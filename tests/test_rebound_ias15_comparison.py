import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "rebound_ias15_comparison.py"
SPEC = importlib.util.spec_from_file_location("rebound_ias15_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def fake_lane(profile, engine_id="fake"):
    epochs = np.asarray(profile.epochs, dtype=np.float64)
    positions, velocities = benchmark.analytic_trajectory(epochs)
    runtime = (
        benchmark._jx_runtime_provenance()
        if engine_id == "jx_rkf78"
        else {"fixture": True}
    )
    return benchmark.Lane(
        engine_id,
        epochs,
        positions,
        velocities,
        0.0,
        0.0,
        {
            "fixture": True,
            "configuration_mode": benchmark.DEFAULT_JX_CONFIGURATION_MODE,
        },
        runtime,
        {"fixture": True},
    )


class AnalyticAndMetricTests(unittest.TestCase):
    def test_decimal_oracle_at_cardinal_and_nontrivial_epochs(self):
        cases = (
            (0.0, (0.0, 1.0)),
            (math.pi / 2.0, (1.0, 6.123233995736766e-17)),
            (2.0 * math.pi, (-2.4492935982947064e-16, 1.0)),
            (0.7, (0.644217687237691, 0.7648421872844885)),
        )
        self.assertGreaterEqual(benchmark.DECIMAL_PRECISION, 80)
        for epoch, expected in cases:
            with self.subTest(epoch=epoch):
                self.assertEqual(benchmark._decimal_sin_cos(epoch), expected)

    def test_decimal_oracle_epoch_bounds_and_fail_closed_conversion(self):
        sine, cosine = benchmark._decimal_sin_cos(benchmark.DECIMAL_EPOCH_ABS_MAX)
        self.assertTrue(math.isfinite(sine) and math.isfinite(cosine))
        for epoch in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "finite"):
                benchmark._decimal_sin_cos(epoch)
        for epoch in (1.0e21, -1.0e21, math.nextafter(1.0e20, math.inf)):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "must not exceed"):
                benchmark._decimal_sin_cos(epoch)
        broken_decimal = SimpleNamespace(
            from_float=mock.Mock(side_effect=InvalidOperation)
        )
        with mock.patch.object(benchmark, "Decimal", broken_decimal):
            with self.assertRaisesRegex(ValueError, "argument reduction failed"):
                benchmark._decimal_sin_cos(1.0)

    def test_locked_equal_mass_orbit_and_invariants(self):
        epochs = np.array((0.0, math.pi / 2.0, 2.0 * math.pi))
        positions, velocities = benchmark.analytic_trajectory(epochs)
        np.testing.assert_allclose(
            positions[0], ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), atol=0.0
        )
        np.testing.assert_allclose(
            velocities[0], ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), atol=0.0
        )
        np.testing.assert_allclose(
            positions[1], ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), atol=1e-15
        )
        np.testing.assert_allclose(positions[2], positions[0], atol=1e-15)
        metrics = benchmark.invariant_metrics(positions, velocities)
        self.assertLess(metrics["relative_total_energy_max"], 2e-15)
        self.assertLess(metrics["relative_angular_momentum_max"], 2e-15)
        self.assertEqual(metrics["center_of_mass_position_max"], 0.0)
        self.assertEqual(metrics["total_momentum_max"], 0.0)

    def test_cartesian_metric_definitions_are_pure(self):
        reference = np.zeros((1, 2, 3), dtype=np.float64)
        candidate = reference.copy()
        candidate[0, 0] = (3.0, 4.0, 0.0)
        metrics = benchmark.cartesian_metrics(candidate, reference)
        self.assertEqual(metrics["component_max_abs"], 4.0)
        self.assertAlmostEqual(metrics["component_rms"], math.sqrt(25.0 / 6.0))
        self.assertEqual(metrics["vector_l2_max"], 5.0)
        self.assertAlmostEqual(metrics["vector_l2_rms"], math.sqrt(25.0 / 2.0))
        with self.assertRaises(ValueError):
            benchmark.cartesian_metrics(np.zeros((2, 3)), np.zeros((2, 3)))


class DependencyBoundaryTests(unittest.TestCase):
    def test_import_and_help_do_not_import_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "rebound.py"
            sentinel.write_text("raise RuntimeError('REBOUND IMPORTED DURING HELP')\n")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (directory, str(ROOT / "src"), environment.get("PYTHONPATH", ""))
            )
            result = subprocess.run(
                (sys.executable, str(SCRIPT), "--help"),
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("REBOUND 5.1.1", result.stdout)

    def test_wrong_or_incomplete_rebound_fails_closed(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "required exactly"):
            benchmark._validate_rebound_module(
                SimpleNamespace(__version__="4.4.11", Simulation=object)
            )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "Simulation API"):
            benchmark._validate_rebound_module(SimpleNamespace(__version__="5.1.1"))


class RuntimeProvenanceTests(unittest.TestCase):
    def test_engine_source_roster_and_tree_hash_are_exact(self):
        manifest = benchmark._engine_sources_manifest()
        self.assertEqual(manifest["schema"], benchmark.ENGINE_SOURCE_MANIFEST_SCHEMA)
        self.assertEqual(manifest["package"], "jxplanetx.engine")
        self.assertEqual(
            manifest["selection"],
            "ALL_TOP_LEVEL_REGULAR_NONSYMLINK_PYTHON_SOURCES",
        )
        self.assertEqual(manifest["classification"], "PROVENANCE_DIAGNOSTIC")
        self.assertFalse(manifest["authority_authorized"])
        paths = tuple(entry["path"] for entry in manifest["files"])
        self.assertEqual(paths, benchmark.REQUIRED_ENGINE_SOURCE_PATHS)
        self.assertEqual(manifest["file_count"], len(paths))
        self.assertEqual(manifest["loaded_source_paths"], sorted(paths))
        for entry in manifest["files"]:
            self.assertEqual(set(entry), {"path", "size_bytes", "sha256"})
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size_bytes"], 0)

        payload = json.dumps(
            manifest["files"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        expected = hashlib.sha256(
            benchmark.ENGINE_SOURCE_TREE_HASH_DOMAIN.encode("ascii")
            + b"\0"
            + payload
        ).hexdigest()
        self.assertEqual(manifest["tree_sha256"], expected)
        benchmark._validate_engine_sources_manifest(manifest)

    def test_engine_source_manifest_mutations_are_detected(self):
        manifest = benchmark._engine_sources_manifest()
        mutated_content = {
            **manifest,
            "files": [dict(entry) for entry in manifest["files"]],
        }
        mutated_content["files"][3]["size_bytes"] += 1
        mutated_digest = benchmark._engine_source_tree_sha256(
            mutated_content["files"]
        )
        self.assertNotEqual(mutated_digest, manifest["tree_sha256"])
        mutated_content["tree_sha256"] = mutated_digest
        with self.assertRaisesRegex(benchmark.BenchmarkError, "content identity"):
            benchmark._validate_engine_sources_manifest(mutated_content)

        mutated_tree = dict(manifest)
        mutated_tree["tree_sha256"] = "0" * 64
        with self.assertRaisesRegex(benchmark.BenchmarkError, "tree content hash"):
            benchmark._validate_engine_sources_manifest(mutated_tree)

    def test_numpy_python_and_native_binary_identities_bind_bytes(self):
        identity = benchmark._numpy_runtime_identity()
        self.assertEqual(identity["classification"], "PROVENANCE_DIAGNOSTIC")
        self.assertFalse(identity["authority_authorized"])
        self.assertEqual(
            identity["identity_scope"],
            "PACKAGE_MODULE_AND_MULTIARRAY_UMATH_NATIVE_BINARY",
        )
        self.assertEqual(identity["version"], np.__version__)
        for label in ("package_module", "multiarray_umath_native_binary"):
            record = identity[label]
            path = Path(record["path"])
            data = path.read_bytes()
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(record["size_bytes"], len(data))
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
        self.assertNotEqual(
            Path(identity["multiarray_umath_native_binary"]["path"]).suffix,
            ".py",
        )

    def test_report_builder_rejects_incomplete_or_mutated_jx_runtime(self):
        profile = benchmark.Profile()
        rebound_lane = fake_lane(profile, "rebound_ias15")
        incomplete = dataclasses.replace(
            fake_lane(profile, "jx_rkf78"), runtime={"fixture": True}
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "schema"):
            benchmark.build_report(
                profile, incomplete, rebound_lane, {"fixture": True}
            )

        runtime = benchmark._jx_runtime_provenance()
        mutated_numpy = {
            **runtime["numpy"],
            "version": "0.0.0",
        }
        mutated = dataclasses.replace(
            fake_lane(profile, "jx_rkf78"),
            runtime={**runtime, "numpy": mutated_numpy},
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "loaded bytes"):
            benchmark.build_report(
                profile, mutated, rebound_lane, {"fixture": True}
            )

    def test_file_identity_rejects_missing_nonregular_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("x = 1\n", encoding="utf-8")
            link = root / "link.py"
            link.symlink_to(target)
            with self.assertRaisesRegex(benchmark.BenchmarkError, "symlink"):
                benchmark._file_identity(link)
            with self.assertRaisesRegex(benchmark.BenchmarkError, "regular file"):
                benchmark._file_identity(root)
            with self.assertRaisesRegex(benchmark.BenchmarkError, "unavailable"):
                benchmark._file_identity(root / "missing.py")


class JXConfigurationTests(unittest.TestCase):
    def test_cli_and_run_default_to_matched_accuracy_mode(self):
        parser = benchmark._parser()
        self.assertEqual(
            parser.parse_args([]).jx_mode,
            benchmark.JX_MATCHED_ACCURACY_MODE,
        )
        self.assertEqual(
            set(benchmark.JX_CONFIGURATION_MODES),
            {
                benchmark.JX_ADAPTIVE_BASELINE_MODE,
                benchmark.JX_MATCHED_ACCURACY_MODE,
            },
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(("--jx-mode", "unknown"))

    def test_matched_and_baseline_specs_are_closed_and_exact(self):
        profile = benchmark.Profile(100, 4)
        matched, matched_settings = benchmark._jx_spec(
            profile, benchmark.JX_MATCHED_ACCURACY_MODE
        )
        self.assertEqual(matched.initial_step, benchmark.PERIOD / 256.0)
        self.assertEqual(matched.maximum_step, benchmark.PERIOD / 256.0)
        self.assertEqual(matched.minimum_step, 1.0e-15)
        self.assertTrue(np.all(matched.position_atol == 1.0e6))
        self.assertTrue(np.all(matched.velocity_atol == 1.0e6))
        self.assertEqual(matched.position_rtol, 1.0e-6)
        self.assertEqual(matched.velocity_rtol, 1.0e-6)
        self.assertEqual(matched.maximum_steps, 30_000)
        self.assertEqual(matched.maximum_rejections, 0)
        self.assertEqual(
            matched_settings["method_classification"], "capped_adaptive_rkf78"
        )

        baseline, baseline_settings = benchmark._jx_spec(
            profile, benchmark.JX_ADAPTIVE_BASELINE_MODE
        )
        self.assertEqual(baseline.initial_step, benchmark.PERIOD / 128.0)
        self.assertEqual(baseline.maximum_step, benchmark.PERIOD / 16.0)
        self.assertEqual(baseline.position_rtol, 1.0e-12)
        self.assertTrue(np.all(baseline.position_atol == 1.0e-13))
        self.assertEqual(baseline.maximum_steps, 2_000_000)
        self.assertEqual(baseline.maximum_rejections, 200_000)
        self.assertEqual(
            baseline_settings["configuration_mode"],
            benchmark.JX_ADAPTIVE_BASELINE_MODE,
        )
        with self.assertRaisesRegex(ValueError, "configuration mode"):
            benchmark._jx_spec(profile, "unknown")

    def test_matched_lane_reports_result_custody_and_provenance(self):
        lane = benchmark._jx_lane(benchmark.Profile())
        self.assertEqual(
            lane.settings["configuration_mode"],
            benchmark.JX_MATCHED_ACCURACY_MODE,
        )
        accounting = lane.accounting
        self.assertEqual(accounting["custody_source"], "TrajectoryResult")
        self.assertEqual(
            accounting["accepted_state_accumulation"],
            "KAHAN_BACKEND_NATIVE_COMPONENTWISE",
        )
        self.assertEqual(
            accounting["time_step_representation"],
            "REPRESENTABLE_ENDPOINT_DELTA",
        )
        self.assertEqual(
            accounting["accepted_step_magnitude_source"],
            "ABS_SIGNED_RKF78_STEP_ARGUMENT",
        )
        self.assertEqual(
            accounting["checkpoint_proposal_policy"],
            "PRESERVE_PRECLIP_PROPOSAL_AFTER_ACCEPTED_CLIP",
        )
        ledger = accounting["accepted_step_ledger"]
        self.assertEqual(ledger["epoch_count"], accounting["accepted_steps"])
        self.assertEqual(ledger["magnitude_count"], accounting["accepted_steps"])
        self.assertRegex(ledger["content_integrity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            ledger["checksum_algorithm"],
            "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1",
        )
        self.assertEqual(
            ledger["checksum_domain"],
            "jxplanetx.accepted-step-ledger.content-integrity.v1",
        )
        self.assertFalse(accounting["registry_authorized"])
        self.assertFalse(accounting["qualification_authorized"])
        runtime = lane.runtime
        benchmark._validate_engine_sources_manifest(runtime["engine_sources"])
        self.assertEqual(
            runtime["benchmark_script"], benchmark._file_identity(benchmark.__file__)
        )
        self.assertEqual(runtime["numpy"], benchmark._numpy_runtime_identity())
        self.assertEqual(runtime["python_executable"], sys.executable)

    def test_common_accuracy_gate_is_exact_and_workload_specific(self):
        position = {"vector_l2_max": 2.0e-12}
        velocity = {"vector_l2_max": 2.0e-12}
        drifts = {
            "relative_total_energy_max": 1.0e-14,
            "relative_angular_momentum_max": 5.0e-15,
            "center_of_mass_position_max": 1.0e-12,
            "total_momentum_max": 1.0e-12,
        }
        self.assertTrue(
            benchmark._within_common_accuracy_gate(position, velocity, drifts)
        )
        position["vector_l2_max"] = math.nextafter(2.0e-12, math.inf)
        self.assertFalse(
            benchmark._within_common_accuracy_gate(position, velocity, drifts)
        )


class ReportContractTests(unittest.TestCase):
    def test_schema_controls_and_vocabulary(self):
        profile = benchmark.Profile()
        report = benchmark.build_report(
            profile,
            fake_lane(profile, "jx_rkf78"),
            fake_lane(profile, "rebound_ias15"),
            {"fixture": True},
        )
        self.assertEqual(report["schema"], benchmark.SCHEMA)
        self.assertTrue(report["checkpoints"]["identical_for_both_lanes"])
        self.assertFalse(report["profile"]["named_full_profile"])
        self.assertEqual(
            report["profile"]["jx_configuration_mode"],
            benchmark.JX_MATCHED_ACCURACY_MODE,
        )
        self.assertEqual(
            report["common_accuracy_gate"],
            {
                "scope": "THIS_ANALYTIC_EQUAL_MASS_BINARY_WORKLOAD_ONLY",
                "thresholds": benchmark.COMMON_ACCURACY_GATE,
                "claim_authorized": False,
            },
        )
        self.assertTrue(report["lanes"]["jx_rkf78"]["within_common_accuracy_gate"])
        self.assertTrue(
            report["lanes"]["rebound_ias15"]["within_common_accuracy_gate"]
        )
        self.assertTrue(report["claim_controls"])
        self.assertTrue(all(value is False for value in report["claim_controls"].values()))
        self.assertEqual(
            report["serialization"],
            {
                "format": "JSON",
                "object_keys_sorted": True,
                "nonfinite_numbers_allowed": False,
                "includes_timing_and_path_diagnostics": True,
            },
        )
        self.assertEqual(report["problem"]["analytic_total_energy"], -0.125)
        self.assertEqual(report["problem"]["analytic_total_angular_momentum"], [0.0, 0.0, 0.25])

        keys = set()
        def collect(value):
            if isinstance(value, dict):
                keys.update(value)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)
        collect(report)
        self.assertFalse(keys & benchmark.FORBIDDEN_RESULT_KEYS)

    def test_named_full_profile_is_exactly_one_hundred_periods(self):
        profile = benchmark.Profile(100, 4)
        self.assertEqual(profile.profile_id, "jx.rebound_ias15.equal_mass_binary.full_100_period.v1")
        self.assertEqual(len(profile.epochs), 401)
        self.assertEqual(profile.epochs[-1], 200.0 * math.pi)
        report = benchmark.build_report(
            profile,
            fake_lane(profile, "jx_rkf78"),
            fake_lane(profile, "rebound_ias15"),
            {"fixture": True},
        )
        self.assertTrue(report["profile"]["named_full_profile"])

    def test_profile_name_and_identity_cannot_diverge(self):
        full = benchmark.Profile(100, 4)
        self.assertEqual(full.name, "full-100-period")
        self.assertTrue(full.named_full_profile)
        with self.assertRaises(TypeError):
            benchmark.Profile(100, 4, "smoke")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            full.periods = 1
        with self.assertRaises(dataclasses.FrozenInstanceError):
            full.name = "smoke"
        smoke = dataclasses.replace(full, periods=1, samples_per_period=16)
        self.assertEqual(smoke.name, "smoke")
        self.assertEqual(smoke.profile_id, "jx.rebound_ias15.equal_mass_binary.smoke.v1")
        self.assertFalse(smoke.named_full_profile)
        custom = dataclasses.replace(full, periods=2, samples_per_period=4)
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.profile_id, "jx.rebound_ias15.equal_mass_binary.custom.v1")
        self.assertFalse(custom.named_full_profile)


class OptionalEndToEndTests(unittest.TestCase):
    def test_exact_rebound_5_1_1_full_one_hundred_period_profile(self):
        try:
            installed = importlib.metadata.version("rebound")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("REBOUND is not installed")
        self.assertEqual(installed, benchmark.REQUIRED_REBOUND_VERSION)
        os.environ["OMP_NUM_THREADS"] = "1"
        report = benchmark.run(benchmark.Profile(100, 4))
        self.assertEqual(
            report["profile"],
            {
                "name": "full-100-period",
                "profile_id": "jx.rebound_ias15.equal_mass_binary.full_100_period.v1",
                "periods": 100,
                "samples_per_period": 4,
                "named_full_profile": True,
                "jx_configuration_mode": benchmark.JX_MATCHED_ACCURACY_MODE,
            },
        )
        self.assertEqual(report["checkpoints"]["count"], 401)
        self.assertEqual(len(report["checkpoints"]["epochs"]), 401)
        self.assertTrue(report["checkpoints"]["identical_for_both_lanes"])
        self.assertEqual(report["claim_controls"], benchmark.CLAIM_CONTROLS)
        self.assertTrue(all(value is False for value in report["claim_controls"].values()))
        self.assertEqual(set(report["lanes"]), {"jx_rkf78", "rebound_ias15"})
        jx_lane = report["lanes"]["jx_rkf78"]
        jx_settings = jx_lane["settings"]
        self.assertEqual(
            jx_settings["configuration_mode"],
            benchmark.JX_MATCHED_ACCURACY_MODE,
        )
        self.assertEqual(jx_settings["initial_step"], benchmark.PERIOD / 256.0)
        self.assertEqual(jx_settings["maximum_step"], benchmark.PERIOD / 256.0)
        self.assertEqual(jx_settings["position_atol"], 1.0e6)
        self.assertEqual(jx_settings["velocity_atol"], 1.0e6)
        self.assertEqual(jx_settings["position_rtol"], 1.0e-6)
        self.assertEqual(jx_settings["velocity_rtol"], 1.0e-6)
        self.assertEqual(jx_settings["maximum_steps"], 30_000)
        self.assertEqual(jx_settings["maximum_rejections"], 0)
        accounting = jx_lane["accounting"]
        self.assertEqual(accounting["attempted_steps"], 26_000)
        self.assertEqual(accounting["accepted_steps"], 26_000)
        self.assertEqual(accounting["rejected_steps"], 0)
        self.assertEqual(accounting["force_evaluations"], 338_000)
        self.assertEqual(
            accounting["accepted_state_accumulation"],
            "KAHAN_BACKEND_NATIVE_COMPONENTWISE",
        )
        self.assertEqual(
            accounting["time_step_representation"],
            "REPRESENTABLE_ENDPOINT_DELTA",
        )
        self.assertEqual(
            accounting["accepted_step_magnitude_source"],
            "ABS_SIGNED_RKF78_STEP_ARGUMENT",
        )
        self.assertEqual(
            accounting["checkpoint_proposal_policy"],
            "PRESERVE_PRECLIP_PROPOSAL_AFTER_ACCEPTED_CLIP",
        )
        ledger = accounting["accepted_step_ledger"]
        self.assertEqual(ledger["epoch_count"], 26_000)
        self.assertEqual(ledger["magnitude_count"], 26_000)
        self.assertLessEqual(ledger["maximum_magnitude"], benchmark.PERIOD / 256.0)
        self.assertLessEqual(
            abs(ledger["magnitude_fsum"] - ledger["epoch_span"]),
            math.ulp(ledger["epoch_span"]),
        )
        self.assertRegex(ledger["content_integrity_sha256"], r"^[0-9a-f]{64}$")
        rebound_settings = report["lanes"]["rebound_ias15"]["settings"]
        self.assertEqual(
            {key: rebound_settings[key] for key in ("G", "gravity", "collision", "softening", "integrator", "epsilon", "min_dt", "adaptive_mode", "N_active", "testparticle_type", "thread_count")},
            {"G": 1.0, "gravity": "basic", "collision": "none", "softening": 0.0, "integrator": "ias15", "epsilon": 1.0e-12, "min_dt": 0.0, "adaptive_mode": "PRS23", "N_active": 2, "testparticle_type": 0, "thread_count": 1},
        )
        persistent = {"G", "gravity", "collision", "softening", "integrator", "epsilon", "min_dt", "adaptive_mode", "initial_dt", "N_active", "testparticle_type"}
        provenance = rebound_settings["provenance"]
        self.assertTrue(all(provenance[key] == "effective_readback" for key in persistent))
        self.assertEqual(provenance["exact_finish_time"], "call_argument")
        self.assertEqual(provenance["thread_count"], "environment_observation")
        self.assertEqual(provenance["OMP_NUM_THREADS"], "environment_observation")
        self.assertEqual(rebound_settings["OMP_NUM_THREADS"], "1")
        for lane in report["lanes"].values():
            self.assertTrue(lane["within_common_accuracy_gate"])
            self.assertLessEqual(
                lane["analytic_oracle_errors"]["position"]["vector_l2_max"],
                2e-12,
            )
            self.assertLessEqual(
                lane["analytic_oracle_errors"]["velocity"]["vector_l2_max"],
                2e-12,
            )
            drifts = lane["invariant_drifts"]
            self.assertLessEqual(drifts["relative_total_energy_max"], 1e-14)
            self.assertLessEqual(drifts["relative_angular_momentum_max"], 5e-15)
            self.assertLessEqual(drifts["center_of_mass_position_max"], 1e-12)
            self.assertLessEqual(drifts["total_momentum_max"], 1e-12)

        keys = set()
        def collect(value):
            if isinstance(value, dict):
                keys.update(value)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)
        collect(report)
        self.assertFalse(keys & benchmark.FORBIDDEN_RESULT_KEYS)


if __name__ == "__main__":
    unittest.main()
