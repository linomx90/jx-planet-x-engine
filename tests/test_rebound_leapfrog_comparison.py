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
SCRIPT = ROOT / "benchmarks" / "rebound_leapfrog_comparison.py"
SPEC = importlib.util.spec_from_file_location("rebound_leapfrog_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def real_jx_lanes(profile):
    return benchmark._jx_kdk_lane(profile), benchmark._jx_rkf78_lane(profile)


def collect_keys(value):
    keys = set()
    if isinstance(value, dict):
        keys.update(value)
        for nested in value.values():
            keys.update(collect_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(collect_keys(nested))
    return keys


class AnalyticMetricTests(unittest.TestCase):
    def test_decimal_oracle_cardinal_nontrivial_and_bounds(self):
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
        for epoch in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "finite"):
                benchmark._decimal_sin_cos(epoch)
        for epoch in (1.0e21, -1.0e21, math.nextafter(1.0e20, math.inf)):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(
                ValueError, "must not exceed"
            ):
                benchmark._decimal_sin_cos(epoch)
        broken_decimal = SimpleNamespace(
            from_float=mock.Mock(side_effect=InvalidOperation)
        )
        with mock.patch.object(benchmark, "Decimal", broken_decimal):
            with self.assertRaisesRegex(ValueError, "argument reduction failed"):
                benchmark._decimal_sin_cos(1.0)

    def test_locked_binary_cartesian_and_invariant_formulas(self):
        epochs = np.array((0.0, math.pi / 2.0, 2.0 * math.pi))
        positions, velocities = benchmark.analytic_trajectory(epochs)
        self.assertFalse(np.any(np.signbit(positions[positions == 0.0])))
        self.assertFalse(np.any(np.signbit(velocities[velocities == 0.0])))
        np.testing.assert_array_equal(
            positions[0], ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
        )
        np.testing.assert_array_equal(
            velocities[0], ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))
        )
        metrics = benchmark.invariant_metrics(positions, velocities)
        self.assertLessEqual(metrics["relative_total_energy_max"], 2.0e-15)
        self.assertLessEqual(metrics["relative_angular_momentum_max"], 2.0e-15)
        self.assertEqual(metrics["center_of_mass_position_max"], 0.0)
        self.assertEqual(metrics["total_momentum_max"], 0.0)

        reference = np.zeros((1, 2, 3), dtype=np.float64)
        candidate = reference.copy()
        candidate[0, 0] = (3.0, 4.0, 0.0)
        cartesian = benchmark.cartesian_metrics(candidate, reference)
        self.assertEqual(cartesian["component_max_abs"], 4.0)
        self.assertEqual(cartesian["vector_l2_max"], 5.0)

    def test_phase_error_is_signed_and_unwrapped_for_this_planar_workload(self):
        angles = np.array((3.0, 3.2), dtype=np.float64)
        reference = np.zeros((2, 2, 3), dtype=np.float64)
        reference[:, 0, 0] = -0.5
        reference[:, 1, 0] = 0.5
        candidate = np.zeros_like(reference)
        candidate[:, 0, 0] = -0.5 * np.cos(angles)
        candidate[:, 0, 1] = -0.5 * np.sin(angles)
        candidate[:, 1, 0] = 0.5 * np.cos(angles)
        candidate[:, 1, 1] = 0.5 * np.sin(angles)
        phase = benchmark.phase_error_metrics(candidate, reference)
        self.assertAlmostEqual(phase["final_signed_radians"], 3.2)
        self.assertAlmostEqual(phase["maximum_abs_radians"], 3.2)
        self.assertIn("UNWRAPPED_SIGNED", phase["interpretation"])
        self.assertEqual(phase["candidate_label"], "candidate")
        self.assertEqual(phase["reference_label"], "reference")


class ProfileAndDependencyTests(unittest.TestCase):
    def test_profiles_are_derived_from_the_fixed_integer_lattice(self):
        smoke = benchmark.Profile()
        self.assertEqual(smoke.name, "smoke")
        self.assertEqual(len(smoke.epochs), 17)
        self.assertEqual(smoke.checkpoint_step_indices, tuple(range(0, 257, 16)))
        self.assertEqual(
            smoke.epochs,
            tuple(benchmark.FIXED_STEP * index for index in smoke.checkpoint_step_indices),
        )
        full = benchmark.Profile(100, 4)
        self.assertTrue(full.named_full_profile)
        self.assertEqual(len(full.epochs), 401)
        self.assertEqual(full.checkpoint_step_indices[-1], 25_600)
        with self.assertRaises(ValueError):
            benchmark.Profile(samples_per_period=3)
        with self.assertRaises(ValueError):
            benchmark.Profile(periods=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            full.periods = 1

    def test_help_and_disabled_mode_never_import_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "rebound.py"
            sentinel.write_text(
                "raise RuntimeError('REBOUND IMPORTED DURING HELP')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (directory, str(ROOT / "src"), environment.get("PYTHONPATH", ""))
            )
            completed = subprocess.run(
                (sys.executable, str(SCRIPT), "--help"),
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("optional exact REBOUND 5.1.1", completed.stdout)

        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("simulation.integrate", source)
        self.assertNotIn("exact_finish_time", source)
        self.assertIn("simulation.steps(count)", source)

    def test_exact_rebound_version_and_simulation_api_fail_closed(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "required exactly"):
            benchmark._validate_rebound_module(
                SimpleNamespace(__version__="4.4.11", Simulation=object)
            )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "Simulation"):
            benchmark._validate_rebound_module(SimpleNamespace(__version__="5.1.1"))
        benchmark._validate_rebound_module(
            SimpleNamespace(__version__="5.1.1", Simulation=lambda: None)
        )


class ProvenanceTests(unittest.TestCase):
    def test_engine_source_roster_tree_and_benchmark_hashes_are_exact(self):
        manifest = benchmark._engine_sources_manifest()
        paths = tuple(entry["path"] for entry in manifest["files"])
        self.assertEqual(paths, benchmark.REQUIRED_ENGINE_SOURCE_PATHS)
        self.assertEqual(manifest["file_count"], 18)
        self.assertEqual(
            manifest["selection"], "EXACT_TOP_LEVEL_SOURCE_TREE_PYTHON_ROSTER"
        )
        self.assertFalse(manifest["authority_authorized"])
        for entry in manifest["files"]:
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
        self.assertEqual(
            benchmark._file_identity(SCRIPT)["sha256"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )

    def test_source_manifest_and_jx_runtime_mutations_are_rejected(self):
        manifest = benchmark._engine_sources_manifest()
        mutated = {**manifest, "files": [dict(entry) for entry in manifest["files"]]}
        mutated["files"][0]["size_bytes"] += 1
        mutated["tree_sha256"] = benchmark._engine_source_tree_sha256(
            mutated["files"]
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "source-tree bytes"):
            benchmark._validate_engine_sources_manifest(mutated)
        runtime = benchmark._jx_runtime_provenance()
        changed = {**runtime, "jxplanetx_version": "altered"}
        with self.assertRaisesRegex(benchmark.BenchmarkError, "loaded source bytes"):
            benchmark._validate_jx_runtime_provenance(changed)

    def test_numpy_identity_binds_package_and_both_native_binaries(self):
        identity = benchmark._numpy_runtime_identity()
        self.assertEqual(
            identity["identity_scope"],
            "PACKAGE_MODULE_MULTIARRAY_UMATH_AND_LINALG_NATIVE_BINARIES",
        )
        self.assertEqual(identity["version"], np.__version__)
        for field in (
            "package_module",
            "multiarray_umath_native_binary",
            "linalg_native_binary",
        ):
            record = identity[field]
            self.assertEqual(record, benchmark._file_identity(record["path"]))
        self.assertFalse(identity["authority_authorized"])


class JXLaneTests(unittest.TestCase):
    def test_smoke_kdk_lane_uses_frozen_public_contract_and_accounting(self):
        profile = benchmark.Profile()
        lane = benchmark._jx_kdk_lane(profile)
        self.assertEqual(lane.engine_id, "jx_kdk")
        self.assertTrue(np.array_equal(lane.declared_epochs, lane.observed_epochs))
        self.assertEqual(lane.settings["fixed_step"], benchmark.PERIOD / 256.0)
        self.assertEqual(lane.settings["method_variant"], "JX_KICK_DRIFT_KICK")
        self.assertFalse(lane.settings["equivalent_to_rebound_leapfrog_map_claimed"])
        self.assertEqual(lane.accounting["completed_steps"], 256)
        self.assertEqual(lane.accounting["force_evaluations"], 257)
        self.assertEqual(lane.accounting["checkpoint_count"], 17)
        self.assertRegex(lane.accounting["schedule_content_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(lane.accounting["result_content_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(lane.accounting["registry_authorized"])
        self.assertFalse(lane.accounting["qualification_authorized"])
        benchmark._validate_kdk_lane(lane, profile)
        benchmark._validate_jx_runtime_provenance(lane.runtime)
        oracle = benchmark.analytic_trajectory(lane.declared_epochs)
        report = benchmark._lane_report(lane, *oracle)
        self.assertTrue(report["within_analytic_workload_gate"])
        self.assertEqual(
            report["analytic_workload_gate_id"],
            "FIXED_MAP_EQUAL_BINARY_P_OVER_256_V1",
        )

    def test_smoke_rkf78_reference_lane_uses_conservative_adaptive_controls(self):
        profile = benchmark.Profile()
        lane = benchmark._jx_rkf78_lane(profile)
        self.assertEqual(lane.engine_id, "jx_rkf78_adaptive_reference")
        self.assertEqual(lane.settings["position_rtol"], 1.0e-12)
        self.assertEqual(lane.settings["velocity_rtol"], 1.0e-12)
        self.assertEqual(lane.settings["position_atol"], 1.0e-13)
        self.assertEqual(lane.settings["initial_step"], benchmark.PERIOD / 128.0)
        accounting = lane.accounting
        self.assertEqual(accounting["checkpoint_count"], 17)
        self.assertEqual(
            accounting["attempted_steps"],
            accounting["accepted_steps"] + accounting["rejected_steps"],
        )
        self.assertEqual(
            accounting["force_evaluations"], 13 * accounting["attempted_steps"]
        )
        self.assertEqual(
            accounting["accepted_step_ledger"]["epoch_count"],
            accounting["accepted_steps"],
        )
        benchmark._validate_rkf78_lane(lane, profile)
        oracle = benchmark.analytic_trajectory(lane.declared_epochs)
        report = benchmark._lane_report(lane, *oracle)
        self.assertTrue(report["within_analytic_workload_gate"])
        self.assertEqual(
            report["analytic_workload_gate_id"],
            "RKF78_ADAPTIVE_REFERENCE_EQUAL_BINARY_V1",
        )


class ReportContractTests(unittest.TestCase):
    def test_unavailable_external_lane_schema_controls_and_vocabulary(self):
        profile = benchmark.Profile()
        kdk, rkf = real_jx_lanes(profile)
        report = benchmark.build_report(
            profile,
            kdk,
            rkf,
            None,
            rebound_unavailability_reason="fixture_not_installed",
        )
        self.assertEqual(report["schema"], benchmark.SCHEMA)
        self.assertEqual(set(report["lanes"]), {
            "jx_kdk",
            "jx_rkf78_adaptive_reference",
            "rebound_leapfrog",
        })
        external = report["lanes"]["rebound_leapfrog"]
        self.assertFalse(external["available"])
        self.assertEqual(external["unavailability_reason"], "fixture_not_installed")
        self.assertTrue(report["lanes"]["jx_kdk"]["within_analytic_workload_gate"])
        self.assertTrue(
            report["lanes"]["jx_rkf78_adaptive_reference"][
                "within_analytic_workload_gate"
            ]
        )
        self.assertEqual(report["claim_controls"], benchmark.CLAIM_CONTROLS)
        self.assertTrue(all(value is False for value in report["claim_controls"].values()))
        self.assertFalse(collect_keys(report) & benchmark.FORBIDDEN_RESULT_KEYS)
        self.assertFalse(report["serialization"]["deterministic_report_bytes_claimed"])
        self.assertEqual(
            set(report["analytic_workload_gates"]),
            {"fixed_map_p_over_256", "rkf78_adaptive_reference"},
        )
        checkpoints = report["checkpoints"]
        self.assertEqual(checkpoints["fixed_map_step_indices"][-1], 256)
        self.assertTrue(checkpoints["declared_output_epochs_shared_by_all_executed_lanes"])
        self.assertFalse(checkpoints["adaptive_rkf78_integration_steps_share_fixed_map_indices"])
        interpretation = report["scientific_interpretation"]
        self.assertEqual(
            interpretation["invariant_maxima_sampling_scope"],
            "RETAINED_OUTPUT_CHECKPOINTS_ONLY",
        )
        self.assertFalse(
            interpretation["small_sampled_energy_envelope_implies_small_phase_error"]
        )
        self.assertFalse(interpretation["general_long_term_boundedness_claimed"])
        for lane_name in ("jx_kdk", "jx_rkf78_adaptive_reference"):
            timing = report["lanes"][lane_name]["timing_seconds"]
            self.assertEqual(timing["scope"], "PRIMARY_LANE_EXECUTION_ONLY")
            self.assertEqual(timing["mandatory_validation_replay_execution_count"], 1)
            self.assertFalse(timing["mandatory_validation_replay_timing_included"])
            self.assertFalse(timing["represents_report_generation_wall_time"])
        pair_phase = report["cross_lane_disagreement"][
            "jx_kdk_vs_jx_rkf78_adaptive_reference"
        ]["relative_orbit_phase"]
        self.assertEqual(pair_phase["candidate_label"], "jx_kdk")
        self.assertEqual(
            pair_phase["reference_label"], "jx_rkf78_adaptive_reference"
        )

    def test_jx_lane_settings_accounting_state_and_replay_mutations_are_rejected(self):
        profile = benchmark.Profile()
        kdk, rkf = real_jx_lanes(profile)

        settings = dict(kdk.settings)
        settings["fixed_step"] = 999.0
        with self.assertRaisesRegex(benchmark.BenchmarkError, "KDK settings"):
            benchmark.build_report(
                profile, dataclasses.replace(kdk, settings=settings), rkf, None
            )

        accounting = dict(kdk.accounting)
        accounting.update(
            schedule_content_sha256="0" * 64,
            result_content_sha256="1" * 64,
            minimum_observed_pair_separations=[0.5],
            minimum_observed_swept_pair_separation=0.5,
            maximum_observed_pair_frequency_step=0.01,
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "independent replay"):
            benchmark.build_report(
                profile, dataclasses.replace(kdk, accounting=accounting), rkf, None
            )

        settings = dict(rkf.settings)
        settings["position_rtol"] = 1.0
        with self.assertRaisesRegex(benchmark.BenchmarkError, "RKF78 settings"):
            benchmark.build_report(
                profile, kdk, dataclasses.replace(rkf, settings=settings), None
            )

        accounting = dict(rkf.accounting)
        accounting.update(
            attempted_steps=16,
            accepted_steps=16,
            rejected_steps=0,
            force_evaluations=208,
        )
        ledger = dict(accounting["accepted_step_ledger"])
        ledger.update(
            epoch_count=16,
            magnitude_count=16,
            minimum_magnitude=benchmark.PERIOD / 16.0,
            maximum_magnitude=benchmark.PERIOD / 16.0,
            last_magnitude=benchmark.PERIOD / 16.0,
            magnitude_fsum=benchmark.PERIOD,
            content_integrity_sha256="2" * 64,
        )
        accounting["accepted_step_ledger"] = ledger
        with self.assertRaisesRegex(benchmark.BenchmarkError, "independent replay"):
            benchmark.build_report(
                profile, kdk, dataclasses.replace(rkf, accounting=accounting), None
            )

        positions = kdk.positions.copy()
        positions[0, 0, 0] += 1.0e-12
        with self.assertRaisesRegex(benchmark.BenchmarkError, "initial positions"):
            benchmark.build_report(
                profile, dataclasses.replace(kdk, positions=positions), rkf, None
            )

    def test_lane_report_separates_declared_grid_and_observed_clock_oracles(self):
        profile = benchmark.Profile()
        declared = np.asarray(profile.epochs)
        observed = declared.copy()
        observed[1:] += 1.0e-10
        positions, velocities = benchmark.analytic_trajectory(observed)
        lane = benchmark.Lane(
            "rebound_leapfrog",
            declared,
            observed,
            positions,
            velocities,
            0.0,
            0.0,
            {},
            {},
            {},
        )
        declared_oracle = benchmark.analytic_trajectory(declared)
        report = benchmark._lane_report(lane, *declared_oracle)
        self.assertGreater(
            report["declared_grid_oracle_errors"]["position"]["vector_l2_max"],
            0.0,
        )
        self.assertEqual(
            report["observed_clock_oracle_errors"]["position"]["vector_l2_max"],
            0.0,
        )
        self.assertAlmostEqual(
            report["clock_alignment"]["maximum_abs_observed_minus_declared"],
            1.0e-10,
        )

    def test_run_auto_marks_only_missing_rebound_as_unavailable(self):
        profile = benchmark.Profile()
        with mock.patch.object(
            benchmark,
            "_rebound_leapfrog_lane",
            side_effect=benchmark.ReboundUnavailable("missing fixture"),
        ):
            report = benchmark.run(profile, "auto")
        self.assertFalse(report["lanes"]["rebound_leapfrog"]["available"])
        with mock.patch.object(
            benchmark,
            "_rebound_leapfrog_lane",
            side_effect=benchmark.BenchmarkError("broken fixture"),
        ):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "broken fixture"):
                benchmark.run(profile, "auto")


class OptionalEndToEndTests(unittest.TestCase):
    def test_exact_rebound_claim_boundary_rejects_coherent_mutations(self):
        try:
            installed = importlib.metadata.version("rebound")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("REBOUND is not installed")
        self.assertEqual(installed, benchmark.REQUIRED_REBOUND_VERSION)
        os.environ["OMP_NUM_THREADS"] = "1"
        profile = benchmark.Profile()
        kdk, rkf = real_jx_lanes(profile)
        rebound = benchmark._rebound_leapfrog_lane(profile)
        benchmark.build_report(profile, kdk, rkf, rebound)

        runtime = json.loads(json.dumps(rebound.runtime))
        runtime["rebound_githash"] = "fabricated"
        with self.assertRaisesRegex(benchmark.BenchmarkError, "distribution bytes"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, runtime=runtime)
            )

        runtime = json.loads(json.dumps(rebound.runtime))
        runtime["module_source"] = benchmark._file_identity(SCRIPT)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "distribution bytes"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, runtime=runtime)
            )

        runtime = json.loads(json.dumps(rebound.runtime))
        sources = runtime["python_sources"]
        sources["files"][0]["size_bytes"] += 1
        sources["tree_sha256"] = benchmark._rebound_source_tree_sha256(
            sources["files"]
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "distribution bytes"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, runtime=runtime)
            )

        settings = json.loads(json.dumps(rebound.settings))
        settings["G"] = 2.0
        with self.assertRaisesRegex(benchmark.BenchmarkError, "effective readback"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, settings=settings)
            )

        settings = json.loads(json.dumps(rebound.settings))
        settings["initial_particle_effective_readback"][0]["m"] = 1.0
        with self.assertRaisesRegex(benchmark.BenchmarkError, "effective readback"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, settings=settings)
            )

        accounting = json.loads(json.dumps(rebound.accounting))
        accounting["completed_integer_steps"] = 999
        with self.assertRaisesRegex(benchmark.BenchmarkError, "step accounting"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, accounting=accounting)
            )

        observed = rebound.observed_epochs.copy()
        observed[1] += 1.0e-12
        with self.assertRaisesRegex(benchmark.BenchmarkError, "exact steps replay"):
            benchmark.build_report(
                profile,
                kdk,
                rkf,
                dataclasses.replace(rebound, observed_epochs=observed),
            )

        class Text(str):
            pass

        accounting = dict(rebound.accounting)
        accounting[Text("step_api")] = accounting.pop("step_api")
        with self.assertRaisesRegex(benchmark.BenchmarkError, "step accounting"):
            benchmark.build_report(
                profile, kdk, rkf, dataclasses.replace(rebound, accounting=accounting)
            )

    def test_exact_rebound_5_1_1_full_profile(self):
        try:
            installed = importlib.metadata.version("rebound")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("REBOUND is not installed")
        self.assertEqual(installed, benchmark.REQUIRED_REBOUND_VERSION)
        os.environ["OMP_NUM_THREADS"] = "1"
        report = benchmark.run(benchmark.Profile(100, 4), "required")
        self.assertEqual(report["profile"], {
            "name": "full-100-period",
            "profile_id": "jx.rebound_leapfrog.equal_mass_binary.full_100_period.v1",
            "periods": 100,
            "samples_per_period": 4,
            "named_full_profile": True,
            "fixed_steps_per_period": 256,
        })
        self.assertEqual(report["checkpoints"]["count"], 401)
        self.assertEqual(report["checkpoints"]["fixed_map_step_indices"][-1], 25_600)
        self.assertEqual(set(report["lanes"]), {
            "jx_kdk",
            "jx_rkf78_adaptive_reference",
            "rebound_leapfrog",
        })
        for lane_name in report["lanes"]:
            lane = report["lanes"][lane_name]
            self.assertTrue(lane["within_analytic_workload_gate"], lane_name)
            self.assertFalse(lane["timing_seconds"]["comparability_authorized"])
            self.assertEqual(
                lane["timing_seconds"]["mandatory_validation_replay_execution_count"],
                1,
            )
            self.assertFalse(
                lane["timing_seconds"]["mandatory_validation_replay_timing_included"]
            )

        kdk = report["lanes"]["jx_kdk"]
        self.assertEqual(kdk["accounting"]["completed_steps"], 25_600)
        self.assertEqual(kdk["accounting"]["force_evaluations"], 25_601)
        self.assertEqual(kdk["accounting"]["checkpoint_count"], 401)
        self.assertLessEqual(
            kdk["declared_grid_oracle_errors"]["position"]["vector_l2_max"],
            0.07,
        )
        self.assertGreater(
            kdk["declared_grid_oracle_errors"]["phase"]["maximum_abs_radians"],
            0.1,
        )

        rkf = report["lanes"]["jx_rkf78_adaptive_reference"]
        self.assertEqual(
            rkf["accounting"]["attempted_steps"],
            rkf["accounting"]["accepted_steps"]
            + rkf["accounting"]["rejected_steps"],
        )
        self.assertEqual(
            rkf["accounting"]["force_evaluations"],
            13 * rkf["accounting"]["attempted_steps"],
        )
        self.assertLessEqual(
            rkf["declared_grid_oracle_errors"]["position"]["vector_l2_max"],
            1.0e-7,
        )

        rebound = report["lanes"]["rebound_leapfrog"]
        self.assertTrue(rebound["available"])
        self.assertEqual(rebound["settings"]["integrator"], "leapfrog")
        self.assertEqual(rebound["settings"]["integrator_order"], 2)
        self.assertEqual(rebound["settings"]["method_variant"], "REBOUND_LEAPFROG_DRIFT_KICK_DRIFT")
        self.assertFalse(rebound["settings"]["equivalent_to_jx_kdk_map_claimed"])
        self.assertEqual(
            rebound["settings"]["official_integrator_documentation"],
            "https://rebound.hanno-rein.de/integrators/leapfrog/",
        )
        self.assertEqual(
            rebound["settings"]["initial_particle_effective_readback"],
            [
                {"m": 0.5, "r": 0.0, "x": -0.5, "y": 0.0, "z": 0.0,
                 "vx": 0.0, "vy": -0.5, "vz": 0.0},
                {"m": 0.5, "r": 0.0, "x": 0.5, "y": 0.0, "z": 0.0,
                 "vx": 0.0, "vy": 0.5, "vz": 0.0},
            ],
        )
        evidence = rebound["settings"]["independent_one_step_method_evidence"]
        self.assertTrue(evidence["bitwise_position_equal"])
        self.assertTrue(evidence["bitwise_velocity_equal"])
        self.assertEqual(
            evidence["expected_state_sha256"], evidence["observed_state_sha256"]
        )
        self.assertEqual(rebound["accounting"]["step_api"], "Simulation.steps")
        self.assertEqual(rebound["accounting"]["step_call_count"], 400)
        self.assertEqual(rebound["accounting"]["requested_integer_steps"], 25_600)
        self.assertEqual(rebound["accounting"]["completed_integer_steps"], 25_600)
        self.assertEqual(set(rebound["accounting"]["step_call_arguments"]), {64})
        clock = rebound["clock_alignment"]
        self.assertGreater(clock["maximum_abs_observed_minus_declared"], 0.0)
        self.assertLessEqual(clock["maximum_abs_observed_minus_declared"], 1.0e-9)
        self.assertNotEqual(
            clock["declared_grid_sha256"], clock["observed_simulation_sha256"]
        )
        self.assertGreater(
            rebound["declared_grid_oracle_errors"]["phase"]["maximum_abs_radians"],
            0.1,
        )
        runtime = rebound["runtime"]
        benchmark._validate_rebound_runtime_provenance(runtime)
        self.assertEqual(runtime["distribution_version"], "5.1.1")
        sources = runtime["python_sources"]
        self.assertGreaterEqual(sources["file_count"], 5)
        self.assertEqual(
            sources["tree_sha256"],
            benchmark._rebound_source_tree_sha256(sources["files"]),
        )
        self.assertTrue(
            set(sources["required_execution_wrappers"]).issubset(
                entry["path"] for entry in sources["files"]
            )
        )
        self.assertEqual(runtime["installed_license_expression"], "GPL-3.0-only")
        self.assertTrue(runtime["project_description_mentions_v3_or_later"])
        self.assertTrue(runtime["license_text_inconsistency_observed"])
        self.assertFalse(runtime["legal_or_redistribution_conclusion_authorized"])
        self.assertFalse(runtime["rebound_code_vendored"])
        self.assertFalse(runtime["rebound_code_copied"])
        self.assertFalse(runtime["jx_core_linked_to_rebound"])
        fixed_cross = report["cross_lane_disagreement"][
            "jx_kdk_vs_rebound_leapfrog"
        ]
        self.assertEqual(
            fixed_cross["relative_orbit_phase"]["candidate_label"], "jx_kdk"
        )
        self.assertEqual(
            fixed_cross["relative_orbit_phase"]["reference_label"],
            "rebound_leapfrog",
        )
        self.assertGreater(fixed_cross["position"]["vector_l2_max"], 1.0e-6)
        self.assertLessEqual(fixed_cross["position"]["vector_l2_max"], 1.2e-4)
        self.assertFalse(collect_keys(report) & benchmark.FORBIDDEN_RESULT_KEYS)
        self.assertEqual(report["claim_controls"], benchmark.CLAIM_CONTROLS)
        self.assertTrue(all(value is False for value in report["claim_controls"].values()))


if __name__ == "__main__":
    unittest.main()
