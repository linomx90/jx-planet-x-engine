import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "rebound_whfast_comparison.py"
SPEC = importlib.util.spec_from_file_location("rebound_whfast_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class ContractAndProvenanceTests(unittest.TestCase):
    def test_help_and_import_do_not_require_rebound(self):
        source = SCRIPT.read_text(encoding="utf-8")
        prefix = source.split("def _load_rebound()", 1)[0]
        self.assertNotIn("import rebound", prefix)
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WHFast", result.stdout)
        self.assertIn("--rebound-mode", result.stdout)

    def test_locked_initial_state_float_hex_and_digest(self):
        self.assertEqual(
            benchmark._initial_state_sha256(), benchmark.INITIAL_STATE_SHA256
        )
        positions, velocities = benchmark._initial_arrays()
        rows = np.concatenate((positions, velocities), axis=1)
        self.assertEqual(
            tuple(tuple(float(value).hex() for value in row) for row in rows),
            tuple(
                tuple(float.fromhex(value).hex() for value in row)
                for row in benchmark.INITIAL_CARTESIAN_HEX
            ),
        )
        self.assertEqual(
            tuple(float(value).hex() for value in benchmark.GRAVITATIONAL_PARAMETERS),
            (
                "0x1.0000000000000p+0",
                "0x1.0624dd2f1a9fcp-10",
                "0x1.0624dd2f1a9fcp-9",
            ),
        )

    def test_named_full_profile_has_exact_integer_lattices(self):
        profile = benchmark.Profile.full()
        self.assertTrue(profile.named_full_profile)
        self.assertEqual(profile.name, "full-10-and-100-period")
        self.assertEqual(len(profile.checkpoint_step_indices("short", 64)), 41)
        self.assertEqual(len(profile.checkpoint_step_indices("short", 128)), 41)
        self.assertEqual(profile.checkpoint_step_indices("short", 64)[-1], 640)
        self.assertEqual(profile.checkpoint_step_indices("short", 128)[-1], 1280)
        self.assertEqual(len(profile.checkpoint_step_indices("long", 64)), 401)
        self.assertEqual(profile.checkpoint_step_indices("long", 64)[-1], 6400)
        with self.assertRaises(ValueError):
            benchmark.Profile(short_periods=True)
        with self.assertRaises(ValueError):
            benchmark.Profile(short_periods=2, long_periods=1)

    def test_engine_tree_and_support_source_are_exactly_bound(self):
        runtime = benchmark._jx_runtime_provenance()
        manifest = runtime["engine_sources"]
        self.assertEqual(manifest["file_count"], 18)
        paths = tuple(entry["path"] for entry in manifest["files"])
        self.assertEqual(paths, benchmark._PROVENANCE_SUPPORT.REQUIRED_ENGINE_SOURCE_PATHS)
        self.assertIn("jxplanetx/engine/wisdom_holman.py", paths)
        self.assertIn("jxplanetx/engine/wisdom_holman_contracts.py", paths)
        for field in ("benchmark_script", "provenance_support_script"):
            identity = runtime[field]
            data = Path(identity["path"]).read_bytes()
            self.assertEqual(identity["size_bytes"], len(data))
            self.assertEqual(identity["sha256"], hashlib.sha256(data).hexdigest())
        benchmark._validate_jx_runtime_provenance(runtime)
        changed = {**runtime, "jxplanetx_version": "forged"}
        with self.assertRaisesRegex(benchmark.BenchmarkError, "source bytes"):
            benchmark._validate_jx_runtime_provenance(changed)

    def test_poisoned_support_module_is_rejected_before_execution(self):
        support = ROOT / "benchmarks" / "rebound_leapfrog_comparison.py"
        digest = hashlib.sha256(support.read_bytes()).hexdigest()
        name = f"jx_rebound_leapfrog_provenance_support_{digest}"
        code = f"""
import importlib.util, pathlib, sys, types
support = pathlib.Path({str(support)!r})
poison = types.ModuleType({name!r})
poison.__file__ = str(support)
poison.REQUIRED_REBOUND_VERSION = '5.1.1'
sys.modules[{name!r}] = poison
path = pathlib.Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('poison_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    assert 'preloaded provenance-support substitution' in str(exc), str(exc)
else:
    raise AssertionError('poisoned support module was accepted')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_whfast_uses_integer_steps_and_never_integrates_to_outputs(self):
        source = inspect.getsource(benchmark._whfast_lane)
        self.assertIn("simulation.steps(delta)", source)
        self.assertNotIn("simulation.integrate", source)
        self.assertNotIn("simulation.integrate(", source)


class MetricTests(unittest.TestCase):
    def test_lane_content_digest_literal_known_answer_and_mutations(self):
        lane = benchmark.Lane(
            "fixture",
            "short",
            64,
            benchmark._owned_readonly_float64((0.0,)),
            benchmark._owned_readonly_float64((0.5,)),
            benchmark._owned_readonly_float64(
                (((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)),)
            ),
            benchmark._owned_readonly_float64(
                (((-0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),)
            ),
            0.0,
            0.0,
            {"alpha": 1.5},
            {"path": "x"},
            {"count": 1},
        )
        literal_payload = (
            b'{"accounting":{"count":1},"declared_epochs":[{"binary64_hex":'
            b'"0x0.0p+0"}],"engine_id":"fixture","horizon":"short",'
            b'"observed_epochs":[{"binary64_hex":"0x1.0000000000000p-1"}],'
            b'"positions":[[[{"binary64_hex":"0x1.0000000000000p+0"},'
            b'{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x0.0p+0"}],'
            b'[{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x1.0000000000000p+1"},'
            b'{"binary64_hex":"0x0.0p+0"}],[{"binary64_hex":"0x0.0p+0"},'
            b'{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x1.8000000000000p+1"}]]],'
            b'"runtime":{"path":"x"},'
            b'"schema":"jx.rebound_whfast.lane_content.v1","settings":{"alpha":{'
            b'"binary64_hex":"0x1.8000000000000p+0"}},"step_divisor":64,'
            b'"timing_fields_excluded":["setup_seconds","integration_seconds"],'
            b'"velocities":[[[{"binary64_hex":"-0x0.0p+0"},'
            b'{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x0.0p+0"}],'
            b'[{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x0.0p+0"},'
            b'{"binary64_hex":"0x0.0p+0"}],[{"binary64_hex":"0x0.0p+0"},'
            b'{"binary64_hex":"0x0.0p+0"},{"binary64_hex":"0x0.0p+0"}]]]}'
        )
        expected = hashlib.sha256(
            benchmark.LANE_CONTENT_CHECKSUM_DOMAIN.encode("ascii")
            + b"\0"
            + literal_payload
        ).hexdigest()
        self.assertEqual(
            expected,
            "d6779c0ccdfbb074a96eb0b95407b01ffdb4558887a5a891c95382ae995cdf84",
        )
        self.assertEqual(benchmark._lane_content_sha256(lane), expected)
        changed_state = dataclasses.replace(
            lane,
            positions=benchmark._owned_readonly_float64(
                (
                    (
                        (1.0000000000000002, 0.0, 0.0),
                        (0.0, 2.0, 0.0),
                        (0.0, 0.0, 3.0),
                    ),
                )
            ),
        )
        changed_epoch = dataclasses.replace(
            lane,
            observed_epochs=benchmark._owned_readonly_float64(
                (math.nextafter(0.5, 1.0),)
            ),
        )
        changed_setting = dataclasses.replace(lane, settings={"alpha": 1.25})
        for changed in (changed_state, changed_epoch, changed_setting):
            self.assertNotEqual(benchmark._lane_content_sha256(changed), expected)

    def test_lane_arrays_reject_views_subclasses_and_overlap(self):
        profile = benchmark.Profile()
        lane = benchmark._jx_lane(profile, "short", 64)
        for array in (
            lane.declared_epochs,
            lane.observed_epochs,
            lane.positions,
            lane.velocities,
        ):
            self.assertIs(type(array), np.ndarray)
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
            self.assertTrue(array.flags.c_contiguous)

        external = np.array(lane.positions, copy=True)
        view = external.view()
        view.setflags(write=False)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "owned"):
            dataclasses.replace(lane, positions=view)

        class ArraySubclass(np.ndarray):
            pass

        subclass = np.array(lane.positions, copy=True).view(ArraySubclass)
        subclass.setflags(write=False)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "exact base ndarray"):
            dataclasses.replace(lane, positions=subclass)

        with self.assertRaisesRegex(benchmark.BenchmarkError, "share memory"):
            dataclasses.replace(lane, observed_epochs=lane.declared_epochs)

    def test_pure_cartesian_phase_and_invariant_metrics(self):
        positions, velocities = benchmark._initial_arrays()
        trajectory_positions = np.stack((positions, positions))
        trajectory_velocities = np.stack((velocities, velocities))
        same = benchmark._cartesian_metrics(
            trajectory_positions, trajectory_positions.copy()
        )
        self.assertEqual(same["vector_l2_max"], 0.0)
        changed = trajectory_positions.copy()
        changed[1, 2, 0] += 3.0
        metrics = benchmark._cartesian_metrics(changed, trajectory_positions)
        self.assertEqual(metrics["vector_l2_max"], 3.0)
        phase = benchmark._phase_proxy_metrics(
            trajectory_positions, trajectory_positions.copy()
        )
        self.assertEqual(phase["max_abs_radians"], 0.0)
        invariants = benchmark._invariant_metrics(
            trajectory_positions, trajectory_velocities
        )
        self.assertEqual(invariants["relative_total_energy_max"], 0.0)
        self.assertEqual(invariants["relative_angular_momentum_max"], 0.0)

    def test_two_half_clock_replay_is_exact_and_distinct_from_declared_grid(self):
        profile = benchmark.Profile.full()
        indices = profile.checkpoint_step_indices("long", 64)
        step = benchmark.PERIOD / 64.0
        replay = benchmark._clock_replay(indices, step)
        declared = np.asarray(profile.declared_epochs("long", 64))
        self.assertEqual(replay[0], 0.0)
        self.assertNotEqual(replay[-1].hex(), declared[-1].hex())
        self.assertGreater(abs(float(replay[-1] - declared[-1])), 0.0)


class DisabledExternalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = benchmark.Profile()
        cls.study = benchmark._execute_study(cls.profile, "disabled")
        cls.report = benchmark.build_report(cls.study)

    def test_real_jx_smoke_and_negative_domain_control(self):
        self.assertEqual(len(self.study.jx_lanes), 3)
        self.assertEqual(self.study.whfast_lanes, ())
        self.assertEqual(self.study.ias15_lanes, ())
        self.assertTrue(self.study.negative_control["domain_rejection_observed"])
        self.assertEqual(
            self.study.negative_control["error_type"], "TrajectoryDomainError"
        )
        for lane in self.study.jx_lanes:
            completed = lane.accounting["completed_steps"]
            self.assertEqual(
                lane.accounting["primary_map_force_evaluations"], completed + 1
            )
            self.assertEqual(
                lane.accounting["validation_replay_force_evaluations"],
                completed + 1,
            )
            self.assertEqual(
                lane.accounting["total_public_call_force_evaluations"],
                2 * (completed + 1),
            )
            self.assertFalse(lane.accounting["floating_point_symplectic"])

    def test_report_schema_claim_vocabulary_and_timing_scopes(self):
        report = self.report
        self.assertEqual(report["schema"], benchmark.SCHEMA)
        self.assertFalse(
            report["runtime_provenance"]["external_status"][
                "external_lanes_executed"
            ]
        )
        self.assertIsNone(report["short_convergence"])
        self.assertEqual(report["claim_controls"], benchmark.CLAIM_CONTROLS)
        text = json.dumps(report, sort_keys=True, allow_nan=False)
        for forbidden in benchmark.FORBIDDEN_RESULT_KEYS:
            self.assertNotIn(f'"{forbidden}"', text.lower())
        for lane in report["lanes"].values():
            timing = lane["raw_timing_seconds"]
            self.assertIn("INCLUDES_CORE_PRIMARY_MAP", timing["scope"])
            self.assertTrue(timing["jx_core_mandatory_semantic_replay_included"])
            self.assertFalse(timing["outer_report_validation_replay_included"])
            self.assertFalse(timing["timing_comparable"])
        self.assertEqual(
            report["timing_disclosure"][
                "mandatory_report_validation_replay_execution_count_per_requested_lane"
            ],
            1,
        )

    def test_outer_replay_rejects_coherent_lane_state_mutation(self):
        lane = self.study.jx_lanes[0]
        positions = lane.positions.copy()
        positions[-1, -1, 0] += 1.0e-12
        changed_lane = dataclasses.replace(
            lane, positions=benchmark._owned_readonly_float64(positions)
        )
        changed = dataclasses.replace(
            self.study,
            jx_lanes=(changed_lane,) + self.study.jx_lanes[1:],
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "independent replay"):
            benchmark.build_report(changed)


class ExactReboundComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            rebound = benchmark._load_rebound()
        except benchmark.ReboundUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if rebound.__version__ != benchmark.REQUIRED_REBOUND_VERSION:
            raise unittest.SkipTest("exact REBOUND 5.1.1 is unavailable")

    def test_smoke_settings_clock_reference_separation_and_mutation_custody(self):
        rebound = benchmark._load_rebound()
        implicit_default = float(rebound.Simulation().dt)
        configured = benchmark._configured_ias15_simulation(rebound)
        self.assertNotEqual(implicit_default.hex(), benchmark.IAS15_INITIAL_DT.hex())
        self.assertEqual(
            float(configured.dt).hex(), benchmark.IAS15_INITIAL_DT.hex()
        )
        study = benchmark._execute_study(benchmark.Profile(), "required")
        report = benchmark.build_report(study)
        self.assertEqual(len(report["lanes"]), 6)
        self.assertEqual(len(report["ias15_numerical_references"]), 5)
        for lane_id in (
            "rebound_whfast_short_p64",
            "rebound_whfast_short_p128",
            "rebound_whfast_long_p64",
        ):
            lane = report["lanes"][lane_id]
            settings = lane["settings"]
            self.assertEqual(settings["integrator"], "whfast")
            self.assertEqual(settings["coordinates"], "JACOBI")
            self.assertEqual(settings["corrector"], 0)
            self.assertEqual(settings["corrector2"], 0)
            self.assertEqual(settings["kernel"], "DEFAULT")
            self.assertEqual(settings["safe_mode"], 1)
            self.assertEqual(settings["keep_unsynchronized"], 0)
            self.assertEqual(settings["N_active"], 3)
            self.assertEqual(settings["testparticle_type"], 0)
            accounting = lane["accounting"]
            self.assertEqual(
                accounting["observed_epoch_hex"],
                accounting["two_half_clock_replay_hex"],
            )
            self.assertTrue(
                accounting["state_synchronized_at_every_retained_checkpoint"]
            )
            self.assertFalse(accounting["integrate_to_endpoint_used"])
            self.assertIn("declared_grid_state_mismatch_without_epoch_relabeling", lane)
        self.assertTrue(
            report["short_convergence"]["jx_wisdom_holman"][
                "each_ratio_within_expected_second_order_window"
            ]
        )
        self.assertTrue(
            report["short_convergence"]["rebound_whfast"][
                "each_ratio_within_expected_second_order_window"
            ]
        )

        lane = study.whfast_lanes[0]
        changed_settings = {**lane.settings, "G": 2.0}
        changed_lane = dataclasses.replace(lane, settings=changed_settings)
        changed_study = dataclasses.replace(
            study, whfast_lanes=(changed_lane,) + study.whfast_lanes[1:]
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "independent replay"):
            benchmark.build_report(changed_study)

        reference = study.ias15_lanes[0]
        self.assertEqual(
            reference.settings["requested_initial_dt_hex"],
            benchmark.IAS15_INITIAL_DT.hex(),
        )
        self.assertEqual(
            reference.settings["effective_initial_dt_hex"],
            benchmark.IAS15_INITIAL_DT.hex(),
        )
        changed_reference = dataclasses.replace(
            reference,
            settings={**reference.settings, "effective_initial_dt": 0.001},
        )
        changed_study = dataclasses.replace(
            study,
            ias15_lanes=(changed_reference,) + study.ias15_lanes[1:],
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "independent replay"):
            benchmark.build_report(changed_study)

    def test_named_full_profile_regression(self):
        report = benchmark.run_comparison(benchmark.Profile.full(), "required")
        self.assertTrue(report["profile"]["named_full_profile"])
        self.assertEqual(report["profile"]["short_periods"], 10)
        self.assertEqual(report["profile"]["long_periods"], 100)
        for method in ("jx_wisdom_holman", "rebound_whfast"):
            convergence = report["short_convergence"][method]
            self.assertTrue(
                convergence["each_ratio_within_expected_second_order_window"]
            )
            for value in convergence["ratios"].values():
                self.assertGreaterEqual(value, 3.7)
                self.assertLessEqual(value, 4.3)

        jx = report["lanes"]["jx_wh_long_p64"]
        whfast = report["lanes"]["rebound_whfast_long_p64"]
        self.assertTrue(jx["within_long_descriptive_envelope"])
        self.assertTrue(whfast["within_long_descriptive_envelope"])
        self.assertEqual(jx["accounting"]["completed_steps"], 6400)
        self.assertEqual(jx["accounting"]["primary_map_force_evaluations"], 6401)
        self.assertEqual(
            jx["accounting"]["validation_replay_force_evaluations"], 6401
        )
        self.assertEqual(
            jx["accounting"]["total_public_call_force_evaluations"], 12802
        )
        self.assertEqual(
            whfast["accounting"]["requested_integer_steps"], 6400
        )
        self.assertEqual(
            whfast["accounting"]["completed_integer_steps"], 6400
        )
        self.assertGreater(whfast["maximum_abs_clock_drift"], 0.0)
        runtime = report["runtime_provenance"]["rebound"]
        self.assertEqual(runtime["distribution_version"], "5.1.1")
        self.assertEqual(runtime["python_sources"]["file_count"], 67)
        self.assertFalse(runtime["legal_or_redistribution_conclusion_authorized"])


if __name__ == "__main__":
    unittest.main()
