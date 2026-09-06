import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "rebound_hybrid_comparison.py"
SPEC = importlib.util.spec_from_file_location("rebound_hybrid_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class ContractAndSerializationTests(unittest.TestCase):
    @staticmethod
    def _literal_lane():
        step = float(benchmark.PERIOD / 256.0)
        declared = benchmark._readonly(np.array((0.0, step)), (2,))
        observed = benchmark._readonly(np.array((0.0, step)), (2,))
        positions = benchmark._readonly(
            np.array(
                (
                    ((0.0, -0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                )
            ),
            (2, 3, 3),
        )
        velocities = benchmark._readonly(
            np.array(
                (
                    ((0.0, 0.0, 0.0), (-0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
                    ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
                )
            ),
            (2, 3, 3),
        )
        return benchmark.Lane(
            engine_id="literal.lane",
            family="LITERAL",
            method_id="literal.method.v1",
            divisor=256,
            fixed_step=step,
            step_count=1,
            declared_epochs=declared,
            observed_epochs=observed,
            positions=positions,
            velocities=velocities,
            settings={"alpha": -0.0, "nested": ("lane", 1)},
            accounting={"count": 1, "zero": 0.0},
            runtime={"enabled": False, "mode": "literal"},
            raw_timing_seconds=0.0,
            final_state_sha256=benchmark._final_state_sha256(
                positions[-1], velocities[-1]
            ),
            trajectory_sha256=benchmark._trajectory_sha256(
                observed, positions, velocities
            ),
        )

    def test_exact_public_roster_and_lazy_optional_import(self):
        self.assertEqual(
            benchmark.__all__,
            (
                "BENCHMARK_ID",
                "BenchmarkError",
                "Profile",
                "ReboundUnavailable",
                "SCHEMA",
                "build_report",
                "main",
                "run_comparison",
            ),
        )
        code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('lazy_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'rebound' not in sys.modules
assert module.Profile().name == 'smoke'
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "PYTHONNOUSERSITE": "1",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_is_dependency_free_and_describes_opt_in_full_profile(self):
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
        self.assertIn("--rebound-mode", result.stdout)
        self.assertIn("--include-limitations", result.stdout)
        self.assertIn("P/256,P/512,P/1024", result.stdout)

    def test_locked_cartesian_fixture_and_exact_lattices(self):
        rows = benchmark._initial_rows()
        self.assertEqual(rows.shape, (3, 8))
        self.assertEqual(
            hashlib.sha256(rows.astype("<f8").tobytes(order="C")).hexdigest(),
            benchmark.INITIAL_RAW_SHA256,
        )
        self.assertEqual(
            tuple(tuple(float(value).hex() for value in row) for row in rows),
            benchmark.INITIAL_ROWS_HEX,
        )
        full = benchmark.Profile.full()
        self.assertEqual(full.jx_divisors, (256, 512, 1024))
        self.assertEqual(tuple(full.step_count(value) for value in full.jx_divisors), (1630, 3260, 6520))
        self.assertEqual(tuple(len(full.jx_checkpoint_indices(value)) for value in full.jx_divisors), (103, 103, 103))
        self.assertEqual(len(benchmark.Profile().jx_checkpoint_indices(256)), 9)
        self.assertEqual(
            benchmark.JX_FINEST_ENVELOPE["phase_max_abs_radians"], 3.0e-4
        )
        self.assertEqual(
            tuple(benchmark.JX_FINAL_STATE_SHA256), (256, 512, 1024)
        )
        self.assertEqual(
            tuple(benchmark.JX_TRAJECTORY_SHA256), (256, 512, 1024)
        )
        with self.assertRaises(ValueError):
            benchmark.Profile(name=True)
        with self.assertRaises(ValueError):
            benchmark.Profile(include_limitations=1)

    def test_limitation_absence_reason_distinguishes_disabled_and_unavailable(self):
        profile = benchmark.Profile(include_limitations=True)
        disabled = benchmark._limitation_probes(profile, None, "disabled")
        unavailable = benchmark._limitation_probes(profile, None, "auto")
        self.assertEqual(disabled["reason"], "EXPLICITLY_DISABLED")
        self.assertEqual(
            unavailable["reason"], "EXACT_REBOUND_5_1_1_UNAVAILABLE"
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._limitation_probes(profile, None, "DISABLED")

    def test_literal_report_domain_preimage_and_signed_zero_mutations(self):
        value = {
            "array": np.array([[0.0, -0.0]], dtype=np.float64),
            "float": -0.0,
            "tuple": (1, "x"),
        }
        literal_json = (
            b'{"array":{"dtype":"float64","shape":[1,2],"values":'
            b'["0x0.0p+0","-0x0.0p+0"]},"float":{"float_hex":'
            b'"-0x0.0p+0"},"tuple":[1,"x"]}'
        )
        literal_preimage = b"jx.rebound-hybrid.report-kat.v1\0" + literal_json
        self.assertEqual(len(literal_json), 127)
        self.assertEqual(len(literal_preimage), 159)
        self.assertEqual(benchmark._canonical_json(value), literal_json)
        self.assertEqual(
            hashlib.sha256(literal_preimage).hexdigest(),
            "1ba0fd508225ecd71fe5e5323525bacc97611c5ad4f86e353e0a982df163f681",
        )
        self.assertEqual(
            benchmark._domain_sha256("jx.rebound-hybrid.report-kat.v1", value),
            "1ba0fd508225ecd71fe5e5323525bacc97611c5ad4f86e353e0a982df163f681",
        )
        changed = copy.deepcopy(value)
        changed["float"] = 0.0
        self.assertNotEqual(
            benchmark._domain_sha256("jx.rebound-hybrid.report-kat.v1", changed),
            benchmark._domain_sha256("jx.rebound-hybrid.report-kat.v1", value),
        )
        self.assertNotEqual(
            benchmark._domain_sha256("jx.rebound-hybrid.report-kat.v2", value),
            benchmark._domain_sha256("jx.rebound-hybrid.report-kat.v1", value),
        )

    def test_literal_array_component_preimage_and_layout_mutations(self):
        value = np.array([[0.0, -0.0]], dtype=np.float64)
        literal_header = b'{"dtype":"float64","shape":[1,2]}'
        literal_preimage = (
            b"jx.rebound-hybrid.positions.v1\0"
            + (33).to_bytes(8, "big")
            + literal_header
            + (16).to_bytes(8, "big")
            + b"\x00\x00\x00\x00\x00\x00\x00\x00"
            + b"\x00\x00\x00\x00\x00\x00\x00\x80"
        )
        self.assertEqual(len(literal_preimage), 96)
        expected = "c688ff55495dcdd020151cd4470293df4155191278ab73af75094e6a8bddd645"
        self.assertEqual(hashlib.sha256(literal_preimage).hexdigest(), expected)
        self.assertEqual(
            benchmark._array_component_sha256(
                "jx.rebound-hybrid.positions.v1", value
            ),
            expected,
        )
        self.assertNotEqual(
            benchmark._array_component_sha256(
                "jx.rebound-hybrid.positions.v1", value.reshape(2, 1)
            ),
            expected,
        )
        changed = value.copy()
        changed[0, 1] = 0.0
        self.assertNotEqual(
            benchmark._array_component_sha256(
                "jx.rebound-hybrid.positions.v1", changed
            ),
            expected,
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._array_component_sha256(
                "jx.rebound-hybrid.positions.v1", value.astype(np.float32)
            )

    def test_literal_production_lane_content_preimage_and_mutations(self):
        lane = self._literal_lane()
        literal_payload = {
            "schema": "jx.rebound-hybrid.lane-content.payload.v1",
            "engine_id": "literal.lane",
            "family": "LITERAL",
            "method_id": "literal.method.v1",
            "divisor": 256,
            "fixed_step": float.fromhex("0x1.921fb54442d18p-6"),
            "step_count": 1,
            "declared_epochs_content_sha256": (
                "b7877a5b639b7dc0dbb357af84b4895d643d9d48309e919081e29e2987049ce1"
            ),
            "observed_epochs_content_sha256": (
                "0fa4646c8a8b71f8d907e4d960b719da8f36d454aa1fd67631caa5e01d2c30c9"
            ),
            "positions_content_sha256": (
                "ab6bb99a63d3355efe43f5057c714a41f2e74e8801d4e80cc0e846fac068fc3d"
            ),
            "velocities_content_sha256": (
                "654d2e6d4e0b9ac634177f3006a99bb5183f151481c64de98a784c80c35bdef1"
            ),
            "settings": {
                "alpha": {"float_hex": "-0x0.0p+0"},
                "nested": ("lane", 1),
            },
            "accounting": {"count": 1, "zero": {"float_hex": "0x0.0p+0"}},
            "runtime": {"enabled": False, "mode": "literal"},
            "final_state_sha256": (
                "bc3d0c30ffc70770e53ef97a838e25dedb6ad3ec3fadd02eb227eae4de7bd9da"
            ),
            "trajectory_sha256": (
                "7328b77ee9cd6e1f14b9213792146e682f10317c17e0b8e3467e7fa9c6628705"
            ),
        }
        literal_json = (
            b'{"accounting":{"count":1,"zero":{"float_hex":"0x0.0p+0"}},'
            b'"declared_epochs_content_sha256":"b7877a5b639b7dc0dbb357af84b4895d'
            b'643d9d48309e919081e29e2987049ce1","divisor":256,"engine_id":"literal.lane",'
            b'"family":"LITERAL","final_state_sha256":"bc3d0c30ffc70770e53ef97a838e25de'
            b'db6ad3ec3fadd02eb227eae4de7bd9da","fixed_step":{"float_hex":"0x1.921f'
            b'b54442d18p-6"},"method_id":"literal.method.v1","observed_epochs_content_sha256":'
            b'"0fa4646c8a8b71f8d907e4d960b719da8f36d454aa1fd67631caa5e01d2c30c9",'
            b'"positions_content_sha256":"ab6bb99a63d3355efe43f5057c714a41f2e74e8801d4e80c'
            b'c0e846fac068fc3d","runtime":{"enabled":false,"mode":"literal"},"schema":"jx.'
            b'rebound-hybrid.lane-content.payload.v1","settings":{"alpha":{"float_hex":"-0x0.'
            b'0p+0"},"nested":["lane",1]},"step_count":1,"trajectory_sha256":"7328b77ee9cd6'
            b'e1f14b9213792146e682f10317c17e0b8e3467e7fa9c6628705","velocities_content_sha256"'
            b':"654d2e6d4e0b9ac634177f3006a99bb5183f151481c64de98a784c80c35bdef1"}'
        )
        literal_preimage = b"jx.rebound-hybrid.lane-content.v1\0" + literal_json
        expected = "0538e6639882c213edadf4a7960742e90a67735750a687c7b4fa623e96485396"
        self.assertEqual(len(literal_json), 944)
        self.assertEqual(len(literal_preimage), 978)
        self.assertEqual(benchmark._lane_content_payload(lane), literal_payload)
        self.assertEqual(benchmark._canonical_json(literal_payload), literal_json)
        self.assertEqual(hashlib.sha256(literal_preimage).hexdigest(), expected)
        self.assertEqual(benchmark._lane_content_sha256(lane), expected)
        mutations = (
            ("schema", "jx.rebound-hybrid.lane-content.payload.v2"),
            ("declared_epochs_content_sha256", "0" * 64),
            ("positions_content_sha256", "1" * 64),
            ("velocities_content_sha256", "2" * 64),
            ("settings", {"alpha": {"float_hex": "0x0.0p+0"}, "nested": ("lane", 1)}),
            ("settings", {"alpha": {"float_hex": "-0x0.0p+0"}, "nested": (1, "lane")}),
            ("accounting", {"count": 2, "zero": {"float_hex": "0x0.0p+0"}}),
            ("runtime", {"enabled": False, "mode": "changed"}),
        )
        for field, replacement in mutations:
            changed = copy.deepcopy(literal_payload)
            changed[field] = replacement
            self.assertNotEqual(
                benchmark._domain_sha256(
                    "jx.rebound-hybrid.lane-content.v1", changed
                ),
                expected,
            )
        self.assertNotEqual(
            benchmark._domain_sha256(
                "jx.rebound-hybrid.lane-content.v2", literal_payload
            ),
            expected,
        )

    def test_literal_production_report_semantic_preimage_and_manifest_mutations(self):
        lane_digest = "0538e6639882c213edadf4a7960742e90a67735750a687c7b4fa623e96485396"
        literal_payload = {
            "schema": benchmark.SCHEMA,
            "benchmark_id": benchmark.BENCHMARK_ID,
            "profile": {"name": "literal"},
            "problem": {"fixture": "literal"},
            "method_provenance": {},
            "runtime_provenance": {},
            "headline_lanes": {},
            "numerical_references": {},
            "refinement": {},
            "regression_gates": {"evaluated": False, "passed": None},
            "all_far_control": {},
            "limitations": {},
            "timing_disclosure": {"timing_is_diagnostic_only": True},
            "interpretation": {"signed_zero": -0.0},
            "claim_controls": {
                "qualification_or_superiority_authorized": False
            },
            "serialization": {
                "canonical_json": (
                    "SORTED_ASCII_JSON_WITH_BINARY64_FLOAT_HEX_OBJECTS"
                )
            },
            "content_integrity": {
                "classification": "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
                "authority_authorized": False,
                "array_component_domain_roster": [
                    "jx.rebound-hybrid.declared-epochs.v1",
                    "jx.rebound-hybrid.observed-epochs.v1",
                    "jx.rebound-hybrid.positions.v1",
                    "jx.rebound-hybrid.velocities.v1",
                ],
                "lane_content_domain": "jx.rebound-hybrid.lane-content.v1",
                "report_semantic_content_domain": (
                    "jx.rebound-hybrid.report-semantic-content.v1"
                ),
                "ordered_lane_ids": ["literal.lane"],
                "ordered_lane_content_sha256": [lane_digest],
                "aggregate_digest_excludes_itself": True,
                "raw_timing_fields_excluded": [
                    "raw_primary_lane_timing_seconds",
                    "summed_primary_lane_timing_seconds",
                    "semantic_replay_wall_seconds",
                ],
            },
        }
        literal_json = (
            b'{"all_far_control":{},"benchmark_id":"jx.rebound_hybrid.all_active_close_scatter.v1",'
            b'"claim_controls":{"qualification_or_superiority_authorized":false},"content_integrity":'
            b'{"aggregate_digest_excludes_itself":true,"array_component_domain_roster":['
            b'"jx.rebound-hybrid.declared-epochs.v1","jx.rebound-hybrid.observed-epochs.v1",'
            b'"jx.rebound-hybrid.positions.v1","jx.rebound-hybrid.velocities.v1"],'
            b'"authority_authorized":false,"classification":"UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",'
            b'"lane_content_domain":"jx.rebound-hybrid.lane-content.v1",'
            b'"ordered_lane_content_sha256":["0538e6639882c213edadf4a7960742e90a67735750a687c7b4fa623e96485396"],'
            b'"ordered_lane_ids":["literal.lane"],"raw_timing_fields_excluded":['
            b'"raw_primary_lane_timing_seconds","summed_primary_lane_timing_seconds",'
            b'"semantic_replay_wall_seconds"],'
            b'"report_semantic_content_domain":"jx.rebound-hybrid.report-semantic-content.v1"},'
            b'"headline_lanes":{},"interpretation":{"signed_zero":{"float_hex":"-0x0.0p+0"}},'
            b'"limitations":{},"method_provenance":{},"numerical_references":{},'
            b'"problem":{"fixture":"literal"},"profile":{"name":"literal"},"refinement":{},'
            b'"regression_gates":{"evaluated":false,"passed":null},"runtime_provenance":{},'
            b'"schema":"jx.rebound_hybrid.close_scatter.v1","serialization":'
            b'{"canonical_json":"SORTED_ASCII_JSON_WITH_BINARY64_FLOAT_HEX_OBJECTS"},'
            b'"timing_disclosure":{"timing_is_diagnostic_only":true}}'
        )
        literal_preimage = (
            b"jx.rebound-hybrid.report-semantic-content.v1\0" + literal_json
        )
        expected = "4acc6265dd6ee881fe70178e85dda727b8a57cfc77acc853d1ba3be09dd8d8c4"
        self.assertEqual(tuple(literal_payload), benchmark.REPORT_FIELD_ROSTER)
        self.assertEqual(len(literal_json), 1373)
        self.assertEqual(len(literal_preimage), 1418)
        self.assertEqual(benchmark._canonical_json(literal_payload), literal_json)
        self.assertEqual(hashlib.sha256(literal_preimage).hexdigest(), expected)
        self.assertEqual(
            benchmark._domain_sha256(
                "jx.rebound-hybrid.report-semantic-content.v1", literal_payload
            ),
            expected,
        )
        with_timing = copy.deepcopy(literal_payload)
        with_timing["timing_disclosure"].update(
            {
                "summed_primary_lane_timing_seconds": 2.0,
                "semantic_replay_wall_seconds": 3.0,
            }
        )
        self.assertEqual(benchmark._strip_timing(with_timing), literal_payload)
        self.assertEqual(
            benchmark._domain_sha256(
                "jx.rebound-hybrid.report-semantic-content.v1",
                benchmark._strip_timing(with_timing),
            ),
            expected,
        )
        mutations = []
        changed = copy.deepcopy(literal_payload)
        changed["interpretation"]["signed_zero"] = 0.0
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["content_integrity"]["lane_content_domain"] += ".changed"
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["content_integrity"]["ordered_lane_ids"] = ["other", "literal.lane"]
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["content_integrity"]["array_component_domain_roster"].reverse()
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["content_integrity"]["raw_timing_fields_excluded"].reverse()
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["content_integrity"]["report_semantic_content_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(literal_payload)
        changed["problem"]["fixture"] = "changed"
        mutations.append(changed)
        for changed in mutations:
            self.assertNotEqual(
                benchmark._domain_sha256(
                    "jx.rebound-hybrid.report-semantic-content.v1", changed
                ),
                expected,
            )
        self.assertNotEqual(
            benchmark._domain_sha256(
                "jx.rebound-hybrid.report-semantic-content.v2", literal_payload
            ),
            expected,
        )

    def test_canonical_preflight_rejects_before_json_conversion(self):
        with mock.patch.object(benchmark, "MAXIMUM_CANONICAL_BYTES", 100), mock.patch.object(
            benchmark, "_json_value", side_effect=AssertionError("encoder called")
        ):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "cumulative"):
                benchmark._canonical_json({"value": "x" * 40})
        with self.assertRaisesRegex(benchmark.BenchmarkError, "bit cap"):
            benchmark._canonical_json({"value": 1 << 4097})
        with self.assertRaisesRegex(benchmark.BenchmarkError, "valid Unicode"):
            benchmark._canonical_json({"value": "\ud800"})
        with mock.patch.object(
            benchmark,
            "_strict_utf8_length",
            side_effect=AssertionError("UTF-8 encoder called"),
        ):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "code-point cap"):
                benchmark._canonical_json("x" * 16_385)


class MetricAndCustodyTests(unittest.TestCase):
    def test_metric_formulas_and_phase_proxy_labels(self):
        rows = benchmark._initial_rows()
        positions = np.stack((rows[:, 2:5], rows[:, 2:5]))
        velocities = np.stack((rows[:, 5:8], rows[:, 5:8]))
        cartesian = benchmark._cartesian_error(positions, positions.copy())
        self.assertEqual(cartesian["vector_l2_max"], 0.0)
        phase = benchmark._phase_error(positions, positions.copy())
        self.assertEqual(phase["maximum_abs_radians"], 0.0)
        self.assertEqual(
            phase["sign_convention"],
            "ATAN2_CROSS_CANDIDATE_REFERENCE_REFERENCE_MINUS_CANDIDATE",
        )
        invariants = benchmark._invariant_metrics(positions, velocities)
        self.assertEqual(invariants["relative_total_energy_max"], 0.0)
        self.assertIn("center_of_mass_position_absolute_max", invariants)
        self.assertIn("center_of_mass_position_drift_max", invariants)
        changed = positions.copy()
        changed[1, 0] += np.array((3.0, 4.0, 0.0))
        nonzero = benchmark._cartesian_error(changed, positions)
        self.assertEqual(nonzero["vector_l2_max"], 5.0)
        self.assertAlmostEqual(nonzero["vector_l2_rms"], 5.0 / math.sqrt(6.0))
        candidate = np.zeros((2, 3, 3), dtype=np.float64)
        reference = np.zeros_like(candidate)
        candidate[:, 2, 1] = 1.0
        reference[:, 2, 0] = 1.0
        signed = benchmark._phase_error(candidate, reference)
        self.assertAlmostEqual(signed["final_signed_radians"], -math.pi / 2.0)

    def test_lane_owns_disjoint_readonly_arrays_and_rejects_views(self):
        rows = benchmark._initial_rows()
        step = benchmark.PERIOD / 256.0
        epoch = benchmark._readonly(np.array((0.0, step)), (2,))
        observed = benchmark._readonly(np.array((0.0, step)), (2,))
        positions = benchmark._readonly(
            np.stack((rows[:, 2:5], rows[:, 2:5])), (2, 3, 3)
        )
        velocities = benchmark._readonly(
            np.stack((rows[:, 5:8], rows[:, 5:8])), (2, 3, 3)
        )
        lane = benchmark.Lane(
            engine_id="fixture",
            family="FIXTURE",
            method_id="fixture.method",
            divisor=256,
            fixed_step=float(step),
            step_count=1,
            declared_epochs=epoch,
            observed_epochs=observed,
            positions=positions,
            velocities=velocities,
            settings={"kind": "fixture"},
            accounting={"count": 1},
            runtime={"kind": "fixture"},
            raw_timing_seconds=0.0,
            final_state_sha256=benchmark._final_state_sha256(
                positions[-1], velocities[-1]
            ),
            trajectory_sha256=benchmark._trajectory_sha256(
                observed, positions, velocities
            ),
        )
        arrays = (
            lane.declared_epochs,
            lane.observed_epochs,
            lane.positions,
            lane.velocities,
        )
        for value in arrays:
            self.assertIs(type(value), np.ndarray)
            self.assertTrue(value.flags.owndata)
            self.assertFalse(value.flags.writeable)
            self.assertIsNone(value.base)
        self.assertFalse(
            any(
                np.shares_memory(left, right)
                for index, left in enumerate(arrays)
                for right in arrays[index + 1 :]
            )
        )
        owner = np.array(lane.positions, copy=True)
        view = owner.view()
        view.setflags(write=False)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "custody"):
            dataclasses.replace(lane, positions=view)
        settings_backing = {"kind": "fixture", "nested": [1, 2]}
        sealed = dataclasses.replace(
            lane, settings=benchmark.MappingProxyType(settings_backing)
        )
        sealed_bytes = benchmark._canonical_json(sealed.settings)
        settings_backing["kind"] = "mutated"
        settings_backing["nested"].append(3)
        self.assertEqual(benchmark._canonical_json(sealed.settings), sealed_bytes)
        positions_backing = np.array(lane.positions, copy=True)
        positions_backing.setflags(write=False)
        array_sealed = dataclasses.replace(lane, positions=positions_backing)
        retained_bytes = array_sealed.positions.tobytes(order="C")
        positions_backing.setflags(write=True)
        positions_backing[0, 0, 0] = math.nextafter(
            float(positions_backing[0, 0, 0]), math.inf
        )
        self.assertEqual(
            array_sealed.positions.tobytes(order="C"), retained_bytes
        )


class SmokeStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = benchmark._execute_study(benchmark.Profile(), "disabled")
        cls.report = benchmark.build_report(cls.primary)

    def test_smoke_report_is_nine_node_jx_only_and_claims_are_false(self):
        report = self.report
        self.assertEqual(tuple(report), benchmark.REPORT_FIELD_ROSTER)
        self.assertEqual(report["interpretation"]["headline_retained_node_count"], 9)
        self.assertFalse(report["interpretation"]["headline_metrics_share_103_labels"])
        self.assertFalse(
            report["runtime_provenance"]["external_status"][
                "external_lanes_executed"
            ]
        )
        self.assertIsNone(report["regression_gates"]["passed"])
        for value in report["claim_controls"].values():
            self.assertIs(value, False)
        self.assertNotIn("winner", json.dumps(report, sort_keys=True))

    def test_all_far_control_binds_state_epochs_diagnostics_and_nineteen_counts(self):
        control = self.report["all_far_control"]
        self.assertEqual(control["case_count"], 4)
        self.assertFalse(control["analytic_accuracy_claimed"])
        for case in control["cases"]:
            self.assertTrue(case["checkpoint_state_bytes_equal"])
            self.assertTrue(case["checkpoint_epoch_bytes_equal"])
            self.assertTrue(case["diagnostics_equal"])
            self.assertTrue(case["public_wh_accounting_projection_equal"])
            self.assertEqual(
                set(case["nineteen_counter_projection"]),
                set(benchmark.HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER),
            )

    def test_report_digest_survives_sorted_json_and_rejects_manifest_mutation(self):
        loaded = json.loads(json.dumps(self.report, sort_keys=True))
        benchmark._validate_report_content_integrity(loaded)
        changed = copy.deepcopy(loaded)
        changed["content_integrity"]["classification"] = "FORGED"
        with self.assertRaisesRegex(benchmark.BenchmarkError, "digest changed"):
            benchmark._validate_report_content_integrity(changed)
        timing = copy.deepcopy(loaded)
        timing["timing_disclosure"]["summed_primary_lane_timing_seconds"] += 1.0
        benchmark._validate_report_content_integrity(timing)
        lane_timing = copy.deepcopy(loaded)
        lane_id = next(iter(lane_timing["headline_lanes"]))
        lane_timing["headline_lanes"][lane_id][
            "raw_primary_lane_timing_seconds"
        ] += 1.0
        benchmark._validate_report_content_integrity(lane_timing)
        for container, key in (
            ("timing_disclosure", "summed_primary_lane_timing_seconds"),
            ("timing_disclosure", "semantic_replay_wall_seconds"),
            ("headline_lanes", "raw_primary_lane_timing_seconds"),
        ):
            missing = copy.deepcopy(loaded)
            invalid = copy.deepcopy(loaded)
            if container == "headline_lanes":
                del missing[container][lane_id][key]
                invalid[container][lane_id][key] = "not-a-float"
            else:
                del missing[container][key]
                invalid[container][key] = "not-a-float"
            with self.subTest(container=container, key=key, mutation="missing"):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "timing field"):
                    benchmark._validate_report_content_integrity(missing)
            with self.subTest(container=container, key=key, mutation="type"):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "timing field"):
                    benchmark._validate_report_content_integrity(invalid)
        for section, key in (
            ("problem", "summed_primary_lane_timing_seconds"),
            ("claim_controls", "raw_primary_lane_timing_seconds"),
            ("limitations", "semantic_replay_wall_seconds"),
        ):
            changed = copy.deepcopy(loaded)
            changed[section][key] = {"not_timing": "forged"}
            with self.subTest(section=section, key=key), self.assertRaisesRegex(
                benchmark.BenchmarkError, "digest changed"
            ):
                benchmark._validate_report_content_integrity(changed)

    def test_saved_report_envelope_precedes_timing_and_digest_traversal(self):
        loaded = json.loads(json.dumps(self.report, sort_keys=True))
        oversized = copy.deepcopy(loaded)
        oversized["headline_lanes"] = {
            f"lane_{index}": {} for index in range(257)
        }
        cyclic = copy.deepcopy(loaded)
        cyclic["problem"]["cycle"] = cyclic["problem"]
        with mock.patch.object(
            benchmark, "_strip_timing", side_effect=AssertionError("strip called")
        ):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "entry cap"):
                benchmark._validate_report_content_integrity(oversized)
            with self.assertRaisesRegex(benchmark.BenchmarkError, "depth cap"):
                benchmark._validate_report_content_integrity(cyclic)

    def test_independent_replay_rejects_coherent_state_mutation(self):
        lane = self.primary.jx_lanes[0]
        positions = np.array(lane.positions, copy=True)
        positions[-1, 0, 0] = math.nextafter(float(positions[-1, 0, 0]), math.inf)
        positions.setflags(write=False)
        changed_lane = dataclasses.replace(
            lane,
            positions=positions,
            final_state_sha256=benchmark._final_state_sha256(
                positions[-1], lane.velocities[-1]
            ),
            trajectory_sha256=benchmark._trajectory_sha256(
                lane.observed_epochs, positions, lane.velocities
            ),
        )
        changed = dataclasses.replace(self.primary, jx_lanes=(changed_lane,))
        with self.assertRaisesRegex(benchmark.BenchmarkError, "semantic replay"):
            benchmark.build_report(changed)

    def test_independent_replay_rejects_coherent_lane_mapping_rewrites(self):
        lane = self.primary.jx_lanes[0]
        for field in ("settings", "accounting", "runtime"):
            replacement = benchmark._plain_tree(getattr(lane, field))
            replacement[f"coherent_{field}_rewrite"] = True
            changed_lane = dataclasses.replace(lane, **{field: replacement})
            with self.subTest(field=field), self.assertRaises(
                benchmark.BenchmarkError
            ):
                changed = dataclasses.replace(
                    self.primary, jx_lanes=(changed_lane,)
                )
                benchmark._validate_study_against_replay(changed, self.primary)

    def test_study_mappingproxy_top_level_cap_precedes_nested_traversal(self):
        at_cap = benchmark.MappingProxyType(
            {f"field_{index}": index for index in range(64)}
        )
        over_cap = benchmark.MappingProxyType(
            {f"field_{index}": index for index in range(65)}
        )
        with self.assertRaisesRegex(
            benchmark.BenchmarkError, "external execution status"
        ):
            dataclasses.replace(self.primary, external_status=at_cap)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "field cap"):
            dataclasses.replace(self.primary, external_status=over_cap)

    def test_study_mappingproxy_inputs_are_deep_copied_before_retention(self):
        status_backing = benchmark._plain_tree(self.primary.external_status)
        all_far_backing = benchmark._plain_tree(self.primary.all_far_control)
        sealed = dataclasses.replace(
            self.primary,
            external_status=benchmark.MappingProxyType(status_backing),
            all_far_control=benchmark.MappingProxyType(all_far_backing),
        )
        status_bytes = benchmark._canonical_json(sealed.external_status)
        all_far_bytes = benchmark._canonical_json(sealed.all_far_control)
        status_backing["reason"] = "FORGED_AFTER_CONSTRUCTION"
        all_far_backing["cases"].clear()
        self.assertEqual(
            benchmark._canonical_json(sealed.external_status), status_bytes
        )
        self.assertEqual(
            benchmark._canonical_json(sealed.all_far_control), all_far_bytes
        )


class OptionalPinnedLimitationTests(unittest.TestCase):
    def test_exact_tunnel_and_central_periapse_witnesses(self):
        try:
            rebound = benchmark._load_rebound()
        except benchmark.ReboundUnavailable as exc:
            self.skipTest(str(exc))
        tunnel = benchmark._tunneling_limitation_probe(rebound)
        self.assertEqual(
            tunnel["classification"], "ENDPOINT_TUNNELING_LIMITATION_WITNESS"
        )
        self.assertEqual(
            tunnel["straight_line_relative_motion_crossing_epoch"].hex(),
            "0x1.0000000000000p-2",
        )
        self.assertFalse(tunnel["straight_line_epoch_is_exact_newtonian_root"])
        self.assertFalse(tunnel["event_or_root_location_claimed"])
        self.assertEqual(
            tunnel["configured_fixed_step"].hex(), "0x1.0000000000000p-1"
        )
        self.assertEqual(
            tunnel["configured_exit_min_distance"].hex(),
            "0x1.999999999999ap-4",
        )
        for method in ("mercurius", "trace"):
            outcome = tunnel["configured_exit_min_distance_outcomes"][method]
            self.assertFalse(outcome["exception_raised"])
            self.assertEqual(
                outcome["initial_effective_readback"]["fixed_step"].hex(),
                "0x1.0000000000000p-1",
            )
            self.assertEqual(
                outcome["post_step_effective_readback"][
                    "exit_min_distance"
                ].hex(),
                "0x1.999999999999ap-4",
            )
            bracket = outcome["signed_relative_y_bracket"]
            self.assertEqual(
                bracket["initial_relative_y"].hex(), "0x1.0000000000000p+1"
            )
            self.assertEqual(
                bracket["endpoint_relative_y"].hex(),
                "-0x1.000000b2f0dbcp+1",
            )
            self.assertTrue(bracket["strict_sign_change"])
        self.assertEqual(tunnel["line_collision_exception_type"], "Collision")
        central = benchmark._central_periapse_limitation_probe(rebound)
        self.assertEqual(
            central["classification"],
            "CENTRAL_PERIAPSE_METHOD_SCOPE_LIMITATION",
        )
        protocol = central["ias15_reference_protocol"]
        self.assertEqual(
            central["final_state_raw_sha256"],
            {
                "mercurius": (
                    "4ec290025f0cbdc4c62ca9563f0d427d3bf20424344edf9b842357d98b2d4308"
                ),
                "trace": (
                    "624bdf8c7e885866d82f7987a4cc526f29bc76affd844c6d247432d8cb1cec09"
                ),
                "ias15": (
                    "a1258fe7578115a7697fcc07dbd3aacfe5d38dc0c08f55b4f3a3b0ed2393ffc0"
                ),
            },
        )
        self.assertEqual(
            central["final_state_raw_sha256_payload"],
            "EXACT_2_BY_8_ROW_MAJOR_LITTLE_ENDIAN_FLOAT64_M_R_X_Y_Z_VX_VY_VZ",
        )
        self.assertEqual(
            protocol["target_epoch"].hex(), "0x1.921fb54442d13p+2"
        )
        self.assertEqual(
            protocol["observed_epoch"].hex(), "0x1.921fb54442d13p+2"
        )
        self.assertTrue(protocol["target_equals_observed_bit_exact"])
        self.assertEqual(protocol["accounting"]["steps_done"], 429)
        self.assertEqual(protocol["accounting"]["iterations_max_exceeded"], 0)
        self.assertEqual(
            protocol["requested"]["initial_dt"].hex(),
            "0x1.921fb54442d18p-4",
        )
        self.assertGreater(
            central["metrics_against_actual_clock_ias15"]["mercurius"]
            ["relative_total_energy_final"],
            4.0,
        )
        self.assertLess(
            central["metrics_against_actual_clock_ias15"]["trace"]
            ["relative_total_energy_final"],
            4.0e-6,
        )


@unittest.skipUnless(
    os.environ.get("JX_RUN_REBOUND_HYBRID_FULL") == "1",
    "set JX_RUN_REBOUND_HYBRID_FULL=1 for the calibrated exact-5.1.1 profile",
)
class FullPinnedProfileTests(unittest.TestCase):
    def test_named_full_profile_and_exact_external_kats(self):
        report = benchmark.run_comparison(benchmark.Profile.full(), "required")
        self.assertTrue(report["regression_gates"]["passed"])
        self.assertEqual(
            tuple(report["profile"]["jx_divisors"]), (256, 512, 1024)
        )
        self.assertEqual(report["interpretation"]["headline_retained_node_count"], 103)
        for method in ("mercurius", "trace"):
            for divisor in benchmark.EXTERNAL_DIVISORS:
                lane = report["headline_lanes"][f"rebound_{method}_p{divisor}"]
                self.assertEqual(
                    lane["accounting"]["every_node_final_state_raw_sha256"],
                    benchmark.EXTERNAL_FINAL_STATE_SHA256[method][divisor],
                )
                self.assertEqual(
                    lane["accounting"]["every_node_trajectory_raw_sha256"],
                    benchmark.EXTERNAL_TRAJECTORY_SHA256[method][divisor],
                )
        for divisor in benchmark.JX_DIVISORS:
            lane = report["headline_lanes"][f"jx_hybrid_p{divisor}"]
            self.assertEqual(
                lane["headline_final_state_raw_sha256"],
                benchmark.JX_FINAL_STATE_SHA256[divisor],
            )
            self.assertEqual(
                lane["headline_trajectory_raw_sha256"],
                benchmark.JX_TRAJECTORY_SHA256[divisor],
            )


if __name__ == "__main__":
    unittest.main()
