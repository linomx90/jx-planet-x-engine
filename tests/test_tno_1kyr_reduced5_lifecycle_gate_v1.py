from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("jx_tno_1kyr_gate_test_subject", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load lifecycle-gate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LifecycleGateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.protocol = cls.m.build_protocol()

    def test_protocol_freezes_only_bounded_one_thousand_year_gate(self) -> None:
        execution = self.protocol["execution"]
        self.assertEqual(execution["ephemeris"], "DE441")
        self.assertEqual(execution["batch_count"], 12)
        self.assertEqual(execution["states_per_batch"], 9)
        self.assertEqual(execution["horizon_julian_years"], 1_000)
        self.assertEqual(execution["steps_per_julian_year"], 64)
        self.assertEqual(execution["segment_steps"], 3_200)
        self.assertEqual(execution["segments_per_batch"], 20)
        self.assertEqual(execution["checkpoint_years"], list(range(0, 1_001, 50)))
        self.assertIs(self.protocol["ten_myr_execution_authorized"], False)
        self.assertIs(self.protocol["outcomes_generated"], False)
        self.assertTrue(all(value is False for value in self.protocol["claims"].values()))

    def test_protocol_and_output_geometry_are_exact(self) -> None:
        outputs = self.protocol["outputs"]
        self.assertEqual(outputs["prefix_states_bytes"], 169_344)
        self.assertEqual(outputs["restart_streams_bytes"], 9_504)
        self.assertEqual(outputs["segment_record_count"], 240)
        self.assertEqual(outputs["routes_jsonl_bytes_on_pass"], 0)
        self.assertEqual(self.m.PREFIX_BYTES, 12 * 21 * 672)
        self.assertEqual(self.m.RESTART_STREAM_BYTES, 2 * 11 * 432)

    def test_protocol_canonical_roundtrip_and_tamper_rejection(self) -> None:
        payload = self.m._canonical_json(self.protocol)
        self.assertEqual(self.m._strict_json(payload), self.protocol)
        self.assertEqual(self.m.validate_protocol(self.protocol), self.protocol["content_sha256"])
        altered = json.loads(payload)
        altered["execution"]["horizon_julian_years"] = 10_000_000
        with self.assertRaises(self.m.LifecycleGateError):
            self.m.validate_protocol(altered)

    def test_prepare_and_validate_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            value = self.m.prepare(root)
            self.assertEqual({item.name for item in root.iterdir()}, {"PROTOCOL.json"})
            self.assertEqual(self.m.validate_prepared(root), value)
            self.assertEqual((root / "PROTOCOL.json").read_bytes(), self.m._canonical_json(value))
            with self.assertRaises(self.m.LifecycleGateError):
                self.m.prepare(root)

    def test_source_locator_exact_mapping_and_bool_rejection(self) -> None:
        first = self.m.source_locator_v1(0)
        last = self.m.source_locator_v1(11)
        self.assertEqual(first["source_order"], 12)
        self.assertEqual(first["byte_offset"], 40 + 12 * 15_360_240)
        self.assertEqual(last["source_order"], 23)
        self.assertEqual(last["byte_offset"], 40 + 23 * 15_360_240)
        self.assertEqual(first["byte_length"], 15_360_240)
        with self.assertRaises(self.m.LifecycleGateError):
            self.m.source_locator_v1(False)
        with self.assertRaises(self.m.LifecycleGateError):
            self.m.source_locator_v1(12)

    def _segment(self) -> dict:
        digest = "1" * 64
        record = {
            "schema": self.m.SEGMENT_SCHEMA,
            "segment_order": 0,
            "batch_order": 0,
            "object_id": self.m.OBJECT_IDS[0],
            "segment_index": 0,
            "start_year": 0,
            "end_year": 50,
            "start_global_source_step": 0,
            "end_global_source_step": 3_200,
            "source_tile_sha256": digest,
            "generated_source_segment_sha256": digest,
            "retained_source_segment_sha256": digest,
            "checkpoint_sha256": digest,
            "active_tracer_count": 9,
            "route_record_count": 0,
            "removal_record_count": 0,
            "minimum_source_distance_AU_hex": (1.01).hex(),
            "maximum_kepler_iterations": 7,
            "engine_runtime_identity_sha256": digest,
            "elapsed_seconds_hex": (0.25).hex(),
            "content_sha256": "",
        }
        record["content_sha256"] = self.m._domain_hash(self.m.SEGMENT_DOMAIN, record)
        return record

    def test_segment_receipt_accepts_only_zero_event_exact_identity(self) -> None:
        record = self._segment()
        self.assertIs(self.m._validate_segment_record(record, expected_order=0), record)
        for key, replacement in (
            ("route_record_count", 1),
            ("removal_record_count", 1),
            ("minimum_source_distance_AU_hex", (1.0).hex()),
            ("object_id", "wrong"),
        ):
            hostile = dict(record)
            hostile[key] = replacement
            hostile["content_sha256"] = self.m._domain_hash(self.m.SEGMENT_DOMAIN, hostile)
            with self.subTest(key=key), self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_segment_record(hostile, expected_order=0)

    def test_segment_rejects_contradictory_source_digests(self) -> None:
        hostile = self._segment()
        hostile["generated_source_segment_sha256"] = "2" * 64
        hostile["retained_source_segment_sha256"] = "3" * 64
        hostile["content_sha256"] = self.m._domain_hash(self.m.SEGMENT_DOMAIN, hostile)
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._validate_segment_record(hostile, expected_order=0)

    def test_segment_integer_fields_reject_json_booleans(self) -> None:
        for key in (
            "segment_order", "batch_order", "segment_index", "start_year",
            "start_global_source_step", "route_record_count", "removal_record_count",
        ):
            hostile = self._segment()
            hostile[key] = False
            hostile["content_sha256"] = self.m._domain_hash(self.m.SEGMENT_DOMAIN, hostile)
            with self.subTest(key=key), self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_segment_record(hostile, expected_order=0)

    def test_restart_request_roundtrip_and_tamper_rejection(self) -> None:
        payload = struct.pack("<54d", *(float(index) for index in range(54)))
        request = self.m._restart_request(payload)
        parsed, restored = self.m._validate_restart_request(request)
        self.assertEqual(parsed, request)
        self.assertEqual(restored, payload)
        hostile = dict(request)
        hostile["midpoint_sha256"] = "2" * 64
        hostile["content_sha256"] = self.m._domain_hash(self.m.RESTART_REQUEST_DOMAIN, hostile)
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._validate_restart_request(hostile)
        nonfinite = struct.pack("<54d", float("nan"), *(0.0 for _ in range(53)))
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._restart_request(nonfinite)

    def test_internal_state_decoder_is_exact_and_finite(self) -> None:
        payload = struct.pack("<54d", *(float(index) for index in range(54)))
        positions, velocities = self.m._unpack_internal_tracer_state(payload)
        self.assertEqual(len(positions), 9)
        self.assertEqual(len(velocities), 9)
        self.assertEqual(positions[0], (0.0, 1.0, 2.0))
        self.assertEqual(velocities[-1], (51.0, 52.0, 53.0))
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._unpack_internal_tracer_state(payload[:-1])
        nonfinite = struct.pack("<54d", float("nan"), *(0.0 for _ in range(53)))
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._unpack_internal_tracer_state(nonfinite)

    def test_runtime_comparison_ignores_only_child_cache_guard(self) -> None:
        base = {
            "schema": self.m.RUNTIME_SCHEMA,
            "python": {"version": "x"},
            "source_only_runtime_import_guard": {"cache_prefix": "/a"},
            "numpy_runtime_identity": {
                "module": {"cached_path": "/a/numpy.pyc", "sha256": "1" * 64},
                "core_module": {"cached_path": "/a/core.pyc", "sha256": "2" * 64},
            },
            "content_sha256": "1" * 64,
        }
        other = dict(base)
        other["numpy_runtime_identity"] = {
            "module": {"cached_path": "/b/numpy.pyc", "sha256": "1" * 64},
            "core_module": {"cached_path": "/b/core.pyc", "sha256": "2" * 64},
        }
        other["source_only_runtime_import_guard"] = {"cache_prefix": "/b"}
        other["content_sha256"] = "2" * 64
        self.assertEqual(
            self.m._runtime_comparison_projection(base),
            self.m._runtime_comparison_projection(other),
        )
        other["python"] = {"version": "different"}
        self.assertNotEqual(
            self.m._runtime_comparison_projection(base),
            self.m._runtime_comparison_projection(other),
        )

    def test_exact_claims_reject_integer_aliases(self) -> None:
        expected = {"claim": False}
        self.m._exact_claims({"claim": False}, expected, "fixture")
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._exact_claims({"claim": 0}, expected, "fixture")

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._strict_json(b'{"a":1,"a":2}\n')
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._strict_json(b'{"a":NaN}\n')

    def test_execute_requires_external_remote_confirmation(self) -> None:
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._git_custody(remote_verified=False)

    def test_public_defaults_and_cli_target_successor_root(self) -> None:
        self.assertEqual(self.m.prepare.__defaults__, (self.m.DEFAULT_OUTPUT_ROOT,))
        self.assertEqual(self.m.validate_prepared.__defaults__, (self.m.DEFAULT_OUTPUT_ROOT,))
        self.assertEqual(self.m.validate_output.__defaults__, (self.m.DEFAULT_OUTPUT_ROOT,))
        self.assertEqual(self.m.execute.__defaults__, (self.m.DEFAULT_OUTPUT_ROOT,))
        args = self.m._parser().parse_args(["build"])
        self.assertEqual(args.output_root, self.m.DEFAULT_OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
