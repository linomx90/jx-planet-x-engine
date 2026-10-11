from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("jx_tno_1kyr_gate_v3_test_subject", SOURCE)
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
        self.assertEqual(
            self.protocol["acceptance"]["prelane_runtime_preflight"],
            "EXECUTION_WORKER_LAUNCHES_TWO_DISTINCT_PARENT_CHALLENGED_FRESH_"
            "CHILDREN_WITH_COMPLETE_RUNTIME_RECEIPTS_EXPLICIT_SYNTHETIC_ROLE_"
            "AND_IDENTICAL_PROCESS_INDEPENDENT_PROJECTION_BEFORE_LANE_ZERO",
        )
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

    def test_protocol_preserves_failed_v1_without_claiming_an_outcome(self) -> None:
        failed = self.protocol["predecessor_failure"]
        self.assertEqual(failed["commit"], self.m.V1_COMMIT)
        self.assertEqual(failed["tag_object"], self.m.V1_TAG_OBJECT)
        self.assertEqual(failed["retained_root_roster"], ["PROTOCOL.json"])
        self.assertEqual(failed["completed_lifecycle_segment_count"], 0)
        self.assertIs(failed["tracer_dynamics_advanced"], False)
        self.assertIs(failed["batch_0_massive_source_dynamics_executed"], True)
        self.assertIs(failed["lifecycle_initialized"], True)
        self.assertIs(failed["result_published"], False)
        self.assertIs(failed["scientific_outcomes_generated"], False)
        self.assertEqual(
            failed["failure_observation_custody"],
            "CODEX_SESSION_LOG_RECOVERED_NOT_RETAINED_BY_V1_ROOT",
        )
        self.assertEqual(failed["exit_code"], 1)

    def test_protocol_preserves_failed_v2_without_upgrading_session_evidence(self) -> None:
        failed = self.protocol["immediate_predecessor_failure"]
        self.assertEqual(failed["commit"], self.m.V2_COMMIT)
        self.assertEqual(failed["tag_object"], self.m.V2_TAG_OBJECT)
        self.assertEqual(failed["retained_root_roster"], ["PROTOCOL.json"])
        self.assertEqual(failed["main_batch_count_completed_in_memory"], 12)
        self.assertEqual(failed["main_lifecycle_segment_count_completed_in_memory"], 240)
        self.assertEqual(failed["main_prefix_checkpoint_count_completed_in_memory"], 252)
        self.assertEqual(failed["retained_h64_prefix_crossbind_count_completed_in_memory"], 12)
        self.assertIs(failed["fresh_restart_started"], False)
        self.assertIs(failed["runtime_receipt_constructed"], False)
        self.assertIs(failed["result_published"], False)
        self.assertIs(failed["qualification_result_retained"], False)
        self.assertEqual(
            failed["failure_observation_custody"],
            "CODEX_SESSION_LOG_AND_CODE_ORDER_NOT_RETAINED_BY_V2_ROOT",
        )

    def test_binary64_array_rows_convert_exactly_to_builtin_tuples(self) -> None:
        import numpy as np

        smallest = math.nextafter(0.0, 1.0)
        rows = np.asarray(
            [
                (0.0, -0.0, smallest),
                (1.0, -1.0, 2.0),
                (3.0, 4.0, 5.0),
                (6.0, 7.0, 8.0),
                (9.0, 10.0, float.fromhex("0x1.fffffffffffffp+1023")),
            ],
            dtype="<f8",
        )
        converted = self.m._builtin_finite_float3_rows(np, rows, 5, "fixture")
        self.assertIs(type(converted), tuple)
        self.assertTrue(all(type(row) is tuple for row in converted))
        self.assertTrue(all(type(item) is float for row in converted for item in row))
        self.assertEqual(
            struct.pack("<15d", *(item for row in converted for item in row)),
            rows.tobytes(order="C"),
        )
        self.assertEqual(math.copysign(1.0, converted[0][1]), -1.0)

    def test_binary64_array_rows_reject_implicit_conversion_and_bad_geometry(self) -> None:
        import numpy as np

        class ArraySubclass(np.ndarray):
            pass

        valid = np.zeros((5, 3), dtype="<f8")
        hostile_values = (
            valid.view(ArraySubclass),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((5, 3), dtype=np.int64),
            np.zeros((5, 3), dtype=np.bool_),
            np.zeros((5, 3), dtype=object),
            np.zeros((5, 3), dtype=np.complex128),
            np.zeros((5, 3), dtype=">f8"),
            np.zeros((15,), dtype="<f8"),
            np.zeros((4, 3), dtype="<f8"),
            tuple(tuple(0.0 for _ in range(3)) for _ in range(5)),
        )
        for hostile in hostile_values:
            with self.subTest(type=type(hostile), dtype=getattr(hostile, "dtype", None)):
                with self.assertRaises(self.m.LifecycleGateError):
                    self.m._builtin_finite_float3_rows(np, hostile, 5, "fixture")
        for nonfinite in (math.nan, math.inf, -math.inf):
            hostile = valid.copy()
            hostile[2, 1] = nonfinite
            with self.subTest(nonfinite=nonfinite), self.assertRaises(self.m.LifecycleGateError):
                self.m._builtin_finite_float3_rows(np, hostile, 5, "fixture")
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._builtin_finite_float3_rows(np, valid, False, "fixture")

    def test_live_checkpoint_wrapper_passes_only_exact_builtin_rows(self) -> None:
        import numpy as np

        captured: dict[str, object] = {}
        grid = types.SimpleNamespace()

        def checkpoint(kernel, **kwargs):
            captured["kernel"] = kernel
            captured.update(kwargs)
            return b"checkpoint"

        grid.checkpoint_from_primary_relative_v1 = checkpoint
        source = np.zeros((5, 3), dtype="<f8")
        tracers = np.arange(27, dtype="<f8").reshape(9, 3)
        result = self.m._checkpoint_from_host_arrays(
            np,
            grid,
            object(),
            batch_order=3,
            source_positions=source,
            source_velocities=source.copy(),
            tracer_positions=tracers,
            tracer_velocities=tracers.copy(),
        )
        self.assertEqual(result, b"checkpoint")
        self.assertEqual(captured["batch_order"], 3)
        for key in (
            "source_relative_positions_AU",
            "source_relative_velocities_AU_per_julian_year",
            "tracer_relative_positions_AU",
            "tracer_relative_velocities_AU_per_julian_year",
        ):
            rows = captured[key]
            self.assertIs(type(rows), tuple)
            self.assertTrue(all(type(row) is tuple for row in rows))
            self.assertTrue(all(type(item) is float for row in rows for item in row))

    def test_frozen_v1_grid_reproduces_numpy_roster_failure(self) -> None:
        import numpy as np

        name = "_jx_tno_1k_v2_grid_root_cause"
        grid = self.m._load_source_module(
            self.m.GRID_PATH, self.m.GRID_BYTES, self.m.GRID_SHA256, name
        )
        try:
            kernel = object.__new__(grid.FrozenClassifierKernelV1)
            kernel._closed = False
            source = np.zeros((5, 3), dtype="<f8")
            tracers = np.zeros((9, 3), dtype="<f8")
            with self.assertRaisesRegex(grid.GridAdapterError, "source positions roster differs"):
                grid.checkpoint_from_primary_relative_v1(
                    kernel,
                    batch_order=0,
                    source_relative_positions_AU=source,
                    source_relative_velocities_AU_per_julian_year=source.copy(),
                    tracer_relative_positions_AU=tracers,
                    tracer_relative_velocities_AU_per_julian_year=tracers.copy(),
                )
        finally:
            sys.modules.pop(name, None)

    def test_post_segment_checkpoint_copies_gpu_rows_before_grid_boundary(self) -> None:
        import numpy as np

        captured: dict[str, object] = {}
        grid = types.SimpleNamespace()
        grid.checkpoint_from_primary_relative_v1 = lambda kernel, **kwargs: (
            captured.update(kwargs) or b"post-segment"
        )

        class FakeCP:
            def __init__(self) -> None:
                self.calls = 0

            def asnumpy(self, value):
                self.calls += 1
                return value.copy()

        cp = FakeCP()
        source = np.zeros((2, 5, 3), dtype="<f8")
        tracer = np.zeros((9, 3), dtype="<f8")
        full = types.SimpleNamespace(
            source_relative_positions_AU=source,
            source_relative_velocities_AU_per_year=source.copy(),
        )
        dynamics = types.SimpleNamespace(
            tracer_relative_positions_AU=tracer,
            tracer_relative_velocities_AU_per_year=tracer.copy(),
        )
        state = types.SimpleNamespace(dynamics_state=dynamics)
        self.assertEqual(
            self.m._checkpoint_payload(
                grid, object(), cp, batch_order=0, full=full, state=state, source_node=1
            ),
            b"post-segment",
        )
        self.assertEqual(cp.calls, 2)
        self.assertTrue(
            all(
                type(row) is tuple and all(type(item) is float for item in row)
                for key, rows in captured.items()
                if key != "batch_order"
                for row in rows
            )
        )

    def test_only_wrapper_and_offline_restart_call_grid_checkpoint_directly(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        parents: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "checkpoint_from_primary_relative_v1"
                ):
                    parents.append(node.name)
        self.assertEqual(
            sorted(parents),
            ["_checkpoint_from_host_arrays", "_crossbind_restart_stream_to_prefix"],
        )

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

    def test_builtin_math_receipt_preserves_exact_no_file_semantics(self) -> None:
        receipt = self.m._builtin_math_receipt()
        self.assertEqual(receipt["module_name"], "math")
        self.assertEqual(receipt["module_kind"], "BUILTIN")
        self.assertIs(receipt["module_file_absent"], True)
        self.assertEqual(receipt["spec_origin"], "built-in")
        self.assertEqual(receipt["spec_loader"], "BuiltinImporter")
        self.assertEqual(receipt["callable_names"], list(self.m.MATH_CALLABLE_NAMES))
        self.assertNotIn("path", receipt)
        self.assertEqual(self.m._strict_json(self.m._canonical_json(receipt)), receipt)
        self.m._validate_builtin_math_receipt(receipt, "fixture")

    def test_builtin_math_receipt_rejects_retained_and_live_spoofs(self) -> None:
        receipt = self.m._builtin_math_receipt()
        hostile = dict(receipt)
        hostile["spec_origin"] = "extension"
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._validate_builtin_math_receipt(hostile, "fixture")
        original_pi = self.m.math.pi
        try:
            self.m.math.pi = 0.5
            with self.assertRaises(self.m.LifecycleGateError):
                self.m._builtin_math_receipt()
            with self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_builtin_math_receipt(receipt, "fixture")
        finally:
            self.m.math.pi = original_pi

    def test_mapped_libm_receipt_is_live_exact_and_tamper_rejected(self) -> None:
        receipt = self.m._libm_receipt()
        self.assertEqual(receipt["path"], str(self.m.PINNED_LIBM_PATH))
        self.assertEqual(receipt["bytes"], self.m.PINNED_LIBM_BYTES)
        self.assertEqual(receipt["sha256"], self.m.PINNED_LIBM_SHA256)
        self.assertIs(receipt["mapped_into_process"], True)
        self.m._validate_libm_receipt(receipt, "fixture")
        hostile = dict(receipt)
        hostile["mapped_inode"] += 1
        with self.assertRaises(self.m.LifecycleGateError):
            self.m._validate_libm_receipt(hostile, "fixture")

    def test_runtime_base_is_constructed_before_any_main_or_restart_lane(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def call_lines(function_name: str, callee: str) -> list[int]:
            return [
                node.lineno
                for node in ast.walk(functions[function_name])
                if isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == callee)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == callee)
                )
            ]

        main_base = call_lines("_execute_worker", "_critical_runtime_base")
        main_specs = call_lines("_execute_worker", "construct_adapter_specs_v1")
        main_lanes = call_lines("_execute_worker", "_run_batch")
        restart_base = call_lines("_restart_worker", "_critical_runtime_base")
        restart_specs = call_lines("_restart_worker", "construct_adapter_specs_v1")
        restart_lane = call_lines("_restart_worker", "_restart_suffix")
        self.assertEqual(len(main_base), 1)
        self.assertEqual(len(main_specs), 1)
        self.assertEqual(len(main_lanes), 1)
        self.assertLess(main_specs[0], main_base[0])
        self.assertLess(main_base[0], main_lanes[0])
        self.assertEqual(len(restart_base), 1)
        self.assertEqual(len(restart_specs), 1)
        self.assertEqual(len(restart_lane), 1)
        self.assertLess(restart_specs[0], restart_base[0])
        self.assertLess(restart_base[0], restart_lane[0])

    def test_execution_worker_runs_two_child_runtime_preflight_before_lane_zero(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_execute_worker"
        )
        preflight_lines = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "runtime_preflight"
        ]
        lane_lines = [
            node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_batch"
        ]
        self.assertEqual(len(preflight_lines), 1)
        self.assertEqual(len(lane_lines), 1)
        self.assertLess(preflight_lines[0], lane_lines[0])
        self.assertEqual(
            self.m._parser().parse_args(["runtime-preflight"]).command,
            "runtime-preflight",
        )
        with self.assertRaisesRegex(
            self.m.LifecycleGateError, "import-probe receipt"
        ):
            self.m.main(["execute-worker"])

    def test_runtime_preflight_rejects_a_duplicated_worker_receipt(self) -> None:
        receipt = {
            "worker_order": 0,
            "parent_challenge_sha256": "1" * 64,
            "process_id": 101,
            "runtime_identity": {
                "source_only_runtime_import_guard": {
                    "pycache_prefix": "/tmp/worker-0-pycache",
                    "cupy_external_kernel_cache": {
                        "directory": "/tmp/worker-0-cupy-cache",
                    },
                },
            },
        }
        document = {
            "schema": self.m.RUNTIME_PREFLIGHT_SCHEMA,
            "worker_receipts": [receipt, receipt],
            "process_independent_projection_sha256": "3" * 64,
            "prelane_binding_projection_sha256": "4" * 64,
            "worker_challenge_sha256s": ["1" * 64, "2" * 64],
            "fresh_child_count": 2,
            "source_step_count": 0,
            "tracer_count": 0,
            "dynamics_executed": False,
            "qualification_result": False,
            "classification_generated": False,
            "science_result": False,
            "content_sha256": "",
        }
        document["content_sha256"] = self.m._domain_hash(
            self.m.RUNTIME_PREFLIGHT_DOMAIN, document
        )
        with mock.patch.object(
            self.m,
            "_validate_runtime_preflight_worker_receipt",
            side_effect=lambda item, _delegate, _label: item,
        ), self.assertRaisesRegex(
            self.m.LifecycleGateError, "worker order differs"
        ):
            self.m._validate_runtime_preflight_document(document, object())

        second = json.loads(self.m._canonical_json(receipt))
        second.update({
            "worker_order": 1,
            "parent_challenge_sha256": "2" * 64,
            "process_id": 102,
        })
        second_guard = second["runtime_identity"][
            "source_only_runtime_import_guard"
        ]
        second_guard["pycache_prefix"] = "/tmp/worker-1-pycache"
        shared_cupy = json.loads(self.m._canonical_json(document))
        shared_cupy["worker_receipts"] = [receipt, second]
        shared_cupy["content_sha256"] = self.m._domain_hash(
            self.m.RUNTIME_PREFLIGHT_DOMAIN, shared_cupy
        )
        with mock.patch.object(
            self.m,
            "_validate_runtime_preflight_worker_receipt",
            side_effect=lambda item, _delegate, _label: item,
        ), self.assertRaisesRegex(
            self.m.LifecycleGateError, "guard paths are not distinct"
        ):
            self.m._validate_runtime_preflight_document(shared_cupy, object())

        second_guard["pycache_prefix"] = "/tmp/worker-0-pycache"
        second_guard["cupy_external_kernel_cache"][
            "directory"
        ] = "/tmp/worker-1-cupy-cache"
        shared_pycache = json.loads(self.m._canonical_json(document))
        shared_pycache["worker_receipts"] = [receipt, second]
        shared_pycache["content_sha256"] = self.m._domain_hash(
            self.m.RUNTIME_PREFLIGHT_DOMAIN, shared_pycache
        )
        with mock.patch.object(
            self.m,
            "_validate_runtime_preflight_worker_receipt",
            side_effect=lambda item, _delegate, _label: item,
        ), self.assertRaisesRegex(
            self.m.LifecycleGateError, "guard paths are not distinct"
        ):
            self.m._validate_runtime_preflight_document(shared_pycache, object())

    def test_frozen_route_policy_literal_matches_bound_constructor(self) -> None:
        name = "_jx_tno_1k_v3_route_policy"
        grid = self.m._load_source_module(
            self.m.GRID_PATH, self.m.GRID_BYTES, self.m.GRID_SHA256, name
        )
        try:
            specs = grid.construct_adapter_specs_v1(
                map_step_count=self.m.SEGMENT_STEPS,
                source_step_count=self.m.HORIZON_YEARS * self.m.STEPS_PER_YEAR,
            )
            from jxplanetx.restricted_wh_route import (
                restricted_wh_route_policy_identity,
            )

            observed = self.m._plain(
                restricted_wh_route_policy_identity(specs["route"])
            )
            self.assertEqual(observed, self.m._expected_route_policy_identity())
        finally:
            sys.modules.pop(name, None)

    def test_production_engine_projection_rejects_semantic_relabels(self) -> None:
        digest_lifecycle = "1" * 64
        digest_hybrid = "2" * 64
        digest_quiet = "3" * 64
        cupy = {
            "cupy_version": "14.2.0",
            "cuda_runtime_version": 13020,
            "cuda_local_runtime_version": 13040,
            "cuda_driver_version": 13020,
            "device_index": 0,
            "device_name": "device",
            "device_uuid": "uuid",
            "compute_capability": "12.0",
            "pci_domain_bus_device": "0000:01:00",
        }
        quiet = {
            "cupy_version": cupy["cupy_version"],
            "cuda_runtime_version": cupy["cuda_runtime_version"],
            "cuda_local_runtime_version": cupy["cuda_local_runtime_version"],
            "cuda_driver_version": cupy["cuda_driver_version"],
            "device_index": cupy["device_index"],
            "device_name": cupy["device_name"],
            "device_uuid": cupy["device_uuid"],
            "device_compute_capability": cupy["compute_capability"],
            "device_pci_domain_bus_device": cupy["pci_domain_bus_device"],
            "implementation_source_sha256": digest_quiet,
        }
        hybrid = {
            "method_id": "JX_CUDA_RESTRICTED_WH_GUARDED_DOP853_V2",
            "implementation_source_sha256": digest_hybrid,
            "quiet_cuda_runtime_identity": quiet,
            "route_runtime_identities": [],
            "route_policy": self.m._expected_route_policy_identity(),
            "route_worker_count": 1,
            "route_worker_start_method": "SERIAL",
            "determinism_scope": (
                "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
            ),
        }
        engine = {
            "method_id": "JX_CUDA_RESTRICTED_WH_FIXED_LANE_LIFECYCLE_V1",
            "implementation_source_sha256": digest_lifecycle,
            "route_worker_count": 1,
            "determinism_scope": (
                "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
            ),
            "hybrid_runtime_identity_without_source_tile": hybrid,
        }
        base = {key: None for key in self.m._runtime_base_keys()}
        base["cupy_cuda"] = cupy
        base["critical_jx_modules"] = [
            {
                "module": "jxplanetx.restricted_wh_lifecycle",
                "sha256": digest_lifecycle,
            },
            {
                "module": "jxplanetx.cuda_restricted_wh_hybrid",
                "sha256": digest_hybrid,
            },
            {
                "module": "jxplanetx.cuda_restricted_wh",
                "sha256": digest_quiet,
            },
        ]
        runtime = {
            "schema": self.m.RUNTIME_SCHEMA,
            **base,
            "source_only_runtime_import_guard": {},
            "engine_runtime_projection": engine,
            "content_sha256": "",
        }
        runtime["content_sha256"] = self.m._domain_hash(
            self.m.RUNTIME_DOMAIN, runtime
        )

        class FakeDelegate:
            @staticmethod
            def _validate_runtime_import_guard(_value, _label):
                return None

            @staticmethod
            def _validate_fused_tracer_runtime_identity(_value):
                return None

        with mock.patch.object(
            self.m, "_validate_runtime_base", return_value=base
        ):
            self.m._validate_runtime_identity(
                runtime,
                FakeDelegate,
                "fixture",
                expected_mode=self.m.RUNTIME_MODE_PRODUCTION,
            )
            mutations = (
                ("method_id", "FORGED"),
                ("implementation_source_sha256", "4" * 64),
                ("route_worker_count", 999),
                ("determinism_scope", "FORGED"),
            )
            for key, replacement in mutations:
                hostile = json.loads(self.m._canonical_json(runtime))
                hostile["engine_runtime_projection"][key] = replacement
                hostile["content_sha256"] = self.m._domain_hash(
                    self.m.RUNTIME_DOMAIN, hostile
                )
                with self.subTest(key=key), self.assertRaises(
                    self.m.LifecycleGateError
                ):
                    self.m._validate_runtime_identity(
                        hostile,
                        FakeDelegate,
                        "fixture",
                        expected_mode=self.m.RUNTIME_MODE_PRODUCTION,
                    )
            for key, replacement in (
                ("method_id", "FORGED"),
                ("route_worker_count", 999),
                ("route_worker_start_method", "spawn"),
                ("determinism_scope", "FORGED"),
                ("route_policy", {"forged": True}),
            ):
                hostile = json.loads(self.m._canonical_json(runtime))
                hostile["engine_runtime_projection"][
                    "hybrid_runtime_identity_without_source_tile"
                ][key] = replacement
                hostile["content_sha256"] = self.m._domain_hash(
                    self.m.RUNTIME_DOMAIN, hostile
                )
                with self.subTest(hybrid_key=key), self.assertRaises(
                    self.m.LifecycleGateError
                ):
                    self.m._validate_runtime_identity(
                        hostile,
                        FakeDelegate,
                        "fixture",
                        expected_mode=self.m.RUNTIME_MODE_PRODUCTION,
                    )

            synthetic = json.loads(self.m._canonical_json(runtime))
            synthetic_engine = synthetic["engine_runtime_projection"]
            synthetic_hybrid = synthetic_engine[
                "hybrid_runtime_identity_without_source_tile"
            ]
            synthetic_engine.update({
                "method_id": "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3",
                "implementation_source_sha256": digest_quiet,
                "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
            })
            synthetic_hybrid.update({
                "method_id": "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3",
                "implementation_source_sha256": digest_quiet,
                "route_policy": "NOT_INVOKED_RUNTIME_PREFLIGHT_ONLY",
                "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
            })
            synthetic["content_sha256"] = self.m._domain_hash(
                self.m.RUNTIME_DOMAIN, synthetic
            )
            self.m._validate_runtime_identity(
                synthetic,
                FakeDelegate,
                "synthetic fixture",
                expected_mode=self.m.RUNTIME_MODE_PREFLIGHT,
            )
            forged_synthetic_method = json.loads(
                self.m._canonical_json(synthetic)
            )
            forged_synthetic_method["engine_runtime_projection"][
                "method_id"
            ] = "FORGED_SYNTHETIC_TOP_LEVEL_METHOD"
            forged_synthetic_method["content_sha256"] = self.m._domain_hash(
                self.m.RUNTIME_DOMAIN, forged_synthetic_method
            )
            with self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_runtime_identity(
                    forged_synthetic_method,
                    FakeDelegate,
                    "forged synthetic top-level method",
                    expected_mode=self.m.RUNTIME_MODE_PREFLIGHT,
                )
            with self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_runtime_identity(
                    synthetic,
                    FakeDelegate,
                    "synthetic as retained main",
                    expected_mode=self.m.RUNTIME_MODE_PRODUCTION,
                )
            with self.assertRaises(self.m.LifecycleGateError):
                self.m._validate_runtime_identity(
                    runtime,
                    FakeDelegate,
                    "production as preflight",
                    expected_mode=self.m.RUNTIME_MODE_PREFLIGHT,
                )

    def test_runtime_base_reaches_builtin_math_and_libm_without_dynamics(self) -> None:
        class FakeDelegate:
            @staticmethod
            def _validate_fresh_child_interpreter_receipt(_value):
                return None

            @staticmethod
            def _validate_numpy_runtime_identity(_value):
                return None

            @staticmethod
            def _numpy_runtime_identity():
                return {"fixture": True}

        class FakeDevice:
            id = 0

        class FakeRuntime:
            @staticmethod
            def getDeviceProperties(_device):
                return {
                    "name": b"fixture-device", "uuid": bytes(range(16)),
                    "major": 12, "minor": 0, "pciDomainID": 0,
                    "pciBusID": 1, "pciDeviceID": 2,
                }

            @staticmethod
            def runtimeGetVersion():
                return 13020

            @staticmethod
            def driverGetVersion():
                return 13020

        fake_cp = types.SimpleNamespace(
            __version__="14.2.0",
            __file__=str(SOURCE),
            cuda=types.SimpleNamespace(
                runtime=FakeRuntime,
                Device=FakeDevice,
                get_local_runtime_version=lambda: 13040,
            ),
        )
        names = (
            "jxplanetx.restricted_wh_source", "jxplanetx.cuda_restricted_wh",
            "jxplanetx.cuda_restricted_wh_hybrid", "jxplanetx.restricted_wh_route",
            "jxplanetx.restricted_wh_lifecycle", "jxplanetx.cuda_tracer_events",
            "jxplanetx.fast_wisdom_holman", "jxplanetx._all_conic_kepler",
            "jxplanetx._wisdom_holman_cpu_v3", "jxplanetx.engine.backends",
            "jxplanetx.engine.contracts",
        )
        saved = {name: sys.modules.get(name) for name in names}
        try:
            for name in names:
                module = types.ModuleType(name)
                module.__file__ = str(SOURCE)
                sys.modules[name] = module
            if sys.flags.optimize == 0:
                base = self.m._critical_runtime_base(
                    FakeDelegate,
                    fake_cp,
                    fresh_child={},
                    fresh_registry={
                        "preloaded_module_names": [],
                        "fresh_registry_observed": True,
                        "claim": (
                            "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_"
                            "NOT_PROCESS_AUTHENTICATION"
                        ),
                    },
                )
                self.assertEqual(base["builtin_math"], self.m._builtin_math_receipt())
                self.assertEqual(
                    base["mapped_libm_point_observation"], self.m._libm_receipt()
                )
                self.assertEqual(set(base), set(self.m._runtime_base_keys()))
            else:
                with self.assertRaises(self.m.LifecycleGateError):
                    self.m._critical_runtime_base(
                        FakeDelegate,
                        fake_cp,
                        fresh_child={},
                        fresh_registry={
                            "preloaded_module_names": [],
                            "fresh_registry_observed": True,
                            "claim": (
                                "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_"
                                "NOT_PROCESS_AUTHENTICATION"
                            ),
                        },
                    )
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

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
