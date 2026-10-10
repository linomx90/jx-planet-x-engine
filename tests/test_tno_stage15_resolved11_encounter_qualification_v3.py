from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v3.py"
MODULE_NAME = "_jx_tno_stage15_resolved11_encounter_qualification_v3_test_bound"


def load_source() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(payload, str(SOURCE), "exec", optimize=sys.flags.optimize),
            module.__dict__,
        )
    finally:
        sys.modules.pop(MODULE_NAME, None)
    return module


M = load_source()


class _FakeVector:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def astype(self, _dtype: str, copy: bool = False) -> "_FakeVector":
        del copy
        return self

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise AssertionError(order)
        return struct.pack(f"<{len(self.values)}d", *self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeVector) and self.values == other.values


class _FakePath:
    def __init__(self, endpoint: _FakeVector) -> None:
        self.endpoint = endpoint

    def __getitem__(self, index: int) -> _FakeVector:
        if index != -1:
            raise AssertionError(index)
        return self.endpoint


class _FakeCP:
    @staticmethod
    def array_equal(left: object, right: object) -> bool:
        return bool(left == right)

    @staticmethod
    def asnumpy(value: object) -> object:
        return value


@dataclass(frozen=True)
class _FakeRoute:
    tracer_index: int
    source_index: int
    global_step: int


@dataclass(frozen=True)
class _FakeRemoval:
    tracer_index: int
    source_index: int
    global_step: int


class _FakeEngine:
    TracerEncounterRouteRecord = _FakeRoute
    TracerRemovalEventRecord = _FakeRemoval


class Stage15V3Tests(unittest.TestCase):
    def test_00_import_is_source_only_and_protocol_reconstructs(self) -> None:
        forbidden = tuple(M.RUNTIME_PREFIXES)
        before = {
            name for name in sys.modules
            if isinstance(name, str)
            and any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
        }
        module = load_source()
        after = {
            name for name in sys.modules
            if isinstance(name, str)
            and any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
        }
        self.assertEqual(after, before)
        protocol = module.build_protocol()
        self.assertEqual(module.validate_protocol(protocol), protocol["content_sha256"])
        self.assertFalse(protocol["claims"]["dynamics_executed"])
        self.assertFalse(protocol["claims"]["qualification_established"])
        self.assertFalse(protocol["claims"]["result_artifacts_published"])

    def test_lineage_commit_tag_branch_and_v1_bytes_are_exact(self) -> None:
        predecessor = M._validate_v1_predecessor()
        self.assertEqual(predecessor["git_custody"]["commit"], M.V1_GIT_COMMIT)
        self.assertEqual(predecessor["git_custody"]["branch"], M.V1_GIT_BRANCH)
        self.assertEqual(predecessor["git_custody"]["tag"], M.V1_GIT_TAG)
        self.assertEqual(predecessor["source"]["file_sha256"], M.V1_SOURCE_SHA256)
        self.assertEqual(predecessor["test"]["file_sha256"], M.V1_TEST_SHA256)
        self.assertEqual(predecessor["protocol"]["file_sha256"], M.V1_PROTOCOL_SHA256)
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            M.V1_PROTOCOL_CONTENT_SHA256,
        )

    def test_v2_is_byte_preserved_and_both_zero_lane_setup_failures_are_explicit(self) -> None:
        predecessor = M._validate_v2_predecessor()
        self.assertEqual(predecessor["source"]["file_sha256"], M.V2_SOURCE_SHA256)
        self.assertEqual(predecessor["test"]["file_sha256"], M.V2_TEST_SHA256)
        self.assertEqual(predecessor["protocol"]["file_sha256"], M.V2_PROTOCOL_SHA256)
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            M.V2_PROTOCOL_CONTENT_SHA256,
        )
        self.assertEqual(predecessor["git_custody"]["commit"], M.V2_GIT_COMMIT)
        self.assertEqual(predecessor["git_custody"]["branch"], M.V2_GIT_BRANCH)
        self.assertEqual(predecessor["git_custody"]["annotated_tag"], M.V2_GIT_TAG)
        self.assertEqual(predecessor["git_custody"]["tag_object"], M.V2_GIT_TAG_OBJECT)
        self.assertEqual(predecessor["failed_root"]["exact_roster"], ["PROTOCOL.json"])
        self.assertEqual(predecessor["failed_root"]["attempt_count"], 2)
        self.assertEqual(predecessor["failed_root"]["completed_lane_count"], 0)
        self.assertFalse(predecessor["failed_root"]["result_artifacts_published"])
        failure = predecessor["setup_failure"]
        self.assertEqual(failure["failure_id"], M.V2_FAILURE_ID)
        self.assertEqual(failure["failure_class"], "ModuleNotFoundError")
        self.assertEqual(failure["failure_message"], "No module named 'cupy.cuda.nvrtc'")
        self.assertEqual(failure["invalid_module_name"], "cupy.cuda.nvrtc")
        self.assertEqual(failure["supported_pinned_module_name"], M.NVRTC_MODULE_NAME)
        self.assertFalse(failure["lane_dynamics_started"])
        self.assertFalse(failure["dynamics_outcome_generated"])
        self.assertFalse(failure["result_artifacts_published"])

    def test_protocol_freezes_exact_experiment_and_claim_ceiling(self) -> None:
        protocol = M.build_protocol()
        self.assertEqual(protocol["scope"]["native_epoch_batch_count"], 12)
        self.assertEqual(protocol["scope"]["tracers_per_batch"], 9)
        self.assertEqual(protocol["scope"]["lane_count"], 24)
        self.assertEqual(
            [row["steps_per_julian_year"] for row in protocol["scope"]["lanes"]],
            [64, 128],
        )
        self.assertEqual(protocol["physics"]["time_scale_token"], "TDB")
        self.assertEqual(protocol["adapter"]["lane_engine"], M.LANE_ENGINE)
        self.assertEqual(protocol["adapter"]["lane_model"], M.LANE_MODEL)
        self.assertEqual(protocol["evidence"]["fixed_exact_bytes"], M.FIXED_ARTIFACTS)
        self.assertEqual(protocol["evidence"]["exact_roster"], list(M.ARTIFACT_ROSTER))
        self.assertEqual(
            protocol["runtime"]["explicit_import_modules_in_order"],
            list(M.RUNTIME_IMPORT_MODULE_NAMES),
        )
        self.assertEqual(protocol["runtime"]["nvrtc_module"], M.NVRTC_MODULE_NAME)
        self.assertNotIn("cupy.cuda.nvrtc", protocol["runtime"]["explicit_import_modules_in_order"])
        self.assertEqual(
            protocol["inputs"]["predecessor_v2_setup_failure"]["setup_failure"]["failure_id"],
            M.V2_FAILURE_ID,
        )
        for key in (
            "dynamical_classification_generated", "resonance_classification_generated",
            "classical_population_inferred", "scattering_population_inferred",
            "detached_population_inferred", "planet_nine_evidence",
            "population_inference", "science_result",
        ):
            self.assertFalse(protocol["claims"][key], key)

    def test_prepare_and_validate_prepared_real_fresh_temp_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "stage15-v3-prepared"
            self.assertFalse(root.exists())
            protocol = M.prepare(root)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["PROTOCOL.json"],
            )
            payload = (root / "PROTOCOL.json").read_bytes()
            self.assertEqual(payload, M._canonical_json(protocol))
            self.assertEqual(M.validate_prepared(root), protocol)
            self.assertEqual((root / "PROTOCOL.json").stat().st_nlink, 1)
            self.assertEqual(
                protocol["evidence"]["restart_stream_binary_order"],
                "CASE_THEN_MONOLITHIC_FRESH_RESUMED_THEN_YEAR_500_1000_THEN_9_POS_THEN_9_VEL",
            )

    def test_adapter_wiring_calls_private_path_then_public_advance_each_year(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        advance = ast.unparse(functions["_advance_one_annual_segment"])
        lane = ast.unparse(functions["_run_lane"])
        restart = ast.unparse(functions["_restart_worker"])
        self.assertLess(advance.index("engine._build_source_path"), advance.index("engine.advance_cuda_massless_tracers"))
        self.assertIn("_require_bitwise_endpoint", advance)
        self.assertIn("for year in range(HORIZON_YEARS)", lane)
        self.assertIn("for year in range(500, 1000)", restart)
        self.assertIn("_advance_one_annual_segment", restart)
        midpoint = ast.unparse(functions["_midpoint_state"])
        self.assertIn("snapshot_id=base.snapshot_id", midpoint)
        self.assertNotIn("year-500-restart", midpoint)

    def test_bitwise_source_endpoint_crossbinding(self) -> None:
        position = _FakeVector((1.0, 2.0, 3.0))
        velocity = _FakeVector((4.0, 5.0, 6.0))
        state = types.SimpleNamespace(source_positions=position, source_velocities=velocity)
        observed = M._require_bitwise_endpoint(
            _FakeCP, _FakePath(position), _FakePath(velocity), state,
        )
        self.assertEqual(observed[0], hashlib.sha256(position.tobytes()).hexdigest())
        self.assertEqual(observed[1], hashlib.sha256(velocity.tobytes()).hexdigest())
        bad = types.SimpleNamespace(
            source_positions=_FakeVector((1.0, 2.0, math.nextafter(3.0, math.inf))),
            source_velocities=velocity,
        )
        with self.assertRaises(M.Stage15QualificationError):
            M._require_bitwise_endpoint(_FakeCP, _FakePath(position), _FakePath(velocity), bad)

    def test_route_jsonl_is_canonical_domain_bound_and_ordered(self) -> None:
        line = M._route_line(
            lane_order=0, batch_order=0, lane_id=M.LANES[0][0],
            object_id=M.OBJECT_IDS[0], segment_year=4, kind="ROUTE",
            record_order=0,
            payload={"tracer_index": 2, "source_index": 3, "global_step": 257},
        )
        lane = {
            "lane_order": 0, "batch_order": 0,
            "lane_id": M.LANES[0][0], "object_id": M.OBJECT_IDS[0],
        }
        routes, removals = M._parse_route_lane_payload(line, lane=lane, engine=_FakeEngine)
        self.assertEqual(routes, (_FakeRoute(2, 3, 257),))
        self.assertEqual(removals, ())
        tampered = line.replace(b'"global_step":257', b'"global_step":258')
        with self.assertRaises(M.Stage15QualificationError):
            M._parse_route_lane_payload(tampered, lane=lane, engine=_FakeEngine)

    def test_capped_writer_rejects_before_overflow_bytes_land(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "evidence.bin"
            writer = M._CappedBinaryWriter(path, 3)
            with self.assertRaises(M.Stage15QualificationError):
                writer.write(b"four")
            self.assertEqual(path.stat().st_size, 0)
            writer.write(b"abc")
            receipt = writer.close(exact=3)
            self.assertEqual(receipt["bytes"], 3)
            self.assertEqual(path.read_bytes(), b"abc")

    def test_global_route_record_cap_rejects_before_short_line_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "routes.jsonl"
            writer = M._CappedBinaryWriter(path, M.ROUTE_MAX_BYTES)
            writer.record_count = M.ROUTE_MAX_RECORDS
            result = types.SimpleNamespace(
                encounter_routes=(_FakeRoute(0, 0, 1),), removal_events=(),
            )
            with self.assertRaises(M.Stage15QualificationError):
                M._append_new_route_records(
                    writer, lane_order=0, batch_order=0,
                    lane_id=M.LANES[0][0], object_id=M.OBJECT_IDS[0], year=0,
                    result=result, prior_route_count=0, prior_removal_count=0,
                    global_record_order=0,
                )
            self.assertEqual(path.stat().st_size, 0)
            writer.abort()

    def test_held_fd_detects_path_replacement_after_semantic_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "artifact.bin"
            replacement = root / "replacement.bin"
            target.write_bytes(b"sealed")
            replacement.write_bytes(b"forged")
            with self.assertRaises(M.Stage15QualificationError):
                with M._stable_fd(target, "synthetic", exact_bytes=6) as (fd, _):
                    self.assertEqual(M._pread_all(fd, 6), b"sealed")
                    os.replace(replacement, target)
            self.assertEqual(target.read_bytes(), b"forged")

    def test_held_fd_detects_hardlink_created_during_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "artifact.bin"
            alias = Path(raw) / "late-hardlink.bin"
            target.write_bytes(b"sealed")
            with self.assertRaises(M.Stage15QualificationError):
                with M._stable_fd(target, "synthetic", exact_bytes=6) as (fd, _):
                    self.assertEqual(M._pread_all(fd, 6), b"sealed")
                    os.link(target, alias)

    def test_validation_receipt_hashes_replayed_document_not_reopened_path(self) -> None:
        validated = {
            "content_sha256": "1" * 64,
            "protocol_content_sha256": "2" * 64,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "RESULT.json").write_bytes(M._canonical_json({"replacement": True}))
            receipt = M._validation_receipt(validated, root)
        self.assertEqual(
            receipt["result_file_sha256"], M._sha256(M._canonical_json(validated)),
        )
        self.assertNotEqual(
            receipt["result_file_sha256"],
            M._sha256(M._canonical_json({"replacement": True})),
        )

    def test_restart_pair_rejects_suffix_and_final_step_tamper(self) -> None:
        payload = bytes(M.RESTART_STREAM_BYTES)
        request = {"content_sha256": "1" * 64}
        lane = {
            "steps_per_julian_year": 64,
            "final_state_sha256": "2" * 64,
            "final_route_ledger_sha256": "3" * 64,
            "final_removal_ledger_sha256": "4" * 64,
            "source_nodes": {"annual_segment_receipts": [{} for _ in range(1_000)]},
        }
        receipt = {
            "schema": M.RESTART_RECEIPT_SCHEMA,
            "request_content_sha256": request["content_sha256"],
            "suffix_file_sha256": M._sha256(payload),
            "final_global_step": 64_000,
            "final_state_sha256": lane["final_state_sha256"],
            "final_route_ledger_sha256": lane["final_route_ledger_sha256"],
            "final_removal_ledger_sha256": lane["final_removal_ledger_sha256"],
            "suffix_checkpoint_count": M.RESTART_YEAR_COUNT,
            "fresh_process": True, "midpoint_consumed": True,
            "source_segment_receipts_500_through_999": [{} for _ in range(500)],
        }
        receipt["content_sha256"] = M._domain_sha(M.RESTART_DOMAIN, receipt)
        M._require_restart_pair(payload, payload, receipt, lane, request)
        forged = bytearray(payload)
        forged[-1] = 1
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, bytes(forged), receipt, lane, request)
        bad = dict(receipt)
        bad["final_global_step"] = -999
        bad["content_sha256"] = M._domain_sha(M.RESTART_DOMAIN, bad)
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, payload, bad, lane, request)

    def test_raw_midpoint_forward_mapping_is_mutation_sensitive(self) -> None:
        v4, _bridge, _custody, batches, _locators = M._load_v4_context()
        batch = batches[0]
        source_values = [
            component
            for item in batch.massive
            for component in item.position_AU
        ] + [
            component
            for item in batch.massive
            for component in item.velocity_AU_per_julian_year
        ]
        tracer_values = [
            component
            for item in batch.tracers
            for component in item.position_AU
        ] + [
            component
            for item in batch.tracers
            for component in item.velocity_AU_per_julian_year
        ]
        source = struct.pack(f"<{len(source_values)}d", *source_values)
        tracer = struct.pack(f"<{len(tracer_values)}d", *tracer_values)
        expected = M._checkpoint_payload_from_raw(
            source_payload=source, tracer_payload=tracer,
            year=0, batch=batch, v4=v4,
        )
        nominal = v4._checkpoint(0, batch.massive, batch.tracers)
        self.assertEqual(expected, M._pack_checkpoint_document(nominal))
        altered = bytearray(tracer)
        value = struct.unpack_from("<d", altered, 0)[0]
        struct.pack_into("<d", altered, 0, math.nextafter(value, math.inf))
        self.assertNotEqual(
            M._checkpoint_payload_from_raw(
                source_payload=source, tracer_payload=bytes(altered),
                year=0, batch=batch, v4=v4,
            ),
            expected,
        )

    def test_midpoint_continuation_epoch_is_exactly_native_epoch_plus_500(self) -> None:
        batch = types.SimpleNamespace(epoch_jd_tdb=2_451_545.25)
        empty_sha = M._sha256(M._canonical_json([]))
        continuation = {
            "global_step": 500 * 64,
            "epoch_hex": M._repeated_year_epoch(batch, 500).hex(),
            "minimum_source_distance_AU_hex": [1.0.hex()] * M.TRACER_COUNT,
            "ever_unbound": [False] * M.TRACER_COUNT,
            "encounter_route_counts": [0] * M.TRACER_COUNT,
            "tracer_lifecycle": [0] * M.TRACER_COUNT,
            "prior_encounter_route_count": 0,
            "prior_removal_event_count": 0,
            "route_prefix_sha256": empty_sha,
            "removal_prefix_sha256": empty_sha,
        }
        M._validate_continuation(
            continuation, steps=64, prefix_routes=(), prefix_removals=(),
            batch=batch,
        )
        forged = dict(continuation)
        forged["epoch_hex"] = math.nextafter(
            M._repeated_year_epoch(batch, 500), math.inf,
        ).hex()
        with self.assertRaises(M.Stage15QualificationError):
            M._validate_continuation(
                forged, steps=64, prefix_routes=(), prefix_removals=(),
                batch=batch,
            )

    def test_full_synthetic_gate_evaluator_accepts_identical_lanes(self) -> None:
        v4, _bridge, _custody, batches, _locators = M._load_v4_context()
        checkpoint = v4._checkpoint(0, batches[0].massive, batches[0].tracers)
        checkpoints = []
        for year in range(M.CHECKPOINT_COUNT):
            row = dict(checkpoint)
            row["year_index"] = year
            row["time_julian_years_hex"] = float(year).hex()
            checkpoints.append(row)
        lane = {"model": "RESOLVED_11_BODY", "checkpoints": checkpoints}
        generated = {
            (batch, steps): lane for batch in range(12) for steps in (64, 128)
        }
        oracle = {batch: lane for batch in range(12)}
        zero = v4._fraction_record(Fraction(0, 1))
        records = [
            {
                "batch_order": batch, "steps_per_julian_year": steps,
                "all_node_summary": {
                    "maximum_relative_massive_energy_drift_exact": zero,
                    "maximum_relative_massive_angular_momentum_drift_hex": 0.0.hex(),
                },
                "no_removal_no_drop": True,
            }
            for batch in range(12) for steps in (64, 128)
        ]
        gates = M._evaluate_gates(generated, oracle, records, v4)
        self.assertEqual(gates["gate_count"], 20)
        self.assertTrue(gates["all_predeclared_gates_passed"])
        self.assertFalse(gates["classification_generated"])

    def test_json_integer_identity_guard_rejects_bool(self) -> None:
        for key in sorted(M._INTEGER_KEYS):
            with self.subTest(key=key):
                with self.assertRaises(M.Stage15QualificationError):
                    M._guard_json_integer_identities({key: False})
        for key in sorted(M._INTEGER_LIST_KEYS):
            with self.subTest(key=key):
                with self.assertRaises(M.Stage15QualificationError):
                    M._guard_json_integer_identities({key: [0, True]})
        payload = M._canonical_json({"lane_order": False})
        with self.assertRaises(M.Stage15QualificationError):
            M._load_canonical_json(payload, "bool mutation")

    def test_fresh_registry_rejects_spoof_even_when_not_moduletype(self) -> None:
        sentinel = object()
        old = sys.modules.get("cupy", sentinel)
        sys.modules["cupy"] = object()  # type: ignore[assignment]
        try:
            with self.assertRaises(M.Stage15QualificationError):
                M._reject_preloaded_runtime_modules()
        finally:
            if old is sentinel:
                sys.modules.pop("cupy", None)
            else:
                sys.modules["cupy"] = old  # type: ignore[assignment]

    def test_exact_venv_fresh_child_imports_supported_nvrtc_without_device_query(self) -> None:
        child_source = r'''
from pathlib import Path
import sys
import types

source = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
payload = source.read_bytes()
module = types.ModuleType("_jx_stage15_v3_import_only_child")
module.__file__ = str(source)
module.__package__ = ""
sys.modules[module.__name__] = module
exec(compile(payload, str(source), "exec", optimize=sys.flags.optimize), module.__dict__)
interpreter = module._fresh_child_interpreter_receipt()
fresh, modules = module._import_runtime_modules_only(cache_root)
nvrtc = module._nvrtc_module_receipt(modules[module.NVRTC_MODULE_NAME])
record = {
    "interpreter": interpreter,
    "fresh": fresh,
    "module_names": list(modules),
    "nvrtc": nvrtc,
    "nvrtc_api": [
        name for name in module.NVRTC_REQUIRED_API
        if callable(getattr(modules[module.NVRTC_MODULE_NAME], name, None))
    ],
}
sys.stdout.buffer.write(module._canonical_json(record))
'''
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_import_runtime_modules_only"
        )
        helper_text = ast.unparse(helper)
        production = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_fresh_runtime_imports"
        )
        production_text = ast.unparse(production)
        self.assertIn("_import_runtime_modules_only(cache_root)", production_text)
        self.assertNotIn("cupy.cuda.nvrtc", production_text)
        for forbidden in (
            "resolve_backend", "BackendSpec", "getDeviceCount",
            "runtimeGetVersion", "cuda_massless_tracer_runtime_identity",
        ):
            self.assertNotIn(forbidden, helper_text)
        with tempfile.TemporaryDirectory() as raw:
            cache_root = Path(raw) / "runtime-cache"
            environment = M._runtime_child_environment(cache_root)
            completed = subprocess.run(
                [
                    str(M.PROJECT_VENV_PYTHON), "-B", "-c", child_source,
                    str(SOURCE), str(cache_root),
                ],
                cwd=M.PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=120,
            )
        self.assertEqual(
            completed.returncode, 0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        record = json.loads(completed.stdout)
        self.assertEqual(record["module_names"], list(M.RUNTIME_IMPORT_MODULE_NAMES))
        self.assertNotIn("cupy.cuda.nvrtc", record["module_names"])
        self.assertEqual(record["nvrtc"]["module"], M.NVRTC_MODULE_NAME)
        self.assertEqual(record["nvrtc"]["bytes"], M.NVRTC_MODULE_BYTES)
        self.assertEqual(record["nvrtc"]["sha256"], M.NVRTC_MODULE_SHA256)
        self.assertEqual(record["nvrtc_api"], list(M.NVRTC_REQUIRED_API))
        self.assertEqual(
            record["interpreter"]["sys_executable"],
            str(M._lexical_absolute(M.PROJECT_VENV_PYTHON)),
        )
        self.assertEqual(record["fresh"]["preloaded_module_names"], [])

    def test_runtime_projection_tamper_fails_before_evidence_replay(self) -> None:
        projected = M._runtime_process_independent_projection(
            {
                "schema": M.RUNTIME_SCHEMA, "distributions": [],
                "critical_modules": [], "module_registry": [{"module": "sentinel"}],
                "adapter_callables": [], "engine_runtime_identity": {},
                "implementation": {},
            }
        )
        self.assertEqual(projected["module_registry"], [{"module": "sentinel"}])
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_output_worker"
        )
        calls = [
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(function) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        ]
        self.assertIn("_fresh_runtime_imports", calls)
        self.assertIn("_validate_runtime_document", calls)
        self.assertLess(calls.index("_validate_runtime_document"), calls.index("_validate_lane_evidence"))
        bad = {
            "path": str(Path(tempfile.gettempdir()) / "outside-runtime.py"),
            "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(),
        }
        with self.assertRaises(M.Stage15QualificationError):
            M._validate_retained_file_receipt(bad, "outside runtime")

    def test_runtime_guard_projection_is_exact_except_cache_root(self) -> None:
        def guard(root: str) -> dict[str, object]:
            return {
                "dont_write_bytecode": True,
                "pycache_prefix": f"{root}/pycache-must-remain-absent",
                "pycache_absent": True,
                "cupy_cache_directory": f"{root}/cupy-kernel-cache",
                "cupy_cache_entries": [], "cupy_cache_in_memory": "1",
                "cuda_cache_disable": "1",
                "claim": "POINT_OBSERVATION_IN_TRUSTED_PROCESS_NOT_CONCURRENT_MUTATION_PROOF",
            }
        self.assertEqual(
            M._normalized_runtime_guard(guard("/tmp/one")),
            M._normalized_runtime_guard(guard("/tmp/two")),
        )
        relative = guard("relative")
        with self.assertRaises(M.Stage15QualificationError):
            M._normalized_runtime_guard(relative)
        for key, replacement in (
            ("dont_write_bytecode", False), ("pycache_absent", False),
            ("cupy_cache_entries", ["forged"]), ("cupy_cache_in_memory", "0"),
            ("cuda_cache_disable", "0"), ("claim", "forged"),
        ):
            changed = guard("/tmp/one")
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(M.Stage15QualificationError):
                M._normalized_runtime_guard(changed)

    def test_exact_result_lane_and_runtime_semantic_key_sets(self) -> None:
        result = {key: None for key in M.RESULT_KEYS}
        lane = {key: None for key in M.LANE_KEYS}
        self.assertIs(M._require_exact_mapping_keys(result, M.RESULT_KEYS, "result"), result)
        self.assertIs(M._require_exact_mapping_keys(lane, M.LANE_KEYS, "lane"), lane)
        result["planet_nine_evidence"] = True
        lane["dynamical_classification"] = "SCATTERING"
        with self.assertRaises(M.Stage15QualificationError):
            M._require_exact_mapping_keys(result, M.RESULT_KEYS, "result")
        with self.assertRaises(M.Stage15QualificationError):
            M._require_exact_mapping_keys(lane, M.LANE_KEYS, "lane")
        runtime_keys = {
            "schema", "fresh_child_interpreter", "fresh_process_registry",
            "guard", "distributions", "critical_modules", "module_registry",
            "adapter_callables", "engine_runtime_identity", "implementation",
            "claim", "content_sha256",
        }
        runtime = {key: None for key in runtime_keys}
        runtime["schema"] = M.RUNTIME_SCHEMA
        runtime["claim"] = "forged custody ceiling"
        runtime["content_sha256"] = M._domain_sha(M.RUNTIME_DOMAIN, runtime)
        with self.assertRaises(M.Stage15QualificationError):
            M._validate_runtime_document(runtime, runtime)

    def test_lane_semantic_relabels_fail_closed(self) -> None:
        runtime_sha = "a" * 64
        batch = types.SimpleNamespace(object_id=M.OBJECT_IDS[0])
        lane = {key: None for key in M.LANE_KEYS}
        lane.update({
            "lane_order": 0,
            "batch_order": 0,
            "object_id": batch.object_id,
            "lane_id": M.LANES[0][0],
            "engine": M.LANE_ENGINE,
            "model": M.LANE_MODEL,
            "steps_per_julian_year": M.LANES[0][1],
            "step_julian_years_hex": (1.0 / M.LANES[0][1]).hex(),
            "final_global_step": M.HORIZON_YEARS * M.LANES[0][1],
            "runtime_projection_sha256": runtime_sha,
            "no_removal_no_drop": True,
        })
        self.assertEqual(
            M._validate_lane_identity(
                lane, expected_order=0, batch=batch,
                runtime_projection_sha256=runtime_sha,
            ),
            (M.LANES[0][0], M.LANES[0][1], 0),
        )
        for key, replacement in (
            ("engine", "RELABELED_ENGINE"),
            ("model", "REDUCED_5_BODY"),
            ("step_julian_years_hex", (1.0 / 128.0).hex()),
        ):
            forged = dict(lane)
            forged[key] = replacement
            with self.subTest(key=key), self.assertRaises(M.Stage15QualificationError):
                M._validate_lane_identity(
                    forged, expected_order=0, batch=batch,
                    runtime_projection_sha256=runtime_sha,
                )

    def test_route_source_witness_is_bound_to_adjacent_retained_nodes(self) -> None:
        class FakeNP:
            float64 = float

            @staticmethod
            def asarray(value: object, dtype: object = None) -> object:
                del dtype
                return value

            @staticmethod
            def ones(count: int, dtype: object = None) -> tuple[float, ...]:
                del dtype
                return (1.0,) * count

            @staticmethod
            def zeros(count: int, dtype: object = None) -> tuple[float, ...]:
                del dtype
                return (0.0,) * count

            @staticmethod
            def full(count: int, value: float, dtype: object = None) -> tuple[float, ...]:
                del dtype
                return (value,) * count

        class FakeWitnessEngine:
            calls = 0

            @staticmethod
            def _source_segment_sha256(*_args: object) -> str:
                return "a" * 64

            @classmethod
            def _validate_route_witness(cls, *_args: object, **_kwargs: object) -> None:
                cls.calls += 1

        massive = tuple(
            types.SimpleNamespace(
                body_id=f"BODY-{index}", gm_AU3_per_julian_year2=float(index + 1),
            )
            for index in range(M.MASSIVE_COUNT)
        )
        batch = types.SimpleNamespace(epoch_jd_tdb=2_451_545.0, massive=massive)
        p0 = tuple((float(index + 1), 0.0, 0.0) for index in range(M.MASSIVE_COUNT))
        v0 = tuple((0.0, float(index + 1), 0.0) for index in range(M.MASSIVE_COUNT))
        p1 = tuple((row[0], row[1] + 0.25, row[2]) for row in p0)
        v1 = v0
        gm = tuple(item.gm_AU3_per_julian_year2 for item in massive)
        node0 = struct.pack(
            f"<{M.MASSIVE_COUNT * 6}d",
            *(component for row in p0 for component in row),
            *(component for row in v0 for component in row),
        )
        node1 = struct.pack(
            f"<{M.MASSIVE_COUNT * 6}d",
            *(component for row in p1 for component in row),
            *(component for row in v1 for component in row),
        )
        record_values = {
            "global_step": 0, "tracer_index": 0, "source_index": 0,
            "source_id": massive[0].body_id, "source_segment_position_0_AU": p0,
            "source_segment_velocity_0_AU_per_year": v0,
            "source_segment_position_1_AU": p1,
            "source_segment_velocity_1_AU_per_year": v1,
            "source_segment_gm_AU3_per_year2": gm,
            "source_segment_step_years": 1.0 / 64.0,
            "source_segment_sha256": "a" * 64,
            "source_segment_epoch_0": 0.0,
            "route_relative_tolerance": 1.0e-12,
            "route_absolute_tolerance": 1.0e-14,
            "route_maximum_substep_fraction": 0.125,
            "route_event_fraction_tolerance": 1.0e-12,
            "route_maximum_primary_dynamical_phase": 0.05,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "source.bin"
            path.write_bytes(node0 + node1)
            fd = os.open(path, os.O_RDONLY)
            try:
                M._validate_one_route_source_witness(
                    record=types.SimpleNamespace(**record_values), kind="ROUTE",
                    segment_year=0, steps=64, batch=batch, source_fd=fd,
                    source_offset=0, np=FakeNP, engine=FakeWitnessEngine,
                    endpoint_position=p1, endpoint_velocity=v1,
                )
                self.assertEqual(FakeWitnessEngine.calls, 1)
                forged = dict(record_values)
                forged_p0 = list(p0)
                forged_p0[0] = (math.nextafter(p0[0][0], math.inf), 0.0, 0.0)
                forged["source_segment_position_0_AU"] = tuple(forged_p0)
                with self.assertRaises(M.Stage15QualificationError):
                    M._validate_one_route_source_witness(
                        record=types.SimpleNamespace(**forged), kind="ROUTE",
                        segment_year=0, steps=64, batch=batch, source_fd=fd,
                        source_offset=0, np=FakeNP, engine=FakeWitnessEngine,
                        endpoint_position=p1, endpoint_velocity=v1,
                    )
            finally:
                os.close(fd)

    def test_publication_is_create_only_result_last_and_failure_never_unlinks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            staging = base / "staging"
            output = base / "output"
            staging.mkdir()
            output.mkdir()
            for name in M.ARTIFACT_ROSTER:
                (staging / name).write_bytes(name.encode("ascii"))
            (output / "PROTOCOL.json").write_bytes(b"PROTOCOL.json")
            seen: list[tuple[str, str]] = []
            original_link = M._link_create_only
            original_write = M._write_create_only

            def recording(source: Path, destination: Path) -> None:
                seen.append(("LINK", destination.name))
                original_link(source, destination)

            def recording_write(path: Path, payload: bytes, *, maximum: int) -> None:
                seen.append(("CREATE", path.name))
                original_write(path, payload, maximum=maximum)

            with mock.patch.object(M, "_link_create_only", side_effect=recording), mock.patch.object(
                M, "_write_create_only", side_effect=recording_write,
            ):
                M._publish_result_last(staging, output)
            self.assertEqual(seen[-1], ("CREATE", "RESULT.json"))
            self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(M.ARTIFACT_ROSTER))
            self.assertEqual(
                sorted(path.name for path in staging.iterdir()),
                ["PROTOCOL.json", "RESULT.json"],
            )
            self.assertTrue(all(path.stat().st_nlink == 1 for path in output.iterdir()))

        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            staging = base / "staging"
            output = base / "output"
            staging.mkdir()
            output.mkdir()
            for name in M.ARTIFACT_ROSTER:
                (staging / name).write_bytes(name.encode("ascii"))
            (output / "PROTOCOL.json").write_bytes(b"PROTOCOL.json")
            original = M._link_create_only

            def fail_after_first(source: Path, destination: Path) -> None:
                if destination.name == "SOURCE_NODES.bin":
                    raise M.Stage15QualificationError("synthetic concurrent failure")
                original(source, destination)

            with mock.patch.object(M, "_link_create_only", side_effect=fail_after_first):
                with self.assertRaises(M.Stage15QualificationError):
                    M._publish_result_last(staging, output)
            self.assertTrue((output / "ARRAYS.bin").exists())
            self.assertFalse((output / "RESULT.json").exists())

        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            staging = base / "staging"
            output = base / "output"
            staging.mkdir(); output.mkdir()
            for name in M.ARTIFACT_ROSTER:
                (staging / name).write_bytes(name.encode("ascii"))
            (output / "PROTOCOL.json").write_bytes(b"PROTOCOL.json")
            original_fsync = M._fsync_directory
            calls = 0

            def fail_post_commit(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic post-result directory fsync failure")
                original_fsync(path)

            with mock.patch.object(M, "_fsync_directory", side_effect=fail_post_commit):
                with self.assertRaises(OSError):
                    M._publish_result_last(staging, output)
            self.assertTrue((output / "RESULT.json").is_file())
            self.assertIn(
                "COMPLETE_ROSTER_MAY_REMAIN",
                M.build_protocol()["evidence"]["post_result_creation_fsync_failure"],
            )

    def test_cli_and_validator_use_fresh_project_venv_children(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH)', source)
        self.assertIn('"validate-worker"', source)
        self.assertIn("_validate_output_in_fresh_child(staging)", source)
        self.assertIn("return _validate_output_in_fresh_child", source)
        self.assertGreaterEqual(
            source.count("_validate_runtime_document(runtime, runtime_after)"), 2,
        )
        self.assertIn('"runtime_fresh_process_registry": runtime["fresh_process_registry"]', source)
        self.assertIn('"runtime_guard": runtime["guard"]', source)
        execute_source = ast.unparse(
            next(
                node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "execute"
            )
        )
        self.assertIn("cache_root = runtime_work / 'runtime-cache'", execute_source)
        self.assertNotIn("cache_root = staging", execute_source)
        parser = M._build_parser()
        parsed = parser.parse_args(["validate"])
        self.assertEqual(parsed.command, "validate")

    def test_large_evidence_validation_uses_one_held_fd_and_pread(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        worker = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_output_worker"
        )
        text = ast.unparse(worker)
        self.assertIn("_stable_fd(root / 'SOURCE_NODES.bin'", text)
        self.assertIn("_stable_fd(root / 'ARRAYS.bin'", text)
        self.assertIn("_stable_fd(root / 'RESTART_STREAMS.bin'", text)
        self.assertIn("_validate_lane_evidence", text)
        self.assertIn("_validate_restart_evidence", text)
        self.assertNotIn("read_bytes", text)


if __name__ == "__main__":
    unittest.main()
