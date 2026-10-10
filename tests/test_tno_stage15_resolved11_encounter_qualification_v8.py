from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import ast
import copy
import hashlib
import importlib.machinery
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
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v8.py"
MODULE_NAME = "_jx_tno_stage15_resolved11_encounter_qualification_v8_test_bound"


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


def _segment_sha(label: str, year: int) -> str:
    return hashlib.sha256(f"{label}:{year}".encode("ascii")).hexdigest()


def _public_segment_receipt(year: int, steps: int = 64) -> dict[str, object]:
    position_sha = _segment_sha("position", year)
    velocity_sha = _segment_sha("velocity", year)
    return {
        "year_start": year,
        "year_end": year + 1,
        "starting_global_step": year * steps,
        "ending_global_step": (year + 1) * steps,
        "prepass_path_node_count": steps + 1,
        "prepass_full_payload_sha256": _segment_sha("prepass", year),
        "precomputed_endpoint_position_sha256": position_sha,
        "public_endpoint_position_sha256": position_sha,
        "precomputed_endpoint_velocity_sha256": velocity_sha,
        "public_endpoint_velocity_sha256": velocity_sha,
        "public_input_state_sha256": _segment_sha("input", year),
        "bitwise_source_endpoint_match": True,
    }


def _merged_segment_receipt(year: int, steps: int = 64) -> dict[str, object]:
    return {
        **_public_segment_receipt(year, steps),
        "retained_unique_node_count": steps + (1 if year == 0 else 0),
        "retained_payload_sha256": _segment_sha("retained", year),
        "prepass_start_position_sha256": _segment_sha("start", year),
        "prepass_endpoint_position_sha256": _segment_sha("position", year),
        "prepass_endpoint_velocity_sha256": _segment_sha("velocity", year),
    }


def _restart_pair_fixture() -> tuple[
    bytes, dict[str, object], dict[str, object], dict[str, object]
]:
    payload = bytes(M.RESTART_STREAM_BYTES)
    request: dict[str, object] = {"content_sha256": "1" * 64}
    monolithic_rows = [_merged_segment_receipt(year) for year in range(1_000)]
    restart_rows = [_public_segment_receipt(year) for year in range(500, 1_000)]
    lane: dict[str, object] = {
        "steps_per_julian_year": 64,
        "final_state_sha256": "2" * 64,
        "final_route_ledger_sha256": "3" * 64,
        "final_removal_ledger_sha256": "4" * 64,
        "source_nodes": {"annual_segment_receipts": monolithic_rows},
    }
    receipt: dict[str, object] = {
        "schema": M.RESTART_RECEIPT_SCHEMA,
        "request_content_sha256": request["content_sha256"],
        "suffix_file_sha256": M._sha256(payload),
        "final_global_step": 64_000,
        "final_state_sha256": lane["final_state_sha256"],
        "final_route_ledger_sha256": lane["final_route_ledger_sha256"],
        "final_removal_ledger_sha256": lane["final_removal_ledger_sha256"],
        "suffix_checkpoint_count": M.RESTART_YEAR_COUNT,
        "fresh_process": True,
        "midpoint_consumed": True,
        "source_segment_receipts_500_through_999": restart_rows,
        "source_segment_receipts_sha256": M._sha256(M._canonical_json(restart_rows)),
    }
    receipt["content_sha256"] = M._domain_sha(M.RESTART_DOMAIN, receipt)
    return payload, request, lane, receipt


def _reseal_restart_receipt(receipt: dict[str, object]) -> None:
    receipt["content_sha256"] = M._domain_sha(M.RESTART_DOMAIN, receipt)


class Stage15V8Tests(unittest.TestCase):
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

    def test_v3_is_byte_preserved_and_zero_lane_runtime_shadow_failure_is_explicit(self) -> None:
        predecessor = M._validate_v3_predecessor()
        self.assertEqual(predecessor["source"]["file_sha256"], M.V3_SOURCE_SHA256)
        self.assertEqual(predecessor["test"]["file_sha256"], M.V3_TEST_SHA256)
        self.assertEqual(predecessor["protocol"]["file_sha256"], M.V3_PROTOCOL_SHA256)
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            M.V3_PROTOCOL_CONTENT_SHA256,
        )
        self.assertEqual(predecessor["git_custody"]["commit"], M.V3_GIT_COMMIT)
        self.assertEqual(predecessor["git_custody"]["branch"], M.V3_GIT_BRANCH)
        self.assertEqual(predecessor["git_custody"]["annotated_tag"], M.V3_GIT_TAG)
        self.assertEqual(predecessor["git_custody"]["tag_object"], M.V3_GIT_TAG_OBJECT)
        self.assertEqual(predecessor["failed_root"]["exact_roster"], ["PROTOCOL.json"])
        self.assertEqual(predecessor["failed_root"]["attempt_count"], 1)
        self.assertEqual(predecessor["failed_root"]["completed_lane_count"], 0)
        self.assertFalse(predecessor["failed_root"]["dynamics_executed"])
        failure = predecessor["setup_failure"]
        self.assertEqual(failure["failure_id"], M.V3_FAILURE_ID)
        self.assertEqual(failure["failure_class"], "AttributeError")
        self.assertEqual(
            failure["failure_message"],
            "module 'cupy.cuda.runtime' has no attribute '_getLocalRuntimeVersion'",
        )
        self.assertEqual(failure["shadowing_import"], "cupy.cuda.runtime")
        self.assertEqual(failure["required_backend_module"], M.CUDA_RUNTIME_MODULE_NAME)
        self.assertFalse(failure["lane_dynamics_started"])
        self.assertFalse(failure["dynamics_outcome_generated"])
        self.assertFalse(failure["result_artifacts_published"])

    def test_v4_is_byte_preserved_and_partial_first_lane_failure_is_explicit(self) -> None:
        predecessor = M._validate_stage15_v4_predecessor()
        self.assertEqual(
            predecessor["source"]["file_sha256"], M.STAGE15_V4_SOURCE_SHA256,
        )
        self.assertEqual(
            predecessor["test"]["file_sha256"], M.STAGE15_V4_TEST_SHA256,
        )
        self.assertEqual(
            predecessor["protocol"]["file_sha256"],
            M.STAGE15_V4_PROTOCOL_SHA256,
        )
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            M.STAGE15_V4_PROTOCOL_CONTENT_SHA256,
        )
        self.assertEqual(
            predecessor["git_custody"]["commit"], M.STAGE15_V4_GIT_COMMIT,
        )
        self.assertEqual(
            predecessor["git_custody"]["branch"], M.STAGE15_V4_GIT_BRANCH,
        )
        self.assertEqual(
            predecessor["git_custody"]["annotated_tag"], M.STAGE15_V4_GIT_TAG,
        )
        self.assertEqual(
            predecessor["git_custody"]["tag_object"],
            M.STAGE15_V4_GIT_TAG_OBJECT,
        )
        failed = predecessor["failed_root"]
        self.assertEqual(failed["exact_roster"], ["PROTOCOL.json"])
        self.assertEqual(failed["attempt_count"], 1)
        self.assertEqual(failed["completed_lane_count"], 0)
        self.assertTrue(failed["partial_dynamics_executed"])
        self.assertFalse(failed["completed_segment_count_exactly_known"])
        self.assertIsNone(failed["completed_segment_count"])
        self.assertFalse(failed["result_artifacts_published"])
        failure = predecessor["execution_failure"]
        self.assertEqual(failure["failure_id"], M.STAGE15_V4_FAILURE_ID)
        self.assertEqual(failure["failure_class"], M.STAGE15_V4_FAILURE_CLASS)
        self.assertEqual(failure["failure_message"], M.STAGE15_V4_FAILURE_MESSAGE)
        self.assertEqual(
            failure["repair_policy"], M.ACTIVE_ENDPOINT_MINIMUM_POLICY,
        )
        self.assertTrue(failure["partial_dynamics_executed"])
        self.assertFalse(failure["qualification_established"])
        self.assertFalse(failure["dynamics_outcome_generated"])

    def test_v5_is_byte_preserved_and_both_smoke_failures_are_disclosed(self) -> None:
        predecessor = M._validate_stage15_v5_predecessor()
        self.assertEqual(
            predecessor["source"]["file_sha256"], M.STAGE15_V5_SOURCE_SHA256,
        )
        self.assertEqual(
            predecessor["test"]["file_sha256"], M.STAGE15_V5_TEST_SHA256,
        )
        protocol = predecessor["reconstructed_protocol"]
        self.assertEqual(protocol["bytes"], M.STAGE15_V5_PROTOCOL_BYTES)
        self.assertEqual(protocol["file_sha256"], M.STAGE15_V5_PROTOCOL_SHA256)
        self.assertEqual(
            protocol["content_sha256"], M.STAGE15_V5_PROTOCOL_CONTENT_SHA256,
        )
        self.assertFalse(predecessor["failed_root"]["exists"])
        self.assertEqual(predecessor["failed_root"]["exact_roster"], [])
        attempts = predecessor["smoke_certification_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            [row["outer_receipt_bytes"] for row in attempts], [0, 0],
        )
        self.assertEqual(
            [row["failure_message"] for row in attempts],
            [M.STAGE15_V5_ATTEMPT_1_MESSAGE, M.STAGE15_V5_ATTEMPT_2_MESSAGE],
        )
        self.assertTrue(
            attempts[1]["post_segment_post_continuation_runtime_capture_reached"],
        )
        self.assertFalse(any(row["certified_smoke_pass"] for row in attempts))

    def test_v6_preregistration_and_all_576_false_bound_routes_are_exact(self) -> None:
        predecessor = M._validate_stage15_v6_predecessor()
        self.assertEqual(
            predecessor["source"]["file_sha256"], M.STAGE15_V6_SOURCE_SHA256,
        )
        self.assertEqual(
            predecessor["test"]["file_sha256"], M.STAGE15_V6_TEST_SHA256,
        )
        self.assertEqual(
            predecessor["protocol"]["file_sha256"],
            M.STAGE15_V6_PROTOCOL_SHA256,
        )
        self.assertEqual(
            predecessor["git_custody"]["preregistration_commit"],
            M.STAGE15_V6_GIT_COMMIT,
        )
        self.assertEqual(
            predecessor["git_custody"]["postrun_evidence_commit"],
            M.STAGE15_V6_POSTRUN_EVIDENCE_COMMIT,
        )
        self.assertEqual(
            predecessor["git_custody"]["tag_object"],
            M.STAGE15_V6_GIT_TAG_OBJECT,
        )
        smoke = predecessor["retained_smoke"]
        self.assertEqual(smoke["file_sha256"], M.STAGE15_V6_SMOKE_SHA256)
        self.assertEqual(smoke["route_count"], 576)
        self.assertEqual(smoke["route_reason"], "ACCELERATION_BOUND")
        self.assertEqual(smoke["route_source_id"], "MERCURY")
        self.assertEqual(smoke["route_outcome"], "REJOINED")
        self.assertEqual(smoke["removal_count"], 0)
        self.assertTrue(smoke["nonqualifying"])
        self.assertFalse(smoke["qualification_established"])
        self.assertEqual(predecessor["failed_root"]["exact_roster"], ["PROTOCOL.json"])

    def test_v7_is_byte_preserved_and_restart_schema_failure_is_explicit(self) -> None:
        predecessor = M._validate_stage15_v7_predecessor()
        self.assertEqual(
            predecessor["source"]["file_sha256"], M.STAGE15_V7_SOURCE_SHA256,
        )
        self.assertEqual(
            predecessor["test"]["file_sha256"], M.STAGE15_V7_TEST_SHA256,
        )
        self.assertEqual(
            predecessor["protocol"]["file_sha256"], M.STAGE15_V7_PROTOCOL_SHA256,
        )
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            M.STAGE15_V7_PROTOCOL_CONTENT_SHA256,
        )
        self.assertEqual(
            predecessor["git_custody"]["preregistration_commit"],
            M.STAGE15_V7_GIT_COMMIT,
        )
        self.assertEqual(
            predecessor["git_custody"]["tag_object"], M.STAGE15_V7_GIT_TAG_OBJECT,
        )
        self.assertEqual(predecessor["failed_root"]["exact_roster"], ["PROTOCOL.json"])
        self.assertEqual(predecessor["failed_root"]["result_artifact_count"], 0)
        self.assertFalse(predecessor["failed_root"]["result_artifacts_published"])
        failure = predecessor["failed_attempt"]
        self.assertEqual(failure["failure_id"], M.STAGE15_V7_FAILURE_ID)
        self.assertEqual(failure["failure_class"], M.STAGE15_V7_FAILURE_CLASS)
        self.assertEqual(failure["exact_failure_message"], M.STAGE15_V7_FAILURE_MESSAGE)
        self.assertTrue(failure["partial_dynamics_executed"])
        self.assertTrue(failure["transient_numerical_state_produced"])
        self.assertFalse(failure["retained_qualification_outcome_generated"])
        self.assertFalse(failure["result_artifacts_published"])
        self.assertFalse(failure["qualification_established"])

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
        self.assertEqual(
            protocol["adapter"]["active_endpoint_minimum_policy"],
            M.ACTIVE_ENDPOINT_MINIMUM_POLICY,
        )
        self.assertIn("NO_TOLERANCE", protocol["adapter"]["active_endpoint_minimum_rule"])
        self.assertEqual(
            protocol["adapter"]["engine_contract_test_sha256"],
            M.ENCOUNTER_TEST_SHA256,
        )
        self.assertIn(
            "_canonicalize_active_endpoint_minimum_host",
            protocol["adapter"]["pinned_private_callables"],
        )
        self.assertIn(
            "_validate_continuation",
            protocol["adapter"]["pinned_private_callables"],
        )
        self.assertEqual(protocol["evidence"]["fixed_exact_bytes"], M.FIXED_ARTIFACTS)
        self.assertEqual(protocol["evidence"]["exact_roster"], list(M.ARTIFACT_ROSTER))
        restart = protocol["restart"]
        self.assertEqual(restart["source_segment_receipt_count_per_case"], 500)
        self.assertEqual(len(M.PUBLIC_SEGMENT_RECEIPT_KEYS), 12)
        self.assertEqual(len(M.PREPASS_STORAGE_ONLY_RECEIPT_KEYS), 5)
        self.assertEqual(len(M.MERGED_SEGMENT_RECEIPT_KEYS), 17)
        self.assertTrue(
            set(M.PUBLIC_SEGMENT_RECEIPT_KEYS).isdisjoint(
                M.PREPASS_STORAGE_ONLY_RECEIPT_KEYS
            )
        )
        self.assertEqual(
            restart["public_segment_receipt_keys_in_order"],
            list(M.PUBLIC_SEGMENT_RECEIPT_KEYS),
        )
        self.assertEqual(
            restart["prepass_storage_receipt_keys"],
            list(M.PREPASS_STORAGE_ONLY_RECEIPT_KEYS),
        )
        self.assertIn("PROJECTED", restart["receipt_projection_rule"])
        self.assertIn("HELD_SOURCE_NODES_FD", restart["prepass_storage_validation_rule"])
        self.assertEqual(
            protocol["runtime"]["explicit_import_modules_in_order"],
            list(M.RUNTIME_IMPORT_MODULE_NAMES),
        )
        self.assertEqual(protocol["runtime"]["nvrtc_module"], M.NVRTC_MODULE_NAME)
        self.assertNotIn("cupy.cuda.nvrtc", protocol["runtime"]["explicit_import_modules_in_order"])
        self.assertEqual(
            protocol["runtime"]["cuda_runtime_module"],
            M.CUDA_RUNTIME_MODULE_NAME,
        )
        self.assertNotIn(
            "cupy.cuda.runtime", protocol["runtime"]["explicit_import_modules_in_order"],
        )
        self.assertEqual(
            protocol["runtime"]["full_runtime_receipt_preflight"]["source_step_count"],
            0,
        )
        self.assertFalse(
            protocol["runtime"]["full_runtime_receipt_preflight"]["dynamics_executed"],
        )
        self.assertEqual(
            protocol["runtime"]["exact_declared_module_name_aliases"],
            {M.SCIPY_UARRAY_REGISTRY_NAME: M.SCIPY_UARRAY_DECLARED_NAME},
        )
        self.assertEqual(
            protocol["runtime"]["runtime_prefix_scan_declared_name_alias_count"],
            1,
        )
        self.assertFalse(
            protocol["runtime"]["arbitrary_module_aliases_or_specless_modules_accepted"],
        )
        self.assertEqual(
            protocol["inputs"]["predecessor_v2_setup_failure"]["setup_failure"]["failure_id"],
            M.V2_FAILURE_ID,
        )
        self.assertEqual(
            protocol["inputs"]["predecessor_v3_setup_failure"]["setup_failure"]["failure_id"],
            M.V3_FAILURE_ID,
        )
        self.assertEqual(
            protocol["inputs"]["predecessor_v4_partial_dynamics_failure"]
            ["execution_failure"]["failure_id"],
            M.STAGE15_V4_FAILURE_ID,
        )
        self.assertEqual(
            protocol["runtime"]["trust_boundary"],
            "NO_HOSTILE_CODE_IN_FRESH_CHILD",
        )
        self.assertFalse(
            protocol["runtime"]["full_interpreter_or_module_registry_authentication_claimed"],
        )
        self.assertTrue(
            protocol["runtime"]["complete_runtime_prefix_inventories_retained_before_and_after"],
        )
        bound = protocol["adapter"]["acceleration_bound_certificate"]
        self.assertEqual(bound["version"], M.ACCELERATION_BOUND_CERTIFICATE_VERSION)
        self.assertEqual(bound["operation_count"], 512)
        self.assertTrue(bound["offline_exact_rederivation_from_held_fd_source_nodes"])
        self.assertFalse(bound["empirical_tolerance_used"])
        self.assertIn("complete 1000 year source", protocol["adapter"]["annual_rule"].lower().replace("_", " "))
        smoke = protocol["runtime"]["two_lane_one_segment_continuation_smoke"]
        self.assertEqual(smoke["smoke_segment_count"], 2)
        self.assertEqual(smoke["steps_per_julian_year_in_order"], [64, 128])
        self.assertTrue(smoke["partial_dynamics_executed"])
        self.assertTrue(smoke["setup_numerical_outcome_generated"])
        self.assertTrue(smoke["setup_numerical_outcome_disclosed"])
        self.assertFalse(smoke["qualification_established"])
        self.assertFalse(smoke["qualification_outcome_generated"])
        self.assertFalse(smoke["default_output_root_mutated"])
        for key in (
            "dynamical_classification_generated", "resonance_classification_generated",
            "classical_population_inferred", "scattering_population_inferred",
            "detached_population_inferred", "planet_nine_evidence",
            "population_inference", "science_result",
        ):
            self.assertFalse(protocol["claims"][key], key)

    def test_prepare_and_validate_prepared_real_fresh_temp_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "stage15-v8-prepared"
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

    def test_adapter_wiring_prepasses_then_restarts_before_public_segments(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        prepass = ast.unparse(functions["_build_lane_source_prepass"])
        advance = ast.unparse(functions["_advance_one_annual_segment"])
        lane = ast.unparse(functions["_run_lane"])
        restart = ast.unparse(functions["_restart_worker"])
        self.assertIn("for year in range(HORIZON_YEARS)", prepass)
        self.assertEqual(prepass.count("engine._build_source_path"), 1)
        self.assertNotIn("engine.advance_cuda_massless_tracers", prepass)
        self.assertIn("engine.advance_cuda_massless_tracers", advance)
        self.assertNotIn("engine._build_source_path", advance)
        self.assertIn("public source endpoint is not bitwise equal", advance)
        self.assertIn("for year in range(HORIZON_YEARS)", lane)
        self.assertLess(
            lane.index("_build_lane_source_prepass"),
            lane.index("_make_engine_state"),
        )
        self.assertIn("acceleration_bounds=acceleration_bounds", lane)
        self.assertIn("for year in range(500, 1000)", restart)
        self.assertIn("_pread_all", restart)
        self.assertIn("_derive_acceleration_bound_certificate", restart)
        self.assertIn(
            "restart acceleration-bound certificate differs from source evidence",
            restart,
        )
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

    def test_directed_rational_bound_certificate_is_outward_and_tamper_sensitive(self) -> None:
        import numpy as np

        gm = [0.5 + 0.125 * index for index in range(M.MASSIVE_COUNT)]
        rows = np.zeros((4, 2, M.MASSIVE_COUNT, 3), dtype="<f8")
        for node in range(4):
            for source in range(M.MASSIVE_COUNT):
                rows[node, 0, source] = (
                    0.2 * source + 0.001 * node,
                    0.03 * source - 0.0005 * node,
                    0.01 * source,
                )
        payload = rows.tobytes(order="C")
        certificate = M._derive_acceleration_bound_certificate(
            np, payload, gm, M.MASSIVE_IDS, block_nodes=2,
        )
        bounds = M._validate_bound_certificate_shape(certificate)
        self.assertEqual(len(bounds), M.MASSIVE_COUNT)
        self.assertEqual(
            certificate["operation_count_proof"], M.BOUND_OPERATION_BREAKDOWN,
        )
        self.assertEqual(
            sum(M.BOUND_OPERATION_BREAKDOWN[key] for key in (
                "force_evaluation_operations_max",
                "relative_subtraction_and_norm_operations",
                "reserved_operations",
            )),
            M.BOUND_OPERATION_COUNT,
        )
        for target, (row, bound) in enumerate(
            zip(certificate["sources"], bounds, strict=True)
        ):
            tracer = M._fraction_from_record(
                row["tracer_envelope_exact"], "test tracer",
            )
            source = M._fraction_from_record(
                row["source_pairwise_sum_exact_at_interval_witness"],
                "test source",
            )
            self.assertGreaterEqual(Fraction.from_float(bound), tracer + source)
            width = M._fraction_from_record(
                row["interval_width_over_exact_witness"], "test width",
            )
            self.assertEqual(Fraction.from_float(bound) - tracer - source, width)
            exact_node_max = max(
                M._exact_pairwise_source_sum_at_node(
                    rows[node, 0].tolist(), gm, target,
                )[0]
                for node in range(rows.shape[0])
            )
            self.assertGreaterEqual(
                Fraction.from_float(bound), tracer + exact_node_max,
            )
        q = Fraction(1, 1 << 1200)
        outward = M._ceil_positive_fraction_binary64(q)
        self.assertGreaterEqual(Fraction.from_float(outward), q)
        self.assertLess(Fraction.from_float(math.nextafter(outward, -math.inf)), q)

        forged = json.loads(json.dumps(certificate))
        forged["sources"][0]["bound_hex"] = math.nextafter(
            float.fromhex(forged["sources"][0]["bound_hex"]), -math.inf,
        ).hex()
        forged["bound_hex_in_source_order"][0] = forged["sources"][0]["bound_hex"]
        forged["content_sha256"] = M._domain_sha(M.BOUND_DOMAIN, forged)
        with self.assertRaises(M.Stage15QualificationError):
            M._validate_bound_certificate_shape(forged)

        forged = json.loads(json.dumps(certificate))
        inflated = forged["sources"][0]["roundoff_inflated_exact_upper"]
        inflated["exact_numerator"] += 1
        forged["content_sha256"] = M._domain_sha(M.BOUND_DOMAIN, forged)
        with self.assertRaises(M.Stage15QualificationError):
            M._validate_bound_certificate_shape(forged)

        changed = bytearray(payload)
        source_one_x_offset = 3 * 8
        first = struct.unpack_from("<d", changed, source_one_x_offset)[0]
        struct.pack_into(
            "<d", changed, source_one_x_offset,
            math.nextafter(first, math.inf),
        )
        rederived = M._derive_acceleration_bound_certificate(
            np, bytes(changed), gm, M.MASSIVE_IDS, block_nodes=2,
        )
        self.assertNotEqual(
            rederived["content_sha256"], certificate["content_sha256"],
        )

    def test_acceleration_bound_routes_outside_envelope_fail_closed(self) -> None:
        far_sources = tuple((float(index), 0.0, 0.0) for index in range(M.MASSIVE_COUNT))
        far = types.SimpleNamespace(
            reason="ACCELERATION_BOUND",
            tracer_position_0_AU=(100.0, 100.0, 100.0),
            source_segment_position_0_AU=far_sources,
        )
        self.assertFalse(M._acceleration_bound_route_is_inside_envelope(far))
        near = types.SimpleNamespace(
            reason="ACCELERATION_BOUND",
            tracer_position_0_AU=(0.5, 0.0, 0.0),
            source_segment_position_0_AU=far_sources,
        )
        self.assertTrue(M._acceleration_bound_route_is_inside_envelope(near))
        other = types.SimpleNamespace(reason="SWEPT_FLOOR")
        self.assertTrue(M._acceleration_bound_route_is_inside_envelope(other))

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

    def test_restart_pair_accepts_exact_500_by_12_projection_and_rejects_stream_tamper(self) -> None:
        payload, request, lane, receipt = _restart_pair_fixture()
        rows = receipt["source_segment_receipts_500_through_999"]
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 500)
        self.assertTrue(all(set(row) == set(M.PUBLIC_SEGMENT_RECEIPT_KEYS) for row in rows))
        M._require_restart_pair(payload, payload, receipt, lane, request)
        forged = bytearray(payload)
        forged[-1] = 1
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, bytes(forged), receipt, lane, request)
        bad = dict(receipt)
        bad["final_global_step"] = -999
        _reseal_restart_receipt(bad)
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, payload, bad, lane, request)

    def test_restart_receipt_rows_reject_missing_extra_and_relabelled_keys(self) -> None:
        for mutation in ("missing", "extra", "relabelled"):
            payload, request, lane, receipt = _restart_pair_fixture()
            rows = copy.deepcopy(receipt["source_segment_receipts_500_through_999"])
            first = rows[0]
            if mutation == "missing":
                del first["public_input_state_sha256"]
            elif mutation == "extra":
                first["unfrozen_extra"] = "forbidden"
            else:
                first["public_input_state_digest"] = first.pop(
                    "public_input_state_sha256"
                )
            receipt["source_segment_receipts_500_through_999"] = rows
            receipt["source_segment_receipts_sha256"] = M._sha256(
                M._canonical_json(rows)
            )
            _reseal_restart_receipt(receipt)
            with self.subTest(mutation=mutation):
                with self.assertRaises(M.Stage15QualificationError):
                    M._require_restart_pair(payload, payload, receipt, lane, request)

    def test_restart_pair_rejects_monolithic_public_projection_mismatch(self) -> None:
        payload, request, lane, receipt = _restart_pair_fixture()
        lane = copy.deepcopy(lane)
        row = lane["source_nodes"]["annual_segment_receipts"][777]
        row["public_input_state_sha256"] = "f" * 64
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, payload, receipt, lane, request)

    def test_restart_pair_recomputes_source_segment_receipt_digest(self) -> None:
        payload, request, lane, receipt = _restart_pair_fixture()
        receipt["source_segment_receipts_sha256"] = "0" * 64
        _reseal_restart_receipt(receipt)
        with self.assertRaises(M.Stage15QualificationError):
            M._require_restart_pair(payload, payload, receipt, lane, request)

    def test_five_prepass_storage_fields_remain_in_held_fd_validation(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_segment_receipts"
        )
        rendered = ast.unparse(function)
        self.assertIn("MERGED_SEGMENT_RECEIPT_KEYS", rendered)
        for key in M.PREPASS_STORAGE_ONLY_RECEIPT_KEYS:
            self.assertIn(key, rendered)
        self.assertIn("_fd_sha256", rendered)
        self.assertIn("_pread_all", rendered)

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

    def test_diagnostic_inventory_accepts_nonmodule_and_fileless_prefix_entries(self) -> None:
        sentinel = object()
        cases = (
            ("cupy.benign_lazy_placeholder", object()),
            (
                "jxplanetx.benign_fileless",
                types.ModuleType("jxplanetx.benign_fileless"),
            ),
        )
        for name, value in cases:
            prior = sys.modules.get(name, sentinel)
            sys.modules[name] = value  # type: ignore[assignment]
            try:
                with self.subTest(name=name):
                    inventory = M._runtime_prefix_inventory()
                    row = next(
                        item for item in inventory
                        if item["registry_name"] == name
                    )
                    self.assertTrue(row["diagnostic_only"])
                    expected = (
                        "EXACT_MODULETYPE"
                        if type(value) is types.ModuleType else "NON_MODULE"
                    )
                    self.assertEqual(row["value_kind"], expected)
            finally:
                if prior is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior  # type: ignore[assignment]

    def test_exact_venv_fresh_child_imports_supported_nvrtc_without_device_query(self) -> None:
        child_source = r'''
from pathlib import Path
import importlib
import json
import sys
import types

source = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
payload = source.read_bytes()
module = types.ModuleType("_jx_stage15_v8_import_only_child")
module.__file__ = str(source)
module.__package__ = ""
sys.modules[module.__name__] = module
exec(compile(payload, str(source), "exec", optimize=sys.flags.optimize), module.__dict__)
interpreter = module._fresh_child_interpreter_receipt()
fresh, modules = module._import_runtime_modules_only(cache_root)
nvrtc = module._nvrtc_module_receipt(modules[module.NVRTC_MODULE_NAME])
cuda_runtime = module._cuda_runtime_module_receipt(
    modules[module.CUDA_RUNTIME_MODULE_NAME]
)
backend_alias = module._require_cuda_runtime_alias_invariant(
    modules["cupy"], modules,
)
engine = modules["jxplanetx.cuda_massless_tracer"]
original_runtime_identity = engine.cuda_massless_tracer_runtime_identity
engine.cuda_massless_tracer_runtime_identity = lambda _spec: {
    "setup_only_test_without_backend_or_device_query": True,
}
try:
    runtime_before = module._runtime_receipt(
        cache_root=cache_root,
        fresh_registry=fresh,
        numpy_module=modules["numpy"],
        scipy_module=modules["scipy"],
        cupy_module=modules["cupy"],
        engine_module=engine,
        backend_spec=object(),
    )
    module._validate_runtime_document(runtime_before, runtime_before)
    importlib.import_module("jxplanetx.claims")
    runtime_after_import = module._runtime_receipt(
        cache_root=cache_root,
        fresh_registry=fresh,
        numpy_module=modules["numpy"],
        scipy_module=modules["scipy"],
        cupy_module=modules["cupy"],
        engine_module=engine,
        backend_spec=object(),
    )
    transition = module._validate_runtime_document(
        runtime_before, runtime_after_import,
    )

    forged_critical = json.loads(json.dumps(runtime_after_import))
    forged_critical["critical_modules"][0]["sha256"] = "0" * 64
    forged_critical["content_sha256"] = module._domain_sha(
        module.RUNTIME_DOMAIN, forged_critical,
    )
    critical_drift_rejected = False
    try:
        module._validate_runtime_document(runtime_before, forged_critical)
    except module.Stage15QualificationError:
        critical_drift_rejected = True

    benign = json.loads(json.dumps(runtime_after_import))
    extra = module._runtime_prefix_inventory_row(
        "jxplanetx.synthetic_additional_import",
        types.ModuleType("jxplanetx.synthetic_additional_import"),
    )
    benign["runtime_prefix_inventory"].append(extra)
    benign["runtime_prefix_inventory"].sort(
        key=lambda row: row["registry_name"],
    )
    benign["content_sha256"] = module._domain_sha(module.RUNTIME_DOMAIN, benign)
    module._validate_runtime_document(runtime_before, benign)

    lazy_before = [module._runtime_prefix_inventory_row(
        "cupyx.synthetic_lazy", object(),
    )]
    lazy_after = [module._runtime_prefix_inventory_row(
        "cupyx.synthetic_lazy", types.ModuleType("cupyx.synthetic_lazy"),
    )]
    lazy_transition = module._runtime_prefix_inventory_transition(
        lazy_before, lazy_after,
    )
finally:
    engine.cuda_massless_tracer_runtime_identity = original_runtime_identity
inventory = runtime_before["runtime_prefix_inventory"]
record = {
    "interpreter": interpreter,
    "fresh": fresh,
    "module_names": list(modules),
    "nvrtc": nvrtc,
    "nvrtc_api": [
        name for name in module.NVRTC_REQUIRED_API
        if callable(getattr(modules[module.NVRTC_MODULE_NAME], name, None))
    ],
    "cuda_runtime": cuda_runtime,
    "cuda_runtime_alias_name": backend_alias.__name__,
    "cuda_runtime_shim_preloaded": "cupy.cuda.runtime" in sys.modules,
    "cuda_runtime_api": [
        name for name in module.CUDA_RUNTIME_REQUIRED_API
        if callable(getattr(backend_alias, name, None))
    ],
    "declared_name_aliases": [
        row for row in inventory
        if row["declared_name"] is not None
        and row["registry_name"] != row["declared_name"]
    ],
    "specless_file_backed_modules": [
        row for row in inventory
        if row["registry_name"] in module.SPECLESS_FILE_BACKED_MODULES
    ],
    "full_runtime_receipt_validated_without_backend_or_device_query": True,
    "post_import_inventory_transition": transition,
    "critical_drift_rejected": critical_drift_rejected,
    "benign_additional_import_accepted": True,
    "lazy_materialization_transition": lazy_transition,
    "trust_boundary": runtime_before["trust_boundary"],
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
        self.assertNotIn("cupy.cuda.runtime", production_text)
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
        self.assertNotIn("cupy.cuda.runtime", record["module_names"])
        self.assertEqual(record["nvrtc"]["module"], M.NVRTC_MODULE_NAME)
        self.assertEqual(record["nvrtc"]["bytes"], M.NVRTC_MODULE_BYTES)
        self.assertEqual(record["nvrtc"]["sha256"], M.NVRTC_MODULE_SHA256)
        self.assertEqual(record["nvrtc_api"], list(M.NVRTC_REQUIRED_API))
        self.assertEqual(
            record["cuda_runtime"]["module"], M.CUDA_RUNTIME_MODULE_NAME,
        )
        self.assertEqual(
            record["cuda_runtime"]["bytes"], M.CUDA_RUNTIME_MODULE_BYTES,
        )
        self.assertEqual(
            record["cuda_runtime"]["sha256"], M.CUDA_RUNTIME_MODULE_SHA256,
        )
        self.assertEqual(
            record["cuda_runtime_alias_name"], M.CUDA_RUNTIME_MODULE_NAME,
        )
        self.assertFalse(record["cuda_runtime_shim_preloaded"])
        self.assertEqual(
            record["cuda_runtime_api"], list(M.CUDA_RUNTIME_REQUIRED_API),
        )
        self.assertEqual(len(record["declared_name_aliases"]), 1)
        alias = record["declared_name_aliases"][0]
        self.assertEqual(alias["registry_name"], M.SCIPY_UARRAY_REGISTRY_NAME)
        self.assertEqual(
            alias["declared_name"], M.SCIPY_UARRAY_DECLARED_NAME,
        )
        self.assertEqual(alias["spec_name"], M.SCIPY_UARRAY_REGISTRY_NAME)
        self.assertEqual(alias["loader_type_qualname"], "ExtensionFileLoader")
        self.assertEqual(
            alias["file_observation"]["file_bytes"], M.SCIPY_UARRAY_BYTES,
        )
        self.assertEqual(
            alias["file_observation"]["file_sha256"], M.SCIPY_UARRAY_SHA256,
        )
        self.assertEqual(
            {
                row["registry_name"]
                for row in record["specless_file_backed_modules"]
            },
            set(M.SPECLESS_FILE_BACKED_MODULES),
        )
        self.assertTrue(
            record["full_runtime_receipt_validated_without_backend_or_device_query"],
        )
        transition = record["post_import_inventory_transition"]
        self.assertIn("jxplanetx.claims", transition["added_names"])
        self.assertGreaterEqual(
            transition["after_count"], transition["before_count"],
        )
        self.assertTrue(record["critical_drift_rejected"])
        self.assertTrue(record["benign_additional_import_accepted"])
        self.assertEqual(
            record["lazy_materialization_transition"]
            ["lazy_placeholder_to_module_names"],
            ["cupyx.synthetic_lazy"],
        )
        self.assertEqual(record["trust_boundary"], M.RUNTIME_TRUST_BOUNDARY)
        self.assertEqual(
            record["interpreter"]["sys_executable"],
            str(M._lexical_absolute(M.PROJECT_VENV_PYTHON)),
        )
        self.assertEqual(record["fresh"]["preloaded_module_names"], [])

    def test_exact_scipy_vendored_alias_and_specless_modules_are_fail_closed(self) -> None:
        alias_path = M._lexical_absolute(
            M.PROJECT_SITE_PACKAGES / M.SCIPY_UARRAY_RELATIVE_PATH,
        )
        alias = types.ModuleType(M.SCIPY_UARRAY_DECLARED_NAME)
        alias.__file__ = str(alias_path)
        alias.__spec__ = importlib.machinery.ModuleSpec(
            M.SCIPY_UARRAY_REGISTRY_NAME,
            importlib.machinery.ExtensionFileLoader(
                M.SCIPY_UARRAY_REGISTRY_NAME, str(alias_path),
            ),
            origin=str(alias_path),
        )
        receipt = M._module_receipt(alias, M.SCIPY_UARRAY_REGISTRY_NAME)
        self.assertEqual(receipt["module"], M.SCIPY_UARRAY_REGISTRY_NAME)
        self.assertEqual(
            receipt["declared_module_name"], M.SCIPY_UARRAY_DECLARED_NAME,
        )
        self.assertEqual(receipt["spec_name"], M.SCIPY_UARRAY_REGISTRY_NAME)
        self.assertEqual(receipt["loader_type"], "ExtensionFileLoader")
        self.assertEqual(receipt["bytes"], M.SCIPY_UARRAY_BYTES)
        self.assertEqual(receipt["sha256"], M.SCIPY_UARRAY_SHA256)

        wrong_declared = types.ModuleType(M.SCIPY_UARRAY_REGISTRY_NAME)
        wrong_declared.__file__ = str(alias_path)
        wrong_declared.__spec__ = alias.__spec__
        with self.assertRaises(M.Stage15QualificationError):
            M._module_receipt(wrong_declared, M.SCIPY_UARRAY_REGISTRY_NAME)

        wrong_spec = types.ModuleType(M.SCIPY_UARRAY_DECLARED_NAME)
        wrong_spec.__file__ = str(alias_path)
        wrong_spec.__spec__ = importlib.machinery.ModuleSpec(
            M.SCIPY_UARRAY_DECLARED_NAME,
            importlib.machinery.ExtensionFileLoader(
                M.SCIPY_UARRAY_DECLARED_NAME, str(alias_path),
            ),
            origin=str(alias_path),
        )
        with self.assertRaises(M.Stage15QualificationError):
            M._module_receipt(wrong_spec, M.SCIPY_UARRAY_REGISTRY_NAME)

        arbitrary = types.ModuleType("vendored.alias")
        arbitrary.__file__ = str(alias_path)
        arbitrary.__spec__ = importlib.machinery.ModuleSpec(
            "scipy.arbitrary", object(), origin=str(alias_path),
        )
        with self.assertRaises(M.Stage15QualificationError):
            M._module_receipt(arbitrary, "scipy.arbitrary")

        highs_path = M._lexical_absolute(
            M.PROJECT_SITE_PACKAGES / M.SCIPY_HIGHS_CORE_RELATIVE_PATH,
        )
        for registry_name in M.SPECLESS_FILE_BACKED_MODULES:
            with self.subTest(registry_name=registry_name):
                synthetic = types.ModuleType(registry_name)
                synthetic.__file__ = str(highs_path)
                synthetic.__spec__ = None
                synthetic_receipt = M._module_receipt(synthetic, registry_name)
                self.assertEqual(synthetic_receipt["module"], registry_name)
                self.assertEqual(
                    synthetic_receipt["declared_module_name"], registry_name,
                )
                self.assertIsNone(synthetic_receipt["spec_name"])
                self.assertIsNone(synthetic_receipt["spec_origin"])
                self.assertIsNone(synthetic_receipt["loader_type"])
                self.assertEqual(
                    synthetic_receipt["sha256"], M.SCIPY_HIGHS_CORE_SHA256,
                )

        unknown_specless = types.ModuleType("scipy.unknown_specless")
        unknown_specless.__file__ = str(highs_path)
        unknown_specless.__spec__ = None
        with self.assertRaises(M.Stage15QualificationError):
            M._module_receipt(unknown_specless, unknown_specless.__name__)

    def test_cuda_runtime_alias_invariant_rejects_python_shim_shadowing(self) -> None:
        backend = types.ModuleType(M.CUDA_RUNTIME_MODULE_NAME)
        for name in M.CUDA_RUNTIME_REQUIRED_API:
            setattr(backend, name, lambda: None)
        cupy = types.ModuleType("cupy")
        cupy.cuda = types.SimpleNamespace(runtime=backend)
        sentinel = object()
        prior = sys.modules.get("cupy.cuda.runtime", sentinel)
        sys.modules.pop("cupy.cuda.runtime", None)
        try:
            self.assertIs(
                M._require_cuda_runtime_alias_invariant(
                    cupy, {M.CUDA_RUNTIME_MODULE_NAME: backend},
                ),
                backend,
            )
            shim = types.ModuleType("cupy.cuda.runtime")
            cupy.cuda.runtime = shim
            sys.modules["cupy.cuda.runtime"] = shim
            with self.assertRaises(M.Stage15QualificationError):
                M._require_cuda_runtime_alias_invariant(
                    cupy, {M.CUDA_RUNTIME_MODULE_NAME: backend},
                )
        finally:
            if prior is sentinel:
                sys.modules.pop("cupy.cuda.runtime", None)
            else:
                sys.modules["cupy.cuda.runtime"] = prior  # type: ignore[assignment]

    def test_runtime_projection_tamper_fails_before_evidence_replay(self) -> None:
        projected = M._runtime_process_independent_projection(
            {
                "schema": M.RUNTIME_SCHEMA, "distributions": [],
                "critical_modules": [],
                "runtime_prefix_inventory": [{"registry_name": "diagnostic"}],
                "adapter_callables": [], "engine_runtime_identity": {},
                "implementation": {},
                "trust_boundary": M.RUNTIME_TRUST_BOUNDARY,
                "claim": M.RUNTIME_CUSTODY_CLAIM,
            }
        )
        self.assertNotIn("runtime_prefix_inventory", projected)
        self.assertEqual(projected["trust_boundary"], M.RUNTIME_TRUST_BOUNDARY)
        self.assertEqual(projected["claim"], M.RUNTIME_CUSTODY_CLAIM)
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

    def test_prefix_inventory_transition_is_diagnostic_not_a_gate(self) -> None:
        lazy_name = "cupyx.synthetic_lazy"
        before = [M._runtime_prefix_inventory_row(lazy_name, object())]
        after = [
            M._runtime_prefix_inventory_row(
                lazy_name, types.ModuleType(lazy_name),
            ),
            M._runtime_prefix_inventory_row(
                "jxplanetx.synthetic_additional",
                types.ModuleType("jxplanetx.synthetic_additional"),
            ),
        ]
        after.sort(key=lambda row: row["registry_name"])
        transition = M._runtime_prefix_inventory_transition(before, after)
        self.assertEqual(transition["policy"], "DIAGNOSTIC_ONLY_NO_QUALIFICATION_GATE")
        self.assertEqual(
            transition["lazy_placeholder_to_module_names"], [lazy_name],
        )
        self.assertIn("jxplanetx.synthetic_additional", transition["added_names"])
        self.assertIn(lazy_name, transition["changed_names"])
        self.assertFalse(transition["qualification_gate"])

        malformed = [dict(before[0])]
        malformed[0]["registry_name"] = "outside_runtime_prefix"
        with self.assertRaises(M.Stage15QualificationError):
            M._runtime_prefix_inventory_transition(malformed, after)

    def test_final_runtime_and_restart_paths_retain_both_inventories(self) -> None:
        before_inventory = [
            M._runtime_prefix_inventory_row("cupyx.synthetic_lazy", object()),
        ]
        after_inventory = [
            M._runtime_prefix_inventory_row(
                "cupyx.synthetic_lazy", types.ModuleType("cupyx.synthetic_lazy"),
            ),
        ]
        before = {
            "runtime_prefix_inventory": before_inventory,
            "retained_post_execution_runtime_prefix_inventory": None,
            "retained_runtime_prefix_inventory_transition": None,
            "content_sha256": "0" * 64,
        }
        after = {
            "runtime_prefix_inventory": after_inventory,
            "retained_post_execution_runtime_prefix_inventory": None,
            "retained_runtime_prefix_inventory_transition": None,
            "content_sha256": "1" * 64,
        }
        with mock.patch.object(M, "_validate_runtime_document"):
            finalized = M._finalize_runtime_receipt(before, after)
        self.assertEqual(
            finalized["retained_post_execution_runtime_prefix_inventory"],
            after_inventory,
        )
        self.assertEqual(
            finalized["retained_runtime_prefix_inventory_transition"]
            ["lazy_placeholder_to_module_names"],
            ["cupyx.synthetic_lazy"],
        )
        self.assertEqual(
            finalized["content_sha256"],
            M._domain_sha(M.RUNTIME_DOMAIN, finalized),
        )

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: ast.unparse(node) for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        execute_worker = functions["_execute_worker"]
        restart_worker = functions["_restart_worker"]
        self.assertIn("final_runtime = _finalize_runtime_receipt", execute_worker)
        self.assertIn("_canonical_json(final_runtime)", execute_worker)
        self.assertIn("runtime_prefix_inventory_before", restart_worker)
        self.assertIn("runtime_prefix_inventory_after", restart_worker)
        self.assertIn("runtime_prefix_inventory_transition", restart_worker)

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
            "guard", "distributions", "critical_modules",
            "runtime_prefix_inventory",
            "retained_post_execution_runtime_prefix_inventory",
            "retained_runtime_prefix_inventory_transition",
            "adapter_callables", "engine_runtime_identity", "implementation",
            "trust_boundary", "claim", "content_sha256",
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
                    acceleration_bounds=[1_000.0] * M.MASSIVE_COUNT,
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
                        acceleration_bounds=[1_000.0] * M.MASSIVE_COUNT,
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
            source.count("allow_post_execution_module_expansion=True"), 4,
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

    def test_full_runtime_preflight_is_fresh_setup_only_and_has_no_dynamics_path(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        worker = ast.unparse(functions["_runtime_preflight_worker"])
        outer = ast.unparse(functions["runtime_preflight"])
        self.assertIn("_fresh_runtime_imports(cache_root)", worker)
        self.assertIn("_runtime_receipt", worker)
        self.assertIn("_build_runtime_preflight_document", worker)
        self.assertIn("str(PROJECT_VENV_PYTHON)", outer)
        self.assertIn("'runtime-preflight-worker'", outer)
        for forbidden in (
            "_run_lane", "_advance_one_annual_segment", "_build_source_path",
            "advance_cuda_massless_tracers", "_execute_worker",
        ):
            self.assertNotIn(forbidden, worker)
            self.assertNotIn(forbidden, outer)
        runtime = {
            "schema": M.RUNTIME_SCHEMA,
            "distributions": [], "critical_modules": [],
            "runtime_prefix_inventory": [],
            "retained_post_execution_runtime_prefix_inventory": None,
            "retained_runtime_prefix_inventory_transition": None,
            "adapter_callables": [],
            "engine_runtime_identity": {}, "implementation": {},
            "trust_boundary": M.RUNTIME_TRUST_BOUNDARY,
            "claim": M.RUNTIME_CUSTODY_CLAIM,
        }
        inventory_transition = M._runtime_prefix_inventory_transition([], [])
        with mock.patch.object(
            M, "_validate_runtime_document", return_value=inventory_transition,
        ) as validate:
            document = M._build_runtime_preflight_document(runtime, runtime)
            M._validate_runtime_preflight_document(document)
            self.assertEqual(validate.call_count, 2)
            forged = dict(document)
            forged["dynamics_executed"] = True
            forged["content_sha256"] = M._domain_sha(
                M.RUNTIME_PREFLIGHT_DOMAIN, forged,
            )
            with self.assertRaises(M.Stage15QualificationError):
                M._validate_runtime_preflight_document(forged)
        parser = M._build_parser()
        self.assertEqual(
            parser.parse_args(["runtime-preflight"]).command,
            "runtime-preflight",
        )

    def test_real_input_continuation_smoke_is_two_lane_fail_closed_and_nonqualifying(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        worker = ast.unparse(functions["_continuation_smoke_worker"])
        lane_runner = ast.unparse(functions["_run_v8_smoke_lane"])
        outer = ast.unparse(functions["continuation_smoke"])
        self.assertIn("for lane_id, steps in LANES", worker)
        self.assertIn("engine._build_source_path", lane_runner)
        self.assertIn("_advance_one_annual_segment", lane_runner)
        self.assertIn("engine._validate_continuation", lane_runner)
        self.assertIn("_validate_continuation_smoke_document", worker)
        self.assertNotIn("_run_lane", worker)
        self.assertNotIn("range(HORIZON_YEARS)", worker)
        self.assertIn("str(PROJECT_VENV_PYTHON)", outer)
        self.assertIn("'continuation-smoke-worker'", outer)
        self.assertNotIn("DEFAULT_OUTPUT_ROOT", worker)
        self.assertNotIn("DEFAULT_OUTPUT_ROOT", outer)

        import numpy as np

        runtime = {
            "schema": M.RUNTIME_SCHEMA,
            "distributions": [], "critical_modules": [],
            "runtime_prefix_inventory": [],
            "retained_post_execution_runtime_prefix_inventory": None,
            "retained_runtime_prefix_inventory_transition": None,
            "adapter_callables": [],
            "engine_runtime_identity": {}, "implementation": {},
            "trust_boundary": M.RUNTIME_TRUST_BOUNDARY,
            "claim": M.RUNTIME_CUSTODY_CLAIM,
        }
        gm = [1.0e-6 * (index + 1) for index in range(M.MASSIVE_COUNT)]
        massive = tuple(
            types.SimpleNamespace(
                body_id=M.MASSIVE_IDS[index],
                gm_AU3_per_julian_year2=gm[index],
            )
            for index in range(M.MASSIVE_COUNT)
        )
        batch = types.SimpleNamespace(
            batch_order=0, object_id=M.OBJECT_IDS[0],
            epoch_jd_tdb=float.fromhex(M.SMOKE_NATIVE_EPOCH_JD_TDB_HEX),
            massive=massive,
        )

        def lane_record(lane_id: str, steps: int) -> dict[str, object]:
            rows = np.zeros((steps + 1, 2, M.MASSIVE_COUNT, 3), dtype="<f8")
            for source_index in range(M.MASSIVE_COUNT):
                rows[:, 0, source_index, 0] = 0.25 * source_index
                rows[:, 0, source_index, 1] = 0.01 * source_index
            payload = rows.tobytes(order="C")
            certificate = M._derive_acceleration_bound_certificate(
                np, payload, gm, M.MASSIVE_IDS,
            )
            endpoint_position = rows[-1, 0].reshape(-1)
            endpoint_velocity = rows[-1, 1].reshape(-1)
            position_bytes = rows[-1, 0].tobytes(order="C")
            velocity_bytes = rows[-1, 1].tobytes(order="C")
            return {
                "lane_id": lane_id, "steps_per_julian_year": steps,
                "source_node_count": steps + 1,
                "source_payload_bytes": len(payload),
                "source_payload_sha256": M._sha256(payload),
                "source_payload_hex": payload.hex(),
                "acceleration_bound_certificate": certificate,
                "segment_receipt": {
                    "year_start": 0, "year_end": 1,
                    "starting_global_step": 0, "ending_global_step": steps,
                    "prepass_path_node_count": steps + 1,
                    "prepass_full_payload_sha256": M._sha256(payload),
                    "precomputed_endpoint_position_sha256": M._sha256(position_bytes),
                    "public_endpoint_position_sha256": M._sha256(position_bytes),
                    "precomputed_endpoint_velocity_sha256": M._sha256(velocity_bytes),
                    "public_endpoint_velocity_sha256": M._sha256(velocity_bytes),
                    "public_input_state_sha256": "4" * 64,
                    "bitwise_source_endpoint_match": True,
                },
                "source_endpoint_position_hex": [float(value).hex() for value in endpoint_position],
                "source_endpoint_velocity_hex": [float(value).hex() for value in endpoint_velocity],
                "tracer_endpoint_position_hex": [(50.0 + index).hex() for index in range(M.TRACER_COUNT * 3)],
                "tracer_endpoint_velocity_hex": [(0.01 * index).hex() for index in range(M.TRACER_COUNT * 3)],
                "minimum_source_distance_AU_hex": [(5.0).hex()] * M.TRACER_COUNT,
                "tracer_lifecycle": [0] * M.TRACER_COUNT,
                "active_tracer_count": M.TRACER_COUNT,
                "route_count": 0, "removal_count": 0,
                "final_global_step": steps, "final_state_sha256": "5" * 64,
                "next_continuation_validation_completed": True,
                "public_prepass_endpoint_bitwise_equal": True,
                "bound_exact_offline_rederive_required": True,
                "all_tracers_finite": True,
                "minimum_source_distance_strictly_greater_than_one_AU": True,
                "result_runtime_identity_sha256": M._sha256(M._canonical_json({})),
            }

        lanes = [lane_record(lane_id, steps) for lane_id, steps in M.LANES]
        inventory_transition = M._runtime_prefix_inventory_transition([], [])
        fake_batches = (batch,) + tuple(
            types.SimpleNamespace(massive=massive) for _ in range(11)
        )
        with mock.patch.object(M, "_validate_runtime_document", return_value=inventory_transition) as validate, mock.patch.object(
            M, "_load_v4_context", return_value=(None, None, None, fake_batches, None),
        ):
            document = M._build_v8_continuation_smoke_document(
                runtime_before=runtime, runtime_after=runtime,
                batch=batch, lanes=lanes,
            )
            M._validate_continuation_smoke_document(document)
            self.assertEqual(validate.call_count, 2)
            self.assertTrue(all(
                call.kwargs.get("allow_post_execution_module_expansion") is True
                for call in validate.call_args_list
            ))
            self.assertTrue(document["claims"]["partial_dynamics_executed"])
            self.assertEqual(document["claims"]["smoke_segment_count"], 2)
            self.assertFalse(document["claims"]["second_year_executed"])
            self.assertFalse(document["claims"]["qualification_established"])
            self.assertFalse(document["claims"]["science_result"])

            forged = json.loads(json.dumps(document))
            forged["lanes"][0]["minimum_source_distance_AU_hex"][0] = (1.0).hex()
            forged["content_sha256"] = M._domain_sha(
                M.CONTINUATION_SMOKE_DOMAIN, forged,
            )
            with self.assertRaises(M.Stage15QualificationError):
                M._validate_continuation_smoke_document(forged)

            forged = json.loads(json.dumps(document))
            forged["lanes"][1]["route_count"] = 1
            forged["content_sha256"] = M._domain_sha(
                M.CONTINUATION_SMOKE_DOMAIN, forged,
            )
            with self.assertRaises(M.Stage15QualificationError):
                M._validate_continuation_smoke_document(forged)
        parser = M._build_parser()
        self.assertEqual(
            parser.parse_args(["continuation-smoke"]).command,
            "continuation-smoke",
        )

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
