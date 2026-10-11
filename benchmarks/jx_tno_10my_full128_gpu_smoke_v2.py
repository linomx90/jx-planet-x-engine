#!/usr/bin/env python3
"""V2 successor for the bounded full-129 TNO GPU smoke.

Production execution remains delegated to the exact preregistered V1
implementation.  V2 changes only namespace/custody and replaces the former
production-shaped CPU test fixture with an explicitly non-executed fixture.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import types
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v2.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v2.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v2"

V1_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v1.py"
V1_BYTES = 120_259
V1_SHA256 = "a472cdf2d30eec48a6f8081e0da97d1b17d68ba8e0f290b1755b367576bfe910"
V1_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v1.py"
V1_TEST_BYTES = 27_822
V1_TEST_SHA256 = "e13c43a342313beb615dd2acce55c6558a8baba13278a723c90b52c22e183f53"
V1_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v1"
V1_PROTOCOL_BYTES = 18_140
V1_PROTOCOL_SHA256 = "3e94befba00737c262917703d27781af8c53a125a9c68862e765976db7d1a333"
V1_PROTOCOL_CONTENT_SHA256 = "3a25f337829d100080f6fdc64b29f55f738b85c82c92960ee01b7533c707cfa3"
V1_BRANCH = "research/tno-10my-reduced5-execution-v1-preregistered-20261011"
V1_COMMIT = "13c17fab588ec6b073a036d245be4f67c48ec002"
V1_TAG = "tno-10my-reduced5-execution-v1-preregistered-20261011"
V1_TAG_OBJECT = "9ea2950aadd358e2276e774cf39780b3dcfd2a25"

PREREGISTRATION_BRANCH = "research/tno-10my-reduced5-execution-v2-preregistered-20261011"
PREREGISTRATION_TAG = "tno-10my-reduced5-execution-v2-preregistered-20261011"

PROTOCOL_SCHEMA = "jx.tno.10my-full128-gpu-smoke.protocol.v2"
SEGMENT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.segment.v2"
RUNTIME_PRE_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-pre.v2"
RUNTIME_POST_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-post.v2"
RESULT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.result.v2"
GIT_CUSTODY_SCHEMA = "jx.tno.10my-full128-gpu-smoke.git-custody.v2"
PROTOCOL_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.protocol.v2\0"
SEGMENT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.segment.v2\0"
RUNTIME_PRE_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-pre.v2\0"
RUNTIME_POST_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-post.v2\0"
RESULT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.result.v2\0"
GIT_CUSTODY_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.git-custody.v2\0"

FIXTURE_SCHEMA = "jx.tno.10my-full128-gpu-smoke.cpu-scientific-fixture.v2"
FIXTURE_SEGMENT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.cpu-scientific-fixture-segment.v2"
FIXTURE_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.cpu-scientific-fixture.v2\0"
FIXTURE_SEGMENT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.cpu-scientific-fixture-segment.v2\0"
FIXTURE_ROSTER = frozenset({
    "FIXTURE_CHECKPOINTS.bin", "FIXTURE_LIFECYCLE.bin",
    "FIXTURE_CONTINUATIONS.bin", "FIXTURE_SEGMENTS.jsonl",
})
HISTORICAL_V3_ENGINE_PROJECTION_SHA256 = "76a591f4a61ea5c0d23f3d4825277bbdd541bda79826444ad73acff5a5cb4474"
HISTORICAL_V3_RUNTIME_PREFLIGHT_CONTENT_SHA256 = "80743cbaeba28be1b131a9470a779df95131752b14a02cb364f59afc4b8a95c2"
HISTORICAL_V3_RUNTIME_CONTENT_SHA256 = "fbbc50a05f3faa87df98052e42bf51e5c9bd9f0feb11bf5acd2516ef857548b5"


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _load_v1() -> types.ModuleType:
    payload = V1_PATH.read_bytes()
    if len(payload) != V1_BYTES or _sha256(payload) != V1_SHA256:
        raise RuntimeError("frozen V1 source binding differs")
    name = f"_jx_tno_full128_smoke_v2_v1_{os.getpid()}_{id(object())}"
    module = types.ModuleType(name)
    module.__file__ = str(V1_PATH)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(
            compile(payload, str(V1_PATH), "exec", dont_inherit=True, optimize=sys.flags.optimize),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    replacements = {
        "SCRIPT_PATH": SCRIPT_PATH,
        "TEST_PATH": TEST_PATH,
        "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
        "PREREGISTRATION_BRANCH": PREREGISTRATION_BRANCH,
        "PREREGISTRATION_TAG": PREREGISTRATION_TAG,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "SEGMENT_SCHEMA": SEGMENT_SCHEMA,
        "RUNTIME_PRE_SCHEMA": RUNTIME_PRE_SCHEMA,
        "RUNTIME_POST_SCHEMA": RUNTIME_POST_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "GIT_CUSTODY_SCHEMA": GIT_CUSTODY_SCHEMA,
        "PROTOCOL_DOMAIN": PROTOCOL_DOMAIN,
        "SEGMENT_DOMAIN": SEGMENT_DOMAIN,
        "RUNTIME_PRE_DOMAIN": RUNTIME_PRE_DOMAIN,
        "RUNTIME_POST_DOMAIN": RUNTIME_POST_DOMAIN,
        "RESULT_DOMAIN": RESULT_DOMAIN,
        "GIT_CUSTODY_DOMAIN": GIT_CUSTODY_DOMAIN,
    }
    for key, value in replacements.items():
        setattr(module, key, value)
    return module


_IMPL = _load_v1()
Full128SmokeError = _IMPL.Full128SmokeError
_V1_BUILD_PROTOCOL = _IMPL.build_protocol


def _exact_file(path: Path, size: int, digest: str, label: str) -> dict[str, object]:
    payload = _IMPL._read_regular(path, label, size)
    if _sha256(payload) != digest:
        raise Full128SmokeError(f"{label} SHA-256 differs")
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": size,
        "sha256": digest,
    }


def _predecessor_v1() -> dict[str, Any]:
    source = _exact_file(V1_PATH, V1_BYTES, V1_SHA256, "V1 predecessor source")
    test = _exact_file(V1_TEST_PATH, V1_TEST_BYTES, V1_TEST_SHA256, "V1 predecessor test")
    with _IMPL._held_directory(V1_ROOT, "V1 predecessor root", expected_roster={"PROTOCOL.json"}) as root_fd:
        protocol_payload = _IMPL._read_regular_at(
            root_fd, "PROTOCOL.json", "V1 predecessor protocol", V1_PROTOCOL_BYTES,
        )
    protocol = _IMPL._strict_json(protocol_payload)
    if (
        protocol_payload != _canonical_json(protocol)
        or _sha256(protocol_payload) != V1_PROTOCOL_SHA256
        or protocol.get("content_sha256") != V1_PROTOCOL_CONTENT_SHA256
        or protocol.get("schema") != "jx.tno.10my-full128-gpu-smoke.protocol.v1"
        or protocol.get("claims", {}).get("bounded_gpu_smoke_executed") is not False
        or protocol.get("claims", {}).get("gpu_executed") is not False
    ):
        raise Full128SmokeError("V1 predecessor protocol custody differs")
    return {
        "source": source,
        "test": test,
        "protocol": {
            "relative_path": "runs/tno_10my_full128_gpu_smoke_v1/PROTOCOL.json",
            "bytes": V1_PROTOCOL_BYTES,
            "sha256": V1_PROTOCOL_SHA256,
            "content_sha256": V1_PROTOCOL_CONTENT_SHA256,
        },
        "branch": V1_BRANCH,
        "commit": V1_COMMIT,
        "annotated_tag": V1_TAG,
        "tag_object": V1_TAG_OBJECT,
        "root_roster": ["PROTOCOL.json"],
        "disposition": {
            "v1_production_execute_invoked": False,
            "v1_gpu_dynamics_executed": False,
            "v1_result_published": False,
            "post_discovery_v3_runtime_preflight_operator_session_check_run": True,
            "post_discovery_check_source_step_count": 0,
            "post_discovery_check_tracer_count": 0,
            "post_discovery_check_dynamics_executed": False,
            "post_discovery_check_retained": False,
            "post_discovery_check_authenticated": False,
            "failure_scope": "CPU_TEST_FIXTURE_RELOCATION_AFTER_PREREGISTRATION_NOT_PRODUCTION_PHYSICS",
        },
    }


def build_protocol() -> dict[str, Any]:
    value = _V1_BUILD_PROTOCOL()
    value["status"] = "PREPARED_EXACT_BATCH0_FULL129_1KYR_GPU_SMOKE_V2_NOT_EXECUTED"
    value["predecessor_v1"] = _predecessor_v1()
    value["cpu_fixture_contract"] = {
        "schema": FIXTURE_SCHEMA,
        "status": "CPU_SYNTHETIC_FIXTURE_NO_GPU_NO_EXECUTION",
        "accepted_by_production_validator": False,
        "constructs_production_runtime_or_result": False,
        "historical_v3_runtime_used_only_as_exact_data": True,
        "historical_v3_engine_projection_sha256": HISTORICAL_V3_ENGINE_PROJECTION_SHA256,
        "current_gpu_observed": False,
        "production_runtime_equivalence_claim": False,
        "nonholdout_post_year_zero_rows_are_deterministic_nominal_copies_for_chain_testing": True,
    }
    value["content_sha256"] = _IMPL._domain_hash(PROTOCOL_DOMAIN, value)
    return value


_IMPL.build_protocol = build_protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    return _IMPL.validate_protocol(value)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _IMPL.prepare(output_root)


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _IMPL.validate_prepared(output_root)


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _IMPL.validate_output(output_root)


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, externally_preregistered: bool = False,
    prereg_commit: str = "", prereg_tag_object: str = "",
) -> dict[str, Any]:
    return _IMPL.execute(
        output_root, externally_preregistered=externally_preregistered,
        prereg_commit=prereg_commit, prereg_tag_object=prereg_tag_object,
    )


def _inverse_sub(desired: float, center: float) -> float:
    candidate = desired + center
    if candidate - center == desired:
        return candidate
    lower = candidate
    upper = candidate
    for _ in range(8):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
        if lower - center == desired:
            return lower
        if upper - center == desired:
            return upper
    raise Full128SmokeError("could not invert exact fixture checkpoint subtraction")


def _relative_receipt(path: Path, label: str) -> dict[str, object]:
    payload = _IMPL._read_regular(path, label)
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _historical_v3_runtime_data() -> dict[str, Any]:
    pre_payload = _IMPL._read_regular(
        _IMPL.V3_ROOT / "RUNTIME_PREFLIGHT.json", "historical V3 preflight",
    )
    post_payload = _IMPL._read_regular(
        _IMPL.V3_ROOT / "RUNTIME.json", "historical V3 runtime",
    )
    pre = _IMPL._strict_json(pre_payload)
    post = _IMPL._strict_json(post_payload)
    if (
        _sha256(pre_payload) != _IMPL.V3_RUNTIME_PREFLIGHT_SHA256
        or _sha256(post_payload) != _IMPL.V3_RUNTIME_SHA256
        or pre_payload != _canonical_json(pre)
        or post_payload != _canonical_json(post)
        or pre.get("content_sha256") != HISTORICAL_V3_RUNTIME_PREFLIGHT_CONTENT_SHA256
        or post.get("content_sha256") != HISTORICAL_V3_RUNTIME_CONTENT_SHA256
        or _sha256(_canonical_json(post.get("engine_runtime_projection")))
        != HISTORICAL_V3_ENGINE_PROJECTION_SHA256
    ):
        raise Full128SmokeError("historical V3 runtime data differs")
    return {
        "runtime_preflight_file_sha256": _IMPL.V3_RUNTIME_PREFLIGHT_SHA256,
        "runtime_preflight_content_sha256": HISTORICAL_V3_RUNTIME_PREFLIGHT_CONTENT_SHA256,
        "runtime_file_sha256": _IMPL.V3_RUNTIME_SHA256,
        "runtime_content_sha256": HISTORICAL_V3_RUNTIME_CONTENT_SHA256,
        "engine_projection_sha256": HISTORICAL_V3_ENGINE_PROJECTION_SHA256,
        "historical_observation_only": True,
        "current_runtime_observation": False,
    }


def _retained_source_crosscheck(v3: types.ModuleType, evidence: Mapping[str, Any]) -> None:
    retained = v3._load_retained_result()
    _lane, record = v3._retained_lane_records(retained, _IMPL.BATCH_ORDER)
    with v3._held_source_file() as descriptor:
        nodes = tuple(
            _IMPL._source_node_payload(
                v3, descriptor, record, year * _IMPL.STEPS_PER_YEAR,
            )
            for year in _IMPL.CHECKPOINT_YEARS
        )
        segments = tuple(
            v3._retained_source_segment(descriptor, record, index)
            for index in range(_IMPL.SEGMENT_COUNT)
        )
    if (
        nodes != tuple(evidence["node_payloads"])
        or segments != tuple(evidence["segment_payloads"])
        or record.get("lane_payload_sha256") != evidence["lane_payload_sha256"]
        or record.get("full_source_tile_sha256") != evidence["full_source_tile_sha256"]
    ):
        raise Full128SmokeError("retained H64 source differs from CPU reconstruction")


def _fixture_segment(
    index: int, source_segment: bytes, source_start: bytes, source_end: bytes,
    checkpoint: bytes, lifecycle: bytes, continuation_start: bytes,
    continuation_end: bytes,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": FIXTURE_SEGMENT_SCHEMA,
        "segment_index": index,
        "start_year": index * _IMPL.SEGMENT_YEARS,
        "end_year": (index + 1) * _IMPL.SEGMENT_YEARS,
        "source_segment_sha256": _sha256(source_segment),
        "source_start_sha256": _sha256(source_start),
        "source_end_sha256": _sha256(source_end),
        "checkpoint_sha256": _sha256(checkpoint),
        "lifecycle_sha256": _sha256(lifecycle),
        "continuation_start_sha256": _sha256(continuation_start),
        "continuation_end_sha256": _sha256(continuation_end),
        "historical_v3_engine_projection_sha256": HISTORICAL_V3_ENGINE_PROJECTION_SHA256,
        "gpu_executed": False,
        "dynamics_executed": False,
        "content_sha256": "",
    }
    value["content_sha256"] = _IMPL._domain_hash(FIXTURE_SEGMENT_DOMAIN, value)
    return value


def _construct_cpu_fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    with _IMPL._delegates() as (preflight, v3):
        evidence = _IMPL._reconstruct_cpu_source_evidence(preflight, v3)
        _retained_source_crosscheck(v3, evidence)
        prefix = _IMPL._read_regular(
            _IMPL.V3_ROOT / "PREFIX_STATES.bin", "V3 retained prefix", 169_344,
        )
        if _sha256(prefix) != _IMPL.V3_PREFIX_SHA256:
            raise Full128SmokeError("V3 retained prefix differs")
        primary_rows = _IMPL._unpack_state_record(_IMPL._batch0_primary_payload(preflight))
        checkpoints: list[bytes] = []
        continuations: list[bytes] = []
        lifecycle_rows = [bytes(_IMPL.TRACER_COUNT) for _ in _IMPL.CHECKPOINT_YEARS]
        with v3._delegates() as (grid, _h64, _d):
            with grid.FrozenClassifierKernelV1() as kernel:
                for index, node in enumerate(evidence["node_payloads"]):
                    source_positions, source_velocities = _IMPL._source_node_rows(node)
                    if index == 0:
                        tracer_positions = tuple(
                            tuple(row[:3]) for row in primary_rows[_IMPL.MASSIVE_COUNT:]
                        )
                        tracer_velocities = tuple(
                            tuple(row[3:]) for row in primary_rows[_IMPL.MASSIVE_COUNT:]
                        )
                    else:
                        selected_payload = preflight._v3_checkpoint(prefix, 0, index)
                        values = struct.unpack("<84d", selected_payload)
                        rows = tuple(
                            tuple(values[row * 6:(row + 1) * 6]) for row in range(14)
                        )
                        center_p = tuple(-item for item in rows[0][:3])
                        center_v = tuple(-item for item in rows[0][3:])
                        nominal_p = tuple(
                            _inverse_sub(rows[5][axis], center_p[axis]) for axis in range(3)
                        )
                        nominal_v = tuple(
                            _inverse_sub(rows[5][3 + axis], center_v[axis]) for axis in range(3)
                        )
                        p_rows = [nominal_p for _ in range(_IMPL.TRACER_COUNT)]
                        v_rows = [nominal_v for _ in range(_IMPL.TRACER_COUNT)]
                        for selected_order, tracer_index in enumerate(preflight.HOLDOUT_TRACER_INDICES):
                            row = rows[_IMPL.MASSIVE_COUNT + selected_order]
                            p_rows[tracer_index] = tuple(
                                _inverse_sub(row[axis], center_p[axis]) for axis in range(3)
                            )
                            v_rows[tracer_index] = tuple(
                                _inverse_sub(row[3 + axis], center_v[axis]) for axis in range(3)
                            )
                        tracer_positions = tuple(p_rows)
                        tracer_velocities = tuple(v_rows)
                    checkpoint = _IMPL._checkpoint_full129(
                        preflight, grid, kernel, source_positions, source_velocities,
                        tracer_positions, tracer_velocities,
                    )
                    checkpoints.append(checkpoint)
                    continuations.append(
                        _IMPL._pack_continuation(tracer_positions, tracer_velocities)
                    )
        checkpoint_payload = b"".join(checkpoints)
        lifecycle_payload = b"".join(lifecycle_rows)
        continuation_payload = b"".join(continuations)
        segments = tuple(
            _fixture_segment(
                index, evidence["segment_payloads"][index],
                evidence["node_payloads"][index], evidence["node_payloads"][index + 1],
                checkpoints[index + 1], lifecycle_rows[index + 1],
                continuations[index], continuations[index + 1],
            )
            for index in range(_IMPL.SEGMENT_COUNT)
        )
        segments_payload = b"".join(_canonical_json(item) for item in segments)
        payloads = {
            "FIXTURE_CHECKPOINTS.bin": checkpoint_payload,
            "FIXTURE_LIFECYCLE.bin": lifecycle_payload,
            "FIXTURE_CONTINUATIONS.bin": continuation_payload,
            "FIXTURE_SEGMENTS.jsonl": segments_payload,
        }
        selected_sha = [
            _sha256(preflight._selected_holdout_payload(item)) for item in checkpoints
        ]
        if any(
            preflight._selected_holdout_payload(item)
            != preflight._v3_checkpoint(prefix, 0, index)
            for index, item in enumerate(checkpoints)
        ):
            raise Full128SmokeError("CPU fixture V3 projection differs")
    receipts = [
        _relative_receipt(SCRIPT_PATH, "V2 fixture implementation"),
        _relative_receipt(TEST_PATH, "V2 fixture test"),
        _relative_receipt(V1_PATH, "V1 delegated production implementation"),
        _relative_receipt(_IMPL.V3_PATH, "V3 lifecycle implementation"),
    ]
    document: dict[str, Any] = {
        "schema": FIXTURE_SCHEMA,
        "status": "CPU_SYNTHETIC_FIXTURE_NO_GPU_NO_EXECUTION",
        "implementation_receipts": receipts,
        "historical_v3_runtime_data": _historical_v3_runtime_data(),
        "fixture_files": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(payloads.items())
        },
        "checkpoint_years": list(_IMPL.CHECKPOINT_YEARS),
        "checkpoint_sha256": [_sha256(item) for item in checkpoints],
        "lifecycle_sha256": [_sha256(item) for item in lifecycle_rows],
        "continuation_sha256": [_sha256(item) for item in continuations],
        "selected_v3_projection_sha256": selected_sha,
        "source_endpoint_sha256": [_sha256(item) for item in evidence["node_payloads"]],
        "source_segment_sha256": [_sha256(item) for item in evidence["segment_payloads"]],
        "segment_content_sha256": [item["content_sha256"] for item in segments],
        "synthetic_rule": "YEAR0_ALL129_AUTHORITATIVE_THEN_V3_HOLDOUTS_AND_NOMINAL_COPY_NONHOLDOUTS",
        "claims": {
            "bounded_gpu_smoke_executed": False,
            "bounded_gpu_smoke_passed": False,
            "gpu_executed": False,
            "production_runtime_observed": False,
            "production_output_generated": False,
            "ten_myr_execution_authorized": False,
            "classification_generated": False,
            "science_result": False,
        },
        "accepted_by_production_validator": False,
        "content_sha256": "",
    }
    document["content_sha256"] = _IMPL._domain_hash(FIXTURE_DOMAIN, document)
    return document, payloads


def build_cpu_synthetic_fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    return _construct_cpu_fixture()


def validate_cpu_synthetic_fixture(
    document: Mapping[str, Any], payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    if type(document) is not dict or type(payloads) is not dict or set(payloads) != FIXTURE_ROSTER:
        raise Full128SmokeError("CPU fixture shape or roster differs")
    if (
        document.get("schema") != FIXTURE_SCHEMA
        or document.get("status") != "CPU_SYNTHETIC_FIXTURE_NO_GPU_NO_EXECUTION"
        or document.get("accepted_by_production_validator") is not False
        or document.get("content_sha256") != _IMPL._domain_hash(FIXTURE_DOMAIN, document)
    ):
        raise Full128SmokeError("CPU fixture identity differs")
    claims = document.get("claims")
    if type(claims) is not dict or not claims or any(type(item) is not bool or item for item in claims.values()):
        raise Full128SmokeError("CPU fixture claim ceiling differs")
    expected_document, expected_payloads = _construct_cpu_fixture()
    if _canonical_json(document) != _canonical_json(expected_document):
        raise Full128SmokeError("CPU fixture document differs from exact reconstruction")
    if payloads != expected_payloads:
        raise Full128SmokeError("CPU fixture payload differs from exact reconstruction")
    checkpoints = tuple(
        payloads["FIXTURE_CHECKPOINTS.bin"][
            index * _IMPL.STATE_RECORD_BYTES:(index + 1) * _IMPL.STATE_RECORD_BYTES
        ]
        for index in range(_IMPL.CHECKPOINT_COUNT)
    )
    lifecycle = tuple(
        payloads["FIXTURE_LIFECYCLE.bin"][
            index * _IMPL.TRACER_COUNT:(index + 1) * _IMPL.TRACER_COUNT
        ]
        for index in range(_IMPL.CHECKPOINT_COUNT)
    )
    if any(row != bytes(_IMPL.TRACER_COUNT) for row in lifecycle):
        raise Full128SmokeError("CPU fixture lifecycle differs")
    with _IMPL._delegates() as (preflight, v3):
        evidence = _IMPL._reconstruct_cpu_source_evidence(preflight, v3)
        _IMPL._validate_continuation_records(
            preflight, v3, evidence["node_payloads"],
            payloads["FIXTURE_CONTINUATIONS.bin"], checkpoints,
        )
    return dict(document)


# Explicit aliases used by focused tests and reviewers.  Production functions
# remain the exact frozen V1 function objects operating on V2-patched globals.
_publish = _IMPL._publish
_validate_output_snapshot = _IMPL._validate_output_snapshot
_validate_runtime_envelopes = _IMPL._validate_runtime_envelopes
_strict_json = _IMPL._strict_json
_domain_hash = _IMPL._domain_hash
_delegates = _IMPL._delegates


def __getattr__(name: str) -> Any:
    return getattr(_IMPL, name)


def main(argv: Sequence[str] | None = None) -> int:
    return _IMPL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
