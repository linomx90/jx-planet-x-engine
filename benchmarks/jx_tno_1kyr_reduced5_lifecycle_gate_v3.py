#!/usr/bin/env python3
"""Preregister, execute, and replay the bounded 1 kyr reduced-five lifecycle gate V3.

This is the stop/go successor to the prepared 10 Myr grid adapter.  It runs
only the twelve DE441 native-epoch batches for 1,000 Julian years at h=1/64,
in twenty public 50-year lifecycle segments.  Every 50-year endpoint must be
bitwise identical to the retained Stage-1 h64 lane.  It cannot classify TNOs
or authorize the 10 Myr experiment.  V3 preserves both failed predecessors,
keeps V2's exact tuple-row repair, and moves complete built-in-math/libm/runtime
custody ahead of lane zero.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v3.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_1kyr_reduced5_lifecycle_gate_v3.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v3"
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

V2_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v2.py"
V2_SOURCE_BYTES = 115_820
V2_SOURCE_SHA256 = "8a293885dd9e761ea64d9ba7c930baa017f09a7733da0ca91b0fe469e03c6544"
V2_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_1kyr_reduced5_lifecycle_gate_v2.py"
V2_TEST_BYTES = 19_212
V2_TEST_SHA256 = "b0d2cebdf1244fee54391367ffd715d5f4faf369c035a5c395cbf0fc00dcde9c"
V2_ROOT = PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v2"
V2_PROTOCOL_PATH = V2_ROOT / "PROTOCOL.json"
V2_PROTOCOL_BYTES = 6_931
V2_PROTOCOL_SHA256 = "d6191b73ee86f8d4b9af38b6c0364274f4b3d749034c33397bb65447756bf1e5"
V2_PROTOCOL_CONTENT_SHA256 = "1b6175afb86c408a6d0e91c9e65e84881f416bd1b0b569af32db8d4762841585"
V2_BRANCH = "research/tno-1kyr-reduced5-lifecycle-gate-v2-preregistered-20261011"
V2_COMMIT = "33e925b2f90836aa50e81b704125c6eb3d5172ba"
V2_TAG = "tno-1kyr-reduced5-lifecycle-gate-v2-preregistered-20261011"
V2_TAG_OBJECT = "01a9c06d7533cbe6a8f848bc6c732fb78f37e2d0"

V1_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v1.py"
V1_SOURCE_BYTES = 110_841
V1_SOURCE_SHA256 = "c4140ee3a321a5882072550b567788ce52c1a986437fbe930f911dbb17008f85"
V1_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_1kyr_reduced5_lifecycle_gate_v1.py"
V1_TEST_BYTES = 10_964
V1_TEST_SHA256 = "1f9d340a5f16ba5ee2fd1eefc1c95acc06fa0aa68bbd4cb5a0ac684f17f887e2"
V1_ROOT = PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v1"
V1_PROTOCOL_PATH = V1_ROOT / "PROTOCOL.json"
V1_PROTOCOL_BYTES = 5_030
V1_PROTOCOL_SHA256 = "445739239a6e9bba2f532ee5e8a32889f3ea683fd0677cdaae7142ba177d3333"
V1_PROTOCOL_CONTENT_SHA256 = "5f8b49fd938f44936235ef5176781338440ff10f0e81ac2ea75da4e794ca9346"
V1_BRANCH = "research/tno-1kyr-reduced5-lifecycle-gate-v1-preregistered-20261011"
V1_COMMIT = "f597fcdb73f3f418e6e0322b5b69bbd21a4ae98d"
V1_TAG = "tno-1kyr-reduced5-lifecycle-gate-v1-preregistered-20261011"
V1_TAG_OBJECT = "60581b6f9325b672c20da067110bb71ce4a779d9"

GRID_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_grid_adapter_v1.py"
GRID_BYTES = 111_493
GRID_SHA256 = "324498cd36726e461ab6910fdf0fe13a1b024346293558f535bea557cdb61b3a"
GRID_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_reduced5_grid_adapter_v1.py"
GRID_TEST_BYTES = 37_454
GRID_TEST_SHA256 = "909e6f63ff229e8909245001728acd58a8d4bb7df80993a14b578659c6dd96b8"
GRID_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_10my_reduced5_grid_adapter_v1" / "PROTOCOL.json"
GRID_PROTOCOL_BYTES = 11_844
GRID_PROTOCOL_SHA256 = "6b10412f06f3c398646179fd163347ac269c03b08486c867e4da821e082be3f2"
GRID_PROTOCOL_CONTENT_SHA256 = "f13b551ea074ce6233c9f3b8aa5e0b75421aa26639141564312087af331812a9"

H64_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_jx_h64_closure_v2.py"
H64_BYTES = 66_627
H64_SHA256 = "04337f7fa63abbcbafd2dcacf6ad26ea5cd73cbccb8562e2a233b0e2c4aaefbd"
H64_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage1_jx_h64_closure_v2.py"
H64_TEST_BYTES = 23_647
H64_TEST_SHA256 = "8dc67ff8ad87c3a34a333fd3f862852f7cf7dfe6ba20f09601952101c458ca5d"
H64_ROOT = PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v2"
H64_PROTOCOL_PATH = H64_ROOT / "PROTOCOL.json"
H64_PROTOCOL_BYTES = 6_965
H64_PROTOCOL_SHA256 = "f3ce7587145ddbda17442e4a47a97c4973d3d656aee7337aec49771731eaf1a0"
H64_RESULT_PATH = H64_ROOT / "RESULT.json"
H64_RESULT_BYTES = 5_464_509
H64_RESULT_SHA256 = "3dd021735f817d4dbfb1a5c92370058347eff3606b47453137b5a3b44d41d54d"
H64_ARRAYS_PATH = H64_ROOT / "ARRAYS.bin"
H64_ARRAYS_BYTES = 16_144_128
H64_ARRAYS_SHA256 = "9da897bb9a386db8429cd607147f241a3a5a1113f716a2f05d2c2d07c876ad41"
H64_SOURCE_PATH = H64_ROOT / "JX_SOURCE_NODES.bin"
H64_SOURCE_BYTES = 368_645_800
H64_SOURCE_SHA256 = "fd9b51e18aa6bb5fac095799955ecc6b3d6afeb2253b1000e2def41ccf376dbd"

STAGE1_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"
STAGE1_BYTES = 436_125
STAGE1_SHA256 = "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"

PROTOCOL_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.protocol.v3"
RESULT_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.result.v3"
RUNTIME_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.runtime.v3"
SEGMENT_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.segment.v3"
RESTART_REQUEST_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.restart-request.v3"
RESTART_RECEIPT_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.restart-receipt.v3"
RUNTIME_PREFLIGHT_WORKER_SCHEMA = (
    "jx.tno.1kyr-reduced5-lifecycle-gate.runtime-preflight-worker.v3"
)
RUNTIME_PREFLIGHT_SCHEMA = "jx.tno.1kyr-reduced5-lifecycle-gate.runtime-preflight.v3"
PROTOCOL_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.protocol.v3\0"
RESULT_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.result.v3\0"
RUNTIME_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.runtime.v3\0"
SEGMENT_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.segment.v3\0"
RESTART_REQUEST_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.restart-request.v3\0"
RESTART_RECEIPT_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.restart-receipt.v3\0"
RUNTIME_PREFLIGHT_WORKER_DOMAIN = (
    b"jx.tno.1kyr-reduced5-lifecycle-gate.runtime-preflight-worker.v3\0"
)
RUNTIME_PREFLIGHT_DOMAIN = b"jx.tno.1kyr-reduced5-lifecycle-gate.runtime-preflight.v3\0"
RUNTIME_MODE_PREFLIGHT = "SYNTHETIC_PREFLIGHT"
RUNTIME_MODE_PRODUCTION = "PRODUCTION"

PREREGISTRATION_BRANCH = "research/tno-1kyr-reduced5-lifecycle-gate-v3-preregistered-20261011"
PREREGISTRATION_TAG = "tno-1kyr-reduced5-lifecycle-gate-v3-preregistered-20261011"

PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.14")
PINNED_PYTHON_BYTES = 7_477_160
PINNED_PYTHON_SHA256 = "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41"
PINNED_PYTHON_VERSION = "3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]"
PINNED_PYTHON_CACHE_TAG = "cpython-314"
PINNED_LIBM_PATH = Path("/usr/lib/x86_64-linux-gnu/libm.so.6")
PINNED_LIBM_BYTES = 1_198_376
PINNED_LIBM_SHA256 = "1c488db6f4fe8cecb72456bc018de000b7a9b539c2994864b0416b5e8371b3d8"
MATH_CALLABLE_NAMES = (
    "acos", "atan2", "cos", "fmod", "fsum", "gcd", "hypot",
    "isfinite", "ldexp", "nextafter", "sin", "sqrt", "ulp",
)
_BOOTSTRAP_MATH_MODULE = math
_BOOTSTRAP_MATH_CALLABLES = tuple(
    getattr(math, name) for name in MATH_CALLABLE_NAMES
)

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
RESTART_BATCH_ORDER = 9
RESTART_OBJECT_ID = "jpl-20042301"
HORIZON_YEARS = 1_000
STEPS_PER_YEAR = 64
SEGMENT_YEARS = 50
SEGMENT_STEPS = 3_200
SEGMENT_COUNT = 20
CHECKPOINT_YEARS = tuple(range(0, HORIZON_YEARS + 1, SEGMENT_YEARS))
CHECKPOINT_BYTES = 672
PREFIX_BYTES = len(OBJECT_IDS) * len(CHECKPOINT_YEARS) * CHECKPOINT_BYTES
TRACER_INTERNAL_BYTES = 9 * 6 * 8
RESTART_RECORD_COUNT = 11
RESTART_STREAM_BYTES = 2 * RESTART_RECORD_COUNT * TRACER_INTERNAL_BYTES
SOURCE_NODE_COUNT = 64_001
SOURCE_COUNT = 5
SOURCE_LANE_BYTES = SOURCE_NODE_COUNT * SOURCE_COUNT * 3 * 8 * 2
SOURCE_COMMON_GM_BYTES = SOURCE_COUNT * 8
EXPECTED_OUTPUT_ROSTER = frozenset({
    "PROTOCOL.json", "PREFIX_STATES.bin", "RESTART_STREAMS.bin",
    "SEGMENTS.jsonl", "ROUTES.jsonl", "RUNTIME_PREFLIGHT.json",
    "RUNTIME.json", "RESULT.json",
})

CLAIMS = {
    "dynamics_executed": False,
    "one_thousand_year_lifecycle_gate_passed": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "encounter_route_qualified": False,
    "classification_generated": False,
    "resonance_classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class LifecycleGateError(RuntimeError):
    """The bounded lifecycle gate failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise LifecycleGateError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LifecycleGateError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleGateError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise LifecycleGateError("JSON root must be an exact object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _read_regular(path: Path, label: str, expected_bytes: int | None = None) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise LifecycleGateError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or (expected_bytes is not None and metadata.st_size != expected_bytes)
    ):
        raise LifecycleGateError(f"{label} is not the expected single-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if (
            metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
        ) != (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns, before.st_nlink,
        ):
            raise LifecycleGateError(f"{label} pathname changed before open")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns, before.st_nlink,
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns, after.st_nlink,
    ):
        raise LifecycleGateError(f"{label} changed during read")
    return bytes(payload)


@contextmanager
def _held_directory(
    path: Path,
    label: str,
    *,
    expected_roster: set[str] | frozenset[str] | None = None,
) -> Iterator[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        lexical = absolute.lstat()
    except OSError as exc:
        raise LifecycleGateError(f"cannot stat {label}") from exc
    if not stat.S_ISDIR(lexical.st_mode):
        raise LifecycleGateError(f"{label} is not a lexical directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode)
        if identity != (
            lexical.st_dev, lexical.st_ino, lexical.st_mode,
        ):
            raise LifecycleGateError(f"{label} pathname changed before open")
        if expected_roster is not None and set(os.listdir(descriptor)) != set(expected_roster):
            raise LifecycleGateError(f"{label} roster differs")
        yield descriptor
        after = os.fstat(descriptor)
        final_path = absolute.lstat()
        if identity != (
            after.st_dev, after.st_ino, after.st_mode,
        ) or identity != (
            final_path.st_dev, final_path.st_ino, final_path.st_mode,
        ):
            raise LifecycleGateError(f"{label} identity changed during use")
        if expected_roster is not None and set(os.listdir(descriptor)) != set(expected_roster):
            raise LifecycleGateError(f"{label} final roster differs")
    finally:
        os.close(descriptor)


def _read_regular_at(
    directory_fd: int,
    name: str,
    label: str,
    expected_bytes: int | None = None,
) -> bytes:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise LifecycleGateError(f"{label} name differs")
    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or (expected_bytes is not None and metadata.st_size != expected_bytes)
    ):
        raise LifecycleGateError(f"{label} is not the expected single-link regular file")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
    )
    try:
        before = os.fstat(descriptor)
        if (
            metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
        ) != (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns, before.st_nlink,
        ):
            raise LifecycleGateError(f"{label} pathname changed before open")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns, before.st_nlink,
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns, after.st_nlink,
    ):
        raise LifecycleGateError(f"{label} changed during read")
    return bytes(payload)


def _file_record(path: Path, label: str, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    payload = _read_regular(path, label, expected_bytes)
    if _sha256(payload) != expected_sha256:
        raise LifecycleGateError(f"{label} SHA-256 differs")
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": expected_sha256,
    }


def _implementation_record(path: Path, label: str) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size <= 0:
        raise LifecycleGateError(f"{label} is not a nonempty single-link regular file")
    payload = _read_regular(path, label, metadata.st_size)
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _bound_inputs() -> list[dict[str, object]]:
    rows = (
        (V2_SOURCE_PATH, "failed_v2_source", V2_SOURCE_BYTES, V2_SOURCE_SHA256),
        (V2_TEST_PATH, "failed_v2_test", V2_TEST_BYTES, V2_TEST_SHA256),
        (V2_PROTOCOL_PATH, "failed_v2_protocol", V2_PROTOCOL_BYTES, V2_PROTOCOL_SHA256),
        (V1_SOURCE_PATH, "failed_v1_source", V1_SOURCE_BYTES, V1_SOURCE_SHA256),
        (V1_TEST_PATH, "failed_v1_test", V1_TEST_BYTES, V1_TEST_SHA256),
        (V1_PROTOCOL_PATH, "failed_v1_protocol", V1_PROTOCOL_BYTES, V1_PROTOCOL_SHA256),
        (GRID_PATH, "grid_adapter_source", GRID_BYTES, GRID_SHA256),
        (GRID_TEST_PATH, "grid_adapter_test", GRID_TEST_BYTES, GRID_TEST_SHA256),
        (GRID_PROTOCOL_PATH, "grid_adapter_protocol", GRID_PROTOCOL_BYTES, GRID_PROTOCOL_SHA256),
        (H64_PATH, "h64_closure_source", H64_BYTES, H64_SHA256),
        (H64_TEST_PATH, "h64_closure_test", H64_TEST_BYTES, H64_TEST_SHA256),
        (H64_PROTOCOL_PATH, "h64_closure_protocol", H64_PROTOCOL_BYTES, H64_PROTOCOL_SHA256),
        (H64_RESULT_PATH, "h64_closure_result", H64_RESULT_BYTES, H64_RESULT_SHA256),
        (H64_ARRAYS_PATH, "h64_closure_arrays", H64_ARRAYS_BYTES, H64_ARRAYS_SHA256),
        (H64_SOURCE_PATH, "h64_closure_source_nodes", H64_SOURCE_BYTES, H64_SOURCE_SHA256),
        (STAGE1_PATH, "stage1_v4_source", STAGE1_BYTES, STAGE1_SHA256),
    )
    records = [
        _file_record(path, label, size, digest)
        for path, label, size, digest in rows
    ]
    grid_protocol = _strict_json(_read_regular(GRID_PROTOCOL_PATH, "grid protocol", GRID_PROTOCOL_BYTES))
    if grid_protocol.get("content_sha256") != GRID_PROTOCOL_CONTENT_SHA256:
        raise LifecycleGateError("grid protocol content identity differs")
    v1_protocol = _strict_json(
        _read_regular(V1_PROTOCOL_PATH, "failed V1 protocol", V1_PROTOCOL_BYTES)
    )
    if v1_protocol.get("content_sha256") != V1_PROTOCOL_CONTENT_SHA256:
        raise LifecycleGateError("failed V1 protocol custody differs")
    with _held_directory(
        V1_ROOT, "failed V1 output root", expected_roster={"PROTOCOL.json"}
    ):
        pass
    v2_protocol = _strict_json(
        _read_regular(V2_PROTOCOL_PATH, "failed V2 protocol", V2_PROTOCOL_BYTES)
    )
    if v2_protocol.get("content_sha256") != V2_PROTOCOL_CONTENT_SHA256:
        raise LifecycleGateError("failed V2 protocol custody differs")
    with _held_directory(
        V2_ROOT, "failed V2 output root", expected_roster={"PROTOCOL.json"}
    ):
        pass
    return records


def source_locator_v1(batch_order: int) -> dict[str, int]:
    if type(batch_order) is not int or not 0 <= batch_order < len(OBJECT_IDS):
        raise LifecycleGateError("batch_order must be an exact cohort index")
    source_order = 12 + batch_order
    return {
        "source_order": source_order,
        "byte_offset": SOURCE_COMMON_GM_BYTES + source_order * SOURCE_LANE_BYTES,
        "byte_length": SOURCE_LANE_BYTES,
        "positions_block_bytes": SOURCE_LANE_BYTES // 2,
        "velocities_block_bytes": SOURCE_LANE_BYTES // 2,
    }


def build_protocol() -> dict[str, Any]:
    source = _implementation_record(SCRIPT_PATH, "lifecycle_gate_source")
    test = _implementation_record(TEST_PATH, "lifecycle_gate_test")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREREGISTERED_BOUNDED_1KYR_GATE_NOT_YET_EXECUTED",
        "objective": (
            "Run exactly twelve DE441 reduced-five lifecycle prefixes for 1000 "
            "Julian years and stop unless every retained h64 checkpoint matches bitwise."
        ),
        "implementation": {
            **source,
            "test": test,
            "fresh_process_required": True,
            "runtime_trust_boundary": "NO_HOSTILE_CODE_IN_FRESH_CHILD_POINT_OBSERVATION",
        },
        "preregistration": {
            "branch": PREREGISTRATION_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "commit_and_tag_object_recorded_by_result": True,
            "external_remote_verification_required_before_execute": True,
        },
        "predecessor_failure": {
            "version": "V1",
            "branch": V1_BRANCH,
            "commit": V1_COMMIT,
            "annotated_tag": V1_TAG,
            "tag_object": V1_TAG_OBJECT,
            "retained_root_roster": ["PROTOCOL.json"],
            "attempt_count": 1,
            "failure_phase": "BATCH_0_INITIAL_CHECKPOINT_BEFORE_FIRST_LIFECYCLE_ADVANCE",
            "failure_id": "GRID_CHECKPOINT_REQUIRES_EXACT_BUILTIN_TUPLE_ROWS",
            "exception_class": "GridAdapterError",
            "exception_message": "source positions roster differs",
            "failure_observation_custody": "CODEX_SESSION_LOG_RECOVERED_NOT_RETAINED_BY_V1_ROOT",
            "session_timestamp_utc": "2026-10-11T19:39:27.037Z",
            "command": ".venv/bin/python -B benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v1.py execute --remote-verified",
            "exit_code": 1,
            "child_wall_seconds_decimal_from_session_log": "7.898684164",
            "traceback_terminus": "GridAdapterError: source positions roster differs",
            "batch_0_source_path_constructed_and_bitwise_checked": True,
            "batch_0_massive_source_dynamics_executed": True,
            "lifecycle_initialized": True,
            "completed_batch_count": 0,
            "completed_lifecycle_segment_count": 0,
            "tracer_dynamics_advanced": False,
            "result_published": False,
            "scientific_outcomes_generated": False,
            "repair_scope": "EXACT_VECTOR_ROWS_BOUNDARY_ONLY",
        },
        "immediate_predecessor_failure": {
            "version": "V2",
            "branch": V2_BRANCH,
            "commit": V2_COMMIT,
            "annotated_tag": V2_TAG,
            "tag_object": V2_TAG_OBJECT,
            "retained_root_roster": ["PROTOCOL.json"],
            "attempt_count": 1,
            "failure_phase": "POST_MAIN_LANES_RUNTIME_CUSTODY_BEFORE_FRESH_RESTART",
            "failure_id": "BUILTIN_MATH_HAS_NO_FILE_RECEIPT",
            "exception_class": "AttributeError",
            "exception_message": "module 'math' has no attribute '__file__'",
            "failure_observation_custody": "CODEX_SESSION_LOG_AND_CODE_ORDER_NOT_RETAINED_BY_V2_ROOT",
            "session_timestamp_utc": "2026-10-11T19:50:22.364Z",
            "final_output_timestamp_utc": "2026-10-11T19:51:38.425Z",
            "command": ".venv/bin/python -B benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v2.py execute --remote-verified",
            "exit_code": 1,
            "main_batch_count_completed_in_memory": 12,
            "main_lifecycle_segment_count_completed_in_memory": 240,
            "main_prefix_checkpoint_count_completed_in_memory": 252,
            "retained_h64_prefix_crossbind_count_completed_in_memory": 12,
            "batch_9_midpoint_and_monolithic_suffix_captured_in_memory": True,
            "fresh_restart_started": False,
            "runtime_receipt_constructed": False,
            "result_published": False,
            "qualification_result_retained": False,
            "scientific_outcomes_generated": False,
            "repair_scope": "BUILTIN_MATH_LIBM_RUNTIME_CUSTODY_AND_PRELANE_PREFLIGHT_ONLY",
        },
        "bound_inputs": _bound_inputs(),
        "execution": {
            "ephemeris": "DE441",
            "batch_count": 12,
            "object_ids": list(OBJECT_IDS),
            "states_per_batch": 9,
            "horizon_julian_years": HORIZON_YEARS,
            "steps_per_julian_year": STEPS_PER_YEAR,
            "step_julian_years_hex": (1.0 / STEPS_PER_YEAR).hex(),
            "segment_julian_years": SEGMENT_YEARS,
            "segment_steps": SEGMENT_STEPS,
            "segments_per_batch": SEGMENT_COUNT,
            "checkpoint_years": list(CHECKPOINT_YEARS),
            "restart": {
                "batch_order": RESTART_BATCH_ORDER,
                "object_id": RESTART_OBJECT_ID,
                "year": 500,
                "fresh_process": True,
                "bitwise_suffix_required": True,
            },
        },
        "acceptance": {
            "prelane_runtime_preflight": (
                "EXECUTION_WORKER_LAUNCHES_TWO_DISTINCT_PARENT_CHALLENGED_FRESH_"
                "CHILDREN_WITH_COMPLETE_RUNTIME_RECEIPTS_EXPLICIT_SYNTHETIC_ROLE_"
                "AND_IDENTICAL_PROCESS_INDEPENDENT_PROJECTION_BEFORE_LANE_ZERO"
            ),
            "generated_source_nodes_bitwise_equal_retained_h64": True,
            "source_energy_relative_drift_max_hex": float(1.0e-8).hex(),
            "source_angular_momentum_relative_drift_max_hex": float(1.0e-10).hex(),
            "active_tracer_count_each_segment": 9,
            "route_record_count": 0,
            "removal_record_count": 0,
            "minimum_source_distance_strictly_greater_than_AU_hex": float(1.0).hex(),
            "checkpoint_comparison": "BITWISE_EQUAL_AT_0_50_..._1000_AFTER_EXACT_LAYOUT_TRANSLATION",
            "fresh_restart_suffix_comparison": "BITWISE_EQUAL_INTERNAL_TRACER_STATE_500_550_..._1000",
            "any_failure": "NO_GO_AND_NO_RESULT_PUBLICATION",
        },
        "outputs": {
            "roster": sorted(EXPECTED_OUTPUT_ROSTER),
            "prefix_states_bytes": PREFIX_BYTES,
            "restart_streams_bytes": RESTART_STREAM_BYTES,
            "segment_record_count": len(OBJECT_IDS) * SEGMENT_COUNT,
            "routes_jsonl_bytes_on_pass": 0,
            "result_published_last": True,
        },
        "claims": dict(CLAIMS),
        "outcomes_generated": False,
        "ten_myr_execution_authorized": False,
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise LifecycleGateError("protocol must be an exact object")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise LifecycleGateError("protocol differs from exact reconstruction")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise LifecycleGateError("protocol content identity differs")
    return str(value["content_sha256"])


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise LifecycleGateError("create-only write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only_at(directory_fd: int, name: str, payload: bytes) -> None:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise LifecycleGateError("create-only output name differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise LifecycleGateError("create-only write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    protocol = build_protocol()
    parent = root.parent
    if root.name in ("", ".", ".."):
        raise LifecycleGateError("output root name differs")
    with _held_directory(parent, "output parent") as parent_fd:
        try:
            os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LifecycleGateError("output root must be absent for create-only preparation")
        os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        created_path = os.stat(
            root.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(created_path.st_mode):
            raise LifecycleGateError("created output root is not a directory")
        directory_fd = os.open(
            root.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            created = os.fstat(directory_fd)
            linked = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                (created.st_dev, created.st_ino)
                != (created_path.st_dev, created_path.st_ino)
                or (created.st_dev, created.st_ino)
                != (linked.st_dev, linked.st_ino)
            ):
                raise LifecycleGateError("created output root identity differs")
            _write_create_only_at(directory_fd, "PROTOCOL.json", _canonical_json(protocol))
            os.fsync(directory_fd)
            if set(os.listdir(directory_fd)) != {"PROTOCOL.json"}:
                raise LifecycleGateError("prepared root roster differs")
            final = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            if (created.st_dev, created.st_ino) != (final.st_dev, final.st_ino):
                raise LifecycleGateError("prepared output root identity changed")
        finally:
            os.close(directory_fd)
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared root", expected_roster={"PROTOCOL.json"}) as root_fd:
        payload = _read_regular_at(root_fd, "PROTOCOL.json", "prepared protocol")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise LifecycleGateError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def _load_source_module(path: Path, expected_bytes: int, expected_sha256: str, name: str) -> types.ModuleType:
    payload = _read_regular(path, name, expected_bytes)
    if _sha256(payload) != expected_sha256:
        raise LifecycleGateError(f"source binding changed: {path}")
    if name in sys.modules:
        raise LifecycleGateError("private source-only module name is occupied")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(compile(payload, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def _delegates() -> Iterator[tuple[types.ModuleType, types.ModuleType, types.ModuleType]]:
    grid_name = f"_jx_1k_grid_{os.getpid()}_{id(object())}"
    h64_name = f"_jx_1k_h64_{os.getpid()}_{id(object())}"
    grid = _load_source_module(GRID_PATH, GRID_BYTES, GRID_SHA256, grid_name)
    h64 = _load_source_module(H64_PATH, H64_BYTES, H64_SHA256, h64_name)
    try:
        delegate = h64._delegate()
        yield grid, h64, delegate
    finally:
        sys.modules.pop(grid_name, None)
        sys.modules.pop(h64_name, None)


def _exact_claims(value: object, expected: Mapping[str, bool], label: str) -> None:
    if type(value) is not dict or set(value) != set(expected):
        raise LifecycleGateError(f"{label} claim roster differs")
    for key, expected_value in expected.items():
        if type(value.get(key)) is not bool or value[key] is not expected_value:
            raise LifecycleGateError(f"{label} claim differs: {key}")


def _validate_segment_record(value: object, *, expected_order: int) -> dict[str, Any]:
    keys = {
        "schema", "segment_order", "batch_order", "object_id", "segment_index",
        "start_year", "end_year", "start_global_source_step", "end_global_source_step",
        "source_tile_sha256", "generated_source_segment_sha256",
        "retained_source_segment_sha256", "checkpoint_sha256",
        "active_tracer_count", "route_record_count", "removal_record_count",
        "minimum_source_distance_AU_hex", "maximum_kepler_iterations",
        "engine_runtime_identity_sha256", "elapsed_seconds_hex", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise LifecycleGateError("segment receipt shape differs")
    integer_keys = (
        "segment_order", "batch_order", "segment_index", "start_year",
        "end_year", "start_global_source_step", "end_global_source_step",
        "active_tracer_count", "route_record_count", "removal_record_count",
        "maximum_kepler_iterations",
    )
    if any(type(value.get(key)) is not int for key in integer_keys):
        raise LifecycleGateError("segment receipt integer type differs")
    batch_order = value.get("batch_order")
    segment_index = value.get("segment_index")
    if (
        value.get("schema") != SEGMENT_SCHEMA
        or type(value.get("segment_order")) is not int
        or value.get("segment_order") != expected_order
        or type(batch_order) is not int or not 0 <= batch_order < len(OBJECT_IDS)
        or type(segment_index) is not int or not 0 <= segment_index < SEGMENT_COUNT
        or expected_order != batch_order * SEGMENT_COUNT + segment_index
        or value.get("object_id") != OBJECT_IDS[batch_order]
        or value.get("start_year") != segment_index * SEGMENT_YEARS
        or value.get("end_year") != (segment_index + 1) * SEGMENT_YEARS
        or value.get("start_global_source_step") != segment_index * SEGMENT_STEPS
        or value.get("end_global_source_step") != (segment_index + 1) * SEGMENT_STEPS
        or value.get("active_tracer_count") != 9
        or value.get("route_record_count") != 0
        or value.get("removal_record_count") != 0
        or type(value.get("maximum_kepler_iterations")) is not int
        or value.get("maximum_kepler_iterations") < 0
    ):
        raise LifecycleGateError("segment receipt identity or zero-event gate differs")
    for key in (
        "source_tile_sha256", "generated_source_segment_sha256",
        "retained_source_segment_sha256", "checkpoint_sha256",
        "engine_runtime_identity_sha256", "content_sha256",
    ):
        if not _is_sha256(value.get(key)):
            raise LifecycleGateError(f"segment digest differs: {key}")
    if (
        value["generated_source_segment_sha256"]
        != value["retained_source_segment_sha256"]
    ):
        raise LifecycleGateError("generated and retained source segment digests differ")
    for key in ("minimum_source_distance_AU_hex", "elapsed_seconds_hex"):
        raw = value.get(key)
        if type(raw) is not str:
            raise LifecycleGateError(f"segment float differs: {key}")
        try:
            number = float.fromhex(raw)
        except ValueError as exc:
            raise LifecycleGateError(f"segment float differs: {key}") from exc
        if not math.isfinite(number) or number < 0.0:
            raise LifecycleGateError(f"segment float is invalid: {key}")
    if not float.fromhex(value["minimum_source_distance_AU_hex"]) > 1.0:
        raise LifecycleGateError("segment source-distance gate failed")
    if value.get("content_sha256") != _domain_hash(SEGMENT_DOMAIN, value):
        raise LifecycleGateError("segment content identity differs")
    return value


def _load_retained_result() -> dict[str, Any]:
    payload = _read_regular(H64_RESULT_PATH, "retained h64 result", H64_RESULT_BYTES)
    if _sha256(payload) != H64_RESULT_SHA256:
        raise LifecycleGateError("retained h64 result changed")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise LifecycleGateError("retained h64 result is not canonical")
    return value


def _retained_lane_records(result: Mapping[str, Any], batch_order: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(batch_order) is not int or not 0 <= batch_order < len(OBJECT_IDS):
        raise LifecycleGateError("retained lane batch order differs")
    lanes = result.get("batch_lanes")
    sources = result.get("jx_source_nodes")
    if type(lanes) is not list or len(lanes) != 24 or type(sources) is not dict:
        raise LifecycleGateError("retained lane/source roster differs")
    wrapper = lanes[12 + batch_order]
    records = sources.get("records")
    if type(wrapper) is not dict or type(records) is not list or len(records) != 24:
        raise LifecycleGateError("retained lane/source record differs")
    source = records[12 + batch_order]
    lane = wrapper.get("lane")
    locator = source_locator_v1(batch_order)
    expected_source_keys = {
        "source_order", "ephemeris", "batch_order", "object_id", "lane_id",
        "steps_per_year", "node_count", "source_count", "byte_offset",
        "byte_length", "positions_block_bytes", "velocities_block_bytes",
        "lane_payload_sha256", "positions_sha256", "velocities_sha256",
        "contract", "full_source_tile_sha256", "annual_source_slice_sha256",
        "record_content_sha256",
    }
    if (
        type(source) is not dict or set(source) != expected_source_keys
        or type(lane) is not dict
        or wrapper.get("ephemeris") != "DE441"
        or wrapper.get("batch_order") != batch_order
        or wrapper.get("object_id") != OBJECT_IDS[batch_order]
        or lane.get("lane_id") != "jx_reduced5_h_1_64"
        or lane.get("steps_per_year") != STEPS_PER_YEAR
        or source.get("source_order") != locator["source_order"]
        or source.get("byte_offset") != locator["byte_offset"]
        or source.get("byte_length") != locator["byte_length"]
        or source.get("positions_block_bytes") != locator["positions_block_bytes"]
        or source.get("velocities_block_bytes") != locator["velocities_block_bytes"]
        or source.get("ephemeris") != "DE441"
        or source.get("batch_order") != batch_order
        or source.get("object_id") != OBJECT_IDS[batch_order]
        or source.get("lane_id") != "jx_reduced5_h_1_64"
        or source.get("steps_per_year") != STEPS_PER_YEAR
        or source.get("node_count") != SOURCE_NODE_COUNT
        or source.get("source_count") != SOURCE_COUNT
    ):
        raise LifecycleGateError("retained DE441 h64 source identity differs")
    return lane, source


@contextmanager
def _held_source_file() -> Iterator[int]:
    metadata = H64_SOURCE_PATH.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or metadata.st_size != H64_SOURCE_BYTES
    ):
        raise LifecycleGateError("retained source evidence file differs")
    descriptor = os.open(H64_SOURCE_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(4 << 20, before.st_size - offset), offset)
            if not block:
                raise LifecycleGateError("retained source evidence ended early")
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != H64_SOURCE_SHA256:
            raise LifecycleGateError("retained source evidence SHA-256 differs")
        yield descriptor
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns, before.st_nlink,
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns, after.st_nlink,
        ):
            raise LifecycleGateError("retained source evidence changed during use")
    finally:
        os.close(descriptor)


def _pread_exact(descriptor: int, offset: int, length: int, label: str) -> bytes:
    if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
        raise LifecycleGateError(f"{label} locator differs")
    payload = bytearray()
    while len(payload) < length:
        block = os.pread(descriptor, length - len(payload), offset + len(payload))
        if not block:
            raise LifecycleGateError(f"{label} ended early")
        payload.extend(block)
    return bytes(payload)


def _pack_internal_tracer_state(cp: Any, state: object) -> bytes:
    try:
        positions = cp.asnumpy(state.tracer_relative_positions_AU)
        velocities = cp.asnumpy(state.tracer_relative_velocities_AU_per_year)
        indices = tuple(int(item) for item in cp.asnumpy(state.tracer_indices).tolist())
    except Exception as exc:
        raise LifecycleGateError("tracer state could not be copied from the device") from exc
    if positions.shape != (9, 3) or velocities.shape != (9, 3) or indices != tuple(range(9)):
        raise LifecycleGateError("tracer state roster differs")
    values = tuple(float(item) for item in positions.reshape(-1)) + tuple(
        float(item) for item in velocities.reshape(-1)
    )
    if any(not math.isfinite(item) for item in values):
        raise LifecycleGateError("tracer state contains non-finite values")
    payload = struct.pack("<54d", *values)
    if len(payload) != TRACER_INTERNAL_BYTES:
        raise LifecycleGateError("internal tracer state width differs")
    return payload


def _unpack_internal_tracer_state(payload: bytes) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    if type(payload) is not bytes or len(payload) != TRACER_INTERNAL_BYTES:
        raise LifecycleGateError("internal tracer state width differs")
    values = struct.unpack("<54d", payload)
    if any(not math.isfinite(item) for item in values):
        raise LifecycleGateError("internal tracer state contains non-finite values")
    positions = tuple(tuple(values[index * 3 : index * 3 + 3]) for index in range(9))
    start = 27
    velocities = tuple(tuple(values[start + index * 3 : start + index * 3 + 3]) for index in range(9))
    return positions, velocities


def _engine_runtime_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain(identity)
    if type(plain) is not dict or set(plain) != {
        "method_id", "implementation_source_sha256", "hybrid_runtime_identities",
        "terminal_route_runtime_identities", "route_worker_count",
        "source_tile_sha256", "determinism_scope",
    }:
        raise LifecycleGateError("lifecycle runtime identity shape differs")
    hybrid = plain.get("hybrid_runtime_identities")
    terminal = plain.get("terminal_route_runtime_identities")
    if type(hybrid) is not list or len(hybrid) != 1 or terminal != []:
        raise LifecycleGateError("lifecycle runtime route identity roster differs")
    child = dict(hybrid[0])
    if child.pop("source_tile_sha256", None) is None:
        raise LifecycleGateError("hybrid runtime source identity is absent")
    if child.get("route_runtime_identities") != []:
        raise LifecycleGateError("hybrid runtime unexpectedly contains a route identity")
    projection = {
        "method_id": plain["method_id"],
        "implementation_source_sha256": plain["implementation_source_sha256"],
        "route_worker_count": plain["route_worker_count"],
        "determinism_scope": plain["determinism_scope"],
        "hybrid_runtime_identity_without_source_tile": child,
    }
    if not _is_sha256(projection["implementation_source_sha256"]):
        raise LifecycleGateError("lifecycle implementation digest differs")
    return projection


def _expected_route_policy_identity() -> dict[str, Any]:
    return {
        "schema": "jx.restricted-wh.route-policy.v2",
        "maximum_primary_dynamical_phase_hex": "0x1.999999999999ap-5",
        "primary_outer_boundary_AU_hex": "0x1.0000000000000p+12",
        "terminal_source_radii_AU_hex": ["0x0.0p+0"] * SOURCE_COUNT,
        "event_locator": {
            "relative_tolerance_hex": "0x1.19799812dea11p-40",
            "absolute_tolerance_hex": "0x1.6849b86a12b9bp-47",
            "maximum_substep_fraction_hex": "0x1.0000000000000p-3",
            "dense_polynomial_validation_points": 17,
            "event_fraction_tolerance_hex": "0x1.19799812dea11p-40",
            "event_distance_tolerance_AU_hex": "0x1.5fd7fe1796495p-37",
            "outward_radial_velocity_tolerance_AU_per_year_hex": (
                "0x1.c25c268497682p-44"
            ),
            "maximum_accepted_steps": 100_000,
        },
        "content_sha256": (
            "fe4f9210c095b6c7cd929e5768f7c06464544b089a5d1f556cb8f4c1f1fa252a"
        ),
    }


def _runtime_file(path: Path, label: str, maximum: int = 200_000_000) -> dict[str, object]:
    absolute = path.resolve(strict=True)
    metadata = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 0 or metadata.st_size > maximum:
        raise LifecycleGateError(f"{label} runtime file differs")
    payload = _read_regular(absolute, label, metadata.st_size)
    return {"path": str(absolute), "bytes": len(payload), "sha256": _sha256(payload)}


def _libm_receipt() -> dict[str, object]:
    if os.environ.get("LD_PRELOAD") not in (None, "") or os.environ.get(
        "LD_AUDIT"
    ) not in (None, ""):
        raise LifecycleGateError("dynamic-loader interposition environment is not empty")
    expected_file = _runtime_file(
        PINNED_LIBM_PATH, "mapped libm", maximum=10_000_000
    )
    if (
        expected_file["path"] != str(PINNED_LIBM_PATH)
        or expected_file["bytes"] != PINNED_LIBM_BYTES
        or expected_file["sha256"] != PINNED_LIBM_SHA256
    ):
        raise LifecycleGateError("mapped libm file identity differs")
    try:
        mapping_lines = Path("/proc/self/maps").read_text(encoding="ascii").splitlines()
        metadata = PINNED_LIBM_PATH.stat()
    except (OSError, UnicodeError) as exc:
        raise LifecycleGateError("libm process mapping could not be inspected") from exc
    expected_device = f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}"
    mapped = False
    for line in mapping_lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5] != str(PINNED_LIBM_PATH):
            continue
        if fields[3].lower() == expected_device and fields[4] == str(metadata.st_ino):
            mapped = True
            break
    if not mapped:
        raise LifecycleGateError("pinned libm file is not mapped into this process")
    return {
        **expected_file,
        "mapped_device": expected_device,
        "mapped_inode": metadata.st_ino,
        "mapped_into_process": True,
    }


def _builtin_math_receipt() -> dict[str, object]:
    resolved = Path(sys.executable).resolve(strict=True)
    executable = _runtime_file(
        PINNED_PYTHON_RESOLVED, "built-in math interpreter", maximum=20_000_000
    )
    spec = getattr(math, "__spec__", None)
    if (
        resolved != PINNED_PYTHON_RESOLVED
        or executable["bytes"] != PINNED_PYTHON_BYTES
        or executable["sha256"] != PINNED_PYTHON_SHA256
        or sys.version != PINNED_PYTHON_VERSION
        or sys.implementation.cache_tag != PINNED_PYTHON_CACHE_TAG
        or math is not _BOOTSTRAP_MATH_MODULE
        or type(math) is not types.ModuleType
        or sys.modules.get("math") is not math
        or "math" not in sys.builtin_module_names
        or getattr(math, "__file__", None) is not None
        or spec is None
        or spec.name != "math"
        or spec.origin != "built-in"
        or spec.loader is not importlib.machinery.BuiltinImporter
        or spec.has_location is not False
        or getattr(math, "__loader__", None) is not importlib.machinery.BuiltinImporter
    ):
        raise LifecycleGateError("built-in math module identity differs")
    for name, bootstrap_function in zip(
        MATH_CALLABLE_NAMES, _BOOTSTRAP_MATH_CALLABLES, strict=True
    ):
        function = getattr(math, name, None)
        if (
            function is not bootstrap_function
            or type(function) is not types.BuiltinFunctionType
            or function.__module__ != "math"
            or function.__name__ != name
        ):
            raise LifecycleGateError(f"built-in math callable identity differs: {name}")
    if (
        type(math.pi) is not float
        or math.pi.hex() != "0x1.921fb54442d18p+1"
        or type(math.tau) is not float
        or math.tau.hex() != "0x1.921fb54442d18p+2"
        or type(math.inf) is not float
        or math.inf != float("inf")
        or type(math.nan) is not float
        or not math.isnan(math.nan)
        or struct.pack(">d", math.nan).hex() != "7ff8000000000000"
    ):
        raise LifecycleGateError("built-in math constants differ")
    return {
        "module_name": "math",
        "module_kind": "BUILTIN",
        "registry_identity_exact": True,
        "module_file_absent": True,
        "spec_name": "math",
        "spec_origin": "built-in",
        "spec_loader": "BuiltinImporter",
        "spec_has_location": False,
        "builtin_module_name_present": True,
        "callable_names": list(MATH_CALLABLE_NAMES),
        "math_pi_hex": math.pi.hex(),
        "math_tau_hex": math.tau.hex(),
        "math_inf_hex": math.inf.hex(),
        "math_nan_binary64_be_hex": struct.pack(">d", math.nan).hex(),
        "python_resolved_executable": str(resolved),
        "python_executable_bytes": executable["bytes"],
        "python_executable_sha256": executable["sha256"],
        "python_version": sys.version,
        "python_cache_tag": sys.implementation.cache_tag,
    }


def _critical_runtime_base(
    d: types.ModuleType,
    cp: Any,
    *,
    fresh_child: Mapping[str, Any],
    fresh_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate every runtime field available before lane zero."""

    module_names = (
        "jxplanetx.restricted_wh_source",
        "jxplanetx.cuda_restricted_wh",
        "jxplanetx.cuda_restricted_wh_hybrid",
        "jxplanetx.restricted_wh_route",
        "jxplanetx.restricted_wh_lifecycle",
        "jxplanetx.cuda_tracer_events",
        "jxplanetx.fast_wisdom_holman",
        "jxplanetx._all_conic_kepler",
        "jxplanetx._wisdom_holman_cpu_v3",
        "jxplanetx.engine.backends",
        "jxplanetx.engine.contracts",
    )
    modules: list[dict[str, object]] = []
    for name in module_names:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if type(module) is not types.ModuleType or type(path) is not str:
            raise LifecycleGateError(f"critical runtime module is absent: {name}")
        modules.append({"module": name, **_runtime_file(Path(path), name)})
    properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    raw_name = properties["name"]
    device_name = raw_name.decode("ascii") if isinstance(raw_name, bytes) else str(raw_name)
    base: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "optimization_level": sys.flags.optimize,
            "executable": _runtime_file(
                Path(sys.executable), "Python executable", maximum=20_000_000
            ),
        },
        "fresh_child_interpreter": _plain(fresh_child),
        "fresh_process_registry": _plain(fresh_registry),
        "numpy_runtime_identity": _plain(d._numpy_runtime_identity()),
        "cupy_cuda": {
            "cupy_version": str(cp.__version__),
            "cupy_module": _runtime_file(Path(cp.__file__), "CuPy module"),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(cp.cuda.get_local_runtime_version()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_index": int(cp.cuda.Device().id),
            "device_name": device_name,
            "device_uuid": bytes(properties["uuid"]).hex(),
            "compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
            "pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
        },
        "critical_jx_modules": modules,
        "builtin_math": _builtin_math_receipt(),
        "mapped_libm_point_observation": _libm_receipt(),
        "bound_input_records_sha256": _sha256(_canonical_json(_bound_inputs())),
        "claim": (
            "TRUSTED_FRESH_PROCESS_EXACT_BUILTIN_OBJECT_LIVE_FILE_AND_MAPPED_LIBM_"
            "POINT_OBSERVATION_NOT_COMPLETE_RUNTIME_AUTHENTICATION"
        ),
    }
    _validate_runtime_base(base, d, "pre-lane runtime")
    parsed = _strict_json(_canonical_json(base))
    if parsed != base:
        raise LifecycleGateError("pre-lane runtime canonical roundtrip differs")
    return base


def _critical_runtime_identity(
    d: types.ModuleType,
    *,
    runtime_base: Mapping[str, Any],
    import_guard: Mapping[str, Any],
    engine_projection: Mapping[str, Any],
    expected_mode: str,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        **dict(runtime_base),
        "source_only_runtime_import_guard": _plain(import_guard),
        "engine_runtime_projection": _plain(engine_projection),
        "content_sha256": "",
    }
    runtime["content_sha256"] = _domain_hash(RUNTIME_DOMAIN, runtime)
    _validate_runtime_identity(
        runtime, d, "constructed runtime", expected_mode=expected_mode
    )
    return runtime


def _validate_runtime_file_receipt(value: object, label: str) -> None:
    if (
        type(value) is not dict or set(value) != {"path", "bytes", "sha256"}
        or type(value.get("path")) is not str
        or type(value.get("bytes")) is not int
        or value.get("bytes") < 0
        or not _is_sha256(value.get("sha256"))
    ):
        raise LifecycleGateError(f"{label} runtime file receipt shape differs")
    if _runtime_file(Path(value["path"]), label) != value:
        raise LifecycleGateError(f"{label} runtime file changed")


def _validate_builtin_math_receipt(value: object, label: str) -> None:
    expected_keys = {
        "module_name", "module_kind", "registry_identity_exact",
        "module_file_absent", "spec_name", "spec_origin", "spec_loader",
        "spec_has_location", "builtin_module_name_present", "callable_names",
        "math_pi_hex", "math_tau_hex", "math_inf_hex",
        "math_nan_binary64_be_hex", "python_resolved_executable",
        "python_executable_bytes", "python_executable_sha256",
        "python_version", "python_cache_tag",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise LifecycleGateError(f"{label} built-in math receipt shape differs")
    if _canonical_json(value) != _canonical_json(_builtin_math_receipt()):
        raise LifecycleGateError(f"{label} built-in math receipt differs")


def _validate_libm_receipt(value: object, label: str) -> None:
    expected_keys = {
        "path", "bytes", "sha256", "mapped_device", "mapped_inode",
        "mapped_into_process",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or type(value.get("path")) is not str
        or type(value.get("bytes")) is not int
        or type(value.get("sha256")) is not str
        or type(value.get("mapped_device")) is not str
        or type(value.get("mapped_inode")) is not int
        or value.get("mapped_inode", 0) <= 0
        or value.get("mapped_into_process") is not True
    ):
        raise LifecycleGateError(f"{label} mapped libm receipt shape differs")
    if _canonical_json(value) != _canonical_json(_libm_receipt()):
        raise LifecycleGateError(f"{label} mapped libm receipt differs")


def _runtime_base_keys() -> frozenset[str]:
    return frozenset({
        "python", "fresh_child_interpreter", "fresh_process_registry",
        "numpy_runtime_identity", "cupy_cuda", "critical_jx_modules",
        "builtin_math", "mapped_libm_point_observation",
        "bound_input_records_sha256", "claim",
    })


def _validate_runtime_base(
    value: object, d: types.ModuleType, label: str
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != _runtime_base_keys()
        or value.get("bound_input_records_sha256")
        != _sha256(_canonical_json(_bound_inputs()))
        or value.get("claim")
        != (
            "TRUSTED_FRESH_PROCESS_EXACT_BUILTIN_OBJECT_LIVE_FILE_AND_MAPPED_LIBM_"
            "POINT_OBSERVATION_NOT_COMPLETE_RUNTIME_AUTHENTICATION"
        )
    ):
        raise LifecycleGateError(f"{label} runtime base differs")
    try:
        d._validate_fresh_child_interpreter_receipt(value["fresh_child_interpreter"])
        d._validate_numpy_runtime_identity(value["numpy_runtime_identity"])
    except Exception as exc:
        raise LifecycleGateError(f"{label} bound runtime receipt differs") from exc
    registry = value.get("fresh_process_registry")
    if (
        type(registry) is not dict
        or registry.get("preloaded_module_names") != []
        or registry.get("fresh_registry_observed") is not True
        or registry.get("claim")
        != "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION"
    ):
        raise LifecycleGateError(f"{label} fresh-process registry differs")
    python = value.get("python")
    if (
        type(python) is not dict
        or set(python) != {
            "version", "full_version", "cache_tag", "implementation",
            "optimization_level", "executable",
        }
        or python.get("version") != PINNED_PYTHON_VERSION.split(" ", 1)[0]
        or python.get("full_version") != PINNED_PYTHON_VERSION
        or python.get("cache_tag") != PINNED_PYTHON_CACHE_TAG
        or python.get("implementation") != "CPython"
        or type(python.get("optimization_level")) is not int
        or python.get("optimization_level") != 0
    ):
        raise LifecycleGateError(f"{label} Python runtime differs")
    _validate_runtime_file_receipt(python["executable"], f"{label} Python executable")
    if (
        python["executable"]["path"] != str(PINNED_PYTHON_RESOLVED)
        or python["executable"]["bytes"] != PINNED_PYTHON_BYTES
        or python["executable"]["sha256"] != PINNED_PYTHON_SHA256
    ):
        raise LifecycleGateError(f"{label} Python executable identity differs")
    cupy = value.get("cupy_cuda")
    cupy_keys = {
        "cupy_version", "cupy_module", "cuda_runtime_version",
        "cuda_local_runtime_version", "cuda_driver_version", "device_index",
        "device_name", "device_uuid", "compute_capability",
        "pci_domain_bus_device",
    }
    if (
        type(cupy) is not dict
        or set(cupy) != cupy_keys
        or type(cupy.get("cupy_version")) is not str
        or not cupy.get("cupy_version")
        or type(cupy.get("device_name")) is not str
        or not cupy.get("device_name")
        or type(cupy.get("device_uuid")) is not str
        or not cupy.get("device_uuid")
        or type(cupy.get("compute_capability")) is not str
        or not cupy.get("compute_capability")
        or type(cupy.get("pci_domain_bus_device")) is not str
        or not cupy.get("pci_domain_bus_device")
        or any(
            type(cupy.get(key)) is not int or cupy.get(key) <= 0
            for key in (
                "cuda_runtime_version", "cuda_local_runtime_version",
                "cuda_driver_version",
            )
        )
        or type(cupy.get("device_index")) is not int
        or cupy.get("device_index") != 0
    ):
        raise LifecycleGateError(f"{label} CuPy/CUDA runtime differs")
    _validate_runtime_file_receipt(cupy["cupy_module"], f"{label} CuPy module")
    expected_modules = (
        "jxplanetx.restricted_wh_source", "jxplanetx.cuda_restricted_wh",
        "jxplanetx.cuda_restricted_wh_hybrid", "jxplanetx.restricted_wh_route",
        "jxplanetx.restricted_wh_lifecycle", "jxplanetx.cuda_tracer_events",
        "jxplanetx.fast_wisdom_holman", "jxplanetx._all_conic_kepler",
        "jxplanetx._wisdom_holman_cpu_v3", "jxplanetx.engine.backends",
        "jxplanetx.engine.contracts",
    )
    modules = value.get("critical_jx_modules")
    if type(modules) is not list or len(modules) != len(expected_modules):
        raise LifecycleGateError(f"{label} critical module roster differs")
    for expected_name, receipt in zip(expected_modules, modules, strict=True):
        if type(receipt) is not dict or receipt.get("module") != expected_name:
            raise LifecycleGateError(f"{label} critical module identity differs")
        _validate_runtime_file_receipt(
            {key: receipt.get(key) for key in ("path", "bytes", "sha256")},
            f"{label} {expected_name}",
        )
    _validate_builtin_math_receipt(value["builtin_math"], label)
    _validate_libm_receipt(value["mapped_libm_point_observation"], label)
    return value


def _validate_runtime_identity(
    value: object,
    d: types.ModuleType,
    label: str,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    if expected_mode not in {RUNTIME_MODE_PREFLIGHT, RUNTIME_MODE_PRODUCTION}:
        raise LifecycleGateError(f"{label} expected runtime mode differs")
    keys = set(_runtime_base_keys()) | {
        "schema", "source_only_runtime_import_guard",
        "engine_runtime_projection", "content_sha256",
    }
    if (
        type(value) is not dict or set(value) != keys
        or value.get("schema") != RUNTIME_SCHEMA
        or value.get("content_sha256") != _domain_hash(RUNTIME_DOMAIN, value)
    ):
        raise LifecycleGateError(f"{label} runtime identity differs")
    base = {key: value[key] for key in _runtime_base_keys()}
    _validate_runtime_base(base, d, label)
    try:
        d._validate_runtime_import_guard(
            value["source_only_runtime_import_guard"], f"{label} import guard"
        )
    except Exception as exc:
        raise LifecycleGateError(f"{label} bound runtime receipt differs") from exc
    cupy = value.get("cupy_cuda")
    engine = value.get("engine_runtime_projection")
    if (
        type(engine) is not dict
        or set(engine) != {
            "method_id", "implementation_source_sha256", "route_worker_count",
            "determinism_scope", "hybrid_runtime_identity_without_source_tile",
        }
        or not _is_sha256(engine.get("implementation_source_sha256"))
        or type(engine.get("route_worker_count")) is not int
        or type(engine.get("hybrid_runtime_identity_without_source_tile")) is not dict
    ):
        raise LifecycleGateError(f"{label} engine runtime projection differs")
    hybrid = engine["hybrid_runtime_identity_without_source_tile"]
    hybrid_keys = {
        "method_id", "implementation_source_sha256",
        "quiet_cuda_runtime_identity", "route_runtime_identities",
        "route_policy", "route_worker_count", "route_worker_start_method",
        "determinism_scope",
    }
    if type(hybrid) is not dict or set(hybrid) != hybrid_keys:
        raise LifecycleGateError(f"{label} hybrid runtime projection differs")
    quiet = hybrid.get("quiet_cuda_runtime_identity")
    try:
        d._validate_fused_tracer_runtime_identity(quiet)
    except Exception as exc:
        raise LifecycleGateError(f"{label} engine CUDA identity differs") from exc
    if (
        cupy["cupy_version"] != quiet.get("cupy_version")
        or cupy["cuda_runtime_version"] != quiet.get("cuda_runtime_version")
        or cupy["cuda_local_runtime_version"]
        != quiet.get("cuda_local_runtime_version")
        or cupy["cuda_driver_version"] != quiet.get("cuda_driver_version")
        or cupy["device_index"] != quiet.get("device_index")
        or cupy["device_name"] != quiet.get("device_name")
        or cupy["device_uuid"] != quiet.get("device_uuid")
        or cupy["compute_capability"]
        != quiet.get("device_compute_capability")
        or cupy["pci_domain_bus_device"]
        != quiet.get("device_pci_domain_bus_device")
    ):
        raise LifecycleGateError(f"{label} CuPy/engine CUDA cross-binding differs")
    module_receipts = {
        item["module"]: item for item in value["critical_jx_modules"]
    }
    common_determinism = "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE"
    production_determinism = (
        "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
    )
    if expected_mode == RUNTIME_MODE_PREFLIGHT:
        if (
            engine["method_id"]
            != "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3"
            or engine["implementation_source_sha256"]
            != module_receipts["jxplanetx.cuda_restricted_wh"]["sha256"]
            or type(engine["route_worker_count"]) is not int
            or engine["route_worker_count"] != 1
            or engine["determinism_scope"] != common_determinism
            or hybrid["method_id"]
            != "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3"
            or hybrid["implementation_source_sha256"]
            != module_receipts["jxplanetx.cuda_restricted_wh"]["sha256"]
            or hybrid["route_runtime_identities"] != []
            or hybrid["route_policy"] != "NOT_INVOKED_RUNTIME_PREFLIGHT_ONLY"
            or type(hybrid["route_worker_count"]) is not int
            or hybrid["route_worker_count"] != 1
            or hybrid["route_worker_start_method"] != "SERIAL"
            or hybrid["determinism_scope"] != common_determinism
        ):
            raise LifecycleGateError(f"{label} preflight engine identity differs")
    elif (
        engine["method_id"]
        != "JX_CUDA_RESTRICTED_WH_FIXED_LANE_LIFECYCLE_V1"
        or engine["implementation_source_sha256"]
        != module_receipts["jxplanetx.restricted_wh_lifecycle"]["sha256"]
        or type(engine["route_worker_count"]) is not int
        or engine["route_worker_count"] != 1
        or engine["determinism_scope"] != production_determinism
        or hybrid["method_id"] != "JX_CUDA_RESTRICTED_WH_GUARDED_DOP853_V2"
        or hybrid["implementation_source_sha256"]
        != module_receipts["jxplanetx.cuda_restricted_wh_hybrid"]["sha256"]
        or hybrid["route_runtime_identities"] != []
        or hybrid["route_policy"] != _expected_route_policy_identity()
        or type(hybrid["route_worker_count"]) is not int
        or hybrid["route_worker_count"] != 1
        or hybrid["route_worker_start_method"] != "SERIAL"
        or hybrid["determinism_scope"] != production_determinism
    ):
        raise LifecycleGateError(f"{label} production engine identity differs")
    return value


def _runtime_preflight_engine_projection(cp: Any) -> dict[str, Any]:
    module = sys.modules.get("jxplanetx.cuda_restricted_wh")
    identity_function = getattr(module, "_runtime_identity", None)
    if type(module) is not types.ModuleType or not callable(identity_function):
        raise LifecycleGateError("preflight CUDA runtime identity API is absent")
    quiet = _plain(identity_function(cp, "cuda:0"))
    if type(quiet) is not dict or not _is_sha256(
        quiet.get("implementation_source_sha256")
    ):
        raise LifecycleGateError("preflight CUDA runtime identity differs")
    return {
        "method_id": "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3",
        "implementation_source_sha256": quiet["implementation_source_sha256"],
        "route_worker_count": 1,
        "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
        "hybrid_runtime_identity_without_source_tile": {
            "method_id": "JX_TNO_1KYR_RUNTIME_PREFLIGHT_NO_DYNAMICS_V3",
            "implementation_source_sha256": quiet["implementation_source_sha256"],
            "quiet_cuda_runtime_identity": quiet,
            "route_runtime_identities": [],
            "route_policy": "NOT_INVOKED_RUNTIME_PREFLIGHT_ONLY",
            "route_worker_count": 1,
            "route_worker_start_method": "SERIAL",
            "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
        },
    }


def _runtime_preflight_worker(
    import_probe_path: Path,
    *,
    worker_order: int,
    parent_challenge_sha256: str,
) -> dict[str, Any]:
    if (
        type(worker_order) is not int
        or worker_order not in (0, 1)
        or not _is_sha256(parent_challenge_sha256)
    ):
        raise LifecycleGateError("runtime-preflight worker challenge differs")
    with _delegates() as (grid, _h64, d):
        fresh_child = d._fresh_child_interpreter_receipt()
        fresh_registry = d._reject_preloaded_runtime_modules()
        _probe_payload, probe = d._load_json(
            import_probe_path, "runtime-preflight import probe"
        )
        d._validate_import_probe_receipt(probe, expected_path=import_probe_path)
        d.validate_execution_prerequisites(d.DE441_ROOT)
        with tempfile.TemporaryDirectory(prefix="jx-tno-1k-runtime-preflight-") as raw:
            cache_root = Path(raw) / "pycache-must-remain-absent"
            with d._source_only_runtime_import_guard(cache_root) as guard:
                cp, _rebound, _target_receipt = d._load_dynamics_runtime(
                    d.DEFAULT_REBOUND_TARGET
                )
                grid.construct_adapter_specs_v1(
                    map_step_count=SEGMENT_STEPS,
                    source_step_count=HORIZON_YEARS * STEPS_PER_YEAR,
                )
                runtime_base = _critical_runtime_base(
                    d,
                    cp,
                    fresh_child=fresh_child,
                    fresh_registry=fresh_registry,
                )
                engine_projection = _runtime_preflight_engine_projection(cp)
            runtime = _critical_runtime_identity(
                d,
                runtime_base=runtime_base,
                import_guard=guard,
                engine_projection=engine_projection,
                expected_mode=RUNTIME_MODE_PREFLIGHT,
            )
        receipt: dict[str, Any] = {
            "schema": RUNTIME_PREFLIGHT_WORKER_SCHEMA,
            "worker_order": worker_order,
            "parent_challenge_sha256": parent_challenge_sha256,
            "process_id": os.getpid(),
            "runtime_identity": runtime,
            "runtime_content_sha256": runtime["content_sha256"],
            "runtime_comparison_projection_sha256": _sha256(
                _canonical_json(_runtime_comparison_projection(runtime))
            ),
            "prelane_binding_projection_sha256": _sha256(
                _canonical_json(_runtime_prelane_binding_projection(runtime))
            ),
            "source_step_count": 0,
            "tracer_count": 0,
            "dynamics_executed": False,
            "qualification_result": False,
            "classification_generated": False,
            "science_result": False,
            "content_sha256": "",
        }
        receipt["content_sha256"] = _domain_hash(
            RUNTIME_PREFLIGHT_WORKER_DOMAIN, receipt
        )
        return receipt


def _validate_runtime_preflight_worker_receipt(
    value: object, d: types.ModuleType, label: str
) -> dict[str, Any]:
    keys = {
        "schema", "worker_order", "parent_challenge_sha256", "process_id",
        "runtime_identity", "runtime_content_sha256",
        "runtime_comparison_projection_sha256",
        "prelane_binding_projection_sha256", "source_step_count", "tracer_count",
        "dynamics_executed", "qualification_result", "classification_generated",
        "science_result", "content_sha256",
    }
    runtime = value.get("runtime_identity") if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != keys
        or value.get("schema") != RUNTIME_PREFLIGHT_WORKER_SCHEMA
        or type(value.get("worker_order")) is not int
        or value.get("worker_order") not in (0, 1)
        or not _is_sha256(value.get("parent_challenge_sha256"))
        or type(value.get("process_id")) is not int
        or value.get("process_id") <= 0
        or type(runtime) is not dict
        or value.get("runtime_content_sha256") != runtime.get("content_sha256")
        or value.get("runtime_comparison_projection_sha256")
        != _sha256(_canonical_json(_runtime_comparison_projection(runtime)))
        or value.get("prelane_binding_projection_sha256")
        != _sha256(_canonical_json(_runtime_prelane_binding_projection(runtime)))
        or type(value.get("source_step_count")) is not int
        or value.get("source_step_count") != 0
        or type(value.get("tracer_count")) is not int
        or value.get("tracer_count") != 0
        or value.get("dynamics_executed") is not False
        or value.get("qualification_result") is not False
        or value.get("classification_generated") is not False
        or value.get("science_result") is not False
        or value.get("content_sha256")
        != _domain_hash(RUNTIME_PREFLIGHT_WORKER_DOMAIN, value)
    ):
        raise LifecycleGateError(f"{label} runtime-preflight receipt differs")
    _validate_runtime_identity(
        runtime, d, f"{label} runtime", expected_mode=RUNTIME_MODE_PREFLIGHT
    )
    return value


def _validate_runtime_preflight_document(
    value: object, d: types.ModuleType
) -> dict[str, Any]:
    keys = {
        "schema", "worker_receipts", "process_independent_projection_sha256",
        "prelane_binding_projection_sha256", "worker_challenge_sha256s",
        "fresh_child_count",
        "source_step_count", "tracer_count", "dynamics_executed",
        "qualification_result", "classification_generated", "science_result",
        "content_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value.get("schema") != RUNTIME_PREFLIGHT_SCHEMA
        or type(value.get("worker_receipts")) is not list
        or len(value["worker_receipts"]) != 2
        or type(value.get("worker_challenge_sha256s")) is not list
        or len(value["worker_challenge_sha256s"]) != 2
        or any(
            not _is_sha256(item)
            for item in value["worker_challenge_sha256s"]
        )
        or len(set(value["worker_challenge_sha256s"])) != 2
        or type(value.get("fresh_child_count")) is not int
        or value.get("fresh_child_count") != 2
        or type(value.get("source_step_count")) is not int
        or value.get("source_step_count") != 0
        or type(value.get("tracer_count")) is not int
        or value.get("tracer_count") != 0
        or value.get("dynamics_executed") is not False
        or value.get("qualification_result") is not False
        or value.get("classification_generated") is not False
        or value.get("science_result") is not False
        or value.get("content_sha256") != _domain_hash(RUNTIME_PREFLIGHT_DOMAIN, value)
    ):
        raise LifecycleGateError("runtime-preflight document differs")
    receipts = [
        _validate_runtime_preflight_worker_receipt(item, d, f"worker {index}")
        for index, item in enumerate(value["worker_receipts"])
    ]
    if [item["worker_order"] for item in receipts] != [0, 1]:
        raise LifecycleGateError("runtime-preflight worker order differs")
    if [item["parent_challenge_sha256"] for item in receipts] != value[
        "worker_challenge_sha256s"
    ]:
        raise LifecycleGateError("runtime-preflight worker challenges differ")
    if len({item["process_id"] for item in receipts}) != 2:
        raise LifecycleGateError("runtime-preflight worker processes are not distinct")
    pycache_paths = [
        item["runtime_identity"]["source_only_runtime_import_guard"][
            "pycache_prefix"
        ]
        for item in receipts
    ]
    cupy_cache_paths = [
        item["runtime_identity"]["source_only_runtime_import_guard"][
            "cupy_external_kernel_cache"
        ]["directory"]
        for item in receipts
    ]
    if len(set(pycache_paths)) != 2 or len(set(cupy_cache_paths)) != 2:
        raise LifecycleGateError("runtime-preflight worker guard paths are not distinct")
    projections = [
        _runtime_comparison_projection(item["runtime_identity"])
        for item in receipts
    ]
    payloads = [_canonical_json(item) for item in projections]
    if payloads[0] != payloads[1]:
        raise LifecycleGateError("fresh-child runtime projections differ")
    digest = _sha256(payloads[0])
    if value.get("process_independent_projection_sha256") != digest:
        raise LifecycleGateError("runtime-preflight projection digest differs")
    binding_payloads = [
        _canonical_json(_runtime_prelane_binding_projection(item["runtime_identity"]))
        for item in receipts
    ]
    if binding_payloads[0] != binding_payloads[1]:
        raise LifecycleGateError("fresh-child prelane runtime bindings differ")
    if value.get("prelane_binding_projection_sha256") != _sha256(
        binding_payloads[0]
    ):
        raise LifecycleGateError("runtime-preflight binding digest differs")
    return value


def _source_payload(np: Any, full: object) -> tuple[bytes, bytes, bytes]:
    positions = np.ascontiguousarray(full.source_relative_positions_AU, dtype=np.dtype("<f8")).tobytes(order="C")
    velocities = np.ascontiguousarray(full.source_relative_velocities_AU_per_year, dtype=np.dtype("<f8")).tobytes(order="C")
    gms = np.ascontiguousarray(full.source_gm_AU3_per_year2, dtype=np.dtype("<f8")).tobytes(order="C")
    if len(positions) != SOURCE_LANE_BYTES // 2 or len(velocities) != SOURCE_LANE_BYTES // 2 or len(gms) != SOURCE_COMMON_GM_BYTES:
        raise LifecycleGateError("generated source payload geometry differs")
    return positions, velocities, gms


def _source_segment_payload(np: Any, full: object, segment_index: int) -> bytes:
    start = segment_index * SEGMENT_STEPS
    stop = (segment_index + 1) * SEGMENT_STEPS + 1
    positions = np.ascontiguousarray(full.source_relative_positions_AU[start:stop], dtype=np.dtype("<f8")).tobytes(order="C")
    velocities = np.ascontiguousarray(full.source_relative_velocities_AU_per_year[start:stop], dtype=np.dtype("<f8")).tobytes(order="C")
    return positions + velocities


def _retained_source_segment(
    descriptor: int, source_record: Mapping[str, Any], segment_index: int,
) -> bytes:
    node_start = segment_index * SEGMENT_STEPS
    node_count = SEGMENT_STEPS + 1
    node_bytes = SOURCE_COUNT * 3 * 8
    position_offset = int(source_record["byte_offset"]) + node_start * node_bytes
    velocity_offset = (
        int(source_record["byte_offset"])
        + int(source_record["positions_block_bytes"])
        + node_start * node_bytes
    )
    length = node_count * node_bytes
    return (
        _pread_exact(descriptor, position_offset, length, "retained source segment positions")
        + _pread_exact(descriptor, velocity_offset, length, "retained source segment velocities")
    )


def _build_batch_context(
    grid: types.ModuleType,
    d: types.ModuleType,
    cp: Any,
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    batch_order: int,
) -> dict[str, Any]:
    import numpy as np
    from jxplanetx.cuda_restricted_wh import COORDINATE_SYSTEM, CUDARestrictedWHState
    from jxplanetx.engine import Provenance

    batch = d._batch_states(bridge, custody, "DE441")[batch_order]
    reduced = d.collapse_reduced_model(batch)
    step = 1.0 / STEPS_PER_YEAR
    central = reduced.massive[0]
    source_positions = np.asarray(
        [d._sub(item.position_AU, central.position_AU) for item in reduced.massive], dtype=np.float64
    )
    source_velocities = np.asarray(
        [d._sub(item.velocity_AU_per_julian_year, central.velocity_AU_per_julian_year) for item in reduced.massive], dtype=np.float64
    )
    source_positions[0] = 0.0
    source_velocities[0] = 0.0
    source_gm = np.asarray([item.gm_AU3_per_julian_year2 for item in reduced.massive], dtype=np.float64)
    radii = np.ones(SOURCE_COUNT, dtype=np.float64)
    required_bound = math.nextafter(2.0 * math.fsum(float(value) for value in source_gm), math.inf)
    bounds = np.full(SOURCE_COUNT, required_bound, dtype=np.float64)
    if required_bound.hex() != grid.REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX:
        raise LifecycleGateError("relative acceleration certificate differs")
    schedule = d._source_schedule_sha256(reduced, step)
    provenance = Provenance(
        source_id=f"jx-tno-stage1-v5-h64-v2-DE441-{batch.object_id}-64",
        citation="Retained Stage-1 v4 TNO native-epoch bridge and common GM contract",
        version="v5-h64-closure-v2",
        sha256=schedule,
    )
    specs = grid.construct_adapter_specs_v1(
        map_step_count=SEGMENT_STEPS, source_step_count=HORIZON_YEARS * STEPS_PER_YEAR,
    )
    build = specs["source_builder"](
        tile_id=f"stage1-v5-h64-v2-DE441-{batch.object_id}-h1_64",
        epoch_origin_years=(batch.epoch_jd_tdb - 2_451_545.0) / 365.25,
        global_source_step=0,
        time_scale="TDB",
        frame="ICRF_J2000",
        axes="J2000_EQUATORIAL",
        source_ids=grid.REDUCED_BODY_IDS,
        primary_source_id=grid.REDUCED_BODY_IDS[0],
        source_schedule_sha256=schedule,
        initial_source_relative_positions_AU=source_positions,
        initial_source_relative_velocities_AU_per_year=source_velocities,
        source_gm_AU3_per_year2=source_gm,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=bounds,
        provenance=provenance,
        spec=specs["source"],
    )
    full = build.tile
    tracer_positions = np.asarray(
        [d._sub(item.position_AU, central.position_AU) for item in reduced.tracers], dtype=np.float64
    )
    tracer_velocities = np.asarray(
        [d._sub(item.velocity_AU_per_julian_year, central.velocity_AU_per_julian_year) for item in reduced.tracers], dtype=np.float64
    )
    state = CUDARestrictedWHState(
        snapshot_id=f"tno-1k-lifecycle-v3-DE441-{batch.object_id}",
        epoch_origin_years=full.epoch_origin_years,
        global_source_step=0,
        source_step_years=step,
        time_scale=full.time_scale,
        frame=full.frame,
        axes=full.axes,
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id=full.unit_system_id,
        source_ids=full.source_ids,
        primary_source_id=full.primary_source_id,
        source_schedule_sha256=schedule,
        tracer_relative_positions_AU=cp.asarray(tracer_positions),
        tracer_relative_velocities_AU_per_year=cp.asarray(tracer_velocities),
        tracer_indices=cp.arange(9, dtype=cp.uint64),
        provenance=provenance,
    )
    return {
        "batch": batch,
        "reduced": reduced,
        "build": build,
        "full": full,
        "specs": specs,
        "state": state,
        "initial_tracer_positions": tracer_positions,
        "initial_tracer_velocities": tracer_velocities,
    }


def _builtin_finite_float3_rows(
    np: Any,
    value: object,
    count: int,
    label: str,
) -> tuple[tuple[float, float, float], ...]:
    if type(count) is not int or count <= 0:
        raise LifecycleGateError(f"{label} count differs")
    if type(value) is not np.ndarray:
        raise LifecycleGateError(f"{label} must be an exact NumPy array")
    if value.dtype != np.dtype("<f8") or value.shape != (count, 3):
        raise LifecycleGateError(f"{label} binary64 geometry differs")
    rows = tuple(
        tuple(float(value[row, axis]) for axis in range(3))
        for row in range(count)
    )
    if any(
        type(component) is not float or not math.isfinite(component)
        for row in rows
        for component in row
    ):
        raise LifecycleGateError(f"{label} contains a non-finite component")
    return rows


def _checkpoint_from_host_arrays(
    np: Any,
    grid: types.ModuleType,
    kernel: object,
    *,
    batch_order: int,
    source_positions: object,
    source_velocities: object,
    tracer_positions: object,
    tracer_velocities: object,
) -> bytes:
    return grid.checkpoint_from_primary_relative_v1(
        kernel,
        batch_order=batch_order,
        source_relative_positions_AU=_builtin_finite_float3_rows(
            np, source_positions, SOURCE_COUNT, "source positions"
        ),
        source_relative_velocities_AU_per_julian_year=_builtin_finite_float3_rows(
            np, source_velocities, SOURCE_COUNT, "source velocities"
        ),
        tracer_relative_positions_AU=_builtin_finite_float3_rows(
            np, tracer_positions, 9, "tracer positions"
        ),
        tracer_relative_velocities_AU_per_julian_year=_builtin_finite_float3_rows(
            np, tracer_velocities, 9, "tracer velocities"
        ),
    )


def _checkpoint_payload(
    grid: types.ModuleType,
    kernel: object,
    cp: Any,
    *,
    batch_order: int,
    full: object,
    state: object,
    source_node: int,
) -> bytes:
    import numpy as np

    dynamics = state.dynamics_state
    return _checkpoint_from_host_arrays(
        np,
        grid,
        kernel,
        batch_order=batch_order,
        source_positions=full.source_relative_positions_AU[source_node],
        source_velocities=full.source_relative_velocities_AU_per_year[source_node],
        tracer_positions=cp.asnumpy(dynamics.tracer_relative_positions_AU),
        tracer_velocities=cp.asnumpy(
            dynamics.tracer_relative_velocities_AU_per_year
        ),
    )


def _run_batch(
    grid: types.ModuleType,
    d: types.ModuleType,
    cp: Any,
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    retained_result: Mapping[str, Any],
    source_descriptor: int,
    kernel: object,
    batch_order: int,
) -> dict[str, Any]:
    import numpy as np

    context = _build_batch_context(grid, d, cp, bridge, custody, batch_order)
    full = context["full"]
    build = context["build"]
    specs = context["specs"]
    lane, source_record = _retained_lane_records(retained_result, batch_order)
    positions_payload, velocities_payload, gm_payload = _source_payload(np, full)
    generated_source_payload = positions_payload + velocities_payload
    retained_source_payload = _pread_exact(
        source_descriptor,
        int(source_record["byte_offset"]),
        int(source_record["byte_length"]),
        "retained full source lane",
    )
    retained_gm = _pread_exact(source_descriptor, 0, SOURCE_COMMON_GM_BYTES, "retained source GM")
    if (
        generated_source_payload != retained_source_payload
        or gm_payload != retained_gm
        or _sha256(generated_source_payload) != source_record["lane_payload_sha256"]
        or _sha256(positions_payload) != source_record["positions_sha256"]
        or _sha256(velocities_payload) != source_record["velocities_sha256"]
        or full.content_sha256 != source_record["full_source_tile_sha256"]
    ):
        raise LifecycleGateError("generated source path differs bitwise from retained h64")
    invariant = d._source_node_invariant_summary(
        full.source_relative_positions_AU,
        full.source_relative_velocities_AU_per_year,
        full.source_gm_AU3_per_year2,
    )
    d._require_source_invariant_summary(lane.get("summary"), invariant)
    if (
        not float.fromhex(invariant["maximum_relative_massive_energy_drift_hex"]) <= 1.0e-8
        or not float.fromhex(
            invariant["maximum_relative_massive_angular_momentum_drift_hex"]
        ) <= 1.0e-10
    ):
        raise LifecycleGateError("source invariant gate failed")

    lifecycle_state = specs["lifecycle_initialize"](context["state"])
    checkpoints: list[tuple[int, bytes]] = []
    initial = _checkpoint_from_host_arrays(
        np,
        grid,
        kernel,
        batch_order=batch_order,
        source_positions=full.source_relative_positions_AU[0],
        source_velocities=full.source_relative_velocities_AU_per_year[0],
        tracer_positions=context["initial_tracer_positions"],
        tracer_velocities=context["initial_tracer_velocities"],
    )
    checkpoints.append((0, initial))
    segments: list[dict[str, Any]] = []
    internal_records: list[bytes] = []
    midpoint_payload: bytes | None = None
    engine_projection: dict[str, Any] | None = None
    started_lane = time.perf_counter()
    for segment_index in range(SEGMENT_COUNT):
        tile = grid.slice_frozen_source_tile_v1(
            full,
            local_start_step=segment_index * SEGMENT_STEPS,
            map_step_count=SEGMENT_STEPS,
            tile_id=f"tno-1k-lifecycle-v3-{OBJECT_IDS[batch_order]}-segment-{segment_index:02d}",
        )
        generated_segment = _source_segment_payload(np, full, segment_index)
        retained_segment = _retained_source_segment(source_descriptor, source_record, segment_index)
        if generated_segment != retained_segment:
            raise LifecycleGateError("generated 50-year source segment differs from retained h64")
        started = time.perf_counter()
        result = specs["lifecycle_advance"](
            lifecycle_state,
            tile,
            specs["fixed_step"],
            route_spec=specs["route"],
            route_worker_count=specs["route_worker_count"],
        )
        elapsed = time.perf_counter() - started
        lifecycle_state = result.state
        end_step = (segment_index + 1) * SEGMENT_STEPS
        if (
            lifecycle_state.dynamics_state.global_source_step != end_step
            or tuple(int(item) for item in lifecycle_state.tracer_lifecycle.tolist()) != (0,) * 9
            or result.route_records != ()
            or result.new_removal_records != ()
            or lifecycle_state.removal_records != ()
        ):
            raise LifecycleGateError("lifecycle segment routed, removed, or incompletely advanced")
        minimum = tuple(float(item) for item in cp.asnumpy(result.minimum_source_distance_AU).tolist())
        if len(minimum) != 9 or any(not math.isfinite(item) or not item > 1.0 for item in minimum):
            raise LifecycleGateError("lifecycle source-distance gate failed")
        projection = _engine_runtime_projection(result.runtime_identity)
        if engine_projection is None:
            engine_projection = projection
        elif _canonical_json(engine_projection) != _canonical_json(projection):
            raise LifecycleGateError("engine runtime projection changed between segments")
        year = (segment_index + 1) * SEGMENT_YEARS
        checkpoint = _checkpoint_payload(
            grid,
            kernel,
            cp,
            batch_order=batch_order,
            full=full,
            state=lifecycle_state,
            source_node=end_step,
        )
        checkpoints.append((year, checkpoint))
        internal = _pack_internal_tracer_state(cp, lifecycle_state.dynamics_state)
        if year == 500:
            midpoint_payload = internal
        if year >= 500:
            internal_records.append(internal)
        record: dict[str, Any] = {
            "schema": SEGMENT_SCHEMA,
            "segment_order": batch_order * SEGMENT_COUNT + segment_index,
            "batch_order": batch_order,
            "object_id": OBJECT_IDS[batch_order],
            "segment_index": segment_index,
            "start_year": segment_index * SEGMENT_YEARS,
            "end_year": year,
            "start_global_source_step": segment_index * SEGMENT_STEPS,
            "end_global_source_step": end_step,
            "source_tile_sha256": tile.content_sha256,
            "generated_source_segment_sha256": _sha256(generated_segment),
            "retained_source_segment_sha256": _sha256(retained_segment),
            "checkpoint_sha256": _sha256(checkpoint),
            "active_tracer_count": 9,
            "route_record_count": 0,
            "removal_record_count": 0,
            "minimum_source_distance_AU_hex": min(minimum).hex(),
            "maximum_kepler_iterations": result.maximum_kepler_iterations,
            "engine_runtime_identity_sha256": _sha256(
                _canonical_json(_plain(result.runtime_identity))
            ),
            "elapsed_seconds_hex": elapsed.hex(),
            "content_sha256": "",
        }
        record["content_sha256"] = _domain_hash(SEGMENT_DOMAIN, record)
        _validate_segment_record(record, expected_order=record["segment_order"])
        segments.append(record)
    if midpoint_payload is None or len(internal_records) != RESTART_RECORD_COUNT or engine_projection is None:
        raise LifecycleGateError("midpoint, suffix, or runtime evidence is incomplete")
    crossbind = grid.validate_first_1000_year_prefix_v1(batch_order, tuple(checkpoints))
    if crossbind.get("passed") is not True:
        raise LifecycleGateError("first-1000-year prefix crossbinding did not pass")
    source_build_identity = _plain(build.runtime_identity)
    lane_record = {
        "batch_order": batch_order,
        "object_id": OBJECT_IDS[batch_order],
        "source_order": source_record["source_order"],
        "full_source_tile_sha256": full.content_sha256,
        "generated_source_payload_sha256": _sha256(generated_source_payload),
        "retained_source_payload_sha256": source_record["lane_payload_sha256"],
        "source_build_input_content_sha256": build.input_content_sha256,
        "source_map_result_content_sha256": build.source_map_result_content_sha256,
        "source_map_verification_content_sha256": build.source_map_verification_content_sha256,
        "source_build_runtime_identity_sha256": _sha256(_canonical_json(source_build_identity)),
        "source_invariants": invariant,
        "prefix_crossbinding": crossbind,
        "segment_count": len(segments),
        "route_record_count": 0,
        "removal_record_count": 0,
        "elapsed_seconds_hex": (time.perf_counter() - started_lane).hex(),
    }
    return {
        "lane_record": lane_record,
        "checkpoint_payload": b"".join(payload for _, payload in checkpoints),
        "segments": segments,
        "midpoint_payload": midpoint_payload,
        "main_restart_suffix": b"".join(internal_records),
        "engine_projection": engine_projection,
    }


def _restart_request(midpoint_payload: bytes) -> dict[str, Any]:
    _unpack_internal_tracer_state(midpoint_payload)
    request: dict[str, Any] = {
        "schema": RESTART_REQUEST_SCHEMA,
        "batch_order": RESTART_BATCH_ORDER,
        "object_id": RESTART_OBJECT_ID,
        "start_year": 500,
        "start_global_source_step": 32_000,
        "midpoint_byte_length": len(midpoint_payload),
        "midpoint_sha256": _sha256(midpoint_payload),
        "midpoint_base64": base64.b64encode(midpoint_payload).decode("ascii"),
        "content_sha256": "",
    }
    request["content_sha256"] = _domain_hash(RESTART_REQUEST_DOMAIN, request)
    return request


def _crossbind_restart_stream_to_prefix(
    grid: types.ModuleType,
    monolithic: bytes,
    prefix: bytes,
    source_descriptor: int,
    source_record: Mapping[str, Any],
) -> None:
    if len(monolithic) != RESTART_RECORD_COUNT * TRACER_INTERNAL_BYTES:
        raise LifecycleGateError("restart monolithic stream geometry differs")
    node_bytes = SOURCE_COUNT * 3 * 8
    with grid.FrozenClassifierKernelV1() as kernel:
        for record_index in range(RESTART_RECORD_COUNT):
            year = 500 + record_index * SEGMENT_YEARS
            source_node = year * STEPS_PER_YEAR
            internal = monolithic[
                record_index * TRACER_INTERNAL_BYTES :
                (record_index + 1) * TRACER_INTERNAL_BYTES
            ]
            tracer_positions, tracer_velocities = _unpack_internal_tracer_state(
                internal
            )
            position_payload = _pread_exact(
                source_descriptor,
                int(source_record["byte_offset"]) + source_node * node_bytes,
                node_bytes,
                "restart crossbind source positions",
            )
            velocity_payload = _pread_exact(
                source_descriptor,
                int(source_record["byte_offset"])
                + int(source_record["positions_block_bytes"])
                + source_node * node_bytes,
                node_bytes,
                "restart crossbind source velocities",
            )
            source_positions_flat = struct.unpack("<15d", position_payload)
            source_velocities_flat = struct.unpack("<15d", velocity_payload)
            source_positions = tuple(
                tuple(source_positions_flat[index * 3 : index * 3 + 3])
                for index in range(SOURCE_COUNT)
            )
            source_velocities = tuple(
                tuple(source_velocities_flat[index * 3 : index * 3 + 3])
                for index in range(SOURCE_COUNT)
            )
            observed = grid.checkpoint_from_primary_relative_v1(
                kernel,
                batch_order=RESTART_BATCH_ORDER,
                source_relative_positions_AU=source_positions,
                source_relative_velocities_AU_per_julian_year=source_velocities,
                tracer_relative_positions_AU=tracer_positions,
                tracer_relative_velocities_AU_per_julian_year=tracer_velocities,
            )
            prefix_index = year // SEGMENT_YEARS
            prefix_offset = (
                RESTART_BATCH_ORDER * len(CHECKPOINT_YEARS) + prefix_index
            ) * CHECKPOINT_BYTES
            expected = prefix[
                prefix_offset : prefix_offset + CHECKPOINT_BYTES
            ]
            if observed != expected:
                raise LifecycleGateError(
                    "restart internal state differs from retained batch-9 prefix"
                )


def _validate_restart_request(value: object) -> tuple[dict[str, Any], bytes]:
    expected_keys = {
        "schema", "batch_order", "object_id", "start_year",
        "start_global_source_step", "midpoint_byte_length", "midpoint_sha256",
        "midpoint_base64", "content_sha256",
    }
    if (
        type(value) is not dict or set(value) != expected_keys
        or value.get("schema") != RESTART_REQUEST_SCHEMA
        or value.get("batch_order") != RESTART_BATCH_ORDER
        or value.get("object_id") != RESTART_OBJECT_ID
        or value.get("start_year") != 500
        or value.get("start_global_source_step") != 32_000
        or value.get("midpoint_byte_length") != TRACER_INTERNAL_BYTES
        or not _is_sha256(value.get("midpoint_sha256"))
        or value.get("content_sha256") != _domain_hash(RESTART_REQUEST_DOMAIN, value)
        or type(value.get("midpoint_base64")) is not str
    ):
        raise LifecycleGateError("restart request differs")
    try:
        payload = base64.b64decode(value["midpoint_base64"], validate=True)
    except Exception as exc:
        raise LifecycleGateError("restart midpoint encoding differs") from exc
    if len(payload) != TRACER_INTERNAL_BYTES or _sha256(payload) != value["midpoint_sha256"]:
        raise LifecycleGateError("restart midpoint payload differs")
    _unpack_internal_tracer_state(payload)
    return value, payload


def _restart_suffix(
    grid: types.ModuleType,
    d: types.ModuleType,
    cp: Any,
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    midpoint_payload: bytes,
) -> tuple[bytes, dict[str, Any]]:
    from dataclasses import replace

    context = _build_batch_context(
        grid, d, cp, bridge, custody, RESTART_BATCH_ORDER
    )
    positions, velocities = _unpack_internal_tracer_state(midpoint_payload)
    base_state = context["state"]
    resumed_state = replace(
        base_state,
        snapshot_id=f"tno-1k-lifecycle-v3-restart-{RESTART_OBJECT_ID}",
        global_source_step=32_000,
        tracer_relative_positions_AU=cp.asarray(positions, dtype=cp.float64),
        tracer_relative_velocities_AU_per_year=cp.asarray(velocities, dtype=cp.float64),
    )
    specs = context["specs"]
    full = context["full"]
    lifecycle_state = specs["lifecycle_initialize"](resumed_state)
    records = [midpoint_payload]
    projection: dict[str, Any] | None = None
    for segment_index in range(10, SEGMENT_COUNT):
        tile = grid.slice_frozen_source_tile_v1(
            full,
            local_start_step=segment_index * SEGMENT_STEPS,
            map_step_count=SEGMENT_STEPS,
            tile_id=f"tno-1k-lifecycle-v3-restart-{RESTART_OBJECT_ID}-segment-{segment_index:02d}",
        )
        result = specs["lifecycle_advance"](
            lifecycle_state,
            tile,
            specs["fixed_step"],
            route_spec=specs["route"],
            route_worker_count=specs["route_worker_count"],
        )
        lifecycle_state = result.state
        current = _engine_runtime_projection(result.runtime_identity)
        if projection is None:
            projection = current
        elif _canonical_json(projection) != _canonical_json(current):
            raise LifecycleGateError("restart engine runtime projection changed")
        if (
            lifecycle_state.dynamics_state.global_source_step != (segment_index + 1) * SEGMENT_STEPS
            or tuple(int(item) for item in lifecycle_state.tracer_lifecycle.tolist()) != (0,) * 9
            or result.route_records != () or result.new_removal_records != ()
        ):
            raise LifecycleGateError("restart route/removal/global-step gate failed")
        minimum = tuple(float(item) for item in cp.asnumpy(result.minimum_source_distance_AU).tolist())
        if any(not math.isfinite(item) or not item > 1.0 for item in minimum):
            raise LifecycleGateError("restart source-distance gate failed")
        records.append(_pack_internal_tracer_state(cp, lifecycle_state.dynamics_state))
    if len(records) != RESTART_RECORD_COUNT or projection is None:
        raise LifecycleGateError("restart suffix geometry differs")
    return b"".join(records), projection


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise LifecycleGateError("Git preregistration identity could not be read")
    return completed.stdout.strip()


def _git_custody(*, remote_verified: bool) -> dict[str, Any]:
    if type(remote_verified) is not bool or not remote_verified:
        raise LifecycleGateError("external remote preregistration was not confirmed")
    branch = _git_value("branch", "--show-current")
    commit = _git_value("rev-parse", "HEAD")
    tag_object = _git_value("rev-parse", f"refs/tags/{PREREGISTRATION_TAG}")
    peeled = _git_value("rev-parse", f"refs/tags/{PREREGISTRATION_TAG}^{{}}")
    if branch != PREREGISTRATION_BRANCH or commit != peeled:
        raise LifecycleGateError("local Git branch/tag does not identify this preregistration")
    for label, value in (("commit", commit), ("tag_object", tag_object), ("peeled_commit", peeled)):
        if type(value) is not str or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise LifecycleGateError(f"Git {label} is not a SHA-1 object identity")
    return {
        "branch": branch,
        "commit": commit,
        "annotated_tag": PREREGISTRATION_TAG,
        "tag_object": tag_object,
        "peeled_commit": peeled,
        "remote_verified_immediately_before_execution": True,
        "claim": "EXTERNAL_GITHUB_CUSTODY_RECORDED_NOT_RECONSTRUCTED_BY_RESULT_VALIDATOR",
    }


def _restart_worker(
    request_path: Path,
    output_root: Path,
    import_probe_path: Path,
) -> dict[str, Any]:
    request_payload = _read_regular(request_path, "restart request")
    request = _strict_json(request_payload)
    if request_payload != _canonical_json(request):
        raise LifecycleGateError("restart request is not canonical")
    request, midpoint = _validate_restart_request(request)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.exists() or root.is_symlink():
        raise LifecycleGateError("restart output root must be absent")
    root.mkdir(mode=0o700, parents=False)
    with _delegates() as (grid, _h64, d):
        fresh_child = d._fresh_child_interpreter_receipt()
        fresh_registry = d._reject_preloaded_runtime_modules()
        _probe_payload, probe = d._load_json(import_probe_path, "restart import probe")
        d._validate_import_probe_receipt(probe, expected_path=import_probe_path)
        custody = d.validate_execution_prerequisites(d.DE441_ROOT)
        bridge = d._load_json(d.BRIDGE_STATES_PATH, "bridge states")[1]
        with d._source_only_runtime_import_guard(
            root / ".restart-runtime-pycache-must-remain-absent"
        ) as guard:
            cp, _rebound, _target_receipt = d._load_dynamics_runtime(d.DEFAULT_REBOUND_TARGET)
            grid.construct_adapter_specs_v1(
                map_step_count=SEGMENT_STEPS,
                source_step_count=HORIZON_YEARS * STEPS_PER_YEAR,
            )
            runtime_base = _critical_runtime_base(
                d,
                cp,
                fresh_child=fresh_child,
                fresh_registry=fresh_registry,
            )
            stream, engine_projection = _restart_suffix(
                grid, d, cp, bridge, custody, midpoint
            )
        runtime = _critical_runtime_identity(
            d,
            runtime_base=runtime_base,
            import_guard=guard,
            engine_projection=engine_projection,
            expected_mode=RUNTIME_MODE_PRODUCTION,
        )
    receipt: dict[str, Any] = {
        "schema": RESTART_RECEIPT_SCHEMA,
        "request_content_sha256": request["content_sha256"],
        "request_file_sha256": _sha256(request_payload),
        "batch_order": RESTART_BATCH_ORDER,
        "object_id": RESTART_OBJECT_ID,
        "start_year": 500,
        "end_year": 1_000,
        "checkpoint_count": RESTART_RECORD_COUNT,
        "stream_byte_length": len(stream),
        "stream_sha256": _sha256(stream),
        "midpoint_consumed_sha256": _sha256(midpoint),
        "final_internal_state_sha256": _sha256(stream[-TRACER_INTERNAL_BYTES:]),
        "route_record_count": 0,
        "removal_record_count": 0,
        "runtime_content_sha256": runtime["content_sha256"],
        "runtime_identity": runtime,
        "parent_bitwise_match_checked": False,
        "content_sha256": "",
    }
    receipt["content_sha256"] = _domain_hash(RESTART_RECEIPT_DOMAIN, receipt)
    _write_create_only(root / "RESTART.bin", stream)
    _write_create_only(root / "RUNTIME.json", _canonical_json(runtime))
    _write_create_only(root / "RECEIPT.json", _canonical_json(receipt))
    return receipt


def _run_restart_child(
    d: types.ModuleType,
    root: Path,
    midpoint: bytes,
    main_suffix: bytes,
    import_probe_path: Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    if len(main_suffix) != RESTART_RECORD_COUNT * TRACER_INTERNAL_BYTES:
        raise LifecycleGateError("main restart suffix geometry differs")
    request = _restart_request(midpoint)
    with tempfile.TemporaryDirectory(prefix="jx-tno-1k-restart-", dir=root.parent) as directory:
        temporary = Path(directory)
        request_path = temporary / "REQUEST.json"
        child_root = temporary / "OUTPUT"
        _write_create_only(request_path, _canonical_json(request))
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "restart-worker",
                "--request", str(request_path), "--worker-output-root", str(child_root),
                "--import-probe-receipt", str(import_probe_path),
            ],
            cwd=PROJECT_ROOT,
            env=d._production_child_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=7_200,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-8_000:]
            raise LifecycleGateError(f"fresh restart worker failed: {detail}")
        if not child_root.is_dir() or {item.name for item in child_root.iterdir()} != {
            "RESTART.bin", "RUNTIME.json", "RECEIPT.json"
        }:
            raise LifecycleGateError("fresh restart output roster differs")
        resumed = _read_regular(
            child_root / "RESTART.bin", "fresh restart stream",
            RESTART_RECORD_COUNT * TRACER_INTERNAL_BYTES,
        )
        runtime_payload = _read_regular(child_root / "RUNTIME.json", "fresh restart runtime")
        runtime = _strict_json(runtime_payload)
        receipt_payload = _read_regular(child_root / "RECEIPT.json", "fresh restart receipt")
        receipt = _strict_json(receipt_payload)
        if runtime_payload != _canonical_json(runtime) or receipt_payload != _canonical_json(receipt):
            raise LifecycleGateError("fresh restart JSON is not canonical")
    if resumed != main_suffix:
        raise LifecycleGateError("fresh restart suffix differs bitwise from monolithic suffix")
    expected_receipt_keys = {
        "schema", "request_content_sha256", "request_file_sha256", "batch_order",
        "object_id", "start_year", "end_year", "checkpoint_count",
        "stream_byte_length", "stream_sha256", "midpoint_consumed_sha256",
        "final_internal_state_sha256", "route_record_count", "removal_record_count",
        "runtime_content_sha256", "runtime_identity",
        "parent_bitwise_match_checked", "content_sha256",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema") != RESTART_RECEIPT_SCHEMA
        or receipt.get("request_content_sha256") != request["content_sha256"]
        or receipt.get("request_file_sha256") != _sha256(_canonical_json(request))
        or type(receipt.get("batch_order")) is not int
        or receipt.get("batch_order") != RESTART_BATCH_ORDER
        or receipt.get("object_id") != RESTART_OBJECT_ID
        or type(receipt.get("start_year")) is not int
        or receipt.get("start_year") != 500
        or type(receipt.get("end_year")) is not int
        or receipt.get("end_year") != 1_000
        or type(receipt.get("checkpoint_count")) is not int
        or receipt.get("checkpoint_count") != RESTART_RECORD_COUNT
        or type(receipt.get("stream_byte_length")) is not int
        or receipt.get("stream_byte_length") != len(resumed)
        or receipt.get("stream_sha256") != _sha256(resumed)
        or receipt.get("midpoint_consumed_sha256") != _sha256(midpoint)
        or receipt.get("final_internal_state_sha256") != _sha256(resumed[-TRACER_INTERNAL_BYTES:])
        or type(receipt.get("route_record_count")) is not int
        or receipt.get("route_record_count") != 0
        or type(receipt.get("removal_record_count")) is not int
        or receipt.get("removal_record_count") != 0
        or receipt.get("parent_bitwise_match_checked") is not False
        or receipt.get("runtime_content_sha256") != runtime.get("content_sha256")
        or receipt.get("runtime_identity") != runtime
        or receipt.get("content_sha256") != _domain_hash(RESTART_RECEIPT_DOMAIN, receipt)
    ):
        raise LifecycleGateError("fresh restart receipt differs")
    receipt = dict(receipt)
    receipt["parent_bitwise_match_checked"] = True
    receipt["content_sha256"] = _domain_hash(RESTART_RECEIPT_DOMAIN, receipt)
    return resumed, receipt, runtime


def _publish_result_artifacts(root: Path, artifacts: Sequence[tuple[str, bytes]]) -> None:
    expected_names = [
        "PREFIX_STATES.bin", "RESTART_STREAMS.bin", "SEGMENTS.jsonl",
        "ROUTES.jsonl", "RUNTIME_PREFLIGHT.json", "RUNTIME.json", "RESULT.json",
    ]
    if [name for name, _ in artifacts] != expected_names:
        raise LifecycleGateError("publication artifact order differs")
    stage = Path(tempfile.mkdtemp(prefix=".tno-1k-publish-", dir=root.parent))
    try:
        for name, payload in artifacts:
            _write_create_only(stage / name, payload)
        with _held_directory(root, "publication root") as root_fd:
            if set(os.listdir(root_fd)) != {"PROTOCOL.json"}:
                raise LifecycleGateError("publication requires an exact prepared root")
            protocol_payload = _read_regular_at(
                root_fd, "PROTOCOL.json", "publication protocol"
            )
            publication_protocol = _strict_json(protocol_payload)
            if protocol_payload != _canonical_json(publication_protocol):
                raise LifecycleGateError("publication protocol is not canonical")
            validate_protocol(publication_protocol)
            linked: list[tuple[str, tuple[int, int]]] = []
            try:
                for name, _payload in artifacts:
                    source = stage / name
                    source_stat = source.stat(follow_symlinks=False)
                    os.link(source, name, dst_dir_fd=root_fd, follow_symlinks=False)
                    destination_stat = os.stat(
                        name, dir_fd=root_fd, follow_symlinks=False
                    )
                    if (
                        source_stat.st_dev != destination_stat.st_dev
                        or source_stat.st_ino != destination_stat.st_ino
                    ):
                        raise LifecycleGateError("publication hard-link identity differs")
                    linked.append(
                        (name, (destination_stat.st_dev, destination_stat.st_ino))
                    )
                if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
                    raise LifecycleGateError("publication final roster differs")
                os.fsync(root_fd)
            except Exception:
                for name, expected_identity in reversed(linked):
                    try:
                        observed = os.stat(
                            name, dir_fd=root_fd, follow_symlinks=False
                        )
                        if (observed.st_dev, observed.st_ino) == expected_identity:
                            os.unlink(name, dir_fd=root_fd)
                    except FileNotFoundError:
                        pass
                os.fsync(root_fd)
                raise
    finally:
        for item in stage.iterdir():
            item.unlink()
        stage.rmdir()
    with _held_directory(
        root, "published root", expected_roster=EXPECTED_OUTPUT_ROSTER
    ) as root_fd:
        for name in EXPECTED_OUTPUT_ROSTER:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LifecycleGateError("published output link count differs")
        os.fsync(root_fd)


def _execute_worker(
    output_root: Path,
    import_probe_path: Path,
    *,
    remote_verified: bool,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    protocol = validate_prepared(root)
    preflight = runtime_preflight()
    preflight_payload = _canonical_json(preflight)
    git_custody = _git_custody(remote_verified=remote_verified)
    retained_result = _load_retained_result()
    with _delegates() as (grid, _h64, d):
        _validate_runtime_preflight_document(preflight, d)
        fresh_child = d._fresh_child_interpreter_receipt()
        fresh_registry = d._reject_preloaded_runtime_modules()
        _probe_payload, probe = d._load_json(import_probe_path, "execution import probe")
        d._validate_import_probe_receipt(probe, expected_path=import_probe_path)
        custody = d.validate_execution_prerequisites(d.DE441_ROOT)
        bridge = d._load_json(d.BRIDGE_STATES_PATH, "bridge states")[1]
        prefix_builder = bytearray()
        segment_records: list[dict[str, Any]] = []
        lane_records: list[dict[str, Any]] = []
        midpoint: bytes | None = None
        main_restart_suffix: bytes | None = None
        main_engine_projection: dict[str, Any] | None = None
        with _held_source_file() as source_descriptor:
            with d._source_only_runtime_import_guard(
                root / ".lifecycle-runtime-pycache-must-remain-absent"
            ) as guard:
                cp, _rebound, _target_receipt = d._load_dynamics_runtime(d.DEFAULT_REBOUND_TARGET)
                grid.construct_adapter_specs_v1(
                    map_step_count=SEGMENT_STEPS,
                    source_step_count=HORIZON_YEARS * STEPS_PER_YEAR,
                )
                runtime_base = _critical_runtime_base(
                    d,
                    cp,
                    fresh_child=fresh_child,
                    fresh_registry=fresh_registry,
                )
                with grid.FrozenClassifierKernelV1() as kernel:
                    for batch_order in range(len(OBJECT_IDS)):
                        output = _run_batch(
                            grid, d, cp, bridge, custody, retained_result,
                            source_descriptor, kernel, batch_order,
                        )
                        prefix_builder.extend(output["checkpoint_payload"])
                        segment_records.extend(output["segments"])
                        lane_records.append(output["lane_record"])
                        if batch_order == RESTART_BATCH_ORDER:
                            midpoint = output["midpoint_payload"]
                            main_restart_suffix = output["main_restart_suffix"]
                            main_engine_projection = output["engine_projection"]
                if midpoint is None or main_restart_suffix is None or main_engine_projection is None:
                    raise LifecycleGateError("selected restart evidence was not captured")
            main_runtime = _critical_runtime_identity(
                d,
                runtime_base=runtime_base,
                import_guard=guard,
                engine_projection=main_engine_projection,
                expected_mode=RUNTIME_MODE_PRODUCTION,
            )
            if preflight["prelane_binding_projection_sha256"] != _sha256(
                _canonical_json(_runtime_prelane_binding_projection(main_runtime))
            ):
                raise LifecycleGateError(
                    "main runtime differs from the retained prelane binding"
                )
        resumed, restart_receipt, restart_runtime = _run_restart_child(
            d, root, midpoint, main_restart_suffix, import_probe_path
        )
    if _canonical_json(_runtime_comparison_projection(main_runtime)) != _canonical_json(
        _runtime_comparison_projection(restart_runtime)
    ):
        raise LifecycleGateError("fresh restart critical runtime identity differs")
    if preflight["prelane_binding_projection_sha256"] != _sha256(
        _canonical_json(_runtime_prelane_binding_projection(restart_runtime))
    ):
        raise LifecycleGateError(
            "restart runtime differs from the retained prelane binding"
        )
    prefix_payload = bytes(prefix_builder)
    restart_payload = main_restart_suffix + resumed
    if len(prefix_payload) != PREFIX_BYTES or len(restart_payload) != RESTART_STREAM_BYTES:
        raise LifecycleGateError("published binary geometry differs")
    segments_payload = b"".join(_canonical_json(item) for item in segment_records)
    routes_payload = b""
    runtime_payload = _canonical_json(main_runtime)
    file_records = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in (
            ("PREFIX_STATES.bin", prefix_payload),
            ("RESTART_STREAMS.bin", restart_payload),
            ("SEGMENTS.jsonl", segments_payload),
            ("ROUTES.jsonl", routes_payload),
            ("RUNTIME_PREFLIGHT.json", preflight_payload),
            ("RUNTIME.json", runtime_payload),
        )
    }
    result_claims = {
        **CLAIMS,
        "dynamics_executed": True,
        "one_thousand_year_lifecycle_gate_passed": True,
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED",
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": git_custody,
        "files": file_records,
        "batch_count": len(lane_records),
        "segment_count": len(segment_records),
        "route_record_count": 0,
        "removal_record_count": 0,
        "batch_lanes": lane_records,
        "restart": {
            "batch_order": RESTART_BATCH_ORDER,
            "object_id": RESTART_OBJECT_ID,
            "record_count_per_stream": RESTART_RECORD_COUNT,
            "monolithic_stream_sha256": _sha256(main_restart_suffix),
            "resumed_stream_sha256": _sha256(resumed),
            "bitwise_equal": True,
            "worker_receipt": restart_receipt,
        },
        "runtime_content_sha256": main_runtime["content_sha256"],
        "runtime_preflight_content_sha256": preflight["content_sha256"],
        "all_predeclared_gates_passed": True,
        "claims": result_claims,
        "content_sha256": "",
    }
    result["content_sha256"] = _domain_hash(RESULT_DOMAIN, result)
    _publish_result_artifacts(
        root,
        (
            ("PREFIX_STATES.bin", prefix_payload),
            ("RESTART_STREAMS.bin", restart_payload),
            ("SEGMENTS.jsonl", segments_payload),
            ("ROUTES.jsonl", routes_payload),
            ("RUNTIME_PREFLIGHT.json", preflight_payload),
            ("RUNTIME.json", runtime_payload),
            ("RESULT.json", _canonical_json(result)),
        ),
    )
    return result


def _runtime_comparison_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or value.get("schema") != RUNTIME_SCHEMA:
        raise LifecycleGateError("runtime identity shape differs")
    projection = _plain(value)
    projection.pop("content_sha256", None)
    projection.pop("source_only_runtime_import_guard", None)
    numpy_identity = projection.get("numpy_runtime_identity")
    if type(numpy_identity) is not dict:
        raise LifecycleGateError("runtime NumPy comparison identity differs")
    for key in ("module", "core_module"):
        module = numpy_identity.get(key)
        if type(module) is not dict or "cached_path" not in module:
            raise LifecycleGateError("runtime NumPy module comparison identity differs")
        module.pop("cached_path")
    return projection


def _runtime_prelane_binding_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projection = _runtime_comparison_projection(value)
    engine = projection.pop("engine_runtime_projection", None)
    hybrid = (
        engine.get("hybrid_runtime_identity_without_source_tile")
        if type(engine) is dict else None
    )
    quiet = hybrid.get("quiet_cuda_runtime_identity") if type(hybrid) is dict else None
    if type(quiet) is not dict:
        raise LifecycleGateError("runtime prelane CUDA binding is absent")
    projection["prelane_quiet_cuda_runtime_identity"] = quiet
    return projection


def _read_output_file(root: Path, name: str, expected_bytes: int | None = None) -> bytes:
    path = root / name
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise LifecycleGateError(f"retained output is not a single-link regular file: {name}")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise LifecycleGateError(f"retained output byte length differs: {name}")
    return _read_regular(path, f"retained {name}", metadata.st_size)


def _parse_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise LifecycleGateError(f"{label} lacks a final line feed")
    records: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        value = _strict_json(line)
        if line != _canonical_json(value):
            raise LifecycleGateError(f"{label} contains a noncanonical row")
        records.append(value)
    return records


def _validate_output_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    protocol_payload = snapshot["PROTOCOL.json"]
    prefix = snapshot["PREFIX_STATES.bin"]
    restart = snapshot["RESTART_STREAMS.bin"]
    segments_payload = snapshot["SEGMENTS.jsonl"]
    routes_payload = snapshot["ROUTES.jsonl"]
    preflight_payload = snapshot["RUNTIME_PREFLIGHT.json"]
    runtime_payload = snapshot["RUNTIME.json"]
    result_payload = snapshot["RESULT.json"]
    protocol = _strict_json(protocol_payload)
    if protocol_payload != _canonical_json(protocol):
        raise LifecycleGateError("retained protocol is not canonical")
    validate_protocol(protocol)
    preflight = _strict_json(preflight_payload)
    runtime = _strict_json(runtime_payload)
    result = _strict_json(result_payload)
    if (
        preflight_payload != _canonical_json(preflight)
        or runtime_payload != _canonical_json(runtime)
        or result_payload != _canonical_json(result)
    ):
        raise LifecycleGateError(
            "retained preflight/runtime/result JSON is not canonical"
        )
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("content_sha256") != _domain_hash(RUNTIME_DOMAIN, runtime)
    ):
        raise LifecycleGateError("retained runtime identity differs")
    with _delegates() as (_grid, _h64, runtime_delegate):
        _validate_runtime_preflight_document(preflight, runtime_delegate)
        _validate_runtime_identity(
            runtime,
            runtime_delegate,
            "retained main",
            expected_mode=RUNTIME_MODE_PRODUCTION,
        )
    if preflight["prelane_binding_projection_sha256"] != _sha256(
        _canonical_json(_runtime_prelane_binding_projection(runtime))
    ):
        raise LifecycleGateError("retained main runtime differs from preflight")
    result_keys = {
        "schema", "status", "protocol_content_sha256", "protocol_file_sha256",
        "git_preregistration", "files", "batch_count", "segment_count",
        "route_record_count", "removal_record_count", "batch_lanes", "restart",
        "runtime_content_sha256", "runtime_preflight_content_sha256",
        "all_predeclared_gates_passed", "claims", "content_sha256",
    }
    if (
        set(result) != result_keys
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status")
        != "ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED"
        or result.get("protocol_content_sha256") != protocol["content_sha256"]
        or result.get("protocol_file_sha256") != _sha256(protocol_payload)
        or type(result.get("batch_count")) is not int
        or result.get("batch_count") != len(OBJECT_IDS)
        or type(result.get("segment_count")) is not int
        or result.get("segment_count") != len(OBJECT_IDS) * SEGMENT_COUNT
        or type(result.get("route_record_count")) is not int
        or result.get("route_record_count") != 0
        or type(result.get("removal_record_count")) is not int
        or result.get("removal_record_count") != 0
        or result.get("runtime_content_sha256") != runtime["content_sha256"]
        or result.get("runtime_preflight_content_sha256")
        != preflight["content_sha256"]
        or result.get("all_predeclared_gates_passed") is not True
        or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result)
    ):
        raise LifecycleGateError("retained result identity differs")
    expected_claims = {
        **CLAIMS,
        "dynamics_executed": True,
        "one_thousand_year_lifecycle_gate_passed": True,
    }
    _exact_claims(result.get("claims"), expected_claims, "result")
    files = result.get("files")
    retained_files = {
        "PREFIX_STATES.bin": prefix,
        "RESTART_STREAMS.bin": restart,
        "SEGMENTS.jsonl": segments_payload,
        "ROUTES.jsonl": routes_payload,
        "RUNTIME_PREFLIGHT.json": preflight_payload,
        "RUNTIME.json": runtime_payload,
    }
    if type(files) is not dict or set(files) != set(retained_files):
        raise LifecycleGateError("retained file receipt roster differs")
    for name, payload in retained_files.items():
        receipt = files.get(name)
        if (
            type(receipt) is not dict or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt.get("bytes") != len(payload)
            or receipt.get("sha256") != _sha256(payload)
        ):
            raise LifecycleGateError(f"retained file receipt differs: {name}")
    if routes_payload != b"":
        raise LifecycleGateError("successful lifecycle gate retained a route record")

    segments = _parse_jsonl(segments_payload, "segments")
    if len(segments) != len(OBJECT_IDS) * SEGMENT_COUNT:
        raise LifecycleGateError("segment receipt count differs")
    retained_result = _load_retained_result()
    with _held_source_file() as source_descriptor:
        for order, record in enumerate(segments):
            _validate_segment_record(record, expected_order=order)
            batch_order = order // SEGMENT_COUNT
            segment_index = order % SEGMENT_COUNT
            checkpoint_offset = (
                (batch_order * len(CHECKPOINT_YEARS) + segment_index + 1)
                * CHECKPOINT_BYTES
            )
            if record["checkpoint_sha256"] != _sha256(
                prefix[checkpoint_offset : checkpoint_offset + CHECKPOINT_BYTES]
            ):
                raise LifecycleGateError("segment checkpoint digest differs from prefix payload")
            _lane, source_record = _retained_lane_records(
                retained_result, batch_order
            )
            retained_segment = _retained_source_segment(
                source_descriptor, source_record, segment_index
            )
            expected_segment_sha256 = _sha256(retained_segment)
            if (
                record["generated_source_segment_sha256"]
                != expected_segment_sha256
                or record["retained_source_segment_sha256"]
                != expected_segment_sha256
            ):
                raise LifecycleGateError(
                    "segment source digest differs from held retained source evidence"
                )
    prefix_receipts: list[dict[str, Any]] = []
    with _delegates() as (grid, _h64, _d):
        for batch_order in range(len(OBJECT_IDS)):
            start = batch_order * len(CHECKPOINT_YEARS) * CHECKPOINT_BYTES
            checkpoints = tuple(
                (
                    year,
                    prefix[
                        start + index * CHECKPOINT_BYTES :
                        start + (index + 1) * CHECKPOINT_BYTES
                    ],
                )
                for index, year in enumerate(CHECKPOINT_YEARS)
            )
            receipt = grid.validate_first_1000_year_prefix_v1(batch_order, checkpoints)
            if receipt.get("passed") is not True:
                raise LifecycleGateError("offline prefix crossbinding did not pass")
            prefix_receipts.append(receipt)

    half = RESTART_RECORD_COUNT * TRACER_INTERNAL_BYTES
    monolithic = restart[:half]
    resumed = restart[half:]
    if monolithic != resumed:
        raise LifecycleGateError("retained restart streams differ bitwise")
    _restart_lane, restart_source_record = _retained_lane_records(
        retained_result, RESTART_BATCH_ORDER
    )
    with _held_source_file() as source_descriptor:
        with _delegates() as (grid, _h64, _d):
            _crossbind_restart_stream_to_prefix(
                grid,
                monolithic,
                prefix,
                source_descriptor,
                restart_source_record,
            )
    restart_record = result.get("restart")
    restart_keys = {
        "batch_order", "object_id", "record_count_per_stream",
        "monolithic_stream_sha256", "resumed_stream_sha256", "bitwise_equal",
        "worker_receipt",
    }
    if (
        type(restart_record) is not dict or set(restart_record) != restart_keys
        or type(restart_record.get("batch_order")) is not int
        or restart_record.get("batch_order") != RESTART_BATCH_ORDER
        or restart_record.get("object_id") != RESTART_OBJECT_ID
        or type(restart_record.get("record_count_per_stream")) is not int
        or restart_record.get("record_count_per_stream") != RESTART_RECORD_COUNT
        or restart_record.get("monolithic_stream_sha256") != _sha256(monolithic)
        or restart_record.get("resumed_stream_sha256") != _sha256(resumed)
        or restart_record.get("bitwise_equal") is not True
    ):
        raise LifecycleGateError("retained restart summary differs")
    worker = restart_record.get("worker_receipt")
    expected_worker_keys = {
        "schema", "request_content_sha256", "request_file_sha256",
        "batch_order", "object_id", "start_year", "end_year",
        "checkpoint_count", "stream_byte_length", "stream_sha256",
        "midpoint_consumed_sha256", "final_internal_state_sha256",
        "route_record_count", "removal_record_count",
        "runtime_content_sha256", "runtime_identity",
        "parent_bitwise_match_checked", "content_sha256",
    }
    midpoint = monolithic[:TRACER_INTERNAL_BYTES]
    expected_request = _restart_request(midpoint)
    worker_runtime = worker.get("runtime_identity") if type(worker) is dict else None
    if (
        type(worker) is not dict or set(worker) != expected_worker_keys
        or worker.get("schema") != RESTART_RECEIPT_SCHEMA
        or worker.get("request_content_sha256")
        != expected_request["content_sha256"]
        or worker.get("request_file_sha256")
        != _sha256(_canonical_json(expected_request))
        or type(worker.get("batch_order")) is not int
        or worker.get("batch_order") != RESTART_BATCH_ORDER
        or worker.get("object_id") != RESTART_OBJECT_ID
        or type(worker.get("start_year")) is not int
        or worker.get("start_year") != 500
        or type(worker.get("end_year")) is not int
        or worker.get("end_year") != 1_000
        or type(worker.get("checkpoint_count")) is not int
        or worker.get("checkpoint_count") != RESTART_RECORD_COUNT
        or type(worker.get("stream_byte_length")) is not int
        or worker.get("stream_byte_length") != len(resumed)
        or worker.get("stream_sha256") != _sha256(resumed)
        or worker.get("midpoint_consumed_sha256") != _sha256(midpoint)
        or worker.get("final_internal_state_sha256")
        != _sha256(resumed[-TRACER_INTERNAL_BYTES:])
        or type(worker.get("route_record_count")) is not int
        or worker.get("route_record_count") != 0
        or type(worker.get("removal_record_count")) is not int
        or worker.get("removal_record_count") != 0
        or type(worker_runtime) is not dict
        or worker.get("runtime_content_sha256")
        != worker_runtime.get("content_sha256")
        or worker.get("parent_bitwise_match_checked") is not True
        or worker.get("content_sha256") != _domain_hash(RESTART_RECEIPT_DOMAIN, worker)
    ):
        raise LifecycleGateError("retained restart worker receipt differs")
    with _delegates() as (_grid, _h64, runtime_delegate):
        _validate_runtime_identity(
            worker_runtime,
            runtime_delegate,
            "retained restart worker",
            expected_mode=RUNTIME_MODE_PRODUCTION,
        )
    if _canonical_json(
        _runtime_comparison_projection(runtime)
    ) != _canonical_json(_runtime_comparison_projection(worker_runtime)):
        raise LifecycleGateError("retained main/restart runtime identities differ")
    if preflight["prelane_binding_projection_sha256"] != _sha256(
        _canonical_json(_runtime_prelane_binding_projection(worker_runtime))
    ):
        raise LifecycleGateError("retained restart runtime differs from preflight")

    git = result.get("git_preregistration")
    git_keys = {
        "branch", "commit", "annotated_tag", "tag_object", "peeled_commit",
        "remote_verified_immediately_before_execution", "claim",
    }
    if (
        type(git) is not dict or set(git) != git_keys
        or git.get("branch") != PREREGISTRATION_BRANCH
        or git.get("annotated_tag") != PREREGISTRATION_TAG
        or git.get("commit") != git.get("peeled_commit")
        or git.get("remote_verified_immediately_before_execution") is not True
        or git.get("claim")
        != "EXTERNAL_GITHUB_CUSTODY_RECORDED_NOT_RECONSTRUCTED_BY_RESULT_VALIDATOR"
    ):
        raise LifecycleGateError("retained Git preregistration record differs")
    for key in ("commit", "tag_object", "peeled_commit"):
        value = git.get(key)
        if type(value) is not str or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise LifecycleGateError("retained Git object identity differs")

    lanes = result.get("batch_lanes")
    if type(lanes) is not list or len(lanes) != len(OBJECT_IDS):
        raise LifecycleGateError("retained batch lane roster differs")
    for batch_order, lane_record in enumerate(lanes):
        retained_lane, retained_source = _retained_lane_records(retained_result, batch_order)
        lane_keys = {
            "batch_order", "object_id", "source_order", "full_source_tile_sha256",
            "generated_source_payload_sha256", "retained_source_payload_sha256",
            "source_build_input_content_sha256",
            "source_map_result_content_sha256",
            "source_map_verification_content_sha256",
            "source_build_runtime_identity_sha256", "source_invariants",
            "prefix_crossbinding", "segment_count", "route_record_count",
            "removal_record_count", "elapsed_seconds_hex",
        }
        if (
            type(lane_record) is not dict or set(lane_record) != lane_keys
            or type(lane_record.get("batch_order")) is not int
            or lane_record.get("batch_order") != batch_order
            or lane_record.get("object_id") != OBJECT_IDS[batch_order]
            or type(lane_record.get("source_order")) is not int
            or lane_record.get("source_order") != 12 + batch_order
            or lane_record.get("full_source_tile_sha256")
            != retained_source["full_source_tile_sha256"]
            or lane_record.get("generated_source_payload_sha256")
            != retained_source["lane_payload_sha256"]
            or lane_record.get("retained_source_payload_sha256")
            != retained_source["lane_payload_sha256"]
            or lane_record.get("source_invariants") != {
                key: retained_lane["summary"][key]
                for key in (
                    "massive_invariant_sampling", "massive_invariant_sample_count",
                    "maximum_relative_massive_energy_drift_hex",
                    "maximum_relative_massive_energy_drift_exact",
                    "maximum_relative_massive_angular_momentum_drift_hex",
                )
            }
            or lane_record.get("prefix_crossbinding") != prefix_receipts[batch_order]
            or type(lane_record.get("segment_count")) is not int
            or lane_record.get("segment_count") != SEGMENT_COUNT
            or type(lane_record.get("route_record_count")) is not int
            or lane_record.get("route_record_count") != 0
            or type(lane_record.get("removal_record_count")) is not int
            or lane_record.get("removal_record_count") != 0
        ):
            raise LifecycleGateError("retained batch lane evidence differs")
        for key in (
            "source_build_input_content_sha256",
            "source_map_result_content_sha256",
            "source_map_verification_content_sha256",
            "source_build_runtime_identity_sha256",
        ):
            if not _is_sha256(lane_record.get(key)):
                raise LifecycleGateError("retained batch lane digest differs")
        elapsed = lane_record.get("elapsed_seconds_hex")
        try:
            elapsed_value = float.fromhex(elapsed) if type(elapsed) is str else math.nan
        except ValueError as exc:
            raise LifecycleGateError("retained batch lane elapsed time differs") from exc
        if not math.isfinite(elapsed_value) or elapsed_value < 0.0:
            raise LifecycleGateError("retained batch lane elapsed time differs")
    return result


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    expected_lengths = {
        "PROTOCOL.json": None,
        "PREFIX_STATES.bin": PREFIX_BYTES,
        "RESTART_STREAMS.bin": RESTART_STREAM_BYTES,
        "SEGMENTS.jsonl": None,
        "ROUTES.jsonl": 0,
        "RUNTIME_PREFLIGHT.json": None,
        "RUNTIME.json": None,
        "RESULT.json": None,
    }
    with _held_directory(
        root, "retained output root", expected_roster=EXPECTED_OUTPUT_ROSTER
    ) as root_fd:
        snapshot: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int, int]] = {}
        for name in sorted(EXPECTED_OUTPUT_ROSTER):
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            identities[name] = (
                metadata.st_dev, metadata.st_ino, metadata.st_size,
                metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
            )
            snapshot[name] = _read_regular_at(
                root_fd, name, f"retained {name}", expected_lengths[name]
            )
        result = _validate_output_snapshot(snapshot)
        if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
            raise LifecycleGateError("retained output root final roster differs")
        for name, expected_identity in identities.items():
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            observed_identity = (
                metadata.st_dev, metadata.st_ino, metadata.st_size,
                metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
            )
            if observed_identity != expected_identity:
                raise LifecycleGateError(
                    f"retained output changed during semantic validation: {name}"
                )
        return result


def runtime_preflight() -> dict[str, Any]:
    _bound_inputs()
    with _delegates() as (_grid, _h64, d):
        d._reject_preloaded_runtime_modules()
        d.validate_execution_prerequisites(d.DE441_ROOT)
        with tempfile.TemporaryDirectory(prefix="jx-tno-1k-preflight-parent-") as raw:
            probe_path, _probe = d._run_import_only_preflight(
                d.DEFAULT_REBOUND_TARGET, Path(raw)
            )
            receipts: list[dict[str, Any]] = []
            challenge_seed = os.urandom(32)
            challenges = [
                _sha256(challenge_seed + index.to_bytes(1, "big"))
                for index in range(2)
            ]
            for worker_order in range(2):
                completed = subprocess.run(
                    [
                        str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                        "runtime-preflight-worker", "--import-probe-receipt",
                        str(probe_path), "--worker-order", str(worker_order),
                        "--worker-challenge-sha256", challenges[worker_order],
                    ],
                    cwd=PROJECT_ROOT,
                    env=d._production_child_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=600,
                )
                if completed.returncode != 0:
                    detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
                    raise LifecycleGateError(
                        f"fresh runtime-preflight worker failed: {detail}"
                    )
                receipt = _strict_json(completed.stdout)
                if completed.stdout != _canonical_json(receipt):
                    raise LifecycleGateError(
                        "runtime-preflight worker output is not canonical"
                    )
                receipts.append(
                    _validate_runtime_preflight_worker_receipt(
                        receipt, d, f"worker {len(receipts)}"
                    )
                )
        projections = [
            _canonical_json(
                _runtime_comparison_projection(item["runtime_identity"])
            )
            for item in receipts
        ]
        if projections[0] != projections[1]:
            raise LifecycleGateError("fresh-child runtime projections differ")
        binding_projections = [
            _canonical_json(
                _runtime_prelane_binding_projection(item["runtime_identity"])
            )
            for item in receipts
        ]
        if binding_projections[0] != binding_projections[1]:
            raise LifecycleGateError("fresh-child prelane runtime bindings differ")
        document: dict[str, Any] = {
            "schema": RUNTIME_PREFLIGHT_SCHEMA,
            "worker_receipts": receipts,
            "process_independent_projection_sha256": _sha256(projections[0]),
            "prelane_binding_projection_sha256": _sha256(binding_projections[0]),
            "worker_challenge_sha256s": challenges,
            "fresh_child_count": 2,
            "source_step_count": 0,
            "tracer_count": 0,
            "dynamics_executed": False,
            "qualification_result": False,
            "classification_generated": False,
            "science_result": False,
            "content_sha256": "",
        }
        document["content_sha256"] = _domain_hash(
            RUNTIME_PREFLIGHT_DOMAIN, document
        )
        return _validate_runtime_preflight_document(document, d)


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    remote_verified: bool = False,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    validate_prepared(root)
    _bound_inputs()
    with _delegates() as (_grid, _h64, d):
        d._reject_preloaded_runtime_modules()
        d.validate_execution_prerequisites(d.DE441_ROOT)
        with tempfile.TemporaryDirectory(prefix="jx-tno-1k-import-probe-") as directory:
            probe_path, _probe = d._run_import_only_preflight(
                d.DEFAULT_REBOUND_TARGET, Path(directory)
            )
            completed = subprocess.run(
                [
                    str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "execute-worker",
                    "--output-root", str(root),
                    "--import-probe-receipt", str(probe_path),
                    *( ["--remote-verified"] if remote_verified else [] ),
                ],
                cwd=PROJECT_ROOT,
                env=d._production_child_environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=7_200,
            )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
        raise LifecycleGateError(f"fresh lifecycle execution worker failed: {detail}")
    return validate_output(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build", "prepare", "validate-prepared", "execute", "validate",
            "runtime-preflight", "runtime-preflight-worker", "execute-worker",
            "restart-worker",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--import-probe-receipt", type=Path)
    parser.add_argument("--worker-order", type=int)
    parser.add_argument("--worker-challenge-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--worker-output-root", type=Path)
    parser.add_argument("--remote-verified", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        sys.stdout.buffer.write(_canonical_json(build_protocol()))
    elif args.command == "prepare":
        prepare(args.output_root)
    elif args.command == "validate-prepared":
        validate_prepared(args.output_root)
    elif args.command == "execute":
        execute(args.output_root, remote_verified=args.remote_verified)
    elif args.command == "validate":
        validate_output(args.output_root)
    elif args.command == "runtime-preflight":
        sys.stdout.buffer.write(_canonical_json(runtime_preflight()))
    elif args.command == "runtime-preflight-worker":
        if (
            args.import_probe_receipt is None
            or args.worker_order is None
            or args.worker_challenge_sha256 is None
        ):
            raise LifecycleGateError(
                "runtime-preflight-worker requires import probe, order, and challenge"
            )
        sys.stdout.buffer.write(
            _canonical_json(
                _runtime_preflight_worker(
                    args.import_probe_receipt,
                    worker_order=args.worker_order,
                    parent_challenge_sha256=args.worker_challenge_sha256,
                )
            )
        )
    elif args.command == "execute-worker":
        if args.import_probe_receipt is None:
            raise LifecycleGateError(
                "execute-worker requires an import-probe receipt"
            )
        _execute_worker(
            args.output_root,
            args.import_probe_receipt,
            remote_verified=args.remote_verified,
        )
    elif args.command == "restart-worker":
        if (
            args.request is None or args.worker_output_root is None
            or args.import_probe_receipt is None
        ):
            raise LifecycleGateError(
                "restart-worker requires request, output root, and import probe"
            )
        _restart_worker(
            args.request, args.worker_output_root, args.import_probe_receipt
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
