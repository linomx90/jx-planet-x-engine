#!/usr/bin/env python3
"""Prepare, execute, and replay the bounded full-129 TNO GPU smoke.

The only dynamics authorized here are batch zero, 129 massless tracers, the
frozen DE441 reduced-five source path, h=1/64 Julian year, and 0..1,000 years
in twenty 50-year lifecycle segments.  This is an engineering qualification
artifact, not a TNO classification or a 10 Myr science result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v1"
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

PREFLIGHT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_production_preflight_v1.py"
PREFLIGHT_BYTES = 58_642
PREFLIGHT_SHA256 = "40a8f6a330792f9d238907a3b9db295ce6de8bf02bfa0fa174d3f89dc90260ad"
PREFLIGHT_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_production_preflight_v1.py"
PREFLIGHT_TEST_BYTES = 16_670
PREFLIGHT_TEST_SHA256 = "de1c6034f903ba1977183a1af7a4f0f926d3feddb329965511c1898c9788f54a"
PREFLIGHT_PROTOCOL_BYTES = 29_086
PREFLIGHT_PROTOCOL_FILE_SHA256 = "7d9f8d80b18e00911d2826977ded05eb43f64ca0b2fd3ad7e1ca97dbbdb1bd90"
PREFLIGHT_PROTOCOL_CONTENT_SHA256 = "b54879ee2f146b9656ffc16e8863b26ee1ddcef6b7982d1131d543117f1328cf"
PREFLIGHT_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_10my_full128_production_preflight_v1" / "PROTOCOL.json"
PREFLIGHT_PRIMARY_AGGREGATE_SHA256 = "0537365c46dd373aca72b04b8c8d2225aed42a60ac9ab1c8c803dcfb6c548bbe"
PREFLIGHT_CHECKPOINT_AGGREGATE_SHA256 = "dc7a009b0c538649fd9dd9c225c074edf2106e1962cca9d24e777a4d421ff961"

FULL_BRIDGE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_full128_reduced5_bridge_v1.py"
FULL_BRIDGE_BYTES = 68_486
FULL_BRIDGE_SHA256 = "7b6e0ee72618410da95639628c74feb5113e62a202d6d51c4a103795fcd8a4df"
FULL_BRIDGE_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_full128_reduced5_bridge_v1.py"
FULL_BRIDGE_TEST_BYTES = 18_430
FULL_BRIDGE_TEST_SHA256 = "df375289cce1a8f66dae0c060a1b952ad38d55c3fe5bf83f5cf78ab372a9494d"
FULL_BRIDGE_ROOT = PROJECT_ROOT / "runs" / "tno_full128_reduced5_bridge_v1"
FULL_BRIDGE_PROTOCOL_BYTES = 10_469
FULL_BRIDGE_PROTOCOL_SHA256 = "a97ce6daefd2417606219744cb168ab3dcb1cb2de676a1f1671bef10493b5ccb"
FULL_BRIDGE_PROTOCOL_CONTENT_SHA256 = "0458f700c64e798ecbfa927e6d722cd6d3b2b4ae8d6f70765f3fa2ea5684a0cd"
FULL_BRIDGE_STATES_BYTES = 1_208_834
FULL_BRIDGE_STATES_SHA256 = "0570e7d81917caad871de2b2a143d4fc083851894d3c72aab1b77f8e9591fcec"
FULL_BRIDGE_STATES_CONTENT_SHA256 = "2c095be924f10481ced30af205836ebd18b4113398a7688b77f06259a2ac780b"

TRAJECTORY_GRID_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_trajectory_grid_v1.py"
TRAJECTORY_GRID_BYTES = 77_247
TRAJECTORY_GRID_SHA256 = "3dbc7ad7e5825b863c47b17ad0311e4986538df98d26ea7545b9fb53bb4562d0"
TRAJECTORY_GRID_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_trajectory_grid_v1.py"
TRAJECTORY_GRID_TEST_BYTES = 22_002
TRAJECTORY_GRID_TEST_SHA256 = "083f68468639183697b1b2ae0df2b3c6e300817d0e8079aab7ff5a84234f9c22"
TRAJECTORY_GRID_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_trajectory_grid_v1"
TRAJECTORY_GRID_PROTOCOL_BYTES = 8_234
TRAJECTORY_GRID_PROTOCOL_SHA256 = "fd8106e150bdf174f631ee25bc30f38bf09e5f006c4fd2e8bf26866f8b4e7e93"
TRAJECTORY_GRID_PROTOCOL_CONTENT_SHA256 = "da6dce31b784f50f96e40b33ddb39d78a4421c47329095dc915a507a1996ad49"

V3_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_lifecycle_gate_v3.py"
V3_BYTES = 151_995
V3_SHA256 = "02e72fb6a635b74dd31dc364e18cbaf03c5b9e1833caf5aba4a063d7198e47c7"
V3_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_1kyr_reduced5_lifecycle_gate_v3.py"
V3_TEST_BYTES = 42_017
V3_TEST_SHA256 = "b4eb43b00e1fab0055a43cc4f26db2bef6bf29d51656bcfab1d792596b7845a8"
V3_ROOT = PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v3"
V3_PROTOCOL_SHA256 = "9c1e7f77b3a2fd9f6af5542aa2ee3fa7dc00d669d496c809da28b675c8519199"
V3_PREFIX_SHA256 = "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b"
V3_RESTART_SHA256 = "3330f6d83b71f32c2a0fb14bc9f2e26ab13def77f1ea17527d69552473c80173"
V3_SEGMENTS_SHA256 = "e9452debe803a7213d0878c6c2bd35199bfe966e7e6891f12d072113c825f64f"
V3_RUNTIME_PREFLIGHT_SHA256 = "7df113a70bcb1012ea05b2145f141bd8b788c504fc61c0f2b075671d0810ea80"
V3_RUNTIME_SHA256 = "43bc1edba432bed75f8a9eb1db9784c13116e1a93e20bb4b43cb634d6abd1dcd"
V3_RESULT_SHA256 = "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4"
V3_RESULT_CONTENT_SHA256 = "d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264"
H64_SOURCE_RELATIVE_PATH = "runs/tno_stage1_jx_h64_closure_v2/JX_SOURCE_NODES.bin"
H64_SOURCE_PATH = PROJECT_ROOT / H64_SOURCE_RELATIVE_PATH
H64_SOURCE_BYTES = 368_645_800
H64_SOURCE_SHA256 = "fd9b51e18aa6bb5fac095799955ecc6b3d6afeb2253b1000e2def41ccf376dbd"

ENGINE_FILES = (
    ("src/jxplanetx/restricted_wh_source.py", 61_803, "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52"),
    ("src/jxplanetx/cuda_restricted_wh.py", 74_758, "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927"),
    ("src/jxplanetx/cuda_restricted_wh_hybrid.py", 37_420, "6b3ea46436dbdb7dc2c434cff005a88395fea6e00cd97c7be104650c75769b10"),
    ("src/jxplanetx/restricted_wh_route.py", 38_194, "7eae131cda32ee3feed525b13226b1b271b0fc54c1dd5d6a82927b4dc4b94075"),
    ("src/jxplanetx/restricted_wh_lifecycle.py", 38_996, "b0211778fcf51dbfef78049e74bafd05d5daee4db7dd18f743f4a5b5a14af346"),
    ("src/jxplanetx/cuda_tracer_events.py", 67_504, "a3c5f1f9de29f504420aa892bd50f8a947b24d6df21cadf91dc243c7976dad89"),
    ("src/jxplanetx/fast_wisdom_holman.py", 36_182, "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72"),
    ("src/jxplanetx/_all_conic_kepler.py", 37_373, "3d38c569764957bc0c781a4c2a7c767ccf2b630a38b0c88688e9716ac2b79691"),
    ("src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so", 42_520, "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3"),
    ("src/jxplanetx/engine/backends.py", 7_367, "c039ae04aab43c0809b200304a47e9d6e63bb7cbea76e0665724c75ab513deba"),
    ("src/jxplanetx/engine/contracts.py", 16_390, "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d"),
)

PROTOCOL_SCHEMA = "jx.tno.10my-full128-gpu-smoke.protocol.v1"
SEGMENT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.segment.v1"
RUNTIME_PRE_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-pre.v1"
RUNTIME_POST_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-post.v1"
RESULT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.result.v1"
GIT_CUSTODY_SCHEMA = "jx.tno.10my-full128-gpu-smoke.git-custody.v1"
PROTOCOL_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.protocol.v1\0"
SEGMENT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.segment.v1\0"
RUNTIME_PRE_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-pre.v1\0"
RUNTIME_POST_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-post.v1\0"
RESULT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.result.v1\0"
GIT_CUSTODY_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.git-custody.v1\0"

PREREGISTRATION_BRANCH = "research/tno-10my-reduced5-execution-v1-preregistered-20261011"
PREREGISTRATION_TAG = "tno-10my-reduced5-execution-v1-preregistered-20261011"
LOCAL_THREAT_BOUNDARY = "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"

BATCH_ORDER = 0
OBJECT_ID = "jpl-20090377"
SPK_ID = 20_090_377
MASSIVE_COUNT = 5
TRACER_COUNT = 129
BODY_COUNT = MASSIVE_COUNT + TRACER_COUNT
STATE_RECORD_BYTES = BODY_COUNT * 6 * 8
CHECKPOINT_YEARS = tuple(range(0, 1_001, 50))
CHECKPOINT_COUNT = len(CHECKPOINT_YEARS)
CHECKPOINTS_BYTES = CHECKPOINT_COUNT * STATE_RECORD_BYTES
LIFECYCLE_BYTES = CHECKPOINT_COUNT * TRACER_COUNT
CONTINUATION_RECORD_BYTES = TRACER_COUNT * 6 * 8
CONTINUATION_BYTES = CHECKPOINT_COUNT * CONTINUATION_RECORD_BYTES
STEPS_PER_YEAR = 64
SEGMENT_YEARS = 50
SEGMENT_STEPS = SEGMENT_YEARS * STEPS_PER_YEAR
SEGMENT_COUNT = 20
SOURCE_COUNT = 5
SOURCE_NODE_BYTES = SOURCE_COUNT * 3 * 8
YEAR_ZERO_CHECKPOINT_SHA256 = "25394a0f2a31caf3c9b47f593ec0511806b94a0ac140d1d44fc9e1939656e00c"
YEAR_ZERO_PRIMARY_SHA256 = "834c03a3cbbb1e7a397d69fb7f92f710f8d7fa9958d91d4efd6ed342bab4c4da"
YEAR_ZERO_DIRECT_SHA256 = "2662b11dafe2acd679d1a91f8902abdc941ffb0c458cd6a22f5cdc3182eea7aa"
EXPECTED_OUTPUT_ROSTER = frozenset({
    "PROTOCOL.json", "CHECKPOINTS.bin", "LIFECYCLE.bin", "CONTINUATION.bin",
    "RUNTIME_PRE.json", "RUNTIME_POST.json", "SEGMENTS.jsonl", "RESULT.json",
})

CLAIMS_PREPARED = {
    "bounded_gpu_smoke_executed": False,
    "bounded_gpu_smoke_passed": False,
    "gpu_executed": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "classification_generated": False,
    "robust_128_clone_classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class Full128SmokeError(RuntimeError):
    """The bounded full-129 smoke contract failed closed."""


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise Full128SmokeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise Full128SmokeError("value is not canonical-JSON encodable") from exc


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise Full128SmokeError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Full128SmokeError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Full128SmokeError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise Full128SmokeError("JSON root must be an exact object")
    return value


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns, value.st_nlink,
    )


def _read_regular(path: Path, label: str, expected_bytes: int | None = None) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        lexical = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise Full128SmokeError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1
        or (expected_bytes is not None and lexical.st_size != expected_bytes)
    ):
        raise Full128SmokeError(f"{label} is not the expected single-link regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(lexical):
            raise Full128SmokeError(f"{label} changed before open")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = absolute.stat(follow_symlinks=False)
    if (
        len(payload) != before.st_size
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(final)
    ):
        raise Full128SmokeError(f"{label} changed during read")
    return bytes(payload)


def _read_regular_at(directory_fd: int, name: str, label: str, expected_bytes: int | None = None) -> bytes:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise Full128SmokeError(f"{label} name differs")
    lexical = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1
        or (expected_bytes is not None and lexical.st_size != expected_bytes)
    ):
        raise Full128SmokeError(f"{label} is not the expected single-link regular file")
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(lexical):
            raise Full128SmokeError(f"{label} changed before open")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        len(payload) != before.st_size
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(final)
    ):
        raise Full128SmokeError(f"{label} changed during read")
    return bytes(payload)


@contextmanager
def _held_directory(
    path: Path, label: str, *, expected_roster: set[str] | frozenset[str] | None = None,
) -> Iterator[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    lexical = absolute.stat(follow_symlinks=False)
    if not stat.S_ISDIR(lexical.st_mode):
        raise Full128SmokeError(f"{label} is not a lexical directory")
    descriptor = os.open(
        absolute, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino):
            raise Full128SmokeError(f"{label} changed before open")
        if expected_roster is not None and set(os.listdir(descriptor)) != set(expected_roster):
            raise Full128SmokeError(f"{label} roster differs")
        yield descriptor
        after = os.fstat(descriptor)
        final = absolute.stat(follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_mode)
            != (after.st_dev, after.st_ino, after.st_mode)
            or (before.st_dev, before.st_ino, before.st_mode)
            != (final.st_dev, final.st_ino, final.st_mode)
        ):
            raise Full128SmokeError(f"{label} changed during use")
    finally:
        os.close(descriptor)


def _write_create_only_at(directory_fd: int, name: str, payload: bytes) -> None:
    if type(name) is not str or not name or "/" in name or name in (".", "..") or type(payload) is not bytes:
        raise Full128SmokeError("create-only output differs")
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise Full128SmokeError("create-only write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _implementation_record(path: Path, label: str) -> dict[str, object]:
    payload = _read_regular(path, label)
    if not payload:
        raise Full128SmokeError(f"{label} is empty")
    return {
        "label": label, "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload), "sha256": _sha256(payload),
    }


def _bound_file(path: Path, size: int, digest: str, label: str) -> dict[str, object]:
    payload = _read_regular(path, label, size)
    if _sha256(payload) != digest:
        raise Full128SmokeError(f"{label} SHA-256 differs")
    return {
        "label": label, "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": size, "sha256": digest,
    }


def _load_source(path: Path, size: int, digest: str, name: str) -> types.ModuleType:
    payload = _read_regular(path, name, size)
    if _sha256(payload) != digest:
        raise Full128SmokeError(f"source binding differs: {path}")
    if name in sys.modules:
        raise Full128SmokeError("private source-only module name is occupied")
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
def _delegates() -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    p_name = f"_jx_full128_smoke_preflight_{os.getpid()}_{id(object())}"
    v_name = f"_jx_full128_smoke_v3_{os.getpid()}_{id(object())}"
    preflight = _load_source(PREFLIGHT_PATH, PREFLIGHT_BYTES, PREFLIGHT_SHA256, p_name)
    v3 = _load_source(V3_PATH, V3_BYTES, V3_SHA256, v_name)
    try:
        yield preflight, v3
    finally:
        sys.modules.pop(p_name, None)
        sys.modules.pop(v_name, None)


@contextmanager
def _qualified_artifact_delegates() -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    bridge_name = f"_jx_full128_smoke_bridge_{os.getpid()}_{id(object())}"
    grid_name = f"_jx_full128_smoke_grid_{os.getpid()}_{id(object())}"
    bridge = _load_source(FULL_BRIDGE_PATH, FULL_BRIDGE_BYTES, FULL_BRIDGE_SHA256, bridge_name)
    grid = _load_source(TRAJECTORY_GRID_PATH, TRAJECTORY_GRID_BYTES, TRAJECTORY_GRID_SHA256, grid_name)
    try:
        yield bridge, grid
    finally:
        sys.modules.pop(bridge_name, None)
        sys.modules.pop(grid_name, None)


def _bound_inputs(preflight: types.ModuleType, v3: types.ModuleType) -> dict[str, Any]:
    source = _bound_file(PREFLIGHT_PATH, PREFLIGHT_BYTES, PREFLIGHT_SHA256, "full128 preflight source")
    test = _bound_file(PREFLIGHT_TEST_PATH, PREFLIGHT_TEST_BYTES, PREFLIGHT_TEST_SHA256, "full128 preflight test")
    preflight_retained = _bound_file(
        PREFLIGHT_PROTOCOL_PATH, PREFLIGHT_PROTOCOL_BYTES, PREFLIGHT_PROTOCOL_FILE_SHA256,
        "full128 preflight retained protocol",
    )
    bridge_source = _bound_file(FULL_BRIDGE_PATH, FULL_BRIDGE_BYTES, FULL_BRIDGE_SHA256, "full128 bridge source")
    bridge_test = _bound_file(
        FULL_BRIDGE_TEST_PATH, FULL_BRIDGE_TEST_BYTES, FULL_BRIDGE_TEST_SHA256, "full128 bridge test",
    )
    bridge_protocol = _bound_file(
        FULL_BRIDGE_ROOT / "PROTOCOL.json", FULL_BRIDGE_PROTOCOL_BYTES,
        FULL_BRIDGE_PROTOCOL_SHA256, "full128 bridge protocol",
    )
    bridge_states = _bound_file(
        FULL_BRIDGE_ROOT / "BRIDGE.json", FULL_BRIDGE_STATES_BYTES,
        FULL_BRIDGE_STATES_SHA256, "full128 bridge states",
    )
    grid_source = _bound_file(
        TRAJECTORY_GRID_PATH, TRAJECTORY_GRID_BYTES, TRAJECTORY_GRID_SHA256,
        "full128 trajectory grid source",
    )
    grid_test = _bound_file(
        TRAJECTORY_GRID_TEST_PATH, TRAJECTORY_GRID_TEST_BYTES, TRAJECTORY_GRID_TEST_SHA256,
        "full128 trajectory grid test",
    )
    grid_protocol = _bound_file(
        TRAJECTORY_GRID_ROOT / "PROTOCOL.json", TRAJECTORY_GRID_PROTOCOL_BYTES,
        TRAJECTORY_GRID_PROTOCOL_SHA256, "full128 trajectory grid protocol",
    )
    v3_source = _bound_file(V3_PATH, V3_BYTES, V3_SHA256, "V3 lifecycle source")
    v3_test = _bound_file(V3_TEST_PATH, V3_TEST_BYTES, V3_TEST_SHA256, "V3 lifecycle test")
    preflight_protocol = preflight.build_protocol()
    preflight_payload = _canonical_json(preflight_protocol)
    if (
        len(preflight_payload) != PREFLIGHT_PROTOCOL_BYTES
        or _sha256(preflight_payload) != PREFLIGHT_PROTOCOL_FILE_SHA256
        or preflight_protocol.get("content_sha256") != PREFLIGHT_PROTOCOL_CONTENT_SHA256
        or preflight_protocol.get("full_clone_initial_states", {}).get("primary_relative_aggregate_sha256")
        != PREFLIGHT_PRIMARY_AGGREGATE_SHA256
        or preflight_protocol.get("full_clone_initial_states", {}).get("barycentric_checkpoint_aggregate_sha256")
        != PREFLIGHT_CHECKPOINT_AGGREGATE_SHA256
        or preflight_protocol.get("full_clone_initial_states", {}).get("batches", [{}])[0].get("primary_relative_payload_sha256")
        != YEAR_ZERO_PRIMARY_SHA256
        or preflight_protocol.get("full_clone_initial_states", {}).get("batches", [{}])[0].get("barycentric_checkpoint_payload_sha256")
        != YEAR_ZERO_CHECKPOINT_SHA256
    ):
        raise Full128SmokeError("frozen full128 preflight reconstruction differs")
    if _read_regular(PREFLIGHT_PROTOCOL_PATH, "full128 retained preflight") != preflight_payload:
        raise Full128SmokeError("retained full128 preflight differs from reconstruction")
    with _qualified_artifact_delegates() as (bridge_delegate, grid_delegate):
        retained_bridge = bridge_delegate.validate_output(FULL_BRIDGE_ROOT)
        retained_grid = grid_delegate.validate_prepared(TRAJECTORY_GRID_ROOT)
    if (
        retained_bridge.get("content_sha256") != FULL_BRIDGE_STATES_CONTENT_SHA256
        or retained_bridge.get("protocol_content_sha256") != FULL_BRIDGE_PROTOCOL_CONTENT_SHA256
        or retained_bridge.get("batch_count") != 12
        or retained_bridge.get("tracer_count_per_batch") != TRACER_COUNT
        or retained_bridge.get("claims", {}).get("dynamics_executed") is not False
        or retained_bridge.get("claims", {}).get("ten_myr_execution_authorized") is not False
        or retained_grid.get("content_sha256") != TRAJECTORY_GRID_PROTOCOL_CONTENT_SHA256
        or retained_grid.get("geometry", {}).get("body_count") != BODY_COUNT
        or retained_grid.get("geometry", {}).get("tracer_count") != TRACER_COUNT
        or retained_grid.get("claims", {}).get("trajectory_acquisition_execution_authorized") is not False
    ):
        raise Full128SmokeError("qualified full128 bridge or trajectory grid differs")
    v3_roster = {
        "PROTOCOL.json", "PREFIX_STATES.bin", "RESTART_STREAMS.bin", "SEGMENTS.jsonl",
        "ROUTES.jsonl", "RUNTIME_PREFLIGHT.json", "RUNTIME.json", "RESULT.json",
    }
    with _held_directory(V3_ROOT, "V3 result root", expected_roster=v3_roster) as v3_fd:
        v3_payloads = {
            name: _read_regular_at(v3_fd, name, f"V3 {name}") for name in sorted(v3_roster)
        }
    retained_v3_payload = v3_payloads["RESULT.json"]
    retained_v3 = _strict_json(retained_v3_payload)
    if (
        retained_v3.get("content_sha256") != V3_RESULT_CONTENT_SHA256
        or _sha256(v3_payloads["PROTOCOL.json"]) != V3_PROTOCOL_SHA256
        or _sha256(v3_payloads["PREFIX_STATES.bin"]) != V3_PREFIX_SHA256
        or _sha256(v3_payloads["RESTART_STREAMS.bin"]) != V3_RESTART_SHA256
        or _sha256(v3_payloads["SEGMENTS.jsonl"]) != V3_SEGMENTS_SHA256
        or v3_payloads["ROUTES.jsonl"] != b""
        or _sha256(v3_payloads["RUNTIME_PREFLIGHT.json"]) != V3_RUNTIME_PREFLIGHT_SHA256
        or _sha256(v3_payloads["RUNTIME.json"]) != V3_RUNTIME_SHA256
        or _sha256(retained_v3_payload) != V3_RESULT_SHA256
        or retained_v3_payload != _canonical_json(retained_v3)
        or retained_v3.get("schema") != "jx.tno.1kyr-reduced5-lifecycle-gate.result.v3"
        or retained_v3.get("status") != "ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED"
        or retained_v3.get("all_predeclared_gates_passed") is not True
        or retained_v3.get("claims", {}).get("ten_myr_execution_authorized") is not False
    ):
        raise Full128SmokeError("frozen V3 result reconstruction differs")
    if (
        Path(v3.H64_SOURCE_PATH) != H64_SOURCE_PATH
        or v3.H64_SOURCE_BYTES != H64_SOURCE_BYTES
        or type(v3.H64_SOURCE_BYTES) is not int
        or v3.H64_SOURCE_SHA256 != H64_SOURCE_SHA256
    ):
        raise Full128SmokeError("frozen V3 large source prerequisite differs")
    engine = [
        _bound_file(PROJECT_ROOT / relative, size, digest, f"engine {relative}")
        for relative, size, digest in ENGINE_FILES
    ]
    return {
        "full128_preflight_source": source,
        "full128_preflight_test": test,
        "full128_preflight_retained_protocol": preflight_retained,
        "full128_preflight_protocol": {
            "canonical_bytes": PREFLIGHT_PROTOCOL_BYTES,
            "file_sha256": PREFLIGHT_PROTOCOL_FILE_SHA256,
            "content_sha256": PREFLIGHT_PROTOCOL_CONTENT_SHA256,
        },
        "full128_bridge_source": bridge_source,
        "full128_bridge_test": bridge_test,
        "full128_bridge_protocol": {
            **bridge_protocol,
            "content_sha256": FULL_BRIDGE_PROTOCOL_CONTENT_SHA256,
        },
        "full128_bridge_states": {
            **bridge_states,
            "content_sha256": FULL_BRIDGE_STATES_CONTENT_SHA256,
        },
        "full128_trajectory_grid_source": grid_source,
        "full128_trajectory_grid_test": grid_test,
        "full128_trajectory_grid_protocol": {
            **grid_protocol,
            "content_sha256": TRAJECTORY_GRID_PROTOCOL_CONTENT_SHA256,
        },
        "v3_source": v3_source,
        "v3_test": v3_test,
        "v3_result_root": {
            "file_count": 8,
            "result_file_sha256": V3_RESULT_SHA256,
            "result_content_sha256": V3_RESULT_CONTENT_SHA256,
        },
        "required_large_runtime_artifact": {
            "relative_path": H64_SOURCE_RELATIVE_PATH,
            "bytes": H64_SOURCE_BYTES,
            "sha256": H64_SOURCE_SHA256,
            "regular_file_no_symlink": True,
            "exact_link_count": 1,
            "availability_checked_by_prepare": False,
            "availability_and_full_hash_checked_before_gpu_lane": True,
            "copy_or_acquisition_custody_must_be_retained_separately": True,
        },
        "engine_files": engine,
    }


def build_protocol() -> dict[str, Any]:
    """Rebuild the exact prepare-only smoke protocol."""
    with _delegates() as (preflight, v3):
        inputs = _bound_inputs(preflight, v3)
    source = _implementation_record(SCRIPT_PATH, "full128 smoke source")
    test = _implementation_record(TEST_PATH, "full128 smoke test")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARED_EXACT_BATCH0_FULL129_1KYR_GPU_SMOKE_NOT_EXECUTED",
        "objective": "QUALIFY_EXACT_FULL129_BATCH0_LIFECYCLE_BEFORE_ANY_10MY_EXECUTION",
        "implementation": {
            **source,
            "test": test,
            "source_only_delegate_loading": True,
            "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        },
        "bound_inputs": inputs,
        "preregistration": {
            "required_before_execute": True,
            "branch": PREREGISTRATION_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "commit_and_tag_object_are_runtime_supplied_and_locally_reconstructed": True,
            "exact_clean_tracked_paths": [
                str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
                str(TEST_PATH.relative_to(PROJECT_ROOT)),
                str((DEFAULT_OUTPUT_ROOT / "PROTOCOL.json").relative_to(PROJECT_ROOT)),
            ],
            "network_verification_by_this_implementation": False,
        },
        "execution": {
            "bounded_smoke_execution_scope_defined": True,
            "execution_requires_all_preregistration_artifact_runtime_and_replay_gates": True,
            "fresh_child_required": True,
            "batch_order": BATCH_ORDER,
            "object_id": OBJECT_ID,
            "spk_id": SPK_ID,
            "ephemeris": "DE441",
            "massive_body_count": MASSIVE_COUNT,
            "tracer_count": TRACER_COUNT,
            "tracer_order": [
                {"sample_kind": "NOMINAL", "clone_index": None},
                *({"sample_kind": "COVARIANCE_CLONE", "clone_index": index} for index in range(128)),
            ],
            "steps_per_julian_year": STEPS_PER_YEAR,
            "step_julian_years_hex": (1.0 / STEPS_PER_YEAR).hex(),
            "horizon_julian_years": 1_000,
            "segment_julian_years": SEGMENT_YEARS,
            "segment_steps": SEGMENT_STEPS,
            "segment_count": SEGMENT_COUNT,
            "checkpoint_years": list(CHECKPOINT_YEARS),
            "initial_state_authority": "runs/tno_full128_reduced5_bridge_v1/BRIDGE.json",
            "initial_state_authority_content_sha256": FULL_BRIDGE_STATES_CONTENT_SHA256,
            "year_zero_direct_reduced5_barycentric_sha256": YEAR_ZERO_DIRECT_SHA256,
            "year_zero_primary_relative_sha256": YEAR_ZERO_PRIMARY_SHA256,
            "year_zero_barycentric_checkpoint_sha256": YEAR_ZERO_CHECKPOINT_SHA256,
        },
        "acceptance": {
            "this_smoke_validator_is_sole_authoritative_full129_smoke_validator": True,
            "preflight_projection_receipt_is_secondary_non_authoritative_evidence": True,
            "all_129_states_finite_at_all_21_checkpoints": True,
            "all_129_lifecycle_codes_exact_active_zero_at_all_21_checkpoints": True,
            "route_record_count": 0,
            "removal_record_count": 0,
            "minimum_source_distance_strictly_greater_than_AU_hex": float(1.0).hex(),
            "massive_plus_frozen_nine_v3_projection_bitwise_equal_at_all_checkpoints": True,
            "exact_retained_h64_source_segments_and_endpoint": True,
            "source_tile_identity_reuses_exact_v3_batch0_labels": True,
            "full129_continuations_reconstruct_all_21_checkpoints_bitwise": True,
            "year_zero_continuation_equals_authoritative_bridge_primary_state": True,
            "next_continuation_structurally_valid_at_global_source_step": 64_000,
            "next_source_step_not_executed": True,
            "elapsed_time_diagnostic_only": True,
        },
        "outputs": {
            "prepared_roster": ["PROTOCOL.json"],
            "published_roster": sorted(EXPECTED_OUTPUT_ROSTER),
            "checkpoints_bytes": CHECKPOINTS_BYTES,
            "checkpoints_layout": "21_CHECKPOINTS_X_134_ROWS_X_POS3_VEL3_BINARY64_LE",
            "lifecycle_bytes": LIFECYCLE_BYTES,
            "lifecycle_layout": "21_CHECKPOINTS_X_129_UINT8_STATUS_CODES",
            "continuation_bytes": CONTINUATION_BYTES,
            "continuation_record_count": CHECKPOINT_COUNT,
            "continuation_record_bytes": CONTINUATION_RECORD_BYTES,
            "continuation_layout": "21_CHECKPOINTS_X_129X3_POSITION_BLOCK_THEN_129X3_VELOCITY_BLOCK_BINARY64_LE",
            "segment_jsonl_record_count": SEGMENT_COUNT,
            "result_published_last": True,
            "create_only": True,
        },
        "interpretation": {
            "single_object_batch_zero_engineering_smoke_only": True,
            "clones_are_local_formal_covariance_sensitivity_draws_not_posterior_or_likelihood_samples": True,
            "one_thousand_year_survival_does_not_establish_ten_myr_classification": True,
            "nine_retained_v3_lanes_are_crosschecks_not_independent_science_replicates": True,
            "no_population_precision_or_tail_probability_claim": True,
        },
        "contract_roles": {
            "full128_bridge": "AUTHORITATIVE_YEAR_ZERO_STATE_GEOMETRY",
            "full128_preflight": "INITIAL_GEOMETRY_AND_SECONDARY_SMOKE_PROJECTION_ONLY",
            "trajectory_grid": "FUTURE_10MY_RETENTION_AND_WRITER_GEOMETRY_NOT_EXECUTED_OR_AUTHORIZED_HERE",
            "v3": "RETAINED_NINE_LANE_1KYR_BITWISE_CROSSCHECK_AND_RUNTIME_LINEAGE",
            "preflight_legacy_8394469632_byte_grid_is_not_THE_BOUND_FUTURE_WRITER_CONTRACT": True,
            "trajectory_grid_4743277848_byte_protocol_is_THE_BOUND_FUTURE_WRITER_CONTRACT": True,
        },
        "blockers_after_pass": [
            "TEN_MYR_FULL128_EXECUTION_NOT_AUTHORIZED_BY_SMOKE_RESULT",
            "TEN_MYR_WRITER_RESTART_AND_FINALIZER_NOT_EXECUTED",
            "TEN_MYR_CLASSIFICATION_NOT_GENERATED",
        ],
        "claims": dict(CLAIMS_PREPARED),
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise Full128SmokeError("protocol must be an exact object")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise Full128SmokeError("protocol differs from exact reconstruction")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise Full128SmokeError("protocol content identity differs")
    return str(value["content_sha256"])


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.name in ("", ".", ".."):
        raise Full128SmokeError("output root name differs")
    protocol = build_protocol()
    with _held_directory(root.parent, "output parent") as parent_fd:
        try:
            os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise Full128SmokeError("output root must be absent for create-only preparation")
        os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        created_path = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        root_fd = os.open(
            root.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(root_fd)
            if (opened.st_dev, opened.st_ino) != (created_path.st_dev, created_path.st_ino):
                raise Full128SmokeError("created output root identity differs")
            _write_create_only_at(root_fd, "PROTOCOL.json", _canonical_json(protocol))
            os.fsync(root_fd)
            if set(os.listdir(root_fd)) != {"PROTOCOL.json"}:
                raise Full128SmokeError("prepared root roster differs")
        finally:
            os.close(root_fd)
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared root", expected_roster={"PROTOCOL.json"}) as root_fd:
        payload = _read_regular_at(root_fd, "PROTOCOL.json", "prepared protocol")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise Full128SmokeError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def _git_text(arguments: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise Full128SmokeError(f"Git {label} failed: {detail}")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise Full128SmokeError(f"Git {label} output is not UTF-8") from exc


def _git_bytes(arguments: Sequence[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise Full128SmokeError(f"Git {label} failed: {detail}")
    return completed.stdout


def _git_custody(
    *, externally_preregistered: bool, prereg_commit: str, prereg_tag_object: str,
) -> dict[str, Any]:
    if externally_preregistered is not True:
        raise Full128SmokeError("bounded smoke execution requires external preregistration")
    if not (
        type(prereg_commit) is str and len(prereg_commit) == 40
        and all(c in "0123456789abcdef" for c in prereg_commit)
    ):
        raise Full128SmokeError("preregistration commit identity differs")
    if not (
        type(prereg_tag_object) is str and len(prereg_tag_object) == 40
        and all(c in "0123456789abcdef" for c in prereg_tag_object)
    ):
        raise Full128SmokeError("preregistration tag-object identity differs")
    branch = _git_text(("symbolic-ref", "--quiet", "--short", "HEAD"), "branch")
    head = _git_text(("rev-parse", "HEAD"), "HEAD")
    tag_object = _git_text(("rev-parse", f"refs/tags/{PREREGISTRATION_TAG}"), "tag object")
    tag_commit = _git_text(("rev-parse", f"{PREREGISTRATION_TAG}^{{commit}}"), "peeled tag")
    tag_type = _git_text(("cat-file", "-t", tag_object), "tag type")
    if (
        branch != PREREGISTRATION_BRANCH or head != prereg_commit
        or tag_object != prereg_tag_object or tag_commit != head or tag_type != "tag"
    ):
        raise Full128SmokeError("local preregistration branch/commit/tag custody differs")
    paths = (
        str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        str(TEST_PATH.relative_to(PROJECT_ROOT)),
        str((DEFAULT_OUTPUT_ROOT / "PROTOCOL.json").relative_to(PROJECT_ROOT)),
    )
    status = _git_text(("status", "--porcelain=v1", "--untracked-files=all", "--", *paths), "pinned-path status")
    if status:
        raise Full128SmokeError("preregistered source/test/protocol paths are not clean")
    records: list[dict[str, Any]] = []
    for relative in paths:
        live = _read_regular(PROJECT_ROOT / relative, f"preregistered {relative}")
        committed = _git_bytes(("show", f"{head}:{relative}"), f"committed {relative}")
        if live != committed:
            raise Full128SmokeError(f"preregistered path differs from commit: {relative}")
        records.append({"relative_path": relative, "bytes": len(live), "sha256": _sha256(live)})
    protocol = build_protocol()
    if records[0]["sha256"] != protocol["implementation"]["sha256"] or records[1]["sha256"] != protocol["implementation"]["test"]["sha256"]:
        raise Full128SmokeError("Git source/test pins differ from reconstructed protocol")
    if records[2]["sha256"] != _sha256(_canonical_json(protocol)):
        raise Full128SmokeError("Git protocol pin differs from reconstructed protocol")
    receipt: dict[str, Any] = {
        "schema": GIT_CUSTODY_SCHEMA,
        "externally_preregistered": True,
        "branch": branch,
        "annotated_tag": PREREGISTRATION_TAG,
        "commit": head,
        "tag_object": tag_object,
        "tag_peeled_commit": tag_commit,
        "tag_object_type": tag_type,
        "clean_tracked_paths": records,
        "network_remote_verification_performed": False,
        "point_observation_only": True,
        "content_sha256": "",
    }
    receipt["content_sha256"] = _domain_hash(GIT_CUSTODY_DOMAIN, receipt)
    return receipt


def _validate_git_custody_offline(
    value: object, protocol: Mapping[str, Any], *, expected_commit: str | None = None,
    expected_tag_object: str | None = None,
) -> dict[str, Any]:
    keys = {
        "schema", "externally_preregistered", "branch", "annotated_tag", "commit",
        "tag_object", "tag_peeled_commit", "tag_object_type", "clean_tracked_paths",
        "network_remote_verification_performed", "point_observation_only", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full128SmokeError("Git custody receipt shape differs")
    records = value.get("clean_tracked_paths")
    expected_paths = protocol["preregistration"]["exact_clean_tracked_paths"]
    if (
        value.get("schema") != GIT_CUSTODY_SCHEMA
        or value.get("externally_preregistered") is not True
        or value.get("branch") != PREREGISTRATION_BRANCH
        or value.get("annotated_tag") != PREREGISTRATION_TAG
        or type(value.get("commit")) is not str or len(value["commit"]) != 40
        or any(c not in "0123456789abcdef" for c in value["commit"])
        or type(value.get("tag_object")) is not str or len(value["tag_object"]) != 40
        or any(c not in "0123456789abcdef" for c in value["tag_object"])
        or value.get("tag_peeled_commit") != value.get("commit")
        or value.get("tag_object_type") != "tag"
        or value.get("network_remote_verification_performed") is not False
        or value.get("point_observation_only") is not True
        or type(records) is not list or len(records) != 3
        or [item.get("relative_path") if type(item) is dict else None for item in records] != expected_paths
        or value.get("content_sha256") != _domain_hash(GIT_CUSTODY_DOMAIN, value)
    ):
        raise Full128SmokeError("Git custody receipt differs")
    if expected_commit is not None and value["commit"] != expected_commit:
        raise Full128SmokeError("Git custody commit differs")
    if expected_tag_object is not None and value["tag_object"] != expected_tag_object:
        raise Full128SmokeError("Git custody tag object differs")
    expected_files = (
        (protocol["implementation"]["bytes"], protocol["implementation"]["sha256"]),
        (protocol["implementation"]["test"]["bytes"], protocol["implementation"]["test"]["sha256"]),
        (len(_canonical_json(protocol)), _sha256(_canonical_json(protocol))),
    )
    for record, (length, digest) in zip(records, expected_files, strict=True):
        if (
            type(record) is not dict or set(record) != {"relative_path", "bytes", "sha256"}
            or type(record.get("bytes")) is not int or record["bytes"] <= 0
            or record["bytes"] != length
            or record.get("sha256") != digest
        ):
            raise Full128SmokeError("Git custody tracked-file receipt differs")
    return value


def _qualified_bridge_batch0_payloads() -> tuple[bytes, bytes, bytes]:
    with _qualified_artifact_delegates() as (bridge_delegate, _grid_delegate):
        bridge = bridge_delegate.validate_output(FULL_BRIDGE_ROOT)
        batches = bridge.get("batches")
        if (
            type(batches) is not list or len(batches) != 12 or type(batches[0]) is not dict
            or batches[0].get("batch_order") != BATCH_ORDER
            or batches[0].get("object_id") != OBJECT_ID
            or batches[0].get("spk_id") != SPK_ID
            or batches[0].get("tracer_count") != TRACER_COUNT
        ):
            raise Full128SmokeError("qualified full128 bridge batch-zero roster differs")
        batch = batches[0]
        direct, primary, checkpoint = bridge_delegate._consumer_payloads(
            batch.get("massive_bodies"), batch.get("tracers")
        )
    receipts = batch.get("consumer_state_payloads")
    if (
        type(receipts) is not dict
        or len(direct) != STATE_RECORD_BYTES or _sha256(direct) != YEAR_ZERO_DIRECT_SHA256
        or len(primary) != STATE_RECORD_BYTES or _sha256(primary) != YEAR_ZERO_PRIMARY_SHA256
        or len(checkpoint) != STATE_RECORD_BYTES or _sha256(checkpoint) != YEAR_ZERO_CHECKPOINT_SHA256
        or receipts.get("direct_reduced5_barycentric", {}).get("sha256") != _sha256(direct)
        or receipts.get("engine_primary_relative", {}).get("sha256") != _sha256(primary)
        or receipts.get("engine_checkpoint_equivalent", {}).get("sha256") != _sha256(checkpoint)
        or receipts.get("selected9_checkpoint_bitwise_equal_to_v3_year_zero") is not True
    ):
        raise Full128SmokeError("qualified full128 bridge consumer payload differs")
    return direct, primary, checkpoint


def _batch0_primary_payload(preflight: types.ModuleType) -> bytes:
    _records, payloads = preflight._bound_inputs()
    stage0, bridge, custody, _v3_prefix = preflight._validate_semantic_inputs(payloads)
    states = stage0.get("states")
    batches = bridge.get("batches")
    de441_rows = custody.get("state_records")
    if type(states) is not list or type(batches) is not list or type(de441_rows) is not list:
        raise Full128SmokeError("full129 initial-state sources differ")
    group = [row for row in states if type(row) is dict and row.get("object_id") == OBJECT_ID]
    if len(group) != TRACER_COUNT or [row.get("clone_index") for row in group] != [None, *range(128)]:
        raise Full128SmokeError("batch0 full129 Stage0 order differs")
    bridge_batch = batches[BATCH_ORDER]
    massive_rows = bridge_batch.get("massive_bodies") if type(bridge_batch) is dict else None
    if type(massive_rows) is not list or len(massive_rows) != 11:
        raise Full128SmokeError("batch0 resolved11 roster differs")
    gms = tuple(
        preflight._exact_float_hex(row.get("gravitational_parameter_m3_per_s2_hex"), "resolved11 GM")
        * (preflight.SECONDS_PER_JULIAN_YEAR_EXACT ** 2) / (preflight.AU_METRES_EXACT ** 3)
        for row in massive_rows
    )
    rows = de441_rows[:11]
    if [row.get("naif_id") for row in rows] != list(preflight.RESOLVED11_NAIF_IDS):
        raise Full128SmokeError("batch0 DE441 body order differs")
    ssb_positions = tuple(preflight._vector_hex(row.get("position_AU_binary64_hex"), "DE441 position") for row in rows)
    ssb_velocities = tuple(
        tuple(value * 365.25 for value in preflight._vector_hex(row.get("velocity_AU_per_day_binary64_hex"), "DE441 velocity"))
        for row in rows
    )
    center_p, center_v = preflight._weighted_center(gms, ssb_positions, ssb_velocities)
    resolved_p = tuple(preflight._sub(row, center_p) for row in ssb_positions)
    resolved_v = tuple(preflight._sub(row, center_v) for row in ssb_velocities)
    helio_p = tuple(
        preflight._rotate_ecliptic_to_j2000(preflight._vector_hex(row.get("position_AU_hex"), "Stage0 position"))
        for row in group
    )
    helio_v = tuple(
        preflight._rotate_ecliptic_to_j2000(preflight._vector_hex(row.get("velocity_AU_per_julian_year_hex"), "Stage0 velocity"))
        for row in group
    )
    tracer_resolved_p = tuple(preflight._sub(preflight._add(row, ssb_positions[0]), center_p) for row in helio_p)
    tracer_resolved_v = tuple(preflight._sub(preflight._add(row, ssb_velocities[0]), center_v) for row in helio_v)
    reduced_gms = (math.fsum(gms[:6]), *gms[6:10])
    active_p = (resolved_p[0], *resolved_p[6:10])
    active_v = (resolved_v[0], *resolved_v[6:10])
    reduced_center_p, reduced_center_v = preflight._weighted_center(reduced_gms, active_p, active_v)
    centered_massive_p = tuple(preflight._sub(row, reduced_center_p) for row in active_p)
    centered_massive_v = tuple(preflight._sub(row, reduced_center_v) for row in active_v)
    centered_tracer_p = tuple(preflight._sub(row, reduced_center_p) for row in tracer_resolved_p)
    centered_tracer_v = tuple(preflight._sub(row, reduced_center_v) for row in tracer_resolved_v)
    central_p, central_v = centered_massive_p[0], centered_massive_v[0]
    massive_p = ((0.0, 0.0, 0.0), *(preflight._sub(row, central_p) for row in centered_massive_p[1:]))
    massive_v = ((0.0, 0.0, 0.0), *(preflight._sub(row, central_v) for row in centered_massive_v[1:]))
    tracer_p = tuple(preflight._sub(row, central_p) for row in centered_tracer_p)
    tracer_v = tuple(preflight._sub(row, central_v) for row in centered_tracer_v)
    payload = preflight._pack_rows(tuple((*massive_p[i], *massive_v[i]) for i in range(5)) + tuple((*tracer_p[i], *tracer_v[i]) for i in range(129)))
    if len(payload) != STATE_RECORD_BYTES or _sha256(payload) != YEAR_ZERO_PRIMARY_SHA256:
        raise Full128SmokeError("batch0 full129 primary-relative payload differs")
    _direct, qualified_primary, _checkpoint = _qualified_bridge_batch0_payloads()
    if payload != qualified_primary:
        raise Full128SmokeError("legacy preflight and qualified full128 bridge primary payload differ")
    return qualified_primary


def _unpack_state_record(payload: bytes) -> tuple[tuple[float, ...], ...]:
    if type(payload) is not bytes or len(payload) != STATE_RECORD_BYTES:
        raise Full128SmokeError("checkpoint record geometry differs")
    values = struct.unpack(f"<{BODY_COUNT * 6}d", payload)
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise Full128SmokeError("checkpoint record contains a non-finite value")
    return tuple(tuple(values[index * 6:(index + 1) * 6]) for index in range(BODY_COUNT))


def _pack_continuation(positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]]) -> bytes:
    if type(positions) not in (tuple, list) or type(velocities) not in (tuple, list) or len(positions) != TRACER_COUNT or len(velocities) != TRACER_COUNT:
        raise Full128SmokeError("continuation roster differs")
    values: list[float] = []
    for rows in (positions, velocities):
        for row in rows:
            if type(row) not in (tuple, list) or len(row) != 3:
                raise Full128SmokeError("continuation row shape differs")
            for item in row:
                if type(item) is not float or not math.isfinite(item):
                    raise Full128SmokeError("continuation contains a non-finite value")
                values.append(item)
    payload = struct.pack(f"<{len(values)}d", *values)
    if len(payload) != CONTINUATION_RECORD_BYTES:
        raise Full128SmokeError("continuation byte geometry differs")
    return payload


def _unpack_continuation(payload: bytes) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    if type(payload) is not bytes or len(payload) != CONTINUATION_RECORD_BYTES:
        raise Full128SmokeError("continuation byte geometry differs")
    values = struct.unpack(f"<{TRACER_COUNT * 6}d", payload)
    if any(not math.isfinite(item) for item in values):
        raise Full128SmokeError("continuation contains a non-finite value")
    split = TRACER_COUNT * 3
    positions = tuple(tuple(values[index * 3:index * 3 + 3]) for index in range(TRACER_COUNT))
    velocities = tuple(tuple(values[split + index * 3:split + index * 3 + 3]) for index in range(TRACER_COUNT))
    return positions, velocities  # type: ignore[return-value]


def _continuation_records(payload: bytes) -> tuple[bytes, ...]:
    if type(payload) is not bytes or len(payload) != CONTINUATION_BYTES:
        raise Full128SmokeError("continuation roster byte geometry differs")
    records = tuple(
        payload[index * CONTINUATION_RECORD_BYTES:(index + 1) * CONTINUATION_RECORD_BYTES]
        for index in range(CHECKPOINT_COUNT)
    )
    for record in records:
        _unpack_continuation(record)
    return records


def _checkpoint_full129(
    preflight: types.ModuleType, grid: types.ModuleType, kernel: object,
    source_positions: Sequence[Sequence[float]], source_velocities: Sequence[Sequence[float]],
    tracer_positions: Sequence[Sequence[float]], tracer_velocities: Sequence[Sequence[float]],
) -> bytes:
    def rows(value: Sequence[Sequence[float]], count: int, label: str) -> tuple[tuple[float, float, float], ...]:
        if type(value) not in (tuple, list) or len(value) != count:
            raise Full128SmokeError(f"{label} roster differs")
        result: list[tuple[float, float, float]] = []
        for row in value:
            if type(row) not in (tuple, list) or len(row) != 3:
                raise Full128SmokeError(f"{label} shape differs")
            converted = tuple(float(item) for item in row)
            if any(not math.isfinite(item) for item in converted):
                raise Full128SmokeError(f"{label} contains non-finite data")
            result.append(converted)  # type: ignore[arg-type]
        return tuple(result)
    source_p = rows(source_positions, MASSIVE_COUNT, "source positions")
    source_v = rows(source_velocities, MASSIVE_COUNT, "source velocities")
    tracer_p = rows(tracer_positions, TRACER_COUNT, "tracer positions")
    tracer_v = rows(tracer_velocities, TRACER_COUNT, "tracer velocities")
    if source_p[0] != (0.0, 0.0, 0.0) or source_v[0] != (0.0, 0.0, 0.0):
        raise Full128SmokeError("primary source is not exact zero")
    stage1 = kernel.stage1
    massive = tuple(
        stage1.MassiveState(
            grid.REDUCED_NAIF_IDS[index], grid.REDUCED_BODY_IDS[index],
            float.fromhex(grid.REDUCED_GM_HEX[index]), source_p[index], source_v[index],
        )
        for index in range(MASSIVE_COUNT)
    )
    center_position, center_velocity = stage1._weighted_centroid(massive)
    packed_rows = tuple(
        (*stage1._sub(source_p[index], center_position),
         *stage1._sub(source_v[index], center_velocity))
        for index in range(MASSIVE_COUNT)
    ) + tuple(
        (*stage1._sub(tracer_p[index], center_position),
         *stage1._sub(tracer_v[index], center_velocity))
        for index in range(TRACER_COUNT)
    )
    return preflight._pack_rows(packed_rows)


def _validate_final_continuation(
    preflight: types.ModuleType, v3: types.ModuleType, source_node_payload: bytes,
    continuation_payload: bytes, expected_checkpoint: bytes,
) -> None:
    from dataclasses import replace
    import numpy as np

    positions, velocities = _unpack_continuation(continuation_payload)
    source_positions, source_velocities = _source_node_rows(source_node_payload)
    _records, payloads = preflight._bound_inputs()
    _stage0, bridge, custody, _prefix = preflight._validate_semantic_inputs(payloads)
    with v3._delegates() as (grid, _h64, _d):
        with grid.FrozenClassifierKernelV1() as kernel:
            reconstructed = _checkpoint_full129(
                preflight, grid, kernel, source_positions, source_velocities,
                positions, velocities,
            )
        context = v3._build_batch_context(grid, _d, np, bridge, custody, BATCH_ORDER)
        base = context["state"]
        resumed = replace(
            base,
            snapshot_id=f"tno-full129-smoke-v1-continuation-DE441-{OBJECT_ID}",
            global_source_step=64_000,
            tracer_relative_positions_AU=np.asarray(positions, dtype=np.dtype("<f8")),
            tracer_relative_velocities_AU_per_year=np.asarray(velocities, dtype=np.dtype("<f8")),
            tracer_indices=np.arange(TRACER_COUNT, dtype=np.uint64),
        )
        lifecycle = context["specs"]["lifecycle_initialize"](resumed)
    if reconstructed != expected_checkpoint:
        raise Full128SmokeError("full129 continuation does not reconstruct the final checkpoint")
    if (
        lifecycle.dynamics_state.global_source_step != 64_000
        or lifecycle.dynamics_state.tracer_relative_positions_AU.dtype != np.dtype("<f8")
        or lifecycle.dynamics_state.tracer_relative_positions_AU.shape != (TRACER_COUNT, 3)
        or lifecycle.dynamics_state.tracer_relative_velocities_AU_per_year.dtype != np.dtype("<f8")
        or lifecycle.dynamics_state.tracer_relative_velocities_AU_per_year.shape != (TRACER_COUNT, 3)
        or tuple(int(item) for item in lifecycle.dynamics_state.tracer_indices.tolist()) != tuple(range(TRACER_COUNT))
        or lifecycle.tracer_lifecycle.tobytes(order="C") != bytes(TRACER_COUNT)
        or lifecycle.removal_records != ()
    ):
        raise Full128SmokeError("full129 continuation is not an exact engine-valid active state")


def _validate_continuation_records(
    preflight: types.ModuleType, v3: types.ModuleType,
    source_node_payloads: Sequence[bytes], continuation_payload: bytes,
    expected_checkpoints: Sequence[bytes],
) -> tuple[str, ...]:
    if (
        type(source_node_payloads) not in (tuple, list)
        or type(expected_checkpoints) not in (tuple, list)
        or len(source_node_payloads) != CHECKPOINT_COUNT
        or len(expected_checkpoints) != CHECKPOINT_COUNT
        or any(type(item) is not bytes for item in source_node_payloads)
        or any(type(item) is not bytes or len(item) != STATE_RECORD_BYTES for item in expected_checkpoints)
    ):
        raise Full128SmokeError("continuation replay roster differs")
    records = _continuation_records(continuation_payload)
    primary_rows = _unpack_state_record(_batch0_primary_payload(preflight))
    authoritative_y0 = _pack_continuation(
        tuple(tuple(row[:3]) for row in primary_rows[MASSIVE_COUNT:]),
        tuple(tuple(row[3:]) for row in primary_rows[MASSIVE_COUNT:]),
    )
    if records[0] != authoritative_y0:
        raise Full128SmokeError("year-zero continuation differs from authoritative bridge primary state")
    with v3._delegates() as (grid, _h64, _d):
        with grid.FrozenClassifierKernelV1() as kernel:
            for index, (source_node, continuation, expected) in enumerate(
                zip(source_node_payloads, records, expected_checkpoints, strict=True)
            ):
                positions, velocities = _unpack_continuation(continuation)
                source_positions, source_velocities = _source_node_rows(source_node)
                reconstructed = _checkpoint_full129(
                    preflight, grid, kernel, source_positions, source_velocities,
                    positions, velocities,
                )
                if reconstructed != expected:
                    raise Full128SmokeError(
                        f"full129 continuation does not reconstruct checkpoint {index}"
                    )
    _validate_final_continuation(
        preflight, v3, source_node_payloads[-1], records[-1], expected_checkpoints[-1],
    )
    return tuple(_sha256(record) for record in records)


def _source_node_payload(v3: types.ModuleType, descriptor: int, source_record: Mapping[str, Any], node: int) -> bytes:
    if type(node) is not int or not 0 <= node <= 64_000:
        raise Full128SmokeError("source node locator differs")
    base = int(source_record["byte_offset"])
    positions_bytes = int(source_record["positions_block_bytes"])
    position = v3._pread_exact(descriptor, base + node * SOURCE_NODE_BYTES, SOURCE_NODE_BYTES, "source node position")
    velocity = v3._pread_exact(
        descriptor, base + positions_bytes + node * SOURCE_NODE_BYTES,
        SOURCE_NODE_BYTES, "source node velocity",
    )
    return position + velocity


def _source_node_rows(payload: bytes) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    if type(payload) is not bytes or len(payload) != SOURCE_NODE_BYTES * 2:
        raise Full128SmokeError("source node payload geometry differs")
    values = struct.unpack(f"<{SOURCE_COUNT * 6}d", payload)
    if any(not math.isfinite(item) for item in values):
        raise Full128SmokeError("source node contains non-finite data")
    split = SOURCE_COUNT * 3
    positions = tuple(tuple(values[index * 3:index * 3 + 3]) for index in range(SOURCE_COUNT))
    velocities = tuple(tuple(values[split + index * 3:split + index * 3 + 3]) for index in range(SOURCE_COUNT))
    return positions, velocities  # type: ignore[return-value]


def _v3_source_tile_id(segment_index: int) -> str:
    if type(segment_index) is not int or not 0 <= segment_index < SEGMENT_COUNT:
        raise Full128SmokeError("source tile segment index differs")
    return f"tno-1k-lifecycle-v3-{OBJECT_ID}-segment-{segment_index:02d}"


def _source_build_identity(v3: types.ModuleType, build: object) -> dict[str, str]:
    value = {
        "input_content_sha256": build.input_content_sha256,
        "map_result_content_sha256": build.source_map_result_content_sha256,
        "map_verification_content_sha256": build.source_map_verification_content_sha256,
        "runtime_identity_sha256": _sha256(_canonical_json(v3._plain(build.runtime_identity))),
    }
    if any(not _is_sha256(item) for item in value.values()):
        raise Full128SmokeError("source construction receipt identity differs")
    return value


def _retained_source_build_identity(lane: Mapping[str, Any]) -> dict[str, object]:
    return {
        "input_content_sha256": lane.get("source_build_input_content_sha256"),
        "map_result_content_sha256": lane.get("source_map_result_content_sha256"),
        "map_verification_content_sha256": lane.get("source_map_verification_content_sha256"),
        "runtime_identity_sha256": lane.get("source_build_runtime_identity_sha256"),
    }


def _v3_gate_batch0_lane() -> dict[str, Any]:
    payload = _read_regular(V3_ROOT / "RESULT.json", "V3 retained result")
    if _sha256(payload) != V3_RESULT_SHA256:
        raise Full128SmokeError("V3 retained result file identity differs")
    result = _strict_json(payload)
    lanes = result.get("batch_lanes")
    if (
        payload != _canonical_json(result)
        or result.get("content_sha256") != V3_RESULT_CONTENT_SHA256
        or type(lanes) is not list or len(lanes) != 12
        or type(lanes[0]) is not dict
        or lanes[0].get("batch_order") != BATCH_ORDER
        or lanes[0].get("object_id") != OBJECT_ID
    ):
        raise Full128SmokeError("V3 retained batch-zero lane identity differs")
    return lanes[0]


def _reconstruct_cpu_source_evidence(
    preflight: types.ModuleType, v3: types.ModuleType,
) -> dict[str, Any]:
    """Rebuild batch-zero source bytes without CUDA or the 369 MB cache file."""
    import numpy as np

    class NumPyConstructionBackend:
        uint64 = np.uint64

        @staticmethod
        def asarray(value: object) -> object:
            return np.asarray(value)

        @staticmethod
        def arange(*args: object, **kwargs: object) -> object:
            return np.arange(*args, **kwargs)

    _records, payloads = preflight._bound_inputs()
    _stage0, bridge, custody, _prefix = preflight._validate_semantic_inputs(payloads)
    retained = v3._load_retained_result()
    lane, source_record = v3._retained_lane_records(retained, BATCH_ORDER)
    with v3._delegates() as (grid, _h64, d):
        context = v3._build_batch_context(
            grid, d, NumPyConstructionBackend, bridge, custody, BATCH_ORDER,
        )
        full = context["full"]
        positions, velocities, gms = v3._source_payload(np, full)
        lane_payload = positions + velocities
        if (
            len(lane_payload) != int(source_record["byte_length"])
            or _sha256(lane_payload) != source_record["lane_payload_sha256"]
            or _sha256(positions) != source_record["positions_sha256"]
            or _sha256(velocities) != source_record["velocities_sha256"]
            or full.content_sha256 != source_record["full_source_tile_sha256"]
        ):
            raise Full128SmokeError("CPU-reconstructed source lane differs from frozen h64 identity")
        invariant = d._source_node_invariant_summary(
            full.source_relative_positions_AU, full.source_relative_velocities_AU_per_year,
            full.source_gm_AU3_per_year2,
        )
        d._require_source_invariant_summary(lane.get("summary"), invariant)
        source_build_identity = _source_build_identity(v3, context["build"])
        if source_build_identity != _retained_source_build_identity(_v3_gate_batch0_lane()):
            raise Full128SmokeError("CPU-reconstructed source construction differs from V3")
        nodes = tuple(
            np.ascontiguousarray(
                full.source_relative_positions_AU[year * STEPS_PER_YEAR], dtype=np.dtype("<f8")
            ).tobytes(order="C")
            + np.ascontiguousarray(
                full.source_relative_velocities_AU_per_year[year * STEPS_PER_YEAR], dtype=np.dtype("<f8")
            ).tobytes(order="C")
            for year in CHECKPOINT_YEARS
        )
        segments = tuple(
            v3._source_segment_payload(np, full, index) for index in range(SEGMENT_COUNT)
        )
        v3_segments = _jsonl(
            _read_regular(V3_ROOT / "SEGMENTS.jsonl", "V3 retained segments"),
            "V3 retained segments",
        )
        tile_sha256s = tuple(
            grid.slice_frozen_source_tile_v1(
                full, local_start_step=index * SEGMENT_STEPS,
                map_step_count=SEGMENT_STEPS, tile_id=_v3_source_tile_id(index),
            ).content_sha256
            for index in range(SEGMENT_COUNT)
        )
        if tile_sha256s != tuple(row.get("source_tile_sha256") for row in v3_segments[:SEGMENT_COUNT]):
            raise Full128SmokeError("CPU-reconstructed source tile identity differs from V3")
    return {
        "node_payloads": nodes,
        "segment_payloads": segments,
        "source_tile_sha256": tile_sha256s,
        "lane_payload_sha256": _sha256(lane_payload),
        "source_build_identity": source_build_identity,
        "gm_payload_sha256": _sha256(gms),
        "full_source_tile_sha256": source_record["full_source_tile_sha256"],
        "invariants": invariant,
        "source_record": source_record,
    }


def _jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    if type(payload) is not bytes or (payload and not payload.endswith(b"\n")):
        raise Full128SmokeError(f"{label} is not newline-terminated JSONL")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line)
        if line != _canonical_json(value):
            raise Full128SmokeError(f"{label} row {index} is not canonical")
        records.append(value)
    return records


def _runtime_pre(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], v3_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": RUNTIME_PRE_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "v3_runtime_preflight": dict(v3_preflight),
        "batch_order": BATCH_ORDER,
        "tracer_count": TRACER_COUNT,
        "source_step_count": 0,
        "dynamics_executed": False,
        "classification_generated": False,
        "science_result": False,
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(RUNTIME_PRE_DOMAIN, value)
    return value


def _runtime_post(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], v3_runtime: Mapping[str, Any],
    v3: types.ModuleType,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": RUNTIME_POST_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "v3_runtime_identity": dict(v3_runtime),
        "v3_runtime_content_sha256": v3_runtime["content_sha256"],
        "v3_runtime_comparison_projection_sha256": "",
        "v3_runtime_prelane_binding_projection_sha256": "",
        "fresh_dynamics_child": True,
        "batch_order": BATCH_ORDER,
        "tracer_count": TRACER_COUNT,
        "global_source_step": 64_000,
        "dynamics_executed": True,
        "bounded_smoke_only": True,
        "ten_myr_execution_authorized": False,
        "classification_generated": False,
        "science_result": False,
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        "content_sha256": "",
    }
    value["v3_runtime_comparison_projection_sha256"] = _sha256(
        _canonical_json(v3_runtime_comparison := v3._runtime_comparison_projection(v3_runtime))
    )
    value["v3_runtime_prelane_binding_projection_sha256"] = _sha256(
        _canonical_json(v3._runtime_prelane_binding_projection(v3_runtime))
    )
    if not v3_runtime_comparison:
        raise Full128SmokeError("runtime comparison projection is empty")
    value["content_sha256"] = _domain_hash(RUNTIME_POST_DOMAIN, value)
    return value


def _validate_runtime_envelopes(
    pre: object, post: object, protocol: Mapping[str, Any], v3: types.ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "git_preregistration",
        "v3_runtime_preflight", "batch_order", "tracer_count", "source_step_count",
        "dynamics_executed", "classification_generated", "science_result",
        "local_threat_boundary", "content_sha256",
    }
    post_keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "git_preregistration",
        "v3_runtime_identity", "v3_runtime_content_sha256",
        "v3_runtime_comparison_projection_sha256", "v3_runtime_prelane_binding_projection_sha256",
        "fresh_dynamics_child", "batch_order", "tracer_count", "global_source_step",
        "dynamics_executed", "bounded_smoke_only", "ten_myr_execution_authorized",
        "classification_generated", "science_result", "local_threat_boundary", "content_sha256",
    }
    if type(pre) is not dict or set(pre) != pre_keys or type(post) is not dict or set(post) != post_keys:
        raise Full128SmokeError("runtime envelope shape differs")
    protocol_file_sha = _sha256(_canonical_json(protocol))
    if (
        pre.get("schema") != RUNTIME_PRE_SCHEMA
        or pre.get("protocol_content_sha256") != protocol["content_sha256"]
        or pre.get("protocol_file_sha256") != protocol_file_sha
        or pre.get("batch_order") != 0 or type(pre.get("batch_order")) is not int
        or pre.get("tracer_count") != 129 or type(pre.get("tracer_count")) is not int
        or pre.get("source_step_count") != 0 or type(pre.get("source_step_count")) is not int
        or pre.get("dynamics_executed") is not False
        or pre.get("classification_generated") is not False
        or pre.get("science_result") is not False
        or pre.get("local_threat_boundary") != LOCAL_THREAT_BOUNDARY
        or pre.get("content_sha256") != _domain_hash(RUNTIME_PRE_DOMAIN, pre)
    ):
        raise Full128SmokeError("pre-lane runtime envelope differs")
    if (
        post.get("schema") != RUNTIME_POST_SCHEMA
        or post.get("protocol_content_sha256") != protocol["content_sha256"]
        or post.get("protocol_file_sha256") != protocol_file_sha
        or post.get("fresh_dynamics_child") is not True
        or post.get("batch_order") != 0 or type(post.get("batch_order")) is not int
        or post.get("tracer_count") != 129 or type(post.get("tracer_count")) is not int
        or post.get("global_source_step") != 64_000 or type(post.get("global_source_step")) is not int
        or post.get("dynamics_executed") is not True
        or post.get("bounded_smoke_only") is not True
        or post.get("ten_myr_execution_authorized") is not False
        or post.get("classification_generated") is not False
        or post.get("science_result") is not False
        or post.get("local_threat_boundary") != LOCAL_THREAT_BOUNDARY
        or post.get("content_sha256") != _domain_hash(RUNTIME_POST_DOMAIN, post)
    ):
        raise Full128SmokeError("post-lane runtime envelope differs")
    custody_pre = _validate_git_custody_offline(pre["git_preregistration"], protocol)
    custody_post = _validate_git_custody_offline(
        post["git_preregistration"], protocol,
        expected_commit=custody_pre["commit"], expected_tag_object=custody_pre["tag_object"],
    )
    if _canonical_json(custody_pre) != _canonical_json(custody_post):
        raise Full128SmokeError("runtime Git custody envelopes differ")
    with v3._delegates() as (_grid, _h64, d):
        v3._validate_runtime_preflight_document(pre["v3_runtime_preflight"], d)
        v3_runtime = v3._validate_runtime_identity(
            post["v3_runtime_identity"], d, "full129 smoke runtime",
            expected_mode=v3.RUNTIME_MODE_PRODUCTION,
        )
    if (
        post.get("v3_runtime_content_sha256") != v3_runtime["content_sha256"]
        or post.get("v3_runtime_comparison_projection_sha256")
        != _sha256(_canonical_json(v3._runtime_comparison_projection(v3_runtime)))
        or post.get("v3_runtime_prelane_binding_projection_sha256")
        != _sha256(_canonical_json(v3._runtime_prelane_binding_projection(v3_runtime)))
        or pre["v3_runtime_preflight"].get("prelane_binding_projection_sha256")
        != post.get("v3_runtime_prelane_binding_projection_sha256")
    ):
        raise Full128SmokeError("pre/post V3 runtime crossbinding differs")
    return pre, post


def _segment_record(
    *, segment_index: int, source_tile_sha256: str, source_segment_sha256: str,
    source_start_sha256: str, source_end_sha256: str, checkpoint_sha256: str,
    lifecycle_sha256: str, continuation_start_sha256: str,
    continuation_end_sha256: str, minimum_distance: float, maximum_kepler_iterations: int,
    engine_runtime_identity_sha256: str, elapsed_seconds: float,
) -> dict[str, Any]:
    if type(elapsed_seconds) is not float or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise Full128SmokeError("segment elapsed time differs")
    value: dict[str, Any] = {
        "schema": SEGMENT_SCHEMA,
        "segment_order": segment_index,
        "batch_order": BATCH_ORDER,
        "object_id": OBJECT_ID,
        "segment_index": segment_index,
        "start_year": segment_index * SEGMENT_YEARS,
        "end_year": (segment_index + 1) * SEGMENT_YEARS,
        "start_global_source_step": segment_index * SEGMENT_STEPS,
        "end_global_source_step": (segment_index + 1) * SEGMENT_STEPS,
        "source_tile_sha256": source_tile_sha256,
        "generated_source_segment_sha256": source_segment_sha256,
        "retained_source_segment_sha256": source_segment_sha256,
        "source_start_node_sha256": source_start_sha256,
        "source_end_node_sha256": source_end_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "lifecycle_sha256": lifecycle_sha256,
        "continuation_start_sha256": continuation_start_sha256,
        "continuation_end_sha256": continuation_end_sha256,
        "active_tracer_count": TRACER_COUNT,
        "route_record_count": 0,
        "removal_record_count": 0,
        "minimum_source_distance_AU_hex": minimum_distance.hex(),
        "maximum_kepler_iterations": maximum_kepler_iterations,
        "engine_runtime_identity_sha256": engine_runtime_identity_sha256,
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE",
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(SEGMENT_DOMAIN, value)
    return value


def _validate_segment_record(
    value: object, *, segment_index: int, checkpoint: bytes, lifecycle: bytes,
    continuation_start: bytes, continuation_end: bytes,
    retained_segment_sha256: str, source_start_sha256: str, source_end_sha256: str,
    v3_source_tile_sha256: str, engine_projection_sha256: str,
) -> dict[str, Any]:
    keys = {
        "schema", "segment_order", "batch_order", "object_id", "segment_index",
        "start_year", "end_year", "start_global_source_step", "end_global_source_step",
        "source_tile_sha256", "generated_source_segment_sha256",
        "retained_source_segment_sha256", "source_start_node_sha256",
        "source_end_node_sha256", "checkpoint_sha256", "lifecycle_sha256",
        "continuation_start_sha256", "continuation_end_sha256",
        "active_tracer_count", "route_record_count", "removal_record_count",
        "minimum_source_distance_AU_hex", "maximum_kepler_iterations",
        "engine_runtime_identity_sha256", "elapsed_seconds_hex", "elapsed_time_claim",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full128SmokeError("segment record shape differs")
    integer_keys = (
        "segment_order", "batch_order", "segment_index", "start_year", "end_year",
        "start_global_source_step", "end_global_source_step", "active_tracer_count",
        "route_record_count", "removal_record_count", "maximum_kepler_iterations",
    )
    if any(type(value.get(key)) is not int for key in integer_keys):
        raise Full128SmokeError("segment integer type differs")
    if (
        value.get("schema") != SEGMENT_SCHEMA
        or value.get("segment_order") != segment_index
        or value.get("batch_order") != BATCH_ORDER
        or value.get("object_id") != OBJECT_ID
        or value.get("segment_index") != segment_index
        or value.get("start_year") != segment_index * SEGMENT_YEARS
        or value.get("end_year") != (segment_index + 1) * SEGMENT_YEARS
        or value.get("start_global_source_step") != segment_index * SEGMENT_STEPS
        or value.get("end_global_source_step") != (segment_index + 1) * SEGMENT_STEPS
        or value.get("source_tile_sha256") != v3_source_tile_sha256
        or value.get("generated_source_segment_sha256") != retained_segment_sha256
        or value.get("retained_source_segment_sha256") != retained_segment_sha256
        or value.get("source_start_node_sha256") != source_start_sha256
        or value.get("source_end_node_sha256") != source_end_sha256
        or value.get("checkpoint_sha256") != _sha256(checkpoint)
        or value.get("lifecycle_sha256") != _sha256(lifecycle)
        or value.get("continuation_start_sha256") != _sha256(continuation_start)
        or value.get("continuation_end_sha256") != _sha256(continuation_end)
        or value.get("active_tracer_count") != TRACER_COUNT
        or value.get("route_record_count") != 0
        or value.get("removal_record_count") != 0
        or value.get("maximum_kepler_iterations", -1) < 0
        or value.get("engine_runtime_identity_sha256") != engine_projection_sha256
        or value.get("elapsed_time_claim") != "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE"
        or value.get("content_sha256") != _domain_hash(SEGMENT_DOMAIN, value)
    ):
        raise Full128SmokeError("segment record identity or gate differs")
    for key in ("minimum_source_distance_AU_hex", "elapsed_seconds_hex"):
        raw = value.get(key)
        if type(raw) is not str:
            raise Full128SmokeError(f"segment {key} type differs")
        try:
            number = float.fromhex(raw)
        except ValueError as exc:
            raise Full128SmokeError(f"segment {key} differs") from exc
        if not math.isfinite(number) or number < 0.0:
            raise Full128SmokeError(f"segment {key} is invalid")
    if not float.fromhex(value["minimum_source_distance_AU_hex"]) > 1.0:
        raise Full128SmokeError("segment minimum-distance gate failed")
    return value


def _result_claims() -> dict[str, bool]:
    return {
        **CLAIMS_PREPARED,
        "bounded_gpu_smoke_executed": True,
        "bounded_gpu_smoke_passed": True,
        "gpu_executed": True,
    }


def _validate_exact_claims(value: object, expected: Mapping[str, bool]) -> None:
    if type(value) is not dict or set(value) != set(expected):
        raise Full128SmokeError("result claim roster differs")
    for key, expected_value in expected.items():
        if type(value.get(key)) is not bool or value[key] is not expected_value:
            raise Full128SmokeError(f"result claim differs: {key}")


def _validate_checkpoint_years(value: object) -> None:
    if (
        type(value) is not list
        or len(value) != CHECKPOINT_COUNT
        or any(type(item) is not int for item in value)
        or tuple(value) != CHECKPOINT_YEARS
    ):
        raise Full128SmokeError("result checkpoint-year roster differs")


def _validate_file_receipts(value: object, snapshot: Mapping[str, bytes]) -> None:
    expected_names = {
        "CHECKPOINTS.bin", "LIFECYCLE.bin", "CONTINUATION.bin", "RUNTIME_PRE.json",
        "RUNTIME_POST.json", "SEGMENTS.jsonl",
    }
    if type(value) is not dict or set(value) != expected_names:
        raise Full128SmokeError("result file roster differs")
    for name in expected_names:
        receipt = value[name]
        if (
            type(receipt) is not dict or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] != len(snapshot[name])
            or receipt.get("sha256") != _sha256(snapshot[name])
        ):
            raise Full128SmokeError(f"result file receipt differs: {name}")


def _validate_output_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    if type(snapshot) is not dict or set(snapshot) != EXPECTED_OUTPUT_ROSTER:
        raise Full128SmokeError("retained output snapshot roster differs")
    if (
        len(snapshot["CHECKPOINTS.bin"]) != CHECKPOINTS_BYTES
        or len(snapshot["LIFECYCLE.bin"]) != LIFECYCLE_BYTES
        or len(snapshot["CONTINUATION.bin"]) != CONTINUATION_BYTES
    ):
        raise Full128SmokeError("retained binary geometry differs")
    protocol = _strict_json(snapshot["PROTOCOL.json"])
    if snapshot["PROTOCOL.json"] != _canonical_json(protocol):
        raise Full128SmokeError("retained protocol is not canonical")
    validate_protocol(protocol)
    pre = _strict_json(snapshot["RUNTIME_PRE.json"])
    post = _strict_json(snapshot["RUNTIME_POST.json"])
    if snapshot["RUNTIME_PRE.json"] != _canonical_json(pre) or snapshot["RUNTIME_POST.json"] != _canonical_json(post):
        raise Full128SmokeError("retained runtime envelope is not canonical")
    result = _strict_json(snapshot["RESULT.json"])
    if snapshot["RESULT.json"] != _canonical_json(result):
        raise Full128SmokeError("retained result is not canonical")
    _validate_checkpoint_years(result.get("checkpoint_years"))

    checkpoints = tuple(
        snapshot["CHECKPOINTS.bin"][index * STATE_RECORD_BYTES:(index + 1) * STATE_RECORD_BYTES]
        for index in range(CHECKPOINT_COUNT)
    )
    for payload in checkpoints:
        _unpack_state_record(payload)
    lifecycle_rows = tuple(
        snapshot["LIFECYCLE.bin"][index * TRACER_COUNT:(index + 1) * TRACER_COUNT]
        for index in range(CHECKPOINT_COUNT)
    )
    if any(row != bytes(TRACER_COUNT) for row in lifecycle_rows):
        raise Full128SmokeError("a retained full129 lifecycle row is not all-active")
    continuation_records = _continuation_records(snapshot["CONTINUATION.bin"])
    segments = _jsonl(snapshot["SEGMENTS.jsonl"], "retained segments")
    if len(segments) != SEGMENT_COUNT:
        raise Full128SmokeError("retained segment count differs")

    with _delegates() as (preflight, v3):
        _validate_runtime_envelopes(pre, post, protocol, v3)
        source_evidence = _reconstruct_cpu_source_evidence(preflight, v3)
        v3_result = _strict_json(_read_regular(V3_ROOT / "RESULT.json", "V3 retained result"))
        v3_segments = _jsonl(_read_regular(V3_ROOT / "SEGMENTS.jsonl", "V3 retained segments"), "V3 retained segments")
        v3_prefix = _read_regular(V3_ROOT / "PREFIX_STATES.bin", "V3 retained prefix", 169_344)
        if _sha256(v3_prefix) != V3_PREFIX_SHA256:
            raise Full128SmokeError("V3 retained prefix differs")
        primary = _batch0_primary_payload(preflight)
        primary_rows = _unpack_state_record(primary)
        engine_projection_sha = _sha256(_canonical_json(post["v3_runtime_identity"]["engine_runtime_projection"]))
        checkpoint_sha = [_sha256(item) for item in checkpoints]
        lifecycle_sha = [_sha256(item) for item in lifecycle_rows]
        selected_sha: list[str] = []
        source_segment_sha: list[str] = []
        node_payloads = source_evidence["node_payloads"]
        source_endpoint_sha = [_sha256(item) for item in node_payloads]
        source0_p, source0_v = _source_node_rows(node_payloads[0])
        if tuple(tuple(row[:3]) for row in primary_rows[:MASSIVE_COUNT]) != source0_p or tuple(tuple(row[3:]) for row in primary_rows[:MASSIVE_COUNT]) != source0_v:
            raise Full128SmokeError("full129 primary state does not match retained source node zero")
        with v3._delegates() as (grid, _h64, _d):
            with grid.FrozenClassifierKernelV1() as kernel:
                initial = _checkpoint_full129(
                    preflight, grid, kernel, source0_p, source0_v,
                    tuple(tuple(row[:3]) for row in primary_rows[MASSIVE_COUNT:]),
                    tuple(tuple(row[3:]) for row in primary_rows[MASSIVE_COUNT:]),
                )
                if initial != checkpoints[0] or _sha256(initial) != YEAR_ZERO_CHECKPOINT_SHA256:
                    raise Full128SmokeError("full129 year-zero checkpoint differs from frozen preflight")
        continuation_sha = _validate_continuation_records(
            preflight, v3, node_payloads, snapshot["CONTINUATION.bin"], checkpoints,
        )
        for index, checkpoint in enumerate(checkpoints):
            selected = preflight._selected_holdout_payload(checkpoint)
            if selected != preflight._v3_checkpoint(v3_prefix, BATCH_ORDER, index):
                raise Full128SmokeError("full129 checkpoint V3 holdout projection differs")
            selected_sha.append(_sha256(selected))
        for index, segment in enumerate(segments):
            retained_sha = _sha256(source_evidence["segment_payloads"][index])
            source_segment_sha.append(retained_sha)
            v3_segment = v3_segments[index]
            _validate_segment_record(
                segment, segment_index=index, checkpoint=checkpoints[index + 1],
                lifecycle=lifecycle_rows[index + 1], retained_segment_sha256=retained_sha,
                continuation_start=continuation_records[index],
                continuation_end=continuation_records[index + 1],
                source_start_sha256=source_endpoint_sha[index],
                source_end_sha256=source_endpoint_sha[index + 1],
                v3_source_tile_sha256=v3_segment["source_tile_sha256"],
                engine_projection_sha256=engine_projection_sha,
            )

        try:
            elapsed = float.fromhex(result.get("elapsed_seconds_hex", ""))
        except (TypeError, ValueError) as exc:
            raise Full128SmokeError("result elapsed time differs") from exc
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise Full128SmokeError("result elapsed time is not positive finite")
        secondary = preflight.validate_smoke_evidence_v1(
            snapshot["CHECKPOINTS.bin"], active_counts=[TRACER_COUNT] * CHECKPOINT_COUNT,
            routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=elapsed,
        )
        lane = v3_result.get("batch_lanes", [{}])[0]

    expected_result_keys = {
        "schema", "status", "protocol_content_sha256", "protocol_file_sha256",
        "git_preregistration", "files", "batch_order", "object_id", "tracer_count",
        "checkpoint_years", "checkpoint_sha256", "lifecycle_sha256",
        "selected_v3_projection_sha256", "year_zero_checkpoint_sha256",
        "continuation_sha256", "continuation_record_sha256",
        "source_segment_sha256", "source_endpoint_sha256",
        "source_full_lane_sha256", "source_full_tile_sha256", "source_build_identity",
        "source_invariants",
        "secondary_preflight_evidence", "runtime_pre_content_sha256",
        "runtime_post_content_sha256", "segment_count", "route_record_count",
        "removal_record_count", "next_continuation_global_source_step",
        "next_source_step_executed", "elapsed_seconds_hex", "elapsed_time_claim",
        "all_predeclared_gates_passed", "claims", "content_sha256",
    }
    _validate_file_receipts(result.get("files"), snapshot)
    if (
        type(result) is not dict or set(result) != expected_result_keys
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "FULL129_BATCH0_1KYR_GPU_SMOKE_PASS_TEN_MYR_NOT_AUTHORIZED"
        or result.get("protocol_content_sha256") != protocol["content_sha256"]
        or result.get("protocol_file_sha256") != _sha256(snapshot["PROTOCOL.json"])
        or _canonical_json(result.get("git_preregistration")) != _canonical_json(pre["git_preregistration"])
        or result.get("batch_order") != BATCH_ORDER or type(result.get("batch_order")) is not int
        or result.get("object_id") != OBJECT_ID
        or result.get("tracer_count") != TRACER_COUNT or type(result.get("tracer_count")) is not int
        or result.get("checkpoint_years") != list(CHECKPOINT_YEARS)
        or result.get("checkpoint_sha256") != checkpoint_sha
        or result.get("lifecycle_sha256") != lifecycle_sha
        or result.get("selected_v3_projection_sha256") != selected_sha
        or result.get("year_zero_checkpoint_sha256") != YEAR_ZERO_CHECKPOINT_SHA256
        or result.get("continuation_sha256") != _sha256(snapshot["CONTINUATION.bin"])
        or result.get("continuation_record_sha256") != list(continuation_sha)
        or result.get("source_segment_sha256") != source_segment_sha
        or result.get("source_endpoint_sha256") != source_endpoint_sha
        or result.get("source_full_lane_sha256") != lane.get("retained_source_payload_sha256")
        or result.get("source_full_tile_sha256") != lane.get("full_source_tile_sha256")
        or result.get("source_build_identity") != _retained_source_build_identity(lane)
        or _canonical_json(result.get("source_invariants")) != _canonical_json(lane.get("source_invariants"))
        or _canonical_json(result.get("secondary_preflight_evidence")) != _canonical_json(secondary)
        or result.get("runtime_pre_content_sha256") != pre["content_sha256"]
        or result.get("runtime_post_content_sha256") != post["content_sha256"]
        or result.get("segment_count") != SEGMENT_COUNT or type(result.get("segment_count")) is not int
        or result.get("route_record_count") != 0 or type(result.get("route_record_count")) is not int
        or result.get("removal_record_count") != 0 or type(result.get("removal_record_count")) is not int
        or result.get("next_continuation_global_source_step") != 64_000
        or result.get("next_source_step_executed") is not False
        or result.get("elapsed_time_claim") != "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE"
        or result.get("all_predeclared_gates_passed") is not True
        or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result)
    ):
        raise Full128SmokeError("retained result semantics differ")
    _validate_exact_claims(result.get("claims"), _result_claims())
    return result


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    expected_lengths = {
        "PROTOCOL.json": None,
        "CHECKPOINTS.bin": CHECKPOINTS_BYTES,
        "LIFECYCLE.bin": LIFECYCLE_BYTES,
        "CONTINUATION.bin": CONTINUATION_BYTES,
        "RUNTIME_PRE.json": None,
        "RUNTIME_POST.json": None,
        "SEGMENTS.jsonl": None,
        "RESULT.json": None,
    }
    with _held_directory(root, "retained smoke root", expected_roster=EXPECTED_OUTPUT_ROSTER) as root_fd:
        snapshot: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        for name in sorted(EXPECTED_OUTPUT_ROSTER):
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            identities[name] = _stat_identity(metadata)
            snapshot[name] = _read_regular_at(root_fd, name, f"retained {name}", expected_lengths[name])
        result = _validate_output_snapshot(snapshot)
        if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
            raise Full128SmokeError("retained smoke root final roster differs")
        for name, identity in identities.items():
            if _stat_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                raise Full128SmokeError(f"retained output changed during replay: {name}")
        return result


def _host_rows(np: Any, value: object, count: int, label: str) -> tuple[tuple[float, float, float], ...]:
    if type(value) is not np.ndarray or value.dtype != np.dtype("<f8") or value.shape != (count, 3):
        raise Full128SmokeError(f"{label} array geometry differs")
    rows = tuple(tuple(float(value[index, axis]) for axis in range(3)) for index in range(count))
    if any(type(item) is not float or not math.isfinite(item) for row in rows for item in row):
        raise Full128SmokeError(f"{label} contains non-finite data")
    return rows  # type: ignore[return-value]


def _lifecycle_row(np: Any, value: object) -> bytes:
    if type(value) is not np.ndarray or value.dtype != np.dtype("uint8") or value.shape != (TRACER_COUNT,):
        raise Full128SmokeError("full129 lifecycle vector geometry differs")
    payload = value.tobytes(order="C")
    if payload != bytes(TRACER_COUNT):
        raise Full128SmokeError("full129 lifecycle vector is not all-active")
    return payload


def _full129_state(cp: Any, context: Mapping[str, Any], primary_payload: bytes, preflight: types.ModuleType) -> tuple[object, object, object]:
    import numpy as np

    rows = _unpack_state_record(primary_payload)
    source_positions = np.asarray([row[:3] for row in rows[:MASSIVE_COUNT]], dtype=np.float64)
    source_velocities = np.asarray([row[3:] for row in rows[:MASSIVE_COUNT]], dtype=np.float64)
    tracer_positions = np.asarray([row[:3] for row in rows[MASSIVE_COUNT:]], dtype=np.float64)
    tracer_velocities = np.asarray([row[3:] for row in rows[MASSIVE_COUNT:]], dtype=np.float64)
    full = context["full"]
    if (
        source_positions.tobytes(order="C") != np.ascontiguousarray(full.source_relative_positions_AU[0], dtype=np.dtype("<f8")).tobytes(order="C")
        or source_velocities.tobytes(order="C") != np.ascontiguousarray(full.source_relative_velocities_AU_per_year[0], dtype=np.dtype("<f8")).tobytes(order="C")
    ):
        raise Full128SmokeError("full129 massive year-zero state differs from frozen source")
    holdouts = tuple(preflight.HOLDOUT_TRACER_INDICES)
    if (
        tracer_positions[list(holdouts)].tobytes(order="C")
        != np.ascontiguousarray(context["initial_tracer_positions"], dtype=np.dtype("<f8")).tobytes(order="C")
        or tracer_velocities[list(holdouts)].tobytes(order="C")
        != np.ascontiguousarray(context["initial_tracer_velocities"], dtype=np.dtype("<f8")).tobytes(order="C")
    ):
        raise Full128SmokeError("full129 holdout engine state differs from V3")
    base = context["state"]
    state = type(base)(
        snapshot_id=f"tno-full129-smoke-v1-DE441-{OBJECT_ID}",
        epoch_origin_years=base.epoch_origin_years,
        global_source_step=0,
        source_step_years=base.source_step_years,
        time_scale=base.time_scale,
        frame=base.frame,
        axes=base.axes,
        coordinate_system=base.coordinate_system,
        length_unit=base.length_unit,
        time_unit=base.time_unit,
        unit_system_id=base.unit_system_id,
        source_ids=base.source_ids,
        primary_source_id=base.primary_source_id,
        source_schedule_sha256=base.source_schedule_sha256,
        tracer_relative_positions_AU=cp.asarray(tracer_positions),
        tracer_relative_velocities_AU_per_year=cp.asarray(tracer_velocities),
        tracer_indices=cp.arange(TRACER_COUNT, dtype=cp.uint64),
        provenance=base.provenance,
    )
    return state, tracer_positions, tracer_velocities


def _run_full129_lane(
    preflight: types.ModuleType, v3: types.ModuleType, grid: types.ModuleType,
    d: types.ModuleType, cp: Any, bridge: Mapping[str, Any], custody: Mapping[str, Any],
    retained_result: Mapping[str, Any], source_descriptor: int, kernel: object,
) -> dict[str, Any]:
    import numpy as np

    context = v3._build_batch_context(grid, d, cp, bridge, custody, BATCH_ORDER)
    full = context["full"]
    build = context["build"]
    specs = context["specs"]
    primary_payload = _batch0_primary_payload(preflight)
    state, initial_positions, initial_velocities = _full129_state(cp, context, primary_payload, preflight)
    lane, source_record = v3._retained_lane_records(retained_result, BATCH_ORDER)
    position_payload, velocity_payload, gm_payload = v3._source_payload(np, full)
    generated_source_payload = position_payload + velocity_payload
    retained_source_payload = v3._pread_exact(
        source_descriptor, int(source_record["byte_offset"]), int(source_record["byte_length"]),
        "retained full source lane",
    )
    retained_gm = v3._pread_exact(source_descriptor, 0, v3.SOURCE_COMMON_GM_BYTES, "retained source GM")
    if (
        generated_source_payload != retained_source_payload or gm_payload != retained_gm
        or _sha256(generated_source_payload) != source_record["lane_payload_sha256"]
        or full.content_sha256 != source_record["full_source_tile_sha256"]
    ):
        raise Full128SmokeError("generated source path differs from retained h64")
    invariant = d._source_node_invariant_summary(
        full.source_relative_positions_AU, full.source_relative_velocities_AU_per_year,
        full.source_gm_AU3_per_year2,
    )
    d._require_source_invariant_summary(lane.get("summary"), invariant)
    source_build_identity = _source_build_identity(v3, build)
    if source_build_identity != _retained_source_build_identity(_v3_gate_batch0_lane()):
        raise Full128SmokeError("source construction receipts differ from retained V3")
    if (
        float.fromhex(invariant["maximum_relative_massive_energy_drift_hex"]) > 1.0e-8
        or float.fromhex(invariant["maximum_relative_massive_angular_momentum_drift_hex"]) > 1.0e-10
    ):
        raise Full128SmokeError("source invariant gate failed")

    lifecycle_state = specs["lifecycle_initialize"](state)
    checkpoints: list[bytes] = []
    lifecycle_rows: list[bytes] = []
    continuation_records: list[bytes] = []
    segment_records: list[dict[str, Any]] = []
    source_segment_sha: list[str] = []
    source_endpoint_sha = [
        _sha256(_source_node_payload(v3, source_descriptor, source_record, year * STEPS_PER_YEAR))
        for year in CHECKPOINT_YEARS
    ]
    initial = _checkpoint_full129(
        preflight, grid, kernel,
        _host_rows(np, full.source_relative_positions_AU[0], MASSIVE_COUNT, "initial source positions"),
        _host_rows(np, full.source_relative_velocities_AU_per_year[0], MASSIVE_COUNT, "initial source velocities"),
        _host_rows(np, initial_positions, TRACER_COUNT, "initial tracer positions"),
        _host_rows(np, initial_velocities, TRACER_COUNT, "initial tracer velocities"),
    )
    if _sha256(initial) != YEAR_ZERO_CHECKPOINT_SHA256:
        raise Full128SmokeError("generated full129 year-zero checkpoint differs")
    checkpoints.append(initial)
    lifecycle_rows.append(_lifecycle_row(np, lifecycle_state.tracer_lifecycle))
    continuation_records.append(_pack_continuation(
        _host_rows(np, initial_positions, TRACER_COUNT, "initial tracer positions"),
        _host_rows(np, initial_velocities, TRACER_COUNT, "initial tracer velocities"),
    ))
    engine_projection: dict[str, Any] | None = None
    engine_projection_sha: str | None = None
    started_lane = time.perf_counter()
    for segment_index in range(SEGMENT_COUNT):
        tile = grid.slice_frozen_source_tile_v1(
            full, local_start_step=segment_index * SEGMENT_STEPS,
            map_step_count=SEGMENT_STEPS,
            tile_id=_v3_source_tile_id(segment_index),
        )
        generated_segment = v3._source_segment_payload(np, full, segment_index)
        retained_segment = v3._retained_source_segment(source_descriptor, source_record, segment_index)
        if generated_segment != retained_segment:
            raise Full128SmokeError("generated source segment differs from retained h64")
        source_sha = _sha256(retained_segment)
        source_segment_sha.append(source_sha)
        started = time.perf_counter()
        advanced = specs["lifecycle_advance"](
            lifecycle_state, tile, specs["fixed_step"], route_spec=specs["route"],
            route_worker_count=specs["route_worker_count"],
        )
        elapsed = time.perf_counter() - started
        lifecycle_state = advanced.state
        end_step = (segment_index + 1) * SEGMENT_STEPS
        lifecycle_payload = _lifecycle_row(np, lifecycle_state.tracer_lifecycle)
        if (
            lifecycle_state.dynamics_state.global_source_step != end_step
            or advanced.route_records != () or advanced.new_removal_records != ()
            or lifecycle_state.removal_records != ()
        ):
            raise Full128SmokeError("full129 lifecycle routed, removed, or incompletely advanced")
        minimum = tuple(float(item) for item in cp.asnumpy(advanced.minimum_source_distance_AU).tolist())
        if len(minimum) != TRACER_COUNT or any(not math.isfinite(item) or not item > 1.0 for item in minimum):
            raise Full128SmokeError("full129 minimum-distance gate failed")
        projection = v3._engine_runtime_projection(advanced.runtime_identity)
        projection_sha = _sha256(_canonical_json(projection))
        if engine_projection is None:
            engine_projection, engine_projection_sha = projection, projection_sha
        elif _canonical_json(engine_projection) != _canonical_json(projection):
            raise Full128SmokeError("engine runtime projection changed between segments")
        dynamics = lifecycle_state.dynamics_state
        tracer_positions = cp.asnumpy(dynamics.tracer_relative_positions_AU)
        tracer_velocities = cp.asnumpy(dynamics.tracer_relative_velocities_AU_per_year)
        host_tracer_positions = _host_rows(np, tracer_positions, TRACER_COUNT, "tracer positions")
        host_tracer_velocities = _host_rows(np, tracer_velocities, TRACER_COUNT, "tracer velocities")
        checkpoint = _checkpoint_full129(
            preflight, grid, kernel,
            _host_rows(np, full.source_relative_positions_AU[end_step], MASSIVE_COUNT, "source positions"),
            _host_rows(np, full.source_relative_velocities_AU_per_year[end_step], MASSIVE_COUNT, "source velocities"),
            host_tracer_positions, host_tracer_velocities,
        )
        end_continuation = _pack_continuation(host_tracer_positions, host_tracer_velocities)
        checkpoints.append(checkpoint)
        lifecycle_rows.append(lifecycle_payload)
        continuation_records.append(end_continuation)
        segment_records.append(_segment_record(
            segment_index=segment_index, source_tile_sha256=tile.content_sha256,
            source_segment_sha256=source_sha,
            source_start_sha256=source_endpoint_sha[segment_index],
            source_end_sha256=source_endpoint_sha[segment_index + 1],
            checkpoint_sha256=_sha256(checkpoint), lifecycle_sha256=_sha256(lifecycle_payload),
            continuation_start_sha256=_sha256(continuation_records[segment_index]),
            continuation_end_sha256=_sha256(end_continuation),
            minimum_distance=min(minimum),
            maximum_kepler_iterations=int(advanced.maximum_kepler_iterations),
            engine_runtime_identity_sha256=projection_sha, elapsed_seconds=elapsed,
        ))
    if engine_projection is None or engine_projection_sha is None:
        raise Full128SmokeError("engine runtime projection is absent")
    continuation = b"".join(continuation_records)
    checkpoint_payload = b"".join(checkpoints)
    lifecycle_payload = b"".join(lifecycle_rows)
    if (
        len(checkpoint_payload) != CHECKPOINTS_BYTES
        or len(lifecycle_payload) != LIFECYCLE_BYTES
        or len(continuation) != CONTINUATION_BYTES
    ):
        raise Full128SmokeError("full129 retained evidence geometry differs")
    return {
        "checkpoints": checkpoint_payload,
        "lifecycle": lifecycle_payload,
        "continuation": continuation,
        "segments": segment_records,
        "source_segment_sha256": source_segment_sha,
        "source_endpoint_sha256": source_endpoint_sha,
        "source_full_lane_sha256": source_record["lane_payload_sha256"],
        "source_full_tile_sha256": full.content_sha256,
        "source_build_identity": source_build_identity,
        "source_invariants": invariant,
        "engine_projection": engine_projection,
        "engine_projection_sha256": engine_projection_sha,
        "elapsed_seconds": time.perf_counter() - started_lane,
    }


def _build_result_and_artifacts(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], runtime_pre: Mapping[str, Any],
    runtime_post: Mapping[str, Any], lane: Mapping[str, Any], preflight: types.ModuleType,
) -> tuple[dict[str, Any], tuple[tuple[str, bytes], ...]]:
    checkpoints = lane["checkpoints"]
    lifecycle = lane["lifecycle"]
    continuation = lane["continuation"]
    segments_payload = b"".join(_canonical_json(item) for item in lane["segments"])
    runtime_pre_payload = _canonical_json(runtime_pre)
    runtime_post_payload = _canonical_json(runtime_post)
    elapsed = float(lane["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise Full128SmokeError("full129 lane elapsed time differs")
    secondary = preflight.validate_smoke_evidence_v1(
        checkpoints, active_counts=[TRACER_COUNT] * CHECKPOINT_COUNT,
        routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=elapsed,
    )
    files_payload = (
        ("CHECKPOINTS.bin", checkpoints),
        ("LIFECYCLE.bin", lifecycle),
        ("CONTINUATION.bin", continuation),
        ("RUNTIME_PRE.json", runtime_pre_payload),
        ("RUNTIME_POST.json", runtime_post_payload),
        ("SEGMENTS.jsonl", segments_payload),
    )
    file_records = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in files_payload
    }
    checkpoint_sha = [
        _sha256(checkpoints[index * STATE_RECORD_BYTES:(index + 1) * STATE_RECORD_BYTES])
        for index in range(CHECKPOINT_COUNT)
    ]
    lifecycle_sha = [
        _sha256(lifecycle[index * TRACER_COUNT:(index + 1) * TRACER_COUNT])
        for index in range(CHECKPOINT_COUNT)
    ]
    continuation_record_sha = [
        _sha256(record) for record in _continuation_records(continuation)
    ]
    v3_prefix = _read_regular(V3_ROOT / "PREFIX_STATES.bin", "V3 prefix", 169_344)
    selected_sha = [
        _sha256(preflight._selected_holdout_payload(
            checkpoints[index * STATE_RECORD_BYTES:(index + 1) * STATE_RECORD_BYTES]
        ))
        for index in range(CHECKPOINT_COUNT)
    ]
    if any(
        preflight._selected_holdout_payload(
            checkpoints[index * STATE_RECORD_BYTES:(index + 1) * STATE_RECORD_BYTES]
        ) != preflight._v3_checkpoint(v3_prefix, BATCH_ORDER, index)
        for index in range(CHECKPOINT_COUNT)
    ):
        raise Full128SmokeError("full129 output differs from V3 holdout projection")
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "FULL129_BATCH0_1KYR_GPU_SMOKE_PASS_TEN_MYR_NOT_AUTHORIZED",
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "files": file_records,
        "batch_order": BATCH_ORDER,
        "object_id": OBJECT_ID,
        "tracer_count": TRACER_COUNT,
        "checkpoint_years": list(CHECKPOINT_YEARS),
        "checkpoint_sha256": checkpoint_sha,
        "lifecycle_sha256": lifecycle_sha,
        "selected_v3_projection_sha256": selected_sha,
        "year_zero_checkpoint_sha256": YEAR_ZERO_CHECKPOINT_SHA256,
        "continuation_sha256": _sha256(continuation),
        "continuation_record_sha256": continuation_record_sha,
        "source_segment_sha256": list(lane["source_segment_sha256"]),
        "source_endpoint_sha256": list(lane["source_endpoint_sha256"]),
        "source_full_lane_sha256": lane["source_full_lane_sha256"],
        "source_full_tile_sha256": lane["source_full_tile_sha256"],
        "source_build_identity": lane["source_build_identity"],
        "source_invariants": lane["source_invariants"],
        "secondary_preflight_evidence": secondary,
        "runtime_pre_content_sha256": runtime_pre["content_sha256"],
        "runtime_post_content_sha256": runtime_post["content_sha256"],
        "segment_count": SEGMENT_COUNT,
        "route_record_count": 0,
        "removal_record_count": 0,
        "next_continuation_global_source_step": 64_000,
        "next_source_step_executed": False,
        "elapsed_seconds_hex": elapsed.hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE",
        "all_predeclared_gates_passed": True,
        "claims": _result_claims(),
        "content_sha256": "",
    }
    result["content_sha256"] = _domain_hash(RESULT_DOMAIN, result)
    return result, (*files_payload, ("RESULT.json", _canonical_json(result)))


def _publish(output_root: Path, artifacts: Sequence[tuple[str, bytes]]) -> None:
    expected_order = [
        "CHECKPOINTS.bin", "LIFECYCLE.bin", "CONTINUATION.bin", "RUNTIME_PRE.json",
        "RUNTIME_POST.json", "SEGMENTS.jsonl", "RESULT.json",
    ]
    if [name for name, _payload in artifacts] != expected_order:
        raise Full128SmokeError("publication order differs")
    root = Path(os.path.abspath(os.fspath(output_root)))
    stage = Path(tempfile.mkdtemp(prefix=".tno-full129-smoke-publish-", dir=root.parent))
    try:
        with _held_directory(stage, "publication stage") as stage_fd:
            for name, payload in artifacts:
                _write_create_only_at(stage_fd, name, payload)
            os.fsync(stage_fd)
        with _held_directory(root, "publication root", expected_roster={"PROTOCOL.json"}) as root_fd:
            protocol_payload = _read_regular_at(root_fd, "PROTOCOL.json", "publication protocol")
            protocol = _strict_json(protocol_payload)
            if protocol_payload != _canonical_json(protocol):
                raise Full128SmokeError("publication protocol is not canonical")
            validate_protocol(protocol)
            linked: list[tuple[str, tuple[int, int]]] = []
            try:
                for name, _payload in artifacts:
                    source = stage / name
                    source_stat = source.stat(follow_symlinks=False)
                    os.link(source, name, dst_dir_fd=root_fd, follow_symlinks=False)
                    destination = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    if (source_stat.st_dev, source_stat.st_ino) != (destination.st_dev, destination.st_ino):
                        raise Full128SmokeError("publication link identity differs")
                    linked.append((name, (destination.st_dev, destination.st_ino)))
                if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
                    raise Full128SmokeError("publication final roster differs")
                os.fsync(root_fd)
            except Exception:
                for name, identity in reversed(linked):
                    try:
                        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                        if (observed.st_dev, observed.st_ino) == identity:
                            os.unlink(name, dir_fd=root_fd)
                    except FileNotFoundError:
                        pass
                os.fsync(root_fd)
                raise
    finally:
        if stage.exists():
            for item in stage.iterdir():
                item.unlink()
            stage.rmdir()
    with _held_directory(root, "published smoke root", expected_roster=EXPECTED_OUTPUT_ROSTER) as root_fd:
        for name in EXPECTED_OUTPUT_ROSTER:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise Full128SmokeError("published output link count differs")


def _execute_worker(output_root: Path, import_probe_path: Path, custody_path: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root != DEFAULT_OUTPUT_ROOT:
        raise Full128SmokeError("preregistered smoke execution requires the exact default output root")
    protocol = validate_prepared(root)
    custody_payload = _read_regular(custody_path, "Git custody transfer")
    custody = _strict_json(custody_payload)
    if custody_payload != _canonical_json(custody):
        raise Full128SmokeError("Git custody transfer is not canonical")
    _validate_git_custody_offline(custody, protocol)
    live_custody = _git_custody(
        externally_preregistered=True,
        prereg_commit=custody["commit"], prereg_tag_object=custody["tag_object"],
    )
    if _canonical_json(live_custody) != _canonical_json(custody):
        raise Full128SmokeError("worker live Git preregistration differs from parent custody")
    custody = live_custody
    with _delegates() as (preflight, v3):
        v3_preflight = v3.runtime_preflight()
        runtime_pre = _runtime_pre(protocol, custody, v3_preflight)
        retained_result = v3._load_retained_result()
        with v3._delegates() as (grid, _h64, d):
            v3._validate_runtime_preflight_document(v3_preflight, d)
            fresh_child = d._fresh_child_interpreter_receipt()
            fresh_registry = d._reject_preloaded_runtime_modules()
            _probe_payload, probe = d._load_json(import_probe_path, "execution import probe")
            d._validate_import_probe_receipt(probe, expected_path=import_probe_path)
            de441_custody = d.validate_execution_prerequisites(d.DE441_ROOT)
            bridge = d._load_json(d.BRIDGE_STATES_PATH, "bridge states")[1]
            with v3._held_source_file() as source_fd:
                with d._source_only_runtime_import_guard(
                    root / ".full129-smoke-runtime-pycache-must-remain-absent"
                ) as guard:
                    cp, _rebound, _target = d._load_dynamics_runtime(d.DEFAULT_REBOUND_TARGET)
                    runtime_base = v3._critical_runtime_base(
                        d, cp, fresh_child=fresh_child, fresh_registry=fresh_registry,
                    )
                    with grid.FrozenClassifierKernelV1() as kernel:
                        lane = _run_full129_lane(
                            preflight, v3, grid, d, cp, bridge, de441_custody,
                            retained_result, source_fd, kernel,
                        )
                v3_runtime = v3._critical_runtime_identity(
                    d, runtime_base=runtime_base, import_guard=guard,
                    engine_projection=lane["engine_projection"],
                    expected_mode=v3.RUNTIME_MODE_PRODUCTION,
                )
        runtime_post = _runtime_post(protocol, custody, v3_runtime, v3)
        if v3_preflight["prelane_binding_projection_sha256"] != runtime_post["v3_runtime_prelane_binding_projection_sha256"]:
            raise Full128SmokeError("smoke runtime differs from pre-lane binding")
        result, artifacts = _build_result_and_artifacts(
            protocol, custody, runtime_pre, runtime_post, lane, preflight,
        )
        snapshot = {
            "PROTOCOL.json": _canonical_json(protocol),
            **{name: payload for name, payload in artifacts},
        }
        validated = _validate_output_snapshot(snapshot)
        if _canonical_json(validated) != _canonical_json(result):
            raise Full128SmokeError("prepublication replay result differs")
    _publish(root, artifacts)
    return result


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, externally_preregistered: bool = False,
    prereg_commit: str = "", prereg_tag_object: str = "",
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root != DEFAULT_OUTPUT_ROOT:
        raise Full128SmokeError("preregistered smoke execution requires the exact default output root")
    validate_prepared(root)
    custody = _git_custody(
        externally_preregistered=externally_preregistered,
        prereg_commit=prereg_commit, prereg_tag_object=prereg_tag_object,
    )
    with _delegates() as (_preflight, v3):
        with v3._delegates() as (_grid, _h64, d):
            d._reject_preloaded_runtime_modules()
            d.validate_execution_prerequisites(d.DE441_ROOT)
            with tempfile.TemporaryDirectory(prefix="jx-tno-full129-smoke-parent-") as raw:
                directory = Path(raw)
                probe_path, _probe = d._run_import_only_preflight(d.DEFAULT_REBOUND_TARGET, directory)
                custody_path = directory / "GIT_CUSTODY.json"
                descriptor = os.open(
                    custody_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
                )
                try:
                    payload = _canonical_json(custody)
                    written = os.write(descriptor, payload)
                    if written != len(payload):
                        raise Full128SmokeError("Git custody transfer write was incomplete")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                completed = subprocess.run(
                    [
                        str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "execute-worker",
                        "--output-root", str(root), "--import-probe-receipt", str(probe_path),
                        "--git-custody", str(custody_path),
                    ],
                    cwd=PROJECT_ROOT, env=d._production_child_environment(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    check=False, timeout=7_200,
                )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
        raise Full128SmokeError(f"fresh full129 smoke worker failed: {detail}")
    return validate_output(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "prepare", "validate-prepared", "execute", "validate", "execute-worker"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--externally-preregistered", action="store_true")
    parser.add_argument("--prereg-commit", default="")
    parser.add_argument("--prereg-tag-object", default="")
    parser.add_argument("--import-probe-receipt", type=Path)
    parser.add_argument("--git-custody", type=Path)
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
        execute(
            args.output_root, externally_preregistered=args.externally_preregistered,
            prereg_commit=args.prereg_commit, prereg_tag_object=args.prereg_tag_object,
        )
    elif args.command == "validate":
        validate_output(args.output_root)
    elif args.command == "execute-worker":
        if args.import_probe_receipt is None or args.git_custody is None:
            raise Full128SmokeError("execute-worker requires import-probe and Git-custody paths")
        _execute_worker(args.output_root, args.import_probe_receipt, args.git_custody)
    else:  # pragma: no cover
        raise Full128SmokeError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
