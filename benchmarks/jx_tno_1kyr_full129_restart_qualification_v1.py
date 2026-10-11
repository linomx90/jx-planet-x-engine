#!/usr/bin/env python3
"""Bounded full-129 fresh-process restart qualification.

This experiment resumes the exact retained batch-zero full-129 V3 trajectory
at year 500 and advances it to year 1000 in a fresh process.  It is deliberately
narrow: the retained midpoint has 129 active lanes and no route or removal
history.  Passing this experiment does not qualify the engine's general disk
checkpoint/restart surface, routed or removed lanes, ten-million-year dynamics,
classification, or a scientific result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_full129_restart_qualification_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_1kyr_full129_restart_qualification_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_1kyr_full129_restart_qualification_v1"
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

V3_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v3.py"
V3_SOURCE_BYTES = 39_277
V3_SOURCE_SHA256 = "d560c4e66082c7d1e3bb5b69dcdd1671e8cf4f22ea22e145e7aaff2dc3880df9"
V3_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v3.py"
V3_TEST_BYTES = 18_460
V3_TEST_SHA256 = "8a464269569d12624ad262faac882c7b367604865bb5da323aec72c89a2e14a0"
V3_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v3"
V3_FILES: dict[str, tuple[int, str]] = {
    "CHECKPOINTS.bin": (135_072, "07bcfb91100fa9a5f9ae23fb225586a0b37502d8b12ec6658e6412ba0166781d"),
    "CONTINUATION.bin": (130_032, "0c13da9af19c55952424da7acecc82c982ff8c3ef381a7822b1f91b1abe96980"),
    "LIFECYCLE.bin": (2_709, "d5fe0dd289a74ae355cbc3e9d7fb7c86ce42fe5a8492683df82cac7331b5ebce"),
    "PROTOCOL.json": (23_245, "92b2ad9069de25795163154ba70a67c80a1a62f8feb07c4bde925b3e437edac3"),
    "RESULT.json": (24_919, "93ee3db258ac21a9a17c964da96688d561f16affaee830297b638912041ed66f"),
    "RUNTIME_POST.json": (15_310, "ab29fbf42b175f8f80b54e0ffd70cba67d17c36919e09b5267ea746fcd8c71ea"),
    "RUNTIME_PRE.json": (28_292, "c7004fff502b4e1a30e6c137e9f055d3c01b7c5cff165da3961752369eadfaf2"),
    "SEGMENTS.jsonl": (30_267, "c552dc4e360ddf93b155310f49ca1bcca73da1a893fa8e9213ddc6475dd64f28"),
}
V3_PROTOCOL_CONTENT_SHA256 = "964b04b6ecbbd37c3f59882275c672488c2aaf37c457bbdb5fb59b682182cb53"
V3_RESULT_CONTENT_SHA256 = "e51ac1578a867e7f2f9e707ef983114992ee9a04cbddecae21d09dd27627a2c8"
V3_PREREG_BRANCH = "research/tno-10my-reduced5-execution-v3-preregistered-20261011"
V3_PREREG_TAG = "tno-10my-reduced5-execution-v3-preregistered-20261011"
V3_PREREG_COMMIT = "cc5394e0b538a3427871678aa05cb7938c65f530"
V3_PREREG_TAG_OBJECT = "bfe5e4b2551ab013e879c64e274db449afa3e135"
V3_RESULT_TAG = "tno-10my-full128-1kyr-gpu-smoke-v3-result-20261011"
V3_RESULT_COMMIT = "001d02e4997162dcd6b3ab7e7ea9cb58ad2fe1a8"
V3_RESULT_TAG_OBJECT = "07065b9acebe9eb3ab34daef71a0ef8b6e4c6c78"

H64_SOURCE_PATH = PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v2" / "JX_SOURCE_NODES.bin"
H64_SOURCE_BYTES = 368_645_800
H64_SOURCE_SHA256 = "fd9b51e18aa6bb5fac095799955ecc6b3d6afeb2253b1000e2def41ccf376dbd"
H64_BATCH0_BYTE_OFFSET = 184_322_920
H64_BATCH0_BYTE_LENGTH = 15_360_240
H64_POSITION_BLOCK_BYTES = 7_680_120
H64_NODE_BYTES = 5 * 3 * 8
H64_NODE_COUNT = 64_001
H64_BATCH0_LANE_SHA256 = "d4c470d6698da111e47816771352b5ba4920ab07a28e739f69d3e2697df29a55"
H64_BATCH0_TILE_SHA256 = "0f584a502ae9d87dc11ef8546cf53481700c468f9e0543ba160dbe2d4b0934c4"

PREREGISTRATION_BRANCH = "research/tno-1kyr-full129-restart-v1-preregistered-20261011"
PREREGISTRATION_TAG = "tno-1kyr-full129-restart-v1-preregistered-20261011"
FUTURE_RESULT_TAG = "tno-1kyr-full129-restart-v1-result-20261011"

PROTOCOL_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.protocol.v1"
REQUEST_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.request.v1"
REQUEST_TRANSFER_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.request-transfer.v1"
SEGMENT_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.segment.v1"
RUNTIME_PRE_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.runtime-pre.v1"
RUNTIME_POST_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.runtime-post.v1"
GIT_CUSTODY_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.git-custody.v1"
RESULT_SCHEMA = "jx.tno.1kyr-full129-restart-qualification.result.v1"

PROTOCOL_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.protocol.v1\0"
REQUEST_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.request.v1\0"
REQUEST_TRANSFER_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.request-transfer.v1\0"
SEGMENT_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.segment.v1\0"
RUNTIME_PRE_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.runtime-pre.v1\0"
RUNTIME_POST_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.runtime-post.v1\0"
GIT_CUSTODY_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.git-custody.v1\0"
RESULT_DOMAIN = b"jx.tno.1kyr-full129-restart-qualification.result.v1\0"

LOCAL_THREAT_BOUNDARY = (
    "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"
)
BATCH_ORDER = 0
OBJECT_ID = "jpl-20090377"
SPK_ID = 20_090_377
TRACER_COUNT = 129
MASSIVE_COUNT = 5
BODY_COUNT = MASSIVE_COUNT + TRACER_COUNT
STEPS_PER_YEAR = 64
SEGMENT_YEARS = 50
SEGMENT_STEPS = SEGMENT_YEARS * STEPS_PER_YEAR
START_YEAR = 500
END_YEAR = 1_000
START_GLOBAL_SOURCE_STEP = START_YEAR * STEPS_PER_YEAR
END_GLOBAL_SOURCE_STEP = END_YEAR * STEPS_PER_YEAR
REFERENCE_START_RECORD = START_YEAR // SEGMENT_YEARS
REFERENCE_END_RECORD = END_YEAR // SEGMENT_YEARS
RESTART_SEGMENT_COUNT = (END_YEAR - START_YEAR) // SEGMENT_YEARS
RESTART_CHECKPOINT_YEARS = tuple(range(START_YEAR, END_YEAR + 1, SEGMENT_YEARS))
STATE_RECORD_BYTES = BODY_COUNT * 6 * 8
CONTINUATION_RECORD_BYTES = TRACER_COUNT * 6 * 8
LIFECYCLE_RECORD_BYTES = TRACER_COUNT
CHECKPOINT_SUFFIX_BYTES = len(RESTART_CHECKPOINT_YEARS) * STATE_RECORD_BYTES
CONTINUATION_SUFFIX_BYTES = len(RESTART_CHECKPOINT_YEARS) * CONTINUATION_RECORD_BYTES
LIFECYCLE_SUFFIX_BYTES = len(RESTART_CHECKPOINT_YEARS) * LIFECYCLE_RECORD_BYTES
REFERENCE_CHECKPOINT_SUFFIX_SHA256 = "be2b9335069b3cfe673fcbaebee63c380337201deb7f76073a8a7d578a30bebf"
REFERENCE_CONTINUATION_SUFFIX_SHA256 = "7dd417e4ad52278dabdeadc84eb5c59974553fe80979e9ad49c56c2b4869673d"
REFERENCE_LIFECYCLE_SUFFIX_SHA256 = "7a5a2b7d1ce40ed7c5612cd9c7823b0e3998fa2d0dfe602e017c65b409b3c9f1"
REFERENCE_MIDPOINT_CONTINUATION_SHA256 = "bc55fc9d82f2096595f26fab2f461f053c2e8c57bbc196bc3055e490ad007217"
REFERENCE_MIDPOINT_CHECKPOINT_SHA256 = "561b857d302e3b6ed444e49c23d2dae8680c244cb73ee5265a9ecff8071d59d9"
REFERENCE_LIFECYCLE_ROW_SHA256 = "2c80619d7e7c58257293cda3a878c13e5856f4e06f6f90601276f7b9179c9e07"
REFERENCE_FINAL_CONTINUATION_SHA256 = "f6fe23760b20994374503e152f8eec0ac705bf9a602dfe209f4a6e733c475544"
REFERENCE_FINAL_CHECKPOINT_SHA256 = "81fc4094ad58216895c30e672c1159ae55a74c99d2fa6551fb461c1fc48fe599"
REFERENCE_ENGINE_PROJECTION_SHA256 = "76a591f4a61ea5c0d23f3d4825277bbdd541bda79826444ad73acff5a5cb4474"
REFERENCE_SOURCE_START_SHA256 = "80d8c654f27ef53eb7081e877cea0f6ef1ea0e99f902764d950f6777fc94d113"

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_OUTPUT_ORDER = (
    "RESTART_REQUEST.json",
    "RESTART_SUFFIX.bin",
    "CHECKPOINT_SUFFIX.bin",
    "LIFECYCLE_SUFFIX.bin",
    "SEGMENTS.jsonl",
    "ROUTES.jsonl",
    "REMOVALS.jsonl",
    "RUNTIME_PRE.json",
    "RUNTIME_POST.json",
    "RESULT.json",
)
EXPECTED_OUTPUT_ROSTER = frozenset({"PROTOCOL.json", *EXPECTED_OUTPUT_ORDER})
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})

CLAIMS_PREPARED = {
    "gpu_executed": False,
    "restart_execution_completed": False,
    "batch0_full129_all_active_restart_equivalence_qualified": False,
    "general_disk_checkpoint_restart_qualified": False,
    "routed_or_removed_restart_qualified": False,
    "ten_myr_execution_authorized": False,
    "classification_generated": False,
    "science_result": False,
}


class Full129RestartError(RuntimeError):
    """The bounded restart contract was violated."""


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(domain) is not bytes or not domain.endswith(b"\0") or type(value) is not dict:
        raise Full129RestartError("domain-hash input differs")
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(unsigned))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: object) -> bool:
    return (
        type(value) is str and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise Full129RestartError(f"{label} is not an exact integer")
    return value


def _strict_json(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise Full129RestartError("JSON payload must be exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Full129RestartError("JSON payload is invalid") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise Full129RestartError("JSON payload is not a canonical object")
    return value


def _jsonl(payload: bytes, label: str) -> tuple[dict[str, Any], ...]:
    if type(payload) is not bytes:
        raise Full129RestartError(f"{label} must be bytes")
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise Full129RestartError(f"{label} is not newline terminated")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        rows.append(_strict_json(line))
    return tuple(rows)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = _absolute_lexical(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:-1]:
        current = current / part
        metadata = current.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise Full129RestartError(f"{label} has a non-directory or symlink ancestor")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_uid,
    )


def _pread_all(descriptor: int, size: int, label: str, *, offset: int = 0) -> bytes:
    _require_int(descriptor, f"{label} descriptor", minimum=0)
    _require_int(size, f"{label} size", minimum=0)
    _require_int(offset, f"{label} offset", minimum=0)
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        chunk = os.pread(descriptor, min(remaining, 8 * 1024 * 1024), position)
        if not chunk:
            raise Full129RestartError(f"{label} ended early")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextmanager
def _held_regular_file(
    path: Path, label: str, *, expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[int]:
    absolute = _absolute_lexical(path)
    _require_no_symlink_ancestors(absolute, label)
    lexical = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1:
        raise Full129RestartError(f"{label} is not a single-link regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(lexical):
            raise Full129RestartError(f"{label} changed before open")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise Full129RestartError(f"{label} byte length differs")
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            position = 0
            while position < before.st_size:
                chunk = os.pread(descriptor, min(8 * 1024 * 1024, before.st_size - position), position)
                if not chunk:
                    raise Full129RestartError(f"{label} ended during hashing")
                digest.update(chunk)
                position += len(chunk)
            if digest.hexdigest() != expected_sha256:
                raise Full129RestartError(f"{label} SHA-256 differs")
        yield descriptor
        after = os.fstat(descriptor)
        final = absolute.stat(follow_symlinks=False)
        if _stat_identity(after) != _stat_identity(before) or _stat_identity(final) != _stat_identity(before):
            raise Full129RestartError(f"{label} changed during use")
    finally:
        os.close(descriptor)


@contextmanager
def _held_directory(
    path: Path, label: str, *, expected_roster: frozenset[str] | set[str] | None = None,
) -> Iterator[int]:
    absolute = _absolute_lexical(path)
    _require_no_symlink_ancestors(absolute, label)
    lexical = absolute.stat(follow_symlinks=False)
    if not stat.S_ISDIR(lexical.st_mode):
        raise Full129RestartError(f"{label} is not a lexical directory")
    descriptor = os.open(
        absolute, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_mode) != (lexical.st_dev, lexical.st_ino, lexical.st_mode):
            raise Full129RestartError(f"{label} changed before open")
        if expected_roster is not None and set(os.listdir(descriptor)) != set(expected_roster):
            raise Full129RestartError(f"{label} roster differs")
        yield descriptor
        after = os.fstat(descriptor)
        final = absolute.stat(follow_symlinks=False)
        expected_identity = (before.st_dev, before.st_ino, before.st_mode)
        if (
            (after.st_dev, after.st_ino, after.st_mode) != expected_identity
            or (final.st_dev, final.st_ino, final.st_mode) != expected_identity
        ):
            raise Full129RestartError(f"{label} changed during use")
    finally:
        os.close(descriptor)


def _read_regular_at(
    directory_fd: int, name: str, label: str, expected_bytes: int | None = None,
) -> bytes:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise Full129RestartError(f"{label} name differs")
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise Full129RestartError(f"{label} is not a single-link regular file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise Full129RestartError(f"{label} byte length differs")
        payload = _pread_all(descriptor, before.st_size, label)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(before):
            raise Full129RestartError(f"{label} changed during read")
        return payload
    finally:
        os.close(descriptor)


def _read_regular(path: Path, label: str, expected_bytes: int | None = None) -> bytes:
    with _held_regular_file(path, label, expected_bytes=expected_bytes) as descriptor:
        size = os.fstat(descriptor).st_size
        return _pread_all(descriptor, size, label)


def _write_create_only_at(directory_fd: int, name: str, payload: bytes) -> None:
    if type(name) is not str or not name or "/" in name or name in (".", "..") or type(payload) is not bytes:
        raise Full129RestartError("create-only output differs")
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
                raise Full129RestartError("create-only write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_source(path: Path, size: int, digest: str, name: str) -> types.ModuleType:
    payload = _read_regular(path, name, size)
    if _sha256(payload) != digest:
        raise Full129RestartError(f"source binding differs: {path}")
    if name in sys.modules:
        raise Full129RestartError("private source-only module name is occupied")
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
def _v3_delegate() -> Iterator[types.ModuleType]:
    name = f"_jx_tno_full129_restart_v1_v3_{os.getpid()}_{id(object())}"
    module = _load_source(V3_SOURCE_PATH, V3_SOURCE_BYTES, V3_SOURCE_SHA256, name)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


def _implementation_record(path: Path, label: str) -> dict[str, object]:
    payload = _read_regular(path, label)
    if not payload:
        raise Full129RestartError(f"{label} is empty")
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _git_text(arguments: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise Full129RestartError(f"Git {label} failed: {detail}")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise Full129RestartError(f"Git {label} output is not UTF-8") from exc


def _git_bytes(arguments: Sequence[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise Full129RestartError(f"Git {label} failed: {detail}")
    return completed.stdout


def _verify_annotated_tag(name: str, object_id: str, commit: str, label: str) -> dict[str, str]:
    if _git_text(("cat-file", "-t", object_id), f"{label} type") != "tag":
        raise Full129RestartError(f"{label} is not an annotated tag")
    if _git_text(("rev-parse", f"refs/tags/{name}"), f"{label} object") != object_id:
        raise Full129RestartError(f"{label} object differs")
    if _git_text(("rev-parse", f"{name}^{{commit}}"), f"{label} commit") != commit:
        raise Full129RestartError(f"{label} peeled commit differs")
    return {"annotated_tag": name, "tag_object": object_id, "peeled_commit": commit}


def _v3_git_lineage() -> dict[str, Any]:
    prereg = _verify_annotated_tag(V3_PREREG_TAG, V3_PREREG_TAG_OBJECT, V3_PREREG_COMMIT, "V3 preregistration tag")
    result = _verify_annotated_tag(V3_RESULT_TAG, V3_RESULT_TAG_OBJECT, V3_RESULT_COMMIT, "V3 result tag")
    tracked = {
        str(V3_SOURCE_PATH.relative_to(PROJECT_ROOT)): (V3_SOURCE_BYTES, V3_SOURCE_SHA256),
        str(V3_TEST_PATH.relative_to(PROJECT_ROOT)): (V3_TEST_BYTES, V3_TEST_SHA256),
        **{
            str((V3_ROOT / name).relative_to(PROJECT_ROOT)): record
            for name, record in V3_FILES.items()
        },
    }
    records: list[dict[str, object]] = []
    for relative, (size, digest) in sorted(tracked.items()):
        committed = _git_bytes(("show", f"{V3_RESULT_COMMIT}:{relative}"), f"V3 result {relative}")
        if len(committed) != size or _sha256(committed) != digest:
            raise Full129RestartError(f"V3 result commit binding differs: {relative}")
        records.append({"relative_path": relative, "bytes": size, "sha256": digest})
    return {
        "preregistration": {"branch": V3_PREREG_BRANCH, **prereg},
        "result": result,
        "tracked_result_files": records,
        "network_remote_verification_performed": False,
        "point_observation_only": True,
    }


@contextmanager
def _held_v3_snapshot() -> Iterator[dict[str, bytes]]:
    with _held_directory(V3_ROOT, "V3 retained root", expected_roster=set(V3_FILES)) as root_fd:
        snapshot: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int, int, int, int]] = {}
        for name, (size, digest) in sorted(V3_FILES.items()):
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            identities[name] = _stat_identity(metadata)
            payload = _read_regular_at(root_fd, name, f"V3 {name}", size)
            if _sha256(payload) != digest:
                raise Full129RestartError(f"V3 retained file differs: {name}")
            snapshot[name] = payload
        protocol = _strict_json(snapshot["PROTOCOL.json"])
        result = _strict_json(snapshot["RESULT.json"])
        segments = _jsonl(snapshot["SEGMENTS.jsonl"], "V3 segments")
        if (
            protocol.get("content_sha256") != V3_PROTOCOL_CONTENT_SHA256
            or result.get("content_sha256") != V3_RESULT_CONTENT_SHA256
            or result.get("status") != "FULL129_BATCH0_1KYR_GPU_SMOKE_PASS_TEN_MYR_NOT_AUTHORIZED"
            or result.get("batch_order") != BATCH_ORDER or type(result.get("batch_order")) is not int
            or result.get("object_id") != OBJECT_ID
            or result.get("tracer_count") != TRACER_COUNT or type(result.get("tracer_count")) is not int
            or result.get("route_record_count") != 0 or type(result.get("route_record_count")) is not int
            or result.get("removal_record_count") != 0 or type(result.get("removal_record_count")) is not int
            or len(segments) != 20
            or any(
                type(row) is not dict
                or row.get("route_record_count") != 0 or type(row.get("route_record_count")) is not int
                or row.get("removal_record_count") != 0 or type(row.get("removal_record_count")) is not int
                for row in segments
            )
            or snapshot["LIFECYCLE.bin"] != bytes(21 * TRACER_COUNT)
        ):
            raise Full129RestartError("V3 retained scientific boundary differs")
        git_record = result.get("git_preregistration")
        if (
            type(git_record) is not dict
            or git_record.get("branch") != V3_PREREG_BRANCH
            or git_record.get("commit") != V3_PREREG_COMMIT
            or git_record.get("tag_object") != V3_PREREG_TAG_OBJECT
            or git_record.get("annotated_tag") != V3_PREREG_TAG
        ):
            raise Full129RestartError("V3 embedded Git preregistration differs")
        yield snapshot
        if set(os.listdir(root_fd)) != set(V3_FILES):
            raise Full129RestartError("V3 retained roster changed during use")
        for name, identity in identities.items():
            if _stat_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                raise Full129RestartError(f"V3 retained file changed during use: {name}")


def _reference_suffix(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    if type(snapshot) is not dict or set(snapshot) != set(V3_FILES):
        raise Full129RestartError("V3 snapshot shape differs")
    continuation_offset = REFERENCE_START_RECORD * CONTINUATION_RECORD_BYTES
    checkpoint_offset = REFERENCE_START_RECORD * STATE_RECORD_BYTES
    lifecycle_offset = REFERENCE_START_RECORD * LIFECYCLE_RECORD_BYTES
    continuation = snapshot["CONTINUATION.bin"][continuation_offset:]
    checkpoints = snapshot["CHECKPOINTS.bin"][checkpoint_offset:]
    lifecycle = snapshot["LIFECYCLE.bin"][lifecycle_offset:]
    segments = _jsonl(snapshot["SEGMENTS.jsonl"], "V3 segments")[REFERENCE_START_RECORD:]
    if (
        len(continuation) != CONTINUATION_SUFFIX_BYTES
        or _sha256(continuation) != REFERENCE_CONTINUATION_SUFFIX_SHA256
        or len(checkpoints) != CHECKPOINT_SUFFIX_BYTES
        or _sha256(checkpoints) != REFERENCE_CHECKPOINT_SUFFIX_SHA256
        or len(lifecycle) != LIFECYCLE_SUFFIX_BYTES
        or _sha256(lifecycle) != REFERENCE_LIFECYCLE_SUFFIX_SHA256
        or lifecycle != bytes(LIFECYCLE_SUFFIX_BYTES)
        or len(segments) != RESTART_SEGMENT_COUNT
        or _sha256(continuation[:CONTINUATION_RECORD_BYTES]) != REFERENCE_MIDPOINT_CONTINUATION_SHA256
        or _sha256(checkpoints[:STATE_RECORD_BYTES]) != REFERENCE_MIDPOINT_CHECKPOINT_SHA256
        or _sha256(continuation[-CONTINUATION_RECORD_BYTES:]) != REFERENCE_FINAL_CONTINUATION_SHA256
        or _sha256(checkpoints[-STATE_RECORD_BYTES:]) != REFERENCE_FINAL_CHECKPOINT_SHA256
    ):
        raise Full129RestartError("V3 retained restart suffix differs")
    return {
        "continuation": continuation,
        "checkpoints": checkpoints,
        "lifecycle": lifecycle,
        "segments": segments,
        "result": _strict_json(snapshot["RESULT.json"]),
        "runtime_pre": _strict_json(snapshot["RUNTIME_PRE.json"]),
        "runtime_post": _strict_json(snapshot["RUNTIME_POST.json"]),
    }


def _bound_predecessor() -> dict[str, Any]:
    source = _implementation_record(V3_SOURCE_PATH, "V3 smoke source")
    test = _implementation_record(V3_TEST_PATH, "V3 smoke test")
    if source["bytes"] != V3_SOURCE_BYTES or source["sha256"] != V3_SOURCE_SHA256:
        raise Full129RestartError("V3 source binding differs")
    if test["bytes"] != V3_TEST_BYTES or test["sha256"] != V3_TEST_SHA256:
        raise Full129RestartError("V3 test binding differs")
    with _held_v3_snapshot() as snapshot:
        reference = _reference_suffix(snapshot)
    return {
        "source": source,
        "test": test,
        "retained_root": {
            "relative_path": str(V3_ROOT.relative_to(PROJECT_ROOT)),
            "exact_roster": [
                {"name": name, "bytes": size, "sha256": digest}
                for name, (size, digest) in sorted(V3_FILES.items())
            ],
            "protocol_content_sha256": V3_PROTOCOL_CONTENT_SHA256,
            "result_content_sha256": V3_RESULT_CONTENT_SHA256,
            "checkpoint_suffix_sha256": _sha256(reference["checkpoints"]),
            "continuation_suffix_sha256": _sha256(reference["continuation"]),
            "lifecycle_suffix_sha256": _sha256(reference["lifecycle"]),
        },
        "git_lineage": _v3_git_lineage(),
    }


def build_protocol() -> dict[str, Any]:
    implementation = _implementation_record(SCRIPT_PATH, "restart qualification source")
    test = _implementation_record(TEST_PATH, "restart qualification test")
    predecessor = _bound_predecessor()
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARED_BATCH0_FULL129_YEAR500_TO1000_RESTART_NOT_EXECUTED",
        "objective": "QUALIFY_ONE_ALL_ACTIVE_FULL129_FRESH_PROCESS_RESTART_SUFFIX_ONLY",
        "implementation": {
            **implementation,
            "test": test,
            "source_only_predecessor_loading": True,
            "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        },
        "predecessor_v3": predecessor,
        "source_evidence": {
            "relative_path": str(H64_SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "bytes": H64_SOURCE_BYTES,
            "sha256": H64_SOURCE_SHA256,
            "batch0_byte_offset": H64_BATCH0_BYTE_OFFSET,
            "batch0_byte_length": H64_BATCH0_BYTE_LENGTH,
            "position_block_bytes": H64_POSITION_BLOCK_BYTES,
            "node_count": H64_NODE_COUNT,
            "node_bytes": H64_NODE_BYTES,
            "lane_sha256": H64_BATCH0_LANE_SHA256,
            "full_tile_sha256": H64_BATCH0_TILE_SHA256,
            "held_fd_full_hash_and_semantic_replay_required": True,
            "copied_into_successor_output": False,
        },
        "preregistration": {
            "required_before_execute": True,
            "branch": PREREGISTRATION_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "future_result_tag": FUTURE_RESULT_TAG,
            "commit_and_tag_object_are_runtime_supplied_and_locally_reconstructed": True,
            "exact_clean_tracked_paths": [
                str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
                str(TEST_PATH.relative_to(PROJECT_ROOT)),
                str((DEFAULT_OUTPUT_ROOT / "PROTOCOL.json").relative_to(PROJECT_ROOT)),
            ],
            "network_verification_by_this_implementation": False,
        },
        "restart": {
            "batch_order": BATCH_ORDER,
            "object_id": OBJECT_ID,
            "spk_id": SPK_ID,
            "ephemeris": "DE441",
            "model": "REDUCED_FIVE_SUN_PLUS_INNER_GM_JUPITER_SATURN_URANUS_NEPTUNE",
            "time_scale": "TDB",
            "frame": "ICRF_J2000",
            "axes": "J2000_EQUATORIAL",
            "coordinate_system": "PRIMARY_RELATIVE_NONROTATING",
            "length_unit": "AU",
            "time_unit": "julian_year",
            "step_julian_years_hex": (1.0 / STEPS_PER_YEAR).hex(),
            "steps_per_julian_year": STEPS_PER_YEAR,
            "tracer_count": TRACER_COUNT,
            "tracer_indices_exact": list(range(TRACER_COUNT)),
            "tracer_order": [
                {"sample_kind": "NOMINAL", "clone_index": None},
                *({"sample_kind": "COVARIANCE_CLONE", "clone_index": index} for index in range(128)),
            ],
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "start_global_source_step": START_GLOBAL_SOURCE_STEP,
            "end_global_source_step": END_GLOBAL_SOURCE_STEP,
            "segment_years": SEGMENT_YEARS,
            "segment_steps": SEGMENT_STEPS,
            "segment_count": RESTART_SEGMENT_COUNT,
            "checkpoint_years": list(RESTART_CHECKPOINT_YEARS),
            "midpoint_authority": "V3_CONTINUATION_RECORD_10_PRIMARY_RELATIVE_BINARY64",
            "midpoint_continuation_byte_offset": REFERENCE_START_RECORD * CONTINUATION_RECORD_BYTES,
            "midpoint_continuation_bytes": CONTINUATION_RECORD_BYTES,
            "midpoint_continuation_sha256": REFERENCE_MIDPOINT_CONTINUATION_SHA256,
            "midpoint_checkpoint_byte_offset": REFERENCE_START_RECORD * STATE_RECORD_BYTES,
            "midpoint_checkpoint_bytes": STATE_RECORD_BYTES,
            "midpoint_checkpoint_sha256": REFERENCE_MIDPOINT_CHECKPOINT_SHA256,
            "midpoint_lifecycle_byte_offset": REFERENCE_START_RECORD * LIFECYCLE_RECORD_BYTES,
            "midpoint_lifecycle_bytes": LIFECYCLE_RECORD_BYTES,
            "midpoint_lifecycle_sha256": REFERENCE_LIFECYCLE_ROW_SHA256,
            "midpoint_lifecycle_exact_all_active": True,
            "midpoint_route_history_exact_empty": True,
            "midpoint_removal_history_exact_empty": True,
            "snapshot_id_must_equal_monolithic": f"tno-full129-smoke-v1-DE441-{OBJECT_ID}",
        },
        "acceptance": {
            "all_11_continuation_records_bitwise_equal_to_v3": True,
            "all_11_checkpoint_records_bitwise_equal_to_v3": True,
            "all_11_lifecycle_records_bitwise_equal_to_v3": True,
            "all_10_route_and_removal_sequences_exact_empty": True,
            "all_10_source_segments_and_11_endpoints_match_held_h64_bytes": True,
            "all_10_source_tile_hashes_match_v3": True,
            "held_full_lane_all_node_energy_and_angular_momentum_invariants_rederived": True,
            "held_full_lane_invariants_exactly_match_v3": True,
            "all_10_minimum_distance_hex_and_maximum_kepler_iterations_match_v3": True,
            "all_10_engine_runtime_projection_hashes_match_v3": True,
            "source_build_identity_and_path_only_relocation_receipt_match_v3": True,
            "runtime_pre_post_critical_identity_stable": True,
            "minimum_source_distance_strictly_greater_than_AU_hex": float(1.0).hex(),
            "final_global_source_step": END_GLOBAL_SOURCE_STEP,
            "no_hidden_multistep_or_router_state_claim": "ONLY_THE_EXACT_ALL_ACTIVE_ZERO_HISTORY_BRANCH_IS_EXERCISED",
        },
        "outputs": {
            "prepared_roster": sorted(PREPARED_ROSTER),
            "published_roster": sorted(EXPECTED_OUTPUT_ROSTER),
            "publication_order": ["PROTOCOL.json", *EXPECTED_OUTPUT_ORDER],
            "result_published_last": True,
            "create_only": True,
            "precommit_failure_may_leave_incomplete_result_absent_root_requiring_manual_cleanup": True,
            "postcommit_cleanup_failure_may_leave_complete_root_requiring_validation": True,
            "continuation_suffix_bytes": CONTINUATION_SUFFIX_BYTES,
            "checkpoint_suffix_bytes": CHECKPOINT_SUFFIX_BYTES,
            "lifecycle_suffix_bytes": LIFECYCLE_SUFFIX_BYTES,
            "routes_bytes": 0,
            "removals_bytes": 0,
        },
        "claim_ceiling": {
            "qualified_if_passed": "BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000_FRESH_PROCESS_SUFFIX_EQUIVALENCE_ONLY",
            "general_disk_checkpoint_restart_qualified": False,
            "routed_or_removed_restart_qualified": False,
            "other_batches_restart_qualified": False,
            "ten_myr_execution_authorized": False,
            "classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "remaining_engine_gap": {
            "public_engine_claim_disk_checkpoint_restart_qualified": False,
            "public_serializer_or_deserializer_exists": False,
            "v3_continuation_retains_only_primary_relative_tracer_position_velocity": True,
            "runner_reconstructs_metadata_indices_and_all_active_lifecycle_from_sealed_inputs": True,
            "any_nonactive_lifecycle_or_nonempty_removal_history_must_fail_closed": True,
        },
        "claims": dict(CLAIMS_PREPARED),
        "content_sha256": "",
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise Full129RestartError("protocol must be an exact object")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise Full129RestartError("protocol differs from exact reconstruction")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise Full129RestartError("protocol content identity differs")
    return str(value["content_sha256"])


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _absolute_lexical(output_root)
    if root.name in ("", ".", ".."):
        raise Full129RestartError("output root name differs")
    protocol = build_protocol()
    with _held_directory(root.parent, "output parent") as parent_fd:
        try:
            os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise Full129RestartError("output root must be absent for create-only preparation")
        os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        created = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        root_fd = os.open(
            root.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(root_fd)
            if (created.st_dev, created.st_ino) != (opened.st_dev, opened.st_ino):
                raise Full129RestartError("created output root identity differs")
            _write_create_only_at(root_fd, "PROTOCOL.json", _canonical_json(protocol))
            os.fsync(root_fd)
            if set(os.listdir(root_fd)) != PREPARED_ROSTER:
                raise Full129RestartError("prepared root roster differs")
        finally:
            os.close(root_fd)
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _absolute_lexical(output_root)
    with _held_directory(root, "prepared restart root", expected_roster=PREPARED_ROSTER) as root_fd:
        payload = _read_regular_at(root_fd, "PROTOCOL.json", "prepared protocol")
    value = _strict_json(payload)
    validate_protocol(value)
    return value


def _git_custody(
    *, externally_preregistered: bool, prereg_commit: str, prereg_tag_object: str,
) -> dict[str, Any]:
    if externally_preregistered is not True:
        raise Full129RestartError("restart execution requires external preregistration")
    if not (type(prereg_commit) is str and len(prereg_commit) == 40 and all(c in "0123456789abcdef" for c in prereg_commit)):
        raise Full129RestartError("preregistration commit identity differs")
    if not (type(prereg_tag_object) is str and len(prereg_tag_object) == 40 and all(c in "0123456789abcdef" for c in prereg_tag_object)):
        raise Full129RestartError("preregistration tag-object identity differs")
    branch = _git_text(("symbolic-ref", "--quiet", "--short", "HEAD"), "branch")
    head = _git_text(("rev-parse", "HEAD"), "HEAD")
    tag_object = _git_text(("rev-parse", f"refs/tags/{PREREGISTRATION_TAG}"), "tag object")
    tag_commit = _git_text(("rev-parse", f"{PREREGISTRATION_TAG}^{{commit}}"), "peeled tag")
    tag_type = _git_text(("cat-file", "-t", tag_object), "tag type")
    if (
        branch != PREREGISTRATION_BRANCH or head != prereg_commit
        or tag_object != prereg_tag_object or tag_commit != head or tag_type != "tag"
    ):
        raise Full129RestartError("local preregistration branch/commit/tag custody differs")
    protocol = build_protocol()
    paths = tuple(protocol["preregistration"]["exact_clean_tracked_paths"])
    status = _git_text(("status", "--porcelain=v1", "--untracked-files=all", "--", *paths), "pinned-path status")
    if status:
        raise Full129RestartError("preregistered source/test/protocol paths are not clean")
    records: list[dict[str, Any]] = []
    for relative in paths:
        live = _read_regular(PROJECT_ROOT / relative, f"preregistered {relative}")
        committed = _git_bytes(("show", f"{head}:{relative}"), f"committed {relative}")
        if live != committed:
            raise Full129RestartError(f"preregistered path differs from commit: {relative}")
        records.append({"relative_path": relative, "bytes": len(live), "sha256": _sha256(live)})
    expected = (
        (protocol["implementation"]["bytes"], protocol["implementation"]["sha256"]),
        (protocol["implementation"]["test"]["bytes"], protocol["implementation"]["test"]["sha256"]),
        (len(_canonical_json(protocol)), _sha256(_canonical_json(protocol))),
    )
    if tuple((row["bytes"], row["sha256"]) for row in records) != expected:
        raise Full129RestartError("Git tracked-file receipt differs from protocol")
    value: dict[str, Any] = {
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
    value["content_sha256"] = _domain_hash(GIT_CUSTODY_DOMAIN, value)
    return value


def _validate_git_custody(
    value: object, protocol: Mapping[str, Any], *, expected_commit: str | None = None,
    expected_tag_object: str | None = None,
) -> dict[str, Any]:
    keys = {
        "schema", "externally_preregistered", "branch", "annotated_tag", "commit",
        "tag_object", "tag_peeled_commit", "tag_object_type", "clean_tracked_paths",
        "network_remote_verification_performed", "point_observation_only", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full129RestartError("Git custody receipt shape differs")
    records = value.get("clean_tracked_paths")
    expected_paths = protocol["preregistration"]["exact_clean_tracked_paths"]
    if (
        value.get("schema") != GIT_CUSTODY_SCHEMA
        or value.get("externally_preregistered") is not True
        or value.get("branch") != PREREGISTRATION_BRANCH
        or value.get("annotated_tag") != PREREGISTRATION_TAG
        or not _is_git_oid(value.get("commit"))
        or not _is_git_oid(value.get("tag_object"))
        or value.get("tag_peeled_commit") != value.get("commit")
        or value.get("tag_object_type") != "tag"
        or value.get("network_remote_verification_performed") is not False
        or value.get("point_observation_only") is not True
        or type(records) is not list or len(records) != 3
        or [row.get("relative_path") if type(row) is dict else None for row in records] != expected_paths
        or value.get("content_sha256") != _domain_hash(GIT_CUSTODY_DOMAIN, value)
    ):
        raise Full129RestartError("Git custody receipt differs")
    if expected_commit is not None and value["commit"] != expected_commit:
        raise Full129RestartError("Git custody commit differs")
    if expected_tag_object is not None and value["tag_object"] != expected_tag_object:
        raise Full129RestartError("Git custody tag object differs")
    expected_files = (
        (protocol["implementation"]["bytes"], protocol["implementation"]["sha256"]),
        (protocol["implementation"]["test"]["bytes"], protocol["implementation"]["test"]["sha256"]),
        (len(_canonical_json(protocol)), _sha256(_canonical_json(protocol))),
    )
    for row, (size, digest) in zip(records, expected_files, strict=True):
        if (
            type(row) is not dict or set(row) != {"relative_path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int or row["bytes"] != size
            or row.get("sha256") != digest
        ):
            raise Full129RestartError("Git tracked-file receipt differs")
    return value


def _request_transfer(protocol: Mapping[str, Any], *, parent_pid: int, challenge_hex: str) -> dict[str, Any]:
    _require_int(parent_pid, "parent PID", minimum=1)
    if type(challenge_hex) is not str or len(challenge_hex) != 64 or any(c not in "0123456789abcdef" for c in challenge_hex):
        raise Full129RestartError("parent challenge differs")
    value: dict[str, Any] = {
        "schema": REQUEST_TRANSFER_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "parent_pid": parent_pid,
        "parent_challenge_hex": challenge_hex,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "midpoint_continuation_sha256": REFERENCE_MIDPOINT_CONTINUATION_SHA256,
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(REQUEST_TRANSFER_DOMAIN, value)
    return value


def _validate_request_transfer(value: object, protocol: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "parent_pid",
        "parent_challenge_hex", "start_year", "end_year",
        "midpoint_continuation_sha256", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full129RestartError("restart request transfer shape differs")
    if (
        value.get("schema") != REQUEST_TRANSFER_SCHEMA
        or value.get("protocol_content_sha256") != protocol["content_sha256"]
        or value.get("protocol_file_sha256") != _sha256(_canonical_json(protocol))
        or type(value.get("parent_pid")) is not int or value["parent_pid"] <= 0
        or type(value.get("parent_challenge_hex")) is not str
        or len(value["parent_challenge_hex"]) != 64
        or any(c not in "0123456789abcdef" for c in value["parent_challenge_hex"])
        or value.get("start_year") != START_YEAR or type(value.get("start_year")) is not int
        or value.get("end_year") != END_YEAR or type(value.get("end_year")) is not int
        or value.get("midpoint_continuation_sha256") != REFERENCE_MIDPOINT_CONTINUATION_SHA256
        or value.get("content_sha256") != _domain_hash(REQUEST_TRANSFER_DOMAIN, value)
    ):
        raise Full129RestartError("restart request transfer differs")
    return value


def _published_request(transfer: Mapping[str, Any]) -> dict[str, Any]:
    child_pid = os.getpid()
    if type(child_pid) is not int or child_pid <= 0 or child_pid == transfer["parent_pid"]:
        raise Full129RestartError("restart worker is not a distinct process")
    value: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "protocol_content_sha256": transfer["protocol_content_sha256"],
        "protocol_file_sha256": transfer["protocol_file_sha256"],
        "parent_pid": transfer["parent_pid"],
        "child_pid": child_pid,
        "parent_challenge_hex": transfer["parent_challenge_hex"],
        "launcher_relative_path": str(PROJECT_VENV_PYTHON.relative_to(PROJECT_ROOT)),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "start_global_source_step": START_GLOBAL_SOURCE_STEP,
        "end_global_source_step": END_GLOBAL_SOURCE_STEP,
        "tracer_count": TRACER_COUNT,
        "tracer_indices": list(range(TRACER_COUNT)),
        "midpoint_continuation_sha256": REFERENCE_MIDPOINT_CONTINUATION_SHA256,
        "claim": "FRESH_CHILD_REQUEST_BINDING_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(REQUEST_DOMAIN, value)
    return value


def _validate_published_request(value: object, protocol: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "parent_pid",
        "child_pid", "parent_challenge_hex", "launcher_relative_path", "start_year",
        "end_year", "start_global_source_step", "end_global_source_step", "tracer_count",
        "tracer_indices", "midpoint_continuation_sha256", "claim", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full129RestartError("published restart request shape differs")
    integer_keys = (
        "parent_pid", "child_pid", "start_year", "end_year", "start_global_source_step",
        "end_global_source_step", "tracer_count",
    )
    indices = value.get("tracer_indices")
    if (
        any(type(value.get(key)) is not int for key in integer_keys)
        or type(indices) is not list or any(type(item) is not int for item in indices)
        or indices != list(range(TRACER_COUNT))
        or value.get("schema") != REQUEST_SCHEMA
        or value.get("protocol_content_sha256") != protocol["content_sha256"]
        or value.get("protocol_file_sha256") != _sha256(_canonical_json(protocol))
        or value["parent_pid"] <= 0 or value["child_pid"] <= 0 or value["parent_pid"] == value["child_pid"]
        or type(value.get("parent_challenge_hex")) is not str or len(value["parent_challenge_hex"]) != 64
        or any(c not in "0123456789abcdef" for c in value["parent_challenge_hex"])
        or value.get("launcher_relative_path") != str(PROJECT_VENV_PYTHON.relative_to(PROJECT_ROOT))
        or value.get("start_year") != START_YEAR or value.get("end_year") != END_YEAR
        or value.get("start_global_source_step") != START_GLOBAL_SOURCE_STEP
        or value.get("end_global_source_step") != END_GLOBAL_SOURCE_STEP
        or value.get("tracer_count") != TRACER_COUNT
        or value.get("midpoint_continuation_sha256") != REFERENCE_MIDPOINT_CONTINUATION_SHA256
        or value.get("claim") != "FRESH_CHILD_REQUEST_BINDING_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION"
        or value.get("content_sha256") != _domain_hash(REQUEST_DOMAIN, value)
    ):
        raise Full129RestartError("published restart request differs")
    return value


def _source_node_payload(descriptor: int, node: int) -> bytes:
    _require_int(node, "source node", minimum=0)
    if node >= H64_NODE_COUNT:
        raise Full129RestartError("source node exceeds retained lane")
    position = _pread_all(
        descriptor, H64_NODE_BYTES, "source node positions",
        offset=H64_BATCH0_BYTE_OFFSET + node * H64_NODE_BYTES,
    )
    velocity = _pread_all(
        descriptor, H64_NODE_BYTES, "source node velocities",
        offset=H64_BATCH0_BYTE_OFFSET + H64_POSITION_BLOCK_BYTES + node * H64_NODE_BYTES,
    )
    return position + velocity


def _source_segment_payload(descriptor: int, reference_segment_index: int) -> bytes:
    _require_int(reference_segment_index, "reference segment index", minimum=0)
    if not REFERENCE_START_RECORD <= reference_segment_index < REFERENCE_END_RECORD:
        raise Full129RestartError("source segment is outside restart suffix")
    start = reference_segment_index * SEGMENT_STEPS
    nodes = SEGMENT_STEPS + 1
    block_bytes = nodes * H64_NODE_BYTES
    position = _pread_all(
        descriptor, block_bytes, "source segment positions",
        offset=H64_BATCH0_BYTE_OFFSET + start * H64_NODE_BYTES,
    )
    velocity = _pread_all(
        descriptor, block_bytes, "source segment velocities",
        offset=H64_BATCH0_BYTE_OFFSET + H64_POSITION_BLOCK_BYTES + start * H64_NODE_BYTES,
    )
    return position + velocity


def _unpack_continuation(payload: bytes) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    if type(payload) is not bytes or len(payload) != CONTINUATION_RECORD_BYTES:
        raise Full129RestartError("continuation record geometry differs")
    values = struct.unpack(f"<{TRACER_COUNT * 6}d", payload)
    if any(type(item) is not float or not math.isfinite(item) for item in values):
        raise Full129RestartError("continuation record contains non-finite data")
    split = TRACER_COUNT * 3
    positions = tuple(tuple(values[index * 3:index * 3 + 3]) for index in range(TRACER_COUNT))
    velocities = tuple(tuple(values[split + index * 3:split + index * 3 + 3]) for index in range(TRACER_COUNT))
    return positions, velocities  # type: ignore[return-value]


def _pack_continuation(
    positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]],
) -> bytes:
    if type(positions) not in (tuple, list) or type(velocities) not in (tuple, list):
        raise Full129RestartError("continuation arrays differ")
    if len(positions) != TRACER_COUNT or len(velocities) != TRACER_COUNT:
        raise Full129RestartError("continuation roster differs")
    values: list[float] = []
    for rows in (positions, velocities):
        for row in rows:
            if type(row) not in (tuple, list) or len(row) != 3:
                raise Full129RestartError("continuation vector shape differs")
            for item in row:
                if type(item) is not float or not math.isfinite(item):
                    raise Full129RestartError("continuation contains non-finite data")
                values.append(item)
    payload = struct.pack(f"<{len(values)}d", *values)
    if len(payload) != CONTINUATION_RECORD_BYTES:
        raise Full129RestartError("continuation byte geometry differs")
    return payload


def _fixed_records(payload: bytes, record_bytes: int, count: int, label: str) -> tuple[bytes, ...]:
    if type(payload) is not bytes or len(payload) != record_bytes * count:
        raise Full129RestartError(f"{label} geometry differs")
    return tuple(payload[index * record_bytes:(index + 1) * record_bytes] for index in range(count))


def _deterministic_v3_segment_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise Full129RestartError("V3 segment must be an exact object")
    omitted = {"schema", "segment_order", "segment_index", "elapsed_seconds_hex", "elapsed_time_claim", "content_sha256"}
    projection = {key: value[key] for key in sorted(set(value) - omitted)}
    required = {
        "active_tracer_count", "batch_order", "checkpoint_sha256", "continuation_end_sha256",
        "continuation_start_sha256", "end_global_source_step", "end_year",
        "engine_runtime_identity_sha256", "generated_source_segment_sha256",
        "lifecycle_sha256", "maximum_kepler_iterations", "minimum_source_distance_AU_hex",
        "object_id", "removal_record_count", "retained_source_segment_sha256",
        "route_record_count", "source_end_node_sha256", "source_start_node_sha256",
        "source_tile_sha256", "start_global_source_step", "start_year",
    }
    if set(projection) != required:
        raise Full129RestartError("V3 deterministic segment projection shape differs")
    integer_keys = {
        "active_tracer_count", "batch_order", "end_global_source_step", "end_year",
        "maximum_kepler_iterations", "removal_record_count", "route_record_count",
        "start_global_source_step", "start_year",
    }
    if any(type(projection[key]) is not int for key in integer_keys):
        raise Full129RestartError("V3 segment integer type differs")
    return projection


def _segment_record(
    *, restart_order: int, reference: Mapping[str, Any], checkpoint: bytes,
    lifecycle: bytes, continuation_start: bytes, continuation_end: bytes,
    source_segment: bytes, source_start: bytes, source_end: bytes,
    source_tile_sha256: str,
    minimum_distance: float, maximum_kepler_iterations: int,
    engine_projection_sha256: str, elapsed_seconds: float,
) -> dict[str, Any]:
    _require_int(restart_order, "restart segment order", minimum=0)
    if restart_order >= RESTART_SEGMENT_COUNT:
        raise Full129RestartError("restart segment order exceeds suffix")
    reference_index = REFERENCE_START_RECORD + restart_order
    if type(elapsed_seconds) is not float or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise Full129RestartError("segment elapsed time differs")
    if type(minimum_distance) is not float or not math.isfinite(minimum_distance) or not minimum_distance > 1.0:
        raise Full129RestartError("segment minimum distance gate failed")
    _require_int(maximum_kepler_iterations, "maximum Kepler iterations", minimum=0)
    if not _is_sha256(source_tile_sha256):
        raise Full129RestartError("restart source-tile identity differs")
    projection = {
        "active_tracer_count": TRACER_COUNT,
        "batch_order": BATCH_ORDER,
        "checkpoint_sha256": _sha256(checkpoint),
        "continuation_end_sha256": _sha256(continuation_end),
        "continuation_start_sha256": _sha256(continuation_start),
        "end_global_source_step": (reference_index + 1) * SEGMENT_STEPS,
        "end_year": (reference_index + 1) * SEGMENT_YEARS,
        "engine_runtime_identity_sha256": engine_projection_sha256,
        "generated_source_segment_sha256": _sha256(source_segment),
        "lifecycle_sha256": _sha256(lifecycle),
        "maximum_kepler_iterations": maximum_kepler_iterations,
        "minimum_source_distance_AU_hex": minimum_distance.hex(),
        "object_id": OBJECT_ID,
        "removal_record_count": 0,
        "retained_source_segment_sha256": _sha256(source_segment),
        "route_record_count": 0,
        "source_end_node_sha256": _sha256(source_end),
        "source_start_node_sha256": _sha256(source_start),
        "source_tile_sha256": source_tile_sha256,
        "start_global_source_step": reference_index * SEGMENT_STEPS,
        "start_year": reference_index * SEGMENT_YEARS,
    }
    expected = _deterministic_v3_segment_projection(reference)
    if _canonical_json(projection) != _canonical_json(expected):
        raise Full129RestartError("restart segment differs from monolithic deterministic projection")
    value: dict[str, Any] = {
        "schema": SEGMENT_SCHEMA,
        "restart_segment_order": restart_order,
        "reference_segment_index": reference_index,
        "reference_segment_content_sha256": reference.get("content_sha256"),
        "deterministic_projection": projection,
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_ACCEPTANCE_GATE",
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(SEGMENT_DOMAIN, value)
    return value


def _validate_segment_record(
    value: object, *, restart_order: int, reference: Mapping[str, Any], checkpoint: bytes,
    lifecycle: bytes, continuation_start: bytes, continuation_end: bytes,
    source_segment: bytes, source_start: bytes, source_end: bytes,
) -> dict[str, Any]:
    keys = {
        "schema", "restart_segment_order", "reference_segment_index",
        "reference_segment_content_sha256", "deterministic_projection",
        "elapsed_seconds_hex", "elapsed_time_claim", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full129RestartError("restart segment record shape differs")
    if type(value.get("restart_segment_order")) is not int or type(value.get("reference_segment_index")) is not int:
        raise Full129RestartError("restart segment integer type differs")
    reference_index = REFERENCE_START_RECORD + restart_order
    if (
        value.get("schema") != SEGMENT_SCHEMA
        or value.get("restart_segment_order") != restart_order
        or value.get("reference_segment_index") != reference_index
        or value.get("reference_segment_content_sha256") != reference.get("content_sha256")
        or value.get("elapsed_time_claim") != "DIAGNOSTIC_ONLY_NO_ACCEPTANCE_GATE"
        or value.get("content_sha256") != _domain_hash(SEGMENT_DOMAIN, value)
    ):
        raise Full129RestartError("restart segment record identity differs")
    elapsed_hex = value.get("elapsed_seconds_hex")
    if type(elapsed_hex) is not str:
        raise Full129RestartError("restart segment elapsed time type differs")
    try:
        elapsed = float.fromhex(elapsed_hex)
    except ValueError as exc:
        raise Full129RestartError("restart segment elapsed time differs") from exc
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise Full129RestartError("restart segment elapsed time is invalid")
    projection = value.get("deterministic_projection")
    expected = _deterministic_v3_segment_projection(reference)
    if type(projection) is not dict or _canonical_json(projection) != _canonical_json(expected):
        raise Full129RestartError("restart deterministic segment projection differs")
    if (
        projection.get("checkpoint_sha256") != _sha256(checkpoint)
        or projection.get("lifecycle_sha256") != _sha256(lifecycle)
        or projection.get("continuation_start_sha256") != _sha256(continuation_start)
        or projection.get("continuation_end_sha256") != _sha256(continuation_end)
        or projection.get("generated_source_segment_sha256") != _sha256(source_segment)
        or projection.get("retained_source_segment_sha256") != _sha256(source_segment)
        or projection.get("source_start_node_sha256") != _sha256(source_start)
        or projection.get("source_end_node_sha256") != _sha256(source_end)
    ):
        raise Full129RestartError("restart segment artifact crossbinding differs")
    return value


def _runtime_pre(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], request: Mapping[str, Any],
    v3_runtime_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": RUNTIME_PRE_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "restart_request_content_sha256": request["content_sha256"],
        "execution_role": "PRODUCTION_FULL129_RESTART_WORKER",
        "fresh_dynamics_child": True,
        "batch_order": BATCH_ORDER,
        "tracer_count": TRACER_COUNT,
        "start_global_source_step": START_GLOBAL_SOURCE_STEP,
        "planned_end_global_source_step": END_GLOBAL_SOURCE_STEP,
        "v3_runtime_preflight": dict(v3_runtime_preflight),
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        "dynamics_executed_at_receipt": False,
        "classification_generated": False,
        "science_result": False,
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(RUNTIME_PRE_DOMAIN, value)
    return value


def _runtime_post(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], request: Mapping[str, Any],
    v3_runtime_identity: Mapping[str, Any], gate: types.ModuleType,
) -> dict[str, Any]:
    if type(gate) is not types.ModuleType:
        raise Full129RestartError("runtime projection gate must be an exact module")
    comparison_callable = getattr(gate, "_runtime_comparison_projection", None)
    prelane_callable = getattr(gate, "_runtime_prelane_binding_projection", None)
    if type(comparison_callable) is not types.FunctionType or type(prelane_callable) is not types.FunctionType:
        raise Full129RestartError("runtime projection gate callables differ")
    comparison = comparison_callable(v3_runtime_identity)
    prelane = prelane_callable(v3_runtime_identity)
    engine = v3_runtime_identity.get("engine_runtime_projection")
    if type(engine) is not dict:
        raise Full129RestartError("runtime engine projection differs")
    value: dict[str, Any] = {
        "schema": RUNTIME_POST_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "restart_request_content_sha256": request["content_sha256"],
        "execution_role": "PRODUCTION_FULL129_RESTART_WORKER",
        "fresh_dynamics_child": True,
        "batch_order": BATCH_ORDER,
        "tracer_count": TRACER_COUNT,
        "final_global_source_step": END_GLOBAL_SOURCE_STEP,
        "v3_runtime_identity": dict(v3_runtime_identity),
        "v3_runtime_content_sha256": v3_runtime_identity.get("content_sha256"),
        "v3_runtime_comparison_projection_sha256": _sha256(_canonical_json(comparison)),
        "v3_runtime_prelane_binding_projection_sha256": _sha256(_canonical_json(prelane)),
        "engine_runtime_projection_sha256": _sha256(_canonical_json(engine)),
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        "dynamics_executed": True,
        "restart_suffix_completed": True,
        "classification_generated": False,
        "science_result": False,
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(RUNTIME_POST_DOMAIN, value)
    return value


def _validate_runtime_envelopes(
    pre: object, post: object, protocol: Mapping[str, Any], request: Mapping[str, Any],
    v3: types.ModuleType, reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "git_preregistration",
        "restart_request_content_sha256", "execution_role", "fresh_dynamics_child",
        "batch_order", "tracer_count", "start_global_source_step",
        "planned_end_global_source_step", "v3_runtime_preflight", "local_threat_boundary",
        "dynamics_executed_at_receipt", "classification_generated", "science_result",
        "content_sha256",
    }
    post_keys = {
        "schema", "protocol_content_sha256", "protocol_file_sha256", "git_preregistration",
        "restart_request_content_sha256", "execution_role", "fresh_dynamics_child",
        "batch_order", "tracer_count", "final_global_source_step", "v3_runtime_identity",
        "v3_runtime_content_sha256", "v3_runtime_comparison_projection_sha256",
        "v3_runtime_prelane_binding_projection_sha256", "engine_runtime_projection_sha256",
        "local_threat_boundary", "dynamics_executed", "restart_suffix_completed",
        "classification_generated", "science_result", "content_sha256",
    }
    if type(pre) is not dict or set(pre) != pre_keys or type(post) is not dict or set(post) != post_keys:
        raise Full129RestartError("runtime envelope shape differs")
    integer_pre = ("batch_order", "tracer_count", "start_global_source_step", "planned_end_global_source_step")
    integer_post = ("batch_order", "tracer_count", "final_global_source_step")
    common_ok = (
        pre.get("schema") == RUNTIME_PRE_SCHEMA
        and post.get("schema") == RUNTIME_POST_SCHEMA
        and pre.get("protocol_content_sha256") == protocol["content_sha256"]
        and post.get("protocol_content_sha256") == protocol["content_sha256"]
        and pre.get("protocol_file_sha256") == _sha256(_canonical_json(protocol))
        and post.get("protocol_file_sha256") == _sha256(_canonical_json(protocol))
        and pre.get("restart_request_content_sha256") == request["content_sha256"]
        and post.get("restart_request_content_sha256") == request["content_sha256"]
        and pre.get("execution_role") == post.get("execution_role") == "PRODUCTION_FULL129_RESTART_WORKER"
        and pre.get("fresh_dynamics_child") is True and post.get("fresh_dynamics_child") is True
        and pre.get("local_threat_boundary") == post.get("local_threat_boundary") == LOCAL_THREAT_BOUNDARY
        and pre.get("dynamics_executed_at_receipt") is False
        and post.get("dynamics_executed") is True and post.get("restart_suffix_completed") is True
        and pre.get("classification_generated") is False and post.get("classification_generated") is False
        and pre.get("science_result") is False and post.get("science_result") is False
        and pre.get("content_sha256") == _domain_hash(RUNTIME_PRE_DOMAIN, pre)
        and post.get("content_sha256") == _domain_hash(RUNTIME_POST_DOMAIN, post)
        and all(type(pre.get(key)) is int for key in integer_pre)
        and all(type(post.get(key)) is int for key in integer_post)
        and pre.get("batch_order") == post.get("batch_order") == BATCH_ORDER
        and pre.get("tracer_count") == post.get("tracer_count") == TRACER_COUNT
        and pre.get("start_global_source_step") == START_GLOBAL_SOURCE_STEP
        and pre.get("planned_end_global_source_step") == END_GLOBAL_SOURCE_STEP
        and post.get("final_global_source_step") == END_GLOBAL_SOURCE_STEP
    )
    if not common_ok:
        raise Full129RestartError("runtime envelope identity differs")
    pre_custody = _validate_git_custody(pre.get("git_preregistration"), protocol)
    post_custody = _validate_git_custody(post.get("git_preregistration"), protocol)
    if _canonical_json(pre_custody) != _canonical_json(post_custody):
        raise Full129RestartError("runtime Git custody differs")
    with v3._INNER._delegates() as (_preflight, gate):
        with gate._delegates() as (_grid, _h64, dynamics):
            gate._validate_runtime_preflight_document(pre.get("v3_runtime_preflight"), dynamics)
            runtime = gate._validate_runtime_identity(
                post.get("v3_runtime_identity"), dynamics, "full129 restart runtime",
                expected_mode=gate.RUNTIME_MODE_PRODUCTION,
            )
        comparison = gate._runtime_comparison_projection(runtime)
        prelane = gate._runtime_prelane_binding_projection(runtime)
    retained_post = reference.get("runtime_post")
    retained_pre = reference.get("runtime_pre")
    if type(retained_post) is not dict or type(retained_pre) is not dict:
        raise Full129RestartError("V3 retained runtime reference differs")
    if (
        post.get("v3_runtime_content_sha256") != runtime.get("content_sha256")
        or post.get("v3_runtime_comparison_projection_sha256") != _sha256(_canonical_json(comparison))
        or post.get("v3_runtime_prelane_binding_projection_sha256") != _sha256(_canonical_json(prelane))
        or post.get("engine_runtime_projection_sha256") != _sha256(_canonical_json(runtime["engine_runtime_projection"]))
        or post.get("engine_runtime_projection_sha256") != REFERENCE_ENGINE_PROJECTION_SHA256
        or post.get("v3_runtime_comparison_projection_sha256") != retained_post.get("v3_runtime_comparison_projection_sha256")
        or post.get("v3_runtime_prelane_binding_projection_sha256") != retained_post.get("v3_runtime_prelane_binding_projection_sha256")
        or pre["v3_runtime_preflight"].get("prelane_binding_projection_sha256")
        != post.get("v3_runtime_prelane_binding_projection_sha256")
        or retained_pre.get("v3_runtime_preflight", {}).get("prelane_binding_projection_sha256")
        != post.get("v3_runtime_prelane_binding_projection_sha256")
    ):
        raise Full129RestartError("restart runtime differs from retained V3 runtime")
    return pre, post


def _result_claims() -> dict[str, bool]:
    return {
        **CLAIMS_PREPARED,
        "gpu_executed": True,
        "restart_execution_completed": True,
        "batch0_full129_all_active_restart_equivalence_qualified": True,
    }


def _validate_exact_claims(value: object, expected: Mapping[str, bool]) -> None:
    if type(value) is not dict or set(value) != set(expected):
        raise Full129RestartError("claim roster differs")
    for key, expected_value in expected.items():
        if type(value.get(key)) is not bool or value[key] is not expected_value:
            raise Full129RestartError(f"claim differs: {key}")


def _file_receipts(payloads: Sequence[tuple[str, bytes]]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, payload in payloads:
        if name in records or name in ("PROTOCOL.json", "RESULT.json"):
            raise Full129RestartError("result file receipt roster differs")
        records[name] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return records


def _build_result_and_artifacts(
    protocol: Mapping[str, Any], custody: Mapping[str, Any], request: Mapping[str, Any],
    runtime_pre: Mapping[str, Any], runtime_post: Mapping[str, Any], lane: Mapping[str, Any],
    source_build_relocation: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, bytes], ...]]:
    continuation = lane.get("continuation")
    checkpoints = lane.get("checkpoints")
    lifecycle = lane.get("lifecycle")
    segments = lane.get("segments")
    elapsed = lane.get("elapsed_seconds")
    if (
        type(continuation) is not bytes or len(continuation) != CONTINUATION_SUFFIX_BYTES
        or type(checkpoints) is not bytes or len(checkpoints) != CHECKPOINT_SUFFIX_BYTES
        or type(lifecycle) is not bytes or len(lifecycle) != LIFECYCLE_SUFFIX_BYTES
        or type(segments) is not list or len(segments) != RESTART_SEGMENT_COUNT
        or type(elapsed) is not float or not math.isfinite(elapsed) or elapsed <= 0.0
    ):
        raise Full129RestartError("restart lane result geometry differs")
    segment_payload = b"".join(_canonical_json(row) for row in segments)
    payloads = (
        ("RESTART_REQUEST.json", _canonical_json(request)),
        ("RESTART_SUFFIX.bin", continuation),
        ("CHECKPOINT_SUFFIX.bin", checkpoints),
        ("LIFECYCLE_SUFFIX.bin", lifecycle),
        ("SEGMENTS.jsonl", segment_payload),
        ("ROUTES.jsonl", b""),
        ("REMOVALS.jsonl", b""),
        ("RUNTIME_PRE.json", _canonical_json(runtime_pre)),
        ("RUNTIME_POST.json", _canonical_json(runtime_post)),
    )
    continuation_records = _fixed_records(continuation, CONTINUATION_RECORD_BYTES, 11, "restart continuation suffix")
    checkpoint_records = _fixed_records(checkpoints, STATE_RECORD_BYTES, 11, "restart checkpoint suffix")
    lifecycle_records = _fixed_records(lifecycle, LIFECYCLE_RECORD_BYTES, 11, "restart lifecycle suffix")
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000_FRESH_PROCESS_RESTART_PASS_TEN_MYR_NOT_AUTHORIZED",
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "git_preregistration": dict(custody),
        "restart_request_content_sha256": request["content_sha256"],
        "files": _file_receipts(payloads),
        "batch_order": BATCH_ORDER,
        "object_id": OBJECT_ID,
        "tracer_count": TRACER_COUNT,
        "tracer_indices": list(range(TRACER_COUNT)),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "start_global_source_step": START_GLOBAL_SOURCE_STEP,
        "final_global_source_step": END_GLOBAL_SOURCE_STEP,
        "checkpoint_years": list(RESTART_CHECKPOINT_YEARS),
        "continuation_suffix_sha256": _sha256(continuation),
        "checkpoint_suffix_sha256": _sha256(checkpoints),
        "lifecycle_suffix_sha256": _sha256(lifecycle),
        "continuation_record_sha256": [_sha256(row) for row in continuation_records],
        "checkpoint_record_sha256": [_sha256(row) for row in checkpoint_records],
        "lifecycle_record_sha256": [_sha256(row) for row in lifecycle_records],
        "segment_count": RESTART_SEGMENT_COUNT,
        "segment_content_sha256": [row["content_sha256"] for row in segments],
        "route_record_count": 0,
        "removal_record_count": 0,
        "source_file_sha256": H64_SOURCE_SHA256,
        "source_full_lane_sha256": H64_BATCH0_LANE_SHA256,
        "source_full_tile_sha256": H64_BATCH0_TILE_SHA256,
        "source_invariants": dict(lane["source_invariants"]),
        "source_build_identity": dict(lane["source_build_identity"]),
        "source_build_runtime_relocation": dict(source_build_relocation),
        "engine_runtime_projection_sha256": REFERENCE_ENGINE_PROJECTION_SHA256,
        "runtime_pre_content_sha256": runtime_pre["content_sha256"],
        "runtime_post_content_sha256": runtime_post["content_sha256"],
        "elapsed_seconds_hex": elapsed.hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_ACCEPTANCE_GATE",
        "all_predeclared_gates_passed": True,
        "claim_ceiling": protocol["claim_ceiling"]["qualified_if_passed"],
        "claims": _result_claims(),
        "content_sha256": "",
    }
    result["content_sha256"] = _domain_hash(RESULT_DOMAIN, result)
    return result, (*payloads, ("RESULT.json", _canonical_json(result)))


def _validate_file_receipts(value: object, snapshot: Mapping[str, bytes]) -> None:
    names = set(EXPECTED_OUTPUT_ROSTER) - {"PROTOCOL.json", "RESULT.json"}
    if type(value) is not dict or set(value) != names:
        raise Full129RestartError("result file receipt roster differs")
    for name in names:
        row = value[name]
        if (
            type(row) is not dict or set(row) != {"bytes", "sha256"}
            or type(row.get("bytes")) is not int or row["bytes"] != len(snapshot[name])
            or row.get("sha256") != _sha256(snapshot[name])
        ):
            raise Full129RestartError(f"result file receipt differs: {name}")


def _replay_suffix_with_source(
    snapshot: Mapping[str, bytes], reference: Mapping[str, Any], v3: types.ModuleType,
) -> tuple[list[str], list[str], dict[str, Any], str]:
    continuation_records = _fixed_records(
        snapshot["RESTART_SUFFIX.bin"], CONTINUATION_RECORD_BYTES, 11, "restart continuation suffix",
    )
    checkpoint_records = _fixed_records(
        snapshot["CHECKPOINT_SUFFIX.bin"], STATE_RECORD_BYTES, 11, "restart checkpoint suffix",
    )
    lifecycle_records = _fixed_records(
        snapshot["LIFECYCLE_SUFFIX.bin"], LIFECYCLE_RECORD_BYTES, 11, "restart lifecycle suffix",
    )
    if (
        snapshot["RESTART_SUFFIX.bin"] != reference["continuation"]
        or snapshot["CHECKPOINT_SUFFIX.bin"] != reference["checkpoints"]
        or snapshot["LIFECYCLE_SUFFIX.bin"] != reference["lifecycle"]
        or any(row != bytes(TRACER_COUNT) for row in lifecycle_records)
        or snapshot["ROUTES.jsonl"] != b""
        or snapshot["REMOVALS.jsonl"] != b""
    ):
        raise Full129RestartError("restart suffix differs bitwise from retained V3")
    segments = _jsonl(snapshot["SEGMENTS.jsonl"], "restart segments")
    if len(segments) != RESTART_SEGMENT_COUNT:
        raise Full129RestartError("restart segment count differs")
    source_segment_hashes: list[str] = []
    source_endpoint_hashes: list[str] = []
    with _held_regular_file(
        H64_SOURCE_PATH, "retained H64 source evidence",
        expected_bytes=H64_SOURCE_BYTES, expected_sha256=H64_SOURCE_SHA256,
    ) as source_fd:
        lane_payload = _pread_all(
            source_fd, H64_BATCH0_BYTE_LENGTH, "retained batch0 source lane",
            offset=H64_BATCH0_BYTE_OFFSET,
        )
        if _sha256(lane_payload) != H64_BATCH0_LANE_SHA256:
            raise Full129RestartError("retained batch0 source lane differs")
        import numpy as np

        source_gm = np.frombuffer(
            _pread_all(source_fd, MASSIVE_COUNT * 8, "retained source GM", offset=0),
            dtype=np.dtype("<f8"), count=MASSIVE_COUNT,
        ).copy()
        source_positions = np.frombuffer(
            lane_payload[:H64_POSITION_BLOCK_BYTES], dtype=np.dtype("<f8"),
        ).reshape(H64_NODE_COUNT, MASSIVE_COUNT, 3)
        source_velocities = np.frombuffer(
            lane_payload[H64_POSITION_BLOCK_BYTES:], dtype=np.dtype("<f8"),
        ).reshape(H64_NODE_COUNT, MASSIVE_COUNT, 3)
        nodes = tuple(
            _source_node_payload(source_fd, year * STEPS_PER_YEAR)
            for year in RESTART_CHECKPOINT_YEARS
        )
        source_endpoint_hashes = [_sha256(row) for row in nodes]
        if source_endpoint_hashes[0] != REFERENCE_SOURCE_START_SHA256:
            raise Full129RestartError("restart source midpoint differs")
        source_segments = tuple(
            _source_segment_payload(source_fd, REFERENCE_START_RECORD + index)
            for index in range(RESTART_SEGMENT_COUNT)
        )
        source_segment_hashes = [_sha256(row) for row in source_segments]
        with v3._INNER._delegates() as (preflight, gate):
            with gate._delegates() as (grid, _h64, dynamics):
                invariant = dynamics._source_node_invariant_summary(
                    source_positions, source_velocities, source_gm,
                )
                dynamics._require_source_invariant_summary(
                    reference["result"].get("source_invariants"), invariant,
                )
                with grid.FrozenClassifierKernelV1() as kernel:
                    for index, (node, continuation, checkpoint) in enumerate(
                        zip(nodes, continuation_records, checkpoint_records, strict=True)
                    ):
                        source_positions, source_velocities = v3._INNER._source_node_rows(node)
                        tracer_positions, tracer_velocities = _unpack_continuation(continuation)
                        rebuilt = v3._INNER._checkpoint_full129(
                            preflight, grid, kernel, source_positions, source_velocities,
                            tracer_positions, tracer_velocities,
                        )
                        if rebuilt != checkpoint:
                            raise Full129RestartError(f"restart checkpoint replay differs at index {index}")
        for index, segment in enumerate(segments):
            _validate_segment_record(
                segment, restart_order=index, reference=reference["segments"][index],
                checkpoint=checkpoint_records[index + 1], lifecycle=lifecycle_records[index + 1],
                continuation_start=continuation_records[index],
                continuation_end=continuation_records[index + 1],
                source_segment=source_segments[index], source_start=nodes[index], source_end=nodes[index + 1],
            )
    return source_segment_hashes, source_endpoint_hashes, invariant, _sha256(snapshot["SEGMENTS.jsonl"])


def _validate_output_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    if type(snapshot) is not dict or set(snapshot) != EXPECTED_OUTPUT_ROSTER:
        raise Full129RestartError("retained restart output roster differs")
    expected_lengths = {
        "RESTART_SUFFIX.bin": CONTINUATION_SUFFIX_BYTES,
        "CHECKPOINT_SUFFIX.bin": CHECKPOINT_SUFFIX_BYTES,
        "LIFECYCLE_SUFFIX.bin": LIFECYCLE_SUFFIX_BYTES,
        "ROUTES.jsonl": 0,
        "REMOVALS.jsonl": 0,
    }
    if any(len(snapshot[name]) != size for name, size in expected_lengths.items()):
        raise Full129RestartError("retained restart binary geometry differs")
    protocol = _strict_json(snapshot["PROTOCOL.json"])
    validate_protocol(protocol)
    request = _strict_json(snapshot["RESTART_REQUEST.json"])
    _validate_published_request(request, protocol)
    pre = _strict_json(snapshot["RUNTIME_PRE.json"])
    post = _strict_json(snapshot["RUNTIME_POST.json"])
    result = _strict_json(snapshot["RESULT.json"])
    with _held_v3_snapshot() as predecessor_snapshot:
        reference = _reference_suffix(predecessor_snapshot)
        with _v3_delegate() as v3:
            _validate_runtime_envelopes(pre, post, protocol, request, v3, reference)
            source_segment_sha, source_endpoint_sha, source_invariants, _segment_file_sha = _replay_suffix_with_source(
                snapshot, reference, v3,
            )
            relocation = result.get("source_build_runtime_relocation")
            v3._validate_relocation_receipt(relocation, post)
    _validate_file_receipts(result.get("files"), snapshot)
    continuation_records = _fixed_records(snapshot["RESTART_SUFFIX.bin"], CONTINUATION_RECORD_BYTES, 11, "restart continuation suffix")
    checkpoint_records = _fixed_records(snapshot["CHECKPOINT_SUFFIX.bin"], STATE_RECORD_BYTES, 11, "restart checkpoint suffix")
    lifecycle_records = _fixed_records(snapshot["LIFECYCLE_SUFFIX.bin"], LIFECYCLE_RECORD_BYTES, 11, "restart lifecycle suffix")
    segments = _jsonl(snapshot["SEGMENTS.jsonl"], "restart segments")
    expected_keys = {
        "schema", "status", "protocol_content_sha256", "protocol_file_sha256",
        "git_preregistration", "restart_request_content_sha256", "files", "batch_order",
        "object_id", "tracer_count", "tracer_indices", "start_year", "end_year",
        "start_global_source_step", "final_global_source_step", "checkpoint_years",
        "continuation_suffix_sha256", "checkpoint_suffix_sha256", "lifecycle_suffix_sha256",
        "continuation_record_sha256", "checkpoint_record_sha256", "lifecycle_record_sha256",
        "segment_count", "segment_content_sha256", "route_record_count", "removal_record_count",
        "source_file_sha256", "source_full_lane_sha256", "source_full_tile_sha256",
        "source_invariants", "source_build_identity", "source_build_runtime_relocation",
        "engine_runtime_projection_sha256", "runtime_pre_content_sha256",
        "runtime_post_content_sha256", "elapsed_seconds_hex", "elapsed_time_claim",
        "all_predeclared_gates_passed", "claim_ceiling", "claims", "content_sha256",
    }
    integer_keys = (
        "batch_order", "tracer_count", "start_year", "end_year", "start_global_source_step",
        "final_global_source_step", "segment_count", "route_record_count", "removal_record_count",
    )
    indices = result.get("tracer_indices")
    years = result.get("checkpoint_years")
    if (
        type(result) is not dict or set(result) != expected_keys
        or any(type(result.get(key)) is not int for key in integer_keys)
        or type(indices) is not list or any(type(item) is not int for item in indices)
        or indices != list(range(TRACER_COUNT))
        or type(years) is not list or any(type(item) is not int for item in years)
        or years != list(RESTART_CHECKPOINT_YEARS)
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000_FRESH_PROCESS_RESTART_PASS_TEN_MYR_NOT_AUTHORIZED"
        or result.get("protocol_content_sha256") != protocol["content_sha256"]
        or result.get("protocol_file_sha256") != _sha256(snapshot["PROTOCOL.json"])
        or result.get("restart_request_content_sha256") != request["content_sha256"]
        or result.get("batch_order") != BATCH_ORDER or result.get("object_id") != OBJECT_ID
        or result.get("tracer_count") != TRACER_COUNT
        or result.get("start_year") != START_YEAR or result.get("end_year") != END_YEAR
        or result.get("start_global_source_step") != START_GLOBAL_SOURCE_STEP
        or result.get("final_global_source_step") != END_GLOBAL_SOURCE_STEP
        or result.get("continuation_suffix_sha256") != _sha256(snapshot["RESTART_SUFFIX.bin"])
        or result.get("checkpoint_suffix_sha256") != _sha256(snapshot["CHECKPOINT_SUFFIX.bin"])
        or result.get("lifecycle_suffix_sha256") != _sha256(snapshot["LIFECYCLE_SUFFIX.bin"])
        or result.get("continuation_record_sha256") != [_sha256(row) for row in continuation_records]
        or result.get("checkpoint_record_sha256") != [_sha256(row) for row in checkpoint_records]
        or result.get("lifecycle_record_sha256") != [_sha256(row) for row in lifecycle_records]
        or result.get("segment_count") != RESTART_SEGMENT_COUNT
        or result.get("segment_content_sha256") != [row["content_sha256"] for row in segments]
        or result.get("route_record_count") != 0 or result.get("removal_record_count") != 0
        or result.get("source_file_sha256") != H64_SOURCE_SHA256
        or result.get("source_full_lane_sha256") != H64_BATCH0_LANE_SHA256
        or result.get("source_full_tile_sha256") != H64_BATCH0_TILE_SHA256
        or type(result.get("source_invariants")) is not dict
        or _canonical_json(result["source_invariants"]) != _canonical_json(source_invariants)
        or _canonical_json(result["source_invariants"])
        != _canonical_json(reference["result"].get("source_invariants"))
        or type(result.get("source_build_identity")) is not dict
        or _canonical_json(result["source_build_identity"])
        != _canonical_json(reference["result"].get("source_build_identity"))
        or result.get("engine_runtime_projection_sha256") != REFERENCE_ENGINE_PROJECTION_SHA256
        or result.get("runtime_pre_content_sha256") != pre["content_sha256"]
        or result.get("runtime_post_content_sha256") != post["content_sha256"]
        or result.get("elapsed_time_claim") != "DIAGNOSTIC_ONLY_NO_ACCEPTANCE_GATE"
        or result.get("all_predeclared_gates_passed") is not True
        or result.get("claim_ceiling") != protocol["claim_ceiling"]["qualified_if_passed"]
        or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result)
    ):
        raise Full129RestartError("retained restart result semantics differ")
    _validate_git_custody(result.get("git_preregistration"), protocol)
    if _canonical_json(result["git_preregistration"]) != _canonical_json(pre["git_preregistration"]):
        raise Full129RestartError("result Git custody differs")
    _validate_exact_claims(result.get("claims"), _result_claims())
    try:
        elapsed = float.fromhex(result.get("elapsed_seconds_hex", ""))
    except (TypeError, ValueError) as exc:
        raise Full129RestartError("restart elapsed time differs") from exc
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise Full129RestartError("restart elapsed time is not positive finite")
    # These values are replayed above.  Keeping the locals explicit prevents a
    # future validator refactor from silently dropping held-source traversal.
    if len(source_segment_sha) != 10 or len(source_endpoint_sha) != 11:
        raise Full129RestartError("held-source replay roster differs")
    return result


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _absolute_lexical(output_root)
    lengths: dict[str, int | None] = {
        "PROTOCOL.json": None,
        "RESTART_REQUEST.json": None,
        "RESTART_SUFFIX.bin": CONTINUATION_SUFFIX_BYTES,
        "CHECKPOINT_SUFFIX.bin": CHECKPOINT_SUFFIX_BYTES,
        "LIFECYCLE_SUFFIX.bin": LIFECYCLE_SUFFIX_BYTES,
        "SEGMENTS.jsonl": None,
        "ROUTES.jsonl": 0,
        "REMOVALS.jsonl": 0,
        "RUNTIME_PRE.json": None,
        "RUNTIME_POST.json": None,
        "RESULT.json": None,
    }
    with _held_directory(root, "retained restart root", expected_roster=EXPECTED_OUTPUT_ROSTER) as root_fd:
        snapshot: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int, int, int, int]] = {}
        for name in sorted(EXPECTED_OUTPUT_ROSTER):
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            identities[name] = _stat_identity(metadata)
            snapshot[name] = _read_regular_at(root_fd, name, f"retained {name}", lengths[name])
        result = _validate_output_snapshot(snapshot)
        if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
            raise Full129RestartError("retained restart roster changed during replay")
        for name, identity in identities.items():
            if _stat_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                raise Full129RestartError(f"retained restart output changed during replay: {name}")
        return result


def _host_rows(np: Any, value: object, count: int, label: str) -> tuple[tuple[float, float, float], ...]:
    if type(value) is not np.ndarray or value.dtype != np.dtype("<f8") or value.shape != (count, 3):
        raise Full129RestartError(f"{label} array geometry differs")
    rows = tuple(tuple(float(value[index, axis]) for axis in range(3)) for index in range(count))
    if any(type(item) is not float or not math.isfinite(item) for row in rows for item in row):
        raise Full129RestartError(f"{label} contains non-finite data")
    return rows  # type: ignore[return-value]


def _lifecycle_row(np: Any, value: object) -> bytes:
    if type(value) is not np.ndarray or value.dtype != np.dtype("uint8") or value.shape != (TRACER_COUNT,):
        raise Full129RestartError("restart lifecycle vector geometry differs")
    payload = value.tobytes(order="C")
    if payload != bytes(TRACER_COUNT):
        raise Full129RestartError("restart lifecycle vector is not exact all-active")
    return payload


def _restart_state(
    cp: Any, year_zero_state: object, midpoint: bytes,
) -> object:
    import numpy as np

    positions, velocities = _unpack_continuation(midpoint)
    host_positions = np.asarray(positions, dtype=np.dtype("<f8"), order="C")
    host_velocities = np.asarray(velocities, dtype=np.dtype("<f8"), order="C")
    if host_positions.shape != (TRACER_COUNT, 3) or host_velocities.shape != (TRACER_COUNT, 3):
        raise Full129RestartError("restart midpoint host arrays differ")
    expected_snapshot = f"tno-full129-smoke-v1-DE441-{OBJECT_ID}"
    if getattr(year_zero_state, "snapshot_id", None) != expected_snapshot:
        raise Full129RestartError("monolithic snapshot identity differs")
    if getattr(year_zero_state, "global_source_step", None) != 0:
        raise Full129RestartError("monolithic base state does not start at zero")
    state = replace(
        year_zero_state,
        global_source_step=START_GLOBAL_SOURCE_STEP,
        tracer_relative_positions_AU=cp.asarray(host_positions),
        tracer_relative_velocities_AU_per_year=cp.asarray(host_velocities),
        tracer_indices=cp.arange(TRACER_COUNT, dtype=cp.uint64),
    )
    if (
        getattr(state, "snapshot_id", None) != expected_snapshot
        or getattr(state, "global_source_step", None) != START_GLOBAL_SOURCE_STEP
        or tuple(int(item) for item in cp.asnumpy(state.tracer_indices).tolist()) != tuple(range(TRACER_COUNT))
        or cp.asnumpy(state.tracer_relative_positions_AU).astype(np.dtype("<f8"), copy=False).tobytes(order="C")
        != host_positions.tobytes(order="C")
        or cp.asnumpy(state.tracer_relative_velocities_AU_per_year).astype(np.dtype("<f8"), copy=False).tobytes(order="C")
        != host_velocities.tobytes(order="C")
    ):
        raise Full129RestartError("restart midpoint device transfer differs")
    return state


def _run_restart_lane(
    v3: types.ModuleType, preflight: types.ModuleType, gate: types.ModuleType,
    grid: types.ModuleType, dynamics: types.ModuleType, cp: Any,
    bridge: Mapping[str, Any], custody: Mapping[str, Any], retained_result: Mapping[str, Any],
    source_descriptor: int, kernel: object, reference: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    context = gate._build_batch_context(grid, dynamics, cp, bridge, custody, BATCH_ORDER)
    full = context["full"]
    build = context["build"]
    specs = context["specs"]
    lane, source_record = gate._retained_lane_records(retained_result, BATCH_ORDER)
    position_payload, velocity_payload, gm_payload = gate._source_payload(np, full)
    generated_source = position_payload + velocity_payload
    retained_source = gate._pread_exact(
        source_descriptor, int(source_record["byte_offset"]), int(source_record["byte_length"]),
        "retained full source lane",
    )
    retained_gm = gate._pread_exact(source_descriptor, 0, gate.SOURCE_COMMON_GM_BYTES, "retained source GM")
    if (
        generated_source != retained_source or gm_payload != retained_gm
        or len(generated_source) != H64_BATCH0_BYTE_LENGTH
        or _sha256(generated_source) != H64_BATCH0_LANE_SHA256
        or source_record.get("lane_payload_sha256") != H64_BATCH0_LANE_SHA256
        or full.content_sha256 != H64_BATCH0_TILE_SHA256
        or source_record.get("full_source_tile_sha256") != H64_BATCH0_TILE_SHA256
    ):
        raise Full129RestartError("generated source path differs from retained H64 evidence")
    invariant = dynamics._source_node_invariant_summary(
        full.source_relative_positions_AU, full.source_relative_velocities_AU_per_year,
        full.source_gm_AU3_per_year2,
    )
    dynamics._require_source_invariant_summary(lane.get("summary"), invariant)
    source_build_identity = v3._source_build_identity(gate, build)
    if source_build_identity != reference["result"].get("source_build_identity"):
        raise Full129RestartError("restart source-build identity differs from V3")
    relocation_base = v3._RELOCATION_CONTEXT.get()
    if type(relocation_base) is not dict:
        raise Full129RestartError("restart source-build relocation base is absent")

    primary_payload = v3._INNER._batch0_primary_payload(preflight)
    year_zero_state, _initial_positions, _initial_velocities = v3._INNER._full129_state(
        cp, context, primary_payload, preflight,
    )
    continuation_records = _fixed_records(
        reference["continuation"], CONTINUATION_RECORD_BYTES, 11, "V3 continuation suffix",
    )
    checkpoint_records = _fixed_records(
        reference["checkpoints"], STATE_RECORD_BYTES, 11, "V3 checkpoint suffix",
    )
    lifecycle_records = _fixed_records(
        reference["lifecycle"], LIFECYCLE_RECORD_BYTES, 11, "V3 lifecycle suffix",
    )
    if lifecycle_records[0] != bytes(TRACER_COUNT):
        raise Full129RestartError("restart midpoint is not all-active")
    state = _restart_state(cp, year_zero_state, continuation_records[0])
    source_positions = _host_rows(
        np, np.ascontiguousarray(full.source_relative_positions_AU[START_GLOBAL_SOURCE_STEP], dtype=np.dtype("<f8")),
        MASSIVE_COUNT, "restart midpoint source positions",
    )
    source_velocities = _host_rows(
        np, np.ascontiguousarray(full.source_relative_velocities_AU_per_year[START_GLOBAL_SOURCE_STEP], dtype=np.dtype("<f8")),
        MASSIVE_COUNT, "restart midpoint source velocities",
    )
    midpoint_positions, midpoint_velocities = _unpack_continuation(continuation_records[0])
    rebuilt_midpoint = v3._INNER._checkpoint_full129(
        preflight, grid, kernel, source_positions, source_velocities,
        midpoint_positions, midpoint_velocities,
    )
    if rebuilt_midpoint != checkpoint_records[0]:
        raise Full129RestartError("restart midpoint checkpoint crossbinding differs")

    lifecycle_state = specs["lifecycle_initialize"](state)
    if _lifecycle_row(np, lifecycle_state.tracer_lifecycle) != lifecycle_records[0]:
        raise Full129RestartError("initialized restart lifecycle differs")
    generated_continuations = [continuation_records[0]]
    generated_checkpoints = [checkpoint_records[0]]
    generated_lifecycle = [lifecycle_records[0]]
    segment_records: list[dict[str, Any]] = []
    engine_projection: dict[str, Any] | None = None
    started_lane = time.perf_counter()
    for restart_order in range(RESTART_SEGMENT_COUNT):
        reference_index = REFERENCE_START_RECORD + restart_order
        local_start = reference_index * SEGMENT_STEPS
        tile = grid.slice_frozen_source_tile_v1(
            full, local_start_step=local_start, map_step_count=SEGMENT_STEPS,
            tile_id=v3._INNER._v3_source_tile_id(reference_index),
        )
        if tile.content_sha256 != reference["segments"][restart_order].get("source_tile_sha256"):
            raise Full129RestartError("restart source-tile identity differs from V3")
        generated_segment = gate._source_segment_payload(np, full, reference_index)
        retained_segment = gate._retained_source_segment(source_descriptor, source_record, reference_index)
        if generated_segment != retained_segment:
            raise Full129RestartError("restart generated source segment differs")
        source_start = v3._INNER._source_node_payload(
            gate, source_descriptor, source_record, reference_index * SEGMENT_STEPS,
        )
        source_end = v3._INNER._source_node_payload(
            gate, source_descriptor, source_record, (reference_index + 1) * SEGMENT_STEPS,
        )
        started_segment = time.perf_counter()
        advanced = specs["lifecycle_advance"](
            lifecycle_state, tile, specs["fixed_step"], route_spec=specs["route"],
            route_worker_count=specs["route_worker_count"],
        )
        elapsed = time.perf_counter() - started_segment
        lifecycle_state = advanced.state
        expected_step = (reference_index + 1) * SEGMENT_STEPS
        lifecycle_payload = _lifecycle_row(np, lifecycle_state.tracer_lifecycle)
        if (
            lifecycle_state.dynamics_state.global_source_step != expected_step
            or advanced.route_records != () or advanced.new_removal_records != ()
            or lifecycle_state.removal_records != ()
            or lifecycle_payload != lifecycle_records[restart_order + 1]
        ):
            raise Full129RestartError("restart lifecycle routed, removed, or incompletely advanced")
        minimum_values = tuple(float(item) for item in cp.asnumpy(advanced.minimum_source_distance_AU).tolist())
        if len(minimum_values) != TRACER_COUNT or any(not math.isfinite(item) or not item > 1.0 for item in minimum_values):
            raise Full129RestartError("restart minimum-distance gate failed")
        projection = gate._engine_runtime_projection(advanced.runtime_identity)
        if type(projection) is not dict or _sha256(_canonical_json(projection)) != REFERENCE_ENGINE_PROJECTION_SHA256:
            raise Full129RestartError("restart engine runtime projection differs")
        if engine_projection is None:
            engine_projection = projection
        elif _canonical_json(engine_projection) != _canonical_json(projection):
            raise Full129RestartError("restart engine runtime projection changed")
        dynamics_state = lifecycle_state.dynamics_state
        host_positions = _host_rows(
            np, cp.asnumpy(dynamics_state.tracer_relative_positions_AU),
            TRACER_COUNT, "restart tracer positions",
        )
        host_velocities = _host_rows(
            np, cp.asnumpy(dynamics_state.tracer_relative_velocities_AU_per_year),
            TRACER_COUNT, "restart tracer velocities",
        )
        checkpoint = v3._INNER._checkpoint_full129(
            preflight, grid, kernel,
            _host_rows(
                np, np.ascontiguousarray(full.source_relative_positions_AU[expected_step], dtype=np.dtype("<f8")),
                MASSIVE_COUNT, "restart source positions",
            ),
            _host_rows(
                np, np.ascontiguousarray(full.source_relative_velocities_AU_per_year[expected_step], dtype=np.dtype("<f8")),
                MASSIVE_COUNT, "restart source velocities",
            ),
            host_positions, host_velocities,
        )
        continuation = _pack_continuation(host_positions, host_velocities)
        if (
            checkpoint != checkpoint_records[restart_order + 1]
            or continuation != continuation_records[restart_order + 1]
        ):
            raise Full129RestartError("restart trajectory suffix differs bitwise from V3")
        generated_checkpoints.append(checkpoint)
        generated_continuations.append(continuation)
        generated_lifecycle.append(lifecycle_payload)
        segment_records.append(_segment_record(
            restart_order=restart_order, reference=reference["segments"][restart_order],
            checkpoint=checkpoint, lifecycle=lifecycle_payload,
            continuation_start=generated_continuations[restart_order],
            continuation_end=continuation, source_segment=retained_segment,
            source_start=source_start, source_end=source_end,
            source_tile_sha256=tile.content_sha256,
            minimum_distance=min(minimum_values),
            maximum_kepler_iterations=int(advanced.maximum_kepler_iterations),
            engine_projection_sha256=REFERENCE_ENGINE_PROJECTION_SHA256,
            elapsed_seconds=elapsed,
        ))
    if engine_projection is None or lifecycle_state.dynamics_state.global_source_step != END_GLOBAL_SOURCE_STEP:
        raise Full129RestartError("restart lane did not reach the exact final step")
    continuation_payload = b"".join(generated_continuations)
    checkpoint_payload = b"".join(generated_checkpoints)
    lifecycle_payload = b"".join(generated_lifecycle)
    if (
        continuation_payload != reference["continuation"]
        or checkpoint_payload != reference["checkpoints"]
        or lifecycle_payload != reference["lifecycle"]
    ):
        raise Full129RestartError("restart final suffix equality failed")
    return {
        "continuation": continuation_payload,
        "checkpoints": checkpoint_payload,
        "lifecycle": lifecycle_payload,
        "segments": segment_records,
        "source_invariants": invariant,
        "source_build_identity": source_build_identity,
        "source_build_runtime_relocation_base": relocation_base,
        "engine_projection": engine_projection,
        "elapsed_seconds": time.perf_counter() - started_lane,
    }


def _publish(output_root: Path, artifacts: Sequence[tuple[str, bytes]]) -> None:
    if [name for name, _payload in artifacts] != list(EXPECTED_OUTPUT_ORDER):
        raise Full129RestartError("publication order differs")
    root = _absolute_lexical(output_root)
    stage = Path(tempfile.mkdtemp(prefix=".tno-full129-restart-publish-", dir=root.parent))
    committed = False
    try:
        with _held_directory(stage, "restart publication stage") as stage_fd:
            for name, payload in artifacts:
                _write_create_only_at(stage_fd, name, payload)
            os.fsync(stage_fd)
        with _held_directory(root, "restart publication root", expected_roster=PREPARED_ROSTER) as root_fd:
            protocol_payload = _read_regular_at(root_fd, "PROTOCOL.json", "publication protocol")
            protocol = _strict_json(protocol_payload)
            validate_protocol(protocol)
            for name, _payload in artifacts[:-1]:
                source = stage / name
                source_stat = source.stat(follow_symlinks=False)
                os.link(source, name, dst_dir_fd=root_fd, follow_symlinks=False)
                destination = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if (source_stat.st_dev, source_stat.st_ino) != (destination.st_dev, destination.st_ino):
                    raise Full129RestartError("publication link identity differs")
                source.unlink()
                if os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_nlink != 1:
                    raise Full129RestartError("publication non-result link count differs")
            if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER - {"RESULT.json"}:
                raise Full129RestartError("pre-result publication roster differs")
            os.fsync(root_fd)
            result_source = stage / "RESULT.json"
            result_stat = result_source.stat(follow_symlinks=False)
            os.link(result_source, "RESULT.json", dst_dir_fd=root_fd, follow_symlinks=False)
            destination = os.stat("RESULT.json", dir_fd=root_fd, follow_symlinks=False)
            if (result_stat.st_dev, result_stat.st_ino) != (destination.st_dev, destination.st_ino):
                raise Full129RestartError("publication RESULT identity differs")
            committed = True
            result_source.unlink()
            if os.stat("RESULT.json", dir_fd=root_fd, follow_symlinks=False).st_nlink != 1:
                raise Full129RestartError("publication RESULT link count differs")
            if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
                raise Full129RestartError("publication final roster differs")
            os.fsync(root_fd)
    except Exception:
        # Never remove names from the destination.  Before RESULT this leaves an
        # exact-roster-invalid fail-closed root.  After RESULT it leaves a root
        # whose validity is decided only by validate_output.
        raise
    finally:
        try:
            for item in stage.iterdir():
                item.unlink()
            stage.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            if not committed:
                raise
    with _held_directory(root, "published restart root", expected_roster=EXPECTED_OUTPUT_ROSTER) as root_fd:
        for name in EXPECTED_OUTPUT_ROSTER:
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise Full129RestartError("published restart output link count differs")


def _write_temp_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        payload = _canonical_json(value)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise Full129RestartError(f"{label} write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _execute_worker(
    output_root: Path, import_probe_path: Path, custody_path: Path, request_path: Path,
) -> dict[str, Any]:
    root = _absolute_lexical(output_root)
    if root != DEFAULT_OUTPUT_ROOT:
        raise Full129RestartError("preregistered restart execution requires the exact default output root")
    protocol = validate_prepared(root)
    custody_payload = _read_regular(custody_path, "Git custody transfer")
    custody = _strict_json(custody_payload)
    _validate_git_custody(custody, protocol)
    if custody_payload != _canonical_json(custody):
        raise Full129RestartError("Git custody transfer is not canonical")
    live_custody = _git_custody(
        externally_preregistered=True,
        prereg_commit=custody["commit"], prereg_tag_object=custody["tag_object"],
    )
    if _canonical_json(live_custody) != _canonical_json(custody):
        raise Full129RestartError("worker live Git custody differs from parent")
    transfer_payload = _read_regular(request_path, "restart request transfer")
    transfer = _strict_json(transfer_payload)
    _validate_request_transfer(transfer, protocol)
    request = _published_request(transfer)
    with _held_v3_snapshot() as predecessor_snapshot:
        reference = _reference_suffix(predecessor_snapshot)
        with _v3_delegate() as v3:
            with v3._INNER._delegates() as (preflight, gate):
                v3_preflight = gate.runtime_preflight()
                runtime_pre = _runtime_pre(protocol, custody, request, v3_preflight)
                retained_result = gate._load_retained_result()
                with gate._delegates() as (grid, _h64, dynamics):
                    gate._validate_runtime_preflight_document(v3_preflight, dynamics)
                    fresh_child = dynamics._fresh_child_interpreter_receipt()
                    fresh_registry = dynamics._reject_preloaded_runtime_modules()
                    _probe_payload, probe = dynamics._load_json(import_probe_path, "restart import probe")
                    dynamics._validate_import_probe_receipt(probe, expected_path=import_probe_path)
                    de441_custody = dynamics.validate_execution_prerequisites(dynamics.DE441_ROOT)
                    bridge = dynamics._load_json(dynamics.BRIDGE_STATES_PATH, "bridge states")[1]
                    with gate._held_source_file() as source_fd:
                        with dynamics._source_only_runtime_import_guard(
                            root / ".full129-restart-runtime-pycache-must-remain-absent"
                        ) as guard:
                            cp, _rebound, _target = dynamics._load_dynamics_runtime(dynamics.DEFAULT_REBOUND_TARGET)
                            runtime_base = gate._critical_runtime_base(
                                dynamics, cp, fresh_child=fresh_child, fresh_registry=fresh_registry,
                            )
                            with grid.FrozenClassifierKernelV1() as kernel:
                                lane = _run_restart_lane(
                                    v3, preflight, gate, grid, dynamics, cp, bridge,
                                    de441_custody, retained_result, source_fd, kernel, reference,
                                )
                        v3_runtime = gate._critical_runtime_identity(
                            dynamics, runtime_base=runtime_base, import_guard=guard,
                            engine_projection=lane["engine_projection"],
                            expected_mode=gate.RUNTIME_MODE_PRODUCTION,
                        )
            runtime_post = _runtime_post(protocol, custody, request, v3_runtime, gate)
            if (
                v3_preflight.get("prelane_binding_projection_sha256")
                != runtime_post["v3_runtime_prelane_binding_projection_sha256"]
            ):
                raise Full129RestartError("restart runtime differs from pre-lane binding")
            relocation = v3._bind_relocation_to_runtime_post(
                lane["source_build_runtime_relocation_base"], runtime_post,
            )
            result, artifacts = _build_result_and_artifacts(
                protocol, custody, request, runtime_pre, runtime_post, lane, relocation,
            )
            snapshot = {
                "PROTOCOL.json": _canonical_json(protocol),
                **{name: payload for name, payload in artifacts},
            }
            validated = _validate_output_snapshot(snapshot)
            if _canonical_json(validated) != _canonical_json(result):
                raise Full129RestartError("prepublication restart replay differs")
    _publish(root, artifacts)
    return result


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, externally_preregistered: bool = False,
    prereg_commit: str = "", prereg_tag_object: str = "",
) -> dict[str, Any]:
    root = _absolute_lexical(output_root)
    if root != DEFAULT_OUTPUT_ROOT:
        raise Full129RestartError("preregistered restart execution requires the exact default output root")
    protocol = validate_prepared(root)
    custody = _git_custody(
        externally_preregistered=externally_preregistered,
        prereg_commit=prereg_commit, prereg_tag_object=prereg_tag_object,
    )
    challenge = secrets.token_hex(32)
    transfer = _request_transfer(protocol, parent_pid=os.getpid(), challenge_hex=challenge)
    with _v3_delegate() as v3:
        with v3._INNER._delegates() as (_preflight, gate):
            with gate._delegates() as (_grid, _h64, dynamics):
                dynamics._reject_preloaded_runtime_modules()
                dynamics.validate_execution_prerequisites(dynamics.DE441_ROOT)
                with tempfile.TemporaryDirectory(prefix="jx-tno-full129-restart-parent-") as raw:
                    directory = Path(raw)
                    probe_path, _probe = dynamics._run_import_only_preflight(
                        dynamics.DEFAULT_REBOUND_TARGET, directory,
                    )
                    custody_path = directory / "GIT_CUSTODY.json"
                    request_path = directory / "RESTART_REQUEST_TRANSFER.json"
                    _write_temp_json(custody_path, custody, "Git custody transfer")
                    _write_temp_json(request_path, transfer, "restart request transfer")
                    completed = subprocess.run(
                        [
                            str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "execute-worker",
                            "--output-root", str(root), "--import-probe-receipt", str(probe_path),
                            "--git-custody", str(custody_path), "--request-transfer", str(request_path),
                        ],
                        cwd=PROJECT_ROOT, env=dynamics._production_child_environment(),
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        check=False, timeout=7_200,
                    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
        raise Full129RestartError(f"fresh full129 restart worker failed: {detail}")
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
    parser.add_argument("--request-transfer", type=Path)
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
        if args.import_probe_receipt is None or args.git_custody is None or args.request_transfer is None:
            raise Full129RestartError("execute-worker requires import-probe, Git-custody, and request-transfer paths")
        _execute_worker(args.output_root, args.import_probe_receipt, args.git_custody, args.request_transfer)
    else:  # pragma: no cover
        raise Full129RestartError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
