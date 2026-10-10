#!/usr/bin/env python3
"""One-shot resolved-11 source-map convergence diagnostic.

This successor investigates the failed source-map gates in the frozen
Stage-1.5 v8 result.  It reuses the exact retained v8 h64/h128 source nodes,
adds h256 and h512 paths for all twelve DE441 native epochs, conditionally
adds h1024 under a predeclared outcome-blind trigger, and creates one tighter
source-only REBOUND 5.1.1 IAS15 replicate per epoch.  No tracer state is
constructed and the public tracer integrator is never called.

Import, protocol construction, prepare, and prepared-root validation are
source-only.  Execution and output validation occur in fresh project-venv
children.  Every output remains a diagnostic: it cannot establish Stage-1.5
qualification, a TNO class, a resonance, a population inference, or a Planet
Nine result, and it never launches an adaptive follow-up run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterator, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)
TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "tno_stage15_resolved11_source_convergence_diagnostic_v1"
)
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

V8_SOURCE_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_stage15_resolved11_encounter_qualification_v8.py"
)
V8_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_tno_stage15_resolved11_encounter_qualification_v8.py"
)
V8_ROOT = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v8"
V8_PROTOCOL_PATH = V8_ROOT / "PROTOCOL.json"
V8_RESULT_PATH = V8_ROOT / "RESULT.json"
V8_SOURCE_NODES_PATH = V8_ROOT / "SOURCE_NODES.bin"

V4_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"
V4_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "PROTOCOL.json"
V4_RESULT_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "RESULT.json"
V4_ARRAYS_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "ARRAYS.bin"
BRIDGE_PATH = PROJECT_ROOT / "runs" / "tno_native_epoch_resolved11_bridge_v1" / "BRIDGE.json"
DE441_CUSTODY_PATH = (
    PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1" / "CUSTODY.json"
)
STAGE15_V1_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "tno_stage15_resolved11_encounter_qualification_v1"
    / "PROTOCOL.json"
)
REBOUND_WHEEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "tno_stage1_rebound511_runtime_v1"
    / "rebound-5.1.1-cp314-manylinux_x86_64.whl"
)

PROTOCOL_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v1"
RESULT_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.result.v1"
RUNTIME_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.runtime.v1"
SOURCE_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.source-nodes.v1"
ANNUAL_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.annual.v1"
VALIDATION_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.validation.v1"

PROTOCOL_DOMAIN = b"jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v1\0"
RESULT_DOMAIN = b"jx.tno.stage15-resolved11-source-convergence-diagnostic.result.v1\0"
RUNTIME_DOMAIN = b"jx.tno.stage15-resolved11-source-convergence-diagnostic.runtime.v1\0"
VALIDATION_DOMAIN = b"jx.tno.stage15-resolved11-source-convergence-diagnostic.validation.v1\0"

V8_SOURCE_SHA256 = "7cd4f355e6f6519f6226f68dbe467b50a7d11265e28ade779f68ab95b437bb89"
V8_SOURCE_BYTES = 329_681
V8_TEST_SHA256 = "431e479f6981d08da95d24f553a4e672ec4ee419c755b2803b4044072088331f"
V8_TEST_BYTES = 91_211
V8_PROTOCOL_SHA256 = "81c29c710ce15fb7cd44695d3219c681e911063561e39604a41ac5964a6f801e"
V8_PROTOCOL_CONTENT_SHA256 = "2be99a0449cfd84b0b9f9d1d679560cd6dae34c5e1808c2222ad15e130702223"
V8_PROTOCOL_BYTES = 35_207
V8_RESULT_SHA256 = "8fc56b4db653eeebb0cb59674284926ea7f71c644ee562cdc1f9dcd96f2a3adf"
V8_RESULT_CONTENT_SHA256 = "e85059b61cadcddebb7e35401c4dfed7946cc57f8b398058dd857f19359d18e2"
V8_RESULT_BYTES = 42_387_995
V8_SOURCE_NODES_SHA256 = "f96624e1ecbbf287a83aaa489106bcb181627d7d9aab8e3adb2473d5c5690005"
V8_SOURCE_NODES_BYTES = 1_216_524_672
V8_GIT_BRANCH = "research/tno-stage15-resolved11-encounter-v8-preregistered-20261010"
V8_GIT_COMMIT = "0fe340feb3fa8b666264e0128ed084697cfb642f"
V8_GIT_TAG = "tno-stage15-resolved11-encounter-v8-preregistered-20261010"
V8_GIT_TAG_OBJECT = "ba548eda088fbb171a61e7efa2906e2bb36ca81d"

V4_SOURCE_SHA256 = "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
V4_SOURCE_BYTES = 436_125
V4_PROTOCOL_SHA256 = "5aaa8a2db5808c1b343fb25473a50aa2bcbc28059e2d724170cd94063f72c05d"
V4_PROTOCOL_CONTENT_SHA256 = "3ea78828498b71e688ad12e6210f3c532689265a6ba202a4d5a6bae5d08f3647"
V4_PROTOCOL_BYTES = 74_186
V4_RESULT_SHA256 = "bebd9186da0ef8943e05a8c4afc057073d48a8a47e9a4d006dae6e59921d5c42"
V4_RESULT_CONTENT_SHA256 = "df4e089aad8d436e99871c943343ea8b631ca714c946edb388387307c75d87d2"
V4_RESULT_BYTES = 10_602_296
V4_ARRAYS_SHA256 = "e7ce12bed175e51831b4d55e558b3e813fbd555fd18c5833f91203feb2921346"
V4_ARRAYS_BYTES = 71_495_424
BRIDGE_SHA256 = "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"
BRIDGE_BYTES = 168_684
DE441_CUSTODY_SHA256 = "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"
DE441_CUSTODY_BYTES = 141_501
STAGE15_V1_PROTOCOL_SHA256 = "7cd7d81c9fb094ed5e01f8e0a1c513b309268e90e92ddc782d9b42ca1d293aee"
STAGE15_V1_PROTOCOL_CONTENT_SHA256 = "48abc9f4e58c57e4b1c5e28c00ed829aaae10fafbca48216d90695b7b255918d"
STAGE15_V1_PROTOCOL_BYTES = 31_382
REBOUND_WHEEL_SHA256 = "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19"
REBOUND_WHEEL_BYTES = 903_973

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
MASSIVE_IDS = (
    "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS_BARYCENTER",
    "JUPITER_BARYCENTER", "SATURN_BARYCENTER", "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER", "PLUTO_BARYCENTER",
)
MASSIVE_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
BATCH_COUNT = 12
MASSIVE_COUNT = 11
HORIZON_YEARS = 1_000
CHECKPOINT_COUNT = 1_001
SOURCE_NODE_RECORD_DOUBLES = 2 * MASSIVE_COUNT * 3
SOURCE_NODE_RECORD_BYTES = SOURCE_NODE_RECORD_DOUBLES * 8
ANNUAL_RECORD_BYTES = SOURCE_NODE_RECORD_BYTES
BASE_STEPS = (256, 512)
ADJUDICATION_STEPS = 1_024
RETAINED_STEPS = (64, 128)
ALL_STEP_LEVELS = (64, 128, 256, 512, 1024)
SOURCE_CHUNK_NODES = 8_192

H256_NODE_COUNT = HORIZON_YEARS * 256 + 1
H512_NODE_COUNT = HORIZON_YEARS * 512 + 1
H1024_NODE_COUNT = HORIZON_YEARS * 1024 + 1
SOURCE_BASE_EXACT_BYTES = BATCH_COUNT * (H256_NODE_COUNT + H512_NODE_COUNT) * SOURCE_NODE_RECORD_BYTES
SOURCE_H1024_MAXIMUM_BYTES = BATCH_COUNT * H1024_NODE_COUNT * SOURCE_NODE_RECORD_BYTES
SOURCE_MAXIMUM_BYTES = SOURCE_BASE_EXACT_BYTES + SOURCE_H1024_MAXIMUM_BYTES
ANNUAL_BASE_EXACT_BYTES = BATCH_COUNT * 3 * CHECKPOINT_COUNT * ANNUAL_RECORD_BYTES
ANNUAL_H1024_MAXIMUM_BYTES = BATCH_COUNT * CHECKPOINT_COUNT * ANNUAL_RECORD_BYTES
ANNUAL_MAXIMUM_BYTES = ANNUAL_BASE_EXACT_BYTES + ANNUAL_H1024_MAXIMUM_BYTES
TOTAL_BASE_FIXED_STEPS = BATCH_COUNT * HORIZON_YEARS * sum(BASE_STEPS)
TOTAL_H1024_MAXIMUM_STEPS = BATCH_COUNT * HORIZON_YEARS * ADJUDICATION_STEPS
TOTAL_MAXIMUM_FIXED_STEPS = TOTAL_BASE_FIXED_STEPS + TOTAL_H1024_MAXIMUM_STEPS

POSITION_BUDGET = Fraction(1, 500)
VELOCITY_BUDGET = Fraction(1, 50_000)
REFERENCE_POSITION_BUDGET = POSITION_BUDGET / 10
REFERENCE_VELOCITY_BUDGET = VELOCITY_BUDGET / 10
ENERGY_BUDGET = Fraction(1, 100_000_000)
ANGULAR_BUDGET = Fraction(1, 10_000_000_000)
ENERGY_CONTRACTION_FLOOR = Fraction(1, 2**46)
ORDER_LOWER_SQUARED = 9
ORDER_UPPER_SQUARED = 25
IAS15_RETAINED_EPSILON = 1.0e-13
IAS15_TIGHT_EPSILON = 1.0e-15
IAS15_INITIAL_DT = 1.0 / 64.0
IAS15_MIN_DT = 0.0
IAS15_ADAPTIVE_MODE = "PRS23"

EXACT_ROSTER = (
    "PROTOCOL.json", "SOURCE_NODES.bin", "ANNUAL_CHECKPOINTS.bin",
    "RUNTIME.json", "RESULT.json",
)
CLAIM_CEILING = (
    "SOURCE_ONLY_RESOLVED11_CONVERGENCE_DIAGNOSTIC_NO_STAGE15_QUALIFICATION_"
    "NO_CLASSIFICATION_NO_SCIENCE"
)
DISPOSITIONS = (
    "PASS", "TIMESTEP_LIMITED", "NON_TIMESTEP_REFERENCE_DISCREPANCY",
    "INCONCLUSIVE_AT_MAX_RESOLUTION",
)


class SourceConvergenceDiagnosticError(RuntimeError):
    """Fail-closed diagnostic contract violation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SourceConvergenceDiagnosticError("value is not canonical JSON") from exc


def _domain_sha(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise SourceConvergenceDiagnosticError(f"{label} is not an exact integer")
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        type(value) is not str or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceConvergenceDiagnosticError(f"{label} is not a SHA-256")
    return value


def _require_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise SourceConvergenceDiagnosticError(f"{label} key set differs")
    return value


def _regular_bytes(
    path: Path, label: str, *, exact_bytes: int | None = None,
    maximum: int = 200_000_000,
) -> bytes:
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceConvergenceDiagnosticError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or metadata.st_size < 0 or metadata.st_size > maximum
        or (exact_bytes is not None and metadata.st_size != exact_bytes)
    ):
        raise SourceConvergenceDiagnosticError(f"{label} file shape differs")
    with _stable_fd(path, label, exact_bytes=metadata.st_size) as fd:
        return _pread_all(fd, metadata.st_size, 0)


def _load_canonical_json(
    payload: bytes, label: str, *, trailing_newline: bool = False,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceConvergenceDiagnosticError(f"{label} is not canonical JSON") from exc
    if type(trailing_newline) is not bool:
        raise SourceConvergenceDiagnosticError("canonical JSON newline policy differs")
    expected = _canonical_json(value)
    if trailing_newline:
        expected += b"\n"
    if type(value) is not dict or expected != payload:
        raise SourceConvergenceDiagnosticError(f"{label} canonical bytes differ")
    return value


def _load_source_module(path: Path, expected_sha: str, name: str) -> types.ModuleType:
    payload = _regular_bytes(path, name, maximum=2_000_000)
    if _sha256(payload) != expected_sha:
        raise SourceConvergenceDiagnosticError(f"{name} source bytes differ")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(payload, str(path), "exec", optimize=sys.flags.optimize)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _file_receipt(path: Path, label: str, *, maximum: int = 200_000_000) -> dict[str, Any]:
    payload = _regular_bytes(path, label, maximum=maximum)
    return {
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(payload), "file_sha256": _sha256(payload),
    }


def _fraction_record(value: Fraction) -> dict[str, int]:
    if type(value) is not Fraction or value < 0:
        raise SourceConvergenceDiagnosticError("fraction record requires nonnegative Fraction")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_record(value: Any, label: str) -> Fraction:
    _require_keys(value, {"numerator", "denominator"}, label)
    numerator = _require_exact_int(value["numerator"], f"{label} numerator", minimum=0)
    denominator = _require_exact_int(value["denominator"], f"{label} denominator", minimum=1)
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise SourceConvergenceDiagnosticError(f"{label} is not canonical")
    return result


def _outward_sqrt(value: Fraction) -> float:
    if type(value) is not Fraction or value < 0:
        raise SourceConvergenceDiagnosticError("outward square root input differs")
    if value == 0:
        return 0.0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    half = exponent // 2
    scaled = value / Fraction(2) ** (2 * half)
    candidate = math.ldexp(math.sqrt(float(scaled)), half)
    if not math.isfinite(candidate):
        raise SourceConvergenceDiagnosticError("outward square root overflow")
    while Fraction.from_float(candidate) ** 2 < value:
        candidate = math.nextafter(candidate, math.inf)
    while candidate > 0.0:
        predecessor = math.nextafter(candidate, -math.inf)
        if Fraction.from_float(predecessor) ** 2 < value:
            break
        candidate = predecessor
    return candidate


def _fraction_observation(value: Fraction) -> dict[str, Any]:
    return {
        "exact": _fraction_record(value),
        "outward_sqrt_hex": _outward_sqrt(value).hex(),
    }


def _pread_all(fd: int, length: int, offset: int) -> bytes:
    _require_exact_int(length, "pread length", minimum=0)
    _require_exact_int(offset, "pread offset", minimum=0)
    chunks: list[bytes] = []
    remaining = length
    cursor = offset
    while remaining:
        try:
            payload = os.pread(fd, min(remaining, 16 * 1024 * 1024), cursor)
        except OSError as exc:
            raise SourceConvergenceDiagnosticError("held-fd read failed") from exc
        if not payload:
            raise SourceConvergenceDiagnosticError("held-fd read ended early")
        chunks.append(payload)
        cursor += len(payload)
        remaining -= len(payload)
    return b"".join(chunks)


def _fd_sha256(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < size:
        payload = _pread_all(fd, min(16 * 1024 * 1024, size - cursor), cursor)
        digest.update(payload)
        cursor += len(payload)
    return digest.hexdigest()


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SourceConvergenceDiagnosticError(
                f"path component is unavailable: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SourceConvergenceDiagnosticError(
                f"symlinked path component is forbidden: {current}"
            )


@contextmanager
def _stable_fd(
    path: Path, label: str, *, exact_bytes: int, expected_sha256: str | None = None,
) -> Iterator[int]:
    path = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SourceConvergenceDiagnosticError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(fd)
        identity = (
            before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
            before.st_size, before.st_mtime_ns, before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != exact_bytes
        ):
            raise SourceConvergenceDiagnosticError(f"{label} held-fd shape differs")
        if expected_sha256 is not None and _fd_sha256(fd, exact_bytes) != expected_sha256:
            raise SourceConvergenceDiagnosticError(f"{label} held-fd digest differs")
        yield fd
        after = os.fstat(fd)
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise SourceConvergenceDiagnosticError(
                f"{label} path changed during replay"
            ) from exc
        final = (
            after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
            after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        )
        path_final = (
            path_after.st_dev, path_after.st_ino, path_after.st_mode,
            path_after.st_nlink, path_after.st_size, path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            final != identity or path_final != identity
            or after.st_nlink != 1 or not stat.S_ISREG(after.st_mode)
        ):
            raise SourceConvergenceDiagnosticError(f"{label} changed during replay")
    finally:
        os.close(fd)


class _CappedWriter:
    def __init__(self, path: Path, cap: int) -> None:
        _require_exact_int(cap, "writer cap", minimum=1)
        self.path = path
        self.cap = cap
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self.fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise SourceConvergenceDiagnosticError(f"cannot create {path.name}") from exc
        self.count = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def write(self, payload: bytes) -> None:
        if self.closed or type(payload) is not bytes:
            raise SourceConvergenceDiagnosticError("binary writer state differs")
        if self.count + len(payload) > self.cap:
            raise SourceConvergenceDiagnosticError(f"{self.path.name} cap exceeded before write")
        view = memoryview(payload)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise SourceConvergenceDiagnosticError("binary write ended early")
            view = view[written:]
        self.count += len(payload)
        self.digest.update(payload)

    def finish(self) -> dict[str, Any]:
        if self.closed:
            raise SourceConvergenceDiagnosticError("binary writer already closed")
        os.fsync(self.fd)
        metadata = os.fstat(self.fd)
        if metadata.st_size != self.count or metadata.st_nlink != 1:
            raise SourceConvergenceDiagnosticError("binary writer final shape differs")
        os.close(self.fd)
        self.closed = True
        return {"bytes": self.count, "file_sha256": self.digest.hexdigest()}

    def abort_close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def _write_create_only(path: Path, payload: bytes) -> None:
    writer = _CappedWriter(path, max(1, len(payload)))
    try:
        writer.write(payload)
        receipt = writer.finish()
    except BaseException:
        writer.abort_close()
        raise
    if receipt != {"bytes": len(payload), "file_sha256": _sha256(payload)}:
        raise SourceConvergenceDiagnosticError(f"{path.name} write receipt differs")


def _v8_module() -> types.ModuleType:
    return _load_source_module(V8_SOURCE_PATH, V8_SOURCE_SHA256, "_jx_stage15_v8_bound_for_source_diagnostic")


def _validate_v8_failure_result() -> dict[str, Any]:
    payload = _regular_bytes(
        V8_RESULT_PATH, "v8 failed result", exact_bytes=V8_RESULT_BYTES,
        maximum=50_000_000,
    )
    if _sha256(payload) != V8_RESULT_SHA256:
        raise SourceConvergenceDiagnosticError("v8 failed result digest differs")
    result = _load_canonical_json(
        payload, "v8 failed result", trailing_newline=True,
    )
    claims = result.get("claims")
    gates = result.get("qualification_gates")
    rows = gates.get("gates") if type(gates) is dict else None
    if (
        result.get("schema") != "jx.tno.stage15-resolved11-encounter-qualification.result.v8"
        or result.get("content_sha256") != V8_RESULT_CONTENT_SHA256
        or result.get("status") != "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
        or type(claims) is not dict
        or claims.get("dynamics_executed") is not True
        or claims.get("qualification_established") is not False
        or claims.get("science_result") is not False
        or type(rows) is not dict or len(rows) != 20
        or type(gates.get("gate_count")) is not int or gates.get("gate_count") != 20
        or any(type(name) is not str or type(row) is not dict for name, row in rows.items())
        or sum(row.get("passed") is True for row in rows.values()) != 12
        or sum(row.get("passed") is False for row in rows.values()) != 8
        or gates.get("all_predeclared_gates_passed") is not False
    ):
        raise SourceConvergenceDiagnosticError("v8 failed-result status differs")
    return result


def _input_bindings() -> list[dict[str, Any]]:
    return [
        {"relative_path": V8_SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V8_SOURCE_BYTES, "file_sha256": V8_SOURCE_SHA256},
        {"relative_path": V8_TEST_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V8_TEST_BYTES, "file_sha256": V8_TEST_SHA256},
        {"relative_path": V8_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V8_PROTOCOL_BYTES, "file_sha256": V8_PROTOCOL_SHA256, "content_sha256": V8_PROTOCOL_CONTENT_SHA256},
        {"relative_path": V8_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V8_RESULT_BYTES, "file_sha256": V8_RESULT_SHA256, "content_sha256": V8_RESULT_CONTENT_SHA256},
        {"relative_path": V8_SOURCE_NODES_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V8_SOURCE_NODES_BYTES, "file_sha256": V8_SOURCE_NODES_SHA256},
        {"relative_path": V4_SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V4_SOURCE_BYTES, "file_sha256": V4_SOURCE_SHA256},
        {"relative_path": V4_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V4_PROTOCOL_BYTES, "file_sha256": V4_PROTOCOL_SHA256, "content_sha256": V4_PROTOCOL_CONTENT_SHA256},
        {"relative_path": V4_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V4_RESULT_BYTES, "file_sha256": V4_RESULT_SHA256, "content_sha256": V4_RESULT_CONTENT_SHA256},
        {"relative_path": V4_ARRAYS_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": V4_ARRAYS_BYTES, "file_sha256": V4_ARRAYS_SHA256},
        {"relative_path": BRIDGE_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": BRIDGE_BYTES, "file_sha256": BRIDGE_SHA256},
        {"relative_path": DE441_CUSTODY_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": DE441_CUSTODY_BYTES, "file_sha256": DE441_CUSTODY_SHA256},
        {"relative_path": STAGE15_V1_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": STAGE15_V1_PROTOCOL_BYTES, "file_sha256": STAGE15_V1_PROTOCOL_SHA256, "content_sha256": STAGE15_V1_PROTOCOL_CONTENT_SHA256},
        {"relative_path": REBOUND_WHEEL_PATH.relative_to(PROJECT_ROOT).as_posix(), "bytes": REBOUND_WHEEL_BYTES, "file_sha256": REBOUND_WHEEL_SHA256},
    ]


def build_protocol() -> dict[str, Any]:
    implementation = _file_receipt(SCRIPT_PATH, "diagnostic implementation", maximum=1_000_000)
    test = _file_receipt(TEST_PATH, "diagnostic test", maximum=500_000)
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": "TNO_STAGE15_RESOLVED11_SOURCE_CONVERGENCE_DIAGNOSTIC_ONLY",
        "claim_ceiling": CLAIM_CEILING,
        "status": "PREDECLARED_ONE_SHOT_DIAGNOSTIC_NOT_EXECUTED",
        "claims": {
            "dynamics_executed": False,
            "source_only_dynamics": True,
            "public_tracer_engine_called": False,
            "tracer_state_constructed": False,
            "stage15_qualification_established": False,
            "dynamical_classification_generated": False,
            "resonance_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
            "adaptive_rerun_authorized": False,
            "externally_preregistered": False,
            "protocol_predeclared_before_execution": True,
        },
        "implementation": {**implementation, "test": test},
        "predecessor_v8": {
            "git_custody": {
                "branch": V8_GIT_BRANCH,
                "preregistration_commit": V8_GIT_COMMIT,
                "annotated_tag": V8_GIT_TAG,
                "tag_object": V8_GIT_TAG_OBJECT,
                "tag_peeled_commit": V8_GIT_COMMIT,
                "external_git_custody_not_reconstructed_by_this_program": True,
            },
            "failed_result": {
                "status": "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS",
                "gate_count": 20, "passed_gate_count": 12,
                "failed_gate_count": 8,
                "dynamics_executed": True,
                "qualification_established": False,
                "science_result": False,
                "relative_path": V8_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": V8_RESULT_BYTES, "file_sha256": V8_RESULT_SHA256,
                "content_sha256": V8_RESULT_CONTENT_SHA256,
            },
        },
        "inputs": {
            "files": _input_bindings(),
            "v8_h64_h128_source_nodes_reused_exactly": True,
            "v4_de441_resolved11_ias15_annual_oracle_reused_exactly": True,
            "v4_result_status": "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS",
            "v4_all_predeclared_gates_passed": False,
            "retained_ias15_is_not_silently_treated_as_exact": True,
        },
        "scope": {
            "ephemeris": "DE441", "model": "RESOLVED_11_BODY",
            "native_epoch_batch_count": BATCH_COUNT,
            "object_ids_in_order": list(OBJECT_IDS),
            "massive_body_count": MASSIVE_COUNT,
            "massive_ids_in_order": list(MASSIVE_IDS),
            "massive_naif_ids_in_order": list(MASSIVE_NAIF_IDS),
            "horizon_julian_years": HORIZON_YEARS,
            "annual_checkpoint_count": CHECKPOINT_COUNT,
            "retained_steps_per_year": list(RETAINED_STEPS),
            "required_new_steps_per_year": list(BASE_STEPS),
            "conditional_adjudication_steps_per_year": ADJUDICATION_STEPS,
            "required_new_fixed_steps": TOTAL_BASE_FIXED_STEPS,
            "maximum_conditional_fixed_steps": TOTAL_H1024_MAXIMUM_STEPS,
            "maximum_total_new_fixed_steps": TOTAL_MAXIMUM_FIXED_STEPS,
            "required_jx_source_integration_count": BATCH_COUNT * len(BASE_STEPS),
            "maximum_conditional_h1024_integration_count": BATCH_COUNT,
            "required_tight_ias15_replicate_count": BATCH_COUNT,
            "required_tight_ias15_annual_integrate_calls": BATCH_COUNT * HORIZON_YEARS,
            "conditional_h1024_is_part_of_same_one_shot_execution": True,
            "no_tracers": True,
        },
        "physics": {
            "force_law": "NEWTONIAN_POINT_MASS",
            "frame": "ICRF_J2000_INERTIAL",
            "origin": "RESOLVED11_COMMON_GM_BARYCENTER_AT_NATIVE_EPOCH_WITH_INSTANTANEOUS_GM_CENTERING_FOR_COMPARISONS",
            "time_scale_token": "TDB",
            "time_unit": "JULIAN_YEAR_EXACTLY_365_25_TDB_DAYS",
            "epoch_origin": "NATIVE_COVARIANCE_EPOCH_JD_TDB_MINUS_2451545_DIVIDED_BY_365_25",
            "gm_policy": "EXACT_COMMON_STAGE1_V4_BINARY64_GMS",
            "ode_identity_required_across_jx_steps": True,
            "annual_grid_identity_required": True,
        },
        "ias15_reference": {
            "retained_epsilon_hex": IAS15_RETAINED_EPSILON.hex(),
            "tight_replicate_epsilon_hex": IAS15_TIGHT_EPSILON.hex(),
            "tight_initial_dt_hex": IAS15_INITIAL_DT.hex(),
            "tight_min_dt_hex": IAS15_MIN_DT.hex(),
            "adaptive_mode": IAS15_ADAPTIVE_MODE,
            "engine": "REBOUND_5.1.1_IAS15",
            "G_hex": 1.0.hex(), "gravity": "basic",
            "collision": "none", "boundary": "none",
            "softening_hex": 0.0.hex(),
            "source_only": True,
            "all_1001_cumulative_integrator_step_counts_retained": True,
            "reference_qualification_position_budget_exact": "1/5000_AU",
            "reference_qualification_velocity_budget_exact": "1/500000_AU_PER_JULIAN_YEAR",
            "reference_qualification_requires_every_operational_sun_relative_and_pairwise_series": True,
            "failure_is_explicit_and_forces_inconclusive_disposition": True,
        },
        "representations": {
            "operational_barycentric": "EACH_OF_11_SOURCES_AFTER_INSTANTANEOUS_COMMON_GM_CENTERING",
            "sun_relative": "TEN_NON_SUN_SOURCE_MINUS_SUN_VECTORS",
            "pairwise_separation_rate": "ALL_55_DIRECTED_LOW_INDEX_TO_HIGH_INDEX_RELATIVE_POSITION_AND_VELOCITY_VECTORS",
            "metric": "MAX_OVER_1001_YEARS_OF_EXACT_BINARY64_COMPONENT_DIFFERENCE_SQUARED_NORM",
        },
        "gates": {
            "order_window": "FINE_AT_OR_BELOW_EXACT_64_ULP_SQUARED_FLOOR_OR_9_FINE_SQUARED_LE_COARSE_SQUARED_LE_25_FINE_SQUARED",
            "h1024_trigger": "FOR_EACH_EPOCH_TRIGGER_IF_REFERENCE_NOT_QUALIFIED_OR_EITHER_OF_THE_TWO_H64_H128_H256_AND_H128_H256_H512_ORDER_WINDOWS_FAILS_OR_XSTAR256_XSTAR512_EXCEEDS_V8_BUDGET_OR_H512_TIGHT_IAS15_EXCEEDS_V8_BUDGET",
            "richardson": "XSTAR_2N_EQUALS_X_2N_PLUS_EXACT_RATIONAL_(X_2N_MINUS_X_N)_DIVIDED_BY_3",
            "position_budget_exact": "1/500_AU",
            "velocity_budget_exact": "1/50000_AU_PER_JULIAN_YEAR",
            "all_node_energy_exact": "1/100000000",
            "all_node_angular_momentum_exact": "1/10000000000",
            "adjacent_energy_contraction": "FINE_LE_MAX(COARSE_DIVIDED_BY_3,1/2^46)",
            "all_available_h64_h128_h256_h512_AND_CONDITIONAL_H1024_absolute_invariants_are_retained_as_diagnostics": True,
            "all_adjacent_energy_contractions_are_retained_as_diagnostics": True,
            "diagnostic_invariant_failures_at_nondecisive_coarser_levels_do_not_veto_a_disposition": True,
            "decisive_finest_absolute_invariants_required_for_pass_or_timestep_limited_and_any_step_estimate": "H512_WITHOUT_H1024_TRIGGER_ELSE_H1024",
            "step_estimate_allowed_only_after_reference_qualification_two_consecutive_order_windows_relevant_final_richardson_and_extrapolate_gates_decisive_finest_absolute_invariants_and_pass_or_timestep_limited_disposition": True,
            "dispositions": list(DISPOSITIONS),
            "disposition_rules": {
                "PASS": "WITHOUT_H1024_TRIGGER_REFERENCE_QUALIFIED_TWO_ORDER_WINDOWS_XSTAR256_XSTAR512_AND_RAW_H512_REFERENCE_GATES_PASS_AND_H512_ABSOLUTE_GATES_PASS;WITH_H1024_TRIGGER_REFERENCE_QUALIFIED_TWO_FINAL_ORDER_WINDOWS_XSTAR512_XSTAR1024_RAW_H1024_AND_EXTRAPOLATED_H1024_REFERENCE_GATES_PASS_AND_H1024_ABSOLUTE_GATES_PASS",
                "TIMESTEP_LIMITED": "H1024_TRIGGERED_AND_REFERENCE_QUALIFIED_TWO_FINAL_ORDER_WINDOWS_FINAL_RICHARDSON_EXTRAPOLATED_REFERENCE_AND_H1024_ABSOLUTE_INVARIANT_GATES_PASS_BUT_RAW_H1024_REFERENCE_GATE_FAILS",
                "NON_TIMESTEP_REFERENCE_DISCREPANCY": "H1024_TRIGGERED_AND_REFERENCE_QUALIFIED_TWO_FINAL_ORDER_WINDOWS_FINAL_RICHARDSON_AND_H1024_ABSOLUTE_INVARIANT_GATES_PASS_BUT_EXTRAPOLATED_H1024_REFERENCE_GATE_FAILS",
                "INCONCLUSIVE_AT_MAX_RESOLUTION": "ANY_OTHER_CONDITION_INCLUDING_ANY_DECISIVE_FINEST_ABSOLUTE_INVARIANT_FAILURE_UNQUALIFIED_IAS15_REFERENCE_FAILED_ORDER_OR_FAILED_FINAL_RICHARDSON;FINEST_INVARIANT_FAILURE_SUPPRESSES_ANY_COORDINATE_ONLY_STEP_ESTIMATE",
            },
            "pass_does_not_establish_stage15_qualification": True,
        },
        "diagnostics": {
            "all_source_nodes_retained": True,
            "energy_and_angular_momentum_rederived_from_every_node": True,
            "linear_momentum_and_barycenter_drift_retained": True,
            "linear_momentum_units": "GM_TIMES_AU_PER_JULIAN_YEAR",
            "barycenter_diagnostic_units": ["AU", "AU_PER_JULIAN_YEAR"],
            "h1024_not_run_for_resolved_epochs": True,
            "one_shot_no_adaptive_followup": True,
        },
        "evidence": {
            "exact_roster": list(EXACT_ROSTER),
            "create_only": True, "result_created_last": True,
            "publication_staging": "ABSENT_TEMPORARY_SIBLING_ON_OUTPUT_PARENT_FILESYSTEM",
            "multi_file_atomicity_claimed": False,
            "partial_failure": "NO_RESULT_MEANS_INVALID_PARTIAL_ROSTER_REQUIRING_MANUAL_CLEANUP",
            "post_result_creation_failure": "A_COMPLETE_ROSTER_MAY_REMAIN_BUT_NO_CALLER_SUCCESS_IS_CLAIMED_WITHOUT_EXPLICIT_OFFLINE_VALIDATION",
            "held_fd_validation": "ONE_O_NOFOLLOW_FD_HELD_ACROSS_GLOBAL_HASH_ALL_SEMANTIC_PREADS_AND_FINAL_FSTAT",
            "source_order": "BATCH_THEN_H256_H512_THEN_H1024_IF_TRIGGERED_THEN_NODE_THEN_POSITION_VELOCITY",
            "annual_order": "BATCH_THEN_H256_H512_TIGHT_IAS15_THEN_H1024_IF_TRIGGERED_THEN_YEAR_THEN_POSITION_VELOCITY",
            "initial_state_crossbinding": "EVERY_RETAINED_AND_NEW_JX_SOURCE_PATH_FIRST_RECORD_BITWISE_EQUALS_THE_SEALED_BATCH_MASSIVE_STATE;TIGHT_IAS15_YEAR_ZERO_EQUALS_THE_SAME_STATE_AFTER_FROZEN_INSTANTANEOUS_GM_CENTERING",
            "source_base_exact_bytes": SOURCE_BASE_EXACT_BYTES,
            "source_h1024_maximum_bytes": SOURCE_H1024_MAXIMUM_BYTES,
            "source_maximum_bytes": SOURCE_MAXIMUM_BYTES,
            "annual_base_exact_bytes": ANNUAL_BASE_EXACT_BYTES,
            "annual_h1024_maximum_bytes": ANNUAL_H1024_MAXIMUM_BYTES,
            "annual_maximum_bytes": ANNUAL_MAXIMUM_BYTES,
            "runtime_maximum_bytes": 10_000_000,
            "result_maximum_bytes": 100_000_000,
        },
        "execution": {
            "fresh_project_venv_child": True,
            "source_kernel_only": True,
            "public_tracer_entrypoint_forbidden": "advance_cuda_massless_tracers",
            "conditional_h1024_is_mechanical_not_operator_selected": True,
            "successor_or_rerun_is_not_authorized_by_any_disposition": True,
        },
        "nonclaim": "NOT_A_STAGE15_QUALIFICATION_CLASSIFICATION_RESONANCE_POPULATION_PLANET_NINE_OR_SCIENCE_RESULT",
    }
    protocol["content_sha256"] = _domain_sha(PROTOCOL_DOMAIN, protocol)
    return protocol


def protocol_bytes() -> bytes:
    return _canonical_json(build_protocol())


def prepare(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    if root.exists() or root.is_symlink():
        raise SourceConvergenceDiagnosticError("prepare root must be absent")
    root.mkdir(mode=0o700)
    _write_create_only(root / "PROTOCOL.json", protocol_bytes())
    return validate_prepared(root)


def validate_prepared(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    if not root.is_dir() or root.is_symlink():
        raise SourceConvergenceDiagnosticError("prepared root is not a real directory")
    if sorted(item.name for item in root.iterdir()) != ["PROTOCOL.json"]:
        raise SourceConvergenceDiagnosticError("prepared root roster differs")
    payload = _regular_bytes(root / "PROTOCOL.json", "prepared protocol", maximum=2_000_000)
    expected = protocol_bytes()
    if payload != expected:
        raise SourceConvergenceDiagnosticError("prepared protocol bytes differ")
    return _load_canonical_json(payload, "prepared protocol")


def _exact_squared_vector_difference(
    left: Sequence[float], right: Sequence[float],
) -> Fraction:
    if (
        type(left) not in (tuple, list) or type(right) not in (tuple, list)
        or len(left) != 3 or len(right) != 3
    ):
        raise SourceConvergenceDiagnosticError("exact vector comparison shape differs")
    result = Fraction(0, 1)
    for left_value, right_value in zip(left, right, strict=True):
        if (
            type(left_value) is not float or type(right_value) is not float
            or not math.isfinite(left_value) or not math.isfinite(right_value)
        ):
            raise SourceConvergenceDiagnosticError("exact vector comparison is nonfinite")
        difference = Fraction.from_float(left_value) - Fraction.from_float(right_value)
        result += difference * difference
    return result


def _exact_squared_ulp_floor(reference: Sequence[float]) -> Fraction:
    if type(reference) not in (tuple, list) or len(reference) != 3:
        raise SourceConvergenceDiagnosticError("ULP floor vector shape differs")
    result = Fraction(0, 1)
    for component in reference:
        if type(component) is not float or not math.isfinite(component):
            raise SourceConvergenceDiagnosticError("ULP floor vector is nonfinite")
        term = Fraction.from_float(64.0 * math.ulp(component))
        result += term * term
    return result


def _center_annual(np: types.ModuleType, raw: Any, gm_values: Sequence[float]) -> Any:
    array = np.asarray(raw, dtype=np.float64)
    gm = np.asarray(tuple(gm_values), dtype=np.float64)
    if (
        array.ndim != 4 or array.shape[1:] != (2, MASSIVE_COUNT, 3)
        or gm.shape != (MASSIVE_COUNT,)
        or not bool(np.isfinite(array).all()) or not bool(np.isfinite(gm).all())
        or not bool(np.all(gm > 0.0))
    ):
        raise SourceConvergenceDiagnosticError("annual source state shape differs")
    total = float(np.sum(gm, dtype=np.float64))
    positions = array[:, 0]
    velocities = array[:, 1]
    center_position = np.sum(positions * gm[None, :, None], axis=1) / total
    center_velocity = np.sum(velocities * gm[None, :, None], axis=1) / total
    centered = np.empty_like(array)
    centered[:, 0] = positions - center_position[:, None, :]
    centered[:, 1] = velocities - center_velocity[:, None, :]
    if not bool(np.isfinite(centered).all()):
        raise SourceConvergenceDiagnosticError("centered annual state is nonfinite")
    return centered


def _annual_payload(np: types.ModuleType, centered: Any) -> bytes:
    array = np.ascontiguousarray(centered, dtype="<f8")
    if array.ndim != 4 or array.shape[1:] != (2, MASSIVE_COUNT, 3):
        raise SourceConvergenceDiagnosticError("annual payload shape differs")
    return array.tobytes(order="C")


def _representation_series(annual: Any) -> dict[str, dict[str, tuple[tuple[float, float, float], ...]]]:
    shape = getattr(annual, "shape", None)
    if type(shape) is not tuple or len(shape) != 4 or shape[1:] != (2, MASSIVE_COUNT, 3):
        raise SourceConvergenceDiagnosticError("comparison annual shape differs")
    result: dict[str, dict[str, tuple[tuple[float, float, float], ...]]] = {}

    def vector(year: int, kind: int, source: int) -> tuple[float, float, float]:
        row = tuple(float(annual[year, kind, source, component]) for component in range(3))
        if any(not math.isfinite(value) for value in row):
            raise SourceConvergenceDiagnosticError("comparison state is nonfinite")
        return row

    def relative(
        year: int, kind: int, left: int, right: int,
    ) -> tuple[float, float, float]:
        left_vector = vector(year, kind, left)
        right_vector = vector(year, kind, right)
        return tuple(
            float(right_vector[component] - left_vector[component])
            for component in range(3)
        )

    years = shape[0]
    for source, source_id in enumerate(MASSIVE_IDS):
        result[f"BARYCENTRIC:{source_id}"] = {
            "position": tuple(vector(year, 0, source) for year in range(years)),
            "velocity": tuple(vector(year, 1, source) for year in range(years)),
        }
    for source in range(1, MASSIVE_COUNT):
        source_id = MASSIVE_IDS[source]
        result[f"SUN_RELATIVE:{source_id}"] = {
            "position": tuple(relative(year, 0, 0, source) for year in range(years)),
            "velocity": tuple(relative(year, 1, 0, source) for year in range(years)),
        }
    for left in range(MASSIVE_COUNT):
        for right in range(left + 1, MASSIVE_COUNT):
            key = f"PAIRWISE:{MASSIVE_IDS[left]}->{MASSIVE_IDS[right]}"
            result[key] = {
                "position": tuple(relative(year, 0, left, right) for year in range(years)),
                "velocity": tuple(relative(year, 1, left, right) for year in range(years)),
            }
    if len(result) != 76:
        raise SourceConvergenceDiagnosticError("representation roster differs")
    return result


def _series_comparison(left: Any, right: Any) -> list[dict[str, Any]]:
    left_series = _representation_series(left)
    right_series = _representation_series(right)
    if tuple(left_series) != tuple(right_series):
        raise SourceConvergenceDiagnosticError("comparison representation roster differs")
    records: list[dict[str, Any]] = []
    for key in left_series:
        representation, subject = key.split(":", 1)
        row: dict[str, Any] = {"representation": representation, "subject": subject}
        for component in ("position", "velocity"):
            left_values = left_series[key][component]
            right_values = right_series[key][component]
            if len(left_values) != len(right_values) or not left_values:
                raise SourceConvergenceDiagnosticError("comparison series length differs")
            squared = max(
                _exact_squared_vector_difference(a, b)
                for a, b in zip(left_values, right_values, strict=True)
            )
            floor = max(_exact_squared_ulp_floor(value) for value in right_values)
            row[f"{component}_squared"] = _fraction_observation(squared)
            row[f"{component}_reference_64ulp_floor_squared"] = _fraction_record(floor)
        records.append(row)
    return records


def _comparison_index(records: Any, label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    if type(records) is not list or len(records) != 76:
        raise SourceConvergenceDiagnosticError(f"{label} comparison roster differs")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in records:
        if type(row) is not dict:
            raise SourceConvergenceDiagnosticError(f"{label} comparison row differs")
        representation = row.get("representation")
        subject = row.get("subject")
        if type(representation) is not str or type(subject) is not str:
            raise SourceConvergenceDiagnosticError(f"{label} comparison identity differs")
        key = (representation, subject)
        if key in result:
            raise SourceConvergenceDiagnosticError(f"{label} comparison duplicate")
        for component in ("position", "velocity"):
            observation = row.get(f"{component}_squared")
            _require_keys(observation, {"exact", "outward_sqrt_hex"}, f"{label} observation")
            exact = _fraction_from_record(observation["exact"], f"{label} exact observation")
            try:
                display = float.fromhex(observation["outward_sqrt_hex"])
            except (TypeError, ValueError) as exc:
                raise SourceConvergenceDiagnosticError(f"{label} display differs") from exc
            if display != _outward_sqrt(exact):
                raise SourceConvergenceDiagnosticError(f"{label} display is not outward exact")
            _fraction_from_record(
                row.get(f"{component}_reference_64ulp_floor_squared"),
                f"{label} ULP floor",
            )
        result[key] = row
    if len(result) != 76:
        raise SourceConvergenceDiagnosticError(f"{label} comparison count differs")
    return result


def _order_window(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]], label: str,
) -> dict[str, Any]:
    coarse_index = _comparison_index(coarse, f"{label} coarse")
    fine_index = _comparison_index(fine, f"{label} fine")
    rows: list[dict[str, Any]] = []
    for key in coarse_index:
        coarse_row = coarse_index[key]
        fine_row = fine_index[key]
        for component in ("position", "velocity"):
            coarse_value = _fraction_from_record(
                coarse_row[f"{component}_squared"]["exact"], "coarse error"
            )
            fine_value = _fraction_from_record(
                fine_row[f"{component}_squared"]["exact"], "fine error"
            )
            floor = _fraction_from_record(
                fine_row[f"{component}_reference_64ulp_floor_squared"],
                "fine ULP floor",
            )
            saturated = fine_value <= floor
            within = (
                saturated
                or ORDER_LOWER_SQUARED * fine_value <= coarse_value
                <= ORDER_UPPER_SQUARED * fine_value
            )
            rows.append({
                "representation": key[0], "subject": key[1],
                "component": component,
                "coarse_squared": _fraction_record(coarse_value),
                "fine_squared": _fraction_record(fine_value),
                "fine_64ulp_floor_squared": _fraction_record(floor),
                "saturated": saturated, "passed": within,
            })
    return {
        "label": label, "predicate_count": len(rows),
        "all_passed": all(row["passed"] is True for row in rows),
        "predicates": rows,
    }


def _richardson_comparison(
    coarse_left: Any, fine_left: Any, coarse_right: Any, fine_right: Any,
) -> list[dict[str, Any]]:
    a = _representation_series(coarse_left)
    b = _representation_series(fine_left)
    c = _representation_series(coarse_right)
    d = _representation_series(fine_right)
    if not (tuple(a) == tuple(b) == tuple(c) == tuple(d)):
        raise SourceConvergenceDiagnosticError("Richardson representation roster differs")
    records: list[dict[str, Any]] = []
    for key in a:
        representation, subject = key.split(":", 1)
        row: dict[str, Any] = {"representation": representation, "subject": subject}
        for component in ("position", "velocity"):
            maxima = Fraction(0, 1)
            series = (a[key][component], b[key][component], c[key][component], d[key][component])
            if len({len(item) for item in series}) != 1:
                raise SourceConvergenceDiagnosticError("Richardson series length differs")
            for values in zip(*series, strict=True):
                squared = Fraction(0, 1)
                for axis in range(3):
                    left_star = (
                        4 * Fraction.from_float(values[1][axis])
                        - Fraction.from_float(values[0][axis])
                    ) / 3
                    right_star = (
                        4 * Fraction.from_float(values[3][axis])
                        - Fraction.from_float(values[2][axis])
                    ) / 3
                    difference = left_star - right_star
                    squared += difference * difference
                maxima = max(maxima, squared)
            row[f"{component}_squared"] = _fraction_observation(maxima)
        records.append(row)
    return records


def _richardson_reference_comparison(
    coarse: Any, fine: Any, reference: Any,
) -> list[dict[str, Any]]:
    a = _representation_series(coarse)
    b = _representation_series(fine)
    c = _representation_series(reference)
    if not (tuple(a) == tuple(b) == tuple(c)):
        raise SourceConvergenceDiagnosticError("Richardson reference roster differs")
    records: list[dict[str, Any]] = []
    for key in a:
        representation, subject = key.split(":", 1)
        row: dict[str, Any] = {"representation": representation, "subject": subject}
        for component in ("position", "velocity"):
            maxima = Fraction(0, 1)
            for coarse_value, fine_value, reference_value in zip(
                a[key][component], b[key][component], c[key][component], strict=True,
            ):
                squared = Fraction(0, 1)
                for axis in range(3):
                    extrapolated = (
                        4 * Fraction.from_float(fine_value[axis])
                        - Fraction.from_float(coarse_value[axis])
                    ) / 3
                    difference = extrapolated - Fraction.from_float(reference_value[axis])
                    squared += difference * difference
                maxima = max(maxima, squared)
            row[f"{component}_squared"] = _fraction_observation(maxima)
        records.append(row)
    return records


def _comparison_within_budgets(
    records: list[dict[str, Any]], position_budget: Fraction,
    velocity_budget: Fraction,
) -> bool:
    index = _comparison_index(records, "budget")
    for row in index.values():
        position = _fraction_from_record(row["position_squared"]["exact"], "position budget")
        velocity = _fraction_from_record(row["velocity_squared"]["exact"], "velocity budget")
        if position > position_budget * position_budget:
            return False
        if velocity > velocity_budget * velocity_budget:
            return False
    return True


def _richardson_within_budgets(
    records: list[dict[str, Any]], position_budget: Fraction,
    velocity_budget: Fraction,
) -> bool:
    if type(records) is not list or len(records) != 76:
        raise SourceConvergenceDiagnosticError("Richardson budget roster differs")
    for row in records:
        for component, budget in (
            ("position", position_budget), ("velocity", velocity_budget),
        ):
            observation = row.get(f"{component}_squared")
            _require_keys(observation, {"exact", "outward_sqrt_hex"}, "Richardson observation")
            value = _fraction_from_record(observation["exact"], "Richardson budget")
            if value > budget * budget:
                return False
    return True


def _log2_fraction(value: Fraction) -> float:
    if type(value) is not Fraction or value <= 0:
        raise SourceConvergenceDiagnosticError("log2 fraction must be positive")
    numerator = value.numerator
    denominator = value.denominator
    shift = numerator.bit_length() - denominator.bit_length()
    scaled = value / Fraction(2) ** shift
    return float(shift + math.log2(float(scaled)))


def _observed_order(coarse_squared: Fraction, fine_squared: Fraction) -> float | None:
    if coarse_squared <= 0 or fine_squared <= 0:
        return None
    return 0.5 * _log2_fraction(coarse_squared / fine_squared)


def _estimated_power_of_two_steps(
    *, current_steps: int, error_squared: Fraction, budget: Fraction,
    observed_order: float, two_consecutive_windows_passed: bool,
) -> int | None:
    _require_exact_int(current_steps, "current steps", minimum=1)
    if not two_consecutive_windows_passed:
        return None
    if type(observed_order) is not float or not math.isfinite(observed_order) or observed_order <= 0.0:
        return None
    if type(error_squared) is not Fraction or error_squared < 0 or budget <= 0:
        raise SourceConvergenceDiagnosticError("step estimate operands differ")
    if error_squared <= budget * budget:
        return current_steps
    ratio = _outward_sqrt(error_squared) / float(budget)
    required = current_steps * ratio ** (1.0 / observed_order)
    if not math.isfinite(required) or required > 2**30:
        return 2**30
    result = 1
    while result < math.ceil(required):
        result *= 2
    return max(current_steps, result)


def _source_invariant_summary(
    fd: int, locator: Mapping[str, Any], gm_values: Sequence[float],
    np: types.ModuleType,
) -> dict[str, Any]:
    offset = _require_exact_int(locator.get("byte_offset"), "source offset", minimum=0)
    length = _require_exact_int(locator.get("byte_length"), "source length", minimum=1)
    node_count = _require_exact_int(locator.get("node_count"), "source node count", minimum=2)
    if length != node_count * SOURCE_NODE_RECORD_BYTES:
        raise SourceConvergenceDiagnosticError("source locator length differs")
    gm = np.asarray(tuple(gm_values), dtype=np.float64)
    if (
        gm.shape != (MASSIVE_COUNT,) or not bool(np.isfinite(gm).all())
        or not bool(np.all(gm > 0.0))
    ):
        raise SourceConvergenceDiagnosticError("source invariant GM roster differs")
    total_gm = float(np.sum(gm, dtype=np.float64))
    initial_energy: float | None = None
    minimum_energy = math.inf
    maximum_energy = -math.inf
    initial_angular: tuple[float, float, float] | None = None
    initial_momentum: tuple[float, float, float] | None = None
    initial_center_position: tuple[float, float, float] | None = None
    initial_center_velocity: tuple[float, float, float] | None = None
    maximum_angular_relative = 0.0
    maximum_momentum_change = 0.0
    maximum_center_position_change = 0.0
    maximum_center_velocity_change = 0.0
    cursor = 0
    while cursor < node_count:
        count = min(SOURCE_CHUNK_NODES, node_count - cursor)
        payload = _pread_all(
            fd, count * SOURCE_NODE_RECORD_BYTES,
            offset + cursor * SOURCE_NODE_RECORD_BYTES,
        )
        rows = np.frombuffer(payload, dtype="<f8").reshape(count, 2, MASSIVE_COUNT, 3)
        if not bool(np.isfinite(rows).all()):
            raise SourceConvergenceDiagnosticError("source evidence contains nonfinite state")
        position = rows[:, 0]
        velocity = rows[:, 1]
        center_position = np.sum(position * gm[None, :, None], axis=1) / total_gm
        center_velocity = np.sum(velocity * gm[None, :, None], axis=1) / total_gm
        centered_position = position - center_position[:, None, :]
        centered_velocity = velocity - center_velocity[:, None, :]
        kinetic = np.sum(
            0.5 * gm[None, :] * np.sum(centered_velocity * centered_velocity, axis=2),
            axis=1,
        )
        potential = np.zeros(count, dtype=np.float64)
        for left in range(MASSIVE_COUNT):
            for right in range(left + 1, MASSIVE_COUNT):
                delta = centered_position[:, left] - centered_position[:, right]
                distance = np.sqrt(np.sum(delta * delta, axis=1))
                if not bool(np.isfinite(distance).all()) or bool(np.any(distance <= 0.0)):
                    raise SourceConvergenceDiagnosticError("source evidence contains coincident bodies")
                potential -= gm[left] * gm[right] / distance
        energy = kinetic + potential
        angular = np.sum(
            gm[None, :, None] * np.cross(centered_position, centered_velocity, axis=2),
            axis=1,
        )
        momentum = np.sum(gm[None, :, None] * velocity, axis=1)
        if not (
            bool(np.isfinite(energy).all())
            and bool(np.isfinite(angular).all())
            and bool(np.isfinite(momentum).all())
            and bool(np.isfinite(center_position).all())
            and bool(np.isfinite(center_velocity).all())
        ):
            raise SourceConvergenceDiagnosticError(
                "source invariant diagnostic is nonfinite"
            )
        if cursor == 0:
            initial_energy = float(energy[0])
            initial_angular = tuple(float(value) for value in angular[0])
            initial_momentum = tuple(float(value) for value in momentum[0])
            initial_center_position = tuple(float(value) for value in center_position[0])
            initial_center_velocity = tuple(float(value) for value in center_velocity[0])
        if (
            initial_angular is None or initial_momentum is None
            or initial_center_position is None or initial_center_velocity is None
        ):
            raise SourceConvergenceDiagnosticError("source invariant initial state is absent")
        minimum_energy = min(minimum_energy, float(np.min(energy)))
        maximum_energy = max(maximum_energy, float(np.max(energy)))
        angular_norm = math.hypot(*initial_angular)
        if not angular_norm > 0.0:
            raise SourceConvergenceDiagnosticError("source angular denominator is zero")
        angular_difference = angular - np.asarray(initial_angular, dtype=np.float64)[None, :]
        momentum_difference = momentum - np.asarray(initial_momentum, dtype=np.float64)[None, :]
        center_position_difference = (
            center_position - np.asarray(initial_center_position, dtype=np.float64)[None, :]
        )
        center_velocity_difference = (
            center_velocity - np.asarray(initial_center_velocity, dtype=np.float64)[None, :]
        )

        def conservative_maximum_norm(values: Any) -> float:
            norms = np.sqrt(np.sum(values * values, axis=1))
            if not bool(np.isfinite(norms).all()):
                raise SourceConvergenceDiagnosticError("source diagnostic norm is nonfinite")
            return math.nextafter(float(np.max(norms)), math.inf)

        angular_delta = conservative_maximum_norm(angular_difference)
        angular_relative = math.nextafter(
            angular_delta / math.nextafter(angular_norm, -math.inf), math.inf,
        )
        maximum_angular_relative = max(maximum_angular_relative, angular_relative)
        maximum_momentum_change = max(
            maximum_momentum_change,
            conservative_maximum_norm(momentum_difference),
        )
        maximum_center_position_change = max(
            maximum_center_position_change,
            conservative_maximum_norm(center_position_difference),
        )
        maximum_center_velocity_change = max(
            maximum_center_velocity_change,
            conservative_maximum_norm(center_velocity_difference),
        )
        cursor += count
    if initial_energy is None or initial_energy == 0.0:
        raise SourceConvergenceDiagnosticError("source energy denominator is zero")
    energy_difference = max(
        abs(Fraction.from_float(minimum_energy) - Fraction.from_float(initial_energy)),
        abs(Fraction.from_float(maximum_energy) - Fraction.from_float(initial_energy)),
    )
    energy_fraction = energy_difference / abs(Fraction.from_float(initial_energy))
    return {
        "sample_count": node_count,
        "maximum_relative_energy_drift": _fraction_observation(energy_fraction),
        "maximum_relative_angular_momentum_drift_hex": maximum_angular_relative.hex(),
        "maximum_absolute_linear_momentum_change_hex": maximum_momentum_change.hex(),
        "maximum_barycenter_position_change_AU_hex": maximum_center_position_change.hex(),
        "maximum_barycenter_velocity_change_AU_per_julian_year_hex": maximum_center_velocity_change.hex(),
        "sampling": "EVERY_RETAINED_FIXED_SOURCE_NODE",
    }


def _decode_annual_locator(
    fd: int, locator: Mapping[str, Any], np: types.ModuleType,
) -> Any:
    keys = {
        "schema", "byte_offset", "byte_length", "record_bytes",
        "checkpoint_count", "lane_payload_sha256", "lane_id",
        "steps_per_julian_year", "kind",
    }
    _require_keys(locator, keys, "annual locator")
    offset = _require_exact_int(locator["byte_offset"], "annual offset", minimum=0)
    length = _require_exact_int(locator["byte_length"], "annual length", minimum=1)
    if (
        locator["schema"] != ANNUAL_SCHEMA
        or locator["record_bytes"] != ANNUAL_RECORD_BYTES
        or locator["checkpoint_count"] != CHECKPOINT_COUNT
        or length != CHECKPOINT_COUNT * ANNUAL_RECORD_BYTES
    ):
        raise SourceConvergenceDiagnosticError("annual locator shape differs")
    payload = _pread_all(fd, length, offset)
    if _sha256(payload) != locator["lane_payload_sha256"]:
        raise SourceConvergenceDiagnosticError("annual lane digest differs")
    values = np.frombuffer(payload, dtype="<f8").reshape(
        CHECKPOINT_COUNT, 2, MASSIVE_COUNT, 3,
    )
    if not bool(np.isfinite(values).all()):
        raise SourceConvergenceDiagnosticError("annual evidence is nonfinite")
    return values.copy()


def _center_single_node(
    np: types.ModuleType, raw_payload: bytes, gm_values: Sequence[float],
) -> bytes:
    if len(raw_payload) != SOURCE_NODE_RECORD_BYTES:
        raise SourceConvergenceDiagnosticError("source node record length differs")
    raw = np.frombuffer(raw_payload, dtype="<f8").reshape(1, 2, MASSIVE_COUNT, 3)
    return _annual_payload(np, _center_annual(np, raw, gm_values))


def _batch_initial_source_payload(np: types.ModuleType, batch: Any) -> bytes:
    massive = getattr(batch, "massive", None)
    if type(massive) is not tuple or len(massive) != MASSIVE_COUNT:
        raise SourceConvergenceDiagnosticError("batch initial source roster differs")
    raw = np.empty((1, 2, MASSIVE_COUNT, 3), dtype="<f8")
    try:
        raw[0, 0] = np.asarray(
            [item.position_AU for item in massive], dtype=np.float64,
        )
        raw[0, 1] = np.asarray(
            [item.velocity_AU_per_julian_year for item in massive],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SourceConvergenceDiagnosticError(
            "batch initial source state differs"
        ) from exc
    if not bool(np.isfinite(raw).all()):
        raise SourceConvergenceDiagnosticError("batch initial source state is nonfinite")
    return raw.tobytes(order="C")


def _require_initial_source_payload(
    fd: int, locator: Mapping[str, Any], expected: bytes, label: str,
) -> None:
    if type(expected) is not bytes or len(expected) != SOURCE_NODE_RECORD_BYTES:
        raise SourceConvergenceDiagnosticError("expected initial source payload differs")
    offset = _require_exact_int(locator.get("byte_offset"), f"{label} offset", minimum=0)
    if _pread_all(fd, SOURCE_NODE_RECORD_BYTES, offset) != expected:
        raise SourceConvergenceDiagnosticError(f"{label} initial source state differs")


def _crossbind_source_annual(
    *, source_fd: int, source_locator: Mapping[str, Any], annual_fd: int,
    annual_locator: Mapping[str, Any], gm_values: Sequence[float],
    np: types.ModuleType,
) -> None:
    source_offset = _require_exact_int(source_locator.get("byte_offset"), "source offset", minimum=0)
    node_count = _require_exact_int(source_locator.get("node_count"), "source nodes", minimum=2)
    steps = _require_exact_int(annual_locator.get("steps_per_julian_year"), "annual steps", minimum=1)
    if node_count != HORIZON_YEARS * steps + 1:
        raise SourceConvergenceDiagnosticError("source/annual node count differs")
    annual_offset = _require_exact_int(annual_locator.get("byte_offset"), "annual offset", minimum=0)
    for year in range(CHECKPOINT_COUNT):
        source_payload = _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            source_offset + year * steps * SOURCE_NODE_RECORD_BYTES,
        )
        expected = _center_single_node(np, source_payload, gm_values)
        observed = _pread_all(
            annual_fd, ANNUAL_RECORD_BYTES,
            annual_offset + year * ANNUAL_RECORD_BYTES,
        )
        if observed != expected:
            raise SourceConvergenceDiagnosticError("annual checkpoint differs from source node")


def _extract_v8_annual(
    fd: int, locator: Mapping[str, Any], steps: int,
    gm_values: Sequence[float], np: types.ModuleType,
) -> Any:
    offset = _require_exact_int(locator.get("byte_offset"), "v8 source offset", minimum=0)
    length = _require_exact_int(locator.get("byte_length"), "v8 source length", minimum=1)
    node_count = _require_exact_int(locator.get("node_count"), "v8 source nodes", minimum=2)
    if (
        locator.get("schema") != "jx.tno.stage15-resolved11-encounter-qualification.source-nodes.v8"
        or locator.get("record_bytes") != SOURCE_NODE_RECORD_BYTES
        or node_count != HORIZON_YEARS * steps + 1
        or length != node_count * SOURCE_NODE_RECORD_BYTES
    ):
        raise SourceConvergenceDiagnosticError("v8 source locator shape differs")
    lane_digest = hashlib.sha256()
    raw_annual = np.empty((CHECKPOINT_COUNT, 2, MASSIVE_COUNT, 3), dtype=np.float64)
    cursor = 0
    next_year = 0
    while cursor < node_count:
        count = min(SOURCE_CHUNK_NODES, node_count - cursor)
        payload = _pread_all(
            fd, count * SOURCE_NODE_RECORD_BYTES,
            offset + cursor * SOURCE_NODE_RECORD_BYTES,
        )
        lane_digest.update(payload)
        rows = np.frombuffer(payload, dtype="<f8").reshape(count, 2, MASSIVE_COUNT, 3)
        while next_year < CHECKPOINT_COUNT and next_year * steps < cursor + count:
            node = next_year * steps
            if node >= cursor:
                raw_annual[next_year] = rows[node - cursor]
                next_year += 1
            else:
                raise SourceConvergenceDiagnosticError("v8 annual extraction order differs")
        cursor += count
    if (
        next_year != CHECKPOINT_COUNT
        or lane_digest.hexdigest() != locator.get("lane_payload_sha256")
    ):
        raise SourceConvergenceDiagnosticError("v8 source evidence digest differs")
    return _center_annual(np, raw_annual, gm_values)


def _extract_v4_oracle_annual(
    fd: int, locator: Mapping[str, Any], batch_order: int,
    np: types.ModuleType,
) -> Any:
    expected_keys = {
        "batch_order", "byte_length", "byte_offset", "checkpoint_count",
        "lane_id", "lane_payload_sha256", "massive_count", "object_id",
        "record_bytes", "tracer_count",
    }
    _require_keys(locator, expected_keys, "v4 IAS15 oracle locator")
    offset = _require_exact_int(locator["byte_offset"], "v4 oracle offset", minimum=0)
    length = _require_exact_int(locator["byte_length"], "v4 oracle length", minimum=1)
    retained_batch_order = _require_exact_int(
        locator["batch_order"], "v4 oracle batch order", minimum=0,
    )
    if (
        retained_batch_order != batch_order
        or retained_batch_order >= BATCH_COUNT
        or locator["object_id"] != OBJECT_IDS[batch_order]
        or locator["lane_id"] != "rebound511_ias15_resolved11"
        or locator["checkpoint_count"] != CHECKPOINT_COUNT
        or locator["massive_count"] != MASSIVE_COUNT
        or locator["tracer_count"] != 9
        or locator["record_bytes"] != 960
        or length != CHECKPOINT_COUNT * 960
    ):
        raise SourceConvergenceDiagnosticError("v4 IAS15 oracle shape differs")
    payload = _pread_all(fd, length, offset)
    if _sha256(payload) != locator["lane_payload_sha256"]:
        raise SourceConvergenceDiagnosticError("v4 IAS15 oracle digest differs")
    result = np.empty((CHECKPOINT_COUNT, 2, MASSIVE_COUNT, 3), dtype=np.float64)
    for year in range(CHECKPOINT_COUNT):
        start = year * 960
        massive = payload[start : start + ANNUAL_RECORD_BYTES]
        result[year] = np.frombuffer(massive, dtype="<f8").reshape(2, MASSIVE_COUNT, 3)
    if not bool(np.isfinite(result).all()):
        raise SourceConvergenceDiagnosticError("v4 IAS15 oracle is nonfinite")
    return result


def _v8_source_locator_map(result: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    lanes = result.get("lanes")
    if type(lanes) is not list or len(lanes) != BATCH_COUNT * 2:
        raise SourceConvergenceDiagnosticError("v8 lane roster differs")
    output: dict[tuple[int, int], Mapping[str, Any]] = {}
    for lane_order, lane in enumerate(lanes):
        if type(lane) is not dict:
            raise SourceConvergenceDiagnosticError("v8 lane record differs")
        batch_order = _require_exact_int(lane.get("batch_order"), "v8 batch order", minimum=0)
        steps = _require_exact_int(lane.get("steps_per_julian_year"), "v8 steps", minimum=1)
        expected_lane_order = batch_order * 2 + RETAINED_STEPS.index(steps) if steps in RETAINED_STEPS else -1
        if (
            batch_order >= BATCH_COUNT or lane_order != expected_lane_order
            or lane.get("object_id") != OBJECT_IDS[batch_order]
            or lane.get("lane_id") != f"jx_resolved11_encounter_h{steps}"
            or lane.get("engine") != "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2"
            or lane.get("model") != "RESOLVED_11_BODY"
        ):
            raise SourceConvergenceDiagnosticError("v8 lane identity differs")
        key = (batch_order, steps)
        if key in output:
            raise SourceConvergenceDiagnosticError("v8 lane identity duplicates")
        output[key] = lane.get("source_nodes")
    if len(output) != BATCH_COUNT * 2:
        raise SourceConvergenceDiagnosticError("v8 source locator count differs")
    return output


def _gate_invariants(
    invariants: Mapping[int, Mapping[str, Any]], triggered_h1024: bool,
) -> dict[str, Any]:
    required = ALL_STEP_LEVELS if triggered_h1024 else ALL_STEP_LEVELS[:-1]
    rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    contraction_rows: list[dict[str, Any]] = []
    energy_values: dict[int, Fraction] = {}
    for steps in required:
        summary = invariants.get(steps)
        if type(summary) is not dict:
            raise SourceConvergenceDiagnosticError("invariant summary is absent")
        energy = _fraction_from_record(
            summary["maximum_relative_energy_drift"]["exact"], "energy drift",
        )
        try:
            angular = float.fromhex(summary["maximum_relative_angular_momentum_drift_hex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceConvergenceDiagnosticError("angular drift encoding differs") from exc
        energy_values[steps] = energy
        energy_row = {
            "gate": f"h{steps}_all_node_energy", "passed": energy <= ENERGY_BUDGET,
            "observation": _fraction_record(energy), "budget": _fraction_record(ENERGY_BUDGET),
        }
        angular_row = {
            "gate": f"h{steps}_all_node_angular_momentum",
            "passed": Fraction.from_float(angular) <= ANGULAR_BUDGET,
            "observation_hex": angular.hex(), "budget": _fraction_record(ANGULAR_BUDGET),
        }
        absolute_rows.extend((energy_row, angular_row))
        rows.extend((energy_row, angular_row))
    pairs = [(64, 128), (128, 256), (256, 512)]
    if triggered_h1024:
        pairs.append((512, 1024))
    for coarse_steps, fine_steps in pairs:
        coarse_summary = invariants.get(coarse_steps)
        if type(coarse_summary) is not dict:
            raise SourceConvergenceDiagnosticError("coarse invariant summary is absent")
        coarse = _fraction_from_record(
            coarse_summary["maximum_relative_energy_drift"]["exact"], "coarse energy drift",
        )
        fine = energy_values[fine_steps]
        limit = max(coarse / 3, ENERGY_CONTRACTION_FLOOR)
        contraction_row = {
            "gate": f"h{coarse_steps}_h{fine_steps}_energy_contraction",
            "passed": fine <= limit, "fine": _fraction_record(fine),
            "limit": _fraction_record(limit),
        }
        contraction_rows.append(contraction_row)
        rows.append(contraction_row)
    finest_steps = required[-1]
    finest_absolute_rows = tuple(
        row for row in absolute_rows if row["gate"].startswith(f"h{finest_steps}_")
    )
    if len(finest_absolute_rows) != 2:
        raise SourceConvergenceDiagnosticError("finest invariant gate roster differs")
    return {
        "all_passed": all(row["passed"] for row in rows),
        "all_absolute_levels_passed": all(row["passed"] for row in absolute_rows),
        "all_adjacent_energy_contractions_passed": all(
            row["passed"] for row in contraction_rows
        ),
        "finest_steps_per_year": finest_steps,
        "finest_absolute_energy_passed": finest_absolute_rows[0]["passed"],
        "finest_absolute_angular_momentum_passed": finest_absolute_rows[1]["passed"],
        "finest_absolute_gates_passed": all(
            row["passed"] for row in finest_absolute_rows
        ),
        "nondecisive_absolute_and_adjacent_contraction_rows_are_diagnostic_only": True,
        "gates": rows,
    }


def _pre_adjudication_diagnostic(
    levels: Mapping[int, Any], retained_ias: Any, tight_ias: Any,
) -> dict[str, Any]:
    required = {64, 128, 256, 512}
    if set(levels) != required:
        raise SourceConvergenceDiagnosticError("pre-adjudication step roster differs")
    comparisons = {
        "h64_h128": _series_comparison(levels[64], levels[128]),
        "h128_h256": _series_comparison(levels[128], levels[256]),
        "h256_h512": _series_comparison(levels[256], levels[512]),
        "h512_tight_ias15": _series_comparison(levels[512], tight_ias),
        "tight_retained_ias15": _series_comparison(tight_ias, retained_ias),
    }
    windows = {
        "h64_h128_h256": _order_window(
            comparisons["h64_h128"], comparisons["h128_h256"], "h64_h128_h256",
        ),
        "h128_h256_h512": _order_window(
            comparisons["h128_h256"], comparisons["h256_h512"], "h128_h256_h512",
        ),
    }
    richardson = _richardson_comparison(
        levels[128], levels[256], levels[256], levels[512],
    )
    reference_qualified = _comparison_within_budgets(
        comparisons["tight_retained_ias15"],
        REFERENCE_POSITION_BUDGET, REFERENCE_VELOCITY_BUDGET,
    )
    h512_reference_passed = _comparison_within_budgets(
        comparisons["h512_tight_ias15"], POSITION_BUDGET, VELOCITY_BUDGET,
    )
    richardson_passed = _richardson_within_budgets(
        richardson, POSITION_BUDGET, VELOCITY_BUDGET,
    )
    trigger = not (
        reference_qualified
        and windows["h64_h128_h256"]["all_passed"]
        and windows["h128_h256_h512"]["all_passed"]
        and richardson_passed and h512_reference_passed
    )
    return {
        "reference_qualified": reference_qualified,
        "comparisons": comparisons, "order_windows": windows,
        "richardson_xstar256_xstar512": richardson,
        "richardson_xstar256_xstar512_passed": richardson_passed,
        "h512_tight_ias15_passed": h512_reference_passed,
        "h1024_triggered": trigger,
        "trigger_rule": "EXACT_PREDECLARED_OUTCOME_BLIND_RULE",
    }


def _step_estimate_from_records(
    previous_error: list[dict[str, Any]], final_error: list[dict[str, Any]],
    raw_reference: list[dict[str, Any]], *, current_steps: int,
    two_windows_passed: bool, reference_qualified: bool,
    relevant_final_richardson_and_extrapolate_gates_passed: bool,
    finest_absolute_gates_passed: bool, final_disposition: str,
) -> dict[str, Any]:
    previous = _comparison_index(previous_error, "step estimate previous")
    final = _comparison_index(final_error, "step estimate final")
    reference = _comparison_index(raw_reference, "step estimate reference")
    estimates: list[int] = []
    orders: list[float] = []
    if two_windows_passed:
        for key in previous:
            for component, budget in (
                ("position", POSITION_BUDGET), ("velocity", VELOCITY_BUDGET),
            ):
                coarse = _fraction_from_record(previous[key][f"{component}_squared"]["exact"], "order coarse")
                fine = _fraction_from_record(final[key][f"{component}_squared"]["exact"], "order fine")
                order = _observed_order(coarse, fine)
                if order is None:
                    continue
                orders.append(order)
                reference_error = _fraction_from_record(
                    reference[key][f"{component}_squared"]["exact"], "reference error",
                )
                estimate = _estimated_power_of_two_steps(
                    current_steps=current_steps, error_squared=reference_error,
                    budget=budget, observed_order=order,
                    two_consecutive_windows_passed=True,
                )
                if estimate is not None:
                    estimates.append(estimate)
    suppression_reason: str | None
    if not reference_qualified:
        suppression_reason = "IAS15_REFERENCE_NOT_QUALIFIED"
    elif not two_windows_passed:
        suppression_reason = "TWO_CONSECUTIVE_FINAL_ORDER_WINDOWS_NOT_PASSED"
    elif not relevant_final_richardson_and_extrapolate_gates_passed:
        suppression_reason = "RELEVANT_FINAL_RICHARDSON_OR_EXTRAPOLATE_GATE_FAILED"
    elif not finest_absolute_gates_passed:
        suppression_reason = "DECISIVE_FINEST_ABSOLUTE_INVARIANT_GATE_FAILED"
    elif final_disposition not in ("PASS", "TIMESTEP_LIMITED"):
        suppression_reason = "FINAL_DISPOSITION_NOT_STEP_ESTIMABLE"
    elif not estimates:
        suppression_reason = "NO_REQUIRED_STEP_ESTIMATE_DERIVED"
    else:
        suppression_reason = None
    authorized = suppression_reason is None
    return {
        "authorized": authorized,
        "estimated_required_steps_per_year": max(estimates) if authorized else None,
        "minimum_observed_order_hex": min(orders).hex() if orders else None,
        "decisive_finest_absolute_gates_passed": finest_absolute_gates_passed,
        "coordinate_only_estimate_suppressed_due_to_finest_invariant_failure": (
            not finest_absolute_gates_passed
        ),
        "suppression_reason": suppression_reason,
        "advisory_only_no_execution_authority": True,
    }


def _finalize_epoch_diagnostic(
    pre: Mapping[str, Any], levels: Mapping[int, Any], tight_ias: Any,
    invariants: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    triggered = pre.get("h1024_triggered") is True
    invariant_gates = _gate_invariants(invariants, triggered)
    result = dict(pre)
    if triggered:
        if 1024 not in levels:
            raise SourceConvergenceDiagnosticError("triggered h1024 evidence is absent")
        h512_h1024 = _series_comparison(levels[512], levels[1024])
        h1024_reference = _series_comparison(levels[1024], tight_ias)
        window = _order_window(
            pre["comparisons"]["h256_h512"], h512_h1024,
            "h256_h512_h1024",
        )
        richardson = _richardson_comparison(
            levels[256], levels[512], levels[512], levels[1024],
        )
        extrapolated_reference = _richardson_reference_comparison(
            levels[512], levels[1024], tight_ias,
        )
        richardson_passed = _richardson_within_budgets(
            richardson, POSITION_BUDGET, VELOCITY_BUDGET,
        )
        raw_passed = _comparison_within_budgets(
            h1024_reference, POSITION_BUDGET, VELOCITY_BUDGET,
        )
        extrapolated_passed = _richardson_within_budgets(
            extrapolated_reference, POSITION_BUDGET, VELOCITY_BUDGET,
        )
        two_windows = (
            pre["order_windows"]["h128_h256_h512"]["all_passed"]
            and window["all_passed"]
        )
        convergence_core = (
            pre["reference_qualified"] and two_windows and richardson_passed
        )
        finest_absolute_passed = invariant_gates["finest_absolute_gates_passed"]
        if not finest_absolute_passed:
            disposition = "INCONCLUSIVE_AT_MAX_RESOLUTION"
        elif (
            convergence_core and raw_passed and extrapolated_passed
        ):
            disposition = "PASS"
        elif convergence_core and extrapolated_passed and not raw_passed:
            disposition = "TIMESTEP_LIMITED"
        elif convergence_core and not extrapolated_passed:
            disposition = "NON_TIMESTEP_REFERENCE_DISCREPANCY"
        else:
            disposition = "INCONCLUSIVE_AT_MAX_RESOLUTION"
        result.update({
            "h512_h1024": h512_h1024,
            "h1024_tight_ias15": h1024_reference,
            "h256_h512_h1024_order_window": window,
            "richardson_xstar512_xstar1024": richardson,
            "richardson_xstar512_xstar1024_passed": richardson_passed,
            "richardson_xstar1024_tight_ias15": extrapolated_reference,
            "richardson_xstar1024_tight_ias15_passed": extrapolated_passed,
            "h1024_tight_ias15_passed": raw_passed,
            "two_consecutive_final_order_windows_passed": two_windows,
            "step_estimate": _step_estimate_from_records(
                pre["comparisons"]["h256_h512"], h512_h1024,
                h1024_reference, current_steps=1024,
                two_windows_passed=two_windows,
                reference_qualified=pre["reference_qualified"],
                relevant_final_richardson_and_extrapolate_gates_passed=(
                    richardson_passed and extrapolated_passed
                ),
                finest_absolute_gates_passed=finest_absolute_passed,
                final_disposition=disposition,
            ),
        })
    else:
        two_windows = (
            pre["order_windows"]["h64_h128_h256"]["all_passed"]
            and pre["order_windows"]["h128_h256_h512"]["all_passed"]
        )
        convergence_core = (
            pre["reference_qualified"] and two_windows
            and pre["richardson_xstar256_xstar512_passed"]
            and pre["h512_tight_ias15_passed"]
        )
        if convergence_core and invariant_gates["finest_absolute_gates_passed"]:
            disposition = "PASS"
        else:
            disposition = "INCONCLUSIVE_AT_MAX_RESOLUTION"
        result.update({
            "two_consecutive_final_order_windows_passed": two_windows,
            "step_estimate": _step_estimate_from_records(
                pre["comparisons"]["h128_h256"],
                pre["comparisons"]["h256_h512"],
                pre["comparisons"]["h512_tight_ias15"],
                current_steps=512, two_windows_passed=two_windows,
                reference_qualified=pre["reference_qualified"],
                relevant_final_richardson_and_extrapolate_gates_passed=pre[
                    "richardson_xstar256_xstar512_passed"
                ],
                finest_absolute_gates_passed=invariant_gates[
                    "finest_absolute_gates_passed"
                ],
                final_disposition=disposition,
            ),
        })
    if disposition not in DISPOSITIONS:
        raise SourceConvergenceDiagnosticError("diagnostic disposition differs")
    result["invariant_gates"] = invariant_gates
    result["disposition"] = disposition
    result["stage15_qualification_established"] = False
    result["science_result"] = False
    result["adaptive_rerun_authorized"] = False
    return result


def _validate_small_prerequisites() -> tuple[types.ModuleType, dict[str, Any]]:
    checks = (
        (V8_SOURCE_PATH, V8_SOURCE_BYTES, V8_SOURCE_SHA256, "v8 source"),
        (V8_TEST_PATH, V8_TEST_BYTES, V8_TEST_SHA256, "v8 test"),
        (V8_PROTOCOL_PATH, V8_PROTOCOL_BYTES, V8_PROTOCOL_SHA256, "v8 protocol"),
        (V8_RESULT_PATH, V8_RESULT_BYTES, V8_RESULT_SHA256, "v8 result"),
        (V4_SOURCE_PATH, V4_SOURCE_BYTES, V4_SOURCE_SHA256, "Stage-1 v4 source"),
        (V4_PROTOCOL_PATH, V4_PROTOCOL_BYTES, V4_PROTOCOL_SHA256, "Stage-1 v4 protocol"),
        (V4_RESULT_PATH, V4_RESULT_BYTES, V4_RESULT_SHA256, "Stage-1 v4 result"),
        (V4_ARRAYS_PATH, V4_ARRAYS_BYTES, V4_ARRAYS_SHA256, "v4 arrays"),
        (BRIDGE_PATH, BRIDGE_BYTES, BRIDGE_SHA256, "resolved-11 bridge"),
        (
            DE441_CUSTODY_PATH, DE441_CUSTODY_BYTES,
            DE441_CUSTODY_SHA256, "DE441 custody",
        ),
        (
            STAGE15_V1_PROTOCOL_PATH, STAGE15_V1_PROTOCOL_BYTES,
            STAGE15_V1_PROTOCOL_SHA256, "Stage-1.5 v1 protocol",
        ),
        (REBOUND_WHEEL_PATH, REBOUND_WHEEL_BYTES, REBOUND_WHEEL_SHA256, "REBOUND wheel"),
    )
    # The retained V8 source-node binary is hashed from a held descriptor at
    # semantic use.  All remaining prerequisite bytes are checked here.
    for path, size, digest, label in checks:
        maximum = max(size, 1)
        payload = _regular_bytes(path, label, exact_bytes=size, maximum=maximum)
        if _sha256(payload) != digest:
            raise SourceConvergenceDiagnosticError(f"{label} digest differs")
    protocol = _load_canonical_json(
        _regular_bytes(V8_PROTOCOL_PATH, "v8 protocol", exact_bytes=V8_PROTOCOL_BYTES),
        "v8 protocol", trailing_newline=True,
    )
    if protocol.get("content_sha256") != V8_PROTOCOL_CONTENT_SHA256:
        raise SourceConvergenceDiagnosticError("v8 protocol content digest differs")
    for path, size, content_digest, label in (
        (
            V4_PROTOCOL_PATH, V4_PROTOCOL_BYTES,
            V4_PROTOCOL_CONTENT_SHA256, "Stage-1 v4 protocol",
        ),
        (
            V4_RESULT_PATH, V4_RESULT_BYTES,
            V4_RESULT_CONTENT_SHA256, "Stage-1 v4 result",
        ),
    ):
        value = _load_canonical_json(
            _regular_bytes(path, label, exact_bytes=size),
            label, trailing_newline=True,
        )
        if value.get("content_sha256") != content_digest:
            raise SourceConvergenceDiagnosticError(f"{label} content digest differs")
        if label == "Stage-1 v4 result":
            claims = value.get("claims")
            gates = value.get("qualification_gates")
            if (
                value.get("status") != "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
                or type(claims) is not dict
                or claims.get("science_result") is not False
                or type(gates) is not dict
                or gates.get("all_predeclared_gates_passed") is not False
            ):
                raise SourceConvergenceDiagnosticError(
                    "Stage-1 v4 failed-result semantics differ"
                )
    v1_protocol = _load_canonical_json(
        _regular_bytes(
            STAGE15_V1_PROTOCOL_PATH, "Stage-1.5 v1 protocol",
            exact_bytes=STAGE15_V1_PROTOCOL_BYTES,
        ),
        "Stage-1.5 v1 protocol", trailing_newline=True,
    )
    if v1_protocol.get("content_sha256") != STAGE15_V1_PROTOCOL_CONTENT_SHA256:
        raise SourceConvergenceDiagnosticError("Stage-1.5 v1 protocol content digest differs")
    result = _validate_v8_failure_result()
    return _v8_module(), result


def _extract_rebound_runtime(target: Path) -> dict[str, Any]:
    target = Path(os.path.abspath(os.fspath(target)))
    if target.exists() or target.is_symlink():
        raise SourceConvergenceDiagnosticError("REBOUND extraction target must be absent")
    target.mkdir(mode=0o700)
    wheel_payload = _regular_bytes(
        REBOUND_WHEEL_PATH, "REBOUND wheel", exact_bytes=REBOUND_WHEEL_BYTES,
        maximum=2_000_000,
    )
    if _sha256(wheel_payload) != REBOUND_WHEEL_SHA256:
        raise SourceConvergenceDiagnosticError("REBOUND wheel digest differs")
    members: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(REBOUND_WHEEL_PATH, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SourceConvergenceDiagnosticError("REBOUND wheel members duplicate")
            for info in infos:
                relative = Path(info.filename)
                if (
                    relative.is_absolute() or ".." in relative.parts
                    or "\\" in info.filename
                ):
                    raise SourceConvergenceDiagnosticError("REBOUND wheel member path is unsafe")
                destination = target.joinpath(*relative.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode not in (0, stat.S_IFREG):
                    raise SourceConvergenceDiagnosticError("REBOUND wheel has nonregular member")
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = archive.read(info)
                _write_create_only(destination, payload)
                members.append({
                    "wheel_member": info.filename,
                    "bytes": len(payload), "sha256": _sha256(payload),
                })
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise SourceConvergenceDiagnosticError("REBOUND wheel extraction failed") from exc
    return {
        "wheel_sha256": REBOUND_WHEEL_SHA256,
        "member_count": len(members), "members": members,
    }


def _load_rebound_runtime(
    v4: types.ModuleType, target: Path, expected_extraction: Mapping[str, Any],
) -> tuple[types.ModuleType, dict[str, Any]]:
    try:
        _cp, rebound, target_receipt = v4._load_dynamics_runtime(
            target, require_cuda_device=False,
        )
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("exact REBOUND runtime load failed") from exc
    if type(rebound) is not types.ModuleType or rebound.__name__ != "rebound":
        raise SourceConvergenceDiagnosticError("REBOUND runtime module identity differs")
    if getattr(rebound, "__version__", None) != "5.1.1":
        raise SourceConvergenceDiagnosticError("REBOUND runtime version differs")
    normalized = {
        "wheel_sha256": target_receipt.get("wheel_sha256"),
        "members": [
            {
                "wheel_member": row.get("wheel_member"),
                "bytes": row.get("bytes"), "sha256": row.get("sha256"),
            }
            for row in target_receipt.get("members", [])
        ],
    }
    expected_members = {
        row["wheel_member"]: (row["bytes"], row["sha256"])
        for row in expected_extraction.get("members", [])
    }
    for row in normalized["members"]:
        if expected_members.get(row["wheel_member"]) != (row["bytes"], row["sha256"]):
            raise SourceConvergenceDiagnosticError("REBOUND imported member differs from wheel")
    normalized["verified_executable_member_count"] = len(normalized["members"])
    normalized["module_version"] = rebound.__version__
    return rebound, normalized


def _runtime_wrapper(
    *, v8: types.ModuleType, base_before: Mapping[str, Any],
    base_after: Mapping[str, Any], rebound_receipt: Mapping[str, Any],
    engine: types.ModuleType,
) -> dict[str, Any]:
    finalized = v8._finalize_runtime_receipt(base_before, base_after)
    source_text = getattr(engine, "SOURCE_PATH_CUDA_SOURCE", None)
    options = getattr(engine, "CUDA_OPTIONS", None)
    if type(source_text) is not str or type(options) is not tuple or not options:
        raise SourceConvergenceDiagnosticError("source kernel contract differs")
    record: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "base_v8_runtime": finalized,
        "base_v8_runtime_projection_sha256": _sha256(
            _canonical_json(v8._runtime_process_independent_projection(finalized))
        ),
        "rebound_runtime": dict(rebound_receipt),
        "implementation": _file_receipt(
            SCRIPT_PATH, "diagnostic runtime implementation", maximum=1_000_000,
        ),
        "source_only_adapter": {
            "engine_module": "jxplanetx.cuda_massless_tracer",
            "private_builder": "_build_source_path",
            "source_kernel_name": "jx_build_massive_source_path",
            "source_kernel_text_sha256": _sha256(source_text.encode("utf-8")),
            "cuda_options": list(options),
            "public_tracer_entrypoint_called": False,
            "tracer_state_constructed": False,
        },
        "ias15_tight_settings": {
            "epsilon_hex": IAS15_TIGHT_EPSILON.hex(),
            "initial_dt_hex": IAS15_INITIAL_DT.hex(),
            "min_dt_hex": IAS15_MIN_DT.hex(),
            "adaptive_mode": IAS15_ADAPTIVE_MODE,
        },
        "trust_boundary": "NO_HOSTILE_CODE_IN_FRESH_CHILD",
        "claim": "EXACT_CRITICAL_RUNTIME_POINT_OBSERVATIONS_NOT_FULL_INTERPRETER_AUTHENTICATION",
    }
    record["content_sha256"] = _domain_sha(RUNTIME_DOMAIN, record)
    return record


def _validate_runtime_wrapper(
    retained: Mapping[str, Any], *, v8: types.ModuleType,
    current_base: Mapping[str, Any], current_rebound: Mapping[str, Any],
    engine: types.ModuleType,
) -> None:
    keys = {
        "schema", "base_v8_runtime", "base_v8_runtime_projection_sha256",
        "rebound_runtime", "implementation", "source_only_adapter",
        "ias15_tight_settings", "trust_boundary", "claim", "content_sha256",
    }
    _require_keys(retained, keys, "diagnostic runtime")
    if (
        retained["schema"] != RUNTIME_SCHEMA
        or retained["content_sha256"] != _domain_sha(RUNTIME_DOMAIN, retained)
        or retained["trust_boundary"] != "NO_HOSTILE_CODE_IN_FRESH_CHILD"
        or retained["claim"]
        != "EXACT_CRITICAL_RUNTIME_POINT_OBSERVATIONS_NOT_FULL_INTERPRETER_AUTHENTICATION"
    ):
        raise SourceConvergenceDiagnosticError("diagnostic runtime identity differs")
    try:
        v8._validate_runtime_document(retained["base_v8_runtime"], current_base)
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("live v8 runtime differs") from exc
    projection = v8._runtime_process_independent_projection(retained["base_v8_runtime"])
    if retained["base_v8_runtime_projection_sha256"] != _sha256(_canonical_json(projection)):
        raise SourceConvergenceDiagnosticError("retained v8 runtime projection differs")
    if retained["rebound_runtime"] != current_rebound:
        raise SourceConvergenceDiagnosticError("live REBOUND runtime differs")
    if retained["implementation"] != _file_receipt(
        SCRIPT_PATH, "live diagnostic implementation", maximum=1_000_000,
    ):
        raise SourceConvergenceDiagnosticError("live diagnostic implementation differs")
    source_text = getattr(engine, "SOURCE_PATH_CUDA_SOURCE", None)
    options = getattr(engine, "CUDA_OPTIONS", None)
    if type(source_text) is not str or type(options) is not tuple or not options:
        raise SourceConvergenceDiagnosticError("live source kernel contract differs")
    expected_adapter = {
        "engine_module": "jxplanetx.cuda_massless_tracer",
        "private_builder": "_build_source_path",
        "source_kernel_name": "jx_build_massive_source_path",
        "source_kernel_text_sha256": _sha256(source_text.encode("utf-8")),
        "cuda_options": list(options),
        "public_tracer_entrypoint_called": False,
        "tracer_state_constructed": False,
    }
    if retained["source_only_adapter"] != expected_adapter:
        raise SourceConvergenceDiagnosticError("source-only adapter runtime differs")
    if retained["ias15_tight_settings"] != {
        "epsilon_hex": IAS15_TIGHT_EPSILON.hex(),
        "initial_dt_hex": IAS15_INITIAL_DT.hex(),
        "min_dt_hex": IAS15_MIN_DT.hex(),
        "adaptive_mode": IAS15_ADAPTIVE_MODE,
    }:
        raise SourceConvergenceDiagnosticError("tight IAS15 runtime settings differ")


def _source_kernel(cp: types.ModuleType, engine: types.ModuleType) -> Any:
    try:
        return cp.RawKernel(
            engine.SOURCE_PATH_CUDA_SOURCE,
            "jx_build_massive_source_path",
            options=engine.CUDA_OPTIONS,
        )
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("source-only CUDA kernel compilation failed") from exc


def _build_source_lane(
    *, np: types.ModuleType, cp: types.ModuleType, engine: types.ModuleType,
    backend: Any, source_kernel: Any, batch: Any, steps: int,
    source_writer: _CappedWriter, annual_writer: _CappedWriter,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if steps not in (256, 512, 1024):
        raise SourceConvergenceDiagnosticError("source-only step level differs")
    step_count = HORIZON_YEARS * steps
    gm_values = tuple(float(item.gm_AU3_per_julian_year2) for item in batch.massive)
    positions = np.asarray([item.position_AU for item in batch.massive], dtype=np.float64)
    velocities = np.asarray(
        [item.velocity_AU_per_julian_year for item in batch.massive], dtype=np.float64,
    )
    gm = np.asarray(gm_values, dtype=np.float64)
    source_offset = source_writer.count
    annual_offset = annual_writer.count
    lane_digest = hashlib.sha256()
    try:
        with backend.activate():
            path = engine._build_source_path(
                backend, source_kernel,
                cp.asarray(positions), cp.asarray(velocities), cp.asarray(gm),
                step_count=step_count, step=1.0 / steps,
            )
            annual_raw = np.empty(
                (CHECKPOINT_COUNT, 2, MASSIVE_COUNT, 3), dtype=np.float64,
            )
            annual_raw[:, 0] = cp.asnumpy(path.positions[::steps])
            annual_raw[:, 1] = cp.asnumpy(path.velocities[::steps])
            for start in range(0, step_count + 1, SOURCE_CHUNK_NODES):
                stop = min(step_count + 1, start + SOURCE_CHUNK_NODES)
                host_position = cp.asnumpy(path.positions[start:stop]).astype("<f8", copy=False)
                host_velocity = cp.asnumpy(path.velocities[start:stop]).astype("<f8", copy=False)
                interleaved = np.empty(
                    (stop - start, 2, MASSIVE_COUNT, 3), dtype="<f8",
                )
                interleaved[:, 0] = host_position
                interleaved[:, 1] = host_velocity
                payload = interleaved.tobytes(order="C")
                source_writer.write(payload)
                lane_digest.update(payload)
            del path
            cp.get_default_memory_pool().free_all_blocks()
    except SourceConvergenceDiagnosticError:
        raise
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError(f"h{steps} source-only integration failed") from exc
    centered = _center_annual(np, annual_raw, gm_values)
    annual_payload = _annual_payload(np, centered)
    annual_writer.write(annual_payload)
    expected_source_length = (step_count + 1) * SOURCE_NODE_RECORD_BYTES
    if source_writer.count - source_offset != expected_source_length:
        raise SourceConvergenceDiagnosticError("source-only lane byte length differs")
    source_locator = {
        "schema": SOURCE_SCHEMA, "kind": "JX_SOURCE_ONLY_FIXED_STEP",
        "lane_id": f"jx_resolved11_source_h{steps}",
        "steps_per_julian_year": steps, "step_julian_years_hex": (1.0 / steps).hex(),
        "byte_offset": source_offset, "byte_length": expected_source_length,
        "record_bytes": SOURCE_NODE_RECORD_BYTES, "node_count": step_count + 1,
        "lane_payload_sha256": lane_digest.hexdigest(),
    }
    annual_locator = {
        "schema": ANNUAL_SCHEMA, "kind": "JX_SOURCE_ONLY_FIXED_STEP",
        "lane_id": f"jx_resolved11_source_h{steps}",
        "steps_per_julian_year": steps,
        "byte_offset": annual_offset, "byte_length": len(annual_payload),
        "record_bytes": ANNUAL_RECORD_BYTES, "checkpoint_count": CHECKPOINT_COUNT,
        "lane_payload_sha256": _sha256(annual_payload),
    }
    return source_locator, annual_locator, centered


def _run_tight_ias15(
    *, np: types.ModuleType, rebound: types.ModuleType, batch: Any,
    annual_writer: _CappedWriter,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "ias15"
    simulation.dt = IAS15_INITIAL_DT
    simulation.integrator.epsilon = IAS15_TIGHT_EPSILON
    simulation.integrator.min_dt = IAS15_MIN_DT
    simulation.integrator.adaptive_mode = IAS15_ADAPTIVE_MODE
    observed_initial_dt = float(simulation.dt)
    if (
        observed_initial_dt.hex() != IAS15_INITIAL_DT.hex()
        or float(simulation.integrator.epsilon).hex()
        != IAS15_TIGHT_EPSILON.hex()
        or float(simulation.integrator.min_dt).hex() != IAS15_MIN_DT.hex()
        or str(simulation.integrator.adaptive_mode) != IAS15_ADAPTIVE_MODE
    ):
        raise SourceConvergenceDiagnosticError(
            "tight IAS15 initial settings differ"
        )
    for item in batch.massive:
        simulation.add(
            m=item.gm_AU3_per_julian_year2,
            x=item.position_AU[0], y=item.position_AU[1], z=item.position_AU[2],
            vx=item.velocity_AU_per_julian_year[0],
            vy=item.velocity_AU_per_julian_year[1],
            vz=item.velocity_AU_per_julian_year[2],
        )
    simulation.N_active = MASSIVE_COUNT
    simulation.testparticle_type = 0
    raw = np.empty((CHECKPOINT_COUNT, 2, MASSIVE_COUNT, 3), dtype=np.float64)
    step_counts: list[int] = []

    def capture(year: int) -> None:
        if int(simulation.N) != MASSIVE_COUNT or int(simulation.N_active) != MASSIVE_COUNT:
            raise SourceConvergenceDiagnosticError("tight IAS15 source roster differs")
        particles = simulation.particles
        for source in range(MASSIVE_COUNT):
            particle = particles[source]
            raw[year, 0, source] = (float(particle.x), float(particle.y), float(particle.z))
            raw[year, 1, source] = (float(particle.vx), float(particle.vy), float(particle.vz))
        if not bool(np.isfinite(raw[year]).all()):
            raise SourceConvergenceDiagnosticError("tight IAS15 state is nonfinite")
        steps_done = _require_exact_int(int(simulation.steps_done), "IAS15 steps done", minimum=0)
        if step_counts and steps_done <= step_counts[-1]:
            raise SourceConvergenceDiagnosticError("tight IAS15 step count did not advance")
        step_counts.append(steps_done)

    capture(0)
    for year in range(1, CHECKPOINT_COUNT):
        simulation.integrate(float(year), exact_finish_time=1)
        if float(simulation.t).hex() != float(year).hex():
            raise SourceConvergenceDiagnosticError("tight IAS15 annual time differs")
        capture(year)
    gm_values = tuple(float(item.gm_AU3_per_julian_year2) for item in batch.massive)
    centered = _center_annual(np, raw, gm_values)
    payload = _annual_payload(np, centered)
    offset = annual_writer.count
    annual_writer.write(payload)
    locator = {
        "schema": ANNUAL_SCHEMA, "kind": "REBOUND_5.1.1_IAS15_TIGHT_SOURCE_ONLY",
        "lane_id": "rebound511_ias15_tight_resolved11_source_only",
        "steps_per_julian_year": 0,
        "byte_offset": offset, "byte_length": len(payload),
        "record_bytes": ANNUAL_RECORD_BYTES, "checkpoint_count": CHECKPOINT_COUNT,
        "lane_payload_sha256": _sha256(payload),
    }
    receipt = {
        "epsilon_hex": float(simulation.integrator.epsilon).hex(),
        "initial_dt_hex": observed_initial_dt.hex(),
        "min_dt_hex": float(simulation.integrator.min_dt).hex(),
        "adaptive_mode": str(simulation.integrator.adaptive_mode),
        "annual_integrate_call_count": HORIZON_YEARS,
        "initial_steps_done": step_counts[0],
        "final_steps_done": step_counts[-1],
        "annual_steps_done": step_counts,
        "annual_steps_done_sha256": _sha256(
            struct.pack(f"<{len(step_counts)}Q", *step_counts)
        ),
        "final_time_hex": float(simulation.t).hex(),
        "gravitational_constant_hex": float(simulation.G).hex(),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening_hex": float(simulation.softening).hex(),
        "testparticle_type": int(simulation.testparticle_type),
        "active_source_count": int(simulation.N_active),
        "source_count": MASSIVE_COUNT,
        "tracer_count": 0,
    }
    if receipt != {
        **receipt,
        "epsilon_hex": IAS15_TIGHT_EPSILON.hex(),
        "initial_dt_hex": IAS15_INITIAL_DT.hex(),
        "min_dt_hex": IAS15_MIN_DT.hex(),
        "adaptive_mode": IAS15_ADAPTIVE_MODE,
        "annual_integrate_call_count": HORIZON_YEARS,
        "final_time_hex": float(HORIZON_YEARS).hex(),
        "gravitational_constant_hex": 1.0.hex(),
        "gravity": "basic", "collision": "none", "boundary": "none",
        "softening_hex": 0.0.hex(), "testparticle_type": 0,
        "active_source_count": MASSIVE_COUNT,
        "source_count": MASSIVE_COUNT,
        "tracer_count": 0,
    }:
        raise SourceConvergenceDiagnosticError("tight IAS15 settings changed")
    return locator, centered, receipt


def _hash_fd_range(fd: int, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < length:
        payload = _pread_all(fd, min(16 * 1024 * 1024, length - cursor), offset + cursor)
        digest.update(payload)
        cursor += len(payload)
    return digest.hexdigest()


def _validate_source_locator(
    locator: Mapping[str, Any], *, expected_offset: int, batch_order: int,
    steps: int,
) -> int:
    keys = {
        "schema", "kind", "lane_id", "steps_per_julian_year",
        "step_julian_years_hex", "byte_offset", "byte_length",
        "record_bytes", "node_count", "lane_payload_sha256",
    }
    _require_keys(locator, keys, "source locator")
    node_count = HORIZON_YEARS * steps + 1
    length = node_count * SOURCE_NODE_RECORD_BYTES
    for key, minimum in (
        ("steps_per_julian_year", 1), ("byte_offset", 0),
        ("byte_length", 1), ("record_bytes", 1), ("node_count", 2),
    ):
        _require_exact_int(locator[key], f"source locator {key}", minimum=minimum)
    if (
        locator["schema"] != SOURCE_SCHEMA
        or locator["kind"] != "JX_SOURCE_ONLY_FIXED_STEP"
        or locator["lane_id"] != f"jx_resolved11_source_h{steps}"
        or locator["steps_per_julian_year"] != steps
        or locator["step_julian_years_hex"] != (1.0 / steps).hex()
        or locator["byte_offset"] != expected_offset
        or locator["byte_length"] != length
        or locator["record_bytes"] != SOURCE_NODE_RECORD_BYTES
        or locator["node_count"] != node_count
    ):
        raise SourceConvergenceDiagnosticError(
            f"batch {batch_order} h{steps} source locator differs"
        )
    _require_sha(locator["lane_payload_sha256"], "source lane digest")
    return expected_offset + length


def _validate_annual_locator(
    locator: Mapping[str, Any], *, expected_offset: int, batch_order: int,
    kind: str, steps: int,
) -> int:
    keys = {
        "schema", "kind", "lane_id", "steps_per_julian_year",
        "byte_offset", "byte_length", "record_bytes", "checkpoint_count",
        "lane_payload_sha256",
    }
    _require_keys(locator, keys, "annual locator")
    expected_lane = (
        f"jx_resolved11_source_h{steps}"
        if kind == "JX_SOURCE_ONLY_FIXED_STEP"
        else "rebound511_ias15_tight_resolved11_source_only"
    )
    if (
        type(locator) is not dict
        or locator.get("schema") != ANNUAL_SCHEMA
        or locator.get("kind") != kind
        or locator.get("lane_id") != expected_lane
        or type(locator.get("steps_per_julian_year")) is not int
        or locator["steps_per_julian_year"] != steps
        or type(locator.get("byte_offset")) is not int
        or locator["byte_offset"] != expected_offset
        or type(locator.get("byte_length")) is not int
        or locator["byte_length"] != CHECKPOINT_COUNT * ANNUAL_RECORD_BYTES
        or type(locator.get("record_bytes")) is not int
        or locator["record_bytes"] != ANNUAL_RECORD_BYTES
        or type(locator.get("checkpoint_count")) is not int
        or locator["checkpoint_count"] != CHECKPOINT_COUNT
    ):
        raise SourceConvergenceDiagnosticError(
            f"batch {batch_order} {expected_lane} annual locator differs"
        )
    _require_sha(locator.get("lane_payload_sha256"), "annual lane digest")
    return expected_offset + CHECKPOINT_COUNT * ANNUAL_RECORD_BYTES


def _validate_ias_receipt(value: Any) -> None:
    keys = {
        "epsilon_hex", "initial_dt_hex", "min_dt_hex", "adaptive_mode",
        "annual_integrate_call_count", "initial_steps_done", "final_steps_done",
        "annual_steps_done", "annual_steps_done_sha256", "final_time_hex",
        "gravitational_constant_hex", "gravity", "collision", "boundary",
        "softening_hex", "testparticle_type", "active_source_count",
        "source_count", "tracer_count",
    }
    _require_keys(value, keys, "tight IAS15 receipt")
    for key, minimum in (
        ("annual_integrate_call_count", 1), ("initial_steps_done", 0),
        ("final_steps_done", 1), ("testparticle_type", 0),
        ("active_source_count", 1), ("source_count", 1),
        ("tracer_count", 0),
    ):
        _require_exact_int(value[key], f"tight IAS15 {key}", minimum=minimum)
    if (
        value["epsilon_hex"] != IAS15_TIGHT_EPSILON.hex()
        or value["initial_dt_hex"] != IAS15_INITIAL_DT.hex()
        or value["min_dt_hex"] != IAS15_MIN_DT.hex()
        or value["adaptive_mode"] != IAS15_ADAPTIVE_MODE
        or value["annual_integrate_call_count"] != HORIZON_YEARS
        or value["final_time_hex"] != float(HORIZON_YEARS).hex()
        or value["gravitational_constant_hex"] != 1.0.hex()
        or value["gravity"] != "basic"
        or value["collision"] != "none"
        or value["boundary"] != "none"
        or value["softening_hex"] != 0.0.hex()
        or value["testparticle_type"] != 0
        or value["active_source_count"] != MASSIVE_COUNT
        or value["source_count"] != MASSIVE_COUNT
        or value["tracer_count"] != 0
    ):
        raise SourceConvergenceDiagnosticError("tight IAS15 receipt semantics differ")
    initial = _require_exact_int(value["initial_steps_done"], "IAS initial steps", minimum=0)
    final = _require_exact_int(value["final_steps_done"], "IAS final steps", minimum=1)
    annual = value["annual_steps_done"]
    if type(annual) is not list or len(annual) != CHECKPOINT_COUNT:
        raise SourceConvergenceDiagnosticError("tight IAS15 annual step roster differs")
    for index, item in enumerate(annual):
        _require_exact_int(item, f"tight IAS15 annual step {index}", minimum=0)
    if (
        annual[0] != initial or annual[-1] != final or final <= initial
        or any(right <= left for left, right in zip(annual, annual[1:], strict=True))
        or _sha256(struct.pack(f"<{len(annual)}Q", *annual))
        != value["annual_steps_done_sha256"]
    ):
        raise SourceConvergenceDiagnosticError("tight IAS15 final step count differs")
    _require_sha(value["annual_steps_done_sha256"], "IAS annual step digest")


def _replay_evidence(
    *, v8: types.ModuleType, v8_result: Mapping[str, Any], v8_source_fd: int,
    v4_arrays_fd: int, source_fd: int, source_bytes: int,
    annual_fd: int, annual_bytes: int, skeleton_batches: Any,
    np: types.ModuleType,
) -> list[dict[str, Any]]:
    if type(skeleton_batches) is not list or len(skeleton_batches) != BATCH_COUNT:
        raise SourceConvergenceDiagnosticError("diagnostic batch skeleton differs")
    try:
        v4, _bridge, _custody, batches, oracle_locators = v8._load_v4_context()
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("Stage-1 v4 context reconstruction failed") from exc
    v8_locators = _v8_source_locator_map(v8_result)
    source_cursor = 0
    annual_cursor = 0
    result: list[dict[str, Any]] = []
    for batch_order, skeleton in enumerate(skeleton_batches):
        keys = {
            "batch_order", "object_id", "source_lanes", "annual_lanes",
            "tight_ias15_execution_receipt",
        }
        _require_keys(skeleton, keys, "diagnostic batch skeleton")
        if (
            skeleton["batch_order"] != batch_order
            or type(skeleton["batch_order"]) is not int
            or skeleton["object_id"] != OBJECT_IDS[batch_order]
        ):
            raise SourceConvergenceDiagnosticError("diagnostic batch identity differs")
        batch = batches[batch_order]
        gm_values = tuple(float(item.gm_AU3_per_julian_year2) for item in batch.massive)
        expected_initial = _batch_initial_source_payload(np, batch)
        expected_centered_initial = _center_single_node(np, expected_initial, gm_values)
        for steps in RETAINED_STEPS:
            _require_initial_source_payload(
                v8_source_fd, v8_locators[(batch_order, steps)],
                expected_initial, f"retained h{steps}",
            )
        source_lanes = skeleton["source_lanes"]
        annual_lanes = skeleton["annual_lanes"]
        if type(source_lanes) is not list or type(annual_lanes) is not list:
            raise SourceConvergenceDiagnosticError("diagnostic lane container differs")
        source_by_steps: dict[int, Mapping[str, Any]] = {}
        for expected_steps in BASE_STEPS:
            matches = [row for row in source_lanes if type(row) is dict and row.get("steps_per_julian_year") == expected_steps]
            if len(matches) != 1:
                raise SourceConvergenceDiagnosticError("required source lane roster differs")
            locator = matches[0]
            source_cursor = _validate_source_locator(
                locator, expected_offset=source_cursor,
                batch_order=batch_order, steps=expected_steps,
            )
            source_by_steps[expected_steps] = locator
        annual_by_key: dict[str, Mapping[str, Any]] = {}
        for kind, steps in (
            ("JX_SOURCE_ONLY_FIXED_STEP", 256),
            ("JX_SOURCE_ONLY_FIXED_STEP", 512),
            ("REBOUND_5.1.1_IAS15_TIGHT_SOURCE_ONLY", 0),
        ):
            lane_id = (
                f"jx_resolved11_source_h{steps}" if steps
                else "rebound511_ias15_tight_resolved11_source_only"
            )
            matches = [row for row in annual_lanes if type(row) is dict and row.get("lane_id") == lane_id]
            if len(matches) != 1:
                raise SourceConvergenceDiagnosticError("required annual lane roster differs")
            locator = matches[0]
            annual_cursor = _validate_annual_locator(
                locator, expected_offset=annual_cursor, batch_order=batch_order,
                kind=kind, steps=steps,
            )
            annual_by_key[lane_id] = locator
        levels = {
            64: _extract_v8_annual(
                v8_source_fd, v8_locators[(batch_order, 64)], 64, gm_values, np,
            ),
            128: _extract_v8_annual(
                v8_source_fd, v8_locators[(batch_order, 128)], 128, gm_values, np,
            ),
            256: _decode_annual_locator(
                annual_fd, annual_by_key["jx_resolved11_source_h256"], np,
            ),
            512: _decode_annual_locator(
                annual_fd, annual_by_key["jx_resolved11_source_h512"], np,
            ),
        }
        tight_ias = _decode_annual_locator(
            annual_fd, annual_by_key["rebound511_ias15_tight_resolved11_source_only"], np,
        )
        tight_offset = annual_by_key[
            "rebound511_ias15_tight_resolved11_source_only"
        ]["byte_offset"]
        if _pread_all(annual_fd, ANNUAL_RECORD_BYTES, tight_offset) != expected_centered_initial:
            raise SourceConvergenceDiagnosticError(
                "tight IAS15 initial source state differs"
            )
        retained_ias = _extract_v4_oracle_annual(
            v4_arrays_fd, oracle_locators[batch_order], batch_order, np,
        )
        pre = _pre_adjudication_diagnostic(levels, retained_ias, tight_ias)
        h1024_sources = [
            row for row in source_lanes
            if type(row) is dict and row.get("steps_per_julian_year") == 1024
        ]
        h1024_annual = [
            row for row in annual_lanes
            if type(row) is dict and row.get("lane_id") == "jx_resolved11_source_h1024"
        ]
        if pre["h1024_triggered"]:
            if len(h1024_sources) != 1 or len(h1024_annual) != 1:
                raise SourceConvergenceDiagnosticError("triggered h1024 lane is absent")
            source_locator = h1024_sources[0]
            source_cursor = _validate_source_locator(
                source_locator, expected_offset=source_cursor,
                batch_order=batch_order, steps=1024,
            )
            annual_locator = h1024_annual[0]
            annual_cursor = _validate_annual_locator(
                annual_locator, expected_offset=annual_cursor,
                batch_order=batch_order, kind="JX_SOURCE_ONLY_FIXED_STEP", steps=1024,
            )
            source_by_steps[1024] = source_locator
            annual_by_key["jx_resolved11_source_h1024"] = annual_locator
            levels[1024] = _decode_annual_locator(annual_fd, annual_locator, np)
        elif h1024_sources or h1024_annual:
            raise SourceConvergenceDiagnosticError("untriggered h1024 evidence is present")
        if len(source_lanes) != (3 if pre["h1024_triggered"] else 2):
            raise SourceConvergenceDiagnosticError("source lane count differs")
        if len(annual_lanes) != (4 if pre["h1024_triggered"] else 3):
            raise SourceConvergenceDiagnosticError("annual lane count differs")
        invariants: dict[int, Mapping[str, Any]] = {}
        for steps in RETAINED_STEPS:
            invariants[steps] = _source_invariant_summary(
                v8_source_fd, v8_locators[(batch_order, steps)], gm_values, np,
            )
        for steps, locator in source_by_steps.items():
            _require_initial_source_payload(
                source_fd, locator, expected_initial, f"new h{steps}",
            )
            if _hash_fd_range(
                source_fd, locator["byte_offset"], locator["byte_length"],
            ) != locator["lane_payload_sha256"]:
                raise SourceConvergenceDiagnosticError("new source lane digest differs")
            _crossbind_source_annual(
                source_fd=source_fd, source_locator=locator,
                annual_fd=annual_fd,
                annual_locator=annual_by_key[f"jx_resolved11_source_h{steps}"],
                gm_values=gm_values, np=np,
            )
            invariants[steps] = _source_invariant_summary(
                source_fd, locator, gm_values, np,
            )
        _validate_ias_receipt(skeleton["tight_ias15_execution_receipt"])
        diagnostic = _finalize_epoch_diagnostic(
            pre, levels, tight_ias, invariants,
        )
        result.append({
            "batch_order": batch_order, "object_id": OBJECT_IDS[batch_order],
            "native_epoch_jd_tdb_hex": float(batch.epoch_jd_tdb).hex(),
            "gm_hex_in_source_order": [value.hex() for value in gm_values],
            "source_lanes": source_lanes, "annual_lanes": annual_lanes,
            "tight_ias15_execution_receipt": skeleton["tight_ias15_execution_receipt"],
            "invariants_by_steps_per_year": {
                str(steps): invariants[steps] for steps in sorted(invariants)
            },
            "diagnostic": diagnostic,
        })
    if source_cursor != source_bytes or annual_cursor != annual_bytes:
        raise SourceConvergenceDiagnosticError("diagnostic evidence has trailing bytes")
    return result


def _current_v8_runtime_receipt(
    v8: types.ModuleType, *, cache_root: Path, before: Mapping[str, Any],
    np: types.ModuleType, scipy: types.ModuleType, cp: types.ModuleType,
    engine: types.ModuleType, backend_spec: Any,
) -> dict[str, Any]:
    try:
        return v8._runtime_receipt(
            cache_root=cache_root,
            fresh_registry=before["fresh_process_registry"],
            numpy_module=np, scipy_module=scipy, cupy_module=cp,
            engine_module=engine, backend_spec=backend_spec,
        )
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("post-execution v8 runtime receipt failed") from exc


def _artifact_receipt(relative_path: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    size = _require_exact_int(receipt.get("bytes"), f"{relative_path} bytes", minimum=0)
    digest = _require_sha(receipt.get("file_sha256"), f"{relative_path} digest")
    return {"relative_path": relative_path, "bytes": size, "file_sha256": digest}


def _execute_worker(staging_root: Path, cache_root: Path) -> dict[str, Any]:
    staging_root = Path(os.path.abspath(os.fspath(staging_root)))
    cache_root = Path(os.path.abspath(os.fspath(cache_root)))
    if not staging_root.is_dir() or staging_root.is_symlink() or any(staging_root.iterdir()):
        raise SourceConvergenceDiagnosticError("execution staging root must be an empty real directory")
    _write_create_only(staging_root / "PROTOCOL.json", protocol_bytes())
    v8, v8_result = _validate_small_prerequisites()
    try:
        np, scipy, cp, engine, backend_spec, backend, _Provenance, base_before = (
            v8._fresh_runtime_imports(cache_root)
        )
        v4, _bridge, _custody, batches, oracle_locators = v8._load_v4_context()
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("fresh source-only runtime initialization failed") from exc
    if (
        type(np) is not types.ModuleType or np.__name__ != "numpy"
        or type(cp) is not types.ModuleType or cp.__name__ != "cupy"
        or type(engine) is not types.ModuleType
        or len(batches) != BATCH_COUNT or len(oracle_locators) != BATCH_COUNT
    ):
        raise SourceConvergenceDiagnosticError("fresh source-only runtime identity differs")
    rebound_target = cache_root / "rebound511-exact-wheel"
    extraction = _extract_rebound_runtime(rebound_target)
    rebound, rebound_receipt = _load_rebound_runtime(v4, rebound_target, extraction)
    kernel = _source_kernel(cp, engine)
    source_writer = _CappedWriter(staging_root / "SOURCE_NODES.bin", SOURCE_MAXIMUM_BYTES)
    annual_writer = _CappedWriter(
        staging_root / "ANNUAL_CHECKPOINTS.bin", ANNUAL_MAXIMUM_BYTES,
    )
    skeletons: list[dict[str, Any]] = []
    source_receipt: dict[str, Any] | None = None
    annual_receipt: dict[str, Any] | None = None
    try:
        with _stable_fd(
            V8_SOURCE_NODES_PATH, "retained v8 source nodes",
            exact_bytes=V8_SOURCE_NODES_BYTES, expected_sha256=V8_SOURCE_NODES_SHA256,
        ) as v8_source_fd, _stable_fd(
            V4_ARRAYS_PATH, "retained v4 arrays",
            exact_bytes=V4_ARRAYS_BYTES, expected_sha256=V4_ARRAYS_SHA256,
        ) as v4_arrays_fd:
            v8_locators = _v8_source_locator_map(v8_result)
            for batch_order, batch in enumerate(batches):
                gm_values = tuple(
                    float(item.gm_AU3_per_julian_year2) for item in batch.massive
                )
                source_lanes: list[dict[str, Any]] = []
                annual_lanes: list[dict[str, Any]] = []
                levels = {
                    64: _extract_v8_annual(
                        v8_source_fd, v8_locators[(batch_order, 64)],
                        64, gm_values, np,
                    ),
                    128: _extract_v8_annual(
                        v8_source_fd, v8_locators[(batch_order, 128)],
                        128, gm_values, np,
                    ),
                }
                for steps in BASE_STEPS:
                    source_locator, annual_locator, centered = _build_source_lane(
                        np=np, cp=cp, engine=engine, backend=backend,
                        source_kernel=kernel, batch=batch, steps=steps,
                        source_writer=source_writer, annual_writer=annual_writer,
                    )
                    source_lanes.append(source_locator)
                    annual_lanes.append(annual_locator)
                    levels[steps] = centered
                tight_locator, tight_ias, tight_receipt = _run_tight_ias15(
                    np=np, rebound=rebound, batch=batch, annual_writer=annual_writer,
                )
                annual_lanes.append(tight_locator)
                retained_ias = _extract_v4_oracle_annual(
                    v4_arrays_fd, oracle_locators[batch_order], batch_order, np,
                )
                pre = _pre_adjudication_diagnostic(levels, retained_ias, tight_ias)
                if pre["h1024_triggered"]:
                    source_locator, annual_locator, centered = _build_source_lane(
                        np=np, cp=cp, engine=engine, backend=backend,
                        source_kernel=kernel, batch=batch, steps=1024,
                        source_writer=source_writer, annual_writer=annual_writer,
                    )
                    source_lanes.append(source_locator)
                    annual_lanes.append(annual_locator)
                    levels[1024] = centered
                skeletons.append({
                    "batch_order": batch_order, "object_id": OBJECT_IDS[batch_order],
                    "source_lanes": source_lanes, "annual_lanes": annual_lanes,
                    "tight_ias15_execution_receipt": tight_receipt,
                })
            source_receipt = source_writer.finish()
            annual_receipt = annual_writer.finish()
            with _stable_fd(
                staging_root / "SOURCE_NODES.bin", "generated source nodes",
                exact_bytes=source_receipt["bytes"],
                expected_sha256=source_receipt["file_sha256"],
            ) as source_fd, _stable_fd(
                staging_root / "ANNUAL_CHECKPOINTS.bin", "generated annual checkpoints",
                exact_bytes=annual_receipt["bytes"],
                expected_sha256=annual_receipt["file_sha256"],
            ) as annual_fd:
                replayed_batches = _replay_evidence(
                    v8=v8, v8_result=v8_result, v8_source_fd=v8_source_fd,
                    v4_arrays_fd=v4_arrays_fd,
                    source_fd=source_fd, source_bytes=source_receipt["bytes"],
                    annual_fd=annual_fd, annual_bytes=annual_receipt["bytes"],
                    skeleton_batches=skeletons, np=np,
                )
    except BaseException:
        source_writer.abort_close()
        annual_writer.abort_close()
        raise
    if source_receipt is None or annual_receipt is None:
        raise SourceConvergenceDiagnosticError("generated evidence receipts are absent")
    base_after = _current_v8_runtime_receipt(
        v8, cache_root=cache_root, before=base_before,
        np=np, scipy=scipy, cp=cp, engine=engine, backend_spec=backend_spec,
    )
    runtime = _runtime_wrapper(
        v8=v8, base_before=base_before, base_after=base_after,
        rebound_receipt=rebound_receipt, engine=engine,
    )
    runtime_payload = _canonical_json(runtime)
    if len(runtime_payload) > 10_000_000:
        raise SourceConvergenceDiagnosticError("RUNTIME.json exceeds predeclared cap")
    _write_create_only(staging_root / "RUNTIME.json", runtime_payload)
    runtime_receipt = {"bytes": len(runtime_payload), "file_sha256": _sha256(runtime_payload)}
    protocol_payload = protocol_bytes()
    disposition_counts = {name: 0 for name in DISPOSITIONS}
    triggered_count = 0
    reference_qualified_count = 0
    for batch in replayed_batches:
        diagnostic = batch["diagnostic"]
        disposition_counts[diagnostic["disposition"]] += 1
        triggered_count += int(diagnostic["h1024_triggered"] is True)
        reference_qualified_count += int(diagnostic["reference_qualified"] is True)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "SOURCE_CONVERGENCE_DIAGNOSTIC_EXECUTED",
        "protocol_content_sha256": build_protocol()["content_sha256"],
        "claim_ceiling": CLAIM_CEILING,
        "claims": {
            "source_only_dynamics_executed": True,
            "public_tracer_engine_called": False,
            "tracer_state_constructed": False,
            "stage15_qualification_established": False,
            "dynamical_classification_generated": False,
            "resonance_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
            "adaptive_rerun_authorized": False,
            "diagnostic_outcomes_generated": True,
        },
        "predecessor_v8_status": "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS",
        "batch_count": BATCH_COUNT,
        "h1024_triggered_batch_count": triggered_count,
        "reference_qualified_batch_count": reference_qualified_count,
        "disposition_counts": disposition_counts,
        "batches": replayed_batches,
        "artifacts": {
            "PROTOCOL.json": {
                "relative_path": "PROTOCOL.json", "bytes": len(protocol_payload),
                "file_sha256": _sha256(protocol_payload),
            },
            "SOURCE_NODES.bin": _artifact_receipt("SOURCE_NODES.bin", source_receipt),
            "ANNUAL_CHECKPOINTS.bin": _artifact_receipt(
                "ANNUAL_CHECKPOINTS.bin", annual_receipt,
            ),
            "RUNTIME.json": _artifact_receipt("RUNTIME.json", runtime_receipt),
        },
        "runtime_content_sha256": runtime["content_sha256"],
        "no_followup_execution_authority": True,
        "classification_generated": False,
        "science_result": False,
    }
    result["content_sha256"] = _domain_sha(RESULT_DOMAIN, result)
    result_payload = _canonical_json(result)
    if len(result_payload) > 100_000_000:
        raise SourceConvergenceDiagnosticError("RESULT.json exceeds predeclared cap")
    _write_create_only(staging_root / "RESULT.json", result_payload)
    return result


def _skeletons_from_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    batches = result.get("batches")
    if type(batches) is not list or len(batches) != BATCH_COUNT:
        raise SourceConvergenceDiagnosticError("retained diagnostic batches differ")
    return [
        {
            "batch_order": batch.get("batch_order"),
            "object_id": batch.get("object_id"),
            "source_lanes": batch.get("source_lanes"),
            "annual_lanes": batch.get("annual_lanes"),
            "tight_ias15_execution_receipt": batch.get("tight_ias15_execution_receipt"),
        }
        for batch in batches
        if type(batch) is dict
    ]


def _validate_result_shape(result: Mapping[str, Any]) -> None:
    keys = {
        "schema", "status", "protocol_content_sha256", "claim_ceiling",
        "claims", "predecessor_v8_status", "batch_count",
        "h1024_triggered_batch_count", "reference_qualified_batch_count",
        "disposition_counts", "batches", "artifacts", "runtime_content_sha256",
        "no_followup_execution_authority", "classification_generated",
        "science_result", "content_sha256",
    }
    _require_keys(result, keys, "diagnostic result")
    expected_claims = {
        "source_only_dynamics_executed": True,
        "public_tracer_engine_called": False,
        "tracer_state_constructed": False,
        "stage15_qualification_established": False,
        "dynamical_classification_generated": False,
        "resonance_classification_generated": False,
        "population_inference": False,
        "planet_nine_evidence": False,
        "science_result": False,
        "adaptive_rerun_authorized": False,
        "diagnostic_outcomes_generated": True,
    }
    counts = result.get("disposition_counts")
    if (
        type(counts) is not dict or set(counts) != set(DISPOSITIONS)
        or any(type(value) is not int or value < 0 for value in counts.values())
        or sum(counts.values()) != BATCH_COUNT
    ):
        raise SourceConvergenceDiagnosticError("diagnostic disposition counts differ")
    if (
        result["schema"] != RESULT_SCHEMA
        or result["status"] != "SOURCE_CONVERGENCE_DIAGNOSTIC_EXECUTED"
        or result["protocol_content_sha256"] != build_protocol()["content_sha256"]
        or result["claim_ceiling"] != CLAIM_CEILING
        or result["claims"] != expected_claims
        or result["predecessor_v8_status"] != "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
        or type(result["batch_count"]) is not int or result["batch_count"] != BATCH_COUNT
        or result["no_followup_execution_authority"] is not True
        or result["classification_generated"] is not False
        or result["science_result"] is not False
        or result["content_sha256"] != _domain_sha(RESULT_DOMAIN, result)
    ):
        raise SourceConvergenceDiagnosticError("diagnostic result semantics differ")


def _validate_output_worker(root: Path, cache_root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    cache_root = Path(os.path.abspath(os.fspath(cache_root)))
    if not root.is_dir() or root.is_symlink():
        raise SourceConvergenceDiagnosticError("output root is not a real directory")
    if sorted(item.name for item in root.iterdir()) != sorted(EXACT_ROSTER):
        raise SourceConvergenceDiagnosticError("output artifact roster differs")
    protocol_payload = _regular_bytes(root / "PROTOCOL.json", "output protocol", maximum=2_000_000)
    if protocol_payload != protocol_bytes():
        raise SourceConvergenceDiagnosticError("output protocol bytes differ")
    result_payload = _regular_bytes(root / "RESULT.json", "diagnostic result", maximum=100_000_000)
    result = _load_canonical_json(result_payload, "diagnostic result")
    _validate_result_shape(result)
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(EXACT_ROSTER) - {"RESULT.json"}:
        raise SourceConvergenceDiagnosticError("result artifact receipt roster differs")
    v8, v8_result = _validate_small_prerequisites()
    try:
        np, scipy, cp, engine, backend_spec, _backend, _Provenance, current_base = (
            v8._fresh_runtime_imports(cache_root)
        )
        v4, _bridge, _custody, _batches, _oracle = v8._load_v4_context()
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("fresh validation runtime failed") from exc
    rebound_target = cache_root / "rebound511-exact-wheel"
    extraction = _extract_rebound_runtime(rebound_target)
    _rebound, current_rebound = _load_rebound_runtime(v4, rebound_target, extraction)
    runtime_receipt = artifacts["RUNTIME.json"]
    source_receipt = artifacts["SOURCE_NODES.bin"]
    annual_receipt = artifacts["ANNUAL_CHECKPOINTS.bin"]
    for name, receipt in artifacts.items():
        _require_keys(receipt, {"relative_path", "bytes", "file_sha256"}, f"{name} receipt")
        if receipt["relative_path"] != name:
            raise SourceConvergenceDiagnosticError(f"{name} artifact path differs")
    if artifacts["PROTOCOL.json"] != {
        "relative_path": "PROTOCOL.json", "bytes": len(protocol_payload),
        "file_sha256": _sha256(protocol_payload),
    }:
        raise SourceConvergenceDiagnosticError("protocol artifact receipt differs")
    source_size = _require_exact_int(source_receipt["bytes"], "source bytes", minimum=1)
    annual_size = _require_exact_int(annual_receipt["bytes"], "annual bytes", minimum=1)
    runtime_size = _require_exact_int(runtime_receipt["bytes"], "runtime bytes", minimum=1)
    if source_size > SOURCE_MAXIMUM_BYTES or annual_size > ANNUAL_MAXIMUM_BYTES or runtime_size > 10_000_000:
        raise SourceConvergenceDiagnosticError("artifact exceeds protocol cap")
    runtime_payload = _regular_bytes(
        root / "RUNTIME.json", "diagnostic runtime", exact_bytes=runtime_size,
        maximum=10_000_000,
    )
    if _sha256(runtime_payload) != runtime_receipt["file_sha256"]:
        raise SourceConvergenceDiagnosticError("runtime artifact digest differs")
    retained_runtime = _load_canonical_json(runtime_payload, "diagnostic runtime")
    _validate_runtime_wrapper(
        retained_runtime, v8=v8, current_base=current_base,
        current_rebound=current_rebound, engine=engine,
    )
    if retained_runtime["content_sha256"] != result["runtime_content_sha256"]:
        raise SourceConvergenceDiagnosticError("runtime/result content cross-binding differs")
    with _stable_fd(
        V8_SOURCE_NODES_PATH, "retained v8 source nodes",
        exact_bytes=V8_SOURCE_NODES_BYTES, expected_sha256=V8_SOURCE_NODES_SHA256,
    ) as v8_source_fd, _stable_fd(
        V4_ARRAYS_PATH, "retained v4 arrays", exact_bytes=V4_ARRAYS_BYTES,
        expected_sha256=V4_ARRAYS_SHA256,
    ) as v4_arrays_fd, _stable_fd(
        root / "SOURCE_NODES.bin", "diagnostic source nodes",
        exact_bytes=source_size, expected_sha256=source_receipt["file_sha256"],
    ) as source_fd, _stable_fd(
        root / "ANNUAL_CHECKPOINTS.bin", "diagnostic annual checkpoints",
        exact_bytes=annual_size, expected_sha256=annual_receipt["file_sha256"],
    ) as annual_fd:
        replayed = _replay_evidence(
            v8=v8, v8_result=v8_result, v8_source_fd=v8_source_fd,
            v4_arrays_fd=v4_arrays_fd,
            source_fd=source_fd, source_bytes=source_size,
            annual_fd=annual_fd, annual_bytes=annual_size,
            skeleton_batches=_skeletons_from_result(result), np=np,
        )
    if _canonical_json(replayed) != _canonical_json(result["batches"]):
        raise SourceConvergenceDiagnosticError("retained diagnostic decisions do not replay")
    disposition_counts = {name: 0 for name in DISPOSITIONS}
    triggered = 0
    references = 0
    for batch in replayed:
        diagnostic = batch["diagnostic"]
        disposition_counts[diagnostic["disposition"]] += 1
        triggered += int(diagnostic["h1024_triggered"] is True)
        references += int(diagnostic["reference_qualified"] is True)
    if (
        type(result["h1024_triggered_batch_count"]) is not int
        or result["h1024_triggered_batch_count"] != triggered
        or type(result["reference_qualified_batch_count"]) is not int
        or result["reference_qualified_batch_count"] != references
        or result["disposition_counts"] != disposition_counts
    ):
        raise SourceConvergenceDiagnosticError("diagnostic aggregate differs")
    validation: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "result_file_sha256": _sha256(result_payload),
        "result_content_sha256": result["content_sha256"],
        "batch_count": BATCH_COUNT,
        "h1024_triggered_batch_count": triggered,
        "disposition_counts": disposition_counts,
        "stage15_qualification_established": False,
        "classification_generated": False,
        "science_result": False,
    }
    validation["content_sha256"] = _domain_sha(VALIDATION_DOMAIN, validation)
    return validation


def _runtime_environment(v8: types.ModuleType, cache_root: Path) -> dict[str, str]:
    try:
        return v8._runtime_child_environment(cache_root)
    except BaseException as exc:
        raise SourceConvergenceDiagnosticError("fresh-child runtime environment failed") from exc


def _run_validation_child(root: Path) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix="jx-stage15-source-diagnostic-validation-"))
    cache_root = temporary / "runtime-cache"
    receipt_path = temporary / "VALIDATION.json"
    try:
        v8 = _v8_module()
        environment = _runtime_environment(v8, cache_root)
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "validate-worker", "--output-root", str(root),
                "--cache-root", str(cache_root),
                "--validation-receipt", str(receipt_path),
            ],
            cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            check=False, timeout=604_800,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise SourceConvergenceDiagnosticError(f"fresh validation worker failed: {detail}")
        payload = _regular_bytes(receipt_path, "validation receipt", maximum=2_000_000)
        receipt = _load_canonical_json(payload, "validation receipt")
        if receipt.get("schema") != VALIDATION_SCHEMA or receipt.get("content_sha256") != _domain_sha(VALIDATION_DOMAIN, receipt):
            raise SourceConvergenceDiagnosticError("validation receipt differs")
        return receipt
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_output(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _run_validation_child(Path(os.path.abspath(os.fspath(root))))


def _publish_result_last(staging: Path, root: Path) -> None:
    if sorted(item.name for item in root.iterdir()) != ["PROTOCOL.json"]:
        raise SourceConvergenceDiagnosticError("publication root is not exactly prepared")
    for name in ("SOURCE_NODES.bin", "ANNUAL_CHECKPOINTS.bin", "RUNTIME.json"):
        source = staging / name
        destination = root / name
        try:
            os.link(source, destination, follow_symlinks=False)
            source.unlink()
        except OSError as exc:
            raise SourceConvergenceDiagnosticError(f"publication failed before RESULT: {name}") from exc
    try:
        os.link(staging / "RESULT.json", root / "RESULT.json", follow_symlinks=False)
        (staging / "RESULT.json").unlink()
    except OSError as exc:
        raise SourceConvergenceDiagnosticError("RESULT publication commit failed") from exc
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def execute(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    validate_prepared(root)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{root.name}.source-diagnostic-execute-", dir=root.parent,
    ))
    staging = temporary / "staging"
    staging.mkdir(mode=0o700)
    cache_root = temporary / "runtime-cache"
    try:
        v8 = _v8_module()
        environment = _runtime_environment(v8, cache_root)
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "execute-worker", "--staging-root", str(staging),
                "--cache-root", str(cache_root),
            ],
            cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            check=False, timeout=604_800,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise SourceConvergenceDiagnosticError(f"fresh execution worker failed: {detail}")
        _run_validation_child(staging)
        (staging / "PROTOCOL.json").unlink()
        _publish_result_last(staging, root)
        return validate_output(root)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate-prepared", "execute", "validate-output"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    worker = subparsers.add_parser("execute-worker")
    worker.add_argument("--staging-root", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    validator = subparsers.add_parser("validate-worker")
    validator.add_argument("--output-root", type=Path, required=True)
    validator.add_argument("--cache-root", type=Path, required=True)
    validator.add_argument("--validation-receipt", type=Path, required=True)
    subparsers.add_parser("protocol")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.output_root)
        elif args.command == "validate-prepared":
            value = validate_prepared(args.output_root)
        elif args.command == "execute":
            value = execute(args.output_root)
        elif args.command == "validate-output":
            value = validate_output(args.output_root)
        elif args.command == "execute-worker":
            value = _execute_worker(args.staging_root, args.cache_root)
        elif args.command == "validate-worker":
            value = _validate_output_worker(args.output_root, args.cache_root)
            _write_create_only(args.validation_receipt, _canonical_json(value))
        elif args.command == "protocol":
            sys.stdout.buffer.write(protocol_bytes() + b"\n")
            return 0
        else:  # pragma: no cover - argparse enforces this branch.
            raise SourceConvergenceDiagnosticError("unknown command")
    except SourceConvergenceDiagnosticError as exc:
        print(f"Stage-1.5 source convergence diagnostic error: {exc}", file=sys.stderr)
        return 2
    if args.command not in ("execute-worker", "validate-worker"):
        sys.stdout.buffer.write(_canonical_json(value) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
