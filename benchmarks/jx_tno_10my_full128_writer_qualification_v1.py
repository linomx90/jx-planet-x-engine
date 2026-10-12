#!/usr/bin/env python3
"""Qualify the full-128 10-Myr trajectory writer with synthetic bytes only.

This gate exercises the production shard cardinality, byte geometry, storage
reservation, crash/resume rules, held-descriptor validation, and atomic
publication surface needed by a later dynamics runner.  It never integrates an
orbit.  Its payloads are deterministic finite sentinels and therefore cannot be
used as trajectories or as scientific evidence.

The default production action is deliberately expensive (roughly 9.5 GB of
retained payload plus metadata) and is never implicit.  ``dry-metadata`` checks
the 12 x 1000 plan without writing payload bytes.  Tests use an explicitly
test-only tiny profile.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import sys
import types
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks/jx_tno_10my_full128_writer_qualification_v1.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tno_10my_full128_writer_qualification_v1.py"
GRID_SOURCE_PATH = PROJECT_ROOT / "benchmarks/jx_tno_10my_full128_trajectory_grid_v1.py"
GRID_TEST_PATH = PROJECT_ROOT / "tests/test_tno_10my_full128_trajectory_grid_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/tno_10my_full128_writer_qualification_v1"

EXPECTED_GRID_SOURCE_BYTES = 77_247
EXPECTED_GRID_SOURCE_SHA256 = "3dbc7ad7e5825b863c47b17ad0311e4986538df98d26ea7545b9fb53bb4562d0"
EXPECTED_GRID_TEST_BYTES = 22_002
EXPECTED_GRID_TEST_SHA256 = "083f68468639183697b1b2ae0df2b3c6e300817d0e8079aab7ff5a84234f9c22"

EXPECTED_BRANCH = "research/tno-10my-full128-writer-qualification-v1-preregistered-20261011"
PREREGISTRATION_TAG = "tno-10my-full128-writer-qualification-v1-preregistered-20261011"
FUTURE_RESULT_TAG = "tno-10my-full128-writer-qualification-v1-result-20261011"
FILESYSTEM_TRUST_BOUNDARY = "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"

PROTOCOL_SCHEMA = "jx.tno.10my-full128-writer-qualification.protocol.v1"
PROFILE_SCHEMA = "jx.tno.10my-full128-writer-qualification.profile.v1"
ATTEMPT_SCHEMA = "jx.tno.10my-full128-writer-qualification.attempt.v1"
GIT_CUSTODY_SCHEMA = "jx.tno.10my-full128-writer-qualification.git-preregistration-custody.v1"
FINALIZATION_PLAN_SCHEMA = "jx.tno.10my-full128-writer-qualification.finalization-plan.v1"
FINALIZATION_STORAGE_SCHEMA = "jx.tno.10my-full128-writer-qualification.finalization-storage.v1"
RECEIPT_SCHEMA = "jx.tno.10my-full128-writer-qualification.role-receipt.v1"
SHARD_SCHEMA = "jx.tno.10my-full128-writer-qualification.shard.v1"
FINAL_SCHEMA = "jx.tno.10my-full128-writer-qualification.final.v1"
DIGEST_ROSTER_SCHEMA = "jx.tno.10my-full128-writer-qualification.digest-roster-row.v1"
DIGEST_ROSTER_HEADER_SCHEMA = "jx.tno.10my-full128-writer-qualification.digest-roster-header.v1"
RESULT_SCHEMA = "jx.tno.10my-full128-writer-qualification.result.v1"

PROTOCOL_DOMAIN = b"jx.tno.10my-full128-writer-qualification.protocol.v1\0"
PROFILE_DOMAIN = b"jx.tno.10my-full128-writer-qualification.profile.v1\0"
ATTEMPT_DOMAIN = b"jx.tno.10my-full128-writer-qualification.attempt.v1\0"
FINALIZATION_PLAN_DOMAIN = b"jx.tno.10my-full128-writer-qualification.finalization-plan.v1\0"
FINALIZATION_STORAGE_DOMAIN = b"jx.tno.10my-full128-writer-qualification.finalization-storage.v1\0"
RECEIPT_DOMAIN = b"jx.tno.10my-full128-writer-qualification.role-receipt.v1\0"
SHARD_DOMAIN = b"jx.tno.10my-full128-writer-qualification.shard.v1\0"
FINAL_DOMAIN = b"jx.tno.10my-full128-writer-qualification.final.v1\0"
DIGEST_ROSTER_DOMAIN = b"jx.tno.10my-full128-writer-qualification.digest-roster-row.v1\0"
DIGEST_ROSTER_HEADER_DOMAIN = b"jx.tno.10my-full128-writer-qualification.digest-roster-header.v1\0"
RESULT_DOMAIN = b"jx.tno.10my-full128-writer-qualification.result.v1\0"
CONTINUATION_DOMAIN = b"jx.tno.10my-full128-writer-qualification.continuation.v1\0"
SOURCE_DOMAIN = b"jx.tno.10my-full128-writer-qualification.source.v1\0"
ROUTE_DOMAIN = b"jx.tno.10my-full128-writer-qualification.route.v1\0"
RUNTIME_DOMAIN = b"jx.tno.10my-full128-writer-qualification.runtime.v1\0"
FIRST_1K_DOMAIN = b"jx.tno.10my-full128-writer-qualification.first-1k.v1\0"

ZERO_SHA256 = "0" * 64
UNRESERVED_FLOOR_BYTES = 2 * 1024**3
DIGEST_ROSTER_MAX_BYTES = 100_000_000
ARCHIVE_FILE_MAX_BYTES = 100_000_000
METADATA_RESERVE_PER_SHARD = 64 * 1024
METADATA_RESERVE_FIXED = 64 * 1024 * 1024
INODE_RESERVE_FIXED = 1024
FINALIZATION_TAIL_BUDGET_BYTES = 16 * 1024 * 1024
RESULT_PUBLICATION_MARGIN_BYTES = 1 * 1024 * 1024
RECEIPT_ROLES = (
    "start_continuation",
    "end_continuation",
    "source",
    "route",
    "runtime",
)
FRAGMENT_TO_FINAL = {
    "B22_STATES.fragment": "B22_STATES.f64le",
    "B22_NODE_SHA256.fragment": "B22_NODE_SHA256.raw",
    "B22_LIFECYCLE.fragment": "B22_LIFECYCLE.u8",
    "ML_SHORT_STATES.fragment": "ML_SHORT_STATES.f64le",
    "ML_SHORT_LIFECYCLE.fragment": "ML_SHORT_LIFECYCLE.u8",
}
ARCHIVE_ROSTER = frozenset({"PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"})
SUCCESS_ROOT_ROSTER = frozenset({"PROTOCOL.json", "EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"})
PREREG_PROTOCOL_RELATIVE_PATH = "runs/tno_10my_full128_writer_qualification_v1/PROTOCOL.json"
GIT_BOUND_RELATIVE_PATHS = (
    "benchmarks/jx_tno_10my_full128_writer_qualification_v1.py",
    "tests/test_tno_10my_full128_writer_qualification_v1.py",
    PREREG_PROTOCOL_RELATIVE_PATH,
)


class WriterQualificationError(RuntimeError):
    """The synthetic writer-qualification contract failed closed."""


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise WriterQualificationError("hash payload must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise WriterQualificationError("value is not canonical finite JSON") from exc


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise WriterQualificationError("JSON has a duplicate or non-text key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WriterQualificationError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise WriterQualificationError("JSON payload must be exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WriterQualificationError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise WriterQualificationError("JSON root must be an exact object")
    return value


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(domain) is not bytes or type(value) is not dict:
        raise WriterQualificationError("domain hash arguments differ")
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WriterQualificationError(f"{label} must be an exact integer >= {minimum}")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise WriterQualificationError(f"{label} must be an exact boolean")
    return value


def _file_record(path: Path, label: str, *, maximum: int = 128_000_000) -> dict[str, object]:
    payload = _read_regular(path, label, maximum=maximum)
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "byte_length": len(payload), "sha256": _sha256(payload)}


def _load_grid() -> Any:
    def read_exact(path: Path, label: str, expected_bytes: int) -> bytes:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise WriterQualificationError(f"{label} is not a singly-linked regular file")
            if before.st_size != expected_bytes:
                raise WriterQualificationError(f"{label} byte length differs before read")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            lexical = os.stat(path, follow_symlinks=False)
            identity_before = (
                before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
                before.st_size, before.st_mtime_ns, before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns,
            )
            identity_lexical = (
                lexical.st_dev, lexical.st_ino, lexical.st_mode, lexical.st_nlink,
                lexical.st_size, lexical.st_mtime_ns, lexical.st_ctime_ns,
            )
            if identity_before != identity_after or identity_before != identity_lexical:
                raise WriterQualificationError(f"{label} changed while held")
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise WriterQualificationError(f"{label} byte count differs")
            return payload
        finally:
            os.close(fd)

    payload = read_exact(
        GRID_SOURCE_PATH,
        "pinned trajectory-grid source",
        EXPECTED_GRID_SOURCE_BYTES,
    )
    if len(payload) != EXPECTED_GRID_SOURCE_BYTES or _sha256(payload) != EXPECTED_GRID_SOURCE_SHA256:
        raise WriterQualificationError("pinned trajectory-grid source bytes differ")
    test_payload = read_exact(
        GRID_TEST_PATH,
        "pinned trajectory-grid test",
        EXPECTED_GRID_TEST_BYTES,
    )
    if len(test_payload) != EXPECTED_GRID_TEST_BYTES or _sha256(test_payload) != EXPECTED_GRID_TEST_SHA256:
        raise WriterQualificationError("pinned trajectory-grid test bytes differ")
    name = "_jx_tno_full128_trajectory_grid_v1_for_writer_qualification"
    # Compile and execute the exact bytes just hashed. Importlib's source loader
    # is intentionally not used: it may accept a timestamp-valid stale .pyc.
    code = compile(payload, os.fspath(GRID_SOURCE_PATH), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    module = types.ModuleType(name)
    module.__file__ = os.fspath(GRID_SOURCE_PATH)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


grid = _load_grid()
GridGeometry = grid.GridGeometry
PRODUCTION_GEOMETRY = grid.PRODUCTION_GEOMETRY
FRAGMENT_NAMES = tuple(grid.FRAGMENT_NAMES)
FINAL_BINARY_NAMES = tuple(grid.FINAL_BINARY_NAMES)
ACTIVE = grid.ACTIVE
REMOVED_SOURCE_RADIUS = grid.REMOVED_SOURCE_RADIUS
REMOVED_OUTER_BOUNDARY = grid.REMOVED_OUTER_BOUNDARY
_RUNTIME_PAYLOAD_CACHE: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class WriterProfile:
    name: str
    geometry: GridGeometry
    first_1k_final_step: int
    exact_production: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or len(self.name) > 64:
            raise WriterQualificationError("profile name must be bounded nonempty text")
        if type(self.geometry) is not GridGeometry:
            raise WriterQualificationError("profile geometry must be the exact pinned type")
        _require_int(self.first_1k_final_step, "first_1k_final_step")
        _require_bool(self.exact_production, "exact_production")
        if self.first_1k_final_step > self.geometry.total_steps:
            raise WriterQualificationError("first-1k endpoint exceeds profile horizon")
        if self.first_1k_final_step % self.geometry.short_interval_steps:
            raise WriterQualificationError("first-1k endpoint is off the retained grid")
        if self.exact_production is True and (
            self.name != "production-12x1000" or self.geometry != PRODUCTION_GEOMETRY
            or self.first_1k_final_step != 64_000
        ):
            raise WriterQualificationError("production profile identity differs")
        if self.exact_production is False and self.geometry == PRODUCTION_GEOMETRY:
            raise WriterQualificationError("production geometry cannot be relabeled test-only")

    @property
    def shard_total(self) -> int:
        return self.geometry.batch_count * self.geometry.shard_count

    @property
    def receipt_total(self) -> int:
        return self.shard_total * len(RECEIPT_ROLES)


PRODUCTION_PROFILE = WriterProfile("production-12x1000", PRODUCTION_GEOMETRY, 64_000, True)


def _geometry_record(geometry: GridGeometry) -> dict[str, int]:
    return {
        **asdict(geometry),
        "shard_count": geometry.shard_count,
        "body_count": geometry.body_count,
        "state_record_bytes": geometry.state_record_bytes,
        "lifecycle_record_bytes": geometry.lifecycle_record_bytes,
        "b22_node_count": geometry.b22_node_count,
        "short_node_count": geometry.short_node_count,
        "union_node_count": geometry.union_node_count,
    }


def _profile_record(profile: WriterProfile) -> dict[str, Any]:
    if type(profile) is not WriterProfile:
        raise WriterQualificationError("profile must be the exact writer-profile type")
    value: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "name": profile.name,
        "geometry": _geometry_record(profile.geometry),
        "first_1k_final_step": profile.first_1k_final_step,
        "exact_production": profile.exact_production,
        "shard_total": profile.shard_total,
        "receipt_roles": list(RECEIPT_ROLES),
        "receipt_total": profile.receipt_total,
        "content_sha256": "",
    }
    value["content_sha256"] = _domain_hash(PROFILE_DOMAIN, value)
    return value


def _require_profile(profile: WriterProfile, *, test_only: bool) -> None:
    if type(profile) is not WriterProfile:
        raise WriterQualificationError("profile must be the exact writer-profile type")
    _require_bool(test_only, "test_only")
    if profile.exact_production is True:
        if profile != PRODUCTION_PROFILE or test_only is True:
            raise WriterQualificationError("production profile authority differs")
    elif test_only is not True:
        raise WriterQualificationError("non-production profile requires explicit test-only authority")


def _round_up(value: int, unit: int) -> int:
    size = _require_int(value, "rounding value")
    block = _require_int(unit, "rounding unit", minimum=1)
    return ((size + block - 1) // block) * block


def _fragment_sizes_for_shard(profile: WriterProfile, shard_index: int) -> dict[str, int]:
    geometry = profile.geometry
    shard = _require_int(shard_index, "shard_index")
    if shard >= geometry.shard_count:
        raise WriterQualificationError("shard index exceeds profile")
    steps = geometry.expected_steps(shard)
    b22 = sum(1 for step in steps if geometry.is_b22(step))
    short = sum(1 for step in steps if geometry.is_short(step))
    return {
        "B22_STATES.fragment": b22 * geometry.state_record_bytes,
        "B22_NODE_SHA256.fragment": b22 * 32,
        "B22_LIFECYCLE.fragment": b22 * geometry.lifecycle_record_bytes,
        "ML_SHORT_STATES.fragment": short * geometry.state_record_bytes,
        "ML_SHORT_LIFECYCLE.fragment": short * geometry.lifecycle_record_bytes,
    }


def _reservation_plan(profile: WriterProfile, *, allocation_unit: int = 4096) -> dict[str, int]:
    if type(profile) is not WriterProfile:
        raise WriterQualificationError("reservation profile type differs")
    unit = _require_int(allocation_unit, "allocation_unit", minimum=1)
    final_logical = sum(profile.geometry.output_sizes().values())
    final_allocated = sum(_round_up(size, unit) for size in profile.geometry.output_sizes().values())
    per_batch_fragment_allocated = 0
    largest_shard = 0
    for shard in range(profile.geometry.shard_count):
        shard_allocated = sum(_round_up(size, unit) for size in _fragment_sizes_for_shard(profile, shard).values())
        per_batch_fragment_allocated += shard_allocated
        largest_shard = max(largest_shard, shard_allocated + METADATA_RESERVE_PER_SHARD)
    evidence_fragment_allocated = per_batch_fragment_allocated * profile.geometry.batch_count
    fixed_metadata_reserve = METADATA_RESERVE_FIXED if profile.exact_production else 1 * 1024 * 1024
    metadata_reserve = profile.shard_total * METADATA_RESERVE_PER_SHARD + fixed_metadata_reserve
    evidence_reserve = evidence_fragment_allocated + metadata_reserve
    required_inodes = profile.shard_total * (2 + len(RECEIPT_ROLES) + len(FRAGMENT_NAMES) + 1) + len(FINAL_BINARY_NAMES) + 16
    return {
        "allocation_unit": unit,
        "final_logical_bytes": final_logical,
        "final_allocated_bytes": final_allocated,
        "evidence_fragment_allocated_bytes": evidence_fragment_allocated,
        "metadata_reserve_bytes": metadata_reserve,
        "initial_evidence_reserve_bytes": evidence_reserve,
        "largest_transient_shard_bytes": largest_shard,
        "required_inodes": required_inodes,
        "required_inodes_with_margin": required_inodes + INODE_RESERVE_FIXED,
    }


def production_metadata_plan_v1() -> dict[str, Any]:
    """Return an exact 12,000-shard plan without creating payload artifacts."""
    profile = PRODUCTION_PROFILE
    identities = hashlib.sha256()
    first = None
    last = None
    count = 0
    for batch in range(profile.geometry.batch_count):
        for shard in range(profile.geometry.shard_count):
            row = {
                "batch_order": batch,
                "shard_index": shard,
                "shard_name": _shard_name(batch, shard),
                "first_global_step": profile.geometry.expected_steps(shard)[0],
                "last_global_step": profile.geometry.expected_steps(shard)[-1],
                "fragment_sizes": _fragment_sizes_for_shard(profile, shard),
            }
            encoded = _canonical_json(row)
            identities.update(encoded)
            if first is None:
                first = row
            last = row
            count += 1
    estimated_digest_roster_bytes = count * 3_000
    if (
        count != 12_000 or profile.receipt_total != 60_000
        or sum(profile.geometry.output_sizes().values()) != 4_743_277_848
        or estimated_digest_roster_bytes >= DIGEST_ROSTER_MAX_BYTES
    ):
        raise WriterQualificationError("production metadata cardinality invariant differs")
    return {
        "schema": "jx.tno.10my-full128-writer-qualification.production-metadata-plan.v1",
        "profile": _profile_record(profile),
        "shard_total": count,
        "receipt_total": profile.receipt_total,
        "final_binary_bytes": sum(profile.geometry.output_sizes().values()),
        "identity_stream_sha256": identities.hexdigest(),
        "first_shard": first,
        "last_shard": last,
        "estimated_digest_roster_bytes": estimated_digest_roster_bytes,
        "digest_roster_cap_bytes": DIGEST_ROSTER_MAX_BYTES,
        "reservation": _reservation_plan(profile),
        "claims": _claim_ceiling(False),
    }


def _claim_ceiling(production_scale_writer_qualified: bool) -> dict[str, bool]:
    qualified = _require_bool(production_scale_writer_qualified, "production_scale_writer_qualified")
    return {
        "synthetic_io_only": True,
        "production_scale_writer_qualified": qualified,
        "dynamics_executed": False,
        "ten_myr_execution_authorized": False,
        "ten_myr_trajectory_acquired": False,
        "classification_generated": False,
        "resonance_classification_generated": False,
        "science_result": False,
    }


def _test_git_custody() -> dict[str, Any]:
    return {
        "schema": GIT_CUSTODY_SCHEMA,
        "mode": "TEST_ONLY_NO_GIT_OR_PREREGISTRATION_AUTHORITY",
        "externally_preregistered": False,
    }


def _git_oid(value: object, object_format: str | None = None) -> bool:
    if type(value) is not str or any(character not in "0123456789abcdef" for character in value):
        return False
    if object_format == "sha1":
        return len(value) == 40
    if object_format == "sha256":
        return len(value) == 64
    return len(value) in (40, 64)


def _git_command(arguments: Sequence[str], *, permitted_returncodes: tuple[int, ...] = (0,)) -> tuple[int, bytes]:
    if type(arguments) not in (tuple, list) or not arguments or any(type(item) is not str or not item for item in arguments):
        raise WriterQualificationError("Git command arguments differ")
    if type(permitted_returncodes) is not tuple or not permitted_returncodes or any(type(item) is not int for item in permitted_returncodes):
        raise WriterQualificationError("Git return-code contract differs")
    completed = subprocess.run(
        ["git", "-C", os.fspath(PROJECT_ROOT), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in permitted_returncodes:
        raise WriterQualificationError(
            f"read-only Git preregistration check failed: {' '.join(arguments)}"
        )
    return completed.returncode, bytes(completed.stdout)


def _git_text(arguments: Sequence[str]) -> str:
    _, payload = _git_command(arguments)
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise WriterQualificationError("Git preregistration output is not UTF-8") from exc
    if not value or "\x00" in value or "\n" in value:
        raise WriterQualificationError("Git preregistration scalar output differs")
    return value


def _expected_git_file_records() -> dict[str, dict[str, object]]:
    protocol_payload = _canonical_json(build_protocol())
    source_payload = _read_regular(SCRIPT_PATH, "Git-bound writer source")
    test_payload = _read_regular(TEST_PATH, "Git-bound writer test")
    return {
        GIT_BOUND_RELATIVE_PATHS[0]: {
            "path": GIT_BOUND_RELATIVE_PATHS[0],
            "byte_length": len(source_payload),
            "sha256": _sha256(source_payload),
        },
        GIT_BOUND_RELATIVE_PATHS[1]: {
            "path": GIT_BOUND_RELATIVE_PATHS[1],
            "byte_length": len(test_payload),
            "sha256": _sha256(test_payload),
        },
        GIT_BOUND_RELATIVE_PATHS[2]: {
            "path": GIT_BOUND_RELATIVE_PATHS[2],
            "byte_length": len(protocol_payload),
            "sha256": _sha256(protocol_payload),
        },
    }


def _validate_git_custody_record(value: object, profile: WriterProfile) -> dict[str, Any]:
    if profile.exact_production is False:
        expected = _test_git_custody()
        if not _exact_equal(value, expected):
            raise WriterQualificationError("test-only Git custody cannot acquire production authority")
        return expected
    keys = {
        "schema", "mode", "externally_preregistered", "repository_root",
        "object_format", "branch", "head_commit", "head_tree",
        "preregistration_tag", "preregistration_tag_object", "preregistration_tag_peeled_commit",
        "annotated_tag", "tag_object_byte_length", "tag_object_payload_sha256",
        "future_result_tag", "future_result_tag_absent", "tracked_files",
        "status_porcelain_byte_length", "status_porcelain_sha256", "clean_bound_paths",
    }
    if type(value) is not dict or set(value) != keys:
        raise WriterQualificationError("production Git custody schema differs")
    object_format = value.get("object_format")
    if object_format not in ("sha1", "sha256"):
        raise WriterQualificationError("Git object format differs")
    if (
        value.get("schema") != GIT_CUSTODY_SCHEMA
        or value.get("mode") != "EXTERNALLY_PREREGISTERED_LOCAL_GIT_CUSTODY"
        or value.get("externally_preregistered") is not True
        or value.get("repository_root") != "."
        or value.get("branch") != EXPECTED_BRANCH
        or not _git_oid(value.get("head_commit"), object_format)
        or value.get("head_commit") != value.get("preregistration_tag_peeled_commit")
        or not _git_oid(value.get("head_tree"), object_format)
        or value.get("preregistration_tag") != PREREGISTRATION_TAG
        or not _git_oid(value.get("preregistration_tag_object"), object_format)
        or value.get("annotated_tag") is not True
        or type(value.get("tag_object_byte_length")) is not int or value["tag_object_byte_length"] <= 0
        or not _is_sha256(value.get("tag_object_payload_sha256"))
        or value.get("future_result_tag") != FUTURE_RESULT_TAG
        or value.get("future_result_tag_absent") is not True
        or type(value.get("status_porcelain_byte_length")) is not int
        or value.get("status_porcelain_byte_length") != 0
        or value.get("status_porcelain_sha256") != _sha256(b"")
        or value.get("clean_bound_paths") is not True
    ):
        raise WriterQualificationError("production Git custody identity/authority differs")
    files = value.get("tracked_files")
    expected_files = _expected_git_file_records()
    if type(files) is not dict or set(files) != set(GIT_BOUND_RELATIVE_PATHS):
        raise WriterQualificationError("production Git-bound file roster differs")
    for path in GIT_BOUND_RELATIVE_PATHS:
        record = files[path]
        expected = expected_files[path]
        if (
            type(record) is not dict
            or set(record) != {"path", "byte_length", "sha256", "git_blob_oid", "committed_byte_identical"}
            or record.get("path") != path
            or type(record.get("byte_length")) is not int or record["byte_length"] != expected["byte_length"]
            or record.get("sha256") != expected["sha256"]
            or not _git_oid(record.get("git_blob_oid"), object_format)
            or record.get("committed_byte_identical") is not True
        ):
            raise WriterQualificationError("production Git-bound file receipt differs")
    return value


def _git_preregistration_custody(prereg_commit: str, prereg_tag_object: str) -> dict[str, Any]:
    if not _git_oid(prereg_commit) or not _git_oid(prereg_tag_object):
        raise WriterQualificationError("external preregistration Git OIDs differ")
    top = Path(_git_text(("rev-parse", "--show-toplevel"))).resolve()
    if top != PROJECT_ROOT.resolve():
        raise WriterQualificationError("Git preregistration repository root differs")
    object_format = _git_text(("rev-parse", "--show-object-format"))
    if object_format not in ("sha1", "sha256") or not _git_oid(prereg_commit, object_format) or not _git_oid(prereg_tag_object, object_format):
        raise WriterQualificationError("external preregistration OID/object format differs")
    branch = _git_text(("symbolic-ref", "--quiet", "--short", "HEAD"))
    head = _git_text(("rev-parse", "--verify", "HEAD^{commit}"))
    tree = _git_text(("rev-parse", "--verify", "HEAD^{tree}"))
    tag_ref = f"refs/tags/{PREREGISTRATION_TAG}"
    tag_object = _git_text(("rev-parse", "--verify", f"{tag_ref}^{{tag}}"))
    tag_peeled = _git_text(("rev-parse", "--verify", f"{tag_ref}^{{commit}}"))
    tag_type = _git_text(("cat-file", "-t", tag_object))
    _, tag_payload = _git_command(("cat-file", "-p", tag_object))
    future_code, _ = _git_command(
        ("show-ref", "--verify", "--quiet", f"refs/tags/{FUTURE_RESULT_TAG}"),
        permitted_returncodes=(0, 1),
    )
    _, status = _git_command(("status", "--porcelain=v1", "--untracked-files=all", "--", *GIT_BOUND_RELATIVE_PATHS))
    if (
        branch != EXPECTED_BRANCH or head != prereg_commit or tag_object != prereg_tag_object
        or tag_peeled != prereg_commit or tag_type != "tag" or future_code != 1 or status != b""
    ):
        raise WriterQualificationError("live Git branch/tag/clean preregistration custody differs")
    expected_files = _expected_git_file_records()
    tracked: dict[str, Any] = {}
    for path in GIT_BOUND_RELATIVE_PATHS:
        _git_command(("ls-files", "--error-unmatch", "--", path))
        blob_oid = _git_text(("rev-parse", "--verify", f"{prereg_commit}:{path}"))
        _, committed = _git_command(("cat-file", "blob", blob_oid))
        live_path = PROJECT_ROOT / path
        live = _read_regular(live_path, f"live Git-bound {path}", maximum=ARCHIVE_FILE_MAX_BYTES)
        if committed != live or len(live) != expected_files[path]["byte_length"] or _sha256(live) != expected_files[path]["sha256"]:
            raise WriterQualificationError("committed/live preregistration bytes differ")
        tracked[path] = {
            **expected_files[path],
            "git_blob_oid": blob_oid,
            "committed_byte_identical": True,
        }
    record: dict[str, Any] = {
        "schema": GIT_CUSTODY_SCHEMA,
        "mode": "EXTERNALLY_PREREGISTERED_LOCAL_GIT_CUSTODY",
        "externally_preregistered": True,
        "repository_root": ".",
        "object_format": object_format,
        "branch": branch,
        "head_commit": head,
        "head_tree": tree,
        "preregistration_tag": PREREGISTRATION_TAG,
        "preregistration_tag_object": tag_object,
        "preregistration_tag_peeled_commit": tag_peeled,
        "annotated_tag": True,
        "tag_object_byte_length": len(tag_payload),
        "tag_object_payload_sha256": _sha256(tag_payload),
        "future_result_tag": FUTURE_RESULT_TAG,
        "future_result_tag_absent": True,
        "tracked_files": tracked,
        "status_porcelain_byte_length": len(status),
        "status_porcelain_sha256": _sha256(status),
        "clean_bound_paths": True,
    }
    return _validate_git_custody_record(record, PRODUCTION_PROFILE)


def _resolve_git_custody(
    profile: WriterProfile,
    *,
    externally_preregistered: bool,
    prereg_commit: str | None,
    prereg_tag_object: str | None,
) -> dict[str, Any]:
    _require_bool(externally_preregistered, "externally_preregistered")
    if profile.exact_production is False:
        if externally_preregistered is not False or prereg_commit is not None or prereg_tag_object is not None:
            raise WriterQualificationError("test profile cannot receive Git preregistration authority")
        return _test_git_custody()
    if externally_preregistered is not True or not _git_oid(prereg_commit) or not _git_oid(prereg_tag_object):
        raise WriterQualificationError("production qualification requires explicit external preregistration commit and annotated tag object")
    return _git_preregistration_custody(prereg_commit, prereg_tag_object)


def build_protocol() -> dict[str, Any]:
    source = _file_record(SCRIPT_PATH, "writer qualification source")
    test = _file_record(TEST_PATH, "writer qualification test")
    grid_source = _file_record(GRID_SOURCE_PATH, "pinned grid source")
    grid_test = _file_record(GRID_TEST_PATH, "pinned grid test")
    if (
        grid_source["byte_length"] != EXPECTED_GRID_SOURCE_BYTES
        or grid_source["sha256"] != EXPECTED_GRID_SOURCE_SHA256
        or grid_test["byte_length"] != EXPECTED_GRID_TEST_BYTES
        or grid_test["sha256"] != EXPECTED_GRID_TEST_SHA256
    ):
        raise WriterQualificationError("pinned trajectory-grid lineage differs")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREREGISTERED_SYNTHETIC_IO_WRITER_QUALIFICATION_ONLY",
        "source": source,
        "test": test,
        "pinned_trajectory_grid": {"source": grid_source, "test": grid_test},
        "profile": _profile_record(PRODUCTION_PROFILE),
        "branch": EXPECTED_BRANCH,
        "preregistration_tag": PREREGISTRATION_TAG,
        "future_result_tag": FUTURE_RESULT_TAG,
        "production_git_authority": {
            "required_cli_inputs": ["--externally-preregistered", "--prereg-commit", "--prereg-tag-object"],
            "preregistration_tag_kind": "ANNOTATED_TAG_OBJECT_WITH_PEEL_EQUAL_TO_HEAD_AND_EXTERNAL_COMMIT",
            "bound_clean_tracked_paths": list(GIT_BOUND_RELATIVE_PATHS),
            "future_result_tag_must_be_absent_before_run": True,
            "gate_order": "VERIFY_BEFORE_STORAGE_RESERVATION_OR_ANY_ATTEMPT_WRITE",
        },
        "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        "storage": {
            "reservation_api": "POSIX_FALLOCATE_PHYSICAL_SAME_ST_DEV",
            "unreserved_floor_bytes": UNRESERVED_FLOOR_BYTES,
            "failure_errnos": ["EDQUOT", "ENOSPC", "EOPNOTSUPP"],
            "reservation_formula": "FINAL_ALLOCATED_PLUS_EVIDENCE_FRAGMENT_ALLOCATED_PLUS_64K_PER_SHARD_PLUS_64M_FIXED_PLUS_2G_FLOOR_PLUS_LARGEST_TRANSIENT_SHARD",
            "required_inodes_formula": "SHARDS_X_(SHARD_DIR_PLUS_RECEIPT_DIR_PLUS_5_RECEIPTS_PLUS_5_FRAGMENTS_PLUS_MANIFEST)_PLUS_FINAL_AND_MARGIN",
            "finalization_reconciliation": "COMMIT_EXACT_PLAN_THEN_RELEASE_RESERVATION_TO_ZERO_BEFORE_STREAMS_AND_REQUIRE_FLOOR_PLUS_EXACT_STREAM_BUDGET_PLUS_16M_TAIL",
            "post_finalization_floor_check": True,
            "result_publication_margin_bytes": RESULT_PUBLICATION_MARGIN_BYTES,
            "production_plan": _reservation_plan(PRODUCTION_PROFILE),
        },
        "custody": {
            "attempt_directory": f".{DEFAULT_OUTPUT_ROOT.name}.writer-qualification-v1.partial",
            "shard_publication": "MANIFEST_LAST_THEN_LINUX_RENAMEAT2_RENAME_NOREPLACE",
            "root_publication": "EVIDENCE_THEN_FINAL_THEN_DIGEST_ROSTER_THEN_QUALIFICATION_RESULT_LAST",
            "resume_rule": "ONLY_CANONICAL_SHARDS_THAT_PASS_FULL_HELD_FD_VALIDATION_ARE_REUSED",
            "finalization_evidence": ["EVIDENCE/FINALIZATION_PLAN.json", "EVIDENCE/FINALIZATION_STORAGE.json"],
            "receipt_roles": list(RECEIPT_ROLES),
            "exact_production_receipt_count": 60_000,
        },
        "archive_surface": {
            "git_result_custody": ["PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"],
            "per_file_cap_bytes": ARCHIVE_FILE_MAX_BYTES,
            "large_payloads_excluded_from_git": ["EVIDENCE/", "FINAL/"],
            "live_revalidation_requires_local_large_payloads": True,
            "statement": "Git result custody covers the canonical summary and bounded digest/roster manifest, not the large payload bytes; full EVIDENCE and FINAL remain local retained artifacts required for live validation.",
        },
        "commands": ["prepare", "dry-metadata", "qualify", "resume", "validate", "validate-archive"],
        "claims": _claim_ceiling(False),
        "content_sha256": "",
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise WriterQualificationError("protocol must be an exact object")
    expected = build_protocol()
    if not _exact_equal(value, expected) or value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise WriterQualificationError("protocol bytes or claim ceiling differ")
    return value["content_sha256"]


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WriterQualificationError(f"{label} has a symlink or non-directory ancestor")


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


@contextmanager
def _held_directory(path: Path, label: str, *, roster: frozenset[str] | None = None) -> Iterator[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(absolute.parent, flags)
    try:
        lexical = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(lexical.st_mode):
            raise WriterQualificationError(f"{label} is not a directory")
        fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        before = os.fstat(fd)
        if _directory_identity(before) != _directory_identity(lexical):
            os.close(fd)
            raise WriterQualificationError(f"{label} was substituted before open")
        try:
            if roster is not None and frozenset(os.listdir(fd)) != roster:
                raise WriterQualificationError(f"{label} roster differs")
            yield fd
            after = os.fstat(fd)
            path_after = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            if _directory_identity(after) != _directory_identity(before) or _directory_identity(path_after) != _directory_identity(before):
                raise WriterQualificationError(f"{label} changed while held")
            if roster is not None and frozenset(os.listdir(fd)) != roster:
                raise WriterQualificationError(f"{label} final roster differs")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


@contextmanager
def _held_directory_at(parent_fd: int, name: str, label: str, *, roster: frozenset[str] | None = None) -> Iterator[int]:
    if type(parent_fd) is not int or type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise WriterQualificationError(f"{label} arguments differ")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    lexical = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(lexical.st_mode):
        raise WriterQualificationError(f"{label} is not a directory")
    fd = os.open(name, flags, dir_fd=parent_fd)
    before = os.fstat(fd)
    if _directory_identity(before) != _directory_identity(lexical):
        os.close(fd)
        raise WriterQualificationError(f"{label} was substituted")
    try:
        if roster is not None and frozenset(os.listdir(fd)) != roster:
            raise WriterQualificationError(f"{label} roster differs")
        yield fd
        after = os.fstat(fd)
        path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _directory_identity(after) != _directory_identity(before) or _directory_identity(path_after) != _directory_identity(before):
            raise WriterQualificationError(f"{label} changed while held")
        if roster is not None and frozenset(os.listdir(fd)) != roster:
            raise WriterQualificationError(f"{label} final roster differs")
    finally:
        os.close(fd)


class _HeldFile:
    def __init__(self, parent_fd: int, name: str, label: str) -> None:
        if type(name) is not str or not name or "/" in name or name in (".", ".."):
            raise WriterQualificationError(f"{label} leaf name differs")
        self.parent_fd = parent_fd
        self.name = name
        self.label = label
        self.fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(self.fd)
            raise WriterQualificationError(f"{label} is not a singly-linked regular file")
        self.before = _file_identity(metadata)
        self.size = metadata.st_size

    def pread(self, count: int, offset: int) -> bytes:
        amount = _require_int(count, "pread count")
        position = _require_int(offset, "pread offset")
        payload = os.pread(self.fd, amount, position)
        if len(payload) != amount:
            raise WriterQualificationError(f"{self.label} ended early")
        return payload

    def read_all(self, *, maximum: int = 128_000_000) -> bytes:
        if self.size > maximum:
            raise WriterQualificationError(f"{self.label} exceeds bounded read cap")
        return self.pread(self.size, 0)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < self.size:
            chunk = os.pread(self.fd, min(1 << 20, self.size - offset), offset)
            if not chunk:
                raise WriterQualificationError(f"{self.label} ended during hashing")
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    def close(self) -> None:
        try:
            after = os.fstat(self.fd)
            path_after = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
            if _file_identity(after) != self.before or _file_identity(path_after) != self.before:
                raise WriterQualificationError(f"{self.label} changed while held")
        finally:
            os.close(self.fd)


def _read_regular(path: Path, label: str, *, maximum: int = 128_000_000) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    with _held_directory(absolute.parent, f"{label} parent") as parent_fd:
        held = _HeldFile(parent_fd, absolute.name, label)
        try:
            return held.read_all(maximum=maximum)
        finally:
            held.close()


@contextmanager
def _held_file(path: Path, label: str) -> Iterator[_HeldFile]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    with _held_directory(absolute.parent, f"{label} parent") as parent_fd:
        held = _HeldFile(parent_fd, absolute.name, label)
        try:
            yield held
        finally:
            held.close()


def _write_all(fd: int, payload: bytes) -> None:
    if type(fd) is not int or type(payload) is not bytes:
        raise WriterQualificationError("write arguments differ")
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise WriterQualificationError("write made no progress")
        view = view[written:]


def _write_create_only(parent_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    if type(name) is not str or not name or "/" in name or name in (".", "..") or type(payload) is not bytes:
        raise WriterQualificationError("create-only file arguments differ")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise WriterQualificationError(f"create-only file already exists: {name}") from exc
    try:
        _write_all(fd, payload)
        os.fsync(fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != len(payload):
            raise WriterQualificationError(f"created file identity differs: {name}")
    finally:
        os.close(fd)
    os.fsync(parent_fd)
    return {"name": name, "byte_length": len(payload), "sha256": _sha256(payload)}


def _rename_noreplace(source_parent_fd: int, source_name: str, destination_parent_fd: int, destination_name: str) -> None:
    for value, label in ((source_name, "source"), (destination_name, "destination")):
        if type(value) is not str or not value or "/" in value or value in (".", ".."):
            raise WriterQualificationError(f"rename {label} leaf differs")
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise WriterQualificationError("Linux renameat2(RENAME_NOREPLACE) is unavailable") from exc
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    result = function(
        source_parent_fd, os.fsencode(source_name), destination_parent_fd,
        os.fsencode(destination_name), 1,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise WriterQualificationError("atomic no-replace publication failed") from OSError(number, os.strerror(number))
    os.fsync(source_parent_fd)
    if destination_parent_fd != source_parent_fd:
        os.fsync(destination_parent_fd)


def _write_atomic_noreplace(parent_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    """Publish canonical bytes only after the complete temporary file is durable."""
    partial = f".{name}.partial"
    record = _write_create_only(parent_fd, partial, payload)
    _rename_noreplace(parent_fd, partial, parent_fd, name)
    return {"name": name, "byte_length": record["byte_length"], "sha256": record["sha256"]}


def _mkdir_at(parent_fd: int, name: str) -> int:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise WriterQualificationError("directory leaf differs")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise WriterQualificationError(f"create-only directory already exists: {name}") from exc
    os.fsync(parent_fd)
    return os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)


def _unlink_tree_at(parent_fd: int, name: str) -> None:
    """Remove only an attempt-local incomplete directory without following links."""
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise WriterQualificationError("incomplete attempt entry is not a directory")
    child_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        for child in os.listdir(child_fd):
            child_metadata = os.stat(child, dir_fd=child_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_metadata.st_mode):
                _unlink_tree_at(child_fd, child)
            elif stat.S_ISREG(child_metadata.st_mode) and child_metadata.st_nlink == 1:
                os.unlink(child, dir_fd=child_fd)
            else:
                raise WriterQualificationError("incomplete attempt contains an unsafe entry")
        os.fsync(child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _unlink_regular_at(parent_fd: int, name: str, label: str) -> None:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WriterQualificationError(f"{label} is not a singly-linked regular file")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _shard_name(batch_order: int, shard_index: int) -> str:
    return f"shard-b{_require_int(batch_order, 'batch_order'):03d}-s{_require_int(shard_index, 'shard_index'):04d}"


def _partial_shard_name(batch_order: int, shard_index: int) -> str:
    return f".{_shard_name(batch_order, shard_index)}.partial"


def _attempt_name(output_root: Path) -> str:
    return f".{output_root.name}.writer-qualification-v1.partial"


def _sentinel_state(profile: WriterProfile, batch: int, step: int) -> bytes:
    geometry = profile.geometry
    batch_order = _require_int(batch, "batch")
    global_step = _require_int(step, "step")
    if batch_order >= geometry.batch_count or global_step > geometry.total_steps:
        raise WriterQualificationError("sentinel state identity exceeds profile")
    payload = bytearray(geometry.state_record_bytes)
    head = (
        float(batch_order + 1), float(global_step), float(geometry.body_count),
        float(geometry.tracer_count), 0.125, -0.25,
    )
    tail = (
        float(batch_order + 101), float(global_step + 1), float(geometry.total_steps),
        float(geometry.shard_steps), 0.5, -0.75,
    )
    struct.pack_into("<6d", payload, 0, *head)
    struct.pack_into("<6d", payload, len(payload) - 48, *tail)
    result = bytes(payload)
    grid.decode_state_record_v1(result, geometry)
    return result


def _sentinel_lifecycle(profile: WriterProfile, batch: int, step: int) -> bytes:
    geometry = profile.geometry
    batch_order = _require_int(batch, "batch")
    global_step = _require_int(step, "step")
    if batch_order >= geometry.batch_count or global_step > geometry.total_steps:
        raise WriterQualificationError("sentinel lifecycle identity exceeds profile")
    values: list[int] = []
    for lane in range(geometry.tracer_count):
        slot = ((lane * 17 + batch_order * 13) % 11) + 1
        threshold = min(geometry.total_steps, slot * geometry.total_steps // 12)
        if global_step < threshold:
            values.append(ACTIVE)
        else:
            values.append(REMOVED_SOURCE_RADIUS if (lane + batch_order) % 2 == 0 else REMOVED_OUTER_BOUNDARY)
    return grid.lifecycle_record_v1(values, geometry)


def _continuation_hash(profile: WriterProfile, batch: int, step: int) -> str:
    state = _sentinel_state(profile, batch, step)
    lifecycle = _sentinel_lifecycle(profile, batch, step)
    metadata = {
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "batch_order": batch,
        "global_step": step,
        "state_sha256": _sha256(state),
        "lifecycle_sha256": _sha256(lifecycle),
    }
    return _sha256(CONTINUATION_DOMAIN + _canonical_json(metadata) + state + lifecycle)


def _runtime_payload_hash(profile: WriterProfile) -> str:
    profile_sha = _profile_record(profile)["content_sha256"]
    cached = _RUNTIME_PAYLOAD_CACHE.get(profile_sha)
    if cached is not None:
        return cached
    value = {
        "profile_content_sha256": profile_sha,
        "python_implementation": sys.implementation.name,
        "python_version_hex": sys.hexversion,
        "python_optimize": sys.flags.optimize,
        "byteorder": sys.byteorder,
        "os_name": os.name,
        "grid_source_sha256": EXPECTED_GRID_SOURCE_SHA256,
        "writer_source_sha256": _sha256(_read_regular(SCRIPT_PATH, "writer runtime source")),
    }
    result = _sha256(RUNTIME_DOMAIN + _canonical_json(value))
    _RUNTIME_PAYLOAD_CACHE[profile_sha] = result
    return result


def _source_payload_hash(profile: WriterProfile, batch: int, shard: int) -> str:
    geometry = profile.geometry
    value = {
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "batch_order": batch,
        "shard_index": shard,
        "range_start_step": shard * geometry.shard_steps,
        "range_end_step": (shard + 1) * geometry.shard_steps,
        "generator": "DETERMINISTIC_FINITE_SENTINEL_V1_NO_DYNAMICS",
        "grid_source_sha256": EXPECTED_GRID_SOURCE_SHA256,
    }
    return _sha256(SOURCE_DOMAIN + _canonical_json(value))


def _route_payload_hash(profile: WriterProfile, batch: int, shard: int) -> str:
    value = {
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "batch_order": batch,
        "shard_index": shard,
        "routes": [],
        "removed_payloads_retained": True,
    }
    return _sha256(ROUTE_DOMAIN + _canonical_json(value))


def _advance_first_1k(profile: WriterProfile, batch: int, shard: int, previous: str) -> str:
    if not _is_sha256(previous):
        raise WriterQualificationError("first-1k predecessor hash differs")
    digest = hashlib.sha256()
    digest.update(FIRST_1K_DOMAIN)
    digest.update(bytes.fromhex(previous))
    digest.update(_canonical_json({"batch_order": batch, "shard_index": shard, "profile_content_sha256": _profile_record(profile)["content_sha256"]}))
    geometry = profile.geometry
    for step in geometry.expected_steps(shard):
        if step > profile.first_1k_final_step:
            continue
        state = _sentinel_state(profile, batch, step)
        lifecycle = _sentinel_lifecycle(profile, batch, step)
        digest.update(struct.pack("<QQ", batch, step))
        digest.update(state)
        digest.update(lifecycle)
    return digest.hexdigest()


def _receipt_record(
    profile: WriterProfile,
    *,
    role: str,
    batch: int,
    shard: int,
    predecessor_manifest_content_sha256: str,
    previous_role_content_sha256: str,
    first_1k_chain_sha256: str,
) -> dict[str, Any]:
    if type(role) is not str or role not in RECEIPT_ROLES:
        raise WriterQualificationError("receipt role differs")
    for value, label in (
        (predecessor_manifest_content_sha256, "manifest predecessor"),
        (previous_role_content_sha256, "role predecessor"),
        (first_1k_chain_sha256, "first-1k chain"),
    ):
        if not _is_sha256(value):
            raise WriterQualificationError(f"{label} digest differs")
    geometry = profile.geometry
    start = shard * geometry.shard_steps
    end = (shard + 1) * geometry.shard_steps
    if role == "start_continuation":
        payload_sha256 = _continuation_hash(profile, batch, start)
    elif role == "end_continuation":
        payload_sha256 = _continuation_hash(profile, batch, end)
    elif role == "source":
        payload_sha256 = _source_payload_hash(profile, batch, shard)
    elif role == "route":
        payload_sha256 = _route_payload_hash(profile, batch, shard)
    else:
        payload_sha256 = _runtime_payload_hash(profile)
    record: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "role": role,
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "batch_order": batch,
        "shard_index": shard,
        "range_start_step": start,
        "range_end_step": end,
        "predecessor_manifest_content_sha256": predecessor_manifest_content_sha256,
        "previous_role_content_sha256": previous_role_content_sha256,
        "payload_sha256": payload_sha256,
        "previous_end_continuation_payload_sha256": (
            ZERO_SHA256 if shard == 0 else _continuation_hash(profile, batch, start)
        ),
        "first_1k_chain_sha256": first_1k_chain_sha256,
        "claims": _claim_ceiling(False),
        "content_sha256": "",
    }
    record["content_sha256"] = _domain_hash(RECEIPT_DOMAIN, record)
    return record


class _DigestWriter:
    def __init__(self, directory_fd: int, name: str, *, preallocated_size: int | None = None) -> None:
        self.name = name
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        self.digest = hashlib.sha256()
        self.length = 0
        self.closed = False
        self.preallocated_size = preallocated_size
        if preallocated_size is not None:
            _physical_fallocate(self.fd, preallocated_size, name)

    def write(self, payload: bytes) -> None:
        if self.closed or type(payload) is not bytes:
            raise WriterQualificationError("writer payload differs")
        _write_all(self.fd, payload)
        self.digest.update(payload)
        self.length += len(payload)

    def pwrite(self, payload: bytes, offset: int) -> None:
        if self.closed or type(payload) is not bytes:
            raise WriterQualificationError("pwrite payload differs")
        position = _require_int(offset, "pwrite offset")
        view = memoryview(payload)
        while view:
            count = os.pwrite(self.fd, view, position)
            if count <= 0:
                raise WriterQualificationError("pwrite made no progress")
            view = view[count:]
            position += count

    def finish(self, *, hash_existing: bool = False) -> dict[str, Any]:
        if self.closed:
            raise WriterQualificationError("writer already closed")
        os.fsync(self.fd)
        metadata = os.fstat(self.fd)
        expected_size = self.preallocated_size if self.preallocated_size is not None else self.length
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != expected_size:
            raise WriterQualificationError(f"writer file identity differs: {self.name}")
        if hash_existing:
            digest = hashlib.sha256()
            offset = 0
            while offset < metadata.st_size:
                payload = os.pread(self.fd, min(1 << 20, metadata.st_size - offset), offset)
                if not payload:
                    raise WriterQualificationError("preallocated file ended during hashing")
                digest.update(payload)
                offset += len(payload)
            sha = digest.hexdigest()
        else:
            sha = self.digest.hexdigest()
        os.close(self.fd)
        self.closed = True
        return {"name": self.name, "byte_length": metadata.st_size, "sha256": sha, "st_blocks": metadata.st_blocks}

    def abort(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def _physical_fallocate(fd: int, length: int, label: str) -> dict[str, int]:
    size = _require_int(length, f"{label} fallocate length")
    if not hasattr(os, "posix_fallocate"):
        raise WriterQualificationError("posix_fallocate is unavailable")
    try:
        os.posix_fallocate(fd, 0, size)
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT, errno.EOPNOTSUPP):
            failure_name = "EOPNOTSUPP" if exc.errno == errno.EOPNOTSUPP else errno.errorcode.get(exc.errno, str(exc.errno))
            raise WriterQualificationError(f"physical reservation failed before success: {failure_name}") from exc
        raise WriterQualificationError("physical reservation failed") from exc
    os.fsync(fd)
    metadata = os.fstat(fd)
    if metadata.st_size != size or (size > 0 and metadata.st_blocks * 512 < size):
        raise WriterQualificationError(f"{label} is not physically allocated")
    return {"byte_length": size, "st_blocks": metadata.st_blocks, "allocated_bytes": metadata.st_blocks * 512}


def _statvfs_receipt(path: Path) -> dict[str, int]:
    value = os.statvfs(path)
    return {
        "f_bsize": value.f_bsize,
        "f_frsize": value.f_frsize,
        "f_blocks": value.f_blocks,
        "f_bfree": value.f_bfree,
        "f_bavail": value.f_bavail,
        "f_files": value.f_files,
        "f_ffree": value.f_ffree,
        "f_favail": value.f_favail,
        "available_bytes": value.f_bavail * value.f_frsize,
    }


def _validate_attempt_record_intrinsic(value: object, profile: WriterProfile) -> dict[str, Any]:
    keys = {
        "schema", "profile", "output_root_name", "attempt_name", "same_st_dev",
        "unreserved_floor_bytes", "reservation_plan", "statvfs_before", "statvfs_after",
        "final_file_reservations", "evidence_reservation", "git_preregistration_custody",
        "claims", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise WriterQualificationError("attempt receipt schema differs")
    root_name = value.get("output_root_name")
    if (
        value.get("schema") != ATTEMPT_SCHEMA
        or not _exact_equal(value.get("profile"), _profile_record(profile))
        or type(root_name) is not str or not root_name or "/" in root_name
        or value.get("attempt_name") != f".{root_name}.writer-qualification-v1.partial"
        or type(value.get("same_st_dev")) is not int or value["same_st_dev"] < 0
        or type(value.get("unreserved_floor_bytes")) is not int or value["unreserved_floor_bytes"] < 0
        or (profile.exact_production and value["unreserved_floor_bytes"] != UNRESERVED_FLOOR_BYTES)
        or not _exact_equal(value.get("claims"), _claim_ceiling(False))
        or value.get("content_sha256") != _domain_hash(ATTEMPT_DOMAIN, value)
    ):
        raise WriterQualificationError("attempt receipt intrinsic identity differs")
    plan = value.get("reservation_plan")
    if type(plan) is not dict or type(plan.get("allocation_unit")) is not int or not _exact_equal(
        plan, _reservation_plan(profile, allocation_unit=plan["allocation_unit"])
    ):
        raise WriterQualificationError("attempt receipt reservation plan differs")
    statvfs_keys = {
        "f_bsize", "f_frsize", "f_blocks", "f_bfree", "f_bavail",
        "f_files", "f_ffree", "f_favail", "available_bytes",
    }
    for receipt in (value.get("statvfs_before"), value.get("statvfs_after")):
        if type(receipt) is not dict or set(receipt) != statvfs_keys or any(
            type(receipt[key]) is not int or receipt[key] < 0 for key in receipt
        ) or receipt["available_bytes"] != receipt["f_bavail"] * receipt["f_frsize"]:
            raise WriterQualificationError("attempt receipt statvfs evidence differs")
    required_before = (
        plan["final_allocated_bytes"] + plan["initial_evidence_reserve_bytes"]
        + plan["largest_transient_shard_bytes"] + value["unreserved_floor_bytes"]
    )
    if (
        plan["allocation_unit"] != max(512, value["statvfs_before"]["f_frsize"])
        or value["statvfs_before"]["available_bytes"] < required_before
        or value["statvfs_before"]["f_favail"] < plan["required_inodes_with_margin"]
        or value["statvfs_after"]["available_bytes"] < value["unreserved_floor_bytes"] + plan["largest_transient_shard_bytes"]
    ):
        raise WriterQualificationError("attempt receipt storage admission evidence differs")
    reservations = value.get("final_file_reservations")
    if type(reservations) is not dict or set(reservations) != set(FINAL_BINARY_NAMES):
        raise WriterQualificationError("attempt receipt final reservation roster differs")
    for name, size in profile.geometry.output_sizes().items():
        receipt = reservations[name]
        if (
            type(receipt) is not dict or set(receipt) != {"byte_length", "st_blocks", "allocated_bytes"}
            or type(receipt.get("byte_length")) is not int or receipt["byte_length"] != size
            or type(receipt.get("st_blocks")) is not int or receipt["st_blocks"] < 0
            or type(receipt.get("allocated_bytes")) is not int
            or receipt["allocated_bytes"] != receipt["st_blocks"] * 512
            or (size > 0 and receipt["allocated_bytes"] < size)
        ):
            raise WriterQualificationError("attempt receipt final physical reservation differs")
    if sum(reservations[name]["allocated_bytes"] for name in FINAL_BINARY_NAMES) != plan["final_allocated_bytes"]:
        raise WriterQualificationError("attempt receipt final reservation sum differs")
    evidence_reservation = value.get("evidence_reservation")
    if (
        type(evidence_reservation) is not dict
        or set(evidence_reservation) != {"byte_length", "st_blocks", "allocated_bytes"}
        or evidence_reservation.get("byte_length") != plan["initial_evidence_reserve_bytes"]
        or type(evidence_reservation.get("st_blocks")) is not int
        or evidence_reservation.get("allocated_bytes") != evidence_reservation["st_blocks"] * 512
        or evidence_reservation["allocated_bytes"] < evidence_reservation["byte_length"]
    ):
        raise WriterQualificationError("attempt receipt evidence physical reservation differs")
    _validate_git_custody_record(value.get("git_preregistration_custody"), profile)
    return value


def _initialize_attempt(
    output_root: Path,
    profile: WriterProfile,
    *,
    floor_bytes: int,
    git_custody: Mapping[str, Any],
) -> Path:
    root = Path(os.path.abspath(os.fspath(output_root)))
    attempt = root.parent / _attempt_name(root)
    if attempt.exists():
        raise WriterQualificationError("attempt-local partial already exists; use explicit resume")
    parent_stat = root.parent.stat()
    root_stat = root.stat()
    if parent_stat.st_dev != root_stat.st_dev:
        raise WriterQualificationError("attempt and output root are not on the same st_dev")
    stat_before = _statvfs_receipt(root.parent)
    plan = _reservation_plan(profile, allocation_unit=max(512, stat_before["f_frsize"]))
    required_free = (
        plan["final_allocated_bytes"] + plan["initial_evidence_reserve_bytes"]
        + plan["largest_transient_shard_bytes"] + floor_bytes
    )
    if stat_before["available_bytes"] < required_free:
        raise WriterQualificationError("insufficient available bytes for physical reservation plus unreserved floor")
    if stat_before["f_favail"] < plan["required_inodes_with_margin"]:
        raise WriterQualificationError("insufficient available inodes for exact production roster plus margin")
    with _held_directory(root.parent, "output parent") as parent_fd:
        attempt_fd = _mkdir_at(parent_fd, attempt.name)
        try:
            evidence_fd = _mkdir_at(attempt_fd, "EVIDENCE")
            os.close(evidence_fd)
            final_fd = _mkdir_at(attempt_fd, "FINAL")
            try:
                final_receipts: dict[str, Any] = {}
                for name, size in profile.geometry.output_sizes().items():
                    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=final_fd)
                    try:
                        final_receipts[name] = _physical_fallocate(fd, size, name)
                    finally:
                        os.close(fd)
                os.fsync(final_fd)
            finally:
                os.close(final_fd)
            reserve_fd = os.open("RESERVATION.bin", os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=attempt_fd)
            try:
                reserve_receipt = _physical_fallocate(reserve_fd, plan["initial_evidence_reserve_bytes"], "evidence reserve")
            finally:
                os.close(reserve_fd)
            stat_after = _statvfs_receipt(attempt)
            if stat_after["available_bytes"] < floor_bytes + plan["largest_transient_shard_bytes"]:
                raise WriterQualificationError("physical reservation did not preserve the unreserved floor")
            attempt_record: dict[str, Any] = {
                "schema": ATTEMPT_SCHEMA,
                "profile": _profile_record(profile),
                "output_root_name": root.name,
                "attempt_name": attempt.name,
                "same_st_dev": root_stat.st_dev,
                "unreserved_floor_bytes": floor_bytes,
                "reservation_plan": plan,
                "statvfs_before": stat_before,
                "statvfs_after": stat_after,
                "final_file_reservations": final_receipts,
                "evidence_reservation": reserve_receipt,
                "git_preregistration_custody": dict(git_custody),
                "claims": _claim_ceiling(False),
                "content_sha256": "",
            }
            attempt_record["content_sha256"] = _domain_hash(ATTEMPT_DOMAIN, attempt_record)
            attempt_payload = _canonical_json(attempt_record)
            with _held_directory_at(attempt_fd, "EVIDENCE", "initial evidence") as evidence_fd:
                _write_create_only(evidence_fd, "ATTEMPT.json", attempt_payload)
                _write_create_only(evidence_fd, "STORAGE_RESERVATION.json", attempt_payload)
            # This top-level record is the initialization commit point and is
            # deliberately created only after all reservation evidence exists.
            _write_create_only(attempt_fd, "ATTEMPT.json", attempt_payload)
        finally:
            os.close(attempt_fd)
    return attempt


def _load_attempt(
    attempt: Path,
    output_root: Path,
    profile: WriterProfile,
    floor_bytes: int,
    expected_git_custody: Mapping[str, Any],
) -> dict[str, Any]:
    with _held_directory(attempt, "attempt root") as attempt_fd:
        held = _HeldFile(attempt_fd, "ATTEMPT.json", "attempt record")
        try:
            payload = held.read_all(maximum=4_000_000)
        finally:
            held.close()
        value = _strict_json(payload)
        _validate_attempt_record_intrinsic(value, profile)
        if payload != _canonical_json(value) or set(value) != {
            "schema", "profile", "output_root_name", "attempt_name", "same_st_dev",
            "unreserved_floor_bytes", "reservation_plan", "statvfs_before", "statvfs_after",
            "final_file_reservations", "evidence_reservation", "git_preregistration_custody",
            "claims", "content_sha256",
        }:
            raise WriterQualificationError("attempt record schema differs")
        if (
            value.get("schema") != ATTEMPT_SCHEMA
            or not _exact_equal(value.get("profile"), _profile_record(profile))
            or value.get("output_root_name") != output_root.name
            or value.get("attempt_name") != attempt.name
            or type(value.get("same_st_dev")) is not int
            or value.get("same_st_dev") != output_root.stat().st_dev
            or type(value.get("unreserved_floor_bytes")) is not int
            or value.get("unreserved_floor_bytes") != floor_bytes
            or not _exact_equal(value.get("git_preregistration_custody"), expected_git_custody)
            or value.get("content_sha256") != _domain_hash(ATTEMPT_DOMAIN, value)
            or not _exact_equal(value.get("claims"), _claim_ceiling(False))
        ):
            raise WriterQualificationError("attempt identity or claim ceiling differs")
        plan = value.get("reservation_plan")
        if type(plan) is not dict or type(plan.get("allocation_unit")) is not int:
            raise WriterQualificationError("attempt reservation plan shape differs")
        if not _exact_equal(plan, _reservation_plan(profile, allocation_unit=plan["allocation_unit"])):
            raise WriterQualificationError("attempt reservation plan reconstruction differs")
        if type(value.get("statvfs_before")) is not dict or type(value.get("statvfs_after")) is not dict:
            raise WriterQualificationError("attempt statvfs receipts differ")
        statvfs_keys = {
            "f_bsize", "f_frsize", "f_blocks", "f_bfree", "f_bavail",
            "f_files", "f_ffree", "f_favail", "available_bytes",
        }
        for receipt in (value["statvfs_before"], value["statvfs_after"]):
            if set(receipt) != statvfs_keys or any(type(receipt[key]) is not int or receipt[key] < 0 for key in receipt):
                raise WriterQualificationError("attempt statvfs receipt schema differs")
            if receipt["available_bytes"] != receipt["f_bavail"] * receipt["f_frsize"]:
                raise WriterQualificationError("attempt statvfs byte arithmetic differs")
        expected_device = value["same_st_dev"]
        if os.fstat(attempt_fd).st_dev != expected_device:
            raise WriterQualificationError("attempt root moved to another device")
        with _held_directory_at(attempt_fd, "EVIDENCE", "attempt evidence") as evidence_fd:
            if os.fstat(evidence_fd).st_dev != expected_device:
                raise WriterQualificationError("attempt evidence moved to another device")
            storage = _HeldFile(evidence_fd, "STORAGE_RESERVATION.json", "retained storage reservation")
            retained_attempt = _HeldFile(evidence_fd, "ATTEMPT.json", "retained attempt receipt")
            try:
                if storage.read_all(maximum=4_000_000) != payload or retained_attempt.read_all(maximum=4_000_000) != payload:
                    raise WriterQualificationError("retained storage reservation differs from attempt receipt")
            finally:
                storage.close()
                retained_attempt.close()
        final_reservations = value.get("final_file_reservations")
        if type(final_reservations) is not dict or set(final_reservations) != set(FINAL_BINARY_NAMES):
            raise WriterQualificationError("attempt final reservation roster differs")
        with _held_directory_at(attempt_fd, "FINAL", "attempt preallocated FINAL") as final_fd:
            if os.fstat(final_fd).st_dev != expected_device:
                raise WriterQualificationError("attempt FINAL moved to another device")
            for name in FINAL_BINARY_NAMES:
                receipt = final_reservations[name]
                if type(receipt) is not dict or set(receipt) != {"byte_length", "st_blocks", "allocated_bytes"}:
                    raise WriterQualificationError("final reservation receipt schema differs")
                held_final = _HeldFile(final_fd, name, f"reserved {name}")
                try:
                    metadata = os.fstat(held_final.fd)
                    if (
                        metadata.st_dev != expected_device or held_final.size != profile.geometry.output_sizes()[name]
                        or receipt.get("byte_length") != held_final.size
                        or type(receipt.get("st_blocks")) is not int or receipt["st_blocks"] != metadata.st_blocks
                        or receipt.get("allocated_bytes") != metadata.st_blocks * 512
                        or (held_final.size > 0 and metadata.st_blocks * 512 < held_final.size)
                    ):
                        raise WriterQualificationError("held final reservation differs from its exact receipt")
                finally:
                    held_final.close()
        # Reconstruct the only permitted reserve shrink from fully validated shards.
        completed_shards, _, _, _, committed_allocated = _existing_chain_state(attempt / "EVIDENCE", profile)
        reserve_receipt = value.get("evidence_reservation")
        if type(reserve_receipt) is not dict or set(reserve_receipt) != {"byte_length", "st_blocks", "allocated_bytes"}:
            raise WriterQualificationError("evidence reservation receipt schema differs")
        reserve = _HeldFile(attempt_fd, "RESERVATION.bin", "live evidence reservation")
        try:
            metadata = os.fstat(reserve.fd)
            expected_size = max(0, plan["initial_evidence_reserve_bytes"] - committed_allocated)
            permitted_sizes = {expected_size}
            if completed_shards > 0:
                order = completed_shards - 1
                last_batch, last_shard = divmod(order, profile.geometry.shard_count)
                with _held_directory(attempt / "EVIDENCE", "last committed allocation evidence") as live_evidence_fd:
                    with _held_directory_at(live_evidence_fd, _shard_name(last_batch, last_shard), "last committed allocation shard") as last_shard_fd:
                        last_manifest_held = _HeldFile(last_shard_fd, "MANIFEST.json", "last committed allocation manifest")
                        try:
                            last_manifest, _ = _parse_canonical_file(last_manifest_held, maximum=8_000_000)
                        finally:
                            last_manifest_held.close()
                    permitted_sizes.add(expected_size + _shard_allocation_at(live_evidence_fd, last_manifest, profile)["total_allocated_bytes"])
            finalization_plan_path = attempt / "EVIDENCE" / "FINALIZATION_PLAN.json"
            if completed_shards == profile.shard_total and finalization_plan_path.exists():
                plan_metadata = finalization_plan_path.stat(follow_symlinks=False)
                if not stat.S_ISREG(plan_metadata.st_mode) or plan_metadata.st_nlink != 1:
                    raise WriterQualificationError("finalization plan is not a singly-linked regular file")
                permitted_sizes.add(0)
            if (
                metadata.st_dev != expected_device or reserve.size not in permitted_sizes
                or reserve_receipt.get("byte_length") != plan["initial_evidence_reserve_bytes"]
                or type(reserve_receipt.get("st_blocks")) is not int
                or reserve_receipt.get("allocated_bytes") != reserve_receipt["st_blocks"] * 512
                or (reserve.size > 0 and metadata.st_blocks * 512 < reserve.size)
            ):
                raise WriterQualificationError("live evidence reservation reconstruction differs")
        finally:
            reserve.close()
        return value


def _reservation_remaining(attempt: Path, attempt_record: Mapping[str, Any], committed_allocated: int, floor_bytes: int) -> None:
    if type(attempt_record) is not dict:
        raise WriterQualificationError("attempt record type differs")
    committed = _require_int(committed_allocated, "committed allocated bytes")
    initial = attempt_record["reservation_plan"]["initial_evidence_reserve_bytes"]
    target = max(0, initial - committed)
    with _held_directory(attempt, "reservation attempt") as attempt_fd:
        fd = os.open("RESERVATION.bin", os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=attempt_fd)
        try:
            os.ftruncate(fd, target)
            os.fsync(fd)
            metadata = os.fstat(fd)
            if metadata.st_size != target or (target > 0 and metadata.st_blocks * 512 < target):
                raise WriterQualificationError("reservation reconciliation size/physical allocation differs")
        finally:
            os.close(fd)
    available = _statvfs_receipt(attempt)["available_bytes"]
    if available < floor_bytes:
        raise WriterQualificationError("unreserved floor was crossed during evidence creation")


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.exists():
        raise WriterQualificationError("prepare destination must not exist")
    root.parent.mkdir(parents=True, exist_ok=True)
    with _held_directory(root.parent, "prepare parent") as parent_fd:
        root_fd = _mkdir_at(parent_fd, root.name)
        try:
            protocol = build_protocol()
            _write_create_only(root_fd, "PROTOCOL.json", _canonical_json(protocol))
        finally:
            os.close(root_fd)
    validate_prepared(root)
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared root", roster=frozenset({"PROTOCOL.json"})) as root_fd:
        held = _HeldFile(root_fd, "PROTOCOL.json", "prepared protocol")
        try:
            payload = held.read_all(maximum=8_000_000)
        finally:
            held.close()
        value = _strict_json(payload)
        if payload != _canonical_json(value):
            raise WriterQualificationError("prepared protocol is not canonical")
        validate_protocol(value)
        return value


def _fragment_record(fd: int, name: str, digest: hashlib._Hash, length: int) -> dict[str, Any]:
    os.fsync(fd)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != length:
        raise WriterQualificationError(f"fragment identity differs: {name}")
    return {"name": name, "byte_length": length, "sha256": digest.hexdigest(), "st_blocks": metadata.st_blocks}


def _write_one_shard(
    evidence: Path,
    profile: WriterProfile,
    *,
    batch: int,
    shard: int,
    predecessor_manifest_sha: str,
    previous_role_sha: Mapping[str, str],
    previous_first_1k_sha: str,
) -> dict[str, Any]:
    geometry = profile.geometry
    canonical = _shard_name(batch, shard)
    partial = _partial_shard_name(batch, shard)
    first_1k_sha = _advance_first_1k(profile, batch, shard, previous_first_1k_sha)
    with _held_directory(evidence, "attempt evidence") as evidence_fd:
        partial_fd = _mkdir_at(evidence_fd, partial)
        try:
            receipts_fd = _mkdir_at(partial_fd, "RECEIPTS")
            receipt_locators: dict[str, Any] = {}
            try:
                for role in RECEIPT_ROLES:
                    record = _receipt_record(
                        profile, role=role, batch=batch, shard=shard,
                        predecessor_manifest_content_sha256=predecessor_manifest_sha,
                        previous_role_content_sha256=previous_role_sha[role],
                        first_1k_chain_sha256=first_1k_sha,
                    )
                    payload = _canonical_json(record)
                    file_record = _write_create_only(receipts_fd, f"{role}.json", payload)
                    receipt_locators[role] = {
                        "role": role,
                        "relative_path": f"{canonical}/RECEIPTS/{role}.json",
                        "byte_length": file_record["byte_length"],
                        "file_sha256": file_record["sha256"],
                        "content_sha256": record["content_sha256"],
                    }
            finally:
                os.close(receipts_fd)
            fragment_fds: dict[str, int] = {}
            fragment_digests: dict[str, Any] = {}
            fragment_lengths = {name: 0 for name in FRAGMENT_NAMES}
            try:
                for name in FRAGMENT_NAMES:
                    fragment_fds[name] = os.open(
                        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o600, dir_fd=partial_fd,
                    )
                    fragment_digests[name] = hashlib.sha256()
                first_lifecycle: bytes | None = None
                last_lifecycle: bytes | None = None
                b22_count = 0
                short_count = 0
                steps = geometry.expected_steps(shard)
                for step in steps:
                    state = _sentinel_state(profile, batch, step)
                    lifecycle = _sentinel_lifecycle(profile, batch, step)
                    if first_lifecycle is None:
                        first_lifecycle = lifecycle
                    last_lifecycle = lifecycle
                    if geometry.is_b22(step):
                        digest = grid.node_digest_v1(batch, step, state, lifecycle, geometry)
                        for name, payload in (
                            ("B22_STATES.fragment", state),
                            ("B22_NODE_SHA256.fragment", digest),
                            ("B22_LIFECYCLE.fragment", lifecycle),
                        ):
                            _write_all(fragment_fds[name], payload)
                            fragment_digests[name].update(payload)
                            fragment_lengths[name] += len(payload)
                        b22_count += 1
                    if geometry.is_short(step):
                        for name, payload in (
                            ("ML_SHORT_STATES.fragment", state),
                            ("ML_SHORT_LIFECYCLE.fragment", lifecycle),
                        ):
                            _write_all(fragment_fds[name], payload)
                            fragment_digests[name].update(payload)
                            fragment_lengths[name] += len(payload)
                        short_count += 1
                if first_lifecycle is None or last_lifecycle is None:
                    raise WriterQualificationError("shard has no retained nodes")
                fragments = {
                    name: _fragment_record(fragment_fds[name], name, fragment_digests[name], fragment_lengths[name])
                    for name in FRAGMENT_NAMES
                }
            finally:
                for fd in fragment_fds.values():
                    os.close(fd)
            expected_sizes = _fragment_sizes_for_shard(profile, shard)
            if any(fragments[name]["byte_length"] != expected_sizes[name] for name in FRAGMENT_NAMES):
                raise WriterQualificationError("shard fragment byte arithmetic differs")
            manifest: dict[str, Any] = {
                "schema": SHARD_SCHEMA,
                "profile_content_sha256": _profile_record(profile)["content_sha256"],
                "batch_order": batch,
                "shard_index": shard,
                "shard_name": canonical,
                "first_global_step": geometry.expected_steps(shard)[0],
                "last_global_step": geometry.expected_steps(shard)[-1],
                "predecessor_manifest_content_sha256": predecessor_manifest_sha,
                "first_1k_chain_sha256": first_1k_sha,
                "start_continuation_sha256": _continuation_hash(profile, batch, shard * geometry.shard_steps),
                "end_continuation_sha256": _continuation_hash(profile, batch, (shard + 1) * geometry.shard_steps),
                "previous_end_continuation_sha256": (
                    ZERO_SHA256 if shard == 0 else _continuation_hash(profile, batch, shard * geometry.shard_steps)
                ),
                "union_checkpoint_count": len(geometry.expected_steps(shard)),
                "b22_checkpoint_count": b22_count,
                "short_checkpoint_count": short_count,
                "first_lifecycle_hex": first_lifecycle.hex(),
                "last_lifecycle_hex": last_lifecycle.hex(),
                "receipt_locators": receipt_locators,
                "fragments": fragments,
                "claims": _claim_ceiling(False),
                "content_sha256": "",
            }
            manifest["content_sha256"] = _domain_hash(SHARD_DOMAIN, manifest)
            _write_create_only(partial_fd, "MANIFEST.json", _canonical_json(manifest))
            os.fsync(partial_fd)
        finally:
            os.close(partial_fd)
        _rename_noreplace(evidence_fd, partial, evidence_fd, canonical)
    return validate_shard(evidence, profile, batch=batch, shard=shard,
                          predecessor_manifest_sha=predecessor_manifest_sha,
                          previous_role_sha=previous_role_sha,
                          previous_first_1k_sha=previous_first_1k_sha)


def _parse_canonical_file(held: _HeldFile, *, maximum: int) -> tuple[dict[str, Any], bytes]:
    payload = held.read_all(maximum=maximum)
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise WriterQualificationError(f"{held.label} is not canonical JSON")
    return value, payload


def _validate_receipt(
    shard_fd: int,
    profile: WriterProfile,
    locator: object,
    *,
    role: str,
    batch: int,
    shard: int,
    predecessor_manifest_sha: str,
    previous_role_sha: str,
    first_1k_sha: str,
) -> dict[str, Any]:
    expected_locator_keys = {"role", "relative_path", "byte_length", "file_sha256", "content_sha256"}
    if type(locator) is not dict or set(locator) != expected_locator_keys or locator.get("role") != role:
        raise WriterQualificationError(f"{role} receipt locator differs")
    expected_path = f"{_shard_name(batch, shard)}/RECEIPTS/{role}.json"
    if locator.get("relative_path") != expected_path or type(locator.get("byte_length")) is not int:
        raise WriterQualificationError(f"{role} receipt locator path/type differs")
    if not _is_sha256(locator.get("file_sha256")) or not _is_sha256(locator.get("content_sha256")):
        raise WriterQualificationError(f"{role} receipt locator digest differs")
    with _held_directory_at(shard_fd, "RECEIPTS", "receipt directory", roster=frozenset(f"{item}.json" for item in RECEIPT_ROLES)) as receipts_fd:
        held = _HeldFile(receipts_fd, f"{role}.json", f"{role} receipt")
        try:
            value, payload = _parse_canonical_file(held, maximum=1_000_000)
        finally:
            held.close()
    expected = _receipt_record(
        profile, role=role, batch=batch, shard=shard,
        predecessor_manifest_content_sha256=predecessor_manifest_sha,
        previous_role_content_sha256=previous_role_sha,
        first_1k_chain_sha256=first_1k_sha,
    )
    if (
        locator["byte_length"] != len(payload) or locator["file_sha256"] != _sha256(payload)
        or locator["content_sha256"] != expected["content_sha256"]
        or not _exact_equal(value, expected)
    ):
        raise WriterQualificationError(f"{role} exact receipt differs")
    return value


def _validate_fragment(held: _HeldFile, record: object, expected_name: str) -> None:
    if type(record) is not dict or set(record) != {"name", "byte_length", "sha256", "st_blocks"}:
        raise WriterQualificationError(f"{expected_name} fragment record differs")
    metadata = os.fstat(held.fd)
    if (
        record.get("name") != expected_name or type(record.get("byte_length")) is not int
        or record.get("byte_length") != held.size or type(record.get("st_blocks")) is not int
        or record.get("st_blocks") != metadata.st_blocks
        or (held.size > 0 and metadata.st_blocks * 512 < held.size)
        or not _is_sha256(record.get("sha256")) or held.sha256() != record["sha256"]
    ):
        raise WriterQualificationError(f"{expected_name} fragment identity differs")


def _shard_allocation_at(evidence_fd: int, manifest: Mapping[str, Any], profile: WriterProfile) -> dict[str, int]:
    """Measure one held shard, including directories and receipt/manifest blocks."""
    if type(manifest) is not dict:
        raise WriterQualificationError("allocation manifest type differs")
    name = manifest.get("shard_name")
    if type(name) is not str:
        raise WriterQualificationError("allocation shard name differs")
    total_blocks = 0
    fragment_blocks = 0
    with _held_directory_at(evidence_fd, name, "allocation shard") as shard_fd:
        shard_metadata = os.fstat(shard_fd)
        total_blocks += shard_metadata.st_blocks
        for entry in os.listdir(shard_fd):
            metadata = os.stat(entry, dir_fd=shard_fd, follow_symlinks=False)
            if entry == "RECEIPTS":
                if not stat.S_ISDIR(metadata.st_mode):
                    raise WriterQualificationError("receipt allocation entry is not a directory")
                receipts_fd = os.open(entry, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=shard_fd)
                try:
                    total_blocks += os.fstat(receipts_fd).st_blocks
                    if frozenset(os.listdir(receipts_fd)) != frozenset(f"{role}.json" for role in RECEIPT_ROLES):
                        raise WriterQualificationError("receipt allocation roster differs")
                    for receipt_name in os.listdir(receipts_fd):
                        receipt_stat = os.stat(receipt_name, dir_fd=receipts_fd, follow_symlinks=False)
                        if not stat.S_ISREG(receipt_stat.st_mode) or receipt_stat.st_nlink != 1:
                            raise WriterQualificationError("receipt allocation entry is unsafe")
                        total_blocks += receipt_stat.st_blocks
                finally:
                    os.close(receipts_fd)
            elif entry in (*FRAGMENT_NAMES, "MANIFEST.json"):
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise WriterQualificationError("shard allocation entry is unsafe")
                total_blocks += metadata.st_blocks
                if entry in FRAGMENT_NAMES:
                    fragment_blocks += metadata.st_blocks
            else:
                raise WriterQualificationError("shard allocation roster has an unexpected entry")
    total_bytes = total_blocks * 512
    fragment_bytes = fragment_blocks * 512
    metadata_bytes = total_bytes - fragment_bytes
    if metadata_bytes < 0 or metadata_bytes > METADATA_RESERVE_PER_SHARD:
        raise WriterQualificationError("per-shard metadata allocation exceeds its frozen allowance")
    allocation_unit = max(512, os.statvfs(evidence_fd).f_frsize)
    planned_fragment = sum(_round_up(size, allocation_unit) for size in _fragment_sizes_for_shard(profile, manifest["shard_index"]).values())
    if fragment_bytes > planned_fragment:
        raise WriterQualificationError("fragment physical allocation exceeds the frozen reservation formula")
    return {
        "total_allocated_bytes": total_bytes,
        "fragment_allocated_bytes": fragment_bytes,
        "metadata_allocated_bytes": metadata_bytes,
        "metadata_allowance_bytes": METADATA_RESERVE_PER_SHARD,
    }


def validate_shard(
    evidence: Path,
    profile: WriterProfile,
    *,
    batch: int,
    shard: int,
    predecessor_manifest_sha: str,
    previous_role_sha: Mapping[str, str],
    previous_first_1k_sha: str,
) -> dict[str, Any]:
    geometry = profile.geometry
    canonical = _shard_name(batch, shard)
    expected_roster = frozenset({"RECEIPTS", "MANIFEST.json", *FRAGMENT_NAMES})
    with _held_directory(evidence, "validated evidence") as evidence_fd:
        with _held_directory_at(evidence_fd, canonical, "validated shard", roster=expected_roster) as shard_fd:
            held_manifest = _HeldFile(shard_fd, "MANIFEST.json", "shard manifest")
            try:
                manifest, manifest_payload = _parse_canonical_file(held_manifest, maximum=8_000_000)
            finally:
                held_manifest.close()
            expected_keys = {
                "schema", "profile_content_sha256", "batch_order", "shard_index", "shard_name",
                "first_global_step", "last_global_step", "predecessor_manifest_content_sha256",
                "first_1k_chain_sha256", "start_continuation_sha256", "end_continuation_sha256",
                "previous_end_continuation_sha256",
                "union_checkpoint_count", "b22_checkpoint_count", "short_checkpoint_count",
                "first_lifecycle_hex", "last_lifecycle_hex", "receipt_locators", "fragments",
                "claims", "content_sha256",
            }
            steps = geometry.expected_steps(shard)
            first_1k_sha = _advance_first_1k(profile, batch, shard, previous_first_1k_sha)
            if (
                set(manifest) != expected_keys or manifest.get("schema") != SHARD_SCHEMA
                or manifest.get("profile_content_sha256") != _profile_record(profile)["content_sha256"]
                or type(manifest.get("batch_order")) is not int or manifest.get("batch_order") != batch
                or type(manifest.get("shard_index")) is not int or manifest.get("shard_index") != shard
                or manifest.get("shard_name") != canonical
                or type(manifest.get("first_global_step")) is not int or manifest.get("first_global_step") != steps[0]
                or type(manifest.get("last_global_step")) is not int or manifest.get("last_global_step") != steps[-1]
                or manifest.get("predecessor_manifest_content_sha256") != predecessor_manifest_sha
                or manifest.get("first_1k_chain_sha256") != first_1k_sha
                or manifest.get("start_continuation_sha256") != _continuation_hash(profile, batch, shard * geometry.shard_steps)
                or manifest.get("end_continuation_sha256") != _continuation_hash(profile, batch, (shard + 1) * geometry.shard_steps)
                or manifest.get("previous_end_continuation_sha256") != (
                    ZERO_SHA256 if shard == 0 else manifest.get("start_continuation_sha256")
                )
                or manifest.get("content_sha256") != _domain_hash(SHARD_DOMAIN, manifest)
                or not _exact_equal(manifest.get("claims"), _claim_ceiling(False))
            ):
                raise WriterQualificationError("shard manifest identity/chain/claim differs")
            b22_steps = tuple(step for step in steps if geometry.is_b22(step))
            short_steps = tuple(step for step in steps if geometry.is_short(step))
            for field, expected in (
                ("union_checkpoint_count", len(steps)),
                ("b22_checkpoint_count", len(b22_steps)),
                ("short_checkpoint_count", len(short_steps)),
            ):
                if type(manifest.get(field)) is not int or manifest[field] != expected:
                    raise WriterQualificationError("shard checkpoint count differs")
            locators = manifest.get("receipt_locators")
            if type(locators) is not dict or set(locators) != set(RECEIPT_ROLES):
                raise WriterQualificationError("receipt locator roster differs")
            for role in RECEIPT_ROLES:
                _validate_receipt(
                    shard_fd, profile, locators[role], role=role, batch=batch, shard=shard,
                    predecessor_manifest_sha=predecessor_manifest_sha,
                    previous_role_sha=previous_role_sha[role], first_1k_sha=first_1k_sha,
                )
            fragments = manifest.get("fragments")
            if type(fragments) is not dict or set(fragments) != set(FRAGMENT_NAMES):
                raise WriterQualificationError("fragment roster differs")
            held_fragments: dict[str, _HeldFile] = {}
            try:
                for name in FRAGMENT_NAMES:
                    held_fragments[name] = _HeldFile(shard_fd, name, name)
                    _validate_fragment(held_fragments[name], fragments[name], name)
                expected_sizes = _fragment_sizes_for_shard(profile, shard)
                if any(held_fragments[name].size != expected_sizes[name] for name in FRAGMENT_NAMES):
                    raise WriterQualificationError("fragment size arithmetic differs")
                prior_lifecycle: tuple[int, ...] | None = None
                first_lifecycle: bytes | None = None
                last_lifecycle: bytes | None = None
                b22_index = 0
                short_index = 0
                for step in steps:
                    expected_state = _sentinel_state(profile, batch, step)
                    expected_lifecycle = _sentinel_lifecycle(profile, batch, step)
                    if geometry.is_b22(step):
                        state = held_fragments["B22_STATES.fragment"].pread(geometry.state_record_bytes, b22_index * geometry.state_record_bytes)
                        lifecycle = held_fragments["B22_LIFECYCLE.fragment"].pread(geometry.lifecycle_record_bytes, b22_index * geometry.lifecycle_record_bytes)
                        digest = held_fragments["B22_NODE_SHA256.fragment"].pread(32, b22_index * 32)
                        if state != expected_state or lifecycle != expected_lifecycle or digest != grid.node_digest_v1(batch, step, state, lifecycle, geometry):
                            raise WriterQualificationError("B22 sentinel/node replay differs")
                        b22_index += 1
                    if geometry.is_short(step):
                        short_state = held_fragments["ML_SHORT_STATES.fragment"].pread(geometry.state_record_bytes, short_index * geometry.state_record_bytes)
                        short_lifecycle = held_fragments["ML_SHORT_LIFECYCLE.fragment"].pread(geometry.lifecycle_record_bytes, short_index * geometry.lifecycle_record_bytes)
                        if short_state != expected_state or short_lifecycle != expected_lifecycle:
                            raise WriterQualificationError("short-grid sentinel replay differs")
                        state, lifecycle = short_state, short_lifecycle
                        short_index += 1
                    decoded = grid.decode_lifecycle_record_v1(lifecycle, geometry)
                    grid._require_terminal_transition(prior_lifecycle, decoded)
                    prior_lifecycle = decoded
                    if first_lifecycle is None:
                        first_lifecycle = lifecycle
                    last_lifecycle = lifecycle
                if (
                    first_lifecycle is None or last_lifecycle is None
                    or manifest.get("first_lifecycle_hex") != first_lifecycle.hex()
                    or manifest.get("last_lifecycle_hex") != last_lifecycle.hex()
                ):
                    raise WriterQualificationError("lifecycle endpoint evidence differs")
            finally:
                for held in held_fragments.values():
                    held.close()
            if manifest_payload != _canonical_json(manifest):
                raise WriterQualificationError("shard manifest changed during validation")
            return manifest


def _existing_chain_state(evidence: Path, profile: WriterProfile) -> tuple[int, dict[int, str], dict[int, dict[str, str]], dict[int, str], int]:
    completed = 0
    predecessor = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
    role_chain = {batch: {role: ZERO_SHA256 for role in RECEIPT_ROLES} for batch in range(profile.geometry.batch_count)}
    first_1k_chain = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
    committed_allocated = 0
    with _held_directory(evidence, "resume evidence") as evidence_fd:
        names = frozenset(os.listdir(evidence_fd))
        custody_names = {"ATTEMPT.json", "STORAGE_RESERVATION.json"}
        finalization_names = {
            "FINALIZATION_PLAN.json",
            "FINALIZATION_STORAGE.json",
            ".FINALIZATION_PLAN.json.partial",
            ".FINALIZATION_STORAGE.json.partial",
        }
        if not custody_names.issubset(names):
            raise WriterQualificationError("retained attempt/storage reservation is absent")
        names = names - custody_names
        retained_finalization_names = names.intersection(finalization_names)
        names = names - finalization_names
        canonical_names = {
            _shard_name(batch, shard)
            for batch in range(profile.geometry.batch_count)
            for shard in range(profile.geometry.shard_count)
        }
        partial_names = {
            _partial_shard_name(batch, shard)
            for batch in range(profile.geometry.batch_count)
            for shard in range(profile.geometry.shard_count)
        }
        if not names.issubset(canonical_names | partial_names):
            raise WriterQualificationError("resume evidence has an unexpected entry")
        for batch in range(profile.geometry.batch_count):
            for shard in range(profile.geometry.shard_count):
                canonical = _shard_name(batch, shard)
                partial = _partial_shard_name(batch, shard)
                if canonical in names and partial in names:
                    raise WriterQualificationError("resume evidence has both committed and partial forms of one shard")
                if partial in names:
                    _unlink_tree_at(evidence_fd, partial)
                    names = frozenset(os.listdir(evidence_fd)) - custody_names
                if canonical not in names:
                    if retained_finalization_names:
                        raise WriterQualificationError("finalization evidence exists before the complete shard set")
                    expected_remaining = {
                        _shard_name(later_batch, later_shard)
                        for later_batch in range(batch, profile.geometry.batch_count)
                        for later_shard in range(profile.geometry.shard_count)
                        if later_batch > batch or later_shard >= shard
                    }
                    if names.intersection(expected_remaining):
                        raise WriterQualificationError("resume evidence has a non-prefix canonical shard set")
                    if names.intersection(partial_names):
                        raise WriterQualificationError("resume evidence has a future partial shard")
                    return completed, predecessor, role_chain, first_1k_chain, committed_allocated
                manifest = validate_shard(
                    evidence, profile, batch=batch, shard=shard,
                    predecessor_manifest_sha=predecessor[batch],
                    previous_role_sha=role_chain[batch],
                    previous_first_1k_sha=first_1k_chain[batch],
                )
                predecessor[batch] = manifest["content_sha256"]
                first_1k_chain[batch] = manifest["first_1k_chain_sha256"]
                for role in RECEIPT_ROLES:
                    role_chain[batch][role] = manifest["receipt_locators"][role]["content_sha256"]
                committed_allocated += _shard_allocation_at(evidence_fd, manifest, profile)["total_allocated_bytes"]
                completed += 1
        if names != canonical_names:
            raise WriterQualificationError("resume evidence roster has an unexpected entry")
        for partial_name in (
            ".FINALIZATION_PLAN.json.partial",
            ".FINALIZATION_STORAGE.json.partial",
        ):
            if partial_name in retained_finalization_names:
                _unlink_regular_at(
                    evidence_fd,
                    partial_name,
                    "incomplete finalization evidence",
                )
    return completed, predecessor, role_chain, first_1k_chain, committed_allocated


def _write_all_shards(
    attempt: Path,
    output_root: Path,
    profile: WriterProfile,
    attempt_record: Mapping[str, Any],
    *,
    floor_bytes: int,
    fault_after_shards: int | None,
) -> None:
    evidence = attempt / "EVIDENCE"
    completed, predecessor, role_chain, first_1k_chain, committed = _existing_chain_state(evidence, profile)
    finalization_plan_exists = (evidence / "FINALIZATION_PLAN.json").exists()
    if finalization_plan_exists:
        if completed != profile.shard_total:
            raise WriterQualificationError("finalization plan exists before all shards are committed")
    else:
        _reservation_remaining(attempt, attempt_record, committed, floor_bytes)
    if fault_after_shards is not None:
        limit = _require_int(fault_after_shards, "fault_after_shards", minimum=1)
        if completed >= limit:
            raise WriterQualificationError("injected crash after committed shard")
    for batch in range(profile.geometry.batch_count):
        for shard in range(profile.geometry.shard_count):
            order = batch * profile.geometry.shard_count + shard
            if order < completed:
                continue
            manifest = _write_one_shard(
                evidence, profile, batch=batch, shard=shard,
                predecessor_manifest_sha=predecessor[batch],
                previous_role_sha=role_chain[batch],
                previous_first_1k_sha=first_1k_chain[batch],
            )
            predecessor[batch] = manifest["content_sha256"]
            first_1k_chain[batch] = manifest["first_1k_chain_sha256"]
            for role in RECEIPT_ROLES:
                role_chain[batch][role] = manifest["receipt_locators"][role]["content_sha256"]
            with _held_directory(evidence, "allocation evidence") as evidence_fd:
                committed += _shard_allocation_at(evidence_fd, manifest, profile)["total_allocated_bytes"]
            completed += 1
            _reservation_remaining(attempt, attempt_record, committed, floor_bytes)
            if fault_after_shards is not None and completed >= fault_after_shards:
                raise WriterQualificationError("injected crash after committed shard")


def _copy_held_to_fd(source: _HeldFile, destination_fd: int, destination_offset: int) -> int:
    offset = 0
    while offset < source.size:
        payload = source.pread(min(1 << 20, source.size - offset), offset)
        position = destination_offset + offset
        view = memoryview(payload)
        while view:
            count = os.pwrite(destination_fd, view, position)
            if count <= 0:
                raise WriterQualificationError("final pwrite made no progress")
            view = view[count:]
            position += count
        offset += len(payload)
    return source.size


def _digest_roster_row(manifest: Mapping[str, Any], manifest_payload: bytes) -> dict[str, Any]:
    if type(manifest) is not dict or type(manifest_payload) is not bytes:
        raise WriterQualificationError("digest roster source differs")
    row: dict[str, Any] = {
        "schema": DIGEST_ROSTER_SCHEMA,
        "batch_order": manifest["batch_order"],
        "shard_index": manifest["shard_index"],
        "shard_name": manifest["shard_name"],
        "manifest_byte_length": len(manifest_payload),
        "manifest_file_sha256": _sha256(manifest_payload),
        "manifest_content_sha256": manifest["content_sha256"],
        "predecessor_manifest_content_sha256": manifest["predecessor_manifest_content_sha256"],
        "first_1k_chain_sha256": manifest["first_1k_chain_sha256"],
        "fragment_sha256": {name: manifest["fragments"][name]["sha256"] for name in FRAGMENT_NAMES},
        "receipt_file_sha256": {role: manifest["receipt_locators"][role]["file_sha256"] for role in RECEIPT_ROLES},
        "receipt_content_sha256": {role: manifest["receipt_locators"][role]["content_sha256"] for role in RECEIPT_ROLES},
        "content_sha256": "",
    }
    row["content_sha256"] = _domain_hash(DIGEST_ROSTER_DOMAIN, row)
    return row


def _retained_attempt_custody(evidence: Path, profile: WriterProfile) -> tuple[dict[str, Any], bytes]:
    with _held_directory(evidence, "custody evidence") as evidence_fd:
        attempt_held = _HeldFile(evidence_fd, "ATTEMPT.json", "retained ATTEMPT")
        storage_held = _HeldFile(evidence_fd, "STORAGE_RESERVATION.json", "retained storage reservation")
        try:
            attempt_value, attempt_payload = _parse_canonical_file(attempt_held, maximum=4_000_000)
            storage_payload = storage_held.read_all(maximum=4_000_000)
        finally:
            attempt_held.close()
            storage_held.close()
    if storage_payload != attempt_payload:
        raise WriterQualificationError("retained ATTEMPT and storage reservation bytes differ")
    _validate_attempt_record_intrinsic(attempt_value, profile)
    if (
        attempt_value.get("schema") != ATTEMPT_SCHEMA
        or not _exact_equal(attempt_value.get("profile"), _profile_record(profile))
        or attempt_value.get("content_sha256") != _domain_hash(ATTEMPT_DOMAIN, attempt_value)
        or not _exact_equal(attempt_value.get("claims"), _claim_ceiling(False))
    ):
        raise WriterQualificationError("retained attempt/storage custody differs")
    locator = {
        "attempt_path": "EVIDENCE/ATTEMPT.json",
        "storage_reservation_path": "EVIDENCE/STORAGE_RESERVATION.json",
        "byte_length_each": len(attempt_payload),
        "file_sha256_each": _sha256(attempt_payload),
        "content_sha256_each": attempt_value["content_sha256"],
    }
    return locator, attempt_payload


def _retained_finalization_records(
    evidence: Path,
    profile: WriterProfile,
    attempt_record: Mapping[str, Any],
    allocation_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _held_directory(evidence, "finalization reconciliation evidence") as evidence_fd:
        plan_held = _HeldFile(evidence_fd, "FINALIZATION_PLAN.json", "retained finalization plan")
        storage_held = _HeldFile(
            evidence_fd,
            "FINALIZATION_STORAGE.json",
            "retained finalization storage reconciliation",
        )
        try:
            plan, _ = _parse_canonical_file(plan_held, maximum=2_000_000)
            reconciliation, _ = _parse_canonical_file(storage_held, maximum=2_000_000)
        finally:
            plan_held.close()
            storage_held.close()
    if type(plan.get("shard_manifests_logical_bytes")) is not int or type(plan.get("digest_roster_logical_bytes")) is not int:
        raise WriterQualificationError("retained finalization plan lengths differ")
    expected_plan = _finalization_plan_record(
        profile,
        attempt_record,
        allocation_summary,
        shard_manifests_bytes=plan["shard_manifests_logical_bytes"],
        digest_roster_bytes=plan["digest_roster_logical_bytes"],
    )
    _validate_finalization_plan_record(plan, expected_plan)
    _validate_finalization_storage_record(reconciliation, plan)
    return plan, reconciliation


def _evidence_allocation_summary(evidence: Path, profile: WriterProfile) -> dict[str, int]:
    totals = {"total_allocated_bytes": 0, "fragment_allocated_bytes": 0, "metadata_allocated_bytes": 0}
    maximum_metadata = 0
    with _held_directory(evidence, "allocation-summary evidence") as evidence_fd:
        for batch in range(profile.geometry.batch_count):
            for shard in range(profile.geometry.shard_count):
                with _held_directory_at(evidence_fd, _shard_name(batch, shard), "allocation-summary shard") as shard_fd:
                    held = _HeldFile(shard_fd, "MANIFEST.json", "allocation-summary manifest")
                    try:
                        manifest, _ = _parse_canonical_file(held, maximum=8_000_000)
                    finally:
                        held.close()
                row = _shard_allocation_at(evidence_fd, manifest, profile)
                for key in totals:
                    totals[key] += row[key]
                maximum_metadata = max(maximum_metadata, row["metadata_allocated_bytes"])
    return {
        **totals,
        "maximum_shard_metadata_allocated_bytes": maximum_metadata,
        "per_shard_metadata_allowance_bytes": METADATA_RESERVE_PER_SHARD,
        "total_metadata_allowance_bytes": profile.shard_total * METADATA_RESERVE_PER_SHARD,
        "shard_count": profile.shard_total,
    }


def _digest_roster_header(
    profile: WriterProfile,
    attempt_custody: Mapping[str, Any],
    allocation_summary: Mapping[str, Any],
    attempt_record: Mapping[str, Any],
) -> dict[str, Any]:
    if type(attempt_custody) is not dict or type(allocation_summary) is not dict or type(attempt_record) is not dict:
        raise WriterQualificationError("digest roster header inputs differ")
    header: dict[str, Any] = {
        "schema": DIGEST_ROSTER_HEADER_SCHEMA,
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "attempt_and_storage_custody": dict(attempt_custody),
        "attempt_record": dict(attempt_record),
        "evidence_allocation_summary": dict(allocation_summary),
        "shard_record_count": profile.shard_total,
        "role_receipt_count": profile.receipt_total,
        "content_sha256": "",
    }
    header["content_sha256"] = _domain_hash(DIGEST_ROSTER_HEADER_DOMAIN, header)
    return header


def _finalization_stream_lengths(
    evidence_fd: int,
    profile: WriterProfile,
    header: Mapping[str, Any],
) -> tuple[int, int]:
    shard_bytes = 0
    digest_bytes = len(_canonical_json(header))
    for batch in range(profile.geometry.batch_count):
        for shard in range(profile.geometry.shard_count):
            with _held_directory_at(
                evidence_fd,
                _shard_name(batch, shard),
                "finalization-plan shard",
            ) as shard_fd:
                held = _HeldFile(shard_fd, "MANIFEST.json", "finalization-plan manifest")
                try:
                    manifest, payload = _parse_canonical_file(held, maximum=8_000_000)
                finally:
                    held.close()
            shard_bytes += len(payload)
            digest_bytes += len(_canonical_json(_digest_roster_row(manifest, payload)))
    if digest_bytes >= DIGEST_ROSTER_MAX_BYTES:
        raise WriterQualificationError("planned digest roster reaches its Git archive cap")
    return shard_bytes, digest_bytes


def _finalization_plan_record(
    profile: WriterProfile,
    attempt_record: Mapping[str, Any],
    allocation_summary: Mapping[str, Any],
    *,
    shard_manifests_bytes: int,
    digest_roster_bytes: int,
) -> dict[str, Any]:
    shard_bytes = _require_int(shard_manifests_bytes, "planned shard-manifest bytes", minimum=1)
    digest_bytes = _require_int(digest_roster_bytes, "planned digest-roster bytes", minimum=1)
    if type(attempt_record) is not dict or type(allocation_summary) is not dict:
        raise WriterQualificationError("finalization plan inputs differ")
    allocation_unit = attempt_record["reservation_plan"]["allocation_unit"]
    expected_remaining = (
        attempt_record["reservation_plan"]["initial_evidence_reserve_bytes"]
        - allocation_summary["total_allocated_bytes"]
    )
    if expected_remaining < 0:
        raise WriterQualificationError("committed evidence exceeded its physical reservation")
    shard_budget = _round_up(shard_bytes, allocation_unit)
    digest_budget = _round_up(digest_bytes, allocation_unit)
    stream_budget = shard_budget + digest_budget
    record: dict[str, Any] = {
        "schema": FINALIZATION_PLAN_SCHEMA,
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "attempt_content_sha256": attempt_record["content_sha256"],
        "evidence_allocation_summary_sha256": _sha256(_canonical_json(dict(allocation_summary))),
        "allocation_unit": allocation_unit,
        "shard_manifests_logical_bytes": shard_bytes,
        "digest_roster_logical_bytes": digest_bytes,
        "shard_manifests_allocation_budget_bytes": shard_budget,
        "digest_roster_allocation_budget_bytes": digest_budget,
        "stream_allocation_budget_bytes": stream_budget,
        "tail_allocation_budget_bytes": FINALIZATION_TAIL_BUDGET_BYTES,
        "total_finalization_budget_bytes": stream_budget + FINALIZATION_TAIL_BUDGET_BYTES,
        "expected_reservation_byte_length_before_release": expected_remaining,
        "unreserved_floor_bytes": attempt_record["unreserved_floor_bytes"],
        "claims": _claim_ceiling(False),
        "content_sha256": "",
    }
    record["content_sha256"] = _domain_hash(FINALIZATION_PLAN_DOMAIN, record)
    return record


def _validate_finalization_plan_record(
    value: object,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or type(expected) is not dict or not _exact_equal(value, expected):
        raise WriterQualificationError("finalization reservation plan differs")
    if value.get("content_sha256") != _domain_hash(FINALIZATION_PLAN_DOMAIN, value):
        raise WriterQualificationError("finalization reservation plan hash differs")
    return value


def _validate_statvfs_snapshot(value: object, label: str) -> dict[str, int]:
    keys = {
        "f_bsize", "f_frsize", "f_blocks", "f_bfree", "f_bavail",
        "f_files", "f_ffree", "f_favail", "available_bytes",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or any(type(value[key]) is not int or value[key] < 0 for key in value)
        or value["available_bytes"] != value["f_bavail"] * value["f_frsize"]
    ):
        raise WriterQualificationError(f"{label} statvfs receipt differs")
    return value


def _release_reservation_for_finalization(
    attempt_fd: int,
    attempt: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    before_statvfs = _statvfs_receipt(attempt)
    fd = os.open(
        "RESERVATION.bin",
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=attempt_fd,
    )
    try:
        before = os.fstat(fd)
        expected = plan["expected_reservation_byte_length_before_release"]
        if before.st_size not in (expected, 0):
            raise WriterQualificationError("finalization reservation has an unrecognized release state")
        resumed = before.st_size == 0
        if before.st_size > 0 and before.st_blocks * 512 < before.st_size:
            raise WriterQualificationError("finalization reservation is not physically allocated")
        if before.st_size != 0:
            os.ftruncate(fd, 0)
        os.fsync(fd)
        after = os.fstat(fd)
        if after.st_size != 0 or after.st_blocks != 0:
            raise WriterQualificationError("finalization reservation did not release all physical blocks")
    finally:
        os.close(fd)
    os.fsync(attempt_fd)
    after_statvfs = _statvfs_receipt(attempt)
    if (
        before_statvfs["available_bytes"] < plan["unreserved_floor_bytes"]
        or after_statvfs["available_bytes"]
        < plan["unreserved_floor_bytes"] + plan["total_finalization_budget_bytes"]
    ):
        raise WriterQualificationError("released finalization reservation cannot preserve the unreserved floor")
    return {
        "reservation_byte_length_before_release": before.st_size,
        "reservation_st_blocks_before_release": before.st_blocks,
        "reservation_allocated_bytes_before_release": before.st_blocks * 512,
        "release_was_already_committed_on_resume": resumed,
        "reservation_byte_length_after_release": after.st_size,
        "reservation_st_blocks_after_release": after.st_blocks,
        "reservation_allocated_bytes_after_release": after.st_blocks * 512,
        "statvfs_before_release": before_statvfs,
        "statvfs_after_release": after_statvfs,
    }


def _validate_finalization_storage_record(
    value: object,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    keys = {
        "schema", "plan_content_sha256", "reservation_byte_length_before_release",
        "reservation_st_blocks_before_release", "reservation_allocated_bytes_before_release",
        "release_was_already_committed_on_resume", "reservation_byte_length_after_release",
        "reservation_st_blocks_after_release", "reservation_allocated_bytes_after_release",
        "statvfs_before_release", "statvfs_after_release", "statvfs_after_streams",
        "streams", "unreserved_floor_preserved", "claims", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise WriterQualificationError("finalization storage reconciliation schema differs")
    expected_before = plan["expected_reservation_byte_length_before_release"]
    observed_before = value.get("reservation_byte_length_before_release")
    resumed = value.get("release_was_already_committed_on_resume")
    if (
        value.get("schema") != FINALIZATION_STORAGE_SCHEMA
        or value.get("plan_content_sha256") != plan["content_sha256"]
        or type(observed_before) is not int or observed_before not in (expected_before, 0)
        or type(value.get("reservation_st_blocks_before_release")) is not int
        or value["reservation_st_blocks_before_release"] < 0
        or type(value.get("reservation_allocated_bytes_before_release")) is not int
        or value["reservation_allocated_bytes_before_release"]
        != value["reservation_st_blocks_before_release"] * 512
        or type(resumed) is not bool or resumed != (observed_before == 0)
        or (observed_before > 0 and value["reservation_allocated_bytes_before_release"] < observed_before)
        or value.get("reservation_byte_length_after_release") != 0
        or value.get("reservation_st_blocks_after_release") != 0
        or value.get("reservation_allocated_bytes_after_release") != 0
        or value.get("unreserved_floor_preserved") is not True
        or not _exact_equal(value.get("claims"), _claim_ceiling(False))
        or value.get("content_sha256") != _domain_hash(FINALIZATION_STORAGE_DOMAIN, value)
    ):
        raise WriterQualificationError("finalization storage reconciliation identity differs")
    before_statvfs = _validate_statvfs_snapshot(value.get("statvfs_before_release"), "pre-release")
    after_release = _validate_statvfs_snapshot(value.get("statvfs_after_release"), "post-release")
    after_streams = _validate_statvfs_snapshot(value.get("statvfs_after_streams"), "post-stream")
    if (
        before_statvfs["available_bytes"] < plan["unreserved_floor_bytes"]
        or after_release["available_bytes"]
        < plan["unreserved_floor_bytes"] + plan["total_finalization_budget_bytes"]
        or after_streams["available_bytes"]
        < plan["unreserved_floor_bytes"] + plan["tail_allocation_budget_bytes"]
    ):
        raise WriterQualificationError("finalization statvfs floor evidence differs")
    streams = value.get("streams")
    expected_streams = {
        "FINAL/SHARD_MANIFESTS.jsonl": (
            plan["shard_manifests_logical_bytes"],
            plan["shard_manifests_allocation_budget_bytes"],
        ),
        "DIGEST_ROSTER.jsonl": (
            plan["digest_roster_logical_bytes"],
            plan["digest_roster_allocation_budget_bytes"],
        ),
    }
    if type(streams) is not dict or set(streams) != set(expected_streams):
        raise WriterQualificationError("finalization stream allocation roster differs")
    allocated_sum = 0
    for name, (logical, budget) in expected_streams.items():
        record = streams[name]
        if (
            type(record) is not dict
            or set(record) != {"name", "byte_length", "st_blocks", "allocated_bytes"}
            or record.get("name") != name
            or type(record.get("byte_length")) is not int or record["byte_length"] != logical
            or type(record.get("st_blocks")) is not int or record["st_blocks"] < 0
            or type(record.get("allocated_bytes")) is not int
            or record["allocated_bytes"] != record["st_blocks"] * 512
            or (logical > 0 and record["allocated_bytes"] < logical)
            or record["allocated_bytes"] > budget
        ):
            raise WriterQualificationError("finalization stream physical allocation differs")
        allocated_sum += record["allocated_bytes"]
    if allocated_sum > plan["stream_allocation_budget_bytes"]:
        raise WriterQualificationError("finalization stream allocation exceeds its released budget")
    return value


def _finalize_attempt(attempt: Path, profile: WriterProfile) -> dict[str, Any]:
    evidence = attempt / "EVIDENCE"
    final = attempt / "FINAL"
    _, predecessor, role_chain, first_1k_chain, _ = _existing_chain_state(evidence, profile)
    if any(predecessor[batch] == ZERO_SHA256 for batch in predecessor):
        raise WriterQualificationError("cannot finalize an incomplete shard set")
    attempt_custody, attempt_payload = _retained_attempt_custody(evidence, profile)
    attempt_record = _strict_json(attempt_payload)
    allocation_summary = _evidence_allocation_summary(evidence, profile)
    header = _digest_roster_header(profile, attempt_custody, allocation_summary, attempt_record)
    with _held_directory(evidence, "finalizer evidence") as evidence_fd, _held_directory(final, "finalizer FINAL") as final_fd, _held_directory(attempt, "finalizer attempt") as attempt_fd:
        expected_final_start = frozenset(FINAL_BINARY_NAMES)
        if frozenset(os.listdir(final_fd)) != expected_final_start:
            raise WriterQualificationError("FINAL does not begin with exact preallocated binary roster")
        planned_shard_bytes, planned_digest_bytes = _finalization_stream_lengths(
            evidence_fd,
            profile,
            header,
        )
        expected_plan = _finalization_plan_record(
            profile,
            attempt_record,
            allocation_summary,
            shard_manifests_bytes=planned_shard_bytes,
            digest_roster_bytes=planned_digest_bytes,
        )
        evidence_names = frozenset(os.listdir(evidence_fd))
        if "FINALIZATION_STORAGE.json" in evidence_names:
            raise WriterQualificationError("uncommitted finalization storage receipt was not recovered")
        if "FINALIZATION_PLAN.json" in evidence_names:
            held_plan = _HeldFile(evidence_fd, "FINALIZATION_PLAN.json", "retained finalization plan")
            try:
                plan, plan_payload = _parse_canonical_file(held_plan, maximum=2_000_000)
            finally:
                held_plan.close()
            _validate_finalization_plan_record(plan, expected_plan)
        else:
            plan = expected_plan
            plan_payload = _canonical_json(plan)
            stat_before_plan = _statvfs_receipt(attempt)
            if stat_before_plan["available_bytes"] < (
                plan["unreserved_floor_bytes"]
                + _round_up(len(plan_payload), plan["allocation_unit"])
            ):
                raise WriterQualificationError("finalization plan cannot be retained above the unreserved floor")
            _write_atomic_noreplace(evidence_fd, "FINALIZATION_PLAN.json", plan_payload)
        release = _release_reservation_for_finalization(attempt_fd, attempt, plan)
        final_fds = {
            name: os.open(name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=final_fd)
            for name in FINAL_BINARY_NAMES
        }
        final_offsets = {name: 0 for name in FINAL_BINARY_NAMES}
        shard_stream_fd = os.open("SHARD_MANIFESTS.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=final_fd)
        digest_roster_fd = os.open("DIGEST_ROSTER.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=attempt_fd)
        shard_stream_hash = hashlib.sha256()
        digest_roster_hash = hashlib.sha256()
        shard_stream_bytes = 0
        digest_roster_bytes = 0
        try:
            header_payload = _canonical_json(header)
            _write_all(digest_roster_fd, header_payload)
            digest_roster_hash.update(header_payload)
            digest_roster_bytes += len(header_payload)
            for batch in range(profile.geometry.batch_count):
                chain_manifest = ZERO_SHA256
                chain_roles = {role: ZERO_SHA256 for role in RECEIPT_ROLES}
                chain_first_1k = ZERO_SHA256
                for shard in range(profile.geometry.shard_count):
                    manifest = validate_shard(
                        evidence, profile, batch=batch, shard=shard,
                        predecessor_manifest_sha=chain_manifest,
                        previous_role_sha=chain_roles,
                        previous_first_1k_sha=chain_first_1k,
                    )
                    chain_manifest = manifest["content_sha256"]
                    chain_first_1k = manifest["first_1k_chain_sha256"]
                    for role in RECEIPT_ROLES:
                        chain_roles[role] = manifest["receipt_locators"][role]["content_sha256"]
                    shard_name = _shard_name(batch, shard)
                    with _held_directory_at(evidence_fd, shard_name, "finalizer shard") as shard_fd:
                        held_manifest = _HeldFile(shard_fd, "MANIFEST.json", "finalizer shard manifest")
                        try:
                            manifest_payload = held_manifest.read_all(maximum=8_000_000)
                        finally:
                            held_manifest.close()
                        if manifest_payload != _canonical_json(manifest):
                            raise WriterQualificationError("held shard manifest differs during finalization")
                        _write_all(shard_stream_fd, manifest_payload)
                        shard_stream_hash.update(manifest_payload)
                        shard_stream_bytes += len(manifest_payload)
                        row_payload = _canonical_json(_digest_roster_row(manifest, manifest_payload))
                        _write_all(digest_roster_fd, row_payload)
                        digest_roster_hash.update(row_payload)
                        digest_roster_bytes += len(row_payload)
                        if digest_roster_bytes >= DIGEST_ROSTER_MAX_BYTES:
                            raise WriterQualificationError("bounded digest roster reached its Git archive cap")
                        for fragment_name, final_name in FRAGMENT_TO_FINAL.items():
                            held = _HeldFile(shard_fd, fragment_name, f"finalizer {fragment_name}")
                            try:
                                copied = _copy_held_to_fd(held, final_fds[final_name], final_offsets[final_name])
                            finally:
                                held.close()
                            final_offsets[final_name] += copied
            os.fsync(shard_stream_fd)
            os.fsync(digest_roster_fd)
            for fd in final_fds.values():
                os.fsync(fd)
        finally:
            os.close(shard_stream_fd)
            os.close(digest_roster_fd)
            for fd in final_fds.values():
                os.close(fd)
        # FINAL/MANIFEST is the finalization commit point.  Make the sibling
        # digest-roster directory entry durable before that commit can exist.
        os.fsync(attempt_fd)
        if final_offsets != profile.geometry.output_sizes():
            raise WriterQualificationError("final binary offset arithmetic differs")
        shard_stream_stat = os.stat(
            "SHARD_MANIFESTS.jsonl",
            dir_fd=final_fd,
            follow_symlinks=False,
        )
        digest_stream_stat = os.stat(
            "DIGEST_ROSTER.jsonl",
            dir_fd=attempt_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(shard_stream_stat.st_mode)
            or shard_stream_stat.st_nlink != 1
            or shard_stream_stat.st_size != planned_shard_bytes
            or not stat.S_ISREG(digest_stream_stat.st_mode)
            or digest_stream_stat.st_nlink != 1
            or digest_stream_stat.st_size != planned_digest_bytes
        ):
            raise WriterQualificationError("finalization stream plan/identity differs")
        reconciliation: dict[str, Any] = {
            "schema": FINALIZATION_STORAGE_SCHEMA,
            "plan_content_sha256": plan["content_sha256"],
            **release,
            "statvfs_after_streams": _statvfs_receipt(attempt),
            "streams": {
                "FINAL/SHARD_MANIFESTS.jsonl": {
                    "name": "FINAL/SHARD_MANIFESTS.jsonl",
                    "byte_length": shard_stream_stat.st_size,
                    "st_blocks": shard_stream_stat.st_blocks,
                    "allocated_bytes": shard_stream_stat.st_blocks * 512,
                },
                "DIGEST_ROSTER.jsonl": {
                    "name": "DIGEST_ROSTER.jsonl",
                    "byte_length": digest_stream_stat.st_size,
                    "st_blocks": digest_stream_stat.st_blocks,
                    "allocated_bytes": digest_stream_stat.st_blocks * 512,
                },
            },
            "unreserved_floor_preserved": True,
            "claims": _claim_ceiling(False),
            "content_sha256": "",
        }
        reconciliation["content_sha256"] = _domain_hash(
            FINALIZATION_STORAGE_DOMAIN,
            reconciliation,
        )
        _validate_finalization_storage_record(reconciliation, plan)
        reconciliation_payload = _canonical_json(reconciliation)
        _write_atomic_noreplace(
            evidence_fd,
            "FINALIZATION_STORAGE.json",
            reconciliation_payload,
        )
        if _statvfs_receipt(attempt)["available_bytes"] < (
            plan["unreserved_floor_bytes"] + plan["tail_allocation_budget_bytes"]
            - _round_up(len(reconciliation_payload), plan["allocation_unit"])
        ):
            raise WriterQualificationError("finalization receipt crossed its retained tail budget")
        final_files: dict[str, Any] = {}
        for name in FINAL_BINARY_NAMES:
            held = _HeldFile(final_fd, name, f"final {name}")
            try:
                metadata = os.fstat(held.fd)
                if held.size != profile.geometry.output_sizes()[name] or (held.size > 0 and metadata.st_blocks * 512 < held.size):
                    raise WriterQualificationError("final binary lost physical allocation")
                final_files[name] = {"name": name, "byte_length": held.size, "sha256": held.sha256(), "st_blocks": metadata.st_blocks}
            finally:
                held.close()
        manifest: dict[str, Any] = {
            "schema": FINAL_SCHEMA,
            "profile": _profile_record(profile),
            "files": final_files,
            "shard_manifests": {
                "name": "SHARD_MANIFESTS.jsonl", "byte_length": shard_stream_bytes,
                "sha256": shard_stream_hash.hexdigest(), "record_count": profile.shard_total,
                "st_blocks": shard_stream_stat.st_blocks,
                "order": "BATCH_MAJOR_THEN_SHARD_MAJOR_CANONICAL_JSON_LINES",
            },
            "terminal_manifest_content_sha256_by_batch": {str(batch): predecessor[batch] for batch in predecessor},
            "terminal_receipt_content_sha256_by_batch_and_role": {str(batch): role_chain[batch] for batch in role_chain},
            "terminal_first_1k_chain_sha256_by_batch": {str(batch): first_1k_chain[batch] for batch in first_1k_chain},
            "attempt_and_storage_custody": attempt_custody,
            "git_preregistration_custody": attempt_record["git_preregistration_custody"],
            "evidence_allocation_summary": allocation_summary,
            "finalization_plan": plan,
            "finalization_storage_reconciliation": reconciliation,
            "digest_roster_header_content_sha256": header["content_sha256"],
            "claims": _claim_ceiling(profile.exact_production),
            "content_sha256": "",
        }
        manifest["content_sha256"] = _domain_hash(FINAL_DOMAIN, manifest)
        manifest_file = _write_atomic_noreplace(final_fd, "MANIFEST.json", _canonical_json(manifest))
        os.fsync(final_fd)
        if _statvfs_receipt(attempt)["available_bytes"] < (
            plan["unreserved_floor_bytes"] + RESULT_PUBLICATION_MARGIN_BYTES
        ):
            raise WriterQualificationError("post-finalization storage floor leaves no result-publication margin")
        return {
            "manifest": manifest,
            "manifest_file": manifest_file,
            "digest_roster": {
                "name": "DIGEST_ROSTER.jsonl", "byte_length": digest_roster_bytes,
                "sha256": digest_roster_hash.hexdigest(), "record_count": profile.shard_total + 1,
                "header_record_count": 1, "shard_record_count": profile.shard_total,
                "st_blocks": digest_stream_stat.st_blocks,
                "cap_bytes": DIGEST_ROSTER_MAX_BYTES,
            },
        }


def _finalize_or_recover_attempt(attempt: Path, profile: WriterProfile) -> dict[str, Any]:
    """Resume cleanly at either side of FINAL/MANIFEST's commit point."""
    final = attempt / "FINAL"
    digest = attempt / "DIGEST_ROSTER.jsonl"
    binary_roster = frozenset(FINAL_BINARY_NAMES)
    with _held_directory(attempt, "finalization recovery attempt") as attempt_fd:
        attempt_names = frozenset(os.listdir(attempt_fd))
        with _held_directory_at(attempt_fd, "FINAL", "finalization recovery FINAL") as final_fd:
            names = frozenset(os.listdir(final_fd))
    if names == frozenset({*binary_roster, "SHARD_MANIFESTS.jsonl", "MANIFEST.json"}):
        if "DIGEST_ROSTER.jsonl" not in attempt_names:
            raise WriterQualificationError("committed FINAL lost its digest roster")
        return _validate_payload_set(attempt / "EVIDENCE", final, digest, profile)
    allowed_incomplete = frozenset({*binary_roster, "SHARD_MANIFESTS.jsonl", ".MANIFEST.json.partial"})
    if not binary_roster.issubset(names) or not names.issubset(allowed_incomplete):
        raise WriterQualificationError("incomplete FINAL roster differs")
    # A crash before MANIFEST is not a commit. Rebuild the two streams and
    # overwrite the already-reserved binaries from fully validated shards.
    with _held_directory(final, "recoverable FINAL") as final_fd:
        if ".MANIFEST.json.partial" in names:
            _unlink_regular_at(final_fd, ".MANIFEST.json.partial", "incomplete FINAL manifest")
        if "SHARD_MANIFESTS.jsonl" in names:
            _unlink_regular_at(final_fd, "SHARD_MANIFESTS.jsonl", "incomplete shard-manifest stream")
    if "DIGEST_ROSTER.jsonl" in attempt_names:
        with _held_directory(attempt, "recoverable attempt") as attempt_fd:
            _unlink_regular_at(attempt_fd, "DIGEST_ROSTER.jsonl", "incomplete digest roster")
    with _held_directory(attempt / "EVIDENCE", "recoverable finalization evidence") as evidence_fd:
        evidence_names = frozenset(os.listdir(evidence_fd))
        if "FINALIZATION_STORAGE.json" in evidence_names:
            _unlink_regular_at(
                evidence_fd,
                "FINALIZATION_STORAGE.json",
                "incomplete finalization storage reconciliation",
            )
    return _finalize_attempt(attempt, profile)


def _cleanup_empty_attempt_shell(output_root: Path, attempt: Path) -> None:
    if not attempt.exists():
        return
    with _held_directory(output_root.parent, "cleanup parent") as parent_fd:
        with _held_directory(attempt, "cleanup attempt") as attempt_fd:
            names = frozenset(os.listdir(attempt_fd))
            if names != frozenset({"ATTEMPT.json", "RESERVATION.bin"}):
                raise WriterQualificationError("published attempt shell roster differs")
            _unlink_regular_at(attempt_fd, "RESERVATION.bin", "released reservation")
            _unlink_regular_at(attempt_fd, "ATTEMPT.json", "released attempt receipt")
        os.rmdir(attempt.name, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _best_effort_cleanup_after_durable_result(output_root: Path, attempt: Path) -> None:
    """Cleanup is never allowed to turn a durable, validated RESULT into failure."""
    try:
        _cleanup_empty_attempt_shell(output_root, attempt)
    except (OSError, WriterQualificationError):
        return


def _result_record(protocol: Mapping[str, Any], profile: WriterProfile, finalized: Mapping[str, Any]) -> dict[str, Any]:
    manifest = finalized["manifest"]
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "SYNTHETIC_IO_WRITER_QUALIFICATION_COMPLETE_NO_DYNAMICS_OR_SCIENCE_CLAIM",
        "protocol_content_sha256": protocol["content_sha256"],
        "profile_content_sha256": _profile_record(profile)["content_sha256"],
        "final_manifest_content_sha256": manifest["content_sha256"],
        "final_manifest_file_sha256": finalized["manifest_file"]["sha256"],
        "final_files": manifest["files"],
        "shard_manifests": manifest["shard_manifests"],
        "terminal_manifest_content_sha256_by_batch": manifest["terminal_manifest_content_sha256_by_batch"],
        "terminal_receipt_content_sha256_by_batch_and_role": manifest["terminal_receipt_content_sha256_by_batch_and_role"],
        "terminal_first_1k_chain_sha256_by_batch": manifest["terminal_first_1k_chain_sha256_by_batch"],
        "digest_roster": finalized["digest_roster"],
        "attempt_and_storage_custody": manifest["attempt_and_storage_custody"],
        "git_preregistration_custody": manifest["git_preregistration_custody"],
        "evidence_allocation_summary": manifest["evidence_allocation_summary"],
        "finalization_plan": manifest["finalization_plan"],
        "finalization_storage_reconciliation": manifest["finalization_storage_reconciliation"],
        "digest_roster_header_content_sha256": manifest["digest_roster_header_content_sha256"],
        "publication_order": ["EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json_LAST"],
        "archive_surface": {
            "git_result_custody": ["PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"],
            "large_payload_bytes_in_git_custody": False,
            "local_live_validation_requires": ["EVIDENCE/", "FINAL/"],
            "statement": "Git result custody covers summary plus digest roster only, not the retained large payload bytes.",
        },
        "claims": _claim_ceiling(profile.exact_production),
        "content_sha256": "",
    }
    record["content_sha256"] = _domain_hash(RESULT_DOMAIN, record)
    return record


def _read_protocol_from_root(root: Path) -> dict[str, Any]:
    with _held_directory(root, "qualification root") as root_fd:
        held = _HeldFile(root_fd, "PROTOCOL.json", "qualification protocol")
        try:
            value, _ = _parse_canonical_file(held, maximum=8_000_000)
        finally:
            held.close()
    validate_protocol(value)
    return value


def _publish_attempt(
    output_root: Path,
    attempt: Path,
    profile: WriterProfile,
    finalized: Mapping[str, Any],
    expected_git_custody: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_protocol_from_root(output_root)
    if (
        type(finalized) is not dict
        or type(finalized.get("manifest")) is not dict
        or not _exact_equal(
            finalized["manifest"].get("git_preregistration_custody"),
            expected_git_custody,
        )
    ):
        raise WriterQualificationError("publication Git preregistration custody differs from the live gate")
    result = _result_record(protocol, profile, finalized)
    with _held_directory(output_root.parent, "publication parent") as parent_fd, _held_directory(output_root, "publication root") as root_fd, _held_directory(attempt, "publication attempt") as attempt_fd:
        allowed_root_before_result = frozenset({"PROTOCOL.json", "EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl"})
        initial_root_names = frozenset(os.listdir(root_fd))
        if "PROTOCOL.json" not in initial_root_names or not initial_root_names.issubset(allowed_root_before_result):
            raise WriterQualificationError("publication root pre-result roster differs")
        for name in ("EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl"):
            root_names = frozenset(os.listdir(root_fd))
            attempt_names = frozenset(os.listdir(attempt_fd))
            if name in root_names:
                if name in attempt_names:
                    raise WriterQualificationError("publication name exists in both attempt and destination")
                continue
            if name not in attempt_names:
                raise WriterQualificationError("publication source is absent")
            _rename_noreplace(attempt_fd, name, root_fd, name)
        if frozenset(os.listdir(root_fd)) != allowed_root_before_result:
            raise WriterQualificationError("publication payload roster differs before result-last commit")
        if frozenset(os.listdir(attempt_fd)) != frozenset({"ATTEMPT.json", "RESERVATION.bin"}):
            raise WriterQualificationError("publication attempt shell roster differs before result-last commit")
        result_payload = _canonical_json(result)
        plan = result["finalization_plan"]
        required_before_result = (
            plan["unreserved_floor_bytes"]
            + _round_up(len(result_payload), plan["allocation_unit"])
            + RESULT_PUBLICATION_MARGIN_BYTES
        )
        if _statvfs_receipt(output_root)["available_bytes"] < required_before_result:
            raise WriterQualificationError("result publication would cross the unreserved floor")
        partial_name = ".QUALIFICATION_RESULT.json.partial"
        _write_create_only(root_fd, partial_name, result_payload)
        if _statvfs_receipt(output_root)["available_bytes"] < plan["unreserved_floor_bytes"]:
            raise WriterQualificationError("result partial crossed the unreserved floor before commit")
        _rename_noreplace(root_fd, partial_name, root_fd, "QUALIFICATION_RESULT.json")
        os.fsync(root_fd)
    return result


def _resume_publication_state(
    output_root: Path,
    attempt: Path,
    profile: WriterProfile,
    expected_git_custody: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not output_root.exists():
        raise WriterQualificationError("prepared output root is absent")
    with _held_directory(output_root, "publication-resume root") as root_fd:
        root_names = frozenset(os.listdir(root_fd))
        partial_result = ".QUALIFICATION_RESULT.json.partial"
        if partial_result in root_names:
            if "QUALIFICATION_RESULT.json" in root_names:
                raise WriterQualificationError("result and incomplete result coexist")
            _unlink_regular_at(root_fd, partial_result, "incomplete qualification result")
            root_names = frozenset(os.listdir(root_fd))
    if "QUALIFICATION_RESULT.json" in root_names:
        if root_names != SUCCESS_ROOT_ROSTER:
            raise WriterQualificationError("result exists without the exact success roster")
        value = validate(output_root, profile=profile, test_only=not profile.exact_production)
        if not _exact_equal(value.get("git_preregistration_custody"), expected_git_custody):
            raise WriterQualificationError("published result Git custody differs from the live gate")
        _best_effort_cleanup_after_durable_result(output_root, attempt)
        return value
    allowed = frozenset({"PROTOCOL.json", "EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl"})
    if not root_names.issubset(allowed) or "PROTOCOL.json" not in root_names:
        raise WriterQualificationError("partial publication root roster differs")
    if any(name in root_names for name in ("EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl")):
        if not attempt.exists():
            raise WriterQualificationError("partial publication lost its attempt root")
        # The finalized summary is reconstructed only after fully validating the mixed locations.
        evidence = output_root / "EVIDENCE" if "EVIDENCE" in root_names else attempt / "EVIDENCE"
        final = output_root / "FINAL" if "FINAL" in root_names else attempt / "FINAL"
        digest = output_root / "DIGEST_ROSTER.jsonl" if "DIGEST_ROSTER.jsonl" in root_names else attempt / "DIGEST_ROSTER.jsonl"
        finalized = _validate_payload_set(evidence, final, digest, profile)
        return _publish_attempt(output_root, attempt, profile, finalized, expected_git_custody)
    return None


def _recover_uncommitted_initialization(
    output_root: Path,
    attempt: Path,
    profile: WriterProfile,
    floor_bytes: int,
    git_custody: Mapping[str, Any],
) -> None:
    """Discard only an attempt that never reached its top-level commit record."""
    with _held_directory(attempt, "initialization recovery attempt") as attempt_fd:
        if "ATTEMPT.json" in os.listdir(attempt_fd):
            return
    with _held_directory(output_root.parent, "initialization recovery parent") as parent_fd:
        _unlink_tree_at(parent_fd, attempt.name)
    _initialize_attempt(output_root, profile, floor_bytes=floor_bytes, git_custody=git_custody)


def qualify(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    profile: WriterProfile = PRODUCTION_PROFILE,
    test_only: bool = False,
    resume: bool = False,
    fault_after_shards: int | None = None,
    fault_after_finalize: bool = False,
    unreserved_floor_bytes: int | None = None,
    externally_preregistered: bool = False,
    prereg_commit: str | None = None,
    prereg_tag_object: str | None = None,
) -> dict[str, Any]:
    _require_profile(profile, test_only=test_only)
    _require_bool(resume, "resume")
    _require_bool(fault_after_finalize, "fault_after_finalize")
    if fault_after_finalize and test_only is not True:
        raise WriterQualificationError("finalization fault injection is test-only")
    root = Path(os.path.abspath(os.fspath(output_root)))
    if profile.exact_production is True and root != DEFAULT_OUTPUT_ROOT:
        raise WriterQualificationError("production qualification is confined to the frozen default root")
    if profile.exact_production is False and root == DEFAULT_OUTPUT_ROOT:
        raise WriterQualificationError("test-only profile cannot occupy the production default root")
    git_custody = _resolve_git_custody(
        profile,
        externally_preregistered=externally_preregistered,
        prereg_commit=prereg_commit,
        prereg_tag_object=prereg_tag_object,
    )
    floor = UNRESERVED_FLOOR_BYTES if profile.exact_production else (
        0 if unreserved_floor_bytes is None else _require_int(unreserved_floor_bytes, "test floor")
    )
    if profile.exact_production and unreserved_floor_bytes is not None:
        raise WriterQualificationError("production unreserved floor cannot be overridden")
    attempt = root.parent / _attempt_name(root)
    if resume:
        published = _resume_publication_state(root, attempt, profile, git_custody)
        if published is not None:
            return published
        validate_prepared(root)
        if not attempt.exists():
            raise WriterQualificationError("resume requested without an attempt-local partial")
        _recover_uncommitted_initialization(root, attempt, profile, floor, git_custody)
    else:
        validate_prepared(root)
        _initialize_attempt(root, profile, floor_bytes=floor, git_custody=git_custody)
    attempt_record = _load_attempt(attempt, root, profile, floor, git_custody)
    _write_all_shards(
        attempt, root, profile, attempt_record, floor_bytes=floor,
        fault_after_shards=fault_after_shards,
    )
    finalized = _finalize_or_recover_attempt(attempt, profile)
    if fault_after_finalize:
        raise WriterQualificationError("injected crash after finalized attempt before publication")
    result = _publish_attempt(root, attempt, profile, finalized, git_custody)
    observed = validate(root, profile=profile, test_only=test_only)
    if not _exact_equal(result, observed):
        raise WriterQualificationError("post-publication validation result differs")
    _best_effort_cleanup_after_durable_result(root, attempt)
    return result


def _read_json_line(line: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(line)
    if line != _canonical_json(value):
        raise WriterQualificationError(f"{label} is not a canonical JSON line")
    return value


def _stream_lines(file: BinaryIO, label: str, *, maximum_line: int = 8_000_000) -> Iterator[bytes]:
    while True:
        line = file.readline(maximum_line + 1)
        if not line:
            return
        if len(line) > maximum_line or not line.endswith(b"\n"):
            raise WriterQualificationError(f"{label} has an overlong or unterminated line")
        yield line


def _validate_payload_set(evidence: Path, final: Path, digest_roster: Path, profile: WriterProfile) -> dict[str, Any]:
    expected_evidence_roster = frozenset({
        "ATTEMPT.json", "STORAGE_RESERVATION.json",
        "FINALIZATION_PLAN.json", "FINALIZATION_STORAGE.json",
        *(_shard_name(batch, shard) for batch in range(profile.geometry.batch_count) for shard in range(profile.geometry.shard_count)),
    })
    expected_final_roster = frozenset({*FINAL_BINARY_NAMES, "SHARD_MANIFESTS.jsonl", "MANIFEST.json"})
    attempt_custody, attempt_payload = _retained_attempt_custody(evidence, profile)
    attempt_record = _strict_json(attempt_payload)
    allocation_summary = _evidence_allocation_summary(evidence, profile)
    expected_header = _digest_roster_header(profile, attempt_custody, allocation_summary, attempt_record)
    finalization_plan, finalization_reconciliation = _retained_finalization_records(
        evidence,
        profile,
        attempt_record,
        allocation_summary,
    )
    with _held_directory(evidence, "local retained EVIDENCE", roster=expected_evidence_roster) as evidence_fd, _held_directory(final, "local retained FINAL", roster=expected_final_roster) as final_fd:
        final_manifest_held = _HeldFile(final_fd, "MANIFEST.json", "FINAL manifest")
        try:
            final_manifest, final_manifest_payload = _parse_canonical_file(final_manifest_held, maximum=32_000_000)
        finally:
            final_manifest_held.close()
        if (
            set(final_manifest) != {
                "schema", "profile", "files", "shard_manifests",
                "terminal_manifest_content_sha256_by_batch",
                "terminal_receipt_content_sha256_by_batch_and_role",
                "terminal_first_1k_chain_sha256_by_batch",
                "attempt_and_storage_custody", "evidence_allocation_summary",
                "git_preregistration_custody",
                "finalization_plan", "finalization_storage_reconciliation",
                "digest_roster_header_content_sha256", "claims", "content_sha256",
            }
            or final_manifest.get("schema") != FINAL_SCHEMA
            or not _exact_equal(final_manifest.get("profile"), _profile_record(profile))
            or final_manifest.get("content_sha256") != _domain_hash(FINAL_DOMAIN, final_manifest)
            or not _exact_equal(final_manifest.get("claims"), _claim_ceiling(profile.exact_production))
            or not _exact_equal(final_manifest.get("attempt_and_storage_custody"), attempt_custody)
            or not _exact_equal(final_manifest.get("git_preregistration_custody"), attempt_record["git_preregistration_custody"])
            or not _exact_equal(final_manifest.get("evidence_allocation_summary"), allocation_summary)
            or not _exact_equal(final_manifest.get("finalization_plan"), finalization_plan)
            or not _exact_equal(
                final_manifest.get("finalization_storage_reconciliation"),
                finalization_reconciliation,
            )
            or final_manifest.get("digest_roster_header_content_sha256") != expected_header["content_sha256"]
        ):
            raise WriterQualificationError("FINAL manifest schema/identity/claim differs")
        files = final_manifest.get("files")
        if type(files) is not dict or set(files) != set(FINAL_BINARY_NAMES):
            raise WriterQualificationError("FINAL binary receipt roster differs")
        held_final: dict[str, _HeldFile] = {}
        try:
            for name in FINAL_BINARY_NAMES:
                held_final[name] = _HeldFile(final_fd, name, f"FINAL {name}")
                _validate_fragment(held_final[name], files[name], name)
                metadata = os.fstat(held_final[name].fd)
                if held_final[name].size > 0 and metadata.st_blocks * 512 < held_final[name].size:
                    raise WriterQualificationError("FINAL binary is not physically allocated")
            shard_stream_held = _HeldFile(final_fd, "SHARD_MANIFESTS.jsonl", "shard manifest stream")
            try:
                shard_stream_record = final_manifest.get("shard_manifests")
                if type(shard_stream_record) is not dict or set(shard_stream_record) != {"name", "byte_length", "sha256", "record_count", "st_blocks", "order"}:
                    raise WriterQualificationError("shard manifest stream receipt differs")
                if (
                    shard_stream_record.get("name") != "SHARD_MANIFESTS.jsonl"
                    or type(shard_stream_record.get("byte_length")) is not int
                    or shard_stream_record.get("byte_length") != shard_stream_held.size
                    or shard_stream_record.get("byte_length") != finalization_plan["shard_manifests_logical_bytes"]
                    or shard_stream_record.get("sha256") != shard_stream_held.sha256()
                    or type(shard_stream_record.get("st_blocks")) is not int
                    or shard_stream_record.get("st_blocks") != os.fstat(shard_stream_held.fd).st_blocks
                    or shard_stream_record.get("st_blocks") != finalization_reconciliation["streams"]["FINAL/SHARD_MANIFESTS.jsonl"]["st_blocks"]
                    or type(shard_stream_record.get("record_count")) is not int
                    or shard_stream_record.get("record_count") != profile.shard_total
                    or shard_stream_record.get("order") != "BATCH_MAJOR_THEN_SHARD_MAJOR_CANONICAL_JSON_LINES"
                ):
                    raise WriterQualificationError("shard manifest stream identity differs")
                digest_path = Path(os.path.abspath(os.fspath(digest_roster)))
                with _held_file(digest_path, "held digest roster replay") as digest_held, os.fdopen(os.dup(digest_held.fd), "rb") as digest_file, os.fdopen(os.dup(shard_stream_held.fd), "rb") as shard_file:
                    shard_file.seek(0)
                    digest_file.seek(0)
                    final_offsets = {name: 0 for name in FINAL_BINARY_NAMES}
                    predecessor = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
                    roles = {batch: {role: ZERO_SHA256 for role in RECEIPT_ROLES} for batch in range(profile.geometry.batch_count)}
                    first_1k = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
                    digest_hasher = hashlib.sha256()
                    digest_bytes = 0
                    count = 0
                    shard_lines = _stream_lines(shard_file, "shard manifest stream")
                    digest_lines = _stream_lines(digest_file, "digest roster", maximum_line=2_000_000)
                    try:
                        header_line = next(digest_lines)
                    except StopIteration as exc:
                        raise WriterQualificationError("digest roster header is absent") from exc
                    header = _read_json_line(header_line, "digest roster header")
                    if not _exact_equal(header, expected_header):
                        raise WriterQualificationError("digest roster header/custody differs")
                    digest_hasher.update(header_line)
                    digest_bytes += len(header_line)
                    for order, pair in enumerate(zip(shard_lines, digest_lines, strict=True)):
                        shard_line, digest_line = pair
                        batch, shard = divmod(order, profile.geometry.shard_count)
                        manifest = validate_shard(
                            evidence, profile, batch=batch, shard=shard,
                            predecessor_manifest_sha=predecessor[batch],
                            previous_role_sha=roles[batch],
                            previous_first_1k_sha=first_1k[batch],
                        )
                        if shard_line != _canonical_json(manifest):
                            raise WriterQualificationError("shard stream differs from held evidence")
                        expected_row = _digest_roster_row(manifest, shard_line)
                        row = _read_json_line(digest_line, "digest roster row")
                        if not _exact_equal(row, expected_row):
                            raise WriterQualificationError("digest roster row differs from held evidence")
                        predecessor[batch] = manifest["content_sha256"]
                        first_1k[batch] = manifest["first_1k_chain_sha256"]
                        for role in RECEIPT_ROLES:
                            roles[batch][role] = manifest["receipt_locators"][role]["content_sha256"]
                        with _held_directory_at(evidence_fd, _shard_name(batch, shard), "replay shard") as shard_fd:
                            for fragment_name, final_name in FRAGMENT_TO_FINAL.items():
                                source = _HeldFile(shard_fd, fragment_name, f"replay {fragment_name}")
                                try:
                                    offset = 0
                                    while offset < source.size:
                                        amount = min(1 << 20, source.size - offset)
                                        if source.pread(amount, offset) != held_final[final_name].pread(amount, final_offsets[final_name] + offset):
                                            raise WriterQualificationError("FINAL binary differs from retained shard fragment")
                                        offset += amount
                                    final_offsets[final_name] += source.size
                                finally:
                                    source.close()
                        digest_hasher.update(digest_line)
                        digest_bytes += len(digest_line)
                        count += 1
                    if count != profile.shard_total or any(final_offsets[name] != held_final[name].size for name in FINAL_BINARY_NAMES):
                        raise WriterQualificationError("streaming replay cardinality/offset differs")
                    if digest_bytes != digest_held.size or digest_bytes >= DIGEST_ROSTER_MAX_BYTES:
                        raise WriterQualificationError("digest roster exceeds Git archive cap")
                    if (
                        digest_bytes != finalization_plan["digest_roster_logical_bytes"]
                        or os.fstat(digest_held.fd).st_blocks
                        != finalization_reconciliation["streams"]["DIGEST_ROSTER.jsonl"]["st_blocks"]
                    ):
                        raise WriterQualificationError("digest roster finalization reconciliation differs")
                    expected_terminals = {
                        "terminal_manifest_content_sha256_by_batch": {str(batch): predecessor[batch] for batch in predecessor},
                        "terminal_receipt_content_sha256_by_batch_and_role": {str(batch): roles[batch] for batch in roles},
                        "terminal_first_1k_chain_sha256_by_batch": {str(batch): first_1k[batch] for batch in first_1k},
                    }
                    for field, expected in expected_terminals.items():
                        if not _exact_equal(final_manifest.get(field), expected):
                            raise WriterQualificationError("FINAL terminal chain differs")
                    return {
                        "manifest": final_manifest,
                        "manifest_file": {"name": "MANIFEST.json", "byte_length": len(final_manifest_payload), "sha256": _sha256(final_manifest_payload)},
                        "digest_roster": {
                            "name": "DIGEST_ROSTER.jsonl", "byte_length": digest_bytes,
                            "sha256": digest_hasher.hexdigest(), "record_count": count + 1,
                            "header_record_count": 1, "shard_record_count": count,
                            "st_blocks": os.fstat(digest_held.fd).st_blocks,
                            "cap_bytes": DIGEST_ROSTER_MAX_BYTES,
                        },
                    }
            finally:
                shard_stream_held.close()
        finally:
            for held in held_final.values():
                held.close()


def validate(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    profile: WriterProfile = PRODUCTION_PROFILE,
    test_only: bool = False,
) -> dict[str, Any]:
    _require_profile(profile, test_only=test_only)
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "successful qualification root", roster=SUCCESS_ROOT_ROSTER) as root_fd:
        protocol_held = _HeldFile(root_fd, "PROTOCOL.json", "result protocol")
        result_held = _HeldFile(root_fd, "QUALIFICATION_RESULT.json", "qualification result")
        digest_held = _HeldFile(root_fd, "DIGEST_ROSTER.jsonl", "digest roster")
        try:
            protocol, protocol_payload = _parse_canonical_file(protocol_held, maximum=8_000_000)
            result, result_payload = _parse_canonical_file(result_held, maximum=8_000_000)
            if digest_held.size >= DIGEST_ROSTER_MAX_BYTES or digest_held.size >= ARCHIVE_FILE_MAX_BYTES:
                raise WriterQualificationError("digest roster exceeds archive cap")
        finally:
            protocol_held.close()
            result_held.close()
            digest_held.close()
        validate_protocol(protocol)
    finalized = _validate_payload_set(root / "EVIDENCE", root / "FINAL", root / "DIGEST_ROSTER.jsonl", profile)
    expected = _result_record(protocol, profile, finalized)
    if result_payload != _canonical_json(result) or not _exact_equal(result, expected) or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result):
        raise WriterQualificationError("qualification result identity/claim ceiling differs")
    if len(protocol_payload) >= ARCHIVE_FILE_MAX_BYTES or len(result_payload) >= ARCHIVE_FILE_MAX_BYTES:
        raise WriterQualificationError("archive summary exceeds per-file cap")
    return result


def _expected_sentinel_fragment_hashes(
    profile: WriterProfile,
    batch: int,
    shard: int,
    final_hashers: Mapping[str, Any],
) -> dict[str, str]:
    """Recreate one deterministic shard without retaining its payload bytes."""
    if type(final_hashers) is not dict or set(final_hashers) != set(FINAL_BINARY_NAMES):
        raise WriterQualificationError("archive final hash accumulator roster differs")
    fragment_hashers = {name: hashlib.sha256() for name in FRAGMENT_NAMES}
    geometry = profile.geometry
    for step in geometry.expected_steps(shard):
        state = _sentinel_state(profile, batch, step)
        lifecycle = _sentinel_lifecycle(profile, batch, step)
        if geometry.is_b22(step):
            node = grid.node_digest_v1(batch, step, state, lifecycle, geometry)
            for fragment_name, payload in (
                ("B22_STATES.fragment", state),
                ("B22_NODE_SHA256.fragment", node),
                ("B22_LIFECYCLE.fragment", lifecycle),
            ):
                fragment_hashers[fragment_name].update(payload)
                final_hashers[FRAGMENT_TO_FINAL[fragment_name]].update(payload)
        if geometry.is_short(step):
            for fragment_name, payload in (
                ("ML_SHORT_STATES.fragment", state),
                ("ML_SHORT_LIFECYCLE.fragment", lifecycle),
            ):
                fragment_hashers[fragment_name].update(payload)
                final_hashers[FRAGMENT_TO_FINAL[fragment_name]].update(payload)
    return {name: fragment_hashers[name].hexdigest() for name in FRAGMENT_NAMES}


def _validate_archive_digest_rows(
    digest_path: Path,
    profile: WriterProfile,
    expected_header: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
    role_chain = {
        batch: {role: ZERO_SHA256 for role in RECEIPT_ROLES}
        for batch in range(profile.geometry.batch_count)
    }
    first_1k = {batch: ZERO_SHA256 for batch in range(profile.geometry.batch_count)}
    final_hashers = {name: hashlib.sha256() for name in FINAL_BINARY_NAMES}
    with _held_file(digest_path, "archive digest semantic replay") as held:
        with os.fdopen(os.dup(held.fd), "rb") as stream:
            lines = _stream_lines(stream, "archive digest semantic replay", maximum_line=2_000_000)
            try:
                header_line = next(lines)
            except StopIteration as exc:
                raise WriterQualificationError("archive digest header is absent") from exc
            if not _exact_equal(_read_json_line(header_line, "archive digest header"), expected_header):
                raise WriterQualificationError("archive digest header semantic identity differs")
            count = 0
            expected_keys = {
                "schema", "batch_order", "shard_index", "shard_name", "manifest_byte_length",
                "manifest_file_sha256", "manifest_content_sha256",
                "predecessor_manifest_content_sha256", "first_1k_chain_sha256",
                "fragment_sha256", "receipt_file_sha256", "receipt_content_sha256", "content_sha256",
            }
            for order, line in enumerate(lines):
                if order >= profile.shard_total:
                    raise WriterQualificationError("archive digest has excess shard rows")
                row = _read_json_line(line, "archive digest shard row")
                batch, shard = divmod(order, profile.geometry.shard_count)
                expected_first_1k = _advance_first_1k(profile, batch, shard, first_1k[batch])
                if (
                    set(row) != expected_keys or row.get("schema") != DIGEST_ROSTER_SCHEMA
                    or type(row.get("batch_order")) is not int or row.get("batch_order") != batch
                    or type(row.get("shard_index")) is not int or row.get("shard_index") != shard
                    or row.get("shard_name") != _shard_name(batch, shard)
                    or type(row.get("manifest_byte_length")) is not int or row["manifest_byte_length"] <= 0
                    or not _is_sha256(row.get("manifest_file_sha256"))
                    or not _is_sha256(row.get("manifest_content_sha256"))
                    or row.get("predecessor_manifest_content_sha256") != predecessor[batch]
                    or row.get("first_1k_chain_sha256") != expected_first_1k
                    or type(row.get("fragment_sha256")) is not dict
                    or set(row["fragment_sha256"]) != set(FRAGMENT_NAMES)
                    or any(not _is_sha256(value) for value in row["fragment_sha256"].values())
                    or type(row.get("receipt_file_sha256")) is not dict
                    or set(row["receipt_file_sha256"]) != set(RECEIPT_ROLES)
                    or any(not _is_sha256(value) for value in row["receipt_file_sha256"].values())
                    or type(row.get("receipt_content_sha256")) is not dict
                    or set(row["receipt_content_sha256"]) != set(RECEIPT_ROLES)
                    or any(not _is_sha256(value) for value in row["receipt_content_sha256"].values())
                    or row.get("content_sha256") != _domain_hash(DIGEST_ROSTER_DOMAIN, row)
                ):
                    raise WriterQualificationError("archive digest shard-row semantics differ")
                expected_fragment_sha256 = _expected_sentinel_fragment_hashes(
                    profile, batch, shard, final_hashers
                )
                if not _exact_equal(row["fragment_sha256"], expected_fragment_sha256):
                    raise WriterQualificationError("archive digest fragment semantics differ")
                for role in RECEIPT_ROLES:
                    receipt = _receipt_record(
                        profile,
                        role=role,
                        batch=batch,
                        shard=shard,
                        predecessor_manifest_content_sha256=predecessor[batch],
                        previous_role_content_sha256=role_chain[batch][role],
                        first_1k_chain_sha256=expected_first_1k,
                    )
                    receipt_payload = _canonical_json(receipt)
                    if (
                        row["receipt_content_sha256"][role] != receipt["content_sha256"]
                        or row["receipt_file_sha256"][role] != _sha256(receipt_payload)
                    ):
                        raise WriterQualificationError("archive digest exact receipt semantics differ")
                    role_chain[batch][role] = receipt["content_sha256"]
                predecessor[batch] = row["manifest_content_sha256"]
                first_1k[batch] = expected_first_1k
                count += 1
            if count != profile.shard_total:
                raise WriterQualificationError("archive digest shard-row cardinality differs")
    return {
        "terminal_manifest_content_sha256_by_batch": {
            str(batch): predecessor[batch] for batch in predecessor
        },
        "terminal_receipt_content_sha256_by_batch_and_role": {
            str(batch): role_chain[batch] for batch in role_chain
        },
        "terminal_first_1k_chain_sha256_by_batch": {
            str(batch): first_1k[batch] for batch in first_1k
        },
        "final_file_sha256": {
            name: final_hashers[name].hexdigest() for name in FINAL_BINARY_NAMES
        },
    }


def validate_archive_surface(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    profile: WriterProfile = PRODUCTION_PROFILE,
    test_only: bool = False,
) -> dict[str, Any]:
    """Validate only the small Git custody surface, without claiming payload replay."""
    _require_profile(profile, test_only=test_only)
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "archive surface") as root_fd:
        names = frozenset(os.listdir(root_fd))
        if names != ARCHIVE_ROSTER and names != SUCCESS_ROOT_ROSTER:
            raise WriterQualificationError("archive/full root roster differs")
        held_protocol = _HeldFile(root_fd, "PROTOCOL.json", "archive protocol")
        held_digest = _HeldFile(root_fd, "DIGEST_ROSTER.jsonl", "archive digest roster")
        held_result = _HeldFile(root_fd, "QUALIFICATION_RESULT.json", "archive result")
        try:
            protocol, _ = _parse_canonical_file(held_protocol, maximum=ARCHIVE_FILE_MAX_BYTES - 1)
            result, _ = _parse_canonical_file(held_result, maximum=ARCHIVE_FILE_MAX_BYTES - 1)
            digest_sha = held_digest.sha256()
            digest_size = held_digest.size
            with os.fdopen(os.dup(held_digest.fd), "rb") as digest_file:
                digest_file.seek(0)
                header_line = digest_file.readline(2_000_001)
            if not header_line or len(header_line) > 2_000_000 or not header_line.endswith(b"\n"):
                raise WriterQualificationError("archive digest header is absent or overlong")
            header = _read_json_line(header_line, "archive digest header")
        finally:
            held_protocol.close()
            held_digest.close()
            held_result.close()
    validate_protocol(protocol)
    digest_record = result.get("digest_roster")
    result_keys = {
        "schema", "status", "protocol_content_sha256", "profile_content_sha256",
        "final_manifest_content_sha256", "final_manifest_file_sha256", "final_files",
        "shard_manifests", "terminal_manifest_content_sha256_by_batch",
        "terminal_receipt_content_sha256_by_batch_and_role",
        "terminal_first_1k_chain_sha256_by_batch", "digest_roster",
        "attempt_and_storage_custody", "git_preregistration_custody", "evidence_allocation_summary",
        "finalization_plan", "finalization_storage_reconciliation",
        "digest_roster_header_content_sha256", "publication_order", "archive_surface",
        "claims", "content_sha256",
    }
    expected_archive = {
        "git_result_custody": ["PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"],
        "large_payload_bytes_in_git_custody": False,
        "local_live_validation_requires": ["EVIDENCE/", "FINAL/"],
        "statement": "Git result custody covers summary plus digest roster only, not the retained large payload bytes.",
    }
    header_attempt = header.get("attempt_record") if type(header) is dict else None
    if type(header_attempt) is dict:
        _validate_attempt_record_intrinsic(header_attempt, profile)
    allocation = result.get("evidence_allocation_summary")
    allocation_keys = {
        "total_allocated_bytes", "fragment_allocated_bytes", "metadata_allocated_bytes",
        "maximum_shard_metadata_allocated_bytes", "per_shard_metadata_allowance_bytes",
        "total_metadata_allowance_bytes", "shard_count",
    }
    allocation_valid = (
        type(allocation) is dict and set(allocation) == allocation_keys
        and all(type(allocation[key]) is int and allocation[key] >= 0 for key in allocation)
        and allocation["total_allocated_bytes"] == allocation["fragment_allocated_bytes"] + allocation["metadata_allocated_bytes"]
        and allocation["maximum_shard_metadata_allocated_bytes"] <= METADATA_RESERVE_PER_SHARD
        and allocation["per_shard_metadata_allowance_bytes"] == METADATA_RESERVE_PER_SHARD
        and allocation["total_metadata_allowance_bytes"] == profile.shard_total * METADATA_RESERVE_PER_SHARD
        and allocation["metadata_allocated_bytes"] <= allocation["total_metadata_allowance_bytes"]
        and allocation["shard_count"] == profile.shard_total
    )
    custody = result.get("attempt_and_storage_custody")
    custody_valid = (
        type(custody) is dict
        and set(custody) == {"attempt_path", "storage_reservation_path", "byte_length_each", "file_sha256_each", "content_sha256_each"}
        and custody.get("attempt_path") == "EVIDENCE/ATTEMPT.json"
        and custody.get("storage_reservation_path") == "EVIDENCE/STORAGE_RESERVATION.json"
        and type(custody.get("byte_length_each")) is int and custody["byte_length_each"] > 0
        and _is_sha256(custody.get("file_sha256_each"))
    )
    if custody_valid and type(header_attempt) is dict:
        custody_valid = (
            custody.get("file_sha256_each") == _sha256(_canonical_json(header_attempt))
            and custody.get("content_sha256_each") == header_attempt.get("content_sha256")
        )
    else:
        custody_valid = False
    expected_header = _digest_roster_header(
        profile, result.get("attempt_and_storage_custody"), result.get("evidence_allocation_summary"), header_attempt
    ) if (
        type(result.get("attempt_and_storage_custody")) is dict
        and type(result.get("evidence_allocation_summary")) is dict
        and type(header_attempt) is dict
    ) else None
    replay = None
    if expected_header is not None:
        replay = _validate_archive_digest_rows(root / "DIGEST_ROSTER.jsonl", profile, expected_header)
    finalization_plan = result.get("finalization_plan")
    finalization_reconciliation = result.get("finalization_storage_reconciliation")
    finalization_valid = False
    if (
        type(finalization_plan) is dict
        and type(header_attempt) is dict
        and type(allocation) is dict
        and type(finalization_plan.get("shard_manifests_logical_bytes")) is int
        and type(finalization_plan.get("digest_roster_logical_bytes")) is int
    ):
        expected_finalization_plan = _finalization_plan_record(
            profile,
            header_attempt,
            allocation,
            shard_manifests_bytes=finalization_plan["shard_manifests_logical_bytes"],
            digest_roster_bytes=finalization_plan["digest_roster_logical_bytes"],
        )
        _validate_finalization_plan_record(finalization_plan, expected_finalization_plan)
        _validate_finalization_storage_record(
            finalization_reconciliation,
            finalization_plan,
        )
        finalization_valid = True
    final_files = result.get("final_files")
    final_files_valid = type(final_files) is dict and set(final_files) == set(FINAL_BINARY_NAMES)
    if final_files_valid and replay is not None:
        for name, expected_size in profile.geometry.output_sizes().items():
            record = final_files[name]
            if (
                type(record) is not dict
                or set(record) != {"name", "byte_length", "sha256", "st_blocks"}
                or record.get("name") != name
                or type(record.get("byte_length")) is not int
                or record["byte_length"] != expected_size
                or record.get("sha256") != replay["final_file_sha256"][name]
                or type(record.get("st_blocks")) is not int
                or record["st_blocks"] < 0
                or (expected_size > 0 and record["st_blocks"] * 512 < expected_size)
            ):
                final_files_valid = False
                break
    else:
        final_files_valid = False
    shard_manifests = result.get("shard_manifests")
    shard_manifests_valid = (
        type(shard_manifests) is dict
        and set(shard_manifests) == {"name", "byte_length", "sha256", "record_count", "st_blocks", "order"}
        and shard_manifests.get("name") == "SHARD_MANIFESTS.jsonl"
        and type(shard_manifests.get("byte_length")) is int
        and shard_manifests["byte_length"] > 0
        and type(shard_manifests.get("st_blocks")) is int
        and shard_manifests["st_blocks"] >= 0
        and shard_manifests["st_blocks"] * 512 >= shard_manifests["byte_length"]
        and _is_sha256(shard_manifests.get("sha256"))
        and type(shard_manifests.get("record_count")) is int
        and shard_manifests["record_count"] == profile.shard_total
        and shard_manifests.get("order") == "BATCH_MAJOR_THEN_SHARD_MAJOR_CANONICAL_JSON_LINES"
        and finalization_valid
        and shard_manifests["byte_length"] == finalization_plan["shard_manifests_logical_bytes"]
        and shard_manifests["st_blocks"]
        == finalization_reconciliation["streams"]["FINAL/SHARD_MANIFESTS.jsonl"]["st_blocks"]
    )
    if (
        set(result) != result_keys or result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "SYNTHETIC_IO_WRITER_QUALIFICATION_COMPLETE_NO_DYNAMICS_OR_SCIENCE_CLAIM"
        or result.get("protocol_content_sha256") != protocol["content_sha256"]
        or result.get("profile_content_sha256") != _profile_record(profile)["content_sha256"]
        or not _is_sha256(result.get("final_manifest_content_sha256"))
        or not _is_sha256(result.get("final_manifest_file_sha256"))
        or not final_files_valid or not shard_manifests_valid or replay is None or not finalization_valid
        or not _exact_equal(
            result.get("terminal_manifest_content_sha256_by_batch"),
            replay["terminal_manifest_content_sha256_by_batch"],
        )
        or not _exact_equal(
            result.get("terminal_receipt_content_sha256_by_batch_and_role"),
            replay["terminal_receipt_content_sha256_by_batch_and_role"],
        )
        or not _exact_equal(
            result.get("terminal_first_1k_chain_sha256_by_batch"),
            replay["terminal_first_1k_chain_sha256_by_batch"],
        )
        or type(digest_record) is not dict
        or set(digest_record) != {"name", "byte_length", "sha256", "record_count", "header_record_count", "shard_record_count", "st_blocks", "cap_bytes"}
        or digest_record.get("name") != "DIGEST_ROSTER.jsonl"
        or type(digest_record.get("byte_length")) is not int or digest_record.get("byte_length") != digest_size
        or digest_record.get("sha256") != digest_sha or digest_size >= DIGEST_ROSTER_MAX_BYTES
        or type(digest_record.get("record_count")) is not int or digest_record.get("record_count") != profile.shard_total + 1
        or type(digest_record.get("header_record_count")) is not int or digest_record.get("header_record_count") != 1
        or type(digest_record.get("shard_record_count")) is not int or digest_record.get("shard_record_count") != profile.shard_total
        or type(digest_record.get("st_blocks")) is not int or digest_record["st_blocks"] < 0
        or digest_record["st_blocks"] * 512 < digest_size
        or digest_record["byte_length"] != finalization_plan["digest_roster_logical_bytes"]
        or digest_record["st_blocks"]
        != finalization_reconciliation["streams"]["DIGEST_ROSTER.jsonl"]["st_blocks"]
        or type(digest_record.get("cap_bytes")) is not int or digest_record.get("cap_bytes") != DIGEST_ROSTER_MAX_BYTES
        or expected_header is None or not _exact_equal(header, expected_header)
        or not allocation_valid or not custody_valid
        or result.get("digest_roster_header_content_sha256") != header.get("content_sha256")
        or not _exact_equal(result.get("git_preregistration_custody"), header_attempt.get("git_preregistration_custody"))
        or not _exact_equal(result.get("publication_order"), ["EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json_LAST"])
        or not _exact_equal(result.get("archive_surface"), expected_archive)
        or not _exact_equal(result.get("claims"), _claim_ceiling(profile.exact_production))
        or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result)
    ):
        raise WriterQualificationError("bounded archive custody differs")
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dry-metadata", "qualify", "resume", "validate", "validate-archive"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--externally-preregistered", action="store_true")
    parser.add_argument("--prereg-commit")
    parser.add_argument("--prereg-tag-object")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.output_root)
    elif args.command == "dry-metadata":
        value = production_metadata_plan_v1()
    elif args.command == "qualify":
        value = qualify(
            args.output_root,
            externally_preregistered=args.externally_preregistered,
            prereg_commit=args.prereg_commit,
            prereg_tag_object=args.prereg_tag_object,
        )
    elif args.command == "resume":
        value = qualify(
            args.output_root,
            resume=True,
            externally_preregistered=args.externally_preregistered,
            prereg_commit=args.prereg_commit,
            prereg_tag_object=args.prereg_tag_object,
        )
    elif args.command == "validate":
        value = validate(args.output_root)
    else:
        value = validate_archive_surface(args.output_root)
    print(json.dumps({"status": value.get("status", value.get("schema")), "content_sha256": value.get("content_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
