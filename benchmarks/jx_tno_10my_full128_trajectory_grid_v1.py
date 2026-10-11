#!/usr/bin/env python3
"""Prepare the full-cohort 10 Myr TNO trajectory-acquisition grid.

This module is deliberately *not* a dynamics runner and is deliberately *not*
a classifier.  It freezes the byte geometry, identities, lifecycle vocabulary,
first-1-kyr projection gate, create-only shard format, and held-descriptor
offline reconstruction needed by a later separately preregistered execution
successor.  The only public CLI actions are ``prepare`` and
``validate-prepared``.

Production records contain five reduced massive bodies followed by one nominal
TNO and covariance clones 0..127.  A Cartesian state record is therefore
``134 * 6 * 8 == 6432`` bytes.  Its lifecycle record is exactly 129 bytes.
Terminal removed states remain finite retained trajectory states; they never
receive a dynamical or scientific class in this layer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_trajectory_grid_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_trajectory_grid_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_trajectory_grid_v1"

PROTOCOL_SCHEMA = "jx.tno.10my-full128-trajectory-grid.protocol.v1"
SHARD_SCHEMA = "jx.tno.10my-full128-trajectory-grid.shard.v1"
FINAL_MANIFEST_SCHEMA = "jx.tno.10my-full128-trajectory-grid.final-manifest.v1"
FINAL_RESULT_SCHEMA = "jx.tno.10my-full128-trajectory-grid.final-result.v1"
RECEIPT_SCHEMA = "jx.tno.10my-full128-trajectory-grid.synthetic-receipt.v1"
PROTOCOL_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.protocol.v1\0"
SHARD_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.shard.v1\0"
FINAL_MANIFEST_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.final-manifest.v1\0"
FINAL_RESULT_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.final-result.v1\0"
NODE_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.node.v1\0"
PREFIX_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.first-1k.v1\0"
RECEIPT_DOMAIN = b"jx.tno.10my-full128-trajectory-grid.synthetic-receipt.v1\0"
FILESYSTEM_TRUST_BOUNDARY = "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
SPK_IDS = (
    20090377, 20474640, 54526835, 20541132, 50666516, 20148209,
    20574372, 20303775, 20026181, 20042301, 20026375, 20015874,
)
NATIVE_EPOCHS_JD_TDB_HEX = (
    "0x1.2c195c0000000p+21", "0x1.2bd3dc0000000p+21",
    "0x1.2c4c140000000p+21", "0x1.2c0e240000000p+21",
    "0x1.2be0a40000000p+21", "0x1.2b62c40000000p+21",
    "0x1.2c15dc0000000p+21", "0x1.2c247c0000000p+21",
    "0x1.2c11640000000p+21", "0x1.2c16a40000000p+21",
    "0x1.2c0b4c0000000p+21", "0x1.2c0f940000000p+21",
)
REDUCED_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER", "SATURN_BARYCENTER", "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
REDUCED_NAIF_IDS = (10, 5, 6, 7, 8)
REDUCED_GM_HEX = (
    "0x1.3bd13a81d2261p+5", "0x1.34c6609ebf23fp-5",
    "0x1.71d0e2fd3b84dp-7", "0x1.c3d8f106aa648p-10",
    "0x1.0a8c8f8c38c6ap-9",
)

NOMINAL_AND_CLONE_COUNT = 129
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
HOLDOUT_LANES = (0,) + tuple(index + 1 for index in HOLDOUT_CLONE_INDICES)
ACTIVE = 0
REMOVED_SOURCE_RADIUS = 1
REMOVED_OUTER_BOUNDARY = 2
LIFECYCLE_CODES = {
    ACTIVE: "ACTIVE",
    REMOVED_SOURCE_RADIUS: "REMOVED_SOURCE_RADIUS",
    REMOVED_OUTER_BOUNDARY: "REMOVED_OUTER_BOUNDARY",
}

FRAGMENT_NAMES = (
    "B22_STATES.fragment",
    "B22_NODE_SHA256.fragment",
    "B22_LIFECYCLE.fragment",
    "ML_SHORT_STATES.fragment",
    "ML_SHORT_LIFECYCLE.fragment",
)
FINAL_BINARY_NAMES = (
    "B22_STATES.f64le",
    "B22_NODE_SHA256.raw",
    "B22_LIFECYCLE.u8",
    "ML_SHORT_STATES.f64le",
    "ML_SHORT_LIFECYCLE.u8",
)
FINAL_ROSTER = frozenset(
    ("PROTOCOL.json", *FINAL_BINARY_NAMES, "SHARD_MANIFESTS.jsonl", "MANIFEST.json", "RESULT.json")
)
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})
RECEIPT_ROLES = (
    "start_continuation", "end_continuation", "source", "route", "runtime"
)

BOUND_FILES = (
    ("grid_adapter_v1_source", "benchmarks/jx_tno_10my_reduced5_grid_adapter_v1.py", 111_493, "324498cd36726e461ab6910fdf0fe13a1b024346293558f535bea557cdb61b3a"),
    ("grid_adapter_v1_test", "tests/test_tno_10my_reduced5_grid_adapter_v1.py", 37_454, "909e6f63ff229e8909245001728acd58a8d4bb7df80993a14b578659c6dd96b8"),
    ("grid_adapter_v1_protocol", "runs/tno_10my_reduced5_grid_adapter_v1/PROTOCOL.json", 11_844, "6b10412f06f3c398646179fd163347ac269c03b08486c867e4da821e082be3f2"),
    ("pilot_v2_source", "benchmarks/jx_tno_10my_reduced5_classification_pilot_v2.py", 59_307, "33ad38cbc89a15a83cf180161544a721b8635affd0d4890a8a39d04389694332"),
    ("pilot_v2_test", "tests/test_tno_10my_reduced5_classification_pilot_v2.py", 28_773, "060793a45cd91590d484a610b9c6d48f6fae6c2a468c6e2663541599a831fbd8"),
    ("pilot_v2_protocol", "runs/tno_10my_reduced5_classification_pilot_v2/PROTOCOL.json", 34_347, "b510cbf7a62f45d8713ca8ef02dd2229f5a79966d00d2482843e814b939cded6"),
    ("lifecycle_v3_source", "benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v3.py", 151_995, "02e72fb6a635b74dd31dc364e18cbaf03c5b9e1833caf5aba4a063d7198e47c7"),
    ("lifecycle_v3_test", "tests/test_tno_1kyr_reduced5_lifecycle_gate_v3.py", 42_017, "b4eb43b00e1fab0055a43cc4f26db2bef6bf29d51656bcfab1d792596b7845a8"),
    ("lifecycle_v3_protocol", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PROTOCOL.json", 9_178, "9c1e7f77b3a2fd9f6af5542aa2ee3fa7dc00d669d496c809da28b675c8519199"),
    ("lifecycle_v3_result", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json", 66_669, "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4"),
    ("lifecycle_v3_prefix", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin", 169_344, "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b"),
    ("covariance_clones", "runs/tno_covariance_clones_v1/CLONES.json", 531_609, "395ce9914df0efe235e2144c24abd157f5aa3ba87beba87fa39891ec42312014"),
    ("stage0_states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
)
BOUND_CONTENT = {
    "grid_adapter_v1_protocol": "f13b551ea074ce6233c9f3b8aa5e0b75421aa26639141564312087af331812a9",
    "pilot_v2_protocol": "eb6fe0b5ac6485f4175ebe44ab779d88ebf92986ef279d949e8eaea2b2a5c7d3",
    "lifecycle_v3_protocol": "538e076274ab5e1c850cce069405ba67c3cab41226fa684a686aaaa81cdc32d9",
    "lifecycle_v3_result": "d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264",
    "covariance_clones": "9f1c0df03a2001f6249aafd423c4c9feb8bde9a75467416d35b81b88bb2d59d8",
    "stage0_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
}

V3_PREFIX_PATH = PROJECT_ROOT / "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin"
V3_PREFIX_BYTES = 169_344
V3_PREFIX_SHA256 = "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b"
V3_RESULT_PATH = PROJECT_ROOT / "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json"
V3_RESULT_SHA256 = "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4"
V3_RESULT_CONTENT_SHA256 = "d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264"
V3_CHECKPOINT_YEARS = tuple(range(0, 1_001, 50))
V3_RECORD_BYTES = 672


class TrajectoryGridError(RuntimeError):
    """The trajectory acquisition contract failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise TrajectoryGridError("JSON has a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrajectoryGridError(f"JSON has non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrajectoryGridError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise TrajectoryGridError("JSON root must be an exact object")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrajectoryGridError(f"{label} must be an exact integer >= {minimum}")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TrajectoryGridError(f"{label} must be nonempty text")
    return value


def _exact_equal(left: object, right: object) -> bool:
    """Recursive equality that never aliases bool and int."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise TrajectoryGridError(f"{label} has a symlink or non-directory ancestor")


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


class _HeldFile:
    def __init__(self, parent_fd: int, name: str, label: str) -> None:
        if type(name) is not str or not name or "/" in name or name in (".", ".."):
            raise TrajectoryGridError(f"{label} leaf name differs")
        self.parent_fd = parent_fd
        self.name = name
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(name, flags, dir_fd=parent_fd)
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(self.fd)
            raise TrajectoryGridError(f"{label} is not a singly-linked regular file")
        self.before = _identity(metadata)
        self.size = metadata.st_size

    def pread(self, count: int, offset: int) -> bytes:
        _require_int(count, "pread count")
        _require_int(offset, "pread offset")
        payload = os.pread(self.fd, count, offset)
        if len(payload) != count:
            raise TrajectoryGridError(f"{self.label} ended early")
        return payload

    def read_all(self, *, maximum: int = 128_000_000) -> bytes:
        if self.size > maximum:
            raise TrajectoryGridError(f"{self.label} exceeds the bounded read cap")
        return self.pread(self.size, 0)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < self.size:
            chunk = os.pread(self.fd, min(1 << 20, self.size - offset), offset)
            if not chunk:
                raise TrajectoryGridError(f"{self.label} ended during hashing")
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    def close(self) -> None:
        try:
            after = os.fstat(self.fd)
            try:
                path_after = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise TrajectoryGridError(f"{self.label} directory entry disappeared") from exc
            if _identity(after) != self.before or _identity(path_after) != self.before:
                raise TrajectoryGridError(f"{self.label} changed while held")
        finally:
            os.close(self.fd)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


@contextmanager
def _held_directory_at(
    parent_fd: int,
    name: str,
    label: str,
    *,
    expected_roster: frozenset[str] | None = None,
    expected_identity: tuple[int, int, int] | None = None,
) -> Iterator[int]:
    if type(parent_fd) is not int or type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise TrajectoryGridError(f"{label} held-directory arguments differ")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    lexical = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    lexical_identity = _directory_identity(lexical)
    if not stat.S_ISDIR(lexical.st_mode) or (
        expected_identity is not None and lexical_identity != expected_identity
    ):
        raise TrajectoryGridError(f"{label} directory identity differs before open")
    fd = os.open(name, flags, dir_fd=parent_fd)
    before = os.fstat(fd)
    if (
        not stat.S_ISDIR(before.st_mode) or before.st_nlink < 1
        or _directory_identity(before) != lexical_identity
        or (expected_identity is not None and _directory_identity(before) != expected_identity)
    ):
        os.close(fd)
        raise TrajectoryGridError(f"{label} was substituted before open")
    try:
        if expected_roster is not None and frozenset(os.listdir(fd)) != expected_roster:
            raise TrajectoryGridError(f"{label} roster differs")
        yield fd
        after = os.fstat(fd)
        if _directory_identity(after) != _directory_identity(before):
            raise TrajectoryGridError(f"{label} identity changed")
        try:
            path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise TrajectoryGridError(f"{label} directory entry disappeared") from exc
        if _directory_identity(path_after) != _directory_identity(before):
            raise TrajectoryGridError(f"{label} directory entry changed while held")
        if expected_roster is not None and frozenset(os.listdir(fd)) != expected_roster:
            raise TrajectoryGridError(f"{label} final roster differs")
    finally:
        os.close(fd)


@contextmanager
def _held_directory(path: Path, label: str, *, expected_roster: frozenset[str] | None = None) -> Iterator[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_lexical = os.stat(absolute.parent, follow_symlinks=False)
    parent_fd = os.open(absolute.parent, flags)
    parent_before = os.fstat(parent_fd)
    if _directory_identity(parent_lexical) != _directory_identity(parent_before):
        os.close(parent_fd)
        raise TrajectoryGridError(f"{label} parent was substituted before open")
    try:
        with _held_directory_at(
            parent_fd, absolute.name, label, expected_roster=expected_roster
        ) as fd:
            yield fd
        parent_after = os.fstat(parent_fd)
        parent_path_after = os.stat(absolute.parent, follow_symlinks=False)
        if (
            _directory_identity(parent_after) != _directory_identity(parent_before)
            or _directory_identity(parent_path_after) != _directory_identity(parent_before)
        ):
            raise TrajectoryGridError(f"{label} parent changed while held")
    finally:
        os.close(parent_fd)


@contextmanager
def _create_and_hold_directory_at(parent_fd: int, name: str, label: str) -> Iterator[int]:
    if type(parent_fd) is not int or type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise TrajectoryGridError(f"{label} create-directory arguments differ")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise TrajectoryGridError(f"{label} already exists") from exc
    created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(created.st_mode):
        raise TrajectoryGridError(f"{label} created entry is not a directory")
    created_identity = _directory_identity(created)
    os.fsync(parent_fd)
    with _held_directory_at(
        parent_fd, name, label, expected_identity=created_identity
    ) as fd:
        yield fd


@contextmanager
def _held_relative(root_fd: int, relative_path: str, label: str) -> Iterator[_HeldFile]:
    if type(relative_path) is not str:
        raise TrajectoryGridError(f"{label} relative path is not text")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise TrajectoryGridError(f"{label} relative path escapes its root")
    opened: list[int] = []
    parent_fd = root_fd
    try:
        for component in pure.parts[:-1]:
            child = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            opened.append(child)
            parent_fd = child
        held = _HeldFile(parent_fd, pure.parts[-1], label)
        try:
            yield held
        finally:
            held.close()
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _read_regular(path: Path, label: str, *, maximum: int = 128_000_000) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_ancestors(absolute, label)
    parent_fd = os.open(absolute.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        held = _HeldFile(parent_fd, absolute.name, label)
        try:
            return held.read_all(maximum=maximum)
        finally:
            held.close()
    finally:
        os.close(parent_fd)


@dataclass(frozen=True, slots=True)
class GridGeometry:
    batch_count: int
    total_steps: int
    shard_steps: int
    b22_interval_steps: int
    short_interval_steps: int
    short_final_step: int
    massive_count: int = 5
    tracer_count: int = NOMINAL_AND_CLONE_COUNT

    def __post_init__(self) -> None:
        for name in (
            "batch_count", "total_steps", "shard_steps", "b22_interval_steps",
            "short_interval_steps", "short_final_step", "massive_count", "tracer_count",
        ):
            _require_int(getattr(self, name), name, minimum=1)
        if self.total_steps % self.shard_steps or self.total_steps % self.b22_interval_steps:
            raise TrajectoryGridError("horizon must divide exactly into shards and B22 cadence")
        if self.short_final_step > self.total_steps or self.short_final_step % self.short_interval_steps:
            raise TrajectoryGridError("short-grid endpoint/cadence differs")
        if self.b22_interval_steps % self.short_interval_steps:
            raise TrajectoryGridError("B22 and short grids are not nested")
        if self.shard_steps % self.b22_interval_steps or self.shard_steps % self.short_interval_steps:
            raise TrajectoryGridError("shard boundaries are off a retention grid")

    @property
    def shard_count(self) -> int:
        return self.total_steps // self.shard_steps

    @property
    def body_count(self) -> int:
        return self.massive_count + self.tracer_count

    @property
    def state_record_bytes(self) -> int:
        return self.body_count * 6 * 8

    @property
    def lifecycle_record_bytes(self) -> int:
        return self.tracer_count

    @property
    def b22_node_count(self) -> int:
        return self.total_steps // self.b22_interval_steps + 1

    @property
    def short_node_count(self) -> int:
        return self.short_final_step // self.short_interval_steps + 1

    @property
    def union_node_count(self) -> int:
        overlap = self.short_final_step // self.b22_interval_steps + 1
        return self.b22_node_count + self.short_node_count - overlap

    def expected_steps(self, shard_index: int) -> tuple[int, ...]:
        shard = _require_int(shard_index, "shard_index")
        if shard >= self.shard_count:
            raise TrajectoryGridError("shard index exceeds geometry")
        start = shard * self.shard_steps
        end = start + self.shard_steps
        lower = start if shard == 0 else start + 1
        first_b22 = ((lower + self.b22_interval_steps - 1) // self.b22_interval_steps) * self.b22_interval_steps
        values = set(range(first_b22, end + 1, self.b22_interval_steps))
        short_end = min(end, self.short_final_step)
        if lower <= short_end:
            first_short = ((lower + self.short_interval_steps - 1) // self.short_interval_steps) * self.short_interval_steps
            values.update(range(first_short, short_end + 1, self.short_interval_steps))
        return tuple(sorted(values))

    def is_b22(self, step: int) -> bool:
        return step % self.b22_interval_steps == 0

    def is_short(self, step: int) -> bool:
        return step <= self.short_final_step and step % self.short_interval_steps == 0

    def output_sizes(self) -> dict[str, int]:
        return {
            "B22_STATES.f64le": self.batch_count * self.b22_node_count * self.state_record_bytes,
            "B22_NODE_SHA256.raw": self.batch_count * self.b22_node_count * 32,
            "B22_LIFECYCLE.u8": self.batch_count * self.b22_node_count * self.lifecycle_record_bytes,
            "ML_SHORT_STATES.f64le": self.batch_count * self.short_node_count * self.state_record_bytes,
            "ML_SHORT_LIFECYCLE.u8": self.batch_count * self.short_node_count * self.lifecycle_record_bytes,
        }


PRODUCTION_GEOMETRY = GridGeometry(
    batch_count=12,
    total_steps=640_000_000,
    shard_steps=640_000,
    b22_interval_steps=12_800,
    short_interval_steps=3_200,
    short_final_step=32_000_000,
)


def _geometry_record(geometry: GridGeometry) -> dict[str, int]:
    return {**asdict(geometry), "shard_count": geometry.shard_count, "body_count": geometry.body_count,
            "state_record_bytes": geometry.state_record_bytes, "lifecycle_record_bytes": geometry.lifecycle_record_bytes,
            "b22_node_count": geometry.b22_node_count, "short_node_count": geometry.short_node_count,
            "union_node_count": geometry.union_node_count}


def _require_geometry(geometry: GridGeometry, *, test_only_geometry: bool) -> None:
    if type(geometry) is not GridGeometry:
        raise TrajectoryGridError("geometry must be the exact contract type")
    if geometry != PRODUCTION_GEOMETRY and test_only_geometry is not True:
        raise TrajectoryGridError("non-production geometry requires explicit test-only authority")


def tracer_identity_v1(batch_order: int, lane: int) -> dict[str, object]:
    batch = _require_int(batch_order, "batch_order")
    lane_index = _require_int(lane, "tracer lane")
    if batch >= len(OBJECT_IDS) or lane_index >= NOMINAL_AND_CLONE_COUNT:
        raise TrajectoryGridError("tracer identity index exceeds the production cohort")
    return {
        "batch_order": batch,
        "object_id": OBJECT_IDS[batch],
        "spk_id": SPK_IDS[batch],
        "tracer_lane": lane_index,
        "sample_kind": "NOMINAL" if lane_index == 0 else "COVARIANCE_CLONE",
        "clone_index": None if lane_index == 0 else lane_index - 1,
        "stage0_state_order": batch * NOMINAL_AND_CLONE_COUNT + lane_index,
    }


def state_record_v1(rows: Sequence[Sequence[float]], geometry: GridGeometry = PRODUCTION_GEOMETRY) -> bytes:
    if type(rows) not in (tuple, list) or len(rows) != geometry.body_count:
        raise TrajectoryGridError("state record body count differs")
    values: list[float] = []
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 6:
            raise TrajectoryGridError("state row must be an exact six-vector")
        for value in row:
            if type(value) is not float or not math.isfinite(value):
                raise TrajectoryGridError("state values must be finite built-in floats")
            values.append(value)
    payload = struct.pack(f"<{len(values)}d", *values)
    if len(payload) != geometry.state_record_bytes:
        raise TrajectoryGridError("state record byte length differs")
    return payload


def decode_state_record_v1(payload: bytes, geometry: GridGeometry = PRODUCTION_GEOMETRY) -> tuple[tuple[float, ...], ...]:
    if type(payload) is not bytes or len(payload) != geometry.state_record_bytes:
        raise TrajectoryGridError("state record byte length differs")
    values = struct.unpack(f"<{geometry.body_count * 6}d", payload)
    if any(not math.isfinite(value) for value in values):
        raise TrajectoryGridError("state record contains a non-finite value")
    return tuple(tuple(values[index:index + 6]) for index in range(0, len(values), 6))


def lifecycle_record_v1(codes: Sequence[int], geometry: GridGeometry = PRODUCTION_GEOMETRY) -> bytes:
    if type(codes) not in (tuple, list) or len(codes) != geometry.tracer_count:
        raise TrajectoryGridError("lifecycle record tracer count differs")
    values: list[int] = []
    for value in codes:
        if type(value) is not int or value not in LIFECYCLE_CODES:
            raise TrajectoryGridError("lifecycle record has an unknown or non-integer code")
        values.append(value)
    payload = bytes(values)
    if len(payload) != geometry.lifecycle_record_bytes:
        raise TrajectoryGridError("lifecycle record byte length differs")
    return payload


def decode_lifecycle_record_v1(payload: bytes, geometry: GridGeometry = PRODUCTION_GEOMETRY) -> tuple[int, ...]:
    if type(payload) is not bytes or len(payload) != geometry.lifecycle_record_bytes:
        raise TrajectoryGridError("lifecycle record byte length differs")
    values = tuple(payload)
    if any(value not in LIFECYCLE_CODES for value in values):
        raise TrajectoryGridError("lifecycle record contains an unknown code")
    return values


def _require_terminal_transition(previous: tuple[int, ...] | None,
                                 current: tuple[int, ...]) -> None:
    if previous is None:
        return
    if len(previous) != len(current):
        raise TrajectoryGridError("lifecycle transition roster differs")
    for before, after in zip(previous, current, strict=True):
        if before != ACTIVE and after != before:
            raise TrajectoryGridError("terminal lifecycle code reverted or changed")


def node_digest_v1(batch_order: int, global_step: int, state_payload: bytes,
                   lifecycle_payload: bytes, geometry: GridGeometry = PRODUCTION_GEOMETRY) -> bytes:
    batch = _require_int(batch_order, "batch_order")
    step = _require_int(global_step, "global_step")
    if batch >= geometry.batch_count or not geometry.is_b22(step) or step > geometry.total_steps:
        raise TrajectoryGridError("node identity is outside the B22 grid")
    decode_state_record_v1(state_payload, geometry)
    decode_lifecycle_record_v1(lifecycle_payload, geometry)
    metadata: dict[str, Any] = {
        "schema": "jx.tno.10my-full128-trajectory-grid.node-identity.v1",
        "batch_order": batch,
        "global_step": step,
        "geometry": _geometry_record(geometry),
        "object_id": OBJECT_IDS[batch] if geometry == PRODUCTION_GEOMETRY else f"SYNTHETIC_BATCH_{batch}",
        "tracer_order": "NOMINAL_THEN_COVARIANCE_CLONES_0_THROUGH_127" if geometry == PRODUCTION_GEOMETRY else "SYNTHETIC_TEST_ORDER",
        "state_field_order": "BODY_MAJOR_X_Y_Z_VX_VY_VZ_BINARY64_LITTLE_ENDIAN",
        "lifecycle_field_order": "TRACER_LANE_MAJOR_UINT8",
    }
    digest = hashlib.sha256()
    digest.update(NODE_DOMAIN)
    digest.update(_canonical_json(metadata))
    digest.update(state_payload)
    digest.update(lifecycle_payload)
    return digest.digest()


def project_holdout_record_v1(full_state_payload: bytes) -> bytes:
    rows = decode_state_record_v1(full_state_payload, PRODUCTION_GEOMETRY)
    projected = rows[:5] + tuple(rows[5 + lane] for lane in HOLDOUT_LANES)
    payload = struct.pack("<84d", *(value for row in projected for value in row))
    if len(payload) != V3_RECORD_BYTES:
        raise TrajectoryGridError("holdout projection produced the wrong byte length")
    return payload


def validate_first_1000_year_projection_v1(batch_order: int,
                                            checkpoints: Sequence[tuple[int, bytes, bytes]]) -> dict[str, Any]:
    batch = _require_int(batch_order, "batch_order")
    if batch >= len(OBJECT_IDS) or type(checkpoints) not in (tuple, list) or len(checkpoints) != len(V3_CHECKPOINT_YEARS):
        raise TrajectoryGridError("first-1k projection roster differs")
    v3_result_payload = _read_regular(V3_RESULT_PATH, "retained V3 result")
    v3_result = _strict_json(v3_result_payload)
    v3_files = v3_result.get("files")
    v3_claims = v3_result.get("claims")
    if (
        _sha256(v3_result_payload) != V3_RESULT_SHA256
        or v3_result_payload != _canonical_json(v3_result)
        or v3_result.get("content_sha256") != V3_RESULT_CONTENT_SHA256
        or v3_result.get("status") != "ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED"
        or type(v3_files) is not dict
        or v3_files.get("PREFIX_STATES.bin") != {"bytes": V3_PREFIX_BYTES, "sha256": V3_PREFIX_SHA256}
        or type(v3_claims) is not dict
        or v3_claims.get("one_thousand_year_lifecycle_gate_passed") is not True
        or v3_claims.get("ten_myr_execution_authorized") is not False
        or v3_claims.get("classification_generated") is not False
        or v3_claims.get("science_result") is not False
    ):
        raise TrajectoryGridError("retained V3 prefix/result custody differs")
    parent_fd = os.open(V3_PREFIX_PATH.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    digests: list[str] = []
    try:
        held = _HeldFile(parent_fd, V3_PREFIX_PATH.name, "retained V3 prefix")
        try:
            if held.size != V3_PREFIX_BYTES or held.sha256() != V3_PREFIX_SHA256:
                raise TrajectoryGridError("retained V3 prefix identity differs")
            for index, (expected_year, item) in enumerate(zip(V3_CHECKPOINT_YEARS, checkpoints, strict=True)):
                if type(item) is not tuple or len(item) != 3:
                    raise TrajectoryGridError("first-1k checkpoint shape differs")
                year, state_payload, lifecycle_payload = item
                if type(year) is not int or year != expected_year:
                    raise TrajectoryGridError("first-1k checkpoint year/order differs")
                lifecycle = decode_lifecycle_record_v1(lifecycle_payload)
                if tuple(lifecycle[lane] for lane in HOLDOUT_LANES) != (ACTIVE,) * len(HOLDOUT_LANES):
                    raise TrajectoryGridError("a projected V3 tracer is not active")
                projected = project_holdout_record_v1(state_payload)
                offset = (batch * len(V3_CHECKPOINT_YEARS) + index) * V3_RECORD_BYTES
                if projected != held.pread(V3_RECORD_BYTES, offset):
                    raise TrajectoryGridError("full-record holdout projection differs from retained V3")
                digests.append(_sha256(projected))
        finally:
            held.close()
    finally:
        os.close(parent_fd)
    record: dict[str, Any] = {
        "schema": "jx.tno.10my-full128-trajectory-grid.first-1k.v1",
        "batch_order": batch,
        "object_id": OBJECT_IDS[batch],
        "projected_lanes": list(HOLDOUT_LANES),
        "checkpoint_years": list(V3_CHECKPOINT_YEARS),
        "checkpoint_sha256": digests,
        "retained_v3_prefix_sha256": V3_PREFIX_SHA256,
        "passed": True,
        "content_sha256": "",
    }
    record["content_sha256"] = _domain_hash(PREFIX_DOMAIN, record)
    return record


def _bound_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, relative, expected_bytes, expected_sha in BOUND_FILES:
        payload = _read_regular(PROJECT_ROOT / relative, role)
        if len(payload) != expected_bytes or _sha256(payload) != expected_sha:
            raise TrajectoryGridError(f"bound input differs: {relative}")
        record: dict[str, Any] = {
            "role": role, "relative_path": relative,
            "byte_length": expected_bytes, "file_sha256": expected_sha,
        }
        if role in BOUND_CONTENT:
            parsed = _strict_json(payload)
            if parsed.get("content_sha256") != BOUND_CONTENT[role]:
                raise TrajectoryGridError(f"bound content identity differs: {relative}")
            record["content_sha256"] = BOUND_CONTENT[role]
        records.append(record)
    return records


def _implementation_record(path: Path, role: str) -> dict[str, Any]:
    payload = _read_regular(path, role)
    return {
        "role": role,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def build_protocol() -> dict[str, Any]:
    sizes = PRODUCTION_GEOMETRY.output_sizes()
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARE_ONLY_TRAJECTORY_ACQUISITION_MECHANICS_NO_DYNAMICS",
        "objective": "Freeze a full 12-by-129 reduced-five trajectory byte grid and evidence chain; do not execute dynamics or classify objects.",
        "implementation": {
            **_implementation_record(SCRIPT_PATH, "trajectory_grid_source"),
            "test": _implementation_record(TEST_PATH, "trajectory_grid_test"),
            "cli_surface": ["prepare", "validate-prepared"],
            "dynamics_surface_present": False,
            "classifier_surface_present": False,
            "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        },
        "bound_inputs": _bound_records(),
        "cohort": {
            "batch_count": 12,
            "object_ids_in_order": list(OBJECT_IDS),
            "spk_ids_in_order": list(SPK_IDS),
            "native_epochs_jd_tdb_hex": list(NATIVE_EPOCHS_JD_TDB_HEX),
            "tracer_count_per_batch": 129,
            "total_tracer_count": 1_548,
            "tracer_order": "NOMINAL_THEN_COVARIANCE_CLONES_0_THROUGH_127",
            "identity_formula": "state_order=batch_order*129+tracer_lane; clone_index=null_if_lane_0_else_lane-1",
            "v3_holdout_lanes": list(HOLDOUT_LANES),
        },
        "dynamics_contract_not_executed": {
            "ephemeris": "DE441",
            "massive_model": "SUN_PLUS_INNER_GM_AT_SUN_THEN_JUPITER_SATURN_URANUS_NEPTUNE",
            "reduced_body_ids": list(REDUCED_BODY_IDS),
            "reduced_naif_ids": list(REDUCED_NAIF_IDS),
            "reduced_gm_hex": list(REDUCED_GM_HEX),
            "time_scale": "TDB",
            "time_unit": "JULIAN_YEAR_EXACTLY_31557600_SI_SECONDS",
            "step_julian_years_exact": "1/64",
            "horizon_julian_years": 10_000_000,
            "massless_noninteracting_tracers": True,
        },
        "geometry": {
            **_geometry_record(PRODUCTION_GEOMETRY),
            "shards_total": PRODUCTION_GEOMETRY.batch_count * PRODUCTION_GEOMETRY.shard_count,
            "state_record_bytes": 6_432,
            "lifecycle_record_bytes": 129,
            "b22_cadence_julian_years": 200,
            "ml_short_cadence_julian_years": 50,
            "ml_short_horizon_julian_years": 500_000,
            "output_file_bytes": sizes,
            "fixed_binary_bytes": sum(sizes.values()),
            "fixed_binary_bytes_expected": 4_743_277_848,
            "no_element_file_in_this_version": True,
        },
        "record_contract": {
            "state_order": "FIVE_MASSIVE_THEN_NOMINAL_THEN_CLONES_0_THROUGH_127",
            "state_fields": "BODY_MAJOR_X_Y_Z_VX_VY_VZ_FLOAT64_LITTLE_ENDIAN",
            "lifecycle_order": "NOMINAL_THEN_CLONES_0_THROUGH_127_UINT8",
            "lifecycle_codes": {str(key): value for key, value in LIFECYCLE_CODES.items()},
            "removed_terminal_state_policy": "RETAIN_EXACT_FINITE_TERMINAL_STATE_NEVER_CLASSIFY",
            "node_digest": "DOMAIN_SEPARATED_SHA256_OF_IDENTITY_STATE_AND_LIFECYCLE",
        },
        "retention": {
            "b22": "steps_12800*k_for_k_0_through_50000",
            "ml_short": "steps_3200*j_for_j_0_through_10000",
            "overlap": "ML_SHORT[4*k]_BITWISE_EQUALS_B22[k]_FOR_k_0_THROUGH_2500_FOR_STATE_AND_LIFECYCLE",
            "first_1k_gate": "PROJECT_FULL_RECORD_TO_V3_LANES_AT_YEARS_0_50_..._1000_AND_REQUIRE_BITWISE_EQUALITY",
        },
        "evidence": {
            "shard_manifest_retention": "FULL_CANONICAL_MANIFEST_ROWS_IN_SHARD_MANIFESTS_JSONL",
            "required_receipt_locators": list(RECEIPT_ROLES),
            "locator_fields": ["role", "relative_path", "schema", "byte_length", "file_sha256", "content_sha256"],
            "filesystem": "CREATE_AT_HELD_PARENT_CAPTURE_CREATED_INODE_OPENAT_O_NOFOLLOW_AND_RECHECK_EVERY_OPENED_DIRECTORY_ENTRY",
            "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
            "publication": "CREATE_ONLY_PROTOCOL_FIRST_RESULT_COMMIT_LAST;PRE_RESULT_FAILURE_HAS_NO_RESULT;POST_RESULT_VALIDATION_IS_FAIL_CLOSED_AND_NEVER_DELETES_DESTINATIONS",
        },
        "claims": {
            "trajectory_acquisition_mechanics_defined": True,
            "trajectory_acquisition_execution_authorized": False,
            "dynamics_executed": False,
            "ten_myr_trajectory_acquired": False,
            "ten_myr_execution_authorized": False,
            "classification_generated": False,
            "resonance_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "content_sha256": "",
    }
    if sizes != {
        "B22_STATES.f64le": 3_859_277_184,
        "B22_NODE_SHA256.raw": 19_200_384,
        "B22_LIFECYCLE.u8": 77_401_548,
        "ML_SHORT_STATES.f64le": 771_917_184,
        "ML_SHORT_LIFECYCLE.u8": 15_481_548,
    }:
        raise TrajectoryGridError("production geometry arithmetic differs")
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict or not _exact_equal(value, build_protocol()):
        raise TrajectoryGridError("protocol does not reconstruct exactly")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise TrajectoryGridError("protocol content identity differs")
    return str(value["content_sha256"])


def _write_create_only(directory_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not name or "/" in name:
        raise TrajectoryGridError("create-only write arguments differ")
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise TrajectoryGridError("create-only write made no progress")
            view = view[written:]
        os.fsync(fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != len(payload):
            raise TrajectoryGridError("created file identity differs")
    finally:
        os.close(fd)
    return {"name": name, "byte_length": len(payload), "sha256": _sha256(payload)}


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    _require_no_symlink_ancestors(root, "prepared output")
    with _held_directory(root.parent, "prepared output parent") as parent_fd:
        with _create_and_hold_directory_at(parent_fd, root.name, "prepared output root") as root_fd:
            protocol = build_protocol()
            _write_create_only(root_fd, "PROTOCOL.json", _canonical_json(protocol))
            os.fsync(root_fd)
    validate_prepared(root)
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared output", expected_roster=PREPARED_ROSTER) as root_fd:
        held = _HeldFile(root_fd, "PROTOCOL.json", "prepared protocol")
        try:
            payload = held.read_all(maximum=1_000_000)
        finally:
            held.close()
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TrajectoryGridError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def write_synthetic_receipt_v1(evidence_root: Path, relative_path: str, role: str,
                               payload_label: str) -> dict[str, Any]:
    """Create one bounded receipt for tests; production successors provide theirs."""
    if role not in RECEIPT_ROLES and role != "predecessor_manifest":
        raise TrajectoryGridError("receipt role differs")
    root = Path(os.path.abspath(os.fspath(evidence_root)))
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or len(pure.parts) != 1:
        raise TrajectoryGridError("synthetic receipt must be a root leaf")
    record: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "role": role, "payload_label": _require_text(payload_label, "payload label"),
        "content_sha256": "",
    }
    record["content_sha256"] = _domain_hash(RECEIPT_DOMAIN, record)
    payload = _canonical_json(record)
    with _held_directory(root, "synthetic evidence root") as root_fd:
        _write_create_only(root_fd, relative_path, payload)
    return {
        "role": role, "relative_path": relative_path, "schema": RECEIPT_SCHEMA,
        "byte_length": len(payload), "file_sha256": _sha256(payload),
        "content_sha256": record["content_sha256"],
    }


def validate_receipt_locator_v1(evidence_root_fd: int, locator: object, expected_role: str) -> dict[str, Any]:
    keys = {"role", "relative_path", "schema", "byte_length", "file_sha256", "content_sha256"}
    if type(locator) is not dict or set(locator) != keys or locator.get("role") != expected_role:
        raise TrajectoryGridError(f"{expected_role} receipt locator shape/role differs")
    relative = locator.get("relative_path")
    schema = locator.get("schema")
    length = locator.get("byte_length")
    if type(relative) is not str or type(schema) is not str or type(length) is not int or length <= 0:
        raise TrajectoryGridError(f"{expected_role} receipt locator types differ")
    if not _is_sha256(locator.get("file_sha256")) or not _is_sha256(locator.get("content_sha256")):
        raise TrajectoryGridError(f"{expected_role} receipt locator digests differ")
    with _held_relative(evidence_root_fd, relative, f"{expected_role} receipt") as held:
        if held.size != length or held.sha256() != locator["file_sha256"]:
            raise TrajectoryGridError(f"{expected_role} receipt file identity differs")
        payload = held.read_all(maximum=16_000_000)
    value = _strict_json(payload)
    if payload != _canonical_json(value) or value.get("schema") != schema or value.get("content_sha256") != locator["content_sha256"]:
        raise TrajectoryGridError(f"{expected_role} receipt content identity differs")
    return dict(locator)


def _validate_receipt_locator_path(evidence_root: Path, locator: object, expected_role: str) -> dict[str, Any]:
    with _held_directory(evidence_root, "receipt evidence root") as root_fd:
        return validate_receipt_locator_v1(root_fd, locator, expected_role)


class _FragmentWriter:
    def __init__(self, directory_fd: int, name: str) -> None:
        self.name = name
        self.fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
        self.digest = hashlib.sha256()
        self.length = 0
        self.closed = False

    def write(self, payload: bytes) -> None:
        if self.closed or type(payload) is not bytes:
            raise TrajectoryGridError("fragment write differs")
        view = memoryview(payload)
        while view:
            count = os.write(self.fd, view)
            if count <= 0:
                raise TrajectoryGridError("fragment write made no progress")
            view = view[count:]
        self.digest.update(payload)
        self.length += len(payload)

    def finish(self) -> dict[str, Any]:
        if self.closed:
            raise TrajectoryGridError("fragment writer already closed")
        os.fsync(self.fd)
        metadata = os.fstat(self.fd)
        os.close(self.fd)
        self.closed = True
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != self.length:
            raise TrajectoryGridError("fragment identity differs")
        return {"name": self.name, "byte_length": self.length, "sha256": self.digest.hexdigest()}

    def abort(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def _shard_name(batch_order: int, shard_index: int) -> str:
    return f"shard-b{batch_order:03d}-s{shard_index:04d}"


def write_shard_v1(evidence_root: Path, *, batch_order: int, shard_index: int,
                   checkpoints: Iterable[tuple[int, bytes, bytes]],
                   receipt_locators: Mapping[str, object],
                   geometry: GridGeometry = PRODUCTION_GEOMETRY,
                   test_only_geometry: bool = False) -> dict[str, Any]:
    _require_geometry(geometry, test_only_geometry=test_only_geometry)
    batch = _require_int(batch_order, "batch_order")
    shard = _require_int(shard_index, "shard_index")
    if batch >= geometry.batch_count or shard >= geometry.shard_count:
        raise TrajectoryGridError("shard identity exceeds geometry")
    if type(receipt_locators) is not dict or set(receipt_locators) != set(RECEIPT_ROLES):
        raise TrajectoryGridError("shard receipt locator roster differs")
    root = Path(os.path.abspath(os.fspath(evidence_root)))
    name = _shard_name(batch, shard)
    with _held_directory(root, "evidence root") as root_fd:
        validated_locators = {
            role: validate_receipt_locator_v1(root_fd, receipt_locators[role], role)
            for role in RECEIPT_ROLES
        }
        with _create_and_hold_directory_at(root_fd, name, "shard directory") as shard_fd:
            writers = {fragment: _FragmentWriter(shard_fd, fragment) for fragment in FRAGMENT_NAMES}
            try:
                expected = geometry.expected_steps(shard)
                observed = b22_count = short_count = 0
                previous_lifecycle: tuple[int, ...] | None = None
                first_lifecycle_payload: bytes | None = None
                last_lifecycle_payload: bytes | None = None
                for expected_step, item in zip(expected, checkpoints, strict=True):
                    if type(item) is not tuple or len(item) != 3:
                        raise TrajectoryGridError("shard checkpoint shape differs")
                    step, state_payload, lifecycle_payload = item
                    if type(step) is not int or step != expected_step:
                        raise TrajectoryGridError("shard checkpoint step/order differs")
                    decode_state_record_v1(state_payload, geometry)
                    lifecycle = decode_lifecycle_record_v1(lifecycle_payload, geometry)
                    _require_terminal_transition(previous_lifecycle, lifecycle)
                    previous_lifecycle = lifecycle
                    if first_lifecycle_payload is None:
                        first_lifecycle_payload = lifecycle_payload
                    last_lifecycle_payload = lifecycle_payload
                    if geometry.is_b22(step):
                        writers[FRAGMENT_NAMES[0]].write(state_payload)
                        writers[FRAGMENT_NAMES[1]].write(node_digest_v1(batch, step, state_payload, lifecycle_payload, geometry))
                        writers[FRAGMENT_NAMES[2]].write(lifecycle_payload)
                        b22_count += 1
                    if geometry.is_short(step):
                        writers[FRAGMENT_NAMES[3]].write(state_payload)
                        writers[FRAGMENT_NAMES[4]].write(lifecycle_payload)
                        short_count += 1
                    observed += 1
                if observed != len(expected):
                    raise TrajectoryGridError("shard checkpoint stream ended early")
                if first_lifecycle_payload is None or last_lifecycle_payload is None:
                    raise TrajectoryGridError("shard lifecycle endpoints are absent")
                fragments = {item["name"]: item for item in (writers[name].finish() for name in FRAGMENT_NAMES)}
                manifest: dict[str, Any] = {
                    "schema": SHARD_SCHEMA,
                    "batch_order": batch,
                    "object_id": OBJECT_IDS[batch] if geometry == PRODUCTION_GEOMETRY else f"SYNTHETIC_BATCH_{batch}",
                    "shard_index": shard,
                    "shard_name": name,
                    "geometry": _geometry_record(geometry),
                    "first_global_step": expected[0],
                    "last_global_step": expected[-1],
                    "union_checkpoint_count": observed,
                    "b22_checkpoint_count": b22_count,
                    "short_checkpoint_count": short_count,
                    "first_lifecycle_hex": first_lifecycle_payload.hex(),
                    "last_lifecycle_hex": last_lifecycle_payload.hex(),
                    "receipt_locators": validated_locators,
                    "fragments": fragments,
                    "claims": {"trajectory_bytes_retained": True, "dynamics_qualified": False, "classification_generated": False, "science_result": False},
                    "content_sha256": "",
                }
                manifest["content_sha256"] = _domain_hash(SHARD_DOMAIN, manifest)
                _write_create_only(shard_fd, "MANIFEST.json", _canonical_json(manifest))
                os.fsync(shard_fd)
            except Exception:
                for writer in writers.values():
                    writer.abort()
                raise
        validate_shard_v1(root, batch_order=batch, shard_index=shard, geometry=geometry,
                          test_only_geometry=test_only_geometry, _root_fd=root_fd)
    return manifest


def _validate_fragment(held: _HeldFile, receipt: object, label: str) -> None:
    if type(receipt) is not dict or set(receipt) != {"name", "byte_length", "sha256"}:
        raise TrajectoryGridError(f"{label} fragment receipt differs")
    if receipt.get("name") != held.name or type(receipt.get("byte_length")) is not int or receipt.get("byte_length") != held.size:
        raise TrajectoryGridError(f"{label} fragment size/name differs")
    if not _is_sha256(receipt.get("sha256")) or held.sha256() != receipt["sha256"]:
        raise TrajectoryGridError(f"{label} fragment digest differs")


def validate_shard_v1(evidence_root: Path, *, batch_order: int, shard_index: int,
                      geometry: GridGeometry = PRODUCTION_GEOMETRY,
                      test_only_geometry: bool = False,
                      _root_fd: int | None = None) -> dict[str, Any]:
    _require_geometry(geometry, test_only_geometry=test_only_geometry)
    batch = _require_int(batch_order, "batch_order")
    shard = _require_int(shard_index, "shard_index")
    name = _shard_name(batch, shard)
    root = Path(os.path.abspath(os.fspath(evidence_root)))
    expected_roster = frozenset((*FRAGMENT_NAMES, "MANIFEST.json"))
    root_context = _held_directory(root, "evidence root") if _root_fd is None else nullcontext(_root_fd)
    with root_context as root_fd:
        with _held_directory_at(root_fd, name, "validated shard directory") as shard_fd:
            if frozenset(os.listdir(shard_fd)) != expected_roster:
                raise TrajectoryGridError("shard roster differs")
            manifest_file = _HeldFile(shard_fd, "MANIFEST.json", "shard manifest")
            try:
                manifest_payload = manifest_file.read_all(maximum=8_000_000)
            finally:
                manifest_file.close()
            manifest = _strict_json(manifest_payload)
            expected_keys = {
                "schema", "batch_order", "object_id", "shard_index", "shard_name", "geometry",
                "first_global_step", "last_global_step", "union_checkpoint_count", "b22_checkpoint_count",
                "short_checkpoint_count", "first_lifecycle_hex", "last_lifecycle_hex",
                "receipt_locators", "fragments", "claims", "content_sha256",
            }
            expected_steps = geometry.expected_steps(shard)
            if (
                set(manifest) != expected_keys or manifest.get("schema") != SHARD_SCHEMA
                or type(manifest.get("batch_order")) is not int or manifest.get("batch_order") != batch
                or type(manifest.get("shard_index")) is not int or manifest.get("shard_index") != shard
                or manifest.get("shard_name") != name or not _exact_equal(manifest.get("geometry"), _geometry_record(geometry))
                or manifest.get("object_id") != (OBJECT_IDS[batch] if geometry == PRODUCTION_GEOMETRY else f"SYNTHETIC_BATCH_{batch}")
                or type(manifest.get("first_global_step")) is not int or manifest.get("first_global_step") != expected_steps[0]
                or type(manifest.get("last_global_step")) is not int or manifest.get("last_global_step") != expected_steps[-1]
                or manifest.get("content_sha256") != _domain_hash(SHARD_DOMAIN, manifest)
                or manifest_payload != _canonical_json(manifest)
            ):
                raise TrajectoryGridError("shard manifest identity differs")
            locators = manifest.get("receipt_locators")
            if type(locators) is not dict or set(locators) != set(RECEIPT_ROLES):
                raise TrajectoryGridError("shard receipt locator roster differs")
            for role in RECEIPT_ROLES:
                validate_receipt_locator_v1(root_fd, locators[role], role)
            fragments = manifest.get("fragments")
            if type(fragments) is not dict or set(fragments) != set(FRAGMENT_NAMES):
                raise TrajectoryGridError("shard fragment roster differs")
            held_fragments: dict[str, _HeldFile] = {}
            try:
                for fragment in FRAGMENT_NAMES:
                    held_fragments[fragment] = _HeldFile(shard_fd, fragment, fragment)
                    _validate_fragment(held_fragments[fragment], fragments[fragment], fragment)
                b22_steps = [step for step in expected_steps if geometry.is_b22(step)]
                short_steps = [step for step in expected_steps if geometry.is_short(step)]
                if (
                    type(manifest.get("union_checkpoint_count")) is not int or manifest["union_checkpoint_count"] != len(expected_steps)
                    or type(manifest.get("b22_checkpoint_count")) is not int or manifest["b22_checkpoint_count"] != len(b22_steps)
                    or type(manifest.get("short_checkpoint_count")) is not int or manifest["short_checkpoint_count"] != len(short_steps)
                ):
                    raise TrajectoryGridError("shard checkpoint counts differ")
                b22_states = held_fragments[FRAGMENT_NAMES[0]]
                b22_digests = held_fragments[FRAGMENT_NAMES[1]]
                b22_lifecycle = held_fragments[FRAGMENT_NAMES[2]]
                short_states = held_fragments[FRAGMENT_NAMES[3]]
                short_lifecycle = held_fragments[FRAGMENT_NAMES[4]]
                if b22_states.size != len(b22_steps) * geometry.state_record_bytes or b22_digests.size != len(b22_steps) * 32 or b22_lifecycle.size != len(b22_steps) * geometry.lifecycle_record_bytes:
                    raise TrajectoryGridError("B22 fragment byte arithmetic differs")
                if short_states.size != len(short_steps) * geometry.state_record_bytes or short_lifecycle.size != len(short_steps) * geometry.lifecycle_record_bytes:
                    raise TrajectoryGridError("short fragment byte arithmetic differs")
                short_lookup = {step: index for index, step in enumerate(short_steps)}
                b22_lookup = {step: index for index, step in enumerate(b22_steps)}
                for index, step in enumerate(b22_steps):
                    state_payload = b22_states.pread(geometry.state_record_bytes, index * geometry.state_record_bytes)
                    lifecycle_payload = b22_lifecycle.pread(geometry.lifecycle_record_bytes, index * geometry.lifecycle_record_bytes)
                    if b22_digests.pread(32, index * 32) != node_digest_v1(batch, step, state_payload, lifecycle_payload, geometry):
                        raise TrajectoryGridError("B22 node digest replay differs")
                    if step in short_lookup:
                        short_index = short_lookup[step]
                        if state_payload != short_states.pread(geometry.state_record_bytes, short_index * geometry.state_record_bytes) or lifecycle_payload != short_lifecycle.pread(geometry.lifecycle_record_bytes, short_index * geometry.lifecycle_record_bytes):
                            raise TrajectoryGridError("B22/short overlap differs")
                previous_lifecycle: tuple[int, ...] | None = None
                first_payload: bytes | None = None
                last_payload: bytes | None = None
                for step in expected_steps:
                    if step in short_lookup:
                        index = short_lookup[step]
                        state_payload = short_states.pread(geometry.state_record_bytes, index * geometry.state_record_bytes)
                        lifecycle_payload = short_lifecycle.pread(geometry.lifecycle_record_bytes, index * geometry.lifecycle_record_bytes)
                    else:
                        index = b22_lookup[step]
                        state_payload = b22_states.pread(geometry.state_record_bytes, index * geometry.state_record_bytes)
                        lifecycle_payload = b22_lifecycle.pread(geometry.lifecycle_record_bytes, index * geometry.lifecycle_record_bytes)
                    decode_state_record_v1(state_payload, geometry)
                    lifecycle = decode_lifecycle_record_v1(lifecycle_payload, geometry)
                    _require_terminal_transition(previous_lifecycle, lifecycle)
                    previous_lifecycle = lifecycle
                    if first_payload is None:
                        first_payload = lifecycle_payload
                    last_payload = lifecycle_payload
                if (
                    first_payload is None or last_payload is None
                    or type(manifest.get("first_lifecycle_hex")) is not str
                    or type(manifest.get("last_lifecycle_hex")) is not str
                    or manifest["first_lifecycle_hex"] != first_payload.hex()
                    or manifest["last_lifecycle_hex"] != last_payload.hex()
                ):
                    raise TrajectoryGridError("shard lifecycle endpoint evidence differs")
            finally:
                for held in held_fragments.values():
                    held.close()
            if not _exact_equal(
                manifest.get("claims"),
                {"trajectory_bytes_retained": True, "dynamics_qualified": False, "classification_generated": False, "science_result": False},
            ):
                raise TrajectoryGridError("shard claim ceiling differs")
            return manifest


def _copy_held(source: _HeldFile, destination: _FragmentWriter) -> None:
    offset = 0
    while offset < source.size:
        payload = source.pread(min(1 << 20, source.size - offset), offset)
        destination.write(payload)
        offset += len(payload)


def finalize_shards_v1(evidence_root: Path, output_root: Path, *,
                       geometry: GridGeometry = PRODUCTION_GEOMETRY,
                       test_only_geometry: bool = False) -> dict[str, Any]:
    """Stream validated fragments into create-only final files; RESULT is last."""
    _require_geometry(geometry, test_only_geometry=test_only_geometry)
    evidence = Path(os.path.abspath(os.fspath(evidence_root)))
    output = Path(os.path.abspath(os.fspath(output_root)))
    validate_prepared(output)
    with _held_directory(evidence, "evidence root") as evidence_fd, _held_directory(output, "final output") as output_fd:
        if frozenset(os.listdir(output_fd)) != PREPARED_ROSTER:
            raise TrajectoryGridError("final output does not begin with the prepared one-file roster")
        held_protocol = _HeldFile(output_fd, "PROTOCOL.json", "finalizer prepared protocol")
        try:
            protocol_payload = held_protocol.read_all(maximum=1_000_000)
        finally:
            held_protocol.close()
        protocol = _strict_json(protocol_payload)
        if protocol_payload != _canonical_json(protocol):
            raise TrajectoryGridError("finalizer prepared protocol is not canonical")
        validate_protocol(protocol)
        writers = {name: _FragmentWriter(output_fd, name) for name in FINAL_BINARY_NAMES}
        manifest_lines = bytearray()
        manifests: list[dict[str, Any]] = []
        try:
            for batch in range(geometry.batch_count):
                previous_terminal: tuple[int, ...] | None = None
                for shard in range(geometry.shard_count):
                    manifest = validate_shard_v1(evidence, batch_order=batch, shard_index=shard,
                                                 geometry=geometry, test_only_geometry=test_only_geometry,
                                                 _root_fd=evidence_fd)
                    manifests.append(manifest)
                    manifest_lines.extend(_canonical_json(manifest))
                    first_lifecycle = decode_lifecycle_record_v1(bytes.fromhex(manifest["first_lifecycle_hex"]), geometry)
                    last_lifecycle = decode_lifecycle_record_v1(bytes.fromhex(manifest["last_lifecycle_hex"]), geometry)
                    _require_terminal_transition(previous_terminal, first_lifecycle)
                    _require_terminal_transition(first_lifecycle, last_lifecycle)
                    previous_terminal = last_lifecycle
                    shard_name = _shard_name(batch, shard)
                    with _held_directory_at(evidence_fd, shard_name, "finalizer shard directory") as shard_fd:
                        held_manifest = _HeldFile(shard_fd, "MANIFEST.json", "held finalizer shard manifest")
                        try:
                            held_manifest_payload = held_manifest.read_all(maximum=8_000_000)
                        finally:
                            held_manifest.close()
                        if held_manifest_payload != _canonical_json(manifest):
                            raise TrajectoryGridError("held finalizer shard differs from validated shard")
                        for role in RECEIPT_ROLES:
                            validate_receipt_locator_v1(evidence_fd, manifest["receipt_locators"][role], role)
                        mapping = {
                            "B22_STATES.f64le": "B22_STATES.fragment",
                            "B22_NODE_SHA256.raw": "B22_NODE_SHA256.fragment",
                            "B22_LIFECYCLE.u8": "B22_LIFECYCLE.fragment",
                            "ML_SHORT_STATES.f64le": "ML_SHORT_STATES.fragment",
                            "ML_SHORT_LIFECYCLE.u8": "ML_SHORT_LIFECYCLE.fragment",
                        }
                        for final_name, fragment_name in mapping.items():
                            held = _HeldFile(shard_fd, fragment_name, fragment_name)
                            try:
                                _validate_fragment(held, manifest["fragments"][fragment_name], fragment_name)
                                _copy_held(held, writers[final_name])
                            finally:
                                held.close()
            file_records = {record["name"]: record for record in (writers[name].finish() for name in FINAL_BINARY_NAMES)}
        except Exception:
            for writer in writers.values():
                writer.abort()
            raise
        expected_sizes = geometry.output_sizes()
        if any(file_records[name]["byte_length"] != expected_sizes[name] for name in FINAL_BINARY_NAMES):
            raise TrajectoryGridError("final binary size arithmetic differs")
        shard_manifest_record = _write_create_only(output_fd, "SHARD_MANIFESTS.jsonl", bytes(manifest_lines))
        final_manifest: dict[str, Any] = {
            "schema": FINAL_MANIFEST_SCHEMA,
            "geometry": _geometry_record(geometry),
            "files": file_records,
            "shard_manifests": {
                **shard_manifest_record,
                "record_count": len(manifests),
                "order": "BATCH_MAJOR_THEN_SHARD_MAJOR_CANONICAL_JSON_LINES",
                "content_sha256_roster": [manifest["content_sha256"] for manifest in manifests],
            },
            "claims": {"trajectory_bytes_finalized": True, "dynamics_qualified": False, "classification_generated": False, "science_result": False},
            "content_sha256": "",
        }
        final_manifest["content_sha256"] = _domain_hash(FINAL_MANIFEST_DOMAIN, final_manifest)
        manifest_payload = _canonical_json(final_manifest)
        manifest_file = _write_create_only(output_fd, "MANIFEST.json", manifest_payload)
        result: dict[str, Any] = {
            "schema": FINAL_RESULT_SCHEMA,
            "status": "TRAJECTORY_BYTES_FINALIZED_NO_DYNAMICS_OR_SCIENCE_CLAIM",
            "protocol_content_sha256": protocol["content_sha256"],
            "manifest_content_sha256": final_manifest["content_sha256"],
            "manifest_file_sha256": manifest_file["sha256"],
            "claims": {
                "trajectory_acquisition_mechanics_exercised": True,
                "dynamics_executed": False,
                "ten_myr_trajectory_acquired": False,
                "ten_myr_execution_authorized": False,
                "classification_generated": False,
                "resonance_classification_generated": False,
                "science_result": False,
            },
            "content_sha256": "",
        }
        result["content_sha256"] = _domain_hash(FINAL_RESULT_DOMAIN, result)
        _write_create_only(output_fd, "RESULT.json", _canonical_json(result))
        os.fsync(output_fd)
    validate_finalized_v1(evidence, output, geometry=geometry, test_only_geometry=test_only_geometry)
    return result


def validate_finalized_v1(evidence_root: Path, output_root: Path, *,
                          geometry: GridGeometry = PRODUCTION_GEOMETRY,
                          test_only_geometry: bool = False) -> dict[str, Any]:
    _require_geometry(geometry, test_only_geometry=test_only_geometry)
    evidence = Path(os.path.abspath(os.fspath(evidence_root)))
    output = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(evidence, "final evidence root") as evidence_fd, _held_directory(output, "final output", expected_roster=FINAL_ROSTER) as output_fd:
        held: dict[str, _HeldFile] = {}
        try:
            for name in FINAL_ROSTER:
                held[name] = _HeldFile(output_fd, name, f"final {name}")
            protocol_payload = held["PROTOCOL.json"].read_all(maximum=1_000_000)
            protocol = _strict_json(protocol_payload)
            if protocol_payload != _canonical_json(protocol):
                raise TrajectoryGridError("final protocol is not canonical")
            validate_protocol(protocol)
            manifest_payload = held["MANIFEST.json"].read_all(maximum=32_000_000)
            result_payload = held["RESULT.json"].read_all(maximum=2_000_000)
            manifest = _strict_json(manifest_payload)
            result = _strict_json(result_payload)
            if manifest_payload != _canonical_json(manifest) or result_payload != _canonical_json(result):
                raise TrajectoryGridError("final JSON is not canonical")
            if set(manifest) != {"schema", "geometry", "files", "shard_manifests", "claims", "content_sha256"} or manifest.get("schema") != FINAL_MANIFEST_SCHEMA or not _exact_equal(manifest.get("geometry"), _geometry_record(geometry)) or manifest.get("content_sha256") != _domain_hash(FINAL_MANIFEST_DOMAIN, manifest):
                raise TrajectoryGridError("final manifest identity differs")
            if not _exact_equal(
                manifest.get("claims"),
                {"trajectory_bytes_finalized": True, "dynamics_qualified": False, "classification_generated": False, "science_result": False},
            ):
                raise TrajectoryGridError("final manifest claim ceiling differs")
            expected_result = {
                "schema": FINAL_RESULT_SCHEMA,
                "status": "TRAJECTORY_BYTES_FINALIZED_NO_DYNAMICS_OR_SCIENCE_CLAIM",
                "protocol_content_sha256": protocol["content_sha256"],
                "manifest_content_sha256": manifest["content_sha256"],
                "manifest_file_sha256": _sha256(manifest_payload),
                "claims": {
                    "trajectory_acquisition_mechanics_exercised": True,
                    "dynamics_executed": False,
                    "ten_myr_trajectory_acquired": False,
                    "ten_myr_execution_authorized": False,
                    "classification_generated": False,
                    "resonance_classification_generated": False,
                    "science_result": False,
                },
                "content_sha256": result.get("content_sha256"),
            }
            if not _exact_equal(result, expected_result) or result.get("content_sha256") != _domain_hash(FINAL_RESULT_DOMAIN, result):
                raise TrajectoryGridError("final result identity or claim ceiling differs")
            files = manifest.get("files")
            if type(files) is not dict or set(files) != set(FINAL_BINARY_NAMES):
                raise TrajectoryGridError("final binary receipt roster differs")
            for name in FINAL_BINARY_NAMES:
                _validate_fragment(held[name], files[name], name)
            shard_stream = held["SHARD_MANIFESTS.jsonl"].read_all(maximum=128_000_000)
            shard_record = manifest.get("shard_manifests")
            if (
                type(shard_record) is not dict
                or set(shard_record) != {"name", "byte_length", "sha256", "record_count", "order", "content_sha256_roster"}
                or shard_record.get("name") != "SHARD_MANIFESTS.jsonl"
                or type(shard_record.get("byte_length")) is not int
                or shard_record.get("sha256") != _sha256(shard_stream)
                or shard_record.get("byte_length") != len(shard_stream)
                or shard_record.get("order") != "BATCH_MAJOR_THEN_SHARD_MAJOR_CANONICAL_JSON_LINES"
            ):
                raise TrajectoryGridError("retained shard-manifest stream differs")
            lines = shard_stream.splitlines(keepends=True)
            if len(lines) != geometry.batch_count * geometry.shard_count or shard_record.get("record_count") != len(lines):
                raise TrajectoryGridError("retained shard-manifest count differs")
            contents: list[str] = []
            final_offsets = {name: 0 for name in FINAL_BINARY_NAMES}
            fragment_for_final = {
                "B22_STATES.f64le": "B22_STATES.fragment",
                "B22_NODE_SHA256.raw": "B22_NODE_SHA256.fragment",
                "B22_LIFECYCLE.u8": "B22_LIFECYCLE.fragment",
                "ML_SHORT_STATES.f64le": "ML_SHORT_STATES.fragment",
                "ML_SHORT_LIFECYCLE.u8": "ML_SHORT_LIFECYCLE.fragment",
            }
            previous_by_batch: dict[int, tuple[int, ...]] = {}
            for order, line in enumerate(lines):
                row = _strict_json(line)
                if line != _canonical_json(row):
                    raise TrajectoryGridError("retained shard-manifest line is not canonical")
                batch, shard = divmod(order, geometry.shard_count)
                observed = validate_shard_v1(evidence, batch_order=batch, shard_index=shard,
                                             geometry=geometry, test_only_geometry=test_only_geometry,
                                             _root_fd=evidence_fd)
                if not _exact_equal(row, observed):
                    raise TrajectoryGridError("retained shard manifest differs from held evidence")
                contents.append(row["content_sha256"])
                first_lifecycle = decode_lifecycle_record_v1(bytes.fromhex(row["first_lifecycle_hex"]), geometry)
                last_lifecycle = decode_lifecycle_record_v1(bytes.fromhex(row["last_lifecycle_hex"]), geometry)
                _require_terminal_transition(previous_by_batch.get(batch), first_lifecycle)
                _require_terminal_transition(first_lifecycle, last_lifecycle)
                previous_by_batch[batch] = last_lifecycle
                shard_name = _shard_name(batch, shard)
                shard_fd = os.open(shard_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=evidence_fd)
                try:
                    held_manifest = _HeldFile(shard_fd, "MANIFEST.json", "held replay manifest")
                    try:
                        if held_manifest.read_all(maximum=8_000_000) != line:
                            raise TrajectoryGridError("replay shard manifest path differs")
                    finally:
                        held_manifest.close()
                    for role in RECEIPT_ROLES:
                        validate_receipt_locator_v1(evidence_fd, row["receipt_locators"][role], role)
                    for final_name, fragment_name in fragment_for_final.items():
                        source = _HeldFile(shard_fd, fragment_name, "replay fragment")
                        try:
                            offset = 0
                            while offset < source.size:
                                count = min(1 << 20, source.size - offset)
                                if source.pread(count, offset) != held[final_name].pread(count, final_offsets[final_name] + offset):
                                    raise TrajectoryGridError("final binary differs from shard evidence")
                                offset += count
                            final_offsets[final_name] += source.size
                        finally:
                            source.close()
                finally:
                    os.close(shard_fd)
            if contents != shard_record.get("content_sha256_roster"):
                raise TrajectoryGridError("shard manifest content roster differs")
            if any(final_offsets[name] != held[name].size for name in FINAL_BINARY_NAMES):
                raise TrajectoryGridError("final binary replay length differs")
            return result
        finally:
            for item in held.values():
                item.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate-prepared"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    value = prepare(args.output_root) if args.command == "prepare" else validate_prepared(args.output_root)
    print(json.dumps({"status": value["status"], "content_sha256": value["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
