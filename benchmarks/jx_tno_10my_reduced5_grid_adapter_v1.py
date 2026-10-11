#!/usr/bin/env python3
"""Prepare and qualify the 10 Myr reduced-five grid/adapter boundary.

This module deliberately does **not** expose a dynamics execution command.  It
implements the deterministic binary codec, immutable shard writer, shard-chain
validator, field-major finalizer, and offline replay validator needed by the
prepared 10 Myr TNO pilot.  It also freezes the exact Wisdom--Holman lifecycle
composition which a later, separately preregistered execution successor must
use.

The production geometry is large (12,000 immutable 10 kyr shards and
791,062,272 fixed output bytes), so the mechanics are parameterized for bounded
synthetic qualification.  Public production entry points reject any geometry
or classifier kernel other than the frozen values below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_grid_adapter_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_reduced5_grid_adapter_v1.py"
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "runs" / "tno_10my_reduced5_grid_adapter_v1"

PROTOCOL_SCHEMA = "jx.tno.10my-reduced5-grid-adapter.protocol.v1"
SHARD_SCHEMA = "jx.tno.10my-reduced5-grid-adapter.shard.v1"
FINAL_MANIFEST_SCHEMA = "jx.tno.10my-reduced5-grid-adapter.final-manifest.v1"
FINAL_RESULT_SCHEMA = "jx.tno.10my-reduced5-grid-adapter.final-result.v1"
PROTOCOL_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.protocol.v1\0"
SHARD_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.shard.v1\0"
FINAL_MANIFEST_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.final-manifest.v1\0"
FINAL_RESULT_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.final-result.v1\0"
PREFIX_CROSSBIND_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.first-1k-crossbind.v1\0"
BATCH_STREAM_ROOT_DOMAIN = b"jx.tno.10my-reduced5-grid-adapter.batch-stream-root.v1\0"

OBJECT_IDS = (
    "jpl-20090377",
    "jpl-20474640",
    "jpl-54526835",
    "jpl-20541132",
    "jpl-50666516",
    "jpl-20148209",
    "jpl-20574372",
    "jpl-20303775",
    "jpl-20026181",
    "jpl-20042301",
    "jpl-20026375",
    "jpl-20015874",
)
SPK_IDS = (
    20090377,
    20474640,
    54526835,
    20541132,
    50666516,
    20148209,
    20574372,
    20303775,
    20026181,
    20042301,
    20026375,
    20015874,
)
NATIVE_EPOCHS_JD_TDB_HEX = (
    "0x1.2c195c0000000p+21",
    "0x1.2bd3dc0000000p+21",
    "0x1.2c4c140000000p+21",
    "0x1.2c0e240000000p+21",
    "0x1.2be0a40000000p+21",
    "0x1.2b62c40000000p+21",
    "0x1.2c15dc0000000p+21",
    "0x1.2c247c0000000p+21",
    "0x1.2c11640000000p+21",
    "0x1.2c16a40000000p+21",
    "0x1.2c0b4c0000000p+21",
    "0x1.2c0f940000000p+21",
)
NATIVE_EPOCH_ORIGINS_JULIAN_YEARS_HEX = tuple(
    ((float.fromhex(value) - 2_451_545.0) / 365.25).hex()
    for value in NATIVE_EPOCHS_JD_TDB_HEX
)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
REDUCED_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
REDUCED_NAIF_IDS = (10, 5, 6, 7, 8)
REDUCED_GM_HEX = (
    "0x1.3bd13a81d2261p+5",
    "0x1.34c6609ebf23fp-5",
    "0x1.71d0e2fd3b84dp-7",
    "0x1.c3d8f106aa648p-10",
    "0x1.0a8c8f8c38c6ap-9",
)

ELEMENT_FIELDS = (
    "semimajor_axis_AU",
    "eccentricity",
    "inclination_radians",
    "longitude_of_ascending_node_radians",
    "argument_of_perihelion_radians",
    "mean_anomaly_radians",
)
FRAGMENT_NAMES = (
    "B22_STATES.fragment",
    "B22_ELEMENTS_NODE_MAJOR.fragment",
    "B22_NODE_SHA256.fragment",
    "ML_SHORT_STATES.fragment",
)
FINAL_BINARY_NAMES = (
    "B22_STATES.f64le",
    "B22_ELEMENTS.f64le",
    "B22_NODE_SHA256.raw",
    "ML_SHORT_STATES.f64le",
)
FINAL_ROSTER = (*FINAL_BINARY_NAMES, "MANIFEST.json", "RESULT.json")
ZERO_SHA256 = "0" * 64
SHARD_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "batch_identity",
        "geometry",
        "shard_index",
        "ownership",
        "first_global_step",
        "last_global_step",
        "union_checkpoint_count",
        "b22_checkpoint_count",
        "short_checkpoint_count",
        "predecessor_manifest_content_sha256",
        "start_continuation_sha256",
        "final_continuation_sha256",
        "source_chunk_receipt_sha256",
        "route_prefix_sha256",
        "runtime_projection_sha256",
        "first_1000_year_prefix_crossbinding_sha256",
        "fragments",
        "claims",
        "content_sha256",
    }
)
FINAL_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "geometry",
        "files",
        "shard_manifests_by_batch",
        "batch_stream_roots",
        "layout",
        "content_sha256",
    }
)
FINAL_RESULT_KEYS = frozenset(
    {"schema", "status", "manifest_content_sha256", "claims", "content_sha256"}
)

BOUND_FILES = (
    ("pilot_v2_source", "benchmarks/jx_tno_10my_reduced5_classification_pilot_v2.py", 59_307, "33ad38cbc89a15a83cf180161544a721b8635affd0d4890a8a39d04389694332"),
    ("pilot_v2_test", "tests/test_tno_10my_reduced5_classification_pilot_v2.py", 28_773, "060793a45cd91590d484a610b9c6d48f6fae6c2a468c6e2663541599a831fbd8"),
    ("pilot_v2_protocol", "runs/tno_10my_reduced5_classification_pilot_v2/PROTOCOL.json", 34_347, "b510cbf7a62f45d8713ca8ef02dd2229f5a79966d00d2482843e814b939cded6"),
    ("classifier_v2_source", "benchmarks/jx_tno_b22_classifier_v2.py", 94_534, "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98"),
    ("stage1_v4_source", "benchmarks/jx_tno_stage1_dynamics_qualification_v4.py", 436_125, "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"),
    ("wh_source", "src/jxplanetx/restricted_wh_source.py", 61_803, "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52"),
    ("quiet_tracer", "src/jxplanetx/cuda_restricted_wh.py", 74_758, "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927"),
    ("hybrid", "src/jxplanetx/cuda_restricted_wh_hybrid.py", 37_420, "6b3ea46436dbdb7dc2c434cff005a88395fea6e00cd97c7be104650c75769b10"),
    ("route", "src/jxplanetx/restricted_wh_route.py", 38_194, "7eae131cda32ee3feed525b13226b1b271b0fc54c1dd5d6a82927b4dc4b94075"),
    ("lifecycle", "src/jxplanetx/restricted_wh_lifecycle.py", 38_996, "b0211778fcf51dbfef78049e74bafd05d5daee4db7dd18f743f4a5b5a14af346"),
    ("fast_wh", "src/jxplanetx/fast_wisdom_holman.py", 36_182, "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72"),
    ("native_wh", "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so", 42_520, "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3"),
    ("engine_contracts", "src/jxplanetx/engine/contracts.py", 16_390, "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d"),
    ("jxplanetx_package", "src/jxplanetx/__init__.py", 6_057, "86bd065b9c16a448b217d960c6dd8d79a7d7ccc79209e7708ff79bc07ce681e0"),
    ("engine_package", "src/jxplanetx/engine/__init__.py", 16_617, "0eb03741a4b37ef6cc8a4a8c982639d4fbbbe1b1caedb376b8c419be3e51c500"),
    ("engine_api", "src/jxplanetx/engine/api.py", 16_757, "1664c4c69e4b9fa30b983c8390f76aa7a05259e89dd009d57f7099f8ffe088f2"),
    ("cuda_tracer_events", "src/jxplanetx/cuda_tracer_events.py", 67_504, "a3c5f1f9de29f504420aa892bd50f8a947b24d6df21cadf91dc243c7976dad89"),
    ("h64_closure_result", "runs/tno_stage1_jx_h64_closure_v2/RESULT.json", 5_464_509, "3dd021735f817d4dbfb1a5c92370058347eff3606b47453137b5a3b44d41d54d"),
    ("h64_closure_arrays", "runs/tno_stage1_jx_h64_closure_v2/ARRAYS.bin", 16_144_128, "9da897bb9a386db8429cd607147f241a3a5a1113f716a2f05d2c2d07c876ad41"),
)
PILOT_V2_PROTOCOL_CONTENT_SHA256 = "eb6fe0b5ac6485f4175ebe44ab779d88ebf92986ef279d949e8eaea2b2a5c7d3"
H64_CLOSURE_RESULT_PATH = PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v2" / "RESULT.json"
H64_CLOSURE_ARRAYS_PATH = PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v2" / "ARRAYS.bin"
H64_CLOSURE_ARRAYS_BYTES = 16_144_128
H64_CLOSURE_ARRAYS_SHA256 = "9da897bb9a386db8429cd607147f241a3a5a1113f716a2f05d2c2d07c876ad41"
H64_CLOSURE_LANE_BYTES = 672_672
H64_CLOSURE_RECORD_BYTES = 672
H64_PREFIX_YEARS = tuple(range(0, 1_001, 50))
H64_STEP_HEX = "0x1.0000000000000p-6"
REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX = "0x1.3c3d3b0c49e7ep+6"


class GridAdapterError(RuntimeError):
    """The frozen grid/adapter contract failed closed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise GridAdapterError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise GridAdapterError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridAdapterError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise GridAdapterError("JSON root must be an object")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GridAdapterError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha(value: object, label: str, *, allow_zero: bool = True) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_zero and value == ZERO_SHA256)
    ):
        raise GridAdapterError(f"{label} is not an exact SHA-256 digest")
    return value


_INTEGER_IDENTITY_KEYS = frozenset(
    {
        "batch_order",
        "spk_id",
        "clone_index",
        "shard_index",
        "first_global_step",
        "last_global_step",
        "union_checkpoint_count",
        "b22_checkpoint_count",
        "short_checkpoint_count",
        "byte_length",
        "batch_count",
        "total_steps",
        "shard_steps",
        "shard_count",
        "b22_interval_steps",
        "b22_node_count",
        "short_interval_steps",
        "short_final_step",
        "short_node_count",
        "union_node_count",
        "massive_count",
        "tracer_count",
        "state_record_bytes",
        "element_entity_count",
        "element_field_count",
        "element_record_bytes",
    }
)


def _guard_integer_identities(value: object, key: str | None = None) -> None:
    if key in _INTEGER_IDENTITY_KEYS and value is not None and type(value) is not int:
        raise GridAdapterError(f"integer identity {key} has the wrong exact type")
    if type(value) is dict:
        for child_key, child in value.items():
            _guard_integer_identities(child, child_key)
    elif type(value) is list:
        for child in value:
            _guard_integer_identities(child, key)


def _exact_tree_equal(observed: object, expected: object) -> bool:
    """Compare JSON-like values without Python's bool/int aliasing."""

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return set(observed) == set(expected) and all(
            _exact_tree_equal(observed[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(observed) == len(expected) and all(
            _exact_tree_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GridAdapterError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GridAdapterError(f"{label} is linked or nonregular")
        try:
            path_before = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise GridAdapterError(f"cannot stat {label}") from exc
        if (
            not stat.S_ISREG(path_before.st_mode)
            or (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise GridAdapterError(f"{label} path differs from opened inode")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise GridAdapterError(f"cannot restat {label}") from exc
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(path_before) != _stat_identity(path_after)
            or (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise GridAdapterError(f"{label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


_BOOTSTRAP_SOURCE = _read_regular_bytes(SCRIPT_PATH, "implementation source")
_BOOTSTRAP_SOURCE_SHA256 = _sha256(_BOOTSTRAP_SOURCE)
_BOOTSTRAP_SOURCE_BYTES = len(_BOOTSTRAP_SOURCE)
del _BOOTSTRAP_SOURCE


def _implementation_identity() -> dict[str, object]:
    payload = _read_regular_bytes(SCRIPT_PATH, "implementation source reread")
    if len(payload) != _BOOTSTRAP_SOURCE_BYTES or _sha256(payload) != _BOOTSTRAP_SOURCE_SHA256:
        raise GridAdapterError("implementation source changed after module bootstrap")
    return {
        "relative_path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        "byte_length": _BOOTSTRAP_SOURCE_BYTES,
        "sha256": _BOOTSTRAP_SOURCE_SHA256,
        "custody_scope": "MODULE_BOOTSTRAP_POINT_OBSERVATION_WITH_LIVE_REREAD",
    }


def _binding_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for role, relative, expected_size, expected_sha in BOUND_FILES:
        payload = _read_regular_bytes(PROJECT_ROOT / relative, role)
        if len(payload) != expected_size or _sha256(payload) != expected_sha:
            raise GridAdapterError(f"bound file differs: {relative}")
        records.append(
            {
                "role": role,
                "relative_path": relative,
                "byte_length": expected_size,
                "sha256": expected_sha,
            }
        )
    pilot = _strict_json(
        _read_regular_bytes(
            PROJECT_ROOT / "runs/tno_10my_reduced5_classification_pilot_v2/PROTOCOL.json",
            "pilot v2 protocol",
        )
    )
    if pilot.get("content_sha256") != PILOT_V2_PROTOCOL_CONTENT_SHA256:
        raise GridAdapterError("pilot v2 protocol content identity differs")
    return records


@dataclass(frozen=True, slots=True)
class GridGeometry:
    batch_count: int
    total_steps: int
    shard_steps: int
    b22_interval_steps: int
    short_interval_steps: int
    short_final_step: int
    massive_count: int = 5
    tracer_count: int = 9
    element_entity_count: int = 10
    element_field_count: int = 6

    def __post_init__(self) -> None:
        for name in (
            "batch_count",
            "total_steps",
            "shard_steps",
            "b22_interval_steps",
            "short_interval_steps",
            "short_final_step",
            "massive_count",
            "tracer_count",
            "element_entity_count",
            "element_field_count",
        ):
            _require_int(getattr(self, name), name, minimum=1 if name != "short_final_step" else 0)
        if self.total_steps % self.shard_steps != 0:
            raise GridAdapterError("total steps must divide exactly into shards")
        if self.total_steps % self.b22_interval_steps != 0:
            raise GridAdapterError("B22 cadence must divide the horizon")
        if self.short_final_step > self.total_steps:
            raise GridAdapterError("short grid exceeds the horizon")
        if self.short_final_step % self.short_interval_steps != 0:
            raise GridAdapterError("short endpoint is off cadence")
        if self.b22_interval_steps % self.short_interval_steps != 0:
            raise GridAdapterError("B22/short grids are not nested on their overlap")
        if self.shard_steps % self.b22_interval_steps != 0 or self.shard_steps % self.short_interval_steps != 0:
            raise GridAdapterError("shard boundaries must lie on both grids")

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
    def element_record_bytes(self) -> int:
        return self.element_entity_count * self.element_field_count * 8

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
        _require_int(shard_index, "shard_index")
        if shard_index >= self.shard_count:
            raise GridAdapterError("shard index exceeds geometry")
        start = shard_index * self.shard_steps
        end = start + self.shard_steps
        lower = start if shard_index == 0 else start + 1
        b22_first = ((lower + self.b22_interval_steps - 1) // self.b22_interval_steps) * self.b22_interval_steps
        values = set(range(b22_first, end + 1, self.b22_interval_steps))
        short_end = min(end, self.short_final_step)
        if lower <= short_end:
            short_first = ((lower + self.short_interval_steps - 1) // self.short_interval_steps) * self.short_interval_steps
            values.update(range(short_first, short_end + 1, self.short_interval_steps))
        return tuple(sorted(values))

    def b22_index(self, global_step: int) -> int | None:
        if global_step % self.b22_interval_steps:
            return None
        return global_step // self.b22_interval_steps

    def short_index(self, global_step: int) -> int | None:
        if global_step > self.short_final_step or global_step % self.short_interval_steps:
            return None
        return global_step // self.short_interval_steps

    def output_sizes(self) -> dict[str, int]:
        return {
            "B22_STATES.f64le": self.batch_count * self.b22_node_count * self.state_record_bytes,
            "B22_ELEMENTS.f64le": self.batch_count * self.element_entity_count * self.element_field_count * self.b22_node_count * 8,
            "B22_NODE_SHA256.raw": self.batch_count * self.b22_node_count * 32,
            "ML_SHORT_STATES.f64le": self.batch_count * self.short_node_count * self.state_record_bytes,
        }


PRODUCTION_GEOMETRY = GridGeometry(
    batch_count=12,
    total_steps=640_000_000,
    shard_steps=640_000,
    b22_interval_steps=12_800,
    short_interval_steps=3_200,
    short_final_step=32_000_000,
)


def state_record_v1(rows: Sequence[Sequence[float]], geometry: GridGeometry = PRODUCTION_GEOMETRY) -> bytes:
    if type(rows) not in (tuple, list) or len(rows) != geometry.body_count:
        raise GridAdapterError("state record has the wrong body count")
    values: list[float] = []
    for body_index, row in enumerate(rows):
        if type(row) not in (tuple, list) or len(row) != 6:
            raise GridAdapterError(f"state row {body_index} is not a six-vector")
        for component in row:
            if type(component) is not float or not math.isfinite(component):
                raise GridAdapterError("state components must be finite built-in floats")
            values.append(component)
    payload = struct.pack(f"<{len(values)}d", *values)
    if len(payload) != geometry.state_record_bytes:
        raise GridAdapterError("state codec produced the wrong byte length")
    return payload


def decode_state_record_v1(payload: bytes, geometry: GridGeometry = PRODUCTION_GEOMETRY) -> tuple[tuple[float, ...], ...]:
    if type(payload) is not bytes or len(payload) != geometry.state_record_bytes:
        raise GridAdapterError("state record byte length differs")
    values = struct.unpack(f"<{geometry.body_count * 6}d", payload)
    if any(not math.isfinite(value) for value in values):
        raise GridAdapterError("state record contains a non-finite value")
    return tuple(tuple(values[index * 6 : (index + 1) * 6]) for index in range(geometry.body_count))


def state_record_from_objects_v1(massive: Sequence[object], tracers: Sequence[object]) -> bytes:
    if len(massive) != 5 or len(tracers) != 9:
        raise GridAdapterError("production checkpoint roster must be five plus nine")
    rows = []
    for item in (*massive, *tracers):
        position = getattr(item, "position_AU", None)
        velocity = getattr(item, "velocity_AU_per_julian_year", None)
        if type(position) is not tuple or type(velocity) is not tuple:
            raise GridAdapterError("checkpoint objects do not expose exact tuple vectors")
        rows.append((*position, *velocity))
    return state_record_v1(tuple(rows))


def checkpoint_from_primary_relative_v1(
    kernel: "FrozenClassifierKernelV1",
    *,
    batch_order: int,
    source_relative_positions_AU: Sequence[Sequence[object]],
    source_relative_velocities_AU_per_julian_year: Sequence[Sequence[object]],
    tracer_relative_positions_AU: Sequence[Sequence[object]],
    tracer_relative_velocities_AU_per_julian_year: Sequence[Sequence[object]],
) -> bytes:
    """Canonicalize one engine endpoint with the exact bound Stage-1 routine.

    The restricted-WH engine is primary-relative.  Published grid states are
    instantaneous reduced-five GM-barycentric.  This function is the sole
    frozen boundary between those representations; it calls the bound Stage-1
    ``_checkpoint`` implementation rather than duplicating its arithmetic.
    """

    if type(kernel) is not FrozenClassifierKernelV1 or kernel._closed:
        raise GridAdapterError("endpoint conversion requires a live frozen classifier kernel")
    _require_int(batch_order, "batch_order")
    if batch_order >= len(OBJECT_IDS):
        raise GridAdapterError("batch order exceeds the cohort")

    def vectors(value: Sequence[Sequence[object]], count: int, label: str) -> tuple[tuple[float, float, float], ...]:
        if type(value) not in (tuple, list) or len(value) != count:
            raise GridAdapterError(f"{label} roster differs")
        result: list[tuple[float, float, float]] = []
        for row in value:
            if type(row) not in (tuple, list) or len(row) != 3:
                raise GridAdapterError(f"{label} row shape differs")
            converted: list[float] = []
            for component in row:
                if type(component) is bool:
                    raise GridAdapterError(f"{label} contains a Boolean component")
                try:
                    number = float(component)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise GridAdapterError(f"{label} contains a non-binary64 component") from exc
                if not math.isfinite(number):
                    raise GridAdapterError(f"{label} contains a non-finite component")
                converted.append(number)
            result.append(tuple(converted))
        return tuple(result)

    source_positions = vectors(source_relative_positions_AU, 5, "source positions")
    source_velocities = vectors(source_relative_velocities_AU_per_julian_year, 5, "source velocities")
    tracer_positions = vectors(tracer_relative_positions_AU, 9, "tracer positions")
    tracer_velocities = vectors(tracer_relative_velocities_AU_per_julian_year, 9, "tracer velocities")
    if source_positions[0] != (0.0, 0.0, 0.0) or source_velocities[0] != (0.0, 0.0, 0.0):
        raise GridAdapterError("primary-relative source slot zero must be exact zero")
    massive = tuple(
        kernel.stage1.MassiveState(
            REDUCED_NAIF_IDS[index],
            REDUCED_BODY_IDS[index],
            float.fromhex(REDUCED_GM_HEX[index]),
            source_positions[index],
            source_velocities[index],
        )
        for index in range(5)
    )
    tracers = tuple(
        kernel.stage1.TracerState(
            OBJECT_IDS[batch_order],
            SPK_IDS[batch_order],
            "NOMINAL" if index == 0 else "COVARIANCE_CLONE",
            None if index == 0 else HOLDOUT_CLONE_INDICES[index - 1],
            tracer_positions[index],
            tracer_velocities[index],
        )
        for index in range(9)
    )
    try:
        centered = kernel.stage1._checkpoint(0, massive, tracers)
    except Exception as exc:
        raise GridAdapterError("bound Stage-1 endpoint canonicalization failed") from exc
    rows = tuple(
        (*centered["massive_positions_AU"][index], *centered["massive_velocities_AU_per_julian_year"][index])
        for index in range(5)
    ) + tuple(
        (*centered["tracer_positions_AU"][index], *centered["tracer_velocities_AU_per_julian_year"][index])
        for index in range(9)
    )
    return state_record_v1(rows)


def _legacy_h64_record_to_row_interleaved_v1(payload: bytes) -> bytes:
    """Translate the retained Stage-1 block layout into the pilot row layout."""

    if type(payload) is not bytes or len(payload) != H64_CLOSURE_RECORD_BYTES:
        raise GridAdapterError("retained h64 record byte length differs")
    values = struct.unpack("<84d", payload)
    if any(not math.isfinite(value) for value in values):
        raise GridAdapterError("retained h64 record contains a non-finite value")
    rows: list[tuple[float, ...]] = []
    for index in range(5):
        rows.append((*values[index * 3 : index * 3 + 3], *values[15 + index * 3 : 18 + index * 3]))
    for index in range(9):
        rows.append((*values[30 + index * 3 : 33 + index * 3], *values[57 + index * 3 : 60 + index * 3]))
    return state_record_v1(tuple(rows))


def _h64_closure_lane_record(batch_order: int, result: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_int(batch_order, "batch_order")
    if batch_order >= len(OBJECT_IDS) or type(result) is not dict:
        raise GridAdapterError("h64 closure batch/result differs")
    arrays = result.get("cartesian_arrays")
    lanes = result.get("batch_lanes")
    if (
        type(arrays) is not dict
        or arrays.get("relative_path") != "ARRAYS.bin"
        or arrays.get("byte_length") != H64_CLOSURE_ARRAYS_BYTES
        or arrays.get("file_sha256") != H64_CLOSURE_ARRAYS_SHA256
        or arrays.get("ordering")
        != "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL"
        or type(lanes) is not list
        or len(lanes) != 24
    ):
        raise GridAdapterError("h64 closure array contract differs")
    lane_index = 12 + batch_order
    row = lanes[lane_index]
    if type(row) is not dict or set(row) != {
        "batch_order",
        "ephemeris",
        "native_epoch_jd_tdb_hex",
        "object_id",
        "lane",
    }:
        raise GridAdapterError("h64 closure lane envelope differs")
    lane = row.get("lane")
    locator = lane.get("cartesian_array") if type(lane) is dict else None
    if (
        row.get("batch_order") != batch_order
        or row.get("ephemeris") != "DE441"
        or row.get("object_id") != OBJECT_IDS[batch_order]
        or row.get("native_epoch_jd_tdb_hex") != NATIVE_EPOCHS_JD_TDB_HEX[batch_order]
        or lane.get("lane_id") != "jx_reduced5_h_1_64"
        or lane.get("step_julian_years_hex") != H64_STEP_HEX
        or type(locator) is not dict
        or locator.get("byte_offset") != lane_index * H64_CLOSURE_LANE_BYTES
        or locator.get("byte_length") != H64_CLOSURE_LANE_BYTES
        or locator.get("checkpoint_count") != 1_001
        or locator.get("record_bytes") != H64_CLOSURE_RECORD_BYTES
        or locator.get("record_doubles") != 84
        or locator.get("massive_count") != 5
        or locator.get("tracer_count") != 9
        or locator.get("field_order")
        != [
            "massive_positions_AU",
            "massive_velocities_AU_per_julian_year",
            "tracer_positions_AU",
            "tracer_velocities_AU_per_julian_year",
        ]
    ):
        raise GridAdapterError("h64 closure DE441 lane contract differs")
    return locator


def validate_first_1000_year_prefix_v1(
    batch_order: int,
    checkpoints: Sequence[tuple[int, bytes]],
) -> dict[str, object]:
    """Bitwise-crossbind a future DE441 h64 prefix to the retained Stage-1 lane.

    The caller must stop at year 1,000 and supply exactly the 21 checkpoints at
    years 0, 50, ..., 1,000.  The comparison is made after an explicit layout
    translation; there is no numerical tolerance or fallback.
    """

    _require_int(batch_order, "batch_order")
    if batch_order >= len(OBJECT_IDS) or type(checkpoints) not in (tuple, list):
        raise GridAdapterError("first-1k prefix input roster differs")
    result_payload = _read_regular_bytes(H64_CLOSURE_RESULT_PATH, "h64 closure result")
    if _sha256(result_payload) != "3dd021735f817d4dbfb1a5c92370058347eff3606b47453137b5a3b44d41d54d":
        raise GridAdapterError("h64 closure result identity differs")
    result = _strict_json(result_payload)
    locator = _h64_closure_lane_record(batch_order, result)
    if len(checkpoints) != len(H64_PREFIX_YEARS):
        raise GridAdapterError("first-1k prefix checkpoint count differs")

    parent_fd = os.open(
        H64_CLOSURE_ARRAYS_PATH.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    held: _HeldFile | None = None
    checkpoint_digests: list[str] = []
    try:
        held = _HeldFile(parent_fd, H64_CLOSURE_ARRAYS_PATH.name, "h64 closure arrays")
        if held.size != H64_CLOSURE_ARRAYS_BYTES or held.sha256() != H64_CLOSURE_ARRAYS_SHA256:
            raise GridAdapterError("h64 closure arrays identity differs")
        for expected_year, item in zip(H64_PREFIX_YEARS, checkpoints, strict=True):
            if type(item) is not tuple or len(item) != 2:
                raise GridAdapterError("first-1k checkpoint item differs")
            year, future_payload = item
            if type(year) is not int or year != expected_year:
                raise GridAdapterError("first-1k checkpoint year/order differs")
            decode_state_record_v1(future_payload)
            retained_offset = locator["byte_offset"] + year * H64_CLOSURE_RECORD_BYTES
            retained_legacy = held.pread(H64_CLOSURE_RECORD_BYTES, retained_offset)
            retained = _legacy_h64_record_to_row_interleaved_v1(retained_legacy)
            if future_payload != retained:
                raise GridAdapterError("first-1k h64 prefix differs bitwise")
            checkpoint_digests.append(_sha256(future_payload))
    finally:
        if held is not None:
            held.close()
        os.close(parent_fd)
    record: dict[str, object] = {
        "schema": "jx.tno.10my-reduced5-grid-adapter.first-1k-crossbind.v1",
        "batch_identity": _batch_identity(batch_order),
        "retained_arrays_file_sha256": H64_CLOSURE_ARRAYS_SHA256,
        "retained_lane_byte_offset": locator["byte_offset"],
        "years": list(H64_PREFIX_YEARS),
        "translated_checkpoint_sha256": checkpoint_digests,
        "comparison": "BITWISE_EQUAL_AFTER_EXACT_STAGE1_BLOCK_TO_ROW_LAYOUT_TRANSLATION",
        "passed": True,
    }
    record["content_sha256"] = _domain_hash(PREFIX_CROSSBIND_DOMAIN, record)
    return record


def construct_adapter_specs_v1(
    *,
    map_step_count: int,
    source_step_count: int,
) -> dict[str, object]:
    """Construct the exact bound WH, CUDA, route, and lifecycle call policy.

    This is a constructor-only boundary.  It neither resolves a backend nor
    performs source or tracer dynamics.
    """

    _require_int(map_step_count, "map_step_count", minimum=1)
    if map_step_count not in (3_200, 12_800):
        raise GridAdapterError("map step count is outside the frozen 50/200-year schedule")
    _require_int(source_step_count, "source_step_count", minimum=1)
    if source_step_count not in (64_000, 576_000, 640_000):
        raise GridAdapterError("source step count is outside the frozen phased schedule")
    required_bound = math.nextafter(
        2.0 * math.fsum(float.fromhex(value) for value in REDUCED_GM_HEX),
        math.inf,
    )
    if required_bound.hex() != REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX:
        raise GridAdapterError("relative acceleration certificate reconstruction differs")
    _binding_records()
    try:
        from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
        from jxplanetx.cuda_tracer_events import DOP853EventLocatorSpec
        from jxplanetx.engine.contracts import BackendSpec
        from jxplanetx.restricted_wh_lifecycle import (
            advance_cuda_restricted_wh_lifecycle,
            initialize_restricted_wh_lifecycle,
        )
        from jxplanetx.restricted_wh_route import RestrictedWHRouteSpec
        from jxplanetx.restricted_wh_source import (
            WisdomHolmanSourceTileSpec,
            build_wisdom_holman_source_tile,
        )
    except Exception as exc:
        raise GridAdapterError("exact adapter constructor imports failed") from exc

    backend = BackendSpec("cupy", "cuda:0", 128)
    source = WisdomHolmanSourceTileSpec(
        source_step_years=float.fromhex(H64_STEP_HEX),
        source_step_count=source_step_count,
        minimum_encounter_pair_separation_AU=1.0,
        minimum_jacobi_periapse_AU=0.1,
        maximum_initial_barycenter_position_norm_AU=1.0e-12,
        maximum_initial_barycenter_velocity_norm_AU_per_year=1.0e-12,
    )
    locator = DOP853EventLocatorSpec(
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-14,
        maximum_substep_fraction=0.125,
        dense_polynomial_validation_points=17,
        event_fraction_tolerance=1.0e-12,
        event_distance_tolerance_AU=1.0e-11,
        outward_radial_velocity_tolerance_AU_per_year=1.0e-13,
        maximum_accepted_steps=100_000,
    )
    route = RestrictedWHRouteSpec(
        event_locator=locator,
        maximum_primary_dynamical_phase=0.05,
        primary_outer_boundary_AU=4096.0,
        terminal_source_radii_AU=(0.0,) * 5,
    )
    fixed = FixedStepCUDARestrictedWHSpec(
        map_step_count=map_step_count,
        source_steps_per_map_step=1,
        backend=backend,
        maximum_primary_dynamical_phase=0.05,
        threads_per_block=128,
        primary_outer_boundary_AU=4096.0,
    )
    return {
        "backend": backend,
        "source": source,
        "event_locator": locator,
        "route": route,
        "fixed_step": fixed,
        "source_builder": build_wisdom_holman_source_tile,
        "lifecycle_initialize": initialize_restricted_wh_lifecycle,
        "lifecycle_advance": advance_cuda_restricted_wh_lifecycle,
        "time_scale": "TDB",
        "frame": "ICRF_J2000",
        "axes": "J2000_EQUATORIAL",
        "source_step_years_hex": H64_STEP_HEX,
        "source_encounter_radii_AU_hex": [float(1.0).hex()] * 5,
        "certified_relative_acceleration_bounds_AU_per_year2_hex": [
            REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX
        ] * 5,
        "route_worker_count": 1,
    }


def slice_frozen_source_tile_v1(
    full: object,
    *,
    local_start_step: int,
    map_step_count: int,
    tile_id: str,
) -> object:
    """Create the exact contiguous source-tile slice consumed by one map call."""

    _require_int(local_start_step, "local_start_step")
    _require_int(map_step_count, "map_step_count", minimum=1)
    if map_step_count not in (3_200, 12_800):
        raise GridAdapterError("source slice is outside the frozen 50/200-year schedule")
    if type(tile_id) is not str or not tile_id:
        raise GridAdapterError("source slice tile_id differs")
    _binding_records()
    try:
        import numpy as np
        from jxplanetx.cuda_restricted_wh import FrozenSourcePathTile
    except Exception as exc:
        raise GridAdapterError("exact source slice imports failed") from exc
    if type(full) is not FrozenSourcePathTile:
        raise GridAdapterError("source slicing requires an exact FrozenSourcePathTile")
    source_step_count = full.node_count - 1
    phase_is_exact = (
        (full.global_source_step == 0 and source_step_count == 64_000)
        or (full.global_source_step == 64_000 and source_step_count == 576_000)
        or (
            source_step_count == 640_000
            and full.global_source_step % 640_000 == 0
            and 1 <= full.global_source_step // 640_000 < 1_000
        )
    )
    if not phase_is_exact:
        raise GridAdapterError("source tile does not belong to the frozen phased schedule")
    absolute_start_step = full.global_source_step + local_start_step
    expected_map_step_count = 3_200 if absolute_start_step < 32_000_000 else 12_800
    if (
        map_step_count != expected_map_step_count
        or absolute_start_step % expected_map_step_count != 0
        or absolute_start_step + map_step_count > 640_000_000
    ):
        raise GridAdapterError("source slice start/length differs from the frozen segment schedule")
    expected_scalar_metadata = {
        "time_scale": "TDB",
        "frame": "ICRF_J2000",
        "axes": "J2000_EQUATORIAL",
        "coordinate_system": "PRIMARY_RELATIVE_NONROTATING",
        "length_unit": "AU",
        "time_unit": "year",
        "unit_system_id": "AU-YEAR",
        "primary_source_id": REDUCED_BODY_IDS[0],
    }
    if any(getattr(full, key) != value for key, value in expected_scalar_metadata.items()):
        raise GridAdapterError("source tile frame/unit/primary metadata differs")
    if (
        full.source_step_years.hex() != H64_STEP_HEX
        or full.epoch_origin_years.hex() not in NATIVE_EPOCH_ORIGINS_JULIAN_YEARS_HEX
        or full.source_ids != REDUCED_BODY_IDS
        or full.provenance.sha256 != full.source_schedule_sha256
        or tuple(value.hex() for value in full.source_gm_AU3_per_year2)
        != REDUCED_GM_HEX
        or tuple(value.hex() for value in full.source_encounter_radii_AU)
        != (float(1.0).hex(),) * len(REDUCED_BODY_IDS)
        or tuple(
            value.hex()
            for value in full.certified_relative_acceleration_bounds_AU_per_year2
        )
        != (REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX,) * len(REDUCED_BODY_IDS)
    ):
        raise GridAdapterError("source tile numerical identity differs")
    node_stop = local_start_step + map_step_count + 1
    if node_stop > full.node_count:
        raise GridAdapterError("source slice exceeds the full tile")
    return FrozenSourcePathTile(
        tile_id=tile_id,
        epoch_origin_years=full.epoch_origin_years,
        global_source_step=full.global_source_step + local_start_step,
        source_step_years=full.source_step_years,
        time_scale=full.time_scale,
        frame=full.frame,
        axes=full.axes,
        coordinate_system=full.coordinate_system,
        length_unit=full.length_unit,
        time_unit=full.time_unit,
        unit_system_id=full.unit_system_id,
        source_ids=full.source_ids,
        primary_source_id=full.primary_source_id,
        source_schedule_sha256=full.source_schedule_sha256,
        source_relative_positions_AU=np.ascontiguousarray(
            full.source_relative_positions_AU[local_start_step:node_stop], dtype=np.float64
        ),
        source_relative_velocities_AU_per_year=np.ascontiguousarray(
            full.source_relative_velocities_AU_per_year[local_start_step:node_stop], dtype=np.float64
        ),
        source_gm_AU3_per_year2=np.ascontiguousarray(full.source_gm_AU3_per_year2, dtype=np.float64),
        source_encounter_radii_AU=np.ascontiguousarray(full.source_encounter_radii_AU, dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.ascontiguousarray(
            full.certified_relative_acceleration_bounds_AU_per_year2, dtype=np.float64
        ),
        provenance=full.provenance,
    )


def _load_source_only(path: Path, expected_sha: str, name: str) -> types.ModuleType:
    payload = _read_regular_bytes(path, name)
    if _sha256(payload) != expected_sha:
        raise GridAdapterError(f"source binding differs: {path}")
    if name in sys.modules:
        raise GridAdapterError("private source-only module name is occupied")
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


class FrozenClassifierKernelV1:
    """The exact v2 classifier functions with one validated cached Stage-1 module."""

    def __init__(self) -> None:
        self._module_name = f"_jx_grid_classifier_v2_{id(self)}_{sys.flags.optimize}"
        self.classifier = _load_source_only(
            PROJECT_ROOT / "benchmarks/jx_tno_b22_classifier_v2.py",
            "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98",
            self._module_name,
        )
        self.stage1 = self.classifier.load_bound_stage1_v2()
        self.classifier.validate_stage1_source_v2()
        self.classifier.validate_stage1_source_v2 = lambda: {
            "sha256": "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
        }
        self.classifier.load_bound_stage1_v2 = lambda: self.stage1
        self._closed = False

    def derive(self, batch_order: int, state_payload: bytes) -> tuple[bytes, bytes]:
        if self._closed:
            raise GridAdapterError("classifier kernel is closed")
        _require_int(batch_order, "batch_order")
        if batch_order >= len(OBJECT_IDS):
            raise GridAdapterError("batch order exceeds the frozen cohort")
        rows = decode_state_record_v1(state_payload)
        massive = tuple(
            self.stage1.MassiveState(
                REDUCED_NAIF_IDS[index],
                REDUCED_BODY_IDS[index],
                float.fromhex(REDUCED_GM_HEX[index]),
                tuple(rows[index][:3]),
                tuple(rows[index][3:]),
            )
            for index in range(5)
        )
        tracers = tuple(
            self.stage1.TracerState(
                OBJECT_IDS[batch_order],
                SPK_IDS[batch_order],
                "NOMINAL" if index == 0 else "COVARIANCE_CLONE",
                None if index == 0 else HOLDOUT_CLONE_INDICES[index - 1],
                tuple(rows[5 + index][:3]),
                tuple(rows[5 + index][3:]),
            )
            for index in range(9)
        )
        epoch = float.fromhex(NATIVE_EPOCHS_JD_TDB_HEX[batch_order])
        digest_hex = self.classifier.node_state_sha256_v2(massive, tracers, batch_order, epoch)
        extracted = self.classifier.extract_batch_elements_v2(
            massive,
            tracers,
            batch_order=batch_order,
            native_epoch_jd_tdb=epoch,
            upstream_node_sha256=digest_hex,
        )
        entities = (extracted["neptune"], *extracted["tracers"])
        values: list[float] = []
        for entity in entities:
            if type(entity) is not dict:
                raise GridAdapterError("classifier element row is not a mapping")
            for field in ELEMENT_FIELDS:
                value = entity.get(field)
                if type(value) is not float or not math.isfinite(value):
                    raise GridAdapterError("classifier returned a non-finite element")
                values.append(value)
        element_payload = struct.pack("<60d", *values)
        if len(element_payload) != PRODUCTION_GEOMETRY.element_record_bytes:
            raise GridAdapterError("classifier element record size differs")
        return bytes.fromhex(digest_hex), element_payload

    def close(self) -> None:
        if self._closed:
            return
        for path, expected in (
            (PROJECT_ROOT / "benchmarks/jx_tno_b22_classifier_v2.py", "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98"),
            (PROJECT_ROOT / "benchmarks/jx_tno_stage1_dynamics_qualification_v4.py", "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"),
        ):
            if _sha256(_read_regular_bytes(path, "classifier-kernel closing binding")) != expected:
                raise GridAdapterError("classifier kernel source changed during use")
        self._closed = True
        sys.modules.pop(self._module_name, None)

    def __enter__(self) -> "FrozenClassifierKernelV1":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


DeriveCallable = Callable[[int, bytes], tuple[bytes, bytes]]
PrefixCrossbindCallable = Callable[[int, Sequence[tuple[int, bytes]]], Mapping[str, Any]]


def _require_production_derive(derive: DeriveCallable) -> None:
    owner = getattr(derive, "__self__", None)
    function = getattr(derive, "__func__", None)
    if (
        type(owner) is not FrozenClassifierKernelV1
        or function is not FrozenClassifierKernelV1.derive
        or owner._closed
    ):
        raise GridAdapterError("production replay requires the exact live frozen classifier kernel method")


def _require_prefix_crossbind(
    prefix_crossbind: PrefixCrossbindCallable,
    geometry: GridGeometry,
    *,
    test_only_geometry: bool,
) -> None:
    if geometry == PRODUCTION_GEOMETRY:
        if prefix_crossbind is not validate_first_1000_year_prefix_v1:
            raise GridAdapterError("production finalization requires the exact first-1k crossbinder")
    elif not test_only_geometry or not callable(prefix_crossbind):
        raise GridAdapterError("nonproduction prefix crossbinding requires the test-only boundary")


def _prefix_crossbinding_from_reader_v1(
    batch_order: int,
    geometry: GridGeometry,
    prefix_crossbind: PrefixCrossbindCallable,
    reader: Callable[[int, int], bytes],
) -> str:
    if geometry == PRODUCTION_GEOMETRY:
        labels = H64_PREFIX_YEARS
    else:
        labels = tuple(range(min(3, geometry.short_node_count)))
    checkpoints: list[tuple[int, bytes]] = []
    for short_node, label in enumerate(labels):
        offset = (
            batch_order * geometry.short_node_count + short_node
        ) * geometry.state_record_bytes
        payload = reader(geometry.state_record_bytes, offset)
        decode_state_record_v1(payload, geometry)
        checkpoints.append((label, payload))
    receipt = prefix_crossbind(batch_order, tuple(checkpoints))
    if type(receipt) is not dict:
        raise GridAdapterError("first-1k crossbinding receipt is not an exact mapping")
    digest = _require_sha(
        receipt.get("content_sha256"),
        "rederived first-1k crossbinding",
        allow_zero=False,
    )
    if geometry == PRODUCTION_GEOMETRY:
        if (
            receipt.get("schema")
            != "jx.tno.10my-reduced5-grid-adapter.first-1k-crossbind.v1"
            or receipt.get("batch_identity") != _batch_identity(batch_order)
            or receipt.get("years") != list(H64_PREFIX_YEARS)
            or receipt.get("passed") is not True
            or digest != _domain_hash(PREFIX_CROSSBIND_DOMAIN, receipt)
        ):
            raise GridAdapterError("production first-1k crossbinding receipt differs")
    return digest


class _FragmentWriter:
    def __init__(self, directory_fd: int, name: str) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            raise GridAdapterError(f"cannot create fragment {name}") from exc
        self.name = name
        self.digest = hashlib.sha256()
        self.size = 0
        self.closed = False

    def write(self, payload: bytes) -> None:
        if self.closed or type(payload) is not bytes:
            raise GridAdapterError("fragment writer input differs")
        view = memoryview(payload)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise GridAdapterError("fragment write made no progress")
            view = view[written:]
        self.digest.update(payload)
        self.size += len(payload)

    def finish(self) -> dict[str, object]:
        if self.closed:
            raise GridAdapterError("fragment writer was already closed")
        os.fsync(self.fd)
        observed = os.fstat(self.fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or observed.st_size != self.size:
            raise GridAdapterError("fragment inode differs at close")
        os.close(self.fd)
        self.closed = True
        return {"name": self.name, "byte_length": self.size, "sha256": self.digest.hexdigest()}

    def abort_close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def _write_create_only_at(directory_fd: int, name: str, payload: bytes) -> None:
    writer = _FragmentWriter(directory_fd, name)
    try:
        writer.write(payload)
        writer.finish()
    except Exception:
        writer.abort_close()
        raise


def _batch_identity(batch_order: int) -> dict[str, object]:
    _require_int(batch_order, "batch_order")
    if batch_order >= len(OBJECT_IDS):
        raise GridAdapterError("batch order exceeds the cohort")
    return {
        "ephemeris": "DE441",
        "batch_order": batch_order,
        "object_id": OBJECT_IDS[batch_order],
        "spk_id": SPK_IDS[batch_order],
        "native_epoch_jd_tdb_hex": NATIVE_EPOCHS_JD_TDB_HEX[batch_order],
        "massive_body_ids": list(REDUCED_BODY_IDS),
        "massive_naif_ids": list(REDUCED_NAIF_IDS),
        "massive_gm_hex": list(REDUCED_GM_HEX),
        "tracer_order": [
            {"sample_kind": "NOMINAL", "clone_index": None},
            *(
                {"sample_kind": "COVARIANCE_CLONE", "clone_index": clone}
                for clone in HOLDOUT_CLONE_INDICES
            ),
        ],
    }


def _geometry_record(geometry: GridGeometry) -> dict[str, int]:
    return {
        "batch_count": geometry.batch_count,
        "total_steps": geometry.total_steps,
        "shard_steps": geometry.shard_steps,
        "shard_count": geometry.shard_count,
        "b22_interval_steps": geometry.b22_interval_steps,
        "b22_node_count": geometry.b22_node_count,
        "short_interval_steps": geometry.short_interval_steps,
        "short_final_step": geometry.short_final_step,
        "short_node_count": geometry.short_node_count,
        "union_node_count": geometry.union_node_count,
        "massive_count": geometry.massive_count,
        "tracer_count": geometry.tracer_count,
        "state_record_bytes": geometry.state_record_bytes,
        "element_entity_count": geometry.element_entity_count,
        "element_field_count": geometry.element_field_count,
        "element_record_bytes": geometry.element_record_bytes,
    }


def write_shard_v1(
    root: Path,
    *,
    batch_order: int,
    shard_index: int,
    checkpoints: Iterable[tuple[int, bytes]],
    predecessor_manifest_content_sha256: str,
    start_continuation_sha256: str,
    final_continuation_sha256: str,
    source_chunk_receipt_sha256: str,
    route_prefix_sha256: str,
    runtime_projection_sha256: str,
    first_1000_year_prefix_crossbinding_sha256: str,
    geometry: GridGeometry = PRODUCTION_GEOMETRY,
    derive: DeriveCallable,
    test_only_geometry: bool = False,
) -> dict[str, Any]:
    """Create one immutable shard, publishing ``MANIFEST.json`` last."""

    if geometry != PRODUCTION_GEOMETRY and not test_only_geometry:
        raise GridAdapterError("nonproduction geometry requires the explicit test-only boundary")
    if geometry == PRODUCTION_GEOMETRY:
        _require_production_derive(derive)
    _require_int(batch_order, "batch_order")
    if batch_order >= geometry.batch_count or batch_order >= len(OBJECT_IDS):
        raise GridAdapterError("batch order exceeds geometry")
    _require_int(shard_index, "shard_index")
    if shard_index >= geometry.shard_count:
        raise GridAdapterError("shard index exceeds geometry")
    predecessor = _require_sha(predecessor_manifest_content_sha256, "predecessor manifest")
    if (shard_index == 0) != (predecessor == ZERO_SHA256):
        raise GridAdapterError("genesis/predecessor relationship differs")
    for label, value in (
        ("start continuation", start_continuation_sha256),
        ("final continuation", final_continuation_sha256),
        ("source chunk receipt", source_chunk_receipt_sha256),
        ("route prefix", route_prefix_sha256),
        ("runtime projection", runtime_projection_sha256),
        ("first-1k prefix crossbinding", first_1000_year_prefix_crossbinding_sha256),
    ):
        _require_sha(value, label, allow_zero=False)
    if root.exists() or root.is_symlink():
        raise GridAdapterError("shard root already exists")
    root.mkdir(mode=0o700, parents=False)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    writers = {name: _FragmentWriter(directory_fd, name) for name in FRAGMENT_NAMES}
    try:
        expected_steps = geometry.expected_steps(shard_index)
        observed_count = 0
        b22_count = 0
        short_count = 0
        for expected_step, item in zip(expected_steps, checkpoints, strict=True):
            if type(item) is not tuple or len(item) != 2:
                raise GridAdapterError("checkpoint stream item differs")
            step, state_payload = item
            if type(step) is not int or step != expected_step:
                raise GridAdapterError("checkpoint step/order differs")
            decode_state_record_v1(state_payload, geometry)
            b22_index = geometry.b22_index(step)
            short_index = geometry.short_index(step)
            if b22_index is not None:
                digest, elements = derive(batch_order, state_payload)
                if type(digest) is not bytes or len(digest) != 32:
                    raise GridAdapterError("node digest derivation differs")
                if type(elements) is not bytes or len(elements) != geometry.element_record_bytes:
                    raise GridAdapterError("element derivation differs")
                writers[FRAGMENT_NAMES[0]].write(state_payload)
                writers[FRAGMENT_NAMES[1]].write(elements)
                writers[FRAGMENT_NAMES[2]].write(digest)
                b22_count += 1
            if short_index is not None:
                writers[FRAGMENT_NAMES[3]].write(state_payload)
                short_count += 1
            observed_count += 1
        if observed_count != len(expected_steps):
            raise GridAdapterError("checkpoint stream ended early")
        fragment_records = [writers[name].finish() for name in FRAGMENT_NAMES]
        manifest: dict[str, Any] = {
            "schema": SHARD_SCHEMA,
            "batch_identity": _batch_identity(batch_order),
            "geometry": _geometry_record(geometry),
            "shard_index": shard_index,
            "ownership": "SHARD_ZERO_OWNS_GENESIS_AND_RIGHT_ENDPOINTS_LATER_SHARDS_OWN_RIGHT_ENDPOINTS_ONLY",
            "first_global_step": expected_steps[0],
            "last_global_step": expected_steps[-1],
            "union_checkpoint_count": observed_count,
            "b22_checkpoint_count": b22_count,
            "short_checkpoint_count": short_count,
            "predecessor_manifest_content_sha256": predecessor,
            "start_continuation_sha256": start_continuation_sha256,
            "final_continuation_sha256": final_continuation_sha256,
            "source_chunk_receipt_sha256": source_chunk_receipt_sha256,
            "route_prefix_sha256": route_prefix_sha256,
            "runtime_projection_sha256": runtime_projection_sha256,
            "first_1000_year_prefix_crossbinding_sha256": (
                first_1000_year_prefix_crossbinding_sha256
            ),
            "fragments": fragment_records,
            "claims": {
                "writer_executed_dynamics": False,
                "writer_generated_classification": False,
                "upstream_dynamics_status": "NOT_ATTESTED_BY_THIS_MECHANICAL_SHARD_WRITER",
                "science_result": False,
                "shard_writer_mechanics_only": True,
            },
        }
        manifest["content_sha256"] = _domain_hash(SHARD_DOMAIN, manifest)
        _write_create_only_at(directory_fd, "MANIFEST.json", _canonical_json(manifest))
        os.fsync(directory_fd)
        return manifest
    except Exception:
        for writer in writers.values():
            writer.abort_close()
        raise
    finally:
        os.close(directory_fd)


class _HeldFile:
    def __init__(self, directory_fd: int, name: str, label: str) -> None:
        self.name = name
        self.label = label
        try:
            self.fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        except OSError as exc:
            raise GridAdapterError(f"cannot open {label}") from exc
        self.before = os.fstat(self.fd)
        if not stat.S_ISREG(self.before.st_mode) or self.before.st_nlink != 1:
            os.close(self.fd)
            raise GridAdapterError(f"{label} is linked or nonregular")
        self.directory_fd = directory_fd
        try:
            self.path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            os.close(self.fd)
            raise GridAdapterError(f"cannot stat {label}") from exc
        if (
            not stat.S_ISREG(self.path_before.st_mode)
            or (self.path_before.st_dev, self.path_before.st_ino)
            != (self.before.st_dev, self.before.st_ino)
        ):
            os.close(self.fd)
            raise GridAdapterError(f"{label} path differs from opened inode")

    @property
    def size(self) -> int:
        return self.before.st_size

    def pread(self, size: int, offset: int) -> bytes:
        _require_int(size, "pread size")
        _require_int(offset, "pread offset")
        result = bytearray()
        while len(result) < size:
            block = os.pread(self.fd, size - len(result), offset + len(result))
            if not block:
                break
            result.extend(block)
        if len(result) != size:
            raise GridAdapterError(f"{self.label} ended early")
        return bytes(result)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < self.size:
            block = self.pread(min(1 << 20, self.size - offset), offset)
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    def close(self) -> None:
        after = os.fstat(self.fd)
        try:
            path_after = os.stat(self.name, dir_fd=self.directory_fd, follow_symlinks=False)
        except OSError as exc:
            os.close(self.fd)
            raise GridAdapterError(f"cannot restat {self.label}") from exc
        os.close(self.fd)
        if (
            _stat_identity(self.before) != _stat_identity(after)
            or _stat_identity(self.path_before) != _stat_identity(path_after)
            or (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise GridAdapterError(f"{self.label} changed while held")


def _load_shard_manifest(directory_fd: int) -> dict[str, Any]:
    held = _HeldFile(directory_fd, "MANIFEST.json", "shard manifest")
    try:
        payload = held.pread(held.size, 0)
        manifest = _strict_json(payload)
        if payload != _canonical_json(manifest):
            raise GridAdapterError("shard manifest is not canonical")
        if manifest.get("content_sha256") != _domain_hash(SHARD_DOMAIN, manifest):
            raise GridAdapterError("shard manifest content digest differs")
        return manifest
    finally:
        held.close()


def _validate_retained_shard_manifest_v1(
    manifest: object,
    geometry: GridGeometry,
    *,
    expected_batch_order: int,
    expected_shard_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate one fully retained shard manifest without its source directory."""

    if type(manifest) is not dict:
        raise GridAdapterError("retained shard manifest is not an exact mapping")
    _guard_integer_identities(manifest)
    if (
        set(manifest) != SHARD_MANIFEST_KEYS
        or manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("geometry") != _geometry_record(geometry)
        or manifest.get("content_sha256") != _domain_hash(SHARD_DOMAIN, manifest)
    ):
        raise GridAdapterError("retained shard manifest identity differs")
    if manifest.get("batch_identity") != _batch_identity(expected_batch_order):
        raise GridAdapterError("retained shard batch identity differs")
    if type(manifest.get("shard_index")) is not int or manifest["shard_index"] != expected_shard_index:
        raise GridAdapterError("retained shard index differs")
    steps = geometry.expected_steps(expected_shard_index)
    b22_steps = tuple(step for step in steps if geometry.b22_index(step) is not None)
    short_steps = tuple(step for step in steps if geometry.short_index(step) is not None)
    if (
        manifest.get("ownership")
        != "SHARD_ZERO_OWNS_GENESIS_AND_RIGHT_ENDPOINTS_LATER_SHARDS_OWN_RIGHT_ENDPOINTS_ONLY"
        or manifest.get("first_global_step") != steps[0]
        or manifest.get("last_global_step") != steps[-1]
        or manifest.get("union_checkpoint_count") != len(steps)
        or manifest.get("b22_checkpoint_count") != len(b22_steps)
        or manifest.get("short_checkpoint_count") != len(short_steps)
    ):
        raise GridAdapterError("retained shard schedule summary differs")
    fragments = manifest.get("fragments")
    if (
        type(fragments) is not list
        or [row.get("name") if type(row) is dict else None for row in fragments]
        != list(FRAGMENT_NAMES)
    ):
        raise GridAdapterError("retained shard fragment roster differs")
    expected_sizes = (
        len(b22_steps) * geometry.state_record_bytes,
        len(b22_steps) * geometry.element_record_bytes,
        len(b22_steps) * 32,
        len(short_steps) * geometry.state_record_bytes,
    )
    for row, name, size in zip(fragments, FRAGMENT_NAMES, expected_sizes, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"name", "byte_length", "sha256"}
            or row.get("name") != name
            or type(row.get("byte_length")) is not int
            or row.get("byte_length") != size
        ):
            raise GridAdapterError("retained shard fragment record differs")
        _require_sha(row.get("sha256"), f"retained {name} digest")
    for key in (
        "predecessor_manifest_content_sha256",
        "start_continuation_sha256",
        "final_continuation_sha256",
        "source_chunk_receipt_sha256",
        "route_prefix_sha256",
        "runtime_projection_sha256",
        "first_1000_year_prefix_crossbinding_sha256",
    ):
        _require_sha(
            manifest.get(key),
            key,
            allow_zero=key == "predecessor_manifest_content_sha256",
        )
    predecessor = manifest["predecessor_manifest_content_sha256"]
    if (expected_shard_index == 0) != (predecessor == ZERO_SHA256):
        raise GridAdapterError("retained shard genesis/predecessor relationship differs")
    if not _exact_tree_equal(manifest.get("claims"), {
        "writer_executed_dynamics": False,
        "writer_generated_classification": False,
        "upstream_dynamics_status": "NOT_ATTESTED_BY_THIS_MECHANICAL_SHARD_WRITER",
        "science_result": False,
        "shard_writer_mechanics_only": True,
    }):
        raise GridAdapterError("retained shard claim ceiling differs")
    return steps, b22_steps, short_steps


def validate_shard_v1(
    root: Path,
    *,
    geometry: GridGeometry = PRODUCTION_GEOMETRY,
    derive: DeriveCallable,
    test_only_geometry: bool = False,
) -> dict[str, Any]:
    if geometry != PRODUCTION_GEOMETRY and not test_only_geometry:
        raise GridAdapterError("nonproduction validation requires test-only boundary")
    if geometry == PRODUCTION_GEOMETRY:
        _require_production_derive(derive)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    held_files: dict[str, _HeldFile] = {}
    try:
        if tuple(sorted(os.listdir(directory_fd))) != tuple(sorted((*FRAGMENT_NAMES, "MANIFEST.json"))):
            raise GridAdapterError("shard root roster differs")
        manifest = _load_shard_manifest(directory_fd)
        _guard_integer_identities(manifest)
        if set(manifest) != SHARD_MANIFEST_KEYS:
            raise GridAdapterError("shard manifest key roster differs")
        if manifest.get("schema") != SHARD_SCHEMA or manifest.get("geometry") != _geometry_record(geometry):
            raise GridAdapterError("shard schema or geometry differs")
        batch_identity = manifest.get("batch_identity")
        if type(batch_identity) is not dict:
            raise GridAdapterError("shard batch identity differs")
        batch_order = batch_identity.get("batch_order")
        if type(batch_order) is not int or batch_identity != _batch_identity(batch_order):
            raise GridAdapterError("shard batch identity is not exact")
        shard_index = manifest.get("shard_index")
        if type(shard_index) is not int:
            raise GridAdapterError("shard index type differs")
        expected_steps = geometry.expected_steps(shard_index)
        b22_steps = tuple(step for step in expected_steps if geometry.b22_index(step) is not None)
        short_steps = tuple(step for step in expected_steps if geometry.short_index(step) is not None)
        if (
            manifest.get("ownership")
            != "SHARD_ZERO_OWNS_GENESIS_AND_RIGHT_ENDPOINTS_LATER_SHARDS_OWN_RIGHT_ENDPOINTS_ONLY"
            or
            manifest.get("first_global_step") != expected_steps[0]
            or manifest.get("last_global_step") != expected_steps[-1]
            or manifest.get("union_checkpoint_count") != len(expected_steps)
            or manifest.get("b22_checkpoint_count") != len(b22_steps)
            or manifest.get("short_checkpoint_count") != len(short_steps)
        ):
            raise GridAdapterError("shard schedule summary differs")
        fragments = manifest.get("fragments")
        if type(fragments) is not list or [row.get("name") if type(row) is dict else None for row in fragments] != list(FRAGMENT_NAMES):
            raise GridAdapterError("shard fragment roster differs")
        records = {row["name"]: row for row in fragments}
        expected_sizes = {
            FRAGMENT_NAMES[0]: len(b22_steps) * geometry.state_record_bytes,
            FRAGMENT_NAMES[1]: len(b22_steps) * geometry.element_record_bytes,
            FRAGMENT_NAMES[2]: len(b22_steps) * 32,
            FRAGMENT_NAMES[3]: len(short_steps) * geometry.state_record_bytes,
        }
        for name in FRAGMENT_NAMES:
            held = _HeldFile(directory_fd, name, name)
            held_files[name] = held
            record = records[name]
            if (
                set(record) != {"name", "byte_length", "sha256"}
                or record["byte_length"] != expected_sizes[name]
                or held.size != expected_sizes[name]
                or held.sha256() != record["sha256"]
            ):
                raise GridAdapterError(f"fragment identity differs: {name}")
        for local_index, step in enumerate(b22_steps):
            state_payload = held_files[FRAGMENT_NAMES[0]].pread(geometry.state_record_bytes, local_index * geometry.state_record_bytes)
            digest, elements = derive(batch_order, state_payload)
            if digest != held_files[FRAGMENT_NAMES[2]].pread(32, local_index * 32):
                raise GridAdapterError("retained B22 node digest replay differs")
            if elements != held_files[FRAGMENT_NAMES[1]].pread(geometry.element_record_bytes, local_index * geometry.element_record_bytes):
                raise GridAdapterError("retained B22 element replay differs")
            short_index = geometry.short_index(step)
            if short_index is not None:
                short_local = short_steps.index(step)
                if state_payload != held_files[FRAGMENT_NAMES[3]].pread(geometry.state_record_bytes, short_local * geometry.state_record_bytes):
                    raise GridAdapterError("B22/short overlap differs")
        for short_local in range(len(short_steps)):
            decode_state_record_v1(
                held_files[FRAGMENT_NAMES[3]].pread(
                    geometry.state_record_bytes,
                    short_local * geometry.state_record_bytes,
                ),
                geometry,
            )
        for key in (
            "predecessor_manifest_content_sha256",
            "start_continuation_sha256",
            "final_continuation_sha256",
            "source_chunk_receipt_sha256",
            "route_prefix_sha256",
            "runtime_projection_sha256",
            "first_1000_year_prefix_crossbinding_sha256",
        ):
            _require_sha(
                manifest.get(key),
                key,
                allow_zero=key == "predecessor_manifest_content_sha256",
            )
        predecessor = manifest["predecessor_manifest_content_sha256"]
        if (shard_index == 0) != (predecessor == ZERO_SHA256):
            raise GridAdapterError("shard genesis/predecessor relationship differs")
        claims = manifest.get("claims")
        if not _exact_tree_equal(claims, {
            "writer_executed_dynamics": False,
            "writer_generated_classification": False,
            "upstream_dynamics_status": "NOT_ATTESTED_BY_THIS_MECHANICAL_SHARD_WRITER",
            "science_result": False,
            "shard_writer_mechanics_only": True,
        }):
            raise GridAdapterError("shard claim ceiling differs")
        return manifest
    finally:
        errors: list[Exception] = []
        for held in held_files.values():
            try:
                held.close()
            except Exception as exc:
                errors.append(exc)
        os.close(directory_fd)
        if errors:
            raise errors[0]


def _read_verified_fragment(root: Path, manifest: Mapping[str, Any], name: str) -> bytes:
    records = manifest["fragments"]
    record = next(row for row in records if row["name"] == name)
    payload = _read_regular_bytes(root / name, f"finalizer input {name}")
    if len(payload) != record["byte_length"] or _sha256(payload) != record["sha256"]:
        raise GridAdapterError("finalizer fragment differs after validation")
    return payload


def _pwrite_all(fd: int, payload: bytes, offset: int) -> None:
    view = memoryview(payload)
    written_total = 0
    while view:
        written = os.pwrite(fd, view, offset + written_total)
        if written <= 0:
            raise GridAdapterError("pwrite made no progress")
        view = view[written:]
        written_total += written


def _sha_fd_range(fd: int, offset: int, size: int) -> str:
    _require_int(offset, "hash range offset")
    _require_int(size, "hash range size")
    digest = hashlib.sha256()
    consumed = 0
    while consumed < size:
        block = os.pread(fd, min(1 << 20, size - consumed), offset + consumed)
        if not block:
            raise GridAdapterError("hashed file range ended early")
        digest.update(block)
        consumed += len(block)
    return digest.hexdigest()


def _sha_held_range(held: _HeldFile, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    while consumed < size:
        block = held.pread(min(1 << 20, size - consumed), offset + consumed)
        digest.update(block)
        consumed += len(block)
    return digest.hexdigest()


def _batch_stream_root_record_v1(
    batch_order: int,
    manifests: Sequence[Mapping[str, Any]],
    binary_range_sha256: Mapping[str, str],
) -> dict[str, object]:
    if len(manifests) == 0:
        raise GridAdapterError("batch stream root requires a nonempty chain")
    prefix_values = [item.get("first_1000_year_prefix_crossbinding_sha256") for item in manifests]
    if len(set(prefix_values)) != 1:
        raise GridAdapterError("first-1k crossbinding changed within a batch chain")
    for name in FINAL_BINARY_NAMES:
        _require_sha(binary_range_sha256.get(name), f"batch {name} series")
    record: dict[str, object] = {
        "batch_identity": _batch_identity(batch_order),
        "shard_count": len(manifests),
        "final_shard_manifest_content_sha256": manifests[-1]["content_sha256"],
        "b22_states_series_sha256": binary_range_sha256[FINAL_BINARY_NAMES[0]],
        "b22_elements_series_sha256": binary_range_sha256[FINAL_BINARY_NAMES[1]],
        "b22_node_digest_series_sha256": binary_range_sha256[FINAL_BINARY_NAMES[2]],
        "ml_short_states_series_sha256": binary_range_sha256[FINAL_BINARY_NAMES[3]],
        "source_chunk_receipt_roster_sha256": _sha256(
            BATCH_STREAM_ROOT_DOMAIN
            + b"source\0"
            + _canonical_json([item["source_chunk_receipt_sha256"] for item in manifests])
        ),
        "route_prefix_roster_sha256": _sha256(
            BATCH_STREAM_ROOT_DOMAIN
            + b"route\0"
            + _canonical_json([item["route_prefix_sha256"] for item in manifests])
        ),
        "runtime_projection_roster_sha256": _sha256(
            BATCH_STREAM_ROOT_DOMAIN
            + b"runtime\0"
            + _canonical_json([item["runtime_projection_sha256"] for item in manifests])
        ),
        "continuation_roster_sha256": _sha256(
            BATCH_STREAM_ROOT_DOMAIN
            + b"continuation\0"
            + _canonical_json(
                [
                    [item["start_continuation_sha256"], item["final_continuation_sha256"]]
                    for item in manifests
                ]
            )
        ),
        "first_1000_year_prefix_crossbinding_sha256": prefix_values[0],
    }
    record["content_sha256"] = _domain_hash(BATCH_STREAM_ROOT_DOMAIN, record)
    return record


def finalize_shards_v1(
    shard_roots_by_batch: Sequence[Sequence[Path]],
    output_root: Path,
    *,
    geometry: GridGeometry = PRODUCTION_GEOMETRY,
    derive: DeriveCallable,
    prefix_crossbind: PrefixCrossbindCallable = validate_first_1000_year_prefix_v1,
    test_only_geometry: bool = False,
) -> dict[str, Any]:
    """Validate a complete chain and publish the six-file final roster."""

    if geometry != PRODUCTION_GEOMETRY and not test_only_geometry:
        raise GridAdapterError("nonproduction finalization requires test-only boundary")
    _require_prefix_crossbind(
        prefix_crossbind, geometry, test_only_geometry=test_only_geometry
    )
    if len(shard_roots_by_batch) != geometry.batch_count:
        raise GridAdapterError("finalizer batch roster differs")
    if output_root.exists() or output_root.is_symlink():
        raise GridAdapterError("final output root already exists")
    output_root.mkdir(mode=0o700, parents=False)
    directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    output_fds: dict[str, int] = {}
    try:
        sizes = geometry.output_sizes()
        for name in FINAL_BINARY_NAMES:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
            os.ftruncate(fd, sizes[name])
            output_fds[name] = fd
        shard_manifests: list[list[dict[str, Any]]] = []
        for batch_order, roots in enumerate(shard_roots_by_batch):
            if len(roots) != geometry.shard_count:
                raise GridAdapterError("finalizer shard roster differs")
            predecessor = ZERO_SHA256
            previous_final: str | None = None
            batch_manifests: list[dict[str, Any]] = []
            for shard_index, root in enumerate(roots):
                manifest = validate_shard_v1(root, geometry=geometry, derive=derive, test_only_geometry=test_only_geometry)
                if (
                    manifest["batch_identity"] != _batch_identity(batch_order)
                    or manifest["shard_index"] != shard_index
                    or manifest["predecessor_manifest_content_sha256"] != predecessor
                ):
                    raise GridAdapterError("shard chain identity differs")
                if shard_index and manifest["start_continuation_sha256"] != previous_final:
                    raise GridAdapterError("shard continuation chain differs")
                predecessor = manifest["content_sha256"]
                previous_final = manifest["final_continuation_sha256"]
                batch_manifests.append(manifest)
                steps = geometry.expected_steps(shard_index)
                b22_steps = tuple(step for step in steps if geometry.b22_index(step) is not None)
                short_steps = tuple(step for step in steps if geometry.short_index(step) is not None)
                b22_states = _read_verified_fragment(root, manifest, FRAGMENT_NAMES[0])
                node_elements = _read_verified_fragment(root, manifest, FRAGMENT_NAMES[1])
                node_digests = _read_verified_fragment(root, manifest, FRAGMENT_NAMES[2])
                short_states = _read_verified_fragment(root, manifest, FRAGMENT_NAMES[3])
                first_b22 = geometry.b22_index(b22_steps[0])
                first_short = geometry.short_index(short_steps[0]) if short_steps else None
                if first_b22 is None:
                    raise GridAdapterError("validated shard has no B22 anchor")
                _pwrite_all(
                    output_fds[FINAL_BINARY_NAMES[0]],
                    b22_states,
                    (batch_order * geometry.b22_node_count + first_b22) * geometry.state_record_bytes,
                )
                _pwrite_all(
                    output_fds[FINAL_BINARY_NAMES[2]],
                    node_digests,
                    (batch_order * geometry.b22_node_count + first_b22) * 32,
                )
                if first_short is not None:
                    _pwrite_all(
                        output_fds[FINAL_BINARY_NAMES[3]],
                        short_states,
                        (batch_order * geometry.short_node_count + first_short) * geometry.state_record_bytes,
                    )
                node_count = len(b22_steps)
                for entity in range(geometry.element_entity_count):
                    for field in range(geometry.element_field_count):
                        pieces = bytearray()
                        value_index = entity * geometry.element_field_count + field
                        for node in range(node_count):
                            source_offset = node * geometry.element_record_bytes + value_index * 8
                            pieces.extend(node_elements[source_offset : source_offset + 8])
                        destination_value_index = (
                            ((batch_order * geometry.element_entity_count + entity) * geometry.element_field_count + field)
                            * geometry.b22_node_count
                            + first_b22
                        )
                        _pwrite_all(output_fds[FINAL_BINARY_NAMES[1]], bytes(pieces), destination_value_index * 8)
            shard_manifests.append(batch_manifests)
        file_records: list[dict[str, object]] = []
        for name in FINAL_BINARY_NAMES:
            fd = output_fds[name]
            os.fsync(fd)
            size = os.fstat(fd).st_size
            digest = hashlib.sha256()
            offset = 0
            while offset < size:
                block = os.pread(fd, min(1 << 20, size - offset), offset)
                if not block:
                    raise GridAdapterError("final output ended early while hashing")
                digest.update(block)
                offset += len(block)
            file_records.append({"name": name, "byte_length": size, "sha256": digest.hexdigest()})
        batch_stream_roots: list[dict[str, object]] = []
        per_batch_sizes = {
            FINAL_BINARY_NAMES[0]: geometry.b22_node_count * geometry.state_record_bytes,
            FINAL_BINARY_NAMES[1]: (
                geometry.element_entity_count
                * geometry.element_field_count
                * geometry.b22_node_count
                * 8
            ),
            FINAL_BINARY_NAMES[2]: geometry.b22_node_count * 32,
            FINAL_BINARY_NAMES[3]: geometry.short_node_count * geometry.state_record_bytes,
        }
        for batch_order, manifests in enumerate(shard_manifests):
            rederived_prefix = _prefix_crossbinding_from_reader_v1(
                batch_order,
                geometry,
                prefix_crossbind,
                lambda size, offset, fd=output_fds[FINAL_BINARY_NAMES[3]]: os.pread(
                    fd, size, offset
                ),
            )
            if any(
                item["first_1000_year_prefix_crossbinding_sha256"]
                != rederived_prefix
                for item in manifests
            ):
                raise GridAdapterError("shard first-1k crossbinding differs from final short states")
            range_hashes = {
                name: _sha_fd_range(
                    output_fds[name],
                    batch_order * per_batch_sizes[name],
                    per_batch_sizes[name],
                )
                for name in FINAL_BINARY_NAMES
            }
            batch_stream_roots.append(
                _batch_stream_root_record_v1(batch_order, manifests, range_hashes)
            )
        manifest: dict[str, Any] = {
            "schema": FINAL_MANIFEST_SCHEMA,
            "geometry": _geometry_record(geometry),
            "files": file_records,
            "shard_manifests_by_batch": shard_manifests,
            "batch_stream_roots": batch_stream_roots,
            "layout": {
                "state": "BATCH_NODE_BODY_COMPONENT_ROW_INTERLEAVED_F64LE",
                "elements": "BATCH_ENTITY_FIELD_NODE_F64LE",
                "node_digest": "BATCH_NODE_RAW_SHA256",
                "ml_long": "EXACT_B22_STRIDE_5_VIEW_NO_DUPLICATE_FILE",
            },
        }
        manifest["content_sha256"] = _domain_hash(FINAL_MANIFEST_DOMAIN, manifest)
        _write_create_only_at(directory_fd, "MANIFEST.json", _canonical_json(manifest))
        result: dict[str, Any] = {
            "schema": FINAL_RESULT_SCHEMA,
            "status": "GRID_ARTIFACTS_ASSEMBLED_AND_REPLAYED_BY_MECHANICAL_FINALIZER",
            "manifest_content_sha256": manifest["content_sha256"],
            "claims": {
                "grid_codec_mechanically_validated": True,
                "finalizer_executed_dynamics": False,
                "finalizer_generated_classification": False,
                "upstream_dynamics_status": "NOT_ATTESTED_BY_THIS_MECHANICAL_FINALIZER",
                "ten_myr_execution_authorized": False,
                "planet_nine_evidence": False,
                "science_result": False,
            },
        }
        result["content_sha256"] = _domain_hash(FINAL_RESULT_DOMAIN, result)
        _write_create_only_at(directory_fd, "RESULT.json", _canonical_json(result))
        os.fsync(directory_fd)
        return result
    finally:
        for fd in output_fds.values():
            os.close(fd)
        os.close(directory_fd)


def validate_finalized_v1(
    output_root: Path,
    *,
    geometry: GridGeometry = PRODUCTION_GEOMETRY,
    derive: DeriveCallable,
    prefix_crossbind: PrefixCrossbindCallable = validate_first_1000_year_prefix_v1,
    test_only_geometry: bool = False,
) -> dict[str, Any]:
    if geometry != PRODUCTION_GEOMETRY and not test_only_geometry:
        raise GridAdapterError("nonproduction final validation requires test-only boundary")
    _require_prefix_crossbind(
        prefix_crossbind, geometry, test_only_geometry=test_only_geometry
    )
    if geometry == PRODUCTION_GEOMETRY:
        _require_production_derive(derive)
    directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    held: dict[str, _HeldFile] = {}
    try:
        if tuple(sorted(os.listdir(directory_fd))) != tuple(sorted(FINAL_ROSTER)):
            raise GridAdapterError("final output roster differs")
        for name in FINAL_ROSTER:
            held[name] = _HeldFile(directory_fd, name, name)
        manifest_payload = held["MANIFEST.json"].pread(held["MANIFEST.json"].size, 0)
        result_payload = held["RESULT.json"].pread(held["RESULT.json"].size, 0)
        manifest = _strict_json(manifest_payload)
        result = _strict_json(result_payload)
        _guard_integer_identities(manifest)
        _guard_integer_identities(result)
        if manifest_payload != _canonical_json(manifest) or result_payload != _canonical_json(result):
            raise GridAdapterError("final JSON is not canonical")
        if (
            set(manifest) != FINAL_MANIFEST_KEYS
            or set(result) != FINAL_RESULT_KEYS
            or
            manifest.get("schema") != FINAL_MANIFEST_SCHEMA
            or manifest.get("geometry") != _geometry_record(geometry)
            or manifest.get("layout")
            != {
                "state": "BATCH_NODE_BODY_COMPONENT_ROW_INTERLEAVED_F64LE",
                "elements": "BATCH_ENTITY_FIELD_NODE_F64LE",
                "node_digest": "BATCH_NODE_RAW_SHA256",
                "ml_long": "EXACT_B22_STRIDE_5_VIEW_NO_DUPLICATE_FILE",
            }
            or manifest.get("content_sha256") != _domain_hash(FINAL_MANIFEST_DOMAIN, manifest)
            or result.get("schema") != FINAL_RESULT_SCHEMA
            or result.get("status")
            != "GRID_ARTIFACTS_ASSEMBLED_AND_REPLAYED_BY_MECHANICAL_FINALIZER"
            or result.get("content_sha256") != _domain_hash(FINAL_RESULT_DOMAIN, result)
            or result.get("manifest_content_sha256") != manifest.get("content_sha256")
        ):
            raise GridAdapterError("final JSON identity differs")
        records = manifest.get("files")
        if type(records) is not list or [row.get("name") if type(row) is dict else None for row in records] != list(FINAL_BINARY_NAMES):
            raise GridAdapterError("final binary manifest differs")
        if any(set(row) != {"name", "byte_length", "sha256"} for row in records):
            raise GridAdapterError("final binary record schema differs")
        shard_manifests = manifest.get("shard_manifests_by_batch")
        batch_stream_roots = manifest.get("batch_stream_roots")
        if (
            type(shard_manifests) is not list
            or len(shard_manifests) != geometry.batch_count
            or any(
                type(batch) is not list or len(batch) != geometry.shard_count
                for batch in shard_manifests
            )
            or type(batch_stream_roots) is not list
            or len(batch_stream_roots) != geometry.batch_count
        ):
            raise GridAdapterError("final retained shard-chain roster differs")
        expected_sizes = geometry.output_sizes()
        for record in records:
            name = record["name"]
            _require_sha(record.get("sha256"), f"final {name} digest")
            if held[name].size != expected_sizes[name] or record.get("byte_length") != expected_sizes[name] or held[name].sha256() != record.get("sha256"):
                raise GridAdapterError("final binary identity differs")

        per_batch_sizes = {
            FINAL_BINARY_NAMES[0]: geometry.b22_node_count * geometry.state_record_bytes,
            FINAL_BINARY_NAMES[1]: (
                geometry.element_entity_count
                * geometry.element_field_count
                * geometry.b22_node_count
                * 8
            ),
            FINAL_BINARY_NAMES[2]: geometry.b22_node_count * 32,
            FINAL_BINARY_NAMES[3]: geometry.short_node_count * geometry.state_record_bytes,
        }
        for batch_order, retained_batch in enumerate(shard_manifests):
            rederived_prefix = _prefix_crossbinding_from_reader_v1(
                batch_order,
                geometry,
                prefix_crossbind,
                held[FINAL_BINARY_NAMES[3]].pread,
            )
            if any(
                item.get("first_1000_year_prefix_crossbinding_sha256")
                != rederived_prefix
                for item in retained_batch
                if type(item) is dict
            ) or any(type(item) is not dict for item in retained_batch):
                raise GridAdapterError("final first-1k crossbinding differs from short states")
            predecessor = ZERO_SHA256
            previous_final: str | None = None
            for shard_index, retained_shard in enumerate(retained_batch):
                _steps, b22_steps, short_steps = _validate_retained_shard_manifest_v1(
                    retained_shard,
                    geometry,
                    expected_batch_order=batch_order,
                    expected_shard_index=shard_index,
                )
                if retained_shard["predecessor_manifest_content_sha256"] != predecessor:
                    raise GridAdapterError("retained shard predecessor chain differs")
                if shard_index and retained_shard["start_continuation_sha256"] != previous_final:
                    raise GridAdapterError("retained shard continuation chain differs")
                predecessor = retained_shard["content_sha256"]
                previous_final = retained_shard["final_continuation_sha256"]
                records_by_name = {row["name"]: row for row in retained_shard["fragments"]}
                first_b22 = geometry.b22_index(b22_steps[0])
                if first_b22 is None:
                    raise GridAdapterError("retained shard has no B22 anchor")
                observed_fragment_hashes = {
                    FRAGMENT_NAMES[0]: _sha_held_range(
                        held[FINAL_BINARY_NAMES[0]],
                        (batch_order * geometry.b22_node_count + first_b22)
                        * geometry.state_record_bytes,
                        len(b22_steps) * geometry.state_record_bytes,
                    ),
                    FRAGMENT_NAMES[2]: _sha_held_range(
                        held[FINAL_BINARY_NAMES[2]],
                        (batch_order * geometry.b22_node_count + first_b22) * 32,
                        len(b22_steps) * 32,
                    ),
                }
                if short_steps:
                    first_short = geometry.short_index(short_steps[0])
                    if first_short is None:
                        raise GridAdapterError("retained shard short anchor differs")
                    observed_fragment_hashes[FRAGMENT_NAMES[3]] = _sha_held_range(
                        held[FINAL_BINARY_NAMES[3]],
                        (batch_order * geometry.short_node_count + first_short)
                        * geometry.state_record_bytes,
                        len(short_steps) * geometry.state_record_bytes,
                    )
                else:
                    observed_fragment_hashes[FRAGMENT_NAMES[3]] = hashlib.sha256(b"").hexdigest()
                element_columns: list[bytes] = []
                for value_index in range(
                    geometry.element_entity_count * geometry.element_field_count
                ):
                    destination_value_index = (
                        (batch_order * geometry.element_entity_count * geometry.element_field_count + value_index)
                        * geometry.b22_node_count
                        + first_b22
                    )
                    element_columns.append(
                        held[FINAL_BINARY_NAMES[1]].pread(
                            len(b22_steps) * 8, destination_value_index * 8
                        )
                    )
                element_digest = hashlib.sha256()
                for local_node in range(len(b22_steps)):
                    for column in element_columns:
                        element_digest.update(column[local_node * 8 : local_node * 8 + 8])
                observed_fragment_hashes[FRAGMENT_NAMES[1]] = element_digest.hexdigest()
                for name in FRAGMENT_NAMES:
                    if observed_fragment_hashes[name] != records_by_name[name]["sha256"]:
                        raise GridAdapterError(f"retained shard fragment/final crossbinding differs: {name}")

            range_hashes = {
                name: _sha_held_range(
                    held[name],
                    batch_order * per_batch_sizes[name],
                    per_batch_sizes[name],
                )
                for name in FINAL_BINARY_NAMES
            }
            expected_stream_root = _batch_stream_root_record_v1(
                batch_order, retained_batch, range_hashes
            )
            if batch_stream_roots[batch_order] != expected_stream_root:
                raise GridAdapterError("final batch stream root differs")

        for batch_order in range(geometry.batch_count):
            for node in range(geometry.b22_node_count):
                state_offset = (batch_order * geometry.b22_node_count + node) * geometry.state_record_bytes
                state_payload = held[FINAL_BINARY_NAMES[0]].pread(geometry.state_record_bytes, state_offset)
                digest, element_payload = derive(batch_order, state_payload)
                if digest != held[FINAL_BINARY_NAMES[2]].pread(32, (batch_order * geometry.b22_node_count + node) * 32):
                    raise GridAdapterError("final node digest replay differs")
                for entity in range(geometry.element_entity_count):
                    for field in range(geometry.element_field_count):
                        source_offset = (entity * geometry.element_field_count + field) * 8
                        destination_value_index = (
                            ((batch_order * geometry.element_entity_count + entity) * geometry.element_field_count + field)
                            * geometry.b22_node_count
                            + node
                        )
                        if element_payload[source_offset : source_offset + 8] != held[FINAL_BINARY_NAMES[1]].pread(8, destination_value_index * 8):
                            raise GridAdapterError("final element replay differs")
                global_step = node * geometry.b22_interval_steps
                short_index = geometry.short_index(global_step)
                if short_index is not None:
                    short_offset = (batch_order * geometry.short_node_count + short_index) * geometry.state_record_bytes
                    if state_payload != held[FINAL_BINARY_NAMES[3]].pread(geometry.state_record_bytes, short_offset):
                        raise GridAdapterError("final B22/short overlap differs")
            for short_node in range(geometry.short_node_count):
                short_offset = (
                    batch_order * geometry.short_node_count + short_node
                ) * geometry.state_record_bytes
                decode_state_record_v1(
                    held[FINAL_BINARY_NAMES[3]].pread(
                        geometry.state_record_bytes, short_offset
                    ),
                    geometry,
                )
        if not _exact_tree_equal(result.get("claims"), {
            "grid_codec_mechanically_validated": True,
            "finalizer_executed_dynamics": False,
            "finalizer_generated_classification": False,
            "upstream_dynamics_status": "NOT_ATTESTED_BY_THIS_MECHANICAL_FINALIZER",
            "ten_myr_execution_authorized": False,
            "planet_nine_evidence": False,
            "science_result": False,
        }):
            raise GridAdapterError("final result claim ceiling differs")
        return result
    finally:
        errors: list[Exception] = []
        for item in held.values():
            try:
                item.close()
            except Exception as exc:
                errors.append(exc)
        os.close(directory_fd)
        if errors:
            raise errors[0]


def adapter_contract_v1() -> dict[str, object]:
    """Return the frozen, non-executing WH/lifecycle composition."""

    return {
        "model": "DE441_REDUCED5_MASSLESS_9_TRACERS_PER_NATIVE_EPOCH_BATCH",
        "central_source_identity": {
            "pilot_v2_descriptive_label": "SUN_WITH_TERRESTRIAL_MASSES_FOLDED_IN",
            "stage1_and_classifier_runtime_body_id": "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
            "runtime_naif_id": 10,
            "runtime_gm_binary64_hex": REDUCED_GM_HEX[0],
            "policy": "DESCRIPTIVE_ALIAS_ONLY_RUNTIME_ID_AND_GM_MUST_MATCH_STAGE1_EXACTLY",
        },
        "massive_source_integrator": "WISDOM_HOLMAN_SOURCE_TILE_H64",
        "quiet_tracer_map": "CUDA_RESTRICTED_WH_FIXED_STEP",
        "encounter_lifecycle": "RESTRICTED_WH_LIFECYCLE_DOP853_ROUTE",
        "time_scale": "TDB",
        "frame": "ICRF_J2000",
        "axes": "J2000_EQUATORIAL",
        "source_step_julian_years_hex": H64_STEP_HEX,
        "source_steps_per_map_step": 1,
        "backend": {
            "backend_id": "cupy",
            "device": "cuda:0",
            "tile_size": 128,
            "dtype": "float64",
            "allow_fallback": False,
            "deterministic_reductions": True,
            "fast_math": False,
            "determinism_scope": "SAME_RUNTIME_DEVICE",
        },
        "source_build_phases": {
            "phase_0_prefix_gate": {
                "julian_years": 1_000,
                "source_steps": 64_000,
                "must_complete_before_any_later_source_build": True,
            },
            "phase_1_shard_zero_complement": {
                "julian_years": 9_000,
                "source_steps": 576_000,
                "requires_prefix_gate_pass": True,
            },
            "phase_1_regular_shards": {
                "count_per_batch": 999,
                "julian_years_each": 10_000,
                "source_steps_each": 640_000,
                "requires_prefix_gate_pass": True,
            },
            "source_build_count_per_batch": 1_001,
            "source_build_count_total": 12_012,
            "published_shard_count_per_batch": 1_000,
            "published_shard_count_total": 12_000,
        },
        "source_builder_step_ceiling": 1_000_000,
        "source_builder_spec": {
            "callable": "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile",
            "minimum_encounter_pair_separation_AU_hex": float(1.0).hex(),
            "minimum_jacobi_periapse_AU_hex": float(0.1).hex(),
            "maximum_initial_barycenter_position_norm_AU_hex": float(1.0e-12).hex(),
            "maximum_initial_barycenter_velocity_norm_AU_per_year_hex": float(1.0e-12).hex(),
            "source_tile_slice": (
                "EXACT_FROZEN_SOURCE_PATH_TILE_CONTIGUOUS_NODES_START_THROUGH_STOP_INCLUSIVE_"
                "WITH_GLOBAL_SOURCE_STEP_CHAINING"
            ),
        },
        "fixed_tracer_spec": {
            "maximum_primary_dynamical_phase_hex": float(0.05).hex(),
            "threads_per_block": 128,
            "primary_outer_boundary_AU_hex": float(4096.0).hex(),
        },
        "route_policy": {
            "source_encounter_radii_AU_hex": [float(1.0).hex()] * 5,
            "certified_relative_acceleration_bounds_AU_per_year2_hex": [
                REQUIRED_RELATIVE_ACCELERATION_BOUND_HEX
            ] * 5,
            "terminal_source_radii_AU_hex": [float(0.0).hex()] * 5,
            "primary_outer_boundary_AU_hex": float(4096.0).hex(),
            "route_worker_count": 1,
            "neptune_guard_routes_instead_of_one_AU_collision": True,
            "event_locator": {
                "relative_tolerance_hex": float(1.0e-12).hex(),
                "absolute_tolerance_hex": float(1.0e-14).hex(),
                "maximum_substep_fraction_hex": float(0.125).hex(),
                "dense_polynomial_validation_points": 17,
                "event_fraction_tolerance_hex": float(1.0e-12).hex(),
                "event_distance_tolerance_AU_hex": float(1.0e-11).hex(),
                "outward_radial_velocity_tolerance_AU_per_year_hex": float(1.0e-13).hex(),
                "maximum_accepted_steps": 100_000,
            },
        },
        "segment_schedule": {
            "through_500k_julian_years": {"years": 50, "h64_steps": 3_200},
            "after_500k_julian_years": {"years": 200, "h64_steps": 12_800},
            "segments_per_batch": 57_500,
            "segments_total": 690_000,
        },
        "checkpoint_frame": "INSTANTANEOUS_REDUCED5_GM_BARYCENTER_ICRF_J2000_EQUATORIAL",
        "first_1000_year_stop_gate": {
            "implemented_callable": "validate_first_1000_year_prefix_v1",
            "required": "BITWISE_EQUAL_AT_YEARS_0_50_100_..._1000_AFTER_LEGACY_LAYOUT_TRANSLATION",
            "fallback": "NO_FALLBACK_FROZEN_IN_THIS_SUCCESSOR",
            "failure_action": "STOP_BEFORE_CROSSING_YEAR_1000",
            "retained_h64_lane": "DE441_LANE_INDEX_12_PLUS_BATCH_ORDER",
        },
        "restart": {
            "shard_boundary_julian_years": 10_000,
            "fresh_process_suffix_year": 5_000_000,
            "selected_batch_count": 4,
            "selection_rule": "MUST_BE_EXTERNALLY_PREREGISTERED_BEFORE_EXECUTION",
            "bitwise_suffix_required": True,
        },
        "execution_surface_present": False,
        "scientifically_qualified": False,
    }


CLAIMS = {
    "writer_and_validator_implemented": True,
    "synthetic_mechanical_qualification_possible": True,
    "production_scale_writer_qualified": False,
    "reduced5_encounter_adapter_composition_frozen": True,
    "reduced5_encounter_adapter_executed": False,
    "reduced5_encounter_adapter_qualified": False,
    "first_1000_year_prefix_gate_implemented": True,
    "first_1000_year_prefix_crossbound": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "classification_generated": False,
    "planet_nine_evidence": False,
    "science_result": False,
}

BLOCKERS = (
    "PRODUCTION_SCALE_12000_SHARD_WRITER_PROFILE_NOT_EXECUTED",
    "REDUCED5_LIFECYCLE_ADAPTER_1KYR_PREFIX_NOT_EXECUTED_OR_CROSSBOUND",
    "CLASSIFIER_RUNTIME_AND_LIBM_CUSTODY_NOT_YET_CROSSBOUND_IN_WRITER_PROCESS",
    "TEN_MYR_ROUTE_INVARIANT_RUNTIME_AND_STORAGE_CAPS_NOT_QUALIFIED",
    "INDEPENDENT_CLASSIFIER_AND_ML_PORT_GATES_REMAIN_OPEN",
    "FUTURE_EXECUTION_NOT_EXTERNALLY_PREREGISTERED",
)


def build_protocol() -> dict[str, Any]:
    bindings = _binding_records()
    test_payload = _read_regular_bytes(TEST_PATH, "grid-adapter test")
    geometry = _geometry_record(PRODUCTION_GEOMETRY)
    sizes = PRODUCTION_GEOMETRY.output_sizes()
    if sum(sizes.values()) != 791_062_272:
        raise GridAdapterError("production fixed output size differs")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARE_ONLY_WRITER_IMPLEMENTED_ADAPTER_COMPOSITION_FROZEN_EXECUTION_BLOCKED",
        "objective": "Qualify exact multi-grid persistence and freeze the reduced-five WH encounter adapter without running 10 Myr dynamics.",
        "implementation": {
            **_implementation_identity(),
            "test_relative_path": str(TEST_PATH.relative_to(PROJECT_ROOT)),
            "test_byte_length": len(test_payload),
            "test_sha256": _sha256(test_payload),
            "cli_surface": "BUILD_PREPARE_VALIDATE_ONLY",
            "dynamics_execution_surface_present": False,
            "runtime_trust_boundary": "NO_HOSTILE_CODE_IN_THE_FRESH_WRITER_OR_VALIDATOR_PROCESS",
        },
        "lineage": {
            "pilot_v2_protocol_content_sha256": PILOT_V2_PROTOCOL_CONTENT_SHA256,
            "bound_files": bindings,
            "frozen_inputs_revalidated": True,
        },
        "production_geometry": geometry,
        "fixed_output_byte_lengths": sizes,
        "fixed_output_total_bytes": sum(sizes.values()),
        "sharding": {
            "shard_julian_years": 10_000,
            "shards_per_batch": 1_000,
            "total_shards": 12_000,
            "internal_restart_segment_julian_years": 1_000,
            "fragment_layout": {
                FRAGMENT_NAMES[0]: "NODE_MAJOR_ROW_INTERLEAVED_STATE",
                FRAGMENT_NAMES[1]: "NODE_MAJOR_ENTITY_FIELD_TRANSPOSED_ONLY_AT_FINALIZATION",
                FRAGMENT_NAMES[2]: "NODE_MAJOR_RAW_SHA256",
                FRAGMENT_NAMES[3]: "NODE_MAJOR_ROW_INTERLEAVED_STATE",
            },
            "publication": "IMMUTABLE_FRAGMENTS_THEN_MANIFEST_LAST_NO_AUTOMATIC_ROLLBACK",
            "chain": (
                "FULL_CANONICAL_SHARD_MANIFESTS_RETAINED_IN_FINAL_MANIFEST_AND_OFFLINE_"
                "REPLAYED_WITH_EXACT_PREDECESSOR_CONTINUATION_SOURCE_ROUTE_RUNTIME_AND_1KYR_BINDINGS"
            ),
        },
        "offline_replay": {
            "every_b22_node_count": 600_012,
            "every_b22_digest_rederived": True,
            "every_b22_element_row_rederived": True,
            "all_b22_short_overlap_nodes_bitwise_compared": 30_012,
            "every_ml_short_state_record_finite_and_shape_validated": True,
            "all_shard_fragment_hashes_reconstructed_from_final_binary_ranges": True,
            "per_batch_stream_roots_rederived": True,
            "ml_long": "EXACT_STRIDE_5_VIEW_NO_DUPLICATE_FILE",
            "held_fd_validation": True,
            "whole_file_materialization_required": False,
            "classifier_kernel_cache_policy": (
                "CALL_EXACT_V2_NODE_DIGEST_AND_ELEMENT_EXTRACTOR_BYTECODE_WITH_ONE_"
                "PREVALIDATED_EXACT_STAGE1_MODULE_CACHED_FOR_THE_HELD_VALIDATION_PASS"
            ),
            "engine_endpoint_conversion": (
                "PRIMARY_RELATIVE_ENDPOINTS_CANONICALIZED_BY_EXACT_BOUND_STAGE1_"
                "CHECKPOINT_THEN_SERIALIZED_ROW_INTERLEAVED"
            ),
        },
        "adapter": adapter_contract_v1(),
        "full_source_node_retention": {
            "per_source_node_bytes": 240,
            "nodes_per_batch": 640_000_001,
            "all_batches_bytes": 1_843_200_002_880,
            "retained": False,
            "replacement": "FIXED_WIDTH_SOURCE_CHUNK_RECEIPTS_PLUS_DETERMINISTIC_CHUNK_REPLAY",
        },
        "blockers": list(BLOCKERS),
        "claims": dict(CLAIMS),
        "outcomes_generated": False,
        "execution_authorized": False,
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise GridAdapterError("protocol must be an exact mapping")
    rebuilt = build_protocol()
    if _canonical_json(value) != _canonical_json(rebuilt):
        raise GridAdapterError("protocol differs from reconstruction")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise GridAdapterError("protocol content identity differs")
    return value["content_sha256"]


def prepare(root: Path = DEFAULT_PREPARED_ROOT) -> Path:
    protocol = build_protocol()
    if root.exists() or root.is_symlink():
        raise GridAdapterError("prepared root already exists")
    root.mkdir(mode=0o700, parents=False)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        _write_create_only_at(directory_fd, "PROTOCOL.json", _canonical_json(protocol))
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return root


def validate_prepared(root: Path = DEFAULT_PREPARED_ROOT) -> dict[str, Any]:
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.listdir(directory_fd) != ["PROTOCOL.json"]:
            raise GridAdapterError("prepared root roster differs")
        held = _HeldFile(directory_fd, "PROTOCOL.json", "prepared protocol")
        try:
            payload = held.pread(held.size, 0)
        finally:
            held.close()
    finally:
        os.close(directory_fd)
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise GridAdapterError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "prepare", "validate"))
    parser.add_argument("--root", type=Path, default=DEFAULT_PREPARED_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        sys.stdout.buffer.write(_canonical_json(build_protocol()))
    elif args.command == "prepare":
        prepare(args.root)
        sys.stdout.write(str(args.root) + "\n")
    else:
        sys.stdout.buffer.write(_canonical_json(validate_prepared(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
