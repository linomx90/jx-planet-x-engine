"""Verified, read-only access to the retained M2 fine source path.

This module does not propagate or interpolate a source state.  It only admits
the exact, completed M2 v2 evidence bundle, verifies its custody chain and all
retained arrays, and returns immutable contiguous views copied from the
81,921-node fine lattice.  Guard data is deliberately a separate overlay: a
change to a caller certification can never change the dynamic-node identity.

Nothing in this module is a scientific, production, tracer, encounter, or
long-duration qualification claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    FrozenSourcePathTile,
    verify_source_path_tile,
)
from jxplanetx.engine.contracts import Provenance
from jxplanetx.long_horizon.physical_models import (
    KNOWN_SOLAR_SYSTEM_V2,
    KNOWN_SYSTEM_BODY_IDS,
    KNOWN_SYSTEM_BODY_NAMES,
)


OFFICIAL_RUN_RELATIVE_PATH = Path("runs/long_horizon_m2_source_preflight_v2")
OFFICIAL_RUN_DIRECTORY_NAME = OFFICIAL_RUN_RELATIVE_PATH.name
RUN_ID = "jx-long-horizon-m2-known-source-10y-v2"

PROTOCOL_SCHEMA = "jx.long-horizon.m2-known-source-protocol.v2"
RESULT_SCHEMA = "jx.long-horizon.m2-known-source-result.v2"
ARTIFACT_SCHEMA = "jx.long-horizon.m2-known-source-artifacts.v2"
COMPLETION_SCHEMA = "jx.long-horizon.m2-known-source-completion.v2"

_PROTOCOL_DOMAIN = b"jx.long-horizon.m2-known-source.protocol.v2\0"
_RESULT_DOMAIN = b"jx.long-horizon.m2-known-source.result.v2\0"
_ARTIFACT_DOMAIN = b"jx.long-horizon.m2-known-source.artifacts.v2\0"
_COMPLETION_DOMAIN = b"jx.long-horizon.m2-known-source.completion.v2\0"
_STATE_DOMAIN = b"jx.long-horizon.m2-known-source.state.v2\0"
_SOURCE_SCHEDULE_DOMAIN = b"jx.long-horizon.m2-source-schedule.v2\0"
_DYNAMIC_DOMAIN = b"jxplanetx.long-horizon.m2-retained-dynamic-path.v1\0"
_PATH_DOMAIN = b"jxplanetx.long-horizon.m2-retained-path.v1\0"
_SLICE_DOMAIN = b"jxplanetx.long-horizon.m2-retained-slice.v1\0"
_OVERLAY_DOMAIN = b"jxplanetx.long-horizon.m2-guard-overlay.v1\0"
_TILE_LINEAGE_DOMAIN = b"jxplanetx.long-horizon.m2-frozen-tile-lineage.v1\0"
_TILE_BINDING_DOMAIN = b"jxplanetx.long-horizon.m2-frozen-tile-binding.v1\0"

PROTOCOL_FILE_SHA256 = (
    "8b1b152181e177653b86ac50ca698c05671d47b1590f8fca3b8d0e1b78d54b76"
)
PROTOCOL_CONTENT_SHA256 = (
    "ac499179478892a173e169d9dec818125c0f0e51f3e58c4b439f4e8a809930cc"
)
RESULT_FILE_SHA256 = (
    "ee9b7d5e727ae12d48757f3919fdf7d1acba49807caea730c4b6fd0c1a73b461"
)
RESULT_CONTENT_SHA256 = (
    "81674069f9f538007b9db07eded745886c6963f6671b7919fe273069edcfff31"
)
ARTIFACTS_FILE_SHA256 = (
    "a408aba26b53e548a877236504704ed4f91e6677afb9ac8bb61ae5ea9df7b902"
)
ARTIFACTS_CONTENT_SHA256 = (
    "38779d56fe688a9f39bbf9e0fb942071b42a748a4cf8a90c8212cfa4412cf120"
)
COMPLETED_FILE_SHA256 = (
    "081ac54da1a7915ea8d1d506ba6a312b716fb77fcdfd79bfcdc81c10b98f9c5a"
)
COMPLETED_CONTENT_SHA256 = (
    "c582877a2c314950ea543a57cc2155f0b5f5627a1128b454af20cc59badaf8ab"
)
STATES_FILE_SHA256 = (
    "49382bb15d99add32e83b30554154c5fd1f1ac32e94479ffb9c2851845388717"
)
EXECUTED_SOURCES_FILE_SHA256 = (
    "9e1d49c641170a935b641d4606aa8381b2085b2d013685f232065411e126e413"
)

KNOWN_SYSTEM_MODEL_SHA256 = (
    "19d33011f45d07d15a2aff8acf56a48c57e726828d8a34f1354f070987b86387"
)
KNOWN_SYSTEM_SOURCE_STATE_SHA256 = (
    "d545472608abc19360b5fd020918c48bcd8cb9fe95da4b8adc315ba8a1513e29"
)
KNOWN_SYSTEM_CONVERTED_STATE_SHA256 = (
    "3b6de264608aa07fca7281c28c2dfbccc66bd9bf7bcb84c098cb0b81decbdbee"
)

FINE_SOURCE_STEP_YEARS = float.fromhex("0x1.0000000000000p-13")
FINE_SOURCE_STEPS_PER_YEAR = 8192
FINE_SOURCE_STEP_COUNT = 81920
FINE_SOURCE_NODE_COUNT = FINE_SOURCE_STEP_COUNT + 1
FINE_SOURCE_SCHEDULE_SHA256 = (
    "5055aa75a3e4cfa322dded369cffc7390d6423ff13f5c574edf01f4647752485"
)
FULL_FINE_DYNAMIC_NODE_SHA256 = (
    "86293bb87aba3f3219aaa3d90263a07c271496f30ee0ee8a997428dfc2ce8487"
)
FULL_FINE_PATH_CONTENT_SHA256 = (
    "c332e9ff937f0aa7157a095e196d3a30adde5f26998b16a75c0740f3d836945d"
)
M2_GUARD_RADIUS_AU = 0.1
M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2 = float.fromhex(
    "0x1.ee1fac800e17ap+12"
)

SOURCE_BODY_IDS = KNOWN_SYSTEM_BODY_IDS
SOURCE_IDS = KNOWN_SYSTEM_BODY_NAMES
SOURCE_NAMES = KNOWN_SYSTEM_BODY_NAMES
PRIMARY_SOURCE_ID = SOURCE_IDS[0]
EPOCH_JD_TDB = 2461200.5
EPOCH_ORIGIN_YEARS = 0.0
TIME_SCALE = "TDB"
FRAME = "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000"
AXES = "IAU76_80_ECLIPTIC_J2000"
PHYSICAL_COORDINATE_SYSTEM = "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED"
LENGTH_UNIT = "AU"
TIME_UNIT = "year"
VELOCITY_UNIT = "AU_PER_JULIAN_YEAR_365.25_D"
GM_UNIT = "AU3_PER_JULIAN_YEAR_365.25_D2"
UNIT_SYSTEM_ID = "AU-YEAR"
JULIAN_YEAR_DAYS = 365.25

_EXPECTED_FILES = (
    "ARTIFACTS.json",
    "COMPLETED.json",
    "EXECUTED_SOURCES.zip",
    "PROTOCOL.json",
    "RESULT.json",
    "STATES.npz",
)
_FILE_SHA256 = MappingProxyType(
    {
        "ARTIFACTS.json": ARTIFACTS_FILE_SHA256,
        "COMPLETED.json": COMPLETED_FILE_SHA256,
        "EXECUTED_SOURCES.zip": EXECUTED_SOURCES_FILE_SHA256,
        "PROTOCOL.json": PROTOCOL_FILE_SHA256,
        "RESULT.json": RESULT_FILE_SHA256,
        "STATES.npz": STATES_FILE_SHA256,
    }
)
_FILE_SIZES = MappingProxyType(
    {
        "ARTIFACTS.json": 1370,
        "COMPLETED.json": 1463,
        "EXECUTED_SOURCES.zip": 5_387_779,
        "PROTOCOL.json": 45_279,
        "RESULT.json": 162_223,
        "STATES.npz": 48_335_826,
    }
)

_CLAIMS = MappingProxyType(
    {
        "complete_de441_force_model": False,
        "detection_claim": False,
        "ephemeris_reconstruction": False,
        "exclusion_claim": False,
        "four_point_five_gyr_accuracy_qualified": False,
        "inference_authorized": False,
        "long_duration_accuracy_qualified": False,
        "observationally_fitted": False,
        "planet_nine_modeled": False,
        "production_qualified": False,
        "science_result": False,
        "tracer_dynamics_qualified": False,
        "uncertainty_calibrated": False,
    }
)

_ARRAY_SHAPES = MappingProxyType(
    {
        "sample_years": (11,),
        "epochs_jd_tdb": (11,),
        "gm_AU3_per_year2": (10,),
        "jx_coarse_all_node_position_AU": (40_961, 10, 3),
        "jx_coarse_all_node_velocity_AU_per_year": (40_961, 10, 3),
        "jx_fine_all_node_position_AU": (81_921, 10, 3),
        "jx_fine_all_node_velocity_AU_per_year": (81_921, 10, 3),
        "jx_coarse_position_AU": (11, 10, 3),
        "jx_coarse_velocity_AU_per_year": (11, 10, 3),
        "jx_fine_position_AU": (11, 10, 3),
        "jx_fine_velocity_AU_per_year": (11, 10, 3),
        "jx_partition_position_AU": (11, 10, 3),
        "jx_partition_velocity_AU_per_year": (11, 10, 3),
        "dop853_barycentric_position_AU": (11, 10, 3),
        "dop853_barycentric_velocity_AU_per_year": (11, 10, 3),
        "ias15_barycentric_position_AU": (11, 10, 3),
        "ias15_barycentric_velocity_AU_per_year": (11, 10, 3),
        "de441_primary_relative_position_AU": (11, 10, 3),
        "de441_primary_relative_velocity_AU_per_year": (11, 10, 3),
    }
)
_ARRAY_CONTENT_SHA256 = MappingProxyType(
    {
        "de441_primary_relative_position_AU": "e1242b24889a58159fd59b8a9dae1b79241970e23dd9cca28cfea2fee406ce0c",
        "de441_primary_relative_velocity_AU_per_year": "5750f613777f08f3b7c2155ecaebe9a997c94dcf6733558e76252aee3aa3e59a",
        "dop853_barycentric_position_AU": "e2564b33ac4f79c216a8ed3cb0542cc3ac540ec7714976d943f26333545b4edd",
        "dop853_barycentric_velocity_AU_per_year": "b9f141dc83cad6e06ccd98d37e48574e12a56a68c4714a20f3d6db4d2cab1628",
        "epochs_jd_tdb": "cd973e491de8e583fb5909003e36ddb454a1b0d3c17e7671de090f964b5c452b",
        "gm_AU3_per_year2": "dadd81c51e3222feb272aa09a4368f79be97927189d93575ea23fcb07f320d95",
        "ias15_barycentric_position_AU": "42a5494d3370007ad2c148449b0dd37e2b0219faf793a21df82eaa6be2a0894a",
        "ias15_barycentric_velocity_AU_per_year": "82656ff380d998b5a83da6c34a9ef4b779f96a9786772ee988f7538098402194",
        "jx_coarse_all_node_position_AU": "b7037ef708cbe2afa6c6a20327f9d0090a3dd1502db1413be195b6d9e3db5b30",
        "jx_coarse_all_node_velocity_AU_per_year": "53771415b7ed84ab1bc325ea28710f7934844cc8b9004a0e24843ea6e6f9f096",
        "jx_coarse_position_AU": "9cabd923bbe83ab556c4993820b273490e3699547d0b3b44f674cfac7bc11aad",
        "jx_coarse_velocity_AU_per_year": "3c6e339ef8a70b9e59cd7d89d4b2dd8869e2fb1794d91fa59b246af44e1af757",
        "jx_fine_all_node_position_AU": "77dfee267cf6d56c0f251d7b81440b80af5a8c040d2b348c80448739efca4d55",
        "jx_fine_all_node_velocity_AU_per_year": "e223a95d5ffc8a5347054923fde2c510f569b0048c94714c25991c971d5fa2ae",
        "jx_fine_position_AU": "65f6af8a2dafcef29dd5e2a714a67c60190cca4b3d6d902414fc1249d6eb6e6f",
        "jx_fine_velocity_AU_per_year": "8bfe5133b4c520f2eb2a4bf0eabe41998e4f8f39d10e1e711808af0d6a61be2d",
        "jx_partition_position_AU": "79545bb656abf4c851147a33e77dade4c5197a8049537bc159a6c527ab4b1504",
        "jx_partition_velocity_AU_per_year": "d47befb5871fdeb7e9093bbb5f6c4963e1aa70c29a9fbdd1f57ea136d993dfb0",
        "sample_years": "5b2e37749965d21cf875f5d94bee401165dd2ddc4fdc771db2784423943cc35c",
    }
)
_NPZ_MEMBERS = tuple(f"{name}.npy" for name in _ARRAY_SHAPES)


class M2SourcePathVerificationError(RuntimeError):
    """The retained M2 path or a derived immutable binding failed closed."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise M2SourcePathVerificationError("record is not strict canonical JSON") from exc


def _record_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(_STATE_DOMAIN)
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise M2SourcePathVerificationError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _frozen_float64_array(
    value: object, label: str, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise M2SourcePathVerificationError(f"{label} must be an exact NumPy array")
    if value.dtype != np.dtype("<f8"):
        raise M2SourcePathVerificationError(f"{label} must have exact little-endian float64 dtype")
    if shape is not None and value.shape != shape:
        raise M2SourcePathVerificationError(
            f"{label} has shape {value.shape}; expected {shape}"
        )
    if not value.flags.c_contiguous:
        raise M2SourcePathVerificationError(f"{label} must be C-contiguous")
    if not bool(np.all(np.isfinite(value))):
        raise M2SourcePathVerificationError(f"{label} contains a nonfinite value")
    raw = value.tobytes(order="C")
    return np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(value.shape)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise M2SourcePathVerificationError("JSON object key is not a string")
        if key in result:
            raise M2SourcePathVerificationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise M2SourcePathVerificationError(f"JSON contains nonfinite constant {value}")


def _load_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except M2SourcePathVerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M2SourcePathVerificationError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise M2SourcePathVerificationError(f"{label} must contain one JSON object")
    return value


def _open_directory_no_symlinks(path: Path) -> int:
    """Open every path component relative to retained no-follow directory FDs."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise M2SourcePathVerificationError(
            "cannot open the retained path root without following a symlink"
        ) from exc
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise M2SourcePathVerificationError(
                    "cannot open retained path component without following "
                    f"a symlink: {component}"
                ) from exc
            os.close(descriptor)
            descriptor = child
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise M2SourcePathVerificationError(
                    f"retained path component is not a directory: {component}"
                )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    expected_size: int,
) -> bytes:
    if type(name) is not str or Path(name).name != name:
        raise M2SourcePathVerificationError("retained evidence filename is unsafe")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise M2SourcePathVerificationError(
            f"cannot open retained evidence file: {name}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M2SourcePathVerificationError(
                f"retained evidence entry is not a regular file: {name}"
            )
        if before.st_size != expected_size:
            raise M2SourcePathVerificationError(
                f"{name} size changed before reading"
            )
        chunks: list[bytes] = []
        remaining = expected_size + 1
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                raise M2SourcePathVerificationError(
                    f"{name} grew beyond its pinned size while reading"
                )
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise M2SourcePathVerificationError(
                f"retained evidence changed while reading: {name}"
            )
        payload = b"".join(chunks)
        if len(payload) != expected_size:
            raise M2SourcePathVerificationError(
                f"short read of retained evidence: {name}"
            )
        return payload
    finally:
        os.close(descriptor)


def _validate_content_record(
    value: dict[str, Any], *, schema: str, domain: bytes, expected_content_sha256: str, label: str
) -> None:
    if value.get("schema") != schema:
        raise M2SourcePathVerificationError(f"{label} schema is not the canonical v2 schema")
    if value.get("run_id") != RUN_ID:
        raise M2SourcePathVerificationError(f"{label} run_id is not the canonical v2 run")
    actual = value.get("content_sha256")
    if actual != expected_content_sha256:
        raise M2SourcePathVerificationError(f"{label} content_sha256 changed")
    unhashed = {key: item for key, item in value.items() if key != "content_sha256"}
    if _record_sha256(domain, unhashed) != actual:
        raise M2SourcePathVerificationError(f"{label} canonical content hash is invalid")


def _validate_claim_ceiling(value: object, label: str) -> None:
    if type(value) is not dict or value != dict(_CLAIMS):
        raise M2SourcePathVerificationError(f"{label} claim ceiling changed")
    if any(type(item) is not bool or item for item in value.values()):
        raise M2SourcePathVerificationError(f"{label} contains a widened claim")


def _validate_evidence_records(
    protocol: dict[str, Any],
    result: dict[str, Any],
    artifacts: dict[str, Any],
    completed: dict[str, Any],
) -> None:
    _validate_content_record(
        protocol,
        schema=PROTOCOL_SCHEMA,
        domain=_PROTOCOL_DOMAIN,
        expected_content_sha256=PROTOCOL_CONTENT_SHA256,
        label="PROTOCOL.json",
    )
    _validate_content_record(
        result,
        schema=RESULT_SCHEMA,
        domain=_RESULT_DOMAIN,
        expected_content_sha256=RESULT_CONTENT_SHA256,
        label="RESULT.json",
    )
    _validate_content_record(
        artifacts,
        schema=ARTIFACT_SCHEMA,
        domain=_ARTIFACT_DOMAIN,
        expected_content_sha256=ARTIFACTS_CONTENT_SHA256,
        label="ARTIFACTS.json",
    )
    _validate_content_record(
        completed,
        schema=COMPLETION_SCHEMA,
        domain=_COMPLETION_DOMAIN,
        expected_content_sha256=COMPLETED_CONTENT_SHA256,
        label="COMPLETED.json",
    )
    for label, record in (
        ("protocol", protocol),
        ("result", result),
        ("artifacts", artifacts),
        ("completion", completed),
    ):
        _validate_claim_ceiling(record.get("claims"), label)
    if protocol.get("status") != "PREDECLARED_BEFORE_EXECUTION":
        raise M2SourcePathVerificationError("protocol predeclaration status changed")
    if result.get("status") != "PASSED" or result.get("all_gates_passed") is not True:
        raise M2SourcePathVerificationError("M2 result is not the exact passing result")
    gates = result.get("gates")
    if type(gates) is not dict or not gates or any(
        type(item) is not dict or item.get("passed") is not True for item in gates.values()
    ):
        raise M2SourcePathVerificationError("M2 result does not retain an all-pass gate set")

    model = protocol.get("model")
    expected_model = {
        "body_ids": list(SOURCE_BODY_IDS),
        "body_names": list(SOURCE_NAMES),
        "epoch_jd_tdb_hex": EPOCH_JD_TDB.hex(),
        "known_system_model_id": KNOWN_SOLAR_SYSTEM_V2.model_id,
        "known_system_schema": "jx.long-horizon.known-solar-system-source-model.v2",
    }
    if type(model) is not dict or any(model.get(key) != item for key, item in expected_model.items()):
        raise M2SourcePathVerificationError("protocol source roster or physical model changed")
    inputs = protocol.get("inputs")
    if type(inputs) is not dict or (
        inputs.get("known_system_model_sha256") != KNOWN_SYSTEM_MODEL_SHA256
        or inputs.get("known_system_source_state_sha256") != KNOWN_SYSTEM_SOURCE_STATE_SHA256
        or inputs.get("known_system_converted_state_sha256") != KNOWN_SYSTEM_CONVERTED_STATE_SHA256
    ):
        raise M2SourcePathVerificationError("protocol known-system identity changed")
    profile = protocol.get("profile")
    lanes = profile.get("jx_lanes") if type(profile) is dict else None
    if type(lanes) is not list or len(lanes) != 2:
        raise M2SourcePathVerificationError("protocol JX lane roster changed")
    fine = lanes[1]
    if type(fine) is not dict or fine != {
        "exact_replay_required": True,
        "name": "h_over_2",
        "schedule_sha256": FINE_SOURCE_SCHEDULE_SHA256,
        "step_count": FINE_SOURCE_STEP_COUNT,
        "step_years_hex": FINE_SOURCE_STEP_YEARS.hex(),
    }:
        raise M2SourcePathVerificationError("protocol fine source schedule changed")

    if (
        result.get("protocol_content_sha256") != PROTOCOL_CONTENT_SHA256
        or result.get("protocol_file_sha256") != PROTOCOL_FILE_SHA256
    ):
        raise M2SourcePathVerificationError("result-to-protocol cross-link changed")
    state_record = result.get("state_arrays")
    if type(state_record) is not dict or (
        state_record.get("path") != "STATES.npz"
        or state_record.get("sha256") != STATES_FILE_SHA256
        or state_record.get("array_content_sha256") != dict(_ARRAY_CONTENT_SHA256)
    ):
        raise M2SourcePathVerificationError("result retained-state identity changed")

    expected_artifact_entries = [
        {
            "path": name,
            "sha256": _FILE_SHA256[name],
            "size_bytes": _FILE_SIZES[name],
        }
        for name in ("EXECUTED_SOURCES.zip", "PROTOCOL.json", "RESULT.json", "STATES.npz")
    ]
    if artifacts.get("file_count") != 4 or artifacts.get("files") != expected_artifact_entries:
        raise M2SourcePathVerificationError("artifact inventory changed")
    expected_completion_links = {
        "protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "result_content_sha256": RESULT_CONTENT_SHA256,
        "result_file_sha256": RESULT_FILE_SHA256,
        "artifact_manifest_content_sha256": ARTIFACTS_CONTENT_SHA256,
        "artifact_manifest_file_sha256": ARTIFACTS_FILE_SHA256,
        "result_status": "PASSED",
        "all_gates_passed": True,
    }
    if any(completed.get(key) != item for key, item in expected_completion_links.items()):
        raise M2SourcePathVerificationError("completion custody cross-link changed")


def _validate_npz_member_roster(names: tuple[str, ...]) -> None:
    if len(names) != len(set(names)):
        raise M2SourcePathVerificationError("STATES.npz contains duplicate members")
    if set(names) != set(_NPZ_MEMBERS) or len(names) != len(_NPZ_MEMBERS):
        raise M2SourcePathVerificationError("STATES.npz member roster changed")
    for name in names:
        path = Path(name)
        if path.name != name or path.suffix != ".npy" or name.startswith(("/", "\\")):
            raise M2SourcePathVerificationError("STATES.npz contains an unsafe member path")


def _validate_array_contract(name: str, array: object) -> np.ndarray:
    expected_shape = _ARRAY_SHAPES.get(name)
    if expected_shape is None:
        raise M2SourcePathVerificationError(f"unexpected retained array {name!r}")
    if type(array) is not np.ndarray:
        raise M2SourcePathVerificationError(f"retained array {name!r} is not an exact NumPy array")
    if array.dtype != np.dtype("<f8"):
        raise M2SourcePathVerificationError(f"retained array {name!r} has the wrong dtype")
    if array.shape != expected_shape:
        raise M2SourcePathVerificationError(f"retained array {name!r} has the wrong shape")
    if not array.flags.c_contiguous:
        raise M2SourcePathVerificationError(f"retained array {name!r} is not C-contiguous")
    if not bool(np.all(np.isfinite(array))):
        raise M2SourcePathVerificationError(f"retained array {name!r} contains nonfinite data")
    if _array_content_sha256(array) != _ARRAY_CONTENT_SHA256[name]:
        raise M2SourcePathVerificationError(f"retained array {name!r} content hash changed")
    return array


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )
    )


def _validate_state_boundaries(arrays: Mapping[str, np.ndarray]) -> None:
    sample = arrays["sample_years"]
    epochs = arrays["epochs_jd_tdb"]
    expected_sample = np.arange(11, dtype=np.float64)
    expected_epochs = np.asarray(
        [EPOCH_JD_TDB + year * JULIAN_YEAR_DAYS for year in range(11)],
        dtype=np.float64,
    )
    if not _bitwise_equal(sample, expected_sample) or not _bitwise_equal(epochs, expected_epochs):
        raise M2SourcePathVerificationError("retained annual epoch lattice changed")
    for lane, stride in (("coarse", 4096), ("fine", 8192)):
        for suffix in ("position_AU", "velocity_AU_per_year"):
            nodes = arrays[f"jx_{lane}_all_node_{suffix}"]
            annual = arrays[f"jx_{lane}_{suffix}"]
            if not _bitwise_equal(np.ascontiguousarray(nodes[::stride]), annual):
                raise M2SourcePathVerificationError(f"{lane} annual boundary array mismatch")

    known_position = np.asarray(
        KNOWN_SOLAR_SYSTEM_V2.positions_primary_relative_ecliptic_AU,
        dtype=np.dtype("<f8"),
    )
    known_velocity = np.asarray(
        KNOWN_SOLAR_SYSTEM_V2.velocities_primary_relative_ecliptic_AU_per_julian_year,
        dtype=np.dtype("<f8"),
    )
    known_gm = np.asarray(
        KNOWN_SOLAR_SYSTEM_V2.gm_AU3_per_julian_year2,
        dtype=np.dtype("<f8"),
    )
    if (
        not _bitwise_equal(arrays["jx_fine_all_node_position_AU"][0], known_position)
        or not _bitwise_equal(arrays["jx_fine_all_node_velocity_AU_per_year"][0], known_velocity)
        or not _bitwise_equal(arrays["gm_AU3_per_year2"], known_gm)
    ):
        raise M2SourcePathVerificationError("fine source genesis boundary changed")
    primary_position = arrays["jx_fine_all_node_position_AU"][:, 0, :]
    primary_velocity = arrays["jx_fine_all_node_velocity_AU_per_year"][:, 0, :]
    if bool(np.any(primary_position.view(np.uint64) != 0)) or bool(
        np.any(primary_velocity.view(np.uint64) != 0)
    ):
        raise M2SourcePathVerificationError("fine source primary is not exact positive zero")


def _load_and_verify_state_arrays(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            _validate_npz_member_roster(names)
            if any(
                info.flag_bits & 0x1
                or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
                for info in infos
            ):
                raise M2SourcePathVerificationError("STATES.npz contains a linked or encrypted member")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != tuple(_ARRAY_SHAPES):
                raise M2SourcePathVerificationError("STATES.npz logical array roster changed")
            arrays = {
                name: _validate_array_contract(name, archive[name])
                for name in _ARRAY_SHAPES
            }
    except M2SourcePathVerificationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise M2SourcePathVerificationError("STATES.npz cannot be decoded safely") from exc
    _validate_state_boundaries(arrays)
    return arrays


def _dynamic_record(
    *, start_step: int, final_step: int, positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> dict[str, Any]:
    return {
        "schema": "jxplanetx.long-horizon.m2-retained-dynamic-path.v1",
        "run_id": RUN_ID,
        "known_system_model_sha256": KNOWN_SYSTEM_MODEL_SHA256,
        "source_body_ids": list(SOURCE_BODY_IDS),
        "source_ids": list(SOURCE_IDS),
        "source_names": list(SOURCE_NAMES),
        "primary_source_id": PRIMARY_SOURCE_ID,
        "epoch_jd_tdb_hex": EPOCH_JD_TDB.hex(),
        "epoch_origin_years_hex": EPOCH_ORIGIN_YEARS.hex(),
        "time_scale": TIME_SCALE,
        "frame": FRAME,
        "axes": AXES,
        "physical_coordinate_system": PHYSICAL_COORDINATE_SYSTEM,
        "engine_coordinate_system": COORDINATE_SYSTEM,
        "length_unit": LENGTH_UNIT,
        "time_unit": TIME_UNIT,
        "velocity_unit": VELOCITY_UNIT,
        "gm_unit": GM_UNIT,
        "unit_system_id": UNIT_SYSTEM_ID,
        "source_step_years_hex": FINE_SOURCE_STEP_YEARS.hex(),
        "source_schedule_sha256": FINE_SOURCE_SCHEDULE_SHA256,
        "start_step": start_step,
        "final_step": final_step,
        "node_count": final_step - start_step + 1,
        "position_array_content_sha256": _array_content_sha256(positions),
        "velocity_array_content_sha256": _array_content_sha256(velocities),
        "gm_array_content_sha256": _array_content_sha256(gm),
    }


def _dynamic_sha256(
    *, start_step: int, final_step: int, positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> str:
    return _record_sha256(
        _DYNAMIC_DOMAIN,
        _dynamic_record(
            start_step=start_step,
            final_step=final_step,
            positions=positions,
            velocities=velocities,
            gm=gm,
        ),
    )


def _path_content_sha256(dynamic_node_sha256: str) -> str:
    return _record_sha256(
        _PATH_DOMAIN,
        {
            "schema": "jxplanetx.long-horizon.m2-retained-path.v1",
            "dynamic_node_sha256": dynamic_node_sha256,
            "parent_file_sha256": dict(_FILE_SHA256),
            "parent_content_sha256": {
                "ARTIFACTS.json": ARTIFACTS_CONTENT_SHA256,
                "COMPLETED.json": COMPLETED_CONTENT_SHA256,
                "PROTOCOL.json": PROTOCOL_CONTENT_SHA256,
                "RESULT.json": RESULT_CONTENT_SHA256,
            },
            "array_content_sha256": dict(_ARRAY_CONTENT_SHA256),
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class VerifiedM2SourcePath:
    """Immutable exact M2 fine path admitted from the official evidence bundle."""

    run_directory: Path
    source_relative_positions_AU: np.ndarray
    source_relative_velocities_AU_per_year: np.ndarray
    source_gm_AU3_per_year2: np.ndarray
    dynamic_node_sha256: str
    content_sha256: str
    parent_file_sha256: Mapping[str, str]
    parent_content_sha256: Mapping[str, str]
    array_content_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        positions = _frozen_float64_array(
            self.source_relative_positions_AU,
            "source_relative_positions_AU",
            shape=(FINE_SOURCE_NODE_COUNT, len(SOURCE_IDS), 3),
        )
        velocities = _frozen_float64_array(
            self.source_relative_velocities_AU_per_year,
            "source_relative_velocities_AU_per_year",
            shape=positions.shape,
        )
        gm = _frozen_float64_array(
            self.source_gm_AU3_per_year2,
            "source_gm_AU3_per_year2",
            shape=(len(SOURCE_IDS),),
        )
        object.__setattr__(self, "source_relative_positions_AU", positions)
        object.__setattr__(self, "source_relative_velocities_AU_per_year", velocities)
        object.__setattr__(self, "source_gm_AU3_per_year2", gm)
        object.__setattr__(self, "parent_file_sha256", MappingProxyType(dict(self.parent_file_sha256)))
        object.__setattr__(self, "parent_content_sha256", MappingProxyType(dict(self.parent_content_sha256)))
        object.__setattr__(self, "array_content_sha256", MappingProxyType(dict(self.array_content_sha256)))
        verify_m2_source_path(self)

    @property
    def source_body_ids(self) -> tuple[int, ...]:
        return SOURCE_BODY_IDS

    @property
    def source_ids(self) -> tuple[str, ...]:
        return SOURCE_IDS

    @property
    def source_names(self) -> tuple[str, ...]:
        return SOURCE_NAMES

    @property
    def primary_source_id(self) -> str:
        return PRIMARY_SOURCE_ID

    @property
    def node_count(self) -> int:
        return FINE_SOURCE_NODE_COUNT

    @property
    def source_step_years(self) -> float:
        return FINE_SOURCE_STEP_YEARS

    @property
    def source_schedule_sha256(self) -> str:
        return FINE_SOURCE_SCHEDULE_SHA256

    @property
    def epoch_jd_tdb(self) -> float:
        return EPOCH_JD_TDB

    @property
    def epoch_origin_years(self) -> float:
        return EPOCH_ORIGIN_YEARS

    @property
    def time_scale(self) -> str:
        return TIME_SCALE

    @property
    def frame(self) -> str:
        return FRAME

    @property
    def axes(self) -> str:
        return AXES

    @property
    def coordinate_system(self) -> str:
        return COORDINATE_SYSTEM

    @property
    def length_unit(self) -> str:
        return LENGTH_UNIT

    @property
    def time_unit(self) -> str:
        return TIME_UNIT

    @property
    def unit_system_id(self) -> str:
        return UNIT_SYSTEM_ID

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            source_id="jx.m2.retained-fine-source-path.v2",
            citation="JX retained M2 v2 ten-Julian-year Newtonian known-source path",
            version="v2",
            sha256=self.content_sha256,
        )

    def slice_steps(self, start_step: int, final_step: int) -> "VerifiedM2SourcePathSlice":
        return _slice_verified_path(self, start_step, final_step)

    def slice_years(self, start_year: int, final_year: int) -> "VerifiedM2SourcePathSlice":
        if type(start_year) is not int or type(final_year) is not int:
            raise M2SourcePathVerificationError("year boundaries must be exact integers")
        return self.slice_steps(
            start_year * FINE_SOURCE_STEPS_PER_YEAR,
            final_year * FINE_SOURCE_STEPS_PER_YEAR,
        )


# Strong references deliberately keep the exact fully-hashed array objects
# alive for the duration of the process.  The retained path is a singleton-sized
# qualification asset, so this bounded registry avoids both id reuse and a
# per-map 39 MiB re-hash.  No execution slice ever reads a replaceable field
# without first matching it by object identity to this registry.
_VERIFIED_PATH_WITNESSES: dict[
    int,
    tuple[
        VerifiedM2SourcePath,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
] = {}


@dataclass(frozen=True, slots=True, eq=False)
class VerifiedM2SourcePathSlice:
    """One immutable inclusive interval on the retained integer source lattice."""

    parent: VerifiedM2SourcePath
    global_source_step: int
    final_global_source_step: int
    source_relative_positions_AU: np.ndarray
    source_relative_velocities_AU_per_year: np.ndarray
    source_gm_AU3_per_year2: np.ndarray
    position_array_content_sha256: str
    velocity_array_content_sha256: str
    gm_array_content_sha256: str
    dynamic_node_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if type(self.parent) is not VerifiedM2SourcePath:
            raise M2SourcePathVerificationError("slice parent must be an exact verified M2 path")
        if (
            type(self.global_source_step) is not int
            or type(self.final_global_source_step) is not int
            or self.global_source_step < 0
            or self.final_global_source_step <= self.global_source_step
            or self.final_global_source_step > FINE_SOURCE_STEP_COUNT
        ):
            raise M2SourcePathVerificationError("slice boundaries are not a valid inclusive interval")
        shape = (self.node_count, len(SOURCE_IDS), 3)
        positions = _frozen_float64_array(
            self.source_relative_positions_AU, "slice source positions", shape=shape
        )
        velocities = _frozen_float64_array(
            self.source_relative_velocities_AU_per_year, "slice source velocities", shape=shape
        )
        gm = _frozen_float64_array(
            self.source_gm_AU3_per_year2, "slice source GM", shape=(len(SOURCE_IDS),)
        )
        object.__setattr__(self, "source_relative_positions_AU", positions)
        object.__setattr__(self, "source_relative_velocities_AU_per_year", velocities)
        object.__setattr__(self, "source_gm_AU3_per_year2", gm)
        verify_m2_source_path_slice(self)

    @property
    def node_count(self) -> int:
        return self.final_global_source_step - self.global_source_step + 1

    @property
    def initial_epoch_years(self) -> float:
        return EPOCH_ORIGIN_YEARS + self.global_source_step * FINE_SOURCE_STEP_YEARS

    @property
    def final_epoch_years(self) -> float:
        return EPOCH_ORIGIN_YEARS + self.final_global_source_step * FINE_SOURCE_STEP_YEARS

    @property
    def source_ids(self) -> tuple[str, ...]:
        return SOURCE_IDS

    @property
    def primary_source_id(self) -> str:
        return PRIMARY_SOURCE_ID

    @property
    def source_step_years(self) -> float:
        return FINE_SOURCE_STEP_YEARS

    @property
    def source_schedule_sha256(self) -> str:
        return FINE_SOURCE_SCHEDULE_SHA256

    @property
    def epoch_origin_years(self) -> float:
        return EPOCH_ORIGIN_YEARS

    @property
    def initial_epoch_jd_tdb(self) -> float:
        return EPOCH_JD_TDB + self.initial_epoch_years * JULIAN_YEAR_DAYS

    @property
    def final_epoch_jd_tdb(self) -> float:
        return EPOCH_JD_TDB + self.final_epoch_years * JULIAN_YEAR_DAYS

    @property
    def time_scale(self) -> str:
        return TIME_SCALE

    @property
    def frame(self) -> str:
        return FRAME

    @property
    def axes(self) -> str:
        return AXES

    @property
    def coordinate_system(self) -> str:
        return COORDINATE_SYSTEM

    @property
    def length_unit(self) -> str:
        return LENGTH_UNIT

    @property
    def time_unit(self) -> str:
        return TIME_UNIT

    @property
    def unit_system_id(self) -> str:
        return UNIT_SYSTEM_ID

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            source_id="jx.m2.retained-fine-source-path.slice.v2",
            citation="Exact contiguous slice of the verified retained JX M2 v2 fine path",
            version="v2",
            sha256=self.content_sha256,
        )

    def bind_frozen_source_path_tile(
        self, overlay: "M2GuardOverlay"
    ) -> "M2FrozenSourcePathTileBinding":
        return build_m2_frozen_source_path_tile(self, overlay)

    def to_frozen_source_path_tile(self, overlay: "M2GuardOverlay") -> FrozenSourcePathTile:
        """Return the exact engine tile; use ``bind_*`` to retain its side receipt."""

        return self.bind_frozen_source_path_tile(overlay).tile


def verify_m2_source_path(source: VerifiedM2SourcePath) -> str:
    if type(source) is not VerifiedM2SourcePath:
        raise M2SourcePathVerificationError("source must be an exact VerifiedM2SourcePath")
    if Path(source.run_directory).name != OFFICIAL_RUN_DIRECTORY_NAME:
        raise M2SourcePathVerificationError("verified path no longer names the canonical v2 run")
    if dict(source.parent_file_sha256) != dict(_FILE_SHA256):
        raise M2SourcePathVerificationError("verified path parent file hashes changed")
    expected_parent_content = {
        "ARTIFACTS.json": ARTIFACTS_CONTENT_SHA256,
        "COMPLETED.json": COMPLETED_CONTENT_SHA256,
        "PROTOCOL.json": PROTOCOL_CONTENT_SHA256,
        "RESULT.json": RESULT_CONTENT_SHA256,
    }
    if dict(source.parent_content_sha256) != expected_parent_content:
        raise M2SourcePathVerificationError("verified path parent content hashes changed")
    if dict(source.array_content_sha256) != dict(_ARRAY_CONTENT_SHA256):
        raise M2SourcePathVerificationError("verified path retained array hashes changed")
    if _array_content_sha256(source.source_relative_positions_AU) != _ARRAY_CONTENT_SHA256[
        "jx_fine_all_node_position_AU"
    ]:
        raise M2SourcePathVerificationError("verified fine positions changed")
    if _array_content_sha256(source.source_relative_velocities_AU_per_year) != _ARRAY_CONTENT_SHA256[
        "jx_fine_all_node_velocity_AU_per_year"
    ]:
        raise M2SourcePathVerificationError("verified fine velocities changed")
    if _array_content_sha256(source.source_gm_AU3_per_year2) != _ARRAY_CONTENT_SHA256[
        "gm_AU3_per_year2"
    ]:
        raise M2SourcePathVerificationError("verified source GM changed")
    dynamic = _dynamic_sha256(
        start_step=0,
        final_step=FINE_SOURCE_STEP_COUNT,
        positions=source.source_relative_positions_AU,
        velocities=source.source_relative_velocities_AU_per_year,
        gm=source.source_gm_AU3_per_year2,
    )
    if dynamic != FULL_FINE_DYNAMIC_NODE_SHA256 or source.dynamic_node_sha256 != dynamic:
        raise M2SourcePathVerificationError("verified path dynamic-node hash changed")
    expected_content = _path_content_sha256(dynamic)
    if (
        expected_content != FULL_FINE_PATH_CONTENT_SHA256
        or source.content_sha256 != expected_content
    ):
        raise M2SourcePathVerificationError("verified path content hash changed")
    _VERIFIED_PATH_WITNESSES[id(source)] = (
        source,
        source.source_relative_positions_AU,
        source.source_relative_velocities_AU_per_year,
        source.source_gm_AU3_per_year2,
    )
    return expected_content


def _verify_m2_source_path_witness(
    source: VerifiedM2SourcePath,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check the immutable in-process witness without re-hashing 39 MiB.

    ``VerifiedM2SourcePath`` copies every admitted array into read-only,
    bytes-backed storage and performs the complete content verification in its
    constructor.  Re-hashing the whole ten-year parent for every 8- or 16-node
    execution slice would repeat that proof thousands of times.  This helper
    checks the sealed object identity and immutable storage invariants; formal
    runs still call :func:`verify_m2_source_path` before and after execution.
    """

    if type(source) is not VerifiedM2SourcePath:
        raise M2SourcePathVerificationError(
            "source must be an exact VerifiedM2SourcePath"
        )
    if (
        Path(source.run_directory).name != OFFICIAL_RUN_DIRECTORY_NAME
        or source.dynamic_node_sha256 != FULL_FINE_DYNAMIC_NODE_SHA256
        or source.content_sha256 != FULL_FINE_PATH_CONTENT_SHA256
        or dict(source.parent_file_sha256) != dict(_FILE_SHA256)
        or dict(source.parent_content_sha256)
        != {
            "ARTIFACTS.json": ARTIFACTS_CONTENT_SHA256,
            "COMPLETED.json": COMPLETED_CONTENT_SHA256,
            "PROTOCOL.json": PROTOCOL_CONTENT_SHA256,
            "RESULT.json": RESULT_CONTENT_SHA256,
        }
        or dict(source.array_content_sha256) != dict(_ARRAY_CONTENT_SHA256)
    ):
        raise M2SourcePathVerificationError(
            "verified M2 parent witness identity changed"
        )
    witness = _VERIFIED_PATH_WITNESSES.get(id(source))
    if (
        witness is None
        or witness[0] is not source
        or source.source_relative_positions_AU is not witness[1]
        or source.source_relative_velocities_AU_per_year is not witness[2]
        or source.source_gm_AU3_per_year2 is not witness[3]
    ):
        raise M2SourcePathVerificationError(
            "verified M2 parent arrays differ from the fully hashed session witness"
        )
    for array, shape, label in (
        (
            witness[1],
            (FINE_SOURCE_NODE_COUNT, len(SOURCE_IDS), 3),
            "positions",
        ),
        (
            witness[2],
            (FINE_SOURCE_NODE_COUNT, len(SOURCE_IDS), 3),
            "velocities",
        ),
        (witness[3], (len(SOURCE_IDS),), "GM"),
    ):
        if (
            type(array) is not np.ndarray
            or array.dtype != np.dtype("<f8")
            or array.shape != shape
            or not array.flags.c_contiguous
            or array.flags.writeable
        ):
            raise M2SourcePathVerificationError(
                f"verified M2 parent {label} storage is no longer sealed"
            )
    return witness[1], witness[2], witness[3]


def _slice_content_sha256(source_slice: VerifiedM2SourcePathSlice, dynamic: str) -> str:
    return _record_sha256(
        _SLICE_DOMAIN,
        {
            "schema": "jxplanetx.long-horizon.m2-retained-slice.v1",
            "parent_path_content_sha256": source_slice.parent.content_sha256,
            "parent_dynamic_node_sha256": source_slice.parent.dynamic_node_sha256,
            "parent_protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
            "parent_result_content_sha256": RESULT_CONTENT_SHA256,
            "parent_states_file_sha256": STATES_FILE_SHA256,
            "global_source_step": source_slice.global_source_step,
            "final_global_source_step": source_slice.final_global_source_step,
            "dynamic_node_sha256": dynamic,
        },
    )


def verify_m2_source_path_slice(source_slice: VerifiedM2SourcePathSlice) -> str:
    if type(source_slice) is not VerifiedM2SourcePathSlice:
        raise M2SourcePathVerificationError("source slice must be an exact verified M2 slice")
    parent_positions, parent_velocities, parent_gm = (
        _verify_m2_source_path_witness(source_slice.parent)
    )
    start = source_slice.global_source_step
    final = source_slice.final_global_source_step
    expected_position = np.ascontiguousarray(
        parent_positions[start : final + 1]
    )
    expected_velocity = np.ascontiguousarray(
        parent_velocities[start : final + 1]
    )
    if not _bitwise_equal(source_slice.source_relative_positions_AU, expected_position) or not _bitwise_equal(
        source_slice.source_relative_velocities_AU_per_year, expected_velocity
    ):
        raise M2SourcePathVerificationError("slice is not an exact contiguous parent interval")
    if not _bitwise_equal(source_slice.source_gm_AU3_per_year2, parent_gm):
        raise M2SourcePathVerificationError("slice source GM differs from its parent")
    position_hash = _array_content_sha256(source_slice.source_relative_positions_AU)
    velocity_hash = _array_content_sha256(source_slice.source_relative_velocities_AU_per_year)
    gm_hash = _array_content_sha256(source_slice.source_gm_AU3_per_year2)
    if (
        source_slice.position_array_content_sha256 != position_hash
        or source_slice.velocity_array_content_sha256 != velocity_hash
        or source_slice.gm_array_content_sha256 != gm_hash
    ):
        raise M2SourcePathVerificationError("slice node-array hash changed")
    dynamic = _dynamic_sha256(
        start_step=start,
        final_step=final,
        positions=source_slice.source_relative_positions_AU,
        velocities=source_slice.source_relative_velocities_AU_per_year,
        gm=source_slice.source_gm_AU3_per_year2,
    )
    if source_slice.dynamic_node_sha256 != dynamic:
        raise M2SourcePathVerificationError("slice dynamic-node hash changed")
    content = _slice_content_sha256(source_slice, dynamic)
    if source_slice.content_sha256 != content:
        raise M2SourcePathVerificationError("slice content hash changed")
    return content


def _slice_verified_path(
    source: VerifiedM2SourcePath, start_step: int, final_step: int
) -> VerifiedM2SourcePathSlice:
    parent_positions, parent_velocities, parent_gm = (
        _verify_m2_source_path_witness(source)
    )
    if (
        type(start_step) is not int
        or type(final_step) is not int
        or start_step < 0
        or final_step <= start_step
        or final_step > FINE_SOURCE_STEP_COUNT
    ):
        raise M2SourcePathVerificationError(
            "slice steps must be an increasing inclusive interval inside [0, 81920]"
        )
    positions = np.ascontiguousarray(parent_positions[start_step : final_step + 1])
    velocities = np.ascontiguousarray(
        parent_velocities[start_step : final_step + 1]
    )
    gm = np.ascontiguousarray(parent_gm)
    position_hash = _array_content_sha256(positions)
    velocity_hash = _array_content_sha256(velocities)
    gm_hash = _array_content_sha256(gm)
    dynamic = _dynamic_sha256(
        start_step=start_step,
        final_step=final_step,
        positions=positions,
        velocities=velocities,
        gm=gm,
    )
    provisional = object.__new__(VerifiedM2SourcePathSlice)
    object.__setattr__(provisional, "parent", source)
    object.__setattr__(provisional, "global_source_step", start_step)
    object.__setattr__(provisional, "final_global_source_step", final_step)
    object.__setattr__(provisional, "source_relative_positions_AU", positions)
    object.__setattr__(provisional, "source_relative_velocities_AU_per_year", velocities)
    object.__setattr__(provisional, "source_gm_AU3_per_year2", gm)
    object.__setattr__(provisional, "position_array_content_sha256", position_hash)
    object.__setattr__(provisional, "velocity_array_content_sha256", velocity_hash)
    object.__setattr__(provisional, "gm_array_content_sha256", gm_hash)
    object.__setattr__(provisional, "dynamic_node_sha256", dynamic)
    content = _slice_content_sha256(provisional, dynamic)
    return VerifiedM2SourcePathSlice(
        parent=source,
        global_source_step=start_step,
        final_global_source_step=final_step,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        source_gm_AU3_per_year2=gm,
        position_array_content_sha256=position_hash,
        velocity_array_content_sha256=velocity_hash,
        gm_array_content_sha256=gm_hash,
        dynamic_node_sha256=dynamic,
        content_sha256=content,
    )


def load_verified_m2_source_path(run_directory: str | os.PathLike[str]) -> VerifiedM2SourcePath:
    """Load only the exact official retained M2 v2 source-path bundle."""

    if isinstance(run_directory, bool) or not isinstance(run_directory, (str, os.PathLike)):
        raise M2SourcePathVerificationError("run_directory must be a filesystem path")
    root = Path(run_directory)
    if root.name != OFFICIAL_RUN_DIRECTORY_NAME:
        raise M2SourcePathVerificationError(
            f"only {OFFICIAL_RUN_DIRECTORY_NAME!r} is an admissible retained run path"
        )
    if (
        KNOWN_SOLAR_SYSTEM_V2.content_sha256 != KNOWN_SYSTEM_MODEL_SHA256
        or KNOWN_SOLAR_SYSTEM_V2.source_barycentric_state_sha256
        != KNOWN_SYSTEM_SOURCE_STATE_SHA256
        or KNOWN_SOLAR_SYSTEM_V2.converted_state_sha256
        != KNOWN_SYSTEM_CONVERTED_STATE_SHA256
    ):
        raise M2SourcePathVerificationError("installed known-system v2 identity changed")
    derived_schedule = hashlib.sha256(
        _SOURCE_SCHEDULE_DOMAIN
        + KNOWN_SYSTEM_MODEL_SHA256.encode("ascii")
        + b"\0"
        + FINE_SOURCE_STEP_YEARS.hex().encode("ascii")
        + b"\0"
        + b"10"
    ).hexdigest()
    if derived_schedule != FINE_SOURCE_SCHEDULE_SHA256:
        raise M2SourcePathVerificationError("fine source schedule derivation changed")
    root_descriptor = _open_directory_no_symlinks(root)
    try:
        root_before = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_before.st_mode):
            raise M2SourcePathVerificationError(
                "retained M2 run path is not a directory"
            )
        try:
            names = tuple(sorted(os.listdir(root_descriptor)))
        except OSError as exc:
            raise M2SourcePathVerificationError(
                "cannot enumerate retained M2 run"
            ) from exc
        if names != _EXPECTED_FILES:
            raise M2SourcePathVerificationError(
                f"retained M2 file roster changed: {names}"
            )
        payloads = {
            name: _read_regular_file_at(
                root_descriptor,
                name,
                expected_size=_FILE_SIZES[name],
            )
            for name in _EXPECTED_FILES
        }
        names_after = tuple(sorted(os.listdir(root_descriptor)))
        root_after = os.fstat(root_descriptor)
        if (
            names_after != names
            or _directory_identity(root_after) != _directory_identity(root_before)
        ):
            raise M2SourcePathVerificationError(
                "retained M2 directory changed while reading"
            )
    finally:
        os.close(root_descriptor)

    # The retained FD made the read itself safe.  Reopen component-by-component
    # to ensure the caller's path still names the same directory after the read.
    confirmation_descriptor = _open_directory_no_symlinks(root)
    try:
        if _directory_identity(os.fstat(confirmation_descriptor)) != (
            _directory_identity(root_after)
        ):
            raise M2SourcePathVerificationError(
                "retained M2 path identity changed while reading"
            )
    finally:
        os.close(confirmation_descriptor)

    for name, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        if digest != _FILE_SHA256[name]:
            raise M2SourcePathVerificationError(f"{name} file SHA-256 changed")

    protocol = _load_json(payloads["PROTOCOL.json"], "PROTOCOL.json")
    result = _load_json(payloads["RESULT.json"], "RESULT.json")
    artifacts = _load_json(payloads["ARTIFACTS.json"], "ARTIFACTS.json")
    completed = _load_json(payloads["COMPLETED.json"], "COMPLETED.json")
    _validate_evidence_records(protocol, result, artifacts, completed)
    arrays = _load_and_verify_state_arrays(payloads["STATES.npz"])
    positions = np.ascontiguousarray(arrays["jx_fine_all_node_position_AU"])
    velocities = np.ascontiguousarray(arrays["jx_fine_all_node_velocity_AU_per_year"])
    gm = np.ascontiguousarray(arrays["gm_AU3_per_year2"])
    dynamic = _dynamic_sha256(
        start_step=0,
        final_step=FINE_SOURCE_STEP_COUNT,
        positions=positions,
        velocities=velocities,
        gm=gm,
    )
    parent_content = {
        "ARTIFACTS.json": ARTIFACTS_CONTENT_SHA256,
        "COMPLETED.json": COMPLETED_CONTENT_SHA256,
        "PROTOCOL.json": PROTOCOL_CONTENT_SHA256,
        "RESULT.json": RESULT_CONTENT_SHA256,
    }
    return VerifiedM2SourcePath(
        run_directory=root,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        source_gm_AU3_per_year2=gm,
        dynamic_node_sha256=dynamic,
        content_sha256=_path_content_sha256(dynamic),
        parent_file_sha256=_FILE_SHA256,
        parent_content_sha256=parent_content,
        array_content_sha256=_ARRAY_CONTENT_SHA256,
    )


def _pinned_m2_uniform_relative_acceleration_bound() -> float:
    gm = np.asarray(
        KNOWN_SOLAR_SYSTEM_V2.gm_AU3_per_julian_year2,
        dtype=np.dtype("<f8"),
    )
    if _array_content_sha256(gm) != _ARRAY_CONTENT_SHA256["gm_AU3_per_year2"]:
        raise M2SourcePathVerificationError(
            "installed known-system GM bytes differ from the retained M2 path"
        )
    total = sum((Fraction.from_float(float(item)) for item in gm), Fraction())
    radius = Fraction.from_float(M2_GUARD_RADIUS_AU)
    exact = 2 * total / (radius * radius)
    bound = float(exact)
    if Fraction.from_float(bound) < exact:
        bound = math.nextafter(bound, math.inf)
    if (
        not math.isfinite(bound)
        or bound <= 0.0
        or bound.hex()
        != M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2.hex()
    ):
        raise M2SourcePathVerificationError(
            "pinned M2 acceleration-bound derivation changed"
        )
    return bound


def derive_m2_uniform_relative_acceleration_bound(
    source: VerifiedM2SourcePath | VerifiedM2SourcePathSlice,
    *,
    encounter_radius_AU: float = M2_GUARD_RADIUS_AU,
) -> float:
    """Derive the upward-rounded M2 swept-guard bound; this is not integration."""

    if type(source) is VerifiedM2SourcePath:
        verify_m2_source_path(source)
        gm = source.source_gm_AU3_per_year2
    elif type(source) is VerifiedM2SourcePathSlice:
        verify_m2_source_path_slice(source)
        gm = source.source_gm_AU3_per_year2
    else:
        raise M2SourcePathVerificationError("guard source must be a verified M2 path or slice")
    if type(encounter_radius_AU) is not float or not math.isfinite(encounter_radius_AU):
        raise M2SourcePathVerificationError("encounter radius must be a finite built-in float")
    if encounter_radius_AU.hex() != M2_GUARD_RADIUS_AU.hex():
        raise M2SourcePathVerificationError("the M2 guard overlay requires exact 0.1-AU radii")
    if _array_content_sha256(gm) != _ARRAY_CONTENT_SHA256["gm_AU3_per_year2"]:
        raise M2SourcePathVerificationError(
            "verified path GM bytes differ from the retained M2 identity"
        )
    return _pinned_m2_uniform_relative_acceleration_bound()


@dataclass(frozen=True, slots=True, eq=False)
class M2GuardOverlay:
    """Sealed caller certification kept separate from retained dynamics."""

    source_encounter_radii_AU: np.ndarray
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray
    gm_array_content_sha256: str
    minimum_derived_uniform_bound_AU_per_year2: float
    content_sha256: str

    def __post_init__(self) -> None:
        radii = _frozen_float64_array(
            self.source_encounter_radii_AU,
            "source_encounter_radii_AU",
            shape=(len(SOURCE_IDS),),
        )
        bounds = _frozen_float64_array(
            self.certified_relative_acceleration_bounds_AU_per_year2,
            "certified_relative_acceleration_bounds_AU_per_year2",
            shape=(len(SOURCE_IDS),),
        )
        object.__setattr__(self, "source_encounter_radii_AU", radii)
        object.__setattr__(self, "certified_relative_acceleration_bounds_AU_per_year2", bounds)
        _require_sha256(self.gm_array_content_sha256, "gm_array_content_sha256")
        if self.gm_array_content_sha256 != _ARRAY_CONTENT_SHA256["gm_AU3_per_year2"]:
            raise M2SourcePathVerificationError(
                "guard overlay GM hash is not the retained M2 GM identity"
            )
        expected_radii = np.full(len(SOURCE_IDS), M2_GUARD_RADIUS_AU, dtype=np.float64)
        if not _bitwise_equal(radii, expected_radii):
            raise M2SourcePathVerificationError("guard overlay radii must all be exact 0.1 AU")
        expected_minimum = _pinned_m2_uniform_relative_acceleration_bound()
        if (
            type(self.minimum_derived_uniform_bound_AU_per_year2) is not float
            or not math.isfinite(self.minimum_derived_uniform_bound_AU_per_year2)
            or self.minimum_derived_uniform_bound_AU_per_year2.hex()
            != expected_minimum.hex()
            or not bool(np.all(bounds > 0.0))
            or bool(np.any(bounds < expected_minimum))
        ):
            raise M2SourcePathVerificationError("guard overlay undercuts the derived uniform bound")
        actual = _guard_overlay_sha256(
            radii,
            bounds,
            self.gm_array_content_sha256,
            self.minimum_derived_uniform_bound_AU_per_year2,
        )
        if self.content_sha256 != actual:
            raise M2SourcePathVerificationError("guard overlay content hash changed")


def _guard_overlay_sha256(
    radii: np.ndarray, bounds: np.ndarray, gm_hash: str, minimum_bound: float
) -> str:
    return _record_sha256(
        _OVERLAY_DOMAIN,
        {
            "schema": "jxplanetx.long-horizon.m2-guard-overlay.v1",
            "known_system_model_sha256": KNOWN_SYSTEM_MODEL_SHA256,
            "gm_array_content_sha256": gm_hash,
            "source_ids": list(SOURCE_IDS),
            "radius_policy": "EVERY_SOURCE_EXACT_0P1_AU",
            "bound_derivation": "UPWARD_ROUNDED_2_SUM_GM_OVER_MINIMUM_RADIUS_SQUARED",
            "minimum_derived_uniform_bound_AU_per_year2_hex": minimum_bound.hex(),
            "source_encounter_radii_array_content_sha256": _array_content_sha256(radii),
            "certified_relative_acceleration_bounds_array_content_sha256": _array_content_sha256(bounds),
        },
    )


def seal_m2_guard_overlay(
    source: VerifiedM2SourcePath | VerifiedM2SourcePathSlice,
    *,
    source_encounter_radii_AU: np.ndarray,
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray,
) -> M2GuardOverlay:
    """Seal a caller-supplied 0.1-AU overlay after checking its lower bound."""

    if type(source) is VerifiedM2SourcePath:
        verify_m2_source_path(source)
        gm = source.source_gm_AU3_per_year2
    elif type(source) is VerifiedM2SourcePathSlice:
        verify_m2_source_path_slice(source)
        gm = source.source_gm_AU3_per_year2
    else:
        raise M2SourcePathVerificationError("guard source must be a verified M2 path or slice")
    radii = _frozen_float64_array(
        source_encounter_radii_AU,
        "source_encounter_radii_AU",
        shape=(len(SOURCE_IDS),),
    )
    bounds = _frozen_float64_array(
        certified_relative_acceleration_bounds_AU_per_year2,
        "certified_relative_acceleration_bounds_AU_per_year2",
        shape=(len(SOURCE_IDS),),
    )
    minimum = derive_m2_uniform_relative_acceleration_bound(source)
    gm_hash = _array_content_sha256(gm)
    digest = _guard_overlay_sha256(radii, bounds, gm_hash, minimum)
    return M2GuardOverlay(
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=bounds,
        gm_array_content_sha256=gm_hash,
        minimum_derived_uniform_bound_AU_per_year2=minimum,
        content_sha256=digest,
    )


def verify_m2_guard_overlay(overlay: M2GuardOverlay) -> str:
    if type(overlay) is not M2GuardOverlay:
        raise M2SourcePathVerificationError(
            "overlay must be an exact sealed M2GuardOverlay"
        )
    checked = M2GuardOverlay(
        source_encounter_radii_AU=overlay.source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            overlay.certified_relative_acceleration_bounds_AU_per_year2
        ),
        gm_array_content_sha256=overlay.gm_array_content_sha256,
        minimum_derived_uniform_bound_AU_per_year2=(
            overlay.minimum_derived_uniform_bound_AU_per_year2
        ),
        content_sha256=overlay.content_sha256,
    )
    return checked.content_sha256


def _tile_lineage_sha256(
    source_slice: VerifiedM2SourcePathSlice,
    overlay: M2GuardOverlay,
) -> str:
    return _record_sha256(
        _TILE_LINEAGE_DOMAIN,
        {
            "schema": "jxplanetx.long-horizon.m2-frozen-tile-lineage.v1",
            "slice_content_sha256": source_slice.content_sha256,
            "dynamic_node_sha256": source_slice.dynamic_node_sha256,
            "guard_overlay_sha256": overlay.content_sha256,
            "parent_protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
            "parent_result_content_sha256": RESULT_CONTENT_SHA256,
            "parent_states_file_sha256": STATES_FILE_SHA256,
        },
    )


_TILE_PROVENANCE_SOURCE_ID = "jx.m2.retained-fine-source-path.v2"
_TILE_PROVENANCE_CITATION = (
    "JX retained M2 v2 ten-Julian-year Newtonian known-source path; "
    "verified custody and caller guard overlay"
)


@dataclass(frozen=True, slots=True, eq=False)
class M2FrozenSourcePathTileBinding:
    """Exact engine tile plus separate dynamic, guard, lineage, and tile hashes."""

    source_slice: VerifiedM2SourcePathSlice
    guard_overlay: M2GuardOverlay
    tile: FrozenSourcePathTile
    dynamic_node_sha256: str
    guard_overlay_sha256: str
    lineage_content_sha256: str
    tile_content_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_slice) is not VerifiedM2SourcePathSlice:
            raise M2SourcePathVerificationError(
                "binding source_slice must be an exact verified M2 slice"
            )
        verify_m2_source_path_slice(self.source_slice)
        verify_m2_guard_overlay(self.guard_overlay)
        if type(self.tile) is not FrozenSourcePathTile:
            raise M2SourcePathVerificationError("binding tile must be an exact FrozenSourcePathTile")
        for value, label in (
            (self.dynamic_node_sha256, "dynamic_node_sha256"),
            (self.guard_overlay_sha256, "guard_overlay_sha256"),
            (self.lineage_content_sha256, "lineage_content_sha256"),
            (self.tile_content_sha256, "tile_content_sha256"),
            (self.content_sha256, "content_sha256"),
        ):
            _require_sha256(value, label)
        expected_lineage = _tile_lineage_sha256(
            self.source_slice, self.guard_overlay
        )
        expected_tile_id = (
            f"m2-v2-{self.source_slice.global_source_step}-"
            f"{self.source_slice.final_global_source_step}-"
            f"{self.source_slice.dynamic_node_sha256[:16]}-"
            f"{self.guard_overlay.content_sha256[:16]}"
        )
        context_matches = (
            self.tile.tile_id == expected_tile_id
            and self.tile.epoch_origin_years.hex() == EPOCH_ORIGIN_YEARS.hex()
            and self.tile.global_source_step
            == self.source_slice.global_source_step
            and self.tile.final_global_source_step
            == self.source_slice.final_global_source_step
            and self.tile.source_step_years.hex()
            == FINE_SOURCE_STEP_YEARS.hex()
            and self.tile.time_scale == TIME_SCALE
            and self.tile.frame == FRAME
            and self.tile.axes == AXES
            and self.tile.coordinate_system == COORDINATE_SYSTEM
            and self.tile.length_unit == LENGTH_UNIT
            and self.tile.time_unit == TIME_UNIT
            and self.tile.unit_system_id == UNIT_SYSTEM_ID
            and self.tile.source_ids == SOURCE_IDS
            and self.tile.primary_source_id == PRIMARY_SOURCE_ID
            and self.tile.source_schedule_sha256
            == FINE_SOURCE_SCHEDULE_SHA256
        )
        if not context_matches:
            raise M2SourcePathVerificationError(
                "binding tile context differs from its verified slice"
            )
        if (
            not _bitwise_equal(
                self.tile.source_relative_positions_AU,
                self.source_slice.source_relative_positions_AU,
            )
            or not _bitwise_equal(
                self.tile.source_relative_velocities_AU_per_year,
                self.source_slice.source_relative_velocities_AU_per_year,
            )
            or not _bitwise_equal(
                self.tile.source_gm_AU3_per_year2,
                self.source_slice.source_gm_AU3_per_year2,
            )
            or not _bitwise_equal(
                self.tile.source_encounter_radii_AU,
                self.guard_overlay.source_encounter_radii_AU,
            )
            or not _bitwise_equal(
                self.tile.certified_relative_acceleration_bounds_AU_per_year2,
                self.guard_overlay.certified_relative_acceleration_bounds_AU_per_year2,
            )
        ):
            raise M2SourcePathVerificationError(
                "binding tile arrays differ from their verified witnesses"
            )
        provenance = self.tile.provenance
        if (
            provenance.source_id != _TILE_PROVENANCE_SOURCE_ID
            or provenance.citation != _TILE_PROVENANCE_CITATION
            or provenance.version != "v2"
            or provenance.sha256 != expected_lineage
        ):
            raise M2SourcePathVerificationError(
                "binding tile provenance differs from its verified lineage"
            )
        if (
            self.dynamic_node_sha256 != self.source_slice.dynamic_node_sha256
            or self.guard_overlay_sha256 != self.guard_overlay.content_sha256
            or self.lineage_content_sha256 != expected_lineage
        ):
            raise M2SourcePathVerificationError(
                "binding receipt fields differ from their verified witnesses"
            )
        if verify_source_path_tile(self.tile) != self.tile_content_sha256:
            raise M2SourcePathVerificationError("binding tile content hash changed")
        actual = _record_sha256(
            _TILE_BINDING_DOMAIN,
            {
                "schema": "jxplanetx.long-horizon.m2-frozen-tile-binding.v1",
                "dynamic_node_sha256": self.dynamic_node_sha256,
                "guard_overlay_sha256": self.guard_overlay_sha256,
                "lineage_content_sha256": self.lineage_content_sha256,
                "tile_content_sha256": self.tile_content_sha256,
            },
        )
        if self.content_sha256 != actual:
            raise M2SourcePathVerificationError("frozen tile binding content hash changed")


def verify_m2_frozen_source_path_tile_binding(
    binding: M2FrozenSourcePathTileBinding,
) -> str:
    if type(binding) is not M2FrozenSourcePathTileBinding:
        raise M2SourcePathVerificationError(
            "binding must be an exact M2FrozenSourcePathTileBinding"
        )
    binding.__post_init__()
    return binding.content_sha256


def build_m2_frozen_source_path_tile(
    source_slice: VerifiedM2SourcePathSlice,
    overlay: M2GuardOverlay,
) -> M2FrozenSourcePathTileBinding:
    """Bind a verified slice and sealed guard overlay without integrating sources."""

    # The exact immutable witnesses are checked once by the binding
    # constructor below.  Avoid a duplicate hash pass over the same slice.
    if type(source_slice) is not VerifiedM2SourcePathSlice:
        raise M2SourcePathVerificationError(
            "source slice must be an exact verified M2 slice"
        )
    if type(overlay) is not M2GuardOverlay:
        raise M2SourcePathVerificationError(
            "overlay must be an exact sealed M2GuardOverlay"
        )
    if overlay.gm_array_content_sha256 != source_slice.gm_array_content_sha256:
        raise M2SourcePathVerificationError("guard overlay GM lineage differs from the slice")
    lineage = _tile_lineage_sha256(source_slice, overlay)
    tile = FrozenSourcePathTile(
        tile_id=(
            f"m2-v2-{source_slice.global_source_step}-{source_slice.final_global_source_step}-"
            f"{source_slice.dynamic_node_sha256[:16]}-{overlay.content_sha256[:16]}"
        ),
        epoch_origin_years=EPOCH_ORIGIN_YEARS,
        global_source_step=source_slice.global_source_step,
        source_step_years=FINE_SOURCE_STEP_YEARS,
        time_scale=TIME_SCALE,
        frame=FRAME,
        axes=AXES,
        coordinate_system=COORDINATE_SYSTEM,
        length_unit=LENGTH_UNIT,
        time_unit=TIME_UNIT,
        unit_system_id=UNIT_SYSTEM_ID,
        source_ids=SOURCE_IDS,
        primary_source_id=PRIMARY_SOURCE_ID,
        source_schedule_sha256=FINE_SOURCE_SCHEDULE_SHA256,
        source_relative_positions_AU=source_slice.source_relative_positions_AU,
        source_relative_velocities_AU_per_year=source_slice.source_relative_velocities_AU_per_year,
        source_gm_AU3_per_year2=source_slice.source_gm_AU3_per_year2,
        source_encounter_radii_AU=overlay.source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            overlay.certified_relative_acceleration_bounds_AU_per_year2
        ),
        provenance=Provenance(
            source_id=_TILE_PROVENANCE_SOURCE_ID,
            citation=_TILE_PROVENANCE_CITATION,
            version="v2",
            sha256=lineage,
        ),
    )
    tile_digest = verify_source_path_tile(tile)
    binding_digest = _record_sha256(
        _TILE_BINDING_DOMAIN,
        {
            "schema": "jxplanetx.long-horizon.m2-frozen-tile-binding.v1",
            "dynamic_node_sha256": source_slice.dynamic_node_sha256,
            "guard_overlay_sha256": overlay.content_sha256,
            "lineage_content_sha256": lineage,
            "tile_content_sha256": tile_digest,
        },
    )
    return M2FrozenSourcePathTileBinding(
        source_slice=source_slice,
        guard_overlay=overlay,
        tile=tile,
        dynamic_node_sha256=source_slice.dynamic_node_sha256,
        guard_overlay_sha256=overlay.content_sha256,
        lineage_content_sha256=lineage,
        tile_content_sha256=tile_digest,
        content_sha256=binding_digest,
    )


__all__ = [
    "ARTIFACTS_CONTENT_SHA256",
    "ARTIFACTS_FILE_SHA256",
    "AXES",
    "COMPLETED_CONTENT_SHA256",
    "COMPLETED_FILE_SHA256",
    "EPOCH_JD_TDB",
    "EPOCH_ORIGIN_YEARS",
    "EXECUTED_SOURCES_FILE_SHA256",
    "FINE_SOURCE_NODE_COUNT",
    "FINE_SOURCE_SCHEDULE_SHA256",
    "FINE_SOURCE_STEP_COUNT",
    "FINE_SOURCE_STEP_YEARS",
    "FINE_SOURCE_STEPS_PER_YEAR",
    "FRAME",
    "FULL_FINE_DYNAMIC_NODE_SHA256",
    "FULL_FINE_PATH_CONTENT_SHA256",
    "KNOWN_SYSTEM_MODEL_SHA256",
    "M2FrozenSourcePathTileBinding",
    "M2GuardOverlay",
    "M2SourcePathVerificationError",
    "M2_DERIVED_ACCELERATION_BOUND_AU_PER_YEAR2",
    "M2_GUARD_RADIUS_AU",
    "OFFICIAL_RUN_DIRECTORY_NAME",
    "OFFICIAL_RUN_RELATIVE_PATH",
    "PRIMARY_SOURCE_ID",
    "PROTOCOL_CONTENT_SHA256",
    "PROTOCOL_FILE_SHA256",
    "RESULT_CONTENT_SHA256",
    "RESULT_FILE_SHA256",
    "RUN_ID",
    "SOURCE_BODY_IDS",
    "SOURCE_IDS",
    "SOURCE_NAMES",
    "STATES_FILE_SHA256",
    "TIME_SCALE",
    "UNIT_SYSTEM_ID",
    "VerifiedM2SourcePath",
    "VerifiedM2SourcePathSlice",
    "build_m2_frozen_source_path_tile",
    "derive_m2_uniform_relative_acceleration_bound",
    "load_verified_m2_source_path",
    "seal_m2_guard_overlay",
    "verify_m2_frozen_source_path_tile_binding",
    "verify_m2_guard_overlay",
    "verify_m2_source_path",
    "verify_m2_source_path_slice",
]
