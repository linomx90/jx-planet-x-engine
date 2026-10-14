#!/usr/bin/env python3
"""Prepare the bounded reduced-five H64/H128 source preflight V1.

This artifact is deliberately prepare-only.  It freezes a 1,000-Julian-year
autonomous massive-source comparison at the common epoch JD 2461200.5 TDB:

* JX's compiled ordered-Jacobi Wisdom--Holman map at h=1/64 and h=1/128;
* a complete same-runtime bitwise replay of each JX lane;
* annual endpoint-inclusive comparison checkpoints and all-node invariants;
* an independently implemented, retained REBOUND 5.1.1 IAS15 reference; and
* the exact numerical thresholds sealed by the 3-Gyr preregistration.

No dynamics command is exposed here.  A successor implementation may execute
only after this exact protocol has been committed and annotated-tagged.  Even
a passing successor result would qualify only the autonomous reduced-five
massive source; it cannot qualify any tracer, covariance bridge, external
forcing model, long-horizon run, or scientific conclusion.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_h128_source_preflight_v1.py"
)
TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_tno_1kyr_reduced5_h128_source_preflight_v1.py"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_h128_source_preflight_v1"
)
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})

PROTOCOL_SCHEMA = "jx.tno.1kyr-reduced5-h128-source-preflight.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.1kyr-reduced5-h128-source-preflight.protocol.v1\0"
INITIAL_STATE_SCHEMA = "jx.tno.common-epoch-reduced5-source-state.v1"
INITIAL_STATE_DOMAIN = b"jx.tno.common-epoch-reduced5-source-state.v1\0"
SOURCE_SCHEDULE_SCHEMA = "jx.tno.1kyr-reduced5-source-schedule.v1"
SOURCE_SCHEDULE_DOMAIN = b"jx.tno.1kyr-reduced5-source-schedule.v1\0"

PARENT_PROTOCOL_PATH = (
    PROJECT_ROOT / "runs" / "tno_3gyr_full129_robustness_v1" / "PROTOCOL.json"
)
PARENT_PROTOCOL_BYTES = 37_244
PARENT_PROTOCOL_SHA256 = (
    "d17f99472ed4580ce18dbb7af1ec4248052c98197beec365ea5e3fd400a0f7a9"
)
PARENT_PROTOCOL_CONTENT_SHA256 = (
    "85d6058563fd67bbb77df095e707b0f155123b291959488c31274f2439f86bea"
)
PARENT_PREREGISTRATION_COMMIT = "5acedb9dbc7cdb6a477bba34fe21de295899adb9"
PARENT_PREREGISTRATION_TAG = (
    "tno-3gyr-full129-robustness-v1-preregistered-20261014"
)
PARENT_PREREGISTRATION_TAG_OBJECT = (
    "1504e185aceea8093ae47250374483f0af6adc3f"
)

PHYSICAL_MODEL_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "long_horizon" / "physical_models.py"
KNOWN_SYSTEM_DATA_ROOT = (
    PROJECT_ROOT / "src" / "jxplanetx" / "data" / "long_horizon_known_system_v2"
)
KNOWN_SYSTEM_SOURCE_PATH = (
    KNOWN_SYSTEM_DATA_ROOT / "horizons_de441_barycentric_icrf_jd2461200p5.csv"
)
KNOWN_SYSTEM_GM_PATH = KNOWN_SYSTEM_DATA_ROOT / "gm_de440_major_barycenters.csv"
KNOWN_SYSTEM_DERIVED_PATH = (
    KNOWN_SYSTEM_DATA_ROOT / "known_system_primary_relative_ecliptic_float_hex.csv"
)
KNOWN_SYSTEM_MANIFEST_PATH = KNOWN_SYSTEM_DATA_ROOT / "MANIFEST.json"
SOURCE_PROVIDER_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "restricted_wh_source.py"
FAST_WH_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "fast_wisdom_holman.py"
WH_CONTRACT_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "engine" / "wisdom_holman_contracts.py"
WH_SEMANTIC_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "engine" / "wisdom_holman.py"
WH_TRAJECTORY_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "engine" / "trajectory.py"
NATIVE_WRAPPER_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "_wisdom_holman_cpu_v3_module.c"
NATIVE_LOOP_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "_wisdom_holman_loop_v3.c"
NATIVE_BASE_LOOP_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "_wisdom_holman_loop_v2.c"
NATIVE_EXTENSION_PATH = (
    PROJECT_ROOT / "src" / "jxplanetx" / "_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
)
REBOUND_WHEEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "tno_stage1_rebound511_runtime_v1"
    / "rebound-5.1.1-cp314-manylinux_x86_64.whl"
)

INPUT_BINDINGS = (
    (PHYSICAL_MODEL_PATH, 60_256, "0383d5a87e7580f0e0f08b16ad410a9cf4a930d73e099d0b8f771c86b35fdbac", "known-system verifier"),
    (KNOWN_SYSTEM_SOURCE_PATH, 2_007, "97af9f841445a26d1c69dca46cfde73aa1a44c0651f19a1269a3facce6af8c9b", "DE441 common-epoch source rows"),
    (KNOWN_SYSTEM_GM_PATH, 448, "11fe03048f125e5283c3ef0d8054bc366baaf74dd86093ac3e61798e93769217", "DE440 GM rows"),
    (KNOWN_SYSTEM_DERIVED_PATH, 1_800, "6b3a2b1fbcee18821783c581a2e81ecdb9198a6da6550c81fbd03d579011bb8f", "verified common-epoch binary64 state"),
    (KNOWN_SYSTEM_MANIFEST_PATH, 4_013, "40d3e77c9ab5081f08354fe8e565e92008755059ef48ed9b7f43848c4f6f8231", "known-system custody manifest"),
    (SOURCE_PROVIDER_PATH, 61_803, "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52", "ordered-Jacobi source provider"),
    (FAST_WH_PATH, 36_182, "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72", "compiled-map Python boundary"),
    (WH_CONTRACT_PATH, 25_447, "cc9d08abcaf166804d1876d9a8c1c0d55b1b5a7df161f245f1842f3b1b7beef7", "Wisdom-Holman contract"),
    (WH_SEMANTIC_PATH, 230_474, "2c3ed08e04d6a622af63f6be82afa1a222582e772a891179cdc01aa0f2a7383f", "Wisdom-Holman semantic boundary"),
    (WH_TRAJECTORY_PATH, 68_392, "2a34e30bb248e0dfb60b7bd559bc1a540ebb5133419b49c80ac973480971a1e0", "trajectory failure boundary"),
    (NATIVE_WRAPPER_PATH, 8_178, "ee0e40a075b481516d8d30f8824b1fe31c9051c8c0de1dc8b321155882c411b3", "native CPython wrapper source"),
    (NATIVE_LOOP_PATH, 15_642, "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629", "native complete map source"),
    (NATIVE_BASE_LOOP_PATH, 58_558, "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750", "native exact fallback source"),
    (NATIVE_EXTENSION_PATH, 42_520, "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3", "compiled ordered-Jacobi extension"),
    (REBOUND_WHEEL_PATH, 903_973, "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19", "retained REBOUND 5.1.1 wheel"),
)

COMMON_EPOCH_JD_TDB = 2_461_200.5
J2000_JD_TDB = 2_451_545.0
JULIAN_YEAR_DAYS = 365.25
EPOCH_ORIGIN_YEARS = (COMMON_EPOCH_JD_TDB - J2000_JD_TDB) / JULIAN_YEAR_DAYS
HORIZON_JULIAN_YEARS = 1_000
CHECKPOINT_INTERVAL_JULIAN_YEARS = 1
CHECKPOINT_COUNT = 1_001
SOURCE_COUNT = 5
STEPS_PER_YEAR = (64, 128)
SOURCE_NODE_COUNTS = {64: 64_001, 128: 128_001}
SOURCE_NODE_BYTES = SOURCE_COUNT * 3 * 2 * 8
INVARIANT_NODE_BYTES = 4 * 8
SOURCE_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
SOURCE_NAIF_IDS = (10, 5, 6, 7, 8)
INNER_COLLAPSE_SOURCE_BODY_IDS = (10, 1, 2, 3, 4)
OMITTED_SOURCE_BODY_IDS = (9,)
KNOWN_SYSTEM_MODEL_CONTENT_SHA256 = (
    "19d33011f45d07d15a2aff8acf56a48c57e726828d8a34f1354f070987b86387"
)

EXPECTED_REDUCED_ROWS = (
    (
        "0x1.3bd13a81d2261p+5",
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0"),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0"),
    ),
    (
        "0x1.34c6609ebf23fp-5",
        ("-0x1.651216b0a70cdp+1", "0x1.1e0834079cc84p+2", "0x1.6734deb24ce45p-5"),
        ("-0x1.2fa4a88feac34p+1", "-0x1.54f7ff15d7e24p+0", "0x1.e01e220a346cbp-5"),
    ),
    (
        "0x1.71d0e2fd3b84cp-7",
        ("0x1.2c8eaba3f9093p+3", "0x1.23d6bc67bc276p+0", "-0x1.932d50e67e2acp-2"),
        ("-0x1.6f8cac076c7c6p-2", "0x1.024a97aaf9b84p+1", "-0x1.55c680b7f83bap-6"),
    ),
    (
        "0x1.c3d8f106aa649p-10",
        ("0x1.2a949e3159bd9p+3", "0x1.114453419b901p+4", "-0x1.d77b5d43ed91bp-5"),
        ("-0x1.460ed62fb9dddp+0", "0x1.3e0ed02b9d578p-1", "0x1.34768d0132a46p-6"),
    ),
    (
        "0x1.0a8c8f8c38c6ap-9",
        ("0x1.ddade925845f2p+4", "0x1.054ca186f62aep+0", "-0x1.6b037f8a9c9b9p-1"),
        ("-0x1.90c876e2b4023p-5", "0x1.26ee6e2e734dep+0", "-0x1.728d0fb09543cp-6"),
    ),
)

REBOUND_VERSION = "5.1.1"
REBOUND_WHEEL_SHA256 = "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19"
REBOUND_INIT_SHA256 = "924e19bfddb22bca9cbff0c641eaf3b2f791287f33df2d8d852f80aac78f8cdd"
REBOUND_LIBRARY_SHA256 = "52df4fe1050b98d6b0db11f31c18ec14ed00697a987e904bd7587cce4a0ccdef"
PINNED_PYTHON_PATH = "/usr/bin/python3.14"
PINNED_PYTHON_BYTES = 7_477_160
PINNED_PYTHON_SHA256 = "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41"
PINNED_PYTHON_VERSION = "3.14.4"
PINNED_NUMPY_VERSION = "2.5.3"
PINNED_NUMPY_RECORD_BYTES = 111_517
PINNED_NUMPY_RECORD_SHA256 = "9d23d1a798b32c85a52dd64d7f484ef38cc29b84692b05c64c23a3a1aada2ca9"
PINNED_NUMPY_INIT_BYTES = 25_220
PINNED_NUMPY_INIT_SHA256 = "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11"

CLAIMS = {
    "prepare_only_protocol_frozen": True,
    "dynamics_executed": False,
    "source_preflight_executed": False,
    "source_preflight_passed": False,
    "common_epoch_full_covariance_bridge_qualified": False,
    "h128_tracer_route_qualified": False,
    "all12_full129_qualification_passed": False,
    "external_forcing_qualified": False,
    "three_gyr_execution_authorized": False,
    "three_gyr_execution_completed": False,
    "classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class SourcePreflightError(RuntimeError):
    """The prepare-only source preflight contract failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise SourcePreflightError("record is not strict canonical JSON") from exc


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise SourcePreflightError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SourcePreflightError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except SourcePreflightError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise SourcePreflightError("JSON root must be an exact object")
    return value


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _regular_bytes(path: Path, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise SourcePreflightError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink():
        raise SourcePreflightError(f"{label} is not a direct regular file")
    try:
        return absolute.read_bytes()
    except OSError as exc:
        raise SourcePreflightError(f"cannot read {label}") from exc


def _binding(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    role: str,
) -> dict[str, Any]:
    payload = _regular_bytes(path, role)
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
        raise SourcePreflightError(f"{role} identity changed")
    return {
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_length": expected_bytes,
        "sha256": expected_sha256,
        "role": role,
    }


def _live_binding(path: Path, role: str) -> dict[str, Any]:
    payload = _regular_bytes(path, role)
    return {
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_length": len(payload),
        "sha256": _sha256(payload),
        "role": role,
    }


def _parse_exact_float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise SourcePreflightError(f"{label} is not float-hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise SourcePreflightError(f"{label} is not valid float-hex") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise SourcePreflightError(f"{label} is not canonical finite binary64")
    return parsed


def _load_reduced5_initial_state() -> dict[str, Any]:
    payload = _regular_bytes(KNOWN_SYSTEM_DERIVED_PATH, "known-system derived state")
    if len(payload) != 1_800 or _sha256(payload) != INPUT_BINDINGS[3][2]:
        raise SourcePreflightError("known-system derived state identity changed")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("ascii"), newline=""))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SourcePreflightError("known-system derived state is not canonical CSV") from exc
    expected_header = (
        "slot", "body_id", "body_name", "gm_au3_per_julian_year2_hex",
        "x_au_hex", "y_au_hex", "z_au_hex",
        "vx_au_per_julian_year_hex", "vy_au_per_julian_year_hex",
        "vz_au_per_julian_year_hex",
    )
    if tuple(reader.fieldnames or ()) != expected_header or len(rows) != 10:
        raise SourcePreflightError("known-system derived state geometry changed")
    expected_ids = (10, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    parsed: list[tuple[float, tuple[float, ...], tuple[float, ...]]] = []
    for index, (row, body_id) in enumerate(zip(rows, expected_ids, strict=True)):
        if row["slot"] != str(index) or row["body_id"] != str(body_id):
            raise SourcePreflightError("known-system derived state order changed")
        gm = _parse_exact_float_hex(
            row["gm_au3_per_julian_year2_hex"], f"GM row {index}"
        )
        position = tuple(
            _parse_exact_float_hex(row[name], f"position row {index}")
            for name in ("x_au_hex", "y_au_hex", "z_au_hex")
        )
        velocity = tuple(
            _parse_exact_float_hex(row[name], f"velocity row {index}")
            for name in (
                "vx_au_per_julian_year_hex",
                "vy_au_per_julian_year_hex",
                "vz_au_per_julian_year_hex",
            )
        )
        parsed.append((gm, position, velocity))
    central_gm = math.fsum(item[0] for item in parsed[:5])
    selected = (
        (central_gm, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        parsed[5], parsed[6], parsed[7], parsed[8],
    )
    observed = tuple(
        (
            gm.hex(),
            tuple(component.hex() for component in position),
            tuple(component.hex() for component in velocity),
        )
        for gm, position, velocity in selected
    )
    if observed != EXPECTED_REDUCED_ROWS:
        raise SourcePreflightError("reduced-five common-epoch projection changed")
    record: dict[str, Any] = {
        "schema": INITIAL_STATE_SCHEMA,
        "epoch_jd_tdb_hex": COMMON_EPOCH_JD_TDB.hex(),
        "epoch_scale": "TDB",
        "frame": "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000",
        "coordinate_system": "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED",
        "length_unit": "AU",
        "time_unit": "JULIAN_YEAR_365.25_D",
        "velocity_unit": "AU_PER_JULIAN_YEAR_365.25_D",
        "gm_unit": "AU3_PER_JULIAN_YEAR_365.25_D2",
        "known_system_model_content_sha256": KNOWN_SYSTEM_MODEL_CONTENT_SHA256,
        "collapse": {
            "central_members_body_ids": list(INNER_COLLAPSE_SOURCE_BODY_IDS),
            "central_gm_operation": "PYTHON_MATH_FSUM_IN_ASCENDING_SOURCE_SLOT_ORDER_ONE_BINARY64_RESULT",
            "central_state_operation": "RETAIN_EXACT_SUN_PRIMARY_RELATIVE_POSITION_AND_VELOCITY",
            "separate_outer_body_ids": [5, 6, 7, 8],
            "omitted_body_ids": list(OMITTED_SOURCE_BODY_IDS),
            "pluto_folded_into_central_gm": False,
        },
        "source_ids": list(SOURCE_IDS),
        "source_naif_ids": list(SOURCE_NAIF_IDS),
        "gm_AU3_per_julian_year2_hex": [item[0] for item in observed],
        "primary_relative_positions_AU_hex": [list(item[1]) for item in observed],
        "primary_relative_velocities_AU_per_julian_year_hex": [
            list(item[2]) for item in observed
        ],
    }
    record["content_sha256"] = _domain_hash(INITIAL_STATE_DOMAIN, record)
    return record


def _load_parent_contract() -> dict[str, Any]:
    payload = _regular_bytes(PARENT_PROTOCOL_PATH, "sealed 3-Gyr protocol")
    if len(payload) != PARENT_PROTOCOL_BYTES or _sha256(payload) != PARENT_PROTOCOL_SHA256:
        raise SourcePreflightError("sealed 3-Gyr protocol identity changed")
    value = _strict_json(payload)
    if value.get("content_sha256") != PARENT_PROTOCOL_CONTENT_SHA256:
        raise SourcePreflightError("sealed 3-Gyr protocol content identity changed")
    try:
        contract = value["cpu_gpu_integrator"]["short_horizon_qualification_contract"]
    except (KeyError, TypeError) as exc:
        raise SourcePreflightError("sealed qualification contract is absent") from exc
    if type(contract) is not dict:
        raise SourcePreflightError("sealed qualification contract is not an object")
    return contract


def _schedule_record(steps_per_year: int, initial_state_sha256: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SOURCE_SCHEDULE_SCHEMA,
        "initial_state_content_sha256": initial_state_sha256,
        "epoch_jd_tdb_hex": COMMON_EPOCH_JD_TDB.hex(),
        "epoch_origin_julian_years_since_j2000_hex": EPOCH_ORIGIN_YEARS.hex(),
        "steps_per_julian_year": steps_per_year,
        "fixed_step_julian_years_hex": (1.0 / steps_per_year).hex(),
        "source_step_count": HORIZON_JULIAN_YEARS * steps_per_year,
        "source_node_count": SOURCE_NODE_COUNTS[steps_per_year],
        "checkpoint_source_node_indices": {
            "first": 0,
            "last": HORIZON_JULIAN_YEARS * steps_per_year,
            "stride": steps_per_year,
            "count": CHECKPOINT_COUNT,
        },
        "exact_replay_required": True,
    }
    record["content_sha256"] = _domain_hash(SOURCE_SCHEDULE_DOMAIN, record)
    return record


def _fraction_record(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "nearest_binary64_hex": (numerator / denominator).hex(),
        "decision_arithmetic": "EXACT_INTEGER_RATIO_CROSS_MULTIPLICATION_NO_DISPLAY_ROUNDING",
    }


def _output_contract() -> dict[str, Any]:
    h64_source = SOURCE_NODE_COUNTS[64] * SOURCE_NODE_BYTES
    h128_source = SOURCE_NODE_COUNTS[128] * SOURCE_NODE_BYTES
    h64_invariants = SOURCE_NODE_COUNTS[64] * INVARIANT_NODE_BYTES
    h128_invariants = SOURCE_NODE_COUNTS[128] * INVARIANT_NODE_BYTES
    ias15_annual = CHECKPOINT_COUNT * SOURCE_NODE_BYTES
    return {
        "create_only_no_overwrite": True,
        "success_roster": [
            "PROTOCOL.json",
            "SOURCE_GM.bin",
            "JX_H64_SOURCE_NODES.bin",
            "JX_H128_SOURCE_NODES.bin",
            "JX_H64_ALL_NODE_INVARIANTS.bin",
            "JX_H128_ALL_NODE_INVARIANTS.bin",
            "IAS15_ANNUAL_SOURCE_STATES.bin",
            "RUNTIME.json",
            "RESULT.json",
        ],
        "binary_files": {
            "SOURCE_GM.bin": {
                "bytes": SOURCE_COUNT * 8,
                "order": "SOURCE_ORDER_GM",
            },
            "JX_H64_SOURCE_NODES.bin": {
                "bytes": h64_source,
                "order": "NODE_THEN_SOURCE_THEN_POSITION_XYZ_THEN_VELOCITY_XYZ",
            },
            "JX_H128_SOURCE_NODES.bin": {
                "bytes": h128_source,
                "order": "NODE_THEN_SOURCE_THEN_POSITION_XYZ_THEN_VELOCITY_XYZ",
            },
            "JX_H64_ALL_NODE_INVARIANTS.bin": {
                "bytes": h64_invariants,
                "order": "NODE_THEN_TOTAL_ENERGY_THEN_ANGULAR_MOMENTUM_XYZ",
            },
            "JX_H128_ALL_NODE_INVARIANTS.bin": {
                "bytes": h128_invariants,
                "order": "NODE_THEN_TOTAL_ENERGY_THEN_ANGULAR_MOMENTUM_XYZ",
            },
            "IAS15_ANNUAL_SOURCE_STATES.bin": {
                "bytes": ias15_annual,
                "order": "YEAR_0_THROUGH_1000_THEN_SOURCE_THEN_POSITION_XYZ_THEN_VELOCITY_XYZ",
            },
        },
        "binary_encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
        "fixed_binary_evidence_bytes": (
            SOURCE_COUNT * 8
            + h64_source + h128_source
            + h64_invariants + h128_invariants
            + ias15_annual
        ),
        "replay_evidence": (
            "SECOND_COMPLETE_FRESH_WORKSPACE_EXECUTION_COMPARED_BITWISE_BEFORE_"
            "RETAINING_ONE_CANONICAL_NODE_STREAM_AND_BOTH_RUNTIME_DIGESTS"
        ),
        "failure_transaction": (
            "ANY_NONFINITE_ROSTER_RUNTIME_REPLAY_INVARIANT_OR_ACCEPTANCE_FAILURE_"
            "PUBLISHES_NO_RESULT_AND_NEVER_RETRIES_WITH_CHANGED_PARAMETERS"
        ),
    }


def build_protocol() -> dict[str, Any]:
    initial_state = _load_reduced5_initial_state()
    inherited = _load_parent_contract()
    input_bindings = [
        _binding(path, length, digest, role)
        for path, length, digest, role in INPUT_BINDINGS
    ]
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": "PREPARE_ONLY_AUTONOMOUS_REDUCED5_MASSIVE_SOURCE_NUMERICAL_PREFLIGHT",
        "status": "PREPARED_SOURCE_ONLY_PREREGISTRATION_NO_EXECUTION",
        "evidence_class": (
            "PREPARED_1KYR_AUTONOMOUS_REDUCED5_SOURCE_NUMERICAL_"
            "PREFLIGHT_NO_MODEL_OUTPUT"
        ),
        "claim_ceiling": "PROTOCOL_ONLY_NO_DYNAMICS_NO_QUALIFICATION_NO_SCIENCE_RESULT",
        "objective": (
            "Freeze the first bounded H128 milestone: compare the autonomous common-epoch "
            "reduced-five massive source at H64 and H128 against pinned REBOUND 5.1.1 "
            "IAS15 for exactly 1,000 Julian years without propagating any tracer."
        ),
        "authority": {
            "parent_preregistration": {
                "commit": PARENT_PREREGISTRATION_COMMIT,
                "tag": PARENT_PREREGISTRATION_TAG,
                "tag_object": PARENT_PREREGISTRATION_TAG_OBJECT,
                "protocol_relative_path": PARENT_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "protocol_file_sha256": PARENT_PROTOCOL_SHA256,
                "protocol_content_sha256": PARENT_PROTOCOL_CONTENT_SHA256,
            },
            "preregistration_required_before_execution": True,
            "execution_successor_must_bind": [
                "THIS_EXACT_PROTOCOL_FILE_SHA256",
                "THIS_EXACT_PROTOCOL_CONTENT_SHA256",
                "PREREGISTRATION_COMMIT_SHA",
                "ANNOTATED_TAG_OBJECT_SHA",
                "EVERY_INPUT_BINDING",
            ],
            "execution_in_this_artifact": False,
            "explicitly_absent_public_cli": ["execute", "resume", "validate-result"],
            "filesystem_trust_boundary": "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        },
        "implementation": {
            "source": _live_binding(SCRIPT_PATH, "prepare-only source"),
            "test": _live_binding(TEST_PATH, "prepare-only tests"),
            "prepare_imports_numpy_or_dynamics_runtime": False,
        },
        "input_bindings": input_bindings,
        "initial_state": initial_state,
        "execution_specification": {
            "implemented": False,
            "authorized": False,
            "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
            "force_law": "NEWTONIAN_POINT_MASS_G_EQUALS_ONE",
            "tno_or_test_particle_count": 0,
            "external_forcing": "NONE",
            "collisions": "NONE",
            "removals": "FORBIDDEN",
            "epoch_jd_tdb_hex": COMMON_EPOCH_JD_TDB.hex(),
            "epoch_origin_julian_years_since_j2000_hex": EPOCH_ORIGIN_YEARS.hex(),
            "horizon_julian_years": HORIZON_JULIAN_YEARS,
            "annual_checkpoint_indices_inclusive": [0, HORIZON_JULIAN_YEARS],
            "annual_checkpoint_count": CHECKPOINT_COUNT,
            "jx_lanes": [
                {
                    "lane_id": "jx_ordered_jacobi_h64",
                    "engine": "JX_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN",
                    "steps_per_julian_year": 64,
                    "fixed_step_julian_years_hex": (1.0 / 64.0).hex(),
                    "source_step_count": 64_000,
                    "source_node_count": 64_001,
                    "schedule": _schedule_record(64, initial_state["content_sha256"]),
                },
                {
                    "lane_id": "jx_ordered_jacobi_h128",
                    "engine": "JX_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN",
                    "steps_per_julian_year": 128,
                    "fixed_step_julian_years_hex": (1.0 / 128.0).hex(),
                    "source_step_count": 128_000,
                    "source_node_count": 128_001,
                    "schedule": _schedule_record(128, initial_state["content_sha256"]),
                },
            ],
            "jx_map": {
                "public_builder": "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile",
                "jacobi_body_order": list(SOURCE_IDS),
                "dtype": "IEEE754_BINARY64",
                "backend": "NUMPY_CPU_PLUS_PINNED_NATIVE_EXTENSION",
                "compiler_flags": "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm",
                "source_encounter_radius_AU_hex": "0x1.0000000000000p+0",
                "minimum_massive_pair_separation_AU_hex": "0x1.0000000000000p+0",
                "minimum_jacobi_periapse_AU_hex": "0x1.999999999999ap-5",
                "all_fixed_nodes_retained": True,
                "exact_replay": (
                    "TWO_COMPLETE_SAME_RUNTIME_EXECUTIONS_FROM_SEPARATELY_OWNED_"
                    "INITIAL_ARRAYS_WITH_BITWISE_IDENTICAL_GM_POSITIONS_VELOCITIES_"
                    "METRICS_COUNTERS_AND_TERMINAL_STATE"
                ),
                "cross_runtime_bitwise_claim": False,
            },
            "ias15_reference": {
                "independent_implementation": True,
                "engine": "REBOUND_5.1.1_IAS15",
                "required_version": REBOUND_VERSION,
                "retained_wheel_relative_path": REBOUND_WHEEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "retained_wheel_sha256": REBOUND_WHEEL_SHA256,
                "wheel_rebound_init_sha256": REBOUND_INIT_SHA256,
                "wheel_librebound_sha256": REBOUND_LIBRARY_SHA256,
                "integrator": "ias15",
                "adaptive_mode": "PRS23",
                "epsilon_hex": (1.0e-13).hex(),
                "minimum_dt_julian_years_hex": "0x0.0p+0",
                "initial_dt_julian_years_hex": (1.0 / 32.0).hex(),
                "exact_finish_time": 1,
                "integration_calls": "SEQUENTIAL_ABSOLUTE_YEARS_1_THROUGH_1000",
                "heartbeat_binding": "PINNED_REBOUND_5_1_1_C_FIELD_WITH_LOCAL_CALLBACK_REFERENCE_DUE_MISSING_HB_SLOT",
                "annual_roster_and_finite_state_check": True,
                "no_jx_code_in_force_or_step_implementation": True,
            },
            "runtime_identity_required_before_lane_zero": {
                "python_executable": {
                    "resolved_path": PINNED_PYTHON_PATH,
                    "bytes": PINNED_PYTHON_BYTES,
                    "sha256": PINNED_PYTHON_SHA256,
                    "version_prefix": PINNED_PYTHON_VERSION,
                },
                "numpy_distribution": {
                    "version": PINNED_NUMPY_VERSION,
                    "record_bytes": PINNED_NUMPY_RECORD_BYTES,
                    "record_sha256": PINNED_NUMPY_RECORD_SHA256,
                    "numpy_init_bytes": PINNED_NUMPY_INIT_BYTES,
                    "numpy_init_sha256": PINNED_NUMPY_INIT_SHA256,
                },
                "native_extension_sha256": INPUT_BINDINGS[13][2],
                "rebound_wheel_sha256": REBOUND_WHEEL_SHA256,
                "loaded_module_and_shared_library_manifest_required": True,
                "blas_thread_environment_and_platform_receipt_required": True,
                "mismatch_action": "STOP_BEFORE_LANE_ZERO",
            },
        },
        "acceptance": {
            "inherited_full_qualification_contract_unmodified": inherited,
            "source_only_projection_cannot_satisfy_full_contract": True,
            "applicable_source_predicates": {
                "all_individual_predicates_must_pass": True,
                "body_scope_cartesian": list(SOURCE_IDS),
                "time_scope": "ALL_1001_ANNUAL_CHECKPOINTS_EXCEPT_ALL_NODE_INVARIANTS",
                "richardson_divisor": 3,
                "richardson_position_AU_max": _fraction_record(1, 500),
                "richardson_velocity_AU_per_julian_year_max": _fraction_record(1, 50000),
                "h128_vs_ias15_position_AU_max": _fraction_record(1, 500),
                "h128_vs_ias15_velocity_AU_per_julian_year_max": _fraction_record(1, 50000),
                "fine_relative_energy_drift_max": _fraction_record(1, 100_000_000),
                "fine_angular_vector_drift_max": _fraction_record(1, 10_000_000_000),
                "energy_contraction_floor_hex": float.fromhex("0x1.0000000000000p-46").hex(),
            },
            "tracer_only_predicates_not_evaluated": {
                "richardson_normalized_semimajor_axis_max": "3/8000",
                "h128_vs_ias15_normalized_semimajor_axis_max": "3/4000",
                "h128_vs_ias15_scattering_diagnostic_disagreement_max": "3/4000",
                "scalar_contraction_rule": inherited["scalar_contraction_rule"],
                "reason": "NO_TRACER_STATE_EXISTS_IN_THIS_SOURCE_ONLY_PREFLIGHT",
                "omission_counts_as_pass": False,
                "full_qualification_still_requires_every_inherited_TRACER_predicate": True,
            },
            "coordinate_comparison": (
                "AT_EACH_ANNUAL_EPOCH_RECONSTRUCT_EACH_LANE_IN_ITS_"
                "INSTANTANEOUS_COMMON_GM_BARYCENTER_THEN_COMPARE_BODY_MATCHED_"
                "CARTESIAN_VECTORS_WITH_EXACT_SQUARED_RATIONAL_TESTS"
            ),
            "richardson_decision": (
                "SOURCE_CARTESIAN_SQUARED_DIFFERENCE_DIVIDED_BY_9"
            ),
            "vector_contraction": (
                "FOR_EACH_SOURCE_AND_ANNUAL_VECTOR_FINE_SQUARED_LE_EXACT_SUM_"
                "OF_THREE_SQUARED_64_ULP_REFERENCE_COMPONENT_FLOORS_OR_"
                "COARSE_SQUARED_GE_9_TIMES_FINE_SQUARED"
            ),
            "invariants": {
                "jx_sampling": "EVERY_FIXED_NODE_INCLUDING_BOTH_ENDPOINTS",
                "ias15_sampling": "ANNUAL_CHECKPOINTS_ONLY_NOT_USED_FOR_JX_ALL_NODE_GATE",
                "barycentric_reconstruction": (
                    "PRIMARY_RELATIVE_TO_ZERO_COMMON_GM_BARYCENTER_USING_"
                    "ASCENDING_SOURCE_ORDER_MATH_FSUM_COMPONENTWISE"
                ),
                "energy": "SUM_HALF_GM_V_SQUARED_MINUS_SUM_I_LT_J_GM_I_GM_J_DIV_DISTANCE",
                "angular_vector": "SUM_GM_TIMES_CROSS_POSITION_VELOCITY",
                "relative_energy_drift": "ABS_E_NODE_MINUS_E_ZERO_DIV_ABS_E_ZERO",
                "relative_angular_vector_drift": "NORM_L_NODE_MINUS_L_ZERO_DIV_NORM_L_ZERO",
                "energy_contraction": "FINE_LE_MAX_COARSE_DIV_3_2_POW_MINUS_46",
                "fine_energy_max": "1/100000000",
                "fine_angular_max": "1/10000000000",
                "zero_or_nonfinite_denominator": "FAIL",
            },
            "ulp_floor": (
                "PER_CHECKPOINT_EXACT_64_TIMES_REFERENCE_SCALAR_ULP; VECTOR_"
                "FLOOR_IS_EXACT_SUM_OF_THREE_SQUARED_COMPONENT_FLOORS;_"
                "TAKE_MAXIMUM_OVER_THE_SAME_1001_CHECKPOINTS"
            ),
            "threshold_tuning_after_execution": "FORBIDDEN",
            "failure_interpretation": "QUALIFICATION_NOT_ESTABLISHED_NOT_SCIENTIFIC_DISPROOF",
        },
        "outputs": _output_contract(),
        "mandatory_phase_order": [
            "PREPARE_PROTOCOL_ONLY",
            "COMMIT_EXACT_SOURCE_TEST_AND_PROTOCOL",
            "CREATE_ANNOTATED_PREREGISTRATION_TAG",
            "IMPLEMENT_EXECUTOR_ONLY_IN_A_SUCCESSOR_COMMIT_WITHOUT_CHANGING_GATES",
            "VERIFY_PINNED_RUNTIME_AND_INPUTS_BEFORE_LANE_ZERO",
            "EXECUTE_H64_EXACT_REPLAY_AND_RETAIN_ALL_NODES",
            "EXECUTE_H128_EXACT_REPLAY_AND_RETAIN_ALL_NODES",
            "EXECUTE_PINNED_REBOUND511_IAS15_ANNUAL_REFERENCE",
            "EVALUATE_EVERY_FROZEN_SOURCE_PREDICATE",
            "PUBLISH_BOUNDED_SOURCE_RESULT_OR_FAIL_CLOSED",
            "REVIEW_BEFORE_ANY_TRACER_OR_LONGER_HORIZON_EXECUTION",
        ],
        "execution_blockers": [
            "THIS_PREPARE_ONLY_PROTOCOL_NOT_YET_COMMITTED_AND_ANNOTATED_TAGGED",
            "EXECUTION_SUCCESSOR_NOT_IMPLEMENTED",
            "PINNED_PROJECT_VENV_NOT_PRESENT_IN_THIS_WORKTREE",
            "NO_RESULT_SCHEMA_OR_OFFLINE_RESULT_VALIDATOR_IMPLEMENTED",
        ],
        "nonclaims": [
            "NO_TRACER_OR_COVARIANCE_PROPAGATION",
            "NO_COMMON_EPOCH_FULL_COVARIANCE_BRIDGE_QUALIFICATION",
            "NO_GPU_EXECUTION_OR_CPU_GPU_ROUTE_QUALIFICATION",
            "NO_GALACTIC_TIDE_STELLAR_COLLISION_OR_OUTER_BOUNDARY_MODEL",
            "NO_LONG_HORIZON_ACCURACY_OR_STABILITY_CLAIM",
            "NO_TNO_CLASSIFICATION_POPULATION_INFERENCE_OR_PLANET_NINE_EVIDENCE",
            "NO_THREE_GYR_EXECUTION_AUTHORIZATION",
        ],
        "claims": dict(CLAIMS),
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise SourcePreflightError("protocol must be an exact object")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise SourcePreflightError("protocol differs from the frozen source contract")
    observed = value.get("content_sha256")
    if observed != _domain_hash(PROTOCOL_DOMAIN, value):
        raise SourcePreflightError("protocol content digest differs")
    return str(observed)


def _stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in reversed([absolute, *absolute.parents]):
        if component.exists() and component.is_symlink():
            raise SourcePreflightError(f"{label} has a symlink ancestor")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SourcePreflightError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise SourcePreflightError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        source.encode("ascii"),
        directory_fd,
        destination.encode("ascii"),
        1,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise SourcePreflightError("protocol destination already exists")
        raise SourcePreflightError(
            f"renameat2 failed: {os.strerror(observed_errno)}"
        )


def _validate_prepared_directory(
    root_fd: int,
    expected_payload: bytes,
) -> dict[str, Any]:
    root_metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != os.geteuid()
    ):
        raise SourcePreflightError(
            "prepared root is not an owner-private directory"
        )
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise SourcePreflightError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise SourcePreflightError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                raise SourcePreflightError("prepared protocol ended early")
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise SourcePreflightError("prepared protocol changed while held")
    finally:
        os.close(descriptor)
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise SourcePreflightError("prepared root changed while held")
    payload = bytes(chunks)
    if payload != expected_payload:
        raise SourcePreflightError("prepared protocol bytes differ")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise SourcePreflightError("prepared protocol is not canonical JSON")
    validate_protocol(value)
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise SourcePreflightError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(root.parent, flags)
    except OSError as exc:
        raise SourcePreflightError("cannot open prepared parent") from exc
    created = False
    prepared: dict[str, Any] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise SourcePreflightError(
                "prepared root already exists; create-only publication refused"
            ) from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            if os.listdir(root_fd):
                raise SourcePreflightError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            descriptor = os.open(
                partial,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_directory(root_fd, payload)
            os.fsync(parent_fd)
        except Exception:
            for name in tuple(os.listdir(root_fd)):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    os.unlink(name, dir_fd=root_fd)
            os.fsync(root_fd)
            raise
        finally:
            os.close(root_fd)
    except SourcePreflightError:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    except OSError as exc:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise SourcePreflightError("cannot create prepared protocol") from exc
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise SourcePreflightError("prepared protocol validation did not complete")
    return prepared


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise SourcePreflightError("cannot stat prepared root") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root.is_symlink()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != os.getuid()
    ):
        raise SourcePreflightError("prepared root identity or permissions differ")
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise SourcePreflightError("cannot enumerate prepared root") from exc
    if tuple(item.name for item in entries) != ("PROTOCOL.json",):
        raise SourcePreflightError("prepared root must contain only PROTOCOL.json")
    protocol_metadata = entries[0].lstat()
    if (
        not stat.S_ISREG(protocol_metadata.st_mode)
        or entries[0].is_symlink()
        or protocol_metadata.st_nlink != 1
        or stat.S_IMODE(protocol_metadata.st_mode) != 0o600
        or protocol_metadata.st_uid != os.getuid()
    ):
        raise SourcePreflightError("prepared protocol identity or permissions differ")
    payload = _regular_bytes(entries[0], "prepared protocol")
    value = _strict_json(payload)
    validate_protocol(value)
    if payload != _canonical_json(value):
        raise SourcePreflightError("prepared protocol is not canonical JSON")
    return value


def scalar_contraction_pass(coarse: float, fine: float, reference: float) -> bool:
    """Apply the frozen scalar contraction rule with exact binary64 ratios."""

    if any(type(item) is not float or not math.isfinite(item) or item < 0.0 for item in (coarse, fine)):
        raise SourcePreflightError("scalar errors must be finite nonnegative floats")
    if type(reference) is not float or not math.isfinite(reference):
        raise SourcePreflightError("scalar reference must be a finite float")
    floor = 64 * Fraction.from_float(math.ulp(reference))
    return Fraction.from_float(fine) <= floor or Fraction.from_float(coarse) >= 3 * Fraction.from_float(fine)


def vector_contraction_pass(
    coarse_components: Sequence[float],
    fine_components: Sequence[float],
    reference_components: Sequence[float],
) -> bool:
    """Apply the frozen vector contraction using exact binary64 squares."""

    vectors = (coarse_components, fine_components, reference_components)
    if any(type(vector) not in (tuple, list) or len(vector) != 3 for vector in vectors):
        raise SourcePreflightError("vectors must contain exactly three components")
    if any(
        type(item) is not float or not math.isfinite(item)
        for vector in vectors
        for item in vector
    ):
        raise SourcePreflightError("vector components must be finite floats")
    coarse_exact = sum(
        (Fraction.from_float(item) ** 2 for item in coarse_components),
        Fraction(0, 1),
    )
    fine_exact = sum(
        (Fraction.from_float(item) ** 2 for item in fine_components),
        Fraction(0, 1),
    )
    floor_squared = sum(
        ((64 * Fraction.from_float(math.ulp(item))) ** 2 for item in reference_components),
        Fraction(0, 1),
    )
    return fine_exact <= floor_squared or coarse_exact >= 9 * fine_exact


def energy_contraction_pass(coarse_drift: float, fine_drift: float) -> bool:
    """Apply fine <= max(coarse/3, 2^-46) without decimal rounding."""

    if any(type(item) is not float or not math.isfinite(item) or item < 0.0 for item in (coarse_drift, fine_drift)):
        raise SourcePreflightError("energy drifts must be finite nonnegative floats")
    coarse_limit = Fraction.from_float(coarse_drift) / 3
    floor = Fraction(1, 1 << 46)
    return Fraction.from_float(fine_drift) <= max(coarse_limit, floor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="create only PROTOCOL.json")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser(
        "validate-prepared", help="validate the create-only prepared root"
    )
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        value = prepare(arguments.output_root)
    elif arguments.command == "validate-prepared":
        value = validate_prepared(arguments.output_root)
    else:  # pragma: no cover - argparse owns this boundary.
        raise SourcePreflightError("unsupported command")
    print(json.dumps({"content_sha256": value["content_sha256"], "status": value["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourcePreflightError as exc:
        print(f"source preflight error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
