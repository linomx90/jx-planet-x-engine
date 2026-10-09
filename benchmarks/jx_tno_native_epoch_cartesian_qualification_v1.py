#!/usr/bin/env python3
"""Qualify native-epoch TNO elements as heliocentric Cartesian states.

This is a deterministic CPU initial-state adapter, not a dynamics runner.  It
consumes the sealed 12-object/128-clone covariance artifact, binds the official
Horizons BODY10 GM semantic, checks nominal states against retained independent
Horizons vectors, and writes create-only protocol and state artifacts.  It has
no integration, classifier, GPU, population, or Planet Nine execution surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import types
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_native_epoch_cartesian_qualification_v1.py"
ACQUISITION_ROOT = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1"
CLONE_ARTIFACT_PATH = PROJECT_ROOT / "runs" / "tno_covariance_clones_v1" / "CLONES.json"
PILOT_TOOL = PROJECT_ROOT / "benchmarks" / "jx_tno_covariance_classification_pilot_v1.py"
PILOT_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_covariance_classification_pilot_v1" / "PROTOCOL.json"
ORACLE_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_horizons_oracle_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1"

PROTOCOL_SCHEMA = "jx.tno.native-epoch-cartesian.protocol.v1"
STATES_SCHEMA = "jx.tno.native-epoch-cartesian.states.v1"
PROTOCOL_DOMAIN = b"jx.tno.native-epoch-cartesian.protocol.v1\0"
STATES_DOMAIN = b"jx.tno.native-epoch-cartesian.states.v1\0"
NATIVE_VECTOR_DOMAIN = b"jx.tno.native-epoch-cartesian.native-vector.v1\0"
BINARY_PAYLOAD_DOMAIN = b"jx.tno.native-epoch-cartesian.binary-payload.v1\0"
CLASSIFICATION = "NATIVE_EPOCH_CARTESIAN_INPUT_QUALIFICATION_ONLY"

ELEMENT_LABELS = ("e", "q", "tp", "node", "peri", "i")
ELEMENT_UNITS = (None, "au", "JD TDB; covariance differential in days", "deg", "deg", "deg")
JULIAN_YEAR_DAYS = 365.25
SECONDS_PER_DAY_EXACT = "86400"
SECONDS_PER_JULIAN_YEAR_EXACT = "31557600"
KM_PER_AU_EXACT = "149597870.7"
HORIZONS_BODY10_GM_KM3_PER_S2_EXACT = "132712440041.27942"
HORIZONS_BODY10_GM_TOKEN = "1.3271244004127942E+11"
HORIZONS_BODY10_GM_AU3_PER_DAY2_HEX = "0x1.3649541cf2bdcp-12"
HORIZONS_BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX = "0x1.3bd0beccac6afp+5"

KEPLER_MAXIMUM_ITERATIONS = 128
KEPLER_RESIDUAL_TOLERANCE = 2.0e-14
ROUNDTRIP_A_RELATIVE_TOLERANCE = 2.0e-12
ROUNDTRIP_Q_RELATIVE_TOLERANCE = 2.0e-12
ROUNDTRIP_E_ABSOLUTE_TOLERANCE = 2.0e-13
ROUNDTRIP_ANGLE_TOLERANCE_RAD = 2.0e-11
ORACLE_POSITION_COMPONENT_TOLERANCE_AU = 2.0e-11
ORACLE_VELOCITY_COMPONENT_TOLERANCE_AU_PER_DAY = 2.0e-15
ORACLE_TP_PRINTING_TOLERANCE_DAYS = "2.17E-10"
ORACLE_TP_PRINTING_MAXIMUM_ULP_DISTANCE = 1

# These bindings are the exact current acquisition, clone, and pilot custody
# that this Stage-0 adapter is allowed to consume.
UPSTREAM_BINDINGS = (
    ("acquisition_implementation", "tools/jx_tno_sbdb_covariance_acquisition_v1.py", 31_175, "51e96414e136e2870542383eb505c75b58c46b0e38f70fe0417b81791047eeca"),
    ("acquisition_test", "tests/test_tno_sbdb_covariance_acquisition_v1.py", 11_462, "a9c39da396c43f06b34c813e70ca2a267d1ff7007150f6e9ad171dcee8f352f4"),
    ("acquisition_protocol", "runs/tno_sbdb_covariance_acquisition_v1/PROTOCOL.json", 6_261, "4f3de80ba98b43c907bb846bd35724a0f1bd0ef9ef152e7642e3af65a2229acc"),
    ("acquisition_custody", "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json", 16_520, "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288"),
    ("clone_implementation", "tools/jx_tno_covariance_clone_materialization_v1.py", 27_114, "49f00b856810225a495cbd16e34282db2199e2295d755c2bb2aaba4897b7ba53"),
    ("clone_test", "tests/test_tno_covariance_clone_materialization_v1.py", 9_607, "c2fdd4eddd85a9a5655e72bb8ad899bd6d3dc5db9c904b4259cee23b5cacbf0d"),
    ("clone_protocol", "runs/tno_covariance_clones_v1/PROTOCOL.json", 3_358, "88745a6f8286cf1c6f717033973fa972457f8e088eebbfe1d6047b1286a1f21b"),
    ("clone_artifact", "runs/tno_covariance_clones_v1/CLONES.json", 531_609, "395ce9914df0efe235e2144c24abd157f5aa3ba87beba87fa39891ec42312014"),
    ("pilot_implementation", "benchmarks/jx_tno_covariance_classification_pilot_v1.py", 31_585, "7d90ff40e71400071eb7d0ecf315113342e7c76cf8ae91935310f8895fcde2e1"),
    ("pilot_test", "tests/test_tno_covariance_classification_pilot_v1.py", 13_134, "10d3a8b162ecc325370b7e973bceb4da5b9c6bbc4dacff93a3cfa2d3d93bfc6c"),
    ("pilot_protocol", "runs/tno_covariance_classification_pilot_v1/PROTOCOL.json", 13_599, "583260b2d9886f0745f1cffd37bd800d18238dfb791d976d5a539a72d5d87f4f"),
    ("repo_de440_gm_crosscheck", "runs/v5_solar_1pn_qualification/sources/gm_de440.tpc", 12_406, "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"),
)
ACQUISITION_PROTOCOL_CONTENT_SHA256 = "176b56c4784bef58a8eacc6e34d99edb17d21d70c559a245b7c0677cdb1a4a60"
ACQUISITION_CUSTODY_CONTENT_SHA256 = "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f"
CLONE_PROTOCOL_CONTENT_SHA256 = "939e6c3f8e3773f2af4675bab95369e702461ae00edf311b71c02f16785ab798"
CLONE_ARTIFACT_CONTENT_SHA256 = "9f1c0df03a2001f6249aafd423c4c9feb8bde9a75467416d35b81b88bb2d59d8"
CLONE_BINARY_PAYLOAD_SHA256 = "3f296d6bf9489598674e152fc7511641023c9d513fd4f6ca6d33ab6ec7293278"
PILOT_PROTOCOL_CONTENT_SHA256 = "af27a4c5014b83d6b7c0241edba807e10ae635aaf641e6c5ad1eaf1fc1176db0"

# Exact independently retained nominal-state oracle custody.  The adapter is
# entirely offline: the acquisition implementation revalidates the retained
# raw responses and official Horizons GM source from these bound bytes.
ORACLE_BINDINGS = (
    (
        "oracle_implementation",
        "tools/jx_tno_native_epoch_horizons_oracle_v1.py",
        23_376,
        "48964a69e7597bfabb4ae89b15e586c96c48b0b39008a98a4bac28df64a23b6e",
    ),
    (
        "oracle_test",
        "tests/test_tno_native_epoch_horizons_oracle_v1.py",
        8_981,
        "2e8b842d2bfd892df726c0edbd4e770466c768e9b65247f7d4bf82de5bf204f7",
    ),
    (
        "oracle_protocol",
        "runs/tno_native_epoch_horizons_oracle_v1/PROTOCOL.json",
        7_933,
        "38326ac4338ea59951aa70f80cbcb2658d13ad55b941f48c3ee268cf89097c24",
    ),
    (
        "oracle_custody",
        "runs/tno_native_epoch_horizons_oracle_v1/CUSTODY.json",
        22_587,
        "e2ad87ad294cf3c4682146f4b278c26f898afce517047de7e1ab1b60a8588bd7",
    ),
    (
        "official_horizons_gm",
        "runs/tno_native_epoch_horizons_oracle_v1/source/gm_Horizons.pck",
        15_428,
        "169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c",
    ),
)
ORACLE_CUSTODY_CONTENT_SHA256 = "45dd99dbce486f173516bc14304e03d6129ebb44e404bfe5c3a3b4519c3af978"
ORACLE_RECORD_ROSTER_SHA256 = "1bc04b68289469f1ee4ba58d467cfeb36eca30f56a3e3ec973ccf8783f9c81a2"
ORACLE_PROTOCOL_CONTENT_SHA256 = "44d9b643d5df7c5a3f913d9d4cadd86efa20013e6f9e501498dd1df96b66a4e0"
ORACLE_IMPLEMENTATION_RELATIVE_PATH = "tools/jx_tno_native_epoch_horizons_oracle_v1.py"

ORACLE_EXPECTED_RECORDS = (
    ("jpl-20090377", 20_090_377, "90377", "51", "2458411.5"),
    ("jpl-20474640", 20_474_640, "474640", "12", "2456187.5"),
    ("jpl-54526835", 54_526_835, "2023 KQ14", "3", "2460034.5"),
    ("jpl-20541132", 20_541_132, "541132", "7", "2458052.5"),
    ("jpl-50666516", 50_666_516, "2012 VP113", "11", "2456596.5"),
    ("jpl-20148209", 20_148_209, "148209", "5", "2452568.5"),
    ("jpl-20574372", 20_574_372, "574372", "14", "2458299.5"),
    ("jpl-20303775", 20_303_775, "303775", "29", "2458767.5"),
    ("jpl-20026181", 20_026_181, "26181", "25", "2458156.5"),
    ("jpl-20042301", 20_042_301, "42301", "28", "2458324.5"),
    ("jpl-20026375", 20_026_375, "26375", "38", "2457961.5"),
    ("jpl-20015874", 20_015_874, "15874", "34", "2458098.5"),
)


class CartesianQualificationError(ValueError):
    """A typed input, numerical gate, or sealed artifact failed closed."""


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise CartesianQualificationError(f"{label} must be a finite built-in float")
    return value


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise CartesianQualificationError(f"{label} must be positive")
    return result


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CartesianQualificationError(f"{label} must be a nonempty trimmed string")
    return value


def _sha256_text(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CartesianQualificationError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if (
        type(value) is not tuple
        or len(value) != 3
        or any(type(component) is not float or not math.isfinite(component) for component in value)
    ):
        raise CartesianQualificationError(f"{label} must be an immutable finite float triplet")
    return value


@dataclass(frozen=True, slots=True)
class CentralGMSemantic:
    semantic_id: str
    primary_id: str
    gm_AU3_per_julian_year2: float
    source_relative_path: str
    source_file_sha256: str
    source_token_km3_per_s2: str

    def __post_init__(self) -> None:
        _text(self.semantic_id, "GM semantic_id")
        _text(self.primary_id, "GM primary_id")
        _positive_float(self.gm_AU3_per_julian_year2, "central GM")
        _text(self.source_relative_path, "GM source path")
        _sha256_text(self.source_file_sha256, "GM source digest")
        _text(self.source_token_km3_per_s2, "GM source token")


@dataclass(frozen=True, slots=True)
class NativeElementRecord:
    object_id: str
    spk_id: int
    cohort: str
    covariance_epoch_jd_tdb: float
    clone_index: int | None
    values: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        _text(self.object_id, "object_id")
        if type(self.spk_id) is not int or self.spk_id <= 0:
            raise CartesianQualificationError("spk_id must be a positive built-in int")
        _text(self.cohort, "cohort")
        _finite_float(self.covariance_epoch_jd_tdb, "covariance epoch")
        if self.clone_index is not None and (
            type(self.clone_index) is not int or not 0 <= self.clone_index < 128
        ):
            raise CartesianQualificationError("clone_index must be None or an int in [0,127]")
        if (
            type(self.values) is not tuple
            or len(self.values) != 6
            or any(type(value) is not float or not math.isfinite(value) for value in self.values)
        ):
            raise CartesianQualificationError("native elements must be six immutable finite floats")
        eccentricity, perihelion, _tp, _node, _peri, inclination = self.values
        if not 0.0 < eccentricity < 1.0:
            raise CartesianQualificationError("eccentricity must be in the open elliptic interval")
        if perihelion <= 0.0:
            raise CartesianQualificationError("perihelion distance must be positive")
        if not 0.0 <= inclination <= 180.0:
            raise CartesianQualificationError("inclination must be in [0,180] degrees")


@dataclass(frozen=True, slots=True)
class CartesianState:
    semimajor_axis_AU: float
    mean_anomaly_rad: float
    eccentric_anomaly_rad: float
    position_AU: tuple[float, float, float]
    velocity_AU_per_julian_year: tuple[float, float, float]
    kepler_residual_rad: float

    def __post_init__(self) -> None:
        _positive_float(self.semimajor_axis_AU, "semimajor axis")
        _finite_float(self.mean_anomaly_rad, "mean anomaly")
        _finite_float(self.eccentric_anomaly_rad, "eccentric anomaly")
        _vec3(self.position_AU, "position")
        _vec3(self.velocity_AU_per_julian_year, "velocity")
        _finite_float(self.kepler_residual_rad, "Kepler residual")


@dataclass(frozen=True, slots=True)
class RecoveredElements:
    semimajor_axis_AU: float
    eccentricity: float
    inclination_rad: float
    ascending_node_rad: float
    argument_of_perihelion_rad: float
    mean_anomaly_rad: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256_bytes(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise CartesianQualificationError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CartesianQualificationError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CartesianQualificationError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise CartesianQualificationError("JSON root must be an object")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CartesianQualificationError(f"required regular file cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CartesianQualificationError(f"file is linked or nonregular: {path}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as exc:
        raise CartesianQualificationError(f"file disappeared while read: {path}") from exc
    first = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    second = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    third = (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns)
    if first != second or first != third or len(payload) != before.st_size:
        raise CartesianQualificationError(f"file changed while read: {path}")
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = _read_regular_bytes(path)
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise CartesianQualificationError(f"retained JSON is not canonical: {path}")
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise CartesianQualificationError(f"{label} differs from the frozen contract")


def _bound_file_payload(binding: tuple[str, str, int, str]) -> bytes:
    if type(binding) is not tuple or len(binding) != 4:
        raise CartesianQualificationError("file binding must be an immutable four-field tuple")
    role, relative_path, expected_size, expected_sha256 = binding
    _text(role, "bound file role")
    _text(relative_path, "bound file relative path")
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise CartesianQualificationError("bound file path must remain inside the project")
    if type(expected_size) is not int or expected_size <= 0:
        raise CartesianQualificationError("bound file size must be a positive built-in int")
    _sha256_text(expected_sha256, "bound file digest")
    payload = _read_regular_bytes(PROJECT_ROOT / relative_path)
    if len(payload) != expected_size:
        raise CartesianQualificationError(f"bound file size differs: {relative_path}")
    if _sha256_bytes(payload) != expected_sha256:
        raise CartesianQualificationError(f"bound file digest differs: {relative_path}")
    return payload


def _bound_file_record(binding: tuple[str, str, int, str]) -> dict[str, object]:
    role, relative_path, expected_size, expected_sha256 = binding
    _bound_file_payload(binding)
    return {
        "role": role,
        "relative_path": relative_path,
        "bytes": expected_size,
        "file_sha256": expected_sha256,
    }


def _binding_for_role(
    bindings: tuple[tuple[str, str, int, str], ...], role: str
) -> tuple[str, str, int, str]:
    matches = tuple(binding for binding in bindings if binding[0] == role)
    if len(matches) != 1:
        raise CartesianQualificationError(f"bound role must occur exactly once: {role}")
    return matches[0]


def _load_bound_module(name: str, binding: tuple[str, str, int, str]):
    """Compile the exact hash-checked source bytes, never an adjacent .pyc."""

    _text(name, "module name")
    payload = _bound_file_payload(binding)
    path = PROJECT_ROOT / binding[1]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(payload, str(path), "exec", dont_inherit=True, optimize=-1)
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise CartesianQualificationError(f"verifier failed to load: {path}") from exc
    return module


def _float_from_canonical_hex(value: object, label: str) -> float:
    token = _text(value, label)
    try:
        result = float.fromhex(token)
    except ValueError as exc:
        raise CartesianQualificationError(f"{label} is not a binary64 hex literal") from exc
    if not math.isfinite(result) or result.hex() != token:
        raise CartesianQualificationError(f"{label} is not canonical finite binary64 hex")
    return result


def _wrap_signed_radians(value: float) -> float:
    result = math.remainder(value, math.tau)
    if not math.isfinite(result):
        raise CartesianQualificationError("angle reduction produced a nonfinite value")
    return result


def solve_elliptic_kepler(mean_anomaly_rad: float, eccentricity: float) -> float:
    """Solve E-e*sin(E)=M with safeguarded Newton steps in [-pi, pi]."""

    mean = _finite_float(mean_anomaly_rad, "mean anomaly")
    eccentricity = _finite_float(eccentricity, "eccentricity")
    if not 0.0 <= eccentricity < 1.0:
        raise CartesianQualificationError("eccentricity must be in [0,1)")
    mean = _wrap_signed_radians(mean)
    lower = -math.pi
    upper = math.pi
    if eccentricity < 0.8:
        estimate = mean
    elif mean > 0.0:
        estimate = math.pi
    elif mean < 0.0:
        estimate = -math.pi
    else:
        estimate = 0.0
    for _ in range(KEPLER_MAXIMUM_ITERATIONS):
        residual = estimate - eccentricity * math.sin(estimate) - mean
        if residual == 0.0:
            return estimate
        if residual > 0.0:
            upper = estimate
        else:
            lower = estimate
        derivative = 1.0 - eccentricity * math.cos(estimate)
        candidate = estimate - residual / derivative
        if not math.isfinite(candidate) or not lower < candidate < upper:
            candidate = 0.5 * (lower + upper)
        if candidate == estimate:
            break
        estimate = candidate
    residual = estimate - eccentricity * math.sin(estimate) - mean
    if not math.isfinite(residual) or abs(residual) > KEPLER_RESIDUAL_TOLERANCE:
        raise CartesianQualificationError("elliptic Kepler solve did not converge")
    return estimate


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(value, value))


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def native_elements_to_cartesian(
    native: NativeElementRecord, gm: CentralGMSemantic
) -> CartesianState:
    """Convert one native element vector using an explicit central-GM semantic."""

    if type(native) is not NativeElementRecord or type(gm) is not CentralGMSemantic:
        raise CartesianQualificationError("conversion requires exact immutable typed records")
    eccentricity, perihelion, tp_jd_tdb, node_deg, peri_deg, inclination_deg = native.values
    semimajor_axis = perihelion / (1.0 - eccentricity)
    central_gm = gm.gm_AU3_per_julian_year2
    mean_motion = math.sqrt(central_gm / (semimajor_axis * semimajor_axis * semimajor_axis))
    delta_days = math.fsum((native.covariance_epoch_jd_tdb, -tp_jd_tdb))
    mean_anomaly = _wrap_signed_radians(mean_motion * (delta_days / JULIAN_YEAR_DAYS))
    eccentric_anomaly = solve_elliptic_kepler(mean_anomaly, eccentricity)
    sine_e = math.sin(eccentric_anomaly)
    cosine_e = math.cos(eccentric_anomaly)
    beta = math.sqrt((1.0 - eccentricity) * (1.0 + eccentricity))
    denominator = 1.0 - eccentricity * cosine_e
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CartesianQualificationError("elliptic anomaly produced an invalid radius")
    perifocal_x = semimajor_axis * (cosine_e - eccentricity)
    perifocal_y = semimajor_axis * beta * sine_e
    speed_scale = math.sqrt(central_gm / semimajor_axis) / denominator
    perifocal_vx = -speed_scale * sine_e
    perifocal_vy = speed_scale * beta * cosine_e

    node = math.radians(node_deg)
    argument = math.radians(peri_deg)
    inclination = math.radians(inclination_deg)
    cosine_node, sine_node = math.cos(node), math.sin(node)
    cosine_argument, sine_argument = math.cos(argument), math.sin(argument)
    cosine_inclination, sine_inclination = math.cos(inclination), math.sin(inclination)
    r11 = cosine_node * cosine_argument - sine_node * sine_argument * cosine_inclination
    r12 = -cosine_node * sine_argument - sine_node * cosine_argument * cosine_inclination
    r21 = sine_node * cosine_argument + cosine_node * sine_argument * cosine_inclination
    r22 = -sine_node * sine_argument + cosine_node * cosine_argument * cosine_inclination
    r31 = sine_argument * sine_inclination
    r32 = cosine_argument * sine_inclination
    position = (
        r11 * perifocal_x + r12 * perifocal_y,
        r21 * perifocal_x + r22 * perifocal_y,
        r31 * perifocal_x + r32 * perifocal_y,
    )
    velocity = (
        r11 * perifocal_vx + r12 * perifocal_vy,
        r21 * perifocal_vx + r22 * perifocal_vy,
        r31 * perifocal_vx + r32 * perifocal_vy,
    )
    residual = eccentric_anomaly - eccentricity * sine_e - mean_anomaly
    result = CartesianState(
        semimajor_axis_AU=semimajor_axis,
        mean_anomaly_rad=mean_anomaly,
        eccentric_anomaly_rad=eccentric_anomaly,
        position_AU=position,
        velocity_AU_per_julian_year=velocity,
        kepler_residual_rad=residual,
    )
    validate_roundtrip(native, result, gm)
    return result


def cartesian_to_elements(state: CartesianState, gm: CentralGMSemantic) -> RecoveredElements:
    """Independently recover nonsingular elliptic elements from vector invariants."""

    if type(state) is not CartesianState or type(gm) is not CentralGMSemantic:
        raise CartesianQualificationError("inverse conversion requires exact typed records")
    position = state.position_AU
    velocity = state.velocity_AU_per_julian_year
    central_gm = gm.gm_AU3_per_julian_year2
    radius = _norm(position)
    angular_momentum = _cross(position, velocity)
    hnorm = _norm(angular_momentum)
    if radius <= 0.0 or hnorm <= 0.0:
        raise CartesianQualificationError("Cartesian state has singular radius or angular momentum")
    node = (-angular_momentum[1], angular_momentum[0], 0.0)
    nnorm = _norm(node)
    velocity_cross_h = _cross(velocity, angular_momentum)
    eccentricity_vector = tuple(
        velocity_cross_h[index] / central_gm - position[index] / radius
        for index in range(3)
    )
    eccentricity = _norm(eccentricity_vector)
    energy = 0.5 * _dot(velocity, velocity) - central_gm / radius
    if energy >= 0.0 or not 0.0 < eccentricity < 1.0 or nnorm <= 0.0:
        raise CartesianQualificationError("inverse state is not a nonsingular bound inclined ellipse")
    semimajor_axis = -central_gm / (2.0 * energy)
    inclination = math.acos(_clamp_unit(angular_momentum[2] / hnorm))
    ascending_node = math.atan2(node[1], node[0]) % math.tau
    argument_sine = _dot(_cross(node, eccentricity_vector), angular_momentum) / (
        nnorm * eccentricity * hnorm
    )
    argument_cosine = _dot(node, eccentricity_vector) / (nnorm * eccentricity)
    argument = math.atan2(argument_sine, argument_cosine) % math.tau
    cosine_e = _clamp_unit((1.0 - radius / semimajor_axis) / eccentricity)
    sine_e = _clamp_unit(
        _dot(position, velocity) / (eccentricity * math.sqrt(central_gm * semimajor_axis))
    )
    eccentric_anomaly = math.atan2(sine_e, cosine_e)
    mean_anomaly = _wrap_signed_radians(
        eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
    )
    values = (
        semimajor_axis,
        eccentricity,
        inclination,
        ascending_node,
        argument,
        mean_anomaly,
    )
    if any(not math.isfinite(value) for value in values):
        raise CartesianQualificationError("inverse conversion produced a nonfinite value")
    return RecoveredElements(*values)


def _angular_error(left: float, right: float) -> float:
    return abs(math.remainder(left - right, math.tau))


def roundtrip_errors(
    native: NativeElementRecord, state: CartesianState, gm: CentralGMSemantic
) -> dict[str, float]:
    recovered = cartesian_to_elements(state, gm)
    eccentricity, perihelion, _tp, node_deg, peri_deg, inclination_deg = native.values
    expected_node = math.radians(node_deg) % math.tau
    expected_argument = math.radians(peri_deg) % math.tau
    expected_inclination = math.radians(inclination_deg)
    recovered_perihelion = recovered.semimajor_axis_AU * (1.0 - recovered.eccentricity)
    return {
        "semimajor_axis_relative": abs(recovered.semimajor_axis_AU - state.semimajor_axis_AU) / state.semimajor_axis_AU,
        "perihelion_relative": abs(recovered_perihelion - perihelion) / perihelion,
        "eccentricity_absolute": abs(recovered.eccentricity - eccentricity),
        "inclination_rad": abs(recovered.inclination_rad - expected_inclination),
        "ascending_node_rad": _angular_error(recovered.ascending_node_rad, expected_node),
        "argument_of_perihelion_rad": _angular_error(recovered.argument_of_perihelion_rad, expected_argument),
        "mean_anomaly_rad": _angular_error(recovered.mean_anomaly_rad, state.mean_anomaly_rad),
    }


def validate_roundtrip(
    native: NativeElementRecord, state: CartesianState, gm: CentralGMSemantic
) -> dict[str, float]:
    errors = roundtrip_errors(native, state, gm)
    gates = {
        "semimajor_axis_relative": ROUNDTRIP_A_RELATIVE_TOLERANCE,
        "perihelion_relative": ROUNDTRIP_Q_RELATIVE_TOLERANCE,
        "eccentricity_absolute": ROUNDTRIP_E_ABSOLUTE_TOLERANCE,
        "inclination_rad": ROUNDTRIP_ANGLE_TOLERANCE_RAD,
        "ascending_node_rad": ROUNDTRIP_ANGLE_TOLERANCE_RAD,
        "argument_of_perihelion_rad": ROUNDTRIP_ANGLE_TOLERANCE_RAD,
        "mean_anomaly_rad": ROUNDTRIP_ANGLE_TOLERANCE_RAD,
    }
    for key, tolerance in gates.items():
        if errors[key] > tolerance:
            raise CartesianQualificationError(f"Cartesian inverse roundtrip failed: {key}")
    return errors


def _derive_official_gm(source_path: Path, source_sha256: str) -> CentralGMSemantic:
    payload = _read_regular_bytes(source_path)
    if _sha256_bytes(payload) != source_sha256:
        raise CartesianQualificationError("official Horizons GM artifact digest differs")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CartesianQualificationError("official Horizons GM artifact is not ASCII") from exc
    matches = re.findall(r"\bBODY10_GM\s*=\s*\(\s*([0-9.+\-EDed]+)\s*\)", text)
    if not matches or set(matches) != {HORIZONS_BODY10_GM_TOKEN}:
        raise CartesianQualificationError("official Horizons BODY10_GM token differs")
    repo_payload = _read_regular_bytes(
        PROJECT_ROOT / "runs/v5_solar_1pn_qualification/sources/gm_de440.tpc"
    )
    if _sha256_bytes(repo_payload) != "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140":
        raise CartesianQualificationError("repo DE440 GM crosscheck digest differs")
    try:
        repo_text = repo_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CartesianQualificationError("repo DE440 GM crosscheck is not ASCII") from exc
    repo_matches = re.findall(
        r"\bBODY10_GM\s*=\s*\(\s*([0-9.+\-EDed]+)\s*\)", repo_text
    )
    if not repo_matches or set(repo_matches) != {HORIZONS_BODY10_GM_TOKEN}:
        raise CartesianQualificationError("official Horizons and repo BODY10_GM tokens differ")
    with localcontext() as context:
        context.prec = 100
        gm_km = Decimal(matches[0].replace("D", "E").replace("d", "e"))
        au_km = Decimal(KM_PER_AU_EXACT)
        gm_day = gm_km * Decimal(SECONDS_PER_DAY_EXACT) ** 2 / au_km**3
        gm_year = gm_km * Decimal(SECONDS_PER_JULIAN_YEAR_EXACT) ** 2 / au_km**3
    if str(gm_km) != HORIZONS_BODY10_GM_KM3_PER_S2_EXACT:
        raise CartesianQualificationError("official Horizons BODY10_GM decimal differs")
    if float(gm_day).hex() != HORIZONS_BODY10_GM_AU3_PER_DAY2_HEX:
        raise CartesianQualificationError("official Horizons daily GM conversion differs")
    if float(gm_year).hex() != HORIZONS_BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX:
        raise CartesianQualificationError("official Horizons yearly GM conversion differs")
    return CentralGMSemantic(
        semantic_id="OFFICIAL_HORIZONS_DE441_STATE_TO_OSCULATING_BODY10_GM_EXACT_DECIMAL_V1",
        primary_id="NAIF_10_SUN",
        gm_AU3_per_julian_year2=float(gm_year),
        source_relative_path=source_path.relative_to(PROJECT_ROOT).as_posix(),
        source_file_sha256=source_sha256,
        source_token_km3_per_s2=matches[0],
    )


def _parse_native_records(clone_artifact: Mapping[str, Any]) -> tuple[NativeElementRecord, ...]:
    objects = clone_artifact.get("objects")
    if type(objects) is not list or len(objects) != 12:
        raise CartesianQualificationError("clone artifact object roster differs")
    records: list[NativeElementRecord] = []
    for object_record in objects:
        if type(object_record) is not dict:
            raise CartesianQualificationError("clone object record must be an object")
        object_id = _text(object_record.get("object_id"), "clone object_id")
        spk_id = object_record.get("spk_id")
        cohort = _text(object_record.get("cohort"), "clone cohort")
        if type(spk_id) is not int or spk_id <= 0:
            raise CartesianQualificationError("clone SPK ID is invalid")
        epoch_token = _text(object_record.get("covariance_epoch_jd_tdb"), "covariance epoch")
        try:
            epoch = float(epoch_token)
        except ValueError as exc:
            raise CartesianQualificationError("covariance epoch is not decimal") from exc
        nominal_tokens = object_record.get("nominal_native_elements_hex")
        if type(nominal_tokens) is not list or len(nominal_tokens) != 6:
            raise CartesianQualificationError("nominal native element vector differs")
        nominal = tuple(
            _float_from_canonical_hex(token, f"{object_id} nominal element {index}")
            for index, token in enumerate(nominal_tokens)
        )
        records.append(
            NativeElementRecord(object_id, spk_id, cohort, epoch, None, nominal)
        )
        clones = object_record.get("clones")
        if type(clones) is not list or len(clones) != 128:
            raise CartesianQualificationError("per-object clone roster differs")
        for expected_index, clone in enumerate(clones):
            if type(clone) is not dict:
                raise CartesianQualificationError("clone record must be an object")
            clone_index = clone.get("clone_index")
            if type(clone_index) is not int or clone_index != expected_index:
                raise CartesianQualificationError("clone index roster differs")
            tokens = clone.get("native_elements_hex")
            if type(tokens) is not list or len(tokens) != 6:
                raise CartesianQualificationError("clone native element vector differs")
            values = tuple(
                _float_from_canonical_hex(token, f"{object_id} clone {expected_index} element {index}")
                for index, token in enumerate(tokens)
            )
            records.append(
                NativeElementRecord(object_id, spk_id, cohort, epoch, clone_index, values)
            )
    if len(records) != 1_548 or sum(row.clone_index is None for row in records) != 12:
        raise CartesianQualificationError("native state count differs")
    return tuple(records)


def _native_vector_sha256(record: NativeElementRecord) -> str:
    value = {
        "object_id": record.object_id,
        "spk_id": record.spk_id,
        "covariance_epoch_jd_tdb_hex": record.covariance_epoch_jd_tdb.hex(),
        "clone_index": record.clone_index,
        "native_elements_hex": [value.hex() for value in record.values],
    }
    return _sha256_bytes(NATIVE_VECTOR_DOMAIN + _canonical_json(value))


def _validate_upstream() -> tuple[list[dict[str, object]], dict[str, Any], tuple[NativeElementRecord, ...]]:
    bindings = [_bound_file_record(row) for row in UPSTREAM_BINDINGS]
    pilot = _load_bound_module(
        "jx_tno_covariance_classification_pilot_for_stage0",
        _binding_for_role(UPSTREAM_BINDINGS, "pilot_implementation"),
    )
    try:
        pilot_protocol = pilot.validate_file(PILOT_PROTOCOL_PATH)
    except Exception as exc:
        raise CartesianQualificationError("sealed pilot reconstruction failed") from exc
    _require_exact(
        pilot_protocol.get("content_sha256"),
        PILOT_PROTOCOL_CONTENT_SHA256,
        "pilot protocol content digest",
    )
    clone_artifact = _load_canonical(CLONE_ARTIFACT_PATH)
    _require_exact(clone_artifact.get("content_sha256"), CLONE_ARTIFACT_CONTENT_SHA256, "clone artifact content digest")
    _require_exact(clone_artifact.get("protocol_content_sha256"), CLONE_PROTOCOL_CONTENT_SHA256, "clone protocol content digest")
    _require_exact(clone_artifact.get("binary_payload_sha256"), CLONE_BINARY_PAYLOAD_SHA256, "clone payload digest")
    _require_exact(clone_artifact.get("object_count"), 12, "clone object count")
    _require_exact(clone_artifact.get("total_clone_count"), 1_536, "clone state count")
    _require_exact(clone_artifact.get("execution_authorized"), False, "clone execution claim")
    records = _parse_native_records(clone_artifact)
    _require_exact([_bound_file_record(row) for row in UPSTREAM_BINDINGS], bindings, "upstream bindings after replay")
    return bindings, clone_artifact, records


def _validate_oracle() -> tuple[list[dict[str, object]], dict[str, Any], CentralGMSemantic]:
    if not ORACLE_BINDINGS or not ORACLE_CUSTODY_CONTENT_SHA256 or not ORACLE_IMPLEMENTATION_RELATIVE_PATH:
        raise CartesianQualificationError(
            "independent Horizons oracle and official central-GM custody are not yet bound"
        )
    bindings = [_bound_file_record(row) for row in ORACLE_BINDINGS]
    implementation_binding = _binding_for_role(ORACLE_BINDINGS, "oracle_implementation")
    _require_exact(
        implementation_binding[1],
        ORACLE_IMPLEMENTATION_RELATIVE_PATH,
        "oracle implementation path",
    )
    oracle = _load_bound_module(
        "jx_tno_native_epoch_horizons_oracle_for_stage0", implementation_binding
    )
    try:
        custody = oracle.validate_output(ORACLE_ROOT)
    except Exception as exc:
        raise CartesianQualificationError("independent Horizons oracle validation failed") from exc
    _require_exact(custody.get("content_sha256"), ORACLE_CUSTODY_CONTENT_SHA256, "oracle custody content digest")
    _require_exact(custody.get("protocol_content_sha256"), ORACLE_PROTOCOL_CONTENT_SHA256, "oracle protocol content digest")
    _require_exact(custody.get("record_roster_sha256"), ORACLE_RECORD_ROSTER_SHA256, "oracle record roster digest")
    _require_exact(custody.get("record_count"), 12, "oracle record count")
    _require_exact(custody.get("execution_authorized"), False, "oracle execution claim")
    _require_exact(custody.get("outcomes_generated"), False, "oracle outcome claim")
    gm_binding = _binding_for_role(ORACLE_BINDINGS, "official_horizons_gm")
    gm = _derive_official_gm(PROJECT_ROOT / gm_binding[1], gm_binding[3])
    _require_exact([_bound_file_record(row) for row in ORACLE_BINDINGS], bindings, "oracle bindings after replay")
    return bindings, custody, gm


def _oracle_vectors(custody: Mapping[str, Any]) -> dict[int, tuple[float, tuple[float, ...]]]:
    """Extract the 12 exact nominal vectors from the validated custody record."""

    if type(custody) is not dict:
        raise CartesianQualificationError("oracle custody must be a built-in dict")
    _require_exact(
        custody.get("schema"),
        "jx.tno.native-epoch-horizons-oracle.custody.v1",
        "oracle custody schema",
    )
    _require_exact(
        custody.get("classification"),
        "RETAINED_HORIZONS_NOMINAL_NATIVE_EPOCH_STATE_ORACLE_INPUT_ONLY",
        "oracle custody classification",
    )
    gm_source = custody.get("gm_source")
    expected_gm_source = {
        "official_url": "https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck",
        "retained_relative_path": "source/gm_Horizons.pck",
        "raw_byte_count": 15_428,
        "raw_sha256": "169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c",
        "body10_gm_km3_per_s2_decimal": HORIZONS_BODY10_GM_KM3_PER_S2_EXACT,
        "body10_gm_AU3_per_day2_binary64_hex": HORIZONS_BODY10_GM_AU3_PER_DAY2_HEX,
        "body10_gm_AU3_per_julian_year2_binary64_hex": HORIZONS_BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX,
        "conversion_constants": {
            "AU_km_decimal": KM_PER_AU_EXACT,
            "seconds_per_day": 86_400,
            "days_per_julian_year_decimal": str(JULIAN_YEAR_DAYS),
        },
    }
    _require_exact(gm_source, expected_gm_source, "oracle GM custody")
    records = custody.get("records")
    if type(records) is not list or len(records) != len(ORACLE_EXPECTED_RECORDS):
        raise CartesianQualificationError("oracle record roster differs")
    result: dict[int, tuple[float, tuple[float, ...]]] = {}
    for row, expected in zip(records, ORACLE_EXPECTED_RECORDS, strict=True):
        if type(row) is not dict:
            raise CartesianQualificationError("oracle record must be a built-in dict")
        object_id, spk_id, designation, solution_id, epoch_token = expected
        for key, expected_value in (
            ("object_id", object_id),
            ("spk_id", spk_id),
            ("primary_designation", designation),
            ("orbit_solution_id", solution_id),
            ("covariance_epoch_jd_tdb", epoch_token),
            ("center", "SUN_NAIF_10"),
            ("frame", "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000"),
            ("corrections", "GEOMETRIC_NONE"),
            ("planetary_ephemeris", "DE441"),
        ):
            _require_exact(row.get(key), expected_value, f"oracle {spk_id} {key}")
        try:
            epoch_decimal = Decimal(epoch_token)
        except Exception as exc:
            raise CartesianQualificationError("oracle epoch is not decimal") from exc
        if not epoch_decimal.is_finite():
            raise CartesianQualificationError("oracle epoch must be finite")
        epoch = float(epoch_decimal)
        vector: list[float] = []
        for decimal_key, hex_key in (
            ("position_AU_decimal_tokens", "position_AU_float_hex"),
            ("velocity_AU_per_day_decimal_tokens", "velocity_AU_per_day_float_hex"),
        ):
            decimal_tokens = row.get(decimal_key)
            hex_tokens = row.get(hex_key)
            if (
                type(decimal_tokens) is not list
                or type(hex_tokens) is not list
                or len(decimal_tokens) != 3
                or len(hex_tokens) != 3
            ):
                raise CartesianQualificationError(f"oracle {spk_id} vector shape differs")
            for index, (decimal_token, hex_token) in enumerate(
                zip(decimal_tokens, hex_tokens, strict=True)
            ):
                token = _text(decimal_token, f"oracle {spk_id} decimal component {index}")
                try:
                    decimal_value = Decimal(token)
                except Exception as exc:
                    raise CartesianQualificationError(
                        f"oracle {spk_id} vector component is not decimal"
                    ) from exc
                if not decimal_value.is_finite():
                    raise CartesianQualificationError(
                        f"oracle {spk_id} vector component is nonfinite"
                    )
                value = float(decimal_value)
                from_hex = _float_from_canonical_hex(
                    hex_token, f"oracle {spk_id} binary64 component {index}"
                )
                if value.hex() != from_hex.hex():
                    raise CartesianQualificationError(
                        f"oracle {spk_id} decimal and binary64 component differ"
                    )
                vector.append(value)
        if spk_id in result:
            raise CartesianQualificationError("oracle SPK roster contains a duplicate")
        result[spk_id] = (epoch, tuple(vector))
    if tuple(result) != tuple(row[1] for row in ORACLE_EXPECTED_RECORDS):
        raise CartesianQualificationError("oracle SPK order differs")
    return result


def _positive_binary64_ulp_distance(left: float, right: float) -> int:
    if (
        type(left) is not float
        or type(right) is not float
        or not math.isfinite(left)
        or not math.isfinite(right)
        or left <= 0.0
        or right <= 0.0
    ):
        raise CartesianQualificationError("ULP comparison requires positive finite floats")
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def _oracle_element_crosscheck(
    records: tuple[NativeElementRecord, ...],
    clone_artifact: Mapping[str, Any],
    custody: Mapping[str, Any],
) -> dict[str, object]:
    """Cross-bind Horizons' printed osculating elements to the SBDB mean."""

    nominal_records = {row.spk_id: row for row in records if row.clone_index is None}
    if len(nominal_records) != 12:
        raise CartesianQualificationError("nominal native element roster differs")
    clone_objects = clone_artifact.get("objects")
    oracle_records = custody.get("records")
    if (
        type(clone_objects) is not list
        or type(oracle_records) is not list
        or len(clone_objects) != 12
        or len(oracle_records) != 12
    ):
        raise CartesianQualificationError("element crosscheck roster differs")
    clone_by_spk = {
        row.get("spk_id"): row for row in clone_objects if type(row) is dict
    }
    oracle_by_spk = {
        row.get("spk_id"): row for row in oracle_records if type(row) is dict
    }
    if len(clone_by_spk) != 12 or len(oracle_by_spk) != 12:
        raise CartesianQualificationError("element crosscheck contains duplicate SPK IDs")

    tolerance_days = Decimal(ORACLE_TP_PRINTING_TOLERANCE_DAYS)
    maximum_tp_difference_days = Decimal(0)
    one_ulp_spk_ids: list[int] = []
    rows: list[dict[str, object]] = []
    for object_id, spk_id, _designation, _solution_id, epoch_token in ORACLE_EXPECTED_RECORDS:
        clone = clone_by_spk.get(spk_id)
        oracle = oracle_by_spk.get(spk_id)
        native = nominal_records.get(spk_id)
        if type(clone) is not dict or type(oracle) is not dict or native is None:
            raise CartesianQualificationError("element crosscheck target is absent")
        _require_exact(clone.get("object_id"), object_id, "clone element crosscheck object")
        _require_exact(oracle.get("object_id"), object_id, "oracle element crosscheck object")
        _require_exact(clone.get("covariance_epoch_jd_tdb"), epoch_token, "clone element crosscheck epoch")
        raw_sha256 = _sha256_text(clone.get("raw_sha256"), "SBDB raw response digest")
        raw_path = ACQUISITION_ROOT / "raw" / f"{spk_id}.json"
        raw_payload = _read_regular_bytes(raw_path)
        if _sha256_bytes(raw_payload) != raw_sha256:
            raise CartesianQualificationError("SBDB raw response differs during element crosscheck")
        raw = _strict_json(raw_payload)
        try:
            covariance = raw["orbit"]["covariance"]
            raw_labels = covariance["labels"]
            raw_epoch = covariance["epoch"]
            raw_elements = covariance["elements"]
        except (KeyError, TypeError) as exc:
            raise CartesianQualificationError("SBDB covariance element schema differs") from exc
        _require_exact(raw_labels, list(ELEMENT_LABELS), "SBDB covariance labels")
        _require_exact(raw_epoch, epoch_token, "SBDB covariance epoch")
        if type(raw_elements) is not list or len(raw_elements) != 6:
            raise CartesianQualificationError("SBDB covariance mean shape differs")
        sbdb_tokens: list[str] = []
        for index, raw_element in enumerate(raw_elements):
            if type(raw_element) is not dict:
                raise CartesianQualificationError("SBDB covariance element record differs")
            _require_exact(raw_element.get("label"), ELEMENT_LABELS[index], "SBDB element label")
            _require_exact(raw_element.get("units"), ELEMENT_UNITS[index] if index != 2 else "TDB", "SBDB element unit")
            sbdb_tokens.append(_text(raw_element.get("value"), "SBDB covariance mean token"))

        _require_exact(
            oracle.get("printed_osculating_element_labels"),
            list(ELEMENT_LABELS),
            "Horizons printed element labels",
        )
        _require_exact(
            oracle.get("printed_osculating_element_units"),
            [None, "au", "JD TDB", "deg", "deg", "deg"],
            "Horizons printed element units",
        )
        oracle_tokens = oracle.get("printed_osculating_element_decimal_tokens")
        oracle_hex = oracle.get("printed_osculating_element_float_hex")
        if (
            type(oracle_tokens) is not list
            or type(oracle_hex) is not list
            or len(oracle_tokens) != 6
            or len(oracle_hex) != 6
        ):
            raise CartesianQualificationError("Horizons printed element vector shape differs")
        sbdb_decimals: list[Decimal] = []
        oracle_decimals: list[Decimal] = []
        for index, (sbdb_token, oracle_token, oracle_hex_token) in enumerate(
            zip(sbdb_tokens, oracle_tokens, oracle_hex, strict=True)
        ):
            oracle_text = _text(oracle_token, "Horizons printed element token")
            try:
                sbdb_decimal = Decimal(sbdb_token)
                oracle_decimal = Decimal(oracle_text)
            except Exception as exc:
                raise CartesianQualificationError("element crosscheck token is not decimal") from exc
            if not sbdb_decimal.is_finite() or not oracle_decimal.is_finite():
                raise CartesianQualificationError("element crosscheck token is nonfinite")
            oracle_float = _float_from_canonical_hex(
                oracle_hex_token, f"Horizons printed element {index}"
            )
            if float(oracle_decimal).hex() != oracle_float.hex():
                raise CartesianQualificationError("Horizons printed decimal and binary64 differ")
            if float(sbdb_decimal).hex() != native.values[index].hex():
                raise CartesianQualificationError("SBDB decimal and sealed native element differ")
            sbdb_decimals.append(sbdb_decimal)
            oracle_decimals.append(oracle_decimal)

        for index in (0, 1, 3, 4, 5):
            if sbdb_decimals[index] != oracle_decimals[index]:
                raise CartesianQualificationError("SBDB and Horizons printed elements differ")
        tp_difference_days = abs(sbdb_decimals[2] - oracle_decimals[2])
        if tp_difference_days > tolerance_days:
            raise CartesianQualificationError("Horizons TP printing delta exceeds its frozen gate")
        tp_ulp_distance = _positive_binary64_ulp_distance(
            native.values[2], float(oracle_decimals[2])
        )
        if tp_ulp_distance > ORACLE_TP_PRINTING_MAXIMUM_ULP_DISTANCE:
            raise CartesianQualificationError("Horizons TP printing ULP distance exceeds its frozen gate")
        if tp_ulp_distance == 1:
            one_ulp_spk_ids.append(spk_id)
        maximum_tp_difference_days = max(maximum_tp_difference_days, tp_difference_days)
        rows.append(
            {
                "object_id": object_id,
                "spk_id": spk_id,
                "exact_decimal_equal_labels": ["e", "q", "node", "peri", "i"],
                "tp_absolute_printing_difference_days_decimal": str(tp_difference_days),
                "tp_binary64_ulp_distance": tp_ulp_distance,
            }
        )
    _require_exact(one_ulp_spk_ids, [20_574_372], "one-ULP TP roster")
    maximum_tp_difference_seconds = maximum_tp_difference_days * Decimal(86_400)
    _require_exact(
        str(maximum_tp_difference_seconds.normalize()),
        "0.0000187488",
        "maximum Horizons TP printing delta",
    )
    return {
        "records": rows,
        "all_e_q_node_peri_i_decimal_tokens_exactly_equal": True,
        "tp_printing_tolerance_days_decimal": ORACLE_TP_PRINTING_TOLERANCE_DAYS,
        "maximum_tp_printing_difference_days_decimal": str(maximum_tp_difference_days),
        "maximum_tp_printing_difference_seconds_decimal": str(
            maximum_tp_difference_seconds.normalize()
        ),
        "maximum_tp_binary64_ulp_distance": max(
            int(row["tp_binary64_ulp_distance"]) for row in rows
        ),
        "one_ulp_tp_spk_ids": one_ulp_spk_ids,
        "all_printed_element_crosschecks_pass": True,
    }


def _roundtrip_summary(
    records: tuple[NativeElementRecord, ...], states: tuple[CartesianState, ...], gm: CentralGMSemantic
) -> dict[str, str]:
    keys = (
        "semimajor_axis_relative",
        "perihelion_relative",
        "eccentricity_absolute",
        "inclination_rad",
        "ascending_node_rad",
        "argument_of_perihelion_rad",
        "mean_anomaly_rad",
    )
    maxima = {key: 0.0 for key in keys}
    for native, state in zip(records, states, strict=True):
        errors = validate_roundtrip(native, state, gm)
        for key in keys:
            maxima[key] = max(maxima[key], errors[key])
    return {f"maximum_{key}_hex": maxima[key].hex() for key in keys}


def _oracle_comparison(
    records: tuple[NativeElementRecord, ...],
    states: tuple[CartesianState, ...],
    custody: Mapping[str, Any],
) -> dict[str, object]:
    vectors = _oracle_vectors(custody)
    rows: list[dict[str, object]] = []
    maximum_position = 0.0
    maximum_velocity_day = 0.0
    for native, state in zip(records, states, strict=True):
        if native.clone_index is not None:
            continue
        oracle = vectors.get(native.spk_id)
        if oracle is None:
            raise CartesianQualificationError("nominal Horizons oracle is absent")
        epoch, vector = oracle
        if epoch != native.covariance_epoch_jd_tdb or len(vector) != 6:
            raise CartesianQualificationError("nominal Horizons oracle epoch or shape differs")
        position_error = max(abs(state.position_AU[index] - vector[index]) for index in range(3))
        velocity_error = max(
            abs(state.velocity_AU_per_julian_year[index] / JULIAN_YEAR_DAYS - vector[index + 3])
            for index in range(3)
        )
        maximum_position = max(maximum_position, position_error)
        maximum_velocity_day = max(maximum_velocity_day, velocity_error)
        rows.append(
            {
                "object_id": native.object_id,
                "spk_id": native.spk_id,
                "covariance_epoch_jd_tdb_hex": epoch.hex(),
                "maximum_absolute_position_component_error_AU_hex": position_error.hex(),
                "maximum_absolute_velocity_component_error_AU_per_day_hex": velocity_error.hex(),
            }
        )
    if len(rows) != 12:
        raise CartesianQualificationError("nominal oracle comparison count differs")
    if maximum_position > ORACLE_POSITION_COMPONENT_TOLERANCE_AU:
        raise CartesianQualificationError("Horizons nominal position comparison failed")
    if maximum_velocity_day > ORACLE_VELOCITY_COMPONENT_TOLERANCE_AU_PER_DAY:
        raise CartesianQualificationError("Horizons nominal velocity comparison failed")
    return {
        "records": rows,
        "maximum_absolute_position_component_error_AU_hex": maximum_position.hex(),
        "maximum_absolute_velocity_component_error_AU_per_day_hex": maximum_velocity_day.hex(),
        "position_component_tolerance_AU_hex": ORACLE_POSITION_COMPONENT_TOLERANCE_AU.hex(),
        "velocity_component_tolerance_AU_per_day_hex": ORACLE_VELOCITY_COMPONENT_TOLERANCE_AU_PER_DAY.hex(),
        "all_nominal_oracles_pass": True,
    }


def _state_record(native: NativeElementRecord, state: CartesianState) -> dict[str, object]:
    return {
        "object_id": native.object_id,
        "spk_id": native.spk_id,
        "cohort": native.cohort,
        "sample_kind": "NOMINAL" if native.clone_index is None else "COVARIANCE_CLONE",
        "clone_index": native.clone_index,
        "native_vector_sha256": _native_vector_sha256(native),
        "covariance_epoch_jd_tdb_hex": native.covariance_epoch_jd_tdb.hex(),
        "semimajor_axis_AU_hex": state.semimajor_axis_AU.hex(),
        "mean_anomaly_rad_hex": state.mean_anomaly_rad.hex(),
        "eccentric_anomaly_rad_hex": state.eccentric_anomaly_rad.hex(),
        "kepler_residual_rad_hex": state.kepler_residual_rad.hex(),
        "position_AU_hex": [value.hex() for value in state.position_AU],
        "velocity_AU_per_julian_year_hex": [
            value.hex() for value in state.velocity_AU_per_julian_year
        ],
    }


def _binary_payload_sha256(rows: list[dict[str, object]]) -> str:
    payload = bytearray()
    for row in rows:
        payload.extend(int(row["spk_id"]).to_bytes(8, "little", signed=False))
        index = -1 if row["clone_index"] is None else int(row["clone_index"])
        payload.extend(index.to_bytes(4, "little", signed=True))
        for key in (
            "covariance_epoch_jd_tdb_hex",
            "semimajor_axis_AU_hex",
            "mean_anomaly_rad_hex",
            "eccentric_anomaly_rad_hex",
            "kepler_residual_rad_hex",
        ):
            payload.extend(struct.pack("<d", float.fromhex(str(row[key]))))
        for key in ("position_AU_hex", "velocity_AU_per_julian_year_hex"):
            for token in row[key]:
                payload.extend(struct.pack("<d", float.fromhex(token)))
    return _sha256_bytes(BINARY_PAYLOAD_DOMAIN + bytes(payload))


def build_protocol() -> dict[str, object]:
    upstream_bindings, clone_artifact, records = _validate_upstream()
    oracle_bindings, oracle_custody, gm = _validate_oracle()
    states = tuple(native_elements_to_cartesian(record, gm) for record in records)
    roundtrip = _roundtrip_summary(records, states, gm)
    oracle_comparison = _oracle_comparison(records, states, oracle_custody)
    oracle_element_crosscheck = _oracle_element_crosscheck(
        records, clone_artifact, oracle_custody
    )
    record: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "STAGE_0_NATIVE_EPOCH_CARTESIAN_INPUT_QUALIFIED_NO_DYNAMICS",
        "claim_ceiling": "CARTESIAN_INITIAL_STATE_INPUT_QUALIFICATION_ONLY",
        "input_bindings": {
            "upstream_files": upstream_bindings,
            "oracle_files": oracle_bindings,
            "acquisition_protocol_content_sha256": ACQUISITION_PROTOCOL_CONTENT_SHA256,
            "acquisition_custody_content_sha256": ACQUISITION_CUSTODY_CONTENT_SHA256,
            "clone_protocol_content_sha256": CLONE_PROTOCOL_CONTENT_SHA256,
            "clone_artifact_content_sha256": CLONE_ARTIFACT_CONTENT_SHA256,
            "clone_binary_payload_sha256": CLONE_BINARY_PAYLOAD_SHA256,
            "pilot_protocol_content_sha256": PILOT_PROTOCOL_CONTENT_SHA256,
            "oracle_protocol_content_sha256": ORACLE_PROTOCOL_CONTENT_SHA256,
            "oracle_custody_content_sha256": ORACLE_CUSTODY_CONTENT_SHA256,
            "oracle_record_roster_sha256": ORACLE_RECORD_ROSTER_SHA256,
        },
        "input_contract": {
            "labels_in_exact_order": list(ELEMENT_LABELS),
            "units_in_exact_order": list(ELEMENT_UNITS),
            "tp_semantics": "ABSOLUTE_JD_TDB_WITH_COVARIANCE_DIFFERENTIALS_IN_DAYS",
            "horizons_printed_tp_caution": "HORIZONS_PRINTS_TP_TO_FEWER_DECIMAL_DIGITS_THAN_SBDB_FULL_PRECISION;_TRANSFORM_USES_THE_SEALED_SBDB_NATIVE_TP",
            "epoch_semantics": "OBJECT_SPECIFIC_NATIVE_COVARIANCE_EPOCH_JD_TDB",
            "element_frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
            "ecliptic_obliquity_arcseconds_exact": "84381.448",
            "output_frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_CARTESIAN",
            "output_position_unit": "AU",
            "output_velocity_unit": "AU_PER_JULIAN_YEAR_365.25_D",
            "nominal_state_count": 12,
            "clone_state_count": 1_536,
            "total_state_count": 1_548,
            "clone_artifact_classification": clone_artifact["classification"],
            "clone_interpretation": "LOCAL_FORMAL_COVARIANCE_GAUSSIAN_SENSITIVITY_DRAWS_NOT_POSTERIOR_OR_LIKELIHOOD_SAMPLES",
            "clone_count_caution": "128_CLONES_PER_OBJECT_DOES_NOT_ESTABLISH_TAIL_PROBABILITIES_OR_POPULATION_PRECISION",
        },
        "central_gm": {
            "semantic_id": gm.semantic_id,
            "primary_id": gm.primary_id,
            "official_source_relative_path": gm.source_relative_path,
            "official_source_file_sha256": gm.source_file_sha256,
            "repo_de440_crosscheck_relative_path": "runs/v5_solar_1pn_qualification/sources/gm_de440.tpc",
            "repo_de440_crosscheck_file_sha256": "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140",
            "official_and_repo_BODY10_tokens_exactly_equal": True,
            "source_token_km3_per_s2": gm.source_token_km3_per_s2,
            "exact_decimal_km3_per_s2": HORIZONS_BODY10_GM_KM3_PER_S2_EXACT,
            "AU_km_exact": KM_PER_AU_EXACT,
            "seconds_per_day_exact": SECONDS_PER_DAY_EXACT,
            "seconds_per_julian_year_exact": SECONDS_PER_JULIAN_YEAR_EXACT,
            "gm_AU3_per_day2_binary64_hex": HORIZONS_BODY10_GM_AU3_PER_DAY2_HEX,
            "gm_AU3_per_julian_year2_binary64_hex": gm.gm_AU3_per_julian_year2.hex(),
            "inner_system_monopole_used": False,
        },
        "references": [
            "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
            "https://ssd-api.jpl.nasa.gov/doc/horizons.html",
            "https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck",
        ],
        "downstream_cautions": {
            "two_body_scope": "THIS_ADAPTER_ONLY_CONVERTS_OSCULATING_ELEMENTS_TO_INITIAL_CARTESIAN_STATES; IT DOES_NOT QUALIFY_N_BODY_PROPAGATION",
            "planetary_ephemeris_scope": "THE_DE441_ORBIT_FIT_AND_HORIZONS_STATE_ORACLE_DO_NOT_BIND_OR_QUALIFY_FUTURE_MASSIVE_BODY_INITIAL_STATES",
            "de440_de441_bridge": "A_FUTURE_DE440_OR_DE440S_PLANETARY_SOURCE_REMAINS_A_MODEL_MISMATCH_REQUIRING_THE_PILOT_PROTOCOL_SENSITIVITY_BRIDGE",
            "oracle_authentication": "RETAINED_HORIZONS_BYTES_ARE_SCHEMA_AND_TEXT_CONFORMANT_BUT_OFFLINE_CUSTODY_DOES_NOT_CRYPTOGRAPHICALLY_PROVE_PROVIDER_ORIGIN_OR_TRANSPORT",
        },
        "algorithm": {
            "semimajor_axis": "a=q/(1-e)",
            "mean_motion": "n=sqrt(GM/a^3)_RAD_PER_JULIAN_YEAR",
            "phase": "M=n*(covariance_epoch_JD_TDB-tp_JD_TDB)/365.25_REDUCED_WITH_IEEE_REMAINDER",
            "kepler_equation": "E-e*sin(E)=M",
            "kepler_solver": "SAFEGUARDED_NEWTON_WITH_BISECTION_ELLIPTIC_BINARY64_V1",
            "kepler_maximum_iterations": KEPLER_MAXIMUM_ITERATIONS,
            "kepler_residual_tolerance_rad_hex": KEPLER_RESIDUAL_TOLERANCE.hex(),
            "perifocal_to_ecliptic_rotation": "R3(ASCENDING_NODE)*R1(INCLINATION)*R3(ARGUMENT_OF_PERIHELION)",
            "binary64_rounding": "IEEE_754_ROUND_TO_NEAREST_TIES_TO_EVEN_VIA_CPYTHON_FLOAT_AND_LIBM",
        },
        "qualification": {
            "roundtrip": roundtrip,
            "roundtrip_tolerances": {
                "semimajor_axis_relative_hex": ROUNDTRIP_A_RELATIVE_TOLERANCE.hex(),
                "perihelion_relative_hex": ROUNDTRIP_Q_RELATIVE_TOLERANCE.hex(),
                "eccentricity_absolute_hex": ROUNDTRIP_E_ABSOLUTE_TOLERANCE.hex(),
                "angle_rad_hex": ROUNDTRIP_ANGLE_TOLERANCE_RAD.hex(),
            },
            "independent_horizons_nominal_oracles": oracle_comparison,
            "sbdb_to_horizons_printed_element_crosscheck": oracle_element_crosscheck,
            "all_1548_inverse_roundtrips_pass": True,
        },
        "implementation": {
            "relative_path": "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py",
            "file_sha256": _sha256_bytes(_read_regular_bytes(SCRIPT_PATH)),
            "runtime_surface": "CPU_QUALIFICATION_AND_CREATE_ONLY_STATE_MATERIALIZATION",
        },
        "claims": {
            "official_horizons_central_gm_bound": True,
            "native_epoch_cartesian_transform_qualified": True,
            "cartesian_input_states_materialized": True,
            "independent_nominal_horizons_oracles_passed": True,
            "sbdb_horizons_printed_elements_crosschecked": True,
            "retained_sbdb_schema_conformant_covariance_bytes_revalidated": True,
            "retained_oracle_schema_and_text_contract_validated": True,
            "retained_oracle_provider_transport_authenticated": False,
            "retained_oracle_provider_origin_cryptographically_proven": False,
            "clone_dynamics_executed": False,
            "one_thousand_year_holdout_executed": False,
            "gpu_executed": False,
            "dynamical_classifications_generated": False,
            "long_horizon_execution_authorized": False,
            "observational_likelihood_evaluated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": "These are qualified two-body Cartesian initial states only, not trajectories, dynamical classes, population inference, or Planet Nine evidence.",
    }
    record["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, record)
    return record


def build_states(protocol: Mapping[str, Any]) -> dict[str, object]:
    validate_protocol(protocol)
    _upstream, _clone_artifact, records = _validate_upstream()
    _oracle_bindings, _oracle_custody, gm = _validate_oracle()
    states = tuple(native_elements_to_cartesian(record, gm) for record in records)
    rows = [_state_record(native, state) for native, state in zip(records, states, strict=True)]
    artifact: dict[str, object] = {
        "schema": STATES_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "TWELVE_NOMINAL_PLUS_1536_COVARIANCE_CLONE_CARTESIAN_INPUT_STATES_NO_DYNAMICS",
        "claim_ceiling": "CARTESIAN_INITIAL_STATE_INPUT_QUALIFICATION_ONLY",
        "protocol_content_sha256": protocol["content_sha256"],
        "clone_artifact_content_sha256": CLONE_ARTIFACT_CONTENT_SHA256,
        "central_gm_semantic_id": gm.semantic_id,
        "central_gm_AU3_per_julian_year2_hex": gm.gm_AU3_per_julian_year2.hex(),
        "object_count": 12,
        "nominal_state_count": 12,
        "clone_state_count": 1_536,
        "total_state_count": len(rows),
        "states": rows,
        "binary_payload_sha256": _binary_payload_sha256(rows),
        "claims": dict(protocol["claims"]),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    artifact["content_sha256"] = _content_sha256(STATES_DOMAIN, artifact)
    return artifact


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or not _exact_equal(value, expected):
        raise CartesianQualificationError("protocol differs from exact reconstruction")
    digest = _content_sha256(PROTOCOL_DOMAIN, value)
    _require_exact(value.get("content_sha256"), digest, "protocol content digest")
    _require_exact(value.get("execution_authorized"), False, "protocol execution claim")
    _require_exact(value.get("outcomes_generated"), False, "protocol outcome claim")
    return digest


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CartesianQualificationError(f"artifact path is not create-only: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def materialize(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink():
        raise CartesianQualificationError("output root cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir() or output_root.is_symlink() or any(output_root.iterdir()):
        raise CartesianQualificationError("dedicated output root must be a real empty directory")
    protocol = build_protocol()
    artifact = build_states(protocol)
    _write_create_only(output_root / "PROTOCOL.json", _canonical_json(protocol))
    _write_create_only(output_root / "STATES.json", _canonical_json(artifact))
    validate_output(output_root)
    return artifact


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise CartesianQualificationError("output root must be a real directory")
    if {entry.name for entry in output_root.iterdir()} != {"PROTOCOL.json", "STATES.json"}:
        raise CartesianQualificationError("output root roster differs")
    protocol = _load_canonical(output_root / "PROTOCOL.json")
    validate_protocol(protocol)
    actual = _load_canonical(output_root / "STATES.json")
    expected = build_states(protocol)
    if not _exact_equal(actual, expected):
        raise CartesianQualificationError("state artifact differs from exact reconstruction")
    _require_exact(actual.get("content_sha256"), _content_sha256(STATES_DOMAIN, actual), "state artifact content digest")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualify-only",
        action="store_true",
        help="qualify and materialize initial states; never integrate dynamics",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.qualify_only:
        parser.error("--qualify-only is required; this tool has no dynamics mode")
    try:
        artifact = materialize(args.output_root)
    except CartesianQualificationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
