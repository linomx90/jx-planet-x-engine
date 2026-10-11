#!/usr/bin/env python3
"""Prepare and validate the full-128-clone DE441 reduced-five TNO bridge.

This module is deliberately limited to deterministic input preparation.  It
does not integrate, classify, authorize a ten-million-year run, or make a
scientific claim.  The retained artifact contains twelve independent native-
epoch batches, each with five massive bodies and the nominal state followed by
clone 0 through clone 127 for one TNO.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE = "benchmarks/jx_tno_full128_reduced5_bridge_v1.py"
TEST_RELATIVE = "tests/test_tno_full128_reduced5_bridge_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_full128_reduced5_bridge_v1"

PROTOCOL_SCHEMA = "jx.tno.full128-reduced5-bridge.protocol.v1"
BRIDGE_SCHEMA = "jx.tno.full128-reduced5-bridge.states.v1"
PROTOCOL_DOMAIN = b"jx.tno.full128-reduced5-bridge.protocol.v1\0"
BRIDGE_DOMAIN = b"jx.tno.full128-reduced5-bridge.states.v1\0"
BATCH_DOMAIN = b"jx.tno.full128-reduced5-bridge.batch.v1\0"
BINARY_DOMAIN = b"jx.tno.full128-reduced5-bridge.binary.v1\0"
PROJECTION_DOMAIN = b"jx.tno.full128-reduced5-bridge.projection.v1\0"
CLASSIFICATION = "DE441_NATIVE_EPOCH_FULL128_REDUCED5_PREPARE_ONLY_INPUT_BRIDGE"

AU_METRES_EXACT = 149_597_870_700
SECONDS_PER_JULIAN_YEAR_EXACT = 31_557_600
OBLIQUITY_ARCSECONDS_EXACT = "84381.448"
OBLIQUITY_RADIANS_HEX = "0x1.a2e9395ca7c40p-2"
OBLIQUITY_SIN_HEX = "0x1.9752e50f4b399p-2"
OBLIQUITY_COS_HEX = "0x1.d5c0357681ef3p-1"
PI_DECIMAL_120 = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
    "208998628034825342117067982148086513282306647"
)

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
RESOLVED11_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
REDUCED5_NAIF_IDS = (10, 5, 6, 7, 8)
REDUCED5_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER", "SATURN_BARYCENTER", "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
PROJECTED_TRACER_INDICES = (0, 1, 19, 37, 55, 74, 92, 110, 128)
EXPECTED_OUTPUT_ROSTER = frozenset({"PROTOCOL.json", "BRIDGE.json"})
STATE_ROW_COUNT = 5 + 129
STATE_RECORD_BYTES = STATE_ROW_COUNT * 6 * 8
V3_CHECKPOINT_COUNT_PER_BATCH = 21
V3_CHECKPOINT_BYTES = 14 * 6 * 8
EXPECTED_BATCH0_DIRECT_BARYCENTRIC_SHA256 = "2662b11dafe2acd679d1a91f8902abdc941ffb0c458cd6a22f5cdc3182eea7aa"
EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256 = "25394a0f2a31caf3c9b47f593ec0511806b94a0ac140d1d44fc9e1939656e00c"

STAGE0_BINDINGS = (
    ("stage0_source", "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", 70_064, "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367"),
    ("stage0_test", "tests/test_tno_native_epoch_cartesian_qualification_v1.py", 15_944, "1225cc19a78aa4ee412c11bf42e931ac8b6881c41845ad9b359c8ff547cb9d0c"),
    ("stage0_protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"),
    ("stage0_states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
)
STAGE0_PROTOCOL_CONTENT_SHA256 = "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e"
STAGE0_STATES_CONTENT_SHA256 = "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b"
STAGE0_BINARY_PAYLOAD_SHA256 = "334d3fb579e3be752e44cee91680a0c73bd4ac2cb5ab2bff108d66c389118b83"

DE441_BINDINGS = (
    ("de441_acquisition_source", "tools/jx_tno_de441_resolved11_acquisition_v1.py", 37_145, "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00"),
    ("de441_acquisition_test", "tests/test_tno_de441_resolved11_acquisition_v1.py", 14_556, "220215bae324d8119d03417f44997920b7a4304fb59eae19a375d94a8555d8b3"),
    ("de441_protocol", "runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", 11_986, "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b"),
    ("de441_custody", "runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", 141_501, "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"),
)
DE441_PROTOCOL_CONTENT_SHA256 = "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e"
DE441_CUSTODY_CONTENT_SHA256 = "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3"

LEGACY_BRIDGE_BINDINGS = (
    ("selected_bridge_source", "benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py", 81_161, "5cb0b5380e7eac3e1c3c5932715dcbe43372b03581e65ef981ba491e78087e7e"),
    ("selected_bridge_test", "tests/test_tno_native_epoch_resolved11_bridge_v1.py", 13_008, "ad52731288e89e1425cc315bcbcdef705144a4bf22fa92f9a2c78576ae4a3497"),
    ("selected_bridge_protocol", "runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", 11_386, "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459"),
    ("selected_bridge_receipts", "runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", 5_810_062, "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5"),
    ("selected_bridge_states", "runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", 168_684, "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"),
)
LEGACY_BRIDGE_CONTENT_SHA256 = "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13"

COLLAPSE_BINDINGS = (
    ("stage1_v4_collapse_source", "benchmarks/jx_tno_stage1_dynamics_qualification_v4.py", 436_125, "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"),
    ("stage1_v4_collapse_test", "tests/test_tno_stage1_dynamics_qualification_v4.py", 89_619, "dc71007054e9afd1cb4d4182de253ac17f495ff835cc6b8677cdf2f36b5c2f8d"),
)

V3_BINDINGS = (
    ("v3_source", "benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v3.py", 151_995, "02e72fb6a635b74dd31dc364e18cbaf03c5b9e1833caf5aba4a063d7198e47c7"),
    ("v3_test", "tests/test_tno_1kyr_reduced5_lifecycle_gate_v3.py", 42_017, "b4eb43b00e1fab0055a43cc4f26db2bef6bf29d51656bcfab1d792596b7845a8"),
    ("v3_prefix", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin", 169_344, "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b"),
    ("v3_protocol", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PROTOCOL.json", 9_178, "9c1e7f77b3a2fd9f6af5542aa2ee3fa7dc00d669d496c809da28b675c8519199"),
    ("v3_restart", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESTART_STREAMS.bin", 9_504, "3330f6d83b71f32c2a0fb14bc9f2e26ab13def77f1ea17527d69552473c80173"),
    ("v3_result", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json", 66_669, "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4"),
    ("v3_routes", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/ROUTES.jsonl", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("v3_runtime", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME.json", 12_980, "43bc1edba432bed75f8a9eb1db9784c13116e1a93e20bb4b43cb634d6abd1dcd"),
    ("v3_runtime_preflight", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME_PREFLIGHT.json", 26_179, "7df113a70bcb1012ea05b2145f141bd8b788c504fc61c0f2b075671d0810ea80"),
    ("v3_segments", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/SEGMENTS.jsonl", 239_895, "e9452debe803a7213d0878c6c2bd35199bfe966e7e6891f12d072113c825f64f"),
)
V3_PROTOCOL_CONTENT_SHA256 = "538e076274ab5e1c850cce069405ba67c3cab41226fa684a686aaaa81cdc32d9"
V3_RESULT_CONTENT_SHA256 = "d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264"
V3_GIT = {
    "annotated_tag": "tno-1kyr-reduced5-lifecycle-gate-v3-preregistered-20261011",
    "branch": "research/tno-1kyr-reduced5-lifecycle-gate-v3-preregistered-20261011",
    "claim": "EXTERNAL_GITHUB_CUSTODY_RECORDED_NOT_RECONSTRUCTED_BY_RESULT_VALIDATOR",
    "commit": "9ec5151e5a979e46a516ec7efdf7c6e4b90da620",
    "peeled_commit": "9ec5151e5a979e46a516ec7efdf7c6e4b90da620",
    "remote_verified_immediately_before_execution": True,
    "tag_object": "94297db889f5b5ba2f4d9eaef2adfa792a3740cd",
}

CLAIMS = {
    "artifact_prepared": True,
    "dynamics_executed": False,
    "classification_generated": False,
    "resonance_classification_generated": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "observational_likelihood_evaluated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class FullBridgeError(ValueError):
    """A frozen input, numerical, schema, or publication invariant failed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(payload))


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise FullBridgeError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FullBridgeError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullBridgeError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise FullBridgeError("JSON root must be an exact object")
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise FullBridgeError(f"{label} differs from the frozen contract")


def _require_int(value: object, expected: int | None, label: str) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        raise FullBridgeError(f"{label} must be an exact integer")
    return value


def _require_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise FullBridgeError(f"{label} must be exact Boolean {expected}")


def _require_digest(value: object, label: str) -> str:
    if (
        type(value) is not str or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullBridgeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _hex_float(value: object, label: str) -> float:
    if type(value) is not str:
        raise FullBridgeError(f"{label} must be a canonical binary64 hex string")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise FullBridgeError(f"{label} is not binary64 hex") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise FullBridgeError(f"{label} is not canonical finite binary64 hex")
    return number


def _vector(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise FullBridgeError(f"{label} must be an exact three-vector")
    return tuple(_hex_float(item, f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
        stat_result.st_mtime_ns, stat_result.st_ctime_ns, stat_result.st_nlink,
    )


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if (
        type(relative) is not str or not relative or path.is_absolute()
        or ".." in path.parts or "." in path.parts
        or any(not part for part in path.parts)
    ):
        raise FullBridgeError("project-relative path is unsafe")
    return path.parts


def _read_descriptor(descriptor: int, label: str) -> tuple[bytes, tuple[int, ...]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise FullBridgeError(f"{label} is nonregular or multiply linked")
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            break
        chunks.append(block)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after) or len(payload) != before.st_size:
        raise FullBridgeError(f"{label} changed while held and read")
    return payload, _identity(after)


def _read_project(relative: str) -> bytes:
    parts = _relative_parts(relative)
    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(PROJECT_ROOT, flags_dir)
        descriptors.append(descriptor)
        for part in parts[:-1]:
            descriptor = os.open(part, flags_dir, dir_fd=descriptor)
            descriptors.append(descriptor)
        file_descriptor = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        try:
            payload, _observed = _read_descriptor(file_descriptor, relative)
        finally:
            os.close(file_descriptor)
        return payload
    except OSError as exc:
        raise FullBridgeError(f"project input cannot be held: {relative}") from exc
    finally:
        for item in reversed(descriptors):
            os.close(item)


def _binding(role: str, relative: str, size: int, digest: str) -> dict[str, object]:
    payload = _read_project(relative)
    if len(payload) != size or _sha256(payload) != digest:
        raise FullBridgeError(f"bound input bytes differ: {relative}")
    return {
        "role": role, "relative_path": relative, "bytes": size,
        "file_sha256": digest,
    }


def _bindings(rows: Sequence[tuple[str, str, int, str]]) -> list[dict[str, object]]:
    return [_binding(*row) for row in rows]


def _canonical_bound_json(relative: str) -> dict[str, Any]:
    payload = _read_project(relative)
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise FullBridgeError(f"bound JSON is not canonical: {relative}")
    return value


def _decimal_sin(value: Decimal) -> Decimal:
    term = value
    total = term
    index = 1
    while True:
        term = -term * value * value / Decimal((2 * index) * (2 * index + 1))
        updated = total + term
        if updated == total:
            return updated
        total = updated
        index += 1


def _decimal_cos(value: Decimal) -> Decimal:
    term = Decimal(1)
    total = term
    index = 1
    while True:
        term = -term * value * value / Decimal((2 * index - 1) * (2 * index))
        updated = total + term
        if updated == total:
            return updated
        total = updated
        index += 1


def _obliquity() -> tuple[float, float, float]:
    with localcontext() as context:
        context.prec = 110
        radians = Decimal(OBLIQUITY_ARCSECONDS_EXACT) * Decimal(PI_DECIMAL_120) / (
            Decimal(180) * Decimal(3600)
        )
        values = (float(radians), float(_decimal_sin(radians)), float(_decimal_cos(radians)))
    _require(
        tuple(item.hex() for item in values),
        (OBLIQUITY_RADIANS_HEX, OBLIQUITY_SIN_HEX, OBLIQUITY_COS_HEX),
        "independently rounded obliquity",
    )
    return values


def _round_fraction(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FullBridgeError("exact dyadic arithmetic overflowed binary64")
    return result


def _rotate_ecliptic_to_j2000(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    if type(vector) is not tuple or len(vector) != 3 or any(
        type(item) is not float or not math.isfinite(item) for item in vector
    ):
        raise FullBridgeError("rotation input must be an immutable finite binary64 triplet")
    _radians, sine, cosine = _obliquity()
    x, y, z = vector
    exact_sine = Fraction.from_float(sine)
    exact_cosine = Fraction.from_float(cosine)
    exact_y = Fraction.from_float(y)
    exact_z = Fraction.from_float(z)
    return (
        x,
        _round_fraction(exact_cosine * exact_y - exact_sine * exact_z),
        _round_fraction(exact_sine * exact_y + exact_cosine * exact_z),
    )


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _gm_si_to_au_year(value: float) -> float:
    result = value * (SECONDS_PER_JULIAN_YEAR_EXACT**2) / (AU_METRES_EXACT**3)
    if not math.isfinite(result) or not result > 0.0:
        raise FullBridgeError("converted GM is not positive finite binary64")
    return result


def _weighted_centroid(
    gms: Sequence[float],
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not gms or len(gms) != len(positions) or len(gms) != len(velocities):
        raise FullBridgeError("centroid roster differs")
    if any(type(item) is not float or not math.isfinite(item) or not item > 0.0 for item in gms):
        raise FullBridgeError("centroid GM roster differs")
    total = math.fsum(gms)
    position = tuple(
        math.fsum(gms[index] * positions[index][axis] for index in range(len(gms))) / total
        for axis in range(3)
    )
    velocity = tuple(
        math.fsum(gms[index] * velocities[index][axis] for index in range(len(gms))) / total
        for axis in range(3)
    )
    if any(not math.isfinite(item) for item in (*position, *velocity)):
        raise FullBridgeError("centroid produced a non-finite component")
    return position, velocity  # type: ignore[return-value]


def _validate_claims(value: object, *, prepared: bool, label: str) -> None:
    expected = dict(CLAIMS)
    expected["artifact_prepared"] = prepared
    if type(value) is not dict or set(value) != set(expected):
        raise FullBridgeError(f"{label} claim roster differs")
    for key, expected_value in expected.items():
        _require_bool(value.get(key), expected_value, f"{label}.{key}")


def _load_stage0() -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    states = _canonical_bound_json(STAGE0_BINDINGS[-1][1])
    _require(states.get("schema"), "jx.tno.native-epoch-cartesian.states.v1", "Stage0 schema")
    _require(states.get("content_sha256"), STAGE0_STATES_CONTENT_SHA256, "Stage0 content digest")
    _require(states.get("protocol_content_sha256"), STAGE0_PROTOCOL_CONTENT_SHA256, "Stage0 protocol digest")
    _require(states.get("binary_payload_sha256"), STAGE0_BINARY_PAYLOAD_SHA256, "Stage0 binary digest")
    _require_int(states.get("object_count"), 12, "Stage0 object_count")
    _require_int(states.get("nominal_state_count"), 12, "Stage0 nominal_state_count")
    _require_int(states.get("clone_state_count"), 1_536, "Stage0 clone_state_count")
    _require_int(states.get("total_state_count"), 1_548, "Stage0 total_state_count")
    rows = states.get("states")
    if type(rows) is not list or len(rows) != 1_548:
        raise FullBridgeError("Stage0 state roster differs")
    groups: list[list[dict[str, Any]]] = []
    for batch_order, object_id in enumerate(OBJECT_IDS):
        group = rows[batch_order * 129 : (batch_order + 1) * 129]
        if len(group) != 129:
            raise FullBridgeError("Stage0 group width differs")
        epoch: str | None = None
        spk_id: int | None = None
        for tracer_order, row in enumerate(group):
            if type(row) is not dict:
                raise FullBridgeError("Stage0 row is not an exact object")
            _require(row.get("object_id"), object_id, "Stage0 object order")
            observed_spk = _require_int(row.get("spk_id"), None, "Stage0 spk_id")
            if spk_id is None:
                spk_id = observed_spk
            _require(observed_spk, spk_id, "Stage0 per-object spk_id")
            expected_clone = None if tracer_order == 0 else tracer_order - 1
            expected_kind = "NOMINAL" if tracer_order == 0 else "COVARIANCE_CLONE"
            if expected_clone is None:
                _require(row.get("clone_index"), None, "Stage0 nominal clone identity")
            else:
                _require_int(row.get("clone_index"), expected_clone, "Stage0 clone order")
            _require(row.get("sample_kind"), expected_kind, "Stage0 sample order")
            _require_digest(row.get("native_vector_sha256"), "Stage0 native vector digest")
            _vector(row.get("position_AU_hex"), "Stage0 position")
            _vector(row.get("velocity_AU_per_julian_year_hex"), "Stage0 velocity")
            current_epoch = row.get("covariance_epoch_jd_tdb_hex")
            _hex_float(current_epoch, "Stage0 native epoch")
            if epoch is None:
                epoch = current_epoch
            _require(current_epoch, epoch, "Stage0 per-object epoch")
        groups.append(group)
    return states, groups


def _load_custody(groups: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    custody = _canonical_bound_json(DE441_BINDINGS[-1][1])
    _require(custody.get("schema"), "jx.tno.de441-resolved11-native-epochs.custody.v1", "DE441 schema")
    _require(custody.get("content_sha256"), DE441_CUSTODY_CONTENT_SHA256, "DE441 content digest")
    _require(custody.get("protocol_content_sha256"), DE441_PROTOCOL_CONTENT_SHA256, "DE441 protocol digest")
    _require_int(custody.get("epoch_count"), 12, "DE441 epoch_count")
    _require_int(custody.get("body_count"), 11, "DE441 body_count")
    _require_int(custody.get("state_record_count"), 132, "DE441 state_record_count")
    _require(custody.get("state_record_order"), "EPOCH_MAJOR_THEN_RESOLVED11_BODY_ORDER", "DE441 record order")
    rows = custody.get("state_records")
    if type(rows) is not list or len(rows) != 132:
        raise FullBridgeError("DE441 state roster differs")
    batches: list[list[dict[str, Any]]] = []
    for batch_order, object_id in enumerate(OBJECT_IDS):
        batch = rows[batch_order * 11 : (batch_order + 1) * 11]
        source = groups[batch_order][0]
        for body_order, row in enumerate(batch):
            if type(row) is not dict:
                raise FullBridgeError("DE441 state row is not an exact object")
            _require_int(row.get("state_order"), batch_order * 11 + body_order, "DE441 state order")
            _require_int(row.get("epoch_order"), batch_order, "DE441 epoch order")
            _require_int(row.get("body_order"), body_order, "DE441 body order")
            _require_int(row.get("naif_id"), RESOLVED11_NAIF_IDS[body_order], "DE441 NAIF order")
            _require(row.get("object_id"), object_id, "DE441 object order")
            _require_int(row.get("spk_id"), source["spk_id"], "DE441 spk_id")
            _require(row.get("frame"), "ICRF", "DE441 frame")
            _require(row.get("corrections"), "GEOMETRIC_NONE", "DE441 corrections")
            _require_int(row.get("center_naif_id"), 0, "DE441 center")
            requested = row.get("requested_jd_tdb")
            if type(requested) is not str:
                raise FullBridgeError("DE441 requested epoch token differs")
            if float(requested).hex() != source["covariance_epoch_jd_tdb_hex"]:
                raise FullBridgeError("DE441 epoch differs from Stage0")
            _vector(row.get("position_AU_binary64_hex"), "DE441 position")
            _vector(row.get("velocity_AU_per_day_binary64_hex"), "DE441 velocity")
        batches.append(batch)
    return custody, batches


def _load_legacy_bridge() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge = _canonical_bound_json(LEGACY_BRIDGE_BINDINGS[-1][1])
    _require(bridge.get("schema"), "jx.tno.native-epoch-resolved11-bridge.states.v1", "selected bridge schema")
    _require(bridge.get("content_sha256"), LEGACY_BRIDGE_CONTENT_SHA256, "selected bridge content digest")
    _require_int(bridge.get("batch_count"), 12, "selected bridge batch_count")
    batches = bridge.get("batches")
    if type(batches) is not list or len(batches) != 12:
        raise FullBridgeError("selected bridge batch roster differs")
    for batch_order, batch in enumerate(batches):
        if type(batch) is not dict:
            raise FullBridgeError("selected bridge batch is not an exact object")
        _require(batch.get("object_id"), OBJECT_IDS[batch_order], "selected bridge object order")
        _require_int(batch.get("massive_body_count"), 11, "selected bridge massive count")
        _require_int(batch.get("tno_state_count"), 9, "selected bridge TNO count")
        massive = batch.get("massive_bodies")
        tnos = batch.get("tnos")
        if type(massive) is not list or len(massive) != 11 or type(tnos) is not list or len(tnos) != 9:
            raise FullBridgeError("selected bridge batch roster differs")
        for index, row in enumerate(massive):
            if type(row) is not dict:
                raise FullBridgeError("selected bridge massive row differs")
            _require_int(row.get("naif_id"), RESOLVED11_NAIF_IDS[index], "selected bridge NAIF order")
            gm = _hex_float(row.get("gravitational_parameter_m3_per_s2_hex"), "selected bridge GM")
            if not gm > 0.0:
                raise FullBridgeError("selected bridge GM must be positive")
        for index, row in enumerate(tnos):
            if type(row) is not dict:
                raise FullBridgeError("selected bridge TNO row differs")
            expected_clone = None if index == 0 else HOLDOUT_CLONE_INDICES[index - 1]
            if expected_clone is None:
                _require(row.get("clone_index"), None, "selected bridge nominal clone identity")
            else:
                _require_int(row.get("clone_index"), expected_clone, "selected bridge clone order")
            _require_digest(row.get("native_vector_sha256"), "selected bridge native vector digest")
            _vector(row.get("heliocentric_j2000_position_AU_hex"), "selected bridge heliocentric position")
            _vector(row.get("heliocentric_j2000_velocity_AU_per_julian_year_hex"), "selected bridge heliocentric velocity")
    return bridge, batches


def _load_v3_lineage() -> dict[str, object]:
    root = PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v3"
    if root.is_symlink() or not root.is_dir():
        raise FullBridgeError("V3 retained root is unavailable or linked")
    actual = {item.name for item in root.iterdir()}
    expected = {PurePosixPath(row[1]).name for row in V3_BINDINGS[2:]}
    if actual != expected or len(actual) != 8:
        raise FullBridgeError("V3 retained root roster differs")
    protocol = _canonical_bound_json("runs/tno_1kyr_reduced5_lifecycle_gate_v3/PROTOCOL.json")
    result = _canonical_bound_json("runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json")
    _require(protocol.get("content_sha256"), V3_PROTOCOL_CONTENT_SHA256, "V3 protocol content digest")
    _require(result.get("content_sha256"), V3_RESULT_CONTENT_SHA256, "V3 result content digest")
    _require(result.get("protocol_content_sha256"), V3_PROTOCOL_CONTENT_SHA256, "V3 result/protocol binding")
    _require(result.get("git_preregistration"), V3_GIT, "V3 Git lineage")
    claims = result.get("claims")
    if type(claims) is not dict:
        raise FullBridgeError("V3 result claims differ")
    _require_bool(claims.get("dynamics_executed"), True, "V3 dynamics claim")
    _require_bool(claims.get("one_thousand_year_lifecycle_gate_passed"), True, "V3 lifecycle claim")
    for name in (
        "ten_myr_execution_authorized", "ten_myr_execution_completed",
        "classification_generated", "resonance_classification_generated",
        "population_inference", "planet_nine_evidence", "science_result",
    ):
        _require_bool(claims.get(name), False, f"V3 claim {name}")
    return {
        "root_relative_path": "runs/tno_1kyr_reduced5_lifecycle_gate_v3",
        "exact_root_roster": sorted(expected),
        "protocol_content_sha256": V3_PROTOCOL_CONTENT_SHA256,
        "result_content_sha256": V3_RESULT_CONTENT_SHA256,
        "git_preregistration": dict(V3_GIT),
        "interpretation": "PASSED_1KYR_SELECTED9_LIFECYCLE_GATE_LINEAGE_ONLY_NO_10MYR_AUTHORIZATION",
    }


def _input_bundle() -> dict[str, Any]:
    stage0_records = _bindings(STAGE0_BINDINGS)
    de441_records = _bindings(DE441_BINDINGS)
    legacy_records = _bindings(LEGACY_BRIDGE_BINDINGS)
    collapse_records = _bindings(COLLAPSE_BINDINGS)
    v3_records = _bindings(V3_BINDINGS)
    stage0, groups = _load_stage0()
    custody, custody_batches = _load_custody(groups)
    legacy, legacy_batches = _load_legacy_bridge()
    v3_lineage = _load_v3_lineage()
    v3_prefix = _read_project("runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin")
    if len(v3_prefix) != 169_344 or _sha256(v3_prefix) != V3_BINDINGS[2][3]:
        raise FullBridgeError("V3 prefix bytes differ during consumer crossbinding")
    # Close the read/use interval by repeating every byte binding after parsing.
    _require(_bindings(STAGE0_BINDINGS), stage0_records, "Stage0 rebound bindings")
    _require(_bindings(DE441_BINDINGS), de441_records, "DE441 rebound bindings")
    _require(_bindings(LEGACY_BRIDGE_BINDINGS), legacy_records, "legacy bridge rebound bindings")
    _require(_bindings(COLLAPSE_BINDINGS), collapse_records, "collapse implementation rebound bindings")
    _require(_bindings(V3_BINDINGS), v3_records, "V3 rebound bindings")
    return {
        "stage0": stage0, "groups": groups,
        "custody": custody, "custody_batches": custody_batches,
        "legacy": legacy, "legacy_batches": legacy_batches,
        "stage0_records": stage0_records, "de441_records": de441_records,
        "legacy_records": legacy_records, "v3_records": v3_records,
        "collapse_records": collapse_records,
        "v3_lineage": v3_lineage, "v3_prefix": v3_prefix,
    }


def _own_implementation() -> dict[str, object]:
    source = _read_project(SCRIPT_RELATIVE)
    test = _read_project(TEST_RELATIVE)
    return {
        "source": {
            "relative_path": SCRIPT_RELATIVE, "bytes": len(source),
            "file_sha256": _sha256(source),
        },
        "test": {
            "relative_path": TEST_RELATIVE, "bytes": len(test),
            "file_sha256": _sha256(test),
        },
    }


def build_protocol(bundle: Mapping[str, Any] | None = None) -> dict[str, object]:
    if bundle is None:
        bundle = _input_bundle()
    implementation = _own_implementation()
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PREPARE_ONLY_FULL128_NATIVE_EPOCH_BRIDGE_NO_DYNAMICS",
        "claim_ceiling": "INPUT_STATE_COORDINATE_AND_MODEL_BRIDGE_ONLY",
        "implementation": implementation,
        "input_bindings": {
            "stage0": bundle["stage0_records"],
            "de441": bundle["de441_records"],
            "selected9_bridge": bundle["legacy_records"],
            "stage1_v4_model_collapse": bundle["collapse_records"],
            "passed_v3_root_source_test": bundle["v3_records"],
        },
        "input_content_bindings": {
            "stage0_protocol_content_sha256": STAGE0_PROTOCOL_CONTENT_SHA256,
            "stage0_states_content_sha256": STAGE0_STATES_CONTENT_SHA256,
            "stage0_binary_payload_sha256": STAGE0_BINARY_PAYLOAD_SHA256,
            "de441_protocol_content_sha256": DE441_PROTOCOL_CONTENT_SHA256,
            "de441_custody_content_sha256": DE441_CUSTODY_CONTENT_SHA256,
            "selected9_bridge_content_sha256": LEGACY_BRIDGE_CONTENT_SHA256,
        },
        "geometry": {
            "batch_count": 12,
            "massive_body_count_per_batch": 5,
            "tracer_count_per_batch": 129,
            "tracer_order": "NOMINAL_THEN_CLONE_0_THROUGH_CLONE_127",
            "tracer_state_rows_total": 1_548,
            "massive_state_rows_total": 60,
            "object_order": list(OBJECT_IDS),
            "projected_selected9_tracer_indices": list(PROJECTED_TRACER_INDICES),
            "projected_selected8_clone_indices": list(HOLDOUT_CLONE_INDICES),
        },
        "coordinate_contract": {
            "source_tracer_frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_CARTESIAN",
            "retained_heliocentric_frame": "HELIOCENTRIC_J2000_EQUATORIAL_ICRF_AXES",
            "retained_dynamics_frame": "INSTANTANEOUS_REDUCED5_MASSIVE_GM_BARYCENTER_J2000_EQUATORIAL_ICRF_AXES",
            "length_unit": "AU",
            "velocity_unit": "AU_PER_JULIAN_YEAR",
            "time_scale": "TDB",
            "obliquity_arcseconds_exact": OBLIQUITY_ARCSECONDS_EXACT,
            "obliquity_radians_binary64_hex": OBLIQUITY_RADIANS_HEX,
            "obliquity_sine_binary64_hex": OBLIQUITY_SIN_HEX,
            "obliquity_cosine_binary64_hex": OBLIQUITY_COS_HEX,
            "rotation_arithmetic": "EXACT_INPUT_DYADS_EXACT_TWO_TERM_DOT_PRODUCT_ROUNDED_ONCE_TO_BINARY64",
            "de441_collapse_arithmetic": "EXACT_STAGE1_V4_BINARY64_OPERATION_ORDER",
            "tracers_excluded_from_all_centroids": True,
        },
        "projection_crosscheck": {
            "reference": "RETAINED_SELECTED9_BRIDGE_V1_THEN_STAGE1_V4_DE441_REDUCED5_COLLAPSE",
            "comparison": "BITWISE_EQUAL_BINARY64_HELIOCENTRIC_AND_REDUCED5_VECTORS",
            "required_in_every_batch": True,
        },
        "local_threat_boundary": "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        "consumer_payload_contract": {
            "layout": "LITTLE_ENDIAN_ROW_INTERLEAVED_5_MASSIVE_THEN_129_TRACERS_EACH_POSITION_XYZ_THEN_VELOCITY_XYZ",
            "binary64_count_per_batch": 804,
            "byte_length_per_batch": 6_432,
            "raw_sha256_retained_per_batch": True,
            "representations": {
                "direct_reduced5_barycentric": "STAGE1_COLLAPSE_OUTPUT_BEFORE_PRIMARY_RELATIVE_CONVERSION",
                "engine_primary_relative": "SUBTRACT_DIRECT_CENTRAL_STATE_AND_FORCE_PRIMARY_POSITION_VELOCITY_TO_POSITIVE_ZERO",
                "engine_checkpoint_equivalent": "REAPPLY_STAGE1_GM_CENTERING_TO_ENGINE_PRIMARY_RELATIVE_ROWS",
            },
            "conversion_operation_order_is_part_of_the_frozen_contract": True,
            "selected9_engine_checkpoint_crossbound_to_v3_year_zero": True,
            "batch0_direct_barycentric_sha256": EXPECTED_BATCH0_DIRECT_BARYCENTRIC_SHA256,
            "batch0_engine_checkpoint_sha256": EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256,
        },
        "v3_lineage": bundle["v3_lineage"],
        "publication": {
            "output_roster": ["PROTOCOL.json", "BRIDGE.json"],
            "root_must_not_exist": True,
            "files_create_only": True,
            "protocol_published_first": True,
            "bridge_published_last": True,
            "validation_uses_held_directory_and_file_descriptors": True,
        },
        "claims": dict(CLAIMS, artifact_prepared=False),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    protocol["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, protocol)
    return protocol


def _massive_and_center(
    legacy_batch: Mapping[str, Any], custody_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, object]], tuple[float, float, float], tuple[float, float, float],
    tuple[float, float, float], tuple[float, float, float],
    tuple[float, ...], tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...],
]:
    legacy_massive = legacy_batch["massive_bodies"]
    gms = tuple(
        _gm_si_to_au_year(_hex_float(row["gravitational_parameter_m3_per_s2_hex"], "legacy GM"))
        for row in legacy_massive
    )
    ssb_positions = tuple(_vector(row["position_AU_binary64_hex"], "DE441 position") for row in custody_rows)
    ssb_velocities = tuple(
        tuple(component * 365.25 for component in _vector(row["velocity_AU_per_day_binary64_hex"], "DE441 velocity"))
        for row in custody_rows
    )
    center11_position, center11_velocity = _weighted_centroid(gms, ssb_positions, ssb_velocities)
    centered11_positions = tuple(_sub(item, center11_position) for item in ssb_positions)
    centered11_velocities = tuple(_sub(item, center11_velocity) for item in ssb_velocities)
    central_gm = math.fsum(gms[:6])
    active_gms = (central_gm, *gms[6:10])
    active_positions = (centered11_positions[0], *centered11_positions[6:10])
    active_velocities = (centered11_velocities[0], *centered11_velocities[6:10])
    center5_position, center5_velocity = _weighted_centroid(active_gms, active_positions, active_velocities)
    reduced_positions = tuple(_sub(item, center5_position) for item in active_positions)
    reduced_velocities = tuple(_sub(item, center5_velocity) for item in active_velocities)
    rows: list[dict[str, object]] = []
    for index in range(5):
        rows.append({
            "massive_order": index,
            "body_id": REDUCED5_BODY_IDS[index],
            "naif_id": REDUCED5_NAIF_IDS[index],
            "gravitational_parameter_AU3_per_julian_year2_hex": active_gms[index].hex(),
            "position_AU_hex": [item.hex() for item in reduced_positions[index]],
            "velocity_AU_per_julian_year_hex": [item.hex() for item in reduced_velocities[index]],
        })
    return (
        rows, center11_position, center11_velocity, center5_position, center5_velocity,
        gms, ssb_positions, ssb_velocities,
    )


def _reduced_tracer(
    heliocentric_position: Sequence[float], heliocentric_velocity: Sequence[float],
    sun_ssb_position: Sequence[float], sun_ssb_velocity: Sequence[float],
    center11_position: Sequence[float], center11_velocity: Sequence[float],
    center5_position: Sequence[float], center5_velocity: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    resolved11_position = _sub(_add(heliocentric_position, sun_ssb_position), center11_position)
    resolved11_velocity = _sub(_add(heliocentric_velocity, sun_ssb_velocity), center11_velocity)
    return _sub(resolved11_position, center5_position), _sub(resolved11_velocity, center5_velocity)


def _binary_payload(massive: Sequence[Mapping[str, Any]], tracers: Sequence[Mapping[str, Any]]) -> bytes:
    if len(massive) != 5 or len(tracers) != 129:
        raise FullBridgeError("binary payload roster differs")
    payload = bytearray()
    for index, row in enumerate(massive):
        naif_id = _require_int(row.get("naif_id"), REDUCED5_NAIF_IDS[index], "massive payload NAIF ID")
        payload.extend(struct.pack(">q", naif_id))
        payload.extend(struct.pack(">d", _hex_float(row["gravitational_parameter_AU3_per_julian_year2_hex"], "massive GM")))
        for name in ("position_AU_hex", "velocity_AU_per_julian_year_hex"):
            payload.extend(struct.pack(">3d", *_vector(row[name], name)))
    for index, row in enumerate(tracers):
        spk_id = _require_int(row.get("spk_id"), None, "tracer payload SPK ID")
        expected_clone = None if index == 0 else index - 1
        if expected_clone is None:
            _require(row.get("clone_index"), None, "tracer payload nominal clone identity")
            clone_index = -1
        else:
            clone_index = _require_int(row.get("clone_index"), expected_clone, "tracer payload clone index")
        payload.extend(struct.pack(">q", spk_id))
        payload.extend(struct.pack(">i", clone_index))
        for name in (
            "heliocentric_j2000_position_AU_hex",
            "heliocentric_j2000_velocity_AU_per_julian_year_hex",
            "reduced5_barycentric_position_AU_hex",
            "reduced5_barycentric_velocity_AU_per_julian_year_hex",
        ):
            payload.extend(struct.pack(">3d", *_vector(row[name], name)))
    return bytes(payload)


def _projection_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if len(rows) != 9:
        raise FullBridgeError("selected projection row count differs")
    payload = bytearray()
    for index, row in enumerate(rows):
        expected_clone = None if index == 0 else HOLDOUT_CLONE_INDICES[index - 1]
        if expected_clone is None:
            _require(row.get("clone_index"), None, "projection nominal clone identity")
            clone_index = -1
        else:
            clone_index = _require_int(row.get("clone_index"), expected_clone, "projection clone index")
        payload.extend(struct.pack(">i", clone_index))
        for name in (
            "heliocentric_j2000_position_AU_hex",
            "heliocentric_j2000_velocity_AU_per_julian_year_hex",
            "reduced5_barycentric_position_AU_hex",
            "reduced5_barycentric_velocity_AU_per_julian_year_hex",
        ):
            payload.extend(struct.pack(">3d", *_vector(row[name], name)))
    return bytes(payload)


def _direct_barycentric_rows(
    massive: Sequence[Mapping[str, Any]], tracers: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], ...]:
    if len(massive) != 5 or len(tracers) != 129:
        raise FullBridgeError("barycentric consumer payload roster differs")
    rows: list[tuple[float, ...]] = []
    for row in massive:
        rows.append((
            *_vector(row["position_AU_hex"], "massive barycentric position"),
            *_vector(row["velocity_AU_per_julian_year_hex"], "massive barycentric velocity"),
        ))
    for row in tracers:
        rows.append((
            *_vector(row["reduced5_barycentric_position_AU_hex"], "tracer barycentric position"),
            *_vector(row["reduced5_barycentric_velocity_AU_per_julian_year_hex"], "tracer barycentric velocity"),
        ))
    return tuple(rows)


def _pack_state_rows(rows: Sequence[Sequence[float]], label: str) -> bytes:
    if type(rows) not in (tuple, list) or len(rows) != STATE_ROW_COUNT:
        raise FullBridgeError(f"{label} row roster differs")
    values: list[float] = []
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 6:
            raise FullBridgeError(f"{label} row shape differs")
        for item in row:
            if type(item) is not float or not math.isfinite(item):
                raise FullBridgeError(f"{label} contains a non-finite or non-built-in binary64 value")
            values.append(item)
    if len(values) != 804:
        raise FullBridgeError(f"{label} binary64 count differs")
    payload = struct.pack("<804d", *values)
    if len(payload) != STATE_RECORD_BYTES:
        raise FullBridgeError(f"{label} byte length differs")
    return payload


def _barycentric_payload(
    massive: Sequence[Mapping[str, Any]], tracers: Sequence[Mapping[str, Any]],
) -> bytes:
    """Return the direct Stage1-collapse barycentric rows, before engine conversion."""

    return _pack_state_rows(
        _direct_barycentric_rows(massive, tracers),
        "direct reduced5 barycentric payload",
    )


def _consumer_payloads(
    massive: Sequence[Mapping[str, Any]], tracers: Sequence[Mapping[str, Any]],
) -> tuple[bytes, bytes, bytes]:
    """Return direct-barycentric, engine-primary, and checkpoint-equivalent bytes.

    The subtraction sequence is intentionally explicit.  Algebraically
    equivalent rearrangements are not accepted because they need not be
    bitwise equivalent in binary64.
    """

    direct_rows = _direct_barycentric_rows(massive, tracers)
    direct = _pack_state_rows(direct_rows, "direct reduced5 barycentric payload")
    gms = tuple(
        _hex_float(row.get("gravitational_parameter_AU3_per_julian_year2_hex"), "consumer massive GM")
        for row in massive
    )
    if any(not value > 0.0 for value in gms):
        raise FullBridgeError("consumer massive GM roster is not positive")
    massive_positions = tuple(tuple(row[:3]) for row in direct_rows[:5])
    massive_velocities = tuple(tuple(row[3:]) for row in direct_rows[:5])
    tracer_positions = tuple(tuple(row[:3]) for row in direct_rows[5:])
    tracer_velocities = tuple(tuple(row[3:]) for row in direct_rows[5:])
    central_position = massive_positions[0]
    central_velocity = massive_velocities[0]
    primary_massive_positions = tuple(_sub(row, central_position) for row in massive_positions)
    primary_massive_velocities = tuple(_sub(row, central_velocity) for row in massive_velocities)
    primary_massive_positions = ((0.0, 0.0, 0.0), *primary_massive_positions[1:])
    primary_massive_velocities = ((0.0, 0.0, 0.0), *primary_massive_velocities[1:])
    primary_tracer_positions = tuple(_sub(row, central_position) for row in tracer_positions)
    primary_tracer_velocities = tuple(_sub(row, central_velocity) for row in tracer_velocities)
    primary_rows = tuple(
        (*primary_massive_positions[index], *primary_massive_velocities[index])
        for index in range(5)
    ) + tuple(
        (*primary_tracer_positions[index], *primary_tracer_velocities[index])
        for index in range(129)
    )
    primary = _pack_state_rows(primary_rows, "engine primary-relative payload")
    checkpoint_center_position, checkpoint_center_velocity = _weighted_centroid(
        gms, primary_massive_positions, primary_massive_velocities
    )
    checkpoint_rows = tuple(
        (
            *_sub(primary_massive_positions[index], checkpoint_center_position),
            *_sub(primary_massive_velocities[index], checkpoint_center_velocity),
        )
        for index in range(5)
    ) + tuple(
        (
            *_sub(primary_tracer_positions[index], checkpoint_center_position),
            *_sub(primary_tracer_velocities[index], checkpoint_center_velocity),
        )
        for index in range(129)
    )
    checkpoint = _pack_state_rows(checkpoint_rows, "engine checkpoint-equivalent payload")
    return direct, primary, checkpoint


def _selected9_checkpoint_payload(payload: bytes) -> bytes:
    if type(payload) is not bytes or len(payload) != STATE_RECORD_BYTES:
        raise FullBridgeError("checkpoint payload width differs")
    values = struct.unpack("<804d", payload)
    if any(not math.isfinite(item) for item in values):
        raise FullBridgeError("checkpoint payload contains a non-finite value")
    rows = tuple(tuple(values[index * 6:(index + 1) * 6]) for index in range(STATE_ROW_COUNT))
    selected = (*rows[:5], *(rows[5 + index] for index in PROJECTED_TRACER_INDICES))
    return struct.pack("<84d", *(item for row in selected for item in row))


def _v3_year_zero_checkpoint(prefix: bytes, batch_order: int) -> bytes:
    _require_int(batch_order, None, "V3 batch order")
    if not 0 <= batch_order < 12 or type(prefix) is not bytes or len(prefix) != 169_344:
        raise FullBridgeError("V3 year-zero locator differs")
    offset = batch_order * V3_CHECKPOINT_COUNT_PER_BATCH * V3_CHECKPOINT_BYTES
    payload = prefix[offset:offset + V3_CHECKPOINT_BYTES]
    if len(payload) != V3_CHECKPOINT_BYTES:
        raise FullBridgeError("V3 year-zero checkpoint ended early")
    return payload


def build_bridge(protocol: Mapping[str, Any], bundle: Mapping[str, Any] | None = None) -> dict[str, object]:
    if bundle is None:
        bundle = _input_bundle()
    expected_protocol = build_protocol(bundle)
    if not _exact_equal(protocol, expected_protocol):
        raise FullBridgeError("protocol differs from exact reconstruction")
    groups = bundle["groups"]
    custody_batches = bundle["custody_batches"]
    legacy_batches = bundle["legacy_batches"]
    batches: list[dict[str, object]] = []
    global_payload = bytearray()
    global_projection = bytearray()
    global_direct = bytearray()
    global_primary = bytearray()
    global_checkpoint = bytearray()
    v3_prefix = bundle.get("v3_prefix")
    if type(v3_prefix) is not bytes or len(v3_prefix) != 169_344:
        raise FullBridgeError("V3 prefix is unavailable for consumer crossbinding")
    for batch_order, (group, custody_rows, legacy_batch) in enumerate(
        zip(groups, custody_batches, legacy_batches, strict=True)
    ):
        (
            massive, center11_position, center11_velocity,
            center5_position, center5_velocity, _gms,
            ssb_positions, ssb_velocities,
        ) = _massive_and_center(legacy_batch, custody_rows)
        tracers: list[dict[str, object]] = []
        for tracer_order, source in enumerate(group):
            expected_clone = None if tracer_order == 0 else tracer_order - 1
            expected_kind = "NOMINAL" if tracer_order == 0 else "COVARIANCE_CLONE"
            if expected_clone is None:
                _require(source.get("clone_index"), None, "build nominal clone identity")
            else:
                _require_int(source.get("clone_index"), expected_clone, "build clone index")
            _require(source.get("sample_kind"), expected_kind, "build sample kind")
            _require_int(source.get("spk_id"), None, "build SPK ID")
            _require_digest(source.get("native_vector_sha256"), "build native vector digest")
            position_ecliptic = _vector(source["position_AU_hex"], "Stage0 position")
            velocity_ecliptic = _vector(source["velocity_AU_per_julian_year_hex"], "Stage0 velocity")
            position_j2000 = _rotate_ecliptic_to_j2000(position_ecliptic)
            velocity_j2000 = _rotate_ecliptic_to_j2000(velocity_ecliptic)
            reduced_position, reduced_velocity = _reduced_tracer(
                position_j2000, velocity_j2000,
                ssb_positions[0], ssb_velocities[0],
                center11_position, center11_velocity,
                center5_position, center5_velocity,
            )
            tracers.append({
                "tracer_order": tracer_order,
                "object_id": source["object_id"],
                "spk_id": source["spk_id"],
                "sample_kind": source["sample_kind"],
                "clone_index": source["clone_index"],
                "native_vector_sha256": source["native_vector_sha256"],
                "heliocentric_j2000_position_AU_hex": [item.hex() for item in position_j2000],
                "heliocentric_j2000_velocity_AU_per_julian_year_hex": [item.hex() for item in velocity_j2000],
                "reduced5_barycentric_position_AU_hex": [item.hex() for item in reduced_position],
                "reduced5_barycentric_velocity_AU_per_julian_year_hex": [item.hex() for item in reduced_velocity],
                "lifecycle_status": "ACTIVE_AT_NATIVE_EPOCH_NOT_PROPAGATED",
            })
        projected = [tracers[index] for index in PROJECTED_TRACER_INDICES]
        references: list[dict[str, object]] = []
        for index, (observed, legacy_row) in enumerate(zip(projected, legacy_batch["tnos"], strict=True)):
            _require(observed["clone_index"], legacy_row["clone_index"], "projection clone identity")
            _require(observed["native_vector_sha256"], legacy_row["native_vector_sha256"], "projection native vector")
            _require(observed["heliocentric_j2000_position_AU_hex"], legacy_row["heliocentric_j2000_position_AU_hex"], "projection heliocentric position")
            _require(observed["heliocentric_j2000_velocity_AU_per_julian_year_hex"], legacy_row["heliocentric_j2000_velocity_AU_per_julian_year_hex"], "projection heliocentric velocity")
            reference_position, reference_velocity = _reduced_tracer(
                _vector(legacy_row["heliocentric_j2000_position_AU_hex"], "legacy projected position"),
                _vector(legacy_row["heliocentric_j2000_velocity_AU_per_julian_year_hex"], "legacy projected velocity"),
                ssb_positions[0], ssb_velocities[0],
                center11_position, center11_velocity,
                center5_position, center5_velocity,
            )
            _require(observed["reduced5_barycentric_position_AU_hex"], [item.hex() for item in reference_position], "projection reduced position")
            _require(observed["reduced5_barycentric_velocity_AU_per_julian_year_hex"], [item.hex() for item in reference_velocity], "projection reduced velocity")
            references.append({
                "projection_order": index,
                "full_tracer_order": PROJECTED_TRACER_INDICES[index],
                "clone_index": observed["clone_index"],
                "native_vector_sha256": observed["native_vector_sha256"],
            })
        binary = _binary_payload(massive, tracers)
        projection = _projection_payload(projected)
        direct, primary, checkpoint = _consumer_payloads(massive, tracers)
        if direct == primary or direct == checkpoint or primary == checkpoint:
            raise FullBridgeError("consumer payload representations were relabelled or collapsed")
        selected_checkpoint = _selected9_checkpoint_payload(checkpoint)
        v3_year_zero = _v3_year_zero_checkpoint(v3_prefix, batch_order)
        if selected_checkpoint != v3_year_zero:
            raise FullBridgeError("engine checkpoint selected9 projection differs from retained V3 year zero")
        if batch_order == 0:
            _require(_sha256(direct), EXPECTED_BATCH0_DIRECT_BARYCENTRIC_SHA256, "batch0 direct barycentric payload digest")
            _require(_sha256(checkpoint), EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256, "batch0 engine checkpoint payload digest")
        global_payload.extend(binary)
        global_projection.extend(projection)
        global_direct.extend(direct)
        global_primary.extend(primary)
        global_checkpoint.extend(checkpoint)
        batch: dict[str, object] = {
            "batch_order": batch_order,
            "object_id": group[0]["object_id"],
            "spk_id": group[0]["spk_id"],
            "covariance_epoch_jd_tdb_hex": group[0]["covariance_epoch_jd_tdb_hex"],
            "massive_body_count": 5,
            "tracer_count": 129,
            "massive_bodies": massive,
            "tracers": tracers,
            "centering_receipt": {
                "resolved11_gm_barycenter_position_AU_hex": [item.hex() for item in center11_position],
                "resolved11_gm_barycenter_velocity_AU_per_julian_year_hex": [item.hex() for item in center11_velocity],
                "reduced5_center_in_resolved11_frame_position_AU_hex": [item.hex() for item in center5_position],
                "reduced5_center_in_resolved11_frame_velocity_AU_per_julian_year_hex": [item.hex() for item in center5_velocity],
                "tracers_excluded": True,
            },
            "selected9_projection": {
                "full_tracer_indices": list(PROJECTED_TRACER_INDICES),
                "rows": references,
                "binary_payload_byte_length": len(projection),
                "binary_payload_sha256": _sha256(PROJECTION_DOMAIN + projection),
                "heliocentric_and_reduced5_bitwise_equal": True,
            },
            "binary_payload_byte_length": len(binary),
            "binary_payload_sha256": _sha256(BINARY_DOMAIN + binary),
            "consumer_state_payloads": {
                "layout": "LITTLE_ENDIAN_ROW_INTERLEAVED_5_MASSIVE_THEN_129_TRACERS_POSITION_XYZ_VELOCITY_XYZ",
                "direct_reduced5_barycentric": {
                    "operation": "STAGE1_COLLAPSE_OUTPUT_WITHOUT_ENGINE_PRIMARY_CONVERSION",
                    "bytes": len(direct), "sha256": _sha256(direct),
                },
                "engine_primary_relative": {
                    "operation": "SUBTRACT_DIRECT_CENTRAL_STATE_THEN_FORCE_PRIMARY_TO_POSITIVE_ZERO",
                    "bytes": len(primary), "sha256": _sha256(primary),
                },
                "engine_checkpoint_equivalent": {
                    "operation": "GM_RECENTER_ENGINE_PRIMARY_RELATIVE_ROWS_IN_STAGE1_CHECKPOINT_ORDER",
                    "bytes": len(checkpoint), "sha256": _sha256(checkpoint),
                },
                "selected9_checkpoint_payload_bytes": len(selected_checkpoint),
                "selected9_checkpoint_payload_sha256": _sha256(selected_checkpoint),
                "retained_v3_year_zero_payload_sha256": _sha256(v3_year_zero),
                "selected9_checkpoint_bitwise_equal_to_v3_year_zero": True,
            },
        }
        batch["content_sha256"] = _content_sha256(BATCH_DOMAIN, batch)
        batches.append(batch)
    artifact: dict[str, object] = {
        "schema": BRIDGE_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "TWELVE_FULL128_NATIVE_EPOCH_REDUCED5_BATCHES_PREPARED_NO_DYNAMICS",
        "claim_ceiling": protocol["claim_ceiling"],
        "protocol_content_sha256": protocol["content_sha256"],
        "batch_count": 12,
        "massive_state_rows_total": 60,
        "tracer_state_rows_total": 1_548,
        "tracer_count_per_batch": 129,
        "batches": batches,
        "binary_payload_byte_length": len(global_payload),
        "binary_payload_sha256": _sha256(BINARY_DOMAIN + bytes(global_payload)),
        "selected9_projection_binary_payload_byte_length": len(global_projection),
        "selected9_projection_binary_payload_sha256": _sha256(PROJECTION_DOMAIN + bytes(global_projection)),
        "all_twelve_selected9_projections_bitwise_equal": True,
        "consumer_state_payload_aggregates": {
            "batch_count": 12,
            "bytes_per_batch_per_representation": STATE_RECORD_BYTES,
            "direct_reduced5_barycentric": {
                "bytes": len(global_direct), "sha256": _sha256(bytes(global_direct)),
            },
            "engine_primary_relative": {
                "bytes": len(global_primary), "sha256": _sha256(bytes(global_primary)),
            },
            "engine_checkpoint_equivalent": {
                "bytes": len(global_checkpoint), "sha256": _sha256(bytes(global_checkpoint)),
            },
            "batch0_direct_reduced5_barycentric_sha256": batches[0]["consumer_state_payloads"]["direct_reduced5_barycentric"]["sha256"],
            "batch0_engine_primary_relative_sha256": batches[0]["consumer_state_payloads"]["engine_primary_relative"]["sha256"],
            "batch0_engine_checkpoint_equivalent_sha256": batches[0]["consumer_state_payloads"]["engine_checkpoint_equivalent"]["sha256"],
            "expected_batch0_direct_reduced5_barycentric_sha256": EXPECTED_BATCH0_DIRECT_BARYCENTRIC_SHA256,
            "expected_batch0_engine_checkpoint_equivalent_sha256": EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256,
            "all_twelve_engine_checkpoint_selected9_crossbindings_pass": True,
        },
        "v3_lineage": bundle["v3_lineage"],
        "claims": dict(CLAIMS),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    artifact["content_sha256"] = _content_sha256(BRIDGE_DOMAIN, artifact)
    return artifact


def build_documents() -> tuple[dict[str, object], dict[str, object]]:
    bundle = _input_bundle()
    protocol = build_protocol(bundle)
    bridge = build_bridge(protocol, bundle)
    # The implementation and test are inputs too; close their read/use interval.
    _require(_own_implementation(), protocol["implementation"], "implementation rebound bytes")
    return protocol, bridge


def _open_output_root(path: Path) -> tuple[int, int, str]:
    absolute = path.absolute()
    if absolute.name in ("", ".", ".."):
        raise FullBridgeError("output root name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent = os.open(absolute.parent, flags)
        root = os.open(absolute.name, flags, dir_fd=parent)
    except OSError as exc:
        try:
            os.close(parent)
        except (UnboundLocalError, OSError):
            pass
        raise FullBridgeError("output root must be an existing real directory") from exc
    current = os.fstat(root)
    if not stat.S_ISDIR(current.st_mode):
        os.close(root)
        os.close(parent)
        raise FullBridgeError("output root is not a directory")
    return parent, root, absolute.name


def _read_output_file(root_fd: int, name: str) -> tuple[bytes, tuple[int, ...]]:
    try:
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
    except OSError as exc:
        raise FullBridgeError(f"retained output cannot be held: {name}") from exc
    try:
        return _read_descriptor(descriptor, f"retained {name}")
    finally:
        os.close(descriptor)


def _write_create_only(root_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=root_fd)
    except OSError as exc:
        raise FullBridgeError(f"artifact is not create-only: {name}") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise FullBridgeError(f"short write while publishing {name}")
            offset += written
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or observed.st_size != len(payload):
            raise FullBridgeError(f"published file metadata differs: {name}")
    finally:
        os.close(descriptor)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    protocol, bridge = build_documents()
    absolute = output_root.absolute()
    if absolute.name in ("", ".", ".."):
        raise FullBridgeError("output root name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(absolute.parent, flags)
    except OSError as exc:
        raise FullBridgeError("output parent must be an existing real directory") from exc
    root_fd: int | None = None
    try:
        try:
            os.mkdir(absolute.name, 0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise FullBridgeError("output root must not already exist") from exc
        root_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        if os.listdir(root_fd):
            raise FullBridgeError("new output root is not empty")
        _write_create_only(root_fd, "PROTOCOL.json", _canonical_json(protocol))
        _write_create_only(root_fd, "BRIDGE.json", _canonical_json(bridge))
        os.fsync(root_fd)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
    return validate_output(absolute)


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    parent_fd, root_fd, root_name = _open_output_root(output_root)
    try:
        if set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER:
            raise FullBridgeError("output root roster differs")
        protocol_payload, protocol_identity = _read_output_file(root_fd, "PROTOCOL.json")
        bridge_payload, bridge_identity = _read_output_file(root_fd, "BRIDGE.json")
        protocol = _strict_json(protocol_payload)
        bridge = _strict_json(bridge_payload)
        if protocol_payload != _canonical_json(protocol) or bridge_payload != _canonical_json(bridge):
            raise FullBridgeError("retained output JSON is not canonical")
        expected_protocol, expected_bridge = build_documents()
        if not _exact_equal(protocol, expected_protocol):
            raise FullBridgeError("retained protocol differs from exact reconstruction")
        if not _exact_equal(bridge, expected_bridge):
            raise FullBridgeError("retained bridge differs from exact reconstruction")
        _require(protocol.get("content_sha256"), _content_sha256(PROTOCOL_DOMAIN, protocol), "protocol content digest")
        _require(bridge.get("content_sha256"), _content_sha256(BRIDGE_DOMAIN, bridge), "bridge content digest")
        _validate_claims(protocol.get("claims"), prepared=False, label="protocol")
        _validate_claims(bridge.get("claims"), prepared=True, label="bridge")
        _require_bool(protocol.get("execution_authorized"), False, "protocol execution_authorized")
        _require_bool(protocol.get("outcomes_generated"), False, "protocol outcomes_generated")
        _require_bool(bridge.get("execution_authorized"), False, "bridge execution_authorized")
        _require_bool(bridge.get("outcomes_generated"), False, "bridge outcomes_generated")
        protocol_again, protocol_identity_again = _read_output_file(root_fd, "PROTOCOL.json")
        bridge_again, bridge_identity_again = _read_output_file(root_fd, "BRIDGE.json")
        if (
            protocol_again != protocol_payload or protocol_identity_again != protocol_identity
            or bridge_again != bridge_payload or bridge_identity_again != bridge_identity
            or set(os.listdir(root_fd)) != EXPECTED_OUTPUT_ROSTER
        ):
            raise FullBridgeError("retained output changed during held validation")
        try:
            rebound = os.open(root_name, flags := os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        except OSError as exc:
            raise FullBridgeError("output root changed during validation") from exc
        try:
            if _identity(os.fstat(rebound)) != _identity(os.fstat(root_fd)):
                raise FullBridgeError("output root identity changed during validation")
        finally:
            os.close(rebound)
        return bridge
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="reconstruct in memory without publication")
    prepare_parser = subparsers.add_parser("prepare", help="create the immutable two-file artifact")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate", help="validate a retained artifact")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            _protocol, artifact = build_documents()
        elif args.command == "prepare":
            artifact = prepare(args.output_root)
        else:
            artifact = validate_output(args.output_root)
    except FullBridgeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps({
        "schema": artifact["schema"],
        "content_sha256": artifact["content_sha256"],
        "batch_count": artifact["batch_count"],
        "tracer_state_rows_total": artifact["tracer_state_rows_total"],
        "execution_authorized": artifact["execution_authorized"],
        "outcomes_generated": artifact["outcomes_generated"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
