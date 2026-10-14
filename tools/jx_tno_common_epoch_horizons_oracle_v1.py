#!/usr/bin/env python3
"""Preregister the all-12 common-epoch Horizons nominal-state oracle.

This module is deliberately prepare-only.  It freezes exact designation-only
Horizons requests and response acceptance gates for a future, separately
reviewed acquisition successor.  It performs no network access, acquisition,
dynamics, residual comparison, execution, or resume operation.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "tools" / "jx_tno_common_epoch_horizons_oracle_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_common_epoch_horizons_oracle_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_common_epoch_horizons_oracle_v1"
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})

PROTOCOL_SCHEMA = "jx.tno.common-epoch-horizons-oracle.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.common-epoch-horizons-oracle.protocol.v1\0"
CUSTODY_SCHEMA = "jx.tno.common-epoch-horizons-oracle.custody.v1"
CUSTODY_DOMAIN = b"jx.tno.common-epoch-horizons-oracle.custody.v1\0"
VECTOR_DOMAIN = b"jx.tno.common-epoch-horizons-oracle.ordered-12x6-vector.v1\0"
EXPECTED_BRANCH = "research/tno-common-epoch-horizons-oracle-v1-20261014"
FUTURE_PREREGISTRATION_TAG = "tno-common-epoch-horizons-oracle-v1-preregistered-20261014"

PARENT_TAG = "tno-common-epoch-full129-bridge-v1-preregistered-20261014"
PARENT_TAG_OBJECT = "4620bf745540da9569d9f94bc3677ab176bca8ef"
PARENT_TAG_PAYLOAD_BYTES = 327
PARENT_TAG_PAYLOAD_SHA256 = "671d24cf40b4cc29ef7e9c5126880b3c8d0f645e2246c697e2cec3b9ab6ab863"
PARENT_COMMIT = "fca1d7b9ea3a97a8127665dcd0b87c600971a803"
PARENT_TREE = "300507c69d9c66996da51bb01ce4b7b7bd909979"

BRIDGE_SOURCE_SHA256 = "b13cd45d35bc56a8a6a1864d8f8719a3edc540715d7acac999a7dea4c6c8a5be"
BRIDGE_TEST_SHA256 = "38119e03c970af9b6f7a9486e3440a152849fc3641c881db9d7572691eb770c0"
BRIDGE_PROTOCOL_SHA256 = "895025b3c1c37e27147e8555a74763e10d43d29396182a91decd02476694521d"
BRIDGE_PROTOCOL_CONTENT_SHA256 = "e3c6fa8f9cb9701f9faf667068a87202204815e1bac942da46fc934dc6e66bc8"

SBDB_PROTOCOL_SHA256 = "4f3de80ba98b43c907bb846bd35724a0f1bd0ef9ef152e7642e3af65a2229acc"
SBDB_CUSTODY_SHA256 = "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288"
SBDB_PROTOCOL_CONTENT_SHA256 = "176b56c4784bef58a8eacc6e34d99edb17d21d70c559a245b7c0677cdb1a4a60"
SBDB_CUSTODY_CONTENT_SHA256 = "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f"

HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_DOCUMENTATION = "https://ssd-api.jpl.nasa.gov/doc/horizons.html"
MAX_RESPONSE_BYTES = 1_048_576
TARGET_JD_TDB_DECIMAL = "2461200.5"
TARGET_JD_TDB_HEX = "0x1.2c70840000000p+21"
ELEMENT_LABELS = ("e", "q", "tp", "node", "peri", "i")
ELEMENT_UNITS = (None, "au", "JD TDB", "deg", "deg", "deg")
SOLUTION_IDS = (51, 12, 3, 7, 11, 5, 14, 29, 25, 28, 38, 34)


class HorizonsOracleProtocolError(RuntimeError):
    """A frozen input, response gate, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class Target:
    order: int
    object_id: str
    spk_id: int
    primary_designation: str
    full_name: str
    solution_id: int
    element_tokens: tuple[str, str, str, str, str, str]
    raw_bytes: int
    raw_sha256: str

    @property
    def raw_role(self) -> str:
        return f"raw_{self.spk_id}"

    @property
    def raw_relative_path(self) -> str:
        return f"runs/tno_sbdb_covariance_acquisition_v1/raw/{self.spk_id}.json"

    @property
    def query_parameter_pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("format", "json"),
            ("COMMAND", f"'DES={self.spk_id};'"),
            ("OBJ_DATA", "'NO'"),
            ("MAKE_EPHEM", "'YES'"),
            ("EPHEM_TYPE", "'VECTORS'"),
            ("CENTER", "'500@10'"),
            ("TLIST", f"'{TARGET_JD_TDB_DECIMAL}'"),
            ("TLIST_TYPE", "'JD'"),
            ("TIME_TYPE", "'TDB'"),
            ("REF_PLANE", "'ECLIPTIC'"),
            ("REF_SYSTEM", "'J2000'"),
            ("OUT_UNITS", "'AU-D'"),
            ("VEC_TABLE", "'2'"),
            ("VEC_CORR", "'NONE'"),
            ("CSV_FORMAT", "'YES'"),
            ("VEC_LABELS", "'YES'"),
            ("TIME_DIGITS", "'FRACSEC'"),
        )

    @property
    def query_url(self) -> str:
        query = "&".join(
            f"{_percent_encode(name)}={_percent_encode(value)}"
            for name, value in self.query_parameter_pairs
        )
        return f"{HORIZONS_ENDPOINT}?{query}"


TARGETS = (
    Target(0, "jpl-20090377", 20090377, "90377", "90377 Sedna (2003 VB12)", 51,
           (".8598824585187618", "76.18464364627253", "2479264.750687138669", "144.5061662673739", "311.0987725939751", "11.92527582847476"),
           3855, "62bad94b65a7ab2472e20d771b5289003eafb9dedc6eca351935b98ba447bcc8"),
    Target(1, "jpl-20474640", 20474640, "474640", "474640 Alicanto (2004 VN112)", 12,
           (".861547579461181", "47.37148771531466", "2455139.573512951511", "66.03582012284039", "327.2807650284388", "25.51048085878516"),
           3855, "4b6e601b1cb41afb72428cfe7112622307c5a08c032a98277b1c9a688459035c"),
    Target(2, "jpl-54526835", 54526835, "2023 KQ14", "(2023 KQ14)", 3,
           (".7335661561347354", "65.92333155513307", "2474743.035190263161", "72.07902619292427", "198.7037510664639", "11.00095782821624"),
           3808, "a47327d6de9c5b30a5d12aec549aef1454ec575fc4dd764bc49105a6c6471aef"),
    Target(3, "jpl-20541132", 20541132, "541132", "541132 Leleakuhonua (2015 TG387)", 7,
           (".9519448010157889", "64.68108100288764", "2480288.494978737181", "301.1346719755806", "118.2317322688105", "11.67889561905244"),
           3857, "6ce8dac67eaf465bde689d4e782ffa65a7c551195075d06d865c69d860d5b297"),
    Target(4, "jpl-50666516", 50666516, "2012 VP113", "(2012 VP113)", 11,
           (".6985589359876868", "80.64181072397446", "2444178.261423504482", "90.8926204194851", "294.2852296107872", "24.00179208811373"),
           3796, "a43a00f133ad1b435416917030512038e23a4f7b52619f42a76a9c8763bc80ce"),
    Target(5, "jpl-20148209", 20148209, "148209", "148209 (2000 CR105)", 5,
           (".8073365317057665", "44.16931657232168", "2438948.969895525519", "128.2037509183512", "317.0408950104643", "22.70799015441454"),
           3798, "e2e4e08acff0fb4a4d37bec7cecf9656ac4633adc480676e3ac9b6ba25dae4b5"),
    Target(6, "jpl-20574372", 20574372, "574372", "574372 (2010 JO179)", 14,
           (".4988701464793244", "39.13505094828052", "2433891.765322347685", "147.0462304885865", "9.496145292923686", "32.06386260847041"),
           3824, "bcf5b50ee3bc4437ab0c55333249a128ba239a9fc8ba23d765acfcc6db9aeb91"),
    Target(7, "jpl-20303775", 20303775, "303775", "303775 (2005 QU182)", 29,
           (".6686928465820892", "37.01605048711487", "2441134.224829422470", "78.51532092257791", "224.1252348407445", "14.01023423630377"),
           3837, "a1e779145a7c91c7d954c63047a5e03302d4c38b0e9869f022937ba647a4a2ef"),
    Target(8, "jpl-20026181", 20026181, "26181", "26181 (1996 GQ21)", 25,
           (".5888130202724389", "38.15355696859051", "2447752.956642173835", "194.18417395", "356.1020162135479", "13.3537460632102"),
           3823, "13ae906ea1c001d02b5c528ee3a2e5bfc9d0884860699bed2f0350fc0d17da73"),
    Target(9, "jpl-20042301", 20042301, "42301", "42301 (2001 UR163)", 28,
           (".2805075574056742", "37.1850899540823", "2428519.940385409034", "302.4037703262449", "343.1295377754446", ".7510091755756962"),
           3835, "0856c73adc67c8fb70813afe09fd4f933aa3216a15683935d62e24d73bdbdac8"),
    Target(10, "jpl-20026375", 20026375, "26375", "26375 (1999 DE9)", 38,
           (".4206896190844759", "32.21772021987648", "2446518.591585215929", "322.8597924267265", "159.5941686692768", "7.610567083727258"),
           3834, "09722418980836833b11ee8fae108964408ff7a8397c70c03e46e24a4e88a293"),
    Target(11, "jpl-20015874", 20015874, "15874", "15874 (1996 TL66)", 34,
           (".5856282476985989", "35.10392324097838", "2452159.869144556968", "217.7307666730197", "185.1373827636416", "23.96488441448197"),
           3839, "db9a0bd37a44e52527bbcb061dd65e725564240da773d36e1f5746f4bfcd00e3"),
)

CUSTODY_TOP_LEVEL_KEYS = (
    "claims",
    "content_sha256",
    "nonclaims",
    "ordered_vector_sha256",
    "preregistration_identity",
    "protocol_content_sha256",
    "protocol_file_sha256",
    "record_count",
    "records",
    "schema",
    "status",
)
CUSTODY_PREREGISTRATION_KEYS = (
    "commit",
    "protocol_content_sha256",
    "protocol_file_bytes",
    "protocol_file_sha256",
    "source_bytes",
    "source_sha256",
    "tag",
    "tag_object",
    "tag_payload_bytes",
    "tag_payload_sha256",
    "test_bytes",
    "test_sha256",
    "tree",
)
CUSTODY_RECORD_KEYS = (
    "calendar_tdb",
    "center",
    "corrections",
    "epoch_jd_tdb",
    "frame",
    "full_name",
    "object_id",
    "order",
    "planetary_ephemeris",
    "position_AU_binary64_hex",
    "position_AU_decimal_tokens",
    "primary_designation",
    "printed_element_binary64_hex",
    "printed_element_decimal_tokens",
    "printed_element_labels",
    "query_url",
    "raw_bytes",
    "raw_sha256",
    "retained_relative_path",
    "small_body_perturbers",
    "solution_id",
    "spk_id",
    "tp_binary64_ulp_distance",
    "vector_component_labels",
    "velocity_AU_per_day_binary64_hex",
    "velocity_AU_per_day_decimal_tokens",
)
VECTOR_COMPONENT_LABELS = (
    "x_AU",
    "y_AU",
    "z_AU",
    "vx_AU_per_day",
    "vy_AU_per_day",
    "vz_AU_per_day",
)
RESULT_ROOT_ROSTER = ("CUSTODY.json", "raw")
RESULT_RAW_ROSTER = tuple(f"{target.spk_id}.json" for target in TARGETS)
CUSTODY_CLAIMS = {
    "all_twelve_exact_raw_responses_retained": True,
    "all_twelve_hard_response_gates_passed": True,
    "common_epoch_nominal_vectors_retained": True,
    "designation_only_query_contract_replayed": True,
    "diagnostic_oracle_input_only": True,
    "provider_origin_cryptographically_proven": False,
    "provider_transport_attested_by_custody": False,
    "residual_tolerance_defined": False,
    "dynamics_executed": False,
    "common_epoch_bridge_qualified": False,
    "classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}
CUSTODY_NONCLAIMS = {
    "horizons_full_force_model_equivalence": False,
    "numerical_bridge_acceptance_or_qualification": False,
    "covariance_posterior_or_probability": False,
    "dynamical_classification": False,
    "physical_stability_collision_or_ejection": False,
    "population_inference": False,
    "planet_nine_detection_exclusion_or_ranking": False,
    "long_horizon_accuracy_or_execution_authority": False,
}


@dataclass(frozen=True, slots=True)
class FileBinding:
    group: str
    role: str
    relative_path: str
    byte_length: int
    sha256: str


def _b(group: str, role: str, path: str, size: int, digest: str) -> FileBinding:
    return FileBinding(group, role, path, size, digest)


INPUT_BINDINGS = (
    _b("parent_bridge", "source", "benchmarks/jx_tno_common_epoch_full129_bridge_v1.py", 54_643, BRIDGE_SOURCE_SHA256),
    _b("parent_bridge", "test", "tests/test_tno_common_epoch_full129_bridge_v1.py", 15_059, BRIDGE_TEST_SHA256),
    _b("parent_bridge", "protocol", "runs/tno_common_epoch_full129_bridge_v1/PROTOCOL.json", 29_996, BRIDGE_PROTOCOL_SHA256),
    _b("sbdb", "protocol", "runs/tno_sbdb_covariance_acquisition_v1/PROTOCOL.json", 6_261, SBDB_PROTOCOL_SHA256),
    _b("sbdb", "custody", "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json", 16_520, SBDB_CUSTODY_SHA256),
    *tuple(_b("sbdb_raw", target.raw_role, target.raw_relative_path, target.raw_bytes, target.raw_sha256) for target in TARGETS),
)

CONTENT_BINDINGS = {
    "parent_bridge_protocol": BRIDGE_PROTOCOL_CONTENT_SHA256,
    "sbdb_protocol": SBDB_PROTOCOL_CONTENT_SHA256,
    "sbdb_custody": SBDB_CUSTODY_CONTENT_SHA256,
}

_FLOAT_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_TARGET_LINE = re.compile(r"^Target body name:\s*(.+?)\s*\{source:\s*JPL#([0-9]+)\}\s*$", re.MULTILINE)
_CENTER_LINE = re.compile(r"^Center body name:\s*Sun \(10\)\s*\{source:\s*(DE441)\}\s*$", re.MULTILINE)
_SMALL_LINE = re.compile(r"^Small perturbers:\s*Yes\s*\{source:\s*(SB441-N16)\}\s*$", re.MULTILINE)
_EPOCH_LINE = re.compile(r"^\s*EPOCH=\s*([^\s]+)\s*!.*\((TDB)\).*$", re.MULTILINE)
_ELEMENT_LINES = re.compile(
    r"^\s*EC=\s*([^\s]+)\s+QR=\s*([^\s]+)\s+TP=\s*([^\s]+)\s*$"
    r"\n^\s*OM=\s*([^\s]+)\s+W=\s*([^\s]+)\s+IN=\s*([^\s]+)\s*$",
    re.MULTILINE,
)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise HorizonsOracleProtocolError("record is not strict canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(PROTOCOL_DOMAIN + _canonical_json(record))


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise HorizonsOracleProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HorizonsOracleProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except HorizonsOracleProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise HorizonsOracleProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise HorizonsOracleProtocolError("JSON root must be an exact object")
    return value


def _percent_encode(value: str) -> str:
    unreserved = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    return "".join(chr(byte) if byte in unreserved else f"%{byte:02X}" for byte in value.encode("ascii"))


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise HorizonsOracleProtocolError("binding path is not a contained POSIX path")
    return pure.parts


def _read_relative(root_fd: int, binding: FileBinding) -> bytes:
    parts = _relative_parts(binding.relative_path)
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != binding.byte_length:
                raise HorizonsOracleProtocolError(f"{binding.role} type, link count, or byte length differs")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(file_fd)
            if _stat_identity(before) != _stat_identity(after):
                raise HorizonsOracleProtocolError(f"{binding.role} changed while held")
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise HorizonsOracleProtocolError(f"cannot read frozen binding {binding.role}") from exc
    finally:
        os.close(directory_fd)
    payload = b"".join(chunks)
    if len(payload) != binding.byte_length or _sha256(payload) != binding.sha256:
        raise HorizonsOracleProtocolError(f"{binding.role} frozen bytes differ")
    return payload


def _read_regular_path(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HorizonsOracleProtocolError(f"cannot open implementation file {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise HorizonsOracleProtocolError(f"implementation file is linked or nonregular: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise HorizonsOracleProtocolError(f"implementation file changed while held: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _git(*args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise HorizonsOracleProtocolError("Git ancestry check failed")
    return process.stdout


def _verify_parent_ancestry() -> dict[str, object]:
    if _git("rev-parse", f"{PARENT_TAG}^{{tag}}").decode("ascii").strip() != PARENT_TAG_OBJECT:
        raise HorizonsOracleProtocolError("parent tag object differs")
    if _git("rev-parse", f"{PARENT_TAG}^{{commit}}").decode("ascii").strip() != PARENT_COMMIT:
        raise HorizonsOracleProtocolError("parent tag commit differs")
    if _git("rev-parse", f"{PARENT_TAG}^{{tree}}").decode("ascii").strip() != PARENT_TREE:
        raise HorizonsOracleProtocolError("parent tag tree differs")
    payload = _git("cat-file", "tag", PARENT_TAG)
    if len(payload) != PARENT_TAG_PAYLOAD_BYTES or _sha256(payload) != PARENT_TAG_PAYLOAD_SHA256:
        raise HorizonsOracleProtocolError("parent tag payload differs")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestry.returncode != 0:
        raise HorizonsOracleProtocolError("HEAD does not descend from the parent preregistration")
    return {
        "tag": PARENT_TAG,
        "tag_object": PARENT_TAG_OBJECT,
        "tag_payload_bytes": PARENT_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": PARENT_TAG_PAYLOAD_SHA256,
        "commit": PARENT_COMMIT,
        "tree": PARENT_TREE,
    }


def _binding_record(binding: FileBinding) -> dict[str, object]:
    return {
        "role": binding.role,
        "relative_path": binding.relative_path,
        "bytes": binding.byte_length,
        "sha256": binding.sha256,
    }


def _float_token(token: str, label: str) -> float:
    if type(token) is not str or _FLOAT_TOKEN.fullmatch(token) is None:
        raise HorizonsOracleProtocolError(f"{label} is not an exact decimal float token")
    value = float(token)
    if not math.isfinite(value):
        raise HorizonsOracleProtocolError(f"{label} is non-finite")
    return value


def _target_record(target: Target) -> dict[str, object]:
    values = [_float_token(token, f"{target.object_id} element {label}") for label, token in zip(ELEMENT_LABELS, target.element_tokens, strict=True)]
    return {
        "order": target.order,
        "object_id": target.object_id,
        "spk_id": target.spk_id,
        "primary_designation": target.primary_designation,
        "full_name": target.full_name,
        "frozen_solution_id": target.solution_id,
        "sbdb_standard_epoch_jd_tdb": TARGET_JD_TDB_DECIMAL,
        "sbdb_models": {"planetary_ephemeris": "DE441", "small_body_perturbers": "SB441-N16"},
        "sbdb_raw": {
            "relative_path": target.raw_relative_path,
            "bytes": target.raw_bytes,
            "sha256": target.raw_sha256,
        },
        "standard_osculating_elements": {
            "labels": list(ELEMENT_LABELS),
            "units": list(ELEMENT_UNITS),
            "decimal_tokens": list(target.element_tokens),
            "binary64_hex": [value.hex() for value in values],
        },
        "horizons_query": {
            "ordered_parameter_pairs": [list(pair) for pair in target.query_parameter_pairs],
            "url": target.query_url,
        },
    }


def _validate_sbdb_target(target: Target, payload: bytes, custody_record: object) -> dict[str, object]:
    if type(custody_record) is not dict:
        raise HorizonsOracleProtocolError("SBDB custody record type differs")
    expected_custody = {
        "object_id": target.object_id,
        "spk_id": target.spk_id,
        "primary_designation": target.primary_designation,
        "full_name": target.full_name,
        "orbit_solution_id": str(target.solution_id),
        "standard_epoch_jd_tdb": TARGET_JD_TDB_DECIMAL,
        "planetary_ephemeris": "DE441",
        "retained_relative_path": f"raw/{target.spk_id}.json",
        "raw_byte_count": target.raw_bytes,
        "raw_sha256": target.raw_sha256,
    }
    if any(custody_record.get(key) != value for key, value in expected_custody.items()):
        raise HorizonsOracleProtocolError("SBDB custody target identity differs")
    value = _strict_json(payload)
    if value.get("signature") != {
        "source": "NASA/JPL Small-Body Database (SBDB) API",
        "version": "1.3",
    }:
        raise HorizonsOracleProtocolError("SBDB provider signature differs")
    object_record = value.get("object")
    orbit = value.get("orbit")
    if type(object_record) is not dict or type(orbit) is not dict:
        raise HorizonsOracleProtocolError("SBDB object or orbit record differs")
    if (
        object_record.get("spkid") != str(target.spk_id)
        or object_record.get("des") != target.primary_designation
        or object_record.get("fullname") != target.full_name
        or object_record.get("orbit_id") != str(target.solution_id)
    ):
        raise HorizonsOracleProtocolError("SBDB raw object identity differs")
    if (
        orbit.get("orbit_id") != str(target.solution_id)
        or orbit.get("epoch") != TARGET_JD_TDB_DECIMAL
        or orbit.get("pe_used") != "DE441"
        or orbit.get("sb_used") != "SB441-N16"
    ):
        raise HorizonsOracleProtocolError("SBDB raw solution, epoch, or model differs")
    elements = orbit.get("elements")
    if type(elements) is not list:
        raise HorizonsOracleProtocolError("SBDB standard element roster differs")
    by_label: dict[str, str] = {}
    for element in elements:
        if type(element) is not dict:
            raise HorizonsOracleProtocolError("SBDB standard element type differs")
        label = element.get("label")
        if label in ELEMENT_LABELS:
            if label in by_label or type(element.get("value")) is not str:
                raise HorizonsOracleProtocolError("SBDB standard element label differs")
            by_label[label] = element["value"]
    observed = tuple(by_label.get(label) for label in ELEMENT_LABELS)
    if observed != target.element_tokens:
        raise HorizonsOracleProtocolError("SBDB frozen standard element tokens differ")
    for label, token in zip(ELEMENT_LABELS, target.element_tokens, strict=True):
        _float_token(token, f"SBDB {target.object_id} {label}")
    return _target_record(target)


def audit_prerequisites() -> dict[str, object]:
    root_fd = os.open(PROJECT_ROOT, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    payloads: dict[tuple[str, str], bytes] = {}
    try:
        for binding in INPUT_BINDINGS:
            payloads[(binding.group, binding.role)] = _read_relative(root_fd, binding)
    finally:
        os.close(root_fd)

    bridge_payload = payloads[("parent_bridge", "protocol")]
    bridge = _strict_json(bridge_payload)
    if bridge_payload != _canonical_json(bridge):
        raise HorizonsOracleProtocolError("parent bridge protocol is not canonical")
    parent_oracle = bridge.get("acceptance", {}).get("common_epoch_nominal_cartesian_oracle", {})
    if (
        bridge.get("schema") != "jx.tno.common-epoch-full129-bridge.protocol.v1"
        or bridge.get("content_sha256") != BRIDGE_PROTOCOL_CONTENT_SHA256
        or bridge.get("target", {}).get("jd_tdb_decimal") != TARGET_JD_TDB_DECIMAL
        or parent_oracle.get("status") != "UNBOUND_EXECUTION_BLOCKER"
        or parent_oracle.get("frozen_designation_solution_ids_in_object_order") != list(SOLUTION_IDS)
    ):
        raise HorizonsOracleProtocolError("parent bridge semantic binding differs")

    sbdb_protocol_payload = payloads[("sbdb", "protocol")]
    sbdb_custody_payload = payloads[("sbdb", "custody")]
    sbdb_protocol = _strict_json(sbdb_protocol_payload)
    sbdb_custody = _strict_json(sbdb_custody_payload)
    if sbdb_protocol_payload != _canonical_json(sbdb_protocol) or sbdb_custody_payload != _canonical_json(sbdb_custody):
        raise HorizonsOracleProtocolError("SBDB protocol or custody is not canonical")
    if (
        sbdb_protocol.get("schema") != "jx.tno.sbdb-covariance-acquisition.protocol.v1"
        or sbdb_protocol.get("content_sha256") != SBDB_PROTOCOL_CONTENT_SHA256
        or sbdb_custody.get("schema") != "jx.tno.sbdb-covariance-acquisition.custody.v1"
        or sbdb_custody.get("content_sha256") != SBDB_CUSTODY_CONTENT_SHA256
        or sbdb_custody.get("record_count") != len(TARGETS)
    ):
        raise HorizonsOracleProtocolError("SBDB protocol or custody semantic binding differs")
    custody_records = sbdb_custody.get("records")
    if type(custody_records) is not list or len(custody_records) != len(TARGETS):
        raise HorizonsOracleProtocolError("SBDB custody roster differs")
    target_roster = [
        _validate_sbdb_target(target, payloads[("sbdb_raw", target.raw_role)], custody_records[target.order])
        for target in TARGETS
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for binding in INPUT_BINDINGS:
        grouped.setdefault(binding.group, []).append(_binding_record(binding))
    return {
        "parent_preregistration": _verify_parent_ancestry(),
        "input_bindings": grouped,
        "input_content_bindings": dict(CONTENT_BINDINGS),
        "target_roster": target_roster,
    }


def _implementation_record(path: Path) -> dict[str, object]:
    payload = _read_regular_path(path)
    return {
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def build_protocol() -> dict[str, object]:
    prerequisites = audit_prerequisites()
    implementation = {
        "source": _implementation_record(SOURCE_PATH),
        "test": _implementation_record(TEST_PATH),
        "network_client_implemented": False,
        "dynamics_runtime_imported": False,
    }
    fixed_pairs = [
        list(pair)
        for pair in TARGETS[0].query_parameter_pairs
        if pair[0] != "COMMAND"
    ]
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARED_COMMON_EPOCH_HORIZONS_NOMINAL_ORACLE_PREREGISTRATION_NO_ACQUISITION",
        "evidence_class": "PREPARE_ONLY_QUERY_AND_RESPONSE_GATE_CONTRACT_NO_PROVIDER_RESPONSE_NO_MODEL_OUTPUT",
        "authority": {
            "expected_branch_for_preparation": EXPECTED_BRANCH,
            "parent_preregistration": prerequisites["parent_preregistration"],
            "preregistration_required_before_any_acquisition_successor_is_added": True,
            "successor_enforcement": "ANY_ACQUISITION_SUCCESSOR_MUST_BIND_THE_FUTURE_ANNOTATED_PREREGISTRATION_TAG_OBJECT_PEELED_COMMIT_TREE_AND_EXACT_SOURCE_TEST_PROTOCOL_BYTES_BEFORE_REQUEST_ZERO",
            "acquisition_in_this_artifact": False,
            "execution_in_this_artifact": False,
            "network_access": False,
            "public_cli": ["build", "audit-prerequisites", "prepare", "validate-prepared"],
            "explicitly_absent_public_cli": ["acquire", "execute", "execute-worker", "resume", "validate-result"],
        },
        "implementation": implementation,
        "future_preregistration": {
            "exact_annotated_tag_name": FUTURE_PREREGISTRATION_TAG,
            "lightweight_tag_forbidden": True,
            "tag_target": "THE_EXACT_PREPARE_ONLY_COMMIT_CONTAINING_THIS_SOURCE_TEST_PROTOCOL_AND_MANIFEST_ENTRY",
            "ordered_release_and_successor_phases": [
                "COMMIT_EXACT_PREPARE_ONLY_SOURCE_TEST_PROTOCOL_AND_MANIFEST_ENTRY",
                "CREATE_ANNOTATED_TAG_WITH_EXACT_FROZEN_NAME_POINTING_TO_THAT_PREPARE_COMMIT",
                "PUSH_PREPARE_BRANCH_AND_ANNOTATED_TAG",
                "REFETCH_BRANCH_AND_TAG_FROM_ORIGIN_WITH_AN_EXPLICIT_TAG_REFSPEC",
                "VERIFY_REFETCHED_TAG_OBJECT_PEELED_COMMIT_TREE_TAG_PAYLOAD_BYTE_LENGTH_AND_TAG_PAYLOAD_SHA256",
                "ONLY_THEN_CREATE_THE_ACQUISITION_SUCCESSOR_COMMIT",
                "SUCCESSOR_VERIFIES_EVERY_EXACT_BINDING_BEFORE_REQUEST_ZERO",
            ],
            "refetch_verification_must_complete_before_successor_exists": True,
            "successor_exact_literal_bindings_required_before_request_zero": [
                "EXACT_ANNOTATED_TAG_NAME",
                "TAG_OBJECT_SHA",
                "PEELED_COMMIT_SHA",
                "PEELED_TREE_SHA",
                "TAG_PAYLOAD_BYTE_LENGTH",
                "TAG_PAYLOAD_SHA256",
                "SOURCE_RELATIVE_PATH_BYTE_LENGTH_SHA256",
                "TEST_RELATIVE_PATH_BYTE_LENGTH_SHA256",
                "PROTOCOL_RELATIVE_PATH_BYTE_LENGTH_FILE_SHA256_CONTENT_SHA256",
            ],
            "source_binding_frozen_here": implementation["source"],
            "test_binding_frozen_here": implementation["test"],
            "protocol_relative_path": "runs/tno_common_epoch_horizons_oracle_v1/PROTOCOL.json",
            "protocol_file_byte_length_and_sha256_must_be_frozen_by_successor": True,
            "protocol_content_sha256_field_must_be_bound_by_successor": True,
            "request_count_allowed_before_all_bindings_pass": 0,
            "mismatch_action": "STOP_BEFORE_REQUEST_ZERO_AND_PUBLISH_NO_RESULT",
        },
        "parent_bridge": {
            "status": "BOUND_EXACT_PARENT_WITH_ORACLE_BLOCKER_STILL_UNRESOLVED",
            "source_sha256": BRIDGE_SOURCE_SHA256,
            "test_sha256": BRIDGE_TEST_SHA256,
            "protocol_file_sha256": BRIDGE_PROTOCOL_SHA256,
            "protocol_content_sha256": BRIDGE_PROTOCOL_CONTENT_SHA256,
            "blocker_resolved_by_this_prepare_only_artifact": False,
        },
        "inputs": {
            "bindings": prerequisites["input_bindings"],
            "content_bindings": prerequisites["input_content_bindings"],
            "targets": prerequisites["target_roster"],
            "target_count": len(TARGETS),
            "object_order": [target.object_id for target in TARGETS],
            "spk_id_order": [target.spk_id for target in TARGETS],
            "frozen_solution_ids_in_object_order": list(SOLUTION_IDS),
        },
        "target_epoch": {
            "time_scale": "TDB",
            "jd_tdb_decimal": TARGET_JD_TDB_DECIMAL,
            "jd_tdb_binary64_hex": TARGET_JD_TDB_HEX,
            "one_identical_epoch_for_all_targets": True,
        },
        "query_contract": {
            "transport_method_for_future_successor": "HTTPS_GET",
            "endpoint": HORIZONS_ENDPOINT,
            "documentation": HORIZONS_DOCUMENTATION,
            "request_order": "EXACT_TARGET_ORDER_ZERO_THROUGH_ELEVEN",
            "one_request_per_target": True,
            "designation_only": True,
            "command_template": "'DES=<FROZEN_SPK_ID>;'",
            "user_supplied_orbital_elements_forbidden": True,
            "ordered_fixed_parameter_pairs_excluding_command": fixed_pairs,
            "center": "500@10",
            "epoch": TARGET_JD_TDB_DECIMAL,
            "time_scale": "TDB",
            "reference_plane": "ECLIPTIC",
            "reference_system": "J2000",
            "output_units": "AU-D",
            "vector_table": "2",
            "corrections": "NONE",
            "csv_format": True,
            "vector_labels": True,
            "time_digits": "FRACSEC",
            "object_data_suppressed": True,
            "printed_initial_osculating_elements_still_required": True,
            "exact_per_target_parameter_pairs_and_urls_in_inputs": True,
        },
        "future_transport_contract": {
            "implemented_in_this_artifact": False,
            "method": "GET",
            "ordered_exact_urls": [target.query_url for target in TARGETS],
            "request_order": list(range(len(TARGETS))),
            "request_count": len(TARGETS),
            "requests_per_url": 1,
            "parallel_requests_forbidden": True,
            "retries_for_any_reason": 0,
            "redirect_policy": "FORBIDDEN;ANY_3XX_IS_A_TERMINAL_FAILURE",
            "proxy_use": "FORBIDDEN",
            "http_success_policy": "STATUS_CODE_EXACTLY_200_REQUIRED",
            "application_request_headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "JX-TNO-common-epoch-horizons-oracle-v1",
            },
            "cookies_and_authorization_headers": "FORBIDDEN",
            "tls_policy": "DEFAULT_PLATFORM_CA_AND_HOSTNAME_VERIFICATION_REQUIRED;NO_VERIFICATION_BYPASS",
            "timeouts_seconds_per_request": {
                "connect": 30,
                "read_idle": 30,
                "total_wall_clock": 120,
                "timeout_action": "FAIL_CLOSED_NO_RETRY",
            },
            "response_headers": {
                "content_encoding": "MUST_BE_ABSENT_OR_EXACTLY_IDENTITY",
                "content_length_if_present": "ONE_STRICT_BASE10_NONNEGATIVE_INTEGER_LE_1048576_AND_MUST_EQUAL_FINAL_ENTITY_BYTE_COUNT",
                "duplicate_or_malformed_content_length": "FAIL",
            },
            "streaming_read": {
                "maximum_chunk_bytes": 65_536,
                "maximum_buffered_bytes": MAX_RESPONSE_BYTES + 1,
                "algorithm": "READ_BINARY_CHUNKS_IN_ORDER;STOP_IMMEDIATELY_ON_BYTE_1048577;ONLY_AFTER_CLEAN_EOF_AT_OR_BELOW_1048576_MAY_UTF8_JSON_DECODING_BEGIN",
                "cap_applies_after_transfer_coding_removal": True,
                "cap_applies_before_content_decoding_or_text_decoding": True,
            },
            "custody_raw_byte_semantics": "EXACT_HTTP_ENTITY_OCTETS_AFTER_TRANSFER_CODING_REMOVAL_WITH_CONTENT_ENCODING_ABSENT_OR_IDENTITY;NO_DECOMPRESSION_CHARSET_DECODING_NEWLINE_NORMALIZATION_TEXT_REENCODING_OR_JSON_RESERIALIZATION",
            "failure_action": "CLOSE_RESPONSE_STOP_SEQUENCE_REMOVE_PRIVATE_STAGING_AND_PUBLISH_NO_FINAL_RESULT_OR_CUSTODY",
        },
        "response_acceptance": {
            "all_hard_gates_must_pass": True,
            "failure_action": "REJECT_RESPONSE_AND_PUBLISH_NO_ORACLE_RESULT",
            "hard_acceptance_gates": [
                "RAW_RESPONSE_BYTE_LENGTH_LE_1048576_ENFORCED_BEFORE_UTF8_JSON_DECODE",
                "EXACT_TOP_LEVEL_JSON_KEY_ROSTER_SIGNATURE_AND_RESULT_WITH_NO_ADDITIONAL_KEYS",
                "EXACT_TWELVE_OBJECT_ORDER_SPK_DESIGNATION_FULL_NAME_AND_FROZEN_SOLUTION_ID_ROSTER",
                "EXACT_QUERY_COMMAND_CENTER_EPOCH_TIME_SCALE_FRAME_UNITS_TABLE_CORRECTIONS_CSV_LABELS_AND_FRACSEC",
                "EXACT_PROVIDER_TARGET_CENTER_AND_SMALL_PERTURBER_SOURCE_HEADERS_DE441_AND_SB441_N16",
                "EXACT_STANDARD_E_Q_NODE_PERI_I_DECIMAL_TOKENS_AND_TP_BINARY64_ULP_DISTANCE_LE_1",
                "EXACTLY_ONE_FINITE_SIX_COMPONENT_VECTOR_ROW_AT_JD_TDB_2461200_5_PER_OBJECT",
                "EXACT_RESPONSE_BYTE_HASH_ROSTER_STRICT_JSON_PARSE_AND_CANONICAL_CUSTODY_REPLAY",
            ],
            "raw_json_envelope": {
                "maximum_bytes_inclusive": MAX_RESPONSE_BYTES,
                "byte_ceiling_enforced_before_utf8_json_decode": True,
                "exact_top_level_keys": ["signature", "result"],
                "additional_top_level_keys_forbidden": True,
            },
            "provider_signature": {"source": "NASA/JPL Horizons API", "version": "1.2"},
            "identity": {
                "target_full_name_match": "EXACT_STRING_NOT_SUBSTRING",
                "solution_id_match": "EXACT_DECIMAL_INTEGER_TOKEN",
                "center_header": "Sun (10)",
                "center_source": "DE441",
                "small_perturbers": "Yes",
                "small_perturber_source": "SB441-N16",
            },
            "printed_osculating_elements": {
                "epoch": TARGET_JD_TDB_DECIMAL,
                "epoch_scale": "TDB",
                "exact_token_labels": ["e", "q", "node", "peri", "i"],
                "tp_gate": "ABS_ORDERED_BINARY64_REPRESENTATION_DISTANCE_LE_1_ULP",
                "source": "BOUND_SBDB_STANDARD_EPOCH_ELEMENT_TOKENS_IN_INPUT_TARGET_ROSTER",
            },
            "vector": {
                "row_count": 1,
                "component_count": 6,
                "component_order": ["x_AU", "y_AU", "z_AU", "vx_AU_per_day", "vy_AU_per_day", "vz_AU_per_day"],
                "all_components_finite_binary64": True,
                "csv_calendar_token_requires_fractional_seconds": True,
                "frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
                "corrections": "GEOMETRIC_NONE",
            },
            "pure_response_gate_function": "validate_horizons_response",
        },
        "diagnostic_role": {
            "classification": "COMMON_EPOCH_NOMINAL_CARTESIAN_PHYSICAL_MODEL_DISCREPANCY_DIAGNOSTIC_ONLY",
            "coverage": "NOMINAL_TRACER_ORDER_0_FOR_ALL_TWELVE_OBJECTS",
            "numerical_bridge_acceptance_gate": False,
            "horizons_force_model_equivalence_claimed": False,
            "residual_pass_tolerance": None,
            "residual_pass_fail_decision_authorized": False,
            "future_residuals_to_retain": "EXACT_COMPONENT_NORM_AND_RADIAL_TRANSVERSE_NORMAL_VALUES_WITHOUT_PASS_FAIL_THRESHOLD",
            "reason_no_tolerance": "HORIZONS_SB441_N16_AND_THE_PARENT_RESTRICTED_RESOLVED11_BRIDGE_HAVE_DIFFERENT_FORCE_MODELS",
        },
        "future_acquisition_contract": {
            "implemented": False,
            "successor_commit_required": True,
            "result_relative_path": "runs/tno_common_epoch_horizons_oracle_v1_result",
            "network_transport_attestation_by_retained_JSON_bytes_alone": False,
            "raw_response_count": len(TARGETS),
            "raw_response_roster": [f"raw/{target.spk_id}.json" for target in TARGETS],
            "maximum_raw_response_bytes_each_inclusive": MAX_RESPONSE_BYTES,
            "canonical_custody_required": True,
            "response_byte_length_and_sha256_required": True,
            "no_retry_selection_or_post_response_gate_tuning": True,
        },
        "future_custody_contract": {
            "schema": CUSTODY_SCHEMA,
            "status_exact": "RETAINED_COMMON_EPOCH_HORIZONS_NOMINAL_ORACLE_DIAGNOSTIC_INPUT_ONLY",
            "strict_canonical_json": "UTF8_ASCII_SUBSET_SORT_KEYS_COMPACT_SEPARATORS_ENSURE_ASCII_ALLOW_NAN_FALSE_ONE_TRAILING_LF",
            "content_hash": {
                "algorithm": "SHA-256",
                "domain_separator_ascii_with_terminal_nul": CUSTODY_DOMAIN.decode("ascii"),
                "domain_separator_hex": CUSTODY_DOMAIN.hex(),
                "construction": "SHA256(DOMAIN_SEPARATOR || CANONICAL_JSON_OF_EXACT_TOP_LEVEL_OBJECT_WITH_CONTENT_SHA256_KEY_REMOVED)",
            },
            "exact_top_level_key_roster_in_lexicographic_canonical_order": list(CUSTODY_TOP_LEVEL_KEYS),
            "exact_top_level_types": {
                "claims": "exact_object",
                "content_sha256": "lowercase_hex_string_length_64",
                "nonclaims": "exact_object",
                "ordered_vector_sha256": "lowercase_hex_string_length_64",
                "preregistration_identity": "exact_object",
                "protocol_content_sha256": "lowercase_hex_string_length_64",
                "protocol_file_sha256": "lowercase_hex_string_length_64",
                "record_count": "exact_integer_not_boolean_equal_12",
                "records": "exact_array_length_12_in_frozen_target_order",
                "schema": "exact_string",
                "status": "exact_string",
            },
            "preregistration_identity": {
                "exact_key_roster_in_lexicographic_canonical_order": list(CUSTODY_PREREGISTRATION_KEYS),
                "exact_types": {
                    "commit": "lowercase_git_sha1_string_length_40",
                    "protocol_content_sha256": "lowercase_hex_string_length_64",
                    "protocol_file_bytes": "positive_exact_integer_not_boolean",
                    "protocol_file_sha256": "lowercase_hex_string_length_64",
                    "source_bytes": "positive_exact_integer_not_boolean",
                    "source_sha256": "lowercase_hex_string_length_64",
                    "tag": "exact_string",
                    "tag_object": "lowercase_git_sha1_string_length_40",
                    "tag_payload_bytes": "positive_exact_integer_not_boolean",
                    "tag_payload_sha256": "lowercase_hex_string_length_64",
                    "test_bytes": "positive_exact_integer_not_boolean",
                    "test_sha256": "lowercase_hex_string_length_64",
                    "tree": "lowercase_git_sha1_string_length_40",
                },
                "tag_exact": FUTURE_PREREGISTRATION_TAG,
                "all_values_are_successor_compile_time_literals_verified_before_request_zero": True,
            },
            "record_contract": {
                "exact_key_roster_in_lexicographic_canonical_order": list(CUSTODY_RECORD_KEYS),
                "exact_scalar_types": {
                    "calendar_tdb": "string",
                    "center": "string",
                    "corrections": "string",
                    "epoch_jd_tdb": "string",
                    "frame": "string",
                    "full_name": "string",
                    "object_id": "string",
                    "order": "integer_not_boolean",
                    "planetary_ephemeris": "string",
                    "primary_designation": "string",
                    "query_url": "string",
                    "raw_bytes": "integer_not_boolean_0_through_1048576",
                    "raw_sha256": "lowercase_hex_string_length_64",
                    "retained_relative_path": "string",
                    "small_body_perturbers": "string",
                    "solution_id": "integer_not_boolean",
                    "spk_id": "integer_not_boolean",
                    "tp_binary64_ulp_distance": "integer_not_boolean_0_or_1",
                },
                "exact_array_types_and_lengths": {
                    "position_AU_binary64_hex": "array_of_3_canonical_finite_binary64_hex_strings",
                    "position_AU_decimal_tokens": "array_of_3_exact_provider_decimal_strings",
                    "printed_element_binary64_hex": "array_of_6_canonical_finite_binary64_hex_strings",
                    "printed_element_decimal_tokens": "array_of_6_exact_provider_decimal_strings",
                    "printed_element_labels": "exact_array_e_q_tp_node_peri_i",
                    "vector_component_labels": "exact_array_x_y_z_vx_vy_vz_with_frozen_units",
                    "velocity_AU_per_day_binary64_hex": "array_of_3_canonical_finite_binary64_hex_strings",
                    "velocity_AU_per_day_decimal_tokens": "array_of_3_exact_provider_decimal_strings",
                },
                "record_array_order_object_ids": [target.object_id for target in TARGETS],
                "record_array_order_spk_ids": [target.spk_id for target in TARGETS],
                "record_array_order_solution_ids": list(SOLUTION_IDS),
                "printed_element_order": list(ELEMENT_LABELS),
                "vector_component_order": list(VECTOR_COMPONENT_LABELS),
                "fixed_values": {
                    "epoch_jd_tdb": TARGET_JD_TDB_DECIMAL,
                    "center": "SUN_NAIF_10",
                    "planetary_ephemeris": "DE441",
                    "small_body_perturbers": "SB441-N16",
                    "frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
                    "corrections": "GEOMETRIC_NONE",
                },
                "raw_binding": "RAW_BYTES_AND_SHA256_MATCH_THE_EXACT_RETAINED_RAW_FILE;EVERY_PARSED_TOKEN_AND_BINARY64_HEX_VALUE_IS_REBUILT_FROM_THOSE_BYTES",
            },
            "ordered_vector_digest": {
                "custody_field": "ordered_vector_sha256",
                "algorithm": "SHA-256",
                "domain_separator_ascii_with_terminal_nul": VECTOR_DOMAIN.decode("ascii"),
                "domain_separator_hex": VECTOR_DOMAIN.hex(),
                "shape": [len(TARGETS), len(VECTOR_COMPONENT_LABELS)],
                "payload_bytes": len(TARGETS) * len(VECTOR_COMPONENT_LABELS) * 8,
                "byte_encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER_NO_PADDING",
                "row_order": "FROZEN_TARGET_ORDER_ZERO_THROUGH_ELEVEN",
                "component_order": list(VECTOR_COMPONENT_LABELS),
                "construction": "SHA256(DOMAIN_SEPARATOR || CONCAT_FOR_EACH_RECORD_POSITION_BINARY64_XYZ_THEN_VELOCITY_BINARY64_XYZ_PACKED_WITH_PYTHON_STRUCT_FORMAT_<6d)",
                "decimal_token_to_hex_crosscheck_required_before_digest": True,
                "pure_digest_function": "ordered_vector_sha256",
            },
            "claims": {
                "exact_key_roster_in_lexicographic_canonical_order": sorted(CUSTODY_CLAIMS),
                "exact_values": dict(CUSTODY_CLAIMS),
            },
            "nonclaims": {
                "exact_key_roster_in_lexicographic_canonical_order": sorted(CUSTODY_NONCLAIMS),
                "exact_values": dict(CUSTODY_NONCLAIMS),
            },
        },
        "future_result_publication_contract": {
            "exact_result_root_roster": list(RESULT_ROOT_ROSTER),
            "exact_raw_directory_roster": list(RESULT_RAW_ROSTER),
            "exact_logical_file_roster": ["CUSTODY.json", *[f"raw/{name}" for name in RESULT_RAW_ROSTER]],
            "directory_mode_octal": "0700",
            "file_mode_octal": "0600",
            "every_directory_must_be_real_owned_by_effective_uid": True,
            "every_file_must_be_regular_owned_by_effective_uid_with_nlink_1": True,
            "symlinks_hardlinks_special_files_and_extra_entries": "FORBIDDEN",
            "private_staging": {
                "required": True,
                "same_parent_and_filesystem_as_final_root": True,
                "unpredictable_exclusive_name": True,
                "mode_octal": "0700",
                "visible_final_root_before_commit": False,
            },
            "ordered_publication_phases": [
                "CREATE_PRIVATE_OWNER_ONLY_STAGING_ROOT_AND_RAW_DIRECTORY",
                "FOR_EACH_EXACT_REQUEST_STREAM_CAP_VALIDATE_AND_WRITE_EXACT_RAW_BYTES_CREATE_ONLY_MODE_0600_THEN_FSYNC",
                "REBUILD_ALL_TWELVE_RECORDS_AND_ORDERED_VECTOR_DIGEST_FROM_STAGED_RAW_BYTES",
                "BUILD_STRICT_CANONICAL_DOMAIN_HASHED_CUSTODY",
                "WRITE_AND_FSYNC_DOT_CUSTODY_JSON_PARTIAL_MODE_0600",
                "ATOMIC_RENAME_NOREPLACE_PARTIAL_TO_CUSTODY_JSON_AS_THE_LAST_STAGED_FILE_AND_COMMIT_MARKER",
                "FSYNC_STAGING_ROOT_AND_RUN_COMPLETE_OFFLINE_REPLAY_VALIDATION",
                "ATOMIC_RENAMEAT2_RENAME_NOREPLACE_STAGING_ROOT_TO_FINAL_ROOT",
                "FSYNC_FINAL_PARENT_DIRECTORY",
            ],
            "custody_is_last_file_and_commit_marker": True,
            "final_root_publication": "ONE_ATOMIC_RENAMEAT2_RENAME_NOREPLACE_OF_THE_COMPLETE_VALIDATED_STAGING_ROOT",
            "existing_final_root": "REFUSE_WITHOUT_READING_WRITING_REPLACING_OR_REMOVING_IT",
            "prepublication_failure": "CLOSE_DESCRIPTORS_REMOVE_ONLY_THIS_PRIVATE_STAGING_TREE_AND_LEAVE_FINAL_ROOT_ABSENT",
            "crash_before_final_rename": "MAY_LEAVE_ONLY_A_PRIVATE_STAGING_TREE_WITH_NO_FINAL_RESULT_NAME;NEVER_COUNTS_AS_A_RESULT",
            "failure_after_complete_final_rename": "FINAL_ROOT_IS_STRUCTURALLY_COMPLETE_NOT_PARTIAL;REPORT_DURABILITY_INDETERMINATE_AND_REQUIRE_OFFLINE_REPLAY_BEFORE_USE",
            "partial_final_result_semantics": "FORBIDDEN_AND_NEVER_COUNTS_AS_SUCCESS",
            "file_and_directory_fsync_required": True,
        },
        "future_offline_replay_validator": {
            "required_successor_cli": "validate-result",
            "network_access": False,
            "read_only": True,
            "must_not_mutate_timestamps_content_permissions_or_rosters": True,
            "validation_order": [
                "OPEN_FINAL_ROOT_AND_RAW_DIRECTORY_WITH_DIRECTORY_FDS_AND_NOFOLLOW",
                "VERIFY_EXACT_ROOT_AND_RAW_ROSTERS_MODES_OWNERSHIP_REGULARITY_AND_NLINK_ONE",
                "READ_STRICT_CANONICAL_CUSTODY_WITH_HELD_FD_AND_BEFORE_AFTER_IDENTITY_CHECK",
                "VERIFY_EXACT_CUSTODY_SCHEMA_KEY_ROSTERS_TYPES_VALUES_AND_DOMAIN_SEPARATED_CONTENT_HASH",
                "FOR_EACH_RAW_FILE_IN_FROZEN_ORDER_ENFORCE_1048576_BYTE_CAP_BEFORE_JSON_DECODE_AND_VERIFY_RECORDED_BYTE_LENGTH_SHA256",
                "REPLAY_VALIDATE_HORIZONS_RESPONSE_AND_REQUIRE_EXACT_REBUILT_RECORD_EQUALITY",
                "RECOMPUTE_DOMAIN_SEPARATED_ORDERED_12X6_VECTOR_DIGEST",
                "REBUILD_COMPLETE_CUSTODY_OBJECT_AND_REQUIRE_EXACT_CANONICAL_BYTE_EQUALITY",
                "RECHECK_DIRECTORY_ROSTERS_AND_HELD_FILE_IDENTITIES_BEFORE_SUCCESS",
            ],
            "provider_contact_or_dns_tls_http_use": "FORBIDDEN",
            "mismatch_action": "FAIL_CLOSED_WITH_NO_REPAIR_REWRITE_OR_PARTIAL_ACCEPTANCE",
            "custody_missing": "NO_COMMITTED_RESULT_EXISTS",
        },
        "claims": {
            "prepare_only_protocol_frozen": True,
            "all12_designation_queries_frozen": True,
            "all12_sbdb_solution_and_element_lineage_frozen": True,
            "parent_bridge_ancestry_bound": True,
            "future_preregistration_tag_and_phase_order_frozen": True,
            "future_transport_and_streaming_contract_frozen": True,
            "future_custody_and_publication_contract_frozen": True,
            "provider_contacted": False,
            "responses_acquired": False,
            "common_epoch_nominal_vectors_retained": False,
            "common_epoch_nominal_oracle_bound": False,
            "parent_bridge_oracle_blocker_resolved": False,
            "residual_tolerance_defined": False,
            "dynamics_executed": False,
            "common_epoch_bridge_executed": False,
            "common_epoch_bridge_qualified": False,
            "classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "claim_ceiling": "PREPARED_FUTURE_COMMON_EPOCH_HORIZONS_NOMINAL_ORACLE_ACQUISITION_CONTRACT_ONLY",
        "publication": {
            "prepared_roster": ["PROTOCOL.json"],
            "root_must_not_exist": True,
            "create_only_no_overwrite": True,
            "directory_mode_octal": "0700",
            "file_mode_octal": "0600",
            "atomic_noreplace_publication": True,
            "file_and_directory_fsync_required": True,
            "validation_is_read_only": True,
            "filesystem_trust_boundary": "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        },
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    return protocol


def _one_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise HorizonsOracleProtocolError(f"Horizons {label} count differs")
    return matches[0]


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def validate_horizons_response(payload: bytes, target: Target) -> dict[str, object]:
    """Apply the frozen pure response gates without performing I/O."""
    if type(payload) is not bytes:
        raise HorizonsOracleProtocolError("Horizons response must be exact bytes")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise HorizonsOracleProtocolError("Horizons response exceeds the 1048576-byte ceiling")
    value = _strict_json(payload)
    if set(value) != {"signature", "result"}:
        raise HorizonsOracleProtocolError("Horizons top-level key roster differs")
    if "error" in value:
        raise HorizonsOracleProtocolError("Horizons returned an error")
    if value.get("signature") != {"source": "NASA/JPL Horizons API", "version": "1.2"}:
        raise HorizonsOracleProtocolError("Horizons provider signature differs")
    result = value.get("result")
    if type(result) is not str:
        raise HorizonsOracleProtocolError("Horizons result must be a string")
    target_match = _one_match(_TARGET_LINE, result, "target header")
    center_match = _one_match(_CENTER_LINE, result, "center header")
    small_match = _one_match(_SMALL_LINE, result, "small-perturber header")
    epoch_match = _one_match(_EPOCH_LINE, result, "element epoch")
    element_match = _one_match(_ELEMENT_LINES, result, "element rows")
    if target_match.group(1).strip() != target.full_name:
        raise HorizonsOracleProtocolError("Horizons target full name differs")
    if target_match.group(2) != str(target.solution_id):
        raise HorizonsOracleProtocolError("Horizons solution ID differs")
    if center_match.group(1) != "DE441" or small_match.group(1) != "SB441-N16":
        raise HorizonsOracleProtocolError("Horizons DE441/SB441-N16 source differs")
    try:
        epoch = Decimal(epoch_match.group(1))
    except InvalidOperation as exc:
        raise HorizonsOracleProtocolError("Horizons element epoch is not decimal") from exc
    if epoch != Decimal(TARGET_JD_TDB_DECIMAL) or epoch_match.group(2) != "TDB":
        raise HorizonsOracleProtocolError("Horizons element epoch or scale differs")
    observed_elements = tuple(element_match.groups())
    for index in (0, 1, 3, 4, 5):
        if observed_elements[index] != target.element_tokens[index]:
            raise HorizonsOracleProtocolError(f"Horizons exact {ELEMENT_LABELS[index]} token differs")
    observed_tp = _float_token(observed_elements[2], "Horizons printed tp")
    expected_tp = _float_token(target.element_tokens[2], "frozen SBDB tp")
    tp_ulp_distance = abs(_ordered_binary64(observed_tp) - _ordered_binary64(expected_tp))
    if tp_ulp_distance > 1:
        raise HorizonsOracleProtocolError("Horizons tp exceeds one binary64 ULP")
    required_text = (
        "Output units    : AU-D",
        "Output type     : GEOMETRIC cartesian states",
        "Reference frame : Ecliptic of J2000.0",
        "Initial IAU76/J2000 heliocentric ecliptic osculating elements",
        "Geometric state vectors have NO corrections or aberrations applied.",
        "Barycentric Dynamical Time (\"TDB\" or T_eph) output was requested.",
    )
    if any(result.count(text) != 1 for text in required_text):
        raise HorizonsOracleProtocolError("Horizons frame, unit, time, or correction semantics differ")
    if result.count("$$SOE") != 1 or result.count("$$EOE") != 1:
        raise HorizonsOracleProtocolError("Horizons state delimiter count differs")
    start = result.index("$$SOE") + len("$$SOE")
    stop = result.index("$$EOE", start)
    rows = [line for line in result[start:stop].splitlines() if line.strip()]
    if len(rows) != 1:
        raise HorizonsOracleProtocolError("Horizons response must contain exactly one vector row")
    columns = [item.strip() for item in rows[0].split(",")]
    if columns and columns[-1] == "":
        columns.pop()
    if len(columns) != 8:
        raise HorizonsOracleProtocolError("Horizons CSV vector column count differs")
    epoch_token, calendar_token, *vector_tokens = columns
    try:
        vector_epoch = Decimal(epoch_token)
    except InvalidOperation as exc:
        raise HorizonsOracleProtocolError("Horizons vector epoch is not decimal") from exc
    if vector_epoch != Decimal(TARGET_JD_TDB_DECIMAL):
        raise HorizonsOracleProtocolError("Horizons vector epoch differs")
    if re.fullmatch(r"(?:A\.D\.|B\.C\.).*\d{2}:\d{2}:\d{2}\.\d+", calendar_token) is None:
        raise HorizonsOracleProtocolError("Horizons calendar token lacks fractional seconds")
    vector = tuple(_float_token(token, f"Horizons vector component {index}") for index, token in enumerate(vector_tokens))
    if len(vector) != 6:
        raise HorizonsOracleProtocolError("Horizons vector dimension differs")
    return {
        "calendar_tdb": calendar_token,
        "center": "SUN_NAIF_10",
        "corrections": "GEOMETRIC_NONE",
        "epoch_jd_tdb": TARGET_JD_TDB_DECIMAL,
        "frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
        "full_name": target.full_name,
        "object_id": target.object_id,
        "order": target.order,
        "planetary_ephemeris": "DE441",
        "position_AU_binary64_hex": [value.hex() for value in vector[:3]],
        "position_AU_decimal_tokens": vector_tokens[:3],
        "primary_designation": target.primary_designation,
        "printed_element_binary64_hex": [_float_token(token, f"printed element {index}").hex() for index, token in enumerate(observed_elements)],
        "printed_element_decimal_tokens": list(observed_elements),
        "printed_element_labels": list(ELEMENT_LABELS),
        "query_url": target.query_url,
        "raw_bytes": len(payload),
        "raw_sha256": _sha256(payload),
        "retained_relative_path": f"raw/{target.spk_id}.json",
        "small_body_perturbers": "SB441-N16",
        "solution_id": target.solution_id,
        "spk_id": target.spk_id,
        "tp_binary64_ulp_distance": tp_ulp_distance,
        "vector_component_labels": list(VECTOR_COMPONENT_LABELS),
        "velocity_AU_per_day_binary64_hex": [value.hex() for value in vector[3:]],
        "velocity_AU_per_day_decimal_tokens": vector_tokens[3:],
    }


def ordered_vector_sha256(records: list[Mapping[str, Any]]) -> str:
    """Hash the frozen target-major 12x6 binary64 vector roster."""
    if type(records) is not list or len(records) != len(TARGETS):
        raise HorizonsOracleProtocolError("custody vector record roster differs")
    encoded = bytearray()
    for target, record in zip(TARGETS, records, strict=True):
        if type(record) is not dict or set(record) != set(CUSTODY_RECORD_KEYS):
            raise HorizonsOracleProtocolError("custody vector record key roster differs")
        if (
            type(record.get("order")) is not int
            or record["order"] != target.order
            or type(record.get("spk_id")) is not int
            or record["spk_id"] != target.spk_id
            or record.get("object_id") != target.object_id
            or record.get("vector_component_labels") != list(VECTOR_COMPONENT_LABELS)
        ):
            raise HorizonsOracleProtocolError("custody vector record order or identity differs")
        decimal_tokens = record.get("position_AU_decimal_tokens")
        velocity_tokens = record.get("velocity_AU_per_day_decimal_tokens")
        position_hex = record.get("position_AU_binary64_hex")
        velocity_hex = record.get("velocity_AU_per_day_binary64_hex")
        arrays = (decimal_tokens, velocity_tokens, position_hex, velocity_hex)
        if any(type(array) is not list or len(array) != 3 for array in arrays):
            raise HorizonsOracleProtocolError("custody vector component array differs")
        values: list[float] = []
        for index, (decimal_token, hex_token) in enumerate(
            zip(decimal_tokens + velocity_tokens, position_hex + velocity_hex, strict=True)
        ):
            value = _float_token(decimal_token, f"custody vector decimal component {index}")
            if type(hex_token) is not str or value.hex() != hex_token:
                raise HorizonsOracleProtocolError("custody vector decimal/hex binding differs")
            try:
                reconstructed = float.fromhex(hex_token)
            except ValueError as exc:
                raise HorizonsOracleProtocolError("custody vector hex token differs") from exc
            if not math.isfinite(reconstructed) or reconstructed.hex() != hex_token:
                raise HorizonsOracleProtocolError("custody vector hex token is noncanonical")
            values.append(reconstructed)
        encoded.extend(struct.pack("<6d", *values))
    if len(encoded) != len(TARGETS) * len(VECTOR_COMPONENT_LABELS) * 8:
        raise HorizonsOracleProtocolError("custody vector encoding length differs")
    return _sha256(VECTOR_DOMAIN + bytes(encoded))


def _custody_content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(CUSTODY_DOMAIN + _canonical_json(record))


def _validate_protocol(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or value.get("schema") != PROTOCOL_SCHEMA:
        raise HorizonsOracleProtocolError("protocol schema differs")
    if value.get("content_sha256") != _content_sha256(value):
        raise HorizonsOracleProtocolError("protocol content SHA-256 differs")
    authority = value.get("authority")
    claims = value.get("claims")
    diagnostic = value.get("diagnostic_role")
    future = value.get("future_preregistration")
    transport = value.get("future_transport_contract")
    custody = value.get("future_custody_contract")
    result_publication = value.get("future_result_publication_contract")
    offline = value.get("future_offline_replay_validator")
    custody_record_contract = custody.get("record_contract") if type(custody) is dict else None
    if (
        type(authority) is not dict
        or authority.get("network_access") is not False
        or authority.get("acquisition_in_this_artifact") is not False
        or authority.get("execution_in_this_artifact") is not False
        or authority.get("public_cli") != ["build", "audit-prerequisites", "prepare", "validate-prepared"]
        or authority.get("explicitly_absent_public_cli") != ["acquire", "execute", "execute-worker", "resume", "validate-result"]
    ):
        raise HorizonsOracleProtocolError("prepare-only authority differs")
    if type(claims) is not dict or claims.get("responses_acquired") is not False or claims.get("dynamics_executed") is not False:
        raise HorizonsOracleProtocolError("claim ceiling differs")
    if type(diagnostic) is not dict or diagnostic.get("residual_pass_tolerance") is not None or diagnostic.get("residual_pass_fail_decision_authorized") is not False:
        raise HorizonsOracleProtocolError("diagnostic no-tolerance contract differs")
    if (
        type(future) is not dict
        or future.get("exact_annotated_tag_name") != FUTURE_PREREGISTRATION_TAG
        or future.get("request_count_allowed_before_all_bindings_pass") != 0
        or type(transport) is not dict
        or transport.get("ordered_exact_urls") != [target.query_url for target in TARGETS]
        or transport.get("requests_per_url") != 1
        or transport.get("retries_for_any_reason") != 0
    ):
        raise HorizonsOracleProtocolError("future tag or transport contract differs")
    if (
        type(custody) is not dict
        or custody.get("schema") != CUSTODY_SCHEMA
        or custody.get("exact_top_level_key_roster_in_lexicographic_canonical_order") != list(CUSTODY_TOP_LEVEL_KEYS)
        or type(custody_record_contract) is not dict
        or custody_record_contract.get("exact_key_roster_in_lexicographic_canonical_order") != list(CUSTODY_RECORD_KEYS)
        or type(result_publication) is not dict
        or result_publication.get("exact_result_root_roster") != list(RESULT_ROOT_ROSTER)
        or result_publication.get("custody_is_last_file_and_commit_marker") is not True
        or type(offline) is not dict
        or offline.get("network_access") is not False
        or offline.get("read_only") is not True
    ):
        raise HorizonsOracleProtocolError("future custody or offline replay contract differs")


def _current_branch() -> str:
    return _git("branch", "--show-current").decode("utf-8").strip()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
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
    for component in reversed((absolute, *absolute.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise HorizonsOracleProtocolError(f"{label} has a symlink ancestor")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise HorizonsOracleProtocolError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise HorizonsOracleProtocolError("renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, source.encode("ascii"), directory_fd, destination.encode("ascii"), 1) != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise HorizonsOracleProtocolError("protocol destination already exists")
        raise HorizonsOracleProtocolError(f"renameat2 failed: {os.strerror(observed_errno)}")


def _validate_prepared_fd(root_fd: int, expected_payload: bytes) -> dict[str, object]:
    root_metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != os.geteuid()
    ):
        raise HorizonsOracleProtocolError("prepared root is not an owner-private directory")
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise HorizonsOracleProtocolError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise HorizonsOracleProtocolError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - len(payload)), len(payload))
            if not chunk:
                raise HorizonsOracleProtocolError("prepared protocol ended early")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise HorizonsOracleProtocolError("prepared protocol changed while held")
    finally:
        os.close(descriptor)
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise HorizonsOracleProtocolError("prepared root changed while held")
    exact_payload = bytes(payload)
    if exact_payload != expected_payload:
        raise HorizonsOracleProtocolError("prepared protocol bytes differ")
    value = _strict_json(exact_payload)
    if exact_payload != _canonical_json(value):
        raise HorizonsOracleProtocolError("prepared protocol is not canonical JSON")
    _validate_protocol(value)
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if _current_branch() != EXPECTED_BRANCH:
        raise HorizonsOracleProtocolError("preparation branch differs")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise HorizonsOracleProtocolError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        parent_fd = os.open(root.parent, flags)
    except OSError as exc:
        raise HorizonsOracleProtocolError("cannot open prepared parent") from exc
    created = False
    prepared: dict[str, object] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise HorizonsOracleProtocolError("prepared root already exists; create-only publication refused") from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            if os.listdir(root_fd):
                raise HorizonsOracleProtocolError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            file_fd = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                os.fchmod(file_fd, 0o600)
                _write_all(file_fd, payload)
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_fd(root_fd, payload)
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
    except HorizonsOracleProtocolError:
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
        raise HorizonsOracleProtocolError("cannot create prepared protocol") from exc
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise HorizonsOracleProtocolError("prepared protocol validation did not complete")
    return prepared


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    protocol = build_protocol()
    expected_payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    _require_no_symlink_ancestors(root, "prepared root")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise HorizonsOracleProtocolError("cannot open prepared root") from exc
    try:
        value = _validate_prepared_fd(root_fd, expected_payload)
    finally:
        os.close(root_fd)
    if value != protocol:
        raise HorizonsOracleProtocolError("prepared protocol differs from rebuilt protocol")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="build and print the protocol without writing")
    subparsers.add_parser("audit-prerequisites", help="verify every frozen input without writing")
    prepare_parser = subparsers.add_parser("prepare", help="create only PROTOCOL.json")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate-prepared", help="read-only validation of a prepared root")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        value = build_protocol()
        print(_canonical_json(value).decode("ascii"), end="")
    elif args.command == "audit-prerequisites":
        value = audit_prerequisites()
        print(json.dumps({"binding_count": len(INPUT_BINDINGS), "parent_commit": value["parent_preregistration"]["commit"], "status": "PASSED"}, sort_keys=True))
    elif args.command == "prepare":
        value = prepare(args.output_root)
        print(json.dumps({"content_sha256": value["content_sha256"], "status": value["status"]}, sort_keys=True))
    elif args.command == "validate-prepared":
        value = validate_prepared(args.output_root)
        print(json.dumps({"content_sha256": value["content_sha256"], "status": "VALID"}, sort_keys=True))
    else:
        raise HorizonsOracleProtocolError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
