#!/usr/bin/env python3
"""Acquire and validate 12 one-epoch Horizons TNO state-vector oracles.

The retained responses are an independent nominal-state transform oracle only.
They do not authorize clone dynamics, classification, inference, or a Planet
Nine claim.  HTTPS was requested, but the serialized bundle does not prove the
remote server identity or preserve a transport attestation.
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
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_horizons_oracle_v1"
SBDB_CUSTODY = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1" / "CUSTODY.json"

SCHEMA_PROTOCOL = "jx.tno.native-epoch-horizons-oracle.protocol.v1"
SCHEMA_CUSTODY = "jx.tno.native-epoch-horizons-oracle.custody.v1"
DOMAIN_PROTOCOL = b"jx.tno.native-epoch-horizons-oracle.protocol.v1\0"
DOMAIN_CUSTODY = b"jx.tno.native-epoch-horizons-oracle.custody.v1\0"
DOMAIN_ROSTER = b"jx.tno.native-epoch-horizons-oracle.roster.v1\0"

HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_DOC_URL = "https://ssd-api.jpl.nasa.gov/doc/horizons.html"
SBDB_DOC_URL = "https://ssd-api.jpl.nasa.gov/doc/sbdb.html"
GM_URL = "https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck"
GM_BYTES = 15_428
GM_SHA256 = "169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c"
BODY10_GM_TOKEN = "1.3271244004127942E+11"
BODY10_GM_KM3_PER_S2 = Decimal("132712440041.27942")
AU_KM = Decimal("149597870.7")
SECONDS_PER_DAY = Decimal(86_400)
DAYS_PER_JULIAN_YEAR = Decimal("365.25")
BODY10_GM_AU3_PER_DAY2_HEX = "0x1.3649541cf2bdcp-12"
BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX = "0x1.3bd0beccac6afp+5"

SBDB_CUSTODY_FILE_SHA256 = "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288"
SBDB_CUSTODY_CONTENT_SHA256 = "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f"

CLASSIFICATION = "RETAINED_HORIZONS_NOMINAL_NATIVE_EPOCH_STATE_ORACLE_INPUT_ONLY"
CLAIMS = {
    "twelve_nominal_native_epoch_vectors_retained": True,
    "twelve_printed_native_epoch_element_sets_retained": True,
    "horizons_response_schema_and_text_contract_validated": True,
    "horizons_gm_source_retained": True,
    "provider_transport_authenticated": False,
    "provider_origin_cryptographically_proven": False,
    "element_to_cartesian_transform_qualified": False,
    "clone_cartesian_states_materialized": False,
    "dynamics_executed": False,
    "classification_generated": False,
    "observational_likelihood_evaluated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class HorizonsOracleError(ValueError):
    """The oracle acquisition, response, or retained custody failed closed."""


@dataclass(frozen=True, slots=True)
class Target:
    object_id: str
    spk_id: int
    primary_designation: str
    covariance_epoch_jd_tdb: str
    orbit_solution_id: str

    @property
    def filename(self) -> str:
        return f"{self.spk_id}.json"

    @property
    def query_parameters(self) -> tuple[tuple[str, str], ...]:
        return (
            ("format", "json"),
            ("COMMAND", f"'DES={self.spk_id};'"),
            ("OBJ_DATA", "'NO'"),
            ("MAKE_EPHEM", "'YES'"),
            ("EPHEM_TYPE", "'VECTORS'"),
            ("CENTER", "'500@10'"),
            ("TLIST", f"'{self.covariance_epoch_jd_tdb}'"),
            ("REF_PLANE", "'ECLIPTIC'"),
            ("REF_SYSTEM", "'J2000'"),
            ("OUT_UNITS", "'AU-D'"),
            ("VEC_TABLE", "'2'"),
            ("VEC_CORR", "'NONE'"),
            ("CSV_FORMAT", "'YES'"),
        )

    @property
    def query_url(self) -> str:
        return f"{HORIZONS_ENDPOINT}?{urlencode(self.query_parameters)}"


TARGETS = (
    Target("jpl-20090377", 20090377, "90377", "2458411.5", "51"),
    Target("jpl-20474640", 20474640, "474640", "2456187.5", "12"),
    Target("jpl-54526835", 54526835, "2023 KQ14", "2460034.5", "3"),
    Target("jpl-20541132", 20541132, "541132", "2458052.5", "7"),
    Target("jpl-50666516", 50666516, "2012 VP113", "2456596.5", "11"),
    Target("jpl-20148209", 20148209, "148209", "2452568.5", "5"),
    Target("jpl-20574372", 20574372, "574372", "2458299.5", "14"),
    Target("jpl-20303775", 20303775, "303775", "2458767.5", "29"),
    Target("jpl-20026181", 20026181, "26181", "2458156.5", "25"),
    Target("jpl-20042301", 20042301, "42301", "2458324.5", "28"),
    Target("jpl-20026375", 20026375, "26375", "2457961.5", "38"),
    Target("jpl-20015874", 20015874, "15874", "2458098.5", "34"),
)

_FLOAT_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_TARGET_LINE = re.compile(r"^Target body name:\s*(.+?)\s*\{source:\s*JPL#(\d+)\}\s*$", re.MULTILINE)
_CENTER_LINE = re.compile(r"^Center body name:\s*Sun \(10\)\s*\{source:\s*(DE441)\}\s*$", re.MULTILINE)
_EPOCH_LINE = re.compile(r"^\s*EPOCH=\s*([0-9.]+)\s*!", re.MULTILINE)
_ELEMENT_LINES = re.compile(
    r"^\s*EC=\s*([^\s]+)\s+QR=\s*([^\s]+)\s+TP=\s*([^\s]+)\s*$"
    r"\n^\s*OM=\s*([^\s]+)\s+W=\s*([^\s]+)\s+IN=\s*([^\s]+)\s*$",
    re.MULTILINE,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise HorizonsOracleError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HorizonsOracleError(f"JSON contains non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HorizonsOracleError("response is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise HorizonsOracleError("JSON root must be an object")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HorizonsOracleError(f"cannot open required file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise HorizonsOracleError(f"file is linked or nonregular: {path}")
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
    path_after = path.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise HorizonsOracleError(f"file changed while read: {path}")
    if identity != (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns):
        raise HorizonsOracleError(f"file path changed while read: {path}")
    if len(payload) != before.st_size:
        raise HorizonsOracleError(f"file length changed while read: {path}")
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = _read_regular_bytes(path)
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise HorizonsOracleError(f"retained JSON is not canonical: {path}")
    return value


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _float_token(token: str, label: str) -> float:
    if type(token) is not str or _FLOAT_TOKEN.fullmatch(token.strip()) is None:
        raise HorizonsOracleError(f"{label} is not a decimal float token")
    value = float(token)
    if not math.isfinite(value):
        raise HorizonsOracleError(f"{label} is non-finite")
    return value


def validate_response(payload: bytes, target: Target) -> dict[str, object]:
    value = _strict_json(payload)
    if "error" in value:
        raise HorizonsOracleError("Horizons returned an error")
    signature = value.get("signature")
    if type(signature) is not dict or signature != {
        "source": "NASA/JPL Horizons API",
        "version": "1.2",
    }:
        raise HorizonsOracleError("Horizons signature differs")
    result = value.get("result")
    if type(result) is not str:
        raise HorizonsOracleError("Horizons result must be a string")
    target_match = _TARGET_LINE.search(result)
    center_match = _CENTER_LINE.search(result)
    epoch_match = _EPOCH_LINE.search(result)
    element_match = _ELEMENT_LINES.search(result)
    if (
        target_match is None
        or center_match is None
        or epoch_match is None
        or element_match is None
    ):
        raise HorizonsOracleError("Horizons identity/center/epoch header differs")
    if target_match.group(2) != target.orbit_solution_id:
        raise HorizonsOracleError("Horizons orbit solution ID differs")
    if target.primary_designation not in target_match.group(1):
        raise HorizonsOracleError("Horizons target designation differs")
    if Decimal(epoch_match.group(1)) != Decimal(target.covariance_epoch_jd_tdb):
        raise HorizonsOracleError("Horizons osculation epoch differs")
    required_text = (
        "Output units    : AU-D",
        "Output type     : GEOMETRIC cartesian states",
        "Reference frame : Ecliptic of J2000.0",
        "Initial IAU76/J2000 heliocentric ecliptic osculating elements",
        "Geometric state vectors have NO corrections or aberrations applied.",
        "Barycentric Dynamical Time (\"TDB\" or T_eph) output was requested.",
    )
    if any(text not in result for text in required_text):
        raise HorizonsOracleError("Horizons frame/unit/correction semantics differ")
    try:
        body = result.split("$$SOE\n", 1)[1].split("\n$$EOE", 1)[0]
    except IndexError as exc:
        raise HorizonsOracleError("Horizons state delimiters differ") from exc
    lines = [line for line in body.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HorizonsOracleError("Horizons response must contain exactly one state")
    columns = [item.strip() for item in lines[0].split(",")]
    if columns and columns[-1] == "":
        columns.pop()
    if len(columns) != 8:
        raise HorizonsOracleError("Horizons CSV state column count differs")
    epoch_token, calendar_token, *vector_tokens = columns
    if Decimal(epoch_token) != Decimal(target.covariance_epoch_jd_tdb):
        raise HorizonsOracleError("Horizons vector epoch differs")
    if not calendar_token.startswith(("A.D. ", "B.C. ")):
        raise HorizonsOracleError("Horizons calendar token differs")
    vector = tuple(_float_token(token, f"state[{index}]") for index, token in enumerate(vector_tokens))
    if len(vector) != 6:
        raise HorizonsOracleError("Horizons vector dimension differs")
    element_tokens = list(element_match.groups())
    element_values = tuple(
        _float_token(token, f"printed element[{index}]")
        for index, token in enumerate(element_tokens)
    )
    eccentricity, perihelion, _tp, _node, _peri, inclination = element_values
    if not 0.0 < eccentricity < 1.0 or perihelion <= 0.0 or not 0.0 <= inclination <= 180.0:
        raise HorizonsOracleError("Horizons printed osculating elements are nonphysical")
    return {
        "object_id": target.object_id,
        "spk_id": target.spk_id,
        "primary_designation": target.primary_designation,
        "orbit_solution_id": target.orbit_solution_id,
        "target_name": target_match.group(1).strip(),
        "center": "SUN_NAIF_10",
        "planetary_ephemeris": center_match.group(1),
        "covariance_epoch_jd_tdb": target.covariance_epoch_jd_tdb,
        "query_url": target.query_url,
        "retained_relative_path": f"raw/{target.filename}",
        "raw_byte_count": len(payload),
        "raw_sha256": _sha256(payload),
        "calendar_tdb": calendar_token,
        "printed_osculating_element_labels": ["e", "q", "tp", "node", "peri", "i"],
        "printed_osculating_element_units": [None, "au", "JD TDB", "deg", "deg", "deg"],
        "printed_osculating_element_decimal_tokens": element_tokens,
        "printed_osculating_element_float_hex": [value.hex() for value in element_values],
        "position_AU_decimal_tokens": vector_tokens[:3],
        "velocity_AU_per_day_decimal_tokens": vector_tokens[3:],
        "position_AU_float_hex": [value.hex() for value in vector[:3]],
        "velocity_AU_per_day_float_hex": [value.hex() for value in vector[3:]],
        "frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
        "corrections": "GEOMETRIC_NONE",
    }


def _gm_record(payload: bytes) -> dict[str, object]:
    if len(payload) != GM_BYTES or _sha256(payload) != GM_SHA256:
        raise HorizonsOracleError("gm_Horizons.pck bytes differ")
    text = payload.decode("ascii")
    matches = re.findall(r"BODY10_GM\s*=\s*\(\s*([0-9.Ee+-]+)\s*\)", text)
    if not matches or any(token != BODY10_GM_TOKEN for token in matches):
        raise HorizonsOracleError("BODY10_GM token differs")
    with localcontext() as context:
        context.prec = 100
        daily = BODY10_GM_KM3_PER_S2 * SECONDS_PER_DAY**2 / AU_KM**3
        yearly = daily * DAYS_PER_JULIAN_YEAR**2
    daily_float = float(daily)
    yearly_float = float(yearly)
    if daily_float.hex() != BODY10_GM_AU3_PER_DAY2_HEX:
        raise HorizonsOracleError("daily BODY10_GM binary64 differs")
    if yearly_float.hex() != BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX:
        raise HorizonsOracleError("yearly BODY10_GM binary64 differs")
    return {
        "official_url": GM_URL,
        "retained_relative_path": "source/gm_Horizons.pck",
        "raw_byte_count": len(payload),
        "raw_sha256": _sha256(payload),
        "body10_gm_km3_per_s2_decimal": str(BODY10_GM_KM3_PER_S2),
        "body10_gm_AU3_per_day2_binary64_hex": daily_float.hex(),
        "body10_gm_AU3_per_julian_year2_binary64_hex": yearly_float.hex(),
        "conversion_constants": {
            "AU_km_decimal": str(AU_KM),
            "seconds_per_day": int(SECONDS_PER_DAY),
            "days_per_julian_year_decimal": str(DAYS_PER_JULIAN_YEAR),
        },
    }


def build_protocol() -> dict[str, object]:
    custody_payload = _read_regular_bytes(SBDB_CUSTODY)
    if _sha256(custody_payload) != SBDB_CUSTODY_FILE_SHA256:
        raise HorizonsOracleError("bound SBDB custody file differs")
    custody = _strict_json(custody_payload)
    if custody.get("content_sha256") != SBDB_CUSTODY_CONTENT_SHA256:
        raise HorizonsOracleError("bound SBDB custody content differs")
    roster = {
        (row.get("object_id"), row.get("spk_id")): (
            row.get("primary_designation"), row.get("covariance_epoch_jd_tdb"), row.get("orbit_solution_id")
        )
        for row in custody.get("records", [])
        if type(row) is dict
    }
    for target in TARGETS:
        if roster.get((target.object_id, target.spk_id)) != (
            target.primary_designation,
            target.covariance_epoch_jd_tdb,
            target.orbit_solution_id,
        ):
            raise HorizonsOracleError("target roster differs from SBDB custody")
    record: dict[str, object] = {
        "schema": SCHEMA_PROTOCOL,
        "classification": CLASSIFICATION,
        "status": "ACQUISITION_INPUT_ONLY_NOT_TRANSFORM_QUALIFICATION",
        "targets": [
            {
                "object_id": target.object_id,
                "spk_id": target.spk_id,
                "primary_designation": target.primary_designation,
                "covariance_epoch_jd_tdb": target.covariance_epoch_jd_tdb,
                "orbit_solution_id": target.orbit_solution_id,
                "query_url": target.query_url,
            }
            for target in TARGETS
        ],
        "query_contract": {
            "endpoint": HORIZONS_ENDPOINT,
            "center": "500@10",
            "ephemeris_type": "VECTORS",
            "time_selection": "EXACT_SINGLE_TLIST_JD_TDB_PER_OBJECT",
            "reference_plane": "ECLIPTIC",
            "reference_system": "J2000",
            "output_units": "AU-D",
            "vector_table": 2,
            "vector_corrections": "NONE",
            "csv_format": True,
            "expected_provider_signature_version": "1.2",
            "printed_initial_osculating_elements_required": True,
        },
        "source_binding": {
            "sbdb_custody_relative_path": "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json",
            "sbdb_custody_file_sha256": SBDB_CUSTODY_FILE_SHA256,
            "sbdb_custody_content_sha256": SBDB_CUSTODY_CONTENT_SHA256,
        },
        "gm_source": {
            "official_url": GM_URL,
            "expected_bytes": GM_BYTES,
            "expected_sha256": GM_SHA256,
            "required_body10_gm_token": BODY10_GM_TOKEN,
        },
        "references": [HORIZONS_DOC_URL, SBDB_DOC_URL, GM_URL],
        "implementation": {
            "relative_path": "tools/jx_tno_native_epoch_horizons_oracle_v1.py",
            "file_sha256": _sha256(_read_regular_bytes(Path(__file__))),
        },
        "claims": dict(CLAIMS),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    record["content_sha256"] = _content_sha256(DOMAIN_PROTOCOL, record)
    return record


def build_custody(root: Path, protocol: Mapping[str, Any]) -> dict[str, object]:
    if protocol != build_protocol():
        raise HorizonsOracleError("protocol differs from exact reconstruction")
    records = [
        validate_response(_read_regular_bytes(root / "raw" / target.filename), target)
        for target in TARGETS
    ]
    gm = _gm_record(_read_regular_bytes(root / "source" / "gm_Horizons.pck"))
    roster_sha256 = _sha256(DOMAIN_ROSTER + _canonical_json(records))
    custody: dict[str, object] = {
        "schema": SCHEMA_CUSTODY,
        "classification": CLASSIFICATION,
        "protocol_content_sha256": protocol["content_sha256"],
        "record_count": len(records),
        "records": records,
        "record_roster_sha256": roster_sha256,
        "gm_source": gm,
        "claims": dict(CLAIMS),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    custody["content_sha256"] = _content_sha256(DOMAIN_CUSTODY, custody)
    return custody


def validate_output(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise HorizonsOracleError("oracle root must be a real directory")
    if {entry.name for entry in root.iterdir()} != {"PROTOCOL.json", "CUSTODY.json", "raw", "source"}:
        raise HorizonsOracleError("oracle root roster differs")
    raw = root / "raw"
    source = root / "source"
    if raw.is_symlink() or source.is_symlink() or not raw.is_dir() or not source.is_dir():
        raise HorizonsOracleError("oracle subdirectories must be real")
    if {entry.name for entry in raw.iterdir()} != {target.filename for target in TARGETS}:
        raise HorizonsOracleError("oracle raw roster differs")
    if {entry.name for entry in source.iterdir()} != {"gm_Horizons.pck"}:
        raise HorizonsOracleError("oracle source roster differs")
    protocol = _load_canonical(root / "PROTOCOL.json")
    expected_protocol = build_protocol()
    if protocol != expected_protocol:
        raise HorizonsOracleError("retained protocol differs from exact reconstruction")
    custody = _load_canonical(root / "CUSTODY.json")
    expected_custody = build_custody(root, protocol)
    if custody != expected_custody:
        raise HorizonsOracleError("retained custody differs from exact reconstruction")
    return custody


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "JX-TNO-native-epoch-oracle-v1"})
    with urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise HorizonsOracleError(f"provider returned HTTP {response.status}")
        return response.read()


def acquire(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if root.exists() or root.is_symlink():
        raise HorizonsOracleError("oracle output root already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        (staging / "raw").mkdir()
        (staging / "source").mkdir()
        protocol = build_protocol()
        _write_create_only(staging / "PROTOCOL.json", _canonical_json(protocol))
        for target in TARGETS:
            payload = _fetch(target.query_url)
            validate_response(payload, target)
            _write_create_only(staging / "raw" / target.filename, payload)
        gm_payload = _fetch(GM_URL)
        _gm_record(gm_payload)
        _write_create_only(staging / "source" / "gm_Horizons.pck", gm_payload)
        custody = build_custody(staging, protocol)
        _write_create_only(staging / "CUSTODY.json", _canonical_json(custody))
        validate_output(staging)
        if root.exists() or root.is_symlink():
            raise HorizonsOracleError("oracle output appeared during acquisition")
        os.rename(staging, root)
        return validate_output(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = acquire(args.output_root) if args.command == "acquire" else validate_output(args.output_root)
    except HorizonsOracleError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
