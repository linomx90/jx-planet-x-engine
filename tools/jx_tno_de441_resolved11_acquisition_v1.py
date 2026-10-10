#!/usr/bin/env python3
"""Acquire exact DE441 resolved-11 states at the 12 sealed TNO epochs.

This source-only tool retains Horizons response bytes and reconstructible
custody metadata.  It does not authenticate the remote provider, run dynamics,
compare DE440 with DE441, classify a TNO, or produce a scientific result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1"
STAGE0_PROTOCOL_PATH = (
    PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1" / "PROTOCOL.json"
)
STAGE0_STATES_PATH = (
    PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1" / "STATES.json"
)

HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_DOCUMENTATION = "https://ssd-api.jpl.nasa.gov/doc/horizons.html"
HORIZONS_MANUAL = "https://ssd.jpl.nasa.gov/horizons/manual.html"
EXPECTED_API_SIGNATURE = {"source": "NASA/JPL Horizons API", "version": "1.2"}

SCHEMA_PROTOCOL = "jx.tno.de441-resolved11-native-epochs.protocol.v1"
SCHEMA_CUSTODY = "jx.tno.de441-resolved11-native-epochs.custody.v1"
CLASSIFICATION = "RETAINED_DE441_RESOLVED11_NATIVE_EPOCH_STATE_INPUT_ONLY"
CLAIM_CEILING = "DE441_HORIZONS_RESOLVED11_NATIVE_EPOCH_INPUT_CUSTODY_ONLY"
DOMAIN_PROTOCOL = b"jx.tno.de441-resolved11-native-epochs.protocol.v1\0"
DOMAIN_CUSTODY = b"jx.tno.de441-resolved11-native-epochs.custody.v1\0"
DOMAIN_RAW_ROSTER = b"jx.tno.de441-resolved11-native-epochs.raw-roster.v1\0"
DOMAIN_STATE_ROSTER = b"jx.tno.de441-resolved11-native-epochs.state-roster.v1\0"

STAGE0_PROTOCOL_FILE_SHA256 = (
    "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"
)
STAGE0_PROTOCOL_CONTENT_SHA256 = (
    "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e"
)
STAGE0_STATES_FILE_SHA256 = (
    "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"
)
STAGE0_STATES_CONTENT_SHA256 = (
    "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b"
)
STAGE0_STATES_BINARY_PAYLOAD_SHA256 = (
    "334d3fb579e3be752e44cee91680a0c73bd4ac2cb5ab2bff108d66c389118b83"
)

MAXIMUM_RESPONSE_BYTES = 2_000_000
MAXIMUM_FILE_BYTES = 5_000_000


class AcquisitionError(ValueError):
    """The acquisition contract, provider bytes, or custody failed closed."""


@dataclass(frozen=True, slots=True)
class Epoch:
    object_id: str
    spk_id: int
    jd_tdb: str
    jd_tdb_binary64_hex: str


@dataclass(frozen=True, slots=True)
class Body:
    naif_id: int
    semantic_id: str

    @property
    def filename(self) -> str:
        return f"body_{self.naif_id:03d}.json"


EPOCHS = (
    Epoch("jpl-20090377", 20090377, "2458411.5", "0x1.2c195c0000000p+21"),
    Epoch("jpl-20474640", 20474640, "2456187.5", "0x1.2bd3dc0000000p+21"),
    Epoch("jpl-54526835", 54526835, "2460034.5", "0x1.2c4c140000000p+21"),
    Epoch("jpl-20541132", 20541132, "2458052.5", "0x1.2c0e240000000p+21"),
    Epoch("jpl-50666516", 50666516, "2456596.5", "0x1.2be0a40000000p+21"),
    Epoch("jpl-20148209", 20148209, "2452568.5", "0x1.2b62c40000000p+21"),
    Epoch("jpl-20574372", 20574372, "2458299.5", "0x1.2c15dc0000000p+21"),
    Epoch("jpl-20303775", 20303775, "2458767.5", "0x1.2c247c0000000p+21"),
    Epoch("jpl-20026181", 20026181, "2458156.5", "0x1.2c11640000000p+21"),
    Epoch("jpl-20042301", 20042301, "2458324.5", "0x1.2c16a40000000p+21"),
    Epoch("jpl-20026375", 20026375, "2457961.5", "0x1.2c0b4c0000000p+21"),
    Epoch("jpl-20015874", 20015874, "2458098.5", "0x1.2c0f940000000p+21"),
)

BODIES = (
    Body(10, "SUN"),
    Body(199, "MERCURY"),
    Body(299, "VENUS"),
    Body(399, "EARTH"),
    Body(301, "MOON"),
    Body(4, "MARS_BARYCENTER"),
    Body(5, "JUPITER_BARYCENTER"),
    Body(6, "SATURN_BARYCENTER"),
    Body(7, "URANUS_BARYCENTER"),
    Body(8, "NEPTUNE_BARYCENTER"),
    Body(9, "PLUTO_BARYCENTER"),
)

PROTOCOL_CLAIMS = {
    "stage0_epoch_roster_content_bound": True,
    "exact_twelve_native_epochs_requested": True,
    "exact_resolved_eleven_body_roster_requested": True,
    "de441_provider_responses_retained": False,
    "provider_transport_authenticated": False,
    "provider_origin_cryptographically_proven": False,
    "continuous_custody_proven": False,
    "de440_de441_sensitivity_gate_executed": False,
    "dynamics_executed": False,
    "classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}

CUSTODY_CLAIMS = {
    **PROTOCOL_CLAIMS,
    "de441_provider_responses_retained": True,
    "exact_132_state_rows_retained_and_schema_validated": True,
}

_DECIMAL_TOKEN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?$"
)
_TARGET_LINE = re.compile(
    r"^Target body name:\s*(.+?)\s+\(([0-9]+)\)\s+\{source:\s*(DE441)\}\s*$",
    re.MULTILINE,
)
_CENTER_LINE = re.compile(
    r"^Center body name:\s*Solar System Barycenter \(0\)\s+"
    r"\{source:\s*(DE441)\}\s*$",
    re.MULTILINE,
)


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionError("value cannot be encoded as canonical JSON") from exc
    return text.encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise AcquisitionError("SHA-256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(domain) is not bytes or type(value) is not dict:
        raise AcquisitionError("content-hash inputs have invalid types")
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise AcquisitionError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_FILE_BYTES:
        raise AcquisitionError(f"{label} has an invalid bounded byte size")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AcquisitionError(f"{label} contains non-finite token {token}")
            ),
        )
    except AcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AcquisitionError(f"{label} root must be an object")
    return value


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AcquisitionError(f"cannot open required file: {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAXIMUM_FILE_BYTES
        ):
            raise AcquisitionError(f"required file is linked, nonregular, or oversized: {label}")
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
    except OSError as exc:
        raise AcquisitionError(f"required path disappeared after read: {label}") from exc
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AcquisitionError(f"required file changed while read: {label}")
    if identity != (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    ):
        raise AcquisitionError(f"required file path changed while read: {label}")
    if len(payload) != before.st_size:
        raise AcquisitionError(f"required file length changed while read: {label}")
    return payload


def _load_canonical(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read_regular_bytes(path, label)
    value = _strict_json(payload, label)
    if payload != _canonical_json(value):
        raise AcquisitionError(f"retained JSON is not canonical: {label}")
    return payload, value


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AcquisitionError(f"required directory is absent: {label}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise AcquisitionError(f"required directory is not a real directory: {label}")


def _write_create_only(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise AcquisitionError(f"invalid create-only payload: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise AcquisitionError(f"cannot create artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise AcquisitionError(f"cannot durably publish directory: {path}") from exc


def _require_output_absent(root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(root)))
    if not root.is_absolute() or not root.name:
        raise AcquisitionError("output root must name an absolute child directory")
    try:
        root.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AcquisitionError("cannot inspect output root") from exc
    else:
        raise AcquisitionError("acquisition output root already exists")
    _require_real_directory(root.parent, "output parent")
    return root


def _decimal(token: str, label: str) -> Decimal:
    if type(token) is not str or _DECIMAL_TOKEN.fullmatch(token.strip()) is None:
        raise AcquisitionError(f"{label} is not a decimal token")
    try:
        value = Decimal(token.strip())
    except InvalidOperation as exc:
        raise AcquisitionError(f"{label} is not a valid decimal") from exc
    if not value.is_finite():
        raise AcquisitionError(f"{label} is non-finite")
    return value


def _float_hex(token: str, label: str) -> str:
    value = float(_decimal(token, label))
    if not math.isfinite(value):
        raise AcquisitionError(f"{label} is outside finite binary64")
    return value.hex()


def _query_parameters(body: Body) -> tuple[tuple[str, str], ...]:
    if type(body) is not Body or body not in BODIES:
        raise AcquisitionError("unknown resolved-11 target")
    return (
        ("format", "json"),
        ("COMMAND", f"'{body.naif_id}'"),
        ("OBJ_DATA", "'NO'"),
        ("MAKE_EPHEM", "'YES'"),
        ("EPHEM_TYPE", "'VECTORS'"),
        ("CENTER", "'@0'"),
        ("TLIST", ",".join(f"'{epoch.jd_tdb}'" for epoch in EPOCHS)),
        ("TLIST_TYPE", "'JD'"),
        ("TIME_TYPE", "'TDB'"),
        ("REF_PLANE", "'FRAME'"),
        ("REF_SYSTEM", "'ICRF'"),
        ("OUT_UNITS", "'AU-D'"),
        ("VEC_TABLE", "'2'"),
        ("VEC_CORR", "'NONE'"),
        ("CSV_FORMAT", "'YES'"),
        ("VEC_LABELS", "'YES'"),
        ("TIME_DIGITS", "'FRACSEC'"),
    )


def build_query_url(body: Body) -> str:
    return HORIZONS_ENDPOINT + "?" + urlencode(
        _query_parameters(body), quote_via=quote
    )


def _verify_stage0() -> dict[str, object]:
    protocol_payload, protocol = _load_canonical(STAGE0_PROTOCOL_PATH, "Stage-0 PROTOCOL.json")
    states_payload, states = _load_canonical(STAGE0_STATES_PATH, "Stage-0 STATES.json")
    if _sha256(protocol_payload) != STAGE0_PROTOCOL_FILE_SHA256:
        raise AcquisitionError("bound Stage-0 protocol file differs")
    if _sha256(states_payload) != STAGE0_STATES_FILE_SHA256:
        raise AcquisitionError("bound Stage-0 states file differs")
    if (
        protocol.get("schema") != "jx.tno.native-epoch-cartesian.protocol.v1"
        or protocol.get("content_sha256") != STAGE0_PROTOCOL_CONTENT_SHA256
        or protocol.get("execution_authorized") is not False
        or protocol.get("outcomes_generated") is not False
    ):
        raise AcquisitionError("bound Stage-0 protocol semantics differ")
    if (
        states.get("schema") != "jx.tno.native-epoch-cartesian.states.v1"
        or states.get("content_sha256") != STAGE0_STATES_CONTENT_SHA256
        or states.get("binary_payload_sha256") != STAGE0_STATES_BINARY_PAYLOAD_SHA256
        or type(states.get("object_count")) is not int
        or states.get("object_count") != len(EPOCHS)
        or type(states.get("nominal_state_count")) is not int
        or states.get("nominal_state_count") != len(EPOCHS)
        or type(states.get("clone_state_count")) is not int
        or states.get("clone_state_count") != 1_536
        or type(states.get("total_state_count")) is not int
        or states.get("total_state_count") != 1_548
        or states.get("execution_authorized") is not False
        or states.get("outcomes_generated") is not False
    ):
        raise AcquisitionError("bound Stage-0 state semantics differ")
    rows = states.get("states")
    if type(rows) is not list or len(rows) != 1_548:
        raise AcquisitionError("bound Stage-0 state roster differs")
    for epoch in EPOCHS:
        object_rows = [
            row for row in rows
            if type(row) is dict and row.get("object_id") == epoch.object_id
        ]
        nominal_rows = [row for row in object_rows if row.get("sample_kind") == "NOMINAL"]
        if len(object_rows) != 129 or len(nominal_rows) != 1:
            raise AcquisitionError("bound Stage-0 per-object state count differs")
        for row in object_rows:
            if (
                type(row.get("spk_id")) is not int
                or row.get("spk_id") != epoch.spk_id
                or row.get("covariance_epoch_jd_tdb_hex") != epoch.jd_tdb_binary64_hex
            ):
                raise AcquisitionError("bound Stage-0 object/epoch identity differs")
        if float(epoch.jd_tdb).hex() != epoch.jd_tdb_binary64_hex:
            raise AcquisitionError("query epoch decimal/binary64 contract differs")
    if [row.get("object_id") for row in rows if row.get("sample_kind") == "NOMINAL"] != [
        epoch.object_id for epoch in EPOCHS
    ]:
        raise AcquisitionError("bound Stage-0 object order differs")
    return {
        "protocol_relative_path": (
            "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json"
        ),
        "protocol_file_sha256": STAGE0_PROTOCOL_FILE_SHA256,
        "protocol_content_sha256": STAGE0_PROTOCOL_CONTENT_SHA256,
        "states_relative_path": (
            "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json"
        ),
        "states_file_sha256": STAGE0_STATES_FILE_SHA256,
        "states_content_sha256": STAGE0_STATES_CONTENT_SHA256,
        "states_binary_payload_sha256": STAGE0_STATES_BINARY_PAYLOAD_SHA256,
    }


def build_protocol() -> dict[str, object]:
    source_binding = _verify_stage0()
    protocol: dict[str, object] = {
        "schema": SCHEMA_PROTOCOL,
        "classification": CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING,
        "status": "ACQUISITION_PROTOCOL_ONLY_NO_PROVIDER_BYTES_IN_THIS_RECORD",
        "nonclaim": (
            "Resolved-11 DE441 input acquisition only; no provider authentication, "
            "DE440/DE441 outcome comparison, dynamics, classification, inference, "
            "Planet Nine evidence, or science result."
        ),
        "stage0_input_binding": source_binding,
        "epoch_roster": [
            {
                "epoch_order": order,
                "object_id": epoch.object_id,
                "spk_id": epoch.spk_id,
                "jd_tdb": epoch.jd_tdb,
                "jd_tdb_binary64_hex": epoch.jd_tdb_binary64_hex,
            }
            for order, epoch in enumerate(EPOCHS)
        ],
        "body_roster": [
            {
                "body_order": order,
                "naif_id": body.naif_id,
                "semantic_id": body.semantic_id,
                "query_url": build_query_url(body),
            }
            for order, body in enumerate(BODIES)
        ],
        "query_contract": {
            "endpoint": HORIZONS_ENDPOINT,
            "request_count": len(BODIES),
            "epochs_per_request": len(EPOCHS),
            "expected_state_row_count": len(BODIES) * len(EPOCHS),
            "command_target_naif_ids_in_order": [body.naif_id for body in BODIES],
            "tlist_jd_tdb_tokens_in_order": [epoch.jd_tdb for epoch in EPOCHS],
            "center_parameter": "@0",
            "center_identity": "SOLAR_SYSTEM_BARYCENTER_NAIF_0",
            "ephemeris_source_header_required": "DE441",
            "ephemeris_type": "VECTORS",
            "time_type": "TDB",
            "tlist_type": "JD",
            "reference_plane": "FRAME",
            "reference_system": "ICRF",
            "output_units": "AU-D",
            "vector_table": 2,
            "vector_corrections": "NONE",
            "csv_format": True,
            "vector_labels": True,
            "state_semantics": "SSB_CENTERED_ICRF_GEOMETRIC_POSITION_VELOCITY",
            "response_epoch_policy": (
                "EXACT_DECIMAL_SET_IN_PROVIDER_ORDER_MAPPED_TO_STAGE0_ORDER"
            ),
            "custody_state_order": "EPOCH_MAJOR_THEN_RESOLVED11_BODY_ORDER",
        },
        "provider_evidence_limit": (
            "The retained JSON signature and DE441 text headers are schema evidence "
            "only; serialized bytes do not authenticate transport, origin, or custody."
        ),
        "references": [HORIZONS_DOCUMENTATION, HORIZONS_MANUAL],
        "implementation": {
            "relative_path": "tools/jx_tno_de441_resolved11_acquisition_v1.py",
            "file_sha256": _sha256(
                _read_regular_bytes(Path(__file__), "acquisition implementation")
            ),
        },
        "claims": dict(PROTOCOL_CLAIMS),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    protocol["content_sha256"] = _content_sha256(DOMAIN_PROTOCOL, protocol)
    return protocol


def _parse_outer_response(payload: bytes) -> str:
    if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_RESPONSE_BYTES:
        raise AcquisitionError("Horizons response has an invalid bounded byte size")
    value = _strict_json(payload, "Horizons response")
    if set(value) != {"signature", "result"}:
        raise AcquisitionError("Horizons outer JSON shape differs")
    if value.get("signature") != EXPECTED_API_SIGNATURE:
        raise AcquisitionError("Horizons API signature differs")
    result = value.get("result")
    if type(result) is not str or not result or "\x00" in result:
        raise AcquisitionError("Horizons result text is invalid")
    return result


def validate_response(payload: bytes, body: Body) -> dict[str, object]:
    if type(body) is not Body or body not in BODIES:
        raise AcquisitionError("unknown resolved-11 target")
    result = _parse_outer_response(payload)
    target_matches = list(_TARGET_LINE.finditer(result))
    center_matches = list(_CENTER_LINE.finditer(result))
    if len(target_matches) != 1 or len(center_matches) != 1:
        raise AcquisitionError("Horizons DE441 target/center header differs")
    target_match = target_matches[0]
    if int(target_match.group(2)) != body.naif_id:
        raise AcquisitionError("Horizons resolved target identity differs")
    target_name = target_match.group(1).strip()
    if not target_name or any(character in target_name for character in "{}\r\n"):
        raise AcquisitionError("Horizons target display name is invalid")
    required_markers = (
        "Step-size       : DISCRETE TIME-LIST",
        "Output units    : AU-D",
        "Output type     : GEOMETRIC cartesian states",
        "Output format   : 2 (position and velocity)",
        "Reference frame : ICRF",
        "Geometric state vectors have NO corrections or aberrations applied.",
        'Barycentric Dynamical Time ("TDB" or T_eph) output was requested.',
    )
    if any(result.count(marker) != 1 for marker in required_markers):
        raise AcquisitionError("Horizons unit/frame/time/correction markers differ")
    lines = result.splitlines()
    start_indexes = [index for index, line in enumerate(lines) if line.strip() == "$$SOE"]
    stop_indexes = [index for index, line in enumerate(lines) if line.strip() == "$$EOE"]
    if (
        len(start_indexes) != 1
        or len(stop_indexes) != 1
        or start_indexes[0] >= stop_indexes[0]
    ):
        raise AcquisitionError("Horizons vector delimiters differ")
    expected_header = (
        "JDTDB",
        "Calendar Date (TDB)",
        "X",
        "Y",
        "Z",
        "VX",
        "VY",
        "VZ",
        "",
    )
    header_count = 0
    for line in lines:
        try:
            fields = tuple(field.strip() for field in next(csv.reader([line], strict=True)))
        except csv.Error:
            continue
        if fields == expected_header:
            header_count += 1
    if header_count != 1:
        raise AcquisitionError("Horizons labeled vector header differs")
    block = "\n".join(lines[start_indexes[0] + 1 : stop_indexes[0]])
    rows: list[dict[str, object]] = []
    epochs_by_decimal = {
        Decimal(epoch.jd_tdb): (epoch_order, epoch)
        for epoch_order, epoch in enumerate(EPOCHS)
    }
    if len(epochs_by_decimal) != len(EPOCHS):
        raise AcquisitionError("requested epoch roster contains a duplicate")
    observed_epochs: set[Decimal] = set()
    try:
        reader = csv.reader(io.StringIO(block), strict=True)
        for raw_fields in reader:
            fields = [field.strip() for field in raw_fields]
            if not fields or not any(fields):
                continue
            if len(fields) != 9 or fields[-1] != "":
                raise AcquisitionError("Horizons vector row shape differs")
            provider_row_index = len(rows)
            if provider_row_index >= len(EPOCHS):
                raise AcquisitionError("Horizons returned extra state rows")
            epoch_token, calendar_tdb, *state_tokens = fields[:-1]
            epoch_decimal = _decimal(epoch_token, "JDTDB")
            if epoch_decimal not in epochs_by_decimal:
                raise AcquisitionError("Horizons returned an unexpected epoch token")
            if epoch_decimal in observed_epochs:
                raise AcquisitionError("Horizons returned a duplicate epoch token")
            observed_epochs.add(epoch_decimal)
            epoch_order, expected_epoch = epochs_by_decimal[epoch_decimal]
            if not calendar_tdb.startswith(("A.D. ", "B.C. ")):
                raise AcquisitionError("Horizons calendar TDB token differs")
            position = state_tokens[:3]
            velocity = state_tokens[3:]
            position_hex = [
                _float_hex(token, f"position component {index}")
                for index, token in enumerate(position)
            ]
            velocity_hex = [
                _float_hex(token, f"velocity component {index}")
                for index, token in enumerate(velocity)
            ]
            rows.append(
                {
                    "epoch_order": epoch_order,
                    "provider_row_index": provider_row_index,
                    "object_id": expected_epoch.object_id,
                    "spk_id": expected_epoch.spk_id,
                    "requested_jd_tdb": expected_epoch.jd_tdb,
                    "returned_jd_tdb_decimal_token": epoch_token,
                    "calendar_tdb": calendar_tdb,
                    "position_AU_decimal_tokens": position,
                    "velocity_AU_per_day_decimal_tokens": velocity,
                    "position_AU_binary64_hex": position_hex,
                    "velocity_AU_per_day_binary64_hex": velocity_hex,
                }
            )
    except csv.Error as exc:
        raise AcquisitionError("Horizons vector block is malformed CSV") from exc
    if len(rows) != len(EPOCHS) or observed_epochs != set(epochs_by_decimal):
        raise AcquisitionError("Horizons epoch roster is incomplete")
    return {
        "target_name": target_name,
        "target_naif_id": body.naif_id,
        "target_semantic_id": body.semantic_id,
        "target_ephemeris_source": target_match.group(3),
        "center_naif_id": 0,
        "center_ephemeris_source": center_matches[0].group(1),
        "frame": "ICRF",
        "center": "SOLAR_SYSTEM_BARYCENTER_NAIF_0",
        "units": "AU_AND_AU_PER_DAY",
        "corrections": "GEOMETRIC_NONE",
        "rows": rows,
    }


def build_custody_from_payloads(
    payloads: Mapping[int, bytes], protocol: Mapping[str, Any]
) -> dict[str, object]:
    if type(payloads) is not dict or set(payloads) != {body.naif_id for body in BODIES}:
        raise AcquisitionError("raw response mapping roster differs")
    expected_protocol = build_protocol()
    if type(protocol) is not dict or _canonical_json(protocol) != _canonical_json(expected_protocol):
        raise AcquisitionError("protocol differs from exact reconstruction")
    parsed_by_body: dict[int, dict[str, object]] = {}
    requests: list[dict[str, object]] = []
    for body_order, body in enumerate(BODIES):
        payload = payloads[body.naif_id]
        if type(payload) is not bytes:
            raise AcquisitionError("raw response mapping values must be bytes")
        parsed = validate_response(payload, body)
        parsed_by_body[body.naif_id] = parsed
        requests.append(
            {
                "body_order": body_order,
                "naif_id": body.naif_id,
                "semantic_id": body.semantic_id,
                "target_name": parsed["target_name"],
                "query_url": build_query_url(body),
                "retained_relative_path": f"raw/{body.filename}",
                "raw_byte_count": len(payload),
                "raw_sha256": _sha256(payload),
                "source_row_count": len(EPOCHS),
                "provider_row_epoch_orders": [
                    row["epoch_order"] for row in parsed["rows"]
                ],
                "state_order_indexes": [
                    epoch_order * len(BODIES) + body_order
                    for epoch_order in range(len(EPOCHS))
                ],
            }
        )
    states: list[dict[str, object]] = []
    for epoch_order, epoch in enumerate(EPOCHS):
        for body_order, body in enumerate(BODIES):
            parsed = parsed_by_body[body.naif_id]
            parsed_rows = parsed["rows"]
            if type(parsed_rows) is not list or len(parsed_rows) != len(EPOCHS):
                raise AcquisitionError("parsed response row roster differs")
            rows_by_epoch_order = {
                row.get("epoch_order"): row
                for row in parsed_rows
                if type(row) is dict
                and type(row.get("epoch_order")) is int
            }
            if set(rows_by_epoch_order) != set(range(len(EPOCHS))):
                raise AcquisitionError("parsed response epoch mapping differs")
            source_row = rows_by_epoch_order[epoch_order]
            if type(source_row) is not dict:
                raise AcquisitionError("parsed response row type differs")
            states.append(
                {
                    "state_order": epoch_order * len(BODIES) + body_order,
                    "epoch_order": epoch_order,
                    "body_order": body_order,
                    "object_id": epoch.object_id,
                    "spk_id": epoch.spk_id,
                    "requested_jd_tdb": epoch.jd_tdb,
                    "returned_jd_tdb_decimal_token": source_row[
                        "returned_jd_tdb_decimal_token"
                    ],
                    "calendar_tdb": source_row["calendar_tdb"],
                    "naif_id": body.naif_id,
                    "semantic_id": body.semantic_id,
                    "target_name": parsed["target_name"],
                    "center_naif_id": 0,
                    "ephemeris_source": "DE441",
                    "frame": "ICRF",
                    "corrections": "GEOMETRIC_NONE",
                    "position_AU_decimal_tokens": source_row[
                        "position_AU_decimal_tokens"
                    ],
                    "velocity_AU_per_day_decimal_tokens": source_row[
                        "velocity_AU_per_day_decimal_tokens"
                    ],
                    "position_AU_binary64_hex": source_row[
                        "position_AU_binary64_hex"
                    ],
                    "velocity_AU_per_day_binary64_hex": source_row[
                        "velocity_AU_per_day_binary64_hex"
                    ],
                    "raw_response_relative_path": f"raw/{body.filename}",
                    "raw_response_sha256": requests[body_order]["raw_sha256"],
                    "raw_response_row_index": source_row["provider_row_index"],
                }
            )
    if len(states) != 132:
        raise AcquisitionError("resolved-11 state support count differs")
    custody: dict[str, object] = {
        "schema": SCHEMA_CUSTODY,
        "classification": CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING,
        "status": "RETAINED_SCHEMA_CONFORMANT_UNAUTHENTICATED_DE441_INPUT_BYTES",
        "protocol_content_sha256": expected_protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(expected_protocol)),
        "request_count": len(requests),
        "epoch_count": len(EPOCHS),
        "body_count": len(BODIES),
        "state_record_count": len(states),
        "state_record_order": "EPOCH_MAJOR_THEN_RESOLVED11_BODY_ORDER",
        "requests": requests,
        "raw_roster_sha256": _sha256(
            DOMAIN_RAW_ROSTER + _canonical_json(requests)
        ),
        "state_records": states,
        "state_roster_sha256": _sha256(
            DOMAIN_STATE_ROSTER + _canonical_json(states)
        ),
        "provider_evidence": {
            "api_signature_schema_matched": True,
            "target_and_center_de441_text_headers_matched": True,
            "transport_authenticated_in_retained_bundle": False,
            "provider_origin_cryptographically_proven": False,
            "continuous_custody_proven": False,
        },
        "claims": dict(CUSTODY_CLAIMS),
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    custody["content_sha256"] = _content_sha256(DOMAIN_CUSTODY, custody)
    return custody


def _output_roster(root: Path) -> None:
    _require_real_directory(root, "acquisition root")
    observed_root = {entry.name for entry in root.iterdir()}
    if observed_root != {"PROTOCOL.json", "CUSTODY.json", "raw"}:
        raise AcquisitionError("acquisition root file roster differs")
    raw = root / "raw"
    _require_real_directory(raw, "raw response directory")
    observed_raw = {entry.name for entry in raw.iterdir()}
    expected_raw = {body.filename for body in BODIES}
    if observed_raw != expected_raw:
        raise AcquisitionError("raw response file roster differs")


def validate_output(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(root)))
    _output_roster(root)
    protocol_payload, protocol = _load_canonical(root / "PROTOCOL.json", "PROTOCOL.json")
    expected_protocol = build_protocol()
    if protocol_payload != _canonical_json(expected_protocol):
        raise AcquisitionError("retained protocol differs from exact reconstruction")
    payloads = {
        body.naif_id: _read_regular_bytes(
            root / "raw" / body.filename, f"raw/{body.filename}"
        )
        for body in BODIES
    }
    expected_custody = build_custody_from_payloads(payloads, expected_protocol)
    custody_payload, _custody = _load_canonical(root / "CUSTODY.json", "CUSTODY.json")
    if custody_payload != _canonical_json(expected_custody):
        raise AcquisitionError("retained custody differs from exact reconstruction")
    return expected_custody


def _fetch(url: str) -> bytes:
    if type(url) is not str or not url.startswith(HORIZONS_ENDPOINT + "?"):
        raise AcquisitionError("invalid Horizons query URL")
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "JX-TNO-DE441-resolved11-native-epochs-v1",
        },
    )
    try:
        with urlopen(request, timeout=90) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise AcquisitionError(f"Horizons returned HTTP status {status}")
            payload = response.read(MAXIMUM_RESPONSE_BYTES + 1)
    except AcquisitionError:
        raise
    except OSError as exc:
        raise AcquisitionError("Horizons request failed") from exc
    if not payload or len(payload) > MAXIMUM_RESPONSE_BYTES:
        raise AcquisitionError("Horizons response has an invalid bounded byte size")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AcquisitionError("Horizons response is not strict UTF-8") from exc
    return payload


def acquire(
    root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    fetcher: Callable[[str], bytes] = _fetch,
) -> dict[str, Any]:
    root = _require_output_absent(root)
    if not callable(fetcher):
        raise AcquisitionError("fetcher must be callable")
    protocol = build_protocol()
    payloads: dict[int, bytes] = {}
    for body in BODIES:
        payload = fetcher(build_query_url(body))
        if type(payload) is not bytes:
            raise AcquisitionError("fetcher must return bytes")
        validate_response(payload, body)
        payloads[body.naif_id] = payload
    custody = build_custody_from_payloads(payloads, protocol)
    root = _require_output_absent(root)
    created = False
    try:
        root.mkdir(mode=0o700)
        created = True
        raw = root / "raw"
        raw.mkdir(mode=0o700)
        _write_create_only(root / "PROTOCOL.json", _canonical_json(protocol))
        for body in BODIES:
            _write_create_only(raw / body.filename, payloads[body.naif_id])
        _fsync_directory(raw)
        _write_create_only(root / "CUSTODY.json", _canonical_json(custody))
        _fsync_directory(root)
        return validate_output(root)
    except Exception:
        if created:
            shutil.rmtree(root, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol", "acquire", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "protocol":
            value = build_protocol()
        elif arguments.command == "acquire":
            value = acquire(arguments.output_root)
        else:
            value = validate_output(arguments.output_root)
    except AcquisitionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
