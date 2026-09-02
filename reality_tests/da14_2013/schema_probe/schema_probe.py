#!/usr/bin/env python3
"""JX Reality Test 1: value-redacted live schema diagnostic.

This program uses the already-audited restricted adapter transport and only
emits structural metadata. It never writes the raw HTTP response or any source
measurement value. It is intended to diagnose live API-schema drift without
unblinding the historical holdout.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import http.server
import json
import math
import os
import re
import socket
import sys
import tempfile
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from jx_source_adapter.errors import SourceSchemaError, TransportError, PolicyError
from jx_source_adapter.http_client import fetch_json
from jx_source_adapter.normalize import (
    normalize_jpl_radar,
    normalize_mpc_observations,
    normalize_mpc_observatories,
)
from jx_source_adapter.policy import build_requests
from jx_source_adapter.util import canonical_json_bytes, harden_process, sha256_bytes

SCHEMA = "jx-reality-test-1-live-schema-probe/v1"
AUDIT_SCHEMA = "jx-reality-test-1-schema-probe-audit/v1"
APPROVED_SOURCE_IDS = (
    "mpc_observations",
    "mpc_observatory_codes",
    "jpl_radar",
)
_SAFE_JPL_FIELD_NAMES = {
    "des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit",
    "bp", "observer", "notes", "ref", "fullname", "modified",
}
_DATETIME_TZ_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:\d{2})$"
)
_DATETIME_NO_TZ_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CODE_RE = re.compile(r"^[A-Za-z0-9_.+\-/]{1,40}$")
_URL_RE = re.compile(r"^https?://", re.I)
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_NONFINITE_RE = re.compile(r"^[+-]?(?:nan|inf(?:inity)?)$", re.I)
_MPC_CODE_RE = re.compile(r"^[A-Z0-9][0-9]{2}$")
_RADAR_CODE_RE = re.compile(r"^-\d{1,4}$")


class ProbeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unsupported"


def string_pattern(value: str) -> str:
    if value == "":
        return "empty"
    text = value.strip()
    if text == "":
        return "whitespace_only"
    if _DATETIME_TZ_RE.fullmatch(text):
        return "datetime_explicit_timezone"
    if _DATETIME_NO_TZ_RE.fullmatch(text):
        return "datetime_no_timezone"
    if _DATE_RE.fullmatch(text):
        return "date"
    if _NONFINITE_RE.fullmatch(text):
        return "nonfinite_numeric_token"
    if _NUMERIC_RE.fullmatch(text):
        try:
            number = Decimal(text)
        except InvalidOperation:
            return "numeric_like_invalid"
        return "finite_numeric_string" if number.is_finite() else "nonfinite_numeric_token"
    if _URL_RE.match(text):
        return "url"
    if _CODE_RE.fullmatch(text):
        return "code_token"
    return "free_text"


def safe_exception_message(error: BaseException) -> str:
    """Retain only static-looking words; erase quoted/numeric/value fragments."""
    message = str(error)
    message = re.sub(r"(['\"]).*?\1", "<redacted>", message)
    message = re.sub(r"\b[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?\b", "<n>", message)
    message = re.sub(r"[^A-Za-z0-9_ .:/<>-]", "?", message)
    message = re.sub(r"\s+", " ", message).strip()
    return message[:240] or error.__class__.__name__


@dataclass
class FieldStats:
    present: int = 0
    null: int = 0
    type_counts: collections.Counter[str] | None = None
    string_patterns: collections.Counter[str] | None = None
    string_min_bytes: int | None = None
    string_max_bytes: int | None = None
    array_min_length: int | None = None
    array_max_length: int | None = None
    object_key_union: set[str] | None = None

    def __post_init__(self) -> None:
        self.type_counts = collections.Counter()
        self.string_patterns = collections.Counter()
        self.object_key_union = set()

    def add(self, value: Any) -> None:
        self.present += 1
        kind = json_type(value)
        self.type_counts[kind] += 1
        if value is None:
            self.null += 1
        elif isinstance(value, str):
            self.string_patterns[string_pattern(value)] += 1
            length = len(value.encode("utf-8"))
            self.string_min_bytes = length if self.string_min_bytes is None else min(self.string_min_bytes, length)
            self.string_max_bytes = length if self.string_max_bytes is None else max(self.string_max_bytes, length)
        elif isinstance(value, list):
            length = len(value)
            self.array_min_length = length if self.array_min_length is None else min(self.array_min_length, length)
            self.array_max_length = length if self.array_max_length is None else max(self.array_max_length, length)
        elif isinstance(value, dict):
            self.object_key_union.update(map(str, value.keys()))

    def export(self, parent_count: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "present_count": self.present,
            "missing_count": max(0, parent_count - self.present),
            "null_count": self.null,
            "json_type_counts": dict(sorted(self.type_counts.items())),
        }
        if self.string_patterns:
            result["string_pattern_counts"] = dict(sorted(self.string_patterns.items()))
            result["string_byte_length_range"] = [self.string_min_bytes, self.string_max_bytes]
        if self.array_min_length is not None:
            result["array_length_range"] = [self.array_min_length, self.array_max_length]
        if self.object_key_union:
            result["object_key_union"] = sorted(self.object_key_union)
        return result


def record_field_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, FieldStats] = {}
    for record in records:
        for key, value in record.items():
            fields.setdefault(str(key), FieldStats()).add(value)
    return {
        "record_count": len(records),
        "field_union": sorted(fields),
        "field_intersection": sorted(
            key for key, stats in fields.items() if stats.present == len(records)
        ),
        "fields": {
            key: fields[key].export(len(records)) for key in sorted(fields)
        },
    }


def safe_dynamic_key_summary(keys: Iterable[str]) -> dict[str, Any]:
    counts = collections.Counter()
    lengths: list[int] = []
    for key in keys:
        text = str(key)
        lengths.append(len(text.encode("utf-8")))
        if _MPC_CODE_RE.fullmatch(text):
            counts["mpc_observatory_code"] += 1
        elif _RADAR_CODE_RE.fullmatch(text):
            counts["radar_station_code"] += 1
        elif _CODE_RE.fullmatch(text):
            counts["generic_code_token"] += 1
        else:
            counts["other_dynamic_key"] += 1
    return {
        "count": len(lengths),
        "key_pattern_counts": dict(sorted(counts.items())),
        "key_byte_length_range": [min(lengths), max(lengths)] if lengths else [None, None],
    }


def summarize_mpc_observations(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"root_type": json_type(payload)}
    if not isinstance(payload, list):
        return result
    result["envelope_count"] = len(payload)
    envelope_types = collections.Counter(json_type(item) for item in payload)
    result["envelope_type_counts"] = dict(sorted(envelope_types.items()))
    envelopes = [item for item in payload if isinstance(item, dict)]
    result["envelope_key_union"] = sorted({str(key) for item in envelopes for key in item})
    result["envelope_key_intersection"] = sorted(
        set.intersection(*(set(map(str, item.keys())) for item in envelopes))
    ) if envelopes else []
    if len(envelopes) == 1:
        rows = envelopes[0].get("ADES_DF")
        result["ades_df_type"] = json_type(rows)
        if isinstance(rows, list):
            result["ades_df_length"] = len(rows)
            row_type_counts = collections.Counter(json_type(row) for row in rows)
            result["row_type_counts"] = dict(sorted(row_type_counts.items()))
            dict_rows = [row for row in rows if isinstance(row, dict)]
            result["record_schema"] = record_field_stats(dict_rows)
            structure_counts = collections.Counter()
            for row in dict_rows:
                has_ra = "ra" in row and row.get("ra") not in (None, "")
                has_dec = "dec" in row and row.get("dec") not in (None, "")
                has_delay = "delay" in row and row.get("delay") not in (None, "")
                has_doppler = "doppler" in row and row.get("doppler") not in (None, "")
                if (has_ra or has_dec) and not (has_delay or has_doppler):
                    structure_counts["optical_key_signature"] += 1
                elif (has_delay or has_doppler) and not (has_ra or has_dec):
                    structure_counts["radar_key_signature"] += 1
                elif (has_ra or has_dec) and (has_delay or has_doppler):
                    structure_counts["ambiguous_mixed_signature"] += 1
                else:
                    structure_counts["unsupported_no_measurement_signature"] += 1
            result["observation_key_signature_counts"] = dict(sorted(structure_counts.items()))
    return result


def summarize_mpc_observatories(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"root_type": json_type(payload)}
    if not isinstance(payload, dict):
        return result
    if "obscode" in payload:
        result["response_mode"] = "single_record"
        result["record_schema"] = record_field_stats([payload])
    else:
        result["response_mode"] = "dynamic_record_map"
        result["dynamic_keys"] = safe_dynamic_key_summary(map(str, payload.keys()))
        records = [value for value in payload.values() if isinstance(value, dict)]
        result["value_type_counts"] = dict(sorted(collections.Counter(json_type(value) for value in payload.values()).items()))
        result["record_schema"] = record_field_stats(records)
    return result


def summarize_jpl_radar(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"root_type": json_type(payload)}
    if not isinstance(payload, dict):
        return result
    result["top_level_keys"] = sorted(map(str, payload.keys()))
    signature = payload.get("signature")
    result["signature_type"] = json_type(signature)
    if isinstance(signature, dict):
        result["signature_keys"] = sorted(map(str, signature.keys()))
        version = signature.get("version")
        if isinstance(version, (str, int, float)):
            result["signature_version"] = str(version)[:32]
        source = signature.get("source")
        if isinstance(source, str):
            result["signature_source_sha256"] = sha256_bytes(source.encode("utf-8"))
            result["signature_source_byte_length"] = len(source.encode("utf-8"))
    fields = payload.get("fields")
    result["fields_type"] = json_type(fields)
    if isinstance(fields, list):
        safe_fields: list[str] = []
        unsafe_count = 0
        for item in fields:
            if isinstance(item, str) and item in _SAFE_JPL_FIELD_NAMES:
                safe_fields.append(item)
            else:
                unsafe_count += 1
        result["declared_field_count"] = len(fields)
        result["declared_safe_field_names"] = safe_fields
        result["undeclared_or_nonstring_field_name_count"] = unsafe_count
    data = payload.get("data")
    result["data_type"] = json_type(data)
    if isinstance(data, list):
        result["data_row_count"] = len(data)
        widths = [len(row) for row in data if isinstance(row, list)]
        result["data_row_type_counts"] = dict(sorted(collections.Counter(json_type(row) for row in data).items()))
        result["data_row_width_range"] = [min(widths), max(widths)] if widths else [None, None]
        if isinstance(fields, list) and all(isinstance(item, str) for item in fields):
            column_stats: dict[str, FieldStats] = {str(name): FieldStats() for name in fields}
            valid_width_rows = 0
            for row in data:
                if isinstance(row, list) and len(row) == len(fields):
                    valid_width_rows += 1
                    for name, value in zip(fields, row):
                        column_stats[str(name)].add(value)
            result["valid_width_row_count"] = valid_width_rows
            result["column_shapes"] = {
                name: stats.export(valid_width_rows) for name, stats in sorted(column_stats.items())
            }
    coords = payload.get("coords")
    result["coords_type"] = json_type(coords)
    if isinstance(coords, dict):
        result["coords_dynamic_keys"] = safe_dynamic_key_summary(map(str, coords.keys()))
        records = [value for value in coords.values() if isinstance(value, dict)]
        result["coords_record_schema"] = record_field_stats(records)
    count = payload.get("count")
    result["declared_count_type"] = json_type(count)
    if isinstance(count, str):
        result["declared_count_string_pattern"] = string_pattern(count)
        result["declared_count_string_length"] = len(count.encode("utf-8"))
    return result


def structural_summary(source_id: str, payload: Any) -> dict[str, Any]:
    if source_id == "mpc_observations":
        return summarize_mpc_observations(payload)
    if source_id == "mpc_observatory_codes":
        return summarize_mpc_observatories(payload)
    if source_id == "jpl_radar":
        return summarize_jpl_radar(payload)
    raise ProbeError("unsupported source id")


def run_compatibility(payloads: dict[str, Any], designation: str, maximum_records: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    mpc_records: list[dict[str, Any]] = []
    try:
        mpc_records, metadata = normalize_mpc_observations(
            payloads["mpc_observations"], designation=designation, maximum_records=maximum_records
        )
        result["mpc_observations"] = {
            "status": "COMPATIBLE",
            "normalized_record_count": len(mpc_records),
            "record_counts_by_type": metadata.get("record_counts_by_type", {}),
            "response_keys": metadata.get("response_keys", []),
        }
    except Exception as error:
        result["mpc_observations"] = {
            "status": "INCOMPATIBLE",
            "error_class": error.__class__.__name__,
            "safe_error": safe_exception_message(error),
        }

    try:
        radar_records, metadata, radar_stations = normalize_jpl_radar(
            payloads["jpl_radar"], designation=designation, maximum_records=maximum_records
        )
        result["jpl_radar"] = {
            "status": "COMPATIBLE",
            "normalized_record_count": len(radar_records),
            "api_signature_version": metadata.get("api_signature_version"),
            "declared_fields": metadata.get("fields", []),
            "radar_station_count": len(radar_stations),
        }
    except Exception as error:
        result["jpl_radar"] = {
            "status": "INCOMPATIBLE",
            "error_class": error.__class__.__name__,
            "safe_error": safe_exception_message(error),
        }

    try:
        required_codes = {
            str(record["station_code"])
            for record in mpc_records
            if record.get("measurement_type") == "optical"
        }
        stations = normalize_mpc_observatories(
            payloads["mpc_observatory_codes"], required_codes=required_codes
        )
        result["mpc_observatory_codes"] = {
            "status": "COMPATIBLE",
            "required_station_count": len(required_codes),
            "selected_station_count": len(stations),
            "dependency_note": "required station set derived in memory from MPC normalization",
        }
    except Exception as error:
        result["mpc_observatory_codes"] = {
            "status": "INCOMPATIBLE",
            "error_class": error.__class__.__name__,
            "safe_error": safe_exception_message(error),
            "dependency_note": "may be downstream of an MPC observation normalization failure",
        }
    mpc_records.clear()
    return result


def probe_sources(
    *,
    designation: str,
    output_path: Path,
    limits: dict[str, Any],
    audit_mode: bool = False,
    audit_base_url: str | None = None,
) -> dict[str, Any]:
    harden_process()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requests = build_requests(
        designation=designation, audit_mode=audit_mode, audit_base_url=audit_base_url
    )
    payloads: dict[str, Any] = {}
    source_reports: dict[str, Any] = {}
    for spec in requests:
        try:
            fetched = fetch_json(
                spec,
                audit_mode=audit_mode,
                timeout_seconds=float(limits["request_timeout_seconds"]),
                maximum_response_bytes=int(limits["maximum_response_bytes_each"]),
                maximum_json_depth=int(limits["maximum_json_depth"]),
                maximum_string_bytes=int(limits["maximum_string_bytes"]),
            )
            payloads[spec.source_id] = fetched.payload
            source_reports[spec.source_id] = {
                "fetch_status": "FETCHED_IN_MEMORY_NO_RAW_RETENTION",
                "receipt": fetched.receipt,
                "structure": structural_summary(spec.source_id, fetched.payload),
            }
        except (TransportError, SourceSchemaError, PolicyError) as error:
            source_reports[spec.source_id] = {
                "fetch_status": "FETCH_BLOCKED",
                "error_class": error.__class__.__name__,
                "safe_error": safe_exception_message(error),
            }
        except Exception as error:
            source_reports[spec.source_id] = {
                "fetch_status": "FETCH_BLOCKED_UNEXPECTED",
                "error_class": error.__class__.__name__,
                "safe_error": safe_exception_message(error),
            }

    all_fetched = all(source_id in payloads for source_id in APPROVED_SOURCE_IDS)
    compatibility = (
        run_compatibility(
            payloads, designation=designation, maximum_records=int(limits["maximum_records_total"])
        )
        if all_fetched
        else {"status": "NOT_RUN_ALL_SOURCES_NOT_FETCHED"}
    )
    result = {
        "schema": SCHEMA,
        "status": "LIVE_SCHEMA_PROBE_COMPLETE" if all_fetched else "LIVE_SCHEMA_PROBE_PARTIAL",
        "audit_mode": audit_mode,
        "target_identity_only": designation,
        "measurement_values_emitted": False,
        "raw_http_response_retention": "NONE",
        "complete_normalized_plaintext_retention": "NEVER_WRITTEN",
        "private_key_required": False,
        "holdout_unblinded": False,
        "orbit_fitting_performed": False,
        "approved_source_ids": list(APPROVED_SOURCE_IDS),
        "sources": source_reports,
        "current_normalizer_compatibility": compatibility,
        "claim_ceiling": (
            "Value-redacted live API-structure diagnostic only; no observation dataset, "
            "orbit fit, prediction, unblinding, or physical-validation result."
        ),
    }
    atomic_json(output_path, result)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8")
    os.chmod(checksum_path, 0o444)
    payloads.clear()
    return result


class AuditHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    responses: dict[str, bytes] = {}
    modes: dict[str, str] = {}

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)
        path = self.path.split("?", 1)[0]
        mode = self.modes.get(path, "normal")
        if mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/forbidden")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "wrong_content_type":
            body = self.responses[path]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if mode == "malformed":
            body = b"{not-json"
        else:
            body = self.responses[path]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def audit_server(responses: dict[str, Any], modes: dict[str, str] | None = None):
    encoded = {path: canonical_json_bytes(value) for path, value in responses.items()}
    handler = type("BoundAuditHandler", (AuditHandler,), {"responses": encoded, "modes": modes or {}})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def synthetic_payloads() -> tuple[dict[str, Any], list[str]]:
    canaries = [
        "123.456789123456789",
        "-12.345678912345678",
        "ZXQ-HOLDOUT-OBSERVATORY-CANARY",
        "987654321.123456789",
        "0.000000123456789",
    ]
    mpc = [{
        "ADES_DF": [
            {
                "permID": "SYNTHETIC-001",
                "obsTime": "2012-12-31T23:59:59Z",
                "stn": "500",
                "ra": canaries[0],
                "dec": canaries[1],
                "rmsRA": "0.1",
                "rmsDec": "0.2",
            },
            {
                "permID": "SYNTHETIC-001",
                "obsTime": "2013-02-15T00:00:00Z",
                "rcv": "-14",
                "trx": "-14",
                "delay": "0.0012345",
                "rmsDelay": "1.0",
                "frq": "8560",
                "com": "1",
            },
        ]
    }]
    obscodes = {
        "500": {
            "obscode": "500",
            "longitude": "0",
            "rhocosphi": "1",
            "rhosinphi": "0",
            "name": canaries[2],
            "uses_two_line_observations": False,
        }
    }
    radar = {
        "signature": {
            "source": "NASA/JPL Small-Body Radar Astrometry API",
            "version": "1.1",
        },
        "count": "1",
        "fields": ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp"],
        "data": [[
            "SYNTHETIC-001", "2013-02-16 00:00:00", canaries[3], canaries[4],
            "us", "8560", "-14", "-14", "C"
        ]],
        "coords": {
            "-14": {
                "longitude": "243.1105",
                "latitude": "35.4259",
                "altitude": "1006.94",
                "alt_units": "m",
                "name": "DSS 14",
            }
        },
    }
    return {
        "/api/get-obs": mpc,
        "/api/obscodes": obscodes,
        "/sb_radar.api": radar,
    }, canaries


def run_self_audit(output_dir: Path, limits: dict[str, Any]) -> dict[str, Any]:
    harden_process()
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads, canaries = synthetic_payloads()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool) -> None:
        checks.append({"check": name, "passed": bool(condition)})

    with audit_server(payloads) as base_url:
        first_path = output_dir / "synthetic_schema_probe_a.json"
        second_path = output_dir / "synthetic_schema_probe_b.json"
        first = probe_sources(
            designation="SYNTHETIC-001",
            output_path=first_path,
            limits=limits,
            audit_mode=True,
            audit_base_url=base_url,
        )
        second = probe_sources(
            designation="SYNTHETIC-001",
            output_path=second_path,
            limits=limits,
            audit_mode=True,
            audit_base_url=base_url,
        )

    output_text = first_path.read_text(encoding="utf-8") + second_path.read_text(encoding="utf-8")
    check("synthetic_probe_completed", first["status"] == "LIVE_SCHEMA_PROBE_COMPLETE")
    check("all_sources_fetched", all(first["sources"][sid]["fetch_status"].startswith("FETCHED") for sid in APPROVED_SOURCE_IDS))
    compatibility = first["current_normalizer_compatibility"]
    check("current_normalizers_compatible_with_valid_mock", all(compatibility[sid]["status"] == "COMPATIBLE" for sid in APPROVED_SOURCE_IDS))
    check("no_measurement_canary_emitted", not any(canary in output_text for canary in canaries))
    check("no_raw_response_file_created", not any("raw" in path.name.lower() for path in output_dir.iterdir()))
    check("measurement_values_flag_false", first["measurement_values_emitted"] is False)
    check("private_key_not_required", first["private_key_required"] is False)
    check("holdout_remains_blinded", first["holdout_unblinded"] is False)
    check("orbit_fit_not_performed", first["orbit_fitting_performed"] is False)
    mpc_structure = first["sources"]["mpc_observations"]["structure"]
    check("mpc_field_names_exposed_without_values", "ra" in mpc_structure["record_schema"]["field_union"] and "dec" in mpc_structure["record_schema"]["field_union"])
    radar_structure = first["sources"]["jpl_radar"]["structure"]
    check("jpl_schema_field_names_recorded", radar_structure.get("declared_safe_field_names") == ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp"])
    obscode_structure = first["sources"]["mpc_observatory_codes"]["structure"]
    check("dynamic_observatory_keys_generalized", obscode_structure.get("dynamic_keys", {}).get("count") == 1 and "500" not in json.dumps(obscode_structure))
    check("response_hashes_recorded", all(len(first["sources"][sid]["receipt"]["response_sha256"]) == 64 for sid in APPROVED_SOURCE_IDS))
    check("request_hashes_recorded", all(len(first["sources"][sid]["receipt"]["request_body_sha256"]) == 64 for sid in APPROVED_SOURCE_IDS))
    check("structural_output_deterministic", first == second)
    check("no_private_key_marker", "PRIVATE KEY" not in output_text)

    bad_checks: list[tuple[str, dict[str, str]]] = [
        ("redirect_blocked", {"/api/get-obs": "redirect"}),
        ("wrong_content_type_blocked", {"/api/get-obs": "wrong_content_type"}),
        ("malformed_json_blocked", {"/api/get-obs": "malformed"}),
    ]
    for name, modes in bad_checks:
        with audit_server(payloads, modes=modes) as base_url:
            request = build_requests(
                designation="SYNTHETIC-002", audit_mode=True, audit_base_url=base_url
            )[0]
            blocked = False
            try:
                fetch_json(
                    request,
                    audit_mode=True,
                    timeout_seconds=float(limits["request_timeout_seconds"]),
                    maximum_response_bytes=int(limits["maximum_response_bytes_each"]),
                    maximum_json_depth=int(limits["maximum_json_depth"]),
                    maximum_string_bytes=int(limits["maximum_string_bytes"]),
                )
            except (TransportError, SourceSchemaError):
                blocked = True
            check(name, blocked)

    passed = sum(1 for item in checks if item["passed"])
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "SCHEMA_PROBE_SYNTHETIC_AUDIT_PASSED" if passed == len(checks) else "SCHEMA_PROBE_SYNTHETIC_AUDIT_FAILED",
        "check_count": len(checks),
        "passed_count": passed,
        "checks": checks,
        "external_network_used": False,
        "real_target_query_performed": False,
        "protected_canary_values_emitted": False if not any(canary in output_text for canary in canaries) else True,
        "claim_ceiling": "Synthetic safety audit of the schema-only diagnostic; no live-source or scientific result.",
    }
    audit_path = output_dir / "SCHEMA_PROBE_AUDIT_RESULT.json"
    atomic_json(audit_path, result)
    sums = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.name}")
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
    os.chmod(checksum_path, 0o444)
    return result


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "jx-reality-test-1-live-schema-probe-contract/v1":
        raise ProbeError("unsupported schema probe contract")
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("self-audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    live_parser = subparsers.add_parser("live")
    live_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    contract = load_contract(args.contract)
    limits = contract["limits"]
    if args.command == "self-audit":
        result = run_self_audit(args.output, limits)
        print(json.dumps({
            "status": result["status"],
            "passed_count": result["passed_count"],
            "check_count": result["check_count"],
        }, sort_keys=True))
        return 0 if result["status"] == "SCHEMA_PROBE_SYNTHETIC_AUDIT_PASSED" else 2

    result = probe_sources(
        designation=str(contract["target"]["designation_query"]),
        output_path=args.output,
        limits=limits,
        audit_mode=False,
    )
    print(json.dumps({
        "status": result["status"],
        "source_fetch_statuses": {
            source_id: report["fetch_status"] for source_id, report in result["sources"].items()
        },
        "compatibility_statuses": {
            source_id: report.get("status")
            for source_id, report in result["current_normalizer_compatibility"].items()
        } if isinstance(result["current_normalizer_compatibility"], dict) else {},
    }, sort_keys=True))
    return 0 if result["status"] == "LIVE_SCHEMA_PROBE_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
