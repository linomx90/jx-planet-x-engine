from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .errors import ValidationError
from .util import canonical_json_bytes

_TYPE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError("obs_time_utc must be a non-empty string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValidationError("obs_time_utc is not valid ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValidationError("obs_time_utc must include an explicit UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    return parsed


def canonical_utc(value: str) -> str:
    return parse_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValidationError("each normalized observation must be a JSON object")
    required = ("record_id", "obs_time_utc", "station_code", "measurement_type", "measurement")
    for key in required:
        if key not in record:
            raise ValidationError(f"missing required observation field: {key}")
    record_id = record["record_id"]
    station = record["station_code"]
    measurement_type = record["measurement_type"]
    if not isinstance(record_id, str) or not record_id or len(record_id) > 256:
        raise ValidationError("record_id must be a non-empty string no longer than 256 characters")
    if not isinstance(station, str) or not station or len(station) > 64:
        raise ValidationError("station_code must be a non-empty string no longer than 64 characters")
    if not isinstance(measurement_type, str) or not _TYPE_RE.fullmatch(measurement_type):
        raise ValidationError("measurement_type contains forbidden characters")
    if not isinstance(record["measurement"], dict) or not record["measurement"]:
        raise ValidationError("measurement must be a non-empty object")
    if "uncertainty" in record and not isinstance(record["uncertainty"], dict):
        raise ValidationError("uncertainty must be an object when present")
    normalized = dict(record)
    normalized["obs_time_utc"] = canonical_utc(record["obs_time_utc"])
    for optional in ("transmitter_code", "receiver_code"):
        if optional in normalized and normalized[optional] is not None:
            if not isinstance(normalized[optional], str) or len(normalized[optional]) > 64:
                raise ValidationError(f"{optional} must be a short string or null")
    # This also rejects NaN/Infinity and non-JSON types without echoing values.
    try:
        canonical_json_bytes(normalized)
    except (TypeError, ValueError) as error:
        raise ValidationError("observation contains a non-JSON or non-finite value") from error
    return normalized


def schedule_row(record: dict[str, Any], dataset_class: str, schedule_id: str) -> dict[str, str]:
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValidationError("schedule_id must be a non-empty string")
    return {
        "schedule_id": schedule_id,
        "obs_time_utc": record["obs_time_utc"],
        "station_code": record["station_code"],
        "measurement_type": record["measurement_type"],
        "transmitter_code": record.get("transmitter_code") or "",
        "receiver_code": record.get("receiver_code") or "",
        "dataset_class": dataset_class,
    }
