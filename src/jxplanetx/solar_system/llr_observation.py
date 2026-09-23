"""Screening-only lunar laser-ranging observation foundations.

The parser accepts the ILRS Consolidated Laser Ranging Data Format (CRD) v2
subset required for passive lunar-surface normal points.  The light-time
solver is deliberately provider-agnostic: callers must supply station and
reflector positions in one common inertial frame and coordinate time.

This module does not provide UTC-to-coordinate-time conversion, Earth
orientation, station displacement, lunar reflector coordinates, atmospheric
delay, relativistic propagation delay, parameter estimation, or a raw-LLR
validation claim.  Those omissions are explicit in every computed result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np


MODEL_ID = "observation.llr.crd-v2-two-way-screen.v2"
SPEED_OF_LIGHT_KM_S = 299_792.458
DEFAULT_BINARY64_TWO_CYCLE_TOLERANCE_SECONDS = 1.0e-11
MAXIMUM_PAYLOAD_BYTES = 16 * 1024 * 1024
MAXIMUM_LINE_BYTES = 4096
MAXIMUM_RECORDS = 1_000_000
SUPPORTED_STATION_TIME_SCALE_CODES = frozenset((3, 4, 7))
SUPPORTED_EPOCH_EVENTS = frozenset((0, 1, 2))
CLAIMS: Mapping[str, object] = MappingProxyType(
    {
        "scientific_claim_state": "SCREENING_ONLY",
        "crd_v2_ingestion": True,
        "retarded_two_way_geometric_light_time": True,
        "raw_llr_validated": False,
        "utc_to_coordinate_time_implemented": False,
        "earth_orientation_implemented": False,
        "station_displacement_implemented": False,
        "troposphere_implemented": False,
        "relativistic_signal_delay_implemented": False,
        "parameter_estimation_implemented": False,
        "registry_authorized": False,
        "qualification_authorized": False,
        "production_ready": False,
    }
)


class LLRObservationError(ValueError):
    """An LLR observation contract or numerical evaluation failed."""


def _require_token(value: object, label: str) -> str:
    if type(value) is not str or not value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise LLRObservationError(f"{label} must be a nonempty visible-ASCII token")
    return value


def _integer(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError) as exc:
        raise LLRObservationError(f"{label} must be an integer") from exc
    return parsed


def _float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LLRObservationError(f"{label} must be a finite float") from exc
    if not math.isfinite(parsed):
        raise LLRObservationError(f"{label} must be a finite float")
    return parsed


def _optional_float(value: str, label: str) -> float | None:
    if value.lower() == "na":
        return None
    return _float(value, label)


def _optional_integer(value: str, label: str) -> int | None:
    if value.lower() == "na":
        return None
    return _integer(value, label)


def _utc_datetime(values: list[str], label: str) -> datetime:
    if len(values) != 6:
        raise LLRObservationError(f"{label} requires six UTC fields")
    fields = tuple(_integer(value, label) for value in values)
    try:
        return datetime(*fields, tzinfo=timezone.utc)
    except ValueError as exc:
        raise LLRObservationError(f"{label} is not a valid UTC date-time") from exc


@dataclass(frozen=True, slots=True)
class CRDStationHeader:
    station_name: str
    system_identifier: int
    system_number: int
    system_occupancy: int
    epoch_time_scale_code: int
    network: str

    def __post_init__(self) -> None:
        _require_token(self.station_name, "station_name")
        _require_token(self.network, "station_network")
        if self.epoch_time_scale_code not in SUPPORTED_STATION_TIME_SCALE_CODES:
            raise LLRObservationError(
                "only CRD v2 UTC station time-scale codes 3, 4, and 7 are supported"
            )
        if any(
            type(value) is not int or value < 0
            for value in (
                self.system_identifier,
                self.system_number,
                self.system_occupancy,
            )
        ):
            raise LLRObservationError("station identifiers must be nonnegative integers")

    @property
    def sod(self) -> str:
        return (
            f"{self.system_identifier:04d}"
            f"{self.system_number:02d}"
            f"{self.system_occupancy:02d}"
        )


@dataclass(frozen=True, slots=True)
class CRDTargetHeader:
    target_name: str
    ilrs_identifier: int
    sic: int | None
    norad_identifier: int | None
    spacecraft_epoch_time_scale: int
    target_class: int
    target_location: int

    def __post_init__(self) -> None:
        _require_token(self.target_name, "target_name")
        if type(self.ilrs_identifier) is not int or self.ilrs_identifier < 0:
            raise LLRObservationError("ILRS identifier must be nonnegative")
        if self.sic is not None and (type(self.sic) is not int or self.sic < 0):
            raise LLRObservationError("SIC must be nonnegative or unavailable")
        if self.norad_identifier is not None and (
            type(self.norad_identifier) is not int or self.norad_identifier < 0
        ):
            raise LLRObservationError("NORAD identifier must be nonnegative or unavailable")
        if (
            self.spacecraft_epoch_time_scale != 0
            or self.target_class != 1
            or self.target_location != 3
        ):
            raise LLRObservationError(
                "only passive retroreflectors on the lunar surface are supported"
            )


@dataclass(frozen=True, slots=True)
class CRDSystemConfiguration:
    configuration_id: str
    transmit_wavelength_nm: float
    component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_token(self.configuration_id, "configuration_id")
        if (
            type(self.transmit_wavelength_nm) is not float
            or not math.isfinite(self.transmit_wavelength_nm)
            or self.transmit_wavelength_nm <= 0.0
        ):
            raise LLRObservationError("transmit wavelength must be positive and finite")
        if type(self.component_ids) is not tuple:
            raise LLRObservationError("component_ids must be an exact tuple")
        for component in self.component_ids:
            _require_token(component, "component_id")


@dataclass(frozen=True, slots=True)
class CRDSessionHeader:
    start_utc: datetime
    end_utc: datetime | None
    release_number: int
    tropospheric_refraction_applied: bool
    center_of_mass_applied: bool
    receive_amplitude_applied: bool
    station_system_delay_applied: bool
    spacecraft_system_delay_applied: bool
    range_type: int
    data_quality: int

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is not timezone.utc:
            raise LLRObservationError("session start must be timezone-aware UTC")
        if self.end_utc is not None and (
            self.end_utc.tzinfo is not timezone.utc or self.end_utc < self.start_utc
        ):
            raise LLRObservationError("session end must be UTC and not precede start")
        if type(self.release_number) is not int or self.release_number < 0:
            raise LLRObservationError("release number must be nonnegative")
        for label, value in (
            ("tropospheric_refraction_applied", self.tropospheric_refraction_applied),
            ("center_of_mass_applied", self.center_of_mass_applied),
            ("receive_amplitude_applied", self.receive_amplitude_applied),
            ("station_system_delay_applied", self.station_system_delay_applied),
            ("spacecraft_system_delay_applied", self.spacecraft_system_delay_applied),
        ):
            if type(value) is not bool:
                raise LLRObservationError(f"{label} must be a built-in bool")
        if self.range_type != 2:
            raise LLRObservationError("only two-way CRD ranges are supported")
        if not self.station_system_delay_applied:
            raise LLRObservationError("normal points must have station system delay applied")
        if self.spacecraft_system_delay_applied:
            raise LLRObservationError(
                "passive lunar reflectors cannot have spacecraft system delay applied"
            )
        if self.data_quality not in (0, 1):
            raise LLRObservationError("poor or unknown-quality CRD sessions are rejected")


@dataclass(frozen=True, slots=True)
class CRDMeteorology:
    seconds_of_day: float
    pressure_millibar: float
    temperature_kelvin: float
    relative_humidity_percent: float
    origin: int

    def __post_init__(self) -> None:
        values = (
            self.seconds_of_day,
            self.pressure_millibar,
            self.temperature_kelvin,
            self.relative_humidity_percent,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise LLRObservationError("meteorology fields must be finite built-in floats")
        if not 0.0 <= self.seconds_of_day <= 86_400.0:
            raise LLRObservationError("meteorology seconds of day is outside UTC day")
        if self.pressure_millibar <= 0.0 or self.temperature_kelvin <= 0.0:
            raise LLRObservationError("pressure and temperature must be positive")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise LLRObservationError("relative humidity is outside [0, 100]")
        if self.origin not in (0, 1):
            raise LLRObservationError("meteorology origin must be measured or interpolated")


@dataclass(frozen=True, slots=True)
class CRDRangeSupplement:
    seconds_of_day: float
    configuration_id: str
    tropospheric_refraction_one_way_ps: float | None
    target_center_of_mass_one_way_m: float | None
    neutral_density_filter: float | None
    time_bias_applied_seconds: float | None
    range_rate_seconds_per_second: float | None

    def __post_init__(self) -> None:
        if (
            type(self.seconds_of_day) is not float
            or not math.isfinite(self.seconds_of_day)
            or not 0.0 <= self.seconds_of_day <= 86_400.0
        ):
            raise LLRObservationError("supplement seconds of day is invalid")
        _require_token(self.configuration_id, "configuration_id")
        for value in (
            self.tropospheric_refraction_one_way_ps,
            self.target_center_of_mass_one_way_m,
            self.neutral_density_filter,
            self.time_bias_applied_seconds,
            self.range_rate_seconds_per_second,
        ):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise LLRObservationError("range supplement contains a nonfinite value")


@dataclass(frozen=True, slots=True)
class CRDNormalPoint:
    epoch_utc: datetime
    seconds_of_day: float
    time_of_flight_seconds: float
    configuration_id: str
    epoch_event: int
    window_seconds: float
    raw_range_count: int
    bin_rms_ps: float | None
    bin_skew: float | None
    bin_kurtosis: float | None
    bin_peak_minus_mean_ps: float | None
    return_rate_percent: float | None
    detector_channel: int
    signal_to_noise_ratio: float | None

    def __post_init__(self) -> None:
        if self.epoch_utc.tzinfo is not timezone.utc:
            raise LLRObservationError("normal-point epoch must be timezone-aware UTC")
        if (
            type(self.seconds_of_day) is not float
            or not math.isfinite(self.seconds_of_day)
            or not 0.0 <= self.seconds_of_day <= 86_400.0
        ):
            raise LLRObservationError("normal-point seconds of day is invalid")
        if (
            type(self.time_of_flight_seconds) is not float
            or not math.isfinite(self.time_of_flight_seconds)
            or not 0.0 < self.time_of_flight_seconds < 3.0
        ):
            raise LLRObservationError("lunar two-way time of flight must be in (0, 3) s")
        _require_token(self.configuration_id, "configuration_id")
        if self.epoch_event not in SUPPORTED_EPOCH_EVENTS:
            raise LLRObservationError("unsupported CRD epoch event")
        if (
            type(self.window_seconds) is not float
            or not math.isfinite(self.window_seconds)
            or self.window_seconds <= 0.0
        ):
            raise LLRObservationError("normal-point window must be positive")
        if type(self.raw_range_count) is not int or self.raw_range_count < 0:
            raise LLRObservationError("raw range count must be nonnegative")
        if type(self.detector_channel) is not int or self.detector_channel < 0:
            raise LLRObservationError("detector channel must be nonnegative")
        for value in (
            self.bin_rms_ps,
            self.bin_skew,
            self.bin_kurtosis,
            self.bin_peak_minus_mean_ps,
            self.return_rate_percent,
            self.signal_to_noise_ratio,
        ):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise LLRObservationError("normal-point statistic is nonfinite")

    @property
    def observed_one_way_range_metres(self) -> float:
        return 0.5 * self.time_of_flight_seconds * SPEED_OF_LIGHT_KM_S * 1.0e3


@dataclass(frozen=True, slots=True)
class CRDSession:
    header: CRDSessionHeader
    meteorology: tuple[CRDMeteorology, ...]
    range_supplements: tuple[CRDRangeSupplement, ...]
    normal_points: tuple[CRDNormalPoint, ...]

    def __post_init__(self) -> None:
        if type(self.header) is not CRDSessionHeader:
            raise LLRObservationError("session header has the wrong type")
        if type(self.meteorology) is not tuple or any(
            type(item) is not CRDMeteorology for item in self.meteorology
        ):
            raise LLRObservationError("session meteorology must use exact CRD types")
        if type(self.range_supplements) is not tuple or any(
            type(item) is not CRDRangeSupplement for item in self.range_supplements
        ):
            raise LLRObservationError("session supplements must use exact CRD types")
        if type(self.normal_points) is not tuple or any(
            type(item) is not CRDNormalPoint for item in self.normal_points
        ):
            raise LLRObservationError("session normal points must use exact CRD types")
        if not self.normal_points:
            raise LLRObservationError("a retained LLR session requires a normal point")
        if not self.meteorology:
            raise LLRObservationError("a retained LLR session requires meteorology")


@dataclass(frozen=True, slots=True)
class CRDObservationFile:
    format_version: int
    production_utc: datetime
    station: CRDStationHeader
    target: CRDTargetHeader
    configurations: tuple[CRDSystemConfiguration, ...]
    sessions: tuple[CRDSession, ...]
    source_name: str
    source_sha256: str
    block_index: int | None = None
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        if self.format_version != 2:
            raise LLRObservationError("only CRD format version 2 is supported")
        if self.production_utc.tzinfo is not timezone.utc:
            raise LLRObservationError("production epoch must be UTC")
        if type(self.station) is not CRDStationHeader or type(self.target) is not CRDTargetHeader:
            raise LLRObservationError("file headers have the wrong type")
        if type(self.configurations) is not tuple or any(
            type(item) is not CRDSystemConfiguration for item in self.configurations
        ):
            raise LLRObservationError("file configurations must use exact CRD types")
        if type(self.sessions) is not tuple or any(
            type(item) is not CRDSession for item in self.sessions
        ):
            raise LLRObservationError("file sessions must use exact CRD types")
        if not self.configurations or not self.sessions:
            raise LLRObservationError("CRD file lacks configurations or sessions")
        _require_token(self.source_name, "source_name")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise LLRObservationError("source SHA-256 is invalid")
        if self.block_index is not None and (
            type(self.block_index) is not int or self.block_index < 0
        ):
            raise LLRObservationError("block index must be nonnegative or unavailable")
        if self.claims is not CLAIMS:
            raise LLRObservationError("LLR file claims must use the immutable claim object")

    @property
    def normal_point_count(self) -> int:
        return sum(len(session.normal_points) for session in self.sessions)


@dataclass(frozen=True, slots=True)
class CRDRejectedBlock:
    block_index: int
    source_sha256: str
    error: str

    def __post_init__(self) -> None:
        if type(self.block_index) is not int or self.block_index < 0:
            raise LLRObservationError("rejected block index must be nonnegative")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise LLRObservationError("rejected block SHA-256 is invalid")
        if type(self.error) is not str or not self.error:
            raise LLRObservationError("rejected block must retain its error")


@dataclass(frozen=True, slots=True)
class CRDObservationStream:
    source_name: str
    source_sha256: str
    block_count: int
    accepted_blocks: tuple[CRDObservationFile, ...]
    rejected_blocks: tuple[CRDRejectedBlock, ...]
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        _require_token(self.source_name, "source_name")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise LLRObservationError("stream SHA-256 is invalid")
        if (
            type(self.block_count) is not int
            or self.block_count <= 0
            or self.block_count
            != len(self.accepted_blocks) + len(self.rejected_blocks)
        ):
            raise LLRObservationError("stream block accounting changed")
        if type(self.accepted_blocks) is not tuple or type(self.rejected_blocks) is not tuple:
            raise LLRObservationError("stream block collections must be exact tuples")
        if any(type(block) is not CRDObservationFile for block in self.accepted_blocks):
            raise LLRObservationError("accepted stream blocks have the wrong type")
        if any(type(block) is not CRDRejectedBlock for block in self.rejected_blocks):
            raise LLRObservationError("rejected stream blocks have the wrong type")
        indices = tuple(
            block.block_index for block in self.accepted_blocks
        ) + tuple(block.block_index for block in self.rejected_blocks)
        if any(index is None for index in indices) or sorted(indices) != list(
            range(self.block_count)
        ):
            raise LLRObservationError("stream block indices are not exhaustive")
        if self.claims is not CLAIMS:
            raise LLRObservationError("LLR stream claims must use the immutable claim object")

    @property
    def normal_point_count(self) -> int:
        return sum(block.normal_point_count for block in self.accepted_blocks)

    @property
    def all_blocks_accepted(self) -> bool:
        return not self.rejected_blocks


@dataclass(frozen=True, slots=True)
class CRDObservationCollection:
    """One source containing multiple independently terminated CRD streams."""

    source_name: str
    source_sha256: str
    segment_count: int
    block_count: int
    accepted_blocks: tuple[CRDObservationFile, ...]
    rejected_blocks: tuple[CRDRejectedBlock, ...]
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        _require_token(self.source_name, "source_name")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_sha256
            )
        ):
            raise LLRObservationError("collection SHA-256 is invalid")
        if type(self.segment_count) is not int or self.segment_count <= 0:
            raise LLRObservationError("collection segment count must be positive")
        if (
            type(self.block_count) is not int
            or self.block_count <= 0
            or self.block_count
            != len(self.accepted_blocks) + len(self.rejected_blocks)
        ):
            raise LLRObservationError("collection block accounting changed")
        if type(self.accepted_blocks) is not tuple or type(
            self.rejected_blocks
        ) is not tuple:
            raise LLRObservationError("collection blocks must be exact tuples")
        if any(
            type(block) is not CRDObservationFile for block in self.accepted_blocks
        ) or any(
            type(block) is not CRDRejectedBlock for block in self.rejected_blocks
        ):
            raise LLRObservationError("collection block types changed")
        indices = tuple(
            block.block_index for block in self.accepted_blocks
        ) + tuple(block.block_index for block in self.rejected_blocks)
        if any(index is None for index in indices) or sorted(indices) != list(
            range(self.block_count)
        ):
            raise LLRObservationError("collection block indices are not exhaustive")
        if self.claims is not CLAIMS:
            raise LLRObservationError(
                "LLR collection claims must use the immutable claim object"
            )

    @property
    def normal_point_count(self) -> int:
        return sum(block.normal_point_count for block in self.accepted_blocks)

    @property
    def all_blocks_accepted(self) -> bool:
        return not self.rejected_blocks


@dataclass(frozen=True, slots=True)
class _SessionBuilder:
    header: CRDSessionHeader
    meteorology: list[CRDMeteorology]
    range_supplements: list[CRDRangeSupplement]
    normal_points: list[CRDNormalPoint]


def _flag(value: str, label: str) -> bool:
    parsed = _integer(value, label)
    if parsed not in (0, 1):
        raise LLRObservationError(f"{label} must be 0 or 1")
    return bool(parsed)


def _normal_point_epoch(header: CRDSessionHeader, seconds_of_day: float) -> datetime:
    midnight = header.start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    epoch = midnight + timedelta(seconds=seconds_of_day)
    if epoch < header.start_utc and header.end_utc is not None:
        epoch += timedelta(days=1)
    if epoch < header.start_utc or (
        header.end_utc is not None and epoch > header.end_utc
    ):
        raise LLRObservationError("normal-point epoch is outside its H4 session")
    return epoch


def parse_crd_v2_llr_normal_points(
    payload: bytes,
    *,
    source_name: str,
) -> CRDObservationFile:
    """Parse a strict CRD v2 passive-lunar normal-point file."""

    if type(payload) is not bytes or not payload or len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise LLRObservationError("CRD payload must be nonempty bounded bytes")
    _require_token(source_name, "source_name")
    if b"\x00" in payload:
        raise LLRObservationError("CRD payload contains a NUL byte")
    try:
        text = payload.decode("latin-1")
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 is total
        raise LLRObservationError("CRD payload is not ISO-8859-1") from exc
    raw_lines = text.splitlines()
    if not raw_lines or len(raw_lines) > MAXIMUM_RECORDS:
        raise LLRObservationError("CRD record count is invalid")

    version: int | None = None
    production: datetime | None = None
    station: CRDStationHeader | None = None
    target: CRDTargetHeader | None = None
    configurations: dict[str, CRDSystemConfiguration] = {}
    sessions: list[CRDSession] = []
    builder: _SessionBuilder | None = None
    saw_h9 = False

    for line_number, raw_line in enumerate(raw_lines, start=1):
        if len(raw_line.encode("latin-1")) > MAXIMUM_LINE_BYTES:
            raise LLRObservationError(f"CRD line {line_number} is too long")
        stripped = raw_line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        record = tokens[0].upper()
        try:
            if saw_h9:
                raise LLRObservationError("data follows the terminal H9 record")
            if record == "00":
                continue
            if record == "H1":
                if version is not None or len(tokens) != 7 or tokens[1].upper() != "CRD":
                    raise LLRObservationError("invalid or duplicate H1 record")
                version = _integer(tokens[2], "CRD version")
                if version != 2:
                    raise LLRObservationError("only CRD version 2 is supported")
                production = _utc_datetime(
                    [*tokens[3:7], "0", "0"], "H1 production epoch"
                )
            elif record == "H2":
                if station is not None or len(tokens) != 7:
                    raise LLRObservationError("invalid or duplicate H2 record")
                station = CRDStationHeader(
                    station_name=tokens[1],
                    system_identifier=_integer(tokens[2], "station system identifier"),
                    system_number=_integer(tokens[3], "station system number"),
                    system_occupancy=_integer(tokens[4], "station occupancy"),
                    epoch_time_scale_code=_integer(tokens[5], "station time scale"),
                    network=tokens[6],
                )
            elif record == "H3":
                parsed_target = CRDTargetHeader(
                    target_name=tokens[1] if len(tokens) == 8 else "",
                    ilrs_identifier=_integer(tokens[2], "ILRS target identifier")
                    if len(tokens) == 8
                    else -1,
                    sic=_optional_integer(tokens[3], "SIC") if len(tokens) == 8 else None,
                    norad_identifier=_optional_integer(tokens[4], "NORAD identifier")
                    if len(tokens) == 8
                    else None,
                    spacecraft_epoch_time_scale=_integer(tokens[5], "spacecraft time scale")
                    if len(tokens) == 8
                    else -1,
                    target_class=_integer(tokens[6], "target class")
                    if len(tokens) == 8
                    else -1,
                    target_location=_integer(tokens[7], "target location")
                    if len(tokens) == 8
                    else -1,
                )
                if target is not None and parsed_target != target:
                    raise LLRObservationError("multiple CRD targets are not supported")
                target = parsed_target
            elif record == "H4":
                if builder is not None or len(tokens) != 22:
                    raise LLRObservationError("invalid H4 nesting or field count")
                if _integer(tokens[1], "H4 data type") != 1:
                    raise LLRObservationError("only normal-point H4 sessions are supported")
                end = None
                if not all(value.lower() == "na" for value in tokens[8:14]):
                    end = _utc_datetime(tokens[8:14], "H4 end")
                header = CRDSessionHeader(
                    start_utc=_utc_datetime(tokens[2:8], "H4 start"),
                    end_utc=end,
                    release_number=_integer(tokens[14], "release number"),
                    tropospheric_refraction_applied=_flag(tokens[15], "refraction flag"),
                    center_of_mass_applied=_flag(tokens[16], "center-of-mass flag"),
                    receive_amplitude_applied=_flag(tokens[17], "receive-amplitude flag"),
                    station_system_delay_applied=_flag(tokens[18], "station-delay flag"),
                    spacecraft_system_delay_applied=_flag(tokens[19], "spacecraft-delay flag"),
                    range_type=_integer(tokens[20], "range type"),
                    data_quality=_integer(tokens[21], "data quality"),
                )
                builder = _SessionBuilder(header, [], [], [])
            elif record == "H8":
                if builder is None or len(tokens) != 1:
                    raise LLRObservationError("H8 has no open session")
                sessions.append(
                    CRDSession(
                        builder.header,
                        tuple(builder.meteorology),
                        tuple(builder.range_supplements),
                        tuple(builder.normal_points),
                    )
                )
                builder = None
            elif record == "H9":
                if builder is not None or len(tokens) != 1:
                    raise LLRObservationError("H9 encountered before session closure")
                saw_h9 = True
            elif record == "C0":
                if len(tokens) < 4 or _integer(tokens[1], "C0 detail type") != 0:
                    raise LLRObservationError("invalid C0 record")
                configuration = CRDSystemConfiguration(
                    configuration_id=tokens[3],
                    transmit_wavelength_nm=_float(tokens[2], "transmit wavelength"),
                    component_ids=tuple(tokens[4:]),
                )
                if configuration.configuration_id in configurations:
                    raise LLRObservationError("duplicate C0 configuration identifier")
                configurations[configuration.configuration_id] = configuration
            elif record == "20":
                if builder is None or len(tokens) != 6:
                    raise LLRObservationError("meteorology record is outside a session")
                builder.meteorology.append(
                    CRDMeteorology(
                        seconds_of_day=_float(tokens[1], "meteorology seconds of day"),
                        pressure_millibar=_float(tokens[2], "pressure"),
                        temperature_kelvin=_float(tokens[3], "temperature"),
                        relative_humidity_percent=_float(tokens[4], "humidity"),
                        origin=_integer(tokens[5], "meteorology origin"),
                    )
                )
            elif record == "12":
                if builder is None or len(tokens) != 8:
                    raise LLRObservationError("range supplement is outside a session")
                builder.range_supplements.append(
                    CRDRangeSupplement(
                        seconds_of_day=_float(tokens[1], "supplement seconds of day"),
                        configuration_id=tokens[2],
                        tropospheric_refraction_one_way_ps=_optional_float(
                            tokens[3], "tropospheric correction"
                        ),
                        target_center_of_mass_one_way_m=_optional_float(
                            tokens[4], "target center-of-mass correction"
                        ),
                        neutral_density_filter=_optional_float(tokens[5], "ND filter"),
                        time_bias_applied_seconds=_optional_float(tokens[6], "time bias"),
                        range_rate_seconds_per_second=_optional_float(
                            tokens[7], "range rate"
                        ),
                    )
                )
            elif record == "11":
                if builder is None or len(tokens) != 14:
                    raise LLRObservationError("normal point is outside a session or malformed")
                seconds = _float(tokens[1], "normal-point seconds of day")
                builder.normal_points.append(
                    CRDNormalPoint(
                        epoch_utc=_normal_point_epoch(builder.header, seconds),
                        seconds_of_day=seconds,
                        time_of_flight_seconds=_float(tokens[2], "time of flight"),
                        configuration_id=tokens[3],
                        epoch_event=_integer(tokens[4], "epoch event"),
                        window_seconds=_float(tokens[5], "normal-point window"),
                        raw_range_count=_integer(tokens[6], "raw range count"),
                        bin_rms_ps=_optional_float(tokens[7], "bin RMS"),
                        bin_skew=_optional_float(tokens[8], "bin skew"),
                        bin_kurtosis=_optional_float(tokens[9], "bin kurtosis"),
                        bin_peak_minus_mean_ps=_optional_float(
                            tokens[10], "bin peak minus mean"
                        ),
                        return_rate_percent=_optional_float(tokens[11], "return rate"),
                        detector_channel=_integer(tokens[12], "detector channel"),
                        signal_to_noise_ratio=_optional_float(tokens[13], "SNR"),
                    )
                )
            elif record in {
                "H5",
                "C1",
                "C2",
                "C3",
                "C5",
                "C6",
                "C7",
                "21",
                "30",
                "40",
                "41",
                "42",
                "50",
                "60",
            }:
                continue
            else:
                raise LLRObservationError(f"unsupported CRD record {record!r}")
        except (IndexError, LLRObservationError) as exc:
            if isinstance(exc, LLRObservationError):
                raise LLRObservationError(f"CRD line {line_number}: {exc}") from exc
            raise LLRObservationError(f"CRD line {line_number}: missing field") from exc

    if builder is not None:
        raise LLRObservationError("CRD payload ended with an open session")
    if None in (version, production, station, target) or not saw_h9:
        raise LLRObservationError("CRD payload lacks required headers or H9")
    configuration_ids = frozenset(configurations)
    for session in sessions:
        for point in session.normal_points:
            if point.configuration_id not in configuration_ids:
                raise LLRObservationError(
                    "normal point references an unknown C0 configuration"
                )
        for supplement in session.range_supplements:
            if supplement.configuration_id not in configuration_ids:
                raise LLRObservationError(
                    "range supplement references an unknown C0 configuration"
                )
    return CRDObservationFile(
        format_version=version,
        production_utc=production,
        station=station,
        target=target,
        configurations=tuple(configurations.values()),
        sessions=tuple(sessions),
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def parse_crd_v2_llr_normal_point_stream(
    payload: bytes,
    *,
    source_name: str,
) -> CRDObservationStream:
    """Parse concatenated CRD v2 blocks with explicit block-level rejection.

    ILRS data-center daily files may concatenate self-contained ``H1`` through
    ``H8`` blocks and place one ``H9`` at the end of the stream.  Contradictory
    or unsupported blocks are retained as hashed rejection records; they are
    never silently reinterpreted as lunar observations.
    """

    if type(payload) is not bytes or not payload or len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise LLRObservationError("CRD stream must be nonempty bounded bytes")
    _require_token(source_name, "source_name")
    if b"\x00" in payload:
        raise LLRObservationError("CRD stream contains a NUL byte")
    raw_lines = payload.splitlines()
    if not raw_lines or len(raw_lines) > MAXIMUM_RECORDS:
        raise LLRObservationError("CRD stream record count is invalid")
    nonempty_records = [
        raw_line.strip().split(maxsplit=1)[0].upper()
        for raw_line in raw_lines
        if raw_line.strip()
    ]
    if nonempty_records.count(b"H9") != 1 or nonempty_records[-1] != b"H9":
        raise LLRObservationError("CRD stream requires one terminal H9 record")

    blocks: list[list[bytes]] = []
    current: list[bytes] | None = None
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if len(raw_line) > MAXIMUM_LINE_BYTES:
            raise LLRObservationError(f"CRD stream line {line_number} is too long")
        token = raw_line.strip().split(maxsplit=1)
        record = token[0].upper() if token else b""
        if record == b"H1":
            if current is not None:
                blocks.append(current)
            current = [raw_line]
        elif record == b"H9":
            continue
        elif current is None:
            if record:
                raise LLRObservationError("CRD stream contains data before its first H1")
        else:
            current.append(raw_line)
    if current is not None:
        blocks.append(current)
    if not blocks:
        raise LLRObservationError("CRD stream contains no H1 blocks")

    accepted: list[CRDObservationFile] = []
    rejected: list[CRDRejectedBlock] = []
    for block_index, block_lines in enumerate(blocks):
        block_payload = b"\n".join((*block_lines, b"H9")) + b"\n"
        block_sha256 = hashlib.sha256(block_payload).hexdigest()
        try:
            accepted.append(
                replace(
                    parse_crd_v2_llr_normal_points(
                    block_payload,
                    source_name=f"{source_name}#block-{block_index}",
                    ),
                    block_index=block_index,
                )
            )
        except LLRObservationError as exc:
            rejected.append(
                CRDRejectedBlock(
                    block_index=block_index,
                    source_sha256=block_sha256,
                    error=str(exc),
                )
            )
    return CRDObservationStream(
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        block_count=len(blocks),
        accepted_blocks=tuple(accepted),
        rejected_blocks=tuple(rejected),
    )


def parse_crd_v2_llr_normal_point_collection(
    payload: bytes,
    *,
    source_name: str,
) -> CRDObservationCollection:
    """Parse a source containing multiple H9-terminated CRD v2 streams.

    This separate boundary preserves the stricter single-terminal-H9 contract
    of :func:`parse_crd_v2_llr_normal_point_stream`.  Every collection segment
    must begin with H1, end with H9, and contain no records after H9 except the
    next segment's H1.
    """

    if type(payload) is not bytes or not payload or len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise LLRObservationError("CRD collection must be nonempty bounded bytes")
    _require_token(source_name, "source_name")
    if b"\x00" in payload:
        raise LLRObservationError("CRD collection contains a NUL byte")
    raw_lines = payload.splitlines()
    if not raw_lines or len(raw_lines) > MAXIMUM_RECORDS:
        raise LLRObservationError("CRD collection record count is invalid")

    segments: list[bytes] = []
    current: list[bytes] | None = None
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if len(raw_line) > MAXIMUM_LINE_BYTES:
            raise LLRObservationError(
                f"CRD collection line {line_number} is too long"
            )
        tokens = raw_line.strip().split(maxsplit=1)
        record = tokens[0].upper() if tokens else b""
        if current is None:
            if not record:
                continue
            if record != b"H1":
                raise LLRObservationError(
                    "CRD collection segment must begin with H1"
                )
            current = [raw_line]
            continue
        if record == b"H9":
            current.append(raw_line)
            segments.append(b"\n".join(current) + b"\n")
            current = None
            continue
        current.append(raw_line)
    if current is not None:
        raise LLRObservationError("CRD collection has an unterminated segment")
    if not segments:
        raise LLRObservationError("CRD collection contains no complete segment")

    accepted: list[CRDObservationFile] = []
    rejected: list[CRDRejectedBlock] = []
    offset = 0
    for segment_index, segment in enumerate(segments):
        parsed = parse_crd_v2_llr_normal_point_stream(
            segment,
            source_name=f"{source_name}#segment-{segment_index}",
        )
        accepted.extend(
            replace(block, block_index=offset + int(block.block_index))
            for block in parsed.accepted_blocks
        )
        rejected.extend(
            CRDRejectedBlock(
                block_index=offset + block.block_index,
                source_sha256=block.source_sha256,
                error=block.error,
            )
            for block in parsed.rejected_blocks
        )
        offset += parsed.block_count
    return CRDObservationCollection(
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        segment_count=len(segments),
        block_count=offset,
        accepted_blocks=tuple(accepted),
        rejected_blocks=tuple(rejected),
    )


def interpolate_meteorology(
    records: tuple[CRDMeteorology, ...],
    seconds_of_day: float,
) -> CRDMeteorology:
    """Linearly interpolate CRD meteorology without claiming a delay model."""

    if type(records) is not tuple or not records:
        raise LLRObservationError("meteorology records must be a nonempty exact tuple")
    if (
        type(seconds_of_day) is not float
        or not math.isfinite(seconds_of_day)
        or not 0.0 <= seconds_of_day <= 86_400.0
    ):
        raise LLRObservationError("requested meteorology epoch is invalid")
    ordered = tuple(sorted(records, key=lambda record: record.seconds_of_day))
    if any(type(record) is not CRDMeteorology for record in ordered):
        raise LLRObservationError("meteorology tuple contains the wrong type")
    if seconds_of_day <= ordered[0].seconds_of_day:
        return ordered[0]
    if seconds_of_day >= ordered[-1].seconds_of_day:
        return ordered[-1]
    for left, right in zip(ordered, ordered[1:], strict=True):
        if left.seconds_of_day <= seconds_of_day <= right.seconds_of_day:
            width = right.seconds_of_day - left.seconds_of_day
            if width <= 0.0:
                raise LLRObservationError("meteorology epochs must be unique")
            fraction = (seconds_of_day - left.seconds_of_day) / width
            return CRDMeteorology(
                seconds_of_day=seconds_of_day,
                pressure_millibar=left.pressure_millibar
                + fraction * (right.pressure_millibar - left.pressure_millibar),
                temperature_kelvin=left.temperature_kelvin
                + fraction * (right.temperature_kelvin - left.temperature_kelvin),
                relative_humidity_percent=left.relative_humidity_percent
                + fraction
                * (right.relative_humidity_percent - left.relative_humidity_percent),
                origin=1,
            )
    raise LLRObservationError("meteorology interpolation bracket was not found")


@dataclass(frozen=True, slots=True)
class LLRCorrectionBudget:
    troposphere_round_trip_seconds: float
    relativistic_round_trip_seconds: float
    other_round_trip_seconds: float
    corrections_complete: bool
    model_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.troposphere_round_trip_seconds,
            self.relativistic_round_trip_seconds,
            self.other_round_trip_seconds,
        ):
            if type(value) is not float or not math.isfinite(value) or abs(value) > 0.01:
                raise LLRObservationError(
                    "each round-trip correction must be a finite built-in float within 0.01 s"
                )
        if type(self.corrections_complete) is not bool:
            raise LLRObservationError("corrections_complete must be a built-in bool")
        if type(self.model_ids) is not tuple:
            raise LLRObservationError("model_ids must be an exact tuple")
        for model_id in self.model_ids:
            _require_token(model_id, "correction model id")
        if self.corrections_complete and not self.model_ids:
            raise LLRObservationError(
                "a complete correction budget must identify its models"
            )

    @property
    def total_round_trip_seconds(self) -> float:
        return math.fsum(
            (
                self.troposphere_round_trip_seconds,
                self.relativistic_round_trip_seconds,
                self.other_round_trip_seconds,
            )
        )


@dataclass(frozen=True, slots=True)
class LLRLightTimeResult:
    model_id: str
    provider_model_id: str
    epoch_event: int
    transmit_epoch_seconds: float
    bounce_epoch_seconds: float
    receive_epoch_seconds: float
    uplink_distance_km: float
    downlink_distance_km: float
    geometric_round_trip_seconds: float
    applied_correction_seconds: float
    predicted_round_trip_seconds: float
    observed_round_trip_seconds: float
    round_trip_residual_seconds: float
    one_way_range_residual_metres: float
    iteration_count: int
    converged: bool
    corrections_complete: bool
    convergence_mode: str
    convergence_bound_seconds: float
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise LLRObservationError("light-time model id changed")
        _require_token(self.provider_model_id, "provider_model_id")
        if self.epoch_event not in SUPPORTED_EPOCH_EVENTS:
            raise LLRObservationError("light-time result has an unsupported event")
        numerical = (
            self.transmit_epoch_seconds,
            self.bounce_epoch_seconds,
            self.receive_epoch_seconds,
            self.uplink_distance_km,
            self.downlink_distance_km,
            self.geometric_round_trip_seconds,
            self.applied_correction_seconds,
            self.predicted_round_trip_seconds,
            self.observed_round_trip_seconds,
            self.round_trip_residual_seconds,
            self.one_way_range_residual_metres,
            self.convergence_bound_seconds,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in numerical):
            raise LLRObservationError("light-time result contains a nonfinite value")
        if not (
            self.transmit_epoch_seconds
            < self.bounce_epoch_seconds
            < self.receive_epoch_seconds
        ):
            raise LLRObservationError("light-time epochs are not ordered")
        if self.uplink_distance_km <= 0.0 or self.downlink_distance_km <= 0.0:
            raise LLRObservationError("light-time path distance must be positive")
        if type(self.iteration_count) is not int or self.iteration_count <= 0:
            raise LLRObservationError("iteration count must be positive")
        if self.converged is not True:
            raise LLRObservationError("only converged light-time results may be created")
        if type(self.corrections_complete) is not bool:
            raise LLRObservationError("corrections_complete must be a bool")
        if self.convergence_mode not in (
            "FIXED_POINT_TOLERANCE",
            "BOUNDED_BINARY64_TWO_CYCLE",
        ):
            raise LLRObservationError("light-time convergence mode changed")
        if (
            self.convergence_bound_seconds < 0.0
            or self.convergence_bound_seconds
            > DEFAULT_BINARY64_TWO_CYCLE_TOLERANCE_SECONDS
        ):
            raise LLRObservationError("light-time convergence bound is invalid")
        if (
            self.convergence_mode == "BOUNDED_BINARY64_TWO_CYCLE"
            and self.convergence_bound_seconds == 0.0
        ):
            raise LLRObservationError("two-cycle convergence requires a positive bound")
        if self.claims is not CLAIMS:
            raise LLRObservationError("LLR result claims must remain immutable")


PositionProvider = Callable[[float], np.ndarray]


def _position(provider: PositionProvider, epoch: float, label: str) -> np.ndarray:
    try:
        value = provider(epoch)
    except Exception as exc:
        raise LLRObservationError(f"{label} provider failed") from exc
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LLRObservationError(
            f"{label} provider must return a finite C-contiguous float64 (3,) array"
        )
    return value


def _distance(
    left_provider: PositionProvider,
    left_epoch: float,
    right_provider: PositionProvider,
    right_epoch: float,
    left_label: str,
    right_label: str,
) -> float:
    left = _position(left_provider, left_epoch, left_label)
    right = _position(right_provider, right_epoch, right_label)
    distance = float(np.linalg.norm(right - left))
    if not math.isfinite(distance) or distance <= 0.0:
        raise LLRObservationError("light-time path has an invalid distance")
    return distance


def _fixed_point_delay(
    update: Callable[[float], float],
    initial: float,
    *,
    tolerance_seconds: float,
    maximum_iterations: int,
    binary64_two_cycle_tolerance_seconds: float,
) -> tuple[float, int, str, float]:
    delay = initial
    two_steps_back: float | None = None
    for iteration in range(1, maximum_iterations + 1):
        candidate = update(delay)
        if not math.isfinite(candidate) or candidate <= 0.0 or candidate >= 3.0:
            raise LLRObservationError("light-time iteration produced an invalid delay")
        difference = abs(candidate - delay)
        if difference <= tolerance_seconds:
            return candidate, iteration, "FIXED_POINT_TOLERANCE", difference
        if candidate == two_steps_back:
            if difference > binary64_two_cycle_tolerance_seconds:
                raise LLRObservationError(
                    "light-time fixed point entered an excessive two-cycle"
                )
            return (
                0.5 * math.fsum((candidate, delay)),
                iteration,
                "BOUNDED_BINARY64_TWO_CYCLE",
                difference,
            )
        two_steps_back = delay
        delay = candidate
    raise LLRObservationError("light-time iteration did not converge")


def solve_two_way_llr_normal_point(
    normal_point: CRDNormalPoint,
    session: CRDSessionHeader,
    *,
    event_epoch_seconds: float,
    station_position_km: PositionProvider,
    reflector_position_km: PositionProvider,
    correction_budget: LLRCorrectionBudget,
    provider_model_id: str,
    tolerance_seconds: float = 1.0e-13,
    maximum_iterations: int = 32,
    binary64_two_cycle_tolerance_seconds: float = (
        DEFAULT_BINARY64_TWO_CYCLE_TOLERANCE_SECONDS
    ),
) -> LLRLightTimeResult:
    """Solve retarded two-way geometry for one CRD normal point.

    ``event_epoch_seconds`` must already represent the CRD event in the same
    coordinate time used by both position providers.  No UTC conversion is
    performed here.
    """

    if type(normal_point) is not CRDNormalPoint or type(session) is not CRDSessionHeader:
        raise LLRObservationError("normal point and session must use exact CRD types")
    if type(correction_budget) is not LLRCorrectionBudget:
        raise LLRObservationError("correction_budget has the wrong type")
    _require_token(provider_model_id, "provider_model_id")
    if (
        type(event_epoch_seconds) is not float
        or not math.isfinite(event_epoch_seconds)
        or type(tolerance_seconds) is not float
        or not math.isfinite(tolerance_seconds)
        or tolerance_seconds <= 0.0
        or type(maximum_iterations) is not int
        or maximum_iterations <= 0
        or type(binary64_two_cycle_tolerance_seconds) is not float
        or not math.isfinite(binary64_two_cycle_tolerance_seconds)
        or binary64_two_cycle_tolerance_seconds < tolerance_seconds
    ):
        raise LLRObservationError("light-time iteration controls are invalid")
    if session.tropospheric_refraction_applied and (
        correction_budget.troposphere_round_trip_seconds != 0.0
    ):
        raise LLRObservationError(
            "troposphere correction would be double counted against CRD flags"
        )

    event = normal_point.epoch_event
    initial = normal_point.time_of_flight_seconds * 0.5
    iterations = 0
    convergence_modes: list[str] = []
    convergence_bounds: list[float] = []
    if event == 2:
        transmit = event_epoch_seconds
        uplink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                station_position_km,
                transmit,
                reflector_position_km,
                transmit + delay,
                "station",
                "reflector",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        bounce = transmit + uplink
        downlink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                reflector_position_km,
                bounce,
                station_position_km,
                bounce + delay,
                "reflector",
                "station",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        receive = bounce + downlink
    elif event == 1:
        bounce = event_epoch_seconds
        uplink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                station_position_km,
                bounce - delay,
                reflector_position_km,
                bounce,
                "station",
                "reflector",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        transmit = bounce - uplink
        downlink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                reflector_position_km,
                bounce,
                station_position_km,
                bounce + delay,
                "reflector",
                "station",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        receive = bounce + downlink
    else:
        receive = event_epoch_seconds
        downlink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                reflector_position_km,
                receive - delay,
                station_position_km,
                receive,
                "reflector",
                "station",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        bounce = receive - downlink
        uplink, count, mode, bound = _fixed_point_delay(
            lambda delay: _distance(
                station_position_km,
                bounce - delay,
                reflector_position_km,
                bounce,
                "station",
                "reflector",
            )
            / SPEED_OF_LIGHT_KM_S,
            initial,
            tolerance_seconds=tolerance_seconds,
            maximum_iterations=maximum_iterations,
            binary64_two_cycle_tolerance_seconds=(
                binary64_two_cycle_tolerance_seconds
            ),
        )
        iterations += count
        convergence_modes.append(mode)
        convergence_bounds.append(bound)
        transmit = bounce - uplink

    uplink_distance = _distance(
        station_position_km,
        transmit,
        reflector_position_km,
        bounce,
        "station",
        "reflector",
    )
    downlink_distance = _distance(
        reflector_position_km,
        bounce,
        station_position_km,
        receive,
        "reflector",
        "station",
    )
    geometric = (uplink_distance + downlink_distance) / SPEED_OF_LIGHT_KM_S
    correction = correction_budget.total_round_trip_seconds
    predicted = geometric + correction
    observed = normal_point.time_of_flight_seconds
    residual = observed - predicted
    return LLRLightTimeResult(
        model_id=MODEL_ID,
        provider_model_id=provider_model_id,
        epoch_event=event,
        transmit_epoch_seconds=float(transmit),
        bounce_epoch_seconds=float(bounce),
        receive_epoch_seconds=float(receive),
        uplink_distance_km=uplink_distance,
        downlink_distance_km=downlink_distance,
        geometric_round_trip_seconds=geometric,
        applied_correction_seconds=correction,
        predicted_round_trip_seconds=predicted,
        observed_round_trip_seconds=observed,
        round_trip_residual_seconds=residual,
        one_way_range_residual_metres=(
            0.5 * residual * SPEED_OF_LIGHT_KM_S * 1.0e3
        ),
        iteration_count=iterations,
        converged=True,
        corrections_complete=correction_budget.corrections_complete,
        convergence_mode=(
            "BOUNDED_BINARY64_TWO_CYCLE"
            if "BOUNDED_BINARY64_TWO_CYCLE" in convergence_modes
            else "FIXED_POINT_TOLERANCE"
        ),
        convergence_bound_seconds=max(convergence_bounds),
    )


@dataclass(frozen=True, slots=True)
class LLRResidualSummary:
    count: int
    mean_one_way_range_residual_metres: float
    rms_one_way_range_residual_metres: float
    maximum_absolute_one_way_range_residual_metres: float
    all_corrections_complete: bool
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count <= 0:
            raise LLRObservationError("residual summary count must be positive")
        values = (
            self.mean_one_way_range_residual_metres,
            self.rms_one_way_range_residual_metres,
            self.maximum_absolute_one_way_range_residual_metres,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise LLRObservationError("residual summary contains a nonfinite value")
        if self.rms_one_way_range_residual_metres < 0.0 or (
            self.maximum_absolute_one_way_range_residual_metres < 0.0
        ):
            raise LLRObservationError("residual norms must be nonnegative")
        if type(self.all_corrections_complete) is not bool:
            raise LLRObservationError("all_corrections_complete must be a bool")
        if self.claims is not CLAIMS:
            raise LLRObservationError("LLR summary claims must remain immutable")


def summarize_llr_residuals(
    results: tuple[LLRLightTimeResult, ...],
) -> LLRResidualSummary:
    """Compute descriptive, unweighted residual statistics without fitting."""

    if type(results) is not tuple or not results or any(
        type(result) is not LLRLightTimeResult for result in results
    ):
        raise LLRObservationError(
            "results must be a nonempty exact tuple of LLRLightTimeResult"
        )
    residuals = tuple(result.one_way_range_residual_metres for result in results)
    return LLRResidualSummary(
        count=len(residuals),
        mean_one_way_range_residual_metres=math.fsum(residuals) / len(residuals),
        rms_one_way_range_residual_metres=math.sqrt(
            math.fsum(value * value for value in residuals) / len(residuals)
        ),
        maximum_absolute_one_way_range_residual_metres=max(
            abs(value) for value in residuals
        ),
        all_corrections_complete=all(
            result.corrections_complete for result in results
        ),
    )


__all__ = [
    "CLAIMS",
    "CRDMeteorology",
    "CRDNormalPoint",
    "CRDObservationCollection",
    "CRDObservationFile",
    "CRDObservationStream",
    "CRDRangeSupplement",
    "CRDRejectedBlock",
    "CRDSession",
    "CRDSessionHeader",
    "CRDStationHeader",
    "CRDSystemConfiguration",
    "CRDTargetHeader",
    "LLRCorrectionBudget",
    "LLRLightTimeResult",
    "LLRObservationError",
    "LLRResidualSummary",
    "MODEL_ID",
    "SPEED_OF_LIGHT_KM_S",
    "interpolate_meteorology",
    "parse_crd_v2_llr_normal_points",
    "parse_crd_v2_llr_normal_point_stream",
    "parse_crd_v2_llr_normal_point_collection",
    "solve_two_way_llr_normal_point",
    "summarize_llr_residuals",
]
