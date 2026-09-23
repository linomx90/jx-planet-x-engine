"""Bounded HARPOS harmonic station-displacement support.

HARPOS is the public format used by NASA's International Mass Loading Service
for harmonic site-position variations.  This module parses a complete HARPOS
source fail-closed, selects one named site, and evaluates the published phase,
frequency, acceleration, and local displacement coefficients in TT.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


MODEL_ID = "station-displacement.ocean-loading.harpos-harmonics.v1"
HARPOS_FORMAT = "HARPOS  Format version of 2005.03.28"
HARPOS_SPECIFICATION_URL = "https://massloading.net/harpos_format.txt"
FES2014B_APOLLO_SOURCE_URL = (
    "https://massloading.net/imls/load_har_list/toc/fes2014b/"
    "toc_fes2014b_harmod.hps"
)
FES2014B_APOLLO_SOURCE_SHA256 = (
    "aa5519ce4c747a165396ddb03c3a22c3089e560a2cb538e6b2aa82f3dfdae70c"
)
FES2014B_HARMONIC_COUNT = 44
MAXIMUM_HARPOS_BYTES = 16 * 1024 * 1024
MAXIMUM_HARPOS_RECORDS = 100_000
J2000_JD = 2_451_545.0
SECONDS_PER_DAY = 86_400.0


class OceanLoadingError(ValueError):
    """A HARPOS source or ocean-loading evaluation violated its contract."""


def _token(value: object, label: str) -> str:
    if type(value) is not str or not value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise OceanLoadingError(f"{label} must be a visible-ASCII token")
    return value


def _float(value: str, label: str) -> float:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise OceanLoadingError(f"{label} is not a finite float") from exc
    if not math.isfinite(result):
        raise OceanLoadingError(f"{label} is not a finite float")
    return result


@dataclass(frozen=True, slots=True)
class HARPOSHarmonic:
    name: str
    phase_radians: float
    frequency_radians_per_second: float
    acceleration_radians_per_second2: float

    def __post_init__(self) -> None:
        _token(self.name, "harmonic name")
        for value in (
            self.phase_radians,
            self.frequency_radians_per_second,
            self.acceleration_radians_per_second2,
        ):
            if type(value) is not float or not math.isfinite(value):
                raise OceanLoadingError("harmonic values must be finite floats")


@dataclass(frozen=True, slots=True)
class HARPOSSite:
    name: str
    position_ecef_metres: tuple[float, float, float]

    def __post_init__(self) -> None:
        _token(self.name, "site name")
        if type(self.position_ecef_metres) is not tuple or len(
            self.position_ecef_metres
        ) != 3 or any(
            type(value) is not float or not math.isfinite(value)
            for value in self.position_ecef_metres
        ):
            raise OceanLoadingError("site position must be a finite float 3-tuple")
        radius = math.sqrt(
            math.fsum(value * value for value in self.position_ecef_metres)
        )
        if not 6.0e6 <= radius <= 7.0e6:
            raise OceanLoadingError("site radius is outside the terrestrial bound")


@dataclass(frozen=True, slots=True)
class HARPOSDisplacement:
    harmonic_name: str
    cosine_up_east_north_metres: tuple[float, float, float]
    sine_up_east_north_metres: tuple[float, float, float]

    def __post_init__(self) -> None:
        _token(self.harmonic_name, "displacement harmonic name")
        for vector in (
            self.cosine_up_east_north_metres,
            self.sine_up_east_north_metres,
        ):
            if type(vector) is not tuple or len(vector) != 3 or any(
                type(value) is not float or not math.isfinite(value)
                for value in vector
            ):
                raise OceanLoadingError(
                    "HARPOS amplitude must be a finite float 3-tuple"
                )
            if any(abs(value) > 1.0 for value in vector):
                raise OceanLoadingError("HARPOS amplitude exceeds one metre")


@dataclass(frozen=True, slots=True)
class HARPOSStationModel:
    model_id: str
    site: HARPOSSite
    applicability_radius_metres: float
    harmonics: Mapping[str, HARPOSHarmonic]
    displacements: tuple[HARPOSDisplacement, ...]
    source_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise OceanLoadingError("ocean-loading model id changed")
        if type(self.site) is not HARPOSSite:
            raise OceanLoadingError("HARPOS station model has the wrong site type")
        if (
            type(self.applicability_radius_metres) is not float
            or not math.isfinite(self.applicability_radius_metres)
            or self.applicability_radius_metres <= 0.0
        ):
            raise OceanLoadingError("HARPOS applicability radius is invalid")
        if type(self.harmonics) is not MappingProxyType or not self.harmonics:
            raise OceanLoadingError("HARPOS harmonics must be a nonempty immutable map")
        if type(self.displacements) is not tuple or not self.displacements or any(
            type(item) is not HARPOSDisplacement for item in self.displacements
        ):
            raise OceanLoadingError("HARPOS displacements must be a nonempty tuple")
        names = tuple(item.harmonic_name for item in self.displacements)
        if len(set(names)) != len(names) or set(names) != set(self.harmonics):
            raise OceanLoadingError("HARPOS harmonic/displacement roster is incomplete")
        _token(self.source_name, "HARPOS source name")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise OceanLoadingError("HARPOS source SHA-256 is invalid")

    def distance_from_site_metres(self, position_ecef_metres: object) -> float:
        value = np.asarray(position_ecef_metres, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise OceanLoadingError("requested station position is invalid")
        distance = float(
            np.linalg.norm(value - np.asarray(self.site.position_ecef_metres))
        )
        if distance > self.applicability_radius_metres:
            raise OceanLoadingError("requested station lies outside HARPOS applicability")
        return distance

    def displacement_ecef_metres(
        self,
        tt_jd: tuple[float, float],
        *,
        position_ecef_metres: object,
    ) -> np.ndarray:
        if type(tt_jd) is not tuple or len(tt_jd) != 2 or any(
            type(value) is not float or not math.isfinite(value) for value in tt_jd
        ):
            raise OceanLoadingError("TT epoch must be a finite float JD pair")
        self.distance_from_site_metres(position_ecef_metres)
        delta_seconds = math.fsum((tt_jd[0] - J2000_JD, tt_jd[1])) * SECONDS_PER_DAY
        local = np.zeros(3, dtype=np.float64)
        by_name = {item.harmonic_name: item for item in self.displacements}
        for name, harmonic in self.harmonics.items():
            amplitudes = by_name[name]
            angle = math.remainder(
                harmonic.phase_radians
                + harmonic.frequency_radians_per_second * delta_seconds
                + 0.5
                * harmonic.acceleration_radians_per_second2
                * delta_seconds**2,
                2.0 * math.pi,
            )
            local += (
                np.asarray(amplitudes.cosine_up_east_north_metres) * math.cos(angle)
                + np.asarray(amplitudes.sine_up_east_north_metres) * math.sin(angle)
            )

        site = np.asarray(self.site.position_ecef_metres, dtype=np.float64)
        radius = float(np.linalg.norm(site))
        horizontal = float(math.hypot(site[0], site[1]))
        if horizontal <= 0.0:
            raise OceanLoadingError("polar HARPOS sites are unsupported")
        sin_phi = float(site[2] / radius)
        cos_phi = horizontal / radius
        sin_lam = float(site[1] / horizontal)
        cos_lam = float(site[0] / horizontal)
        up = np.asarray((cos_phi * cos_lam, cos_phi * sin_lam, sin_phi))
        east = np.asarray((-sin_lam, cos_lam, 0.0))
        north = np.asarray((-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi))
        result = local[0] * up + local[1] * east + local[2] * north
        if not np.all(np.isfinite(result)) or float(np.linalg.norm(result)) > 1.0:
            raise OceanLoadingError("ocean-loading displacement is invalid")
        return np.ascontiguousarray(result, dtype=np.float64)


def parse_harpos_station(
    payload: bytes,
    *,
    source_name: str,
    site_name: str,
) -> HARPOSStationModel:
    """Parse and fully account for a HARPOS source, returning one named site."""

    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > MAXIMUM_HARPOS_BYTES
    ):
        raise OceanLoadingError("HARPOS payload must be nonempty bounded bytes")
    _token(source_name, "HARPOS source name")
    _token(site_name, "HARPOS requested site")
    if b"\x00" in payload:
        raise OceanLoadingError("HARPOS payload contains a NUL byte")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise OceanLoadingError("HARPOS payload must be ASCII") from exc
    if not lines or len(lines) > MAXIMUM_HARPOS_RECORDS:
        raise OceanLoadingError("HARPOS record count is invalid")
    meaningful = [line.rstrip() for line in lines if line.strip() and not line.startswith("#")]
    if len(meaningful) < 5 or meaningful[0] != HARPOS_FORMAT or meaningful[-1] != HARPOS_FORMAT:
        raise OceanLoadingError("HARPOS header/trailer contract failed")

    harmonics: dict[str, HARPOSHarmonic] = {}
    sites: dict[str, HARPOSSite] = {}
    displacements: dict[tuple[str, str], HARPOSDisplacement] = {}
    applicability: float | None = None
    stage = "definition"
    for line_number, line in enumerate(meaningful[1:-1], start=2):
        tokens = line.split()
        record = tokens[0] if tokens else ""
        try:
            if record == "A" and len(tokens) == 2 and stage == "definition":
                if applicability is not None:
                    raise OceanLoadingError("HARPOS contains duplicate A records")
                applicability = _float(tokens[1], "HARPOS applicability radius")
            elif record == "H" and len(tokens) == 5 and stage == "definition":
                harmonic = HARPOSHarmonic(
                    name=tokens[1],
                    phase_radians=_float(tokens[2], "HARPOS phase"),
                    frequency_radians_per_second=_float(tokens[3], "HARPOS frequency"),
                    acceleration_radians_per_second2=_float(
                        tokens[4], "HARPOS acceleration"
                    ),
                )
                if harmonic.name in harmonics:
                    raise OceanLoadingError("HARPOS contains duplicate harmonics")
                harmonics[harmonic.name] = harmonic
            elif record == "S" and len(tokens) == 8 and stage != "data":
                stage = "site"
                site = HARPOSSite(
                    name=tokens[1],
                    position_ecef_metres=tuple(
                        _float(value, "HARPOS site coordinate") for value in tokens[2:5]
                    ),
                )
                if site.name in sites:
                    raise OceanLoadingError("HARPOS contains duplicate sites")
                sites[site.name] = site
                for value in tokens[5:]:
                    _float(value, "HARPOS informational site field")
            elif record == "D" and len(tokens) == 9 and stage in ("site", "data"):
                stage = "data"
                if tokens[1] not in harmonics or tokens[2] not in sites:
                    raise OceanLoadingError("HARPOS D record has an undefined reference")
                key = (tokens[2], tokens[1])
                if key in displacements:
                    raise OceanLoadingError("HARPOS contains duplicate D records")
                values = tuple(
                    _float(value, "HARPOS displacement amplitude") for value in tokens[3:]
                )
                displacements[key] = HARPOSDisplacement(
                    harmonic_name=tokens[1],
                    cosine_up_east_north_metres=values[:3],
                    sine_up_east_north_metres=values[3:],
                )
            else:
                raise OceanLoadingError("HARPOS record order or arity is invalid")
        except OceanLoadingError as exc:
            raise OceanLoadingError(f"HARPOS line {line_number}: {exc}") from exc
    if applicability is None or applicability <= 0.0 or not harmonics or not sites:
        raise OceanLoadingError("HARPOS definitions are incomplete")
    expected_keys = {(site, harmonic) for site in sites for harmonic in harmonics}
    if set(displacements) != expected_keys:
        raise OceanLoadingError("HARPOS source has an incomplete displacement matrix")
    try:
        selected_site = sites[site_name]
    except KeyError as exc:
        raise OceanLoadingError("requested HARPOS site is absent") from exc
    selected = tuple(
        displacements[(site_name, name)] for name in harmonics
    )
    return HARPOSStationModel(
        model_id=MODEL_ID,
        site=selected_site,
        applicability_radius_metres=float(applicability),
        harmonics=MappingProxyType(harmonics),
        displacements=selected,
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "FES2014B_APOLLO_SOURCE_SHA256",
    "FES2014B_APOLLO_SOURCE_URL",
    "FES2014B_HARMONIC_COUNT",
    "HARPOSDisplacement",
    "HARPOSHarmonic",
    "HARPOSSite",
    "HARPOSStationModel",
    "HARPOS_FORMAT",
    "HARPOS_SPECIFICATION_URL",
    "MODEL_ID",
    "OceanLoadingError",
    "parse_harpos_station",
]
