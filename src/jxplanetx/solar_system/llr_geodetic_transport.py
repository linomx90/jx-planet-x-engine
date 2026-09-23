"""Screening-only geodetic transport and propagation terms for LLR.

This module connects CRD v2 normal points to DE440 geometry using IAU 2006/
2000A Earth rotation, observed IERS finals2000A Earth-orientation parameters,
and the DE440 lunar principal-axis frame.  It also implements independent
Python versions of the published Mendes--Pavlis FCUL zenith-delay and FCULa
mapping equations, the separately licensed IERS-2010-compatible solid-Earth-
tide adaptation, the IERS working-version solid pole tide, and a first-order
PPN gamma=1 point-mass signal delay.

The implementation is not IERS Conventions software and is not endorsed by
IERS.  It uses the published equations and official reference test vectors.
Station velocity, ocean pole/other site loading, the relativistic GCRS-to-BCRS
site transform, atmospheric gradients, parameter estimation, and independent
held-out validation remain absent, so all outputs remain screening-only.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import llr_observation as llr
from . import ocean_loading
from . import solid_earth_tide as solid_tide
from . import solid_pole_tide


TRANSPORT_MODEL_ID = "observation.llr.de440-iers-transport-screen.v4"
ATMOSPHERE_MODEL_ID = "propagation.optical.fculzd-fcula-buck-screen.v1"
RELATIVITY_MODEL_ID = "propagation.ppn-gamma1-static-midpoint-screen.v1"
SECONDS_PER_DAY = 86_400.0
J2000_JD = 2_451_545.0
MJD_ZERO_JD = 2_400_000.5
ARCSECOND_TO_RADIAN = math.pi / (180.0 * 3600.0)
SPEED_OF_LIGHT_M_S = llr.SPEED_OF_LIGHT_KM_S * 1.0e3
MAXIMUM_EOP_BYTES = 8 * 1024 * 1024
MAXIMUM_EOP_ROWS = 100_000
MINIMUM_FCULA_ELEVATION_DEGREES = 3.0
DE440_POINT_MASS_GM_KM3_S2 = MappingProxyType(
    {
        "SUN": 132_712_440_041.27942,
        "EARTH": 398_600.43550702266,
        "MOON": 4_902.8001184575496,
    }
)
MISSING_REDUCTION_COMPONENTS = (
    "STATION_REFERENCE_FRAME_VELOCITY_AND_DISCONTINUITY_HISTORY",
    "OCEAN_POLE_TIDE_AND_OTHER_SITE_DISPLACEMENT",
    "RELATIVISTIC_GCRS_TO_BCRS_STATION_POSITION_TRANSFORM",
    "ATMOSPHERIC_HORIZONTAL_GRADIENTS_AND_RAY_BENDING",
    "FULL_MOVING_BODY_RELATIVISTIC_TIME_TRANSFER_FUNCTION",
    "FROZEN_DATA_EDITING_WEIGHTING_AND_PARAMETER_ESTIMATION",
    "INDEPENDENT_HELD_OUT_RAW_LLR_RESIDUAL_GATE",
)
CLAIMS: Mapping[str, object] = MappingProxyType(
    {
        "scientific_claim_state": "SCREENING_ONLY",
        "crd_v2_ingestion": True,
        "utc_to_tt_tdb_with_erfa": True,
        "observed_iers_eop_rotation": True,
        "de440_lunar_pa_reflector_transport": True,
        "iers2010_solid_earth_tide_station_displacement": True,
        "fes2014b_harpos_ocean_loading_station_displacement": True,
        "iers_working_2018_solid_pole_tide_station_displacement": True,
        "fcul_atmosphere_screen": True,
        "static_midpoint_ppn_signal_delay_screen": True,
        "station_velocity_implemented": False,
        "station_displacement_complete": False,
        "complete_relativistic_signal_delay_implemented": False,
        "parameter_estimation_implemented": False,
        "raw_llr_validated": False,
        "independent_physical_validation": False,
        "registry_authorized": False,
        "qualification_authorized": False,
        "production_ready": False,
    }
)


class LLRGeodeticTransportError(ValueError):
    """A geodetic transport or propagation contract failed."""


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LLRGeodeticTransportError(f"{label} must be a finite built-in float")
    return value


def _visible_ascii(value: object, label: str) -> str:
    if type(value) is not str or not value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise LLRGeodeticTransportError(
            f"{label} must be a nonempty visible-ASCII token"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LLRGeodeticTransportError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class IERSEarthOrientationSample:
    mjd_utc: float
    polar_motion_x_arcseconds: float
    polar_motion_y_arcseconds: float
    ut1_minus_utc_seconds: float
    celestial_pole_x_milliarcseconds: float
    celestial_pole_y_milliarcseconds: float
    polar_motion_flag: str
    ut1_flag: str
    celestial_pole_flag: str

    def __post_init__(self) -> None:
        for label, value in (
            ("mjd_utc", self.mjd_utc),
            ("polar_motion_x_arcseconds", self.polar_motion_x_arcseconds),
            ("polar_motion_y_arcseconds", self.polar_motion_y_arcseconds),
            ("ut1_minus_utc_seconds", self.ut1_minus_utc_seconds),
            (
                "celestial_pole_x_milliarcseconds",
                self.celestial_pole_x_milliarcseconds,
            ),
            (
                "celestial_pole_y_milliarcseconds",
                self.celestial_pole_y_milliarcseconds,
            ),
        ):
            _finite_float(value, label)
        if self.mjd_utc < 30_000.0 or self.mjd_utc > 100_000.0:
            raise LLRGeodeticTransportError("EOP MJD is outside the supported bound")
        for label, value in (
            ("polar_motion_flag", self.polar_motion_flag),
            ("ut1_flag", self.ut1_flag),
            ("celestial_pole_flag", self.celestial_pole_flag),
        ):
            if value not in ("I", "P"):
                raise LLRGeodeticTransportError(f"{label} must be I or P")


@dataclass(frozen=True, slots=True)
class InterpolatedEarthOrientation:
    mjd_utc: float
    lower_mjd_utc: float
    upper_mjd_utc: float
    interpolation_fraction: float
    polar_motion_x_arcseconds: float
    polar_motion_y_arcseconds: float
    ut1_minus_utc_seconds: float
    celestial_pole_x_milliarcseconds: float
    celestial_pole_y_milliarcseconds: float
    source_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.mjd_utc,
            self.lower_mjd_utc,
            self.upper_mjd_utc,
            self.interpolation_fraction,
            self.polar_motion_x_arcseconds,
            self.polar_motion_y_arcseconds,
            self.ut1_minus_utc_seconds,
            self.celestial_pole_x_milliarcseconds,
            self.celestial_pole_y_milliarcseconds,
        ):
            _finite_float(value, "interpolated EOP value")
        if not (
            self.lower_mjd_utc
            <= self.mjd_utc
            <= self.upper_mjd_utc
            and 0.0 <= self.interpolation_fraction <= 1.0
        ):
            raise LLRGeodeticTransportError("EOP interpolation bracket is invalid")
        if type(self.source_flags) is not tuple or any(
            flag not in ("I", "P") for flag in self.source_flags
        ):
            raise LLRGeodeticTransportError("EOP source flags are invalid")


@dataclass(frozen=True, slots=True)
class IERSFinals2000A:
    samples: tuple[IERSEarthOrientationSample, ...]
    source_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        _visible_ascii(self.source_name, "EOP source_name")
        _sha256(self.source_sha256, "EOP source_sha256")
        if type(self.samples) is not tuple or len(self.samples) < 2:
            raise LLRGeodeticTransportError("EOP series requires at least two samples")
        if any(type(item) is not IERSEarthOrientationSample for item in self.samples):
            raise LLRGeodeticTransportError("EOP series contains the wrong type")
        epochs = tuple(item.mjd_utc for item in self.samples)
        if any(right <= left for left, right in zip(epochs, epochs[1:])):
            raise LLRGeodeticTransportError("EOP epochs must increase strictly")

    def interpolate(self, mjd_utc: float) -> InterpolatedEarthOrientation:
        _finite_float(mjd_utc, "requested EOP MJD")
        epochs = tuple(item.mjd_utc for item in self.samples)
        if mjd_utc < epochs[0] or mjd_utc > epochs[-1]:
            raise LLRGeodeticTransportError("EOP request would extrapolate")
        upper_index = bisect_right(epochs, mjd_utc)
        if upper_index == 0:
            upper_index = 1
        if upper_index == len(epochs):
            upper_index -= 1
        left = self.samples[upper_index - 1]
        right = self.samples[upper_index]
        fraction = (mjd_utc - left.mjd_utc) / (right.mjd_utc - left.mjd_utc)

        def blend(name: str) -> float:
            start = getattr(left, name)
            end = getattr(right, name)
            return float(start + fraction * (end - start))

        return InterpolatedEarthOrientation(
            mjd_utc=mjd_utc,
            lower_mjd_utc=left.mjd_utc,
            upper_mjd_utc=right.mjd_utc,
            interpolation_fraction=float(fraction),
            polar_motion_x_arcseconds=blend("polar_motion_x_arcseconds"),
            polar_motion_y_arcseconds=blend("polar_motion_y_arcseconds"),
            ut1_minus_utc_seconds=blend("ut1_minus_utc_seconds"),
            celestial_pole_x_milliarcseconds=blend(
                "celestial_pole_x_milliarcseconds"
            ),
            celestial_pole_y_milliarcseconds=blend(
                "celestial_pole_y_milliarcseconds"
            ),
            source_flags=tuple(
                sorted(
                    {
                        left.polar_motion_flag,
                        left.ut1_flag,
                        left.celestial_pole_flag,
                        right.polar_motion_flag,
                        right.ut1_flag,
                        right.celestial_pole_flag,
                    }
                )
            ),
        )


def parse_iers_finals2000a(payload: bytes, *, source_name: str) -> IERSFinals2000A:
    """Parse the Bulletin-A fields needed for IAU 2006/2000A transport."""

    if type(payload) is not bytes or not payload or len(payload) > MAXIMUM_EOP_BYTES:
        raise LLRGeodeticTransportError("EOP payload must be nonempty bounded bytes")
    _visible_ascii(source_name, "EOP source_name")
    if b"\x00" in payload:
        raise LLRGeodeticTransportError("EOP payload contains a NUL byte")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LLRGeodeticTransportError("EOP payload must be ASCII") from exc
    if len(lines) > MAXIMUM_EOP_ROWS:
        raise LLRGeodeticTransportError("EOP payload exceeds the row cap")
    samples: list[IERSEarthOrientationSample] = []
    for line in lines:
        if len(line) < 125:
            continue
        flags = (line[16:17], line[57:58], line[95:96])
        if any(flag == " " for flag in flags):
            # finals2000A retains partial predictions and date/MJD stubs after
            # one or more of the required products ends.
            continue
        if any(flag not in ("I", "P") for flag in flags):
            raise LLRGeodeticTransportError(
                "EOP payload contains a malformed complete Bulletin-A row"
            )
        try:
            sample = IERSEarthOrientationSample(
                mjd_utc=float(line[7:15]),
                polar_motion_x_arcseconds=float(line[18:27]),
                polar_motion_y_arcseconds=float(line[37:46]),
                ut1_minus_utc_seconds=float(line[58:68]),
                celestial_pole_x_milliarcseconds=float(line[97:106]),
                celestial_pole_y_milliarcseconds=float(line[116:125]),
                polar_motion_flag=flags[0],
                ut1_flag=flags[1],
                celestial_pole_flag=flags[2],
            )
        except (ValueError, LLRGeodeticTransportError) as exc:
            raise LLRGeodeticTransportError(
                "EOP payload contains a malformed complete Bulletin-A row"
            ) from exc
        samples.append(sample)
    if len(samples) < 2:
        raise LLRGeodeticTransportError(
            "EOP payload has fewer than two complete Bulletin-A rows"
        )
    return IERSFinals2000A(
        samples=tuple(samples),
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class GroundStation:
    station_name: str
    station_sod: str
    domes_number: str
    longitude_east_radians: float
    geodetic_latitude_radians: float
    ellipsoidal_height_metres: float
    mean_sea_level_height_metres: float
    coordinate_reference_epoch_utc: str | None
    source_document_date_utc: str
    stated_accuracy_metres: float
    coordinate_velocity_available: bool
    source_url: str
    source_sha256: str
    site_log_url: str
    site_log_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("station_name", self.station_name),
            ("station_sod", self.station_sod),
            ("domes_number", self.domes_number),
            ("source_document_date_utc", self.source_document_date_utc),
            ("source_url", self.source_url),
            ("site_log_url", self.site_log_url),
        ):
            _visible_ascii(value, label)
        if self.coordinate_reference_epoch_utc is not None:
            _visible_ascii(
                self.coordinate_reference_epoch_utc,
                "coordinate_reference_epoch_utc",
            )
        _sha256(self.source_sha256, "station source SHA-256")
        _sha256(self.site_log_sha256, "station site-log SHA-256")
        for label, value in (
            ("longitude_east_radians", self.longitude_east_radians),
            ("geodetic_latitude_radians", self.geodetic_latitude_radians),
            ("ellipsoidal_height_metres", self.ellipsoidal_height_metres),
            ("mean_sea_level_height_metres", self.mean_sea_level_height_metres),
            ("stated_accuracy_metres", self.stated_accuracy_metres),
        ):
            _finite_float(value, label)
        if not -math.pi <= self.longitude_east_radians <= 2.0 * math.pi:
            raise LLRGeodeticTransportError("station longitude is invalid")
        if not -math.pi / 2.0 <= self.geodetic_latitude_radians <= math.pi / 2.0:
            raise LLRGeodeticTransportError("station latitude is invalid")
        if self.stated_accuracy_metres <= 0.0:
            raise LLRGeodeticTransportError("station accuracy must be positive")
        if type(self.coordinate_velocity_available) is not bool:
            raise LLRGeodeticTransportError("station velocity flag must be a bool")


def apache_point_2005_station(
    *,
    source_sha256: str,
    site_log_sha256: str,
) -> GroundStation:
    """Return the published 2005 WGS84 APOL telescope-axis solution."""

    return GroundStation(
        station_name="APOL",
        station_sod="70459501",
        domes_number="49447S001",
        longitude_east_radians=math.radians(254.1795786),
        geodetic_latitude_radians=math.radians(32.78035394),
        ellipsoidal_height_metres=2786.62,
        mean_sea_level_height_metres=2788.0,
        coordinate_reference_epoch_utc=None,
        source_document_date_utc="2005-08-23",
        stated_accuracy_metres=0.1,
        coordinate_velocity_available=False,
        source_url="https://tmurphy.physics.ucsd.edu/apollo/doc/APO_coords.pdf",
        source_sha256=source_sha256,
        site_log_url="https://ilrs.gsfc.nasa.gov/network/stations/active/APOL_sitelog.html",
        site_log_sha256=site_log_sha256,
    )


def wgs84_cartesian_metres(station: GroundStation) -> np.ndarray:
    if type(station) is not GroundStation:
        raise LLRGeodeticTransportError("station has the wrong type")
    semimajor = 6_378_137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = 2.0 * flattening - flattening * flattening
    sine = math.sin(station.geodetic_latitude_radians)
    cosine = math.cos(station.geodetic_latitude_radians)
    prime_vertical = semimajor / math.sqrt(1.0 - eccentricity_squared * sine * sine)
    radius = prime_vertical + station.ellipsoidal_height_metres
    return np.ascontiguousarray(
        np.asarray(
            (
                radius * cosine * math.cos(station.longitude_east_radians),
                radius * cosine * math.sin(station.longitude_east_radians),
                (
                    prime_vertical * (1.0 - eccentricity_squared)
                    + station.ellipsoidal_height_metres
                )
                * sine,
            ),
            dtype=np.float64,
        )
    )


@dataclass(frozen=True, slots=True)
class LinearStationState:
    """Explicit empirical ECEF offset and velocity at a declared UTC MJD."""

    model_id: str
    reference_mjd_utc: float
    offset_ecef_metres: tuple[float, float, float]
    velocity_ecef_metres_per_year: tuple[float, float, float]
    source_sha256: str

    def __post_init__(self) -> None:
        _visible_ascii(self.model_id, "station-state model_id")
        _finite_float(self.reference_mjd_utc, "station-state reference MJD")
        _sha256(self.source_sha256, "station-state source SHA-256")
        if not 30_000.0 <= self.reference_mjd_utc <= 100_000.0:
            raise LLRGeodeticTransportError(
                "station-state reference MJD is outside the supported bound"
            )
        for label, vector in (
            ("offset", self.offset_ecef_metres),
            ("velocity", self.velocity_ecef_metres_per_year),
        ):
            if type(vector) is not tuple or len(vector) != 3 or any(
                type(value) is not float or not math.isfinite(value)
                for value in vector
            ):
                raise LLRGeodeticTransportError(
                    f"station-state {label} must be a finite float 3-tuple"
                )
        if math.sqrt(math.fsum(value * value for value in self.offset_ecef_metres)) > 100.0:
            raise LLRGeodeticTransportError("station-state offset exceeds 100 metres")
        if math.sqrt(
            math.fsum(
                value * value for value in self.velocity_ecef_metres_per_year
            )
        ) > 1.0:
            raise LLRGeodeticTransportError(
                "station-state velocity exceeds one metre per year"
            )

    def offset_at_mjd_metres(self, mjd_utc: float) -> np.ndarray:
        _finite_float(mjd_utc, "station-state evaluation MJD")
        years = (mjd_utc - self.reference_mjd_utc) / 365.25
        return np.ascontiguousarray(
            np.asarray(self.offset_ecef_metres, dtype=np.float64)
            + years
            * np.asarray(self.velocity_ecef_metres_per_year, dtype=np.float64)
        )


@dataclass(frozen=True, slots=True)
class LunarReflector:
    target_name: str
    frame_name: str
    position_km: tuple[float, float, float]
    source_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("target_name", self.target_name),
            ("frame_name", self.frame_name),
            ("source_url", self.source_url),
        ):
            _visible_ascii(value, label)
        if type(self.position_km) is not tuple or len(self.position_km) != 3:
            raise LLRGeodeticTransportError("reflector position must be an exact 3-tuple")
        for value in self.position_km:
            _finite_float(value, "reflector position")


_DE440_REFLECTOR_SOURCE = "https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf"
DE440_LUNAR_REFLECTORS: Mapping[str, LunarReflector] = MappingProxyType(
    {
        "apollo11": LunarReflector(
            "apollo11",
            "MOON_PA_DE440",
            (1591.967049, 690.698573, 21.004461),
            _DE440_REFLECTOR_SOURCE,
        ),
        "apollo14": LunarReflector(
            "apollo14",
            "MOON_PA_DE440",
            (1652.689369, -520.998431, -109.729869),
            _DE440_REFLECTOR_SOURCE,
        ),
        "apollo15": LunarReflector(
            "apollo15",
            "MOON_PA_DE440",
            (1554.678104, 98.094498, 765.005863),
            _DE440_REFLECTOR_SOURCE,
        ),
        "luna17": LunarReflector(
            "luna17",
            "MOON_PA_DE440",
            (1114.291452, -781.299273, 1076.059049),
            _DE440_REFLECTOR_SOURCE,
        ),
        "luna21": LunarReflector(
            "luna21",
            "MOON_PA_DE440",
            (1339.363598, 801.870995, 756.359260),
            _DE440_REFLECTOR_SOURCE,
        ),
    }
)
APOLLO_15_DE440_PA = DE440_LUNAR_REFLECTORS["apollo15"]


def de440_lunar_reflector(target_name: str) -> LunarReflector:
    _visible_ascii(target_name, "lunar reflector target name")
    try:
        return DE440_LUNAR_REFLECTORS[target_name.lower()]
    except KeyError as exc:
        raise LLRGeodeticTransportError(
            "target has no registered DE440 lunar-reflector coordinate"
        ) from exc


@dataclass(frozen=True, slots=True)
class LLRTimeScales:
    utc_jd: tuple[float, float]
    ut1_jd: tuple[float, float]
    tt_jd: tuple[float, float]
    tdb_jd: tuple[float, float]
    spice_tdb_et_seconds: float
    tdb_minus_tt_seconds: float
    eop: InterpolatedEarthOrientation

    def __post_init__(self) -> None:
        for pair in (self.utc_jd, self.ut1_jd, self.tt_jd, self.tdb_jd):
            if type(pair) is not tuple or len(pair) != 2 or any(
                type(value) is not float or not math.isfinite(value) for value in pair
            ):
                raise LLRGeodeticTransportError("time scale must be a finite JD pair")
        _finite_float(self.spice_tdb_et_seconds, "SPICE ET")
        _finite_float(self.tdb_minus_tt_seconds, "TDB-TT")
        if type(self.eop) is not InterpolatedEarthOrientation:
            raise LLRGeodeticTransportError("time scales require interpolated EOP")


def _ut_fraction(ut1_jd: tuple[float, float]) -> float:
    return float(((ut1_jd[0] - MJD_ZERO_JD) + ut1_jd[1]) % 1.0)


def _mjd(jd: tuple[float, float]) -> float:
    return float((jd[0] - MJD_ZERO_JD) + jd[1])


def _site_uv_km(station_position_metres: np.ndarray) -> tuple[float, float]:
    return (
        float(math.hypot(station_position_metres[0], station_position_metres[1]) / 1e3),
        float(station_position_metres[2] / 1e3),
    )


def utc_calendar_to_time_scales(
    year: int,
    month: int,
    day: int,
    seconds_of_day: float,
    *,
    station: GroundStation,
    eop_series: IERSFinals2000A,
    erfa_module: Any,
) -> LLRTimeScales:
    if any(type(value) is not int for value in (year, month, day)):
        raise LLRGeodeticTransportError("UTC calendar fields must be exact integers")
    _finite_float(seconds_of_day, "UTC seconds of day")
    if not 0.0 <= seconds_of_day < SECONDS_PER_DAY:
        raise LLRGeodeticTransportError("UTC seconds of day is outside [0, 86400)")
    hour = int(seconds_of_day // 3600.0)
    minute = int((seconds_of_day - 3600.0 * hour) // 60.0)
    second = float(seconds_of_day - 3600.0 * hour - 60.0 * minute)
    try:
        utc_raw = erfa_module.dtf2d("UTC", year, month, day, hour, minute, second)
        utc = (float(utc_raw[0]), float(utc_raw[1]))
        tai_raw = erfa_module.utctai(*utc)
        tt_raw = erfa_module.taitt(float(tai_raw[0]), float(tai_raw[1]))
        tt = (float(tt_raw[0]), float(tt_raw[1]))
        eop = eop_series.interpolate(_mjd(utc))
        ut1_raw = erfa_module.utcut1(*utc, eop.ut1_minus_utc_seconds)
        ut1 = (float(ut1_raw[0]), float(ut1_raw[1]))
        station_position = wgs84_cartesian_metres(station)
        u_km, v_km = _site_uv_km(station_position)
        dtr = float(
            erfa_module.dtdb(
                *tt,
                _ut_fraction(ut1),
                station.longitude_east_radians,
                u_km,
                v_km,
            )
        )
        tdb_raw = erfa_module.tttdb(*tt, dtr)
        tdb = (float(tdb_raw[0]), float(tdb_raw[1]))
    except Exception as exc:
        raise LLRGeodeticTransportError("ERFA UTC-to-TDB conversion failed") from exc
    et = float(math.fsum((tdb[0] - J2000_JD, tdb[1])) * SECONDS_PER_DAY)
    return LLRTimeScales(utc, ut1, tt, tdb, et, dtr, eop)


def crd_normal_point_time_scales(
    point: llr.CRDNormalPoint,
    *,
    station: GroundStation,
    eop_series: IERSFinals2000A,
    erfa_module: Any,
) -> LLRTimeScales:
    if type(point) is not llr.CRDNormalPoint:
        raise LLRGeodeticTransportError("point must be an exact CRDNormalPoint")
    epoch = point.epoch_utc
    return utc_calendar_to_time_scales(
        epoch.year,
        epoch.month,
        epoch.day,
        point.seconds_of_day,
        station=station,
        eop_series=eop_series,
        erfa_module=erfa_module,
    )


class DE440LLRGeometry:
    """Loaded-kernel DE440 geometry providers for the generic light-time solver."""

    __slots__ = (
        "eop_series",
        "erfa",
        "reflector",
        "ocean_loading_model",
        "spice",
        "station",
        "station_state",
        "apply_solid_earth_tide",
        "apply_solid_pole_tide",
        "station_position_metres",
        "u_km",
        "v_km",
    )

    def __init__(
        self,
        *,
        station: GroundStation,
        reflector: LunarReflector,
        eop_series: IERSFinals2000A,
        erfa_module: Any,
        spice_module: Any,
        station_state: LinearStationState | None = None,
        ocean_loading_model: ocean_loading.HARPOSStationModel | None = None,
        apply_solid_earth_tide: bool = True,
        apply_solid_pole_tide: bool = True,
    ) -> None:
        if type(station) is not GroundStation or type(reflector) is not LunarReflector:
            raise LLRGeodeticTransportError("geometry station or reflector type changed")
        if type(eop_series) is not IERSFinals2000A:
            raise LLRGeodeticTransportError("geometry EOP series type changed")
        if type(apply_solid_earth_tide) is not bool:
            raise LLRGeodeticTransportError("solid-Earth-tide flag must be a bool")
        if type(apply_solid_pole_tide) is not bool:
            raise LLRGeodeticTransportError("solid-pole-tide flag must be a bool")
        if station_state is not None and type(station_state) is not LinearStationState:
            raise LLRGeodeticTransportError("station state has the wrong type")
        if ocean_loading_model is not None and type(
            ocean_loading_model
        ) is not ocean_loading.HARPOSStationModel:
            raise LLRGeodeticTransportError("ocean-loading model has the wrong type")
        self.station = station
        self.reflector = reflector
        self.eop_series = eop_series
        self.erfa = erfa_module
        self.spice = spice_module
        self.station_state = station_state
        self.ocean_loading_model = ocean_loading_model
        self.apply_solid_earth_tide = apply_solid_earth_tide
        self.apply_solid_pole_tide = apply_solid_pole_tide
        self.station_position_metres = wgs84_cartesian_metres(station)
        if ocean_loading_model is not None:
            try:
                ocean_loading_model.distance_from_site_metres(
                    self.station_position_metres
                )
            except ocean_loading.OceanLoadingError as exc:
                raise LLRGeodeticTransportError(
                    "station lies outside the ocean-loading site"
                ) from exc
        self.u_km, self.v_km = _site_uv_km(self.station_position_metres)

    def _et_time_scales(self, et_seconds: float) -> LLRTimeScales:
        _finite_float(et_seconds, "SPICE ET")
        tdb = (J2000_JD, et_seconds / SECONDS_PER_DAY)
        dtr = 0.0
        try:
            for _ in range(4):
                tt_raw = self.erfa.tdbtt(*tdb, dtr)
                tt = (float(tt_raw[0]), float(tt_raw[1]))
                tai_raw = self.erfa.tttai(*tt)
                utc_raw = self.erfa.taiutc(float(tai_raw[0]), float(tai_raw[1]))
                utc = (float(utc_raw[0]), float(utc_raw[1]))
                eop = self.eop_series.interpolate(_mjd(utc))
                ut1_raw = self.erfa.utcut1(*utc, eop.ut1_minus_utc_seconds)
                ut1 = (float(ut1_raw[0]), float(ut1_raw[1]))
                dtr = float(
                    self.erfa.dtdb(
                        *tt,
                        _ut_fraction(ut1),
                        self.station.longitude_east_radians,
                        self.u_km,
                        self.v_km,
                    )
                )
        except Exception as exc:
            raise LLRGeodeticTransportError("ERFA TDB-to-UTC iteration failed") from exc
        return LLRTimeScales(utc, ut1, tt, tdb, et_seconds, dtr, eop)

    def _celestial_to_terrestrial(self, et_seconds: float) -> np.ndarray:
        scales = self._et_time_scales(et_seconds)
        eop = scales.eop
        try:
            x, y, s = self.erfa.xys06a(*scales.tt_jd)
            x = float(x) + (
                eop.celestial_pole_x_milliarcseconds
                * ARCSECOND_TO_RADIAN
                / 1.0e3
            )
            y = float(y) + (
                eop.celestial_pole_y_milliarcseconds
                * ARCSECOND_TO_RADIAN
                / 1.0e3
            )
            rc2i = self.erfa.c2ixys(x, y, float(s))
            era = self.erfa.era00(*scales.ut1_jd)
            sp = self.erfa.sp00(*scales.tt_jd)
            rpom = self.erfa.pom00(
                eop.polar_motion_x_arcseconds * ARCSECOND_TO_RADIAN,
                eop.polar_motion_y_arcseconds * ARCSECOND_TO_RADIAN,
                sp,
            )
            matrix = np.asarray(
                self.erfa.c2tcio(rc2i, era, rpom),
                dtype=np.float64,
            )
        except Exception as exc:
            raise LLRGeodeticTransportError("ERFA terrestrial rotation failed") from exc
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise LLRGeodeticTransportError("ERFA returned an invalid rotation")
        return matrix

    def body_position_km(self, body: str, et_seconds: float) -> np.ndarray:
        _visible_ascii(body, "SPICE body")
        _finite_float(et_seconds, "SPICE ET")
        try:
            position = self.spice.spkpos(
                body,
                et_seconds,
                "J2000",
                "NONE",
                "SOLAR SYSTEM BARYCENTER",
            )[0]
        except Exception as exc:
            raise LLRGeodeticTransportError("SPICE body query failed") from exc
        value = np.ascontiguousarray(np.asarray(position, dtype=np.float64))
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise LLRGeodeticTransportError("SPICE body position is invalid")
        return value

    def station_position_km(self, et_seconds: float) -> np.ndarray:
        rotation = self._celestial_to_terrestrial(et_seconds)
        station_ecef_metres = self.reference_station_position_metres(et_seconds)
        if self.apply_solid_earth_tide:
            station_ecef_metres = (
                station_ecef_metres
                + self.solid_earth_tide_displacement_metres(et_seconds)
            )
        if self.ocean_loading_model is not None:
            station_ecef_metres = (
                station_ecef_metres
                + self.ocean_loading_displacement_metres(et_seconds)
            )
        if self.apply_solid_pole_tide:
            station_ecef_metres = (
                station_ecef_metres
                + self.solid_pole_tide_displacement_metres(et_seconds)
            )
        geocentric = np.asarray(
            rotation.T @ station_ecef_metres,
            dtype=np.float64,
        ) / 1.0e3
        return np.ascontiguousarray(
            self.body_position_km("EARTH", et_seconds) + geocentric
        )

    def reference_station_position_metres(self, et_seconds: float) -> np.ndarray:
        """Return static WGS84 plus any explicit empirical linear state."""

        _finite_float(et_seconds, "SPICE ET")
        position = self.station_position_metres
        if self.station_state is not None:
            scales = self._et_time_scales(et_seconds)
            position = position + self.station_state.offset_at_mjd_metres(
                scales.eop.mjd_utc
            )
        return np.ascontiguousarray(position, dtype=np.float64)

    def solid_earth_tide_displacement_metres(
        self,
        et_seconds: float,
    ) -> np.ndarray:
        """Return the IERS-compatible ECEF displacement used by this geometry."""

        _finite_float(et_seconds, "SPICE ET")
        scales = self._et_time_scales(et_seconds)
        try:
            year, month, day, hmsf = self.erfa.d2dtf(
                "UTC",
                9,
                *scales.utc_jd,
            )
            seconds_of_day = float(
                int(hmsf["h"]) * 3600
                + int(hmsf["m"]) * 60
                + int(hmsf["s"])
                + int(hmsf["f"]) * 1.0e-9
            )
            rotation = self._celestial_to_terrestrial(et_seconds)
            earth = self.body_position_km("EARTH", et_seconds)
            sun_ecef_metres = rotation @ (
                self.body_position_km("SUN", et_seconds) - earth
            ) * 1.0e3
            moon_ecef_metres = rotation @ (
                self.body_position_km("MOON", et_seconds) - earth
            ) * 1.0e3
            displacement = solid_tide.solid_earth_tide_displacement_metres(
                self.reference_station_position_metres(et_seconds),
                sun_ecef_metres,
                moon_ecef_metres,
                year=int(year),
                month=int(month),
                day=int(day),
                seconds_of_day=seconds_of_day,
                erfa_module=self.erfa,
            )
        except solid_tide.SolidEarthTideError as exc:
            raise LLRGeodeticTransportError(
                "solid-Earth-tide station displacement failed"
            ) from exc
        return displacement

    def ocean_loading_displacement_metres(self, et_seconds: float) -> np.ndarray:
        """Return the source-bound HARPOS displacement used by this geometry."""

        _finite_float(et_seconds, "SPICE ET")
        if self.ocean_loading_model is None:
            raise LLRGeodeticTransportError("ocean-loading model is absent")
        scales = self._et_time_scales(et_seconds)
        station_position = self.reference_station_position_metres(et_seconds)
        try:
            return self.ocean_loading_model.displacement_ecef_metres(
                scales.tt_jd,
                position_ecef_metres=station_position,
            )
        except ocean_loading.OceanLoadingError as exc:
            raise LLRGeodeticTransportError(
                "ocean-loading station displacement failed"
            ) from exc

    def solid_pole_tide_displacement_metres(self, et_seconds: float) -> np.ndarray:
        """Return the observed-EOP solid pole-tide displacement in ECEF."""

        _finite_float(et_seconds, "SPICE ET")
        scales = self._et_time_scales(et_seconds)
        eop = scales.eop
        try:
            return solid_pole_tide.solid_pole_tide_displacement_metres(
                self.reference_station_position_metres(et_seconds),
                mjd_utc=eop.mjd_utc,
                polar_motion_x_arcseconds=eop.polar_motion_x_arcseconds,
                polar_motion_y_arcseconds=eop.polar_motion_y_arcseconds,
            )
        except solid_pole_tide.SolidPoleTideError as exc:
            raise LLRGeodeticTransportError(
                "solid-pole-tide station displacement failed"
            ) from exc

    def reflector_position_km(self, et_seconds: float) -> np.ndarray:
        try:
            rotation = np.asarray(
                self.spice.pxform(self.reflector.frame_name, "J2000", et_seconds),
                dtype=np.float64,
            )
        except Exception as exc:
            raise LLRGeodeticTransportError("SPICE lunar-frame rotation failed") from exc
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise LLRGeodeticTransportError("SPICE lunar-frame rotation is invalid")
        offset = rotation @ np.asarray(self.reflector.position_km, dtype=np.float64)
        return np.ascontiguousarray(
            self.body_position_km("MOON", et_seconds) + offset
        )

    def elevation_degrees(
        self,
        station_et_seconds: float,
        reflector_et_seconds: float,
    ) -> float:
        ray = self.reflector_position_km(
            reflector_et_seconds
        ) - self.station_position_km(station_et_seconds)
        norm = float(np.linalg.norm(ray))
        if not math.isfinite(norm) or norm <= 0.0:
            raise LLRGeodeticTransportError("elevation ray is invalid")
        terrestrial_ray = self._celestial_to_terrestrial(station_et_seconds) @ ray
        terrestrial_ray /= norm
        latitude = self.station.geodetic_latitude_radians
        longitude = self.station.longitude_east_radians
        up = np.asarray(
            (
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ),
            dtype=np.float64,
        )
        sine = min(1.0, max(-1.0, float(np.dot(terrestrial_ray, up))))
        return float(math.degrees(math.asin(sine)))


def station_ecef_one_way_residual_derivative(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
) -> np.ndarray:
    """Return d(observed-predicted one-way range)/d(station ECEF offset).

    This is the preregistered geometric design row: one half the sum of the
    transmit and receive station-to-reflector unit vectors after rotation into
    the terrestrial frame.  It intentionally does not differentiate the
    incomplete propagation screens or site-displacement models.
    """

    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    bounce = geometry.reflector_position_km(light_time.bounce_epoch_seconds)
    transmit = geometry.station_position_km(light_time.transmit_epoch_seconds)
    receive = geometry.station_position_km(light_time.receive_epoch_seconds)
    uplink = bounce - transmit
    downlink = bounce - receive
    uplink_norm = float(np.linalg.norm(uplink))
    downlink_norm = float(np.linalg.norm(downlink))
    if (
        not math.isfinite(uplink_norm)
        or not math.isfinite(downlink_norm)
        or uplink_norm <= 0.0
        or downlink_norm <= 0.0
    ):
        raise LLRGeodeticTransportError("station design geometry is invalid")
    derivative = 0.5 * (
        geometry._celestial_to_terrestrial(light_time.transmit_epoch_seconds)
        @ (uplink / uplink_norm)
        + geometry._celestial_to_terrestrial(light_time.receive_epoch_seconds)
        @ (downlink / downlink_norm)
    )
    derivative = np.ascontiguousarray(derivative, dtype=np.float64)
    norm = float(np.linalg.norm(derivative))
    if not np.all(np.isfinite(derivative)) or norm <= 0.0 or norm > 1.0000000001:
        raise LLRGeodeticTransportError("station design derivative is invalid")
    return derivative


@dataclass(frozen=True, slots=True)
class FCULZenithDelay:
    hydrostatic_metres: float
    nonhydrostatic_metres: float
    total_metres: float

    def __post_init__(self) -> None:
        for value in (
            self.hydrostatic_metres,
            self.nonhydrostatic_metres,
            self.total_metres,
        ):
            _finite_float(value, "FCUL zenith delay")
        if min(
            self.hydrostatic_metres,
            self.nonhydrostatic_metres,
            self.total_metres,
        ) < 0.0:
            raise LLRGeodeticTransportError("FCUL zenith delays must be nonnegative")
        if not math.isclose(
            self.total_metres,
            self.hydrostatic_metres + self.nonhydrostatic_metres,
            rel_tol=0.0,
            abs_tol=2.0e-15,
        ):
            raise LLRGeodeticTransportError("FCUL zenith components do not close")


def buck_water_vapor_pressure_hpa(
    temperature_kelvin: float,
    relative_humidity_percent: float,
) -> float:
    _finite_float(temperature_kelvin, "temperature")
    _finite_float(relative_humidity_percent, "relative humidity")
    if not 180.0 <= temperature_kelvin <= 330.0:
        raise LLRGeodeticTransportError("temperature is outside the screening bound")
    if not 0.0 <= relative_humidity_percent <= 100.0:
        raise LLRGeodeticTransportError("relative humidity is outside [0, 100]")
    celsius = temperature_kelvin - 273.15
    saturation = 6.1121 * math.exp(
        (18.678 - celsius / 234.5) * (celsius / (257.14 + celsius))
    )
    return float(relative_humidity_percent * saturation / 100.0)


def fcul_zenith_delay_metres(
    *,
    latitude_degrees: float,
    ellipsoidal_height_metres: float,
    pressure_hpa: float,
    water_vapor_pressure_hpa: float,
    wavelength_micrometres: float,
) -> FCULZenithDelay:
    for label, value in (
        ("latitude_degrees", latitude_degrees),
        ("ellipsoidal_height_metres", ellipsoidal_height_metres),
        ("pressure_hpa", pressure_hpa),
        ("water_vapor_pressure_hpa", water_vapor_pressure_hpa),
        ("wavelength_micrometres", wavelength_micrometres),
    ):
        _finite_float(value, label)
    if not -90.0 <= latitude_degrees <= 90.0:
        raise LLRGeodeticTransportError("latitude is outside [-90, 90]")
    if pressure_hpa <= 0.0 or water_vapor_pressure_hpa < 0.0:
        raise LLRGeodeticTransportError("atmospheric pressure is invalid")
    if not 0.3 <= wavelength_micrometres <= 1.5:
        raise LLRGeodeticTransportError("wavelength is outside the optical bound")
    sigma = 1.0 / wavelength_micrometres
    gravity = 1.0 - 0.00266 * math.cos(
        2.0 * math.radians(latitude_degrees)
    ) - 0.00028e-3 * ellipsoidal_height_metres
    co2_correction = 1.0 + 0.534e-6 * (375.0 - 450.0)
    k0, k1, k2, k3 = 238.0185, 19990.975, 57.362, 579.55174
    hydrostatic_dispersion = 0.01 * co2_correction * (
        k1 * (k0 + sigma * sigma) / (k0 - sigma * sigma) ** 2
        + k3 * (k2 + sigma * sigma) / (k2 - sigma * sigma) ** 2
    )
    nonhydrostatic_dispersion = 0.003101 * (
        295.235
        + 3.0 * 2.6422 * sigma**2
        + 5.0 * -0.032380 * sigma**4
        + 7.0 * 0.004028 * sigma**6
    )
    hydrostatic = (
        2.416579e-3 * hydrostatic_dispersion * pressure_hpa / gravity
    )
    nonhydrostatic = (
        1.0e-4
        * (5.316 * nonhydrostatic_dispersion - 3.759 * hydrostatic_dispersion)
        * water_vapor_pressure_hpa
        / gravity
    )
    return FCULZenithDelay(
        hydrostatic_metres=float(hydrostatic),
        nonhydrostatic_metres=float(nonhydrostatic),
        total_metres=float(hydrostatic + nonhydrostatic),
    )


def fcul_a_mapping(
    *,
    latitude_degrees: float,
    mean_sea_level_height_metres: float,
    temperature_kelvin: float,
    elevation_degrees: float,
) -> float:
    for label, value in (
        ("latitude_degrees", latitude_degrees),
        ("mean_sea_level_height_metres", mean_sea_level_height_metres),
        ("temperature_kelvin", temperature_kelvin),
        ("elevation_degrees", elevation_degrees),
    ):
        _finite_float(value, label)
    if not MINIMUM_FCULA_ELEVATION_DEGREES <= elevation_degrees <= 90.0:
        raise LLRGeodeticTransportError("FCULa elevation is outside [3, 90] degrees")
    temperature_celsius = temperature_kelvin - 273.15
    cosine_latitude = math.cos(math.radians(latitude_degrees))
    a1 = (
        0.121008e-2
        + 0.17295e-5 * temperature_celsius
        + 0.3191e-4 * cosine_latitude
        - 0.18478e-7 * mean_sea_level_height_metres
    )
    a2 = (
        0.304965e-2
        + 0.2346e-5 * temperature_celsius
        - 0.1035e-3 * cosine_latitude
        - 0.1856e-7 * mean_sea_level_height_metres
    )
    a3 = (
        0.68777e-1
        + 0.1972e-4 * temperature_celsius
        - 0.3458e-2 * cosine_latitude
        + 0.1060e-6 * mean_sea_level_height_metres
    )
    sine = math.sin(math.radians(elevation_degrees))
    zenith = 1.0 + a1 / (1.0 + a2 / (1.0 + a3))
    return float(zenith / (sine + a1 / (sine + a2 / (sine + a3))))


@dataclass(frozen=True, slots=True)
class AtmosphericDelayScreen:
    model_id: str
    uplink_elevation_degrees: float
    downlink_elevation_degrees: float
    zenith_delay_metres: float
    uplink_delay_metres: float
    downlink_delay_metres: float
    round_trip_seconds: float

    def __post_init__(self) -> None:
        if self.model_id != ATMOSPHERE_MODEL_ID:
            raise LLRGeodeticTransportError("atmosphere model id changed")
        for value in (
            self.uplink_elevation_degrees,
            self.downlink_elevation_degrees,
            self.zenith_delay_metres,
            self.uplink_delay_metres,
            self.downlink_delay_metres,
            self.round_trip_seconds,
        ):
            _finite_float(value, "atmosphere result")
        if min(
            self.zenith_delay_metres,
            self.uplink_delay_metres,
            self.downlink_delay_metres,
            self.round_trip_seconds,
        ) < 0.0:
            raise LLRGeodeticTransportError("atmosphere delays must be nonnegative")
        if not math.isclose(
            self.round_trip_seconds,
            (self.uplink_delay_metres + self.downlink_delay_metres)
            / SPEED_OF_LIGHT_M_S,
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise LLRGeodeticTransportError("atmosphere delay budget does not close")


def atmospheric_delay_screen(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
    meteorology: llr.CRDMeteorology,
    *,
    wavelength_nm: float,
) -> AtmosphericDelayScreen:
    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    if type(meteorology) is not llr.CRDMeteorology:
        raise LLRGeodeticTransportError("meteorology has the wrong type")
    _finite_float(wavelength_nm, "laser wavelength")
    uplink_elevation = geometry.elevation_degrees(
        light_time.transmit_epoch_seconds,
        light_time.bounce_epoch_seconds,
    )
    downlink_elevation = geometry.elevation_degrees(
        light_time.receive_epoch_seconds,
        light_time.bounce_epoch_seconds,
    )
    latitude_degrees = math.degrees(geometry.station.geodetic_latitude_radians)
    vapor = buck_water_vapor_pressure_hpa(
        meteorology.temperature_kelvin,
        meteorology.relative_humidity_percent,
    )
    zenith = fcul_zenith_delay_metres(
        latitude_degrees=latitude_degrees,
        ellipsoidal_height_metres=geometry.station.ellipsoidal_height_metres,
        pressure_hpa=meteorology.pressure_millibar,
        water_vapor_pressure_hpa=vapor,
        wavelength_micrometres=wavelength_nm / 1.0e3,
    )
    uplink = zenith.total_metres * fcul_a_mapping(
        latitude_degrees=latitude_degrees,
        mean_sea_level_height_metres=geometry.station.mean_sea_level_height_metres,
        temperature_kelvin=meteorology.temperature_kelvin,
        elevation_degrees=uplink_elevation,
    )
    downlink = zenith.total_metres * fcul_a_mapping(
        latitude_degrees=latitude_degrees,
        mean_sea_level_height_metres=geometry.station.mean_sea_level_height_metres,
        temperature_kelvin=meteorology.temperature_kelvin,
        elevation_degrees=downlink_elevation,
    )
    return AtmosphericDelayScreen(
        model_id=ATMOSPHERE_MODEL_ID,
        uplink_elevation_degrees=uplink_elevation,
        downlink_elevation_degrees=downlink_elevation,
        zenith_delay_metres=zenith.total_metres,
        uplink_delay_metres=float(uplink),
        downlink_delay_metres=float(downlink),
        round_trip_seconds=float((uplink + downlink) / SPEED_OF_LIGHT_M_S),
    )


@dataclass(frozen=True, slots=True)
class RelativisticDelayScreen:
    model_id: str
    body_ids: tuple[str, ...]
    uplink_seconds_by_body: tuple[float, ...]
    downlink_seconds_by_body: tuple[float, ...]
    round_trip_seconds: float

    def __post_init__(self) -> None:
        if self.model_id != RELATIVITY_MODEL_ID:
            raise LLRGeodeticTransportError("relativity model id changed")
        if self.body_ids != tuple(DE440_POINT_MASS_GM_KM3_S2):
            raise LLRGeodeticTransportError("relativistic body roster changed")
        for values in (self.uplink_seconds_by_body, self.downlink_seconds_by_body):
            if type(values) is not tuple or len(values) != len(self.body_ids):
                raise LLRGeodeticTransportError("relativistic delay roster changed")
            for value in values:
                _finite_float(value, "relativistic delay")
                if value < 0.0:
                    raise LLRGeodeticTransportError(
                        "relativistic delay must be nonnegative"
                    )
        _finite_float(self.round_trip_seconds, "relativistic round-trip delay")
        if not math.isclose(
            self.round_trip_seconds,
            math.fsum((*self.uplink_seconds_by_body, *self.downlink_seconds_by_body)),
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise LLRGeodeticTransportError("relativistic delay budget does not close")


@dataclass(frozen=True, slots=True)
class SolidEarthTideScreen:
    model_id: str
    transmit_displacement_ecef_metres: tuple[float, float, float]
    receive_displacement_ecef_metres: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.model_id != solid_tide.MODEL_ID:
            raise LLRGeodeticTransportError("solid-Earth-tide model id changed")
        for vector in (
            self.transmit_displacement_ecef_metres,
            self.receive_displacement_ecef_metres,
        ):
            if type(vector) is not tuple or len(vector) != 3 or any(
                type(value) is not float or not math.isfinite(value)
                for value in vector
            ):
                raise LLRGeodeticTransportError(
                    "solid-Earth-tide displacement must be a finite float 3-tuple"
                )
            if math.sqrt(math.fsum(value * value for value in vector)) > 1.0:
                raise LLRGeodeticTransportError(
                    "solid-Earth-tide displacement exceeds the physical bound"
                )


@dataclass(frozen=True, slots=True)
class OceanLoadingScreen:
    model_id: str
    source_sha256: str
    site_name: str
    site_distance_metres: float
    harmonic_count: int
    transmit_displacement_ecef_metres: tuple[float, float, float]
    receive_displacement_ecef_metres: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.model_id != ocean_loading.MODEL_ID:
            raise LLRGeodeticTransportError("ocean-loading model id changed")
        _sha256(self.source_sha256, "ocean-loading source SHA-256")
        _visible_ascii(self.site_name, "ocean-loading site name")
        _finite_float(self.site_distance_metres, "ocean-loading site distance")
        if self.site_distance_metres < 0.0:
            raise LLRGeodeticTransportError(
                "ocean-loading site distance must be nonnegative"
            )
        if type(self.harmonic_count) is not int or self.harmonic_count <= 0:
            raise LLRGeodeticTransportError(
                "ocean-loading harmonic count must be positive"
            )
        for vector in (
            self.transmit_displacement_ecef_metres,
            self.receive_displacement_ecef_metres,
        ):
            if type(vector) is not tuple or len(vector) != 3 or any(
                type(value) is not float or not math.isfinite(value)
                for value in vector
            ):
                raise LLRGeodeticTransportError(
                    "ocean-loading displacement must be a finite float 3-tuple"
                )
            if math.sqrt(math.fsum(value * value for value in vector)) > 1.0:
                raise LLRGeodeticTransportError(
                    "ocean-loading displacement exceeds the physical bound"
                )


@dataclass(frozen=True, slots=True)
class SolidPoleTideScreen:
    model_id: str
    transmit_displacement_ecef_metres: tuple[float, float, float]
    receive_displacement_ecef_metres: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.model_id != solid_pole_tide.MODEL_ID:
            raise LLRGeodeticTransportError("solid-pole-tide model id changed")
        for vector in (
            self.transmit_displacement_ecef_metres,
            self.receive_displacement_ecef_metres,
        ):
            if type(vector) is not tuple or len(vector) != 3 or any(
                type(value) is not float or not math.isfinite(value)
                for value in vector
            ):
                raise LLRGeodeticTransportError(
                    "solid-pole-tide displacement must be a finite float 3-tuple"
                )
            if math.sqrt(math.fsum(value * value for value in vector)) > 0.1:
                raise LLRGeodeticTransportError(
                    "solid-pole-tide displacement exceeds the physical bound"
                )


def ocean_loading_screen(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
) -> OceanLoadingScreen:
    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    model = geometry.ocean_loading_model
    if model is None:
        raise LLRGeodeticTransportError(
            "physical screen requires a source-bound ocean-loading model"
        )
    transmit = geometry.ocean_loading_displacement_metres(
        light_time.transmit_epoch_seconds
    )
    receive = geometry.ocean_loading_displacement_metres(
        light_time.receive_epoch_seconds
    )
    distance = model.distance_from_site_metres(geometry.station_position_metres)
    return OceanLoadingScreen(
        model_id=ocean_loading.MODEL_ID,
        source_sha256=model.source_sha256,
        site_name=model.site.name,
        site_distance_metres=distance,
        harmonic_count=len(model.harmonics),
        transmit_displacement_ecef_metres=tuple(float(value) for value in transmit),
        receive_displacement_ecef_metres=tuple(float(value) for value in receive),
    )


def solid_earth_tide_screen(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
) -> SolidEarthTideScreen:
    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    if not geometry.apply_solid_earth_tide:
        raise LLRGeodeticTransportError(
            "solid-Earth-tide screen requires displacement-enabled geometry"
        )
    transmit = geometry.solid_earth_tide_displacement_metres(
        light_time.transmit_epoch_seconds
    )
    receive = geometry.solid_earth_tide_displacement_metres(
        light_time.receive_epoch_seconds
    )
    return SolidEarthTideScreen(
        model_id=solid_tide.MODEL_ID,
        transmit_displacement_ecef_metres=tuple(float(value) for value in transmit),
        receive_displacement_ecef_metres=tuple(float(value) for value in receive),
    )


def solid_pole_tide_screen(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
) -> SolidPoleTideScreen:
    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    if not geometry.apply_solid_pole_tide:
        raise LLRGeodeticTransportError(
            "solid-pole-tide screen requires displacement-enabled geometry"
        )
    transmit = geometry.solid_pole_tide_displacement_metres(
        light_time.transmit_epoch_seconds
    )
    receive = geometry.solid_pole_tide_displacement_metres(
        light_time.receive_epoch_seconds
    )
    return SolidPoleTideScreen(
        model_id=solid_pole_tide.MODEL_ID,
        transmit_displacement_ecef_metres=tuple(float(value) for value in transmit),
        receive_displacement_ecef_metres=tuple(float(value) for value in receive),
    )


def _shapiro_leg_seconds(
    geometry: DE440LLRGeometry,
    position_1_km: np.ndarray,
    epoch_1_seconds: float,
    position_2_km: np.ndarray,
    epoch_2_seconds: float,
) -> tuple[float, ...]:
    path = float(np.linalg.norm(position_2_km - position_1_km))
    midpoint = 0.5 * (epoch_1_seconds + epoch_2_seconds)
    values = []
    for body, gm in DE440_POINT_MASS_GM_KM3_S2.items():
        mass_position = geometry.body_position_km(body, midpoint)
        radius_1 = float(np.linalg.norm(position_1_km - mass_position))
        radius_2 = float(np.linalg.norm(position_2_km - mass_position))
        numerator = radius_1 + radius_2 + path
        denominator = radius_1 + radius_2 - path
        if denominator <= 0.0 or numerator <= denominator:
            raise LLRGeodeticTransportError("Shapiro logarithm geometry is invalid")
        values.append(
            float(
                2.0
                * gm
                / llr.SPEED_OF_LIGHT_KM_S**3
                * math.log(numerator / denominator)
            )
        )
    return tuple(values)


def relativistic_delay_screen(
    geometry: DE440LLRGeometry,
    light_time: llr.LLRLightTimeResult,
) -> RelativisticDelayScreen:
    if type(geometry) is not DE440LLRGeometry:
        raise LLRGeodeticTransportError("geometry has the wrong type")
    if type(light_time) is not llr.LLRLightTimeResult:
        raise LLRGeodeticTransportError("light_time has the wrong type")
    transmit = geometry.station_position_km(light_time.transmit_epoch_seconds)
    bounce = geometry.reflector_position_km(light_time.bounce_epoch_seconds)
    receive = geometry.station_position_km(light_time.receive_epoch_seconds)
    uplink = _shapiro_leg_seconds(
        geometry,
        transmit,
        light_time.transmit_epoch_seconds,
        bounce,
        light_time.bounce_epoch_seconds,
    )
    downlink = _shapiro_leg_seconds(
        geometry,
        bounce,
        light_time.bounce_epoch_seconds,
        receive,
        light_time.receive_epoch_seconds,
    )
    return RelativisticDelayScreen(
        model_id=RELATIVITY_MODEL_ID,
        body_ids=tuple(DE440_POINT_MASS_GM_KM3_S2),
        uplink_seconds_by_body=uplink,
        downlink_seconds_by_body=downlink,
        round_trip_seconds=float(math.fsum((*uplink, *downlink))),
    )


@dataclass(frozen=True, slots=True)
class LLRPhysicalScreenResult:
    model_id: str
    time_scales: LLRTimeScales
    geometric: llr.LLRLightTimeResult
    solid_earth_tide: SolidEarthTideScreen
    ocean_loading: OceanLoadingScreen
    solid_pole_tide: SolidPoleTideScreen
    atmosphere: AtmosphericDelayScreen
    relativity: RelativisticDelayScreen
    corrected: llr.LLRLightTimeResult
    missing_components: tuple[str, ...]
    claims: Mapping[str, object] = CLAIMS

    def __post_init__(self) -> None:
        if self.model_id != TRANSPORT_MODEL_ID:
            raise LLRGeodeticTransportError("physical screen model id changed")
        if type(self.time_scales) is not LLRTimeScales:
            raise LLRGeodeticTransportError("physical screen time type changed")
        if type(self.solid_earth_tide) is not SolidEarthTideScreen:
            raise LLRGeodeticTransportError(
                "physical station-displacement type changed"
            )
        if type(self.ocean_loading) is not OceanLoadingScreen:
            raise LLRGeodeticTransportError(
                "physical ocean-loading type changed"
            )
        if type(self.solid_pole_tide) is not SolidPoleTideScreen:
            raise LLRGeodeticTransportError(
                "physical solid-pole-tide type changed"
            )
        if type(self.geometric) is not llr.LLRLightTimeResult or type(
            self.corrected
        ) is not llr.LLRLightTimeResult:
            raise LLRGeodeticTransportError("physical screen result type changed")
        if type(self.atmosphere) is not AtmosphericDelayScreen or type(
            self.relativity
        ) is not RelativisticDelayScreen:
            raise LLRGeodeticTransportError("physical correction type changed")
        if self.corrected.corrections_complete:
            raise LLRGeodeticTransportError("incomplete reduction was marked complete")
        expected_correction = (
            self.atmosphere.round_trip_seconds + self.relativity.round_trip_seconds
        )
        if not math.isclose(
            self.corrected.applied_correction_seconds,
            expected_correction,
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise LLRGeodeticTransportError("physical correction budget does not close")
        if self.geometric.applied_correction_seconds != 0.0:
            raise LLRGeodeticTransportError("geometric result contains a correction")
        if self.missing_components != MISSING_REDUCTION_COMPONENTS:
            raise LLRGeodeticTransportError("missing-component roster changed")
        if self.claims is not CLAIMS:
            raise LLRGeodeticTransportError("physical screen claims changed")


def evaluate_normal_point_screen(
    point: llr.CRDNormalPoint,
    session: llr.CRDSessionHeader,
    meteorology: llr.CRDMeteorology,
    *,
    wavelength_nm: float,
    geometry: DE440LLRGeometry,
    erfa_module: Any,
) -> LLRPhysicalScreenResult:
    if session.tropospheric_refraction_applied:
        raise LLRGeodeticTransportError(
            "the physical screen requires an uncorrected CRD atmosphere flag"
        )
    scales = crd_normal_point_time_scales(
        point,
        station=geometry.station,
        eop_series=geometry.eop_series,
        erfa_module=erfa_module,
    )
    geometric = llr.solve_two_way_llr_normal_point(
        point,
        session,
        event_epoch_seconds=scales.spice_tdb_et_seconds,
        station_position_km=geometry.station_position_km,
        reflector_position_km=geometry.reflector_position_km,
        correction_budget=llr.LLRCorrectionBudget(0.0, 0.0, 0.0, False, ()),
        provider_model_id=TRANSPORT_MODEL_ID,
    )
    tide = solid_earth_tide_screen(geometry, geometric)
    ocean = ocean_loading_screen(geometry, geometric)
    pole = solid_pole_tide_screen(geometry, geometric)
    atmosphere = atmospheric_delay_screen(
        geometry,
        geometric,
        meteorology,
        wavelength_nm=wavelength_nm,
    )
    relativity = relativistic_delay_screen(geometry, geometric)
    corrected = llr.solve_two_way_llr_normal_point(
        point,
        session,
        event_epoch_seconds=scales.spice_tdb_et_seconds,
        station_position_km=geometry.station_position_km,
        reflector_position_km=geometry.reflector_position_km,
        correction_budget=llr.LLRCorrectionBudget(
            atmosphere.round_trip_seconds,
            relativity.round_trip_seconds,
            0.0,
            False,
            (ATMOSPHERE_MODEL_ID, RELATIVITY_MODEL_ID),
        ),
        provider_model_id=TRANSPORT_MODEL_ID,
    )
    return LLRPhysicalScreenResult(
        model_id=TRANSPORT_MODEL_ID,
        time_scales=scales,
        geometric=geometric,
        solid_earth_tide=tide,
        ocean_loading=ocean,
        solid_pole_tide=pole,
        atmosphere=atmosphere,
        relativity=relativity,
        corrected=corrected,
        missing_components=MISSING_REDUCTION_COMPONENTS,
    )


__all__ = [
    "APOLLO_15_DE440_PA",
    "ATMOSPHERE_MODEL_ID",
    "AtmosphericDelayScreen",
    "CLAIMS",
    "DE440LLRGeometry",
    "DE440_POINT_MASS_GM_KM3_S2",
    "DE440_LUNAR_REFLECTORS",
    "FCULZenithDelay",
    "GroundStation",
    "IERSEarthOrientationSample",
    "IERSFinals2000A",
    "InterpolatedEarthOrientation",
    "LLRGeodeticTransportError",
    "LLRPhysicalScreenResult",
    "LLRTimeScales",
    "LinearStationState",
    "LunarReflector",
    "MISSING_REDUCTION_COMPONENTS",
    "OceanLoadingScreen",
    "RELATIVITY_MODEL_ID",
    "RelativisticDelayScreen",
    "SolidEarthTideScreen",
    "SolidPoleTideScreen",
    "TRANSPORT_MODEL_ID",
    "apache_point_2005_station",
    "atmospheric_delay_screen",
    "buck_water_vapor_pressure_hpa",
    "crd_normal_point_time_scales",
    "de440_lunar_reflector",
    "evaluate_normal_point_screen",
    "fcul_a_mapping",
    "fcul_zenith_delay_metres",
    "parse_iers_finals2000a",
    "ocean_loading_screen",
    "relativistic_delay_screen",
    "solid_earth_tide_screen",
    "solid_pole_tide_screen",
    "station_ecef_one_way_residual_derivative",
    "utc_calendar_to_time_scales",
    "wgs84_cartesian_metres",
]
