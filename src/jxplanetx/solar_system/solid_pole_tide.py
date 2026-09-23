"""IERS working-version solid-Earth pole-tide station displacement.

This is an independent implementation of the compact equations in the
updated IERS Conventions Chapter 7 working version (2018), equations 7.24 and
7.25.  It is not IERS software and is not endorsed by IERS.  The working
version replaces the older piecewise mean-pole model with the secular pole.

The ocean pole tide is a separate loading correction and is deliberately not
implemented here.
"""

from __future__ import annotations

import math

import numpy as np


MODEL_ID = "station-displacement.solid-pole-tide.iers-working-2018.v1"
IERS_CHAPTER_7_WORKING_URL = (
    "https://iers-conventions.obspm.fr/content/chapter7/icc7.pdf"
)
J2000_MJD = 51_544.5
DAYS_PER_JULIAN_YEAR = 365.25
MILLIMETRES_PER_METRE = 1_000.0


class SolidPoleTideError(ValueError):
    """The solid-Earth pole-tide contract was not satisfied."""


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SolidPoleTideError(f"{label} must be a finite built-in float")
    return value


def _vector3(value: object, label: str) -> np.ndarray:
    try:
        vector = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise SolidPoleTideError(f"{label} must be a numeric 3-vector") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SolidPoleTideError(f"{label} must be a finite numeric 3-vector")
    return vector


def secular_pole_arcseconds(mjd_utc: float) -> tuple[float, float]:
    """Return the IERS 2018 working-version secular pole ``(x_s, y_s)``."""

    _finite_float(mjd_utc, "MJD UTC")
    if mjd_utc < 30_000.0 or mjd_utc > 100_000.0:
        raise SolidPoleTideError("MJD UTC is outside the supported bound")
    years_since_2000 = (mjd_utc - J2000_MJD) / DAYS_PER_JULIAN_YEAR
    # Equation (7.25): coefficients are milliarcseconds.
    x_arcseconds = (55.0 + 1.677 * years_since_2000) / 1_000.0
    y_arcseconds = (320.5 + 3.460 * years_since_2000) / 1_000.0
    return float(x_arcseconds), float(y_arcseconds)


def solid_pole_tide_displacement_metres(
    station_position_ecef_metres: object,
    *,
    mjd_utc: float,
    polar_motion_x_arcseconds: float,
    polar_motion_y_arcseconds: float,
) -> np.ndarray:
    """Return solid pole-tide station displacement in ECEF metres.

    ``polar_motion_x_arcseconds`` and ``polar_motion_y_arcseconds`` are the
    observed IERS polar-motion coordinates.  Local components follow IERS
    equation (7.24): radial is up, lambda is east, and theta is positive
    south.  The returned vector is contiguous binary64 ECEF XYZ.
    """

    station = _vector3(station_position_ecef_metres, "station position")
    _finite_float(mjd_utc, "MJD UTC")
    _finite_float(polar_motion_x_arcseconds, "polar motion x")
    _finite_float(polar_motion_y_arcseconds, "polar motion y")
    radius = float(np.linalg.norm(station))
    horizontal = float(math.hypot(station[0], station[1]))
    if radius < 6.0e6 or radius > 7.0e6 or horizontal <= 0.0:
        raise SolidPoleTideError("station position is outside the terrestrial bound")

    longitude = math.atan2(float(station[1]), float(station[0]))
    colatitude = math.atan2(horizontal, float(station[2]))
    sin_lambda = math.sin(longitude)
    cos_lambda = math.cos(longitude)
    sin_theta = math.sin(colatitude)
    cos_theta = math.cos(colatitude)

    secular_x, secular_y = secular_pole_arcseconds(mjd_utc)
    m1 = polar_motion_x_arcseconds - secular_x
    m2 = -(polar_motion_y_arcseconds - secular_y)
    wobble_projection = m1 * cos_lambda + m2 * sin_lambda

    south_mm = -9.0 * math.cos(2.0 * colatitude) * wobble_projection
    east_mm = 9.0 * cos_theta * (m1 * sin_lambda - m2 * cos_lambda)
    up_mm = -33.0 * math.sin(2.0 * colatitude) * wobble_projection

    up = np.asarray(
        (sin_theta * cos_lambda, sin_theta * sin_lambda, cos_theta),
        dtype=np.float64,
    )
    east = np.asarray((-sin_lambda, cos_lambda, 0.0), dtype=np.float64)
    north = np.asarray(
        (-cos_theta * cos_lambda, -cos_theta * sin_lambda, sin_theta),
        dtype=np.float64,
    )
    displacement = (
        up_mm * up + east_mm * east - south_mm * north
    ) / MILLIMETRES_PER_METRE
    norm = float(np.linalg.norm(displacement))
    if not np.all(np.isfinite(displacement)) or norm > 0.1:
        raise SolidPoleTideError("solid pole-tide displacement is invalid")
    return np.ascontiguousarray(displacement, dtype=np.float64)


__all__ = [
    "DAYS_PER_JULIAN_YEAR",
    "IERS_CHAPTER_7_WORKING_URL",
    "J2000_MJD",
    "MODEL_ID",
    "SolidPoleTideError",
    "secular_pole_arcseconds",
    "solid_pole_tide_displacement_metres",
]
