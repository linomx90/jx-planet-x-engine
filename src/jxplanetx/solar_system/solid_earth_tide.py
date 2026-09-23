"""IERS-2010-compatible solid-Earth-tide station displacement.

This is a renamed Python adaptation of the algorithms in the IERS
Conventions routine ``DEHANTTIDEINEL`` and its ``ST1*``/``STEP2*`` helpers.
It is not IERS Conventions software and is neither distributed nor endorsed
by the IERS Conventions Center.  Differences from the reference routines:

* the public API and every helper have JX-specific Python names;
* NumPy arrays replace Fortran arrays;
* PyERFA supplies UTC-to-TT conversion instead of the reference routine's
  embedded leap-second table; and
* validation rejects non-finite, incorrectly shaped, or out-of-range inputs.

The permanent-tide step is deliberately omitted, matching the conventional
zero-tide output of the reference routine.  Equations, coefficient tables,
and reference vectors are from the official IERS Chapter 7 software at
https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/.

IERS Conventions Software License
=================================

NOTICE TO USER:

BY USING THIS SOFTWARE YOU ACCEPT THE FOLLOWING TERMS AND CONDITIONS WHICH
APPLY TO ITS USE.

1. The Software is provided by the IERS Conventions Center ("the Center").

2. Permission is granted to anyone to use the Software for any purpose,
   including commercial applications, free of charge, subject to the
   conditions and restrictions listed below.

3. You (the user) may adapt the Software and its algorithms for your own
   purposes and you may distribute the resulting "derived work" to others,
   provided that the derived work complies with the following requirements:

   a) Your work shall be clearly identified so that it cannot be mistaken
      for IERS Conventions software and that it has been neither distributed
      by nor endorsed by the Center.

   b) Your work (including source code) must contain descriptions of how the
      derived work is based upon and/or differs from the original Software.

   c) The name(s) of all modified routine(s) that you distribute shall be
      changed.

   d) The origin of the IERS Conventions components of your derived work must
      not be misrepresented; you must not claim that you wrote the original
      Software.

   e) The source code must be included for all routine(s) that you distribute.
      This notice must be reproduced intact in any source distribution.

4. In any published work produced by the user and which includes results
   achieved by using the Software, you shall acknowledge that the Software
   was used in obtaining those results.

5. The Software is provided to the user "as is" and the Center makes no
   warranty as to its use or performance. The Center does not and cannot
   warrant the performance or results which the user may obtain by using the
   Software. The Center makes no warranties, express or implied, as to
   non-infringement of third party rights, merchantability, or fitness for any
   particular purpose. In no event will the Center be liable to the user for
   any consequential, incidental, or special damages, including any lost
   profits or lost savings, even if a Center representative has been advised
   of such damages, or for any claim by any third party.

Correspondence concerning IERS Conventions software should be addressed as
follows:

                    Gerard Petit
    Internet email: gpetit[at]bipm.org
    Postal address: IERS Conventions Center
                    Time, frequency and gravimetry section, BIPM
                    Pavillon de Breteuil
                    92312 Sevres  FRANCE

or

                    Brian Luzum
    Internet email: brian.luzum[at]usno.navy.mil
    Postal address: IERS Conventions Center
                    Earth Orientation Department
                    3450 Massachusetts Ave, NW
                    Washington, DC 20392
"""

from __future__ import annotations

from datetime import datetime
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


MODEL_ID = "station-displacement.solid-earth-tide.iers2010-adaptation.v1"
IERS_SOURCE_URL = (
    "https://iers-conventions.obspm.fr/content/chapter7/software/"
    "dehanttideinel/"
)
IERS_SOURCE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "DEHANTTIDEINEL.F": "bc6039a1704761881bb785ce44ce084ea82783107ff64c576e69155a4914e2cb",
        "ST1IDIU.F": "d2976b8b76be8dd1d57e57a8d6b48f5764676126515bada3592753a07d3acd1e",
        "ST1ISEM.F": "efdf284bd977826a1f4aea4c79c5dbd0c38fc1c403a2376010048b511a11f2c6",
        "ST1L1.F": "b1dfd0e797a3ce950631ad7dbbf5f576bf6b58dc515688253a3d84ca059bc282",
        "STEP2DIU.F": "898c70d4b8d50e09e0c717c911c4117b3ad1ca4996d369c258b68d00ea3a5674",
        "STEP2LON.F": "f9d3bf0317222986d22e53557020bb13a6fbb90f8e3c9915137da6184d82813a",
    }
)
J2000_JD = 2_451_545.0
JULIAN_CENTURY_DAYS = 36_525.0
SECONDS_PER_DAY = 86_400.0
DEGREE_TO_RADIAN = math.pi / 180.0


_DIURNAL_TERMS = (
    (-3, 0, 2, 0, 0, -0.01, 0.0, 0.0, 0.0),
    (-3, 2, 0, 0, 0, -0.01, 0.0, 0.0, 0.0),
    (-2, 0, 1, -1, 0, -0.02, 0.0, 0.0, 0.0),
    (-2, 0, 1, 0, 0, -0.08, 0.0, -0.01, 0.01),
    (-2, 2, -1, 0, 0, -0.02, 0.0, 0.0, 0.0),
    (-1, 0, 0, -1, 0, -0.10, 0.0, 0.0, 0.0),
    (-1, 0, 0, 0, 0, -0.51, 0.0, -0.02, 0.03),
    (-1, 2, 0, 0, 0, 0.01, 0.0, 0.0, 0.0),
    (0, -2, 1, 0, 0, 0.01, 0.0, 0.0, 0.0),
    (0, 0, -1, 0, 0, 0.02, 0.0, 0.0, 0.0),
    (0, 0, 1, 0, 0, 0.06, 0.0, 0.0, 0.0),
    (0, 0, 1, 1, 0, 0.01, 0.0, 0.0, 0.0),
    (0, 2, -1, 0, 0, 0.01, 0.0, 0.0, 0.0),
    (1, -3, 0, 0, 1, -0.06, 0.0, 0.0, 0.0),
    (1, -2, 0, -1, 0, 0.01, 0.0, 0.0, 0.0),
    (1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01),
    (1, -1, 0, 0, -1, 0.02, 0.0, 0.0, 0.0),
    (1, -1, 0, 0, 1, 0.04, 0.0, 0.0, 0.0),
    (1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0.0),
    (1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03),
    (1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0.0),
    (1, 0, 0, 2, 0, -0.04, 0.0, 0.0, 0.0),
    (1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0.0),
    (1, 1, 0, 0, 1, 0.01, 0.0, 0.0, 0.0),
    (0, 1, 0, 1, -1, -0.01, 0.0, 0.0, 0.0),
    (1, 2, -2, 0, 0, -0.01, 0.0, 0.0, 0.0),
    (1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0.0),
    (2, -2, 1, 0, 0, -0.01, 0.0, 0.0, 0.0),
    (2, 0, -1, 0, 0, -0.02, 0.0, 0.0, 0.0),
    (3, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0),
    (3, 0, 0, 1, 0, 0.0, 0.0, 0.0, 0.0),
)

_LONG_PERIOD_TERMS = (
    (0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07),
    (0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05),
    (1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04),
    (2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07),
    (2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03),
)


class SolidEarthTideError(ValueError):
    """A solid-Earth-tide input or result violated the model contract."""


def _jx_vector(value: object, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SolidEarthTideError(f"{label} must be a finite three-vector") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SolidEarthTideError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SolidEarthTideError(f"{label} must have positive finite norm")
    return np.ascontiguousarray(vector)


def _jx_local_basis(station: np.ndarray) -> tuple[float, ...]:
    radius = float(np.linalg.norm(station))
    sine_phi = float(station[2] / radius)
    cosine_phi = float(math.hypot(station[0], station[1]) / radius)
    if cosine_phi <= 0.0:
        raise SolidEarthTideError("polar stations are outside this adaptation's bound")
    cosine_lambda = float(station[0] / (cosine_phi * radius))
    sine_lambda = float(station[1] / (cosine_phi * radius))
    return radius, sine_phi, cosine_phi, sine_lambda, cosine_lambda


def _jx_local_to_ecef(
    radial: float,
    east: float,
    north: float,
    sine_phi: float,
    cosine_phi: float,
    sine_lambda: float,
    cosine_lambda: float,
) -> np.ndarray:
    return np.asarray(
        (
            radial * cosine_lambda * cosine_phi
            - east * sine_lambda
            - north * sine_phi * cosine_lambda,
            radial * sine_lambda * cosine_phi
            + east * cosine_lambda
            - north * sine_phi * sine_lambda,
            radial * sine_phi + north * cosine_phi,
        ),
        dtype=np.float64,
    )


def _jx_step1_diurnal(
    station: np.ndarray,
    sun: np.ndarray,
    moon: np.ndarray,
    factor_sun: float,
    factor_moon: float,
) -> np.ndarray:
    _, sin_phi, cos_phi, sin_lam, cos_lam = _jx_local_basis(station)
    cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi
    r_sun = float(np.linalg.norm(sun))
    r_moon = float(np.linalg.norm(moon))
    dhi = -0.0025
    dli = -0.0007

    def terms(body: np.ndarray, radius: float, factor: float) -> tuple[float, ...]:
        cross = float(body[0] * sin_lam - body[1] * cos_lam)
        along = float(body[0] * cos_lam + body[1] * sin_lam)
        radial = -3.0 * dhi * sin_phi * cos_phi * factor * body[2] * cross / radius**2
        north = -3.0 * dli * cos_2phi * factor * body[2] * cross / radius**2
        east = -3.0 * dli * sin_phi * factor * body[2] * along / radius**2
        return radial, east, north

    sun_terms = terms(sun, r_sun, factor_sun)
    moon_terms = terms(moon, r_moon, factor_moon)
    return _jx_local_to_ecef(
        sun_terms[0] + moon_terms[0],
        sun_terms[1] + moon_terms[1],
        sun_terms[2] + moon_terms[2],
        sin_phi,
        cos_phi,
        sin_lam,
        cos_lam,
    )


def _jx_step1_semidiurnal(
    station: np.ndarray,
    sun: np.ndarray,
    moon: np.ndarray,
    factor_sun: float,
    factor_moon: float,
) -> np.ndarray:
    _, sin_phi, cos_phi, sin_lam, cos_lam = _jx_local_basis(station)
    cos_2lam = cos_lam * cos_lam - sin_lam * sin_lam
    sin_2lam = 2.0 * cos_lam * sin_lam
    r_sun = float(np.linalg.norm(sun))
    r_moon = float(np.linalg.norm(moon))
    dhi = -0.0022
    dli = -0.0007

    def terms(body: np.ndarray, radius: float, factor: float) -> tuple[float, ...]:
        quadrature = (
            (body[0] ** 2 - body[1] ** 2) * sin_2lam
            - 2.0 * body[0] * body[1] * cos_2lam
        )
        in_phase = (
            (body[0] ** 2 - body[1] ** 2) * cos_2lam
            + 2.0 * body[0] * body[1] * sin_2lam
        )
        radial = -0.75 * dhi * cos_phi**2 * factor * quadrature / radius**2
        north = 1.5 * dli * sin_phi * cos_phi * factor * quadrature / radius**2
        east = -1.5 * dli * cos_phi * factor * in_phase / radius**2
        return radial, east, north

    sun_terms = terms(sun, r_sun, factor_sun)
    moon_terms = terms(moon, r_moon, factor_moon)
    return _jx_local_to_ecef(
        sun_terms[0] + moon_terms[0],
        sun_terms[1] + moon_terms[1],
        sun_terms[2] + moon_terms[2],
        sin_phi,
        cos_phi,
        sin_lam,
        cos_lam,
    )


def _jx_step1_latitude(
    station: np.ndarray,
    sun: np.ndarray,
    moon: np.ndarray,
    factor_sun: float,
    factor_moon: float,
) -> np.ndarray:
    _, sin_phi, cos_phi, sin_lam, cos_lam = _jx_local_basis(station)
    r_sun = float(np.linalg.norm(sun))
    r_moon = float(np.linalg.norm(moon))

    def diurnal(body: np.ndarray, radius: float, factor: float) -> tuple[float, float]:
        north = (
            -0.0012
            * sin_phi**2
            * factor
            * body[2]
            * (body[0] * cos_lam + body[1] * sin_lam)
            / radius**2
        )
        east = (
            0.0012
            * sin_phi
            * (cos_phi**2 - sin_phi**2)
            * factor
            * body[2]
            * (body[0] * sin_lam - body[1] * cos_lam)
            / radius**2
        )
        return 3.0 * east, 3.0 * north

    east_sun, north_sun = diurnal(sun, r_sun, factor_sun)
    east_moon, north_moon = diurnal(moon, r_moon, factor_moon)
    result = _jx_local_to_ecef(
        0.0,
        east_sun + east_moon,
        north_sun + north_moon,
        sin_phi,
        cos_phi,
        sin_lam,
        cos_lam,
    )

    cos_2lam = cos_lam**2 - sin_lam**2
    sin_2lam = 2.0 * cos_lam * sin_lam

    def semidiurnal(
        body: np.ndarray, radius: float, factor: float
    ) -> tuple[float, float]:
        in_phase = (
            (body[0] ** 2 - body[1] ** 2) * cos_2lam
            + 2.0 * body[0] * body[1] * sin_2lam
        )
        quadrature = (
            (body[0] ** 2 - body[1] ** 2) * sin_2lam
            - 2.0 * body[0] * body[1] * cos_2lam
        )
        north = -0.0012 * sin_phi * cos_phi * factor * in_phase / radius**2
        east = -0.0012 * sin_phi**2 * cos_phi * factor * quadrature / radius**2
        return 3.0 * east, 3.0 * north

    east_sun, north_sun = semidiurnal(sun, r_sun, factor_sun)
    east_moon, north_moon = semidiurnal(moon, r_moon, factor_moon)
    return result + _jx_local_to_ecef(
        0.0,
        east_sun + east_moon,
        north_sun + north_moon,
        sin_phi,
        cos_phi,
        sin_lam,
        cos_lam,
    )


def _jx_fundamental_arguments(tt_centuries: float) -> tuple[float, ...]:
    t = tt_centuries
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t
    precession = (
        1.396971278
        + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t
    ) * t
    s += precession
    h = 280.46645 + (
        36000.7697489
        + (0.00030322222 + (0.000000020 - 0.00000000654 * t) * t) * t
    ) * t
    p = 83.35324312 + (
        4069.01363525
        + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t
    ) * t
    node = 234.95544499 + (
        1934.13626197
        + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t
    ) * t
    perihelion = 282.93734098 + (
        1.71945766667
        + (0.00045688889 + (-0.00000001778 - 0.00000000334 * t) * t) * t
    ) * t
    return tuple(math.fmod(value, 360.0) for value in (s, h, p, node, perihelion))


def _jx_step2_diurnal(
    station: np.ndarray,
    fractional_utc_hours: float,
    tt_centuries: float,
) -> np.ndarray:
    _, sin_phi, cos_phi, sin_lam, cos_lam = _jx_local_basis(station)
    s, h, p, node, perihelion = _jx_fundamental_arguments(tt_centuries)
    tau = math.fmod(
        fractional_utc_hours * 15.0
        + 280.4606184
        + (36000.7700536 + (0.00038793 - 0.0000000258 * tt_centuries) * tt_centuries)
        * tt_centuries
        - (s - (
            1.396971278
            + (0.000308889 + (0.000000021 + 0.000000007 * tt_centuries) * tt_centuries)
            * tt_centuries
        ) * tt_centuries),
        360.0,
    )
    longitude = math.atan2(station[1], station[0])
    output = np.zeros(3, dtype=np.float64)
    for term in _DIURNAL_TERMS:
        angle = (
            tau
            + term[0] * s
            + term[1] * h
            + term[2] * p
            + term[3] * node
            + term[4] * perihelion
        ) * DEGREE_TO_RADIAN + longitude
        radial = (
            term[5] * 2.0 * sin_phi * cos_phi * math.sin(angle)
            + term[6] * 2.0 * sin_phi * cos_phi * math.cos(angle)
        )
        north = (
            term[7] * (cos_phi**2 - sin_phi**2) * math.sin(angle)
            + term[8] * (cos_phi**2 - sin_phi**2) * math.cos(angle)
        )
        east = term[7] * sin_phi * math.cos(angle) - term[8] * sin_phi * math.sin(angle)
        output += _jx_local_to_ecef(
            radial, east, north, sin_phi, cos_phi, sin_lam, cos_lam
        )
    return output / 1000.0


def _jx_step2_long_period(station: np.ndarray, tt_centuries: float) -> np.ndarray:
    _, sin_phi, cos_phi, sin_lam, cos_lam = _jx_local_basis(station)
    arguments = _jx_fundamental_arguments(tt_centuries)
    output = np.zeros(3, dtype=np.float64)
    for term in _LONG_PERIOD_TERMS:
        angle = math.fsum(term[index] * arguments[index] for index in range(5))
        angle *= DEGREE_TO_RADIAN
        radial = (
            term[5] * (3.0 * sin_phi**2 - 1.0) / 2.0 * math.cos(angle)
            + term[7] * (3.0 * sin_phi**2 - 1.0) / 2.0 * math.sin(angle)
        )
        north = (
            term[6] * 2.0 * cos_phi * sin_phi * math.cos(angle)
            + term[8] * 2.0 * cos_phi * sin_phi * math.sin(angle)
        )
        output += _jx_local_to_ecef(
            radial, 0.0, north, sin_phi, cos_phi, sin_lam, cos_lam
        )
    return output / 1000.0


def _jx_tt_centuries(
    year: int,
    month: int,
    day: int,
    seconds_of_day: float,
    erfa_module: Any,
) -> float:
    hour = int(seconds_of_day // 3600.0)
    minute = int((seconds_of_day - hour * 3600.0) // 60.0)
    second = float(seconds_of_day - hour * 3600.0 - minute * 60.0)
    try:
        utc = erfa_module.dtf2d("UTC", year, month, day, hour, minute, second)
        tai = erfa_module.utctai(float(utc[0]), float(utc[1]))
        tt = erfa_module.taitt(float(tai[0]), float(tai[1]))
        centuries = math.fsum((float(tt[0]) - J2000_JD, float(tt[1]))) / JULIAN_CENTURY_DAYS
    except Exception as exc:
        raise SolidEarthTideError("PyERFA UTC-to-TT conversion failed") from exc
    if not math.isfinite(centuries):
        raise SolidEarthTideError("PyERFA returned a non-finite TT epoch")
    return float(centuries)


def solid_earth_tide_displacement_metres(
    station_ecef_metres: object,
    sun_geocentric_ecef_metres: object,
    moon_geocentric_ecef_metres: object,
    *,
    year: int,
    month: int,
    day: int,
    seconds_of_day: float,
    erfa_module: Any,
) -> np.ndarray:
    """Return the conventional zero-tide station displacement in ECEF metres."""

    if any(type(value) is not int for value in (year, month, day)):
        raise SolidEarthTideError("UTC calendar fields must be exact integers")
    if type(seconds_of_day) is not float or not math.isfinite(seconds_of_day):
        raise SolidEarthTideError("UTC seconds of day must be a finite float")
    if not 0.0 <= seconds_of_day < SECONDS_PER_DAY:
        raise SolidEarthTideError("UTC seconds of day is outside [0, 86400)")
    try:
        datetime(year, month, day)
    except ValueError as exc:
        raise SolidEarthTideError("UTC calendar date is invalid") from exc

    station = _jx_vector(station_ecef_metres, "station ECEF position")
    sun = _jx_vector(sun_geocentric_ecef_metres, "Sun geocentric ECEF position")
    moon = _jx_vector(moon_geocentric_ecef_metres, "Moon geocentric ECEF position")
    station_radius = float(np.linalg.norm(station))
    sun_radius = float(np.linalg.norm(sun))
    moon_radius = float(np.linalg.norm(moon))
    if not 6.0e6 <= station_radius <= 7.0e6:
        raise SolidEarthTideError("station ECEF radius is outside the terrestrial bound")
    if sun_radius < 1.0e10 or moon_radius < 1.0e8:
        raise SolidEarthTideError("Sun or Moon distance is outside the geocentric bound")

    cosine_phi = math.hypot(station[0], station[1]) / station_radius
    h2 = 0.6078 - 0.0006 * (1.0 - 1.5 * cosine_phi**2)
    l2 = 0.0847 + 0.0002 * (1.0 - 1.5 * cosine_phi**2)
    h3 = 0.292
    l3 = 0.015
    cosine_sun = float(np.dot(station, sun) / (station_radius * sun_radius))
    cosine_moon = float(np.dot(station, moon) / (station_radius * moon_radius))

    earth_radius = 6_378_136.6
    factor2_sun = 332_946.0482 * earth_radius * (earth_radius / sun_radius) ** 3
    factor2_moon = 0.0123000371 * earth_radius * (earth_radius / moon_radius) ** 3
    factor3_sun = factor2_sun * earth_radius / sun_radius
    factor3_moon = factor2_moon * earth_radius / moon_radius

    def nominal(
        body: np.ndarray,
        body_radius: float,
        cosine: float,
        factor2: float,
        factor3: float,
    ) -> np.ndarray:
        p2 = 3.0 * (h2 / 2.0 - l2) * cosine**2 - h2 / 2.0
        p3 = 2.5 * (h3 - 3.0 * l3) * cosine**3 + 1.5 * (l3 - h3) * cosine
        x2 = 3.0 * l2 * cosine
        x3 = 1.5 * l3 * (5.0 * cosine**2 - 1.0)
        return (
            factor2 * (x2 * body / body_radius + p2 * station / station_radius)
            + factor3 * (x3 * body / body_radius + p3 * station / station_radius)
        )

    displacement = nominal(sun, sun_radius, cosine_sun, factor2_sun, factor3_sun)
    displacement += nominal(moon, moon_radius, cosine_moon, factor2_moon, factor3_moon)
    displacement += _jx_step1_diurnal(
        station, sun, moon, factor2_sun, factor2_moon
    )
    displacement += _jx_step1_semidiurnal(
        station, sun, moon, factor2_sun, factor2_moon
    )
    displacement += _jx_step1_latitude(
        station, sun, moon, factor2_sun, factor2_moon
    )
    tt_centuries = _jx_tt_centuries(
        year, month, day, seconds_of_day, erfa_module
    )
    displacement += _jx_step2_diurnal(
        station, seconds_of_day / 3600.0, tt_centuries
    )
    displacement += _jx_step2_long_period(station, tt_centuries)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise SolidEarthTideError("solid-Earth-tide displacement is invalid")
    return np.ascontiguousarray(displacement, dtype=np.float64)


__all__ = [
    "IERS_SOURCE_SHA256",
    "IERS_SOURCE_URL",
    "MODEL_ID",
    "SolidEarthTideError",
    "solid_earth_tide_displacement_metres",
]
