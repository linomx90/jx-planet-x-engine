"""Source-fixed lunar geodetic-precession transport primitive.

This opt-in component evaluates the inertial geodetic-precession rate used by
the retained DE440 lunar rotational equation and transports it into the
instantaneous lunar mantle frame.  Earth and Sun states are supplied by the
caller relative to the Moon; the component does not query an ephemeris or
select constants.

The returned rate belongs in the relative-frame gyroscopic terms of the
coupled mantle/core equations.  It is not an additive empirical torque.  This
screening-only primitive is not a DE440 reproduction, an LLR model, or a
registered production force.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import lunar_coupled


LUNAR_GEODETIC_PRECESSION_MODEL_ID = (
    "solar-system.rotation.lunar-geodetic-precession.experimental.v1"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MODEL_OUTPUT = "MODEL_OUTPUT"


class LunarGeodeticPrecessionError(ValueError):
    """A lunar geodetic-precession request violated its narrow contract."""


class LunarGeodeticPrecessionDependencyError(LunarGeodeticPrecessionError):
    """The explicitly required numerical runtime is unavailable."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarGeodeticPrecessionDependencyError(
            "lunar geodetic-precession evaluation requires NumPy"
        ) from exc
    return np


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise LunarGeodeticPrecessionError(
            f"{label} must be a positive finite built-in float"
        )
    return value


def _source_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise LunarGeodeticPrecessionError(
            "parameter_source_id must be bounded printable ASCII without spaces"
        )
    return value


def _vector(value: object, shape: tuple[int, ...], label: str):
    np = _numpy()
    if type(value) is not np.ndarray or (
        value.dtype != np.dtype("float64")
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarGeodeticPrecessionError(
            f"{label} must be finite contiguous float64 with shape {shape}"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class LunarGeodeticPrecessionParameters:
    """Explicit constants for the fixed geodetic-precession equation."""

    earth_gm_km3_s2: float
    sun_gm_km3_s2: float
    speed_of_light_km_s: float
    ppn_gamma: float
    parameter_source_id: str
    model_id: str = LUNAR_GEODETIC_PRECESSION_MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.earth_gm_km3_s2, "earth_gm_km3_s2"),
            (self.sun_gm_km3_s2, "sun_gm_km3_s2"),
            (self.speed_of_light_km_s, "speed_of_light_km_s"),
            (self.ppn_gamma, "ppn_gamma"),
        ):
            _positive_float(value, label)
        _source_id(self.parameter_source_id)
        if self.model_id != LUNAR_GEODETIC_PRECESSION_MODEL_ID:
            raise LunarGeodeticPrecessionError("model_id changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise LunarGeodeticPrecessionError("scientific_claim_state changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarGeodeticPrecessionError("evidence_class changed")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarGeodeticPrecessionError(f"{label} must be exact false")


@dataclass(frozen=True, slots=True, eq=False)
class LunarGeodeticPrecessionEvaluation:
    """Geodetic-precession rate in inertial and mantle-body coordinates."""

    inertial_rate_radians_per_second: object
    body_rate_radians_per_second: object
    gravitating_acceleration_km_s2: object


def evaluate_lunar_geodetic_precession(
    parameters: LunarGeodeticPrecessionParameters,
    *,
    quaternion_body_to_inertial: object,
    earth_relative_to_moon_state_inertial_km_km_s: object,
    sun_relative_to_moon_state_inertial_km_km_s: object,
) -> LunarGeodeticPrecessionEvaluation:
    """Evaluate and transport the fixed DE440 geodetic-precession rate.

    The state convention is source relative to Moon in J2000: position in km
    followed by velocity in km/s.  The Earth-relative velocity is the velocity
    appearing in the retained equation; only the Sun position contributes to
    the acceleration sum.
    """

    if type(parameters) is not LunarGeodeticPrecessionParameters:
        raise LunarGeodeticPrecessionError(
            "parameters must be exact LunarGeodeticPrecessionParameters"
        )
    np = _numpy()
    quaternion = _vector(
        quaternion_body_to_inertial, (4,), "quaternion_body_to_inertial"
    )
    earth = _vector(
        earth_relative_to_moon_state_inertial_km_km_s,
        (6,),
        "earth_relative_to_moon_state_inertial_km_km_s",
    )
    sun = _vector(
        sun_relative_to_moon_state_inertial_km_km_s,
        (6,),
        "sun_relative_to_moon_state_inertial_km_km_s",
    )
    earth_radius = float(np.linalg.norm(earth[:3]))
    sun_radius = float(np.linalg.norm(sun[:3]))
    if earth_radius <= 0.0 or sun_radius <= 0.0:
        raise LunarGeodeticPrecessionError(
            "Earth, Sun, and Moon must have nonzero separation"
        )
    acceleration = np.ascontiguousarray(
        parameters.earth_gm_km3_s2 * earth[:3] / earth_radius**3
        + parameters.sun_gm_km3_s2 * sun[:3] / sun_radius**3,
        dtype=np.float64,
    )
    prefactor = (1.0 + 2.0 * parameters.ppn_gamma) / (
        2.0 * parameters.speed_of_light_km_s**2
    )
    inertial_rate = np.ascontiguousarray(
        prefactor * np.cross(earth[3:], acceleration), dtype=np.float64
    )
    body_to_inertial = lunar_coupled._body_to_inertial(quaternion)
    body_rate = np.ascontiguousarray(
        body_to_inertial.T @ inertial_rate, dtype=np.float64
    )
    for value in (acceleration, inertial_rate, body_rate):
        value[value == 0.0] = 0.0
        if not np.all(np.isfinite(value)):
            raise LunarGeodeticPrecessionError(
                "geodetic-precession evaluation became nonfinite"
            )
    return LunarGeodeticPrecessionEvaluation(
        inertial_rate_radians_per_second=inertial_rate,
        body_rate_radians_per_second=body_rate,
        gravitating_acceleration_km_s2=acceleration,
    )
