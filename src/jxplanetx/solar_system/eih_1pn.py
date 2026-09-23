"""Screening-only barycentric point-mass EIH 1PN correction.

This module implements the general-relativistic (beta = gamma = 1)
Einstein--Infeld--Hoffmann point-mass correction from equation 27 of the JPL
DE440/DE441 description.  It returns only the velocity-dependent 1PN
correction; callers must add mutual Newtonian gravity exactly once and must use
an integrator that reevaluates velocities at every stage.

The acceleration-dependent source terms are evaluated with simultaneous
Newtonian point-mass accelerations, which is equivalent through retained 1PN
order.  The model excludes figures, tides, spins, frame dragging, asteroids,
Kuiper-belt sources, signal propagation, and observation reduction.  It is
``SCREENING_ONLY`` and conveys no registry or production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


EIH_1PN_MODEL_ID = "force.relativity.eih_1pn_gr"
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
COORDINATE_SCOPE = "BARYCENTRIC_INERTIAL_POINT_MASS_EIH_1PN_TDB_COMPATIBLE"


class EIH1PNError(ValueError):
    """An EIH configuration or simultaneous state violated the contract."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise EIH1PNError("EIH 1PN evaluation requires NumPy") from exc
    return np


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise EIH1PNError(f"{label} must be a positive finite built-in float")
    return value


def _positive_fraction(value: object, label: str) -> float:
    result = _positive_float(value, label)
    if result >= 1.0:
        raise EIH1PNError(f"{label} must be strictly less than one")
    return result


def _state_matrix(value: object, label: str, *, rows: int | None = None):
    np = _numpy()
    if type(value) is not np.ndarray:
        raise EIH1PNError(f"{label} must be an exact NumPy array")
    if value.dtype != np.dtype("float64") or value.ndim != 2 or value.shape[1] != 3:
        raise EIH1PNError(f"{label} must have float64 shape (N,3)")
    if rows is not None and value.shape[0] != rows:
        raise EIH1PNError(f"{label} has the wrong body count")
    if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise EIH1PNError(f"{label} must be finite and C-contiguous")
    return value


def _gm_vector(value: object, rows: int):
    np = _numpy()
    if type(value) is not np.ndarray:
        raise EIH1PNError("gravitational_parameters_km3_s2 must be an exact NumPy array")
    if value.dtype != np.dtype("float64") or value.shape != (rows,):
        raise EIH1PNError(
            "gravitational_parameters_km3_s2 must have float64 shape (N,)"
        )
    if (
        not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
        or np.any(value <= 0.0)
    ):
        raise EIH1PNError(
            "gravitational_parameters_km3_s2 must be finite, positive, and C-contiguous"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class EIH1PNParameters:
    """Fixed GR and weak-field domain contract for one EIH evaluation."""

    speed_of_light_km_s: float
    maximum_compactness: float = 1.0e-4
    maximum_speed_fraction_squared: float = 1.0e-4
    beta: float = 1.0
    gamma: float = 1.0
    model_id: str = EIH_1PN_MODEL_ID
    coordinate_scope: str = COORDINATE_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    correction_only: bool = True
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_ready: bool = False

    def __post_init__(self) -> None:
        _positive_float(self.speed_of_light_km_s, "speed_of_light_km_s")
        _positive_fraction(self.maximum_compactness, "maximum_compactness")
        _positive_fraction(
            self.maximum_speed_fraction_squared,
            "maximum_speed_fraction_squared",
        )
        if type(self.beta) is not float or self.beta != 1.0:
            raise EIH1PNError("beta must be exact built-in float 1.0")
        if type(self.gamma) is not float or self.gamma != 1.0:
            raise EIH1PNError("gamma must be exact built-in float 1.0")
        if self.model_id != EIH_1PN_MODEL_ID:
            raise EIH1PNError("model_id changed")
        if self.coordinate_scope != COORDINATE_SCOPE:
            raise EIH1PNError("coordinate_scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise EIH1PNError("scientific_claim_state changed")
        for value, label, required in (
            (self.correction_only, "correction_only", True),
            (self.registry_authorized, "registry_authorized", False),
            (self.qualification_authorized, "qualification_authorized", False),
            (self.production_ready, "production_ready", False),
        ):
            if type(value) is not bool or value is not required:
                raise EIH1PNError(f"{label} must be exact {required!r}")


@dataclass(frozen=True, slots=True, eq=False)
class EIH1PNEvaluation:
    """Owned correction and audited weak-field domain values."""

    correction_accelerations_km_s2: object
    maximum_compactness: float
    maximum_speed_fraction_squared: float
    body_count: int
    model_id: str = EIH_1PN_MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    correction_only: bool = True
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_ready: bool = False

    def __post_init__(self) -> None:
        np = _numpy()
        if type(self.body_count) is not int or self.body_count < 2:
            raise EIH1PNError("body_count must be an integer of at least two")
        correction = _state_matrix(
            self.correction_accelerations_km_s2,
            "correction_accelerations_km_s2",
            rows=self.body_count,
        )
        owned = np.array(correction, dtype=np.float64, order="C", copy=True)
        owned[owned == 0.0] = 0.0
        owned.setflags(write=False)
        object.__setattr__(self, "correction_accelerations_km_s2", owned)
        for value, label in (
            (self.maximum_compactness, "maximum_compactness"),
            (
                self.maximum_speed_fraction_squared,
                "maximum_speed_fraction_squared",
            ),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise EIH1PNError(f"{label} must be a nonnegative finite float")
        if self.model_id != EIH_1PN_MODEL_ID:
            raise EIH1PNError("evaluation model_id changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise EIH1PNError("evaluation scientific_claim_state changed")
        for value, label, required in (
            (self.correction_only, "correction_only", True),
            (self.registry_authorized, "registry_authorized", False),
            (self.qualification_authorized, "qualification_authorized", False),
            (self.production_ready, "production_ready", False),
        ):
            if type(value) is not bool or value is not required:
                raise EIH1PNError(f"evaluation {label} must be exact {required!r}")


def newtonian_point_mass_accelerations(
    positions_km: object,
    gravitational_parameters_km3_s2: object,
):
    """Return simultaneous Newtonian accelerations in deterministic pair order."""

    np = _numpy()
    positions = _state_matrix(positions_km, "positions_km")
    gm = _gm_vector(gravitational_parameters_km3_s2, positions.shape[0])
    if positions.shape[0] < 2:
        raise EIH1PNError("point-mass evaluation requires at least two bodies")
    acceleration = np.zeros_like(positions, dtype=np.float64)
    for left in range(positions.shape[0]):
        for right in range(left + 1, positions.shape[0]):
            difference = positions[left] - positions[right]
            radius_squared = float(np.dot(difference, difference))
            if not math.isfinite(radius_squared) or radius_squared <= 0.0:
                raise EIH1PNError("point masses must have positive separation")
            inverse_radius_cubed = 1.0 / (
                radius_squared * math.sqrt(radius_squared)
            )
            direction = difference * inverse_radius_cubed
            acceleration[left] -= gm[right] * direction
            acceleration[right] += gm[left] * direction
    if not np.all(np.isfinite(acceleration)):
        raise EIH1PNError("Newtonian acceleration became nonfinite")
    return np.ascontiguousarray(acceleration, dtype=np.float64)


def evaluate_eih_1pn_correction(
    positions_km: object,
    velocities_km_s: object,
    gravitational_parameters_km3_s2: object,
    parameters: EIH1PNParameters,
) -> EIH1PNEvaluation:
    """Evaluate equation-27 EIH acceleration minus its Newtonian base."""

    np = _numpy()
    if type(parameters) is not EIH1PNParameters:
        raise EIH1PNError("parameters must be exact EIH1PNParameters")
    positions = _state_matrix(positions_km, "positions_km")
    velocities = _state_matrix(
        velocities_km_s,
        "velocities_km_s",
        rows=positions.shape[0],
    )
    gm = _gm_vector(gravitational_parameters_km3_s2, positions.shape[0])
    count = positions.shape[0]
    if count < 2:
        raise EIH1PNError("EIH evaluation requires at least two bodies")

    c_squared = parameters.speed_of_light_km_s**2
    if not math.isfinite(c_squared):
        raise EIH1PNError("speed_of_light_km_s squared became nonfinite")
    newtonian = newtonian_point_mass_accelerations(positions, gm)
    separations = np.empty((count, count), dtype=np.float64)
    potentials = np.zeros(count, dtype=np.float64)
    for body in range(count):
        separations[body, body] = math.inf
        for source in range(count):
            if source == body:
                continue
            radius = float(np.linalg.norm(positions[body] - positions[source]))
            if not math.isfinite(radius) or radius <= 0.0:
                raise EIH1PNError("EIH point masses must have positive separation")
            separations[body, source] = radius
            potentials[body] += gm[source] / radius

    compactness = np.ascontiguousarray(potentials / c_squared)
    speed_fractions = np.ascontiguousarray(
        np.sum(velocities * velocities, axis=1) / c_squared
    )
    maximum_compactness = float(np.max(compactness))
    maximum_speed_fraction = float(np.max(speed_fractions))
    if maximum_compactness > parameters.maximum_compactness:
        raise EIH1PNError("state exceeds maximum_compactness")
    if maximum_speed_fraction > parameters.maximum_speed_fraction_squared:
        raise EIH1PNError("state exceeds maximum_speed_fraction_squared")

    correction = np.zeros_like(positions, dtype=np.float64)
    for body in range(count):
        velocity_body = velocities[body]
        speed_body_squared = float(np.dot(velocity_body, velocity_body))
        for source in range(count):
            if source == body:
                continue
            difference = positions[body] - positions[source]
            radius = float(separations[body, source])
            radius_squared = radius * radius
            radius_cubed = radius_squared * radius
            velocity_source = velocities[source]
            radial_source_velocity = float(np.dot(difference, velocity_source))
            bracket = (
                4.0 * float(potentials[body])
                + float(potentials[source])
                - speed_body_squared
                - 2.0 * float(np.dot(velocity_source, velocity_source))
                + 4.0 * float(np.dot(velocity_body, velocity_source))
                + 1.5
                * radial_source_velocity
                * radial_source_velocity
                / radius_squared
                + 0.5 * float(np.dot(difference, newtonian[source]))
            ) / c_squared
            velocity_bracket = float(
                np.dot(difference, 4.0 * velocity_body - 3.0 * velocity_source)
            )
            correction[body] += gm[source] * (
                difference * (bracket / radius_cubed)
                + (
                    velocity_bracket
                    * (velocity_body - velocity_source)
                    / radius_cubed
                    + 3.5 * newtonian[source] / radius
                )
                / c_squared
            )

    if not np.all(np.isfinite(correction)):
        raise EIH1PNError("EIH correction became nonfinite")
    return EIH1PNEvaluation(
        correction_accelerations_km_s2=np.ascontiguousarray(correction),
        maximum_compactness=maximum_compactness,
        maximum_speed_fraction_squared=maximum_speed_fraction,
        body_count=count,
    )


__all__ = [
    "COORDINATE_SCOPE",
    "EIH_1PN_MODEL_ID",
    "EIH1PNError",
    "EIH1PNEvaluation",
    "EIH1PNParameters",
    "SCIENTIFIC_CLAIM_STATE",
    "evaluate_eih_1pn_correction",
    "newtonian_point_mass_accelerations",
]
