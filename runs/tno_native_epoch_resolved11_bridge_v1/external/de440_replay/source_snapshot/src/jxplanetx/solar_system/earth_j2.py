"""Correction-only Earth J2 pair force for experimental Step 5 models.

This module connects :mod:`jxplanetx.solar_system.earth_pole` policies to one
axisymmetric, unnormalized Earth-J2 source/target correction.  It returns an
acceleration matrix only: no Newtonian monopole, trajectory integration,
complete DE440 force model, or qualification is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .earth_pole import (
    FIXED_J2000_POSITIVE_Z,
    PreparedEarthPolePolicy,
    prepare_earth_pole_policy,
)


EARTH_J2_PAIR_MODEL_ID = "solar-system.force.earth-j2-axisymmetric-pair"
DE440_EARTH_J2 = 1.0826359e-3
DE440_EARTH_REFERENCE_RADIUS_METRES = 6_378_136.6
MODEL_OUTPUT = "MODEL_OUTPUT"


class EarthJ2Error(ValueError):
    """An Earth-J2 configuration or evaluation violated its contract."""


class EarthJ2DependencyError(EarthJ2Error):
    """The explicitly requested NumPy force runtime is unavailable."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise EarthJ2Error(f"{label} must be bounded nonempty text")
    if value.strip() != value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise EarthJ2Error(f"{label} must be trimmed printable ASCII without spaces")
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EarthJ2Error(f"{label} must be a nonnegative built-in integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise EarthJ2Error(f"{label} must be a finite positive built-in float")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class EarthJ2PairForce:
    """Immutable configuration for one Earth-J2 source/target pair."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    j2: float
    reference_radius_metres: float
    pole_policy: PreparedEarthPolePolicy
    model_id: str = EARTH_J2_PAIR_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        source_id = _identifier(self.source_id, "source_id")
        target_id = _identifier(self.target_id, "target_id")
        if source_id != "EARTH":
            raise EarthJ2Error("the Earth-J2 source_id must be exact EARTH")
        if source_id == target_id:
            raise EarthJ2Error("source_id and target_id must be distinct")
        source_index = _index(self.source_index, "source_index")
        target_index = _index(self.target_index, "target_index")
        if source_index == target_index:
            raise EarthJ2Error("source_index and target_index must be distinct")
        _positive_float(self.j2, "j2")
        _positive_float(self.reference_radius_metres, "reference_radius_metres")
        if type(self.pole_policy) is not PreparedEarthPolePolicy:
            raise EarthJ2Error("pole_policy must be a prepared Earth-pole policy")
        if self.model_id != EARTH_J2_PAIR_MODEL_ID:
            raise EarthJ2Error("model_id is not the exact Earth-J2 pair model")
        if self.evidence_class != MODEL_OUTPUT:
            raise EarthJ2Error("Earth-J2 output must remain MODEL_OUTPUT")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise EarthJ2Error(f"{label} must be exact false")

    @property
    def pole_mode(self) -> str:
        """Return the exact configured pole-policy token."""

        return self.pole_policy.mode


def prepare_earth_j2_pair_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    j2: float = DE440_EARTH_J2,
    reference_radius_metres: float = DE440_EARTH_REFERENCE_RADIUS_METRES,
    pole_policy: PreparedEarthPolePolicy | None = None,
) -> EarthJ2PairForce:
    """Prepare one opt-in Earth-J2 correction with a fixed-+Z default."""

    if pole_policy is None:
        pole_policy = prepare_earth_pole_policy(FIXED_J2000_POSITIVE_Z)
    return EarthJ2PairForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        j2=j2,
        reference_radius_metres=reference_radius_metres,
        pole_policy=pole_policy,
    )


def evaluate_earth_j2_pair_correction(
    force: EarthJ2PairForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    gravitational_parameters: object,
    elapsed_time: float = 0.0,
):
    """Return the correction acceleration for the configured pair.

    ``positions[target] - positions[source]`` is the Earth-to-target vector.
    The target receives the direct J2 correction and the source receives the
    GM-weighted reaction used by the retained DE440 equation-29 experiment.
    Other rows remain exact zero.  Inputs are inspected but never mutated.
    """

    if type(force) is not EarthJ2PairForce:
        raise EarthJ2Error("force must be an exact EarthJ2PairForce")
    if type(body_ids) is not tuple or not body_ids:
        raise EarthJ2Error("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        _identifier(body_id, f"body_ids[{index}]")
        for index, body_id in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise EarthJ2Error("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise EarthJ2Error("configured pair index is outside body_ids")
    if checked_ids[force.source_index] != force.source_id:
        raise EarthJ2Error("source index does not match source_id")
    if checked_ids[force.target_index] != force.target_id:
        raise EarthJ2Error("target index does not match target_id")

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise EarthJ2DependencyError("Earth-J2 evaluation requires NumPy") from exc
    if type(positions) is not np.ndarray:
        raise EarthJ2Error("positions must be an exact NumPy array")
    if (
        positions.dtype != np.dtype("float64")
        or positions.ndim != 2
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise EarthJ2Error(
            "positions must be finite C-contiguous float64 with shape (N,3)"
        )
    if type(gravitational_parameters) is not np.ndarray:
        raise EarthJ2Error("gravitational_parameters must be an exact NumPy array")
    if (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or np.any(gravitational_parameters < 0.0)
    ):
        raise EarthJ2Error(
            "gravitational_parameters must be finite nonnegative C-contiguous float64"
        )
    source_gm = float(gravitational_parameters[force.source_index])
    target_gm = float(gravitational_parameters[force.target_index])
    if source_gm <= 0.0:
        raise EarthJ2Error("the J2 source must have positive GM")

    try:
        pole_values = force.pole_policy.raw_pole_at_elapsed_time(elapsed_time)
    except ValueError as exc:
        raise EarthJ2Error(str(exc)) from exc
    pole = np.ascontiguousarray(pole_values, dtype=np.float64)
    pole_norm = float(np.linalg.norm(pole))
    if (
        pole.shape != (3,)
        or not np.all(np.isfinite(pole))
        or not math.isfinite(pole_norm)
        or pole_norm <= 0.0
    ):
        raise EarthJ2Error("prepared pole is not a finite nonzero three-vector")
    unit_pole = np.ascontiguousarray(pole / pole_norm, dtype=np.float64)

    relative = positions[force.target_index] - positions[force.source_index]
    radius_squared = float(np.dot(relative, relative))
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise EarthJ2Error("source and target must have positive finite separation")
    radius = math.sqrt(radius_squared)
    axial = float(np.dot(relative, unit_pole))
    factor = (
        1.5
        * force.j2
        * source_gm
        * force.reference_radius_metres
        * force.reference_radius_metres
        / (radius_squared * radius_squared * radius)
    )
    direct = factor * (
        (5.0 * axial * axial / radius_squared - 1.0) * relative
        - 2.0 * axial * unit_pole
    )
    correction = np.zeros_like(positions, dtype=np.float64)
    correction[force.target_index] = direct
    correction[force.source_index] = -(target_gm / source_gm) * direct
    if not np.all(np.isfinite(correction)):
        raise EarthJ2Error("Earth-J2 correction became nonfinite")
    return np.ascontiguousarray(correction, dtype=np.float64)


__all__ = [
    "DE440_EARTH_J2",
    "DE440_EARTH_REFERENCE_RADIUS_METRES",
    "EARTH_J2_PAIR_MODEL_ID",
    "EarthJ2DependencyError",
    "EarthJ2Error",
    "EarthJ2PairForce",
    "evaluate_earth_j2_pair_correction",
    "prepare_earth_j2_pair_force",
]
