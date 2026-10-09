"""Correction-only axisymmetric Earth zonals through degree five.

This additive post-V5 module evaluates unnormalized ``J_l`` terms for one
Earth/source and target pair.  It deliberately contains no monopole,
trajectory integration, tesseral harmonics, tides, or qualification claim.
The target receives the direct correction and Earth receives the same
GM-weighted reaction convention used by the retained Step 5 J2 experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .earth_pole import (
    FIXED_J2000_POSITIVE_Z,
    PreparedEarthPolePolicy,
    prepare_earth_pole_policy,
)


EARTH_ZONAL_PAIR_MODEL_ID = "solar-system.force.earth-zonal-j2-j5-axisymmetric-pair"
DE440_TECH_COMMENTS_EARTH_REFERENCE_RADIUS_METRES = 6_378_136.6
DE440_TECH_COMMENTS_EARTH_J2 = 1.0826253900000000e-3
DE440_TECH_COMMENTS_EARTH_J3 = -2.5324100000000000e-6
DE440_TECH_COMMENTS_EARTH_J4 = -1.6198980000000001e-6
DE440_TECH_COMMENTS_EARTH_J5 = -2.2773450000000001e-7
DE440_TECH_COMMENTS_ZONALS = (
    (2, DE440_TECH_COMMENTS_EARTH_J2),
    (3, DE440_TECH_COMMENTS_EARTH_J3),
    (4, DE440_TECH_COMMENTS_EARTH_J4),
    (5, DE440_TECH_COMMENTS_EARTH_J5),
)
MODEL_OUTPUT = "MODEL_OUTPUT"


class EarthZonalError(ValueError):
    """An Earth-zonal configuration or evaluation violated its contract."""


class EarthZonalDependencyError(EarthZonalError):
    """The explicitly requested NumPy force runtime is unavailable."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise EarthZonalError(f"{label} must be bounded nonempty text")
    if value.strip() != value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise EarthZonalError(
            f"{label} must be trimmed printable ASCII without spaces"
        )
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EarthZonalError(f"{label} must be a nonnegative built-in integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise EarthZonalError(f"{label} must be a finite positive built-in float")
    return value


def _coefficients(value: object) -> tuple[tuple[int, float], ...]:
    if type(value) is not tuple or not value or len(value) > 4:
        raise EarthZonalError(
            "zonal_coefficients must be a nonempty exact tuple with at most four terms"
        )
    checked: list[tuple[int, float]] = []
    previous_degree = 1
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise EarthZonalError("each zonal coefficient must be an exact pair")
        degree, coefficient = item
        if type(degree) is not int or degree < 2 or degree > 5:
            raise EarthZonalError("zonal degrees must be exact integers from two to five")
        if degree <= previous_degree:
            raise EarthZonalError("zonal degrees must be unique and strictly increasing")
        if (
            type(coefficient) is not float
            or not math.isfinite(coefficient)
            or coefficient == 0.0
        ):
            raise EarthZonalError("zonal coefficients must be finite nonzero floats")
        checked.append((degree, coefficient))
        previous_degree = degree
    if checked[0][0] != 2:
        raise EarthZonalError("the bounded Earth-zonal model must start with degree two")
    return tuple(checked)


@dataclass(frozen=True, slots=True, eq=False)
class EarthZonalPairForce:
    """Immutable configuration for one axisymmetric Earth-zonal pair."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    zonal_coefficients: tuple[tuple[int, float], ...]
    reference_radius_metres: float
    pole_policy: PreparedEarthPolePolicy
    model_id: str = EARTH_ZONAL_PAIR_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        source_id = _identifier(self.source_id, "source_id")
        target_id = _identifier(self.target_id, "target_id")
        if source_id != "EARTH":
            raise EarthZonalError("the Earth-zonal source_id must be exact EARTH")
        if source_id == target_id:
            raise EarthZonalError("source_id and target_id must be distinct")
        source_index = _index(self.source_index, "source_index")
        target_index = _index(self.target_index, "target_index")
        if source_index == target_index:
            raise EarthZonalError("source_index and target_index must be distinct")
        _coefficients(self.zonal_coefficients)
        _positive_float(self.reference_radius_metres, "reference_radius_metres")
        if type(self.pole_policy) is not PreparedEarthPolePolicy:
            raise EarthZonalError("pole_policy must be a prepared Earth-pole policy")
        if self.model_id != EARTH_ZONAL_PAIR_MODEL_ID:
            raise EarthZonalError("model_id is not the exact Earth-zonal pair model")
        if self.evidence_class != MODEL_OUTPUT:
            raise EarthZonalError("Earth-zonal output must remain MODEL_OUTPUT")
        for observed, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(observed) is not bool or observed:
                raise EarthZonalError(f"{label} must be exact false")

    @property
    def maximum_degree(self) -> int:
        return self.zonal_coefficients[-1][0]

    @property
    def pole_mode(self) -> str:
        return self.pole_policy.mode


def prepare_earth_zonal_pair_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    zonal_coefficients: tuple[tuple[int, float], ...] = (
        DE440_TECH_COMMENTS_ZONALS
    ),
    reference_radius_metres: float = (
        DE440_TECH_COMMENTS_EARTH_REFERENCE_RADIUS_METRES
    ),
    pole_policy: PreparedEarthPolePolicy | None = None,
) -> EarthZonalPairForce:
    """Prepare one explicit Earth-zonal correction through degree five."""

    if pole_policy is None:
        pole_policy = prepare_earth_pole_policy(FIXED_J2000_POSITIVE_Z)
    return EarthZonalPairForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        zonal_coefficients=_coefficients(zonal_coefficients),
        reference_radius_metres=reference_radius_metres,
        pole_policy=pole_policy,
    )


def _legendre_and_derivative(degree: int, axial_ratio: float) -> tuple[float, float]:
    """Return unnormalized P_l(s) and dP_l/ds for registered degrees."""

    s = axial_ratio
    s2 = s * s
    if degree == 2:
        return (0.5 * (3.0 * s2 - 1.0), 3.0 * s)
    if degree == 3:
        return (0.5 * (5.0 * s2 * s - 3.0 * s), 0.5 * (15.0 * s2 - 3.0))
    if degree == 4:
        s4 = s2 * s2
        return (
            0.125 * (35.0 * s4 - 30.0 * s2 + 3.0),
            0.125 * (140.0 * s2 * s - 60.0 * s),
        )
    if degree == 5:
        s3 = s2 * s
        s4 = s2 * s2
        s5 = s4 * s
        return (
            0.125 * (63.0 * s5 - 70.0 * s3 + 15.0 * s),
            0.125 * (315.0 * s4 - 210.0 * s2 + 15.0),
        )
    raise EarthZonalError("unregistered zonal degree")


def evaluate_earth_zonal_pair_correction(
    force: EarthZonalPairForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    gravitational_parameters: object,
    elapsed_time: float = 0.0,
):
    """Return the correction-only acceleration for the configured pair."""

    if type(force) is not EarthZonalPairForce:
        raise EarthZonalError("force must be an exact EarthZonalPairForce")
    if type(body_ids) is not tuple or not body_ids:
        raise EarthZonalError("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        _identifier(body_id, f"body_ids[{index}]")
        for index, body_id in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise EarthZonalError("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise EarthZonalError("configured pair index is outside body_ids")
    if checked_ids[force.source_index] != force.source_id:
        raise EarthZonalError("source index does not match source_id")
    if checked_ids[force.target_index] != force.target_id:
        raise EarthZonalError("target index does not match target_id")

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise EarthZonalDependencyError("Earth-zonal evaluation requires NumPy") from exc
    if type(positions) is not np.ndarray:
        raise EarthZonalError("positions must be an exact NumPy array")
    if (
        positions.dtype != np.dtype("float64")
        or positions.ndim != 2
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise EarthZonalError(
            "positions must be finite C-contiguous float64 with shape (N,3)"
        )
    if type(gravitational_parameters) is not np.ndarray:
        raise EarthZonalError(
            "gravitational_parameters must be an exact NumPy array"
        )
    if (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or np.any(gravitational_parameters < 0.0)
    ):
        raise EarthZonalError(
            "gravitational_parameters must be finite nonnegative C-contiguous float64"
        )
    source_gm = float(gravitational_parameters[force.source_index])
    target_gm = float(gravitational_parameters[force.target_index])
    if source_gm <= 0.0:
        raise EarthZonalError("the zonal source must have positive GM")

    try:
        pole_values = force.pole_policy.raw_pole_at_elapsed_time(elapsed_time)
    except ValueError as exc:
        raise EarthZonalError(str(exc)) from exc
    pole = np.ascontiguousarray(pole_values, dtype=np.float64)
    pole_norm = float(np.linalg.norm(pole))
    if (
        pole.shape != (3,)
        or not np.all(np.isfinite(pole))
        or not math.isfinite(pole_norm)
        or pole_norm <= 0.0
    ):
        raise EarthZonalError("prepared pole is not a finite nonzero three-vector")
    unit_pole = np.ascontiguousarray(pole / pole_norm, dtype=np.float64)

    relative = positions[force.target_index] - positions[force.source_index]
    radius_squared = float(np.dot(relative, relative))
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise EarthZonalError("source and target must have positive finite separation")
    radius = math.sqrt(radius_squared)
    radius_unit = relative / radius
    axial = float(np.dot(relative, unit_pole))
    axial_ratio = axial / radius

    direct = np.zeros(3, dtype=np.float64)
    for degree, coefficient in force.zonal_coefficients:
        if degree == 2:
            factor = (
                1.5
                * coefficient
                * source_gm
                * force.reference_radius_metres
                * force.reference_radius_metres
                / (radius_squared * radius_squared * radius)
            )
            term = factor * (
                (5.0 * axial * axial / radius_squared - 1.0) * relative
                - 2.0 * axial * unit_pole
            )
        else:
            legendre, derivative = _legendre_and_derivative(degree, axial_ratio)
            scale = (
                source_gm
                * coefficient
                * (force.reference_radius_metres / radius) ** degree
                / radius_squared
            )
            term = scale * (
                ((degree + 1.0) * legendre + axial_ratio * derivative)
                * radius_unit
                - derivative * unit_pole
            )
        direct += term

    correction = np.zeros_like(positions, dtype=np.float64)
    correction[force.target_index] = direct
    correction[force.source_index] = -(target_gm / source_gm) * direct
    if not np.all(np.isfinite(correction)):
        raise EarthZonalError("Earth-zonal correction became nonfinite")
    return np.ascontiguousarray(correction, dtype=np.float64)


__all__ = [
    "DE440_TECH_COMMENTS_EARTH_J2",
    "DE440_TECH_COMMENTS_EARTH_J3",
    "DE440_TECH_COMMENTS_EARTH_J4",
    "DE440_TECH_COMMENTS_EARTH_J5",
    "DE440_TECH_COMMENTS_EARTH_REFERENCE_RADIUS_METRES",
    "DE440_TECH_COMMENTS_ZONALS",
    "EARTH_ZONAL_PAIR_MODEL_ID",
    "EarthZonalDependencyError",
    "EarthZonalError",
    "EarthZonalPairForce",
    "evaluate_earth_zonal_pair_correction",
    "prepare_earth_zonal_pair_force",
]
