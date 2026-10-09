"""Correction-only instantaneous Earth-raised elastic lunar tide.

This additive post-V5 module implements only the first, tidal-distortion term
of Park et al. (2021), Equation 54.  It converts that instantaneous inertia
increment to the unnormalized lunar degree-two coefficients of Equations
30-34 and evaluates the resulting Moon-figure/Earth pair interaction.

The registered model deliberately omits the lunar response delay, spin
distortion, integrated libration/core state, and every Earth-tide term.  It is
therefore a bounded equilibrium-tide diagnostic, not a reproduction of the
complete DE440 lunar model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from . import lunar_figure


LUNAR_INSTANTANEOUS_EARTH_TIDE_MODEL_ID = (
    "solar-system.force.lunar-instantaneous-earth-tide-degree2-pair"
)
DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2 = 2.4190000000000000e-2
DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES = 1_738_000.0
MODEL_OUTPUT = "MODEL_OUTPUT"


class LunarTideError(ValueError):
    """A lunar equilibrium-tide request violated its narrow contract."""


class LunarTideDependencyError(LunarTideError):
    """The explicitly requested NumPy runtime is unavailable."""


class LunarTideOrientationProvider(Protocol):
    """Callable returning the inertial-to-principal-axis rotation matrix."""

    def __call__(self, absolute_et: float) -> object: ...


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise LunarTideError(f"{label} must be bounded nonempty text")
    if value.strip() != value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise LunarTideError(
            f"{label} must be trimmed printable ASCII without spaces"
        )
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise LunarTideError(f"{label} must be a nonnegative built-in integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LunarTideError(f"{label} must be a finite built-in float")
    return value


def _positive_float(value: object, label: str) -> float:
    checked = _finite_float(value, label)
    if checked <= 0.0:
        raise LunarTideError(f"{label} must be positive")
    return checked


def _rotation_matrix(value: object):
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarTideDependencyError(
            "lunar equilibrium-tide evaluation requires NumPy"
        ) from exc
    if type(value) is not np.ndarray:
        raise LunarTideError("orientation provider must return an exact NumPy array")
    if (
        value.dtype != np.dtype("float64")
        or value.shape != (3, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarTideError(
            "orientation matrix must be finite contiguous float64 with shape (3,3)"
        )
    tolerance = 64.0 * float(np.finfo(np.float64).eps)
    if float(np.max(np.abs(value @ value.T - np.eye(3)))) > tolerance:
        raise LunarTideError("orientation matrix must be orthonormal")
    determinant = float(np.linalg.det(value))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > tolerance:
        raise LunarTideError("orientation matrix must be a proper rotation")
    return value


def induced_lunar_degree2_coefficients(
    relative_earth_in_principal_axes: object,
    *,
    moon_gm: float,
    earth_gm: float,
    love_number_k2: float = DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2,
    reference_radius_metres: float = (
        DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
) -> tuple[float, float, float, float, float]:
    """Return the instantaneous unnormalized (J2,C21,S21,C22,S22) increment."""

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarTideDependencyError(
            "lunar equilibrium-tide evaluation requires NumPy"
        ) from exc
    if type(relative_earth_in_principal_axes) is not np.ndarray:
        raise LunarTideError("relative vector must be an exact NumPy array")
    relative = relative_earth_in_principal_axes
    if (
        relative.dtype != np.dtype("float64")
        or relative.shape != (3,)
        or not relative.flags.c_contiguous
        or not np.all(np.isfinite(relative))
    ):
        raise LunarTideError(
            "relative vector must be finite contiguous float64 with shape (3,)"
        )
    moon_gm = _positive_float(moon_gm, "moon_gm")
    earth_gm = _positive_float(earth_gm, "earth_gm")
    love_number_k2 = _positive_float(love_number_k2, "love_number_k2")
    reference_radius_metres = _positive_float(
        reference_radius_metres, "reference_radius_metres"
    )
    x, y, z = (float(value) for value in relative)
    radius_squared = x * x + y * y + z * z
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise LunarTideError("Earth and Moon must be separated")
    radius = math.sqrt(radius_squared)
    amplitude = (
        love_number_k2
        * (earth_gm / moon_gm)
        * (reference_radius_metres / radius) ** 3
    )
    inverse_radius_squared = 1.0 / radius_squared
    coefficients = (
        0.5 * amplitude * (1.0 - 3.0 * z * z * inverse_radius_squared),
        amplitude * x * z * inverse_radius_squared,
        amplitude * y * z * inverse_radius_squared,
        0.25 * amplitude * (x * x - y * y) * inverse_radius_squared,
        0.5 * amplitude * x * y * inverse_radius_squared,
    )
    if not all(math.isfinite(value) for value in coefficients):
        raise LunarTideError("induced lunar degree-two coefficients became nonfinite")
    return coefficients


@dataclass(frozen=True, slots=True, eq=False)
class LunarInstantaneousEarthTideForce:
    """Immutable correction for the Earth-raised equilibrium tide on the Moon."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    absolute_start_et: float
    orientation_provider: LunarTideOrientationProvider
    love_number_k2: float = DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2
    reference_radius_metres: float = (
        DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    )
    response_delay_days: float = 0.0
    include_spin_distortion: bool = False
    orientation_frame: str = "MOON_PA_DE440"
    inertial_frame: str = "J2000"
    model_id: str = LUNAR_INSTANTANEOUS_EARTH_TIDE_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if _identifier(self.source_id, "source_id") != "MOON":
            raise LunarTideError("source_id must be exact MOON")
        if _identifier(self.target_id, "target_id") != "EARTH":
            raise LunarTideError("target_id must be exact EARTH")
        if _index(self.source_index, "source_index") == _index(
            self.target_index, "target_index"
        ):
            raise LunarTideError("source and target indices must differ")
        _finite_float(self.absolute_start_et, "absolute_start_et")
        _positive_float(self.love_number_k2, "love_number_k2")
        _positive_float(self.reference_radius_metres, "reference_radius_metres")
        if type(self.response_delay_days) is not float or self.response_delay_days != 0.0:
            raise LunarTideError("the registered equilibrium tide requires zero delay")
        if type(self.include_spin_distortion) is not bool or self.include_spin_distortion:
            raise LunarTideError("the registered equilibrium tide excludes spin distortion")
        if not callable(self.orientation_provider):
            raise LunarTideError("orientation_provider must be callable")
        if self.orientation_frame != "MOON_PA_DE440" or self.inertial_frame != "J2000":
            raise LunarTideError("the registered frame pair changed")
        if self.model_id != LUNAR_INSTANTANEOUS_EARTH_TIDE_MODEL_ID:
            raise LunarTideError("model_id changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarTideError("output must remain MODEL_OUTPUT")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarTideError(f"{label} must be exact false")


def prepare_lunar_instantaneous_earth_tide_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    absolute_start_et: float,
    orientation_provider: LunarTideOrientationProvider,
    love_number_k2: float = DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2,
    reference_radius_metres: float = (
        DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
) -> LunarInstantaneousEarthTideForce:
    """Prepare the registered zero-delay, no-spin lunar equilibrium tide."""

    return LunarInstantaneousEarthTideForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        absolute_start_et=absolute_start_et,
        orientation_provider=orientation_provider,
        love_number_k2=love_number_k2,
        reference_radius_metres=reference_radius_metres,
    )


def evaluate_lunar_instantaneous_earth_tide_correction(
    force: LunarInstantaneousEarthTideForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    gravitational_parameters: object,
    elapsed_time: float = 0.0,
):
    """Return the correction-only acceleration for the registered pair."""

    if type(force) is not LunarInstantaneousEarthTideForce:
        raise LunarTideError(
            "force must be an exact LunarInstantaneousEarthTideForce"
        )
    if type(body_ids) is not tuple or not body_ids:
        raise LunarTideError("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        _identifier(value, f"body_ids[{index}]")
        for index, value in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise LunarTideError("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise LunarTideError("configured pair index is outside body_ids")
    if checked_ids[force.source_index] != "MOON" or checked_ids[force.target_index] != "EARTH":
        raise LunarTideError("configured indices do not match MOON and EARTH")
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarTideDependencyError(
            "lunar equilibrium-tide evaluation requires NumPy"
        ) from exc
    if type(positions) is not np.ndarray or (
        positions.dtype != np.dtype("float64")
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise LunarTideError(
            "positions must be finite contiguous float64 with shape (body_count,3)"
        )
    if type(gravitational_parameters) is not np.ndarray or (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or not np.all(gravitational_parameters > 0.0)
    ):
        raise LunarTideError(
            "gravitational_parameters must be finite positive contiguous float64"
        )
    elapsed_time = _finite_float(elapsed_time, "elapsed_time")
    absolute_et = force.absolute_start_et + elapsed_time
    if not math.isfinite(absolute_et):
        raise LunarTideError("absolute orientation epoch became nonfinite")
    rotation = _rotation_matrix(force.orientation_provider(absolute_et))
    relative_inertial = np.ascontiguousarray(
        positions[force.target_index] - positions[force.source_index],
        dtype=np.float64,
    )
    relative_body = np.ascontiguousarray(rotation @ relative_inertial, dtype=np.float64)
    moon_gm = float(gravitational_parameters[force.source_index])
    earth_gm = float(gravitational_parameters[force.target_index])
    coefficients = induced_lunar_degree2_coefficients(
        relative_body,
        moon_gm=moon_gm,
        earth_gm=earth_gm,
        love_number_k2=force.love_number_k2,
        reference_radius_metres=force.reference_radius_metres,
    )
    earth_body_acceleration = lunar_figure._degree2_body_acceleration(
        relative_body,
        moon_gm,
        force.reference_radius_metres,
        coefficients,
    )
    earth_inertial_acceleration = np.ascontiguousarray(
        rotation.T @ earth_body_acceleration, dtype=np.float64
    )
    result = np.zeros_like(positions)
    result[force.target_index] += earth_inertial_acceleration
    result[force.source_index] -= (
        earth_gm / moon_gm
    ) * earth_inertial_acceleration
    if not np.all(np.isfinite(result)):
        raise LunarTideError("lunar equilibrium-tide correction became nonfinite")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2",
    "DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES",
    "LUNAR_INSTANTANEOUS_EARTH_TIDE_MODEL_ID",
    "LunarInstantaneousEarthTideForce",
    "LunarTideDependencyError",
    "LunarTideError",
    "LunarTideOrientationProvider",
    "evaluate_lunar_instantaneous_earth_tide_correction",
    "induced_lunar_degree2_coefficients",
    "prepare_lunar_instantaneous_earth_tide_force",
]
