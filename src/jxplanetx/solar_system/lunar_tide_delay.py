"""Correction-only delayed Earth-raised elastic lunar tide.

This additive post-V5 module isolates the response-delay part of the first
tidal-distortion term in Park et al. (2021), Equation 54.  The Earth-relative
vector and lunar principal-axis orientation used to form the induced lunar
degree-two coefficients are evaluated at ``t - TAUM``.  The resulting
body-fixed coefficients are then used with the current Earth-Moon geometry.

The caller owns the delay history and supplies its interpolated past relative
vector plus both rotation matrices.  That boundary is deliberate: a delay
differential equation requires a declared prehistory, which cannot be inferred
from one instantaneous Cartesian state.  This module does not implement spin
distortion, libration, fluid-core dynamics, or the complete DE440 lunar model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import lunar_figure
from . import lunar_tide


LUNAR_DELAYED_EARTH_TIDE_MODEL_ID = (
    "solar-system.force.lunar-delayed-earth-tide-degree2-pair"
)
DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS = 1.4519388258636395e-1
SECONDS_PER_DAY = 86_400.0
DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_SECONDS = (
    DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS * SECONDS_PER_DAY
)
MODEL_OUTPUT = lunar_tide.MODEL_OUTPUT


class LunarDelayedTideError(lunar_tide.LunarTideError):
    """A delayed lunar-tide request violated its narrow contract."""


class LunarDelayedTideDependencyError(LunarDelayedTideError):
    """The explicitly required numerical runtime is unavailable."""


def _relative_vector(value: object, label: str):
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarDelayedTideDependencyError(
            "delayed lunar-tide evaluation requires NumPy"
        ) from exc
    if type(value) is not np.ndarray:
        raise LunarDelayedTideError(f"{label} must be an exact NumPy array")
    if (
        value.dtype != np.dtype("float64")
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarDelayedTideError(
            f"{label} must be finite contiguous float64 with shape (3,)"
        )
    radius_squared = float(value @ value)
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise LunarDelayedTideError(f"{label} must be nonzero")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class LunarDelayedEarthTideForce:
    """Immutable parameters for the registered delayed lunar tide."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    absolute_start_et: float
    love_number_k2: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2
    )
    reference_radius_metres: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    )
    response_delay_days: float = (
        DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
    )
    include_spin_distortion: bool = False
    orientation_frame: str = "MOON_PA_DE440"
    inertial_frame: str = "J2000"
    model_id: str = LUNAR_DELAYED_EARTH_TIDE_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if lunar_tide._identifier(self.source_id, "source_id") != "MOON":
            raise LunarDelayedTideError("source_id must be exact MOON")
        if lunar_tide._identifier(self.target_id, "target_id") != "EARTH":
            raise LunarDelayedTideError("target_id must be exact EARTH")
        if lunar_tide._index(self.source_index, "source_index") == lunar_tide._index(
            self.target_index, "target_index"
        ):
            raise LunarDelayedTideError("source and target indices must differ")
        lunar_tide._finite_float(self.absolute_start_et, "absolute_start_et")
        lunar_tide._positive_float(self.love_number_k2, "love_number_k2")
        lunar_tide._positive_float(
            self.reference_radius_metres, "reference_radius_metres"
        )
        if (
            type(self.response_delay_days) is not float
            or self.response_delay_days
            != DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
        ):
            raise LunarDelayedTideError(
                "response_delay_days must equal the exact retained TAUM value"
            )
        if type(self.include_spin_distortion) is not bool or self.include_spin_distortion:
            raise LunarDelayedTideError("the registered delayed tide excludes spin distortion")
        if self.orientation_frame != "MOON_PA_DE440" or self.inertial_frame != "J2000":
            raise LunarDelayedTideError("the registered frame pair changed")
        if self.model_id != LUNAR_DELAYED_EARTH_TIDE_MODEL_ID:
            raise LunarDelayedTideError("model_id changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarDelayedTideError("output must remain MODEL_OUTPUT")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarDelayedTideError(f"{label} must be exact false")

    @property
    def response_delay_seconds(self) -> float:
        """Return the exact binary64 days-to-seconds conversion used here."""

        return self.response_delay_days * SECONDS_PER_DAY


def prepare_lunar_delayed_earth_tide_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    absolute_start_et: float,
    love_number_k2: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2
    ),
    reference_radius_metres: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
    response_delay_days: float = (
        DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
    ),
) -> LunarDelayedEarthTideForce:
    """Prepare the registered delayed, no-spin lunar equilibrium tide."""

    return LunarDelayedEarthTideForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        absolute_start_et=absolute_start_et,
        love_number_k2=love_number_k2,
        reference_radius_metres=reference_radius_metres,
        response_delay_days=response_delay_days,
    )


def evaluate_lunar_delayed_earth_tide_correction(
    force: LunarDelayedEarthTideForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    delayed_relative_earth_inertial: object,
    current_inertial_to_principal_axes: object,
    delayed_inertial_to_principal_axes: object,
    gravitational_parameters: object,
    elapsed_time: float,
    delayed_elapsed_time: float,
):
    """Return the delayed tidal correction for the registered Moon-Earth pair.

    ``delayed_relative_earth_inertial`` is Earth minus Moon at
    ``delayed_elapsed_time``.  Its direction is converted to material-frame
    coefficients with the delayed orientation.  Those coefficients are then
    evaluated at the current Earth position using the current orientation.
    """

    if type(force) is not LunarDelayedEarthTideForce:
        raise LunarDelayedTideError(
            "force must be an exact LunarDelayedEarthTideForce"
        )
    if type(body_ids) is not tuple or not body_ids:
        raise LunarDelayedTideError("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        lunar_tide._identifier(value, f"body_ids[{index}]")
        for index, value in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise LunarDelayedTideError("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise LunarDelayedTideError("configured pair index is outside body_ids")
    if (
        checked_ids[force.source_index] != "MOON"
        or checked_ids[force.target_index] != "EARTH"
    ):
        raise LunarDelayedTideError("configured indices do not match MOON and EARTH")
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarDelayedTideDependencyError(
            "delayed lunar-tide evaluation requires NumPy"
        ) from exc
    if type(positions) is not np.ndarray or (
        positions.dtype != np.dtype("float64")
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise LunarDelayedTideError(
            "positions must be finite contiguous float64 with shape (body_count,3)"
        )
    if type(gravitational_parameters) is not np.ndarray or (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or not np.all(gravitational_parameters > 0.0)
    ):
        raise LunarDelayedTideError(
            "gravitational_parameters must be finite positive contiguous float64"
        )
    delayed_relative = _relative_vector(
        delayed_relative_earth_inertial, "delayed_relative_earth_inertial"
    )
    current_rotation = lunar_tide._rotation_matrix(
        current_inertial_to_principal_axes
    )
    delayed_rotation = lunar_tide._rotation_matrix(
        delayed_inertial_to_principal_axes
    )
    elapsed = lunar_tide._finite_float(elapsed_time, "elapsed_time")
    delayed_elapsed = lunar_tide._finite_float(
        delayed_elapsed_time, "delayed_elapsed_time"
    )
    expected_delayed = elapsed - force.response_delay_seconds
    if delayed_elapsed.hex() != expected_delayed.hex():
        raise LunarDelayedTideError(
            "delayed_elapsed_time must equal elapsed_time minus exact TAUM"
        )
    absolute_current = force.absolute_start_et + elapsed
    absolute_delayed = force.absolute_start_et + delayed_elapsed
    if not math.isfinite(absolute_current) or not math.isfinite(absolute_delayed):
        raise LunarDelayedTideError("absolute tide epoch became nonfinite")

    current_relative_inertial = np.ascontiguousarray(
        positions[force.target_index] - positions[force.source_index],
        dtype=np.float64,
    )
    current_relative_body = np.ascontiguousarray(
        current_rotation @ current_relative_inertial, dtype=np.float64
    )
    delayed_relative_body = np.ascontiguousarray(
        delayed_rotation @ delayed_relative, dtype=np.float64
    )
    moon_gm = float(gravitational_parameters[force.source_index])
    earth_gm = float(gravitational_parameters[force.target_index])
    coefficients = lunar_tide.induced_lunar_degree2_coefficients(
        delayed_relative_body,
        moon_gm=moon_gm,
        earth_gm=earth_gm,
        love_number_k2=force.love_number_k2,
        reference_radius_metres=force.reference_radius_metres,
    )
    earth_body_acceleration = lunar_figure._degree2_body_acceleration(
        current_relative_body,
        moon_gm,
        force.reference_radius_metres,
        coefficients,
    )
    earth_inertial_acceleration = np.ascontiguousarray(
        current_rotation.T @ earth_body_acceleration, dtype=np.float64
    )
    result = np.zeros_like(positions)
    result[force.target_index] += earth_inertial_acceleration
    result[force.source_index] -= (
        earth_gm / moon_gm
    ) * earth_inertial_acceleration
    if not np.all(np.isfinite(result)):
        raise LunarDelayedTideError("delayed lunar-tide correction became nonfinite")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS",
    "DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_SECONDS",
    "LUNAR_DELAYED_EARTH_TIDE_MODEL_ID",
    "LunarDelayedEarthTideForce",
    "LunarDelayedTideDependencyError",
    "LunarDelayedTideError",
    "MODEL_OUTPUT",
    "SECONDS_PER_DAY",
    "evaluate_lunar_delayed_earth_tide_correction",
    "prepare_lunar_delayed_earth_tide_force",
]
