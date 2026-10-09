"""Correction-only lunar degree-two spin-distortion surrogate.

This additive post-V5 module implements only the second, spin-distortion
matrix in Park et al. (2021), Equation 54.  The lunar mantle angular velocity
is supplied in ``MOON_PA_DE440`` at ``t - TAUM`` and converted to unnormalized
degree-two coefficients through Equations 30--34.  The resulting body-fixed
coefficients are evaluated at the current Earth--Moon geometry.

The caller owns the delayed orientation history.  This module does not
integrate lunar libration or core rotation and does not reproduce the complete
DE440 lunar force model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import lunar_figure
from . import lunar_tide
from . import lunar_tide_delay


LUNAR_SPIN_DISTORTION_MODEL_ID = (
    "solar-system.force.lunar-spin-distortion-degree2-pair"
)
JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND = 2.6616995e-6
MODEL_OUTPUT = lunar_tide.MODEL_OUTPUT


class LunarSpinDistortionError(lunar_tide.LunarTideError):
    """A lunar spin-distortion request violated its narrow contract."""


class LunarSpinDistortionDependencyError(LunarSpinDistortionError):
    """The explicitly required numerical runtime is unavailable."""


def _vector3(value: object, label: str):
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarSpinDistortionDependencyError(
            "lunar spin-distortion evaluation requires NumPy"
        ) from exc
    if type(value) is not np.ndarray or (
        value.dtype != np.dtype("float64")
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarSpinDistortionError(
            f"{label} must be finite contiguous float64 with shape (3,)"
        )
    return value


def mantle_angular_velocity_from_state_transform(state_transform: object):
    """Extract target-frame angular velocity from an inertial-to-body xform.

    For ``r_body = R r_inertial``, a fixed inertial vector obeys
    ``d(r_body)/dt = -omega_body x r_body``.  Therefore
    ``[omega_body]x = -dR/dt R.T``.  The returned vector has the reciprocal
    time unit used by the state transform (seconds for SPICE ``sxform``).
    """

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarSpinDistortionDependencyError(
            "lunar spin-distortion evaluation requires NumPy"
        ) from exc
    if type(state_transform) is not np.ndarray or (
        state_transform.dtype != np.dtype("float64")
        or state_transform.shape != (6, 6)
        or not state_transform.flags.c_contiguous
        or not np.all(np.isfinite(state_transform))
    ):
        raise LunarSpinDistortionError(
            "state_transform must be finite contiguous float64 with shape (6,6)"
        )
    rotation = np.ascontiguousarray(state_transform[:3, :3], dtype=np.float64)
    derivative = np.ascontiguousarray(state_transform[3:, :3], dtype=np.float64)
    if not np.array_equal(state_transform[:3, 3:], np.zeros((3, 3), dtype=np.float64)):
        raise LunarSpinDistortionError("state transform upper-right block changed")
    if not np.array_equal(state_transform[3:, 3:], rotation):
        raise LunarSpinDistortionError("state transform rotation blocks differ")
    lunar_tide._rotation_matrix(rotation)
    cross_matrix = np.ascontiguousarray(-(derivative @ rotation.T), dtype=np.float64)
    tolerance = 128.0 * float(np.finfo(np.float64).eps) * max(
        1.0, float(np.max(np.abs(cross_matrix)))
    )
    if float(np.max(np.abs(cross_matrix + cross_matrix.T))) > tolerance:
        raise LunarSpinDistortionError("state transform derivative is not rotational")
    omega = np.ascontiguousarray(
        [cross_matrix[2, 1], cross_matrix[0, 2], cross_matrix[1, 0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(omega)):
        raise LunarSpinDistortionError("mantle angular velocity became nonfinite")
    return omega


def spin_distortion_degree2_coefficients(
    mantle_angular_velocity_body: object,
    *,
    moon_gm: float,
    mean_motion_radians_per_second: float = (
        JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND
    ),
    love_number_k2: float = lunar_tide.DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2,
    reference_radius_metres: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
) -> tuple[float, float, float, float, float]:
    """Return ``(J2,C21,S21,C22,S22)`` from Eq. 54 spin distortion."""

    omega = _vector3(mantle_angular_velocity_body, "mantle_angular_velocity_body")
    moon_gm = lunar_tide._positive_float(moon_gm, "moon_gm")
    mean_motion = lunar_tide._positive_float(
        mean_motion_radians_per_second, "mean_motion_radians_per_second"
    )
    love_number = lunar_tide._positive_float(love_number_k2, "love_number_k2")
    radius = lunar_tide._positive_float(
        reference_radius_metres, "reference_radius_metres"
    )
    wx, wy, wz = (float(omega[0]), float(omega[1]), float(omega[2]))
    omega_squared = wx * wx + wy * wy + wz * wz
    n_squared = mean_motion * mean_motion
    qxx = wx * wx - (omega_squared - n_squared) / 3.0
    qyy = wy * wy - (omega_squared - n_squared) / 3.0
    qzz = wz * wz - (omega_squared + 2.0 * n_squared) / 3.0
    qxy = wx * wy
    qxz = wx * wz
    qyz = wy * wz
    trace = qxx + qyy + qzz
    scale_q = max(abs(qxx), abs(qyy), abs(qzz), n_squared)
    if abs(trace) > 16.0 * math.ulp(scale_q):
        raise LunarSpinDistortionError("spin-distortion tensor lost tracelessness")
    inertia_scale = love_number * radius * radius * radius / (3.0 * moon_gm)
    coefficients = (
        inertia_scale * (qzz - 0.5 * (qxx + qyy)),
        -inertia_scale * qxz,
        -inertia_scale * qyz,
        0.25 * inertia_scale * (qyy - qxx),
        -0.5 * inertia_scale * qxy,
    )
    if not all(math.isfinite(value) for value in coefficients):
        raise LunarSpinDistortionError("spin-distortion coefficients became nonfinite")
    return coefficients


@dataclass(frozen=True, slots=True, eq=False)
class LunarSpinDistortionForce:
    """Immutable parameters for the registered Moon--Earth correction."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    absolute_start_et: float
    mean_motion_radians_per_second: float = (
        JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND
    )
    love_number_k2: float = lunar_tide.DE440_TECH_COMMENTS_LUNAR_LOVE_NUMBER_K2
    reference_radius_metres: float = (
        lunar_tide.DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    )
    response_delay_days: float = (
        lunar_tide_delay.DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
    )
    orientation_frame: str = "MOON_PA_DE440"
    inertial_frame: str = "J2000"
    angular_velocity_source: str = "DERIVATIVE_OF_RETAINED_MOON_PA_DE440_XFORM"
    model_id: str = LUNAR_SPIN_DISTORTION_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if lunar_tide._identifier(self.source_id, "source_id") != "MOON":
            raise LunarSpinDistortionError("source_id must be exact MOON")
        if lunar_tide._identifier(self.target_id, "target_id") != "EARTH":
            raise LunarSpinDistortionError("target_id must be exact EARTH")
        if lunar_tide._index(self.source_index, "source_index") == lunar_tide._index(
            self.target_index, "target_index"
        ):
            raise LunarSpinDistortionError("source and target indices must differ")
        lunar_tide._finite_float(self.absolute_start_et, "absolute_start_et")
        lunar_tide._positive_float(
            self.mean_motion_radians_per_second,
            "mean_motion_radians_per_second",
        )
        if (
            self.mean_motion_radians_per_second
            != JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND
        ):
            raise LunarSpinDistortionError("registered lunar mean motion changed")
        lunar_tide._positive_float(self.love_number_k2, "love_number_k2")
        lunar_tide._positive_float(
            self.reference_radius_metres, "reference_radius_metres"
        )
        if (
            type(self.response_delay_days) is not float
            or self.response_delay_days
            != lunar_tide_delay.DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
        ):
            raise LunarSpinDistortionError("registered TAUM delay changed")
        if self.orientation_frame != "MOON_PA_DE440" or self.inertial_frame != "J2000":
            raise LunarSpinDistortionError("registered frame pair changed")
        if self.angular_velocity_source != "DERIVATIVE_OF_RETAINED_MOON_PA_DE440_XFORM":
            raise LunarSpinDistortionError("angular velocity source changed")
        if self.model_id != LUNAR_SPIN_DISTORTION_MODEL_ID:
            raise LunarSpinDistortionError("model_id changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarSpinDistortionError("output must remain MODEL_OUTPUT")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarSpinDistortionError(f"{label} must be exact false")

    @property
    def response_delay_seconds(self) -> float:
        return self.response_delay_days * lunar_tide_delay.SECONDS_PER_DAY


def prepare_lunar_spin_distortion_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    absolute_start_et: float,
) -> LunarSpinDistortionForce:
    """Prepare the registered delayed spin-distortion pair correction."""

    return LunarSpinDistortionForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        absolute_start_et=absolute_start_et,
    )


def evaluate_lunar_spin_distortion_correction(
    force: LunarSpinDistortionForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    current_inertial_to_principal_axes: object,
    delayed_mantle_angular_velocity_body: object,
    gravitational_parameters: object,
    elapsed_time: float,
    delayed_elapsed_time: float,
):
    """Return the correction-only acceleration for Equation 54 spin distortion."""

    if type(force) is not LunarSpinDistortionForce:
        raise LunarSpinDistortionError(
            "force must be an exact LunarSpinDistortionForce"
        )
    if type(body_ids) is not tuple or not body_ids:
        raise LunarSpinDistortionError("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        lunar_tide._identifier(value, f"body_ids[{index}]")
        for index, value in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise LunarSpinDistortionError("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise LunarSpinDistortionError("configured pair index is outside body_ids")
    if (
        checked_ids[force.source_index] != "MOON"
        or checked_ids[force.target_index] != "EARTH"
    ):
        raise LunarSpinDistortionError("configured indices do not match MOON and EARTH")
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarSpinDistortionDependencyError(
            "lunar spin-distortion evaluation requires NumPy"
        ) from exc
    if type(positions) is not np.ndarray or (
        positions.dtype != np.dtype("float64")
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise LunarSpinDistortionError(
            "positions must be finite contiguous float64 with shape (body_count,3)"
        )
    if type(gravitational_parameters) is not np.ndarray or (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or not np.all(gravitational_parameters > 0.0)
    ):
        raise LunarSpinDistortionError(
            "gravitational_parameters must be finite positive contiguous float64"
        )
    current_rotation = lunar_tide._rotation_matrix(
        current_inertial_to_principal_axes
    )
    delayed_omega = _vector3(
        delayed_mantle_angular_velocity_body,
        "delayed_mantle_angular_velocity_body",
    )
    elapsed = lunar_tide._finite_float(elapsed_time, "elapsed_time")
    delayed_elapsed = lunar_tide._finite_float(
        delayed_elapsed_time, "delayed_elapsed_time"
    )
    if delayed_elapsed.hex() != (elapsed - force.response_delay_seconds).hex():
        raise LunarSpinDistortionError(
            "delayed_elapsed_time must equal elapsed_time minus exact TAUM"
        )
    if not math.isfinite(force.absolute_start_et + elapsed) or not math.isfinite(
        force.absolute_start_et + delayed_elapsed
    ):
        raise LunarSpinDistortionError("absolute spin-distortion epoch became nonfinite")

    current_relative_inertial = np.ascontiguousarray(
        positions[force.target_index] - positions[force.source_index],
        dtype=np.float64,
    )
    current_relative_body = np.ascontiguousarray(
        current_rotation @ current_relative_inertial, dtype=np.float64
    )
    moon_gm = float(gravitational_parameters[force.source_index])
    earth_gm = float(gravitational_parameters[force.target_index])
    coefficients = spin_distortion_degree2_coefficients(
        delayed_omega,
        moon_gm=moon_gm,
        mean_motion_radians_per_second=force.mean_motion_radians_per_second,
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
        raise LunarSpinDistortionError("spin-distortion correction became nonfinite")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND",
    "LUNAR_SPIN_DISTORTION_MODEL_ID",
    "LunarSpinDistortionDependencyError",
    "LunarSpinDistortionError",
    "LunarSpinDistortionForce",
    "MODEL_OUTPUT",
    "evaluate_lunar_spin_distortion_correction",
    "mantle_angular_velocity_from_state_transform",
    "prepare_lunar_spin_distortion_force",
    "spin_distortion_degree2_coefficients",
]
