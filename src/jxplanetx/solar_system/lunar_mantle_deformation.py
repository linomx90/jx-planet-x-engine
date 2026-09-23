"""Complete delayed lunar mantle-deformation balance primitives.

This opt-in research component promotes the internally consistent deformation
package used by the retained JX lunar balance diagnostic.  It evaluates the
Earth-raised tidal inertia, spin-distortion inertia, both analytic time
derivatives, the deforming mantle inertia, and the corresponding exterior
gravity figure.  A general symmetric-tensor quadrupole interaction supplies
force and torque from the same potential.

The caller owns the delay history.  In particular, the delayed lunar
orientation, mantle rate and acceleration, and Earth--Moon relative state
must all describe the same epoch ``t - TAUM``.  This module neither invents a
prehistory nor queries an ephemeris.  It is screening-only research code, not
an exact DE440 generator, an LLR model, or a registered production force.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import lunar_coupled
from . import lunar_spin_distortion
from . import lunar_tide_delay


LUNAR_MANTLE_DEFORMATION_MODEL_ID = (
    "solar-system.rotation.lunar-complete-delayed-mantle-deformation.experimental.v1"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MODEL_OUTPUT = "MODEL_OUTPUT"


class LunarMantleDeformationError(ValueError):
    """A complete mantle-deformation request violated its contract."""


class LunarMantleDeformationDependencyError(LunarMantleDeformationError):
    """The explicitly required numerical runtime is unavailable."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarMantleDeformationDependencyError(
            "lunar mantle-deformation evaluation requires NumPy"
        ) from exc
    return np


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LunarMantleDeformationError(
            f"{label} must be a finite built-in float"
        )
    if positive and value <= 0.0:
        raise LunarMantleDeformationError(f"{label} must be positive")
    return value


def _positive_tuple3(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise LunarMantleDeformationError(
            f"{label} must be an exact three-float tuple"
        )
    checked = tuple(
        _finite_float(item, f"{label}[{index}]", positive=True)
        for index, item in enumerate(value)
    )
    return checked  # type: ignore[return-value]


def _vector(value: object, shape: tuple[int, ...], label: str):
    np = _numpy()
    if type(value) is not np.ndarray or (
        value.dtype != np.dtype("float64")
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarMantleDeformationError(
            f"{label} must be finite contiguous float64 with shape {shape}"
        )
    return value


def _symmetric_tensor(value: object, label: str, *, positive_definite: bool):
    np = _numpy()
    tensor = _vector(value, (3, 3), label)
    scale = max(float(np.max(np.abs(tensor))), np.finfo(np.float64).tiny)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if float(np.max(np.abs(tensor - tensor.T))) > tolerance:
        raise LunarMantleDeformationError(f"{label} must be symmetric")
    if positive_definite and float(np.min(np.linalg.eigvalsh(tensor))) <= 0.0:
        raise LunarMantleDeformationError(f"{label} must be positive definite")
    return tensor


@dataclass(frozen=True, slots=True, eq=False)
class LunarMantleDeformationParameters:
    """Explicit parameters and asymmetric inertia ownership for the package."""

    undistorted_total_inertia_over_mr2: tuple[float, float, float]
    fixed_core_inertia_over_mr2: tuple[float, float, float]
    undistorted_mantle_inertia_over_mr2: tuple[float, float, float]
    earth_gm_km3_s2: float
    moon_gm_km3_s2: float
    love_number_k2: float
    lunar_reference_radius_km: float
    response_delay_days: float = (
        lunar_tide_delay.DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
    )
    mean_motion_radians_per_second: float = (
        lunar_spin_distortion.JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND
    )
    parameter_source_id: str = "CALLER_SUPPLIED_RESEARCH_PARAMETERS"
    model_id: str = LUNAR_MANTLE_DEFORMATION_MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        total = _positive_tuple3(
            self.undistorted_total_inertia_over_mr2,
            "undistorted_total_inertia_over_mr2",
        )
        core = _positive_tuple3(
            self.fixed_core_inertia_over_mr2,
            "fixed_core_inertia_over_mr2",
        )
        mantle = _positive_tuple3(
            self.undistorted_mantle_inertia_over_mr2,
            "undistorted_mantle_inertia_over_mr2",
        )
        if any(mantle[index] + core[index] != total[index] for index in range(3)):
            raise LunarMantleDeformationError(
                "mantle plus fixed core must equal total inertia exactly"
            )
        if not (core[0] == core[1] <= core[2]):
            raise LunarMantleDeformationError(
                "fixed core inertia must be axisymmetric and oblate or spherical"
            )
        for value, label in (
            (self.earth_gm_km3_s2, "earth_gm_km3_s2"),
            (self.moon_gm_km3_s2, "moon_gm_km3_s2"),
            (self.love_number_k2, "love_number_k2"),
            (self.lunar_reference_radius_km, "lunar_reference_radius_km"),
            (self.response_delay_days, "response_delay_days"),
            (self.mean_motion_radians_per_second, "mean_motion_radians_per_second"),
        ):
            _finite_float(value, label, positive=True)
        if (
            self.response_delay_days
            != lunar_tide_delay.DE440_TECH_COMMENTS_LUNAR_RESPONSE_DELAY_DAYS
        ):
            raise LunarMantleDeformationError("retained TAUM value changed")
        if (
            self.mean_motion_radians_per_second
            != lunar_spin_distortion.JPL_HORIZONS_LUNAR_MEAN_MOTION_RADIANS_PER_SECOND
        ):
            raise LunarMantleDeformationError(
                "retained spin-distortion mean-motion convention changed"
            )
        if (
            type(self.parameter_source_id) is not str
            or not self.parameter_source_id
            or len(self.parameter_source_id) > 256
            or self.parameter_source_id.strip() != self.parameter_source_id
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E
                for character in self.parameter_source_id
            )
        ):
            raise LunarMantleDeformationError(
                "parameter_source_id must be bounded printable ASCII without spaces"
            )
        if self.model_id != LUNAR_MANTLE_DEFORMATION_MODEL_ID:
            raise LunarMantleDeformationError("model_id changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise LunarMantleDeformationError("scientific_claim_state changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarMantleDeformationError("evidence_class changed")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarMantleDeformationError(f"{label} must be exact false")

    @property
    def response_delay_seconds(self) -> float:
        return self.response_delay_days * lunar_tide_delay.SECONDS_PER_DAY


@dataclass(frozen=True, slots=True, eq=False)
class LunarMantleDeformationEvaluation:
    """One complete delayed deformation evaluation in mantle axes."""

    delayed_tidal_inertia_over_mr2: object
    delayed_tidal_inertia_rate_over_mr2_per_second: object
    delayed_spin_inertia_over_mr2: object
    delayed_spin_inertia_rate_over_mr2_per_second: object
    mantle_inertia_over_mr2: object
    exterior_figure_inertia_over_mr2: object
    mantle_inertia_rate_over_mr2_per_second: object
    delayed_earth_position_body_km: object
    delayed_earth_velocity_body_km_s: object


@dataclass(frozen=True, slots=True, eq=False)
class LunarTensorQuadrupoleInteraction:
    """One source--Moon force/torque pair from a common tensor potential."""

    source_acceleration_km_s2: object
    moon_acceleration_km_s2: object
    torque_on_mantle_over_mr2: object
    potential_gm_scaled: float


def _coefficients_to_traceless_tensor(
    coefficients: tuple[float, float, float, float, float],
):
    np = _numpy()
    j2, c21, s21, c22, s22 = coefficients
    return np.ascontiguousarray(
        (
            (-j2 / 3.0 - 2.0 * c22, -2.0 * s22, -c21),
            (-2.0 * s22, -j2 / 3.0 + 2.0 * c22, -s21),
            (-c21, -s21, 2.0 * j2 / 3.0),
        ),
        dtype=np.float64,
    )


def _tidal_tensor_and_rate(
    quaternion: object,
    body_omega: object,
    earth_relative_state_inertial: object,
    parameters: LunarMantleDeformationParameters,
):
    np = _numpy()
    q = _vector(quaternion, (4,), "delayed_quaternion_body_to_inertial")
    omega = _vector(
        body_omega, (3,), "delayed_mantle_angular_velocity_body_s"
    )
    state = _vector(
        earth_relative_state_inertial,
        (6,),
        "delayed_earth_relative_state_inertial_km_km_s",
    )
    rotation = lunar_coupled._body_to_inertial(q)
    position = np.ascontiguousarray(rotation.T @ state[:3], dtype=np.float64)
    velocity = np.ascontiguousarray(
        rotation.T @ state[3:] - np.cross(omega, position), dtype=np.float64
    )
    radius_squared = float(np.dot(position, position))
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise LunarMantleDeformationError("delayed Earth and Moon must be separated")
    radius = math.sqrt(radius_squared)
    radial_rate = float(np.dot(position, velocity))
    identity = np.eye(3, dtype=np.float64)
    quadrupole = (
        np.outer(position, position) - (radius_squared / 3.0) * identity
    )
    quadrupole_rate = (
        np.outer(velocity, position)
        + np.outer(position, velocity)
        - (2.0 * radial_rate / 3.0) * identity
    )
    scale = (
        parameters.love_number_k2
        * (parameters.earth_gm_km3_s2 / parameters.moon_gm_km3_s2)
        * parameters.lunar_reference_radius_km**3
    )
    tensor = np.ascontiguousarray(-scale * quadrupole / radius**5)
    rate = np.ascontiguousarray(
        -scale
        * (
            quadrupole_rate / radius**5
            - 5.0 * radial_rate * quadrupole / radius**7
        )
    )
    tensor[tensor == 0.0] = 0.0
    rate[rate == 0.0] = 0.0
    return tensor, rate, position, velocity


def _spin_tensor_and_rate(
    body_omega: object,
    body_alpha: object,
    parameters: LunarMantleDeformationParameters,
):
    np = _numpy()
    omega = _vector(
        body_omega, (3,), "delayed_mantle_angular_velocity_body_s"
    )
    alpha = _vector(
        body_alpha, (3,), "delayed_mantle_angular_acceleration_body_s2"
    )
    coefficients = lunar_spin_distortion.spin_distortion_degree2_coefficients(
        omega,
        moon_gm=parameters.moon_gm_km3_s2 * 1.0e9,
        mean_motion_radians_per_second=parameters.mean_motion_radians_per_second,
        love_number_k2=parameters.love_number_k2,
        reference_radius_metres=parameters.lunar_reference_radius_km * 1_000.0,
    )
    tensor = _coefficients_to_traceless_tensor(coefficients)
    dot = float(np.dot(omega, alpha))
    wx, wy, wz = (float(value) for value in omega)
    ax, ay, az = (float(value) for value in alpha)
    dqxx = 2.0 * wx * ax - 2.0 * dot / 3.0
    dqyy = 2.0 * wy * ay - 2.0 * dot / 3.0
    dqzz = 2.0 * wz * az - 2.0 * dot / 3.0
    dqxy = ax * wy + wx * ay
    dqxz = ax * wz + wx * az
    dqyz = ay * wz + wy * az
    scale = (
        parameters.love_number_k2
        * parameters.lunar_reference_radius_km**3
        / (3.0 * parameters.moon_gm_km3_s2)
    )
    rate_coefficients = (
        scale * (dqzz - 0.5 * (dqxx + dqyy)),
        -scale * dqxz,
        -scale * dqyz,
        0.25 * scale * (dqyy - dqxx),
        -0.5 * scale * dqxy,
    )
    rate = _coefficients_to_traceless_tensor(rate_coefficients)
    return tensor, rate


def evaluate_lunar_mantle_deformation(
    parameters: LunarMantleDeformationParameters,
    *,
    delayed_quaternion_body_to_inertial: object,
    delayed_mantle_angular_velocity_body_s: object,
    delayed_mantle_angular_acceleration_body_s2: object,
    delayed_earth_relative_state_inertial_km_km_s: object,
) -> LunarMantleDeformationEvaluation:
    """Evaluate the complete time-variable mantle-deformation package."""

    if type(parameters) is not LunarMantleDeformationParameters:
        raise LunarMantleDeformationError(
            "parameters must be exact LunarMantleDeformationParameters"
        )
    np = _numpy()
    tide, tide_rate, earth_body, earth_velocity_body = _tidal_tensor_and_rate(
        delayed_quaternion_body_to_inertial,
        delayed_mantle_angular_velocity_body_s,
        delayed_earth_relative_state_inertial_km_km_s,
        parameters,
    )
    spin, spin_rate = _spin_tensor_and_rate(
        delayed_mantle_angular_velocity_body_s,
        delayed_mantle_angular_acceleration_body_s2,
        parameters,
    )
    mantle = np.ascontiguousarray(
        np.diag(parameters.undistorted_mantle_inertia_over_mr2) + tide + spin
    )
    figure = np.ascontiguousarray(
        np.diag(parameters.undistorted_total_inertia_over_mr2) + tide + spin
    )
    rate = np.ascontiguousarray(tide_rate + spin_rate)
    for value, label, positive in (
        (tide, "delayed tidal inertia", False),
        (tide_rate, "delayed tidal inertia rate", False),
        (spin, "delayed spin inertia", False),
        (spin_rate, "delayed spin inertia rate", False),
        (mantle, "mantle inertia", True),
        (figure, "exterior figure inertia", True),
        (rate, "mantle inertia rate", False),
    ):
        _symmetric_tensor(value, label, positive_definite=positive)
    return LunarMantleDeformationEvaluation(
        delayed_tidal_inertia_over_mr2=tide,
        delayed_tidal_inertia_rate_over_mr2_per_second=tide_rate,
        delayed_spin_inertia_over_mr2=spin,
        delayed_spin_inertia_rate_over_mr2_per_second=spin_rate,
        mantle_inertia_over_mr2=mantle,
        exterior_figure_inertia_over_mr2=figure,
        mantle_inertia_rate_over_mr2_per_second=rate,
        delayed_earth_position_body_km=earth_body,
        delayed_earth_velocity_body_km_s=earth_velocity_body,
    )


def lunar_gravity_gradient_torque_over_mr2(
    quaternion_body_to_inertial: object,
    source_minus_moon_position_inertial_km: object,
    exterior_figure_inertia_over_mr2: object,
    *,
    source_gm_km3_s2: float,
):
    """Return the point-source torque on the full deforming lunar figure."""

    np = _numpy()
    q = _vector(quaternion_body_to_inertial, (4,), "quaternion_body_to_inertial")
    separation = _vector(
        source_minus_moon_position_inertial_km,
        (3,),
        "source_minus_moon_position_inertial_km",
    )
    tensor = _symmetric_tensor(
        exterior_figure_inertia_over_mr2,
        "exterior_figure_inertia_over_mr2",
        positive_definite=True,
    )
    gm = _finite_float(source_gm_km3_s2, "source_gm_km3_s2", positive=True)
    radius = float(np.linalg.norm(separation))
    if not math.isfinite(radius) or radius <= 0.0:
        raise LunarMantleDeformationError("source and Moon must be separated")
    direction = np.ascontiguousarray(
        lunar_coupled._body_to_inertial(q).T @ (separation / radius)
    )
    torque = np.ascontiguousarray(
        3.0 * gm / radius**3 * np.cross(direction, tensor @ direction)
    )
    torque[torque == 0.0] = 0.0
    return torque


def evaluate_lunar_tensor_quadrupole_interaction(
    quaternion_body_to_inertial: object,
    source_position_inertial_km: object,
    moon_position_inertial_km: object,
    exterior_figure_inertia_over_mr2: object,
    *,
    source_gm_km3_s2: float,
    moon_gm_km3_s2: float,
    lunar_reference_radius_km: float,
) -> LunarTensorQuadrupoleInteraction:
    """Evaluate force, reaction and torque from one common tensor potential."""

    np = _numpy()
    q = _vector(quaternion_body_to_inertial, (4,), "quaternion_body_to_inertial")
    source = _vector(
        source_position_inertial_km, (3,), "source_position_inertial_km"
    )
    moon = _vector(moon_position_inertial_km, (3,), "moon_position_inertial_km")
    tensor = _symmetric_tensor(
        exterior_figure_inertia_over_mr2,
        "exterior_figure_inertia_over_mr2",
        positive_definite=True,
    )
    source_gm = _finite_float(
        source_gm_km3_s2, "source_gm_km3_s2", positive=True
    )
    moon_gm = _finite_float(moon_gm_km3_s2, "moon_gm_km3_s2", positive=True)
    radius_km = _finite_float(
        lunar_reference_radius_km, "lunar_reference_radius_km", positive=True
    )
    separation = np.ascontiguousarray(source - moon)
    distance = float(np.linalg.norm(separation))
    if not math.isfinite(distance) or distance <= 0.0:
        raise LunarMantleDeformationError("source and Moon must be separated")
    rotation = lunar_coupled._body_to_inertial(q)
    relative_body = np.ascontiguousarray(rotation.T @ separation)
    radius_squared = distance * distance
    trace = float(np.trace(tensor))
    quadratic = float(relative_body @ tensor @ relative_body)
    figure_polynomial = 3.0 * quadratic - radius_squared * trace
    body_force = (
        moon_gm
        * radius_km**2
        / distance**5
        * (
            (trace * np.eye(3, dtype=np.float64) - 3.0 * tensor) @ relative_body
            + 2.5 * figure_polynomial / radius_squared * relative_body
        )
    )
    source_acceleration = np.ascontiguousarray(rotation @ body_force)
    moon_acceleration = np.ascontiguousarray(
        -(source_gm / moon_gm) * source_acceleration
    )
    torque = lunar_gravity_gradient_torque_over_mr2(
        q,
        separation,
        tensor,
        source_gm_km3_s2=source_gm,
    )
    potential = (
        source_gm
        * moon_gm
        * radius_km**2
        / (2.0 * distance**5)
        * figure_polynomial
    )
    if (
        not np.all(np.isfinite(source_acceleration))
        or not np.all(np.isfinite(moon_acceleration))
        or not np.all(np.isfinite(torque))
        or not math.isfinite(potential)
    ):
        raise LunarMantleDeformationError(
            "tensor quadrupole interaction became nonfinite"
        )
    return LunarTensorQuadrupoleInteraction(
        source_acceleration_km_s2=source_acceleration,
        moon_acceleration_km_s2=moon_acceleration,
        torque_on_mantle_over_mr2=torque,
        potential_gm_scaled=potential,
    )


__all__ = [
    "LUNAR_MANTLE_DEFORMATION_MODEL_ID",
    "LunarMantleDeformationDependencyError",
    "LunarMantleDeformationError",
    "LunarMantleDeformationEvaluation",
    "LunarMantleDeformationParameters",
    "LunarTensorQuadrupoleInteraction",
    "MODEL_OUTPUT",
    "SCIENTIFIC_CLAIM_STATE",
    "evaluate_lunar_mantle_deformation",
    "evaluate_lunar_tensor_quadrupole_interaction",
    "lunar_gravity_gradient_torque_over_mr2",
]
