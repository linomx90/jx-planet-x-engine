"""Screening-only Sun--Earth--Moon orbit and lunar rotation model.

This additive v2 research boundary advances the Sun, Earth, and Moon as three
simultaneous translational bodies together with the lunar mantle quaternion,
mantle angular velocity, and fluid-core angular velocity.  It includes:

* mutual point-mass gravity among all three bodies;
* static lunar degree-two force and torque from the Earth and the Sun, with
  each source reaction and lunar torque derived from one common potential;
* an axisymmetric Earth-J2 correction on the Earth--Moon orbit, including the
  GM-weighted translational reaction; and
* equal-and-opposite core--mantle boundary torque.

The Earth pole is a caller-supplied, inertially fixed unit direction.  Earth
spin is not a state, so the J2 field does not conserve the full angular-
momentum vector; its component along the fixed symmetry axis is the applicable
conservation diagnostic.  Delayed tides, time-varying inertia, Earth rotation,
other planets, relativity, and an LLR observation model remain omitted.  This
module is unregistered ``SCREENING_ONLY`` research code, not a DE440 model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT, _rkf78_step

from . import lunar_coupled as _v1
from .lunar_fluid_core import (
    LunarFluidCoreError,
    lunar_mantle_core_angular_accelerations,
)


SOLAR_COUPLED_LUNAR_MODEL_ID = (
    "solar-system.experimental.sun-earth-moon-static-figure-mantle-fluid-core-j2.v2"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MODEL_SCOPE = (
    "SUN_EARTH_MOON_MONOPOLES_STATIC_LUNAR_DEGREE2_EARTH_J2_FIXED_POLE_"
    "MANTLE_QUATERNION_AND_FLUID_CORE_RATE"
)
BODY_IDS = ("SUN", "EARTH", "MOON")
SUN_INDEX = 0
EARTH_INDEX = 1
MOON_INDEX = 2
LEDGER_DOMAIN = b"jxplanetx.solar-coupled-lunar.accepted-step-ledger.v2\0"


class SolarCoupledLunarError(ValueError):
    """The v2 screening contract or integration failed."""


class SolarCoupledLunarStepLimitError(SolarCoupledLunarError):
    """The fixed-step integration exhausted its declared work cap."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise SolarCoupledLunarError("solar coupled lunar integration requires NumPy") from exc
    return np


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SolarCoupledLunarError(f"{label} must be a positive finite built-in float")
    return value


def _nonnegative_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise SolarCoupledLunarError(
            f"{label} must be a nonnegative finite built-in float"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SolarCoupledLunarError(f"{label} must be a finite built-in float")
    return value


def _owned_array(value: object, shape: tuple[int, ...], label: str):
    np = _numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("float64")
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise SolarCoupledLunarError(
            f"{label} must be finite C-contiguous NumPy float64 with shape {shape}"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _unit_vector_tuple(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise SolarCoupledLunarError(f"{label} must be an exact three-float tuple")
    checked = tuple(_finite_float(item, f"{label}[{i}]") for i, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in checked))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SolarCoupledLunarError(f"{label} must be nonzero")
    tolerance = 64.0 * math.ulp(1.0)
    if abs(norm - 1.0) > tolerance:
        raise SolarCoupledLunarError(f"{label} must be unit length")
    return checked  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class SolarCoupledLunarParameters:
    """Caller-sourced constants for the additive v2 model."""

    sun_gm_km3_s2: float
    lunar: _v1.CoupledLunarParameters
    earth_j2: float
    earth_reference_radius_km: float
    earth_pole_inertial: tuple[float, float, float] = (0.0, 0.0, 1.0)
    model_id: str = SOLAR_COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        _positive_float(self.sun_gm_km3_s2, "sun_gm_km3_s2")
        if type(self.lunar) is not _v1.CoupledLunarParameters:
            raise SolarCoupledLunarError(
                "lunar must be an exact CoupledLunarParameters"
            )
        _nonnegative_float(self.earth_j2, "earth_j2")
        _positive_float(self.earth_reference_radius_km, "earth_reference_radius_km")
        _unit_vector_tuple(self.earth_pole_inertial, "earth_pole_inertial")
        if self.model_id != SOLAR_COUPLED_LUNAR_MODEL_ID:
            raise SolarCoupledLunarError("model_id changed")
        if self.scope != MODEL_SCOPE:
            raise SolarCoupledLunarError("scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise SolarCoupledLunarError("scientific_claim_state changed")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise SolarCoupledLunarError(f"{label} must be exact false")

    @property
    def gravitational_parameters_km3_s2(self) -> tuple[float, float, float]:
        return (
            self.sun_gm_km3_s2,
            self.lunar.earth_gm_km3_s2,
            self.lunar.moon_gm_km3_s2,
        )


@dataclass(frozen=True, slots=True, eq=False)
class SolarCoupledLunarState:
    """One owned simultaneous three-body and lunar-rotation state."""

    epoch: float
    positions_km: object
    velocities_km_s: object
    mantle_quaternion_body_to_inertial: object
    mantle_angular_velocity_body_s: object
    core_angular_velocity_body_s: object

    def __post_init__(self) -> None:
        _finite_float(self.epoch, "epoch")
        object.__setattr__(self, "positions_km", _owned_array(self.positions_km, (3, 3), "positions_km"))
        object.__setattr__(self, "velocities_km_s", _owned_array(self.velocities_km_s, (3, 3), "velocities_km_s"))
        quaternion = _owned_array(
            self.mantle_quaternion_body_to_inertial,
            (4,),
            "mantle_quaternion_body_to_inertial",
        )
        try:
            normalized = _v1._unit_quaternion(quaternion)
        except _v1.CoupledLunarError as exc:
            raise SolarCoupledLunarError(str(exc)) from exc
        normalized.setflags(write=False)
        object.__setattr__(self, "mantle_quaternion_body_to_inertial", normalized)
        object.__setattr__(
            self,
            "mantle_angular_velocity_body_s",
            _owned_array(self.mantle_angular_velocity_body_s, (3,), "mantle_angular_velocity_body_s"),
        )
        object.__setattr__(
            self,
            "core_angular_velocity_body_s",
            _owned_array(self.core_angular_velocity_body_s, (3,), "core_angular_velocity_body_s"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class SolarCoupledLunarResult:
    """Owned checkpoints and conservative-structure diagnostics."""

    parameters: SolarCoupledLunarParameters
    integration_spec: _v1.CoupledLunarIntegrationSpec
    checkpoints: tuple[SolarCoupledLunarState, ...]
    accepted_step_epochs: tuple[float, ...]
    accepted_step_magnitudes: tuple[float, ...]
    accepted_steps: int
    rkf78_stage_evaluations: int
    accepted_step_ledger_sha256: str
    maximum_quaternion_preprojection_norm_error: float
    maximum_rotation_orthogonality_error: float
    maximum_rotation_determinant_error: float
    maximum_linear_momentum_relative_drift: float
    maximum_barycenter_relative_drift: float
    maximum_total_angular_momentum_vector_relative_change: float
    maximum_fixed_pole_angular_momentum_relative_drift: float
    maximum_energy_relative_change: float
    maximum_lunar_figure_torque_balance_relative: float
    maximum_pair_reaction_relative: float
    maximum_earth_j2_fixed_pole_torque_relative: float
    maximum_positive_viscous_power_over_mr2: float
    maximum_earth_lunar_figure_acceleration_km_s2: float
    maximum_solar_lunar_figure_acceleration_km_s2: float
    maximum_earth_j2_acceleration_km_s2: float
    maximum_cmb_torque_over_mr2: float
    minimum_pair_separation_km: float
    earth_pole_is_held_fixed: bool = True
    model_id: str = SOLAR_COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    @property
    def final_state(self) -> SolarCoupledLunarState:
        return self.checkpoints[-1]


@dataclass(frozen=True, slots=True, eq=False)
class _FigureInteraction:
    source_acceleration_km_s2: object
    moon_acceleration_km_s2: object
    torque_on_mantle_over_mr2: object
    potential_gm_scaled: float


@dataclass(frozen=True, slots=True, eq=False)
class _J2Interaction:
    earth_acceleration_km_s2: object
    moon_acceleration_km_s2: object
    potential_gm_scaled: float


def _source_lunar_quadrupole(
    quaternion: object,
    source_position_km: object,
    moon_position_km: object,
    *,
    source_gm_km3_s2: float,
    parameters: SolarCoupledLunarParameters,
) -> _FigureInteraction:
    """Evaluate one source--lunar-figure common potential interaction."""

    if type(parameters) is not SolarCoupledLunarParameters:
        raise SolarCoupledLunarError("parameters must be exact SolarCoupledLunarParameters")
    _positive_float(source_gm_km3_s2, "source_gm_km3_s2")
    np = _numpy()
    source = np.asarray(source_position_km, dtype=np.float64)
    moon = np.asarray(moon_position_km, dtype=np.float64)
    if source.shape != (3,) or moon.shape != (3,) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(moon)):
        raise SolarCoupledLunarError("source and Moon positions must be finite 3-vectors")
    separation = np.ascontiguousarray(source - moon, dtype=np.float64)
    distance = float(np.linalg.norm(separation))
    if not math.isfinite(distance) or distance <= 0.0:
        raise SolarCoupledLunarError("source and Moon must remain separated")
    rotation = _v1._body_to_inertial(quaternion)
    relative_body = np.ascontiguousarray(rotation.T @ separation, dtype=np.float64)
    x, y, z = (float(value) for value in relative_body)
    a, b, c = parameters.lunar.figure_inertia_over_mr2
    lunar_j2 = 0.5 * (2.0 * c - a - b)
    lunar_c22 = 0.25 * (b - a)
    radius_squared = distance * distance
    harmonic = (
        0.5 * lunar_j2 * (x * x + y * y - 2.0 * z * z)
        + 3.0 * lunar_c22 * (x * x - y * y)
    )
    gradient = np.asarray(
        (
            lunar_j2 * x + 6.0 * lunar_c22 * x,
            lunar_j2 * y - 6.0 * lunar_c22 * y,
            -2.0 * lunar_j2 * z,
        ),
        dtype=np.float64,
    )
    scale = (
        parameters.lunar.moon_gm_km3_s2
        * parameters.lunar.lunar_reference_radius_km**2
        / (radius_squared * radius_squared * distance)
    )
    source_acceleration = np.ascontiguousarray(
        rotation @ (scale * (gradient - (5.0 * harmonic / radius_squared) * relative_body)),
        dtype=np.float64,
    )
    moon_acceleration = np.ascontiguousarray(
        -(source_gm_km3_s2 / parameters.lunar.moon_gm_km3_s2) * source_acceleration,
        dtype=np.float64,
    )
    direction_body = np.ascontiguousarray(relative_body / distance)
    moments = np.asarray(parameters.lunar.figure_inertia_over_mr2, dtype=np.float64)
    torque = np.ascontiguousarray(
        3.0 * source_gm_km3_s2 / distance**3
        * np.cross(direction_body, moments * direction_body),
        dtype=np.float64,
    )
    potential = (
        source_gm_km3_s2
        * parameters.lunar.moon_gm_km3_s2
        * parameters.lunar.lunar_reference_radius_km**2
        / (2.0 * distance**3)
        * (3.0 * float(np.dot(direction_body, moments * direction_body)) - float(np.sum(moments)))
    )
    if not all(np.all(np.isfinite(value)) for value in (source_acceleration, moon_acceleration, torque)) or not math.isfinite(potential):
        raise SolarCoupledLunarError("lunar figure interaction became nonfinite")
    return _FigureInteraction(source_acceleration, moon_acceleration, torque, potential)


def _earth_j2_interaction(
    earth_position_km: object,
    moon_position_km: object,
    parameters: SolarCoupledLunarParameters,
) -> _J2Interaction:
    """Evaluate the conservative fixed-axis Earth-J2 orbital pair."""

    if type(parameters) is not SolarCoupledLunarParameters:
        raise SolarCoupledLunarError("parameters must be exact SolarCoupledLunarParameters")
    np = _numpy()
    earth = np.asarray(earth_position_km, dtype=np.float64)
    moon = np.asarray(moon_position_km, dtype=np.float64)
    relative = np.ascontiguousarray(moon - earth, dtype=np.float64)
    radius_squared = float(np.dot(relative, relative))
    if earth.shape != (3,) or moon.shape != (3,) or not np.all(np.isfinite(earth)) or not np.all(np.isfinite(moon)) or not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise SolarCoupledLunarError("Earth and Moon must be finite and separated")
    radius = math.sqrt(radius_squared)
    pole = np.asarray(parameters.earth_pole_inertial, dtype=np.float64)
    axial = float(np.dot(relative, pole))
    factor = (
        1.5
        * parameters.earth_j2
        * parameters.lunar.earth_gm_km3_s2
        * parameters.earth_reference_radius_km**2
        / (radius_squared * radius_squared * radius)
    )
    moon_acceleration = np.ascontiguousarray(
        factor * ((5.0 * axial * axial / radius_squared - 1.0) * relative - 2.0 * axial * pole),
        dtype=np.float64,
    )
    earth_acceleration = np.ascontiguousarray(
        -(parameters.lunar.moon_gm_km3_s2 / parameters.lunar.earth_gm_km3_s2) * moon_acceleration,
        dtype=np.float64,
    )
    potential = (
        parameters.lunar.earth_gm_km3_s2
        * parameters.lunar.moon_gm_km3_s2
        * parameters.earth_j2
        * parameters.earth_reference_radius_km**2
        / (2.0 * radius**3)
        * (3.0 * axial * axial / radius_squared - 1.0)
    )
    if not np.all(np.isfinite(earth_acceleration)) or not np.all(np.isfinite(moon_acceleration)) or not math.isfinite(potential):
        raise SolarCoupledLunarError("Earth-J2 interaction became nonfinite")
    return _J2Interaction(earth_acceleration, moon_acceleration, potential)


def _pack_state(state: SolarCoupledLunarState):
    np = _numpy()
    return (
        np.ascontiguousarray(
            np.concatenate((state.positions_km.reshape(9), state.mantle_quaternion_body_to_inertial)),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            np.concatenate((state.velocities_km_s.reshape(9), state.mantle_angular_velocity_body_s, state.core_angular_velocity_body_s)),
            dtype=np.float64,
        ),
    )


def _state_from_packed(epoch: float, positions: object, velocities: object) -> SolarCoupledLunarState:
    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    return SolarCoupledLunarState(
        epoch=float(epoch),
        positions_km=np.ascontiguousarray(p[:9].reshape(3, 3)),
        velocities_km_s=np.ascontiguousarray(v[:9].reshape(3, 3)),
        mantle_quaternion_body_to_inertial=np.ascontiguousarray(p[9:13]),
        mantle_angular_velocity_body_s=np.ascontiguousarray(v[9:12]),
        core_angular_velocity_body_s=np.ascontiguousarray(v[12:15]),
    )


def _invariants(positions: object, velocities: object, parameters: SolarCoupledLunarParameters) -> dict[str, Any]:
    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    xyz = p[:9].reshape(3, 3)
    uvw = v[:9].reshape(3, 3)
    quaternion = _v1._unit_quaternion(p[9:13])
    mantle_rate = v[9:12]
    core_rate = v[12:15]
    gms = np.asarray(parameters.gravitational_parameters_km3_s2, dtype=np.float64)
    orbital_kinetic = 0.5 * sum(float(gms[i] * np.dot(uvw[i], uvw[i])) for i in range(3))
    point_potential = 0.0
    minimum_separation = math.inf
    for left in range(3):
        for right in range(left + 1, 3):
            distance = float(np.linalg.norm(xyz[left] - xyz[right]))
            if not math.isfinite(distance) or distance <= 0.0:
                raise SolarCoupledLunarError("two translational bodies collided or became invalid")
            minimum_separation = min(minimum_separation, distance)
            point_potential -= float(gms[left] * gms[right] / distance)
    earth_figure = _source_lunar_quadrupole(
        quaternion, xyz[EARTH_INDEX], xyz[MOON_INDEX],
        source_gm_km3_s2=parameters.lunar.earth_gm_km3_s2,
        parameters=parameters,
    )
    solar_figure = _source_lunar_quadrupole(
        quaternion, xyz[SUN_INDEX], xyz[MOON_INDEX],
        source_gm_km3_s2=parameters.sun_gm_km3_s2,
        parameters=parameters,
    )
    earth_j2 = _earth_j2_interaction(xyz[EARTH_INDEX], xyz[MOON_INDEX], parameters)
    mantle = np.asarray(parameters.lunar.mantle_inertia_over_mr2, dtype=np.float64)
    core = np.asarray(parameters.lunar.core_coupling.core_inertia_over_mr2, dtype=np.float64)
    spin_scale = parameters.lunar.moon_gm_km3_s2 * parameters.lunar.lunar_reference_radius_km**2
    rotational_kinetic = 0.5 * spin_scale * (
        float(np.dot(mantle * mantle_rate, mantle_rate))
        + float(np.dot(core * core_rate, core_rate))
    )
    rotation = _v1._body_to_inertial(quaternion)
    orbital_angular = sum((np.cross(xyz[i], gms[i] * uvw[i]) for i in range(3)), start=np.zeros(3))
    spin_angular = spin_scale * (rotation @ (mantle * mantle_rate + core * core_rate))
    total_angular = np.ascontiguousarray(orbital_angular + spin_angular)
    total_momentum = np.ascontiguousarray(np.sum(gms[:, None] * uvw, axis=0))
    total_gm = float(np.sum(gms))
    barycenter = np.ascontiguousarray(np.sum(gms[:, None] * xyz, axis=0) / total_gm)
    relative_speed_scale = max(
        float(np.linalg.norm(uvw[EARTH_INDEX] - uvw[MOON_INDEX])),
        float(np.linalg.norm(uvw[SUN_INDEX] - uvw[EARTH_INDEX])),
    )
    return {
        "energy": float(orbital_kinetic + rotational_kinetic + point_potential + earth_figure.potential_gm_scaled + solar_figure.potential_gm_scaled + earth_j2.potential_gm_scaled),
        "total_angular": total_angular,
        "fixed_pole_angular": float(np.dot(total_angular, np.asarray(parameters.earth_pole_inertial))),
        "total_momentum": total_momentum,
        "barycenter": barycenter,
        "minimum_separation": minimum_separation,
        "relative_speed_scale": relative_speed_scale,
        "total_gm": total_gm,
    }


def integrate_solar_coupled_lunar_trajectory(
    initial_state: SolarCoupledLunarState,
    parameters: SolarCoupledLunarParameters,
    integration_spec: _v1.CoupledLunarIntegrationSpec,
) -> SolarCoupledLunarResult:
    """Advance the additive v2 screening model through exact checkpoints."""

    if type(initial_state) is not SolarCoupledLunarState:
        raise SolarCoupledLunarError("initial_state must be exact SolarCoupledLunarState")
    if type(parameters) is not SolarCoupledLunarParameters:
        raise SolarCoupledLunarError("parameters must be exact SolarCoupledLunarParameters")
    if type(integration_spec) is not _v1.CoupledLunarIntegrationSpec:
        raise SolarCoupledLunarError("integration_spec must be exact CoupledLunarIntegrationSpec")
    if initial_state.epoch != integration_spec.checkpoint_epochs[0]:
        raise SolarCoupledLunarError("initial state epoch must equal the first checkpoint exactly")
    np = _numpy()
    backend = resolve_backend("numpy")
    positions, velocities = _pack_state(initial_state)
    position_carry = np.zeros(13, dtype=np.float64)
    velocity_carry = np.zeros(15, dtype=np.float64)
    mantle_inertia = np.ascontiguousarray(np.diag(parameters.lunar.mantle_inertia_over_mr2), dtype=np.float64)
    zero_matrix = np.zeros((3, 3), dtype=np.float64)
    zero_vector = np.zeros(3, dtype=np.float64)
    gms = np.asarray(parameters.gravitational_parameters_km3_s2, dtype=np.float64)
    pole = np.asarray(parameters.earth_pole_inertial, dtype=np.float64)
    initial = _invariants(positions, velocities, parameters)
    tiny = np.finfo(np.float64).tiny
    energy_scale = max(abs(initial["energy"]), tiny)
    angular_scale = max(float(np.linalg.norm(initial["total_angular"])), tiny)
    axial_scale = max(abs(initial["fixed_pole_angular"]), angular_scale * math.ulp(1.0), tiny)
    momentum_scale = max(parameters.lunar.moon_gm_km3_s2 * initial["relative_speed_scale"], tiny)
    position_scale = max(
        float(np.linalg.norm(initial_state.positions_km[EARTH_INDEX] - initial_state.positions_km[MOON_INDEX])),
        tiny,
    )
    maxima = {name: 0.0 for name in (
        "preprojection", "orthogonality", "determinant", "momentum", "barycenter",
        "angular_vector", "angular_axis", "energy", "torque_balance", "pair_reaction",
        "j2_axis_torque", "positive_viscous_power", "earth_figure_acceleration",
        "solar_figure_acceleration", "j2_acceleration", "cmb_torque",
    )}
    minimum_pair_separation = initial["minimum_separation"]
    accepted_epochs: list[float] = []
    accepted_magnitudes: list[float] = []
    checkpoints: list[SolarCoupledLunarState] = [initial_state]
    ledger = hashlib.sha256(LEDGER_DOMAIN)
    epoch = initial_state.epoch
    direction = 1.0 if integration_spec.checkpoint_epochs[-1] > epoch else -1.0

    for checkpoint_epoch in integration_spec.checkpoint_epochs[1:]:
        while epoch != checkpoint_epoch:
            if len(accepted_epochs) >= integration_spec.maximum_steps:
                raise SolarCoupledLunarStepLimitError("maximum_steps exhausted")
            remaining = checkpoint_epoch - epoch
            if direction * remaining <= 0.0 or not math.isfinite(remaining):
                raise SolarCoupledLunarError("checkpoint schedule lost monotone progress")
            magnitude = min(integration_spec.maximum_step, abs(remaining))
            endpoint = checkpoint_epoch if magnitude >= abs(remaining) else epoch + math.copysign(magnitude, direction)
            if endpoint == epoch or not math.isfinite(endpoint):
                raise SolarCoupledLunarError("fixed step cannot advance binary64 epoch")
            step = endpoint - epoch

            def derivative(_stage_epoch: float, stage_positions: object, stage_velocities: object) -> tuple[object, object]:
                state = np.asarray(stage_positions, dtype=np.float64)
                rates = np.asarray(stage_velocities, dtype=np.float64)
                xyz = state[:9].reshape(3, 3)
                uvw = rates[:9].reshape(3, 3)
                quaternion = _v1._unit_quaternion(state[9:13])
                mantle_rate = np.ascontiguousarray(rates[9:12])
                core_rate = np.ascontiguousarray(rates[12:15])
                accelerations = np.zeros((3, 3), dtype=np.float64)
                for left in range(3):
                    for right in range(left + 1, 3):
                        separation = xyz[left] - xyz[right]
                        distance = float(np.linalg.norm(separation))
                        if not math.isfinite(distance) or distance <= 0.0:
                            raise SolarCoupledLunarError("two translational bodies collided or became invalid")
                        inverse_cube = 1.0 / distance**3
                        accelerations[left] -= gms[right] * inverse_cube * separation
                        accelerations[right] += gms[left] * inverse_cube * separation
                figure_interactions = (
                    (SUN_INDEX, _source_lunar_quadrupole(
                        quaternion, xyz[SUN_INDEX], xyz[MOON_INDEX],
                        source_gm_km3_s2=parameters.sun_gm_km3_s2, parameters=parameters,
                    )),
                    (EARTH_INDEX, _source_lunar_quadrupole(
                        quaternion, xyz[EARTH_INDEX], xyz[MOON_INDEX],
                        source_gm_km3_s2=parameters.lunar.earth_gm_km3_s2, parameters=parameters,
                    )),
                )
                external_torque = np.zeros(3, dtype=np.float64)
                body_to_inertial = _v1._body_to_inertial(quaternion)
                for source_index, interaction in figure_interactions:
                    accelerations[source_index] += interaction.source_acceleration_km_s2
                    accelerations[MOON_INDEX] += interaction.moon_acceleration_km_s2
                    external_torque += interaction.torque_on_mantle_over_mr2
                    reaction = gms[source_index] * interaction.source_acceleration_km_s2 + gms[MOON_INDEX] * interaction.moon_acceleration_km_s2
                    reaction_scale = max(float(np.linalg.norm(gms[source_index] * interaction.source_acceleration_km_s2)), tiny)
                    maxima["pair_reaction"] = max(maxima["pair_reaction"], float(np.linalg.norm(reaction)) / reaction_scale)
                    separation = xyz[source_index] - xyz[MOON_INDEX]
                    orbital_torque = np.cross(separation, gms[source_index] * interaction.source_acceleration_km_s2)
                    spin_torque = (
                        parameters.lunar.moon_gm_km3_s2
                        * parameters.lunar.lunar_reference_radius_km**2
                        * (body_to_inertial @ interaction.torque_on_mantle_over_mr2)
                    )
                    anisotropy = max(parameters.lunar.figure_inertia_over_mr2) - min(parameters.lunar.figure_inertia_over_mr2)
                    torque_scale = max(
                        float(np.linalg.norm(spin_torque)),
                        gms[source_index] * gms[MOON_INDEX] * parameters.lunar.lunar_reference_radius_km**2 * anisotropy / float(np.linalg.norm(separation))**3,
                        tiny,
                    )
                    maxima["torque_balance"] = max(maxima["torque_balance"], float(np.linalg.norm(orbital_torque + spin_torque)) / torque_scale)
                j2 = _earth_j2_interaction(xyz[EARTH_INDEX], xyz[MOON_INDEX], parameters)
                accelerations[EARTH_INDEX] += j2.earth_acceleration_km_s2
                accelerations[MOON_INDEX] += j2.moon_acceleration_km_s2
                j2_reaction = gms[EARTH_INDEX] * j2.earth_acceleration_km_s2 + gms[MOON_INDEX] * j2.moon_acceleration_km_s2
                j2_reaction_scale = max(float(np.linalg.norm(gms[MOON_INDEX] * j2.moon_acceleration_km_s2)), tiny)
                maxima["pair_reaction"] = max(maxima["pair_reaction"], float(np.linalg.norm(j2_reaction)) / j2_reaction_scale)
                j2_orbital_torque = np.cross(
                    xyz[MOON_INDEX] - xyz[EARTH_INDEX],
                    gms[MOON_INDEX] * j2.moon_acceleration_km_s2,
                )
                maxima["j2_axis_torque"] = max(
                    maxima["j2_axis_torque"],
                    abs(float(np.dot(j2_orbital_torque, pole))) / max(float(np.linalg.norm(j2_orbital_torque)), j2_reaction_scale * position_scale, tiny),
                )
                try:
                    rotation = lunar_mantle_core_angular_accelerations(
                        parameters.lunar.core_coupling,
                        mantle_angular_velocity_body=mantle_rate,
                        core_angular_velocity_body=core_rate,
                        mantle_inertia_over_mr2=mantle_inertia,
                        mantle_inertia_rate_over_mr2_per_second=zero_matrix,
                        external_torque_over_mr2=np.ascontiguousarray(external_torque),
                        geodetic_precession_body=zero_vector,
                    )
                except LunarFluidCoreError as exc:
                    raise SolarCoupledLunarError(str(exc)) from exc
                viscous_power = float(np.dot(mantle_rate - core_rate, rotation.viscous_torque_on_mantle_body))
                maxima["positive_viscous_power"] = max(maxima["positive_viscous_power"], viscous_power)
                maxima["solar_figure_acceleration"] = max(maxima["solar_figure_acceleration"], float(np.linalg.norm(figure_interactions[0][1].moon_acceleration_km_s2)))
                maxima["earth_figure_acceleration"] = max(maxima["earth_figure_acceleration"], float(np.linalg.norm(figure_interactions[1][1].moon_acceleration_km_s2)))
                maxima["j2_acceleration"] = max(maxima["j2_acceleration"], float(np.linalg.norm(j2.moon_acceleration_km_s2)))
                maxima["cmb_torque"] = max(maxima["cmb_torque"], float(np.linalg.norm(rotation.cmb_torque_on_mantle_body)))
                return (
                    np.ascontiguousarray(np.concatenate((uvw.reshape(9), _v1._quaternion_derivative(quaternion, mantle_rate))), dtype=np.float64),
                    np.ascontiguousarray(np.concatenate((accelerations.reshape(9), rotation.mantle_angular_acceleration_body, rotation.core_angular_acceleration_body)), dtype=np.float64),
                )

            trial = _rkf78_step(
                backend=backend,
                epoch=epoch,
                endpoint_epoch=endpoint,
                step=step,
                positions=positions,
                velocities=velocities,
                position_carry=position_carry,
                velocity_carry=velocity_carry,
                derivative=derivative,
            )
            accepted_positions = np.ascontiguousarray(trial.accepted_positions, dtype=np.float64)
            preprojection_norm = float(np.linalg.norm(accepted_positions[9:13]))
            maxima["preprojection"] = max(maxima["preprojection"], abs(preprojection_norm - 1.0))
            accepted_positions[9:13] = _v1._unit_quaternion(accepted_positions[9:13])
            positions = accepted_positions
            velocities = np.ascontiguousarray(trial.accepted_velocities, dtype=np.float64)
            position_carry = np.ascontiguousarray(trial.accepted_position_carry, dtype=np.float64)
            position_carry[9:13] = 0.0
            velocity_carry = np.ascontiguousarray(trial.accepted_velocity_carry, dtype=np.float64)
            epoch = endpoint
            accepted_epochs.append(epoch)
            accepted_magnitudes.append(abs(step))
            ledger.update(struct.pack(">dd", epoch, abs(step)))
            ledger.update(positions.tobytes(order="C"))
            ledger.update(velocities.tobytes(order="C"))
            current = _invariants(positions, velocities, parameters)
            minimum_pair_separation = min(minimum_pair_separation, current["minimum_separation"])
            elapsed = epoch - initial_state.epoch
            expected_barycenter = initial["barycenter"] + initial["total_momentum"] / initial["total_gm"] * elapsed
            maxima["momentum"] = max(maxima["momentum"], float(np.linalg.norm(current["total_momentum"] - initial["total_momentum"])) / momentum_scale)
            maxima["barycenter"] = max(maxima["barycenter"], float(np.linalg.norm(current["barycenter"] - expected_barycenter)) / position_scale)
            maxima["angular_vector"] = max(maxima["angular_vector"], float(np.linalg.norm(current["total_angular"] - initial["total_angular"])) / angular_scale)
            maxima["angular_axis"] = max(maxima["angular_axis"], abs(current["fixed_pole_angular"] - initial["fixed_pole_angular"]) / axial_scale)
            maxima["energy"] = max(maxima["energy"], abs(current["energy"] - initial["energy"]) / energy_scale)
            orthogonality, determinant = _v1._rotation_metrics(positions[9:13])
            maxima["orthogonality"] = max(maxima["orthogonality"], orthogonality)
            maxima["determinant"] = max(maxima["determinant"], determinant)
        checkpoints.append(_state_from_packed(epoch, positions, velocities))

    return SolarCoupledLunarResult(
        parameters=parameters,
        integration_spec=integration_spec,
        checkpoints=tuple(checkpoints),
        accepted_step_epochs=tuple(accepted_epochs),
        accepted_step_magnitudes=tuple(accepted_magnitudes),
        accepted_steps=len(accepted_epochs),
        rkf78_stage_evaluations=len(accepted_epochs) * RKF78_STAGE_COUNT,
        accepted_step_ledger_sha256=ledger.hexdigest(),
        maximum_quaternion_preprojection_norm_error=maxima["preprojection"],
        maximum_rotation_orthogonality_error=maxima["orthogonality"],
        maximum_rotation_determinant_error=maxima["determinant"],
        maximum_linear_momentum_relative_drift=maxima["momentum"],
        maximum_barycenter_relative_drift=maxima["barycenter"],
        maximum_total_angular_momentum_vector_relative_change=maxima["angular_vector"],
        maximum_fixed_pole_angular_momentum_relative_drift=maxima["angular_axis"],
        maximum_energy_relative_change=maxima["energy"],
        maximum_lunar_figure_torque_balance_relative=maxima["torque_balance"],
        maximum_pair_reaction_relative=maxima["pair_reaction"],
        maximum_earth_j2_fixed_pole_torque_relative=maxima["j2_axis_torque"],
        maximum_positive_viscous_power_over_mr2=maxima["positive_viscous_power"],
        maximum_earth_lunar_figure_acceleration_km_s2=maxima["earth_figure_acceleration"],
        maximum_solar_lunar_figure_acceleration_km_s2=maxima["solar_figure_acceleration"],
        maximum_earth_j2_acceleration_km_s2=maxima["j2_acceleration"],
        maximum_cmb_torque_over_mr2=maxima["cmb_torque"],
        minimum_pair_separation_km=minimum_pair_separation,
    )


__all__ = [
    "BODY_IDS",
    "EARTH_INDEX",
    "MODEL_SCOPE",
    "MOON_INDEX",
    "SCIENTIFIC_CLAIM_STATE",
    "SOLAR_COUPLED_LUNAR_MODEL_ID",
    "SUN_INDEX",
    "SolarCoupledLunarError",
    "SolarCoupledLunarParameters",
    "SolarCoupledLunarResult",
    "SolarCoupledLunarState",
    "SolarCoupledLunarStepLimitError",
    "integrate_solar_coupled_lunar_trajectory",
]
