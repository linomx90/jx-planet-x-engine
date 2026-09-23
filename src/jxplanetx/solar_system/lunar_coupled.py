"""Restricted simultaneous Earth--Moon orbit and mantle/core rotation model.

This module advances one isolated Earth--Moon pair, the lunar mantle attitude
and angular velocity, and the lunar fluid-core angular velocity in one RKF78
state.  The attitude-dependent lunar degree-two force and gravity-gradient
torque come from one common potential, so their linear and angular reactions
are applied together.  Core--mantle boundary torque is equal and opposite.

The model is deliberately narrow: static principal moments, no delayed tides,
no spin-distortion inertia, no Sun or planetary forces, no relativity, no
Earth figure, and no raw LLR observation model.  It is an unregistered
``SCREENING_ONLY`` research boundary, not a DE440 reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT, _rkf78_step

from .lunar_fluid_core import (
    LunarFluidCoreCoupling,
    LunarFluidCoreError,
    lunar_mantle_core_angular_accelerations,
)


COUPLED_LUNAR_MODEL_ID = (
    "solar-system.experimental.earth-moon-static-figure-mantle-fluid-core.v1"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MODEL_SCOPE = (
    "ISOLATED_EARTH_MOON_MONOPOLES_STATIC_LUNAR_DEGREE2_"
    "MANTLE_QUATERNION_AND_FLUID_CORE_RATE"
)
LEDGER_DOMAIN = b"jxplanetx.coupled-lunar.accepted-step-ledger.v1\0"


class CoupledLunarError(ValueError):
    """The restricted coupled lunar contract or integration failed."""


class CoupledLunarStepLimitError(CoupledLunarError):
    """The fixed-step integration exhausted its declared work cap."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise CoupledLunarError("coupled lunar integration requires NumPy") from exc
    return np


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise CoupledLunarError(f"{label} must be a positive finite built-in float")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise CoupledLunarError(f"{label} must be a finite built-in float")
    return value


def _moments(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise CoupledLunarError(f"{label} must be an exact three-float tuple")
    checked = tuple(
        _positive_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if not checked[0] <= checked[1] <= checked[2]:
        raise CoupledLunarError(f"{label} must satisfy A <= B <= C")
    return checked  # type: ignore[return-value]


def _owned_array(value: object, shape: tuple[int, ...], label: str):
    np = _numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("float64")
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise CoupledLunarError(
            f"{label} must be finite C-contiguous NumPy float64 with shape {shape}"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarParameters:
    """Caller-sourced parameters for the restricted simultaneous model."""

    earth_gm_km3_s2: float
    moon_gm_km3_s2: float
    lunar_reference_radius_km: float
    figure_inertia_over_mr2: tuple[float, float, float]
    mantle_inertia_over_mr2: tuple[float, float, float]
    core_coupling: LunarFluidCoreCoupling
    model_id: str = COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        _positive_float(self.earth_gm_km3_s2, "earth_gm_km3_s2")
        _positive_float(self.moon_gm_km3_s2, "moon_gm_km3_s2")
        _positive_float(
            self.lunar_reference_radius_km, "lunar_reference_radius_km"
        )
        figure = _moments(self.figure_inertia_over_mr2, "figure_inertia_over_mr2")
        mantle = _moments(self.mantle_inertia_over_mr2, "mantle_inertia_over_mr2")
        if type(self.core_coupling) is not LunarFluidCoreCoupling:
            raise CoupledLunarError(
                "core_coupling must be an exact LunarFluidCoreCoupling"
            )
        core = self.core_coupling.core_inertia_over_mr2
        for index, (total, mantle_value, core_value) in enumerate(
            zip(figure, mantle, core, strict=True)
        ):
            reconstructed = mantle_value + core_value
            tolerance = 128.0 * math.ulp(max(abs(total), abs(reconstructed), 1.0))
            if abs(total - reconstructed) > tolerance:
                raise CoupledLunarError(
                    "figure inertia must equal mantle plus core inertia "
                    f"at component {index}"
                )
        if self.model_id != COUPLED_LUNAR_MODEL_ID:
            raise CoupledLunarError("model_id changed")
        if self.scope != MODEL_SCOPE:
            raise CoupledLunarError("scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise CoupledLunarError("scientific_claim_state changed")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise CoupledLunarError("registry_authorized must be exact false")
        if (
            type(self.qualification_authorized) is not bool
            or self.qualification_authorized
        ):
            raise CoupledLunarError("qualification_authorized must be exact false")


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarState:
    """One owned simultaneous translational and rotational state."""

    epoch: float
    positions_km: object
    velocities_km_s: object
    mantle_quaternion_body_to_inertial: object
    mantle_angular_velocity_body_s: object
    core_angular_velocity_body_s: object

    def __post_init__(self) -> None:
        _finite_float(self.epoch, "epoch")
        object.__setattr__(
            self,
            "positions_km",
            _owned_array(self.positions_km, (2, 3), "positions_km"),
        )
        object.__setattr__(
            self,
            "velocities_km_s",
            _owned_array(self.velocities_km_s, (2, 3), "velocities_km_s"),
        )
        quaternion = _owned_array(
            self.mantle_quaternion_body_to_inertial,
            (4,),
            "mantle_quaternion_body_to_inertial",
        )
        norm = float(_numpy().linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 0.0:
            raise CoupledLunarError("mantle quaternion norm must be positive")
        normalized = _numpy().ascontiguousarray(quaternion / norm)
        normalized.setflags(write=False)
        object.__setattr__(
            self, "mantle_quaternion_body_to_inertial", normalized
        )
        object.__setattr__(
            self,
            "mantle_angular_velocity_body_s",
            _owned_array(
                self.mantle_angular_velocity_body_s,
                (3,),
                "mantle_angular_velocity_body_s",
            ),
        )
        object.__setattr__(
            self,
            "core_angular_velocity_body_s",
            _owned_array(
                self.core_angular_velocity_body_s,
                (3,),
                "core_angular_velocity_body_s",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarIntegrationSpec:
    """Exact checkpoint schedule and fixed maximum step magnitude."""

    checkpoint_epochs: tuple[float, ...]
    maximum_step: float
    maximum_steps: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.checkpoint_epochs) is not tuple or len(self.checkpoint_epochs) < 2:
            raise CoupledLunarError(
                "checkpoint_epochs must be an exact tuple with at least two epochs"
            )
        epochs = tuple(
            _finite_float(value, f"checkpoint_epochs[{index}]")
            for index, value in enumerate(self.checkpoint_epochs)
        )
        differences = tuple(
            right - left for left, right in zip(epochs, epochs[1:])
        )
        if not (
            all(value > 0.0 for value in differences)
            or all(value < 0.0 for value in differences)
        ):
            raise CoupledLunarError(
                "checkpoint epochs must be strictly monotone in one direction"
            )
        _positive_float(self.maximum_step, "maximum_step")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise CoupledLunarError("maximum_steps must be a positive integer")


@dataclass(frozen=True, slots=True, eq=False)
class CoupledLunarResult:
    """Owned checkpoints, accounting, and conservative-structure diagnostics."""

    parameters: CoupledLunarParameters
    integration_spec: CoupledLunarIntegrationSpec
    checkpoints: tuple[CoupledLunarState, ...]
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
    maximum_total_angular_momentum_relative_drift: float
    maximum_energy_relative_change: float
    maximum_orbital_spin_torque_balance_relative: float
    maximum_pair_reaction_relative: float
    maximum_positive_viscous_power_over_mr2: float
    maximum_quadrupole_acceleration_km_s2: float
    maximum_cmb_torque_over_mr2: float
    minimum_separation_km: float
    model_id: str = COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    @property
    def final_state(self) -> CoupledLunarState:
        return self.checkpoints[-1]


@dataclass(frozen=True, slots=True, eq=False)
class _Interaction:
    earth_acceleration_km_s2: object
    moon_acceleration_km_s2: object
    torque_on_mantle_over_mr2: object
    potential_gm_scaled: float


def _unit_quaternion(value: object):
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise CoupledLunarError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CoupledLunarError("quaternion norm must be positive")
    return np.ascontiguousarray(array / norm, dtype=np.float64)


def _quaternion_product(left: object, right: object):
    np = _numpy()
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aw, ax, ay, az = (float(value) for value in a)
    bw, bx, by, bz = (float(value) for value in b)
    return np.ascontiguousarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )


def _quaternion_derivative(quaternion: object, angular_velocity_body: object):
    np = _numpy()
    q = _unit_quaternion(quaternion)
    omega = np.asarray(angular_velocity_body, dtype=np.float64)
    return np.ascontiguousarray(
        0.5 * _quaternion_product(q, np.concatenate((np.zeros(1), omega))),
        dtype=np.float64,
    )


def _body_to_inertial(quaternion: object):
    np = _numpy()
    w, x, y, z = (float(value) for value in _unit_quaternion(quaternion))
    return np.ascontiguousarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quadrupole_interaction(
    quaternion: object,
    earth_position_km: object,
    moon_position_km: object,
    parameters: CoupledLunarParameters,
) -> _Interaction:
    """Evaluate one common degree-two potential, force, and body torque."""

    if type(parameters) is not CoupledLunarParameters:
        raise CoupledLunarError("parameters must be exact CoupledLunarParameters")
    np = _numpy()
    earth = np.asarray(earth_position_km, dtype=np.float64)
    moon = np.asarray(moon_position_km, dtype=np.float64)
    if (
        earth.shape != (3,)
        or moon.shape != (3,)
        or not np.all(np.isfinite(earth))
        or not np.all(np.isfinite(moon))
    ):
        raise CoupledLunarError("Earth and Moon positions must be finite 3-vectors")
    separation = np.ascontiguousarray(earth - moon, dtype=np.float64)
    distance = float(np.linalg.norm(separation))
    if not math.isfinite(distance) or distance <= 0.0:
        raise CoupledLunarError("Earth and Moon must remain separated")
    rotation = _body_to_inertial(quaternion)
    relative_body = np.ascontiguousarray(rotation.T @ separation, dtype=np.float64)
    x, y, z = (float(value) for value in relative_body)
    a, b, c = parameters.figure_inertia_over_mr2
    j2 = 0.5 * (2.0 * c - a - b)
    c22 = 0.25 * (b - a)
    distance_squared = distance * distance
    harmonic = (
        0.5 * j2 * (x * x + y * y - 2.0 * z * z)
        + 3.0 * c22 * (x * x - y * y)
    )
    gradient = np.ascontiguousarray(
        (j2 * x + 6.0 * c22 * x, j2 * y - 6.0 * c22 * y, -2.0 * j2 * z),
        dtype=np.float64,
    )
    scale = (
        parameters.moon_gm_km3_s2
        * parameters.lunar_reference_radius_km**2
        / (distance_squared * distance_squared * distance)
    )
    earth_body_acceleration = scale * (
        gradient - (5.0 * harmonic / distance_squared) * relative_body
    )
    earth_acceleration = np.ascontiguousarray(
        rotation @ earth_body_acceleration, dtype=np.float64
    )
    moon_acceleration = np.ascontiguousarray(
        -(
            parameters.earth_gm_km3_s2
            / parameters.moon_gm_km3_s2
        )
        * earth_acceleration,
        dtype=np.float64,
    )
    direction_body = np.ascontiguousarray(relative_body / distance)
    moments = np.asarray(parameters.figure_inertia_over_mr2, dtype=np.float64)
    torque = np.ascontiguousarray(
        3.0
        * parameters.earth_gm_km3_s2
        / distance**3
        * np.cross(direction_body, moments * direction_body),
        dtype=np.float64,
    )
    potential = (
        parameters.earth_gm_km3_s2
        * parameters.moon_gm_km3_s2
        * parameters.lunar_reference_radius_km**2
        / (2.0 * distance**3)
        * (3.0 * float(np.dot(direction_body, moments * direction_body)) - float(np.sum(moments)))
    )
    for value in (earth_acceleration, moon_acceleration, torque):
        value[value == 0.0] = 0.0
        if not np.all(np.isfinite(value)):
            raise CoupledLunarError("quadrupole interaction became nonfinite")
    if not math.isfinite(potential):
        raise CoupledLunarError("quadrupole potential became nonfinite")
    return _Interaction(
        earth_acceleration_km_s2=earth_acceleration,
        moon_acceleration_km_s2=moon_acceleration,
        torque_on_mantle_over_mr2=torque,
        potential_gm_scaled=potential,
    )


def _pack_state(state: CoupledLunarState):
    np = _numpy()
    generalized_positions = np.ascontiguousarray(
        np.concatenate(
            (
                state.positions_km.reshape(6),
                state.mantle_quaternion_body_to_inertial,
            )
        ),
        dtype=np.float64,
    )
    generalized_velocities = np.ascontiguousarray(
        np.concatenate(
            (
                state.velocities_km_s.reshape(6),
                state.mantle_angular_velocity_body_s,
                state.core_angular_velocity_body_s,
            )
        ),
        dtype=np.float64,
    )
    return generalized_positions, generalized_velocities


def _state_from_packed(epoch: float, positions: object, velocities: object) -> CoupledLunarState:
    np = _numpy()
    generalized_positions = np.asarray(positions, dtype=np.float64)
    generalized_velocities = np.asarray(velocities, dtype=np.float64)
    return CoupledLunarState(
        epoch=float(epoch),
        positions_km=np.ascontiguousarray(generalized_positions[:6].reshape(2, 3)),
        velocities_km_s=np.ascontiguousarray(generalized_velocities[:6].reshape(2, 3)),
        mantle_quaternion_body_to_inertial=np.ascontiguousarray(
            generalized_positions[6:10]
        ),
        mantle_angular_velocity_body_s=np.ascontiguousarray(
            generalized_velocities[6:9]
        ),
        core_angular_velocity_body_s=np.ascontiguousarray(
            generalized_velocities[9:12]
        ),
    )


def _invariants(
    generalized_positions: object,
    generalized_velocities: object,
    parameters: CoupledLunarParameters,
) -> dict[str, Any]:
    np = _numpy()
    positions = np.asarray(generalized_positions, dtype=np.float64)
    velocities = np.asarray(generalized_velocities, dtype=np.float64)
    earth = positions[:3]
    moon = positions[3:6]
    quaternion = _unit_quaternion(positions[6:10])
    earth_velocity = velocities[:3]
    moon_velocity = velocities[3:6]
    mantle_rate = velocities[6:9]
    core_rate = velocities[9:12]
    separation = earth - moon
    distance = float(np.linalg.norm(separation))
    interaction = _quadrupole_interaction(
        quaternion, earth, moon, parameters
    )
    mantle = np.asarray(parameters.mantle_inertia_over_mr2, dtype=np.float64)
    core = np.asarray(
        parameters.core_coupling.core_inertia_over_mr2, dtype=np.float64
    )
    orbital_kinetic = (
        0.5 * parameters.earth_gm_km3_s2 * float(np.dot(earth_velocity, earth_velocity))
        + 0.5 * parameters.moon_gm_km3_s2 * float(np.dot(moon_velocity, moon_velocity))
    )
    rotational_kinetic = (
        0.5
        * parameters.moon_gm_km3_s2
        * parameters.lunar_reference_radius_km**2
        * (
            float(np.dot(mantle * mantle_rate, mantle_rate))
            + float(np.dot(core * core_rate, core_rate))
        )
    )
    point_potential = (
        -parameters.earth_gm_km3_s2 * parameters.moon_gm_km3_s2 / distance
    )
    rotation = _body_to_inertial(quaternion)
    orbital_angular = (
        np.cross(earth, parameters.earth_gm_km3_s2 * earth_velocity)
        + np.cross(moon, parameters.moon_gm_km3_s2 * moon_velocity)
    )
    spin_angular = (
        parameters.moon_gm_km3_s2
        * parameters.lunar_reference_radius_km**2
        * (rotation @ (mantle * mantle_rate + core * core_rate))
    )
    total_momentum = (
        parameters.earth_gm_km3_s2 * earth_velocity
        + parameters.moon_gm_km3_s2 * moon_velocity
    )
    barycenter = (
        parameters.earth_gm_km3_s2 * earth
        + parameters.moon_gm_km3_s2 * moon
    ) / (parameters.earth_gm_km3_s2 + parameters.moon_gm_km3_s2)
    return {
        "energy": float(
            orbital_kinetic
            + rotational_kinetic
            + point_potential
            + interaction.potential_gm_scaled
        ),
        "orbital_angular": np.ascontiguousarray(orbital_angular),
        "spin_angular": np.ascontiguousarray(spin_angular),
        "total_angular": np.ascontiguousarray(orbital_angular + spin_angular),
        "total_momentum": np.ascontiguousarray(total_momentum),
        "barycenter": np.ascontiguousarray(barycenter),
        "distance": distance,
        "relative_speed": float(np.linalg.norm(earth_velocity - moon_velocity)),
    }


def _rotation_metrics(quaternion: object) -> tuple[float, float]:
    np = _numpy()
    matrix = _body_to_inertial(quaternion)
    identity = np.eye(3, dtype=np.float64)
    orthogonality = float(np.max(np.abs(matrix.T @ matrix - identity)))
    determinant = abs(float(np.linalg.det(matrix)) - 1.0)
    return orthogonality, determinant


def integrate_coupled_lunar_trajectory(
    initial_state: CoupledLunarState,
    parameters: CoupledLunarParameters,
    integration_spec: CoupledLunarIntegrationSpec,
) -> CoupledLunarResult:
    """Advance the restricted simultaneous model through exact checkpoints."""

    if type(initial_state) is not CoupledLunarState:
        raise CoupledLunarError("initial_state must be exact CoupledLunarState")
    if type(parameters) is not CoupledLunarParameters:
        raise CoupledLunarError("parameters must be exact CoupledLunarParameters")
    if type(integration_spec) is not CoupledLunarIntegrationSpec:
        raise CoupledLunarError(
            "integration_spec must be exact CoupledLunarIntegrationSpec"
        )
    if initial_state.epoch != integration_spec.checkpoint_epochs[0]:
        raise CoupledLunarError(
            "initial state epoch must equal the first checkpoint exactly"
        )
    np = _numpy()
    backend = resolve_backend("numpy")
    generalized_positions, generalized_velocities = _pack_state(initial_state)
    position_carry = np.zeros(10, dtype=np.float64)
    velocity_carry = np.zeros(12, dtype=np.float64)
    mantle_inertia = np.ascontiguousarray(
        np.diag(parameters.mantle_inertia_over_mr2), dtype=np.float64
    )
    zero_matrix = np.zeros((3, 3), dtype=np.float64)
    zero_vector = np.zeros(3, dtype=np.float64)
    initial_invariants = _invariants(
        generalized_positions, generalized_velocities, parameters
    )
    energy_scale = max(abs(initial_invariants["energy"]), np.finfo(np.float64).tiny)
    angular_scale = max(
        float(np.linalg.norm(initial_invariants["total_angular"])),
        float(np.linalg.norm(initial_invariants["orbital_angular"])),
        float(np.linalg.norm(initial_invariants["spin_angular"])),
        np.finfo(np.float64).tiny,
    )
    momentum_scale = max(
        parameters.moon_gm_km3_s2 * initial_invariants["relative_speed"],
        np.finfo(np.float64).tiny,
    )
    position_scale = max(initial_invariants["distance"], np.finfo(np.float64).tiny)
    maxima = {
        "preprojection": 0.0,
        "orthogonality": 0.0,
        "determinant": 0.0,
        "momentum": 0.0,
        "barycenter": 0.0,
        "angular": 0.0,
        "energy": 0.0,
        "torque_balance": 0.0,
        "pair_reaction": 0.0,
        "positive_viscous_power": 0.0,
        "quadrupole_acceleration": 0.0,
        "cmb_torque": 0.0,
    }
    minimum_separation = initial_invariants["distance"]
    accepted_epochs: list[float] = []
    accepted_magnitudes: list[float] = []
    checkpoints: list[CoupledLunarState] = [initial_state]
    ledger = hashlib.sha256(LEDGER_DOMAIN)
    epoch = initial_state.epoch
    direction = 1.0 if integration_spec.checkpoint_epochs[-1] > epoch else -1.0

    for checkpoint_epoch in integration_spec.checkpoint_epochs[1:]:
        while epoch != checkpoint_epoch:
            if len(accepted_epochs) >= integration_spec.maximum_steps:
                raise CoupledLunarStepLimitError("maximum_steps exhausted")
            remaining = checkpoint_epoch - epoch
            if direction * remaining <= 0.0 or not math.isfinite(remaining):
                raise CoupledLunarError("checkpoint schedule lost monotone progress")
            magnitude = min(integration_spec.maximum_step, abs(remaining))
            endpoint = (
                checkpoint_epoch
                if magnitude >= abs(remaining)
                else epoch + math.copysign(magnitude, direction)
            )
            if endpoint == epoch or not math.isfinite(endpoint):
                raise CoupledLunarError("fixed step cannot advance binary64 epoch")
            step = endpoint - epoch

            def derivative(
                _stage_epoch: float,
                stage_positions: object,
                stage_velocities: object,
            ) -> tuple[object, object]:
                state = np.asarray(stage_positions, dtype=np.float64)
                rates = np.asarray(stage_velocities, dtype=np.float64)
                earth = state[:3]
                moon = state[3:6]
                quaternion = _unit_quaternion(state[6:10])
                earth_velocity = rates[:3]
                moon_velocity = rates[3:6]
                mantle_rate = np.ascontiguousarray(rates[6:9])
                core_rate = np.ascontiguousarray(rates[9:12])
                separation = earth - moon
                distance = float(np.linalg.norm(separation))
                if not math.isfinite(distance) or distance <= 0.0:
                    raise CoupledLunarError("Earth and Moon collided or became invalid")
                inverse_cube = 1.0 / distance**3
                earth_acceleration = np.ascontiguousarray(
                    -parameters.moon_gm_km3_s2 * inverse_cube * separation,
                    dtype=np.float64,
                )
                moon_acceleration = np.ascontiguousarray(
                    parameters.earth_gm_km3_s2 * inverse_cube * separation,
                    dtype=np.float64,
                )
                interaction = _quadrupole_interaction(
                    quaternion, earth, moon, parameters
                )
                earth_acceleration += interaction.earth_acceleration_km_s2
                moon_acceleration += interaction.moon_acceleration_km_s2
                try:
                    rotation = lunar_mantle_core_angular_accelerations(
                        parameters.core_coupling,
                        mantle_angular_velocity_body=mantle_rate,
                        core_angular_velocity_body=core_rate,
                        mantle_inertia_over_mr2=mantle_inertia,
                        mantle_inertia_rate_over_mr2_per_second=zero_matrix,
                        external_torque_over_mr2=(
                            interaction.torque_on_mantle_over_mr2
                        ),
                        geodetic_precession_body=zero_vector,
                    )
                except LunarFluidCoreError as exc:
                    raise CoupledLunarError(str(exc)) from exc
                body_to_inertial = _body_to_inertial(quaternion)
                orbital_torque = np.cross(
                    separation,
                    parameters.earth_gm_km3_s2
                    * interaction.earth_acceleration_km_s2,
                )
                spin_torque = (
                    parameters.moon_gm_km3_s2
                    * parameters.lunar_reference_radius_km**2
                    * (body_to_inertial @ interaction.torque_on_mantle_over_mr2)
                )
                figure_moments = parameters.figure_inertia_over_mr2
                figure_anisotropy = max(figure_moments) - min(figure_moments)
                characteristic_quadrupole_torque = (
                    parameters.earth_gm_km3_s2
                    * parameters.moon_gm_km3_s2
                    * parameters.lunar_reference_radius_km**2
                    * figure_anisotropy
                    / distance**3
                )
                torque_scale = max(
                    float(np.linalg.norm(spin_torque)),
                    characteristic_quadrupole_torque,
                    np.finfo(np.float64).tiny,
                )
                maxima["torque_balance"] = max(
                    maxima["torque_balance"],
                    float(np.linalg.norm(orbital_torque + spin_torque))
                    / torque_scale,
                )
                reaction = (
                    parameters.earth_gm_km3_s2
                    * interaction.earth_acceleration_km_s2
                    + parameters.moon_gm_km3_s2
                    * interaction.moon_acceleration_km_s2
                )
                reaction_scale = max(
                    float(
                        np.linalg.norm(
                            parameters.earth_gm_km3_s2
                            * interaction.earth_acceleration_km_s2
                        )
                    ),
                    np.finfo(np.float64).tiny,
                )
                maxima["pair_reaction"] = max(
                    maxima["pair_reaction"],
                    float(np.linalg.norm(reaction)) / reaction_scale,
                )
                viscous_power = float(
                    np.dot(
                        mantle_rate - core_rate,
                        rotation.viscous_torque_on_mantle_body,
                    )
                )
                maxima["positive_viscous_power"] = max(
                    maxima["positive_viscous_power"], viscous_power
                )
                maxima["quadrupole_acceleration"] = max(
                    maxima["quadrupole_acceleration"],
                    float(np.linalg.norm(interaction.moon_acceleration_km_s2)),
                )
                maxima["cmb_torque"] = max(
                    maxima["cmb_torque"],
                    float(np.linalg.norm(rotation.cmb_torque_on_mantle_body)),
                )
                return (
                    np.ascontiguousarray(
                        np.concatenate(
                            (
                                earth_velocity,
                                moon_velocity,
                                _quaternion_derivative(quaternion, mantle_rate),
                            )
                        ),
                        dtype=np.float64,
                    ),
                    np.ascontiguousarray(
                        np.concatenate(
                            (
                                earth_acceleration,
                                moon_acceleration,
                                rotation.mantle_angular_acceleration_body,
                                rotation.core_angular_acceleration_body,
                            )
                        ),
                        dtype=np.float64,
                    ),
                )

            trial = _rkf78_step(
                backend=backend,
                epoch=epoch,
                endpoint_epoch=endpoint,
                step=step,
                positions=generalized_positions,
                velocities=generalized_velocities,
                position_carry=position_carry,
                velocity_carry=velocity_carry,
                derivative=derivative,
            )
            accepted_positions = np.ascontiguousarray(
                trial.accepted_positions, dtype=np.float64
            )
            preprojection_norm = float(np.linalg.norm(accepted_positions[6:10]))
            maxima["preprojection"] = max(
                maxima["preprojection"], abs(preprojection_norm - 1.0)
            )
            accepted_positions[6:10] = _unit_quaternion(
                accepted_positions[6:10]
            )
            generalized_positions = accepted_positions
            generalized_velocities = np.ascontiguousarray(
                trial.accepted_velocities, dtype=np.float64
            )
            position_carry = np.ascontiguousarray(
                trial.accepted_position_carry, dtype=np.float64
            )
            position_carry[6:10] = 0.0
            velocity_carry = np.ascontiguousarray(
                trial.accepted_velocity_carry, dtype=np.float64
            )
            epoch = endpoint
            accepted_epochs.append(epoch)
            accepted_magnitudes.append(abs(step))
            ledger.update(struct.pack(">dd", epoch, abs(step)))
            ledger.update(generalized_positions.tobytes(order="C"))
            ledger.update(generalized_velocities.tobytes(order="C"))

            current = _invariants(
                generalized_positions, generalized_velocities, parameters
            )
            minimum_separation = min(minimum_separation, current["distance"])
            elapsed = epoch - initial_state.epoch
            expected_barycenter = initial_invariants["barycenter"] + (
                initial_invariants["total_momentum"]
                / (parameters.earth_gm_km3_s2 + parameters.moon_gm_km3_s2)
            ) * elapsed
            maxima["momentum"] = max(
                maxima["momentum"],
                float(
                    np.linalg.norm(
                        current["total_momentum"]
                        - initial_invariants["total_momentum"]
                    )
                )
                / momentum_scale,
            )
            maxima["barycenter"] = max(
                maxima["barycenter"],
                float(np.linalg.norm(current["barycenter"] - expected_barycenter))
                / position_scale,
            )
            maxima["angular"] = max(
                maxima["angular"],
                float(
                    np.linalg.norm(
                        current["total_angular"]
                        - initial_invariants["total_angular"]
                    )
                )
                / angular_scale,
            )
            maxima["energy"] = max(
                maxima["energy"],
                abs(current["energy"] - initial_invariants["energy"])
                / energy_scale,
            )
            orthogonality, determinant = _rotation_metrics(
                generalized_positions[6:10]
            )
            maxima["orthogonality"] = max(
                maxima["orthogonality"], orthogonality
            )
            maxima["determinant"] = max(maxima["determinant"], determinant)

        checkpoints.append(
            _state_from_packed(epoch, generalized_positions, generalized_velocities)
        )

    return CoupledLunarResult(
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
        maximum_total_angular_momentum_relative_drift=maxima["angular"],
        maximum_energy_relative_change=maxima["energy"],
        maximum_orbital_spin_torque_balance_relative=maxima["torque_balance"],
        maximum_pair_reaction_relative=maxima["pair_reaction"],
        maximum_positive_viscous_power_over_mr2=maxima["positive_viscous_power"],
        maximum_quadrupole_acceleration_km_s2=maxima["quadrupole_acceleration"],
        maximum_cmb_torque_over_mr2=maxima["cmb_torque"],
        minimum_separation_km=minimum_separation,
    )


__all__ = [
    "COUPLED_LUNAR_MODEL_ID",
    "MODEL_SCOPE",
    "SCIENTIFIC_CLAIM_STATE",
    "CoupledLunarError",
    "CoupledLunarIntegrationSpec",
    "CoupledLunarParameters",
    "CoupledLunarResult",
    "CoupledLunarState",
    "CoupledLunarStepLimitError",
    "integrate_coupled_lunar_trajectory",
]
