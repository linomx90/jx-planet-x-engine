"""Simultaneous Sun--Earth--Moon dynamics with delayed lunar deformation.

This additive v3 research model advances the three translational bodies and
the lunar mantle/core rotation in one fixed-step RKF78 solve.  The delayed
Earth tide, spin distortion, mantle inertia, inertia derivative, exterior
figure, source reactions, and lunar torques are evaluated as one package.
Earth and Sun quadrupole forces and torques come from the same tensor
potential.  The caller supplies only the pre-start delay history; once the
integration starts, every delayed value comes from the matching RKF78 stage.

The delay must be an integer number of fixed steps.  This deliberate
restriction makes the history ownership auditable and prevents interpolation
from becoming an unregistered physical increment.  Geodetic transport is not
included because its separate promotion gate failed.  This module is
``SCREENING_ONLY`` research code, not a DE440 generator or an LLR model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Callable

from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT, _C, _rkf78_step

from . import lunar_coupled as _v1
from . import lunar_coupled_solar as _v2
from .lunar_fluid_core import (
    LunarFluidCoreError,
    lunar_mantle_core_angular_accelerations,
)
from .lunar_mantle_deformation import (
    LunarMantleDeformationError,
    LunarMantleDeformationParameters,
    evaluate_lunar_mantle_deformation,
    evaluate_lunar_tensor_quadrupole_interaction,
)


DEFORMABLE_SOLAR_COUPLED_LUNAR_MODEL_ID = (
    "solar-system.experimental.sun-earth-moon-delayed-deformable-"
    "mantle-fluid-core-j2.v3"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
MODEL_SCOPE = (
    "SUN_EARTH_MOON_MONOPOLES_DELAYED_LUNAR_TIDE_AND_SPIN_FIGURE_"
    "COMMON_POTENTIAL_REACTIONS_EARTH_J2_FIXED_POLE_MANTLE_FLUID_CORE"
)
LEDGER_DOMAIN = b"jxplanetx.deformable-solar-coupled-lunar.stage-ledger.v3\0"
RKF78_STAGE_ABSCISSAE = tuple(_C)


class DeformableSolarCoupledLunarError(ValueError):
    """The v3 delay, physics, or ownership contract was violated."""


class DeformableSolarCoupledLunarStepLimitError(
    DeformableSolarCoupledLunarError
):
    """The declared exact fixed-step work cap was exhausted."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise DeformableSolarCoupledLunarError(
            "deformable coupled lunar integration requires NumPy"
        ) from exc
    return np


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DeformableSolarCoupledLunarError(
            f"{label} must be a finite built-in float"
        )
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
        raise DeformableSolarCoupledLunarError(
            f"{label} must be finite C-contiguous NumPy float64 with shape {shape}"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class DeformableSolarCoupledLunarParameters:
    """Consistent v2 orbit parameters plus the complete deformation package."""

    solar: _v2.SolarCoupledLunarParameters
    deformation: LunarMantleDeformationParameters
    model_id: str = DEFORMABLE_SOLAR_COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.solar) is not _v2.SolarCoupledLunarParameters:
            raise DeformableSolarCoupledLunarError(
                "solar must be an exact SolarCoupledLunarParameters"
            )
        if type(self.deformation) is not LunarMantleDeformationParameters:
            raise DeformableSolarCoupledLunarError(
                "deformation must be exact LunarMantleDeformationParameters"
            )
        lunar = self.solar.lunar
        checks = (
            (self.deformation.undistorted_total_inertia_over_mr2, lunar.figure_inertia_over_mr2, "total inertia"),
            (self.deformation.fixed_core_inertia_over_mr2, lunar.core_coupling.core_inertia_over_mr2, "core inertia"),
            (self.deformation.undistorted_mantle_inertia_over_mr2, lunar.mantle_inertia_over_mr2, "mantle inertia"),
            (self.deformation.earth_gm_km3_s2, lunar.earth_gm_km3_s2, "Earth GM"),
            (self.deformation.moon_gm_km3_s2, lunar.moon_gm_km3_s2, "Moon GM"),
            (self.deformation.lunar_reference_radius_km, lunar.lunar_reference_radius_km, "lunar radius"),
        )
        for left, right, label in checks:
            if left != right:
                raise DeformableSolarCoupledLunarError(
                    f"deformation and solar {label} must match exactly"
                )
        if self.model_id != DEFORMABLE_SOLAR_COUPLED_LUNAR_MODEL_ID:
            raise DeformableSolarCoupledLunarError("model_id changed")
        if self.scope != MODEL_SCOPE:
            raise DeformableSolarCoupledLunarError("scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise DeformableSolarCoupledLunarError(
                "scientific_claim_state changed"
            )
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise DeformableSolarCoupledLunarError(
                    f"{label} must be exact false"
                )


@dataclass(frozen=True, slots=True, eq=False)
class DeformableSolarCoupledLunarIntegrationSpec:
    """An exact fixed-step delay lattice and checkpoint roster."""

    delay_divisor: int
    checkpoint_step_indices: tuple[int, ...]
    maximum_steps: int = 10_000_000

    def __post_init__(self) -> None:
        if type(self.delay_divisor) is not int or self.delay_divisor < 1:
            raise DeformableSolarCoupledLunarError(
                "delay_divisor must be a positive exact integer"
            )
        if (
            type(self.checkpoint_step_indices) is not tuple
            or len(self.checkpoint_step_indices) < 2
            or any(type(value) is not int for value in self.checkpoint_step_indices)
            or self.checkpoint_step_indices[0] != 0
            or any(
                right <= left
                for left, right in zip(
                    self.checkpoint_step_indices,
                    self.checkpoint_step_indices[1:],
                )
            )
        ):
            raise DeformableSolarCoupledLunarError(
                "checkpoint_step_indices must be strictly increasing exact integers starting at zero"
            )
        if type(self.maximum_steps) is not int or self.maximum_steps < 1:
            raise DeformableSolarCoupledLunarError(
                "maximum_steps must be a positive exact integer"
            )
        if self.checkpoint_step_indices[-1] > self.maximum_steps:
            raise DeformableSolarCoupledLunarStepLimitError(
                "final checkpoint exceeds maximum_steps"
            )

    def step_seconds(
        self, parameters: DeformableSolarCoupledLunarParameters
    ) -> float:
        if type(parameters) is not DeformableSolarCoupledLunarParameters:
            raise DeformableSolarCoupledLunarError(
                "parameters must be exact DeformableSolarCoupledLunarParameters"
            )
        step = parameters.deformation.response_delay_seconds / self.delay_divisor
        if self.delay_divisor * step != parameters.deformation.response_delay_seconds:
            raise DeformableSolarCoupledLunarError(
                "response delay is not exactly recoverable from the fixed-step lattice"
            )
        return step


@dataclass(frozen=True, slots=True, eq=False)
class DelayedLunarHistorySample:
    """One caller-owned pre-start sample on an exact RKF78 stage epoch."""

    epoch: float
    mantle_quaternion_body_to_inertial: object
    mantle_angular_velocity_body_s: object
    mantle_angular_acceleration_body_s2: object
    earth_minus_moon_state_inertial_km_km_s: object

    def __post_init__(self) -> None:
        _finite_float(self.epoch, "epoch")
        quaternion = _owned_array(
            self.mantle_quaternion_body_to_inertial,
            (4,),
            "mantle_quaternion_body_to_inertial",
        )
        try:
            normalized = _v1._unit_quaternion(quaternion)
        except _v1.CoupledLunarError as exc:
            raise DeformableSolarCoupledLunarError(str(exc)) from exc
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
            "mantle_angular_acceleration_body_s2",
            _owned_array(
                self.mantle_angular_acceleration_body_s2,
                (3,),
                "mantle_angular_acceleration_body_s2",
            ),
        )
        object.__setattr__(
            self,
            "earth_minus_moon_state_inertial_km_km_s",
            _owned_array(
                self.earth_minus_moon_state_inertial_km_km_s,
                (6,),
                "earth_minus_moon_state_inertial_km_km_s",
            ),
        )


PrehistoryProvider = Callable[[float], DelayedLunarHistorySample]


@dataclass(frozen=True, slots=True, eq=False)
class DeformableSolarCoupledLunarResult:
    """Owned checkpoints and fail-closed v3 structural diagnostics."""

    parameters: DeformableSolarCoupledLunarParameters
    integration_spec: DeformableSolarCoupledLunarIntegrationSpec
    checkpoints: tuple[_v2.SolarCoupledLunarState, ...]
    step_seconds: float
    accepted_steps: int
    rkf78_stage_evaluations: int
    prehistory_provider_queries: int
    post_start_prehistory_provider_queries: int
    maximum_stage_history_steps: int
    stage_ledger_sha256: str
    maximum_quaternion_preprojection_norm_error: float
    maximum_rotation_orthogonality_error: float
    maximum_rotation_determinant_error: float
    maximum_linear_momentum_relative_drift: float
    maximum_barycenter_relative_drift: float
    maximum_pair_reaction_relative: float
    maximum_lunar_figure_torque_balance_relative: float
    maximum_rotational_balance_relative: float
    maximum_positive_viscous_power_over_mr2: float
    minimum_mantle_inertia_eigenvalue_over_mr2: float
    maximum_earth_lunar_figure_acceleration_km_s2: float
    maximum_solar_lunar_figure_acceleration_km_s2: float
    maximum_earth_j2_acceleration_km_s2: float
    geodetic_transport_included: bool = False
    model_id: str = DEFORMABLE_SOLAR_COUPLED_LUNAR_MODEL_ID
    scope: str = MODEL_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False

    @property
    def final_state(self) -> _v2.SolarCoupledLunarState:
        return self.checkpoints[-1]


@dataclass(frozen=True, slots=True, eq=False)
class _DerivativeAudit:
    delayed_stage_sample: DelayedLunarHistorySample
    pair_reaction_relative: float
    torque_balance_relative: float
    rotational_balance_relative: float
    positive_viscous_power_over_mr2: float
    minimum_mantle_inertia_eigenvalue_over_mr2: float
    earth_figure_acceleration_km_s2: float
    solar_figure_acceleration_km_s2: float
    earth_j2_acceleration_km_s2: float


def _pack_state(state: _v2.SolarCoupledLunarState):
    return _v2._pack_state(state)


def _history_sample_from_stage(
    epoch: float,
    xyz: object,
    uvw: object,
    quaternion: object,
    mantle_rate: object,
    mantle_acceleration: object,
) -> DelayedLunarHistorySample:
    np = _numpy()
    positions = np.asarray(xyz, dtype=np.float64)
    velocities = np.asarray(uvw, dtype=np.float64)
    return DelayedLunarHistorySample(
        epoch=float(epoch),
        mantle_quaternion_body_to_inertial=np.ascontiguousarray(quaternion),
        mantle_angular_velocity_body_s=np.ascontiguousarray(mantle_rate),
        mantle_angular_acceleration_body_s2=np.ascontiguousarray(
            mantle_acceleration
        ),
        earth_minus_moon_state_inertial_km_km_s=np.ascontiguousarray(
            np.concatenate(
                (
                    positions[_v2.EARTH_INDEX] - positions[_v2.MOON_INDEX],
                    velocities[_v2.EARTH_INDEX] - velocities[_v2.MOON_INDEX],
                )
            )
        ),
    )


def _evaluate_derivative(
    stage_epoch: float,
    stage_positions: object,
    stage_velocities: object,
    delayed: DelayedLunarHistorySample,
    parameters: DeformableSolarCoupledLunarParameters,
) -> tuple[object, object, _DerivativeAudit]:
    """Evaluate one simultaneous v3 derivative and its structural audit."""

    if type(delayed) is not DelayedLunarHistorySample:
        raise DeformableSolarCoupledLunarError(
            "delayed must be an exact DelayedLunarHistorySample"
        )
    _finite_float(stage_epoch, "stage_epoch")
    np = _numpy()
    state = np.asarray(stage_positions, dtype=np.float64)
    rates = np.asarray(stage_velocities, dtype=np.float64)
    if (
        state.shape != (13,)
        or rates.shape != (15,)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(rates))
    ):
        raise DeformableSolarCoupledLunarError(
            "packed stage state is invalid"
        )
    xyz = state[:9].reshape(3, 3)
    uvw = rates[:9].reshape(3, 3)
    quaternion = _v1._unit_quaternion(np.ascontiguousarray(state[9:13]))
    mantle_rate = np.ascontiguousarray(rates[9:12])
    core_rate = np.ascontiguousarray(rates[12:15])
    solar = parameters.solar
    lunar = solar.lunar
    gms = np.asarray(solar.gravitational_parameters_km3_s2, dtype=np.float64)
    accelerations = np.zeros((3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(left + 1, 3):
            separation = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(separation))
            if not math.isfinite(distance) or distance <= 0.0:
                raise DeformableSolarCoupledLunarError(
                    "two translational bodies collided or became invalid"
                )
            inverse_cube = 1.0 / distance**3
            accelerations[left] -= gms[right] * inverse_cube * separation
            accelerations[right] += gms[left] * inverse_cube * separation
    try:
        package = evaluate_lunar_mantle_deformation(
            parameters.deformation,
            delayed_quaternion_body_to_inertial=(
                delayed.mantle_quaternion_body_to_inertial
            ),
            delayed_mantle_angular_velocity_body_s=(
                delayed.mantle_angular_velocity_body_s
            ),
            delayed_mantle_angular_acceleration_body_s2=(
                delayed.mantle_angular_acceleration_body_s2
            ),
            delayed_earth_relative_state_inertial_km_km_s=(
                delayed.earth_minus_moon_state_inertial_km_km_s
            ),
        )
    except LunarMantleDeformationError as exc:
        raise DeformableSolarCoupledLunarError(str(exc)) from exc
    source_rows = (
        (_v2.SUN_INDEX, solar.sun_gm_km3_s2),
        (_v2.EARTH_INDEX, lunar.earth_gm_km3_s2),
    )
    external_torque = np.zeros(3, dtype=np.float64)
    maximum_reaction = 0.0
    maximum_torque_balance = 0.0
    figure_accelerations: dict[int, float] = {}
    rotation = _v1._body_to_inertial(quaternion)
    tiny = np.finfo(np.float64).tiny
    for source_index, source_gm in source_rows:
        try:
            interaction = evaluate_lunar_tensor_quadrupole_interaction(
                quaternion,
                np.ascontiguousarray(xyz[source_index]),
                np.ascontiguousarray(xyz[_v2.MOON_INDEX]),
                np.ascontiguousarray(package.exterior_figure_inertia_over_mr2),
                source_gm_km3_s2=float(source_gm),
                moon_gm_km3_s2=float(lunar.moon_gm_km3_s2),
                lunar_reference_radius_km=float(
                    lunar.lunar_reference_radius_km
                ),
            )
        except LunarMantleDeformationError as exc:
            raise DeformableSolarCoupledLunarError(str(exc)) from exc
        accelerations[source_index] += interaction.source_acceleration_km_s2
        accelerations[_v2.MOON_INDEX] += interaction.moon_acceleration_km_s2
        external_torque += interaction.torque_on_mantle_over_mr2
        reaction = (
            source_gm * interaction.source_acceleration_km_s2
            + lunar.moon_gm_km3_s2 * interaction.moon_acceleration_km_s2
        )
        reaction_scale = max(
            float(
                np.linalg.norm(
                    source_gm * interaction.source_acceleration_km_s2
                )
            ),
            tiny,
        )
        maximum_reaction = max(
            maximum_reaction, float(np.linalg.norm(reaction)) / reaction_scale
        )
        separation = xyz[source_index] - xyz[_v2.MOON_INDEX]
        orbital_torque = np.cross(
            separation,
            source_gm * interaction.source_acceleration_km_s2,
        )
        spin_torque = (
            lunar.moon_gm_km3_s2
            * lunar.lunar_reference_radius_km**2
            * (rotation @ interaction.torque_on_mantle_over_mr2)
        )
        torque_scale = max(
            float(np.linalg.norm(orbital_torque)),
            float(np.linalg.norm(spin_torque)),
            tiny,
        )
        maximum_torque_balance = max(
            maximum_torque_balance,
            float(np.linalg.norm(orbital_torque + spin_torque)) / torque_scale,
        )
        figure_accelerations[source_index] = float(
            np.linalg.norm(interaction.moon_acceleration_km_s2)
        )
    j2 = _v2._earth_j2_interaction(
        np.ascontiguousarray(xyz[_v2.EARTH_INDEX]),
        np.ascontiguousarray(xyz[_v2.MOON_INDEX]),
        solar,
    )
    accelerations[_v2.EARTH_INDEX] += j2.earth_acceleration_km_s2
    accelerations[_v2.MOON_INDEX] += j2.moon_acceleration_km_s2
    j2_reaction = (
        lunar.earth_gm_km3_s2 * j2.earth_acceleration_km_s2
        + lunar.moon_gm_km3_s2 * j2.moon_acceleration_km_s2
    )
    j2_scale = max(
        float(
            np.linalg.norm(
                lunar.moon_gm_km3_s2 * j2.moon_acceleration_km_s2
            )
        ),
        tiny,
    )
    maximum_reaction = max(
        maximum_reaction, float(np.linalg.norm(j2_reaction)) / j2_scale
    )
    zero_geodetic = np.zeros(3, dtype=np.float64)
    try:
        angular = lunar_mantle_core_angular_accelerations(
            lunar.core_coupling,
            mantle_angular_velocity_body=mantle_rate,
            core_angular_velocity_body=core_rate,
            mantle_inertia_over_mr2=np.ascontiguousarray(
                package.mantle_inertia_over_mr2
            ),
            mantle_inertia_rate_over_mr2_per_second=np.ascontiguousarray(
                package.mantle_inertia_rate_over_mr2_per_second
            ),
            external_torque_over_mr2=np.ascontiguousarray(external_torque),
            geodetic_precession_body=zero_geodetic,
        )
    except LunarFluidCoreError as exc:
        raise DeformableSolarCoupledLunarError(str(exc)) from exc
    alpha_m = np.ascontiguousarray(angular.mantle_angular_acceleration_body)
    alpha_c = np.ascontiguousarray(angular.core_angular_acceleration_body)
    mantle_inertia = package.mantle_inertia_over_mr2
    inertia_rate = package.mantle_inertia_rate_over_mr2_per_second
    mantle_balance = (
        mantle_inertia @ alpha_m
        + inertia_rate @ mantle_rate
        + np.cross(mantle_rate, mantle_inertia @ mantle_rate)
        - external_torque
        - angular.cmb_torque_on_mantle_body
    )
    core = np.asarray(
        lunar.core_coupling.core_inertia_over_mr2, dtype=np.float64
    )
    core_balance = (
        core * alpha_c
        + np.cross(mantle_rate, core * core_rate)
        + angular.cmb_torque_on_mantle_body
    )
    balance_scale = max(
        float(np.max(np.abs(external_torque))),
        float(np.max(np.abs(angular.cmb_torque_on_mantle_body))),
        float(np.max(np.abs(mantle_inertia @ alpha_m))),
        float(np.max(np.abs(core * alpha_c))),
        tiny,
    )
    stage_sample = _history_sample_from_stage(
        stage_epoch,
        xyz,
        uvw,
        quaternion,
        mantle_rate,
        alpha_m,
    )
    return (
        np.ascontiguousarray(
            np.concatenate(
                (
                    uvw.reshape(9),
                    _v1._quaternion_derivative(quaternion, mantle_rate),
                )
            )
        ),
        np.ascontiguousarray(
            np.concatenate((accelerations.reshape(9), alpha_m, alpha_c))
        ),
        _DerivativeAudit(
            delayed_stage_sample=stage_sample,
            pair_reaction_relative=maximum_reaction,
            torque_balance_relative=maximum_torque_balance,
            rotational_balance_relative=max(
                float(np.max(np.abs(mantle_balance))) / balance_scale,
                float(np.max(np.abs(core_balance))) / balance_scale,
            ),
            positive_viscous_power_over_mr2=max(
                0.0,
                float(
                    np.dot(
                        mantle_rate - core_rate,
                        angular.viscous_torque_on_mantle_body,
                    )
                ),
            ),
            minimum_mantle_inertia_eigenvalue_over_mr2=float(
                np.min(np.linalg.eigvalsh(mantle_inertia))
            ),
            earth_figure_acceleration_km_s2=figure_accelerations[
                _v2.EARTH_INDEX
            ],
            solar_figure_acceleration_km_s2=figure_accelerations[
                _v2.SUN_INDEX
            ],
            earth_j2_acceleration_km_s2=float(
                np.linalg.norm(j2.moon_acceleration_km_s2)
            ),
        ),
    )


def integrate_deformable_solar_coupled_lunar_trajectory(
    initial_state: _v2.SolarCoupledLunarState,
    parameters: DeformableSolarCoupledLunarParameters,
    integration_spec: DeformableSolarCoupledLunarIntegrationSpec,
    prehistory_provider: PrehistoryProvider,
) -> DeformableSolarCoupledLunarResult:
    """Advance v3 on an exact fixed-step delay lattice."""

    if type(initial_state) is not _v2.SolarCoupledLunarState:
        raise DeformableSolarCoupledLunarError(
            "initial_state must be exact SolarCoupledLunarState"
        )
    if type(parameters) is not DeformableSolarCoupledLunarParameters:
        raise DeformableSolarCoupledLunarError(
            "parameters must be exact DeformableSolarCoupledLunarParameters"
        )
    if type(integration_spec) is not DeformableSolarCoupledLunarIntegrationSpec:
        raise DeformableSolarCoupledLunarError(
            "integration_spec must be exact DeformableSolarCoupledLunarIntegrationSpec"
        )
    if not callable(prehistory_provider):
        raise DeformableSolarCoupledLunarError(
            "prehistory_provider must be callable"
        )
    if len(RKF78_STAGE_ABSCISSAE) != RKF78_STAGE_COUNT:
        raise DeformableSolarCoupledLunarError("RKF78 stage roster changed")
    np = _numpy()
    step = integration_spec.step_seconds(parameters)
    delay = parameters.deformation.response_delay_seconds
    divisor = integration_spec.delay_divisor
    positions, velocities = _pack_state(initial_state)
    position_carry = np.zeros(13, dtype=np.float64)
    velocity_carry = np.zeros(15, dtype=np.float64)
    backend = resolve_backend("numpy")
    stage_history: dict[int, tuple[DelayedLunarHistorySample, ...]] = {}
    checkpoint_set = set(integration_spec.checkpoint_step_indices[1:])
    checkpoints = [initial_state]
    ledger = hashlib.sha256(
        LEDGER_DOMAIN + struct.pack(">qdd", divisor, step, delay)
    )
    initial_epoch = initial_state.epoch
    gms = np.asarray(
        parameters.solar.gravitational_parameters_km3_s2, dtype=np.float64
    )
    initial_momentum = np.sum(
        gms[:, None] * initial_state.velocities_km_s, axis=0
    )
    total_gm = float(np.sum(gms))
    initial_barycenter = (
        np.sum(gms[:, None] * initial_state.positions_km, axis=0) / total_gm
    )
    relative_speed = max(
        float(
            np.linalg.norm(
                initial_state.velocities_km_s[_v2.EARTH_INDEX]
                - initial_state.velocities_km_s[_v2.MOON_INDEX]
            )
        ),
        np.finfo(np.float64).tiny,
    )
    momentum_scale = max(
        parameters.solar.lunar.moon_gm_km3_s2 * relative_speed,
        np.finfo(np.float64).tiny,
    )
    position_scale = max(
        float(
            np.linalg.norm(
                initial_state.positions_km[_v2.EARTH_INDEX]
                - initial_state.positions_km[_v2.MOON_INDEX]
            )
        ),
        np.finfo(np.float64).tiny,
    )
    maxima = {
        name: 0.0
        for name in (
            "preprojection",
            "orthogonality",
            "determinant",
            "momentum",
            "barycenter",
            "reaction",
            "torque_balance",
            "rotation_balance",
            "positive_viscous_power",
            "earth_figure",
            "solar_figure",
            "earth_j2",
        )
    }
    minimum_inertia = math.inf
    prehistory_queries = 0
    post_start_queries = 0
    maximum_history_steps = 0
    final_step_index = integration_spec.checkpoint_step_indices[-1]
    for step_index in range(final_step_index):
        if step_index >= integration_spec.maximum_steps:
            raise DeformableSolarCoupledLunarStepLimitError(
                "maximum_steps exhausted"
            )
        stage_index = 0
        current_stage_rows: list[DelayedLunarHistorySample] = []

        def derivative(
            stage_elapsed: float,
            stage_positions: object,
            stage_velocities: object,
        ) -> tuple[object, object]:
            nonlocal stage_index, prehistory_queries, post_start_queries
            nonlocal minimum_inertia
            if stage_index >= RKF78_STAGE_COUNT:
                raise DeformableSolarCoupledLunarError(
                    "RKF78 evaluated an unexpected extra stage"
                )
            abscissa = RKF78_STAGE_ABSCISSAE[stage_index]
            expected_elapsed = (step_index + abscissa) * step
            tolerance = 8.0 * math.ulp(max(abs(expected_elapsed), step))
            if abs(stage_elapsed - expected_elapsed) > tolerance:
                raise DeformableSolarCoupledLunarError(
                    "RKF78 stage left the exact delay lattice"
                )
            delayed_elapsed = (step_index - divisor + abscissa) * step
            delayed_epoch = initial_epoch + delayed_elapsed
            prior_step = step_index - divisor
            if prior_step >= 0:
                delayed = stage_history[prior_step][stage_index]
            else:
                if delayed_epoch > initial_epoch + 8.0 * math.ulp(
                    max(abs(initial_epoch), 1.0)
                ):
                    post_start_queries += 1
                    raise DeformableSolarCoupledLunarError(
                        "prehistory provider was queried after the start epoch"
                    )
                delayed = prehistory_provider(float(delayed_epoch))
                prehistory_queries += 1
                if type(delayed) is not DelayedLunarHistorySample:
                    raise DeformableSolarCoupledLunarError(
                        "prehistory provider must return exact DelayedLunarHistorySample"
                    )
                epoch_tolerance = 8.0 * math.ulp(
                    max(abs(delayed_epoch), abs(delay), 1.0)
                )
                if abs(delayed.epoch - delayed_epoch) > epoch_tolerance:
                    raise DeformableSolarCoupledLunarError(
                        "prehistory sample epoch does not match the requested delay epoch"
                    )
            derivative_positions, derivative_velocities, audit = (
                _evaluate_derivative(
                    float(initial_epoch + expected_elapsed),
                    stage_positions,
                    stage_velocities,
                    delayed,
                    parameters,
                )
            )
            current_stage_rows.append(audit.delayed_stage_sample)
            maxima["reaction"] = max(
                maxima["reaction"], audit.pair_reaction_relative
            )
            maxima["torque_balance"] = max(
                maxima["torque_balance"], audit.torque_balance_relative
            )
            maxima["rotation_balance"] = max(
                maxima["rotation_balance"], audit.rotational_balance_relative
            )
            maxima["positive_viscous_power"] = max(
                maxima["positive_viscous_power"],
                audit.positive_viscous_power_over_mr2,
            )
            maxima["earth_figure"] = max(
                maxima["earth_figure"],
                audit.earth_figure_acceleration_km_s2,
            )
            maxima["solar_figure"] = max(
                maxima["solar_figure"],
                audit.solar_figure_acceleration_km_s2,
            )
            maxima["earth_j2"] = max(
                maxima["earth_j2"], audit.earth_j2_acceleration_km_s2
            )
            minimum_inertia = min(
                minimum_inertia,
                audit.minimum_mantle_inertia_eigenvalue_over_mr2,
            )
            ledger.update(
                struct.pack(
                    ">qidd",
                    step_index,
                    stage_index,
                    initial_epoch + expected_elapsed,
                    delayed_epoch,
                )
            )
            stage_index += 1
            return derivative_positions, derivative_velocities

        left = step_index * step
        right = (step_index + 1) * step
        trial = _rkf78_step(
            backend=backend,
            epoch=left,
            endpoint_epoch=right,
            step=step,
            positions=positions,
            velocities=velocities,
            position_carry=position_carry,
            velocity_carry=velocity_carry,
            derivative=derivative,
        )
        if (
            stage_index != RKF78_STAGE_COUNT
            or len(current_stage_rows) != RKF78_STAGE_COUNT
        ):
            raise DeformableSolarCoupledLunarError(
                "RKF78 stage roster changed"
            )
        positions = np.ascontiguousarray(
            trial.accepted_positions, dtype=np.float64
        )
        norm = float(np.linalg.norm(positions[9:13]))
        maxima["preprojection"] = max(
            maxima["preprojection"], abs(norm - 1.0)
        )
        positions[9:13] = _v1._unit_quaternion(
            np.ascontiguousarray(positions[9:13])
        )
        velocities = np.ascontiguousarray(
            trial.accepted_velocities, dtype=np.float64
        )
        position_carry = np.ascontiguousarray(
            trial.accepted_position_carry, dtype=np.float64
        )
        position_carry[9:13] = 0.0
        velocity_carry = np.ascontiguousarray(
            trial.accepted_velocity_carry, dtype=np.float64
        )
        stage_history[step_index] = tuple(current_stage_rows)
        expired = step_index - divisor
        if expired in stage_history:
            del stage_history[expired]
        maximum_history_steps = max(maximum_history_steps, len(stage_history))
        completed = step_index + 1
        epoch = initial_epoch + right
        xyz = positions[:9].reshape(3, 3)
        uvw = velocities[:9].reshape(3, 3)
        momentum = np.sum(gms[:, None] * uvw, axis=0)
        barycenter = np.sum(gms[:, None] * xyz, axis=0) / total_gm
        expected_barycenter = initial_barycenter + initial_momentum / total_gm * right
        maxima["momentum"] = max(
            maxima["momentum"],
            float(np.linalg.norm(momentum - initial_momentum)) / momentum_scale,
        )
        maxima["barycenter"] = max(
            maxima["barycenter"],
            float(np.linalg.norm(barycenter - expected_barycenter))
            / position_scale,
        )
        orthogonality, determinant = _v1._rotation_metrics(positions[9:13])
        maxima["orthogonality"] = max(
            maxima["orthogonality"], orthogonality
        )
        maxima["determinant"] = max(maxima["determinant"], determinant)
        ledger.update(struct.pack(">qd", completed, epoch))
        ledger.update(positions.tobytes(order="C"))
        ledger.update(velocities.tobytes(order="C"))
        if completed in checkpoint_set:
            checkpoints.append(_v2._state_from_packed(epoch, positions, velocities))
    if len(checkpoints) != len(integration_spec.checkpoint_step_indices):
        raise DeformableSolarCoupledLunarError(
            "checkpoint roster was not completed exactly"
        )
    return DeformableSolarCoupledLunarResult(
        parameters=parameters,
        integration_spec=integration_spec,
        checkpoints=tuple(checkpoints),
        step_seconds=step,
        accepted_steps=final_step_index,
        rkf78_stage_evaluations=final_step_index * RKF78_STAGE_COUNT,
        prehistory_provider_queries=prehistory_queries,
        post_start_prehistory_provider_queries=post_start_queries,
        maximum_stage_history_steps=maximum_history_steps,
        stage_ledger_sha256=ledger.hexdigest(),
        maximum_quaternion_preprojection_norm_error=maxima["preprojection"],
        maximum_rotation_orthogonality_error=maxima["orthogonality"],
        maximum_rotation_determinant_error=maxima["determinant"],
        maximum_linear_momentum_relative_drift=maxima["momentum"],
        maximum_barycenter_relative_drift=maxima["barycenter"],
        maximum_pair_reaction_relative=maxima["reaction"],
        maximum_lunar_figure_torque_balance_relative=maxima["torque_balance"],
        maximum_rotational_balance_relative=maxima["rotation_balance"],
        maximum_positive_viscous_power_over_mr2=maxima[
            "positive_viscous_power"
        ],
        minimum_mantle_inertia_eigenvalue_over_mr2=minimum_inertia,
        maximum_earth_lunar_figure_acceleration_km_s2=maxima["earth_figure"],
        maximum_solar_lunar_figure_acceleration_km_s2=maxima["solar_figure"],
        maximum_earth_j2_acceleration_km_s2=maxima["earth_j2"],
    )


__all__ = [
    "DEFORMABLE_SOLAR_COUPLED_LUNAR_MODEL_ID",
    "DelayedLunarHistorySample",
    "DeformableSolarCoupledLunarError",
    "DeformableSolarCoupledLunarIntegrationSpec",
    "DeformableSolarCoupledLunarParameters",
    "DeformableSolarCoupledLunarResult",
    "DeformableSolarCoupledLunarStepLimitError",
    "MODEL_SCOPE",
    "RKF78_STAGE_ABSCISSAE",
    "SCIENTIFIC_CLAIM_STATE",
    "integrate_deformable_solar_coupled_lunar_trajectory",
]
