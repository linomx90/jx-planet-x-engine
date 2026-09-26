"""Declarative capability catalog for the public experimental JX engine.

Catalog membership is not execution authority.  Only rows marked IMPLEMENTED
may be represented by a typed force configuration; every DECLARED row accepts
and validates its known parameter roster, then fails closed if execution is
requested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .contracts import (
    CapabilityUnavailableError,
    ContractError,
    ParameterMetadata,
)
from .coupled_lunar_contracts import COUPLED_LUNAR_RKF78_METHOD_ID
from .encounter_contracts import ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID
from .hybrid_contracts import HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID
from .lunar_ephemeris_contracts import LUNAR_EPHEMERIS_V1_METHOD_ID
from .trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID
from .symplectic_contracts import FIXED_STEP_KDK_METHOD_ID
from .wisdom_holman_contracts import FIXED_STEP_WISDOM_HOLMAN_METHOD_ID


class CodeStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    DECLARED = "DECLARED"


class Maturity(str, Enum):
    UNQUALIFIED = "UNQUALIFIED"


class AccelerationSemantics(str, Enum):
    BASE = "BASE_ACCELERATION"
    CORRECTION = "ACCELERATION_CORRECTION"
    SERVICE = "STATE_SERVICE"
    EVENT = "EVENT_OR_STATE_TRANSITION"


@dataclass(frozen=True)
class ParameterSpec:
    parameter_id: str
    value_kind: str
    unit_dimension: str
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        for label in ("parameter_id", "value_kind", "unit_dimension", "description"):
            value = getattr(self, label)
            if type(value) is not str or (label != "description" and (not value or value.strip() != value)):
                raise ContractError(f"{label} must be a trimmed string")
        if type(self.required) is not bool:
            raise ContractError("required must be bool")


@dataclass(frozen=True)
class CapabilitySpec:
    model_id: str
    family: str
    code_status: CodeStatus
    maturity: Maturity
    semantics: AccelerationSemantics
    velocity_dependent: bool
    global_snapshot_required: bool
    compatible_integrators: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    dependencies: tuple[str, ...] = ()
    mutually_exclusive_with: tuple[str, ...] = ()
    restrictions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id or self.model_id.strip() != self.model_id:
            raise ContractError("model_id must be a nonempty, trimmed string")
        if type(self.family) is not str or not self.family:
            raise ContractError("family must be nonempty")
        if type(self.code_status) is not CodeStatus or type(self.maturity) is not Maturity:
            raise ContractError("capability status enums are required")
        if type(self.semantics) is not AccelerationSemantics:
            raise ContractError("acceleration semantics enum is required")
        if type(self.velocity_dependent) is not bool or type(self.global_snapshot_required) is not bool:
            raise ContractError("dependency flags must be bool")
        for label in ("compatible_integrators", "parameters", "dependencies", "mutually_exclusive_with", "restrictions"):
            if type(getattr(self, label)) is not tuple:
                raise ContractError(f"{label} must be an immutable tuple")
        names = tuple(parameter.parameter_id for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ContractError(f"duplicate parameter in {self.model_id}")

    @property
    def execution_available(self) -> bool:
        return self.code_status is CodeStatus.IMPLEMENTED


@dataclass(frozen=True, eq=False)
class ParameterBinding:
    """One immutable descriptor around a possibly backend-native value."""

    parameter_id: str
    value: object
    metadata: ParameterMetadata

    def __post_init__(self) -> None:
        if type(self.parameter_id) is not str or not self.parameter_id or self.parameter_id.strip() != self.parameter_id:
            raise ContractError("parameter_id must be a nonempty, trimmed string")
        if self.value is None:
            raise ContractError("a parameter binding cannot have a null value")
        if type(self.metadata) is not ParameterMetadata:
            raise ContractError("metadata must be ParameterMetadata")
        if self.metadata.parameter_id != self.parameter_id:
            raise ContractError("binding and metadata parameter identifiers differ")


@dataclass(frozen=True, eq=False)
class DeclaredModelConfig:
    """Validated configuration for a known but unavailable model."""

    config_id: str
    model_id: str
    epoch: float
    unit_system_id: str
    parameters: tuple[ParameterBinding, ...]

    def __post_init__(self) -> None:
        if type(self.config_id) is not str or not self.config_id or self.config_id.strip() != self.config_id:
            raise ContractError("config_id must be a nonempty, trimmed string")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, (int, float)) or not math.isfinite(float(self.epoch)):
            raise ContractError("epoch must be a finite scalar")
        if type(self.unit_system_id) is not str or not self.unit_system_id or self.unit_system_id.strip() != self.unit_system_id:
            raise ContractError("unit_system_id must be a nonempty, trimmed string")
        capability = get_capability(self.model_id)
        if capability.code_status is not CodeStatus.DECLARED:
            raise ContractError("implemented models require their typed configuration class")
        if type(self.parameters) is not tuple:
            raise ContractError("parameters must be an immutable tuple")
        bindings: dict[str, ParameterBinding] = {}
        for binding in self.parameters:
            if type(binding) is not ParameterBinding:
                raise ContractError("parameters entries must be ParameterBinding")
            if binding.parameter_id in bindings:
                raise ContractError(f"duplicate parameter {binding.parameter_id!r}")
            bindings[binding.parameter_id] = binding
        roster = {parameter.parameter_id: parameter for parameter in capability.parameters}
        unknown = set(bindings) - set(roster)
        missing = {name for name, spec in roster.items() if spec.required} - set(bindings)
        if unknown:
            raise ContractError(f"unknown parameters for {self.model_id}: {sorted(unknown)!r}")
        if missing:
            raise ContractError(f"missing parameters for {self.model_id}: {sorted(missing)!r}")
        for name, binding in bindings.items():
            spec = roster[name]
            _validate_value(spec.value_kind, binding.value, name)
            if binding.metadata.units != spec.unit_dimension:
                raise ContractError(
                    f"{name} units must equal catalog binding {spec.unit_dimension!r}"
                )
            if (
                binding.metadata.validity_start is not None
                and float(self.epoch) < float(binding.metadata.validity_start)
            ) or (
                binding.metadata.validity_end is not None
                and float(self.epoch) > float(binding.metadata.validity_end)
            ):
                raise ContractError(f"{name} is outside its declared validity interval")

    @property
    def code_status(self) -> CodeStatus:
        return CodeStatus.DECLARED

    @property
    def maturity(self) -> Maturity:
        return Maturity.UNQUALIFIED

    def require_executable(self) -> None:
        raise CapabilityUnavailableError(
            f"{self.model_id} is declared but has no executable JX implementation"
        )


def _validate_value(kind: str, value: object, label: str) -> None:
    if kind in {"scalar", "positive_scalar", "nonnegative_scalar"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError(f"{label} must be a finite scalar")
        if kind == "positive_scalar" and float(value) <= 0.0:
            raise ContractError(f"{label} must be positive")
        if kind == "nonnegative_scalar" and float(value) < 0.0:
            raise ContractError(f"{label} cannot be negative")
    elif kind == "integer":
        if type(value) is not int:
            raise ContractError(f"{label} must be an integer")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ContractError(f"{label} must be bool")
    elif kind in {"string", "identifier"}:
        if type(value) is not str or not value or value.strip() != value:
            raise ContractError(f"{label} must be a nonempty, trimmed string")
    elif kind == "identifier_tuple":
        if type(value) is not tuple or not value or any(type(item) is not str or not item for item in value):
            raise ContractError(f"{label} must be a nonempty immutable identifier tuple")
    elif kind == "vector3":
        if type(value) is not tuple or len(value) != 3:
            raise ContractError(f"{label} must be an immutable 3-vector")
        for component in value:
            _validate_value("scalar", component, label)
    elif kind in {"array", "record", "callable"}:
        # Opaque values are intentionally left on their owning backend.
        if value is None:
            raise ContractError(f"{label} cannot be null")
    else:
        raise ContractError(f"unknown catalog value kind {kind!r}")


def _p(name: str, kind: str, dimension: str, required: bool = True) -> ParameterSpec:
    return ParameterSpec(name, kind, dimension, required)


GENERAL = ("GENERAL_FIRST_ORDER",)
POSITION = ("GENERAL_FIRST_ORDER", "POSITION_FORCE_KICK_DRIFT")
IMPLICIT = ("GENERAL_FIRST_ORDER", "IMPLICIT_VELOCITY_DEPENDENT")
SERVICE = ("EXTERNAL_STATE_PROVIDER",)
EVENT = ("EVENT_CAPABLE_GENERAL",)


_ROWS = (
    CapabilitySpec("force.newtonian.point_mass", "GRAVITY", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.BASE, False, True, POSITION, (_p("source_ids", "identifier_tuple", "1"), _p("target_ids", "identifier_tuple", "1"), _p("unit_system_id", "identifier", "1"), _p("state.gravitational_parameters", "array", "STATE_LENGTH_UNIT^3/STATE_TIME_UNIT^2")), restrictions=("direct unsoftened summation", "collision singularities fail closed", "softening unavailable")),
    CapabilitySpec("relativity.solar_schwarzschild_test_particle_1pn", "RELATIVITY", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("central_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("unit_system_id", "identifier", "1"), _p("speed_of_light", "positive_scalar", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"), _p("maximum_compactness", "positive_scalar", "1"), _p("maximum_speed_fraction_squared", "positive_scalar", "1")), dependencies=("force.newtonian.point_mass",), mutually_exclusive_with=("force.relativity.eih_1pn_gr", "force.relativity.restricted_ppn_beta_gamma"), restrictions=("CENTRAL_BODY_INERTIAL frame", "origin equals static central source", "massless test particles", "correction only")),
    CapabilitySpec("force.nongrav.srp_cannonball", "NONGRAVITATIONAL", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, GENERAL, (_p("radiation_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("unit_system_id", "identifier", "1"), _p("reference_pressure", "positive_scalar", "STATE_MASS_UNIT/(STATE_LENGTH_UNIT*STATE_TIME_UNIT^2)"), _p("reference_distance", "positive_scalar", "STATE_LENGTH_UNIT"), _p("area_to_mass", "array", "STATE_LENGTH_UNIT^2/STATE_MASS_UNIT"), _p("radiation_pressure_coefficient", "array", "1"), _p("coefficient_convention", "string", "1"), _p("attitude_model", "string", "1"), _p("shadow_model", "string", "1")), dependencies=("force.newtonian.point_mass",), restrictions=("isotropic cannonball", "unshadowed")),
    CapabilitySpec("force.newtonian.softened_point_mass", "GRAVITY", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.BASE, False, True, POSITION, (_p("source_ids", "identifier_tuple", "1"), _p("target_ids", "identifier_tuple", "1"), _p("gravitational_parameters", "array", "L^3/T^2"), _p("softening_kernel", "string", "1"), _p("softening_lengths", "array", "L"))),
    CapabilitySpec("force.newtonian.barnes_hut_tree", "GRAVITY", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.BASE, False, True, POSITION, (_p("body_ids", "identifier_tuple", "1"), _p("gravitational_parameters", "array", "L^3/T^2"), _p("opening_angle", "positive_scalar", "1"), _p("opening_criterion", "string", "1"), _p("leaf_capacity", "integer", "1"), _p("multipole_order", "integer", "1"), _p("singularity_policy", "string", "1"))),
    CapabilitySpec("force.harmonics.solar_j2_j4", "GRAVITY_HARMONICS", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, POSITION, (_p("central_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("reference_radius", "positive_scalar", "L"), _p("j2", "scalar", "1"), _p("j4", "scalar", "1"), _p("pole_vector", "vector3", "1"), _p("orientation_frame", "string", "1"))),
    CapabilitySpec("solar-system.force.earth-zonal-j2-j5-axisymmetric-pair", "GRAVITY_HARMONICS", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, GENERAL, (_p("source_id", "identifier", "1"), _p("target_id", "identifier", "1"), _p("unit_system_id", "identifier", "1"), _p("reference_radius", "positive_scalar", "STATE_LENGTH_UNIT"), _p("zonal_coefficients", "record", "1"), _p("earth_pole_model", "record", "1")), dependencies=("force.newtonian.point_mass",), restrictions=("NumPy CPU only", "metre-second J2000 state", "one Earth-target pair", "unnormalized J2 through J5", "correction only")),
    CapabilitySpec("solar-system.force.lunar-static-degree2-principal-axis-pair", "GRAVITY_HARMONICS", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, GENERAL, (_p("source_id", "identifier", "1"), _p("target_id", "identifier", "1"), _p("unit_system_id", "identifier", "1"), _p("reference_radius", "positive_scalar", "STATE_LENGTH_UNIT"), _p("degree2_coefficients", "record", "1"), _p("lunar_orientation_model", "record", "1")), dependencies=("force.newtonian.point_mass",), restrictions=("NumPy CPU only", "metre-second J2000 state", "one Moon-target pair", "static unnormalized degree two", "correction only")),
    CapabilitySpec("solar-system.force.lunar-static-degree3-principal-axis-pair", "GRAVITY_HARMONICS", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, GENERAL, (_p("source_id", "identifier", "1"), _p("target_id", "identifier", "1"), _p("unit_system_id", "identifier", "1"), _p("reference_radius", "positive_scalar", "STATE_LENGTH_UNIT"), _p("degree3_coefficients", "record", "1"), _p("lunar_orientation_model", "record", "1")), dependencies=("force.newtonian.point_mass",), restrictions=("NumPy CPU only", "metre-second J2000 state", "one Moon-target pair", "static unnormalized degree three", "correction only")),
    CapabilitySpec("force.harmonics.planetary", "GRAVITY_HARMONICS", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, False, False, POSITION, (_p("source_ids", "identifier_tuple", "1"), _p("target_ids", "identifier_tuple", "1"), _p("reference_radii", "array", "L"), _p("coefficient_sets", "record", "1"), _p("orientation_model", "record", "1"), _p("maximum_degree", "integer", "1"), _p("maximum_order", "integer", "1"))),
    CapabilitySpec("force.relativity.eih_1pn_gr", "RELATIVITY", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, True, IMPLICIT, (_p("body_ids", "identifier_tuple", "1"), _p("unit_system_id", "identifier", "1"), _p("speed_of_light", "positive_scalar", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"), _p("maximum_compactness", "positive_scalar", "1"), _p("maximum_speed_fraction_squared", "positive_scalar", "1")), dependencies=("force.newtonian.point_mass",), mutually_exclusive_with=("relativity.solar_schwarzschild_test_particle_1pn", "force.relativity.restricted_ppn_beta_gamma"), restrictions=("TDB barycentric inertial state", "all state bodies are massive mutual sources and targets", "correction only", "beta=gamma=1")),
    CapabilitySpec("force.relativity.restricted_ppn_beta_gamma", "RELATIVITY", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("central_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("speed_of_light", "positive_scalar", "L/T"), _p("beta", "scalar", "1"), _p("gamma", "scalar", "1"), _p("maximum_compactness", "positive_scalar", "1"), _p("maximum_speed_fraction_squared", "positive_scalar", "1")), dependencies=("force.newtonian.point_mass",), mutually_exclusive_with=("relativity.solar_schwarzschild_test_particle_1pn", "force.relativity.eih_1pn_gr")),
    CapabilitySpec("force.relativity.solar_lense_thirring", "RELATIVITY", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("central_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("speed_of_light", "positive_scalar", "L/T"), _p("spin_angular_momentum", "vector3", "M*L^2/T"), _p("orientation_frame", "string", "1")), dependencies=("force.newtonian.point_mass",)),
    CapabilitySpec("force.tides.constant_time_lag", "TIDES", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, True, IMPLICIT, (_p("interacting_pairs", "record", "1"), _p("love_numbers", "array", "1"), _p("time_lags", "array", "T"), _p("radii", "array", "L"), _p("spin_states", "array", "1/T"), _p("dissipation_convention", "string", "1"))),
    CapabilitySpec("force.nongrav.srp_pr_burns_1979", "NONGRAVITATIONAL", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("radiation_source_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("speed_of_light", "positive_scalar", "L/T"), _p("reference_pressure", "positive_scalar", "M/(L*T^2)"), _p("reference_distance", "positive_scalar", "L"), _p("area_to_mass", "array", "L^2/M"), _p("radiation_pressure_coefficient", "array", "1"), _p("attitude_model", "string", "1"), _p("shadow_model", "string", "1"))),
    CapabilitySpec("force.nongrav.yarkovsky.empirical_a2", "NONGRAVITATIONAL", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("target_ids", "identifier_tuple", "1"), _p("a2", "array", "L/T^2"), _p("reference_distance", "positive_scalar", "L"), _p("radial_exponent", "scalar", "1"), _p("transverse_convention", "string", "1"))),
    CapabilitySpec("force.nongrav.yarkovsky.linear_sphere", "NONGRAVITATIONAL", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("target_ids", "identifier_tuple", "1"), _p("radii", "array", "L"), _p("densities", "array", "M/L^3"), _p("thermal_conductivities", "array", "M*L/(T^3*K)"), _p("heat_capacities", "array", "L^2/(T^2*K)"), _p("emissivities", "array", "1"), _p("albedos", "array", "1"), _p("spin_states", "array", "1/T"), _p("pole_vectors", "array", "1"), _p("radiation_source", "identifier", "1"))),
    CapabilitySpec("force.nongrav.yarkovsky.facet_thermophysical", "NONGRAVITATIONAL", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, GENERAL, (_p("target_ids", "identifier_tuple", "1"), _p("shape_meshes", "record", "1"), _p("facet_materials", "record", "1"), _p("spin_states", "array", "1/T"), _p("attitude_model", "record", "1"), _p("thermal_solver", "record", "1"), _p("shadow_model", "record", "1"))),
    CapabilitySpec("force.nongrav.comet.marsden_esm", "OUTGASSING", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("target_ids", "identifier_tuple", "1"), _p("a1_a2_a3", "array", "L/T^2"), _p("g_law_parameters", "record", "1"), _p("time_shifts", "array", "T"), _p("orbital_frame_convention", "string", "1"))),
    CapabilitySpec("force.nongrav.comet.rotating_jet", "OUTGASSING", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, GENERAL, (_p("target_ids", "identifier_tuple", "1"), _p("jet_geometry", "record", "1"), _p("mass_flow_model", "record", "M/T"), _p("exhaust_velocity", "array", "L/T"), _p("spin_states", "array", "1/T"), _p("pole_vectors", "array", "1"), _p("attitude_model", "record", "1"))),
    CapabilitySpec("force.drag.gas", "DRAG", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("target_ids", "identifier_tuple", "1"), _p("gas_density_model", "record", "M/L^3"), _p("gas_velocity_model", "record", "L/T"), _p("drag_coefficients", "array", "1"), _p("area_to_mass", "array", "L^2/M"), _p("flow_regime_model", "string", "1"))),
    CapabilitySpec("force.drag.atmospheric", "DRAG", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, IMPLICIT, (_p("central_body_id", "identifier", "1"), _p("target_ids", "identifier_tuple", "1"), _p("atmosphere_model", "record", "M/L^3"), _p("atmosphere_rotation", "vector3", "1/T"), _p("drag_coefficients", "array", "1"), _p("area_to_mass", "array", "L^2/M"), _p("space_weather_inputs", "record", "1"))),
    CapabilitySpec("force.thrust.prescribed", "THRUST", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.CORRECTION, True, False, GENERAL, (_p("target_ids", "identifier_tuple", "1"), _p("burn_windows", "record", "T"), _p("force_or_acceleration_profile", "callable", "1"), _p("direction_law", "callable", "1"), _p("mass_flow_profile", "callable", "M/T"), _p("attitude_model", "record", "1"))),
    CapabilitySpec("ephemeris.external.interpolated", "EPHEMERIS", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, SERVICE, (_p("source_ids", "identifier_tuple", "1"), _p("kernel_or_table", "record", "1"), _p("frame", "string", "1"), _p("origin", "string", "1"), _p("time_scale", "string", "1"), _p("validity_interval", "record", "T"), _p("interpolation_method", "string", "1"), _p("interpolation_tolerance", "positive_scalar", "L"))),
    CapabilitySpec("force.collision.hard_sphere", "COLLISION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.EVENT, True, True, EVENT, (_p("body_ids", "identifier_tuple", "1"), _p("radii", "array", "L"), _p("collision_response", "string", "1"), _p("restitution_coefficients", "array", "1"), _p("fragmentation_model", "record", "1", False), _p("event_tolerance", "positive_scalar", "T"))),
    CapabilitySpec("force.encounter.hybrid_switching", "CLOSE_ENCOUNTER", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.EVENT, True, True, EVENT, (_p("body_ids", "identifier_tuple", "1"), _p("switch_distance", "positive_scalar", "L"), _p("switch_hysteresis", "positive_scalar", "L"), _p("far_integrator", "string", "1"), _p("near_integrator", "string", "1"), _p("event_tolerance", "positive_scalar", "T"))),
    CapabilitySpec("force.regularization.algorithmic", "REGULARIZATION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, EVENT, (_p("body_ids", "identifier_tuple", "1"), _p("regularization_method", "string", "1"), _p("activation_distance", "positive_scalar", "L"), _p("termination_distance", "positive_scalar", "L"), _p("error_tolerance", "positive_scalar", "1"))),
    CapabilitySpec("backend.numpy.cpu", "BACKEND", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("device", "string", "1"), _p("dtype", "string", "1"), _p("tile_size", "integer", "1")), restrictions=("float64 only", "same-runtime/device determinism only", "no fallback")),
    CapabilitySpec("backend.cupy.cuda", "BACKEND", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("device", "string", "1"), _p("dtype", "string", "1"), _p("tile_size", "integer", "1")), restrictions=("optional CuPy runtime", "code path present but not locally GPU-qualified", "no fallback", "no implicit transfer")),
    CapabilitySpec("backend.hip", "BACKEND", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("device", "string", "1"), _p("runtime", "string", "1"), _p("dtype", "string", "1"), _p("tile_size", "integer", "1"))),
    CapabilitySpec("backend.sycl", "BACKEND", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("device", "string", "1"), _p("runtime", "string", "1"), _p("dtype", "string", "1"), _p("tile_size", "integer", "1"))),
    CapabilitySpec("backend.metal", "BACKEND", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("device", "string", "1"), _p("runtime", "string", "1"), _p("dtype", "string", "1"), _p("tile_size", "integer", "1"))),
    CapabilitySpec("precision.float64", "PRECISION", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("dtype", "string", "1"),), restrictions=("only executable precision in the public engine",)),
    CapabilitySpec("precision.float32", "PRECISION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("dtype", "string", "1"), _p("error_budget", "record", "1"))),
    CapabilitySpec("precision.mixed", "PRECISION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("storage_dtype", "string", "1"), _p("compute_dtype", "string", "1"), _p("accumulation_dtype", "string", "1"), _p("error_budget", "record", "1"))),
    CapabilitySpec("precision.decimal_reference", "PRECISION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("decimal_precision", "integer", "1"), _p("rounding", "string", "1"), _p("trap_policy", "record", "1")), restrictions=("audit/reference path only",)),
    CapabilitySpec("determinism.same_runtime_device", "DETERMINISM", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("runtime_fingerprint", "record", "1"), _p("device_fingerprint", "record", "1"), _p("tile_size", "integer", "1"), _p("fast_math", "boolean", "1")), restrictions=("does not claim cross-device bitwise identity",)),
    CapabilitySpec("determinism.cross_device_bitwise", "DETERMINISM", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, False, SERVICE, (_p("backend_matrix", "record", "1"), _p("reduction_policy", "record", "1"), _p("compiler_policy", "record", "1"))),
    CapabilitySpec(ADAPTIVE_RKF78_METHOD_ID, "INTEGRATOR", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, GENERAL, (_p("checkpoint_epochs", "record", "STATE_TIME_UNIT"), _p("initial_step", "positive_scalar", "STATE_TIME_UNIT"), _p("minimum_step", "positive_scalar", "STATE_TIME_UNIT"), _p("maximum_step", "positive_scalar", "STATE_TIME_UNIT"), _p("position_atol", "array", "STATE_LENGTH_UNIT"), _p("position_rtol", "positive_scalar", "1"), _p("velocity_atol", "array", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"), _p("velocity_rtol", "positive_scalar", "1"), _p("maximum_steps", "integer", "1"), _p("maximum_rejections", "integer", "1"), _p("safety_factor", "positive_scalar", "1"), _p("minimum_scale_factor", "positive_scalar", "1"), _p("maximum_scale_factor", "positive_scalar", "1")), restrictions=("explicit nonstiff Fehlberg RK7(8), 13 stages", "accept hatted order-8 solution; embedded ordinary order-7 defect", "supports velocity-dependent force plans", "normalized maximum per-component error", "exact checkpoint endpoints by step clipping; no interpolation", "float64 only")),
    CapabilitySpec(
        COUPLED_LUNAR_RKF78_METHOD_ID,
        "INTEGRATOR",
        CodeStatus.IMPLEMENTED,
        Maturity.UNQUALIFIED,
        AccelerationSemantics.SERVICE,
        True,
        True,
        GENERAL,
        (
            _p("initial_state", "record", "COUPLED_STATE_UNITS"),
            _p("parameters", "record", "1"),
            _p("integration_spec", "record", "1"),
            _p("prehistory_provider", "callable", "COUPLED_STATE_UNITS"),
        ),
        dependencies=("backend.numpy.cpu", "precision.float64"),
        restrictions=(
            "simultaneous Sun-Earth-Moon translation, lunar mantle attitude and angular velocity, delayed mantle deformation, and fluid-core angular velocity",
            "fixed-step 13-stage RKF78 delay lattice with caller-supplied exact pre-start history",
            "TDB barycentric-inertial J2000 state in kilometres and seconds with exact SUN, EARTH, MOON body order",
            "retained native coupled physics bundle; not yet dispatched through force ABI v1",
            "no geodetic transport, dense output, event location, collision response, continuation archive, CUDA, production ephemeris, or general superiority claim",
            "all results remain SCREENING_ONLY unqualified model output",
        ),
    ),
    CapabilitySpec(
        LUNAR_EPHEMERIS_V1_METHOD_ID,
        "INTEGRATOR",
        CodeStatus.IMPLEMENTED,
        Maturity.UNQUALIFIED,
        AccelerationSemantics.SERVICE,
        True,
        True,
        GENERAL,
        (
            _p("initial_state", "record", "COUPLED_STATE_UNITS"),
            _p("parameters", "record", "1"),
            _p("integration_spec", "record", "1"),
        ),
        dependencies=(
            "backend.numpy.cpu",
            "precision.float64",
            "force.relativity.eih_1pn_gr",
        ),
        restrictions=(
            "fixed resolved-eleven body roster in exact order",
            "simultaneous barycentric translation, lunar mantle attitude and rate, and fluid-core rate",
            "mutual Newtonian plus EIH 1PN, reacting Sun/Earth lunar quadrupole, fixed-axis reacting Earth J2, and mantle-core coupling",
            "fixed-step 13-stage RKF78 with exact checkpoint clipping",
            "kilometre-second TDB barycentric-inertial J2000 state",
            "NumPy CPU only; no continuation archive, CUDA, dense output, event location, or collision response",
            "omits time-variable deformation, delayed tides, Earth J3-J5, lunar degree three and higher, minor bodies, observation reduction, and parameter fitting",
            "all results remain SCREENING_ONLY and are not a production ephemeris",
        ),
    ),
    CapabilitySpec(
        ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
        "INTEGRATOR",
        CodeStatus.IMPLEMENTED,
        Maturity.UNQUALIFIED,
        AccelerationSemantics.SERVICE,
        False,
        True,
        GENERAL,
        (
            _p("body_order", "identifier_tuple", "1"),
            _p("initial_epoch", "scalar", "STATE_TIME_UNIT"),
            _p("endpoint_epoch", "scalar", "STATE_TIME_UNIT"),
            _p("duration", "scalar", "STATE_TIME_UNIT"),
            _p("initial_step_magnitude", "positive_scalar", "STATE_TIME_UNIT"),
            _p("minimum_step_magnitude", "positive_scalar", "STATE_TIME_UNIT"),
            _p("maximum_step_magnitude", "positive_scalar", "STATE_TIME_UNIT"),
            _p("pair_certification_floors", "array", "STATE_LENGTH_UNIT"),
            _p("pair_position_atols", "array", "STATE_LENGTH_UNIT"),
            _p("pair_position_rtol", "positive_scalar", "1"),
            _p("pair_velocity_atols", "array", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"),
            _p("pair_velocity_rtol", "positive_scalar", "1"),
            _p("gm_centroid_position_atol", "positive_scalar", "STATE_LENGTH_UNIT"),
            _p("gm_centroid_position_rtol", "positive_scalar", "1"),
            _p("gm_centroid_velocity_atol", "positive_scalar", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"),
            _p("gm_centroid_velocity_rtol", "positive_scalar", "1"),
            _p("maximum_substep_proposals", "integer", "1"),
            _p("maximum_accepted_substeps", "integer", "1"),
            _p("maximum_rejected_substeps", "integer", "1"),
            _p("maximum_consecutive_rejections", "integer", "1"),
            _p("maximum_force_evaluations", "integer", "1"),
            _p("safety_factor", "positive_scalar", "1"),
            _p("minimum_scale_factor", "positive_scalar", "1"),
            _p("maximum_scale_factor", "positive_scalar", "1"),
            _p("exact_rational_resources", "record", "1"),
        ),
        dependencies=(
            "force.newtonian.point_mass",
            "backend.numpy.cpu",
            "precision.float64",
        ),
        restrictions=(
            "standalone exact-duration full-Cartesian adaptive Fehlberg RK7(8) local encounter IVP segment",
            "NumPy CPU binary64 only; barycentric inertial state; all-active positive-GM mutual unsoftened Newtonian force only",
            "caller-supplied positive canonical pair floors and an exact all-pair simultaneous first-crossing certificate at every numerical node",
            "the certificate proves noncollision only for the exact local IVP issuing from each accepted numerical node on that proposed substep",
            "endpoint_epoch is an independent externally supplied provenance label; signed duration and exact local offsets alone advance state",
            "pair-relative and GM-centroid control uses the returned RKF78 defect arrays; accepted updates use componentwise Kahan accumulation",
            "initialization and every proposal use capped deterministic abstract exact-work, GCD-iteration, transcript, and witness ledgers",
            "one mandatory deterministic semantic replay has separate accounting; public totals include primary plus replay",
            "an uncertified or guarded proposal is a retry or fail-closed condition and is never a collision or event determination",
            "no dense output, event detection or location, collision response, regularization, global trajectory clearance, hybrid switching, symplecticity, or exact reversibility claim",
            "all results remain unqualified MODEL_OUTPUT",
        ),
    ),
    CapabilitySpec(FIXED_STEP_KDK_METHOD_ID, "INTEGRATOR", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, True, POSITION, (_p("checkpoint_step_indices", "record", "1"), _p("fixed_step", "scalar", "STATE_TIME_UNIT"), _p("maximum_steps", "integer", "1"), _p("minimum_swept_pair_separation", "positive_scalar", "STATE_LENGTH_UNIT"), _p("maximum_pair_frequency_step", "positive_scalar", "1")), restrictions=("second-order kick-drift-kick map; symplectic and time-reversible only in exact arithmetic", "NumPy CPU only", "one fully mutual all-body Newtonian point-mass force plan", "positive GM and all bodies active; passive/massless tracers unavailable", "barycentric inertial Cartesian state with continuous coordinate time", "constant signed binary64 map step", "integer step-index checkpoints; no clipping or interpolation", "mandatory swept-drift encounter/contact and pair-frequency resolution guards")),
    CapabilitySpec(FIXED_STEP_WISDOM_HOLMAN_METHOD_ID, "INTEGRATOR", CodeStatus.IMPLEMENTED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, True, POSITION, (_p("checkpoint_step_indices", "record", "1"), _p("fixed_step", "scalar", "STATE_TIME_UNIT"), _p("maximum_steps", "integer", "1"), _p("jacobi_body_order", "identifier_tuple", "1"), _p("minimum_encounter_pair_separation", "positive_scalar", "STATE_LENGTH_UNIT"), _p("minimum_jacobi_periapse", "positive_scalar", "STATE_LENGTH_UNIT"), _p("maximum_initial_barycenter_position_norm", "positive_scalar", "STATE_LENGTH_UNIT"), _p("maximum_initial_barycenter_velocity_norm", "positive_scalar", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"), _p("kepler_solver", "record", "1", False)), restrictions=("second-order ordered-Jacobi interaction-kick/Kepler-drift/interaction-kick map; symplectic and time-reversible only for the formal exact subflows", "NumPy CPU binary64 only", "one fully mutual all-body Newtonian point-mass force plan", "central body first; positive GM and all bodies active; passive/massless tracers unavailable", "barycentric inertial Cartesian state with continuous coordinate time and near-rest barycenter", "bound elliptic strictly hierarchical low-mass-secondary domain only", "constant signed binary64 map step", "integer step-index checkpoints; no clipping, interpolation, subdivision, or fallback", "mandatory periapse, interaction-force, step-resolution, mutual-Hill, swept-path, contact, and hierarchy guards", "one mandatory deterministic semantic validation replay; all results remain unqualified MODEL_OUTPUT")),
    CapabilitySpec(
        HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
        "INTEGRATOR",
        CodeStatus.IMPLEMENTED,
        Maturity.UNQUALIFIED,
        AccelerationSemantics.SERVICE,
        False,
        True,
        POSITION,
        (
            _p("wisdom_holman_spec", "record", "1"),
            _p("encounter_control", "record", "1"),
        ),
        dependencies=(
            "force.newtonian.point_mass",
            "backend.numpy.cpu",
            "precision.float64",
        ),
        restrictions=(
            "specific whole-system Wisdom-Holman Jacobi / guarded full-Cartesian adaptive RKF78 hybrid on one fixed signed integer outer lattice",
            "NumPy CPU binary64 only; barycentric inertial state; central body first; at most 16 all-active positive-GM bodies under one fully mutual unsoftened Newtonian force plan",
            "each outer step performs one typed complete provisional WH far probe; finite named dynamic guard exits alone select near mode, while static, numerical, resource, custody, and replay failures are fatal and return no partial result",
            "a far pass commits the WH candidate; a near decision discards the entire provisional candidate and redoes the untouched original node over the full outer interval with the private Cartesian encounter execution",
            "the encounter certificate proves noncollision only for the exact local IVP issuing from each accepted numerical node; it is not collision or event detection and makes no global clearance claim",
            "there is no hysteresis, latch, grouping, partial-prefix commit, event location, endpoint clipping, interpolation, or dense output",
            "one mandatory full hybrid semantic replay recomputes every decision, state, private child record, digest, and accounting lane; no public child wrapper or nested public child replay is used",
            "primary, replay, and public-total accounting retain accepted far work, discarded provisional far work, private near work, and coordinate transforms separately",
            "hard ceilings per primary or replay lane are 65536 outer records, 4096 near macrosteps, 524288 retained near accepted substeps, and 917504 retained digest bytes; public-total ceilings are exactly twice those values",
            "an all-far execution preserves the frozen public WH accepted-state, checkpoint, diagnostic, force-ledger, checksum, and accounting projection bit-for-bit, with only the pure pre-force path screen hoisted",
            "not globally symplectic, formally or exactly reversible, regularized, event-capable, collision-detecting, collision-responding, globally order-qualified, superior, or qualified",
            "finite library ceilings do not guarantee denial-of-service resistance, wall time, memory, allocator, or concurrency behavior; external process isolation, timeout, memory, and concurrency limits are required",
            "all results remain unqualified MODEL_OUTPUT",
        ),
    ),
    CapabilitySpec("integrator.symplectic.split", "INTEGRATOR", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, False, True, POSITION, (_p("method", "string", "1"), _p("fixed_step", "positive_scalar", "T"), _p("splitting", "record", "1"), _p("corrector", "record", "1", False), _p("output_schedule", "record", "T")), restrictions=("position-only separable force plans",)),
    CapabilitySpec("integrator.implicit.velocity_dependent", "INTEGRATOR", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, IMPLICIT, (_p("method", "string", "1"), _p("step", "positive_scalar", "T"), _p("nonlinear_tolerance", "positive_scalar", "1"), _p("maximum_iterations", "integer", "1"), _p("failure_policy", "string", "1"))),
    CapabilitySpec("integrator.hybrid.close_encounter", "INTEGRATOR", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, EVENT, (_p("far_integrator", "record", "1"), _p("near_integrator", "record", "1"), _p("switch_policy", "record", "L"), _p("regularization", "record", "1"), _p("event_tolerance", "positive_scalar", "T"))),
    CapabilitySpec("analysis.variational_equations", "SENSITIVITY", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, GENERAL, (_p("parameter_ids", "identifier_tuple", "1"), _p("initial_state_transition", "array", "1"), _p("jacobian_method", "string", "1"), _p("differentiation_step", "positive_scalar", "1", False))),
    CapabilitySpec("orbit_determination.batch_least_squares", "ORBIT_DETERMINATION", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, GENERAL, (_p("estimated_parameters", "identifier_tuple", "1"), _p("prior_state", "record", "STATE_UNITS"), _p("prior_covariance", "array", "STATE_UNITS^2"), _p("measurement_set", "record", "MEASUREMENT_UNITS"), _p("weight_model", "record", "1"), _p("convergence_policy", "record", "1"))),
    CapabilitySpec("measurement.astrometry", "MEASUREMENT", CodeStatus.DECLARED, Maturity.UNQUALIFIED, AccelerationSemantics.SERVICE, True, True, GENERAL, (_p("observer_ephemeris", "record", "STATE_UNITS"), _p("time_scale", "string", "1"), _p("frame", "string", "1"), _p("light_time_model", "record", "1"), _p("aberration_model", "record", "1"), _p("bias_model", "record", "MEASUREMENT_UNITS"), _p("covariance_model", "record", "MEASUREMENT_UNITS^2"))),
)


CAPABILITY_CATALOG: Mapping[str, CapabilitySpec] = MappingProxyType(
    {row.model_id: row for row in _ROWS}
)


def get_capability(model_id: str) -> CapabilitySpec:
    if type(model_id) is not str or not model_id or model_id.strip() != model_id:
        raise ContractError("model_id must be a nonempty, trimmed string")
    try:
        return CAPABILITY_CATALOG[model_id]
    except KeyError as exc:
        raise ContractError(f"unknown JX engine model_id {model_id!r}") from exc


def list_capabilities() -> tuple[CapabilitySpec, ...]:
    """Return capabilities in stable catalog order."""

    return _ROWS


__all__ = [
    "AccelerationSemantics",
    "CAPABILITY_CATALOG",
    "CapabilitySpec",
    "CodeStatus",
    "DeclaredModelConfig",
    "Maturity",
    "ParameterBinding",
    "ParameterSpec",
    "get_capability",
    "list_capabilities",
]
