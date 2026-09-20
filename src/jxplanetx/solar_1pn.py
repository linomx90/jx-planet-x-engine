"""JX V5 Decimal reference kernel for the restricted Solar Schwarzschild 1PN term.

This module deliberately exposes a correction-only primitive.  It is not a
full relativistic N-body model, an Einstein--Infeld--Hoffmann implementation,
or an integrator.  In particular, it must not be inserted into the existing
velocity-independent symplectic kick without a separately validated
velocity-dependent integration scheme.

For relative ``r = target - central`` and ``v = target - central``, the GR
test-particle correction implemented here is

    delta_a = mu/(c^2 |r|^3) *
              ((4 mu/|r| - |v|^2) r + 4 (r dot v) v).

The Newtonian term ``-mu r/|r|^3`` is intentionally absent.  All dimensional
inputs must use the single coherent length/time system declared by the
contract; this module performs no unit or time-scale conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from enum import Enum

from .decimal_math import Vec3


SOLAR_SCHWARZSCHILD_1PN_MODEL_ID = "relativity.solar_schwarzschild_test_particle_1pn"


class RelativityContractError(ValueError):
    """The caller supplied an ambiguous or unsupported model contract."""


class RelativityDomainError(ValueError):
    """The state or physical parameter lies outside the declared 1PN domain."""


class DuplicateForceApplicationError(ValueError):
    """A tagged acceleration contribution would be applied more than once."""


class CoherentUnitSystem(Enum):
    """Supported coherent numerical systems; no conversion is performed."""

    AU_DAY = "position=au;time=day;velocity=au/day;mu=au^3/day^2;c=au/day"
    KILOMETRE_SECOND = "position=km;time=s;velocity=km/s;mu=km^3/s^2;c=km/s"
    METRE_SECOND = "position=m;time=s;velocity=m/s;mu=m^3/s^2;c=m/s"


class InertialFrame(Enum):
    """Coordinate-frame contract supported by this restricted kernel."""

    BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED = (
        "simultaneous BCRS-derived Sun-relative state with ICRS-aligned axes; "
        "restricted static-Sun approximation"
    )


class CoordinateTimeScale(Enum):
    """Scaling shared by state velocities, mu, c, and the time coordinate."""

    TCB_COMPATIBLE = "TCB-compatible"
    TDB_COMPATIBLE = "TDB-compatible"


class AccelerationSemantics(Enum):
    CORRECTION_ONLY = "CORRECTION_ONLY"
    TOTAL_ACCELERATION = "TOTAL_ACCELERATION"


class CentralSourceModel(Enum):
    """The only central-source approximation implemented by this kernel."""

    STATIC_SPHERICAL_SOLAR_MONOPOLE = "STATIC_SPHERICAL_SOLAR_MONOPOLE"


class TargetTreatment(Enum):
    """Explicitly exclude massive targets and backreaction."""

    MASSLESS_NO_BACKREACTION = "MASSLESS_NO_BACKREACTION"


def _require_name(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise RelativityContractError(f"{label} must be a nonempty, trimmed string")
    return value


def _require_decimal(value: object, label: str) -> Decimal:
    # Deliberately reject int/bool/float rather than silently changing the
    # caller's representation or importing binary-float rounding.
    if type(value) is not Decimal:
        raise RelativityContractError(f"{label} must be decimal.Decimal")
    if not value.is_finite():
        raise RelativityDomainError(f"{label} must be finite")
    return value


def _require_vec3(value: object, label: str) -> Vec3:
    if type(value) is not tuple or len(value) != 3:
        raise RelativityContractError(f"{label} must be an immutable 3-tuple")
    checked = tuple(_require_decimal(item, f"{label}[{index}]") for index, item in enumerate(value))
    return checked  # type: ignore[return-value]


@dataclass(frozen=True)
class SolarSchwarzschild1PNContract:
    """Complete contract for the static-central, negligible-target GR term.

    ``mu`` and ``speed_of_light`` must already be scaled for ``time_scale`` and
    ``units``.  ``maximum_compactness`` bounds ``mu/(r*c^2)`` and
    ``maximum_speed_fraction_squared`` bounds ``v^2/c^2``.  These caller-owned
    thresholds make the weak-field/slow-motion applicability decision explicit.
    """

    central_source: str
    gravitational_parameter: Decimal
    speed_of_light: Decimal
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    central_source_model: CentralSourceModel
    target_treatment: TargetTreatment
    output_semantics: AccelerationSemantics
    maximum_compactness: Decimal
    maximum_speed_fraction_squared: Decimal

    def __post_init__(self) -> None:
        central = _require_name(self.central_source, "central_source")
        if central != "SUN":
            raise RelativityContractError("the Solar model requires canonical central_source='SUN'")
        mu = _require_decimal(self.gravitational_parameter, "gravitational_parameter")
        c = _require_decimal(self.speed_of_light, "speed_of_light")
        compactness = _require_decimal(self.maximum_compactness, "maximum_compactness")
        speed_fraction = _require_decimal(
            self.maximum_speed_fraction_squared, "maximum_speed_fraction_squared"
        )
        if mu <= 0:
            raise RelativityDomainError("gravitational_parameter must be positive")
        if c <= 0:
            raise RelativityDomainError("speed_of_light must be positive")
        if type(self.units) is not CoherentUnitSystem:
            raise RelativityContractError("units must be an explicit CoherentUnitSystem")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise RelativityContractError(
                "frame must be the restricted BCRS-derived Sun-relative, ICRS-aligned frame"
            )
        if type(self.time_scale) is not CoordinateTimeScale:
            raise RelativityContractError("time_scale must be explicitly TCB- or TDB-compatible")
        if self.central_source_model is not CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE:
            raise RelativityContractError("this kernel requires a static spherical Solar monopole")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise RelativityContractError("this kernel requires a massless target with no backreaction")
        if self.output_semantics is not AccelerationSemantics.CORRECTION_ONLY:
            raise RelativityContractError("this kernel authorizes CORRECTION_ONLY output, never a total acceleration")
        if not (0 < compactness < 1):
            raise RelativityDomainError("maximum_compactness must lie strictly between zero and one")
        if not (0 < speed_fraction < 1):
            raise RelativityDomainError(
                "maximum_speed_fraction_squared must lie strictly between zero and one"
            )


@dataclass(frozen=True)
class RelativeState:
    """Target-minus-central state with independently checkable metadata."""

    central_source: str
    target: str
    position: Vec3
    velocity: Vec3
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    target_treatment: TargetTreatment

    def __post_init__(self) -> None:
        central = _require_name(self.central_source, "central_source")
        target = _require_name(self.target, "target")
        if target == central:
            raise RelativityContractError("target must differ from central_source")
        _require_vec3(self.position, "position")
        _require_vec3(self.velocity, "velocity")
        if type(self.units) is not CoherentUnitSystem:
            raise RelativityContractError("state units must be an explicit CoherentUnitSystem")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise RelativityContractError("state frame must be the restricted Sun-relative frame")
        if type(self.time_scale) is not CoordinateTimeScale:
            raise RelativityContractError("state time_scale must be explicitly TCB- or TDB-compatible")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise RelativityContractError("state target_treatment must be MASSLESS_NO_BACKREACTION")


@dataclass(frozen=True)
class ForceEvaluationContext:
    """Per-target stable IDs already present in one force evaluation.

    The caller must create a separate immutable context for each target and
    evaluation epoch.  This object prevents duplicate model application; it
    does not itself bind the epoch or the Decimal precision policy required by
    a future executable registry runner.
    """

    applied_model_ids: frozenset[str]

    def __post_init__(self) -> None:
        if type(self.applied_model_ids) is not frozenset:
            raise RelativityContractError("applied_model_ids must be an immutable frozenset")
        for model_id in self.applied_model_ids:
            _require_name(model_id, "applied_model_id")


@dataclass(frozen=True)
class AccelerationContribution:
    """Immutable, tagged correction; its existence conveys no registry approval."""

    model_id: str
    acceleration: Vec3
    semantics: AccelerationSemantics
    registry_authorized: bool
    central_source: str
    target: str
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    target_treatment: TargetTreatment

    def __post_init__(self) -> None:
        if self.model_id != SOLAR_SCHWARZSCHILD_1PN_MODEL_ID:
            raise RelativityContractError("unexpected acceleration model_id")
        _require_vec3(self.acceleration, "acceleration")
        if self.semantics is not AccelerationSemantics.CORRECTION_ONLY:
            raise RelativityContractError("a 1PN contribution must be correction-only")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise RelativityContractError("the reference kernel cannot assert registry authorization")
        if self.central_source != "SUN":
            raise RelativityContractError("a Solar 1PN contribution must identify central_source='SUN'")
        _require_name(self.target, "target")
        if type(self.units) is not CoherentUnitSystem:
            raise RelativityContractError("contribution units are invalid")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise RelativityContractError("contribution frame is invalid")
        if type(self.time_scale) is not CoordinateTimeScale:
            raise RelativityContractError("contribution time scale is invalid")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise RelativityContractError("contribution target treatment is invalid")


def solar_schwarzschild_1pn_correction(
    contract: SolarSchwarzschild1PNContract,
    state: RelativeState,
    context: ForceEvaluationContext,
) -> AccelerationContribution:
    """Return only the restricted Solar Schwarzschild 1PN acceleration term.

    Assumptions: a static spherical central monopole, a massless/negligible
    target, no target backreaction, and one coherent inertial coordinate state.
    Planet--planet 1PN interactions, source motion terms, spins, multipoles,
    and the full EIH equations are outside this function's scope.
    """

    if type(contract) is not SolarSchwarzschild1PNContract:
        raise RelativityContractError("contract must be SolarSchwarzschild1PNContract")
    if type(state) is not RelativeState:
        raise RelativityContractError("state must be RelativeState")
    if type(context) is not ForceEvaluationContext:
        raise RelativityContractError("context must be ForceEvaluationContext")
    # Refuse a duplicate before performing any arithmetic.
    if SOLAR_SCHWARZSCHILD_1PN_MODEL_ID in context.applied_model_ids:
        raise DuplicateForceApplicationError(
            f"model already applied: {SOLAR_SCHWARZSCHILD_1PN_MODEL_ID}"
        )
    if state.central_source != contract.central_source:
        raise RelativityContractError("state central_source does not match the model contract")
    if state.units is not contract.units:
        raise RelativityContractError("state and physical parameters use different unit systems")
    if state.frame is not contract.frame:
        raise RelativityContractError("state and model frames differ")
    if state.time_scale is not contract.time_scale:
        raise RelativityContractError("state and physical parameters use different time scalings")
    if state.target_treatment is not contract.target_treatment:
        raise RelativityContractError("state and contract target treatments differ")

    r = state.position
    v = state.velocity
    mu = contract.gravitational_parameter
    c = contract.speed_of_light
    try:
        radius_squared = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
        if radius_squared == 0:
            raise RelativityDomainError("the target and central source cannot be coincident")
        radius = radius_squared.sqrt()
        speed_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        c_squared = c * c
        compactness = mu / (radius * c_squared)
        speed_fraction_squared = speed_squared / c_squared
        if compactness > contract.maximum_compactness:
            raise RelativityDomainError("state exceeds the declared weak-field compactness bound")
        if speed_fraction_squared > contract.maximum_speed_fraction_squared:
            raise RelativityDomainError("state exceeds the declared slow-motion speed bound")

        radial_velocity_product = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
        radial_coefficient = 4 * mu / radius - speed_squared
        common = mu / (c_squared * radius_squared * radius)
        correction = tuple(
            common * (radial_coefficient * r[index] + 4 * radial_velocity_product * v[index])
            for index in range(3)
        )
    except DecimalException as exc:
        raise RelativityDomainError("Decimal arithmetic failed in the 1PN domain") from exc

    checked = _require_vec3(correction, "computed correction")
    return AccelerationContribution(
        model_id=SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,
        acceleration=checked,
        semantics=AccelerationSemantics.CORRECTION_ONLY,
        registry_authorized=False,
        central_source=state.central_source,
        target=state.target,
        units=state.units,
        frame=state.frame,
        time_scale=state.time_scale,
        target_treatment=state.target_treatment,
    )


__all__ = [
    "AccelerationContribution",
    "AccelerationSemantics",
    "CentralSourceModel",
    "CoherentUnitSystem",
    "CoordinateTimeScale",
    "DuplicateForceApplicationError",
    "ForceEvaluationContext",
    "InertialFrame",
    "RelativeState",
    "RelativityContractError",
    "RelativityDomainError",
    "SOLAR_SCHWARZSCHILD_1PN_MODEL_ID",
    "SolarSchwarzschild1PNContract",
    "TargetTreatment",
    "solar_schwarzschild_1pn_correction",
]
