"""Isolated Decimal oracle candidate for restricted Solar 1PN dynamics.

This source is deliberately outside :mod:`jxplanetx` and imports no JX code.
It is a non-authorizing implementation candidate awaiting independent review,
not an operationally independent oracle and not a qualification result.

The adaptive integrator uses Fehlberg's 13-stage RK7(8) pair.  Coefficients
are transcribed as exact rationals from NASA TR R-287, Table X (printed p. 65,
PDF p. 72).  Equation (105) (printed p. 52, PDF p. 59) fixes the orientation:
the hatted formula is eighth order and is the accepted solution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Underflow,
    localcontext,
)
from enum import Enum
from typing import Final, TypeAlias


METHOD_ID: Final = "fehlberg.nasa-tr-r-287.rk7-8.table-x.decimal.v1"
SOURCE_SHA256: Final = "5553a2a3eb53785a461762cc2b29428015f1b32c3ad0a5cb57f85a421256a0c8"
SOURCE_SIZE_BYTES: Final = 2_625_098
REVIEW_STATUS: Final = "CANDIDATE_PENDING_INDEPENDENT_REVIEW"
OUTPUT_CLASS: Final = "MODEL_OUTPUT"
REGISTRY_AUTHORIZED: Final = False
CONTROLLER_ID: Final = "fehlberg.rk7-8.normalized-max.exponent-one-eighth.v1"
CONTROLLER_ERROR_EXPONENT_RATIONAL: Final[Rational] = (1, 8)

NEWTONIAN_MODEL_ID: Final = "force.newtonian.point_mass"
SOLAR_1PN_MODEL_ID: Final = "relativity.solar_schwarzschild_test_particle_1pn"

Rational: TypeAlias = tuple[int, int]
StateVector: TypeAlias = tuple[Decimal, ...]
Vec3: TypeAlias = tuple[Decimal, Decimal, Decimal]
RightHandSide: TypeAlias = Callable[[Decimal, StateVector], StateVector]


def _r(numerator: int, denominator: int = 1) -> Rational:
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise AssertionError("internal rational coefficient is invalid")
    return (numerator, denominator)


# NASA TR R-287, Table X.  Rows use zero-based stages and contain only the
# beta[K,lambda] entries lambda < K.  Keeping integer pairs avoids constructing
# any coefficient under the ambient Decimal context.
TABLEAU_ALPHA_RATIONAL: Final[tuple[Rational, ...]] = (
    _r(0),
    _r(2, 27),
    _r(1, 9),
    _r(1, 6),
    _r(5, 12),
    _r(1, 2),
    _r(5, 6),
    _r(1, 6),
    _r(2, 3),
    _r(1, 3),
    _r(1),
    _r(0),
    _r(1),
)

TABLEAU_A_RATIONAL: Final[tuple[tuple[Rational, ...], ...]] = (
    (),
    (_r(2, 27),),
    (_r(1, 36), _r(1, 12)),
    (_r(1, 24), _r(0), _r(1, 8)),
    (_r(5, 12), _r(0), _r(-25, 16), _r(25, 16)),
    (_r(1, 20), _r(0), _r(0), _r(1, 4), _r(1, 5)),
    (
        _r(-25, 108),
        _r(0),
        _r(0),
        _r(125, 108),
        _r(-65, 27),
        _r(125, 54),
    ),
    (
        _r(31, 300),
        _r(0),
        _r(0),
        _r(0),
        _r(61, 225),
        _r(-2, 9),
        _r(13, 900),
    ),
    (
        _r(2),
        _r(0),
        _r(0),
        _r(-53, 6),
        _r(704, 45),
        _r(-107, 9),
        _r(67, 90),
        _r(3),
    ),
    (
        _r(-91, 108),
        _r(0),
        _r(0),
        _r(23, 108),
        _r(-976, 135),
        _r(311, 54),
        _r(-19, 60),
        _r(17, 6),
        _r(-1, 12),
    ),
    (
        _r(2383, 4100),
        _r(0),
        _r(0),
        _r(-341, 164),
        _r(4496, 1025),
        _r(-301, 82),
        _r(2133, 4100),
        _r(45, 82),
        _r(45, 164),
        _r(18, 41),
    ),
    (
        _r(3, 205),
        _r(0),
        _r(0),
        _r(0),
        _r(0),
        _r(-6, 41),
        _r(-3, 205),
        _r(-3, 41),
        _r(3, 41),
        _r(6, 41),
        _r(0),
    ),
    (
        _r(-1777, 4100),
        _r(0),
        _r(0),
        _r(-341, 164),
        _r(4496, 1025),
        _r(-289, 82),
        _r(2193, 4100),
        _r(51, 82),
        _r(33, 164),
        _r(12, 41),
        _r(0),
        _r(1),
    ),
)

# Equation (105): c gives the seventh-order formula and c-hat gives the
# eighth-order formula.  The latter is the accepted state in this candidate.
EMBEDDED_SEVENTH_WEIGHTS_RATIONAL: Final[tuple[Rational, ...]] = (
    _r(41, 840),
    _r(0),
    _r(0),
    _r(0),
    _r(0),
    _r(34, 105),
    _r(9, 35),
    _r(9, 35),
    _r(9, 280),
    _r(9, 280),
    _r(41, 840),
    _r(0),
    _r(0),
)

ACCEPTED_EIGHTH_WEIGHTS_RATIONAL: Final[tuple[Rational, ...]] = (
    _r(0),
    _r(0),
    _r(0),
    _r(0),
    _r(0),
    _r(34, 105),
    _r(9, 35),
    _r(9, 35),
    _r(9, 280),
    _r(9, 280),
    _r(0),
    _r(41, 840),
    _r(41, 840),
)


class OracleContractError(ValueError):
    """The requested oracle configuration is ambiguous or unsupported."""


class OracleDomainError(ValueError):
    """The state or parameters are outside the candidate model domain."""


class OracleIntegrationError(RuntimeError):
    """The adaptive algorithm cannot safely complete the requested step."""


class CoherentUnitSystem(Enum):
    AU_DAY = "position=au;time=day;velocity=au/day;mu=au^3/day^2;c=au/day"
    KILOMETRE_SECOND = "position=km;time=s;velocity=km/s;mu=km^3/s^2;c=km/s"
    METRE_SECOND = "position=m;time=s;velocity=m/s;mu=m^3/s^2;c=m/s"


class InertialFrame(Enum):
    BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED = (
        "simultaneous BCRS-derived Sun-relative state with ICRS-aligned axes; "
        "restricted static-Sun approximation"
    )


class CoordinateTimeScale(Enum):
    TCB_COMPATIBLE = "TCB-compatible"
    TDB_COMPATIBLE = "TDB-compatible"


class CentralSourceModel(Enum):
    STATIC_SPHERICAL_SOLAR_MONOPOLE = "STATIC_SPHERICAL_SOLAR_MONOPOLE"


class TargetTreatment(Enum):
    MASSLESS_NO_BACKREACTION = "MASSLESS_NO_BACKREACTION"


class ForceMode(Enum):
    NEWTONIAN = "NEWTONIAN_SOLAR_MONOPOLE"
    NEWTONIAN_PLUS_SOLAR_1PN = "NEWTONIAN_SOLAR_MONOPOLE_PLUS_RESTRICTED_SOLAR_1PN"


_TRAPPED_SIGNALS: Final = (
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Underflow,
)


def _require_name(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise OracleContractError(f"{label} must be a nonempty, trimmed string")
    return value


def _require_decimal(value: object, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise OracleContractError(f"{label} must be decimal.Decimal (binary floats are forbidden)")
    if not value.is_finite():
        raise OracleDomainError(f"{label} must be finite")
    return value


def _require_vec3(value: object, label: str) -> Vec3:
    if type(value) is not tuple or len(value) != 3:
        raise OracleContractError(f"{label} must be an immutable three-tuple")
    result = tuple(_require_decimal(item, f"{label}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _require_state_vector(value: object, label: str, expected_length: int | None = None) -> StateVector:
    if type(value) is not tuple or not value:
        raise OracleContractError(f"{label} must be a nonempty immutable tuple")
    if expected_length is not None and len(value) != expected_length:
        raise OracleContractError(f"{label} must contain exactly {expected_length} components")
    return tuple(_require_decimal(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class DecimalContextSpec:
    """Complete arithmetic context; ambient process settings are never used."""

    context_id: str
    precision: int
    rounding: str = ROUND_HALF_EVEN
    emin: int = -999_999
    emax: int = 999_999
    capitals: int = 1
    clamp: int = 0

    def __post_init__(self) -> None:
        if type(self.precision) is not int or self.precision not in (50, 70, 90):
            raise OracleContractError("precision must be one of the frozen values 50, 70, or 90")
        expected_id = f"decimal.context.precision_{self.precision}"
        if self.context_id != expected_id:
            raise OracleContractError(f"context_id must be {expected_id!r}")
        if self.rounding != ROUND_HALF_EVEN:
            raise OracleContractError("rounding must be ROUND_HALF_EVEN")
        if type(self.emin) is not int or self.emin != -999_999:
            raise OracleContractError("emin must equal the frozen value -999999")
        if type(self.emax) is not int or self.emax != 999_999:
            raise OracleContractError("emax must equal the frozen value 999999")
        if type(self.capitals) is not int or self.capitals != 1:
            raise OracleContractError("capitals must equal the frozen value 1")
        if type(self.clamp) is not int or self.clamp != 0:
            raise OracleContractError("clamp must equal the frozen value 0")

    def build(self) -> Context:
        context = Context(
            prec=self.precision,
            rounding=self.rounding,
            Emin=self.emin,
            Emax=self.emax,
            capitals=self.capitals,
            clamp=self.clamp,
        )
        trapped = frozenset(_TRAPPED_SIGNALS)
        for signal in context.traps:
            context.traps[signal] = signal in trapped
        context.clear_flags()
        return context


@dataclass(frozen=True)
class OracleContract:
    """Restricted static-Sun, massless-target model and applicability domain."""

    central_source: str
    target: str
    gravitational_parameter: Decimal
    speed_of_light: Decimal
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    central_source_model: CentralSourceModel
    target_treatment: TargetTreatment
    maximum_compactness: Decimal
    maximum_speed_fraction_squared: Decimal

    def __post_init__(self) -> None:
        central = _require_name(self.central_source, "central_source")
        target = _require_name(self.target, "target")
        if central != "SUN":
            raise OracleContractError("central_source must be canonical 'SUN'")
        if target == central:
            raise OracleContractError("target must differ from central_source")
        mu = _require_decimal(self.gravitational_parameter, "gravitational_parameter")
        c = _require_decimal(self.speed_of_light, "speed_of_light")
        compactness = _require_decimal(self.maximum_compactness, "maximum_compactness")
        speed_fraction = _require_decimal(
            self.maximum_speed_fraction_squared, "maximum_speed_fraction_squared"
        )
        if mu <= 0:
            raise OracleDomainError("gravitational_parameter must be positive")
        if c <= 0:
            raise OracleDomainError("speed_of_light must be positive")
        if type(self.units) is not CoherentUnitSystem:
            raise OracleContractError("units must be an explicit CoherentUnitSystem")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise OracleContractError("only the declared restricted Sun-relative frame is supported")
        if type(self.time_scale) is not CoordinateTimeScale:
            raise OracleContractError("time_scale must be explicitly TCB- or TDB-compatible")
        if self.central_source_model is not CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE:
            raise OracleContractError("only a static spherical Solar monopole is supported")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise OracleContractError("only a massless target without backreaction is supported")
        if not (0 < compactness < 1):
            raise OracleDomainError("maximum_compactness must lie strictly between zero and one")
        if not (0 < speed_fraction < 1):
            raise OracleDomainError(
                "maximum_speed_fraction_squared must lie strictly between zero and one"
            )


@dataclass(frozen=True)
class OracleState:
    """Target-minus-Sun Cartesian state in one coherent unit system."""

    central_source: str
    target: str
    position: tuple[Decimal, Decimal, Decimal]
    velocity: tuple[Decimal, Decimal, Decimal]
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    target_treatment: TargetTreatment

    def __post_init__(self) -> None:
        central = _require_name(self.central_source, "central_source")
        target = _require_name(self.target, "target")
        if central != "SUN":
            raise OracleContractError("state central_source must be canonical 'SUN'")
        if target == central:
            raise OracleContractError("state target must differ from central_source")
        _require_vec3(self.position, "position")
        _require_vec3(self.velocity, "velocity")
        if type(self.units) is not CoherentUnitSystem:
            raise OracleContractError("state units must be an explicit CoherentUnitSystem")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise OracleContractError("state frame is unsupported")
        if type(self.time_scale) is not CoordinateTimeScale:
            raise OracleContractError("state time_scale must be explicit")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise OracleContractError("state target_treatment is unsupported")


@dataclass(frozen=True)
class OracleResultMetadata:
    output_class: str = OUTPUT_CLASS
    review_status: str = REVIEW_STATUS
    registry_authorized: bool = REGISTRY_AUTHORIZED
    qualification_outcomes_generated: bool = False

    def __post_init__(self) -> None:
        if self.output_class != OUTPUT_CLASS:
            raise OracleContractError("oracle results must remain MODEL_OUTPUT")
        if self.review_status != REVIEW_STATUS:
            raise OracleContractError("oracle result review status is immutable")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise OracleContractError("oracle candidate cannot assert registry authorization")
        if (
            type(self.qualification_outcomes_generated) is not bool
            or self.qualification_outcomes_generated
        ):
            raise OracleContractError("oracle candidate cannot assert qualification outcomes")


@dataclass(frozen=True)
class ForceEvaluation:
    acceleration: Vec3
    component_model_ids: tuple[str, ...]
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        _require_vec3(self.acceleration, "acceleration")
        expected_ledgers = (
            (NEWTONIAN_MODEL_ID,),
            (NEWTONIAN_MODEL_ID, SOLAR_1PN_MODEL_ID),
        )
        if self.component_model_ids not in expected_ledgers:
            raise OracleContractError("force ledger is incomplete, duplicated, or unsupported")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("force metadata must be immutable OracleResultMetadata")


def _require_state_matches_contract(contract: OracleContract, state: OracleState) -> None:
    if type(contract) is not OracleContract:
        raise OracleContractError("contract must be exactly OracleContract")
    if type(state) is not OracleState:
        raise OracleContractError("state must be exactly OracleState")
    mismatches = (
        state.central_source != contract.central_source,
        state.target != contract.target,
        state.units is not contract.units,
        state.frame is not contract.frame,
        state.time_scale is not contract.time_scale,
        state.target_treatment is not contract.target_treatment,
    )
    if any(mismatches):
        raise OracleContractError("state metadata does not exactly match the model contract")


def _evaluate_acceleration_core(
    contract: OracleContract,
    state: OracleState,
    mode: ForceMode,
) -> ForceEvaluation:
    _require_state_matches_contract(contract, state)
    if type(mode) is not ForceMode:
        raise OracleContractError("mode must be exactly ForceMode")

    r = state.position
    v = state.velocity
    mu = contract.gravitational_parameter
    c = contract.speed_of_light
    radius_squared = sum((component * component for component in r), Decimal(0))
    if radius_squared <= 0:
        raise OracleDomainError("the target-to-Sun radius must be nonzero")
    radius = radius_squared.sqrt()
    radius_cubed = radius_squared * radius
    speed_squared = sum((component * component for component in v), Decimal(0))
    c_squared = c * c
    compactness = mu / (radius * c_squared)
    speed_fraction_squared = speed_squared / c_squared
    if compactness > contract.maximum_compactness:
        raise OracleDomainError("mu/(r*c^2) exceeds the frozen weak-field bound")
    if speed_fraction_squared > contract.maximum_speed_fraction_squared:
        raise OracleDomainError("v^2/c^2 exceeds the frozen slow-motion bound")

    newtonian_scale = -mu / radius_cubed
    total = tuple(newtonian_scale * component for component in r)
    ledger = (NEWTONIAN_MODEL_ID,)
    if mode is ForceMode.NEWTONIAN_PLUS_SOLAR_1PN:
        position_dot_velocity = sum((r[index] * v[index] for index in range(3)), Decimal(0))
        radial_coefficient = Decimal(4) * mu / radius - speed_squared
        correction_scale = mu / (c_squared * radius_cubed)
        correction = tuple(
            correction_scale
            * (
                radial_coefficient * r[index]
                + Decimal(4) * position_dot_velocity * v[index]
            )
            for index in range(3)
        )
        total = tuple(total[index] + correction[index] for index in range(3))
        ledger = (NEWTONIAN_MODEL_ID, SOLAR_1PN_MODEL_ID)
    return ForceEvaluation(total, ledger)  # type: ignore[arg-type]


def evaluate_acceleration(
    contract: OracleContract,
    state: OracleState,
    mode: ForceMode,
    context_spec: DecimalContextSpec,
) -> ForceEvaluation:
    """Evaluate the total restricted acceleration under an isolated context.

    For ``NEWTONIAN_PLUS_SOLAR_1PN`` the separately transcribed equation is

    ``-mu*r/r^3 + mu/(c^2*r^3)*((4*mu/r-v^2)*r + 4*(r.v)*v)``.

    Relative vectors are target minus Sun.  The ledger always includes the
    Newtonian monopole exactly once and optionally the correction exactly once.
    """

    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    try:
        with localcontext(context_spec.build()):
            return _evaluate_acceleration_core(contract, state, mode)
    except DecimalException as exc:
        raise OracleDomainError(f"trapped Decimal signal during force evaluation: {type(exc).__name__}") from exc


def _state_to_vector(state: OracleState) -> StateVector:
    return state.position + state.velocity


def _vector_to_state(contract: OracleContract, vector: StateVector) -> OracleState:
    checked = _require_state_vector(vector, "state vector", 6)
    return OracleState(
        central_source=contract.central_source,
        target=contract.target,
        position=checked[:3],  # type: ignore[arg-type]
        velocity=checked[3:],  # type: ignore[arg-type]
        units=contract.units,
        frame=contract.frame,
        time_scale=contract.time_scale,
        target_treatment=contract.target_treatment,
    )


def _restricted_rhs_core(
    contract: OracleContract,
    mode: ForceMode,
    _epoch: Decimal,
    vector: StateVector,
) -> StateVector:
    state = _vector_to_state(contract, vector)
    acceleration = _evaluate_acceleration_core(contract, state, mode).acceleration
    return state.velocity + acceleration


def evaluate_rhs(
    contract: OracleContract,
    state: OracleState,
    mode: ForceMode,
    epoch: Decimal,
    context_spec: DecimalContextSpec,
) -> StateVector:
    """Return ``(velocity, total_acceleration)`` without importing JX code."""

    checked_epoch = _require_decimal(epoch, "epoch")
    _require_state_matches_contract(contract, state)
    if type(mode) is not ForceMode:
        raise OracleContractError("mode must be exactly ForceMode")
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    try:
        with localcontext(context_spec.build()):
            return _restricted_rhs_core(contract, mode, checked_epoch, _state_to_vector(state))
    except DecimalException as exc:
        raise OracleDomainError(f"trapped Decimal signal during RHS evaluation: {type(exc).__name__}") from exc


def _as_decimal(coefficient: Rational) -> Decimal:
    numerator, denominator = coefficient
    return Decimal(numerator) / Decimal(denominator)


@dataclass(frozen=True)
class RK78StepResult:
    start_epoch: Decimal
    step_size: Decimal
    accepted_eighth: StateVector
    embedded_seventh: StateVector
    error_embedded_minus_accepted: StateVector
    method_id: str = METHOD_ID
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        _require_decimal(self.start_epoch, "start_epoch")
        step = _require_decimal(self.step_size, "step_size")
        if step <= 0:
            raise OracleDomainError("step_size must be positive")
        accepted = _require_state_vector(self.accepted_eighth, "accepted_eighth")
        embedded = _require_state_vector(
            self.embedded_seventh, "embedded_seventh", len(accepted)
        )
        _require_state_vector(
            self.error_embedded_minus_accepted,
            "error_embedded_minus_accepted",
            len(accepted),
        )
        if embedded == accepted:
            # Equality is valid for simple differential equations, so this is
            # intentionally not an error.  The branch documents that both
            # approximations remain independently present.
            pass
        if self.method_id != METHOD_ID:
            raise OracleContractError("step result method_id is immutable")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("step result metadata is invalid")


def _rk78_step_core(
    rhs: RightHandSide,
    start_epoch: Decimal,
    state: StateVector,
    step_size: Decimal,
) -> RK78StepResult:
    stages: list[StateVector] = []
    dimension = len(state)
    for stage_index in range(13):
        stage_state = list(state)
        for previous_index, rational in enumerate(TABLEAU_A_RATIONAL[stage_index]):
            coefficient = _as_decimal(rational)
            if coefficient:
                derivative = stages[previous_index]
                for component in range(dimension):
                    stage_state[component] += step_size * coefficient * derivative[component]
        stage_epoch = start_epoch + step_size * _as_decimal(
            TABLEAU_ALPHA_RATIONAL[stage_index]
        )
        derivative = rhs(stage_epoch, tuple(stage_state))
        stages.append(
            _require_state_vector(
                derivative,
                f"rhs return at stage {stage_index}",
                dimension,
            )
        )

    embedded = list(state)
    accepted = list(state)
    for stage_index, derivative in enumerate(stages):
        weight_seven = _as_decimal(EMBEDDED_SEVENTH_WEIGHTS_RATIONAL[stage_index])
        weight_eight = _as_decimal(ACCEPTED_EIGHTH_WEIGHTS_RATIONAL[stage_index])
        for component in range(dimension):
            embedded[component] += step_size * weight_seven * derivative[component]
            accepted[component] += step_size * weight_eight * derivative[component]
    accepted_tuple = tuple(accepted)
    embedded_tuple = tuple(embedded)
    # NASA TR R-287 Eq. (134): +41/840*(f0+f10-f11-f12)*h.
    error = tuple(
        embedded_tuple[component] - accepted_tuple[component]
        for component in range(dimension)
    )
    return RK78StepResult(
        start_epoch=start_epoch,
        step_size=step_size,
        accepted_eighth=accepted_tuple,
        embedded_seventh=embedded_tuple,
        error_embedded_minus_accepted=error,
    )


def rk78_step(
    rhs: RightHandSide,
    start_epoch: Decimal,
    state: StateVector,
    step_size: Decimal,
    context_spec: DecimalContextSpec,
) -> RK78StepResult:
    """Take one generic Fehlberg step and return both formula orientations."""

    if not callable(rhs):
        raise OracleContractError("rhs must be callable")
    checked_epoch = _require_decimal(start_epoch, "start_epoch")
    checked_state = _require_state_vector(state, "state")
    checked_step = _require_decimal(step_size, "step_size")
    if checked_step <= 0:
        raise OracleDomainError("step_size must be positive")
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    try:
        with localcontext(context_spec.build()):
            return _rk78_step_core(rhs, checked_epoch, checked_state, checked_step)
    except DecimalException as exc:
        raise OracleIntegrationError(
            f"trapped Decimal signal during RK7(8) step: {type(exc).__name__}"
        ) from exc


@dataclass(frozen=True)
class AdaptiveControllerSpec:
    """Frozen normalized-max controller using the pair's order-seven defect.

    ``absolute_tolerances`` follows the state-vector component order.  For the
    restricted model that order is ``(x, y, z, vx, vy, vz)``, so length and
    velocity tolerances occupy distinct, explicit slots.
    """

    absolute_tolerances: tuple[Decimal, ...]
    relative_tolerance: Decimal
    initial_step: Decimal
    minimum_step: Decimal
    maximum_step: Decimal
    safety_factor: Decimal = Decimal("0.9")
    minimum_factor: Decimal = Decimal("0.2")
    maximum_factor: Decimal = Decimal("5")
    maximum_attempts: int = 1_000_000
    controller_id: str = CONTROLLER_ID

    def __post_init__(self) -> None:
        if type(self.absolute_tolerances) is not tuple or not self.absolute_tolerances:
            raise OracleContractError(
                "absolute_tolerances must be a nonempty immutable per-component tuple"
            )
        absolute = tuple(
            _require_decimal(value, f"absolute_tolerances[{index}]")
            for index, value in enumerate(self.absolute_tolerances)
        )
        relative = _require_decimal(self.relative_tolerance, "relative_tolerance")
        initial = _require_decimal(self.initial_step, "initial_step")
        minimum = _require_decimal(self.minimum_step, "minimum_step")
        maximum = _require_decimal(self.maximum_step, "maximum_step")
        safety = _require_decimal(self.safety_factor, "safety_factor")
        minimum_factor = _require_decimal(self.minimum_factor, "minimum_factor")
        maximum_factor = _require_decimal(self.maximum_factor, "maximum_factor")
        if any(value <= 0 for value in absolute) or relative <= 0:
            raise OracleDomainError(
                "every component absolute tolerance and the relative tolerance must be positive"
            )
        if minimum <= 0 or not (minimum <= initial <= maximum):
            raise OracleDomainError("step sizes must satisfy 0 < minimum <= initial <= maximum")
        if safety != Decimal("0.9"):
            raise OracleContractError("safety_factor is frozen at 0.9")
        if minimum_factor != Decimal("0.2"):
            raise OracleContractError("minimum_factor is frozen at 0.2")
        if maximum_factor != Decimal("5"):
            raise OracleContractError("maximum_factor is frozen at 5")
        if type(self.maximum_attempts) is not int or not (1 <= self.maximum_attempts <= 10_000_000):
            raise OracleContractError("maximum_attempts must be an integer in [1, 10000000]")
        if self.controller_id != CONTROLLER_ID:
            raise OracleContractError("controller_id is immutable")


def controller_sha256(controller: AdaptiveControllerSpec) -> str:
    """Bind every adaptive-control field to canonical UTF-8 JSON bytes."""

    if type(controller) is not AdaptiveControllerSpec:
        raise OracleContractError("controller must be exactly AdaptiveControllerSpec")
    record = {
        "absolute_tolerances": [str(value) for value in controller.absolute_tolerances],
        "controller_id": controller.controller_id,
        "initial_step": str(controller.initial_step),
        "maximum_attempts": controller.maximum_attempts,
        "maximum_factor": str(controller.maximum_factor),
        "maximum_step": str(controller.maximum_step),
        "minimum_factor": str(controller.minimum_factor),
        "minimum_step": str(controller.minimum_step),
        "relative_tolerance": str(controller.relative_tolerance),
        "safety_factor": str(controller.safety_factor),
    }
    canonical = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalized_max_error(
    start: StateVector,
    accepted: StateVector,
    defect: StateVector,
    controller: AdaptiveControllerSpec,
) -> Decimal:
    if not (len(start) == len(accepted) == len(defect)):
        raise OracleContractError("normalized-error vectors have different dimensions")
    if len(controller.absolute_tolerances) != len(start):
        raise OracleContractError(
            "per-component absolute-tolerance roster does not match state dimension"
        )
    ratios: list[Decimal] = []
    for index in range(len(start)):
        scale = controller.absolute_tolerances[index] + controller.relative_tolerance * max(
            abs(start[index]), abs(accepted[index])
        )
        if scale <= 0:
            raise OracleIntegrationError("normalized error scale is not positive")
        ratios.append(abs(defect[index]) / scale)
    result = max(ratios)
    if not result.is_finite():
        raise OracleIntegrationError("normalized error is not finite")
    return result


def normalized_max_error(
    start: StateVector,
    accepted: StateVector,
    defect: StateVector,
    controller: AdaptiveControllerSpec,
    context_spec: DecimalContextSpec,
) -> Decimal:
    """Evaluate the frozen per-component normalized maximum error."""

    checked_start = _require_state_vector(start, "start")
    checked_accepted = _require_state_vector(accepted, "accepted", len(checked_start))
    checked_defect = _require_state_vector(defect, "defect", len(checked_start))
    if type(controller) is not AdaptiveControllerSpec:
        raise OracleContractError("controller must be exactly AdaptiveControllerSpec")
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    try:
        with localcontext(context_spec.build()):
            return _normalized_max_error(
                checked_start,
                checked_accepted,
                checked_defect,
                controller,
            )
    except DecimalException as exc:
        raise OracleIntegrationError(
            f"trapped Decimal signal during error normalization: {type(exc).__name__}"
        ) from exc


def _adaptive_step_factor_core(
    error_norm: Decimal,
    controller: AdaptiveControllerSpec,
) -> Decimal:
    if error_norm < 0 or not error_norm.is_finite():
        raise OracleDomainError("error_norm must be finite and nonnegative")
    if error_norm == 0:
        return controller.maximum_factor
    # Three Decimal square roots are exactly the frozen exponent 1/8.  No
    # binary float or implementation-defined fractional power is introduced.
    eighth_root = error_norm.sqrt().sqrt().sqrt()
    factor = controller.safety_factor / eighth_root
    return min(controller.maximum_factor, max(controller.minimum_factor, factor))


def adaptive_step_factor(
    error_norm: Decimal,
    controller: AdaptiveControllerSpec,
    context_spec: DecimalContextSpec,
) -> Decimal:
    checked_error = _require_decimal(error_norm, "error_norm")
    if type(controller) is not AdaptiveControllerSpec:
        raise OracleContractError("controller must be exactly AdaptiveControllerSpec")
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    try:
        with localcontext(context_spec.build()):
            return _adaptive_step_factor_core(checked_error, controller)
    except DecimalException as exc:
        raise OracleIntegrationError(
            f"trapped Decimal signal in adaptive controller: {type(exc).__name__}"
        ) from exc


@dataclass(frozen=True)
class VectorCheckpoint:
    epoch: Decimal
    state: StateVector
    accepted_steps: int
    rejected_steps: int
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        _require_decimal(self.epoch, "checkpoint epoch")
        _require_state_vector(self.state, "checkpoint state")
        if type(self.accepted_steps) is not int or self.accepted_steps < 0:
            raise OracleContractError("accepted_steps must be a nonnegative integer")
        if type(self.rejected_steps) is not int or self.rejected_steps < 0:
            raise OracleContractError("rejected_steps must be a nonnegative integer")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("checkpoint metadata is invalid")


@dataclass(frozen=True)
class ExactCheckpointIntegrationResult:
    checkpoints: tuple[VectorCheckpoint, ...]
    context_id: str
    initial_epoch: Decimal
    initial_state: StateVector
    controller: AdaptiveControllerSpec
    controller_sha256: str
    controller_id: str = CONTROLLER_ID
    method_id: str = METHOD_ID
    checkpoint_policy: str = "CLIP_EACH_STEP_TO_EXACT_CHECKPOINT_NO_INTERPOLATION"
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        initial_epoch = _require_decimal(self.initial_epoch, "integration initial_epoch")
        initial_state = _require_state_vector(self.initial_state, "integration initial_state")
        if type(self.checkpoints) is not tuple or not self.checkpoints:
            raise OracleContractError("integration result must contain immutable checkpoints")
        if any(type(item) is not VectorCheckpoint for item in self.checkpoints):
            raise OracleContractError("integration result contains an invalid checkpoint")
        expected_dimension = len(initial_state)
        previous_epoch: Decimal | None = None
        previous_accepted = -1
        previous_rejected = -1
        for item in self.checkpoints:
            if len(item.state) != expected_dimension:
                raise OracleContractError("checkpoint state dimension changed within one result")
            if previous_epoch is not None and item.epoch <= previous_epoch:
                raise OracleContractError("checkpoint epochs must be strictly increasing")
            if item.epoch < initial_epoch:
                raise OracleContractError("checkpoint precedes the retained initial_epoch")
            if item.epoch == initial_epoch and (
                item.state != initial_state
                or item.accepted_steps != 0
                or item.rejected_steps != 0
            ):
                raise OracleContractError(
                    "checkpoint at initial_epoch does not reproduce the retained initial condition"
                )
            if item.accepted_steps < previous_accepted or item.rejected_steps < previous_rejected:
                raise OracleContractError("checkpoint step counters must be monotone")
            if item.metadata != self.metadata:
                raise OracleContractError("checkpoint metadata is not closed over the result")
            previous_epoch = item.epoch
            previous_accepted = item.accepted_steps
            previous_rejected = item.rejected_steps
        if self.context_id not in {
            "decimal.context.precision_50",
            "decimal.context.precision_70",
            "decimal.context.precision_90",
        }:
            raise OracleContractError("integration result context_id is not frozen")
        if type(self.controller) is not AdaptiveControllerSpec:
            raise OracleContractError("integration result controller record is invalid")
        if self.controller_sha256 != controller_sha256(self.controller):
            raise OracleContractError("integration result controller_sha256 does not bind its record")
        if self.controller_id != self.controller.controller_id:
            raise OracleContractError("integration result controller_id does not bind its record")
        if self.controller_id != CONTROLLER_ID or self.method_id != METHOD_ID:
            raise OracleContractError("integration method/controller identity is immutable")
        if self.checkpoint_policy != "CLIP_EACH_STEP_TO_EXACT_CHECKPOINT_NO_INTERPOLATION":
            raise OracleContractError("checkpoint policy is immutable")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("integration metadata is invalid")


def _clamp_step(step: Decimal, controller: AdaptiveControllerSpec) -> Decimal:
    return min(controller.maximum_step, max(controller.minimum_step, step))


def integrate_rhs_exact_checkpoints(
    rhs: RightHandSide,
    initial_epoch: Decimal,
    initial_state: StateVector,
    checkpoints: tuple[Decimal, ...],
    controller: AdaptiveControllerSpec,
    context_spec: DecimalContextSpec,
) -> ExactCheckpointIntegrationResult:
    """Adapt a generic RHS to exact forward checkpoints without interpolation."""

    if not callable(rhs):
        raise OracleContractError("rhs must be callable")
    checked_epoch = _require_decimal(initial_epoch, "initial_epoch")
    checked_state = _require_state_vector(initial_state, "initial_state")
    if type(checkpoints) is not tuple or not checkpoints:
        raise OracleContractError("checkpoints must be a nonempty immutable tuple")
    checked_checkpoints = tuple(
        _require_decimal(epoch, f"checkpoints[{index}]")
        for index, epoch in enumerate(checkpoints)
    )
    previous = checked_epoch
    for index, epoch in enumerate(checked_checkpoints):
        if epoch < checked_epoch:
            raise OracleDomainError("backward checkpoints are unsupported")
        if index and epoch <= previous:
            raise OracleDomainError("checkpoints must be strictly increasing")
        previous = epoch
    if type(controller) is not AdaptiveControllerSpec:
        raise OracleContractError("controller must be exactly AdaptiveControllerSpec")
    if len(controller.absolute_tolerances) != len(checked_state):
        raise OracleContractError(
            "per-component absolute-tolerance roster does not match initial_state dimension"
        )
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")

    try:
        with localcontext(context_spec.build()):
            epoch = checked_epoch
            state = checked_state
            next_step = controller.initial_step
            attempts = 0
            accepted_steps = 0
            rejected_steps = 0
            retained: list[VectorCheckpoint] = []
            for checkpoint in checked_checkpoints:
                while epoch < checkpoint:
                    if attempts >= controller.maximum_attempts:
                        raise OracleIntegrationError("maximum_attempts exhausted before checkpoint")
                    remaining = checkpoint - epoch
                    step_size = min(next_step, remaining)
                    if step_size < controller.minimum_step:
                        raise OracleIntegrationError(
                            "exact-checkpoint clip fell below the frozen minimum_step"
                        )
                    if epoch + step_size == epoch:
                        raise OracleIntegrationError("time failed to advance at the frozen precision")
                    attempt = _rk78_step_core(rhs, epoch, state, step_size)
                    attempts += 1
                    error_norm = _normalized_max_error(
                        state,
                        attempt.accepted_eighth,
                        attempt.error_embedded_minus_accepted,
                        controller,
                    )
                    factor = _adaptive_step_factor_core(error_norm, controller)
                    proposed_step = _clamp_step(step_size * factor, controller)
                    if error_norm <= 1:
                        advanced_epoch = epoch + step_size
                        if step_size == remaining:
                            if advanced_epoch != checkpoint:
                                raise OracleIntegrationError(
                                    "Decimal endpoint is not exactly the requested checkpoint"
                                )
                            advanced_epoch = checkpoint
                        if advanced_epoch <= epoch:
                            raise OracleIntegrationError("accepted step did not advance time")
                        epoch = advanced_epoch
                        state = attempt.accepted_eighth
                        accepted_steps += 1
                        next_step = proposed_step
                    else:
                        rejected_steps += 1
                        if step_size <= controller.minimum_step:
                            raise OracleIntegrationError(
                                "tolerance cannot be met at the frozen minimum_step"
                            )
                        next_step = proposed_step
                        if next_step >= step_size:
                            raise OracleIntegrationError(
                                "rejected step did not shrink under the frozen controller"
                            )
                if epoch != checkpoint:
                    raise OracleIntegrationError("checkpoint was not reached exactly")
                retained.append(
                    VectorCheckpoint(
                        epoch=checkpoint,
                        state=state,
                        accepted_steps=accepted_steps,
                        rejected_steps=rejected_steps,
                    )
                )
            return ExactCheckpointIntegrationResult(
                checkpoints=tuple(retained),
                context_id=context_spec.context_id,
                initial_epoch=checked_epoch,
                initial_state=checked_state,
                controller=controller,
                controller_sha256=controller_sha256(controller),
            )
    except DecimalException as exc:
        raise OracleIntegrationError(
            f"trapped Decimal signal during adaptive integration: {type(exc).__name__}"
        ) from exc


@dataclass(frozen=True)
class OracleCheckpoint:
    epoch: Decimal
    state: OracleState
    accepted_steps: int
    rejected_steps: int
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        _require_decimal(self.epoch, "checkpoint epoch")
        if type(self.state) is not OracleState:
            raise OracleContractError("checkpoint state must be exactly OracleState")
        if type(self.accepted_steps) is not int or self.accepted_steps < 0:
            raise OracleContractError("accepted_steps must be a nonnegative integer")
        if type(self.rejected_steps) is not int or self.rejected_steps < 0:
            raise OracleContractError("rejected_steps must be a nonnegative integer")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("checkpoint metadata is invalid")


@dataclass(frozen=True)
class OracleIntegrationResult:
    checkpoints: tuple[OracleCheckpoint, ...]
    force_mode: ForceMode
    context_id: str
    contract: OracleContract
    initial_epoch: Decimal
    initial_state: OracleState
    controller: AdaptiveControllerSpec
    controller_sha256: str
    controller_id: str = CONTROLLER_ID
    method_id: str = METHOD_ID
    checkpoint_policy: str = "CLIP_EACH_STEP_TO_EXACT_CHECKPOINT_NO_INTERPOLATION"
    metadata: OracleResultMetadata = OracleResultMetadata()

    def __post_init__(self) -> None:
        if type(self.contract) is not OracleContract:
            raise OracleContractError("oracle result contract record is invalid")
        initial_epoch = _require_decimal(self.initial_epoch, "oracle initial_epoch")
        _require_state_matches_contract(self.contract, self.initial_state)
        if type(self.checkpoints) is not tuple or not self.checkpoints:
            raise OracleContractError("oracle result must contain immutable checkpoints")
        if any(type(item) is not OracleCheckpoint for item in self.checkpoints):
            raise OracleContractError("oracle result contains an invalid checkpoint")
        expected_state_metadata = (
            self.contract.central_source,
            self.contract.target,
            self.contract.units,
            self.contract.frame,
            self.contract.time_scale,
            self.contract.target_treatment,
        )
        previous_epoch: Decimal | None = None
        previous_accepted = -1
        previous_rejected = -1
        for item in self.checkpoints:
            state_metadata = (
                item.state.central_source,
                item.state.target,
                item.state.units,
                item.state.frame,
                item.state.time_scale,
                item.state.target_treatment,
            )
            if state_metadata != expected_state_metadata:
                raise OracleContractError("checkpoint state metadata changed within one result")
            if previous_epoch is not None and item.epoch <= previous_epoch:
                raise OracleContractError("checkpoint epochs must be strictly increasing")
            if item.epoch < initial_epoch:
                raise OracleContractError("checkpoint precedes the retained initial_epoch")
            if item.epoch == initial_epoch and (
                item.state != self.initial_state
                or item.accepted_steps != 0
                or item.rejected_steps != 0
            ):
                raise OracleContractError(
                    "checkpoint at initial_epoch does not reproduce the retained initial condition"
                )
            if item.accepted_steps < previous_accepted or item.rejected_steps < previous_rejected:
                raise OracleContractError("checkpoint step counters must be monotone")
            if item.metadata != self.metadata:
                raise OracleContractError("checkpoint metadata is not closed over the result")
            previous_epoch = item.epoch
            previous_accepted = item.accepted_steps
            previous_rejected = item.rejected_steps
        if type(self.force_mode) is not ForceMode:
            raise OracleContractError("oracle result force_mode is invalid")
        if self.context_id not in {
            "decimal.context.precision_50",
            "decimal.context.precision_70",
            "decimal.context.precision_90",
        }:
            raise OracleContractError("oracle result context_id is not frozen")
        if type(self.controller) is not AdaptiveControllerSpec:
            raise OracleContractError("oracle result controller record is invalid")
        if self.controller_sha256 != controller_sha256(self.controller):
            raise OracleContractError("oracle result controller_sha256 does not bind its record")
        if self.controller_id != self.controller.controller_id:
            raise OracleContractError("oracle result controller_id does not bind its record")
        if self.controller_id != CONTROLLER_ID or self.method_id != METHOD_ID:
            raise OracleContractError("oracle result method/controller identity is immutable")
        if self.checkpoint_policy != "CLIP_EACH_STEP_TO_EXACT_CHECKPOINT_NO_INTERPOLATION":
            raise OracleContractError("oracle result checkpoint policy is immutable")
        if type(self.metadata) is not OracleResultMetadata:
            raise OracleContractError("oracle result metadata is invalid")


def integrate_exact_checkpoints(
    contract: OracleContract,
    initial_epoch: Decimal,
    initial_state: OracleState,
    checkpoints: tuple[Decimal, ...],
    force_mode: ForceMode,
    controller: AdaptiveControllerSpec,
    context_spec: DecimalContextSpec,
) -> OracleIntegrationResult:
    """Integrate the restricted model in memory; never write qualification output."""

    _require_state_matches_contract(contract, initial_state)
    if type(force_mode) is not ForceMode:
        raise OracleContractError("force_mode must be exactly ForceMode")

    def rhs(epoch: Decimal, vector: StateVector) -> StateVector:
        return _restricted_rhs_core(contract, force_mode, epoch, vector)

    generic = integrate_rhs_exact_checkpoints(
        rhs=rhs,
        initial_epoch=initial_epoch,
        initial_state=_state_to_vector(initial_state),
        checkpoints=checkpoints,
        controller=controller,
        context_spec=context_spec,
    )
    retained = tuple(
        OracleCheckpoint(
            epoch=item.epoch,
            state=_vector_to_state(contract, item.state),
            accepted_steps=item.accepted_steps,
            rejected_steps=item.rejected_steps,
        )
        for item in generic.checkpoints
    )
    return OracleIntegrationResult(
        checkpoints=retained,
        force_mode=force_mode,
        context_id=context_spec.context_id,
        contract=contract,
        initial_epoch=generic.initial_epoch,
        initial_state=initial_state,
        controller=generic.controller,
        controller_sha256=generic.controller_sha256,
    )


def max_scaled_checkpoint_discrepancy(
    first: ExactCheckpointIntegrationResult,
    second: ExactCheckpointIntegrationResult,
    absolute_scale: Decimal,
    relative_scale: Decimal,
    context_spec: DecimalContextSpec,
) -> Decimal:
    """Return a strict normalized discrepancy for generic self-convergence."""

    if type(first) is not ExactCheckpointIntegrationResult:
        raise OracleContractError("first result has the wrong type")
    if type(second) is not ExactCheckpointIntegrationResult:
        raise OracleContractError("second result has the wrong type")
    absolute = _require_decimal(absolute_scale, "absolute_scale")
    relative = _require_decimal(relative_scale, "relative_scale")
    if absolute <= 0 or relative <= 0:
        raise OracleDomainError("self-convergence scales must both be positive")
    if type(context_spec) is not DecimalContextSpec:
        raise OracleContractError("context_spec must be exactly DecimalContextSpec")
    if len(first.checkpoints) != len(second.checkpoints):
        raise OracleContractError("self-convergence results have different checkpoint rosters")
    if first.initial_epoch != second.initial_epoch or first.initial_state != second.initial_state:
        raise OracleContractError("self-convergence results have different initial conditions")
    try:
        with localcontext(context_spec.build()):
            maximum = Decimal(0)
            for left, right in zip(first.checkpoints, second.checkpoints, strict=True):
                if left.epoch != right.epoch or len(left.state) != len(right.state):
                    raise OracleContractError(
                        "self-convergence results are not aligned at exact checkpoints"
                    )
                for left_value, right_value in zip(left.state, right.state, strict=True):
                    scale = absolute + relative * max(abs(left_value), abs(right_value))
                    maximum = max(maximum, abs(left_value - right_value) / scale)
            return maximum
    except DecimalException as exc:
        raise OracleIntegrationError(
            f"trapped Decimal signal during self-convergence: {type(exc).__name__}"
        ) from exc


__all__ = [
    "ACCEPTED_EIGHTH_WEIGHTS_RATIONAL",
    "AdaptiveControllerSpec",
    "CONTROLLER_ERROR_EXPONENT_RATIONAL",
    "CONTROLLER_ID",
    "CentralSourceModel",
    "CoherentUnitSystem",
    "CoordinateTimeScale",
    "DecimalContextSpec",
    "EMBEDDED_SEVENTH_WEIGHTS_RATIONAL",
    "ExactCheckpointIntegrationResult",
    "ForceEvaluation",
    "ForceMode",
    "InertialFrame",
    "METHOD_ID",
    "NEWTONIAN_MODEL_ID",
    "OUTPUT_CLASS",
    "OracleCheckpoint",
    "OracleContract",
    "OracleContractError",
    "OracleDomainError",
    "OracleIntegrationError",
    "OracleIntegrationResult",
    "OracleResultMetadata",
    "OracleState",
    "REGISTRY_AUTHORIZED",
    "REVIEW_STATUS",
    "RK78StepResult",
    "SOLAR_1PN_MODEL_ID",
    "SOURCE_SHA256",
    "SOURCE_SIZE_BYTES",
    "TABLEAU_ALPHA_RATIONAL",
    "TABLEAU_A_RATIONAL",
    "TargetTreatment",
    "VectorCheckpoint",
    "adaptive_step_factor",
    "controller_sha256",
    "evaluate_acceleration",
    "evaluate_rhs",
    "integrate_exact_checkpoints",
    "integrate_rhs_exact_checkpoints",
    "max_scaled_checkpoint_discrepancy",
    "normalized_max_error",
    "rk78_step",
]
