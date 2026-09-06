"""Decimal-only, nonauthorizing JX V5 Q2 perihelion observer candidate.

The candidate observes immutable in-memory checkpoint series.  It owns its
dense-state rule: position is the cubic Hermite polynomial determined by the
endpoint positions and velocities, and velocity is the derivative of that
same polynomial.  It never integrates dynamics, loads fixtures, writes files,
or imports JX/oracle implementation code.

This same-repository source is development ``MODEL_OUTPUT`` pending independent
review.  It does not fill the external Q2 registration and does not perform
step, event-tolerance, inverse-c-squared, analytic-target, budget, or gate
adjudication.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    Underflow,
    localcontext,
)
from enum import Enum
from typing import Final, TypeAlias


OBSERVER_ID: Final = "jx.v5.solar_1pn.q2.perihelion_observer_candidate.v1"
DENSE_STATE_METHOD_ID: Final = "decimal.cubic-hermite.position-with-derivative-velocity.v1"
ROOT_METHOD_ID: Final = "decimal.bisection.rdotv.negative-to-positive.v1"
ROOT_ISOLATION_METHOD_ID: Final = "decimal.bernstein-degree5.sign-variation.v1"
ANGLE_METHOD_ID: Final = "decimal.oriented-atan2.dlmf-reduced-series.v1"
UNWRAP_METHOD_ID: Final = "nearest-principal-branch.exact-pi-tie-rejected.v1"
REGRESSION_METHOD_ID: Final = "decimal.ols.angle-on-integer-orbit-index.v1"
OUTPUT_CLASS: Final = "MODEL_OUTPUT"
REVIEW_STATUS: Final = "CANDIDATE_PENDING_INDEPENDENT_REVIEW"
REGISTRY_AUTHORIZED: Final = False
EXECUTION_AUTHORIZED: Final = False
QUALIFICATION_OUTCOMES_GENERATED: Final = False

Vec3: TypeAlias = tuple[Decimal, Decimal, Decimal]
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TRAPPED_SIGNALS: Final = (
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Underflow,
)


class ObserverContractError(ValueError):
    """An observer input is ambiguous, mutable, or unsupported."""


class ObserverDomainError(ValueError):
    """The supplied series cannot define the declared observable."""


class ObserverArithmeticError(RuntimeError):
    """A bounded Decimal calculation could not complete safely."""


class ForceMode(Enum):
    NEWTONIAN = "NEWTONIAN_SOLAR_MONOPOLE"
    NEWTONIAN_PLUS_SOLAR_1PN = (
        "NEWTONIAN_SOLAR_MONOPOLE_PLUS_RESTRICTED_SOLAR_1PN"
    )


def _decimal(value: object, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise ObserverContractError(
            f"{label} must be decimal.Decimal; binary floats are forbidden"
        )
    if not value.is_finite():
        raise ObserverDomainError(f"{label} must be finite")
    return value


def _vec3(value: object, label: str) -> Vec3:
    if type(value) is not tuple or len(value) != 3:
        raise ObserverContractError(f"{label} must be an immutable three-tuple")
    checked = tuple(
        _decimal(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return checked  # type: ignore[return-value]


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ObserverContractError(f"{label} must be a canonical identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ObserverContractError(f"{label} must be a lowercase SHA-256")
    return value


def _literal(value: object, expected: str, label: str) -> None:
    if type(value) is not str or value != expected:
        raise ObserverContractError(f"{label} must equal the frozen literal {expected!r}")


def _state_encoding(state: "CartesianState") -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(str(value) for value in state.position),
        tuple(str(value) for value in state.velocity),
    )


def _canonical_sha256(record: object) -> str:
    raw = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DecimalContextSpec:
    context_id: str = "decimal.context.precision_90"
    precision: int = 90
    rounding: str = ROUND_HALF_EVEN
    emin: int = -999_999
    emax: int = 999_999
    capitals: int = 1
    clamp: int = 0

    def __post_init__(self) -> None:
        _literal(self.context_id, "decimal.context.precision_90", "context_id")
        _literal(self.rounding, "ROUND_HALF_EVEN", "rounding")
        expected = (
            self.context_id == "decimal.context.precision_90",
            type(self.precision) is int and self.precision == 90,
            self.rounding == ROUND_HALF_EVEN,
            type(self.emin) is int and self.emin == -999_999,
            type(self.emax) is int and self.emax == 999_999,
            type(self.capitals) is int and self.capitals == 1,
            type(self.clamp) is int and self.clamp == 0,
        )
        if not all(expected):
            raise ObserverContractError("Decimal context must equal the frozen p90 contract")

    def build(self, *, guard_digits: int = 0) -> Context:
        if type(guard_digits) is not int or not (0 <= guard_digits <= 32):
            raise ObserverContractError("guard_digits must be an integer in [0, 32]")
        context = Context(
            prec=self.precision + guard_digits,
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
class ObserverMetadata:
    output_class: str = OUTPUT_CLASS
    review_status: str = REVIEW_STATUS
    registry_authorized: bool = REGISTRY_AUTHORIZED
    execution_authorized: bool = EXECUTION_AUTHORIZED
    qualification_outcomes_generated: bool = QUALIFICATION_OUTCOMES_GENERATED

    def __post_init__(self) -> None:
        _literal(self.output_class, "MODEL_OUTPUT", "output_class")
        _literal(
            self.review_status,
            "CANDIDATE_PENDING_INDEPENDENT_REVIEW",
            "review_status",
        )
        flags = (
            self.registry_authorized,
            self.execution_authorized,
            self.qualification_outcomes_generated,
        )
        if any(type(flag) is not bool or flag for flag in flags):
            raise ObserverContractError("observer candidate cannot assert authority or outcomes")


@dataclass(frozen=True)
class CartesianState:
    position: Vec3
    velocity: Vec3

    def __post_init__(self) -> None:
        _vec3(self.position, "position")
        _vec3(self.velocity, "velocity")


@dataclass(frozen=True)
class StateSample:
    epoch: Decimal
    state: CartesianState

    def __post_init__(self) -> None:
        _decimal(self.epoch, "sample epoch")
        if type(self.state) is not CartesianState:
            raise ObserverContractError("sample state must be exactly CartesianState")


@dataclass(frozen=True)
class InitialInput:
    input_id: str
    initial_epoch: Decimal
    initial_state: CartesianState
    length_unit_id: str
    time_unit_id: str
    velocity_unit_id: str
    frame_id: str
    orientation_id: str
    time_scale_id: str
    origin_id: str
    target_id: str

    def __post_init__(self) -> None:
        _identifier(self.input_id, "input_id")
        _decimal(self.initial_epoch, "initial_epoch")
        if type(self.initial_state) is not CartesianState:
            raise ObserverContractError("initial_state must be exactly CartesianState")
        for name in (
            "length_unit_id",
            "time_unit_id",
            "velocity_unit_id",
            "frame_id",
            "orientation_id",
            "time_scale_id",
            "origin_id",
            "target_id",
        ):
            _identifier(getattr(self, name), name)
        frozen_coordinates = {
            "length_unit_id": "unit.au",
            "time_unit_id": "unit.day",
            "velocity_unit_id": "unit.au_per_day",
            "frame_id": "BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED",
            "orientation_id": "ICRS_ALIGNED_RESTRICTED",
            "time_scale_id": "TDB_COMPATIBLE",
            "origin_id": "SUN",
        }
        for name, expected in frozen_coordinates.items():
            if getattr(self, name) != expected:
                raise ObserverContractError(f"{name} must equal the frozen Q2 value")
        if self.origin_id == self.target_id:
            raise ObserverContractError("origin_id and target_id must differ")


def initial_input_sha256(value: InitialInput) -> str:
    if type(value) is not InitialInput:
        raise ObserverContractError("value must be exactly InitialInput")
    state = value.initial_state
    return _canonical_sha256(
        {
            "schema": "jx-v5-q2-observer-initial-input/v1",
            "frame_id": value.frame_id,
            "initial_epoch": str(value.initial_epoch),
            "input_id": value.input_id,
            "length_unit_id": value.length_unit_id,
            "orientation_id": value.orientation_id,
            "origin_id": value.origin_id,
            "position": [str(item) for item in state.position],
            "target_id": value.target_id,
            "time_scale_id": value.time_scale_id,
            "time_unit_id": value.time_unit_id,
            "velocity": [str(item) for item in state.velocity],
            "velocity_unit_id": value.velocity_unit_id,
        }
    )


@dataclass(frozen=True)
class TrajectorySeries:
    trajectory_id: str
    schedule_pair_id: str
    force_mode: ForceMode
    source_configuration_id: str
    source_configuration_sha256: str
    initial_input: InitialInput
    samples: tuple[StateSample, ...]

    def __post_init__(self) -> None:
        _identifier(self.trajectory_id, "trajectory_id")
        _identifier(self.schedule_pair_id, "schedule_pair_id")
        if type(self.force_mode) is not ForceMode:
            raise ObserverContractError("force_mode must be exactly ForceMode")
        _identifier(self.source_configuration_id, "source_configuration_id")
        _sha256(self.source_configuration_sha256, "source_configuration_sha256")
        if type(self.initial_input) is not InitialInput:
            raise ObserverContractError("initial_input must be exactly InitialInput")
        if type(self.samples) is not tuple or len(self.samples) < 5:
            raise ObserverContractError("samples must be an immutable tuple with at least five items")
        if any(type(item) is not StateSample for item in self.samples):
            raise ObserverContractError("every sample must be exactly StateSample")
        first = self.samples[0]
        if (
            str(first.epoch) != str(self.initial_input.initial_epoch)
            or _state_encoding(first.state) != _state_encoding(self.initial_input.initial_state)
        ):
            raise ObserverContractError("first sample must reproduce the initial input")
        previous = first.epoch
        for item in self.samples[1:]:
            if item.epoch <= previous:
                raise ObserverContractError("sample epochs must be strictly increasing")
            previous = item.epoch


def sampling_schedule_sha256(series: TrajectorySeries) -> str:
    if type(series) is not TrajectorySeries:
        raise ObserverContractError("series must be exactly TrajectorySeries")
    return _canonical_sha256(
        {
            "schema": "jx-v5-q2-observer-sampling-schedule/v1",
            "initial_input_sha256": initial_input_sha256(series.initial_input),
            "schedule_pair_id": series.schedule_pair_id,
            "sample_epochs": [str(item.epoch) for item in series.samples],
        }
    )


def trajectory_series_sha256(series: TrajectorySeries) -> str:
    if type(series) is not TrajectorySeries:
        raise ObserverContractError("series must be exactly TrajectorySeries")
    return _canonical_sha256(
        {
            "schema": "jx-v5-q2-observer-trajectory-series/v1",
            "force_mode": series.force_mode.value,
            "sampling_schedule_sha256": sampling_schedule_sha256(series),
            "samples": [
                {
                    "epoch": str(item.epoch),
                    "position": [str(value) for value in item.state.position],
                    "velocity": [str(value) for value in item.state.velocity],
                }
                for item in series.samples
            ],
            "source_configuration_id": series.source_configuration_id,
            "source_configuration_sha256": series.source_configuration_sha256,
            "trajectory_id": series.trajectory_id,
        }
    )


@dataclass(frozen=True)
class ObserverConfiguration:
    configuration_id: str
    root_bracket_width_tolerance: Decimal
    maximum_relative_off_plane: Decimal
    maximum_root_iterations: int = 256
    certificate_precision: int = 4096
    context: DecimalContextSpec = DecimalContextSpec()
    expected_event_count: int = 2
    atan_guard_digits: int = 12
    maximum_atan_series_iterations: int = 10_000
    epoch_unit_id: str = "unit.day"
    root_tolerance_unit_id: str = "unit.day"
    angle_unit_id: str = "unit.radian"
    observer_id: str = OBSERVER_ID
    dense_state_method_id: str = DENSE_STATE_METHOD_ID
    root_method_id: str = ROOT_METHOD_ID
    root_isolation_method_id: str = ROOT_ISOLATION_METHOD_ID
    angle_method_id: str = ANGLE_METHOD_ID
    unwrap_method_id: str = UNWRAP_METHOD_ID
    regression_method_id: str = REGRESSION_METHOD_ID

    def __post_init__(self) -> None:
        _identifier(self.configuration_id, "configuration_id")
        tolerance = _decimal(
            self.root_bracket_width_tolerance, "root_bracket_width_tolerance"
        )
        off_plane = _decimal(self.maximum_relative_off_plane, "maximum_relative_off_plane")
        frozen_root_tolerances = (Decimal("1e-20"), Decimal("1e-30"))
        if not any(tolerance.as_tuple() == item.as_tuple() for item in frozen_root_tolerances):
            raise ObserverContractError(
                "root_bracket_width_tolerance must be one frozen Q2 event-grid value"
            )
        if not (0 <= off_plane < 1):
            raise ObserverDomainError("observer tolerances are outside their declared domains")
        if type(self.maximum_root_iterations) is not int or self.maximum_root_iterations != 256:
            raise ObserverContractError("maximum_root_iterations is frozen at 256")
        if type(self.certificate_precision) is not int or self.certificate_precision != 4096:
            raise ObserverContractError("certificate_precision is frozen at 4096")
        if type(self.context) is not DecimalContextSpec:
            raise ObserverContractError("context must be exactly DecimalContextSpec")
        if type(self.expected_event_count) is not int or self.expected_event_count != 2:
            raise ObserverContractError("expected_event_count is frozen at two")
        if type(self.atan_guard_digits) is not int or self.atan_guard_digits != 12:
            raise ObserverContractError("atan_guard_digits is frozen at 12")
        if (
            type(self.maximum_atan_series_iterations) is not int
            or self.maximum_atan_series_iterations != 10_000
        ):
            raise ObserverContractError("maximum_atan_series_iterations is frozen at 10000")
        constants = {
            "epoch_unit_id": "unit.day",
            "root_tolerance_unit_id": "unit.day",
            "angle_unit_id": "unit.radian",
            "observer_id": "jx.v5.solar_1pn.q2.perihelion_observer_candidate.v1",
            "dense_state_method_id": "decimal.cubic-hermite.position-with-derivative-velocity.v1",
            "root_method_id": "decimal.bisection.rdotv.negative-to-positive.v1",
            "root_isolation_method_id": "decimal.bernstein-degree5.sign-variation.v1",
            "angle_method_id": "decimal.oriented-atan2.dlmf-reduced-series.v1",
            "unwrap_method_id": "nearest-principal-branch.exact-pi-tie-rejected.v1",
            "regression_method_id": "decimal.ols.angle-on-integer-orbit-index.v1",
        }
        for name, expected in constants.items():
            _literal(getattr(self, name), expected, name)


def observer_configuration_sha256(value: ObserverConfiguration) -> str:
    if type(value) is not ObserverConfiguration:
        raise ObserverContractError("value must be exactly ObserverConfiguration")
    context = value.context
    return _canonical_sha256(
        {
            "schema": "jx-v5-q2-observer-configuration/v1",
            "angle_method_id": value.angle_method_id,
            "angle_unit_id": value.angle_unit_id,
            "atan_guard_digits": value.atan_guard_digits,
            "configuration_id": value.configuration_id,
            "context": {
                "capitals": context.capitals,
                "clamp": context.clamp,
                "context_id": context.context_id,
                "emax": context.emax,
                "emin": context.emin,
                "precision": context.precision,
                "rounding": context.rounding,
            },
            "dense_state_method_id": value.dense_state_method_id,
            "epoch_unit_id": value.epoch_unit_id,
            "expected_event_count": value.expected_event_count,
            "maximum_atan_series_iterations": value.maximum_atan_series_iterations,
            "certificate_precision": value.certificate_precision,
            "maximum_relative_off_plane": str(value.maximum_relative_off_plane),
            "maximum_root_iterations": value.maximum_root_iterations,
            "observer_id": value.observer_id,
            "regression_method_id": value.regression_method_id,
            "root_bracket_width_tolerance": str(value.root_bracket_width_tolerance),
            "root_method_id": value.root_method_id,
            "root_isolation_method_id": value.root_isolation_method_id,
            "root_tolerance_unit_id": value.root_tolerance_unit_id,
            "unwrap_method_id": value.unwrap_method_id,
        }
    )


def _dot(left: Vec3, right: Vec3) -> Decimal:
    return sum((left[index] * right[index] for index in range(3)), Decimal(0))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(vector: Vec3, factor: Decimal) -> Vec3:
    return tuple(item * factor for item in vector)  # type: ignore[return-value]


def _norm(vector: Vec3) -> Decimal:
    squared = _dot(vector, vector)
    if squared <= 0:
        raise ObserverDomainError("required direction is degenerate")
    return squared.sqrt()


def _radial(state: CartesianState) -> Decimal:
    return _dot(state.position, state.velocity)


def radial_product(state: CartesianState, configuration: ObserverConfiguration) -> Decimal:
    if type(state) is not CartesianState or type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("radial_product requires exact state and configuration types")
    try:
        with localcontext(configuration.context.build()):
            return _radial(state)
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal evaluating r dot v: {type(exc).__name__}"
        ) from exc


def _binomial(n: int, k: int) -> int:
    if not (0 <= k <= n <= 5):
        raise AssertionError("internal binomial index is invalid")
    k = min(k, n - k)
    result = 1
    for index in range(1, k + 1):
        result = result * (n - k + index) // index
    return result


def _certificate_context(configuration: ObserverConfiguration) -> Context:
    context = Context(
        prec=configuration.certificate_precision,
        rounding=ROUND_HALF_EVEN,
        Emin=configuration.context.emin,
        Emax=configuration.context.emax,
        capitals=configuration.context.capitals,
        clamp=configuration.context.clamp,
    )
    trapped = frozenset(_TRAPPED_SIGNALS)
    for signal in context.traps:
        context.traps[signal] = signal in trapped
    context.clear_flags()
    return context


def _bernstein_radial_coefficients(
    left: StateSample,
    right: StateSample,
    configuration: ObserverConfiguration,
) -> tuple[Decimal, ...]:
    """Exact degree-five Bernstein coefficients of the Hermite ``r dot v``.

    Let ``Q=(3r0, 3r0+h*v0, 3r1-h*v1, 3r1)`` and
    ``D_j=Q_(j+1)-Q_j``.  The Hermite position and velocity have controls
    ``Q/3`` and ``D/h``.  The omitted common factor ``1/(3h)`` is positive,
    so the product-form coefficients below have exactly the event function's
    signs.
    """

    if not left.epoch < right.epoch:
        raise ObserverContractError("certificate interval must be forward and nondegenerate")
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            duration = right.epoch - left.epoch
            three = Decimal(3)
            q0 = _scale(left.state.position, three)
            q1 = tuple(
                q0[index] + duration * left.state.velocity[index]
                for index in range(3)
            )
            q3 = _scale(right.state.position, three)
            q2 = tuple(
                q3[index] - duration * right.state.velocity[index]
                for index in range(3)
            )
            q = (q0, q1, q2, q3)
            d = tuple(
                tuple(q[index + 1][axis] - q[index][axis] for axis in range(3))
                for index in range(3)
            )
            coefficients: list[Decimal] = []
            for degree in range(6):
                total = Decimal(0)
                lower = max(0, degree - 2)
                upper = min(3, degree)
                for position_index in range(lower, upper + 1):
                    velocity_index = degree - position_index
                    numerator = _binomial(3, position_index) * _binomial(
                        2, velocity_index
                    )
                    weight = Decimal(numerator) / Decimal(_binomial(5, degree))
                    total += weight * _dot(q[position_index], d[velocity_index])
                coefficients.append(total)
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "Bernstein certificate arithmetic was not exact at frozen precision"
                )
            return tuple(coefficients)
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal constructing root certificate: {type(exc).__name__}"
        ) from exc


def _sign(value: Decimal) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _certify_hermite_intervals(
    series: TrajectorySeries,
    configuration: ObserverConfiguration,
) -> None:
    """Certify zero or exactly one simple root on every dense interval."""

    for index in range(1, len(series.samples)):
        left = series.samples[index - 1]
        right = series.samples[index]
        coefficients = _bernstein_radial_coefficients(left, right, configuration)
        signs = tuple(_sign(value) for value in coefficients)
        left_sign = signs[0]
        right_sign = signs[-1]
        p90_left_sign = _sign(radial_product(left.state, configuration))
        p90_right_sign = _sign(radial_product(right.state, configuration))
        if (p90_left_sign, p90_right_sign) != (left_sign, right_sign):
            raise ObserverArithmeticError(
                "p90 endpoint sign disagrees with the exact root certificate"
            )
        if left_sign == 0 and right_sign == 0:
            raise ObserverDomainError(
                "consecutive zero endpoints make the dense event roster ambiguous"
            )
        if left_sign == right_sign:
            if left_sign == 0 or any(sign != left_sign for sign in signs):
                raise ObserverDomainError(
                    "same-sign Hermite interval is not certified root-free"
                )
            continue
        if left_sign == 0:
            if any(sign != right_sign for sign in signs[1:]):
                raise ObserverDomainError(
                    "left-endpoint root is not isolated from internal roots"
                )
            continue
        if right_sign == 0:
            if any(sign != left_sign for sign in signs[:-1]):
                raise ObserverDomainError(
                    "right-endpoint root is not isolated from internal roots"
                )
            continue
        if any(sign == 0 for sign in signs):
            raise ObserverDomainError("root certificate contains an unresolved zero coefficient")
        variations = sum(
            signs[position] != signs[position - 1]
            for position in range(1, len(signs))
        )
        if variations != 1:
            raise ObserverDomainError(
                "sign-changing Hermite interval is not certified to one simple root"
            )


def _subdivide_bernstein_half(
    coefficients: tuple[Decimal, ...],
    configuration: ObserverConfiguration,
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...], Decimal]:
    """Exact de Casteljau subdivision at one half for a degree-five record."""

    if len(coefficients) != 6:
        raise AssertionError("internal Bernstein record must have degree five")
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            levels = [coefficients]
            while len(levels[-1]) > 1:
                prior = levels[-1]
                levels.append(
                    tuple(
                        (prior[index] + prior[index + 1]) / Decimal(2)
                        for index in range(len(prior) - 1)
                    )
                )
            left = tuple(level[0] for level in levels)
            right = tuple(level[-1] for level in reversed(levels))
            midpoint_value = levels[-1][0]
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "Bernstein subdivision was not exact at frozen precision"
                )
            return left, right, midpoint_value
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal subdividing root certificate: {type(exc).__name__}"
        ) from exc


def _exact_time_difference(
    left: Decimal,
    right: Decimal,
    configuration: ObserverConfiguration,
) -> Decimal:
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            width = right - left
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "time bracket width was not exact at certificate precision"
                )
            return width
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal constructing time width: {type(exc).__name__}"
        ) from exc


def _exact_epoch_at_unit_parameter(
    left_epoch: Decimal,
    duration: Decimal,
    unit_parameter: Decimal,
    configuration: ObserverConfiguration,
) -> Decimal:
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            epoch = left_epoch + duration * unit_parameter
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "event epoch was not exact at certificate precision"
                )
            return epoch
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal constructing event epoch: {type(exc).__name__}"
        ) from exc


def _exact_width_at_unit_interval(
    duration: Decimal,
    left_unit: Decimal,
    right_unit: Decimal,
    configuration: ObserverConfiguration,
) -> Decimal:
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            width = duration * (right_unit - left_unit)
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "unit-parameter bracket width was not exact"
                )
            return width
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal constructing unit bracket width: {type(exc).__name__}"
        ) from exc


def _cubic_hermite_state_at_unit_parameter(
    left: StateSample,
    right: StateSample,
    unit_parameter: Decimal,
    configuration: ObserverConfiguration,
) -> CartesianState:
    checked_u = _decimal(unit_parameter, "Hermite unit parameter")
    if not (0 <= checked_u <= 1):
        raise ObserverDomainError("Hermite unit parameter must lie in [0, 1]")
    if checked_u == 0:
        return left.state
    if checked_u == 1:
        return right.state
    exact_duration = _exact_time_difference(left.epoch, right.epoch, configuration)
    try:
        with localcontext(configuration.context.build()) as p90:
            duration = +exact_duration
            if p90.flags[Inexact]:
                raise ObserverArithmeticError(
                    "Hermite duration is not representable at p90"
                )
            p90.clear_flags()
            u = +checked_u
            # Deep bisection can need more than 90 digits to encode the exact
            # dyadic u.  State evaluation deliberately rounds u once to p90;
            # the caller must compare its r-dot-v sign with the exact
            # de Casteljau apex before retaining or choosing a half.
            p90.clear_flags()
            with localcontext(_certificate_context(configuration)) as exact:
                time_displacement = abs(u - checked_u) * exact_duration
                if exact.flags[Inexact] or exact.flags[Rounded]:
                    raise ObserverArithmeticError(
                        "p90 unit-parameter displacement was not exactly measurable"
                    )
                if time_displacement > configuration.root_bracket_width_tolerance:
                    raise ObserverArithmeticError(
                        "p90 unit-parameter rounding exceeds the root time tolerance"
                    )
            u2 = u * u
            u3 = u2 * u
            h00 = Decimal(2) * u3 - Decimal(3) * u2 + Decimal(1)
            h10 = u3 - Decimal(2) * u2 + u
            h01 = -Decimal(2) * u3 + Decimal(3) * u2
            h11 = u3 - u2
            dh00 = Decimal(6) * u2 - Decimal(6) * u
            dh10 = Decimal(3) * u2 - Decimal(4) * u + Decimal(1)
            dh01 = -Decimal(6) * u2 + Decimal(6) * u
            dh11 = Decimal(3) * u2 - Decimal(2) * u
            position = tuple(
                h00 * left.state.position[index]
                + h10 * duration * left.state.velocity[index]
                + h01 * right.state.position[index]
                + h11 * duration * right.state.velocity[index]
                for index in range(3)
            )
            velocity = tuple(
                (dh00 * left.state.position[index] + dh01 * right.state.position[index])
                / duration
                + dh10 * left.state.velocity[index]
                + dh11 * right.state.velocity[index]
                for index in range(3)
            )
            return CartesianState(position, velocity)  # type: ignore[arg-type]
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal evaluating cubic Hermite state: {type(exc).__name__}"
        ) from exc


def cubic_hermite_state(
    left: StateSample,
    right: StateSample,
    epoch: Decimal,
    configuration: ObserverConfiguration,
) -> CartesianState:
    """Evaluate endpoint-bound cubic Hermite position and its derivative."""

    if type(left) is not StateSample or type(right) is not StateSample:
        raise ObserverContractError("Hermite endpoints must be exactly StateSample")
    checked_epoch = _decimal(epoch, "Hermite epoch")
    if type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("configuration must be exactly ObserverConfiguration")
    if not (left.epoch < right.epoch) or not (left.epoch <= checked_epoch <= right.epoch):
        raise ObserverDomainError("Hermite epoch must lie in a nondegenerate endpoint interval")
    if checked_epoch == left.epoch:
        return left.state
    if checked_epoch == right.epoch:
        return right.state
    try:
        with localcontext(_certificate_context(configuration)) as exact:
            duration = right.epoch - left.epoch
            unit_parameter = (checked_epoch - left.epoch) / duration
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "public Hermite epoch does not define an exact Decimal unit parameter"
                )
        return _cubic_hermite_state_at_unit_parameter(
            left, right, unit_parameter, configuration
        )
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal evaluating cubic Hermite state: {type(exc).__name__}"
        ) from exc


def _atan_series(value: Decimal, tolerance: Decimal, maximum: int) -> Decimal:
    if not (0 <= value <= Decimal("0.125")):
        raise AssertionError("internal atan-series argument is outside [0, 0.125]")
    if value == 0:
        return Decimal(0)
    squared = value * value
    total = value
    term = value
    for index in range(maximum):
        next_term = -(
            term * squared * Decimal(2 * index + 1) / Decimal(2 * index + 3)
        )
        if abs(next_term) <= tolerance:
            return total
        updated = total + next_term
        if updated == total:
            raise ObserverArithmeticError("atan series stagnated before its error bound")
        total = updated
        term = next_term
    raise ObserverArithmeticError("atan series exceeded its frozen iteration bound")


def _atan_nonnegative(value: Decimal, tolerance: Decimal, maximum: int) -> Decimal:
    if value < 0:
        raise AssertionError("internal atan argument is negative")
    reduced = value
    doublings = 0
    # NIST DLMF 4.45.8, repeatedly, before DLMF 4.24.3's ascending series.
    while reduced > Decimal("0.125"):
        next_value = reduced / (Decimal(1) + (Decimal(1) + reduced * reduced).sqrt())
        if not (0 <= next_value < reduced):
            raise ObserverArithmeticError("atan argument reduction did not contract")
        reduced = next_value
        doublings += 1
        if doublings > 64:
            raise ObserverArithmeticError("atan argument reduction exceeded its bound")
    inner_tolerance = tolerance / (Decimal(2) ** doublings)
    return _atan_series(reduced, inner_tolerance, maximum) * (Decimal(2) ** doublings)


def _pi(tolerance: Decimal, maximum: int) -> Decimal:
    # Machin identity.  Dividing by 20 bounds the weighted 16+4 atan remainder.
    component_tolerance = tolerance / Decimal(20)
    return Decimal(16) * _atan_nonnegative(
        Decimal(1) / Decimal(5), component_tolerance, maximum
    ) - Decimal(4) * _atan_nonnegative(
        Decimal(1) / Decimal(239), component_tolerance, maximum
    )


def _angle_tolerance(configuration: ObserverConfiguration) -> Decimal:
    work_precision = configuration.context.precision + configuration.atan_guard_digits
    return Decimal(1).scaleb(-(work_precision - 4))


def _bounded_unit_ratio(
    smaller: Decimal, larger: Decimal, tolerance: Decimal
) -> Decimal:
    """Form ``smaller/larger`` without underflow outside the needed angle bound."""

    if not (0 <= smaller <= larger) or larger <= 0:
        raise AssertionError("internal atan2 ratio operands are unordered")
    if smaller == 0:
        return Decimal(0)
    # adjusted() is the base-ten exponent of the leading digit.  With the
    # two-decade margin below, replacing the ratio by zero contributes less
    # than the already-declared absolute angle tolerance.
    if smaller.adjusted() - larger.adjusted() <= tolerance.adjusted() - 2:
        return Decimal(0)
    return smaller / larger


def decimal_pi(configuration: ObserverConfiguration) -> Decimal:
    if type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("configuration must be exactly ObserverConfiguration")
    try:
        with localcontext(configuration.context.build(guard_digits=configuration.atan_guard_digits)):
            value = _pi(
                _angle_tolerance(configuration),
                configuration.maximum_atan_series_iterations,
            )
            if not value.is_finite() or value <= 3:
                raise ObserverArithmeticError("Decimal pi failed its domain check")
            with localcontext(configuration.context.build()):
                return +value
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal constructing pi: {type(exc).__name__}"
        ) from exc


def decimal_atan2(y: Decimal, x: Decimal, configuration: ObserverConfiguration) -> Decimal:
    """Return a Decimal principal angle in ``(-pi, pi]`` without float trig."""

    checked_y = _decimal(y, "atan2 y")
    checked_x = _decimal(x, "atan2 x")
    if type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("configuration must be exactly ObserverConfiguration")
    if checked_x == 0 and checked_y == 0:
        raise ObserverDomainError("atan2 is undefined at the origin")
    try:
        with localcontext(configuration.context.build(guard_digits=configuration.atan_guard_digits)):
            tolerance = _angle_tolerance(configuration)
            pi = _pi(tolerance / Decimal(2), configuration.maximum_atan_series_iterations)
            if checked_x == 0:
                result = pi / Decimal(2) if checked_y > 0 else -(pi / Decimal(2))
            elif checked_y == 0:
                result = Decimal(0) if checked_x > 0 else pi
            else:
                ax = abs(checked_x)
                ay = abs(checked_y)
                if ay <= ax:
                    acute = _atan_nonnegative(
                        _bounded_unit_ratio(ay, ax, tolerance),
                        tolerance / Decimal(2),
                        configuration.maximum_atan_series_iterations,
                    )
                else:
                    acute = pi / Decimal(2) - _atan_nonnegative(
                        _bounded_unit_ratio(ax, ay, tolerance),
                        tolerance / Decimal(2),
                        configuration.maximum_atan_series_iterations,
                    )
                if checked_x > 0 and checked_y > 0:
                    result = acute
                elif checked_x < 0 < checked_y:
                    result = pi - acute
                elif checked_x < 0 and checked_y < 0:
                    result = acute - pi
                else:
                    result = -acute
            with localcontext(configuration.context.build()):
                return +result
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal in atan2: {type(exc).__name__}"
        ) from exc


def _orientation_basis(
    state: CartesianState, configuration: ObserverConfiguration
) -> tuple[Vec3, Vec3, Vec3]:
    e1 = _scale(state.position, Decimal(1) / _norm(state.position))
    normal = _cross(state.position, state.velocity)
    n = _scale(normal, Decimal(1) / _norm(normal))
    e2_raw = _cross(n, e1)
    e2 = _scale(e2_raw, Decimal(1) / _norm(e2_raw))
    sanity = Decimal(1).scaleb(-(configuration.context.precision - 12))
    defects = (
        abs(_dot(e1, e2)),
        abs(_dot(e1, n)),
        abs(_dot(e2, n)),
        abs(_dot(e1, e1) - 1),
        abs(_dot(e2, e2) - 1),
        abs(_dot(n, n) - 1),
    )
    if max(defects) > sanity or _dot(_cross(e1, e2), n) <= 0:
        raise ObserverArithmeticError("frozen orientation basis failed its orthonormal check")
    return e1, e2, n


@dataclass(frozen=True)
class _AngleMeasurement:
    principal_angle: Decimal
    plane_normal_component: Decimal
    relative_off_plane: Decimal


def _measure_angle(
    position: Vec3,
    initial_state: CartesianState,
    configuration: ObserverConfiguration,
) -> _AngleMeasurement:
    with localcontext(configuration.context.build()):
        e1, e2, normal = _orientation_basis(initial_state, configuration)
        radius = _norm(position)
        normal_component = _dot(position, normal)
        relative = abs(normal_component) / radius
        if relative > configuration.maximum_relative_off_plane:
            raise ObserverDomainError("event position exceeds the frozen off-plane bound")
        x = _dot(position, e1)
        y = _dot(position, e2)
        if x == 0 and y == 0:
            raise ObserverDomainError("event position has zero orbital-plane projection")
    return _AngleMeasurement(
        decimal_atan2(y, x, configuration), normal_component, relative
    )


def oriented_position_angle(
    position: Vec3,
    initial_state: CartesianState,
    configuration: ObserverConfiguration,
) -> Decimal:
    checked = _vec3(position, "position")
    if type(initial_state) is not CartesianState:
        raise ObserverContractError("initial_state must be exactly CartesianState")
    if type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("configuration must be exactly ObserverConfiguration")
    try:
        return _measure_angle(checked, initial_state, configuration).principal_angle
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal measuring oriented angle: {type(exc).__name__}"
        ) from exc


def unwrap_oriented_angles(
    principal_angles: tuple[Decimal, ...], configuration: ObserverConfiguration
) -> tuple[Decimal, ...]:
    if type(principal_angles) is not tuple or not principal_angles:
        raise ObserverContractError("principal_angles must be a nonempty immutable tuple")
    checked = tuple(
        _decimal(value, f"principal_angles[{index}]")
        for index, value in enumerate(principal_angles)
    )
    pi = decimal_pi(configuration)
    if any(not (-pi < value <= pi) for value in checked):
        raise ObserverDomainError("principal angle lies outside (-pi, pi]")
    with localcontext(configuration.context.build()):
        two_pi = Decimal(2) * pi
        retained = [checked[0]]
        previous = checked[0]
        for value in checked[1:]:
            delta = value - previous
            while delta < -pi:
                delta += two_pi
            while delta > pi:
                delta -= two_pi
            if abs(delta) == pi:
                raise ObserverDomainError("exact pi unwrap tie has no unique nearest branch")
            retained.append(retained[-1] + delta)
            previous = value
        return tuple(retained)


@dataclass(frozen=True)
class _LocatedRoot:
    source_left_index: int
    source_right_index: int
    source_left_value: Decimal
    source_right_value: Decimal
    direction_confirmation_epoch: Decimal
    direction_confirmation_value: Decimal
    bracket_left_epoch: Decimal
    bracket_right_epoch: Decimal
    bracket_left_value: Decimal
    bracket_right_value: Decimal
    epoch: Decimal
    state: CartesianState
    event_value: Decimal
    iterations: int
    exact_root: bool


def _refine_root(
    series: TrajectorySeries,
    left_index: int,
    right_index: int,
    confirmation_index: int,
    configuration: ObserverConfiguration,
) -> _LocatedRoot:
    source_left = series.samples[left_index]
    source_right = series.samples[right_index]
    confirmation = series.samples[confirmation_index]
    source_left_value = _radial(source_left.state)
    source_right_value = _radial(source_right.state)
    confirmation_value = _radial(confirmation.state)
    if not (source_left_value < 0 <= source_right_value) or confirmation_value <= 0:
        raise ObserverDomainError("source interval is not a confirmed outward crossing")
    source_duration = _exact_time_difference(
        source_left.epoch, source_right.epoch, configuration
    )
    left_unit = Decimal(0)
    right_unit = Decimal(1)
    left_value = source_left_value
    right_value = source_right_value
    exact_coefficients = _bernstein_radial_coefficients(
        source_left, source_right, configuration
    )
    if (_sign(exact_coefficients[0]), _sign(exact_coefficients[-1])) != (
        _sign(source_left_value),
        _sign(source_right_value),
    ):
        raise ObserverArithmeticError(
            "p90 source bracket disagrees with the exact root certificate"
        )
    iterations = 0
    exact_root = right_value == 0
    if exact_root:
        left_unit = right_unit
        left_value = Decimal(0)
    while not exact_root and (
        _exact_width_at_unit_interval(
            source_duration, left_unit, right_unit, configuration
        )
        > configuration.root_bracket_width_tolerance
    ):
        if iterations >= configuration.maximum_root_iterations:
            raise ObserverArithmeticError("bisection exceeded its frozen iteration bound")
        with localcontext(_certificate_context(configuration)) as exact:
            midpoint_unit = (left_unit + right_unit) / Decimal(2)
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "unit-parameter midpoint was not exact at certificate precision"
                )
        if midpoint_unit == left_unit or midpoint_unit == right_unit:
            raise ObserverArithmeticError("bisection stagnated before width termination")
        state = _cubic_hermite_state_at_unit_parameter(
            source_left, source_right, midpoint_unit, configuration
        )
        value = _radial(state)
        exact_left, exact_right, exact_midpoint = _subdivide_bernstein_half(
            exact_coefficients, configuration
        )
        if _sign(value) != _sign(exact_midpoint):
            raise ObserverArithmeticError(
                "p90 midpoint sign disagrees with the exact root certificate"
            )
        iterations += 1
        if exact_midpoint == 0:
            left_unit = right_unit = midpoint_unit
            left_value = right_value = Decimal(0)
            exact_root = True
        elif exact_midpoint < 0:
            left_unit, left_value = midpoint_unit, value
            exact_coefficients = exact_right
        else:
            right_unit, right_value = midpoint_unit, value
            exact_coefficients = exact_left
    if exact_root:
        event_unit = right_unit
    else:
        with localcontext(_certificate_context(configuration)) as exact:
            event_unit = (left_unit + right_unit) / Decimal(2)
            if exact.flags[Inexact] or exact.flags[Rounded]:
                raise ObserverArithmeticError(
                    "reported unit parameter was not exact at certificate precision"
                )
    bracket_left_epoch = _exact_epoch_at_unit_parameter(
        source_left.epoch, source_duration, left_unit, configuration
    )
    bracket_right_epoch = _exact_epoch_at_unit_parameter(
        source_left.epoch, source_duration, right_unit, configuration
    )
    epoch = _exact_epoch_at_unit_parameter(
        source_left.epoch, source_duration, event_unit, configuration
    )
    state = _cubic_hermite_state_at_unit_parameter(
        source_left, source_right, event_unit, configuration
    )
    event_value = _radial(state)
    if not exact_root:
        _, _, exact_event_value = _subdivide_bernstein_half(
            exact_coefficients, configuration
        )
        if _sign(event_value) != _sign(exact_event_value):
            raise ObserverArithmeticError(
                "p90 reported-event sign disagrees with the exact root certificate"
            )
    return _LocatedRoot(
        left_index,
        right_index,
        source_left_value,
        source_right_value,
        confirmation.epoch,
        confirmation_value,
        bracket_left_epoch,
        bracket_right_epoch,
        left_value,
        right_value,
        epoch,
        state,
        event_value,
        iterations,
        exact_root,
    )


def _locate_roots(
    series: TrajectorySeries, configuration: ObserverConfiguration
) -> tuple[_LocatedRoot, ...]:
    _certify_hermite_intervals(series, configuration)
    values = tuple(_radial(item.state) for item in series.samples)
    if values[0] != 0:
        raise ObserverDomainError("initial checkpoint must be the exact perihelion event g=0")
    for index in range(1, len(values)):
        if values[index] != 0:
            continue
        if index + 1 >= len(values):
            raise ObserverDomainError("terminal endpoint root has no direction confirmation")
        left_sign = _sign(values[index - 1])
        right_sign = _sign(values[index + 1])
        if left_sign == 0 or right_sign == 0 or left_sign == right_sign:
            raise ObserverDomainError(
                "interior endpoint zero is a plateau, tangent, or unresolved root"
            )
    roots: list[_LocatedRoot] = []
    phase = "AWAIT_OUTWARD"
    for index in range(1, len(values)):
        value = values[index]
        if phase == "AWAIT_OUTWARD":
            if value > 0:
                phase = "AWAIT_INWARD"
            elif value < 0:
                raise ObserverDomainError("inward motion before initial-event arming")
            continue
        if phase == "AWAIT_INWARD":
            if value < 0:
                phase = "SEARCH_PERIHELION"
            continue
        previous = values[index - 1]
        if previous < 0 < value:
            roots.append(_refine_root(series, index - 1, index, index, configuration))
            phase = "AWAIT_INWARD"
        elif previous < 0 and value == 0:
            if index + 1 >= len(values) or values[index + 1] <= 0:
                continue
            roots.append(
                _refine_root(series, index - 1, index, index + 1, configuration)
            )
            phase = "AWAIT_INWARD"
    if len(roots) != configuration.expected_event_count:
        raise ObserverDomainError(
            "full series must contain exactly the frozen two postinitial perihelia"
        )
    return tuple(roots)


@dataclass(frozen=True)
class PerihelionEvent:
    orbit_index: int
    source_left_index: int
    source_right_index: int
    source_left_value: Decimal
    source_right_value: Decimal
    direction_confirmation_epoch: Decimal
    direction_confirmation_value: Decimal
    bracket_left_epoch: Decimal
    bracket_right_epoch: Decimal
    bracket_left_value: Decimal
    bracket_right_value: Decimal
    epoch: Decimal
    state: CartesianState
    event_value: Decimal
    bisection_iterations: int
    exact_root: bool
    principal_angle: Decimal
    unwrapped_angle: Decimal
    plane_normal_component: Decimal
    relative_off_plane: Decimal

    def __post_init__(self) -> None:
        integer_fields = (
            self.orbit_index,
            self.source_left_index,
            self.source_right_index,
            self.bisection_iterations,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ObserverContractError("event integer fields must be nonnegative integers")
        if self.orbit_index < 1 or self.source_right_index != self.source_left_index + 1:
            raise ObserverContractError("event index roster is invalid")
        for name in (
            "source_left_value",
            "source_right_value",
            "direction_confirmation_epoch",
            "direction_confirmation_value",
            "bracket_left_epoch",
            "bracket_right_epoch",
            "bracket_left_value",
            "bracket_right_value",
            "epoch",
            "event_value",
            "principal_angle",
            "unwrapped_angle",
            "plane_normal_component",
            "relative_off_plane",
        ):
            _decimal(getattr(self, name), name)
        if type(self.state) is not CartesianState:
            raise ObserverContractError("event state must be exactly CartesianState")
        if type(self.exact_root) is not bool:
            raise ObserverContractError("exact_root must be bool")


def _event_record(event: PerihelionEvent) -> dict[str, object]:
    return {
        "orbit_index": event.orbit_index,
        "source_left_index": event.source_left_index,
        "source_right_index": event.source_right_index,
        "source_left_value": str(event.source_left_value),
        "source_right_value": str(event.source_right_value),
        "direction_confirmation_epoch": str(event.direction_confirmation_epoch),
        "direction_confirmation_value": str(event.direction_confirmation_value),
        "bracket_left_epoch": str(event.bracket_left_epoch),
        "bracket_right_epoch": str(event.bracket_right_epoch),
        "bracket_left_value": str(event.bracket_left_value),
        "bracket_right_value": str(event.bracket_right_value),
        "epoch": str(event.epoch),
        "position": [str(value) for value in event.state.position],
        "velocity": [str(value) for value in event.state.velocity],
        "event_value": str(event.event_value),
        "bisection_iterations": event.bisection_iterations,
        "exact_root": event.exact_root,
        "principal_angle": str(event.principal_angle),
        "unwrapped_angle": str(event.unwrapped_angle),
        "plane_normal_component": str(event.plane_normal_component),
        "relative_off_plane": str(event.relative_off_plane),
    }


def _build_events(
    series: TrajectorySeries, configuration: ObserverConfiguration
) -> tuple[PerihelionEvent, ...]:
    roots = _locate_roots(series, configuration)
    measurements = tuple(
        _measure_angle(root.state.position, series.initial_input.initial_state, configuration)
        for root in roots
    )
    unwrapped = unwrap_oriented_angles(
        tuple(item.principal_angle for item in measurements), configuration
    )
    return tuple(
        PerihelionEvent(
            orbit_index=index + 1,
            source_left_index=root.source_left_index,
            source_right_index=root.source_right_index,
            source_left_value=root.source_left_value,
            source_right_value=root.source_right_value,
            direction_confirmation_epoch=root.direction_confirmation_epoch,
            direction_confirmation_value=root.direction_confirmation_value,
            bracket_left_epoch=root.bracket_left_epoch,
            bracket_right_epoch=root.bracket_right_epoch,
            bracket_left_value=root.bracket_left_value,
            bracket_right_value=root.bracket_right_value,
            epoch=root.epoch,
            state=root.state,
            event_value=root.event_value,
            bisection_iterations=root.iterations,
            exact_root=root.exact_root,
            principal_angle=measurements[index].principal_angle,
            unwrapped_angle=unwrapped[index],
            plane_normal_component=measurements[index].plane_normal_component,
            relative_off_plane=measurements[index].relative_off_plane,
        )
        for index, root in enumerate(roots)
    )


def _slope(
    events: tuple[PerihelionEvent, ...], configuration: ObserverConfiguration
) -> Decimal:
    if len(events) != 2 or tuple(item.orbit_index for item in events) != (1, 2):
        raise ObserverContractError("the frozen OLS roster is exactly orbit indices (1, 2)")
    with localcontext(configuration.context.build()):
        # Exact algebraic reduction of unweighted OLS with a free intercept at
        # k=(1,2): slope = theta_2 - theta_1.  Avoiding mean/divide round trips
        # preserves the uniquely defined p90 result.
        return events[1].unwrapped_angle - events[0].unwrapped_angle


@dataclass(frozen=True)
class TrajectoryObservation:
    series: TrajectorySeries
    trajectory_sha256: str
    sampling_schedule_sha256: str
    initial_input_sha256: str
    observer_configuration: ObserverConfiguration
    observer_configuration_sha256: str
    events: tuple[PerihelionEvent, ...]
    secular_advance_per_orbit: Decimal
    metadata: ObserverMetadata = ObserverMetadata()

    def __post_init__(self) -> None:
        if type(self.series) is not TrajectorySeries:
            raise ObserverContractError("series must be exactly TrajectorySeries")
        _sha256(self.trajectory_sha256, "trajectory_sha256")
        _sha256(self.sampling_schedule_sha256, "sampling_schedule_sha256")
        _sha256(self.initial_input_sha256, "initial_input_sha256")
        if type(self.observer_configuration) is not ObserverConfiguration:
            raise ObserverContractError("observer_configuration has the wrong type")
        _sha256(self.observer_configuration_sha256, "observer_configuration_sha256")
        if type(self.events) is not tuple:
            raise ObserverContractError("events must be an immutable tuple")
        _decimal(self.secular_advance_per_orbit, "secular_advance_per_orbit")
        if type(self.metadata) is not ObserverMetadata:
            raise ObserverContractError("metadata must be exactly ObserverMetadata")
        try:
            with localcontext(self.observer_configuration.context.build()):
                if self.trajectory_sha256 != trajectory_series_sha256(self.series):
                    raise ObserverContractError("trajectory hash does not bind the retained series")
                if self.sampling_schedule_sha256 != sampling_schedule_sha256(self.series):
                    raise ObserverContractError("schedule hash does not bind the retained series")
                if self.initial_input_sha256 != initial_input_sha256(self.series.initial_input):
                    raise ObserverContractError("initial-input hash does not bind the retained input")
                if self.observer_configuration_sha256 != observer_configuration_sha256(
                    self.observer_configuration
                ):
                    raise ObserverContractError(
                        "observer hash does not bind the retained configuration"
                    )
                expected_events = _build_events(self.series, self.observer_configuration)
                if tuple(_event_record(item) for item in self.events) != tuple(
                    _event_record(item) for item in expected_events
                ):
                    raise ObserverContractError("events do not reproduce from the retained series")
                if str(self.secular_advance_per_orbit) != str(
                    _slope(expected_events, self.observer_configuration)
                ):
                    raise ObserverContractError("secular slope does not reproduce")
        except DecimalException as exc:
            raise ObserverArithmeticError(
                f"trapped Decimal signal validating observation: {type(exc).__name__}"
            ) from exc


def observe_trajectory(
    series: TrajectorySeries, configuration: ObserverConfiguration
) -> TrajectoryObservation:
    """Observe exactly two postinitial perihelia from the complete bound series."""

    if type(series) is not TrajectorySeries or type(configuration) is not ObserverConfiguration:
        raise ObserverContractError("observe_trajectory requires exact series/configuration types")
    try:
        with localcontext(configuration.context.build()):
            events = _build_events(series, configuration)
            slope = _slope(events, configuration)
            return TrajectoryObservation(
                series=series,
                trajectory_sha256=trajectory_series_sha256(series),
                sampling_schedule_sha256=sampling_schedule_sha256(series),
                initial_input_sha256=initial_input_sha256(series.initial_input),
                observer_configuration=configuration,
                observer_configuration_sha256=observer_configuration_sha256(configuration),
                events=events,
                secular_advance_per_orbit=slope,
            )
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal observing trajectory: {type(exc).__name__}"
        ) from exc


@dataclass(frozen=True)
class PairedPerihelionObservation:
    newtonian: TrajectoryObservation
    one_pn: TrajectoryObservation
    paired_advance_per_orbit: Decimal
    prograde: bool
    sampling_schedule_sha256: str
    initial_input_sha256: str
    observer_configuration_sha256: str
    metadata: ObserverMetadata = ObserverMetadata()

    def __post_init__(self) -> None:
        if type(self.newtonian) is not TrajectoryObservation or type(
            self.one_pn
        ) is not TrajectoryObservation:
            raise ObserverContractError("paired arms must be exact TrajectoryObservation values")
        if self.newtonian.series.force_mode is not ForceMode.NEWTONIAN:
            raise ObserverContractError("newtonian arm force mode is wrong")
        if self.one_pn.series.force_mode is not ForceMode.NEWTONIAN_PLUS_SOLAR_1PN:
            raise ObserverContractError("one_pn arm force mode is wrong")
        if self.newtonian.series.initial_input != self.one_pn.series.initial_input:
            raise ObserverContractError("paired initial inputs differ")
        if (
            self.newtonian.initial_input_sha256 != self.one_pn.initial_input_sha256
            or self.newtonian.observer_configuration != self.one_pn.observer_configuration
            or self.newtonian.observer_configuration_sha256
            != self.one_pn.observer_configuration_sha256
        ):
            raise ObserverContractError("paired observer configurations differ")
        if (
            self.newtonian.series.source_configuration_id
            != self.one_pn.series.source_configuration_id
            or self.newtonian.series.source_configuration_sha256
            != self.one_pn.series.source_configuration_sha256
        ):
            raise ObserverContractError("paired numerical source configurations differ")
        if (
            self.newtonian.series.schedule_pair_id != self.one_pn.series.schedule_pair_id
            or self.newtonian.sampling_schedule_sha256
            != self.one_pn.sampling_schedule_sha256
        ):
            raise ObserverContractError("paired checkpoint schedules differ")
        if tuple(item.orbit_index for item in self.newtonian.events) != tuple(
            item.orbit_index for item in self.one_pn.events
        ):
            raise ObserverContractError("paired orbit-index rosters differ")
        expected_bindings = (
            self.newtonian.sampling_schedule_sha256,
            self.newtonian.initial_input_sha256,
            self.newtonian.observer_configuration_sha256,
        )
        actual_bindings = (
            self.sampling_schedule_sha256,
            self.initial_input_sha256,
            self.observer_configuration_sha256,
        )
        if actual_bindings != expected_bindings:
            raise ObserverContractError("paired identity bindings are inconsistent")
        _decimal(self.paired_advance_per_orbit, "paired_advance_per_orbit")
        configuration = self.newtonian.observer_configuration
        with localcontext(configuration.context.build()):
            expected = (
                self.one_pn.secular_advance_per_orbit
                - self.newtonian.secular_advance_per_orbit
            )
            if str(self.paired_advance_per_orbit) != str(expected):
                raise ObserverContractError("paired subtraction must be 1PN minus Newtonian")
        if type(self.prograde) is not bool or self.prograde is not (expected > 0):
            raise ObserverContractError("prograde must retain the paired signed result")
        if type(self.metadata) is not ObserverMetadata:
            raise ObserverContractError("metadata must be exactly ObserverMetadata")


def pair_perihelion_observations(
    newtonian: TrajectoryObservation, one_pn: TrajectoryObservation
) -> PairedPerihelionObservation:
    if type(newtonian) is not TrajectoryObservation or type(
        one_pn
    ) is not TrajectoryObservation:
        raise ObserverContractError("pairing requires exact TrajectoryObservation values")
    configuration = newtonian.observer_configuration
    try:
        with localcontext(configuration.context.build()):
            paired = one_pn.secular_advance_per_orbit - newtonian.secular_advance_per_orbit
        return PairedPerihelionObservation(
            newtonian=newtonian,
            one_pn=one_pn,
            paired_advance_per_orbit=paired,
            prograde=paired > 0,
            sampling_schedule_sha256=newtonian.sampling_schedule_sha256,
            initial_input_sha256=newtonian.initial_input_sha256,
            observer_configuration_sha256=newtonian.observer_configuration_sha256,
        )
    except DecimalException as exc:
        raise ObserverArithmeticError(
            f"trapped Decimal signal pairing observations: {type(exc).__name__}"
        ) from exc


__all__ = (
    "ANGLE_METHOD_ID",
    "DENSE_STATE_METHOD_ID",
    "EXECUTION_AUTHORIZED",
    "OBSERVER_ID",
    "OUTPUT_CLASS",
    "QUALIFICATION_OUTCOMES_GENERATED",
    "REGISTRY_AUTHORIZED",
    "REGRESSION_METHOD_ID",
    "REVIEW_STATUS",
    "ROOT_ISOLATION_METHOD_ID",
    "ROOT_METHOD_ID",
    "UNWRAP_METHOD_ID",
    "CartesianState",
    "DecimalContextSpec",
    "ForceMode",
    "InitialInput",
    "ObserverArithmeticError",
    "ObserverConfiguration",
    "ObserverContractError",
    "ObserverDomainError",
    "ObserverMetadata",
    "PairedPerihelionObservation",
    "PerihelionEvent",
    "StateSample",
    "TrajectoryObservation",
    "TrajectorySeries",
    "cubic_hermite_state",
    "decimal_atan2",
    "decimal_pi",
    "initial_input_sha256",
    "observe_trajectory",
    "observer_configuration_sha256",
    "oriented_position_angle",
    "pair_perihelion_observations",
    "radial_product",
    "sampling_schedule_sha256",
    "trajectory_series_sha256",
    "unwrap_oriented_angles",
)
