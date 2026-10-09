"""Fixed-step Decimal implicit-midpoint reference integration for JX V5.

The implicit midpoint rule advances a general first-order system with

    y1 = y0 + h f(t0 + h/2, (y0 + y1)/2).

It is appropriate for the velocity-dependent restricted Solar 1PN reference
equation.  This module makes no symplecticity claim for the noncanonical
``(position, coordinate velocity)`` state and grants no registry authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, Inexact, localcontext
import re
from typing import Callable

from .provenance import sha256_data
from .v5_reference_dynamics import (
    DecimalContextSpec,
    REFERENCE_EVIDENCE_CLASS,
    REFERENCE_FORCE_MODEL_IDS,
    ReferenceDynamicsError,
    ReferenceForcePlan,
    ReferencePhaseState,
    bind_reference_state,
    evaluate_reference_rhs,
    validate_reference_state_binding,
)


IMPLICIT_MIDPOINT_METHOD_ID = "jx.v5.reference.implicit_midpoint.fixed_step_decimal.v1"
INITIAL_GUESS_POLICY = "EXPLICIT_EULER_FROM_FRESH_RHS"

SixVector = tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
SixVectorRHS = Callable[[Decimal, SixVector], SixVector]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReferenceIntegrationError(ValueError):
    """The reference integration contract or result was invalid."""


class NonlinearSolveError(ArithmeticError):
    """The implicit midpoint equation did not satisfy its residual gate."""


def _require_decimal(value: object, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise ReferenceIntegrationError(f"{label} must be decimal.Decimal")
    if not value.is_finite():
        raise ReferenceIntegrationError(f"{label} must be finite")
    return value


def _require_six(value: object, label: str) -> SixVector:
    if type(value) is not tuple or len(value) != 6:
        raise ReferenceIntegrationError(f"{label} must be an immutable 6-tuple")
    checked = tuple(
        _require_decimal(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return checked  # type: ignore[return-value]


@dataclass(frozen=True)
class ImplicitMidpointSpec:
    """Frozen fixed-step and nonlinear-solver policy."""

    step_size: Decimal
    step_count: int
    position_atol: Decimal
    velocity_atol: Decimal
    relative_tolerance: Decimal
    maximum_iterations: int
    initial_guess_policy: str = INITIAL_GUESS_POLICY
    method_id: str = IMPLICIT_MIDPOINT_METHOD_ID
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        step = _require_decimal(self.step_size, "step_size")
        if step == 0:
            raise ReferenceIntegrationError("step_size must be nonzero")
        if type(self.step_count) is not int or not (1 <= self.step_count <= 1_000_000):
            raise ReferenceIntegrationError("step_count must be an integer in [1, 1000000]")
        position_atol = _require_decimal(self.position_atol, "position_atol")
        velocity_atol = _require_decimal(self.velocity_atol, "velocity_atol")
        relative = _require_decimal(self.relative_tolerance, "relative_tolerance")
        if position_atol <= 0 or velocity_atol <= 0:
            raise ReferenceIntegrationError("absolute tolerances must be positive")
        if relative < 0:
            raise ReferenceIntegrationError("relative_tolerance must be nonnegative")
        if type(self.maximum_iterations) is not int or not (1 <= self.maximum_iterations <= 10000):
            raise ReferenceIntegrationError("maximum_iterations must be an integer in [1, 10000]")
        if self.initial_guess_policy != INITIAL_GUESS_POLICY:
            raise ReferenceIntegrationError("unsupported implicit initial-guess policy")
        if self.method_id != IMPLICIT_MIDPOINT_METHOD_ID:
            raise ReferenceIntegrationError("unexpected integration method identifier")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceIntegrationError("the reference integrator cannot authorize the registry")

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "jx-v5-implicit-midpoint-spec/v1",
            "method_id": self.method_id,
            "step_size": str(self.step_size),
            "step_count": self.step_count,
            "position_atol": str(self.position_atol),
            "velocity_atol": str(self.velocity_atol),
            "relative_tolerance": str(self.relative_tolerance),
            "maximum_iterations": self.maximum_iterations,
            "initial_guess_policy": self.initial_guess_policy,
            "order": 2,
            "adaptive": False,
            "state_coordinates": "POSITION_AND_COORDINATE_VELOCITY_NONCANONICAL",
            "symplectic_claim": False,
            "registry_authorized": self.registry_authorized,
        }

    @property
    def sha256(self) -> str:
        return sha256_data(self.as_record())


@dataclass(frozen=True)
class ImplicitMidpointStepDiagnostics:
    step_index: int
    start_epoch: Decimal
    end_epoch: Decimal
    iterations: int
    rhs_evaluations: int
    scaled_residual: Decimal
    method_id: str
    spec_sha256: str
    applied_model_ids: tuple[str, ...]
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ReferenceIntegrationError("step_index must be a nonnegative integer")
        _require_decimal(self.start_epoch, "start_epoch")
        _require_decimal(self.end_epoch, "end_epoch")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ReferenceIntegrationError("iterations must be positive")
        if (
            type(self.rhs_evaluations) is not int
            or self.rhs_evaluations != 1 + 2 * self.iterations
        ):
            raise ReferenceIntegrationError("rhs_evaluations is inconsistent")
        residual = _require_decimal(self.scaled_residual, "scaled_residual")
        if residual < 0 or residual > 1:
            raise ReferenceIntegrationError("accepted scaled_residual must lie in [0, 1]")
        if self.method_id != IMPLICIT_MIDPOINT_METHOD_ID:
            raise ReferenceIntegrationError("diagnostic method identifier differs")
        if type(self.spec_sha256) is not str or _SHA256.fullmatch(self.spec_sha256) is None:
            raise ReferenceIntegrationError("diagnostic spec digest is invalid")
        if self.applied_model_ids != REFERENCE_FORCE_MODEL_IDS:
            raise ReferenceIntegrationError("diagnostic force ledger differs")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceIntegrationError("step diagnostics cannot authorize a registry")


@dataclass(frozen=True)
class ReferenceTrajectoryResult:
    method_id: str
    states: tuple[ReferencePhaseState, ...]
    diagnostics: tuple[ImplicitMidpointStepDiagnostics, ...]
    force_plan_sha256: str
    decimal_context_sha256: str
    spec_sha256: str
    evidence_class: str = REFERENCE_EVIDENCE_CLASS
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        if self.method_id != IMPLICIT_MIDPOINT_METHOD_ID:
            raise ReferenceIntegrationError("trajectory method identifier differs")
        if type(self.states) is not tuple or len(self.states) < 2:
            raise ReferenceIntegrationError("trajectory must retain its initial and final states")
        if type(self.diagnostics) is not tuple or len(self.diagnostics) != len(self.states) - 1:
            raise ReferenceIntegrationError("trajectory diagnostics do not close over the states")
        if any(type(state) is not ReferencePhaseState for state in self.states):
            raise ReferenceIntegrationError("trajectory contains a non-reference state")
        if any(
            type(diagnostic) is not ImplicitMidpointStepDiagnostics
            for diagnostic in self.diagnostics
        ):
            raise ReferenceIntegrationError("trajectory contains invalid step diagnostics")
        if self.states[-1].epoch != self.diagnostics[-1].end_epoch:
            raise ReferenceIntegrationError("trajectory final epoch differs from diagnostics")
        for digest in (
            self.force_plan_sha256,
            self.decimal_context_sha256,
            self.spec_sha256,
        ):
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                raise ReferenceIntegrationError("trajectory digest is invalid")
        for index, state in enumerate(self.states):
            if (
                state.force_plan_sha256 != self.force_plan_sha256
                or state.decimal_context_sha256 != self.decimal_context_sha256
                or state.registry_sha256 != self.states[0].registry_sha256
                or state.central_source != self.states[0].central_source
                or state.target != self.states[0].target
                or state.units is not self.states[0].units
                or state.frame is not self.states[0].frame
                or state.time_scale is not self.states[0].time_scale
                or state.target_treatment is not self.states[0].target_treatment
                or state.evidence_class != REFERENCE_EVIDENCE_CLASS
                or state.registry_authorized
            ):
                raise ReferenceIntegrationError("trajectory state bindings differ")
            if index:
                diagnostic = self.diagnostics[index - 1]
                if (
                    diagnostic.step_index != index - 1
                    or diagnostic.start_epoch != self.states[index - 1].epoch
                    or diagnostic.end_epoch != state.epoch
                    or diagnostic.spec_sha256 != self.spec_sha256
                    or diagnostic.method_id != self.method_id
                    or diagnostic.applied_model_ids != REFERENCE_FORCE_MODEL_IDS
                    or diagnostic.registry_authorized
                ):
                    raise ReferenceIntegrationError("trajectory epoch chain is not contiguous")
        if self.evidence_class != REFERENCE_EVIDENCE_CLASS:
            raise ReferenceIntegrationError("trajectory evidence must remain MODEL_OUTPUT")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceIntegrationError("a reference trajectory cannot authorize the registry")

    @property
    def final_state(self) -> ReferencePhaseState:
        return self.states[-1]


def _scaled_midpoint_residual(
    initial: SixVector,
    candidate: SixVector,
    step_size: Decimal,
    midpoint_rhs: SixVector,
    position_atol: Decimal,
    velocity_atol: Decimal,
    relative_tolerance: Decimal,
) -> Decimal:
    ratios: list[Decimal] = []
    for index in range(6):
        residual = candidate[index] - initial[index] - step_size * midpoint_rhs[index]
        absolute_tolerance = position_atol if index < 3 else velocity_atol
        scale = max(
            abs(initial[index]),
            abs(candidate[index]),
            abs(step_size * midpoint_rhs[index]),
        )
        denominator = absolute_tolerance + relative_tolerance * scale
        if not denominator.is_finite() or denominator <= 0:
            raise ReferenceIntegrationError("residual denominator must be finite and positive")
        ratio = abs(residual) / denominator
        if not ratio.is_finite():
            raise ReferenceIntegrationError("scaled nonlinear residual must be finite")
        ratios.append(ratio)
    return max(ratios)


def _exact_epoch_offset(epoch: Decimal, step_size: Decimal, numerator: int, denominator: int) -> Decimal:
    """Return one supported offset only when its fused evaluation is exact.

    Inspecting the Decimal context's ``Inexact`` flag avoids converting values
    with implementation-limit exponents into enormous integer ratios.
    """

    epoch_value = _require_decimal(epoch, "epoch")
    step = _require_decimal(step_size, "step_size")
    if step == 0:
        raise ReferenceIntegrationError("step_size must be nonzero")
    if numerator != 1 or denominator not in (1, 2):
        raise ReferenceIntegrationError("unsupported coordinate-time offset")
    multiplier = Decimal(1) if denominator == 1 else Decimal("0.5")
    try:
        with localcontext() as offset_context:
            offset_context.clear_flags()
            computed = step.fma(multiplier, epoch_value)
            inexact = offset_context.flags[Inexact]
    except DecimalException as exc:
        raise ReferenceIntegrationError(
            "Decimal arithmetic failed while computing the coordinate-time offset"
        ) from exc
    wrong_direction = (step > 0 and computed <= epoch_value) or (
        step < 0 and computed >= epoch_value
    )
    if inexact or not computed.is_finite() or wrong_direction:
        raise ReferenceIntegrationError(
            "Decimal context cannot represent the requested coordinate-time offset exactly"
        )
    return computed


def _solve_implicit_midpoint_six(
    *,
    initial_epoch: Decimal,
    initial: SixVector,
    step_size: Decimal,
    position_atol: Decimal,
    velocity_atol: Decimal,
    relative_tolerance: Decimal,
    maximum_iterations: int,
    rhs: SixVectorRHS,
) -> tuple[SixVector, int, int, Decimal]:
    """Solve one six-dimensional midpoint equation with a strict residual gate.

    This small generic core is kept separate so the numerical method can be
    tested against analytic first-order systems without sharing the Solar RHS.
    """

    epoch = _require_decimal(initial_epoch, "initial_epoch")
    y0 = _require_six(initial, "initial")
    h = _require_decimal(step_size, "step_size")
    if h == 0:
        raise ReferenceIntegrationError("step_size must be nonzero")
    p_atol = _require_decimal(position_atol, "position_atol")
    v_atol = _require_decimal(velocity_atol, "velocity_atol")
    rtol = _require_decimal(relative_tolerance, "relative_tolerance")
    if p_atol <= 0 or v_atol <= 0 or rtol < 0:
        raise ReferenceIntegrationError("invalid nonlinear residual tolerances")
    if type(maximum_iterations) is not int or maximum_iterations < 1:
        raise ReferenceIntegrationError("maximum_iterations must be positive")
    if not callable(rhs):
        raise ReferenceIntegrationError("rhs must be callable")

    try:
        half = Decimal(1) / Decimal(2)
        midpoint_epoch = _exact_epoch_offset(epoch, h, 1, 2)
        _exact_epoch_offset(epoch, h, 1, 1)
        initial_rhs = _require_six(rhs(epoch, y0), "initial RHS")
        rhs_evaluations = 1
        guess = _require_six(
            tuple(y0[index] + h * initial_rhs[index] for index in range(6)),
            "explicit-Euler initial guess",
        )
        last_residual: Decimal | None = None

        for iteration in range(1, maximum_iterations + 1):
            midpoint = _require_six(
                tuple((y0[index] + guess[index]) * half for index in range(6)),
                "iteration midpoint",
            )
            midpoint_rhs = _require_six(rhs(midpoint_epoch, midpoint), "iteration RHS")
            rhs_evaluations += 1
            candidate = _require_six(
                tuple(y0[index] + h * midpoint_rhs[index] for index in range(6)),
                "implicit candidate",
            )

            # Acceptance always uses a new RHS call and recomputes the full
            # implicit equation from the candidate, never an iteration delta.
            verification_midpoint = _require_six(
                tuple((y0[index] + candidate[index]) * half for index in range(6)),
                "verification midpoint",
            )
            verification_rhs = _require_six(
                rhs(midpoint_epoch, verification_midpoint),
                "verification RHS",
            )
            rhs_evaluations += 1
            last_residual = _scaled_midpoint_residual(
                y0,
                candidate,
                h,
                verification_rhs,
                p_atol,
                v_atol,
                rtol,
            )
            if last_residual <= 1:
                return candidate, iteration, rhs_evaluations, last_residual
            guess = candidate
    except DecimalException as exc:
        raise ReferenceIntegrationError("Decimal arithmetic failed in the midpoint solve") from exc

    residual_text = "unavailable" if last_residual is None else str(last_residual)
    raise NonlinearSolveError(
        f"implicit midpoint failed after {maximum_iterations} iterations; "
        f"scaled residual={residual_text}"
    )


def _state_vector(state: ReferencePhaseState) -> SixVector:
    return state.position + state.velocity


def implicit_midpoint_step(
    state: ReferencePhaseState,
    plan: ReferenceForcePlan,
    arithmetic: DecimalContextSpec,
    spec: ImplicitMidpointSpec,
    *,
    step_index: int = 0,
) -> tuple[ReferencePhaseState, ImplicitMidpointStepDiagnostics]:
    """Advance one immutable JX reference state or raise without a result."""

    if type(state) is not ReferencePhaseState:
        raise ReferenceIntegrationError("state must be ReferencePhaseState")
    if type(plan) is not ReferenceForcePlan:
        raise ReferenceIntegrationError("plan must be ReferenceForcePlan")
    if type(arithmetic) is not DecimalContextSpec:
        raise ReferenceIntegrationError("arithmetic must be DecimalContextSpec")
    if type(spec) is not ImplicitMidpointSpec:
        raise ReferenceIntegrationError("spec must be ImplicitMidpointSpec")
    if type(step_index) is not int or step_index < 0:
        raise ReferenceIntegrationError("step_index must be a nonnegative integer")
    try:
        validate_reference_state_binding(state, plan, arithmetic)
    except ReferenceDynamicsError as exc:
        raise ReferenceIntegrationError(
            "state bindings differ from the integration contracts"
        ) from exc

    with localcontext(arithmetic.make_context()):
        def rhs(epoch: Decimal, vector: SixVector) -> SixVector:
            substage = bind_reference_state(
                epoch=epoch,
                target=state.target,
                position=vector[:3],  # type: ignore[arg-type]
                velocity=vector[3:],  # type: ignore[arg-type]
                plan=plan,
                arithmetic=arithmetic,
            )
            evaluation = evaluate_reference_rhs(substage, plan, arithmetic)
            if evaluation.applied_model_ids != REFERENCE_FORCE_MODEL_IDS:
                raise ReferenceIntegrationError("substage force ledger did not close")
            return evaluation.position_rate + evaluation.velocity_rate

        candidate, iterations, rhs_calls, residual = _solve_implicit_midpoint_six(
            initial_epoch=state.epoch,
            initial=_state_vector(state),
            step_size=spec.step_size,
            position_atol=spec.position_atol,
            velocity_atol=spec.velocity_atol,
            relative_tolerance=spec.relative_tolerance,
            maximum_iterations=spec.maximum_iterations,
            rhs=rhs,
        )
        end_epoch = _exact_epoch_offset(state.epoch, spec.step_size, 1, 1)
        final_state = bind_reference_state(
            epoch=end_epoch,
            target=state.target,
            position=candidate[:3],  # type: ignore[arg-type]
            velocity=candidate[3:],  # type: ignore[arg-type]
            plan=plan,
            arithmetic=arithmetic,
        )

    diagnostics = ImplicitMidpointStepDiagnostics(
        step_index=step_index,
        start_epoch=state.epoch,
        end_epoch=final_state.epoch,
        iterations=iterations,
        rhs_evaluations=rhs_calls,
        scaled_residual=residual,
        method_id=spec.method_id,
        spec_sha256=spec.sha256,
        applied_model_ids=REFERENCE_FORCE_MODEL_IDS,
    )
    return final_state, diagnostics


def integrate_reference_trajectory(
    initial_state: ReferencePhaseState,
    plan: ReferenceForcePlan,
    arithmetic: DecimalContextSpec,
    spec: ImplicitMidpointSpec,
) -> ReferenceTrajectoryResult:
    """Run a fixed number of steps and retain all reference states.

    If any substage or nonlinear solve fails, no trajectory object is returned.
    The immutable input state is never modified.
    """

    if type(spec) is not ImplicitMidpointSpec:
        raise ReferenceIntegrationError("spec must be ImplicitMidpointSpec")
    states = [initial_state]
    diagnostics: list[ImplicitMidpointStepDiagnostics] = []
    current = initial_state
    for step_index in range(spec.step_count):
        current, step_diagnostics = implicit_midpoint_step(
            current,
            plan,
            arithmetic,
            spec,
            step_index=step_index,
        )
        states.append(current)
        diagnostics.append(step_diagnostics)
    return ReferenceTrajectoryResult(
        method_id=spec.method_id,
        states=tuple(states),
        diagnostics=tuple(diagnostics),
        force_plan_sha256=plan.sha256,
        decimal_context_sha256=arithmetic.sha256,
        spec_sha256=spec.sha256,
    )


__all__ = [
    "IMPLICIT_MIDPOINT_METHOD_ID",
    "ImplicitMidpointSpec",
    "ImplicitMidpointStepDiagnostics",
    "NonlinearSolveError",
    "ReferenceIntegrationError",
    "ReferenceTrajectoryResult",
    "implicit_midpoint_step",
    "integrate_reference_trajectory",
]
