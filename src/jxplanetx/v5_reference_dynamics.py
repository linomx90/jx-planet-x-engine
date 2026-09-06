"""Nonauthorizing JX V5 reference dynamics for restricted Solar 1PN tests.

This module composes exactly one Newtonian Solar monopole base acceleration
with exactly one correction from :mod:`jxplanetx.solar_1pn`.  It exists only to
exercise a separately validated velocity-dependent reference integrator.  It
does not make the draft V5 registry executable, and it is deliberately
disconnected from the legacy JX propagation paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import (
    Clamped,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    Overflow,
    Rounded,
    Subnormal,
    Underflow,
    localcontext,
)
from pathlib import Path

from .decimal_math import Vec3
from .force_registry_v5 import inspect_registry_file
from .provenance import sha256_data
from .solar_1pn import (
    AccelerationSemantics,
    CoherentUnitSystem,
    CoordinateTimeScale,
    ForceEvaluationContext,
    InertialFrame,
    RelativeState,
    RelativityContractError,
    RelativityDomainError,
    SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,
    SolarSchwarzschild1PNContract,
    TargetTreatment,
    solar_schwarzschild_1pn_correction,
)


NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID = "force.newtonian.point_mass"
REFERENCE_FORCE_MODEL_IDS = (
    NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID,
    SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,
)
REFERENCE_EVIDENCE_CLASS = "MODEL_OUTPUT"
REFERENCE_REGISTRY_STATE = "DRAFT_NONEXECUTABLE"
REFERENCE_REGISTRY_ID = "jx.force_parameter_registry.v5.draft"
REFERENCE_ROUNDING = "ROUND_HALF_EVEN"
REFERENCE_TRAPS = (
    "DivisionByZero",
    "FloatOperation",
    "InvalidOperation",
    "Overflow",
    "Underflow",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIGNALS = {
    signal.__name__: signal
    for signal in (
        Clamped,
        DivisionByZero,
        FloatOperation,
        Inexact,
        InvalidOperation,
        Overflow,
        Rounded,
        Subnormal,
        Underflow,
    )
}


class ReferenceDynamicsError(ValueError):
    """A V5 reference-dynamics contract was invalid or ambiguous."""


class ReferenceDynamicsDomainError(ValueError):
    """A reference state lies outside the implemented numerical domain."""


class ReferenceForceCompositionError(ValueError):
    """The requested force ledger is incomplete, duplicated, or unsupported."""


def _require_decimal(value: object, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise ReferenceDynamicsError(f"{label} must be decimal.Decimal")
    if not value.is_finite():
        raise ReferenceDynamicsDomainError(f"{label} must be finite")
    return value


def _require_vec3(value: object, label: str) -> Vec3:
    if type(value) is not tuple or len(value) != 3:
        raise ReferenceDynamicsError(f"{label} must be an immutable 3-tuple")
    checked = tuple(
        _require_decimal(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return checked  # type: ignore[return-value]


def _require_name(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ReferenceDynamicsError(f"{label} must be a nonempty, trimmed string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ReferenceDynamicsError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _contract_record(contract: SolarSchwarzschild1PNContract) -> dict[str, str]:
    return {
        "central_source": contract.central_source,
        "gravitational_parameter": str(contract.gravitational_parameter),
        "speed_of_light": str(contract.speed_of_light),
        "units": contract.units.name,
        "frame": contract.frame.name,
        "time_scale": contract.time_scale.name,
        "central_source_model": contract.central_source_model.name,
        "target_treatment": contract.target_treatment.name,
        "output_semantics": contract.output_semantics.name,
        "maximum_compactness": str(contract.maximum_compactness),
        "maximum_speed_fraction_squared": str(
            contract.maximum_speed_fraction_squared
        ),
    }


@dataclass(frozen=True)
class DecimalContextSpec:
    """Complete Decimal arithmetic policy for one reference trajectory."""

    precision: int
    rounding: str = REFERENCE_ROUNDING
    emin: int = -999999
    emax: int = 999999
    capitals: int = 1
    clamp: int = 0
    traps: tuple[str, ...] = REFERENCE_TRAPS

    def __post_init__(self) -> None:
        if type(self.precision) is not int or not (34 <= self.precision <= MAX_PREC):
            raise ReferenceDynamicsError(
                "precision must be an integer within the Decimal implementation bounds and at least 34"
            )
        if self.rounding != REFERENCE_ROUNDING:
            raise ReferenceDynamicsError("the reference path requires explicit ROUND_HALF_EVEN")
        if (
            type(self.emin) is not int
            or type(self.emax) is not int
            or not (MIN_EMIN <= self.emin < 0)
            or not (0 < self.emax <= MAX_EMAX)
        ):
            raise ReferenceDynamicsError(
                "emin and emax must be explicit signed integers within Decimal implementation bounds"
            )
        if type(self.capitals) is not int or self.capitals not in (0, 1):
            raise ReferenceDynamicsError("capitals must be integer 0 or 1")
        if type(self.clamp) is not int or self.clamp not in (0, 1):
            raise ReferenceDynamicsError("clamp must be integer 0 or 1")
        if type(self.traps) is not tuple or self.traps != REFERENCE_TRAPS:
            raise ReferenceDynamicsError(
                "traps must equal the frozen V5 reference signal policy"
            )

    def as_record(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "rounding": self.rounding,
            "emin": self.emin,
            "emax": self.emax,
            "capitals": self.capitals,
            "clamp": self.clamp,
            "traps": list(self.traps),
        }

    @property
    def sha256(self) -> str:
        return sha256_data(self.as_record())

    def make_context(self) -> Context:
        context = Context(
            prec=self.precision,
            rounding=self.rounding,
            Emin=self.emin,
            Emax=self.emax,
            capitals=self.capitals,
            clamp=self.clamp,
        )
        selected = set(self.traps)
        for name, signal in _SIGNALS.items():
            context.traps[signal] = name in selected
        context.clear_flags()
        return context


@dataclass(frozen=True)
class ReferenceForcePlan:
    """An inspected draft-registry binding that grants no execution authority."""

    registry_id: str
    registry_sha256: str
    registry_state: str
    scientific_contract_sha256: str
    model_ids: tuple[str, ...]
    solar_contract: SolarSchwarzschild1PNContract
    evidence_class: str = REFERENCE_EVIDENCE_CLASS
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        if self.registry_id != REFERENCE_REGISTRY_ID:
            raise ReferenceDynamicsError("unexpected V5 registry identifier")
        _require_sha256(self.registry_sha256, "registry_sha256")
        _require_sha256(self.scientific_contract_sha256, "scientific_contract_sha256")
        if self.registry_state != REFERENCE_REGISTRY_STATE:
            raise ReferenceDynamicsError("the reference plan requires the draft nonexecutable registry")
        if type(self.model_ids) is not tuple:
            raise ReferenceDynamicsError("model_ids must be an immutable ordered tuple")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ReferenceForceCompositionError("duplicate force model identifier")
        if self.model_ids != REFERENCE_FORCE_MODEL_IDS:
            raise ReferenceForceCompositionError(
                "the reference plan requires exactly one Newtonian base followed by one Solar 1PN correction"
            )
        if type(self.solar_contract) is not SolarSchwarzschild1PNContract:
            raise ReferenceDynamicsError("solar_contract has the wrong type")
        if self.evidence_class != REFERENCE_EVIDENCE_CLASS:
            raise ReferenceDynamicsError("reference output must remain MODEL_OUTPUT")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceDynamicsError("a reference force plan cannot authorize the registry")

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "jx-v5-reference-force-plan/v1",
            "registry_id": self.registry_id,
            "registry_sha256": self.registry_sha256,
            "registry_state": self.registry_state,
            "scientific_contract_sha256": self.scientific_contract_sha256,
            "model_ids": list(self.model_ids),
            "solar_contract": _contract_record(self.solar_contract),
            "evidence_class": self.evidence_class,
            "registry_authorized": self.registry_authorized,
        }

    @property
    def sha256(self) -> str:
        return sha256_data(self.as_record())


def inspect_reference_force_plan(
    registry_path: str | Path,
    solar_contract: SolarSchwarzschild1PNContract,
    *,
    project_root: str | Path | None = None,
) -> ReferenceForcePlan:
    """Inspect the draft registry and construct a nonauthorizing force plan."""

    registry, inspection = inspect_registry_file(
        registry_path,
        project_root=project_root,
    )
    if inspection.execution_authorized:
        raise ReferenceDynamicsError("this reference path must never consume executable authority")
    if inspection.registry_id != REFERENCE_REGISTRY_ID or inspection.registry_state != REFERENCE_REGISTRY_STATE:
        raise ReferenceDynamicsError("registry identity or state does not match the V5 draft")
    models = {row["model_id"]: row for row in registry["force_models"]}
    if not set(REFERENCE_FORCE_MODEL_IDS).issubset(models):
        raise ReferenceDynamicsError("registry is missing a required reference force model")
    for model_id in REFERENCE_FORCE_MODEL_IDS:
        model = models[model_id]
        if (
            model["treatment"] != "BLOCKED"
            or model["implementation_status"] == "IMPLEMENTED"
            or model["qualification_status"] == "QUALIFIED"
        ):
            raise ReferenceDynamicsError(
                f"reference model {model_id!r} must remain blocked and unqualified"
            )
    framework = registry["framework"]
    return ReferenceForcePlan(
        registry_id=inspection.registry_id,
        registry_sha256=inspection.registry_sha256,
        registry_state=inspection.registry_state,
        scientific_contract_sha256=framework["scientific_contract_sha256"],
        model_ids=REFERENCE_FORCE_MODEL_IDS,
        solar_contract=solar_contract,
    )


@dataclass(frozen=True)
class ReferencePhaseState:
    """Immutable target-minus-Sun phase state at one coordinate-time epoch."""

    epoch: Decimal
    central_source: str
    target: str
    position: Vec3
    velocity: Vec3
    units: CoherentUnitSystem
    frame: InertialFrame
    time_scale: CoordinateTimeScale
    target_treatment: TargetTreatment
    force_plan_sha256: str
    registry_sha256: str
    decimal_context_sha256: str
    evidence_class: str = REFERENCE_EVIDENCE_CLASS
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        _require_decimal(self.epoch, "epoch")
        if self.central_source != "SUN":
            raise ReferenceDynamicsError("reference state requires central_source='SUN'")
        target = _require_name(self.target, "target")
        if target == self.central_source:
            raise ReferenceDynamicsError("target must differ from central_source")
        _require_vec3(self.position, "position")
        _require_vec3(self.velocity, "velocity")
        if type(self.units) is not CoherentUnitSystem:
            raise ReferenceDynamicsError("units must be explicit")
        if self.frame is not InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED:
            raise ReferenceDynamicsError("state frame is outside the restricted reference scope")
        if type(self.time_scale) is not CoordinateTimeScale:
            raise ReferenceDynamicsError("time_scale must be explicit")
        if self.target_treatment is not TargetTreatment.MASSLESS_NO_BACKREACTION:
            raise ReferenceDynamicsError("reference state requires a massless target")
        _require_sha256(self.force_plan_sha256, "force_plan_sha256")
        _require_sha256(self.registry_sha256, "registry_sha256")
        _require_sha256(self.decimal_context_sha256, "decimal_context_sha256")
        if self.evidence_class != REFERENCE_EVIDENCE_CLASS:
            raise ReferenceDynamicsError("reference state evidence must remain MODEL_OUTPUT")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceDynamicsError("a reference state cannot assert registry authority")


def bind_reference_state(
    *,
    epoch: Decimal,
    target: str,
    position: Vec3,
    velocity: Vec3,
    plan: ReferenceForcePlan,
    arithmetic: DecimalContextSpec,
) -> ReferencePhaseState:
    """Bind state metadata to the exact plan and Decimal policy."""

    if type(plan) is not ReferenceForcePlan or type(arithmetic) is not DecimalContextSpec:
        raise ReferenceDynamicsError("plan and arithmetic contracts must be exact V5 types")
    contract = plan.solar_contract
    return ReferencePhaseState(
        epoch=epoch,
        central_source=contract.central_source,
        target=target,
        position=position,
        velocity=velocity,
        units=contract.units,
        frame=contract.frame,
        time_scale=contract.time_scale,
        target_treatment=contract.target_treatment,
        force_plan_sha256=plan.sha256,
        registry_sha256=plan.registry_sha256,
        decimal_context_sha256=arithmetic.sha256,
    )


@dataclass(frozen=True)
class ReferenceAccelerationTerm:
    model_id: str
    role: str
    acceleration: Vec3

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_FORCE_MODEL_IDS:
            raise ReferenceForceCompositionError("unexpected acceleration model identifier")
        if self.role not in ("NEWTONIAN_BASE", "CORRECTION_ONLY"):
            raise ReferenceForceCompositionError("unexpected acceleration role")
        _require_vec3(self.acceleration, "acceleration")


@dataclass(frozen=True)
class DerivativeEvaluation:
    epoch: Decimal
    target: str
    position_rate: Vec3
    velocity_rate: Vec3
    contributions: tuple[ReferenceAccelerationTerm, ...]
    applied_model_ids: tuple[str, ...]
    evidence_class: str = REFERENCE_EVIDENCE_CLASS
    registry_authorized: bool = False

    def __post_init__(self) -> None:
        _require_decimal(self.epoch, "derivative epoch")
        _require_name(self.target, "derivative target")
        _require_vec3(self.position_rate, "position_rate")
        _require_vec3(self.velocity_rate, "velocity_rate")
        if type(self.contributions) is not tuple or len(self.contributions) != 2:
            raise ReferenceForceCompositionError("reference derivative requires exactly two terms")
        if type(self.applied_model_ids) is not tuple or self.applied_model_ids != REFERENCE_FORCE_MODEL_IDS:
            raise ReferenceForceCompositionError("reference derivative force ledger is incomplete")
        if tuple(term.model_id for term in self.contributions) != self.applied_model_ids:
            raise ReferenceForceCompositionError("force ledger and contributions differ")
        if tuple(term.role for term in self.contributions) != (
            "NEWTONIAN_BASE",
            "CORRECTION_ONLY",
        ):
            raise ReferenceForceCompositionError("force contribution roles differ")
        if self.evidence_class != REFERENCE_EVIDENCE_CLASS:
            raise ReferenceDynamicsError("derivative evidence must remain MODEL_OUTPUT")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ReferenceDynamicsError("a derivative cannot assert registry authorization")


def validate_reference_state_binding(
    state: ReferencePhaseState,
    plan: ReferenceForcePlan,
    arithmetic: DecimalContextSpec,
) -> None:
    if type(state) is not ReferencePhaseState:
        raise ReferenceDynamicsError("state must be ReferencePhaseState")
    if type(plan) is not ReferenceForcePlan:
        raise ReferenceDynamicsError("plan must be ReferenceForcePlan")
    if type(arithmetic) is not DecimalContextSpec:
        raise ReferenceDynamicsError("arithmetic must be DecimalContextSpec")
    if state.force_plan_sha256 != plan.sha256:
        raise ReferenceDynamicsError("state is bound to a different force plan")
    if state.registry_sha256 != plan.registry_sha256:
        raise ReferenceDynamicsError("state is bound to a different registry")
    if state.decimal_context_sha256 != arithmetic.sha256:
        raise ReferenceDynamicsError("state is bound to a different Decimal context")
    contract = plan.solar_contract
    if (
        state.central_source != contract.central_source
        or state.units is not contract.units
        or state.frame is not contract.frame
        or state.time_scale is not contract.time_scale
        or state.target_treatment is not contract.target_treatment
    ):
        raise ReferenceDynamicsError("state metadata differs from the force plan")


def evaluate_reference_rhs(
    state: ReferencePhaseState,
    plan: ReferenceForcePlan,
    arithmetic: DecimalContextSpec,
) -> DerivativeEvaluation:
    """Evaluate ``(v, a_Newton + delta_a_1PN)`` with an atomic force ledger."""

    validate_reference_state_binding(state, plan, arithmetic)
    # Recheck cardinality before any numerical work.  Each call owns a fresh
    # ledger, so substage evaluation cannot inherit a prior target or epoch.
    if plan.model_ids != REFERENCE_FORCE_MODEL_IDS or len(set(plan.model_ids)) != 2:
        raise ReferenceForceCompositionError("invalid force composition")

    contract = plan.solar_contract
    with localcontext(arithmetic.make_context()):
        try:
            r = state.position
            radius_squared = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
            if radius_squared == 0:
                raise ReferenceDynamicsDomainError("target and Sun cannot be coincident")
            radius = radius_squared.sqrt()
            common = -contract.gravitational_parameter / (radius_squared * radius)
            newtonian = tuple(common * component for component in r)
        except DecimalException as exc:
            raise ReferenceDynamicsDomainError(
                "Decimal arithmetic failed in the Newtonian reference term"
            ) from exc
        newtonian_checked = _require_vec3(newtonian, "Newtonian acceleration")

        ledger = [NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID]
        try:
            correction = solar_schwarzschild_1pn_correction(
                contract,
                RelativeState(
                    central_source=state.central_source,
                    target=state.target,
                    position=state.position,
                    velocity=state.velocity,
                    units=state.units,
                    frame=state.frame,
                    time_scale=state.time_scale,
                    target_treatment=state.target_treatment,
                ),
                ForceEvaluationContext(frozenset(ledger)),
            )
        except RelativityDomainError as exc:
            raise ReferenceDynamicsDomainError(
                "state lies outside the Solar 1PN reference domain"
            ) from exc
        except RelativityContractError as exc:
            raise ReferenceDynamicsError("Solar 1PN contract validation failed") from exc
        if (
            correction.model_id != SOLAR_SCHWARZSCHILD_1PN_MODEL_ID
            or correction.semantics is not AccelerationSemantics.CORRECTION_ONLY
            or correction.registry_authorized
            or correction.central_source != state.central_source
            or correction.target != state.target
            or correction.units is not state.units
            or correction.frame is not state.frame
            or correction.time_scale is not state.time_scale
            or correction.target_treatment is not state.target_treatment
        ):
            raise ReferenceForceCompositionError("Solar correction metadata drifted")
        ledger.append(correction.model_id)
        if tuple(ledger) != REFERENCE_FORCE_MODEL_IDS:
            raise ReferenceForceCompositionError("force ledger did not close exactly")
        total = tuple(
            newtonian_checked[index] + correction.acceleration[index]
            for index in range(3)
        )

    return DerivativeEvaluation(
        epoch=state.epoch,
        target=state.target,
        position_rate=state.velocity,
        velocity_rate=_require_vec3(total, "total acceleration"),
        contributions=(
            ReferenceAccelerationTerm(
                NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID,
                "NEWTONIAN_BASE",
                newtonian_checked,
            ),
            ReferenceAccelerationTerm(
                SOLAR_SCHWARZSCHILD_1PN_MODEL_ID,
                "CORRECTION_ONLY",
                correction.acceleration,
            ),
        ),
        applied_model_ids=tuple(ledger),
    )


__all__ = [
    "DecimalContextSpec",
    "DerivativeEvaluation",
    "NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID",
    "REFERENCE_EVIDENCE_CLASS",
    "REFERENCE_FORCE_MODEL_IDS",
    "ReferenceAccelerationTerm",
    "ReferenceDynamicsDomainError",
    "ReferenceDynamicsError",
    "ReferenceForceCompositionError",
    "ReferenceForcePlan",
    "ReferencePhaseState",
    "bind_reference_state",
    "evaluate_reference_rhs",
    "inspect_reference_force_plan",
    "validate_reference_state_binding",
]
