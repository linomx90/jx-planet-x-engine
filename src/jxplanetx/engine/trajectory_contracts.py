"""Fail-closed contracts for adaptive JX trajectory integration.

Only the method and controller contract lives here.  Runtime checkpoint and
trajectory result types live with the integrator so there is one authoritative
array-bearing result representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ContractError


ADAPTIVE_RKF78_METHOD_ID = "integrator.adaptive.rkf78.fehlberg_1968"
RKF78_SOURCE_REPORT = "NASA-TR-R-287"
RKF78_SOURCE_DOCUMENT_ID = "19680027281"
RKF78_SOURCE_TITLE = (
    "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta "
    "Formulas with Stepsize Control"
)
RKF78_SOURCE_AUTHOR = "Erwin Fehlberg"
RKF78_SOURCE_PUBLICATION_DATE = "1968-10-01"
RKF78_SOURCE_URL = "https://ntrs.nasa.gov/citations/19680027281"

RKF78_METHOD_CLASS = "EXPLICIT_EMBEDDED_RUNGE_KUTTA"
RKF78_PRINCIPAL_ORDER = 8
RKF78_ACCEPTED_ORDER = 8
RKF78_EMBEDDED_ORDER = 7
RKF78_STAGE_COUNT = 13
RKF78_ACCEPTED_SOLUTION = "ORDER_8_HATTED"
RKF78_EMBEDDED_SOLUTION = "ORDER_7_ORDINARY"
RKF78_ACCEPTED_DISTINGUISHING_STAGES = (11, 12)
RKF78_EMBEDDED_DISTINGUISHING_STAGES = (0, 10)
RKF78_DEFECT_ORIENTATION = "ACCEPTED_ORDER_8_MINUS_EMBEDDED_ORDER_7"
RKF78_CONTROLLER_EXPONENT = 0.125

RKF78_ERROR_NORM = "NORMALIZED_MAX_PER_COMPONENT"
RKF78_ERROR_SCALE = "ATOL_PLUS_RTOL_TIMES_MAX_ABS_CURRENT_AND_CANDIDATE"
RKF78_CHECKPOINT_POLICY = "CLIP_EACH_STEP_TO_NEXT_CHECKPOINT_NO_INTERPOLATION"
RKF78_MINIMUM_STEP_FAILURE_POLICY = "ERROR"
RKF78_ZERO_DEFECT_SCALE_POLICY = "USE_MAXIMUM_SCALE_FACTOR"
RKF78_DTYPE = "float64"
RKF78_ACCEPTED_STATE_ACCUMULATION = "KAHAN_BACKEND_NATIVE_COMPONENTWISE"
RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE = "ABS_SIGNED_RKF78_STEP_ARGUMENT"
RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN = (
    "jxplanetx.accepted-step-ledger.content-integrity.v1"
)
RKF78_TIME_STEP_REPRESENTATION = "REPRESENTABLE_ENDPOINT_DELTA"
RKF78_CHECKPOINT_PROPOSAL_POLICY = (
    "PRESERVE_PRECLIP_PROPOSAL_AFTER_ACCEPTED_CLIP"
)
RKF78_EVIDENCE_CLASS = "MODEL_OUTPUT"


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite real number")
    checked = float(value)
    if not math.isfinite(checked):
        raise ContractError(f"{label} must be a finite real number")
    return checked


def _positive(value: object, label: str) -> float:
    checked = _finite(value, label)
    if checked <= 0.0:
        raise ContractError(f"{label} must be positive")
    return checked


@dataclass(frozen=True, eq=False)
class AdaptiveRKF78Spec:
    """Complete Fehlberg RK7(8) adaptive-controller request.

    ``position_atol`` and ``velocity_atol`` remain caller-owned backend-native
    float64 arrays.  They are neither inspected nor copied here; the trajectory
    runtime validates their backend, device, shape, dtype, finiteness, and
    strict positivity against the initial state before any force evaluation.

    Despite the conventional RK7(8) name, JX advances the hatted eighth-order
    solution and forms the local defect as accepted order eight minus embedded
    order seven.  That orientation is fixed below and cannot be silently
    replaced by another RKF78 convention.
    """

    checkpoint_epochs: tuple[float, ...]
    initial_step: float
    minimum_step: float
    maximum_step: float
    position_atol: object
    position_rtol: float
    velocity_atol: object
    velocity_rtol: float
    maximum_steps: int
    maximum_rejections: int
    safety_factor: float
    minimum_scale_factor: float
    maximum_scale_factor: float

    method_id: str = ADAPTIVE_RKF78_METHOD_ID
    source_report: str = RKF78_SOURCE_REPORT
    source_document_id: str = RKF78_SOURCE_DOCUMENT_ID
    source_title: str = RKF78_SOURCE_TITLE
    source_author: str = RKF78_SOURCE_AUTHOR
    source_publication_date: str = RKF78_SOURCE_PUBLICATION_DATE
    source_url: str = RKF78_SOURCE_URL
    method_class: str = RKF78_METHOD_CLASS
    principal_order: int = RKF78_PRINCIPAL_ORDER
    accepted_order: int = RKF78_ACCEPTED_ORDER
    embedded_order: int = RKF78_EMBEDDED_ORDER
    stage_count: int = RKF78_STAGE_COUNT
    force_evaluations_per_attempt: int = RKF78_STAGE_COUNT
    accepted_solution: str = RKF78_ACCEPTED_SOLUTION
    embedded_solution: str = RKF78_EMBEDDED_SOLUTION
    accepted_distinguishing_stages: tuple[int, ...] = RKF78_ACCEPTED_DISTINGUISHING_STAGES
    embedded_distinguishing_stages: tuple[int, ...] = RKF78_EMBEDDED_DISTINGUISHING_STAGES
    defect_orientation: str = RKF78_DEFECT_ORIENTATION
    controller_exponent: float = RKF78_CONTROLLER_EXPONENT
    error_norm: str = RKF78_ERROR_NORM
    error_scale: str = RKF78_ERROR_SCALE
    checkpoint_policy: str = RKF78_CHECKPOINT_POLICY
    minimum_step_failure_policy: str = RKF78_MINIMUM_STEP_FAILURE_POLICY
    zero_defect_scale_policy: str = RKF78_ZERO_DEFECT_SCALE_POLICY
    dtype: str = RKF78_DTYPE
    accepted_state_accumulation: str = RKF78_ACCEPTED_STATE_ACCUMULATION
    time_step_representation: str = RKF78_TIME_STEP_REPRESENTATION
    checkpoint_proposal_policy: str = RKF78_CHECKPOINT_PROPOSAL_POLICY
    supports_velocity_dependent_forces: bool = True
    dense_output: bool = False
    evidence_class: str = RKF78_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.checkpoint_epochs) is not tuple or len(self.checkpoint_epochs) < 2:
            raise ContractError(
                "checkpoint_epochs must be an immutable tuple containing the initial epoch and at least one later epoch"
            )
        epochs = tuple(
            _finite(epoch, f"checkpoint_epochs[{index}]")
            for index, epoch in enumerate(self.checkpoint_epochs)
        )
        differences = tuple(right - left for left, right in zip(epochs, epochs[1:]))
        if not (all(delta > 0.0 for delta in differences) or all(delta < 0.0 for delta in differences)):
            raise ContractError(
                "checkpoint_epochs must be strictly monotone in one forward or backward direction"
            )

        initial_step = _positive(self.initial_step, "initial_step")
        minimum_step = _positive(self.minimum_step, "minimum_step")
        maximum_step = _positive(self.maximum_step, "maximum_step")
        if not minimum_step <= initial_step <= maximum_step:
            raise ContractError(
                "step magnitudes must satisfy minimum_step <= initial_step <= maximum_step"
            )
        if self.position_atol is None or self.velocity_atol is None:
            raise ContractError(
                "position_atol and velocity_atol must be backend-native float64 arrays"
            )
        _positive(self.position_rtol, "position_rtol")
        _positive(self.velocity_rtol, "velocity_rtol")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise ContractError("maximum_steps must be a positive integer")
        if type(self.maximum_rejections) is not int or self.maximum_rejections < 0:
            raise ContractError("maximum_rejections must be a nonnegative integer")

        safety = _positive(self.safety_factor, "safety_factor")
        minimum_scale = _positive(self.minimum_scale_factor, "minimum_scale_factor")
        maximum_scale = _positive(self.maximum_scale_factor, "maximum_scale_factor")
        if safety >= 1.0:
            raise ContractError("safety_factor must be below one")
        if minimum_scale > 1.0:
            raise ContractError("minimum_scale_factor cannot exceed one")
        if maximum_scale < 1.0:
            raise ContractError("maximum_scale_factor cannot be below one")
        if minimum_scale > maximum_scale:
            raise ContractError(
                "minimum_scale_factor cannot exceed maximum_scale_factor"
            )

        exact = {
            "method_id": ADAPTIVE_RKF78_METHOD_ID,
            "source_report": RKF78_SOURCE_REPORT,
            "source_document_id": RKF78_SOURCE_DOCUMENT_ID,
            "source_title": RKF78_SOURCE_TITLE,
            "source_author": RKF78_SOURCE_AUTHOR,
            "source_publication_date": RKF78_SOURCE_PUBLICATION_DATE,
            "source_url": RKF78_SOURCE_URL,
            "method_class": RKF78_METHOD_CLASS,
            "principal_order": RKF78_PRINCIPAL_ORDER,
            "accepted_order": RKF78_ACCEPTED_ORDER,
            "embedded_order": RKF78_EMBEDDED_ORDER,
            "stage_count": RKF78_STAGE_COUNT,
            "force_evaluations_per_attempt": RKF78_STAGE_COUNT,
            "accepted_solution": RKF78_ACCEPTED_SOLUTION,
            "embedded_solution": RKF78_EMBEDDED_SOLUTION,
            "accepted_distinguishing_stages": RKF78_ACCEPTED_DISTINGUISHING_STAGES,
            "embedded_distinguishing_stages": RKF78_EMBEDDED_DISTINGUISHING_STAGES,
            "defect_orientation": RKF78_DEFECT_ORIENTATION,
            "controller_exponent": RKF78_CONTROLLER_EXPONENT,
            "error_norm": RKF78_ERROR_NORM,
            "error_scale": RKF78_ERROR_SCALE,
            "checkpoint_policy": RKF78_CHECKPOINT_POLICY,
            "minimum_step_failure_policy": RKF78_MINIMUM_STEP_FAILURE_POLICY,
            "zero_defect_scale_policy": RKF78_ZERO_DEFECT_SCALE_POLICY,
            "dtype": RKF78_DTYPE,
            "accepted_state_accumulation": RKF78_ACCEPTED_STATE_ACCUMULATION,
            "time_step_representation": RKF78_TIME_STEP_REPRESENTATION,
            "checkpoint_proposal_policy": RKF78_CHECKPOINT_PROPOSAL_POLICY,
            "supports_velocity_dependent_forces": True,
            "dense_output": False,
            "evidence_class": RKF78_EVIDENCE_CLASS,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for field, expected in exact.items():
            value = getattr(self, field)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(f"{field} must equal the fixed RKF78 contract value {expected!r}")

    @property
    def direction(self) -> str:
        return "FORWARD" if self.checkpoint_epochs[-1] > self.checkpoint_epochs[0] else "BACKWARD"

    @property
    def controller_formula(self) -> str:
        return "CLAMP(SAFETY_FACTOR*ERROR_NORM^(-1/8),MINIMUM_SCALE_FACTOR,MAXIMUM_SCALE_FACTOR)"

    @property
    def acceptance_rule(self) -> str:
        return "ACCEPT_IF_NORMALIZED_MAX_ERROR_LESS_THAN_OR_EQUAL_TO_ONE"


__all__ = [
    "ADAPTIVE_RKF78_METHOD_ID",
    "AdaptiveRKF78Spec",
    "RKF78_ACCEPTED_STATE_ACCUMULATION",
    "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM",
    "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN",
    "RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE",
    "RKF78_ACCEPTED_ORDER",
    "RKF78_ACCEPTED_DISTINGUISHING_STAGES",
    "RKF78_ACCEPTED_SOLUTION",
    "RKF78_CHECKPOINT_POLICY",
    "RKF78_CHECKPOINT_PROPOSAL_POLICY",
    "RKF78_CONTROLLER_EXPONENT",
    "RKF78_DEFECT_ORIENTATION",
    "RKF78_DTYPE",
    "RKF78_EMBEDDED_ORDER",
    "RKF78_EMBEDDED_DISTINGUISHING_STAGES",
    "RKF78_EMBEDDED_SOLUTION",
    "RKF78_ERROR_NORM",
    "RKF78_ERROR_SCALE",
    "RKF78_EVIDENCE_CLASS",
    "RKF78_METHOD_CLASS",
    "RKF78_MINIMUM_STEP_FAILURE_POLICY",
    "RKF78_PRINCIPAL_ORDER",
    "RKF78_SOURCE_AUTHOR",
    "RKF78_SOURCE_DOCUMENT_ID",
    "RKF78_SOURCE_PUBLICATION_DATE",
    "RKF78_SOURCE_REPORT",
    "RKF78_SOURCE_TITLE",
    "RKF78_SOURCE_URL",
    "RKF78_STAGE_COUNT",
    "RKF78_TIME_STEP_REPRESENTATION",
    "RKF78_ZERO_DEFECT_SCALE_POLICY",
]
