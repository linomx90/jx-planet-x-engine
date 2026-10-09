"""Fail-closed contracts for the fixed-step Newtonian KDK map."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ContractError


FIXED_STEP_KDK_METHOD_ID = "integrator.symplectic.kdk_leapfrog_2"
KDK_METHOD_CLASS = "EXPLICIT_SYMMETRIC_KICK_DRIFT_KICK_SPLITTING"
KDK_PRINCIPAL_ORDER = 2
KDK_KICK_COEFFICIENTS = (0.5, 0.5)
KDK_DRIFT_COEFFICIENTS = (1.0,)
KDK_COMPOSITION = "KICK_HALF_DRIFT_FULL_KICK_HALF"
KDK_STEP_REPRESENTATION = "CONSTANT_SIGNED_BINARY64_MAP_STEP"
KDK_CHECKPOINT_POLICY = "INTEGER_STEP_LATTICE_NO_CLIPPING_NO_INTERPOLATION"
KDK_FORCE_PLAN_SCOPE = "MUTUAL_ALL_BODY_NEWTONIAN_POINT_MASS_ONLY"
KDK_BACKEND_SCOPE = "NUMPY_CPU_ONLY"
KDK_TIME_SEMANTICS = "CONTINUOUS_COORDINATE_TIME"
KDK_ALLOWED_TIME_SCALES = ("TDB", "SYNTHETIC")
KDK_REQUIRED_FRAME = "BARYCENTRIC_INERTIAL"
KDK_REQUIRED_ORIGIN = "BARYCENTER"
KDK_ALLOWED_AXES = ("ICRS_ALIGNED", "CARTESIAN_RIGHT_HANDED")
KDK_ENCOUNTER_GUARD = (
    "LINEAR_DRIFT_PAIRWISE_MINIMUM_SEPARATION_STRICTLY_ABOVE_"
    "MAX_CONTACT_AND_CALLER_FLOOR"
)
KDK_PAIR_FREQUENCY_GUARD = (
    "ABS_H_TIMES_SQRT_PAIR_GM_OVER_SWEPT_MINIMUM_SEPARATION_CUBED_"
    "LESS_THAN_OR_EQUAL_TO_CALLER_LIMIT"
)
KDK_FORCE_EVALUATION_ACCOUNTING = "ONE_INITIAL_PLUS_ONE_PER_COMPLETED_MAP_STEP"
KDK_SCHEDULE_CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
KDK_SCHEDULE_CHECKSUM_DOMAIN = "jxplanetx.kdk-schedule.content-integrity.v1"
KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
KDK_RESULT_CONTENT_CHECKSUM_DOMAIN = "jxplanetx.kdk-result.content-integrity.v1"
KDK_EVIDENCE_CLASS = "MODEL_OUTPUT"


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ContractError(f"{label} must be a finite positive built-in float")
    return value


@dataclass(frozen=True, eq=False)
class FixedStepKDKSpec:
    """One constant signed KDK step and an integer output lattice.

    The two safety bounds are mandatory caller choices.  The swept-distance
    floor is expressed in the snapshot length unit; the pair-frequency limit
    is the dimensionless maximum ``abs(h)*sqrt((GM_i+GM_j)/r_min**3)``.
    """

    checkpoint_step_indices: tuple[int, ...]
    fixed_step: float
    maximum_steps: int
    minimum_swept_pair_separation: float
    maximum_pair_frequency_step: float

    method_id: str = FIXED_STEP_KDK_METHOD_ID
    method_class: str = KDK_METHOD_CLASS
    principal_order: int = KDK_PRINCIPAL_ORDER
    kick_coefficients: tuple[float, ...] = KDK_KICK_COEFFICIENTS
    drift_coefficients: tuple[float, ...] = KDK_DRIFT_COEFFICIENTS
    composition: str = KDK_COMPOSITION
    step_representation: str = KDK_STEP_REPRESENTATION
    checkpoint_policy: str = KDK_CHECKPOINT_POLICY
    force_plan_scope: str = KDK_FORCE_PLAN_SCOPE
    backend_scope: str = KDK_BACKEND_SCOPE
    time_semantics: str = KDK_TIME_SEMANTICS
    encounter_guard: str = KDK_ENCOUNTER_GUARD
    pair_frequency_guard: str = KDK_PAIR_FREQUENCY_GUARD
    force_evaluation_accounting: str = KDK_FORCE_EVALUATION_ACCOUNTING
    exact_arithmetic_symplectic: bool = True
    floating_point_symplectic: bool = False
    exact_arithmetic_time_reversible: bool = True
    dense_output: bool = False
    adaptive: bool = False
    evidence_class: str = KDK_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        indices = self.checkpoint_step_indices
        if type(indices) is not tuple or len(indices) < 2:
            raise ContractError(
                "checkpoint_step_indices must be an immutable tuple containing "
                "zero and at least one later map-step index"
            )
        if any(type(index) is not int or index < 0 for index in indices):
            raise ContractError(
                "checkpoint_step_indices entries must be nonnegative built-in integers"
            )
        if indices[0] != 0 or any(right <= left for left, right in zip(indices, indices[1:])):
            raise ContractError(
                "checkpoint_step_indices must begin at zero and be strictly increasing"
            )
        if type(self.fixed_step) is not float or not math.isfinite(self.fixed_step):
            raise ContractError("fixed_step must be a finite nonzero built-in float")
        if self.fixed_step == 0.0:
            raise ContractError("fixed_step must be nonzero")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise ContractError("maximum_steps must be a positive built-in integer")
        if indices[-1] > self.maximum_steps:
            raise ContractError(
                "the final checkpoint step index cannot exceed maximum_steps"
            )
        _positive_float(
            self.minimum_swept_pair_separation,
            "minimum_swept_pair_separation",
        )
        frequency_limit = _positive_float(
            self.maximum_pair_frequency_step,
            "maximum_pair_frequency_step",
        )
        if frequency_limit > 1.0:
            raise ContractError("maximum_pair_frequency_step cannot exceed one")
        for field in ("kick_coefficients", "drift_coefficients"):
            values = getattr(self, field)
            if type(values) is not tuple or any(type(value) is not float for value in values):
                raise ContractError(
                    f"{field} must be an exact tuple of built-in floats"
                )

        exact = {
            "method_id": FIXED_STEP_KDK_METHOD_ID,
            "method_class": KDK_METHOD_CLASS,
            "principal_order": KDK_PRINCIPAL_ORDER,
            "kick_coefficients": KDK_KICK_COEFFICIENTS,
            "drift_coefficients": KDK_DRIFT_COEFFICIENTS,
            "composition": KDK_COMPOSITION,
            "step_representation": KDK_STEP_REPRESENTATION,
            "checkpoint_policy": KDK_CHECKPOINT_POLICY,
            "force_plan_scope": KDK_FORCE_PLAN_SCOPE,
            "backend_scope": KDK_BACKEND_SCOPE,
            "time_semantics": KDK_TIME_SEMANTICS,
            "encounter_guard": KDK_ENCOUNTER_GUARD,
            "pair_frequency_guard": KDK_PAIR_FREQUENCY_GUARD,
            "force_evaluation_accounting": KDK_FORCE_EVALUATION_ACCOUNTING,
            "exact_arithmetic_symplectic": True,
            "floating_point_symplectic": False,
            "exact_arithmetic_time_reversible": True,
            "dense_output": False,
            "adaptive": False,
            "evidence_class": KDK_EVIDENCE_CLASS,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for field, expected in exact.items():
            value = getattr(self, field)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(
                    f"{field} must equal the fixed KDK contract value {expected!r}"
                )

    @property
    def direction(self) -> str:
        return "FORWARD" if self.fixed_step > 0.0 else "BACKWARD"

    @property
    def completed_steps(self) -> int:
        return self.checkpoint_step_indices[-1]


__all__ = [
    "FIXED_STEP_KDK_METHOD_ID",
    "FixedStepKDKSpec",
    "KDK_ALLOWED_AXES",
    "KDK_ALLOWED_TIME_SCALES",
    "KDK_BACKEND_SCOPE",
    "KDK_CHECKPOINT_POLICY",
    "KDK_COMPOSITION",
    "KDK_DRIFT_COEFFICIENTS",
    "KDK_ENCOUNTER_GUARD",
    "KDK_EVIDENCE_CLASS",
    "KDK_FORCE_EVALUATION_ACCOUNTING",
    "KDK_FORCE_PLAN_SCOPE",
    "KDK_KICK_COEFFICIENTS",
    "KDK_METHOD_CLASS",
    "KDK_PAIR_FREQUENCY_GUARD",
    "KDK_PRINCIPAL_ORDER",
    "KDK_REQUIRED_FRAME",
    "KDK_REQUIRED_ORIGIN",
    "KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "KDK_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "KDK_SCHEDULE_CHECKSUM_ALGORITHM",
    "KDK_SCHEDULE_CHECKSUM_DOMAIN",
    "KDK_STEP_REPRESENTATION",
    "KDK_TIME_SEMANTICS",
]
