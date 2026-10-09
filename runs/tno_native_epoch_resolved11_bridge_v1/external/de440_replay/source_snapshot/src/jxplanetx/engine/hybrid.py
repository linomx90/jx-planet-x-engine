"""Transactional whole-macrostep WH/RKF78 hybrid runtime.

The runtime composes the frozen private Wisdom--Holman proposer with the
frozen private Cartesian encounter executor.  A provisional WH macrostep is
either committed in full or discarded in full; a switched interval always
starts again from the untouched committed Cartesian node.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from . import encounter as _enc
from . import wisdom_holman as _wh
from .backends import resolve_backend
from .contracts import BackendSpec, ForcePlan, StateSnapshot
from .encounter import EncounterExecutionCounts
from .encounter_contracts import (
    ENCOUNTER_HARD_MAXIMUM_BODY_COUNT,
    ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
)
from .evaluator import ForceLedgerEntry
from .hybrid_contracts import (
    HYBRID_ALL_FAR_REASON,
    HYBRID_CLAIM_SCOPE,
    HYBRID_EVIDENCE_CLASS,
    HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER,
    HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE,
    HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL,
    HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE,
    HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL,
    HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE,
    HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL,
    HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE,
    HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL,
    HYBRID_NONCLAIMS,
    HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM,
    HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER,
    HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN,
    HYBRID_SCHEDULE_CHECKSUM_ALGORITHM,
    HYBRID_SCHEDULE_CHECKSUM_DOMAIN,
    HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM,
    HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN,
    HYBRID_VALIDATION_REPLAY_COUNT,
    HYBRID_VALIDATION_REPLAY_POLICY,
    HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
    HybridFarProbeDecision,
    HybridFarProbeWork,
    HybridPrivateEncounterRecord,
    HybridWisdomHolmanRKF78Spec,
    _bind_encounter_segment,
)
from .trajectory import (
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryError,
    TrajectoryStepLimitError,
)


_HYBRID_SCHEDULE_SCHEMA = "jxplanetx.hybrid-wh-rkf78-schedule.payload.v1"
_HYBRID_STEP_LEDGER_SCHEMA = "jxplanetx.hybrid-wh-rkf78-step-ledger.payload.v1"
_HYBRID_RESULT_SCHEMA = "jxplanetx.hybrid-wh-rkf78-result.payload.v1"
_PRIVATE_DIGEST_BYTES_PER_NEAR = 32 * len(
    HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER
)
_MAXIMUM_CANONICAL_BINARY64_ITEM_BYTES = 40
_MAXIMUM_UINT64 = (1 << 64) - 1


class HybridError(TrajectoryError):
    """A transactional WH/RKF78 request could not be completed."""


class HybridContractError(HybridError, TrajectoryContractError):
    """Hybrid input, custody, accounting, or replay was inconsistent."""


class HybridDomainError(HybridError, TrajectoryDomainError):
    """A hybrid execution left its finite numerical domain."""


class HybridStepLimitError(HybridError, TrajectoryStepLimitError):
    """A hard hybrid aggregate or child step ceiling was exhausted."""


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HybridContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    return _enc._readonly_copy(value, np.dtype(np.float64))


def _same_float(left: object, right: object) -> bool:
    return type(left) is float and type(right) is float and left.hex() == right.hex()


def _same_content(left: object, right: object) -> bool:
    return _enc._canonical_json(left) == _enc._canonical_json(right)


def _domain_sha256(domain: str, value: object) -> str:
    return _enc._domain_sha256(domain, value)


def _streaming_sequence_sha256(
    *,
    domain: str,
    schema: str,
    values: tuple[object, ...],
    max_count: int,
    max_item_bytes: int,
    max_cumulative_bytes: int,
) -> str:
    if type(domain) is not str or type(schema) is not str or type(values) is not tuple:
        raise HybridContractError("streaming digest inputs must have exact built-in types")
    for name, value in (
        ("max_count", max_count),
        ("max_item_bytes", max_item_bytes),
        ("max_cumulative_bytes", max_cumulative_bytes),
    ):
        if type(value) is not int or value < 0 or value > _MAXIMUM_UINT64:
            raise HybridContractError(
                f"{name} must be an exact unsigned-64-bit integer"
            )
    # Count is rejected before inspecting or encoding a caller-controlled item.
    if len(values) > max_count:
        raise HybridContractError("streaming sequence count exceeds its retained cap")
    if values and max_item_bytes == 0:
        raise HybridContractError(
            "a nonempty streaming sequence requires positive item capacity"
        )
    domain_bytes = domain.encode("utf-8")
    schema_bytes = schema.encode("utf-8")
    prefix = domain_bytes + b"\x00" + schema_bytes + b"\x00"
    framed_count = len(prefix) + 8
    if framed_count > max_cumulative_bytes:
        raise HybridContractError(
            "streaming sequence framing exceeds its cumulative byte cap"
        )
    digest = hashlib.sha256()
    digest.update(domain_bytes)
    digest.update(b"\x00")
    digest.update(schema_bytes)
    digest.update(b"\x00")
    digest.update(len(values).to_bytes(8, "big", signed=False))
    cumulative_bytes = framed_count
    for value in values:
        # Require room for an item-length frame before asking the canonical
        # encoder to materialize this already schema-bounded exact item.
        if cumulative_bytes + 8 + max_item_bytes > max_cumulative_bytes:
            raise HybridContractError(
                "streaming sequence cannot reserve its next item before encoding"
            )
        encoded = _enc._canonical_json(value)
        item_bytes = len(encoded)
        if item_bytes > max_item_bytes:
            raise HybridContractError(
                "streaming sequence item exceeds its retained byte cap"
            )
        prospective = cumulative_bytes + 8 + item_bytes
        if prospective > max_cumulative_bytes:
            raise HybridContractError(
                "streaming sequence exceeds its cumulative byte cap"
            )
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
        cumulative_bytes = prospective
    return digest.hexdigest()


def _streaming_cumulative_cap(
    *, domain: str, schema: str, max_count: int, max_item_payload_bytes: int
) -> int:
    """Exact framing ceiling from a bounded item-payload byte budget."""

    if (
        type(domain) is not str
        or type(schema) is not str
        or type(max_count) is not int
        or type(max_item_payload_bytes) is not int
        or max_count < 0
        or max_item_payload_bytes < 0
    ):
        raise HybridContractError("streaming cap inputs require exact bounded types")
    total = (
        len(domain.encode("utf-8"))
        + 1
        + len(schema.encode("utf-8"))
        + 1
        + 8
        + 8 * max_count
        + max_item_payload_bytes
    )
    if total > _MAXIMUM_UINT64:
        raise HybridContractError("streaming cumulative cap exceeds unsigned-64-bit")
    return total


def _bounded_canonical_json(
    value: object,
    *,
    max_item_bytes: int,
    remaining_cumulative_bytes: int,
    label: str,
) -> bytes:
    """Encode one schema-bounded witness only while retained capacity remains."""

    if (
        type(max_item_bytes) is not int
        or type(remaining_cumulative_bytes) is not int
        or max_item_bytes < 0
        or remaining_cumulative_bytes < 0
    ):
        raise HybridContractError(f"{label} byte caps require exact integers")
    if max_item_bytes == 0 or remaining_cumulative_bytes == 0:
        raise HybridContractError(f"{label} has no prospective byte capacity")
    if max_item_bytes > remaining_cumulative_bytes:
        raise HybridContractError(
            f"{label} cannot reserve its per-item maximum before encoding"
        )
    encoded = _enc._canonical_json(value)
    if len(encoded) > max_item_bytes:
        raise HybridContractError(f"{label} exceeds its per-item byte cap")
    if len(encoded) > remaining_cumulative_bytes:
        raise HybridContractError(f"{label} exceeds its cumulative byte cap")
    return encoded


@dataclass(frozen=True, eq=False)
class HybridProbeWorkTotals:
    """Componentwise sum of the nineteen numeric far-probe counters."""

    force_calls_entered: int
    force_evaluations_completed: int
    interaction_force_assemblies: int
    kepler_subflow_calls_entered: int
    kepler_subflow_solves_completed: int
    universal_solver_iterations: int
    universal_solver_bracket_expansions: int
    universal_g_bundle_calls_entered: int
    universal_g_bundle_calls_completed: int
    universal_series_terms_evaluated: int
    cartesian_to_jacobi_calls_entered: int
    cartesian_to_jacobi_transforms_completed: int
    jacobi_to_cartesian_calls_entered: int
    jacobi_to_cartesian_transforms_completed: int
    node_guard_evaluations: int
    path_guard_evaluations: int
    first_half_kicks_completed: int
    center_of_mass_drifts_completed: int
    second_half_kicks_completed: int

    def __post_init__(self) -> None:
        if type(self) is not HybridProbeWorkTotals:
            raise HybridContractError("probe totals must have their exact public type")
        if tuple(descriptor.name for descriptor in fields(self)) != (
            HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
        ):
            raise HybridContractError("hybrid probe-total field roster changed")
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if type(value) is not int or value < 0:
                raise HybridContractError(
                    f"{descriptor.name} must be an exact nonnegative integer"
                )


def _zero_probe_totals() -> HybridProbeWorkTotals:
    return HybridProbeWorkTotals(
        **{name: 0 for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER}
    )


def _probe_totals(work: HybridFarProbeWork) -> HybridProbeWorkTotals:
    if type(work) is not HybridFarProbeWork:
        raise HybridContractError("probe work must be exact HybridFarProbeWork")
    work.__post_init__()
    return HybridProbeWorkTotals(
        **{
            name: getattr(work, name)
            for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
        }
    )


def _sum_probe_totals(
    left: HybridProbeWorkTotals, right: HybridProbeWorkTotals
) -> HybridProbeWorkTotals:
    if type(left) is not HybridProbeWorkTotals or type(right) is not HybridProbeWorkTotals:
        raise HybridContractError("probe totals must have exact types")
    return HybridProbeWorkTotals(
        **{
            name: getattr(left, name) + getattr(right, name)
            for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
        }
    )


def _zero_encounter_counts() -> EncounterExecutionCounts:
    return EncounterExecutionCounts(
        **{descriptor.name: 0 for descriptor in fields(EncounterExecutionCounts)}
    )


def _sum_encounter_counts(
    left: EncounterExecutionCounts, right: EncounterExecutionCounts
) -> EncounterExecutionCounts:
    if type(left) is not EncounterExecutionCounts or type(right) is not EncounterExecutionCounts:
        raise HybridContractError("encounter counts must have exact frozen types")
    return EncounterExecutionCounts(
        **{
            descriptor.name: getattr(left, descriptor.name)
            + getattr(right, descriptor.name)
            for descriptor in fields(EncounterExecutionCounts)
        }
    )


@dataclass(frozen=True, eq=False)
class HybridLaneExecutionCounts:
    """Exact accounting for one primary or replay hybrid lane."""

    outer_records: int
    far_probe_count: int
    far_accepted_count: int
    far_discarded_count: int
    private_encounter_execution_count: int
    near_accepted_substeps: int
    retained_near_digest_bytes: int
    checkpoint_count: int
    accepted_wh_work: HybridProbeWorkTotals
    discarded_wh_work: HybridProbeWorkTotals
    total_probe_work: HybridProbeWorkTotals
    private_encounter_counts: EncounterExecutionCounts

    def __post_init__(self) -> None:
        if type(self) is not HybridLaneExecutionCounts:
            raise HybridContractError("lane counts must have their exact public type")
        for name in (
            "outer_records",
            "far_probe_count",
            "far_accepted_count",
            "far_discarded_count",
            "private_encounter_execution_count",
            "near_accepted_substeps",
            "retained_near_digest_bytes",
            "checkpoint_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise HybridContractError(f"{name} must be an exact nonnegative integer")
        for name in ("accepted_wh_work", "discarded_wh_work", "total_probe_work"):
            value = getattr(self, name)
            if type(value) is not HybridProbeWorkTotals:
                raise HybridContractError(f"{name} must be exact HybridProbeWorkTotals")
            value.__post_init__()
        if type(self.private_encounter_counts) is not EncounterExecutionCounts:
            raise HybridContractError(
                "private_encounter_counts must be exact EncounterExecutionCounts"
            )
        self.private_encounter_counts.__post_init__()
        if (
            self.outer_records != self.far_probe_count
            or self.outer_records != self.far_accepted_count + self.far_discarded_count
            or self.private_encounter_execution_count != self.far_discarded_count
            or self.total_probe_work
            != _sum_probe_totals(self.accepted_wh_work, self.discarded_wh_work)
        ):
            # Dataclass equality is deliberately disabled; compare canonical bytes.
            if not _same_content(
                self.total_probe_work,
                _sum_probe_totals(self.accepted_wh_work, self.discarded_wh_work),
            ) or (
                self.outer_records != self.far_probe_count
                or self.outer_records
                != self.far_accepted_count + self.far_discarded_count
                or self.private_encounter_execution_count != self.far_discarded_count
            ):
                raise HybridContractError("hybrid lane accounting equations do not close")
        if (
            self.near_accepted_substeps
            != self.private_encounter_counts.accepted_substeps
            or self.retained_near_digest_bytes
            != self.private_encounter_execution_count
            * _PRIVATE_DIGEST_BYTES_PER_NEAR
            or self.checkpoint_count > self.outer_records + 2
        ):
            raise HybridContractError(
                "hybrid near/checkpoint accounting differs from retained work"
            )
        if (
            self.outer_records > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL
            or self.far_discarded_count
            > HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL
            or self.near_accepted_substeps
            > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL
            or self.retained_near_digest_bytes
            > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL
        ):
            raise HybridContractError("hybrid counts exceed the public hard envelope")
        # The same exact schema is used for one lane and for the public sum.
        # Context-specific lane/public ceilings are enforced by the enclosing
        # result validator before any retained tuple traversal.


def _sum_lane_counts(
    left: HybridLaneExecutionCounts, right: HybridLaneExecutionCounts
) -> HybridLaneExecutionCounts:
    return HybridLaneExecutionCounts(
        outer_records=left.outer_records + right.outer_records,
        far_probe_count=left.far_probe_count + right.far_probe_count,
        far_accepted_count=left.far_accepted_count + right.far_accepted_count,
        far_discarded_count=left.far_discarded_count + right.far_discarded_count,
        private_encounter_execution_count=(
            left.private_encounter_execution_count
            + right.private_encounter_execution_count
        ),
        near_accepted_substeps=(
            left.near_accepted_substeps + right.near_accepted_substeps
        ),
        retained_near_digest_bytes=(
            left.retained_near_digest_bytes + right.retained_near_digest_bytes
        ),
        checkpoint_count=left.checkpoint_count + right.checkpoint_count,
        accepted_wh_work=_sum_probe_totals(
            left.accepted_wh_work, right.accepted_wh_work
        ),
        discarded_wh_work=_sum_probe_totals(
            left.discarded_wh_work, right.discarded_wh_work
        ),
        total_probe_work=_sum_probe_totals(
            left.total_probe_work, right.total_probe_work
        ),
        private_encounter_counts=_sum_encounter_counts(
            left.private_encounter_counts, right.private_encounter_counts
        ),
    )


@dataclass(frozen=True, eq=False)
class HybridOuterStepRecord:
    """One committed outer interval and its complete typed probe custody."""

    outer_step_index: int
    start_epoch: float
    endpoint_epoch: float
    signed_step: float
    mode: str
    probe_committed: bool
    decision: HybridFarProbeDecision
    private_encounter: HybridPrivateEncounterRecord | None

    def __post_init__(self) -> None:
        if type(self) is not HybridOuterStepRecord:
            raise HybridContractError("outer record must have its exact public type")
        if (
            type(self.outer_step_index) is not int
            or not 1
            <= self.outer_step_index
            <= HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
        ):
            raise HybridContractError("outer_step_index exceeds its hard lane cap")
        for name in ("start_epoch", "endpoint_epoch", "signed_step"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise HybridContractError(f"{name} must be a finite built-in float")
        if self.signed_step == 0.0:
            raise HybridContractError("signed_step cannot be zero")
        if (
            self.signed_step > 0.0
            and self.endpoint_epoch <= self.start_epoch
        ) or (
            self.signed_step < 0.0
            and self.endpoint_epoch >= self.start_epoch
        ):
            raise HybridContractError("outer labels do not advance in signed-step direction")
        if type(self.mode) is not str or type(self.probe_committed) is not bool:
            raise HybridContractError("outer mode/commit fields require exact types")
        if type(self.decision) is not HybridFarProbeDecision:
            raise HybridContractError("decision must be exact HybridFarProbeDecision")
        self.decision.__post_init__()
        if self.mode == "WISDOM_HOLMAN_FAR":
            if (
                self.probe_committed is not True
                or self.decision.outcome != "FAR_PASS"
                or self.decision.reason != HYBRID_ALL_FAR_REASON
                or self.private_encounter is not None
            ):
                raise HybridContractError("far outer record is internally inconsistent")
        elif self.mode == "CARTESIAN_RKF78_NEAR_FULL_INTERVAL":
            if (
                self.probe_committed is not False
                or self.decision.outcome != "NEAR_SWITCH"
                or type(self.private_encounter) is not HybridPrivateEncounterRecord
            ):
                raise HybridContractError("near outer record is internally inconsistent")
            self.private_encounter.__post_init__()
            if self.private_encounter.outer_step_index != self.outer_step_index:
                raise HybridContractError("near child index differs from its outer record")
        else:
            raise HybridContractError("outer record mode is outside the closed roster")


@dataclass(frozen=True, eq=False)
class HybridWisdomHolmanDiagnostics:
    """Committed-WH scalar evidence; complete public-WH evidence when all-far."""

    maximum_universal_solver_iterations: int
    maximum_universal_solver_bracket_expansions: int
    maximum_kepler_time_residual: float
    maximum_kepler_residual_tolerance: float
    maximum_kepler_lagrange_identity_error: float
    maximum_kepler_energy_error: float
    maximum_kepler_angular_momentum_error: float
    maximum_translation_force_residual: float
    minimum_jacobi_periapses: tuple[float, ...]
    maximum_jacobi_eccentricities: tuple[float, ...]
    maximum_interaction_force_ratios: tuple[float, ...]
    maximum_orbit_step_fractions: tuple[float, ...]
    maximum_periapse_step_fractions: tuple[float, ...]
    minimum_pair_endpoint_separations: tuple[float, ...]
    minimum_pair_path_lower_bounds: tuple[float, ...]
    minimum_pair_clearance_after_margins: tuple[float, ...]
    minimum_secondary_hill_floor_ratios: tuple[float, ...]
    maximum_barycenter_position_norm: float
    maximum_barycenter_velocity_norm: float
    initial_barycenter_position_cap: float
    initial_barycenter_velocity_cap: float

    def __post_init__(self) -> None:
        if type(self) is not HybridWisdomHolmanDiagnostics:
            raise HybridContractError("WH diagnostics must have their exact public type")
        for name in (
            "maximum_universal_solver_iterations",
            "maximum_universal_solver_bracket_expansions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise HybridContractError(f"{name} must be an exact nonnegative integer")
        tuple_caps = {
            "minimum_jacobi_periapses": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1,
            "maximum_jacobi_eccentricities": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1,
            "maximum_interaction_force_ratios": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1,
            "maximum_orbit_step_fractions": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1,
            "maximum_periapse_step_fractions": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1,
            "minimum_pair_endpoint_separations": ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
            "minimum_pair_path_lower_bounds": ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
            "minimum_pair_clearance_after_margins": ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
            "minimum_secondary_hill_floor_ratios": (
                (ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1)
                * (ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 2)
                // 2
            ),
        }
        # Exact tuple types and absolute lengths are checked before any nested
        # entry traversal, including on a directly constructed diagnostics value.
        for name, cap in tuple_caps.items():
            value = getattr(self, name)
            if type(value) is not tuple or len(value) > cap:
                raise HybridContractError(
                    f"{name} must be an exact tuple within its hard body/pair cap"
                )
        for descriptor in fields(self)[2:]:
            value = getattr(self, descriptor.name)
            if type(value) is float:
                if not math.isfinite(value) or value < 0.0:
                    raise HybridContractError(
                        f"{descriptor.name} must be finite and nonnegative"
                    )
            elif descriptor.name in tuple_caps:
                if any(
                    type(item) is not float
                    or not math.isfinite(item)
                    or item < 0.0
                    for item in value
                ):
                    raise HybridContractError(
                        f"{descriptor.name} must contain finite nonnegative floats"
                    )
            else:
                raise HybridContractError(
                    f"{descriptor.name} has an invalid exact diagnostics type"
                )


@dataclass
class _DiagnosticsBuilder:
    minimum_periapses: list[float]
    maximum_eccentricities: list[float]
    maximum_interaction_ratios: list[float]
    maximum_orbit_fractions: list[float]
    maximum_periapse_fractions: list[float]
    minimum_pair_endpoints: list[float]
    minimum_path_lower_bounds: list[float]
    minimum_path_clearances: list[float]
    minimum_hill_ratios: list[float]
    maximum_barycenter_position_norm: float
    maximum_barycenter_velocity_norm: float
    initial_barycenter_position_cap: float
    initial_barycenter_velocity_cap: float
    maximum_solver_iterations: int = 0
    maximum_bracket_expansions: int = 0
    maximum_time_residual: float = 0.0
    maximum_residual_tolerance: float = 0.0
    maximum_lagrange_error: float = 0.0
    maximum_energy_error: float = 0.0
    maximum_angular_error: float = 0.0
    maximum_translation_residual: float = 0.0

    @classmethod
    def from_preparation(cls, preparation: Any, cache: Any) -> _DiagnosticsBuilder:
        guard = cache.accepted_start_guard
        if guard is None:
            raise HybridContractError("successful WH rebind lacks start-guard evidence")
        body_count = len(cache.positions)
        pair_count = body_count * (body_count - 1) // 2
        return cls(
            minimum_periapses=list(guard.periapses),
            maximum_eccentricities=list(guard.eccentricities),
            maximum_interaction_ratios=list(guard.interaction_force_ratios),
            maximum_orbit_fractions=list(guard.orbit_step_fractions),
            maximum_periapse_fractions=list(guard.periapse_step_fractions),
            minimum_pair_endpoints=list(guard.pair_separations),
            minimum_path_lower_bounds=[math.inf] * pair_count,
            minimum_path_clearances=[math.inf] * pair_count,
            minimum_hill_ratios=list(guard.secondary_hill_floor_ratios),
            maximum_barycenter_position_norm=preparation.initial_barycenter_position_norm,
            maximum_barycenter_velocity_norm=preparation.initial_barycenter_velocity_norm,
            initial_barycenter_position_cap=preparation.initial_barycenter_position_cap,
            initial_barycenter_velocity_cap=preparation.initial_barycenter_velocity_cap,
            maximum_translation_residual=cache.accepted_start_translation_residual,
        )

    def merge_guard(self, evidence: Any) -> None:
        bary_position, bary_velocity = _wh._merge_node_extrema(
            evidence=evidence,
            minimum_periapses=self.minimum_periapses,
            maximum_eccentricities=self.maximum_eccentricities,
            maximum_interaction_ratios=self.maximum_interaction_ratios,
            maximum_orbit_fractions=self.maximum_orbit_fractions,
            maximum_periapse_fractions=self.maximum_periapse_fractions,
            minimum_pair_endpoints=self.minimum_pair_endpoints,
            minimum_hill_ratios=self.minimum_hill_ratios,
        )
        self.maximum_barycenter_position_norm = max(
            self.maximum_barycenter_position_norm, bary_position
        )
        self.maximum_barycenter_velocity_norm = max(
            self.maximum_barycenter_velocity_norm, bary_velocity
        )

    def merge_rebind(self, cache: Any) -> None:
        if cache.accepted_start_guard is None:
            raise HybridContractError("WH rebind lacks accepted-start evidence")
        self.merge_guard(cache.accepted_start_guard)
        self.maximum_translation_residual = max(
            self.maximum_translation_residual,
            cache.accepted_start_translation_residual,
        )

    def merge_candidate(self, candidate: Any) -> None:
        self.maximum_translation_residual = max(
            self.maximum_translation_residual, candidate.translation_residual
        )
        self.merge_guard(candidate.drift_guard)
        self.merge_guard(candidate.completed_guard)
        for index, value in enumerate(
            candidate.path_evidence.endpoint_minimum_separations
        ):
            self.minimum_pair_endpoints[index] = min(
                self.minimum_pair_endpoints[index], value
            )
        for index, value in enumerate(candidate.path_evidence.path_lower_bounds):
            self.minimum_path_lower_bounds[index] = min(
                self.minimum_path_lower_bounds[index], value
            )
        for index, value in enumerate(
            candidate.path_evidence.clearance_after_margins
        ):
            self.minimum_path_clearances[index] = min(
                self.minimum_path_clearances[index], value
            )
        for subflow in candidate.solver_records:
            self.maximum_solver_iterations = max(
                self.maximum_solver_iterations, subflow.iterations
            )
            self.maximum_bracket_expansions = max(
                self.maximum_bracket_expansions, subflow.bracket_expansions
            )
            self.maximum_time_residual = max(
                self.maximum_time_residual, subflow.time_residual
            )
            self.maximum_residual_tolerance = max(
                self.maximum_residual_tolerance, subflow.residual_tolerance
            )
            self.maximum_lagrange_error = max(
                self.maximum_lagrange_error, subflow.lagrange_identity_error
            )
            self.maximum_energy_error = max(
                self.maximum_energy_error, subflow.energy_error
            )
            self.maximum_angular_error = max(
                self.maximum_angular_error, subflow.angular_momentum_error
            )

    def freeze(self) -> HybridWisdomHolmanDiagnostics:
        return HybridWisdomHolmanDiagnostics(
            maximum_universal_solver_iterations=self.maximum_solver_iterations,
            maximum_universal_solver_bracket_expansions=self.maximum_bracket_expansions,
            maximum_kepler_time_residual=float(self.maximum_time_residual),
            maximum_kepler_residual_tolerance=float(self.maximum_residual_tolerance),
            maximum_kepler_lagrange_identity_error=float(self.maximum_lagrange_error),
            maximum_kepler_energy_error=float(self.maximum_energy_error),
            maximum_kepler_angular_momentum_error=float(self.maximum_angular_error),
            maximum_translation_force_residual=float(
                self.maximum_translation_residual
            ),
            minimum_jacobi_periapses=tuple(float(v) for v in self.minimum_periapses),
            maximum_jacobi_eccentricities=tuple(
                float(v) for v in self.maximum_eccentricities
            ),
            maximum_interaction_force_ratios=tuple(
                float(v) for v in self.maximum_interaction_ratios
            ),
            maximum_orbit_step_fractions=tuple(
                float(v) for v in self.maximum_orbit_fractions
            ),
            maximum_periapse_step_fractions=tuple(
                float(v) for v in self.maximum_periapse_fractions
            ),
            minimum_pair_endpoint_separations=tuple(
                float(v) for v in self.minimum_pair_endpoints
            ),
            minimum_pair_path_lower_bounds=tuple(
                float(v) for v in self.minimum_path_lower_bounds
            ),
            minimum_pair_clearance_after_margins=tuple(
                float(v) for v in self.minimum_path_clearances
            ),
            minimum_secondary_hill_floor_ratios=tuple(
                float(v) for v in self.minimum_hill_ratios
            ),
            maximum_barycenter_position_norm=float(
                self.maximum_barycenter_position_norm
            ),
            maximum_barycenter_velocity_norm=float(
                self.maximum_barycenter_velocity_norm
            ),
            initial_barycenter_position_cap=float(self.initial_barycenter_position_cap),
            initial_barycenter_velocity_cap=float(self.initial_barycenter_velocity_cap),
        )


def _private_digest_maps() -> tuple[dict[str, str], dict[str, tuple[str, tuple[str, ...]]]]:
    domains = {
        field_name: domain
        for field_name, domain, _description in (
            HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER
        )
    }
    schemas = {
        field_name: (schema, keys)
        for field_name, schema, keys, _description in (
            HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER
        )
    }
    return domains, schemas


def _private_encounter_record(
    *, outer_step_index: int, segment_spec: Any, run: Any
) -> HybridPrivateEncounterRecord:
    domains, schemas = _private_digest_maps()
    resources = segment_spec.exact_rational_resources
    final_schema = schemas["final_state_content_sha256"][0]
    initialization_schema = schemas["initialization_content_sha256"][0]
    counts_schema = schemas["primary_counts_content_sha256"][0]
    final_positions = _readonly(run.final_positions)
    final_velocities = _readonly(run.final_velocities)
    component_digests: dict[str, str] = {}
    component_digests["final_state_content_sha256"] = _domain_sha256(
        domains["final_state_content_sha256"],
        {
            "schema": final_schema,
            "positions": final_positions,
            "velocities": final_velocities,
        },
    )
    component_digests["initialization_content_sha256"] = _domain_sha256(
        domains["initialization_content_sha256"],
        {"schema": initialization_schema, "record": run.initialization_record},
    )
    component_digests["proposal_ledger_content_sha256"] = (
        _streaming_sequence_sha256(
            domain=domains["proposal_ledger_content_sha256"],
            schema=schemas["proposal_ledger_content_sha256"][0],
            values=run.proposal_ledger,
            max_count=segment_spec.maximum_substep_proposals,
            max_item_bytes=(
                resources.maximum_witness_diagnostic_bytes_per_proposal
            ),
            max_cumulative_bytes=_streaming_cumulative_cap(
                domain=domains["proposal_ledger_content_sha256"],
                schema=schemas["proposal_ledger_content_sha256"][0],
                max_count=segment_spec.maximum_substep_proposals,
                max_item_payload_bytes=resources.maximum_witness_ledger_bytes,
            ),
        )
    )
    component_digests["force_ledger_content_sha256"] = _streaming_sequence_sha256(
        domain=domains["force_ledger_content_sha256"],
        schema=schemas["force_ledger_content_sha256"][0],
        values=run.force_ledger,
        max_count=1,
        max_item_bytes=resources.maximum_witness_ledger_bytes,
        max_cumulative_bytes=_streaming_cumulative_cap(
            domain=domains["force_ledger_content_sha256"],
            schema=schemas["force_ledger_content_sha256"][0],
            max_count=1,
            max_item_payload_bytes=resources.maximum_witness_ledger_bytes,
        ),
    )
    component_digests["primary_counts_content_sha256"] = _domain_sha256(
        domains["primary_counts_content_sha256"],
        {"schema": counts_schema, "counts": run.counts},
    )
    component_digests["accepted_steps_content_sha256"] = (
        _streaming_sequence_sha256(
            domain=domains["accepted_steps_content_sha256"],
            schema=schemas["accepted_steps_content_sha256"][0],
            values=run.accepted_signed_substeps,
            max_count=segment_spec.maximum_accepted_substeps,
            max_item_bytes=_MAXIMUM_CANONICAL_BINARY64_ITEM_BYTES,
            max_cumulative_bytes=_streaming_cumulative_cap(
                domain=domains["accepted_steps_content_sha256"],
                schema=schemas["accepted_steps_content_sha256"][0],
                max_count=segment_spec.maximum_accepted_substeps,
                max_item_payload_bytes=(
                    segment_spec.maximum_accepted_substeps
                    * _MAXIMUM_CANONICAL_BINARY64_ITEM_BYTES
                ),
            ),
        )
    )
    proposal_records = run.proposal_ledger
    if (
        type(proposal_records) is not tuple
        or len(proposal_records) > segment_spec.maximum_substep_proposals
    ):
        raise HybridContractError("private proposal ledger exceeds its count cap")
    maximum_integer_bits = max(
        (run.initialization_record.maximum_integer_bits,)
        + tuple(record.maximum_integer_bits for record in proposal_records)
    )
    maximum_exponent = max(
        (run.initialization_record.maximum_rational_exponent_magnitude,)
        + tuple(
            record.maximum_rational_exponent_magnitude
            for record in proposal_records
        )
    )
    witness_item_cap = resources.maximum_witness_diagnostic_bytes_per_proposal
    witness_cumulative_cap = resources.maximum_witness_ledger_bytes
    initialization_bytes = _bounded_canonical_json(
        run.initialization_record,
        max_item_bytes=witness_item_cap,
        remaining_cumulative_bytes=witness_cumulative_cap,
        label="private encounter initialization witness",
    )
    streamed_witness_bytes = len(initialization_bytes)
    for proposal_index, record in enumerate(proposal_records, start=1):
        encoded_record = _bounded_canonical_json(
            record,
            max_item_bytes=witness_item_cap,
            remaining_cumulative_bytes=(
                witness_cumulative_cap - streamed_witness_bytes
            ),
            label=f"private encounter proposal witness {proposal_index}",
        )
        streamed_witness_bytes += len(encoded_record)
    summary = {
        "proposal_count": run.counts.substep_proposals,
        "accepted_substep_count": run.counts.accepted_substeps,
        "rejected_substep_count": run.counts.rejected_substeps,
        "force_evaluations": run.counts.force_evaluations,
        "general_rational_operation_count": run.counts.total_rational_operations,
        "dyadic_operation_count": run.counts.total_dyadic_operations,
        "gcd_iteration_count": run.counts.total_gcd_iterations,
        "transcript_byte_count": run.counts.total_transcript_bytes,
        "maximum_integer_bits": maximum_integer_bits,
        "maximum_rational_exponent_magnitude": maximum_exponent,
        "streamed_witness_ledger_bytes": streamed_witness_bytes,
    }
    manifest_schema = schemas["private_execution_content_sha256"][0]
    component_digests["private_execution_content_sha256"] = _domain_sha256(
        domains["private_execution_content_sha256"],
        {
            "schema": manifest_schema,
            "method_id": segment_spec.method_id,
            "outer_step_index": outer_step_index,
            "segment_spec": segment_spec,
            "summary": summary,
            "component_digests": dict(component_digests),
        },
    )
    return HybridPrivateEncounterRecord(
        outer_step_index=outer_step_index,
        segment_spec=segment_spec,
        accepted_signed_steps=run.accepted_signed_substeps,
        **summary,
        **component_digests,
    )


@dataclass(frozen=True, eq=False)
class _HybridRun:
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    outer_step_records: tuple[HybridOuterStepRecord, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    counts: HybridLaneExecutionCounts
    wisdom_holman_diagnostics: HybridWisdomHolmanDiagnostics | None
    wisdom_holman_schedule_content_sha256: str | None
    wisdom_holman_result_content_sha256: str | None


def _make_node_snapshot(
    initial_snapshot: StateSnapshot,
    epoch: float,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> StateSnapshot:
    return replace(
        initial_snapshot,
        epoch=epoch,
        positions=_readonly(positions),
        velocities=_readonly(velocities),
    )


def _wh_run_from_hybrid(
    *,
    checkpoints: tuple[TrajectoryCheckpoint, ...],
    force_model_ids: tuple[str, ...],
    force_ledger: tuple[ForceLedgerEntry, ...],
    work: HybridProbeWorkTotals,
    diagnostics: HybridWisdomHolmanDiagnostics,
) -> Any:
    return _wh._WisdomHolmanRun(
        checkpoints=checkpoints,
        force_model_ids=force_model_ids,
        force_ledger=force_ledger,
        force_evaluations=work.force_evaluations_completed,
        interaction_force_assemblies=work.interaction_force_assemblies,
        kepler_subflow_solves=work.kepler_subflow_solves_completed,
        universal_solver_iterations=work.universal_solver_iterations,
        maximum_universal_solver_iterations=(
            diagnostics.maximum_universal_solver_iterations
        ),
        universal_solver_bracket_expansions=(
            work.universal_solver_bracket_expansions
        ),
        maximum_universal_solver_bracket_expansions=(
            diagnostics.maximum_universal_solver_bracket_expansions
        ),
        universal_g_function_evaluations=(
            work.universal_g_bundle_calls_completed
        ),
        universal_series_terms_evaluated=work.universal_series_terms_evaluated,
        coordinate_forward_transforms=(
            work.cartesian_to_jacobi_transforms_completed
        ),
        coordinate_inverse_transforms=(
            work.jacobi_to_cartesian_transforms_completed
        ),
        node_guard_evaluations=work.node_guard_evaluations,
        path_guard_evaluations=work.path_guard_evaluations,
        **{
            descriptor.name: getattr(diagnostics, descriptor.name)
            for descriptor in fields(HybridWisdomHolmanDiagnostics)
            if descriptor.name
            not in (
                "maximum_universal_solver_iterations",
                "maximum_universal_solver_bracket_expansions",
            )
        },
    )


def _execute_hybrid_lane(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: HybridWisdomHolmanRKF78Spec,
    binding: Any,
    all_epochs: tuple[float, ...],
    checkpoint_epochs: tuple[float, ...],
) -> _HybridRun:
    wh_spec = integration_spec.wisdom_holman_spec
    body_count = len(initial_snapshot.body_ids)
    accepted_work = _zero_probe_totals()
    discarded_work = _zero_probe_totals()
    encounter_counts = _zero_encounter_counts()
    records: list[HybridOuterStepRecord] = []
    near_count = 0
    near_accepted_substeps = 0
    retained_digest_bytes = 0
    current_positions = initial_snapshot.positions
    current_velocities = initial_snapshot.velocities
    cache: Any | None = None
    force_custody: Any | None = None
    force_model_ids: tuple[str, ...] | None = None
    force_ledger: tuple[ForceLedgerEntry, ...] | None = None
    diagnostics_builder: _DiagnosticsBuilder | None = None
    initial_preparation_seen = False

    checkpoints: list[TrajectoryCheckpoint] = [
        TrajectoryCheckpoint(
            index=0,
            epoch=checkpoint_epochs[0],
            body_ids=initial_snapshot.body_ids,
            backend_id="numpy",
            device="cpu",
            positions=_readonly(initial_snapshot.positions),
            velocities=_readonly(initial_snapshot.velocities),
            accepted_steps=0,
            rejected_steps=0,
        )
    ]
    next_checkpoint = 1

    for step_index in range(1, wh_spec.completed_steps + 1):
        if len(records) >= HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE:
            raise HybridStepLimitError("hybrid outer-record cap exhausted")
        start_epoch = all_epochs[step_index - 1]
        endpoint_epoch = all_epochs[step_index]
        original_positions = _readonly(current_positions)
        original_velocities = _readonly(current_velocities)

        if cache is None:
            node_snapshot = _make_node_snapshot(
                initial_snapshot,
                start_epoch,
                original_positions,
                original_velocities,
            )
            preparation = _wh._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=node_snapshot,
                force_plan=force_plan,
                integration_spec=wh_spec,
                binding=binding,
                accepted_outer_step_index=step_index - 1,
                validate_initial_barycenter=not initial_preparation_seen,
                retained_force_custody=force_custody,
                require_hybrid_decision=True,
            )
            initial_preparation_seen = True
            if preparation.terminal is None:
                if preparation.cache is None:
                    raise HybridContractError("successful WH preparation lacks a cache")
                cache = preparation.cache
                force_custody = cache.force_custody
                if force_model_ids is None:
                    force_model_ids = cache.force_model_ids
                    force_ledger = cache.force_ledger
                elif cache.force_model_ids != force_model_ids:
                    raise HybridContractError("WH force identity changed after near rebind")
                if diagnostics_builder is None:
                    diagnostics_builder = _DiagnosticsBuilder.from_preparation(
                        preparation, cache
                    )
                else:
                    diagnostics_builder.merge_rebind(cache)
                proposal = _wh._propose_wisdom_holman_macrostep(
                    cache=cache,
                    initial_snapshot=node_snapshot,
                    force_plan=force_plan,
                    integration_spec=wh_spec,
                    binding=binding,
                    expected_cache_epoch=start_epoch,
                    expected_rebind_work_pending=True,
                    candidate_epoch=endpoint_epoch,
                    candidate_outer_step_index=step_index,
                    require_hybrid_decision=True,
                )
            else:
                proposal = preparation.terminal
        else:
            node_snapshot = cache.authoritative_snapshot
            proposal = _wh._propose_wisdom_holman_macrostep(
                cache=cache,
                initial_snapshot=node_snapshot,
                force_plan=force_plan,
                integration_spec=wh_spec,
                binding=binding,
                expected_cache_epoch=start_epoch,
                expected_rebind_work_pending=False,
                candidate_epoch=endpoint_epoch,
                candidate_outer_step_index=step_index,
                require_hybrid_decision=True,
            )

        decision = proposal.decision
        if type(decision) is not HybridFarProbeDecision:
            raise HybridContractError("hybrid probe returned the wrong decision schema")
        decision.__post_init__()
        if decision.outcome == "FATAL_FAILURE":
            if proposal.public_error is None:
                raise HybridContractError("fatal WH decision lacks its typed failure")
            raise proposal.public_error

        if decision.outcome == "FAR_PASS":
            if cache is None or proposal.candidate is None:
                raise HybridContractError("FAR_PASS lacks a cache/candidate transaction")
            cache = _wh._commit_wisdom_holman_macrostep(
                cache=cache,
                proposal=proposal,
                initial_snapshot=node_snapshot,
                force_plan=force_plan,
                integration_spec=wh_spec,
                binding=binding,
                expected_cache_epoch=start_epoch,
                expected_rebind_work_pending=cache.rebind_work_pending,
                expected_candidate_epoch=endpoint_epoch,
                expected_outer_step_index=step_index,
            )
            current_positions = cache.positions
            current_velocities = cache.velocities
            accepted_work = _sum_probe_totals(
                accepted_work, _probe_totals(decision.work)
            )
            if diagnostics_builder is None:
                raise HybridContractError("committed WH candidate lacks diagnostics")
            diagnostics_builder.merge_candidate(proposal.candidate)
            record = HybridOuterStepRecord(
                outer_step_index=step_index,
                start_epoch=start_epoch,
                endpoint_epoch=endpoint_epoch,
                signed_step=wh_spec.fixed_step,
                mode="WISDOM_HOLMAN_FAR",
                probe_committed=True,
                decision=decision,
                private_encounter=None,
            )
        elif decision.outcome == "NEAR_SWITCH":
            if near_count >= HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE:
                raise HybridStepLimitError("hybrid near-macrostep cap exhausted")
            if proposal.candidate is not None:
                raise HybridContractError("NEAR_SWITCH leaked a provisional candidate")
            discarded_work = _sum_probe_totals(
                discarded_work, _probe_totals(decision.work)
            )
            child_spec = _bind_encounter_segment(
                profile=integration_spec.encounter_control,
                body_order=integration_spec.body_order,
                initial_epoch=start_epoch,
                endpoint_epoch=endpoint_epoch,
                duration=wh_spec.fixed_step,
            )
            child_snapshot = _make_node_snapshot(
                initial_snapshot,
                start_epoch,
                original_positions,
                original_velocities,
            )
            child_run = _enc._execute(child_snapshot, force_plan, child_spec)
            child_record = _private_encounter_record(
                outer_step_index=step_index,
                segment_spec=child_spec,
                run=child_run,
            )
            prospective_substeps = (
                near_accepted_substeps + child_record.accepted_substep_count
            )
            prospective_digest_bytes = retained_digest_bytes + _PRIVATE_DIGEST_BYTES_PER_NEAR
            if (
                prospective_substeps
                > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE
                or prospective_digest_bytes
                > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE
            ):
                raise HybridStepLimitError("hybrid retained near-child cap exhausted")
            near_count += 1
            near_accepted_substeps = prospective_substeps
            retained_digest_bytes = prospective_digest_bytes
            encounter_counts = _sum_encounter_counts(
                encounter_counts, child_run.counts
            )
            if force_model_ids is None:
                force_model_ids = child_run.force_model_ids
                force_ledger = child_run.force_ledger
            elif child_run.force_model_ids != force_model_ids:
                raise HybridContractError("near child changed force-model identity")
            current_positions = _readonly(child_run.final_positions)
            current_velocities = _readonly(child_run.final_velocities)
            if cache is not None:
                force_custody = cache.force_custody
            cache = None
            record = HybridOuterStepRecord(
                outer_step_index=step_index,
                start_epoch=start_epoch,
                endpoint_epoch=endpoint_epoch,
                signed_step=wh_spec.fixed_step,
                mode="CARTESIAN_RKF78_NEAR_FULL_INTERVAL",
                probe_committed=False,
                decision=decision,
                private_encounter=child_record,
            )
        else:
            raise HybridContractError("probe outcome is outside the closed hybrid roster")

        records.append(record)
        if (
            next_checkpoint < len(wh_spec.checkpoint_step_indices)
            and step_index == wh_spec.checkpoint_step_indices[next_checkpoint]
        ):
            checkpoints.append(
                TrajectoryCheckpoint(
                    index=next_checkpoint,
                    epoch=checkpoint_epochs[next_checkpoint],
                    body_ids=initial_snapshot.body_ids,
                    backend_id="numpy",
                    device="cpu",
                    positions=_readonly(current_positions),
                    velocities=_readonly(current_velocities),
                    accepted_steps=step_index,
                    rejected_steps=0,
                )
            )
            next_checkpoint += 1

    if next_checkpoint != len(wh_spec.checkpoint_step_indices):
        raise HybridContractError("hybrid did not materialize every outer checkpoint")
    if force_model_ids is None or force_ledger is None:
        raise HybridContractError("hybrid execution retained no force identity")
    # A switched probe can terminate before complete path evidence exists.  The
    # public-WH diagnostic projection is therefore retained only for the exact
    # all-far case where every frozen diagnostic lane is complete.
    diagnostics = (
        diagnostics_builder.freeze()
        if near_count == 0 and diagnostics_builder is not None
        else None
    )
    total_probe_work = _sum_probe_totals(accepted_work, discarded_work)
    counts = HybridLaneExecutionCounts(
        outer_records=len(records),
        far_probe_count=len(records),
        far_accepted_count=len(records) - near_count,
        far_discarded_count=near_count,
        private_encounter_execution_count=near_count,
        near_accepted_substeps=near_accepted_substeps,
        retained_near_digest_bytes=retained_digest_bytes,
        checkpoint_count=len(checkpoints),
        accepted_wh_work=accepted_work,
        discarded_wh_work=discarded_work,
        total_probe_work=total_probe_work,
        private_encounter_counts=encounter_counts,
    )

    wh_schedule_sha256: str | None = None
    wh_result_sha256: str | None = None
    if near_count == 0:
        if diagnostics is None:
            raise HybridContractError("all-far execution lacks WH diagnostics")
        wh_run = _wh_run_from_hybrid(
            checkpoints=tuple(checkpoints),
            force_model_ids=force_model_ids,
            force_ledger=force_ledger,
            work=accepted_work,
            diagnostics=diagnostics,
        )
        wh_schedule_sha256 = _wh._schedule_content_sha256(
            checkpoint_step_indices=wh_spec.checkpoint_step_indices,
            checkpoint_epochs=checkpoint_epochs,
            fixed_step=wh_spec.fixed_step,
            direction=wh_spec.direction,
            completed_steps=wh_spec.completed_steps,
            body_count=body_count,
        )
        wh_result_sha256 = _wh._result_content_sha256(
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=wh_spec,
            coordinate_binding=binding,
            checkpoint_step_indices=wh_spec.checkpoint_step_indices,
            checkpoint_epochs=checkpoint_epochs,
            run=wh_run,
            direction=wh_spec.direction,
            schedule_content_sha256=wh_schedule_sha256,
        )
    return _HybridRun(
        checkpoints=tuple(checkpoints),
        outer_step_records=tuple(records),
        force_model_ids=force_model_ids,
        force_ledger=force_ledger,
        counts=counts,
        wisdom_holman_diagnostics=diagnostics,
        wisdom_holman_schedule_content_sha256=wh_schedule_sha256,
        wisdom_holman_result_content_sha256=wh_result_sha256,
    )


def _schedule_sha256(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: HybridWisdomHolmanRKF78Spec,
    coordinate_binding: Any,
    all_epochs: tuple[float, ...],
    checkpoint_epochs: tuple[float, ...],
) -> str:
    return _domain_sha256(
        HYBRID_SCHEDULE_CHECKSUM_DOMAIN,
        {
            "schema": _HYBRID_SCHEDULE_SCHEMA,
            "snapshot_id": initial_snapshot.snapshot_id,
            "plan_id": force_plan.plan_id,
            "integration_spec": integration_spec,
            "coordinate_binding": coordinate_binding,
            "outer_epochs": all_epochs,
            "checkpoint_step_indices": (
                integration_spec.wisdom_holman_spec.checkpoint_step_indices
            ),
            "checkpoint_epochs": checkpoint_epochs,
        },
    )


def _step_ledger_sha256(records: tuple[HybridOuterStepRecord, ...]) -> str:
    return _domain_sha256(
        HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN,
        {"schema": _HYBRID_STEP_LEDGER_SCHEMA, "records": records},
    )


def _result_sha256(
    *,
    initial_snapshot: StateSnapshot,
    force_plan: ForcePlan,
    integration_spec: HybridWisdomHolmanRKF78Spec,
    coordinate_binding: Any,
    all_epochs: tuple[float, ...],
    checkpoint_epochs: tuple[float, ...],
    run: _HybridRun,
    primary_counts: HybridLaneExecutionCounts,
    replay_counts: HybridLaneExecutionCounts,
    total_counts: HybridLaneExecutionCounts,
    schedule_sha256: str,
    step_ledger_sha256: str,
) -> str:
    return _domain_sha256(
        HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN,
        {
            "schema": _HYBRID_RESULT_SCHEMA,
            "initial_snapshot": initial_snapshot,
            "force_plan": force_plan,
            "integration_spec": integration_spec,
            "coordinate_binding": coordinate_binding,
            "outer_epochs": all_epochs,
            "checkpoint_epochs": checkpoint_epochs,
            "runtime": run,
            "primary_counts": primary_counts,
            "validation_replay_counts": replay_counts,
            "total_public_call_counts": total_counts,
            "schedule_content_sha256": schedule_sha256,
            "step_ledger_content_sha256": step_ledger_sha256,
            "claim_scope": HYBRID_CLAIM_SCOPE,
            "nonclaims": HYBRID_NONCLAIMS,
        },
    )


@dataclass(frozen=True, eq=False)
class HybridWisdomHolmanRKF78Result:
    """Owned hybrid trajectory with exact mode, replay, and work custody."""

    snapshot_id: str
    plan_id: str
    backend_id: str
    device: str
    dtype: str
    backend_spec: BackendSpec
    initial_snapshot: StateSnapshot
    force_plan: ForcePlan
    integration_spec: HybridWisdomHolmanRKF78Spec
    coordinate_binding: _wh.JacobiCoordinateBinding
    outer_epochs: tuple[float, ...]
    checkpoint_step_indices: tuple[int, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    outer_step_records: tuple[HybridOuterStepRecord, ...]
    force_model_ids: tuple[str, ...]
    force_ledger: tuple[ForceLedgerEntry, ...]
    primary_counts: HybridLaneExecutionCounts
    validation_replay_counts: HybridLaneExecutionCounts
    total_public_call_counts: HybridLaneExecutionCounts
    wisdom_holman_diagnostics: HybridWisdomHolmanDiagnostics | None
    wisdom_holman_schedule_content_sha256: str | None
    wisdom_holman_result_content_sha256: str | None
    schedule_content_sha256: str
    step_ledger_content_sha256: str
    result_content_sha256: str
    method_id: str = HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID
    validation_replay_policy: str = HYBRID_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = HYBRID_VALIDATION_REPLAY_COUNT
    schedule_checksum_algorithm: str = HYBRID_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = HYBRID_SCHEDULE_CHECKSUM_DOMAIN
    step_ledger_checksum_algorithm: str = HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM
    step_ledger_checksum_domain: str = HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN
    private_encounter_checksum_algorithm: str = (
        HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM
    )
    result_content_checksum_algorithm: str = (
        HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM
    )
    result_content_checksum_domain: str = HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN
    claim_scope: str = HYBRID_CLAIM_SCOPE
    nonclaims: str = HYBRID_NONCLAIMS
    outer_step_adaptive: bool = False
    encounter_substeps_adaptive: bool = True
    symplectic: bool = False
    time_reversible: bool = False
    dense_output: bool = False
    event_detection: bool = False
    collision_detection: bool = False
    collision_response: bool = False
    regularized: bool = False
    global_clearance_claimed: bool = False
    global_order_claimed: bool = False
    superiority_claimed: bool = False
    evidence_class: str = HYBRID_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        _validate_result(self)

    @property
    def completed_steps(self) -> int:
        return self.integration_spec.completed_steps

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    @property
    def direction(self) -> str:
        return self.integration_spec.direction

    @property
    def final_epoch(self) -> float:
        return self.checkpoint_epochs[-1]

    @property
    def positions(self) -> tuple[np.ndarray, ...]:
        return tuple(checkpoint.positions for checkpoint in self.checkpoints)

    @property
    def velocities(self) -> tuple[np.ndarray, ...]:
        return tuple(checkpoint.velocities for checkpoint in self.checkpoints)

    @property
    def final_positions(self) -> np.ndarray:
        return self.checkpoints[-1].positions

    @property
    def final_velocities(self) -> np.ndarray:
        return self.checkpoints[-1].velocities

    @property
    def all_far(self) -> bool:
        return self.primary_counts.far_discarded_count == 0

    @property
    def integrated(self) -> bool:
        return True

    @property
    def qualified(self) -> bool:
        return False


def _run_from_result(result: HybridWisdomHolmanRKF78Result) -> _HybridRun:
    return _HybridRun(
        checkpoints=result.checkpoints,
        outer_step_records=result.outer_step_records,
        force_model_ids=result.force_model_ids,
        force_ledger=result.force_ledger,
        counts=result.primary_counts,
        wisdom_holman_diagnostics=result.wisdom_holman_diagnostics,
        wisdom_holman_schedule_content_sha256=(
            result.wisdom_holman_schedule_content_sha256
        ),
        wisdom_holman_result_content_sha256=(
            result.wisdom_holman_result_content_sha256
        ),
    )


def _validate_result(result: HybridWisdomHolmanRKF78Result) -> None:
    if type(result) is not HybridWisdomHolmanRKF78Result:
        raise HybridContractError("result must be exact HybridWisdomHolmanRKF78Result")
    exact = {
        "backend_id": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "method_id": HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
        "validation_replay_policy": HYBRID_VALIDATION_REPLAY_POLICY,
        "validation_replay_count": HYBRID_VALIDATION_REPLAY_COUNT,
        "schedule_checksum_algorithm": HYBRID_SCHEDULE_CHECKSUM_ALGORITHM,
        "schedule_checksum_domain": HYBRID_SCHEDULE_CHECKSUM_DOMAIN,
        "step_ledger_checksum_algorithm": HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM,
        "step_ledger_checksum_domain": HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN,
        "private_encounter_checksum_algorithm": HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM,
        "result_content_checksum_algorithm": HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM,
        "result_content_checksum_domain": HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN,
        "claim_scope": HYBRID_CLAIM_SCOPE,
        "nonclaims": HYBRID_NONCLAIMS,
        "outer_step_adaptive": False,
        "encounter_substeps_adaptive": True,
        "symplectic": False,
        "time_reversible": False,
        "dense_output": False,
        "event_detection": False,
        "collision_detection": False,
        "collision_response": False,
        "regularized": False,
        "global_clearance_claimed": False,
        "global_order_claimed": False,
        "superiority_claimed": False,
        "evidence_class": HYBRID_EVIDENCE_CLASS,
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    for name, expected in exact.items():
        value = getattr(result, name)
        if type(value) is not type(expected) or value != expected:
            raise HybridContractError(f"{name} differs from its frozen hybrid value")
    if type(result.snapshot_id) is not str or result.snapshot_id != result.initial_snapshot.snapshot_id:
        raise HybridContractError("snapshot_id differs from retained initial snapshot")
    if type(result.plan_id) is not str or result.plan_id != result.force_plan.plan_id:
        raise HybridContractError("plan_id differs from retained force plan")
    if type(result.integration_spec) is not HybridWisdomHolmanRKF78Spec:
        raise HybridContractError("integration_spec must have its exact hybrid type")
    result.integration_spec.__post_init__()
    _wh._validate_snapshot_schema(result.initial_snapshot, "initial_snapshot")
    _wh._validate_force_plan_schema(result.force_plan, "force_plan")
    _wh._validate_backend_schema(result.backend_spec, "backend_spec")
    if type(result.backend_spec) is not BackendSpec or result.backend_spec is not result.force_plan.backend:
        raise HybridContractError("backend_spec must be the retained force-plan backend")
    if type(result.coordinate_binding) is not _wh.JacobiCoordinateBinding:
        raise HybridContractError("coordinate_binding must have its exact WH type")
    result.coordinate_binding.__post_init__()
    if (
        result.coordinate_binding.body_ids != result.initial_snapshot.body_ids
        or result.coordinate_binding.gravitational_parameters.tobytes(order="C")
        != result.initial_snapshot.gravitational_parameters.tobytes(order="C")
    ):
        raise HybridContractError("coordinate binding differs from retained input")
    _wh._validate_semantic_boundary(result.initial_snapshot)
    retained_model = _wh._validate_force_plan_scope(
        result.initial_snapshot,
        result.force_plan,
        result.integration_spec.wisdom_holman_spec,
    )
    _wh._validate_parameter_metadata_schema(retained_model)
    _wh._validate_model_sequence(result.force_plan.models)
    _wh._validate_dependencies(result.force_plan.models)
    _wh._validate_parameter_contracts(result.initial_snapshot, result.force_plan)
    backend = resolve_backend(result.backend_spec)
    if backend.name != "numpy" or backend.device != "cpu":
        raise HybridContractError("retained hybrid backend is outside NumPy CPU")
    with backend.activate():
        retained_state = _wh._validate_state(backend, result.initial_snapshot)
        if bool(np.any(~retained_state[5])) or bool(
            np.any(retained_state[2] <= np.float64(0.0))
        ):
            raise HybridContractError("retained hybrid state lost active positive-GM scope")
    secondary_ratio = _wh._sum_1d_fixed(retained_state[2], 1) / float(
        retained_state[2][0]
    )
    if (
        not math.isfinite(secondary_ratio)
        or secondary_ratio > _wh.WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
    ):
        raise HybridContractError("retained hybrid state exceeds the WH mass-ratio scope")
    if (
        type(result.outer_epochs) is not tuple
        or len(result.outer_epochs) != result.completed_steps + 1
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in result.outer_epochs
        )
    ):
        raise HybridContractError("outer_epochs has an invalid bounded length")
    if type(result.outer_step_records) is not tuple or len(result.outer_step_records) != result.completed_steps:
        raise HybridContractError("outer_step_records has an invalid bounded length")
    if len(result.outer_step_records) > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE:
        raise HybridContractError("outer_step_records exceeds its hard cap")
    expected_checkpoint_count = len(
        result.integration_spec.wisdom_holman_spec.checkpoint_step_indices
    )
    for name in (
        "checkpoint_step_indices",
        "checkpoint_epochs",
        "checkpoints",
    ):
        value = getattr(result, name)
        if type(value) is not tuple or len(value) != expected_checkpoint_count:
            raise HybridContractError(
                f"{name} must be an exact tuple with its expected bounded length"
            )
    # The exact, bounded tuple lengths above are established before inspecting
    # any caller-controlled nested checkpoint or label entry.
    if any(type(value) is not int for value in result.checkpoint_step_indices):
        raise HybridContractError("checkpoint indices require exact integers")
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in result.checkpoint_epochs
    ):
        raise HybridContractError("checkpoint epochs require finite built-in floats")
    if any(type(value) is not TrajectoryCheckpoint for value in result.checkpoints):
        raise HybridContractError("checkpoints require their exact public type")
    if result.checkpoint_step_indices != result.integration_spec.wisdom_holman_spec.checkpoint_step_indices:
        raise HybridContractError("checkpoint indices differ from the WH schedule")
    all_epochs = _wh._step_epochs(
        result.initial_snapshot.epoch, result.integration_spec.wisdom_holman_spec
    )
    checkpoint_epochs = tuple(
        all_epochs[index] for index in result.checkpoint_step_indices
    )
    _wh._validate_metadata_range(
        result.force_plan,
        min(all_epochs[0], all_epochs[-1]),
        max(all_epochs[0], all_epochs[-1]),
    )
    if not _same_content(result.outer_epochs, all_epochs) or not _same_content(
        result.checkpoint_epochs, checkpoint_epochs
    ):
        raise HybridContractError("retained labels differ from the global integer lattice")
    for output_index, (checkpoint, step_index, epoch) in enumerate(
        zip(
            result.checkpoints,
            result.checkpoint_step_indices,
            result.checkpoint_epochs,
        )
    ):
        _wh._validate_checkpoint_schema(checkpoint, f"checkpoints[{output_index}]")
        if (
            type(checkpoint) is not TrajectoryCheckpoint
            or checkpoint.index != output_index
            or type(checkpoint.epoch) is not float
            or not _same_float(float(checkpoint.epoch), epoch)
            or checkpoint.accepted_steps != step_index
            or checkpoint.rejected_steps != 0
            or checkpoint.body_ids != result.initial_snapshot.body_ids
            or checkpoint.backend_id != "numpy"
            or checkpoint.device != "cpu"
            or checkpoint.dtype != "float64"
            or checkpoint.evidence_class != HYBRID_EVIDENCE_CLASS
            or checkpoint.registry_authorized is not False
            or checkpoint.qualification_authorized is not False
        ):
            raise HybridContractError("checkpoint lost its outer-lattice binding")
    if (
        result.checkpoints[0].positions.tobytes(order="C")
        != result.initial_snapshot.positions.tobytes(order="C")
        or result.checkpoints[0].velocities.tobytes(order="C")
        != result.initial_snapshot.velocities.tobytes(order="C")
    ):
        raise HybridContractError("initial checkpoint differs from initial_snapshot")
    for expected_index, record in enumerate(result.outer_step_records, start=1):
        if type(record) is not HybridOuterStepRecord:
            raise HybridContractError("outer ledger entries require their exact type")
        record.__post_init__()
        if (
            record.outer_step_index != expected_index
            or not _same_float(record.start_epoch, all_epochs[expected_index - 1])
            or not _same_float(record.endpoint_epoch, all_epochs[expected_index])
            or not _same_float(
                record.signed_step,
                result.integration_spec.wisdom_holman_spec.fixed_step,
            )
            or record.decision.body_count != len(result.initial_snapshot.body_ids)
        ):
            raise HybridContractError("outer record lost its lattice/body binding")
        if record.private_encounter is not None:
            child = record.private_encounter.segment_spec
            if (
                child.body_order != result.integration_spec.body_order
                or not _same_float(child.initial_epoch, record.start_epoch)
                or not _same_float(child.endpoint_epoch, record.endpoint_epoch)
                or not _same_float(child.duration, record.signed_step)
            ):
                raise HybridContractError("private child lost its outer-interval binding")
    for name in (
        "primary_counts",
        "validation_replay_counts",
        "total_public_call_counts",
    ):
        value = getattr(result, name)
        if type(value) is not HybridLaneExecutionCounts:
            raise HybridContractError(f"{name} must have exact lane-count type")
        value.__post_init__()
    if not _same_content(result.primary_counts, result.validation_replay_counts):
        raise HybridContractError("mandatory replay counts differ from primary")
    for lane_name, lane in (
        ("primary", result.primary_counts),
        ("validation replay", result.validation_replay_counts),
    ):
        if (
            lane.outer_records > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
            or lane.far_discarded_count
            > HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE
            or lane.near_accepted_substeps
            > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE
            or lane.retained_near_digest_bytes
            > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE
        ):
            raise HybridContractError(f"{lane_name} hybrid lane exceeds a hard cap")
    expected_total = _sum_lane_counts(result.primary_counts, result.validation_replay_counts)
    if not _same_content(result.total_public_call_counts, expected_total):
        raise HybridContractError("public-total hybrid accounting does not add exactly")
    if (
        result.total_public_call_counts.outer_records
        > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL
        or result.total_public_call_counts.far_discarded_count
        > HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL
        or result.total_public_call_counts.near_accepted_substeps
        > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL
        or result.total_public_call_counts.retained_near_digest_bytes
        > HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL
    ):
        raise HybridContractError("public-total hybrid accounting exceeds a hard cap")
    expected_near = sum(
        record.mode == "CARTESIAN_RKF78_NEAR_FULL_INTERVAL"
        for record in result.outer_step_records
    )
    expected_accepted_work = _zero_probe_totals()
    expected_discarded_work = _zero_probe_totals()
    expected_near_substeps = 0
    for record in result.outer_step_records:
        if record.probe_committed:
            expected_accepted_work = _sum_probe_totals(
                expected_accepted_work, _probe_totals(record.decision.work)
            )
        else:
            expected_discarded_work = _sum_probe_totals(
                expected_discarded_work, _probe_totals(record.decision.work)
            )
            assert record.private_encounter is not None
            expected_near_substeps += record.private_encounter.accepted_substep_count
    if (
        result.primary_counts.outer_records != len(result.outer_step_records)
        or result.primary_counts.checkpoint_count != len(result.checkpoints)
        or result.primary_counts.far_discarded_count != expected_near
        or result.primary_counts.near_accepted_substeps != expected_near_substeps
        or result.primary_counts.retained_near_digest_bytes
        != expected_near * _PRIVATE_DIGEST_BYTES_PER_NEAR
        or not _same_content(
            result.primary_counts.accepted_wh_work, expected_accepted_work
        )
        or not _same_content(
            result.primary_counts.discarded_wh_work, expected_discarded_work
        )
    ):
        raise HybridContractError("retained step ledger and primary accounting differ")
    if type(result.force_model_ids) is not tuple or type(result.force_ledger) is not tuple:
        raise HybridContractError("force custody requires exact tuples")
    if len(result.force_model_ids) != 1 or len(result.force_ledger) != 1:
        raise HybridContractError(
            "hybrid force custody must retain exactly one Newtonian model entry"
        )
    for value in result.force_model_ids:
        if type(value) is not str or not value:
            raise HybridContractError("force model IDs must be built-in nonempty strings")
    for index, value in enumerate(result.force_ledger):
        entry = _wh._validate_ledger_schema(value)
        metadata = entry.state_metadata
        if (
            entry.order != index
            or entry.model_id != result.force_model_ids[index]
            or entry.source_ids != result.initial_snapshot.body_ids
            or entry.target_ids != result.initial_snapshot.body_ids
            or metadata.snapshot_id != result.initial_snapshot.snapshot_id
            or not _same_float(metadata.epoch, result.initial_snapshot.epoch)
            or metadata.body_ids != result.initial_snapshot.body_ids
        ):
            raise HybridContractError("retained force ledger lost its input binding")
    if expected_near == 0:
        if type(result.wisdom_holman_diagnostics) is not HybridWisdomHolmanDiagnostics:
            raise HybridContractError("all-far result lacks exact WH diagnostics")
        diagnostics = result.wisdom_holman_diagnostics
        body_count = len(result.initial_snapshot.body_ids)
        secondary_count = body_count - 1
        pair_count = body_count * (body_count - 1) // 2
        hill_pair_count = secondary_count * (secondary_count - 1) // 2
        for name, length in (
            ("minimum_jacobi_periapses", secondary_count),
            ("maximum_jacobi_eccentricities", secondary_count),
            ("maximum_interaction_force_ratios", secondary_count),
            ("maximum_orbit_step_fractions", secondary_count),
            ("maximum_periapse_step_fractions", secondary_count),
            ("minimum_pair_endpoint_separations", pair_count),
            ("minimum_pair_path_lower_bounds", pair_count),
            ("minimum_pair_clearance_after_margins", pair_count),
            ("minimum_secondary_hill_floor_ratios", hill_pair_count),
        ):
            value = getattr(diagnostics, name)
            if type(value) is not tuple or len(value) != length:
                raise HybridContractError("all-far WH diagnostic roster changed")
        # Expected body/pair cardinalities are checked before the diagnostics
        # validator traverses any retained scalar-evidence tuple.
        diagnostics.__post_init__()
        _sha256(
            result.wisdom_holman_schedule_content_sha256,
            "wisdom_holman_schedule_content_sha256",
        )
        _sha256(
            result.wisdom_holman_result_content_sha256,
            "wisdom_holman_result_content_sha256",
        )
        projected_run = _wh_run_from_hybrid(
            checkpoints=result.checkpoints,
            force_model_ids=result.force_model_ids,
            force_ledger=result.force_ledger,
            work=result.primary_counts.accepted_wh_work,
            diagnostics=result.wisdom_holman_diagnostics,
        )
        projected_schedule = _wh._schedule_content_sha256(
            checkpoint_step_indices=result.checkpoint_step_indices,
            checkpoint_epochs=result.checkpoint_epochs,
            fixed_step=result.integration_spec.wisdom_holman_spec.fixed_step,
            direction=result.integration_spec.direction,
            completed_steps=result.completed_steps,
            body_count=len(result.initial_snapshot.body_ids),
        )
        projected_result = _wh._result_content_sha256(
            initial_snapshot=result.initial_snapshot,
            force_plan=result.force_plan,
            integration_spec=result.integration_spec.wisdom_holman_spec,
            coordinate_binding=result.coordinate_binding,
            checkpoint_step_indices=result.checkpoint_step_indices,
            checkpoint_epochs=result.checkpoint_epochs,
            run=projected_run,
            direction=result.integration_spec.direction,
            schedule_content_sha256=projected_schedule,
        )
        if (
            result.wisdom_holman_schedule_content_sha256 != projected_schedule
            or result.wisdom_holman_result_content_sha256 != projected_result
        ):
            raise HybridContractError("all-far public-WH projection checksum mismatch")
    elif (
        result.wisdom_holman_diagnostics is not None
        or result.wisdom_holman_schedule_content_sha256 is not None
        or result.wisdom_holman_result_content_sha256 is not None
    ):
        raise HybridContractError("mixed-mode result cannot claim a complete WH projection")
    _wh._validate_owned_readonly_arrays(
        result.initial_snapshot, result.coordinate_binding, result.checkpoints
    )
    expected_schedule = _schedule_sha256(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        coordinate_binding=result.coordinate_binding,
        all_epochs=all_epochs,
        checkpoint_epochs=checkpoint_epochs,
    )
    expected_step = _step_ledger_sha256(result.outer_step_records)
    _sha256(result.schedule_content_sha256, "schedule_content_sha256")
    _sha256(result.step_ledger_content_sha256, "step_ledger_content_sha256")
    _sha256(result.result_content_sha256, "result_content_sha256")
    if result.schedule_content_sha256 != expected_schedule:
        raise HybridContractError("hybrid schedule checksum mismatch")
    if result.step_ledger_content_sha256 != expected_step:
        raise HybridContractError("hybrid step-ledger checksum mismatch")
    run = _run_from_result(result)
    expected_result = _result_sha256(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        coordinate_binding=result.coordinate_binding,
        all_epochs=all_epochs,
        checkpoint_epochs=checkpoint_epochs,
        run=run,
        primary_counts=result.primary_counts,
        replay_counts=result.validation_replay_counts,
        total_counts=result.total_public_call_counts,
        schedule_sha256=expected_schedule,
        step_ledger_sha256=expected_step,
    )
    if result.result_content_sha256 != expected_result:
        raise HybridContractError("hybrid result checksum mismatch")
    replay = _execute_hybrid_lane(
        initial_snapshot=result.initial_snapshot,
        force_plan=result.force_plan,
        integration_spec=result.integration_spec,
        binding=result.coordinate_binding,
        all_epochs=all_epochs,
        checkpoint_epochs=checkpoint_epochs,
    )
    if not _same_content(run, replay):
        raise HybridContractError("mandatory full hybrid replay differs from primary")


def integrate_hybrid_wisdom_holman_rkf78_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: HybridWisdomHolmanRKF78Spec,
) -> HybridWisdomHolmanRKF78Result:
    """Integrate one fixed outer lattice with transactional whole-step switching."""

    if type(snapshot) is not StateSnapshot:
        raise HybridContractError("snapshot must be an exact StateSnapshot")
    if type(plan) is not ForcePlan:
        raise HybridContractError("plan must be an exact ForcePlan")
    if type(spec) is not HybridWisdomHolmanRKF78Spec:
        raise HybridContractError("spec must be exact HybridWisdomHolmanRKF78Spec")
    _wh._validate_snapshot_schema(snapshot, "snapshot")
    _wh._validate_force_plan_schema(plan, "plan")
    spec.__post_init__()
    wh_spec = spec.wisdom_holman_spec
    _wh._validate_semantic_boundary(snapshot)
    model = _wh._validate_force_plan_scope(snapshot, plan, wh_spec)
    _wh._validate_parameter_metadata_schema(model)
    _wh._validate_model_sequence(plan.models)
    _wh._validate_dependencies(plan.models)
    _wh._validate_parameter_contracts(snapshot, plan)
    backend = resolve_backend(plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise HybridContractError("hybrid v1 requires the exact NumPy CPU backend")
    for name in (
        "positions",
        "velocities",
        "gravitational_parameters",
        "masses",
        "radii",
        "massive",
    ):
        if type(getattr(snapshot, name)) is not np.ndarray:
            raise HybridContractError(f"hybrid input {name} must be exact ndarray")
    all_epochs = _wh._step_epochs(snapshot.epoch, wh_spec)
    checkpoint_epochs = tuple(
        all_epochs[index] for index in wh_spec.checkpoint_step_indices
    )
    _wh._validate_metadata_range(
        plan, min(all_epochs[0], all_epochs[-1]), max(all_epochs[0], all_epochs[-1])
    )
    with backend.activate():
        validated_state = _wh._validate_state(backend, snapshot)
        if bool(np.any(~validated_state[5])):
            raise HybridContractError("hybrid requires every body active/massive")
        if bool(np.any(validated_state[2] <= np.float64(0.0))):
            raise HybridContractError("hybrid requires positive GM for every body")
        total_secondary_ratio = _wh._sum_1d_fixed(validated_state[2], 1) / float(
            validated_state[2][0]
        )
        if (
            not math.isfinite(total_secondary_ratio)
            or total_secondary_ratio > _wh.WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
        ):
            raise HybridDomainError(
                "total secondary-to-primary GM ratio exceeds the WH v1 envelope"
            )
        initial_snapshot = _wh._readonly_snapshot_copy(snapshot, validated_state)
        force_plan = replace(
            plan,
            backend=replace(plan.backend),
            models=(replace(plan.models[0]),),
        )
        integration_spec = replace(
            spec,
            wisdom_holman_spec=replace(
                wh_spec, kepler_solver=replace(wh_spec.kepler_solver)
            ),
            encounter_control=replace(
                spec.encounter_control,
                exact_rational_resources=replace(
                    spec.encounter_control.exact_rational_resources
                ),
            ),
        )
        binding = _wh._build_jacobi_coordinate_binding(
            initial_snapshot.body_ids,
            initial_snapshot.gravitational_parameters,
        )
        primary = _execute_hybrid_lane(
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
            all_epochs=all_epochs,
            checkpoint_epochs=checkpoint_epochs,
        )
    replay_counts = primary.counts
    total_counts = _sum_lane_counts(primary.counts, replay_counts)
    schedule_sha256 = _schedule_sha256(
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        coordinate_binding=binding,
        all_epochs=all_epochs,
        checkpoint_epochs=checkpoint_epochs,
    )
    step_sha256 = _step_ledger_sha256(primary.outer_step_records)
    result_sha256 = _result_sha256(
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        coordinate_binding=binding,
        all_epochs=all_epochs,
        checkpoint_epochs=checkpoint_epochs,
        run=primary,
        primary_counts=primary.counts,
        replay_counts=replay_counts,
        total_counts=total_counts,
        schedule_sha256=schedule_sha256,
        step_ledger_sha256=step_sha256,
    )
    return HybridWisdomHolmanRKF78Result(
        snapshot_id=initial_snapshot.snapshot_id,
        plan_id=force_plan.plan_id,
        backend_id="numpy",
        device="cpu",
        dtype="float64",
        backend_spec=force_plan.backend,
        initial_snapshot=initial_snapshot,
        force_plan=force_plan,
        integration_spec=integration_spec,
        coordinate_binding=binding,
        outer_epochs=all_epochs,
        checkpoint_step_indices=wh_spec.checkpoint_step_indices,
        checkpoint_epochs=checkpoint_epochs,
        checkpoints=primary.checkpoints,
        outer_step_records=primary.outer_step_records,
        force_model_ids=primary.force_model_ids,
        force_ledger=primary.force_ledger,
        primary_counts=primary.counts,
        validation_replay_counts=replay_counts,
        total_public_call_counts=total_counts,
        wisdom_holman_diagnostics=primary.wisdom_holman_diagnostics,
        wisdom_holman_schedule_content_sha256=(
            primary.wisdom_holman_schedule_content_sha256
        ),
        wisdom_holman_result_content_sha256=(
            primary.wisdom_holman_result_content_sha256
        ),
        schedule_content_sha256=schedule_sha256,
        step_ledger_content_sha256=step_sha256,
        result_content_sha256=result_sha256,
    )


__all__ = [
    "HybridContractError",
    "HybridDomainError",
    "HybridError",
    "HybridLaneExecutionCounts",
    "HybridOuterStepRecord",
    "HybridProbeWorkTotals",
    "HybridStepLimitError",
    "HybridWisdomHolmanDiagnostics",
    "HybridWisdomHolmanRKF78Result",
    "integrate_hybrid_wisdom_holman_rkf78_trajectory",
]
