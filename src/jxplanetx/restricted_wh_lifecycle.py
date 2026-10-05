"""Fixed-lane terminal lifecycle for the experimental restricted-WH map.

Active tracers use the transactional CUDA/DOP853 hybrid scheduler.  A first
source-surface contact or outward primary-boundary crossing freezes that lane
at the continuously located event, records an immutable removal witness, and
allows every other lane to continue.  Lanes are never compacted in the public
state, so tracer identity and output ordering remain stable.

This direct-import M1 component has no disk-checkpoint or science claim.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .cuda_restricted_wh import (
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    _validate_compatibility,
)
from .cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridFailureHandoff,
    CUDARestrictedWHHybridIntegrationError,
    CUDARestrictedWHRouteRecord,
    DEFAULT_ROUTE_WORKER_COUNT,
    _child_spec,
    _slice_tile,
    advance_cuda_restricted_wh_hybrid,
)
from .cuda_tracer_events import (
    EVENT_PRIMARY_BOUNDARY_OUTWARD,
    EVENT_SOURCE_RADIUS_CONTACT,
)
from .engine.backends import resolve_backend
from .restricted_wh_route import (
    RestrictedWHRouteEvent,
    RestrictedWHRouteSpec,
)


LIFECYCLE_METHOD_ID = "JX_CUDA_RESTRICTED_WH_FIXED_LANE_LIFECYCLE_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

ACTIVE = 0
REMOVED_SOURCE_RADIUS = 1
REMOVED_OUTER_BOUNDARY = 2
_VALID_LIFECYCLE_CODES = frozenset(
    {ACTIVE, REMOVED_SOURCE_RADIUS, REMOVED_OUTER_BOUNDARY}
)
_EVENT_TO_LIFECYCLE = {
    EVENT_SOURCE_RADIUS_CONTACT: REMOVED_SOURCE_RADIUS,
    EVENT_PRIMARY_BOUNDARY_OUTWARD: REMOVED_OUTER_BOUNDARY,
}

LIFECYCLE_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "close_encounter_episode_qualified": False,
        "physical_collision_inferred": False,
        "physical_ejection_inferred": False,
        "long_horizon_accuracy_qualified": False,
        "science_result": False,
    }
)


class RestrictedWHLifecycleError(RuntimeError):
    """Base error for the fixed-lane lifecycle wrapper."""


class RestrictedWHLifecycleContractError(RestrictedWHLifecycleError, ValueError):
    """A lifecycle request violates the explicit M1 contract."""


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _plain_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_tree(item) for item in value]
    return value


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_tree(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"jx.restricted-wh.runtime-identity.v1\0" + payload).hexdigest()


def _owned_uint8_vector(value: object, lane_count: int) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.uint8)
        or value.shape != (lane_count,)
        or not value.flags.c_contiguous
    ):
        raise RestrictedWHLifecycleContractError(
            "tracer_lifecycle must be an exact C-contiguous uint8 vector"
        )
    owned = np.frombuffer(value.tobytes(order="C"), dtype=np.uint8)
    if any(int(item) not in _VALID_LIFECYCLE_CODES for item in owned):
        raise RestrictedWHLifecycleContractError(
            "tracer_lifecycle contains an unknown code"
        )
    return owned


def _event_record(event: RestrictedWHRouteEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "source_index": event.source_index,
        "source_id": event.source_id,
        "map_step_fraction": event.map_step_fraction.hex(),
        "source_interval_offset": event.source_interval_offset,
        "interval_fraction": event.interval_fraction.hex(),
        "relative_position_AU": [item.hex() for item in event.relative_position_AU],
        "relative_velocity_AU_per_year": [
            item.hex() for item in event.relative_velocity_AU_per_year
        ],
        "surface_distance_AU": event.surface_distance_AU.hex(),
        "surface_threshold_AU": event.surface_threshold_AU.hex(),
        "surface_residual_AU": event.surface_residual_AU.hex(),
        "surface_tolerance_AU": event.surface_tolerance_AU.hex(),
        "primary_specific_energy_AU2_per_year2": (
            event.primary_specific_energy_AU2_per_year2.hex()
        ),
        "primary_radial_velocity_AU_per_year": (
            event.primary_radial_velocity_AU_per_year.hex()
        ),
        "surface_radial_velocity_AU_per_year": (
            event.surface_radial_velocity_AU_per_year.hex()
        ),
        "location_method": event.location_method,
    }


@dataclass(frozen=True, eq=False)
class RestrictedWHRemovalRecord:
    """Canonical terminal event bound to one stable public tracer lane."""

    lane_index: int
    tracer_index: int
    lifecycle_code: int
    global_source_step: int
    source_steps_per_map_step: int
    guard_phase: str
    guard_reason: str
    guard_source_index: int | None
    source_segment_sha256: str
    route_runtime_identity_sha256: str
    accepted_step_count: int
    derivative_evaluation_count: int
    minimum_committed_stage_distance_AU: float
    maximum_committed_primary_dynamical_phase_radians: float
    events: tuple[RestrictedWHRouteEvent, ...]
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.lane_index, "lane_index"),
            (self.tracer_index, "tracer_index"),
            (self.global_source_step, "global_source_step"),
        ):
            if type(value) is not int or value < 0:
                raise RestrictedWHLifecycleContractError(
                    f"{label} must be a nonnegative integer"
                )
        if self.lifecycle_code not in (
            REMOVED_SOURCE_RADIUS,
            REMOVED_OUTER_BOUNDARY,
        ):
            raise RestrictedWHLifecycleContractError(
                "removal lifecycle_code is not terminal"
            )
        if (
            type(self.source_steps_per_map_step) is not int
            or self.source_steps_per_map_step <= 0
        ):
            raise RestrictedWHLifecycleContractError(
                "source_steps_per_map_step must be positive"
            )
        for value, label in (
            (self.guard_phase, "guard_phase"),
            (self.guard_reason, "guard_reason"),
        ):
            if type(value) is not str or not value:
                raise RestrictedWHLifecycleContractError(
                    f"{label} must be nonempty"
                )
        if self.guard_source_index is not None and (
            type(self.guard_source_index) is not int
            or self.guard_source_index < 0
        ):
            raise RestrictedWHLifecycleContractError(
                "guard_source_index must be nonnegative or None"
            )
        for value, label in (
            (self.source_segment_sha256, "source_segment_sha256"),
            (
                self.route_runtime_identity_sha256,
                "route_runtime_identity_sha256",
            ),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RestrictedWHLifecycleContractError(
                    f"{label} must be a lowercase SHA-256 digest"
                )
        for value, label in (
            (self.accepted_step_count, "accepted_step_count"),
            (
                self.derivative_evaluation_count,
                "derivative_evaluation_count",
            ),
        ):
            if type(value) is not int or value < 0:
                raise RestrictedWHLifecycleContractError(
                    f"{label} must be nonnegative"
                )
        for value, label in (
            (
                self.minimum_committed_stage_distance_AU,
                "minimum_committed_stage_distance_AU",
            ),
            (
                self.maximum_committed_primary_dynamical_phase_radians,
                "maximum_committed_primary_dynamical_phase_radians",
            ),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise RestrictedWHLifecycleContractError(
                    f"{label} must be a finite built-in float"
                )
        if self.minimum_committed_stage_distance_AU <= 0.0:
            raise RestrictedWHLifecycleContractError(
                "minimum committed source distance must be positive"
            )
        if self.maximum_committed_primary_dynamical_phase_radians < 0.0:
            raise RestrictedWHLifecycleContractError(
                "maximum committed primary phase cannot be negative"
            )
        if not self.events or any(
            type(item) is not RestrictedWHRouteEvent for item in self.events
        ):
            raise RestrictedWHLifecycleContractError(
                "events must be a nonempty exact RestrictedWHRouteEvent tuple"
            )
        expected_code = _EVENT_TO_LIFECYCLE.get(self.events[0].event_type)
        if expected_code != self.lifecycle_code:
            raise RestrictedWHLifecycleContractError(
                "first event and lifecycle code disagree"
            )
        first_position = self.events[0].relative_position_AU
        first_velocity = self.events[0].relative_velocity_AU_per_year
        if any(
            not np.array_equal(event.relative_position_AU, first_position)
            or not np.array_equal(event.relative_velocity_AU_per_year, first_velocity)
            for event in self.events[1:]
        ):
            raise RestrictedWHLifecycleContractError(
                "simultaneous terminal events must share one located state"
            )
        actual = removal_record_sha256(self)
        if self.content_sha256 is not None and self.content_sha256 != actual:
            raise RestrictedWHLifecycleContractError(
                "content_sha256 does not match the removal record"
            )
        object.__setattr__(self, "content_sha256", actual)

    @property
    def event(self) -> RestrictedWHRouteEvent:
        return self.events[0]


def removal_record_sha256(record: RestrictedWHRemovalRecord) -> str:
    payload = {
        "schema": "jx.restricted-wh.removal-record.v1",
        "lane_index": record.lane_index,
        "tracer_index": record.tracer_index,
        "lifecycle_code": record.lifecycle_code,
        "global_source_step": record.global_source_step,
        "source_steps_per_map_step": record.source_steps_per_map_step,
        "guard_phase": record.guard_phase,
        "guard_reason": record.guard_reason,
        "guard_source_index": record.guard_source_index,
        "source_segment_sha256": record.source_segment_sha256,
        "route_runtime_identity_sha256": record.route_runtime_identity_sha256,
        "accepted_step_count": record.accepted_step_count,
        "derivative_evaluation_count": record.derivative_evaluation_count,
        "minimum_committed_stage_distance_AU": (
            record.minimum_committed_stage_distance_AU.hex()
        ),
        "maximum_committed_primary_dynamical_phase_radians": (
            record.maximum_committed_primary_dynamical_phase_radians.hex()
        ),
        "events": [_event_record(event) for event in record.events],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"jx.restricted-wh.removal-record.v1\0" + encoded).hexdigest()


@dataclass(frozen=True, eq=False)
class RestrictedWHLifecycleState:
    """Fixed-lane dynamics plus immutable terminal lifecycle history."""

    dynamics_state: CUDARestrictedWHState
    tracer_lifecycle: np.ndarray
    removal_records: tuple[RestrictedWHRemovalRecord, ...] = ()

    def __post_init__(self) -> None:
        if type(self.dynamics_state) is not CUDARestrictedWHState:
            raise RestrictedWHLifecycleContractError(
                "dynamics_state must be an exact CUDARestrictedWHState"
            )
        lane_count = int(self.dynamics_state.tracer_indices.shape[0])
        lifecycle = _owned_uint8_vector(self.tracer_lifecycle, lane_count)
        if type(self.removal_records) is not tuple or any(
            type(item) is not RestrictedWHRemovalRecord
            for item in self.removal_records
        ):
            raise RestrictedWHLifecycleContractError(
                "removal_records must be an immutable exact tuple"
            )
        keys = tuple(
            (
                record.global_source_step,
                record.event.map_step_fraction,
                record.lane_index,
                record.tracer_index,
            )
            for record in self.removal_records
        )
        if keys != tuple(sorted(keys)) or len(
            {record.lane_index for record in self.removal_records}
        ) != len(self.removal_records):
            raise RestrictedWHLifecycleContractError(
                "removal records must be canonical with one per lane"
            )
        for record in self.removal_records:
            if record.lane_index >= lane_count:
                raise RestrictedWHLifecycleContractError(
                    "removal record lane is outside the fixed state"
                )
            if int(lifecycle[record.lane_index]) != record.lifecycle_code:
                raise RestrictedWHLifecycleContractError(
                    "removal record and lifecycle vector disagree"
                )
        terminal_lanes = set(np.flatnonzero(lifecycle != ACTIVE).tolist())
        record_lanes = {record.lane_index for record in self.removal_records}
        if terminal_lanes != record_lanes:
            raise RestrictedWHLifecycleContractError(
                "every terminal lane must have exactly one removal record"
            )
        object.__setattr__(self, "tracer_lifecycle", lifecycle)


def initialize_restricted_wh_lifecycle(
    state: CUDARestrictedWHState,
) -> RestrictedWHLifecycleState:
    """Create an all-active fixed-lane lifecycle state."""

    if type(state) is not CUDARestrictedWHState:
        raise RestrictedWHLifecycleContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    return RestrictedWHLifecycleState(
        dynamics_state=state,
        tracer_lifecycle=np.zeros(
            int(state.tracer_indices.shape[0]), dtype=np.uint8
        ),
    )


@dataclass(frozen=True)
class RestrictedWHLifecycleFailureHandoff:
    state: RestrictedWHLifecycleState
    failure_map_step_index: int
    reason: str
    hybrid_handoff: CUDARestrictedWHHybridFailureHandoff | None
    committed_map_steps: int = 0

    def __post_init__(self) -> None:
        if type(self.state) is not RestrictedWHLifecycleState:
            raise RestrictedWHLifecycleContractError(
                "handoff state must be the exact lifecycle input"
            )
        if (
            type(self.failure_map_step_index) is not int
            or self.failure_map_step_index < 0
        ):
            raise RestrictedWHLifecycleContractError(
                "failure_map_step_index must be nonnegative"
            )
        if type(self.reason) is not str or not self.reason:
            raise RestrictedWHLifecycleContractError("reason must be nonempty")
        if self.hybrid_handoff is not None and type(
            self.hybrid_handoff
        ) is not CUDARestrictedWHHybridFailureHandoff:
            raise RestrictedWHLifecycleContractError(
                "hybrid_handoff has the wrong type"
            )
        if self.committed_map_steps != 0:
            raise RestrictedWHLifecycleContractError(
                "a public lifecycle failure cannot commit partial work"
            )


class RestrictedWHLifecycleIntegrationError(RestrictedWHLifecycleError):
    """Lifecycle advancement failed with the public input unchanged."""

    def __init__(
        self,
        message: str,
        handoff: RestrictedWHLifecycleFailureHandoff,
    ) -> None:
        super().__init__(message)
        self.handoff = handoff


@dataclass(frozen=True, eq=False)
class RestrictedWHLifecycleResult:
    state: RestrictedWHLifecycleState
    source_tile_sha256: str
    minimum_source_distance_AU: object
    maximum_kepler_iterations: int
    route_records: tuple[CUDARestrictedWHRouteRecord, ...]
    new_removal_records: tuple[RestrictedWHRemovalRecord, ...]
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = LIFECYCLE_CLAIMS
    method_id: str = LIFECYCLE_METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.state) is not RestrictedWHLifecycleState:
            raise RestrictedWHLifecycleContractError(
                "result state must be an exact RestrictedWHLifecycleState"
            )
        if (
            type(self.source_tile_sha256) is not str
            or len(self.source_tile_sha256) != 64
        ):
            raise RestrictedWHLifecycleContractError(
                "source_tile_sha256 must be a SHA-256 digest"
            )
        if (
            type(self.maximum_kepler_iterations) is not int
            or self.maximum_kepler_iterations < 0
        ):
            raise RestrictedWHLifecycleContractError(
                "maximum_kepler_iterations must be nonnegative"
            )
        if type(self.route_records) is not tuple or any(
            type(item) is not CUDARestrictedWHRouteRecord
            for item in self.route_records
        ):
            raise RestrictedWHLifecycleContractError(
                "route_records must be an exact tuple"
            )
        if type(self.new_removal_records) is not tuple or any(
            type(item) is not RestrictedWHRemovalRecord
            for item in self.new_removal_records
        ):
            raise RestrictedWHLifecycleContractError(
                "new_removal_records must be an exact tuple"
            )
        if (
            self.method_id != LIFECYCLE_METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(LIFECYCLE_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHLifecycleContractError(
                "lifecycle identity or claim ceiling changed"
            )
        object.__setattr__(
            self, "claims", MappingProxyType(dict(LIFECYCLE_CLAIMS))
        )
        object.__setattr__(
            self, "runtime_identity", _freeze_tree(self.runtime_identity)
        )


def _validate_lifecycle_state(
    state: RestrictedWHLifecycleState,
    source_tile: FrozenSourcePathTile,
    route_spec: RestrictedWHRouteSpec,
    cp: Any,
) -> np.ndarray:
    indices = np.ascontiguousarray(cp.asnumpy(state.dynamics_state.tracer_indices))
    positions = np.ascontiguousarray(
        cp.asnumpy(state.dynamics_state.tracer_relative_positions_AU)
    )
    velocities = np.ascontiguousarray(
        cp.asnumpy(state.dynamics_state.tracer_relative_velocities_AU_per_year)
    )
    records = {record.lane_index: record for record in state.removal_records}
    for lane, code in enumerate(state.tracer_lifecycle.tolist()):
        if code == ACTIVE:
            for source_index, radius in enumerate(
                source_tile.source_encounter_radii_AU
            ):
                distance = float(
                    np.linalg.norm(
                        source_tile.source_relative_positions_AU[0, source_index]
                        - positions[lane]
                    )
                )
                if not distance > float(radius):
                    raise RestrictedWHLifecycleContractError(
                        "active tracer begins on or inside a source surface"
                    )
            if route_spec.primary_outer_boundary_AU is not None:
                primary_distance = float(np.linalg.norm(positions[lane]))
                if not primary_distance < route_spec.primary_outer_boundary_AU:
                    raise RestrictedWHLifecycleContractError(
                        "active tracer begins on or outside the outer boundary"
                    )
            continue
        record = records[lane]
        if int(indices[lane]) != record.tracer_index:
            raise RestrictedWHLifecycleContractError(
                "removal record tracer identity changed"
            )
        if record.global_source_step >= state.dynamics_state.global_source_step:
            raise RestrictedWHLifecycleContractError(
                "removal record is not in the committed state history"
            )
        if not np.array_equal(positions[lane], record.event.relative_position_AU):
            raise RestrictedWHLifecycleContractError(
                "removed tracer position changed after its terminal event"
            )
        if not np.array_equal(
            velocities[lane], record.event.relative_velocity_AU_per_year
        ):
            raise RestrictedWHLifecycleContractError(
                "removed tracer velocity changed after its terminal event"
            )
    return indices


def _failure(
    input_state: RestrictedWHLifecycleState,
    map_step_index: int,
    reason: str,
    hybrid_handoff: CUDARestrictedWHHybridFailureHandoff | None,
) -> RestrictedWHLifecycleIntegrationError:
    return RestrictedWHLifecycleIntegrationError(
        "restricted-WH lifecycle advance failed; input state is unchanged",
        RestrictedWHLifecycleFailureHandoff(
            state=input_state,
            failure_map_step_index=map_step_index,
            reason=reason,
            hybrid_handoff=hybrid_handoff,
        ),
    )


def _removal_from_handoff(
    handoff: CUDARestrictedWHHybridFailureHandoff,
    fixed_lane: int,
) -> RestrictedWHRemovalRecord:
    route = handoff.route_result
    if (
        route is None
        or not route.terminated_at_event
        or handoff.guard_phase is None
        or handoff.guard_reason is None
    ):
        raise RestrictedWHLifecycleContractError(
            "terminal hybrid handoff is incomplete"
        )
    lifecycle_code = _EVENT_TO_LIFECYCLE.get(route.events[0].event_type)
    if lifecycle_code is None:
        raise RestrictedWHLifecycleContractError(
            "terminal event has no lifecycle mapping"
        )
    route_runtime_identity = dict(_plain_tree(route.runtime_identity))
    route_runtime_identity.pop("source_tile_sha256", None)
    return RestrictedWHRemovalRecord(
        lane_index=fixed_lane,
        tracer_index=route.tracer_index,
        lifecycle_code=lifecycle_code,
        global_source_step=route.global_source_step,
        source_steps_per_map_step=route.source_steps_per_map_step,
        guard_phase=handoff.guard_phase,
        guard_reason=handoff.guard_reason,
        guard_source_index=handoff.guard_source_index,
        source_segment_sha256=route.source_segment_sha256,
        route_runtime_identity_sha256=_mapping_sha256(route_runtime_identity),
        accepted_step_count=route.accepted_step_count,
        derivative_evaluation_count=route.derivative_evaluation_count,
        minimum_committed_stage_distance_AU=(
            route.minimum_committed_stage_distance_AU
        ),
        maximum_committed_primary_dynamical_phase_radians=(
            route.maximum_committed_primary_dynamical_phase_radians
        ),
        events=route.events,
    )


def advance_cuda_restricted_wh_lifecycle(
    state: RestrictedWHLifecycleState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    *,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
) -> RestrictedWHLifecycleResult:
    """Advance fixed lanes; freeze terminal lanes and continue active lanes."""

    if type(state) is not RestrictedWHLifecycleState:
        raise RestrictedWHLifecycleContractError(
            "state must be an exact RestrictedWHLifecycleState"
        )
    try:
        _validate_compatibility(state.dynamics_state, source_tile, spec)
    except Exception as exc:
        raise RestrictedWHLifecycleContractError(str(exc)) from exc
    if route_spec is None:
        route_spec = RestrictedWHRouteSpec(
            maximum_primary_dynamical_phase=(
                spec.maximum_primary_dynamical_phase
            ),
            primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
        )
    if type(route_spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHLifecycleContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    if (
        route_spec.maximum_primary_dynamical_phase
        != spec.maximum_primary_dynamical_phase
        or route_spec.primary_outer_boundary_AU
        != spec.primary_outer_boundary_AU
    ):
        raise RestrictedWHLifecycleContractError(
            "CUDA and DOP853 lifecycle policies must match exactly"
        )
    if (
        type(route_worker_count) is not int
        or not 1 <= route_worker_count <= 32
    ):
        raise RestrictedWHLifecycleContractError(
            "route_worker_count must be an integer in [1, 32]"
        )

    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    with resolved.activate():
        try:
            fixed_indices = _validate_lifecycle_state(
                state, source_tile, route_spec, cp
            )
            input_dynamics = state.dynamics_state
            full_positions = cp.array(
                input_dynamics.tracer_relative_positions_AU,
                dtype=cp.float64,
                copy=True,
                order="C",
            )
            full_velocities = cp.array(
                input_dynamics.tracer_relative_velocities_AU_per_year,
                dtype=cp.float64,
                copy=True,
                order="C",
            )
            full_indices = cp.array(
                input_dynamics.tracer_indices,
                dtype=cp.uint64,
                copy=True,
                order="C",
            )
            lifecycle = np.array(state.tracer_lifecycle, copy=True)
            lane_count = len(lifecycle)
            segment_minimum = cp.full(
                lane_count, cp.inf, dtype=cp.float64
            )
        except RestrictedWHLifecycleContractError:
            raise
        except Exception as exc:
            raise _failure(
                state,
                0,
                f"LIFECYCLE_VALIDATION_{type(exc).__name__}",
                None,
            ) from exc
        route_records: list[CUDARestrictedWHRouteRecord] = []
        new_removals: list[RestrictedWHRemovalRecord] = []
        hybrid_runtime_identities: list[Mapping[str, Any]] = []
        terminal_route_runtime_identities: list[Mapping[str, Any]] = []
        maximum_kepler_iterations = 0
        current_map = 0
        current_global_step = input_dynamics.global_source_step

        try:
            while current_map < spec.map_step_count:
                active_lanes = np.flatnonzero(lifecycle == ACTIVE).astype(
                    np.int64, copy=False
                )
                if not len(active_lanes):
                    current_global_step = (
                        input_dynamics.global_source_step
                        + spec.map_step_count
                        * spec.source_steps_per_map_step
                    )
                    break
                device_lanes = cp.asarray(active_lanes, dtype=cp.int64)
                subset = replace(
                    input_dynamics,
                    global_source_step=current_global_step,
                    tracer_relative_positions_AU=cp.ascontiguousarray(
                        full_positions[device_lanes]
                    ),
                    tracer_relative_velocities_AU_per_year=cp.ascontiguousarray(
                        full_velocities[device_lanes]
                    ),
                    tracer_indices=cp.ascontiguousarray(
                        full_indices[device_lanes]
                    ),
                )
                remaining_maps = spec.map_step_count - current_map
                start_interval = (
                    current_map * spec.source_steps_per_map_step
                )
                candidate_tile = _slice_tile(
                    source_tile,
                    start_interval,
                    remaining_maps * spec.source_steps_per_map_step,
                )
                candidate_spec = _child_spec(spec, remaining_maps)
                try:
                    result = advance_cuda_restricted_wh_hybrid(
                        subset,
                        candidate_tile,
                        candidate_spec,
                        route_spec=route_spec,
                        route_worker_count=route_worker_count,
                    )
                except CUDARestrictedWHHybridIntegrationError as exc:
                    handoff = exc.handoff
                    if handoff.reason != "TERMINAL_EVENT":
                        raise _failure(
                            state,
                            current_map + handoff.failure_map_step_index,
                            handoff.reason,
                            handoff,
                        ) from exc
                    if handoff.failure_map_step_index > 0:
                        prefix_maps = handoff.failure_map_step_index
                        prefix_tile = _slice_tile(
                            source_tile,
                            start_interval,
                            prefix_maps * spec.source_steps_per_map_step,
                        )
                        prefix = advance_cuda_restricted_wh_hybrid(
                            subset,
                            prefix_tile,
                            _child_spec(spec, prefix_maps),
                            route_spec=route_spec,
                            route_worker_count=route_worker_count,
                        )
                        full_positions[device_lanes] = (
                            prefix.state.tracer_relative_positions_AU
                        )
                        full_velocities[device_lanes] = (
                            prefix.state.tracer_relative_velocities_AU_per_year
                        )
                        segment_minimum[device_lanes] = cp.minimum(
                            segment_minimum[device_lanes],
                            prefix.minimum_source_distance_AU,
                        )
                        maximum_kepler_iterations = max(
                            maximum_kepler_iterations,
                            prefix.maximum_kepler_iterations,
                        )
                        hybrid_runtime_identities.append(prefix.runtime_identity)
                        for record in prefix.route_records:
                            route_records.append(
                                replace(
                                    record,
                                    map_step_index=(
                                        current_map + record.map_step_index
                                    ),
                                    lane_index=int(
                                        active_lanes[record.lane_index]
                                    ),
                                )
                            )
                        current_map += prefix_maps
                        current_global_step += (
                            prefix_maps * spec.source_steps_per_map_step
                        )
                        continue
                    if handoff.lane_index is None or handoff.route_result is None:
                        raise _failure(
                            state,
                            current_map,
                            "INCOMPLETE_TERMINAL_HANDOFF",
                            handoff,
                        ) from exc
                    fixed_lane = int(active_lanes[handoff.lane_index])
                    removal = _removal_from_handoff(handoff, fixed_lane)
                    terminal_route_runtime_identities.append(
                        handoff.route_result.runtime_identity
                    )
                    if int(fixed_indices[fixed_lane]) != removal.tracer_index:
                        raise _failure(
                            state,
                            current_map,
                            "TERMINAL_TRACER_IDENTITY_CHANGED",
                            handoff,
                        ) from exc
                    full_positions[fixed_lane] = cp.asarray(
                        removal.event.relative_position_AU,
                        dtype=cp.float64,
                    )
                    full_velocities[fixed_lane] = cp.asarray(
                        removal.event.relative_velocity_AU_per_year,
                        dtype=cp.float64,
                    )
                    segment_minimum[fixed_lane] = cp.minimum(
                        segment_minimum[fixed_lane],
                        cp.float64(
                            removal.minimum_committed_stage_distance_AU
                        ),
                    )
                    lifecycle[fixed_lane] = np.uint8(removal.lifecycle_code)
                    new_removals.append(removal)
                    continue

                full_positions[device_lanes] = (
                    result.state.tracer_relative_positions_AU
                )
                full_velocities[device_lanes] = (
                    result.state.tracer_relative_velocities_AU_per_year
                )
                segment_minimum[device_lanes] = cp.minimum(
                    segment_minimum[device_lanes],
                    result.minimum_source_distance_AU,
                )
                maximum_kepler_iterations = max(
                    maximum_kepler_iterations,
                    result.maximum_kepler_iterations,
                )
                hybrid_runtime_identities.append(result.runtime_identity)
                for record in result.route_records:
                    route_records.append(
                        replace(
                            record,
                            map_step_index=current_map + record.map_step_index,
                            lane_index=int(active_lanes[record.lane_index]),
                        )
                    )
                current_map = spec.map_step_count
                current_global_step = result.state.global_source_step

            output_dynamics = replace(
                input_dynamics,
                global_source_step=current_global_step,
                tracer_relative_positions_AU=full_positions,
                tracer_relative_velocities_AU_per_year=full_velocities,
                tracer_indices=full_indices,
            )
            new_removals.sort(
                key=lambda record: (
                    record.global_source_step,
                    record.event.map_step_fraction,
                    record.lane_index,
                    record.tracer_index,
                )
            )
            output_state = RestrictedWHLifecycleState(
                dynamics_state=output_dynamics,
                tracer_lifecycle=lifecycle,
                removal_records=state.removal_records + tuple(new_removals),
            )
            route_records.sort(
                key=lambda record: (
                    record.map_step_index,
                    record.lane_index,
                    record.tracer_index,
                )
            )
            cp.cuda.get_current_stream().synchronize()
            return RestrictedWHLifecycleResult(
                state=output_state,
                source_tile_sha256=source_tile.content_sha256 or "",
                minimum_source_distance_AU=segment_minimum,
                maximum_kepler_iterations=maximum_kepler_iterations,
                route_records=tuple(route_records),
                new_removal_records=tuple(new_removals),
                runtime_identity={
                    "method_id": LIFECYCLE_METHOD_ID,
                    "implementation_source_sha256": (
                        IMPLEMENTATION_SOURCE_SHA256
                    ),
                    "hybrid_runtime_identities": tuple(
                        hybrid_runtime_identities
                    ),
                    "terminal_route_runtime_identities": tuple(
                        terminal_route_runtime_identities
                    ),
                    "route_worker_count": route_worker_count,
                    "source_tile_sha256": source_tile.content_sha256,
                    "determinism_scope": (
                        "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
                    ),
                },
            )
        except RestrictedWHLifecycleIntegrationError:
            raise
        except RestrictedWHLifecycleContractError:
            raise
        except Exception as exc:
            raise _failure(
                state,
                current_map,
                f"LIFECYCLE_RUNTIME_{type(exc).__name__}",
                None,
            ) from exc


__all__ = [
    "ACTIVE",
    "IMPLEMENTATION_SOURCE_SHA256",
    "LIFECYCLE_CLAIMS",
    "LIFECYCLE_METHOD_ID",
    "PROJECT_SCOPE",
    "REMOVED_OUTER_BOUNDARY",
    "REMOVED_SOURCE_RADIUS",
    "RestrictedWHLifecycleContractError",
    "RestrictedWHLifecycleError",
    "RestrictedWHLifecycleFailureHandoff",
    "RestrictedWHLifecycleIntegrationError",
    "RestrictedWHLifecycleResult",
    "RestrictedWHLifecycleState",
    "RestrictedWHRemovalRecord",
    "SCIENTIFIC_CLASSIFICATION",
    "advance_cuda_restricted_wh_lifecycle",
    "initialize_restricted_wh_lifecycle",
    "removal_record_sha256",
]
