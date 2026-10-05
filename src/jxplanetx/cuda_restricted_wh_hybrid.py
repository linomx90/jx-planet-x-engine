"""Transactional CUDA/CPU guarded-step orchestration for restricted WH M1.

Quiet tracer lanes remain on the fused CUDA map.  When that map reports a
guarded lane, this module redoes exactly that coarse step with the continuous
restricted-frame DOP853 route, removes the routed lane from the provisional
GPU batch, and retries the remaining lanes.  Results are merged in the exact
input lane order.  A located terminal event, numerical ambiguity, or
non-routable device failure aborts the whole requested segment and returns a
handoff to the unchanged input state.

This correctness-first implementation discovers guarded lanes iteratively and
dispatches each canonical guarded wave to deterministic spawned CPU workers.
It is not yet the high-throughput whole-encounter-episode implementation.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
import multiprocessing
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .cuda_restricted_wh import (
    CUDARestrictedWHContractError,
    CUDARestrictedWHIntegrationError,
    CUDARestrictedWHResult,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    _runtime_identity,
    advance_cuda_restricted_wh_fused,
    verify_source_path_tile,
)
from .engine.backends import resolve_backend
from .restricted_wh_route import (
    RestrictedWHRouteResult,
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
    route_restricted_wh_map_step,
)


HYBRID_METHOD_ID = "JX_CUDA_RESTRICTED_WH_GUARDED_DOP853_V2"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
DEFAULT_ROUTE_WORKER_COUNT = 6
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROUTABLE_REASONS = frozenset(
    {
        "SOURCE_RADIUS_GUARD",
        "SWEPT_SOURCE_RADIUS_GUARD",
        "OUTER_BOUNDARY_GUARD",
        "PRIMARY_PHASE_GUARD",
    }
)
HYBRID_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "close_encounter_qualified": False,
        "collision_response_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "parallel_route_wavefront_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "science_result": False,
    }
)


@dataclass(frozen=True)
class _RouteJob:
    original_lane: int
    tracer_index: int
    guard_phase: str
    guard_reason: str
    guard_source_index: int | None
    source_node_index: int
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class _RoutePayload:
    values: dict[str, Any]


def _plain_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_tree(item) for item in value)
    return value


def _route_payload(result: RestrictedWHRouteResult) -> _RoutePayload:
    return _RoutePayload(
        {
            "tracer_index": result.tracer_index,
            "source_tile_sha256": result.source_tile_sha256,
            "source_segment_sha256": result.source_segment_sha256,
            "global_source_step": result.global_source_step,
            "source_steps_per_map_step": result.source_steps_per_map_step,
            "events": result.events,
            "endpoint_relative_position_AU": (
                result.endpoint_relative_position_AU
            ),
            "endpoint_relative_velocity_AU_per_year": (
                result.endpoint_relative_velocity_AU_per_year
            ),
            "accepted_step_count": result.accepted_step_count,
            "derivative_evaluation_count": result.derivative_evaluation_count,
            "minimum_evaluated_stage_distance_AU": (
                result.minimum_evaluated_stage_distance_AU
            ),
            "minimum_committed_stage_distance_AU": (
                result.minimum_committed_stage_distance_AU
            ),
            "maximum_accepted_primary_dynamical_phase_radians": (
                result.maximum_accepted_primary_dynamical_phase_radians
            ),
            "maximum_committed_primary_dynamical_phase_radians": (
                result.maximum_committed_primary_dynamical_phase_radians
            ),
            "completed_map_fraction": result.completed_map_fraction,
            "runtime_identity": _plain_tree(result.runtime_identity),
        }
    )


def _route_from_payload(payload: _RoutePayload) -> RestrictedWHRouteResult:
    if type(payload) is not _RoutePayload or type(payload.values) is not dict:
        raise RuntimeError("route worker returned an invalid payload")
    return RestrictedWHRouteResult(**payload.values)


class _RouteDispatchFailure(RuntimeError):
    def __init__(self, job: _RouteJob, cause: Exception) -> None:
        super().__init__(str(cause))
        self.job = job
        self.cause = cause


_WORKER_SOURCE_TILE: FrozenSourcePathTile | None = None
_WORKER_ROUTE_SPEC: RestrictedWHRouteSpec | None = None
_WORKER_SOURCE_STEPS_PER_MAP_STEP: int | None = None


def _initialize_route_worker(
    source_tile: FrozenSourcePathTile,
    route_spec: RestrictedWHRouteSpec,
    source_steps_per_map_step: int,
) -> None:
    global _WORKER_SOURCE_TILE
    global _WORKER_ROUTE_SPEC
    global _WORKER_SOURCE_STEPS_PER_MAP_STEP
    _WORKER_SOURCE_TILE = source_tile
    _WORKER_ROUTE_SPEC = route_spec
    _WORKER_SOURCE_STEPS_PER_MAP_STEP = source_steps_per_map_step


def _execute_route_job(job: _RouteJob) -> _RoutePayload:
    if (
        _WORKER_SOURCE_TILE is None
        or _WORKER_ROUTE_SPEC is None
        or _WORKER_SOURCE_STEPS_PER_MAP_STEP is None
    ):
        raise RuntimeError("route worker was not initialized")
    return _route_payload(
        route_restricted_wh_map_step(
            job.position,
            job.velocity,
            tracer_index=job.tracer_index,
            source_tile=_WORKER_SOURCE_TILE,
            source_node_index=job.source_node_index,
            source_steps_per_map_step=_WORKER_SOURCE_STEPS_PER_MAP_STEP,
            spec=_WORKER_ROUTE_SPEC,
        )
    )


class _RouteDispatcher:
    def __init__(
        self,
        *,
        source_tile: FrozenSourcePathTile,
        route_spec: RestrictedWHRouteSpec,
        source_steps_per_map_step: int,
        worker_count: int,
    ) -> None:
        self.source_tile = source_tile
        self.route_spec = route_spec
        self.source_steps_per_map_step = source_steps_per_map_step
        self.worker_count = worker_count
        self.executor: ProcessPoolExecutor | None = None

    def route(
        self, jobs: tuple[_RouteJob, ...]
    ) -> tuple[RestrictedWHRouteResult, ...]:
        if not jobs:
            return ()
        if self.worker_count == 1:
            results: list[RestrictedWHRouteResult] = []
            for job in jobs:
                try:
                    results.append(
                        route_restricted_wh_map_step(
                            job.position,
                            job.velocity,
                            tracer_index=job.tracer_index,
                            source_tile=self.source_tile,
                            source_node_index=job.source_node_index,
                            source_steps_per_map_step=(
                                self.source_steps_per_map_step
                            ),
                            spec=self.route_spec,
                        )
                    )
                except Exception as exc:
                    raise _RouteDispatchFailure(job, exc) from exc
            return tuple(results)
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_route_worker,
                initargs=(
                    self.source_tile,
                    self.route_spec,
                    self.source_steps_per_map_step,
                ),
            )
        futures = [self.executor.submit(_execute_route_job, job) for job in jobs]
        results = []
        for job, future in zip(jobs, futures, strict=True):
            try:
                results.append(_route_from_payload(future.result()))
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise _RouteDispatchFailure(job, exc) from exc
        return tuple(results)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


class CUDARestrictedWHHybridError(RuntimeError):
    """Base error for the guarded restricted-WH scheduler."""


class CUDARestrictedWHHybridContractError(CUDARestrictedWHHybridError, ValueError):
    """The hybrid request violates its fixed contract."""


@dataclass(frozen=True)
class CUDARestrictedWHHybridFailureHandoff:
    """Transactional failure witness referring to the exact input state."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    failure_map_step_index: int
    reason: str
    tracer_index: int | None
    route_result: RestrictedWHRouteResult | None
    lane_index: int | None = None
    guard_phase: str | None = None
    guard_reason: str | None = None
    guard_source_index: int | None = None
    committed_map_steps: int = 0
    checkpoint_qualified: bool = False

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise CUDARestrictedWHHybridContractError(
                "handoff state must be the exact input state"
            )
        if (
            type(self.source_tile_sha256) is not str
            or len(self.source_tile_sha256) != 64
        ):
            raise CUDARestrictedWHHybridContractError(
                "source_tile_sha256 must be a SHA-256 digest"
            )
        if (
            type(self.failure_map_step_index) is not int
            or self.failure_map_step_index < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "failure_map_step_index must be nonnegative"
            )
        if type(self.reason) is not str or not self.reason:
            raise CUDARestrictedWHHybridContractError("reason must be nonempty")
        if self.tracer_index is not None and (
            type(self.tracer_index) is not int or self.tracer_index < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "tracer_index must be nonnegative or None"
            )
        if (
            self.route_result is not None
            and type(self.route_result) is not RestrictedWHRouteResult
        ):
            raise CUDARestrictedWHHybridContractError(
                "route_result must be an exact RestrictedWHRouteResult or None"
            )
        if self.lane_index is not None and (
            type(self.lane_index) is not int or self.lane_index < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "lane_index must be nonnegative or None"
            )
        for value, label in (
            (self.guard_phase, "guard_phase"),
            (self.guard_reason, "guard_reason"),
        ):
            if value is not None and (type(value) is not str or not value):
                raise CUDARestrictedWHHybridContractError(
                    f"{label} must be nonempty or None"
                )
        if self.guard_source_index is not None and (
            type(self.guard_source_index) is not int
            or self.guard_source_index < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "guard_source_index must be nonnegative or None"
            )
        if self.route_result is not None and self.route_result.terminated_at_event:
            if (
                self.lane_index is None
                or self.guard_phase is None
                or self.guard_reason is None
            ):
                raise CUDARestrictedWHHybridContractError(
                    "a terminal route handoff must retain its triggering guard"
                )
        if self.committed_map_steps != 0 or self.checkpoint_qualified is not False:
            raise CUDARestrictedWHHybridContractError(
                "hybrid handoff cannot commit partial work or claim a checkpoint"
            )


class CUDARestrictedWHHybridIntegrationError(CUDARestrictedWHHybridError):
    """The hybrid segment failed closed without exposing partial progress."""

    def __init__(
        self,
        message: str,
        handoff: CUDARestrictedWHHybridFailureHandoff,
    ) -> None:
        super().__init__(message)
        self.handoff = handoff


@dataclass(frozen=True)
class CUDARestrictedWHRouteRecord:
    """One rejoined DOP853 redo linked to its triggering CUDA guard."""

    map_step_index: int
    lane_index: int
    tracer_index: int
    guard_phase: str
    guard_reason: str
    guard_source_index: int | None
    route_result: RestrictedWHRouteResult

    def __post_init__(self) -> None:
        for value, label in (
            (self.map_step_index, "map_step_index"),
            (self.lane_index, "lane_index"),
            (self.tracer_index, "tracer_index"),
        ):
            if type(value) is not int or value < 0:
                raise CUDARestrictedWHHybridContractError(
                    f"{label} must be nonnegative"
                )
        if type(self.guard_phase) is not str or not self.guard_phase:
            raise CUDARestrictedWHHybridContractError(
                "guard_phase must be nonempty"
            )
        if self.guard_reason not in ROUTABLE_REASONS and not self.guard_reason.startswith(
            "KEPLER_"
        ):
            raise CUDARestrictedWHHybridContractError(
                "guard_reason is not routable"
            )
        if self.guard_source_index is not None and (
            type(self.guard_source_index) is not int
            or self.guard_source_index < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "guard_source_index must be nonnegative or None"
            )
        if type(self.route_result) is not RestrictedWHRouteResult:
            raise CUDARestrictedWHHybridContractError(
                "route_result must be an exact RestrictedWHRouteResult"
            )
        if self.route_result.terminated_at_event:
            raise CUDARestrictedWHHybridContractError(
                "a committed route record must have rejoined at the endpoint"
            )
        if self.route_result.tracer_index != self.tracer_index:
            raise CUDARestrictedWHHybridContractError(
                "route result tracer identity changed"
            )


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHHybridResult:
    """One completely committed hybrid segment."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    minimum_source_distance_AU: object
    maximum_kepler_iterations: int
    route_records: tuple[CUDARestrictedWHRouteRecord, ...]
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = HYBRID_CLAIMS
    method_id: str = HYBRID_METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise CUDARestrictedWHHybridContractError(
                "result state must be an exact CUDARestrictedWHState"
            )
        if (
            type(self.source_tile_sha256) is not str
            or len(self.source_tile_sha256) != 64
        ):
            raise CUDARestrictedWHHybridContractError(
                "source_tile_sha256 must be a SHA-256 digest"
            )
        if (
            type(self.maximum_kepler_iterations) is not int
            or self.maximum_kepler_iterations < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "maximum_kepler_iterations must be nonnegative"
            )
        if type(self.route_records) is not tuple or any(
            type(item) is not CUDARestrictedWHRouteRecord
            for item in self.route_records
        ):
            raise CUDARestrictedWHHybridContractError(
                "route_records must be an immutable exact tuple"
            )
        order = tuple(
            (record.map_step_index, record.lane_index, record.tracer_index)
            for record in self.route_records
        )
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise CUDARestrictedWHHybridContractError(
                "route records must be unique and canonically ordered"
            )
        if (
            self.method_id != HYBRID_METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(HYBRID_CLAIMS)
            or any(self.claims.values())
        ):
            raise CUDARestrictedWHHybridContractError(
                "hybrid identity or claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(HYBRID_CLAIMS)))
        object.__setattr__(self, "runtime_identity", _freeze_tree(self.runtime_identity))


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _slice_tile(
    tile: FrozenSourcePathTile,
    start_interval: int,
    interval_count: int,
) -> FrozenSourcePathTile:
    stop_node = start_interval + interval_count + 1
    return FrozenSourcePathTile(
        tile_id=f"{tile.tile_id}.hybrid.{start_interval}.{interval_count}",
        epoch_origin_years=tile.epoch_origin_years,
        global_source_step=tile.global_source_step + start_interval,
        source_step_years=tile.source_step_years,
        time_scale=tile.time_scale,
        frame=tile.frame,
        axes=tile.axes,
        coordinate_system=tile.coordinate_system,
        length_unit=tile.length_unit,
        time_unit=tile.time_unit,
        unit_system_id=tile.unit_system_id,
        source_ids=tile.source_ids,
        primary_source_id=tile.primary_source_id,
        source_schedule_sha256=tile.source_schedule_sha256,
        source_relative_positions_AU=np.ascontiguousarray(
            tile.source_relative_positions_AU[
                start_interval:stop_node
            ]
        ),
        source_relative_velocities_AU_per_year=np.ascontiguousarray(
            tile.source_relative_velocities_AU_per_year[
                start_interval:stop_node
            ]
        ),
        source_gm_AU3_per_year2=tile.source_gm_AU3_per_year2,
        source_encounter_radii_AU=tile.source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            tile.certified_relative_acceleration_bounds_AU_per_year2
        ),
        provenance=tile.provenance,
    )


def _child_spec(
    parent: FixedStepCUDARestrictedWHSpec,
    map_step_count: int,
) -> FixedStepCUDARestrictedWHSpec:
    return FixedStepCUDARestrictedWHSpec(
        map_step_count=map_step_count,
        source_steps_per_map_step=parent.source_steps_per_map_step,
        backend=parent.backend,
        maximum_primary_dynamical_phase=(
            parent.maximum_primary_dynamical_phase
        ),
        threads_per_block=parent.threads_per_block,
        primary_outer_boundary_AU=parent.primary_outer_boundary_AU,
    )


def _routable(reason: str) -> bool:
    return reason in ROUTABLE_REASONS or reason.startswith("KEPLER_")


def _failure(
    *,
    input_state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    map_step_index: int,
    reason: str,
    tracer_index: int | None,
    route_result: RestrictedWHRouteResult | None,
    lane_index: int | None = None,
    guard_phase: str | None = None,
    guard_reason: str | None = None,
    guard_source_index: int | None = None,
) -> CUDARestrictedWHHybridIntegrationError:
    return CUDARestrictedWHHybridIntegrationError(
        "hybrid restricted-WH segment failed closed; input state is unchanged",
        CUDARestrictedWHHybridFailureHandoff(
            state=input_state,
            source_tile_sha256=tile.content_sha256 or "",
            failure_map_step_index=map_step_index,
            reason=reason,
            tracer_index=tracer_index,
            route_result=route_result,
            lane_index=lane_index,
            guard_phase=guard_phase,
            guard_reason=guard_reason,
            guard_source_index=guard_source_index,
        ),
    )


def advance_cuda_restricted_wh_hybrid(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    *,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
) -> CUDARestrictedWHHybridResult:
    """Advance a segment, redoing guarded lanes with continuous DOP853."""

    if type(state) is not CUDARestrictedWHState:
        raise CUDARestrictedWHHybridContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(source_tile) is not FrozenSourcePathTile:
        raise CUDARestrictedWHHybridContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    tile_digest = verify_source_path_tile(source_tile)
    if type(spec) is not FixedStepCUDARestrictedWHSpec:
        raise CUDARestrictedWHHybridContractError(
            "spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    if route_spec is None:
        route_spec = RestrictedWHRouteSpec(
            maximum_primary_dynamical_phase=(
                spec.maximum_primary_dynamical_phase
            ),
            primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
        )
    if type(route_spec) is not RestrictedWHRouteSpec:
        raise CUDARestrictedWHHybridContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    if (
        type(route_worker_count) is not int
        or not 1 <= route_worker_count <= 32
    ):
        raise CUDARestrictedWHHybridContractError(
            "route_worker_count must be an integer in [1, 32]"
        )
    if (
        route_spec.maximum_primary_dynamical_phase
        != spec.maximum_primary_dynamical_phase
    ):
        raise CUDARestrictedWHHybridContractError(
            "CUDA and DOP853 primary phase ceilings must be exactly equal"
        )
    if route_spec.primary_outer_boundary_AU != spec.primary_outer_boundary_AU:
        raise CUDARestrictedWHHybridContractError(
            "CUDA and DOP853 primary outer boundaries must be exactly equal"
        )
    required_intervals = spec.map_step_count * spec.source_steps_per_map_step
    if source_tile.global_source_step != state.global_source_step:
        raise CUDARestrictedWHHybridContractError(
            "source tile must begin at state.global_source_step"
        )
    if source_tile.node_count < required_intervals + 1:
        raise CUDARestrictedWHHybridContractError(
            "source tile does not cover the complete hybrid segment"
        )

    resolved = resolve_backend(spec.backend)
    if resolved.name != "cupy":
        raise CUDARestrictedWHHybridContractError(
            "hybrid restricted-WH execution requires CuPy"
        )
    cp = resolved.xp
    ratio = spec.source_steps_per_map_step
    current_state = state
    current_map = 0
    route_records: list[CUDARestrictedWHRouteRecord] = []
    maximum_kepler_iterations = 0
    dispatcher = _RouteDispatcher(
        source_tile=source_tile,
        route_spec=route_spec,
        source_steps_per_map_step=ratio,
        worker_count=route_worker_count,
    )

    with resolved.activate():
        try:
            lane_count = int(state.tracer_indices.shape[0])
            cumulative_minimum = cp.full(lane_count, cp.inf, dtype=cp.float64)
            while current_map < spec.map_step_count:
                remaining_maps = spec.map_step_count - current_map
                start_interval = current_map * ratio
                candidate_tile = _slice_tile(
                    source_tile,
                    start_interval,
                    remaining_maps * ratio,
                )
                try:
                    candidate = advance_cuda_restricted_wh_fused(
                        current_state,
                        candidate_tile,
                        _child_spec(spec, remaining_maps),
                    )
                except CUDARestrictedWHIntegrationError as exc:
                    failed_map = exc.handoff.map_step_index
                    if failed_map > 0:
                        prefix_tile = _slice_tile(
                            source_tile,
                            start_interval,
                            failed_map * ratio,
                        )
                        prefix = advance_cuda_restricted_wh_fused(
                            current_state,
                            prefix_tile,
                            _child_spec(spec, failed_map),
                        )
                        cumulative_minimum = cp.minimum(
                            cumulative_minimum,
                            prefix.minimum_endpoint_source_distance_AU,
                        )
                        maximum_kepler_iterations = max(
                            maximum_kepler_iterations,
                            prefix.maximum_kepler_iterations,
                        )
                        current_state = prefix.state
                        current_map += failed_map
                        continue
                    current_state, step_minimum, step_maximum, records = (
                        _redo_guarded_wave(
                            input_state=state,
                            current_state=current_state,
                            source_tile=source_tile,
                            spec=spec,
                            map_step_index=current_map,
                            cp=cp,
                            dispatcher=dispatcher,
                        )
                    )
                    cumulative_minimum = cp.minimum(
                        cumulative_minimum, step_minimum
                    )
                    maximum_kepler_iterations = max(
                        maximum_kepler_iterations, step_maximum
                    )
                    route_records.extend(records)
                    current_map += 1
                    continue
                cumulative_minimum = cp.minimum(
                    cumulative_minimum,
                    candidate.minimum_endpoint_source_distance_AU,
                )
                maximum_kepler_iterations = max(
                    maximum_kepler_iterations,
                    candidate.maximum_kepler_iterations,
                )
                current_state = candidate.state
                current_map = spec.map_step_count

            output_indices = cp.array(
                current_state.tracer_indices, dtype=cp.uint64, copy=True, order="C"
            )
            current_state = replace(current_state, tracer_indices=output_indices)
            cp.cuda.get_current_stream().synchronize()
            return CUDARestrictedWHHybridResult(
                state=current_state,
                source_tile_sha256=tile_digest,
                minimum_source_distance_AU=cumulative_minimum,
                maximum_kepler_iterations=maximum_kepler_iterations,
                route_records=tuple(route_records),
                runtime_identity={
                    "method_id": HYBRID_METHOD_ID,
                    "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
                    "quiet_cuda_runtime_identity": _runtime_identity(
                        cp,
                        resolved.device,
                    ),
                    "route_runtime_identities": tuple(
                        record.route_result.runtime_identity
                        for record in route_records
                    ),
                    "route_policy": restricted_wh_route_policy_identity(
                        route_spec
                    ),
                    "route_worker_count": route_worker_count,
                    "route_worker_start_method": (
                        "SERIAL" if route_worker_count == 1 else "spawn"
                    ),
                    "source_tile_sha256": tile_digest,
                    "determinism_scope": (
                        "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
                    ),
                },
            )
        except CUDARestrictedWHHybridIntegrationError:
            raise
        except CUDARestrictedWHContractError as exc:
            raise CUDARestrictedWHHybridContractError(str(exc)) from exc
        except Exception as exc:
            raise _failure(
                input_state=state,
                tile=source_tile,
                map_step_index=current_map,
                reason=f"HYBRID_RUNTIME_{type(exc).__name__}",
                tracer_index=None,
                route_result=None,
            ) from exc
        finally:
            dispatcher.close()


def _redo_guarded_wave(
    *,
    input_state: CUDARestrictedWHState,
    current_state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    map_step_index: int,
    cp: Any,
    dispatcher: _RouteDispatcher,
) -> tuple[
    CUDARestrictedWHState,
    Any,
    int,
    tuple[CUDARestrictedWHRouteRecord, ...],
]:
    ratio = spec.source_steps_per_map_step
    start_interval = map_step_index * ratio
    step_tile = _slice_tile(source_tile, start_interval, ratio)
    one_step_spec = _child_spec(spec, 1)
    original_indices = cp.asnumpy(current_state.tracer_indices)
    if original_indices.dtype != np.dtype(np.uint64):
        raise CUDARestrictedWHHybridContractError(
            "hybrid input tracer indices must be uint64"
        )
    destination = {
        int(tracer_index): lane
        for lane, tracer_index in enumerate(original_indices.tolist())
    }
    remaining_state = current_state
    jobs: list[_RouteJob] = []
    quiet: CUDARestrictedWHResult | None = None

    while int(remaining_state.tracer_indices.shape[0]):
        try:
            quiet = advance_cuda_restricted_wh_fused(
                remaining_state,
                step_tile,
                one_step_spec,
            )
            break
        except CUDARestrictedWHIntegrationError as exc:
            handoff = exc.handoff
            if (
                handoff.map_step_index != 0
                or handoff.lane_index is None
                or handoff.tracer_index is None
                or not _routable(handoff.reason)
            ):
                raise _failure(
                    input_state=input_state,
                    tile=source_tile,
                    map_step_index=map_step_index,
                    reason=handoff.reason,
                    tracer_index=handoff.tracer_index,
                    route_result=None,
                ) from exc
            lane = handoff.lane_index
            tracer_index = handoff.tracer_index
            position = np.ascontiguousarray(
                cp.asnumpy(remaining_state.tracer_relative_positions_AU[lane])
            )
            velocity = np.ascontiguousarray(
                cp.asnumpy(remaining_state.tracer_relative_velocities_AU_per_year[lane])
            )
            original_lane = destination[tracer_index]
            jobs.append(
                _RouteJob(
                    original_lane=original_lane,
                    tracer_index=tracer_index,
                    guard_phase=handoff.phase,
                    guard_reason=handoff.reason,
                    guard_source_index=handoff.source_index,
                    source_node_index=start_interval,
                    position=position,
                    velocity=velocity,
                )
            )
            keep = cp.arange(
                int(remaining_state.tracer_indices.shape[0]), dtype=cp.int64
            ) != cp.int64(lane)
            remaining_state = replace(
                remaining_state,
                tracer_relative_positions_AU=cp.ascontiguousarray(
                    remaining_state.tracer_relative_positions_AU[keep]
                ),
                tracer_relative_velocities_AU_per_year=cp.ascontiguousarray(
                    remaining_state.tracer_relative_velocities_AU_per_year[keep]
                ),
                tracer_indices=cp.ascontiguousarray(
                    remaining_state.tracer_indices[keep]
                ),
            )

    ordered_jobs = tuple(sorted(jobs, key=lambda item: item.original_lane))
    try:
        route_results = dispatcher.route(ordered_jobs)
    except _RouteDispatchFailure as dispatch_error:
        cause = dispatch_error.cause
        raise _failure(
            input_state=input_state,
            tile=source_tile,
            map_step_index=map_step_index,
            reason=f"DOP853_{type(cause).__name__}",
            tracer_index=dispatch_error.job.tracer_index,
            route_result=None,
        ) from cause

    routed: list[
        tuple[int, RestrictedWHRouteResult, CUDARestrictedWHRouteRecord]
    ] = []
    for job, route in zip(ordered_jobs, route_results, strict=True):
        if route.terminated_at_event:
            raise _failure(
                input_state=input_state,
                tile=source_tile,
                map_step_index=map_step_index,
                reason="TERMINAL_EVENT",
                tracer_index=job.tracer_index,
                route_result=route,
                lane_index=job.original_lane,
                guard_phase=job.guard_phase,
                guard_reason=job.guard_reason,
                guard_source_index=job.guard_source_index,
            )
        record = CUDARestrictedWHRouteRecord(
            map_step_index=map_step_index,
            lane_index=job.original_lane,
            tracer_index=job.tracer_index,
            guard_phase=job.guard_phase,
            guard_reason=job.guard_reason,
            guard_source_index=job.guard_source_index,
            route_result=route,
        )
        routed.append((job.original_lane, route, record))

    lane_count = len(original_indices)
    next_positions = cp.empty_like(current_state.tracer_relative_positions_AU)
    next_velocities = cp.empty_like(current_state.tracer_relative_velocities_AU_per_year)
    step_minimum = cp.empty(lane_count, dtype=cp.float64)
    maximum_iterations = 0
    if quiet is not None:
        quiet_indices = cp.asnumpy(quiet.state.tracer_indices).tolist()
        quiet_destinations = cp.asarray(
            [destination[int(value)] for value in quiet_indices],
            dtype=cp.int64,
        )
        next_positions[quiet_destinations] = quiet.state.tracer_relative_positions_AU
        next_velocities[quiet_destinations] = quiet.state.tracer_relative_velocities_AU_per_year
        step_minimum[quiet_destinations] = quiet.minimum_endpoint_source_distance_AU
        maximum_iterations = quiet.maximum_kepler_iterations
    for original_lane, route, _record in routed:
        next_positions[original_lane] = cp.asarray(
            route.endpoint_relative_position_AU, dtype=cp.float64
        )
        next_velocities[original_lane] = cp.asarray(
            route.endpoint_relative_velocity_AU_per_year, dtype=cp.float64
        )
        step_minimum[original_lane] = cp.float64(
            route.minimum_committed_stage_distance_AU
        )
    next_indices = cp.array(
        current_state.tracer_indices,
        dtype=cp.uint64,
        copy=True,
        order="C",
    )
    cp.cuda.get_current_stream().synchronize()
    next_state = replace(
        current_state,
        global_source_step=current_state.global_source_step + ratio,
        tracer_relative_positions_AU=next_positions,
        tracer_relative_velocities_AU_per_year=next_velocities,
        tracer_indices=next_indices,
    )
    records = tuple(
        record
        for _lane, _route, record in sorted(
            routed, key=lambda item: (item[0], item[2].tracer_index)
        )
    )
    return next_state, step_minimum, maximum_iterations, records


__all__ = [
    "CUDARestrictedWHHybridContractError",
    "CUDARestrictedWHHybridError",
    "CUDARestrictedWHHybridFailureHandoff",
    "CUDARestrictedWHHybridIntegrationError",
    "CUDARestrictedWHHybridResult",
    "CUDARestrictedWHRouteRecord",
    "DEFAULT_ROUTE_WORKER_COUNT",
    "HYBRID_CLAIMS",
    "HYBRID_METHOD_ID",
    "IMPLEMENTATION_SOURCE_SHA256",
    "PROJECT_SCOPE",
    "ROUTABLE_REASONS",
    "SCIENTIFIC_CLASSIFICATION",
    "advance_cuda_restricted_wh_hybrid",
]
