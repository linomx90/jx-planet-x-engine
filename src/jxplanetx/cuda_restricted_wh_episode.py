"""Episode-aware CUDA/CPU scheduler for the restricted WH prototype.

The qualified guarded-step scheduler remains the numerical reference.  This
additive implementation changes only how a guarded wave is *discovered*: one
fused CUDA classification launch exposes every lane's first failure for the
current map step.  Quiet lanes commit the exact fused CUDA endpoint, while
guarded lanes are redone by the same continuous DOP853 route and the same
spawned CPU worker implementation used by the reference scheduler.

After the first guarded step, classification stays in episode mode until a
complete map step contains no routes.  Consecutive route records for a tracer
are committed as an immutable operational episode.  An episode is therefore
consecutive guarded macrosteps, not a claim about a physical encounter.

The component is experimental and direct-module only.  It has no disk
checkpoint contract and every production, science, accuracy, and close-
encounter claim remains false.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .cuda_restricted_wh import (
    CUDA_OPTIONS,
    FUSED_BATCH_KERNEL_SOURCE,
    FUSED_CUDA_SOURCE,
    FUSED_CUDA_SOURCE_SHA256,
    FUSED_KERNEL_ID,
    KICK_STATUS_NAMES,
    MAXIMUM_BINARY64,
    CUDARestrictedWHContractError,
    CUDARestrictedWHIntegrationError,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    _runtime_identity,
    _validate_compatibility,
    _validate_tracer_arrays,
    advance_cuda_restricted_wh_fused,
    verify_source_path_tile,
)
from ._all_conic_kepler import STATUS_NAMES as KEPLER_STATUS_NAMES
from .cuda_restricted_wh_hybrid import (
    DEFAULT_ROUTE_WORKER_COUNT,
    IMPLEMENTATION_SOURCE_SHA256 as REFERENCE_HYBRID_SOURCE_SHA256,
    ROUTABLE_REASONS,
    CUDARestrictedWHHybridContractError,
    CUDARestrictedWHHybridFailureHandoff,
    CUDARestrictedWHHybridIntegrationError,
    CUDARestrictedWHRouteRecord,
    _RouteDispatchFailure,
    _RouteDispatcher,
    _RouteJob,
    _child_spec,
    _freeze_tree,
    _slice_tile,
)
from .engine.backends import resolve_backend
from .restricted_wh_route import (
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
)


EPISODE_METHOD_ID = "JX_CUDA_RESTRICTED_WH_EPISODE_WAVE_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EPISODE_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "close_encounter_qualified": False,
        "whole_encounter_episode_qualified": False,
        "collision_response_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "science_result": False,
    }
)


def _routable(reason: str) -> bool:
    return reason in ROUTABLE_REASONS or reason.startswith("KEPLER_")


def _phase_name(phase_code: int) -> str:
    return {
        0: "PRE_KICK_GUARD",
        1: "KEPLER_DRIFT",
        2: "POST_KICK_GUARD",
    }.get(phase_code, "CUDA_RUNTIME")


def _reason_name(phase_code: int, status_code: int) -> str:
    if phase_code == 1:
        return f"KEPLER_{KEPLER_STATUS_NAMES.get(status_code, status_code)}"
    return KICK_STATUS_NAMES.get(status_code, f"KICK_STATUS_{status_code}")


@dataclass(frozen=True)
class CUDARestrictedWHRouteEpisode:
    """One tracer's maximal consecutive sequence of guarded macrosteps."""

    lane_index: int
    tracer_index: int
    first_map_step_index: int
    stop_map_step_index: int
    route_records: tuple[CUDARestrictedWHRouteRecord, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.lane_index, "lane_index"),
            (self.tracer_index, "tracer_index"),
            (self.first_map_step_index, "first_map_step_index"),
            (self.stop_map_step_index, "stop_map_step_index"),
        ):
            if type(value) is not int or value < 0:
                raise CUDARestrictedWHHybridContractError(
                    f"{label} must be a nonnegative integer"
                )
        if self.stop_map_step_index <= self.first_map_step_index:
            raise CUDARestrictedWHHybridContractError(
                "episode stop must be after its first map step"
            )
        if type(self.route_records) is not tuple or not self.route_records:
            raise CUDARestrictedWHHybridContractError(
                "episode route_records must be a nonempty exact tuple"
            )
        expected_steps = tuple(
            range(self.first_map_step_index, self.stop_map_step_index)
        )
        actual_steps = tuple(record.map_step_index for record in self.route_records)
        if actual_steps != expected_steps:
            raise CUDARestrictedWHHybridContractError(
                "episode records must cover consecutive map steps exactly"
            )
        if any(
            type(record) is not CUDARestrictedWHRouteRecord
            or record.lane_index != self.lane_index
            or record.tracer_index != self.tracer_index
            for record in self.route_records
        ):
            raise CUDARestrictedWHHybridContractError(
                "episode lane or tracer identity changed"
            )

    @property
    def route_count(self) -> int:
        return len(self.route_records)


def group_restricted_wh_route_episodes(
    route_records: tuple[CUDARestrictedWHRouteRecord, ...],
) -> tuple[CUDARestrictedWHRouteEpisode, ...]:
    """Group canonical route records into maximal per-tracer episodes."""

    if type(route_records) is not tuple or any(
        type(record) is not CUDARestrictedWHRouteRecord for record in route_records
    ):
        raise CUDARestrictedWHHybridContractError(
            "route_records must be an exact route-record tuple"
        )
    order = tuple(
        (record.map_step_index, record.lane_index, record.tracer_index)
        for record in route_records
    )
    if order != tuple(sorted(order)) or len(order) != len(set(order)):
        raise CUDARestrictedWHHybridContractError(
            "route records must be unique and canonically ordered"
        )

    by_tracer: dict[tuple[int, int], list[CUDARestrictedWHRouteRecord]] = (
        defaultdict(list)
    )
    for record in route_records:
        by_tracer[(record.lane_index, record.tracer_index)].append(record)

    episodes: list[CUDARestrictedWHRouteEpisode] = []
    for (lane_index, tracer_index), records in by_tracer.items():
        start = 0
        for cursor in range(1, len(records) + 1):
            closes = cursor == len(records) or (
                records[cursor].map_step_index
                != records[cursor - 1].map_step_index + 1
            )
            if not closes:
                continue
            group = tuple(records[start:cursor])
            episodes.append(
                CUDARestrictedWHRouteEpisode(
                    lane_index=lane_index,
                    tracer_index=tracer_index,
                    first_map_step_index=group[0].map_step_index,
                    stop_map_step_index=group[-1].map_step_index + 1,
                    route_records=group,
                )
            )
            start = cursor
    return tuple(
        sorted(
            episodes,
            key=lambda item: (
                item.first_map_step_index,
                item.lane_index,
                item.tracer_index,
            ),
        )
    )


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHEpisodeResult:
    """One completely committed episode-aware hybrid segment."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    minimum_source_distance_AU: object
    maximum_kepler_iterations: int
    route_records: tuple[CUDARestrictedWHRouteRecord, ...]
    episodes: tuple[CUDARestrictedWHRouteEpisode, ...]
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = EPISODE_CLAIMS
    method_id: str = EPISODE_METHOD_ID
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
            or any(
                character not in "0123456789abcdef"
                for character in self.source_tile_sha256
            )
        ):
            raise CUDARestrictedWHHybridContractError(
                "source_tile_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            type(self.maximum_kepler_iterations) is not int
            or self.maximum_kepler_iterations < 0
        ):
            raise CUDARestrictedWHHybridContractError(
                "maximum_kepler_iterations must be nonnegative"
            )
        expected_episodes = group_restricted_wh_route_episodes(self.route_records)
        if type(self.episodes) is not tuple or self.episodes != expected_episodes:
            raise CUDARestrictedWHHybridContractError(
                "episodes must exactly group the canonical route ledger"
            )
        if (
            self.method_id != EPISODE_METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(EPISODE_CLAIMS)
            or any(self.claims.values())
        ):
            raise CUDARestrictedWHHybridContractError(
                "episode identity or claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(EPISODE_CLAIMS)))
        object.__setattr__(self, "runtime_identity", _freeze_tree(self.runtime_identity))


@dataclass(frozen=True)
class _PartitionedStep:
    state: CUDARestrictedWHState
    minimum_distance_AU: Any
    maximum_kepler_iterations: int
    records: tuple[CUDARestrictedWHRouteRecord, ...]


def _classify_and_route_step(
    *,
    input_state: CUDARestrictedWHState,
    current_state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    map_step_index: int,
    cp: Any,
    resolved: Any,
    dispatcher: _RouteDispatcher,
    partition_kernel: Any,
) -> _PartitionedStep:
    """Advance one map step with one all-lane CUDA classification launch."""

    ratio = spec.source_steps_per_map_step
    start_interval = map_step_index * ratio
    step_tile = _slice_tile(source_tile, start_interval, ratio)
    one_step_spec = _child_spec(spec, 1)
    _validate_compatibility(current_state, step_tile, one_step_spec)
    arrays = _validate_tracer_arrays(current_state, resolved)
    lane_count = arrays.lane_count
    int32_max = int(np.iinfo(np.int32).max)
    if lane_count > int32_max:
        raise CUDARestrictedWHHybridContractError(
            "episode tracer lane count exceeds the signed CUDA int32 limit"
        )

    source_positions = cp.asarray(
        step_tile.source_relative_positions_AU, dtype=cp.float64
    )
    source_velocities = cp.asarray(
        step_tile.source_relative_velocities_AU_per_year, dtype=cp.float64
    )
    source_gm = cp.asarray(step_tile.source_gm_AU3_per_year2, dtype=cp.float64)
    source_radii = cp.asarray(
        step_tile.source_encounter_radii_AU, dtype=cp.float64
    )
    acceleration_bounds = cp.asarray(
        step_tile.certified_relative_acceleration_bounds_AU_per_year2,
        dtype=cp.float64,
    )
    positions = cp.array(arrays.positions, dtype=cp.float64, copy=True, order="C")
    velocities = cp.array(arrays.velocities, dtype=cp.float64, copy=True, order="C")
    minimum_distance = cp.full(lane_count, cp.inf, dtype=cp.float64)
    maximum_iterations = cp.zeros(lane_count, dtype=cp.int32)
    status = cp.zeros(lane_count, dtype=cp.int32)
    failure_map = cp.full(lane_count, -1, dtype=cp.int32)
    failure_phase = cp.full(lane_count, -1, dtype=cp.int32)
    failure_source = cp.full(lane_count, -1, dtype=cp.int32)

    full_step = step_tile.source_step_years * ratio
    primary_outer_boundary = (
        MAXIMUM_BINARY64
        if spec.primary_outer_boundary_AU is None
        else spec.primary_outer_boundary_AU
    )
    if lane_count:
        blocks = (lane_count + spec.threads_per_block - 1) // spec.threads_per_block
        partition_kernel(
            (blocks,),
            (spec.threads_per_block,),
            (
                arrays.positions,
                arrays.velocities,
                source_positions,
                source_velocities,
                source_gm,
                source_radii,
                acceleration_bounds,
                cp.int32(lane_count),
                cp.int32(len(step_tile.source_ids)),
                cp.int32(1),
                cp.int32(ratio),
                cp.float64(0.5 * full_step),
                cp.float64(full_step),
                cp.float64(spec.maximum_primary_dynamical_phase),
                cp.float64(primary_outer_boundary),
                positions,
                velocities,
                minimum_distance,
                maximum_iterations,
                status,
                failure_map,
                failure_phase,
                failure_source,
            ),
        )

    host_indices = cp.asnumpy(arrays.indices)
    host_failure_map = cp.asnumpy(failure_map)
    host_failure_phase = cp.asnumpy(failure_phase)
    host_status = cp.asnumpy(status)
    host_failure_source = cp.asnumpy(failure_source)
    failures: list[tuple[int, int, int, int, int, str, str, int | None]] = []
    for lane in range(lane_count):
        if int(host_failure_map[lane]) < 0:
            continue
        phase_code = int(host_failure_phase[lane])
        status_code = int(host_status[lane])
        source_value = int(host_failure_source[lane])
        tracer_index = int(host_indices[lane])
        phase = _phase_name(phase_code)
        reason = _reason_name(phase_code, status_code)
        failures.append(
            (
                phase_code,
                lane,
                tracer_index,
                status_code,
                source_value,
                phase,
                reason,
                source_value if source_value >= 0 else None,
            )
        )

    nonroutable = [failure for failure in failures if not _routable(failure[6])]
    if nonroutable:
        failure = min(nonroutable, key=lambda item: (item[0], item[1]))
        raise CUDARestrictedWHHybridIntegrationError(
            "episode restricted-WH segment failed closed; input state is unchanged",
            CUDARestrictedWHHybridFailureHandoff(
                state=input_state,
                source_tile_sha256=source_tile.content_sha256 or "",
                failure_map_step_index=map_step_index,
                reason=failure[6],
                tracer_index=failure[2],
                route_result=None,
                lane_index=failure[1],
                guard_phase=failure[5],
                guard_reason=failure[6],
                guard_source_index=failure[7],
            ),
        )

    host_positions = cp.asnumpy(arrays.positions) if failures else None
    host_velocities = cp.asnumpy(arrays.velocities) if failures else None
    jobs = tuple(
        _RouteJob(
            original_lane=failure[1],
            tracer_index=failure[2],
            guard_phase=failure[5],
            guard_reason=failure[6],
            guard_source_index=failure[7],
            source_node_index=start_interval,
            position=np.ascontiguousarray(host_positions[failure[1]]),
            velocity=np.ascontiguousarray(host_velocities[failure[1]]),
        )
        for failure in sorted(failures, key=lambda item: item[1])
    )
    try:
        routes = dispatcher.route(jobs)
    except _RouteDispatchFailure as dispatch_error:
        raise CUDARestrictedWHHybridIntegrationError(
            "episode restricted-WH segment failed closed; input state is unchanged",
            CUDARestrictedWHHybridFailureHandoff(
                state=input_state,
                source_tile_sha256=source_tile.content_sha256 or "",
                failure_map_step_index=map_step_index,
                reason=f"DOP853_{type(dispatch_error.cause).__name__}",
                tracer_index=dispatch_error.job.tracer_index,
                route_result=None,
            ),
        ) from dispatch_error.cause

    records: list[CUDARestrictedWHRouteRecord] = []
    for job, route in zip(jobs, routes, strict=True):
        if route.terminated_at_event:
            raise CUDARestrictedWHHybridIntegrationError(
                "episode restricted-WH segment failed closed; input state is unchanged",
                CUDARestrictedWHHybridFailureHandoff(
                    state=input_state,
                    source_tile_sha256=source_tile.content_sha256 or "",
                    failure_map_step_index=map_step_index,
                    reason="TERMINAL_EVENT",
                    tracer_index=job.tracer_index,
                    route_result=route,
                    lane_index=job.original_lane,
                    guard_phase=job.guard_phase,
                    guard_reason=job.guard_reason,
                    guard_source_index=job.guard_source_index,
                ),
            )
        records.append(
            CUDARestrictedWHRouteRecord(
                map_step_index=map_step_index,
                lane_index=job.original_lane,
                tracer_index=job.tracer_index,
                guard_phase=job.guard_phase,
                guard_reason=job.guard_reason,
                guard_source_index=job.guard_source_index,
                route_result=route,
            )
        )

    if jobs:
        routed_lanes = cp.asarray(
            np.asarray([job.original_lane for job in jobs], dtype=np.int64)
        )
        routed_positions = cp.asarray(
            np.stack(
                [route.endpoint_relative_position_AU for route in routes], axis=0
            ),
            dtype=cp.float64,
        )
        routed_velocities = cp.asarray(
            np.stack(
                [
                    route.endpoint_relative_velocity_AU_per_year
                    for route in routes
                ],
                axis=0,
            ),
            dtype=cp.float64,
        )
        routed_minima = cp.asarray(
            np.asarray(
                [
                    route.minimum_committed_stage_distance_AU
                    for route in routes
                ],
                dtype=np.float64,
            )
        )
        positions[routed_lanes] = routed_positions
        velocities[routed_lanes] = routed_velocities
        minimum_distance[routed_lanes] = routed_minima

    next_indices = cp.array(arrays.indices, dtype=cp.uint64, copy=True, order="C")
    cp.cuda.get_current_stream().synchronize()
    next_state = replace(
        current_state,
        global_source_step=current_state.global_source_step + ratio,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
        tracer_indices=next_indices,
    )
    if lane_count:
        quiet_mask = failure_map < 0
        quiet_count = int(cp.count_nonzero(quiet_mask).item())
        maximum = (
            int(cp.max(maximum_iterations[quiet_mask]).item())
            if quiet_count
            else 0
        )
    else:
        maximum = 0
    return _PartitionedStep(
        state=next_state,
        minimum_distance_AU=minimum_distance,
        maximum_kepler_iterations=maximum,
        records=tuple(records),
    )


def advance_cuda_restricted_wh_episodes(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    *,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
) -> CUDARestrictedWHEpisodeResult:
    """Advance with all-lane episode waves and exact DOP853 guarded redos."""

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
            maximum_primary_dynamical_phase=spec.maximum_primary_dynamical_phase,
            primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
        )
    if type(route_spec) is not RestrictedWHRouteSpec:
        raise CUDARestrictedWHHybridContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    if type(route_worker_count) is not int or not 1 <= route_worker_count <= 32:
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
            "source tile does not cover the complete episode segment"
        )

    resolved = resolve_backend(spec.backend)
    if resolved.name != "cupy":
        raise CUDARestrictedWHHybridContractError(
            "episode restricted-WH execution requires CuPy"
        )
    cp = resolved.xp
    ratio = spec.source_steps_per_map_step
    current_state = state
    current_map = 0
    route_records: list[CUDARestrictedWHRouteRecord] = []
    maximum_kepler_iterations = 0
    partitioned_gpu_step_count = 0
    quiet_fused_attempt_count = 0
    episode_mode = False
    partition_kernel: Any | None = None
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
                if not episode_mode:
                    remaining_maps = spec.map_step_count - current_map
                    start_interval = current_map * ratio
                    candidate_tile = _slice_tile(
                        source_tile, start_interval, remaining_maps * ratio
                    )
                    quiet_fused_attempt_count += 1
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
                                source_tile, start_interval, failed_map * ratio
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
                        episode_mode = True
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
                    continue

                if partition_kernel is None:
                    partition_kernel = cp.RawKernel(
                        FUSED_CUDA_SOURCE,
                        "jx_restricted_wh_fused_segment",
                        options=CUDA_OPTIONS,
                    )
                step = _classify_and_route_step(
                    input_state=state,
                    current_state=current_state,
                    source_tile=source_tile,
                    spec=spec,
                    route_spec=route_spec,
                    map_step_index=current_map,
                    cp=cp,
                    resolved=resolved,
                    dispatcher=dispatcher,
                    partition_kernel=partition_kernel,
                )
                partitioned_gpu_step_count += 1
                cumulative_minimum = cp.minimum(
                    cumulative_minimum, step.minimum_distance_AU
                )
                maximum_kepler_iterations = max(
                    maximum_kepler_iterations,
                    step.maximum_kepler_iterations,
                )
                current_state = step.state
                route_records.extend(step.records)
                current_map += 1
                episode_mode = bool(step.records)

            output_indices = cp.array(
                current_state.tracer_indices, dtype=cp.uint64, copy=True, order="C"
            )
            current_state = replace(current_state, tracer_indices=output_indices)
            cp.cuda.get_current_stream().synchronize()
            records = tuple(route_records)
            episodes = group_restricted_wh_route_episodes(records)
            return CUDARestrictedWHEpisodeResult(
                state=current_state,
                source_tile_sha256=tile_digest,
                minimum_source_distance_AU=cumulative_minimum,
                maximum_kepler_iterations=maximum_kepler_iterations,
                route_records=records,
                episodes=episodes,
                runtime_identity={
                    "method_id": EPISODE_METHOD_ID,
                    "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
                    "reference_hybrid_source_sha256": (
                        REFERENCE_HYBRID_SOURCE_SHA256
                    ),
                    "fused_kernel_id": FUSED_KERNEL_ID,
                    "fused_cuda_source_sha256": FUSED_CUDA_SOURCE_SHA256,
                    "fused_batch_kernel_sha256": hashlib.sha256(
                        FUSED_BATCH_KERNEL_SOURCE.encode("ascii")
                    ).hexdigest(),
                    "cuda_options": tuple(CUDA_OPTIONS),
                    "quiet_cuda_runtime_identity": _runtime_identity(
                        cp,
                        resolved.device,
                        execution_kernel_id=FUSED_KERNEL_ID,
                        execution_kernel_source_sha256=FUSED_CUDA_SOURCE_SHA256,
                    ),
                    "route_runtime_identities": tuple(
                        record.route_result.runtime_identity for record in records
                    ),
                    "route_policy": restricted_wh_route_policy_identity(route_spec),
                    "route_worker_count": route_worker_count,
                    "route_worker_start_method": (
                        "SERIAL" if route_worker_count == 1 else "spawn"
                    ),
                    "route_dispatch_policy": (
                        "SERIAL"
                        if route_worker_count == 1
                        else "REFERENCE_ONE_FUTURE_PER_ROUTE"
                    ),
                    "partitioned_gpu_step_count": partitioned_gpu_step_count,
                    "quiet_fused_attempt_count": quiet_fused_attempt_count,
                    "episode_count": len(episodes),
                    "longest_episode_route_count": max(
                        (episode.route_count for episode in episodes), default=0
                    ),
                    "source_tile_sha256": tile_digest,
                    "determinism_scope": (
                        "SAME_PYTHON_NUMPY_SCIPY_CUPY_CUDA_RUNTIME_AND_DEVICE"
                    ),
                },
            )
        except CUDARestrictedWHHybridIntegrationError:
            raise
        except CUDARestrictedWHHybridContractError:
            raise
        except CUDARestrictedWHContractError as exc:
            raise CUDARestrictedWHHybridContractError(str(exc)) from exc
        except Exception as exc:
            raise CUDARestrictedWHHybridIntegrationError(
                "episode restricted-WH segment failed closed; input state is unchanged",
                CUDARestrictedWHHybridFailureHandoff(
                    state=state,
                    source_tile_sha256=tile_digest,
                    failure_map_step_index=current_map,
                    reason=f"EPISODE_RUNTIME_{type(exc).__name__}",
                    tracer_index=None,
                    route_result=None,
                ),
            ) from exc
        finally:
            dispatcher.close()


__all__ = [
    "CUDARestrictedWHEpisodeResult",
    "CUDARestrictedWHRouteEpisode",
    "EPISODE_CLAIMS",
    "EPISODE_METHOD_ID",
    "IMPLEMENTATION_SOURCE_SHA256",
    "PROJECT_SCOPE",
    "SCIENTIFIC_CLASSIFICATION",
    "advance_cuda_restricted_wh_episodes",
    "group_restricted_wh_route_episodes",
]
