"""Persistent-worker episode routing for the restricted WH prototype.

This additive scheduler preserves the committed numerical path of
``cuda_restricted_wh_episode`` while reducing Python/process dispatch.  A
tracer that enters a maximal consecutive routed episode is assigned to one
dedicated spawned CPU worker.  That worker computes a small, bounded chain of
prospective DOP853 map-step routes.  A prospective result is committed only
after the CUDA classifier independently requests a route at that exact next
map step and the binary64 route-start vectors match bit-for-bit.  A quiet CUDA
step releases the owner and discards every unused prospective result.

The implementation is experimental and direct-module only.  It has no disk
checkpoint contract.  Every production, science, accuracy, and long-horizon
claim remains false.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import multiprocessing
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
    CUDARestrictedWHContractError,
    CUDARestrictedWHIntegrationError,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    _runtime_identity,
    advance_cuda_restricted_wh_fused,
    verify_source_path_tile,
)
from .cuda_restricted_wh_episode import (
    IMPLEMENTATION_SOURCE_SHA256 as EPISODE_SOURCE_SHA256,
    CUDARestrictedWHRouteEpisode,
    _classify_and_route_step,
    group_restricted_wh_route_episodes,
)
from .cuda_restricted_wh_hybrid import (
    DEFAULT_ROUTE_WORKER_COUNT,
    IMPLEMENTATION_SOURCE_SHA256 as HYBRID_SOURCE_SHA256,
    CUDARestrictedWHHybridContractError,
    CUDARestrictedWHHybridFailureHandoff,
    CUDARestrictedWHHybridIntegrationError,
    CUDARestrictedWHRouteRecord,
    _RouteDispatchFailure,
    _RouteJob,
    _child_spec,
    _freeze_tree,
    _route_from_payload,
    _route_payload,
    _slice_tile,
)
from .engine.backends import resolve_backend
from .restricted_wh_route import (
    RestrictedWHRouteResult,
    RestrictedWHRouteSpec,
    _VerifiedRouteContext,
    _prepare_verified_route_context,
    _route_restricted_wh_map_step_verified,
    restricted_wh_route_policy_identity,
)


OWNED_EPISODE_METHOD_ID = "JX_CUDA_RESTRICTED_WH_OWNED_EPISODE_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
DEFAULT_ROUTE_LOOKAHEAD_STEPS = 32
MAXIMUM_ROUTE_LOOKAHEAD_STEPS = 64
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OWNED_EPISODE_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "close_encounter_qualified": False,
        "whole_encounter_episode_qualified": False,
        "speculative_route_lookahead_qualified": False,
        "collision_response_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "science_result": False,
    }
)


@dataclass(frozen=True)
class _OwnedChainRequest:
    job: _RouteJob
    maximum_steps: int


@dataclass(frozen=True)
class _OwnedChainResponse:
    payloads: tuple[Any, ...]
    failure_offset: int | None
    failure: _SerializedRouteFailure | None


@dataclass(frozen=True)
class _SerializedRouteFailure:
    type_name: str
    message: str


@dataclass(frozen=True)
class _CachedRouteOutcome:
    source_node_index: int
    position: np.ndarray
    velocity: np.ndarray
    payload: Any | None
    failure: _SerializedRouteFailure | None


_OWNED_SOURCE_TILE: FrozenSourcePathTile | None = None
_OWNED_ROUTE_CONTEXT: _VerifiedRouteContext | None = None
_OWNED_ROUTE_SPEC: RestrictedWHRouteSpec | None = None
_OWNED_SOURCE_STEPS_PER_MAP_STEP: int | None = None


def _initialize_owned_worker(
    source_tile: FrozenSourcePathTile,
    route_spec: RestrictedWHRouteSpec,
    source_steps_per_map_step: int,
) -> None:
    global _OWNED_SOURCE_TILE
    global _OWNED_ROUTE_CONTEXT
    global _OWNED_ROUTE_SPEC
    global _OWNED_SOURCE_STEPS_PER_MAP_STEP
    _OWNED_SOURCE_TILE = source_tile
    _OWNED_ROUTE_CONTEXT = _prepare_verified_route_context(source_tile)
    _OWNED_ROUTE_SPEC = route_spec
    _OWNED_SOURCE_STEPS_PER_MAP_STEP = source_steps_per_map_step


def _execute_owned_chain(request: _OwnedChainRequest) -> _OwnedChainResponse:
    if (
        _OWNED_SOURCE_TILE is None
        or _OWNED_ROUTE_CONTEXT is None
        or _OWNED_ROUTE_SPEC is None
        or _OWNED_SOURCE_STEPS_PER_MAP_STEP is None
    ):
        raise RuntimeError("owned route worker was not initialized")
    return _execute_owned_chain_core(
        request,
        _OWNED_ROUTE_CONTEXT,
        _OWNED_ROUTE_SPEC,
        _OWNED_SOURCE_STEPS_PER_MAP_STEP,
    )


def _execute_owned_chain_core(
    request: _OwnedChainRequest,
    route_context: _VerifiedRouteContext,
    route_spec: RestrictedWHRouteSpec,
    source_steps_per_map_step: int,
) -> _OwnedChainResponse:
    job = request.job
    position = np.ascontiguousarray(job.position, dtype=np.float64)
    velocity = np.ascontiguousarray(job.velocity, dtype=np.float64)
    payloads: list[Any] = []
    for offset in range(request.maximum_steps):
        try:
            result = _route_restricted_wh_map_step_verified(
                route_context,
                position,
                velocity,
                tracer_index=job.tracer_index,
                source_node_index=(
                    job.source_node_index
                    + offset * source_steps_per_map_step
                ),
                source_steps_per_map_step=source_steps_per_map_step,
                spec=route_spec,
            )
        except Exception as exc:
            return _OwnedChainResponse(
                payloads=tuple(payloads),
                failure_offset=offset,
                failure=_SerializedRouteFailure(
                    type_name=type(exc).__name__, message=str(exc)
                ),
            )
        payloads.append(_route_payload(result))
        if result.terminated_at_event:
            break
        assert result.endpoint_relative_position_AU is not None
        assert result.endpoint_relative_velocity_AU_per_year is not None
        position = np.ascontiguousarray(result.endpoint_relative_position_AU)
        velocity = np.ascontiguousarray(result.endpoint_relative_velocity_AU_per_year)
    return _OwnedChainResponse(
        payloads=tuple(payloads), failure_offset=None, failure=None
    )


class _OwnedRouteContinuityError(RuntimeError):
    """A prospective route did not start from the GPU-classified state."""


def _materialize_route_failure(failure: _SerializedRouteFailure) -> Exception:
    if (
        type(failure) is not _SerializedRouteFailure
        or type(failure.type_name) is not str
        or not failure.type_name
        or type(failure.message) is not str
    ):
        return RuntimeError("owned route worker returned an invalid failure")
    remote_type = type(failure.type_name, (RuntimeError,), {})
    return remote_type(failure.message)


class _PersistentOwnedRouteDispatcher:
    """Deterministic dedicated-worker routing with bounded lookahead."""

    def __init__(
        self,
        *,
        source_tile: FrozenSourcePathTile,
        route_spec: RestrictedWHRouteSpec,
        source_steps_per_map_step: int,
        worker_count: int,
        lookahead_steps: int,
        stop_source_node_index: int,
    ) -> None:
        self.source_tile = source_tile
        self.route_spec = route_spec
        self.source_steps_per_map_step = source_steps_per_map_step
        self.worker_count = worker_count
        self.lookahead_steps = lookahead_steps
        self.stop_source_node_index = stop_source_node_index
        self.executors: list[ProcessPoolExecutor | None] = [
            None for _ in range(worker_count)
        ]
        self.owners: dict[tuple[int, int], int] = {}
        self.cache: dict[tuple[int, int], deque[_CachedRouteOutcome]] = {}
        self.last_assignments: tuple[int, ...] = ()
        self.local_route_context = (
            _prepare_verified_route_context(source_tile)
            if worker_count == 1
            else None
        )
        self.chain_submission_count = 0
        self.speculative_step_count = 0
        self.committed_cached_step_count = 0
        self.discarded_speculative_step_count = 0
        self.new_episode_assignment_count = 0
        self.continuation_route_count = 0

    def _executor(self, worker_index: int) -> ProcessPoolExecutor:
        executor = self.executors[worker_index]
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_owned_worker,
                initargs=(
                    self.source_tile,
                    self.route_spec,
                    self.source_steps_per_map_step,
                ),
            )
            self.executors[worker_index] = executor
        return executor

    @staticmethod
    def _key(job: _RouteJob) -> tuple[int, int]:
        return (job.original_lane, job.tracer_index)

    def _release_absent(self, current_keys: set[tuple[int, int]]) -> None:
        for key in tuple(self.owners):
            if key in current_keys:
                continue
            pending = self.cache.pop(key, deque())
            self.discarded_speculative_step_count += len(pending)
            del self.owners[key]

    def _assign_owner(self, key: tuple[int, int]) -> int:
        if key in self.owners:
            return self.owners[key]
        loads = [0 for _ in range(self.worker_count)]
        for owner in self.owners.values():
            loads[owner] += 1
        worker_index = min(range(self.worker_count), key=lambda item: (loads[item], item))
        self.owners[key] = worker_index
        self.new_episode_assignment_count += 1
        return worker_index

    @staticmethod
    def _validate_cached_start(
        job: _RouteJob, outcome: _CachedRouteOutcome
    ) -> None:
        if outcome.source_node_index != job.source_node_index:
            raise _OwnedRouteContinuityError(
                "owned route source-node continuity changed"
            )
        for actual, expected, label in (
            (job.position, outcome.position, "position"),
            (job.velocity, outcome.velocity, "velocity"),
        ):
            actual_array = np.asarray(actual)
            if (
                actual_array.dtype != np.dtype(np.float64)
                or actual_array.shape != (3,)
                or not actual_array.flags.c_contiguous
                or not np.array_equal(
                    actual_array.view(np.uint64), expected.view(np.uint64)
                )
            ):
                raise _OwnedRouteContinuityError(
                    f"owned route {label} continuity changed"
                )

    def _install_response(
        self,
        key: tuple[int, int],
        request: _OwnedChainRequest,
        response: _OwnedChainResponse,
    ) -> None:
        if type(response) is not _OwnedChainResponse:
            raise RuntimeError("owned route worker returned an invalid response")
        job = request.job
        position = np.ascontiguousarray(job.position, dtype=np.float64)
        velocity = np.ascontiguousarray(job.velocity, dtype=np.float64)
        outcomes: deque[_CachedRouteOutcome] = deque()
        for offset, payload in enumerate(response.payloads):
            outcomes.append(
                _CachedRouteOutcome(
                    source_node_index=(
                        job.source_node_index
                        + offset * self.source_steps_per_map_step
                    ),
                    position=position,
                    velocity=velocity,
                    payload=payload,
                    failure=None,
                )
            )
            result = _route_from_payload(payload)
            if result.terminated_at_event:
                break
            assert result.endpoint_relative_position_AU is not None
            assert result.endpoint_relative_velocity_AU_per_year is not None
            position = np.ascontiguousarray(result.endpoint_relative_position_AU)
            velocity = np.ascontiguousarray(
                result.endpoint_relative_velocity_AU_per_year
            )
        if response.failure_offset is not None:
            if response.failure is None:
                raise RuntimeError("owned route worker omitted its failure")
            if response.failure_offset != len(response.payloads):
                raise RuntimeError("owned route worker returned a noncanonical failure")
            outcomes.append(
                _CachedRouteOutcome(
                    source_node_index=(
                        job.source_node_index
                        + response.failure_offset * self.source_steps_per_map_step
                    ),
                    position=position,
                    velocity=velocity,
                    payload=None,
                    failure=response.failure,
                )
            )
        if not outcomes:
            raise RuntimeError("owned route worker returned an empty chain")
        self.speculative_step_count += len(outcomes)
        self.cache[key] = outcomes

    def route(
        self, jobs: tuple[_RouteJob, ...]
    ) -> tuple[RestrictedWHRouteResult, ...]:
        if type(jobs) is not tuple:
            raise RuntimeError("owned route jobs must be an exact tuple")
        keys = tuple(self._key(job) for job in jobs)
        if len(keys) != len(set(keys)):
            raise RuntimeError("owned route jobs contain a duplicate tracer")
        self._release_absent(set(keys))
        if not jobs:
            self.last_assignments = ()
            return ()

        assignments: list[int] = []
        cache_hit_indices: set[int] = set()
        futures: dict[int, tuple[_RouteJob, _OwnedChainRequest, Future[Any]]] = {}
        for item_index, (job, key) in enumerate(zip(jobs, keys, strict=True)):
            was_owned = key in self.owners
            owner = self._assign_owner(key)
            assignments.append(owner)
            pending = self.cache.get(key)
            if pending:
                self.continuation_route_count += 1
                cache_hit_indices.add(item_index)
                continue
            if was_owned:
                self.continuation_route_count += 1
            remaining_steps = (
                self.stop_source_node_index - job.source_node_index
            ) // self.source_steps_per_map_step
            # The first routed step is intentionally non-speculative.  Only a
            # second consecutive CUDA route request opens the bounded lookahead,
            # avoiding up to N wasted DOP853 solves for isolated guard chatter.
            maximum_steps = min(
                self.lookahead_steps if was_owned else 1,
                remaining_steps,
            )
            if maximum_steps <= 0:
                raise _RouteDispatchFailure(
                    job, RuntimeError("owned route extends past the segment horizon")
                )
            request = _OwnedChainRequest(job=job, maximum_steps=maximum_steps)
            self.chain_submission_count += 1
            if self.worker_count == 1:
                try:
                    if self.local_route_context is None:
                        raise RuntimeError(
                            "serial owned route context was not initialized"
                        )
                    response = _execute_owned_chain_local(
                        request,
                        self.local_route_context,
                        self.route_spec,
                        self.source_steps_per_map_step,
                    )
                    self._install_response(key, request, response)
                except Exception as exc:
                    raise _RouteDispatchFailure(job, exc) from exc
            else:
                future = self._executor(owner).submit(_execute_owned_chain, request)
                futures[item_index] = (job, request, future)

        for item_index in range(len(jobs)):
            if item_index not in futures:
                continue
            job, request, future = futures[item_index]
            try:
                response = future.result()
                self._install_response(keys[item_index], request, response)
            except Exception as exc:
                for _other_job, _other_request, pending in futures.values():
                    pending.cancel()
                raise _RouteDispatchFailure(job, exc) from exc

        results: list[RestrictedWHRouteResult] = []
        for item_index, (job, key) in enumerate(zip(jobs, keys, strict=True)):
            pending = self.cache.get(key)
            if not pending:
                raise _RouteDispatchFailure(
                    job, RuntimeError("owned route cache unexpectedly became empty")
                )
            outcome = pending.popleft()
            try:
                self._validate_cached_start(job, outcome)
            except Exception as exc:
                raise _RouteDispatchFailure(job, exc) from exc
            if outcome.failure is not None:
                raise _RouteDispatchFailure(
                    job, _materialize_route_failure(outcome.failure)
                )
            if outcome.payload is None:
                raise _RouteDispatchFailure(
                    job, RuntimeError("owned route cache omitted its payload")
                )
            try:
                results.append(_route_from_payload(outcome.payload))
            except Exception as exc:
                raise _RouteDispatchFailure(job, exc) from exc
            if item_index in cache_hit_indices:
                self.committed_cached_step_count += 1
            if not pending:
                self.cache.pop(key, None)
        self.last_assignments = tuple(assignments)
        return tuple(results)

    def close(self) -> None:
        for pending in self.cache.values():
            self.discarded_speculative_step_count += len(pending)
        self.cache.clear()
        self.owners.clear()
        for executor in self.executors:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)


def _execute_owned_chain_local(
    request: _OwnedChainRequest,
    route_context: _VerifiedRouteContext,
    route_spec: RestrictedWHRouteSpec,
    source_steps_per_map_step: int,
) -> _OwnedChainResponse:
    return _execute_owned_chain_core(
        request, route_context, route_spec, source_steps_per_map_step
    )


def _assignment_sha256(
    records: tuple[CUDARestrictedWHRouteRecord, ...],
    worker_indices: tuple[int, ...],
) -> str:
    payload = [
        {
            "map_step_index": record.map_step_index,
            "lane_index": record.lane_index,
            "tracer_index": record.tracer_index,
            "worker_index": worker_index,
        }
        for record, worker_index in zip(records, worker_indices, strict=True)
    ]
    return hashlib.sha256(
        b"jx.cuda-restricted-wh-owned-assignments.v1\0"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHOwnedEpisodeResult:
    """One committed persistent-worker episode segment."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    minimum_source_distance_AU: object
    maximum_kepler_iterations: int
    route_records: tuple[CUDARestrictedWHRouteRecord, ...]
    episodes: tuple[CUDARestrictedWHRouteEpisode, ...]
    route_worker_indices: tuple[int, ...]
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = OWNED_EPISODE_CLAIMS
    method_id: str = OWNED_EPISODE_METHOD_ID
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
        if type(self.route_records) is not tuple or any(
            type(record) is not CUDARestrictedWHRouteRecord
            for record in self.route_records
        ):
            raise CUDARestrictedWHHybridContractError(
                "route_records must be an exact immutable tuple"
            )
        expected_episodes = group_restricted_wh_route_episodes(self.route_records)
        if type(self.episodes) is not tuple or self.episodes != expected_episodes:
            raise CUDARestrictedWHHybridContractError(
                "episodes must exactly group the canonical route ledger"
            )
        if (
            type(self.route_worker_indices) is not tuple
            or len(self.route_worker_indices) != len(self.route_records)
            or any(
                type(value) is not int or value < 0
                for value in self.route_worker_indices
            )
        ):
            raise CUDARestrictedWHHybridContractError(
                "route_worker_indices must exactly cover the route ledger"
            )
        if not isinstance(self.runtime_identity, Mapping):
            raise CUDARestrictedWHHybridContractError(
                "runtime_identity must be a mapping"
            )
        worker_count = self.runtime_identity.get("route_worker_count")
        lookahead_steps = self.runtime_identity.get("route_lookahead_steps")
        if (
            type(worker_count) is not int
            or not 1 <= worker_count <= 32
            or type(lookahead_steps) is not int
            or not 1 <= lookahead_steps <= MAXIMUM_ROUTE_LOOKAHEAD_STEPS
            or any(value >= worker_count for value in self.route_worker_indices)
        ):
            raise CUDARestrictedWHHybridContractError(
                "route worker identity is outside the bound execution policy"
            )
        if self.runtime_identity.get(
            "route_worker_assignment_sha256"
        ) != _assignment_sha256(self.route_records, self.route_worker_indices):
            raise CUDARestrictedWHHybridContractError(
                "route worker assignment digest changed"
            )
        worker_by_record = {
            (record.map_step_index, record.lane_index, record.tracer_index): worker
            for record, worker in zip(
                self.route_records, self.route_worker_indices, strict=True
            )
        }
        for episode in self.episodes:
            workers = {
                worker_by_record[
                    (record.map_step_index, record.lane_index, record.tracer_index)
                ]
                for record in episode.route_records
            }
            if len(workers) != 1:
                raise CUDARestrictedWHHybridContractError(
                    "a routed episode changed its dedicated worker"
                )
        if (
            self.method_id != OWNED_EPISODE_METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(OWNED_EPISODE_CLAIMS)
            or any(self.claims.values())
        ):
            raise CUDARestrictedWHHybridContractError(
                "owned-episode identity or claim ceiling changed"
            )
        object.__setattr__(
            self, "claims", MappingProxyType(dict(OWNED_EPISODE_CLAIMS))
        )
        object.__setattr__(self, "runtime_identity", _freeze_tree(self.runtime_identity))


def advance_cuda_restricted_wh_owned_episodes(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    *,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
    route_lookahead_steps: int = DEFAULT_ROUTE_LOOKAHEAD_STEPS,
) -> CUDARestrictedWHOwnedEpisodeResult:
    """Advance with persistent route owners and CUDA-certified commitments."""

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
        type(route_lookahead_steps) is not int
        or not 1 <= route_lookahead_steps <= MAXIMUM_ROUTE_LOOKAHEAD_STEPS
    ):
        raise CUDARestrictedWHHybridContractError(
            "route_lookahead_steps must be an integer in [1, 64]"
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
            "source tile does not cover the complete owned-episode segment"
        )

    resolved = resolve_backend(spec.backend)
    if resolved.name != "cupy":
        raise CUDARestrictedWHHybridContractError(
            "owned-episode restricted-WH execution requires CuPy"
        )
    cp = resolved.xp
    ratio = spec.source_steps_per_map_step
    current_state = state
    current_map = 0
    route_records: list[CUDARestrictedWHRouteRecord] = []
    route_worker_indices: list[int] = []
    maximum_kepler_iterations = 0
    partitioned_gpu_step_count = 0
    quiet_fused_attempt_count = 0
    episode_mode = False
    partition_kernel: Any | None = None
    dispatcher = _PersistentOwnedRouteDispatcher(
        source_tile=source_tile,
        route_spec=route_spec,
        source_steps_per_map_step=ratio,
        worker_count=route_worker_count,
        lookahead_steps=route_lookahead_steps,
        stop_source_node_index=required_intervals,
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
                if len(dispatcher.last_assignments) != len(step.records):
                    raise RuntimeError(
                        "owned worker assignments do not cover the routed wave"
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
                route_worker_indices.extend(dispatcher.last_assignments)
                current_map += 1
                episode_mode = bool(step.records)

            output_indices = cp.array(
                current_state.tracer_indices, dtype=cp.uint64, copy=True, order="C"
            )
            current_state = replace(current_state, tracer_indices=output_indices)
            cp.cuda.get_current_stream().synchronize()
            records = tuple(route_records)
            worker_indices = tuple(route_worker_indices)
            episodes = group_restricted_wh_route_episodes(records)
            assignment_digest = _assignment_sha256(records, worker_indices)
            runtime_identity = {
                "method_id": OWNED_EPISODE_METHOD_ID,
                "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
                "episode_source_sha256": EPISODE_SOURCE_SHA256,
                "reference_hybrid_source_sha256": HYBRID_SOURCE_SHA256,
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
                    "SERIAL_BOUNDED_LOOKAHEAD"
                    if route_worker_count == 1
                    else "DEDICATED_PROCESS_PER_OWNER_BOUNDED_LOOKAHEAD"
                ),
                "route_lookahead_steps": route_lookahead_steps,
                "route_chain_submission_count": dispatcher.chain_submission_count,
                "speculative_route_step_count": dispatcher.speculative_step_count,
                "committed_cached_route_step_count": (
                    dispatcher.committed_cached_step_count
                ),
                "discarded_speculative_route_step_count": (
                    dispatcher.discarded_speculative_step_count
                ),
                "new_episode_assignment_count": (
                    dispatcher.new_episode_assignment_count
                ),
                "continuation_route_count": dispatcher.continuation_route_count,
                "route_worker_assignment_sha256": assignment_digest,
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
            }
            return CUDARestrictedWHOwnedEpisodeResult(
                state=current_state,
                source_tile_sha256=tile_digest,
                minimum_source_distance_AU=cumulative_minimum,
                maximum_kepler_iterations=maximum_kepler_iterations,
                route_records=records,
                episodes=episodes,
                route_worker_indices=worker_indices,
                runtime_identity=runtime_identity,
            )
        except CUDARestrictedWHHybridIntegrationError:
            raise
        except CUDARestrictedWHHybridContractError:
            raise
        except CUDARestrictedWHContractError as exc:
            raise CUDARestrictedWHHybridContractError(str(exc)) from exc
        except Exception as exc:
            raise CUDARestrictedWHHybridIntegrationError(
                "owned-episode restricted-WH segment failed closed; input state is unchanged",
                CUDARestrictedWHHybridFailureHandoff(
                    state=state,
                    source_tile_sha256=tile_digest,
                    failure_map_step_index=current_map,
                    reason=f"OWNED_EPISODE_RUNTIME_{type(exc).__name__}",
                    tracer_index=None,
                    route_result=None,
                ),
            ) from exc
        finally:
            dispatcher.close()


__all__ = [
    "CUDARestrictedWHOwnedEpisodeResult",
    "DEFAULT_ROUTE_LOOKAHEAD_STEPS",
    "IMPLEMENTATION_SOURCE_SHA256",
    "MAXIMUM_ROUTE_LOOKAHEAD_STEPS",
    "OWNED_EPISODE_CLAIMS",
    "OWNED_EPISODE_METHOD_ID",
    "PROJECT_SCOPE",
    "SCIENTIFIC_CLASSIFICATION",
    "advance_cuda_restricted_wh_owned_episodes",
]
