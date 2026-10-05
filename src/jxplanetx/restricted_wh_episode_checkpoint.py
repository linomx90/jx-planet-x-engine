"""Delta checkpoints for the persistent-owner restricted-WH prototype.

This module is additive and direct-import only.  One checkpoint record stores
one fully committed owned-episode segment.  Route witnesses, worker ownership,
and episode summaries are *deltas*: a successor never repeats any predecessor
route row.  A rolling route-chain digest and exact cumulative counters bind the
ordered history without turning every checkpoint into a cumulative ledger
copy.

Loading is precision-first and intentionally expensive.  The complete chain
is hash-checked and every segment is deterministically replayed on the exact
bound Python/NumPy/SciPy/CuPy/CPU/CUDA runtime before the latest state is
released.  Checkpoint boundaries terminate in-memory speculative caches;
worker ownership is therefore proved per persisted segment, not claimed to
survive a process restart.

A locked append session may amortize predecessor verification inside one
process.  It fully replays an externally anchored existing chain once, rehashes
every committed JSON and NPZ before every append, deterministically replays
only the new segment, and derives its next private authorization capability
from the bytes read back after create-only publication.  The capability is
bound to one process, thread, implementation, runtime, and namespace lock.
Closing or poisoning the session discards it; reopening and every public load
again require full replay from genesis.

Publication is create-only.  The content-addressed NumPy archive is committed
before its canonical JSON record, so a process crash may leave an unreferenced
archive but cannot expose a record whose arrays are absent.  An external head
index and digest are mandatory when loading, which makes tail deletion fail
closed within the stated local-filesystem threat model.

No production, long-horizon accuracy, physical encounter, collision,
ejection, completion, or science claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import cuda_restricted_wh as _restricted_module
from . import cuda_restricted_wh_episode as _episode_module
from . import cuda_restricted_wh_episode_owner as _owner_module
from . import restricted_wh_checkpoint as _checkpoint_codec
from .cuda_restricted_wh import (
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    verify_source_path_tile,
)
from .cuda_restricted_wh_episode_owner import (
    CUDARestrictedWHOwnedEpisodeResult,
    MAXIMUM_ROUTE_LOOKAHEAD_STEPS,
    OWNED_EPISODE_CLAIMS,
    OWNED_EPISODE_METHOD_ID,
    advance_cuda_restricted_wh_owned_episodes,
)
from .engine.backends import resolve_backend
from .engine.contracts import BackendSpec
from .restricted_wh_route import RestrictedWHRouteSpec


CHECKPOINT_SCHEMA = "jxplanetx.restricted-wh-owned-episode-delta-checkpoint.v1"
CHECKPOINT_METHOD_ID = "JX_RESTRICTED_WH_OWNED_EPISODE_DELTA_CHECKPOINT_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"

CHECKPOINT_DOMAIN = b"jxplanetx.restricted-wh.owned-episode-delta.record.v1\0"
CHECKPOINT_NAMESPACE_DOMAIN = (
    b"jxplanetx.restricted-wh.owned-episode-delta.namespace.v1\0"
)
STATE_DOMAIN = b"jxplanetx.restricted-wh.owned-episode-delta.state.v1\0"
LINEAGE_DOMAIN = b"jxplanetx.restricted-wh.owned-episode-delta.lineage.v1\0"
SEGMENT_DOMAIN = b"jxplanetx.restricted-wh.owned-episode-delta.segment.v1\0"
ROUTE_DELTA_DOMAIN = b"jxplanetx.restricted-wh.owned-episode.route-delta.v1\0"
ROUTE_CHAIN_DOMAIN = b"jxplanetx.restricted-wh.owned-episode.route-chain.v1\0"
APPEND_INVENTORY_DOMAIN = (
    b"jxplanetx.restricted-wh.owned-episode.append-inventory.v1\0"
)
MAXIMUM_CHECKPOINT_INDEX = 10**20 - 1

IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CHECKPOINT_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "four_point_five_gyr_accuracy_qualified": False,
        "cross_runtime_bitwise_reproducible": False,
        "whole_encounter_episode_qualified": False,
        "checkpoint_boundary_owner_persistence_qualified": False,
        "physical_collision_inferred": False,
        "physical_ejection_inferred": False,
        "completion_guaranteed": False,
        "science_result": False,
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECORD_PATTERN = re.compile(r"^checkpoint_([0-9]{20})\.json$")
_ARRAY_PATTERN = re.compile(
    r"^checkpoint_([0-9]{20})\.array-([0-9a-f]{64})\.npz$"
)
_TEMP_ARRAY_PATTERN = re.compile(
    r"^\.checkpoint_[0-9]{20}\.[A-Za-z0-9_-]+\.npz$"
)
_TEMP_RECORD_PATTERN = re.compile(
    r"^\.checkpoint_[0-9]{20}\.json\.[A-Za-z0-9_-]+\.tmp$"
)

_STATE_METADATA_KEYS = frozenset(
    {
        "snapshot_id",
        "epoch_origin_years_hex",
        "global_source_step",
        "source_step_years_hex",
        "time_scale",
        "frame",
        "axes",
        "coordinate_system",
        "length_unit",
        "time_unit",
        "unit_system_id",
        "source_ids",
        "primary_source_id",
        "source_schedule_sha256",
        "provenance",
    }
)
_ARRAY_KEYS = frozenset(
    {
        "input_tracer_relative_positions_AU",
        "input_tracer_relative_velocities_AU_per_year",
        "input_tracer_indices",
        "output_tracer_relative_positions_AU",
        "output_tracer_relative_velocities_AU_per_year",
        "output_tracer_indices",
        "source_relative_positions_AU",
        "source_relative_velocities_AU_per_year",
        "source_gm_AU3_per_year2",
        "source_encounter_radii_AU",
        "certified_relative_acceleration_bounds_AU_per_year2",
        "minimum_source_distance_AU",
    }
)
_RECORD_KEYS = frozenset(
    {
        "schema",
        "method_id",
        "run_id",
        "protocol_sha256",
        "genesis_state_sha256",
        "checkpoint_namespace",
        "checkpoint_index",
        "previous_checkpoint_content_sha256",
        "array_file",
        "array_file_size_bytes",
        "array_file_sha256",
        "input_state_metadata",
        "state_metadata",
        "source_tile_metadata",
        "input_state_sha256",
        "state_sha256",
        "lineage_sha256",
        "source_lineage_sha256",
        "source_initial_node_sha256",
        "source_final_node_sha256",
        "segment_sha256",
        "spec",
        "route_policy",
        "route_worker_count",
        "route_lookahead_steps",
        "route_delta_sha256",
        "previous_route_chain_sha256",
        "route_chain_sha256",
        "route_record_count_delta",
        "route_record_count_cumulative",
        "segment_episode_count_delta",
        "segment_episode_count_cumulative",
        "result",
        "implementation_identity",
        "checkpoint_runtime_identity",
        "claim_ceiling",
        "checkpoint_content_sha256",
    }
)


class OwnedEpisodeCheckpointError(RuntimeError):
    """Base error for replay-verified owned-episode delta checkpoints."""


class OwnedEpisodeCheckpointContractError(OwnedEpisodeCheckpointError, ValueError):
    """A checkpoint request violates the immutable contract."""


class OwnedEpisodeCheckpointIntegrityError(OwnedEpisodeCheckpointError):
    """Stored checkpoint evidence failed a hash, chain, or replay check."""


@dataclass(frozen=True, eq=False)
class LoadedOwnedEpisodeCheckpoint:
    """Latest state released after exact full-chain replay."""

    state: CUDARestrictedWHState
    checkpoint_index: int
    checkpoint_content_sha256: str
    input_state_sha256: str
    state_sha256: str
    route_chain_sha256: str
    route_record_count_cumulative: int
    segment_episode_count_cumulative: int
    run_id: str
    protocol_sha256: str
    record: Mapping[str, Any]
    claims: Mapping[str, bool] = CHECKPOINT_CLAIMS
    method_id: str = CHECKPOINT_METHOD_ID

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise OwnedEpisodeCheckpointContractError(
                "state must be an exact CUDARestrictedWHState"
            )
        if (
            type(self.checkpoint_index) is not int
            or not 0 <= self.checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
        ):
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint_index must be an integer in [0, 10**20)"
            )
        for value, label in (
            (self.checkpoint_content_sha256, "checkpoint_content_sha256"),
            (self.input_state_sha256, "input_state_sha256"),
            (self.state_sha256, "state_sha256"),
            (self.route_chain_sha256, "route_chain_sha256"),
            (self.protocol_sha256, "protocol_sha256"),
        ):
            _sha256(value, label)
        for value, label in (
            (self.route_record_count_cumulative, "route_record_count_cumulative"),
            (
                self.segment_episode_count_cumulative,
                "segment_episode_count_cumulative",
            ),
        ):
            if type(value) is not int or value < 0:
                raise OwnedEpisodeCheckpointContractError(
                    f"{label} must be a nonnegative integer"
                )
        _text(self.run_id, "run_id")
        if self.method_id != CHECKPOINT_METHOD_ID:
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint method identity changed"
            )
        if not _checkpoint_codec._canonical_tree_equal(
            dict(self.claims), dict(CHECKPOINT_CLAIMS)
        ) or any(self.claims.values()):
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(CHECKPOINT_CLAIMS)))
        object.__setattr__(self, "record", _checkpoint_codec._freeze_tree(self.record))


_APPEND_SESSION_SEAL = object()


@dataclass(frozen=True, eq=False)
class _VerifiedAppendHead:
    """Private authorization capability for one locked append frontier."""

    seal: object
    process_id: int
    thread_id: int
    run_id: str
    protocol_sha256: str
    genesis_state_sha256: str
    checkpoint_index: int
    checkpoint_content_sha256: str
    state_sha256: str
    route_chain_sha256: str
    route_record_count_cumulative: int
    segment_episode_count_cumulative: int
    artifact_inventory_sha256: str
    record: Mapping[str, Any]


def _append_artifact_inventory(
    records: tuple[tuple[int, Path, dict[str, Any]], ...],
) -> str:
    """Rehash every committed artifact without replaying old dynamics."""

    entries: list[dict[str, Any]] = []
    for index, path, record in records:
        arrays = _load_arrays(path, index, record)
        _validate_loaded_arrays(arrays)
        entries.append(
            {
                "checkpoint_index": index,
                "checkpoint_content_sha256": record[
                    "checkpoint_content_sha256"
                ],
                "array_file": record["array_file"],
                "array_file_size_bytes": record["array_file_size_bytes"],
                "array_file_sha256": record["array_file_sha256"],
            }
        )
    return hashlib.sha256(
        APPEND_INVENTORY_DOMAIN + _checkpoint_codec._canonical_json(entries)
    ).hexdigest()


def _verified_append_head(
    loaded: LoadedOwnedEpisodeCheckpoint,
    records: tuple[tuple[int, Path, dict[str, Any]], ...],
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    artifact_inventory_sha256: str | None = None,
) -> _VerifiedAppendHead:
    if not records:
        raise OwnedEpisodeCheckpointIntegrityError(
            "cannot authorize an append head for an empty chain"
        )
    latest = records[-1][2]
    if (
        type(loaded) is not LoadedOwnedEpisodeCheckpoint
        or loaded.run_id != run_id
        or loaded.protocol_sha256 != protocol_sha256
        or loaded.checkpoint_index != latest["checkpoint_index"]
        or loaded.checkpoint_content_sha256
        != latest["checkpoint_content_sha256"]
        or loaded.state_sha256 != latest["state_sha256"]
        or loaded.route_chain_sha256 != latest["route_chain_sha256"]
        or loaded.route_record_count_cumulative
        != latest["route_record_count_cumulative"]
        or loaded.segment_episode_count_cumulative
        != latest["segment_episode_count_cumulative"]
        or not _checkpoint_codec._canonical_tree_equal(
            _checkpoint_codec._plain_tree(loaded.record), latest
        )
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "replayed append head differs from committed evidence"
        )
    inventory = (
        _append_artifact_inventory(records)
        if artifact_inventory_sha256 is None
        else _sha256(
            artifact_inventory_sha256, "artifact_inventory_sha256"
        )
    )
    return _VerifiedAppendHead(
        seal=_APPEND_SESSION_SEAL,
        process_id=os.getpid(),
        thread_id=threading.get_ident(),
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        checkpoint_index=latest["checkpoint_index"],
        checkpoint_content_sha256=latest["checkpoint_content_sha256"],
        state_sha256=latest["state_sha256"],
        route_chain_sha256=latest["route_chain_sha256"],
        route_record_count_cumulative=latest[
            "route_record_count_cumulative"
        ],
        segment_episode_count_cumulative=latest[
            "segment_episode_count_cumulative"
        ],
        artifact_inventory_sha256=inventory,
        record=_checkpoint_codec._freeze_tree(latest),
    )


def _validate_verified_append_head(
    head: _VerifiedAppendHead,
    records: tuple[tuple[int, Path, dict[str, Any]], ...],
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
) -> None:
    if (
        type(head) is not _VerifiedAppendHead
        or head.seal is not _APPEND_SESSION_SEAL
        or head.process_id != os.getpid()
        or head.thread_id != threading.get_ident()
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "append authorization belongs to another process or thread"
        )
    if not records:
        raise OwnedEpisodeCheckpointIntegrityError(
            "verified append head requires a nonempty chain"
        )
    latest = records[-1][2]
    inventory = _append_artifact_inventory(records)
    if (
        head.run_id != run_id
        or head.protocol_sha256 != protocol_sha256
        or head.genesis_state_sha256 != genesis_state_sha256
        or head.checkpoint_index != latest["checkpoint_index"]
        or head.checkpoint_content_sha256
        != latest["checkpoint_content_sha256"]
        or head.state_sha256 != latest["state_sha256"]
        or head.route_chain_sha256 != latest["route_chain_sha256"]
        or head.route_record_count_cumulative
        != latest["route_record_count_cumulative"]
        or head.segment_episode_count_cumulative
        != latest["segment_episode_count_cumulative"]
        or head.artifact_inventory_sha256 != inventory
        or not _checkpoint_codec._canonical_tree_equal(
            _checkpoint_codec._plain_tree(head.record), latest
        )
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "verified append head or committed artifact inventory changed"
        )


def _read_published_append_head(
    root: Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    backend: BackendSpec,
    expected_record: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> tuple[LoadedOwnedEpisodeCheckpoint, _VerifiedAppendHead]:
    """Read back new bytes and derive the next capability from those bytes."""

    records = _scan_records(root, run_id, protocol_sha256)
    if not records:
        raise OwnedEpisodeCheckpointIntegrityError(
            "published append did not produce a checkpoint record"
        )
    index, path, record = records[-1]
    if not _checkpoint_codec._canonical_tree_equal(
        record, _checkpoint_codec._plain_tree(expected_record)
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "published checkpoint record differs from verified replay output"
        )
    _verify_record_static(
        record,
        implementation_identity,
        runtime_identity,
        genesis_state_sha256,
    )
    inventory = _append_artifact_inventory(records)
    arrays = _load_arrays(path, index, record)
    _validate_loaded_arrays(arrays)
    input_metadata = record.get("input_state_metadata")
    output_metadata = record.get("state_metadata")
    if type(input_metadata) is not dict or type(output_metadata) is not dict:
        raise OwnedEpisodeCheckpointIntegrityError(
            "published checkpoint state metadata is invalid"
        )
    input_hash = _state_sha256(input_metadata, arrays, prefix="input")
    output_hash = _state_sha256(output_metadata, arrays, prefix="output")
    if (
        input_hash != record.get("input_state_sha256")
        or output_hash != record.get("state_sha256")
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "published checkpoint state hash failed read-back"
        )
    state = _state_from_storage(
        output_metadata, arrays, prefix="output", backend=backend
    )
    loaded = LoadedOwnedEpisodeCheckpoint(
        state=state,
        checkpoint_index=index,
        checkpoint_content_sha256=record["checkpoint_content_sha256"],
        input_state_sha256=record["input_state_sha256"],
        state_sha256=record["state_sha256"],
        route_chain_sha256=record["route_chain_sha256"],
        route_record_count_cumulative=record[
            "route_record_count_cumulative"
        ],
        segment_episode_count_cumulative=record[
            "segment_episode_count_cumulative"
        ],
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        record=record,
    )
    head = _verified_append_head(
        loaded,
        records,
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        artifact_inventory_sha256=inventory,
    )
    return loaded, head


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise OwnedEpisodeCheckpointContractError(
            f"{label} must be a nonempty trimmed string"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise OwnedEpisodeCheckpointContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _from_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise OwnedEpisodeCheckpointIntegrityError(
            f"{label} must be a hexadecimal binary64 string"
        )
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise OwnedEpisodeCheckpointIntegrityError(
            f"{label} must be a hexadecimal binary64 string"
        ) from exc
    if not math.isfinite(result) or result.hex() != value:
        raise OwnedEpisodeCheckpointIntegrityError(
            f"{label} is not canonical finite binary64"
        )
    return result


def _state_metadata(state: CUDARestrictedWHState) -> dict[str, Any]:
    if type(state) is not CUDARestrictedWHState:
        raise OwnedEpisodeCheckpointContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    return {
        "snapshot_id": state.snapshot_id,
        "epoch_origin_years_hex": float(state.epoch_origin_years).hex(),
        "global_source_step": state.global_source_step,
        "source_step_years_hex": float(state.source_step_years).hex(),
        "time_scale": state.time_scale,
        "frame": state.frame,
        "axes": state.axes,
        "coordinate_system": state.coordinate_system,
        "length_unit": state.length_unit,
        "time_unit": state.time_unit,
        "unit_system_id": state.unit_system_id,
        "source_ids": list(state.source_ids),
        "primary_source_id": state.primary_source_id,
        "source_schedule_sha256": state.source_schedule_sha256,
        "provenance": _checkpoint_codec._provenance_record(state.provenance),
    }


def _state_arrays(
    state: CUDARestrictedWHState, cp: Any, *, prefix: str
) -> dict[str, np.ndarray]:
    if prefix not in {"input", "output", "state"}:
        raise OwnedEpisodeCheckpointContractError("state array prefix is invalid")
    arrays = {
        f"{prefix}_tracer_relative_positions_AU": np.ascontiguousarray(
            cp.asnumpy(state.tracer_relative_positions_AU)
        ),
        f"{prefix}_tracer_relative_velocities_AU_per_year": np.ascontiguousarray(
            cp.asnumpy(state.tracer_relative_velocities_AU_per_year)
        ),
        f"{prefix}_tracer_indices": np.ascontiguousarray(
            cp.asnumpy(state.tracer_indices)
        ),
    }
    positions = arrays[f"{prefix}_tracer_relative_positions_AU"]
    velocities = arrays[f"{prefix}_tracer_relative_velocities_AU_per_year"]
    indices = arrays[f"{prefix}_tracer_indices"]
    if (
        positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or indices.dtype != np.dtype(np.uint64)
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or velocities.shape != positions.shape
        or indices.shape != (positions.shape[0],)
        or not positions.flags.c_contiguous
        or not velocities.flags.c_contiguous
        or not indices.flags.c_contiguous
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(velocities)))
        or len(np.unique(indices)) != indices.size
    ):
        raise OwnedEpisodeCheckpointContractError(
            "state arrays are not exact finite float64/uint64 canonical arrays"
        )
    return arrays


def _state_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *, prefix: str
) -> str:
    digest = hashlib.sha256()
    digest.update(STATE_DOMAIN)
    digest.update(_checkpoint_codec._canonical_json(dict(metadata)))
    for suffix in (
        "tracer_relative_positions_AU",
        "tracer_relative_velocities_AU_per_year",
        "tracer_indices",
    ):
        _checkpoint_codec._array_digest_update(
            digest, suffix, arrays[f"{prefix}_{suffix}"]
        )
    return digest.hexdigest()


def _lineage_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *, prefix: str
) -> str:
    static = dict(metadata)
    static.pop("global_source_step", None)
    digest = hashlib.sha256()
    digest.update(LINEAGE_DOMAIN)
    digest.update(_checkpoint_codec._canonical_json(static))
    _checkpoint_codec._array_digest_update(
        digest, "tracer_indices", arrays[f"{prefix}_tracer_indices"]
    )
    return digest.hexdigest()


def _state_from_storage(
    metadata: object,
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
    backend: BackendSpec,
) -> CUDARestrictedWHState:
    if type(metadata) is not dict or set(metadata) != _STATE_METADATA_KEYS:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint state metadata roster changed"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    try:
        provenance = _checkpoint_codec._provenance_from_record(
            metadata["provenance"], prefix
        )
        with resolved.activate():
            state = CUDARestrictedWHState(
                snapshot_id=metadata["snapshot_id"],
                epoch_origin_years=_from_hex(
                    metadata["epoch_origin_years_hex"],
                    f"{prefix}.epoch_origin_years",
                ),
                global_source_step=metadata["global_source_step"],
                source_step_years=_from_hex(
                    metadata["source_step_years_hex"],
                    f"{prefix}.source_step_years",
                ),
                time_scale=metadata["time_scale"],
                frame=metadata["frame"],
                axes=metadata["axes"],
                coordinate_system=metadata["coordinate_system"],
                length_unit=metadata["length_unit"],
                time_unit=metadata["time_unit"],
                unit_system_id=metadata["unit_system_id"],
                source_ids=tuple(metadata["source_ids"]),
                primary_source_id=metadata["primary_source_id"],
                source_schedule_sha256=metadata["source_schedule_sha256"],
                tracer_relative_positions_AU=cp.asarray(
                    arrays[f"{prefix}_tracer_relative_positions_AU"],
                    dtype=cp.float64,
                    order="C",
                ),
                tracer_relative_velocities_AU_per_year=cp.asarray(
                    arrays[f"{prefix}_tracer_relative_velocities_AU_per_year"],
                    dtype=cp.float64,
                    order="C",
                ),
                tracer_indices=cp.asarray(
                    arrays[f"{prefix}_tracer_indices"],
                    dtype=cp.uint64,
                    order="C",
                ),
                provenance=provenance,
            )
            cp.cuda.get_current_stream().synchronize()
            return state
    except OwnedEpisodeCheckpointIntegrityError:
        raise
    except Exception as exc:
        raise OwnedEpisodeCheckpointIntegrityError(
            f"stored {prefix} state cannot be reconstructed"
        ) from exc


def _episode_summaries(
    result: CUDARestrictedWHOwnedEpisodeResult, segment_map_step_count: int
) -> list[dict[str, Any]]:
    worker_by_key = {
        (record.map_step_index, record.lane_index, record.tracer_index): worker
        for record, worker in zip(
            result.route_records, result.route_worker_indices, strict=True
        )
    }
    summaries: list[dict[str, Any]] = []
    for episode in result.episodes:
        first = episode.route_records[0]
        last = episode.route_records[-1]
        worker = worker_by_key[
            (first.map_step_index, first.lane_index, first.tracer_index)
        ]
        summaries.append(
            {
                "lane_index": episode.lane_index,
                "tracer_index": episode.tracer_index,
                "first_map_step_index": episode.first_map_step_index,
                "stop_map_step_index": episode.stop_map_step_index,
                "route_count": episode.route_count,
                "worker_index": worker,
                "first_global_source_step": (
                    first.route_result.global_source_step
                ),
                "stop_global_source_step": (
                    last.route_result.global_source_step
                    + last.route_result.source_steps_per_map_step
                ),
                "starts_at_segment_boundary": (
                    episode.first_map_step_index == 0
                ),
                "ends_at_segment_boundary": (
                    episode.stop_map_step_index == segment_map_step_count
                ),
            }
        )
    return summaries


def _result_semantics(
    result: CUDARestrictedWHOwnedEpisodeResult,
    cp: Any,
    *,
    segment_map_step_count: int,
) -> dict[str, Any]:
    if type(result) is not CUDARestrictedWHOwnedEpisodeResult:
        raise OwnedEpisodeCheckpointContractError(
            "result must be an exact CUDARestrictedWHOwnedEpisodeResult"
        )
    minimum = np.ascontiguousarray(cp.asnumpy(result.minimum_source_distance_AU))
    if (
        minimum.dtype != np.dtype(np.float64)
        or minimum.ndim != 1
        or not minimum.flags.c_contiguous
        or not bool(np.all(np.isfinite(minimum)))
        or not bool(np.all(minimum > 0.0))
    ):
        raise OwnedEpisodeCheckpointContractError(
            "minimum-source-distance diagnostics must be finite positive float64"
        )
    minimum_digest = hashlib.sha256()
    _checkpoint_codec._array_digest_update(
        minimum_digest, "minimum_source_distance_AU", minimum
    )
    return {
        "method_id": result.method_id,
        "scientific_classification": result.scientific_classification,
        "project_scope": result.project_scope,
        "claims": dict(result.claims),
        "source_tile_sha256": result.source_tile_sha256,
        "maximum_kepler_iterations": result.maximum_kepler_iterations,
        "minimum_source_distance_array_sha256": minimum_digest.hexdigest(),
        "route_records": [
            _checkpoint_codec._route_record(row) for row in result.route_records
        ],
        "route_worker_indices": list(result.route_worker_indices),
        "segment_episode_summaries": _episode_summaries(
            result, segment_map_step_count
        ),
        "runtime_identity": _checkpoint_codec._evidence_tree(
            result.runtime_identity
        ),
    }


def _route_delta_sha256(result_record: Mapping[str, Any]) -> str:
    payload = {
        "route_records": result_record.get("route_records"),
        "route_worker_indices": result_record.get("route_worker_indices"),
        "segment_episode_summaries": result_record.get(
            "segment_episode_summaries"
        ),
    }
    return hashlib.sha256(
        ROUTE_DELTA_DOMAIN + _checkpoint_codec._canonical_json(payload)
    ).hexdigest()


def _route_chain_sha256(
    *,
    previous_route_chain_sha256: str | None,
    checkpoint_index: int,
    input_global_source_step: int,
    output_global_source_step: int,
    route_delta_sha256: str,
) -> str:
    payload = {
        "previous_route_chain_sha256": previous_route_chain_sha256,
        "checkpoint_index": checkpoint_index,
        "input_global_source_step": input_global_source_step,
        "output_global_source_step": output_global_source_step,
        "route_delta_sha256": route_delta_sha256,
    }
    return hashlib.sha256(
        ROUTE_CHAIN_DOMAIN + _checkpoint_codec._canonical_json(payload)
    ).hexdigest()


def _segment_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        SEGMENT_DOMAIN + _checkpoint_codec._canonical_json(dict(record))
    ).hexdigest()


def owned_episode_checkpoint_implementation_identity() -> Mapping[str, Any]:
    """Return every local code/kernel identity used by checkpoint replay."""

    return _checkpoint_codec._freeze_tree(
        {
            "checkpoint_method_id": CHECKPOINT_METHOD_ID,
            "checkpoint_implementation_source_sha256": (
                IMPLEMENTATION_SOURCE_SHA256
            ),
            "checkpoint_codec_implementation_source_sha256": (
                _checkpoint_codec.IMPLEMENTATION_SOURCE_SHA256
            ),
            "checkpoint_codec_dependency_identity": dict(
                _checkpoint_codec._plain_tree(
                    _checkpoint_codec.checkpoint_implementation_identity()
                )
            ),
            "owned_episode_method_id": OWNED_EPISODE_METHOD_ID,
            "owned_episode_implementation_source_sha256": (
                _owner_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "episode_method_id": _episode_module.EPISODE_METHOD_ID,
            "episode_implementation_source_sha256": (
                _episode_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "restricted_wh_method_id": _restricted_module.METHOD_ID,
            "restricted_wh_implementation_source_sha256": (
                _restricted_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "restricted_wh_fused_cuda_source_sha256": (
                _restricted_module.FUSED_CUDA_SOURCE_SHA256
            ),
        }
    )


def owned_episode_checkpoint_runtime_identity(
    backend: BackendSpec,
) -> Mapping[str, Any]:
    """Return the exact host/CUDA runtime required for deterministic replay."""

    base = dict(
        _checkpoint_codec._plain_tree(
            _checkpoint_codec.checkpoint_runtime_identity(backend)
        )
    )
    base["checkpoint_method_id"] = CHECKPOINT_METHOD_ID
    return _checkpoint_codec._freeze_tree(base)


def _current_checkpoint_identities(
    backend: BackendSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = dict(
        _checkpoint_codec._plain_tree(
            owned_episode_checkpoint_implementation_identity()
        )
    )
    runtime = dict(
        _checkpoint_codec._plain_tree(
            owned_episode_checkpoint_runtime_identity(backend)
        )
    )
    return implementation, runtime


def owned_episode_state_sha256(
    state: CUDARestrictedWHState, backend: BackendSpec
) -> str:
    """Hash an exact state for the protocol genesis commitment."""

    if type(state) is not CUDARestrictedWHState:
        raise OwnedEpisodeCheckpointContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise OwnedEpisodeCheckpointContractError(
            "state hashing requires an exact CuPy BackendSpec"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        arrays = _state_arrays(state, cp, prefix="state")
        cp.cuda.get_current_stream().synchronize()
    return _state_sha256(_state_metadata(state), arrays, prefix="state")


def _checkpoint_namespace(
    directory: Path, run_id: str, protocol_sha256: str
) -> Path:
    digest = hashlib.sha256(
        CHECKPOINT_NAMESPACE_DOMAIN
        + run_id.encode("utf-8")
        + b"\0"
        + protocol_sha256.encode("ascii")
    ).hexdigest()
    return directory / f"owned-episode-delta-{digest}"


def _record_path(namespace: Path, index: int) -> Path:
    return namespace / f"checkpoint_{index:020d}.json"


def _load_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint record is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OwnedEpisodeCheckpointIntegrityError(
            f"checkpoint record is unreadable: {path}"
        ) from exc
    if type(value) is not dict:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint record must be a JSON object"
        )
    recorded = value.pop("checkpoint_content_sha256", None)
    expected = hashlib.sha256(
        CHECKPOINT_DOMAIN + _checkpoint_codec._canonical_json(value)
    ).hexdigest()
    if recorded != expected:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint record content hash failed"
        )
    value["checkpoint_content_sha256"] = recorded
    if set(value) != _RECORD_KEYS:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint record field roster changed"
        )
    return value


def _scan_records(
    directory: Path, run_id: str, protocol_sha256: str
) -> tuple[tuple[int, Path, dict[str, Any]], ...]:
    namespace = _checkpoint_namespace(directory, run_id, protocol_sha256)
    if not namespace.exists():
        return ()
    if namespace.is_symlink() or not namespace.is_dir():
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint namespace exists but is not a real directory"
        )
    records: list[tuple[int, Path, dict[str, Any]]] = []
    for path in namespace.iterdir():
        if path.name.startswith("."):
            allowed_temporary = (
                _TEMP_ARRAY_PATTERN.fullmatch(path.name) is not None
                or _TEMP_RECORD_PATTERN.fullmatch(path.name) is not None
            )
            if path.name == ".append.lock":
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != 0
                ):
                    raise OwnedEpisodeCheckpointIntegrityError(
                        "checkpoint append lock artifact is invalid"
                    )
            elif not allowed_temporary or path.is_symlink() or not path.is_file():
                raise OwnedEpisodeCheckpointIntegrityError(
                    f"unexpected hidden checkpoint artifact: {path.name}"
                )
            continue
        record_match = _RECORD_PATTERN.fullmatch(path.name)
        array_match = _ARRAY_PATTERN.fullmatch(path.name)
        if record_match is None and array_match is None:
            raise OwnedEpisodeCheckpointIntegrityError(
                f"unexpected checkpoint namespace artifact: {path.name}"
            )
        if record_match is None:
            if path.is_symlink() or not path.is_file():
                raise OwnedEpisodeCheckpointIntegrityError(
                    "checkpoint array artifact is not a regular file"
                )
            continue
        index = int(record_match.group(1))
        record = _load_record(path)
        expected_identity = {
            "schema": CHECKPOINT_SCHEMA,
            "method_id": CHECKPOINT_METHOD_ID,
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "checkpoint_namespace": namespace.name,
            "checkpoint_index": index,
        }
        observed_identity = {key: record.get(key) for key in expected_identity}
        if (
            not _checkpoint_codec._canonical_tree_equal(
                observed_identity, expected_identity
            )
            or _record_path(namespace, index) != path
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint identity or canonical path failed"
            )
        records.append((index, path, record))
    records.sort(key=lambda item: item[0])
    predecessor: str | None = None
    for expected_index, (index, _path, record) in enumerate(records):
        if index != expected_index:
            raise OwnedEpisodeCheckpointIntegrityError(
                "delta checkpoint indices must be contiguous from zero"
            )
        if record.get("previous_checkpoint_content_sha256") != predecessor:
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint predecessor hash chain failed"
            )
        predecessor = record["checkpoint_content_sha256"]
    return tuple(records)


def _load_arrays(
    record_path: Path, index: int, record: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    digest = record.get("array_file_sha256")
    name = record.get("array_file")
    expected = f"checkpoint_{index:020d}.array-{digest}.npz"
    if (
        type(digest) is not str
        or _SHA256_PATTERN.fullmatch(digest) is None
        or type(name) is not str
        or name != expected
        or Path(name).name != name
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint array path identity failed"
        )
    path = record_path.parent / name
    if path.is_symlink() or not path.is_file():
        raise OwnedEpisodeCheckpointIntegrityError(
            "committed checkpoint is missing its regular array file"
        )
    size = record.get("array_file_size_bytes")
    if type(size) is not int or size <= 0 or path.stat().st_size != size:
        raise OwnedEpisodeCheckpointIntegrityError("checkpoint array size failed")
    if _checkpoint_codec._sha256_file(path) != digest:
        raise OwnedEpisodeCheckpointIntegrityError("checkpoint array SHA-256 failed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(_ARRAY_KEYS)
                or set(archive.files) != _ARRAY_KEYS
            ):
                raise OwnedEpisodeCheckpointIntegrityError(
                    "checkpoint array member roster changed"
                )
            arrays = {
                key: np.ascontiguousarray(np.array(archive[key], copy=True))
                for key in _ARRAY_KEYS
            }
    except OwnedEpisodeCheckpointIntegrityError:
        raise
    except (OSError, ValueError) as exc:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint array archive is unreadable"
        ) from exc
    return arrays


def _validate_loaded_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for prefix in ("input", "output"):
        positions = arrays[f"{prefix}_tracer_relative_positions_AU"]
        velocities = arrays[f"{prefix}_tracer_relative_velocities_AU_per_year"]
        indices = arrays[f"{prefix}_tracer_indices"]
        if (
            positions.dtype != np.dtype(np.float64)
            or velocities.dtype != np.dtype(np.float64)
            or indices.dtype != np.dtype(np.uint64)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or velocities.shape != positions.shape
            or indices.shape != (positions.shape[0],)
            or not positions.flags.c_contiguous
            or not velocities.flags.c_contiguous
            or not indices.flags.c_contiguous
            or not bool(np.all(np.isfinite(positions)))
            or not bool(np.all(np.isfinite(velocities)))
            or len(np.unique(indices)) != indices.size
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                f"stored {prefix} state arrays are noncanonical"
            )
    if not np.array_equal(
        arrays["input_tracer_indices"], arrays["output_tracer_indices"]
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint output changed tracer identity or order"
        )
    minimum = arrays["minimum_source_distance_AU"]
    lane_count = arrays["output_tracer_indices"].size
    if (
        minimum.dtype != np.dtype(np.float64)
        or minimum.shape != (lane_count,)
        or not minimum.flags.c_contiguous
        or not bool(np.all(np.isfinite(minimum)))
        or not bool(np.all(minimum > 0.0))
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint minimum-distance array is noncanonical"
        )


def _validate_segment_shape(
    input_state: CUDARestrictedWHState,
    result: CUDARestrictedWHOwnedEpisodeResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    route_worker_count: int,
    route_lookahead_steps: int,
) -> None:
    if type(input_state) is not CUDARestrictedWHState:
        raise OwnedEpisodeCheckpointContractError(
            "input_state must be an exact CUDARestrictedWHState"
        )
    if type(result) is not CUDARestrictedWHOwnedEpisodeResult:
        raise OwnedEpisodeCheckpointContractError(
            "result must be an exact CUDARestrictedWHOwnedEpisodeResult"
        )
    if type(source_tile) is not FrozenSourcePathTile:
        raise OwnedEpisodeCheckpointContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    verify_source_path_tile(source_tile)
    if type(spec) is not FixedStepCUDARestrictedWHSpec:
        raise OwnedEpisodeCheckpointContractError(
            "spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    if type(route_spec) is not RestrictedWHRouteSpec:
        raise OwnedEpisodeCheckpointContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    if type(route_worker_count) is not int or not 1 <= route_worker_count <= 32:
        raise OwnedEpisodeCheckpointContractError(
            "route_worker_count must be an integer in [1, 32]"
        )
    if (
        type(route_lookahead_steps) is not int
        or not 1 <= route_lookahead_steps <= MAXIMUM_ROUTE_LOOKAHEAD_STEPS
    ):
        raise OwnedEpisodeCheckpointContractError(
            "route_lookahead_steps must be an integer in [1, 64]"
        )
    intervals = spec.map_step_count * spec.source_steps_per_map_step
    if source_tile.node_count != intervals + 1:
        raise OwnedEpisodeCheckpointContractError(
            "a delta checkpoint source tile must contain exactly one segment"
        )
    if source_tile.global_source_step != input_state.global_source_step:
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint source tile does not start at its input state"
        )
    if result.state.global_source_step != input_state.global_source_step + intervals:
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint output is not at the exact segment lattice endpoint"
        )
    if result.source_tile_sha256 != source_tile.content_sha256:
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint result names a different source tile"
        )
    if result.method_id != OWNED_EPISODE_METHOD_ID or not (
        _checkpoint_codec._canonical_tree_equal(
            dict(result.claims), dict(OWNED_EPISODE_CLAIMS)
        )
        and not any(result.claims.values())
    ):
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint result owned-episode identity changed"
        )
    if (
        route_spec.maximum_primary_dynamical_phase
        != spec.maximum_primary_dynamical_phase
        or route_spec.primary_outer_boundary_AU != spec.primary_outer_boundary_AU
    ):
        raise OwnedEpisodeCheckpointContractError(
            "CUDA and route policies must match exactly"
        )
    if (
        result.runtime_identity.get("route_worker_count") != route_worker_count
        or result.runtime_identity.get("route_lookahead_steps")
        != route_lookahead_steps
    ):
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint result worker/lookahead policy changed"
        )


def _verify_result_against_replay(
    input_state: CUDARestrictedWHState,
    expected: CUDARestrictedWHOwnedEpisodeResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    route_worker_count: int,
    route_lookahead_steps: int,
) -> CUDARestrictedWHOwnedEpisodeResult:
    replay = advance_cuda_restricted_wh_owned_episodes(
        input_state,
        source_tile,
        spec,
        route_spec=route_spec,
        route_worker_count=route_worker_count,
        route_lookahead_steps=route_lookahead_steps,
    )
    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    with resolved.activate():
        expected_arrays = _state_arrays(expected.state, cp, prefix="output")
        replay_arrays = _state_arrays(replay.state, cp, prefix="output")
        expected_hash = _state_sha256(
            _state_metadata(expected.state), expected_arrays, prefix="output"
        )
        replay_hash = _state_sha256(
            _state_metadata(replay.state), replay_arrays, prefix="output"
        )
        if expected_hash != replay_hash:
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint result output differs from deterministic replay"
            )
        expected_semantics = _result_semantics(
            expected, cp, segment_map_step_count=spec.map_step_count
        )
        replay_semantics = _result_semantics(
            replay, cp, segment_map_step_count=spec.map_step_count
        )
        if not _checkpoint_codec._canonical_tree_equal(
            expected_semantics, replay_semantics
        ):
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint result diagnostics or route delta differ from replay"
            )
        cp.cuda.get_current_stream().synchronize()
    return replay


def _core_error(exc: Exception, *, integrity: bool) -> OwnedEpisodeCheckpointError:
    if integrity:
        return OwnedEpisodeCheckpointIntegrityError(str(exc))
    return OwnedEpisodeCheckpointContractError(str(exc))


def dump_owned_episode_delta_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    checkpoint_index: int,
    input_state: CUDARestrictedWHState,
    result: CUDARestrictedWHOwnedEpisodeResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = 6,
    route_lookahead_steps: int = 32,
    previous_checkpoint_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Replay, verify, and create one route-delta checkpoint record."""

    try:
        root = Path(directory)
        _text(run_id, "run_id")
        _sha256(protocol_sha256, "protocol_sha256")
        _sha256(genesis_state_sha256, "genesis_state_sha256")
        if (
            type(checkpoint_index) is not int
            or not 0 <= checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
        ):
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint_index must be an integer in [0, 10**20)"
            )
        if previous_checkpoint_content_sha256 is not None:
            _sha256(
                previous_checkpoint_content_sha256,
                "previous_checkpoint_content_sha256",
            )
        if type(spec) is not FixedStepCUDARestrictedWHSpec:
            raise OwnedEpisodeCheckpointContractError(
                "spec must be an exact FixedStepCUDARestrictedWHSpec"
            )
        if route_spec is None:
            route_spec = RestrictedWHRouteSpec(
                maximum_primary_dynamical_phase=(
                    spec.maximum_primary_dynamical_phase
                ),
                primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
            )
        _validate_segment_shape(
            input_state,
            result,
            source_tile,
            spec,
            route_spec,
            route_worker_count,
            route_lookahead_steps,
        )
        namespace = _checkpoint_namespace(root, run_id, protocol_sha256)
        _checkpoint_codec._ensure_namespace(root, namespace)
        with _checkpoint_codec._checkpoint_append_lock(namespace):
            return _dump_locked(
                root,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_state_sha256,
                checkpoint_index=checkpoint_index,
                input_state=input_state,
                result=result,
                source_tile=source_tile,
                spec=spec,
                route_spec=route_spec,
                route_worker_count=route_worker_count,
                route_lookahead_steps=route_lookahead_steps,
                previous_checkpoint_content_sha256=(
                    previous_checkpoint_content_sha256
                ),
            )
    except OwnedEpisodeCheckpointError:
        raise
    except _checkpoint_codec.RestrictedWHCheckpointIntegrityError as exc:
        raise _core_error(exc, integrity=True) from exc
    except _checkpoint_codec.RestrictedWHCheckpointError as exc:
        raise _core_error(exc, integrity=False) from exc


def _dump_locked(
    root: Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    checkpoint_index: int,
    input_state: CUDARestrictedWHState,
    result: CUDARestrictedWHOwnedEpisodeResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    route_worker_count: int,
    route_lookahead_steps: int,
    previous_checkpoint_content_sha256: str | None,
    verified_head: _VerifiedAppendHead | None = None,
    bound_implementation_identity: Mapping[str, Any] | None = None,
    bound_runtime_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if (bound_implementation_identity is None) != (
        bound_runtime_identity is None
    ):
        raise OwnedEpisodeCheckpointContractError(
            "append implementation and runtime identities must be bound together"
        )
    if bound_implementation_identity is None:
        implementation_identity, runtime_identity = (
            _current_checkpoint_identities(spec.backend)
        )
    else:
        implementation_identity = dict(
            _checkpoint_codec._plain_tree(bound_implementation_identity)
        )
        runtime_identity = dict(
            _checkpoint_codec._plain_tree(bound_runtime_identity)
        )
    records = _scan_records(root, run_id, protocol_sha256)
    if checkpoint_index != len(records):
        raise OwnedEpisodeCheckpointContractError(
            "delta checkpoint index must exactly extend the current chain"
        )
    latest = records[-1][2] if records else None
    expected_predecessor = (
        None if latest is None else latest["checkpoint_content_sha256"]
    )
    if previous_checkpoint_content_sha256 != expected_predecessor:
        error_type = (
            OwnedEpisodeCheckpointIntegrityError
            if verified_head is not None
            else OwnedEpisodeCheckpointContractError
        )
        raise error_type(
            "previous checkpoint digest does not name the current chain head"
        )
    # Appending is an authorization decision, not only a hash-link operation.
    # Replay the complete predecessor chain with the current code/runtime while
    # the namespace lock is held.  This prevents a rehashed predecessor or a
    # code/runtime transition from being sealed into an irreversible mixed
    # chain that no runtime can subsequently load.
    if latest is not None:
        if verified_head is None:
            verified_latest = load_latest_owned_episode_delta_checkpoint(
                root,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_state_sha256,
                expected_checkpoint_index=latest["checkpoint_index"],
                expected_checkpoint_content_sha256=latest[
                    "checkpoint_content_sha256"
                ],
                backend=spec.backend,
            )
            verified_head = _verified_append_head(
                verified_latest,
                records,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_state_sha256,
            )
        _validate_verified_append_head(
            verified_head,
            records,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            genesis_state_sha256=genesis_state_sha256,
        )
    elif verified_head is not None:
        raise OwnedEpisodeCheckpointIntegrityError(
            "verified append head was supplied for an empty chain"
        )

    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    input_metadata = _state_metadata(input_state)
    with resolved.activate():
        input_arrays = _state_arrays(input_state, cp, prefix="input")
        cp.cuda.get_current_stream().synchronize()
    verified_input_state = _state_from_storage(
        input_metadata,
        input_arrays,
        prefix="input",
        backend=spec.backend,
    )
    verified_result = _verify_result_against_replay(
        verified_input_state,
        result,
        source_tile,
        spec,
        route_spec,
        route_worker_count,
        route_lookahead_steps,
    )
    with resolved.activate():
        output_arrays = _state_arrays(
            verified_result.state, cp, prefix="output"
        )
        minimum = np.ascontiguousarray(
            cp.asnumpy(verified_result.minimum_source_distance_AU)
        )
        cp.cuda.get_current_stream().synchronize()
    arrays = {
        **input_arrays,
        **output_arrays,
        **_checkpoint_codec._tile_arrays(source_tile),
        "minimum_source_distance_AU": minimum,
    }
    _validate_loaded_arrays(arrays)
    output_metadata = _state_metadata(verified_result.state)
    input_hash = _state_sha256(input_metadata, arrays, prefix="input")
    output_hash = _state_sha256(output_metadata, arrays, prefix="output")
    lineage_hash = _lineage_sha256(input_metadata, arrays, prefix="input")
    if _lineage_sha256(output_metadata, arrays, prefix="output") != lineage_hash:
        raise OwnedEpisodeCheckpointContractError(
            "checkpoint output changed immutable state lineage"
        )
    source_lineage = _checkpoint_codec._source_lineage_sha256(source_tile)
    source_initial = _checkpoint_codec._source_boundary_sha256(source_tile, 0)
    source_final = _checkpoint_codec._source_boundary_sha256(
        source_tile, source_tile.node_count - 1
    )
    if latest is None:
        if input_hash != genesis_state_sha256:
            raise OwnedEpisodeCheckpointContractError(
                "first checkpoint input differs from protocol genesis"
            )
    else:
        assert verified_head is not None
        if latest.get("genesis_state_sha256") != genesis_state_sha256:
            raise OwnedEpisodeCheckpointContractError(
                "successor changed the protocol genesis commitment"
            )
        if (
            input_hash != latest.get("state_sha256")
            or input_hash != verified_head.state_sha256
        ):
            raise OwnedEpisodeCheckpointContractError(
                "successor input differs from predecessor output"
            )
        if lineage_hash != latest.get("lineage_sha256"):
            raise OwnedEpisodeCheckpointContractError(
                "successor changed immutable state lineage"
            )
        if source_lineage != latest.get("source_lineage_sha256"):
            raise OwnedEpisodeCheckpointContractError(
                "successor changed immutable source lineage"
            )
        if source_initial != latest.get("source_final_node_sha256"):
            raise OwnedEpisodeCheckpointContractError(
                "successor source tile is discontinuous"
            )

    result_record = _result_semantics(
        verified_result, cp, segment_map_step_count=spec.map_step_count
    )
    route_delta = _route_delta_sha256(result_record)
    previous_route_chain = (
        None if latest is None else latest["route_chain_sha256"]
    )
    route_chain = _route_chain_sha256(
        previous_route_chain_sha256=previous_route_chain,
        checkpoint_index=checkpoint_index,
        input_global_source_step=verified_input_state.global_source_step,
        output_global_source_step=verified_result.state.global_source_step,
        route_delta_sha256=route_delta,
    )
    route_count_delta = len(verified_result.route_records)
    episode_count_delta = len(verified_result.episodes)
    prior_route_count = (
        0 if latest is None else latest["route_record_count_cumulative"]
    )
    prior_episode_count = (
        0 if latest is None else latest["segment_episode_count_cumulative"]
    )
    spec_record = _checkpoint_codec._spec_record(spec)
    route_policy_record = _checkpoint_codec._route_spec_record(route_spec)
    segment_record = {
        "input_state_sha256": input_hash,
        "state_sha256": output_hash,
        "lineage_sha256": lineage_hash,
        "source_tile_sha256": source_tile.content_sha256,
        "spec": spec_record,
        "route_policy": route_policy_record,
        "route_worker_count": route_worker_count,
        "route_lookahead_steps": route_lookahead_steps,
        "route_delta_sha256": route_delta,
        "route_chain_sha256": route_chain,
        "result": result_record,
    }
    segment_sha = _segment_sha256(segment_record)

    record_path = _record_path(
        _checkpoint_namespace(root, run_id, protocol_sha256), checkpoint_index
    )
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint_index}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".checkpoint_{checkpoint_index:020d}.",
        suffix=".npz",
        dir=record_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_sha = _checkpoint_codec._sha256_file(temporary)
        array_path = record_path.parent / (
            f"checkpoint_{checkpoint_index:020d}.array-{array_sha}.npz"
        )
        if array_path.exists():
            if array_path.is_symlink() or not array_path.is_file():
                raise OwnedEpisodeCheckpointIntegrityError(
                    "content-addressed array path is not a regular file"
                )
            if _checkpoint_codec._sha256_file(array_path) != array_sha:
                raise OwnedEpisodeCheckpointIntegrityError(
                    "content-addressed array collision"
                )
            temporary.unlink()
        else:
            _checkpoint_codec._commit_create_only(temporary, array_path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    record: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "method_id": CHECKPOINT_METHOD_ID,
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "genesis_state_sha256": genesis_state_sha256,
        "checkpoint_namespace": record_path.parent.name,
        "checkpoint_index": checkpoint_index,
        "previous_checkpoint_content_sha256": (
            previous_checkpoint_content_sha256
        ),
        "array_file": array_path.name,
        "array_file_size_bytes": array_path.stat().st_size,
        "array_file_sha256": array_sha,
        "input_state_metadata": input_metadata,
        "state_metadata": output_metadata,
        "source_tile_metadata": _checkpoint_codec._tile_metadata(source_tile),
        "input_state_sha256": input_hash,
        "state_sha256": output_hash,
        "lineage_sha256": lineage_hash,
        "source_lineage_sha256": source_lineage,
        "source_initial_node_sha256": source_initial,
        "source_final_node_sha256": source_final,
        "segment_sha256": segment_sha,
        "spec": spec_record,
        "route_policy": route_policy_record,
        "route_worker_count": route_worker_count,
        "route_lookahead_steps": route_lookahead_steps,
        "route_delta_sha256": route_delta,
        "previous_route_chain_sha256": previous_route_chain,
        "route_chain_sha256": route_chain,
        "route_record_count_delta": route_count_delta,
        "route_record_count_cumulative": prior_route_count + route_count_delta,
        "segment_episode_count_delta": episode_count_delta,
        "segment_episode_count_cumulative": (
            prior_episode_count + episode_count_delta
        ),
        "result": result_record,
        "implementation_identity": implementation_identity,
        "checkpoint_runtime_identity": runtime_identity,
        "claim_ceiling": {
            "scientific_classification": SCIENTIFIC_CLASSIFICATION,
            "project_scope": PROJECT_SCOPE,
            "claims": dict(CHECKPOINT_CLAIMS),
        },
    }
    record["checkpoint_content_sha256"] = hashlib.sha256(
        CHECKPOINT_DOMAIN + _checkpoint_codec._canonical_json(record)
    ).hexdigest()
    current_implementation, current_runtime = _current_checkpoint_identities(
        spec.backend
    )
    if not _checkpoint_codec._canonical_tree_equal(
        current_implementation, implementation_identity
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "append implementation identity changed before publication"
        )
    if not _checkpoint_codec._canonical_tree_equal(
        current_runtime, runtime_identity
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "append host/CUDA runtime identity changed before publication"
        )
    _checkpoint_codec._atomic_json_create(record_path, record)
    return _checkpoint_codec._freeze_tree(record)


def _verify_record_static(
    record: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    genesis_state_sha256: str,
) -> None:
    if not _checkpoint_codec._canonical_tree_equal(
        record.get("implementation_identity"), dict(implementation_identity)
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint implementation/kernel identity changed"
        )
    expected_claims = {
        "scientific_classification": SCIENTIFIC_CLASSIFICATION,
        "project_scope": PROJECT_SCOPE,
        "claims": dict(CHECKPOINT_CLAIMS),
    }
    if not _checkpoint_codec._canonical_tree_equal(
        record.get("claim_ceiling"), expected_claims
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint claim ceiling changed"
        )
    if not _checkpoint_codec._canonical_tree_equal(
        record.get("checkpoint_runtime_identity"), dict(runtime_identity)
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint host/CUDA runtime identity changed"
        )
    if record.get("genesis_state_sha256") != genesis_state_sha256:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint protocol genesis commitment changed"
        )


def _verify_loaded_segment(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    backend: BackendSpec,
    genesis_state_sha256: str,
    predecessor_state_sha256: str | None,
    predecessor_lineage_sha256: str | None,
    predecessor_source_final_node_sha256: str | None,
    predecessor_source_lineage_sha256: str | None,
    predecessor_route_chain_sha256: str | None,
    predecessor_route_count: int,
    predecessor_episode_count: int,
) -> tuple[CUDARestrictedWHState, str, str, str, str, str, int, int]:
    _validate_loaded_arrays(arrays)
    input_metadata = record.get("input_state_metadata")
    output_metadata = record.get("state_metadata")
    result_record = record.get("result")
    if (
        type(input_metadata) is not dict
        or type(output_metadata) is not dict
        or type(result_record) is not dict
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint state/result metadata is invalid"
        )
    input_hash = _state_sha256(input_metadata, arrays, prefix="input")
    output_hash = _state_sha256(output_metadata, arrays, prefix="output")
    lineage = _lineage_sha256(input_metadata, arrays, prefix="input")
    if (
        input_hash != record.get("input_state_sha256")
        or output_hash != record.get("state_sha256")
        or lineage != record.get("lineage_sha256")
        or _lineage_sha256(output_metadata, arrays, prefix="output") != lineage
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint semantic state or lineage hash failed"
        )
    if predecessor_state_sha256 is None:
        if input_hash != genesis_state_sha256:
            raise OwnedEpisodeCheckpointIntegrityError(
                "first checkpoint input differs from required genesis state"
            )
    elif input_hash != predecessor_state_sha256:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint successor input is not predecessor output"
        )
    if (
        predecessor_lineage_sha256 is not None
        and lineage != predecessor_lineage_sha256
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint successor changed immutable state lineage"
        )
    input_state = _state_from_storage(
        input_metadata, arrays, prefix="input", backend=backend
    )
    stored_output = _state_from_storage(
        output_metadata, arrays, prefix="output", backend=backend
    )
    try:
        tile = _checkpoint_codec._tile_from_storage(
            record.get("source_tile_metadata"), arrays
        )
    except _checkpoint_codec.RestrictedWHCheckpointError as exc:
        raise OwnedEpisodeCheckpointIntegrityError(str(exc)) from exc
    source_lineage = _checkpoint_codec._source_lineage_sha256(tile)
    source_initial = _checkpoint_codec._source_boundary_sha256(tile, 0)
    source_final = _checkpoint_codec._source_boundary_sha256(
        tile, tile.node_count - 1
    )
    if (
        source_lineage != record.get("source_lineage_sha256")
        or source_initial != record.get("source_initial_node_sha256")
        or source_final != record.get("source_final_node_sha256")
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint source lineage or boundary digest failed"
        )
    if (
        predecessor_source_lineage_sha256 is not None
        and source_lineage != predecessor_source_lineage_sha256
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint successor changed immutable source lineage"
        )
    if (
        predecessor_source_final_node_sha256 is not None
        and source_initial != predecessor_source_final_node_sha256
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint successor source tile is discontinuous"
        )
    try:
        spec = _checkpoint_codec._spec_from_record(record.get("spec"), backend)
        route_spec = _checkpoint_codec._route_spec_from_record(
            record.get("route_policy")
        )
    except _checkpoint_codec.RestrictedWHCheckpointError as exc:
        raise OwnedEpisodeCheckpointIntegrityError(str(exc)) from exc
    worker_count = record.get("route_worker_count")
    lookahead = record.get("route_lookahead_steps")
    if type(worker_count) is not int or not 1 <= worker_count <= 32:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint route worker count is invalid"
        )
    if (
        type(lookahead) is not int
        or not 1 <= lookahead <= MAXIMUM_ROUTE_LOOKAHEAD_STEPS
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint route lookahead is invalid"
        )
    intervals = spec.map_step_count * spec.source_steps_per_map_step
    if tile.node_count != intervals + 1:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint tile contains extra or missing segment nodes"
        )
    replay = advance_cuda_restricted_wh_owned_episodes(
        input_state,
        tile,
        spec,
        route_spec=route_spec,
        route_worker_count=worker_count,
        route_lookahead_steps=lookahead,
    )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        replay_arrays = _state_arrays(replay.state, cp, prefix="output")
        replay_hash = _state_sha256(
            _state_metadata(replay.state), replay_arrays, prefix="output"
        )
        stored_arrays = _state_arrays(stored_output, cp, prefix="output")
        stored_hash = _state_sha256(
            _state_metadata(stored_output), stored_arrays, prefix="output"
        )
        replay_record = _result_semantics(
            replay, cp, segment_map_step_count=spec.map_step_count
        )
        if replay_hash != output_hash or stored_hash != output_hash:
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint endpoint differs from deterministic replay"
            )
        if not _checkpoint_codec._canonical_tree_equal(
            replay_record, result_record
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint route delta differs from deterministic replay"
            )
        if not np.array_equal(
            cp.asnumpy(replay.minimum_source_distance_AU),
            arrays["minimum_source_distance_AU"],
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint minimum-distance array differs from replay"
            )
        cp.cuda.get_current_stream().synchronize()
    minimum_digest = hashlib.sha256()
    _checkpoint_codec._array_digest_update(
        minimum_digest,
        "minimum_source_distance_AU",
        arrays["minimum_source_distance_AU"],
    )
    if minimum_digest.hexdigest() != result_record.get(
        "minimum_source_distance_array_sha256"
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint minimum-distance semantic hash failed"
        )
    route_delta = _route_delta_sha256(result_record)
    if route_delta != record.get("route_delta_sha256"):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint route-delta semantic hash failed"
        )
    if record.get("previous_route_chain_sha256") != predecessor_route_chain_sha256:
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint previous route-chain digest failed"
        )
    route_chain = _route_chain_sha256(
        previous_route_chain_sha256=predecessor_route_chain_sha256,
        checkpoint_index=record["checkpoint_index"],
        input_global_source_step=input_state.global_source_step,
        output_global_source_step=replay.state.global_source_step,
        route_delta_sha256=route_delta,
    )
    if route_chain != record.get("route_chain_sha256"):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint route-chain digest failed"
        )
    route_count_delta = len(result_record.get("route_records", ()))
    episode_count_delta = len(result_record.get("segment_episode_summaries", ()))
    for label in (
        "route_record_count_delta",
        "route_record_count_cumulative",
        "segment_episode_count_delta",
        "segment_episode_count_cumulative",
    ):
        value = record.get(label)
        if type(value) is not int or value < 0:
            raise OwnedEpisodeCheckpointIntegrityError(
                f"checkpoint {label} is not a nonnegative integer"
            )
    if (
        record.get("route_record_count_delta") != route_count_delta
        or record.get("route_record_count_cumulative")
        != predecessor_route_count + route_count_delta
        or record.get("segment_episode_count_delta") != episode_count_delta
        or record.get("segment_episode_count_cumulative")
        != predecessor_episode_count + episode_count_delta
    ):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint delta or cumulative route accounting failed"
        )
    segment_record = {
        "input_state_sha256": input_hash,
        "state_sha256": output_hash,
        "lineage_sha256": lineage,
        "source_tile_sha256": tile.content_sha256,
        "spec": record.get("spec"),
        "route_policy": record.get("route_policy"),
        "route_worker_count": worker_count,
        "route_lookahead_steps": lookahead,
        "route_delta_sha256": route_delta,
        "route_chain_sha256": route_chain,
        "result": result_record,
    }
    if _segment_sha256(segment_record) != record.get("segment_sha256"):
        raise OwnedEpisodeCheckpointIntegrityError(
            "checkpoint segment semantic hash failed"
        )
    return (
        replay.state,
        output_hash,
        lineage,
        source_final,
        source_lineage,
        route_chain,
        predecessor_route_count + route_count_delta,
        predecessor_episode_count + episode_count_delta,
    )


def load_latest_owned_episode_delta_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    expected_checkpoint_index: int,
    expected_checkpoint_content_sha256: str,
    backend: BackendSpec,
) -> LoadedOwnedEpisodeCheckpoint:
    """Verify and replay the complete delta chain, then release its head."""

    try:
        root = Path(directory)
        _text(run_id, "run_id")
        _sha256(protocol_sha256, "protocol_sha256")
        _sha256(genesis_state_sha256, "genesis_state_sha256")
        if (
            type(expected_checkpoint_index) is not int
            or not 0 <= expected_checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
        ):
            raise OwnedEpisodeCheckpointContractError(
                "expected_checkpoint_index must be an integer in [0, 10**20)"
            )
        _sha256(
            expected_checkpoint_content_sha256,
            "expected_checkpoint_content_sha256",
        )
        if type(backend) is not BackendSpec or backend.backend_id != "cupy":
            raise OwnedEpisodeCheckpointContractError(
                "checkpoint load requires an exact CuPy BackendSpec"
            )
        if not root.exists():
            raise OwnedEpisodeCheckpointIntegrityError(
                "expected checkpoint root is missing"
            )
        _checkpoint_codec._validate_root(root, create=False)
        records = _scan_records(root, run_id, protocol_sha256)
        if not records:
            raise OwnedEpisodeCheckpointIntegrityError(
                "expected checkpoint chain is missing"
            )
        if (
            records[-1][0] != expected_checkpoint_index
            or records[-1][2].get("checkpoint_content_sha256")
            != expected_checkpoint_content_sha256
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint chain head differs from required external anchor"
            )
        implementation = dict(
            _checkpoint_codec._plain_tree(
                owned_episode_checkpoint_implementation_identity()
            )
        )
        runtime = dict(
            _checkpoint_codec._plain_tree(
                owned_episode_checkpoint_runtime_identity(backend)
            )
        )
        predecessor_state: str | None = None
        predecessor_lineage: str | None = None
        predecessor_source_final: str | None = None
        predecessor_source_lineage: str | None = None
        predecessor_route_chain: str | None = None
        route_count = 0
        episode_count = 0
        state: CUDARestrictedWHState | None = None
        for index, path, record in records:
            _verify_record_static(record, implementation, runtime, genesis_state_sha256)
            arrays = _load_arrays(path, index, record)
            (
                state,
                predecessor_state,
                predecessor_lineage,
                predecessor_source_final,
                predecessor_source_lineage,
                predecessor_route_chain,
                route_count,
                episode_count,
            ) = _verify_loaded_segment(
                record,
                arrays,
                backend,
                genesis_state_sha256,
                predecessor_state,
                predecessor_lineage,
                predecessor_source_final,
                predecessor_source_lineage,
                predecessor_route_chain,
                route_count,
                episode_count,
            )
        if (
            state is None
            or predecessor_state is None
            or predecessor_route_chain is None
        ):
            raise OwnedEpisodeCheckpointIntegrityError(
                "checkpoint history did not yield a state"
            )
        index, _path, record = records[-1]
        resolved = resolve_backend(backend)
        with resolved.activate():
            resolved.xp.cuda.get_current_stream().synchronize()
        return LoadedOwnedEpisodeCheckpoint(
            state=state,
            checkpoint_index=index,
            checkpoint_content_sha256=record["checkpoint_content_sha256"],
            input_state_sha256=record["input_state_sha256"],
            state_sha256=record["state_sha256"],
            route_chain_sha256=predecessor_route_chain,
            route_record_count_cumulative=route_count,
            segment_episode_count_cumulative=episode_count,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            record=record,
        )
    except OwnedEpisodeCheckpointError:
        raise
    except _checkpoint_codec.RestrictedWHCheckpointIntegrityError as exc:
        raise _core_error(exc, integrity=True) from exc
    except _checkpoint_codec.RestrictedWHCheckpointError as exc:
        raise _core_error(exc, integrity=False) from exc


class OwnedEpisodeDeltaAppendSession:
    """One locked writer that replay-verifies its predecessor chain once.

    The session holds the namespace append lock for its complete lifetime.
    Opening an existing chain performs the ordinary full replay.  Each append
    then verifies only the new deterministic segment, compares the exact
    locked head with the session's verified head, and advances that in-memory
    verification frontier.  Closing and reopening always requires another
    full replay from genesis.
    """

    def __init__(
        self,
        *,
        seal: object,
        root: Path,
        run_id: str,
        protocol_sha256: str,
        genesis_state_sha256: str,
        backend: BackendSpec,
        lock_context: Any,
        loaded_head: LoadedOwnedEpisodeCheckpoint | None,
        verified_head: _VerifiedAppendHead | None,
        implementation_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
    ) -> None:
        if seal is not _APPEND_SESSION_SEAL:
            raise OwnedEpisodeCheckpointContractError(
                "append sessions must be created by the checkpoint factory"
            )
        self._root = root
        self._run_id = run_id
        self._protocol_sha256 = protocol_sha256
        self._genesis_state_sha256 = genesis_state_sha256
        self._backend = backend
        self._lock_context = lock_context
        self._loaded_head = loaded_head
        self._verified_head = verified_head
        self._implementation_identity = implementation_identity
        self._runtime_identity = runtime_identity
        self._process_id = os.getpid()
        self._thread_id = threading.get_ident()
        self._closed = False

    def __enter__(self) -> OwnedEpisodeDeltaAppendSession:
        if self._closed:
            raise OwnedEpisodeCheckpointContractError(
                "append session is already closed"
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def loaded_head(self) -> LoadedOwnedEpisodeCheckpoint | None:
        return self._loaded_head

    def _require_owner(self) -> None:
        if self._closed:
            raise OwnedEpisodeCheckpointContractError(
                "append session is already closed"
            )
        if os.getpid() != self._process_id:
            raise OwnedEpisodeCheckpointIntegrityError(
                "append session cannot be used after process fork"
            )
        if threading.get_ident() != self._thread_id:
            raise OwnedEpisodeCheckpointIntegrityError(
                "append session cannot move between threads"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._require_owner()
        self._closed = True
        self._lock_context.__exit__(None, None, None)

    def append(
        self,
        *,
        input_state: CUDARestrictedWHState,
        result: CUDARestrictedWHOwnedEpisodeResult,
        source_tile: FrozenSourcePathTile,
        spec: FixedStepCUDARestrictedWHSpec,
        route_spec: RestrictedWHRouteSpec | None = None,
        route_worker_count: int = 6,
        route_lookahead_steps: int = 32,
    ) -> LoadedOwnedEpisodeCheckpoint:
        """Replay and append exactly one new segment at the locked frontier."""

        self._require_owner()
        try:
            if type(spec) is not FixedStepCUDARestrictedWHSpec:
                raise OwnedEpisodeCheckpointContractError(
                    "spec must be an exact FixedStepCUDARestrictedWHSpec"
                )
            if spec.backend != self._backend:
                raise OwnedEpisodeCheckpointContractError(
                    "append session backend cannot change"
                )
            if route_spec is None:
                route_spec = RestrictedWHRouteSpec(
                    maximum_primary_dynamical_phase=(
                        spec.maximum_primary_dynamical_phase
                    ),
                    primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
                )
            _validate_segment_shape(
                input_state,
                result,
                source_tile,
                spec,
                route_spec,
                route_worker_count,
                route_lookahead_steps,
            )
            current_implementation, current_runtime = (
                _current_checkpoint_identities(self._backend)
            )
            if not _checkpoint_codec._canonical_tree_equal(
                current_implementation, self._implementation_identity
            ):
                raise OwnedEpisodeCheckpointIntegrityError(
                    "append session implementation identity changed"
                )
            if not _checkpoint_codec._canonical_tree_equal(
                current_runtime, self._runtime_identity
            ):
                raise OwnedEpisodeCheckpointIntegrityError(
                    "append session host/CUDA runtime identity changed"
                )
            checkpoint_index = (
                0
                if self._verified_head is None
                else self._verified_head.checkpoint_index + 1
            )
            predecessor = (
                None
                if self._verified_head is None
                else self._verified_head.checkpoint_content_sha256
            )
            record = _dump_locked(
                self._root,
                run_id=self._run_id,
                protocol_sha256=self._protocol_sha256,
                genesis_state_sha256=self._genesis_state_sha256,
                checkpoint_index=checkpoint_index,
                input_state=input_state,
                result=result,
                source_tile=source_tile,
                spec=spec,
                route_spec=route_spec,
                route_worker_count=route_worker_count,
                route_lookahead_steps=route_lookahead_steps,
                previous_checkpoint_content_sha256=predecessor,
                verified_head=self._verified_head,
                bound_implementation_identity=(
                    self._implementation_identity
                ),
                bound_runtime_identity=self._runtime_identity,
            )
            loaded, verified_head = _read_published_append_head(
                self._root,
                run_id=self._run_id,
                protocol_sha256=self._protocol_sha256,
                genesis_state_sha256=self._genesis_state_sha256,
                backend=self._backend,
                expected_record=record,
                implementation_identity=self._implementation_identity,
                runtime_identity=self._runtime_identity,
            )
        except BaseException:
            self.close()
            raise
        self._loaded_head = loaded
        self._verified_head = verified_head
        return loaded


def open_owned_episode_delta_append_session(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    backend: BackendSpec,
    expected_checkpoint_index: int | None = None,
    expected_checkpoint_content_sha256: str | None = None,
) -> OwnedEpisodeDeltaAppendSession:
    """Lock one namespace and establish a replay-verified append frontier."""

    root = Path(directory)
    _text(run_id, "run_id")
    _sha256(protocol_sha256, "protocol_sha256")
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise OwnedEpisodeCheckpointContractError(
            "append sessions require an exact CuPy BackendSpec"
        )
    if (expected_checkpoint_index is None) != (
        expected_checkpoint_content_sha256 is None
    ):
        raise OwnedEpisodeCheckpointContractError(
            "expected checkpoint index and digest must be supplied together"
        )
    if expected_checkpoint_index is not None:
        if (
            type(expected_checkpoint_index) is not int
            or not 0 <= expected_checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
        ):
            raise OwnedEpisodeCheckpointContractError(
                "expected_checkpoint_index must be an integer in [0, 10**20)"
            )
        _sha256(
            expected_checkpoint_content_sha256,
            "expected_checkpoint_content_sha256",
        )
    namespace = _checkpoint_namespace(root, run_id, protocol_sha256)
    _checkpoint_codec._ensure_namespace(root, namespace)
    lock_context = _checkpoint_codec._checkpoint_append_lock(namespace)
    lock_context.__enter__()
    try:
        records = _scan_records(root, run_id, protocol_sha256)
        if records:
            if expected_checkpoint_index is None:
                raise OwnedEpisodeCheckpointIntegrityError(
                    "existing append session requires an external head anchor"
                )
            loaded = load_latest_owned_episode_delta_checkpoint(
                root,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_state_sha256,
                expected_checkpoint_index=expected_checkpoint_index,
                expected_checkpoint_content_sha256=(
                    expected_checkpoint_content_sha256
                ),
                backend=backend,
            )
            verified_head = _verified_append_head(
                loaded,
                records,
                run_id=run_id,
                protocol_sha256=protocol_sha256,
                genesis_state_sha256=genesis_state_sha256,
            )
        else:
            if expected_checkpoint_index is not None:
                raise OwnedEpisodeCheckpointIntegrityError(
                    "external append head was supplied for an empty chain"
                )
            loaded = None
            verified_head = None
        implementation, runtime = _current_checkpoint_identities(backend)
        if loaded is not None:
            if not _checkpoint_codec._canonical_tree_equal(
                implementation,
                _checkpoint_codec._plain_tree(
                    loaded.record.get("implementation_identity")
                ),
            ):
                raise OwnedEpisodeCheckpointIntegrityError(
                    "append implementation identity changed after head replay"
                )
            if not _checkpoint_codec._canonical_tree_equal(
                runtime,
                _checkpoint_codec._plain_tree(
                    loaded.record.get("checkpoint_runtime_identity")
                ),
            ):
                raise OwnedEpisodeCheckpointIntegrityError(
                    "append host/CUDA runtime identity changed after head replay"
                )
        return OwnedEpisodeDeltaAppendSession(
            seal=_APPEND_SESSION_SEAL,
            root=root,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            genesis_state_sha256=genesis_state_sha256,
            backend=backend,
            lock_context=lock_context,
            loaded_head=loaded,
            verified_head=verified_head,
            implementation_identity=implementation,
            runtime_identity=runtime,
        )
    except BaseException:
        lock_context.__exit__(None, None, None)
        raise


__all__ = [
    "CHECKPOINT_CLAIMS",
    "CHECKPOINT_METHOD_ID",
    "CHECKPOINT_SCHEMA",
    "IMPLEMENTATION_SOURCE_SHA256",
    "LoadedOwnedEpisodeCheckpoint",
    "OwnedEpisodeDeltaAppendSession",
    "OwnedEpisodeCheckpointContractError",
    "OwnedEpisodeCheckpointError",
    "OwnedEpisodeCheckpointIntegrityError",
    "dump_owned_episode_delta_checkpoint",
    "load_latest_owned_episode_delta_checkpoint",
    "open_owned_episode_delta_append_session",
    "owned_episode_checkpoint_implementation_identity",
    "owned_episode_checkpoint_runtime_identity",
    "owned_episode_state_sha256",
]
