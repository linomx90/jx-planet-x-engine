"""Restricted-WH adapters for the experimental long-horizon façade.

The existing restricted-WH objects remain authoritative.  These adapters
produce backend-neutral immutable receipts without changing numerical arrays,
route decisions, checkpoint identities, or restart behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .. import restricted_wh_checkpoint as _checkpoint_codec
from ..cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHResult,
    CUDARestrictedWHState,
    FrozenSourcePathTile,
)
from ..cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridResult,
    CUDARestrictedWHRouteRecord,
)
from ..restricted_wh_checkpoint import LoadedRestrictedWHCheckpoint
from ..restricted_wh_episode_checkpoint import LoadedOwnedEpisodeCheckpoint
from ..restricted_wh_lifecycle import (
    ACTIVE,
    REMOVED_OUTER_BOUNDARY,
    REMOVED_SOURCE_RADIUS,
    RestrictedWHLifecycleResult,
    RestrictedWHLifecycleState,
)
from ..restricted_wh_route import (
    ROUTE_METHOD_ID,
    RestrictedWHRouteEvent,
)
from ..restricted_wh_source import (
    WisdomHolmanRestrictedWHHybridSegmentResult,
    WisdomHolmanRestrictedWHLifecycleSegmentResult,
    WisdomHolmanRestrictedWHSegmentResult,
)
from .contracts import (
    CLAIM_CEILING,
    PROJECT_SCOPE,
    SCIENTIFIC_CLASSIFICATION,
    LongHorizonArchitectureContract,
    LongHorizonCheckpointContract,
    LongHorizonContractError,
    LongHorizonEventContract,
    LongHorizonResultContract,
)


CANONICAL_ARCHITECTURE_ID = "jxplanetx.long-horizon.restricted-wh.v1"
CANONICAL_FAMILY = "RESTRICTED_WISDOM_HOLMAN"

RESTRICTED_WH_ARCHITECTURE = LongHorizonArchitectureContract(
    architecture_id=CANONICAL_ARCHITECTURE_ID,
    family=CANONICAL_FAMILY,
    components=(
        "REPLAY_VERIFIED_ORDERED_JACOBI_SOURCE",
        "BINARY64_PRIMARY_RELATIVE_RESTRICTED_WH_MAP",
        "CONTINUOUS_DOP853_GUARD_ROUTE",
        "FIXED_LANE_TERMINAL_LIFECYCLE",
        "REPLAY_VERIFIED_CHECKPOINT_CHAIN",
    ),
    entry_points={
        "initial_conditions": (
            "jxplanetx.long_horizon:build_predeclared_initial_condition_inventory"
        ),
        "start": "jxplanetx.long_horizon:start_fixed_lifecycle_run",
        "resume": "jxplanetx.long_horizon:resume_fixed_lifecycle_run",
        "advance_segment": (
            "jxplanetx.long_horizon:advance_fixed_lifecycle_segment"
        ),
    },
)


_DIRECT_RESULTS = (
    CUDARestrictedWHResult,
    CUDARestrictedWHHybridResult,
    RestrictedWHLifecycleResult,
)
_SOURCE_RESULTS = (
    WisdomHolmanRestrictedWHSegmentResult,
    WisdomHolmanRestrictedWHHybridSegmentResult,
    WisdomHolmanRestrictedWHLifecycleSegmentResult,
)


def _verify_source_claim_ceiling(value: object) -> None:
    claims = getattr(value, "claims", None)
    if not isinstance(claims, Mapping):
        raise LongHorizonContractError("restricted-WH source has no claim mapping")
    if any(type(item) is not bool or item for item in claims.values()):
        raise LongHorizonContractError(
            "restricted-WH source attempted to widen a scientific claim"
        )
    project_scope = getattr(value, "project_scope", PROJECT_SCOPE)
    classification = getattr(
        value, "scientific_classification", SCIENTIFIC_CLASSIFICATION
    )
    if project_scope != PROJECT_SCOPE or classification != SCIENTIFIC_CLASSIFICATION:
        raise LongHorizonContractError(
            "restricted-WH source is outside MODEL_OUTPUT / SCREENING_ONLY"
        )


def _float(value: object, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LongHorizonContractError(f"{label} is not scalar numeric metadata") from exc
    if not math.isfinite(converted):
        raise LongHorizonContractError(f"{label} must be finite")
    return converted


def _minimum_distance_diagnostics(
    value: object,
    *,
    tracer_count: int,
) -> dict[str, Any]:
    """Normalize a real scalar fixture or the authoritative per-lane vector.

    Production restricted-WH results retain one binary64 minimum per fixed
    lane.  The façade records that vector's byte identity and an explicitly
    labelled finite-lane reduction; it never misrepresents a device vector as
    scalar metadata.  Scalar support remains only for older CPU contract
    fixtures that predate the real CUDA result shape.
    """

    shape = getattr(value, "shape", None)
    if shape is None:
        return {
            "global_minimum_source_distance_AU": _float(
                value, "minimum source distance"
            ),
            "minimum_source_distance_reduction": "SCALAR_CONTRACT_FIXTURE",
        }
    if type(shape) is not tuple or shape != (tracer_count,):
        raise LongHorizonContractError(
            "minimum source distance must be one value per tracer lane"
        )
    getter = getattr(value, "get", None)
    try:
        host = getter() if callable(getter) else np.asarray(value)
    except Exception as exc:
        raise LongHorizonContractError(
            "minimum source distance vector could not be transferred"
        ) from exc
    array = np.asarray(host)
    if array.dtype != np.dtype(np.float64) or array.shape != shape:
        raise LongHorizonContractError(
            "minimum source distance vector must retain exact float64 shape"
        )
    array = np.ascontiguousarray(array)
    if bool(np.any(np.isnan(array))) or bool(np.any(array < 0.0)):
        raise LongHorizonContractError(
            "minimum source distance vector contains invalid values"
        )
    metadata = json.dumps(
        {"dtype": "float64", "shape": [tracer_count]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    vector_sha256 = hashlib.sha256(
        b"jx.long-horizon.minimum-source-distance-vector.v1\0"
        + metadata
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    diagnostics: dict[str, Any] = {
        "minimum_source_distance_lane_count": tracer_count,
        "minimum_source_distance_finite_lane_count": finite_count,
        "minimum_source_distance_vector_sha256": vector_sha256,
        "minimum_source_distance_reduction": (
            "MINIMUM_OVER_FINITE_FIXED_LANES"
        ),
    }
    if finite_count:
        diagnostics["global_minimum_source_distance_AU"] = float(
            np.min(array[finite])
        )
    return diagnostics


def _state_context(
    state: CUDARestrictedWHState,
) -> tuple[int, float, int, str, str, str, str]:
    if type(state) is not CUDARestrictedWHState:
        raise LongHorizonContractError(
            "restricted-WH state must be an exact CUDARestrictedWHState"
        )
    _positions, _velocities, indices, _lifecycle = _validated_state_arrays(
        state, None
    )
    tracer_count = int(indices.shape[0])
    return (
        state.global_source_step,
        _float(state.epoch_years, "state epoch"),
        tracer_count,
        state.frame,
        state.coordinate_system,
        state.length_unit,
        state.time_unit,
    )


def _host_state_array(
    value: object,
    label: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    getter = getattr(value, "get", None)
    try:
        host = getter() if callable(getter) else np.asarray(value)
    except Exception as exc:
        raise LongHorizonContractError(
            f"{label} could not be transferred for state identity"
        ) from exc
    array = np.asarray(host)
    if array.dtype != dtype or array.ndim != ndim:
        raise LongHorizonContractError(
            f"{label} changed dtype or rank before state identity"
        )
    return np.ascontiguousarray(array)


def _validated_state_arrays(
    state: CUDARestrictedWHState,
    lifecycle: RestrictedWHLifecycleState | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Transfer and validate the exact numerical state before attestation."""

    positions = _host_state_array(
        state.tracer_relative_positions_AU,
        "tracer_relative_positions_AU",
        dtype=np.dtype(np.float64),
        ndim=2,
    )
    velocities = _host_state_array(
        state.tracer_relative_velocities_AU_per_year,
        "tracer_relative_velocities_AU_per_year",
        dtype=np.dtype(np.float64),
        ndim=2,
    )
    indices = _host_state_array(
        state.tracer_indices,
        "tracer_indices",
        dtype=np.dtype(np.uint64),
        ndim=1,
    )
    lane_count = int(indices.shape[0])
    if positions.shape != (lane_count, 3) or velocities.shape != (
        lane_count,
        3,
    ):
        raise LongHorizonContractError(
            "restricted-WH tracer state arrays must have exact matching "
            "(lane_count, 3) shapes"
        )
    if not bool(np.all(np.isfinite(positions))) or not bool(
        np.all(np.isfinite(velocities))
    ):
        raise LongHorizonContractError(
            "restricted-WH tracer state contains non-finite values"
        )
    if int(np.unique(indices).size) != lane_count:
        raise LongHorizonContractError(
            "restricted-WH tracer indices must be unique"
        )

    lifecycle_array: np.ndarray | None = None
    if lifecycle is not None:
        lifecycle_array = _host_state_array(
            lifecycle.tracer_lifecycle,
            "tracer_lifecycle",
            dtype=np.dtype(np.uint8),
            ndim=1,
        )
        if lifecycle_array.shape != (lane_count,) or any(
            int(item)
            not in (ACTIVE, REMOVED_SOURCE_RADIUS, REMOVED_OUTER_BOUNDARY)
            for item in lifecycle_array.tolist()
        ):
            raise LongHorizonContractError(
                "restricted-WH lifecycle vector is inconsistent with its lanes"
            )
        records = {record.lane_index: record for record in lifecycle.removal_records}
        terminal_lanes = set(np.flatnonzero(lifecycle_array != ACTIVE).tolist())
        if len(records) != len(lifecycle.removal_records) or set(records) != terminal_lanes:
            raise LongHorizonContractError(
                "restricted-WH terminal lanes and removal records disagree"
            )
        for lane, record in records.items():
            if (
                lane >= lane_count
                or int(indices[lane]) != record.tracer_index
                or int(lifecycle_array[lane]) != record.lifecycle_code
                or record.global_source_step >= state.global_source_step
                or not np.array_equal(
                    positions[lane], record.event.relative_position_AU
                )
                or not np.array_equal(
                    velocities[lane], record.event.relative_velocity_AU_per_year
                )
            ):
                raise LongHorizonContractError(
                    "restricted-WH removal history is inconsistent with the "
                    "committed numerical state"
                )
            if any(
                event.source_index >= len(state.source_ids)
                or state.source_ids[event.source_index] != event.source_id
                for event in record.events
            ):
                raise LongHorizonContractError(
                    "restricted-WH removal event changed source identity"
                )
    return positions, velocities, indices, lifecycle_array


def _output_state_sha256(
    state: CUDARestrictedWHState,
    lifecycle: RestrictedWHLifecycleState | None,
) -> str:
    """Hash exact state semantics and array bytes for the shared receipt."""

    metadata = {
        "schema": "jxplanetx.long-horizon.output-state-identity.v1",
        "snapshot_id": state.snapshot_id,
        "epoch_origin_years": {"float_hex": float(state.epoch_origin_years).hex()},
        "global_source_step": state.global_source_step,
        "source_step_years": {"float_hex": float(state.source_step_years).hex()},
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
        "provenance": {
            "source_id": state.provenance.source_id,
            "citation": state.provenance.citation,
            "version": state.provenance.version,
            "sha256": state.provenance.sha256,
        },
        "removal_record_sha256": (
            []
            if lifecycle is None
            else [record.content_sha256 for record in lifecycle.removal_records]
        ),
    }
    positions, velocities, indices, lifecycle_array = _validated_state_arrays(
        state, lifecycle
    )
    arrays: list[tuple[str, np.ndarray]] = [
        ("tracer_relative_positions_AU", positions),
        ("tracer_relative_velocities_AU_per_year", velocities),
        ("tracer_indices", indices),
    ]
    if lifecycle_array is not None:
        arrays.append(
            (
                "tracer_lifecycle",
                lifecycle_array,
            )
        )
    digest = hashlib.sha256(
        b"jxplanetx.long-horizon.output-state-identity.v1\0"
    )
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    for name, array in arrays:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0" + array.tobytes(order="C"))
    return digest.hexdigest()


def _route_ledger_sha256(
    result: object,
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile | None,
) -> str:
    """Bind every committed route row and any worker assignment exactly."""

    route_records = getattr(result, "route_records", ())
    if type(route_records) is not tuple or any(
        type(record) is not CUDARestrictedWHRouteRecord
        for record in route_records
    ):
        raise LongHorizonContractError(
            "restricted-WH route ledger must be an exact immutable tuple"
        )
    _positions, _velocities, indices, _lifecycle = _validated_state_arrays(
        state, None
    )
    lane_count = int(indices.shape[0])
    order = tuple(
        (record.map_step_index, record.lane_index, record.tracer_index)
        for record in route_records
    )
    if order != tuple(sorted(order)) or len(order) != len(set(order)):
        raise LongHorizonContractError(
            "restricted-WH route ledger is not unique and canonical"
        )
    ratios = {
        record.route_result.source_steps_per_map_step
        for record in route_records
    }
    if route_records and len(ratios) != 1:
        raise LongHorizonContractError(
            "restricted-WH route ledger changed its segment lattice"
        )
    if route_records and source_tile is None:
        raise LongHorizonContractError(
            "routed restricted-WH results require their exact source wrapper tile"
        )
    if source_tile is not None:
        if (
            type(source_tile) is not FrozenSourcePathTile
            or state.epoch_origin_years != source_tile.epoch_origin_years
            or state.source_step_years != source_tile.source_step_years
            or state.time_scale != source_tile.time_scale
            or state.frame != source_tile.frame
            or state.axes != source_tile.axes
            or state.coordinate_system != source_tile.coordinate_system
            or state.length_unit != source_tile.length_unit
            or state.time_unit != source_tile.time_unit
            or state.unit_system_id != source_tile.unit_system_id
            or state.source_ids != source_tile.source_ids
            or state.primary_source_id != source_tile.primary_source_id
            or state.source_schedule_sha256
            != source_tile.source_schedule_sha256
            or state.global_source_step
            != source_tile.final_global_source_step
        ):
            raise LongHorizonContractError(
                "restricted-WH source wrapper changed its exact tile context"
            )
        ratio = next(iter(ratios)) if route_records else None
        tile_span = (
            source_tile.final_global_source_step
            - source_tile.global_source_step
        )
        if route_records and (
            tile_span <= 0
            or tile_span % ratio != 0
            or any(
                record.route_result.global_source_step
                != source_tile.global_source_step
                + record.map_step_index * ratio
                or record.route_result.source_steps_per_map_step != ratio
                or record.map_step_index >= tile_span // ratio
                for record in route_records
            )
        ):
            raise LongHorizonContractError(
                "restricted-WH route ledger disagrees with its source tile lattice"
            )
    for record in route_records:
        route = record.route_result
        if (
            record.lane_index >= lane_count
            or int(indices[record.lane_index]) != record.tracer_index
            or route.source_tile_sha256 != result.source_tile_sha256
            or route.global_source_step + route.source_steps_per_map_step
            > state.global_source_step
            or (
                record.guard_source_index is not None
                and record.guard_source_index >= len(state.source_ids)
            )
        ):
            raise LongHorizonContractError(
                "restricted-WH route ledger is inconsistent with the "
                "committed state or source identity"
            )
    worker_indices = getattr(result, "route_worker_indices", None)
    if worker_indices is not None and (
        type(worker_indices) is not tuple
        or len(worker_indices) != len(route_records)
        or any(type(item) is not int or item < 0 for item in worker_indices)
    ):
        raise LongHorizonContractError(
            "restricted-WH route worker assignment is inconsistent"
        )
    try:
        payload = {
            "schema": "jxplanetx.long-horizon.route-ledger.v1",
            "route_records": [
                _checkpoint_codec._route_record(record)
                for record in route_records
            ],
            "route_worker_indices": (
                None if worker_indices is None else list(worker_indices)
            ),
        }
        encoded = _checkpoint_codec._canonical_json(payload)
    except Exception as exc:
        raise LongHorizonContractError(
            "restricted-WH route ledger could not be canonically encoded"
        ) from exc
    return hashlib.sha256(
        b"jxplanetx.long-horizon.route-ledger.v1\0" + encoded
    ).hexdigest()


def _validate_new_removal_suffix(
    result: RestrictedWHLifecycleResult,
) -> None:
    committed = result.state.removal_records
    new = result.new_removal_records
    if len(new) > len(committed):
        raise LongHorizonContractError(
            "new removal records exceed committed lifecycle history"
        )
    suffix = committed[len(committed) - len(new) :] if new else ()
    if tuple(item.content_sha256 for item in new) != tuple(
        item.content_sha256 for item in suffix
    ):
        raise LongHorizonContractError(
            "new removal records are not the exact committed history suffix"
        )


def adapt_restricted_wh_event(
    event: RestrictedWHRouteEvent,
    *,
    tracer_index: int,
    route_start_global_source_step: int,
    source_steps_per_map_step: int,
    frame: str,
    coordinate_system: str = COORDINATE_SYSTEM,
    length_unit: str = "AU",
    time_unit: str = "year",
    lane_index: int | None = None,
) -> LongHorizonEventContract:
    """Normalize one exact restricted-WH route event.

    Context absent from the low-level event must be supplied explicitly; the
    adapter never guesses tracer, lattice, frame, or lane identity.
    """

    if type(event) is not RestrictedWHRouteEvent:
        raise LongHorizonContractError(
            "event must be an exact RestrictedWHRouteEvent"
        )
    if (
        type(route_start_global_source_step) is not int
        or route_start_global_source_step < 0
    ):
        raise LongHorizonContractError(
            "route_start_global_source_step must be a nonnegative integer"
        )
    if (
        type(source_steps_per_map_step) is not int
        or source_steps_per_map_step <= 0
        or event.source_interval_offset >= source_steps_per_map_step
    ):
        raise LongHorizonContractError(
            "source_steps_per_map_step must contain the event interval"
        )
    expected_map_fraction = (
        event.source_interval_offset + event.interval_fraction
    ) / source_steps_per_map_step
    if event.map_step_fraction != expected_map_fraction:
        raise LongHorizonContractError(
            "event map fraction is inconsistent with its fine source interval"
        )
    canonical_step = (
        route_start_global_source_step + event.source_interval_offset
    )
    canonical_fraction = float(event.interval_fraction)
    if canonical_fraction == 1.0:
        canonical_step += 1
        canonical_fraction = 0.0
    return LongHorizonEventContract(
        event_type=event.event_type,
        tracer_index=tracer_index,
        lane_index=lane_index,
        global_step=canonical_step,
        step_fraction=canonical_fraction,
        source_index=event.source_index,
        source_id=event.source_id,
        relative_position=tuple(
            float(item) for item in event.relative_position_AU.tolist()
        ),
        relative_velocity=tuple(
            float(item) for item in event.relative_velocity_AU_per_year.tolist()
        ),
        coordinate_system=coordinate_system,
        frame=frame,
        length_unit=length_unit,
        time_unit=time_unit,
        location_method=event.location_method,
        origin_method_id=ROUTE_METHOD_ID,
        terminal=True,
        diagnostics={
            "route_start_global_source_step": route_start_global_source_step,
            "source_steps_per_map_step": source_steps_per_map_step,
            "map_step_fraction": float(event.map_step_fraction),
            "source_interval_offset": event.source_interval_offset,
            "interval_fraction": float(event.interval_fraction),
            "surface_distance": float(event.surface_distance_AU),
            "surface_threshold": float(event.surface_threshold_AU),
            "surface_residual": float(event.surface_residual_AU),
            "surface_tolerance": float(event.surface_tolerance_AU),
            "primary_specific_energy": float(
                event.primary_specific_energy_AU2_per_year2
            ),
            "primary_radial_velocity": float(
                event.primary_radial_velocity_AU_per_year
            ),
            "surface_radial_velocity": float(
                event.surface_radial_velocity_AU_per_year
            ),
        },
    )


def _unwrap_result(
    value: object,
) -> tuple[object, FrozenSourcePathTile | None]:
    if type(value) in _SOURCE_RESULTS:
        _verify_source_claim_ceiling(value)
        tile = value.source_build.tile
        if type(tile) is not FrozenSourcePathTile:
            raise LongHorizonContractError(
                "restricted-WH source wrapper has no exact source tile"
            )
        return value.tracer_result, tile
    return value, None


def adapt_restricted_wh_result(value: object) -> LongHorizonResultContract:
    """Normalize a fully constructed restricted-WH segment result.

    Only exact known result classes are accepted.  Failure handoffs and partial
    work are deliberately not adaptable as successful result receipts.
    """

    result, source_tile = _unwrap_result(value)
    if type(result) not in _DIRECT_RESULTS:
        raise LongHorizonContractError(
            "unsupported restricted-WH result type; partial or foreign results "
            "cannot be normalized"
        )
    _verify_source_claim_ceiling(result)

    if type(result) is RestrictedWHLifecycleResult:
        lifecycle = result.state
        if type(lifecycle) is not RestrictedWHLifecycleState:
            raise LongHorizonContractError("lifecycle result state type changed")
        _validate_new_removal_suffix(result)
        state = lifecycle.dynamics_state
        active_count = sum(
            1 for item in lifecycle.tracer_lifecycle.tolist() if int(item) == ACTIVE
        )
        event_rows = [
            adapt_restricted_wh_event(
                event,
                tracer_index=record.tracer_index,
                lane_index=record.lane_index,
                route_start_global_source_step=record.global_source_step,
                source_steps_per_map_step=record.source_steps_per_map_step,
                frame=state.frame,
                coordinate_system=state.coordinate_system,
                length_unit=state.length_unit,
                time_unit=state.time_unit,
            )
            for record in lifecycle.removal_records
            for event in record.events
        ]
    else:
        state = result.state
        lifecycle = None
        event_rows = []
        active_count = None

    (
        global_step,
        epoch,
        tracer_count,
        _frame,
        _coordinate_system,
        _length_unit,
        _time_unit,
    ) = _state_context(state)
    if active_count is None:
        active_count = tracer_count

    events = tuple(
        sorted(
            event_rows,
            key=lambda item: (
                item.global_step,
                item.step_fraction,
                -1 if item.lane_index is None else item.lane_index,
                item.tracer_index,
                item.event_type,
                item.content_sha256,
            ),
        )
    )
    route_records = getattr(result, "route_records", ())
    route_ledger_sha256 = _route_ledger_sha256(
        result, state, source_tile
    )
    minimum = getattr(
        result,
        "minimum_source_distance_AU",
        getattr(result, "minimum_endpoint_source_distance_AU", None),
    )
    diagnostics: dict[str, Any] = {
        "maximum_kepler_iterations": int(result.maximum_kepler_iterations),
    }
    if minimum is not None:
        diagnostics.update(
            _minimum_distance_diagnostics(
                minimum,
                tracer_count=tracer_count,
            )
        )
    if hasattr(result, "episodes"):
        diagnostics["episode_count"] = len(result.episodes)
    if lifecycle is not None:
        diagnostics["event_ledger_scope"] = (
            "CUMULATIVE_COMMITTED_LIFECYCLE_HISTORY"
        )

    return LongHorizonResultContract(
        method_id=result.method_id,
        status="COMPLETED",
        global_step=global_step,
        epoch=epoch,
        source_tile_sha256=result.source_tile_sha256,
        output_state_sha256=_output_state_sha256(state, lifecycle),
        tracer_count=tracer_count,
        active_tracer_count=active_count,
        route_count=len(route_records),
        route_ledger_sha256=route_ledger_sha256,
        events=events,
        runtime_identity=result.runtime_identity,
        diagnostics=diagnostics,
        claims=CLAIM_CEILING,
    )


def adapt_restricted_wh_checkpoint(value: object) -> LongHorizonCheckpointContract:
    """Normalize an already replay-verified restricted-WH checkpoint head."""

    if type(value) is LoadedRestrictedWHCheckpoint:
        _verify_source_claim_ceiling(value)
        state = value.state.dynamics_state
        lineage: dict[str, Any] = {
            "variant": "FIXED_LANE_LIFECYCLE",
        }
    elif type(value) is LoadedOwnedEpisodeCheckpoint:
        _verify_source_claim_ceiling(value)
        state = value.state
        lineage = {
            "variant": "OWNED_EPISODE_DELTA",
            "route_chain_sha256": value.route_chain_sha256,
            "route_record_count_cumulative": value.route_record_count_cumulative,
            "segment_episode_count_cumulative": (
                value.segment_episode_count_cumulative
            ),
        }
    else:
        raise LongHorizonContractError(
            "checkpoint must be an exact replay-verified restricted-WH load result"
        )

    global_step, epoch, _count, _frame, _coordinates, _length, _time = (
        _state_context(state)
    )
    return LongHorizonCheckpointContract(
        method_id=value.method_id,
        status="VERIFIED",
        checkpoint_index=value.checkpoint_index,
        run_id=value.run_id,
        protocol_sha256=value.protocol_sha256,
        checkpoint_content_sha256=value.checkpoint_content_sha256,
        input_state_sha256=value.input_state_sha256,
        state_sha256=value.state_sha256,
        global_step=global_step,
        epoch=epoch,
        lineage=lineage,
        claims=CLAIM_CEILING,
    )


__all__ = [
    "CANONICAL_ARCHITECTURE_ID",
    "CANONICAL_FAMILY",
    "RESTRICTED_WH_ARCHITECTURE",
    "adapt_restricted_wh_checkpoint",
    "adapt_restricted_wh_event",
    "adapt_restricted_wh_result",
]
