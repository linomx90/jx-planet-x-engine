"""Replay-verified checkpoints for the experimental restricted-WH lifecycle.

This module is deliberately additive and direct-import only.  It persists one
fully committed lifecycle segment per checkpoint record.  Every record binds
the exact input state, output state, immutable source tile, numerical policies,
runtime evidence, per-segment route/removal witnesses, and predecessor digest.

Loading is intentionally expensive and precision-first: the complete committed
chain is verified and every segment is replayed on the requested CUDA runtime
before the latest state is released.  A JSON record is published create-only
after its content-addressed NumPy archive.  An archive orphaned by a crash is
harmless; a committed record with missing or changed bytes fails closed.
Here, crash safety means process interruption on the local POSIX filesystem;
it is not a hardware power-loss or storage-controller durability claim.

The checkpoint layer proves restart integrity only within the explicitly bound
runtime and inputs.  It makes no production, long-horizon-accuracy, encounter,
collision, ejection, or science claim.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import _all_conic_kepler as _all_conic_module
from . import cuda_restricted_wh as _restricted_module
from . import cuda_restricted_wh_hybrid as _hybrid_module
from . import restricted_wh_lifecycle as _lifecycle_module
from . import restricted_wh_route as _route_module
from .cuda_restricted_wh import (
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    verify_source_path_tile,
)
from .cuda_restricted_wh_hybrid import (
    CUDARestrictedWHRouteRecord,
    DEFAULT_ROUTE_WORKER_COUNT,
)
from .cuda_tracer_events import DOP853EventLocatorSpec
from .engine import backends as _backends_module
from .engine import contracts as _contracts_module
from .engine import wisdom_holman_contracts as _wh_contracts_module
from .engine.backends import resolve_backend
from .engine.contracts import BackendSpec, Provenance
from .restricted_wh_lifecycle import (
    LIFECYCLE_CLAIMS,
    LIFECYCLE_METHOD_ID,
    RestrictedWHLifecycleResult,
    RestrictedWHLifecycleState,
    RestrictedWHRemovalRecord,
    advance_cuda_restricted_wh_lifecycle,
)
from .restricted_wh_route import (
    ROUTE_CLAIMS,
    ROUTE_METHOD_ID,
    RestrictedWHRouteEvent,
    RestrictedWHRouteResult,
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
)


CHECKPOINT_SCHEMA = "jxplanetx.restricted-wh-lifecycle-checkpoint.v1"
CHECKPOINT_METHOD_ID = "JX_RESTRICTED_WH_REPLAY_VERIFIED_CHECKPOINT_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"

CHECKPOINT_DOMAIN = b"jxplanetx.restricted-wh.checkpoint-record.v1\0"
CHECKPOINT_NAMESPACE_DOMAIN = b"jxplanetx.restricted-wh.checkpoint-namespace.v1\0"
STATE_DOMAIN = b"jxplanetx.restricted-wh.lifecycle-state.v1\0"
LINEAGE_DOMAIN = b"jxplanetx.restricted-wh.lifecycle-lineage.v1\0"
SEGMENT_DOMAIN = b"jxplanetx.restricted-wh.checkpoint-segment.v1\0"
SOURCE_BOUNDARY_DOMAIN = b"jxplanetx.restricted-wh.source-boundary.v1\0"
SOURCE_LINEAGE_DOMAIN = b"jxplanetx.restricted-wh.source-lineage.v1\0"
MAXIMUM_CHECKPOINT_INDEX = 10**20 - 1

IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EVENT_LOCATOR_SOURCE_SHA256 = _route_module.EVENT_LOCATOR_SOURCE_SHA256
ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256 = (
    _restricted_module.ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256
)
ALL_CONIC_DEVICE_SOURCE_SHA256 = hashlib.sha256(
    _all_conic_module.CUDA_DEVICE_SOURCE.encode("ascii")
).hexdigest()
BACKENDS_IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(
    Path(_backends_module.__file__).read_bytes()
).hexdigest()
CONTRACTS_IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(
    Path(_contracts_module.__file__).read_bytes()
).hexdigest()
WH_CONTRACTS_IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(
    Path(_wh_contracts_module.__file__).read_bytes()
).hexdigest()

CHECKPOINT_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "four_point_five_gyr_accuracy_qualified": False,
        "cross_runtime_bitwise_reproducible": False,
        "close_encounter_episode_qualified": False,
        "physical_collision_inferred": False,
        "physical_ejection_inferred": False,
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

_ARRAY_KEYS = frozenset(
    {
        "input_tracer_relative_positions_AU",
        "input_tracer_relative_velocities_AU_per_year",
        "input_tracer_indices",
        "input_tracer_lifecycle",
        "output_tracer_relative_positions_AU",
        "output_tracer_relative_velocities_AU_per_year",
        "output_tracer_indices",
        "output_tracer_lifecycle",
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
        "genesis_removal_records",
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
        "result",
        "implementation_identity",
        "checkpoint_runtime_identity",
        "claim_ceiling",
        "checkpoint_content_sha256",
    }
)


class RestrictedWHCheckpointError(RuntimeError):
    """Base error for replay-verified restricted-WH checkpoints."""


class RestrictedWHCheckpointContractError(RestrictedWHCheckpointError, ValueError):
    """A checkpoint request violates the immutable public contract."""


class RestrictedWHCheckpointIntegrityError(RestrictedWHCheckpointError):
    """Stored checkpoint evidence failed a hash, chain, or replay check."""


_LOADER_VERIFICATION_SEAL = object()


@dataclass(frozen=True, eq=False)
class LoadedRestrictedWHCheckpoint:
    """Latest state released only after complete chain verification and replay."""

    state: RestrictedWHLifecycleState
    checkpoint_index: int
    checkpoint_content_sha256: str
    input_state_sha256: str
    state_sha256: str
    run_id: str
    protocol_sha256: str
    record: Mapping[str, Any]
    claims: Mapping[str, bool] = CHECKPOINT_CLAIMS
    method_id: str = CHECKPOINT_METHOD_ID
    _verification_seal: object = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._verification_seal is not _LOADER_VERIFICATION_SEAL:
            raise RestrictedWHCheckpointContractError(
                "LoadedRestrictedWHCheckpoint is released only by the "
                "replay-verifying loader"
            )
        if type(self.state) is not RestrictedWHLifecycleState:
            raise RestrictedWHCheckpointContractError(
                "state must be an exact RestrictedWHLifecycleState"
            )
        if (
            type(self.checkpoint_index) is not int
            or not 0 <= self.checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
        ):
            raise RestrictedWHCheckpointContractError(
                "checkpoint_index must be an integer in [0, 10**20)"
            )
        for value, label in (
            (self.checkpoint_content_sha256, "checkpoint_content_sha256"),
            (self.input_state_sha256, "input_state_sha256"),
            (self.state_sha256, "state_sha256"),
            (self.protocol_sha256, "protocol_sha256"),
        ):
            _sha256(value, label)
        _text(self.run_id, "run_id")
        if self.method_id != CHECKPOINT_METHOD_ID:
            raise RestrictedWHCheckpointContractError(
                "checkpoint method identity changed"
            )
        if not _canonical_tree_equal(
            dict(self.claims), dict(CHECKPOINT_CLAIMS)
        ) or any(
            self.claims.values()
        ):
            raise RestrictedWHCheckpointContractError(
                "checkpoint claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(CHECKPOINT_CLAIMS)))
        object.__setattr__(self, "record", _freeze_tree(self.record))


def _release_verified_checkpoint(
    *,
    state: RestrictedWHLifecycleState,
    checkpoint_index: int,
    checkpoint_content_sha256: str,
    input_state_sha256: str,
    state_sha256: str,
    run_id: str,
    protocol_sha256: str,
    record: Mapping[str, Any],
) -> LoadedRestrictedWHCheckpoint:
    """Create a loaded head only inside the replay-verifying loader."""

    value = object.__new__(LoadedRestrictedWHCheckpoint)
    for name, item in (
        ("state", state),
        ("checkpoint_index", checkpoint_index),
        ("checkpoint_content_sha256", checkpoint_content_sha256),
        ("input_state_sha256", input_state_sha256),
        ("state_sha256", state_sha256),
        ("run_id", run_id),
        ("protocol_sha256", protocol_sha256),
        ("record", record),
        ("claims", CHECKPOINT_CLAIMS),
        ("method_id", CHECKPOINT_METHOD_ID),
        ("_verification_seal", _LOADER_VERIFICATION_SEAL),
    ):
        object.__setattr__(value, name, item)
    value.__post_init__()
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise RestrictedWHCheckpointContractError(
            f"{label} must be a nonempty trimmed string"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise RestrictedWHCheckpointContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _hex_float(value: object, label: str) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise RestrictedWHCheckpointContractError(
            f"{label} must be a finite built-in float"
        )
    return value.hex()


def _from_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise RestrictedWHCheckpointIntegrityError(
            f"{label} is not a hexadecimal binary64 string"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise RestrictedWHCheckpointIntegrityError(
            f"{label} is not a hexadecimal binary64 string"
        ) from exc
    if not math.isfinite(parsed):
        raise RestrictedWHCheckpointIntegrityError(f"{label} is not finite")
    return parsed


def _plain_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evidence_tree(value: Any) -> Any:
    """Convert runtime evidence to a type-explicit canonical JSON tree."""

    if isinstance(value, MappingABC):
        return {
            str(key): _evidence_tree(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_evidence_tree(item) for item in value]
    if isinstance(value, np.generic):
        return _evidence_tree(value.item())
    if type(value) is float:
        if not math.isfinite(value):
            raise RestrictedWHCheckpointContractError(
                "runtime evidence contains a non-finite float"
            )
        return {"binary64_hex": value.hex()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise RestrictedWHCheckpointContractError(
        f"runtime evidence contains unsupported type {type(value).__name__}"
    )


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RestrictedWHCheckpointContractError(
            "checkpoint content is not canonical JSON"
        ) from exc


def _canonical_tree_equal(left: Any, right: Any) -> bool:
    """Compare JSON evidence without Python's bool/integer coercion."""

    return _canonical_json(left) == _canonical_json(right)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise RestrictedWHCheckpointIntegrityError(
            f"cannot read checkpoint artifact: {path}"
        ) from exc
    return digest.hexdigest()


def _array_digest_update(
    digest: "hashlib._Hash", label: str, array: np.ndarray
) -> None:
    value = np.ascontiguousarray(array)
    digest.update(label.encode("ascii") + b"\0")
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical_json(list(value.shape)) + b"\0")
    digest.update(value.tobytes(order="C"))


def _provenance_record(value: Provenance) -> dict[str, str]:
    if type(value) is not Provenance:
        raise RestrictedWHCheckpointContractError(
            "checkpoint provenance must be an exact Provenance"
        )
    return {
        "source_id": value.source_id,
        "citation": value.citation,
        "version": value.version,
        "sha256": value.sha256,
    }


def _provenance_from_record(value: object, label: str) -> Provenance:
    if type(value) is not dict or set(value) != {
        "source_id",
        "citation",
        "version",
        "sha256",
    }:
        raise RestrictedWHCheckpointIntegrityError(
            f"{label} provenance record is invalid"
        )
    try:
        return Provenance(**value)
    except Exception as exc:
        raise RestrictedWHCheckpointIntegrityError(
            f"{label} provenance record is invalid"
        ) from exc


def _state_metadata(state: RestrictedWHLifecycleState) -> dict[str, Any]:
    if type(state) is not RestrictedWHLifecycleState:
        raise RestrictedWHCheckpointContractError(
            "state must be an exact RestrictedWHLifecycleState"
        )
    dynamics = state.dynamics_state
    return {
        "snapshot_id": dynamics.snapshot_id,
        "epoch_origin_years_hex": float(dynamics.epoch_origin_years).hex(),
        "global_source_step": dynamics.global_source_step,
        "source_step_years_hex": float(dynamics.source_step_years).hex(),
        "time_scale": dynamics.time_scale,
        "frame": dynamics.frame,
        "axes": dynamics.axes,
        "coordinate_system": dynamics.coordinate_system,
        "length_unit": dynamics.length_unit,
        "time_unit": dynamics.time_unit,
        "unit_system_id": dynamics.unit_system_id,
        "source_ids": list(dynamics.source_ids),
        "primary_source_id": dynamics.primary_source_id,
        "source_schedule_sha256": dynamics.source_schedule_sha256,
        "provenance": _provenance_record(dynamics.provenance),
    }


def _state_arrays(
    state: RestrictedWHLifecycleState,
    cp: Any,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    dynamics = state.dynamics_state
    arrays = {
        f"{prefix}_tracer_relative_positions_AU": cp.asnumpy(
            dynamics.tracer_relative_positions_AU
        ),
        f"{prefix}_tracer_relative_velocities_AU_per_year": cp.asnumpy(
            dynamics.tracer_relative_velocities_AU_per_year
        ),
        f"{prefix}_tracer_indices": cp.asnumpy(dynamics.tracer_indices),
        f"{prefix}_tracer_lifecycle": np.asarray(state.tracer_lifecycle),
    }
    return {
        key: np.ascontiguousarray(value)
        for key, value in arrays.items()
    }


def _state_sha256(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    removal_rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> str:
    digest = hashlib.sha256(STATE_DOMAIN)
    digest.update(_canonical_json(dict(metadata)))
    for suffix in (
        "tracer_relative_positions_AU",
        "tracer_relative_velocities_AU_per_year",
        "tracer_indices",
        "tracer_lifecycle",
    ):
        key = f"{prefix}_{suffix}"
        _array_digest_update(digest, suffix, arrays[key])
    digest.update(_canonical_json(removal_rows))
    return digest.hexdigest()


def _lineage_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *, prefix: str
) -> str:
    invariant = {
        key: value
        for key, value in metadata.items()
        if key != "global_source_step"
    }
    digest = hashlib.sha256(LINEAGE_DOMAIN)
    digest.update(_canonical_json(invariant))
    _array_digest_update(digest, "tracer_indices", arrays[f"{prefix}_tracer_indices"])
    return digest.hexdigest()


def _tile_metadata(tile: FrozenSourcePathTile) -> dict[str, Any]:
    return {
        "tile_id": tile.tile_id,
        "epoch_origin_years_hex": float(tile.epoch_origin_years).hex(),
        "global_source_step": tile.global_source_step,
        "source_step_years_hex": float(tile.source_step_years).hex(),
        "time_scale": tile.time_scale,
        "frame": tile.frame,
        "axes": tile.axes,
        "coordinate_system": tile.coordinate_system,
        "length_unit": tile.length_unit,
        "time_unit": tile.time_unit,
        "unit_system_id": tile.unit_system_id,
        "source_ids": list(tile.source_ids),
        "primary_source_id": tile.primary_source_id,
        "source_schedule_sha256": tile.source_schedule_sha256,
        "provenance": _provenance_record(tile.provenance),
        "content_sha256": verify_source_path_tile(tile),
    }


def _tile_arrays(tile: FrozenSourcePathTile) -> dict[str, np.ndarray]:
    return {
        "source_relative_positions_AU": np.ascontiguousarray(
            tile.source_relative_positions_AU
        ),
        "source_relative_velocities_AU_per_year": np.ascontiguousarray(
            tile.source_relative_velocities_AU_per_year
        ),
        "source_gm_AU3_per_year2": np.ascontiguousarray(
            tile.source_gm_AU3_per_year2
        ),
        "source_encounter_radii_AU": np.ascontiguousarray(
            tile.source_encounter_radii_AU
        ),
        "certified_relative_acceleration_bounds_AU_per_year2": (
            np.ascontiguousarray(
                tile.certified_relative_acceleration_bounds_AU_per_year2
            )
        ),
    }


def _source_lineage_sha256(tile: FrozenSourcePathTile) -> str:
    """Bind source identities and immutable physical parameters across tiles."""

    digest = hashlib.sha256(SOURCE_LINEAGE_DOMAIN)
    digest.update(
        _canonical_json(
            {
                "epoch_origin_years_hex": float(tile.epoch_origin_years).hex(),
                "source_step_years_hex": float(tile.source_step_years).hex(),
                "time_scale": tile.time_scale,
                "frame": tile.frame,
                "axes": tile.axes,
                "coordinate_system": tile.coordinate_system,
                "length_unit": tile.length_unit,
                "time_unit": tile.time_unit,
                "unit_system_id": tile.unit_system_id,
                "source_ids": list(tile.source_ids),
                "primary_source_id": tile.primary_source_id,
                "source_schedule_sha256": tile.source_schedule_sha256,
                "provenance": _provenance_record(tile.provenance),
            }
        )
    )
    _array_digest_update(
        digest, "source_gm_AU3_per_year2", tile.source_gm_AU3_per_year2
    )
    _array_digest_update(
        digest,
        "source_encounter_radii_AU",
        tile.source_encounter_radii_AU,
    )
    return digest.hexdigest()


def _source_boundary_sha256(tile: FrozenSourcePathTile, node_index: int) -> str:
    """Hash one shared physical source node plus immutable source lineage."""

    if type(node_index) is not int or not 0 <= node_index < tile.node_count:
        raise RestrictedWHCheckpointContractError(
            "source boundary node index is outside the tile"
        )
    digest = hashlib.sha256(SOURCE_BOUNDARY_DOMAIN)
    digest.update(_source_lineage_sha256(tile).encode("ascii"))
    global_step = tile.global_source_step + node_index
    # The engine contract permits arbitrary nonnegative Python lattice
    # integers.  Canonical JSON avoids imposing an accidental uint64 ceiling
    # in the persistence layer.
    digest.update(_canonical_json(global_step))
    _array_digest_update(
        digest,
        "source_relative_positions_AU",
        tile.source_relative_positions_AU[node_index],
    )
    _array_digest_update(
        digest,
        "source_relative_velocities_AU_per_year",
        tile.source_relative_velocities_AU_per_year[node_index],
    )
    return digest.hexdigest()


def _vector_record(value: np.ndarray | None) -> list[str] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != (3,):
        raise RestrictedWHCheckpointContractError(
            "route vector is not an exact binary64 three-vector"
        )
    return [float(item).hex() for item in array]


def _vector_from_record(value: object, label: str) -> np.ndarray:
    if type(value) is not list or len(value) != 3:
        raise RestrictedWHCheckpointIntegrityError(
            f"{label} must be a hexadecimal binary64 three-vector"
        )
    return np.ascontiguousarray(
        np.asarray(
            [_from_hex(item, f"{label}[{index}]") for index, item in enumerate(value)],
            dtype=np.float64,
        )
    )


def _event_record(event: RestrictedWHRouteEvent) -> dict[str, Any]:
    if type(event) is not RestrictedWHRouteEvent:
        raise RestrictedWHCheckpointContractError(
            "route event must be an exact RestrictedWHRouteEvent"
        )
    return {
        "event_type": event.event_type,
        "source_index": event.source_index,
        "source_id": event.source_id,
        "map_step_fraction_hex": event.map_step_fraction.hex(),
        "source_interval_offset": event.source_interval_offset,
        "interval_fraction_hex": event.interval_fraction.hex(),
        "relative_position_AU_hex": _vector_record(event.relative_position_AU),
        "relative_velocity_AU_per_year_hex": _vector_record(
            event.relative_velocity_AU_per_year
        ),
        "surface_distance_AU_hex": event.surface_distance_AU.hex(),
        "surface_threshold_AU_hex": event.surface_threshold_AU.hex(),
        "surface_residual_AU_hex": event.surface_residual_AU.hex(),
        "surface_tolerance_AU_hex": event.surface_tolerance_AU.hex(),
        "primary_specific_energy_AU2_per_year2_hex": (
            event.primary_specific_energy_AU2_per_year2.hex()
        ),
        "primary_radial_velocity_AU_per_year_hex": (
            event.primary_radial_velocity_AU_per_year.hex()
        ),
        "surface_radial_velocity_AU_per_year_hex": (
            event.surface_radial_velocity_AU_per_year.hex()
        ),
        "location_method": event.location_method,
    }


def _event_from_record(value: object) -> RestrictedWHRouteEvent:
    if type(value) is not dict:
        raise RestrictedWHCheckpointIntegrityError("removal event row is invalid")
    try:
        return RestrictedWHRouteEvent(
            event_type=value["event_type"],
            source_index=value["source_index"],
            source_id=value["source_id"],
            map_step_fraction=_from_hex(
                value["map_step_fraction_hex"], "map_step_fraction"
            ),
            source_interval_offset=value["source_interval_offset"],
            interval_fraction=_from_hex(
                value["interval_fraction_hex"], "interval_fraction"
            ),
            relative_position_AU=_vector_from_record(
                value["relative_position_AU_hex"], "relative_position_AU"
            ),
            relative_velocity_AU_per_year=_vector_from_record(
                value["relative_velocity_AU_per_year_hex"],
                "relative_velocity_AU_per_year",
            ),
            surface_distance_AU=_from_hex(
                value["surface_distance_AU_hex"], "surface_distance_AU"
            ),
            surface_threshold_AU=_from_hex(
                value["surface_threshold_AU_hex"], "surface_threshold_AU"
            ),
            surface_residual_AU=_from_hex(
                value["surface_residual_AU_hex"], "surface_residual_AU"
            ),
            surface_tolerance_AU=_from_hex(
                value["surface_tolerance_AU_hex"], "surface_tolerance_AU"
            ),
            primary_specific_energy_AU2_per_year2=_from_hex(
                value["primary_specific_energy_AU2_per_year2_hex"],
                "primary_specific_energy_AU2_per_year2",
            ),
            primary_radial_velocity_AU_per_year=_from_hex(
                value["primary_radial_velocity_AU_per_year_hex"],
                "primary_radial_velocity_AU_per_year",
            ),
            surface_radial_velocity_AU_per_year=_from_hex(
                value["surface_radial_velocity_AU_per_year_hex"],
                "surface_radial_velocity_AU_per_year",
            ),
            location_method=value["location_method"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "removal event row is invalid"
        ) from exc


def _route_result_record(result: RestrictedWHRouteResult) -> dict[str, Any]:
    if type(result) is not RestrictedWHRouteResult:
        raise RestrictedWHCheckpointContractError(
            "route result must be an exact RestrictedWHRouteResult"
        )
    return {
        "method_id": result.method_id,
        "scientific_classification": result.scientific_classification,
        "project_scope": result.project_scope,
        "claims": dict(result.claims),
        "tracer_index": result.tracer_index,
        "source_tile_sha256": result.source_tile_sha256,
        "source_segment_sha256": result.source_segment_sha256,
        "global_source_step": result.global_source_step,
        "source_steps_per_map_step": result.source_steps_per_map_step,
        "events": [_event_record(event) for event in result.events],
        "endpoint_relative_position_AU_hex": _vector_record(
            result.endpoint_relative_position_AU
        ),
        "endpoint_relative_velocity_AU_per_year_hex": _vector_record(
            result.endpoint_relative_velocity_AU_per_year
        ),
        "accepted_step_count": result.accepted_step_count,
        "derivative_evaluation_count": result.derivative_evaluation_count,
        "minimum_evaluated_stage_distance_AU_hex": (
            result.minimum_evaluated_stage_distance_AU.hex()
        ),
        "minimum_committed_stage_distance_AU_hex": (
            result.minimum_committed_stage_distance_AU.hex()
        ),
        "maximum_accepted_primary_dynamical_phase_radians_hex": (
            result.maximum_accepted_primary_dynamical_phase_radians.hex()
        ),
        "maximum_committed_primary_dynamical_phase_radians_hex": (
            result.maximum_committed_primary_dynamical_phase_radians.hex()
        ),
        "completed_map_fraction_hex": result.completed_map_fraction.hex(),
        "runtime_identity": _evidence_tree(result.runtime_identity),
    }


def _route_record(record: CUDARestrictedWHRouteRecord) -> dict[str, Any]:
    if type(record) is not CUDARestrictedWHRouteRecord:
        raise RestrictedWHCheckpointContractError(
            "route record must be an exact CUDARestrictedWHRouteRecord"
        )
    return {
        "map_step_index": record.map_step_index,
        "lane_index": record.lane_index,
        "tracer_index": record.tracer_index,
        "guard_phase": record.guard_phase,
        "guard_reason": record.guard_reason,
        "guard_source_index": record.guard_source_index,
        "route_result": _route_result_record(record.route_result),
    }


def _removal_record(record: RestrictedWHRemovalRecord) -> dict[str, Any]:
    if type(record) is not RestrictedWHRemovalRecord:
        raise RestrictedWHCheckpointContractError(
            "removal record must be an exact RestrictedWHRemovalRecord"
        )
    return {
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
        "minimum_committed_stage_distance_AU_hex": (
            record.minimum_committed_stage_distance_AU.hex()
        ),
        "maximum_committed_primary_dynamical_phase_radians_hex": (
            record.maximum_committed_primary_dynamical_phase_radians.hex()
        ),
        "events": [_event_record(event) for event in record.events],
        "content_sha256": record.content_sha256,
    }


def _removal_from_record(value: object) -> RestrictedWHRemovalRecord:
    if type(value) is not dict:
        raise RestrictedWHCheckpointIntegrityError("removal record row is invalid")
    try:
        return RestrictedWHRemovalRecord(
            lane_index=value["lane_index"],
            tracer_index=value["tracer_index"],
            lifecycle_code=value["lifecycle_code"],
            global_source_step=value["global_source_step"],
            source_steps_per_map_step=value["source_steps_per_map_step"],
            guard_phase=value["guard_phase"],
            guard_reason=value["guard_reason"],
            guard_source_index=value["guard_source_index"],
            source_segment_sha256=value["source_segment_sha256"],
            route_runtime_identity_sha256=value[
                "route_runtime_identity_sha256"
            ],
            accepted_step_count=value["accepted_step_count"],
            derivative_evaluation_count=value["derivative_evaluation_count"],
            minimum_committed_stage_distance_AU=_from_hex(
                value["minimum_committed_stage_distance_AU_hex"],
                "minimum_committed_stage_distance_AU",
            ),
            maximum_committed_primary_dynamical_phase_radians=_from_hex(
                value[
                    "maximum_committed_primary_dynamical_phase_radians_hex"
                ],
                "maximum_committed_primary_dynamical_phase_radians",
            ),
            events=tuple(_event_from_record(event) for event in value["events"]),
            content_sha256=value["content_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "removal record row is invalid"
        ) from exc


def _backend_record(backend: BackendSpec) -> dict[str, Any]:
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise RestrictedWHCheckpointContractError(
            "checkpoint backend must be an exact CuPy BackendSpec"
        )
    resolved = resolve_backend(backend)
    return {
        "backend_id": backend.backend_id,
        "device": resolved.device,
        "tile_size": backend.tile_size,
        "dtype": backend.dtype,
        "allow_fallback": backend.allow_fallback,
        "deterministic_reductions": backend.deterministic_reductions,
        "fast_math": backend.fast_math,
        "determinism_scope": backend.determinism_scope,
    }


def _spec_record(spec: FixedStepCUDARestrictedWHSpec) -> dict[str, Any]:
    if type(spec) is not FixedStepCUDARestrictedWHSpec:
        raise RestrictedWHCheckpointContractError(
            "spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    return {
        "map_step_count": spec.map_step_count,
        "source_steps_per_map_step": spec.source_steps_per_map_step,
        "backend": _backend_record(spec.backend),
        "maximum_primary_dynamical_phase_hex": (
            float(spec.maximum_primary_dynamical_phase).hex()
        ),
        "threads_per_block": spec.threads_per_block,
        "primary_outer_boundary_AU_hex": (
            None
            if spec.primary_outer_boundary_AU is None
            else float(spec.primary_outer_boundary_AU).hex()
        ),
    }


def _route_spec_record(spec: RestrictedWHRouteSpec) -> dict[str, Any]:
    if type(spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHCheckpointContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    return dict(_plain_tree(restricted_wh_route_policy_identity(spec)))


def _route_spec_from_record(value: object) -> RestrictedWHRouteSpec:
    if type(value) is not dict:
        raise RestrictedWHCheckpointIntegrityError("route policy record is invalid")
    try:
        event = value["event_locator"]
        spec = RestrictedWHRouteSpec(
            event_locator=DOP853EventLocatorSpec(
                relative_tolerance=_from_hex(
                    event["relative_tolerance_hex"], "relative_tolerance"
                ),
                absolute_tolerance=_from_hex(
                    event["absolute_tolerance_hex"], "absolute_tolerance"
                ),
                maximum_substep_fraction=_from_hex(
                    event["maximum_substep_fraction_hex"],
                    "maximum_substep_fraction",
                ),
                dense_polynomial_validation_points=event[
                    "dense_polynomial_validation_points"
                ],
                event_fraction_tolerance=_from_hex(
                    event["event_fraction_tolerance_hex"],
                    "event_fraction_tolerance",
                ),
                event_distance_tolerance_AU=_from_hex(
                    event["event_distance_tolerance_AU_hex"],
                    "event_distance_tolerance_AU",
                ),
                outward_radial_velocity_tolerance_AU_per_year=_from_hex(
                    event[
                        "outward_radial_velocity_tolerance_AU_per_year_hex"
                    ],
                    "outward_radial_velocity_tolerance_AU_per_year",
                ),
                maximum_accepted_steps=event["maximum_accepted_steps"],
            ),
            maximum_primary_dynamical_phase=_from_hex(
                value["maximum_primary_dynamical_phase_hex"],
                "maximum_primary_dynamical_phase",
            ),
            primary_outer_boundary_AU=(
                None
                if value["primary_outer_boundary_AU_hex"] is None
                else _from_hex(
                    value["primary_outer_boundary_AU_hex"],
                    "primary_outer_boundary_AU",
                )
            ),
            terminal_source_radii_AU=(
                None
                if value["terminal_source_radii_AU_hex"] is None
                else tuple(
                    _from_hex(item, "terminal_source_radius_AU")
                    for item in value["terminal_source_radii_AU_hex"]
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "route policy record is invalid"
        ) from exc
    if not _canonical_tree_equal(_route_spec_record(spec), value):
        raise RestrictedWHCheckpointIntegrityError(
            "route policy content hash or canonical form failed"
        )
    return spec


def checkpoint_implementation_identity() -> Mapping[str, Any]:
    """Return the frozen local code/kernel identity bound by each checkpoint."""

    return _freeze_tree(
        {
            "checkpoint_method_id": CHECKPOINT_METHOD_ID,
            "checkpoint_implementation_source_sha256": (
                IMPLEMENTATION_SOURCE_SHA256
            ),
            "lifecycle_method_id": LIFECYCLE_METHOD_ID,
            "lifecycle_implementation_source_sha256": (
                _lifecycle_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "hybrid_method_id": _hybrid_module.HYBRID_METHOD_ID,
            "hybrid_implementation_source_sha256": (
                _hybrid_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "route_method_id": ROUTE_METHOD_ID,
            "route_implementation_source_sha256": (
                _route_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "event_locator_source_sha256": EVENT_LOCATOR_SOURCE_SHA256,
            "restricted_wh_method_id": _restricted_module.METHOD_ID,
            "restricted_wh_implementation_source_sha256": (
                _restricted_module.IMPLEMENTATION_SOURCE_SHA256
            ),
            "restricted_wh_kick_cuda_source_sha256": (
                _restricted_module.KICK_CUDA_SOURCE_SHA256
            ),
            "restricted_wh_fused_cuda_source_sha256": (
                _restricted_module.FUSED_CUDA_SOURCE_SHA256
            ),
            "all_conic_implementation_source_sha256": (
                ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256
            ),
            "all_conic_device_source_sha256": ALL_CONIC_DEVICE_SOURCE_SHA256,
            "backend_implementation_source_sha256": (
                BACKENDS_IMPLEMENTATION_SOURCE_SHA256
            ),
            "contracts_implementation_source_sha256": (
                CONTRACTS_IMPLEMENTATION_SOURCE_SHA256
            ),
            "wisdom_holman_contracts_source_sha256": (
                WH_CONTRACTS_IMPLEMENTATION_SOURCE_SHA256
            ),
            "cuda_options": list(_all_conic_module.CUDA_OPTIONS),
        }
    )


def _cpu_runtime_identity() -> dict[str, Any]:
    selected: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            first = cpuinfo.read_text(encoding="utf-8").split("\n\n", 1)[0]
        except (OSError, UnicodeError) as exc:
            raise RestrictedWHCheckpointContractError(
                "CPU runtime identity is unreadable"
            ) from exc
        fields = {
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "microcode",
            "flags",
        }
        for line in first.splitlines():
            key, separator, value = line.partition(":")
            key = key.strip()
            if separator and key in fields:
                selected[key] = " ".join(value.split())
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(int(item) for item in os.sched_getaffinity(0))
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": list(platform.libc_ver()),
        "cpuinfo": selected,
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "numerical_thread_environment": thread_environment,
    }


def checkpoint_runtime_identity(backend: BackendSpec) -> Mapping[str, Any]:
    """Return the exact host/CUDA runtime identity required for replay."""

    resolved = resolve_backend(backend)
    if resolved.name != "cupy":
        raise RestrictedWHCheckpointContractError(
            "checkpoint runtime identity requires a CuPy backend"
        )
    cp = resolved.xp
    try:
        import scipy

        with resolved.activate():
            device_index = int(cp.cuda.Device().id)
            properties = cp.cuda.runtime.getDeviceProperties(device_index)
            raw_name = properties["name"]
            device_name = (
                raw_name.decode("ascii")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            identity = {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_compiler": platform.python_compiler(),
                "python_build": list(platform.python_build()),
                "python_executable": str(Path(sys.executable).resolve()),
                "python_executable_sha256": _sha256_file(
                    Path(sys.executable).resolve()
                ),
                "cpu_runtime": _cpu_runtime_identity(),
                "numpy_version": np.__version__,
                "numpy_build": _evidence_tree(np.show_config(mode="dicts")),
                "scipy_version": scipy.__version__,
                "scipy_build": _evidence_tree(scipy.show_config(mode="dicts")),
                "cupy_version": str(cp.__version__),
                "device": resolved.device,
                "device_index": device_index,
                "device_name": device_name,
                "device_uuid": bytes(properties["uuid"]).hex(),
                "device_compute_capability": (
                    f"{int(properties['major'])}.{int(properties['minor'])}"
                ),
                "device_pci_domain_bus_device": (
                    f"{int(properties['pciDomainID']):04x}:"
                    f"{int(properties['pciBusID']):02x}:"
                    f"{int(properties['pciDeviceID']):02x}"
                ),
                "cuda_runtime_version": int(
                    cp.cuda.runtime.runtimeGetVersion()
                ),
                "cuda_local_runtime_version": int(
                    cp.cuda.get_local_runtime_version()
                ),
                "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
                "nvrtc_version": [
                    int(item) for item in cp.cuda.nvrtc.getVersion()
                ],
                "determinism_scope": (
                    "SAME_PYTHON_NUMPY_SCIPY_CPU_CUPY_CUDA_RUNTIME_AND_DEVICE"
                ),
            }
    except Exception as exc:
        raise RestrictedWHCheckpointContractError(
            "checkpoint runtime identity is unavailable"
        ) from exc
    return _freeze_tree(identity)


def restricted_wh_lifecycle_state_sha256(
    state: RestrictedWHLifecycleState,
    backend: BackendSpec,
) -> str:
    """Hash one exact lifecycle state for a protocol genesis commitment."""

    if type(state) is not RestrictedWHLifecycleState:
        raise RestrictedWHCheckpointContractError(
            "state must be an exact RestrictedWHLifecycleState"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        arrays = _state_arrays(state, cp, prefix="state")
        cp.cuda.get_current_stream().synchronize()
    removals = [_removal_record(row) for row in state.removal_records]
    return _state_sha256(
        _state_metadata(state), arrays, removals, prefix="state"
    )


def _checkpoint_namespace(directory: Path, run_id: str, protocol_sha256: str) -> Path:
    digest = hashlib.sha256(
        CHECKPOINT_NAMESPACE_DOMAIN
        + run_id.encode("utf-8")
        + b"\0"
        + protocol_sha256.encode("ascii")
    ).hexdigest()
    return directory / f"run-{digest}"


def _record_path(namespace: Path, index: int) -> Path:
    return namespace / f"checkpoint_{index:020d}.json"


def _commit_create_only(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite committed artifact: {destination}"
        ) from None
    else:
        temporary.unlink()
        descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint record: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_root(directory: Path, *, create: bool) -> None:
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint root exists but is not a real directory"
            )
        return
    if create:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise RestrictedWHCheckpointIntegrityError(
                    "checkpoint root exists but is not a real directory"
                )


def _ensure_namespace(directory: Path, namespace: Path) -> None:
    _validate_root(directory, create=True)
    if namespace.exists():
        if namespace.is_symlink() or not namespace.is_dir():
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint namespace exists but is not a real directory"
            )
        return
    try:
        namespace.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        if namespace.is_symlink() or not namespace.is_dir():
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint namespace exists but is not a real directory"
            )


@contextmanager
def _checkpoint_append_lock(namespace: Path):
    """Serialize the complete verify-and-publish transaction per namespace."""

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        directory_descriptor = os.open(namespace, directory_flags)
        lock_descriptor = os.open(
            ".append.lock",
            lock_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint append lock could not be acquired"
        ) from exc

    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint append lock is not a regular file"
            )
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint append lock could not be acquired"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)
            os.close(directory_descriptor)


def _load_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint record is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            f"checkpoint record is unreadable: {path}"
        ) from exc
    if type(value) is not dict:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint record must be a JSON object"
        )
    recorded = value.pop("checkpoint_content_sha256", None)
    expected = hashlib.sha256(CHECKPOINT_DOMAIN + _canonical_json(value)).hexdigest()
    if recorded != expected:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint record content hash failed"
        )
    value["checkpoint_content_sha256"] = recorded
    if set(value) != _RECORD_KEYS:
        raise RestrictedWHCheckpointIntegrityError(
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
        raise RestrictedWHCheckpointIntegrityError(
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
                    raise RestrictedWHCheckpointIntegrityError(
                        "checkpoint append lock artifact is invalid"
                    )
            elif not allowed_temporary or path.is_symlink() or not path.is_file():
                raise RestrictedWHCheckpointIntegrityError(
                    f"unexpected hidden checkpoint artifact: {path.name}"
                )
            continue
        record_match = _RECORD_PATTERN.fullmatch(path.name)
        array_match = _ARRAY_PATTERN.fullmatch(path.name)
        if record_match is None and array_match is None:
            raise RestrictedWHCheckpointIntegrityError(
                f"unexpected checkpoint namespace artifact: {path.name}"
            )
        if record_match is None:
            if path.is_symlink() or not path.is_file():
                raise RestrictedWHCheckpointIntegrityError(
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
        observed_identity = {
            key: record.get(key) for key in expected_identity
        }
        if (
            not _canonical_tree_equal(observed_identity, expected_identity)
            or _record_path(namespace, index) != path
        ):
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint identity or canonical path failed"
            )
        records.append((index, path, record))
    records.sort(key=lambda item: item[0])
    predecessor: str | None = None
    previous_index = -1
    for index, _, record in records:
        if index <= previous_index:
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint indices are duplicated or nonmonotonic"
            )
        if record.get("previous_checkpoint_content_sha256") != predecessor:
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint predecessor hash chain failed"
            )
        previous_index = index
        predecessor = record["checkpoint_content_sha256"]
    if records and records[0][0] != 0:
        raise RestrictedWHCheckpointIntegrityError(
            "a checkpoint chain must begin at index zero"
        )
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
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint array path identity failed"
        )
    path = record_path.parent / name
    if path.is_symlink() or not path.is_file():
        raise RestrictedWHCheckpointIntegrityError(
            "committed checkpoint is missing its regular array file"
        )
    recorded_size = record.get("array_file_size_bytes")
    if (
        type(recorded_size) is not int
        or recorded_size <= 0
        or path.stat().st_size != recorded_size
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint array size failed"
        )
    if _sha256_file(path) != digest:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint array SHA-256 failed"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(_ARRAY_KEYS)
                or set(archive.files) != _ARRAY_KEYS
            ):
                raise RestrictedWHCheckpointIntegrityError(
                    "checkpoint array member roster changed"
                )
            arrays = {
                key: np.ascontiguousarray(np.array(archive[key], copy=True))
                for key in _ARRAY_KEYS
            }
    except RestrictedWHCheckpointIntegrityError:
        raise
    except (OSError, ValueError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint array archive is unreadable"
        ) from exc
    return arrays


def _state_from_storage(
    metadata: object,
    arrays: Mapping[str, np.ndarray],
    removal_rows: object,
    *,
    prefix: str,
    backend: BackendSpec,
) -> RestrictedWHLifecycleState:
    if type(metadata) is not dict or type(removal_rows) is not list:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint state metadata or removal ledger is invalid"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    try:
        provenance = _provenance_from_record(metadata["provenance"], prefix)
        removals = tuple(_removal_from_record(row) for row in removal_rows)
        with resolved.activate():
            dynamics = CUDARestrictedWHState(
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
            state = RestrictedWHLifecycleState(
                dynamics_state=dynamics,
                tracer_lifecycle=arrays[f"{prefix}_tracer_lifecycle"],
                removal_records=removals,
            )
            cp.cuda.get_current_stream().synchronize()
            return state
    except RestrictedWHCheckpointIntegrityError:
        raise
    except Exception as exc:
        raise RestrictedWHCheckpointIntegrityError(
            f"stored {prefix} state cannot be reconstructed"
        ) from exc


def _tile_from_storage(
    metadata: object, arrays: Mapping[str, np.ndarray]
) -> FrozenSourcePathTile:
    if type(metadata) is not dict:
        raise RestrictedWHCheckpointIntegrityError(
            "source tile metadata is invalid"
        )
    try:
        tile = FrozenSourcePathTile(
            tile_id=metadata["tile_id"],
            epoch_origin_years=_from_hex(
                metadata["epoch_origin_years_hex"],
                "tile.epoch_origin_years",
            ),
            global_source_step=metadata["global_source_step"],
            source_step_years=_from_hex(
                metadata["source_step_years_hex"], "tile.source_step_years"
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
            source_relative_positions_AU=arrays[
                "source_relative_positions_AU"
            ],
            source_relative_velocities_AU_per_year=arrays[
                "source_relative_velocities_AU_per_year"
            ],
            source_gm_AU3_per_year2=arrays["source_gm_AU3_per_year2"],
            source_encounter_radii_AU=arrays[
                "source_encounter_radii_AU"
            ],
            certified_relative_acceleration_bounds_AU_per_year2=arrays[
                "certified_relative_acceleration_bounds_AU_per_year2"
            ],
            provenance=_provenance_from_record(metadata["provenance"], "tile"),
            content_sha256=metadata["content_sha256"],
        )
        if not _canonical_tree_equal(_tile_metadata(tile), metadata):
            raise RestrictedWHCheckpointIntegrityError(
                "source tile metadata canonical form failed"
            )
        return tile
    except RestrictedWHCheckpointIntegrityError:
        raise
    except Exception as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "stored source tile cannot be reconstructed"
        ) from exc


def _spec_from_record(
    value: object, requested_backend: BackendSpec
) -> FixedStepCUDARestrictedWHSpec:
    if type(value) is not dict:
        raise RestrictedWHCheckpointIntegrityError("map spec record is invalid")
    if not _canonical_tree_equal(
        value.get("backend"), _backend_record(requested_backend)
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint backend/device policy differs from the requested runtime"
        )
    try:
        spec = FixedStepCUDARestrictedWHSpec(
            map_step_count=value["map_step_count"],
            source_steps_per_map_step=value["source_steps_per_map_step"],
            backend=requested_backend,
            maximum_primary_dynamical_phase=_from_hex(
                value["maximum_primary_dynamical_phase_hex"],
                "maximum_primary_dynamical_phase",
            ),
            threads_per_block=value["threads_per_block"],
            primary_outer_boundary_AU=(
                None
                if value["primary_outer_boundary_AU_hex"] is None
                else _from_hex(
                    value["primary_outer_boundary_AU_hex"],
                    "primary_outer_boundary_AU",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RestrictedWHCheckpointIntegrityError(
            "map spec record is invalid"
        ) from exc
    if not _canonical_tree_equal(_spec_record(spec), value):
        raise RestrictedWHCheckpointIntegrityError(
            "map spec canonical form failed"
        )
    return spec


def _result_semantics(
    result: RestrictedWHLifecycleResult, cp: Any
) -> dict[str, Any]:
    if type(result) is not RestrictedWHLifecycleResult:
        raise RestrictedWHCheckpointContractError(
            "result must be an exact RestrictedWHLifecycleResult"
        )
    minimum = np.ascontiguousarray(cp.asnumpy(result.minimum_source_distance_AU))
    minimum_digest = hashlib.sha256()
    _array_digest_update(minimum_digest, "minimum_source_distance_AU", minimum)
    return {
        "method_id": result.method_id,
        "scientific_classification": result.scientific_classification,
        "project_scope": result.project_scope,
        "claims": dict(result.claims),
        "source_tile_sha256": result.source_tile_sha256,
        "maximum_kepler_iterations": result.maximum_kepler_iterations,
        "minimum_source_distance_array_sha256": minimum_digest.hexdigest(),
        "route_records": [_route_record(row) for row in result.route_records],
        "new_removal_records": [
            _removal_record(row) for row in result.new_removal_records
        ],
        "runtime_identity": _evidence_tree(result.runtime_identity),
    }


def _segment_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(SEGMENT_DOMAIN + _canonical_json(dict(record))).hexdigest()


def _verify_result_against_replay(
    input_state: RestrictedWHLifecycleState,
    expected_result: RestrictedWHLifecycleResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    route_worker_count: int,
) -> RestrictedWHLifecycleResult:
    replay = advance_cuda_restricted_wh_lifecycle(
        input_state,
        source_tile,
        spec,
        route_spec=route_spec,
        route_worker_count=route_worker_count,
    )
    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    with resolved.activate():
        expected_arrays = _state_arrays(expected_result.state, cp, prefix="output")
        replay_arrays = _state_arrays(replay.state, cp, prefix="output")
        expected_metadata = _state_metadata(expected_result.state)
        replay_metadata = _state_metadata(replay.state)
        expected_removals = [
            _removal_record(row) for row in expected_result.state.removal_records
        ]
        replay_removals = [
            _removal_record(row) for row in replay.state.removal_records
        ]
        expected_state_hash = _state_sha256(
            expected_metadata,
            expected_arrays,
            expected_removals,
            prefix="output",
        )
        replay_state_hash = _state_sha256(
            replay_metadata,
            replay_arrays,
            replay_removals,
            prefix="output",
        )
        if expected_state_hash != replay_state_hash:
            raise RestrictedWHCheckpointContractError(
                "checkpoint result output differs from deterministic replay"
            )
        if _result_semantics(expected_result, cp) != _result_semantics(replay, cp):
            raise RestrictedWHCheckpointContractError(
                "checkpoint result diagnostics or ledgers differ from replay"
            )
        cp.cuda.get_current_stream().synchronize()
    return replay


def _validate_segment_shape(
    input_state: RestrictedWHLifecycleState,
    result: RestrictedWHLifecycleResult,
    tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec,
    route_worker_count: int,
) -> None:
    if type(input_state) is not RestrictedWHLifecycleState:
        raise RestrictedWHCheckpointContractError(
            "input_state must be an exact RestrictedWHLifecycleState"
        )
    if type(result) is not RestrictedWHLifecycleResult:
        raise RestrictedWHCheckpointContractError(
            "result must be an exact RestrictedWHLifecycleResult"
        )
    if type(tile) is not FrozenSourcePathTile:
        raise RestrictedWHCheckpointContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    if type(spec) is not FixedStepCUDARestrictedWHSpec:
        raise RestrictedWHCheckpointContractError(
            "spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    if type(route_spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHCheckpointContractError(
            "route_spec must be an exact RestrictedWHRouteSpec"
        )
    if type(route_worker_count) is not int or not 1 <= route_worker_count <= 32:
        raise RestrictedWHCheckpointContractError(
            "route_worker_count must be an integer in [1, 32]"
        )
    required_intervals = spec.map_step_count * spec.source_steps_per_map_step
    if tile.node_count != required_intervals + 1:
        raise RestrictedWHCheckpointContractError(
            "a checkpoint source tile must contain exactly one segment"
        )
    if tile.global_source_step != input_state.dynamics_state.global_source_step:
        raise RestrictedWHCheckpointContractError(
            "checkpoint source tile does not start at its input state"
        )
    expected_output_step = (
        input_state.dynamics_state.global_source_step + required_intervals
    )
    if result.state.dynamics_state.global_source_step != expected_output_step:
        raise RestrictedWHCheckpointContractError(
            "checkpoint output is not at the exact segment lattice endpoint"
        )
    if result.source_tile_sha256 != tile.content_sha256:
        raise RestrictedWHCheckpointContractError(
            "checkpoint result names a different source tile"
        )
    if result.method_id != LIFECYCLE_METHOD_ID or not _canonical_tree_equal(
        dict(result.claims), dict(LIFECYCLE_CLAIMS)
    ):
        raise RestrictedWHCheckpointContractError(
            "checkpoint result lifecycle identity changed"
        )
    if (
        route_spec.maximum_primary_dynamical_phase
        != spec.maximum_primary_dynamical_phase
        or route_spec.primary_outer_boundary_AU
        != spec.primary_outer_boundary_AU
    ):
        raise RestrictedWHCheckpointContractError(
            "CUDA and route policies must match exactly"
        )
    prior = input_state.removal_records
    if result.state.removal_records[: len(prior)] != prior:
        raise RestrictedWHCheckpointContractError(
            "checkpoint result removal history is not an exact prefix extension"
        )
    if result.state.removal_records[len(prior) :] != result.new_removal_records:
        raise RestrictedWHCheckpointContractError(
            "checkpoint new-removal delta disagrees with output history"
        )


def dump_restricted_wh_lifecycle_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    checkpoint_index: int,
    input_state: RestrictedWHLifecycleState,
    result: RestrictedWHLifecycleResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
    previous_checkpoint_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Replay, verify, and publish one namespace-serialized checkpoint."""

    root = Path(directory)
    _text(run_id, "run_id")
    _sha256(protocol_sha256, "protocol_sha256")
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    namespace = _checkpoint_namespace(root, run_id, protocol_sha256)
    _ensure_namespace(root, namespace)
    with _checkpoint_append_lock(namespace):
        return _dump_restricted_wh_lifecycle_checkpoint_locked(
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
            previous_checkpoint_content_sha256=(
                previous_checkpoint_content_sha256
            ),
        )


def _dump_restricted_wh_lifecycle_checkpoint_locked(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    checkpoint_index: int,
    input_state: RestrictedWHLifecycleState,
    result: RestrictedWHLifecycleResult,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    route_spec: RestrictedWHRouteSpec | None = None,
    route_worker_count: int = DEFAULT_ROUTE_WORKER_COUNT,
    previous_checkpoint_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Publish one immutable segment only after exact deterministic replay."""

    root = Path(directory)
    _text(run_id, "run_id")
    _sha256(protocol_sha256, "protocol_sha256")
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    if (
        type(checkpoint_index) is not int
        or not 0 <= checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
    ):
        raise RestrictedWHCheckpointContractError(
            "checkpoint_index must be an integer in [0, 10**20)"
        )
    if previous_checkpoint_content_sha256 is not None:
        _sha256(
            previous_checkpoint_content_sha256,
            "previous_checkpoint_content_sha256",
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
    )
    replay = _verify_result_against_replay(
        input_state,
        result,
        source_tile,
        spec,
        route_spec,
        route_worker_count,
    )
    if replay.state is result.state:
        raise AssertionError("deterministic replay unexpectedly reused result state")

    records = _scan_records(root, run_id, protocol_sha256) if root.exists() else ()
    verified_predecessor: LoadedRestrictedWHCheckpoint | None = None
    if records:
        # Appending is an authorization decision.  Do not trust the predecessor
        # JSON hash alone: verify its complete history and deterministic replay
        # before allowing a new record to extend the chain.
        verified_predecessor = load_latest_restricted_wh_lifecycle_checkpoint(
            root,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            genesis_state_sha256=genesis_state_sha256,
            expected_checkpoint_index=records[-1][0],
            expected_checkpoint_content_sha256=records[-1][2][
                "checkpoint_content_sha256"
            ],
            backend=spec.backend,
        )
        if (
            verified_predecessor is None
            or verified_predecessor.checkpoint_index != records[-1][0]
            or verified_predecessor.checkpoint_content_sha256
            != records[-1][2]["checkpoint_content_sha256"]
        ):
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint predecessor verification did not yield the latest record"
            )
        latest_index, _, latest = records[-1]
        if checkpoint_index <= latest_index:
            raise RestrictedWHCheckpointContractError(
                "checkpoint indices must increase monotonically"
            )
        if previous_checkpoint_content_sha256 != latest[
            "checkpoint_content_sha256"
        ]:
            raise RestrictedWHCheckpointContractError(
                "predecessor digest must name the latest checkpoint"
            )
    else:
        if checkpoint_index != 0:
            raise RestrictedWHCheckpointContractError(
                "the first checkpoint index must be zero"
            )
        if previous_checkpoint_content_sha256 is not None:
            raise RestrictedWHCheckpointContractError(
                "the first checkpoint cannot name a predecessor"
            )

    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    with resolved.activate():
        input_arrays = _state_arrays(input_state, cp, prefix="input")
        output_arrays = _state_arrays(result.state, cp, prefix="output")
        minimum = np.ascontiguousarray(
            cp.asnumpy(result.minimum_source_distance_AU)
        )
        cp.cuda.get_current_stream().synchronize()
    arrays = {
        **input_arrays,
        **output_arrays,
        **_tile_arrays(source_tile),
        "minimum_source_distance_AU": minimum,
    }
    if set(arrays) != _ARRAY_KEYS:
        raise AssertionError("internal checkpoint array roster changed")

    input_metadata = _state_metadata(input_state)
    output_metadata = _state_metadata(result.state)
    input_removals = [_removal_record(row) for row in input_state.removal_records]
    output_removals = [
        _removal_record(row) for row in result.state.removal_records
    ]
    input_state_sha256 = _state_sha256(
        input_metadata, arrays, input_removals, prefix="input"
    )
    output_state_sha256 = _state_sha256(
        output_metadata, arrays, output_removals, prefix="output"
    )
    lineage_sha256 = _lineage_sha256(
        input_metadata, arrays, prefix="input"
    )
    source_lineage_sha256 = _source_lineage_sha256(source_tile)
    source_initial_node_sha256 = _source_boundary_sha256(source_tile, 0)
    source_final_node_sha256 = _source_boundary_sha256(
        source_tile, source_tile.node_count - 1
    )
    if _lineage_sha256(output_metadata, arrays, prefix="output") != lineage_sha256:
        raise RestrictedWHCheckpointContractError(
            "checkpoint segment changed immutable state lineage"
        )
    if records:
        latest = records[-1][2]
        if verified_predecessor is None:
            raise AssertionError("verified checkpoint predecessor is missing")
        if input_state_sha256 != latest.get("state_sha256"):
            raise RestrictedWHCheckpointContractError(
                "checkpoint input is not the exact predecessor output"
            )
        if input_state_sha256 != verified_predecessor.state_sha256:
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint input differs from the replayed predecessor"
            )
        if lineage_sha256 != latest.get("lineage_sha256"):
            raise RestrictedWHCheckpointContractError(
                "checkpoint successor changed immutable state lineage"
            )
        if genesis_state_sha256 != latest.get("genesis_state_sha256"):
            raise RestrictedWHCheckpointContractError(
                "checkpoint successor changed the protocol genesis commitment"
            )
        if source_lineage_sha256 != latest.get("source_lineage_sha256"):
            raise RestrictedWHCheckpointContractError(
                "checkpoint successor changed immutable source lineage"
            )
        if source_initial_node_sha256 != latest.get(
            "source_final_node_sha256"
        ):
            raise RestrictedWHCheckpointContractError(
                "checkpoint successor source tile does not join its predecessor"
            )
    elif input_state_sha256 != genesis_state_sha256:
        raise RestrictedWHCheckpointContractError(
            "first checkpoint input differs from the protocol genesis commitment"
        )

    implementation_identity = dict(_plain_tree(checkpoint_implementation_identity()))
    result_semantics = _result_semantics(result, cp)
    segment_record = {
        "input_state_sha256": input_state_sha256,
        "state_sha256": output_state_sha256,
        "lineage_sha256": lineage_sha256,
        "source_tile_sha256": source_tile.content_sha256,
        "spec": _spec_record(spec),
        "route_policy": _route_spec_record(route_spec),
        "route_worker_count": route_worker_count,
        "result": result_semantics,
    }
    segment_sha256 = _segment_sha256(segment_record)

    namespace = _checkpoint_namespace(root, run_id, protocol_sha256)
    _ensure_namespace(root, namespace)
    record_path = _record_path(namespace, checkpoint_index)
    if record_path.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint {checkpoint_index}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".checkpoint_{checkpoint_index:020d}.",
        suffix=".npz",
        dir=namespace,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_sha256 = _sha256_file(temporary)
        array_path = namespace / (
            f"checkpoint_{checkpoint_index:020d}.array-{array_sha256}.npz"
        )
        if array_path.exists():
            if array_path.is_symlink() or not array_path.is_file():
                raise RestrictedWHCheckpointIntegrityError(
                    "content-addressed array path is not a regular file"
                )
            if _sha256_file(array_path) != array_sha256:
                raise RestrictedWHCheckpointIntegrityError(
                    "content-addressed array collision"
                )
            temporary.unlink()
        else:
            _commit_create_only(temporary, array_path)
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
        "checkpoint_namespace": namespace.name,
        "checkpoint_index": checkpoint_index,
        "previous_checkpoint_content_sha256": (
            previous_checkpoint_content_sha256
        ),
        "array_file": array_path.name,
        "array_file_size_bytes": array_path.stat().st_size,
        "array_file_sha256": array_sha256,
        "input_state_metadata": input_metadata,
        "state_metadata": output_metadata,
        "source_tile_metadata": _tile_metadata(source_tile),
        # A successor derives its cumulative input ledger from the verified
        # predecessor.  Only the first record carries a genesis ledger; every
        # segment's new removals live once inside ``result`` below.
        "genesis_removal_records": input_removals if not records else None,
        "input_state_sha256": input_state_sha256,
        "state_sha256": output_state_sha256,
        "lineage_sha256": lineage_sha256,
        "source_lineage_sha256": source_lineage_sha256,
        "source_initial_node_sha256": source_initial_node_sha256,
        "source_final_node_sha256": source_final_node_sha256,
        "segment_sha256": segment_sha256,
        "spec": segment_record["spec"],
        "route_policy": segment_record["route_policy"],
        "route_worker_count": route_worker_count,
        "result": result_semantics,
        "implementation_identity": implementation_identity,
        "checkpoint_runtime_identity": dict(
            _plain_tree(checkpoint_runtime_identity(spec.backend))
        ),
        "claim_ceiling": {
            "scientific_classification": SCIENTIFIC_CLASSIFICATION,
            "project_scope": PROJECT_SCOPE,
            "claims": dict(CHECKPOINT_CLAIMS),
        },
    }
    record["checkpoint_content_sha256"] = hashlib.sha256(
        CHECKPOINT_DOMAIN + _canonical_json(record)
    ).hexdigest()
    _atomic_json_create(record_path, record)
    return _freeze_tree(record)


def _verify_record_static(
    record: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    genesis_state_sha256: str,
) -> None:
    if not _canonical_tree_equal(
        record.get("implementation_identity"), dict(implementation_identity)
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint implementation/kernel identity changed"
        )
    expected_claims = {
        "scientific_classification": SCIENTIFIC_CLASSIFICATION,
        "project_scope": PROJECT_SCOPE,
        "claims": dict(CHECKPOINT_CLAIMS),
    }
    if not _canonical_tree_equal(record.get("claim_ceiling"), expected_claims):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint claim ceiling changed"
        )
    if not _canonical_tree_equal(
        record.get("checkpoint_runtime_identity"), dict(runtime_identity)
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint host/CUDA runtime identity changed"
        )
    if record.get("genesis_state_sha256") != genesis_state_sha256:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint protocol genesis commitment changed"
        )
    worker_count = record.get("route_worker_count")
    if type(worker_count) is not int or not 1 <= worker_count <= 32:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint route worker policy is invalid"
        )


def _verify_loaded_segment(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    backend: BackendSpec,
    genesis_state_sha256: str,
    predecessor_state_sha256: str | None,
    predecessor_lineage_sha256: str | None,
    predecessor_removal_rows: list[dict[str, Any]] | None,
    predecessor_source_final_node_sha256: str | None,
    predecessor_source_lineage_sha256: str | None,
) -> tuple[
    RestrictedWHLifecycleState,
    str,
    str,
    list[dict[str, Any]],
    str,
    str,
]:
    input_metadata = record.get("input_state_metadata")
    output_metadata = record.get("state_metadata")
    result_record = record.get("result")
    if (
        type(input_metadata) is not dict
        or type(output_metadata) is not dict
        or type(result_record) is not dict
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint state metadata or removal history is invalid"
        )
    new_removals = result_record.get("new_removal_records")
    if type(new_removals) is not list:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint segment removal delta is invalid"
        )
    genesis_removals = record.get("genesis_removal_records")
    if predecessor_removal_rows is None:
        if type(genesis_removals) is not list:
            raise RestrictedWHCheckpointIntegrityError(
                "first checkpoint lacks its genesis removal ledger"
            )
        input_removals = genesis_removals
    else:
        if genesis_removals is not None:
            raise RestrictedWHCheckpointIntegrityError(
                "a checkpoint successor repeated or replaced the genesis ledger"
            )
        input_removals = predecessor_removal_rows
    output_removals = input_removals + new_removals
    minimum = arrays["minimum_source_distance_AU"]
    output_positions = arrays["output_tracer_relative_positions_AU"]
    if (
        minimum.dtype != np.dtype(np.float64)
        or minimum.ndim != 1
        or output_positions.ndim != 2
        or output_positions.shape[1:] != (3,)
        or minimum.shape != (output_positions.shape[0],)
        or not minimum.flags.c_contiguous
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint minimum-distance array has the wrong binary64 shape"
        )
    minimum_digest = hashlib.sha256()
    _array_digest_update(
        minimum_digest, "minimum_source_distance_AU", minimum
    )
    if minimum_digest.hexdigest() != result_record.get(
        "minimum_source_distance_array_sha256"
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint minimum-distance semantic hash failed"
        )
    input_hash = _state_sha256(
        input_metadata, arrays, input_removals, prefix="input"
    )
    output_hash = _state_sha256(
        output_metadata, arrays, output_removals, prefix="output"
    )
    lineage_hash = _lineage_sha256(input_metadata, arrays, prefix="input")
    if (
        input_hash != record.get("input_state_sha256")
        or output_hash != record.get("state_sha256")
        or lineage_hash != record.get("lineage_sha256")
        or _lineage_sha256(output_metadata, arrays, prefix="output")
        != lineage_hash
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint semantic state or lineage hash failed"
        )
    if (
        predecessor_state_sha256 is None
        and input_hash != genesis_state_sha256
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "first checkpoint input differs from the required genesis state"
        )
    if predecessor_state_sha256 is not None and input_hash != predecessor_state_sha256:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint successor input is not the predecessor output"
        )
    if (
        predecessor_lineage_sha256 is not None
        and lineage_hash != predecessor_lineage_sha256
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint successor changed immutable lineage"
        )

    input_state = _state_from_storage(
        input_metadata,
        arrays,
        input_removals,
        prefix="input",
        backend=backend,
    )
    stored_output = _state_from_storage(
        output_metadata,
        arrays,
        output_removals,
        prefix="output",
        backend=backend,
    )
    tile = _tile_from_storage(record.get("source_tile_metadata"), arrays)
    source_lineage_sha256 = _source_lineage_sha256(tile)
    source_initial_node_sha256 = _source_boundary_sha256(tile, 0)
    source_final_node_sha256 = _source_boundary_sha256(
        tile, tile.node_count - 1
    )
    if (
        source_lineage_sha256 != record.get("source_lineage_sha256")
        or source_initial_node_sha256
        != record.get("source_initial_node_sha256")
        or source_final_node_sha256 != record.get("source_final_node_sha256")
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint source lineage or boundary digest failed"
        )
    if (
        predecessor_source_lineage_sha256 is not None
        and source_lineage_sha256 != predecessor_source_lineage_sha256
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint successor changed immutable source lineage"
        )
    if (
        predecessor_source_final_node_sha256 is not None
        and source_initial_node_sha256
        != predecessor_source_final_node_sha256
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint successor source tile is discontinuous"
        )
    spec = _spec_from_record(record.get("spec"), backend)
    route_spec = _route_spec_from_record(record.get("route_policy"))
    worker_count = record["route_worker_count"]
    required_intervals = spec.map_step_count * spec.source_steps_per_map_step
    if tile.node_count != required_intervals + 1:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint tile contains extra or missing segment nodes"
        )
    replay = advance_cuda_restricted_wh_lifecycle(
        input_state,
        tile,
        spec,
        route_spec=route_spec,
        route_worker_count=worker_count,
    )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        replay_arrays = _state_arrays(replay.state, cp, prefix="output")
        replay_removals = [
            _removal_record(row) for row in replay.state.removal_records
        ]
        replay_hash = _state_sha256(
            _state_metadata(replay.state),
            replay_arrays,
            replay_removals,
            prefix="output",
        )
        stored_arrays = _state_arrays(stored_output, cp, prefix="output")
        stored_hash = _state_sha256(
            _state_metadata(stored_output),
            stored_arrays,
            output_removals,
            prefix="output",
        )
        if replay_hash != output_hash or stored_hash != output_hash:
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint endpoint differs from deterministic replay"
            )
        if not _canonical_tree_equal(_result_semantics(replay, cp), result_record):
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint result evidence differs from deterministic replay"
            )
        if not np.array_equal(
            cp.asnumpy(replay.minimum_source_distance_AU),
            arrays["minimum_source_distance_AU"],
        ):
            raise RestrictedWHCheckpointIntegrityError(
                "checkpoint minimum-distance array differs from replay"
            )
        cp.cuda.get_current_stream().synchronize()

    segment_record = {
        "input_state_sha256": input_hash,
        "state_sha256": output_hash,
        "lineage_sha256": lineage_hash,
        "source_tile_sha256": tile.content_sha256,
        "spec": record.get("spec"),
        "route_policy": record.get("route_policy"),
        "route_worker_count": worker_count,
        "result": result_record,
    }
    if _segment_sha256(segment_record) != record.get("segment_sha256"):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint segment semantic hash failed"
        )
    return (
        replay.state,
        output_hash,
        lineage_hash,
        output_removals,
        source_final_node_sha256,
        source_lineage_sha256,
    )


def load_latest_restricted_wh_lifecycle_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    genesis_state_sha256: str,
    expected_checkpoint_index: int,
    expected_checkpoint_content_sha256: str,
    backend: BackendSpec,
) -> LoadedRestrictedWHCheckpoint:
    """Verify and replay the complete chain, then release its latest GPU state."""

    root = Path(directory)
    _text(run_id, "run_id")
    _sha256(protocol_sha256, "protocol_sha256")
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    if (
        type(expected_checkpoint_index) is not int
        or not 0 <= expected_checkpoint_index <= MAXIMUM_CHECKPOINT_INDEX
    ):
        raise RestrictedWHCheckpointContractError(
            "expected_checkpoint_index must be an integer in [0, 10**20)"
        )
    _sha256(
        expected_checkpoint_content_sha256,
        "expected_checkpoint_content_sha256",
    )
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise RestrictedWHCheckpointContractError(
            "checkpoint load requires an exact CuPy BackendSpec"
        )
    if not root.exists():
        raise RestrictedWHCheckpointIntegrityError(
            "expected checkpoint root is missing"
        )
    _validate_root(root, create=False)
    records = _scan_records(root, run_id, protocol_sha256)
    if not records:
        raise RestrictedWHCheckpointIntegrityError(
            "expected checkpoint chain is missing"
        )
    if (
        records[-1][0] != expected_checkpoint_index
        or records[-1][2].get("checkpoint_content_sha256")
        != expected_checkpoint_content_sha256
    ):
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint chain head differs from the required external anchor"
        )
    implementation_identity = dict(_plain_tree(checkpoint_implementation_identity()))
    runtime_identity = dict(_plain_tree(checkpoint_runtime_identity(backend)))
    predecessor_state_sha256: str | None = None
    predecessor_lineage_sha256: str | None = None
    predecessor_removal_rows: list[dict[str, Any]] | None = None
    predecessor_source_final_node_sha256: str | None = None
    predecessor_source_lineage_sha256: str | None = None
    state: RestrictedWHLifecycleState | None = None
    for index, path, record in records:
        _verify_record_static(
            record,
            implementation_identity,
            runtime_identity,
            genesis_state_sha256,
        )
        arrays = _load_arrays(path, index, record)
        (
            state,
            predecessor_state_sha256,
            predecessor_lineage_sha256,
            predecessor_removal_rows,
            predecessor_source_final_node_sha256,
            predecessor_source_lineage_sha256,
        ) = _verify_loaded_segment(
            record,
            arrays,
            backend,
            genesis_state_sha256,
            predecessor_state_sha256,
            predecessor_lineage_sha256,
            predecessor_removal_rows,
            predecessor_source_final_node_sha256,
            predecessor_source_lineage_sha256,
        )
    if state is None or predecessor_state_sha256 is None:
        raise RestrictedWHCheckpointIntegrityError(
            "checkpoint history did not yield a state"
        )
    index, _, record = records[-1]
    resolved = resolve_backend(backend)
    with resolved.activate():
        resolved.xp.cuda.get_current_stream().synchronize()
    return _release_verified_checkpoint(
        state=state,
        checkpoint_index=index,
        checkpoint_content_sha256=record["checkpoint_content_sha256"],
        input_state_sha256=record["input_state_sha256"],
        state_sha256=record["state_sha256"],
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        record=record,
    )


__all__ = [
    "CHECKPOINT_CLAIMS",
    "CHECKPOINT_METHOD_ID",
    "CHECKPOINT_SCHEMA",
    "IMPLEMENTATION_SOURCE_SHA256",
    "LoadedRestrictedWHCheckpoint",
    "RestrictedWHCheckpointContractError",
    "RestrictedWHCheckpointError",
    "RestrictedWHCheckpointIntegrityError",
    "checkpoint_implementation_identity",
    "checkpoint_runtime_identity",
    "dump_restricted_wh_lifecycle_checkpoint",
    "load_latest_restricted_wh_lifecycle_checkpoint",
    "restricted_wh_lifecycle_state_sha256",
]
