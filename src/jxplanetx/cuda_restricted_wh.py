"""Experimental source-decoupled CUDA restricted Wisdom--Holman map.

This module is deliberately a direct-import experimental surface.  It does
not alter the qualified :mod:`jxplanetx.engine` API and it makes no production,
checkpoint, long-horizon, encounter, or inference claim.

The caller supplies an immutable, content-addressed fine source-path tile in
primary-relative, nonrotating Cartesian coordinates.  Tracers remain in the
same primary-relative coordinates throughout a segment and across in-memory
segment boundaries.  One coarse tracer step consumes an integer number of
fine source steps and applies

``perturbing half kick -> exact all-conic primary drift -> half kick``.

The primary monopole is present only in the exact Kepler drift.  Secondary
kicks include the indirect primary acceleration.  Source bodies never depend
on tracer state, so back-reaction is structurally absent.

Failures are transactional: numerical work is performed on private device
buffers, and any endpoint guard, Kepler-solver, or CUDA failure raises with a
handoff referring to the unchanged input state.  There is intentionally no
disk checkpoint API in this milestone.
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

from . import _all_conic_kepler as _all_conic_kepler_module
from ._all_conic_kepler import (
    CUDA_OPTIONS,
    STATUS_NAMES as KEPLER_STATUS_NAMES,
    drift_all_conic_cuda_batch,
)
from .engine.backends import BackendArrayError, resolve_backend
from .engine.contracts import BackendSpec, Provenance


METHOD_ID = "JX_CUDA_RESTRICTED_WH_PRIMARY_RELATIVE_V1"
KERNEL_ID = "jx.cuda.restricted-wh.primary-relative.v1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
COORDINATE_SYSTEM = "PRIMARY_RELATIVE_NONROTATING"
MAXIMUM_SOURCE_COUNT = 32
MAXIMUM_THREADS_PER_BLOCK = 128
MAXIMUM_PRIMARY_DYNAMICAL_PHASE = 0.25
MAXIMUM_BINARY64 = float.fromhex("0x1.fffffffffffffp+1023")
SOURCE_TILE_DOMAIN = b"jxplanetx.cuda-restricted-wh.source-tile.v1\x00"
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(
    Path(_all_conic_kepler_module.__file__).read_bytes()
).hexdigest()

RESULT_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "four_point_five_gyr_accuracy_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "cross_runtime_bitwise_reproducible": False,
        "globally_symplectic_with_external_source_tile": False,
        "close_encounter_qualified": False,
        "collision_qualified": False,
        "planet_nine_detected": False,
        "planet_nine_excluded": False,
        "stellar_flyby_inferred": False,
        "unbound_rogue_flyby_inferred": False,
        "independently_reproduced": False,
    }
)

KICK_STATUS_SUCCESS = 0
KICK_STATUS_NONFINITE = 1
KICK_STATUS_SOURCE_RADIUS_GUARD = 2
KICK_STATUS_FORCE_SINGULARITY = 3
KICK_STATUS_PRIMARY_PHASE_GUARD = 4
KICK_STATUS_SWEPT_SOURCE_RADIUS_GUARD = 5
KICK_STATUS_OUTER_BOUNDARY_GUARD = 6
KICK_STATUS_NAMES = {
    KICK_STATUS_SUCCESS: "SUCCESS",
    KICK_STATUS_NONFINITE: "NONFINITE",
    KICK_STATUS_SOURCE_RADIUS_GUARD: "SOURCE_RADIUS_GUARD",
    KICK_STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    KICK_STATUS_PRIMARY_PHASE_GUARD: "PRIMARY_PHASE_GUARD",
    KICK_STATUS_SWEPT_SOURCE_RADIUS_GUARD: "SWEPT_SOURCE_RADIUS_GUARD",
    KICK_STATUS_OUTER_BOUNDARY_GUARD: "OUTER_BOUNDARY_GUARD",
}


class CUDARestrictedWHError(RuntimeError):
    """Base error for the experimental restricted-WH component."""


class CUDARestrictedWHContractError(CUDARestrictedWHError, ValueError):
    """An input violates the explicit restricted-WH contract."""


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHState:
    """Device-resident tracer state on the authoritative fine source lattice.

    ``tracer_relative_*`` are always ``tracer - primary`` in the declared
    nonrotating frame.  The constructor deliberately leaves numerical buffers
    opaque; :func:`advance_cuda_restricted_wh` validates exact CuPy ownership,
    device, dtype, shape, contiguity, and domain before doing any work.

    ``epoch`` is not stored.  It is derived from ``epoch_origin_years`` and the
    integer ``global_source_step``, preventing a recursively accumulated
    floating epoch from becoming restart authority.
    """

    snapshot_id: str
    epoch_origin_years: float
    global_source_step: int
    source_step_years: float
    time_scale: str
    frame: str
    axes: str
    coordinate_system: str
    length_unit: str
    time_unit: str
    unit_system_id: str
    source_ids: tuple[str, ...]
    primary_source_id: str
    source_schedule_sha256: str
    tracer_relative_positions_AU: object
    tracer_relative_velocities_AU_per_year: object
    tracer_indices: object
    provenance: Provenance

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _finite(self.epoch_origin_years, "epoch_origin_years")
        if type(self.global_source_step) is not int or self.global_source_step < 0:
            raise CUDARestrictedWHContractError(
                "global_source_step must be a nonnegative integer"
            )
        step = _finite(self.source_step_years, "source_step_years")
        if step == 0.0:
            raise CUDARestrictedWHContractError("source_step_years cannot be zero")
        _derived_epoch(
            self.epoch_origin_years,
            self.global_source_step,
            step,
            "state epoch",
        )
        _validate_context(self)
        source_count = _source_ids(self.source_ids, self.primary_source_id)
        if source_count > MAXIMUM_SOURCE_COUNT:
            raise CUDARestrictedWHContractError(
                f"at most {MAXIMUM_SOURCE_COUNT} sources are supported"
            )
        _sha256(self.source_schedule_sha256, "source_schedule_sha256")
        for label in (
            "tracer_relative_positions_AU",
            "tracer_relative_velocities_AU_per_year",
            "tracer_indices",
        ):
            if getattr(self, label) is None:
                raise CUDARestrictedWHContractError(
                    f"{label} must be an explicit backend-native array"
                )
        if type(self.provenance) is not Provenance:
            raise CUDARestrictedWHContractError("provenance must be Provenance")

    @property
    def epoch_years(self) -> float:
        """Epoch derived only from the integer source-lattice coordinate."""

        return _derived_epoch(
            self.epoch_origin_years,
            self.global_source_step,
            self.source_step_years,
            "state epoch",
        )


@dataclass(frozen=True, eq=False)
class FrozenSourcePathTile:
    """Immutable host-canonical source path consumed by the CUDA map.

    Array inputs must be native binary64 NumPy arrays.  Construction copies
    them into read-only ``bytes``-backed arrays, so neither caller mutation nor
    a later writeable-flag change can alter the tile.  ``content_sha256`` is
    recomputed from canonical metadata and array bytes.  Supplying an expected
    digest makes ingestion fail closed if transport or storage changed.

    Source slot zero is the primary and must be exact zero in both position
    and velocity at every node.  Nodes are separated by exactly one signed
    ``source_step_years`` interval.  Each acceleration-bound entry is a caller
    certification that bounds the magnitude of ``tracer - source`` relative
    acceleration throughout every coarse map interval that uses this tile.
    The pre-drift swept guard combines that certificate with initial relative
    velocity to prove a conservative lower separation bound; an insufficient
    bound is a contract error by the caller, never an engine inference.
    """

    tile_id: str
    epoch_origin_years: float
    global_source_step: int
    source_step_years: float
    time_scale: str
    frame: str
    axes: str
    coordinate_system: str
    length_unit: str
    time_unit: str
    unit_system_id: str
    source_ids: tuple[str, ...]
    primary_source_id: str
    source_schedule_sha256: str
    source_relative_positions_AU: np.ndarray
    source_relative_velocities_AU_per_year: np.ndarray
    source_gm_AU3_per_year2: np.ndarray
    source_encounter_radii_AU: np.ndarray
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray
    provenance: Provenance
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.tile_id, "tile_id")
        _finite(self.epoch_origin_years, "epoch_origin_years")
        if type(self.global_source_step) is not int or self.global_source_step < 0:
            raise CUDARestrictedWHContractError(
                "global_source_step must be a nonnegative integer"
            )
        step = _finite(self.source_step_years, "source_step_years")
        if step == 0.0:
            raise CUDARestrictedWHContractError("source_step_years cannot be zero")
        _validate_context(self)
        source_count = _source_ids(self.source_ids, self.primary_source_id)
        if source_count > MAXIMUM_SOURCE_COUNT:
            raise CUDARestrictedWHContractError(
                f"at most {MAXIMUM_SOURCE_COUNT} sources are supported"
            )
        _sha256(self.source_schedule_sha256, "source_schedule_sha256")
        if type(self.provenance) is not Provenance:
            raise CUDARestrictedWHContractError("provenance must be Provenance")

        positions = _frozen_float64_array(
            self.source_relative_positions_AU,
            "source_relative_positions_AU",
            ndim=3,
        )
        velocities = _frozen_float64_array(
            self.source_relative_velocities_AU_per_year,
            "source_relative_velocities_AU_per_year",
            ndim=3,
        )
        gm = _frozen_float64_array(
            self.source_gm_AU3_per_year2,
            "source_gm_AU3_per_year2",
            ndim=1,
        )
        radii = _frozen_float64_array(
            self.source_encounter_radii_AU,
            "source_encounter_radii_AU",
            ndim=1,
        )
        acceleration_bounds = _frozen_float64_array(
            self.certified_relative_acceleration_bounds_AU_per_year2,
            "certified_relative_acceleration_bounds_AU_per_year2",
            ndim=1,
        )
        if positions.shape[0] < 2 or positions.shape[1:] != (source_count, 3):
            raise CUDARestrictedWHContractError(
                "source_relative_positions_AU must have shape "
                "(at least 2 nodes, source_count, 3)"
            )
        if velocities.shape != positions.shape:
            raise CUDARestrictedWHContractError(
                "source relative position and velocity tile shapes must match"
            )
        if (
            gm.shape != (source_count,)
            or radii.shape != (source_count,)
            or acceleration_bounds.shape != (source_count,)
        ):
            raise CUDARestrictedWHContractError(
                "source GM, encounter-radius, and acceleration-bound arrays "
                "must have shape (source_count,)"
            )
        if not bool(np.all(gm > 0.0)):
            raise CUDARestrictedWHContractError("every source GM must be positive")
        if not bool(np.all(radii > 0.0)):
            raise CUDARestrictedWHContractError(
                "every source encounter radius must be positive"
            )
        if not bool(np.all(acceleration_bounds > 0.0)):
            raise CUDARestrictedWHContractError(
                "every certified relative acceleration bound must be positive"
            )
        if not bool(np.all(positions[:, 0, :] == 0.0)) or not bool(
            np.all(velocities[:, 0, :] == 0.0)
        ):
            raise CUDARestrictedWHContractError(
                "primary source slot zero must be exact zero at every tile node"
            )
        if source_count > 1:
            secondary_squared = np.sum(positions[:, 1:, :] ** 2, axis=2)
            if bool(np.any(secondary_squared == 0.0)):
                raise CUDARestrictedWHContractError(
                    "a secondary source cannot coincide with the primary"
                )
        object.__setattr__(self, "source_relative_positions_AU", positions)
        object.__setattr__(
            self, "source_relative_velocities_AU_per_year", velocities
        )
        object.__setattr__(self, "source_gm_AU3_per_year2", gm)
        object.__setattr__(self, "source_encounter_radii_AU", radii)
        object.__setattr__(
            self,
            "certified_relative_acceleration_bounds_AU_per_year2",
            acceleration_bounds,
        )

        _derived_epoch(
            self.epoch_origin_years,
            self.global_source_step + int(positions.shape[0]) - 1,
            step,
            "source tile final epoch",
        )

        actual = source_path_tile_sha256(self)
        if self.content_sha256 is not None:
            expected = _sha256(self.content_sha256, "content_sha256")
            if expected != actual:
                raise CUDARestrictedWHContractError(
                    "source tile content_sha256 does not match canonical content"
                )
        object.__setattr__(self, "content_sha256", actual)

    @property
    def node_count(self) -> int:
        return int(self.source_relative_positions_AU.shape[0])

    @property
    def final_global_source_step(self) -> int:
        return self.global_source_step + self.node_count - 1

    @property
    def initial_epoch_years(self) -> float:
        return _derived_epoch(
            self.epoch_origin_years,
            self.global_source_step,
            self.source_step_years,
            "source tile initial epoch",
        )

    @property
    def final_epoch_years(self) -> float:
        return _derived_epoch(
            self.epoch_origin_years,
            self.final_global_source_step,
            self.source_step_years,
            "source tile final epoch",
        )


@dataclass(frozen=True)
class FixedStepCUDARestrictedWHSpec:
    """One fixed map segment indexed by an integer source-step ratio."""

    map_step_count: int
    source_steps_per_map_step: int
    backend: BackendSpec
    maximum_primary_dynamical_phase: float = 0.05
    threads_per_block: int = MAXIMUM_THREADS_PER_BLOCK
    primary_outer_boundary_AU: float | None = None

    def __post_init__(self) -> None:
        if type(self.map_step_count) is not int or self.map_step_count <= 0:
            raise CUDARestrictedWHContractError(
                "map_step_count must be a positive integer"
            )
        if (
            type(self.source_steps_per_map_step) is not int
            or self.source_steps_per_map_step <= 0
        ):
            raise CUDARestrictedWHContractError(
                "source_steps_per_map_step must be a positive integer"
            )
        if type(self.backend) is not BackendSpec:
            raise CUDARestrictedWHContractError("backend must be BackendSpec")
        if self.backend.backend_id != "cupy":
            raise CUDARestrictedWHContractError(
                "the restricted-WH component requires backend_id='cupy'"
            )
        phase = _finite(
            self.maximum_primary_dynamical_phase,
            "maximum_primary_dynamical_phase",
        )
        if not 0.0 < phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE:
            raise CUDARestrictedWHContractError(
                "maximum_primary_dynamical_phase must be in (0, 0.25]"
            )
        if (
            type(self.threads_per_block) is not int
            or not 32 <= self.threads_per_block <= MAXIMUM_THREADS_PER_BLOCK
            or self.threads_per_block % 32 != 0
        ):
            raise CUDARestrictedWHContractError(
                "threads_per_block must be a warp multiple in [32, 128]"
            )
        if self.primary_outer_boundary_AU is not None:
            boundary = _finite(
                self.primary_outer_boundary_AU,
                "primary_outer_boundary_AU",
            )
            if not boundary > 0.0:
                raise CUDARestrictedWHContractError(
                    "primary_outer_boundary_AU must be positive"
                )


@dataclass(frozen=True)
class CUDARestrictedWHFailureHandoff:
    """Fail-closed in-memory handoff; no partial state is committed."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    failure_global_source_step: int
    map_step_index: int
    phase: str
    reason: str
    lane_index: int | None
    tracer_index: int | None
    source_index: int | None
    committed_map_steps: int = 0
    checkpoint_qualified: bool = False

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise CUDARestrictedWHContractError(
                "handoff state must be the exact CUDARestrictedWHState input"
            )
        _sha256(self.source_tile_sha256, "source_tile_sha256")
        if (
            type(self.failure_global_source_step) is not int
            or self.failure_global_source_step < 0
        ):
            raise CUDARestrictedWHContractError(
                "failure_global_source_step must be nonnegative"
            )
        if type(self.map_step_index) is not int or self.map_step_index < 0:
            raise CUDARestrictedWHContractError("map_step_index must be nonnegative")
        _text(self.phase, "phase")
        _text(self.reason, "reason")
        for value, label in (
            (self.lane_index, "lane_index"),
            (self.tracer_index, "tracer_index"),
            (self.source_index, "source_index"),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise CUDARestrictedWHContractError(
                    f"{label} must be a nonnegative integer or None"
                )
        if self.committed_map_steps != 0 or self.checkpoint_qualified is not False:
            raise CUDARestrictedWHContractError(
                "M1 handoffs must commit zero steps and make no checkpoint claim"
            )


class CUDARestrictedWHIntegrationError(CUDARestrictedWHError):
    """A guard, all-conic solve, or CUDA operation failed transactionally."""

    def __init__(
        self, message: str, handoff: CUDARestrictedWHFailureHandoff
    ) -> None:
        super().__init__(message)
        self.handoff = handoff


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHResult:
    """One fully committed in-memory segment result."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    minimum_endpoint_source_distance_AU: object
    maximum_kepler_iterations: int
    runtime_identity: Mapping[str, Any]
    method_id: str = METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    claims: Mapping[str, bool] = RESULT_CLAIMS

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise CUDARestrictedWHContractError(
                "result state must be an exact CUDARestrictedWHState"
            )
        _sha256(self.source_tile_sha256, "source_tile_sha256")
        if (
            type(self.maximum_kepler_iterations) is not int
            or self.maximum_kepler_iterations < 0
        ):
            raise CUDARestrictedWHContractError(
                "maximum_kepler_iterations must be nonnegative"
            )
        if self.method_id != METHOD_ID:
            raise CUDARestrictedWHContractError("result method_id is immutable")
        if self.scientific_classification != SCIENTIFIC_CLASSIFICATION:
            raise CUDARestrictedWHContractError(
                "scientific classification cannot be widened"
            )
        if self.project_scope != PROJECT_SCOPE:
            raise CUDARestrictedWHContractError("project scope cannot be widened")
        if dict(self.claims) != dict(RESULT_CLAIMS) or any(self.claims.values()):
            raise CUDARestrictedWHContractError(
                "result claim controls must remain the fixed all-false map"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(RESULT_CLAIMS)))
        object.__setattr__(
            self, "runtime_identity", _freeze_tree(self.runtime_identity)
        )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CUDARestrictedWHContractError(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _freeze_tree(value: Any) -> Any:
    """Recursively detach and freeze runtime/provenance container values."""

    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CUDARestrictedWHContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CUDARestrictedWHContractError(f"{label} must be a finite number")
    return result


def _derived_epoch(
    origin: float, global_source_step: int, source_step_years: float, label: str
) -> float:
    try:
        result = origin + global_source_step * source_step_years
    except OverflowError as exc:
        raise CUDARestrictedWHContractError(f"{label} must remain finite") from exc
    if not math.isfinite(result):
        raise CUDARestrictedWHContractError(f"{label} must remain finite")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise CUDARestrictedWHContractError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _source_ids(source_ids: object, primary_source_id: object) -> int:
    if type(source_ids) is not tuple or not source_ids:
        raise CUDARestrictedWHContractError(
            "source_ids must be a nonempty immutable tuple"
        )
    for index, source_id in enumerate(source_ids):
        _text(source_id, f"source_ids[{index}]")
    if len(set(source_ids)) != len(source_ids):
        raise CUDARestrictedWHContractError("source_ids must be unique")
    _text(primary_source_id, "primary_source_id")
    if primary_source_id != source_ids[0]:
        raise CUDARestrictedWHContractError(
            "primary_source_id must explicitly identify source slot zero"
        )
    return len(source_ids)


def _validate_context(value: Any) -> None:
    for label in ("time_scale", "frame", "axes", "coordinate_system"):
        _text(getattr(value, label), label)
    if value.coordinate_system != COORDINATE_SYSTEM:
        raise CUDARestrictedWHContractError(
            f"coordinate_system must be {COORDINATE_SYSTEM!r}"
        )
    if (
        value.length_unit != "AU"
        or value.time_unit != "year"
        or value.unit_system_id != "AU-YEAR"
    ):
        raise CUDARestrictedWHContractError(
            "M1 accepts only length_unit='AU', time_unit='year', and "
            "unit_system_id='AU-YEAR'"
        )


def _frozen_float64_array(value: object, label: str, *, ndim: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise CUDARestrictedWHContractError(
            f"{label} must be an exact NumPy array"
        )
    if value.dtype != np.float64 or value.ndim != ndim:
        raise CUDARestrictedWHContractError(
            f"{label} must be a {ndim}-dimensional float64 NumPy array"
        )
    if not bool(np.all(np.isfinite(value))):
        raise CUDARestrictedWHContractError(f"{label} must be finite")
    canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    # ``bytes`` owns immutable storage; unlike a normal read-only ndarray, its
    # WRITEABLE flag cannot subsequently be re-enabled.
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        canonical.shape
    )


def _canonical_tile_metadata(tile: FrozenSourcePathTile) -> bytes:
    metadata = {
        "axes": tile.axes,
        "coordinate_system": tile.coordinate_system,
        "epoch_origin_years_hex": float(tile.epoch_origin_years).hex(),
        "frame": tile.frame,
        "global_source_step": tile.global_source_step,
        "length_unit": tile.length_unit,
        "primary_source_id": tile.primary_source_id,
        "provenance": {
            "citation": tile.provenance.citation,
            "sha256": tile.provenance.sha256,
            "source_id": tile.provenance.source_id,
            "version": tile.provenance.version,
        },
        "source_ids": list(tile.source_ids),
        "source_schedule_sha256": tile.source_schedule_sha256,
        "source_step_years_hex": float(tile.source_step_years).hex(),
        "tile_id": tile.tile_id,
        "time_scale": tile.time_scale,
        "time_unit": tile.time_unit,
        "unit_system_id": tile.unit_system_id,
    }
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def source_path_tile_sha256(tile: FrozenSourcePathTile) -> str:
    """Return the canonical digest of a validated frozen source tile."""

    if type(tile) is not FrozenSourcePathTile:
        raise CUDARestrictedWHContractError(
            "source_path_tile_sha256 requires an exact FrozenSourcePathTile"
        )
    digest = hashlib.sha256()
    digest.update(SOURCE_TILE_DOMAIN)
    metadata = _canonical_tile_metadata(tile)
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    for label, array in (
        ("source_relative_positions_AU", tile.source_relative_positions_AU),
        (
            "source_relative_velocities_AU_per_year",
            tile.source_relative_velocities_AU_per_year,
        ),
        ("source_gm_AU3_per_year2", tile.source_gm_AU3_per_year2),
        ("source_encounter_radii_AU", tile.source_encounter_radii_AU),
        (
            "certified_relative_acceleration_bounds_AU_per_year2",
            tile.certified_relative_acceleration_bounds_AU_per_year2,
        ),
    ):
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "little"))
        digest.update(encoded_label)
        digest.update(array.ndim.to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def verify_source_path_tile(tile: FrozenSourcePathTile) -> str:
    """Recompute a tile digest, rejecting any reflective tampering."""

    if type(tile) is not FrozenSourcePathTile:
        raise CUDARestrictedWHContractError(
            "source tile must be an exact FrozenSourcePathTile"
        )
    actual = source_path_tile_sha256(tile)
    if actual != tile.content_sha256:
        raise CUDARestrictedWHContractError(
            "frozen source tile no longer matches content_sha256"
        )
    return actual


KICK_DEVICE_SOURCE = r'''
#define JX_KICK_SUCCESS 0
#define JX_KICK_NONFINITE 1
#define JX_KICK_SOURCE_RADIUS 2
#define JX_KICK_FORCE_SINGULARITY 3
#define JX_KICK_PRIMARY_PHASE 4
#define JX_KICK_SWEPT_SOURCE_RADIUS 5
#define JX_KICK_OUTER_BOUNDARY 6
#define JX_KICK_DBL_MAX 1.79769313486231570814527423731704357e308

__device__ __forceinline__ int jx_restricted_wh_half_kick_device(
    const double* input_position,
    const double* input_velocity,
    const double* source_position,
    const double* source_velocity,
    const double* source_gm,
    const double* source_radius,
    const double* relative_acceleration_bound,
    const int source_count,
    const double half_step,
    const double full_step,
    const double maximum_primary_phase,
    const double primary_outer_boundary,
    const int apply_pre_drift_guards,
    double* output_velocity,
    double* minimum_distance,
    int* output_source)
{
    const double qx = input_position[0];
    const double qy = input_position[1];
    const double qz = input_position[2];
    const double ux = input_velocity[0];
    const double uy = input_velocity[1];
    const double uz = input_velocity[2];
    output_velocity[0] = ux;
    output_velocity[1] = uy;
    output_velocity[2] = uz;
    *minimum_distance = JX_KICK_DBL_MAX;
    *output_source = -1;
    if (!isfinite(qx) || !isfinite(qy) || !isfinite(qz)
        || !isfinite(ux) || !isfinite(uy) || !isfinite(uz))
        return JX_KICK_NONFINITE;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double primary_distance = nan("");
    for (int source = 0; source < source_count; ++source) {
        const int source_offset = source * 3;
        const double sx = source_position[source_offset];
        const double sy = source_position[source_offset + 1];
        const double sz = source_position[source_offset + 2];
        const double dx = sx - qx;
        const double dy = sy - qy;
        const double dz = sz - qz;
        const double distance_squared = (dx * dx + dy * dy) + dz * dz;
        if (!isfinite(distance_squared) || !(distance_squared > 0.0)) {
            *output_source = source;
            return JX_KICK_FORCE_SINGULARITY;
        }
        const double distance = sqrt(distance_squared);
        *minimum_distance = fmin(*minimum_distance, distance);
        if (!(distance > source_radius[source])) {
            *output_source = source;
            return JX_KICK_SOURCE_RADIUS;
        }
        if (source == 0) primary_distance = distance;
        if (source > 0) {
            const double source_squared = (sx * sx + sy * sy) + sz * sz;
            if (!isfinite(source_squared) || !(source_squared > 0.0)) {
                *output_source = source;
                return JX_KICK_FORCE_SINGULARITY;
            }
            const double source_distance = sqrt(source_squared);
            const double inverse_source = 1.0 / source_distance;
            const double inverse_source3 =
                (inverse_source * inverse_source) * inverse_source;
            const double mu = source_gm[source];
            const double tracer_squared = (qx * qx + qy * qy) + qz * qz;

            /*
             * The direct difference
             *
             *   (s-q)/|s-q|^3 - s/|s|^3
             *
             * catastrophically cancels for a distant source.  In the
             * scale-separated regime use the exact Encke-style identity
             * below.  eta=(|s-q|^2-|s|^2)/|s|^2 is formed from dot products,
             * and 1-(1+eta)^(3/2) is evaluated without subtracting from one.
             */
            if (isfinite(tracer_squared)
                && tracer_squared <= 0.0625 * source_squared) {
                const double source_dot_tracer =
                    (sx * qx + sy * qy) + sz * qz;
                const double eta =
                    (tracer_squared - 2.0 * source_dot_tracer)
                    / source_squared;
                const double one_plus_eta = 1.0 + eta;
                if (!isfinite(eta) || !isfinite(one_plus_eta)
                    || !(one_plus_eta > 0.0)) {
                    *output_source = source;
                    return JX_KICK_NONFINITE;
                }
                const double eta_root = sqrt(one_plus_eta);
                const double eta_power = one_plus_eta * eta_root;
                const double one_minus_eta_power =
                    -eta * ((eta * eta + 3.0 * eta) + 3.0)
                    / (1.0 + eta_power);
                const double scale =
                    (mu * inverse_source3) / eta_power;
                ax += scale * (-qx + sx * one_minus_eta_power);
                ay += scale * (-qy + sy * one_minus_eta_power);
                az += scale * (-qz + sz * one_minus_eta_power);
            } else {
                const double inverse_distance = 1.0 / distance;
                const double inverse_distance3 =
                    (inverse_distance * inverse_distance) * inverse_distance;
                ax += mu * (dx * inverse_distance3 - sx * inverse_source3);
                ay += mu * (dy * inverse_distance3 - sy * inverse_source3);
                az += mu * (dz * inverse_distance3 - sz * inverse_source3);
            }
        }
    }
    if (!(primary_distance < primary_outer_boundary)) {
        *output_source = 0;
        return JX_KICK_OUTER_BOUNDARY;
    }
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) {
        return JX_KICK_NONFINITE;
    }
    const double next_ux = ux + half_step * ax;
    const double next_uy = uy + half_step * ay;
    const double next_uz = uz + half_step * az;
    if (!isfinite(next_ux) || !isfinite(next_uy) || !isfinite(next_uz)) {
        return JX_KICK_NONFINITE;
    }
    if (apply_pre_drift_guards != 0) {
        const double absolute_step = fabs(full_step);
        const double half_step_squared = 0.5 * absolute_step * absolute_step;
        double primary_certified_lower_distance = nan("");
        double primary_certified_upper_distance = nan("");
        for (int source = 0; source < source_count; ++source) {
            const int source_offset = source * 3;
            const double dx = source_position[source_offset] - qx;
            const double dy = source_position[source_offset + 1] - qy;
            const double dz = source_position[source_offset + 2] - qz;
            const double distance = sqrt((dx * dx + dy * dy) + dz * dz);
            const double wx = next_ux - source_velocity[source_offset];
            const double wy = next_uy - source_velocity[source_offset + 1];
            const double wz = next_uz - source_velocity[source_offset + 2];
            const double relative_speed = sqrt((wx * wx + wy * wy) + wz * wz);
            const double displacement_bound = relative_speed * absolute_step
                + relative_acceleration_bound[source] * half_step_squared;
            const double certified_lower_distance = distance - displacement_bound;
            if (!isfinite(relative_speed) || !isfinite(displacement_bound)
                || !isfinite(certified_lower_distance)) {
                *output_source = source;
                return JX_KICK_NONFINITE;
            }
            if (!(certified_lower_distance > source_radius[source])) {
                *output_source = source;
                return JX_KICK_SWEPT_SOURCE_RADIUS;
            }
            if (source == 0) {
                primary_certified_lower_distance = certified_lower_distance;
                primary_certified_upper_distance = distance + displacement_bound;
            }
        }
        if (!isfinite(primary_certified_upper_distance)
            || !(primary_certified_upper_distance < primary_outer_boundary)) {
            *output_source = 0;
            return JX_KICK_OUTER_BOUNDARY;
        }
        const double lower_squared = primary_certified_lower_distance
            * primary_certified_lower_distance;
        const double phase = absolute_step * sqrt(
            source_gm[0]
            / (lower_squared * primary_certified_lower_distance));
        if (!isfinite(phase) || !(phase < maximum_primary_phase)) {
            *output_source = 0;
            return JX_KICK_PRIMARY_PHASE;
        }
    }
    output_velocity[0] = next_ux;
    output_velocity[1] = next_uy;
    output_velocity[2] = next_uz;
    return JX_KICK_SUCCESS;
}
'''

KICK_BATCH_KERNEL_SOURCE = r'''
extern "C" __global__
void jx_restricted_wh_half_kick(
    const double* input_position,
    const double* input_velocity,
    const double* source_position,
    const double* source_velocity,
    const double* source_gm,
    const double* source_radius,
    const double* relative_acceleration_bound,
    const int lane_count,
    const int source_count,
    const double half_step,
    const double full_step,
    const double maximum_primary_phase,
    const double primary_outer_boundary,
    const int apply_pre_drift_guards,
    double* output_velocity,
    double* minimum_distance,
    int* output_status,
    int* output_source)
{
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lane_count) return;
    const int offset = lane * 3;
    output_status[lane] = jx_restricted_wh_half_kick_device(
        input_position + offset,
        input_velocity + offset,
        source_position,
        source_velocity,
        source_gm,
        source_radius,
        relative_acceleration_bound,
        source_count,
        half_step,
        full_step,
        maximum_primary_phase,
        primary_outer_boundary,
        apply_pre_drift_guards,
        output_velocity + offset,
        minimum_distance + lane,
        output_source + lane);
}
'''

KICK_CUDA_SOURCE = KICK_DEVICE_SOURCE + KICK_BATCH_KERNEL_SOURCE
KICK_CUDA_SOURCE_SHA256 = hashlib.sha256(KICK_CUDA_SOURCE.encode("ascii")).hexdigest()

FUSED_KERNEL_ID = "jx.cuda.restricted-wh.primary-relative.fused.v1"
FUSED_BATCH_KERNEL_SOURCE = r'''
extern "C" __global__
void jx_restricted_wh_fused_segment(
    const double* input_position,
    const double* input_velocity,
    const double* source_position,
    const double* source_velocity,
    const double* source_gm,
    const double* source_radius,
    const double* relative_acceleration_bound,
    const int lane_count,
    const int source_count,
    const int map_step_count,
    const int source_steps_per_map_step,
    const double half_step,
    const double full_step,
    const double maximum_primary_phase,
    const double primary_outer_boundary,
    double* output_position,
    double* output_velocity,
    double* output_minimum_distance,
    int* output_maximum_iterations,
    int* output_status,
    int* output_failure_map,
    int* output_failure_phase,
    int* output_failure_source)
{
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lane_count) return;
    const int lane_offset = lane * 3;
    double position[3] = {
        input_position[lane_offset],
        input_position[lane_offset + 1],
        input_position[lane_offset + 2]
    };
    double velocity[3] = {
        input_velocity[lane_offset],
        input_velocity[lane_offset + 1],
        input_velocity[lane_offset + 2]
    };
    output_position[lane_offset] = position[0];
    output_position[lane_offset + 1] = position[1];
    output_position[lane_offset + 2] = position[2];
    output_velocity[lane_offset] = velocity[0];
    output_velocity[lane_offset + 1] = velocity[1];
    output_velocity[lane_offset + 2] = velocity[2];
    output_minimum_distance[lane] = JX_KICK_DBL_MAX;
    output_maximum_iterations[lane] = 0;
    output_status[lane] = JX_KICK_SUCCESS;
    output_failure_map[lane] = -1;
    output_failure_phase[lane] = -1;
    output_failure_source[lane] = -1;

    double cumulative_minimum = JX_KICK_DBL_MAX;
    int maximum_iterations = 0;
    const long long source_node_stride = (long long)source_count * 3LL;
    for (int map_step = 0; map_step < map_step_count; ++map_step) {
        const long long start_node =
            (long long)map_step * (long long)source_steps_per_map_step;
        const long long end_node =
            start_node + (long long)source_steps_per_map_step;
        const double* start_position =
            source_position + start_node * source_node_stride;
        const double* start_velocity =
            source_velocity + start_node * source_node_stride;
        const double* end_position =
            source_position + end_node * source_node_stride;
        const double* end_velocity =
            source_velocity + end_node * source_node_stride;

        double kicked_velocity[3];
        double pre_minimum;
        int failure_source = -1;
        int status = jx_restricted_wh_half_kick_device(
            position,
            velocity,
            start_position,
            start_velocity,
            source_gm,
            source_radius,
            relative_acceleration_bound,
            source_count,
            half_step,
            full_step,
            maximum_primary_phase,
            primary_outer_boundary,
            1,
            kicked_velocity,
            &pre_minimum,
            &failure_source);
        cumulative_minimum = fmin(cumulative_minimum, pre_minimum);
        if (status != JX_KICK_SUCCESS) {
            output_minimum_distance[lane] = cumulative_minimum;
            output_maximum_iterations[lane] = maximum_iterations;
            output_status[lane] = status;
            output_failure_map[lane] = map_step;
            output_failure_phase[lane] = 0;
            output_failure_source[lane] = failure_source;
            return;
        }

        double drift_position[3];
        double drift_velocity[3];
        int conic = 0;
        double beta = nan("");
        double anomaly = nan("");
        int iterations = 0;
        int expansions = 0;
        double residual = nan("");
        double tolerance = nan("");
        double lagrange = nan("");
        double energy = nan("");
        double angular = nan("");
        status = jx_all_conic_drift_device(
            position,
            kicked_velocity,
            source_gm[0],
            full_step,
            drift_position,
            drift_velocity,
            &conic,
            &beta,
            &anomaly,
            &iterations,
            &expansions,
            &residual,
            &tolerance,
            &lagrange,
            &energy,
            &angular);
        if (iterations > maximum_iterations) maximum_iterations = iterations;
        if (status != JX_ACK_SUCCESS) {
            output_minimum_distance[lane] = cumulative_minimum;
            output_maximum_iterations[lane] = maximum_iterations;
            output_status[lane] = status;
            output_failure_map[lane] = map_step;
            output_failure_phase[lane] = 1;
            output_failure_source[lane] = -1;
            return;
        }

        double final_velocity[3];
        double post_minimum;
        failure_source = -1;
        status = jx_restricted_wh_half_kick_device(
            drift_position,
            drift_velocity,
            end_position,
            end_velocity,
            source_gm,
            source_radius,
            relative_acceleration_bound,
            source_count,
            half_step,
            full_step,
            maximum_primary_phase,
            primary_outer_boundary,
            0,
            final_velocity,
            &post_minimum,
            &failure_source);
        cumulative_minimum = fmin(cumulative_minimum, post_minimum);
        if (status != JX_KICK_SUCCESS) {
            output_minimum_distance[lane] = cumulative_minimum;
            output_maximum_iterations[lane] = maximum_iterations;
            output_status[lane] = status;
            output_failure_map[lane] = map_step;
            output_failure_phase[lane] = 2;
            output_failure_source[lane] = failure_source;
            return;
        }

        position[0] = drift_position[0];
        position[1] = drift_position[1];
        position[2] = drift_position[2];
        velocity[0] = final_velocity[0];
        velocity[1] = final_velocity[1];
        velocity[2] = final_velocity[2];
    }

    output_position[lane_offset] = position[0];
    output_position[lane_offset + 1] = position[1];
    output_position[lane_offset + 2] = position[2];
    output_velocity[lane_offset] = velocity[0];
    output_velocity[lane_offset + 1] = velocity[1];
    output_velocity[lane_offset + 2] = velocity[2];
    output_minimum_distance[lane] = cumulative_minimum;
    output_maximum_iterations[lane] = maximum_iterations;
}
'''
FUSED_CUDA_SOURCE = (
    _all_conic_kepler_module.CUDA_DEVICE_SOURCE
    + KICK_DEVICE_SOURCE
    + FUSED_BATCH_KERNEL_SOURCE
)
FUSED_CUDA_SOURCE_SHA256 = hashlib.sha256(
    FUSED_CUDA_SOURCE.encode("ascii")
).hexdigest()


def _validate_compatibility(
    state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
) -> None:
    if type(state) is not CUDARestrictedWHState:
        raise CUDARestrictedWHContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(tile) is not FrozenSourcePathTile:
        raise CUDARestrictedWHContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    if type(spec) is not FixedStepCUDARestrictedWHSpec:
        raise CUDARestrictedWHContractError(
            "spec must be an exact FixedStepCUDARestrictedWHSpec"
        )
    verify_source_path_tile(tile)
    fields = (
        "epoch_origin_years",
        "source_step_years",
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
    )
    for field in fields:
        if getattr(state, field) != getattr(tile, field):
            raise CUDARestrictedWHContractError(
                f"state and source tile disagree on {field}"
            )
    if tile.global_source_step != state.global_source_step:
        raise CUDARestrictedWHContractError(
            "source tile must begin at state.global_source_step"
        )
    required_intervals = spec.map_step_count * spec.source_steps_per_map_step
    if tile.node_count < required_intervals + 1:
        raise CUDARestrictedWHContractError(
            "source tile does not contain every required fine source node"
        )
    if (
        spec.primary_outer_boundary_AU is not None
        and not spec.primary_outer_boundary_AU
        > float(tile.source_encounter_radii_AU[0])
    ):
        raise CUDARestrictedWHContractError(
            "primary_outer_boundary_AU must exceed the primary encounter radius"
        )


@dataclass(frozen=True)
class _ValidatedTracerArrays:
    positions: Any
    velocities: Any
    indices: Any
    lane_count: int


def _validate_tracer_arrays(
    state: CUDARestrictedWHState, resolved: Any
) -> _ValidatedTracerArrays:
    cp = resolved.xp
    try:
        positions = resolved.require_native_array(
            state.tracer_relative_positions_AU,
            "tracer_relative_positions_AU",
        )
        velocities = resolved.require_native_array(
            state.tracer_relative_velocities_AU_per_year,
            "tracer_relative_velocities_AU_per_year",
        )
        indices = resolved.require_native_array(
            state.tracer_indices, "tracer_indices"
        )
    except BackendArrayError as exc:
        raise CUDARestrictedWHContractError(str(exc)) from exc
    if (
        positions.dtype != cp.float64
        or positions.ndim != 2
        or positions.shape[1] != 3
        or not positions.flags.c_contiguous
    ):
        raise CUDARestrictedWHContractError(
            "tracer_relative_positions_AU must be a C-contiguous float64 "
            "CuPy array with shape (lanes, 3)"
        )
    if (
        velocities.dtype != cp.float64
        or velocities.shape != positions.shape
        or not velocities.flags.c_contiguous
    ):
        raise CUDARestrictedWHContractError(
            "tracer_relative_velocities_AU_per_year must be a C-contiguous "
            "float64 CuPy array matching positions"
        )
    lane_count = int(positions.shape[0])
    if (
        indices.dtype != cp.uint64
        or indices.shape != (lane_count,)
        or not indices.flags.c_contiguous
    ):
        raise CUDARestrictedWHContractError(
            "tracer_indices must be a C-contiguous uint64 CuPy vector"
        )
    if lane_count and (
        not resolved.scalar_bool(cp.all(cp.isfinite(positions)), "finite positions")
        or not resolved.scalar_bool(
            cp.all(cp.isfinite(velocities)), "finite velocities"
        )
    ):
        raise CUDARestrictedWHContractError("tracer state arrays must be finite")
    if lane_count and int(cp.unique(indices).size) != lane_count:
        raise CUDARestrictedWHContractError("tracer_indices must be unique")
    return _ValidatedTracerArrays(positions, velocities, indices, lane_count)


def _record_first_failure(
    cp: Any,
    first_map: Any,
    first_phase: Any,
    first_reason: Any,
    first_source: Any,
    status: Any,
    source: Any,
    map_index: int,
    phase_code: int,
) -> tuple[Any, Any, Any, Any]:
    fresh = (first_map < 0) & (status != 0)
    return (
        cp.where(fresh, cp.int32(map_index), first_map),
        cp.where(fresh, cp.int32(phase_code), first_phase),
        cp.where(fresh, status.astype(cp.int32), first_reason),
        cp.where(fresh, source.astype(cp.int32), first_source),
    )


def _failure_handoff(
    *,
    state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    map_index: int,
    phase_code: int,
    reason_code: int,
    lane_index: int | None,
    tracer_index: int | None,
    source_index: int | None,
    reason_override: str | None = None,
) -> CUDARestrictedWHFailureHandoff:
    phases = {0: "PRE_KICK_GUARD", 1: "KEPLER_DRIFT", 2: "POST_KICK_GUARD"}
    if reason_override is not None:
        reason = reason_override
    elif phase_code == 1:
        reason = f"KEPLER_{KEPLER_STATUS_NAMES.get(reason_code, reason_code)}"
    else:
        reason = KICK_STATUS_NAMES.get(reason_code, f"KICK_STATUS_{reason_code}")
    failure_step = state.global_source_step + (
        map_index * spec.source_steps_per_map_step
    )
    if phase_code == 2:
        failure_step += spec.source_steps_per_map_step
    return CUDARestrictedWHFailureHandoff(
        state=state,
        source_tile_sha256=tile.content_sha256 or "",
        failure_global_source_step=failure_step,
        map_step_index=map_index,
        phase=phases.get(phase_code, "CUDA_RUNTIME"),
        reason=reason,
        lane_index=lane_index,
        tracer_index=tracer_index,
        source_index=source_index,
    )


def _runtime_identity(
    cp: Any,
    device: str,
    *,
    execution_kernel_id: str = KERNEL_ID,
    execution_kernel_source_sha256: str = KICK_CUDA_SOURCE_SHA256,
) -> Mapping[str, Any]:
    device_index = int(cp.cuda.Device().id)
    properties = cp.cuda.runtime.getDeviceProperties(device_index)
    raw_name = properties["name"]
    device_name = (
        raw_name.decode("ascii") if isinstance(raw_name, bytes) else str(raw_name)
    )
    return MappingProxyType(
        {
            "method_id": METHOD_ID,
            "backend": "cupy",
            "device": device,
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
            "cupy_version": str(cp.__version__),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(
                cp.cuda.get_local_runtime_version()
            ),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc_version": [int(value) for value in cp.cuda.nvrtc.getVersion()],
            "kernel_id": execution_kernel_id,
            "execution_kernel_source_sha256": (
                execution_kernel_source_sha256
            ),
            "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
            "all_conic_implementation_source_sha256": (
                ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256
            ),
            "all_conic_device_source_sha256": hashlib.sha256(
                _all_conic_kepler_module.CUDA_DEVICE_SOURCE.encode("ascii")
            ).hexdigest(),
            "kick_cuda_source_sha256": KICK_CUDA_SOURCE_SHA256,
            "fused_cuda_source_sha256": FUSED_CUDA_SOURCE_SHA256,
            "cuda_options": list(CUDA_OPTIONS),
            "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
        }
    )


def advance_cuda_restricted_wh(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
) -> CUDARestrictedWHResult:
    """Advance a quiet restricted-WH segment or raise without partial commit.

    Source-radius checks are endpoint guards, not continuous encounter
    localization.  A guard hit fails the whole segment closed.  Successful
    state may be passed directly to another call whose tile starts at the
    returned ``global_source_step``; no disk-checkpoint claim is made.
    """

    _validate_compatibility(state, source_tile, spec)
    resolved = resolve_backend(spec.backend)
    if resolved.name != "cupy":  # defensive; the spec already rejects NumPy.
        raise CUDARestrictedWHContractError("resolved backend must be CuPy")
    cp = resolved.xp
    ratio = spec.source_steps_per_map_step
    full_step = source_tile.source_step_years * ratio
    half_step = 0.5 * full_step
    primary_outer_boundary = (
        MAXIMUM_BINARY64
        if spec.primary_outer_boundary_AU is None
        else spec.primary_outer_boundary_AU
    )
    map_index = 0

    with resolved.activate():
        try:
            arrays = _validate_tracer_arrays(state, resolved)
            # Canonical tile-to-device transfer is explicit at this boundary.
            source_positions = cp.asarray(
                source_tile.source_relative_positions_AU, dtype=cp.float64
            )
            source_gm = cp.asarray(
                source_tile.source_gm_AU3_per_year2, dtype=cp.float64
            )
            source_radii = cp.asarray(
                source_tile.source_encounter_radii_AU, dtype=cp.float64
            )
            relative_acceleration_bounds = cp.asarray(
                source_tile.certified_relative_acceleration_bounds_AU_per_year2,
                dtype=cp.float64,
            )
            source_velocities = cp.asarray(
                source_tile.source_relative_velocities_AU_per_year,
                dtype=cp.float64,
            )
            positions = cp.array(
                arrays.positions, dtype=cp.float64, copy=True, order="C"
            )
            velocities = cp.array(
                arrays.velocities, dtype=cp.float64, copy=True, order="C"
            )
            lane_count = arrays.lane_count
            first_map = cp.full(lane_count, -1, dtype=cp.int32)
            first_phase = cp.full(lane_count, -1, dtype=cp.int32)
            first_reason = cp.zeros(lane_count, dtype=cp.int32)
            first_source = cp.full(lane_count, -1, dtype=cp.int32)
            minimum_distance = cp.full(lane_count, cp.inf, dtype=cp.float64)
            maximum_iterations = cp.zeros(lane_count, dtype=cp.int32)
            kick = cp.RawKernel(
                KICK_CUDA_SOURCE,
                "jx_restricted_wh_half_kick",
                options=CUDA_OPTIONS,
            )
            blocks = (
                (lane_count + spec.threads_per_block - 1)
                // spec.threads_per_block
            )

            for map_index in range(spec.map_step_count):
                start_node = map_index * ratio
                end_node = start_node + ratio
                kick_status = cp.empty(lane_count, dtype=cp.int32)
                kick_source = cp.empty(lane_count, dtype=cp.int32)
                endpoint_minimum = cp.empty(lane_count, dtype=cp.float64)
                kicked_velocity = cp.empty_like(velocities)
                if lane_count:
                    kick(
                        (blocks,),
                        (spec.threads_per_block,),
                        (
                            positions,
                            velocities,
                            source_positions[start_node],
                            source_velocities[start_node],
                            source_gm,
                            source_radii,
                            relative_acceleration_bounds,
                            cp.int32(lane_count),
                            cp.int32(len(source_tile.source_ids)),
                            cp.float64(half_step),
                            cp.float64(full_step),
                            cp.float64(spec.maximum_primary_dynamical_phase),
                            cp.float64(primary_outer_boundary),
                            cp.int32(1),
                            kicked_velocity,
                            endpoint_minimum,
                            kick_status,
                            kick_source,
                        ),
                    )
                    minimum_distance = cp.minimum(
                        minimum_distance, endpoint_minimum
                    )
                    first_map, first_phase, first_reason, first_source = (
                        _record_first_failure(
                            cp,
                            first_map,
                            first_phase,
                            first_reason,
                            first_source,
                            kick_status,
                            kick_source,
                            map_index,
                            0,
                        )
                    )

                drift = drift_all_conic_cuda_batch(
                    cp,
                    positions,
                    kicked_velocity,
                    float(source_tile.source_gm_AU3_per_year2[0]),
                    full_step,
                    threads_per_block=spec.threads_per_block,
                )
                if lane_count:
                    no_source = cp.full(lane_count, -1, dtype=cp.int32)
                    first_map, first_phase, first_reason, first_source = (
                        _record_first_failure(
                            cp,
                            first_map,
                            first_phase,
                            first_reason,
                            first_source,
                            drift.status,
                            no_source,
                            map_index,
                            1,
                        )
                    )
                    maximum_iterations = cp.maximum(
                        maximum_iterations, drift.iterations
                    )

                post_velocity = cp.empty_like(velocities)
                post_status = cp.empty(lane_count, dtype=cp.int32)
                post_source = cp.empty(lane_count, dtype=cp.int32)
                post_minimum = cp.empty(lane_count, dtype=cp.float64)
                if lane_count:
                    kick(
                        (blocks,),
                        (spec.threads_per_block,),
                        (
                            drift.positions,
                            drift.velocities,
                            source_positions[end_node],
                            source_velocities[end_node],
                            source_gm,
                            source_radii,
                            relative_acceleration_bounds,
                            cp.int32(lane_count),
                            cp.int32(len(source_tile.source_ids)),
                            cp.float64(half_step),
                            cp.float64(full_step),
                            cp.float64(spec.maximum_primary_dynamical_phase),
                            cp.float64(primary_outer_boundary),
                            cp.int32(0),
                            post_velocity,
                            post_minimum,
                            post_status,
                            post_source,
                        ),
                    )
                    minimum_distance = cp.minimum(minimum_distance, post_minimum)
                    first_map, first_phase, first_reason, first_source = (
                        _record_first_failure(
                            cp,
                            first_map,
                            first_phase,
                            first_reason,
                            first_source,
                            post_status,
                            post_source,
                            map_index,
                            2,
                        )
                    )
                positions = drift.positions
                velocities = post_velocity

            if arrays.lane_count:
                failure_key = cp.where(
                    first_map >= 0,
                    (
                        (
                            first_map.astype(cp.int64) * cp.int64(3)
                            + first_phase.astype(cp.int64)
                        )
                        * cp.int64(arrays.lane_count)
                    )
                    + cp.arange(arrays.lane_count, dtype=cp.int64),
                    cp.iinfo(cp.int64).max,
                )
                lane = int(cp.argmin(failure_key).item())
                failed_map = int(first_map[lane].item())
                if failed_map >= 0:
                    phase_code = int(first_phase[lane].item())
                    reason_code = int(first_reason[lane].item())
                    source_value = int(first_source[lane].item())
                    tracer_value = int(arrays.indices[lane].item())
                    handoff = _failure_handoff(
                        state=state,
                        tile=source_tile,
                        spec=spec,
                        map_index=failed_map,
                        phase_code=phase_code,
                        reason_code=reason_code,
                        lane_index=lane,
                        tracer_index=tracer_value,
                        source_index=(source_value if source_value >= 0 else None),
                    )
                    raise CUDARestrictedWHIntegrationError(
                        "restricted-WH segment failed closed at "
                        f"source step {handoff.failure_global_source_step}: "
                        f"{handoff.reason}",
                        handoff,
                    )
                maximum_kepler_iterations = int(cp.max(maximum_iterations).item())
            else:
                maximum_kepler_iterations = 0

            output_indices = cp.array(arrays.indices, copy=True, order="C")
            # Complete every final gather/copy before exposing result buffers to
            # another stream or allowing an asynchronous CUDA error to escape.
            cp.cuda.get_current_stream().synchronize()
            output_state = replace(
                state,
                global_source_step=(
                    state.global_source_step
                    + spec.map_step_count * spec.source_steps_per_map_step
                ),
                tracer_relative_positions_AU=positions,
                tracer_relative_velocities_AU_per_year=velocities,
                tracer_indices=output_indices,
            )
            return CUDARestrictedWHResult(
                state=output_state,
                source_tile_sha256=source_tile.content_sha256 or "",
                minimum_endpoint_source_distance_AU=minimum_distance,
                maximum_kepler_iterations=maximum_kepler_iterations,
                runtime_identity=_runtime_identity(cp, resolved.device),
            )
        except CUDARestrictedWHIntegrationError:
            raise
        except CUDARestrictedWHContractError:
            raise
        except Exception as exc:
            handoff = _failure_handoff(
                state=state,
                tile=source_tile,
                spec=spec,
                map_index=map_index,
                phase_code=3,
                reason_code=-1,
                lane_index=None,
                tracer_index=None,
                source_index=None,
                reason_override=f"CUDA_RUNTIME_{type(exc).__name__}",
            )
            raise CUDARestrictedWHIntegrationError(
                "restricted-WH CUDA execution failed closed; input state is unchanged",
                handoff,
            ) from exc


def advance_cuda_restricted_wh_fused(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
) -> CUDARestrictedWHResult:
    """Advance the same quiet map as :func:`advance_cuda_restricted_wh`.

    This execution path fuses every map step in the supplied tile into one
    CUDA launch.  It intentionally uses the same binary64 kick and all-conic
    device functions as the stepwise reference path.  A failed lane records
    its first failure, the host selects the same canonical
    ``(map step, phase, lane)`` failure, and no output is committed.

    The fused path is an experimental performance implementation.  It does
    not widen the result claims or add a disk-checkpoint contract.
    """

    _validate_compatibility(state, source_tile, spec)
    resolved = resolve_backend(spec.backend)
    if resolved.name != "cupy":  # defensive; the spec already rejects NumPy.
        raise CUDARestrictedWHContractError("resolved backend must be CuPy")
    cp = resolved.xp
    ratio = spec.source_steps_per_map_step
    full_step = source_tile.source_step_years * ratio
    half_step = 0.5 * full_step
    primary_outer_boundary = (
        MAXIMUM_BINARY64
        if spec.primary_outer_boundary_AU is None
        else spec.primary_outer_boundary_AU
    )

    int32_max = int(np.iinfo(np.int32).max)
    if spec.map_step_count > int32_max:
        raise CUDARestrictedWHContractError(
            "fused map_step_count exceeds the signed CUDA int32 limit"
        )
    if ratio > int32_max:
        raise CUDARestrictedWHContractError(
            "fused source_steps_per_map_step exceeds the signed CUDA int32 limit"
        )

    with resolved.activate():
        try:
            arrays = _validate_tracer_arrays(state, resolved)
            if arrays.lane_count > int32_max:
                raise CUDARestrictedWHContractError(
                    "fused tracer lane count exceeds the signed CUDA int32 limit"
                )

            # The immutable host tile is copied once.  The fused device loop
            # then consumes its canonical boundary nodes without host traffic.
            source_positions = cp.asarray(
                source_tile.source_relative_positions_AU, dtype=cp.float64
            )
            source_velocities = cp.asarray(
                source_tile.source_relative_velocities_AU_per_year,
                dtype=cp.float64,
            )
            source_gm = cp.asarray(
                source_tile.source_gm_AU3_per_year2, dtype=cp.float64
            )
            source_radii = cp.asarray(
                source_tile.source_encounter_radii_AU, dtype=cp.float64
            )
            relative_acceleration_bounds = cp.asarray(
                source_tile.certified_relative_acceleration_bounds_AU_per_year2,
                dtype=cp.float64,
            )

            positions = cp.array(
                arrays.positions, dtype=cp.float64, copy=True, order="C"
            )
            velocities = cp.array(
                arrays.velocities, dtype=cp.float64, copy=True, order="C"
            )
            lane_count = arrays.lane_count
            minimum_distance = cp.full(lane_count, cp.inf, dtype=cp.float64)
            maximum_iterations = cp.zeros(lane_count, dtype=cp.int32)
            status = cp.zeros(lane_count, dtype=cp.int32)
            failure_map = cp.full(lane_count, -1, dtype=cp.int32)
            failure_phase = cp.full(lane_count, -1, dtype=cp.int32)
            failure_source = cp.full(lane_count, -1, dtype=cp.int32)

            if lane_count:
                kernel = cp.RawKernel(
                    FUSED_CUDA_SOURCE,
                    "jx_restricted_wh_fused_segment",
                    options=CUDA_OPTIONS,
                )
                blocks = (
                    lane_count + spec.threads_per_block - 1
                ) // spec.threads_per_block
                kernel(
                    (blocks,),
                    (spec.threads_per_block,),
                    (
                        arrays.positions,
                        arrays.velocities,
                        source_positions,
                        source_velocities,
                        source_gm,
                        source_radii,
                        relative_acceleration_bounds,
                        cp.int32(lane_count),
                        cp.int32(len(source_tile.source_ids)),
                        cp.int32(spec.map_step_count),
                        cp.int32(ratio),
                        cp.float64(half_step),
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

                failure_key = cp.where(
                    failure_map >= 0,
                    (
                        (
                            failure_map.astype(cp.int64) * cp.int64(3)
                            + failure_phase.astype(cp.int64)
                        )
                        * cp.int64(lane_count)
                    )
                    + cp.arange(lane_count, dtype=cp.int64),
                    cp.iinfo(cp.int64).max,
                )
                lane = int(cp.argmin(failure_key).item())
                failed_map = int(failure_map[lane].item())
                if failed_map >= 0:
                    phase_code = int(failure_phase[lane].item())
                    reason_code = int(status[lane].item())
                    source_value = int(failure_source[lane].item())
                    tracer_value = int(arrays.indices[lane].item())
                    handoff = _failure_handoff(
                        state=state,
                        tile=source_tile,
                        spec=spec,
                        map_index=failed_map,
                        phase_code=phase_code,
                        reason_code=reason_code,
                        lane_index=lane,
                        tracer_index=tracer_value,
                        source_index=(
                            source_value if source_value >= 0 else None
                        ),
                    )
                    raise CUDARestrictedWHIntegrationError(
                        "fused restricted-WH segment failed closed at "
                        f"source step {handoff.failure_global_source_step}: "
                        f"{handoff.reason}",
                        handoff,
                    )
                maximum_kepler_iterations = int(
                    cp.max(maximum_iterations).item()
                )
            else:
                maximum_kepler_iterations = 0

            output_indices = cp.array(arrays.indices, copy=True, order="C")
            # The result may immediately cross streams.  Surface every pending
            # kernel/copy failure before constructing the public result.
            cp.cuda.get_current_stream().synchronize()
            output_state = replace(
                state,
                global_source_step=(
                    state.global_source_step
                    + spec.map_step_count * spec.source_steps_per_map_step
                ),
                tracer_relative_positions_AU=positions,
                tracer_relative_velocities_AU_per_year=velocities,
                tracer_indices=output_indices,
            )
            return CUDARestrictedWHResult(
                state=output_state,
                source_tile_sha256=source_tile.content_sha256 or "",
                minimum_endpoint_source_distance_AU=minimum_distance,
                maximum_kepler_iterations=maximum_kepler_iterations,
                runtime_identity=_runtime_identity(
                    cp,
                    resolved.device,
                    execution_kernel_id=FUSED_KERNEL_ID,
                    execution_kernel_source_sha256=FUSED_CUDA_SOURCE_SHA256,
                ),
            )
        except CUDARestrictedWHIntegrationError:
            raise
        except CUDARestrictedWHContractError:
            raise
        except Exception as exc:
            handoff = _failure_handoff(
                state=state,
                tile=source_tile,
                spec=spec,
                map_index=0,
                phase_code=3,
                reason_code=-1,
                lane_index=None,
                tracer_index=None,
                source_index=None,
                reason_override=f"CUDA_RUNTIME_{type(exc).__name__}",
            )
            raise CUDARestrictedWHIntegrationError(
                "fused restricted-WH CUDA execution failed closed; input state "
                "is unchanged",
                handoff,
            ) from exc


__all__ = [
    "COORDINATE_SYSTEM",
    "CUDARestrictedWHContractError",
    "CUDARestrictedWHError",
    "CUDARestrictedWHFailureHandoff",
    "CUDARestrictedWHIntegrationError",
    "CUDARestrictedWHResult",
    "CUDARestrictedWHState",
    "FixedStepCUDARestrictedWHSpec",
    "FrozenSourcePathTile",
    "METHOD_ID",
    "PROJECT_SCOPE",
    "RESULT_CLAIMS",
    "SCIENTIFIC_CLASSIFICATION",
    "advance_cuda_restricted_wh",
    "advance_cuda_restricted_wh_fused",
    "source_path_tile_sha256",
    "verify_source_path_tile",
]
