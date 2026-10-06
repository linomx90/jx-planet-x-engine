"""Transactional fixed-step coordinator for the experimental long-horizon path.

This module deliberately composes, rather than replaces, the authoritative
restricted-WH source, lifecycle, and checkpoint implementations.  A segment
is first prepared entirely in memory.  A committed segment is returned only
after the low-level checkpoint layer has replayed the segment, published its
create-only record, loaded and replayed the complete chain, and reproduced the
prepared output state.

Only the fixed-lane terminal lifecycle can currently satisfy all three
requirements at once.  The owned-episode result carries a plain dynamics state
and its delta checkpoint does not carry the fixed-lane terminal lifecycle;
combining those two histories here would invent restart semantics.  Multirate
execution is similarly outside this v1 coordinator and is rejected explicitly.

Everything in this module remains EXPERIMENTAL / SCREENING_ONLY / MODEL_OUTPUT.
No successful call widens a scientific claim.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
from ..engine import Provenance
from ..restricted_wh_checkpoint import (
    LoadedRestrictedWHCheckpoint,
    dump_restricted_wh_lifecycle_checkpoint,
    load_latest_restricted_wh_lifecycle_checkpoint,
    restricted_wh_lifecycle_state_sha256,
)
from ..restricted_wh_lifecycle import (
    RestrictedWHLifecycleResult,
    RestrictedWHLifecycleState,
)
from ..restricted_wh_route import (
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
)
from ..restricted_wh_source import (
    WisdomHolmanRestrictedWHLifecycleSegmentResult,
    WisdomHolmanSourceTileSpec,
    advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source,
)
from .contracts import (
    CLAIM_CEILING,
    PROJECT_SCOPE,
    PUBLIC_STABILITY,
    SCIENTIFIC_CLASSIFICATION,
    LongHorizonCheckpointContract,
    LongHorizonResultContract,
)
from .restricted_wh import (
    CANONICAL_ARCHITECTURE_ID,
    adapt_restricted_wh_checkpoint,
    adapt_restricted_wh_result,
)


RUNNER_SCHEMA = "jxplanetx.long-horizon.fixed-lifecycle-runner.v1"
RUNNER_ID = "JX_LONG_HORIZON_FIXED_LIFECYCLE_RUNNER_V1"
EXECUTION_MODE = "FIXED_LANE_LIFECYCLE"
CHECKPOINT_POLICY = "REPLAY_VERIFY_EVERY_COMMIT"
RUNNER_IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

OWNED_EPISODE_UNSUPPORTED_REASON = (
    "owned-episode results do not carry the fixed-lane terminal lifecycle, "
    "so lifecycle and owned-episode delta histories cannot yet share one "
    "honest restart authority"
)
MULTIRATE_UNSUPPORTED_REASON = (
    "multirate scheduling has no lifecycle checkpoint composition in the "
    "experimental long-horizon runner"
)


class LongHorizonRunnerError(RuntimeError):
    """Base error for coordinator-level failures."""


class LongHorizonRunnerContractError(LongHorizonRunnerError, ValueError):
    """A runner request is ambiguous or outside the supported v1 envelope."""


class LongHorizonRunnerIntegrityError(LongHorizonRunnerError):
    """Prepared, published, and replay-loaded identities disagree."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise LongHorizonRunnerContractError(
            f"{label} must be a nonempty trimmed string"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise LongHorizonRunnerContractError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LongHorizonRunnerContractError(
            f"{label} must be a finite built-in float"
        )
    return value


def _plain_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_tree(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise LongHorizonRunnerContractError(
                "identity metadata contains a nonfinite float"
            )
        return {"float_hex": value.hex()}
    if value is None or type(value) in (str, bool, int):
        return value
    raise LongHorizonRunnerContractError(
        f"identity metadata contains unsupported {type(value).__name__}"
    )


def _freeze_tree(value: Any, label: str = "metadata") -> Any:
    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            _text(key, f"{label} key")
            if key in frozen:
                raise LongHorizonRunnerContractError(
                    f"{label} contains a duplicate key"
                )
            frozen[key] = _freeze_tree(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if type(value) in (list, tuple):
        return tuple(
            _freeze_tree(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise LongHorizonRunnerContractError(
        f"{label} contains unsupported {type(value).__name__}"
    )


def _canonical_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_tree(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + payload).hexdigest()


def _backend_identity(spec: FixedStepCUDARestrictedWHSpec) -> Mapping[str, Any]:
    backend = spec.backend
    return {
        "backend_id": backend.backend_id,
        "device": backend.device,
        "tile_size": backend.tile_size,
        "dtype": backend.dtype,
        "allow_fallback": backend.allow_fallback,
        "deterministic_reductions": backend.deterministic_reductions,
        "fast_math": backend.fast_math,
        "determinism_scope": backend.determinism_scope,
    }


def _claims(value: object) -> Mapping[str, bool]:
    if not isinstance(value, MappingABC) or dict(value) != dict(CLAIM_CEILING):
        raise LongHorizonRunnerContractError("runner claim ceiling changed")
    if any(type(item) is not bool or item for item in value.values()):
        raise LongHorizonRunnerContractError(
            "every runner claim must remain exact false"
        )
    return MappingProxyType(dict(CLAIM_CEILING))


def _owned_float64_array(value: object, label: str, *, ndim: int) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != ndim
        or not value.flags.c_contiguous
        or not bool(np.all(np.isfinite(value)))
    ):
        raise LongHorizonRunnerContractError(
            f"{label} must be a finite C-contiguous float64 array of rank {ndim}"
        )
    copied = np.array(value, dtype=np.float64, order="C", copy=True)
    # A bytes-backed reconstruction cannot be made writeable later with
    # ``setflags(write=True)``; a merely read-only owned array can.
    return np.frombuffer(copied.tobytes(order="C"), dtype=np.float64).reshape(
        copied.shape
    )


def _source_material_sha256(source: "FixedLongHorizonSourceSegment") -> str:
    metadata = {
        "schema": "jxplanetx.long-horizon.fixed-source-segment.v1",
        "tile_id": source.tile_id,
        "context": {
            "epoch_origin_years": source.epoch_origin_years,
            "global_source_step": source.global_source_step,
            "source_step_years": source.source_step_years,
            "time_scale": source.time_scale,
            "frame": source.frame,
            "axes": source.axes,
            "coordinate_system": source.coordinate_system,
            "length_unit": source.length_unit,
            "time_unit": source.time_unit,
            "unit_system_id": source.unit_system_id,
            "source_ids": source.source_ids,
            "primary_source_id": source.primary_source_id,
            "source_schedule_sha256": source.source_schedule_sha256,
        },
        "provenance": {
            "source_id": source.source_provenance.source_id,
            "citation": source.source_provenance.citation,
            "version": source.source_provenance.version,
            "sha256": source.source_provenance.sha256,
        },
        "arrays": tuple(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": tuple(int(item) for item in array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
            for name, array in source._arrays()
        ),
    }
    return _canonical_sha256(
        b"jxplanetx.long-horizon.fixed-source-segment.v1\0", metadata
    )


@dataclass(frozen=True, eq=False)
class FixedLongHorizonRunConfig:
    """Exact, content-bound policy for one fixed-size checkpoint segment."""

    run_id: str
    source_spec: WisdomHolmanSourceTileSpec
    tracer_spec: FixedStepCUDARestrictedWHSpec
    route_spec: RestrictedWHRouteSpec | None = None
    route_worker_count: int = 6
    execution_mode: str = EXECUTION_MODE
    multirate_enabled: bool = False
    owned_episode_enabled: bool = False
    public_stability: str = PUBLIC_STABILITY
    project_scope: str = PROJECT_SCOPE
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)
    schema: str = RUNNER_SCHEMA
    protocol_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if type(self.source_spec) is not WisdomHolmanSourceTileSpec:
            raise LongHorizonRunnerContractError(
                "source_spec must be an exact WisdomHolmanSourceTileSpec"
            )
        if type(self.tracer_spec) is not FixedStepCUDARestrictedWHSpec:
            raise LongHorizonRunnerContractError(
                "tracer_spec must be an exact FixedStepCUDARestrictedWHSpec"
            )
        self.source_spec.__post_init__()
        self.tracer_spec.__post_init__()
        required_source_steps = (
            self.tracer_spec.map_step_count
            * self.tracer_spec.source_steps_per_map_step
        )
        if self.source_spec.source_step_count != required_source_steps:
            raise LongHorizonRunnerContractError(
                "source_spec must cover exactly one tracer segment"
            )
        if self.route_spec is None:
            route_spec = RestrictedWHRouteSpec(
                maximum_primary_dynamical_phase=(
                    self.tracer_spec.maximum_primary_dynamical_phase
                ),
                primary_outer_boundary_AU=(
                    self.tracer_spec.primary_outer_boundary_AU
                ),
            )
            object.__setattr__(self, "route_spec", route_spec)
        elif type(self.route_spec) is not RestrictedWHRouteSpec:
            raise LongHorizonRunnerContractError(
                "route_spec must be an exact RestrictedWHRouteSpec or None"
            )
        assert self.route_spec is not None
        self.route_spec.__post_init__()
        if (
            self.route_spec.maximum_primary_dynamical_phase
            != self.tracer_spec.maximum_primary_dynamical_phase
            or self.route_spec.primary_outer_boundary_AU
            != self.tracer_spec.primary_outer_boundary_AU
        ):
            raise LongHorizonRunnerContractError(
                "route and tracer primary guard policies must be identical"
            )
        if (
            type(self.route_worker_count) is not int
            or not 1 <= self.route_worker_count <= 32
        ):
            raise LongHorizonRunnerContractError(
                "route_worker_count must be an integer in [1, 32]"
            )
        if self.execution_mode != EXECUTION_MODE:
            raise LongHorizonRunnerContractError(
                f"execution_mode must remain {EXECUTION_MODE}"
            )
        if type(self.multirate_enabled) is not bool:
            raise LongHorizonRunnerContractError(
                "multirate_enabled must be an exact bool"
            )
        if self.multirate_enabled:
            raise LongHorizonRunnerContractError(MULTIRATE_UNSUPPORTED_REASON)
        if type(self.owned_episode_enabled) is not bool:
            raise LongHorizonRunnerContractError(
                "owned_episode_enabled must be an exact bool"
            )
        if self.owned_episode_enabled:
            raise LongHorizonRunnerContractError(
                OWNED_EPISODE_UNSUPPORTED_REASON
            )
        if (
            self.public_stability != PUBLIC_STABILITY
            or self.project_scope != PROJECT_SCOPE
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.schema != RUNNER_SCHEMA
        ):
            raise LongHorizonRunnerContractError(
                "runner scope or schema identity changed"
            )
        object.__setattr__(self, "claims", _claims(self.claims))
        object.__setattr__(
            self,
            "protocol_sha256",
            _canonical_sha256(
                b"jxplanetx.long-horizon.fixed-lifecycle-protocol.v1\0",
                self._protocol_record(),
            ),
        )

    def _protocol_record(self) -> Mapping[str, Any]:
        assert self.route_spec is not None
        source_spec = {
            "source_step_years": self.source_spec.source_step_years,
            "source_step_count": self.source_spec.source_step_count,
            "minimum_encounter_pair_separation_AU": (
                self.source_spec.minimum_encounter_pair_separation_AU
            ),
            "minimum_jacobi_periapse_AU": (
                self.source_spec.minimum_jacobi_periapse_AU
            ),
            "maximum_initial_barycenter_position_norm_AU": (
                self.source_spec.maximum_initial_barycenter_position_norm_AU
            ),
            "maximum_initial_barycenter_velocity_norm_AU_per_year": (
                self.source_spec.maximum_initial_barycenter_velocity_norm_AU_per_year
            ),
        }
        tracer_spec = {
            "map_step_count": self.tracer_spec.map_step_count,
            "source_steps_per_map_step": (
                self.tracer_spec.source_steps_per_map_step
            ),
            "maximum_primary_dynamical_phase": (
                self.tracer_spec.maximum_primary_dynamical_phase
            ),
            "threads_per_block": self.tracer_spec.threads_per_block,
            "primary_outer_boundary_AU": (
                self.tracer_spec.primary_outer_boundary_AU
            ),
            "backend": _backend_identity(self.tracer_spec),
        }
        return {
            "schema": self.schema,
            "runner_id": RUNNER_ID,
            "runner_implementation_source_sha256": (
                RUNNER_IMPLEMENTATION_SOURCE_SHA256
            ),
            "architecture_id": CANONICAL_ARCHITECTURE_ID,
            "run_id": self.run_id,
            "execution_mode": self.execution_mode,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "source_spec": source_spec,
            "tracer_spec": tracer_spec,
            "route_policy": restricted_wh_route_policy_identity(
                self.route_spec
            ),
            "route_worker_count": self.route_worker_count,
            "multirate_enabled": self.multirate_enabled,
            "owned_episode_enabled": self.owned_episode_enabled,
            "public_stability": self.public_stability,
            "project_scope": self.project_scope,
            "scientific_classification": self.scientific_classification,
            "claims": self.claims,
        }


@dataclass(frozen=True, eq=False)
class FixedLongHorizonSourceSegment:
    """Owned, immutable massive-source boundary data for one segment."""

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
    initial_source_relative_positions_AU: np.ndarray
    initial_source_relative_velocities_AU_per_year: np.ndarray
    source_gm_AU3_per_year2: np.ndarray
    source_encounter_radii_AU: np.ndarray
    certified_relative_acceleration_bounds_AU_per_year2: np.ndarray
    source_provenance: Provenance
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.tile_id, "tile_id")
        _finite_float(self.epoch_origin_years, "epoch_origin_years")
        if type(self.global_source_step) is not int or self.global_source_step < 0:
            raise LongHorizonRunnerContractError(
                "global_source_step must be a nonnegative integer"
            )
        if _finite_float(self.source_step_years, "source_step_years") == 0.0:
            raise LongHorizonRunnerContractError(
                "source_step_years must be nonzero"
            )
        for value, label in (
            (self.time_scale, "time_scale"),
            (self.frame, "frame"),
            (self.axes, "axes"),
            (self.coordinate_system, "coordinate_system"),
            (self.length_unit, "length_unit"),
            (self.time_unit, "time_unit"),
            (self.unit_system_id, "unit_system_id"),
            (self.primary_source_id, "primary_source_id"),
        ):
            _text(value, label)
        if (
            type(self.source_ids) is not tuple
            or not self.source_ids
            or any(type(item) is not str or not item for item in self.source_ids)
            or len(set(self.source_ids)) != len(self.source_ids)
            or self.source_ids[0] != self.primary_source_id
        ):
            raise LongHorizonRunnerContractError(
                "source_ids must be a unique exact tuple led by primary_source_id"
            )
        _sha256(self.source_schedule_sha256, "source_schedule_sha256")
        if type(self.source_provenance) is not Provenance:
            raise LongHorizonRunnerContractError(
                "source_provenance must be an exact Provenance"
            )
        positions = _owned_float64_array(
            self.initial_source_relative_positions_AU,
            "initial_source_relative_positions_AU",
            ndim=2,
        )
        velocities = _owned_float64_array(
            self.initial_source_relative_velocities_AU_per_year,
            "initial_source_relative_velocities_AU_per_year",
            ndim=2,
        )
        gm = _owned_float64_array(
            self.source_gm_AU3_per_year2,
            "source_gm_AU3_per_year2",
            ndim=1,
        )
        radii = _owned_float64_array(
            self.source_encounter_radii_AU,
            "source_encounter_radii_AU",
            ndim=1,
        )
        bounds = _owned_float64_array(
            self.certified_relative_acceleration_bounds_AU_per_year2,
            "certified_relative_acceleration_bounds_AU_per_year2",
            ndim=1,
        )
        source_count = int(gm.shape[0])
        if (
            source_count <= 0
            or source_count != len(self.source_ids)
            or positions.shape != (source_count, 3)
            or velocities.shape != (source_count, 3)
            or radii.shape != (source_count,)
            or bounds.shape != (source_count,)
        ):
            raise LongHorizonRunnerContractError(
                "source segment arrays have inconsistent shapes"
            )
        if not bool(np.all(positions[0] == 0.0)) or not bool(
            np.all(velocities[0] == 0.0)
        ):
            raise LongHorizonRunnerContractError(
                "source slot zero position and velocity must be exact zero"
            )
        if (
            not bool(np.all(gm > 0.0))
            or not bool(np.all(radii > 0.0))
            or not bool(np.all(bounds > 0.0))
        ):
            raise LongHorizonRunnerContractError(
                "source GM, encounter radii, and acceleration bounds must be positive"
            )
        for name, value in (
            ("initial_source_relative_positions_AU", positions),
            ("initial_source_relative_velocities_AU_per_year", velocities),
            ("source_gm_AU3_per_year2", gm),
            ("source_encounter_radii_AU", radii),
            ("certified_relative_acceleration_bounds_AU_per_year2", bounds),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "content_sha256", _source_material_sha256(self))

    def _arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            (
                "initial_source_relative_positions_AU",
                self.initial_source_relative_positions_AU,
            ),
            (
                "initial_source_relative_velocities_AU_per_year",
                self.initial_source_relative_velocities_AU_per_year,
            ),
            ("source_gm_AU3_per_year2", self.source_gm_AU3_per_year2),
            ("source_encounter_radii_AU", self.source_encounter_radii_AU),
            (
                "certified_relative_acceleration_bounds_AU_per_year2",
                self.certified_relative_acceleration_bounds_AU_per_year2,
            ),
        )


@dataclass(frozen=True, eq=False)
class LongHorizonPreparedSegment:
    """One complete in-memory segment that has not been checkpoint-published."""

    config_protocol_sha256: str
    source_content_sha256: str
    input_state_sha256: str
    output_state_sha256: str
    input_state: RestrictedWHLifecycleState
    authoritative_result: WisdomHolmanRestrictedWHLifecycleSegmentResult
    result_contract: LongHorizonResultContract
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)

    def __post_init__(self) -> None:
        for value, label in (
            (self.config_protocol_sha256, "config_protocol_sha256"),
            (self.source_content_sha256, "source_content_sha256"),
            (self.input_state_sha256, "input_state_sha256"),
            (self.output_state_sha256, "output_state_sha256"),
        ):
            _sha256(value, label)
        if type(self.input_state) is not RestrictedWHLifecycleState:
            raise LongHorizonRunnerContractError(
                "input_state must be an exact RestrictedWHLifecycleState"
            )
        if (
            type(self.authoritative_result)
            is not WisdomHolmanRestrictedWHLifecycleSegmentResult
        ):
            raise LongHorizonRunnerContractError(
                "authoritative_result has the wrong exact type"
            )
        if type(self.result_contract) is not LongHorizonResultContract:
            raise LongHorizonRunnerContractError(
                "result_contract has the wrong exact type"
            )
        expected_result_contract = adapt_restricted_wh_result(
            self.authoritative_result
        )
        if (
            self.result_contract.source_tile_sha256
            != self.authoritative_result.source_build.tile.content_sha256
            or self.result_contract.global_step
            != self.authoritative_result.tracer_result.state.dynamics_state.global_source_step
            or self.result_contract.output_state_sha256
            != expected_result_contract.output_state_sha256
            or self.result_contract.content_sha256
            != expected_result_contract.content_sha256
        ):
            raise LongHorizonRunnerIntegrityError(
                "result façade is not bound to the authoritative segment"
            )
        object.__setattr__(self, "claims", _claims(self.claims))


@dataclass(frozen=True, eq=False)
class LongHorizonCommittedSegment:
    """Prepared segment released after checkpoint publication and full replay."""

    prepared: LongHorizonPreparedSegment
    loaded_checkpoint: LoadedRestrictedWHCheckpoint
    checkpoint_contract: LongHorizonCheckpointContract
    checkpoint_record: Mapping[str, Any]
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)

    def __post_init__(self) -> None:
        if type(self.prepared) is not LongHorizonPreparedSegment:
            raise LongHorizonRunnerContractError(
                "prepared must be an exact LongHorizonPreparedSegment"
            )
        if type(self.loaded_checkpoint) is not LoadedRestrictedWHCheckpoint:
            raise LongHorizonRunnerContractError(
                "loaded_checkpoint has the wrong exact type"
            )
        if type(self.checkpoint_contract) is not LongHorizonCheckpointContract:
            raise LongHorizonRunnerContractError(
                "checkpoint_contract has the wrong exact type"
            )
        if not isinstance(self.checkpoint_record, MappingABC):
            raise LongHorizonRunnerContractError(
                "checkpoint_record must be a mapping"
            )
        if (
            self.loaded_checkpoint.state_sha256 != self.prepared.output_state_sha256
            or self.checkpoint_contract.state_sha256
            != self.prepared.output_state_sha256
            or self.checkpoint_contract.checkpoint_content_sha256
            != self.loaded_checkpoint.checkpoint_content_sha256
            or self.checkpoint_record.get("checkpoint_content_sha256")
            != self.loaded_checkpoint.checkpoint_content_sha256
        ):
            raise LongHorizonRunnerIntegrityError(
                "checkpoint replay is not bound to the prepared segment"
            )
        object.__setattr__(
            self,
            "checkpoint_record",
            _freeze_tree(self.checkpoint_record, "checkpoint_record"),
        )
        object.__setattr__(self, "claims", _claims(self.claims))


def _validate_state_against_config(
    state: RestrictedWHLifecycleState,
    config: FixedLongHorizonRunConfig,
) -> None:
    if type(state) is not RestrictedWHLifecycleState:
        raise LongHorizonRunnerContractError(
            "state must be an exact RestrictedWHLifecycleState"
        )
    if type(config) is not FixedLongHorizonRunConfig:
        raise LongHorizonRunnerContractError(
            "config must be an exact FixedLongHorizonRunConfig"
        )
    if state.dynamics_state.source_step_years != config.source_spec.source_step_years:
        raise LongHorizonRunnerContractError(
            "state and source_spec must use the identical source step"
        )


def _validate_source_against_state(
    source: FixedLongHorizonSourceSegment,
    state: RestrictedWHLifecycleState,
    config: FixedLongHorizonRunConfig,
) -> None:
    dynamics = state.dynamics_state
    expected = (
        source.epoch_origin_years == dynamics.epoch_origin_years
        and source.global_source_step == dynamics.global_source_step
        and source.source_step_years == dynamics.source_step_years
        and source.source_step_years == config.source_spec.source_step_years
        and source.time_scale == dynamics.time_scale
        and source.frame == dynamics.frame
        and source.axes == dynamics.axes
        and source.coordinate_system == dynamics.coordinate_system
        and source.length_unit == dynamics.length_unit
        and source.time_unit == dynamics.time_unit
        and source.unit_system_id == dynamics.unit_system_id
        and source.source_ids == dynamics.source_ids
        and source.primary_source_id == dynamics.primary_source_id
        and source.source_schedule_sha256 == dynamics.source_schedule_sha256
    )
    if not expected:
        raise LongHorizonRunnerContractError(
            "source segment context differs from the exact dynamics state"
        )


def advance_fixed_lifecycle_segment(
    state: RestrictedWHLifecycleState,
    *,
    config: FixedLongHorizonRunConfig,
    source: FixedLongHorizonSourceSegment,
) -> LongHorizonPreparedSegment:
    """Prepare one fixed-lane segment without publishing a checkpoint.

    The input lifecycle state is hashed before and after execution.  Any
    mutation by a lower layer fails the coordinator instead of becoming
    restart authority.
    """

    _validate_state_against_config(state, config)
    if type(source) is not FixedLongHorizonSourceSegment:
        raise LongHorizonRunnerContractError(
            "source must be an exact FixedLongHorizonSourceSegment"
        )
    _validate_source_against_state(source, state, config)
    input_digest = restricted_wh_lifecycle_state_sha256(
        state, config.tracer_spec.backend
    )
    result = advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source(
        state,
        tile_id=source.tile_id,
        initial_source_relative_positions_AU=(
            source.initial_source_relative_positions_AU
        ),
        initial_source_relative_velocities_AU_per_year=(
            source.initial_source_relative_velocities_AU_per_year
        ),
        source_gm_AU3_per_year2=source.source_gm_AU3_per_year2,
        source_encounter_radii_AU=source.source_encounter_radii_AU,
        certified_relative_acceleration_bounds_AU_per_year2=(
            source.certified_relative_acceleration_bounds_AU_per_year2
        ),
        source_provenance=source.source_provenance,
        source_spec=config.source_spec,
        tracer_spec=config.tracer_spec,
        route_spec=config.route_spec,
        route_worker_count=config.route_worker_count,
    )
    if (
        restricted_wh_lifecycle_state_sha256(
            state, config.tracer_spec.backend
        )
        != input_digest
    ):
        raise LongHorizonRunnerIntegrityError(
            "segment execution mutated its caller-owned input state"
        )
    if result.source_build.tile.global_source_step != (
        state.dynamics_state.global_source_step
    ):
        raise LongHorizonRunnerIntegrityError(
            "source build started on a different lattice node"
        )
    result_contract = adapt_restricted_wh_result(result)
    output_digest = restricted_wh_lifecycle_state_sha256(
        result.tracer_result.state, config.tracer_spec.backend
    )
    return LongHorizonPreparedSegment(
        config_protocol_sha256=config.protocol_sha256,
        source_content_sha256=source.content_sha256,
        input_state_sha256=input_digest,
        output_state_sha256=output_digest,
        input_state=state,
        authoritative_result=result,
        result_contract=result_contract,
    )


def _commit_prepared_segment(
    prepared: LongHorizonPreparedSegment,
    *,
    config: FixedLongHorizonRunConfig,
    checkpoint_directory: str | Path,
    genesis_state_sha256: str,
    checkpoint_index: int,
    previous_checkpoint_content_sha256: str | None,
) -> LongHorizonCommittedSegment:
    if type(prepared) is not LongHorizonPreparedSegment:
        raise LongHorizonRunnerContractError(
            "prepared must be an exact LongHorizonPreparedSegment"
        )
    if type(config) is not FixedLongHorizonRunConfig:
        raise LongHorizonRunnerContractError(
            "config must be an exact FixedLongHorizonRunConfig"
        )
    if prepared.config_protocol_sha256 != config.protocol_sha256:
        raise LongHorizonRunnerContractError(
            "prepared segment belongs to a different protocol"
        )
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    if type(checkpoint_index) is not int or checkpoint_index < 0:
        raise LongHorizonRunnerContractError(
            "checkpoint_index must be a nonnegative integer"
        )
    if previous_checkpoint_content_sha256 is not None:
        _sha256(
            previous_checkpoint_content_sha256,
            "previous_checkpoint_content_sha256",
        )
    if (
        restricted_wh_lifecycle_state_sha256(
            prepared.input_state, config.tracer_spec.backend
        )
        != prepared.input_state_sha256
    ):
        raise LongHorizonRunnerIntegrityError(
            "prepared input state changed before checkpoint commit"
        )
    result: RestrictedWHLifecycleResult = prepared.authoritative_result.tracer_result
    source_tile = prepared.authoritative_result.source_build.tile
    record = dump_restricted_wh_lifecycle_checkpoint(
        checkpoint_directory,
        run_id=config.run_id,
        protocol_sha256=config.protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        checkpoint_index=checkpoint_index,
        input_state=prepared.input_state,
        result=result,
        source_tile=source_tile,
        spec=config.tracer_spec,
        route_spec=config.route_spec,
        route_worker_count=config.route_worker_count,
        previous_checkpoint_content_sha256=(
            previous_checkpoint_content_sha256
        ),
    )
    record_index = record.get("checkpoint_index")
    record_digest = record.get("checkpoint_content_sha256")
    if record_index != checkpoint_index or type(record_digest) is not str:
        raise LongHorizonRunnerIntegrityError(
            "checkpoint publication returned an invalid head anchor"
        )
    loaded = load_latest_restricted_wh_lifecycle_checkpoint(
        checkpoint_directory,
        run_id=config.run_id,
        protocol_sha256=config.protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        expected_checkpoint_index=checkpoint_index,
        expected_checkpoint_content_sha256=record_digest,
        backend=config.tracer_spec.backend,
    )
    if loaded.state_sha256 != prepared.output_state_sha256:
        raise LongHorizonRunnerIntegrityError(
            "checkpoint replay changed the prepared output state"
        )
    checkpoint_contract = adapt_restricted_wh_checkpoint(loaded)
    return LongHorizonCommittedSegment(
        prepared=prepared,
        loaded_checkpoint=loaded,
        checkpoint_contract=checkpoint_contract,
        checkpoint_record=record,
    )


def start_fixed_lifecycle_run(
    state: RestrictedWHLifecycleState,
    *,
    config: FixedLongHorizonRunConfig,
    source: FixedLongHorizonSourceSegment,
    checkpoint_directory: str | Path,
) -> LongHorizonCommittedSegment:
    """Prepare, replay-verify, publish, and reload checkpoint zero."""

    _validate_state_against_config(state, config)
    genesis_digest = restricted_wh_lifecycle_state_sha256(
        state, config.tracer_spec.backend
    )
    prepared = advance_fixed_lifecycle_segment(
        state, config=config, source=source
    )
    if prepared.input_state_sha256 != genesis_digest:
        raise LongHorizonRunnerIntegrityError(
            "prepared checkpoint zero changed the genesis identity"
        )
    return _commit_prepared_segment(
        prepared,
        config=config,
        checkpoint_directory=checkpoint_directory,
        genesis_state_sha256=genesis_digest,
        checkpoint_index=0,
        previous_checkpoint_content_sha256=None,
    )


def resume_fixed_lifecycle_run(
    *,
    config: FixedLongHorizonRunConfig,
    source: FixedLongHorizonSourceSegment,
    checkpoint_directory: str | Path,
    genesis_state_sha256: str,
    expected_checkpoint_index: int,
    expected_checkpoint_content_sha256: str,
) -> LongHorizonCommittedSegment:
    """Load an externally anchored chain, then append exactly one segment."""

    if type(config) is not FixedLongHorizonRunConfig:
        raise LongHorizonRunnerContractError(
            "config must be an exact FixedLongHorizonRunConfig"
        )
    if type(source) is not FixedLongHorizonSourceSegment:
        raise LongHorizonRunnerContractError(
            "source must be an exact FixedLongHorizonSourceSegment"
        )
    _sha256(genesis_state_sha256, "genesis_state_sha256")
    if type(expected_checkpoint_index) is not int or expected_checkpoint_index < 0:
        raise LongHorizonRunnerContractError(
            "expected_checkpoint_index must be a nonnegative integer"
        )
    _sha256(
        expected_checkpoint_content_sha256,
        "expected_checkpoint_content_sha256",
    )
    predecessor = load_latest_restricted_wh_lifecycle_checkpoint(
        checkpoint_directory,
        run_id=config.run_id,
        protocol_sha256=config.protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        expected_checkpoint_index=expected_checkpoint_index,
        expected_checkpoint_content_sha256=(
            expected_checkpoint_content_sha256
        ),
        backend=config.tracer_spec.backend,
    )
    predecessor_contract = adapt_restricted_wh_checkpoint(predecessor)
    if (
        predecessor_contract.checkpoint_index != expected_checkpoint_index
        or predecessor_contract.checkpoint_content_sha256
        != expected_checkpoint_content_sha256
    ):
        raise LongHorizonRunnerIntegrityError(
            "loaded predecessor differs from its external anchor"
        )
    prepared = advance_fixed_lifecycle_segment(
        predecessor.state, config=config, source=source
    )
    if prepared.input_state_sha256 != predecessor.state_sha256:
        raise LongHorizonRunnerIntegrityError(
            "resumed segment did not start from the verified checkpoint state"
        )
    return _commit_prepared_segment(
        prepared,
        config=config,
        checkpoint_directory=checkpoint_directory,
        genesis_state_sha256=genesis_state_sha256,
        checkpoint_index=expected_checkpoint_index + 1,
        previous_checkpoint_content_sha256=(
            expected_checkpoint_content_sha256
        ),
    )


__all__ = [
    "CHECKPOINT_POLICY",
    "EXECUTION_MODE",
    "FixedLongHorizonRunConfig",
    "FixedLongHorizonSourceSegment",
    "LongHorizonCommittedSegment",
    "LongHorizonPreparedSegment",
    "LongHorizonRunnerContractError",
    "LongHorizonRunnerError",
    "LongHorizonRunnerIntegrityError",
    "MULTIRATE_UNSUPPORTED_REASON",
    "OWNED_EPISODE_UNSUPPORTED_REASON",
    "RUNNER_ID",
    "RUNNER_IMPLEMENTATION_SOURCE_SHA256",
    "RUNNER_SCHEMA",
    "advance_fixed_lifecycle_segment",
    "resume_fixed_lifecycle_run",
    "start_fixed_lifecycle_run",
]
