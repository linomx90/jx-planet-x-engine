"""Experimental fixed-rung multirate restricted-WH CUDA scheduler.

This module is an additive precision prototype.  It groups tracers by a
predeclared, immutable source-step ratio and advances every group to one common
fine-lattice endpoint with the already qualified fixed-step fused CUDA map.
The source path and tracer ordering are shared by every group; only the number
of fine source intervals consumed by one tracer map step differs.

The first milestone deliberately forbids state-dependent or intra-segment
rung changes.  That restriction avoids presenting an uncorrected adaptive
step switch as symplectic.  A later transition method must have its own
identity and convergence evidence.  Until then, callers must create a new
explicit plan and qualification lineage rather than silently changing rungs.

No production, long-horizon accuracy, encounter, completion, or science claim
is made.  A failure in any group releases no partial multirate result.
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

from . import cuda_restricted_wh as _fixed
from .cuda_restricted_wh import (
    CUDARestrictedWHFailureHandoff,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    verify_source_path_tile,
)
from .engine.backends import BackendArrayError, resolve_backend
from .engine.contracts import BackendSpec, Provenance


METHOD_ID = "JX_CUDA_RESTRICTED_WH_STATIC_MULTIRATE_V1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
PLAN_DOMAIN = b"jxplanetx.cuda-restricted-wh.static-rung-plan.v1\0"
INPUT_STATE_DOMAIN = (
    b"jxplanetx.cuda-restricted-wh.static-rung-input-state.v1\0"
)
EXECUTION_POLICY_DOMAIN = (
    b"jxplanetx.cuda-restricted-wh.static-multirate-policy.v1\0"
)
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAXIMUM_RUNG_RATIO = 1024
STATIC_MULTIRATE_KERNEL_ID = "jx.cuda.restricted-wh.static-multirate.fused.v1"

_MR_CLEARANCE = 101
_MR_RELATIVE_SWEEP = 102
_MR_ALL_SOURCE_PHASE = 103
_MR_STATUS_NAMES = {
    _MR_CLEARANCE: "MULTIRATE_CLEARANCE_GUARD",
    _MR_RELATIVE_SWEEP: "MULTIRATE_RELATIVE_SWEEP_GUARD",
    _MR_ALL_SOURCE_PHASE: "MULTIRATE_ALL_SOURCE_PHASE_GUARD",
}

RESULT_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "four_point_five_gyr_accuracy_qualified": False,
        "dynamic_rung_transition_qualified": False,
        "symplectic_corrector_qualified": False,
        "globally_symplectic_with_external_source_tile": False,
        "close_encounter_qualified": False,
        "completion_guaranteed": False,
        "science_result": False,
    }
)


class CUDARestrictedWHMultirateError(RuntimeError):
    """Base error for the experimental static-rung scheduler."""


class CUDARestrictedWHMultirateContractError(
    CUDARestrictedWHMultirateError, ValueError
):
    """A multirate request violates its explicit immutable contract."""


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHMultirateFailureHandoff:
    """Transactional failure referring to the unchanged complete input."""

    state: CUDARestrictedWHState
    rung_plan_sha256: str
    failed_source_steps_per_map_step: int | None
    fixed_step_handoff: CUDARestrictedWHFailureHandoff | None


class CUDARestrictedWHMultirateIntegrationError(
    CUDARestrictedWHMultirateError
):
    """One group failed and the complete multirate segment was discarded."""

    def __init__(
        self,
        message: str,
        handoff: CUDARestrictedWHMultirateFailureHandoff,
    ) -> None:
        super().__init__(message)
        self.handoff = handoff


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CUDARestrictedWHMultirateContractError(
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
        raise CUDARestrictedWHMultirateContractError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _frozen_array(
    value: object,
    label: str,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype or value.ndim != 1:
        raise CUDARestrictedWHMultirateContractError(
            f"{label} must be an exact one-dimensional {dtype.name} NumPy array"
        )
    canonical = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(canonical.tobytes(order="C"), dtype=dtype)


def _allowed_ratios(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise CUDARestrictedWHMultirateContractError(
            "allowed_source_steps_per_map_step must be a nonempty tuple"
        )
    if value[0] != 1:
        raise CUDARestrictedWHMultirateContractError(
            "the finest allowed source-step ratio must be one"
        )
    previous = 0
    for index, ratio in enumerate(value):
        if (
            type(ratio) is not int
            or ratio <= previous
            or ratio > MAXIMUM_RUNG_RATIO
            or ratio & (ratio - 1)
        ):
            raise CUDARestrictedWHMultirateContractError(
                "allowed ratios must be strictly increasing powers of two "
                f"not exceeding {MAXIMUM_RUNG_RATIO}; invalid item {index}"
            )
        previous = ratio
    return value


@dataclass(frozen=True, eq=False)
class FrozenStaticRungPlan:
    """Immutable per-tracer rung assignment for one complete segment.

    The assignment is fixed for the full segment.  Ratios are powers of two
    on the authoritative fine source lattice, divide the segment horizon, and
    begin on a matching absolute block boundary.  The plan binds the exact
    input-state bytes and execution policy for which its external accuracy
    evidence was prepared.  The evidence digest remains a provenance pointer;
    this experimental module does not infer scientific validity from it.
    """

    plan_id: str
    global_source_step: int
    total_source_steps: int
    source_schedule_sha256: str
    source_tile_sha256: str
    tracer_indices: np.ndarray
    source_steps_per_map_step: np.ndarray
    allowed_source_steps_per_map_step: tuple[int, ...]
    selection_basis: str
    accuracy_policy_id: str
    accuracy_validation_sha256: str
    input_state_sha256: str
    execution_policy_sha256: str
    provenance: Provenance
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        if type(self.global_source_step) is not int or self.global_source_step < 0:
            raise CUDARestrictedWHMultirateContractError(
                "global_source_step must be a nonnegative integer"
            )
        if type(self.total_source_steps) is not int or self.total_source_steps <= 0:
            raise CUDARestrictedWHMultirateContractError(
                "total_source_steps must be a positive integer"
            )
        _sha256(self.source_schedule_sha256, "source_schedule_sha256")
        _sha256(self.source_tile_sha256, "source_tile_sha256")
        _text(self.selection_basis, "selection_basis")
        _text(self.accuracy_policy_id, "accuracy_policy_id")
        _sha256(
            self.accuracy_validation_sha256,
            "accuracy_validation_sha256",
        )
        _sha256(self.input_state_sha256, "input_state_sha256")
        _sha256(self.execution_policy_sha256, "execution_policy_sha256")
        if type(self.provenance) is not Provenance:
            raise CUDARestrictedWHMultirateContractError(
                "provenance must be an exact Provenance"
            )
        allowed = _allowed_ratios(self.allowed_source_steps_per_map_step)
        indices = _frozen_array(
            self.tracer_indices,
            "tracer_indices",
            dtype=np.dtype("<u8"),
        )
        ratios = _frozen_array(
            self.source_steps_per_map_step,
            "source_steps_per_map_step",
            dtype=np.dtype("<u4"),
        )
        if ratios.shape != indices.shape:
            raise CUDARestrictedWHMultirateContractError(
                "rung assignment must exactly cover the tracer roster"
            )
        if len(np.unique(indices)) != indices.size:
            raise CUDARestrictedWHMultirateContractError(
                "tracer_indices must be unique"
            )
        allowed_set = set(allowed)
        for lane, raw_ratio in enumerate(ratios):
            ratio = int(raw_ratio)
            if ratio not in allowed_set:
                raise CUDARestrictedWHMultirateContractError(
                    f"lane {lane} uses a ratio outside the declared roster"
                )
            if self.total_source_steps % ratio != 0:
                raise CUDARestrictedWHMultirateContractError(
                    f"lane {lane} ratio does not divide the segment horizon"
                )
            if self.global_source_step % ratio != 0:
                raise CUDARestrictedWHMultirateContractError(
                    f"lane {lane} does not begin on its absolute block boundary"
                )
        object.__setattr__(self, "tracer_indices", indices)
        object.__setattr__(self, "source_steps_per_map_step", ratios)
        object.__setattr__(self, "allowed_source_steps_per_map_step", allowed)
        actual = static_rung_plan_sha256(self)
        if self.content_sha256 is not None:
            expected = _sha256(self.content_sha256, "content_sha256")
            if expected != actual:
                raise CUDARestrictedWHMultirateContractError(
                    "rung-plan content_sha256 does not match canonical content"
                )
        object.__setattr__(self, "content_sha256", actual)


def static_rung_plan_sha256(plan: FrozenStaticRungPlan) -> str:
    """Return the canonical content digest of a static rung plan."""

    if type(plan) is not FrozenStaticRungPlan:
        raise CUDARestrictedWHMultirateContractError(
            "static_rung_plan_sha256 requires an exact FrozenStaticRungPlan"
        )
    metadata = {
        "allowed_source_steps_per_map_step": list(
            plan.allowed_source_steps_per_map_step
        ),
        "global_source_step": plan.global_source_step,
        "accuracy_policy_id": plan.accuracy_policy_id,
        "accuracy_validation_sha256": plan.accuracy_validation_sha256,
        "execution_policy_sha256": plan.execution_policy_sha256,
        "input_state_sha256": plan.input_state_sha256,
        "plan_id": plan.plan_id,
        "provenance": {
            "citation": plan.provenance.citation,
            "sha256": plan.provenance.sha256,
            "source_id": plan.provenance.source_id,
            "version": plan.provenance.version,
        },
        "selection_basis": plan.selection_basis,
        "source_schedule_sha256": plan.source_schedule_sha256,
        "source_tile_sha256": plan.source_tile_sha256,
        "total_source_steps": plan.total_source_steps,
    }
    digest = hashlib.sha256()
    digest.update(PLAN_DOMAIN)
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    for label, array in (
        ("tracer_indices", plan.tracer_indices),
        ("source_steps_per_map_step", plan.source_steps_per_map_step),
    ):
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "little"))
        digest.update(encoded_label)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.size.to_bytes(8, "little"))
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def verify_static_rung_plan(plan: FrozenStaticRungPlan) -> str:
    """Rehash a plan so reflective tampering fails before numerical work."""

    actual = static_rung_plan_sha256(plan)
    if actual != plan.content_sha256:
        raise CUDARestrictedWHMultirateContractError(
            "frozen static rung plan no longer matches content_sha256"
        )
    return actual


@dataclass(frozen=True)
class StaticMultirateCUDARestrictedWHSpec:
    """One common-endpoint segment with immutable per-lane rungs.

    The fixed engine's primary phase ceiling is strengthened here into an
    all-source ceiling.  ``maximum_relative_sweep_fraction`` bounds a
    tracer--source displacement certificate at the assigned rung, while
    ``maximum_source_motion_sweep_fraction`` independently bounds each
    secondary's cubic-Hermite motion relative to the primary.
    """

    total_source_steps: int
    backend: BackendSpec
    maximum_source_dynamical_phase: float = 0.05
    threads_per_block: int = 128
    primary_outer_boundary_AU: float | None = None
    clearance_multiplier: float = 2.0
    maximum_relative_sweep_fraction: float = 0.1
    maximum_source_motion_sweep_fraction: float = 0.1

    def __post_init__(self) -> None:
        if type(self.total_source_steps) is not int or self.total_source_steps <= 0:
            raise CUDARestrictedWHMultirateContractError(
                "total_source_steps must be a positive integer"
            )
        if type(self.backend) is not BackendSpec or self.backend.backend_id != "cupy":
            raise CUDARestrictedWHMultirateContractError(
                "static multirate execution requires an exact CuPy BackendSpec"
            )
        # Reuse the fixed-step contract for shared numerical policy fields.
        try:
            FixedStepCUDARestrictedWHSpec(
                1,
                1,
                self.backend,
                maximum_primary_dynamical_phase=(
                    self.maximum_source_dynamical_phase
                ),
                threads_per_block=self.threads_per_block,
                primary_outer_boundary_AU=self.primary_outer_boundary_AU,
            )
        except _fixed.CUDARestrictedWHContractError as exc:
            raise CUDARestrictedWHMultirateContractError(str(exc)) from exc
        for value, label, lower, upper in (
            (
                self.clearance_multiplier,
                "clearance_multiplier",
                1.0,
                math.inf,
            ),
            (
                self.maximum_relative_sweep_fraction,
                "maximum_relative_sweep_fraction",
                0.0,
                0.5,
            ),
            (
                self.maximum_source_motion_sweep_fraction,
                "maximum_source_motion_sweep_fraction",
                0.0,
                0.5,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower < float(value) <= upper
            ):
                relation = "(1, inf)" if math.isinf(upper) else f"(0, {upper}]"
                raise CUDARestrictedWHMultirateContractError(
                    f"{label} must be finite and in {relation}"
                )


def static_multirate_spec_sha256(
    spec: StaticMultirateCUDARestrictedWHSpec,
) -> str:
    """Hash every execution-relevant field of one multirate policy.

    This digest binds a rung plan to the exact numerical and hardware-request
    policy for which its external accuracy evidence was prepared.  It does
    not, by itself, assert that the external evidence is scientifically
    sufficient.
    """

    if type(spec) is not StaticMultirateCUDARestrictedWHSpec:
        raise CUDARestrictedWHMultirateContractError(
            "static_multirate_spec_sha256 requires an exact "
            "StaticMultirateCUDARestrictedWHSpec"
        )
    backend = spec.backend
    metadata = {
        "backend": {
            "allow_fallback": backend.allow_fallback,
            "backend_id": backend.backend_id,
            "determinism_scope": backend.determinism_scope,
            "deterministic_reductions": backend.deterministic_reductions,
            "device": backend.device,
            "dtype": backend.dtype,
            "fast_math": backend.fast_math,
            "tile_size": backend.tile_size,
        },
        "clearance_multiplier_hex": float(spec.clearance_multiplier).hex(),
        "maximum_relative_sweep_fraction_hex": float(
            spec.maximum_relative_sweep_fraction
        ).hex(),
        "maximum_source_dynamical_phase_hex": float(
            spec.maximum_source_dynamical_phase
        ).hex(),
        "maximum_source_motion_sweep_fraction_hex": float(
            spec.maximum_source_motion_sweep_fraction
        ).hex(),
        "primary_outer_boundary_AU_hex": (
            None
            if spec.primary_outer_boundary_AU is None
            else float(spec.primary_outer_boundary_AU).hex()
        ),
        "threads_per_block": spec.threads_per_block,
        "total_source_steps": spec.total_source_steps,
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(EXECUTION_POLICY_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return digest.hexdigest()


def _input_state_sha256_from_host(
    state: CUDARestrictedWHState,
    positions: np.ndarray,
    velocities: np.ndarray,
    indices: np.ndarray,
) -> str:
    """Hash validated canonical host copies of one exact CUDA input state."""

    metadata = {
        "axes": state.axes,
        "coordinate_system": state.coordinate_system,
        "epoch_origin_years_hex": float(state.epoch_origin_years).hex(),
        "frame": state.frame,
        "global_source_step": state.global_source_step,
        "length_unit": state.length_unit,
        "primary_source_id": state.primary_source_id,
        "provenance": {
            "citation": state.provenance.citation,
            "sha256": state.provenance.sha256,
            "source_id": state.provenance.source_id,
            "version": state.provenance.version,
        },
        "snapshot_id": state.snapshot_id,
        "source_ids": list(state.source_ids),
        "source_schedule_sha256": state.source_schedule_sha256,
        "source_step_years_hex": float(state.source_step_years).hex(),
        "time_scale": state.time_scale,
        "time_unit": state.time_unit,
        "unit_system_id": state.unit_system_id,
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(INPUT_STATE_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    for label, value, dtype in (
        ("tracer_relative_positions_AU", positions, np.dtype("<f8")),
        (
            "tracer_relative_velocities_AU_per_year",
            velocities,
            np.dtype("<f8"),
        ),
        ("tracer_indices", indices, np.dtype("<u8")),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "little"))
        digest.update(encoded_label)
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded_dtype).to_bytes(4, "little"))
        digest.update(encoded_dtype)
        digest.update(array.ndim.to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def static_multirate_input_state_sha256(
    state: CUDARestrictedWHState,
    backend: BackendSpec,
) -> str:
    """Hash an exact validated CUDA state for a static-rung plan.

    The transfer is intentional: the accuracy assignment is content-addressed
    against canonical host bytes before any numerical result can be committed.
    """

    if type(state) is not CUDARestrictedWHState:
        raise CUDARestrictedWHMultirateContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise CUDARestrictedWHMultirateContractError(
            "backend must be an exact CuPy BackendSpec"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        try:
            arrays = _fixed._validate_tracer_arrays(state, resolved)
        except (BackendArrayError, _fixed.CUDARestrictedWHContractError) as exc:
            raise CUDARestrictedWHMultirateContractError(str(exc)) from exc
        positions = np.ascontiguousarray(
            cp.asnumpy(arrays.positions), dtype=np.dtype("<f8")
        )
        velocities = np.ascontiguousarray(
            cp.asnumpy(arrays.velocities), dtype=np.dtype("<f8")
        )
        indices = np.ascontiguousarray(
            cp.asnumpy(arrays.indices), dtype=np.dtype("<u8")
        )
        cp.cuda.get_current_stream().synchronize()
    return _input_state_sha256_from_host(
        state,
        positions,
        velocities,
        indices,
    )


@dataclass(frozen=True, eq=False)
class CUDARestrictedWHMultirateResult:
    """One fully committed common-endpoint static-rung result."""

    state: CUDARestrictedWHState
    source_tile_sha256: str
    rung_plan_sha256: str
    rung_lane_counts: tuple[tuple[int, int], ...]
    minimum_endpoint_source_distance_AU: object
    maximum_kepler_iterations: int
    runtime_identity: Mapping[str, Any]
    method_id: str = METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    claims: Mapping[str, bool] = RESULT_CLAIMS

    def __post_init__(self) -> None:
        if type(self.state) is not CUDARestrictedWHState:
            raise CUDARestrictedWHMultirateContractError(
                "result state must be an exact CUDARestrictedWHState"
            )
        _sha256(self.source_tile_sha256, "source_tile_sha256")
        _sha256(self.rung_plan_sha256, "rung_plan_sha256")
        if type(self.rung_lane_counts) is not tuple:
            raise CUDARestrictedWHMultirateContractError(
                "rung_lane_counts must be an immutable tuple"
            )
        prior = 0
        total = 0
        for ratio, count in self.rung_lane_counts:
            if (
                type(ratio) is not int
                or ratio <= prior
                or type(count) is not int
                or count < 0
            ):
                raise CUDARestrictedWHMultirateContractError(
                    "rung_lane_counts is noncanonical"
                )
            prior = ratio
            total += count
        tracer_shape = getattr(self.state.tracer_indices, "shape", None)
        if (
            type(tracer_shape) is not tuple
            or len(tracer_shape) != 1
            or total != int(tracer_shape[0])
        ):
            raise CUDARestrictedWHMultirateContractError(
                "rung lane counts do not exactly cover the result roster"
            )
        if (
            type(self.maximum_kepler_iterations) is not int
            or self.maximum_kepler_iterations < 0
        ):
            raise CUDARestrictedWHMultirateContractError(
                "maximum_kepler_iterations must be nonnegative"
            )
        if (
            self.method_id != METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(RESULT_CLAIMS)
            or any(self.claims.values())
        ):
            raise CUDARestrictedWHMultirateContractError(
                "multirate result identity or claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(RESULT_CLAIMS)))
        object.__setattr__(self, "runtime_identity", _freeze_tree(self.runtime_identity))


def _fixed_spec(
    spec: StaticMultirateCUDARestrictedWHSpec,
    ratio: int,
) -> FixedStepCUDARestrictedWHSpec:
    return FixedStepCUDARestrictedWHSpec(
        spec.total_source_steps // ratio,
        ratio,
        spec.backend,
        maximum_primary_dynamical_phase=spec.maximum_source_dynamical_phase,
        threads_per_block=spec.threads_per_block,
        primary_outer_boundary_AU=spec.primary_outer_boundary_AU,
    )


STATIC_MULTIRATE_BATCH_KERNEL_SOURCE = r'''
#define JX_MR_CLEARANCE 101
#define JX_MR_RELATIVE_SWEEP 102
#define JX_MR_ALL_SOURCE_PHASE 103

__device__ __forceinline__ int jx_multirate_clearance_guard(
    const double* position,
    const double* source_position,
    const double* source_radius,
    const int source_count,
    const double clearance_multiplier,
    int* output_source)
{
    for (int source = 0; source < source_count; ++source) {
        const int offset = source * 3;
        const double dx = source_position[offset] - position[0];
        const double dy = source_position[offset + 1] - position[1];
        const double dz = source_position[offset + 2] - position[2];
        const double distance_squared = (dx * dx + dy * dy) + dz * dz;
        const double threshold = clearance_multiplier * source_radius[source];
        if (!isfinite(distance_squared) || !(distance_squared > 0.0)) {
            *output_source = source;
            return JX_KICK_NONFINITE;
        }
        const double distance = sqrt(distance_squared);
        if (!(distance > source_radius[source])) {
            *output_source = source;
            return JX_KICK_SOURCE_RADIUS;
        }
        if (!(distance > threshold)) {
            *output_source = source;
            return JX_MR_CLEARANCE;
        }
    }
    return JX_KICK_SUCCESS;
}

__device__ __forceinline__ int jx_multirate_all_source_guard(
    const double* position,
    const double* kicked_velocity,
    const double* source_position,
    const double* source_velocity,
    const double* source_gm,
    const double* source_radius,
    const double* relative_acceleration_bound,
    const int source_count,
    const double full_step,
    const double maximum_source_phase,
    const double clearance_multiplier,
    const double maximum_relative_sweep_fraction,
    const double primary_outer_boundary,
    int* output_source)
{
    const double absolute_step = fabs(full_step);
    const double half_step_squared = 0.5 * absolute_step * absolute_step;
    for (int source = 0; source < source_count; ++source) {
        const int offset = source * 3;
        const double dx = source_position[offset] - position[0];
        const double dy = source_position[offset + 1] - position[1];
        const double dz = source_position[offset + 2] - position[2];
        const double distance_squared = (dx * dx + dy * dy) + dz * dz;
        const double wx = kicked_velocity[0] - source_velocity[offset];
        const double wy = kicked_velocity[1] - source_velocity[offset + 1];
        const double wz = kicked_velocity[2] - source_velocity[offset + 2];
        const double speed_squared = (wx * wx + wy * wy) + wz * wz;
        if (!isfinite(distance_squared) || !(distance_squared > 0.0)
            || !isfinite(speed_squared) || !(speed_squared >= 0.0)) {
            *output_source = source;
            return JX_KICK_NONFINITE;
        }
        const double distance = sqrt(distance_squared);
        const double relative_speed = sqrt(speed_squared);
        const double displacement = relative_speed * absolute_step
            + relative_acceleration_bound[source] * half_step_squared;
        const double lower_distance = distance - displacement;
        if (!isfinite(displacement) || !isfinite(lower_distance)
            || !(lower_distance > 0.0)) {
            *output_source = source;
            return JX_KICK_NONFINITE;
        }
        if (!(lower_distance > clearance_multiplier * source_radius[source])) {
            *output_source = source;
            return JX_MR_CLEARANCE;
        }
        const double sweep_fraction = displacement / distance;
        if (!isfinite(sweep_fraction)
            || !(sweep_fraction < maximum_relative_sweep_fraction)) {
            *output_source = source;
            return JX_MR_RELATIVE_SWEEP;
        }
        const double phase = absolute_step * sqrt(
            source_gm[source]
            / ((lower_distance * lower_distance) * lower_distance));
        if (!isfinite(phase) || !(phase < maximum_source_phase)) {
            *output_source = source;
            return JX_MR_ALL_SOURCE_PHASE;
        }
        if (source == 0) {
            const double upper_distance = distance + displacement;
            if (!isfinite(upper_distance)
                || !(upper_distance < primary_outer_boundary)) {
                *output_source = 0;
                return JX_KICK_OUTER_BOUNDARY;
            }
        }
    }
    return JX_KICK_SUCCESS;
}

extern "C" __global__
void jx_restricted_wh_static_multirate_segment(
    const double* input_position,
    const double* input_velocity,
    const unsigned int* source_steps_per_map_step,
    const double* source_position,
    const double* source_velocity,
    const double* source_gm,
    const double* source_radius,
    const double* relative_acceleration_bound,
    const int lane_count,
    const int source_count,
    const int total_source_steps,
    const double source_step_years,
    const double maximum_source_phase,
    const double clearance_multiplier,
    const double maximum_relative_sweep_fraction,
    const double primary_outer_boundary,
    double* output_position,
    double* output_velocity,
    double* output_minimum_distance,
    int* output_maximum_iterations,
    int* output_status,
    int* output_failure_map,
    int* output_failure_phase,
    int* output_failure_source,
    int* output_failure_source_step)
{
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lane_count) return;
    const int lane_offset = lane * 3;
    const int ratio = (int)source_steps_per_map_step[lane];
    const int map_step_count = total_source_steps / ratio;
    const double full_step = source_step_years * (double)ratio;
    const double half_step = 0.5 * full_step;
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
    output_failure_source_step[lane] = -1;

    double cumulative_minimum = JX_KICK_DBL_MAX;
    int maximum_iterations = 0;
    const long long source_node_stride = (long long)source_count * 3LL;
    for (int map_step = 0; map_step < map_step_count; ++map_step) {
        const int source_step = map_step * ratio;
        const int end_source_step = source_step + ratio;
        const double* start_position = source_position
            + (long long)source_step * source_node_stride;
        const double* start_velocity = source_velocity
            + (long long)source_step * source_node_stride;
        const double* end_position = source_position
            + (long long)end_source_step * source_node_stride;
        const double* end_velocity = source_velocity
            + (long long)end_source_step * source_node_stride;

        double kicked_velocity[3];
        double pre_minimum;
        int failure_source = -1;
        int status = jx_multirate_clearance_guard(
            position,
            start_position,
            source_radius,
            source_count,
            clearance_multiplier,
            &failure_source);
        if (status == JX_KICK_SUCCESS) {
            status = jx_restricted_wh_half_kick_device(
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
                maximum_source_phase,
                primary_outer_boundary,
                0,
                kicked_velocity,
                &pre_minimum,
                &failure_source);
        } else {
            pre_minimum = JX_KICK_DBL_MAX;
        }
        cumulative_minimum = fmin(cumulative_minimum, pre_minimum);
        if (status == JX_KICK_SUCCESS) {
            status = jx_multirate_all_source_guard(
                position,
                kicked_velocity,
                start_position,
                start_velocity,
                source_gm,
                source_radius,
                relative_acceleration_bound,
                source_count,
                full_step,
                maximum_source_phase,
                clearance_multiplier,
                maximum_relative_sweep_fraction,
                primary_outer_boundary,
                &failure_source);
        }
        if (status != JX_KICK_SUCCESS) {
            output_minimum_distance[lane] = cumulative_minimum;
            output_maximum_iterations[lane] = maximum_iterations;
            output_status[lane] = status;
            output_failure_map[lane] = map_step;
            output_failure_phase[lane] = 0;
            output_failure_source[lane] = failure_source;
            output_failure_source_step[lane] = source_step;
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
            output_failure_source_step[lane] = source_step;
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
            maximum_source_phase,
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
            output_failure_source_step[lane] = end_source_step;
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

STATIC_MULTIRATE_CUDA_SOURCE = (
    _fixed._all_conic_kepler_module.CUDA_DEVICE_SOURCE
    + _fixed.KICK_DEVICE_SOURCE
    + STATIC_MULTIRATE_BATCH_KERNEL_SOURCE
)
STATIC_MULTIRATE_CUDA_SOURCE_SHA256 = hashlib.sha256(
    STATIC_MULTIRATE_CUDA_SOURCE.encode("ascii")
).hexdigest()


def _hermite_source_interval_length_bounds(
    tile: FrozenSourcePathTile,
    total_source_steps: int,
) -> np.ndarray:
    """Bound secondary path length inside every fine Hermite interval.

    For ``p(u)-p0 = a*u + b*u**2 + c*u**3`` on ``u in [0, 1]``, the
    integrated speed is no larger than ``|a| + |b| + |c|``.  This uses the
    exact position/velocity preimage consumed by the DOP853 source
    reconstruction and therefore covers interior overshoot, not only nodes.
    """

    if len(tile.source_ids) <= 1:
        return np.empty((total_source_steps, 0), dtype=np.float64)
    positions = tile.source_relative_positions_AU[
        : total_source_steps + 1, 1:, :
    ]
    velocities = tile.source_relative_velocities_AU_per_year[
        : total_source_steps + 1, 1:, :
    ]
    dt = float(tile.source_step_years)
    delta = positions[1:] - positions[:-1]
    a = dt * velocities[:-1]
    b = 3.0 * delta - dt * (2.0 * velocities[:-1] + velocities[1:])
    c = -2.0 * delta + dt * (velocities[:-1] + velocities[1:])
    return np.ascontiguousarray(
        np.linalg.norm(a, axis=2)
        + np.linalg.norm(b, axis=2)
        + np.linalg.norm(c, axis=2),
        dtype=np.float64,
    )


def _certify_source_motion(
    tile: FrozenSourcePathTile,
    spec: StaticMultirateCUDARestrictedWHSpec,
    used_ratios: tuple[int, ...],
) -> None:
    """Reject a rung that undersamples any secondary's indirect kick."""

    if len(tile.source_ids) <= 1:
        return
    lengths = _hermite_source_interval_length_bounds(
        tile, spec.total_source_steps
    )
    positions = tile.source_relative_positions_AU[
        : spec.total_source_steps + 1, 1:, :
    ]
    gm = tile.source_gm_AU3_per_year2
    absolute_source_step = abs(float(tile.source_step_years))
    for ratio in used_ratios:
        block_count = spec.total_source_steps // ratio
        block_lengths = lengths.reshape(
            block_count, ratio, lengths.shape[1]
        ).sum(axis=1)
        start_distance = np.linalg.norm(
            positions[::ratio][:block_count], axis=2
        )
        lower_distance = start_distance - block_lengths
        sweep = block_lengths / start_distance
        pair_gm = gm[0] + gm[1:]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            phase = absolute_source_step * ratio * np.sqrt(
                pair_gm[None, :] / lower_distance**3
            )
        valid = (
            np.isfinite(lower_distance)
            & np.isfinite(sweep)
            & np.isfinite(phase)
            & (lower_distance > 0.0)
            & (sweep < spec.maximum_source_motion_sweep_fraction)
            & (phase < spec.maximum_source_dynamical_phase)
        )
        if not bool(np.all(valid)):
            block, secondary_offset = np.argwhere(~valid)[0]
            raise CUDARestrictedWHMultirateContractError(
                "source-motion certificate rejects ratio "
                f"{ratio} at block {int(block)} for source "
                f"{tile.source_ids[int(secondary_offset) + 1]}"
            )


def advance_cuda_restricted_wh_static_multirate(
    state: CUDARestrictedWHState,
    source_tile: FrozenSourcePathTile,
    spec: StaticMultirateCUDARestrictedWHSpec,
    rung_plan: FrozenStaticRungPlan,
) -> CUDARestrictedWHMultirateResult:
    """Advance predeclared fixed-rung groups to one common lattice endpoint."""

    if type(state) is not CUDARestrictedWHState:
        raise CUDARestrictedWHMultirateContractError(
            "state must be an exact CUDARestrictedWHState"
        )
    if type(source_tile) is not FrozenSourcePathTile:
        raise CUDARestrictedWHMultirateContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    if type(spec) is not StaticMultirateCUDARestrictedWHSpec:
        raise CUDARestrictedWHMultirateContractError(
            "spec must be an exact StaticMultirateCUDARestrictedWHSpec"
        )
    if type(rung_plan) is not FrozenStaticRungPlan:
        raise CUDARestrictedWHMultirateContractError(
            "rung_plan must be an exact FrozenStaticRungPlan"
        )
    try:
        _fixed._validate_compatibility(
            state,
            source_tile,
            _fixed_spec(spec, 1),
        )
    except _fixed.CUDARestrictedWHContractError as exc:
        raise CUDARestrictedWHMultirateContractError(str(exc)) from exc
    tile_sha = verify_source_path_tile(source_tile)
    plan_sha = verify_static_rung_plan(rung_plan)
    if (
        rung_plan.global_source_step != state.global_source_step
        or rung_plan.global_source_step != source_tile.global_source_step
    ):
        raise CUDARestrictedWHMultirateContractError(
            "state, source tile, and rung plan must begin at one lattice node"
        )
    if rung_plan.total_source_steps != spec.total_source_steps:
        raise CUDARestrictedWHMultirateContractError(
            "rung plan and multirate spec disagree on the segment horizon"
        )
    if rung_plan.source_schedule_sha256 != state.source_schedule_sha256:
        raise CUDARestrictedWHMultirateContractError(
            "rung plan names a different source schedule"
        )
    if rung_plan.source_tile_sha256 != tile_sha:
        raise CUDARestrictedWHMultirateContractError(
            "rung plan names a different immutable source tile"
        )
    policy_sha = static_multirate_spec_sha256(spec)
    if rung_plan.execution_policy_sha256 != policy_sha:
        raise CUDARestrictedWHMultirateContractError(
            "rung plan names a different multirate execution policy"
        )
    if source_tile.node_count < spec.total_source_steps + 1:
        raise CUDARestrictedWHMultirateContractError(
            "source tile does not contain the complete common horizon"
        )

    resolved = resolve_backend(spec.backend)
    cp = resolved.xp
    current_ratio: int | None = None
    try:
        with resolved.activate():
            try:
                arrays = _fixed._validate_tracer_arrays(state, resolved)
            except (BackendArrayError, _fixed.CUDARestrictedWHContractError) as exc:
                raise CUDARestrictedWHMultirateContractError(str(exc)) from exc
            host_positions = np.ascontiguousarray(
                cp.asnumpy(arrays.positions), dtype=np.dtype("<f8")
            )
            host_velocities = np.ascontiguousarray(
                cp.asnumpy(arrays.velocities), dtype=np.dtype("<f8")
            )
            host_indices = np.ascontiguousarray(
                cp.asnumpy(arrays.indices), dtype=np.dtype("<u8")
            )
            cp.cuda.get_current_stream().synchronize()
            if not np.array_equal(host_indices, rung_plan.tracer_indices):
                raise CUDARestrictedWHMultirateContractError(
                    "rung plan tracer roster or order differs from the state"
                )
            input_state_sha = _input_state_sha256_from_host(
                state,
                host_positions,
                host_velocities,
                host_indices,
            )
            if rung_plan.input_state_sha256 != input_state_sha:
                raise CUDARestrictedWHMultirateContractError(
                    "rung plan names different input-state bytes or metadata"
                )
            used_ratios = tuple(
                int(value)
                for value in np.unique(rung_plan.source_steps_per_map_step)
            )
            _certify_source_motion(source_tile, spec, used_ratios)

            int32_max = int(np.iinfo(np.int32).max)
            if spec.total_source_steps > int32_max:
                raise CUDARestrictedWHMultirateContractError(
                    "total_source_steps exceeds the signed CUDA int32 limit"
                )
            if arrays.lane_count > int32_max:
                raise CUDARestrictedWHMultirateContractError(
                    "tracer lane count exceeds the signed CUDA int32 limit"
                )
            if (
                spec.total_source_steps
                * 3
                * max(1, arrays.lane_count)
                > int(np.iinfo(np.int64).max)
            ):
                raise CUDARestrictedWHMultirateContractError(
                    "canonical CUDA failure-order key would overflow int64"
                )

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
            device_ratios = cp.asarray(
                rung_plan.source_steps_per_map_step, dtype=cp.uint32
            )
            positions = cp.array(
                arrays.positions, dtype=cp.float64, copy=True, order="C"
            )
            velocities = cp.array(
                arrays.velocities, dtype=cp.float64, copy=True, order="C"
            )
            output_indices = cp.array(arrays.indices, copy=True, order="C")
            minimum = cp.full(arrays.lane_count, cp.inf, dtype=cp.float64)
            maximum_iterations = 0
            counts = tuple(
                (
                    ratio,
                    int(
                        np.count_nonzero(
                            rung_plan.source_steps_per_map_step == ratio
                        )
                    ),
                )
                for ratio in rung_plan.allowed_source_steps_per_map_step
            )
            execution_identity: Mapping[str, Any]

            if arrays.lane_count:
                status = cp.zeros(arrays.lane_count, dtype=cp.int32)
                failure_map = cp.full(arrays.lane_count, -1, dtype=cp.int32)
                failure_phase = cp.full(arrays.lane_count, -1, dtype=cp.int32)
                failure_source = cp.full(arrays.lane_count, -1, dtype=cp.int32)
                failure_source_step = cp.full(
                    arrays.lane_count, -1, dtype=cp.int32
                )
                per_lane_iterations = cp.zeros(
                    arrays.lane_count, dtype=cp.int32
                )
                primary_outer_boundary = (
                    _fixed.MAXIMUM_BINARY64
                    if spec.primary_outer_boundary_AU is None
                    else spec.primary_outer_boundary_AU
                )
                kernel = cp.RawKernel(
                    STATIC_MULTIRATE_CUDA_SOURCE,
                    "jx_restricted_wh_static_multirate_segment",
                    options=_fixed.CUDA_OPTIONS,
                )
                blocks = (
                    arrays.lane_count + spec.threads_per_block - 1
                ) // spec.threads_per_block
                kernel(
                    (blocks,),
                    (spec.threads_per_block,),
                    (
                        arrays.positions,
                        arrays.velocities,
                        device_ratios,
                        source_positions,
                        source_velocities,
                        source_gm,
                        source_radii,
                        relative_acceleration_bounds,
                        cp.int32(arrays.lane_count),
                        cp.int32(len(source_tile.source_ids)),
                        cp.int32(spec.total_source_steps),
                        cp.float64(source_tile.source_step_years),
                        cp.float64(spec.maximum_source_dynamical_phase),
                        cp.float64(spec.clearance_multiplier),
                        cp.float64(spec.maximum_relative_sweep_fraction),
                        cp.float64(primary_outer_boundary),
                        positions,
                        velocities,
                        minimum,
                        per_lane_iterations,
                        status,
                        failure_map,
                        failure_phase,
                        failure_source,
                        failure_source_step,
                    ),
                )
                failure_key = cp.where(
                    failure_source_step >= 0,
                    (
                        (
                            failure_source_step.astype(cp.int64) * cp.int64(3)
                            + failure_phase.astype(cp.int64)
                        )
                        * cp.int64(arrays.lane_count)
                    )
                    + cp.arange(arrays.lane_count, dtype=cp.int64),
                    cp.iinfo(cp.int64).max,
                )
                failed_lane = int(cp.argmin(failure_key).item())
                failed_source_step = int(failure_source_step[failed_lane].item())
                if failed_source_step >= 0:
                    failed_ratio = int(
                        rung_plan.source_steps_per_map_step[failed_lane]
                    )
                    current_ratio = failed_ratio
                    failed_map = int(failure_map[failed_lane].item())
                    failed_phase = int(failure_phase[failed_lane].item())
                    failed_status = int(status[failed_lane].item())
                    raw_source = int(failure_source[failed_lane].item())
                    fixed_handoff = _fixed._failure_handoff(
                        state=state,
                        tile=source_tile,
                        spec=_fixed_spec(spec, failed_ratio),
                        map_index=failed_map,
                        phase_code=failed_phase,
                        reason_code=failed_status,
                        lane_index=failed_lane,
                        tracer_index=int(host_indices[failed_lane]),
                        source_index=(raw_source if raw_source >= 0 else None),
                        reason_override=_MR_STATUS_NAMES.get(failed_status),
                    )
                    raise CUDARestrictedWHMultirateIntegrationError(
                        "static multirate segment failed closed at source step "
                        f"{fixed_handoff.failure_global_source_step}: "
                        f"{fixed_handoff.reason}",
                        CUDARestrictedWHMultirateFailureHandoff(
                            state=state,
                            rung_plan_sha256=plan_sha,
                            failed_source_steps_per_map_step=failed_ratio,
                            fixed_step_handoff=fixed_handoff,
                        ),
                    )
                maximum_iterations = int(cp.max(per_lane_iterations).item())
                execution_identity = _fixed._runtime_identity(
                    cp,
                    resolved.device,
                    execution_kernel_id=STATIC_MULTIRATE_KERNEL_ID,
                    execution_kernel_source_sha256=(
                        STATIC_MULTIRATE_CUDA_SOURCE_SHA256
                    ),
                )
            else:
                execution_identity = _fixed._runtime_identity(
                    cp,
                    resolved.device,
                    execution_kernel_id=STATIC_MULTIRATE_KERNEL_ID,
                    execution_kernel_source_sha256=(
                        STATIC_MULTIRATE_CUDA_SOURCE_SHA256
                    ),
                )

            cp.cuda.get_current_stream().synchronize()
            output_state = replace(
                state,
                global_source_step=(
                    state.global_source_step + spec.total_source_steps
                ),
                tracer_relative_positions_AU=positions,
                tracer_relative_velocities_AU_per_year=velocities,
                tracer_indices=output_indices,
            )
            runtime_identity = {
                "method_id": METHOD_ID,
                "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
                "fixed_implementation_source_sha256": (
                    _fixed.IMPLEMENTATION_SOURCE_SHA256
                ),
                "source_tile_sha256": tile_sha,
                "rung_plan_sha256": plan_sha,
                "accuracy_policy_id": rung_plan.accuracy_policy_id,
                "accuracy_validation_sha256": (
                    rung_plan.accuracy_validation_sha256
                ),
                "input_state_sha256": input_state_sha,
                "execution_policy_sha256": policy_sha,
                "fixed_rungs_for_complete_segment": True,
                "dynamic_rung_changes_allowed": False,
                "eligibility_policy": {
                    "method": (
                        "PER_MAP_STEP_ALL_SOURCE_ACCELERATION_ENVELOPE_"
                        "PLUS_HERMITE_SOURCE_MOTION_BOUND_V1"
                    ),
                    "clearance_multiplier_hex": float(
                        spec.clearance_multiplier
                    ).hex(),
                    "maximum_source_dynamical_phase_hex": float(
                        spec.maximum_source_dynamical_phase
                    ).hex(),
                    "maximum_relative_sweep_fraction_hex": float(
                        spec.maximum_relative_sweep_fraction
                    ).hex(),
                    "maximum_source_motion_sweep_fraction_hex": float(
                        spec.maximum_source_motion_sweep_fraction
                    ).hex(),
                },
                "execution_runtime_identity": dict(execution_identity),
                "determinism_scope": "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE",
            }
            return CUDARestrictedWHMultirateResult(
                state=output_state,
                source_tile_sha256=tile_sha,
                rung_plan_sha256=plan_sha,
                rung_lane_counts=counts,
                minimum_endpoint_source_distance_AU=minimum,
                maximum_kepler_iterations=maximum_iterations,
                runtime_identity=runtime_identity,
            )
    except CUDARestrictedWHMultirateError:
        raise
    except _fixed.CUDARestrictedWHIntegrationError as exc:
        raise CUDARestrictedWHMultirateIntegrationError(
            "one fixed-rung group failed; no multirate result was committed",
            CUDARestrictedWHMultirateFailureHandoff(
                state=state,
                rung_plan_sha256=plan_sha,
                failed_source_steps_per_map_step=current_ratio,
                fixed_step_handoff=exc.handoff,
            ),
        ) from exc
    except _fixed.CUDARestrictedWHContractError as exc:
        raise CUDARestrictedWHMultirateContractError(str(exc)) from exc
    except Exception as exc:
        raise CUDARestrictedWHMultirateIntegrationError(
            "static multirate CUDA execution failed; input state is unchanged",
            CUDARestrictedWHMultirateFailureHandoff(
                state=state,
                rung_plan_sha256=plan_sha,
                failed_source_steps_per_map_step=current_ratio,
                fixed_step_handoff=None,
            ),
        ) from exc


__all__ = [
    "CUDARestrictedWHMultirateContractError",
    "CUDARestrictedWHMultirateError",
    "CUDARestrictedWHMultirateFailureHandoff",
    "CUDARestrictedWHMultirateIntegrationError",
    "CUDARestrictedWHMultirateResult",
    "FrozenStaticRungPlan",
    "IMPLEMENTATION_SOURCE_SHA256",
    "MAXIMUM_RUNG_RATIO",
    "METHOD_ID",
    "PROJECT_SCOPE",
    "RESULT_CLAIMS",
    "SCIENTIFIC_CLASSIFICATION",
    "StaticMultirateCUDARestrictedWHSpec",
    "advance_cuda_restricted_wh_static_multirate",
    "static_multirate_input_state_sha256",
    "static_multirate_spec_sha256",
    "static_rung_plan_sha256",
    "verify_static_rung_plan",
]
