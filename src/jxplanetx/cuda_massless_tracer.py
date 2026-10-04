"""Experimental CUDA restricted-N-body engine for massless tracer populations.

This module is deliberately additive: it does not widen the qualified public
Wisdom--Holman or encounter backends.  One mutually interacting point-mass
source system is advanced once per CUDA block while independent, zero-mass
tracer lanes feel the sources without back-reacting on them.  The quiet map is
fixed-step inertial kick--drift--kick (velocity Verlet).

An encounter guard is transactional.  A guarded tracer step is not committed
on the GPU; that complete outer step is re-integrated on the CPU with DOP853
against the exact cubic-Hermite source segment, then the lane rejoins the GPU
at the next outer node.  This is an ``encounter_route``, not an array-backend
fallback and not evidence that a collision or physical encounter occurred.

The implementation and its results remain ``MODEL_OUTPUT / SCREENING_ONLY``.
In particular, this module does not qualify a 4.5-Gyr integration, a real TNO
population, or any Planet Nine, stellar-flyby, or temporary-rogue inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .engine.backends import ArrayBackend, resolve_backend
from .engine.contracts import BackendSpec, ContractError, Provenance


def _import_time_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


METHOD_ID = "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V1"
KERNEL_ID = "jx.cuda.massless-tracer.restricted-kdk.v1"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
DETERMINISM_SCOPE = "SAME_PYTHON_NUMPY_SCIPY_CPU_AND_CUDA_RUNTIME_DEVICE"
IMPLEMENTATION_SOURCE_SHA256 = _import_time_file_sha256(Path(__file__))

STATUS_SUCCESS = 0
STATUS_ENCOUNTER_ROUTE = 1
STATUS_NONFINITE = 2
STATUS_FORCE_SINGULARITY = 3
STATUS_ACCELERATION_BOUND = 4
STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_ENCOUNTER_ROUTE: "ENCOUNTER_ROUTE",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    STATUS_ACCELERATION_BOUND: "ACCELERATION_BOUND",
}

REASON_NONE = 0
REASON_SWEPT_FLOOR = 1
REASON_ACCELERATION_BOUND = 2
REASON_NAMES = {
    REASON_NONE: "NONE",
    REASON_SWEPT_FLOOR: "SWEPT_FLOOR",
    REASON_ACCELERATION_BOUND: "ACCELERATION_BOUND",
}

CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)

MAXIMUM_SOURCE_COUNT = 32
MAXIMUM_THREADS_PER_BLOCK = 128
MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER = 256
CHECKPOINT_SCHEMA = "jxplanetx.cuda-massless-tracer-checkpoint.v3"
CHECKPOINT_DOMAIN = b"jxplanetx.cuda-massless-tracer-checkpoint.v3\x00"
CHECKPOINT_NAMESPACE_DOMAIN = b"jxplanetx.cuda-massless-tracer-namespace.v1\x00"
STATE_DOMAIN = b"jxplanetx.cuda-massless-tracer-state.v2\x00"
LINEAGE_IDENTITY_DOMAIN = b"jxplanetx.cuda-massless-tracer-lineage.v1\x00"
SOURCE_SEGMENT_DOMAIN = b"jxplanetx.cuda-massless-tracer-source-segment.v1\x00"
RESULT_CLAIMS = MappingProxyType(
    {
        "four_point_five_gyr_accuracy_qualified": False,
        "planet_nine_detected": False,
        "planet_nine_excluded": False,
        "stellar_flyby_inferred": False,
        "temporary_rogue_inferred": False,
        "close_encounter_confirmed": False,
        "collision_confirmed": False,
        "independently_reproduced": False,
    }
)


class CUDAMasslessTracerError(RuntimeError):
    """Base error for the experimental restricted tracer engine."""


class CUDAMasslessTracerContractError(CUDAMasslessTracerError, ValueError):
    """An input violates the explicit restricted-tracer contract."""


class CUDAMasslessTracerIntegrationError(CUDAMasslessTracerError):
    """A source or tracer lane failed closed during integration."""


class EncounterRouteError(CUDAMasslessTracerIntegrationError):
    """A guarded outer step could not be transactionally re-integrated."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CUDAMasslessTracerContractError(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CUDAMasslessTracerContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CUDAMasslessTracerContractError(f"{label} must be a finite number")
    return result


def _sha256_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CUDAMasslessTracerContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    if value == "0" * 64:
        raise CUDAMasslessTracerContractError(f"{label} cannot be all zero")
    return value


@dataclass(frozen=True, eq=False)
class CUDAMasslessTracerState:
    """One device-resident restricted source/tracer state.

    Numerical buffers remain opaque in the constructor so creating a contract
    never causes an implicit host/device transfer.  ``advance`` performs strict
    native-CuPy, device, dtype, shape, contiguity, and domain validation.
    """

    snapshot_id: str
    epoch: float
    global_step: int
    time_scale: str
    frame: str
    origin: str
    axes: str
    length_unit: str
    time_unit: str
    unit_system_id: str
    source_ids: tuple[str, ...]
    primary_source_id: str
    source_positions: object
    source_velocities: object
    source_gm: object
    source_encounter_radii: object
    relative_acceleration_bounds: object
    tracer_positions: object
    tracer_velocities: object
    tracer_indices: object
    provenance: Provenance
    minimum_source_distance_AU: object | None = None
    ever_unbound: object | None = None
    encounter_route_counts: object | None = None
    prior_encounter_routes: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _finite(self.epoch, "epoch")
        if type(self.global_step) is not int or self.global_step < 0:
            raise CUDAMasslessTracerContractError(
                "global_step must be a nonnegative integer"
            )
        for label in (
            "time_scale",
            "frame",
            "origin",
            "axes",
            "length_unit",
            "time_unit",
            "unit_system_id",
        ):
            _text(getattr(self, label), label)
        if type(self.source_ids) is not tuple or not self.source_ids:
            raise CUDAMasslessTracerContractError(
                "source_ids must be a nonempty immutable tuple"
            )
        for index, source_id in enumerate(self.source_ids):
            _text(source_id, f"source_ids[{index}]")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise CUDAMasslessTracerContractError("source_ids must be unique")
        if len(self.source_ids) > MAXIMUM_SOURCE_COUNT:
            raise CUDAMasslessTracerContractError(
                f"at most {MAXIMUM_SOURCE_COUNT} sources are supported"
            )
        _text(self.primary_source_id, "primary_source_id")
        if self.primary_source_id != self.source_ids[0]:
            raise CUDAMasslessTracerContractError(
                "primary_source_id must explicitly identify source slot zero"
            )
        if (
            self.length_unit != "AU"
            or self.time_unit != "year"
            or self.unit_system_id != "AU-YEAR"
        ):
            raise CUDAMasslessTracerContractError(
                "v1 accepts only the canonical AU/year unit_system_id='AU-YEAR'"
            )
        for label in (
            "source_positions",
            "source_velocities",
            "source_gm",
            "source_encounter_radii",
            "relative_acceleration_bounds",
            "tracer_positions",
            "tracer_velocities",
            "tracer_indices",
        ):
            if getattr(self, label) is None:
                raise CUDAMasslessTracerContractError(
                    f"{label} must be an explicit backend-native array"
                )
        if type(self.provenance) is not Provenance:
            raise CUDAMasslessTracerContractError("provenance must be Provenance")
        if type(self.prior_encounter_routes) is not tuple:
            raise CUDAMasslessTracerContractError(
                "prior_encounter_routes must be an immutable tuple"
            )


@dataclass(frozen=True)
class FixedStepCUDAMasslessTracerSpec:
    """Fixed CUDA segment and guarded CPU encounter-route policy."""

    step_years: float
    step_count: int
    backend: BackendSpec
    maximum_encounter_routes_per_tracer: int = 256
    encounter_route_method: str = "SCIPY_DOP853_FULL_OUTER_STEP_V1"
    encounter_route_rtol: float = 1.0e-12
    encounter_route_atol: float = 1.0e-14
    encounter_route_maximum_substep_fraction: float = 0.125
    source_path_steps_per_chunk: int = 65_536
    threads_per_block: int = MAXIMUM_THREADS_PER_BLOCK

    def __post_init__(self) -> None:
        step = _finite(self.step_years, "step_years")
        if step == 0.0:
            raise CUDAMasslessTracerContractError("step_years cannot be zero")
        if type(self.step_count) is not int or self.step_count <= 0:
            raise CUDAMasslessTracerContractError(
                "step_count must be a positive integer"
            )
        if type(self.backend) is not BackendSpec:
            raise CUDAMasslessTracerContractError("backend must be BackendSpec")
        if self.backend.backend_id != "cupy":
            raise CUDAMasslessTracerContractError(
                "the CUDA tracer engine requires backend_id='cupy'"
            )
        if (
            type(self.maximum_encounter_routes_per_tracer) is not int
            or self.maximum_encounter_routes_per_tracer < 0
            or self.maximum_encounter_routes_per_tracer
            > MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
        ):
            raise CUDAMasslessTracerContractError(
                "maximum_encounter_routes_per_tracer must be in "
                f"[0, {MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER}]"
            )
        if self.encounter_route_method != "SCIPY_DOP853_FULL_OUTER_STEP_V1":
            raise CUDAMasslessTracerContractError(
                "the only implemented encounter route is "
                "SCIPY_DOP853_FULL_OUTER_STEP_V1"
            )
        for label in ("encounter_route_rtol", "encounter_route_atol"):
            value = _finite(getattr(self, label), label)
            if value <= 0.0:
                raise CUDAMasslessTracerContractError(f"{label} must be positive")
        fraction = _finite(
            self.encounter_route_maximum_substep_fraction,
            "encounter_route_maximum_substep_fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise CUDAMasslessTracerContractError(
                "encounter_route_maximum_substep_fraction must be in (0, 1]"
            )
        if (
            type(self.source_path_steps_per_chunk) is not int
            or self.source_path_steps_per_chunk <= 0
        ):
            raise CUDAMasslessTracerContractError(
                "source_path_steps_per_chunk must be a positive integer"
            )
        if (
            type(self.threads_per_block) is not int
            or not 32 <= self.threads_per_block <= MAXIMUM_THREADS_PER_BLOCK
            or self.threads_per_block % 32 != 0
        ):
            raise CUDAMasslessTracerContractError(
                "threads_per_block must be a multiple of 32 in [32, 128]"
            )


@dataclass(frozen=True)
class TracerEncounterRouteRecord:
    """Audit record for one discarded GPU step and full CPU redo."""

    tracer_index: int
    source_index: int
    source_id: str
    global_step: int
    reason: str
    source_segment_sha256: str
    cpu_accepted_steps: int
    cpu_rejected_steps: int
    minimum_evaluated_stage_distance_AU: float
    endpoint_unbound: bool

    def __post_init__(self) -> None:
        if (
            type(self.tracer_index) is not int
            or not 0 <= self.tracer_index < 2**64
        ):
            raise CUDAMasslessTracerContractError(
                "route tracer_index must be a uint64-compatible integer"
            )
        if type(self.source_index) is not int or self.source_index < 0:
            raise CUDAMasslessTracerContractError(
                "route source_index must be nonnegative"
            )
        _text(self.source_id, "route source_id")
        if type(self.global_step) is not int or self.global_step < 0:
            raise CUDAMasslessTracerContractError(
                "route global_step must be nonnegative"
            )
        if self.reason not in REASON_NAMES.values() or self.reason == "NONE":
            raise CUDAMasslessTracerContractError("route reason is invalid")
        _sha256_text(self.source_segment_sha256, "route source_segment_sha256")
        if type(self.cpu_accepted_steps) is not int or self.cpu_accepted_steps <= 0:
            raise CUDAMasslessTracerContractError(
                "route cpu_accepted_steps must be positive"
            )
        if self.cpu_rejected_steps != -1:
            raise CUDAMasslessTracerContractError(
                "route cpu_rejected_steps must be -1 when unavailable"
            )
        distance = _finite(
            self.minimum_evaluated_stage_distance_AU,
            "route minimum_evaluated_stage_distance_AU",
        )
        if distance < 0.0:
            raise CUDAMasslessTracerContractError(
                "route minimum evaluated stage distance cannot be negative"
            )
        if type(self.endpoint_unbound) is not bool:
            raise CUDAMasslessTracerContractError(
                "route endpoint_unbound must be bool"
            )


@dataclass(frozen=True, eq=False)
class CUDAMasslessTracerResult:
    """Device-resident result of one completely committed segment."""

    state: CUDAMasslessTracerState
    minimum_source_distance_AU: object
    ever_unbound: object
    encounter_route_counts: object
    encounter_routes: tuple[TracerEncounterRouteRecord, ...]
    runtime_identity: Mapping[str, Any]
    input_state_sha256: str
    method_id: str = METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    claims: Mapping[str, bool] = RESULT_CLAIMS

    def __post_init__(self) -> None:
        if type(self.state) is not CUDAMasslessTracerState:
            raise CUDAMasslessTracerContractError(
                "result state must be an exact CUDAMasslessTracerState"
            )
        if self.method_id != METHOD_ID:
            raise CUDAMasslessTracerContractError("result method_id is immutable")
        if self.scientific_classification != SCIENTIFIC_CLASSIFICATION:
            raise CUDAMasslessTracerContractError(
                "result scientific classification cannot be widened"
            )
        if self.project_scope != PROJECT_SCOPE:
            raise CUDAMasslessTracerContractError(
                "result project scope cannot be widened"
            )
        if dict(self.claims) != dict(RESULT_CLAIMS):
            raise CUDAMasslessTracerContractError(
                "result claim controls must remain the fixed all-false map"
            )
        _sha256_text(self.input_state_sha256, "result input_state_sha256")
        object.__setattr__(
            self, "claims", MappingProxyType(dict(RESULT_CLAIMS))
        )
        object.__setattr__(
            self, "runtime_identity", MappingProxyType(dict(self.runtime_identity))
        )


@dataclass(frozen=True)
class _ValidatedState:
    source_positions: Any
    source_velocities: Any
    source_gm: Any
    source_encounter_radii: Any
    relative_acceleration_bounds: Any
    tracer_positions: Any
    tracer_velocities: Any
    tracer_indices: Any
    source_count: int
    tracer_count: int


@dataclass(frozen=True)
class _ValidatedContinuation:
    minimum_source_distance_AU: Any
    ever_unbound: Any
    encounter_route_counts_host: np.ndarray
    encounter_routes: tuple[TracerEncounterRouteRecord, ...]


@dataclass(frozen=True)
class _LaunchResult:
    source_positions: Any
    source_velocities: Any
    tracer_positions: Any
    tracer_velocities: Any
    minimum_source_distance: Any
    ever_unbound: Any
    status: Any
    reason: Any
    trigger_source: Any
    completed_steps: Any
    trigger_source_position_0: Any
    trigger_source_velocity_0: Any
    trigger_source_position_1: Any
    trigger_source_velocity_1: Any


@dataclass(frozen=True)
class _SourcePath:
    positions: Any
    velocities: Any
    accelerations: Any


@dataclass(frozen=True)
class _PathLaunchResult:
    tracer_positions: Any
    tracer_velocities: Any
    minimum_source_distance: Any
    ever_unbound: Any
    status: Any
    reason: Any
    trigger_source: Any
    completed_steps: Any


CUDA_SOURCE = r'''
#define JX_MAX_SOURCES 32
#define JX_MAX_COMPONENTS 96

#define JX_SUCCESS 0
#define JX_ENCOUNTER_ROUTE 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3

#define JX_REASON_NONE 0
#define JX_REASON_SWEPT_FLOOR 1
#define JX_REASON_ACCELERATION_BOUND 2

__device__ __forceinline__ int
jx_finite3(const double x, const double y, const double z)
{
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ int
jx_source_acceleration(
    const double* position,
    const double* gm,
    const int source_count,
    const int target,
    double* ax,
    double* ay,
    double* az)
{
    const int i = target * 3;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        if (source == target) continue;
        const int j = source * 3;
        const double dx = position[j] - position[i];
        const double dy = position[j+1] - position[i+1];
        const double dz = position[j+2] - position[i+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return jx_finite3(x, y, z);
}

__device__ __forceinline__ int
jx_tracer_acceleration(
    const double tx,
    const double ty,
    const double tz,
    const double* position,
    const double* gm,
    const int source_count,
    double* ax,
    double* ay,
    double* az)
{
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        const int j = source * 3;
        const double dx = position[j] - tx;
        const double dy = position[j+1] - ty;
        const double dz = position[j+2] - tz;
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return jx_finite3(x, y, z);
}

__device__ __forceinline__ int
jx_bound_to_primary(
    const double tx,
    const double ty,
    const double tz,
    const double tvx,
    const double tvy,
    const double tvz,
    const double* source_position,
    const double* source_velocity,
    const double primary_gm)
{
    const double rx = tx - source_position[0];
    const double ry = ty - source_position[1];
    const double rz = tz - source_position[2];
    const double vx = tvx - source_velocity[0];
    const double vy = tvy - source_velocity[1];
    const double vz = tvz - source_velocity[2];
    const double r2 = rx*rx + ry*ry + rz*rz;
    if (!(r2 > 0.0) || !isfinite(r2)) return 0;
    const double energy = 0.5*(vx*vx + vy*vy + vz*vz) - primary_gm/sqrt(r2);
    return isfinite(energy) && energy < 0.0;
}

extern "C" __global__
void jx_massless_tracer_restricted_kdk(
    const double* input_source_position,
    const double* input_source_velocity,
    const double* source_gm,
    const double* source_encounter_radius,
    const double* relative_acceleration_bound,
    const double* input_tracer_position,
    const double* input_tracer_velocity,
    const int source_count,
    const int tracer_count,
    const long long step_count,
    const double step,
    double* output_source_position,
    double* output_source_velocity,
    double* output_tracer_position,
    double* output_tracer_velocity,
    double* output_minimum_source_distance,
    int* output_ever_unbound,
    int* output_status,
    int* output_reason,
    int* output_trigger_source,
    long long* output_completed_steps,
    double* output_trigger_source_position_0,
    double* output_trigger_source_velocity_0,
    double* output_trigger_source_position_1,
    double* output_trigger_source_velocity_1)
{
    const int tid = (int)threadIdx.x;
    const int lane = (int)blockIdx.x * (int)blockDim.x + tid;
    const int tracer_active = lane < tracer_count;
    const int component_count = source_count * 3;

    __shared__ double sp[JX_MAX_COMPONENTS];
    __shared__ double sv[JX_MAX_COMPONENTS];
    __shared__ double sa[JX_MAX_COMPONENTS];
    __shared__ double sp0[JX_MAX_COMPONENTS];
    __shared__ double sv0[JX_MAX_COMPONENTS];
    __shared__ int source_status;

    if (tid == 0) source_status = JX_SUCCESS;
    if (tid < source_count) {
        const int i = tid * 3;
        sp[i] = input_source_position[i];
        sp[i+1] = input_source_position[i+1];
        sp[i+2] = input_source_position[i+2];
        sv[i] = input_source_velocity[i];
        sv[i+1] = input_source_velocity[i+1];
        sv[i+2] = input_source_velocity[i+2];
    }
    __syncthreads();

    if (tid < source_count) {
        const int i = tid * 3;
        double ax, ay, az;
        if (!jx_source_acceleration(sp, source_gm, source_count, tid, &ax, &ay, &az)) {
            atomicCAS(&source_status, JX_SUCCESS, JX_FORCE_SINGULARITY);
            ax = ay = az = 0.0;
        }
        sa[i] = ax; sa[i+1] = ay; sa[i+2] = az;
    }
    __syncthreads();

    double tx = 0.0, ty = 0.0, tz = 0.0;
    double tvx = 0.0, tvy = 0.0, tvz = 0.0;
    double tax = 0.0, tay = 0.0, taz = 0.0;
    double minimum_distance = 1.0/0.0;
    int ever_unbound = 0;
    int status = JX_SUCCESS;
    int reason = JX_REASON_NONE;
    int trigger_source = -1;
    long long completed_steps = 0;

    if (tracer_active) {
        const int i = lane * 3;
        tx = input_tracer_position[i];
        ty = input_tracer_position[i+1];
        tz = input_tracer_position[i+2];
        tvx = input_tracer_velocity[i];
        tvy = input_tracer_velocity[i+1];
        tvz = input_tracer_velocity[i+2];
        if (!jx_finite3(tx,ty,tz) || !jx_finite3(tvx,tvy,tvz)) {
            status = JX_NONFINITE;
        } else if (!jx_tracer_acceleration(
            tx,ty,tz,sp,source_gm,source_count,&tax,&tay,&taz)) {
            status = JX_FORCE_SINGULARITY;
        }
        for (int source = 0; source < source_count; ++source) {
            const int j = source * 3;
            const double dx = tx-sp[j], dy = ty-sp[j+1], dz = tz-sp[j+2];
            const double distance = sqrt(dx*dx + dy*dy + dz*dz);
            if (distance < minimum_distance) minimum_distance = distance;
        }
        ever_unbound = !jx_bound_to_primary(
            tx,ty,tz,tvx,tvy,tvz,sp,sv,source_gm[0]);
    }
    __syncthreads();

    const double absolute_step = fabs(step);
    for (long long iteration = 0; iteration < step_count; ++iteration) {
        int newly_triggered = 0;
        if (tracer_active && status == JX_SUCCESS) {
            double best_margin = 1.0/0.0;
            int best_source = -1;
            int best_reason = JX_REASON_NONE;
            for (int source = 0; source < source_count; ++source) {
                const int j = source * 3;
                const double rx = tx-sp[j], ry = ty-sp[j+1], rz = tz-sp[j+2];
                const double vx = tvx-sv[j], vy = tvy-sv[j+1], vz = tvz-sv[j+2];
                const double r2 = rx*rx + ry*ry + rz*rz;
                const double v2 = vx*vx + vy*vy + vz*vz;
                if (!(r2 > 0.0) || !isfinite(r2) || !isfinite(v2)) {
                    status = JX_FORCE_SINGULARITY;
                    break;
                }
                const double distance = sqrt(r2);
                if (distance < minimum_distance) minimum_distance = distance;
                const double rax = tax-sa[j], ray = tay-sa[j+1], raz = taz-sa[j+2];
                const double relative_acceleration = sqrt(rax*rax + ray*ray + raz*raz);
                const double acceleration_bound = relative_acceleration_bound[source];
                int local_reason = JX_REASON_NONE;
                double margin = 1.0/0.0;
                if (!isfinite(relative_acceleration) || relative_acceleration > acceleration_bound) {
                    local_reason = JX_REASON_ACCELERATION_BOUND;
                    margin = -1.0/0.0;
                } else {
                    const double certified_lower = distance - sqrt(v2)*absolute_step
                        - 0.5*acceleration_bound*absolute_step*absolute_step;
                    double tau = 0.0;
                    if (v2 > 0.0) {
                        tau = -(rx*vx + ry*vy + rz*vz) / (v2*step);
                        if (tau < 0.0) tau = 0.0;
                        if (tau > 1.0) tau = 1.0;
                    }
                    const double sx = rx + tau*step*vx;
                    const double sy = ry + tau*step*vy;
                    const double sz = rz + tau*step*vz;
                    const double linear_distance = sqrt(sx*sx + sy*sy + sz*sz);
                    const double guarded_distance = fmin(certified_lower, linear_distance);
                    margin = guarded_distance - source_encounter_radius[source];
                    if (!(margin > 0.0)) local_reason = JX_REASON_SWEPT_FLOOR;
                }
                if (local_reason != JX_REASON_NONE && margin < best_margin) {
                    best_margin = margin;
                    best_source = source;
                    best_reason = local_reason;
                }
            }
            if (best_source >= 0 && status == JX_SUCCESS) {
                status = JX_ENCOUNTER_ROUTE;
                reason = best_reason;
                trigger_source = best_source;
                completed_steps = iteration;
                newly_triggered = 1;
                const long long offset = (long long)lane * component_count;
                for (int component = 0; component < component_count; ++component) {
                    output_trigger_source_position_0[offset+component] = sp[component];
                    output_trigger_source_velocity_0[offset+component] = sv[component];
                }
            }
        }
        __syncthreads();

        if (tid < source_count) {
            const int i = tid * 3;
            sp0[i] = sp[i]; sp0[i+1] = sp[i+1]; sp0[i+2] = sp[i+2];
            sv0[i] = sv[i]; sv0[i+1] = sv[i+1]; sv0[i+2] = sv[i+2];
            const double vhx = sv[i] + 0.5*step*sa[i];
            const double vhy = sv[i+1] + 0.5*step*sa[i+1];
            const double vhz = sv[i+2] + 0.5*step*sa[i+2];
            sv0[i] = vhx; sv0[i+1] = vhy; sv0[i+2] = vhz;
            sp[i] += step*vhx;
            sp[i+1] += step*vhy;
            sp[i+2] += step*vhz;
        }
        __syncthreads();
        if (tid < source_count) {
            const int i = tid * 3;
            double ax, ay, az;
            if (!jx_source_acceleration(sp, source_gm, source_count, tid, &ax, &ay, &az)) {
                atomicCAS(&source_status, JX_SUCCESS, JX_FORCE_SINGULARITY);
                ax = ay = az = 0.0;
            }
            sa[i] = ax; sa[i+1] = ay; sa[i+2] = az;
        }
        __syncthreads();
        if (tid < source_count) {
            const int i = tid * 3;
            sv[i] = sv0[i] + 0.5*step*sa[i];
            sv[i+1] = sv0[i+1] + 0.5*step*sa[i+1];
            sv[i+2] = sv0[i+2] + 0.5*step*sa[i+2];
        }
        __syncthreads();

        if (tracer_active && newly_triggered) {
            const long long offset = (long long)lane * component_count;
            for (int component = 0; component < component_count; ++component) {
                output_trigger_source_position_1[offset+component] = sp[component];
                output_trigger_source_velocity_1[offset+component] = sv[component];
            }
        }
        if (tracer_active && status == JX_SUCCESS) {
            tvx += 0.5*step*tax;
            tvy += 0.5*step*tay;
            tvz += 0.5*step*taz;
            tx += step*tvx;
            ty += step*tvy;
            tz += step*tvz;
            if (!jx_tracer_acceleration(
                tx,ty,tz,sp,source_gm,source_count,&tax,&tay,&taz)) {
                status = JX_FORCE_SINGULARITY;
                completed_steps = iteration + 1;
            } else {
                tvx += 0.5*step*tax;
                tvy += 0.5*step*tay;
                tvz += 0.5*step*taz;
                if (!jx_finite3(tx,ty,tz) || !jx_finite3(tvx,tvy,tvz)) {
                    status = JX_NONFINITE;
                    completed_steps = iteration + 1;
                } else {
                    completed_steps = iteration + 1;
                    for (int source = 0; source < source_count; ++source) {
                        const int j = source * 3;
                        const double dx=tx-sp[j], dy=ty-sp[j+1], dz=tz-sp[j+2];
                        const double distance=sqrt(dx*dx+dy*dy+dz*dz);
                        if (distance < minimum_distance) minimum_distance=distance;
                    }
                    if (!jx_bound_to_primary(
                        tx,ty,tz,tvx,tvy,tvz,sp,sv,source_gm[0])) ever_unbound=1;
                }
            }
        }
        __syncthreads();
    }

    if (tid < source_count) {
        const int i = tid * 3;
        const long long offset = (long long)blockIdx.x * component_count + i;
        output_source_position[offset] = sp[i];
        output_source_position[offset+1] = sp[i+1];
        output_source_position[offset+2] = sp[i+2];
        output_source_velocity[offset] = sv[i];
        output_source_velocity[offset+1] = sv[i+1];
        output_source_velocity[offset+2] = sv[i+2];
    }
    if (tracer_active) {
        const int i = lane * 3;
        output_tracer_position[i] = tx;
        output_tracer_position[i+1] = ty;
        output_tracer_position[i+2] = tz;
        output_tracer_velocity[i] = tvx;
        output_tracer_velocity[i+1] = tvy;
        output_tracer_velocity[i+2] = tvz;
        output_minimum_source_distance[lane] = minimum_distance;
        output_ever_unbound[lane] = ever_unbound;
        output_status[lane] = source_status != JX_SUCCESS ? source_status : status;
        output_reason[lane] = reason;
        output_trigger_source[lane] = trigger_source;
        output_completed_steps[lane] = completed_steps;
    }
}
'''

SOURCE_PATH_CUDA_SOURCE = r'''
#define JX_MAX_SOURCES 32
#define JX_MAX_COMPONENTS 96
#define JX_SUCCESS 0
#define JX_FORCE_SINGULARITY 3

__device__ __forceinline__ int
jx_source_path_acceleration(
    const double* position,
    const double* gm,
    const int source_count,
    const int target,
    double* ax,
    double* ay,
    double* az)
{
    const int i = target * 3;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        if (source == target) continue;
        const int j = source * 3;
        const double dx = position[j] - position[i];
        const double dy = position[j+1] - position[i+1];
        const double dz = position[j+2] - position[i+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return isfinite(x) && isfinite(y) && isfinite(z);
}

extern "C" __global__
void jx_build_massive_source_path(
    const double* input_position,
    const double* input_velocity,
    const double* gm,
    const int source_count,
    const long long step_count,
    const double step,
    double* output_position,
    double* output_velocity,
    double* output_acceleration,
    int* output_status)
{
    const int source = (int)threadIdx.x;
    __shared__ double position[JX_MAX_COMPONENTS];
    __shared__ double velocity[JX_MAX_COMPONENTS];
    __shared__ double acceleration[JX_MAX_COMPONENTS];
    __shared__ double half_velocity[JX_MAX_COMPONENTS];
    __shared__ int status;
    if (source == 0) status = JX_SUCCESS;
    if (source < source_count) {
        const int i = source * 3;
        for (int axis = 0; axis < 3; ++axis) {
            position[i+axis] = input_position[i+axis];
            velocity[i+axis] = input_velocity[i+axis];
        }
    }
    __syncthreads();
    if (source < source_count) {
        const int i = source * 3;
        double ax, ay, az;
        if (!jx_source_path_acceleration(
            position,gm,source_count,source,&ax,&ay,&az)) {
            atomicCAS(&status,JX_SUCCESS,JX_FORCE_SINGULARITY);
            ax=ay=az=0.0;
        }
        acceleration[i]=ax; acceleration[i+1]=ay; acceleration[i+2]=az;
    }
    __syncthreads();
    if (source < source_count) {
        const int i = source * 3;
        for (int axis = 0; axis < 3; ++axis) {
            output_position[i+axis] = position[i+axis];
            output_velocity[i+axis] = velocity[i+axis];
            output_acceleration[i+axis] = acceleration[i+axis];
        }
    }
    __syncthreads();
    for (long long iteration=0; iteration<step_count; ++iteration) {
        if (source < source_count) {
            const int i=source*3;
            for (int axis=0; axis<3; ++axis) {
                half_velocity[i+axis]=velocity[i+axis]+0.5*step*acceleration[i+axis];
                position[i+axis]+=step*half_velocity[i+axis];
            }
        }
        __syncthreads();
        if (source < source_count) {
            const int i=source*3;
            double ax,ay,az;
            if (!jx_source_path_acceleration(
                position,gm,source_count,source,&ax,&ay,&az)) {
                atomicCAS(&status,JX_SUCCESS,JX_FORCE_SINGULARITY);
                ax=ay=az=0.0;
            }
            acceleration[i]=ax; acceleration[i+1]=ay; acceleration[i+2]=az;
        }
        __syncthreads();
        if (source < source_count) {
            const int i=source*3;
            const long long offset=(iteration+1)*(long long)source_count*3+i;
            for (int axis=0; axis<3; ++axis) {
                velocity[i+axis]=half_velocity[i+axis]+0.5*step*acceleration[i+axis];
                output_position[offset+axis]=position[i+axis];
                output_velocity[offset+axis]=velocity[i+axis];
                output_acceleration[offset+axis]=acceleration[i+axis];
            }
        }
        __syncthreads();
    }
    if (source == 0) output_status[0]=status;
}
'''


TRACER_PATH_CUDA_SOURCE = r'''
#define JX_SUCCESS 0
#define JX_ENCOUNTER_ROUTE 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3
#define JX_REASON_NONE 0
#define JX_REASON_SWEPT_FLOOR 1
#define JX_REASON_ACCELERATION_BOUND 2

__device__ __forceinline__ int
jx_path_tracer_acceleration(
    const double tx,
    const double ty,
    const double tz,
    const double* source_position,
    const double* gm,
    const int source_count,
    double* ax,
    double* ay,
    double* az)
{
    double x=0.0,y=0.0,z=0.0;
    for (int source=0; source<source_count; ++source) {
        const int j=source*3;
        const double dx=source_position[j]-tx;
        const double dy=source_position[j+1]-ty;
        const double dz=source_position[j+2]-tz;
        const double d2=dx*dx+dy*dy+dz*dz;
        if (!(d2>0.0) || !isfinite(d2)) return 0;
        const double inv3=1.0/(d2*sqrt(d2));
        x+=gm[source]*dx*inv3;
        y+=gm[source]*dy*inv3;
        z+=gm[source]*dz*inv3;
    }
    *ax=x; *ay=y; *az=z;
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ int
jx_path_bound_to_primary(
    const double tx,const double ty,const double tz,
    const double tvx,const double tvy,const double tvz,
    const double* source_position,const double* source_velocity,
    const double primary_gm)
{
    const double rx=tx-source_position[0];
    const double ry=ty-source_position[1];
    const double rz=tz-source_position[2];
    const double vx=tvx-source_velocity[0];
    const double vy=tvy-source_velocity[1];
    const double vz=tvz-source_velocity[2];
    const double r2=rx*rx+ry*ry+rz*rz;
    if (!(r2>0.0) || !isfinite(r2)) return 0;
    const double energy=0.5*(vx*vx+vy*vy+vz*vz)-primary_gm/sqrt(r2);
    return isfinite(energy) && energy<0.0;
}

extern "C" __global__
void jx_advance_tracers_on_source_path(
    const double* source_position_path,
    const double* source_velocity_path,
    const double* source_acceleration_path,
    const double* source_gm,
    const double* source_encounter_radius,
    const double* relative_acceleration_bound,
    const double* input_tracer_position,
    const double* input_tracer_velocity,
    const int source_count,
    const int tracer_count,
    const long long step_count,
    const double step,
    double* output_tracer_position,
    double* output_tracer_velocity,
    double* output_minimum_source_distance,
    int* output_ever_unbound,
    int* output_status,
    int* output_reason,
    int* output_trigger_source,
    long long* output_completed_steps)
{
    const int lane=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if (lane>=tracer_count) return;
    const int component_count=source_count*3;
    const int i=lane*3;
    double tx=input_tracer_position[i];
    double ty=input_tracer_position[i+1];
    double tz=input_tracer_position[i+2];
    double tvx=input_tracer_velocity[i];
    double tvy=input_tracer_velocity[i+1];
    double tvz=input_tracer_velocity[i+2];
    double tax=0.0,tay=0.0,taz=0.0;
    double minimum_distance=1.0/0.0;
    int ever_unbound=0;
    int status=JX_SUCCESS;
    int reason=JX_REASON_NONE;
    int trigger_source=-1;
    long long completed_steps=0;
    const double* source_position=source_position_path;
    const double* source_velocity=source_velocity_path;
    const double* source_acceleration=source_acceleration_path;
    if (!isfinite(tx)||!isfinite(ty)||!isfinite(tz)
        ||!isfinite(tvx)||!isfinite(tvy)||!isfinite(tvz)) status=JX_NONFINITE;
    else if (!jx_path_tracer_acceleration(
        tx,ty,tz,source_position,source_gm,source_count,&tax,&tay,&taz))
        status=JX_FORCE_SINGULARITY;
    for (int source=0; source<source_count; ++source) {
        const int j=source*3;
        const double dx=tx-source_position[j];
        const double dy=ty-source_position[j+1];
        const double dz=tz-source_position[j+2];
        const double distance=sqrt(dx*dx+dy*dy+dz*dz);
        if (distance<minimum_distance) minimum_distance=distance;
    }
    ever_unbound=!jx_path_bound_to_primary(
        tx,ty,tz,tvx,tvy,tvz,source_position,source_velocity,source_gm[0]);
    const double absolute_step=fabs(step);
    for (long long iteration=0; iteration<step_count && status==JX_SUCCESS; ++iteration) {
        source_position=source_position_path+iteration*(long long)component_count;
        source_velocity=source_velocity_path+iteration*(long long)component_count;
        source_acceleration=source_acceleration_path+iteration*(long long)component_count;
        double best_margin=1.0/0.0;
        int best_source=-1;
        int best_reason=JX_REASON_NONE;
        for (int source=0; source<source_count; ++source) {
            const int j=source*3;
            const double rx=tx-source_position[j];
            const double ry=ty-source_position[j+1];
            const double rz=tz-source_position[j+2];
            const double vx=tvx-source_velocity[j];
            const double vy=tvy-source_velocity[j+1];
            const double vz=tvz-source_velocity[j+2];
            const double r2=rx*rx+ry*ry+rz*rz;
            const double v2=vx*vx+vy*vy+vz*vz;
            if (!(r2>0.0)||!isfinite(r2)||!isfinite(v2)) {
                status=JX_FORCE_SINGULARITY;
                break;
            }
            const double distance=sqrt(r2);
            if (distance<minimum_distance) minimum_distance=distance;
            const double rax=tax-source_acceleration[j];
            const double ray=tay-source_acceleration[j+1];
            const double raz=taz-source_acceleration[j+2];
            const double relative_acceleration=sqrt(rax*rax+ray*ray+raz*raz);
            const double acceleration_bound=relative_acceleration_bound[source];
            int local_reason=JX_REASON_NONE;
            double margin=1.0/0.0;
            if (!isfinite(relative_acceleration)||relative_acceleration>acceleration_bound) {
                local_reason=JX_REASON_ACCELERATION_BOUND;
                margin=-1.0/0.0;
            } else {
                const double certified_lower=distance-sqrt(v2)*absolute_step
                    -0.5*acceleration_bound*absolute_step*absolute_step;
                double tau=0.0;
                if (v2>0.0) {
                    tau=-(rx*vx+ry*vy+rz*vz)/(v2*step);
                    if (tau<0.0) tau=0.0;
                    if (tau>1.0) tau=1.0;
                }
                const double sx=rx+tau*step*vx;
                const double sy=ry+tau*step*vy;
                const double sz=rz+tau*step*vz;
                const double linear_distance=sqrt(sx*sx+sy*sy+sz*sz);
                const double guarded_distance=fmin(certified_lower,linear_distance);
                margin=guarded_distance-source_encounter_radius[source];
                if (!(margin>0.0)) local_reason=JX_REASON_SWEPT_FLOOR;
            }
            if (local_reason!=JX_REASON_NONE && margin<best_margin) {
                best_margin=margin; best_source=source; best_reason=local_reason;
            }
        }
        if (status!=JX_SUCCESS) break;
        if (best_source>=0) {
            status=JX_ENCOUNTER_ROUTE;
            reason=best_reason;
            trigger_source=best_source;
            completed_steps=iteration;
            break;
        }
        tvx+=0.5*step*tax; tvy+=0.5*step*tay; tvz+=0.5*step*taz;
        tx+=step*tvx; ty+=step*tvy; tz+=step*tvz;
        const double* next_position=source_position_path+(iteration+1)*(long long)component_count;
        const double* next_velocity=source_velocity_path+(iteration+1)*(long long)component_count;
        if (!jx_path_tracer_acceleration(
            tx,ty,tz,next_position,source_gm,source_count,&tax,&tay,&taz)) {
            status=JX_FORCE_SINGULARITY;
            completed_steps=iteration+1;
            break;
        }
        tvx+=0.5*step*tax; tvy+=0.5*step*tay; tvz+=0.5*step*taz;
        completed_steps=iteration+1;
        if (!isfinite(tx)||!isfinite(ty)||!isfinite(tz)
            ||!isfinite(tvx)||!isfinite(tvy)||!isfinite(tvz)) {
            status=JX_NONFINITE;
            break;
        }
        for (int source=0; source<source_count; ++source) {
            const int j=source*3;
            const double dx=tx-next_position[j];
            const double dy=ty-next_position[j+1];
            const double dz=tz-next_position[j+2];
            const double distance=sqrt(dx*dx+dy*dy+dz*dz);
            if (distance<minimum_distance) minimum_distance=distance;
        }
        if (!jx_path_bound_to_primary(
            tx,ty,tz,tvx,tvy,tvz,next_position,next_velocity,source_gm[0]))
            ever_unbound=1;
    }
    output_tracer_position[i]=tx;
    output_tracer_position[i+1]=ty;
    output_tracer_position[i+2]=tz;
    output_tracer_velocity[i]=tvx;
    output_tracer_velocity[i+1]=tvy;
    output_tracer_velocity[i+2]=tvz;
    output_minimum_source_distance[lane]=minimum_distance;
    output_ever_unbound[lane]=ever_unbound;
    output_status[lane]=status;
    output_reason[lane]=reason;
    output_trigger_source[lane]=trigger_source;
    output_completed_steps[lane]=completed_steps;
}
'''


KERNEL_SOURCE_SHA256 = hashlib.sha256(
    SOURCE_PATH_CUDA_SOURCE.encode("utf-8")
    + b"\x00"
    + TRACER_PATH_CUDA_SOURCE.encode("utf-8")
).hexdigest()
_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _require_device_array(
    backend: ArrayBackend,
    value: object,
    label: str,
    *,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != dtype:
        raise CUDAMasslessTracerContractError(
            f"{label} must have dtype {np.dtype(dtype).name}"
        )
    if shape is not None and array.shape != shape:
        raise CUDAMasslessTracerContractError(
            f"{label} must have shape {shape}; received {array.shape}"
        )
    if not bool(array.flags.c_contiguous):
        raise CUDAMasslessTracerContractError(f"{label} must be C-contiguous")
    return array


def _device_truth(backend: ArrayBackend, value: object, label: str) -> bool:
    return backend.scalar_bool(value, label)


def _validate_state(
    backend: ArrayBackend, state: CUDAMasslessTracerState
) -> _ValidatedState:
    if type(state) is not CUDAMasslessTracerState:
        raise CUDAMasslessTracerContractError(
            "state must be an exact CUDAMasslessTracerState"
        )
    xp = backend.xp
    source_count = len(state.source_ids)
    source_shape = (source_count, 3)
    source_positions = _require_device_array(
        backend, state.source_positions, "source_positions", dtype=xp.float64, shape=source_shape
    )
    source_velocities = _require_device_array(
        backend, state.source_velocities, "source_velocities", dtype=xp.float64, shape=source_shape
    )
    source_gm = _require_device_array(
        backend, state.source_gm, "source_gm", dtype=xp.float64, shape=(source_count,)
    )
    source_encounter_radii = _require_device_array(
        backend,
        state.source_encounter_radii,
        "source_encounter_radii",
        dtype=xp.float64,
        shape=(source_count,),
    )
    relative_acceleration_bounds = _require_device_array(
        backend,
        state.relative_acceleration_bounds,
        "relative_acceleration_bounds",
        dtype=xp.float64,
        shape=(source_count,),
    )
    tracer_positions = _require_device_array(
        backend, state.tracer_positions, "tracer_positions", dtype=xp.float64
    )
    if tracer_positions.ndim != 2 or tracer_positions.shape[1:] != (3,):
        raise CUDAMasslessTracerContractError(
            "tracer_positions must have shape (tracer_count, 3)"
        )
    tracer_count = int(tracer_positions.shape[0])
    tracer_velocities = _require_device_array(
        backend,
        state.tracer_velocities,
        "tracer_velocities",
        dtype=xp.float64,
        shape=(tracer_count, 3),
    )
    tracer_indices = _require_device_array(
        backend,
        state.tracer_indices,
        "tracer_indices",
        dtype=xp.uint64,
        shape=(tracer_count,),
    )
    floating = (
        source_positions,
        source_velocities,
        source_gm,
        source_encounter_radii,
        relative_acceleration_bounds,
        tracer_positions,
        tracer_velocities,
    )
    if any(
        _device_truth(backend, xp.any(~xp.isfinite(array)), "finite-state validation")
        for array in floating
    ):
        raise CUDAMasslessTracerContractError("all state arrays must be finite")
    if _device_truth(backend, xp.any(source_gm <= 0.0), "source GM validation"):
        raise CUDAMasslessTracerContractError("source_gm must be strictly positive")
    source_delta = source_positions[:, None, :] - source_positions[None, :, :]
    source_distance2 = xp.sum(source_delta * source_delta, axis=2)
    source_distance2 = source_distance2 + xp.eye(
        source_count, dtype=xp.float64
    )
    if _device_truth(
        backend,
        xp.any(source_distance2 <= 0.0),
        "source-separation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "distinct sources cannot occupy the same position"
        )
    if tracer_count:
        tracer_delta = (
            tracer_positions[:, None, :] - source_positions[None, :, :]
        )
        if _device_truth(
            backend,
            xp.any(xp.sum(tracer_delta * tracer_delta, axis=2) <= 0.0),
            "tracer-source separation validation",
        ):
            raise CUDAMasslessTracerContractError(
                "a tracer cannot begin at a point-mass source position"
            )
    if _device_truth(
        backend,
        xp.any(source_encounter_radii < 0.0),
        "encounter-radius validation",
    ):
        raise CUDAMasslessTracerContractError(
            "source_encounter_radii cannot be negative"
        )
    if _device_truth(
        backend,
        xp.any(relative_acceleration_bounds <= 0.0),
        "acceleration-bound validation",
    ):
        raise CUDAMasslessTracerContractError(
            "relative_acceleration_bounds must be strictly positive"
        )
    if tracer_count and int(xp.unique(tracer_indices).size) != tracer_count:
        raise CUDAMasslessTracerContractError(
            "tracer_indices must contain stable unique uint64 identities"
        )
    return _ValidatedState(
        source_positions,
        source_velocities,
        source_gm,
        source_encounter_radii,
        relative_acceleration_bounds,
        tracer_positions,
        tracer_velocities,
        tracer_indices,
        source_count,
        tracer_count,
    )


def _validate_continuation(
    backend: ArrayBackend,
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
) -> _ValidatedContinuation:
    """Validate cumulative diagnostics and their immutable route ledger."""

    cp = backend.xp
    tracer_count = validated.tracer_count
    optional_values = (
        state.minimum_source_distance_AU,
        state.ever_unbound,
        state.encounter_route_counts,
    )
    missing = tuple(value is None for value in optional_values)
    if all(missing):
        if state.global_step != 0:
            raise CUDAMasslessTracerContractError(
                "cumulative diagnostics may be absent only at global_step zero"
            )
        if state.prior_encounter_routes:
            raise CUDAMasslessTracerContractError(
                "a new state cannot have a prior route ledger without diagnostics"
            )
        return _ValidatedContinuation(
            cp.full(tracer_count, cp.inf, dtype=cp.float64),
            cp.zeros(tracer_count, dtype=cp.bool_),
            np.zeros(tracer_count, dtype=np.int64),
            (),
        )
    if any(missing):
        raise CUDAMasslessTracerContractError(
            "continuation diagnostics must be supplied together"
        )
    minimum = _require_device_array(
        backend,
        state.minimum_source_distance_AU,
        "minimum_source_distance_AU",
        dtype=cp.float64,
        shape=(tracer_count,),
    ).copy()
    if _device_truth(
        backend,
        cp.any(cp.isnan(minimum) | (minimum < 0.0)),
        "minimum-distance continuation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "minimum_source_distance_AU cannot contain NaN or negative values"
        )
    ever_unbound = _require_device_array(
        backend,
        state.ever_unbound,
        "ever_unbound",
        dtype=cp.bool_,
        shape=(tracer_count,),
    ).copy()
    route_counts = _require_device_array(
        backend,
        state.encounter_route_counts,
        "encounter_route_counts",
        dtype=cp.int64,
        shape=(tracer_count,),
    )
    if _device_truth(
        backend,
        cp.any(route_counts < 0),
        "encounter-route continuation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "encounter_route_counts cannot be negative"
        )
    route_counts_host = cp.asnumpy(route_counts)
    records = tuple(state.prior_encounter_routes)
    if any(type(row) is not TracerEncounterRouteRecord for row in records):
        raise CUDAMasslessTracerContractError(
            "prior_encounter_routes entries must be TracerEncounterRouteRecord"
        )
    canonical_records = tuple(
        sorted(
            records,
            key=lambda row: (row.global_step, row.tracer_index, row.source_index),
        )
    )
    if records != canonical_records:
        raise CUDAMasslessTracerContractError(
            "prior encounter routes must be in canonical order"
        )
    tracer_ids_host = cp.asnumpy(validated.tracer_indices)
    minimum_host = cp.asnumpy(minimum)
    ever_unbound_host = cp.asnumpy(ever_unbound)
    slot_for_tracer_id = {
        int(tracer_id): slot for slot, tracer_id in enumerate(tracer_ids_host)
    }
    ledger_counts = np.zeros(tracer_count, dtype=np.int64)
    seen_route_nodes: set[tuple[int, int]] = set()
    for record in records:
        slot = slot_for_tracer_id.get(record.tracer_index)
        if slot is None:
            raise CUDAMasslessTracerContractError(
                "a prior encounter route names a tracer outside this state"
            )
        if (
            record.source_index >= validated.source_count
            or state.source_ids[record.source_index] != record.source_id
        ):
            raise CUDAMasslessTracerContractError(
                "a prior encounter route has inconsistent source identity"
            )
        if record.global_step >= state.global_step:
            raise CUDAMasslessTracerContractError(
                "a prior encounter route is not earlier than the state endpoint"
            )
        route_node = (record.global_step, record.tracer_index)
        if route_node in seen_route_nodes:
            raise CUDAMasslessTracerContractError(
                "prior encounter route ledger contains a duplicate tracer step"
            )
        seen_route_nodes.add(route_node)
        if record.endpoint_unbound and not bool(ever_unbound_host[slot]):
            raise CUDAMasslessTracerContractError(
                "route endpoint-unbound record contradicts cumulative diagnostics"
            )
        if minimum_host[slot] > record.minimum_evaluated_stage_distance_AU:
            raise CUDAMasslessTracerContractError(
                "route evaluated-stage minimum contradicts cumulative diagnostics"
            )
        ledger_counts[slot] += 1
    if not np.array_equal(ledger_counts, route_counts_host):
        raise CUDAMasslessTracerContractError(
            "encounter-route counts and immutable route ledger disagree"
        )
    return _ValidatedContinuation(
        minimum,
        ever_unbound,
        route_counts_host,
        records,
    )


def _kernel(backend: ArrayBackend) -> Any:
    key = ("block-local-reference", int(backend.device_index or 0), KERNEL_SOURCE_SHA256)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = backend.xp.RawKernel(
            CUDA_SOURCE,
            "jx_massless_tracer_restricted_kdk",
            options=CUDA_OPTIONS,
        )
    return _KERNEL_CACHE[key]


def _path_kernels(backend: ArrayBackend) -> tuple[Any, Any]:
    key = ("shared-source-path", int(backend.device_index or 0), KERNEL_SOURCE_SHA256)
    cached = _KERNEL_CACHE.get(key)
    if cached is None or not isinstance(cached, tuple):
        source_kernel = backend.xp.RawKernel(
            SOURCE_PATH_CUDA_SOURCE,
            "jx_build_massive_source_path",
            options=CUDA_OPTIONS,
        )
        tracer_kernel = backend.xp.RawKernel(
            TRACER_PATH_CUDA_SOURCE,
            "jx_advance_tracers_on_source_path",
            options=CUDA_OPTIONS,
        )
        cached = (source_kernel, tracer_kernel)
        _KERNEL_CACHE[key] = cached
    return cached


def _build_source_path(
    backend: ArrayBackend,
    kernel: Any,
    source_positions: Any,
    source_velocities: Any,
    source_gm: Any,
    *,
    step_count: int,
    step: float,
) -> _SourcePath:
    cp = backend.xp
    source_count = int(source_positions.shape[0])
    shape = (step_count + 1, source_count, 3)
    positions = cp.empty(shape, dtype=cp.float64)
    velocities = cp.empty(shape, dtype=cp.float64)
    accelerations = cp.empty(shape, dtype=cp.float64)
    status = cp.empty(1, dtype=cp.int32)
    kernel(
        (1,),
        (32,),
        (
            source_positions,
            source_velocities,
            source_gm,
            cp.int32(source_count),
            cp.int64(step_count),
            cp.float64(step),
            positions,
            velocities,
            accelerations,
            status,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    code = int(cp.asnumpy(status[0]))
    if code != STATUS_SUCCESS:
        raise CUDAMasslessTracerIntegrationError(
            f"massive source path failed with {STATUS_NAMES.get(code, code)}"
        )
    if not _device_truth(
        backend,
        cp.all(cp.isfinite(positions))
        & cp.all(cp.isfinite(velocities))
        & cp.all(cp.isfinite(accelerations)),
        "massive source path finite-state validation",
    ):
        raise CUDAMasslessTracerIntegrationError(
            "massive source path produced a nonfinite state"
        )
    return _SourcePath(positions, velocities, accelerations)


def _launch_on_source_path(
    backend: ArrayBackend,
    kernel: Any,
    source_path: _SourcePath,
    source_gm: Any,
    encounter_radii: Any,
    acceleration_bounds: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    *,
    step: float,
    threads: int,
) -> _PathLaunchResult:
    cp = backend.xp
    source_count = int(source_path.positions.shape[1])
    step_count = int(source_path.positions.shape[0] - 1)
    tracer_count = int(tracer_positions.shape[0])
    output_position = cp.empty_like(tracer_positions)
    output_velocity = cp.empty_like(tracer_velocities)
    minimum_distance = cp.empty(tracer_count, dtype=cp.float64)
    ever_unbound = cp.empty(tracer_count, dtype=cp.int32)
    status = cp.empty(tracer_count, dtype=cp.int32)
    reason = cp.empty(tracer_count, dtype=cp.int32)
    trigger_source = cp.empty(tracer_count, dtype=cp.int32)
    completed_steps = cp.empty(tracer_count, dtype=cp.int64)
    if tracer_count:
        blocks = math.ceil(tracer_count / threads)
        kernel(
            (blocks,),
            (threads,),
            (
                source_path.positions,
                source_path.velocities,
                source_path.accelerations,
                source_gm,
                encounter_radii,
                acceleration_bounds,
                tracer_positions,
                tracer_velocities,
                cp.int32(source_count),
                cp.int32(tracer_count),
                cp.int64(step_count),
                cp.float64(step),
                output_position,
                output_velocity,
                minimum_distance,
                ever_unbound,
                status,
                reason,
                trigger_source,
                completed_steps,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()
        fatal = (status != STATUS_SUCCESS) & (status != STATUS_ENCOUNTER_ROUTE)
        if _device_truth(backend, cp.any(fatal), "CUDA path fatal-status validation"):
            lane = int(cp.asnumpy(cp.flatnonzero(fatal)[0]))
            code = int(cp.asnumpy(status[lane]))
            raise CUDAMasslessTracerIntegrationError(
                f"CUDA tracer lane {lane} failed with {STATUS_NAMES.get(code, code)}"
            )
    return _PathLaunchResult(
        output_position,
        output_velocity,
        minimum_distance,
        ever_unbound,
        status,
        reason,
        trigger_source,
        completed_steps,
    )


def _cpu_runtime_fingerprint() -> Mapping[str, Any]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            first_processor = cpuinfo.read_text(encoding="utf-8").split("\n\n", 1)[0]
        except (OSError, UnicodeError):
            first_processor = ""
        wanted = {
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "microcode",
            "flags",
        }
        for line in first_processor.splitlines():
            key, separator, value = line.partition(":")
            normalized = key.strip()
            if separator and normalized in wanted:
                fields[normalized] = " ".join(value.split())
    if not fields:
        fields = {
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    encoded = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return MappingProxyType(
        {
            "fields": fields,
            "sha256": hashlib.sha256(
                b"jxplanetx.cpu-runtime-fingerprint.v1\x00" + encoded
            ).hexdigest(),
        }
    )


def cuda_massless_tracer_runtime_identity(backend: BackendSpec) -> Mapping[str, Any]:
    """Return the exact GPU and CPU-route runtime identity for checkpoints."""

    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise CUDAMasslessTracerContractError(
            "runtime identity requires an exact CuPy BackendSpec"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    try:
        import cupy.cuda.compiler as cupy_cuda_compiler
        import scipy
        import scipy.integrate._ivp.base as scipy_ivp_base
        import scipy.integrate._ivp.common as scipy_ivp_common
        import scipy.integrate._ivp.dop853_coefficients as scipy_dop853_coefficients
        import scipy.integrate._ivp.rk as scipy_ivp_rk
    except (ImportError, OSError) as exc:  # pragma: no cover - environment failure
        raise CUDAMasslessTracerContractError(
            "SciPy is required by the declared encounter-route contract"
        ) from exc
    with resolved.activate():
        properties = cp.cuda.runtime.getDeviceProperties(resolved.device_index or 0)
        runtime_binary_paths = {
            "python_executable": Path(sys.executable),
            "numpy_multiarray_umath": Path(np._core._multiarray_umath.__file__),
            "cupy_cuda_compiler": Path(cupy_cuda_compiler.__file__),
            "cupy_cuda_runtime_extension": Path(cp.cuda.runtime.__file__),
            "cupy_nvrtc_extension": Path(cp.cuda.nvrtc.__file__),
            "scipy_ivp_base": Path(scipy_ivp_base.__file__),
            "scipy_ivp_common": Path(scipy_ivp_common.__file__),
            "scipy_dop853_coefficients": Path(
                scipy_dop853_coefficients.__file__
            ),
            "scipy_ivp_rk": Path(scipy_ivp_rk.__file__),
        }
        runtime_binary_sha256 = {
            label: _sha256_file(path)
            for label, path in runtime_binary_paths.items()
        }
        numpy_build_sha256 = hashlib.sha256(
            json.dumps(
                np.__config__.show(mode="dicts"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        scipy_build_sha256 = hashlib.sha256(
            json.dumps(
                scipy.show_config(mode="dicts"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        cpu_fingerprint = _cpu_runtime_fingerprint()
        value = {
            "method_id": METHOD_ID,
            "kernel_id": KERNEL_ID,
            "array_backend_id": backend.backend_id,
            "array_backend_device": resolved.device,
            "kernel_source_sha256": KERNEL_SOURCE_SHA256,
            "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
            "cuda_options": list(CUDA_OPTIONS),
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(
                cp.cuda.get_local_runtime_version()
            ),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc_version": [int(value) for value in cp.cuda.nvrtc.getVersion()],
            "device_index": int(resolved.device_index or 0),
            "device_name": properties["name"].decode("ascii"),
            "device_uuid": bytes(properties["uuid"]).hex(),
            "device_compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "device_pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
            "device_total_global_memory_bytes": int(properties["totalGlobalMem"]),
            "device_multiprocessor_count": int(properties["multiProcessorCount"]),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "runtime_binary_sha256": runtime_binary_sha256,
            "numpy_version": np.__version__,
            "numpy_build_configuration_sha256": numpy_build_sha256,
            "scipy_version": scipy.__version__,
            "scipy_build_configuration_sha256": scipy_build_sha256,
            "cpu_system": platform.system(),
            "cpu_release": platform.release(),
            "cpu_machine": platform.machine(),
            "cpu_processor": platform.processor(),
            "cpu_fingerprint": dict(cpu_fingerprint),
            "libc": list(platform.libc_ver()),
            "encounter_route_method": "SCIPY_DOP853_FULL_OUTER_STEP_V1",
            "determinism_scope": DETERMINISM_SCOPE,
        }
    return MappingProxyType(value)


def _launch(
    backend: ArrayBackend,
    kernel: Any,
    source_positions: Any,
    source_velocities: Any,
    source_gm: Any,
    encounter_radii: Any,
    acceleration_bounds: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    *,
    step_count: int,
    step: float,
    threads: int,
) -> _LaunchResult:
    cp = backend.xp
    source_count = int(source_positions.shape[0])
    tracer_count = int(tracer_positions.shape[0])
    blocks = max(1, math.ceil(tracer_count / threads))
    output_source_position = cp.empty((blocks, source_count, 3), dtype=cp.float64)
    output_source_velocity = cp.empty_like(output_source_position)
    output_tracer_position = cp.empty_like(tracer_positions)
    output_tracer_velocity = cp.empty_like(tracer_velocities)
    minimum_distance = cp.empty(tracer_count, dtype=cp.float64)
    ever_unbound = cp.empty(tracer_count, dtype=cp.int32)
    status = cp.empty(tracer_count, dtype=cp.int32)
    reason = cp.empty(tracer_count, dtype=cp.int32)
    trigger_source = cp.empty(tracer_count, dtype=cp.int32)
    completed_steps = cp.empty(tracer_count, dtype=cp.int64)
    trigger_shape = (tracer_count, source_count, 3)
    trigger_p0 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_v0 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_p1 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_v1 = cp.empty(trigger_shape, dtype=cp.float64)
    kernel(
        (blocks,),
        (threads,),
        (
            source_positions,
            source_velocities,
            source_gm,
            encounter_radii,
            acceleration_bounds,
            tracer_positions,
            tracer_velocities,
            cp.int32(source_count),
            cp.int32(tracer_count),
            cp.int64(step_count),
            cp.float64(step),
            output_source_position,
            output_source_velocity,
            output_tracer_position,
            output_tracer_velocity,
            minimum_distance,
            ever_unbound,
            status,
            reason,
            trigger_source,
            completed_steps,
            trigger_p0,
            trigger_v0,
            trigger_p1,
            trigger_v1,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    reference_p = output_source_position[0]
    reference_v = output_source_velocity[0]
    if not _device_truth(
        backend,
        cp.all(output_source_position == reference_p[None, :, :]),
        "source position no-backreaction invariant",
    ) or not _device_truth(
        backend,
        cp.all(output_source_velocity == reference_v[None, :, :]),
        "source velocity no-backreaction invariant",
    ):
        raise CUDAMasslessTracerIntegrationError(
            "source path differs between CUDA tracer blocks"
        )
    if tracer_count:
        fatal = (status != STATUS_SUCCESS) & (status != STATUS_ENCOUNTER_ROUTE)
        if _device_truth(backend, cp.any(fatal), "CUDA fatal-status validation"):
            lane = int(cp.asnumpy(cp.flatnonzero(fatal)[0]))
            code = int(cp.asnumpy(status[lane]))
            raise CUDAMasslessTracerIntegrationError(
                f"CUDA tracer lane {lane} failed with {STATUS_NAMES.get(code, code)}"
            )
    return _LaunchResult(
        reference_p.copy(),
        reference_v.copy(),
        output_tracer_position,
        output_tracer_velocity,
        minimum_distance,
        ever_unbound,
        status,
        reason,
        trigger_source,
        completed_steps,
        trigger_p0,
        trigger_v0,
        trigger_p1,
        trigger_v1,
    )


def _hermite_position(
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    fraction: float,
    step: float,
) -> np.ndarray:
    s = fraction
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * position_0
        + (s3 - 2.0 * s2 + s) * step * velocity_0
        + (-2.0 * s3 + 3.0 * s2) * position_1
        + (s3 - s2) * step * velocity_1
    )


def _unbound_from_primary_host(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    primary_position: np.ndarray,
    primary_velocity: np.ndarray,
    primary_gm: float,
) -> bool:
    """Apply the CUDA diagnostic's node-sampled primary-energy definition."""

    relative_position = np.asarray(tracer_position) - np.asarray(primary_position)
    relative_velocity = np.asarray(tracer_velocity) - np.asarray(primary_velocity)
    distance2 = float(np.dot(relative_position, relative_position))
    speed2 = float(np.dot(relative_velocity, relative_velocity))
    if not distance2 > 0.0 or not math.isfinite(distance2 + speed2):
        raise EncounterRouteError(
            "encounter route endpoint cannot be classified against the primary"
        )
    energy = 0.5 * speed2 - float(primary_gm) / math.sqrt(distance2)
    if not math.isfinite(energy):
        raise EncounterRouteError(
            "encounter route endpoint energy is nonfinite"
        )
    return energy >= 0.0


def _source_segment_sha256(
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    gm: np.ndarray,
    step: float,
) -> str:
    digest = hashlib.sha256(SOURCE_SEGMENT_DOMAIN)
    for label, value in (
        ("position_0", position_0),
        ("velocity_0", velocity_0),
        ("position_1", position_1),
        ("velocity_1", velocity_1),
        ("gm", gm),
    ):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(label.encode("ascii") + b"\x00")
        digest.update(str(array.shape).encode("ascii") + b"\x00")
        digest.update(array.tobytes())
    digest.update(np.asarray([step], dtype="<f8").tobytes())
    return digest.hexdigest()


def _redo_encounter_step(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    source_gm: np.ndarray,
    *,
    step: float,
    rtol: float,
    atol: float,
    maximum_substep_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Redo exactly one untouched outer step against its source segment."""

    try:
        from scipy.integrate import DOP853
    except (ImportError, OSError) as exc:  # pragma: no cover - optional environment
        raise EncounterRouteError(
            "SciPy is required only when the encounter route is exercised"
        ) from exc

    y0 = np.concatenate(
        (
            np.asarray(tracer_position, dtype=np.float64),
            np.asarray(tracer_velocity, dtype=np.float64),
        )
    )

    minimum_evaluated_stage_distance = math.inf

    def derivative(epoch: float, state: np.ndarray) -> np.ndarray:
        nonlocal minimum_evaluated_stage_distance
        fraction = epoch / step
        sources = _hermite_position(
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            fraction,
            step,
        )
        delta = sources - state[:3]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        if (
            not np.isfinite(distance2).all()
            or bool(np.any(distance2 <= 0.0))
        ):
            raise EncounterRouteError(
                "encounter route reached a point-mass singularity"
            )
        minimum_evaluated_stage_distance = min(
            minimum_evaluated_stage_distance, float(np.sqrt(np.min(distance2)))
        )
        acceleration = np.sum(
            source_gm[:, None]
            * delta
            / (distance2 * np.sqrt(distance2))[:, None],
            axis=0,
        )
        return np.concatenate((state[3:], acceleration))

    solver = DOP853(
        derivative,
        0.0,
        y0,
        step,
        rtol=rtol,
        atol=atol,
        max_step=abs(step) * maximum_substep_fraction,
    )
    accepted = 0
    while solver.status == "running":
        solver.step()
        accepted += 1
    if solver.status != "finished" or not np.isfinite(solver.y).all():
        raise EncounterRouteError(
            f"DOP853 encounter route failed: {solver.status}"
        )
    # SciPy's low-level solver exposes function evaluations but not a rejected
    # step counter.  Preserve the observable ledger and mark rejections unknown
    # with -1 rather than inventing a value.
    return (
        solver.y[:3].copy(),
        solver.y[3:].copy(),
        accepted,
        -1,
        minimum_evaluated_stage_distance,
    )


def _copy_state_with_endpoint(
    state: CUDAMasslessTracerState,
    *,
    source_positions: Any,
    source_velocities: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    minimum_source_distance_AU: Any | None = None,
    ever_unbound: Any | None = None,
    encounter_route_counts: Any | None = None,
    prior_encounter_routes: tuple[Any, ...] = (),
    step_years: float,
    step_count: int,
) -> CUDAMasslessTracerState:
    return CUDAMasslessTracerState(
        snapshot_id=state.snapshot_id,
        epoch=state.epoch + step_years * step_count,
        global_step=state.global_step + step_count,
        time_scale=state.time_scale,
        frame=state.frame,
        origin=state.origin,
        axes=state.axes,
        length_unit=state.length_unit,
        time_unit=state.time_unit,
        unit_system_id=state.unit_system_id,
        source_ids=state.source_ids,
        primary_source_id=state.primary_source_id,
        source_positions=source_positions,
        source_velocities=source_velocities,
        source_gm=state.source_gm,
        source_encounter_radii=state.source_encounter_radii,
        relative_acceleration_bounds=state.relative_acceleration_bounds,
        tracer_positions=tracer_positions,
        tracer_velocities=tracer_velocities,
        tracer_indices=state.tracer_indices,
        provenance=state.provenance,
        minimum_source_distance_AU=minimum_source_distance_AU,
        ever_unbound=ever_unbound,
        encounter_route_counts=encounter_route_counts,
        prior_encounter_routes=prior_encounter_routes,
    )


def _advance_block_local_reference(
    state: CUDAMasslessTracerState,
    spec: FixedStepCUDAMasslessTracerSpec,
) -> CUDAMasslessTracerResult:
    """Advance one transactional restricted tracer segment.

    Source and quiet-lane work remain provisional until every guarded lane has
    successfully redone its complete outer step and reached the common segment
    endpoint.  Any failure raises without returning a partially committed
    result.
    """

    if type(spec) is not FixedStepCUDAMasslessTracerSpec:
        raise CUDAMasslessTracerContractError(
            "spec must be an exact FixedStepCUDAMasslessTracerSpec"
        )
    backend = resolve_backend(spec.backend)
    cp = backend.xp
    with backend.activate():
        validated = _validate_state(backend, state)
        continuation = _validate_continuation(backend, state, validated)
        input_state_sha256 = _validated_state_sha256(
            backend, state, validated, continuation
        )
        kernel = _kernel(backend)
        main = _launch(
            backend,
            kernel,
            validated.source_positions,
            validated.source_velocities,
            validated.source_gm,
            validated.source_encounter_radii,
            validated.relative_acceleration_bounds,
            validated.tracer_positions,
            validated.tracer_velocities,
            step_count=spec.step_count,
            step=spec.step_years,
            threads=spec.threads_per_block,
        )
        final_position = cp.empty_like(validated.tracer_positions)
        final_velocity = cp.empty_like(validated.tracer_velocities)
        minimum_distance = cp.full(validated.tracer_count, cp.inf, dtype=cp.float64)
        ever_unbound = cp.zeros(validated.tracer_count, dtype=cp.bool_)
        route_counts_host = np.zeros(validated.tracer_count, dtype=np.int64)
        records: list[TracerEncounterRouteRecord] = []
        expected_source_position = main.source_positions
        expected_source_velocity = main.source_velocities
        gm_host: np.ndarray | None = None

        def commit_launch(
            launch: _LaunchResult,
            lane_slots: Any,
            tracer_indices: Any,
            *,
            absolute_start_step: int,
            steps_in_launch: int,
        ) -> None:
            nonlocal gm_host
            if not _device_truth(
                backend,
                cp.array_equal(launch.source_positions, expected_source_position),
                "routed source endpoint position identity",
            ) or not _device_truth(
                backend,
                cp.array_equal(launch.source_velocities, expected_source_velocity),
                "routed source endpoint velocity identity",
            ):
                raise EncounterRouteError(
                    "a routed lane did not rejoin the bitwise-identical source endpoint"
                )

            current_minimum = minimum_distance[lane_slots]
            minimum_distance[lane_slots] = cp.minimum(
                current_minimum, launch.minimum_source_distance
            )
            ever_unbound[lane_slots] = ever_unbound[lane_slots] | (
                launch.ever_unbound != 0
            )
            quiet = launch.status == STATUS_SUCCESS
            if _device_truth(backend, cp.any(quiet), "quiet-lane selection"):
                quiet_local = cp.flatnonzero(quiet)
                quiet_slots = lane_slots[quiet_local]
                final_position[quiet_slots] = launch.tracer_positions[quiet_local]
                final_velocity[quiet_slots] = launch.tracer_velocities[quiet_local]

            guarded_local_host = cp.asnumpy(
                cp.flatnonzero(launch.status == STATUS_ENCOUNTER_ROUTE)
            )
            for local_value in guarded_local_host:
                local = int(local_value)
                slot = int(cp.asnumpy(lane_slots[local]))
                route_counts_host[slot] += 1
                if (
                    route_counts_host[slot]
                    > spec.maximum_encounter_routes_per_tracer
                ):
                    raise EncounterRouteError(
                        f"tracer {slot} exceeded the encounter-route budget"
                    )
                completed = int(cp.asnumpy(launch.completed_steps[local]))
                source_index = int(cp.asnumpy(launch.trigger_source[local]))
                reason_code = int(cp.asnumpy(launch.reason[local]))
                if not 0 <= source_index < validated.source_count:
                    raise EncounterRouteError("CUDA returned an invalid trigger source")
                if reason_code not in REASON_NAMES or reason_code == REASON_NONE:
                    raise EncounterRouteError("CUDA returned an invalid trigger reason")
                p0 = cp.asnumpy(launch.trigger_source_position_0[local])
                v0 = cp.asnumpy(launch.trigger_source_velocity_0[local])
                p1 = cp.asnumpy(launch.trigger_source_position_1[local])
                v1 = cp.asnumpy(launch.trigger_source_velocity_1[local])
                tp0 = cp.asnumpy(launch.tracer_positions[local])
                tv0 = cp.asnumpy(launch.tracer_velocities[local])
                if gm_host is None:
                    gm_host = cp.asnumpy(validated.source_gm)
                tp1, tv1, accepted, rejected, routed_minimum = _redo_encounter_step(
                    tp0,
                    tv0,
                    p0,
                    v0,
                    p1,
                    v1,
                    gm_host,
                    step=spec.step_years,
                    rtol=spec.encounter_route_rtol,
                    atol=spec.encounter_route_atol,
                    maximum_substep_fraction=(
                        spec.encounter_route_maximum_substep_fraction
                    ),
                )
                tracer_index = int(cp.asnumpy(tracer_indices[local]))
                segment_sha = _source_segment_sha256(
                    p0, v0, p1, v1, gm_host, spec.step_years
                )
                distance = routed_minimum
                endpoint_unbound = _unbound_from_primary_host(
                    tp1, tv1, p1[0], v1[0], float(gm_host[0])
                )
                records.append(
                    TracerEncounterRouteRecord(
                        tracer_index=tracer_index,
                        source_index=source_index,
                        source_id=state.source_ids[source_index],
                        global_step=absolute_start_step + completed,
                        reason=REASON_NAMES[reason_code],
                        source_segment_sha256=segment_sha,
                        cpu_accepted_steps=accepted,
                        cpu_rejected_steps=rejected,
                        minimum_evaluated_stage_distance_AU=distance,
                        endpoint_unbound=endpoint_unbound,
                    )
                )
                minimum_distance[slot] = cp.minimum(
                    minimum_distance[slot], cp.float64(distance)
                )
                ever_unbound[slot] = ever_unbound[slot] | endpoint_unbound
                remaining = steps_in_launch - completed - 1
                if remaining < 0:
                    raise EncounterRouteError("negative routed-step remainder")
                if remaining == 0:
                    if not _device_truth(
                        backend,
                        cp.array_equal(cp.asarray(p1), expected_source_position),
                        "terminal routed source position identity",
                    ) or not _device_truth(
                        backend,
                        cp.array_equal(cp.asarray(v1), expected_source_velocity),
                        "terminal routed source velocity identity",
                    ):
                        raise EncounterRouteError(
                            "terminal routed lane missed the common source endpoint"
                        )
                    final_position[slot] = cp.asarray(tp1)
                    final_velocity[slot] = cp.asarray(tv1)
                    continue
                next_position = cp.asarray(tp1[None, :])
                next_velocity = cp.asarray(tv1[None, :])
                resumed = _launch(
                    backend,
                    kernel,
                    cp.asarray(p1),
                    cp.asarray(v1),
                    validated.source_gm,
                    validated.source_encounter_radii,
                    validated.relative_acceleration_bounds,
                    next_position,
                    next_velocity,
                    step_count=remaining,
                    step=spec.step_years,
                    threads=spec.threads_per_block,
                )
                commit_launch(
                    resumed,
                    cp.asarray([slot], dtype=cp.int64),
                    cp.asarray([tracer_index], dtype=cp.uint64),
                    absolute_start_step=absolute_start_step + completed + 1,
                    steps_in_launch=remaining,
                )

        if validated.tracer_count:
            commit_launch(
                main,
                cp.arange(validated.tracer_count, dtype=cp.int64),
                validated.tracer_indices,
                absolute_start_step=state.global_step,
                steps_in_launch=spec.step_count,
            )
        endpoint_state = _copy_state_with_endpoint(
            state,
            source_positions=expected_source_position,
            source_velocities=expected_source_velocity,
            tracer_positions=final_position,
            tracer_velocities=final_velocity,
            step_years=spec.step_years,
            step_count=spec.step_count,
        )
        runtime = cuda_massless_tracer_runtime_identity(spec.backend)
        return CUDAMasslessTracerResult(
            state=endpoint_state,
            minimum_source_distance_AU=minimum_distance,
            ever_unbound=ever_unbound,
            encounter_route_counts=cp.asarray(route_counts_host),
            encounter_routes=tuple(
                sorted(records, key=lambda row: (row.global_step, row.tracer_index))
            ),
            runtime_identity=runtime,
            input_state_sha256=input_state_sha256,
        )


def advance_cuda_massless_tracers(
    state: CUDAMasslessTracerState,
    spec: FixedStepCUDAMasslessTracerSpec,
) -> CUDAMasslessTracerResult:
    """Advance tracers over one precomputed immutable source path per chunk.

    This is the production path for the experimental v1 module.  Massive
    sources are integrated exactly once per chunk, not once per tracer block.
    A guarded tracer lane stops at its untouched outer node, its complete step
    is redone by DOP853 against two nodes of that same source path, and only a
    successfully synchronized endpoint is committed.
    """

    if type(spec) is not FixedStepCUDAMasslessTracerSpec:
        raise CUDAMasslessTracerContractError(
            "spec must be an exact FixedStepCUDAMasslessTracerSpec"
        )
    if not math.isfinite(state.epoch + spec.step_years * spec.step_count):
        raise CUDAMasslessTracerContractError(
            "the requested segment endpoint epoch is nonfinite"
        )
    backend = resolve_backend(spec.backend)
    cp = backend.xp
    with backend.activate():
        validated = _validate_state(backend, state)
        source_kernel, tracer_kernel = _path_kernels(backend)
        tracer_count = validated.tracer_count

        continuation = _validate_continuation(backend, state, validated)
        input_state_sha256 = _validated_state_sha256(
            backend, state, validated, continuation
        )
        cumulative_minimum = continuation.minimum_source_distance_AU
        cumulative_unbound = continuation.ever_unbound
        route_counts_host = continuation.encounter_route_counts_host
        records = list(continuation.encounter_routes)
        if bool(np.any(route_counts_host > spec.maximum_encounter_routes_per_tracer)):
            raise CUDAMasslessTracerContractError(
                "a cumulative encounter-route count exceeds the declared budget"
            )

        current_source_position = validated.source_positions.copy()
        current_source_velocity = validated.source_velocities.copy()
        current_tracer_position = validated.tracer_positions.copy()
        current_tracer_velocity = validated.tracer_velocities.copy()
        processed_steps = 0
        gm_host: np.ndarray | None = None

        while processed_steps < spec.step_count:
            chunk_steps = min(
                spec.source_path_steps_per_chunk,
                spec.step_count - processed_steps,
            )
            source_path = _build_source_path(
                backend,
                source_kernel,
                current_source_position,
                current_source_velocity,
                validated.source_gm,
                step_count=chunk_steps,
                step=spec.step_years,
            )
            launch = _launch_on_source_path(
                backend,
                tracer_kernel,
                source_path,
                validated.source_gm,
                validated.source_encounter_radii,
                validated.relative_acceleration_bounds,
                current_tracer_position,
                current_tracer_velocity,
                step=spec.step_years,
                threads=spec.threads_per_block,
            )
            chunk_position = cp.empty_like(current_tracer_position)
            chunk_velocity = cp.empty_like(current_tracer_velocity)

            def commit_path_launch(
                path_launch: _PathLaunchResult,
                path: _SourcePath,
                lane_slots: Any,
                tracer_indices: Any,
                *,
                absolute_start_step: int,
            ) -> None:
                nonlocal gm_host
                local_steps = int(path.positions.shape[0] - 1)
                cumulative_minimum[lane_slots] = cp.minimum(
                    cumulative_minimum[lane_slots],
                    path_launch.minimum_source_distance,
                )
                cumulative_unbound[lane_slots] = cumulative_unbound[lane_slots] | (
                    path_launch.ever_unbound != 0
                )
                quiet = path_launch.status == STATUS_SUCCESS
                if _device_truth(backend, cp.any(quiet), "quiet path-lane selection"):
                    quiet_local = cp.flatnonzero(quiet)
                    quiet_slots = lane_slots[quiet_local]
                    chunk_position[quiet_slots] = path_launch.tracer_positions[
                        quiet_local
                    ]
                    chunk_velocity[quiet_slots] = path_launch.tracer_velocities[
                        quiet_local
                    ]
                guarded_local_host = cp.asnumpy(
                    cp.flatnonzero(
                        path_launch.status == STATUS_ENCOUNTER_ROUTE
                    )
                )
                for local_value in guarded_local_host:
                    local = int(local_value)
                    slot = int(cp.asnumpy(lane_slots[local]))
                    route_counts_host[slot] += 1
                    if (
                        route_counts_host[slot]
                        > spec.maximum_encounter_routes_per_tracer
                    ):
                        raise EncounterRouteError(
                            f"tracer {slot} exceeded the encounter-route budget"
                        )
                    completed = int(cp.asnumpy(path_launch.completed_steps[local]))
                    source_index = int(cp.asnumpy(path_launch.trigger_source[local]))
                    reason_code = int(cp.asnumpy(path_launch.reason[local]))
                    if not 0 <= completed < local_steps:
                        raise EncounterRouteError(
                            "CUDA returned an invalid guarded-step index"
                        )
                    if not 0 <= source_index < validated.source_count:
                        raise EncounterRouteError(
                            "CUDA returned an invalid trigger source"
                        )
                    if reason_code not in REASON_NAMES or reason_code == REASON_NONE:
                        raise EncounterRouteError(
                            "CUDA returned an invalid trigger reason"
                        )
                    p0 = cp.asnumpy(path.positions[completed])
                    v0 = cp.asnumpy(path.velocities[completed])
                    p1 = cp.asnumpy(path.positions[completed + 1])
                    v1 = cp.asnumpy(path.velocities[completed + 1])
                    tp0 = cp.asnumpy(path_launch.tracer_positions[local])
                    tv0 = cp.asnumpy(path_launch.tracer_velocities[local])
                    if gm_host is None:
                        gm_host = cp.asnumpy(validated.source_gm)
                    tp1, tv1, accepted, rejected, routed_minimum = _redo_encounter_step(
                        tp0,
                        tv0,
                        p0,
                        v0,
                        p1,
                        v1,
                        gm_host,
                        step=spec.step_years,
                        rtol=spec.encounter_route_rtol,
                        atol=spec.encounter_route_atol,
                        maximum_substep_fraction=(
                            spec.encounter_route_maximum_substep_fraction
                        ),
                    )
                    tracer_index = int(cp.asnumpy(tracer_indices[local]))
                    distance = routed_minimum
                    endpoint_unbound = _unbound_from_primary_host(
                        tp1, tv1, p1[0], v1[0], float(gm_host[0])
                    )
                    records.append(
                        TracerEncounterRouteRecord(
                            tracer_index=tracer_index,
                            source_index=source_index,
                            source_id=state.source_ids[source_index],
                            global_step=absolute_start_step + completed,
                            reason=REASON_NAMES[reason_code],
                            source_segment_sha256=_source_segment_sha256(
                                p0,
                                v0,
                                p1,
                                v1,
                                gm_host,
                                spec.step_years,
                            ),
                            cpu_accepted_steps=accepted,
                            cpu_rejected_steps=rejected,
                            minimum_evaluated_stage_distance_AU=distance,
                            endpoint_unbound=endpoint_unbound,
                        )
                    )
                    cumulative_minimum[slot] = cp.minimum(
                        cumulative_minimum[slot], cp.float64(distance)
                    )
                    cumulative_unbound[slot] = (
                        cumulative_unbound[slot] | endpoint_unbound
                    )
                    remaining = local_steps - completed - 1
                    if remaining == 0:
                        chunk_position[slot] = cp.asarray(tp1)
                        chunk_velocity[slot] = cp.asarray(tv1)
                        continue
                    resumed_path = _SourcePath(
                        path.positions[completed + 1 :],
                        path.velocities[completed + 1 :],
                        path.accelerations[completed + 1 :],
                    )
                    resumed = _launch_on_source_path(
                        backend,
                        tracer_kernel,
                        resumed_path,
                        validated.source_gm,
                        validated.source_encounter_radii,
                        validated.relative_acceleration_bounds,
                        cp.asarray(tp1[None, :]),
                        cp.asarray(tv1[None, :]),
                        step=spec.step_years,
                        threads=spec.threads_per_block,
                    )
                    commit_path_launch(
                        resumed,
                        resumed_path,
                        cp.asarray([slot], dtype=cp.int64),
                        cp.asarray([tracer_index], dtype=cp.uint64),
                        absolute_start_step=absolute_start_step + completed + 1,
                    )

            if tracer_count:
                commit_path_launch(
                    launch,
                    source_path,
                    cp.arange(tracer_count, dtype=cp.int64),
                    validated.tracer_indices,
                    absolute_start_step=state.global_step + processed_steps,
                )
            current_source_position = source_path.positions[-1].copy()
            current_source_velocity = source_path.velocities[-1].copy()
            current_tracer_position = chunk_position
            current_tracer_velocity = chunk_velocity
            processed_steps += chunk_steps

        route_counts_device = cp.asarray(route_counts_host)
        # All gathers/scatters, diagnostic uploads, and CPU-route reinjections are part of the
        # transaction.  Surface asynchronous errors and complete the committed
        # endpoint before returning buffers to a caller or another stream.
        cp.cuda.runtime.deviceSynchronize()
        routes = tuple(
            sorted(records, key=lambda row: (row.global_step, row.tracer_index))
        )
        endpoint_state = _copy_state_with_endpoint(
            state,
            source_positions=current_source_position,
            source_velocities=current_source_velocity,
            tracer_positions=current_tracer_position,
            tracer_velocities=current_tracer_velocity,
            minimum_source_distance_AU=cumulative_minimum,
            ever_unbound=cumulative_unbound,
            encounter_route_counts=route_counts_device,
            prior_encounter_routes=routes,
            step_years=spec.step_years,
            step_count=spec.step_count,
        )
        runtime = cuda_massless_tracer_runtime_identity(spec.backend)
        return CUDAMasslessTracerResult(
            state=endpoint_state,
            minimum_source_distance_AU=cumulative_minimum,
            ever_unbound=cumulative_unbound,
            encounter_route_counts=route_counts_device,
            encounter_routes=routes,
            runtime_identity=runtime,
            input_state_sha256=input_state_sha256,
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_state_arrays(result: CUDAMasslessTracerResult) -> dict[str, np.ndarray]:
    """Perform the checkpoint API's explicit, documented device-to-host copy."""

    state = result.state
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise CUDAMasslessTracerContractError(
            "CuPy is required to dump a CUDA tracer checkpoint"
        ) from exc
    arrays = {
        "source_positions": cp.asnumpy(state.source_positions),
        "source_velocities": cp.asnumpy(state.source_velocities),
        "source_gm": cp.asnumpy(state.source_gm),
        "source_encounter_radii": cp.asnumpy(state.source_encounter_radii),
        "relative_acceleration_bounds": cp.asnumpy(
            state.relative_acceleration_bounds
        ),
        "tracer_positions": cp.asnumpy(state.tracer_positions),
        "tracer_velocities": cp.asnumpy(state.tracer_velocities),
        "tracer_indices": cp.asnumpy(state.tracer_indices),
        "minimum_source_distance_AU": cp.asnumpy(
            result.minimum_source_distance_AU
        ),
        "ever_unbound": cp.asnumpy(result.ever_unbound),
        "encounter_route_counts": cp.asnumpy(result.encounter_route_counts),
    }
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def _validated_state_sha256(
    backend: ArrayBackend,
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
    continuation: _ValidatedContinuation,
) -> str:
    """Hash the exact pre-segment numerical state for checkpoint lineage."""

    cp = backend.xp
    arrays = {
        "source_positions": cp.asnumpy(validated.source_positions),
        "source_velocities": cp.asnumpy(validated.source_velocities),
        "source_gm": cp.asnumpy(validated.source_gm),
        "source_encounter_radii": cp.asnumpy(
            validated.source_encounter_radii
        ),
        "relative_acceleration_bounds": cp.asnumpy(
            validated.relative_acceleration_bounds
        ),
        "tracer_positions": cp.asnumpy(validated.tracer_positions),
        "tracer_velocities": cp.asnumpy(validated.tracer_velocities),
        "tracer_indices": cp.asnumpy(validated.tracer_indices),
        "minimum_source_distance_AU": cp.asnumpy(
            continuation.minimum_source_distance_AU
        ),
        "ever_unbound": cp.asnumpy(continuation.ever_unbound),
        "encounter_route_counts": continuation.encounter_route_counts_host,
    }
    contiguous = {
        key: np.ascontiguousarray(value) for key, value in arrays.items()
    }
    return _state_sha256(_state_metadata(state), contiguous)


def _state_metadata(state: CUDAMasslessTracerState) -> dict[str, Any]:
    return {
        "snapshot_id": state.snapshot_id,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "time_scale": state.time_scale,
        "frame": state.frame,
        "origin": state.origin,
        "axes": state.axes,
        "length_unit": state.length_unit,
        "time_unit": state.time_unit,
        "unit_system_id": state.unit_system_id,
        "source_ids": list(state.source_ids),
        "primary_source_id": state.primary_source_id,
        "provenance": asdict(state.provenance),
    }


def _state_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(STATE_DOMAIN)
    digest.update(_canonical_json(dict(metadata)))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(_canonical_json(list(value.shape)) + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _lineage_identity_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """Bind state attributes that must never change inside a run chain."""

    invariant_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"epoch", "global_step"}
    }
    digest = hashlib.sha256(LINEAGE_IDENTITY_DOMAIN)
    digest.update(_canonical_json(invariant_metadata))
    for key in (
        "source_gm",
        "source_encounter_radii",
        "relative_acceleration_bounds",
        "tracer_indices",
    ):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(_canonical_json(list(value.shape)) + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_run_directory(
    directory: Path, run_id: str, protocol_sha256: str
) -> Path:
    digest = hashlib.sha256(
        CHECKPOINT_NAMESPACE_DOMAIN
        + run_id.encode("utf-8")
        + b"\x00"
        + protocol_sha256.encode("ascii")
    ).hexdigest()
    return directory / f"run-{digest}"


def _checkpoint_record_path(directory: Path, index: int) -> Path:
    stem = f"checkpoint_{index:020d}"
    return directory / f"{stem}.json"


def _checkpoint_array_path_from_record(
    directory: Path,
    index: int,
    record: Mapping[str, Any],
) -> Path:
    name = record.get("array_file")
    digest = record.get("array_file_sha256")
    expected_name = f"checkpoint_{index:020d}.array-{digest}.npz"
    if type(name) is not str or name != expected_name or Path(name).name != name:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint array content-addressed path identity failed"
        )
    return directory / name


def _commit_create_only(temporary: Path, destination: Path) -> None:
    """Atomically publish a same-directory file without overwrite races."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite committed artifact: {destination}"
        ) from None
    else:
        temporary.unlink()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination.parent, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


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


def _expected_claim_ceiling() -> dict[str, Any]:
    return {
        "scientific_classification": SCIENTIFIC_CLASSIFICATION,
        "project_scope": PROJECT_SCOPE,
        "claims": dict(RESULT_CLAIMS),
    }


def _validate_checkpoint_record_invariants(
    record: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> None:
    """Fail closed if any historical record used a different implementation."""

    if record.get("schema") != CHECKPOINT_SCHEMA:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history schema changed"
        )
    if record.get("method_id") != METHOD_ID:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history method changed"
        )
    runtime_identity = record.get("runtime_identity")
    if (
        type(runtime_identity) is not dict
        or runtime_identity.get("method_id") != record.get("method_id")
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint method/runtime identity is inconsistent"
        )
    if runtime_identity != dict(current_runtime):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history runtime/device identity changed"
        )
    if record.get("claim_ceiling") != _expected_claim_ceiling():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history claim ceiling changed"
        )
    routes = record.get("encounter_routes")
    if type(routes) is not list or any(type(row) is not dict for row in routes):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history route ledger is invalid"
        )
    try:
        _sha256_text(record.get("input_state_sha256"), "input_state_sha256")
        _sha256_text(
            record.get("lineage_identity_sha256"),
            "lineage_identity_sha256",
        )
    except CUDAMasslessTracerContractError as exc:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history lineage digest is invalid"
        ) from exc


def dump_cuda_massless_tracer_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    checkpoint_index: int,
    result: CUDAMasslessTracerResult,
    previous_checkpoint_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Create one immutable checkpoint pair after an explicit D2H transfer."""

    directory = Path(directory)
    _text(run_id, "run_id")
    _sha256_text(protocol_sha256, "protocol_sha256")
    if type(checkpoint_index) is not int or checkpoint_index < 0:
        raise CUDAMasslessTracerContractError(
            "checkpoint_index must be a nonnegative integer"
        )
    if type(result) is not CUDAMasslessTracerResult:
        raise CUDAMasslessTracerContractError(
            "result must be an exact CUDAMasslessTracerResult"
        )
    runtime_device_index = result.runtime_identity.get("device_index")
    if type(runtime_device_index) is not int or runtime_device_index < 0:
        raise CUDAMasslessTracerContractError(
            "result runtime identity lacks a valid CUDA device index"
        )
    checkpoint_backend = BackendSpec(
        "cupy", f"cuda:{runtime_device_index}", MAXIMUM_THREADS_PER_BLOCK
    )
    current_runtime = cuda_massless_tracer_runtime_identity(checkpoint_backend)
    if dict(result.runtime_identity) != dict(current_runtime):
        raise CUDAMasslessTracerContractError(
            "checkpoint result runtime identity differs from the current runtime"
        )
    resolved = resolve_backend(checkpoint_backend)
    cp = resolved.xp
    with resolved.activate():
        validated = _validate_state(resolved, result.state)
        continuation = _validate_continuation(resolved, result.state, validated)
        if not _device_truth(
            resolved,
            cp.array_equal(
                result.minimum_source_distance_AU,
                continuation.minimum_source_distance_AU,
            ),
            "checkpoint minimum-distance result/state identity",
        ) or not _device_truth(
            resolved,
            cp.array_equal(result.ever_unbound, continuation.ever_unbound),
            "checkpoint unbound result/state identity",
        ) or not np.array_equal(
            cp.asnumpy(result.encounter_route_counts),
            continuation.encounter_route_counts_host,
        ):
            raise CUDAMasslessTracerContractError(
                "checkpoint result diagnostics differ from its state"
            )
        if result.encounter_routes != continuation.encounter_routes:
            raise CUDAMasslessTracerContractError(
                "checkpoint result route ledger differs from its state"
            )
    if previous_checkpoint_content_sha256 is not None:
        _sha256_text(
            previous_checkpoint_content_sha256,
            "previous_checkpoint_content_sha256",
        )
    if directory.exists() and not directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint root exists but is not a directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    expected_claim_ceiling = _expected_claim_ceiling()
    observed_claim_ceiling = {
        "scientific_classification": result.scientific_classification,
        "project_scope": result.project_scope,
        "claims": dict(result.claims),
    }
    if observed_claim_ceiling != expected_claim_ceiling:
        raise CUDAMasslessTracerContractError(
            "checkpoint result exceeds the immutable claim ceiling"
        )
    if result.method_id != METHOD_ID:
        raise CUDAMasslessTracerContractError(
            "checkpoint result method_id differs from this implementation"
        )
    arrays = _host_state_arrays(result)
    metadata = _state_metadata(result.state)
    state_sha256 = _state_sha256(metadata, arrays)
    lineage_identity_sha256 = _lineage_identity_sha256(metadata, arrays)
    route_rows = [asdict(row) for row in result.encounter_routes]
    existing = _matching_checkpoint_records(directory, run_id, protocol_sha256)
    checkpoint_directory = _checkpoint_run_directory(
        directory, run_id, protocol_sha256
    )
    checkpoint_directory.mkdir(parents=False, exist_ok=True)
    previous_historical: tuple[int, Path, dict[str, Any]] | None = None
    for historical_index, historical_path, historical_record in existing:
        _validate_checkpoint_record_invariants(
            historical_record, current_runtime
        )
        historical_metadata, historical_arrays = _load_checkpoint_arrays(
            historical_path, historical_index, historical_record
        )
        if previous_historical is not None:
            prior_index, prior_path, prior_record = previous_historical
            historical_routes = historical_record.get("encounter_routes")
            if type(historical_routes) is not list:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint route ledger must be a JSON array"
                )
            _validate_checkpoint_successor(
                prior_path,
                prior_index,
                prior_record,
                current_metadata=historical_metadata,
                current_arrays=historical_arrays,
                current_routes=historical_routes,
                current_input_state_sha256=historical_record[
                    "input_state_sha256"
                ],
                current_lineage_identity_sha256=historical_record[
                    "lineage_identity_sha256"
                ],
            )
        previous_historical = (
            historical_index,
            historical_path,
            historical_record,
        )
    if existing:
        previous_index, previous_path, previous_record = existing[-1]
        if checkpoint_index <= previous_index:
            raise CUDAMasslessTracerContractError(
                "checkpoint indices must increase monotonically within a run"
            )
        if (
            previous_checkpoint_content_sha256
            != previous_record["checkpoint_content_sha256"]
        ):
            raise CUDAMasslessTracerContractError(
                "previous checkpoint digest must name the latest committed checkpoint"
            )
        previous_global_step = previous_record.get("state_metadata", {}).get(
            "global_step"
        )
        if (
            type(previous_global_step) is not int
            or result.state.global_step <= previous_global_step
        ):
            raise CUDAMasslessTracerContractError(
                "checkpoint state global_step must advance monotonically"
            )
        _validate_checkpoint_successor(
            previous_path,
            previous_index,
            previous_record,
            current_metadata=metadata,
            current_arrays=arrays,
            current_routes=route_rows,
            current_input_state_sha256=result.input_state_sha256,
            current_lineage_identity_sha256=lineage_identity_sha256,
        )
    elif previous_checkpoint_content_sha256 is not None:
        raise CUDAMasslessTracerContractError(
            "the first checkpoint in a run cannot name a predecessor"
        )
    record_path = _checkpoint_record_path(checkpoint_directory, checkpoint_index)
    if record_path.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint {checkpoint_index}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".checkpoint_{checkpoint_index:020d}.",
        suffix=".npz",
        dir=checkpoint_directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_sha256 = _sha256_file(temporary)
        array_path = checkpoint_directory / (
            f"checkpoint_{checkpoint_index:020d}.array-{array_sha256}.npz"
        )
        if array_path.exists():
            if _sha256_file(array_path) != array_sha256:
                raise CUDAMasslessTracerIntegrationError(
                    "content-addressed checkpoint array collision"
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
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "checkpoint_namespace": checkpoint_directory.name,
        "checkpoint_index": checkpoint_index,
        "array_file": array_path.name,
        "array_file_size_bytes": array_path.stat().st_size,
        "array_file_sha256": array_sha256,
        "state_metadata": metadata,
        "state_sha256": state_sha256,
        "input_state_sha256": result.input_state_sha256,
        "lineage_identity_sha256": lineage_identity_sha256,
        "method_id": result.method_id,
        "runtime_identity": dict(result.runtime_identity),
        "previous_checkpoint_content_sha256": previous_checkpoint_content_sha256,
        "encounter_routes": route_rows,
        "claim_ceiling": expected_claim_ceiling,
    }
    record["checkpoint_content_sha256"] = hashlib.sha256(
        CHECKPOINT_DOMAIN + _canonical_json(record)
    ).hexdigest()
    _atomic_json_create(record_path, record)
    return MappingProxyType(record)


def _load_checkpoint_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CUDAMasslessTracerIntegrationError(
            f"checkpoint record is unreadable: {path}"
        ) from exc
    if type(value) is not dict:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint record must be a JSON object"
        )
    recorded = value.pop("checkpoint_content_sha256", None)
    expected = hashlib.sha256(
        CHECKPOINT_DOMAIN + _canonical_json(value)
    ).hexdigest()
    if recorded != expected:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint record content hash failed"
        )
    value["checkpoint_content_sha256"] = recorded
    return value


_CHECKPOINT_ARRAY_KEYS = frozenset(
    {
        "source_positions",
        "source_velocities",
        "source_gm",
        "source_encounter_radii",
        "relative_acceleration_bounds",
        "tracer_positions",
        "tracer_velocities",
        "tracer_indices",
        "minimum_source_distance_AU",
        "ever_unbound",
        "encounter_route_counts",
    }
)


def _load_checkpoint_arrays(
    record_path: Path,
    index: int,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify and load one checkpoint's bound endpoint state."""

    array_path = _checkpoint_array_path_from_record(
        record_path.parent, index, record
    )
    if not array_path.is_file():
        raise CUDAMasslessTracerIntegrationError(
            "committed checkpoint record is missing its array file"
        )
    if array_path.stat().st_size != record.get("array_file_size_bytes"):
        raise CUDAMasslessTracerIntegrationError("checkpoint array size failed")
    if _sha256_file(array_path) != record.get("array_file_sha256"):
        raise CUDAMasslessTracerIntegrationError("checkpoint array hash failed")
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != _CHECKPOINT_ARRAY_KEYS:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint array member roster changed"
                )
            arrays = {
                key: np.ascontiguousarray(np.array(archive[key], copy=True))
                for key in _CHECKPOINT_ARRAY_KEYS
            }
    except (OSError, ValueError) as exc:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint array archive is unreadable"
        ) from exc
    metadata = record.get("state_metadata")
    if type(metadata) is not dict:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint state metadata is missing"
        )
    if _state_sha256(metadata, arrays) != record.get("state_sha256"):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint semantic state hash failed"
        )
    if _lineage_identity_sha256(metadata, arrays) != record.get(
        "lineage_identity_sha256"
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint immutable lineage identity failed"
        )
    return metadata, arrays


def _validate_checkpoint_successor(
    previous_record_path: Path,
    previous_index: int,
    previous_record: Mapping[str, Any],
    *,
    current_metadata: Mapping[str, Any],
    current_arrays: Mapping[str, np.ndarray],
    current_routes: list[dict[str, Any]],
    current_input_state_sha256: str,
    current_lineage_identity_sha256: str,
) -> None:
    """Verify that one endpoint is an actual continuation of its predecessor."""

    previous_metadata, previous_arrays = _load_checkpoint_arrays(
        previous_record_path, previous_index, previous_record
    )
    if current_input_state_sha256 != previous_record.get("state_sha256"):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor was not advanced from the prior endpoint state"
        )
    if (
        current_lineage_identity_sha256
        != previous_record.get("lineage_identity_sha256")
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor changed immutable state identity"
        )
    previous_step = previous_metadata.get("global_step")
    current_step = current_metadata.get("global_step")
    if (
        type(previous_step) is not int
        or type(current_step) is not int
        or current_step <= previous_step
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor global_step did not increase"
        )
    previous_routes = previous_record.get("encounter_routes")
    if type(previous_routes) is not list:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint predecessor route ledger is invalid"
        )
    if current_routes[: len(previous_routes)] != previous_routes:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor route ledger is not an exact prefix extension"
        )
    suffix = current_routes[len(previous_routes) :]
    for row in suffix:
        route_step = row.get("global_step") if type(row) is dict else None
        if (
            type(route_step) is not int
            or route_step < previous_step
            or route_step >= current_step
        ):
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor route ledger has an out-of-segment step"
            )
    previous_counts = previous_arrays["encounter_route_counts"]
    current_counts = current_arrays["encounter_route_counts"]
    if previous_counts.shape != current_counts.shape or np.any(
        current_counts < previous_counts
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor encounter-route counts decreased"
        )
    tracer_ids = current_arrays["tracer_indices"]
    slot_for_id = {
        int(tracer_id): slot for slot, tracer_id in enumerate(tracer_ids)
    }
    suffix_counts = np.zeros_like(current_counts)
    for row in suffix:
        slot = slot_for_id.get(row.get("tracer_index"))
        if slot is None:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor route suffix names an unknown tracer"
            )
        suffix_counts[slot] += 1
    if not np.array_equal(
        current_counts - previous_counts, suffix_counts
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor route-count delta disagrees with its ledger"
        )
    previous_minimum = previous_arrays["minimum_source_distance_AU"]
    current_minimum = current_arrays["minimum_source_distance_AU"]
    if previous_minimum.shape != current_minimum.shape or np.any(
        current_minimum > previous_minimum
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor cumulative minimum increased"
        )
    previous_unbound = previous_arrays["ever_unbound"]
    current_unbound = current_arrays["ever_unbound"]
    if previous_unbound.shape != current_unbound.shape or np.any(
        previous_unbound & ~current_unbound
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor cleared a cumulative unbound flag"
        )


def _matching_checkpoint_records(
    directory: Path,
    run_id: str,
    protocol_sha256: str,
) -> tuple[tuple[int, Path, dict[str, Any]], ...]:
    """Load, identify, order, and hash-chain one run's committed records."""

    checkpoint_directory = _checkpoint_run_directory(
        directory, run_id, protocol_sha256
    )
    if not checkpoint_directory.exists():
        return ()
    if not checkpoint_directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint run namespace exists but is not a directory"
        )
    matching: list[tuple[int, Path, dict[str, Any]]] = []
    for path in checkpoint_directory.glob("checkpoint_*.json"):
        stem = path.stem
        _, separator, suffix = stem.partition("_")
        if separator != "_" or not suffix.isdigit():
            continue
        index = int(suffix)
        record = _load_checkpoint_record(path)
        expected_path = _checkpoint_record_path(checkpoint_directory, index)
        identity = (
            record.get("schema"),
            record.get("run_id"),
            record.get("protocol_sha256"),
            record.get("checkpoint_namespace"),
            record.get("checkpoint_index"),
            path,
        )
        expected_identity = (
            CHECKPOINT_SCHEMA,
            run_id,
            protocol_sha256,
            checkpoint_directory.name,
            index,
            expected_path,
        )
        if identity != expected_identity:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint record identity or canonical path failed"
            )
        matching.append((index, path, record))
    matching.sort(key=lambda item: item[0])
    previous_digest: str | None = None
    previous_index = -1
    seen_indices: set[int] = set()
    for index, _, record in matching:
        if index in seen_indices or index <= previous_index:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint indices are duplicated or nonmonotonic"
            )
        if record.get("previous_checkpoint_content_sha256") != previous_digest:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint predecessor hash chain failed"
            )
        seen_indices.add(index)
        previous_index = index
        previous_digest = record["checkpoint_content_sha256"]
    return tuple(matching)


def load_latest_cuda_massless_tracer_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    backend: BackendSpec,
) -> CUDAMasslessTracerResult | None:
    """Load the numerically latest valid checkpoint on the same runtime/device.

    Numeric selection intentionally avoids the historical lexicographic bug at
    checkpoint 1,000 and beyond.  A malformed latest *record* fails closed;
    hidden temporary files and an array orphaned before its record was atomically
    installed do not displace the preceding committed checkpoint.
    """

    directory = Path(directory)
    _text(run_id, "run_id")
    _sha256_text(protocol_sha256, "protocol_sha256")
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise CUDAMasslessTracerContractError(
            "checkpoint load requires an exact CuPy BackendSpec"
        )
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint root exists but is not a directory"
        )
    matching = _matching_checkpoint_records(directory, run_id, protocol_sha256)
    if not matching:
        return None
    current_runtime = dict(cuda_massless_tracer_runtime_identity(backend))
    previous_entry: tuple[int, Path, dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    arrays: dict[str, np.ndarray] | None = None
    for historical_index, historical_record_path, historical_record in matching:
        _validate_checkpoint_record_invariants(
            historical_record, current_runtime
        )
        metadata, arrays = _load_checkpoint_arrays(
            historical_record_path, historical_index, historical_record
        )
        if previous_entry is not None:
            previous_index, previous_path, previous_record = previous_entry
            routes = historical_record.get("encounter_routes")
            if type(routes) is not list:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint route ledger must be a JSON array"
                )
            _validate_checkpoint_successor(
                previous_path,
                previous_index,
                previous_record,
                current_metadata=metadata,
                current_arrays=arrays,
                current_routes=routes,
                current_input_state_sha256=historical_record[
                    "input_state_sha256"
                ],
                current_lineage_identity_sha256=historical_record[
                    "lineage_identity_sha256"
                ],
            )
        previous_entry = (
            historical_index,
            historical_record_path,
            historical_record,
        )
    index, record_path, record = matching[-1]
    expected_identity = (
        CHECKPOINT_SCHEMA,
        run_id,
        protocol_sha256,
        index,
    )
    observed_identity = (
        record.get("schema"),
        record.get("run_id"),
        record.get("protocol_sha256"),
        record.get("checkpoint_index"),
    )
    if observed_identity != expected_identity:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint identity does not match the requested run"
        )
    expected_record_path = _checkpoint_record_path(record_path.parent, index)
    if record_path != expected_record_path:
        raise CUDAMasslessTracerIntegrationError("checkpoint path identity failed")
    if metadata is None or arrays is None:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history did not yield an endpoint state"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        state = CUDAMasslessTracerState(
            snapshot_id=metadata["snapshot_id"],
            epoch=metadata["epoch"],
            global_step=metadata["global_step"],
            time_scale=metadata["time_scale"],
            frame=metadata["frame"],
            origin=metadata["origin"],
            axes=metadata["axes"],
            length_unit=metadata["length_unit"],
            time_unit=metadata["time_unit"],
            unit_system_id=metadata["unit_system_id"],
            source_ids=tuple(metadata["source_ids"]),
            primary_source_id=metadata["primary_source_id"],
            source_positions=cp.asarray(arrays["source_positions"]),
            source_velocities=cp.asarray(arrays["source_velocities"]),
            source_gm=cp.asarray(arrays["source_gm"]),
            source_encounter_radii=cp.asarray(
                arrays["source_encounter_radii"]
            ),
            relative_acceleration_bounds=cp.asarray(
                arrays["relative_acceleration_bounds"]
            ),
            tracer_positions=cp.asarray(arrays["tracer_positions"]),
            tracer_velocities=cp.asarray(arrays["tracer_velocities"]),
            tracer_indices=cp.asarray(arrays["tracer_indices"]),
            provenance=Provenance(**metadata["provenance"]),
            minimum_source_distance_AU=cp.asarray(
                arrays["minimum_source_distance_AU"]
            ),
            ever_unbound=cp.asarray(arrays["ever_unbound"]),
            encounter_route_counts=cp.asarray(
                arrays["encounter_route_counts"]
            ),
            prior_encounter_routes=tuple(
                TracerEncounterRouteRecord(**row)
                for row in record.get("encounter_routes", [])
            ),
        )
        validated = _validate_state(resolved, state)
        continuation = _validate_continuation(resolved, state, validated)
        routes = continuation.encounter_routes
        return CUDAMasslessTracerResult(
            state=state,
            minimum_source_distance_AU=state.minimum_source_distance_AU,
            ever_unbound=state.ever_unbound,
            encounter_route_counts=state.encounter_route_counts,
            encounter_routes=routes,
            runtime_identity=MappingProxyType(current_runtime),
            input_state_sha256=record["input_state_sha256"],
        )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CUDA_OPTIONS",
    "CUDAMasslessTracerContractError",
    "CUDAMasslessTracerError",
    "CUDAMasslessTracerIntegrationError",
    "CUDAMasslessTracerResult",
    "CUDAMasslessTracerState",
    "DETERMINISM_SCOPE",
    "EncounterRouteError",
    "FixedStepCUDAMasslessTracerSpec",
    "IMPLEMENTATION_SOURCE_SHA256",
    "KERNEL_ID",
    "KERNEL_SOURCE_SHA256",
    "METHOD_ID",
    "PROJECT_SCOPE",
    "SCIENTIFIC_CLASSIFICATION",
    "TracerEncounterRouteRecord",
    "advance_cuda_massless_tracers",
    "cuda_massless_tracer_runtime_identity",
    "dump_cuda_massless_tracer_checkpoint",
    "load_latest_cuda_massless_tracer_checkpoint",
]
