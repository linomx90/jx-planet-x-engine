"""Supported batched CUDA evaluator for mutual point-mass EIH 1PN gravity.

The kernel evaluates the same screening-only Newtonian-plus-EIH equation as
the native CPU force core used by :mod:`jxplanetx.gr15_eih_1pn`.  It is a
force component, not yet a CUDA implementation of the complete adaptive GR15
trajectory controller.  Inputs and the returned acceleration stay on-device;
only the fail-closed status and weak-field diagnostics cross to the host.

Units are kilometres, seconds, kilometres per second, and km^3/s^2.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

from .solar_system.eih_1pn import (
    COORDINATE_SCOPE,
    EIH1PNParameters,
    SCIENTIFIC_CLAIM_STATE,
)


METHOD_ID = "JX_EIH1PN_CUDA_BATCH_V1"
MODEL_ID = "jx.eih-1pn.mutual-point-mass-total-acceleration.cuda.v1"
GR15_STAGE_METHOD_ID = "JX_GR15_EIH1PN_CUDA_STAGE_FORCE_V1"
GR15_STAGE_COUNT = 8
MINIMUM_BODY_COUNT = 2
MAXIMUM_BODY_COUNT = 32
MAXIMUM_LANE_COUNT = 65_535

STATUS_SUCCESS = 0
STATUS_INPUT_DOMAIN = 1
STATUS_NONFINITE = 2
STATUS_FORCE_SINGULARITY = 3
STATUS_WEAK_FIELD_DOMAIN = 4
STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_INPUT_DOMAIN: "INPUT_DOMAIN",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    STATUS_WEAK_FIELD_DOMAIN: "WEAK_FIELD_DOMAIN",
}

CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


class EIH1PNCUDAError(RuntimeError):
    """Base error for the supported EIH 1PN CUDA force component."""


class EIH1PNCUDAUnavailableError(EIH1PNCUDAError):
    """CuPy or a CUDA device is unavailable to the current process."""


class EIH1PNCUDAContractError(EIH1PNCUDAError, ValueError):
    """A caller violated the explicit device-array component contract."""


class EIH1PNCUDAEvaluationError(EIH1PNCUDAError):
    """A CUDA lane failed closed while evaluating the force equation."""

    def __init__(self, message: str, *, lane: int, status: int) -> None:
        super().__init__(message)
        self.lane = lane
        self.status = status
        self.status_name = STATUS_NAMES.get(status, "UNKNOWN")


CUDA_SOURCE = r'''
#define JX_MAX_BODIES 32
#define JX_MAX_COMPONENTS 96

#define JX_SUCCESS 0
#define JX_INPUT_DOMAIN 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3
#define JX_WEAK_FIELD_DOMAIN 4

__device__ __forceinline__ void
jx_set_status_once(int* status, const int value)
{
    atomicCAS(status, JX_SUCCESS, value);
}

extern "C" __global__
void jx_eih_1pn_total_acceleration_batch(
    const double* input_position,
    const double* input_velocity,
    const double* input_gm,
    const double speed_of_light_squared,
    const double maximum_compactness,
    const double maximum_speed_fraction_squared,
    const int body_count,
    const int lane_count,
    const int gm_is_batched,
    double* output_acceleration,
    double* output_diagnostics,
    int* output_status)
{
    const int lane = (int)blockIdx.x;
    const int body = (int)threadIdx.x;
    const int active = body < body_count;
    const int component_count = body_count * 3;

    __shared__ double position[JX_MAX_COMPONENTS];
    __shared__ double velocity[JX_MAX_COMPONENTS];
    __shared__ double gm[JX_MAX_BODIES];
    __shared__ double newtonian[JX_MAX_COMPONENTS];
    __shared__ double separation[JX_MAX_BODIES * JX_MAX_BODIES];
    __shared__ double potential[JX_MAX_BODIES];
    __shared__ double compactness[JX_MAX_BODIES];
    __shared__ double speed_fraction_squared[JX_MAX_BODIES];
    __shared__ int lane_status;

    if (lane >= lane_count) return;
    if (body == 0) lane_status = JX_SUCCESS;
    __syncthreads();

    if (active) {
        const int lane_component = lane * component_count + body * 3;
        const int gm_offset = gm_is_batched ? lane * body_count : 0;
        const double current_gm = input_gm[gm_offset + body];
        gm[body] = current_gm;
        if (!isfinite(current_gm) || !(current_gm > 0.0)) {
            jx_set_status_once(&lane_status, JX_INPUT_DOMAIN);
        }
        for (int axis = 0; axis < 3; ++axis) {
            const double current_position = input_position[lane_component + axis];
            const double current_velocity = input_velocity[lane_component + axis];
            position[body * 3 + axis] = current_position;
            velocity[body * 3 + axis] = current_velocity;
            if (!isfinite(current_position) || !isfinite(current_velocity)) {
                jx_set_status_once(&lane_status, JX_NONFINITE);
            }
        }
    }
    __syncthreads();

    if (active && lane_status == JX_SUCCESS) {
        double acceleration_x = 0.0;
        double acceleration_y = 0.0;
        double acceleration_z = 0.0;
        double current_potential = 0.0;
        for (int source = 0; source < body_count; ++source) {
            if (source == body) {
                separation[body * JX_MAX_BODIES + source] = 0.0;
                continue;
            }
            const double dx = position[source * 3] - position[body * 3];
            const double dy = position[source * 3 + 1] - position[body * 3 + 1];
            const double dz = position[source * 3 + 2] - position[body * 3 + 2];
            const double radius_squared = dx * dx + dy * dy + dz * dz;
            if (!(radius_squared > 0.0) || !isfinite(radius_squared)) {
                jx_set_status_once(&lane_status, JX_FORCE_SINGULARITY);
                continue;
            }
            const double radius = sqrt(radius_squared);
            const double inverse_radius_cubed = 1.0 / (radius_squared * radius);
            if (!isfinite(inverse_radius_cubed)) {
                jx_set_status_once(&lane_status, JX_NONFINITE);
                continue;
            }
            separation[body * JX_MAX_BODIES + source] = radius;
            current_potential += gm[source] / radius;
            const double weight = gm[source] * inverse_radius_cubed;
            acceleration_x += weight * dx;
            acceleration_y += weight * dy;
            acceleration_z += weight * dz;
        }
        newtonian[body * 3] = acceleration_x;
        newtonian[body * 3 + 1] = acceleration_y;
        newtonian[body * 3 + 2] = acceleration_z;
        potential[body] = current_potential;
        const double vx = velocity[body * 3];
        const double vy = velocity[body * 3 + 1];
        const double vz = velocity[body * 3 + 2];
        compactness[body] = current_potential / speed_of_light_squared;
        speed_fraction_squared[body] =
            (vx * vx + vy * vy + vz * vz) / speed_of_light_squared;
    }
    __syncthreads();

    if (body == 0 && lane_status == JX_SUCCESS) {
        double observed_compactness = 0.0;
        double observed_speed_fraction_squared = 0.0;
        for (int index = 0; index < body_count; ++index) {
            observed_compactness = fmax(
                observed_compactness, compactness[index]);
            observed_speed_fraction_squared = fmax(
                observed_speed_fraction_squared,
                speed_fraction_squared[index]);
        }
        output_diagnostics[lane * 2] = observed_compactness;
        output_diagnostics[lane * 2 + 1] = observed_speed_fraction_squared;
        if (!isfinite(observed_compactness)
            || !isfinite(observed_speed_fraction_squared)) {
            lane_status = JX_NONFINITE;
        } else if (observed_compactness > maximum_compactness
            || observed_speed_fraction_squared
                > maximum_speed_fraction_squared) {
            lane_status = JX_WEAK_FIELD_DOMAIN;
        }
    }
    __syncthreads();

    if (active && lane_status == JX_SUCCESS) {
        const double body_vx = velocity[body * 3];
        const double body_vy = velocity[body * 3 + 1];
        const double body_vz = velocity[body * 3 + 2];
        const double body_speed_squared =
            body_vx * body_vx + body_vy * body_vy + body_vz * body_vz;
        double acceleration_x = newtonian[body * 3];
        double acceleration_y = newtonian[body * 3 + 1];
        double acceleration_z = newtonian[body * 3 + 2];
        for (int source = 0; source < body_count; ++source) {
            if (source == body) continue;
            const double difference_x =
                position[body * 3] - position[source * 3];
            const double difference_y =
                position[body * 3 + 1] - position[source * 3 + 1];
            const double difference_z =
                position[body * 3 + 2] - position[source * 3 + 2];
            const double source_vx = velocity[source * 3];
            const double source_vy = velocity[source * 3 + 1];
            const double source_vz = velocity[source * 3 + 2];
            const double source_speed_squared =
                source_vx * source_vx + source_vy * source_vy
                + source_vz * source_vz;
            const double body_source_velocity_dot =
                body_vx * source_vx + body_vy * source_vy
                + body_vz * source_vz;
            const double radial_source_velocity =
                difference_x * source_vx + difference_y * source_vy
                + difference_z * source_vz;
            const double radius =
                separation[body * JX_MAX_BODIES + source];
            const double radius_squared = radius * radius;
            const double radius_cubed = radius_squared * radius;
            const double source_acceleration_dot =
                difference_x * newtonian[source * 3]
                + difference_y * newtonian[source * 3 + 1]
                + difference_z * newtonian[source * 3 + 2];
            const double bracket = (
                4.0 * potential[body]
                + potential[source]
                - body_speed_squared
                - 2.0 * source_speed_squared
                + 4.0 * body_source_velocity_dot
                + 1.5 * radial_source_velocity * radial_source_velocity
                    / radius_squared
                + 0.5 * source_acceleration_dot
            ) / speed_of_light_squared;
            const double velocity_bracket =
                difference_x * (4.0 * body_vx - 3.0 * source_vx)
                + difference_y * (4.0 * body_vy - 3.0 * source_vy)
                + difference_z * (4.0 * body_vz - 3.0 * source_vz);
            const double common_position = gm[source] * bracket / radius_cubed;
            const double common_velocity =
                gm[source] * velocity_bracket
                / (radius_cubed * speed_of_light_squared);
            const double common_source_acceleration =
                gm[source] * 3.5 / (radius * speed_of_light_squared);
            acceleration_x +=
                common_position * difference_x
                + common_velocity * (body_vx - source_vx)
                + common_source_acceleration * newtonian[source * 3];
            acceleration_y +=
                common_position * difference_y
                + common_velocity * (body_vy - source_vy)
                + common_source_acceleration * newtonian[source * 3 + 1];
            acceleration_z +=
                common_position * difference_z
                + common_velocity * (body_vz - source_vz)
                + common_source_acceleration * newtonian[source * 3 + 2];
        }
        if (!isfinite(acceleration_x) || !isfinite(acceleration_y)
            || !isfinite(acceleration_z)) {
            jx_set_status_once(&lane_status, JX_NONFINITE);
        }
        const int output_offset = lane * component_count + body * 3;
        output_acceleration[output_offset] = acceleration_x;
        output_acceleration[output_offset + 1] = acceleration_y;
        output_acceleration[output_offset + 2] = acceleration_z;
    }
    __syncthreads();
    if (body == 0) output_status[lane] = lane_status;
}
'''

CUDA_SOURCE_SHA256 = hashlib.sha256(CUDA_SOURCE.encode("utf-8")).hexdigest()

_CUDA_KERNEL: Any | None = None


@dataclass(frozen=True, slots=True)
class EIH1PNCUDABatchResult:
    """Device-resident accelerations plus host-audited per-lane diagnostics."""

    accelerations_km_s2: Any
    maximum_compactness_by_lane: tuple[float, ...]
    maximum_speed_fraction_squared_by_lane: tuple[float, ...]
    lane_count: int
    body_count: int
    method_id: str = METHOD_ID
    model_id: str = MODEL_ID
    coordinate_scope: str = COORDINATE_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    kernel_launch_count: int = 1
    host_audit_performed: bool = True
    full_gr15_cuda_claimed: bool = False
    production_authorized: bool = False
    exact_general_relativity_claimed: bool = False

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.model_id != MODEL_ID:
            raise EIH1PNCUDAContractError("CUDA result identity changed")
        if self.coordinate_scope != COORDINATE_SCOPE:
            raise EIH1PNCUDAContractError("CUDA coordinate scope changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise EIH1PNCUDAContractError("CUDA claim state changed")
        if self.kernel_launch_count != 1 or not self.host_audit_performed:
            raise EIH1PNCUDAContractError("CUDA execution accounting is invalid")
        if (
            self.full_gr15_cuda_claimed
            or self.production_authorized
            or self.exact_general_relativity_claimed
        ):
            raise EIH1PNCUDAContractError("CUDA force result elevated a claim")
        if len(self.maximum_compactness_by_lane) != self.lane_count or len(
            self.maximum_speed_fraction_squared_by_lane
        ) != self.lane_count:
            raise EIH1PNCUDAContractError("CUDA diagnostic shape changed")


@dataclass(frozen=True, slots=True)
class GR15EIH1PNCUDAStageResult:
    """One batched EIH force sweep over all eight GR15 corrector stages."""

    stage_accelerations_km_s2: Any
    maximum_compactness_by_system_stage: tuple[tuple[float, ...], ...]
    maximum_speed_fraction_squared_by_system_stage: tuple[
        tuple[float, ...], ...
    ]
    system_count: int
    body_count: int
    stage_count: int = GR15_STAGE_COUNT
    method_id: str = GR15_STAGE_METHOD_ID
    force_method_id: str = METHOD_ID
    model_id: str = MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    kernel_launch_count: int = 1
    gm_device_expansion_performed: bool = False
    complete_corrector_claimed: bool = False
    adaptive_controller_claimed: bool = False
    full_gr15_cuda_claimed: bool = False
    production_authorized: bool = False

    def __post_init__(self) -> None:
        if self.method_id != GR15_STAGE_METHOD_ID or self.force_method_id != METHOD_ID:
            raise EIH1PNCUDAContractError("GR15 CUDA stage identity changed")
        if self.model_id != MODEL_ID or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise EIH1PNCUDAContractError("GR15 CUDA stage claim identity changed")
        if self.stage_count != GR15_STAGE_COUNT or self.kernel_launch_count != 1:
            raise EIH1PNCUDAContractError("GR15 CUDA stage accounting changed")
        if (
            self.complete_corrector_claimed
            or self.adaptive_controller_claimed
            or self.full_gr15_cuda_claimed
            or self.production_authorized
        ):
            raise EIH1PNCUDAContractError("GR15 CUDA stage result elevated a claim")
        expected_shape = (self.system_count, self.stage_count)
        for diagnostics in (
            self.maximum_compactness_by_system_stage,
            self.maximum_speed_fraction_squared_by_system_stage,
        ):
            if len(diagnostics) != expected_shape[0] or any(
                len(row) != expected_shape[1] for row in diagnostics
            ):
                raise EIH1PNCUDAContractError(
                    "GR15 CUDA stage diagnostic shape changed"
                )


def _cupy() -> Any:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - installation dependent
        raise EIH1PNCUDAUnavailableError(f"CuPy is unavailable: {exc}") from exc
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            raise EIH1PNCUDAUnavailableError(
                "no CUDA device is visible to the current process"
            )
    except EIH1PNCUDAUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - driver dependent
        raise EIH1PNCUDAUnavailableError(
            f"the CUDA runtime is unavailable: {exc}"
        ) from exc
    return cp


def _require_device_state(cp: Any, value: Any, label: str) -> Any:
    if type(value) is not cp.ndarray:
        raise EIH1PNCUDAContractError(f"{label} must be an exact CuPy array")
    if value.dtype != cp.dtype(cp.float64) or value.ndim != 3 or value.shape[2] != 3:
        raise EIH1PNCUDAContractError(
            f"{label} must have CuPy float64 shape (lanes,bodies,3)"
        )
    if not value.flags.c_contiguous:
        raise EIH1PNCUDAContractError(f"{label} must be C-contiguous")
    return value


def _kernel(cp: Any) -> Any:
    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        try:
            _CUDA_KERNEL = cp.RawKernel(
                CUDA_SOURCE,
                "jx_eih_1pn_total_acceleration_batch",
                options=CUDA_OPTIONS,
            )
        except Exception as exc:  # pragma: no cover - compiler dependent
            raise EIH1PNCUDAUnavailableError(
                f"the EIH 1PN CUDA kernel could not be constructed: {exc}"
            ) from exc
    return _CUDA_KERNEL


def eih_1pn_cuda_runtime_identity() -> dict[str, Any]:
    """Return the exact CUDA source and active runtime/device identity."""

    cp = _cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", b"UNKNOWN")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "method_id": METHOD_ID,
        "model_id": MODEL_ID,
        "cuda_source_sha256": CUDA_SOURCE_SHA256,
        "cuda_options": CUDA_OPTIONS,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device_id": int(device.id),
        "device_name": str(name),
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
    }


def evaluate_eih_1pn_total_acceleration_cuda(
    positions_km: Any,
    velocities_km_s: Any,
    gravitational_parameters_km3_s2: Any,
    parameters: EIH1PNParameters,
) -> EIH1PNCUDABatchResult:
    """Evaluate batched total Newtonian-plus-EIH acceleration on one GPU.

    All three input arrays must already be CuPy float64 arrays on the current
    device.  Position and velocity shapes are ``(lanes, bodies, 3)``.  The GM
    array may be shared with shape ``(bodies,)`` or lane-specific with shape
    ``(lanes, bodies)``.  No implicit host/device transfer is performed.
    """

    if type(parameters) is not EIH1PNParameters:
        raise EIH1PNCUDAContractError(
            "parameters must be an exact EIH1PNParameters"
        )
    cp = _cupy()
    positions = _require_device_state(cp, positions_km, "positions_km")
    velocities = _require_device_state(cp, velocities_km_s, "velocities_km_s")
    if velocities.shape != positions.shape:
        raise EIH1PNCUDAContractError("position and velocity shapes must match")
    lane_count, body_count, _ = positions.shape
    if not 1 <= lane_count <= MAXIMUM_LANE_COUNT:
        raise EIH1PNCUDAContractError(
            f"lane count must be 1--{MAXIMUM_LANE_COUNT}"
        )
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise EIH1PNCUDAContractError("EIH 1PN CUDA supports 2--32 bodies")

    gm = gravitational_parameters_km3_s2
    if type(gm) is not cp.ndarray or gm.dtype != cp.dtype(cp.float64):
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be an exact CuPy float64 array"
        )
    if gm.shape == (body_count,):
        gm_is_batched = 0
    elif gm.shape == (lane_count, body_count):
        gm_is_batched = 1
    else:
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must have shape (bodies,) "
            "or (lanes,bodies)"
        )
    if not gm.flags.c_contiguous:
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be C-contiguous"
        )

    speed_of_light_squared = (
        parameters.speed_of_light_km_s * parameters.speed_of_light_km_s
    )
    if not math.isfinite(speed_of_light_squared):
        raise EIH1PNCUDAContractError(
            "speed_of_light_km_s squared must remain finite"
        )

    acceleration = cp.empty_like(positions)
    diagnostics = cp.empty((lane_count, 2), dtype=cp.float64)
    status = cp.empty(lane_count, dtype=cp.int32)
    try:
        _kernel(cp)(
            (lane_count,),
            (MAXIMUM_BODY_COUNT,),
            (
                positions,
                velocities,
                gm,
                float(speed_of_light_squared),
                parameters.maximum_compactness,
                parameters.maximum_speed_fraction_squared,
                body_count,
                lane_count,
                gm_is_batched,
                acceleration,
                diagnostics,
                status,
            ),
        )
        status_host = cp.asnumpy(status)
        diagnostics_host = cp.asnumpy(diagnostics)
    except EIH1PNCUDAError:
        raise
    except Exception as exc:  # pragma: no cover - driver dependent
        raise EIH1PNCUDAUnavailableError(
            f"the EIH 1PN CUDA kernel failed to execute: {exc}"
        ) from exc

    for lane, current_status in enumerate(status_host.tolist()):
        current_status = int(current_status)
        if current_status != STATUS_SUCCESS:
            status_name = STATUS_NAMES.get(current_status, "UNKNOWN")
            raise EIH1PNCUDAEvaluationError(
                f"EIH 1PN CUDA lane {lane} failed with {status_name}",
                lane=lane,
                status=current_status,
            )

    compactness = tuple(float(value) for value in diagnostics_host[:, 0])
    speed_fractions = tuple(float(value) for value in diagnostics_host[:, 1])
    return EIH1PNCUDABatchResult(
        accelerations_km_s2=acceleration,
        maximum_compactness_by_lane=compactness,
        maximum_speed_fraction_squared_by_lane=speed_fractions,
        lane_count=lane_count,
        body_count=body_count,
    )


def evaluate_gr15_eih_1pn_stages_cuda(
    stage_positions_km: Any,
    stage_velocities_km_s: Any,
    gravitational_parameters_km3_s2: Any,
    parameters: EIH1PNParameters,
) -> GR15EIH1PNCUDAStageResult:
    """Evaluate one force sweep for every stage of batched GR15 trials.

    Stage states must have CuPy float64 shape ``(systems, 8, bodies, 3)``.
    The GM array may be shared across all systems with shape ``(bodies,)`` or
    system-specific with shape ``(systems, bodies)``.  System-specific masses
    are expanded across the eight stages on-device before the single force
    kernel launch.  This component does not perform the GR15 corrector update,
    convergence decision, error estimate, rejection, or adaptive controller.
    """

    if type(parameters) is not EIH1PNParameters:
        raise EIH1PNCUDAContractError(
            "parameters must be an exact EIH1PNParameters"
        )
    cp = _cupy()
    for value, label in (
        (stage_positions_km, "stage_positions_km"),
        (stage_velocities_km_s, "stage_velocities_km_s"),
    ):
        if type(value) is not cp.ndarray:
            raise EIH1PNCUDAContractError(f"{label} must be an exact CuPy array")
        if (
            value.dtype != cp.dtype(cp.float64)
            or value.ndim != 4
            or value.shape[1] != GR15_STAGE_COUNT
            or value.shape[3] != 3
        ):
            raise EIH1PNCUDAContractError(
                f"{label} must have CuPy float64 shape (systems,8,bodies,3)"
            )
        if not value.flags.c_contiguous:
            raise EIH1PNCUDAContractError(f"{label} must be C-contiguous")
    if stage_velocities_km_s.shape != stage_positions_km.shape:
        raise EIH1PNCUDAContractError("GR15 stage position/velocity shapes must match")
    system_count, _, body_count, _ = stage_positions_km.shape
    if system_count < 1 or system_count * GR15_STAGE_COUNT > MAXIMUM_LANE_COUNT:
        raise EIH1PNCUDAContractError(
            f"GR15 CUDA stage systems must be 1--{MAXIMUM_LANE_COUNT // GR15_STAGE_COUNT}"
        )
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise EIH1PNCUDAContractError("GR15 EIH CUDA stages support 2--32 bodies")

    gm = gravitational_parameters_km3_s2
    if type(gm) is not cp.ndarray or gm.dtype != cp.dtype(cp.float64):
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be an exact CuPy float64 array"
        )
    gm_expanded = False
    if gm.shape == (body_count,):
        launch_gm = gm
    elif gm.shape == (system_count, body_count):
        if not gm.flags.c_contiguous:
            raise EIH1PNCUDAContractError(
                "gravitational_parameters_km3_s2 must be C-contiguous"
            )
        launch_gm = cp.repeat(gm, GR15_STAGE_COUNT, axis=0)
        gm_expanded = True
    else:
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must have shape (bodies,) "
            "or (systems,bodies)"
        )
    if not launch_gm.flags.c_contiguous:
        raise EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be C-contiguous"
        )

    lane_count = system_count * GR15_STAGE_COUNT
    flat_positions = stage_positions_km.reshape(lane_count, body_count, 3)
    flat_velocities = stage_velocities_km_s.reshape(lane_count, body_count, 3)
    force_result = evaluate_eih_1pn_total_acceleration_cuda(
        flat_positions,
        flat_velocities,
        launch_gm,
        parameters,
    )

    def diagnostic_rows(values: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(values[index * GR15_STAGE_COUNT : (index + 1) * GR15_STAGE_COUNT])
            for index in range(system_count)
        )

    return GR15EIH1PNCUDAStageResult(
        stage_accelerations_km_s2=force_result.accelerations_km_s2.reshape(
            system_count, GR15_STAGE_COUNT, body_count, 3
        ),
        maximum_compactness_by_system_stage=diagnostic_rows(
            force_result.maximum_compactness_by_lane
        ),
        maximum_speed_fraction_squared_by_system_stage=diagnostic_rows(
            force_result.maximum_speed_fraction_squared_by_lane
        ),
        system_count=system_count,
        body_count=body_count,
        gm_device_expansion_performed=gm_expanded,
    )


__all__ = [
    "CUDA_OPTIONS",
    "CUDA_SOURCE_SHA256",
    "EIH1PNCUDABatchResult",
    "EIH1PNCUDAContractError",
    "EIH1PNCUDAError",
    "EIH1PNCUDAEvaluationError",
    "EIH1PNCUDAUnavailableError",
    "GR15EIH1PNCUDAStageResult",
    "GR15_STAGE_COUNT",
    "GR15_STAGE_METHOD_ID",
    "MAXIMUM_BODY_COUNT",
    "MAXIMUM_LANE_COUNT",
    "METHOD_ID",
    "MODEL_ID",
    "STATUS_FORCE_SINGULARITY",
    "STATUS_INPUT_DOMAIN",
    "STATUS_NAMES",
    "STATUS_NONFINITE",
    "STATUS_SUCCESS",
    "STATUS_WEAK_FIELD_DOMAIN",
    "eih_1pn_cuda_runtime_identity",
    "evaluate_eih_1pn_total_acceleration_cuda",
    "evaluate_gr15_eih_1pn_stages_cuda",
]
