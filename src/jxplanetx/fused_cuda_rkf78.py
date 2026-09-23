#!/usr/bin/env python3
"""Package-owned fused CUDA core for one JX Fehlberg RKF78 step.

This is an opt-in experimental engine component, not a qualified backend. One CUDA
block advances one independent, fully mutual, unsoftened Newtonian system.  A
single kernel evaluates all 13 RKF78 stages and applies the accepted eighth-
order update with componentwise Kahan carries.  The implementation is limited
to 2--32 bodies so every lane fits in one block's shared memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from jxplanetx.engine import BackendSpec
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine import rkf78


MAX_BODIES = 32
STAGE_COUNT = 13
MAX_COMPONENTS = MAX_BODIES * 3
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--ftz=false")


class FusedPrototypeError(RuntimeError):
    """The fused prototype request or execution is invalid."""


@dataclass(frozen=True)
class FusedStepResult:
    positions: Any
    velocities: Any
    position_carry: Any
    velocity_carry: Any
    position_defect: Any
    velocity_defect: Any
    status: Any
    body_count: int
    lane_count: int
    kernel_launch_count: int = 1


CUDA_SOURCE = r'''
#define JX_STAGE_COUNT 13
#define JX_MAX_BODIES 32
#define JX_MAX_COMPONENTS 96

extern "C" __global__
void jx_fused_rkf78_step(
    const double* input_position,
    const double* input_velocity,
    const double* input_gm,
    const double* input_position_carry,
    const double* input_velocity_carry,
    const double* tableau_a,
    const double* weight_eighth,
    const double* weight_defect,
    const double step,
    double* output_position,
    double* output_velocity,
    double* output_position_carry,
    double* output_velocity_carry,
    double* output_position_defect,
    double* output_velocity_defect,
    int* output_status,
    const int body_count,
    const int lane_count)
{
    const int lane = (int)blockIdx.x;
    const int component = (int)threadIdx.x;
    const int component_count = body_count * 3;
    if (lane >= lane_count || component >= component_count) return;

    __shared__ double position[JX_MAX_COMPONENTS];
    __shared__ double velocity[JX_MAX_COMPONENTS];
    __shared__ double position_carry[JX_MAX_COMPONENTS];
    __shared__ double velocity_carry[JX_MAX_COMPONENTS];
    __shared__ double stage_position[JX_MAX_COMPONENTS];
    __shared__ double stage_velocity[JX_MAX_COMPONENTS];
    __shared__ double k_position[JX_STAGE_COUNT * JX_MAX_COMPONENTS];
    __shared__ double k_velocity[JX_STAGE_COUNT * JX_MAX_COMPONENTS];
    __shared__ int lane_status;

    const int lane_component = lane * component_count + component;
    position[component] = input_position[lane_component];
    velocity[component] = input_velocity[lane_component];
    position_carry[component] = input_position_carry[lane_component];
    velocity_carry[component] = input_velocity_carry[lane_component];
    if (component == 0) lane_status = 0;
    __syncthreads();

    const int body = component / 3;
    const int axis = component - body * 3;
    const int gm_offset = lane * body_count;

    for (int stage = 0; stage < JX_STAGE_COUNT; ++stage) {
        if (stage == 0) {
            stage_position[component] = position[component];
            stage_velocity[component] = velocity[component];
        } else {
            double position_sum = 0.0;
            double velocity_sum = 0.0;
            for (int previous = 0; previous < stage; ++previous) {
                const double coefficient = tableau_a[stage * JX_STAGE_COUNT + previous];
                if (coefficient != 0.0) {
                    position_sum += coefficient
                        * k_position[previous * JX_MAX_COMPONENTS + component];
                    velocity_sum += coefficient
                        * k_velocity[previous * JX_MAX_COMPONENTS + component];
                }
            }
            stage_position[component] = position[component] + step * position_sum;
            stage_velocity[component] = velocity[component] + step * velocity_sum;
        }
        __syncthreads();

        k_position[stage * JX_MAX_COMPONENTS + component] = stage_velocity[component];
        double acceleration = 0.0;
        for (int source = 0; source < body_count; ++source) {
            if (source == body) continue;
            const double dx = stage_position[source * 3]
                - stage_position[body * 3];
            const double dy = stage_position[source * 3 + 1]
                - stage_position[body * 3 + 1];
            const double dz = stage_position[source * 3 + 2]
                - stage_position[body * 3 + 2];
            const double distance_squared = dx * dx + dy * dy + dz * dz;
            if (!(distance_squared > 0.0) || !isfinite(distance_squared)) {
                atomicExch(&lane_status, 1);
                continue;
            }
            const double inverse_distance_cubed =
                1.0 / (distance_squared * sqrt(distance_squared));
            const double displacement = axis == 0 ? dx : (axis == 1 ? dy : dz);
            acceleration += input_gm[gm_offset + source]
                * inverse_distance_cubed * displacement;
        }
        if (!isfinite(acceleration)) atomicExch(&lane_status, 2);
        k_velocity[stage * JX_MAX_COMPONENTS + component] = acceleration;
        __syncthreads();
    }

    double eighth_position = 0.0;
    double eighth_velocity = 0.0;
    double defect_position = 0.0;
    double defect_velocity = 0.0;
    for (int stage = 0; stage < JX_STAGE_COUNT; ++stage) {
        const double eighth = weight_eighth[stage];
        const double defect = weight_defect[stage];
        if (eighth != 0.0) {
            eighth_position += eighth
                * k_position[stage * JX_MAX_COMPONENTS + component];
            eighth_velocity += eighth
                * k_velocity[stage * JX_MAX_COMPONENTS + component];
        }
        if (defect != 0.0) {
            defect_position += defect
                * k_position[stage * JX_MAX_COMPONENTS + component];
            defect_velocity += defect
                * k_velocity[stage * JX_MAX_COMPONENTS + component];
        }
    }

    const double position_delta = step * eighth_position;
    const double velocity_delta = step * eighth_velocity;
    const double adjusted_position = position_delta - position_carry[component];
    const double adjusted_velocity = velocity_delta - velocity_carry[component];
    const double next_position = position[component] + adjusted_position;
    const double next_velocity = velocity[component] + adjusted_velocity;
    const double next_position_carry =
        (next_position - position[component]) - adjusted_position;
    const double next_velocity_carry =
        (next_velocity - velocity[component]) - adjusted_velocity;

    if (!isfinite(next_position) || !isfinite(next_velocity)) {
        atomicExch(&lane_status, 3);
    }
    output_position[lane_component] = next_position;
    output_velocity[lane_component] = next_velocity;
    output_position_carry[lane_component] = next_position_carry;
    output_velocity_carry[lane_component] = next_velocity_carry;
    output_position_defect[lane_component] = step * defect_position;
    output_velocity_defect[lane_component] = step * defect_velocity;
    __syncthreads();
    if (component == 0) output_status[lane] = lane_status;
}
'''


def tableau_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the engine's exact binary64 tableau in dense CUDA layout."""

    coefficients = np.zeros((STAGE_COUNT, STAGE_COUNT), dtype=np.float64)
    for stage, row in enumerate(rkf78._A):
        coefficients[stage, : len(row)] = row
    eighth = np.asarray(rkf78._B_EIGHTH, dtype=np.float64)
    defect = np.asarray(rkf78._B_DEFECT, dtype=np.float64)
    return coefficients, eighth, defect


def _require_state(cp: Any, value: Any, label: str) -> Any:
    if not isinstance(value, cp.ndarray):
        raise FusedPrototypeError(f"{label} must be a CuPy array")
    if value.dtype != cp.float64 or value.ndim != 3 or value.shape[2] != 3:
        raise FusedPrototypeError(
            f"{label} must have shape (lanes, bodies, 3) and float64 dtype"
        )
    return cp.ascontiguousarray(value)


def fused_rkf78_step(
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    step: float,
    *,
    position_carry: Any | None = None,
    velocity_carry: Any | None = None,
) -> FusedStepResult:
    """Advance a batch of 2--32-body systems by one fused RKF78 trial step."""

    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FusedPrototypeError(f"CuPy is unavailable: {exc}") from exc
    checked_step = float(step)
    if not math.isfinite(checked_step) or checked_step == 0.0:
        raise FusedPrototypeError("step must be finite and nonzero")
    positions = _require_state(cp, positions, "positions")
    velocities = _require_state(cp, velocities, "velocities")
    if velocities.shape != positions.shape:
        raise FusedPrototypeError("position and velocity shapes must match")
    lane_count, body_count, _ = positions.shape
    if not 1 <= lane_count or not 2 <= body_count <= MAX_BODIES:
        raise FusedPrototypeError("the prototype requires lanes >= 1 and 2--32 bodies")
    if not isinstance(gravitational_parameters, cp.ndarray):
        raise FusedPrototypeError("gravitational_parameters must be a CuPy array")
    gm = gravitational_parameters
    if gm.dtype != cp.float64:
        raise FusedPrototypeError("gravitational_parameters must use float64")
    if gm.shape == (body_count,):
        gm = cp.broadcast_to(gm[cp.newaxis, :], (lane_count, body_count))
    if gm.shape != (lane_count, body_count):
        raise FusedPrototypeError(
            "gravitational_parameters must have shape (bodies,) or (lanes, bodies)"
        )
    gm = cp.ascontiguousarray(gm)
    if position_carry is None:
        position_carry = cp.zeros_like(positions)
    if velocity_carry is None:
        velocity_carry = cp.zeros_like(velocities)
    position_carry = _require_state(cp, position_carry, "position_carry")
    velocity_carry = _require_state(cp, velocity_carry, "velocity_carry")
    if position_carry.shape != positions.shape or velocity_carry.shape != positions.shape:
        raise FusedPrototypeError("carry shapes must match the state")

    coefficients, eighth, defect = tableau_arrays()
    device_a = cp.asarray(coefficients.reshape(-1), dtype=cp.float64)
    device_eighth = cp.asarray(eighth, dtype=cp.float64)
    device_defect = cp.asarray(defect, dtype=cp.float64)
    outputs = tuple(cp.empty_like(positions) for _ in range(6))
    status = cp.empty(lane_count, dtype=cp.int32)
    kernel = cp.RawKernel(CUDA_SOURCE, "jx_fused_rkf78_step", options=CUDA_OPTIONS)
    kernel(
        (lane_count,),
        (body_count * 3,),
        (
            positions,
            velocities,
            gm,
            position_carry,
            velocity_carry,
            device_a,
            device_eighth,
            device_defect,
            cp.float64(checked_step),
            *outputs,
            status,
            cp.int32(body_count),
            cp.int32(lane_count),
        ),
    )
    return FusedStepResult(
        positions=outputs[0],
        velocities=outputs[1],
        position_carry=outputs[2],
        velocity_carry=outputs[3],
        position_defect=outputs[4],
        velocity_defect=outputs[5],
        status=status,
        body_count=body_count,
        lane_count=lane_count,
    )


def numpy_reference_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    step: float,
    *,
    position_carry: np.ndarray | None = None,
    velocity_carry: np.ndarray | None = None,
) -> FusedStepResult:
    """Evaluate the same one-step contract through the engine's NumPy RKF78."""

    positions = np.ascontiguousarray(positions, dtype=np.float64)
    velocities = np.ascontiguousarray(velocities, dtype=np.float64)
    gm = np.ascontiguousarray(gravitational_parameters, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
        raise FusedPrototypeError("NumPy reference state must have shape (bodies, 3)")
    body_count = positions.shape[0]
    if gm.shape != (body_count,) or not 2 <= body_count <= MAX_BODIES:
        raise FusedPrototypeError("NumPy reference GM shape or body count is invalid")
    if position_carry is None:
        position_carry = np.zeros_like(positions)
    if velocity_carry is None:
        velocity_carry = np.zeros_like(velocities)
    backend = resolve_backend(
        BackendSpec(backend_id="numpy", device="cpu", tile_size=body_count)
    )

    def derivative(_epoch: float, stage_position: np.ndarray, stage_velocity: np.ndarray):
        displacement = stage_position[np.newaxis, :, :] - stage_position[:, np.newaxis, :]
        distance_squared = np.sum(displacement * displacement, axis=2)
        mask = ~np.eye(body_count, dtype=bool)
        if np.any(distance_squared[mask] <= 0.0):
            raise FusedPrototypeError("distinct reference bodies coincide")
        safe = np.where(mask, distance_squared, 1.0)
        inverse_distance_cubed = np.where(mask, 1.0 / (safe * np.sqrt(safe)), 0.0)
        acceleration = np.sum(
            displacement * gm[np.newaxis, :, np.newaxis]
            * inverse_distance_cubed[:, :, np.newaxis],
            axis=1,
        )
        return stage_velocity, acceleration

    trial = rkf78._rkf78_step(
        backend=backend,
        epoch=0.0,
        endpoint_epoch=float(step),
        step=float(step),
        positions=positions,
        velocities=velocities,
        position_carry=np.ascontiguousarray(position_carry, dtype=np.float64),
        velocity_carry=np.ascontiguousarray(velocity_carry, dtype=np.float64),
        derivative=derivative,
    )
    return FusedStepResult(
        positions=trial.accepted_positions[np.newaxis, ...],
        velocities=trial.accepted_velocities[np.newaxis, ...],
        position_carry=trial.accepted_position_carry[np.newaxis, ...],
        velocity_carry=trial.accepted_velocity_carry[np.newaxis, ...],
        position_defect=trial.position_defect[np.newaxis, ...],
        velocity_defect=trial.velocity_defect[np.newaxis, ...],
        status=np.zeros(1, dtype=np.int32),
        body_count=body_count,
        lane_count=1,
        kernel_launch_count=0,
    )


def deterministic_fixture(lanes: int, bodies: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(lanes) is not int or lanes <= 0 or type(bodies) is not int or not 2 <= bodies <= MAX_BODIES:
        raise FusedPrototypeError("fixture requires positive lanes and 2--32 bodies")
    index = np.arange(bodies, dtype=np.float64)
    angle = 2.0 * np.pi * index / bodies
    radius = 1.0 + 0.01 * index
    base_position = np.zeros((bodies, 3), dtype=np.float64)
    base_velocity = np.zeros((bodies, 3), dtype=np.float64)
    base_position[:, 0] = radius * np.cos(angle)
    base_position[:, 1] = radius * np.sin(angle)
    speed = 0.1 / np.sqrt(radius)
    base_velocity[:, 0] = -speed * np.sin(angle)
    base_velocity[:, 1] = speed * np.cos(angle)
    positions = np.repeat(base_position[np.newaxis, ...], lanes, axis=0)
    velocities = np.repeat(base_velocity[np.newaxis, ...], lanes, axis=0)
    for lane in range(lanes):
        phase = 0.0001 * lane
        cosine, sine = math.cos(phase), math.sin(phase)
        x = positions[lane, :, 0].copy()
        y = positions[lane, :, 1].copy()
        positions[lane, :, 0] = cosine * x - sine * y
        positions[lane, :, 1] = sine * x + cosine * y
    gm = np.full((lanes, bodies), 1.0 / bodies, dtype=np.float64)
    return positions, velocities, gm


def run_probe(*, lanes: int, bodies: int, step: float, warmup: int, iterations: int) -> dict[str, Any]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FusedPrototypeError(f"CuPy is unavailable: {exc}") from exc
    host_position, host_velocity, host_gm = deterministic_fixture(lanes, bodies)
    reference = numpy_reference_step(
        host_position[0], host_velocity[0], host_gm[0], step
    )
    position = cp.asarray(host_position)
    velocity = cp.asarray(host_velocity)
    gm = cp.asarray(host_gm)
    result = fused_rkf78_step(position, velocity, gm, step)
    cp.cuda.get_current_stream().synchronize()
    for _ in range(warmup):
        result = fused_rkf78_step(position, velocity, gm, step)
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        result = fused_rkf78_step(position, velocity, gm, step)
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - started
    gpu_position = cp.asnumpy(result.positions[0])
    gpu_velocity = cp.asnumpy(result.velocities[0])
    gpu_position_defect = cp.asnumpy(result.position_defect[0])
    gpu_velocity_defect = cp.asnumpy(result.velocity_defect[0])
    status = cp.asnumpy(result.status)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = properties["name"]
    maximum_difference = max(
        float(np.max(np.abs(gpu_position - reference.positions[0]))),
        float(np.max(np.abs(gpu_velocity - reference.velocities[0]))),
        float(np.max(np.abs(gpu_position_defect - reference.position_defect[0]))),
        float(np.max(np.abs(gpu_velocity_defect - reference.velocity_defect[0]))),
    )
    return {
        "schema": "jxplanetx.fused-rkf78-prototype-probe.v1",
        "status": "PASS" if bool(np.all(status == 0)) and maximum_difference <= 5.0e-13 else "FAIL",
        "scope": "ENGINEERING_PROTOTYPE_NO_PRODUCTION_OR_SCIENTIFIC_QUALIFICATION",
        "workload": {
            "bodies_per_system": bodies,
            "independent_systems": lanes,
            "rkf78_stages_per_step": STAGE_COUNT,
            "step_hex": float(step).hex(),
        },
        "correctness": {
            "all_kernel_status_zero": bool(np.all(status == 0)),
            "lane_zero_maximum_component_difference_vs_numpy_reference_hex": maximum_difference.hex(),
            "comparison_limit_hex": float(5.0e-13).hex(),
        },
        "performance": {
            "warmup_steps": warmup,
            "timed_steps": iterations,
            "kernel_launches_per_batched_step": 1,
            "elapsed_seconds_hex": elapsed.hex(),
            "batched_steps_per_second_hex": (iterations / elapsed).hex(),
            "system_steps_per_second_hex": (iterations * lanes / elapsed).hex(),
        },
        "runtime": {
            "device_name": raw_name.decode("ascii", "replace") if isinstance(raw_name, bytes) else str(raw_name),
            "compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
            "cupy_version": cp.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "claim_controls": {
            "accuracy_superiority_claimed": False,
            "production_authorized": False,
            "scientific_qualification_claimed": False,
            "speed_superiority_claimed": False,
        },
    }


def _write_new(path: Path, encoded: str) -> None:
    raw = encoded.encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--bodies", type=int, default=11)
    parser.add_argument("--step", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    report = run_probe(
        lanes=arguments.lanes,
        bodies=arguments.bodies,
        step=arguments.step,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        _write_new(Path(arguments.output).resolve(), encoded)
    else:
        print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
