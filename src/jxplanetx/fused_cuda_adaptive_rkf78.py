#!/usr/bin/env python3
"""Package-owned adaptive batched CUDA core built on fused JX RKF78.

One block owns one independent 2--32-body mutual Newtonian system. Each launch
runs a persistent bounded controller through complete 13-stage trials and the
entire checkpoint schedule until its lane reaches the final checkpoint, records
a fail-closed status, or hits the configurable safety chunk limit. Host code
observes terminal state only between chunks, never after attempted steps or
intermediate checkpoints.

This remains an experimental screening component. It has one Newtonian force model, no
event location or response, and no production-backend registration. A separate
restricted bridge binds its componentwise tolerances, provenance, checkpoints,
force ledger, and accepted-step ledger to the public trajectory result types.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from . import fused_cuda_rkf78 as fused


CUDA_OPTIONS = fused.CUDA_OPTIONS
MAX_BODIES = fused.MAX_BODIES
STAGE_COUNT = fused.STAGE_COUNT
COUNT_HEADER_ROWS = 6

STATUS_MESSAGES = {
    0: "OK",
    1: "SINGULAR_OR_INVALID_DISTANCE",
    2: "NONFINITE_ACCELERATION",
    3: "NONFINITE_CANDIDATE",
    4: "MAXIMUM_STEPS_EXHAUSTED",
    5: "NONFINITE_NORMALIZED_ERROR",
    6: "MAXIMUM_REJECTIONS_EXHAUSTED",
    7: "REJECTED_AT_MINIMUM_STEP",
    8: "CONTROLLER_FAILED_TO_REDUCE",
    9: "INVALID_EPOCH_PROGRESS",
    10: "FINITE_RADIUS_COLLISION",
    11: "DEVICE_LEDGER_CAPACITY_EXHAUSTED",
}

_DEVICE_ASSET_CACHE: dict[int, tuple[Any, Any, Any, Any]] = {}


class AdaptivePrototypeError(RuntimeError):
    """The adaptive fused prototype contract is invalid."""


@dataclass(frozen=True)
class AdaptiveFusedSpec:
    initial_epoch: float
    final_epoch: float
    initial_step: float
    minimum_step: float
    maximum_step: float
    position_atol: float
    position_rtol: float
    velocity_atol: float
    velocity_rtol: float
    maximum_steps: int = 100_000
    maximum_rejections: int = 10_000
    safety_factor: float = 0.9
    minimum_scale_factor: float = 0.2
    maximum_scale_factor: float = 5.0
    intermediate_epochs: tuple[float, ...] = ()
    ledger_capacity: int = 4_096
    attempts_per_launch: int = 4_096

    def __post_init__(self) -> None:
        finite = (
            self.initial_epoch,
            self.final_epoch,
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.position_atol,
            self.position_rtol,
            self.velocity_atol,
            self.velocity_rtol,
            self.safety_factor,
            self.minimum_scale_factor,
            self.maximum_scale_factor,
        )
        if any(not math.isfinite(float(value)) for value in finite):
            raise AdaptivePrototypeError("adaptive scalar fields must be finite")
        if self.final_epoch == self.initial_epoch:
            raise AdaptivePrototypeError("initial and final epochs must differ")
        if type(self.intermediate_epochs) is not tuple:
            raise AdaptivePrototypeError("intermediate_epochs must be an exact tuple")
        checkpoint_epochs = self.checkpoint_epochs
        if any(not math.isfinite(value) for value in checkpoint_epochs):
            raise AdaptivePrototypeError("checkpoint epochs must be finite")
        differences = tuple(
            right - left
            for left, right in zip(checkpoint_epochs, checkpoint_epochs[1:])
        )
        if not (
            all(value > 0.0 for value in differences)
            or all(value < 0.0 for value in differences)
        ):
            raise AdaptivePrototypeError(
                "checkpoint epochs must be strictly monotone in one direction"
            )
        if not 0.0 < self.minimum_step <= self.initial_step <= self.maximum_step:
            raise AdaptivePrototypeError("step bounds must satisfy 0 < min <= initial <= max")
        if min(
            self.position_atol,
            self.position_rtol,
            self.velocity_atol,
            self.velocity_rtol,
        ) <= 0.0:
            raise AdaptivePrototypeError("all tolerances must be positive")
        if not 0.0 < self.safety_factor < 1.0:
            raise AdaptivePrototypeError("safety_factor must be between zero and one")
        if not 0.0 < self.minimum_scale_factor <= 1.0 <= self.maximum_scale_factor:
            raise AdaptivePrototypeError("controller scale bounds must straddle one")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise AdaptivePrototypeError("maximum_steps must be a positive integer")
        if type(self.maximum_rejections) is not int or self.maximum_rejections < 0:
            raise AdaptivePrototypeError("maximum_rejections must be nonnegative")
        if type(self.ledger_capacity) is not int or self.ledger_capacity <= 0:
            raise AdaptivePrototypeError("ledger_capacity must be a positive integer")
        if type(self.attempts_per_launch) is not int or self.attempts_per_launch <= 0:
            raise AdaptivePrototypeError(
                "attempts_per_launch must be a positive integer"
            )

    @property
    def checkpoint_epochs(self) -> tuple[float, ...]:
        return (self.initial_epoch, *self.intermediate_epochs, self.final_epoch)

    @property
    def direction(self) -> float:
        return 1.0 if self.final_epoch > self.initial_epoch else -1.0


@dataclass(frozen=True)
class AdaptiveFusedResult:
    positions: Any
    velocities: Any
    position_carry: Any
    velocity_carry: Any
    epochs: Any
    proposed_steps: Any
    attempted_steps: Any
    accepted_steps: Any
    rejected_steps: Any
    maximum_normalized_error: Any
    done: Any
    status: Any
    body_count: int
    lane_count: int
    kernel_launch_count: int
    checkpoint_positions: tuple[Any, ...]
    checkpoint_velocities: tuple[Any, ...]
    checkpoint_accepted_steps: tuple[np.ndarray, ...]
    checkpoint_rejected_steps: tuple[np.ndarray, ...]
    accepted_step_epochs: tuple[tuple[float, ...], ...]
    accepted_step_magnitudes: tuple[tuple[float, ...], ...]


@dataclass
class AdaptiveFusedWorkspace:
    """Reusable borrowed device storage for one fixed integration shape.

    Results produced with a workspace reference its buffers and remain valid
    only until that workspace is reused. The public trajectory bridge does not
    use this optimization and continues to return independent owning results.
    """

    device_id: int
    lane_count: int
    body_count: int
    checkpoint_epochs: tuple[float, ...]
    ledger_capacity: int
    buffers: dict[str, Any]
    position_atol_default: float
    velocity_atol_default: float
    radii_are_default: bool = True
    position_atol_is_default: bool = True
    velocity_atol_is_default: bool = True


CUDA_SOURCE = r'''
#define JX_STAGE_COUNT 13
#define JX_MAX_BODIES 32
#define JX_MAX_COMPONENTS 96

extern "C" __global__
void jx_fused_adaptive_rkf78_schedule(
    double* position_global,
    double* velocity_global,
    const double* initial_position_global,
    const double* initial_velocity_global,
    const double* gm_global,
    const double* radius_global,
    double* position_carry_global,
    double* velocity_carry_global,
    const double* tableau_a,
    const double* weight_eighth,
    const double* weight_defect,
    double* epoch_global,
    const double initial_epoch,
    const double* checkpoint_epoch_global,
    const int checkpoint_count,
    long long* checkpoint_index_global,
    double* checkpoint_position_global,
    double* checkpoint_velocity_global,
    long long* checkpoint_accepted_steps_global,
    long long* checkpoint_rejected_steps_global,
    const double direction,
    double* proposed_step_global,
    const double initial_step,
    const double* position_atol_global,
    const double position_rtol,
    const double* velocity_atol_global,
    const double velocity_rtol,
    const double minimum_step,
    const double maximum_step,
    const double safety_factor,
    const double minimum_scale_factor,
    const double maximum_scale_factor,
    double* accepted_epoch_ledger,
    double* accepted_magnitude_ledger,
    const long long ledger_capacity,
    const long long attempts_per_launch,
    long long* attempted_steps,
    long long* accepted_steps,
    long long* rejected_steps,
    double* maximum_normalized_error,
    long long* done,
    long long* status,
    const long long maximum_steps,
    const long long maximum_rejections,
    const int body_count,
    const int lane_count,
    const int initialize_state)
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
    __shared__ double candidate_position[JX_MAX_COMPONENTS];
    __shared__ double candidate_velocity[JX_MAX_COMPONENTS];
    __shared__ double candidate_position_carry[JX_MAX_COMPONENTS];
    __shared__ double candidate_velocity_carry[JX_MAX_COMPONENTS];
    __shared__ double component_error[JX_MAX_COMPONENTS];
    extern __shared__ double single_lane_pair_weight[];
    __shared__ double step;
    __shared__ double proposal;
    __shared__ double endpoint_epoch;
    __shared__ double checkpoint_epoch;
    __shared__ int checkpoint_clipped;
    __shared__ int target_checkpoint_index;
    __shared__ int reached_checkpoint;
    __shared__ int execute_trial;
    __shared__ int accept_trial;
    __shared__ int lane_status;
    __shared__ long long launch_start_attempts;

    const int lane_component = lane * component_count + component;
    const int body = component / 3;
    const int axis = component - body * 3;
    const int gm_offset = lane * body_count;

    if (initialize_state != 0) {
        position_global[lane_component] = initial_position_global[lane_component];
        velocity_global[lane_component] = initial_velocity_global[lane_component];
        position_carry_global[lane_component] = 0.0;
        velocity_carry_global[lane_component] = 0.0;
        checkpoint_position_global[lane_component] =
            initial_position_global[lane_component];
        checkpoint_velocity_global[lane_component] =
            initial_velocity_global[lane_component];
        for (int index = component; index < checkpoint_count;
             index += component_count) {
            checkpoint_accepted_steps_global[index * lane_count + lane] = 0;
            checkpoint_rejected_steps_global[index * lane_count + lane] = 0;
        }
        if (component == 0) {
            epoch_global[lane] = initial_epoch;
            proposed_step_global[lane] = initial_step;
            attempted_steps[lane] = 0;
            accepted_steps[lane] = 0;
            rejected_steps[lane] = 0;
            checkpoint_index_global[lane] = 1;
            maximum_normalized_error[lane] = 0.0;
            done[lane] = 0;
            status[lane] = 0;
        }
    }
    __syncthreads();
    if (done[lane] != 0 || status[lane] != 0) return;

    if (component == 0) {
        launch_start_attempts = attempted_steps[lane];
    }
    __syncthreads();

    while (true) {

    if (component == 0) {
        lane_status = 0;
        accept_trial = 0;
        reached_checkpoint = 0;
        execute_trial = 1;
        if (done[lane] != 0 || status[lane] != 0) {
            execute_trial = 0;
        } else if (checkpoint_index_global[lane] >= checkpoint_count) {
            done[lane] = 1;
            execute_trial = 0;
        } else if (attempted_steps[lane] >= maximum_steps) {
            status[lane] = 4;
            execute_trial = 0;
        } else if (
            attempted_steps[lane] - launch_start_attempts >= attempts_per_launch
        ) {
            execute_trial = 0;
        } else {
            target_checkpoint_index = (int)checkpoint_index_global[lane];
            checkpoint_epoch = checkpoint_epoch_global[target_checkpoint_index];
            const double current_epoch = epoch_global[lane];
            const double remaining = checkpoint_epoch - current_epoch;
            if (!(direction * remaining > 0.0) || !isfinite(remaining)) {
                status[lane] = 9;
                execute_trial = 0;
            } else {
                proposal = fmin(maximum_step, fmax(minimum_step, proposed_step_global[lane]));
                checkpoint_clipped = proposal >= fabs(remaining) ? 1 : 0;
                if (checkpoint_clipped != 0) {
                    endpoint_epoch = checkpoint_epoch;
                    step = remaining;
                } else {
                    endpoint_epoch = current_epoch + copysign(proposal, direction);
                    step = endpoint_epoch - current_epoch;
                    if (fabs(step) > proposal) {
                        endpoint_epoch = nextafter(endpoint_epoch, current_epoch);
                        step = endpoint_epoch - current_epoch;
                    }
                }
                if (!(direction * step > 0.0) || !isfinite(step)
                    || fabs(step) > proposal
                    || direction * (checkpoint_epoch - endpoint_epoch) < 0.0
                    || (checkpoint_clipped == 0
                        && direction * (checkpoint_epoch - endpoint_epoch) <= 0.0)) {
                    status[lane] = 9;
                    execute_trial = 0;
                }
            }
        }
    }
    __syncthreads();
    if (execute_trial == 0) return;

    position[component] = position_global[lane_component];
    velocity[component] = velocity_global[lane_component];
    position_carry[component] = position_carry_global[lane_component];
    velocity_carry[component] = velocity_carry_global[lane_component];
    __syncthreads();

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
        if (lane_count == 1) {
            const int pair_count = body_count * body_count;
            for (int pair = component; pair < pair_count; pair += component_count) {
                const int pair_body = pair / body_count;
                const int pair_source = pair - pair_body * body_count;
                if (pair_source <= pair_body) continue;
                const double body_gm = gm_global[pair_body];
                const double source_gm = gm_global[pair_source];
                const double dx = stage_position[pair_source * 3]
                    - stage_position[pair_body * 3];
                const double dy = stage_position[pair_source * 3 + 1]
                    - stage_position[pair_body * 3 + 1];
                const double dz = stage_position[pair_source * 3 + 2]
                    - stage_position[pair_body * 3 + 2];
                const double distance_squared = dx * dx + dy * dy + dz * dz;
                if (!(body_gm > 0.0) || !isfinite(body_gm)
                    || !(source_gm > 0.0) || !isfinite(source_gm)
                    || !(distance_squared > 0.0) || !isfinite(distance_squared)) {
                    atomicExch(&lane_status, 1);
                    single_lane_pair_weight[pair] = 0.0;
                    single_lane_pair_weight[pair_source * body_count + pair_body] =
                        0.0;
                    continue;
                }
                const double collision_distance = radius_global[pair_body]
                    + radius_global[pair_source];
                if (distance_squared <= collision_distance * collision_distance) {
                    atomicExch(&lane_status, 10);
                    single_lane_pair_weight[pair] = 0.0;
                    single_lane_pair_weight[pair_source * body_count + pair_body] =
                        0.0;
                    continue;
                }
                const double inverse_distance_cubed =
                    1.0 / (distance_squared * sqrt(distance_squared));
                single_lane_pair_weight[pair] = inverse_distance_cubed;
                single_lane_pair_weight[pair_source * body_count + pair_body] =
                    inverse_distance_cubed;
            }
            __syncthreads();
        }
        double acceleration = 0.0;
        for (int source = 0; source < body_count; ++source) {
            if (source == body) continue;
            const double dx = stage_position[source * 3]
                - stage_position[body * 3];
            const double dy = stage_position[source * 3 + 1]
                - stage_position[body * 3 + 1];
            const double dz = stage_position[source * 3 + 2]
                - stage_position[body * 3 + 2];
            const double displacement =
                axis == 0 ? dx : (axis == 1 ? dy : dz);
            if (lane_count == 1) {
                acceleration += gm_global[source]
                    * single_lane_pair_weight[body * body_count + source]
                    * displacement;
            } else {
                const double source_gm = gm_global[gm_offset + source];
                const double distance_squared = dx * dx + dy * dy + dz * dz;
                if (!(source_gm > 0.0) || !isfinite(source_gm)
                    || !(distance_squared > 0.0) || !isfinite(distance_squared)) {
                    atomicExch(&lane_status, 1);
                    continue;
                }
                const double collision_distance =
                    radius_global[gm_offset + body] + radius_global[gm_offset + source];
                if (distance_squared <= collision_distance * collision_distance) {
                    atomicExch(&lane_status, 10);
                    continue;
                }
                const double inverse_distance_cubed =
                    1.0 / (distance_squared * sqrt(distance_squared));
                acceleration += source_gm * inverse_distance_cubed * displacement;
            }
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
    candidate_position[component] = next_position;
    candidate_velocity[component] = next_velocity;
    candidate_position_carry[component] =
        (next_position - position[component]) - adjusted_position;
    candidate_velocity_carry[component] =
        (next_velocity - velocity[component]) - adjusted_velocity;
    if (!isfinite(next_position) || !isfinite(next_velocity)) {
        atomicExch(&lane_status, 3);
    }
    const double position_scale = position_atol_global[lane_component]
        + position_rtol * fmax(fabs(position[component]), fabs(next_position));
    const double velocity_scale = velocity_atol_global[lane_component]
        + velocity_rtol * fmax(fabs(velocity[component]), fabs(next_velocity));
    const double position_error = fabs(step * defect_position) / position_scale;
    const double velocity_error = fabs(step * defect_velocity) / velocity_scale;
    component_error[component] = fmax(position_error, velocity_error);
    __syncthreads();

    if (component == 0) {
        attempted_steps[lane] += 1;
        if (lane_status != 0) {
            status[lane] = lane_status;
        } else {
            double normalized_error = 0.0;
            for (int index = 0; index < component_count; ++index) {
                normalized_error = fmax(normalized_error, component_error[index]);
            }
            if (!isfinite(normalized_error) || normalized_error < 0.0) {
                status[lane] = 5;
            } else {
                maximum_normalized_error[lane] =
                    fmax(maximum_normalized_error[lane], normalized_error);
                const double factor = normalized_error == 0.0
                    ? maximum_scale_factor
                    : fmin(maximum_scale_factor,
                        fmax(minimum_scale_factor,
                            safety_factor * pow(normalized_error, -0.125)));
                if (normalized_error <= 1.0) {
                    const long long ledger_index = accepted_steps[lane];
                    if (ledger_index >= ledger_capacity) {
                        status[lane] = 11;
                    } else {
                        const long long ledger_offset =
                            ((long long)lane) * ledger_capacity + ledger_index;
                        accepted_epoch_ledger[ledger_offset] = endpoint_epoch;
                        accepted_magnitude_ledger[ledger_offset] = fabs(step);
                        accept_trial = 1;
                        accepted_steps[lane] += 1;
                        epoch_global[lane] = endpoint_epoch;
                        if (checkpoint_clipped != 0
                            || endpoint_epoch == checkpoint_epoch) {
                            reached_checkpoint = 1;
                            const long long checkpoint_offset =
                                ((long long)target_checkpoint_index) * lane_count
                                + lane;
                            checkpoint_accepted_steps_global[checkpoint_offset] =
                                accepted_steps[lane];
                            checkpoint_rejected_steps_global[checkpoint_offset] =
                                rejected_steps[lane];
                            checkpoint_index_global[lane] =
                                target_checkpoint_index + 1;
                            if (target_checkpoint_index + 1 >= checkpoint_count) {
                                done[lane] = 1;
                            }
                        } else {
                            proposed_step_global[lane] = fmin(
                                maximum_step,
                                fmax(minimum_step, fabs(step) * factor));
                        }
                    }
                } else {
                    rejected_steps[lane] += 1;
                    if (rejected_steps[lane] > maximum_rejections) {
                        status[lane] = 6;
                    } else if (proposal <= minimum_step
                        || (checkpoint_clipped != 0 && fabs(step) < minimum_step)) {
                        status[lane] = 7;
                    } else {
                        const double reduced = fmax(
                            minimum_step,
                            fabs(step) * factor);
                        if (!(reduced < fabs(step))) {
                            status[lane] = 8;
                        } else {
                            proposed_step_global[lane] = reduced;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (accept_trial != 0 && status[lane] == 0) {
        position_global[lane_component] = candidate_position[component];
        velocity_global[lane_component] = candidate_velocity[component];
        position_carry_global[lane_component] = candidate_position_carry[component];
        velocity_carry_global[lane_component] = candidate_velocity_carry[component];
        if (reached_checkpoint != 0) {
            const long long checkpoint_component_offset =
                (((long long)target_checkpoint_index) * lane_count + lane)
                * component_count + component;
            checkpoint_position_global[checkpoint_component_offset] =
                candidate_position[component];
            checkpoint_velocity_global[checkpoint_component_offset] =
                candidate_velocity[component];
        }
    }
    __syncthreads();
    }
}
'''


def _require_state(cp: Any, value: Any, label: str) -> Any:
    if not isinstance(value, cp.ndarray):
        raise AdaptivePrototypeError(f"{label} must be a CuPy array")
    if value.dtype != cp.float64 or value.ndim != 3 or value.shape[2] != 3:
        raise AdaptivePrototypeError(
            f"{label} must have shape (lanes, bodies, 3) and float64 dtype"
        )
    return cp.ascontiguousarray(value)


def _require_tolerance(
    cp: Any,
    value: Any | None,
    scalar_default: float,
    state_shape: tuple[int, int, int],
    label: str,
) -> Any:
    if value is None:
        return cp.full(state_shape, scalar_default, dtype=cp.float64)
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float64:
        raise AdaptivePrototypeError(f"{label} must be a CuPy float64 array")
    lane_count, body_count, _ = state_shape
    if value.shape == (body_count, 3):
        value = cp.broadcast_to(value[cp.newaxis, :, :], state_shape)
    if value.shape != state_shape:
        raise AdaptivePrototypeError(
            f"{label} must have shape (bodies, 3) or (lanes, bodies, 3)"
        )
    result = cp.ascontiguousarray(value)
    if bool(cp.asnumpy(cp.any((~cp.isfinite(result)) | (result <= 0.0)))):
        raise AdaptivePrototypeError(f"{label} must be finite and positive")
    return result


def _require_radii(
    cp: Any,
    value: Any | None,
    lane_count: int,
    body_count: int,
) -> Any:
    shape = (lane_count, body_count)
    if value is None:
        return cp.zeros(shape, dtype=cp.float64)
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float64:
        raise AdaptivePrototypeError("radii must be a CuPy float64 array")
    if value.shape == (body_count,):
        value = cp.broadcast_to(value[cp.newaxis, :], shape)
    if value.shape != shape:
        raise AdaptivePrototypeError(
            "radii must have shape (bodies,) or (lanes, bodies)"
        )
    result = cp.ascontiguousarray(value)
    if bool(cp.asnumpy(cp.any((~cp.isfinite(result)) | (result < 0.0)))):
        raise AdaptivePrototypeError("radii must be finite and nonnegative")
    return result


def _device_assets(cp: Any) -> tuple[Any, Any, Any, Any]:
    """Return immutable tableau arrays and the compiled kernel for this GPU."""

    device_id = int(cp.cuda.runtime.getDevice())
    cached = _DEVICE_ASSET_CACHE.get(device_id)
    if cached is None:
        coefficients, eighth, defect = fused.tableau_arrays()
        cached = (
            cp.asarray(coefficients.reshape(-1), dtype=cp.float64),
            cp.asarray(eighth, dtype=cp.float64),
            cp.asarray(defect, dtype=cp.float64),
            cp.RawKernel(
                CUDA_SOURCE,
                "jx_fused_adaptive_rkf78_schedule",
                options=CUDA_OPTIONS,
            ),
        )
        _DEVICE_ASSET_CACHE[device_id] = cached
    return cached


def prepare_adaptive_workspace(
    lane_count: int,
    body_count: int,
    spec: AdaptiveFusedSpec,
) -> AdaptiveFusedWorkspace:
    """Allocate reusable device storage outside a latency-critical call."""

    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AdaptivePrototypeError(f"CuPy unavailable: {exc}") from exc
    if type(spec) is not AdaptiveFusedSpec:
        raise AdaptivePrototypeError("spec must be an exact AdaptiveFusedSpec")
    if type(lane_count) is not int or lane_count <= 0:
        raise AdaptivePrototypeError("workspace lane_count must be positive")
    if type(body_count) is not int or not 2 <= body_count <= MAX_BODIES:
        raise AdaptivePrototypeError("workspace requires 2--32 bodies")
    state_shape = (lane_count, body_count, 3)
    checkpoint_count = len(spec.checkpoint_epochs)
    device_counts = cp.empty(
        (COUNT_HEADER_ROWS + 2 * checkpoint_count, lane_count), dtype=cp.int64
    )
    device_ledger = cp.empty(
        (2, lane_count, spec.ledger_capacity), dtype=cp.float64
    )
    buffers = {
        "positions": cp.empty(state_shape, dtype=cp.float64),
        "velocities": cp.empty(state_shape, dtype=cp.float64),
        "gm": cp.empty((lane_count, body_count), dtype=cp.float64),
        "radii": cp.zeros((lane_count, body_count), dtype=cp.float64),
        "position_atol": cp.full(
            state_shape, spec.position_atol, dtype=cp.float64
        ),
        "velocity_atol": cp.full(
            state_shape, spec.velocity_atol, dtype=cp.float64
        ),
        "position_carry": cp.empty(state_shape, dtype=cp.float64),
        "velocity_carry": cp.empty(state_shape, dtype=cp.float64),
        "epochs": cp.empty(lane_count, dtype=cp.float64),
        "proposed_steps": cp.empty(lane_count, dtype=cp.float64),
        "counts": device_counts,
        "attempted_steps": device_counts[0],
        "accepted_steps": device_counts[1],
        "rejected_steps": device_counts[2],
        "done": device_counts[4],
        "status": device_counts[5],
        "maximum_error": cp.empty(lane_count, dtype=cp.float64),
        "ledger": device_ledger,
        "accepted_epochs": device_ledger[0],
        "accepted_magnitudes": device_ledger[1],
        "checkpoint_epochs": cp.asarray(
            spec.checkpoint_epochs, dtype=cp.float64
        ),
        "checkpoint_indices": device_counts[3],
        "checkpoint_positions": cp.empty(
            (checkpoint_count, lane_count, body_count, 3), dtype=cp.float64
        ),
        "checkpoint_velocities": cp.empty(
            (checkpoint_count, lane_count, body_count, 3), dtype=cp.float64
        ),
        "checkpoint_accepted_steps": device_counts[
            COUNT_HEADER_ROWS : COUNT_HEADER_ROWS + checkpoint_count
        ],
        "checkpoint_rejected_steps": device_counts[
            COUNT_HEADER_ROWS + checkpoint_count :
            COUNT_HEADER_ROWS + 2 * checkpoint_count
        ],
    }
    return AdaptiveFusedWorkspace(
        device_id=int(cp.cuda.runtime.getDevice()),
        lane_count=lane_count,
        body_count=body_count,
        checkpoint_epochs=spec.checkpoint_epochs,
        ledger_capacity=spec.ledger_capacity,
        buffers=buffers,
        position_atol_default=spec.position_atol,
        velocity_atol_default=spec.velocity_atol,
    )


def fused_adaptive_integrate(
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    spec: AdaptiveFusedSpec,
    *,
    position_atol: Any | None = None,
    velocity_atol: Any | None = None,
    radii: Any | None = None,
    workspace: AdaptiveFusedWorkspace | None = None,
    phase_timings: dict[str, float] | None = None,
) -> AdaptiveFusedResult:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AdaptivePrototypeError(f"CuPy unavailable: {exc}") from exc
    if phase_timings is not None and type(phase_timings) is not dict:
        raise AdaptivePrototypeError("phase_timings must be a dictionary")
    if phase_timings is not None:
        phase_timings.clear()
    profile_started = time.perf_counter()
    if type(spec) is not AdaptiveFusedSpec:
        raise AdaptivePrototypeError("spec must be an exact AdaptiveFusedSpec")
    input_positions = _require_state(cp, positions, "positions")
    input_velocities = _require_state(cp, velocities, "velocities")
    if input_positions.shape != input_velocities.shape:
        raise AdaptivePrototypeError("position and velocity shapes must match")
    lane_count, body_count, _ = input_positions.shape
    if lane_count < 1 or not 2 <= body_count <= MAX_BODIES:
        raise AdaptivePrototypeError("prototype requires lanes >= 1 and 2--32 bodies")
    if workspace is not None:
        if type(workspace) is not AdaptiveFusedWorkspace:
            raise AdaptivePrototypeError(
                "workspace must be an exact AdaptiveFusedWorkspace"
            )
        if workspace.device_id != int(cp.cuda.runtime.getDevice()):
            raise AdaptivePrototypeError("workspace belongs to another CUDA device")
        if (workspace.lane_count, workspace.body_count) != (
            lane_count,
            body_count,
        ):
            raise AdaptivePrototypeError("workspace state shape does not match")
        if workspace.checkpoint_epochs != spec.checkpoint_epochs:
            raise AdaptivePrototypeError("workspace checkpoint schedule does not match")
        if workspace.ledger_capacity != spec.ledger_capacity:
            raise AdaptivePrototypeError("workspace ledger capacity does not match")
        buffers = workspace.buffers
        positions = buffers["positions"]
        velocities = buffers["velocities"]
    else:
        buffers = None
        positions = input_positions.copy()
        velocities = input_velocities.copy()
    if (
        not isinstance(gravitational_parameters, cp.ndarray)
        or gravitational_parameters.dtype != cp.float64
    ):
        raise AdaptivePrototypeError("gravitational_parameters must be a CuPy float64 array")
    gm = gravitational_parameters
    if gm.shape == (body_count,):
        gm = cp.broadcast_to(gm[cp.newaxis, :], (lane_count, body_count))
    if gm.shape != (lane_count, body_count):
        raise AdaptivePrototypeError("GM shape must be (bodies,) or (lanes, bodies)")
    gm = cp.ascontiguousarray(gm)
    if buffers is None:
        radius_array = _require_radii(cp, radii, lane_count, body_count)
        position_atol_array = _require_tolerance(
            cp,
            position_atol,
            spec.position_atol,
            positions.shape,
            "position_atol",
        )
        velocity_atol_array = _require_tolerance(
            cp,
            velocity_atol,
            spec.velocity_atol,
            positions.shape,
            "velocity_atol",
        )
    else:
        radius_array = buffers["radii"]
        if radii is None:
            if not workspace.radii_are_default:
                radius_array.fill(0.0)
                workspace.radii_are_default = True
        else:
            cp.copyto(
                radius_array,
                _require_radii(cp, radii, lane_count, body_count),
            )
            workspace.radii_are_default = False
        position_atol_array = buffers["position_atol"]
        if position_atol is None:
            if (
                not workspace.position_atol_is_default
                or workspace.position_atol_default != spec.position_atol
            ):
                position_atol_array.fill(spec.position_atol)
                workspace.position_atol_is_default = True
                workspace.position_atol_default = spec.position_atol
        else:
            cp.copyto(
                position_atol_array,
                _require_tolerance(
                    cp,
                    position_atol,
                    spec.position_atol,
                    positions.shape,
                    "position_atol",
                ),
            )
            workspace.position_atol_is_default = False
        velocity_atol_array = buffers["velocity_atol"]
        if velocity_atol is None:
            if (
                not workspace.velocity_atol_is_default
                or workspace.velocity_atol_default != spec.velocity_atol
            ):
                velocity_atol_array.fill(spec.velocity_atol)
                workspace.velocity_atol_is_default = True
                workspace.velocity_atol_default = spec.velocity_atol
        else:
            cp.copyto(
                velocity_atol_array,
                _require_tolerance(
                    cp,
                    velocity_atol,
                    spec.velocity_atol,
                    positions.shape,
                    "velocity_atol",
                ),
            )
            workspace.velocity_atol_is_default = False
    if phase_timings is not None:
        cp.cuda.get_current_stream().synchronize()
        input_finished = time.perf_counter()
        phase_timings["validation_and_input_seconds"] = (
            input_finished - profile_started
        )
    else:
        input_finished = 0.0
    device_a, device_eighth, device_defect, kernel = _device_assets(cp)
    checkpoint_count = len(spec.checkpoint_epochs)
    if buffers is None:
        position_carry = cp.zeros_like(positions)
        velocity_carry = cp.zeros_like(velocities)
        epochs = cp.full(lane_count, spec.initial_epoch, dtype=cp.float64)
        proposed_steps = cp.full(lane_count, spec.initial_step, dtype=cp.float64)
        device_counts = cp.zeros(
            (COUNT_HEADER_ROWS + 2 * checkpoint_count, lane_count),
            dtype=cp.int64,
        )
        attempted_steps = device_counts[0]
        accepted_steps = device_counts[1]
        rejected_steps = device_counts[2]
        done = device_counts[4]
        status = device_counts[5]
        maximum_error = cp.zeros(lane_count, dtype=cp.float64)
        device_ledger = cp.empty(
            (2, lane_count, spec.ledger_capacity), dtype=cp.float64
        )
        device_accepted_epochs = device_ledger[0]
        device_accepted_magnitudes = device_ledger[1]
        device_checkpoint_epochs = cp.asarray(
            spec.checkpoint_epochs, dtype=cp.float64
        )
        checkpoint_indices = device_counts[3]
        checkpoint_indices.fill(1)
        device_checkpoint_positions = cp.empty(
            (checkpoint_count, lane_count, body_count, 3), dtype=cp.float64
        )
        device_checkpoint_velocities = cp.empty_like(
            device_checkpoint_positions
        )
        device_checkpoint_accepted_steps = device_counts[
            COUNT_HEADER_ROWS : COUNT_HEADER_ROWS + checkpoint_count
        ]
        device_checkpoint_rejected_steps = device_counts[
            COUNT_HEADER_ROWS + checkpoint_count :
            COUNT_HEADER_ROWS + 2 * checkpoint_count
        ]
    else:
        position_carry = buffers["position_carry"]
        velocity_carry = buffers["velocity_carry"]
        epochs = buffers["epochs"]
        proposed_steps = buffers["proposed_steps"]
        device_counts = buffers["counts"]
        attempted_steps = buffers["attempted_steps"]
        accepted_steps = buffers["accepted_steps"]
        rejected_steps = buffers["rejected_steps"]
        maximum_error = buffers["maximum_error"]
        done = buffers["done"]
        status = buffers["status"]
        device_ledger = buffers["ledger"]
        device_accepted_epochs = buffers["accepted_epochs"]
        device_accepted_magnitudes = buffers["accepted_magnitudes"]
        device_checkpoint_epochs = buffers["checkpoint_epochs"]
        checkpoint_indices = buffers["checkpoint_indices"]
        device_checkpoint_positions = buffers["checkpoint_positions"]
        device_checkpoint_velocities = buffers["checkpoint_velocities"]
        device_checkpoint_accepted_steps = buffers[
            "checkpoint_accepted_steps"
        ]
        device_checkpoint_rejected_steps = buffers[
            "checkpoint_rejected_steps"
        ]
    if buffers is None:
        device_checkpoint_positions[0] = positions
        device_checkpoint_velocities[0] = velocities
    if phase_timings is not None:
        cp.cuda.get_current_stream().synchronize()
        reset_finished = time.perf_counter()
        phase_timings["workspace_reset_seconds"] = (
            reset_finished - input_finished
        )
    else:
        reset_finished = 0.0
    launch_count = 0
    kernel_seconds = 0.0
    metadata_transfer_seconds = 0.0
    chunks_per_lane = math.ceil(spec.maximum_steps / spec.attempts_per_launch)
    host_launch_ceiling = chunks_per_lane + 1
    while True:
        kernel_started = time.perf_counter()
        kernel(
            (lane_count,),
            (body_count * 3,),
            (
                positions,
                velocities,
                input_positions,
                input_velocities,
                gm,
                radius_array,
                position_carry,
                velocity_carry,
                device_a,
                device_eighth,
                device_defect,
                epochs,
                cp.float64(spec.initial_epoch),
                device_checkpoint_epochs,
                cp.int32(checkpoint_count),
                checkpoint_indices,
                device_checkpoint_positions,
                device_checkpoint_velocities,
                device_checkpoint_accepted_steps,
                device_checkpoint_rejected_steps,
                cp.float64(spec.direction),
                proposed_steps,
                cp.float64(spec.initial_step),
                position_atol_array,
                cp.float64(spec.position_rtol),
                velocity_atol_array,
                cp.float64(spec.velocity_rtol),
                cp.float64(spec.minimum_step),
                cp.float64(spec.maximum_step),
                cp.float64(spec.safety_factor),
                cp.float64(spec.minimum_scale_factor),
                cp.float64(spec.maximum_scale_factor),
                device_accepted_epochs,
                device_accepted_magnitudes,
                cp.int64(spec.ledger_capacity),
                cp.int64(spec.attempts_per_launch),
                attempted_steps,
                accepted_steps,
                rejected_steps,
                maximum_error,
                done,
                status,
                cp.int64(spec.maximum_steps),
                cp.int64(spec.maximum_rejections),
                cp.int32(body_count),
                cp.int32(lane_count),
                cp.int32(1 if buffers is not None and launch_count == 0 else 0),
            ),
            shared_mem=(MAX_BODIES * MAX_BODIES * 8 if lane_count == 1 else 0),
        )
        launch_count += 1
        if phase_timings is not None:
            cp.cuda.get_current_stream().synchronize()
            kernel_finished = time.perf_counter()
            kernel_seconds += kernel_finished - kernel_started
            metadata_started = kernel_finished
        host_counts = cp.asnumpy(device_counts)
        if phase_timings is not None:
            metadata_finished = time.perf_counter()
            metadata_transfer_seconds += metadata_finished - metadata_started
        terminal = (host_counts[4] != 0) | (host_counts[5] != 0)
        if bool(np.all(terminal)):
            break
        if launch_count > host_launch_ceiling:
            raise AdaptivePrototypeError("host launch ceiling exhausted")
    host_checkpoint_indices = host_counts[3]
    retained_checkpoint_count = int(np.min(host_checkpoint_indices))
    checkpoint_positions = tuple(
        device_checkpoint_positions[index]
        for index in range(retained_checkpoint_count)
    )
    checkpoint_velocities = tuple(
        device_checkpoint_velocities[index]
        for index in range(retained_checkpoint_count)
    )
    host_checkpoint_accepted = host_counts[
        COUNT_HEADER_ROWS : COUNT_HEADER_ROWS + retained_checkpoint_count
    ]
    host_checkpoint_rejected = host_counts[
        COUNT_HEADER_ROWS + checkpoint_count :
        COUNT_HEADER_ROWS + checkpoint_count + retained_checkpoint_count
    ]
    checkpoint_accepted_steps = tuple(host_checkpoint_accepted)
    checkpoint_rejected_steps = tuple(host_checkpoint_rejected)
    accepted_counts = host_counts[1]
    maximum_accepted = int(np.max(accepted_counts)) if lane_count else 0
    if maximum_accepted > spec.ledger_capacity:
        raise AdaptivePrototypeError("device ledger count exceeds capacity")
    ledger_started = time.perf_counter()
    if maximum_accepted:
        host_ledger = cp.asnumpy(device_ledger[:, :, :maximum_accepted])
        host_epoch_ledger = host_ledger[0]
        host_magnitude_ledger = host_ledger[1]
    else:
        host_epoch_ledger = np.empty((lane_count, 0), dtype=np.float64)
        host_magnitude_ledger = np.empty((lane_count, 0), dtype=np.float64)
    ledger_finished = time.perf_counter()
    result = AdaptiveFusedResult(
        positions=positions,
        velocities=velocities,
        position_carry=position_carry,
        velocity_carry=velocity_carry,
        epochs=epochs,
        proposed_steps=proposed_steps,
        attempted_steps=attempted_steps,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        maximum_normalized_error=maximum_error,
        done=done,
        status=status,
        body_count=body_count,
        lane_count=lane_count,
        kernel_launch_count=launch_count,
        checkpoint_positions=checkpoint_positions,
        checkpoint_velocities=checkpoint_velocities,
        checkpoint_accepted_steps=checkpoint_accepted_steps,
        checkpoint_rejected_steps=checkpoint_rejected_steps,
        accepted_step_epochs=tuple(
            tuple(float(value) for value in host_epoch_ledger[lane, :count])
            for lane, count in enumerate(accepted_counts)
        ),
        accepted_step_magnitudes=tuple(
            tuple(float(value) for value in host_magnitude_ledger[lane, :count])
            for lane, count in enumerate(accepted_counts)
        ),
    )
    if phase_timings is not None:
        profile_finished = time.perf_counter()
        phase_timings["kernel_seconds"] = kernel_seconds
        phase_timings["metadata_transfer_seconds"] = metadata_transfer_seconds
        phase_timings["ledger_transfer_seconds"] = ledger_finished - ledger_started
        measured = sum(phase_timings.values())
        phase_timings["result_materialization_seconds"] = max(
            0.0,
            profile_finished - profile_started - measured,
        )
        phase_timings["total_seconds"] = profile_finished - profile_started
    return result


def _controller_factor(error: float, spec: AdaptiveFusedSpec) -> float:
    if error == 0.0:
        return spec.maximum_scale_factor
    return min(
        spec.maximum_scale_factor,
        max(spec.minimum_scale_factor, spec.safety_factor * error ** -0.125),
    )


def _trial_endpoint(
    current_epoch: float,
    checkpoint_epoch: float,
    proposed_step: float,
    direction: float,
) -> tuple[float, float, bool]:
    """Construct an endpoint without exceeding the binary64 step proposal."""

    remaining = checkpoint_epoch - current_epoch
    if not direction * remaining > 0.0 or not math.isfinite(remaining):
        raise AdaptivePrototypeError("reference has invalid remaining interval")
    if proposed_step >= abs(remaining):
        return checkpoint_epoch, remaining, True
    endpoint = current_epoch + math.copysign(proposed_step, direction)
    if endpoint == current_epoch or not math.isfinite(endpoint):
        raise AdaptivePrototypeError("reference step cannot advance the epoch")
    step = endpoint - current_epoch
    if abs(step) > proposed_step:
        endpoint = math.nextafter(endpoint, current_epoch)
        step = endpoint - current_epoch
    if (
        not direction * step > 0.0
        or abs(step) > proposed_step
        or direction * (checkpoint_epoch - endpoint) <= 0.0
    ):
        raise AdaptivePrototypeError("reference cannot honor the proposed step")
    return endpoint, step, False


def numpy_adaptive_integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    spec: AdaptiveFusedSpec,
) -> dict[str, Any]:
    """Independent host controller around the engine's NumPy RKF78 step."""

    position = np.ascontiguousarray(positions, dtype=np.float64).copy()
    velocity = np.ascontiguousarray(velocities, dtype=np.float64).copy()
    gm = np.ascontiguousarray(gravitational_parameters, dtype=np.float64)
    position_carry = np.zeros_like(position)
    velocity_carry = np.zeros_like(velocity)
    epoch = spec.initial_epoch
    proposed = spec.initial_step
    attempted = accepted = rejected = 0
    maximum_error = 0.0
    accepted_step_epochs: list[float] = []
    accepted_step_magnitudes: list[float] = []
    checkpoint_positions = [position.copy()]
    checkpoint_velocities = [velocity.copy()]
    checkpoint_accepted_steps = [0]
    checkpoint_rejected_steps = [0]
    for checkpoint_epoch in spec.checkpoint_epochs[1:]:
        while epoch != checkpoint_epoch:
            if attempted >= spec.maximum_steps:
                raise AdaptivePrototypeError("NumPy reference exhausted maximum_steps")
            proposed = min(spec.maximum_step, max(spec.minimum_step, proposed))
            endpoint, step, clipped = _trial_endpoint(
                epoch,
                checkpoint_epoch,
                proposed,
                spec.direction,
            )
            trial = fused.numpy_reference_step(
                position,
                velocity,
                gm,
                step,
                position_carry=position_carry,
                velocity_carry=velocity_carry,
            )
            candidate_position = trial.positions[0]
            candidate_velocity = trial.velocities[0]
            position_scale = spec.position_atol + spec.position_rtol * np.maximum(
                np.abs(position), np.abs(candidate_position)
            )
            velocity_scale = spec.velocity_atol + spec.velocity_rtol * np.maximum(
                np.abs(velocity), np.abs(candidate_velocity)
            )
            error = max(
                float(np.max(np.abs(trial.position_defect[0]) / position_scale)),
                float(np.max(np.abs(trial.velocity_defect[0]) / velocity_scale)),
            )
            attempted += 1
            maximum_error = max(maximum_error, error)
            factor = _controller_factor(error, spec)
            if error <= 1.0:
                position = candidate_position.copy()
                velocity = candidate_velocity.copy()
                position_carry = trial.position_carry[0].copy()
                velocity_carry = trial.velocity_carry[0].copy()
                epoch = endpoint
                accepted += 1
                accepted_step_epochs.append(float(endpoint))
                accepted_step_magnitudes.append(float(abs(step)))
                if not clipped:
                    proposed = abs(step) * factor
            else:
                rejected += 1
                if rejected > spec.maximum_rejections:
                    raise AdaptivePrototypeError(
                        "NumPy reference exhausted rejections"
                    )
                if proposed <= spec.minimum_step or (
                    clipped and abs(step) < spec.minimum_step
                ):
                    raise AdaptivePrototypeError(
                        "NumPy reference rejected at minimum step"
                    )
                reduced = max(spec.minimum_step, abs(step) * factor)
                if not reduced < abs(step):
                    raise AdaptivePrototypeError(
                        "NumPy controller failed to reduce"
                    )
                proposed = reduced
        checkpoint_positions.append(position.copy())
        checkpoint_velocities.append(velocity.copy())
        checkpoint_accepted_steps.append(accepted)
        checkpoint_rejected_steps.append(rejected)
    return {
        "positions": position,
        "velocities": velocity,
        "position_carry": position_carry,
        "velocity_carry": velocity_carry,
        "epoch": epoch,
        "attempted_steps": attempted,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "maximum_normalized_error": maximum_error,
        "checkpoint_positions": tuple(checkpoint_positions),
        "checkpoint_velocities": tuple(checkpoint_velocities),
        "checkpoint_accepted_steps": tuple(checkpoint_accepted_steps),
        "checkpoint_rejected_steps": tuple(checkpoint_rejected_steps),
        "accepted_step_epochs": tuple(accepted_step_epochs),
        "accepted_step_magnitudes": tuple(accepted_step_magnitudes),
    }


def run_probe(
    *,
    lanes: int,
    bodies: int,
    horizon: float,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AdaptivePrototypeError(f"CuPy unavailable: {exc}") from exc
    host_position, host_velocity, host_gm = fused.deterministic_fixture(lanes, bodies)
    spec = AdaptiveFusedSpec(
        initial_epoch=0.0,
        final_epoch=horizon,
        initial_step=min(1.0e-4, horizon),
        minimum_step=1.0e-12,
        maximum_step=min(1.0e-3, horizon),
        position_atol=1.0e-12,
        position_rtol=1.0e-9,
        velocity_atol=1.0e-12,
        velocity_rtol=1.0e-9,
    )
    reference = numpy_adaptive_integrate(
        host_position[0], host_velocity[0], host_gm[0], spec
    )
    position = cp.asarray(host_position)
    velocity = cp.asarray(host_velocity)
    gm = cp.asarray(host_gm)
    for _ in range(warmup):
        result = fused_adaptive_integrate(position, velocity, gm, spec)
    cp.cuda.get_current_stream().synchronize()
    elapsed_rows: list[float] = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = fused_adaptive_integrate(position, velocity, gm, spec)
        cp.cuda.get_current_stream().synchronize()
        elapsed_rows.append(time.perf_counter() - started)
    assert result is not None
    gpu_position = cp.asnumpy(result.positions[0])
    gpu_velocity = cp.asnumpy(result.velocities[0])
    status = cp.asnumpy(result.status)
    done = cp.asnumpy(result.done)
    epochs = cp.asnumpy(result.epochs)
    attempted = cp.asnumpy(result.attempted_steps)
    accepted = cp.asnumpy(result.accepted_steps)
    rejected = cp.asnumpy(result.rejected_steps)
    maximum_difference = max(
        float(np.max(np.abs(gpu_position - reference["positions"]))),
        float(np.max(np.abs(gpu_velocity - reference["velocities"]))),
    )
    accounting_match = bool(
        int(attempted[0]) == reference["attempted_steps"]
        and int(accepted[0]) == reference["accepted_steps"]
        and int(rejected[0]) == reference["rejected_steps"]
    )
    endpoint_exact = bool(np.all(epochs == spec.final_epoch))
    median_elapsed = statistics.median(elapsed_rows)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = properties["name"]
    passed = bool(
        np.all(status == 0)
        and np.all(done == 1)
        and endpoint_exact
        and accounting_match
        and maximum_difference <= 5.0e-12
    )
    return {
        "schema": "jxplanetx.fused-adaptive-rkf78-prototype-probe.v5",
        "status": "PASS" if passed else "FAIL",
        "scope": "ENGINEERING_PROTOTYPE_NO_PRODUCTION_OR_SCIENTIFIC_QUALIFICATION",
        "workload": {
            "bodies_per_system": bodies,
            "independent_systems": lanes,
            "horizon_hex": horizon.hex(),
            "rkf78_stages_per_attempt": STAGE_COUNT,
            "device_ledger_capacity_per_lane": spec.ledger_capacity,
            "maximum_attempts_per_kernel_launch": spec.attempts_per_launch,
        },
        "controller": {
            "initial_step_hex": spec.initial_step.hex(),
            "minimum_step_hex": spec.minimum_step.hex(),
            "maximum_step_hex": spec.maximum_step.hex(),
            "position_atol_hex": spec.position_atol.hex(),
            "position_rtol_hex": spec.position_rtol.hex(),
            "velocity_atol_hex": spec.velocity_atol.hex(),
            "velocity_rtol_hex": spec.velocity_rtol.hex(),
        },
        "correctness": {
            "all_kernel_status_zero": bool(np.all(status == 0)),
            "all_lanes_done": bool(np.all(done == 1)),
            "all_final_epochs_exact": endpoint_exact,
            "lane_zero_accounting_matches_numpy": accounting_match,
            "lane_zero_maximum_endpoint_difference_hex": maximum_difference.hex(),
            "comparison_limit_hex": float(5.0e-12).hex(),
            "attempted_steps_lane_zero": int(attempted[0]),
            "accepted_steps_lane_zero": int(accepted[0]),
            "rejected_steps_lane_zero": int(rejected[0]),
            "kernel_launches": result.kernel_launch_count,
            "accepted_step_ledger_entries_lane_zero": len(
                result.accepted_step_epochs[0]
            ),
        },
        "ledger_transport": {
            "persistent_device_controller": True,
            "whole_checkpoint_schedule_device_resident": True,
            "accepted_entries_appended_on_device": True,
            "per_attempt_terminal_host_copy": False,
            "per_checkpoint_terminal_host_copy": False,
            "per_attempt_accepted_count_host_copy": False,
            "per_attempt_accepted_epoch_host_copy": False,
            "completed_ledger_copied_once": True,
            "packed_metadata_copied_once_per_safety_chunk": True,
            "accepted_epoch_and_magnitude_ledgers_share_one_transfer": True,
            "capacity_exhaustion_policy": "FAIL_CLOSED_STATUS_11",
        },
        "performance": {
            "warmup_integrations": warmup,
            "timed_repetitions": repetitions,
            "median_elapsed_seconds_hex": median_elapsed.hex(),
            "minimum_elapsed_seconds_hex": min(elapsed_rows).hex(),
            "maximum_elapsed_seconds_hex": max(elapsed_rows).hex(),
            "system_integrations_per_second_hex": (lanes / median_elapsed).hex(),
            "accepted_system_steps_per_second_hex": (
                lanes * float(np.mean(accepted)) / median_elapsed
            ).hex(),
        },
        "runtime": {
            "device_name": (
                raw_name.decode("ascii", "replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            ),
            "compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
            "cupy_version": cp.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "claim_controls": {
            "general_adaptive_engine_claimed": False,
            "production_authorized": False,
            "scientific_qualification_claimed": False,
            "speed_superiority_claimed": False,
        },
    }


def _write_new(path: Path, report: dict[str, Any]) -> None:
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--bodies", type=int, default=11)
    parser.add_argument("--horizon", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_probe(
        lanes=arguments.lanes,
        bodies=arguments.bodies,
        horizon=arguments.horizon,
        warmup=arguments.warmup,
        repetitions=arguments.repetitions,
    )
    if arguments.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _write_new(arguments.output.resolve(), report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
