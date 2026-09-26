"""Persistent batched CUDA GR15 integration with mutual EIH 1PN gravity.

Each CUDA block advances one independent system through the complete GR15
predictor, eight-stage corrector, convergence gate, error estimate, rejection
controller, and exact checkpoint schedule.  State and trajectory outputs stay
on the active CUDA device; only status and numerical accounting are audited on
the host.

The scientific scope is deliberately unchanged from the CPU component: this
is a screening-only mutual point-mass Newtonian-plus-EIH-1PN model, not a
production ephemeris or an exact general-relativity solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import sys
import time
from typing import Any

import numpy as np

from .eih_1pn_cuda import (
    CUDA_OPTIONS,
    EIH1PNCUDAContractError,
    EIH1PNCUDAError,
    EIH1PNCUDAUnavailableError,
    _cupy,
)
from .gr15 import GR15Spec, MAXIMUM_BODY_COUNT, MINIMUM_BODY_COUNT, STAGE_COUNT
from .gr15_eih_1pn import STATUS_NAMES
from .solar_system.eih_1pn import (
    COORDINATE_SCOPE,
    EIH1PNParameters,
    SCIENTIFIC_CLAIM_STATE,
)


METHOD_ID = "JX_GR15_EIH1PN_CUDA_V1"
MODEL_ID = "jx.gr15.eih-1pn-mutual-point-mass.cuda.v1"
MAXIMUM_SYSTEM_COUNT = 65_535

STATUS_SUCCESS = 0
STATUS_INPUT_DOMAIN = 1
STATUS_NONFINITE = 2
STATUS_FORCE_SINGULARITY = 3
STATUS_STEP_LIMIT = 4
STATUS_REJECTION_LIMIT = 5
STATUS_MINIMUM_STEP = 6
STATUS_CHECKPOINT_SCHEDULE = 7
STATUS_WEAK_FIELD_DOMAIN = 8


class GR15EIH1PNCUDAError(EIH1PNCUDAError):
    """Base error for the complete CUDA GR15-EIH1PN component."""


class GR15EIH1PNCUDAContractError(
    GR15EIH1PNCUDAError, EIH1PNCUDAContractError
):
    """A caller violated the explicit batched device-array contract."""


class GR15EIH1PNCUDAIntegrationError(GR15EIH1PNCUDAError):
    """One CUDA system failed closed with a named integration status."""

    def __init__(self, message: str, *, system: int, status: int) -> None:
        super().__init__(message)
        self.system = system
        self.status = status
        self.status_name = STATUS_NAMES.get(status, "UNKNOWN")


CUDA_SOURCE = r'''
#define JX_MAX_BODIES 32
#define JX_MAX_COMPONENTS 96
#define JX_STAGES 8

#define JX_SUCCESS 0
#define JX_INPUT_DOMAIN 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3
#define JX_STEP_LIMIT 4
#define JX_REJECTION_LIMIT 5
#define JX_MINIMUM_STEP 6
#define JX_CHECKPOINT_SCHEDULE 7
#define JX_WEAK_FIELD_DOMAIN 8
#define JX_INFINITY (__longlong_as_double(0x7ff0000000000000ULL))

__device__ __constant__ double jx_nodes[JX_STAGES] = {
    0.0,
    0.05626256053692214646565219,
    0.1802406917368923649875799,
    0.3526247171131696373739078,
    0.5471536263305553830014486,
    0.7342101772154105315232106,
    0.8853209468390957680903598,
    0.9775206135612875018911745
};

__device__ __constant__ double jx_weights[JX_STAGES] = {
    0.015625,
    0.09267907740148963927036449,
    0.1520653103233925644878716,
    0.1882587726945592782860646,
    0.1957860837262467965412498,
    0.173507397817250640114338,
    0.1248239506649324816289346,
    0.05725440737212859967117686
};

__device__ __constant__ double jx_matrix[JX_STAGES][JX_STAGES] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0218575357793353541137256, 0.03896658995493952390444515,
     -0.00655597715050407620077529, 0.003047987997701841242050686,
     -0.001628352413490498772971478, 0.0008640846291827019805984866,
     -0.0003967440731372719993749891, 0.0001074358128945721979540281},
    {0.01109019996716397002729751, 0.1074038731595102964556402,
     0.06962828556589075384540129, -0.01119811624863760220170688,
     0.004966269979023949391973244, -0.002444154466489499164429424,
     0.001082958425545010154963901, -0.0002886246451145135215598593},
    {0.01924685020403895099268005, 0.08260943089632460047050209,
     0.1713406191339361332668743, 0.08907176694972627693501597,
     -0.01341616503759130080277322, 0.005419027397884905017646852,
     -0.002219210818477691309457862, 0.0005723983873277628034195499},
    {0.01265460246028185956985975, 0.1004777151793904481256521,
     0.1398180360279955518397087, 0.2094549690931143370409547,
     0.0943551795688308622003237, -0.01290688650124075141876731,
     0.004346289187593836977968735, -0.001046278685410761334251823},
    {0.01804235378259889963420418, 0.08649817648448209743917763,
     0.1610353360774351425974393, 0.1756277504741355714468989,
     0.2161085695632558350760332, 0.08467721591812071483469406,
     -0.00974806918951167399370458, 0.001968844104893944488467863},
    {0.01375292550555745359579391, 0.09740289477548153254460886,
     0.1454582534249695619403189, 0.1968341735426425536397814,
     0.1845095077630667328461543, 0.1902735654385526020187613,
     0.06151604522964667650267326, -0.00442641884082134499773214},
    {0.01684882072563045519190064, 0.08960966405234871034068221,
     0.1562886199734024062385994, 0.1829587316772641566227832,
     0.202279157322782537796746, 0.1654383876133992588169816,
     0.1356456487169481184393683, 0.02845158347951185844411323}
};

__device__ __constant__ double jx_power_from_stage[JX_STAGES][JX_STAGES] = {
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {-31.5, 42.052027226455564, -15.670952569588987,
     7.9201408431608504, -4.3535853390147787, 2.3399024862198341,
     -1.0811358245335345, 0.29360317730105179},
    {315.0, -577.21416126072359, 406.69041857626132,
     -227.02389860295213, 129.18115030530788, -70.519962840869354,
     32.83459893515554, -8.9481451121796418},
    {-1443.75, 2987.0943715855974, -2679.9760150403058,
     1851.0327222669553, -1135.2827096874314, 641.02054258154885,
     -303.46999294770791, 83.331081241343568},
    {3465.0, -7620.5697393149057, 7756.0644384999332,
     -6185.4039244485239, 4210.6004379613469, -2505.1592252272112,
     1218.1102094876376, -338.64219695827626},
    {-4504.5, 10263.723976680187, -11268.144751043623,
     9902.2489954553439, -7389.710288840819, 4695.7156695112635,
     -2370.238886498817, 670.90528473646555},
    {3003.0, -6997.8879332146889, 8072.5982066560664,
     -7594.7233770253169, 6104.9930704174922, -4144.4893052958905,
     2192.7112066703403, -636.20186820800222},
    {-804.375, 1903.114833955942, -2271.9920421295196,
     2246.488603184227, -1916.0856046504753, 1381.9003387178452,
     -769.90929430944618, 230.85816523142665}
};

__device__ __forceinline__ void
jx_set_status_once(int* status, const int value)
{
    atomicCAS(status, JX_SUCCESS, value);
}

__device__ void
jx_eih_force(
    const double* position,
    const double* velocity,
    const double* gm,
    const int body_count,
    const double speed_of_light_squared,
    const double maximum_compactness,
    const double maximum_speed_fraction_squared,
    double* acceleration,
    int* status,
    double* observed_maximum_compactness,
    double* observed_maximum_speed_fraction_squared,
    double* newtonian,
    double* separation,
    double* potential,
    double* compactness,
    double* speed_fraction_squared)
{
    const int body = (int)threadIdx.x;
    const int active = body < body_count;
    if (active && *status == JX_SUCCESS) {
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
                jx_set_status_once(status, JX_FORCE_SINGULARITY);
                continue;
            }
            const double radius = sqrt(radius_squared);
            const double inverse_radius_cubed = 1.0 / (radius_squared * radius);
            if (!isfinite(inverse_radius_cubed)) {
                jx_set_status_once(status, JX_NONFINITE);
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

    if (body == 0 && *status == JX_SUCCESS) {
        double force_compactness = 0.0;
        double force_speed_fraction_squared = 0.0;
        for (int index = 0; index < body_count; ++index) {
            force_compactness = fmax(force_compactness, compactness[index]);
            force_speed_fraction_squared = fmax(
                force_speed_fraction_squared, speed_fraction_squared[index]);
        }
        if (!isfinite(force_compactness)
            || !isfinite(force_speed_fraction_squared)) {
            *status = JX_NONFINITE;
        } else if (force_compactness > maximum_compactness
            || force_speed_fraction_squared
                > maximum_speed_fraction_squared) {
            *status = JX_WEAK_FIELD_DOMAIN;
        } else {
            *observed_maximum_compactness = fmax(
                *observed_maximum_compactness, force_compactness);
            *observed_maximum_speed_fraction_squared = fmax(
                *observed_maximum_speed_fraction_squared,
                force_speed_fraction_squared);
        }
    }
    __syncthreads();

    if (active && *status == JX_SUCCESS) {
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
            const double radius = separation[body * JX_MAX_BODIES + source];
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
            jx_set_status_once(status, JX_NONFINITE);
        }
        acceleration[body * 3] = acceleration_x;
        acceleration[body * 3 + 1] = acceleration_y;
        acceleration[body * 3 + 2] = acceleration_z;
    }
    __syncthreads();
}

extern "C" __global__
void jx_gr15_eih_1pn_integrate_batch(
    const double* input_position,
    const double* input_velocity,
    const double* input_gm,
    const int gm_is_batched,
    const double* checkpoints,
    const int checkpoint_count,
    const int body_count,
    const int system_count,
    const double speed_of_light_squared,
    const double maximum_compactness,
    const double maximum_speed_fraction_squared,
    const double initial_step,
    const double minimum_step,
    const double maximum_step,
    const double epsilon,
    const double safety_factor,
    const double minimum_scale_factor,
    const double maximum_scale_factor,
    const double convergence_factor,
    const int maximum_iterations,
    const long long maximum_steps,
    const long long maximum_rejections,
    double* checkpoint_positions,
    double* checkpoint_velocities,
    long long* checkpoint_accepted,
    long long* checkpoint_rejected,
    long long* output_counters,
    double* output_metrics,
    double* output_diagnostics,
    int* output_status,
    double* previous_stage_acceleration,
    double* stage_position,
    double* stage_velocity,
    double* stage_acceleration,
    double* next_stage_position,
    double* next_stage_velocity)
{
    const int system = (int)blockIdx.x;
    const int body = (int)threadIdx.x;
    const int active = body < body_count;
    const int component_count = body_count * 3;
    const int input_offset = system * component_count;
    const int stage_system_offset = system * JX_STAGES * component_count;

    __shared__ double position[JX_MAX_COMPONENTS];
    __shared__ double velocity[JX_MAX_COMPONENTS];
    __shared__ double position_carry[JX_MAX_COMPONENTS];
    __shared__ double velocity_carry[JX_MAX_COMPONENTS];
    __shared__ double trial_position_carry[JX_MAX_COMPONENTS];
    __shared__ double trial_velocity_carry[JX_MAX_COMPONENTS];
    __shared__ double candidate_position[JX_MAX_COMPONENTS];
    __shared__ double candidate_velocity[JX_MAX_COMPONENTS];
    __shared__ double initial_acceleration[JX_MAX_COMPONENTS];
    __shared__ double gm[JX_MAX_BODIES];
    __shared__ double newtonian[JX_MAX_COMPONENTS];
    __shared__ double separation[JX_MAX_BODIES * JX_MAX_BODIES];
    __shared__ double potential[JX_MAX_BODIES];
    __shared__ double compactness[JX_MAX_BODIES];
    __shared__ double speed_fraction_squared[JX_MAX_BODIES];
    __shared__ double predictor_velocity_weight[JX_STAGES * JX_STAGES];
    __shared__ double predictor_position_weight[JX_STAGES * JX_STAGES];
    __shared__ double reduction_a[JX_MAX_BODIES];
    __shared__ double reduction_b[JX_MAX_BODIES];
    __shared__ long long counters[11];
    __shared__ double epoch;
    __shared__ double direction;
    __shared__ double proposed;
    __shared__ double step;
    __shared__ double previous_step;
    __shared__ double error_ratio;
    __shared__ double required;
    __shared__ double maximum_error;
    __shared__ double minimum_accepted_step;
    __shared__ double maximum_accepted_step;
    __shared__ double maximum_corrector_residual;
    __shared__ double observed_maximum_compactness;
    __shared__ double observed_maximum_speed_fraction_squared;
    __shared__ int status;
    __shared__ int history_valid;
    __shared__ int checkpoint_done;
    __shared__ int polynomial_predictor_used;
    __shared__ int converged;
    __shared__ int terminal_force_reused;
    __shared__ int copy_next_stage;
    __shared__ int accept_step;

    if (system >= system_count) return;
    if (body == 0) {
        status = JX_SUCCESS;
        epoch = checkpoints[0];
        direction = checkpoints[checkpoint_count - 1] > checkpoints[0]
            ? 1.0 : -1.0;
        proposed = initial_step;
        previous_step = 0.0;
        maximum_error = 0.0;
        minimum_accepted_step = JX_INFINITY;
        maximum_accepted_step = 0.0;
        maximum_corrector_residual = 0.0;
        observed_maximum_compactness = 0.0;
        observed_maximum_speed_fraction_squared = 0.0;
        history_valid = 0;
        for (int index = 0; index < 11; ++index) counters[index] = 0;
        if (checkpoint_count < 2 || !isfinite(checkpoints[0])
            || !isfinite(checkpoints[checkpoint_count - 1])
            || checkpoints[checkpoint_count - 1] == checkpoints[0]) {
            status = JX_CHECKPOINT_SCHEDULE;
        } else {
            for (int index = 1; index < checkpoint_count; ++index) {
                if (!isfinite(checkpoints[index])
                    || !(direction * (checkpoints[index]
                        - checkpoints[index - 1]) > 0.0)) {
                    status = JX_CHECKPOINT_SCHEDULE;
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (active) {
        const int gm_offset = gm_is_batched ? system * body_count : 0;
        const double current_gm = input_gm[gm_offset + body];
        gm[body] = current_gm;
        if (!isfinite(current_gm) || !(current_gm > 0.0)) {
            jx_set_status_once(&status, JX_INPUT_DOMAIN);
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int component = body * 3 + axis;
            const double current_position = input_position[input_offset + component];
            const double current_velocity = input_velocity[input_offset + component];
            position[component] = current_position;
            velocity[component] = current_velocity;
            position_carry[component] = 0.0;
            velocity_carry[component] = 0.0;
            if (!isfinite(current_position) || !isfinite(current_velocity)) {
                jx_set_status_once(&status, JX_NONFINITE);
            }
            const int output_component =
                (system * checkpoint_count) * component_count + component;
            checkpoint_positions[output_component] = current_position;
            checkpoint_velocities[output_component] = current_velocity;
        }
        for (int stage = 0; stage < JX_STAGES; ++stage) {
            for (int axis = 0; axis < 3; ++axis) {
                previous_stage_acceleration[
                    stage_system_offset + stage * component_count
                    + body * 3 + axis] = 0.0;
            }
        }
    }
    if (body == 0) {
        checkpoint_accepted[system * checkpoint_count] = 0;
        checkpoint_rejected[system * checkpoint_count] = 0;
    }
    __syncthreads();

    for (int checkpoint_index = 1;
         checkpoint_index < checkpoint_count && status == JX_SUCCESS;
         ++checkpoint_index) {
        const double checkpoint_epoch = checkpoints[checkpoint_index];
        while (status == JX_SUCCESS) {
            if (body == 0) {
                checkpoint_done = epoch == checkpoint_epoch;
                accept_step = 0;
                if (!checkpoint_done) {
                    const double remaining = checkpoint_epoch - epoch;
                    if (counters[0] >= maximum_steps) {
                        status = JX_STEP_LIMIT;
                    } else {
                        proposed = fmin(maximum_step, fmax(minimum_step, proposed));
                        if (!(direction * remaining > 0.0)
                            || !isfinite(remaining)) {
                            status = JX_CHECKPOINT_SCHEDULE;
                        } else {
                            step = copysign(fmin(proposed, fabs(remaining)), direction);
                            if (epoch + step == epoch || !isfinite(step)) {
                                status = JX_MINIMUM_STEP;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (status != JX_SUCCESS || checkpoint_done) break;

            if (active) {
                for (int axis = 0; axis < 3; ++axis) {
                    const int component = body * 3 + axis;
                    trial_position_carry[component] = position_carry[component];
                    trial_velocity_carry[component] = velocity_carry[component];
                }
            }
            if (body == 0) {
                error_ratio = JX_INFINITY;
                converged = 0;
                terminal_force_reused = 0;
                const double ratio = history_valid && previous_step != 0.0
                    ? step / previous_step : 0.0;
                polynomial_predictor_used = history_valid && isfinite(ratio)
                    && ratio >= 0.25 && ratio <= 2.0;
            }
            __syncthreads();

            jx_eih_force(
                position, velocity, gm, body_count, speed_of_light_squared,
                maximum_compactness, maximum_speed_fraction_squared,
                initial_acceleration, &status,
                &observed_maximum_compactness,
                &observed_maximum_speed_fraction_squared,
                newtonian, separation, potential, compactness,
                speed_fraction_squared);
            if (body == 0 && status == JX_SUCCESS) counters[3] += 1;
            __syncthreads();
            if (status != JX_SUCCESS) break;

            if (body == 0 && polynomial_predictor_used) {
                const double ratio = step / previous_step;
                for (int stage = 0; stage < JX_STAGES; ++stage) {
                    const double normalized = 1.0 + ratio * jx_nodes[stage];
                    for (int source = 0; source < JX_STAGES; ++source) {
                        double integral = 0.0;
                        double moment = 0.0;
                        double power_value = normalized;
                        for (int power = 0; power < JX_STAGES; ++power) {
                            const double next_power = power_value * normalized;
                            const double coefficient =
                                jx_power_from_stage[power][source];
                            integral += coefficient * (power_value - 1.0)
                                / (double)(power + 1);
                            moment += coefficient * (next_power - 1.0)
                                / (double)(power + 2);
                            power_value = next_power;
                        }
                        predictor_velocity_weight[stage * JX_STAGES + source] =
                            integral;
                        predictor_position_weight[stage * JX_STAGES + source] =
                            normalized * integral - moment;
                    }
                }
            }
            __syncthreads();

            if (active) {
                const double ratio = history_valid && previous_step != 0.0
                    ? step / previous_step : 0.0;
                for (int stage = 0; stage < JX_STAGES; ++stage) {
                    if (polynomial_predictor_used) {
                        const double normalized = 1.0 + ratio * jx_nodes[stage];
                        for (int axis = 0; axis < 3; ++axis) {
                            const int component = body * 3 + axis;
                            double acceleration_integral = 0.0;
                            double acceleration_double_integral = 0.0;
                            for (int source = 0; source < JX_STAGES; ++source) {
                                const double acceleration =
                                    previous_stage_acceleration[
                                        stage_system_offset
                                        + source * component_count + component];
                                acceleration_integral +=
                                    predictor_velocity_weight[
                                        stage * JX_STAGES + source]
                                    * acceleration;
                                acceleration_double_integral +=
                                    predictor_position_weight[
                                        stage * JX_STAGES + source]
                                    * acceleration;
                            }
                            const double predicted_velocity = velocity[component]
                                + previous_step * acceleration_integral;
                            const double predicted_position = position[component]
                                + previous_step * (normalized - 1.0)
                                    * velocity[component]
                                + previous_step * previous_step
                                    * acceleration_double_integral;
                            const int offset = stage_system_offset
                                + stage * component_count + component;
                            stage_velocity[offset] = predicted_velocity;
                            stage_position[offset] = predicted_position;
                            if (!isfinite(predicted_position)
                                || !isfinite(predicted_velocity)) {
                                jx_set_status_once(&status, JX_NONFINITE);
                            }
                        }
                    } else {
                        const double offset_time = jx_nodes[stage] * step;
                        for (int axis = 0; axis < 3; ++axis) {
                            const int component = body * 3 + axis;
                            const int offset = stage_system_offset
                                + stage * component_count + component;
                            stage_position[offset] = position[component]
                                + offset_time * velocity[component]
                                + 0.5 * offset_time * offset_time
                                    * initial_acceleration[component];
                            stage_velocity[offset] = velocity[component]
                                + offset_time * initial_acceleration[component];
                        }
                    }
                }
            }
            __syncthreads();
            if (status != JX_SUCCESS) break;

            for (int iteration = 0;
                 iteration < maximum_iterations && status == JX_SUCCESS;
                 ++iteration) {
                for (int stage = 0;
                     stage < JX_STAGES && status == JX_SUCCESS;
                     ++stage) {
                    const int offset = stage_system_offset
                        + stage * component_count;
                    jx_eih_force(
                        stage_position + offset,
                        stage_velocity + offset,
                        gm, body_count, speed_of_light_squared,
                        maximum_compactness,
                        maximum_speed_fraction_squared,
                        stage_acceleration + offset, &status,
                        &observed_maximum_compactness,
                        &observed_maximum_speed_fraction_squared,
                        newtonian, separation, potential, compactness,
                        speed_fraction_squared);
                    if (body == 0 && status == JX_SUCCESS) counters[3] += 1;
                    __syncthreads();
                }
                if (status != JX_SUCCESS) break;

                double local_maximum_scaled_change = 0.0;
                if (active) {
                    for (int stage = 0; stage < JX_STAGES; ++stage) {
                        for (int axis = 0; axis < 3; ++axis) {
                            const int component = body * 3 + axis;
                            double position_sum = 0.0;
                            double velocity_sum = 0.0;
                            for (int source = 0; source < JX_STAGES; ++source) {
                                const int source_offset = stage_system_offset
                                    + source * component_count + component;
                                const double coefficient = jx_matrix[stage][source];
                                position_sum += coefficient
                                    * stage_velocity[source_offset];
                                velocity_sum += coefficient
                                    * stage_acceleration[source_offset];
                            }
                            const int offset = stage_system_offset
                                + stage * component_count + component;
                            const double next_position = position[component]
                                + step * position_sum;
                            const double next_velocity = velocity[component]
                                + step * velocity_sum;
                            next_stage_position[offset] = next_position;
                            next_stage_velocity[offset] = next_velocity;
                            if (!isfinite(next_position)
                                || !isfinite(next_velocity)) {
                                jx_set_status_once(&status, JX_NONFINITE);
                            }
                            const double position_scale = 1.0 + fmax(
                                fabs(stage_position[offset]), fabs(next_position));
                            const double velocity_scale = 1.0 + fmax(
                                fabs(stage_velocity[offset]), fabs(next_velocity));
                            local_maximum_scaled_change = fmax(
                                local_maximum_scaled_change,
                                fabs(next_position - stage_position[offset])
                                    / position_scale);
                            local_maximum_scaled_change = fmax(
                                local_maximum_scaled_change,
                                fabs(next_velocity - stage_velocity[offset])
                                    / velocity_scale);
                        }
                    }
                }
                reduction_a[body] = active ? local_maximum_scaled_change : 0.0;
                __syncthreads();
                if (body == 0 && status == JX_SUCCESS) {
                    double maximum_scaled_change = 0.0;
                    for (int index = 0; index < body_count; ++index) {
                        maximum_scaled_change = fmax(
                            maximum_scaled_change, reduction_a[index]);
                    }
                    counters[4] += 1;
                    const double corrector_threshold = fmax(
                        convergence_factor * 2.2204460492503130808472633361816e-16,
                        epsilon * 9.5367431640625e-7);
                    converged = maximum_scaled_change <= corrector_threshold;
                    if (converged) {
                        maximum_corrector_residual = fmax(
                            maximum_corrector_residual,
                            maximum_scaled_change);
                        terminal_force_reused = maximum_scaled_change
                            <= convergence_factor
                                * 2.2204460492503130808472633361816e-16;
                        if (terminal_force_reused) counters[9] += 1;
                    }
                    copy_next_stage = !converged || !terminal_force_reused;
                }
                __syncthreads();
                if (active && copy_next_stage && status == JX_SUCCESS) {
                    for (int stage = 0; stage < JX_STAGES; ++stage) {
                        for (int axis = 0; axis < 3; ++axis) {
                            const int offset = stage_system_offset
                                + stage * component_count + body * 3 + axis;
                            stage_position[offset] = next_stage_position[offset];
                            stage_velocity[offset] = next_stage_velocity[offset];
                        }
                    }
                }
                __syncthreads();
                if (converged) break;
            }
            if (status != JX_SUCCESS) break;

            if (converged && !terminal_force_reused) {
                for (int stage = 0;
                     stage < JX_STAGES && status == JX_SUCCESS;
                     ++stage) {
                    const int offset = stage_system_offset
                        + stage * component_count;
                    jx_eih_force(
                        stage_position + offset,
                        stage_velocity + offset,
                        gm, body_count, speed_of_light_squared,
                        maximum_compactness,
                        maximum_speed_fraction_squared,
                        stage_acceleration + offset, &status,
                        &observed_maximum_compactness,
                        &observed_maximum_speed_fraction_squared,
                        newtonian, separation, potential, compactness,
                        speed_fraction_squared);
                    if (body == 0 && status == JX_SUCCESS) counters[3] += 1;
                    __syncthreads();
                }
                if (body == 0 && status == JX_SUCCESS) counters[10] += 1;
                __syncthreads();
            }
            if (status != JX_SUCCESS) break;

            if (converged) {
                double local_maximum_acceleration = 0.0;
                double local_maximum_highest = 0.0;
                if (active) {
                    for (int axis = 0; axis < 3; ++axis) {
                        const int component = body * 3 + axis;
                        double position_sum = 0.0;
                        double velocity_sum = 0.0;
                        double divided[JX_STAGES];
                        for (int stage = 0; stage < JX_STAGES; ++stage) {
                            const int offset = stage_system_offset
                                + stage * component_count + component;
                            position_sum += jx_weights[stage]
                                * stage_velocity[offset];
                            velocity_sum += jx_weights[stage]
                                * stage_acceleration[offset];
                            local_maximum_acceleration = fmax(
                                local_maximum_acceleration,
                                fabs(stage_acceleration[offset]));
                            divided[stage] = stage_acceleration[offset];
                        }
                        for (int stage = 1; stage < JX_STAGES; ++stage) {
                            for (int index = JX_STAGES - 1;
                                 index >= stage; --index) {
                                divided[index] =
                                    (divided[index] - divided[index - 1])
                                    / (jx_nodes[index]
                                        - jx_nodes[index - stage]);
                            }
                        }
                        local_maximum_highest = fmax(
                            local_maximum_highest, fabs(divided[7]));
                        const double adjusted_position =
                            step * position_sum - trial_position_carry[component];
                        const double adjusted_velocity =
                            step * velocity_sum - trial_velocity_carry[component];
                        candidate_position[component] = position[component]
                            + adjusted_position;
                        candidate_velocity[component] = velocity[component]
                            + adjusted_velocity;
                        trial_position_carry[component] =
                            (candidate_position[component] - position[component])
                            - adjusted_position;
                        trial_velocity_carry[component] =
                            (candidate_velocity[component] - velocity[component])
                            - adjusted_velocity;
                        if (!isfinite(candidate_position[component])
                            || !isfinite(candidate_velocity[component])) {
                            jx_set_status_once(&status, JX_NONFINITE);
                        }
                    }
                }
                reduction_a[body] = active ? local_maximum_acceleration : 0.0;
                reduction_b[body] = active ? local_maximum_highest : 0.0;
                __syncthreads();
                if (body == 0 && status == JX_SUCCESS) {
                    double maximum_acceleration = 0.0;
                    double maximum_highest = 0.0;
                    for (int index = 0; index < body_count; ++index) {
                        maximum_acceleration = fmax(
                            maximum_acceleration, reduction_a[index]);
                        maximum_highest = fmax(maximum_highest, reduction_b[index]);
                    }
                    if (!(maximum_acceleration > 0.0)
                        || !isfinite(maximum_highest)) {
                        status = JX_NONFINITE;
                    } else {
                        error_ratio = maximum_highest / maximum_acceleration;
                        if (!isfinite(error_ratio) || error_ratio < 0.0) {
                            status = JX_NONFINITE;
                        }
                    }
                }
                __syncthreads();
            }
            if (status != JX_SUCCESS) break;

            if (body == 0) {
                counters[0] += 1;
                if (polynomial_predictor_used) counters[6] += 1;
                else counters[7] += 1;
                maximum_error = fmax(maximum_error, error_ratio);
                if (!converged) {
                    counters[2] += 1;
                    counters[5] += 1;
                    if (counters[2] > maximum_rejections) {
                        status = JX_REJECTION_LIMIT;
                    } else if (proposed <= minimum_step) {
                        status = JX_MINIMUM_STEP;
                    } else {
                        proposed = fmax(minimum_step, 0.5 * fabs(step));
                    }
                } else {
                    double next_factor = maximum_scale_factor;
                    if (error_ratio > 0.0) {
                        next_factor = safety_factor
                            * pow(epsilon / error_ratio, 1.0 / 7.0);
                        next_factor = fmin(maximum_scale_factor,
                            fmax(minimum_scale_factor, next_factor));
                    }
                    required = fabs(step) * next_factor;
                    if (error_ratio > epsilon) {
                        counters[2] += 1;
                        if (counters[2] > maximum_rejections) {
                            status = JX_REJECTION_LIMIT;
                        } else if (fabs(step) <= minimum_step) {
                            status = JX_MINIMUM_STEP;
                        } else {
                            proposed = fmax(minimum_step, required);
                        }
                    } else {
                        accept_step = 1;
                    }
                }
            }
            __syncthreads();
            if (status != JX_SUCCESS) break;

            if (accept_step && active) {
                for (int axis = 0; axis < 3; ++axis) {
                    const int component = body * 3 + axis;
                    position[component] = candidate_position[component];
                    velocity[component] = candidate_velocity[component];
                    position_carry[component] = trial_position_carry[component];
                    velocity_carry[component] = trial_velocity_carry[component];
                    for (int stage = 0; stage < JX_STAGES; ++stage) {
                        const int offset = stage_system_offset
                            + stage * component_count + component;
                        previous_stage_acceleration[offset] =
                            stage_acceleration[offset];
                    }
                }
            }
            __syncthreads();
            if (body == 0 && accept_step) {
                previous_step = step;
                history_valid = 1;
                counters[8] += 1;
                epoch += step;
                if (fabs(checkpoint_epoch - epoch)
                    <= 2.0 * 2.2204460492503130808472633361816e-16
                        * fmax(1.0, fabs(checkpoint_epoch))) {
                    epoch = checkpoint_epoch;
                }
                counters[1] += 1;
                minimum_accepted_step = fmin(
                    minimum_accepted_step, fabs(step));
                maximum_accepted_step = fmax(
                    maximum_accepted_step, fabs(step));
                proposed = fmin(maximum_step,
                    fmax(minimum_step, required));
            }
            __syncthreads();
        }

        if (status == JX_SUCCESS) {
            if (active) {
                for (int axis = 0; axis < 3; ++axis) {
                    const int component = body * 3 + axis;
                    const int output_component =
                        (system * checkpoint_count + checkpoint_index)
                            * component_count + component;
                    checkpoint_positions[output_component] = position[component];
                    checkpoint_velocities[output_component] = velocity[component];
                }
            }
            if (body == 0) {
                checkpoint_accepted[
                    system * checkpoint_count + checkpoint_index] = counters[1];
                checkpoint_rejected[
                    system * checkpoint_count + checkpoint_index] = counters[2];
            }
        }
        __syncthreads();
    }

    if (body == 0) {
        for (int index = 0; index < 11; ++index) {
            output_counters[system * 11 + index] = counters[index];
        }
        output_metrics[system * 5] = maximum_error;
        output_metrics[system * 5 + 1] = minimum_accepted_step;
        output_metrics[system * 5 + 2] = maximum_accepted_step;
        output_metrics[system * 5 + 3] = proposed;
        output_metrics[system * 5 + 4] = maximum_corrector_residual;
        output_diagnostics[system * 2] = observed_maximum_compactness;
        output_diagnostics[system * 2 + 1] =
            observed_maximum_speed_fraction_squared;
        output_status[system] = status;
    }
}
'''

CUDA_SOURCE_SHA256 = hashlib.sha256(CUDA_SOURCE.encode("utf-8")).hexdigest()

_CUDA_KERNEL: Any | None = None


@dataclass(frozen=True, slots=True)
class GR15EIH1PNCUDASystemAudit:
    """Host-audited controller accounting for one CUDA system."""

    checkpoint_accepted_steps: tuple[int, ...]
    checkpoint_rejected_steps: tuple[int, ...]
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    force_evaluations: int
    predictor_corrector_iterations: int
    nonconverged_retries: int
    polynomial_predictor_trials: int
    constant_predictor_trials: int
    history_commits: int
    terminal_force_reuses: int
    terminal_force_sweeps: int
    maximum_error_ratio: float
    minimum_accepted_step: float
    maximum_accepted_step: float
    final_proposed_step: float
    maximum_corrector_residual: float
    observed_maximum_compactness: float
    observed_maximum_speed_fraction_squared: float

    def __post_init__(self) -> None:
        if self.attempted_steps != self.accepted_steps + self.rejected_steps:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 attempted-step accounting is inconsistent"
            )
        if self.accepted_steps <= 0 or self.rejected_steps < 0:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 accepted/rejected accounting is invalid"
            )
        if self.history_commits != self.accepted_steps:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 history accounting is inconsistent"
            )
        if self.polynomial_predictor_trials + self.constant_predictor_trials != (
            self.attempted_steps
        ):
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 predictor accounting is inconsistent"
            )
        if self.terminal_force_reuses + self.terminal_force_sweeps != (
            self.attempted_steps - self.nonconverged_retries
        ):
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 terminal-force accounting is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class GR15EIH1PNCUDABatchResult:
    """Device-resident trajectories and host-audited adaptive accounting."""

    checkpoint_positions_km: Any
    checkpoint_velocities_km_s: Any
    positions_km: Any
    velocities_km_s: Any
    checkpoint_epochs: tuple[float, ...]
    system_audits: tuple[GR15EIH1PNCUDASystemAudit, ...]
    system_count: int
    body_count: int
    elapsed_seconds: float
    method_id: str = METHOD_ID
    model_id: str = MODEL_ID
    coordinate_scope: str = COORDINATE_SCOPE
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    kernel_launch_count: int = 1
    full_predictor_corrector_executed: bool = True
    adaptive_controller_executed: bool = True
    host_audit_performed: bool = True
    production_authorized: bool = False
    exact_general_relativity_claimed: bool = False
    ephemeris_equivalence_claimed: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.model_id != MODEL_ID:
            raise GR15EIH1PNCUDAContractError("CUDA GR15 result identity changed")
        if self.coordinate_scope != COORDINATE_SCOPE:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 coordinate scope changed"
            )
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 scientific claim state changed"
            )
        if (
            self.kernel_launch_count != 1
            or not self.full_predictor_corrector_executed
            or not self.adaptive_controller_executed
            or not self.host_audit_performed
        ):
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 execution accounting changed"
            )
        if len(self.system_audits) != self.system_count:
            raise GR15EIH1PNCUDAContractError("CUDA GR15 audit count changed")
        if (
            self.production_authorized
            or self.exact_general_relativity_claimed
            or self.ephemeris_equivalence_claimed
            or self.general_superiority_claimed
        ):
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 screening result elevated a scientific claim"
            )


def _kernel(cp: Any) -> Any:
    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        try:
            _CUDA_KERNEL = cp.RawKernel(
                CUDA_SOURCE,
                "jx_gr15_eih_1pn_integrate_batch",
                options=CUDA_OPTIONS,
            )
        except Exception as exc:  # pragma: no cover - compiler dependent
            raise EIH1PNCUDAUnavailableError(
                f"the complete GR15-EIH1PN CUDA kernel could not be built: {exc}"
            ) from exc
    return _CUDA_KERNEL


def gr15_eih_1pn_cuda_runtime_identity() -> dict[str, Any]:
    """Return the exact persistent kernel and active CUDA device identity."""

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


def _require_device_state(cp: Any, value: Any, label: str) -> Any:
    if type(value) is not cp.ndarray:
        raise GR15EIH1PNCUDAContractError(f"{label} must be an exact CuPy array")
    if value.dtype != cp.dtype(cp.float64) or value.ndim != 3 or value.shape[2] != 3:
        raise GR15EIH1PNCUDAContractError(
            f"{label} must have CuPy float64 shape (systems,bodies,3)"
        )
    if not value.flags.c_contiguous:
        raise GR15EIH1PNCUDAContractError(f"{label} must be C-contiguous")
    return value


def integrate_gr15_eih_1pn_cuda_batch(
    positions_km: Any,
    velocities_km_s: Any,
    gravitational_parameters_km3_s2: Any,
    spec: GR15Spec,
    parameters: EIH1PNParameters,
) -> GR15EIH1PNCUDABatchResult:
    """Advance independent 2--32 body systems in one persistent CUDA launch.

    Position, velocity, and GM inputs must already be exact C-contiguous CuPy
    float64 arrays on the active device.  State arrays have shape
    ``(systems,bodies,3)``. GM may be shared with shape ``(bodies,)`` or have
    shape ``(systems,bodies)``. No implicit state transfer is performed.
    """

    if type(spec) is not GR15Spec:
        raise GR15EIH1PNCUDAContractError("spec must be an exact GR15Spec")
    if type(parameters) is not EIH1PNParameters:
        raise GR15EIH1PNCUDAContractError(
            "parameters must be an exact EIH1PNParameters"
        )
    cp = _cupy()
    positions = _require_device_state(cp, positions_km, "positions_km")
    velocities = _require_device_state(cp, velocities_km_s, "velocities_km_s")
    if velocities.shape != positions.shape:
        raise GR15EIH1PNCUDAContractError(
            "position and velocity shapes must match"
        )
    system_count, body_count, _ = positions.shape
    if not 1 <= system_count <= MAXIMUM_SYSTEM_COUNT:
        raise GR15EIH1PNCUDAContractError(
            f"system count must be 1--{MAXIMUM_SYSTEM_COUNT}"
        )
    if not MINIMUM_BODY_COUNT <= body_count <= MAXIMUM_BODY_COUNT:
        raise GR15EIH1PNCUDAContractError(
            "CUDA GR15-EIH1PN supports exactly 2--32 bodies"
        )

    gm = gravitational_parameters_km3_s2
    if type(gm) is not cp.ndarray or gm.dtype != cp.dtype(cp.float64):
        raise GR15EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be an exact CuPy float64 array"
        )
    if gm.shape == (body_count,):
        gm_is_batched = 0
    elif gm.shape == (system_count, body_count):
        gm_is_batched = 1
    else:
        raise GR15EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must have shape (bodies,) "
            "or (systems,bodies)"
        )
    if not gm.flags.c_contiguous:
        raise GR15EIH1PNCUDAContractError(
            "gravitational_parameters_km3_s2 must be C-contiguous"
        )

    speed_of_light_squared = (
        parameters.speed_of_light_km_s * parameters.speed_of_light_km_s
    )
    if not math.isfinite(speed_of_light_squared):
        raise GR15EIH1PNCUDAContractError(
            "speed_of_light_km_s squared must remain finite"
        )

    checkpoints = cp.asarray(spec.checkpoint_epochs, dtype=cp.float64)
    checkpoint_count = len(spec.checkpoint_epochs)
    output_shape = (system_count, checkpoint_count, body_count, 3)
    checkpoint_positions = cp.empty(output_shape, dtype=cp.float64)
    checkpoint_velocities = cp.empty(output_shape, dtype=cp.float64)
    checkpoint_accepted = cp.empty(
        (system_count, checkpoint_count), dtype=cp.int64
    )
    checkpoint_rejected = cp.empty_like(checkpoint_accepted)
    counters = cp.empty((system_count, 11), dtype=cp.int64)
    metrics = cp.empty((system_count, 5), dtype=cp.float64)
    diagnostics = cp.empty((system_count, 2), dtype=cp.float64)
    status = cp.empty(system_count, dtype=cp.int32)
    stage_shape = (system_count, STAGE_COUNT, body_count, 3)
    previous_stage_acceleration = cp.empty(stage_shape, dtype=cp.float64)
    stage_position = cp.empty(stage_shape, dtype=cp.float64)
    stage_velocity = cp.empty(stage_shape, dtype=cp.float64)
    stage_acceleration = cp.empty(stage_shape, dtype=cp.float64)
    next_stage_position = cp.empty(stage_shape, dtype=cp.float64)
    next_stage_velocity = cp.empty(stage_shape, dtype=cp.float64)

    started = time.perf_counter_ns()
    try:
        _kernel(cp)(
            (system_count,),
            (MAXIMUM_BODY_COUNT,),
            (
                positions,
                velocities,
                gm,
                gm_is_batched,
                checkpoints,
                checkpoint_count,
                body_count,
                system_count,
                float(speed_of_light_squared),
                parameters.maximum_compactness,
                parameters.maximum_speed_fraction_squared,
                spec.initial_step,
                spec.minimum_step,
                spec.maximum_step,
                spec.epsilon,
                spec.safety_factor,
                spec.minimum_scale_factor,
                spec.maximum_scale_factor,
                spec.convergence_factor,
                spec.maximum_iterations,
                spec.maximum_steps,
                spec.maximum_rejections,
                checkpoint_positions,
                checkpoint_velocities,
                checkpoint_accepted,
                checkpoint_rejected,
                counters,
                metrics,
                diagnostics,
                status,
                previous_stage_acceleration,
                stage_position,
                stage_velocity,
                stage_acceleration,
                next_stage_position,
                next_stage_velocity,
            ),
        )
        status_host = cp.asnumpy(status)
        counters_host = cp.asnumpy(counters)
        metrics_host = cp.asnumpy(metrics)
        diagnostics_host = cp.asnumpy(diagnostics)
        accepted_host = cp.asnumpy(checkpoint_accepted)
        rejected_host = cp.asnumpy(checkpoint_rejected)
    except GR15EIH1PNCUDAError:
        raise
    except Exception as exc:  # pragma: no cover - driver dependent
        raise EIH1PNCUDAUnavailableError(
            f"the complete GR15-EIH1PN CUDA kernel failed: {exc}"
        ) from exc
    elapsed_seconds = (time.perf_counter_ns() - started) * 1.0e-9

    for system, current_status in enumerate(status_host.tolist()):
        current_status = int(current_status)
        if current_status != STATUS_SUCCESS:
            status_name = STATUS_NAMES.get(current_status, "UNKNOWN")
            raise GR15EIH1PNCUDAIntegrationError(
                f"CUDA GR15-EIH1PN system {system} failed with {status_name}",
                system=system,
                status=current_status,
            )

    corrector_threshold = max(
        spec.convergence_factor * sys.float_info.epsilon,
        spec.epsilon * 2.0**-20,
    )
    audits: list[GR15EIH1PNCUDASystemAudit] = []
    for system in range(system_count):
        current_counters = counters_host[system]
        current_metrics = metrics_host[system]
        current_diagnostics = diagnostics_host[system]
        if not np.all(np.isfinite(current_diagnostics)) or np.any(
            current_diagnostics < 0.0
        ):
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 force diagnostics are invalid"
            )
        if not 0.0 <= float(current_metrics[4]) <= corrector_threshold:
            raise GR15EIH1PNCUDAContractError(
                "CUDA GR15 corrector residual exceeded its threshold"
            )
        audits.append(
            GR15EIH1PNCUDASystemAudit(
                checkpoint_accepted_steps=tuple(
                    int(value) for value in accepted_host[system]
                ),
                checkpoint_rejected_steps=tuple(
                    int(value) for value in rejected_host[system]
                ),
                attempted_steps=int(current_counters[0]),
                accepted_steps=int(current_counters[1]),
                rejected_steps=int(current_counters[2]),
                force_evaluations=int(current_counters[3]),
                predictor_corrector_iterations=int(current_counters[4]),
                nonconverged_retries=int(current_counters[5]),
                polynomial_predictor_trials=int(current_counters[6]),
                constant_predictor_trials=int(current_counters[7]),
                history_commits=int(current_counters[8]),
                terminal_force_reuses=int(current_counters[9]),
                terminal_force_sweeps=int(current_counters[10]),
                maximum_error_ratio=float(current_metrics[0]),
                minimum_accepted_step=float(current_metrics[1]),
                maximum_accepted_step=float(current_metrics[2]),
                final_proposed_step=float(current_metrics[3]),
                maximum_corrector_residual=float(current_metrics[4]),
                observed_maximum_compactness=float(current_diagnostics[0]),
                observed_maximum_speed_fraction_squared=float(
                    current_diagnostics[1]
                ),
            )
        )

    final_positions = checkpoint_positions[:, -1]
    final_velocities = checkpoint_velocities[:, -1]
    return GR15EIH1PNCUDABatchResult(
        checkpoint_positions_km=checkpoint_positions,
        checkpoint_velocities_km_s=checkpoint_velocities,
        positions_km=final_positions,
        velocities_km_s=final_velocities,
        checkpoint_epochs=spec.checkpoint_epochs,
        system_audits=tuple(audits),
        system_count=system_count,
        body_count=body_count,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = [
    "CUDA_OPTIONS",
    "CUDA_SOURCE_SHA256",
    "GR15EIH1PNCUDABatchResult",
    "GR15EIH1PNCUDAContractError",
    "GR15EIH1PNCUDAError",
    "GR15EIH1PNCUDAIntegrationError",
    "GR15EIH1PNCUDASystemAudit",
    "MAXIMUM_SYSTEM_COUNT",
    "METHOD_ID",
    "MODEL_ID",
    "gr15_eih_1pn_cuda_runtime_identity",
    "integrate_gr15_eih_1pn_cuda_batch",
]
