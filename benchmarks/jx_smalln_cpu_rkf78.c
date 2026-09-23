#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define JX_STAGE_COUNT 13
#define JX_MAX_BODIES 32
#define JX_MAX_COMPONENTS (JX_MAX_BODIES * 3)

int jx_smalln_cpu_rkf78_integrate(
    int body_count,
    int checkpoint_count,
    const double *input_position,
    const double *input_velocity,
    const double *gm,
    const double *checkpoints,
    const double *tableau_a,
    const double *weight_eighth,
    const double *weight_defect,
    double initial_step,
    double minimum_step,
    double maximum_step,
    double position_atol,
    double position_rtol,
    double velocity_atol,
    double velocity_rtol,
    double safety_factor,
    double minimum_scale_factor,
    double maximum_scale_factor,
    int64_t maximum_steps,
    int64_t maximum_rejections,
    int64_t ledger_capacity,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    int64_t *checkpoint_accepted,
    int64_t *checkpoint_rejected,
    double *accepted_epoch_ledger,
    double *accepted_magnitude_ledger,
    int64_t *attempted_out,
    int64_t *accepted_out,
    int64_t *rejected_out,
    double *maximum_error_out)
{
    if (body_count < 2 || body_count > JX_MAX_BODIES || checkpoint_count < 2) {
        return 12;
    }
    const int component_count = body_count * 3;
    const double direction = checkpoints[checkpoint_count - 1] > checkpoints[0]
        ? 1.0 : -1.0;
    double position[JX_MAX_COMPONENTS];
    double velocity[JX_MAX_COMPONENTS];
    double position_carry[JX_MAX_COMPONENTS] = {0.0};
    double velocity_carry[JX_MAX_COMPONENTS] = {0.0};
    double stage_position[JX_MAX_COMPONENTS];
    double stage_velocity[JX_MAX_COMPONENTS];
    double k_position[JX_STAGE_COUNT][JX_MAX_COMPONENTS];
    double k_velocity[JX_STAGE_COUNT][JX_MAX_COMPONENTS];
    double candidate_position[JX_MAX_COMPONENTS];
    double candidate_velocity[JX_MAX_COMPONENTS];
    double candidate_position_carry[JX_MAX_COMPONENTS];
    double candidate_velocity_carry[JX_MAX_COMPONENTS];
    double acceleration[JX_MAX_COMPONENTS];
    memcpy(position, input_position, (size_t)component_count * sizeof(double));
    memcpy(velocity, input_velocity, (size_t)component_count * sizeof(double));
    memcpy(
        checkpoint_positions,
        input_position,
        (size_t)component_count * sizeof(double));
    memcpy(
        checkpoint_velocities,
        input_velocity,
        (size_t)component_count * sizeof(double));
    checkpoint_accepted[0] = 0;
    checkpoint_rejected[0] = 0;

    double epoch = checkpoints[0];
    double proposed = initial_step;
    double maximum_error = 0.0;
    int64_t attempted = 0;
    int64_t accepted = 0;
    int64_t rejected = 0;

    for (int checkpoint_index = 1;
         checkpoint_index < checkpoint_count;
         ++checkpoint_index) {
        const double checkpoint_epoch = checkpoints[checkpoint_index];
        while (epoch != checkpoint_epoch) {
            if (attempted >= maximum_steps) return 4;
            proposed = fmin(maximum_step, fmax(minimum_step, proposed));
            const double remaining = checkpoint_epoch - epoch;
            if (!(direction * remaining > 0.0) || !isfinite(remaining)) return 9;
            int clipped = proposed >= fabs(remaining);
            double endpoint;
            double step;
            if (clipped) {
                endpoint = checkpoint_epoch;
                step = remaining;
            } else {
                endpoint = epoch + copysign(proposed, direction);
                if (endpoint == epoch || !isfinite(endpoint)) return 9;
                step = endpoint - epoch;
                if (fabs(step) > proposed) {
                    endpoint = nextafter(endpoint, epoch);
                    step = endpoint - epoch;
                }
                if (!(direction * step > 0.0)
                    || fabs(step) > proposed
                    || direction * (checkpoint_epoch - endpoint) <= 0.0) {
                    return 9;
                }
            }

            for (int stage = 0; stage < JX_STAGE_COUNT; ++stage) {
                for (int component = 0; component < component_count; ++component) {
                    if (stage == 0) {
                        stage_position[component] = position[component];
                        stage_velocity[component] = velocity[component];
                    } else {
                        double position_sum = 0.0;
                        double velocity_sum = 0.0;
                        for (int previous = 0; previous < stage; ++previous) {
                            const double coefficient =
                                tableau_a[stage * JX_STAGE_COUNT + previous];
                            if (coefficient != 0.0) {
                                position_sum += coefficient
                                    * k_position[previous][component];
                                velocity_sum += coefficient
                                    * k_velocity[previous][component];
                            }
                        }
                        stage_position[component] =
                            position[component] + step * position_sum;
                        stage_velocity[component] =
                            velocity[component] + step * velocity_sum;
                    }
                    k_position[stage][component] = stage_velocity[component];
                    acceleration[component] = 0.0;
                }

                for (int body = 0; body < body_count; ++body) {
                    for (int source = body + 1; source < body_count; ++source) {
                        const double dx = stage_position[source * 3]
                            - stage_position[body * 3];
                        const double dy = stage_position[source * 3 + 1]
                            - stage_position[body * 3 + 1];
                        const double dz = stage_position[source * 3 + 2]
                            - stage_position[body * 3 + 2];
                        const double distance_squared = dx * dx + dy * dy + dz * dz;
                        if (!(distance_squared > 0.0)
                            || !isfinite(distance_squared)) {
                            return 1;
                        }
                        const double inverse_distance_cubed =
                            1.0 / (distance_squared * sqrt(distance_squared));
                        const double body_weight = gm[source] * inverse_distance_cubed;
                        const double source_weight = gm[body] * inverse_distance_cubed;
                        acceleration[body * 3] += body_weight * dx;
                        acceleration[body * 3 + 1] += body_weight * dy;
                        acceleration[body * 3 + 2] += body_weight * dz;
                        acceleration[source * 3] -= source_weight * dx;
                        acceleration[source * 3 + 1] -= source_weight * dy;
                        acceleration[source * 3 + 2] -= source_weight * dz;
                    }
                }
                for (int component = 0; component < component_count; ++component) {
                    if (!isfinite(acceleration[component])) return 2;
                    k_velocity[stage][component] = acceleration[component];
                }
            }

            double normalized_error = 0.0;
            for (int component = 0; component < component_count; ++component) {
                double eighth_position = 0.0;
                double eighth_velocity = 0.0;
                double defect_position = 0.0;
                double defect_velocity = 0.0;
                for (int stage = 0; stage < JX_STAGE_COUNT; ++stage) {
                    const double eighth = weight_eighth[stage];
                    const double defect = weight_defect[stage];
                    if (eighth != 0.0) {
                        eighth_position += eighth * k_position[stage][component];
                        eighth_velocity += eighth * k_velocity[stage][component];
                    }
                    if (defect != 0.0) {
                        defect_position += defect * k_position[stage][component];
                        defect_velocity += defect * k_velocity[stage][component];
                    }
                }
                const double position_delta = step * eighth_position;
                const double velocity_delta = step * eighth_velocity;
                const double adjusted_position =
                    position_delta - position_carry[component];
                const double adjusted_velocity =
                    velocity_delta - velocity_carry[component];
                candidate_position[component] =
                    position[component] + adjusted_position;
                candidate_velocity[component] =
                    velocity[component] + adjusted_velocity;
                candidate_position_carry[component] =
                    (candidate_position[component] - position[component])
                    - adjusted_position;
                candidate_velocity_carry[component] =
                    (candidate_velocity[component] - velocity[component])
                    - adjusted_velocity;
                if (!isfinite(candidate_position[component])
                    || !isfinite(candidate_velocity[component])) {
                    return 3;
                }
                const double position_scale = position_atol
                    + position_rtol * fmax(
                        fabs(position[component]),
                        fabs(candidate_position[component]));
                const double velocity_scale = velocity_atol
                    + velocity_rtol * fmax(
                        fabs(velocity[component]),
                        fabs(candidate_velocity[component]));
                normalized_error = fmax(
                    normalized_error,
                    fabs(step * defect_position) / position_scale);
                normalized_error = fmax(
                    normalized_error,
                    fabs(step * defect_velocity) / velocity_scale);
            }
            if (!isfinite(normalized_error)) return 5;
            ++attempted;
            maximum_error = fmax(maximum_error, normalized_error);
            double factor = maximum_scale_factor;
            if (normalized_error != 0.0) {
                factor = fmin(
                    maximum_scale_factor,
                    fmax(
                        minimum_scale_factor,
                        safety_factor * pow(normalized_error, -0.125)));
            }
            if (normalized_error <= 1.0) {
                memcpy(
                    position,
                    candidate_position,
                    (size_t)component_count * sizeof(double));
                memcpy(
                    velocity,
                    candidate_velocity,
                    (size_t)component_count * sizeof(double));
                memcpy(
                    position_carry,
                    candidate_position_carry,
                    (size_t)component_count * sizeof(double));
                memcpy(
                    velocity_carry,
                    candidate_velocity_carry,
                    (size_t)component_count * sizeof(double));
                epoch = endpoint;
                if (accepted >= ledger_capacity) return 11;
                accepted_epoch_ledger[accepted] = endpoint;
                accepted_magnitude_ledger[accepted] = fabs(step);
                ++accepted;
                if (!clipped) proposed = fabs(step) * factor;
            } else {
                ++rejected;
                if (rejected > maximum_rejections) return 6;
                if (proposed <= minimum_step
                    || (clipped && fabs(step) < minimum_step)) {
                    return 7;
                }
                const double reduced = fmax(minimum_step, fabs(step) * factor);
                if (!(reduced < fabs(step))) return 8;
                proposed = reduced;
            }
        }
        memcpy(
            checkpoint_positions + checkpoint_index * component_count,
            position,
            (size_t)component_count * sizeof(double));
        memcpy(
            checkpoint_velocities + checkpoint_index * component_count,
            velocity,
            (size_t)component_count * sizeof(double));
        checkpoint_accepted[checkpoint_index] = accepted;
        checkpoint_rejected[checkpoint_index] = rejected;
    }

    *attempted_out = attempted;
    *accepted_out = accepted;
    *rejected_out = rejected;
    *maximum_error_out = maximum_error;
    return 0;
}
