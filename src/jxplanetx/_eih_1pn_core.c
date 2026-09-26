#include "_eih_1pn_core.h"

#include <math.h>
#include <stddef.h>

#define JX_EIH_1PN_MAX_COMPONENTS (JX_EIH_1PN_MAX_BODIES * 3)

static double
jx_dot3(const double *left, const double *right)
{
    return left[0] * right[0] + left[1] * right[1]
        + left[2] * right[2];
}

int
jx_eih_1pn_total_acceleration(
    int body_count,
    const double *positions,
    const double *velocities,
    const double *gravitational_parameters,
    double speed_of_light,
    double maximum_compactness,
    double maximum_speed_fraction_squared,
    double *accelerations,
    double *observed_maximum_compactness,
    double *observed_maximum_speed_fraction_squared)
{
    double newtonian[JX_EIH_1PN_MAX_COMPONENTS] = {0.0};
    double separations[JX_EIH_1PN_MAX_BODIES][JX_EIH_1PN_MAX_BODIES];
    double potentials[JX_EIH_1PN_MAX_BODIES] = {0.0};
    double c_squared;
    double observed_compactness = 0.0;
    double observed_speed_fraction_squared = 0.0;
    int body;
    int source;
    int component;

    if (positions == NULL || velocities == NULL
        || gravitational_parameters == NULL || accelerations == NULL
        || observed_maximum_compactness == NULL
        || observed_maximum_speed_fraction_squared == NULL
        || body_count < 2 || body_count > JX_EIH_1PN_MAX_BODIES
        || !isfinite(speed_of_light) || speed_of_light <= 0.0
        || !isfinite(maximum_compactness) || maximum_compactness <= 0.0
        || maximum_compactness >= 1.0
        || !isfinite(maximum_speed_fraction_squared)
        || maximum_speed_fraction_squared <= 0.0
        || maximum_speed_fraction_squared >= 1.0) {
        return JX_EIH_1PN_INPUT_DOMAIN;
    }
    c_squared = speed_of_light * speed_of_light;
    if (!isfinite(c_squared)) {
        return JX_EIH_1PN_INPUT_DOMAIN;
    }
    for (body = 0; body < body_count; ++body) {
        if (!isfinite(gravitational_parameters[body])
            || gravitational_parameters[body] <= 0.0) {
            return JX_EIH_1PN_INPUT_DOMAIN;
        }
        for (component = 0; component < 3; ++component) {
            const int index = body * 3 + component;
            if (!isfinite(positions[index]) || !isfinite(velocities[index])) {
                return JX_EIH_1PN_NONFINITE;
            }
        }
    }

    for (body = 0; body < body_count; ++body) {
        separations[body][body] = INFINITY;
        for (source = body + 1; source < body_count; ++source) {
            const double dx = positions[source * 3] - positions[body * 3];
            const double dy = positions[source * 3 + 1]
                - positions[body * 3 + 1];
            const double dz = positions[source * 3 + 2]
                - positions[body * 3 + 2];
            const double radius_squared = dx * dx + dy * dy + dz * dz;
            double radius;
            double inverse_radius_cubed;
            double body_weight;
            double source_weight;
            if (!(radius_squared > 0.0) || !isfinite(radius_squared)) {
                return JX_EIH_1PN_FORCE_SINGULARITY;
            }
            radius = sqrt(radius_squared);
            inverse_radius_cubed = 1.0 / (radius_squared * radius);
            if (!isfinite(inverse_radius_cubed)) {
                return JX_EIH_1PN_NONFINITE;
            }
            separations[body][source] = radius;
            separations[source][body] = radius;
            body_weight = gravitational_parameters[source]
                * inverse_radius_cubed;
            source_weight = gravitational_parameters[body]
                * inverse_radius_cubed;
            newtonian[body * 3] += body_weight * dx;
            newtonian[body * 3 + 1] += body_weight * dy;
            newtonian[body * 3 + 2] += body_weight * dz;
            newtonian[source * 3] -= source_weight * dx;
            newtonian[source * 3 + 1] -= source_weight * dy;
            newtonian[source * 3 + 2] -= source_weight * dz;
        }
    }

    for (body = 0; body < body_count; ++body) {
        const double *velocity = velocities + body * 3;
        double speed_fraction_squared;
        for (source = 0; source < body_count; ++source) {
            if (source != body) {
                potentials[body] += gravitational_parameters[source]
                    / separations[body][source];
            }
        }
        speed_fraction_squared = jx_dot3(velocity, velocity) / c_squared;
        observed_compactness = fmax(
            observed_compactness, potentials[body] / c_squared);
        observed_speed_fraction_squared = fmax(
            observed_speed_fraction_squared, speed_fraction_squared);
    }
    if (!isfinite(observed_compactness)
        || !isfinite(observed_speed_fraction_squared)) {
        return JX_EIH_1PN_NONFINITE;
    }
    if (observed_compactness > maximum_compactness
        || observed_speed_fraction_squared
            > maximum_speed_fraction_squared) {
        return JX_EIH_1PN_WEAK_FIELD_DOMAIN;
    }

    for (component = 0; component < body_count * 3; ++component) {
        accelerations[component] = newtonian[component];
    }
    for (body = 0; body < body_count; ++body) {
        const double *body_position = positions + body * 3;
        const double *body_velocity = velocities + body * 3;
        const double body_speed_squared = jx_dot3(
            body_velocity, body_velocity);
        for (source = 0; source < body_count; ++source) {
            const double *source_position;
            const double *source_velocity;
            const double *source_newtonian;
            double difference[3];
            double velocity_difference[3];
            double velocity_combination[3];
            double radius;
            double radius_squared;
            double radius_cubed;
            double radial_source_velocity;
            double bracket;
            double velocity_bracket;
            if (source == body) {
                continue;
            }
            source_position = positions + source * 3;
            source_velocity = velocities + source * 3;
            source_newtonian = newtonian + source * 3;
            for (component = 0; component < 3; ++component) {
                difference[component] = body_position[component]
                    - source_position[component];
                velocity_difference[component] = body_velocity[component]
                    - source_velocity[component];
                velocity_combination[component] = 4.0 * body_velocity[component]
                    - 3.0 * source_velocity[component];
            }
            radius = separations[body][source];
            radius_squared = radius * radius;
            radius_cubed = radius_squared * radius;
            radial_source_velocity = jx_dot3(
                difference, source_velocity);
            bracket = (
                4.0 * potentials[body]
                + potentials[source]
                - body_speed_squared
                - 2.0 * jx_dot3(source_velocity, source_velocity)
                + 4.0 * jx_dot3(body_velocity, source_velocity)
                + 1.5 * radial_source_velocity * radial_source_velocity
                    / radius_squared
                + 0.5 * jx_dot3(difference, source_newtonian)
            ) / c_squared;
            velocity_bracket = jx_dot3(
                difference, velocity_combination);
            for (component = 0; component < 3; ++component) {
                accelerations[body * 3 + component]
                    += gravitational_parameters[source] * (
                        difference[component] * bracket / radius_cubed
                        + (
                            velocity_bracket
                                * velocity_difference[component]
                                / radius_cubed
                            + 3.5 * source_newtonian[component] / radius
                        ) / c_squared
                    );
            }
        }
    }
    for (component = 0; component < body_count * 3; ++component) {
        if (!isfinite(accelerations[component])) {
            return JX_EIH_1PN_NONFINITE;
        }
    }
    *observed_maximum_compactness = observed_compactness;
    *observed_maximum_speed_fraction_squared
        = observed_speed_fraction_squared;
    return JX_EIH_1PN_SUCCESS;
}
