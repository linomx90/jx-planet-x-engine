/*
 * Versioned v2 prototype.  The v1 numerical loop remains in
 * _wisdom_holman_loop.c.  This copy changes only universal-G evaluation:
 * it stops the same recurrence once the next decreasing term cannot change
 * the binary64 sum.  The complete fixed-64-term evaluator remains below as
 * an exact fallback and test oracle.
 */

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


#define JX_WH_LOOP_MAX_BODIES 32
#define JX_WH_LOOP_COMPONENTS (JX_WH_LOOP_MAX_BODIES * 3)
#define JX_WH_LOOP_MATRIX (JX_WH_LOOP_MAX_BODIES * JX_WH_LOOP_MAX_BODIES)
#define JX_WH_LOOP_METRIC_COUNT 17
#define JX_WH_LOOP_COUNTER_COUNT 12
#define JX_WH_LOOP_SERIES_TERMS 64
#define JX_WH_LOOP_MAX_ITERATIONS 96
#define JX_WH_LOOP_MAX_BRACKET_EXPANSIONS 32
#define JX_WH_LOOP_PI 3.141592653589793238462643383279502884

enum jx_wh_metric_index {
    JX_METRIC_MINIMUM_PERIAPSE = 0,
    JX_METRIC_MAXIMUM_ECCENTRICITY = 1,
    JX_METRIC_MAXIMUM_INTERACTION_RATIO = 2,
    JX_METRIC_MAXIMUM_ORBIT_FRACTION = 3,
    JX_METRIC_MAXIMUM_PERIAPSE_FRACTION = 4,
    JX_METRIC_MINIMUM_PAIR_ENDPOINT = 5,
    JX_METRIC_MINIMUM_PATH_LOWER_BOUND = 6,
    JX_METRIC_MINIMUM_PATH_CLEARANCE = 7,
    JX_METRIC_MINIMUM_HILL_FLOOR_RATIO = 8,
    JX_METRIC_MAXIMUM_BARYCENTER_POSITION = 9,
    JX_METRIC_MAXIMUM_BARYCENTER_VELOCITY = 10,
    JX_METRIC_MAXIMUM_TRANSLATION_RESIDUAL = 11,
    JX_METRIC_MAXIMUM_KEPLER_TIME_RESIDUAL = 12,
    JX_METRIC_MAXIMUM_KEPLER_TOLERANCE = 13,
    JX_METRIC_MAXIMUM_LAGRANGE_ERROR = 14,
    JX_METRIC_MAXIMUM_ENERGY_ERROR = 15,
    JX_METRIC_MAXIMUM_ANGULAR_ERROR = 16
};

enum jx_wh_counter_index {
    JX_COUNTER_COMPLETED_STEPS = 0,
    JX_COUNTER_FORCE_EVALUATIONS = 1,
    JX_COUNTER_INTERACTION_ASSEMBLIES = 2,
    JX_COUNTER_KEPLER_SOLVES = 3,
    JX_COUNTER_SOLVER_ITERATIONS = 4,
    JX_COUNTER_BRACKET_EXPANSIONS = 5,
    JX_COUNTER_G_EVALUATIONS = 6,
    JX_COUNTER_SERIES_TERMS = 7,
    JX_COUNTER_NODE_GUARDS = 8,
    JX_COUNTER_PATH_GUARDS = 9,
    JX_COUNTER_FORWARD_TRANSFORMS = 10,
    JX_COUNTER_INVERSE_TRANSFORMS = 11
};

enum jx_wh_status {
    JX_WH_SUCCESS = 0,
    JX_WH_INPUT_DOMAIN = 1,
    JX_WH_NONFINITE = 2,
    JX_WH_FORCE_SINGULARITY = 3,
    JX_WH_TRANSLATION_RESIDUAL = 4,
    JX_WH_ORBITAL_GUARD = 5,
    JX_WH_NODE_PAIR_GUARD = 6,
    JX_WH_PATH_GUARD = 7,
    JX_WH_KEPLER_ROOT = 8,
    JX_WH_KEPLER_POSTCONDITION = 9,
    JX_WH_CHECKPOINT_SCHEDULE = 10
};

struct jx_wh_orbit {
    double radius;
    double beta;
    double semimajor_axis;
    double eccentricity;
    double periapse;
    double period;
    double periapse_timescale;
};

struct jx_wh_evaluation {
    double anomaly;
    double residual;
    double radius;
    double tolerance;
    double g[4];
};


static double
jx_wh_max(double left, double right)
{
    return left > right ? left : right;
}


static double
jx_wh_min(double left, double right)
{
    return left < right ? left : right;
}


static double
jx_wh_dot3(const double *left, const double *right)
{
    const double first = left[0] * right[0];
    const double second = left[1] * right[1];
    const double third = left[2] * right[2];
    return (first + second) + third;
}


static void
jx_wh_cross3(const double *left, const double *right, double *result)
{
    result[0] = left[1] * right[2] - left[2] * right[1];
    result[1] = left[2] * right[0] - left[0] * right[2];
    result[2] = left[0] * right[1] - left[1] * right[0];
}


static int
jx_wh_norm3(const double *value, double *result)
{
    const double squared = jx_wh_dot3(value, value);
    if (!isfinite(squared) || squared < 0.0) {
        return JX_WH_NONFINITE;
    }
    *result = sqrt(squared);
    return isfinite(*result) ? JX_WH_SUCCESS : JX_WH_NONFINITE;
}


static double
jx_wh_ulp(double value)
{
    const double magnitude = fabs(value);
    return nextafter(magnitude, INFINITY) - magnitude;
}


static void
jx_wh_initialize_metrics(double *metrics)
{
    int index;
    for (index = 0; index < JX_WH_LOOP_METRIC_COUNT; ++index) {
        metrics[index] = 0.0;
    }
    metrics[JX_METRIC_MINIMUM_PERIAPSE] = INFINITY;
    metrics[JX_METRIC_MINIMUM_PAIR_ENDPOINT] = INFINITY;
    metrics[JX_METRIC_MINIMUM_PATH_LOWER_BOUND] = INFINITY;
    metrics[JX_METRIC_MINIMUM_PATH_CLEARANCE] = INFINITY;
    metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO] = INFINITY;
}


static int
jx_wh_derive_binding(
    int body_count,
    const double *gm,
    double *cumulative,
    double *inertias,
    double *cartesian_from_jacobi)
{
    int index;
    int row;
    int column;
    cumulative[0] = gm[0];
    if (!isfinite(gm[0]) || gm[0] <= 0.0) {
        return JX_WH_INPUT_DOMAIN;
    }
    for (index = 1; index < body_count; ++index) {
        if (!isfinite(gm[index]) || gm[index] <= 0.0) {
            return JX_WH_INPUT_DOMAIN;
        }
        cumulative[index] = cumulative[index - 1] + gm[index];
        if (!isfinite(cumulative[index])
            || !(cumulative[index] > cumulative[index - 1])) {
            return JX_WH_INPUT_DOMAIN;
        }
    }
    inertias[0] = cumulative[body_count - 1];
    for (index = 1; index < body_count; ++index) {
        const double product = gm[index] * cumulative[index - 1];
        inertias[index] = product / cumulative[index];
        if (!isfinite(inertias[index]) || inertias[index] <= 0.0) {
            return JX_WH_INPUT_DOMAIN;
        }
    }
    for (row = 0; row < body_count; ++row) {
        for (column = 0; column < body_count; ++column) {
            cartesian_from_jacobi[row * body_count + column] = 0.0;
        }
        cartesian_from_jacobi[row * body_count] = 1.0;
    }
    for (column = 1; column < body_count; ++column) {
        const double coefficient = -gm[column] / cumulative[column];
        for (row = 0; row < column; ++row) {
            cartesian_from_jacobi[row * body_count + column] = coefficient;
        }
        cartesian_from_jacobi[column * body_count + column] =
            cumulative[column - 1] / cumulative[column];
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_cartesian_to_jacobi(
    int body_count,
    const double *positions,
    const double *velocities,
    const double *gm,
    const double *cumulative,
    const double *inertias,
    double *coordinates,
    double *momenta)
{
    double weighted_position[3];
    double weighted_velocity[3];
    int component;
    int index;
    for (component = 0; component < 3; ++component) {
        weighted_position[component] = gm[0] * positions[component];
        weighted_velocity[component] = gm[0] * velocities[component];
    }
    for (index = 1; index < body_count; ++index) {
        const double prefix = cumulative[index - 1];
        for (component = 0; component < 3; ++component) {
            const int target = index * 3 + component;
            const double prefix_position = weighted_position[component] / prefix;
            const double prefix_velocity = weighted_velocity[component] / prefix;
            coordinates[target] = positions[target] - prefix_position;
            momenta[target] = inertias[index]
                * (velocities[target] - prefix_velocity);
            weighted_position[component] = weighted_position[component]
                + gm[index] * positions[target];
            weighted_velocity[component] = weighted_velocity[component]
                + gm[index] * velocities[target];
        }
    }
    for (component = 0; component < 3; ++component) {
        coordinates[component] = weighted_position[component]
            / cumulative[body_count - 1];
        momenta[component] = weighted_velocity[component];
    }
    for (index = 0; index < body_count * 3; ++index) {
        if (!isfinite(coordinates[index]) || !isfinite(momenta[index])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_jacobi_to_cartesian(
    int body_count,
    const double *coordinates,
    const double *momenta,
    const double *gm,
    const double *cumulative,
    const double *inertias,
    double *positions,
    double *velocities)
{
    double prefix_positions[JX_WH_LOOP_COMPONENTS];
    double prefix_velocities[JX_WH_LOOP_COMPONENTS];
    const double total = cumulative[body_count - 1];
    int component;
    int index;
    for (component = 0; component < 3; ++component) {
        const int final = (body_count - 1) * 3 + component;
        prefix_positions[final] = coordinates[component];
        prefix_velocities[final] = momenta[component] / total;
    }
    for (index = body_count - 1; index > 0; --index) {
        const double position_coefficient = gm[index] / cumulative[index];
        const double velocity_divisor = cumulative[index - 1];
        for (component = 0; component < 3; ++component) {
            const int current = index * 3 + component;
            const int previous = (index - 1) * 3 + component;
            prefix_positions[previous] = prefix_positions[current]
                - position_coefficient * coordinates[current];
            prefix_velocities[previous] = prefix_velocities[current]
                - momenta[current] / velocity_divisor;
        }
    }
    for (component = 0; component < 3; ++component) {
        positions[component] = prefix_positions[component];
        velocities[component] = prefix_velocities[component];
    }
    for (index = 1; index < body_count; ++index) {
        for (component = 0; component < 3; ++component) {
            const int target = index * 3 + component;
            const int prefix = (index - 1) * 3 + component;
            positions[target] = prefix_positions[prefix] + coordinates[target];
            velocities[target] = prefix_velocities[prefix]
                + momenta[target] / inertias[index];
        }
    }
    for (index = 0; index < body_count * 3; ++index) {
        if (!isfinite(positions[index]) || !isfinite(velocities[index])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_gravity(
    int body_count,
    const double *positions,
    const double *gm,
    const double *radii,
    double *acceleration)
{
    int target;
    int source;
    int component;
    for (target = 0; target < body_count; ++target) {
        for (component = 0; component < 3; ++component) {
            acceleration[target * 3 + component] = 0.0;
        }
        for (source = 0; source < body_count; ++source) {
            double dx;
            double dy;
            double dz;
            double distance_squared;
            double collision_distance;
            double inverse_distance_cubed;
            double weight;
            if (source == target) {
                continue;
            }
            dx = positions[source * 3] - positions[target * 3];
            dy = positions[source * 3 + 1] - positions[target * 3 + 1];
            dz = positions[source * 3 + 2] - positions[target * 3 + 2];
            distance_squared = (dx * dx + dy * dy) + dz * dz;
            collision_distance = radii[source] + radii[target];
            if (!isfinite(distance_squared) || distance_squared <= 0.0
                || distance_squared <= collision_distance * collision_distance) {
                return JX_WH_FORCE_SINGULARITY;
            }
            inverse_distance_cubed = 1.0
                / (distance_squared * sqrt(distance_squared));
            weight = gm[source] * inverse_distance_cubed;
            acceleration[target * 3] = acceleration[target * 3] + weight * dx;
            acceleration[target * 3 + 1] = acceleration[target * 3 + 1]
                + weight * dy;
            acceleration[target * 3 + 2] = acceleration[target * 3 + 2]
                + weight * dz;
        }
    }
    for (target = 0; target < body_count * 3; ++target) {
        if (!isfinite(acceleration[target])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_interaction_force(
    int body_count,
    const double *coordinates,
    const double *acceleration,
    const double *gm,
    const double *cumulative,
    const double *inertias,
    const double *cartesian_from_jacobi,
    double *interaction,
    double *translation_residual)
{
    double cartesian_force[JX_WH_LOOP_COMPONENTS];
    double full_force[JX_WH_LOOP_COMPONENTS];
    double force_scale = 0.0;
    double translation_cap;
    int body;
    int coordinate;
    int component;
    int status;
    for (body = 0; body < body_count; ++body) {
        double row_norm;
        for (component = 0; component < 3; ++component) {
            const int slot = body * 3 + component;
            cartesian_force[slot] = gm[body] * acceleration[slot];
        }
        status = jx_wh_norm3(&cartesian_force[body * 3], &row_norm);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        force_scale = jx_wh_max(force_scale, row_norm);
    }
    for (coordinate = 0; coordinate < body_count; ++coordinate) {
        for (component = 0; component < 3; ++component) {
            double total = 0.0;
            for (body = 0; body < body_count; ++body) {
                total = total
                    + cartesian_from_jacobi[body * body_count + coordinate]
                    * cartesian_force[body * 3 + component];
            }
            full_force[coordinate * 3 + component] = total;
            if (!isfinite(total)) {
                return JX_WH_NONFINITE;
            }
        }
    }
    status = jx_wh_norm3(full_force, translation_residual);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    translation_cap = 256.0 * (double)body_count * DBL_EPSILON * force_scale;
    if (*translation_residual > translation_cap) {
        return JX_WH_TRANSLATION_RESIDUAL;
    }
    interaction[0] = 0.0;
    interaction[1] = 0.0;
    interaction[2] = 0.0;
    for (coordinate = 1; coordinate < body_count; ++coordinate) {
        double radius;
        double coefficient;
        status = jx_wh_norm3(&coordinates[coordinate * 3], &radius);
        if (status != JX_WH_SUCCESS || radius <= 0.0) {
            return status == JX_WH_SUCCESS ? JX_WH_FORCE_SINGULARITY : status;
        }
        coefficient = -(inertias[coordinate] * cumulative[coordinate])
            / ((radius * radius) * radius);
        for (component = 0; component < 3; ++component) {
            const int slot = coordinate * 3 + component;
            const double central = coefficient * coordinates[slot];
            interaction[slot] = full_force[slot] - central;
            if (!isfinite(interaction[slot])) {
                return JX_WH_NONFINITE;
            }
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_orbital_elements(
    const double *position,
    const double *velocity,
    double gravitational_parameter,
    struct jx_wh_orbit *orbit)
{
    double speed_squared;
    double radial_dot;
    double eccentricity_vector[3];
    double eccentricity_coefficient;
    int component;
    int status = jx_wh_norm3(position, &orbit->radius);
    if (status != JX_WH_SUCCESS || orbit->radius <= 0.0) {
        return status == JX_WH_SUCCESS ? JX_WH_ORBITAL_GUARD : status;
    }
    speed_squared = jx_wh_dot3(velocity, velocity);
    radial_dot = jx_wh_dot3(position, velocity);
    if (!isfinite(speed_squared) || speed_squared < 0.0 || !isfinite(radial_dot)) {
        return JX_WH_NONFINITE;
    }
    orbit->beta = 2.0 * gravitational_parameter / orbit->radius
        - speed_squared;
    if (!isfinite(orbit->beta) || orbit->beta <= 0.0) {
        return JX_WH_ORBITAL_GUARD;
    }
    orbit->semimajor_axis = gravitational_parameter / orbit->beta;
    eccentricity_coefficient = speed_squared
        - gravitational_parameter / orbit->radius;
    for (component = 0; component < 3; ++component) {
        eccentricity_vector[component] =
            (eccentricity_coefficient * position[component]
             - radial_dot * velocity[component])
            / gravitational_parameter;
    }
    status = jx_wh_norm3(eccentricity_vector, &orbit->eccentricity);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    if (orbit->eccentricity > 0.9) {
        return JX_WH_ORBITAL_GUARD;
    }
    orbit->periapse = orbit->semimajor_axis * (1.0 - orbit->eccentricity);
    orbit->period = 2.0 * JX_WH_LOOP_PI * sqrt(
        (orbit->semimajor_axis * orbit->semimajor_axis
         * orbit->semimajor_axis) / gravitational_parameter);
    orbit->periapse_timescale = 2.0 * JX_WH_LOOP_PI * sqrt(
        ((orbit->semimajor_axis * orbit->semimajor_axis
          * orbit->semimajor_axis) / gravitational_parameter)
        * (((1.0 - orbit->eccentricity)
            * (1.0 - orbit->eccentricity)
            * (1.0 - orbit->eccentricity))
           / (1.0 + orbit->eccentricity)));
    if (!isfinite(orbit->semimajor_axis) || orbit->semimajor_axis <= 0.0
        || !isfinite(orbit->periapse) || orbit->periapse <= 0.0
        || !isfinite(orbit->period) || orbit->period <= 0.0
        || !isfinite(orbit->periapse_timescale)
        || orbit->periapse_timescale <= 0.0) {
        return JX_WH_ORBITAL_GUARD;
    }
    return JX_WH_SUCCESS;
}


static double
jx_wh_hill_radius(
    int left,
    int right,
    const double *semimajor_axes,
    const double *gm)
{
    double average_axis;
    double ratio;
    if (left == 0 || right == 0) {
        return 0.0;
    }
    average_axis = 0.5
        * (semimajor_axes[left - 1] + semimajor_axes[right - 1]);
    ratio = (gm[left] + gm[right]) / (3.0 * gm[0]);
    return average_axis * pow(ratio, 1.0 / 3.0);
}


static int
jx_wh_node_guard(
    int body_count,
    const double *coordinates,
    const double *momenta,
    const double *interaction,
    const double *positions,
    const double *gm,
    const double *radii,
    const double *cumulative,
    const double *inertias,
    double fixed_step,
    double minimum_encounter,
    double minimum_periapse,
    double *semimajor_axes,
    double *eccentricities,
    double *metrics)
{
    int index;
    int left;
    int right;
    int status;
    for (index = 1; index < body_count; ++index) {
        double velocity[3];
        double interaction_norm;
        double acceleration_scale;
        double interaction_acceleration;
        double interaction_ratio;
        double orbit_fraction;
        double periapse_fraction;
        struct jx_wh_orbit orbit;
        int component;
        for (component = 0; component < 3; ++component) {
            velocity[component] = momenta[index * 3 + component]
                / inertias[index];
        }
        status = jx_wh_orbital_elements(
            &coordinates[index * 3],
            velocity,
            cumulative[index],
            &orbit);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        if (orbit.periapse < minimum_periapse) {
            return JX_WH_ORBITAL_GUARD;
        }
        status = jx_wh_norm3(&interaction[index * 3], &interaction_norm);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        acceleration_scale = cumulative[index]
            / (orbit.radius * orbit.radius);
        interaction_acceleration = interaction_norm / inertias[index];
        interaction_ratio = interaction_acceleration / acceleration_scale;
        orbit_fraction = fabs(fixed_step) / orbit.period;
        periapse_fraction = fabs(fixed_step) / orbit.periapse_timescale;
        if (!isfinite(interaction_ratio) || interaction_ratio < 0.0
            || interaction_ratio > 0.1
            || !isfinite(orbit_fraction) || orbit_fraction < 0.0
            || orbit_fraction > 0.05
            || !isfinite(periapse_fraction) || periapse_fraction < 0.0
            || periapse_fraction > 0.0625) {
            return JX_WH_ORBITAL_GUARD;
        }
        semimajor_axes[index - 1] = orbit.semimajor_axis;
        eccentricities[index - 1] = orbit.eccentricity;
        metrics[JX_METRIC_MINIMUM_PERIAPSE] = jx_wh_min(
            metrics[JX_METRIC_MINIMUM_PERIAPSE], orbit.periapse);
        metrics[JX_METRIC_MAXIMUM_ECCENTRICITY] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_ECCENTRICITY], orbit.eccentricity);
        metrics[JX_METRIC_MAXIMUM_INTERACTION_RATIO] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_INTERACTION_RATIO], interaction_ratio);
        metrics[JX_METRIC_MAXIMUM_ORBIT_FRACTION] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_ORBIT_FRACTION], orbit_fraction);
        metrics[JX_METRIC_MAXIMUM_PERIAPSE_FRACTION] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_PERIAPSE_FRACTION], periapse_fraction);
    }
    for (index = 0; index < body_count - 2; ++index) {
        if (!(semimajor_axes[index + 1] > semimajor_axes[index])) {
            return JX_WH_ORBITAL_GUARD;
        }
    }
    for (left = 0; left < body_count - 1; ++left) {
        for (right = left + 1; right < body_count; ++right) {
            double difference[3];
            double separation;
            double hill;
            double hill_floor;
            double floor_value;
            double margin;
            int component;
            for (component = 0; component < 3; ++component) {
                difference[component] = positions[right * 3 + component]
                    - positions[left * 3 + component];
            }
            status = jx_wh_norm3(difference, &separation);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            hill = jx_wh_hill_radius(left, right, semimajor_axes, gm);
            if (!isfinite(hill) || hill < 0.0) {
                return JX_WH_NONFINITE;
            }
            hill_floor = hill > 0.0 ? 3.0 * hill : 0.0;
            floor_value = jx_wh_max(
                radii[left] + radii[right],
                jx_wh_max(minimum_encounter, hill_floor));
            margin = 256.0 * (double)body_count * DBL_EPSILON
                * jx_wh_max(separation, floor_value);
            if (!(separation - margin > floor_value)) {
                return JX_WH_NODE_PAIR_GUARD;
            }
            metrics[JX_METRIC_MINIMUM_PAIR_ENDPOINT] = jx_wh_min(
                metrics[JX_METRIC_MINIMUM_PAIR_ENDPOINT], separation);
            if (hill > 0.0) {
                metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO] = jx_wh_min(
                    metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO],
                    separation / hill_floor);
            }
        }
    }
    {
        double barycenter_position;
        double barycenter_velocity_vector[3];
        double barycenter_velocity;
        int component;
        status = jx_wh_norm3(coordinates, &barycenter_position);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        for (component = 0; component < 3; ++component) {
            barycenter_velocity_vector[component] = momenta[component]
                / cumulative[body_count - 1];
        }
        status = jx_wh_norm3(barycenter_velocity_vector, &barycenter_velocity);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        metrics[JX_METRIC_MAXIMUM_BARYCENTER_POSITION] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_BARYCENTER_POSITION],
            barycenter_position);
        metrics[JX_METRIC_MAXIMUM_BARYCENTER_VELOCITY] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_BARYCENTER_VELOCITY],
            barycenter_velocity);
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_path_guard(
    int body_count,
    const double *start_positions,
    const double *end_positions,
    const double *radii,
    const double *gm,
    const double *cumulative,
    const double *cartesian_from_jacobi,
    const double *semimajor_axes,
    const double *eccentricities,
    double fixed_step,
    double minimum_encounter,
    double *metrics)
{
    double periapse_speeds[JX_WH_LOOP_MAX_BODIES];
    int index;
    int left;
    int right;
    int status;
    for (index = 1; index < body_count; ++index) {
        periapse_speeds[index - 1] = sqrt(
            cumulative[index] * (1.0 + eccentricities[index - 1])
            / (semimajor_axes[index - 1]
               * (1.0 - eccentricities[index - 1])));
        if (!isfinite(periapse_speeds[index - 1])
            || periapse_speeds[index - 1] <= 0.0) {
            return JX_WH_NONFINITE;
        }
    }
    for (left = 0; left < body_count - 1; ++left) {
        for (right = left + 1; right < body_count; ++right) {
            double start_difference[3];
            double end_difference[3];
            double start_distance;
            double end_distance;
            double path_rate_bound = 0.0;
            double path_length_bound;
            double lower_bound;
            double hill;
            double hill_floor;
            double floor_value;
            double margin_scale;
            double margin;
            double minimum_value;
            int component;
            for (component = 0; component < 3; ++component) {
                start_difference[component] =
                    start_positions[right * 3 + component]
                    - start_positions[left * 3 + component];
                end_difference[component] =
                    end_positions[right * 3 + component]
                    - end_positions[left * 3 + component];
            }
            status = jx_wh_norm3(start_difference, &start_distance);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            status = jx_wh_norm3(end_difference, &end_distance);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            for (index = 1; index < body_count; ++index) {
                path_rate_bound = path_rate_bound
                    + fabs(
                        cartesian_from_jacobi[left * body_count + index]
                        - cartesian_from_jacobi[right * body_count + index])
                    * periapse_speeds[index - 1];
            }
            path_length_bound = fabs(fixed_step) * path_rate_bound;
            lower_bound = jx_wh_max(start_distance, end_distance)
                - path_length_bound;
            hill = jx_wh_hill_radius(left, right, semimajor_axes, gm);
            hill_floor = hill > 0.0 ? 3.0 * hill : 0.0;
            floor_value = jx_wh_max(
                radii[left] + radii[right],
                jx_wh_max(minimum_encounter, hill_floor));
            margin_scale = jx_wh_max(
                jx_wh_max(start_distance, end_distance),
                jx_wh_max(path_length_bound, floor_value));
            margin = 256.0 * (double)body_count * DBL_EPSILON * margin_scale;
            minimum_value = jx_wh_min(
                jx_wh_min(start_distance - margin, end_distance - margin),
                lower_bound - margin);
            if (!isfinite(minimum_value) || !(minimum_value > floor_value)) {
                return JX_WH_PATH_GUARD;
            }
            metrics[JX_METRIC_MINIMUM_PAIR_ENDPOINT] = jx_wh_min(
                metrics[JX_METRIC_MINIMUM_PAIR_ENDPOINT],
                jx_wh_min(start_distance, end_distance));
            metrics[JX_METRIC_MINIMUM_PATH_LOWER_BOUND] = jx_wh_min(
                metrics[JX_METRIC_MINIMUM_PATH_LOWER_BOUND], lower_bound);
            metrics[JX_METRIC_MINIMUM_PATH_CLEARANCE] = jx_wh_min(
                metrics[JX_METRIC_MINIMUM_PATH_CLEARANCE],
                minimum_value - floor_value);
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_g_values_exact(
    double beta,
    double anomaly,
    double *values,
    int *used_series)
{
    const double root_beta = sqrt(beta);
    const double argument = root_beta * anomaly;
    int order;
    if (!isfinite(root_beta) || !isfinite(argument)) {
        return JX_WH_NONFINITE;
    }
    if (fabs(argument) <= 0.5) {
        *used_series = 1;
        for (order = 0; order < 4; ++order) {
            double term;
            double total;
            double rho;
            int series_index;
            if (order == 0) {
                term = 1.0;
            } else if (order == 1) {
                term = anomaly;
            } else if (order == 2) {
                term = (anomaly * anomaly) / 2.0;
            } else {
                term = ((anomaly * anomaly) * anomaly) / 6.0;
            }
            rho = ((-beta) * anomaly) * anomaly;
            total = term;
            for (series_index = 0;
                 series_index < JX_WH_LOOP_SERIES_TERMS - 1;
                 ++series_index) {
                const int first = order + 2 * series_index + 1;
                const int second = first + 1;
                term = (term * rho) / (double)(first * second);
                total = total + term;
                if (!isfinite(term) || !isfinite(total)) {
                    return JX_WH_NONFINITE;
                }
            }
            values[order] = total;
        }
    } else {
        const double cosine = cos(argument);
        const double sine = sin(argument);
        *used_series = 0;
        values[0] = cosine;
        values[1] = sine / root_beta;
        values[2] = (1.0 - cosine) / beta;
        values[3] = (anomaly - values[1]) / beta;
    }
    for (order = 0; order < 4; ++order) {
        if (!isfinite(values[order])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_g_values(
    double beta,
    double anomaly,
    double *values,
    int *used_series,
    int *series_terms)
{
    const double root_beta = sqrt(beta);
    const double argument = root_beta * anomaly;
    int order;
    *series_terms = 0;
    if (!isfinite(root_beta) || !isfinite(argument)) {
        return JX_WH_NONFINITE;
    }
    if (fabs(argument) > 0.5) {
        return jx_wh_g_values_exact(beta, anomaly, values, used_series);
    }
    *used_series = 1;
    for (order = 0; order < 4; ++order) {
        double term;
        double total;
        double rho;
        int evaluated_terms = 1;
        int series_index;
        if (order == 0) {
            term = 1.0;
        } else if (order == 1) {
            term = anomaly;
        } else if (order == 2) {
            term = (anomaly * anomaly) / 2.0;
        } else {
            term = ((anomaly * anomaly) * anomaly) / 6.0;
        }
        rho = ((-beta) * anomaly) * anomaly;
        if (!isfinite(term) || !isfinite(rho) || fabs(rho) > 0.25) {
            *series_terms = 4 * JX_WH_LOOP_SERIES_TERMS;
            return jx_wh_g_values_exact(beta, anomaly, values, used_series);
        }
        total = term;
        for (series_index = 0;
             series_index < JX_WH_LOOP_SERIES_TERMS - 1;
             ++series_index) {
            const int first = order + 2 * series_index + 1;
            const int second = first + 1;
            const double previous_total = total;
            term = (term * rho) / (double)(first * second);
            total = total + term;
            evaluated_terms += 1;
            if (!isfinite(term) || !isfinite(total)) {
                *series_terms = 4 * JX_WH_LOOP_SERIES_TERMS;
                return jx_wh_g_values_exact(
                    beta, anomaly, values, used_series);
            }
            if (total == previous_total) {
                break;
            }
        }
        values[order] = total;
        *series_terms += evaluated_terms;
    }
    for (order = 0; order < 4; ++order) {
        if (!isfinite(values[order])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


int
jx_wh_loop_v2_g_bundle(
    int exact,
    double beta,
    double anomaly,
    double *values,
    int *used_series,
    int *series_terms)
{
    if ((exact != 0 && exact != 1)
        || !isfinite(beta) || beta <= 0.0 || !isfinite(anomaly)) {
        return JX_WH_INPUT_DOMAIN;
    }
    if (exact) {
        const int status = jx_wh_g_values_exact(
            beta, anomaly, values, used_series);
        *series_terms = *used_series ? 4 * JX_WH_LOOP_SERIES_TERMS : 0;
        return status;
    }
    return jx_wh_g_values(
        beta, anomaly, values, used_series, series_terms);
}


static int
jx_wh_universal_evaluation(
    double anomaly,
    double beta,
    double radius_initial,
    double radial_moment,
    double gravitational_parameter,
    double signed_step,
    struct jx_wh_evaluation *evaluation,
    int64_t *counters)
{
    const double coefficient = gravitational_parameter
        - beta * radius_initial;
    double first;
    double second;
    double third;
    double scale;
    int used_series = 0;
    int series_terms = 0;
    int status = jx_wh_g_values(
        beta, anomaly, evaluation->g, &used_series, &series_terms);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    counters[JX_COUNTER_G_EVALUATIONS] += 1;
    if (used_series) {
        counters[JX_COUNTER_SERIES_TERMS] += series_terms;
    }
    first = radius_initial * anomaly;
    second = radial_moment * evaluation->g[2];
    third = coefficient * evaluation->g[3];
    evaluation->anomaly = anomaly;
    evaluation->residual = ((first + second) + third) - signed_step;
    evaluation->radius = (radius_initial
        + radial_moment * evaluation->g[1])
        + coefficient * evaluation->g[2];
    scale = fabs(first) + fabs(second) + fabs(third) + fabs(signed_step);
    evaluation->tolerance = jx_wh_max(
        8.0 * jx_wh_ulp(fabs(signed_step)),
        32.0 * DBL_EPSILON * scale);
    if (!isfinite(evaluation->residual)
        || !isfinite(evaluation->radius)
        || !isfinite(evaluation->tolerance)
        || evaluation->radius <= 0.0) {
        return JX_WH_NONFINITE;
    }
    return JX_WH_SUCCESS;
}


static int
jx_wh_kepler_step(
    const double *input_position,
    const double *input_velocity,
    double gravitational_parameter,
    double signed_step,
    double *output_position,
    double *output_velocity,
    double *metrics,
    int64_t *counters)
{
    struct jx_wh_orbit orbit;
    struct jx_wh_evaluation outer_evaluation;
    struct jx_wh_evaluation evaluation;
    struct jx_wh_evaluation accepted;
    double radial_moment;
    double seed;
    double outer;
    double low;
    double high;
    double residual_low;
    double residual_high;
    double current;
    double previous = 0.0;
    double two_iterations_ago = 0.0;
    double final_width = INFINITY;
    int have_previous = 0;
    int have_two_iterations_ago = 0;
    int bracket_expansions = 0;
    int accepted_iteration = 0;
    int iteration;
    int component;
    int status = jx_wh_orbital_elements(
        input_position,
        input_velocity,
        gravitational_parameter,
        &orbit);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    radial_moment = jx_wh_dot3(input_position, input_velocity);
    seed = signed_step / orbit.radius;
    if (!isfinite(radial_moment) || !isfinite(seed) || seed == 0.0) {
        return JX_WH_KEPLER_ROOT;
    }
    outer = seed;
    status = jx_wh_universal_evaluation(
        outer,
        orbit.beta,
        orbit.radius,
        radial_moment,
        gravitational_parameter,
        signed_step,
        &outer_evaluation,
        counters);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    while ((signed_step > 0.0 && outer_evaluation.residual < 0.0)
           || (signed_step < 0.0 && outer_evaluation.residual > 0.0)) {
        if (bracket_expansions >= JX_WH_LOOP_MAX_BRACKET_EXPANSIONS) {
            return JX_WH_KEPLER_ROOT;
        }
        outer = outer * 2.0;
        if (!isfinite(outer)) {
            return JX_WH_KEPLER_ROOT;
        }
        bracket_expansions += 1;
        counters[JX_COUNTER_BRACKET_EXPANSIONS] += 1;
        status = jx_wh_universal_evaluation(
            outer,
            orbit.beta,
            orbit.radius,
            radial_moment,
            gravitational_parameter,
            signed_step,
            &outer_evaluation,
            counters);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
    }
    if (signed_step > 0.0) {
        low = 0.0;
        residual_low = -signed_step;
        high = outer;
        residual_high = outer_evaluation.residual;
    } else {
        low = outer;
        residual_low = outer_evaluation.residual;
        high = 0.0;
        residual_high = -signed_step;
    }
    if (!(low < high) || !(residual_low <= 0.0)
        || !(residual_high >= 0.0)) {
        return JX_WH_KEPLER_ROOT;
    }
    current = outer_evaluation.residual == 0.0
        ? outer : low + 0.5 * (high - low);
    if (!isfinite(current) || current < low || current > high) {
        return JX_WH_KEPLER_ROOT;
    }
    for (iteration = 1; iteration <= JX_WH_LOOP_MAX_ITERATIONS; ++iteration) {
        double ulp_scale;
        int adjacent;
        int settled;
        int cycle;
        double midpoint;
        double newton;
        double candidate;
        counters[JX_COUNTER_SOLVER_ITERATIONS] += 1;
        status = jx_wh_universal_evaluation(
            current,
            orbit.beta,
            orbit.radius,
            radial_moment,
            gravitational_parameter,
            signed_step,
            &evaluation,
            counters);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        if (evaluation.residual == 0.0) {
            low = current;
            high = current;
            residual_low = 0.0;
            residual_high = 0.0;
        } else if (evaluation.residual < 0.0) {
            low = current;
            residual_low = evaluation.residual;
        } else {
            high = current;
            residual_high = evaluation.residual;
        }
        if (!(residual_low <= 0.0) || !(residual_high >= 0.0)
            || !(low <= high)) {
            return JX_WH_KEPLER_ROOT;
        }
        final_width = high - low;
        ulp_scale = jx_wh_max(
            jx_wh_ulp(fabs(low)),
            jx_wh_max(jx_wh_ulp(fabs(high)), jx_wh_ulp(fabs(current))));
        adjacent = low == high || nextafter(low, high) == high;
        settled = adjacent || final_width <= 8.0 * ulp_scale;
        cycle = (have_previous && current == previous)
            || (have_two_iterations_ago && current == two_iterations_ago);
        if (fabs(evaluation.residual) <= evaluation.tolerance
            && (settled || cycle)) {
            accepted = evaluation;
            accepted_iteration = iteration;
            break;
        }
        midpoint = low + 0.5 * (high - low);
        newton = current - evaluation.residual / evaluation.radius;
        candidate = isfinite(newton) && low < newton && newton < high
            ? newton : midpoint;
        if (candidate == current) {
            candidate = midpoint;
        }
        if (!isfinite(candidate) || candidate < low || candidate > high
            || candidate == current) {
            return JX_WH_KEPLER_ROOT;
        }
        if (have_previous) {
            two_iterations_ago = previous;
            have_two_iterations_ago = 1;
        }
        previous = current;
        have_previous = 1;
        current = candidate;
    }
    if (accepted_iteration == 0) {
        return JX_WH_KEPLER_ROOT;
    }
    {
        const double f_value = 1.0
            - (gravitational_parameter / orbit.radius) * accepted.g[2];
        const double g_value = signed_step
            - gravitational_parameter * accepted.g[3];
        double radius_final;
        double fdot;
        double gdot;
        double lagrange_error;
        double lagrange_scale;
        double energy_initial;
        double energy_final;
        double energy_error;
        double energy_scale;
        double angular_initial[3];
        double angular_final[3];
        double angular_difference[3];
        double angular_error;
        double angular_scale;
        double velocity_initial_norm;
        double velocity_final_norm;
        const double speed_initial_squared = jx_wh_dot3(
            input_velocity, input_velocity);
        double speed_final_squared;
        for (component = 0; component < 3; ++component) {
            output_position[component] = f_value * input_position[component]
                + g_value * input_velocity[component];
        }
        status = jx_wh_norm3(output_position, &radius_final);
        if (status != JX_WH_SUCCESS || radius_final <= 0.0) {
            return JX_WH_KEPLER_POSTCONDITION;
        }
        fdot = -(gravitational_parameter / (orbit.radius * radius_final))
            * accepted.g[1];
        gdot = 1.0
            - (gravitational_parameter / radius_final) * accepted.g[2];
        for (component = 0; component < 3; ++component) {
            output_velocity[component] = fdot * input_position[component]
                + gdot * input_velocity[component];
            if (!isfinite(output_position[component])
                || !isfinite(output_velocity[component])) {
                return JX_WH_KEPLER_POSTCONDITION;
            }
        }
        lagrange_error = fabs(f_value * gdot - fdot * g_value - 1.0);
        lagrange_scale = jx_wh_max(
            1.0, fabs(f_value * gdot) + fabs(fdot * g_value));
        if (!isfinite(lagrange_error)
            || lagrange_error > 512.0 * DBL_EPSILON * lagrange_scale) {
            return JX_WH_KEPLER_POSTCONDITION;
        }
        speed_final_squared = jx_wh_dot3(output_velocity, output_velocity);
        energy_initial = 0.5 * speed_initial_squared
            - gravitational_parameter / orbit.radius;
        energy_final = 0.5 * speed_final_squared
            - gravitational_parameter / radius_final;
        energy_error = fabs(energy_final - energy_initial);
        energy_scale = jx_wh_max(
            jx_wh_max(
                gravitational_parameter / orbit.radius,
                gravitational_parameter / radius_final),
            jx_wh_max(0.5 * speed_initial_squared, 0.5 * speed_final_squared));
        if (!isfinite(energy_error)
            || energy_error > 1024.0 * DBL_EPSILON * energy_scale) {
            return JX_WH_KEPLER_POSTCONDITION;
        }
        jx_wh_cross3(input_position, input_velocity, angular_initial);
        jx_wh_cross3(output_position, output_velocity, angular_final);
        for (component = 0; component < 3; ++component) {
            angular_difference[component] = angular_final[component]
                - angular_initial[component];
        }
        status = jx_wh_norm3(angular_difference, &angular_error);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        status = jx_wh_norm3(input_velocity, &velocity_initial_norm);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        status = jx_wh_norm3(output_velocity, &velocity_final_norm);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        angular_scale = jx_wh_max(
            orbit.radius * velocity_initial_norm,
            radius_final * velocity_final_norm);
        if (!isfinite(angular_error)
            || angular_error > 1024.0 * DBL_EPSILON * angular_scale) {
            return JX_WH_KEPLER_POSTCONDITION;
        }
        metrics[JX_METRIC_MAXIMUM_KEPLER_TIME_RESIDUAL] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_KEPLER_TIME_RESIDUAL],
            fabs(accepted.residual));
        metrics[JX_METRIC_MAXIMUM_KEPLER_TOLERANCE] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_KEPLER_TOLERANCE], accepted.tolerance);
        metrics[JX_METRIC_MAXIMUM_LAGRANGE_ERROR] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_LAGRANGE_ERROR], lagrange_error);
        metrics[JX_METRIC_MAXIMUM_ENERGY_ERROR] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_ENERGY_ERROR], energy_error);
        metrics[JX_METRIC_MAXIMUM_ANGULAR_ERROR] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_ANGULAR_ERROR], angular_error);
    }
    counters[JX_COUNTER_KEPLER_SOLVES] += 1;
    (void)bracket_expansions;
    (void)final_width;
    return JX_WH_SUCCESS;
}


static int
jx_wh_initial_barycenter_guard(
    int body_count,
    const double *positions,
    const double *velocities,
    const double *coordinates,
    const double *momenta,
    const double *cumulative,
    double maximum_position,
    double maximum_velocity)
{
    double position_scale = 0.0;
    double velocity_scale = 0.0;
    double barycenter_position;
    double barycenter_velocity_vector[3];
    double barycenter_velocity;
    double derived_position_cap;
    double derived_velocity_cap;
    int left;
    int right;
    int component;
    int status;
    for (left = 0; left < body_count - 1; ++left) {
        for (right = left + 1; right < body_count; ++right) {
            double position_difference[3];
            double velocity_difference[3];
            double position_norm;
            double velocity_norm;
            for (component = 0; component < 3; ++component) {
                position_difference[component] = positions[right * 3 + component]
                    - positions[left * 3 + component];
                velocity_difference[component] = velocities[right * 3 + component]
                    - velocities[left * 3 + component];
            }
            status = jx_wh_norm3(position_difference, &position_norm);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            status = jx_wh_norm3(velocity_difference, &velocity_norm);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            position_scale = jx_wh_max(position_scale, position_norm);
            velocity_scale = jx_wh_max(velocity_scale, velocity_norm);
        }
    }
    status = jx_wh_norm3(coordinates, &barycenter_position);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    for (component = 0; component < 3; ++component) {
        barycenter_velocity_vector[component] = momenta[component]
            / cumulative[body_count - 1];
    }
    status = jx_wh_norm3(barycenter_velocity_vector, &barycenter_velocity);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    derived_position_cap = 256.0 * (double)body_count * DBL_EPSILON
        * position_scale;
    derived_velocity_cap = 256.0 * (double)body_count * DBL_EPSILON
        * velocity_scale;
    if (barycenter_position > jx_wh_min(maximum_position, derived_position_cap)
        || barycenter_velocity > jx_wh_min(
            maximum_velocity, derived_velocity_cap)) {
        return JX_WH_ORBITAL_GUARD;
    }
    return JX_WH_SUCCESS;
}


int
jx_wh_loop_integrate_v2(
    int body_count,
    int checkpoint_count,
    const double *input_positions,
    const double *input_velocities,
    const double *gm,
    const double *radii,
    const int64_t *checkpoint_steps,
    double fixed_step,
    double minimum_encounter,
    double minimum_periapse,
    double maximum_barycenter_position,
    double maximum_barycenter_velocity,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    double *metrics,
    int64_t *counters)
{
    double cumulative[JX_WH_LOOP_MAX_BODIES];
    double inertias[JX_WH_LOOP_MAX_BODIES];
    double cartesian_from_jacobi[JX_WH_LOOP_MATRIX];
    double positions[JX_WH_LOOP_COMPONENTS];
    double velocities[JX_WH_LOOP_COMPONENTS];
    double coordinates[JX_WH_LOOP_COMPONENTS];
    double momenta[JX_WH_LOOP_COMPONENTS];
    double interaction[JX_WH_LOOP_COMPONENTS];
    double acceleration[JX_WH_LOOP_COMPONENTS];
    double half_momenta[JX_WH_LOOP_COMPONENTS];
    double drift_coordinates[JX_WH_LOOP_COMPONENTS];
    double drift_momenta[JX_WH_LOOP_COMPONENTS];
    double drift_positions[JX_WH_LOOP_COMPONENTS];
    double drift_velocities[JX_WH_LOOP_COMPONENTS];
    double next_interaction[JX_WH_LOOP_COMPONENTS];
    double next_momenta[JX_WH_LOOP_COMPONENTS];
    double next_positions[JX_WH_LOOP_COMPONENTS];
    double next_velocities[JX_WH_LOOP_COMPONENTS];
    double semimajor_axes[JX_WH_LOOP_MAX_BODIES];
    double eccentricities[JX_WH_LOOP_MAX_BODIES];
    const int component_count = body_count * 3;
    double secondary_ratio = 0.0;
    double translation_residual;
    int64_t step;
    int checkpoint_index = 1;
    int index;
    int component;
    int status;

    if (body_count < 2 || body_count > JX_WH_LOOP_MAX_BODIES
        || checkpoint_count < 2 || checkpoint_steps[0] != 0
        || !isfinite(fixed_step) || fixed_step == 0.0
        || !isfinite(minimum_encounter) || minimum_encounter <= 0.0
        || !isfinite(minimum_periapse) || minimum_periapse <= 0.0
        || !isfinite(maximum_barycenter_position)
        || maximum_barycenter_position <= 0.0
        || !isfinite(maximum_barycenter_velocity)
        || maximum_barycenter_velocity <= 0.0) {
        return JX_WH_INPUT_DOMAIN;
    }
    for (index = 1; index < checkpoint_count; ++index) {
        if (checkpoint_steps[index] <= checkpoint_steps[index - 1]) {
            return JX_WH_CHECKPOINT_SCHEDULE;
        }
    }
    status = jx_wh_derive_binding(
        body_count, gm, cumulative, inertias, cartesian_from_jacobi);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    for (index = 0; index < body_count; ++index) {
        if (!isfinite(radii[index]) || radii[index] < 0.0) {
            return JX_WH_INPUT_DOMAIN;
        }
        if (index > 0) {
            secondary_ratio = secondary_ratio + gm[index];
        }
    }
    if (!isfinite(secondary_ratio) || secondary_ratio / gm[0] > 0.01) {
        return JX_WH_ORBITAL_GUARD;
    }
    memset(counters, 0, JX_WH_LOOP_COUNTER_COUNT * sizeof(int64_t));
    jx_wh_initialize_metrics(metrics);
    memcpy(positions, input_positions, (size_t)component_count * sizeof(double));
    memcpy(velocities, input_velocities, (size_t)component_count * sizeof(double));
    memcpy(
        checkpoint_positions,
        input_positions,
        (size_t)component_count * sizeof(double));
    memcpy(
        checkpoint_velocities,
        input_velocities,
        (size_t)component_count * sizeof(double));
    status = jx_wh_cartesian_to_jacobi(
        body_count,
        positions,
        velocities,
        gm,
        cumulative,
        inertias,
        coordinates,
        momenta);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    counters[JX_COUNTER_FORWARD_TRANSFORMS] = 1;
    status = jx_wh_initial_barycenter_guard(
        body_count,
        positions,
        velocities,
        coordinates,
        momenta,
        cumulative,
        maximum_barycenter_position,
        maximum_barycenter_velocity);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    status = jx_wh_gravity(body_count, positions, gm, radii, acceleration);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    counters[JX_COUNTER_FORCE_EVALUATIONS] = 1;
    status = jx_wh_interaction_force(
        body_count,
        coordinates,
        acceleration,
        gm,
        cumulative,
        inertias,
        cartesian_from_jacobi,
        interaction,
        &translation_residual);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    counters[JX_COUNTER_INTERACTION_ASSEMBLIES] = 1;
    metrics[JX_METRIC_MAXIMUM_TRANSLATION_RESIDUAL] = translation_residual;
    status = jx_wh_node_guard(
        body_count,
        coordinates,
        momenta,
        interaction,
        positions,
        gm,
        radii,
        cumulative,
        inertias,
        fixed_step,
        minimum_encounter,
        minimum_periapse,
        semimajor_axes,
        eccentricities,
        metrics);
    if (status != JX_WH_SUCCESS) {
        return status;
    }
    counters[JX_COUNTER_NODE_GUARDS] = 1;

    for (step = 1; step <= checkpoint_steps[checkpoint_count - 1]; ++step) {
        for (index = 0; index < body_count; ++index) {
            for (component = 0; component < 3; ++component) {
                const int slot = index * 3 + component;
                half_momenta[slot] = momenta[slot]
                    + 0.5 * fixed_step * interaction[slot];
                if (!isfinite(half_momenta[slot])) {
                    return JX_WH_NONFINITE;
                }
            }
        }
        status = jx_wh_node_guard(
            body_count,
            coordinates,
            half_momenta,
            interaction,
            positions,
            gm,
            radii,
            cumulative,
            inertias,
            fixed_step,
            minimum_encounter,
            minimum_periapse,
            semimajor_axes,
            eccentricities,
            metrics);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_NODE_GUARDS] += 1;
        memcpy(
            drift_coordinates,
            coordinates,
            (size_t)component_count * sizeof(double));
        memcpy(
            drift_momenta,
            half_momenta,
            (size_t)component_count * sizeof(double));
        for (component = 0; component < 3; ++component) {
            drift_coordinates[component] = coordinates[component]
                + fixed_step * half_momenta[component]
                / cumulative[body_count - 1];
        }
        for (index = 1; index < body_count; ++index) {
            double relative_velocity[3];
            double output_position[3];
            double output_velocity[3];
            for (component = 0; component < 3; ++component) {
                relative_velocity[component] = half_momenta[index * 3 + component]
                    / inertias[index];
            }
            status = jx_wh_kepler_step(
                &coordinates[index * 3],
                relative_velocity,
                cumulative[index],
                fixed_step,
                output_position,
                output_velocity,
                metrics,
                counters);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            for (component = 0; component < 3; ++component) {
                drift_coordinates[index * 3 + component] =
                    output_position[component];
                drift_momenta[index * 3 + component] = inertias[index]
                    * output_velocity[component];
            }
        }
        status = jx_wh_jacobi_to_cartesian(
            body_count,
            drift_coordinates,
            drift_momenta,
            gm,
            cumulative,
            inertias,
            drift_positions,
            drift_velocities);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_INVERSE_TRANSFORMS] += 1;
        status = jx_wh_path_guard(
            body_count,
            positions,
            drift_positions,
            radii,
            gm,
            cumulative,
            cartesian_from_jacobi,
            semimajor_axes,
            eccentricities,
            fixed_step,
            minimum_encounter,
            metrics);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_PATH_GUARDS] += 1;
        status = jx_wh_gravity(
            body_count, drift_positions, gm, radii, acceleration);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_FORCE_EVALUATIONS] += 1;
        status = jx_wh_interaction_force(
            body_count,
            drift_coordinates,
            acceleration,
            gm,
            cumulative,
            inertias,
            cartesian_from_jacobi,
            next_interaction,
            &translation_residual);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_INTERACTION_ASSEMBLIES] += 1;
        metrics[JX_METRIC_MAXIMUM_TRANSLATION_RESIDUAL] = jx_wh_max(
            metrics[JX_METRIC_MAXIMUM_TRANSLATION_RESIDUAL],
            translation_residual);
        for (index = 0; index < body_count; ++index) {
            for (component = 0; component < 3; ++component) {
                const int slot = index * 3 + component;
                next_momenta[slot] = drift_momenta[slot]
                    + 0.5 * fixed_step * next_interaction[slot];
                if (!isfinite(next_momenta[slot])) {
                    return JX_WH_NONFINITE;
                }
            }
        }
        status = jx_wh_jacobi_to_cartesian(
            body_count,
            drift_coordinates,
            next_momenta,
            gm,
            cumulative,
            inertias,
            next_positions,
            next_velocities);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_INVERSE_TRANSFORMS] += 1;
        status = jx_wh_node_guard(
            body_count,
            drift_coordinates,
            next_momenta,
            next_interaction,
            next_positions,
            gm,
            radii,
            cumulative,
            inertias,
            fixed_step,
            minimum_encounter,
            minimum_periapse,
            semimajor_axes,
            eccentricities,
            metrics);
        if (status != JX_WH_SUCCESS) {
            return status;
        }
        counters[JX_COUNTER_NODE_GUARDS] += 1;
        memcpy(
            coordinates,
            drift_coordinates,
            (size_t)component_count * sizeof(double));
        memcpy(momenta, next_momenta, (size_t)component_count * sizeof(double));
        memcpy(
            interaction,
            next_interaction,
            (size_t)component_count * sizeof(double));
        memcpy(positions, next_positions, (size_t)component_count * sizeof(double));
        memcpy(velocities, next_velocities, (size_t)component_count * sizeof(double));
        counters[JX_COUNTER_COMPLETED_STEPS] = step;
        if (checkpoint_index < checkpoint_count
            && step == checkpoint_steps[checkpoint_index]) {
            memcpy(
                &checkpoint_positions[checkpoint_index * component_count],
                positions,
                (size_t)component_count * sizeof(double));
            memcpy(
                &checkpoint_velocities[checkpoint_index * component_count],
                velocities,
                (size_t)component_count * sizeof(double));
            checkpoint_index += 1;
        }
    }
    if (checkpoint_index != checkpoint_count) {
        return JX_WH_CHECKPOINT_SCHEDULE;
    }
    if (!isfinite(metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO])) {
        metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO] = 0.0;
    }
    return JX_WH_SUCCESS;
}
