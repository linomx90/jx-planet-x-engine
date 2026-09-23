/*
 * Safeguarded endpoint-seeded Kepler prototype.
 *
 * The v2 arithmetic, postconditions, and v3 checkpoint-synchronized complete
 * map are included unchanged.  V4 changes only the root-search schedule for
 * each elliptic Kepler subflow.  Any non-success candidate result restores
 * the incoming metrics/counters and executes the unchanged v2 solver.
 */

#include "_wisdom_holman_loop_v3.c"


static _Thread_local int64_t jx_wh_v4_fallback_count = 0;


int64_t
jx_wh_loop_v4_last_fallback_count(void)
{
    return jx_wh_v4_fallback_count;
}


static int
jx_wh_kepler_step_v4_candidate(
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
    double radius_seed;
    double semimajor_seed;
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
    radius_seed = signed_step / orbit.radius;
    semimajor_seed = signed_step / orbit.semimajor_axis;
    if (!isfinite(radial_moment)
        || !isfinite(radius_seed) || radius_seed == 0.0
        || !isfinite(semimajor_seed) || semimajor_seed == 0.0) {
        return JX_WH_KEPLER_ROOT;
    }
    outer = fabs(radius_seed) >= fabs(semimajor_seed)
        ? radius_seed : semimajor_seed;
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
    current = outer;
    for (iteration = 1; iteration <= JX_WH_LOOP_MAX_ITERATIONS; ++iteration) {
        double ulp_scale;
        int adjacent;
        int settled;
        int cycle;
        double midpoint;
        double newton;
        double candidate;
        counters[JX_COUNTER_SOLVER_ITERATIONS] += 1;
        if (iteration == 1) {
            evaluation = outer_evaluation;
        } else {
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
jx_wh_kepler_step_v4(
    const double *input_position,
    const double *input_velocity,
    double gravitational_parameter,
    double signed_step,
    double *output_position,
    double *output_velocity,
    double *metrics,
    int64_t *counters)
{
    double saved_metrics[JX_WH_LOOP_METRIC_COUNT];
    int64_t saved_counters[JX_WH_LOOP_COUNTER_COUNT];
    int status;
    memcpy(saved_metrics, metrics, sizeof(saved_metrics));
    memcpy(saved_counters, counters, sizeof(saved_counters));
    status = jx_wh_kepler_step_v4_candidate(
        input_position,
        input_velocity,
        gravitational_parameter,
        signed_step,
        output_position,
        output_velocity,
        metrics,
        counters);
    if (status == JX_WH_SUCCESS) {
        return status;
    }
    memcpy(metrics, saved_metrics, sizeof(saved_metrics));
    memcpy(counters, saved_counters, sizeof(saved_counters));
    jx_wh_v4_fallback_count += 1;
    return jx_wh_kepler_step(
        input_position,
        input_velocity,
        gravitational_parameter,
        signed_step,
        output_position,
        output_velocity,
        metrics,
        counters);
}


int
jx_wh_loop_integrate_v4(
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
    jx_wh_v4_fallback_count = 0;
    return jx_wh_loop_integrate_v3_core(
        body_count,
        checkpoint_count,
        input_positions,
        input_velocities,
        gm,
        radii,
        checkpoint_steps,
        fixed_step,
        minimum_encounter,
        minimum_periapse,
        maximum_barycenter_position,
        maximum_barycenter_velocity,
        checkpoint_positions,
        checkpoint_velocities,
        metrics,
        counters,
        jx_wh_kepler_step_v4);
}
