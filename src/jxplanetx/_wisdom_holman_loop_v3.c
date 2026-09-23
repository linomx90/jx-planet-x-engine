/*
 * Checkpoint-synchronized v3 Wisdom--Holman loop.
 *
 * The v2 implementation remains the immutable arithmetic/reference base.  It
 * is included into this translation unit so v3 uses exactly the same guarded
 * transforms, Kepler solver, force assembly, and postconditions.  The only map
 * scheduling change is that the Cartesian transform after the final momentum
 * kick is executed only at retained checkpoints.  That kick cannot change
 * Cartesian positions, so path and node guards use the already synchronized
 * post-drift positions.  Retained checkpoints still receive the same complete
 * position/velocity transform as v2.
 */

#include "_wisdom_holman_loop_v2.c"


static int
jx_wh_jacobi_positions_v3(
    int body_count,
    const double *coordinates,
    const double *gm,
    const double *cumulative,
    double *positions)
{
    double prefix_positions[JX_WH_LOOP_COMPONENTS];
    int component;
    int index;
    for (component = 0; component < 3; ++component) {
        const int final = (body_count - 1) * 3 + component;
        prefix_positions[final] = coordinates[component];
    }
    for (index = body_count - 1; index > 0; --index) {
        const double position_coefficient = gm[index] / cumulative[index];
        for (component = 0; component < 3; ++component) {
            const int current = index * 3 + component;
            const int previous = (index - 1) * 3 + component;
            prefix_positions[previous] = prefix_positions[current]
                - position_coefficient * coordinates[current];
        }
    }
    for (component = 0; component < 3; ++component) {
        positions[component] = prefix_positions[component];
    }
    for (index = 1; index < body_count; ++index) {
        for (component = 0; component < 3; ++component) {
            const int target = index * 3 + component;
            const int prefix = (index - 1) * 3 + component;
            positions[target] = prefix_positions[prefix] + coordinates[target];
        }
    }
    for (index = 0; index < body_count * 3; ++index) {
        if (!isfinite(positions[index])) {
            return JX_WH_NONFINITE;
        }
    }
    return JX_WH_SUCCESS;
}


typedef int (*jx_wh_kepler_step_function)(
    const double *,
    const double *,
    double,
    double,
    double *,
    double *,
    double *,
    int64_t *);


static int
jx_wh_loop_integrate_v3_core(
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
    int64_t *counters,
    jx_wh_kepler_step_function kepler_step)
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
    double next_interaction[JX_WH_LOOP_COMPONENTS];
    double next_momenta[JX_WH_LOOP_COMPONENTS];
    double checkpoint_position[JX_WH_LOOP_COMPONENTS];
    double checkpoint_velocity[JX_WH_LOOP_COMPONENTS];
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
        const int retain_checkpoint = checkpoint_index < checkpoint_count
            && step == checkpoint_steps[checkpoint_index];
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
            status = kepler_step(
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
        status = jx_wh_jacobi_positions_v3(
            body_count,
            drift_coordinates,
            gm,
            cumulative,
            drift_positions);
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
        status = jx_wh_node_guard(
            body_count,
            drift_coordinates,
            next_momenta,
            next_interaction,
            drift_positions,
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

        if (retain_checkpoint) {
            status = jx_wh_jacobi_to_cartesian(
                body_count,
                drift_coordinates,
                next_momenta,
                gm,
                cumulative,
                inertias,
                checkpoint_position,
                checkpoint_velocity);
            if (status != JX_WH_SUCCESS) {
                return status;
            }
            counters[JX_COUNTER_INVERSE_TRANSFORMS] += 1;
            if (memcmp(
                    checkpoint_position,
                    drift_positions,
                    (size_t)component_count * sizeof(double)) != 0) {
                return JX_WH_KEPLER_POSTCONDITION;
            }
            memcpy(
                &checkpoint_positions[checkpoint_index * component_count],
                checkpoint_position,
                (size_t)component_count * sizeof(double));
            memcpy(
                &checkpoint_velocities[checkpoint_index * component_count],
                checkpoint_velocity,
                (size_t)component_count * sizeof(double));
            checkpoint_index += 1;
        }
        memcpy(
            coordinates,
            drift_coordinates,
            (size_t)component_count * sizeof(double));
        memcpy(momenta, next_momenta, (size_t)component_count * sizeof(double));
        memcpy(
            interaction,
            next_interaction,
            (size_t)component_count * sizeof(double));
        memcpy(
            positions,
            drift_positions,
            (size_t)component_count * sizeof(double));
        counters[JX_COUNTER_COMPLETED_STEPS] = step;
    }
    if (checkpoint_index != checkpoint_count) {
        return JX_WH_CHECKPOINT_SCHEDULE;
    }
    if (!isfinite(metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO])) {
        metrics[JX_METRIC_MINIMUM_HILL_FLOOR_RATIO] = 0.0;
    }
    return JX_WH_SUCCESS;
}


int
jx_wh_loop_integrate_v3(
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
        jx_wh_kepler_step);
}
