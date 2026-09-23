#include "_gr15_core.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#define JX_GR15_MAX_COMPONENTS (JX_GR15_MAX_BODIES * 3)

/* Eight-node left Gauss--Radau coefficients; formal collocation order 15. */
static const double jx_gr15_nodes[JX_GR15_STAGE_COUNT] = {
    0.0,
    0.05626256053692214646565219,
    0.1802406917368923649875799,
    0.3526247171131696373739078,
    0.5471536263305553830014486,
    0.7342101772154105315232106,
    0.8853209468390957680903598,
    0.9775206135612875018911745
};

static const double jx_gr15_weights[JX_GR15_STAGE_COUNT] = {
    0.015625,
    0.09267907740148963927036449,
    0.1520653103233925644878716,
    0.1882587726945592782860646,
    0.1957860837262467965412498,
    0.173507397817250640114338,
    0.1248239506649324816289346,
    0.05725440737212859967117686
};

static const double jx_gr15_matrix[JX_GR15_STAGE_COUNT]
    [JX_GR15_STAGE_COUNT] = {
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

/* Inverse Vandermonde: power coefficient k from acceleration sample j. */
static const double jx_gr15_power_from_stage[JX_GR15_STAGE_COUNT]
    [JX_GR15_STAGE_COUNT] = {
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

int
jx_gr15_tableau(double *nodes, double *weights, double *matrix)
{
    if (nodes == NULL || weights == NULL || matrix == NULL) {
        return JX_GR15_INPUT_DOMAIN;
    }
    memcpy(nodes, jx_gr15_nodes, sizeof(jx_gr15_nodes));
    memcpy(weights, jx_gr15_weights, sizeof(jx_gr15_weights));
    memcpy(matrix, jx_gr15_matrix, sizeof(jx_gr15_matrix));
    return JX_GR15_SUCCESS;
}

static int
jx_gr15_acceleration(
    int body_count,
    const double *position,
    const double *gm,
    double *acceleration)
{
    const int component_count = body_count * 3;
    int component;
    int body;
    int source;
    for (component = 0; component < component_count; ++component) {
        acceleration[component] = 0.0;
    }
    for (body = 0; body < body_count; ++body) {
        for (source = body + 1; source < body_count; ++source) {
            const double dx = position[source * 3] - position[body * 3];
            const double dy = position[source * 3 + 1] - position[body * 3 + 1];
            const double dz = position[source * 3 + 2] - position[body * 3 + 2];
            const double distance_squared = dx * dx + dy * dy + dz * dz;
            double inverse_distance_cubed;
            double body_weight;
            double source_weight;
            if (!(distance_squared > 0.0) || !isfinite(distance_squared)) {
                return JX_GR15_FORCE_SINGULARITY;
            }
            inverse_distance_cubed =
                1.0 / (distance_squared * sqrt(distance_squared));
            body_weight = gm[source] * inverse_distance_cubed;
            source_weight = gm[body] * inverse_distance_cubed;
            acceleration[body * 3] += body_weight * dx;
            acceleration[body * 3 + 1] += body_weight * dy;
            acceleration[body * 3 + 2] += body_weight * dz;
            acceleration[source * 3] -= source_weight * dx;
            acceleration[source * 3 + 1] -= source_weight * dy;
            acceleration[source * 3 + 2] -= source_weight * dz;
        }
    }
    for (component = 0; component < component_count; ++component) {
        if (!isfinite(acceleration[component])) {
            return JX_GR15_NONFINITE;
        }
    }
    return JX_GR15_SUCCESS;
}

static int
jx_gr15_predict(
    int component_count,
    const double *position,
    const double *velocity,
    const double *initial_acceleration,
    double step,
    int history_valid,
    double previous_step,
    double previous_stage_acceleration[JX_GR15_STAGE_COUNT]
        [JX_GR15_MAX_COMPONENTS],
    double stage_position[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS],
    double stage_velocity[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS],
    int *polynomial_predictor_used)
{
    double ratio = 0.0;
    int stage;
    int component;
    *polynomial_predictor_used = 0;
    if (history_valid && previous_step != 0.0) {
        ratio = step / previous_step;
    }
    if (history_valid && isfinite(ratio) && ratio >= 0.25 && ratio <= 2.0) {
        double velocity_weight[JX_GR15_STAGE_COUNT][JX_GR15_STAGE_COUNT];
        double position_weight[JX_GR15_STAGE_COUNT][JX_GR15_STAGE_COUNT];
        *polynomial_predictor_used = 1;
        for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
            const double normalized = 1.0 + ratio * jx_gr15_nodes[stage];
            int source;
            for (source = 0; source < JX_GR15_STAGE_COUNT; ++source) {
                double acceleration_integral = 0.0;
                double acceleration_moment = 0.0;
                double power_value = normalized;
                int power;
                for (power = 0; power < JX_GR15_STAGE_COUNT; ++power) {
                    const double next_power = power_value * normalized;
                    const double coefficient =
                        jx_gr15_power_from_stage[power][source];
                    acceleration_integral += coefficient
                        * (power_value - 1.0) / (double)(power + 1);
                    acceleration_moment += coefficient
                        * (next_power - 1.0) / (double)(power + 2);
                    power_value = next_power;
                }
                velocity_weight[stage][source] = acceleration_integral;
                position_weight[stage][source] =
                    normalized * acceleration_integral - acceleration_moment;
            }
        }
        for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
            const double normalized = 1.0 + ratio * jx_gr15_nodes[stage];
            for (component = 0; component < component_count; ++component) {
                double acceleration_integral = 0.0;
                double acceleration_double_integral = 0.0;
                int source;
                for (source = 0; source < JX_GR15_STAGE_COUNT; ++source) {
                    const double acceleration =
                        previous_stage_acceleration[source][component];
                    acceleration_integral +=
                        velocity_weight[stage][source] * acceleration;
                    acceleration_double_integral +=
                        position_weight[stage][source] * acceleration;
                }
                stage_velocity[stage][component] = velocity[component]
                    + previous_step * acceleration_integral;
                stage_position[stage][component] = position[component]
                    + previous_step * (normalized - 1.0) * velocity[component]
                    + previous_step * previous_step
                        * acceleration_double_integral;
                if (!isfinite(stage_position[stage][component])
                    || !isfinite(stage_velocity[stage][component])) {
                    return JX_GR15_NONFINITE;
                }
            }
        }
        return JX_GR15_SUCCESS;
    }
    for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
        const double offset = jx_gr15_nodes[stage] * step;
        for (component = 0; component < component_count; ++component) {
            stage_position[stage][component] = position[component]
                + offset * velocity[component]
                + 0.5 * offset * offset * initial_acceleration[component];
            stage_velocity[stage][component] = velocity[component]
                + offset * initial_acceleration[component];
        }
    }
    return JX_GR15_SUCCESS;
}

static int
jx_gr15_trial(
    int body_count,
    const double *position,
    const double *velocity,
    const double *gm,
    double step,
    double convergence_factor,
    int maximum_iterations,
    int history_valid,
    double previous_step,
    double previous_stage_acceleration[JX_GR15_STAGE_COUNT]
        [JX_GR15_MAX_COMPONENTS],
    double accepted_stage_acceleration[JX_GR15_STAGE_COUNT]
        [JX_GR15_MAX_COMPONENTS],
    double *candidate_position,
    double *candidate_velocity,
    double *position_carry,
    double *velocity_carry,
    double *error_ratio,
    int64_t *force_evaluations,
    int64_t *iteration_count,
    int *converged_out,
    int *polynomial_predictor_used)
{
    const int component_count = body_count * 3;
    double initial_acceleration[JX_GR15_MAX_COMPONENTS];
    double stage_position[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS];
    double stage_velocity[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS];
    double stage_acceleration[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS];
    double next_position[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS];
    double next_velocity[JX_GR15_STAGE_COUNT][JX_GR15_MAX_COMPONENTS];
    double divided[JX_GR15_STAGE_COUNT];
    double maximum_acceleration = 0.0;
    double maximum_highest = 0.0;
    int iteration;
    int stage;
    int component;
    int status = jx_gr15_acceleration(
        body_count, position, gm, initial_acceleration);
    if (status != JX_GR15_SUCCESS) {
        return status;
    }
    *force_evaluations += 1;
    status = jx_gr15_predict(
        component_count,
        position,
        velocity,
        initial_acceleration,
        step,
        history_valid,
        previous_step,
        previous_stage_acceleration,
        stage_position,
        stage_velocity,
        polynomial_predictor_used);
    if (status != JX_GR15_SUCCESS) {
        return status;
    }
    *converged_out = 0;
    for (iteration = 0; iteration < maximum_iterations; ++iteration) {
        double maximum_scaled_change = 0.0;
        for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
            status = jx_gr15_acceleration(
                body_count,
                stage_position[stage],
                gm,
                stage_acceleration[stage]);
            if (status != JX_GR15_SUCCESS) {
                return status;
            }
            *force_evaluations += 1;
        }
        for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
            for (component = 0; component < component_count; ++component) {
                double position_sum = 0.0;
                double velocity_sum = 0.0;
                int source_stage;
                for (source_stage = 0;
                     source_stage < JX_GR15_STAGE_COUNT;
                     ++source_stage) {
                    const double coefficient =
                        jx_gr15_matrix[stage][source_stage];
                    position_sum += coefficient
                        * stage_velocity[source_stage][component];
                    velocity_sum += coefficient
                        * stage_acceleration[source_stage][component];
                }
                next_position[stage][component] =
                    position[component] + step * position_sum;
                next_velocity[stage][component] =
                    velocity[component] + step * velocity_sum;
                if (!isfinite(next_position[stage][component])
                    || !isfinite(next_velocity[stage][component])) {
                    return JX_GR15_NONFINITE;
                }
                {
                    const double position_scale = 1.0 + fmax(
                        fabs(stage_position[stage][component]),
                        fabs(next_position[stage][component]));
                    const double velocity_scale = 1.0 + fmax(
                        fabs(stage_velocity[stage][component]),
                        fabs(next_velocity[stage][component]));
                    maximum_scaled_change = fmax(
                        maximum_scaled_change,
                        fabs(next_position[stage][component]
                             - stage_position[stage][component])
                            / position_scale);
                    maximum_scaled_change = fmax(
                        maximum_scaled_change,
                        fabs(next_velocity[stage][component]
                             - stage_velocity[stage][component])
                            / velocity_scale);
                }
            }
        }
        memcpy(stage_position, next_position, sizeof(stage_position));
        memcpy(stage_velocity, next_velocity, sizeof(stage_velocity));
        *iteration_count += 1;
        if (maximum_scaled_change <= convergence_factor * DBL_EPSILON) {
            *converged_out = 1;
            break;
        }
    }
    if (!*converged_out) {
        return JX_GR15_SUCCESS;
    }
    for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
        status = jx_gr15_acceleration(
            body_count,
            stage_position[stage],
            gm,
            stage_acceleration[stage]);
        if (status != JX_GR15_SUCCESS) {
            return status;
        }
        *force_evaluations += 1;
    }
    memcpy(
        accepted_stage_acceleration,
        stage_acceleration,
        sizeof(stage_acceleration));
    for (component = 0; component < component_count; ++component) {
        double position_sum = 0.0;
        double velocity_sum = 0.0;
        double adjusted_position;
        double adjusted_velocity;
        for (stage = 0; stage < JX_GR15_STAGE_COUNT; ++stage) {
            position_sum += jx_gr15_weights[stage]
                * stage_velocity[stage][component];
            velocity_sum += jx_gr15_weights[stage]
                * stage_acceleration[stage][component];
            maximum_acceleration = fmax(
                maximum_acceleration,
                fabs(stage_acceleration[stage][component]));
            divided[stage] = stage_acceleration[stage][component];
        }
        for (stage = 1; stage < JX_GR15_STAGE_COUNT; ++stage) {
            int index;
            for (index = JX_GR15_STAGE_COUNT - 1; index >= stage; --index) {
                divided[index] =
                    (divided[index] - divided[index - 1])
                    / (jx_gr15_nodes[index]
                       - jx_gr15_nodes[index - stage]);
            }
        }
        maximum_highest = fmax(maximum_highest, fabs(divided[7]));
        adjusted_position = step * position_sum - position_carry[component];
        adjusted_velocity = step * velocity_sum - velocity_carry[component];
        candidate_position[component] = position[component] + adjusted_position;
        candidate_velocity[component] = velocity[component] + adjusted_velocity;
        position_carry[component] =
            (candidate_position[component] - position[component])
            - adjusted_position;
        velocity_carry[component] =
            (candidate_velocity[component] - velocity[component])
            - adjusted_velocity;
        if (!isfinite(candidate_position[component])
            || !isfinite(candidate_velocity[component])) {
            return JX_GR15_NONFINITE;
        }
    }
    if (!(maximum_acceleration > 0.0) || !isfinite(maximum_highest)) {
        return JX_GR15_NONFINITE;
    }
    *error_ratio = maximum_highest / maximum_acceleration;
    if (!isfinite(*error_ratio) || *error_ratio < 0.0) {
        return JX_GR15_NONFINITE;
    }
    return JX_GR15_SUCCESS;
}

int
jx_gr15_integrate(
    int body_count,
    int checkpoint_count,
    const double *input_position,
    const double *input_velocity,
    const double *gm,
    const double *checkpoints,
    double initial_step,
    double minimum_step,
    double maximum_step,
    double epsilon,
    double safety_factor,
    double minimum_scale_factor,
    double maximum_scale_factor,
    double convergence_factor,
    int maximum_iterations,
    int64_t maximum_steps,
    int64_t maximum_rejections,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    int64_t *checkpoint_accepted,
    int64_t *checkpoint_rejected,
    int64_t *counters,
    double *metrics)
{
    int component_count;
    double direction;
    double position[JX_GR15_MAX_COMPONENTS];
    double velocity[JX_GR15_MAX_COMPONENTS];
    double position_carry[JX_GR15_MAX_COMPONENTS] = {0.0};
    double velocity_carry[JX_GR15_MAX_COMPONENTS] = {0.0};
    double candidate_position[JX_GR15_MAX_COMPONENTS];
    double candidate_velocity[JX_GR15_MAX_COMPONENTS];
    double previous_stage_acceleration[JX_GR15_STAGE_COUNT]
        [JX_GR15_MAX_COMPONENTS] = {{0.0}};
    double trial_stage_acceleration[JX_GR15_STAGE_COUNT]
        [JX_GR15_MAX_COMPONENTS];
    double previous_step = 0.0;
    double epoch;
    double proposed;
    double maximum_error = 0.0;
    double minimum_accepted_step = INFINITY;
    double maximum_accepted_step = 0.0;
    int history_valid = 0;
    int checkpoint_index;
    int component;
    int body;
    int64_t attempted = 0;
    int64_t accepted = 0;
    int64_t rejected = 0;
    int64_t force_evaluations = 0;
    int64_t iteration_count = 0;
    int64_t nonconverged_retries = 0;
    int64_t polynomial_predictor_trials = 0;
    int64_t constant_predictor_trials = 0;
    int64_t history_commits = 0;
    if (input_position == NULL || input_velocity == NULL || gm == NULL
        || checkpoints == NULL || checkpoint_positions == NULL
        || checkpoint_velocities == NULL || checkpoint_accepted == NULL
        || checkpoint_rejected == NULL || counters == NULL || metrics == NULL
        || body_count < 2 || body_count > JX_GR15_MAX_BODIES
        || checkpoint_count < 2
        || !isfinite(initial_step) || !isfinite(minimum_step)
        || !isfinite(maximum_step) || !isfinite(epsilon)
        || !isfinite(safety_factor) || !isfinite(minimum_scale_factor)
        || !isfinite(maximum_scale_factor) || !isfinite(convergence_factor)
        || minimum_step <= 0.0 || initial_step < minimum_step
        || maximum_step < initial_step || epsilon <= 0.0
        || safety_factor <= 0.0 || safety_factor >= 1.0
        || minimum_scale_factor <= 0.0 || minimum_scale_factor > 1.0
        || maximum_scale_factor < 1.0
        || convergence_factor < 1.0 || maximum_iterations < 1
        || maximum_iterations > 12 || maximum_steps < 1
        || maximum_rejections < 0) {
        return JX_GR15_INPUT_DOMAIN;
    }
    direction = checkpoints[checkpoint_count - 1] > checkpoints[0]
        ? 1.0 : -1.0;
    if (!isfinite(checkpoints[0])
        || !isfinite(checkpoints[checkpoint_count - 1])
        || checkpoints[checkpoint_count - 1] == checkpoints[0]) {
        return JX_GR15_CHECKPOINT_SCHEDULE;
    }
    for (checkpoint_index = 1;
         checkpoint_index < checkpoint_count;
         ++checkpoint_index) {
        if (!isfinite(checkpoints[checkpoint_index])
            || !(direction * (checkpoints[checkpoint_index]
                               - checkpoints[checkpoint_index - 1]) > 0.0)) {
            return JX_GR15_CHECKPOINT_SCHEDULE;
        }
    }
    component_count = body_count * 3;
    for (body = 0; body < body_count; ++body) {
        if (!isfinite(gm[body]) || gm[body] <= 0.0) {
            return JX_GR15_INPUT_DOMAIN;
        }
    }
    for (component = 0; component < component_count; ++component) {
        if (!isfinite(input_position[component])
            || !isfinite(input_velocity[component])) {
            return JX_GR15_NONFINITE;
        }
    }
    memcpy(position, input_position, (size_t)component_count * sizeof(double));
    memcpy(velocity, input_velocity, (size_t)component_count * sizeof(double));
    memcpy(checkpoint_positions, position,
           (size_t)component_count * sizeof(double));
    memcpy(checkpoint_velocities, velocity,
           (size_t)component_count * sizeof(double));
    checkpoint_accepted[0] = 0;
    checkpoint_rejected[0] = 0;
    epoch = checkpoints[0];
    proposed = initial_step;
    for (checkpoint_index = 1;
         checkpoint_index < checkpoint_count;
         ++checkpoint_index) {
        const double checkpoint_epoch = checkpoints[checkpoint_index];
        while (epoch != checkpoint_epoch) {
            const double remaining = checkpoint_epoch - epoch;
            double step;
            double error_ratio = INFINITY;
            double next_factor;
            double required;
            double trial_position_carry[JX_GR15_MAX_COMPONENTS];
            double trial_velocity_carry[JX_GR15_MAX_COMPONENTS];
            int converged = 0;
            int polynomial_predictor_used = 0;
            int status;
            if (attempted >= maximum_steps) {
                return JX_GR15_STEP_LIMIT;
            }
            proposed = fmin(maximum_step, fmax(minimum_step, proposed));
            if (!(direction * remaining > 0.0) || !isfinite(remaining)) {
                return JX_GR15_CHECKPOINT_SCHEDULE;
            }
            step = copysign(fmin(proposed, fabs(remaining)), direction);
            if (epoch + step == epoch || !isfinite(step)) {
                return JX_GR15_MINIMUM_STEP;
            }
            memcpy(trial_position_carry, position_carry,
                   (size_t)component_count * sizeof(double));
            memcpy(trial_velocity_carry, velocity_carry,
                   (size_t)component_count * sizeof(double));
            status = jx_gr15_trial(
                body_count,
                position,
                velocity,
                gm,
                step,
                convergence_factor,
                maximum_iterations,
                history_valid,
                previous_step,
                previous_stage_acceleration,
                trial_stage_acceleration,
                candidate_position,
                candidate_velocity,
                trial_position_carry,
                trial_velocity_carry,
                &error_ratio,
                &force_evaluations,
                &iteration_count,
                &converged,
                &polynomial_predictor_used);
            if (status != JX_GR15_SUCCESS) {
                return status;
            }
            attempted += 1;
            if (polynomial_predictor_used) {
                polynomial_predictor_trials += 1;
            } else {
                constant_predictor_trials += 1;
            }
            maximum_error = fmax(maximum_error, error_ratio);
            if (!converged) {
                rejected += 1;
                nonconverged_retries += 1;
                if (rejected > maximum_rejections) {
                    return JX_GR15_REJECTION_LIMIT;
                }
                if (proposed <= minimum_step) {
                    return JX_GR15_MINIMUM_STEP;
                }
                proposed = fmax(minimum_step, 0.5 * fabs(step));
                continue;
            }
            next_factor = maximum_scale_factor;
            if (error_ratio > 0.0) {
                next_factor = safety_factor
                    * pow(epsilon / error_ratio, 1.0 / 7.0);
                next_factor = fmin(
                    maximum_scale_factor,
                    fmax(minimum_scale_factor, next_factor));
            }
            required = fabs(step) * next_factor;
            if (error_ratio > epsilon) {
                rejected += 1;
                if (rejected > maximum_rejections) {
                    return JX_GR15_REJECTION_LIMIT;
                }
                if (fabs(step) <= minimum_step) {
                    return JX_GR15_MINIMUM_STEP;
                }
                proposed = fmax(minimum_step, required);
                continue;
            }
            memcpy(position, candidate_position,
                   (size_t)component_count * sizeof(double));
            memcpy(velocity, candidate_velocity,
                   (size_t)component_count * sizeof(double));
            memcpy(position_carry, trial_position_carry,
                   (size_t)component_count * sizeof(double));
            memcpy(velocity_carry, trial_velocity_carry,
                   (size_t)component_count * sizeof(double));
            memcpy(previous_stage_acceleration, trial_stage_acceleration,
                   sizeof(previous_stage_acceleration));
            previous_step = step;
            history_valid = 1;
            history_commits += 1;
            epoch += step;
            if (fabs(checkpoint_epoch - epoch)
                <= 2.0 * DBL_EPSILON
                    * fmax(1.0, fabs(checkpoint_epoch))) {
                epoch = checkpoint_epoch;
            }
            accepted += 1;
            minimum_accepted_step = fmin(minimum_accepted_step, fabs(step));
            maximum_accepted_step = fmax(maximum_accepted_step, fabs(step));
            proposed = fmin(maximum_step, fmax(minimum_step, required));
        }
        memcpy(
            checkpoint_positions + (size_t)checkpoint_index * component_count,
            position,
            (size_t)component_count * sizeof(double));
        memcpy(
            checkpoint_velocities + (size_t)checkpoint_index * component_count,
            velocity,
            (size_t)component_count * sizeof(double));
        checkpoint_accepted[checkpoint_index] = accepted;
        checkpoint_rejected[checkpoint_index] = rejected;
    }
    counters[0] = attempted;
    counters[1] = accepted;
    counters[2] = rejected;
    counters[3] = force_evaluations;
    counters[4] = iteration_count;
    counters[5] = nonconverged_retries;
    counters[6] = polynomial_predictor_trials;
    counters[7] = constant_predictor_trials;
    counters[8] = history_commits;
    metrics[0] = maximum_error;
    metrics[1] = minimum_accepted_step;
    metrics[2] = maximum_accepted_step;
    metrics[3] = proposed;
    return JX_GR15_SUCCESS;
}
