/*
 * Hill-mass-factor cache v7 prototype.
 *
 * The complete v6 implementation is included unchanged except that its sole
 * libm pow call (the mass-only Hill factor) is routed through a per-map exact
 * binary64 cache.  The cache is reset before every map call.  A miss stores
 * the exact libm result; a hit returns that same value.  The time-varying
 * semimajor-axis multiplication and all guards remain in the v2/v3 core.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>


#define JX_WH_V7_CACHE_CAPACITY 512


static _Thread_local double jx_wh_v7_cache_bases[JX_WH_V7_CACHE_CAPACITY];
static _Thread_local double jx_wh_v7_cache_exponents[JX_WH_V7_CACHE_CAPACITY];
static _Thread_local double jx_wh_v7_cache_values[JX_WH_V7_CACHE_CAPACITY];
static _Thread_local int64_t jx_wh_v7_cache_entries = 0;
static _Thread_local int64_t jx_wh_v7_cache_hits = 0;
static _Thread_local int64_t jx_wh_v7_cache_misses = 0;


static double
jx_wh_v7_cached_pow(double base, double exponent)
{
    int64_t index;
    double value;
    if (!isfinite(base) || !isfinite(exponent)) {
        return pow(base, exponent);
    }
    for (index = 0; index < jx_wh_v7_cache_entries; ++index) {
        if (base == jx_wh_v7_cache_bases[index]
            && exponent == jx_wh_v7_cache_exponents[index]) {
            jx_wh_v7_cache_hits += 1;
            return jx_wh_v7_cache_values[index];
        }
    }
    value = pow(base, exponent);
    jx_wh_v7_cache_misses += 1;
    if (jx_wh_v7_cache_entries < JX_WH_V7_CACHE_CAPACITY) {
        index = jx_wh_v7_cache_entries;
        jx_wh_v7_cache_bases[index] = base;
        jx_wh_v7_cache_exponents[index] = exponent;
        jx_wh_v7_cache_values[index] = value;
        jx_wh_v7_cache_entries += 1;
    }
    return value;
}


#define pow jx_wh_v7_cached_pow
#include "_wisdom_holman_loop_v6.c"
#undef pow


static _Thread_local int64_t jx_wh_v7_fallback_count = 0;


int64_t
jx_wh_loop_v7_last_fallback_count(void)
{
    return jx_wh_v7_fallback_count;
}


int64_t
jx_wh_loop_v7_last_cache_entries(void)
{
    return jx_wh_v7_cache_entries;
}


int64_t
jx_wh_loop_v7_last_cache_hits(void)
{
    return jx_wh_v7_cache_hits;
}


int64_t
jx_wh_loop_v7_last_cache_misses(void)
{
    return jx_wh_v7_cache_misses;
}


static int
jx_wh_kepler_step_v7(
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
    status = jx_wh_kepler_step_v6_candidate(
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
    jx_wh_v7_fallback_count += 1;
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
jx_wh_loop_integrate_v7(
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
    jx_wh_v7_cache_entries = 0;
    jx_wh_v7_cache_hits = 0;
    jx_wh_v7_cache_misses = 0;
    jx_wh_v7_fallback_count = 0;
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
        jx_wh_kepler_step_v7);
}
