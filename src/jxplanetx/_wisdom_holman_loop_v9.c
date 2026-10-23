/*
 * Bundled universal-G series v9 prototype.
 *
 * V8 is included unchanged.  V9 replaces the two hot nextafter operations
 * used by the ULP and adjacent-root checks with an IEEE-754 binary64 bit step.
 * The substitution preserves the returned double while avoiding a libm call.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>


_Static_assert(CHAR_BIT == 8, "v9 requires eight-bit bytes");
_Static_assert(sizeof(double) == sizeof(uint64_t),
    "v9 requires a 64-bit double");
_Static_assert(FLT_RADIX == 2, "v9 requires a binary floating-point radix");
_Static_assert(DBL_MANT_DIG == 53,
    "v9 requires an IEEE-754 binary64 significand");
_Static_assert(DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021,
    "v9 requires IEEE-754 binary64 exponents");
#ifdef DBL_HAS_SUBNORM
_Static_assert(DBL_HAS_SUBNORM == 1,
    "v9 requires IEEE-754 binary64 subnormals");
#endif


static double
jx_wh_nextafter_v9_bits(double from, double toward)
{
    uint64_t bits;
    double result;

    /* Preserve libm behavior for values outside the finite bit-step domain. */
    if (isnan(from) || isnan(toward) || isinf(from) || from == toward
        || fabs(from) == DBL_MAX) {
        return nextafter(from, toward);
    }
    if (from == 0.0) {
        bits = signbit(toward)
            ? UINT64_C(0x8000000000000001)
            : UINT64_C(0x0000000000000001);
    } else {
        memcpy(&bits, &from, sizeof(bits));
        if ((from < toward) == (from > 0.0)) {
            bits += UINT64_C(1);
        } else {
            bits -= UINT64_C(1);
        }
    }
    memcpy(&result, &bits, sizeof(result));
    return result;
}


double
jx_wh_loop_v9_nextafter_bits(double from, double toward)
{
    return jx_wh_nextafter_v9_bits(from, toward);
}


#define nextafter jx_wh_nextafter_v9_bits
#define jx_wh_g_values_v8 jx_wh_g_values_v9
#define jx_wh_loop_v8_g_bundle jx_wh_loop_v9_g_bundle
#define jx_wh_universal_evaluation_v8 jx_wh_universal_evaluation_v9
#define jx_wh_v8_fallback_count jx_wh_v9_fallback_count
#define jx_wh_loop_v8_last_fallback_count jx_wh_loop_v9_last_fallback_count
#define jx_wh_loop_v8_last_cache_entries jx_wh_loop_v9_last_cache_entries
#define jx_wh_loop_v8_last_cache_hits jx_wh_loop_v9_last_cache_hits
#define jx_wh_loop_v8_last_cache_misses jx_wh_loop_v9_last_cache_misses
#define jx_wh_kepler_step_v8_candidate jx_wh_kepler_step_v9_candidate
#define jx_wh_kepler_step_v8 jx_wh_kepler_step_v9
#define jx_wh_loop_integrate_v8 jx_wh_loop_integrate_v9

#include "_wisdom_holman_loop_v8.c"

#undef jx_wh_loop_integrate_v8
#undef jx_wh_kepler_step_v8
#undef jx_wh_kepler_step_v8_candidate
#undef jx_wh_loop_v8_last_cache_misses
#undef jx_wh_loop_v8_last_cache_hits
#undef jx_wh_loop_v8_last_cache_entries
#undef jx_wh_loop_v8_last_fallback_count
#undef jx_wh_v8_fallback_count
#undef jx_wh_universal_evaluation_v8
#undef jx_wh_loop_v8_g_bundle
#undef jx_wh_g_values_v8
#undef nextafter
