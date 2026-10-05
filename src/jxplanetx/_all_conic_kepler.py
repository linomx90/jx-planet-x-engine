"""Private all-conic universal-variable drift for tracer integration work.

This module is deliberately additive: no qualified engine imports it yet.  It
advances a *relative* two-body state for elliptic, parabolic, or hyperbolic
motion.  A future democratic-heliocentric tracer split should apply it to
``q = x_tracer - x_primary`` and ``u = v_tracer - v_primary``; perturbing
kicks must use differential acceleration and the primary endpoint must be
added back after the drift.

The scalar reference and CUDA kernel share the G-function definition, signed
root bracket, convergence test, and reconstruction.  Constants that are not
conic-specific are reused from the qualified elliptic Wisdom--Holman solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .engine.wisdom_holman_contracts import (
    WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR,
    WH_KEPLER_BRACKET_ULP_FACTOR,
    WH_KEPLER_ENERGY_ROUNDOFF_FACTOR,
    WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR,
    WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS,
    WH_KEPLER_MAXIMUM_ITERATIONS,
    WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR,
    WH_KEPLER_RESIDUAL_ULP_FACTOR,
    WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT,
    WH_KEPLER_SERIES_TERM_COUNT,
)


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
)
# CuPy appends ``-ftz=true`` internally.  NVIDIA documents FTZ, precise divide,
# and precise square root as single-precision-only controls; this source uses
# binary64 exclusively.  Omitting contradictory no-op flags avoids a misleading
# NVRTC warning, while ``--fmad=false`` still fixes binary64 operation order.
CUDA_DEVICE_FUNCTION_NAME = "jx_all_conic_drift_device"
HYPERBOLIC_MAX_ABS_ARGUMENT = 700.0
STATUS_SUCCESS = 0
STATUS_INPUT_DOMAIN = 1
STATUS_NONFINITE = 2
STATUS_RADIUS_DOMAIN = 3
STATUS_BRACKET_FAILURE = 4
STATUS_ITERATION_FAILURE = 5
STATUS_RECONSTRUCTION_FAILURE = 6
STATUS_POSTCONDITION_FAILURE = 7
STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_INPUT_DOMAIN: "INPUT_DOMAIN",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_RADIUS_DOMAIN: "RADIUS_DOMAIN",
    STATUS_BRACKET_FAILURE: "BRACKET_FAILURE",
    STATUS_ITERATION_FAILURE: "ITERATION_FAILURE",
    STATUS_RECONSTRUCTION_FAILURE: "RECONSTRUCTION_FAILURE",
    STATUS_POSTCONDITION_FAILURE: "POSTCONDITION_FAILURE",
}
CONIC_HYPERBOLIC = -1
CONIC_PARABOLIC = 0
CONIC_ELLIPTIC = 1
CONIC_NAMES = {
    CONIC_HYPERBOLIC: "hyperbolic",
    CONIC_PARABOLIC: "parabolic",
    CONIC_ELLIPTIC: "elliptic",
}
_EPSILON = np.finfo(np.float64).eps


class AllConicKeplerError(RuntimeError):
    """Base error for the private all-conic drift."""


class AllConicKeplerDomainError(AllConicKeplerError, ValueError):
    """The requested state or an intermediate leaves the finite domain."""


class AllConicKeplerConvergenceError(AllConicKeplerError):
    """The fixed signed root solve did not converge."""


@dataclass(frozen=True)
class UniversalDriftResult:
    position: np.ndarray
    velocity: np.ndarray
    conic: str
    beta: float
    universal_anomaly: float
    iterations: int
    bracket_expansions: int
    time_residual: float
    residual_tolerance: float
    final_bracket_width: float
    lagrange_identity_error: float
    energy_error: float
    angular_momentum_error: float


@dataclass(frozen=True)
class CUDAUniversalDriftBatch:
    positions: Any
    velocities: Any
    status: Any
    conic: Any
    beta: Any
    universal_anomaly: Any
    iterations: Any
    bracket_expansions: Any
    time_residual: Any
    residual_tolerance: Any
    lagrange_identity_error: Any
    energy_error: Any
    angular_momentum_error: Any


@dataclass(frozen=True)
class _GValues:
    g0: float
    g1: float
    g2: float
    g3: float
    used_series: bool


@dataclass(frozen=True)
class _Evaluation:
    anomaly: float
    residual: float
    radius: float
    tolerance: float
    g: _GValues


def _dot3(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        (float(left[0]) * float(right[0]) + float(left[1]) * float(right[1]))
        + float(left[2]) * float(right[2])
    )


def _norm3(value: np.ndarray) -> float:
    squared = _dot3(value, value)
    if not math.isfinite(squared) or squared < 0.0:
        raise AllConicKeplerDomainError("vector norm became nonfinite")
    return math.sqrt(squared)


def _series_g_values(beta: float, anomaly: float) -> _GValues:
    terms = [
        1.0,
        anomaly,
        (anomaly * anomaly) / 2.0,
        ((anomaly * anomaly) * anomaly) / 6.0,
    ]
    totals = terms.copy()
    rho = ((-beta) * anomaly) * anomaly
    for term_index in range(1, WH_KEPLER_SERIES_TERM_COUNT):
        twice = 2 * term_index
        previous = totals.copy()
        for order in range(4):
            denominator = (twice + order - 1) * (twice + order)
            terms[order] = terms[order] * rho / float(denominator)
            totals[order] = totals[order] + terms[order]
        if not all(math.isfinite(value) for value in (*terms, *totals)):
            raise AllConicKeplerDomainError(
                "universal G series produced a nonfinite value"
            )
        # With |rho| <= .25 in this branch, every following term is smaller
        # in magnitude.  Once all four correctly ordered additions round back
        # to the existing binary64 totals, the remaining fixed-series terms
        # are provably bitwise inert.  This preserves the 64-term result while
        # avoiding dozens of useless divisions in the common small-step case.
        if totals == previous:
            break
    return _GValues(*totals, used_series=True)


def _universal_g_values(beta: float, anomaly: float) -> _GValues:
    """Evaluate universal G0..G3 for every finite sign of ``beta``."""

    if not math.isfinite(beta) or not math.isfinite(anomaly):
        raise AllConicKeplerDomainError("beta and anomaly must be finite")
    if beta == 0.0:
        return _GValues(
            1.0,
            anomaly,
            (anomaly * anomaly) / 2.0,
            ((anomaly * anomaly) * anomaly) / 6.0,
            True,
        )
    root_abs_beta = math.sqrt(abs(beta))
    argument = root_abs_beta * anomaly
    if not math.isfinite(argument):
        raise AllConicKeplerDomainError("universal G argument became nonfinite")
    if abs(argument) <= WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT:
        return _series_g_values(beta, anomaly)
    if beta > 0.0:
        cosine = math.cos(argument)
        sine = math.sin(argument)
        values = (
            cosine,
            sine / root_abs_beta,
            (1.0 - cosine) / beta,
            (anomaly - sine / root_abs_beta) / beta,
        )
    else:
        if abs(argument) > HYPERBOLIC_MAX_ABS_ARGUMENT:
            raise AllConicKeplerDomainError(
                "hyperbolic universal argument exceeds the finite envelope"
            )
        cosine = math.cosh(argument)
        sine = math.sinh(argument)
        absolute_beta = -beta
        g1 = sine / root_abs_beta
        values = (
            cosine,
            g1,
            (cosine - 1.0) / absolute_beta,
            (g1 - anomaly) / absolute_beta,
        )
    if not all(math.isfinite(value) for value in values):
        raise AllConicKeplerDomainError(
            "universal G closed-form branch produced a nonfinite value"
        )
    return _GValues(*values, used_series=False)


def _evaluate(
    anomaly: float,
    beta: float,
    radius_initial: float,
    radial_moment: float,
    gravitational_parameter: float,
    signed_step: float,
) -> _Evaluation:
    g = _universal_g_values(beta, anomaly)
    coefficient = gravitational_parameter - beta * radius_initial
    first = radius_initial * anomaly
    second = radial_moment * g.g2
    third = coefficient * g.g3
    residual = ((first + second) + third) - signed_step
    radius = (radius_initial + radial_moment * g.g1) + coefficient * g.g2
    scale = abs(first) + abs(second) + abs(third) + abs(signed_step)
    tolerance = max(
        WH_KEPLER_RESIDUAL_ULP_FACTOR * math.ulp(abs(signed_step)),
        WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR * _EPSILON * scale,
    )
    if not all(
        math.isfinite(value) for value in (residual, radius, scale, tolerance)
    ):
        raise AllConicKeplerDomainError(
            "universal Kepler equation produced a nonfinite value"
        )
    if radius <= 0.0:
        raise AllConicKeplerDomainError(
            "universal Kepler root derivative radius must be positive"
        )
    return _Evaluation(anomaly, residual, radius, tolerance, g)


def _validated_vector(value: Any, label: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not bool(np.all(np.isfinite(result))):
        raise AllConicKeplerDomainError(
            f"{label} must be a finite three-component vector"
        )
    return result


def drift_all_conic(
    position: Any,
    velocity: Any,
    gravitational_parameter: float,
    signed_step: float,
) -> UniversalDriftResult:
    """Advance one finite relative two-body state through a signed time step."""

    q0 = _validated_vector(position, "position")
    u0 = _validated_vector(velocity, "velocity")
    mu = float(gravitational_parameter)
    step = float(signed_step)
    if not math.isfinite(mu) or mu <= 0.0:
        raise AllConicKeplerDomainError(
            "gravitational_parameter must be finite and positive"
        )
    if not math.isfinite(step) or step == 0.0:
        raise AllConicKeplerDomainError("signed_step must be finite and nonzero")

    radius_initial = _norm3(q0)
    if radius_initial <= 0.0:
        raise AllConicKeplerDomainError("initial radius must be positive")
    speed_squared = _dot3(u0, u0)
    beta = (2.0 * mu / radius_initial) - speed_squared
    radial_moment = _dot3(q0, u0)
    if not math.isfinite(beta) or not math.isfinite(radial_moment):
        raise AllConicKeplerDomainError("derived orbital scalars must be finite")
    conic_code = (
        CONIC_ELLIPTIC
        if beta > 0.0
        else CONIC_HYPERBOLIC
        if beta < 0.0
        else CONIC_PARABOLIC
    )

    seed = step / radius_initial
    if beta < 0.0:
        series_edge = WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT / math.sqrt(-beta)
        if abs(seed) > series_edge:
            seed = math.copysign(series_edge, step)
    if not math.isfinite(seed) or seed == 0.0:
        raise AllConicKeplerConvergenceError(
            "root seed cannot make finite representable progress"
        )

    outer = seed
    outer_evaluation = _evaluate(
        outer, beta, radius_initial, radial_moment, mu, step
    )
    bracket_expansions = 0
    while (step > 0.0 and outer_evaluation.residual < 0.0) or (
        step < 0.0 and outer_evaluation.residual > 0.0
    ):
        if bracket_expansions >= WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS:
            raise AllConicKeplerConvergenceError(
                "root could not be bracketed within the fixed expansion cap"
            )
        candidate = outer * 2.0
        if beta < 0.0:
            maximum = HYPERBOLIC_MAX_ABS_ARGUMENT / math.sqrt(-beta)
            if abs(candidate) > maximum:
                candidate = math.copysign(maximum, step)
        if not math.isfinite(candidate) or candidate == outer:
            raise AllConicKeplerConvergenceError(
                "root bracket cannot make finite representable progress"
            )
        outer = candidate
        bracket_expansions += 1
        outer_evaluation = _evaluate(
            outer, beta, radius_initial, radial_moment, mu, step
        )

    zero_residual = -step
    if step > 0.0:
        low, residual_low = 0.0, zero_residual
        high, residual_high = outer, outer_evaluation.residual
    else:
        low, residual_low = outer, outer_evaluation.residual
        high, residual_high = 0.0, zero_residual
    if not (
        low < high
        and residual_low <= 0.0 <= residual_high
        and all(
            math.isfinite(value)
            for value in (low, high, residual_low, residual_high)
        )
    ):
        raise AllConicKeplerConvergenceError(
            "signed root bracket does not preserve R(low)<=0<=R(high)"
        )

    current = outer if outer_evaluation.residual == 0.0 else low + 0.5 * (high - low)
    previous: float | None = None
    two_iterations_ago: float | None = None
    accepted: _Evaluation | None = None
    final_width = math.inf
    accepted_iterations = 0
    for iteration in range(1, WH_KEPLER_MAXIMUM_ITERATIONS + 1):
        evaluation = _evaluate(
            current, beta, radius_initial, radial_moment, mu, step
        )
        if evaluation.residual == 0.0:
            low = high = current
            residual_low = residual_high = 0.0
        elif evaluation.residual < 0.0:
            low, residual_low = current, evaluation.residual
        else:
            high, residual_high = current, evaluation.residual
        if not (residual_low <= 0.0 <= residual_high and low <= high):
            raise AllConicKeplerConvergenceError(
                "iteration lost the signed root bracket"
            )

        final_width = high - low
        ulp_scale = max(math.ulp(abs(low)), math.ulp(abs(high)), math.ulp(abs(current)))
        adjacent = low == high or math.nextafter(low, high) == high
        settled = adjacent or final_width <= WH_KEPLER_BRACKET_ULP_FACTOR * ulp_scale
        cycle = (previous is not None and current == previous) or (
            two_iterations_ago is not None and current == two_iterations_ago
        )
        if abs(evaluation.residual) <= evaluation.tolerance and (settled or cycle):
            accepted = evaluation
            accepted_iterations = iteration
            break

        midpoint = low + 0.5 * (high - low)
        newton = current - evaluation.residual / evaluation.radius
        candidate = newton if math.isfinite(newton) and low < newton < high else midpoint
        if candidate == current:
            candidate = midpoint
        if (
            not math.isfinite(candidate)
            or not low <= candidate <= high
            or candidate == current
        ):
            raise AllConicKeplerConvergenceError(
                "iteration stagnated before satisfying convergence"
            )
        two_iterations_ago, previous, current = previous, current, candidate
    if accepted is None:
        raise AllConicKeplerConvergenceError(
            "root exceeded the fixed iteration cap"
        )

    g = accepted.g
    f_value = 1.0 - (mu / radius_initial) * g.g2
    g_value = step - mu * g.g3
    q1 = f_value * q0 + g_value * u0
    radius_final = _norm3(q1)
    if radius_final <= 0.0:
        raise AllConicKeplerDomainError(
            "reconstruction produced a nonpositive final radius"
        )
    fdot = -(mu / (radius_initial * radius_final)) * g.g1
    gdot = 1.0 - (mu / radius_final) * g.g2
    u1 = fdot * q0 + gdot * u0
    if not bool(np.all(np.isfinite(q1))) or not bool(np.all(np.isfinite(u1))):
        raise AllConicKeplerDomainError(
            "reconstruction produced a nonfinite state"
        )

    lagrange_error = abs(f_value * gdot - fdot * g_value - 1.0)
    lagrange_scale = max(1.0, abs(f_value * gdot) + abs(fdot * g_value))
    lagrange_limit = (
        WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR
        * _EPSILON
        * lagrange_scale
    )
    energy_initial = 0.5 * speed_squared - mu / radius_initial
    speed_final_squared = _dot3(u1, u1)
    energy_final = 0.5 * speed_final_squared - mu / radius_final
    energy_error = abs(energy_final - energy_initial)
    energy_scale = max(
        mu / radius_initial,
        mu / radius_final,
        0.5 * speed_squared,
        0.5 * speed_final_squared,
    )
    energy_limit = WH_KEPLER_ENERGY_ROUNDOFF_FACTOR * _EPSILON * energy_scale
    angular_initial = np.cross(q0, u0)
    angular_final = np.cross(q1, u1)
    angular_error = _norm3(angular_final - angular_initial)
    angular_scale = max(radius_initial * _norm3(u0), radius_final * _norm3(u1))
    angular_limit = (
        WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR
        * _EPSILON
        * angular_scale
    )
    checks = (
        (lagrange_error, lagrange_limit, "Lagrange identity"),
        (energy_error, energy_limit, "specific energy"),
        (angular_error, angular_limit, "angular momentum"),
    )
    for error, limit, label in checks:
        if not math.isfinite(error) or not math.isfinite(limit) or error > limit:
            raise AllConicKeplerDomainError(f"{label} postcondition failed")

    return UniversalDriftResult(
        position=q1,
        velocity=u1,
        conic=CONIC_NAMES[conic_code],
        beta=beta,
        universal_anomaly=accepted.anomaly,
        iterations=accepted_iterations,
        bracket_expansions=bracket_expansions,
        time_residual=abs(accepted.residual),
        residual_tolerance=accepted.tolerance,
        final_bracket_width=final_width,
        lagrange_identity_error=lagrange_error,
        energy_error=energy_error,
        angular_momentum_error=angular_error,
    )


CUDA_DEVICE_SOURCE = r'''
#define JX_ACK_SUCCESS 0
#define JX_ACK_INPUT_DOMAIN 1
#define JX_ACK_NONFINITE 2
#define JX_ACK_RADIUS_DOMAIN 3
#define JX_ACK_BRACKET_FAILURE 4
#define JX_ACK_ITERATION_FAILURE 5
#define JX_ACK_RECONSTRUCTION_FAILURE 6
#define JX_ACK_POSTCONDITION_FAILURE 7
#define JX_ACK_MAX_ITERATIONS 96
#define JX_ACK_MAX_EXPANSIONS 32
#define JX_ACK_SERIES_TERMS 64
#define JX_ACK_SERIES_EDGE 0.5
#define JX_ACK_HYPERBOLIC_EDGE 700.0
#define JX_ACK_EPS 2.220446049250313080847263336181640625e-16
#define JX_ACK_DBL_MAX 1.79769313486231570814527423731704357e308
#define JX_ACK_DBL_MAX_ULP 1.99584030953471981165637271303683857e292

__device__ __forceinline__ double jx_ack_ulp(const double value)
{
    const double magnitude = fabs(value);
    if (magnitude == JX_ACK_DBL_MAX) return JX_ACK_DBL_MAX_ULP;
    return nextafter(magnitude, JX_ACK_DBL_MAX) - magnitude;
}

__device__ __forceinline__ int jx_ack_g_values(
    const double beta,
    const double anomaly,
    double* g0,
    double* g1,
    double* g2,
    double* g3)
{
    if (!isfinite(beta) || !isfinite(anomaly)) return JX_ACK_NONFINITE;
    if (beta == 0.0) {
        *g0 = 1.0;
        *g1 = anomaly;
        *g2 = (anomaly * anomaly) / 2.0;
        *g3 = ((anomaly * anomaly) * anomaly) / 6.0;
    } else {
        const double root_beta = sqrt(fabs(beta));
        const double argument = root_beta * anomaly;
        if (!isfinite(argument)) return JX_ACK_NONFINITE;
        if (fabs(argument) <= JX_ACK_SERIES_EDGE) {
            double t0 = 1.0;
            double t1 = anomaly;
            double t2 = (anomaly * anomaly) / 2.0;
            double t3 = ((anomaly * anomaly) * anomaly) / 6.0;
            *g0 = t0;
            *g1 = t1;
            *g2 = t2;
            *g3 = t3;
            const double rho = ((-beta) * anomaly) * anomaly;
            for (int term = 1; term < JX_ACK_SERIES_TERMS; ++term) {
                const int twice = 2 * term;
                t0 = t0 * rho / ((double)(twice - 1) * (double)twice);
                t1 = t1 * rho / ((double)twice * (double)(twice + 1));
                t2 = t2 * rho / ((double)(twice + 1) * (double)(twice + 2));
                t3 = t3 * rho / ((double)(twice + 2) * (double)(twice + 3));
                const double prior_g0 = *g0;
                const double prior_g1 = *g1;
                const double prior_g2 = *g2;
                const double prior_g3 = *g3;
                *g0 += t0;
                *g1 += t1;
                *g2 += t2;
                *g3 += t3;
                if (!isfinite(t0) || !isfinite(t1) || !isfinite(t2)
                    || !isfinite(t3) || !isfinite(*g0) || !isfinite(*g1)
                    || !isfinite(*g2) || !isfinite(*g3)) return JX_ACK_NONFINITE;
                if (*g0 == prior_g0 && *g1 == prior_g1
                    && *g2 == prior_g2 && *g3 == prior_g3) break;
            }
        } else if (beta > 0.0) {
            const double cosine = cos(argument);
            const double sine = sin(argument);
            *g0 = cosine;
            *g1 = sine / root_beta;
            *g2 = (1.0 - cosine) / beta;
            *g3 = (anomaly - *g1) / beta;
        } else {
            if (fabs(argument) > JX_ACK_HYPERBOLIC_EDGE)
                return JX_ACK_NONFINITE;
            const double cosine = cosh(argument);
            const double sine = sinh(argument);
            const double absolute_beta = -beta;
            *g0 = cosine;
            *g1 = sine / root_beta;
            *g2 = (cosine - 1.0) / absolute_beta;
            *g3 = (*g1 - anomaly) / absolute_beta;
        }
    }
    if (!isfinite(*g0) || !isfinite(*g1) || !isfinite(*g2)
        || !isfinite(*g3)) return JX_ACK_NONFINITE;
    return JX_ACK_SUCCESS;
}

__device__ __forceinline__ int jx_ack_evaluate(
    const double anomaly,
    const double beta,
    const double radius_initial,
    const double radial_moment,
    const double mu,
    const double step,
    double* residual,
    double* radius,
    double* tolerance,
    double* g0,
    double* g1,
    double* g2,
    double* g3)
{
    const int status = jx_ack_g_values(beta, anomaly, g0, g1, g2, g3);
    if (status != JX_ACK_SUCCESS) return status;
    const double coefficient = mu - beta * radius_initial;
    const double first = radius_initial * anomaly;
    const double second = radial_moment * *g2;
    const double third = coefficient * *g3;
    *residual = ((first + second) + third) - step;
    *radius = (radius_initial + radial_moment * *g1) + coefficient * *g2;
    const double scale = fabs(first) + fabs(second) + fabs(third) + fabs(step);
    *tolerance = fmax(
        8.0 * jx_ack_ulp(step), 32.0 * JX_ACK_EPS * scale);
    if (!isfinite(*residual) || !isfinite(*radius) || !isfinite(*tolerance))
        return JX_ACK_NONFINITE;
    if (!(*radius > 0.0)) return JX_ACK_RADIUS_DOMAIN;
    return JX_ACK_SUCCESS;
}

__device__ __forceinline__ int jx_all_conic_drift_device(
    const double* input_position,
    const double* input_velocity,
    const double mu,
    const double step,
    double* output_position,
    double* output_velocity,
    int* output_conic,
    double* output_beta,
    double* output_anomaly,
    int* output_iterations,
    int* output_expansions,
    double* output_residual,
    double* output_tolerance,
    double* output_lagrange,
    double* output_energy,
    double* output_angular)
{
    const double qx = input_position[0];
    const double qy = input_position[1];
    const double qz = input_position[2];
    const double ux = input_velocity[0];
    const double uy = input_velocity[1];
    const double uz = input_velocity[2];
    output_position[0] = nan("");
    output_position[1] = nan("");
    output_position[2] = nan("");
    output_velocity[0] = nan("");
    output_velocity[1] = nan("");
    output_velocity[2] = nan("");
    *output_conic = 0;
    *output_beta = nan("");
    *output_anomaly = nan("");
    *output_iterations = 0;
    *output_expansions = 0;
    *output_residual = nan("");
    *output_tolerance = nan("");
    *output_lagrange = nan("");
    *output_energy = nan("");
    *output_angular = nan("");
    if (!isfinite(qx) || !isfinite(qy) || !isfinite(qz)
        || !isfinite(ux) || !isfinite(uy) || !isfinite(uz)
        || !isfinite(mu) || !(mu > 0.0)
        || !isfinite(step) || step == 0.0) return JX_ACK_INPUT_DOMAIN;

    const double radius_squared = (qx * qx + qy * qy) + qz * qz;
    if (!isfinite(radius_squared) || !(radius_squared > 0.0))
        return JX_ACK_INPUT_DOMAIN;
    const double radius_initial = sqrt(radius_squared);
    const double speed_squared = (ux * ux + uy * uy) + uz * uz;
    const double radial_moment = (qx * ux + qy * uy) + qz * uz;
    const double beta = (2.0 * mu / radius_initial) - speed_squared;
    if (!isfinite(speed_squared) || !isfinite(radial_moment) || !isfinite(beta)) {
        return JX_ACK_NONFINITE;
    }
    *output_beta = beta;
    *output_conic = beta > 0.0 ? 1 : (beta < 0.0 ? -1 : 0);

    double seed = step / radius_initial;
    if (beta < 0.0) {
        const double series_edge = JX_ACK_SERIES_EDGE / sqrt(-beta);
        if (fabs(seed) > series_edge) seed = copysign(series_edge, step);
    }
    if (!isfinite(seed) || seed == 0.0) {
        return JX_ACK_BRACKET_FAILURE;
    }

    double outer = seed;
    double outer_residual, outer_radius, outer_tolerance;
    double outer_g0, outer_g1, outer_g2, outer_g3;
    int status = jx_ack_evaluate(
        outer, beta, radius_initial, radial_moment, mu, step,
        &outer_residual, &outer_radius, &outer_tolerance,
        &outer_g0, &outer_g1, &outer_g2, &outer_g3);
    if (status != JX_ACK_SUCCESS) {
        return status;
    }
    int expansions = 0;
    while ((step > 0.0 && outer_residual < 0.0)
           || (step < 0.0 && outer_residual > 0.0)) {
        if (expansions >= JX_ACK_MAX_EXPANSIONS) {
            return JX_ACK_BRACKET_FAILURE;
        }
        double candidate = outer * 2.0;
        if (beta < 0.0) {
            const double maximum = JX_ACK_HYPERBOLIC_EDGE / sqrt(-beta);
            if (fabs(candidate) > maximum) candidate = copysign(maximum, step);
        }
        if (!isfinite(candidate) || candidate == outer) {
            return JX_ACK_BRACKET_FAILURE;
        }
        outer = candidate;
        ++expansions;
        status = jx_ack_evaluate(
            outer, beta, radius_initial, radial_moment, mu, step,
            &outer_residual, &outer_radius, &outer_tolerance,
            &outer_g0, &outer_g1, &outer_g2, &outer_g3);
        if (status != JX_ACK_SUCCESS) {
            return status;
        }
    }
    *output_expansions = expansions;

    double low, high, residual_low, residual_high;
    if (step > 0.0) {
        low = 0.0;
        residual_low = -step;
        high = outer;
        residual_high = outer_residual;
    } else {
        low = outer;
        residual_low = outer_residual;
        high = 0.0;
        residual_high = -step;
    }
    if (!(low < high) || !(residual_low <= 0.0)
        || !(residual_high >= 0.0)) {
        return JX_ACK_BRACKET_FAILURE;
    }
    double current = outer_residual == 0.0
        ? outer : low + 0.5 * (high - low);
    double previous = nan("");
    double two_ago = nan("");
    double accepted_g1 = nan("");
    double accepted_g2 = nan("");
    double accepted_g3 = nan("");
    double accepted_residual = nan("");
    double accepted_tolerance = nan("");
    double accepted_anomaly = nan("");
    int accepted_iteration = 0;
    for (int iteration = 1; iteration <= JX_ACK_MAX_ITERATIONS; ++iteration) {
        double residual, radius, tolerance, g0, g1, g2, g3;
        status = jx_ack_evaluate(
            current, beta, radius_initial, radial_moment, mu, step,
            &residual, &radius, &tolerance, &g0, &g1, &g2, &g3);
        if (status != JX_ACK_SUCCESS) {
            return status;
        }
        if (residual == 0.0) {
            low = current;
            high = current;
            residual_low = 0.0;
            residual_high = 0.0;
        } else if (residual < 0.0) {
            low = current;
            residual_low = residual;
        } else {
            high = current;
            residual_high = residual;
        }
        if (!(residual_low <= 0.0) || !(residual_high >= 0.0) || !(low <= high)) {
            return JX_ACK_ITERATION_FAILURE;
        }
        const double width = high - low;
        const double ulp_scale = fmax(
            jx_ack_ulp(low), fmax(jx_ack_ulp(high), jx_ack_ulp(current)));
        const bool adjacent = low == high || nextafter(low, high) == high;
        const bool settled = adjacent || width <= 8.0 * ulp_scale;
        const bool cycle = (isfinite(previous) && current == previous)
            || (isfinite(two_ago) && current == two_ago);
        if (fabs(residual) <= tolerance && (settled || cycle)) {
            accepted_g1 = g1;
            accepted_g2 = g2;
            accepted_g3 = g3;
            accepted_residual = residual;
            accepted_tolerance = tolerance;
            accepted_anomaly = current;
            accepted_iteration = iteration;
            break;
        }
        const double midpoint = low + 0.5 * (high - low);
        const double newton = current - residual / radius;
        double candidate = isfinite(newton) && low < newton && newton < high
            ? newton : midpoint;
        if (candidate == current) candidate = midpoint;
        if (!isfinite(candidate) || candidate < low || candidate > high
            || candidate == current) {
            return JX_ACK_ITERATION_FAILURE;
        }
        two_ago = previous;
        previous = current;
        current = candidate;
    }
    if (accepted_iteration == 0) {
        return JX_ACK_ITERATION_FAILURE;
    }

    const double f_value = 1.0 - (mu / radius_initial) * accepted_g2;
    const double g_value = step - mu * accepted_g3;
    const double q1x = f_value * qx + g_value * ux;
    const double q1y = f_value * qy + g_value * uy;
    const double q1z = f_value * qz + g_value * uz;
    const double radius_final_squared = (q1x * q1x + q1y * q1y) + q1z * q1z;
    if (!isfinite(radius_final_squared) || !(radius_final_squared > 0.0)) {
        return JX_ACK_RECONSTRUCTION_FAILURE;
    }
    const double radius_final = sqrt(radius_final_squared);
    const double fdot = -(mu / (radius_initial * radius_final)) * accepted_g1;
    const double gdot = 1.0 - (mu / radius_final) * accepted_g2;
    const double u1x = fdot * qx + gdot * ux;
    const double u1y = fdot * qy + gdot * uy;
    const double u1z = fdot * qz + gdot * uz;
    if (!isfinite(q1x) || !isfinite(q1y) || !isfinite(q1z)
        || !isfinite(u1x) || !isfinite(u1y) || !isfinite(u1z)) {
        return JX_ACK_RECONSTRUCTION_FAILURE;
    }

    const double lagrange = fabs(f_value * gdot - fdot * g_value - 1.0);
    const double lagrange_scale = fmax(
        1.0, fabs(f_value * gdot) + fabs(fdot * g_value));
    const double lagrange_limit = 512.0 * JX_ACK_EPS * lagrange_scale;
    const double speed_final_squared = (u1x * u1x + u1y * u1y) + u1z * u1z;
    const double energy_initial = 0.5 * speed_squared - mu / radius_initial;
    const double energy_final = 0.5 * speed_final_squared - mu / radius_final;
    const double energy_error = fabs(energy_final - energy_initial);
    const double energy_scale = fmax(
        fmax(mu / radius_initial, mu / radius_final),
        fmax(0.5 * speed_squared, 0.5 * speed_final_squared));
    const double energy_limit = 1024.0 * JX_ACK_EPS * energy_scale;
    const double h0x = qy * uz - qz * uy;
    const double h0y = qz * ux - qx * uz;
    const double h0z = qx * uy - qy * ux;
    const double h1x = q1y * u1z - q1z * u1y;
    const double h1y = q1z * u1x - q1x * u1z;
    const double h1z = q1x * u1y - q1y * u1x;
    const double dhx = h1x - h0x;
    const double dhy = h1y - h0y;
    const double dhz = h1z - h0z;
    const double angular_error = sqrt((dhx * dhx + dhy * dhy) + dhz * dhz);
    const double speed_initial = sqrt(speed_squared);
    const double speed_final = sqrt(speed_final_squared);
    const double angular_scale = fmax(
        radius_initial * speed_initial, radius_final * speed_final);
    const double angular_limit = 1024.0 * JX_ACK_EPS * angular_scale;
    if (!isfinite(lagrange) || !isfinite(energy_error)
        || !isfinite(angular_error) || lagrange > lagrange_limit
        || energy_error > energy_limit || angular_error > angular_limit) {
        return JX_ACK_POSTCONDITION_FAILURE;
    }

    output_position[0] = q1x;
    output_position[1] = q1y;
    output_position[2] = q1z;
    output_velocity[0] = u1x;
    output_velocity[1] = u1y;
    output_velocity[2] = u1z;
    *output_anomaly = accepted_anomaly;
    *output_iterations = accepted_iteration;
    *output_residual = fabs(accepted_residual);
    *output_tolerance = accepted_tolerance;
    *output_lagrange = lagrange;
    *output_energy = energy_error;
    *output_angular = angular_error;
    return JX_ACK_SUCCESS;
}
'''


CUDA_BATCH_KERNEL_SOURCE = r'''
extern "C" __global__
void jx_all_conic_universal_drift(
    const double* input_position,
    const double* input_velocity,
    const double* input_mu,
    const double* input_step,
    const int lane_count,
    double* output_position,
    double* output_velocity,
    int* output_status,
    int* output_conic,
    double* output_beta,
    double* output_anomaly,
    int* output_iterations,
    int* output_expansions,
    double* output_residual,
    double* output_tolerance,
    double* output_lagrange,
    double* output_energy,
    double* output_angular)
{
    const int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lane_count) return;
    const int offset = lane * 3;
    output_status[lane] = jx_all_conic_drift_device(
        input_position + offset,
        input_velocity + offset,
        input_mu[lane],
        input_step[lane],
        output_position + offset,
        output_velocity + offset,
        output_conic + lane,
        output_beta + lane,
        output_anomaly + lane,
        output_iterations + lane,
        output_expansions + lane,
        output_residual + lane,
        output_tolerance + lane,
        output_lagrange + lane,
        output_energy + lane,
        output_angular + lane);
}
'''


CUDA_SOURCE = CUDA_DEVICE_SOURCE + CUDA_BATCH_KERNEL_SOURCE


def _cuda_parameter(cp: Any, value: Any, count: int, label: str) -> Any:
    if isinstance(value, cp.ndarray):
        if value.dtype != cp.float64 or value.shape != (count,):
            raise AllConicKeplerDomainError(
                f"{label} must be a float64 CuPy vector with one value per lane"
            )
        return cp.ascontiguousarray(value)
    scalar = float(value)
    return cp.full(count, scalar, dtype=cp.float64)


def drift_all_conic_cuda_batch(
    cp: Any,
    positions: Any,
    velocities: Any,
    gravitational_parameters: Any,
    signed_steps: Any,
    *,
    threads_per_block: int = 128,
) -> CUDAUniversalDriftBatch:
    """Launch the matching one-thread-per-relative-orbit CUDA drift."""

    for value, label in ((positions, "positions"), (velocities, "velocities")):
        if not isinstance(value, cp.ndarray) or value.dtype != cp.float64:
            raise AllConicKeplerDomainError(
                f"{label} must be a float64 CuPy array"
            )
        if value.ndim != 2 or value.shape[1] != 3:
            raise AllConicKeplerDomainError(f"{label} must have shape (lanes, 3)")
    if velocities.shape != positions.shape:
        raise AllConicKeplerDomainError("position and velocity shapes must match")
    if type(threads_per_block) is not int or not 1 <= threads_per_block <= 1024:
        raise AllConicKeplerDomainError("threads_per_block must be in [1, 1024]")
    position = cp.ascontiguousarray(positions)
    velocity = cp.ascontiguousarray(velocities)
    count = int(position.shape[0])
    mu = _cuda_parameter(
        cp, gravitational_parameters, count, "gravitational_parameters"
    )
    step = _cuda_parameter(cp, signed_steps, count, "signed_steps")
    output_position = cp.empty_like(position)
    output_velocity = cp.empty_like(velocity)
    status = cp.empty(count, dtype=cp.int32)
    conic = cp.empty(count, dtype=cp.int32)
    beta = cp.empty(count, dtype=cp.float64)
    anomaly = cp.empty(count, dtype=cp.float64)
    iterations = cp.empty(count, dtype=cp.int32)
    expansions = cp.empty(count, dtype=cp.int32)
    diagnostics = tuple(cp.empty(count, dtype=cp.float64) for _ in range(5))
    if count:
        kernel = cp.RawKernel(
            CUDA_SOURCE,
            "jx_all_conic_universal_drift",
            options=CUDA_OPTIONS,
        )
        blocks = (count + threads_per_block - 1) // threads_per_block
        kernel(
            (blocks,),
            (threads_per_block,),
            (
                position,
                velocity,
                mu,
                step,
                cp.int32(count),
                output_position,
                output_velocity,
                status,
                conic,
                beta,
                anomaly,
                iterations,
                expansions,
                *diagnostics,
            ),
        )
    return CUDAUniversalDriftBatch(
        positions=output_position,
        velocities=output_velocity,
        status=status,
        conic=conic,
        beta=beta,
        universal_anomaly=anomaly,
        iterations=iterations,
        bracket_expansions=expansions,
        time_residual=diagnostics[0],
        residual_tolerance=diagnostics[1],
        lagrange_identity_error=diagnostics[2],
        energy_error=diagnostics[3],
        angular_momentum_error=diagnostics[4],
    )


__all__ = [
    "AllConicKeplerConvergenceError",
    "AllConicKeplerDomainError",
    "AllConicKeplerError",
    "CONIC_ELLIPTIC",
    "CONIC_HYPERBOLIC",
    "CONIC_PARABOLIC",
    "CUDA_BATCH_KERNEL_SOURCE",
    "CUDA_DEVICE_FUNCTION_NAME",
    "CUDA_DEVICE_SOURCE",
    "CUDA_OPTIONS",
    "CUDA_SOURCE",
    "CUDAUniversalDriftBatch",
    "STATUS_NAMES",
    "STATUS_SUCCESS",
    "UniversalDriftResult",
    "drift_all_conic",
    "drift_all_conic_cuda_batch",
]
