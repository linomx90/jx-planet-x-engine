from __future__ import annotations

import math
import unittest

import numpy as np

from jxplanetx._all_conic_kepler import (
    CUDA_DEVICE_FUNCTION_NAME,
    CUDA_DEVICE_SOURCE,
    CUDA_OPTIONS,
    STATUS_INPUT_DOMAIN,
    STATUS_SUCCESS,
    AllConicKeplerDomainError,
    _universal_g_values,
    drift_all_conic,
    drift_all_conic_cuda_batch,
)
from jxplanetx.engine.wisdom_holman import _solve_elliptic_universal_kepler
from jxplanetx.engine.wisdom_holman_contracts import UniversalKeplerSolverSpec


def _cupy_or_none():
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
        return cp
    except Exception:
        return None


CP = _cupy_or_none()
GPU_REQUIRED = unittest.skipIf(CP is None, "CUDA/CuPy device is unavailable")
FLOAT64_EPSILON = np.finfo(np.float64).eps


# Independently generated with a 100-decimal-digit conic oracle, then rounded
# once to binary64.  These deliberately share q, mu, and a long dt while
# straddling beta=0 at one-ULP-scale velocity offsets.
ORACLE_CASES = (
    (
        "eccentricity_0.96",
        float.fromhex("0x1.fadaa8f7eed52p-1"),
        1,
        (-float.fromhex("0x1.f6bc8e5cba992p+3"),
         float.fromhex("0x1.560e0cf1b8ed1p+3"), 0.0),
        (-float.fromhex("0x1.22f042d09540dp-2"),
         float.fromhex("0x1.139bc441bc676p-4"), 0.0),
    ),
    (
        "exact_parabola",
        float.fromhex("0x1p+0"),
        0,
        (-16.0, 12.0, 0.0),
        (-0.3, 0.1, 0.0),
    ),
    (
        "near_parabolic_elliptic",
        float.fromhex("0x1.fffffffffe000p-1"),
        1,
        (-float.fromhex("0x1.fffffffffcc29p+3"),
         float.fromhex("0x1.7fffffffefeb8p+3"), 0.0),
        (-float.fromhex("0x1.333333332d694p-2"),
         float.fromhex("0x1.99999999663bdp-4"), 0.0),
    ),
    (
        "near_parabolic_hyperbolic",
        float.fromhex("0x1.0000000001000p+0"),
        -1,
        (-float.fromhex("0x1.00000000019ecp+4"),
         float.fromhex("0x1.8000000010148p+3"), 0.0),
        (-float.fromhex("0x1.3333333338fd2p-2"),
         float.fromhex("0x1.99999999ccf76p-4"), 0.0),
    ),
    (
        "eccentricity_1.5",
        float.fromhex("0x1.1e3779b97f4a8p+0"),
        -1,
        (-float.fromhex("0x1.0d3384ecb131ap+4"),
         float.fromhex("0x1.91fd1d5df9d21p+4"), 0.0),
        (-float.fromhex("0x1.7c817c8490d03p-2"),
         float.fromhex("0x1.b01afa5977b8dp-2"), 0.0),
    ),
)


def _assert_scaled(
    testcase: unittest.TestCase,
    actual: np.ndarray,
    expected: np.ndarray,
    epsilon_factor: float,
) -> None:
    scale = max(1.0, float(np.max(np.abs(expected))))
    error = float(np.max(np.abs(actual - expected)))
    testcase.assertLessEqual(error, epsilon_factor * FLOAT64_EPSILON * scale)


def _newton(function, derivative, seed: float) -> float:
    value = seed
    for _ in range(64):
        candidate = value - function(value) / derivative(value)
        if candidate == value or abs(candidate - value) <= 4.0 * math.ulp(
            abs(candidate)
        ):
            return candidate
        value = candidate
    raise AssertionError("independent anomaly solve did not converge")


def _elliptic_reference(
    eccentricity: float, periapse: float, mu: float, step: float
) -> tuple[np.ndarray, np.ndarray]:
    semimajor_axis = periapse / (1.0 - eccentricity)
    mean_motion = math.sqrt(mu / semimajor_axis**3)
    mean_anomaly = mean_motion * step
    anomaly = _newton(
        lambda value: value - eccentricity * math.sin(value) - mean_anomaly,
        lambda value: 1.0 - eccentricity * math.cos(value),
        mean_anomaly / (1.0 - eccentricity),
    )
    cosine = math.cos(anomaly)
    sine = math.sin(anomaly)
    root = math.sqrt(1.0 - eccentricity**2)
    anomaly_rate = mean_motion / (1.0 - eccentricity * cosine)
    return (
        np.array(
            (
                semimajor_axis * (cosine - eccentricity),
                semimajor_axis * root * sine,
                0.0,
            )
        ),
        np.array(
            (
                -semimajor_axis * sine * anomaly_rate,
                semimajor_axis * root * cosine * anomaly_rate,
                0.0,
            )
        ),
    )


def _hyperbolic_reference(
    eccentricity: float, periapse: float, mu: float, step: float
) -> tuple[np.ndarray, np.ndarray]:
    semimajor_magnitude = periapse / (eccentricity - 1.0)
    mean_motion = math.sqrt(mu / semimajor_magnitude**3)
    mean_anomaly = mean_motion * step
    anomaly = _newton(
        lambda value: eccentricity * math.sinh(value) - value - mean_anomaly,
        lambda value: eccentricity * math.cosh(value) - 1.0,
        math.asinh(mean_anomaly / eccentricity),
    )
    cosine = math.cosh(anomaly)
    sine = math.sinh(anomaly)
    root = math.sqrt(eccentricity**2 - 1.0)
    anomaly_rate = mean_motion / (eccentricity * cosine - 1.0)
    return (
        np.array(
            (
                semimajor_magnitude * (eccentricity - cosine),
                semimajor_magnitude * root * sine,
                0.0,
            )
        ),
        np.array(
            (
                -semimajor_magnitude * sine * anomaly_rate,
                semimajor_magnitude * root * cosine * anomaly_rate,
                0.0,
            )
        ),
    )


def _parabolic_reference(
    periapse: float, mu: float, step: float
) -> tuple[np.ndarray, np.ndarray]:
    semilatus_rectum = 2.0 * periapse
    timescale = 0.5 * math.sqrt(semilatus_rectum**3 / mu)
    barker = _newton(
        lambda value: timescale * (value + value**3 / 3.0) - step,
        lambda value: timescale * (1.0 + value**2),
        step / timescale,
    )
    factor = math.sqrt(mu / semilatus_rectum)
    denominator = 1.0 + barker**2
    return (
        np.array(
            (periapse * (1.0 - barker**2), 2.0 * periapse * barker, 0.0)
        ),
        np.array(
            (
                -2.0 * barker * factor / denominator,
                2.0 * factor / denominator,
                0.0,
            )
        ),
    )


class AllConicCPUReferenceTests(unittest.TestCase):
    def test_circular_known_answer_for_both_time_directions(self):
        for step in (0.3, -0.3, 1.2, -1.2):
            with self.subTest(step=step):
                result = drift_all_conic((1, 0, 0), (0, 1, 0), 1.0, step)
                expected_position = np.array(
                    (math.cos(step), math.sin(step), 0.0)
                )
                expected_velocity = np.array(
                    (-math.sin(step), math.cos(step), 0.0)
                )
                np.testing.assert_allclose(
                    result.position, expected_position, rtol=0.0, atol=5e-16
                )
                np.testing.assert_allclose(
                    result.velocity, expected_velocity, rtol=0.0, atol=5e-16
                )
                self.assertEqual(result.conic, "elliptic")

    def test_eccentricity_point_96_matches_eccentric_anomaly_solution(self):
        eccentricity = 0.96
        periapse = 1.0
        mu = 1.0
        step = 2.0
        expected_position, expected_velocity = _elliptic_reference(
            eccentricity, periapse, mu, step
        )
        result = drift_all_conic(
            (periapse, 0.0, 0.0),
            (0.0, math.sqrt(mu * (1.0 + eccentricity) / periapse), 0.0),
            mu,
            step,
        )
        np.testing.assert_allclose(
            result.position, expected_position, rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            result.velocity, expected_velocity, rtol=0.0, atol=1e-14
        )
        self.assertAlmostEqual(result.beta, 0.04, delta=4e-16)

    def test_exact_parabola_matches_barker_solution(self):
        # mu=.5, q=1, v=1 makes beta=2*mu/q-v**2 exactly zero in binary64.
        mu = 0.5
        step = 0.7
        expected_position, expected_velocity = _parabolic_reference(1.0, mu, step)
        result = drift_all_conic((1, 0, 0), (0, 1, 0), mu, step)
        self.assertEqual(result.conic, "parabolic")
        self.assertEqual(result.beta, 0.0)
        np.testing.assert_allclose(
            result.position, expected_position, rtol=0.0, atol=3e-16
        )
        np.testing.assert_allclose(
            result.velocity, expected_velocity, rtol=0.0, atol=3e-16
        )

    def test_hyperbola_matches_hyperbolic_anomaly_solution(self):
        eccentricity = 1.5
        periapse = 1.0
        mu = 1.0
        step = 10.0
        expected_position, expected_velocity = _hyperbolic_reference(
            eccentricity, periapse, mu, step
        )
        result = drift_all_conic(
            (periapse, 0.0, 0.0),
            (0.0, math.sqrt(mu * (1.0 + eccentricity) / periapse), 0.0),
            mu,
            step,
        )
        self.assertEqual(result.conic, "hyperbolic")
        self.assertGreater(result.bracket_expansions, 0)
        np.testing.assert_allclose(
            result.position, expected_position, rtol=0.0, atol=3e-15
        )
        np.testing.assert_allclose(
            result.velocity, expected_velocity, rtol=0.0, atol=5e-16
        )

    def test_long_step_independent_high_precision_oracle_vectors(self):
        initial_position = np.array((2.0, 0.0, 0.0))
        for name, periapse_speed, conic_sign, position, velocity in ORACLE_CASES:
            with self.subTest(name=name):
                initial_velocity = np.array((0.0, periapse_speed, 0.0))
                expected_position = np.array(position)
                expected_velocity = np.array(velocity)
                forward = drift_all_conic(
                    initial_position, initial_velocity, 1.0, 48.0
                )
                expected_conic = {
                    1: "elliptic",
                    0: "parabolic",
                    -1: "hyperbolic",
                }[conic_sign]
                self.assertEqual(forward.conic, expected_conic)
                self.assertEqual(
                    (forward.beta > 0.0) - (forward.beta < 0.0), conic_sign
                )
                _assert_scaled(self, forward.position, expected_position, 64.0)
                _assert_scaled(self, forward.velocity, expected_velocity, 64.0)
                reverse = drift_all_conic(
                    forward.position, forward.velocity, 1.0, -48.0
                )
                _assert_scaled(self, reverse.position, initial_position, 256.0)
                _assert_scaled(self, reverse.velocity, initial_velocity, 256.0)
                if conic_sign:
                    self.assertEqual(
                        (reverse.beta > 0.0) - (reverse.beta < 0.0), conic_sign
                    )

    def test_near_parabolic_limit_is_continuous_from_both_sides(self):
        exact = drift_all_conic(
            (1.0, 0.0, 0.0), (0.0, math.sqrt(2.0), 0.0), 1.0, 0.4
        )
        for eccentricity, conic in (
            (1.0 - 1e-12, "elliptic"),
            (1.0 + 1e-12, "hyperbolic"),
        ):
            with self.subTest(eccentricity=eccentricity):
                result = drift_all_conic(
                    (1.0, 0.0, 0.0),
                    (0.0, math.sqrt(1.0 + eccentricity), 0.0),
                    1.0,
                    0.4,
                )
                self.assertEqual(result.conic, conic)
                np.testing.assert_allclose(
                    result.position, exact.position, rtol=0.0, atol=4e-13
                )
                np.testing.assert_allclose(
                    result.velocity, exact.velocity, rtol=0.0, atol=4e-13
                )

    def test_elliptic_overlap_matches_existing_wisdom_holman_solver(self):
        position = np.array((0.3, 0.0, 0.0), dtype=np.float64)
        velocity = np.array(
            (0.0, math.sqrt(1.7 / 0.3), 0.0), dtype=np.float64
        )
        existing = _solve_elliptic_universal_kepler(
            position, velocity, 1.0, 0.1, UniversalKeplerSolverSpec()
        )
        all_conic = drift_all_conic(position, velocity, 1.0, 0.1)
        np.testing.assert_allclose(
            all_conic.position, existing.position, rtol=0.0, atol=2e-15
        )
        np.testing.assert_allclose(
            all_conic.velocity, existing.velocity, rtol=0.0, atol=3e-15
        )

    def test_forward_backward_and_invariants_for_every_conic(self):
        cases = (
            (1.0, 1.0),
            (1.0, math.sqrt(1.96)),
            (0.5, 1.0),
            (1.0, math.sqrt(2.5)),
        )
        for mu, tangential_speed in cases:
            with self.subTest(mu=mu, tangential_speed=tangential_speed):
                initial_position = np.array((1.0, 0.0, 0.0))
                initial_velocity = np.array((0.0, tangential_speed, 0.0))
                forward = drift_all_conic(
                    initial_position, initial_velocity, mu, 0.7
                )
                backward = drift_all_conic(
                    forward.position, forward.velocity, mu, -0.7
                )
                np.testing.assert_allclose(
                    backward.position, initial_position, rtol=0.0, atol=2e-15
                )
                np.testing.assert_allclose(
                    backward.velocity, initial_velocity, rtol=0.0, atol=2e-15
                )
                self.assertLessEqual(
                    forward.time_residual, forward.residual_tolerance
                )
                self.assertLessEqual(forward.lagrange_identity_error, 3e-15)
                self.assertLessEqual(forward.energy_error, 5e-15)
                self.assertLessEqual(forward.angular_momentum_error, 5e-15)

    def test_three_dimensional_rotation_covariance(self):
        rotation = np.array(
            (
                (0.0, -1.0, 0.0),
                (1.0 / math.sqrt(2.0), 0.0, -1.0 / math.sqrt(2.0)),
                (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)),
            )
        )
        position = np.array((1.0, 0.0, 0.0))
        velocity = np.array((0.0, math.sqrt(2.5), 0.0))
        planar = drift_all_conic(position, velocity, 1.0, 0.4)
        rotated = drift_all_conic(
            rotation @ position, rotation @ velocity, 1.0, 0.4
        )
        np.testing.assert_allclose(
            rotated.position, rotation @ planar.position, rtol=0.0, atol=7e-16
        )
        np.testing.assert_allclose(
            rotated.velocity, rotation @ planar.velocity, rtol=0.0, atol=7e-16
        )

    def test_series_boundary_matches_closed_forms_for_both_beta_signs(self):
        for beta, scalar_functions in (
            (
                1.0,
                (
                    math.cos,
                    math.sin,
                    lambda x: 1.0 - math.cos(x),
                    lambda x: x - math.sin(x),
                ),
            ),
            (
                -1.0,
                (
                    math.cosh,
                    math.sinh,
                    lambda x: math.cosh(x) - 1.0,
                    lambda x: math.sinh(x) - x,
                ),
            ),
        ):
            for anomaly in (
                math.nextafter(0.5, 0.0),
                0.5,
                math.nextafter(0.5, math.inf),
                -0.5,
            ):
                with self.subTest(beta=beta, anomaly=anomaly):
                    values = _universal_g_values(beta, anomaly)
                    expected = tuple(function(anomaly) for function in scalar_functions)
                    np.testing.assert_allclose(
                        (values.g0, values.g1, values.g2, values.g3),
                        expected,
                        rtol=0.0,
                        atol=3e-16,
                    )
                    self.assertEqual(values.used_series, abs(anomaly) <= 0.5)

    def test_series_early_exit_is_bitwise_equal_to_fixed_64_term_contract(self):
        def fixed_series(beta: float, anomaly: float) -> tuple[float, ...]:
            terms = [
                1.0,
                anomaly,
                anomaly * anomaly / 2.0,
                anomaly * anomaly * anomaly / 6.0,
            ]
            totals = terms.copy()
            rho = -beta * anomaly * anomaly
            for term_index in range(1, 64):
                twice = 2 * term_index
                for order in range(4):
                    denominator = (twice + order - 1) * (twice + order)
                    terms[order] = terms[order] * rho / float(denominator)
                    totals[order] += terms[order]
            return tuple(totals)

        for beta in (-100.0, -1.0, -1.0e-12, 1.0e-12, 1.0, 100.0):
            root = math.sqrt(abs(beta))
            for argument in (-0.5, -0.1, -1.0e-6, 1.0e-6, 0.1, 0.5):
                anomaly = argument / root
                with self.subTest(beta=beta, argument=argument):
                    actual = _universal_g_values(beta, anomaly)
                    self.assertEqual(
                        (actual.g0, actual.g1, actual.g2, actual.g3),
                        fixed_series(beta, anomaly),
                    )

    def test_invalid_inputs_fail_closed(self):
        bad_requests = (
            ((0, 0, 0), (0, 1, 0), 1.0, 0.1),
            ((1, 0, 0), (0, 1, 0), 0.0, 0.1),
            ((1, 0, 0), (0, 1, 0), 1.0, 0.0),
            ((1, 0, math.inf), (0, 1, 0), 1.0, 0.1),
        )
        for request in bad_requests:
            with self.subTest(request=request), self.assertRaises(
                AllConicKeplerDomainError
            ):
                drift_all_conic(*request)

    def test_device_source_is_composable_without_a_global_kernel(self):
        self.assertEqual(CUDA_DEVICE_FUNCTION_NAME, "jx_all_conic_drift_device")
        self.assertIn(f"int {CUDA_DEVICE_FUNCTION_NAME}(", CUDA_DEVICE_SOURCE)
        self.assertIn("__device__ __forceinline__", CUDA_DEVICE_SOURCE)
        self.assertNotIn("__global__", CUDA_DEVICE_SOURCE)
        self.assertEqual(CUDA_OPTIONS, ("--std=c++11", "--fmad=false"))
        self.assertNotRegex(CUDA_DEVICE_SOURCE, r"\bfloat\b")
        spec = UniversalKeplerSolverSpec()
        expected_directives = (
            f"#define JX_ACK_MAX_ITERATIONS {spec.maximum_iterations}",
            f"#define JX_ACK_MAX_EXPANSIONS {spec.maximum_bracket_expansions}",
            f"#define JX_ACK_SERIES_TERMS {spec.series_term_count}",
            f"#define JX_ACK_SERIES_EDGE {spec.series_switch_abs_argument}",
        )
        for directive in expected_directives:
            self.assertIn(directive, CUDA_DEVICE_SOURCE)


class AllConicCUDAQualificationTests(unittest.TestCase):
    @GPU_REQUIRED
    def test_mixed_conic_warp_matches_independent_oracle_and_repeats(self):
        cp = CP
        count = len(ORACLE_CASES)
        positions = np.zeros((count, 3), dtype=np.float64)
        velocities = np.zeros_like(positions)
        positions[:, 0] = 2.0
        velocities[:, 1] = [case[1] for case in ORACLE_CASES]
        mus = np.ones(count, dtype=np.float64)
        steps = np.full(count, 48.0, dtype=np.float64)
        expected_conic = np.array(
            [case[2] for case in ORACLE_CASES], dtype=np.int32
        )
        expected_position = np.array(
            [case[3] for case in ORACLE_CASES], dtype=np.float64
        )
        expected_velocity = np.array(
            [case[4] for case in ORACLE_CASES], dtype=np.float64
        )
        first = drift_all_conic_cuda_batch(
            cp,
            cp.asarray(positions),
            cp.asarray(velocities),
            cp.asarray(mus),
            cp.asarray(steps),
            threads_per_block=32,
        )
        second = drift_all_conic_cuda_batch(
            cp,
            cp.asarray(positions),
            cp.asarray(velocities),
            cp.asarray(mus),
            cp.asarray(steps),
            threads_per_block=32,
        )
        np.testing.assert_array_equal(cp.asnumpy(first.status), STATUS_SUCCESS)
        np.testing.assert_array_equal(cp.asnumpy(first.conic), expected_conic)
        for lane in range(count):
            _assert_scaled(
                self,
                cp.asnumpy(first.positions[lane]),
                expected_position[lane],
                256.0,
            )
            _assert_scaled(
                self,
                cp.asnumpy(first.velocities[lane]),
                expected_velocity[lane],
                256.0,
            )
        self.assertTrue(bool(cp.array_equal(first.positions, second.positions)))
        self.assertTrue(bool(cp.array_equal(first.velocities, second.velocities)))
        self.assertTrue(bool(cp.array_equal(first.status, second.status)))
        reverse = drift_all_conic_cuda_batch(
            cp,
            first.positions,
            first.velocities,
            cp.asarray(mus),
            cp.asarray(-steps),
            threads_per_block=32,
        )
        np.testing.assert_array_equal(cp.asnumpy(reverse.status), STATUS_SUCCESS)
        for lane in range(count):
            _assert_scaled(
                self,
                cp.asnumpy(reverse.positions[lane]),
                positions[lane],
                256.0,
            )
            _assert_scaled(
                self,
                cp.asnumpy(reverse.velocities[lane]),
                velocities[lane],
                256.0,
            )

    @GPU_REQUIRED
    def test_gpu_invalid_lane_returns_status_and_nan_state(self):
        cp = CP
        result = drift_all_conic_cuda_batch(
            cp,
            cp.asarray(((0.0, 0.0, 0.0),), dtype=cp.float64),
            cp.asarray(((0.0, 1.0, 0.0),), dtype=cp.float64),
            1.0,
            0.1,
        )
        self.assertEqual(int(cp.asnumpy(result.status)[0]), STATUS_INPUT_DOMAIN)
        self.assertTrue(bool(cp.all(cp.isnan(result.positions))))
        self.assertTrue(bool(cp.all(cp.isnan(result.velocities))))


if __name__ == "__main__":
    unittest.main()
