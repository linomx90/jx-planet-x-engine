from __future__ import annotations

import hashlib
import math
from pathlib import Path
import unittest

import numpy as np

import jxplanetx
from jxplanetx import _eih_1pn_cpu
from jxplanetx.gr15 import GR15ContractError, GR15Spec, STAGE_COUNT, integrate_gr15
from jxplanetx.gr15_eih_1pn import (
    EXPECTED_CORE_SOURCE_SHA256,
    EXPECTED_FORCE_CORE_SOURCE_SHA256,
    GR15EIH1PNIntegrationError,
    METHOD_ID,
    MODEL_ID,
    STATUS_WEAK_FIELD_DOMAIN,
    gr15_eih_1pn_runtime_identity,
    integrate_gr15_eih_1pn,
    prepare_gr15_eih_1pn_workspace,
)
from jxplanetx.solar_system.eih_1pn import (
    EIH1PNParameters,
    evaluate_eih_1pn_correction,
    newtonian_point_mass_accelerations,
)


ROOT = Path(__file__).resolve().parents[1]
NEWTONIAN_V3_SHA256 = (
    "036d1dc7e4e2e26c908c6dde9ff27bdf6a5804b88aa3f3e374f1f512642c1e41"
)


def _finite_mass_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = np.ascontiguousarray((1.0, 0.2), dtype=np.float64)
    relative_position = np.ascontiguousarray((1.2, 0.3, -0.1), dtype=np.float64)
    relative_velocity = np.ascontiguousarray((-0.2, 0.9, 0.15), dtype=np.float64)
    total = float(np.sum(gm))
    positions = np.ascontiguousarray(
        (
            -gm[1] / total * relative_position,
            gm[0] / total * relative_position,
        ),
        dtype=np.float64,
    )
    velocities = np.ascontiguousarray(
        (
            -gm[1] / total * relative_velocity,
            gm[0] / total * relative_velocity,
        ),
        dtype=np.float64,
    )
    return positions, velocities, gm


def _circular_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))),
        np.ascontiguousarray(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))),
        np.ascontiguousarray((0.5, 0.5)),
    )


def _eccentric_binary(
    semimajor_axis: float,
    eccentricity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    gm = np.ascontiguousarray((0.8, 0.2), dtype=np.float64)
    total_gm = float(np.sum(gm))
    period = 2.0 * math.pi * math.sqrt(semimajor_axis**3 / total_gm)
    relative_position = np.asarray(
        (semimajor_axis * (1.0 - eccentricity), 0.0, 0.0),
        dtype=np.float64,
    )
    relative_velocity = np.asarray(
        (
            0.0,
            math.sqrt(
                total_gm
                * (1.0 + eccentricity)
                / (semimajor_axis * (1.0 - eccentricity))
            ),
            0.0,
        ),
        dtype=np.float64,
    )
    positions = np.ascontiguousarray(
        (
            -gm[1] / total_gm * relative_position,
            gm[0] / total_gm * relative_position,
        )
    )
    velocities = np.ascontiguousarray(
        (
            -gm[1] / total_gm * relative_velocity,
            gm[0] / total_gm * relative_velocity,
        )
    )
    return positions, velocities, gm, period


def _first_periapsis_advance(
    speed_of_light: float,
    semimajor_axis: float,
    eccentricity: float,
) -> float:
    positions, velocities, gm, period = _eccentric_binary(
        semimajor_axis,
        eccentricity,
    )
    parameters = EIH1PNParameters(
        speed_of_light_km_s=speed_of_light,
        maximum_compactness=1.0e-2,
        maximum_speed_fraction_squared=1.0e-2,
    )

    def radial_motion(epoch: float) -> tuple[float, np.ndarray]:
        result = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            GR15Spec(
                0.0,
                epoch,
                initial_step=1.0e-4,
                minimum_step=1.0e-15,
                maximum_step=5.0e-3,
                epsilon=1.0e-10,
            ),
            parameters,
        )
        relative_position = result.positions[1] - result.positions[0]
        relative_velocity = result.velocities[1] - result.velocities[0]
        return (
            float(np.dot(relative_position, relative_velocity)),
            relative_position,
        )

    lower = 0.8 * period
    upper = 1.2 * period
    lower_motion, _ = radial_motion(lower)
    upper_motion, _ = radial_motion(upper)
    if not lower_motion < 0.0 < upper_motion:
        raise AssertionError("the first post-initial periapsis was not bracketed")
    for _ in range(42):
        midpoint = 0.5 * (lower + upper)
        midpoint_motion, _ = radial_motion(midpoint)
        if midpoint_motion < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    _, relative_position = radial_motion(0.5 * (lower + upper))
    return math.atan2(
        float(relative_position[1]),
        float(relative_position[0]),
    )


class GR15EIH1PNTests(unittest.TestCase):
    def test_native_identity_and_newtonian_v3_isolation(self) -> None:
        identity = gr15_eih_1pn_runtime_identity()
        self.assertEqual(identity["method_id"], METHOD_ID)
        self.assertEqual(identity["model_id"], MODEL_ID)
        self.assertEqual(identity["core_source_sha256"], EXPECTED_CORE_SOURCE_SHA256)
        self.assertEqual(
            identity["force_core_source_sha256"],
            EXPECTED_FORCE_CORE_SOURCE_SHA256,
        )
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(jxplanetx._gr15_eih_1pn_cpu.STAGE_COUNT, STAGE_COUNT)
        self.assertIs(jxplanetx.integrate_gr15_eih_1pn, integrate_gr15_eih_1pn)
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "src/jxplanetx/_gr15_v3_core.c").read_bytes()
            ).hexdigest(),
            NEWTONIAN_V3_SHA256,
        )

    def test_newtonian_limit_is_bitwise_identical(self) -> None:
        positions, velocities, gm = _circular_binary()
        initial_positions = positions.copy()
        initial_velocities = velocities.copy()
        spec = GR15Spec(
            0.0,
            2.0 * math.pi,
            intermediate_epochs=(math.pi,),
            initial_step=0.05,
            minimum_step=1.0e-14,
            maximum_step=0.5,
            epsilon=1.0e-9,
        )
        newtonian = integrate_gr15(positions, velocities, gm, spec)
        relativistic = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            spec,
            EIH1PNParameters(speed_of_light_km_s=1.0e30),
        )
        np.testing.assert_array_equal(
            relativistic.checkpoint_positions,
            newtonian.checkpoint_positions,
        )
        np.testing.assert_array_equal(
            relativistic.checkpoint_velocities,
            newtonian.checkpoint_velocities,
        )
        self.assertEqual(relativistic.accepted_steps, newtonian.accepted_steps)
        self.assertEqual(relativistic.rejected_steps, newtonian.rejected_steps)
        self.assertEqual(relativistic.force_evaluations, newtonian.force_evaluations)
        np.testing.assert_array_equal(positions, initial_positions)
        np.testing.assert_array_equal(velocities, initial_velocities)
        self.assertFalse(relativistic.positions.flags.writeable)
        self.assertFalse(relativistic.velocities.flags.writeable)
        self.assertFalse(relativistic.production_authorized)
        self.assertFalse(relativistic.exact_general_relativity_claimed)
        self.assertFalse(relativistic.ephemeris_equivalence_claimed)
        self.assertFalse(relativistic.general_superiority_claimed)

    def test_native_force_matches_python_equation_reference(self) -> None:
        positions, velocities, gm = _finite_mass_state()
        parameters = EIH1PNParameters(
            speed_of_light_km_s=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
        )
        native_acceleration = np.empty((1, 2, 3), dtype=np.float64)
        native_diagnostics = np.empty(2, dtype=np.float64)
        status = int(
            _eih_1pn_cpu.evaluate_batch(
                1,
                2,
                np.ascontiguousarray(positions[None, :, :]),
                np.ascontiguousarray(velocities[None, :, :]),
                gm,
                parameters.speed_of_light_km_s,
                parameters.maximum_compactness,
                parameters.maximum_speed_fraction_squared,
                native_acceleration,
                native_diagnostics,
            )
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            _eih_1pn_cpu.CORE_SOURCE_SHA256,
            EXPECTED_FORCE_CORE_SOURCE_SHA256,
        )
        reference = newtonian_point_mass_accelerations(positions, gm)
        evaluation = evaluate_eih_1pn_correction(
            positions,
            velocities,
            gm,
            parameters,
        )
        reference += evaluation.correction_accelerations_km_s2
        np.testing.assert_allclose(
            native_acceleration[0],
            reference,
            rtol=0.0,
            atol=5.0e-17,
        )
        np.testing.assert_allclose(
            native_diagnostics,
            (
                evaluation.maximum_compactness,
                evaluation.maximum_speed_fraction_squared,
            ),
            rtol=0.0,
            atol=2.0e-18,
        )

    def test_trajectory_difference_has_inverse_c_squared_limit(self) -> None:
        positions, velocities, gm = _finite_mass_state()
        spec = GR15Spec(
            0.0,
            0.1,
            initial_step=0.001,
            minimum_step=1.0e-14,
            maximum_step=0.01,
            epsilon=1.0e-10,
        )
        newtonian = integrate_gr15(positions, velocities, gm, spec)
        low_c = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            spec,
            EIH1PNParameters(
                speed_of_light_km_s=31.0,
                maximum_compactness=1.0e-2,
                maximum_speed_fraction_squared=1.0e-2,
            ),
        )
        high_c = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            spec,
            EIH1PNParameters(
                speed_of_light_km_s=62.0,
                maximum_compactness=1.0e-2,
                maximum_speed_fraction_squared=1.0e-2,
            ),
        )
        low_delta = low_c.positions - newtonian.positions
        high_delta = high_c.positions - newtonian.positions
        observed_ratio = float(np.linalg.norm(low_delta) / np.linalg.norm(high_delta))
        self.assertAlmostEqual(observed_ratio, 4.0, delta=2.0e-4)
        self.assertGreater(low_c.observed_maximum_compactness, 0.0)
        self.assertGreater(low_c.observed_maximum_speed_fraction_squared, 0.0)

    def test_two_body_periapsis_matches_analytic_1pn_limit(self) -> None:
        semimajor_axis = 1.0
        eccentricity = 0.2
        speed_of_light = 50.0
        base_advance = _first_periapsis_advance(
            speed_of_light,
            semimajor_axis,
            eccentricity,
        )
        doubled_c_advance = _first_periapsis_advance(
            2.0 * speed_of_light,
            semimajor_axis,
            eccentricity,
        )
        inverse_c_squared = 1.0 / speed_of_light**2
        richardson_leading_coefficient = (
            16.0 * doubled_c_advance - base_advance
        ) / (3.0 * inverse_c_squared)
        analytic_leading_coefficient = (
            6.0
            * math.pi
            * 1.0
            / (semimajor_axis * (1.0 - eccentricity**2))
        )
        relative_error = abs(
            richardson_leading_coefficient / analytic_leading_coefficient - 1.0
        )
        self.assertLessEqual(relative_error, 1.0e-5)

    def test_workspace_replay_and_permutation_covariance(self) -> None:
        positions, velocities, gm = _finite_mass_state()
        spec = GR15Spec(
            0.0,
            0.1,
            intermediate_epochs=(0.05,),
            initial_step=0.001,
            minimum_step=1.0e-14,
            maximum_step=0.01,
            epsilon=1.0e-10,
        )
        parameters = EIH1PNParameters(
            speed_of_light_km_s=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
        )
        workspace = prepare_gr15_eih_1pn_workspace(2, spec)
        first = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            spec,
            parameters,
            workspace=workspace,
        )
        saved_positions = first.checkpoint_positions.copy()
        saved_velocities = first.checkpoint_velocities.copy()
        saved_digest = first.replay_digest
        second = integrate_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            spec,
            parameters,
            workspace=workspace,
        )
        np.testing.assert_array_equal(second.checkpoint_positions, saved_positions)
        np.testing.assert_array_equal(second.checkpoint_velocities, saved_velocities)
        self.assertEqual(second.replay_digest, saved_digest)

        permutation = np.asarray((1, 0))
        permuted = integrate_gr15_eih_1pn(
            np.ascontiguousarray(positions[permutation]),
            np.ascontiguousarray(velocities[permutation]),
            np.ascontiguousarray(gm[permutation]),
            spec,
            parameters,
        )
        np.testing.assert_array_equal(
            permuted.checkpoint_positions,
            saved_positions[:, permutation],
        )
        np.testing.assert_array_equal(
            permuted.checkpoint_velocities,
            saved_velocities[:, permutation],
        )

    def test_forward_backward_and_tolerance_convergence(self) -> None:
        positions, velocities, gm = _finite_mass_state()
        parameters = EIH1PNParameters(
            speed_of_light_km_s=31.0,
            maximum_compactness=1.0e-2,
            maximum_speed_fraction_squared=1.0e-2,
        )

        def integrate(
            initial_positions: np.ndarray,
            initial_velocities: np.ndarray,
            initial_epoch: float,
            final_epoch: float,
            epsilon: float,
        ):
            return integrate_gr15_eih_1pn(
                initial_positions,
                initial_velocities,
                gm,
                GR15Spec(
                    initial_epoch,
                    final_epoch,
                    initial_step=1.0e-3,
                    minimum_step=1.0e-15,
                    maximum_step=0.2,
                    epsilon=epsilon,
                ),
                parameters,
            )

        reference = integrate(positions, velocities, 0.0, 4.0, 1.0e-11)
        errors: list[tuple[float, float]] = []
        for epsilon in (1.0e-8, 1.0e-9, 1.0e-10):
            candidate = integrate(positions, velocities, 0.0, 4.0, epsilon)
            errors.append(
                (
                    float(np.linalg.norm(candidate.positions - reference.positions)),
                    float(np.linalg.norm(candidate.velocities - reference.velocities)),
                )
            )
        self.assertGreater(errors[0][0], errors[1][0])
        self.assertGreater(errors[1][0], errors[2][0])
        self.assertGreater(errors[0][1], errors[1][1])
        self.assertGreater(errors[1][1], errors[2][1])
        self.assertLessEqual(errors[2][0], 0.2 * errors[0][0])
        self.assertLessEqual(errors[2][1], 0.2 * errors[0][1])

        forward = integrate(positions, velocities, 0.0, 4.0, 1.0e-9)
        backward = integrate(
            np.ascontiguousarray(forward.positions),
            np.ascontiguousarray(forward.velocities),
            4.0,
            0.0,
            1.0e-9,
        )
        self.assertLessEqual(
            float(np.linalg.norm(backward.positions - positions)),
            1.0e-12,
        )
        self.assertLessEqual(
            float(np.linalg.norm(backward.velocities - velocities)),
            1.0e-12,
        )

    def test_contract_and_weak_field_fail_closed(self) -> None:
        positions, velocities, gm = _finite_mass_state()
        spec = GR15Spec(
            0.0,
            0.01,
            initial_step=0.001,
            maximum_step=0.01,
            epsilon=1.0e-9,
        )
        with self.assertRaises(GR15EIH1PNIntegrationError) as captured:
            integrate_gr15_eih_1pn(
                positions,
                velocities,
                gm,
                spec,
                EIH1PNParameters(
                    speed_of_light_km_s=31.0,
                    maximum_compactness=1.0e-12,
                    maximum_speed_fraction_squared=1.0e-2,
                ),
            )
        self.assertEqual(captured.exception.status, STATUS_WEAK_FIELD_DOMAIN)
        self.assertEqual(captured.exception.status_name, "WEAK_FIELD_DOMAIN")

        with self.assertRaises(GR15ContractError):
            integrate_gr15_eih_1pn(
                positions.astype(np.float32),
                velocities,
                gm,
                spec,
                EIH1PNParameters(speed_of_light_km_s=31.0),
            )


if __name__ == "__main__":
    unittest.main()
