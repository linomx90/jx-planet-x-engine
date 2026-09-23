from __future__ import annotations

import math
import unittest

import numpy as np

import jxplanetx
from jxplanetx.gr15 import (
    EXPECTED_CORE_SOURCE_SHA256,
    GR15ContractError,
    GR15IntegrationError,
    GR15Spec,
    METHOD_ID,
    SCIENTIFIC_CLAIM_STATE,
    STATUS_FORCE_SINGULARITY,
    STATUS_MINIMUM_STEP,
    gr15_runtime_identity,
    gr15_tableau_arrays,
    integrate_gr15,
    prepare_gr15_workspace,
)


TAU = 2.0 * math.pi


def circular_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array(
        ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64
    )
    velocities = np.array(
        ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64
    )
    gm = np.array((0.5, 0.5), dtype=np.float64)
    return positions, velocities, gm


def analytic_circular(epochs: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((len(epochs), 2, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    for index, epoch in enumerate(epochs):
        sine = math.sin(epoch)
        cosine = math.cos(epoch)
        positions[index, 0, :2] = (-0.5 * cosine, -0.5 * sine)
        positions[index, 1, :2] = (0.5 * cosine, 0.5 * sine)
        velocities[index, 0, :2] = (0.5 * sine, -0.5 * cosine)
        velocities[index, 1, :2] = (-0.5 * sine, 0.5 * cosine)
    return positions, velocities


def invariants(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> tuple[float, np.ndarray]:
    kinetic = 0.5 * float(np.sum(gm[:, None] * velocities * velocities))
    potential = 0.0
    for first in range(len(gm)):
        for second in range(first + 1, len(gm)):
            separation = float(np.linalg.norm(positions[second] - positions[first]))
            potential -= float(gm[first] * gm[second] / separation)
    angular = np.sum(gm[:, None] * np.cross(positions, velocities), axis=0)
    return kinetic + potential, angular


class SupportedGR15Tests(unittest.TestCase):
    def test_retained_v2_fallback_is_bitwise_equivalent_to_validated_reference(self) -> None:
        from benchmarks import jx_native_gauss_radau15_prototype as reference_v1
        from benchmarks import jx_native_gauss_radau15_v2_prototype as reference_v2
        from jxplanetx import _gr15_v2_reference as retained_v2

        positions, velocities, gm = circular_binary()
        retained_spec = retained_v2.GR15Spec(
            0.0,
            TAU,
            intermediate_epochs=(TAU / 3.0, 2.0 * TAU / 3.0),
            initial_step=0.05,
            minimum_step=1.0e-14,
            maximum_step=0.5,
            epsilon=1.0e-9,
            convergence_factor=16.0,
        )
        reference_spec = reference_v1.GaussRadau15Spec(
            retained_spec.initial_epoch,
            retained_spec.final_epoch,
            intermediate_epochs=retained_spec.intermediate_epochs,
            initial_step=retained_spec.initial_step,
            minimum_step=retained_spec.minimum_step,
            maximum_step=retained_spec.maximum_step,
            epsilon=retained_spec.epsilon,
            safety_factor=retained_spec.safety_factor,
            minimum_scale_factor=retained_spec.minimum_scale_factor,
            maximum_scale_factor=retained_spec.maximum_scale_factor,
            convergence_factor=retained_spec.convergence_factor,
            maximum_iterations=retained_spec.maximum_iterations,
            maximum_steps=retained_spec.maximum_steps,
            maximum_rejections=retained_spec.maximum_rejections,
        )
        retained = retained_v2.integrate_gr15(
            positions, velocities, gm, retained_spec
        )
        reference = reference_v2.integrate(
            positions,
            velocities,
            gm,
            reference_spec,
        )
        np.testing.assert_array_equal(
            retained.checkpoint_positions, reference.checkpoint_positions
        )
        np.testing.assert_array_equal(
            retained.checkpoint_velocities, reference.checkpoint_velocities
        )
        self.assertEqual(
            (
                retained.checkpoint_accepted_steps,
                retained.checkpoint_rejected_steps,
                retained.attempted_steps,
                retained.accepted_steps,
                retained.rejected_steps,
                retained.force_evaluations,
                retained.predictor_corrector_iterations,
                retained.nonconverged_retries,
                retained.polynomial_predictor_trials,
                retained.constant_predictor_trials,
                retained.history_commits,
                retained.maximum_error_ratio.hex(),
                retained.minimum_accepted_step.hex(),
                retained.maximum_accepted_step.hex(),
                retained.final_proposed_step.hex(),
            ),
            (
                reference.checkpoint_accepted_steps,
                reference.checkpoint_rejected_steps,
                reference.attempted_steps,
                reference.accepted_steps,
                reference.rejected_steps,
                reference.force_evaluations,
                reference.predictor_corrector_iterations,
                reference.nonconverged_retries,
                reference.polynomial_predictor_trials,
                reference.constant_predictor_trials,
                reference.history_commits,
                reference.maximum_error_ratio.hex(),
                reference.minimum_accepted_step.hex(),
                reference.maximum_accepted_step.hex(),
                reference.final_proposed_step.hex(),
            ),
        )

    def test_packaged_identity_tableau_and_lazy_top_level_exports(self) -> None:
        identity = gr15_runtime_identity()
        self.assertEqual(identity["method_id"], METHOD_ID)
        self.assertEqual(identity["source_sha256"], EXPECTED_CORE_SOURCE_SHA256)
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")
        self.assertNotIn("fPIC", identity["compiler_flags"])
        self.assertIs(jxplanetx.GR15Spec, GR15Spec)
        self.assertIs(jxplanetx.integrate_gr15, integrate_gr15)

        nodes, weights, matrix = gr15_tableau_arrays()
        self.assertEqual(nodes.shape, (8,))
        self.assertEqual(weights.shape, (8,))
        self.assertEqual(matrix.shape, (8, 8))
        self.assertFalse(nodes.flags.writeable)
        self.assertFalse(weights.flags.writeable)
        self.assertFalse(matrix.flags.writeable)
        self.assertEqual(float(nodes[0]), 0.0)
        self.assertTrue(np.all(np.diff(nodes) > 0.0))
        for degree in range(15):
            self.assertAlmostEqual(
                float(np.sum(weights * nodes**degree)),
                1.0 / (degree + 1),
                delta=3.0e-15,
            )
        for degree in range(8):
            np.testing.assert_allclose(
                matrix @ (nodes**degree),
                nodes ** (degree + 1) / (degree + 1),
                rtol=0.0,
                atol=3.0e-15,
            )

    def test_analytic_accuracy_conservation_and_claim_boundary(self) -> None:
        positions, velocities, gm = circular_binary()
        initial_positions = positions.copy()
        initial_velocities = velocities.copy()
        initial_energy, initial_angular = invariants(positions, velocities, gm)
        spec = GR15Spec(
            0.0,
            100.0 * TAU,
            intermediate_epochs=(TAU,),
            initial_step=0.05,
            minimum_step=1.0e-14,
            maximum_step=0.5,
            epsilon=1.0e-9,
        )
        result = integrate_gr15(positions, velocities, gm, spec)
        expected_positions, expected_velocities = analytic_circular(
            spec.checkpoint_epochs
        )
        self.assertLessEqual(
            float(
                np.max(
                    np.linalg.norm(
                        result.checkpoint_positions - expected_positions,
                        axis=2,
                    )
                )
            ),
            2.0e-9,
        )
        self.assertLessEqual(
            float(
                np.max(
                    np.linalg.norm(
                        result.checkpoint_velocities - expected_velocities,
                        axis=2,
                    )
                )
            ),
            2.0e-9,
        )
        final_energy, final_angular = invariants(result.positions, result.velocities, gm)
        self.assertLessEqual(
            abs((final_energy - initial_energy) / initial_energy), 2.0e-11
        )
        self.assertLessEqual(
            float(
                np.linalg.norm(final_angular - initial_angular)
                / np.linalg.norm(initial_angular)
            ),
            2.0e-11,
        )
        np.testing.assert_array_equal(positions, initial_positions)
        np.testing.assert_array_equal(velocities, initial_velocities)
        self.assertEqual(result.method_id, METHOD_ID)
        self.assertEqual(result.scientific_claim_state, SCIENTIFIC_CLAIM_STATE)
        self.assertFalse(result.production_authorized)
        self.assertFalse(result.scientific_qualification_claimed)
        self.assertFalse(result.positions.flags.writeable)
        self.assertFalse(result.velocities.flags.writeable)
        self.assertFalse(result.checkpoint_positions.flags.writeable)
        self.assertFalse(result.checkpoint_velocities.flags.writeable)
        self.assertEqual(result.history_commits, result.accepted_steps)
        self.assertEqual(
            result.terminal_force_reuses + result.terminal_force_sweeps,
            result.attempted_steps - result.nonconverged_retries,
        )
        self.assertLessEqual(
            result.maximum_corrector_residual, result.corrector_threshold
        )
        self.assertEqual(
            result.polynomial_predictor_trials + result.constant_predictor_trials,
            result.attempted_steps,
        )
        self.assertGreater(result.polynomial_predictor_trials, 0)
        self.assertGreater(result.constant_predictor_trials, 0)

    def test_workspace_determinism_and_reverse_time(self) -> None:
        positions, velocities, gm = circular_binary()
        spec = GR15Spec(
            0.0,
            TAU,
            intermediate_epochs=(TAU / 4.0, TAU / 2.0),
            initial_step=0.05,
            minimum_step=1.0e-14,
            maximum_step=0.5,
            epsilon=1.0e-9,
        )
        workspace = prepare_gr15_workspace(2, spec)
        first = integrate_gr15(positions, velocities, gm, spec, workspace=workspace)
        first_positions = first.checkpoint_positions.copy()
        first_velocities = first.checkpoint_velocities.copy()
        first_digest = first.replay_digest
        second = integrate_gr15(positions, velocities, gm, spec, workspace=workspace)
        np.testing.assert_array_equal(second.checkpoint_positions, first_positions)
        np.testing.assert_array_equal(second.checkpoint_velocities, first_velocities)
        self.assertEqual(second.replay_digest, first_digest)

        backward_spec = GR15Spec(
            TAU,
            0.0,
            initial_step=0.05,
            minimum_step=1.0e-14,
            maximum_step=0.5,
            epsilon=1.0e-9,
        )
        backward = integrate_gr15(
            np.ascontiguousarray(second.positions),
            np.ascontiguousarray(second.velocities),
            gm,
            backward_spec,
        )
        self.assertLessEqual(
            float(np.max(np.abs(backward.positions - positions))), 2.0e-10
        )
        self.assertLessEqual(
            float(np.max(np.abs(backward.velocities - velocities))), 2.0e-10
        )

    def test_maximum_supported_body_count_executes_native_core(self) -> None:
        body_count = 32
        phases = 2.0 * math.pi * np.arange(body_count, dtype=np.float64) / body_count
        positions = np.zeros((body_count, 3), dtype=np.float64)
        positions[:, 0] = 10.0 * np.cos(phases)
        positions[:, 1] = 10.0 * np.sin(phases)
        velocities = np.zeros_like(positions)
        gm = np.full(body_count, 1.0e-6, dtype=np.float64)
        result = integrate_gr15(
            positions,
            velocities,
            gm,
            GR15Spec(
                0.0,
                1.0e-3,
                initial_step=1.0e-4,
                minimum_step=1.0e-14,
                maximum_step=1.0e-3,
                epsilon=1.0e-8,
            ),
        )
        self.assertGreater(result.accepted_steps, 0)
        self.assertEqual(result.rejected_steps, 0)
        self.assertTrue(np.all(np.isfinite(result.positions)))
        self.assertTrue(np.all(np.isfinite(result.velocities)))

    def test_failures_are_explicit_and_fail_closed(self) -> None:
        positions, velocities, gm = circular_binary()
        spec = GR15Spec(
            0.0,
            1.0,
            initial_step=0.5,
            minimum_step=0.5,
            maximum_step=0.5,
            epsilon=1.0e-9,
            maximum_iterations=1,
        )
        with self.assertRaises(GR15IntegrationError) as captured:
            integrate_gr15(positions, velocities, gm, spec)
        self.assertEqual(captured.exception.status, STATUS_MINIMUM_STEP)
        self.assertEqual(captured.exception.status_name, "MINIMUM_STEP")

        coincident = positions.copy()
        coincident[1] = coincident[0]
        with self.assertRaises(GR15IntegrationError) as captured:
            integrate_gr15(coincident, velocities, gm, spec)
        self.assertEqual(captured.exception.status, STATUS_FORCE_SINGULARITY)
        self.assertEqual(captured.exception.status_name, "FORCE_SINGULARITY")

        with self.assertRaises(GR15ContractError):
            GR15Spec(0.0, 1.0, intermediate_epochs=(0.75, 0.5))
        with self.assertRaises(GR15ContractError):
            integrate_gr15(positions.astype(np.float32), velocities, gm, spec)
        invalid_gm = gm.copy()
        invalid_gm[0] = math.inf
        with self.assertRaises(GR15ContractError):
            integrate_gr15(positions, velocities, invalid_gm, spec)


if __name__ == "__main__":
    unittest.main()
