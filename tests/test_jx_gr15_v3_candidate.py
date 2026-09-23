from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

from benchmarks import jx_gr15_ias15 as comparison
from benchmarks.jx_gr15_v3_calibration import _spec, _workloads
from jxplanetx import _gr15_v2_reference as v2
from jxplanetx import _gr15_v3_candidate as v3
from jxplanetx.gr15 import (
    GR15IntegrationError,
    GR15Spec,
    STATUS_FORCE_SINGULARITY,
    STATUS_MINIMUM_STEP,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_SHA256 = (
    "64c5e9acd9b0a8cfe643c9be0dc2209d3503302dccbb4600973f4c3513d04026"
)


class GR15V3CandidateTests(unittest.TestCase):
    def test_preregistration_identity(self) -> None:
        path = ROOT / "benchmarks" / "jx_gr15_v3_preregistration.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            PREREGISTRATION_SHA256,
        )
        identity = v3.runtime_identity()
        self.assertEqual(identity["method_id"], "JX_GAUSS_RADAU15_V3")
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")

    def test_calibration_accuracy_reverse_time_and_work_reduction(self) -> None:
        for name, workload in _workloads().items():
            with self.subTest(workload=name):
                positions, velocities, gm, epochs = comparison._fixture(workload)
                reference_positions, reference_velocities = comparison._analytic_binary(
                    epochs, gm, float(workload["eccentricity"])
                )
                baseline_spec = v2.GR15Spec(
                    epochs[0],
                    epochs[-1],
                    intermediate_epochs=epochs[1:-1],
                    initial_step=float(workload["initial_step"]),
                    minimum_step=1.0e-15,
                    maximum_step=float(workload["maximum_step"]),
                    epsilon=1.0e-6,
                    convergence_factor=16.0,
                )
                candidate_spec = _spec(epochs, workload, 256.0)
                baseline = v2.integrate_gr15(
                    positions, velocities, gm, baseline_spec
                )
                candidate = v3.integrate(positions, velocities, gm, candidate_spec)
                repeat = v3.integrate(positions, velocities, gm, candidate_spec)
                self.assertEqual(candidate.replay_digest, repeat.replay_digest)
                np.testing.assert_array_equal(
                    candidate.checkpoint_positions, repeat.checkpoint_positions
                )
                np.testing.assert_array_equal(
                    candidate.checkpoint_velocities, repeat.checkpoint_velocities
                )
                metrics = comparison._metrics(
                    candidate.checkpoint_positions,
                    candidate.checkpoint_velocities,
                    reference_positions,
                    reference_velocities,
                    gm,
                )
                self.assertTrue(comparison._passes(metrics, workload["accuracy_gate"]))
                self.assertLessEqual(
                    candidate.force_evaluations, 0.8 * baseline.force_evaluations
                )
                self.assertLessEqual(candidate.rejected_steps, baseline.rejected_steps)
                self.assertEqual(
                    candidate.terminal_force_reuses
                    + candidate.terminal_force_sweeps,
                    candidate.attempted_steps - candidate.nonconverged_retries,
                )
                self.assertLessEqual(
                    candidate.maximum_corrector_residual,
                    candidate.corrector_threshold,
                )
                backward_epochs = tuple(reversed(epochs))
                backward = v3.integrate(
                    np.ascontiguousarray(candidate.positions),
                    np.ascontiguousarray(candidate.velocities),
                    gm,
                    _spec(backward_epochs, workload, 256.0),
                )
                self.assertLessEqual(
                    float(np.max(np.abs(backward.positions - positions))), 2.0e-9
                )
                self.assertLessEqual(
                    float(np.max(np.abs(backward.velocities - velocities))), 2.0e-9
                )

    def test_failure_domain_parity_and_maximum_body_count(self) -> None:
        positions = np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64)
        velocities = np.array(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64)
        gm = np.array((0.5, 0.5), dtype=np.float64)
        failure_spec = GR15Spec(
            0.0,
            1.0,
            initial_step=0.5,
            minimum_step=0.5,
            maximum_step=0.5,
            epsilon=1.0e-9,
            maximum_iterations=1,
        )
        v2_failure_spec = v2.GR15Spec(
            0.0,
            1.0,
            initial_step=0.5,
            minimum_step=0.5,
            maximum_step=0.5,
            epsilon=1.0e-9,
            maximum_iterations=1,
        )
        for integrator, selected_spec, error_type in (
            (v2.integrate_gr15, v2_failure_spec, v2.GR15IntegrationError),
            (v3.integrate, failure_spec, GR15IntegrationError),
        ):
            with self.subTest(integrator=integrator.__module__, failure="minimum_step"):
                with self.assertRaises(error_type) as captured:
                    integrator(positions, velocities, gm, selected_spec)
                self.assertEqual(captured.exception.status, STATUS_MINIMUM_STEP)
            coincident = positions.copy()
            coincident[1] = coincident[0]
            with self.subTest(integrator=integrator.__module__, failure="singularity"):
                with self.assertRaises(error_type) as captured:
                    integrator(coincident, velocities, gm, selected_spec)
                self.assertEqual(captured.exception.status, STATUS_FORCE_SINGULARITY)

        body_count = 32
        angles = np.arange(body_count, dtype=np.float64) * (2.0 * np.pi / body_count)
        many_positions = np.zeros((body_count, 3), dtype=np.float64)
        many_positions[:, 0] = np.cos(angles)
        many_positions[:, 1] = np.sin(angles)
        many_velocities = np.zeros_like(many_positions)
        many_gm = np.full(body_count, 1.0e-6, dtype=np.float64)
        result = v3.integrate(
            many_positions,
            many_velocities,
            many_gm,
            GR15Spec(
                0.0,
                1.0e-4,
                initial_step=1.0e-4,
                minimum_step=1.0e-12,
                maximum_step=1.0e-4,
                epsilon=1.0e-6,
            ),
        )
        self.assertEqual(result.positions.shape, (body_count, 3))
        self.assertTrue(np.all(np.isfinite(result.positions)))


if __name__ == "__main__":
    unittest.main()
