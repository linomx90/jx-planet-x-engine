from __future__ import annotations

from unittest import mock
import unittest

import numpy as np

import jxplanetx
from jxplanetx.gr15 import GR15Spec
from jxplanetx.gr15_eih_1pn import integrate_gr15_eih_1pn
from jxplanetx.gr15_eih_1pn_batch import (
    AUTO_API_ID,
    CPU_BACKEND_ID,
    CUDA_BACKEND_ID,
    DEFAULT_DISPATCH_POLICY,
    GR15EIH1PNBatchContractError,
    GR15EIH1PNBatchIntegrationError,
    integrate_gr15_eih_1pn_batch,
    integrate_gr15_eih_1pn_cpu_batch,
    select_gr15_eih_1pn_backend,
)
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters


def _fixture(
    systems: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.empty((systems, 2, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    gm = np.empty((systems, 2), dtype=np.float64)
    for system in range(systems):
        radius = 1.0 + 0.05 * system
        mass_scale = 1.0 + 0.02 * system
        positions[system] = ((-0.5 * radius, 0.0, 0.0), (0.5 * radius, 0.0, 0.0))
        velocities[system] = ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))
        gm[system] = (0.5 * mass_scale, 0.5 * mass_scale)
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(gm),
    )


def _spec(*, backward: bool = False) -> GR15Spec:
    if backward:
        return GR15Spec(
            0.2,
            0.0,
            intermediate_epochs=(0.11,),
            initial_step=0.01,
            maximum_step=0.02,
            epsilon=1.0e-9,
        )
    return GR15Spec(
        0.0,
        0.2,
        intermediate_epochs=(0.07, 0.13),
        initial_step=0.01,
        maximum_step=0.02,
        epsilon=1.0e-9,
    )


def _parameters() -> EIH1PNParameters:
    return EIH1PNParameters(
        speed_of_light_km_s=100.0,
        maximum_compactness=0.1,
        maximum_speed_fraction_squared=0.1,
    )


class GR15EIH1PNBatchCPUTests(unittest.TestCase):
    def test_public_surface_and_claim_ceiling(self) -> None:
        self.assertIs(
            jxplanetx.integrate_gr15_eih_1pn_batch,
            integrate_gr15_eih_1pn_batch,
        )
        self.assertIs(
            jxplanetx.integrate_gr15_eih_1pn_cpu_batch,
            integrate_gr15_eih_1pn_cpu_batch,
        )
        positions, velocities, gm = _fixture(2)
        result = integrate_gr15_eih_1pn_cpu_batch(
            positions, velocities, gm[0], _spec(), _parameters(), workers=2
        )
        self.assertEqual(result.api_id, AUTO_API_ID)
        self.assertEqual(result.decision.backend_id, CPU_BACKEND_ID)
        self.assertEqual(result.scientific_claim_state, "SCREENING_ONLY")
        self.assertFalse(result.production_authorized)
        self.assertFalse(result.exact_general_relativity_claimed)
        self.assertFalse(result.ephemeris_equivalence_claimed)
        self.assertFalse(result.general_superiority_claimed)
        self.assertFalse(result.checkpoint_positions_km.flags.writeable)
        self.assertFalse(result.checkpoint_velocities_km_s.flags.writeable)

    def test_shared_gm_parallel_batch_is_bitwise_sequential(self) -> None:
        positions, velocities, gm = _fixture(4)
        spec = _spec()
        parameters = _parameters()
        result = integrate_gr15_eih_1pn_cpu_batch(
            positions, velocities, gm[0], spec, parameters, workers=4
        )
        self.assertEqual(result.worker_count, 4)
        self.assertEqual(result.kernel_launch_count, 0)
        for system in range(4):
            expected = integrate_gr15_eih_1pn(
                positions[system], velocities[system], gm[0], spec, parameters
            )
            np.testing.assert_array_equal(
                result.checkpoint_positions_km[system],
                expected.checkpoint_positions,
            )
            np.testing.assert_array_equal(
                result.checkpoint_velocities_km_s[system],
                expected.checkpoint_velocities,
            )
            self.assertEqual(
                result.system_audits[system].replay_digest,
                expected.replay_digest,
            )

    def test_per_system_gm_and_thread_counts_are_deterministic(self) -> None:
        positions, velocities, gm = _fixture(4)
        arguments = (positions, velocities, gm, _spec(), _parameters())
        sequential = integrate_gr15_eih_1pn_cpu_batch(*arguments, workers=1)
        parallel = integrate_gr15_eih_1pn_cpu_batch(*arguments, workers=4)
        np.testing.assert_array_equal(
            parallel.checkpoint_positions_km,
            sequential.checkpoint_positions_km,
        )
        np.testing.assert_array_equal(
            parallel.checkpoint_velocities_km_s,
            sequential.checkpoint_velocities_km_s,
        )
        self.assertEqual(parallel.system_audits, sequential.system_audits)

    def test_backward_batch_matches_independent_cpu(self) -> None:
        positions, velocities, gm = _fixture(3)
        spec = _spec(backward=True)
        parameters = _parameters()
        result = integrate_gr15_eih_1pn_cpu_batch(
            positions, velocities, gm, spec, parameters, workers=3
        )
        for system in range(3):
            expected = integrate_gr15_eih_1pn(
                positions[system], velocities[system], gm[system], spec, parameters
            )
            np.testing.assert_array_equal(
                result.checkpoint_positions_km[system],
                expected.checkpoint_positions,
            )

    def test_auto_falls_back_to_cpu_outside_calibrated_scope(self) -> None:
        positions, velocities, gm = _fixture(2)
        with mock.patch(
            "jxplanetx.gr15_eih_1pn_batch._cuda_identity",
            side_effect=AssertionError("uncalibrated auto must not probe CUDA"),
        ):
            result = integrate_gr15_eih_1pn_batch(
                positions, velocities, gm, _spec(), _parameters()
            )
        self.assertEqual(result.decision.backend_id, CPU_BACKEND_ID)
        self.assertFalse(result.decision.performance_calibrated)
        self.assertIn("outside", result.decision.reason)

    def test_profile_decisions_are_auditable(self) -> None:
        spec = GR15Spec(
            0.0,
            DEFAULT_DISPATCH_POLICY.absolute_duration_seconds,
            initial_step=21_600.0,
            maximum_step=691_200.0,
            epsilon=1.0e-6,
        )
        identity = {"device_name": DEFAULT_DISPATCH_POLICY.cuda_device_name}
        with (
            mock.patch(
                "jxplanetx.gr15_eih_1pn_batch._cpu_model",
                return_value=DEFAULT_DISPATCH_POLICY.cpu_model,
            ),
            mock.patch(
                "jxplanetx.gr15_eih_1pn_batch._cuda_identity",
                return_value=identity,
            ) as probe,
        ):
            small = select_gr15_eih_1pn_backend(64, 11, spec)
            measured = select_gr15_eih_1pn_backend(128, 11, spec)
            extrapolated = select_gr15_eih_1pn_backend(256, 11, spec)
        self.assertEqual(small.backend_id, CPU_BACKEND_ID)
        self.assertTrue(small.performance_calibrated)
        self.assertEqual(measured.backend_id, CUDA_BACKEND_ID)
        self.assertTrue(measured.performance_calibrated)
        self.assertFalse(measured.threshold_extrapolated)
        self.assertEqual(extrapolated.backend_id, CUDA_BACKEND_ID)
        self.assertFalse(extrapolated.performance_calibrated)
        self.assertTrue(extrapolated.threshold_extrapolated)
        self.assertEqual(probe.call_count, 2)

    def test_contract_and_system_failure_are_fail_closed(self) -> None:
        positions, velocities, gm = _fixture(2)
        with self.assertRaises(GR15EIH1PNBatchContractError):
            integrate_gr15_eih_1pn_cpu_batch(
                positions.astype(np.float32),
                velocities,
                gm,
                _spec(),
                _parameters(),
            )
        with self.assertRaises(GR15EIH1PNBatchContractError):
            integrate_gr15_eih_1pn_cpu_batch(
                positions, velocities, gm, _spec(), _parameters(), workers=3
            )
        invalid = positions.copy()
        invalid[1, 1] = invalid[1, 0]
        with self.assertRaises(GR15EIH1PNBatchIntegrationError) as caught:
            integrate_gr15_eih_1pn_cpu_batch(
                invalid, velocities, gm, _spec(), _parameters(), workers=2
            )
        self.assertEqual(caught.exception.system, 1)
        self.assertIsNotNone(caught.exception.status)
if __name__ == "__main__":
    unittest.main()
