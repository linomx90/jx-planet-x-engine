from __future__ import annotations

import dataclasses
import unittest

import numpy as np

import jxplanetx
from jxplanetx.gr15 import GR15Spec
from jxplanetx.gr15_eih_1pn import integrate_gr15_eih_1pn
from jxplanetx.gr15_eih_1pn_cuda import (
    CUDA_OPTIONS,
    CUDA_SOURCE_SHA256,
    GR15EIH1PNCUDAContractError,
    GR15EIH1PNCUDAIntegrationError,
    MAXIMUM_BODY_COUNT,
    METHOD_ID,
    MODEL_ID,
    STATUS_FORCE_SINGULARITY,
    STATUS_INPUT_DOMAIN,
    STATUS_REJECTION_LIMIT,
    STATUS_STEP_LIMIT,
    STATUS_WEAK_FIELD_DOMAIN,
    gr15_eih_1pn_cuda_runtime_identity,
    integrate_gr15_eih_1pn_cuda_batch,
)
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters


EXPECTED_CUDA_SOURCE_SHA256 = (
    "2097f0601910739cd6b485e20b32e942f2fca8ae9c69f7265fb1741f94665922"
)


def _require_cuda(test: unittest.TestCase):
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            test.skipTest("CUDA device unavailable")
    except Exception as exc:
        test.skipTest(f"CUDA runtime unavailable: {exc}")
    return cp


def _parameters(**overrides: float) -> EIH1PNParameters:
    values = {
        "speed_of_light_km_s": 100.0,
        "maximum_compactness": 0.1,
        "maximum_speed_fraction_squared": 0.1,
    }
    values.update(overrides)
    return EIH1PNParameters(**values)


def _binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64),
        np.array(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64),
        np.array((0.5, 0.5), dtype=np.float64),
    )


def _batch_fixture(
    system_count: int, body_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260923 + 101 * system_count + body_count)
    positions = np.empty((system_count, body_count, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    base_x = np.linspace(-12.0, 12.0, body_count)
    base_gm = np.linspace(1.0, 0.2, body_count)
    gm = np.empty((system_count, body_count), dtype=np.float64)
    for system in range(system_count):
        positions[system, :, 0] = base_x + 0.01 * system
        positions[system, :, 1] = 0.2 * np.sin(
            np.arange(body_count, dtype=np.float64) + system
        )
        positions[system, :, 2] = 0.1 * np.cos(
            np.arange(body_count, dtype=np.float64) - system
        )
        velocities[system] = rng.normal(0.0, 0.015, (body_count, 3))
        gm[system] = base_gm * (1.0 + 0.03 * system)
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(gm),
    )


class GR15EIH1PNCUDASourceContractTests(unittest.TestCase):
    def test_identity_options_and_public_surface(self) -> None:
        self.assertEqual(CUDA_SOURCE_SHA256, EXPECTED_CUDA_SOURCE_SHA256)
        self.assertEqual(METHOD_ID, "JX_GR15_EIH1PN_CUDA_V1")
        self.assertEqual(MODEL_ID, "jx.gr15.eih-1pn-mutual-point-mass.cuda.v1")
        self.assertIn("--fmad=false", CUDA_OPTIONS)
        self.assertIn("--prec-div=true", CUDA_OPTIONS)
        self.assertIs(
            jxplanetx.integrate_gr15_eih_1pn_cuda_batch,
            integrate_gr15_eih_1pn_cuda_batch,
        )


class GR15EIH1PNCUDAHardwareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _require_cuda(self)

    def test_runtime_identity_names_real_device(self) -> None:
        identity = gr15_eih_1pn_cuda_runtime_identity()
        self.assertEqual(identity["method_id"], METHOD_ID)
        self.assertEqual(identity["model_id"], MODEL_ID)
        self.assertEqual(identity["cuda_source_sha256"], CUDA_SOURCE_SHA256)
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")
        self.assertTrue(identity["device_name"])
        self.assertGreater(identity["cuda_runtime_version"], 0)
        self.assertGreater(identity["cuda_driver_version"], 0)

    def test_binary_checkpoints_and_accounting_exactly_match_cpu(self) -> None:
        cp = self.cp
        position, velocity, gm = _binary()
        spec = GR15Spec(
            0.0,
            0.2,
            intermediate_epochs=(0.07, 0.13),
            initial_step=0.01,
            maximum_step=0.02,
            epsilon=1.0e-9,
        )
        parameters = _parameters()
        expected = integrate_gr15_eih_1pn(
            position, velocity, gm, spec, parameters
        )
        result = integrate_gr15_eih_1pn_cuda_batch(
            cp.asarray(position[None]),
            cp.asarray(velocity[None]),
            cp.asarray(gm),
            spec,
            parameters,
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.checkpoint_positions_km)[0],
            expected.checkpoint_positions,
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.checkpoint_velocities_km_s)[0],
            expected.checkpoint_velocities,
        )
        audit = result.system_audits[0]
        fields = (
            "checkpoint_accepted_steps",
            "checkpoint_rejected_steps",
            "attempted_steps",
            "accepted_steps",
            "rejected_steps",
            "force_evaluations",
            "predictor_corrector_iterations",
            "nonconverged_retries",
            "polynomial_predictor_trials",
            "constant_predictor_trials",
            "history_commits",
            "terminal_force_reuses",
            "terminal_force_sweeps",
            "maximum_error_ratio",
            "minimum_accepted_step",
            "maximum_accepted_step",
            "final_proposed_step",
            "maximum_corrector_residual",
            "observed_maximum_compactness",
            "observed_maximum_speed_fraction_squared",
        )
        for field in fields:
            self.assertEqual(getattr(audit, field), getattr(expected, field))
        self.assertEqual(result.kernel_launch_count, 1)
        self.assertTrue(result.full_predictor_corrector_executed)
        self.assertTrue(result.adaptive_controller_executed)
        self.assertTrue(result.host_audit_performed)
        self.assertFalse(result.production_authorized)
        self.assertFalse(result.ephemeris_equivalence_claimed)

    def test_system_specific_gm_batch_matches_independent_cpu_runs(self) -> None:
        cp = self.cp
        positions, velocities, gm = _batch_fixture(4, 7)
        spec = GR15Spec(
            0.0,
            0.2,
            intermediate_epochs=(0.07, 0.13),
            initial_step=0.005,
            maximum_step=0.025,
            epsilon=1.0e-9,
        )
        parameters = _parameters()
        result = integrate_gr15_eih_1pn_cuda_batch(
            cp.asarray(positions),
            cp.asarray(velocities),
            cp.asarray(gm),
            spec,
            parameters,
        )
        actual_positions = cp.asnumpy(result.checkpoint_positions_km)
        actual_velocities = cp.asnumpy(result.checkpoint_velocities_km_s)
        for system in range(4):
            expected = integrate_gr15_eih_1pn(
                positions[system], velocities[system], gm[system], spec, parameters
            )
            np.testing.assert_allclose(
                actual_positions[system],
                expected.checkpoint_positions,
                rtol=2.0e-15,
                atol=2.0e-15,
            )
            np.testing.assert_allclose(
                actual_velocities[system],
                expected.checkpoint_velocities,
                rtol=2.0e-14,
                atol=2.0e-16,
            )
            audit = result.system_audits[system]
            self.assertEqual(audit.attempted_steps, expected.attempted_steps)
            self.assertEqual(audit.accepted_steps, expected.accepted_steps)
            self.assertEqual(audit.rejected_steps, expected.rejected_steps)
            self.assertEqual(audit.maximum_error_ratio, expected.maximum_error_ratio)

    def test_backward_integration_and_deterministic_replay(self) -> None:
        cp = self.cp
        position, velocity, gm = _binary()
        spec = GR15Spec(
            0.2,
            0.0,
            intermediate_epochs=(0.11,),
            initial_step=0.01,
            maximum_step=0.02,
            epsilon=1.0e-9,
        )
        parameters = _parameters()
        expected = integrate_gr15_eih_1pn(
            position, velocity, gm, spec, parameters
        )
        arguments = (
            cp.asarray(position[None]),
            cp.asarray(velocity[None]),
            cp.asarray(gm),
            spec,
            parameters,
        )
        first = integrate_gr15_eih_1pn_cuda_batch(*arguments)
        second = integrate_gr15_eih_1pn_cuda_batch(*arguments)
        np.testing.assert_array_equal(
            cp.asnumpy(first.checkpoint_positions_km),
            cp.asnumpy(second.checkpoint_positions_km),
        )
        np.testing.assert_array_equal(
            cp.asnumpy(first.checkpoint_positions_km)[0],
            expected.checkpoint_positions,
        )
        self.assertEqual(first.system_audits, second.system_audits)

    def test_maximum_body_boundary_tracks_cpu(self) -> None:
        cp = self.cp
        positions, velocities, gm = _batch_fixture(1, MAXIMUM_BODY_COUNT)
        spec = GR15Spec(
            0.0,
            0.002,
            initial_step=0.001,
            maximum_step=0.001,
            epsilon=1.0e-8,
        )
        parameters = _parameters()
        result = integrate_gr15_eih_1pn_cuda_batch(
            cp.asarray(positions),
            cp.asarray(velocities),
            cp.asarray(gm[0]),
            spec,
            parameters,
        )
        expected = integrate_gr15_eih_1pn(
            positions[0], velocities[0], gm[0], spec, parameters
        )
        np.testing.assert_allclose(
            cp.asnumpy(result.positions_km)[0],
            expected.positions,
            rtol=2.0e-14,
            atol=2.0e-15,
        )

    def test_contract_rejects_host_state_and_wrong_device_shape(self) -> None:
        cp = self.cp
        position, velocity, gm = _binary()
        spec = GR15Spec(0.0, 0.1)
        with self.assertRaises(GR15EIH1PNCUDAContractError):
            integrate_gr15_eih_1pn_cuda_batch(
                position[None], cp.asarray(velocity[None]), cp.asarray(gm), spec, _parameters()
            )
        with self.assertRaises(GR15EIH1PNCUDAContractError):
            integrate_gr15_eih_1pn_cuda_batch(
                cp.asarray(position), cp.asarray(velocity), cp.asarray(gm), spec, _parameters()
            )

    def test_named_fail_closed_domains(self) -> None:
        cp = self.cp
        position, velocity, gm = _binary()
        base_spec = GR15Spec(
            0.0,
            0.1,
            initial_step=0.01,
            maximum_step=0.02,
            epsilon=1.0e-9,
        )
        cases = (
            (
                "invalid-gm",
                position,
                velocity,
                np.array((0.5, -0.5), dtype=np.float64),
                base_spec,
                _parameters(),
                STATUS_INPUT_DOMAIN,
            ),
            (
                "singularity",
                np.zeros((2, 3), dtype=np.float64),
                velocity,
                gm,
                base_spec,
                _parameters(),
                STATUS_FORCE_SINGULARITY,
            ),
            (
                "weak-field",
                position,
                velocity,
                gm,
                base_spec,
                _parameters(maximum_compactness=1.0e-6),
                STATUS_WEAK_FIELD_DOMAIN,
            ),
            (
                "step-limit",
                position,
                velocity,
                gm,
                dataclasses.replace(base_spec, maximum_steps=1),
                _parameters(),
                STATUS_STEP_LIMIT,
            ),
            (
                "rejection-limit",
                position,
                velocity,
                gm,
                dataclasses.replace(
                    base_spec,
                    maximum_iterations=1,
                    maximum_rejections=0,
                ),
                _parameters(),
                STATUS_REJECTION_LIMIT,
            ),
        )
        for label, current_position, current_velocity, current_gm, spec, parameters, status in cases:
            with self.subTest(label=label):
                with self.assertRaises(GR15EIH1PNCUDAIntegrationError) as caught:
                    integrate_gr15_eih_1pn_cuda_batch(
                        cp.asarray(current_position[None]),
                        cp.asarray(current_velocity[None]),
                        cp.asarray(current_gm),
                        spec,
                        parameters,
                    )
                self.assertEqual(caught.exception.system, 0)
                self.assertEqual(caught.exception.status, status)


if __name__ == "__main__":
    unittest.main()
