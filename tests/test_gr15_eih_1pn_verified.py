from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

import jxplanetx
from jxplanetx.gr15 import GR15Spec
from jxplanetx.gr15_eih_1pn import (
    EXPECTED_CORE_SOURCE_SHA256,
    EXPECTED_FORCE_CORE_SOURCE_SHA256,
    integrate_gr15_eih_1pn,
)
from jxplanetx.gr15_eih_1pn_verified import (
    API_ID,
    GR15EIH1PNOperationalContext,
    GR15EIH1PNVerificationContractError,
    GR15EIH1PNVerificationFailure,
    GR15EIH1PNVerificationGates,
    VERIFICATION_STATUS,
    integrate_verified_gr15_eih_1pn,
)
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = np.ascontiguousarray((1.0, 0.2), dtype=np.float64)
    relative_position = np.asarray((1.2, 0.3, -0.1), dtype=np.float64)
    relative_velocity = np.asarray((-0.2, 0.9, 0.15), dtype=np.float64)
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


def _context(**changes: object) -> GR15EIH1PNOperationalContext:
    values: dict[str, object] = {
        "body_ids": ("PRIMARY", "SECONDARY"),
        "state_source": "analytic finite-mass binary fixture",
        "state_source_sha256": "a" * 64,
        "gm_source": "analytic finite-mass binary fixture",
        "gm_source_sha256": "b" * 64,
    }
    values.update(changes)
    return GR15EIH1PNOperationalContext(**values)


def _specs(*, backward: bool = False) -> tuple[GR15Spec, GR15Spec]:
    if backward:
        primary = GR15Spec(
            0.5,
            0.0,
            intermediate_epochs=(0.31, 0.17),
            initial_step=1.0e-3,
            minimum_step=1.0e-14,
            maximum_step=1.0e-2,
            epsilon=1.0e-8,
        )
        confirmation = GR15Spec(
            0.5,
            0.0,
            intermediate_epochs=(0.31, 0.17),
            initial_step=5.0e-4,
            minimum_step=1.0e-15,
            maximum_step=5.0e-3,
            epsilon=1.0e-10,
        )
        return primary, confirmation
    primary = GR15Spec(
        0.0,
        0.5,
        intermediate_epochs=(0.17, 0.31),
        initial_step=1.0e-3,
        minimum_step=1.0e-14,
        maximum_step=1.0e-2,
        epsilon=1.0e-8,
    )
    confirmation = GR15Spec(
        0.0,
        0.5,
        intermediate_epochs=(0.17, 0.31),
        initial_step=5.0e-4,
        minimum_step=1.0e-15,
        maximum_step=5.0e-3,
        epsilon=1.0e-10,
    )
    return primary, confirmation


def _parameters() -> EIH1PNParameters:
    return EIH1PNParameters(
        speed_of_light_km_s=100.0,
        maximum_compactness=0.1,
        maximum_speed_fraction_squared=0.1,
    )


def _gates() -> GR15EIH1PNVerificationGates:
    return GR15EIH1PNVerificationGates(1.0e-8, 1.0e-8)


class GR15EIH1PNVerifiedExecutionTests(unittest.TestCase):
    def test_public_result_is_primary_exact_and_claim_limited(self) -> None:
        positions, velocities, gm = _state()
        primary_spec, confirmation_spec = _specs()
        result = integrate_verified_gr15_eih_1pn(
            positions,
            velocities,
            gm,
            primary_spec,
            confirmation_spec,
            _parameters(),
            _context(),
            _gates(),
        )
        direct = integrate_gr15_eih_1pn(
            positions, velocities, gm, primary_spec, _parameters()
        )
        np.testing.assert_array_equal(
            result.checkpoint_positions, direct.checkpoint_positions
        )
        np.testing.assert_array_equal(
            result.checkpoint_velocities, direct.checkpoint_velocities
        )
        self.assertIs(
            jxplanetx.integrate_verified_gr15_eih_1pn,
            integrate_verified_gr15_eih_1pn,
        )
        self.assertEqual(result.api_id, API_ID)
        self.assertEqual(result.verification_status, VERIFICATION_STATUS)
        self.assertTrue(result.verified_numerical_execution)
        self.assertFalse(result.independent_implementation_confirmation)
        self.assertFalse(result.production_ephemeris_authorized)
        self.assertFalse(result.navigation_authorized)
        self.assertFalse(result.exact_general_relativity_claimed)
        self.assertFalse(result.de440_equivalence_claimed)
        self.assertEqual(len(result.input_digest), 64)
        self.assertEqual(len(result.verification_digest), 64)

    def test_non_timing_verification_is_bitwise_repeatable(self) -> None:
        arguments = (*_state(), *_specs(), _parameters(), _context(), _gates())
        first = integrate_verified_gr15_eih_1pn(*arguments)
        second = integrate_verified_gr15_eih_1pn(*arguments)
        np.testing.assert_array_equal(
            first.checkpoint_positions, second.checkpoint_positions
        )
        np.testing.assert_array_equal(
            first.checkpoint_velocities, second.checkpoint_velocities
        )
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertEqual(first.verification_digest, second.verification_digest)
        self.assertEqual(
            first.confirmation_replay_digest, second.confirmation_replay_digest
        )

    def test_backward_execution_passes_the_same_contract(self) -> None:
        forward_primary, _ = _specs()
        positions, velocities, gm = _state()
        forward = integrate_gr15_eih_1pn(
            positions, velocities, gm, forward_primary, _parameters()
        )
        backward_primary, backward_confirmation = _specs(backward=True)
        backward = integrate_verified_gr15_eih_1pn(
            np.ascontiguousarray(forward.positions),
            np.ascontiguousarray(forward.velocities),
            gm,
            backward_primary,
            backward_confirmation,
            _parameters(),
            _context(state_source="forward GR15 endpoint"),
            _gates(),
        )
        self.assertEqual(backward.verification_status, VERIFICATION_STATUS)

    def test_operational_metadata_and_arrays_fail_closed(self) -> None:
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            _context(reference_frame="ICRF")
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            _context(body_ids=("SAME", "SAME"))
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            _context(state_source_sha256="A" * 64)
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            _context(coordinate_origin="SUN")
        positions, velocities, gm = _state()
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            integrate_verified_gr15_eih_1pn(
                positions.astype(np.float32),
                velocities,
                gm,
                *_specs(),
                _parameters(),
                _context(),
                _gates(),
            )

    def test_confirmation_must_be_distinct_no_looser_and_same_schedule(self) -> None:
        positions, velocities, gm = _state()
        primary, confirmation = _specs()
        common = (positions, velocities, gm)
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            integrate_verified_gr15_eih_1pn(
                *common, primary, primary, _parameters(), _context(), _gates()
            )
        looser = GR15Spec(
            primary.initial_epoch,
            primary.final_epoch,
            intermediate_epochs=primary.intermediate_epochs,
            initial_step=primary.initial_step,
            minimum_step=primary.minimum_step,
            maximum_step=primary.maximum_step,
            epsilon=1.0e-7,
        )
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            integrate_verified_gr15_eih_1pn(
                *common, primary, looser, _parameters(), _context(), _gates()
            )
        mismatched = GR15Spec(
            primary.initial_epoch,
            primary.final_epoch,
            intermediate_epochs=(0.2, 0.31),
            initial_step=confirmation.initial_step,
            minimum_step=confirmation.minimum_step,
            maximum_step=confirmation.maximum_step,
            epsilon=confirmation.epsilon,
        )
        with self.assertRaises(GR15EIH1PNVerificationContractError):
            integrate_verified_gr15_eih_1pn(
                *common, primary, mismatched, _parameters(), _context(), _gates()
            )

    def test_declared_zero_gate_rejects_a_real_numerical_difference(self) -> None:
        positions, velocities, gm = _state()
        with self.assertRaises(GR15EIH1PNVerificationFailure) as caught:
            integrate_verified_gr15_eih_1pn(
                positions,
                velocities,
                gm,
                *_specs(),
                _parameters(),
                _context(),
                GR15EIH1PNVerificationGates(0.0, 0.0),
            )
        self.assertGreater(
            caught.exception.maximum_position_difference_km
            + caught.exception.maximum_velocity_difference_km_s,
            0.0,
        )

    def test_existing_numerical_and_force_cores_are_unchanged(self) -> None:
        self.assertEqual(
            _sha256(ROOT / "src/jxplanetx/_gr15_eih_1pn_core.c"),
            EXPECTED_CORE_SOURCE_SHA256,
        )
        self.assertEqual(
            _sha256(ROOT / "src/jxplanetx/_eih_1pn_core.c"),
            EXPECTED_FORCE_CORE_SOURCE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
