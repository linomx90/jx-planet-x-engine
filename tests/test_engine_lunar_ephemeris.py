"""Numerical and adversarial tests for the public lunar ephemeris contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import struct
import unittest

import numpy as np
import jxplanetx
import jxplanetx.engine as public_engine

from jxplanetx.engine.contracts import Provenance
from jxplanetx.engine.lunar_ephemeris import (
    BODY_COUNT,
    LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN,
    LUNAR_EPHEMERIS_V1_EPOCH_UNIT,
    LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN,
    LunarEphemerisV1Error,
    LunarEphemerisV1IntegrationSpec,
    LunarEphemerisV1Parameters,
    LunarEphemerisV1State,
    integrate_lunar_ephemeris_v1,
)
from jxplanetx.engine.rkf78 import RKF78_STAGE_COUNT
from jxplanetx.solar_system.eih_1pn import EIH1PNParameters
from jxplanetx.solar_system.lunar_coupled import CoupledLunarParameters
from jxplanetx.solar_system.lunar_coupled_solar import (
    SolarCoupledLunarParameters,
)
from jxplanetx.solar_system.lunar_fluid_core import LunarFluidCoreCoupling


LEDGER_DOMAIN = b"jxplanetx.engine.lunar-ephemeris-v1-ledger.v2\0"


def _fixture() -> tuple[
    LunarEphemerisV1State,
    LunarEphemerisV1Parameters,
    LunarEphemerisV1IntegrationSpec,
]:
    positions = np.array(
        (
            (0.0, 0.0, 0.0),
            (57_900_000.0, 0.0, 0.0),
            (0.0, 108_200_000.0, 0.0),
            (149_600_000.0, 0.0, 0.0),
            (149_984_400.0, 0.0, 0.0),
            (-227_900_000.0, 0.0, 0.0),
            (0.0, 778_500_000.0, 0.0),
            (0.0, -1_434_000_000.0, 0.0),
            (-2_871_000_000.0, 0.0, 0.0),
            (0.0, 4_495_000_000.0, 0.0),
            (5_900_000_000.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    velocities = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.0, 47.4, 0.0),
            (-35.0, 0.0, 0.0),
            (0.0, 29.78, 0.0),
            (0.0, 30.802, 0.0),
            (0.0, -24.1, 0.0),
            (-13.1, 0.0, 0.0),
            (9.7, 0.0, 0.0),
            (0.0, -6.8, 0.0),
            (-5.4, 0.0, 0.0),
            (0.0, 4.7, 0.0),
        ),
        dtype=np.float64,
    )
    gm = np.array(
        (
            132_712_440_041.27942,
            22_031.86855,
            324_858.592,
            398_600.435507,
            4_902.800118,
            42_828.375816,
            126_712_764.1,
            37_940_584.84,
            5_794_556.4,
            6_836_527.1,
            975.5,
        ),
        dtype=np.float64,
    )
    provenance = Provenance(
        source_id="test.lunar-ephemeris.fixture",
        citation="Deterministic synthetic contract fixture",
        version="1",
        sha256="a" * 64,
    )
    state = LunarEphemerisV1State(
        snapshot_id="lunar-contract-fixture",
        epoch=0.0,
        positions_km=positions,
        velocities_km_s=velocities,
        mantle_quaternion_body_to_inertial=np.array(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        ),
        mantle_angular_velocity_body_s=np.array(
            (0.0, 0.0, 2.66e-6), dtype=np.float64
        ),
        core_angular_velocity_body_s=np.array(
            (0.0, 0.0, 2.64e-6), dtype=np.float64
        ),
        provenance=provenance,
    )
    core = LunarFluidCoreCoupling(
        core_inertia_over_mr2=(0.001, 0.001, 0.0011),
        viscous_coefficient_over_mr2_per_second=1.0e-12,
        parameter_source_id="test.lunar-core.fixture",
    )
    lunar = CoupledLunarParameters(
        earth_gm_km3_s2=float(gm[3]),
        moon_gm_km3_s2=float(gm[4]),
        lunar_reference_radius_km=1_737.4,
        figure_inertia_over_mr2=(0.393, 0.3932, 0.394),
        mantle_inertia_over_mr2=(0.392, 0.3922, 0.3929),
        core_coupling=core,
    )
    coupled = SolarCoupledLunarParameters(
        sun_gm_km3_s2=float(gm[0]),
        lunar=lunar,
        earth_j2=0.00108262668,
        earth_reference_radius_km=6_378.1363,
    )
    parameters = LunarEphemerisV1Parameters(
        coupled_lunar=coupled,
        gravitational_parameters_km3_s2=gm,
        eih=EIH1PNParameters(speed_of_light_km_s=299_792.458),
    )
    spec = LunarEphemerisV1IntegrationSpec(
        checkpoint_epochs=(0.0, 1.5, 3.0),
        maximum_step=1.0,
        maximum_steps=4,
    )
    return state, parameters, spec


class LunarEphemerisNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initial, cls.parameters, cls.spec = _fixture()
        cls.result = integrate_lunar_ephemeris_v1(
            cls.initial, cls.parameters, cls.spec
        )

    def test_small_real_integration_is_finite_deterministic_and_accounted(self):
        run = self.result
        self.assertEqual(run.accepted_step_epochs, (1.0, 1.5, 2.5, 3.0))
        self.assertEqual(run.accepted_step_magnitudes, (1.0, 0.5, 1.0, 0.5))
        self.assertEqual(run.accepted_steps, 4)
        self.assertEqual(run.force_evaluations, 4 * RKF78_STAGE_COUNT)
        self.assertEqual(
            tuple(state.epoch for state in run.checkpoints),
            self.spec.checkpoint_epochs,
        )

        self.assertEqual(run.final_state.epoch_unit, LUNAR_EPHEMERIS_V1_EPOCH_UNIT)
        self.assertEqual(run.final_state.epoch_origin, LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN)
        self.assertTrue(np.all(np.isfinite(run.final_state.positions_km)))
        self.assertTrue(np.all(np.isfinite(run.final_state.velocities_km_s)))
        self.assertAlmostEqual(
            float(np.linalg.norm(run.final_state.mantle_quaternion_body_to_inertial)),
            1.0,
        )
        repeated = integrate_lunar_ephemeris_v1(
            self.initial, self.parameters, self.spec
        )
        self.assertEqual(
            repeated.accepted_step_ledger_sha256,
            run.accepted_step_ledger_sha256,
        )
        self.assertEqual(repeated.result_content_sha256, run.result_content_sha256)
        self.assertIn("result.v2", LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN)

    def test_epoch_contract_is_reexported_by_public_namespaces(self):
        for namespace in (jxplanetx, public_engine):
            self.assertEqual(
                namespace.LUNAR_EPHEMERIS_V1_EPOCH_UNIT,
                LUNAR_EPHEMERIS_V1_EPOCH_UNIT,
            )
            self.assertEqual(
                namespace.LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN,
                LUNAR_EPHEMERIS_V1_EPOCH_ORIGIN,
            )

    def test_ledger_is_independently_recomputable_from_retained_roster(self):
        digest = hashlib.sha256(LEDGER_DOMAIN)
        for epoch, magnitude in zip(
            self.result.accepted_step_epochs,
            self.result.accepted_step_magnitudes,
            strict=True,
        ):
            digest.update(struct.pack(">dd", epoch, magnitude))
        self.assertEqual(digest.hexdigest(), self.result.accepted_step_ledger_sha256)

    def test_owned_state_has_complete_resolved_eleven_shape(self):
        self.assertEqual(self.result.final_state.positions_km.shape, (BODY_COUNT, 3))
        self.assertFalse(self.result.final_state.positions_km.flags.writeable)
        self.assertFalse(self.result.final_state.velocities_km_s.flags.writeable)


class LunarEphemerisForgeryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initial, cls.parameters, cls.spec = _fixture()
        cls.result = integrate_lunar_ephemeris_v1(
            cls.initial, cls.parameters, cls.spec
        )

    def assertRejected(self, **changes: object) -> None:
        with self.assertRaises(LunarEphemerisV1Error):
            replace(self.result, **changes)

    def test_epoch_unit_and_origin_are_explicit_and_fail_closed(self):
        self.assertEqual(self.initial.time_scale, "TDB")
        self.assertEqual(self.initial.epoch_unit, "SI_SECOND")
        self.assertEqual(self.initial.epoch_origin, "J2000_TDB_JD_2451545.0")
        with self.assertRaises(LunarEphemerisV1Error):
            replace(self.initial, epoch_unit="DAY")
        with self.assertRaises(LunarEphemerisV1Error):
            replace(self.initial, epoch_origin="CALLER_LOCAL_ZERO")

    def test_empty_or_mistyped_step_rosters_are_rejected(self):
        self.assertRejected(accepted_step_epochs=())
        epochs = list(self.result.accepted_step_epochs)
        epochs[0] = 1
        self.assertRejected(accepted_step_epochs=tuple(epochs))
        magnitudes = list(self.result.accepted_step_magnitudes)
        magnitudes[1] = -0.5
        self.assertRejected(accepted_step_magnitudes=tuple(magnitudes))

    def test_step_schedule_and_work_cap_forgery_are_rejected(self):
        epochs = list(self.result.accepted_step_epochs)
        epochs[0] = math.nextafter(epochs[0], math.inf)
        self.assertRejected(accepted_step_epochs=tuple(epochs))
        looser_spec = replace(self.spec, maximum_steps=5)
        self.assertRejected(integration_spec=looser_spec)

    def test_checkpoint_epoch_context_provenance_and_identity_are_rejected(self):
        checkpoints = list(self.result.checkpoints)
        checkpoints[1] = replace(checkpoints[1], epoch=1.25)
        self.assertRejected(checkpoints=tuple(checkpoints))

        checkpoints = list(self.result.checkpoints)
        other_provenance = Provenance(
            source_id="test.other",
            citation="Different source",
            version="1",
            sha256="b" * 64,
        )
        checkpoints[1] = replace(checkpoints[1], provenance=other_provenance)
        self.assertRejected(checkpoints=tuple(checkpoints))

        checkpoints = list(self.result.checkpoints)
        checkpoints[1] = replace(checkpoints[1], snapshot_id="forged")
        self.assertRejected(checkpoints=tuple(checkpoints))

    def test_ledger_and_result_digest_forgery_are_rejected(self):
        self.assertRejected(accepted_step_ledger_sha256="0" * 64)
        self.assertRejected(result_content_sha256="0" * 64)
        self.assertRejected(
            maximum_eih_compactness=math.nextafter(
                self.result.maximum_eih_compactness, math.inf
            )
        )
        self.assertRejected(
            maximum_eih_compactness=(
                2.0 * self.parameters.eih.maximum_compactness
            )
        )

    def test_initial_state_bytes_are_bound_even_if_first_checkpoint_is_changed(self):
        changed_positions = np.array(self.initial.positions_km, copy=True)
        changed_positions[1, 0] += 1.0
        changed_initial = replace(self.initial, positions_km=changed_positions)
        checkpoints = list(self.result.checkpoints)
        checkpoints[0] = changed_initial
        self.assertRejected(
            initial_state=changed_initial,
            checkpoints=tuple(checkpoints),
        )

    def test_complete_parameter_content_is_bound(self):
        changed_gm = np.array(
            self.parameters.gravitational_parameters_km3_s2, copy=True
        )
        changed_gm[1] *= 2.0
        changed_parameters = replace(
            self.parameters,
            gravitational_parameters_km3_s2=changed_gm,
        )
        self.assertRejected(parameters=changed_parameters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
