from __future__ import annotations

import unittest

import numpy as np

from benchmarks import jx_long_horizon_m3_tracer_preflight as qualification
from jxplanetx.long_horizon import (
    KNOWN_SOLAR_SYSTEM_V2,
    KnownSystemRunProfile,
    prepare_known_system_observed_fixed_run,
)


class M3ExactRebound511Tests(unittest.TestCase):
    """Exercise the exact REBOUND API used by the independent M3 lane."""

    def test_one_refinement_node_is_massless_and_source_invariant(self) -> None:
        rebound = qualification._load_exact_rebound()
        contract = prepare_known_system_observed_fixed_run(
            profile_id=KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL
        )
        source_position = np.asarray(
            KNOWN_SOLAR_SYSTEM_V2.positions_primary_relative_ecliptic_AU,
            dtype=np.float64,
        )
        source_velocity = np.asarray(
            KNOWN_SOLAR_SYSTEM_V2.velocities_primary_relative_ecliptic_AU_per_julian_year,
            dtype=np.float64,
        )
        source_gm = np.asarray(
            KNOWN_SOLAR_SYSTEM_V2.gm_AU3_per_julian_year2,
            dtype=np.float64,
        )
        bary_position, bary_velocity = qualification._primary_relative_to_barycentric(
            source_position, source_velocity, source_gm
        )
        combined = qualification._new_rebound_simulation(
            rebound,
            bary_position,
            bary_velocity,
            source_gm,
            contract.tracer_positions_AU,
            contract.tracer_velocities_AU_per_year,
        )
        source_only = qualification._new_rebound_simulation(
            rebound,
            bary_position,
            bary_velocity,
            source_gm,
            None,
            None,
        )
        target = qualification.REFINEMENT_STEP_YEARS
        combined.integrate(target, exact_finish_time=1)
        source_only.integrate(target, exact_finish_time=1)
        self.assertEqual(float(combined.t).hex(), target.hex())
        self.assertEqual(float(source_only.t).hex(), target.hex())
        self.assertEqual(combined.N, qualification.SOURCE_COUNT + qualification.STATE_COUNT)
        self.assertEqual(combined.N_active, qualification.SOURCE_COUNT)
        self.assertEqual(source_only.N, qualification.SOURCE_COUNT)
        self.assertEqual(source_only.N_active, qualification.SOURCE_COUNT)

        combined_source_position, combined_source_velocity, tracers, tracer_velocity = (
            qualification._rebound_node(combined)
        )
        source_only_position, source_only_velocity, empty_position, empty_velocity = (
            qualification._rebound_node(source_only)
        )
        self.assertEqual(tracers.shape, (qualification.STATE_COUNT, 3))
        self.assertEqual(tracer_velocity.shape, tracers.shape)
        self.assertEqual(empty_position.shape, (0, 3))
        self.assertEqual(empty_velocity.shape, (0, 3))
        np.testing.assert_array_equal(
            combined_source_position.view(np.uint64),
            source_only_position.view(np.uint64),
        )
        np.testing.assert_array_equal(
            combined_source_velocity.view(np.uint64),
            source_only_velocity.view(np.uint64),
        )
        self.assertTrue(np.all(np.isfinite(tracers)))
        self.assertTrue(np.all(np.isfinite(tracer_velocity)))


if __name__ == "__main__":
    unittest.main()
