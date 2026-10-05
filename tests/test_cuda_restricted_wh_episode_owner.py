from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
from jxplanetx.cuda_restricted_wh_episode import (
    advance_cuda_restricted_wh_episodes,
)
from jxplanetx.cuda_restricted_wh_episode_owner import (
    OWNED_EPISODE_CLAIMS,
    CUDARestrictedWHOwnedEpisodeResult,
    advance_cuda_restricted_wh_owned_episodes,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridContractError,
    CUDARestrictedWHHybridIntegrationError,
)

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)
from tests.test_cuda_restricted_wh_episode import (
    _assert_device_array_equal,
    _assert_route_ledgers_equal,
)


@GPU_REQUIRED
class CUDARestrictedWHOwnedEpisodeTests(unittest.TestCase):
    def _assert_matches_episode_reference(
        self, actual: CUDARestrictedWHOwnedEpisodeResult, expected: object
    ) -> None:
        for left, right in (
            (
                actual.state.tracer_relative_positions_AU,
                expected.state.tracer_relative_positions_AU,
            ),
            (
                actual.state.tracer_relative_velocities_AU_per_year,
                expected.state.tracer_relative_velocities_AU_per_year,
            ),
            (actual.state.tracer_indices, expected.state.tracer_indices),
            (
                actual.minimum_source_distance_AU,
                expected.minimum_source_distance_AU,
            ),
        ):
            _assert_device_array_equal(self, left, right)
        self.assertEqual(
            actual.state.global_source_step, expected.state.global_source_step
        )
        self.assertEqual(
            actual.maximum_kepler_iterations,
            expected.maximum_kepler_iterations,
        )
        _assert_route_ledgers_equal(
            self, actual.route_records, expected.route_records
        )
        self.assertEqual(
            [
                (
                    item.lane_index,
                    item.tracer_index,
                    item.first_map_step_index,
                    item.stop_map_step_index,
                    item.route_count,
                )
                for item in actual.episodes
            ],
            [
                (
                    item.lane_index,
                    item.tracer_index,
                    item.first_map_step_index,
                    item.stop_map_step_index,
                    item.route_count,
                )
                for item in expected.episodes
            ],
        )

    @staticmethod
    def _dense_fixture(lane_count: int, step_count: int):
        state = _state(lane_count)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[:, 0] = 1.0
        velocities[:] = velocities[0]
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=step_count + 1,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(step_count, 1, _backend())
        return state, tile, spec

    def test_quiet_segment_matches_episode_reference_bitwise(self):
        state = _state(17)
        tile = _tile(node_count=5)
        spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(actual.route_worker_indices, ())
        self.assertEqual(actual.runtime_identity["route_chain_submission_count"], 0)
        self.assertFalse(any(actual.claims.values()))
        self.assertEqual(dict(actual.claims), dict(OWNED_EPISODE_CLAIMS))

    def test_dense_serial_lookahead_is_bitwise_exact_and_cached(self):
        lane_count = 6
        step_count = 8
        state, tile, spec = self._dense_fixture(lane_count, step_count)
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=1,
            route_lookahead_steps=8,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(len(actual.route_records), lane_count * step_count)
        self.assertEqual(set(actual.route_worker_indices), {0})
        self.assertEqual(
            actual.runtime_identity["route_chain_submission_count"],
            lane_count * 2,
        )
        self.assertEqual(
            actual.runtime_identity["committed_cached_route_step_count"],
            lane_count * (step_count - 2),
        )
        self.assertEqual(
            actual.runtime_identity["discarded_speculative_route_step_count"], 0
        )

    def test_six_dedicated_workers_keep_each_episode_on_one_owner(self):
        lane_count = 12
        step_count = 8
        state, tile, spec = self._dense_fixture(lane_count, step_count)
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=6
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=6,
            route_lookahead_steps=8,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(set(actual.route_worker_indices), set(range(6)))
        assignment = {
            (record.map_step_index, record.lane_index, record.tracer_index): owner
            for record, owner in zip(
                actual.route_records, actual.route_worker_indices, strict=True
            )
        }
        for episode in actual.episodes:
            owners = {
                assignment[
                    (record.map_step_index, record.lane_index, record.tracer_index)
                ]
                for record in episode.route_records
            }
            self.assertEqual(len(owners), 1)
        self.assertEqual(
            actual.runtime_identity["new_episode_assignment_count"], lane_count
        )
        self.assertEqual(
            actual.runtime_identity["route_chain_submission_count"],
            lane_count * 2,
        )

    def test_one_step_lookahead_matches_reference_without_cache_hits(self):
        lane_count = 6
        step_count = 4
        state, tile, spec = self._dense_fixture(lane_count, step_count)
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=6
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=6,
            route_lookahead_steps=1,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(
            actual.runtime_identity["route_chain_submission_count"],
            lane_count * step_count,
        )
        self.assertEqual(
            actual.runtime_identity["committed_cached_route_step_count"], 0
        )

    def test_mixed_wave_matches_reference_and_never_changes_owner_mid_episode(self):
        state = _state(9)
        tile = _tile(
            node_count=5,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=6
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state, tile, spec, route_worker_count=6
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertGreater(len(actual.route_records), 0)
        self.assertLess(len(actual.route_records), 9 * 4)

    def test_moving_secondary_integer_ratio_matches_reference_bitwise(self):
        state = replace(_state(2), source_ids=("Sun", "Jupiter"))
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[:, 0] = 1.0
        velocities[:] = velocities[0]
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=7,
            encounter_radius=0.997,
            acceleration_bound=100.0,
            secondary=True,
        )
        spec = FixedStepCUDARestrictedWHSpec(2, 3, _backend())
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=1,
            route_lookahead_steps=2,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(len(actual.route_records), 4)
        self.assertTrue(
            all(
                record.route_result.source_steps_per_map_step == 3
                for record in actual.route_records
            )
        )

    def test_negative_source_step_matches_reference_bitwise(self):
        state = replace(_state(2), source_step_years=-0.001)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[:, 0] = 1.0
        velocities[:] = velocities[0]
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=3,
            step=-0.001,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(2, 1, _backend())
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=1,
            route_lookahead_steps=2,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(len(actual.route_records), 4)

    def test_single_step_episode_does_not_speculate_and_matches_reference(self):
        state = _state(1)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[0] = CP.asarray((1.0, 0.0, 0.0), dtype=CP.float64)
        velocities[0] = CP.asarray((10.0, 0.0, 0.0), dtype=CP.float64)
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=9,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(8, 1, _backend())
        reference = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        actual = advance_cuda_restricted_wh_owned_episodes(
            state,
            tile,
            spec,
            route_worker_count=1,
            route_lookahead_steps=8,
        )
        self._assert_matches_episode_reference(actual, reference)
        self.assertEqual(len(actual.route_records), 1)
        self.assertEqual(
            actual.runtime_identity["route_chain_submission_count"], 1
        )
        self.assertEqual(
            actual.runtime_identity["speculative_route_step_count"], 1
        )
        self.assertEqual(
            actual.runtime_identity["discarded_speculative_route_step_count"], 0
        )

    def test_terminal_first_event_remains_transactional(self):
        state = _state(1)
        position_before = state.tracer_relative_positions_AU.copy()
        velocity_before = state.tracer_relative_velocities_AU_per_year.copy()
        tile = _tile(node_count=2, encounter_radius=1.0)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as reference:
            advance_cuda_restricted_wh_episodes(
                state, tile, spec, route_worker_count=1
            )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as actual:
            advance_cuda_restricted_wh_owned_episodes(
                state, tile, spec, route_worker_count=1
            )
        self.assertEqual(actual.exception.handoff.reason, "TERMINAL_EVENT")
        self.assertEqual(
            actual.exception.handoff.tracer_index,
            reference.exception.handoff.tracer_index,
        )
        self.assertIs(actual.exception.handoff.state, state)
        _assert_device_array_equal(
            self, state.tracer_relative_positions_AU, position_before
        )
        _assert_device_array_equal(
            self, state.tracer_relative_velocities_AU_per_year, velocity_before
        )

    def test_speculative_terminal_is_deferred_to_the_exact_guarded_step(self):
        state = _state(1)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[0] = CP.asarray((1.0, 0.0, 0.0), dtype=CP.float64)
        velocities[0] = CP.asarray((-150.0, 0.0, 0.0), dtype=CP.float64)
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=9,
            encounter_radius=0.5,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(8, 1, _backend())
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as reference:
            advance_cuda_restricted_wh_episodes(
                state, tile, spec, route_worker_count=1
            )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as actual:
            advance_cuda_restricted_wh_owned_episodes(
                state,
                tile,
                spec,
                route_worker_count=1,
                route_lookahead_steps=8,
            )
        expected = reference.exception.handoff
        observed = actual.exception.handoff
        self.assertEqual(observed.reason, "TERMINAL_EVENT")
        self.assertEqual(
            observed.failure_map_step_index, expected.failure_map_step_index
        )
        self.assertEqual(observed.failure_map_step_index, 3)
        self.assertEqual(observed.tracer_index, expected.tracer_index)
        self.assertEqual(
            observed.route_result.completed_map_fraction,
            expected.route_result.completed_map_fraction,
        )
        observed_event = observed.route_result.events[0]
        expected_event = expected.route_result.events[0]
        for field in (
            "event_type",
            "source_index",
            "source_id",
            "map_step_fraction",
            "source_interval_offset",
            "interval_fraction",
            "surface_distance_AU",
            "surface_threshold_AU",
            "surface_residual_AU",
            "surface_tolerance_AU",
            "primary_specific_energy_AU2_per_year2",
            "primary_radial_velocity_AU_per_year",
            "surface_radial_velocity_AU_per_year",
            "location_method",
        ):
            self.assertEqual(
                getattr(observed_event, field), getattr(expected_event, field)
            )
        self.assertTrue(
            np.array_equal(
                observed_event.relative_velocity_AU_per_year,
                expected_event.relative_velocity_AU_per_year,
            )
        )
        self.assertTrue(
            np.array_equal(
                observed.route_result.terminal_relative_position_AU,
                expected.route_result.terminal_relative_position_AU,
            )
        )
        self.assertIs(observed.state, state)

    def test_nonroutable_failure_remains_transactional(self):
        state = replace(
            _state(1),
            tracer_relative_positions_AU=CP.zeros((1, 3), dtype=CP.float64),
        )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as actual:
            advance_cuda_restricted_wh_owned_episodes(
                state,
                _tile(node_count=2),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
                route_worker_count=6,
            )
        self.assertEqual(actual.exception.handoff.reason, "FORCE_SINGULARITY")
        self.assertIs(actual.exception.handoff.state, state)

    def test_invalid_lookahead_and_state_fail_with_contract_error(self):
        state = _state(1)
        tile = _tile(node_count=2)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        for lookahead in (0, 65, True):
            with self.subTest(lookahead=lookahead):
                with self.assertRaises(CUDARestrictedWHHybridContractError):
                    advance_cuda_restricted_wh_owned_episodes(
                        state,
                        tile,
                        spec,
                        route_worker_count=1,
                        route_lookahead_steps=lookahead,
                    )
        invalid_state = replace(state, source_schedule_sha256="1" * 64)
        with self.assertRaises(CUDARestrictedWHHybridContractError):
            advance_cuda_restricted_wh_owned_episodes(
                invalid_state, tile, spec, route_worker_count=1
            )


if __name__ == "__main__":
    unittest.main()
