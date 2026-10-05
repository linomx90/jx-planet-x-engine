from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
from jxplanetx.cuda_restricted_wh_episode import (
    EPISODE_CLAIMS,
    CUDARestrictedWHEpisodeResult,
    advance_cuda_restricted_wh_episodes,
    group_restricted_wh_route_episodes,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridContractError,
    CUDARestrictedWHHybridIntegrationError,
    advance_cuda_restricted_wh_hybrid,
)

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)


def _assert_device_array_equal(
    case: unittest.TestCase, actual: object, expected: object
) -> None:
    case.assertTrue(bool(CP.array_equal(actual, expected)))


def _assert_route_ledgers_equal(
    case: unittest.TestCase, actual: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    case.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected, strict=True):
        case.assertEqual(
            (
                left.map_step_index,
                left.lane_index,
                left.tracer_index,
                left.guard_phase,
                left.guard_reason,
                left.guard_source_index,
            ),
            (
                right.map_step_index,
                right.lane_index,
                right.tracer_index,
                right.guard_phase,
                right.guard_reason,
                right.guard_source_index,
            ),
        )
        left_route = left.route_result
        right_route = right.route_result
        case.assertEqual(
            (
                left_route.tracer_index,
                left_route.source_tile_sha256,
                left_route.source_segment_sha256,
                left_route.global_source_step,
                left_route.source_steps_per_map_step,
                left_route.accepted_step_count,
                left_route.derivative_evaluation_count,
                left_route.minimum_evaluated_stage_distance_AU,
                left_route.minimum_committed_stage_distance_AU,
                left_route.maximum_accepted_primary_dynamical_phase_radians,
                left_route.maximum_committed_primary_dynamical_phase_radians,
                left_route.completed_map_fraction,
            ),
            (
                right_route.tracer_index,
                right_route.source_tile_sha256,
                right_route.source_segment_sha256,
                right_route.global_source_step,
                right_route.source_steps_per_map_step,
                right_route.accepted_step_count,
                right_route.derivative_evaluation_count,
                right_route.minimum_evaluated_stage_distance_AU,
                right_route.minimum_committed_stage_distance_AU,
                right_route.maximum_accepted_primary_dynamical_phase_radians,
                right_route.maximum_committed_primary_dynamical_phase_radians,
                right_route.completed_map_fraction,
            ),
        )
        case.assertEqual(left_route.events, right_route.events)
        case.assertTrue(
            np.array_equal(
                left_route.endpoint_relative_position_AU,
                right_route.endpoint_relative_position_AU,
            )
        )
        case.assertTrue(
            np.array_equal(
                left_route.endpoint_relative_velocity_AU_per_year,
                right_route.endpoint_relative_velocity_AU_per_year,
            )
        )
        case.assertEqual(
            dict(left_route.runtime_identity), dict(right_route.runtime_identity)
        )


@GPU_REQUIRED
class CUDARestrictedWHEpisodeTests(unittest.TestCase):
    def _assert_result_matches_reference(
        self, actual: CUDARestrictedWHEpisodeResult, expected: object
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

    def test_quiet_segment_matches_reference_bitwise_and_has_no_episode(self):
        state = _state(17)
        tile = _tile(node_count=5)
        spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        result = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_result_matches_reference(result, reference)
        self.assertEqual(result.episodes, ())
        self.assertEqual(result.runtime_identity["partitioned_gpu_step_count"], 0)
        self.assertEqual(result.runtime_identity["quiet_fused_attempt_count"], 1)
        self.assertFalse(any(result.claims.values()))
        self.assertEqual(dict(result.claims), dict(EPISODE_CLAIMS))

    def test_route_heavy_wave_matches_reference_and_forms_whole_episodes(self):
        lane_count = 12
        step_count = 8
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
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        result = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_result_matches_reference(result, reference)
        self.assertEqual(len(result.route_records), lane_count * step_count)
        self.assertEqual(len(result.episodes), lane_count)
        self.assertEqual(
            [episode.route_count for episode in result.episodes],
            [step_count] * lane_count,
        )
        self.assertEqual(
            result.runtime_identity["partitioned_gpu_step_count"], step_count
        )
        self.assertEqual(
            result.runtime_identity["longest_episode_route_count"], step_count
        )

    def test_six_worker_episode_result_matches_serial_and_reference(self):
        lane_count = 12
        step_count = 3
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
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        serial = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        parallel = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=6
        )
        self._assert_result_matches_reference(serial, reference)
        self._assert_result_matches_reference(parallel, reference)
        self.assertEqual(parallel.runtime_identity["route_worker_count"], 6)
        self.assertEqual(
            parallel.runtime_identity["route_worker_start_method"], "spawn"
        )

    def test_mixed_guarded_and_quiet_wave_matches_reference_bitwise(self):
        state = _state(9)
        tile = _tile(
            node_count=4,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(3, 1, _backend())
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        result = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_result_matches_reference(result, reference)
        self.assertGreater(len(result.route_records), 0)
        self.assertLess(len(result.route_records), 9 * 3)

    def test_quiet_prefix_then_episode_matches_reference_bitwise(self):
        base_tile = _tile(
            node_count=4,
            step=0.1,
            encounter_radius=0.1,
            acceleration_bound=1.0,
        )
        tile = replace(
            base_tile,
            source_gm_AU3_per_year2=np.asarray((1.0,), dtype=np.float64),
            content_sha256=None,
        )
        state = replace(
            _state(1),
            source_step_years=0.1,
            tracer_relative_positions_AU=CP.asarray(
                ((2.0, 0.0, 0.0),), dtype=CP.float64
            ),
            tracer_relative_velocities_AU_per_year=CP.asarray(
                ((-1.0, 0.5, 0.0),), dtype=CP.float64
            ),
        )
        spec = FixedStepCUDARestrictedWHSpec(
            3,
            1,
            _backend(),
            maximum_primary_dynamical_phase=0.04,
        )
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        result = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_result_matches_reference(result, reference)
        self.assertGreaterEqual(result.route_records[0].map_step_index, 1)

    def test_nonroutable_force_singularity_fails_transactionally(self):
        state = _state(1)
        state = replace(
            state,
            tracer_relative_positions_AU=CP.zeros((1, 3), dtype=CP.float64),
        )
        position_before = state.tracer_relative_positions_AU.copy()
        velocity_before = state.tracer_relative_velocities_AU_per_year.copy()
        tile = _tile(node_count=2)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as expected:
            advance_cuda_restricted_wh_hybrid(
                state, tile, spec, route_worker_count=1
            )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as actual:
            advance_cuda_restricted_wh_episodes(
                state, tile, spec, route_worker_count=1
            )
        self.assertEqual(expected.exception.handoff.reason, "FORCE_SINGULARITY")
        self.assertEqual(actual.exception.handoff.reason, "FORCE_SINGULARITY")
        self.assertIs(actual.exception.handoff.state, state)
        _assert_device_array_equal(
            self, state.tracer_relative_positions_AU, position_before
        )
        _assert_device_array_equal(
            self, state.tracer_relative_velocities_AU_per_year, velocity_before
        )

    def test_zero_lane_segment_is_defined_and_matches_reference(self):
        state = _state(0)
        tile = _tile(node_count=5)
        spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
        reference = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        result = advance_cuda_restricted_wh_episodes(
            state, tile, spec, route_worker_count=1
        )
        self._assert_result_matches_reference(result, reference)
        self.assertEqual(result.route_records, ())
        self.assertEqual(result.episodes, ())

    def test_invalid_state_uses_the_reference_hybrid_contract_error(self):
        state = replace(_state(1), source_schedule_sha256="1" * 64)
        with self.assertRaises(CUDARestrictedWHHybridContractError):
            advance_cuda_restricted_wh_episodes(
                state,
                _tile(node_count=2),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
                route_worker_count=1,
            )

    def test_episode_grouping_splits_on_a_quiet_map_step(self):
        state = _state(1)
        tile = _tile(
            node_count=3,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        result = advance_cuda_restricted_wh_episodes(
            state,
            tile,
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
            route_worker_count=1,
        )
        first, second = result.route_records
        split = (first, replace(second, map_step_index=3))
        episodes = group_restricted_wh_route_episodes(split)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(
            [
                (episode.first_map_step_index, episode.stop_map_step_index)
                for episode in episodes
            ],
            [(0, 1), (3, 4)],
        )

    def test_terminal_event_is_transactional_and_matches_reference_handoff(self):
        state = _state(1)
        position_before = state.tracer_relative_positions_AU.copy()
        velocity_before = state.tracer_relative_velocities_AU_per_year.copy()
        tile = _tile(node_count=2, encounter_radius=1.0)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as expected:
            advance_cuda_restricted_wh_hybrid(
                state, tile, spec, route_worker_count=1
            )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as actual:
            advance_cuda_restricted_wh_episodes(
                state, tile, spec, route_worker_count=1
            )
        self.assertEqual(actual.exception.handoff.reason, "TERMINAL_EVENT")
        self.assertEqual(
            actual.exception.handoff.tracer_index,
            expected.exception.handoff.tracer_index,
        )
        actual_event = actual.exception.handoff.route_result.events[0]
        expected_event = expected.exception.handoff.route_result.events[0]
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
            self.assertEqual(getattr(actual_event, field), getattr(expected_event, field))
        self.assertTrue(
            np.array_equal(
                actual_event.relative_position_AU,
                expected_event.relative_position_AU,
            )
        )
        self.assertTrue(
            np.array_equal(
                actual_event.relative_velocity_AU_per_year,
                expected_event.relative_velocity_AU_per_year,
            )
        )
        self.assertIs(actual.exception.handoff.state, state)
        _assert_device_array_equal(
            self, state.tracer_relative_positions_AU, position_before
        )
        _assert_device_array_equal(
            self, state.tracer_relative_velocities_AU_per_year, velocity_before
        )


if __name__ == "__main__":
    unittest.main()
