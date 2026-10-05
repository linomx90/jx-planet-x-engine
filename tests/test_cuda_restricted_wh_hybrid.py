from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    CUDARestrictedWHIntegrationError,
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridContractError,
    CUDARestrictedWHHybridIntegrationError,
    DEFAULT_ROUTE_WORKER_COUNT,
    advance_cuda_restricted_wh_hybrid,
)
from jxplanetx.restricted_wh_route import RestrictedWHRouteSpec

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)


@GPU_REQUIRED
class CUDARestrictedWHHybridTests(unittest.TestCase):
    def test_quiet_segment_is_bitwise_identical_to_fused_cuda(self):
        state = _state(17)
        tile = _tile(node_count=5)
        spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
        expected = advance_cuda_restricted_wh_fused(state, tile, spec)
        result = advance_cuda_restricted_wh_hybrid(state, tile, spec)
        self.assertEqual(result.route_records, ())
        self.assertEqual(
            result.state.global_source_step,
            expected.state.global_source_step,
        )
        for actual, reference in (
            (
                result.state.tracer_relative_positions_AU,
                expected.state.tracer_relative_positions_AU,
            ),
            (
                result.state.tracer_relative_velocities_AU_per_year,
                expected.state.tracer_relative_velocities_AU_per_year,
            ),
            (result.state.tracer_indices, expected.state.tracer_indices),
            (
                result.minimum_source_distance_AU,
                expected.minimum_endpoint_source_distance_AU,
            ),
        ):
            self.assertTrue(bool(CP.array_equal(actual, reference)))
        self.assertEqual(
            result.maximum_kepler_iterations,
            expected.maximum_kepler_iterations,
        )
        with self.assertRaises(TypeError):
            result.runtime_identity["method_id"] = "forged"
        with self.assertRaises(TypeError):
            result.runtime_identity["quiet_cuda_runtime_identity"][
                "device_uuid"
            ] = "forged"

    def test_swept_guarded_lane_rejoins_while_quiet_lanes_remain_bitwise(self):
        state = _state(4)
        tile = _tile(
            node_count=2,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        with self.assertRaises(CUDARestrictedWHIntegrationError):
            advance_cuda_restricted_wh_fused(state, tile, spec)

        position_before = state.tracer_relative_positions_AU.copy()
        velocity_before = state.tracer_relative_velocities_AU_per_year.copy()
        result = advance_cuda_restricted_wh_hybrid(state, tile, spec)
        self.assertEqual(len(result.route_records), 1)
        record = result.route_records[0]
        self.assertEqual(record.lane_index, 0)
        self.assertEqual(record.tracer_index, 0)
        self.assertEqual(record.guard_reason, "SWEPT_SOURCE_RADIUS_GUARD")
        self.assertFalse(record.route_result.terminated_at_event)
        self.assertEqual(result.state.global_source_step, 1)

        quiet_state = replace(
            state,
            tracer_relative_positions_AU=CP.ascontiguousarray(
                state.tracer_relative_positions_AU[1:]
            ),
            tracer_relative_velocities_AU_per_year=CP.ascontiguousarray(
                state.tracer_relative_velocities_AU_per_year[1:]
            ),
            tracer_indices=CP.ascontiguousarray(state.tracer_indices[1:]),
        )
        quiet = advance_cuda_restricted_wh_fused(quiet_state, tile, spec)
        self.assertTrue(
            bool(
                CP.array_equal(
                    result.state.tracer_relative_positions_AU[1:],
                    quiet.state.tracer_relative_positions_AU,
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    result.state.tracer_relative_velocities_AU_per_year[1:],
                    quiet.state.tracer_relative_velocities_AU_per_year,
                )
            )
        )
        self.assertTrue(
            bool(CP.array_equal(state.tracer_relative_positions_AU, position_before))
        )
        self.assertTrue(
            bool(CP.array_equal(state.tracer_relative_velocities_AU_per_year, velocity_before))
        )

    def test_consecutive_routes_split_bitwise_like_continuous(self):
        state = _state(1)
        continuous_tile = _tile(
            node_count=3,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        continuous = advance_cuda_restricted_wh_hybrid(
            state,
            continuous_tile,
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        first = advance_cuda_restricted_wh_hybrid(
            state,
            _tile(
                start=0,
                node_count=2,
                encounter_radius=0.997,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )
        second = advance_cuda_restricted_wh_hybrid(
            first.state,
            _tile(
                start=1,
                node_count=2,
                encounter_radius=0.997,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )
        self.assertEqual(len(continuous.route_records), 2)
        self.assertEqual(len(first.route_records), 1)
        self.assertEqual(len(second.route_records), 1)
        self.assertTrue(
            bool(
                CP.array_equal(
                    continuous.state.tracer_relative_positions_AU,
                    second.state.tracer_relative_positions_AU,
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    continuous.state.tracer_relative_velocities_AU_per_year,
                    second.state.tracer_relative_velocities_AU_per_year,
                )
            )
        )

    def test_multiple_guarded_lanes_are_routed_in_input_lane_order(self):
        state = _state(3)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[0, 0] = 1.0
        positions[1, 0] = 1.0
        velocities[1] = velocities[0]
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        result = advance_cuda_restricted_wh_hybrid(
            state,
            _tile(
                node_count=2,
                encounter_radius=0.997,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )
        self.assertEqual(
            [
                (record.lane_index, record.tracer_index)
                for record in result.route_records
            ],
            [(0, 0), (1, 1)],
        )

    def test_spawn_process_wave_is_bitwise_equal_to_serial_routes(self):
        state = _state(7)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        positions[:6, 0] = 1.0
        velocities[:6] = velocities[0]
        state = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        tile = _tile(
            node_count=2,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        serial = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=1
        )
        parallel = advance_cuda_restricted_wh_hybrid(
            state, tile, spec, route_worker_count=3
        )
        for left, right in (
            (
                serial.state.tracer_relative_positions_AU,
                parallel.state.tracer_relative_positions_AU,
            ),
            (
                serial.state.tracer_relative_velocities_AU_per_year,
                parallel.state.tracer_relative_velocities_AU_per_year,
            ),
            (serial.state.tracer_indices, parallel.state.tracer_indices),
            (
                serial.minimum_source_distance_AU,
                parallel.minimum_source_distance_AU,
            ),
        ):
            self.assertTrue(bool(CP.array_equal(left, right)))
        self.assertEqual(len(serial.route_records), 6)
        self.assertEqual(len(parallel.route_records), 6)
        for serial_record, parallel_record in zip(
            serial.route_records, parallel.route_records, strict=True
        ):
            self.assertEqual(
                (
                    serial_record.map_step_index,
                    serial_record.lane_index,
                    serial_record.tracer_index,
                    serial_record.guard_reason,
                ),
                (
                    parallel_record.map_step_index,
                    parallel_record.lane_index,
                    parallel_record.tracer_index,
                    parallel_record.guard_reason,
                ),
            )
            self.assertTrue(
                np.array_equal(
                    serial_record.route_result.endpoint_relative_position_AU,
                    parallel_record.route_result.endpoint_relative_position_AU,
                )
            )
            self.assertTrue(
                np.array_equal(
                    serial_record.route_result.endpoint_relative_velocity_AU_per_year,
                    parallel_record.route_result.endpoint_relative_velocity_AU_per_year,
                )
            )
            self.assertEqual(
                serial_record.route_result.source_segment_sha256,
                parallel_record.route_result.source_segment_sha256,
            )
        self.assertEqual(
            parallel.runtime_identity["route_worker_start_method"], "spawn"
        )
        self.assertEqual(parallel.runtime_identity["route_worker_count"], 3)

    def test_default_route_workers_match_six_physical_cores(self):
        self.assertEqual(DEFAULT_ROUTE_WORKER_COUNT, 6)

    def test_failure_after_quiet_prefix_commits_only_on_final_success(self):
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
        state = _state(1)
        state = replace(
            state,
            source_step_years=0.1,
            tracer_relative_positions_AU=CP.asarray(
                ((2.0, 0.0, 0.0),), dtype=CP.float64
            ),
            tracer_relative_velocities_AU_per_year=CP.asarray(
                ((-1.0, 0.5, 0.0),), dtype=CP.float64
            ),
        )
        result = advance_cuda_restricted_wh_hybrid(
            state,
            tile,
            FixedStepCUDARestrictedWHSpec(
                3,
                1,
                _backend(),
                maximum_primary_dynamical_phase=0.04,
            ),
        )
        self.assertEqual(result.state.global_source_step, 3)
        self.assertGreaterEqual(len(result.route_records), 1)
        self.assertEqual(result.route_records[0].map_step_index, 1)
        self.assertTrue(
            bool(CP.all(CP.isfinite(result.state.tracer_relative_positions_AU)))
        )

    def test_terminal_event_aborts_transactionally_with_route_witness(self):
        state = _state(1)
        position_before = state.tracer_relative_positions_AU.copy()
        velocity_before = state.tracer_relative_velocities_AU_per_year.copy()
        tile = _tile(node_count=2, encounter_radius=1.0)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as raised:
            advance_cuda_restricted_wh_hybrid(state, tile, spec)
        handoff = raised.exception.handoff
        self.assertIs(handoff.state, state)
        self.assertEqual(handoff.reason, "TERMINAL_EVENT")
        self.assertIsNotNone(handoff.route_result)
        self.assertTrue(handoff.route_result.terminated_at_event)
        self.assertEqual(handoff.committed_map_steps, 0)
        self.assertTrue(
            bool(CP.array_equal(state.tracer_relative_positions_AU, position_before))
        )
        self.assertTrue(
            bool(CP.array_equal(state.tracer_relative_velocities_AU_per_year, velocity_before))
        )

    def test_outer_boundary_crossing_is_continuously_located_and_transactional(self):
        state = _state(1)
        position = state.tracer_relative_positions_AU.copy()
        velocity = state.tracer_relative_velocities_AU_per_year.copy()
        position[0] = CP.asarray((0.9, 0.0, 0.0), dtype=CP.float64)
        velocity[0] = CP.asarray((2.0, 0.0, 0.0), dtype=CP.float64)
        state = replace(
            state,
            tracer_relative_positions_AU=position,
            tracer_relative_velocities_AU_per_year=velocity,
        )
        with self.assertRaises(CUDARestrictedWHHybridIntegrationError) as raised:
            advance_cuda_restricted_wh_hybrid(
                state,
                _tile(node_count=2),
                FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    _backend(),
                    primary_outer_boundary_AU=0.9005,
                ),
                route_spec=RestrictedWHRouteSpec(
                    primary_outer_boundary_AU=0.9005
                ),
            )
        self.assertEqual(raised.exception.handoff.reason, "TERMINAL_EVENT")
        route = raised.exception.handoff.route_result
        self.assertIsNotNone(route)
        self.assertEqual(len(route.events), 1)
        self.assertEqual(route.events[0].event_type, "PRIMARY_BOUNDARY_OUTWARD")
        self.assertAlmostEqual(route.events[0].surface_distance_AU, 0.9005)

    def test_outer_boundary_policy_must_match_cuda_spec_exactly(self):
        with self.assertRaisesRegex(
            CUDARestrictedWHHybridContractError,
            "outer boundaries must be exactly equal",
        ):
            advance_cuda_restricted_wh_hybrid(
                _state(1),
                _tile(node_count=2),
                FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    _backend(),
                    primary_outer_boundary_AU=10_000.0,
                ),
                route_spec=RestrictedWHRouteSpec(
                    primary_outer_boundary_AU=9_999.0
                ),
            )

    def test_zero_lane_segment_is_defined(self):
        result = advance_cuda_restricted_wh_hybrid(
            _state(0),
            _tile(node_count=3),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        self.assertEqual(result.state.global_source_step, 2)
        self.assertEqual(result.route_records, ())
        self.assertEqual(result.minimum_source_distance_AU.shape, (0,))


if __name__ == "__main__":
    unittest.main()
