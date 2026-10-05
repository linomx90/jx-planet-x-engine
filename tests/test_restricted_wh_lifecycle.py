from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
from jxplanetx.restricted_wh_lifecycle import (
    ACTIVE,
    REMOVED_OUTER_BOUNDARY,
    REMOVED_SOURCE_RADIUS,
    RestrictedWHLifecycleContractError,
    RestrictedWHLifecycleIntegrationError,
    RestrictedWHLifecycleState,
    advance_cuda_restricted_wh_lifecycle,
    initialize_restricted_wh_lifecycle,
    removal_record_sha256,
)
from jxplanetx.restricted_wh_route import RestrictedWHRouteSpec

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)


def _source_contact_state(count: int = 2):
    state = _state(count)
    positions = state.tracer_relative_positions_AU.copy()
    velocities = state.tracer_relative_velocities_AU_per_year.copy()
    positions[0] = CP.asarray((1.001, 0.0, 0.0), dtype=CP.float64)
    velocities[0] = CP.asarray((-2.0, 0.0, 0.0), dtype=CP.float64)
    return replace(
        state,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
    )


@GPU_REQUIRED
class RestrictedWHLifecycleTests(unittest.TestCase):
    def test_source_contact_freezes_lane_while_neighbor_continues(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        tile = _tile(
            node_count=4,
            encounter_radius=1.0,
            acceleration_bound=100.0,
        )
        result = advance_cuda_restricted_wh_lifecycle(
            initial,
            tile,
            FixedStepCUDARestrictedWHSpec(3, 1, _backend()),
            route_worker_count=1,
        )
        self.assertEqual(result.state.dynamics_state.global_source_step, 3)
        self.assertEqual(result.state.tracer_lifecycle.tolist(), [1, 0])
        self.assertEqual(len(result.new_removal_records), 1)
        removal = result.new_removal_records[0]
        self.assertEqual(removal.lifecycle_code, REMOVED_SOURCE_RADIUS)
        self.assertEqual(removal.tracer_index, 0)
        self.assertEqual(removal.event.event_type, "SOURCE_RADIUS_CONTACT")
        self.assertEqual(removal.content_sha256, removal_record_sha256(removal))
        with self.assertRaises(ValueError):
            result.state.tracer_lifecycle[0] = ACTIVE
        with self.assertRaises(TypeError):
            result.claims["physical_collision_inferred"] = True
        with self.assertRaises(TypeError):
            result.runtime_identity["route_worker_count"] = 99
        with self.assertRaises(TypeError):
            result.runtime_identity["hybrid_runtime_identities"][0][
                "method_id"
            ] = "forged"
        frozen_position = CP.asnumpy(
            result.state.dynamics_state.tracer_relative_positions_AU[0]
        )
        frozen_velocity = CP.asnumpy(
            result.state.dynamics_state.tracer_relative_velocities_AU_per_year[0]
        )
        self.assertTrue(
            np.array_equal(frozen_position, removal.event.relative_position_AU)
        )
        self.assertTrue(
            np.array_equal(
                frozen_velocity,
                removal.event.relative_velocity_AU_per_year,
            )
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(frozen_position)),
            1.0,
            delta=removal.event.surface_tolerance_AU,
        )

        continued = advance_cuda_restricted_wh_lifecycle(
            result.state,
            _tile(
                start=3,
                node_count=3,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
            route_worker_count=1,
        )
        self.assertEqual(continued.state.dynamics_state.global_source_step, 5)
        self.assertEqual(len(continued.new_removal_records), 0)
        self.assertEqual(len(continued.state.removal_records), 1)
        self.assertTrue(
            bool(
                CP.array_equal(
                    continued.state.dynamics_state.tracer_relative_positions_AU[0],
                    result.state.dynamics_state.tracer_relative_positions_AU[0],
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    continued.state.dynamics_state.tracer_relative_velocities_AU_per_year[0],
                    result.state.dynamics_state.tracer_relative_velocities_AU_per_year[0],
                )
            )
        )
        self.assertFalse(
            bool(
                CP.array_equal(
                    continued.state.dynamics_state.tracer_relative_positions_AU[1],
                    result.state.dynamics_state.tracer_relative_positions_AU[1],
                )
            )
        )

    def test_outer_boundary_is_operational_removal_not_ejection_claim(self):
        dynamics = _state(1)
        positions = dynamics.tracer_relative_positions_AU.copy()
        velocities = dynamics.tracer_relative_velocities_AU_per_year.copy()
        positions[0] = CP.asarray((0.9, 0.0, 0.0), dtype=CP.float64)
        velocities[0] = CP.asarray((2.0, 0.0, 0.0), dtype=CP.float64)
        dynamics = replace(
            dynamics,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        boundary = 0.9005
        spec = FixedStepCUDARestrictedWHSpec(
            2,
            1,
            _backend(),
            primary_outer_boundary_AU=boundary,
        )
        result = advance_cuda_restricted_wh_lifecycle(
            initialize_restricted_wh_lifecycle(dynamics),
            _tile(node_count=3),
            spec,
            route_spec=RestrictedWHRouteSpec(
                primary_outer_boundary_AU=boundary
            ),
            route_worker_count=1,
        )
        removal = result.new_removal_records[0]
        self.assertEqual(removal.lifecycle_code, REMOVED_OUTER_BOUNDARY)
        self.assertEqual(
            removal.event.event_type,
            "PRIMARY_BOUNDARY_OUTWARD",
        )
        self.assertAlmostEqual(
            removal.event.surface_distance_AU,
            boundary,
            delta=removal.event.surface_tolerance_AU,
        )
        self.assertFalse(result.claims["physical_ejection_inferred"])

        frozen_position = (
            result.state.dynamics_state.tracer_relative_positions_AU.copy()
        )
        frozen_velocity = (
            result.state.dynamics_state.tracer_relative_velocities_AU_per_year.copy()
        )
        continued = advance_cuda_restricted_wh_lifecycle(
            result.state,
            _tile(start=2, node_count=3),
            FixedStepCUDARestrictedWHSpec(
                2,
                1,
                _backend(),
                primary_outer_boundary_AU=boundary,
            ),
            route_spec=RestrictedWHRouteSpec(
                primary_outer_boundary_AU=boundary
            ),
            route_worker_count=1,
        )
        self.assertEqual(continued.state.dynamics_state.global_source_step, 4)
        self.assertEqual(continued.new_removal_records, ())
        self.assertTrue(
            bool(
                CP.array_equal(
                    continued.state.dynamics_state.tracer_relative_positions_AU,
                    frozen_position,
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    continued.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                    frozen_velocity,
                )
            )
        )

    def test_terminal_split_matches_uninterrupted_bitwise_and_record_hash(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        full = advance_cuda_restricted_wh_lifecycle(
            initial,
            _tile(
                node_count=4,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(3, 1, _backend()),
            route_worker_count=1,
        )
        first = advance_cuda_restricted_wh_lifecycle(
            initial,
            _tile(
                node_count=2,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
            route_worker_count=1,
        )
        second = advance_cuda_restricted_wh_lifecycle(
            first.state,
            _tile(
                start=1,
                node_count=3,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
            route_worker_count=1,
        )
        for left, right in (
            (
                full.state.dynamics_state.tracer_relative_positions_AU,
                second.state.dynamics_state.tracer_relative_positions_AU,
            ),
            (
                full.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                second.state.dynamics_state.tracer_relative_velocities_AU_per_year,
            ),
            (
                full.state.dynamics_state.tracer_indices,
                second.state.dynamics_state.tracer_indices,
            ),
        ):
            self.assertTrue(bool(CP.array_equal(left, right)))
        self.assertTrue(
            np.array_equal(
                full.state.tracer_lifecycle,
                second.state.tracer_lifecycle,
            )
        )
        self.assertEqual(
            full.state.removal_records[0].content_sha256,
            second.state.removal_records[0].content_sha256,
        )

    def test_parallel_simultaneous_terminals_preserve_fixed_lane_order(self):
        dynamics = _state(3)
        positions = dynamics.tracer_relative_positions_AU.copy()
        velocities = dynamics.tracer_relative_velocities_AU_per_year.copy()
        positions[0] = CP.asarray((1.001, 0.0, 0.0), dtype=CP.float64)
        positions[1] = CP.asarray((1.002, 0.0, 0.0), dtype=CP.float64)
        velocities[0] = CP.asarray((-2.0, 0.0, 0.0), dtype=CP.float64)
        velocities[1] = CP.asarray((-3.0, 0.0, 0.0), dtype=CP.float64)
        dynamics = replace(
            dynamics,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        result = advance_cuda_restricted_wh_lifecycle(
            initialize_restricted_wh_lifecycle(dynamics),
            _tile(
                node_count=2,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
            route_worker_count=3,
        )
        self.assertEqual(result.state.tracer_lifecycle.tolist(), [1, 1, 0])
        self.assertEqual(
            [record.lane_index for record in result.new_removal_records],
            [0, 1],
        )
        self.assertEqual(
            [record.tracer_index for record in result.new_removal_records],
            [0, 1],
        )
        for lane, record in enumerate(result.new_removal_records):
            self.assertTrue(
                np.array_equal(
                    CP.asnumpy(
                        result.state.dynamics_state
                        .tracer_relative_positions_AU[lane]
                    ),
                    record.event.relative_position_AU,
                )
            )

    def test_removed_lane_tampering_and_active_initial_surface_reject(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        result = advance_cuda_restricted_wh_lifecycle(
            initial,
            _tile(
                node_count=2,
                encounter_radius=1.0,
                acceleration_bound=100.0,
            ),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
            route_worker_count=1,
        )
        forged_positions = (
            result.state.dynamics_state.tracer_relative_positions_AU.copy()
        )
        forged_positions[0, 0] += 1.0e-6
        forged = RestrictedWHLifecycleState(
            dynamics_state=replace(
                result.state.dynamics_state,
                tracer_relative_positions_AU=forged_positions,
            ),
            tracer_lifecycle=result.state.tracer_lifecycle,
            removal_records=result.state.removal_records,
        )
        with self.assertRaisesRegex(
            RestrictedWHLifecycleContractError,
            "removed tracer position changed",
        ):
            advance_cuda_restricted_wh_lifecycle(
                forged,
                _tile(
                    start=1,
                    node_count=2,
                    encounter_radius=1.0,
                    acceleration_bound=100.0,
                ),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
                route_worker_count=1,
            )

        active = _state(1)
        with self.assertRaisesRegex(
            RestrictedWHLifecycleContractError,
            "active tracer begins on or inside",
        ):
            advance_cuda_restricted_wh_lifecycle(
                initialize_restricted_wh_lifecycle(active),
                _tile(node_count=2, encounter_radius=1.0),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
                route_worker_count=1,
            )

    def test_late_route_failure_is_publicly_transactional(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        positions_before = (
            initial.dynamics_state.tracer_relative_positions_AU.copy()
        )
        with self.assertRaises(RestrictedWHLifecycleIntegrationError) as raised:
            advance_cuda_restricted_wh_lifecycle(
                initial,
                _tile(node_count=2, encounter_radius=1.0e-4),
                FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    _backend(),
                    maximum_primary_dynamical_phase=1.0e-6,
                ),
                route_worker_count=1,
            )
        self.assertIs(raised.exception.handoff.state, initial)
        self.assertEqual(raised.exception.handoff.committed_map_steps, 0)
        self.assertTrue(
            bool(
                CP.array_equal(
                    initial.dynamics_state.tracer_relative_positions_AU,
                    positions_before,
                )
            )
        )

    def test_lifecycle_array_and_ledger_contract_are_strict(self):
        dynamics = _state(2)
        with self.assertRaises(RestrictedWHLifecycleContractError):
            RestrictedWHLifecycleState(
                dynamics_state=dynamics,
                tracer_lifecycle=np.zeros(2, dtype=np.int64),
            )
        with self.assertRaises(RestrictedWHLifecycleContractError):
            RestrictedWHLifecycleState(
                dynamics_state=dynamics,
                tracer_lifecycle=np.asarray((ACTIVE, 9), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
