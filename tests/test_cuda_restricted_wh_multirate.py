from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.cuda_restricted_wh_multirate import (
    CUDARestrictedWHMultirateContractError,
    CUDARestrictedWHMultirateIntegrationError,
    FrozenStaticRungPlan,
    RESULT_CLAIMS,
    StaticMultirateCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_static_multirate,
    static_multirate_input_state_sha256,
    static_multirate_spec_sha256,
    static_rung_plan_sha256,
    verify_static_rung_plan,
)
from jxplanetx.engine import Provenance
from jxplanetx.restricted_wh_route import (
    RestrictedWHRouteSpec,
    route_restricted_wh_map_step,
)

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    SCHEDULE_SHA256,
    _backend,
    _state,
    _tile,
)
from tests.test_cuda_restricted_wh_episode import _assert_device_array_equal


def _provenance() -> Provenance:
    return Provenance(
        source_id="jx-test.static-rung-plan",
        citation="Deterministic multirate contract fixture",
        version="v1",
        sha256=hashlib.sha256(b"static-rung-plan-v1").hexdigest(),
    )


def _plan(
    ratios: tuple[int, ...],
    *,
    start: int = 0,
    total: int = 8,
    allowed: tuple[int, ...] = (1, 2, 4, 8),
    tracer_indices: np.ndarray | None = None,
    source_tile_sha256: str = SCHEDULE_SHA256,
    input_state_sha256: str | None = None,
    execution_policy_sha256: str | None = None,
) -> FrozenStaticRungPlan:
    if tracer_indices is None:
        tracer_indices = np.arange(len(ratios), dtype=np.uint64)
    return FrozenStaticRungPlan(
        plan_id=f"plan-{start}-{total}-{'-'.join(map(str, ratios))}",
        global_source_step=start,
        total_source_steps=total,
        source_schedule_sha256=SCHEDULE_SHA256,
        source_tile_sha256=source_tile_sha256,
        tracer_indices=tracer_indices,
        source_steps_per_map_step=np.asarray(ratios, dtype=np.uint32),
        allowed_source_steps_per_map_step=allowed,
        selection_basis="PREDECLARED_OUTCOME_BLIND_STATIC_RUNG_FIXTURE",
        accuracy_policy_id="INDEPENDENT_DOP853_FINE_LATTICE_TEST_FIXTURE_V1",
        accuracy_validation_sha256=hashlib.sha256(
            b"static-rung-test-accuracy-validation-v1"
        ).hexdigest(),
        input_state_sha256=(
            hashlib.sha256(b"static-rung-test-input-state-v1").hexdigest()
            if input_state_sha256 is None
            else input_state_sha256
        ),
        execution_policy_sha256=(
            hashlib.sha256(b"static-rung-test-policy-v1").hexdigest()
            if execution_policy_sha256 is None
            else execution_policy_sha256
        ),
        provenance=_provenance(),
    )


def _compatible_state(count: int, tile):
    state = _state(count, global_source_step=tile.global_source_step)
    return replace(
        state,
        source_step_years=tile.source_step_years,
        time_scale=tile.time_scale,
        frame=tile.frame,
        axes=tile.axes,
        coordinate_system=tile.coordinate_system,
        length_unit=tile.length_unit,
        time_unit=tile.time_unit,
        unit_system_id=tile.unit_system_id,
        source_ids=tile.source_ids,
        primary_source_id=tile.primary_source_id,
        source_schedule_sha256=tile.source_schedule_sha256,
    )


def _bound_plan(
    tile,
    state,
    ratios: tuple[int, ...],
    *,
    spec: StaticMultirateCUDARestrictedWHSpec | None = None,
    **kwargs,
) -> FrozenStaticRungPlan:
    if spec is None:
        spec = StaticMultirateCUDARestrictedWHSpec(
            int(kwargs.get("total", 8)),
            _backend(),
        )
    return _plan(
        ratios,
        source_tile_sha256=tile.content_sha256,
        input_state_sha256=static_multirate_input_state_sha256(
            state, spec.backend
        ),
        execution_policy_sha256=static_multirate_spec_sha256(spec),
        **kwargs,
    )


def _consistent_secondary_tile(*, node_count: int, step: float):
    """Return a source path whose node velocities differentiate its positions."""

    tile = _tile(
        node_count=node_count,
        step=step,
        secondary=True,
        acceleration_bound=100.0,
    )
    positions = np.array(tile.source_relative_positions_AU, copy=True)
    velocities = np.array(tile.source_relative_velocities_AU_per_year, copy=True)
    radius = 5.2
    angular_rate = float(
        np.sqrt(np.sum(tile.source_gm_AU3_per_year2) / radius**3)
    )
    for node in range(node_count):
        angle = angular_rate * node * step
        positions[node, 1] = (
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0,
        )
        velocities[node, 1] = (
            -radius * angular_rate * np.sin(angle),
            radius * angular_rate * np.cos(angle),
            0.0,
        )
    return replace(
        tile,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        content_sha256=None,
    )


class StaticRungPlanContractTests(unittest.TestCase):
    def test_plan_is_immutable_content_addressed_and_claims_stay_false(self):
        indices = np.arange(4, dtype=np.uint64)
        ratios = np.asarray((1, 2, 4, 8), dtype=np.uint32)
        plan = FrozenStaticRungPlan(
            plan_id="immutable-plan",
            global_source_step=0,
            total_source_steps=8,
            source_schedule_sha256=SCHEDULE_SHA256,
            source_tile_sha256=SCHEDULE_SHA256,
            tracer_indices=indices,
            source_steps_per_map_step=ratios,
            allowed_source_steps_per_map_step=(1, 2, 4, 8),
            selection_basis="PREDECLARED_TEST_ASSIGNMENT",
            accuracy_policy_id="INDEPENDENT_DOP853_FINE_LATTICE_TEST_FIXTURE_V1",
            accuracy_validation_sha256=hashlib.sha256(
                b"static-rung-test-accuracy-validation-v1"
            ).hexdigest(),
            input_state_sha256=hashlib.sha256(
                b"static-rung-test-input-state-v1"
            ).hexdigest(),
            execution_policy_sha256=hashlib.sha256(
                b"static-rung-test-policy-v1"
            ).hexdigest(),
            provenance=_provenance(),
        )
        digest = plan.content_sha256
        indices[0] = 99
        ratios[0] = 8
        self.assertEqual(plan.tracer_indices[0], 0)
        self.assertEqual(plan.source_steps_per_map_step[0], 1)
        self.assertFalse(plan.tracer_indices.flags.writeable)
        self.assertEqual(static_rung_plan_sha256(plan), digest)
        self.assertEqual(verify_static_rung_plan(plan), digest)
        self.assertTrue(RESULT_CLAIMS)
        self.assertFalse(any(RESULT_CLAIMS.values()))

    def test_plan_rejects_nondyadic_unaligned_or_incomplete_rungs(self):
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError, "powers of two"
        ):
            _plan((1,), allowed=(1, 3))
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError, "does not divide"
        ):
            _plan((4,), total=6, allowed=(1, 2, 4))
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError, "block boundary"
        ):
            _plan((2,), start=1, total=8, allowed=(1, 2))
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError, "outside"
        ):
            _plan((4,), allowed=(1, 2))

    def test_expected_digest_detects_changed_assignment(self):
        plan = _plan((1, 2, 4, 8))
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError, "does not match"
        ):
            replace(
                plan,
                source_steps_per_map_step=np.asarray(
                    (1, 2, 8, 8), dtype=np.uint32
                ),
            )


@GPU_REQUIRED
class StaticMultirateGPUTests(unittest.TestCase):
    def assertStateArraysEqual(self, left, right) -> None:
        self.assertEqual(left.global_source_step, right.global_source_step)
        _assert_device_array_equal(
            self,
            left.tracer_relative_positions_AU,
            right.tracer_relative_positions_AU,
        )
        _assert_device_array_equal(
            self,
            left.tracer_relative_velocities_AU_per_year,
            right.tracer_relative_velocities_AU_per_year,
        )
        _assert_device_array_equal(self, left.tracer_indices, right.tracer_indices)

    def test_uniform_fine_rung_is_bitwise_fixed_step(self):
        tile = _tile(node_count=9, secondary=True)
        state = _compatible_state(8, tile)
        spec = StaticMultirateCUDARestrictedWHSpec(8, _backend())
        plan = _bound_plan(tile, state, (1,) * 8, spec=spec, allowed=(1,))
        multirate = advance_cuda_restricted_wh_static_multirate(
            state,
            tile,
            spec,
            plan,
        )
        fixed = advance_cuda_restricted_wh_fused(
            state,
            tile,
            FixedStepCUDARestrictedWHSpec(8, 1, _backend()),
        )
        self.assertStateArraysEqual(multirate.state, fixed.state)
        _assert_device_array_equal(
            self,
            multirate.minimum_endpoint_source_distance_AU,
            fixed.minimum_endpoint_source_distance_AU,
        )
        self.assertEqual(multirate.maximum_kepler_iterations, fixed.maximum_kepler_iterations)
        self.assertEqual(multirate.rung_lane_counts, ((1, 8),))

    def test_state_tile_metadata_mismatch_fails_before_cuda_work(self):
        tile = _tile(node_count=5)
        state = replace(_compatible_state(1, tile), frame="WRONG_FRAME")
        spec = StaticMultirateCUDARestrictedWHSpec(4, _backend())
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError,
            "state and source tile disagree on frame",
        ):
            advance_cuda_restricted_wh_static_multirate(
                state,
                tile,
                spec,
                _bound_plan(
                    tile, state, (1,), spec=spec, total=4, allowed=(1,)
                ),
            )

    def test_plan_is_bound_to_exact_input_state_bytes(self):
        tile = _tile(node_count=5)
        state = _compatible_state(2, tile)
        spec = StaticMultirateCUDARestrictedWHSpec(4, _backend())
        plan = _bound_plan(
            tile,
            state,
            (1, 2),
            spec=spec,
            total=4,
            allowed=(1, 2),
        )
        changed_positions = state.tracer_relative_positions_AU.copy()
        changed_positions[1, 0] = CP.nextafter(
            changed_positions[1, 0], CP.asarray(CP.inf)
        )
        changed = replace(
            state,
            tracer_relative_positions_AU=changed_positions,
        )
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError,
            "different input-state bytes or metadata",
        ):
            advance_cuda_restricted_wh_static_multirate(
                changed,
                tile,
                spec,
                plan,
            )

    def test_plan_is_bound_to_exact_execution_policy(self):
        tile = _tile(node_count=5)
        state = _compatible_state(1, tile)
        original_spec = StaticMultirateCUDARestrictedWHSpec(
            4, _backend(), threads_per_block=128
        )
        plan = _bound_plan(
            tile,
            state,
            (1,),
            spec=original_spec,
            total=4,
            allowed=(1,),
        )
        changed_spec = replace(original_spec, threads_per_block=64)
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError,
            "different multirate execution policy",
        ):
            advance_cuda_restricted_wh_static_multirate(
                state,
                tile,
                changed_spec,
                plan,
            )

    def test_mixed_rungs_equal_independent_fixed_groups_and_repeat_bitwise(self):
        tile = _consistent_secondary_tile(node_count=9, step=0.001)
        state = _compatible_state(8, tile)
        ratios = (1, 1, 2, 2, 4, 4, 8, 8)
        spec = StaticMultirateCUDARestrictedWHSpec(8, _backend())
        plan = _bound_plan(tile, state, ratios, spec=spec)
        first = advance_cuda_restricted_wh_static_multirate(
            state, tile, spec, plan
        )
        second = advance_cuda_restricted_wh_static_multirate(
            state, tile, spec, plan
        )
        self.assertStateArraysEqual(first.state, second.state)
        _assert_device_array_equal(
            self,
            first.minimum_endpoint_source_distance_AU,
            second.minimum_endpoint_source_distance_AU,
        )
        for ratio in (1, 2, 4, 8):
            lanes = np.flatnonzero(np.asarray(ratios) == ratio)
            device_lanes = CP.asarray(lanes, dtype=CP.int64)
            subset = replace(
                state,
                tracer_relative_positions_AU=CP.ascontiguousarray(
                    state.tracer_relative_positions_AU[device_lanes]
                ),
                tracer_relative_velocities_AU_per_year=CP.ascontiguousarray(
                    state.tracer_relative_velocities_AU_per_year[device_lanes]
                ),
                tracer_indices=CP.ascontiguousarray(
                    state.tracer_indices[device_lanes]
                ),
            )
            expected = advance_cuda_restricted_wh_fused(
                subset,
                tile,
                FixedStepCUDARestrictedWHSpec(8 // ratio, ratio, _backend()),
            )
            _assert_device_array_equal(
                self,
                first.state.tracer_relative_positions_AU[device_lanes],
                expected.state.tracer_relative_positions_AU,
            )
            _assert_device_array_equal(
                self,
                first.state.tracer_relative_velocities_AU_per_year[device_lanes],
                expected.state.tracer_relative_velocities_AU_per_year,
            )
        self.assertEqual(
            first.rung_lane_counts, ((1, 2), (2, 2), (4, 2), (8, 2))
        )
        self.assertFalse(any(first.claims.values()))
        self.assertFalse(first.runtime_identity["dynamic_rung_changes_allowed"])
        self.assertEqual(
            first.runtime_identity["input_state_sha256"],
            plan.input_state_sha256,
        )
        self.assertEqual(
            first.runtime_identity["execution_policy_sha256"],
            static_multirate_spec_sha256(spec),
        )

    def test_unsafe_plan_is_rejected_before_commit(self):
        state = _state(2)
        positions = state.tracer_relative_positions_AU.copy()
        positions[1] = CP.asarray((5.0e-5, 0.0, 0.0), dtype=CP.float64)
        failing = replace(state, tracer_relative_positions_AU=positions)
        positions_before = failing.tracer_relative_positions_AU.copy()
        velocities_before = failing.tracer_relative_velocities_AU_per_year.copy()
        tile = _tile(node_count=5, encounter_radius=1.0e-4)
        spec = StaticMultirateCUDARestrictedWHSpec(4, _backend())
        plan = _bound_plan(
            tile,
            failing,
            (1, 4),
            spec=spec,
            total=4,
            allowed=(1, 2, 4),
        )
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateIntegrationError,
            "failed closed at source step 0",
        ) as captured:
            advance_cuda_restricted_wh_static_multirate(
                failing,
                tile,
                spec,
                plan,
            )
        handoff = captured.exception.handoff
        self.assertIs(handoff.state, failing)
        self.assertEqual(handoff.failed_source_steps_per_map_step, 4)
        self.assertIsNotNone(handoff.fixed_step_handoff)
        self.assertEqual(handoff.fixed_step_handoff.lane_index, 1)
        self.assertEqual(handoff.fixed_step_handoff.tracer_index, 1)
        self.assertEqual(handoff.fixed_step_handoff.reason, "SOURCE_RADIUS_GUARD")
        _assert_device_array_equal(
            self,
            failing.tracer_relative_positions_AU,
            positions_before,
        )
        _assert_device_array_equal(
            self,
            failing.tracer_relative_velocities_AU_per_year,
            velocities_before,
        )

    def test_secondary_clearance_rejects_globally_identified_lane(self):
        tile = _consistent_secondary_tile(node_count=5, step=0.001)
        state = _compatible_state(4, tile)
        positions = state.tracer_relative_positions_AU.copy()
        velocities = state.tracer_relative_velocities_AU_per_year.copy()
        secondary_position = tile.source_relative_positions_AU[0, 1]
        secondary_velocity = tile.source_relative_velocities_AU_per_year[0, 1]
        positions[3] = CP.asarray(
            secondary_position - np.asarray((1.5e-4, 0.0, 0.0)),
            dtype=CP.float64,
        )
        velocities[3] = CP.asarray(secondary_velocity, dtype=CP.float64)
        failing = replace(
            state,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        before_position = failing.tracer_relative_positions_AU.copy()
        before_velocity = failing.tracer_relative_velocities_AU_per_year.copy()
        spec = StaticMultirateCUDARestrictedWHSpec(4, _backend())
        plan = _bound_plan(
            tile,
            failing,
            (1, 1, 2, 4),
            spec=spec,
            total=4,
            allowed=(1, 2, 4),
        )
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateIntegrationError,
            "MULTIRATE_CLEARANCE_GUARD",
        ) as captured:
            advance_cuda_restricted_wh_static_multirate(
                failing,
                tile,
                spec,
                plan,
            )
        fixed = captured.exception.handoff.fixed_step_handoff
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed.lane_index, 3)
        self.assertEqual(fixed.tracer_index, 3)
        self.assertEqual(fixed.source_index, 1)
        self.assertEqual(fixed.reason, "MULTIRATE_CLEARANCE_GUARD")
        _assert_device_array_equal(
            self, failing.tracer_relative_positions_AU, before_position
        )
        _assert_device_array_equal(
            self, failing.tracer_relative_velocities_AU_per_year, before_velocity
        )

    def test_source_motion_certificate_accepts_fine_and_rejects_coarse(self):
        tile = _consistent_secondary_tile(node_count=5, step=0.001)
        state = _compatible_state(1, tile)
        spec = StaticMultirateCUDARestrictedWHSpec(
            4,
            _backend(),
            maximum_source_motion_sweep_fraction=0.001,
        )
        fine = advance_cuda_restricted_wh_static_multirate(
            state,
            tile,
            spec,
            _bound_plan(
                tile,
                state,
                (1,),
                spec=spec,
                total=4,
                allowed=(1, 2, 4),
            ),
        )
        self.assertEqual(fine.state.global_source_step, 4)
        with self.assertRaisesRegex(
            CUDARestrictedWHMultirateContractError,
            "source-motion certificate rejects ratio 4",
        ):
            advance_cuda_restricted_wh_static_multirate(
                state,
                tile,
                spec,
                _bound_plan(
                    tile,
                    state,
                    (4,),
                    spec=spec,
                    total=4,
                    allowed=(1, 2, 4),
                ),
            )

    def test_empty_population_advances_clock_without_numerical_claims(self):
        tile = _tile(node_count=5)
        state = _compatible_state(0, tile)
        spec = StaticMultirateCUDARestrictedWHSpec(4, _backend())
        result = advance_cuda_restricted_wh_static_multirate(
            state,
            tile,
            spec,
            _bound_plan(
                tile,
                state,
                (),
                spec=spec,
                total=4,
                allowed=(1, 2, 4),
            ),
        )
        self.assertEqual(result.state.global_source_step, 4)
        self.assertEqual(result.state.tracer_indices.shape, (0,))
        self.assertEqual(result.rung_lane_counts, ((1, 0), (2, 0), (4, 0)))
        self.assertFalse(any(result.claims.values()))

    def test_negative_time_split_and_block_sizes_are_bitwise_identical(self):
        tile = _tile(node_count=9, step=-0.001)
        state = _compatible_state(4, tile)
        ratios = (1, 2, 4, 4)
        whole_spec = StaticMultirateCUDARestrictedWHSpec(
            8, _backend(), threads_per_block=32
        )
        whole = advance_cuda_restricted_wh_static_multirate(
            state,
            tile,
            whole_spec,
            _bound_plan(
                tile,
                state,
                ratios,
                spec=whole_spec,
                total=8,
                allowed=(1, 2, 4),
            ),
        )
        first_tile = replace(
            tile,
            source_relative_positions_AU=np.ascontiguousarray(
                tile.source_relative_positions_AU[:5]
            ),
            source_relative_velocities_AU_per_year=np.ascontiguousarray(
                tile.source_relative_velocities_AU_per_year[:5]
            ),
            content_sha256=None,
        )
        first_spec = StaticMultirateCUDARestrictedWHSpec(
            4, _backend(), threads_per_block=128
        )
        first = advance_cuda_restricted_wh_static_multirate(
            state,
            first_tile,
            first_spec,
            _bound_plan(
                first_tile,
                state,
                ratios,
                spec=first_spec,
                total=4,
                allowed=(1, 2, 4),
            ),
        )
        second_tile = replace(
            tile,
            tile_id="synthetic-negative-split-second",
            global_source_step=4,
            source_relative_positions_AU=np.ascontiguousarray(
                tile.source_relative_positions_AU[4:]
            ),
            source_relative_velocities_AU_per_year=np.ascontiguousarray(
                tile.source_relative_velocities_AU_per_year[4:]
            ),
            content_sha256=None,
        )
        second_spec = StaticMultirateCUDARestrictedWHSpec(
            4, _backend(), threads_per_block=64
        )
        second = advance_cuda_restricted_wh_static_multirate(
            first.state,
            second_tile,
            second_spec,
            _bound_plan(
                second_tile,
                first.state,
                ratios,
                spec=second_spec,
                start=4,
                total=4,
                allowed=(1, 2, 4),
            ),
        )
        self.assertStateArraysEqual(whole.state, second.state)

    def test_lane_permutation_is_bitwise_invariant_by_tracer_id(self):
        tile = _consistent_secondary_tile(node_count=9, step=0.001)
        state = _compatible_state(8, tile)
        ratios = np.asarray((1, 2, 4, 8, 1, 2, 4, 8), dtype=np.uint32)
        spec = StaticMultirateCUDARestrictedWHSpec(8, _backend())
        baseline = advance_cuda_restricted_wh_static_multirate(
            state,
            tile,
            spec,
            _bound_plan(
                tile,
                state,
                tuple(int(item) for item in ratios),
                spec=spec,
            ),
        )
        permutation = np.asarray((7, 0, 5, 2, 6, 1, 4, 3), dtype=np.int64)
        device_permutation = CP.asarray(permutation)
        permuted_state = replace(
            state,
            tracer_relative_positions_AU=CP.ascontiguousarray(
                state.tracer_relative_positions_AU[device_permutation]
            ),
            tracer_relative_velocities_AU_per_year=CP.ascontiguousarray(
                state.tracer_relative_velocities_AU_per_year[device_permutation]
            ),
            tracer_indices=CP.ascontiguousarray(
                state.tracer_indices[device_permutation]
            ),
        )
        permuted = advance_cuda_restricted_wh_static_multirate(
            permuted_state,
            tile,
            spec,
            _bound_plan(
                tile,
                permuted_state,
                tuple(int(item) for item in ratios[permutation]),
                spec=spec,
                tracer_indices=np.asarray(permutation, dtype=np.uint64),
            ),
        )
        inverse = np.argsort(permutation)
        device_inverse = CP.asarray(inverse)
        _assert_device_array_equal(
            self,
            baseline.state.tracer_relative_positions_AU,
            permuted.state.tracer_relative_positions_AU[device_inverse],
        )
        _assert_device_array_equal(
            self,
            baseline.state.tracer_relative_velocities_AU_per_year,
            permuted.state.tracer_relative_velocities_AU_per_year[device_inverse],
        )

    def test_timestep_ladder_converges_toward_independent_dop853(self):
        total = 32
        tile = _consistent_secondary_tile(node_count=total + 1, step=0.001)
        state = _compatible_state(1, tile)
        reference_position = CP.asnumpy(
            state.tracer_relative_positions_AU[0]
        )
        reference_velocity = CP.asnumpy(
            state.tracer_relative_velocities_AU_per_year[0]
        )
        route_spec = RestrictedWHRouteSpec(
            maximum_primary_dynamical_phase=0.25,
            primary_outer_boundary_AU=100.0,
        )
        # Integrate each retained fine source interval independently so the
        # DOP853 reference sees exactly the same piecewise-Hermite forcing as
        # every multirate rung.  A single 32-node endpoint interpolant would be
        # a different physical forcing and is not an accuracy oracle.
        for source_node_index in range(total):
            route = route_restricted_wh_map_step(
                reference_position,
                reference_velocity,
                tracer_index=0,
                source_tile=tile,
                source_node_index=source_node_index,
                source_steps_per_map_step=1,
                spec=route_spec,
            )
            self.assertFalse(route.events)
            reference_position = route.endpoint_relative_position_AU
            reference_velocity = route.endpoint_relative_velocity_AU_per_year
        reference = np.concatenate((reference_position, reference_velocity))
        errors: list[float] = []
        for ratio in (1, 2, 4, 8):
            spec = StaticMultirateCUDARestrictedWHSpec(
                total,
                _backend(),
                maximum_source_dynamical_phase=0.25,
                primary_outer_boundary_AU=100.0,
                maximum_relative_sweep_fraction=0.15,
            )
            plan = _bound_plan(
                tile,
                state,
                (ratio,),
                spec=spec,
                total=total,
                allowed=(1, 2, 4, 8),
            )
            result = advance_cuda_restricted_wh_static_multirate(
                state,
                tile,
                spec,
                plan,
            )
            observed = np.concatenate(
                (
                    CP.asnumpy(result.state.tracer_relative_positions_AU[0]),
                    CP.asnumpy(
                        result.state.tracer_relative_velocities_AU_per_year[0]
                    ),
                )
            )
            errors.append(float(np.max(np.abs(observed - reference))))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])
        self.assertLess(errors[2], errors[3])
        # A second-order KDK ladder should improve by roughly four per halving.
        for fine, coarse in zip(errors[:-1], errors[1:], strict=True):
            self.assertGreater(coarse / fine, 2.5)


if __name__ == "__main__":
    unittest.main()
