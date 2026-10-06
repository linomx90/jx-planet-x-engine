from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import numpy as np
import jxplanetx.long_horizon as long_horizon
import jxplanetx.restricted_wh_source as restricted_wh_source

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHResult,
    CUDARestrictedWHState,
    FrozenSourcePathTile,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    CUDARestrictedWHHybridResult,
    CUDARestrictedWHRouteRecord,
)
from jxplanetx.engine import Provenance
from jxplanetx.long_horizon import (
    CANONICAL_FAMILY,
    CLAIM_CEILING,
    PUBLIC_STABILITY,
    RESTRICTED_WH_ARCHITECTURE,
    LongHorizonContractError,
    adapt_restricted_wh_checkpoint,
    adapt_restricted_wh_event,
    adapt_restricted_wh_result,
)
from jxplanetx.restricted_wh_episode_checkpoint import (
    LoadedOwnedEpisodeCheckpoint,
)
from jxplanetx.restricted_wh_lifecycle import (
    ACTIVE,
    REMOVED_SOURCE_RADIUS,
    RestrictedWHLifecycleResult,
    RestrictedWHLifecycleState,
    RestrictedWHRemovalRecord,
)
from jxplanetx.restricted_wh_route import (
    RestrictedWHRouteEvent,
    RestrictedWHRouteResult,
)
from jxplanetx.restricted_wh_source import (
    WisdomHolmanRestrictedWHHybridSegmentResult,
    WisdomHolmanSourceTileBuild,
)


SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
SHA_C = hashlib.sha256(b"c").hexdigest()
SHA_D = hashlib.sha256(b"d").hexdigest()
SHA_E = hashlib.sha256(b"e").hexdigest()


def _state(*, count: int = 2, step: int = 8) -> CUDARestrictedWHState:
    return CUDARestrictedWHState(
        snapshot_id="cpu-contract-fixture",
        epoch_origin_years=2_451_545.0,
        global_source_step=step,
        source_step_years=0.125,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SHA_A,
        tracer_relative_positions_AU=np.zeros((count, 3), dtype=np.float64),
        tracer_relative_velocities_AU_per_year=np.zeros(
            (count, 3), dtype=np.float64
        ),
        tracer_indices=np.arange(count, dtype=np.uint64),
        provenance=Provenance(
            source_id="jx-test.long-horizon",
            citation="Synthetic façade contract fixture",
            version="v1",
            sha256=SHA_B,
        ),
    )


def _event() -> RestrictedWHRouteEvent:
    return RestrictedWHRouteEvent(
        event_type="SOURCE_RADIUS_CONTACT",
        source_index=0,
        source_id="Sun",
        map_step_fraction=0.25,
        source_interval_offset=0,
        interval_fraction=0.5,
        relative_position_AU=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        relative_velocity_AU_per_year=np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        surface_distance_AU=1.0,
        surface_threshold_AU=1.0,
        surface_residual_AU=0.0,
        surface_tolerance_AU=1.0e-12,
        primary_specific_energy_AU2_per_year2=-1.0,
        primary_radial_velocity_AU_per_year=-1.0,
        surface_radial_velocity_AU_per_year=-1.0,
        location_method="DOP853_DENSE_BRENTQ",
    )


def _source_build() -> WisdomHolmanSourceTileBuild:
    tile = FrozenSourcePathTile(
        tile_id="cpu-contract-source",
        epoch_origin_years=2_451_545.0,
        global_source_step=7,
        source_step_years=0.125,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SHA_A,
        source_relative_positions_AU=np.zeros(
            (2, 1, 3), dtype=np.float64
        ),
        source_relative_velocities_AU_per_year=np.zeros(
            (2, 1, 3), dtype=np.float64
        ),
        source_gm_AU3_per_year2=np.asarray((1.0,), dtype=np.float64),
        source_encounter_radii_AU=np.asarray((0.1,), dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
            (1.0,), dtype=np.float64
        ),
        provenance=Provenance(
            source_id="jx-test.long-horizon-source",
            citation="Synthetic source-wrapper fixture",
            version="v1",
            sha256=SHA_B,
        ),
    )
    positions = np.ascontiguousarray(tile.source_relative_positions_AU[0])
    velocities = np.ascontiguousarray(
        tile.source_relative_velocities_AU_per_year[0]
    )
    gm = np.ascontiguousarray(tile.source_gm_AU3_per_year2)
    input_digest = restricted_wh_source._canonical_input_sha256(
        tile.source_ids, positions, velocities, gm
    )
    barycentric_digest = restricted_wh_source._coordinate_state_sha256(
        b"jx.restricted-wh.source-provider.canonical-barycentric-input.v1",
        tile.source_ids,
        positions,
        velocities,
        gm,
    )
    metrics = {
        "maximum_component_input_ulps": 0.0,
        "maximum_forward_error_bound_fraction": 0.0,
        "maximum_input_relative_error": 0.0,
    }
    receipt = restricted_wh_source._make_coordinate_canonicalization_receipt(
        tile=tile,
        raw_relative_input_sha256=input_digest,
        canonical_relative_input_sha256=input_digest,
        canonical_barycentric_input_sha256=barycentric_digest,
        positions=positions,
        velocities=velocities,
        canonical_positions=positions,
        canonical_velocities=velocities,
        position_metrics=metrics,
        velocity_metrics=metrics,
        fixed_point_iteration_count=1,
    )
    return WisdomHolmanSourceTileBuild(
        tile=tile,
        input_content_sha256=input_digest,
        source_map_input_content_sha256=SHA_B,
        source_map_result_content_sha256=SHA_C,
        source_map_verification_content_sha256=SHA_D,
        source_map_metrics=(("fixture", 1.0),),
        source_map_primary_seconds=0.1,
        source_map_replay_seconds=0.1,
        coordinate_canonicalization=receipt,
        runtime_identity={
            "backend": "cpu-fixture",
            "provider_id": restricted_wh_source.PROVIDER_ID,
            "provider_implementation_source_sha256": (
                restricted_wh_source.IMPLEMENTATION_SOURCE_SHA256
            ),
            "exact_replay_required": True,
            "source_tile_sha256": tile.content_sha256,
            "coordinate_canonicalization": dict(receipt.as_record()),
            "coordinate_canonicalization_receipt_sha256": (
                receipt.content_sha256
            ),
        },
    )


class LongHorizonFacadeTests(unittest.TestCase):
    def test_experimental_facade_exposes_physics_genesis_and_fixed_runner(self):
        expected = {
            "DEFAULT_GENESIS_MODEL",
            "FixedLongHorizonRunConfig",
            "FixedLongHorizonSourceSegment",
            "KNOWN_SOLAR_SYSTEM_V2",
            "KnownSolarSystemSourceModel",
            "ObservedInitialConditionInventory",
            "advance_fixed_lifecycle_segment",
            "build_predeclared_initial_condition_inventory",
            "load_known_solar_system_source_model",
            "resume_fixed_lifecycle_run",
            "start_fixed_lifecycle_run",
        }
        self.assertTrue(expected.issubset(set(long_horizon.__all__)))
        for name in expected:
            self.assertIsNotNone(getattr(long_horizon, name))
        self.assertNotIn(
            "advance_cuda_restricted_wh_static_multirate",
            long_horizon.__all__,
        )

    def test_restricted_wh_is_canonical_but_remains_experimental(self):
        declaration = RESTRICTED_WH_ARCHITECTURE
        self.assertTrue(declaration.canonical)
        self.assertEqual(declaration.family, CANONICAL_FAMILY)
        self.assertEqual(declaration.public_stability, PUBLIC_STABILITY)
        self.assertEqual(declaration.project_scope, "SCREENING_ONLY")
        self.assertEqual(declaration.scientific_classification, "MODEL_OUTPUT")
        self.assertEqual(dict(declaration.claims), dict(CLAIM_CEILING))
        self.assertFalse(any(declaration.claims.values()))
        self.assertEqual(
            dict(declaration.entry_points),
            {
                "known_source_model": (
                    "jxplanetx.long_horizon:"
                    "load_known_solar_system_source_model"
                ),
                "initial_conditions": (
                    "jxplanetx.long_horizon:"
                    "build_predeclared_initial_condition_inventory"
                ),
                "start": "jxplanetx.long_horizon:start_fixed_lifecycle_run",
                "resume": "jxplanetx.long_horizon:resume_fixed_lifecycle_run",
                "advance_segment": (
                    "jxplanetx.long_horizon:advance_fixed_lifecycle_segment"
                ),
            },
        )
        self.assertTrue(
            all(
                value.startswith("jxplanetx.long_horizon:")
                for value in declaration.entry_points.values()
            )
        )
        with self.assertRaises(TypeError):
            declaration.entry_points["start"] = "forged"
        with self.assertRaises(TypeError):
            declaration.claims["production_qualified"] = True

    def test_event_adapter_requires_explicit_context_and_is_content_bound(self):
        receipt = adapt_restricted_wh_event(
            _event(),
            tracer_index=42,
            lane_index=1,
            route_start_global_source_step=8,
            source_steps_per_map_step=2,
            frame="ICRF",
        )
        self.assertEqual(receipt.tracer_index, 42)
        self.assertEqual(receipt.global_step, 8)
        self.assertEqual(receipt.step_fraction, 0.5)
        self.assertEqual(receipt.diagnostics["map_step_fraction"], 0.25)
        self.assertEqual(receipt.relative_position, (1.0, 0.0, 0.0))
        self.assertEqual(receipt.diagnostics["surface_threshold"], 1.0)
        self.assertEqual(len(receipt.content_sha256), 64)
        self.assertFalse(receipt.claims["physical_collision_inferred"])
        with self.assertRaises(TypeError):
            receipt.diagnostics["surface_threshold"] = 2.0
        with self.assertRaises(LongHorizonContractError):
            replace(receipt, public_stability="STABLE", content_sha256=None)
        with self.assertRaises(LongHorizonContractError):
            replace(receipt, diagnostics={"bad": float("nan")}, content_sha256=None)

        endpoint_event = replace(
            _event(),
            map_step_fraction=1.0,
            source_interval_offset=1,
            interval_fraction=1.0,
        )
        endpoint = adapt_restricted_wh_event(
            endpoint_event,
            tracer_index=42,
            route_start_global_source_step=8,
            source_steps_per_map_step=2,
            frame="ICRF",
        )
        self.assertEqual((endpoint.global_step, endpoint.step_fraction), (10, 0.0))

    def test_plain_map_result_normalizes_without_gpu_access(self):
        state = _state()
        result = CUDARestrictedWHResult(
            state=state,
            source_tile_sha256=SHA_C,
            minimum_endpoint_source_distance_AU=np.asarray(
                (0.75, 1.25), dtype=np.float64
            ),
            maximum_kepler_iterations=4,
            runtime_identity={"backend": "cpu-fixture", "workers": 0},
        )
        receipt = adapt_restricted_wh_result(result)
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.global_step, 8)
        self.assertEqual(receipt.tracer_count, 2)
        self.assertEqual(receipt.active_tracer_count, 2)
        self.assertEqual(receipt.route_count, 0)
        self.assertEqual(receipt.events, ())
        self.assertEqual(receipt.diagnostics["maximum_kepler_iterations"], 4)
        self.assertEqual(
            receipt.diagnostics["global_minimum_source_distance_AU"], 0.75
        )
        self.assertEqual(receipt.diagnostics["minimum_source_distance_lane_count"], 2)
        self.assertRegex(
            receipt.diagnostics["minimum_source_distance_vector_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(receipt.source_tile_sha256, SHA_C)
        self.assertRegex(receipt.output_state_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(receipt.route_ledger_sha256, r"^[0-9a-f]{64}$")
        self.assertFalse(any(receipt.claims.values()))

        changed_positions = np.zeros((2, 3), dtype=np.float64)
        changed_positions[0, 0] = 1.0e-12
        changed_state = replace(
            state,
            tracer_relative_positions_AU=changed_positions,
        )
        changed_receipt = adapt_restricted_wh_result(
            replace(result, state=changed_state)
        )
        self.assertNotEqual(
            receipt.output_state_sha256,
            changed_receipt.output_state_sha256,
        )
        self.assertNotEqual(
            receipt.content_sha256,
            changed_receipt.content_sha256,
        )
        invalid_states = (
            replace(
                state,
                tracer_relative_positions_AU=np.zeros(
                    (1, 4), dtype=np.float64
                ),
            ),
            replace(
                state,
                tracer_relative_positions_AU=np.asarray(
                    ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)),
                    dtype=np.float64,
                ),
            ),
            replace(
                state,
                tracer_indices=np.asarray((7, 7), dtype=np.uint64),
            ),
        )
        for invalid_state in invalid_states:
            with self.subTest(shape=invalid_state.tracer_relative_positions_AU.shape):
                with self.assertRaises(LongHorizonContractError):
                    adapt_restricted_wh_result(
                        replace(result, state=invalid_state)
                    )

        after_endpoint = adapt_restricted_wh_event(
            _event(),
            tracer_index=42,
            route_start_global_source_step=receipt.global_step,
            source_steps_per_map_step=2,
            frame="ICRF",
        )
        with self.assertRaisesRegex(LongHorizonContractError, "beyond"):
            replace(
                receipt,
                events=(after_endpoint,),
                content_sha256=None,
            )

    def test_route_ledger_identity_binds_complete_route_semantics(self):
        state = _state()
        route = RestrictedWHRouteResult(
            tracer_index=0,
            source_tile_sha256=SHA_C,
            source_segment_sha256=SHA_D,
            global_source_step=7,
            source_steps_per_map_step=1,
            events=(),
            endpoint_relative_position_AU=np.zeros(3, dtype=np.float64),
            endpoint_relative_velocity_AU_per_year=np.zeros(
                3, dtype=np.float64
            ),
            accepted_step_count=3,
            derivative_evaluation_count=20,
            minimum_evaluated_stage_distance_AU=1.0,
            minimum_committed_stage_distance_AU=1.0,
            maximum_accepted_primary_dynamical_phase_radians=0.01,
            maximum_committed_primary_dynamical_phase_radians=0.01,
            completed_map_fraction=1.0,
            runtime_identity={"solver": "cpu-fixture"},
        )
        record = CUDARestrictedWHRouteRecord(
            map_step_index=0,
            lane_index=0,
            tracer_index=0,
            guard_phase="PRE_KICK_GUARD",
            guard_reason="SOURCE_RADIUS_GUARD",
            guard_source_index=0,
            route_result=route,
        )
        source_build = _source_build()
        wrapped_route = replace(
            route,
            source_tile_sha256=source_build.tile.content_sha256,
        )
        wrapped_record = replace(record, route_result=wrapped_route)
        result = CUDARestrictedWHHybridResult(
            state=state,
            source_tile_sha256=source_build.tile.content_sha256,
            minimum_source_distance_AU=np.asarray(
                (1.0, 2.0), dtype=np.float64
            ),
            maximum_kepler_iterations=5,
            route_records=(wrapped_record,),
            runtime_identity={"backend": "cpu-fixture"},
        )
        wrapped = WisdomHolmanRestrictedWHHybridSegmentResult(
            source_build=source_build,
            tracer_result=result,
        )
        receipt = adapt_restricted_wh_result(wrapped)
        changed_route = replace(wrapped_route, accepted_step_count=4)
        changed_result = replace(
            result,
            route_records=(
                replace(wrapped_record, route_result=changed_route),
            ),
        )
        changed_receipt = adapt_restricted_wh_result(
            replace(wrapped, tracer_result=changed_result)
        )
        self.assertEqual(receipt.route_count, 1)
        self.assertNotEqual(
            receipt.route_ledger_sha256,
            changed_receipt.route_ledger_sha256,
        )
        self.assertNotEqual(
            receipt.content_sha256,
            changed_receipt.content_sha256,
        )
        with self.assertRaisesRegex(LongHorizonContractError, "source wrapper"):
            adapt_restricted_wh_result(result)
        with self.assertRaisesRegex(LongHorizonContractError, "tile lattice"):
            adapt_restricted_wh_result(
                replace(
                    wrapped,
                    tracer_result=replace(
                        result,
                        route_records=(
                            replace(wrapped_record, map_step_index=999),
                        ),
                    ),
                )
            )

        coherent_but_wrong_start = replace(
            result.route_records[0],
            map_step_index=6,
        )
        forged_wrapped = replace(
            wrapped,
            tracer_result=replace(
                result,
                route_records=(coherent_but_wrong_start,),
            ),
        )
        with self.assertRaisesRegex(LongHorizonContractError, "tile lattice"):
            adapt_restricted_wh_result(forged_wrapped)

    def test_lifecycle_result_normalizes_terminal_event_without_inference(self):
        event = _event()
        positions = np.zeros((2, 3), dtype=np.float64)
        velocities = np.zeros((2, 3), dtype=np.float64)
        positions[0] = event.relative_position_AU
        velocities[0] = event.relative_velocity_AU_per_year
        state = replace(
            _state(step=10),
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        removal = RestrictedWHRemovalRecord(
            lane_index=0,
            tracer_index=0,
            lifecycle_code=REMOVED_SOURCE_RADIUS,
            global_source_step=8,
            source_steps_per_map_step=2,
            guard_phase="PRE_DRIFT",
            guard_reason="SOURCE_RADIUS_GUARD",
            guard_source_index=0,
            source_segment_sha256=SHA_D,
            route_runtime_identity_sha256=SHA_E,
            accepted_step_count=3,
            derivative_evaluation_count=20,
            minimum_committed_stage_distance_AU=1.0,
            maximum_committed_primary_dynamical_phase_radians=0.01,
            events=(event,),
        )
        lifecycle = RestrictedWHLifecycleState(
            dynamics_state=state,
            tracer_lifecycle=np.asarray(
                (REMOVED_SOURCE_RADIUS, ACTIVE), dtype=np.uint8
            ),
            removal_records=(removal,),
        )
        result = RestrictedWHLifecycleResult(
            state=lifecycle,
            source_tile_sha256=SHA_C,
            minimum_source_distance_AU=np.asarray(
                (1.0, float("inf")), dtype=np.float64
            ),
            maximum_kepler_iterations=5,
            route_records=(),
            new_removal_records=(removal,),
            runtime_identity={"backend": "cpu-fixture"},
        )
        receipt = adapt_restricted_wh_result(result)
        self.assertEqual(receipt.active_tracer_count, 1)
        self.assertEqual(
            receipt.diagnostics["minimum_source_distance_finite_lane_count"], 1
        )
        self.assertEqual(len(receipt.events), 1)
        self.assertTrue(receipt.events[0].terminal)
        self.assertEqual(receipt.events[0].lane_index, 0)
        self.assertFalse(receipt.events[0].claims["physical_collision_inferred"])
        self.assertEqual(
            receipt.diagnostics["event_ledger_scope"],
            "CUMULATIVE_COMMITTED_LIFECYCLE_HISTORY",
        )

        omitted_delta_receipt = adapt_restricted_wh_result(
            replace(result, new_removal_records=())
        )
        self.assertEqual(
            tuple(item.content_sha256 for item in omitted_delta_receipt.events),
            tuple(item.content_sha256 for item in receipt.events),
        )
        self.assertEqual(
            omitted_delta_receipt.output_state_sha256,
            receipt.output_state_sha256,
        )
        self.assertEqual(
            omitted_delta_receipt.content_sha256,
            receipt.content_sha256,
        )

        forged_active_state = RestrictedWHLifecycleState(
            dynamics_state=state,
            tracer_lifecycle=np.asarray((ACTIVE, ACTIVE), dtype=np.uint8),
            removal_records=(),
        )
        with self.assertRaisesRegex(LongHorizonContractError, "history"):
            adapt_restricted_wh_result(
                replace(result, state=forged_active_state)
            )

    def test_unverified_owned_episode_checkpoint_cannot_be_constructed(self):
        with self.assertRaisesRegex(Exception, "replay-verifying loader"):
            LoadedOwnedEpisodeCheckpoint(
                state=_state(step=16),
                checkpoint_index=3,
                checkpoint_content_sha256=SHA_A,
                input_state_sha256=SHA_B,
                state_sha256=SHA_C,
                route_chain_sha256=SHA_D,
                route_record_count_cumulative=7,
                segment_episode_count_cumulative=2,
                run_id="cpu-contract-run",
                protocol_sha256=SHA_E,
                record={"schema": "fixture"},
            )

    def test_foreign_or_partial_objects_fail_closed(self):
        class Foreign:
            pass

        with self.assertRaises(LongHorizonContractError):
            adapt_restricted_wh_result(Foreign())
        with self.assertRaises(LongHorizonContractError):
            adapt_restricted_wh_checkpoint(Foreign())
        with self.assertRaises(LongHorizonContractError):
            adapt_restricted_wh_event(
                Foreign(),
                tracer_index=0,
                route_start_global_source_step=0,
                source_steps_per_map_step=1,
                frame="ICRF",
            )


if __name__ == "__main__":
    unittest.main()
