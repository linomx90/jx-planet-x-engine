from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import unittest

import numpy as np

import jxplanetx.long_horizon as facade
import jxplanetx.long_horizon.known_system_inputs as known_inputs
from jxplanetx.engine import BackendSpec
from jxplanetx.long_horizon.initial_conditions import (
    DE440_SUN_GENESIS_MODEL,
)
from jxplanetx.long_horizon.known_system_inputs import (
    ALGORITHMIC_ROUTE_RADII_AU,
    ALGORITHMIC_ROUTE_RADIUS_AU,
    CLAIMS,
    ENGINE_EPOCH_ORIGIN_YEARS,
    HORIZON_YEARS,
    KnownSystemInputContractError,
    KnownSystemRunProfile,
    M2_FINE_CANDIDATE_SEGMENT_MAP_STEP_COUNT,
    M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP,
    M2_FINE_CANDIDATE_STEP_YEARS,
    M2_FINE_HORIZON_YEARS,
    M2_FINE_REFINEMENT_SEGMENT_MAP_STEP_COUNT,
    M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP,
    M2_FINE_REFINEMENT_STEP_YEARS,
    M2_FINE_SEGMENT_SOURCE_STEP_COUNT,
    PHYSICAL_TERMINAL_RADII_AU,
    PROFILE_ID,
    ROUTE_WORKER_COUNT,
    RUN_ID,
    SOURCE_STEP_COUNT,
    SOURCE_STEP_YEARS,
    TERMINAL_RADIUS_POLICY,
    TRACER_MAP_STEP_COUNT,
    TRACER_SOURCE_STEPS_PER_MAP_STEP,
    TRACER_STEP_YEARS,
    build_known_system_observed_fixed_run_inputs,
    prepare_known_system_observed_fixed_run,
)
from jxplanetx.long_horizon.physical_models import (
    KNOWN_SOLAR_SYSTEM_V2,
)
from jxplanetx.long_horizon.m2_source_path import (
    FINE_SOURCE_SCHEDULE_SHA256,
    FINE_SOURCE_STEP_COUNT,
    FINE_SOURCE_STEP_YEARS,
)
from jxplanetx.long_horizon.runner import advance_fixed_lifecycle_segment


EXPECTED_CONTRACT_SHA256 = (
    "b7766b414b0462d2a8b2074d962d0b05bf30c773e5f6a885422fc43aeb69cab2"
)
EXPECTED_BOOTSTRAP_PROFILE_SHA256 = (
    "ff2ba6c0736e11a279c2a2327d34b9a6fead6749e676cee11d492fa0b5843f70"
)
EXPECTED_FINE_CANDIDATE_CONTRACT_SHA256 = (
    "ef8f2b61cfdf2b7228971df4a8217949de44f1737341ddc292bba4e363f5aaa0"
)
EXPECTED_FINE_CANDIDATE_PROFILE_SHA256 = (
    "045db073e83919aa9981875db5c5200f621b44b01a808ef974db5e3c21d7dbc7"
)
EXPECTED_FINE_REFINEMENT_CONTRACT_SHA256 = (
    "33e33024629f2b0180c0f5439b9883b479f94800b6b8a4305ce4ad4d8fb47d73"
)
EXPECTED_FINE_REFINEMENT_PROFILE_SHA256 = (
    "f9f71b5429755f62e8488595ca34cbde7b006647fe050eafef0868d94a82efa4"
)
EXPECTED_ADAPTER_SHA256 = (
    "e45b929cc3f0500ad04ef037eda0381747a065214fb96cb2b0e3321f319c316a"
)
EXPECTED_ROSTER_SHA256 = (
    "998352ffa6f76c40c0468192c378dc43a1d27e01734736bf97bd7544e0879d90"
)
EXPECTED_SOURCE_SCHEDULE_SHA256 = (
    "bbd4b218aca8a6746f9f11d313e1681bcc664bafb1a8a0368f3f78bdf26da8f1"
)
EXPECTED_SOURCE_SEGMENT_SHA256 = (
    "a7f07e65e5227d06da910f7f7ef75af0a43de78fc33722d9e43805cba2b35e1b"
)
EXPECTED_RUNNER_PROTOCOL_SHA256 = (
    "1c72571ffd95ab929264f8bb1b7defcf4f52a64586df7bb26b8fc84b3461b7f2"
)
EXPECTED_POSITION_ARRAY_SHA256 = (
    "8aeba1c8e6c0fd2f188a50479b5d35688b3f170c0e2b47dcce8ea51124e6481a"
)
EXPECTED_VELOCITY_ARRAY_SHA256 = (
    "22c6cee1a8326244e8067987b08724af678a837b6ca5ca12f53ff339e4608936"
)


def _cupy_or_none():
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
        with cp.cuda.Device(0):
            cp.zeros(1, dtype=cp.float64)
        return cp
    except Exception:  # pragma: no cover - CPU-only CI
        return None


CP = _cupy_or_none()


class LongHorizonKnownSystemInputContractTests(unittest.TestCase):
    def test_public_api_is_the_single_experimental_facade(self) -> None:
        for name in (
            "KnownSystemRunProfile",
            "KnownSystemObservedFixedRunContract",
            "KnownSystemObservedFixedRunInputs",
            "prepare_known_system_observed_fixed_run",
            "build_known_system_observed_fixed_run_inputs",
        ):
            self.assertIn(name, facade.__all__)
            self.assertIs(getattr(facade, name), getattr(known_inputs, name))

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = prepare_known_system_observed_fixed_run()

    def test_exact_models_inventory_roster_and_hashes_are_bound(self) -> None:
        contract = self.contract
        self.assertIs(contract.known_system_model, KNOWN_SOLAR_SYSTEM_V2)
        self.assertEqual(contract.inventory.model, DE440_SUN_GENESIS_MODEL)
        self.assertEqual(contract.inventory.state_count, 108)
        self.assertEqual(contract.roster.tracer_indices, tuple(range(108)))
        self.assertEqual(
            contract.roster.object_ids,
            tuple(state.object_id for state in contract.inventory.states),
        )
        self.assertEqual(
            tuple(
                contract.roster.object_id_for_tracer_index(index)
                for index in range(108)
            ),
            contract.roster.object_ids,
        )
        self.assertEqual(contract.content_sha256, EXPECTED_CONTRACT_SHA256)
        self.assertEqual(
            contract.profile.content_sha256,
            EXPECTED_BOOTSTRAP_PROFILE_SHA256,
        )
        self.assertEqual(
            contract.adapter.content_sha256, EXPECTED_ADAPTER_SHA256
        )
        self.assertEqual(
            contract.roster.content_sha256, EXPECTED_ROSTER_SHA256
        )
        self.assertEqual(
            contract.source_schedule_sha256,
            EXPECTED_SOURCE_SCHEDULE_SHA256,
        )
        self.assertEqual(
            contract.source.content_sha256, EXPECTED_SOURCE_SEGMENT_SHA256
        )
        self.assertEqual(
            contract.config.protocol_sha256,
            EXPECTED_RUNNER_PROTOCOL_SHA256,
        )
        record = contract.as_record()
        self.assertEqual(
            record["tracer_positions"]["sha256"],
            EXPECTED_POSITION_ARRAY_SHA256,
        )
        self.assertEqual(
            record["tracer_velocities"]["sha256"],
            EXPECTED_VELOCITY_ARRAY_SHA256,
        )
        self.assertEqual(record["claims"], dict(CLAIMS))
        self.assertFalse(any(record["claims"].values()))

    def test_closed_retained_m2_fine_profiles_bind_official_ratios(self) -> None:
        candidate = prepare_known_system_observed_fixed_run(
            profile_id=(
                KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL
            )
        )
        refinement = prepare_known_system_observed_fixed_run(
            profile_id=(
                KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL
            )
        )
        self.assertEqual(
            candidate.content_sha256,
            EXPECTED_FINE_CANDIDATE_CONTRACT_SHA256,
        )
        self.assertEqual(
            candidate.profile.content_sha256,
            EXPECTED_FINE_CANDIDATE_PROFILE_SHA256,
        )
        self.assertEqual(
            refinement.content_sha256,
            EXPECTED_FINE_REFINEMENT_CONTRACT_SHA256,
        )
        self.assertEqual(
            refinement.profile.content_sha256,
            EXPECTED_FINE_REFINEMENT_PROFILE_SHA256,
        )
        for contract in (candidate, refinement):
            self.assertIsNone(contract.source)
            self.assertEqual(
                contract.profile.source_mode,
                "RETAINED_M2_FINE_SOURCE_PATH",
            )
            self.assertEqual(
                contract.source_schedule_sha256,
                FINE_SOURCE_SCHEDULE_SHA256,
            )
            self.assertEqual(
                contract.profile.source_step_years.hex(),
                FINE_SOURCE_STEP_YEARS.hex(),
            )
            self.assertEqual(
                contract.profile.total_source_step_count,
                FINE_SOURCE_STEP_COUNT,
            )
            self.assertEqual(
                contract.profile.total_horizon_years,
                M2_FINE_HORIZON_YEARS,
            )
            self.assertEqual(
                contract.config.source_spec.source_step_count,
                M2_FINE_SEGMENT_SOURCE_STEP_COUNT,
            )
            self.assertEqual(contract.config.route_worker_count, 6)
            self.assertIs(contract.config.multirate_enabled, False)
            self.assertIs(contract.config.owned_episode_enabled, False)
            self.assertEqual(
                contract.config.route_spec.terminal_source_radii_AU,
                PHYSICAL_TERMINAL_RADII_AU,
            )
            self.assertEqual(
                contract.config.tracer_spec.primary_outer_boundary_AU,
                10_000.0,
            )
        self.assertEqual(
            candidate.profile.source_steps_per_map_step,
            M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP,
        )
        self.assertEqual(
            candidate.profile.segment_map_step_count,
            M2_FINE_CANDIDATE_SEGMENT_MAP_STEP_COUNT,
        )
        self.assertEqual(
            candidate.profile.tracer_step_years.hex(),
            M2_FINE_CANDIDATE_STEP_YEARS.hex(),
        )
        self.assertEqual(
            refinement.profile.source_steps_per_map_step,
            M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP,
        )
        self.assertEqual(
            refinement.profile.segment_map_step_count,
            M2_FINE_REFINEMENT_SEGMENT_MAP_STEP_COUNT,
        )
        self.assertEqual(
            refinement.profile.tracer_step_years.hex(),
            M2_FINE_REFINEMENT_STEP_YEARS.hex(),
        )
        self.assertEqual(
            candidate.roster.content_sha256,
            refinement.roster.content_sha256,
        )
        self.assertTrue(
            np.array_equal(
                candidate.tracer_positions_AU.view(np.uint64),
                refinement.tracer_positions_AU.view(np.uint64),
            )
        )
        self.assertTrue(
            np.array_equal(
                candidate.tracer_velocities_AU_per_year.view(np.uint64),
                refinement.tracer_velocities_AU_per_year.view(np.uint64),
            )
        )

    def test_adapter_is_an_exact_no_numerical_change_receipt(self) -> None:
        contract = self.contract
        adapter = contract.adapter
        self.assertEqual(adapter.epoch_jd_tdb, 2_461_200.5)
        self.assertEqual(adapter.epoch_scale, "TDB")
        self.assertEqual(ENGINE_EPOCH_ORIGIN_YEARS.hex(), "0x0.0p+0")
        self.assertEqual(adapter.engine_epoch_origin_years.hex(), "0x0.0p+0")
        self.assertEqual(
            adapter.physical_frame,
            "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000",
        )
        self.assertEqual(adapter.engine_axes, "IAU76_80_ECLIPTIC_J2000")
        self.assertEqual(
            adapter.physical_coordinate_system,
            "PRIMARY_RELATIVE_CARTESIAN_RIGHT_HANDED",
        )
        self.assertEqual(
            adapter.engine_coordinate_system,
            "PRIMARY_RELATIVE_NONROTATING",
        )
        self.assertEqual(adapter.physical_time_unit, "JULIAN_YEAR_365.25_D")
        self.assertEqual(adapter.engine_time_unit, "year")
        self.assertEqual(
            adapter.as_record()["numerical_position_transform"],
            "NONE_BITWISE_COPY",
        )
        self.assertEqual(
            adapter.as_record()["numerical_velocity_transform"],
            "NONE_BITWISE_COPY",
        )
        expected_positions = np.asarray(
            [state.position_AU for state in contract.inventory.states],
            dtype=np.float64,
            order="C",
        )
        expected_velocities = np.asarray(
            [state.velocity_AU_per_year for state in contract.inventory.states],
            dtype=np.float64,
            order="C",
        )
        self.assertTrue(
            np.array_equal(
                contract.tracer_positions_AU.view(np.uint64),
                expected_positions.view(np.uint64),
            )
        )
        self.assertTrue(
            np.array_equal(
                contract.tracer_velocities_AU_per_year.view(np.uint64),
                expected_velocities.view(np.uint64),
            )
        )
        self.assertFalse(contract.tracer_positions_AU.flags.writeable)
        self.assertFalse(contract.tracer_velocities_AU_per_year.flags.writeable)
        with self.assertRaises(ValueError):
            contract.tracer_positions_AU.setflags(write=True)
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "engine_epoch_origin_years changed"
        ):
            replace(adapter, engine_epoch_origin_years=-0.0)

    def test_algorithmic_routing_and_physical_terminal_surfaces_are_separate(
        self,
    ) -> None:
        contract = self.contract
        self.assertEqual(
            tuple(contract.source.source_encounter_radii_AU),
            ALGORITHMIC_ROUTE_RADII_AU,
        )
        self.assertEqual(
            ALGORITHMIC_ROUTE_RADIUS_AU.hex(),
            "0x1.999999999999ap-4",
        )
        self.assertEqual(
            contract.config.route_spec.terminal_source_radii_AU,
            PHYSICAL_TERMINAL_RADII_AU,
        )
        self.assertEqual(PHYSICAL_TERMINAL_RADII_AU, (0.0,) * 10)
        self.assertIn("NO_PROVENANCE_BOUND", TERMINAL_RADIUS_POLICY)
        self.assertFalse(CLAIMS["physical_terminal_radii_modeled"])
        self.assertFalse(CLAIMS["physical_collision_inferred"])

        gm = contract.source.source_gm_AU3_per_year2
        exact = (
            2
            * sum(
                (Fraction.from_float(float(value)) for value in gm),
                Fraction(0, 1),
            )
            / Fraction.from_float(ALGORITHMIC_ROUTE_RADIUS_AU) ** 2
        )
        bound = (
            contract.source
            .required_uniform_relative_acceleration_bound_AU_per_year2
        )
        self.assertEqual(bound.hex(), "0x1.ee1fac800e17ap+12")
        self.assertGreaterEqual(Fraction.from_float(bound), exact)
        self.assertLess(Fraction.from_float(math.nextafter(bound, -math.inf)), exact)
        self.assertTrue(
            np.all(
                contract.source
                .certified_relative_acceleration_bounds_AU_per_year2
                == bound
            )
        )

    def test_fixed_one_year_six_worker_profile_is_exact(self) -> None:
        contract = self.contract
        config = contract.config
        self.assertEqual(PROFILE_ID, "m2-known-system-observed-fixed-1y-v1")
        self.assertEqual(RUN_ID, config.run_id)
        self.assertEqual(HORIZON_YEARS, 1)
        self.assertEqual(SOURCE_STEP_YEARS.hex(), "0x1.0000000000000p-12")
        self.assertEqual(SOURCE_STEP_COUNT, 4096)
        self.assertEqual(TRACER_STEP_YEARS.hex(), "0x1.0000000000000p-8")
        self.assertEqual(TRACER_MAP_STEP_COUNT, 256)
        self.assertEqual(TRACER_SOURCE_STEPS_PER_MAP_STEP, 16)
        self.assertEqual(
            SOURCE_STEP_YEARS * SOURCE_STEP_COUNT, float(HORIZON_YEARS)
        )
        self.assertEqual(
            TRACER_STEP_YEARS * TRACER_MAP_STEP_COUNT,
            float(HORIZON_YEARS),
        )
        self.assertEqual(config.source_spec.source_step_count, 4096)
        self.assertEqual(config.tracer_spec.map_step_count, 256)
        self.assertEqual(config.tracer_spec.source_steps_per_map_step, 16)
        self.assertEqual(config.route_worker_count, ROUTE_WORKER_COUNT)
        self.assertEqual(config.route_worker_count, 6)
        self.assertIs(config.multirate_enabled, False)
        self.assertIs(config.owned_episode_enabled, False)
        self.assertEqual(config.tracer_spec.backend.backend_id, "cupy")
        self.assertEqual(config.tracer_spec.backend.device, "cuda:0")
        self.assertEqual(config.tracer_spec.backend.tile_size, 128)
        self.assertIs(config.tracer_spec.backend.allow_fallback, False)
        self.assertIs(config.tracer_spec.backend.fast_math, False)
        self.assertIs(config.tracer_spec.backend.deterministic_reductions, True)

    def test_source_prefix_is_bitwise_the_exact_m2_model(self) -> None:
        source = self.contract.source
        model = KNOWN_SOLAR_SYSTEM_V2
        self.assertEqual(source.source_ids, model.body_names)
        self.assertEqual(source.primary_source_id, "Sun")
        self.assertTrue(
            np.array_equal(
                source.initial_source_relative_positions_AU.view(np.uint64),
                np.asarray(
                    model.positions_primary_relative_ecliptic_AU,
                    dtype=np.float64,
                ).view(np.uint64),
            )
        )
        self.assertTrue(
            np.array_equal(
                source.initial_source_relative_velocities_AU_per_year.view(
                    np.uint64
                ),
                np.asarray(
                    model.velocities_primary_relative_ecliptic_AU_per_julian_year,
                    dtype=np.float64,
                ).view(np.uint64),
            )
        )
        self.assertTrue(
            np.array_equal(
                source.source_gm_AU3_per_year2.view(np.uint64),
                np.asarray(
                    model.gm_AU3_per_julian_year2,
                    dtype=np.float64,
                ).view(np.uint64),
            )
        )

    def test_all_reviewed_mismatches_fail_closed(self) -> None:
        contract = self.contract
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "tile_size=128"
        ):
            prepare_known_system_observed_fixed_run(
                BackendSpec("cupy", "cuda:0", 64)
            )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "canonical cuda:<index>"
        ):
            prepare_known_system_observed_fixed_run(
                BackendSpec("cupy", "gpu", 128)
            )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "KnownSystemRunProfile"
        ):
            prepare_known_system_observed_fixed_run(
                profile_id="m2-retained-fine-observed-candidate-10y-annual-v1"
            )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "source schedule identity"
        ):
            replace(contract, source_schedule_sha256="1" * 64)

        changed_positions = contract.tracer_positions_AU.copy()
        bits = changed_positions.view(np.uint64)
        bits[0, 0] += np.uint64(1)
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "differ bitwise"
        ):
            replace(contract, tracer_positions_AU=changed_positions)

        swapped_ids = list(contract.roster.object_ids)
        swapped_ids[0], swapped_ids[1] = swapped_ids[1], swapped_ids[0]
        changed_roster = replace(contract.roster, object_ids=tuple(swapped_ids))
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "roster differs"
        ):
            replace(contract, roster=changed_roster)

        changed_config = replace(contract.config, route_worker_count=5)
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "runner configuration"
        ):
            replace(contract, config=changed_config)

        changed_source_spec = replace(
            contract.config.source_spec,
            maximum_initial_barycenter_position_norm_AU=2.0e-12,
        )
        changed_source_config = replace(
            contract.config, source_spec=changed_source_spec
        )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "runner configuration"
        ):
            replace(contract, config=changed_source_config)

        changed_event_locator = replace(
            contract.config.route_spec.event_locator,
            relative_tolerance=5.0e-13,
        )
        changed_event_route = replace(
            contract.config.route_spec,
            event_locator=changed_event_locator,
        )
        changed_event_config = replace(
            contract.config, route_spec=changed_event_route
        )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "runner configuration"
        ):
            replace(contract, config=changed_event_config)

        negative_zero_route = replace(
            contract.config.route_spec,
            terminal_source_radii_AU=(-0.0,) + (0.0,) * 9,
        )
        negative_zero_config = replace(
            contract.config, route_spec=negative_zero_route
        )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "runner configuration"
        ):
            replace(contract, config=negative_zero_config)

        changed_provenance = replace(
            contract.source.source_provenance,
            citation="Falsified source citation",
        )
        changed_source = replace(
            contract.source, source_provenance=changed_provenance
        )
        with self.assertRaisesRegex(
            KnownSystemInputContractError, "source context"
        ):
            replace(contract, source=changed_source)

        with self.assertRaises(KnownSystemInputContractError):
            contract.roster.object_id_for_tracer_index(True)
        with self.assertRaises(KnownSystemInputContractError):
            contract.roster.object_id_for_tracer_index(108)


@unittest.skipIf(CP is None, "CUDA/CuPy device is unavailable")
class LongHorizonKnownSystemInputLiveGPUTests(unittest.TestCase):
    def test_materialization_and_short_one_year_fixed_run(self) -> None:
        materialized = build_known_system_observed_fixed_run_inputs()
        contract = materialized.contract
        state = materialized.lifecycle_state.dynamics_state
        self.assertEqual(materialized.genesis_state_sha256.__len__(), 64)
        self.assertEqual(materialized.content_sha256.__len__(), 64)
        self.assertTrue(
            bool(
                CP.array_equal(
                    state.tracer_indices, CP.arange(108, dtype=CP.uint64)
                )
            )
        )
        result = advance_fixed_lifecycle_segment(
            materialized.lifecycle_state,
            config=materialized.config,
            source=materialized.source,
        )
        lifecycle_result = result.authoritative_result.tracer_result
        self.assertEqual(
            lifecycle_result.state.dynamics_state.global_source_step,
            SOURCE_STEP_COUNT,
        )
        self.assertEqual(len(lifecycle_result.route_records), 0)
        self.assertEqual(len(lifecycle_result.new_removal_records), 0)
        self.assertEqual(
            int(np.count_nonzero(lifecycle_result.state.tracer_lifecycle)), 0
        )
        minimum = np.asarray(
            CP.asnumpy(lifecycle_result.minimum_source_distance_AU),
            dtype=np.float64,
        )
        self.assertEqual(minimum.shape, (108,))
        self.assertTrue(bool(np.all(np.isfinite(minimum))))
        self.assertGreater(float(np.min(minimum)), ALGORITHMIC_ROUTE_RADIUS_AU)
        self.assertFalse(any(result.result_contract.claims.values()))

    def test_retained_fine_states_use_the_selected_source_step_exactly(
        self,
    ) -> None:
        profiles = (
            (
                KnownSystemRunProfile.M2_FINE_CANDIDATE_10Y_ANNUAL,
                M2_FINE_CANDIDATE_SOURCE_STEPS_PER_MAP_STEP,
            ),
            (
                KnownSystemRunProfile.M2_FINE_REFINEMENT_10Y_ANNUAL,
                M2_FINE_REFINEMENT_SOURCE_STEPS_PER_MAP_STEP,
            ),
        )
        for profile_id, ratio in profiles:
            with self.subTest(profile_id=profile_id.value):
                materialized = build_known_system_observed_fixed_run_inputs(
                    profile_id=profile_id
                )
                state = materialized.lifecycle_state.dynamics_state
                self.assertEqual(
                    state.source_step_years.hex(),
                    FINE_SOURCE_STEP_YEARS.hex(),
                )
                self.assertEqual(
                    state.source_schedule_sha256,
                    FINE_SOURCE_SCHEDULE_SHA256,
                )
                self.assertEqual(
                    materialized.config
                    .tracer_spec.source_steps_per_map_step,
                    ratio,
                )
                self.assertEqual(
                    state.provenance.sha256,
                    materialized.contract.content_sha256,
                )
                self.assertIsNone(materialized.source)


if __name__ == "__main__":
    unittest.main()
