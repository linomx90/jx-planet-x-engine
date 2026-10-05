from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import unittest

import numpy as np

import jxplanetx.cuda_restricted_wh as restricted_wh
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHContractError,
    CUDARestrictedWHIntegrationError,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
    RESULT_CLAIMS,
    advance_cuda_restricted_wh,
    advance_cuda_restricted_wh_fused,
    source_path_tile_sha256,
    verify_source_path_tile,
)
from jxplanetx.engine import BackendSpec, Provenance


def _cupy_or_none():
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
        return cp
    except Exception:
        return None


CP = _cupy_or_none()
GPU_REQUIRED = unittest.skipIf(CP is None, "CUDA/CuPy device is unavailable")
SOLAR_GM = 4.0 * math.pi * math.pi
SCHEDULE_SHA256 = hashlib.sha256(
    b"cuda-restricted-wh-test-source-schedule-v1"
).hexdigest()


def _provenance(label: str = "source-tile") -> Provenance:
    return Provenance(
        source_id=f"jx-test.{label}",
        citation="Synthetic restricted-WH test construction",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _backend() -> BackendSpec:
    return BackendSpec("cupy", "cuda:0", 128)


def _tile(
    *,
    start: int = 0,
    node_count: int = 5,
    step: float = 0.001,
    encounter_radius: float = 1.0e-4,
    acceleration_bound: float = 100.0,
    expected_digest: str | None = None,
    secondary: bool = False,
) -> FrozenSourcePathTile:
    source_count = 2 if secondary else 1
    positions = np.zeros((node_count, source_count, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    gm = np.asarray(
        [SOLAR_GM, 9.547919384243266e-4 * SOLAR_GM]
        if secondary
        else [SOLAR_GM],
        dtype=np.float64,
    )
    radii = np.full(source_count, encounter_radius, dtype=np.float64)
    if secondary:
        # The test tile is an externally prescribed path, not a source solver.
        # Smooth node variation is sufficient to exercise differential kicks.
        for node in range(node_count):
            angle = 0.01 * (start + node)
            positions[node, 1] = (5.2 * math.cos(angle), 5.2 * math.sin(angle), 0.0)
            velocities[node, 1] = (
                -0.52 * math.sin(angle),
                0.52 * math.cos(angle),
                0.0,
            )
    return FrozenSourcePathTile(
        tile_id=f"synthetic-{start}-{node_count}",
        epoch_origin_years=2_451_545.0,
        global_source_step=start,
        source_step_years=step,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun", "Jupiter") if secondary else ("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=velocities,
        source_gm_AU3_per_year2=gm,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=np.full(
            source_count, acceleration_bound, dtype=np.float64
        ),
        provenance=_provenance(),
        content_sha256=expected_digest,
    )


def _state(count: int = 4, *, global_source_step: int = 0) -> CUDARestrictedWHState:
    cp = CP
    radius = cp.linspace(1.0, 1.5, count, dtype=cp.float64)
    position = cp.zeros((count, 3), dtype=cp.float64)
    velocity = cp.zeros((count, 3), dtype=cp.float64)
    if count:
        position[:, 0] = radius
        velocity[:, 1] = cp.sqrt(SOLAR_GM / radius)
    return CUDARestrictedWHState(
        snapshot_id="synthetic-primary-relative",
        epoch_origin_years=2_451_545.0,
        global_source_step=global_source_step,
        source_step_years=0.001,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        tracer_relative_positions_AU=position,
        tracer_relative_velocities_AU_per_year=velocity,
        tracer_indices=cp.arange(count, dtype=cp.uint64),
        provenance=_provenance("tracer-state"),
    )


class FrozenSourcePathTileContractTests(unittest.TestCase):
    def test_tile_owns_immutable_canonical_arrays_and_digest(self):
        input_positions = np.zeros((3, 1, 3), dtype=np.float64)
        input_velocities = np.zeros_like(input_positions)
        input_gm = np.asarray([SOLAR_GM], dtype=np.float64)
        input_radii = np.asarray([1.0e-4], dtype=np.float64)
        input_acceleration_bounds = np.asarray([100.0], dtype=np.float64)
        tile = FrozenSourcePathTile(
            tile_id="immutable",
            epoch_origin_years=0.0,
            global_source_step=8,
            source_step_years=-0.125,
            time_scale="TDB",
            frame="ICRF",
            axes="J2000",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=("Sun",),
            primary_source_id="Sun",
            source_schedule_sha256=SCHEDULE_SHA256,
            source_relative_positions_AU=input_positions,
            source_relative_velocities_AU_per_year=input_velocities,
            source_gm_AU3_per_year2=input_gm,
            source_encounter_radii_AU=input_radii,
            certified_relative_acceleration_bounds_AU_per_year2=(
                input_acceleration_bounds
            ),
            provenance=_provenance(),
        )
        digest = tile.content_sha256
        input_positions[0, 0, 0] = 99.0
        input_gm[0] = 1.0
        self.assertEqual(tile.source_relative_positions_AU[0, 0, 0], 0.0)
        self.assertEqual(tile.source_gm_AU3_per_year2[0], SOLAR_GM)
        self.assertFalse(tile.source_relative_positions_AU.flags.writeable)
        with self.assertRaises(ValueError):
            tile.source_relative_positions_AU.flags.writeable = True
        self.assertEqual(source_path_tile_sha256(tile), digest)
        self.assertEqual(verify_source_path_tile(tile), digest)
        self.assertEqual(tile.initial_epoch_years, -1.0)
        self.assertEqual(tile.final_epoch_years, -1.25)

    def test_expected_digest_is_verified_and_changes_with_content(self):
        original = _tile()
        accepted = _tile(expected_digest=original.content_sha256)
        self.assertEqual(accepted.content_sha256, original.content_sha256)
        with self.assertRaisesRegex(
            CUDARestrictedWHContractError, "does not match canonical content"
        ):
            _tile(encounter_radius=2.0e-4, expected_digest=original.content_sha256)

    def test_primary_slot_must_be_exact_zero_in_position_and_velocity(self):
        base = _tile()
        bad_position = np.array(base.source_relative_positions_AU, copy=True)
        bad_position[1, 0, 0] = 1.0e-30
        with self.assertRaisesRegex(
            CUDARestrictedWHContractError, "slot zero must be exact zero"
        ):
            replace(
                base,
                source_relative_positions_AU=bad_position,
                content_sha256=None,
            )
        bad_velocity = np.array(
            base.source_relative_velocities_AU_per_year, copy=True
        )
        bad_velocity[1, 0, 2] = -1.0e-30
        with self.assertRaisesRegex(
            CUDARestrictedWHContractError, "slot zero must be exact zero"
        ):
            replace(
                base,
                source_relative_velocities_AU_per_year=bad_velocity,
                content_sha256=None,
            )

    def test_contract_rejects_ambiguous_arrays_units_and_primary_order(self):
        base = _tile()
        with self.assertRaises(CUDARestrictedWHContractError):
            replace(
                base,
                source_relative_positions_AU=base.source_relative_positions_AU.tolist(),
                content_sha256=None,
            )
        with self.assertRaises(CUDARestrictedWHContractError):
            replace(base, length_unit="km", content_sha256=None)
        with self.assertRaises(CUDARestrictedWHContractError):
            replace(
                base,
                source_ids=("Jupiter", "Sun"),
                primary_source_id="Sun",
                content_sha256=None,
            )

    def test_reflective_tampering_is_detected(self):
        tile = _tile()
        object.__setattr__(tile, "tile_id", "tampered")
        with self.assertRaisesRegex(
            CUDARestrictedWHContractError, "no longer matches"
        ):
            verify_source_path_tile(tile)


class RestrictedWHContractTests(unittest.TestCase):
    def test_spec_is_cuda_only_and_ratio_is_strictly_integer(self):
        for invalid in (0, -1, 1.0, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(CUDARestrictedWHContractError):
                    FixedStepCUDARestrictedWHSpec(1, invalid, _backend())
        with self.assertRaises(CUDARestrictedWHContractError):
            FixedStepCUDARestrictedWHSpec(
                1, 1, BackendSpec("numpy", "cpu", 128)
            )
        with self.assertRaises(CUDARestrictedWHContractError):
            FixedStepCUDARestrictedWHSpec(1, 1, _backend(), threads_per_block=48)
        with self.assertRaises(CUDARestrictedWHContractError):
            FixedStepCUDARestrictedWHSpec(
                1, 1, _backend(), maximum_primary_dynamical_phase=0.3
            )
        for boundary in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(boundary=boundary):
                with self.assertRaises(CUDARestrictedWHContractError):
                    FixedStepCUDARestrictedWHSpec(
                        1,
                        1,
                        _backend(),
                        primary_outer_boundary_AU=boundary,
                    )

    def test_state_epoch_is_derived_from_integer_source_step(self):
        state = CUDARestrictedWHState(
            snapshot_id="opaque-contract",
            epoch_origin_years=100.0,
            global_source_step=7,
            source_step_years=0.125,
            time_scale="TDB",
            frame="ICRF",
            axes="J2000",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=("Sun",),
            primary_source_id="Sun",
            source_schedule_sha256=SCHEDULE_SHA256,
            tracer_relative_positions_AU=object(),
            tracer_relative_velocities_AU_per_year=object(),
            tracer_indices=object(),
            provenance=_provenance("opaque-state"),
        )
        self.assertEqual(state.epoch_years, 100.875)
        self.assertNotIn("epoch", state.__dataclass_fields__)

    def test_claims_are_fixed_all_false_and_no_checkpoint_api_exists(self):
        self.assertTrue(RESULT_CLAIMS)
        self.assertFalse(any(RESULT_CLAIMS.values()))
        self.assertNotIn("dump_cuda_restricted_wh_checkpoint", restricted_wh.__all__)
        self.assertNotIn("load_cuda_restricted_wh_checkpoint", restricted_wh.__all__)
        with self.assertRaises(TypeError):
            RESULT_CLAIMS["production_qualified"] = True


@GPU_REQUIRED
class RestrictedWHGPUTests(unittest.TestCase):
    def assert_fused_matches_stepwise(
        self,
        state: CUDARestrictedWHState,
        tile: FrozenSourcePathTile,
        spec: FixedStepCUDARestrictedWHSpec,
    ) -> tuple[object, object]:
        stepwise = advance_cuda_restricted_wh(state, tile, spec)
        fused = advance_cuda_restricted_wh_fused(state, tile, spec)
        self.assertEqual(
            fused.state.global_source_step, stepwise.state.global_source_step
        )
        self.assertEqual(
            fused.maximum_kepler_iterations,
            stepwise.maximum_kepler_iterations,
        )
        for fused_array, stepwise_array in (
            (
                fused.state.tracer_relative_positions_AU,
                stepwise.state.tracer_relative_positions_AU,
            ),
            (
                fused.state.tracer_relative_velocities_AU_per_year,
                stepwise.state.tracer_relative_velocities_AU_per_year,
            ),
            (fused.state.tracer_indices, stepwise.state.tracer_indices),
            (
                fused.minimum_endpoint_source_distance_AU,
                stepwise.minimum_endpoint_source_distance_AU,
            ),
        ):
            self.assertTrue(bool(CP.array_equal(fused_array, stepwise_array)))
        self.assertEqual(
            fused.runtime_identity["kernel_id"], restricted_wh.FUSED_KERNEL_ID
        )
        self.assertEqual(
            fused.runtime_identity["execution_kernel_source_sha256"],
            restricted_wh.FUSED_CUDA_SOURCE_SHA256,
        )
        self.assertIsInstance(fused.runtime_identity["nvrtc_version"], tuple)
        with self.assertRaises(TypeError):
            fused.runtime_identity["device_uuid"] = "forged"
        with self.assertRaises(TypeError):
            fused.runtime_identity["nvrtc_version"][0] = 0
        self.assertFalse(any(fused.claims.values()))
        return stepwise, fused

    def test_fused_primary_only_matches_stepwise_bitwise_for_block_sizes(self):
        initial = _state(65)
        tile = _tile(start=0, node_count=9)
        fused_results = []
        for threads in (32, 64, 128):
            _, fused = self.assert_fused_matches_stepwise(
                initial,
                tile,
                FixedStepCUDARestrictedWHSpec(
                    8, 1, _backend(), threads_per_block=threads
                ),
            )
            fused_results.append(fused)
        for candidate in fused_results[1:]:
            self.assertTrue(
                bool(
                    CP.array_equal(
                        fused_results[0].state.tracer_relative_positions_AU,
                        candidate.state.tracer_relative_positions_AU,
                    )
                )
            )
            self.assertTrue(
                bool(
                    CP.array_equal(
                        fused_results[0].state.tracer_relative_velocities_AU_per_year,
                        candidate.state.tracer_relative_velocities_AU_per_year,
                    )
                )
            )

    def test_fused_multisource_integer_ratio_matches_stepwise_bitwise(self):
        initial = replace(_state(17), source_ids=("Sun", "Jupiter"))
        self.assert_fused_matches_stepwise(
            initial,
            _tile(start=0, node_count=7, secondary=True),
            FixedStepCUDARestrictedWHSpec(2, 3, _backend()),
        )

    def test_fused_negative_time_matches_stepwise_and_split_bitwise(self):
        initial = replace(_state(8), source_step_years=-0.001)
        tile = _tile(start=0, node_count=5, step=-0.001)
        _, continuous = self.assert_fused_matches_stepwise(
            initial,
            tile,
            FixedStepCUDARestrictedWHSpec(4, 1, _backend()),
        )
        first = advance_cuda_restricted_wh_fused(
            initial,
            _tile(start=0, node_count=3, step=-0.001),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        second = advance_cuda_restricted_wh_fused(
            first.state,
            _tile(start=2, node_count=3, step=-0.001),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
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

    def test_fused_guard_failure_matches_stepwise_handoff(self):
        initial = _state(4)
        tile = _tile(encounter_radius=2.0)
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        handoffs = []
        for advance in (
            advance_cuda_restricted_wh,
            advance_cuda_restricted_wh_fused,
        ):
            with self.assertRaises(CUDARestrictedWHIntegrationError) as raised:
                advance(initial, tile, spec)
            handoffs.append(raised.exception.handoff)
        self.assertIs(handoffs[0].state, initial)
        self.assertIs(handoffs[1].state, initial)
        self.assertEqual(
            (
                handoffs[1].failure_global_source_step,
                handoffs[1].map_step_index,
                handoffs[1].phase,
                handoffs[1].reason,
                handoffs[1].lane_index,
                handoffs[1].tracer_index,
                handoffs[1].source_index,
            ),
            (
                handoffs[0].failure_global_source_step,
                handoffs[0].map_step_index,
                handoffs[0].phase,
                handoffs[0].reason,
                handoffs[0].lane_index,
                handoffs[0].tracer_index,
                handoffs[0].source_index,
            ),
        )

    def test_fused_zero_lane_and_distant_tidal_cases_match_stepwise(self):
        self.assert_fused_matches_stepwise(
            _state(0),
            _tile(start=0, node_count=5),
            FixedStepCUDARestrictedWHSpec(4, 1, _backend()),
        )

        radius = math.ldexp(1.0, 54)
        positions = np.zeros((2, 2, 3), dtype=np.float64)
        positions[:, 1, 0] = radius
        distant_tile = replace(
            _tile(node_count=2, secondary=True),
            source_relative_positions_AU=positions,
            source_relative_velocities_AU_per_year=np.zeros_like(positions),
            source_gm_AU3_per_year2=np.asarray(
                (SOLAR_GM, math.ldexp(1.0, 162)), dtype=np.float64
            ),
            certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
                (100.0, 100.0), dtype=np.float64
            ),
            content_sha256=None,
        )
        self.assert_fused_matches_stepwise(
            replace(_state(1), source_ids=("Sun", "Jupiter")),
            distant_tile,
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )

    def test_quiet_multisource_map_converges_second_order_to_dop853(self):
        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            self.skipTest("SciPy independent-reference extra is unavailable")

        mu_primary = SOLAR_GM
        mu_secondary = 1.0e-3 * SOLAR_GM
        source_radius = 5.2
        source_motion = math.sqrt(
            (mu_primary + mu_secondary) / source_radius**3
        )
        duration = 2.0
        initial_position = np.asarray((10.0, 0.0, 0.0), dtype=np.float64)
        initial_velocity = np.asarray(
            (0.0, math.sqrt(mu_primary / 10.0), 0.0), dtype=np.float64
        )

        def source_position(time: float) -> np.ndarray:
            angle = source_motion * time
            return np.asarray(
                (
                    source_radius * math.cos(angle),
                    source_radius * math.sin(angle),
                    0.0,
                ),
                dtype=np.float64,
            )

        def reference_rhs(time: float, state: np.ndarray) -> np.ndarray:
            position = state[:3]
            velocity = state[3:]
            source = source_position(time)
            relative = source - position
            primary_radius = float(np.linalg.norm(position))
            source_distance = float(np.linalg.norm(source))
            relative_distance = float(np.linalg.norm(relative))
            acceleration = -mu_primary * position / primary_radius**3
            acceleration += mu_secondary * (
                relative / relative_distance**3 - source / source_distance**3
            )
            return np.concatenate((velocity, acceleration))

        reference = solve_ivp(
            reference_rhs,
            (0.0, duration),
            np.concatenate((initial_position, initial_velocity)),
            method="DOP853",
            rtol=2.0e-13,
            atol=1.0e-15,
        )
        self.assertTrue(reference.success)
        expected = reference.y[:, -1]

        errors: list[float] = []
        for step in (0.04, 0.02, 0.01):
            map_steps = int(round(duration / step))
            times = np.arange(map_steps + 1, dtype=np.float64) * step
            positions = np.zeros((map_steps + 1, 2, 3), dtype=np.float64)
            velocities = np.zeros_like(positions)
            angles = source_motion * times
            positions[:, 1, 0] = source_radius * np.cos(angles)
            positions[:, 1, 1] = source_radius * np.sin(angles)
            velocities[:, 1, 0] = -source_radius * source_motion * np.sin(
                angles
            )
            velocities[:, 1, 1] = source_radius * source_motion * np.cos(
                angles
            )
            tile = FrozenSourcePathTile(
                tile_id=f"analytic-circular-secondary-{step.hex()}",
                epoch_origin_years=0.0,
                global_source_step=0,
                source_step_years=step,
                time_scale="TDB",
                frame="ICRF",
                axes="J2000_ECLIPTIC",
                coordinate_system=COORDINATE_SYSTEM,
                length_unit="AU",
                time_unit="year",
                unit_system_id="AU-YEAR",
                source_ids=("Sun", "secondary"),
                primary_source_id="Sun",
                source_schedule_sha256=SCHEDULE_SHA256,
                source_relative_positions_AU=positions,
                source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=np.asarray(
                    (mu_primary, mu_secondary), dtype=np.float64
                ),
                source_encounter_radii_AU=np.asarray(
                    (0.1, 0.1), dtype=np.float64
                ),
                certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
                    (10.0, 10.0), dtype=np.float64
                ),
                provenance=_provenance("analytic-circular-secondary"),
            )
            state = replace(
                _state(1),
                epoch_origin_years=0.0,
                source_step_years=step,
                source_ids=("Sun", "secondary"),
                tracer_relative_positions_AU=CP.asarray(
                    initial_position[None, :]
                ),
                tracer_relative_velocities_AU_per_year=CP.asarray(
                    initial_velocity[None, :]
                ),
            )
            result = advance_cuda_restricted_wh(
                state,
                tile,
                FixedStepCUDARestrictedWHSpec(map_steps, 1, _backend()),
            )
            actual = np.concatenate(
                (
                    CP.asnumpy(result.state.tracer_relative_positions_AU[0]),
                    CP.asnumpy(
                        result.state.tracer_relative_velocities_AU_per_year[0]
                    ),
                )
            )
            errors.append(float(np.max(np.abs(actual - expected))))

        self.assertGreater(errors[0] / errors[1], 3.5)
        self.assertGreater(errors[1] / errors[2], 3.5)
        self.assertLess(errors[1] / errors[2], 4.5)

    def test_primary_only_split_is_bitwise_identical_to_one_call(self):
        initial = _state(8)
        continuous = advance_cuda_restricted_wh(
            initial,
            _tile(start=0, node_count=5),
            FixedStepCUDARestrictedWHSpec(4, 1, _backend()),
        )
        first = advance_cuda_restricted_wh(
            initial,
            _tile(start=0, node_count=3),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        second = advance_cuda_restricted_wh(
            first.state,
            _tile(start=2, node_count=3),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        self.assertEqual(continuous.state.global_source_step, 4)
        self.assertEqual(second.state.global_source_step, 4)
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
        self.assertFalse(any(continuous.claims.values()))

    def test_negative_time_split_is_bitwise_identical_to_one_call(self):
        initial = replace(_state(8), source_step_years=-0.001)
        continuous = advance_cuda_restricted_wh(
            initial,
            _tile(start=0, node_count=5, step=-0.001),
            FixedStepCUDARestrictedWHSpec(4, 1, _backend()),
        )
        first = advance_cuda_restricted_wh(
            initial,
            _tile(start=0, node_count=3, step=-0.001),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        second = advance_cuda_restricted_wh(
            first.state,
            _tile(start=2, node_count=3, step=-0.001),
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        self.assertEqual(continuous.state.epoch_years, second.state.epoch_years)
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

    def test_lane_permutation_and_block_size_are_bitwise_invariant(self):
        initial = _state(65)
        tile = _tile(start=0, node_count=3)
        by_block = []
        for threads in (32, 64, 128):
            by_block.append(
                advance_cuda_restricted_wh(
                    initial,
                    tile,
                    FixedStepCUDARestrictedWHSpec(
                        2, 1, _backend(), threads_per_block=threads
                    ),
                )
            )
        for candidate in by_block[1:]:
            self.assertTrue(
                bool(
                    CP.array_equal(
                        by_block[0].state.tracer_relative_positions_AU,
                        candidate.state.tracer_relative_positions_AU,
                    )
                )
            )
            self.assertTrue(
                bool(
                    CP.array_equal(
                        by_block[0].state.tracer_relative_velocities_AU_per_year,
                        candidate.state.tracer_relative_velocities_AU_per_year,
                    )
                )
            )

        permutation = CP.arange(64, -1, -1, dtype=CP.int64)
        permuted = replace(
            initial,
            tracer_relative_positions_AU=(
                initial.tracer_relative_positions_AU[permutation]
            ),
            tracer_relative_velocities_AU_per_year=(
                initial.tracer_relative_velocities_AU_per_year[permutation]
            ),
            tracer_indices=initial.tracer_indices[permutation],
        )
        permuted_result = advance_cuda_restricted_wh(
            permuted,
            tile,
            FixedStepCUDARestrictedWHSpec(
                2, 1, _backend(), threads_per_block=64
            ),
        )
        canonical_order = CP.argsort(permuted_result.state.tracer_indices)
        self.assertTrue(
            bool(
                CP.array_equal(
                    by_block[0].state.tracer_relative_positions_AU,
                    permuted_result.state.tracer_relative_positions_AU[
                        canonical_order
                    ],
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    by_block[0].state.tracer_relative_velocities_AU_per_year,
                    permuted_result.state.tracer_relative_velocities_AU_per_year[
                        canonical_order
                    ],
                )
            )
        )

    def test_zero_tracer_segment_advances_lattice_without_device_work(self):
        result = advance_cuda_restricted_wh(
            _state(0),
            _tile(start=0, node_count=5),
            FixedStepCUDARestrictedWHSpec(4, 1, _backend()),
        )
        self.assertEqual(result.state.global_source_step, 4)
        self.assertEqual(result.maximum_kepler_iterations, 0)
        self.assertEqual(result.state.tracer_relative_positions_AU.shape, (0, 3))
        self.assertEqual(result.minimum_endpoint_source_distance_AU.shape, (0,))

    def test_integer_ratio_uses_only_matching_fine_source_nodes(self):
        initial = _state(4)
        result = advance_cuda_restricted_wh(
            initial,
            _tile(start=0, node_count=7),
            FixedStepCUDARestrictedWHSpec(2, 3, _backend()),
        )
        self.assertEqual(result.state.global_source_step, 6)
        self.assertTrue(
            bool(CP.all(CP.isfinite(result.state.tracer_relative_positions_AU)))
        )

    def test_guard_failure_is_transactional_and_hands_back_input(self):
        initial = _state(4)
        position_before = initial.tracer_relative_positions_AU.copy()
        velocity_before = initial.tracer_relative_velocities_AU_per_year.copy()
        with self.assertRaises(CUDARestrictedWHIntegrationError) as raised:
            advance_cuda_restricted_wh(
                initial,
                _tile(encounter_radius=2.0),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
            )
        handoff = raised.exception.handoff
        self.assertIs(handoff.state, initial)
        self.assertEqual(handoff.committed_map_steps, 0)
        self.assertFalse(handoff.checkpoint_qualified)
        self.assertEqual(handoff.failure_global_source_step, 0)
        self.assertEqual(handoff.phase, "PRE_KICK_GUARD")
        self.assertEqual(handoff.reason, "SOURCE_RADIUS_GUARD")
        self.assertTrue(
            bool(
                CP.array_equal(
                    initial.tracer_relative_positions_AU, position_before
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    initial.tracer_relative_velocities_AU_per_year,
                    velocity_before,
                )
            )
        )

    def test_swept_guard_rejects_endpoint_safe_but_uncertified_step(self):
        initial = _state(1)
        with self.assertRaises(CUDARestrictedWHIntegrationError) as raised:
            advance_cuda_restricted_wh(
                initial,
                _tile(encounter_radius=0.997, acceleration_bound=100.0),
                FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
            )
        handoff = raised.exception.handoff
        self.assertEqual(handoff.phase, "PRE_KICK_GUARD")
        self.assertEqual(handoff.reason, "SWEPT_SOURCE_RADIUS_GUARD")
        self.assertIs(handoff.state, initial)

    def test_primary_phase_uses_certified_swept_lower_radius(self):
        initial = _state(1)
        tile = _tile(encounter_radius=1.0e-4, acceleration_bound=100.0)
        baseline = advance_cuda_restricted_wh(
            initial,
            tile,
            FixedStepCUDARestrictedWHSpec(
                1, 1, _backend(), maximum_primary_dynamical_phase=0.01
            ),
        )
        self.assertGreater(
            float(
                CP.linalg.norm(
                    baseline.state.tracer_relative_positions_AU[0]
                ).item()
            ),
            tile.source_encounter_radii_AU[0],
        )
        with self.assertRaises(CUDARestrictedWHIntegrationError) as raised:
            advance_cuda_restricted_wh(
                initial,
                tile,
                FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    _backend(),
                    maximum_primary_dynamical_phase=0.0063,
                ),
            )
        self.assertEqual(raised.exception.handoff.phase, "PRE_KICK_GUARD")
        self.assertEqual(raised.exception.handoff.reason, "PRIMARY_PHASE_GUARD")

    def test_outer_boundary_possible_crossing_fails_closed_in_both_paths(self):
        initial = _state(1)
        position = initial.tracer_relative_positions_AU.copy()
        velocity = initial.tracer_relative_velocities_AU_per_year.copy()
        position[0] = CP.asarray((0.9, 0.0, 0.0), dtype=CP.float64)
        velocity[0] = CP.asarray((2.0, 0.0, 0.0), dtype=CP.float64)
        initial = replace(
            initial,
            tracer_relative_positions_AU=position,
            tracer_relative_velocities_AU_per_year=velocity,
        )
        spec = FixedStepCUDARestrictedWHSpec(
            1,
            1,
            _backend(),
            primary_outer_boundary_AU=0.9005,
        )
        for advance in (
            advance_cuda_restricted_wh,
            advance_cuda_restricted_wh_fused,
        ):
            with self.subTest(advance=advance.__name__):
                with self.assertRaises(CUDARestrictedWHIntegrationError) as raised:
                    advance(initial, _tile(node_count=2), spec)
                self.assertEqual(
                    raised.exception.handoff.reason,
                    "OUTER_BOUNDARY_GUARD",
                )
                self.assertIs(raised.exception.handoff.state, initial)

    def test_outer_boundary_must_exceed_primary_encounter_radius(self):
        with self.assertRaisesRegex(
            CUDARestrictedWHContractError,
            "must exceed the primary encounter radius",
        ):
            advance_cuda_restricted_wh_fused(
                _state(1),
                _tile(node_count=2, encounter_radius=1.0),
                FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    _backend(),
                    primary_outer_boundary_AU=1.0,
                ),
            )

    def test_secondary_tile_is_not_modified_and_output_is_finite(self):
        cp = CP
        tile = _tile(secondary=True)
        digest = tile.content_sha256
        initial = _state(4)
        initial = replace(initial, source_ids=("Sun", "Jupiter"))
        result = advance_cuda_restricted_wh(
            initial,
            tile,
            FixedStepCUDARestrictedWHSpec(2, 1, _backend()),
        )
        self.assertEqual(verify_source_path_tile(tile), digest)
        self.assertTrue(
            bool(cp.all(cp.isfinite(result.state.tracer_relative_positions_AU)))
        )
        self.assertTrue(
            bool(
                cp.all(
                    cp.isfinite(
                        result.state.tracer_relative_velocities_AU_per_year
                    )
                )
            )
        )

    def test_distant_secondary_tidal_kick_does_not_cancel(self):
        radius = math.ldexp(1.0, 54)
        positions = np.zeros((2, 2, 3), dtype=np.float64)
        positions[:, 1, 0] = radius
        velocities = np.zeros_like(positions)
        distant_tile = replace(
            _tile(node_count=2, secondary=True),
            source_relative_positions_AU=positions,
            source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=np.asarray(
                (SOLAR_GM, math.ldexp(1.0, 162)), dtype=np.float64
            ),
            certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
                (100.0, 100.0), dtype=np.float64
            ),
            content_sha256=None,
        )
        primary_state = _state(1)
        primary_only = advance_cuda_restricted_wh(
            primary_state,
            _tile(node_count=2),
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )
        with_secondary = advance_cuda_restricted_wh(
            replace(primary_state, source_ids=("Sun", "Jupiter")),
            distant_tile,
            FixedStepCUDARestrictedWHSpec(1, 1, _backend()),
        )
        delta_vx = float(
            (
                with_secondary.state.tracer_relative_velocities_AU_per_year[0, 0]
                - primary_only.state.tracer_relative_velocities_AU_per_year[0, 0]
            ).item()
        )
        # The exact initial differential acceleration tends to +2 AU/year^2.
        # A naive subtraction of the two source accelerations produces zero at
        # this binary64 scale; the stable tidal form must retain the kick.
        self.assertGreater(delta_vx, 1.5e-3)
        self.assertLess(delta_vx, 2.5e-3)


if __name__ == "__main__":
    unittest.main()
