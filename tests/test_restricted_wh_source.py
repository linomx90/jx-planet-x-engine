from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import verify_source_path_tile
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    advance_cuda_restricted_wh_fused,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.restricted_wh_lifecycle import (
    REMOVED_SOURCE_RADIUS,
    initialize_restricted_wh_lifecycle,
)
from jxplanetx.restricted_wh_source import (
    MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
    MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR_IN_EPS,
    MAXIMUM_SOURCE_STEPS_PER_TILE,
    RestrictedWHSourceContractError,
    RestrictedWHSourceIntegrationError,
    SOURCE_PROVIDER_CLAIMS,
    WisdomHolmanSourceTileSpec,
    advance_cuda_restricted_wh_hybrid_with_wisdom_holman_source,
    advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source,
    advance_cuda_restricted_wh_with_wisdom_holman_source,
    build_wisdom_holman_source_tile,
)


try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - exercised on CPU-only CI
    cp = None


MU = 4.0 * math.pi * math.pi
SCHEDULE_SHA256 = hashlib.sha256(b"restricted-wh-source-test").hexdigest()


def _provenance() -> Provenance:
    return Provenance(
        "jx.test.restricted-wh-source",
        "synthetic two-body source-provider regression",
        "v1",
        hashlib.sha256(b"restricted-wh-source-input").hexdigest(),
    )


def _arrays():
    positions = np.asarray(((0.0, 0.0, 0.0), (5.2, 0.0, 0.0)), dtype=np.float64)
    velocities = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, math.sqrt((MU + 1.0e-3 * MU) / 5.2), 0.0),
        ),
        dtype=np.float64,
    )
    gm = np.asarray((MU, 1.0e-3 * MU), dtype=np.float64)
    radii = np.asarray((0.01, 0.01), dtype=np.float64)
    bounds = np.asarray((100.0, 100.0), dtype=np.float64)
    return positions, velocities, gm, radii, bounds


def _build(**changes):
    positions, velocities, gm, radii, bounds = _arrays()
    arguments = {
        "tile_id": "synthetic-source-tile",
        "epoch_origin_years": 0.0,
        "global_source_step": 0,
        "time_scale": "TDB",
        "frame": "SYNTHETIC_INERTIAL",
        "axes": "CARTESIAN_RIGHT_HANDED",
        "source_ids": ("primary", "secondary"),
        "primary_source_id": "primary",
        "source_schedule_sha256": SCHEDULE_SHA256,
        "initial_source_relative_positions_AU": positions,
        "initial_source_relative_velocities_AU_per_year": velocities,
        "source_gm_AU3_per_year2": gm,
        "source_encounter_radii_AU": radii,
        "certified_relative_acceleration_bounds_AU_per_year2": bounds,
        "provenance": _provenance(),
        "spec": WisdomHolmanSourceTileSpec(0.01, 8),
    }
    arguments.update(changes)
    return build_wisdom_holman_source_tile(**arguments)


class RestrictedWHSourceTests(unittest.TestCase):
    def test_build_is_replay_verified_immutable_and_deterministic(self):
        positions, velocities, _, _, _ = _arrays()
        position_before = positions.copy()
        velocity_before = velocities.copy()
        first = _build(
            initial_source_relative_positions_AU=positions,
            initial_source_relative_velocities_AU_per_year=velocities,
        )
        second = _build()

        self.assertEqual(first.tile.node_count, 9)
        np.testing.assert_array_equal(
            first.tile.source_relative_positions_AU[0], positions
        )
        np.testing.assert_array_equal(
            first.tile.source_relative_velocities_AU_per_year[0], velocities
        )
        self.assertTrue(
            np.all(first.tile.source_relative_positions_AU[:, 0, :] == 0.0)
        )
        self.assertTrue(
            np.all(first.tile.source_relative_velocities_AU_per_year[:, 0, :] == 0.0)
        )
        self.assertFalse(first.tile.source_relative_positions_AU.flags.writeable)
        self.assertEqual(verify_source_path_tile(first.tile), first.tile.content_sha256)
        self.assertEqual(first.tile.content_sha256, second.tile.content_sha256)
        self.assertEqual(
            first.source_map_result_content_sha256,
            second.source_map_result_content_sha256,
        )
        self.assertEqual(
            first.source_map_verification_content_sha256,
            second.source_map_verification_content_sha256,
        )
        self.assertTrue(first.runtime_identity["exact_replay_required"])
        self.assertEqual(dict(first.claims), dict(SOURCE_PROVIDER_CLAIMS))
        self.assertFalse(any(first.claims.values()))
        np.testing.assert_array_equal(positions, position_before)
        np.testing.assert_array_equal(velocities, velocity_before)
        self.assertFalse(
            np.array_equal(
                first.tile.source_relative_positions_AU[-1], positions
            )
        )

    def test_realistic_barycentric_roundoff_is_explicit_and_exact(self):
        positions = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (
                    10.771871959743002,
                    -19.893444413801102,
                    0.4066052474663806,
                ),
            ),
            dtype=np.float64,
        )
        gm = np.asarray((39.4769, 0.09270611361871046), dtype=np.float64)
        radius = float(np.linalg.norm(positions[1]))
        speed = math.sqrt(float(np.sum(gm, dtype=np.float64)) / radius)
        velocities = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (
                    -positions[1, 1] * speed / radius,
                    positions[1, 0] * speed / radius,
                    0.0,
                ),
            ),
            dtype=np.float64,
        )
        raw_positions = positions.copy()
        raw_velocities = velocities.copy()
        result = _build(
            initial_source_relative_positions_AU=positions,
            initial_source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=gm,
            source_encounter_radii_AU=np.asarray((0.01, 0.01), dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
                (100.0, 100.0), dtype=np.float64
            ),
        )

        receipt = result.runtime_identity["coordinate_canonicalization"]
        self.assertEqual(
            receipt["maximum_allowed_input_relative_error_in_eps"], 512.0
        )
        self.assertEqual(
            receipt["maximum_allowed_input_relative_error"],
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
        )
        self.assertGreater(receipt["position_changed_component_count"], 0)
        self.assertLessEqual(
            receipt["position_maximum_forward_error_bound_fraction"], 1.0
        )
        self.assertLessEqual(
            receipt["position_maximum_input_relative_error"],
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
        )
        self.assertTrue(receipt["canonicalization_idempotent"])
        self.assertTrue(receipt["tile_node_zero_exact"])
        self.assertEqual(
            receipt["raw_relative_input_sha256"], result.input_content_sha256
        )
        self.assertNotEqual(
            receipt["raw_relative_input_sha256"],
            receipt["canonical_relative_input_sha256"],
        )
        self.assertRegex(
            receipt["canonical_barycentric_input_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertNotEqual(
            receipt["canonical_barycentric_input_sha256"],
            receipt["canonical_relative_input_sha256"],
        )
        self.assertFalse(
            np.array_equal(
                result.tile.source_relative_positions_AU[0], raw_positions
            )
        )
        np.testing.assert_array_equal(positions, raw_positions)
        np.testing.assert_array_equal(velocities, raw_velocities)

    def test_unsafe_coordinate_cancellation_fails_closed(self):
        positions = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0e20, 0.0, 0.0)),
            dtype=np.float64,
        )
        velocities = np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        gm = np.ones(3, dtype=np.float64)
        with self.assertRaisesRegex(
            RestrictedWHSourceContractError,
            "forward-error or conditioning bound",
        ):
            _build(
                source_ids=("primary", "secondary", "tertiary"),
                initial_source_relative_positions_AU=positions,
                initial_source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=gm,
                source_encounter_radii_AU=np.full(3, 0.01, dtype=np.float64),
                certified_relative_acceleration_bounds_AU_per_year2=np.full(
                    3, 100.0, dtype=np.float64
                ),
            )

    def test_contract_rejects_invalid_lattice_and_source_arrays(self):
        with self.assertRaises(RestrictedWHSourceContractError):
            WisdomHolmanSourceTileSpec(0.0, 1)
        with self.assertRaises(RestrictedWHSourceContractError):
            WisdomHolmanSourceTileSpec(0.01, 0)
        with self.assertRaises(RestrictedWHSourceContractError):
            WisdomHolmanSourceTileSpec(
                0.01, MAXIMUM_SOURCE_STEPS_PER_TILE + 1
            )

        positions, _, _, _, _ = _arrays()
        invalid_primary = positions.copy()
        invalid_primary[0, 0] = 1.0
        with self.assertRaises(RestrictedWHSourceContractError):
            _build(initial_source_relative_positions_AU=invalid_primary)

        _, _, gm, _, _ = _arrays()
        invalid_gm = gm.copy()
        invalid_gm[1] = 0.0
        with self.assertRaises(RestrictedWHSourceContractError):
            _build(source_gm_AU3_per_year2=invalid_gm)

    def test_massive_close_approach_policy_fails_closed(self):
        with self.assertRaises(RestrictedWHSourceIntegrationError):
            _build(
                spec=WisdomHolmanSourceTileSpec(
                    0.01,
                    2,
                    minimum_encounter_pair_separation_AU=10.0,
                )
            )

    def test_result_claim_mapping_is_immutable(self):
        result = _build()
        with self.assertRaises(TypeError):
            result.claims["production_qualified"] = True
        with self.assertRaises(TypeError):
            result.runtime_identity["provider_id"] = "changed"
        with self.assertRaises(TypeError):
            result.runtime_identity["source_map_runtime_identity"][
                "kernel_id"
            ] = "changed"
        with self.assertRaises(RestrictedWHSourceContractError):
            replace(result, runtime_identity={})
        forged_runtime = dict(result.runtime_identity)
        forged_runtime["coordinate_canonicalization"] = {
            "policy": "FORGED",
            "content_sha256": "0" * 64,
        }
        with self.assertRaises(RestrictedWHSourceContractError):
            replace(result, runtime_identity=forged_runtime)
        with self.assertRaises(TypeError):
            replace(result.coordinate_canonicalization)
        with self.assertRaises(RestrictedWHSourceContractError):
            replace(result, claims={"production_qualified": True})


@unittest.skipIf(cp is None, "CUDA device is required")
class RestrictedWHSourceGPUTests(unittest.TestCase):
    def test_combined_entry_point_links_source_and_tracer_custody(self):
        positions, velocities, gm, radii, bounds = _arrays()
        initial = CUDARestrictedWHState(
            snapshot_id="synthetic-combined-segment",
            epoch_origin_years=0.0,
            global_source_step=0,
            source_step_years=0.01,
            time_scale="TDB",
            frame="SYNTHETIC_INERTIAL",
            axes="CARTESIAN_RIGHT_HANDED",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=("primary", "secondary"),
            primary_source_id="primary",
            source_schedule_sha256=SCHEDULE_SHA256,
            tracer_relative_positions_AU=cp.asarray(((10.0, 0.0, 0.0),)),
            tracer_relative_velocities_AU_per_year=cp.asarray(
                ((0.0, math.sqrt(MU / 10.0), 0.0),)
            ),
            tracer_indices=cp.asarray((7,), dtype=cp.uint64),
            provenance=_provenance(),
        )
        tracer_spec = FixedStepCUDARestrictedWHSpec(
            8,
            1,
            BackendSpec("cupy", "cuda:0", 128),
        )
        combined = advance_cuda_restricted_wh_with_wisdom_holman_source(
            initial,
            tile_id="synthetic-combined-source-tile",
            initial_source_relative_positions_AU=positions,
            initial_source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=gm,
            source_encounter_radii_AU=radii,
            certified_relative_acceleration_bounds_AU_per_year2=bounds,
            source_provenance=_provenance(),
            source_spec=WisdomHolmanSourceTileSpec(0.01, 8),
            tracer_spec=tracer_spec,
        )
        direct = advance_cuda_restricted_wh_fused(
            initial,
            combined.source_build.tile,
            tracer_spec,
        )
        self.assertEqual(
            combined.tracer_result.source_tile_sha256,
            combined.source_build.tile.content_sha256,
        )
        self.assertTrue(
            bool(
                cp.array_equal(
                    combined.tracer_result.state.tracer_relative_positions_AU,
                    direct.state.tracer_relative_positions_AU,
                )
            )
        )
        self.assertTrue(
            bool(
                cp.array_equal(
                    combined.tracer_result.state.tracer_relative_velocities_AU_per_year,
                    direct.state.tracer_relative_velocities_AU_per_year,
                )
            )
        )
        self.assertFalse(any(combined.claims.values()))

    def test_combined_hybrid_entry_point_routes_and_links_custody(self):
        positions, velocities, gm, radii, bounds = _arrays()
        guarded_radii = radii.copy()
        guarded_radii[0] = 0.997
        initial = CUDARestrictedWHState(
            snapshot_id="synthetic-combined-hybrid-segment",
            epoch_origin_years=0.0,
            global_source_step=0,
            source_step_years=0.01,
            time_scale="TDB",
            frame="SYNTHETIC_INERTIAL",
            axes="CARTESIAN_RIGHT_HANDED",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=("primary", "secondary"),
            primary_source_id="primary",
            source_schedule_sha256=SCHEDULE_SHA256,
            tracer_relative_positions_AU=cp.asarray(((1.0, 0.0, 0.0),)),
            tracer_relative_velocities_AU_per_year=cp.asarray(
                ((0.0, math.sqrt(MU), 0.0),)
            ),
            tracer_indices=cp.asarray((19,), dtype=cp.uint64),
            provenance=_provenance(),
        )
        combined = (
            advance_cuda_restricted_wh_hybrid_with_wisdom_holman_source(
                initial,
                tile_id="synthetic-combined-hybrid-source-tile",
                initial_source_relative_positions_AU=positions,
                initial_source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=gm,
                source_encounter_radii_AU=guarded_radii,
                certified_relative_acceleration_bounds_AU_per_year2=bounds,
                source_provenance=_provenance(),
                source_spec=WisdomHolmanSourceTileSpec(0.01, 1),
                tracer_spec=FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    BackendSpec("cupy", "cuda:0", 128),
                ),
            )
        )
        self.assertEqual(
            combined.tracer_result.source_tile_sha256,
            combined.source_build.tile.content_sha256,
        )
        self.assertEqual(len(combined.tracer_result.route_records), 1)
        self.assertEqual(
            combined.tracer_result.route_records[0].tracer_index, 19
        )
        self.assertFalse(
            combined.tracer_result.route_records[0]
            .route_result.terminated_at_event
        )
        self.assertFalse(any(combined.claims.values()))

    def test_combined_lifecycle_entry_point_freezes_terminal_lane(self):
        positions, velocities, gm, radii, bounds = _arrays()
        terminal_radii = radii.copy()
        terminal_radii[0] = 1.0
        dynamics = CUDARestrictedWHState(
            snapshot_id="synthetic-combined-lifecycle-segment",
            epoch_origin_years=0.0,
            global_source_step=0,
            source_step_years=0.01,
            time_scale="TDB",
            frame="SYNTHETIC_INERTIAL",
            axes="CARTESIAN_RIGHT_HANDED",
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id="AU-YEAR",
            source_ids=("primary", "secondary"),
            primary_source_id="primary",
            source_schedule_sha256=SCHEDULE_SHA256,
            tracer_relative_positions_AU=cp.asarray(((1.001, 0.0, 0.0),)),
            tracer_relative_velocities_AU_per_year=cp.asarray(
                ((-2.0, 0.0, 0.0),)
            ),
            tracer_indices=cp.asarray((23,), dtype=cp.uint64),
            provenance=_provenance(),
        )
        combined = (
            advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source(
                initialize_restricted_wh_lifecycle(dynamics),
                tile_id="synthetic-combined-lifecycle-source-tile",
                initial_source_relative_positions_AU=positions,
                initial_source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=gm,
                source_encounter_radii_AU=terminal_radii,
                certified_relative_acceleration_bounds_AU_per_year2=bounds,
                source_provenance=_provenance(),
                source_spec=WisdomHolmanSourceTileSpec(0.01, 1),
                tracer_spec=FixedStepCUDARestrictedWHSpec(
                    1,
                    1,
                    BackendSpec("cupy", "cuda:0", 128),
                    maximum_primary_dynamical_phase=0.1,
                ),
                route_worker_count=1,
            )
        )
        self.assertEqual(
            combined.tracer_result.source_tile_sha256,
            combined.source_build.tile.content_sha256,
        )
        self.assertEqual(
            combined.tracer_result.state.tracer_lifecycle.tolist(),
            [REMOVED_SOURCE_RADIUS],
        )
        self.assertEqual(len(combined.tracer_result.new_removal_records), 1)
        self.assertFalse(any(combined.claims.values()))


if __name__ == "__main__":
    unittest.main()
