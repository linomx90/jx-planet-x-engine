from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import jxplanetx.long_horizon.m2_source_path as retained
import jxplanetx.long_horizon as facade
from jxplanetx.cuda_restricted_wh import FrozenSourcePathTile, verify_source_path_tile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_RUN = PROJECT_ROOT / retained.OFFICIAL_RUN_RELATIVE_PATH


class RetainedM2SourcePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = retained.load_verified_m2_source_path(OFFICIAL_RUN)

    def test_exact_completed_v2_path_is_loaded_and_frozen(self) -> None:
        source = self.source
        self.assertEqual(source.node_count, 81_921)
        self.assertEqual(source.source_relative_positions_AU.shape, (81_921, 10, 3))
        self.assertEqual(
            source.source_relative_velocities_AU_per_year.shape, (81_921, 10, 3)
        )
        self.assertEqual(source.source_gm_AU3_per_year2.shape, (10,))
        self.assertEqual(source.source_step_years.hex(), "0x1.0000000000000p-13")
        self.assertEqual(source.source_schedule_sha256, retained.FINE_SOURCE_SCHEDULE_SHA256)
        self.assertEqual(source.source_body_ids, (10, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(source.source_ids[0], "Sun")
        self.assertEqual(source.source_ids[-1], "Pluto Barycenter")
        self.assertEqual(source.epoch_jd_tdb, 2_461_200.5)
        self.assertEqual(source.frame, "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000")
        self.assertEqual(source.axes, "IAU76_80_ECLIPTIC_J2000")
        self.assertEqual(source.time_scale, "TDB")
        self.assertEqual(source.unit_system_id, "AU-YEAR")
        self.assertEqual(
            source.parent_file_sha256["RESULT.json"], retained.RESULT_FILE_SHA256
        )
        self.assertEqual(
            source.parent_content_sha256["COMPLETED.json"],
            retained.COMPLETED_CONTENT_SHA256,
        )
        self.assertEqual(len(source.array_content_sha256), 19)
        self.assertEqual(
            source.array_content_sha256["jx_fine_all_node_position_AU"],
            "77dfee267cf6d56c0f251d7b81440b80af5a8c040d2b348c80448739efca4d55",
        )
        self.assertEqual(
            source.array_content_sha256["jx_fine_all_node_velocity_AU_per_year"],
            "e223a95d5ffc8a5347054923fde2c510f569b0048c94714c25991c971d5fa2ae",
        )
        self.assertEqual(
            source.dynamic_node_sha256,
            "86293bb87aba3f3219aaa3d90263a07c271496f30ee0ee8a997428dfc2ce8487",
        )
        self.assertEqual(
            source.content_sha256,
            "c332e9ff937f0aa7157a095e196d3a30adde5f26998b16a75c0740f3d836945d",
        )
        self.assertEqual(retained.verify_m2_source_path(source), source.content_sha256)
        for array in (
            source.source_relative_positions_AU,
            source.source_relative_velocities_AU_per_year,
            source.source_gm_AU3_per_year2,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flags.writeable = True
        with self.assertRaises(TypeError):
            source.parent_file_sha256["RESULT.json"] = "0" * 64

    def test_public_api_is_the_single_experimental_facade(self) -> None:
        for name in (
            "VerifiedM2SourcePath",
            "VerifiedM2SourcePathSlice",
            "M2GuardOverlay",
            "M2FrozenSourcePathTileBinding",
            "load_verified_m2_source_path",
            "seal_m2_guard_overlay",
            "build_m2_frozen_source_path_tile",
            "verify_m2_frozen_source_path_tile_binding",
        ):
            self.assertIn(name, facade.__all__)
            self.assertIs(getattr(facade, name), getattr(retained, name))

    def test_exact_one_year_and_five_to_ten_year_slices(self) -> None:
        one_year = self.source.slice_years(0, 1)
        late = self.source.slice_years(5, 10)
        self.assertEqual(one_year.global_source_step, 0)
        self.assertEqual(one_year.final_global_source_step, 8192)
        self.assertEqual(one_year.node_count, 8193)
        self.assertEqual(one_year.initial_epoch_years, 0.0)
        self.assertEqual(one_year.final_epoch_years, 1.0)
        self.assertEqual(one_year.final_epoch_jd_tdb, 2_461_565.75)
        self.assertEqual(
            one_year.dynamic_node_sha256,
            "7b5f2d6525a05231a88235f8d0c49fee7a18bc8212625e569509a4126cea1472",
        )
        self.assertEqual(late.global_source_step, 40_960)
        self.assertEqual(late.final_global_source_step, 81_920)
        self.assertEqual(late.node_count, 40_961)
        self.assertEqual(late.initial_epoch_years, 5.0)
        self.assertEqual(late.final_epoch_years, 10.0)
        self.assertEqual(
            late.dynamic_node_sha256,
            "ec3120f970f3d6a6e71a98abc1dd75c6c60af0c4a434151fd349d07ee6811818",
        )
        np.testing.assert_array_equal(
            one_year.source_relative_positions_AU,
            self.source.source_relative_positions_AU[:8193],
        )
        np.testing.assert_array_equal(
            late.source_relative_velocities_AU_per_year,
            self.source.source_relative_velocities_AU_per_year[40_960:],
        )
        self.assertEqual(
            retained.verify_m2_source_path_slice(one_year), one_year.content_sha256
        )
        self.assertFalse(one_year.source_relative_positions_AU.flags.writeable)

    def test_adjacent_slices_share_one_exact_boundary_and_reconstruct_parent(self) -> None:
        first = self.source.slice_years(0, 5)
        second = self.source.slice_years(5, 10)
        np.testing.assert_array_equal(
            first.source_relative_positions_AU[-1],
            second.source_relative_positions_AU[0],
        )
        np.testing.assert_array_equal(
            first.source_relative_velocities_AU_per_year[-1],
            second.source_relative_velocities_AU_per_year[0],
        )
        joined_position = np.concatenate(
            (first.source_relative_positions_AU[:-1], second.source_relative_positions_AU),
            axis=0,
        )
        joined_velocity = np.concatenate(
            (
                first.source_relative_velocities_AU_per_year[:-1],
                second.source_relative_velocities_AU_per_year,
            ),
            axis=0,
        )
        np.testing.assert_array_equal(joined_position, self.source.source_relative_positions_AU)
        np.testing.assert_array_equal(
            joined_velocity, self.source.source_relative_velocities_AU_per_year
        )

    def test_guard_overlay_and_engine_tile_keep_dynamic_identity_separate(self) -> None:
        source_slice = self.source.slice_years(0, 1)
        minimum = retained.derive_m2_uniform_relative_acceleration_bound(source_slice)
        self.assertEqual(minimum.hex(), "0x1.ee1fac800e17ap+12")
        radii = np.full(10, 0.1, dtype=np.float64)
        exact_bounds = np.full(10, minimum, dtype=np.float64)
        wider_bounds = np.full(10, math.nextafter(minimum, math.inf), dtype=np.float64)
        exact_overlay = retained.seal_m2_guard_overlay(
            source_slice,
            source_encounter_radii_AU=radii,
            certified_relative_acceleration_bounds_AU_per_year2=exact_bounds,
        )
        wider_overlay = retained.seal_m2_guard_overlay(
            source_slice,
            source_encounter_radii_AU=radii,
            certified_relative_acceleration_bounds_AU_per_year2=wider_bounds,
        )
        exact_binding = retained.build_m2_frozen_source_path_tile(
            source_slice, exact_overlay
        )
        wider_binding = source_slice.bind_frozen_source_path_tile(wider_overlay)
        self.assertIs(type(exact_binding.tile), FrozenSourcePathTile)
        self.assertEqual(exact_binding.dynamic_node_sha256, source_slice.dynamic_node_sha256)
        self.assertEqual(wider_binding.dynamic_node_sha256, source_slice.dynamic_node_sha256)
        self.assertNotEqual(
            exact_binding.guard_overlay_sha256, wider_binding.guard_overlay_sha256
        )
        self.assertNotEqual(exact_binding.tile_content_sha256, wider_binding.tile_content_sha256)
        self.assertEqual(
            verify_source_path_tile(exact_binding.tile), exact_binding.tile_content_sha256
        )
        self.assertEqual(exact_binding.tile.global_source_step, 0)
        self.assertEqual(exact_binding.tile.final_global_source_step, 8192)
        self.assertEqual(
            exact_binding.tile.source_schedule_sha256,
            retained.FINE_SOURCE_SCHEDULE_SHA256,
        )
        self.assertEqual(exact_binding.tile.provenance.sha256, exact_binding.lineage_content_sha256)
        np.testing.assert_array_equal(
            exact_binding.tile.source_relative_positions_AU,
            source_slice.source_relative_positions_AU,
        )
        self.assertIs(type(source_slice.to_frozen_source_path_tile(exact_overlay)), FrozenSourcePathTile)
        self.assertEqual(
            retained.verify_m2_frozen_source_path_tile_binding(exact_binding),
            exact_binding.content_sha256,
        )

    def test_small_execution_slices_do_not_rehash_the_full_parent(self) -> None:
        minimum = retained.derive_m2_uniform_relative_acceleration_bound(self.source)
        overlay = retained.seal_m2_guard_overlay(
            self.source,
            source_encounter_radii_AU=np.full(10, 0.1, dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                10, minimum, dtype=np.float64
            ),
        )
        started = time.perf_counter()
        with mock.patch.object(
            retained,
            "verify_m2_source_path",
            side_effect=AssertionError("per-map path rehashed the complete parent"),
        ):
            bindings = tuple(
                retained.build_m2_frozen_source_path_tile(
                    self.source.slice_steps(index * 8, (index + 1) * 8),
                    overlay,
                )
                for index in range(128)
            )
        elapsed = time.perf_counter() - started
        self.assertEqual(bindings[-1].tile.final_global_source_step, 1024)
        # The former full-parent hashing path takes tens of seconds for this
        # bounded batch; local immutable-witness validation is sub-second on
        # the qualification host and retains generous CI headroom here.
        self.assertLess(elapsed, 5.0)
        self.assertEqual(
            retained.verify_m2_source_path(self.source), self.source.content_sha256
        )

    def test_fast_witness_rejects_reflective_read_only_array_replacement(self) -> None:
        original = self.source.source_relative_positions_AU
        changed = np.array(original, copy=True)
        changed[1, 1, 0] = np.nextafter(changed[1, 1, 0], math.inf)
        forged = np.frombuffer(changed.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
            changed.shape
        )
        self.assertFalse(forged.flags.writeable)
        object.__setattr__(self.source, "source_relative_positions_AU", forged)
        try:
            with self.assertRaisesRegex(
                retained.M2SourcePathVerificationError,
                "fully hashed session witness",
            ):
                self.source.slice_steps(0, 8)
        finally:
            object.__setattr__(
                self.source, "source_relative_positions_AU", original
            )
        self.assertEqual(
            retained.verify_m2_source_path(self.source), self.source.content_sha256
        )

    def test_guard_and_tile_receipts_reject_direct_constructor_forgery(self) -> None:
        source_slice = self.source.slice_years(0, 1)
        radii = np.full(10, 0.1, dtype=np.float64)
        tiny_bounds = np.full(10, 1.0e-300, dtype=np.float64)
        gm_hash = source_slice.gm_array_content_sha256
        forged_overlay_hash = retained._guard_overlay_sha256(
            radii, tiny_bounds, gm_hash, 0.0
        )
        with self.assertRaisesRegex(
            retained.M2SourcePathVerificationError, "derived uniform bound"
        ):
            retained.M2GuardOverlay(
                source_encounter_radii_AU=radii,
                certified_relative_acceleration_bounds_AU_per_year2=tiny_bounds,
                gm_array_content_sha256=gm_hash,
                minimum_derived_uniform_bound_AU_per_year2=0.0,
                content_sha256=forged_overlay_hash,
            )

        minimum = retained.derive_m2_uniform_relative_acceleration_bound(
            source_slice
        )
        overlay = retained.seal_m2_guard_overlay(
            source_slice,
            source_encounter_radii_AU=radii,
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                10, minimum, dtype=np.float64
            ),
        )
        valid = retained.build_m2_frozen_source_path_tile(source_slice, overlay)
        fake_dynamic = "1" * 64
        fake_guard = "2" * 64
        fake_lineage = "3" * 64
        fake_content = retained._record_sha256(
            retained._TILE_BINDING_DOMAIN,
            {
                "schema": "jxplanetx.long-horizon.m2-frozen-tile-binding.v1",
                "dynamic_node_sha256": fake_dynamic,
                "guard_overlay_sha256": fake_guard,
                "lineage_content_sha256": fake_lineage,
                "tile_content_sha256": valid.tile_content_sha256,
            },
        )
        with self.assertRaisesRegex(
            retained.M2SourcePathVerificationError, "verified witnesses"
        ):
            retained.M2FrozenSourcePathTileBinding(
                source_slice=source_slice,
                guard_overlay=overlay,
                tile=valid.tile,
                dynamic_node_sha256=fake_dynamic,
                guard_overlay_sha256=fake_guard,
                lineage_content_sha256=fake_lineage,
                tile_content_sha256=valid.tile_content_sha256,
                content_sha256=fake_content,
            )

    def test_slice_and_guard_inputs_fail_closed(self) -> None:
        with self.assertRaises(retained.M2SourcePathVerificationError):
            self.source.slice_steps(-1, 1)
        with self.assertRaises(retained.M2SourcePathVerificationError):
            self.source.slice_steps(0, 0)
        with self.assertRaises(retained.M2SourcePathVerificationError):
            self.source.slice_steps(0, 81_921)
        with self.assertRaises(retained.M2SourcePathVerificationError):
            self.source.slice_years(0.0, 1)
        source_slice = self.source.slice_years(0, 1)
        minimum = retained.derive_m2_uniform_relative_acceleration_bound(source_slice)
        with self.assertRaisesRegex(
            retained.M2SourcePathVerificationError, "exact 0.1 AU"
        ):
            retained.seal_m2_guard_overlay(
                source_slice,
                source_encounter_radii_AU=np.full(10, 0.10000000000000002),
                certified_relative_acceleration_bounds_AU_per_year2=np.full(10, minimum),
            )
        with self.assertRaisesRegex(
            retained.M2SourcePathVerificationError, "undercuts"
        ):
            retained.seal_m2_guard_overlay(
                source_slice,
                source_encounter_radii_AU=np.full(10, 0.1),
                certified_relative_acceleration_bounds_AU_per_year2=np.full(
                    10, math.nextafter(minimum, 0.0)
                ),
            )


class RetainedM2RejectionTests(unittest.TestCase):
    def test_stale_v1_and_wrong_path_names_are_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "long_horizon_m2_source_preflight_v1"
            wrong.mkdir()
            with self.assertRaisesRegex(
                retained.M2SourcePathVerificationError, "only.*v2"
            ):
                retained.load_verified_m2_source_path(wrong)

    def test_symlink_run_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / retained.OFFICIAL_RUN_DIRECTORY_NAME
            os.symlink(OFFICIAL_RUN, link)
            with self.assertRaisesRegex(
                retained.M2SourcePathVerificationError, "symlink"
            ):
                retained.load_verified_m2_source_path(link)

    def test_ancestor_swap_after_open_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            switch = temporary_path / "switch"
            copied = switch / retained.OFFICIAL_RUN_DIRECTORY_NAME
            copied.mkdir(parents=True)
            for source in OFFICIAL_RUN.iterdir():
                destination = copied / source.name
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            original_reader = retained._read_regular_file_at
            swapped = False

            def swap_then_read(
                directory_descriptor: int,
                name: str,
                *,
                expected_size: int,
            ) -> bytes:
                nonlocal swapped
                if not swapped:
                    swapped = True
                    switch.rename(temporary_path / "switch-original")
                    os.symlink(PROJECT_ROOT / "runs", switch)
                return original_reader(
                    directory_descriptor,
                    name,
                    expected_size=expected_size,
                )

            with mock.patch.object(
                retained, "_read_regular_file_at", swap_then_read
            ):
                with self.assertRaisesRegex(
                    retained.M2SourcePathVerificationError, "symlink"
                ):
                    retained.load_verified_m2_source_path(copied)

    def test_tampered_states_file_is_rejected_without_touching_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / retained.OFFICIAL_RUN_DIRECTORY_NAME
            shutil.copytree(OFFICIAL_RUN, copied)
            state_path = copied / "STATES.npz"
            payload = bytearray(state_path.read_bytes())
            payload[-1] ^= 1
            state_path.write_bytes(payload)
            with self.assertRaisesRegex(
                retained.M2SourcePathVerificationError, "STATES.npz file SHA-256"
            ):
                retained.load_verified_m2_source_path(copied)
        self.assertEqual(
            hashlib.sha256((OFFICIAL_RUN / "STATES.npz").read_bytes()).hexdigest(),
            retained.STATES_FILE_SHA256,
        )

    def test_duplicate_json_and_npz_rosters_and_stale_schema_are_rejected(self) -> None:
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "duplicate key"):
            retained._load_json(b'{"schema":"a","schema":"b"}', "test.json")
        names = tuple(retained._NPZ_MEMBERS) + (retained._NPZ_MEMBERS[0],)
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "duplicate"):
            retained._validate_npz_member_roster(names)
        stale = {
            "schema": "jx.long-horizon.m2-known-source-protocol.v1",
            "run_id": retained.RUN_ID,
            "content_sha256": retained.PROTOCOL_CONTENT_SHA256,
        }
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "schema"):
            retained._validate_content_record(
                stale,
                schema=retained.PROTOCOL_SCHEMA,
                domain=retained._PROTOCOL_DOMAIN,
                expected_content_sha256=retained.PROTOCOL_CONTENT_SHA256,
                label="test protocol",
            )

    def test_dtype_shape_nonfinite_and_boundary_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "dtype"):
            retained._validate_array_contract(
                "sample_years", np.arange(11, dtype=np.float32)
            )
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "shape"):
            retained._validate_array_contract(
                "sample_years", np.arange(10, dtype=np.float64)
            )
        nonfinite = np.arange(11, dtype=np.float64)
        nonfinite[5] = np.nan
        with self.assertRaisesRegex(retained.M2SourcePathVerificationError, "nonfinite"):
            retained._validate_array_contract("sample_years", nonfinite)

        source = retained.load_verified_m2_source_path(OFFICIAL_RUN)
        fine_position = source.source_relative_positions_AU
        fine_velocity = source.source_relative_velocities_AU_per_year
        annual_position = np.ascontiguousarray(fine_position[::8192])
        annual_position[4, 2, 1] = np.nextafter(annual_position[4, 2, 1], math.inf)
        arrays = {
            "sample_years": np.arange(11, dtype=np.float64),
            "epochs_jd_tdb": np.asarray(
                [retained.EPOCH_JD_TDB + i * retained.JULIAN_YEAR_DAYS for i in range(11)],
                dtype=np.float64,
            ),
            "gm_AU3_per_year2": source.source_gm_AU3_per_year2,
            "jx_coarse_all_node_position_AU": np.zeros((40_961, 10, 3)),
            "jx_coarse_all_node_velocity_AU_per_year": np.zeros((40_961, 10, 3)),
            "jx_coarse_position_AU": np.zeros((11, 10, 3)),
            "jx_coarse_velocity_AU_per_year": np.zeros((11, 10, 3)),
            "jx_fine_all_node_position_AU": fine_position,
            "jx_fine_all_node_velocity_AU_per_year": fine_velocity,
            "jx_fine_position_AU": annual_position,
            "jx_fine_velocity_AU_per_year": np.ascontiguousarray(fine_velocity[::8192]),
        }
        with self.assertRaisesRegex(
            retained.M2SourcePathVerificationError, "fine annual boundary"
        ):
            retained._validate_state_boundaries(arrays)


if __name__ == "__main__":
    unittest.main()
