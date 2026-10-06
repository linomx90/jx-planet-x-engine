from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.long_horizon.contracts import CLAIM_CEILING
from jxplanetx.long_horizon.runner import (
    EXECUTION_MODE,
    FixedLongHorizonRunConfig,
    FixedLongHorizonSourceSegment,
    LongHorizonRunnerContractError,
    advance_fixed_lifecycle_segment,
    resume_fixed_lifecycle_run,
    start_fixed_lifecycle_run,
)
from jxplanetx.restricted_wh_checkpoint import (
    RestrictedWHCheckpointIntegrityError,
    restricted_wh_lifecycle_state_sha256,
)
from jxplanetx.restricted_wh_lifecycle import (
    initialize_restricted_wh_lifecycle,
)
from jxplanetx.restricted_wh_source import WisdomHolmanSourceTileSpec


try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU-only CI
    cp = None


MU = 4.0 * math.pi * math.pi
SCHEDULE_SHA256 = hashlib.sha256(b"long-horizon-runner-schedule").hexdigest()


def _provenance() -> Provenance:
    return Provenance(
        "jx.test.long-horizon-runner",
        "Synthetic coordinator contract fixture",
        "v1",
        hashlib.sha256(b"long-horizon-runner-fixture").hexdigest(),
    )


def _source_arrays():
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (5.2, 0.0, 0.0)), dtype=np.float64
    )
    velocities = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, math.sqrt((MU + 1.0e-3 * MU) / 5.2), 0.0),
        ),
        dtype=np.float64,
    )
    gm = np.asarray((MU, 1.0e-3 * MU), dtype=np.float64)
    radii = np.asarray((1.0e-4, 1.0e-4), dtype=np.float64)
    bounds = np.asarray((100.0, 100.0), dtype=np.float64)
    return positions, velocities, gm, radii, bounds


def _source(
    tile_id: str,
    *,
    positions=None,
    velocities=None,
    global_source_step: int = 0,
    frame: str = "SYNTHETIC_INERTIAL",
    source_ids: tuple[str, ...] = ("primary", "secondary"),
    source_schedule_sha256: str = SCHEDULE_SHA256,
):
    base_positions, base_velocities, gm, radii, bounds = _source_arrays()
    return FixedLongHorizonSourceSegment(
        tile_id=tile_id,
        epoch_origin_years=0.0,
        global_source_step=global_source_step,
        source_step_years=0.01,
        time_scale="TDB",
        frame=frame,
        axes="CARTESIAN_RIGHT_HANDED",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=source_ids,
        primary_source_id=source_ids[0],
        source_schedule_sha256=source_schedule_sha256,
        initial_source_relative_positions_AU=(
            base_positions if positions is None else positions
        ),
        initial_source_relative_velocities_AU_per_year=(
            base_velocities if velocities is None else velocities
        ),
        source_gm_AU3_per_year2=gm,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=bounds,
        source_provenance=_provenance(),
    )


def _config(run_id: str = "long-horizon-runner-test"):
    return FixedLongHorizonRunConfig(
        run_id=run_id,
        source_spec=WisdomHolmanSourceTileSpec(0.01, 1),
        tracer_spec=FixedStepCUDARestrictedWHSpec(
            1,
            1,
            BackendSpec("cupy", "cuda:0", 128),
        ),
        route_worker_count=1,
    )


class LongHorizonRunnerContractTests(unittest.TestCase):
    def test_config_is_content_bound_and_rejects_unimplemented_modes(self):
        first = _config()
        second = _config()
        self.assertEqual(first.execution_mode, EXECUTION_MODE)
        self.assertEqual(first.protocol_sha256, second.protocol_sha256)
        self.assertEqual(len(first.protocol_sha256), 64)
        self.assertFalse(any(first.claims.values()))
        self.assertEqual(dict(first.claims), dict(CLAIM_CEILING))
        with self.assertRaises(TypeError):
            first.claims["science_result"] = True
        with self.assertRaisesRegex(
            LongHorizonRunnerContractError, "multirate scheduling"
        ):
            replace(first, multirate_enabled=True)
        with self.assertRaisesRegex(
            LongHorizonRunnerContractError, "owned-episode results"
        ):
            replace(first, owned_episode_enabled=True)
        with self.assertRaises(LongHorizonRunnerContractError):
            replace(first, route_worker_count=True)
        self.assertNotEqual(first.protocol_sha256, _config("other-run").protocol_sha256)

    def test_source_material_is_owned_immutable_and_content_bound(self):
        positions, velocities, _, _, _ = _source_arrays()
        source = _source("runner-source", positions=positions, velocities=velocities)
        digest = source.content_sha256
        positions[1, 0] = 9.0
        velocities[1, 1] = 8.0
        self.assertEqual(source.initial_source_relative_positions_AU[1, 0], 5.2)
        self.assertNotEqual(
            source.initial_source_relative_velocities_AU_per_year[1, 1], 8.0
        )
        self.assertFalse(source.initial_source_relative_positions_AU.flags.writeable)
        with self.assertRaises(ValueError):
            source.initial_source_relative_positions_AU.setflags(write=True)
        self.assertEqual(digest, _source("runner-source").content_sha256)
        self.assertNotEqual(digest, _source("runner-source-other").content_sha256)
        self.assertNotEqual(
            digest,
            replace(source, frame="OTHER_INERTIAL").content_sha256,
        )
        self.assertNotEqual(
            digest,
            replace(
                source,
                source_schedule_sha256=hashlib.sha256(
                    b"different-source-schedule"
                ).hexdigest(),
            ).content_sha256,
        )

    def test_contract_rejects_source_shape_and_guard_policy_ambiguity(self):
        positions, velocities, gm, radii, bounds = _source_arrays()
        with self.assertRaises(LongHorizonRunnerContractError):
            FixedLongHorizonSourceSegment(
                tile_id="bad-source",
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
                initial_source_relative_positions_AU=positions[:, :2].copy(),
                initial_source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=gm,
                source_encounter_radii_AU=radii,
                certified_relative_acceleration_bounds_AU_per_year2=bounds,
                source_provenance=_provenance(),
            )
        config = _config()
        with self.assertRaises(LongHorizonRunnerContractError):
            replace(
                config,
                source_spec=WisdomHolmanSourceTileSpec(0.01, 2),
            )


@unittest.skipIf(cp is None, "CUDA device is required")
class LongHorizonRunnerLiveGPUTests(unittest.TestCase):
    def _state(self):
        dynamics = CUDARestrictedWHState(
            snapshot_id="long-horizon-runner-genesis",
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
            tracer_relative_positions_AU=cp.asarray(
                ((10.0, 0.0, 0.0),), dtype=cp.float64
            ),
            tracer_relative_velocities_AU_per_year=cp.asarray(
                ((0.0, math.sqrt(MU / 10.0), 0.0),), dtype=cp.float64
            ),
            tracer_indices=cp.asarray((101,), dtype=cp.uint64),
            provenance=_provenance(),
        )
        return initialize_restricted_wh_lifecycle(dynamics)

    @staticmethod
    def _successor_source(prepared):
        tile = prepared.authoritative_result.source_build.tile
        return _source(
            "runner-source-0001",
            positions=np.ascontiguousarray(
                tile.source_relative_positions_AU[-1]
            ),
            velocities=np.ascontiguousarray(
                tile.source_relative_velocities_AU_per_year[-1]
            ),
            global_source_step=(
                prepared.authoritative_result.tracer_result.state.dynamics_state.global_source_step
            ),
        )

    def test_uninterrupted_and_checkpoint_resume_are_bitwise_identical(self):
        config = _config()
        genesis = self._state()
        source0 = _source("runner-source-0000")

        direct0 = advance_fixed_lifecycle_segment(
            genesis, config=config, source=source0
        )
        source1 = self._successor_source(direct0)
        direct1 = advance_fixed_lifecycle_segment(
            direct0.authoritative_result.tracer_result.state,
            config=config,
            source=source1,
        )

        with tempfile.TemporaryDirectory() as directory:
            committed0 = start_fixed_lifecycle_run(
                genesis,
                config=config,
                source=source0,
                checkpoint_directory=directory,
            )
            resumed = resume_fixed_lifecycle_run(
                config=config,
                source=self._successor_source(committed0.prepared),
                checkpoint_directory=directory,
                genesis_state_sha256=committed0.checkpoint_record[
                    "genesis_state_sha256"
                ],
                expected_checkpoint_index=0,
                expected_checkpoint_content_sha256=(
                    committed0.loaded_checkpoint.checkpoint_content_sha256
                ),
            )

            self.assertEqual(resumed.loaded_checkpoint.checkpoint_index, 1)
            self.assertEqual(
                resumed.prepared.output_state_sha256,
                direct1.output_state_sha256,
            )
            self.assertEqual(
                resumed.prepared.result_contract.content_sha256,
                direct1.result_contract.content_sha256,
            )
            direct_state = direct1.authoritative_result.tracer_result.state
            resumed_state = resumed.loaded_checkpoint.state
            self.assertTrue(
                bool(
                    cp.array_equal(
                        direct_state.dynamics_state.tracer_relative_positions_AU,
                        resumed_state.dynamics_state.tracer_relative_positions_AU,
                    )
                )
            )
            self.assertTrue(
                bool(
                    cp.array_equal(
                        direct_state.dynamics_state.tracer_relative_velocities_AU_per_year,
                        resumed_state.dynamics_state.tracer_relative_velocities_AU_per_year,
                    )
                )
            )
            self.assertEqual(
                direct_state.tracer_lifecycle.tolist(),
                resumed_state.tracer_lifecycle.tolist(),
            )
            self.assertFalse(any(committed0.claims.values()))
            self.assertFalse(any(resumed.prepared.result_contract.claims.values()))
            self.assertFalse(any(resumed.checkpoint_contract.claims.values()))
            diagnostics = resumed.prepared.result_contract.diagnostics
            self.assertIn("global_minimum_source_distance_AU", diagnostics)
            self.assertEqual(diagnostics["minimum_source_distance_lane_count"], 1)
            self.assertEqual(
                diagnostics["minimum_source_distance_finite_lane_count"], 1
            )
            self.assertEqual(
                len(diagnostics["minimum_source_distance_vector_sha256"]), 64
            )

    def test_source_context_mismatch_fails_before_numerical_execution(self):
        config = _config("long-horizon-runner-context-test")
        genesis = self._state()
        source = _source("runner-context-source")
        for changed in (
            replace(source, frame="OTHER_INERTIAL"),
            replace(source, global_source_step=1),
            replace(source, source_ids=("primary", "different-secondary")),
            replace(
                source,
                source_schedule_sha256=hashlib.sha256(
                    b"different-source-schedule"
                ).hexdigest(),
            ),
        ):
            with self.assertRaisesRegex(
                LongHorizonRunnerContractError,
                "source segment context",
            ):
                advance_fixed_lifecycle_segment(
                    genesis,
                    config=config,
                    source=changed,
                )

    def test_wrong_external_anchor_fails_before_appending(self):
        config = _config("long-horizon-runner-anchor-test")
        genesis = self._state()
        with tempfile.TemporaryDirectory() as directory:
            committed = start_fixed_lifecycle_run(
                genesis,
                config=config,
                source=_source("runner-anchor-source-0000"),
                checkpoint_directory=directory,
            )
            genesis_digest = restricted_wh_lifecycle_state_sha256(
                genesis, config.tracer_spec.backend
            )
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                resume_fixed_lifecycle_run(
                    config=config,
                    source=self._successor_source(committed.prepared),
                    checkpoint_directory=directory,
                    genesis_state_sha256=genesis_digest,
                    expected_checkpoint_index=0,
                    expected_checkpoint_content_sha256=(
                        hashlib.sha256(b"wrong-anchor").hexdigest()
                    ),
                )
            records = tuple(Path(directory).rglob("checkpoint_*.json"))
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
