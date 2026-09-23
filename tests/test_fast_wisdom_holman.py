from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import fast_wisdom_holman as fast_wh
from jxplanetx import _wisdom_holman_cpu_v2 as native_v2
from jxplanetx import _wisdom_holman_cpu_v3 as native_v3
from jxplanetx.engine import wisdom_holman as public_wh


class FastWisdomHolmanTests(unittest.TestCase):
    @staticmethod
    def _case(periods: int = 1) -> tuple[object, object, object]:
        profile = fixture.Profile(1, periods, 4)
        snapshot, plan = fixture._state_and_plan(profile)
        spec = fixture._wh_spec(profile, "long", 64)
        return snapshot, plan, spec

    @staticmethod
    def _native_run(
        native: object,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        gm: np.ndarray,
        radii: np.ndarray,
        checkpoint_steps: tuple[int, ...],
        fixed_step: float,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(checkpoint_steps)
        body_count = int(gm.size)
        output_positions = np.empty((count, body_count, 3), dtype=np.float64)
        output_velocities = np.empty((count, body_count, 3), dtype=np.float64)
        metrics = np.empty(len(fast_wh.METRIC_NAMES), dtype=np.float64)
        counters = np.empty(len(fast_wh.COUNTER_NAMES), dtype=np.int64)
        status = native.integrate_map(
            body_count,
            count,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(velocities, dtype=np.float64),
            np.ascontiguousarray(gm, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.asarray(checkpoint_steps, dtype=np.int64),
            fixed_step,
            0.01,
            0.01,
            1.0e-12,
            1.0e-12,
            output_positions,
            output_velocities,
            metrics,
            counters,
        )
        return status, output_positions, output_velocities, metrics, counters

    def _assert_v3_v2_bitwise_parity(
        self,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        gm: np.ndarray,
        radii: np.ndarray,
        checkpoint_steps: tuple[int, ...],
        fixed_step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        arguments = {
            "positions": positions,
            "velocities": velocities,
            "gm": gm,
            "radii": radii,
            "checkpoint_steps": checkpoint_steps,
            "fixed_step": fixed_step,
        }
        reference = self._native_run(native_v2, **arguments)
        candidate = self._native_run(native_v3, **arguments)
        self.assertEqual(reference[0], 0)
        self.assertEqual(candidate[0], 0)
        for reference_array, candidate_array in zip(
            reference[1:4], candidate[1:4], strict=True
        ):
            self.assertTrue(np.array_equal(reference_array, candidate_array))
        reference_counters = dict(
            zip(fast_wh.COUNTER_NAMES, map(int, reference[4]), strict=True)
        )
        candidate_counters = dict(
            zip(fast_wh.COUNTER_NAMES, map(int, candidate[4]), strict=True)
        )
        for name in fast_wh.COUNTER_NAMES:
            if name != "inverse_transforms":
                self.assertEqual(reference_counters[name], candidate_counters[name])
        final_step = checkpoint_steps[-1]
        self.assertEqual(reference_counters["inverse_transforms"], 2 * final_step)
        self.assertEqual(
            candidate_counters["inverse_transforms"],
            final_step + len(checkpoint_steps) - 1,
        )
        return reference[3], candidate[4]

    def test_runtime_identity_retains_exact_fallback_and_claim_ceiling(self) -> None:
        identity = fast_wh.fast_wisdom_holman_runtime_identity()
        self.assertEqual(identity["kernel_id"], fast_wh.KERNEL_ID)
        self.assertEqual(
            identity["exact_fallback_kernel_id"],
            fast_wh.EXACT_FALLBACK_KERNEL_ID,
        )
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")
        self.assertRegex(
            identity["release_loop_source_sha256"], r"^[0-9a-f]{64}$"
        )

    def test_default_certificate_matches_public_trajectory_bitwise(self) -> None:
        snapshot, plan, spec = self._case()
        expected = public_wh.integrate_wisdom_holman_trajectory(
            snapshot, plan, spec
        )
        candidate = fast_wh.integrate_fast_wisdom_holman_trajectory(
            snapshot, plan, spec
        )
        self.assertTrue(
            np.array_equal(candidate.positions, np.stack(expected.positions))
        )
        self.assertTrue(
            np.array_equal(candidate.velocities, np.stack(expected.velocities))
        )
        self.assertTrue(
            np.array_equal(
                candidate.checkpoint_epochs,
                np.asarray(expected.checkpoint_epochs, dtype=np.float64),
            )
        )
        self.assertEqual(candidate.api_id, fast_wh.API_ID)
        self.assertEqual(candidate.replay_count, 0)
        self.assertEqual(
            candidate.verification_mode,
            fast_wh.CERTIFICATE_VERIFICATION_MODE,
        )
        self.assertEqual(candidate.certified_steps, 64)
        self.assertEqual(candidate.certified_kepler_solves, 128)
        self.assertTrue(candidate.full_step_postcondition_coverage)
        self.assertFalse(candidate.exact_replay_verified)
        self.assertFalse(candidate.independent_numerical_method_verified)
        self.assertFalse(candidate.deterministic_implementation_defect_excluded)
        self.assertEqual(candidate.primary_counters, candidate.total_counters)
        self.assertTrue(
            all(
                not value.flags.writeable
                for value in (
                    candidate.positions,
                    candidate.velocities,
                    candidate.checkpoint_epochs,
                )
            )
        )

    def test_long_default_certifies_every_step_with_one_execution(self) -> None:
        snapshot, plan, spec = self._case(periods=100)
        execute_once = fast_wh._execute_once
        with mock.patch.object(
            fast_wh, "_execute_once", wraps=execute_once
        ) as wrapped:
            result = fast_wh.integrate_fast_wisdom_holman_trajectory(
                snapshot, plan, spec
            )
        self.assertEqual(wrapped.call_count, 1)
        self.assertEqual(result.certified_steps, 6400)
        self.assertEqual(result.certified_kepler_solves, 12800)
        self.assertTrue(result.full_step_postcondition_coverage)

    def test_v3_matches_v2_for_irregular_checkpoint_schedule(self) -> None:
        snapshot, plan, spec = self._case()
        irregular = replace(
            spec,
            checkpoint_step_indices=(0, 1, 2, 7, 19, 31, 64),
        )
        workspace = fast_wh.prepare_fast_wisdom_holman_workspace(
            snapshot, plan, irregular
        )
        self._assert_v3_v2_bitwise_parity(
            positions=workspace.positions,
            velocities=workspace.velocities,
            gm=workspace.gravitational_parameters,
            radii=workspace.radii,
            checkpoint_steps=tuple(map(int, workspace.checkpoint_steps)),
            fixed_step=workspace.fixed_step,
        )

    def test_v3_matches_v2_for_backward_irregular_schedule(self) -> None:
        snapshot, plan, spec = self._case()
        self._assert_v3_v2_bitwise_parity(
            positions=snapshot.positions,
            velocities=snapshot.velocities,
            gm=snapshot.gravitational_parameters,
            radii=snapshot.radii,
            checkpoint_steps=(0, 3, 11, 37, 64),
            fixed_step=-spec.fixed_step,
        )

    def test_v3_two_body_hill_sentinel_and_v2_parity(self) -> None:
        gm = np.asarray((1.0, 0.001), dtype=np.float64)
        total = float(np.sum(gm))
        speed = float(np.sqrt(total))
        positions = np.asarray(
            ((-gm[1] / total, 0.0, 0.0), (gm[0] / total, 0.0, 0.0)),
            dtype=np.float64,
        )
        velocities = np.asarray(
            (
                (0.0, -(gm[1] / total) * speed, 0.0),
                (0.0, (gm[0] / total) * speed, 0.0),
            ),
            dtype=np.float64,
        )
        metrics, _ = self._assert_v3_v2_bitwise_parity(
            positions=positions,
            velocities=velocities,
            gm=gm,
            radii=np.zeros(2, dtype=np.float64),
            checkpoint_steps=(0, 1, 7, 64),
            fixed_step=(2.0 * np.pi / np.sqrt(total)) / 64.0,
        )
        hill_index = fast_wh.METRIC_NAMES.index(
            "minimum_secondary_hill_floor_ratio"
        )
        self.assertEqual(float(metrics[hill_index]), 0.0)

    def test_complete_replay_remains_explicitly_available(self) -> None:
        snapshot, plan, spec = self._case()
        result = fast_wh.integrate_fast_wisdom_holman_trajectory(
            snapshot, plan, spec, exact_replay=True
        )
        self.assertEqual(result.replay_count, 1)
        self.assertEqual(
            result.verification_mode,
            fast_wh.CERTIFICATE_WITH_EXACT_REPLAY_MODE,
        )
        self.assertTrue(result.exact_replay_verified)
        self.assertEqual(result.replay_counters, result.primary_counters)

    def test_explicit_false_uses_the_same_default_certificate(self) -> None:
        snapshot, plan, spec = self._case()
        result = fast_wh.integrate_fast_wisdom_holman_trajectory(
            snapshot, plan, spec, exact_replay=False
        )
        self.assertEqual(result.replay_count, 0)
        self.assertEqual(
            result.verification_mode,
            fast_wh.CERTIFICATE_VERIFICATION_MODE,
        )
        self.assertTrue(result.full_step_postcondition_coverage)
        self.assertFalse(result.exact_replay_verified)

    def test_certificate_is_deterministic_and_fails_closed_on_mutation(self) -> None:
        snapshot, plan, spec = self._case()
        workspace = fast_wh.prepare_fast_wisdom_holman_workspace(
            snapshot, plan, spec
        )
        first = fast_wh.integrate_fast_wisdom_holman_workspace(workspace)
        second = fast_wh.integrate_fast_wisdom_holman_workspace(workspace)
        self.assertEqual(
            first.verification_content_sha256,
            second.verification_content_sha256,
        )
        with self.assertRaisesRegex(
            fast_wh.FastWisdomHolmanLoopError, "certificate digest"
        ):
            replace(first, verification_content_sha256="0" * 64)

        metric_map = dict(first.metrics)
        metric_map["maximum_kepler_time_residual"] = (
            2.0 * metric_map["maximum_kepler_residual_tolerance"]
        )
        changed_metrics = tuple(
            (name, metric_map[name]) for name in fast_wh.METRIC_NAMES
        )
        with self.assertRaisesRegex(
            fast_wh.FastWisdomHolmanLoopError,
            "Kepler residual certificate failed",
        ):
            replace(first, metrics=changed_metrics)

    def test_workspace_custody_and_result_claims_fail_closed(self) -> None:
        snapshot, plan, spec = self._case()
        workspace = fast_wh.prepare_fast_wisdom_holman_workspace(
            snapshot, plan, spec
        )
        writable_positions = workspace.positions.copy()
        changed = replace(workspace, positions=writable_positions)
        with self.assertRaisesRegex(
            fast_wh.FastWisdomHolmanLoopError, "custody"
        ):
            fast_wh.integrate_fast_wisdom_holman_workspace(changed)

        result = fast_wh.integrate_fast_wisdom_holman_workspace(
            workspace, exact_replay=True
        )
        with self.assertRaisesRegex(
            fast_wh.FastWisdomHolmanLoopError, "claim ceiling"
        ):
            replace(result, production_authorized=True)


if __name__ == "__main__":
    unittest.main()
