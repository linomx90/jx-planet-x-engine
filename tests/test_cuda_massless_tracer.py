from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import jxplanetx.cuda_massless_tracer as cuda_module
from jxplanetx.cuda_massless_tracer import (
    CUDAMasslessTracerContractError,
    CUDAMasslessTracerIntegrationError,
    CUDAMasslessTracerState,
    FixedStepCUDAMasslessTracerSpec,
    advance_cuda_massless_tracers,
    dump_cuda_massless_tracer_checkpoint,
    load_latest_cuda_massless_tracer_checkpoint,
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


def _backend(tile_size: int = 128, *, device: str = "cuda:0") -> BackendSpec:
    return BackendSpec(
        backend_id="cupy",
        device=device,
        tile_size=tile_size,
    )


def _provenance() -> Provenance:
    return Provenance(
        source_id="jx-test.synthetic-circular-tracers",
        citation="Analytic test construction",
        version="v1",
        sha256=hashlib.sha256(b"cuda-massless-tracer-tests-v1").hexdigest(),
    )


def _state(
    count: int,
    *,
    encounter_radius: float = 0.001,
    acceleration_bound: float = 100.0,
    indices=None,
) -> CUDAMasslessTracerState:
    cp = CP
    radii = cp.linspace(1.0, 2.0, count, dtype=cp.float64)
    position = cp.zeros((count, 3), dtype=cp.float64)
    velocity = cp.zeros((count, 3), dtype=cp.float64)
    if count:
        position[:, 0] = radii
        velocity[:, 1] = cp.sqrt(SOLAR_GM / radii)
    if indices is None:
        indices = cp.arange(count, dtype=cp.uint64)
    return CUDAMasslessTracerState(
        snapshot_id="synthetic-circular",
        epoch=0.0,
        global_step=0,
        time_scale="TDB",
        frame="ICRF",
        origin="SOLAR_SYSTEM_BARYCENTER",
        axes="J2000_ECLIPTIC",
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_positions=cp.zeros((1, 3), dtype=cp.float64),
        source_velocities=cp.zeros((1, 3), dtype=cp.float64),
        source_gm=cp.asarray([SOLAR_GM], dtype=cp.float64),
        source_encounter_radii=cp.asarray(
            [encounter_radius], dtype=cp.float64
        ),
        relative_acceleration_bounds=cp.asarray(
            [acceleration_bound], dtype=cp.float64
        ),
        tracer_positions=position,
        tracer_velocities=velocity,
        tracer_indices=indices,
        provenance=_provenance(),
    )


class CUDAMasslessTracerContractTests(unittest.TestCase):
    def test_spec_is_cuda_only_and_fixed_binary64_policy(self):
        with self.assertRaises(CUDAMasslessTracerContractError):
            FixedStepCUDAMasslessTracerSpec(
                0.5,
                1,
                BackendSpec("numpy", "cpu", 1),
            )
        with self.assertRaises(CUDAMasslessTracerContractError):
            FixedStepCUDAMasslessTracerSpec(0.0, 1, _backend())
        with self.assertRaises(CUDAMasslessTracerContractError):
            FixedStepCUDAMasslessTracerSpec(0.5, -1, _backend())
        with self.assertRaises(CUDAMasslessTracerContractError):
            FixedStepCUDAMasslessTracerSpec(0.5, 0, _backend())
        with self.assertRaises(CUDAMasslessTracerContractError):
            FixedStepCUDAMasslessTracerSpec(
                0.5,
                1,
                _backend(),
                maximum_encounter_routes_per_tracer=257,
            )

    def test_state_rejects_ambiguous_metadata_before_device_access(self):
        with self.assertRaises(CUDAMasslessTracerContractError):
            CUDAMasslessTracerState(
                snapshot_id="bad ",
                epoch=0.0,
                global_step=0,
                time_scale="TDB",
                frame="ICRF",
                origin="BARYCENTER",
                axes="J2000",
                length_unit="AU",
                time_unit="year",
                unit_system_id="AU-YEAR",
                source_ids=("Sun",),
                primary_source_id="Sun",
                source_positions=object(),
                source_velocities=object(),
                source_gm=object(),
                source_encounter_radii=object(),
                relative_acceleration_bounds=object(),
                tracer_positions=object(),
                tracer_velocities=object(),
                tracer_indices=object(),
                provenance=_provenance(),
            )

    def test_state_rejects_unit_or_primary_slot_ambiguity(self):
        base = {
            "snapshot_id": "contract",
            "epoch": 0.0,
            "global_step": 0,
            "time_scale": "TDB",
            "frame": "ICRF",
            "origin": "BARYCENTER",
            "axes": "J2000",
            "length_unit": "AU",
            "time_unit": "year",
            "unit_system_id": "AU-YEAR",
            "source_ids": ("Sun",),
            "primary_source_id": "Sun",
            "source_positions": object(),
            "source_velocities": object(),
            "source_gm": object(),
            "source_encounter_radii": object(),
            "relative_acceleration_bounds": object(),
            "tracer_positions": object(),
            "tracer_velocities": object(),
            "tracer_indices": object(),
            "provenance": _provenance(),
        }
        with self.assertRaises(CUDAMasslessTracerContractError):
            CUDAMasslessTracerState(**{**base, "length_unit": "km"})
        with self.assertRaises(CUDAMasslessTracerContractError):
            CUDAMasslessTracerState(**{**base, "unit_system_id": "KM-DAY"})
        with self.assertRaises(CUDAMasslessTracerContractError):
            CUDAMasslessTracerState(
                **{
                    **base,
                    "source_ids": ("Jupiter", "Sun"),
                    "primary_source_id": "Sun",
                }
            )


@GPU_REQUIRED
class CUDAMasslessTracerGPUTests(unittest.TestCase):
    def test_quiet_cuda_segment_is_finite_and_has_no_routes(self):
        result = advance_cuda_massless_tracers(
            _state(16),
            FixedStepCUDAMasslessTracerSpec(0.001, 100, _backend()),
        )
        self.assertEqual(result.state.global_step, 100)
        self.assertEqual(result.state.epoch, 0.1)
        self.assertTrue(bool(CP.isfinite(result.state.tracer_positions).all()))
        self.assertEqual(result.encounter_routes, ())
        self.assertTrue(bool((result.encounter_route_counts == 0).all()))
        self.assertFalse(result.claims["four_point_five_gyr_accuracy_qualified"])
        self.assertIn("implementation_source_sha256", result.runtime_identity)
        self.assertIn("scipy_version", result.runtime_identity)
        self.assertIn("device_uuid", result.runtime_identity)
        self.assertIn("cpu_fingerprint", result.runtime_identity)
        self.assertIn("cuda_local_runtime_version", result.runtime_identity)
        self.assertIn("nvrtc_version", result.runtime_identity)
        self.assertIn(
            "cupy_nvrtc_extension",
            result.runtime_identity["runtime_binary_sha256"],
        )
        self.assertTrue(result.runtime_identity["cpu_fingerprint"]["sha256"])
        with self.assertRaises(TypeError):
            result.claims["planet_nine_detected"] = True

    def test_source_path_is_bitwise_independent_of_tracer_count(self):
        zero = advance_cuda_massless_tracers(
            _state(0),
            FixedStepCUDAMasslessTracerSpec(0.001, 100, _backend(64)),
        )
        many = advance_cuda_massless_tracers(
            _state(129),
            FixedStepCUDAMasslessTracerSpec(0.001, 100, _backend(128)),
        )
        self.assertTrue(
            bool(CP.array_equal(zero.state.source_positions, many.state.source_positions))
        )
        self.assertTrue(
            bool(CP.array_equal(zero.state.source_velocities, many.state.source_velocities))
        )

    def test_nonfinite_massive_source_path_fails_closed(self):
        state = _state(0)
        state.source_velocities[0, 0] = 1.0e308
        with self.assertRaises(CUDAMasslessTracerIntegrationError):
            advance_cuda_massless_tracers(
                state,
                FixedStepCUDAMasslessTracerSpec(2.0, 1, _backend()),
            )

    def test_partition_and_lane_permutation_invariance(self):
        count = 65
        baseline_state = _state(
            count, indices=CP.arange(1000, 1000 + count, dtype=CP.uint64)
        )
        baseline = advance_cuda_massless_tracers(
            baseline_state,
            FixedStepCUDAMasslessTracerSpec(0.001, 50, _backend(64), threads_per_block=64),
        )
        permutation = CP.asarray(np.random.default_rng(7).permutation(count))
        permuted_state = CUDAMasslessTracerState(
            **{
                **baseline_state.__dict__,
                "snapshot_id": "synthetic-permuted",
                "tracer_positions": baseline_state.tracer_positions[permutation].copy(),
                "tracer_velocities": baseline_state.tracer_velocities[permutation].copy(),
                "tracer_indices": baseline_state.tracer_indices[permutation].copy(),
            }
        )
        permuted = advance_cuda_massless_tracers(
            permuted_state,
            FixedStepCUDAMasslessTracerSpec(
                0.001, 50, _backend(128), threads_per_block=128
            ),
        )
        left_order = CP.argsort(baseline.state.tracer_indices)
        right_order = CP.argsort(permuted.state.tracer_indices)
        self.assertTrue(
            bool(
                CP.array_equal(
                    baseline.state.tracer_positions[left_order],
                    permuted.state.tracer_positions[right_order],
                )
            )
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    baseline.state.tracer_velocities[left_order],
                    permuted.state.tracer_velocities[right_order],
                )
            )
        )

    def test_forced_guard_discards_gpu_step_and_uses_cpu_route(self):
        result = advance_cuda_massless_tracers(
            _state(1, encounter_radius=1.1),
            FixedStepCUDAMasslessTracerSpec(
                0.001,
                1,
                _backend(),
                maximum_encounter_routes_per_tracer=2,
            ),
        )
        self.assertEqual(len(result.encounter_routes), 1)
        self.assertEqual(result.encounter_routes[0].reason, "SWEPT_FLOOR")
        self.assertEqual(int(CP.asnumpy(result.encounter_route_counts[0])), 1)
        expected = np.asarray(
            [math.cos(2.0 * math.pi * 0.001), math.sin(2.0 * math.pi * 0.001), 0.0]
        )
        observed = CP.asnumpy(result.state.tracer_positions[0])
        self.assertLess(float(np.max(np.abs(observed - expected))), 2.0e-12)

    def test_acceleration_bound_excess_routes_instead_of_committing(self):
        result = advance_cuda_massless_tracers(
            _state(1, encounter_radius=0.0, acceleration_bound=1.0),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        self.assertEqual(len(result.encounter_routes), 1)
        self.assertEqual(result.encounter_routes[0].reason, "ACCELERATION_BOUND")

    def test_finite_hyperbolic_state_is_a_diagnostic_not_an_error(self):
        state = _state(1)
        state.tracer_velocities[0, 1] = 12.0
        result = advance_cuda_massless_tracers(
            state,
            FixedStepCUDAMasslessTracerSpec(0.0001, 1, _backend()),
        )
        self.assertTrue(bool(CP.asnumpy(result.ever_unbound[0])))
        self.assertEqual(result.encounter_routes, ())

    def test_terminal_cpu_route_updates_node_sampled_unbound_diagnostic(self):
        state = _state(1, encounter_radius=2.0, acceleration_bound=100.0)
        state.tracer_velocities[0] = CP.asarray(
            [0.0, math.sqrt(2.0 * SOLAR_GM) * 0.99999, 0.0],
            dtype=CP.float64,
        )
        result = advance_cuda_massless_tracers(
            state,
            FixedStepCUDAMasslessTracerSpec(
                1.0,
                1,
                _backend(),
                encounter_route_rtol=1.0,
                encounter_route_atol=1.0,
                encounter_route_maximum_substep_fraction=1.0,
            ),
        )
        self.assertTrue(bool(CP.asnumpy(result.ever_unbound[0])))
        self.assertEqual(len(result.encounter_routes), 1)
        self.assertTrue(result.encounter_routes[0].endpoint_unbound)

    def test_partial_continuation_or_lowered_route_budget_fails_closed(self):
        initial = _state(1)
        reset_noninitial = CUDAMasslessTracerState(
            **{
                **initial.__dict__,
                "epoch": 0.001,
                "global_step": 1,
            }
        )
        with self.assertRaises(CUDAMasslessTracerContractError):
            advance_cuda_massless_tracers(
                reset_noninitial,
                FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
            )
        partial = CUDAMasslessTracerState(
            **{
                **initial.__dict__,
                "minimum_source_distance_AU": CP.ones(1, dtype=CP.float64),
            }
        )
        with self.assertRaises(CUDAMasslessTracerContractError):
            advance_cuda_massless_tracers(
                partial,
                FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
            )
        routed = advance_cuda_massless_tracers(
            _state(1, encounter_radius=1.1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        with self.assertRaises(CUDAMasslessTracerContractError):
            advance_cuda_massless_tracers(
                routed.state,
                FixedStepCUDAMasslessTracerSpec(
                    0.001,
                    1,
                    _backend(),
                    maximum_encounter_routes_per_tracer=0,
                ),
            )

    def test_route_ledger_cannot_contradict_cumulative_diagnostics(self):
        routed = advance_cuda_massless_tracers(
            _state(1, encounter_radius=1.1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        contradictory_record = replace(
            routed.encounter_routes[0],
            endpoint_unbound=True,
            minimum_evaluated_stage_distance_AU=0.1,
        )
        contradictory = CUDAMasslessTracerState(
            **{
                **routed.state.__dict__,
                "minimum_source_distance_AU": CP.asarray(
                    [1.0], dtype=CP.float64
                ),
                "ever_unbound": CP.asarray([False], dtype=CP.bool_),
                "prior_encounter_routes": (contradictory_record,),
            }
        )
        with self.assertRaises(CUDAMasslessTracerContractError):
            advance_cuda_massless_tracers(
                contradictory,
                FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
            )

    def test_split_segment_dynamical_buffers_are_bitwise_identical(self):
        initial = _state(8)
        first = advance_cuda_massless_tracers(
            initial,
            FixedStepCUDAMasslessTracerSpec(0.001, 10, _backend()),
        )
        split = advance_cuda_massless_tracers(
            first.state,
            FixedStepCUDAMasslessTracerSpec(0.001, 10, _backend()),
        )
        direct = advance_cuda_massless_tracers(
            initial,
            FixedStepCUDAMasslessTracerSpec(0.001, 20, _backend()),
        )
        self.assertTrue(
            bool(CP.array_equal(split.state.source_positions, direct.state.source_positions))
        )
        self.assertTrue(
            bool(CP.array_equal(split.state.tracer_positions, direct.state.tracer_positions))
        )
        self.assertTrue(
            bool(CP.array_equal(split.state.tracer_velocities, direct.state.tracer_velocities))
        )
        self.assertTrue(
            bool(
                CP.array_equal(
                    split.minimum_source_distance_AU,
                    direct.minimum_source_distance_AU,
                )
            )
        )
        self.assertTrue(bool(CP.array_equal(split.ever_unbound, direct.ever_unbound)))
        self.assertTrue(
            bool(
                CP.array_equal(
                    split.encounter_route_counts,
                    direct.encounter_route_counts,
                )
            )
        )

    def test_internal_source_path_chunking_is_bitwise_identical(self):
        initial = _state(33)
        one_chunk = advance_cuda_massless_tracers(
            initial,
            FixedStepCUDAMasslessTracerSpec(
                0.001, 65, _backend(), source_path_steps_per_chunk=65
            ),
        )
        many_chunks = advance_cuda_massless_tracers(
            initial,
            FixedStepCUDAMasslessTracerSpec(
                0.001, 65, _backend(), source_path_steps_per_chunk=7
            ),
        )
        for left, right in (
            (one_chunk.state.source_positions, many_chunks.state.source_positions),
            (one_chunk.state.source_velocities, many_chunks.state.source_velocities),
            (one_chunk.state.tracer_positions, many_chunks.state.tracer_positions),
            (one_chunk.state.tracer_velocities, many_chunks.state.tracer_velocities),
            (
                one_chunk.minimum_source_distance_AU,
                many_chunks.minimum_source_distance_AU,
            ),
            (one_chunk.ever_unbound, many_chunks.ever_unbound),
            (one_chunk.encounter_route_counts, many_chunks.encounter_route_counts),
        ):
            self.assertTrue(bool(CP.array_equal(left, right)))

    def test_checkpoint_uses_numeric_latest_and_detects_corruption(self):
        result = advance_cuda_massless_tracers(
            _state(4),
            FixedStepCUDAMasslessTracerSpec(0.001, 10, _backend()),
        )
        protocol = hashlib.sha256(b"test-protocol").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = dump_cuda_massless_tracer_checkpoint(
                directory,
                run_id="test-run",
                protocol_sha256=protocol,
                checkpoint_index=9,
                result=result,
            )
            second_result = advance_cuda_massless_tracers(
                result.state,
                FixedStepCUDAMasslessTracerSpec(0.001, 10, _backend()),
            )
            second = dump_cuda_massless_tracer_checkpoint(
                directory,
                run_id="test-run",
                protocol_sha256=protocol,
                checkpoint_index=1000,
                result=second_result,
                previous_checkpoint_content_sha256=first[
                    "checkpoint_content_sha256"
                ],
            )
            with self.assertRaises(CUDAMasslessTracerContractError):
                dump_cuda_massless_tracer_checkpoint(
                    directory,
                    run_id="test-run",
                    protocol_sha256=protocol,
                    checkpoint_index=1001,
                    result=second_result,
                    previous_checkpoint_content_sha256=hashlib.sha256(
                        b"false predecessor"
                    ).hexdigest(),
                )
            self.assertEqual(
                second["previous_checkpoint_content_sha256"],
                first["checkpoint_content_sha256"],
            )
            dump_cuda_massless_tracer_checkpoint(
                directory,
                run_id="other-run",
                protocol_sha256=hashlib.sha256(b"other-protocol").hexdigest(),
                checkpoint_index=1000,
                result=result,
            )
            loaded = load_latest_cuda_massless_tracer_checkpoint(
                directory,
                run_id="test-run",
                protocol_sha256=protocol,
                backend=_backend(),
            )
            self.assertIsNotNone(loaded)
            self.assertTrue(
                bool(
                    CP.array_equal(
                        loaded.state.tracer_positions,
                        second_result.state.tracer_positions,
                    )
                )
            )
            resumed = advance_cuda_massless_tracers(
                loaded.state,
                FixedStepCUDAMasslessTracerSpec(0.001, 10, _backend()),
            )
            direct = advance_cuda_massless_tracers(
                _state(4),
                FixedStepCUDAMasslessTracerSpec(0.001, 30, _backend()),
            )
            for left, right in (
                (resumed.state.tracer_positions, direct.state.tracer_positions),
                (resumed.state.tracer_velocities, direct.state.tracer_velocities),
                (
                    resumed.minimum_source_distance_AU,
                    direct.minimum_source_distance_AU,
                ),
                (resumed.ever_unbound, direct.ever_unbound),
                (resumed.encounter_route_counts, direct.encounter_route_counts),
            ):
                self.assertTrue(bool(CP.array_equal(left, right)))
            latest_array = (
                directory
                / second["checkpoint_namespace"]
                / second["array_file"]
            )
            data = bytearray(latest_array.read_bytes())
            data[len(data) // 2] ^= 1
            latest_array.write_bytes(data)
            with self.assertRaises(CUDAMasslessTracerIntegrationError):
                load_latest_cuda_massless_tracer_checkpoint(
                    directory,
                    run_id="test-run",
                    protocol_sha256=protocol,
                    backend=_backend(),
                )

    def test_checkpoint_preserves_route_ledger_and_diagnostics(self):
        result = advance_cuda_massless_tracers(
            _state(1, encounter_radius=1.1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        protocol = hashlib.sha256(b"routed-protocol").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dump_cuda_massless_tracer_checkpoint(
                directory,
                run_id="routed-run",
                protocol_sha256=protocol,
                checkpoint_index=0,
                result=result,
            )
            loaded = load_latest_cuda_massless_tracer_checkpoint(
                directory,
                run_id="routed-run",
                protocol_sha256=protocol,
                backend=_backend(),
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.encounter_routes, result.encounter_routes)
            self.assertTrue(
                bool(
                    CP.array_equal(
                        loaded.encounter_route_counts,
                        result.encounter_route_counts,
                    )
                )
            )
            self.assertTrue(
                bool(
                    CP.array_equal(
                        loaded.minimum_source_distance_AU,
                        result.minimum_source_distance_AU,
                    )
                )
            )

    def test_checkpoint_retry_survives_uncommitted_array_orphan(self):
        result = advance_cuda_massless_tracers(
            _state(1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        protocol = hashlib.sha256(b"orphan-retry-protocol").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch(
                "jxplanetx.cuda_massless_tracer._atomic_json_create",
                side_effect=OSError("injected record commit failure"),
            ):
                with self.assertRaises(OSError):
                    dump_cuda_massless_tracer_checkpoint(
                        directory,
                        run_id="orphan-retry-run",
                        protocol_sha256=protocol,
                        checkpoint_index=0,
                        result=result,
                    )
            self.assertEqual(
                load_latest_cuda_massless_tracer_checkpoint(
                    directory,
                    run_id="orphan-retry-run",
                    protocol_sha256=protocol,
                    backend=_backend(),
                ),
                None,
            )
            committed = dump_cuda_massless_tracer_checkpoint(
                directory,
                run_id="orphan-retry-run",
                protocol_sha256=protocol,
                checkpoint_index=0,
                result=result,
            )
            self.assertTrue(
                (
                    directory
                    / committed["checkpoint_namespace"]
                    / f"checkpoint_{0:020d}.json"
                ).is_file()
            )

    def test_checkpoint_accepts_canonicalized_cuda_device_alias(self):
        alias_backend = _backend(device="gpu:0")
        result = advance_cuda_massless_tracers(
            _state(1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, alias_backend),
        )
        self.assertEqual(result.runtime_identity["array_backend_device"], "cuda:0")
        protocol = hashlib.sha256(b"device-alias-protocol").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            dump_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="device-alias-run",
                protocol_sha256=protocol,
                checkpoint_index=0,
                result=result,
            )
            loaded = load_latest_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="device-alias-run",
                protocol_sha256=protocol,
                backend=_backend(),
            )
            self.assertIsNotNone(loaded)

    def test_checkpoint_lineage_rejects_unrelated_successor_and_preserves_routes(self):
        protocol = hashlib.sha256(b"lineage-protocol").hexdigest()
        routed = advance_cuda_massless_tracers(
            _state(1, encounter_radius=1.1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = dump_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="lineage-run",
                protocol_sha256=protocol,
                checkpoint_index=0,
                result=routed,
            )
            unrelated = advance_cuda_massless_tracers(
                _state(1),
                FixedStepCUDAMasslessTracerSpec(0.001, 2, _backend()),
            )
            with self.assertRaises(CUDAMasslessTracerIntegrationError):
                dump_cuda_massless_tracer_checkpoint(
                    temporary,
                    run_id="lineage-run",
                    protocol_sha256=protocol,
                    checkpoint_index=1,
                    result=unrelated,
                    previous_checkpoint_content_sha256=first[
                        "checkpoint_content_sha256"
                    ],
                )
            loaded = load_latest_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="lineage-run",
                protocol_sha256=protocol,
                backend=_backend(),
            )
            continued = advance_cuda_massless_tracers(
                loaded.state,
                FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
            )
            second = dump_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="lineage-run",
                protocol_sha256=protocol,
                checkpoint_index=2,
                result=continued,
                previous_checkpoint_content_sha256=first[
                    "checkpoint_content_sha256"
                ],
            )
            final = load_latest_cuda_massless_tracer_checkpoint(
                temporary,
                run_id="lineage-run",
                protocol_sha256=protocol,
                backend=_backend(),
            )
            self.assertEqual(len(final.encounter_routes), 2)
            self.assertEqual(int(CP.asnumpy(final.encounter_route_counts[0])), 2)
            self.assertEqual(
                second["input_state_sha256"], first["state_sha256"]
            )

    def test_checkpoint_validates_historical_runtime_and_namespace_type(self):
        protocol = hashlib.sha256(b"history-controls-protocol").hexdigest()
        first_result = advance_cuda_massless_tracers(
            _state(1),
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        second_result = advance_cuda_massless_tracers(
            first_result.state,
            FixedStepCUDAMasslessTracerSpec(0.001, 1, _backend()),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = dump_cuda_massless_tracer_checkpoint(
                root,
                run_id="history-run",
                protocol_sha256=protocol,
                checkpoint_index=0,
                result=first_result,
            )
            second = dump_cuda_massless_tracer_checkpoint(
                root,
                run_id="history-run",
                protocol_sha256=protocol,
                checkpoint_index=1,
                result=second_result,
                previous_checkpoint_content_sha256=first[
                    "checkpoint_content_sha256"
                ],
            )
            namespace = root / first["checkpoint_namespace"]
            first_path = namespace / f"checkpoint_{0:020d}.json"
            second_path = namespace / f"checkpoint_{1:020d}.json"

            first_record = json.loads(first_path.read_text(encoding="utf-8"))
            first_record.pop("checkpoint_content_sha256")
            first_record["runtime_identity"]["python_version"] = "tampered"
            first_digest = hashlib.sha256(
                cuda_module.CHECKPOINT_DOMAIN
                + cuda_module._canonical_json(first_record)
            ).hexdigest()
            first_record["checkpoint_content_sha256"] = first_digest
            first_path.write_text(
                json.dumps(first_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            second_record = json.loads(second_path.read_text(encoding="utf-8"))
            second_record.pop("checkpoint_content_sha256")
            second_record["previous_checkpoint_content_sha256"] = first_digest
            second_digest = hashlib.sha256(
                cuda_module.CHECKPOINT_DOMAIN
                + cuda_module._canonical_json(second_record)
            ).hexdigest()
            second_record["checkpoint_content_sha256"] = second_digest
            second_path.write_text(
                json.dumps(second_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(CUDAMasslessTracerIntegrationError):
                load_latest_cuda_massless_tracer_checkpoint(
                    root,
                    run_id="history-run",
                    protocol_sha256=protocol,
                    backend=_backend(),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = cuda_module._checkpoint_run_directory(
                root, "namespace-run", protocol
            )
            namespace.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(CUDAMasslessTracerIntegrationError):
                load_latest_cuda_massless_tracer_checkpoint(
                    root,
                    run_id="namespace-run",
                    protocol_sha256=protocol,
                    backend=_backend(),
                )


if __name__ == "__main__":
    unittest.main()
