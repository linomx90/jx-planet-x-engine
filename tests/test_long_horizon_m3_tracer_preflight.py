from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from benchmarks import jx_long_horizon_m3_tracer_preflight as m3


ROOT = Path(__file__).resolve().parents[1]


def _threshold_limits(value: float = 1.0) -> dict[str, dict[str, str]]:
    return {
        comparison: {metric: value.hex() for metric in m3.COMPARISON_METRICS}
        for comparison in m3.COMPARISON_IDS
    }


def _passing_analysis(protocol: dict[str, object]) -> dict[str, object]:
    inputs = protocol["inputs"]
    state_ids = inputs["state_ids"]
    family_ids = inputs["family_ids"]
    unique_families = tuple(dict.fromkeys(family_ids))
    comparisons: dict[str, object] = {}
    for comparison in m3.COMPARISON_IDS:
        comparisons[comparison] = {
            "metrics": {
                metric: {
                    "global_maximum_hex": (0.0).hex(),
                    "per_state_hex": {
                        state_id: (0.0).hex() for state_id in state_ids
                    },
                    "per_family_maximum_hex": {
                        family_id: (0.0).hex()
                        for family_id in unique_families
                    },
                }
                for metric in m3.COMPARISON_METRICS
            }
        }
    encounter = {
        lane: {
            "interval_count": interval_count,
            "all_intervals_clear": True,
            "minimum_instantaneous_source_distance_AU": 2.0,
            "minimum_certified_interval_clearance_AU": 1.0,
            "uniform_relative_acceleration_bound_AU_per_year2": float.fromhex(
                inputs[
                    "uniform_relative_acceleration_bound_AU_per_year2_hex"
                ]
            ),
            "bootstrap_policy": (
                "FIRST_CROSSING_CONTRADICTION_FROM_START_DISTANCE_MINUS_"
                "RELATIVE_SPEED_H_MINUS_HALF_UNIFORM_ACCELERATION_H2_V1"
            ),
            "algorithmic_guard_radius_AU": m3.ALGORITHMIC_GUARD_RADIUS_AU,
            "source_pair_safety": {
                "all_source_pair_intervals_clear": True,
                "minimum_instantaneous_source_pair_distance_AU": 2.0,
                "minimum_certified_source_pair_interval_clearance_AU": 1.0,
                "pair_count": 45,
                "interval_count": interval_count,
                "rounding_enclosure_relative_pad_hex": (
                    m3._GUARD_ROUNDING_PAD.hex()
                ),
            },
            "instantaneous_phase_passed": True,
            "maximum_instantaneous_primary_phase_rad": 1.0e-6,
            "periapse_phase_passed": True,
            "maximum_periapse_primary_phase_rad": 1.0e-6,
        }
        for lane, interval_count in (
            ("jx_candidate", m3.CANDIDATE_MAP_STEP_COUNT),
            ("jx_refinement", m3.REFINEMENT_MAP_STEP_COUNT),
            ("rebound_ias15", m3.REFINEMENT_MAP_STEP_COUNT),
        )
    }
    active_lifecycle_sha256 = m3._array_sha256(
        np.zeros(m3.STATE_COUNT, dtype=np.uint8)
    )
    tracer = {
        "tracer_positions_sha256": inputs["tracer_positions_sha256"],
        "tracer_velocities_sha256": inputs["tracer_velocities_sha256"],
    }
    source = {
        "source_positions_sha256": inputs["source_initial_positions_sha256"],
        "source_velocities_sha256": inputs["source_initial_velocities_sha256"],
    }
    parent_sha = inputs["m2_source"]["content_sha256"]
    source_expectations = inputs["m2_execution_source_custody"]

    def source_custody(lane: str, map_count: int) -> dict[str, object]:
        return {
            "parent_path_content_sha256_before": parent_sha,
            "parent_path_content_sha256_after": parent_sha,
            "parent_unchanged": True,
            "map_slice_count": map_count,
            "slice_dynamic_digest": source_expectations[lane][
                "slice_dynamic_digest"
            ],
            "tile_binding_digest": source_expectations[lane][
                "tile_binding_digest"
            ],
            "first_global_source_step": 0,
            "final_global_source_step": m3.SOURCE_STEP_COUNT,
            "every_slice_verified_by_constructor": True,
        }

    candidate_endpoint_hashes = [
        f"{index + 1:064x}"
        for index in range(m3.CANDIDATE_MAP_STEP_COUNT + 1)
    ]
    refinement_endpoint_hashes = [
        f"{index + m3.CANDIDATE_MAP_STEP_COUNT + 2:064x}"
        for index in range(m3.REFINEMENT_MAP_STEP_COUNT + 1)
    ]

    def trace_custody(
        lane: str, map_count: int, ratio: int, offset: int
    ) -> dict[str, object]:
        return m3._content_record(
            {
                "lane_id": lane,
                "node_count": map_count + 1,
                "state_roster_sha256": hashlib.sha256(
                    m3.TRACE_ROSTER_DOMAIN + m3._canonical_json(state_ids)
                ).hexdigest(),
                "source_steps_per_map_step": ratio,
                "first_global_source_step": 0,
                "final_global_source_step": m3.SOURCE_STEP_COUNT,
                "node_sha256": [
                    f"{offset + index + 1:064x}"
                    for index in range(map_count + 1)
                ],
            },
            m3.TRACE_CUSTODY_DOMAIN,
        )

    def split_receipt(
        lane: str, hashes: list[str], split_index: int
    ) -> dict[str, object]:
        split_sha = hashes[split_index]
        terminal_sha = hashes[-1]
        event_sha = "5" * 64
        return {
            "lane_id": lane,
            "genesis_state_sha256": hashes[0],
            "first_half_state_sha256": split_sha,
            "expected_split_state_sha256": split_sha,
            "split_checkpoint_content_sha256": "7" * 64,
            "final_checkpoint_content_sha256": "8" * 64,
            "loaded_split_checkpoint_state_sha256": split_sha,
            "loaded_final_checkpoint_state_sha256": terminal_sha,
            "terminal_state_sha256": terminal_sha,
            "expected_terminal_state_sha256": terminal_sha,
            "first_half_matches_one_map_trace": True,
            "bitwise_exact": True,
            "lifecycle_state_bitwise_exact": True,
            "event_ledger_bitwise_exact": True,
            "expected_event_ledger_sha256": event_sha,
            "split_event_ledger_sha256": event_sha,
            "annual_checkpoint_count": m3.HORIZON_YEARS,
            "annual_source_slice_dynamic_node_sha256": list(
                source_expectations[
                    "annual_source_slice_dynamic_node_sha256"
                ]
            ),
            "annual_source_tile_binding_sha256": list(
                source_expectations["annual_source_tile_binding_sha256"]
            ),
        }

    def event_receipt() -> dict[str, object]:
        return {
            "exact": True,
            "route_record_count": 0,
            "removal_record_count": 0,
            "terminal_lane_count": 0,
            "lifecycle_transition_count": 0,
            "no_reactivation": True,
            "ledger_content_sha256": "b" * 64,
        }

    candidate_custody = source_custody(
        "candidate", m3.CANDIDATE_MAP_STEP_COUNT
    )
    refinement_custody = source_custody(
        "refinement", m3.REFINEMENT_MAP_STEP_COUNT
    )
    candidate_split = split_receipt(
        "candidate", candidate_endpoint_hashes, m3.SPLIT_CANDIDATE_MAP_STEP
    )
    refinement_split = split_receipt(
        "refinement", refinement_endpoint_hashes, m3.SPLIT_REFINEMENT_MAP_STEP
    )
    candidate_events = event_receipt()
    refinement_events = event_receipt()
    runtime_identity = {"unit_runtime": "exact"}
    return {
        "comparisons": comparisons,
        "encounter_and_timestep": encounter,
        "initial_state_custody": {
            "candidate": {**tracer, "lifecycle_sha256": active_lifecycle_sha256},
            "refinement": {**tracer, "lifecycle_sha256": active_lifecycle_sha256},
            "rebound_ias15": {**tracer, **source},
            "rebound_ias15_source_only": dict(source),
            "candidate_source": dict(source),
            "refinement_source": dict(source),
            "source_gm_sha256": inputs["source_gm_sha256"],
            "candidate_node_zero_index": 0,
            "refinement_node_zero_index": 0,
            "rebound_node_zero_index": 0,
        },
        "source_no_backreaction_bitwise_exact": True,
        "source_parent_custody": {
            "parent_path_content_sha256": parent_sha,
            "parent_dynamic_node_sha256": inputs["m2_source"][
                "dynamic_node_sha256"
            ],
            "candidate_exact_parent_slices": True,
            "refinement_exact_parent_slices": True,
            "parent_unchanged_after_execution": True,
            "candidate": candidate_custody,
            "refinement": refinement_custody,
            "guard_overlay_sha256": source_expectations[
                "guard_overlay_sha256"
            ],
        },
        "split_resume": {
            "candidate": candidate_split,
            "refinement": refinement_split,
            "candidate_bitwise_exact": True,
            "refinement_bitwise_exact": True,
        },
        "event_accounting": {
            "candidate": candidate_events,
            "refinement": refinement_events,
            "candidate_exact": True,
            "refinement_exact": True,
        },
        "endpoint_state_custody": {
            "candidate_state_anchors": {
                "genesis": candidate_endpoint_hashes[0],
                "split": candidate_endpoint_hashes[
                    m3.SPLIT_CANDIDATE_MAP_STEP
                ],
                "terminal": candidate_endpoint_hashes[-1],
            },
            "refinement_state_anchors": {
                "genesis": refinement_endpoint_hashes[0],
                "split": refinement_endpoint_hashes[
                    m3.SPLIT_REFINEMENT_MAP_STEP
                ],
                "terminal": refinement_endpoint_hashes[-1],
            },
            "candidate_trace_node_custody": trace_custody(
                "candidate",
                m3.CANDIDATE_MAP_STEP_COUNT,
                m3.CANDIDATE_SOURCE_STEPS_PER_MAP,
                20_000,
            ),
            "refinement_trace_node_custody": trace_custody(
                "refinement",
                m3.REFINEMENT_MAP_STEP_COUNT,
                m3.REFINEMENT_SOURCE_STEPS_PER_MAP,
                40_000,
            ),
        },
        "runtime": {
            "runtime_identity_start": runtime_identity,
            "runtime_identity_end": deepcopy(runtime_identity),
            "runtime_identity_stable": True,
            "runtime_identity_sha256": hashlib.sha256(
                m3.RUNTIME_IDENTITY_DOMAIN
                + m3._canonical_json(runtime_identity)
            ).hexdigest(),
        },
    }


def _mock_bundle() -> dict[str, object]:
    return {
        "arrays": {
            "mock_state": np.asarray([1.0, 2.0], dtype=np.float64),
            "mock_lifecycle": np.asarray([0, 0], dtype=np.uint8),
        },
        "candidate_events": [],
        "refinement_events": [],
        "receipts": {},
    }


def _event_trace(
    lifecycle: np.ndarray,
    *,
    routes: list[dict[str, object]] | None = None,
    removals: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    route_rows = [] if routes is None else routes
    removal_rows = [] if removals is None else removals
    position = np.zeros((*lifecycle.shape, 3), dtype=np.float64)
    velocity = np.zeros_like(position)
    for row in route_rows:
        node = int(row["global_map_step"]) + 1
        lane = int(row["lane_index"])
        position[node, lane] = [
            float.fromhex(value)
            for value in row["endpoint_relative_position_AU_hex"]
        ]
        velocity[node, lane] = [
            float.fromhex(value)
            for value in row["endpoint_relative_velocity_AU_per_year_hex"]
        ]
    for row in removal_rows:
        node = (
            int(row["global_source_step"])
            // int(row["source_steps_per_map_step"])
            + 1
        )
        lane = int(row["lane_index"])
        event = row["event_content"][0]
        terminal_position = np.asarray(
            [float.fromhex(value) for value in event["relative_position_AU_hex"]],
            dtype=np.float64,
        )
        terminal_velocity = np.asarray(
            [
                float.fromhex(value)
                for value in event["relative_velocity_AU_per_year_hex"]
            ],
            dtype=np.float64,
        )
        position[node:, lane] = terminal_position
        velocity[node:, lane] = terminal_velocity
    return {
        "lifecycle": lifecycle,
        "position_AU": position,
        "velocity_AU_per_year": velocity,
        "route_records": route_rows,
        "removal_records": removal_rows,
    }


class LongHorizonM3TracerPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = m3.build_protocol(ROOT)

    def test_literal_v1_schema_run_and_domain_pins(self) -> None:
        self.assertEqual(
            m3.PROTOCOL_SCHEMA,
            "jx.long-horizon.m3-tracer-preflight.protocol.v1",
        )
        self.assertEqual(
            m3.CALIBRATION_SCHEMA,
            "jx.long-horizon.m3-tracer-preflight.calibration.v1",
        )
        self.assertEqual(
            m3.RESULT_SCHEMA,
            "jx.long-horizon.m3-tracer-preflight.result.v1",
        )
        self.assertEqual(
            m3.RUN_ID, "jx-long-horizon-m3-tracer-preflight-10y-v1"
        )
        self.assertEqual(
            m3.PROTOCOL_DOMAIN,
            b"jx.long-horizon.m3-tracer-preflight.protocol.v1\0",
        )
        self.assertEqual(
            m3.CALIBRATION_DOMAIN,
            b"jx.long-horizon.m3-tracer-preflight.calibration.v1\0",
        )
        self.assertEqual(
            m3.RESULT_DOMAIN,
            b"jx.long-horizon.m3-tracer-preflight.result.v1\0",
        )

    def test_prepare_protocol_is_deterministic_and_non_authorizing(self) -> None:
        second = m3.build_protocol(ROOT)
        self.assertEqual(self.protocol, second)
        m3.validate_protocol(self.protocol)
        self.assertEqual(
            self.protocol["status"], "PREPARED_THRESHOLDS_UNSET_NOT_EXECUTED"
        )
        self.assertIsNone(
            self.protocol["comparisons"]["gate_thresholds"]
        )
        self.assertEqual(
            self.protocol["comparisons"]["threshold_status"],
            "UNSET_CALIBRATION_REQUIRED",
        )
        self.assertFalse(self.protocol["execution_authorized"])
        self.assertFalse(self.protocol["official_attempt_authorized"])
        self.assertFalse(
            self.protocol["staged_one_thousand_year_run_authorized"]
        )
        self.assertTrue(all(value is False for value in self.protocol["claims"].values()))

        changed = deepcopy(self.protocol)
        changed["execution_authorized"] = True
        with self.assertRaises(m3.M3ProtocolError):
            m3.validate_protocol(changed)

    def test_exact_retained_m2_and_108_state_bridges_are_bound(self) -> None:
        inputs = self.protocol["inputs"]
        self.assertEqual(inputs["state_count"], 108)
        self.assertEqual(len(inputs["state_ids"]), 108)
        self.assertEqual(len(set(inputs["state_ids"])), 108)
        self.assertEqual(len(inputs["family_ids"]), 108)
        self.assertEqual(len(set(inputs["family_ids"])), 12)
        self.assertEqual(set(inputs["family_state_counts"].values()), {9})
        source = inputs["m2_source"]
        self.assertEqual(source["node_count"], 81_921)
        self.assertEqual(
            source["dynamic_node_sha256"],
            "86293bb87aba3f3219aaa3d90263a07c271496f30ee0ee8a997428dfc2ce8487",
        )
        self.assertEqual(
            source["content_sha256"],
            "c332e9ff937f0aa7157a095e196d3a30adde5f26998b16a75c0740f3d836945d",
        )
        self.assertEqual(
            source["official_result_file_sha256"],
            "ee9b7d5e727ae12d48757f3919fdf7d1acba49807caea730c4b6fd0c1a73b461",
        )
        self.assertEqual(
            source["official_completion_file_sha256"],
            "081ac54da1a7915ea8d1d506ba6a312b716fb77fcdfd79bfcdc81c10b98f9c5a",
        )
        self.assertEqual(
            inputs["profile_receipt_sha256"],
            {
                "candidate": "045db073e83919aa9981875db5c5200f621b44b01a808ef974db5e3c21d7dbc7",
                "refinement": "f9f71b5429755f62e8488595ca34cbde7b006647fe050eafef0868d94a82efa4",
            },
        )

    def test_precision_first_lanes_and_independent_comparator_are_exact(self) -> None:
        candidate = self.protocol["lanes"]["jx_candidate"]
        refinement = self.protocol["lanes"]["jx_refinement"]
        comparator = self.protocol["lanes"]["independent_comparator"]
        self.assertEqual(float.fromhex(candidate["step_years_hex"]), 1.0 / 512.0)
        self.assertEqual(candidate["source_steps_per_map_step"], 16)
        self.assertEqual(candidate["map_step_count"], 5_120)
        self.assertEqual(float.fromhex(refinement["step_years_hex"]), 1.0 / 1024.0)
        self.assertEqual(refinement["source_steps_per_map_step"], 8)
        self.assertEqual(refinement["map_step_count"], 10_240)
        self.assertEqual(comparator["required_exact_version"], "5.1.1")
        self.assertEqual(comparator["adaptive_mode"], "PRS23")
        self.assertEqual(
            float.fromhex(comparator["initial_step_years_hex"]), 1.0e-3
        )
        self.assertEqual(
            float.fromhex(comparator["minimum_step_years_hex"]), 0.0
        )
        self.assertEqual(comparator["active_massive_source_count"], 10)
        self.assertEqual(comparator["exactly_massless_tracer_count"], 108)
        self.assertTrue(comparator["output_every_refinement_node"])
        self.assertFalse(comparator["reference_truth"])
        self.assertFalse(
            self.protocol["precision_selection"][
                "legacy_ratio_32_lane_is_official"
            ]
        )

    def test_executed_source_inventory_is_cwd_independent(self) -> None:
        expected = m3._executed_source_paths(ROOT)
        self.assertIn(Path("src/jxplanetx/long_horizon/__init__.py"), expected)
        self.assertIn(Path("src/jxplanetx/engine/backends.py"), expected)
        self.assertIn(Path("src/jxplanetx/fast_wisdom_holman.py"), expected)
        self.assertIn(
            Path("src/jxplanetx/_wisdom_holman_cpu_v3_module.c"), expected
        )
        self.assertIn(
            Path("src/jxplanetx/_wisdom_holman_loop_v3.c"), expected
        )
        self.assertIn(
            Path(
                "src/jxplanetx/data/long_horizon_known_system_v2/"
                "known_system_primary_relative_ecliptic_float_hex.csv"
            ),
            expected,
        )
        self.assertIn(
            Path(
                "src/jxplanetx/data/long_horizon_known_system_v2/"
                "SHA256SUMS"
            ),
            expected,
        )
        self.assertTrue(any(path.suffix == ".so" for path in expected))
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                actual = m3._executed_source_paths(ROOT)
            finally:
                os.chdir(previous)
        self.assertEqual(actual, expected)

    def test_code_identity_binds_imported_origins_data_and_native_bytes(self) -> None:
        identity = self.protocol["code_identity"]
        rows = {row["path"]: row for row in identity["files"]}
        self.assertEqual(identity["file_count"], len(rows))
        self.assertTrue(
            any("/data/long_horizon_known_system_v2/" in path for path in rows)
        )
        self.assertTrue(any(path.endswith(".so") for path in rows))
        imported = identity["imported_local_modules"]
        self.assertTrue(imported)
        self.assertIn("__benchmark__", {row["module"] for row in imported})
        for row in imported:
            self.assertEqual(row["sha256"], rows[row["path"]]["sha256"])
            self.assertEqual(
                row["size_bytes"], rows[row["path"]]["size_bytes"]
            )

        forged = deepcopy(rows)
        benchmark_path = next(
            row["path"] for row in imported if row["module"] == "__benchmark__"
        )
        forged[benchmark_path]["sha256"] = "f" * 64
        with self.assertRaises(m3.M3ProtocolError):
            m3._imported_local_module_identity(ROOT, forged)

    def test_preparation_runtime_is_exact_and_does_not_load_cuda(self) -> None:
        runtime = self.protocol["preparation_runtime_identity"]
        self.assertFalse(runtime["numerical_packages_included"])
        self.assertEqual(len(runtime["python_executable"]["sha256"]), 64)
        numpy_identity = runtime["numpy_distribution"]
        self.assertGreater(numpy_identity["record_member_count"], 0)
        self.assertGreater(numpy_identity["native_binary_count"], 0)
        self.assertEqual(len(numpy_identity["record_file"]["sha256"]), 64)
        self.assertEqual(len(numpy_identity["record_inventory_sha256"]), 64)
        with mock.patch.object(
            m3,
            "_numerical_runtime_modules",
            side_effect=AssertionError("CUDA numerical closure was imported"),
        ):
            observed = m3._runtime_identity(
                include_numerical_packages=False,
                project_root=ROOT,
                expected_code_identity=self.protocol["code_identity"],
            )
        self.assertFalse(observed["numerical_packages_included"])

    def test_gpu_identity_binds_device_runtime_backend_and_kernel(self) -> None:
        properties = {
            "name": b"unit-gpu",
            "uuid": bytes.fromhex("01" * 16),
            "major": 12,
            "minor": 0,
            "pciDomainID": 0,
            "pciBusID": 1,
            "pciDeviceID": 2,
        }
        runtime = SimpleNamespace(
            getDeviceProperties=lambda _index: properties,
            runtimeGetVersion=lambda: 13000,
            driverGetVersion=lambda: 13010,
        )
        cuda = SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            runtime=runtime,
            get_local_runtime_version=lambda: 13000,
            nvrtc=SimpleNamespace(getVersion=lambda: (13, 0)),
        )
        cp = SimpleNamespace(__version__="14.0.0", cuda=cuda)
        kernel = SimpleNamespace(
            METHOD_ID="method",
            KERNEL_ID="kick",
            FUSED_KERNEL_ID="fused",
            KICK_CUDA_SOURCE_SHA256="a" * 64,
            FUSED_CUDA_SOURCE_SHA256="b" * 64,
            IMPLEMENTATION_SOURCE_SHA256="c" * 64,
            ALL_CONIC_IMPLEMENTATION_SOURCE_SHA256="d" * 64,
            CUDA_OPTIONS=("--std=c++11",),
        )
        identity = m3._gpu_identity(cp, kernel)
        self.assertEqual(identity["backend"], "cupy")
        self.assertEqual(identity["device"], "cuda:0")
        self.assertEqual(identity["device_uuid"], "01" * 16)
        self.assertEqual(identity["device_compute_capability"], "12.0")
        self.assertEqual(identity["cuda_driver_version"], 13010)
        self.assertEqual(identity["nvrtc_version"], [13, 0])
        self.assertEqual(identity["kernel_id"], "kick")
        self.assertEqual(identity["fused_kernel_id"], "fused")

    def test_executed_archive_binds_protocol_and_execution_runtime(self) -> None:
        runtime_identity = {"unit_runtime": "exact"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            m3, "_runtime_identity", return_value=runtime_identity
        ):
            payload = m3._create_executed_archive(
                project_root=ROOT,
                work_directory=Path(temporary),
                expected_code_identity=self.protocol["code_identity"],
                expected_runtime_identity=runtime_identity,
            )
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = set(archive.namelist())
            self.assertIn("MANIFEST.json", names)
            self.assertIn(
                "source/src/jxplanetx/fast_wisdom_holman.py", names
            )
            self.assertIn(
                "source/src/jxplanetx/data/long_horizon_known_system_v2/"
                "MANIFEST.json",
                names,
            )
            manifest = json.loads(archive.read("MANIFEST.json"))
        self.assertTrue(manifest["runtime_matches_execution_receipt"])
        self.assertEqual(manifest["runtime_identity"], runtime_identity)
        self.assertEqual(
            manifest["runtime_identity_sha256"],
            hashlib.sha256(
                m3.RUNTIME_IDENTITY_DOMAIN
                + m3._canonical_json(runtime_identity)
            ).hexdigest(),
        )
        self.assertEqual(
            manifest["protocol_code_files_content_sha256"],
            self.protocol["code_identity"]["files_content_sha256"],
        )

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            m3, "_runtime_identity", return_value={"unit_runtime": "changed"}
        ):
            with self.assertRaises(m3.M3ProtocolError):
                m3._create_executed_archive(
                    project_root=ROOT,
                    work_directory=Path(temporary),
                    expected_code_identity=self.protocol["code_identity"],
                    expected_runtime_identity=runtime_identity,
                )

    def test_exact_directory_and_npz_rosters_reject_extra_links_and_members(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir()
            (root / "A").write_bytes(b"a")
            (root / "B").write_bytes(b"b")
            self.assertEqual(
                m3._read_exact_directory_payloads_no_follow(
                    root, {"A", "B"}, "unit evidence"
                ),
                {"A": b"a", "B": b"b"},
            )
            (root / "EXTRA").write_bytes(b"extra")
            with self.assertRaises(m3.M3ProtocolError):
                m3._read_exact_directory_payloads_no_follow(
                    root, {"A", "B"}, "unit evidence"
                )
            (root / "EXTRA").unlink()
            (root / "B").unlink()
            (root / "B").symlink_to(root / "A")
            with self.assertRaises(m3.M3ProtocolError):
                m3._read_exact_directory_payloads_no_follow(
                    root, {"A", "B"}, "unit evidence"
                )

        buffer = io.BytesIO()
        np.savez(buffer, exact=np.asarray([1.0], dtype=np.float64))
        payload = buffer.getvalue()
        decoded = m3._decode_npz_arrays_payload(
            payload, "unit NPZ", expected_names={"exact"}
        )
        np.testing.assert_array_equal(decoded["exact"], [1.0])
        expanded = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload), "r") as source, zipfile.ZipFile(
            expanded, "w"
        ) as destination:
            for info in source.infolist():
                destination.writestr(info.filename, source.read(info))
            destination.writestr("EXTRA.txt", b"forbidden")
        with self.assertRaises(m3.M3ProtocolError):
            m3._decode_npz_arrays_payload(
                expanded.getvalue(), "unit NPZ", expected_names={"exact"}
            )

    def test_executed_archive_verifier_rejects_extra_and_tampered_members(self) -> None:
        code = self.protocol["code_identity"]
        runtime_identity = {
            "local_files": {
                "file_count": code["file_count"],
                "files_content_sha256": code["files_content_sha256"],
                "imported_module_count_at_preparation": len(
                    code["imported_local_modules"]
                ),
            },
            "local_imported_modules": [],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            m3, "_runtime_identity", return_value=runtime_identity
        ):
            payload = m3._create_executed_archive(
                project_root=ROOT,
                work_directory=Path(temporary),
                expected_code_identity=code,
                expected_runtime_identity=runtime_identity,
            )
        manifest = m3._validate_executed_source_archive_payload(
            payload,
            self.protocol,
            expected_runtime_identity=runtime_identity,
            split_receipt=None,
        )
        self.assertTrue(manifest["runtime_matches_execution_receipt"])

        def rebuilt(*, extra: bool, tamper_source: bool) -> bytes:
            result = io.BytesIO()
            changed = False
            with zipfile.ZipFile(io.BytesIO(payload), "r") as source, zipfile.ZipFile(
                result, "w", compression=zipfile.ZIP_DEFLATED
            ) as destination:
                for info in source.infolist():
                    member_payload = source.read(info)
                    if (
                        tamper_source
                        and not changed
                        and info.filename.startswith("source/")
                    ):
                        member_payload = (
                            bytes([member_payload[0] ^ 1]) + member_payload[1:]
                        )
                        changed = True
                    destination.writestr(info.filename, member_payload)
                if extra:
                    destination.writestr("UNDECLARED", b"x")
            return result.getvalue()

        for forged in (
            rebuilt(extra=True, tamper_source=False),
            rebuilt(extra=False, tamper_source=True),
        ):
            with self.assertRaises(m3.M3ProtocolError):
                m3._validate_executed_source_archive_payload(
                    forged,
                    self.protocol,
                    expected_runtime_identity=runtime_identity,
                    split_receipt=None,
                )

    def test_split_event_hashes_are_derived_not_merely_equal(self) -> None:
        arbitrary = {
            "expected_event_ledger_sha256": "a" * 64,
            "split_event_ledger_sha256": "a" * 64,
        }
        with self.assertRaises(m3.M3ProtocolError):
            m3._require_checkpoint_event_ledger_crosslink(
                arbitrary, [], [], []
            )
        empty_digest = hashlib.sha256(
            m3.EVENT_DOMAIN
            + m3._canonical_json(
                {"route_records": [], "removal_records": []}
            )
        ).hexdigest()
        receipt = {
            "expected_event_ledger_sha256": empty_digest,
            "split_event_ledger_sha256": empty_digest,
        }
        self.assertEqual(
            m3._require_checkpoint_event_ledger_crosslink(
                receipt, [], [], []
            ),
            (empty_digest, empty_digest),
        )

    def test_incomplete_calibration_and_symlink_ancestor_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            for name in (
                m3.PROTOCOL_NAME,
                m3.ARRAY_ARCHIVE_NAME,
                m3.EVENT_LEDGER_NAME,
                m3.SOURCE_ARCHIVE_NAME,
                m3.CALIBRATION_RECORD_NAME,
            ):
                (incomplete / name).write_bytes(b"x")
            with self.assertRaisesRegex(m3.M3ProtocolError, "INCOMPLETE"):
                m3.validate_calibration_bundle(incomplete, self.protocol)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(m3.M3ProtocolError):
                m3._calibration_output_allowed(
                    linked_parent / "calibration", ROOT
                )

    def test_official_publication_stays_on_retained_directory_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt-anchor"
            created, descriptor = m3._mkdir_absolute_no_follow_create_only_open(
                attempt, "unit attempt"
            )
            self.assertEqual(created, attempt)
            moved = root / "moved-original"
            decoy = root / "decoy"
            decoy.mkdir()
            attempt.rename(moved)
            decoy.rename(attempt)
            try:
                m3._write_bytes_at_create_only(
                    descriptor, "SEALED.bin", b"retained-inode"
                )
                self.assertEqual(
                    (moved / "SEALED.bin").read_bytes(), b"retained-inode"
                )
                self.assertFalse((attempt / "SEALED.bin").exists())
                with self.assertRaises(m3.M3ProtocolError):
                    m3._require_directory_path_identity(
                        attempt, descriptor, "swapped attempt"
                    )
            finally:
                os.close(descriptor)

    def test_atomic_directory_publication_never_replaces_a_decoy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_descriptor = m3._open_absolute_directory_no_follow(
                root, "atomic publication parent"
            )
            original_publish = m3._rename_directory_no_replace_at

            def insert_decoy_then_publish(
                descriptor: int,
                source_name: str,
                destination_name: str,
                label: str,
            ) -> None:
                os.mkdir(destination_name, dir_fd=descriptor)
                original_publish(
                    descriptor, source_name, destination_name, label
                )

            try:
                with mock.patch.object(
                    m3,
                    "_rename_directory_no_replace_at",
                    side_effect=insert_decoy_then_publish,
                ):
                    with self.assertRaises(FileExistsError):
                        m3._create_directory_at_no_replace_open(
                            parent_descriptor,
                            "attempt-decoy",
                            "atomic publication test",
                        )
                self.assertEqual(os.listdir(parent_descriptor), ["attempt-decoy"])
                self.assertTrue((root / "attempt-decoy").is_dir())
            finally:
                os.close(parent_descriptor)

    @unittest.skipUnless(
        Path("/proc/self/fd").is_dir(), "Linux descriptor accounting is required"
    )
    def test_official_root_fsync_failure_leaks_no_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            attempt = canonical / "attempt-fsync-failure"
            before = len(os.listdir("/proc/self/fd"))
            with mock.patch.object(m3, "DEFAULT_CANONICAL_ROOT", canonical):
                with mock.patch.object(
                    m3.os, "fsync", side_effect=OSError("injected fsync failure")
                ):
                    with self.assertRaises(m3.M3ProtocolError):
                        m3._official_attempt_directory_open(ROOT, attempt)
            after = len(os.listdir("/proc/self/fd"))
            self.assertEqual(after, before)

    def test_guard_event_and_split_contracts_cannot_be_relaxed(self) -> None:
        route = self.protocol["routing_and_events"]
        self.assertEqual(
            float.fromhex(route["algorithmic_guard_radius_AU_hex"]), 0.1
        )
        self.assertEqual(
            [float.fromhex(value) for value in route["terminal_source_radii_AU_hex"]],
            [0.0] * 10,
        )
        self.assertEqual(route["route_worker_count"], 6)
        self.assertEqual(
            float.fromhex(route["maximum_primary_dynamical_phase_hex"]),
            1.0 / 64.0,
        )
        split = self.protocol["split_resume"]
        self.assertEqual(split["split_epoch_years"], 5)
        self.assertEqual(split["candidate_split_map_step"], 2_560)
        self.assertEqual(split["refinement_split_map_step"], 5_120)
        self.assertEqual(split["checkpoint_segment_years"], 1)
        self.assertEqual(split["checkpoint_count_per_lane"], 10)
        self.assertTrue(split["replay_load_at_split_and_final_head"])
        self.assertTrue(split["bitwise_terminal_state_equality_required"])
        self.assertTrue(split["lifecycle_and_event_ledger_equality_required"])

    def test_threshold_commitment_is_separate_and_fail_closed(self) -> None:
        record = m3.build_threshold_commitment(
            protocol_content_sha256=self.protocol["content_sha256"],
            calibration_content_sha256="a" * 64,
            comparisons=_threshold_limits(),
            review_id="unit-review-v1",
            attempt_id="attempt-unit-threshold",
        )
        self.assertEqual(
            m3.validate_threshold_commitment(record, self.protocol), record
        )
        with self.assertRaises(m3.M3ProtocolError):
            m3.validate_threshold_commitment(
                record,
                self.protocol,
                expected_attempt_id="attempt-someone-else",
            )
        changed = deepcopy(record)
        changed["comparisons"][m3.COMPARISON_IDS[0]][m3.COMPARISON_METRICS[0]] = (
            2.0
        ).hex()
        with self.assertRaises(m3.M3ProtocolError):
            m3.validate_threshold_commitment(changed, self.protocol)

        calibration = {
            "schema": m3.CALIBRATION_SCHEMA,
            "content_sha256": "b" * 64,
            "official_attempt_authorized": False,
        }
        with self.assertRaises(m3.M3ProtocolError):
            m3.validate_threshold_commitment(calibration, self.protocol)

        for mutate in (
            lambda row: row.__setitem__(
                "staged_one_thousand_year_run_authorized", True
            ),
            lambda row: row.pop("review_id"),
            lambda row: row.__setitem__("calibration_content_sha256", "0" * 64),
            lambda row: row.__setitem__("production_run_authorized", True),
            lambda row: row.__setitem__("review_id", " not-normalized "),
            lambda row: row.__setitem__("authorized_attempt_id", "attempt-A"),
            lambda row: row["comparisons"][m3.COMPARISON_IDS[0]].__setitem__(
                m3.COMPARISON_METRICS[0], "0X1.0000000000000P+0"
            ),
        ):
            forged = deepcopy(record)
            mutate(forged)
            forged["content_sha256"] = m3._content_sha256(
                m3.THRESHOLD_DOMAIN, forged
            )
            with self.assertRaises(m3.M3ProtocolError):
                m3.validate_threshold_commitment(forged, self.protocol)

    def test_official_runtime_must_equal_reviewed_calibration_runtime(self) -> None:
        identity = {"unit_runtime": "calibration-device-a"}
        digest = hashlib.sha256(
            m3.RUNTIME_IDENTITY_DOMAIN + m3._canonical_json(identity)
        ).hexdigest()
        runtime = {
            "runtime_identity_start": identity,
            "runtime_identity_end": deepcopy(identity),
            "runtime_identity_sha256": digest,
        }
        calibration = {"measured_analysis": {"runtime": deepcopy(runtime)}}
        self.assertEqual(
            m3._require_exact_calibration_runtime(calibration, runtime), digest
        )
        other_identity = {"unit_runtime": "official-device-b"}
        other = {
            "runtime_identity_start": other_identity,
            "runtime_identity_end": deepcopy(other_identity),
            "runtime_identity_sha256": hashlib.sha256(
                m3.RUNTIME_IDENTITY_DOMAIN
                + m3._canonical_json(other_identity)
            ).hexdigest(),
        }
        with self.assertRaises(m3.M3ProtocolError):
            m3._require_exact_calibration_runtime(calibration, other)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            path.write_text('{"a":1,"a":2}', encoding="ascii")
            with self.assertRaises(m3.M3ProtocolError):
                m3._read_json(path)
            path.write_text('{"a":NaN}', encoding="ascii")
            with self.assertRaises(m3.M3ProtocolError):
                m3._read_json(path)

    def test_state_and_element_metrics_are_individual_not_aggregate_only(self) -> None:
        nodes = 2
        position = np.zeros((nodes, m3.STATE_COUNT, 3), dtype=np.float64)
        velocity = np.zeros_like(position)
        radii = np.linspace(40.0, 80.0, m3.STATE_COUNT)
        position[:, :, 0] = radii
        velocity[:, :, 1] = np.sqrt(1.0 / radii)
        metrics = m3._comparison_per_state(
            position, velocity, position.copy(), velocity.copy(), 1.0
        )
        self.assertEqual(set(metrics), set(m3.COMPARISON_METRICS))
        for values in metrics.values():
            self.assertEqual(values.shape, (m3.STATE_COUNT,))
            self.assertTrue(np.array_equal(values, np.zeros(m3.STATE_COUNT)))

    def test_trace_node_hashes_recompute_from_exact_retained_bytes(self) -> None:
        position = np.zeros((2, m3.STATE_COUNT, 3), dtype=np.float64)
        velocity = np.zeros_like(position)
        lifecycle = np.zeros((2, m3.STATE_COUNT), dtype=np.uint8)
        state_ids = self.protocol["inputs"]["state_ids"]
        receipt = m3._trace_node_custody(
            lane_id="candidate",
            position_AU=position,
            velocity_AU_per_year=velocity,
            lifecycle=lifecycle,
            state_ids=state_ids,
            source_steps_per_map_step=m3.CANDIDATE_SOURCE_STEPS_PER_MAP,
        )
        self.assertEqual(receipt["node_count"], 2)
        self.assertEqual(
            receipt["content_sha256"],
            m3._content_sha256(m3.TRACE_CUSTODY_DOMAIN, receipt),
        )
        changed = position.copy()
        changed[1, 0, 0] = -0.0
        changed_receipt = m3._trace_node_custody(
            lane_id="candidate",
            position_AU=changed,
            velocity_AU_per_year=velocity,
            lifecycle=lifecycle,
            state_ids=state_ids,
            source_steps_per_map_step=m3.CANDIDATE_SOURCE_STEPS_PER_MAP,
        )
        self.assertNotEqual(
            receipt["node_sha256"][1], changed_receipt["node_sha256"][1]
        )

    def test_independent_interval_clearance_and_phase_are_certified(self) -> None:
        tracer_position = np.zeros((2, m3.STATE_COUNT, 3), dtype=np.float64)
        tracer_velocity = np.zeros_like(tracer_position)
        tracer_position[:, :, 0] = 40.0
        tracer_velocity[:, :, 1] = math.sqrt(1.0 / 40.0)
        source_position = np.zeros((2, m3.SOURCE_COUNT, 3), dtype=np.float64)
        source_position[:, :, 0] = np.arange(m3.SOURCE_COUNT) * 2.0
        source_velocity = np.zeros_like(source_position)
        gm = np.ones(m3.SOURCE_COUNT, dtype=np.float64)
        record = m3._independent_interval_guards(
            tracer_position,
            tracer_velocity,
            source_position,
            source_velocity,
            gm,
            m3.REFINEMENT_STEP_YEARS,
        )
        self.assertTrue(record["all_intervals_clear"])
        self.assertTrue(record["instantaneous_phase_passed"])
        self.assertTrue(record["periapse_phase_passed"])
        self.assertGreater(
            record["minimum_certified_interval_clearance_AU"], 0.1
        )
        self.assertTrue(
            record["source_pair_safety"]["all_source_pair_intervals_clear"]
        )
        overlapping = source_position.copy()
        overlapping[:, 1] = overlapping[:, 0]
        rejected = m3._independent_interval_guards(
            tracer_position,
            tracer_velocity,
            overlapping,
            source_velocity,
            gm,
            m3.REFINEMENT_STEP_YEARS,
        )
        self.assertFalse(
            rejected["source_pair_safety"]["all_source_pair_intervals_clear"]
        )

    def test_gate_evaluation_requires_every_state_family_and_structural_gate(self) -> None:
        thresholds = m3.build_threshold_commitment(
            protocol_content_sha256=self.protocol["content_sha256"],
            calibration_content_sha256="c" * 64,
            comparisons=_threshold_limits(),
            review_id="unit-review-v2",
            attempt_id="attempt-unit-gates",
        )
        analysis = _passing_analysis(self.protocol)
        gates = m3.evaluate_gates(analysis, thresholds, self.protocol)
        self.assertTrue(all(row["passed"] for row in gates.values()))
        metric_record = analysis["comparisons"][m3.COMPARISON_IDS[0]][
            "metrics"
        ][m3.COMPARISON_METRICS[0]]
        first_state = self.protocol["inputs"]["state_ids"][0]
        first_family = self.protocol["inputs"]["family_ids"][0]
        metric_record["per_state_hex"][first_state] = (2.0).hex()
        metric_record["per_family_maximum_hex"][first_family] = (2.0).hex()
        metric_record["global_maximum_hex"] = (2.0).hex()
        gates = m3.evaluate_gates(analysis, thresholds, self.protocol)
        self.assertFalse(
            gates[
                f"{m3.COMPARISON_IDS[0]}.{m3.COMPARISON_METRICS[0]}"
            ]["passed"]
        )

        inconsistent_runtime = _passing_analysis(self.protocol)
        inconsistent_runtime["runtime"]["runtime_identity_end"] = {
            "unit_runtime": "changed"
        }
        with self.assertRaisesRegex(
            m3.M3ProtocolError, "runtime custody receipt is inconsistent"
        ):
            m3.evaluate_gates(
                inconsistent_runtime, thresholds, self.protocol
            )

        exact = _passing_analysis(self.protocol)
        exact_gates = m3.evaluate_gates(exact, thresholds, self.protocol)
        self.assertTrue(exact_gates["initial_state_bitwise_exact"]["passed"])
        exact["initial_state_custody"]["rebound_ias15"][
            "tracer_positions_sha256"
        ] = "d" * 64
        changed_gates = m3.evaluate_gates(exact, thresholds, self.protocol)
        self.assertFalse(
            changed_gates["initial_state_bitwise_exact"]["passed"]
        )

    def test_zero_event_lifecycle_accounting_is_exact(self) -> None:
        trace = _event_trace(
            np.zeros((3, m3.STATE_COUNT), dtype=np.uint8)
        )
        record = m3._event_accounting(trace)
        self.assertTrue(record["exact"])
        self.assertEqual(record["route_record_count"], 0)
        self.assertEqual(record["removal_record_count"], 0)

    @staticmethod
    def _event_witness(
        *, source_steps_per_map: int = 8, interval_fraction: float = 0.25
    ) -> object:
        from jxplanetx.restricted_wh_route import RestrictedWHRouteEvent

        return RestrictedWHRouteEvent(
            event_type="PRIMARY_BOUNDARY_OUTWARD",
            source_index=0,
            source_id="Sun",
            map_step_fraction=interval_fraction / source_steps_per_map,
            source_interval_offset=0,
            interval_fraction=interval_fraction,
            relative_position_AU=np.asarray(
                [m3.PRIMARY_OUTER_BOUNDARY_AU, 0.0, 0.0], dtype=np.float64
            ),
            relative_velocity_AU_per_year=np.asarray(
                [1.0, 0.0, 0.0], dtype=np.float64
            ),
            surface_distance_AU=m3.PRIMARY_OUTER_BOUNDARY_AU,
            surface_threshold_AU=m3.PRIMARY_OUTER_BOUNDARY_AU,
            surface_residual_AU=0.0,
            surface_tolerance_AU=1.0e-12,
            primary_specific_energy_AU2_per_year2=-0.5,
            primary_radial_velocity_AU_per_year=1.0,
            surface_radial_velocity_AU_per_year=1.0,
            location_method="EXACT_DENSE_NODE",
        )

    @classmethod
    def _removal_row(
        cls,
        *,
        lane: int,
        tracer: int,
        map_step: int,
        source_steps_per_map: int = 8,
    ) -> dict[str, object]:
        from jxplanetx.restricted_wh_lifecycle import RestrictedWHRemovalRecord

        record = RestrictedWHRemovalRecord(
            lane_index=lane,
            tracer_index=tracer,
            lifecycle_code=2,
            global_source_step=map_step * source_steps_per_map,
            source_steps_per_map_step=source_steps_per_map,
            guard_phase="DRIFT",
            guard_reason="MINIMUM_SOURCE_DISTANCE",
            guard_source_index=1,
            source_segment_sha256="a" * 64,
            route_runtime_identity_sha256="b" * 64,
            accepted_step_count=4,
            derivative_evaluation_count=61,
            minimum_committed_stage_distance_AU=0.1,
            maximum_committed_primary_dynamical_phase_radians=1.0e-3,
            events=(cls._event_witness(source_steps_per_map=source_steps_per_map),),
        )
        return m3._removal_record(record)

    @classmethod
    def _route_row(
        cls,
        *,
        lane: int,
        tracer: int,
        map_step: int,
        source_steps_per_map: int = 8,
        events: tuple[object, ...] = (),
    ) -> dict[str, object]:
        route = SimpleNamespace(
            runtime_identity={
                "source_tile_sha256": "c" * 64,
                "method_id": "UNIT_TEST_ROUTE",
            },
            source_tile_sha256="c" * 64,
            source_segment_sha256="d" * 64,
            global_source_step=map_step * source_steps_per_map,
            source_steps_per_map_step=source_steps_per_map,
            accepted_step_count=4,
            derivative_evaluation_count=61,
            endpoint_relative_position_AU=np.asarray(
                [1.0, 2.0, 3.0], dtype=np.float64
            ),
            endpoint_relative_velocity_AU_per_year=np.asarray(
                [0.01, 0.02, 0.03], dtype=np.float64
            ),
            minimum_evaluated_stage_distance_AU=0.1,
            minimum_committed_stage_distance_AU=0.2,
            maximum_accepted_primary_dynamical_phase_radians=1.0e-3,
            maximum_committed_primary_dynamical_phase_radians=5.0e-4,
            completed_map_fraction=1.0,
            events=events,
        )
        row = SimpleNamespace(
            route_result=route,
            lane_index=lane,
            tracer_index=tracer,
            guard_phase="DRIFT",
            guard_reason="MINIMUM_SOURCE_DISTANCE",
            guard_source_index=1,
        )
        return m3._route_record(row, global_map_step=map_step)

    def test_route_and_removal_serialize_every_event_field(self) -> None:
        event = self._event_witness()
        route = self._route_row(
            lane=0, tracer=0, map_step=0, events=(event,)
        )
        removal = self._removal_row(lane=0, tracer=0, map_step=0)
        self.assertEqual(route["event_count"], 1)
        self.assertEqual(removal["event_count"], 1)
        self.assertEqual(set(route["event_content"][0]), m3._ROUTED_EVENT_KEYS)
        self.assertEqual(route["event_content"], removal["event_content"])

    def test_algorithmic_guard_cannot_be_reported_as_physical_contact(self) -> None:
        event = self._event_witness()
        forged = m3._routed_event_record(event)
        forged["event_type"] = "SOURCE_RADIUS_CONTACT"
        forged["source_index"] = 1
        forged["source_id"] = "Mercury Barycenter"
        forged["surface_distance_AU_hex"] = (0.1).hex()
        forged["surface_threshold_AU_hex"] = (0.1).hex()
        forged["surface_radial_velocity_AU_per_year_hex"] = (-1.0).hex()
        with self.assertRaisesRegex(
            m3.M3NumericalError, "disabled physical contact"
        ):
            m3._validate_routed_events(
                [forged], source_steps_per_map_step=8, label="forged"
            )

    def test_terminal_event_is_bijective_with_lifecycle_transition(self) -> None:
        lifecycle = np.zeros((3, m3.STATE_COUNT), dtype=np.uint8)
        lifecycle[2:, 0] = 2
        trace = _event_trace(
            lifecycle,
            removals=[self._removal_row(lane=0, tracer=0, map_step=1)],
        )
        record = m3._event_accounting(trace)
        self.assertTrue(record["exact"])
        self.assertEqual(record["lifecycle_transition_count"], 1)
        self.assertEqual(record["terminal_lane_count"], 1)

        wrong_lane = deepcopy(trace)
        wrong_lane["removal_records"][0]["lane_index"] = m3.STATE_COUNT - 1
        with self.assertRaises(m3.M3NumericalError):
            m3._event_accounting(wrong_lane)

        wrong_code = deepcopy(trace)
        wrong_code["removal_records"][0]["lifecycle_code"] = 1
        with self.assertRaisesRegex(
            m3.M3NumericalError, "lifecycle code disagree"
        ):
            m3._event_accounting(wrong_code)

        forged_hash = deepcopy(trace)
        forged_hash["removal_records"][0]["content_sha256"] = "f" * 64
        with self.assertRaisesRegex(m3.M3NumericalError, "does not bind"):
            m3._event_accounting(forged_hash)

    def test_event_accounting_rejects_bad_counts_ranges_order_and_uniqueness(
        self,
    ) -> None:
        lifecycle = np.zeros((3, m3.STATE_COUNT), dtype=np.uint8)
        route_0 = self._route_row(lane=0, tracer=0, map_step=0)
        route_1 = self._route_row(lane=1, tracer=1, map_step=1)
        valid_trace = _event_trace(lifecycle, routes=[route_0, route_1])
        self.assertTrue(m3._event_accounting(valid_trace)["exact"])

        cases: dict[str, dict[str, object]] = {}
        bad_count = deepcopy(valid_trace)
        bad_count["route_records"][0]["event_count"] = 1
        cases["event_count"] = bad_count
        bad_lane = deepcopy(valid_trace)
        bad_lane["route_records"][0]["lane_index"] = m3.STATE_COUNT
        cases["lane"] = bad_lane
        bad_map = deepcopy(valid_trace)
        bad_map["route_records"][1]["global_map_step"] = 2
        bad_map["route_records"][1]["global_source_step"] = 16
        cases["map"] = bad_map
        bad_order = deepcopy(valid_trace)
        bad_order["route_records"].reverse()
        cases["order"] = bad_order
        duplicate = deepcopy(valid_trace)
        duplicate["route_records"] = [route_0, deepcopy(route_0)]
        cases["unique"] = duplicate
        terminal_route = deepcopy(valid_trace)
        terminal_route["route_records"][0] = self._route_row(
            lane=0,
            tracer=0,
            map_step=0,
            events=(self._event_witness(),),
        )
        cases["terminal route"] = terminal_route
        for label, trace in cases.items():
            with self.subTest(label=label), self.assertRaises(
                m3.M3NumericalError
            ):
                m3._event_accounting(trace)

    def test_removal_ledger_rejects_bad_order_and_duplicate_lane(self) -> None:
        lifecycle = np.zeros((3, m3.STATE_COUNT), dtype=np.uint8)
        lifecycle[1:, 0] = 2
        lifecycle[2:, 1] = 2
        first = self._removal_row(lane=0, tracer=0, map_step=0)
        second = self._removal_row(lane=1, tracer=1, map_step=1)
        trace = _event_trace(lifecycle, removals=[first, second])
        self.assertTrue(m3._event_accounting(trace)["exact"])

        reversed_trace = deepcopy(trace)
        reversed_trace["removal_records"].reverse()
        with self.assertRaisesRegex(m3.M3NumericalError, "canonically ordered"):
            m3._event_accounting(reversed_trace)

        duplicate_trace = deepcopy(trace)
        duplicate_trace["removal_records"] = [first, deepcopy(first)]
        with self.assertRaises(m3.M3NumericalError):
            m3._event_accounting(duplicate_trace)

    def test_calibration_entrypoint_rejects_injected_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "calibration.json"
            with self.assertRaises(TypeError):
                m3.run_calibration(
                    project_root=ROOT,
                    output_path=output,
                    numerical_executor=mock.Mock(),
                )
            self.assertFalse(output.exists())

    def test_calibration_rejects_canonical_namespace_before_execution(self) -> None:
        target = ROOT / m3.DEFAULT_CANONICAL_ROOT / "attempt-bad" / "calibration.json"
        with self.assertRaises(m3.M3ProtocolError):
            m3.run_calibration(
                project_root=ROOT,
                output_path=target,
            )

    def test_official_entrypoint_rejects_injected_numerics_and_attempts_are_create_only(self) -> None:
        thresholds = m3.build_threshold_commitment(
            protocol_content_sha256=self.protocol["content_sha256"],
            calibration_content_sha256="d" * 64,
            comparisons=_threshold_limits(),
            review_id="unit-review-v3",
            attempt_id="attempt-unit-success",
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            canonical = temp / "canonical"
            attempt = canonical / "attempt-unit-success"
            threshold_path = temp / "thresholds.json"
            threshold_path.write_text(json.dumps(thresholds), encoding="ascii")
            with self.assertRaises(TypeError):
                m3.run_official(
                    project_root=ROOT,
                    threshold_path=threshold_path,
                    attempt_directory=attempt,
                    numerical_executor=mock.Mock(),
                )
            with mock.patch.object(m3, "DEFAULT_CANONICAL_ROOT", canonical):
                created, descriptor = m3._official_attempt_directory_open(
                    ROOT, attempt
                )
            try:
                self.assertEqual(created, attempt)
                m3._require_directory_path_identity(
                    attempt, descriptor, "unit official attempt"
                )
            finally:
                os.close(descriptor)
            with self.assertRaises(m3.M3ProtocolError):
                with mock.patch.object(m3, "DEFAULT_CANONICAL_ROOT", canonical):
                    m3._official_attempt_directory(ROOT, attempt)

    def test_official_numerical_failure_record_is_explicit_and_nonauthorizing(self) -> None:
        thresholds = m3.build_threshold_commitment(
            protocol_content_sha256=self.protocol["content_sha256"],
            calibration_content_sha256="e" * 64,
            comparisons=_threshold_limits(),
            review_id="unit-review-v4",
            attempt_id="attempt-unit-failure",
        )
        result = m3._failure_result(
            protocol=self.protocol,
            thresholds=thresholds,
            calibration={"content_sha256": "e" * 64},
            calibration_files=(),
            error=RuntimeError("mock numerical failure"),
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["all_gates_passed"])
        self.assertEqual(result["failure"]["type"], "RuntimeError")
        self.assertTrue(all(value is False for value in result["claims"].values()))

    def test_interrupted_attempt_has_explicit_incomplete_status(self) -> None:
        thresholds = m3.build_threshold_commitment(
            protocol_content_sha256=self.protocol["content_sha256"],
            calibration_content_sha256="f" * 64,
            comparisons=_threshold_limits(),
            review_id="unit-review-v5",
            attempt_id="attempt-unit-incomplete",
        )
        result = m3._failure_result(
            protocol=self.protocol,
            thresholds=thresholds,
            calibration={"content_sha256": "f" * 64},
            calibration_files=(),
            error=KeyboardInterrupt(),
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["all_gates_passed"])
        self.assertFalse(result["staged_one_thousand_year_run_authorized"])
        self.assertTrue(all(value is False for value in result["claims"].values()))

    def test_final_official_verifier_rejects_extra_symlink_and_tamper(self) -> None:
        calibration_digest = "c" * 64

        def build_attempt(
            root: Path, name: str, *, completed: bool
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object]
        ]:
            attempt = root / name
            attempt.mkdir()
            thresholds = m3.build_threshold_commitment(
                protocol_content_sha256=self.protocol["content_sha256"],
                calibration_content_sha256=calibration_digest,
                comparisons=_threshold_limits(),
                review_id=f"review-{name}",
                attempt_id=name,
            )
            m3._write_json_create_only(attempt / m3.PROTOCOL_NAME, self.protocol)
            m3._write_json_create_only(attempt / "THRESHOLDS.json", thresholds)
            calibration_rows = []
            for _source, retained in m3.OFFICIAL_CALIBRATION_COPY_NAMES:
                payload = f"retained:{retained}\n".encode("ascii")
                m3._write_bytes_create_only(attempt / retained, payload)
                calibration_rows.append(
                    {
                        "path": retained,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            m3._write_bytes_create_only(
                attempt / m3.SOURCE_ARCHIVE_NAME, b"archive"
            )
            result = m3._failure_result(
                protocol=self.protocol,
                thresholds=thresholds,
                calibration={"content_sha256": calibration_digest},
                calibration_files=calibration_rows,
                error=RuntimeError("unit failure"),
            )
            m3._write_json_create_only(attempt / m3.RESULT_NAME, result)
            retained_order = [
                m3.PROTOCOL_NAME,
                "THRESHOLDS.json",
                *(retained for _, retained in m3.OFFICIAL_CALIBRATION_COPY_NAMES),
                m3.SOURCE_ARCHIVE_NAME,
                m3.RESULT_NAME,
            ]
            artifacts = m3._artifact_inventory(attempt, retained_order)
            m3._write_json_create_only(attempt / m3.ARTIFACT_NAME, artifacts)
            if completed:
                completion = m3._completion_record(
                    attempt, self.protocol, result, artifacts
                )
                m3._write_json_create_only(
                    attempt / m3.COMPLETION_NAME, completion
                )
            return thresholds, result, artifacts

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            m3,
            "_validate_retained_calibration_payloads",
            return_value={"content_sha256": calibration_digest},
        ), mock.patch.object(
            m3, "_validate_executed_source_archive_payload", return_value={}
        ):
            root = Path(temporary)
            thresholds, result, artifacts = build_attempt(
                root, "attempt-clean-incomplete", completed=False
            )
            verified = m3.validate_official_attempt(
                root / "attempt-clean-incomplete",
                self.protocol,
                thresholds,
                require_completed=False,
            )
            self.assertEqual(verified["status"], "INCOMPLETE")
            self.assertEqual(
                verified["result_content_sha256"], result["content_sha256"]
            )
            self.assertEqual(
                verified["artifacts_content_sha256"], artifacts["content_sha256"]
            )

            thresholds, _result, _artifacts = build_attempt(
                root, "attempt-clean-completed", completed=True
            )
            self.assertEqual(
                m3.validate_official_attempt(
                    root / "attempt-clean-completed",
                    self.protocol,
                    thresholds,
                    require_completed=True,
                )["status"],
                "COMPLETED",
            )

            thresholds, _result, _artifacts = build_attempt(
                root, "attempt-extra", completed=False
            )
            (root / "attempt-extra" / "EXTRA").write_bytes(b"x")
            with self.assertRaises(m3.M3ProtocolError):
                m3.validate_official_attempt(
                    root / "attempt-extra",
                    self.protocol,
                    thresholds,
                    require_completed=False,
                )

            thresholds, _result, _artifacts = build_attempt(
                root, "attempt-symlink", completed=False
            )
            archive = root / "attempt-symlink" / m3.SOURCE_ARCHIVE_NAME
            outside = root / "outside-archive"
            outside.write_bytes(b"archive")
            archive.unlink()
            archive.symlink_to(outside)
            with self.assertRaises(m3.M3ProtocolError):
                m3.validate_official_attempt(
                    root / "attempt-symlink",
                    self.protocol,
                    thresholds,
                    require_completed=False,
                )

            thresholds, _result, _artifacts = build_attempt(
                root, "attempt-tamper", completed=False
            )
            retained = m3.OFFICIAL_CALIBRATION_COPY_NAMES[0][1]
            (root / "attempt-tamper" / retained).write_bytes(b"tampered")
            with self.assertRaises(m3.M3ProtocolError):
                m3.validate_official_attempt(
                    root / "attempt-tamper",
                    self.protocol,
                    thresholds,
                    require_completed=False,
                )


if __name__ == "__main__":
    unittest.main()
