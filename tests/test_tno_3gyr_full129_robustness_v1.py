from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "benchmarks/jx_tno_3gyr_full129_robustness_v1.py"


def _load_source(name: str = "jx_tno_3gyr_full129_robustness_v1_tested"):
    payload = SOURCE_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE_PATH)
    sys.modules[name] = module
    try:
        exec(
            compile(
                payload,
                str(SOURCE_PATH),
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    if not root.exists():
        return ()
    records: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((relative, "symlink", 0, os.readlink(path)))
        elif path.is_dir():
            records.append((relative, "directory", 0, ""))
        else:
            payload = path.read_bytes()
            records.append((relative, "file", len(payload), _sha256(payload)))
    return tuple(records)


CHILD = r'''
from pathlib import Path
import sys
import types

source = Path(sys.argv[1]).resolve()
name = "jx_tno_3gyr_full129_robustness_v1_child"
module = types.ModuleType(name)
module.__file__ = str(source)
sys.modules[name] = module
exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
root = Path(sys.argv[3])
if sys.argv[2] == "prepare":
    value = module.prepare(root)
elif sys.argv[2] == "validate":
    value = module.validate_prepared(root)
else:
    raise RuntimeError("unknown child command")
print(value["content_sha256"])
'''


class TNOThreeGigayearFull129RobustnessV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_source()
        cls.protocol = cls.module.build_protocol()

    def test_import_and_build_have_no_gpu_network_or_execution_surface(self) -> None:
        before = set(sys.modules)
        module = _load_source(
            f"jx_tno_3gyr_source_only_{sys.flags.optimize}_{id(self)}"
        )
        added = set(sys.modules) - before
        for package in ("cupy", "numpy", "rebound"):
            self.assertFalse(
                any(name == package or name.startswith(f"{package}.") for name in added),
                package,
            )
        for name in ("execute", "execute_worker", "resume", "validate_result"):
            self.assertFalse(hasattr(module, name), name)

        observed_commands: list[list[str]] = []
        real_run = module.subprocess.run

        def record_run(*args, **kwargs):
            observed_commands.append(list(args[0]))
            return real_run(*args, **kwargs)

        with mock.patch.object(module.subprocess, "run", side_effect=record_run):
            rebuilt = module.build_protocol()
        self.assertTrue(module._exact_equal(rebuilt, self.protocol))
        self.assertTrue(observed_commands)
        for command in observed_commands:
            self.assertGreaterEqual(len(command), 3)
            self.assertEqual(command[0], "git")
            self.assertEqual(command[1], "-C")
            self.assertEqual(command[2], os.fspath(PROJECT_ROOT))

    def test_protocol_is_canonical_deterministic_and_prepare_only(self) -> None:
        m = self.module
        second = m.build_protocol()
        self.assertTrue(m._exact_equal(self.protocol, second))
        self.assertEqual(
            self.protocol["content_sha256"],
            m._domain_hash(m.PROTOCOL_DOMAIN, self.protocol),
        )
        self.assertEqual(m._canonical_json(self.protocol), m._canonical_json(second))
        self.assertEqual(
            self.protocol["status"],
            "PREREGISTRATION_CANDIDATE_PREPARE_ONLY_EXECUTION_NOT_IMPLEMENTED",
        )
        self.assertEqual(
            self.protocol["evidence_class"],
            "PREPARED_PREREGISTRATION_NO_MODEL_OUTPUT",
        )
        self.assertEqual(
            self.protocol["claim_ceiling"],
            "PREPARED_PROTOCOL_ONLY_NO_EXECUTION_OR_SCIENCE_RESULT",
        )
        self.assertEqual(self.protocol["future_result_evidence_class"], "MODEL_OUTPUT")
        self.assertEqual(
            self.protocol["future_result_claim_ceiling"],
            "TWELVE_OBJECT_1025_FORCING_CONTEXT_FORWARD_3GYR_CHAOTIC_ENSEMBLE_MODEL_OUTPUT_"
            "CONDITIONAL_ON_FROZEN_INITIAL_COVARIANCE_FORCE_MODEL_AND_RUNTIME",
        )
        self.assertEqual(
            self.protocol["public_cli"],
            ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        )
        self.assertEqual(
            self.protocol["explicitly_absent_public_cli"],
            ["execute", "execute-worker", "resume", "validate-result"],
        )
        self.assertIs(
            self.protocol["preregistration"]
            ["no_automatic_git_network_gpu_or_execution_action"],
            True,
        )
        preregistration = self.protocol["preregistration"]
        self.assertEqual(preregistration["annotated_tag"], self.module.PREREGISTRATION_TAG)
        self.assertEqual(preregistration["future_result_tag"], self.module.FUTURE_RESULT_TAG)
        self.assertEqual(
            preregistration["execution_successor_must_pin"],
            [
                "ANNOTATED_PREREGISTRATION_TAG_OBJECT",
                "TAG_PEELED_COMMIT",
                "SOURCE_BYTES_AND_SHA256",
                "TEST_BYTES_AND_SHA256",
                "PROTOCOL_BYTES_CONTENT_SHA256_AND_FILE_SHA256",
            ],
        )

    def test_time_contract_is_exactly_three_billion_julian_years(self) -> None:
        time = self.protocol["time_contract"]
        self.assertEqual(time["direction"], "FORWARD")
        self.assertIs(type(time["duration_julian_years"]), int)
        self.assertIs(type(time["duration_julian_days"]), int)
        self.assertEqual(time["duration_julian_years"], 3_000_000_000)
        self.assertEqual(time["duration_julian_days"], 1_095_750_000_000)
        self.assertEqual(
            time["duration_julian_days"] * 100,
            time["duration_julian_years"] * 36_525,
        )
        self.assertEqual(
            time["time_representation"],
            "SIGNED_INT64_GLOBAL_STEP_PLUS_EXACT_BINARY_POWER_OF_TWO_STEP",
        )
        self.assertIs(time["floating_point_time_accumulation_forbidden"], True)
        self.assertNotIn("step_budget", self.protocol)
        self.assertNotIn("selection", self.protocol)

    def test_all_twelve_full129_batches_are_bound_without_selection(self) -> None:
        roster = self.protocol["roster"]
        self.assertEqual(
            roster["policy"],
            "ALL_TWELVE_FROZEN_BATCHES_NO_SELECTION_OR_SUBSTITUTION",
        )
        self.assertEqual(roster["batch_count"], 12)
        self.assertEqual(roster["tracers_per_batch"], 129)
        self.assertEqual(roster["trajectory_count"], 1_548)
        self.assertEqual(roster["forcing_context_count"], 1_025)
        self.assertEqual(
            roster["trajectory_instances_across_forcing_contexts"],
            1_586_700,
        )
        self.assertEqual(
            roster["trajectory_instances_across_forcing_contexts"],
            roster["trajectory_count"] * roster["forcing_context_count"],
        )
        self.assertEqual(roster["nominal_trajectories"], 12)
        self.assertEqual(roster["covariance_clones_per_batch"], 128)
        self.assertIs(
            roster["clones_are_alternative_massless_worlds_not_simultaneous_bodies"],
            True,
        )
        records = roster["records"]
        self.assertEqual(len(records), 12)
        self.assertEqual(
            [
                (
                    row["batch_order"],
                    row["object_id"],
                    row["spk_id"],
                    row["designation"],
                    row["cohort"],
                )
                for row in records
            ],
            [
                (
                    target.batch_order,
                    target.object_id,
                    target.spk_id,
                    target.designation,
                    target.cohort,
                )
                for target in self.module.TARGETS
            ],
        )
        for row in records:
            self.assertIs(row["included"], True)
            self.assertNotIn("selected", row)

    def test_common_epoch_is_a_candidate_and_bridge_is_blocked(self) -> None:
        time = self.protocol["time_contract"]
        self.assertEqual(time["candidate_common_epoch_jd_tdb"], "2461200.5")
        self.assertEqual(
            time["candidate_epoch_basis"],
            "STANDARD_EPOCH_RECORDED_FOR_ALL_TWELVE_SBDB_CUSTODY_ROWS",
        )
        self.assertIs(time["native_covariance_epochs_are_object_specific"], True)
        self.assertIs(
            time["qualified_full_covariance_propagation_to_common_epoch_required"],
            True,
        )
        self.assertIn(
            "COMMON_EPOCH_FULL_COVARIANCE_PROPAGATION_NOT_QUALIFIED",
            self.protocol["execution_blockers"],
        )
        phases = self.protocol["mandatory_phase_order"]
        self.assertLess(
            phases.index(
                "FREEZE_AND_QUALIFY_COMMON_EPOCH_AND_COMPLETE_3GYR_PHYSICAL_MODEL_LADDER"
            ),
            phases.index(
                "IMPLEMENT_AND_QUALIFY_H64_H128_3GYR_ADAPTERS_WITH_INT64_STEPS"
            ),
        )

    def test_exact_dual_resolution_work_and_replay_arithmetic(self) -> None:
        work = self.protocol["work_geometry"]
        expected = {
            "h64_steps_per_batch": 192_000_000_000,
            "h128_steps_per_batch": 384_000_000_000,
            "forcing_context_count": 1_025,
            "per_context_h64_batch_map_steps": 2_304_000_000_000,
            "per_context_h128_batch_map_steps": 4_608_000_000_000,
            "per_context_dual_resolution_batch_map_steps": 6_912_000_000_000,
            "ensemble_h64_batch_map_steps": 2_361_600_000_000_000,
            "ensemble_h128_batch_map_steps": 4_723_200_000_000_000,
            "ensemble_dual_resolution_batch_map_steps": 7_084_800_000_000_000,
            "ensemble_maximum_all_lanes_active_tracer_step_capacity": 913_939_200_000_000_000,
            "per_context_canonical_shared_source_steps": 576_000_000_000,
            "ensemble_canonical_shared_source_steps": 590_400_000_000_000,
            "ensemble_canonical_shared_source_advances_with_complete_exact_replay": 1_180_800_000_000_000,
            "h64_shards_per_batch": 300_000,
            "h128_shards_per_batch": 600_000,
            "per_context_h64_total_batch_shards": 3_600_000,
            "per_context_h128_total_batch_shards": 7_200_000,
            "per_context_dual_resolution_total_batch_shards": 10_800_000,
            "ensemble_h64_total_batch_shards": 3_690_000_000,
            "ensemble_h128_total_batch_shards": 7_380_000_000,
            "ensemble_dual_resolution_total_batch_shards": 11_070_000_000,
            "h64_scheduled_physical_fragments_per_batch": 300_002,
            "h128_scheduled_physical_fragments_per_batch": 600_001,
            "ensemble_scheduled_physical_batch_fragment_count_with_prefix_stops": 11_070_036_900,
            "ensemble_canonical_source_tile_count": 922_500_000,
            "ensemble_unique_canonical_source_physical_fragment_count_with_prefix_stops": 922_503_075,
            "ensemble_exact_replay_canonical_source_physical_fragment_attempts_before_retries": 1_845_006_150,
            "distributed_context_resolution_batch_worker_units": 24_600,
            "source_steps_per_logical_640k_tile": 640_000,
            "h64_logical_640k_tile_julian_years": 10_000,
            "h128_logical_640k_tile_julian_years": 5_000,
        }
        for key, value in expected.items():
            self.assertIs(type(work[key]), int, key)
            self.assertEqual(work[key], value, key)
        self.assertEqual(
            work["h64_steps_per_batch"],
            self.protocol["time_contract"]["duration_julian_years"] * 64,
        )
        self.assertEqual(
            work["h128_steps_per_batch"],
            self.protocol["time_contract"]["duration_julian_years"] * 128,
        )
        self.assertEqual(
            work["per_context_dual_resolution_batch_map_steps"],
            work["per_context_h64_batch_map_steps"]
            + work["per_context_h128_batch_map_steps"],
        )
        self.assertEqual(
            work["ensemble_h64_batch_map_steps"],
            work["per_context_h64_batch_map_steps"] * work["forcing_context_count"],
        )
        self.assertEqual(
            work["ensemble_h128_batch_map_steps"],
            work["per_context_h128_batch_map_steps"] * work["forcing_context_count"],
        )
        self.assertEqual(
            work["ensemble_maximum_all_lanes_active_tracer_step_capacity"],
            work["ensemble_dual_resolution_batch_map_steps"] * 129,
        )
        self.assertEqual(
            work["per_context_canonical_shared_source_steps"],
            work["h64_steps_per_batch"] + work["h128_steps_per_batch"],
        )
        self.assertEqual(
            work["ensemble_canonical_shared_source_steps"],
            work["per_context_canonical_shared_source_steps"]
            * work["forcing_context_count"],
        )
        self.assertEqual(
            work["ensemble_canonical_shared_source_advances_with_complete_exact_replay"],
            work["ensemble_canonical_shared_source_steps"] * 2,
        )
        self.assertEqual(
            work["ensemble_canonical_source_tile_count"],
            (work["h64_shards_per_batch"] + work["h128_shards_per_batch"])
            * work["forcing_context_count"],
        )
        self.assertIs(work["batch_shard_counts_are_640k_logical_equivalents"], True)
        self.assertEqual(
            work["h64_scheduled_physical_fragments_per_batch"],
            work["h64_shards_per_batch"] + 2,
        )
        self.assertEqual(
            work["h128_scheduled_physical_fragments_per_batch"],
            work["h128_shards_per_batch"] + 1,
        )
        self.assertEqual(
            work["ensemble_scheduled_physical_batch_fragment_count_with_prefix_stops"],
            (
                work["h64_scheduled_physical_fragments_per_batch"]
                + work["h128_scheduled_physical_fragments_per_batch"]
            )
            * 12
            * work["forcing_context_count"],
        )
        self.assertIs(
            work["canonical_source_tile_count_is_640k_logical_equivalents"],
            True,
        )
        self.assertEqual(
            work["ensemble_unique_canonical_source_physical_fragment_count_with_prefix_stops"],
            (
                work["h64_scheduled_physical_fragments_per_batch"]
                + work["h128_scheduled_physical_fragments_per_batch"]
            )
            * work["forcing_context_count"],
        )
        self.assertEqual(
            work[
                "ensemble_exact_replay_canonical_source_physical_fragment_attempts_before_retries"
            ],
            work[
                "ensemble_unique_canonical_source_physical_fragment_count_with_prefix_stops"
            ]
            * 2,
        )
        self.assertEqual(
            work["distributed_context_resolution_batch_worker_units"],
            work["forcing_context_count"] * 2 * 12,
        )
        self.assertIs(
            work["actual_attempted_tracer_steps_depend_on_frozen_terminal_lifecycle"],
            True,
        )
        self.assertIs(
            work["logical_640k_tile_within_provider_one_million_step_ceiling"],
            True,
        )
        self.assertNotIn("dual_resolution_batch_map_steps", work)
        self.assertNotIn("dual_resolution_total_batch_shards", work)
        self.assertIs(
            work["qualification_failed_and_crash_replay_work_is_additional"],
            True,
        )

    def test_stop_schedules_are_exact_integer_int64_coordinates(self) -> None:
        work = self.protocol["work_geometry"]
        stop_years = [
            1_000,
            5_000,
            10_000,
            100_000,
            500_000,
            5_000_000,
            10_000_000,
            100_000_000,
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
        ]
        signed_int64_max = 2**63 - 1
        for resolution, steps_per_year in (("h64", 64), ("h128", 128)):
            expected = []
            previous_step = 0
            for year in stop_years:
                global_step = year * steps_per_year
                delta_steps = global_step - previous_step
                expected.append({
                    "julian_year": year,
                    "global_step": global_step,
                    "completed_640k_shards_per_batch": global_step // 640_000,
                    "remainder_steps_in_current_fragment": global_step % 640_000,
                    "newly_completed_logical_640k_shards_since_previous_stop": (
                        global_step // 640_000 - previous_step // 640_000
                    ),
                    "partial_fragment_remainder_steps_since_previous_stop": (
                        delta_steps % 640_000
                    ),
                })
                previous_step = global_step
            observed = work[f"{resolution}_stop_schedule"]
            self.assertEqual(observed, expected)
            self.assertEqual(
                [
                    record["newly_completed_logical_640k_shards_since_previous_stop"]
                    for record in observed[:3]
                ],
                [0, 0, 1] if resolution == "h64" else [0, 1, 1],
            )
            for record in observed:
                self.assertEqual(
                    tuple(record),
                    (
                        "julian_year",
                        "global_step",
                        "completed_640k_shards_per_batch",
                        "remainder_steps_in_current_fragment",
                        "newly_completed_logical_640k_shards_since_previous_stop",
                        "partial_fragment_remainder_steps_since_previous_stop",
                    ),
                )
                for key, value in record.items():
                    self.assertIs(type(value), int, f"{resolution}.{key}")
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, signed_int64_max)
            self.assertEqual(observed[-1]["remainder_steps_in_current_fragment"], 0)
            self.assertEqual(
                observed[-1]["global_step"],
                work[f"{resolution}_steps_per_batch"],
            )
            self.assertEqual(
                observed[-1]["completed_640k_shards_per_batch"],
                work[f"{resolution}_shards_per_batch"],
            )

    def test_cpu_wh_gpu_quiet_and_cpu_dop853_split_is_fail_closed(self) -> None:
        design = self.protocol["cpu_gpu_integrator"]
        self.assertEqual(
            design["architecture"],
            "CPU_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN_MASSIVE_SOURCE_"
            "PLUS_GPU_FUSED_RESTRICTED_WISDOM_HOLMAN_QUIET_TRACERS_"
            "PLUS_CPU_DOP853_GUARDED_ROUTE",
        )
        self.assertIs(design["precision_precedes_speed"], True)
        source = design["massive_source"]
        self.assertEqual(
            (source["backend"], source["device"], source["dtype"]),
            ("numpy", "cpu", "float64"),
        )
        self.assertEqual(
            source["callable"],
            "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile",
        )
        self.assertEqual(source["source_body_count"], 5)
        self.assertIs(source["complete_same_runtime_bitwise_replay_required"], True)
        self.assertEqual(
            source["canonical_stream_scope"],
            "ONE_PER_FORCING_CONTEXT_AND_RESOLUTION",
        )
        self.assertIs(source["redundant_per_batch_source_rebuild_forbidden"], True)
        self.assertIs(source["full_physical_fragment_sha256_required"], True)
        self.assertIs(source["physical_fragment_start_and_end_state_receipts_required"], True)
        tracer = design["tracer_propagation"]
        self.assertEqual(
            (tracer["backend"], tracer["device"], tracer["dtype"]),
            ("cupy", "cuda:0", "float64"),
        )
        self.assertEqual(tracer["quiet_map"], "CUDA_RESTRICTED_WH_FIXED_STEP")
        self.assertEqual(tracer["tracers_per_batch"], 129)
        self.assertIs(tracer["deterministic_reductions"], True)
        self.assertIs(tracer["fast_math"], False)
        self.assertIs(tracer["cpu_fallback_allowed"], False)
        guarded = design["guarded_route"]
        self.assertEqual(guarded["backend"], "numpy_cpu_workers")
        self.assertEqual(guarded["method"], "CONTINUOUS_RESTRICTED_FRAME_DOP853")
        self.assertEqual(
            guarded["transaction"],
            "DISCARD_PROVISIONAL_GPU_STEP_AND_REDO_ONLY_THE_GUARDED_LANE",
        )
        self.assertIs(guarded["currently_qualified_for_3gyr"], False)
        self.assertIs(guarded["route_and_removal_counts_are_outcomes_not_gates"], True)
        coupling = design["coupling"]
        self.assertIs(
            coupling["source_physical_fragment_exact_replay_and_sha256_required"],
            True,
        )
        self.assertIs(coupling["gpu_consumes_exact_verified_cpu_source_fragment_bytes"], True)
        self.assertIs(
            coupling["canonical_source_fragment_broadcast_to_all_twelve_batch_workers"],
            True,
        )
        self.assertIs(
            coupling["every_batch_worker_verifies_full_fragment_sha256_before_use"],
            True,
        )
        self.assertIs(coupling["external_history_event_stream_sha256_required"], True)
        self.assertIs(
            coupling["gpu_quiet_and_cpu_guarded_routes_share_exact_external_history_cursor"],
            True,
        )
        self.assertIs(coupling["external_forcing_gpu_and_cpu_route_kernels_implemented"], False)
        self.assertIs(
            coupling["twelve_gpu_batches_are_independent_workers_after_source_verification"],
            True,
        )
        self.assertIs(
            coupling["one_batch_context_and_resolution_stays_on_one_qualified_runtime"],
            True,
        )
        self.assertIs(
            coupling["paired_h64_h128_batch_context_runs_use_same_device_uuid_sequentially"],
            True,
        )
        self.assertIs(
            coupling["runtime_migration_without_separate_bitwise_qualification_allowed"],
            False,
        )
        self.assertEqual(coupling["mismatch_action"], "STOP_BEFORE_WRITING_SUCCESS")

    def test_distributed_identity_fragment_restart_and_retry_contracts_are_exact(self) -> None:
        design = self.protocol["cpu_gpu_integrator"]
        distributed = design["distributed_execution_contract"]
        self.assertEqual(
            distributed,
            {
                "worker_unit_count": 24_600,
                "worker_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                ],
                "signed_int64_fields": [
                    "forcing_context_index",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                    "logical_640k_shard_index",
                    "physical_fragment_index",
                    "start_global_step",
                    "end_global_step",
                ],
                "physical_batch_fragment_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                    "start_global_step",
                    "end_global_step",
                ],
                "canonical_source_fragment_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "start_global_step",
                    "end_global_step",
                ],
                "physical_fragment_index_scope": "ZERO_BASED_PER_STREAM",
                "first_physical_fragment_index_is_zero": True,
                "physical_fragment_indices_increment_by_exactly_one": True,
                "physical_fragments_are_nonoverlapping_and_contiguous": True,
                "first_physical_fragment_start_global_step_is_zero": True,
                "successor_start_step_equals_predecessor_end_step": True,
                "final_physical_fragment_end_global_step_equals_resolution_horizon_step": True,
                "logical_640k_shard_index_retained_for_capacity_accounting_only": True,
                "canonical_aggregation_order": (
                    "FORCING_CONTEXT_THEN_RESOLUTION_H64_H128_THEN_BATCH_0_TO_11_"
                    "THEN_PHYSICAL_FRAGMENT_START_STEP_ASCENDING"
                ),
                "idempotent_create_only_physical_fragment_commit_required": True,
                "fixed_integer_counts_and_fixed_reduction_order_required": True,
                "runtime_migration_requires_separate_bitwise_qualification": True,
                "h64_h128_pair_same_device_uuid_sequential_execution_required": True,
            },
        )

        restart = design["fragment_and_restart_contract"]
        self.assertEqual(
            restart["canonical_incremental_prefix_fragments"],
            {
                "h64": [
                    {"stop_julian_year": 1_000, "steps": 64_000},
                    {"stop_julian_year": 5_000, "steps": 256_000},
                    {"stop_julian_year": 10_000, "steps": 320_000},
                ],
                "h128": [
                    {"stop_julian_year": 1_000, "steps": 128_000},
                    {"stop_julian_year": 5_000, "steps": 512_000},
                    {"stop_julian_year": 10_000, "steps": 640_000},
                ],
            },
        )
        self.assertIs(restart["all_stops_after_10k_are_640k_step_aligned"], True)
        self.assertEqual(
            restart["restart_payload_fields"],
            [
                "signed_int64_start_global_step",
                "signed_int64_end_global_step",
                "signed_int64_logical_640k_shard_index",
                "signed_int64_physical_fragment_index",
                "forcing_context_id",
                "resolution_steps_per_julian_year",
                "batch_order",
                "all_129_tracer_state_and_lifecycle",
                "terminal_and_removal_ledger",
                "pending_route_state_or_exact_proof_none",
                "external_history_cursor",
                "canonical_source_start_state",
                "canonical_source_end_state",
                "canonical_source_full_physical_fragment_sha256",
                "predecessor_manifest_sha256",
                "code_model_runtime_receipts",
            ],
        )
        self.assertIs(restart["fresh_process_bitwise_suffix_validation_required"], True)
        self.assertEqual(
            restart["retry_eligible_failures"],
            [
                "PRECOMMIT_PROCESS_INTERRUPTION",
                "PRECOMMIT_DEVICE_INTERRUPTION",
                "PRECOMMIT_IO_INTERRUPTION",
            ],
        )
        self.assertIs(restart["retry_requires_identical_pinned_runtime_and_inputs"], True)
        self.assertEqual(
            restart["nan_hash_numerical_or_route_failure_action"],
            "STOP_WITHOUT_AUTOMATIC_RETRY",
        )

    def test_h128_means_128_binary64_steps_per_year_not_128_bit_arithmetic(self) -> None:
        design = self.protocol["cpu_gpu_integrator"]
        h128 = design["h128_primary"]
        h64 = design["h64_complete_control"]
        self.assertEqual(
            h128["current_status"],
            "CANDIDATE_REQUIRES_STAGED_QUALIFICATION_BEFORE_EXECUTION",
        )
        self.assertEqual(h128["source_step_julian_years_binary64_hex"], "0x1.0000000000000p-7")
        self.assertEqual(h128["steps_per_julian_year"], 128)
        self.assertIs(h128["binary64_not_128_bit_arithmetic"], True)
        self.assertEqual(h128["horizon_julian_years"], 3_000_000_000)
        self.assertEqual(h128["shards_per_batch"], 600_000)
        self.assertEqual(h64["source_step_julian_years_binary64_hex"], "0x1.0000000000000p-6")
        self.assertEqual(h64["steps_per_julian_year"], 64)
        self.assertEqual(h64["horizon_julian_years"], 3_000_000_000)
        self.assertEqual(h64["shards_per_batch"], 300_000)
        self.assertEqual(
            h128["qualification_stop_years"],
            [row["julian_year"] for row in self.protocol["work_geometry"]["h128_stop_schedule"]],
        )
        self.assertIs(design["long_horizon_accuracy_qualified"], False)
        self.assertIs(
            design["pointwise_h64_h128_closeness_after_chaotic_divergence_is_a_gate"],
            False,
        )

    def test_short_horizon_qualification_contract_and_prior_evidence_are_exact(self) -> None:
        design = self.protocol["cpu_gpu_integrator"]
        qualification = design["short_horizon_qualification_contract"]
        self.assertEqual(
            qualification,
            {
                "scope": {
                    "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
                    "horizon_julian_years": 1_000,
                    "checkpoint_interval_julian_years": 1,
                    "checkpoint_count_endpoint_inclusive": 1_001,
                    "batch_orders": list(range(12)),
                    "tracer_orders": list(range(129)),
                    "reference_integrator": (
                        "INDEPENDENT_CPU_IAS15_EXACT_IMPLEMENTATION_AND_RUNTIME_TO_BE_PINNED"
                    ),
                    "common_epoch_bridge_must_pass_first": True,
                },
                "h64_h128_richardson_divisor": 3,
                "richardson_normalized_semimajor_axis_max": "3/8000",
                "richardson_normalized_semimajor_axis_formula": (
                    "MAX_OVER_CHECKPOINTS_ABS_A_LEFT_MINUS_A_RIGHT_DIV_"
                    "MAX_A_LEFT_INITIAL_A_RIGHT_INITIAL"
                ),
                "richardson_cartesian_position_au_max": "1/500",
                "richardson_cartesian_velocity_au_per_julian_year_max": "1/50000",
                "h128_vs_ias15_normalized_semimajor_axis_max": "3/4000",
                "h128_vs_ias15_scattering_diagnostic_disagreement_max": "3/4000",
                "h128_vs_ias15_cartesian_position_au_max": "1/500",
                "h128_vs_ias15_cartesian_velocity_au_per_julian_year_max": "1/50000",
                "scalar_contraction_rule": (
                    "FINE_LE_EXACT_64_ULP_FLOOR_OR_COARSE_GE_3_TIMES_FINE"
                ),
                "vector_contraction_rule": (
                    "FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_"
                    "COARSE_SQUARED_GE_9_TIMES_FINE_SQUARED"
                ),
                "ulp_floor_rule": (
                    "PER_CHECKPOINT_EXACT_64_TIMES_REFERENCE_SCALAR_ULP;_"
                    "VECTOR_FLOOR_IS_EXACT_SUM_OF_THREE_SQUARED_COMPONENT_FLOORS;_"
                    "TAKE_MAXIMUM_OVER_THE_SAME_1001_CHECKPOINTS"
                ),
                "energy_contraction_rule": (
                    "FINE_LE_MAX_OF_COARSE_OVER_3_AND_2_POW_MINUS_46"
                ),
                "all_node_fine_massive_source_relative_energy_drift_max": "1/100000000",
                "all_node_fine_massive_source_angular_vector_drift_max": "1/10000000000",
                "all_individual_predicates_must_pass": True,
                "energy_and_angular_gates_apply_only_to_autonomous_control": True,
                "external_forcing_work_and_torque_balance_residual_contract_frozen": False,
                "external_forcing_lanes_cannot_use_autonomous_energy_or_angular_gates": True,
            },
        )
        self.assertEqual(len(qualification["scope"]["batch_orders"]), 12)
        self.assertEqual(len(qualification["scope"]["tracer_orders"]), 129)
        self.assertIn(
            "EXTERNAL_FORCING_WORK_AND_TORQUE_BALANCE_RESIDUAL_CONTRACT_NOT_FROZEN",
            self.protocol["execution_blockers"],
        )
        self.assertIn(
            "INDEPENDENT_ALL129_CPU_IAS15_REFERENCE_IMPLEMENTATION_AND_RUNTIME_NOT_PINNED",
            self.protocol["execution_blockers"],
        )

        self.assertEqual(
            design["resolved11_h128_prior_evidence"],
            {
                "status": "UNBOUND_LOCAL_DIAGNOSTIC_ONLY",
                "qualification": "EXECUTED_GATES_NOT_ALL_PASS",
                "passed_gate_count": 12,
                "total_gate_count": 20,
                "selected9_tracer_gates_all_passed": True,
                "massive_source_gate_failure_count": 8,
                "reduced5_or_full129_h128_authority": False,
                "scientific_disproof": False,
            },
        )

    def test_physical_model_ladder_and_boundary_nonclaims_are_exact(self) -> None:
        model = self.protocol["physical_model"]
        self.assertEqual(
            model["bound_control_model"],
            "ISOLATED_NEWTONIAN_REDUCED5_PLUS_129_MASSLESS_TNO_STATES_PER_BATCH",
        )
        self.assertEqual(
            model["minimum_model_ladder_before_broad_physical_interpretation"],
            [
                "QUALIFIED_COMMON_EPOCH_FULL_COVARIANCE_BRIDGE",
                "RESOLVED_OR_VALIDATED_MULTIRATE_KNOWN_PLANETARY_MODEL_WITH_PLUTO_CHARON_SENSITIVITY",
                "GALACTIC_TIDE_IN_A_FROZEN_GALACTIC_FRAME_AND_POTENTIAL",
                "PREREGISTERED_STATISTICAL_STELLAR_ENCOUNTER_HISTORIES",
                "PHYSICAL_COLLISION_SURFACES_WITH_CONTINUOUS_EVENT_LOCATION",
                "OUTER_BARYCENTRIC_OORT_DOMAIN_ROUTE_AND_ESCAPE_OR_CENSORING_RULE",
                "PLANETARY_INITIAL_CONDITION_AND_PARAMETER_SENSITIVITY",
            ],
        )
        self.assertIs(
            model["solar_mass_evolution_1pn_j2_dwarf_planet_kbo_ring_ablations_required"],
            True,
        )
        self.assertEqual(
            model["current_4096_au_crossing_label"],
            "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION",
        )
        self.assertIs(model["current_zero_terminal_source_radii_support_collision_claim"], False)
        self.assertIs(model["external_perturber_model_implemented_and_qualified"], False)
        self.assertIs(model["control_model_is_a_real_solar_system_prediction"], False)
        self.assertIs(model["tno_backreaction"], False)
        self.assertIs(model["mutual_tno_gravity"], False)
        self.assertIn(
            "NO_PHYSICAL_EJECTION_OR_COLLISION_FROM_CURRENT_BOUNDARIES",
            self.protocol["nonclaims"],
        )

        forcing = model["external_forcing_ensemble"]
        self.assertEqual(forcing["statistical_stellar_history_count"], 1_024)
        self.assertEqual(forcing["galactic_tide_only_control_count"], 1)
        self.assertEqual(forcing["total_forcing_context_count"], 1_025)
        self.assertEqual(
            forcing["statistical_history_ids"],
            "STELLAR_HISTORY_0000_THROUGH_1023",
        )
        self.assertEqual(
            forcing["control_id"],
            "GALACTIC_TIDE_ONLY_NO_DISCRETE_STARS",
        )
        self.assertIs(forcing["same_forcing_context_applied_to_all_twelve_objects"], True)
        self.assertIs(forcing["identical_context_ids_paired_across_h64_and_h128"], True)
        self.assertEqual(
            forcing["history_seed_domain"],
            "jx.tno.3gyr-full129-robustness.stellar-history.v1",
        )
        self.assertEqual(
            forcing["history_seed_preimage"],
            "ASCII_DOMAIN_NUL_PLUS_FOUR_DIGIT_ASCII_HISTORY_INDEX",
        )
        self.assertIs(
            forcing["generator_rng_catalog_probability_law_and_truncations_frozen"],
            False,
        )
        self.assertIs(
            forcing["catalog_stars_may_be_used_only_inside_their_validated_time_window"],
            True,
        )
        self.assertIs(forcing["catalog_stars_extrapolated_for_three_gyr"], False)
        self.assertIs(
            forcing["synthetic_history_frequencies_are_model_sensitivity_not_real_probabilities"],
            True,
        )
        self.assertIs(
            forcing["execution_forbidden_until_generator_and_every_history_digest_are_bound"],
            True,
        )
        self.assertIn(
            "STELLAR_HISTORY_GENERATOR_RNG_DISTRIBUTIONS_TRUNCATIONS_AND_DIGESTS_NOT_FROZEN",
            self.protocol["execution_blockers"],
        )

    def test_b22_is_first_ten_myr_only_and_not_a_three_gyr_classifier(self) -> None:
        outcome = self.protocol["outcome_contract"]
        self.assertEqual(
            outcome["first_10myr_diagnostic"],
            "JX_B22_OPERATIONALIZATION_V2_ONLY_AFTER_ITS_OWN_QUALIFICATION",
        )
        self.assertEqual(outcome["first_10myr_b22_nodes"], 50_001)
        self.assertIs(outcome["single_b22_label_over_3gyr_forbidden"], True)
        self.assertEqual(
            outcome["candidate_windowing"],
            "300_NONOVERLAPPING_10MYR_WINDOWS_SHARING_ONLY_BOUNDARY_NODES",
        )
        self.assertIs(outcome["windowing_requires_separate_qualification"], True)
        self.assertNotIn("classification", self.protocol)
        self.assertNotIn("JX_B22_OPERATIONALIZATION_V2", outcome["primary_3gyr_outputs"])
        self.assertEqual(
            outcome["primary_3gyr_outputs"],
            [
                "SURVIVED_TO_HORIZON_UNDER_FROZEN_MODEL",
                "FIRST_TERMINAL_OR_CENSORING_EVENT_TYPE_AND_INTEGER_STEP",
                "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION",
                "STATE_RESIDENCE_AND_TRANSITION_COUNTS",
                "ORBITAL_ELEMENT_ENVELOPES_AMONG_RETAINED_ACTIVE_STATES",
                "PAIRED_H64_H128_OBJECT_AND_CLONE_SENSITIVITY",
            ],
        )
        self.assertIs(outcome["numerical_failure_is_never_a_physical_removal"], True)
        self.assertIs(outcome["pointwise_prediction_after_chaotic_divergence"], False)
        self.assertIs(
            outcome["clone_frequencies_are_covariance_sensitivity_not_probabilities"],
            True,
        )
        self.assertIs(outcome["no_cross_object_population_statistic"], True)

    def test_full_dual_resolution_retention_and_storage_are_exact(self) -> None:
        retention = self.protocol["retention_geometry"]
        self.assertIs(retention["precision_maximal_200_year_state_retention_selected"], True)
        self.assertEqual(retention["state_interval_julian_years"], 200)
        self.assertEqual(retention["h64_state_interval_steps"], 12_800)
        self.assertEqual(retention["h128_state_interval_steps"], 25_600)
        self.assertEqual(retention["state_nodes_per_batch_endpoint_inclusive"], 15_000_001)
        self.assertEqual(retention["state_body_count_per_batch"], 134)
        self.assertEqual(retention["phase_components_per_body"], 6)
        self.assertEqual(retention["phase_component_bytes"], 8)
        self.assertEqual(retention["lifecycle_lanes_per_batch"], 129)
        self.assertEqual(retention["node_digest_bytes_per_batch"], 32)
        self.assertEqual(retention["first_500k_interval_julian_years"], 50)
        self.assertEqual(retention["h64_first_500k_interval_steps"], 3_200)
        self.assertEqual(retention["h128_first_500k_interval_steps"], 6_400)
        self.assertEqual(retention["h64_first_500k_final_step"], 32_000_000)
        self.assertEqual(retention["h128_first_500k_final_step"], 64_000_000)
        self.assertEqual(retention["first_500k_nodes_per_batch_endpoint_inclusive"], 10_001)
        self.assertEqual(retention["ten_myr_windows"], 300)
        self.assertEqual(retention["nodes_per_ten_myr_window_endpoint_inclusive"], 50_001)
        self.assertIs(retention["every_integrator_step_source_nodes_fully_retained"], False)
        self.assertIs(
            retention["source_fragments_regenerated_and_bound_by_full_fragment_sha256"],
            True,
        )
        self.assertIs(retention["source_fragment_start_and_end_state_receipts_required"], True)
        self.assertIs(
            retention["hierarchical_append_only_64_bit_manifest_offset_format_required"],
            True,
        )

        storage = self.protocol["storage"]
        self.assertEqual(storage["forcing_context_count"], 1_025)
        self.assertEqual(storage["geometry_scope"], "REDUCED5_FIXED_ARRAY_SUBTOTAL_ONLY")
        breakdown = storage["per_context_per_resolution_fixed_array_breakdown_bytes"]
        self.assertEqual(
            breakdown,
            {
                "states_200_year": 1_157_760_077_184,
                "lifecycle_200_year": 23_220_001_548,
                "all12_node_sha256_200_year": 5_760_000_384,
                "first_500k_states_50_year": 771_917_184,
                "first_500k_lifecycle_50_year": 15_481_548,
            },
        )
        self.assertEqual(sum(breakdown.values()), 1_187_527_477_848)
        self.assertEqual(storage["node_sha256_200_year_per_batch_bytes"], 480_000_032)
        self.assertEqual(
            storage["per_context_per_resolution_all12_fixed_output_logical_bytes"],
            1_187_527_477_848,
        )
        self.assertEqual(
            storage["per_context_per_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"],
            1_187_527_487_488,
        )
        self.assertEqual(
            storage["per_context_per_resolution_finalized_per_batch_stream_4k_sum_bytes"],
            1_187_527_606_272,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_fixed_output_logical_bytes"],
            2_375_054_955_696,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"],
            2_375_054_974_976,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_finalized_per_batch_stream_4k_sum_bytes"],
            2_375_055_212_544,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_fixed_output_logical_bytes"],
            storage["per_context_per_resolution_all12_fixed_output_logical_bytes"] * 2,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"],
            storage[
                "per_context_per_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"
            ]
            * 2,
        )
        self.assertEqual(
            storage["per_context_dual_resolution_finalized_per_batch_stream_4k_sum_bytes"],
            storage["per_context_per_resolution_finalized_per_batch_stream_4k_sum_bytes"]
            * 2,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_fixed_output_logical_bytes"],
            2_434_431_329_588_400,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"],
            2_434_431_349_350_400,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_finalized_per_batch_stream_4k_sum_bytes"],
            2_434_431_592_857_600,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_fixed_output_logical_bytes"],
            storage["per_context_dual_resolution_fixed_output_logical_bytes"] * 1_025,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"],
            storage[
                "per_context_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes"
            ]
            * 1_025,
        )
        self.assertEqual(
            storage["ensemble_dual_resolution_finalized_per_batch_stream_4k_sum_bytes"],
            storage["per_context_dual_resolution_finalized_per_batch_stream_4k_sum_bytes"]
            * 1_025,
        )
        self.assertEqual(
            storage["per_context_canonical_continuous_source_stream_bytes_if_fully_retained"],
            138_240_000_000_480,
        )
        self.assertEqual(
            storage["ensemble_canonical_continuous_source_stream_bytes_if_fully_retained"],
            141_696_000_000_492_000,
        )
        self.assertEqual(
            storage[
                "per_context_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints"
            ],
            138_240_216_000_000,
        )
        self.assertEqual(
            storage[
                "ensemble_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints"
            ],
            141_696_221_400_000_000,
        )
        self.assertEqual(
            storage[
                "per_context_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints"
            ],
            138_240_216_000_720,
        )
        self.assertEqual(
            storage[
                "ensemble_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints"
            ],
            141_696_221_400_738_000,
        )
        self.assertEqual(
            storage[
                "per_context_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints"
            ]
            - storage[
                "per_context_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints"
            ],
            720,
        )
        self.assertEqual(
            storage[
                "ensemble_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints"
            ]
            - storage[
                "ensemble_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints"
            ],
            738_000,
        )
        self.assertEqual(
            storage["fixed_array_subtotals_exclude"],
            [
                "ROUTE_AND_REMOVAL_RECORDS",
                "RESTART_PAYLOADS",
                "MANIFESTS_AND_RECEIPTS",
                "INTERMEDIATE_PHYSICAL_FRAGMENT_PAYLOADS_AND_PER_FRAGMENT_RECEIPTS_AND_FILESYSTEM_ALLOCATION",
                "REPLICATION",
                "TRANSIENT_STAGING",
                "EXPANDED_PHYSICAL_MODEL_BODIES",
            ],
        )
        self.assertNotIn("per_resolution_all12_fixed_output_logical_bytes", storage)
        self.assertNotIn("dual_resolution_fixed_output_logical_bytes", storage)
        self.assertIs(
            storage["current_writer_physical_fragment_inode_and_roster_format_qualified"],
            False,
        )
        self.assertIs(
            storage["exact_external_storage_replication_route_transient_plan_required"],
            True,
        )
        self.assertIs(storage["success_forbidden_if_space_or_inode_floor_is_not_met"], True)

    def test_no_budget_ceiling_still_has_one_bounded_exact_retry(self) -> None:
        policy = self.protocol["resource_policy"]
        self.assertIs(policy["compute_budget_limit_imposed"], False)
        self.assertIs(policy["storage_budget_limit_imposed"], False)
        self.assertIs(policy["precision_and_complete_custody_precede_speed_or_cost"], True)
        self.assertEqual(
            policy["charged_step_definition"],
            "ONE_ATTEMPTED_FIXED_MAP_STEP_FOR_ONE_FULL129_BATCH;"
            "FAILED_AND_REPLAYED_ATTEMPTS_REMAIN_REPORTED",
        )
        self.assertIs(
            type(policy["maximum_automatic_precommit_interruption_retries_per_physical_fragment"]),
            int,
        )
        self.assertEqual(
            policy["maximum_automatic_precommit_interruption_retries_per_physical_fragment"],
            1,
        )
        self.assertEqual(
            policy["automatic_retry_scope"],
            "PROCESS_DEVICE_OR_IO_INTERRUPTION_BEFORE_CREATE_ONLY_COMMIT_ONLY",
        )
        self.assertEqual(policy["nan_hash_numerical_or_route_failure_automatic_retries"], 0)
        self.assertEqual(
            policy["after_retry_failure"],
            "STOP_RETAIN_FAILURE_AND_REQUIRE_A_NEW_PREREGISTRATION",
        )
        self.assertIs(policy["selective_outcome_dependent_retries_forbidden"], True)
        self.assertIs(policy["all_attempts_and_adaptive_route_accounting_retained"], True)
        self.assertIs(policy["wall_runtime_projection_frozen"], False)
        self.assertIs(
            policy["runtime_requires_measured_qualified_cpu_gpu_route_and_io_throughput"],
            True,
        )
        self.assertIn(
            "CURRENT_10MYR_ADAPTER_IS_H64_AND_EXACTLY_1000_SHARDS_PER_BATCH_ONLY",
            self.protocol["execution_blockers"],
        )
        self.assertIn(
            "CANONICAL_SHARED_SOURCE_SERVICE_AND_FULL_FRAGMENT_BROADCAST_NOT_IMPLEMENTED_OR_QUALIFIED",
            self.protocol["execution_blockers"],
        )

    def test_phase_order_requires_every_gate_before_long_execution(self) -> None:
        phases = self.protocol["mandatory_phase_order"]
        self.assertEqual(
            phases[0],
            "PREPARE_COMMIT_AND_ANNOTATED_TAG_THIS_PREPARE_ONLY_PREREGISTRATION",
        )

        def position(name: str) -> int:
            return phases.index(name)

        self.assertLess(
            position("FREEZE_1024_STELLAR_HISTORIES_PLUS_TIDE_ONLY_CONTROL_AND_BIND_EVERY_DIGEST"),
            position("IMPLEMENT_AND_QUALIFY_H64_H128_3GYR_ADAPTERS_WITH_INT64_STEPS"),
        )
        self.assertLess(
            position("IMPLEMENT_AND_QUALIFY_H64_H128_3GYR_ADAPTERS_WITH_INT64_STEPS"),
            position("EXECUTE_ALL12_FULL129_H64_H128_1KYR_WITH_FROZEN_REFERENCE_LANES"),
        )
        self.assertLess(
            position("IMPLEMENT_AND_QUALIFY_HIERARCHICAL_WRITER_RESTART_AND_EXTERNAL_STORAGE"),
            position("EXECUTE_ALL12_FULL129_H64_H128_1KYR_WITH_FROZEN_REFERENCE_LANES"),
        )
        self.assertLess(
            position("STOP_AND_REVIEW_ONLY_ENGINEERING_EVIDENCE_WITH_SCIENCE_OUTCOMES_SEALED"),
            position("CONTINUE_PREFIXES_THROUGH_5KYR_10KYR_100KYR_500KYR_5MYR_AND_10MYR"),
        )
        self.assertLess(
            position("PASS_EXPANDED_FORCE_MODEL_AND_BOUNDARY_SENSITIVITY_GATES"),
            position("CONTINUE_WITHOUT_DROPPING_OR_TUNING_THROUGH_100MYR_500MYR_1GYR_2GYR_3GYR"),
        )
        self.assertLess(
            position("VERIFY_OUTPUT_RESTART_SOURCE_ROUTE_MODEL_RUNTIME_STORAGE_CROSSBINDINGS"),
            position("PUBLISH_ONLY_CONDITIONAL_CHAOTIC_ENSEMBLE_MODEL_OUTPUT"),
        )

    def test_claim_ceiling_is_exact_boolean_and_only_prepare_is_true(self) -> None:
        claims = self.protocol["claims"]
        expected_keys = {
            "prepare_only_protocol_frozen",
            "execution_implemented",
            "execution_authorized",
            "cpu_source_executed",
            "gpu_tracers_executed",
            "all_12_h64_qualification_passed",
            "all_12_h128_qualification_passed",
            "h128_route_qualified",
            "canonical_shared_source_service_qualified",
            "distributed_batch_runtime_qualified",
            "guarded_cpu_route_qualified",
            "independent_all129_ias15_reference_pinned",
            "external_forcing_work_torque_balance_qualified",
            "three_gyr_physical_model_qualified",
            "h64_three_gyr_control_completed",
            "h128_three_gyr_primary_completed",
            "first_ten_myr_b22_classification_generated",
            "three_gyr_survival_record_generated",
            "cross_resolution_statistical_comparison_generated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        }
        self.assertEqual(set(claims), expected_keys)
        for key, value in claims.items():
            self.assertIs(type(value), bool, key)
            self.assertIs(value, key == "prepare_only_protocol_frozen", key)
        self.assertIn(
            "NO_PLANET_NINE_DETECTION_EXCLUSION_OR_RANKING",
            self.protocol["nonclaims"],
        )

    def test_exact_git_and_file_authority_is_bound(self) -> None:
        authority = self.protocol["authority"]
        git = authority["git_authority"]
        self.assertEqual(git["base_commit"], self.module.BASE_COMMIT)
        self.assertEqual(git["base_tree"], self.module.BASE_TREE)
        self.assertEqual(git["predecessor_tag_object"], self.module.PREDECESSOR_TAG_OBJECT)
        records = authority["exact_files"]
        self.assertEqual(len(records), len(self.module.FILE_PINS))
        self.assertEqual(len({record["relative_path"] for record in records}), len(records))
        for record in records:
            self.assertIs(record["live_equals_committed"], True)
            self.assertIs(type(record["byte_length"]), int)
            self.assertTrue(self.module._is_sha256(record["sha256"]))

    def test_source_and_test_self_receipts_match_exact_bytes(self) -> None:
        implementation = self.protocol["implementation"]
        for role, path in (("source", SOURCE_PATH), ("test", Path(__file__).resolve())):
            payload = path.read_bytes()
            self.assertEqual(implementation[role]["byte_length"], len(payload))
            self.assertEqual(implementation[role]["sha256"], _sha256(payload))
            self.assertEqual(
                implementation[role]["relative_path"],
                path.relative_to(PROJECT_ROOT).as_posix(),
            )

    def test_strict_json_and_exact_comparison_reject_ambiguous_types(self) -> None:
        m = self.module
        for payload in (
            b'{"a":1,"a":2}\n',
            b'{"a":NaN}\n',
            b'{"a":Infinity}\n',
            b'{"a":1.0}\n',
            b'[1,2]\n',
            b'"object"\n',
            b'\xff',
        ):
            with self.subTest(payload=payload), self.assertRaises(m.TNOThreeBillionError):
                m._strict_json(payload)
        self.assertFalse(m._exact_equal(True, 1))
        self.assertFalse(m._exact_equal(False, 0))
        forged = copy.deepcopy(self.protocol)
        forged["claims"]["execution_authorized"] = 0
        forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
        with self.assertRaises(m.TNOThreeBillionError):
            m.validate_protocol(forged)

    def test_semantic_mutations_fail_even_after_coordinated_rehash(self) -> None:
        m = self.module
        mutations = (
            (("claims", "science_result"), True),
            (("evidence_class",), "MODEL_OUTPUT"),
            (("resource_policy", "compute_budget_limit_imposed"), True),
            (
                (
                    "resource_policy",
                    "maximum_automatic_precommit_interruption_retries_per_physical_fragment",
                ),
                2,
            ),
            (("time_contract", "duration_julian_years"), 3_000_000_001),
            (("roster", "batch_count"), 4),
            (("roster", "forcing_context_count"), 1_024),
            (("cpu_gpu_integrator", "tracer_propagation", "fast_math"), True),
            (("cpu_gpu_integrator", "tracer_propagation", "cpu_fallback_allowed"), True),
            (("physical_model", "current_4096_au_crossing_label"), "EJECTED"),
            (
                (
                    "physical_model",
                    "external_forcing_ensemble",
                    "generator_rng_catalog_probability_law_and_truncations_frozen",
                ),
                True,
            ),
            (
                (
                    "physical_model",
                    "external_forcing_ensemble",
                    "synthetic_history_frequencies_are_model_sensitivity_not_real_probabilities",
                ),
                False,
            ),
            (("outcome_contract", "single_b22_label_over_3gyr_forbidden"), False),
        )
        for field, value in mutations:
            forged = copy.deepcopy(self.protocol)
            cursor = forged
            for key in field[:-1]:
                cursor = cursor[key]
            cursor[field[-1]] = value
            forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
            with self.subTest(field=field), self.assertRaises(m.TNOThreeBillionError):
                m.validate_protocol(forged)

    def test_held_file_read_rejects_size_digest_symlink_and_hardlink(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-held-") as temporary:
            root = Path(temporary)
            exact = root / "exact.bin"
            exact.write_bytes(b"exact-payload")
            digest = _sha256(b"exact-payload")
            self.assertEqual(
                m._read_file(exact, expected_bytes=13, expected_sha256=digest, label="exact"),
                b"exact-payload",
            )
            with self.assertRaises(m.TNOThreeBillionError):
                m._read_file(exact, expected_bytes=12, expected_sha256=digest, label="size")
            with self.assertRaises(m.TNOThreeBillionError):
                m._read_file(exact, expected_bytes=13, expected_sha256="0" * 64, label="hash")
            symlink = root / "symlink.bin"
            symlink.symlink_to(exact.name)
            with self.assertRaises(m.TNOThreeBillionError):
                m._read_file(symlink, expected_bytes=13, expected_sha256=digest, label="symlink")
            hardlink = root / "hard.bin"
            os.link(exact, hardlink)
            with self.assertRaises(m.TNOThreeBillionError):
                m._read_file(exact, expected_bytes=13, expected_sha256=digest, label="hardlink")

    def test_prepare_is_create_only_read_only_on_validation_and_failure_atomic(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-prepare-") as temporary:
            root = Path(temporary) / "prepared"
            prepared = m.prepare(root)
            self.assertTrue(m._exact_equal(prepared, self.protocol))
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["PROTOCOL.json"])
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "PROTOCOL.json").stat().st_mode), 0o600)
            snapshot = _tree_snapshot(root)
            validated = m.validate_prepared(root)
            self.assertTrue(m._exact_equal(validated, prepared))
            self.assertEqual(_tree_snapshot(root), snapshot)
            with self.assertRaises(m.TNOThreeBillionError):
                m.prepare(root)
            self.assertEqual(_tree_snapshot(root), snapshot)

        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-fault-") as temporary:
            root = Path(temporary) / "prepared"
            with mock.patch.object(
                m,
                "_rename_noreplace",
                side_effect=m.TNOThreeBillionError("injected failure"),
            ):
                with self.assertRaises(m.TNOThreeBillionError):
                    m.prepare(root)
            self.assertFalse(root.exists())

        with tempfile.TemporaryDirectory(
            prefix="jx-tno-3gyr-postwrite-validation-fault-"
        ) as temporary:
            root = Path(temporary) / "prepared"
            with mock.patch.object(
                m,
                "_validate_prepared_directory",
                side_effect=m.TNOThreeBillionError("injected validation failure"),
            ):
                with self.assertRaises(m.TNOThreeBillionError):
                    m.prepare(root)
            self.assertFalse(root.exists())

        with tempfile.TemporaryDirectory(
            prefix="jx-tno-3gyr-parent-fsync-fault-"
        ) as temporary:
            root = Path(temporary) / "prepared"
            real_fsync = m.os.fsync
            calls = 0

            def fail_third_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected parent directory fsync failure")
                real_fsync(descriptor)

            with mock.patch.object(m.os, "fsync", side_effect=fail_third_fsync):
                with self.assertRaises(OSError):
                    m.prepare(root)
            self.assertFalse(root.exists())

    def test_prepare_and_validate_reject_symlink_hardlink_and_extra_roster(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-symlink-root-") as temporary:
            parent = Path(temporary)
            real = parent / "real"
            real.mkdir()
            alias = parent / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(m.TNOThreeBillionError):
                m.prepare(alias / "prepared")

        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-hardlink-") as temporary:
            root = Path(temporary) / "prepared"
            m.prepare(root)
            os.link(root / "PROTOCOL.json", root / "second-link")
            with self.assertRaises(m.TNOThreeBillionError):
                m.validate_prepared(root)

        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-extra-") as temporary:
            root = Path(temporary) / "prepared"
            m.prepare(root)
            (root / "unexpected").write_bytes(b"")
            with self.assertRaises(m.TNOThreeBillionError):
                m.validate_prepared(root)

    def test_prepared_claim_tamper_is_rejected_even_after_rehash(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-forge-") as temporary:
            root = Path(temporary) / "prepared"
            m.prepare(root)
            protocol_path = root / "PROTOCOL.json"
            forged = m._strict_json(protocol_path.read_bytes())
            forged["claims"]["execution_authorized"] = True
            forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
            protocol_path.write_bytes(m._canonical_json(forged))
            with self.assertRaises(m.TNOThreeBillionError):
                m.validate_prepared(root)

    def test_normal_prepare_then_optimized_validation_is_read_only(self) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONOPTIMIZE", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory(prefix="jx-tno-3gyr-crossmode-") as temporary:
            root = Path(temporary) / "prepared"
            creator = subprocess.run(
                [sys.executable, "-B", "-c", CHILD, str(SOURCE_PATH), "prepare", str(root)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(creator.returncode, 0, creator.stderr)
            snapshot = _tree_snapshot(root)
            validator = subprocess.run(
                [sys.executable, "-OO", "-B", "-c", CHILD, str(SOURCE_PATH), "validate", str(root)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(validator.returncode, 0, validator.stderr)
            self.assertEqual(creator.stdout.strip(), validator.stdout.strip())
            self.assertEqual(_tree_snapshot(root), snapshot)

    def test_cli_rejects_every_execution_surface(self) -> None:
        m = self.module
        for command in ("execute", "execute-worker", "resume", "validate-result"):
            with self.subTest(command=command):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        m._parser().parse_args([command])


if __name__ == "__main__":
    unittest.main()
