#!/usr/bin/env python3
"""Independent stored-artifact verifier for JX-XP2.

This file does not import either numerical runner.  It reconstructs the frozen
design and initial states, validates the primary segment commit chains,
recomputes all endpoint arithmetic and numerical gates, verifies A before B,
requires exact A/B semantic equality, and performs the seven-arm DOP853
cross-method audit against both MERCURIUS resolutions.
"""

from __future__ import annotations

import argparse
import ctypes
import decimal
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v4"
SCIENTIFIC_DESIGN_EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v1"
CONTRACT_SCHEMA = "jx-xp2-robustness-contract/v3"
REGISTRATION_SCHEMA = "jx-xp2-local-registration/v1"
ENGINEERING_REGISTRATION_SCHEMA = "jx-xp2-v4-engineering-registration/v1"
ENGINEERING_RECEIPT_SCHEMA = "jx-xp2-v4-engineering-boundary-verification/v1"
PRIMARY_RESULT_SCHEMA = "jx-xp2-primary-result/v3"
DOP_RESULT_SCHEMA = "jx-xp2-dop853-result/v1"
COMMIT_SCHEMA = "jx-xp2-mercurius-segment-parent-commit/v3"
RECEIPT_SCHEMA = "jx-xp2-mercurius-segment-receipt/v3"
A_RECEIPT_SCHEMA = "jx-xp2-primary-a-verification/v3"
FINAL_RECEIPT_SCHEMA = "jx-xp2-final-verification/v2"
DOP_CHECKPOINT_SCHEMA = "jx-xp2-dop853-segment-checkpoint/v1"
DOP_SEGMENT_RECEIPT_SCHEMA = "jx-xp2-dop853-segment-parent-receipt/v1"
DOP_RUN_MANIFEST_SCHEMA = "jx-xp2-dop853-run-manifest/v1"
DOP_ATTEMPT_SCHEMA = "jx-xp2-dop853-attempt-ledger-row/v2"
DOP_FAILURE_SCHEMA = "jx-xp2-dop853-failure/v2"
PRIMARY_RUN_MANIFEST_SCHEMA = "jx-xp2-primary-run-manifest/v3"
PRIMARY_ATTEMPT_SCHEMA = "jx-xp2-mercurius-segment-attempt/v4"
PRIMARY_FAILURE_SCHEMA = "jx-xp2-primary-failure/v4"
PRIMARY_FAILURE_CLASSES = {
    "CHILD_EXIT_NONZERO", "CHILD_SIGNAL", "SEGMENT_TIMEOUT",
    "CHILD_RSS_LIMIT", "RECOVERED_UNCOMMITTED",
}
V2_DEFECT_EVIDENCE_SHA256 = "7cd515610718eaa9fac3159f988ef924c6df030cc8828719818b5b461789ff47"
V2_DEFECT_EVIDENCE_SIZE_BYTES = 5626
V3_FAILED_STARTUP_EVIDENCE_SHA256 = "eeb5ed87e05aab1ac0fa3cad68391bae1c850090dc48d4113b8b71c58c1dd473"
V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES = 4064
MAX_PRIMARY_CHECKPOINT_BYTES = 1_048_576
MAX_REBOUND_ALLOCATION_CAPACITY = 4096
ENDPOINT_DIGEST_DOMAIN = b"jx-xp2-mercurius-live-archive-endpoint/v1\0"

PRIMARY_ARMS = (
    "M0", *(f"{case}-P{probe}" for case in ("CI01", "CI05", "CI09") for probe in range(8))
)
AUDIT_ARMS = tuple(f"AUDIT-{arm}" for arm in PRIMARY_ARMS)
ALL_ARMS = PRIMARY_ARMS + AUDIT_ARMS
AUDIT_BASE = {f"AUDIT-{arm}": arm for arm in PRIMARY_ARMS}
DOP_ARMS = ("M0", "CI01-P0", "CI01-P4", "CI05-P1", "CI05-P5", "CI09-P2", "CI09-P6")
HORIZONS = (250_000.0, 500_000.0, 1_000_000.0)
CLASS_HORIZONS = (500_000.0, 1_000_000.0)
SEGMENT_DOMAIN = b"jx-xp2-mercurius-semantic-segment-chain/v3\0"
RAW_ARTIFACT_INTEGRITY_DOMAIN = b"jx-xp2-mercurius-raw-artifact-integrity/v1\0"
SEGMENT_GENESIS = hashlib.sha256(SEGMENT_DOMAIN + b"GENESIS").hexdigest()
PRIMARY_SEGMENT_SEMANTIC_FIELD_ORDER = (
    "arm_id", "configuration_id", "arm_class", "dt_years", "segment_index",
    "start_years", "end_years", "first_sample_index", "last_sample_index",
    "new_sample_count", "sample_count_total", "sampled_state_stream_sha256",
    "decoded_integrator_state_sha256", "tracker", "initial_active_invariants",
    "maximum_active_invariant_drifts", "landmarks",
)
PRIMARY_SEGMENT_SEMANTIC_FIELDS = frozenset(PRIMARY_SEGMENT_SEMANTIC_FIELD_ORDER)
CONTINUATION_SIMULATION_FIELDS = (
    "t_hex", "G_hex", "softening_hex", "dt_hex", "dt_last_done_hex", "steps_done",
    "usleep_hex", "save_messages", "status", "N", "particle_capacity_covers_logical_count",
    "particle_storage_present", "active_memory_ranges_pairwise_disjoint",
    "N_var", "N_var_config", "variation_config_present",
    "var_rescale_warning", "N_active", "testparticle_type", "testparticle_hidewarnings",
    "hash_ctr", "particle_lookup_count", "particle_lookup_allocation_count",
    "particle_lookup_present", "integrator", "gravity", "boundary",
    "collision", "exact_finish_time", "force_is_velocity_dependent", "gravity_ignore",
    "exit_max_distance_hex", "exit_min_distance_hex", "track_energy_offset",
    "energy_offset_hex", "opening_angle2_hex", "boxsize_hex", "boxsize_max_hex",
    "root_size_hex", "N_root", "N_root_xyz", "N_ghost_xyz",
    "collision_resolve_keep_sorted", "collisions_N", "minimum_collision_velocity_hex",
    "gravity_compensated_sums_present", "gravity_compensated_sums_allocation_count",
    "tree_root_present", "tree_needs_update", "messages_present", "display_view_present",
    "display_data_present", "server_data_present", "collision_storage_present",
    "collision_allocation_count", "collisions_plog_hex", "collisions_log_n",
    "calculate_megno", "megno_Ys_hex", "megno_Yss_hex", "megno_cov_Yt_hex",
    "megno_var_t_hex", "megno_mean_t_hex", "megno_mean_Y_hex", "megno_initial_t_hex",
    "megno_n", "N_odes", "odes_allocation_count", "odes_warnings", "odes_present",
    "extras_present", "simulationarchive_auto_interval_hex",
    "simulationarchive_auto_walltime_hex", "simulationarchive_auto_step",
    "simulationarchive_next_hex", "simulationarchive_next_step",
    "simulationarchive_filename_present", "callbacks_present",
)
CONTINUATION_CALLBACK_FIELDS = (
    "additional_forces", "pre_timestep_modifications", "post_timestep_modifications",
    "heartbeat", "coefficient_of_restitution", "collision_resolve", "free_particle_ap",
    "key_callback", "extras_cleanup",
)
CONTINUATION_MERCURIUS_FIELDS = (
    "r_crit_hill_hex", "safe_mode", "mode", "is_synchronized",
    "recalculate_coordinates_this_timestep", "recalculate_r_crit_this_timestep",
    "encounter_N", "encounter_N_active", "tponly_encounter",
    "dcrit_storage_present", "dcrit_capacity_covers_logical_count", "dcrit_hex",
    "com_position_hex", "com_velocity_hex", "L_callback_present",
    "allocated_particle_backup_count", "allocated_additional_forces_backup_count",
    "particles_backup_present", "additional_forces_backup_present",
    "encounter_map_present",
)
CONTINUATION_WHFAST_FIELDS = (
    "coordinates", "kernel", "corrector", "corrector2",
    "recalculate_coordinates_this_timestep", "safe_mode", "keep_unsynchronized",
    "is_synchronized", "timestep_warning", "unsynchronized_recalculation_warning",
    "internal_particle_arrays_present",
)
CONTINUATION_IAS15_FIELDS = (
    "epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded",
    "stored_coordinate_count", "direct_array_sha256", "coefficient_array_sha256",
    "map_count", "map_sha256",
)
CONTINUATION_PARTICLE_FIELDS = (
    "index", "hash", "simulation_reference_bound_to_parent", "m_hex", "r_hex",
    "x_hex", "y_hex", "z_hex", "vx_hex",
    "vy_hex", "vz_hex", "ax_hex", "ay_hex", "az_hex", "last_collision_hex",
    "collision_cell_present", "additional_properties_present",
)
CONTINUATION_EXCLUDED_FIELDS = (
    "process_pointers", "unselected_integrator_inactive_allocator_capacities",
    "inactive_nonmercurius_integrator_structs", "walltime",
    "walltime_last_step", "walltime_last_steps", "walltime_last_steps_sum",
    "walltime_last_steps_N", "output_timing_last",
    "python_unit_codes_and_display_metadata", "rand_seed",
    "simulationarchive_format_version",
)


def expected_continuation_declaration_v3() -> dict[str, Any]:
    return {
        "schema": "jx-xp2-mercurius-decoded-continuation-state/v3",
        "state_digest_domain": "jx-xp2-mercurius-decoded-continuation-state/v3\\0",
        "array_digest_domain": "jx-xp2-mercurius-decoded-array/v3\\0",
        "live_archive_endpoint_digest_domain": "jx-xp2-mercurius-live-archive-endpoint/v1\\0",
        "live_archive_endpoint_projection": {
            "schema": "jx-xp2-mercurius-live-archive-endpoint/v1",
            "digest_domain": "jx-xp2-mercurius-live-archive-endpoint/v1\\0",
            "retained_top_level_fields": [
                "simulation", "mercurius", "whfast", "ias15", "particles",
                "excluded_noncontinuation_fields",
            ],
            "normalized_mercurius_fields": [
                "encounter_N", "encounter_N_active", "tponly_encounter",
                "allocated_particle_backup_count",
                "allocated_additional_forces_backup_count", "particles_backup_present",
                "additional_forces_backup_present", "encounter_map_present",
            ],
            "normalized_whfast_fields": ["internal_particle_arrays_present"],
            "normalized_ias15_fields": [
                "stored_coordinate_count", "direct_array_sha256",
                "coefficient_array_sha256", "map_count", "map_sha256",
            ],
            "all_omitted_live_allocations_require_bounded_pointer_count_coherence_alignment_and_pairwise_nonoverlap": True,
            "authorization_requires_registered_full_segment_engineering_boundary_PASS": True,
        },
        "simulation_fields": list(CONTINUATION_SIMULATION_FIELDS),
        "callback_fields": list(CONTINUATION_CALLBACK_FIELDS),
        "mercurius_fields": list(CONTINUATION_MERCURIUS_FIELDS),
        "whfast_fields": list(CONTINUATION_WHFAST_FIELDS),
        "ias15_fields": list(CONTINUATION_IAS15_FIELDS),
        "ias15_direct_arrays": ["at", "x0", "v0", "a0", "csx", "csv", "csa0"],
        "ias15_coefficient_groups": ["g", "b", "csb", "e", "br", "er"],
        "ias15_coefficients_per_group": [f"p{index}" for index in range(7)],
        "allowed_ias15_stored_coordinate_counts": [0, 9],
        "required_ias15_map_count": 0,
        "particle_fields": list(CONTINUATION_PARTICLE_FIELDS),
        "absolute_process_addresses_hashed": False,
        "allocator_capacity_values_hashed": False,
        "logical_particle_and_dcrit_prefix_only_hashed": True,
        "bounded_capacity_and_pointer_coherence_required": True,
        "active_memory_alignment_and_nonoverlap_required": True,
        "particle_parent_simulation_binding_required": True,
        "excluded_noncontinuation_fields": list(CONTINUATION_EXCLUDED_FIELDS),
        "exclusion_rationale": "PROCESS_ADDRESS_INACTIVE_ALLOCATOR_INACTIVE_ARCHIVE_FORMAT_PRESENTATION_AND_WALL_CLOCK_METADATA_DO_NOT_AFFECT_CONTINUATION; AUTOMATIC_ARCHIVE_OUTPUT_IS_SEPARATELY_REQUIRED_DISABLED; RAND_SEED_IS_UNUSED_BECAUSE_REGISTERED_COLLISION_AND_ALL_STOCHASTIC_OR_USER_FORCE_CALLBACKS_ARE_DISABLED",
        "all_projected_binary64_and_array_values_must_be_finite": True,
    }


def expected_raw_artifact_declaration_v1() -> dict[str, Any]:
    return {
        "schema": "jx-xp2-mercurius-raw-artifact-integrity/v1",
        "root_digest_domain": "jx-xp2-mercurius-raw-artifact-integrity/v1\\0",
        "entry_count": 1000,
        "order": "REGISTERED_PRIMARY_ARM_ORDER_THEN_SEGMENT_INDEX_ASCENDING",
        "entry_fields": [
            "arm_id", "segment_index", "commit_filename", "commit_size_bytes",
            "commit_sha256", "receipt_filename", "receipt_size_bytes",
            "receipt_sha256", "checkpoint_filename", "checkpoint_size_bytes",
            "checkpoint_sha256",
        ],
        "mandatory_result_provenance": True,
        "independently_recomputed_by_verifier": True,
        "scientific_semantic_input": False,
        "A_B_equality_required": False,
    }


def expected_v4_fresh_repair_declaration() -> dict[str, Any]:
    return {
        "registration_scope": "BEFORE_FIRST_OFFICIAL_V4_SCIENTIFIC_OUTPUT",
        "primary_A_primary_B_and_DOP853_start_from_registered_v1_initial_state_bytes": True,
        "v1_v2_v3_checkpoint_result_ledger_resume_promotion_or_A_prerequisite": "FORBIDDEN",
        "v1_v2_v3_artifacts_are_diagnosis_only_and_never_scientific_inputs_gates_labels_or_classification": True,
        "resource_accounting_scan_v3": "HELD_DIRECTORY_FD_RECURSION_WITH_O_DIRECTORY_AND_O_NOFOLLOW; TRANSIENT_CHILD_FILE_RENAME_OR_UNLINK_MAY_BE_SKIPPED; MISSING_ROOT_SYMLINK_HARDLINK_SPECIAL_FILE_OR_REPLACED_DIRECTORY_FAILS_CLOSED",
        "rebound_decode_release_v3": "EXPLICIT_SIMULATION_REFERENCE_RELEASE_AND_GARBAGE_COLLECTION_AFTER_PARENT_OR_VERIFIER_CHECKPOINT_VALIDATION",
        "checkpoint_boundary_v4": "SAVE_AND_FSYNC_PENDING_THEN_DECODE_AND_VALIDATE_CANONICAL_ARCHIVE_CONTINUATION_THEN_COMPARE_NORMALIZED_LIVE_ARCHIVE_SCIENTIFIC_ENDPOINT_THEN_ATOMIC_RENAME",
        "allocator_normalization_v4": "ONLY_FIELDS_ENUMERATED_BY_DECODED_CONTINUATION_STATE_V3_LIVE_ARCHIVE_ENDPOINT_PROJECTION_MAY_NORMALIZE; BOUNDED_CAPACITY_POINTER_COHERENCE_ALIGNMENT_NONOVERLAP_AND_LOGICAL_PARTICLE_DCRIT_PREFIX_VALUES_REMAIN_REQUIRED; AUTHORITY_REQUIRES_THE_REGISTERED_ENGINEERING_BOUNDARY_PASS",
        "science_design_gates_seeds_initial_states_sampling_and_dynamics_changed": False,
    }


def expected_engineering_boundary_gate_v1() -> dict[str, Any]:
    return {
        "schema": "jx-xp2-v4-engineering-boundary-gate/v1",
        "purpose": "REGISTERED_NONSCIENTIFIC_FULL_SEGMENT_SAVE_DECODE_AND_RESTART_BOUNDARY_AUTHORIZATION_ONLY",
        "registration_sequence": ["FREEZE_CORE_AND_HARNESS", "CREATE_ENGINEERING_REGISTRATION", "RUN_ONE_EXTERNAL_ENGINEERING_ATTEMPT", "INDEPENDENTLY_VERIFY_AND_PUBLISH_PASS_RECEIPT", "CREATE_FINAL_OFFICIAL_REGISTRATION_LAST"],
        "engineering_registration_schema": ENGINEERING_REGISTRATION_SCHEMA,
        "engineering_registration_path": "engineering_registration_v1.json",
        "final_registration_path": "registration_v1.json",
        "final_registration_engineering_authorization_schema": "jx-xp2-v4-final-engineering-authorization/v1",
        "final_registration_must_be_absent_during_engineering_run_and_verification": True,
        "engineering_registration_locks_every_registered_core_and_harness_file_except_both_registration_files": True,
        "exercise_arm_ids": ["M0", "CI01-P0", "AUDIT-CI01-P0"],
        "start_years": 0.0, "boundary_years": 50_000.0,
        "continuation_probe_years": 50_050.0,
        "integration_call_cadence_years": 50.0, "exact_finish_time": 1,
        "preboundary_trajectories_per_arm": 2,
        "preboundary_trajectory_roles": ["SAVED_CANDIDATE", "UNSAVED_CONTROL"],
        "runner_continuation_trajectories_per_arm": 2,
        "runner_continuation_trajectory_roles": [
            "DECODED_SAVED_CANDIDATE", "UNSAVED_CONTROL",
        ],
        "engineering_registration_authorizes_only": {
            "runner": "run_engineering_boundary.py; ONE SAVED_CANDIDATE_AND_ONE UNSAVED_CONTROL 0_TO_50000_SEGMENT_FOR_EACH_REGISTERED_EXERCISE_ARM; BOTH INTEGRATE_AT_EVERY_50_YEAR_GRID_POINT; SAVE_DECODE_CANDIDATE_AND_ONE UNSAVED_CONTROL_VS_DECODED 50000_TO_50050_RESTART_COMPARISON",
            "independent_verifier": "verify_engineering_boundary.py; REDECODE_ALL_REGISTERED_ARCHIVES_AND_EXACTLY_ONE_50000_TO_50050_RESTART_PER_ARM_FOR_INDEPENDENT_PARITY",
            "runtime": "CONTRACT_RUNTIME_LOCK_LOCAL_CPU_ONLY",
            "network_gpu_observed_data": "FORBIDDEN",
            "official_A_B_DOP_or_scientific_analysis": "NOT_AUTHORIZED",
        },
        "engineering_runner_required_cli": [
            "--contract", "--seed-manifest", "--initial-states",
            "--engineering-registration", "--output-root",
        ],
        "engineering_verifier_required_cli": [
            "--contract", "--initial-states", "--engineering-registration",
            "--engineering-output-root", "--receipt",
        ],
        "engineering_output_root": "../jx_xp2_runs_v4/engineering_boundary",
        "engineering_attempt_schema": "jx-xp2-v4-engineering-attempt/v1",
        "engineering_result_schema": "jx-xp2-v4-engineering-boundary-result/v1",
        "engineering_result_filename": "result_v1.json",
        "engineering_verifier_scratch_root": "../jx_xp2_verification_v4/engineering_boundary_verifier_scratch",
        "engineering_verifier_start_path": "../jx_xp2_verification_v4/engineering_boundary_verifier_start_v1.json",
        "engineering_verifier_terminal_path": "../jx_xp2_verification_v4/engineering_boundary_verifier_terminal_v1.json",
        "engineering_verification_receipt_path": "../jx_xp2_verification_v4/engineering_boundary_receipt_v1.json",
        "engineering_verification_receipt_schema": ENGINEERING_RECEIPT_SCHEMA,
        "required_arm_tree_fingerprint_schema": "jx-xp2-v4-engineering-arm-tree-fingerprint/v1",
        "registered_particle_vector_digest_domain": "jx-xp2-v4-engineering-particle-vectors/v1\\0",
        "registered_particle_vector_fields": ["hash_uint32", "m_binary64_hex", "r_binary64_hex"],
        "registered_particle_vector_sha256_by_configuration": dict(
            ENGINEERING_PARTICLE_VECTOR_SHA256
        ),
        "raw_live_allocator_and_cache_topology_captured_immediately_before_save_to_file": True,
        "engineering_resource_caps": "EXACT_RESOURCE_CAPS_PER_EXECUTION_WITH_PARENT_PROCESS_GROUP_WATCHDOG_POLLING_DURING_EVERY_INTEGRATION_CALL; ANY_CAP_FAILURE_CONSUMES_THE_ONE_SHOT_AND_FORCES_V5",
        "full_normalized_field_set_must_be_captured_structurally_validated_and_compared_for_all_three_arms": True,
        "required_live_vs_decoded_must_differ_fields_in_CI01_or_AUDIT": [
            "simulation.N_allocated", "mercurius.allocated_particle_backup_count",
            "mercurius.particles_backup_present", "mercurius.encounter_map_present",
        ],
        "required_live_nonzero_topology_coverage_in_CI01_or_AUDIT": [
            "whfast.internal_particle_arrays_present_with_decoded_absent",
            "ias15.stored_coordinate_count_positive_with_map_count_zero_and_decoded_strict",
        ],
        "equal_by_design_normalized_fields_do_not_need_to_differ": [
            "mercurius.allocated_additional_forces_backup_count",
            "mercurius.additional_forces_backup_present",
        ],
        "independent_verifier_redecodes_every_archive_and_recomputes_strict_projection_endpoint_and_50050_restart_parity": True,
        "engineering_artifacts_are_nonpromotable_and_never_scientific_inputs_gates_labels_or_classification": True,
        "outcomes_generated_false_means_no_scientific_outcomes_and_does_not_deny_registered_engineering_diagnostics": True,
        "one_attempt_only_no_resume": True,
        "any_START_without_one_exact_PASS_or_any_partial_failure_extra_or_mutation_forces_v5": True,
        "final_registration_must_bind_unchanged_engineering_registration_exact_three_runner_arm_tree_fingerprints_verifier_scratch_tree_result_receipt_and_verifier_PASS_terminal": True,
        "any_core_change_after_engineering_probe_invalidates_gate": True,
        "official_primary_B_DOP_and_verifier_authority_requires_final_registration": True,
    }
EXPANDED_DOMAIN = b"jx-xp2-expanded-barycentric-state/v1\0"
INDEX_DOMAIN = b"jx-xp2-configuration-digest-index/v1\0"
SELECTION_DOMAIN = b"jx-xp2-dop853-sentinel/v1\0"
LHS_DOMAIN = b"jx-xp2-lhs-u64/v1\0"
TRACER_DOMAIN = b"jx-xp2-canonical-tracer-design/v1\0"
PRIMARY_STATE_DOMAIN = b"jx-xp2-mercurius-decoded-continuation-state/v3\0"
CONTINUATION_ARRAY_DOMAIN = b"jx-xp2-mercurius-decoded-array/v3\0"
PRIMARY_ENDPOINT_DOMAIN = b"jx-xp2-mercurius-live-archive-endpoint/v1\0"
REBOUND_TREE_DOMAIN = b"jx-e2-rebound-python-sources/v1\0"
DIMENSIONS = ("LOG_A", "Q", "COS_I", "OMEGA", "OMEGA_ARGUMENT", "MEAN_ANOMALY")
_VERIFICATION_LOCK_FDS: dict[tuple[int, int], int] = {}
_ENGINEERING_RUNNER_GUARD_FD: int | None = None
_ENGINEERING_SCRATCH_GUARD_FD: int | None = None
_FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS: list[Any] = []


class VerificationError(RuntimeError):
    pass


sys.dont_write_bytecode = True


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


IntegrityError = VerificationError
canonical_bytes = canonical
sha256_bytes = digest_bytes


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


def binary64_from_hex(value: Any, label: str = "binary64") -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise VerificationError(f"{label} hex width changed")
    try:
        parsed = struct.unpack(">d", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as exc:
        raise VerificationError(f"{label} is not binary64 hex") from exc
    if not math.isfinite(parsed):
        raise VerificationError(f"{label} is non-finite")
    return parsed


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def primary_segment_semantic_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Independently select decoded/scientific state, never raw archive bytes."""
    if not PRIMARY_SEGMENT_SEMANTIC_FIELDS <= set(value):
        raise VerificationError("primary segment semantic payload is incomplete")
    return {key: value[key] for key in PRIMARY_SEGMENT_SEMANTIC_FIELDS}


def primary_raw_artifact_integrity_inventory(output_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for arm_id in ALL_ARMS:
        segment_dir = output_root / "arms" / arm_id / "segments"
        for segment_index in range(20):
            commit_path = segment_dir / f"segment_{segment_index:02d}_commit.json"
            commit = read_object(commit_path)
            receipt_path = segment_dir / commit["attempt_receipt_filename"]
            receipt = read_object(receipt_path)
            state_path = segment_dir / commit["checkpoint_filename"]
            if (state_path.name != receipt.get("checkpoint_filename")
                    or state_path.is_symlink() or not state_path.is_file()
                    or state_path.stat().st_nlink != 1):
                raise VerificationError("raw artifact integrity inventory path changed")
            entries.append({
                "arm_id": arm_id,
                "segment_index": segment_index,
                "commit_filename": commit_path.name,
                "commit_size_bytes": commit_path.stat().st_size,
                "commit_sha256": digest_file(commit_path),
                "receipt_filename": receipt_path.name,
                "receipt_size_bytes": receipt_path.stat().st_size,
                "receipt_sha256": digest_file(receipt_path),
                "checkpoint_filename": state_path.name,
                "checkpoint_size_bytes": state_path.stat().st_size,
                "checkpoint_sha256": digest_file(state_path),
            })
    if len(entries) != len(ALL_ARMS) * 20:
        raise VerificationError("raw artifact integrity inventory cardinality changed")
    return {
        "schema": "jx-xp2-mercurius-raw-artifact-integrity/v1",
        "entry_count": len(entries),
        "entries": entries,
        "root_sha256": digest_bytes(
            RAW_ARTIFACT_INTEGRITY_DOMAIN + canonical(entries)
        ),
        "scientific_semantic_input": False,
    }


def verify_unlocked_execution_lock(path: Path, label: str) -> int:
    """Acquire and retain an exclusive input-tree lock through publication."""
    path = reject_symlink_path(path, label)
    safe_regular(path, label)
    if path.stat().st_size != 0:
        raise VerificationError(f"{label} size changed")
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        on_disk = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or metadata.st_size != 0 or metadata.st_dev != on_disk.st_dev
                or metadata.st_ino != on_disk.st_ino):
            raise VerificationError(f"{label} binding changed")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in _VERIFICATION_LOCK_FDS:
            return _VERIFICATION_LOCK_FDS[identity]
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise VerificationError(f"{label} is still owned by a live process") from exc
        after_lock = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
                or after_lock.st_size != 0
                or after_lock.st_dev != metadata.st_dev
                or after_lock.st_ino != metadata.st_ino):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise VerificationError(f"{label} path changed during acquisition")
        _VERIFICATION_LOCK_FDS[identity] = descriptor
        retained = descriptor
        descriptor = -1
        return retained
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def release_verification_locks() -> None:
    global _ENGINEERING_RUNNER_GUARD_FD, _ENGINEERING_SCRATCH_GUARD_FD
    for snapshot in reversed(_FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS):
        snapshot.close()
    _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS.clear()
    _ENGINEERING_RUNNER_GUARD_FD = None
    _ENGINEERING_SCRATCH_GUARD_FD = None
    for descriptor in _VERIFICATION_LOCK_FDS.values():
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    _VERIFICATION_LOCK_FDS.clear()


def exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise VerificationError(f"{label} fields changed")


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def lower_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


def safe_dir(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise VerificationError(f"{label} is not a safe directory")


def safe_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise VerificationError(f"{label} is not a safe regular file")


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise VerificationError("duplicate JSON key")
        output[key] = value
    return output


def finite_float(text: str) -> float:
    value = float(text); exact = decimal.Decimal(text)
    if not math.isfinite(value) or not exact.is_finite() or (value == 0.0 and exact != 0):
        raise VerificationError("invalid JSON number")
    return value


def reject_constant(text: str) -> None:
    raise VerificationError(f"invalid JSON constant: {text}")


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise VerificationError("unsafe JSON artifact")
    value = json.loads(path.read_text(), object_pairs_hook=unique_pairs,
                       parse_float=finite_float, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise VerificationError("JSON root is not an object")
    return value


def _open_bound_directory(
    path: Path, label: str,
) -> tuple[list[int], list[tuple[int, str, int, os.stat_result]]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptors = [os.open(absolute.anchor, flags)]
    bindings: list[tuple[int, str, int, os.stat_result]] = []
    try:
        for component in absolute.parts[1:]:
            parent = descriptors[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            child = os.open(component, flags, dir_fd=parent)
            opened = os.fstat(child)
            if (not stat.S_ISDIR(before.st_mode) or not stat.S_ISDIR(opened.st_mode)
                    or before.st_dev != opened.st_dev or before.st_ino != opened.st_ino):
                os.close(child)
                raise VerificationError(f"{label} directory binding changed")
            descriptors.append(child)
            bindings.append((parent, component, child, opened))
        return descriptors, bindings
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _revalidate_bound_directory(
    bindings: Sequence[tuple[int, str, int, os.stat_result]], label: str,
) -> None:
    for parent, name, descriptor, before in bindings:
        after = os.fstat(descriptor)
        on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if any(getattr(before, key) != getattr(after, key) for key in (
                "st_dev", "st_ino", "st_mode")) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in (
                    "st_dev", "st_ino", "st_mode"
                )):
            raise VerificationError(f"{label} directory binding changed")


def _read_bound_regular(
    directory_fd: int, name: str, label: str, *, allowed_links: set[int],
) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink not in allowed_links:
        raise VerificationError(f"{label} is unsafe")
    flags = os.O_RDONLY | (
        os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink not in allowed_links
                or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size):
            raise VerificationError(f"{label} binding changed")
        payload = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        on_disk = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if any(getattr(opened, key) != getattr(after, key) for key in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns")) \
                or any(getattr(after, key) != getattr(on_disk, key) for key in (
                    "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                    "st_mtime_ns", "st_ctime_ns"
                )):
            raise VerificationError(f"{label} changed while held")
        return bytes(payload), after
    finally:
        os.close(descriptor)


def _promote_pending_receipt(
    directory_fd: int, pending_name: str, final_name: str, payload: bytes,
) -> None:
    pending_payload, pending_before = _read_bound_regular(
        directory_fd, pending_name, "pending verification receipt", allowed_links={1},
    )
    if pending_payload != payload:
        raise VerificationError("pending verification receipt diverges")
    flags = os.O_RDONLY | (
        os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    )
    pending_fd = os.open(pending_name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(pending_fd)
        if (opened.st_dev != pending_before.st_dev
                or opened.st_ino != pending_before.st_ino):
            raise VerificationError("pending verification receipt binding changed")
        os.fsync(pending_fd)
    finally:
        os.close(pending_fd)
    try:
        os.link(
            pending_name, final_name, src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd, follow_symlinks=False,
        )
    except FileExistsError as exc:
        try:
            final = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
            pending = os.stat(pending_name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as binding_exc:
            raise VerificationError("raced verification receipt is unstable") from binding_exc
        if final.st_dev != pending.st_dev or final.st_ino != pending.st_ino:
            raise VerificationError("raced verification receipt conflicts") from exc
    os.fsync(directory_fd)
    final = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
    pending = os.stat(pending_name, dir_fd=directory_fd, follow_symlinks=False)
    if (not stat.S_ISREG(final.st_mode) or not stat.S_ISREG(pending.st_mode)
            or final.st_dev != pending.st_dev or final.st_ino != pending.st_ino
            or final.st_nlink != 2 or pending.st_nlink != 2):
        raise VerificationError("verification receipt no-clobber promotion changed")
    os.unlink(pending_name, dir_fd=directory_fd)
    os.fsync(directory_fd)
    final_payload, final = _read_bound_regular(
        directory_fd, final_name, "published verification receipt", allowed_links={1},
    )
    if final_payload != payload or final.st_nlink != 1:
        raise VerificationError("published verification receipt changed")


def publish(path: Path, value: dict[str, Any]) -> None:
    path = validate_receipt_destination(path, "verification receipt destination")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    pending_name = f".{path.name}.pending"
    descriptors, bindings = _open_bound_directory(
        path.parent, "verification receipt parent",
    )
    directory_fd = descriptors[-1]
    try:
        names = set(os.listdir(directory_fd))
        final_exists = path.name in names
        pending_exists = pending_name in names
        if final_exists:
            final_payload, final = _read_bound_regular(
                directory_fd, path.name, "published verification receipt",
                allowed_links={1, 2},
            )
            if pending_exists:
                pending_payload, pending = _read_bound_regular(
                    directory_fd, pending_name, "pending verification receipt",
                    allowed_links={2},
                )
                if (final_payload != payload or pending_payload != payload
                        or final.st_dev != pending.st_dev
                        or final.st_ino != pending.st_ino
                        or final.st_nlink != 2 or pending.st_nlink != 2):
                    raise VerificationError("published verification receipt conflicts")
                os.unlink(pending_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                final_payload, final = _read_bound_regular(
                    directory_fd, path.name, "published verification receipt",
                    allowed_links={1},
                )
            if final_payload != payload or final.st_nlink != 1:
                raise VerificationError("published verification receipt conflicts")
            _revalidate_bound_directory(bindings, "verification receipt parent")
            return
        if pending_exists:
            candidate, _metadata = _read_bound_regular(
                directory_fd, pending_name, "pending verification receipt",
                allowed_links={1},
            )
            if candidate == payload:
                _promote_pending_receipt(
                    directory_fd, pending_name, path.name, payload,
                )
                _revalidate_bound_directory(bindings, "verification receipt parent")
                return
            if not payload.startswith(candidate):
                raise VerificationError("pending verification receipt diverges")
            os.unlink(pending_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | (
            os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
        )
        descriptor = os.open(pending_name, flags, 0o600, dir_fd=directory_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise VerificationError("pending receipt write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        candidate, metadata = _read_bound_regular(
            directory_fd, pending_name, "pending verification receipt",
            allowed_links={1},
        )
        if candidate != payload or metadata.st_size != len(payload):
            raise VerificationError("pending receipt hash/size mismatch")
        _promote_pending_receipt(directory_fd, pending_name, path.name, payload)
        _revalidate_bound_directory(bindings, "verification receipt parent")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def f64(value: str) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise VerificationError("binary64 hex width changed")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise VerificationError("binary64 input is non-finite")
    return result


def unpack6(value: str) -> list[float]:
    if not isinstance(value, str) or len(value) != 96:
        raise VerificationError("packed state width changed")
    return [f64(value[index:index + 16]) for index in range(0, 96, 16)]


def pack6(values: Sequence[float]) -> str:
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise VerificationError("invalid Cartesian vector")
    return "".join(struct.pack(">d", float(value)).hex() for value in values)


def independent_seed(domain: str, design_hash: str, label: str) -> bytes:
    left = domain.encode("ascii"); right = label.encode("ascii")
    return hashlib.sha256(len(left).to_bytes(4, "big") + left + bytes.fromhex(design_hash)
                          + len(right).to_bytes(4, "big") + right + (0).to_bytes(8, "big")).digest()[:16]


def independent_lhs(seed: bytes) -> tuple[list[int], list[float]]:
    counter = 0
    def take() -> int:
        nonlocal counter
        result = int.from_bytes(hashlib.sha256(
            LHS_DOMAIN + seed + counter.to_bytes(8, "big")
        ).digest()[:8], "big")
        counter += 1
        return result
    permutation = list(range(16)); modulus = 1 << 64
    for index in range(15, 0, -1):
        divisor = index + 1; cutoff = modulus - modulus % divisor
        value = take()
        while value >= cutoff: value = take()
        other = value % divisor
        permutation[index], permutation[other] = permutation[other], permutation[index]
    return permutation, [(stratum + (take() >> 11) / float(1 << 53)) / 16.0
                         for stratum in permutation]


def independent_tracers(contract: dict[str, Any], seed_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    policy = contract["seed_policy"]
    stream_rows = seed_manifest.get("streams")
    if not isinstance(stream_rows, list) or len(stream_rows) != 48:
        raise VerificationError("seed manifest stream count changed")
    streams = {row["stream_label"]: row["seed_hex_128"] for row in stream_rows}
    result = []; canonical_rows = []; vector = None
    for block in range(8):
        dimensions = {}
        for suffix in DIMENSIONS:
            label = f"LHS_BLOCK_{block}_{suffix}"
            seed = independent_seed(policy["domain_ascii"], policy["design_core_sha256"], label)
            if streams.get(label) != seed.hex():
                raise VerificationError("seed manifest derivation changed")
            permutation, values = independent_lhs(seed); dimensions[suffix] = values
            if block == 0 and suffix == "LOG_A": vector = permutation, values
        for index in range(16):
            a = math.exp(math.log(150.0) + dimensions["LOG_A"][index]
                         * (math.log(800.0) - math.log(150.0)))
            q = 35.0 + 45.0 * dimensions["Q"][index]; eccentricity = 1.0 - q / a
            inclination = math.acos(math.cos(math.radians(40.0))
                                    + dimensions["COS_I"][index]
                                    * (1.0 - math.cos(math.radians(40.0))))
            row = {
                "logical_id": f"XP2-B{block:02d}-T{index:02d}", "block_index": block,
                "index_within_block": index, "a_AU": a, "q_AU": q, "e": eccentricity,
                "i_rad": inclination, "Omega_rad": 2.0 * math.pi * dimensions["OMEGA"][index],
                "omega_rad": 2.0 * math.pi * dimensions["OMEGA_ARGUMENT"][index],
                "M_rad": 2.0 * math.pi * dimensions["MEAN_ANOMALY"][index],
            }
            result.append(row); canonical_rows.append({
                "logical_id": row["logical_id"], "block_index": block,
                "index_within_block": index, "a_AU_hex": a.hex(), "q_AU_hex": q.hex(),
                "e_hex": eccentricity.hex(), "i_rad_hex": inclination.hex(),
                "Omega_rad_hex": row["Omega_rad"].hex(),
                "omega_rad_hex": row["omega_rad"].hex(), "M_rad_hex": row["M_rad"].hex(),
            })
    if vector is None or vector[0] != policy["block_0_log_a_permutation_test_vector"] \
            or [value.hex() for value in vector[1][:4]] != policy["block_0_log_a_first_four_lhs_float_hex"]:
        raise VerificationError("independent LHS test vector changed")
    if digest_bytes(TRACER_DOMAIN + canonical(canonical_rows)) != policy["canonical_rows_sha256"]:
        raise VerificationError("independent tracer-element digest changed")
    return result


def independent_eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    value = mean_anomaly if eccentricity < 0.8 else math.pi
    for _ in range(32):
        value = value - (value - eccentricity * math.sin(value) - mean_anomaly) \
            / (1.0 - eccentricity * math.cos(value))
    return value


def independent_cartesian(
    G: float, primary_mass: float, orbiting_mass: float, a: float, e: float,
    inc: float, ascending_node: float, omega: float, mean_anomaly: float,
) -> list[float]:
    E = independent_eccentric_anomaly(mean_anomaly, e)
    ce = math.cos(E); se = math.sin(E); beta = math.sqrt(1.0 - e * e)
    x = a * (ce - e); y = a * beta * se
    n = math.sqrt(G * (primary_mass + orbiting_mass) / (a * a * a)); denominator = 1.0 - e * ce
    vx = -a * n * se / denominator; vy = a * n * beta * ce / denominator
    cO = math.cos(ascending_node); sO = math.sin(ascending_node)
    co = math.cos(omega); so = math.sin(omega); ci = math.cos(inc); si = math.sin(inc)
    p = (cO * co - sO * so * ci, sO * co + cO * so * ci, so * si)
    q = (-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si)
    return [x * p[axis] + y * q[axis] for axis in range(3)] + [
        vx * p[axis] + vy * q[axis] for axis in range(3)
    ]


def row(logical_id: str, role: str, mass: float, state: Sequence[float]) -> list[str]:
    return [logical_id, role, struct.pack(">d", mass).hex(), pack6(state)]


def angle_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def inverse_elements(state: Sequence[float], mu: float) -> tuple[float, float, float, float, float, float]:
    r = state[:3]; v = state[3:]; radius = math.sqrt(math.fsum(value * value for value in r))
    speed2 = math.fsum(value * value for value in v); radial = math.fsum(a * b for a, b in zip(r, v))
    h = (r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2],
         r[0] * v[1] - r[1] * v[0]); hmag = math.sqrt(math.fsum(value * value for value in h))
    node = (-h[1], h[0], 0.0); nmag = math.hypot(node[0], node[1])
    evec = tuple(((speed2 - mu / radius) * r[axis] - radial * v[axis]) / mu
                 for axis in range(3)); eccentricity = math.sqrt(math.fsum(value * value for value in evec))
    a = 1.0 / (2.0 / radius - speed2 / mu); inc = math.acos(h[2] / hmag)
    ascending = math.atan2(node[1], node[0]) % (2.0 * math.pi)
    cross_ne = (node[1] * evec[2], -node[0] * evec[2], node[0] * evec[1] - node[1] * evec[0])
    omega = math.atan2(math.fsum(cross_ne[i] * h[i] for i in range(3)) / (nmag * eccentricity * hmag),
                       math.fsum(node[i] * evec[i] for i in range(3)) / (nmag * eccentricity)) % (2.0 * math.pi)
    cross_er = (evec[1] * r[2] - evec[2] * r[1], evec[2] * r[0] - evec[0] * r[2],
                evec[0] * r[1] - evec[1] * r[0])
    anomaly = math.atan2(math.fsum(cross_er[i] * h[i] for i in range(3))
                         / (eccentricity * radius * hmag),
                         math.fsum(evec[i] * r[i] for i in range(3)) / (eccentricity * radius))
    E = 2.0 * math.atan2(math.sqrt(1.0 - eccentricity) * math.sin(anomaly / 2.0),
                         math.sqrt(1.0 + eccentricity) * math.cos(anomaly / 2.0))
    mean = (E - eccentricity * math.sin(E)) % (2.0 * math.pi)
    return a, eccentricity, inc, ascending, omega, mean


def verify_source_rows(
    contract: dict[str, Any], seed_manifest: dict[str, Any], initial: dict[str, Any]
) -> None:
    core = contract["design_core"]; active = core["common_active_system"]
    G = core["units_and_frame"]["G_AU3_Msun_yr2"]; sun_mass = active["sun_mass_Msun"]
    common = [row("Sun", "A", sun_mass, [0.0] * 6)]
    for body in active["giants"]:
        common.append(row(body["name"], "A", body["mass_Msun"], independent_cartesian(
            G, sun_mass, body["mass_Msun"], body["a_AU"], 0.0, 0.0, 0.0, 0.0,
            math.radians(body["initial_longitude_deg"])
        )))
    tracers = independent_tracers(contract, seed_manifest)
    tracer_rows = [row(value["logical_id"], "T", 0.0, independent_cartesian(
        G, sun_mass, 0.0, value["a_AU"], value["e"], value["i_rad"],
        value["Omega_rad"], value["omega_rad"], value["M_rad"]
    )) for value in tracers]
    if common != initial["common_active_sun_centered_rows"] or tracer_rows != initial[
        "tracer_sun_centered_rows"
    ]:
        raise VerificationError("independent exact common/tracer Cartesian regeneration mismatch")
    models = {value["id"]: value for value in core["m1_physical_cases"]}
    probes = {value["id"]: value for value in core["orientation_probes"]}
    configs = {value[0]: value for value in initial["configuration_states"]}
    tolerances = contract["initial_state_policy"]["independent_element_roundtrip_maximum_absolute_error"]
    for arm_id in PRIMARY_ARMS[1:]:
        case_id, probe_id = arm_id.split("-"); model = models[case_id]; probe = probes[probe_id]
        mass = model["mass_Mearth"] * active["earth_to_sun_mass_ratio"]
        omega = math.radians((probe["varpi_deg"] - probe["Omega_deg"]) % 360.0)
        expected = row(f"XP2-{arm_id}", "A", mass, independent_cartesian(
            G, sun_mass, mass, model["a_AU"], model["e"], math.radians(model["i_deg"]),
            math.radians(probe["Omega_deg"]), omega, math.radians(probe["M_deg"])
        ))
        if configs[arm_id][2] != expected:
            raise VerificationError("independent exact M1 Cartesian regeneration mismatch")
        recovered = inverse_elements(unpack6(expected[3]), G * (sun_mass + mass))
        target = (model["a_AU"], model["e"], math.radians(model["i_deg"]),
                  math.radians(probe["Omega_deg"]), omega, math.radians(probe["M_deg"]))
        if (abs(recovered[0] - target[0]) > tolerances["a_AU"]
                or abs(recovered[1] - target[1]) > tolerances["e"]
                or abs(recovered[2] - target[2]) > tolerances["i_rad"]
                or any(angle_distance(recovered[index], target[index])
                       > tolerances["angular_rad_modulo_2pi"] for index in (3, 4, 5))):
            raise VerificationError("independent M1 element roundtrip exceeded tolerance")
    for expected, tracer in zip(tracer_rows, tracers, strict=True):
        recovered = inverse_elements(unpack6(expected[3]), G * sun_mass)
        target = (tracer["a_AU"], tracer["e"], tracer["i_rad"], tracer["Omega_rad"],
                  tracer["omega_rad"], tracer["M_rad"])
        if (abs(recovered[0] - target[0]) > tolerances["a_AU"]
                or abs(recovered[1] - target[1]) > tolerances["e"]
                or abs(recovered[2] - target[2]) > tolerances["i_rad"]
                or any(angle_distance(recovered[index], target[index])
                       > tolerances["angular_rad_modulo_2pi"] for index in (3, 4, 5))):
            raise VerificationError("independent tracer element roundtrip exceeded tolerance")


def validate_v2_replay_lineage(contract: dict[str, Any], package_root: Path) -> dict[str, Any]:
    expected = {
        "role": "INVALID_REPLAY_IDENTITY_DIAGNOSTIC_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT",
        "defect_evidence_path": "v2_replay_defect_evidence_v1.json",
        "defect_evidence_sha256": V2_DEFECT_EVIDENCE_SHA256,
        "defect_evidence_size_bytes": V2_DEFECT_EVIDENCE_SIZE_BYTES,
        "invalid_replay_reason": "RAW_REBOUND_ARCHIVE_HASH_WAS_INCLUDED_IN_THE_V2_SEMANTIC_SEGMENT_CHAIN_DESPITE_EQUAL_CANONICAL_V3_DECODED_CONTINUATION_STATE_AND_SAMPLED_SCIENTIFIC_PAYLOAD",
        "v2_scientific_outcomes_gates_or_classification_consumed": False,
        "v2_artifacts_consumed_only_for_protocol_diagnosis": True,
        "protocol_repair_only_scientific_design_unchanged": True,
        "evolving_v2_b_tree_excluded_non_authorizing": True,
        "v2_b_execution_lock_path": "../jx_xp2_runs_v2/output_b/execution.lock",
        "v4_numerical_output_requires_v2_b_execution_lock_unlocked": True,
        "protected_read_only_trees": [
            "../jx_xp2_robustness_v2", "../jx_xp2_runs_v2",
            "../jx_xp2_verification_v2",
        ],
        "output_receipt_or_checkpoint_tree_overlap": "FORBIDDEN",
    }
    lineage = contract.get("xp2_v2_invalid_replay_lineage")
    if lineage != expected:
        raise VerificationError("XP2-v2 invalid-replay lineage declaration changed")
    evidence_path = package_root / expected["defect_evidence_path"]
    if (evidence_path.is_symlink() or not evidence_path.is_file()
            or evidence_path.stat().st_nlink != 1
            or evidence_path.stat().st_size != V2_DEFECT_EVIDENCE_SIZE_BYTES
            or digest_file(evidence_path) != V2_DEFECT_EVIDENCE_SHA256):
        raise VerificationError("XP2-v2 replay-defect evidence changed")
    evidence = read_object(evidence_path)
    if (
        evidence.get("schema") != "jx-xp2-v2-replay-defect-evidence/v1"
        or evidence.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v3"
        or evidence.get("role")
        != "IMMUTABLE_PROTOCOL_DEFECT_EVIDENCE_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT"
        or evidence.get("diagnosis")
        != "V2_INCORRECTLY_MADE_RAW_REBOUND_ARCHIVE_SERIALIZATION_BYTES_PART_OF_SEMANTIC_REPLAY_IDENTITY"
        or evidence.get("evolving_v2_b_tree_policy")
        != "ALL_UNLISTED_V2_B_ARTIFACTS_AND_THEIR_EVENTUAL_OUTCOME_ARE_EXCLUDED_NON_AUTHORIZING_AND_NOT_CONSUMED"
        or evidence.get("v2_scientific_outcomes_gates_or_classification_consumed") is not False
        or evidence.get("comparison", {}).get("raw_checkpoint_sha256_equal") is not False
        or evidence.get("comparison", {}).get("decoded_integrator_state_sha256_equal") is not True
        or evidence.get("comparison", {}).get("all_v3_semantic_segment_fields_equal") is not True
        or evidence.get("comparison", {}).get("v2_segment_chain_heads_equal") is not False
    ):
        raise VerificationError("XP2-v2 replay-defect diagnosis changed")
    lock_path = package_root / expected["v2_b_execution_lock_path"]
    verify_unlocked_execution_lock(lock_path, "XP2-v2 B lineage execution lock")
    bindings = [
        evidence["v2_registration"], evidence["v2_completed_a_result"],
        evidence["v2_a_verification_receipt"], evidence["v2_a_run_manifest"],
        evidence["v2_b_run_manifest"],
    ]
    for segment_key in ("v2_a_m0_segment_00", "v2_b_m0_segment_00"):
        bindings.extend(evidence[segment_key][kind] for kind in ("commit", "receipt", "state"))
    for binding in bindings:
        if set(binding) not in ({"path", "sha256", "size_bytes"},
                                {"path", "sha256", "size_bytes", "semantic_sha256"}):
            raise VerificationError("XP2-v2 defect-evidence binding shape changed")
        bound = package_root / binding["path"]
        if (bound.is_symlink() or not bound.is_file() or bound.stat().st_nlink != 1
                or bound.stat().st_size != binding["size_bytes"]
                or digest_file(bound) != binding["sha256"]):
            raise VerificationError("XP2-v2 defect-evidence external artifact changed")
    forensic: dict[str, dict[str, Any]] = {}
    rebound = verifier_rebound(contract)
    for run_key, evidence_key in (
        ("a", "v2_a_m0_segment_00"), ("b", "v2_b_m0_segment_00")
    ):
        bound = evidence[evidence_key]
        commit_path = package_root / bound["commit"]["path"]
        receipt_path = package_root / bound["receipt"]["path"]
        state_path = package_root / bound["state"]["path"]
        if not 0 < state_path.stat().st_size <= MAX_PRIMARY_CHECKPOINT_BYTES:
            raise VerificationError("v2 forensic checkpoint exceeds registered byte bound")
        commit = read_object(commit_path); receipt = read_object(receipt_path)
        if (commit.get("attempt_receipt_filename") != receipt_path.name
                or commit.get("attempt_receipt_sha256") != digest_file(receipt_path)
                or commit.get("checkpoint_filename") != state_path.name
                or commit.get("checkpoint_sha256") != digest_file(state_path)
                or receipt.get("checkpoint_filename") != state_path.name
                or receipt.get("checkpoint_sha256") != digest_file(state_path)
                or receipt.get("checkpoint_size_bytes") != state_path.stat().st_size
                or receipt.get("arm_id") != "M0" or receipt.get("segment_index") != 0
                or receipt.get("provenance", {}).get("attempt_index") != 1
                or commit.get("decoded_integrator_state_sha256")
                != receipt.get("decoded_integrator_state_sha256")
                or commit.get("segment_chain_head") != receipt.get("segment_chain_head")):
            raise VerificationError("v2 forensic commit/receipt/state binding changed")
        simulation = rebound.Simulation(str(state_path))
        projection = decoded_primary_continuation_projection(simulation)
        decoded_sha256 = digest_bytes(PRIMARY_STATE_DOMAIN + canonical(projection))
        semantic = primary_segment_semantic_payload(receipt)
        semantic["decoded_integrator_state_sha256"] = decoded_sha256
        semantic_sha256 = digest_bytes(canonical(semantic))
        chain = digest_bytes(
            SEGMENT_DOMAIN + bytes.fromhex(SEGMENT_GENESIS) + canonical(semantic)
        )
        v2_payload = {
            key: receipt[key] for key in (
                *PRIMARY_SEGMENT_SEMANTIC_FIELD_ORDER,
                "checkpoint_sha256", "checkpoint_size_bytes",
            )
        }
        v2_domain = b"jx-xp2-mercurius-segment-chain/v1\0"
        v2_genesis = hashlib.sha256(v2_domain + b"GENESIS").hexdigest()
        v2_recomputed_chain = digest_bytes(
            v2_domain + bytes.fromhex(v2_genesis) + canonical(v2_payload)
        )
        if (receipt.get("previous_segment_chain_head") != v2_genesis
                or receipt.get("segment_chain_head") != v2_recomputed_chain):
            raise VerificationError("v2 forensic semantic chain cannot be reproduced")
        forensic[run_key] = {
            "raw_sha256": digest_file(state_path),
            "decoded_sha256": decoded_sha256,
            "semantic": semantic,
            "semantic_sha256": semantic_sha256,
            "v3_chain": chain,
            "v2_chain": receipt["segment_chain_head"],
            "v2_payload": v2_payload,
            "v2_shallow_decoded_sha256": receipt["decoded_integrator_state_sha256"],
            "checkpoint_size_bytes": receipt["checkpoint_size_bytes"],
            "sampled_stream": receipt["sampled_state_stream_sha256"],
        }
        del simulation
        gc.collect()
    comparison = evidence["comparison"]
    v2_differing_fields = {
        key for key in forensic["a"]["v2_payload"]
        if forensic["a"]["v2_payload"][key] != forensic["b"]["v2_payload"][key]
    }
    if (forensic["a"]["raw_sha256"] == forensic["b"]["raw_sha256"]
            or forensic["a"]["decoded_sha256"] != forensic["b"]["decoded_sha256"]
            or forensic["a"]["semantic"] != forensic["b"]["semantic"]
            or forensic["a"]["v3_chain"] != forensic["b"]["v3_chain"]
            or forensic["a"]["v2_chain"] == forensic["b"]["v2_chain"]
            or forensic["a"]["sampled_stream"] != forensic["b"]["sampled_stream"]
            or forensic["a"]["checkpoint_size_bytes"]
            != forensic["b"]["checkpoint_size_bytes"]
            or forensic["a"]["v2_shallow_decoded_sha256"]
            != forensic["b"]["v2_shallow_decoded_sha256"]
            or v2_differing_fields != {"checkpoint_sha256"}
            or comparison.get("arm_id") != "M0"
            or comparison.get("segment_index") != 0
            or comparison.get("attempt_index") != 1
            or comparison.get("decoded_integrator_state_sha256_equal") is not True
            or comparison.get("v3_decoded_continuation_state_sha256_equal") is not True
            or comparison.get("sampled_state_stream_sha256_equal") is not True
            or comparison.get("v2_shallow_decoded_integrator_state_sha256")
            != forensic["a"]["v2_shallow_decoded_sha256"]
            or comparison.get("v3_decoded_continuation_state_sha256")
            != forensic["a"]["decoded_sha256"]
            or comparison.get("v3_canonical_semantic_segment_payload_sha256")
            != forensic["a"]["semantic_sha256"]
            or comparison.get("v3_semantic_segment_chain_genesis") != SEGMENT_GENESIS
            or comparison.get("v3_expected_segment_chain_head_both_runs")
            != forensic["a"]["v3_chain"]
            or comparison.get("sampled_state_stream_sha256")
            != forensic["a"]["sampled_stream"]
            or comparison.get("v2_a_segment_chain_head") != forensic["a"]["v2_chain"]
            or comparison.get("v2_b_segment_chain_head") != forensic["b"]["v2_chain"]):
        raise VerificationError("v2 forensic replay-defect proof changed")
    return evidence


def validate_v3_failed_startup_lineage(
    contract: dict[str, Any], package_root: Path,
) -> dict[str, Any]:
    expected = {
        "role": "FAILED_STARTUP_DIAGNOSTIC_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT",
        "evidence_path": "v3_failed_startup_evidence_v1.json",
        "evidence_sha256": V3_FAILED_STARTUP_EVIDENCE_SHA256,
        "evidence_size_bytes": V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES,
        "invalid_startup_reason": "LIVE_REBOUND_ALLOCATOR_AND_TRANSIENT_MERCURIUS_CACHE_STATE_WAS_INCORRECTLY_REQUIRED_TO_EQUAL_SAVE_LOAD_NORMALIZED_ARCHIVE_STATE",
        "v3_scientific_outcomes_gates_or_classification_consumed": False,
        "v3_checkpoint_result_ledger_resume_promotion_or_A_prerequisite": "FORBIDDEN",
        "v3_a_execution_lock_path": "../jx_xp2_runs_v3/output_a/execution.lock",
        "v4_numerical_output_requires_v3_a_execution_lock_unlocked": True,
        "protected_read_only_trees": ["../jx_xp2_robustness_v3", "../jx_xp2_runs_v3"],
        "output_receipt_or_checkpoint_tree_overlap": "FORBIDDEN",
    }
    if contract.get("xp2_v3_failed_startup_lineage") != expected:
        raise VerificationError("XP2-v3 failed-startup lineage declaration changed")
    evidence_path = package_root / expected["evidence_path"]
    safe_regular(evidence_path, "XP2-v3 failed-startup evidence")
    if (evidence_path.stat().st_size != V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES
            or digest_file(evidence_path) != V3_FAILED_STARTUP_EVIDENCE_SHA256):
        raise VerificationError("XP2-v3 failed-startup evidence changed")
    evidence = read_object(evidence_path)
    if (evidence.get("schema") != "jx-xp2-v3-failed-startup-evidence/v1"
            or evidence.get("experiment_id") != EXPERIMENT_ID
            or evidence.get("ledger_counts")
            != {"START": 12, "PASS": 0, "FAIL": 9, "OPEN": 3}
            or evidence.get("absence_assertions") != {
                "segment_PASS_rows": 0, "segment_commits": 0, "checkpoints": 0,
                "primary_results": 0, "verification_receipts": 0,
            }):
        raise VerificationError("XP2-v3 failed-startup evidence semantics changed")
    bindings = [evidence[key] for key in (
        "v3_registration", "v3_run_manifest", "v3_attempt_ledger", "v3_execution_lock",
    )] + list(evidence["failure_receipts"])
    for binding in bindings:
        if set(binding) != {"path", "size_bytes", "sha256"}:
            raise VerificationError("XP2-v3 evidence binding shape changed")
        path = package_root / binding["path"]
        safe_regular(path, "XP2-v3 bound artifact")
        if path.stat().st_size != binding["size_bytes"] or digest_file(path) != binding["sha256"]:
            raise VerificationError("XP2-v3 evidence external artifact changed")
    output_root = package_root / evidence["tree_fingerprint"]["root_path"]
    if (evidence["tree_fingerprint"] != {
            "root_path": "../jx_xp2_runs_v3/output_a",
            "schema": "jx-xp2-v3-verifier-tree-fingerprint/v1",
            "digest_formula": "SHA256_CANONICAL_JSON_OF_ROWS",
            "root_entry_included": False, "entry_count": 22,
            "entry_order": "RELATIVE_POSIX_PATH_ASCENDING",
            "directory_row": ["relative_posix_path", "D"],
            "file_row": ["relative_posix_path", "F", "size_bytes", "sha256"],
            "symlinks_hardlinks_and_special_files_allowed": False,
            "sha256": held_verification_tree_fingerprint(
                output_root, output_root / "execution.lock",
                "XP2-v3 failed output tree",
            ),
        }):
        raise VerificationError("XP2-v3 failed-startup tree fingerprint changed")
    rows = []
    for line in (package_root / evidence["v3_attempt_ledger"]["path"]).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise VerificationError("XP2-v3 ledger framing changed")
        row = json.loads(line)
        if canonical(row) + b"\n" != line:
            raise VerificationError("XP2-v3 ledger canonical bytes changed")
        rows.append(row)
    starts = {(row["arm_id"], row["segment_index"], row["attempt_index"]): row
              for row in rows if row.get("event") == "START"}
    fails = [row for row in rows if row.get("event") == "FAIL"]
    if (len(rows) != 21 or len(starts) != 12 or len(fails) != 9
            or [row.get("sequence") for row in rows] != list(range(1, 22))
            or any(row.get("schema") != "jx-xp2-mercurius-segment-attempt/v3"
                   or row.get("execution_label") != "A" for row in rows)):
        raise VerificationError("XP2-v3 failed-startup ledger state changed")
    receipt_bindings = {Path(item["path"]).name: item for item in evidence["failure_receipts"]}
    for row in fails:
        key = (row["arm_id"], row["segment_index"], row["attempt_index"])
        name = row.get("failure_receipt_filename")
        if (key not in starts or name not in receipt_bindings
                or row.get("failure_class") != "CHILD_EXIT_NONZERO"
                or row.get("return_code") != 2
                or row.get("failure_receipt_sha256") != receipt_bindings[name]["sha256"]):
            raise VerificationError("XP2-v3 FAIL row binding changed")
        receipt = read_object(package_root / receipt_bindings[name]["path"])
        if (receipt.get("schema") != "jx-xp2-primary-failure/v3"
                or receipt.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v3"
                or receipt.get("execution_label") != "A" or receipt.get("arm_id") != key[0]
                or receipt.get("segment_index") != key[1] or receipt.get("attempt_index") != key[2]
                or receipt.get("start_sequence") != starts[key]["sequence"]
                or receipt.get("failure_class") != "CHILD_EXIT_NONZERO"
                or receipt.get("return_code") != 2
                or receipt.get("fail_event_sha256") != row.get("fail_event_sha256")):
            raise VerificationError("XP2-v3 failure receipt binding changed")
    manifest = read_object(package_root / evidence["v3_run_manifest"]["path"])
    if (manifest.get("schema") != "jx-xp2-primary-run-manifest/v2"
            or manifest.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v3"
            or manifest.get("execution_label") != "A"
            or manifest.get("registration_sha256") != evidence["v3_registration"]["sha256"]):
        raise VerificationError("XP2-v3 run-manifest binding changed")
    return evidence


def reject_symlink_components(path: Path, label: str) -> Path:
    """Return a lexical absolute path only after every component passes lstat."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise IntegrityError(f"{label} component is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise IntegrityError(f"{label} contains a symlink component")
        if index < len(absolute.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise IntegrityError(f"{label} ancestor is not a directory")
    return absolute


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=unique_pairs,
            parse_float=finite_float, parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid held JSON: {label}") from exc
    if not isinstance(parsed, dict):
        raise IntegrityError(f"held JSON root is not an object: {label}")
    return parsed


class HeldEngineeringEvidence:
    """Retain an exact tree, its bytes, and every path binding through auth."""

    def __init__(
        self, root: Path, label: str, *, lock_fd: int | None = None,
        lock_relative: str | None = None,
    ) -> None:
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        absolute = Path(os.path.abspath(os.fspath(root)))
        self.selected_root = absolute
        self.directories = [os.open(absolute.anchor, flags)]
        self.directory_bindings: list[
            tuple[int, str, int, os.stat_result, bool]
        ] = []
        self.listings: list[tuple[int, list[str]]] = []
        self.files: dict[
            str, tuple[int, str, int, os.stat_result, str, bytes]
        ] = {}
        try:
            components = absolute.parts[1:]
            for index, component in enumerate(components):
                parent = self.directories[-1]
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
                child = os.open(component, flags, dir_fd=parent)
                opened = os.fstat(child)
                if (not stat.S_ISDIR(before.st_mode)
                        or not stat.S_ISDIR(opened.st_mode)
                        or before.st_dev != opened.st_dev
                        or before.st_ino != opened.st_ino):
                    os.close(child)
                    raise IntegrityError(f"{label} component binding changed")
                self.directories.append(child)
                self.directory_bindings.append(
                    (parent, component, child, opened, index == len(components) - 1)
                )
            self.root_fd = self.directories[-1]
            self.selected_root_metadata = os.fstat(self.root_fd)
            rows: list[list[Any]] = []

            def scan(directory_fd: int, prefix: str) -> None:
                names = sorted(os.listdir(directory_fd))
                self.listings.append((directory_fd, names))
                for name in names:
                    if not name or "/" in name or name in {".", ".."}:
                        raise IntegrityError(f"{label} contains an unsafe entry name")
                    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    relative = f"{prefix}/{name}" if prefix else name
                    if stat.S_ISDIR(before.st_mode):
                        child = os.open(name, flags, dir_fd=directory_fd)
                        opened = os.fstat(child)
                        if (not stat.S_ISDIR(opened.st_mode)
                                or opened.st_dev != before.st_dev
                                or opened.st_ino != before.st_ino):
                            os.close(child)
                            raise IntegrityError(f"{label} directory binding changed")
                        self.directories.append(child)
                        self.directory_bindings.append(
                            (directory_fd, name, child, opened, True)
                        )
                        rows.append([relative, "D"])
                        scan(child, relative)
                    elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                        descriptor = os.open(
                            name, os.O_RDONLY | (
                                os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
                            ), dir_fd=directory_fd,
                        )
                        opened = os.fstat(descriptor)
                        if (not stat.S_ISREG(opened.st_mode)
                                or opened.st_nlink != 1
                                or opened.st_dev != before.st_dev
                                or opened.st_ino != before.st_ino
                                or opened.st_size != before.st_size):
                            os.close(descriptor)
                            raise IntegrityError(f"{label} file binding changed")
                        payload = bytearray()
                        while True:
                            block = os.read(descriptor, 1024 * 1024)
                            if not block:
                                break
                            payload.extend(block)
                        frozen = bytes(payload)
                        digest = sha256_bytes(frozen)
                        self.files[relative] = (
                            directory_fd, name, descriptor, opened, digest, frozen,
                        )
                        rows.append([relative, "F", opened.st_size, digest])
                    else:
                        raise IntegrityError(
                            f"{label} contains a symlink, hardlink, or special file"
                        )

            scan(self.root_fd, "")
            self.rows = sorted(rows, key=lambda row: row[0])
            if (lock_fd is None) != (lock_relative is None):
                raise IntegrityError(f"{label} lock binding is incomplete")
            if lock_fd is not None and lock_relative is not None:
                if lock_relative not in self.files:
                    raise IntegrityError(f"{label} held lock is absent from tree")
                lock = os.fstat(lock_fd)
                opened = os.fstat(self.files[lock_relative][2])
                if (not stat.S_ISREG(lock.st_mode) or lock.st_nlink != 1
                        or lock.st_size != 0 or lock.st_dev != opened.st_dev
                        or lock.st_ino != opened.st_ino):
                    raise IntegrityError(f"{label} is not owned by its held lock")
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def payload(self, relative: str) -> bytes:
        try:
            return self.files[relative][5]
        except KeyError as exc:
            raise IntegrityError(f"{self.label} required file is absent") from exc

    def binding(self, relative: str, registered_path: str) -> dict[str, Any]:
        try:
            _parent, _name, _fd, opened, digest, _payload = self.files[relative]
        except KeyError as exc:
            raise IntegrityError(f"{self.label} required binding is absent") from exc
        return {
            "path": registered_path, "size_bytes": opened.st_size, "sha256": digest,
        }

    def fingerprint(self, prefix: str | None = None) -> dict[str, Any]:
        selected = self.rows if prefix is None else [
            [row[0][len(prefix) + 1:], *row[1:]]
            for row in self.rows if row[0].startswith(prefix + "/")
        ]
        return {
            "entry_count": len(selected),
            "sha256": sha256_bytes(canonical_bytes(selected)),
        }

    def require_selected_root(self, root: Path) -> None:
        """Reject reuse of a retained snapshot for any other selected root."""
        selected = Path(os.path.abspath(os.fspath(root)))
        if selected != self.selected_root:
            raise IntegrityError(f"{self.label} selected root changed")
        current = os.stat(selected, follow_symlinks=False)
        held = os.fstat(self.root_fd)
        if (not stat.S_ISDIR(current.st_mode)
                or current.st_dev != self.selected_root_metadata.st_dev
                or current.st_ino != self.selected_root_metadata.st_ino
                or held.st_dev != self.selected_root_metadata.st_dev
                or held.st_ino != self.selected_root_metadata.st_ino):
            raise IntegrityError(f"{self.label} selected root binding changed")

    def revalidate(self) -> None:
        stable = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        for directory_fd, names in self.listings:
            if sorted(os.listdir(directory_fd)) != names:
                raise IntegrityError(f"{self.label} inventory changed")
        for parent, name, descriptor, before, strict in self.directory_bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            keys = stable if strict else ("st_dev", "st_ino", "st_mode")
            if any(getattr(before, key) != getattr(after, key) for key in keys) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in keys):
                raise IntegrityError(f"{self.label} directory changed")
        for parent, name, descriptor, before, expected, _payload in self.files.values():
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if any(getattr(before, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise IntegrityError(f"{self.label} file changed")
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected:
                raise IntegrityError(f"{self.label} file content changed")

    def close(self) -> None:
        files = getattr(self, "files", {})
        for _parent, _name, descriptor, _before, _digest, _payload in reversed(
            list(files.values())
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.files = {}
        for descriptor in reversed(getattr(self, "directories", [])):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.directories = []


ENGINEERING_ARM_IDS = ("M0", "CI01-P0", "AUDIT-CI01-P0")


ENGINEERING_DT = {"M0": 0.125, "CI01-P0": 0.125, "AUDIT-CI01-P0": 0.0625}


ENGINEERING_N = {"M0": 133, "CI01-P0": 134, "AUDIT-CI01-P0": 134}


ENGINEERING_CONFIGURATION = {
    "M0": "M0", "CI01-P0": "CI01-P0", "AUDIT-CI01-P0": "CI01-P0",
}
ENGINEERING_PARTICLE_VECTOR_DIGEST_DOMAIN = (
    b"jx-xp2-v4-engineering-particle-vectors/v1\0"
)
ENGINEERING_PARTICLE_VECTOR_SHA256 = {
    "M0": "9e26615e682f4bfa1a18adddebd7254255421bafde7b9cd8d3b5ac20027a000c",
    "CI01-P0": "efd99bab9b58886fcb799b74c7e2af7313d0f384dc4195d752684d8007f1bbdd",
}


def engineering_binary64_from_hex(value: Any) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise IntegrityError("engineering binary64 hex width changed")
    try:
        parsed = struct.unpack(">d", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as exc:
        raise IntegrityError("engineering value is not binary64 hex") from exc
    if not math.isfinite(parsed):
        raise IntegrityError("engineering binary64 value is non-finite")
    return parsed


ENGINEERING_RUNNER_RECORD_FIELDS = frozenset({
    "schema", "experiment_id", "arm_id", "configuration_id", "dt_years",
    "pre_save_live_topology", "pre_save_unsaved_control_topology",
    "post_save_candidate_topology", "decoded_boundary_topology",
    "continued_live_topology", "continued_decoded_topology",
    "boundary_normalized_endpoint_equal",
    "saved_candidate_unsaved_control_pre_save_endpoint_equal",
    "pre_save_post_save_candidate_normalized_endpoint_equal",
    "restart_50050_normalized_endpoint_equal", "archives",
    "tracer_metrics_or_classification_emitted", "nonpromotable",
})


ENGINEERING_VERIFIER_RECORD_FIELDS = frozenset({
    "schema", "experiment_id", "arm_id", "dt_years", "boundary_years",
    "continuation_years", "archive_filename", "archive_size_bytes",
    "archive_sha256", "decoded_state_sha256",
    "live_pre_save_normalized_endpoint",
    "live_pre_save_normalized_endpoint_sha256",
    "decoded_archive_normalized_endpoint",
    "decoded_archive_normalized_endpoint_sha256",
    "live_pre_save_matches_decoded_archive", "normalized_endpoint_sha256",
    "matches_runner_restarted_archive",
    "scientific_metrics_or_classification_emitted", "nonpromotable",
})


def validate_engineering_endpoint(
    value: Any, *, configuration_id: str, particle_count: int,
    dt_years: float, end_years: float,
) -> str:
    if (particle_count not in {133, 134}
            or configuration_id not in ENGINEERING_PARTICLE_VECTOR_SHA256
            or particle_count != {"M0": 133, "CI01-P0": 134}[configuration_id]
            or dt_years not in {0.125, 0.0625}
            or end_years not in {50_000.0, 50_050.0}):
        raise IntegrityError("engineering endpoint context changed")
    if not isinstance(value, dict) or set(value) != {
        "schema", "simulation", "mercurius", "whfast", "ias15", "particles",
        "save_load_normalized_mercurius_fields",
        "save_load_normalized_whfast_fields", "save_load_normalized_ias15_fields",
        "excluded_noncontinuation_fields",
    } or value.get("schema") != "jx-xp2-mercurius-live-archive-endpoint/v1":
        raise IntegrityError("engineering normalized endpoint shape changed")
    normalized_mercurius = {
        "encounter_N", "encounter_N_active", "tponly_encounter",
        "allocated_particle_backup_count",
        "allocated_additional_forces_backup_count", "particles_backup_present",
        "additional_forces_backup_present", "encounter_map_present",
    }
    particle_fields = {
        "index", "hash", "simulation_reference_bound_to_parent", "m_hex", "r_hex",
        "x_hex", "y_hex", "z_hex", "vx_hex", "vy_hex", "vz_hex", "ax_hex",
        "ay_hex", "az_hex", "last_collision_hex", "collision_cell_present",
        "additional_properties_present",
    }
    if (not isinstance(value["simulation"], dict)
            or set(value["simulation"]) != set(CONTINUATION_SIMULATION_FIELDS)
            or not isinstance(value["mercurius"], dict)
            or set(value["mercurius"])
            != set(CONTINUATION_MERCURIUS_FIELDS) - normalized_mercurius
            or not isinstance(value["whfast"], dict)
            or set(value["whfast"])
            != set(CONTINUATION_WHFAST_FIELDS) - {"internal_particle_arrays_present"}
            or not isinstance(value["ias15"], dict)
            or set(value["ias15"]) != {
                "epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded",
            }
            or not isinstance(value["particles"], list)
            or not all(isinstance(item, dict) for item in value["particles"])
            or any(set(item) != particle_fields for item in value["particles"])
            or type(value["simulation"].get("N")) is not int
            or len(value["particles"]) != value["simulation"]["N"]
            or any(item["index"] != index or type(item["hash"]) is not int
                   or item["simulation_reference_bound_to_parent"] is not True
                   or type(item["collision_cell_present"]) is not bool
                   or type(item["additional_properties_present"]) is not bool
                   for index, item in enumerate(value["particles"]))
            or set(value["simulation"].get("callbacks_present", {}))
            != set(CONTINUATION_CALLBACK_FIELDS)
            or any(type(item) is not bool for item in value["simulation"][
                "callbacks_present"
            ].values())
            or any(isinstance(item, dict) and key != "callbacks_present"
                   for key, item in value["simulation"].items())
            or any(isinstance(item, (dict, list))
                   for item in value["whfast"].values())
            or any(isinstance(item, dict) for item in value["mercurius"].values())
            or any(isinstance(item, (dict, list)) for item in value["ias15"].values())
            or any(isinstance(child, (dict, list))
                   for section in (value["simulation"], value["mercurius"])
                   for item in section.values() if isinstance(item, list)
                   for child in item)
            or value["save_load_normalized_whfast_fields"]
            != ["internal_particle_arrays_present"]
            or value["save_load_normalized_ias15_fields"] != [
                "stored_coordinate_count", "direct_array_sha256",
                "coefficient_array_sha256", "map_count", "map_sha256",
            ]
            or value["save_load_normalized_mercurius_fields"] != [
                "encounter_N", "encounter_N_active", "tponly_encounter",
                "allocated_particle_backup_count",
                "allocated_additional_forces_backup_count",
                "particles_backup_present", "additional_forces_backup_present",
                "encounter_map_present",
            ]
            or value["excluded_noncontinuation_fields"]
            != list(CONTINUATION_EXCLUDED_FIELDS)):
        raise IntegrityError("engineering normalized endpoint content changed")
    simulation = value["simulation"]
    mercurius = value["mercurius"]
    whfast = value["whfast"]
    ias15 = value["ias15"]
    expected_active_count = particle_count - 128
    binary64_hex_value = lambda item: struct.pack(">d", float(item)).hex()
    simulation_integer_fields = {
        "steps_done", "save_messages", "status", "N", "N_var", "N_var_config",
        "var_rescale_warning", "N_active", "testparticle_type",
        "testparticle_hidewarnings", "hash_ctr", "particle_lookup_count",
        "particle_lookup_allocation_count", "exact_finish_time",
        "force_is_velocity_dependent", "gravity_ignore", "track_energy_offset",
        "N_root", "collision_resolve_keep_sorted", "collisions_N",
        "gravity_compensated_sums_allocation_count", "tree_needs_update",
        "collision_allocation_count", "collisions_log_n", "calculate_megno",
        "megno_n", "N_odes", "odes_allocation_count", "odes_warnings",
        "simulationarchive_auto_step", "simulationarchive_next_step",
    }
    simulation_boolean_fields = {
        "particle_capacity_covers_logical_count", "particle_storage_present",
        "active_memory_ranges_pairwise_disjoint", "variation_config_present",
        "particle_lookup_present", "gravity_compensated_sums_present",
        "tree_root_present", "messages_present", "display_view_present",
        "display_data_present", "server_data_present", "collision_storage_present",
        "odes_present", "extras_present", "simulationarchive_filename_present",
    }
    if (any(type(simulation[field]) is not int for field in simulation_integer_fields)
            or any(type(simulation[field]) is not bool
                   for field in simulation_boolean_fields)
            or simulation["N"] != particle_count
            or simulation["N_active"] != expected_active_count
            or not 0 < simulation["N_active"] <= simulation["N"]
            or simulation["t_hex"] != binary64_hex_value(end_years)
            or simulation["dt_hex"] != binary64_hex_value(dt_years)
            or simulation["dt_last_done_hex"] != binary64_hex_value(dt_years)
            or simulation["steps_done"] != int(round(end_years / dt_years))
            or simulation["G_hex"] != binary64_hex_value(39.47841760435743)
            or simulation["integrator"] != "mercurius"
            or simulation["gravity"] != "mercurius"
            or simulation["boundary"] != "none"
            or simulation["collision"] != "none"
            or simulation["testparticle_type"] != 0
            or simulation["softening_hex"] != binary64_hex_value(0.0)
            or simulation["usleep_hex"] != binary64_hex_value(0.0)
            or simulation["save_messages"] != 1 or simulation["status"] != 0
            or simulation["N_var"] != 0 or simulation["N_var_config"] != 0
            or simulation["variation_config_present"] is not False
            or simulation["var_rescale_warning"] != 0
            or simulation["testparticle_hidewarnings"] != 0
            or simulation["hash_ctr"] != 0
            or simulation["particle_lookup_count"] != 0
            or simulation["particle_lookup_allocation_count"] != 0
            or simulation["particle_lookup_present"] is not False
            or simulation["exact_finish_time"] != 1
            or simulation["force_is_velocity_dependent"] != 0
            or simulation["gravity_ignore"] != 0
            or simulation["exit_max_distance_hex"] != binary64_hex_value(0.0)
            or simulation["exit_min_distance_hex"] != binary64_hex_value(0.0)
            or simulation["track_energy_offset"] != 0
            or simulation["energy_offset_hex"] != binary64_hex_value(0.0)
            or simulation["opening_angle2_hex"] != binary64_hex_value(0.25)
            or simulation["boxsize_hex"] != [binary64_hex_value(0.0)] * 3
            or simulation["boxsize_max_hex"] != binary64_hex_value(0.0)
            or simulation["root_size_hex"] != binary64_hex_value(-1.0)
            or simulation["N_root"] != 1 or simulation["N_root_xyz"] != [1, 1, 1]
            or simulation["N_ghost_xyz"] != [0, 0, 0]
            or simulation["collision_resolve_keep_sorted"] != 0
            or simulation["collisions_N"] != 0
            or simulation["minimum_collision_velocity_hex"]
                != binary64_hex_value(0.0)
            or simulation["gravity_compensated_sums_present"] is not False
            or simulation["gravity_compensated_sums_allocation_count"] != 0
            or simulation["tree_root_present"] is not False
            or simulation["tree_needs_update"] != 0
            or simulation["messages_present"] is not False
            or simulation["display_view_present"] is not False
            or simulation["display_data_present"] is not False
            or simulation["server_data_present"] is not False
            or simulation["collision_storage_present"] is not False
            or simulation["collision_allocation_count"] != 0
            or simulation["collisions_plog_hex"] != binary64_hex_value(0.0)
            or simulation["collisions_log_n"] != 0
            or simulation["calculate_megno"] != 0
            or any(simulation[field] != binary64_hex_value(0.0) for field in (
                "megno_Ys_hex", "megno_Yss_hex", "megno_cov_Yt_hex",
                "megno_var_t_hex", "megno_mean_t_hex", "megno_mean_Y_hex",
                "megno_initial_t_hex",
            ))
            or simulation["megno_n"] != 0
            or simulation["N_odes"] != 0
            or simulation["odes_allocation_count"] != 0
            or simulation["odes_warnings"] != 0
            or simulation["odes_present"] is not False
            or simulation["extras_present"] is not False
            or simulation["simulationarchive_auto_interval_hex"]
                != binary64_hex_value(0.0)
            or simulation["simulationarchive_auto_walltime_hex"]
                != binary64_hex_value(0.0)
            or simulation["simulationarchive_auto_step"] != 0
            or simulation["simulationarchive_next_hex"] != binary64_hex_value(0.0)
            or simulation["simulationarchive_next_step"] != 0
            or simulation["simulationarchive_filename_present"] is not False
            or simulation["particle_capacity_covers_logical_count"] is not True
            or simulation["particle_storage_present"] is not True
            or simulation["active_memory_ranges_pairwise_disjoint"] is not True
            or any(simulation["callbacks_present"].values())
            or any(type(item) is not bool
                   for item in simulation["callbacks_present"].values())
            or type(mercurius["safe_mode"]) is not int
            or type(mercurius["mode"]) is not int
            or type(mercurius["is_synchronized"]) is not int
            or type(mercurius["recalculate_coordinates_this_timestep"]) is not int
            or type(mercurius["recalculate_r_crit_this_timestep"]) is not int
            or type(mercurius["dcrit_storage_present"]) is not bool
            or type(mercurius["dcrit_capacity_covers_logical_count"]) is not bool
            or type(mercurius["L_callback_present"]) is not bool
            or mercurius["r_crit_hill_hex"] != binary64_hex_value(3.0)
            or mercurius["safe_mode"] != 1 or mercurius["mode"] != 0
            or mercurius["is_synchronized"] != 1
            or mercurius["recalculate_coordinates_this_timestep"] != 1
            or mercurius["recalculate_r_crit_this_timestep"] != 0
            or mercurius["dcrit_storage_present"] is not True
            or mercurius["dcrit_capacity_covers_logical_count"] is not True
            or len(mercurius["dcrit_hex"]) != particle_count
            or mercurius["L_callback_present"] is not False
            or whfast != {
                "coordinates": "jacobi", "kernel": "default", "corrector": 0,
                "corrector2": 0, "recalculate_coordinates_this_timestep": 0,
                "safe_mode": 1, "keep_unsynchronized": 0, "is_synchronized": 1,
                "timestep_warning": 0, "unsynchronized_recalculation_warning": 0,
            }
            or ias15 != {
                "epsilon_hex": binary64_hex_value(1e-9),
                "min_dt_hex": binary64_hex_value(0.0), "adaptive_mode": "prs23",
                "iterations_max_exceeded": 0,
            }
            or len({item["hash"] for item in value["particles"]}) != particle_count
            or any(not 0 <= item["hash"] <= 0xFFFFFFFF
                   for item in value["particles"])
            or any(engineering_binary64_from_hex(item["m_hex"]) <= 0.0
                   for item in value["particles"][:expected_active_count])
            or any(engineering_binary64_from_hex(item["m_hex"]) != 0.0
                   for item in value["particles"][expected_active_count:])
            or any(engineering_binary64_from_hex(item["r_hex"]) != 0.0
                   for item in value["particles"])
            or any(engineering_binary64_from_hex(item) < 0.0
                   for item in mercurius["dcrit_hex"])
            or sha256_bytes(
                ENGINEERING_PARTICLE_VECTOR_DIGEST_DOMAIN + canonical_bytes([
                    [item["hash"], item["m_hex"], item["r_hex"]]
                    for item in value["particles"]
                ])
            ) != ENGINEERING_PARTICLE_VECTOR_SHA256[configuration_id]
            or any(item["last_collision_hex"] != binary64_hex_value(0.0)
                   or item["collision_cell_present"]
                   or item["additional_properties_present"]
                   for item in value["particles"])):
        raise IntegrityError("engineering normalized endpoint semantics changed")
    try:
        def validate_hex_payload(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    validate_hex_payload(child)
            else:
                engineering_binary64_from_hex(item)

        for section in (value["simulation"], value["mercurius"], value["ias15"]):
            for key, item in section.items():
                if key.endswith("_hex"):
                    validate_hex_payload(item)
        for item in value["particles"]:
            for key in particle_fields:
                if key.endswith("_hex"):
                    engineering_binary64_from_hex(item[key])
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError("engineering endpoint particle binary64 changed") from exc
    return sha256_bytes(ENDPOINT_DIGEST_DOMAIN + canonical_bytes(value))


def validate_engineering_topology(
    value: Any, *, source_mode: str, configuration_id: str, particle_count: int,
    dt_years: float, end_years: float,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source_mode", "structural_projection_validation_passed", "simulation",
        "mercurius", "whfast", "ias15", "normalized_endpoint_projection",
        "normalized_endpoint_sha256", "strict_projection_sha256",
    }:
        raise IntegrityError("engineering topology fields changed")
    simulation = value.get("simulation")
    mercurius = value.get("mercurius")
    whfast = value.get("whfast")
    ias15 = value.get("ias15")
    direct = {"at", "x0", "v0", "a0", "csx", "csv", "csa0"}
    coefficients = {"g", "b", "csb", "e", "br", "er"}
    endpoint_sha = validate_engineering_endpoint(
        value.get("normalized_endpoint_projection"), configuration_id=configuration_id,
        particle_count=particle_count,
        dt_years=dt_years, end_years=end_years,
    )
    if (value.get("source_mode") != source_mode
            or value.get("structural_projection_validation_passed") is not True
            or not isinstance(simulation, dict)
            or set(simulation) != {"N", "N_allocated", "particles_present"}
            or type(simulation["N"]) is not int
            or type(simulation["N_allocated"]) is not int
            or type(simulation["particles_present"]) is not bool
            or simulation["N"] != particle_count
            or not particle_count <= simulation["N_allocated"] <= MAX_REBOUND_ALLOCATION_CAPACITY
            or simulation["particles_present"] is not True
            or not isinstance(mercurius, dict)
            or set(mercurius) != {
                "dcrit_count", "dcrit_present", "allocated_particle_backup_count",
                "particles_backup_present", "encounter_map_present", "encounter_N",
                "encounter_N_active", "tponly_encounter",
                "allocated_additional_forces_backup_count",
                "additional_forces_backup_present",
            }
            or any(type(mercurius[key]) is not int for key in (
                "dcrit_count", "allocated_particle_backup_count", "encounter_N",
                "encounter_N_active", "tponly_encounter",
                "allocated_additional_forces_backup_count",
            ))
            or any(type(mercurius[key]) is not bool for key in (
                "dcrit_present", "particles_backup_present", "encounter_map_present",
                "additional_forces_backup_present",
            ))
            or mercurius["dcrit_count"] < particle_count
            or mercurius["dcrit_count"] > MAX_REBOUND_ALLOCATION_CAPACITY
            or mercurius["dcrit_present"] is not True
            or not 0 <= mercurius["allocated_particle_backup_count"] \
                <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not 0 <= mercurius["allocated_additional_forces_backup_count"] \
                <= MAX_REBOUND_ALLOCATION_CAPACITY
            or (mercurius["allocated_particle_backup_count"] == 0) \
                != (not mercurius["particles_backup_present"])
            or mercurius["encounter_map_present"] \
                != mercurius["particles_backup_present"]
            or (mercurius["allocated_additional_forces_backup_count"] == 0) \
                != (not mercurius["additional_forces_backup_present"])
            or (mercurius["particles_backup_present"]
                and mercurius["allocated_particle_backup_count"] < particle_count)
            or (mercurius["additional_forces_backup_present"]
                and mercurius["allocated_additional_forces_backup_count"]
                < particle_count)
            or not 0 <= mercurius["encounter_N_active"] \
                <= mercurius["encounter_N"] <= particle_count
            or mercurius["encounter_N_active"] \
                > value["normalized_endpoint_projection"]["simulation"]["N_active"]
            or mercurius["tponly_encounter"] not in {0, 1}
            or not isinstance(whfast, dict)
            or set(whfast) != {
                "particle_count", "particle_present", "temporary_count",
                "temporary_present", "internal_particle_arrays_present",
            }
            or type(whfast["particle_count"]) is not int
            or type(whfast["temporary_count"]) is not int
            or any(type(whfast[key]) is not bool for key in (
                "particle_present", "temporary_present",
                "internal_particle_arrays_present",
            ))
            or not 0 <= whfast["particle_count"] <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not 0 <= whfast["temporary_count"] <= MAX_REBOUND_ALLOCATION_CAPACITY
            or (whfast["particle_count"] == 0) != (not whfast["particle_present"])
            or (whfast["temporary_count"] == 0) != (not whfast["temporary_present"])
            or (whfast["particle_present"]
                and whfast["particle_count"] < particle_count)
            or (whfast["temporary_present"]
                and whfast["temporary_count"] < particle_count)
            or whfast["internal_particle_arrays_present"]
            != (whfast["particle_present"] or whfast["temporary_present"])
            or not isinstance(ias15, dict)
            or set(ias15) != {
                "stored_coordinate_count", "map_count", "map_present",
                "direct_pointer_presence", "coefficient_pointer_presence",
                "direct_array_sha256", "coefficient_array_sha256", "map_sha256",
            }
            or type(ias15["stored_coordinate_count"]) is not int
            or type(ias15["map_count"]) is not int
            or type(ias15["map_present"]) is not bool
            or not 0 <= ias15["stored_coordinate_count"] <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not 0 <= ias15["map_count"] <= MAX_REBOUND_ALLOCATION_CAPACITY
            or (ias15["map_count"] == 0) != (not ias15["map_present"])
            or set(ias15.get("direct_pointer_presence", {})) != direct
            or set(ias15.get("direct_array_sha256", {})) != direct
            or set(ias15.get("coefficient_pointer_presence", {})) != coefficients
            or set(ias15.get("coefficient_array_sha256", {})) != coefficients
            or any(type(item) is not bool
                   for item in ias15["direct_pointer_presence"].values())
            or any(item is not None and not valid_sha256(item)
                   for item in ias15["direct_array_sha256"].values())
            or any(not isinstance(rows, list) or len(rows) != 7
                   or any(type(item) is not bool for item in rows)
                   for rows in ias15["coefficient_pointer_presence"].values())
            or any(not isinstance(rows, list) or len(rows) != 7
                   or any(item is not None and not valid_sha256(item) for item in rows)
                   for rows in ias15["coefficient_array_sha256"].values())
            or (ias15["map_sha256"] is not None
                and not valid_sha256(ias15["map_sha256"]))
            or value.get("normalized_endpoint_sha256") != endpoint_sha
            or value["normalized_endpoint_projection"]["simulation"]["N"]
                != simulation["N"]
            or mercurius["dcrit_count"] < len(
                value["normalized_endpoint_projection"]["mercurius"]["dcrit_hex"]
            )
            or (source_mode == "ARCHIVE" and (
                mercurius["allocated_particle_backup_count"] != 0
                or mercurius["allocated_additional_forces_backup_count"] != 0
                or mercurius["particles_backup_present"]
                or mercurius["additional_forces_backup_present"]
                or mercurius["encounter_map_present"]
                or mercurius["encounter_N"] != 0
                or mercurius["encounter_N_active"] != 0
                or mercurius["tponly_encounter"] != 0
                or whfast["particle_count"] != 0 or whfast["temporary_count"] != 0
                or whfast["particle_present"] or whfast["temporary_present"]
                or whfast["internal_particle_arrays_present"]
                or ias15["stored_coordinate_count"] not in {0, 9}
                or ias15["map_count"] != 0 or ias15["map_present"]
            ))
            or (source_mode == "ARCHIVE"
                and not valid_sha256(value.get("strict_projection_sha256")))
            or (source_mode == "LIVE_BOUNDARY"
                and value.get("strict_projection_sha256") is not None)):
        raise IntegrityError("engineering topology content changed")
    ias_count = ias15["stored_coordinate_count"]
    if any(present != (ias_count > 0)
           for present in ias15["direct_pointer_presence"].values()) \
            or any(any(present != (ias_count > 0) for present in rows)
                   for rows in ias15["coefficient_pointer_presence"].values()) \
            or any((ias15["direct_array_sha256"][name] is not None) != present
                   for name, present in ias15["direct_pointer_presence"].items()) \
            or any(any((digest is not None) != present
                       for digest, present in zip(
                           ias15["coefficient_array_sha256"][name], rows,
                       )) for name, rows in ias15[
                           "coefficient_pointer_presence"
                       ].items()) \
            or (ias15["map_sha256"] is not None) != ias15["map_present"]:
        raise IntegrityError("engineering IAS15 pointer/count coherence changed")


def validate_engineering_runner_record(
    record: Any, arm: str, artifacts: dict[str, dict[str, Any]],
) -> None:
    if (not isinstance(record, dict) or set(record) != ENGINEERING_RUNNER_RECORD_FIELDS
            or record.get("schema") != "jx-xp2-v4-engineering-arm-result/v1"
            or record.get("experiment_id") != EXPERIMENT_ID
            or record.get("arm_id") != arm
            or record.get("configuration_id") != ENGINEERING_CONFIGURATION[arm]
            or record.get("dt_years") != ENGINEERING_DT[arm]
            or record.get("nonpromotable") is not True
            or record.get("tracer_metrics_or_classification_emitted") is not False
            or any(record.get(flag) is not True for flag in (
                "boundary_normalized_endpoint_equal",
                "saved_candidate_unsaved_control_pre_save_endpoint_equal",
                "pre_save_post_save_candidate_normalized_endpoint_equal",
                "restart_50050_normalized_endpoint_equal",
            ))):
        raise IntegrityError("engineering runner arm record identity changed")
    for key, end_years in (
        ("pre_save_live_topology", 50_000.0),
        ("pre_save_unsaved_control_topology", 50_000.0),
        ("post_save_candidate_topology", 50_000.0),
        ("continued_live_topology", 50_050.0),
        ("continued_decoded_topology", 50_050.0),
    ):
        validate_engineering_topology(
            record[key], source_mode="LIVE_BOUNDARY",
            configuration_id=ENGINEERING_CONFIGURATION[arm],
            particle_count=ENGINEERING_N[arm], dt_years=ENGINEERING_DT[arm],
            end_years=end_years,
        )
    validate_engineering_topology(
        record["decoded_boundary_topology"], source_mode="ARCHIVE",
        configuration_id=ENGINEERING_CONFIGURATION[arm],
        particle_count=ENGINEERING_N[arm], dt_years=ENGINEERING_DT[arm],
        end_years=50_000.0,
    )
    boundary = record["decoded_boundary_topology"]["normalized_endpoint_projection"]
    if (record["pre_save_live_topology"]["normalized_endpoint_projection"] != boundary
            or record["pre_save_unsaved_control_topology"][
                "normalized_endpoint_projection"
            ] != boundary
            or record["post_save_candidate_topology"][
                "normalized_endpoint_projection"
            ] != boundary
            or record["continued_live_topology"]["normalized_endpoint_projection"]
            != record["continued_decoded_topology"]["normalized_endpoint_projection"]):
        raise IntegrityError("engineering normalized endpoint parity changed")
    archives = record.get("archives")
    expected_names = {
        "boundary_50000.bin", "continued_live_50050.bin",
        "continued_decoded_50050.bin",
    }
    if not isinstance(archives, dict) or set(archives) != expected_names:
        raise IntegrityError("engineering archive inventory changed")
    for filename, value in archives.items():
        if (not isinstance(value, list) or len(value) != 3
                or type(value[0]) is not int or value[0] <= 0
                or not valid_sha256(value[1]) or not valid_sha256(value[2])
                or value[0] != artifacts[filename]["size_bytes"]
                or value[1] != artifacts[filename]["sha256"]):
            raise IntegrityError("engineering archive binding changed")
    if (archives["boundary_50000.bin"][2]
            != record["decoded_boundary_topology"]["strict_projection_sha256"]
            or archives["continued_live_50050.bin"][2]
            != archives["continued_decoded_50050.bin"][2]):
        raise IntegrityError("engineering decoded archive digest binding changed")


def validate_engineering_verifier_record(
    record: Any, arm: str, artifact: dict[str, Any], runner_record: dict[str, Any],
) -> None:
    if (not isinstance(record, dict) or set(record) != ENGINEERING_VERIFIER_RECORD_FIELDS
            or record.get("schema") != "jx-xp2-v4-engineering-verifier-arm/v1"
            or record.get("experiment_id") != EXPERIMENT_ID
            or record.get("arm_id") != arm
            or record.get("dt_years") != ENGINEERING_DT[arm]
            or record.get("boundary_years") != 50_000.0
            or record.get("continuation_years") != 50_050.0
            or record.get("archive_filename") != "independent_50050.bin"
            or record.get("archive_size_bytes") != artifact["size_bytes"]
            or record.get("archive_sha256") != artifact["sha256"]
            or record.get("decoded_state_sha256")
            != runner_record["archives"]["continued_decoded_50050.bin"][2]
            or record.get("live_pre_save_matches_decoded_archive") is not True
            or record.get("matches_runner_restarted_archive") is not True
            or record.get("scientific_metrics_or_classification_emitted") is not False
            or record.get("nonpromotable") is not True):
        raise IntegrityError("engineering verifier arm record identity changed")
    live = record.get("live_pre_save_normalized_endpoint")
    decoded = record.get("decoded_archive_normalized_endpoint")
    live_sha = validate_engineering_endpoint(
        live, configuration_id=ENGINEERING_CONFIGURATION[arm],
        particle_count=ENGINEERING_N[arm], dt_years=ENGINEERING_DT[arm],
        end_years=50_050.0,
    )
    decoded_sha = validate_engineering_endpoint(
        decoded, configuration_id=ENGINEERING_CONFIGURATION[arm],
        particle_count=ENGINEERING_N[arm], dt_years=ENGINEERING_DT[arm],
        end_years=50_050.0,
    )
    runner = runner_record["continued_decoded_topology"]
    if (live != decoded
            or decoded != runner["normalized_endpoint_projection"]
            or live_sha != decoded_sha
            or live_sha != record.get("live_pre_save_normalized_endpoint_sha256")
            or decoded_sha != record.get("decoded_archive_normalized_endpoint_sha256")
            or decoded_sha != record.get("normalized_endpoint_sha256")
            or decoded_sha != runner["normalized_endpoint_sha256"]):
        raise IntegrityError("engineering verifier endpoint binding changed")


def validate_engineering_evidence_inventory(
    runner: HeldEngineeringEvidence, verification: HeldEngineeringEvidence,
    scratch_name: str,
) -> None:
    runner_expected: dict[str, str] = {
        "execution.lock": "F", "runner_attempt_v1.json": "F",
        "result_v1.json": "F", "runner_terminal_v1.json": "F",
    }
    for arm in ENGINEERING_ARM_IDS:
        runner_expected[arm] = "D"
        for filename in (
            "execution.lock", "arm_result_v1.json", "boundary_50000.bin",
            "continued_live_50050.bin", "continued_decoded_50050.bin",
        ):
            runner_expected[f"{arm}/{filename}"] = "F"
    verification_expected: dict[str, str] = {
        "engineering_boundary_verifier_start_v1.json": "F",
        "engineering_boundary_receipt_v1.json": "F",
        "engineering_boundary_verifier_terminal_v1.json": "F",
        scratch_name: "D", f"{scratch_name}/execution.lock": "F",
    }
    for arm in ENGINEERING_ARM_IDS:
        verification_expected[f"{scratch_name}/{arm}"] = "D"
        for filename in (
            "execution.lock", "verification_arm_v1.json", "independent_50050.bin",
        ):
            verification_expected[f"{scratch_name}/{arm}/{filename}"] = "F"
    for snapshot, expected in (
        (runner, runner_expected), (verification, verification_expected),
    ):
        if {row[0]: row[1] for row in snapshot.rows} != expected:
            raise IntegrityError(f"{snapshot.label} exact inventory changed")
        for relative in (name for name in expected if name.endswith("execution.lock")):
            row = next(row for row in snapshot.rows if row[0] == relative)
            if row != [relative, "F", 0, sha256_bytes(b"")]:
                raise IntegrityError(f"{snapshot.label} execution lock changed")


def _validate_final_engineering_authorization_held(
    registration: dict[str, Any], contract: dict[str, Any], package_root: Path,
    package_snapshot: HeldEngineeringEvidence,
    runner_snapshot: HeldEngineeringEvidence,
    verification_snapshot: HeldEngineeringEvidence,
) -> None:
    if _ENGINEERING_RUNNER_GUARD_FD is None or _ENGINEERING_SCRATCH_GUARD_FD is None:
        raise IntegrityError("final engineering evidence locks are not held")
    gate = contract["engineering_boundary_gate_v1"]
    scratch_name = Path(gate["engineering_verifier_scratch_root"]).name
    runner_root_fp = runner_snapshot.fingerprint()
    scratch_root_fp = verification_snapshot.fingerprint(scratch_name)
    runner_arm_fps = {arm: {
        "schema": "jx-xp2-v4-engineering-arm-tree-fingerprint/v1",
        **runner_snapshot.fingerprint(arm),
    } for arm in ("M0", "CI01-P0", "AUDIT-CI01-P0")}
    scratch_arm_fps = {
        arm: verification_snapshot.fingerprint(f"{scratch_name}/{arm}")
        for arm in ("M0", "CI01-P0", "AUDIT-CI01-P0")
    }
    relative_paths = {
        "engineering_registration": "engineering_registration_v1.json",
        "runner_start": gate["engineering_output_root"] + "/runner_attempt_v1.json",
        "runner_result": gate["engineering_output_root"] + "/result_v1.json",
        "runner_terminal": gate["engineering_output_root"] + "/runner_terminal_v1.json",
        "verifier_start": gate["engineering_verifier_start_path"],
        "verifier_receipt": gate["engineering_verification_receipt_path"],
        "verifier_terminal": gate["engineering_verifier_terminal_path"],
    }
    def binding(name: str) -> dict[str, Any]:
        relative = relative_paths[name]
        if name == "engineering_registration":
            return package_snapshot.binding("engineering_registration_v1.json", relative)
        if name.startswith("runner_"):
            return runner_snapshot.binding(Path(relative).name, relative)
        return verification_snapshot.binding(Path(relative).name, relative)

    runner_arm_artifacts = {
        arm: {
            filename: runner_snapshot.binding(
                f"{arm}/{filename}",
                gate["engineering_output_root"] + f"/{arm}/{filename}",
            )
            for filename in (
                "arm_result_v1.json", "boundary_50000.bin",
                "continued_live_50050.bin", "continued_decoded_50050.bin",
            )
        } for arm in runner_arm_fps
    }
    verifier_arm_artifacts = {
        arm: {
            filename: verification_snapshot.binding(
                f"{scratch_name}/{arm}/{filename}",
                gate["engineering_verifier_scratch_root"] + f"/{arm}/{filename}",
            )
            for filename in ("verification_arm_v1.json", "independent_50050.bin")
        } for arm in scratch_arm_fps
    }

    expected_authority = {
        "schema": gate["final_registration_engineering_authorization_schema"],
        **{name: binding(name) for name in relative_paths},
        "runner_root_tree_fingerprint": runner_root_fp,
        "runner_arm_tree_fingerprints": runner_arm_fps,
        "runner_arm_artifacts": runner_arm_artifacts,
        "verifier_scratch_tree_fingerprint": scratch_root_fp,
        "verifier_arm_tree_fingerprints": scratch_arm_fps,
        "verifier_arm_artifacts": verifier_arm_artifacts,
    }
    if (set(registration) != {
            "schema", "experiment_id", "outcomes_generated",
            "scientific_evidence_artifact", "locked_files",
            "engineering_boundary_authorization",
        } or registration.get("engineering_boundary_authorization") != expected_authority):
        raise IntegrityError("final registration engineering authorization changed")
    engineering = strict_json_bytes(
        package_snapshot.payload("engineering_registration_v1.json"),
        "engineering registration",
    )
    runner_start = strict_json_bytes(
        runner_snapshot.payload("runner_attempt_v1.json"), "runner START",
    )
    runner_result = strict_json_bytes(
        runner_snapshot.payload("result_v1.json"), "runner result",
    )
    runner_terminal = strict_json_bytes(
        runner_snapshot.payload("runner_terminal_v1.json"), "runner PASS",
    )
    verifier_start = strict_json_bytes(
        verification_snapshot.payload(Path(relative_paths["verifier_start"]).name),
        "verifier START",
    )
    verifier_receipt = strict_json_bytes(
        verification_snapshot.payload(Path(relative_paths["verifier_receipt"]).name),
        "verifier receipt",
    )
    verifier_terminal = strict_json_bytes(
        verification_snapshot.payload(Path(relative_paths["verifier_terminal"]).name),
        "verifier PASS",
    )
    runner_arm_records = {
        arm: strict_json_bytes(
            runner_snapshot.payload(f"{arm}/arm_result_v1.json"),
            f"runner {arm} result",
        )
        for arm, artifacts in runner_arm_artifacts.items()
    }
    verifier_arm_records = {
        arm: strict_json_bytes(
            verification_snapshot.payload(
                f"{scratch_name}/{arm}/verification_arm_v1.json"
            ), f"verifier {arm} result",
        )
        for arm, artifacts in verifier_arm_artifacts.items()
    }
    for arm in ENGINEERING_ARM_IDS:
        validate_engineering_runner_record(
            runner_arm_records[arm], arm, runner_arm_artifacts[arm],
        )
        validate_engineering_verifier_record(
            verifier_arm_records[arm], arm,
            verifier_arm_artifacts[arm]["independent_50050.bin"],
            runner_arm_records[arm],
        )
    engineering_sha = expected_authority["engineering_registration"]["sha256"]
    core_files = set(registration["locked_files"]) - {"engineering_registration_v1.json"}
    core_rows = [[name, registration["locked_files"][name]] for name in sorted(core_files)]
    core_digest = sha256_bytes(canonical_bytes(core_rows))
    required_checks = {
        "registered_core_and_engineering_scope",
        "runner_START_result_PASS_bijection", "all_raw_archives_redecoded_and_strict",
        "required_allocator_cache_topology_coverage",
        "unsaved_control_vs_restarted_endpoint_parity",
        "three_independent_50050_restart_probes",
        "no_scientific_metrics_gates_labels_or_classification",
    }
    runner_records_valid = all(
        record.get("schema") == "jx-xp2-v4-engineering-arm-result/v1"
        and record.get("experiment_id") == EXPERIMENT_ID
        and record.get("arm_id") == arm
        and record.get("nonpromotable") is True
        and record.get("tracer_metrics_or_classification_emitted") is False
        and all(record.get(flag) is True for flag in (
            "boundary_normalized_endpoint_equal",
            "saved_candidate_unsaved_control_pre_save_endpoint_equal",
            "pre_save_post_save_candidate_normalized_endpoint_equal",
            "restart_50050_normalized_endpoint_equal",
        ))
        and set(record.get("archives", {})) == {
            "boundary_50000.bin", "continued_live_50050.bin",
            "continued_decoded_50050.bin",
        }
        and all(
            isinstance(record["archives"][filename], list)
            and len(record["archives"][filename]) == 3
            and record["archives"][filename][0]
            == runner_arm_artifacts[arm][filename]["size_bytes"]
            and record["archives"][filename][1]
            == runner_arm_artifacts[arm][filename]["sha256"]
            for filename in record["archives"]
        )
        for arm, record in runner_arm_records.items()
    )
    verifier_records_valid = all(
        record.get("schema") == "jx-xp2-v4-engineering-verifier-arm/v1"
        and record.get("experiment_id") == EXPERIMENT_ID
        and record.get("arm_id") == arm
        and record.get("archive_filename") == "independent_50050.bin"
        and record.get("archive_size_bytes")
        == verifier_arm_artifacts[arm]["independent_50050.bin"]["size_bytes"]
        and record.get("archive_sha256")
        == verifier_arm_artifacts[arm]["independent_50050.bin"]["sha256"]
        and record.get("live_pre_save_matches_decoded_archive") is True
        and record.get("matches_runner_restarted_archive") is True
        and record.get("scientific_metrics_or_classification_emitted") is False
        and record.get("nonpromotable") is True
        for arm, record in verifier_arm_records.items()
    )
    eligible = [runner_arm_records[arm] for arm in ("CI01-P0", "AUDIT-CI01-P0")]
    coverage_accessors = {
        "simulation.N_allocated": lambda row, side: row[side]["simulation"]["N_allocated"],
        "mercurius.allocated_particle_backup_count": (
            lambda row, side: row[side]["mercurius"]["allocated_particle_backup_count"]
        ),
        "mercurius.particles_backup_present": (
            lambda row, side: row[side]["mercurius"]["particles_backup_present"]
        ),
        "mercurius.encounter_map_present": (
            lambda row, side: row[side]["mercurius"]["encounter_map_present"]
        ),
    }
    expected_differences = {
        field: [
            row["arm_id"] for row in eligible
            if accessor(row, "pre_save_live_topology")
            != accessor(row, "decoded_boundary_topology")
        ] for field, accessor in coverage_accessors.items()
    }
    expected_coverage = {
        "required_live_decoded_differences": expected_differences,
        "whfast_live_present_decoded_absent_arms": [
            row["arm_id"] for row in eligible
            if row["pre_save_live_topology"]["whfast"]["internal_particle_arrays_present"]
            and not row["decoded_boundary_topology"]["whfast"]["internal_particle_arrays_present"]
        ],
        "ias15_live_stored_positive_map_zero_decoded_strict_arms": [
            row["arm_id"] for row in eligible
            if row["pre_save_live_topology"]["ias15"]["stored_coordinate_count"] > 0
            and row["pre_save_live_topology"]["ias15"]["map_count"] == 0
            and row["decoded_boundary_topology"]["ias15"]["map_count"] == 0
            and row["decoded_boundary_topology"]["structural_projection_validation_passed"]
            is True
        ],
        "status": "PASS",
    }
    if (runner_root_fp.get("entry_count") != 22
            or any(value.get("entry_count") != 5 for value in runner_arm_fps.values())
            or scratch_root_fp.get("entry_count") != 13
            or any(value.get("entry_count") != 3 for value in scratch_arm_fps.values())
            or set(engineering) != {
                "schema", "experiment_id", "artifact_class", "outcomes_generated",
                "scientific_evidence_artifact", "created_before_engineering_output",
                "final_registration_absent", "authorization", "locked_core_files",
                "core_inventory_sha256",
            }
            or engineering.get("schema") != "jx-xp2-v4-engineering-registration/v1"
            or engineering.get("experiment_id") != EXPERIMENT_ID
            or engineering.get("artifact_class")
            != "LOCAL_PREOUTPUT_ENGINEERING_BOUNDARY_REGISTRATION"
            or engineering.get("created_before_engineering_output") is not True
            or engineering.get("authorization")
            != gate["engineering_registration_authorizes_only"]
            or engineering.get("outcomes_generated") is not False
            or engineering.get("scientific_evidence_artifact") is not False
            or engineering.get("final_registration_absent") is not True
            or engineering.get("locked_core_files")
            != {name: registration["locked_files"][name] for name in core_files}
            or engineering.get("core_inventory_sha256") != core_digest
            or set(runner_start) != {
                "schema", "experiment_id", "event", "attempt_index",
                "engineering_registration_sha256", "core_inventory_sha256",
                "arm_ids", "resume_allowed", "scientific_output_authorized",
            }
            or runner_start.get("schema") != "jx-xp2-v4-engineering-attempt/v1"
            or runner_start.get("experiment_id") != EXPERIMENT_ID
            or runner_start.get("event") != "START"
            or runner_start.get("attempt_index") != 1
            or runner_start.get("engineering_registration_sha256") != engineering_sha
            or runner_start.get("core_inventory_sha256") != core_digest
            or runner_start.get("arm_ids") != list(runner_arm_fps)
            or runner_start.get("resume_allowed") is not False
            or runner_start.get("scientific_output_authorized") is not False
            or set(runner_result) != {
                "schema", "experiment_id", "status", "artifact_class",
                "engineering_registration_sha256", "core_inventory_sha256",
                "runner_start_sha256", "arm_tree_fingerprints", "arms",
                "required_topology_coverage",
                "scientific_outcomes_gates_labels_or_classification",
                "nonpromotable", "authorizes_official_execution",
            }
            or runner_result.get("schema") != "jx-xp2-v4-engineering-boundary-result/v1"
            or runner_result.get("experiment_id") != EXPERIMENT_ID
            or runner_result.get("status") != "PASS"
            or runner_result.get("artifact_class") != "NONSCIENTIFIC_ENGINEERING_DIAGNOSTIC"
            or runner_result.get("engineering_registration_sha256") != engineering_sha
            or runner_result.get("core_inventory_sha256") != core_digest
            or runner_result.get("runner_start_sha256")
            != expected_authority["runner_start"]["sha256"]
            or runner_result.get("authorizes_official_execution") is not False
            or runner_result.get("nonpromotable") is not True
            or runner_result.get("scientific_outcomes_gates_labels_or_classification") is not None
            or set(runner_result.get("arms", {})) != set(runner_arm_fps)
            or runner_result.get("arms") != runner_arm_records
            or not runner_records_valid
            or any(not arms for arms in expected_differences.values())
            or not expected_coverage["whfast_live_present_decoded_absent_arms"]
            or not expected_coverage[
                "ias15_live_stored_positive_map_zero_decoded_strict_arms"
            ]
            or runner_result.get("required_topology_coverage") != expected_coverage
            or runner_result.get("arm_tree_fingerprints") != runner_arm_fps
            or set(runner_terminal) != {
                "schema", "experiment_id", "event", "attempt_index", "start_sha256",
                "engineering_registration_sha256", "arm_tree_fingerprints",
                "result_filename", "result_size_bytes", "result_sha256",
                "scientific_output_emitted",
            }
            or runner_terminal.get("schema") != "jx-xp2-v4-engineering-attempt/v1"
            or runner_terminal.get("experiment_id") != EXPERIMENT_ID
            or runner_terminal.get("event") != "PASS"
            or runner_terminal.get("attempt_index") != 1
            or runner_terminal.get("start_sha256")
            != expected_authority["runner_start"]["sha256"]
            or runner_terminal.get("engineering_registration_sha256") != engineering_sha
            or runner_terminal.get("result_filename") != "result_v1.json"
            or runner_terminal.get("result_size_bytes")
            != expected_authority["runner_result"]["size_bytes"]
            or runner_terminal.get("result_sha256")
            != expected_authority["runner_result"]["sha256"]
            or runner_terminal.get("arm_tree_fingerprints") != runner_arm_fps
            or runner_terminal.get("scientific_output_emitted") is not False
            or set(verifier_start) != {
                "schema", "experiment_id", "event", "attempt_index",
                "engineering_registration_sha256", "core_inventory_sha256",
                "runner_start_sha256", "runner_result_sha256", "runner_terminal_sha256",
                "arm_ids", "resume_allowed", "scientific_output_authorized",
            }
            or verifier_start.get("schema")
            != "jx-xp2-v4-engineering-verifier-attempt/v1"
            or verifier_start.get("experiment_id") != EXPERIMENT_ID
            or verifier_start.get("event") != "START"
            or verifier_start.get("attempt_index") != 1
            or verifier_start.get("engineering_registration_sha256") != engineering_sha
            or verifier_start.get("core_inventory_sha256") != core_digest
            or verifier_start.get("runner_start_sha256")
            != expected_authority["runner_start"]["sha256"]
            or verifier_start.get("runner_result_sha256")
            != expected_authority["runner_result"]["sha256"]
            or verifier_start.get("runner_terminal_sha256")
            != expected_authority["runner_terminal"]["sha256"]
            or verifier_start.get("arm_ids") != list(runner_arm_fps)
            or verifier_start.get("resume_allowed") is not False
            or verifier_start.get("scientific_output_authorized") is not False
            or set(verifier_receipt) != {
                "schema", "experiment_id", "status", "artifact_class",
                "engineering_registration_sha256", "core_inventory_sha256",
                "runner_start_sha256", "runner_result_sha256", "runner_terminal_sha256",
                "runner_root_tree_fingerprint", "runner_arm_tree_fingerprints",
                "verifier_start_sha256", "verifier_scratch_tree_fingerprint",
                "verifier_arm_tree_fingerprints", "verifier_arm_results", "checks",
                "scientific_outcomes_gates_labels_or_classification", "nonpromotable",
                "final_registration_reference",
                "authorizes_final_registration_only_with_exact_verifier_PASS_terminal",
            }
            or verifier_receipt.get("schema")
            != "jx-xp2-v4-engineering-boundary-verification/v1"
            or verifier_receipt.get("experiment_id") != EXPERIMENT_ID
            or verifier_receipt.get("status") != "PASS"
            or verifier_receipt.get("artifact_class")
            != "INDEPENDENT_NONSCIENTIFIC_ENGINEERING_BOUNDARY_VERIFICATION"
            or verifier_receipt.get("engineering_registration_sha256") != engineering_sha
            or verifier_receipt.get("core_inventory_sha256") != core_digest
            or verifier_receipt.get("runner_start_sha256")
            != expected_authority["runner_start"]["sha256"]
            or verifier_receipt.get("runner_result_sha256")
            != expected_authority["runner_result"]["sha256"]
            or verifier_receipt.get("runner_terminal_sha256")
            != expected_authority["runner_terminal"]["sha256"]
            or verifier_receipt.get("verifier_start_sha256")
            != expected_authority["verifier_start"]["sha256"]
            or verifier_receipt.get("runner_root_tree_fingerprint") != runner_root_fp
            or verifier_receipt.get("runner_arm_tree_fingerprints") != runner_arm_fps
            or verifier_receipt.get("verifier_scratch_tree_fingerprint") != scratch_root_fp
            or verifier_receipt.get("verifier_arm_tree_fingerprints") != scratch_arm_fps
            or verifier_receipt.get("final_registration_reference") is not None
            or verifier_receipt.get("nonpromotable") is not True
            or verifier_receipt.get("scientific_outcomes_gates_labels_or_classification") is not None
            or verifier_receipt.get(
                "authorizes_final_registration_only_with_exact_verifier_PASS_terminal"
            ) is not True
            or set(verifier_receipt.get("checks", {})) != required_checks
            or any(value is not True for value in verifier_receipt["checks"].values())
            or set(verifier_receipt.get("verifier_arm_results", {})) != set(scratch_arm_fps)
            or verifier_receipt.get("verifier_arm_results") != verifier_arm_records
            or not verifier_records_valid
            or set(verifier_terminal) != {
                "schema", "experiment_id", "event", "attempt_index", "start_sha256",
                "engineering_registration_sha256", "receipt_filename",
                "receipt_size_bytes", "receipt_sha256",
                "verifier_scratch_tree_fingerprint", "scientific_output_emitted",
            }
            or verifier_terminal.get("schema")
            != "jx-xp2-v4-engineering-verifier-attempt/v1"
            or verifier_terminal.get("experiment_id") != EXPERIMENT_ID
            or verifier_terminal.get("event") != "PASS"
            or verifier_terminal.get("attempt_index") != 1
            or verifier_terminal.get("start_sha256")
            != expected_authority["verifier_start"]["sha256"]
            or verifier_terminal.get("engineering_registration_sha256") != engineering_sha
            or verifier_terminal.get("receipt_filename")
            != Path(relative_paths["verifier_receipt"]).name
            or verifier_terminal.get("receipt_size_bytes")
            != expected_authority["verifier_receipt"]["size_bytes"]
            or verifier_terminal.get("receipt_sha256")
            != expected_authority["verifier_receipt"]["sha256"]
            or verifier_terminal.get("verifier_scratch_tree_fingerprint") != scratch_root_fp
            or verifier_terminal.get("scientific_output_emitted") is not False):
        raise IntegrityError("final engineering PASS evidence is inconsistent")


def validate_final_engineering_authorization(
    registration: dict[str, Any], contract: dict[str, Any], package_root: Path,
) -> None:
    if _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS:
        package_snapshot, runner_snapshot, verification_snapshot = (
            _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS
        )
        gate = contract["engineering_boundary_gate_v1"]
        runner_root = package_root / gate["engineering_output_root"]
        scratch_root = package_root / gate["engineering_verifier_scratch_root"]
        package_snapshot.require_selected_root(package_root)
        runner_snapshot.require_selected_root(runner_root)
        verification_snapshot.require_selected_root(scratch_root.parent)
        for snapshot in _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS:
            snapshot.revalidate()
        _validate_final_engineering_authorization_held(
            registration, contract, package_root, package_snapshot,
            runner_snapshot, verification_snapshot,
        )
        for snapshot in _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS:
            snapshot.revalidate()
        return
    gate = contract["engineering_boundary_gate_v1"]
    runner_root = reject_symlink_components(
        package_root / gate["engineering_output_root"], "engineering runner evidence",
    )
    scratch_root = reject_symlink_components(
        package_root / gate["engineering_verifier_scratch_root"],
        "engineering verifier scratch evidence",
    )
    verification_root = reject_symlink_components(
        scratch_root.parent, "engineering verification evidence",
    )
    snapshots: list[HeldEngineeringEvidence] = []
    try:
        package_snapshot = HeldEngineeringEvidence(
            package_root, "registered v4 package",
        )
        snapshots.append(package_snapshot)
        runner_snapshot = HeldEngineeringEvidence(
            runner_root, "engineering runner evidence",
            lock_fd=_ENGINEERING_RUNNER_GUARD_FD, lock_relative="execution.lock",
        )
        snapshots.append(runner_snapshot)
        scratch_name = scratch_root.name
        verification_snapshot = HeldEngineeringEvidence(
            verification_root, "engineering verification evidence",
            lock_fd=_ENGINEERING_SCRATCH_GUARD_FD,
            lock_relative=f"{scratch_name}/execution.lock",
        )
        snapshots.append(verification_snapshot)
        inventory = set(contract["result_policy"]["registered_package_inventory"])
        if ({row[0]: row[1] for row in package_snapshot.rows}
                != {name: "F" for name in inventory}
                or len(inventory) != 17
                or strict_json_bytes(
                    package_snapshot.payload("registration_v1.json"),
                    "final registration",
                ) != registration
                or strict_json_bytes(
                    package_snapshot.payload("contract_v1.json"), "contract",
                ) != contract
                or set(registration.get("locked_files", {}))
                != inventory - {"registration_v1.json"}
                or any(
                    package_snapshot.files[name][4] != digest
                    for name, digest in registration["locked_files"].items()
                )):
            raise IntegrityError("retained registered package inventory changed")
        validate_engineering_evidence_inventory(
            runner_snapshot, verification_snapshot, scratch_name,
        )
        _validate_final_engineering_authorization_held(
            registration, contract, package_root, package_snapshot,
            runner_snapshot, verification_snapshot,
        )
        for snapshot in snapshots:
            snapshot.revalidate()
        _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS.extend(snapshots)
        snapshots = []
    finally:
        for snapshot in reversed(snapshots):
            snapshot.close()


def revalidate_final_engineering_evidence() -> None:
    if len(_FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS) != 3:
        raise IntegrityError("final engineering evidence snapshot is not retained")
    for snapshot in _FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS:
        snapshot.revalidate()


def validate_package(
    contract_path: Path, seed_path: Path, selection_path: Path,
    initial_path: Path, registration_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    global _ENGINEERING_RUNNER_GUARD_FD, _ENGINEERING_SCRATCH_GUARD_FD
    contract = read_object(contract_path)
    seed_manifest = read_object(seed_path)
    registration = read_object(registration_path)
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("experiment_id") != EXPERIMENT_ID:
        raise VerificationError("contract identity changed")
    root = registration_path.resolve().parent
    if (contract.get("xp2_v2_invalid_replay_lineage", {}).get("v2_b_execution_lock_path")
            != "../jx_xp2_runs_v2/output_b/execution.lock"
            or contract.get("xp2_v3_failed_startup_lineage", {}).get(
                "v3_a_execution_lock_path"
            ) != "../jx_xp2_runs_v3/output_a/execution.lock"):
        raise VerificationError("lineage lock path changed before acquisition")
    verify_unlocked_execution_lock(
        root / "../jx_xp2_runs_v2/output_b/execution.lock", "XP2-v2 B lineage lock"
    )
    verify_unlocked_execution_lock(
        root / "../jx_xp2_runs_v3/output_a/execution.lock", "XP2-v3 A lineage lock"
    )
    gate = contract.get("engineering_boundary_gate_v1", {})
    if gate != expected_engineering_boundary_gate_v1():
        raise VerificationError("engineering boundary gate changed before evidence locking")
    engineering_runner_root = root / gate["engineering_output_root"]
    engineering_scratch_root = root / gate["engineering_verifier_scratch_root"]
    _ENGINEERING_RUNNER_GUARD_FD = verify_unlocked_execution_lock(
        engineering_runner_root / "execution.lock", "engineering runner evidence lock",
    )
    _ENGINEERING_SCRATCH_GUARD_FD = verify_unlocked_execution_lock(
        engineering_scratch_root / "execution.lock", "engineering scratch evidence lock",
    )
    for arm in ("M0", "CI01-P0", "AUDIT-CI01-P0"):
        verify_unlocked_execution_lock(
            engineering_runner_root / arm / "execution.lock",
            f"engineering runner {arm} evidence lock",
        )
        verify_unlocked_execution_lock(
            engineering_scratch_root / arm / "execution.lock",
            f"engineering scratch {arm} evidence lock",
        )
    checkpoint = contract.get("checkpoint_and_resume", {})
    result_policy = contract.get("result_policy", {})
    if (
        contract.get("registration_status")
        != "ENGINEERING_BOUNDARY_PASS_AND_FINAL_LOCAL_HASH_LOCK_REQUIRED_BEFORE_FIRST_OFFICIAL_V4_SCIENTIFIC_OUTPUT"
        or contract.get("v4_fresh_execution_and_operational_repair")
        != expected_v4_fresh_repair_declaration()
        or contract.get("engineering_boundary_gate_v1")
        != expected_engineering_boundary_gate_v1()
        or checkpoint.get("primary_semantic_segment_chain_v3")
        != "SHA256_OVER_PREVIOUS_SEMANTIC_CHAIN_AND_CANONICAL_DETERMINISTIC_DECODED_INTEGRATOR_STATE_PLUS_SAMPLED_SCIENTIFIC_PAYLOAD_ONLY; RAW_REBOUND_ARCHIVE_FILENAME_SIZE_AND_SHA256_ARE_EXCLUDED"
        or checkpoint.get("primary_semantic_segment_fields_v3")
        != list(PRIMARY_SEGMENT_SEMANTIC_FIELD_ORDER)
        or checkpoint.get("raw_rebound_archive_identity_policy_v3")
        != "INTEGRITY_ONLY_REQUIRED_FILENAME_SIZE_SHA256_AND_DECODED_STATE_VALIDATION; FORBIDDEN_AS_SEMANTIC_REPLAY_IDENTITY"
        or checkpoint.get("decoded_integrator_state_identity_policy_v3")
        != "REQUIRED_AS_SEMANTIC_REPLAY_IDENTITY_AND_RECOMPUTED_FROM_EVERY_RAW_ARCHIVE_AFTER_PENDING_SAVE_FSYNC"
        or checkpoint.get("decoded_continuation_state_v3")
        != expected_continuation_declaration_v3()
        or checkpoint.get("max_primary_checkpoint_bytes") != MAX_PRIMARY_CHECKPOINT_BYTES
        or checkpoint.get("primary_failure_receipt_v4")
        != "DETERMINISTIC_ATTEMPT_KEY_FILENAME; RECEIPT_FSYNCED_BEFORE_FAIL_LEDGER_PUBLICATION; STABLE_START_BOUND_EVENT_DIGEST; EXACT_CLASS_RETURN_CODE_AND_QUARANTINE_BINDING; EVERY_FULLY_PUBLISHED_ATTEMPT_ACROSS_SOURCE_PENDING_SPLIT_OR_QUARANTINE_CRASH_CUTS_BINDS_CANONICAL_DECODED_AND_SAMPLED_SEMANTIC_EVIDENCE_IN_RECEIPT_AND_FAIL; LATER_PASS_MUST_MATCH_EVERY_COMPLETE_FAILED_ATTEMPT_WHILE_RAW_ARCHIVE_BYTES_MAY_DIFFER; TORN_UNPUBLISHED_FAILURE_RECEIPT_BYTES_ARE_PRESERVED_AND_HASH_BOUND_IN_RECOVERED_QUARANTINE; RECEIPT_WITHOUT_FAIL_RECONCILED_ON_RESUME; FAIL_WITHOUT_RECEIPT_OR_ANY_EXTRA_REJECTED"
        or checkpoint.get("retry_identical_inputs_decoded_state_scientific_payload_and_seed_only")
        is not True
        or checkpoint.get("retry_raw_archive_byte_equality_required") is not False
        or result_policy.get("primary_semantic_payload_excludes_raw_rebound_archive_filename_size_and_sha256")
        is not True
        or result_policy.get("primary_raw_rebound_archive_integrity_validation_required")
        is not True
        or result_policy.get("verification_receipt_publication_v4")
        != "HELD_INPUT_AND_LINEAGE_LOCKS_THROUGH_FSYNCED_ATOMIC_PUBLICATION; EXACT_FINAL_IS_IDEMPOTENT; COMPLETE_PENDING_IS_PROMOTED; EXACT_PAYLOAD_PREFIX_IS_DISCARDED_AND_REBUILT; DIVERGENT_PENDING_OR_FINAL_FAILS_CLOSED"
        or result_policy.get("primary_raw_artifact_integrity_v1")
        != expected_raw_artifact_declaration_v1()
        or result_policy.get("primary_failure_receipt_schema") != PRIMARY_FAILURE_SCHEMA
        or result_policy.get("primary_attempt_ledger_schema") != PRIMARY_ATTEMPT_SCHEMA
        or result_policy.get("primary_checkpoint_receipt_schema") != RECEIPT_SCHEMA
        or result_policy.get("primary_segment_commit_schema") != COMMIT_SCHEMA
        or result_policy.get("primary_result_schema") != PRIMARY_RESULT_SCHEMA
        or result_policy.get("primary_semantic_schema") != "jx-xp2-primary-semantic/v3"
        or checkpoint.get("dop853_failure_receipt_v2")
        != "DETERMINISTIC_ARM_SEGMENT_ATTEMPT_FILENAME; RECEIPT_FSYNCED_BEFORE_SEGMENT_ATTEMPT_FAILED_LEDGER_PUBLICATION; START_SEQUENCE_BOUND_EVENT_DIGEST; EXACT_CLOSED_FAILURE_TO_RECEIPT_FILENAME_SHA256_AND_CLASS_BIJECTION; COMPLETE_ORPHAN_RECEIPT_RECONCILED; PARTIAL_UNPUBLISHED_RECEIPT_BECOMES_INTERRUPTED_ATTEMPT; MISSING_EXTRA_DUPLICATE_OR_TAMPERED_RECEIPT_REJECTED"
        or checkpoint.get("dop853_failure_classes_v2") != [
            "InterruptedAttempt", "IntegrityError", "NumericalError",
            "ResourceLimitError", "UnexpectedFailure",
        ]
        or result_policy.get("dop853_failure_receipt_bijection_required") is not True
        or result_policy.get("dop853_attempt_ledger_schema") != DOP_ATTEMPT_SCHEMA
        or result_policy.get("dop853_failure_receipt_schema") != DOP_FAILURE_SCHEMA
    ):
        raise VerificationError("v4 semantic/raw or DOP853 failure protocol declaration changed")
    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("experiment_id") != EXPERIMENT_ID:
        raise VerificationError("registration identity changed")
    if registration.get("outcomes_generated") is not False:
        raise VerificationError("registration was not pre-output")
    locked = registration.get("locked_files")
    inventory = set(contract["result_policy"]["registered_package_inventory"])
    if not isinstance(locked, dict) or inventory != set(locked) | {"registration_v1.json"}:
        raise VerificationError("registered inventory changed")
    root_entries = list(root.iterdir())
    if ({item.name for item in root_entries} != inventory
            or any(item.is_symlink() or not item.is_file() or item.stat().st_nlink != 1
                   for item in root_entries)):
        raise VerificationError("registered package has an extra or missing file")
    for name, expected in locked.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise VerificationError("unsafe registered file")
        if digest_file(path) != expected:
            raise VerificationError("registered file hash mismatch")
    validate_final_engineering_authorization(registration, contract, root)
    canonical_paths = {
        "contract_v1.json": contract_path, "seed_manifest_v1.json": seed_path,
        "selection_manifest_v1.json": selection_path, "initial_states_v1.json": initial_path,
        "verify_replay.py": Path(__file__).resolve(),
    }
    if any(path.resolve() != root / name for name, path in canonical_paths.items()):
        raise VerificationError("verifier input is not a canonical registered package path")
    design_hash = digest_bytes(canonical(contract["design_core"]))
    if design_hash != contract["seed_policy"]["design_core_sha256"]:
        raise VerificationError("design-core digest mismatch")
    historical = contract["xp1_historical_binding"]
    xp1_registration = (root / historical["registration_path"]).resolve()
    if digest_file(xp1_registration) != historical["registration_sha256"]:
        raise VerificationError("XP1 read-only registration binding mismatch")
    lineage = contract["xp2_v1_invalid_protocol_lineage"]
    v1_registration = root / lineage["v1_registration_path"]
    v1_diagnostic = root / lineage["v1_final_invalid_diagnostic_path"]
    if (
        lineage["role"]
        != "INVALID_PROTOCOL_DIAGNOSTIC_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT"
        or lineage["v1_output_or_outcome_consumed"] is not False
        or lineage["v1_scientific_classification_consumed"] is not False
        or lineage["protocol_repair_only_scientific_design_unchanged"] is not True
        or v1_registration.is_symlink() or not v1_registration.is_file()
        or v1_registration.stat().st_nlink != 1
        or digest_file(v1_registration) != lineage["v1_registration_sha256"]
        or v1_diagnostic.is_symlink() or not v1_diagnostic.is_file()
        or v1_diagnostic.stat().st_nlink != 1
        or v1_diagnostic.stat().st_size != lineage["v1_final_invalid_diagnostic_size_bytes"]
        or digest_file(v1_diagnostic) != lineage["v1_final_invalid_diagnostic_sha256"]
    ):
        raise VerificationError("XP2-v1 invalid-protocol lineage binding changed")
    validate_v2_replay_lineage(contract, root)
    validate_v3_failed_startup_lineage(contract, root)
    selection = read_object(selection_path)
    initial = read_object(initial_path)
    science = contract["frozen_v1_scientific_design_inputs"]
    if (
        science["role"]
        != "EXACT_SCIENTIFIC_DESIGN_INPUT_REUSE_WITH_V4_EXECUTION_PROTOCOL_ONLY"
        or science["scientific_design_experiment_id"]
        != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or science["tracer_or_state_bytes_regenerated_for_v4"] is not False
        or science["v1_dynamics_or_outcomes_consumed"] is not False
        or digest_file(seed_path) != science["seed_manifest_sha256"]
        or digest_file(selection_path) != science["selection_manifest_sha256"]
        or digest_file(initial_path) != science["initial_states_sha256"]
    ):
        raise VerificationError("frozen v1 scientific-design input lineage changed")
    if (seed_manifest.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID
            or selection.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID
            or initial.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID):
        raise VerificationError("frozen v1 scientific-design identity changed")
    if digest_file(initial_path) != contract["initial_state_policy"]["artifact_sha256"]:
        raise VerificationError("initial-state artifact digest mismatch")
    verify_source_rows(contract, seed_manifest, initial)
    return contract, seed_manifest, selection, initial


def reconstruct_states(
    initial: dict[str, Any], contract: dict[str, Any], seed_manifest: dict[str, Any]
) -> dict[str, list[list[Any]]]:
    verify_source_rows(contract, seed_manifest, initial)
    common = initial["common_active_sun_centered_rows"]
    tracers = initial["tracer_sun_centered_rows"]
    output: dict[str, list[list[Any]]] = {}
    index: list[list[str]] = []
    for config in initial["configuration_states"]:
        arm_id, active_count, added, pos_hex, vel_hex, expected = config
        active_rows = common + ([] if added is None else [added])
        masses = [f64(row[2]) for row in active_rows]
        states = [unpack6(row[3]) for row in active_rows]
        total_mass = math.fsum(masses)
        position = [math.fsum(mass * state[axis]
                              for mass, state in zip(masses, states, strict=True)) / total_mass
                    for axis in range(3)]
        velocity = [math.fsum(mass * state[axis + 3]
                              for mass, state in zip(masses, states, strict=True)) / total_mass
                    for axis in range(3)]
        if "".join(struct.pack(">d", value).hex() for value in position) != pos_hex \
                or "".join(struct.pack(">d", value).hex() for value in velocity) != vel_hex:
            raise VerificationError("independently recomputed active COM bytes changed")
        source = active_rows + tracers
        if len(source) != active_count + 128:
            raise VerificationError("expanded state cardinality changed")
        rows = []
        for logical_id, role, mass, packed in source:
            state = unpack6(packed)
            shifted = [state[axis] - position[axis] for axis in range(3)] + [
                state[axis + 3] - velocity[axis] for axis in range(3)
            ]
            rows.append([logical_id, role, mass, pack6(shifted)])
        actual = digest_bytes(EXPANDED_DOMAIN + canonical(rows))
        if actual != expected:
            raise VerificationError("expanded initial-state hash mismatch")
        output[arm_id] = rows; index.append([arm_id, actual])
    if set(output) != set(PRIMARY_ARMS):
        raise VerificationError("expanded initial-state arm matrix changed")
    if digest_bytes(INDEX_DOMAIN + canonical(index)) != initial["configuration_digest_index_sha256"]:
        raise VerificationError("initial-state digest index mismatch")
    return output


def selected_ids(selection: dict[str, Any], initial: dict[str, Any]) -> list[str]:
    tracer_ids = [row[0] for row in initial["tracer_sun_centered_rows"]]
    expected: list[str] = []
    for block in range(8):
        ids = [logical_id for logical_id in tracer_ids if logical_id.startswith(f"XP2-B{block:02d}-T")]
        ranked = sorted((digest_bytes(SELECTION_DOMAIN + logical_id.encode("ascii")), logical_id)
                        for logical_id in ids)[:4]
        manifest_row = selection["sentinels_by_block"][str(block)]
        if manifest_row["ordered_logical_ids"] != [logical_id for _digest, logical_id in ranked]:
            raise VerificationError("DOP853 sentinel selection changed")
        if manifest_row["rank_sha256"] != [digest for digest, _logical_id in ranked]:
            raise VerificationError("DOP853 selection rank hashes changed")
        expected.extend(manifest_row["ordered_logical_ids"])
    if selection["ordered_logical_ids"] != expected or selection["selected_arm_ids"] != list(DOP_ARMS):
        raise VerificationError("DOP853 selected IDs/arms changed")
    return expected


def primary_active_invariants(rows: Sequence[Sequence[Any]], G: float) -> dict[str, Any]:
    active = [row for row in rows if row[1] == "A"]
    masses = [f64(row[2]) for row in active]
    states = [unpack6(row[3]) for row in active]
    total_mass = math.fsum(masses)
    positions = [state[:3] for state in states]; velocities = [state[3:] for state in states]
    momentum = [math.fsum(mass * velocities[index][axis]
                          for index, mass in enumerate(masses)) for axis in range(3)]
    r_com = [math.fsum(mass * positions[index][axis]
                       for index, mass in enumerate(masses)) / total_mass for axis in range(3)]
    v_com = [value / total_mass for value in momentum]
    angular = []; kinetic = []; scale = []
    for mass, position, velocity in zip(masses, positions, velocities, strict=True):
        r = [position[i] - r_com[i] for i in range(3)]
        v = [velocity[i] - v_com[i] for i in range(3)]
        angular.append([mass * (r[1] * v[2] - r[2] * v[1]),
                        mass * (r[2] * v[0] - r[0] * v[2]),
                        mass * (r[0] * v[1] - r[1] * v[0])])
        kinetic.append(0.5 * mass * math.fsum(value * value for value in v))
        scale.append(mass * math.sqrt(math.fsum(value * value for value in v)))
    potential = []
    for left in range(len(active)):
        for right in range(left + 1, len(active)):
            distance = math.sqrt(math.fsum(
                (positions[left][axis] - positions[right][axis]) ** 2 for axis in range(3)
            ))
            potential.append(-G * masses[left] * masses[right] / distance)
    return {
        "momentum": momentum, "r_com": r_com, "v_com": v_com,
        "com_angular": [math.fsum(row[axis] for row in angular) for axis in range(3)],
        "linear_internal_scale": math.fsum(scale),
        "intrinsic_energy": math.fsum((math.fsum(kinetic), math.fsum(potential))),
    }


_VERIFIER_REBOUND: Any | None = None


def verifier_rebound(contract: dict[str, Any]) -> Any:
    global _VERIFIER_REBOUND
    if _VERIFIER_REBOUND is None:
        module = importlib.import_module("rebound")
        root = Path(module.__file__).resolve().parent
        files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
        tree = hashlib.sha256(REBOUND_TREE_DOMAIN)
        for path in files:
            relative = path.relative_to(root).as_posix().encode(); payload = path.read_bytes()
            tree.update(len(relative).to_bytes(4, "big")); tree.update(relative)
            tree.update(len(payload).to_bytes(8, "big")); tree.update(payload)
        runtime = contract["runtime_lock"]
        binary = Path(module.clibrebound._name).resolve()
        actual = {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "python_executable_sha256": digest_file(Path(sys.executable).resolve()),
            "rebound_version": module.__version__, "rebound_build": module.__build__,
            "rebound_binary_sha256": digest_file(binary),
            "rebound_python_source_file_count": len(files),
            "rebound_python_source_sha256": tree.hexdigest(),
        }
        if actual != {key: runtime[key] for key in actual}:
            raise VerificationError("verifier REBOUND runtime differs from frozen primary runtime")
        _VERIFIER_REBOUND = module
    return _VERIFIER_REBOUND


def decoded_primary_continuation_projection(
    simulation: Any, *, source_mode: str = "ARCHIVE",
) -> dict[str, Any]:
    if source_mode not in {"ARCHIVE", "LIVE_BOUNDARY"}:
        raise VerificationError("unknown continuation projection source mode")
    def h(value: Any) -> str:
        number = float(value)
        if not math.isfinite(number):
            raise VerificationError("decoded continuation contains a non-finite value")
        return struct.pack(">d", number).hex()

    mercurius = simulation.ri_mercurius
    whfast = simulation.ri_whfast
    ias15 = simulation.ri_ias15
    box = simulation.boxsize
    particle_count = int(simulation.N)
    particle_capacity = int(simulation.N_allocated)
    if (particle_count not in {133, 134}
            or not particle_count <= particle_capacity <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not bool(simulation._particles)):
        raise VerificationError("decoded particle allocation/count is unsafe")
    dcrit_capacity = int(mercurius._N_allocated_dcrit)
    dcrit_present = bool(mercurius._dcrit)
    if ((dcrit_capacity == 0) != (not dcrit_present)
            or dcrit_capacity < 0 or dcrit_capacity > MAX_REBOUND_ALLOCATION_CAPACITY
            or (dcrit_present and dcrit_capacity < particle_count)):
        raise VerificationError("MERCURIUS dcrit allocation/state is unsafe")
    backup_capacity = int(mercurius._N_allocated)
    backup_present = bool(mercurius._particles_backup)
    encounter_map_present = bool(mercurius._encounter_map)
    additional_capacity = int(mercurius._N_allocated_additional_forces)
    additional_present = bool(mercurius._particles_backup_additional_forces)
    if ((backup_capacity == 0) != (not backup_present)
            or backup_present != encounter_map_present
            or backup_capacity < 0 or backup_capacity > MAX_REBOUND_ALLOCATION_CAPACITY
            or (backup_present and backup_capacity < particle_count)
            or (additional_capacity == 0) != (not additional_present)
            or additional_capacity < 0
            or additional_capacity > MAX_REBOUND_ALLOCATION_CAPACITY
            or (additional_present and additional_capacity < particle_count)
            or not 0 <= int(mercurius._encounter_N) <= particle_count
            or not 0 <= int(mercurius._encounter_N_active) <= int(simulation.N_active)
            or int(mercurius._encounter_N_active) > int(mercurius._encounter_N)
            or int(mercurius._tponly_encounter) not in {0, 1}):
        raise VerificationError("MERCURIUS transient cache topology is unsafe")
    ias_count = int(ias15._N_allocated)
    map_count = int(ias15._map_allocated_n)
    direct_pointers = [getattr(ias15, name) for name in (
        "_at", "_x0", "_v0", "_a0", "_csx", "_csv", "_csa0",
    )]
    coefficient_pointers = [
        getattr(getattr(ias15, group), f"p{index}")
        for group in ("_g", "_b", "_csb", "_e", "_br", "_er")
        for index in range(7)
    ]
    if ((source_mode == "ARCHIVE" and ias_count not in {0, 9})
            or (source_mode == "LIVE_BOUNDARY"
                and not 0 <= ias_count <= MAX_REBOUND_ALLOCATION_CAPACITY)
            or (ias_count == 0 and any(bool(pointer) for pointer in (
                *direct_pointers, *coefficient_pointers
            )))
            or (ias_count > 0 and any(not bool(pointer) for pointer in (
                *direct_pointers, *coefficient_pointers
            )))
            or ((map_count == 0) != (not bool(ias15._map)))
            or map_count < 0 or map_count > MAX_REBOUND_ALLOCATION_CAPACITY
            or (source_mode == "ARCHIVE" and (map_count != 0 or bool(ias15._map)))):
        raise VerificationError("IAS15 decoded array allocation/state is unsafe")
    whfast_capacity = int(whfast._N_allocated)
    whfast_tmp_capacity = int(whfast._N_allocated_tmp)
    whfast_present = bool(whfast._p_jh)
    whfast_tmp_present = bool(whfast._p_temp)
    if (((whfast_capacity == 0) != (not whfast_present))
            or ((whfast_tmp_capacity == 0) != (not whfast_tmp_present))
            or not 0 <= whfast_capacity <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not 0 <= whfast_tmp_capacity <= MAX_REBOUND_ALLOCATION_CAPACITY
            or (whfast_present and whfast_capacity < particle_count)
            or (whfast_tmp_present and whfast_tmp_capacity < particle_count)
            or (source_mode == "ARCHIVE" and (
                whfast_capacity != 0 or whfast_tmp_capacity != 0
                or whfast_present or whfast_tmp_present
            ))):
        raise VerificationError("WHFast decoded internal arrays are unexpectedly live")

    def active_interval(
        pointer: Any, count: int, item_size: int, alignment: int,
    ) -> tuple[int, int]:
        if count <= 0 or not bool(pointer) or item_size <= 0:
            raise VerificationError("decoded active-memory interval is unsafe")
        start = ctypes.cast(pointer, ctypes.c_void_p).value
        if type(start) is not int:
            raise VerificationError("decoded active-memory pointer is null")
        end = start + count * item_size
        max_address = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
        if (start <= 0 or end <= start or end - 1 > max_address
                or alignment <= 0 or start % alignment != 0):
            raise VerificationError("decoded active-memory interval overflowed")
        return start, end

    simulation_start = ctypes.addressof(simulation)
    simulation_size = ctypes.sizeof(simulation)
    if (simulation_start <= 0 or simulation_size <= 0
            or simulation_start % ctypes.alignment(type(simulation)) != 0):
        raise VerificationError("decoded Simulation memory range is unsafe")
    active_ranges = [
        (simulation_start, simulation_start + simulation_size),
        active_interval(
            simulation._particles, particle_capacity,
            ctypes.sizeof(simulation._particles._type_),
            ctypes.alignment(simulation._particles._type_),
        ),
    ]
    if dcrit_present:
        active_ranges.append(active_interval(
            mercurius._dcrit, dcrit_capacity, ctypes.sizeof(ctypes.c_double),
            ctypes.alignment(ctypes.c_double),
        ))
    if backup_present:
        active_ranges.extend((
            active_interval(
                mercurius._particles_backup, backup_capacity,
                ctypes.sizeof(mercurius._particles_backup._type_),
                ctypes.alignment(mercurius._particles_backup._type_),
            ),
            active_interval(
                mercurius._encounter_map, backup_capacity,
                ctypes.sizeof(mercurius._encounter_map._type_),
                ctypes.alignment(mercurius._encounter_map._type_),
            ),
        ))
    if additional_present:
        active_ranges.append(active_interval(
            mercurius._particles_backup_additional_forces, additional_capacity,
            ctypes.sizeof(mercurius._particles_backup_additional_forces._type_),
            ctypes.alignment(mercurius._particles_backup_additional_forces._type_),
        ))
    if ias_count:
        active_ranges.extend(
            active_interval(
                pointer, ias_count, ctypes.sizeof(ctypes.c_double),
                ctypes.alignment(ctypes.c_double),
            )
            for pointer in (*direct_pointers, *coefficient_pointers)
        )
    if map_count:
        active_ranges.append(active_interval(
            ias15._map, map_count, ctypes.sizeof(ctypes.c_int),
            ctypes.alignment(ctypes.c_int),
        ))
    if whfast_present:
        active_ranges.append(active_interval(
            whfast._p_jh, whfast_capacity,
            ctypes.sizeof(whfast._p_jh._type_), ctypes.alignment(whfast._p_jh._type_),
        ))
    if whfast_tmp_present:
        active_ranges.append(active_interval(
            whfast._p_temp, whfast_tmp_capacity,
            ctypes.sizeof(whfast._p_temp._type_),
            ctypes.alignment(whfast._p_temp._type_),
        ))
    active_ranges.sort()
    if any(left[1] > right[0]
           for left, right in zip(active_ranges, active_ranges[1:])):
        raise VerificationError("decoded active-memory allocations overlap")
    dcrit = [] if not dcrit_present else [
        h(mercurius._dcrit[index]) for index in range(particle_count)
    ]
    particles = []
    for index, particle in enumerate(simulation.particles):
        parent_bound = bool(particle._sim) and (
            ctypes.cast(particle._sim, ctypes.c_void_p).value == simulation_start
        )
        if not parent_bound:
            raise VerificationError(
                "decoded particle simulation reference is not parent-bound"
            )
        particles.append({
            "index": index, "hash": int(particle.hash.value),
            "simulation_reference_bound_to_parent": True,
            "m_hex": h(particle.m), "r_hex": h(particle.r),
            "x_hex": h(particle.x), "y_hex": h(particle.y), "z_hex": h(particle.z),
            "vx_hex": h(particle.vx), "vy_hex": h(particle.vy), "vz_hex": h(particle.vz),
            "ax_hex": h(particle.ax), "ay_hex": h(particle.ay), "az_hex": h(particle.az),
            "last_collision_hex": h(particle.last_collision),
            "collision_cell_present": particle.c is not None,
            "additional_properties_present": particle.ap is not None,
        })
    ias_direct = {
        name.removeprefix("_"): decoded_primary_double_array_sha256(
            getattr(ias15, name), ias_count
        ) if source_mode == "ARCHIVE" else None
        for name in ("_at", "_x0", "_v0", "_a0", "_csx", "_csv", "_csa0")
    }
    ias_coefficients = {
        group.removeprefix("_"): {
            f"p{index}": decoded_primary_double_array_sha256(
                getattr(getattr(ias15, group), f"p{index}"), ias_count
            ) if source_mode == "ARCHIVE" else None for index in range(7)
        } for group in ("_g", "_b", "_csb", "_e", "_br", "_er")
    }
    map_sha256 = None if source_mode == "LIVE_BOUNDARY" or not bool(ias15._map) else digest_bytes(
        CONTINUATION_ARRAY_DOMAIN + b"IAS15_MAP\0" + b"".join(
            struct.pack(">i", int(ias15._map[index])) for index in range(map_count)
        )
    )
    return {
        "schema": "jx-xp2-mercurius-decoded-continuation-state/v3",
        "simulation": {
            "t_hex": h(simulation.t), "G_hex": h(simulation.G),
            "softening_hex": h(simulation.softening), "dt_hex": h(simulation.dt),
            "dt_last_done_hex": h(simulation.dt_last_done),
            "steps_done": int(simulation.steps_done),
            "usleep_hex": h(simulation.usleep), "save_messages": int(simulation.save_messages),
            "status": int(simulation._status),
            "N": int(simulation.N),
            "particle_capacity_covers_logical_count": True,
            "particle_storage_present": bool(simulation._particles),
            "active_memory_ranges_pairwise_disjoint": True,
            "N_var": int(simulation.N_var),
            "N_var_config": int(simulation.N_var_config),
            "variation_config_present": bool(simulation.var_config),
            "var_rescale_warning": int(simulation._var_rescale_warning),
            "N_active": int(simulation.N_active),
            "testparticle_type": int(simulation.testparticle_type),
            "testparticle_hidewarnings": int(simulation.testparticle_hidewarnings),
            "hash_ctr": int(simulation.hash_ctr),
            "particle_lookup_count": int(simulation.N_lookup),
            "particle_lookup_allocation_count": int(simulation.N_allocated_lookup),
            "particle_lookup_present": bool(simulation._particle_lookup_table),
            "integrator": str(simulation.integrator), "gravity": str(simulation.gravity),
            "boundary": str(simulation.boundary), "collision": str(simulation.collision),
            "exact_finish_time": int(simulation.exact_finish_time),
            "force_is_velocity_dependent": int(simulation.force_is_velocity_dependent),
            "gravity_ignore": int(simulation.gravity_ignore),
            "exit_max_distance_hex": h(simulation.exit_max_distance),
            "exit_min_distance_hex": h(simulation.exit_min_distance),
            "track_energy_offset": int(simulation.track_energy_offset),
            "energy_offset_hex": h(simulation.energy_offset),
            "opening_angle2_hex": h(simulation.opening_angle2),
            "boxsize_hex": [h(value) for value in (box.x, box.y, box.z)],
            "boxsize_max_hex": h(simulation.boxsize_max),
            "root_size_hex": h(simulation.root_size), "N_root": int(simulation.N_root),
            "N_root_xyz": [int(simulation.N_root_x), int(simulation.N_root_y),
                           int(simulation.N_root_z)],
            "N_ghost_xyz": [int(simulation.N_ghost_x), int(simulation.N_ghost_y),
                            int(simulation.N_ghost_z)],
            "collision_resolve_keep_sorted": int(simulation.collision_resolve_keep_sorted),
            "collisions_N": int(simulation.collisions_N),
            "minimum_collision_velocity_hex": h(simulation.minimum_collision_velocity),
            "gravity_compensated_sums_present": bool(simulation.gravity_cs),
            "gravity_compensated_sums_allocation_count": int(
                simulation.N_allocated_gravity_cs
            ),
            "tree_root_present": bool(simulation._tree_root),
            "tree_needs_update": int(simulation._tree_needs_update),
            "messages_present": bool(simulation.messages),
            "display_view_present": bool(simulation._display_view),
            "display_data_present": bool(simulation._display_data),
            "server_data_present": bool(simulation._server_data),
            "collision_storage_present": bool(simulation.collisions),
            "collision_allocation_count": int(simulation.N_allocated_collisions),
            "collisions_plog_hex": h(simulation.collisions_plog),
            "collisions_log_n": int(simulation.collisions_log_n),
            "calculate_megno": int(simulation._calculate_megno),
            "megno_Ys_hex": h(simulation._megno_Ys),
            "megno_Yss_hex": h(simulation._megno_Yss),
            "megno_cov_Yt_hex": h(simulation._megno_cov_Yt),
            "megno_var_t_hex": h(simulation._megno_var_t),
            "megno_mean_t_hex": h(simulation._megno_mean_t),
            "megno_mean_Y_hex": h(simulation._megno_mean_Y),
            "megno_initial_t_hex": h(simulation._megno_initial_t),
            "megno_n": int(simulation._megno_n),
            "N_odes": int(simulation._N_odes),
            "odes_allocation_count": int(simulation._N_allocated_odes),
            "odes_warnings": int(simulation._odes_warnings),
            "odes_present": bool(simulation._odes),
            "extras_present": simulation.extras is not None,
            "simulationarchive_auto_interval_hex": h(
                simulation.simulationarchive_auto_interval
            ),
            "simulationarchive_auto_walltime_hex": h(
                simulation.simulationarchive_auto_walltime
            ),
            "simulationarchive_auto_step": int(simulation.simulationarchive_auto_step),
            "simulationarchive_next_hex": h(simulation.simulationarchive_next),
            "simulationarchive_next_step": int(simulation.simulationarchive_next_step),
            "simulationarchive_filename_present": (
                simulation.simulationarchive_filename is not None
            ),
            "callbacks_present": {
                "additional_forces": bool(simulation._additional_forces),
                "pre_timestep_modifications": bool(simulation._pre_timestep_modifications),
                "post_timestep_modifications": bool(simulation._post_timestep_modifications),
                "heartbeat": bool(simulation._heartbeat),
                "coefficient_of_restitution": bool(simulation._coefficient_of_restitution),
                "collision_resolve": bool(simulation._collision_resolve),
                "free_particle_ap": bool(simulation._free_particle_ap),
                "key_callback": bool(simulation._key_callback),
                "extras_cleanup": bool(simulation._extras_cleanup),
            },
        },
        "mercurius": {
            "r_crit_hill_hex": h(mercurius.r_crit_hill),
            "safe_mode": int(mercurius.safe_mode), "mode": int(mercurius.mode),
            "is_synchronized": int(mercurius.is_synchronized),
            "recalculate_coordinates_this_timestep": int(
                mercurius.recalculate_coordinates_this_timestep
            ),
            "recalculate_r_crit_this_timestep": int(
                mercurius.recalculate_r_crit_this_timestep
            ),
            "encounter_N": int(mercurius._encounter_N),
            "encounter_N_active": int(mercurius._encounter_N_active),
            "tponly_encounter": int(mercurius._tponly_encounter),
            "dcrit_storage_present": dcrit_present,
            "dcrit_capacity_covers_logical_count": (
                dcrit_present and dcrit_capacity >= particle_count
            ),
            "dcrit_hex": dcrit,
            "com_position_hex": [h(value) for value in (
                mercurius._com_pos.x, mercurius._com_pos.y, mercurius._com_pos.z
            )],
            "com_velocity_hex": [h(value) for value in (
                mercurius._com_vel.x, mercurius._com_vel.y, mercurius._com_vel.z
            )],
            "L_callback_present": bool(mercurius._L),
            "allocated_particle_backup_count": int(mercurius._N_allocated),
            "allocated_additional_forces_backup_count": int(
                mercurius._N_allocated_additional_forces
            ),
            "particles_backup_present": bool(mercurius._particles_backup),
            "additional_forces_backup_present": bool(
                mercurius._particles_backup_additional_forces
            ),
            "encounter_map_present": bool(mercurius._encounter_map),
        },
        "whfast": {
            "coordinates": str(whfast.coordinates), "kernel": str(whfast.kernel),
            "corrector": int(whfast.corrector), "corrector2": int(whfast.corrector2),
            "recalculate_coordinates_this_timestep": int(
                whfast.recalculate_coordinates_this_timestep
            ),
            "safe_mode": int(whfast.safe_mode),
            "keep_unsynchronized": int(whfast.keep_unsynchronized),
            "is_synchronized": int(whfast.is_synchronized),
            "timestep_warning": int(whfast._timestep_warning),
            "unsynchronized_recalculation_warning": int(
                whfast._recalculate_coordinates_but_not_synchronized_warning
            ),
            "internal_particle_arrays_present": bool(whfast._p_jh) or bool(whfast._p_temp),
        },
        "ias15": {
            "epsilon_hex": h(ias15.epsilon), "min_dt_hex": h(ias15.min_dt),
            "adaptive_mode": str(ias15.adaptive_mode),
            "iterations_max_exceeded": int(ias15._iterations_max_exceeded),
            "stored_coordinate_count": ias_count,
            "direct_array_sha256": ias_direct,
            "coefficient_array_sha256": ias_coefficients,
            "map_count": map_count, "map_sha256": map_sha256,
        },
        "particles": particles,
        "excluded_noncontinuation_fields": list(CONTINUATION_EXCLUDED_FIELDS),
    }


def primary_live_archive_endpoint_projection(simulation: Any) -> dict[str, Any]:
    """Independent normalized live/archive endpoint comparator."""
    projection = decoded_primary_continuation_projection(
        simulation, source_mode="LIVE_BOUNDARY",
    )
    normalized_mercurius = dict(projection["mercurius"])
    normalized_fields = (
        "encounter_N", "encounter_N_active", "tponly_encounter",
        "allocated_particle_backup_count",
        "allocated_additional_forces_backup_count", "particles_backup_present",
        "additional_forces_backup_present", "encounter_map_present",
    )
    for field in normalized_fields:
        normalized_mercurius.pop(field)
    normalized_whfast = dict(projection["whfast"])
    normalized_whfast.pop("internal_particle_arrays_present")
    normalized_ias15 = {
        field: projection["ias15"][field]
        for field in ("epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded")
    }
    return {
        "schema": "jx-xp2-mercurius-live-archive-endpoint/v1",
        "simulation": projection["simulation"],
        "mercurius": normalized_mercurius,
        "whfast": normalized_whfast,
        "ias15": normalized_ias15,
        "particles": projection["particles"],
        "save_load_normalized_mercurius_fields": list(normalized_fields),
        "save_load_normalized_whfast_fields": ["internal_particle_arrays_present"],
        "save_load_normalized_ias15_fields": [
            "stored_coordinate_count", "direct_array_sha256",
            "coefficient_array_sha256", "map_count", "map_sha256",
        ],
        "excluded_noncontinuation_fields": projection["excluded_noncontinuation_fields"],
    }


def primary_live_archive_endpoint_sha256(simulation: Any) -> str:
    return digest_bytes(
        PRIMARY_ENDPOINT_DOMAIN + canonical(primary_live_archive_endpoint_projection(simulation))
    )


def decoded_primary_double_array_sha256(pointer: Any, count: int) -> str | None:
    if count < 0:
        raise VerificationError("negative decoded array length")
    if not bool(pointer):
        return None
    digest = hashlib.sha256(CONTINUATION_ARRAY_DOMAIN)
    for index in range(count):
        value = float(pointer[index])
        if not math.isfinite(value):
            raise VerificationError(
                "decoded continuation array contains a non-finite value"
            )
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def decoded_primary_state_sha256(simulation: Any) -> str:
    return digest_bytes(
        PRIMARY_STATE_DOMAIN + canonical(decoded_primary_continuation_projection(simulation))
    )


def valid_primary_ias15_continuation(value: dict[str, Any]) -> bool:
    if set(value) != {
        "epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded",
        "stored_coordinate_count", "direct_array_sha256",
        "coefficient_array_sha256", "map_count", "map_sha256",
    }:
        return False
    count = value["stored_coordinate_count"]
    direct = value["direct_array_sha256"]
    coefficients = value["coefficient_array_sha256"]
    if (type(count) is not int or count not in {0, 9}
            or set(direct) != {"at", "x0", "v0", "a0", "csx", "csv", "csa0"}
            or set(coefficients) != {"g", "b", "csb", "e", "br", "er"}
            or any(set(rows) != {f"p{index}" for index in range(7)}
                   for rows in coefficients.values())):
        return False
    digests = list(direct.values()) + [
        digest for rows in coefficients.values() for digest in rows.values()
    ]
    if (count == 0 and any(digest is not None for digest in digests)) \
            or (count > 0 and any(not lower_sha256(digest) for digest in digests)):
        return False
    map_count = value["map_count"]
    return (type(map_count) is int and map_count == 0
            and value["map_sha256"] is None)


def verify_primary_checkpoint_endpoint(
    state_path: Path, receipt: dict[str, Any], contract: dict[str, Any],
    expanded_rows: list[list[Any]],
) -> None:
    if not 0 < state_path.stat().st_size <= MAX_PRIMARY_CHECKPOINT_BYTES:
        raise VerificationError("primary checkpoint exceeds registered byte bound")
    rebound = verifier_rebound(contract)
    simulation = rebound.Simulation(str(state_path))
    active_count = sum(row[1] == "A" for row in expanded_rows)
    projection = decoded_primary_continuation_projection(simulation)
    if (type(projection) is not dict
            or list(projection) != [
                "schema", "simulation", "mercurius", "whfast", "ias15",
                "particles", "excluded_noncontinuation_fields",
            ]
            or projection.get("schema")
            != "jx-xp2-mercurius-decoded-continuation-state/v3"
            or list(projection.get("simulation", {}))
            != list(CONTINUATION_SIMULATION_FIELDS)
            or list(projection.get("simulation", {}).get("callbacks_present", {}))
            != list(CONTINUATION_CALLBACK_FIELDS)
            or list(projection.get("mercurius", {}))
            != list(CONTINUATION_MERCURIUS_FIELDS)
            or list(projection.get("whfast", {})) != list(CONTINUATION_WHFAST_FIELDS)
            or list(projection.get("ias15", {})) != list(CONTINUATION_IAS15_FIELDS)
            or any(list(row) != list(CONTINUATION_PARTICLE_FIELDS)
                   for row in projection.get("particles", []))
            or projection.get("excluded_noncontinuation_fields")
            != list(CONTINUATION_EXCLUDED_FIELDS)):
        raise VerificationError("decoded primary continuation projection schema changed")
    settings = projection["simulation"]
    mercurius = projection["mercurius"]
    whfast = projection["whfast"]
    ias15 = projection["ias15"]
    h = lambda value: struct.pack(">d", float(value)).hex()
    if (decoded_primary_state_sha256(simulation) != receipt["decoded_integrator_state_sha256"]
            or float(simulation.t) != receipt["end_years"]
            or float(simulation.G) != contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
            or simulation.N != len(expanded_rows) or simulation.N_active != active_count
            or simulation.integrator != "mercurius" or simulation.dt != receipt["dt_years"]
            or simulation.testparticle_type != 0 or simulation.ri_mercurius.r_crit_hill != 3.0
            or int(simulation.ri_mercurius.safe_mode) != 1 or str(simulation.collision) != "none"):
        raise VerificationError("decoded primary checkpoint settings/state digest changed")
    if (
        settings["softening_hex"] != h(0.0)
        or settings["dt_last_done_hex"] != h(receipt["dt_years"])
        or settings["steps_done"] != int(round(receipt["end_years"] / receipt["dt_years"]))
        or settings["usleep_hex"] != h(0.0) or settings["save_messages"] != 1
        or settings["status"] != 0
        or settings["particle_capacity_covers_logical_count"] is not True
        or settings["particle_storage_present"] is not True
        or settings["active_memory_ranges_pairwise_disjoint"] is not True
        or settings["N_var"] != 0 or settings["N_var_config"] != 0
        or settings["variation_config_present"] is not False
        or settings["var_rescale_warning"] != 0
        or settings["testparticle_hidewarnings"] != 0
        or settings["hash_ctr"] != 0
        or settings["particle_lookup_count"] != 0
        or settings["particle_lookup_allocation_count"] != 0
        or settings["particle_lookup_present"] is not False
        or settings["gravity"] != "mercurius"
        or settings["boundary"] != "none" or settings["collision"] != "none"
        or settings["exact_finish_time"] != 1
        or settings["force_is_velocity_dependent"] != 0
        or settings["gravity_ignore"] != 0
        or settings["exit_max_distance_hex"] != h(0.0)
        or settings["exit_min_distance_hex"] != h(0.0)
        or settings["track_energy_offset"] != 0
        or settings["energy_offset_hex"] != h(0.0)
        or settings["opening_angle2_hex"] != h(0.25)
        or settings["boxsize_hex"] != [h(0.0)] * 3
        or settings["boxsize_max_hex"] != h(0.0)
        or settings["root_size_hex"] != h(-1.0) or settings["N_root"] != 1
        or settings["N_root_xyz"] != [1, 1, 1]
        or settings["N_ghost_xyz"] != [0, 0, 0]
        or settings["collision_resolve_keep_sorted"] != 0
        or settings["collisions_N"] != 0
        or settings["minimum_collision_velocity_hex"] != h(0.0)
        or settings["gravity_compensated_sums_present"] is not False
        or settings["gravity_compensated_sums_allocation_count"] != 0
        or settings["tree_root_present"] is not False
        or settings["tree_needs_update"] != 0
        or settings["messages_present"] is not False
        or settings["display_view_present"] is not False
        or settings["display_data_present"] is not False
        or settings["server_data_present"] is not False
        or settings["collision_storage_present"] is not False
        or settings["collision_allocation_count"] != 0
        or settings["collisions_plog_hex"] != h(0.0)
        or settings["collisions_log_n"] != 0
        or settings["calculate_megno"] != 0
        or any(settings[field] != h(0.0) for field in (
            "megno_Ys_hex", "megno_Yss_hex", "megno_cov_Yt_hex", "megno_var_t_hex",
            "megno_mean_t_hex", "megno_mean_Y_hex", "megno_initial_t_hex",
        ))
        or settings["megno_n"] != 0
        or settings["N_odes"] != 0 or settings["odes_allocation_count"] != 0
        or settings["odes_warnings"] != 0 or settings["odes_present"] is not False
        or settings["extras_present"] is not False
        or settings["simulationarchive_auto_interval_hex"] != h(0.0)
        or settings["simulationarchive_auto_walltime_hex"] != h(0.0)
        or settings["simulationarchive_auto_step"] != 0
        or settings["simulationarchive_next_hex"] != h(0.0)
        or settings["simulationarchive_next_step"] != 0
        or settings["simulationarchive_filename_present"] is not False
        or any(settings["callbacks_present"].values())
        or mercurius["r_crit_hill_hex"] != h(3.0)
        or mercurius["safe_mode"] != 1 or mercurius["mode"] != 0
        or mercurius["is_synchronized"] != 1
        or mercurius["recalculate_coordinates_this_timestep"] != 1
        or mercurius["recalculate_r_crit_this_timestep"] != 0
        or mercurius["encounter_N"] != 0 or mercurius["encounter_N_active"] != 0
        or mercurius["tponly_encounter"] != 0
        or mercurius["dcrit_storage_present"] is not True
        or mercurius["dcrit_capacity_covers_logical_count"] is not True
        or len(mercurius["dcrit_hex"]) != len(expanded_rows)
        or mercurius["L_callback_present"] is not False
        or mercurius["allocated_particle_backup_count"] != 0
        or mercurius["allocated_additional_forces_backup_count"] != 0
        or mercurius["particles_backup_present"] is not False
        or mercurius["additional_forces_backup_present"] is not False
        or mercurius["encounter_map_present"] is not False
        or whfast != {
            "coordinates": "jacobi", "kernel": "default", "corrector": 0,
            "corrector2": 0, "recalculate_coordinates_this_timestep": 0,
            "safe_mode": 1, "keep_unsynchronized": 0, "is_synchronized": 1,
            "timestep_warning": 0, "unsynchronized_recalculation_warning": 0,
            "internal_particle_arrays_present": False,
        }
        or ias15["epsilon_hex"] != h(1e-9) or ias15["min_dt_hex"] != h(0.0)
        or ias15["adaptive_mode"] != "prs23"
        or ias15["iterations_max_exceeded"] != 0
        or not valid_primary_ias15_continuation(ias15)
        or any(row["simulation_reference_bound_to_parent"] is not True
               or row["last_collision_hex"] != h(0.0)
               or row["collision_cell_present"]
               or row["additional_properties_present"]
               for row in projection["particles"])
    ):
        raise VerificationError("decoded primary continuation settings changed")
    for index, expected in enumerate(expanded_rows):
        particle = simulation.particles[index]
        if (int(particle.hash.value) != int(rebound.hash(expected[0]).value)
                or float(particle.m) != f64(expected[2]) or float(particle.r) != 0.0):
            raise VerificationError("decoded primary particle identity/order/mass changed")
    active_rows = []
    for index, expected in enumerate(expanded_rows[:active_count]):
        particle = simulation.particles[index]
        active_rows.append([expected[0], "A", expected[2], pack6([
            float(particle.x), float(particle.y), float(particle.z),
            float(particle.vx), float(particle.vy), float(particle.vz),
        ])])
    current = primary_active_invariants(
        active_rows, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
    )
    baseline = receipt["initial_active_invariants"]
    norm = lambda values: math.sqrt(math.fsum(value * value for value in values))
    endpoint_drifts = {
        "relative_compensated_active_energy_drift": abs(
            current["intrinsic_energy"] - baseline["intrinsic_energy"]
        ) / abs(baseline["intrinsic_energy"]),
        "relative_active_com_angular_momentum_vector_drift": norm([
            current["com_angular"][i] - baseline["com_angular"][i] for i in range(3)
        ]) / norm(baseline["com_angular"]),
        "scale_normalized_active_linear_momentum_residual": norm([
            current["momentum"][i] - baseline["momentum"][i] for i in range(3)
        ]) / baseline["linear_internal_scale"],
    }
    if any(receipt["maximum_active_invariant_drifts"][key] < value
           for key, value in endpoint_drifts.items()):
        raise VerificationError("primary stored maximum drift is below decoded endpoint")
    sun = simulation.particles[0]
    landmark_key = str(int(receipt["end_years"]))
    landmark_rows = None if landmark_key not in receipt["landmarks"] else {
        row["logical_id"]: row for row in receipt["landmarks"][landmark_key]["particles"]
    }
    for offset, tracker in enumerate(receipt["tracker"]):
        particle = simulation.particles[active_count + offset]
        orbit = particle.orbit(primary=sun)
        a = float(orbit.a); e = float(orbit.e); inclination = math.degrees(float(orbit.inc))
        q = a * (1.0 - e)
        distance = math.sqrt((float(particle.x) - float(sun.x)) ** 2
                             + (float(particle.y) - float(sun.y)) ** 2
                             + (float(particle.z) - float(sun.z)) ** 2)
        if (not all(math.isfinite(value) for value in (a, e, inclination, q, distance))
                or e < 0.0 or q < 0.0 or distance <= 0.0
                or tracker["minimum_sampled_q_AU"] > q):
            raise VerificationError("decoded primary endpoint contradicts compact tracker")
        if landmark_rows is not None:
            stored = landmark_rows[tracker["logical_id"]]
            expected = {
                "final_a_AU": a, "final_e": e, "final_i_deg": inclination,
                "final_q_AU": q, "final_distance_AU": distance,
                "final_finite_and_bound": bool(a > 0.0 and e < 1.0),
            }
            if any(stored[key] != value for key, value in expected.items()):
                raise VerificationError("primary landmark differs from decoded endpoint")
    del particle, sun, simulation
    gc.collect()


def validate_primary_drifts(value: Any, label: str) -> None:
    exact_keys(value, {
        "relative_compensated_active_energy_drift",
        "relative_active_com_angular_momentum_vector_drift",
        "scale_normalized_active_linear_momentum_residual",
    }, label)
    if any(not finite_number(number) or number < 0.0 for number in value.values()):
        raise VerificationError(f"{label} contains an invalid value")


def validate_primary_crossings(row: dict[str, Any], minimum_q: float, horizon: float) -> None:
    values = []
    for threshold in (40, 35, 30):
        value = row[f"first_sampled_q_below_{threshold}_time_year"]
        if (value is not None) != (minimum_q < threshold):
            raise VerificationError("primary first-passage presence contradicts prefix minimum")
        if value is not None and (not finite_number(value) or value < 0.0
                                  or value > horizon or value % 50.0 != 0.0):
            raise VerificationError("primary first passage is off the exact grid")
        values.append(value)
    if ((values[1] is not None and values[0] is None)
            or (values[2] is not None and values[1] is None)):
        raise VerificationError("primary first-passage presence is not nested")
    finite = [value for value in values if value is not None]
    if finite != sorted(finite):
        raise VerificationError("primary first-passage times are not nested")


def validate_primary_tracker(tracker: Any, horizon: float) -> list[str]:
    if not isinstance(tracker, list) or len(tracker) != 128:
        raise VerificationError("primary tracker cardinality changed")
    ids = [f"XP2-B{block:02d}-T{index:02d}" for block in range(8) for index in range(16)]
    fields = {
        "logical_id", "block_index", "index_within_block", "minimum_sampled_q_AU",
        "first_sampled_q_below_30_time_year", "first_sampled_q_below_35_time_year",
        "first_sampled_q_below_40_time_year",
        "all_samples_finite_cartesian_and_osculating",
    }
    for logical_id, row in zip(ids, tracker, strict=True):
        exact_keys(row, fields, "primary tracker row")
        if (row["logical_id"] != logical_id or row["block_index"] != int(logical_id[5:7])
                or row["index_within_block"] != int(logical_id[9:11])
                or row["all_samples_finite_cartesian_and_osculating"] is not True
                or not finite_number(row["minimum_sampled_q_AU"])
                or row["minimum_sampled_q_AU"] < 0.0):
            raise VerificationError("primary tracker identity/value changed")
        validate_primary_crossings(row, row["minimum_sampled_q_AU"], horizon)
    return ids


def validate_primary_landmark(value: Any, horizon: float, ids: Sequence[str]) -> None:
    exact_keys(value, {
        "horizon_years", "particles", "summary", "maximum_active_invariant_drifts"
    }, "primary landmark")
    if (value["horizon_years"] != horizon or not isinstance(value["particles"], list)
            or len(value["particles"]) != 128):
        raise VerificationError("primary landmark identity/cardinality changed")
    fields = {
        "logical_id", "block_index", "index_within_block", "minimum_sampled_q_AU",
        "first_sampled_q_below_30_time_year", "first_sampled_q_below_35_time_year",
        "first_sampled_q_below_40_time_year",
        "all_samples_finite_cartesian_and_osculating", "final_a_AU", "final_e",
        "final_i_deg", "final_q_AU", "final_distance_AU", "final_finite_and_bound",
    }
    for logical_id, row in zip(ids, value["particles"], strict=True):
        exact_keys(row, fields, "primary landmark particle")
        if (row["logical_id"] != logical_id or row["block_index"] != int(logical_id[5:7])
                or row["index_within_block"] != int(logical_id[9:11])
                or row["all_samples_finite_cartesian_and_osculating"] is not True
                or type(row["final_finite_and_bound"]) is not bool):
            raise VerificationError("primary landmark identity/boolean changed")
        for field in ("minimum_sampled_q_AU", "final_a_AU", "final_e", "final_i_deg",
                      "final_q_AU", "final_distance_AU"):
            if not finite_number(row[field]):
                raise VerificationError("primary landmark metric is non-finite")
        if (row["minimum_sampled_q_AU"] < 0.0 or row["final_e"] < 0.0
                or not 0.0 <= row["final_i_deg"] <= 180.0 or row["final_q_AU"] < 0.0
                or row["final_distance_AU"] <= 0.0
                or row["final_finite_and_bound"]
                != (row["final_a_AU"] > 0.0 and row["final_e"] < 1.0)):
            raise VerificationError("primary landmark metric domain changed")
        validate_primary_crossings(row, row["minimum_sampled_q_AU"], horizon)
    if value["summary"] != recompute_landmark(value):
        raise VerificationError("primary landmark summary changed")
    validate_primary_drifts(value["maximum_active_invariant_drifts"], "primary landmark drifts")


def segment_semantic(receipt: dict[str, Any]) -> dict[str, Any]:
    return primary_segment_semantic_payload(receipt)


def primary_arm_identity(arm_id: str, contract: dict[str, Any]) -> tuple[str, str, float]:
    if arm_id not in ALL_ARMS:
        raise VerificationError("unknown primary arm")
    configuration = AUDIT_BASE.get(arm_id, arm_id)
    dynamics = contract["design_core"]["dynamics"]
    if arm_id in AUDIT_BASE:
        return configuration, "HALF_TIMESTEP", dynamics["audit_dt_years"]
    return configuration, "PRIMARY_TIMESTEP", dynamics["primary_dt_years"]


def verify_segment_tree(
    output_root: Path, arm_id: str, contract: dict[str, Any],
    expanded: dict[str, list[list[Any]]],
) -> dict[str, Any]:
    arm_dir = output_root / "arms" / arm_id
    segment_dir = arm_dir / "segments"
    safe_dir(arm_dir, "primary arm directory"); safe_dir(segment_dir, "primary segment directory")
    if arm_dir.resolve().parent != (output_root / "arms").resolve() \
            or segment_dir.resolve().parent != arm_dir.resolve():
        raise VerificationError("primary segment ancestry escaped")
    previous = SEGMENT_GENESIS
    final: dict[str, Any] | None = None
    previous_receipt: dict[str, Any] | None = None
    expected_names: set[str] = set()
    configuration, expected_arm_class, expected_dt = primary_arm_identity(arm_id, contract)
    expected_initial = primary_active_invariants(
        expanded[configuration], contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
    )
    for segment in range(20):
        commit_path = segment_dir / f"segment_{segment:02d}_commit.json"
        commit = read_object(commit_path)
        exact_keys(commit, {
            "schema", "experiment_id", "arm_id", "segment_index",
            "attempt_receipt_filename", "attempt_receipt_sha256", "checkpoint_filename",
            "checkpoint_sha256", "raw_checkpoint_integrity_only",
            "decoded_integrator_state_sha256", "segment_chain_head",
            "parent_terminal_validation",
        }, "primary parent commit")
        if (commit["schema"] != COMMIT_SCHEMA or commit["experiment_id"] != EXPERIMENT_ID
                or commit["arm_id"] != arm_id or commit["segment_index"] != segment
                or commit["parent_terminal_validation"]
                != "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS"):
            raise VerificationError("parent segment commit identity changed")
        receipt_name = commit["attempt_receipt_filename"]
        if (not isinstance(receipt_name, str) or Path(receipt_name).name != receipt_name
                or not receipt_name.startswith(f"segment_{segment:02d}_attempt_")
                or not receipt_name.endswith("_receipt.json")):
            raise VerificationError("primary receipt basename changed")
        receipt_path = segment_dir / receipt_name
        if receipt_path.resolve().parent != segment_dir.resolve():
            raise VerificationError("primary receipt path escaped")
        receipt = read_object(receipt_path)
        semantic_fields = set(segment_semantic(receipt))
        exact_keys(receipt, semantic_fields | {
            "schema", "experiment_id", "previous_segment_chain_head", "segment_chain_head",
            "checkpoint_filename", "checkpoint_sha256", "checkpoint_size_bytes",
            "raw_checkpoint_integrity_only", "provenance",
        }, "primary segment receipt")
        state_name = receipt["checkpoint_filename"]
        if (not isinstance(state_name, str) or Path(state_name).name != state_name
                or not state_name.startswith(f"segment_{segment:02d}_attempt_")
                or not state_name.endswith("_state.bin") or commit["checkpoint_filename"] != state_name):
            raise VerificationError("primary checkpoint basename changed")
        state_path = segment_dir / state_name
        safe_regular(state_path, "primary checkpoint")
        if state_path.resolve().parent != segment_dir.resolve():
            raise VerificationError("primary checkpoint path escaped")
        if (receipt["schema"] != RECEIPT_SCHEMA or receipt["experiment_id"] != EXPERIMENT_ID
                or receipt["arm_id"] != arm_id or receipt["segment_index"] != segment
                or receipt["configuration_id"] != configuration
                or receipt["arm_class"] != expected_arm_class
                or receipt["dt_years"] != expected_dt):
            raise VerificationError("segment attempt receipt identity changed")
        if (digest_file(receipt_path) != commit["attempt_receipt_sha256"]
                or digest_file(state_path) != receipt["checkpoint_sha256"]
                or state_path.stat().st_size != receipt["checkpoint_size_bytes"]
                or commit["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or receipt["raw_checkpoint_integrity_only"] is not True
                or commit["raw_checkpoint_integrity_only"] is not True
                or commit["decoded_integrator_state_sha256"]
                != receipt["decoded_integrator_state_sha256"]):
            raise VerificationError("segment stored hash/size mismatch")
        semantic = segment_semantic(receipt)
        chain = digest_bytes(SEGMENT_DOMAIN + bytes.fromhex(previous) + canonical(semantic))
        if (receipt["previous_segment_chain_head"] != previous
                or receipt["segment_chain_head"] != chain
                or commit["segment_chain_head"] != chain):
            raise VerificationError("primary segment hash chain mismatch")
        first = segment * 1000 + (0 if segment == 0 else 1)
        last = (segment + 1) * 1000
        if (receipt["first_sample_index"] != first or receipt["last_sample_index"] != last
                or receipt["new_sample_count"] != last - first + 1
                or receipt["sample_count_total"] != last + 1
                or receipt["start_years"] != segment * 50_000.0
                or receipt["end_years"] != (segment + 1) * 50_000.0):
            raise VerificationError("segment sampling ownership changed")
        ids = validate_primary_tracker(receipt["tracker"], receipt["end_years"])
        exact_keys(receipt["initial_active_invariants"], {
            "momentum", "r_com", "v_com", "com_angular", "linear_internal_scale",
            "intrinsic_energy",
        }, "primary initial invariants")
        if receipt["initial_active_invariants"] != expected_initial:
            raise VerificationError("primary invariant baseline differs from registered rows")
        validate_primary_drifts(receipt["maximum_active_invariant_drifts"], "primary maximum drifts")
        expected_landmarks = {str(int(year)) for year in HORIZONS if year <= receipt["end_years"]}
        if not isinstance(receipt["landmarks"], dict) or set(receipt["landmarks"]) != expected_landmarks:
            raise VerificationError("primary landmark ownership changed")
        for key, landmark in receipt["landmarks"].items():
            validate_primary_landmark(landmark, float(key), ids)
        if previous_receipt is not None:
            if receipt["initial_active_invariants"] != previous_receipt["initial_active_invariants"]:
                raise VerificationError("primary invariant baseline changed across segments")
            if any(receipt["maximum_active_invariant_drifts"][key]
                   < previous_receipt["maximum_active_invariant_drifts"][key]
                   for key in receipt["maximum_active_invariant_drifts"]):
                raise VerificationError("primary maximum drift decreased")
            for key, landmark in previous_receipt["landmarks"].items():
                if receipt["landmarks"].get(key) != landmark:
                    raise VerificationError("primary historical landmark changed")
            for old, new in zip(previous_receipt["tracker"], receipt["tracker"], strict=True):
                if new["logical_id"] != old["logical_id"] or new["minimum_sampled_q_AU"] > old[
                    "minimum_sampled_q_AU"
                ]:
                    raise VerificationError("primary tracker prefix regressed")
                for threshold in (30, 35, 40):
                    field = f"first_sampled_q_below_{threshold}_time_year"
                    if old[field] is not None and new[field] != old[field]:
                        raise VerificationError("primary first passage was rewritten")
                    if old[field] is None and new[field] is not None and not (
                        receipt["start_years"] < new[field] <= receipt["end_years"]
                    ):
                        raise VerificationError("primary new first passage is backdated")
        new_landmark = str(int(receipt["end_years"]))
        if new_landmark in receipt["landmarks"]:
            landmark = receipt["landmarks"][new_landmark]
            if landmark["maximum_active_invariant_drifts"] != receipt[
                "maximum_active_invariant_drifts"
            ]:
                raise VerificationError("primary terminal landmark drift differs")
            prefix_fields = set(receipt["tracker"][0])
            for tracker, particle in zip(receipt["tracker"], landmark["particles"], strict=True):
                if any(tracker[field] != particle[field] for field in prefix_fields):
                    raise VerificationError("primary terminal landmark tracker differs")
        provenance = receipt["provenance"]
        exact_keys(provenance, {"attempt_index", "wall_seconds", "peak_rss_bytes"},
                   "primary attempt provenance")
        if (not isinstance(provenance["attempt_index"], int) or isinstance(
                provenance["attempt_index"], bool) or not 1 <= provenance["attempt_index"] <= 3
                or not finite_number(provenance["wall_seconds"]) or provenance["wall_seconds"] < 0.0
                or provenance["wall_seconds"]
                >= contract["resource_caps_per_execution"]["max_wall_seconds_per_segment_attempt"]
                or not isinstance(provenance["peak_rss_bytes"], int)
                or isinstance(provenance["peak_rss_bytes"], bool)
                or not 0 <= provenance["peak_rss_bytes"]
                <= contract["resource_caps_per_execution"]["max_peak_rss_bytes_per_process"]
                or any(not lower_sha256(receipt[key]) for key in (
                    "sampled_state_stream_sha256", "decoded_integrator_state_sha256",
                    "checkpoint_sha256", "previous_segment_chain_head", "segment_chain_head",
                ))):
            raise VerificationError("primary attempt provenance/digest changed")
        attempt = provenance["attempt_index"]
        if (receipt_path.name != f"segment_{segment:02d}_attempt_{attempt:02d}_receipt.json"
                or state_path.name != f"segment_{segment:02d}_attempt_{attempt:02d}_state.bin"):
            raise VerificationError("primary selected attempt filenames disagree with provenance")
        verify_primary_checkpoint_endpoint(
            state_path, receipt, contract, expanded[configuration]
        )
        expected_names.update({commit_path.name, receipt_path.name, state_path.name})
        previous = chain; final = receipt
        previous_receipt = receipt
    if final is None or final["sample_count_total"] != 20_001:
        raise VerificationError("arm is incomplete")
    entries = list(segment_dir.iterdir())
    if (any(entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1
            for entry in entries) or {entry.name for entry in entries} != expected_names):
        raise VerificationError("primary segment tree contains an extra/uncommitted artifact")
    return final


def indicator(row: dict[str, Any], threshold: float) -> int:
    return int(row["minimum_sampled_q_AU"] < threshold)


def index_particles(landmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = landmark["particles"]
    result = {row["logical_id"]: row for row in rows}
    if len(rows) != 128 or len(result) != 128:
        raise VerificationError("primary landmark particle cardinality changed")
    return result


def recompute_landmark(landmark: dict[str, Any]) -> dict[str, Any]:
    rows = landmark["particles"]
    horizon = landmark["horizon_years"]
    summary = {
        "particle_count": 128,
        "all_particles_all_samples_finite_cartesian_and_osculating": all(
            row["all_samples_finite_cartesian_and_osculating"] for row in rows
        ),
        "final_finite_bound_count": sum(row["final_finite_and_bound"] for row in rows),
    }
    for threshold in (30, 35, 40):
        hits = sum(indicator(row, float(threshold)) for row in rows)
        summary[f"q_below_{threshold}_hit_count"] = hits
        summary[f"q_below_{threshold}_fraction"] = hits / 128.0
        key = f"first_sampled_q_below_{threshold}_time_year"
        summary[f"restricted_mean_censored_first_q{threshold}_years"] = math.fsum(
            horizon if row[key] is None else row[key] for row in rows
        ) / 128.0
    return summary


def w1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise VerificationError("invalid W1 sample shapes")
    a = sorted(map(float, left)); b = sorted(map(float, right))
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise VerificationError("non-finite W1 input")
    return math.fsum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def effects(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    maps = {arm_id: index_particles(landmark) for arm_id, landmark in arms.items()}
    ids = sorted(maps["M0"])
    mixture = 0
    blocks = {index: 0 for index in range(8)}
    cases = {case: 0 for case in ("CI01", "CI05", "CI09")}
    probes = {f"P{index}": 0 for index in range(8)}
    for logical_id in ids:
        control = indicator(maps["M0"][logical_id], 35.0)
        block = maps["M0"][logical_id]["block_index"]
        if block not in blocks or any(maps[arm][logical_id]["block_index"] != block
                                      for arm in PRIMARY_ARMS):
            raise VerificationError("matched block grouping changed")
        delta = sum(indicator(maps[arm][logical_id], 35.0) for arm in PRIMARY_ARMS[1:]) - 24 * control
        mixture += delta; blocks[block] += delta
        for case in cases:
            cases[case] += sum(indicator(maps[f"{case}-P{probe}"][logical_id], 35.0)
                               for probe in range(8)) - 8 * control
        for probe in range(8):
            key = f"P{probe}"
            probes[key] += sum(indicator(maps[f"{case}-{key}"][logical_id], 35.0)
                               for case in cases) - 3 * control
    return {
        "threshold_AU": 35.0, "mixture_numerator": mixture, "mixture_denominator": 3072,
        "block_numerators": {str(key): value for key, value in blocks.items()},
        "block_denominator_each": 384, "physical_case_numerators": cases,
        "physical_case_denominator_each": 1024, "orientation_numerators": probes,
        "orientation_denominator_each": 384,
    }


def raw_label(row: dict[str, Any]) -> str:
    n = row["mixture_numerator"]
    blocks = list(row["block_numerators"].values())
    cases = list(row["physical_case_numerators"].values())
    probes = list(row["orientation_numerators"].values())
    if (n >= 154 and all(value > 0 for value in blocks) and all(value > 0 for value in cases)
            and sum(value > 0 for value in probes) >= 6 and not any(value < 0 for value in probes)):
        return "DESIGN_GRID_DIRECTIONALLY_ROBUST_INCREASE"
    if (n <= -154 and all(value < 0 for value in blocks) and all(value < 0 for value in cases)
            and sum(value < 0 for value in probes) >= 6 and not any(value > 0 for value in probes)):
        return "DESIGN_GRID_DIRECTIONALLY_ROBUST_DECREASE"
    if (abs(n) <= 61 and max(map(abs, blocks)) <= 19 and max(map(abs, cases)) <= 51
            and max(map(abs, probes)) <= 19):
        return "DESIGN_GRID_Q35_PRACTICALLY_SMALL_WITH_EVENT_SUPPORT"
    return "INCONCLUSIVE"


def support(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    maps = {arm: index_particles(value) for arm, value in arms.items()}
    ids = sorted(maps["M0"])
    control = {logical_id for logical_id in ids if indicator(maps["M0"][logical_id], 35.0)}
    any_m1 = {logical_id for logical_id in ids if any(
        indicator(maps[arm][logical_id], 35.0) for arm in PRIMARY_ARMS[1:]
    )}
    discordant = {logical_id for logical_id in ids if any(
        indicator(maps[arm][logical_id], 35.0) != indicator(maps["M0"][logical_id], 35.0)
        for arm in PRIMARY_ARMS[1:]
    )}
    return {
        "unique_M0_q35_hitter_count": len(control),
        "unique_any_M1_q35_hitter_count": len(any_m1),
        "unique_union_q35_hitter_count": len(control | any_m1),
        "unique_discordant_q35_tracer_count": len(discordant),
        "M0_logical_ids_sha256": digest_bytes(canonical(sorted(control))),
        "any_M1_logical_ids_sha256": digest_bytes(canonical(sorted(any_m1))),
        "union_logical_ids_sha256": digest_bytes(canonical(sorted(control | any_m1))),
        "discordant_logical_ids_sha256": digest_bytes(canonical(sorted(discordant))),
    }


def promote(label: str, row: dict[str, Any]) -> str:
    if label == "DESIGN_GRID_Q35_PRACTICALLY_SMALL_WITH_EVENT_SUPPORT":
        return label if (row["unique_M0_q35_hitter_count"] >= 16
                         and row["unique_any_M1_q35_hitter_count"] >= 16) else "ENDPOINT_FLOOR_LIMITED"
    if label in ("DESIGN_GRID_DIRECTIONALLY_ROBUST_INCREASE",
                 "DESIGN_GRID_DIRECTIONALLY_ROBUST_DECREASE"):
        return label if (row["unique_union_q35_hitter_count"] >= 24
                         and row["unique_discordant_q35_tracer_count"] >= 16) else "ENDPOINT_FLOOR_LIMITED"
    return label


def censored(rows: Sequence[dict[str, Any]], threshold: int, horizon: float) -> list[float]:
    key = f"first_sampled_q_below_{threshold}_time_year"
    return [(horizon if row[key] is None else row[key]) / horizon for row in rows]


def pair_gate(primary: dict[str, Any], audit: dict[str, Any], horizon: float,
              contract: dict[str, Any], configuration: str) -> dict[str, Any]:
    left = index_particles(primary); right = index_particles(audit)
    ids = sorted(left)
    if set(left) != set(right):
        raise VerificationError("paired tracer identities changed")
    limits = contract["numerical_gates"]["timestep_pairs"]
    metrics: dict[str, Any] = {"configuration_id": configuration, "horizon_years": horizon}
    checks: dict[str, bool] = {}
    for threshold in (30, 35, 40):
        li = [indicator(left[logical_id], threshold) for logical_id in ids]
        ri = [indicator(right[logical_id], threshold) for logical_id in ids]
        count = abs(sum(li) - sum(ri)); discordance = sum(a != b for a, b in zip(li, ri, strict=True))
        metrics[f"q{threshold}_hit_count_absolute_difference"] = count
        metrics[f"q{threshold}_paired_indicator_discordance"] = discordance
        checks[f"q{threshold}_count_within_gate"] = count <= limits[f"max_q{threshold}_count_difference"]
        checks[f"q{threshold}_discordance_within_gate"] = discordance <= limits[
            f"max_q{threshold}_indicator_discordance"
        ]
    lrows = [left[key] for key in ids]; rrows = [right[key] for key in ids]
    bound = abs(sum(row["final_finite_and_bound"] for row in lrows)
                - sum(row["final_finite_and_bound"] for row in rrows))
    min_q = w1([row["minimum_sampled_q_AU"] for row in lrows],
               [row["minimum_sampled_q_AU"] for row in rrows])
    final_q = w1([row["final_q_AU"] for row in lrows], [row["final_q_AU"] for row in rrows])
    final_i = w1([row["final_i_deg"] for row in lrows], [row["final_i_deg"] for row in rrows])
    first30 = w1(censored(lrows, 30, horizon), censored(rrows, 30, horizon))
    first35 = w1(censored(lrows, 35, horizon), censored(rrows, 35, horizon))
    metrics.update({
        "final_bound_count_absolute_difference": bound,
        "w1_minimum_sampled_q_AU": min_q, "w1_final_q_AU": final_q,
        "w1_final_i_deg": final_i,
        "w1_censored_first_q30_divided_by_horizon": first30,
        "w1_censored_first_q35_divided_by_horizon": first35,
    })
    checks.update({
        "bound_count_within_gate": bound <= limits["max_bound_count_difference"],
        "w1_minimum_q_within_gate": min_q <= limits["max_w1_minimum_sampled_q_AU"],
        "w1_final_q_within_gate": final_q <= limits["max_w1_final_q_AU"],
        "w1_final_i_within_gate": final_i <= limits["max_w1_final_i_deg"],
        "w1_first_q30_within_gate": first30 <= limits[
            "max_w1_censored_first_q30_divided_by_horizon"
        ],
        "w1_first_q35_within_gate": first35 <= limits[
            "max_w1_censored_first_q35_divided_by_horizon"
        ],
    })
    metrics["checks"] = checks; metrics["passes"] = all(checks.values())
    return metrics


def invariant_gate(landmark: dict[str, Any], contract: dict[str, Any]) -> dict[str, bool]:
    values = landmark["maximum_active_invariant_drifts"]
    limits = contract["numerical_gates"]
    return {
        "all_finite": landmark["summary"]["all_particles_all_samples_finite_cartesian_and_osculating"],
        "energy": values["relative_compensated_active_energy_drift"]
        <= limits["max_relative_compensated_active_energy_drift"],
        "angular": values["relative_active_com_angular_momentum_vector_drift"]
        <= limits["max_relative_active_com_angular_momentum_vector_drift"],
        "linear": values["scale_normalized_active_linear_momentum_residual"]
        <= limits["max_scale_normalized_active_linear_momentum_residual"],
    }


def recompute_analysis(finals: dict[str, dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {"primary": {}, "audit": {}}
    invariant_rows = []
    for horizon in HORIZONS:
        key = str(int(horizon)); table["primary"][key] = {}; table["audit"][key] = {}
        for arm_id, receipt in finals.items():
            landmark = receipt["landmarks"][key]
            if recompute_landmark(landmark) != landmark["summary"]:
                raise VerificationError("stored primary landmark summary mismatch")
            checks = invariant_gate(landmark, contract)
            invariant_rows.append({"arm_id": arm_id, "horizon_years": horizon,
                                   "checks": checks, "passes": all(checks.values())})
            if arm_id.startswith("AUDIT-"):
                table["audit"][key][AUDIT_BASE[arm_id]] = landmark
            else:
                table["primary"][key][arm_id] = landmark
    pairs = [pair_gate(table["primary"][str(int(horizon))][arm],
                       table["audit"][str(int(horizon))][arm], horizon, contract, arm)
             for horizon in HORIZONS for arm in PRIMARY_ARMS]
    numerical = all(row["passes"] for row in pairs) and all(row["passes"] for row in invariant_rows)
    effect_rows: dict[str, dict[str, Any]] = {"primary": {}, "audit": {}}
    labels: dict[str, dict[str, str]] = {"primary": {}, "audit": {}}
    for resolution in ("primary", "audit"):
        for horizon in CLASS_HORIZONS:
            key = str(int(horizon)); row = effects(table[resolution][key])
            effect_rows[resolution][key] = row; labels[resolution][key] = raw_label(row)
    supports: dict[str, Any] = {}; promoted: dict[str, str] = {}
    official: str | None = None
    if not numerical:
        state = "NUMERICALLY_UNRESOLVED"
    elif any(labels["primary"][str(int(h))] != labels["audit"][str(int(h))]
             for h in CLASS_HORIZONS):
        state = "TIMESTEP_SENSITIVE"
    elif labels["primary"]["500000"] != labels["primary"]["1000000"]:
        state = "HORIZON_SENSITIVE"
    else:
        for resolution in ("primary", "audit"):
            supports[resolution] = support(table[resolution]["1000000"])
            promoted[resolution] = promote(labels[resolution]["1000000"], supports[resolution])
        if promoted["primary"] != promoted["audit"]:
            state = "TIMESTEP_SENSITIVE"
        else:
            state = "PRIMARY_NUMERICS_COMPLETE_AWAITING_REPLAY_AND_DOP853"
            official = promoted["primary"]
    bridge = {resolution: effects(table[resolution]["250000"])
              for resolution in ("primary", "audit")}
    return {
        "analysis_state": state, "official_classification": None,
        "primary_screen_label": official,
        "classification_suppressed_until_replay_and_dop853": True,
        "structural_effects_q35": effect_rows, "structural_raw_labels_q35": labels,
        "event_support_1M_by_resolution": supports,
        "event_floor_promoted_labels_by_resolution": promoted,
        "bridge_250k_effects_q35_descriptive_only": bridge,
        "timestep_pair_gates": pairs, "active_invariant_gates": invariant_rows,
        "all_primary_numerical_gates_pass": numerical,
    }


def read_primary_ledger(
    path: Path, label: str, registration_hash: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    safe_regular(path, "primary attempt ledger")
    rows: list[dict[str, Any]] = []
    open_keys: set[tuple[str, int, int]] = set()
    attempts: dict[tuple[str, int], int] = {}
    completed = {arm: 0 for arm in ALL_ARMS}
    chains = {arm: SEGMENT_GENESIS for arm in ALL_ARMS}
    passes: dict[tuple[str, int], dict[str, Any]] = {}
    starts: dict[tuple[str, int, int], dict[str, Any]] = {}
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise VerificationError("primary attempt ledger has a nonterminated tail")
    for sequence, raw in enumerate(payload.splitlines(), start=1):
        row = json.loads(raw, object_pairs_hook=unique_pairs, parse_float=finite_float,
                         parse_constant=reject_constant)
        if canonical(row) != raw:
            raise VerificationError("primary attempt ledger row is noncanonical")
        if (row.get("schema") != PRIMARY_ATTEMPT_SCHEMA
                or type(row.get("sequence")) is not int
                or row.get("sequence") != sequence or row.get("execution_label") != label):
            raise VerificationError("primary attempt ledger identity/order changed")
        arm = row.get("arm_id"); segment = row.get("segment_index"); attempt = row.get("attempt_index")
        if (arm not in ALL_ARMS or not isinstance(segment, int) or isinstance(segment, bool)
                or not 0 <= segment < 20):
            raise VerificationError("primary ledger arm/segment changed")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= 3:
            raise VerificationError("primary ledger attempt index changed")
        key = (arm, segment, attempt); pair = (arm, segment)
        if row.get("event") == "START":
            exact_keys(row, {
                "schema", "sequence", "event", "execution_label", "arm_id",
                "segment_index", "attempt_index", "predecessor_segment_chain_head",
                "input_key_sha256",
            }, "primary ledger START")
            expected = attempts.get(pair, 0) + 1
            expected_key = digest_bytes(canonical({
                "registration_sha256": registration_hash, "execution_label": label,
                "arm_id": arm, "segment_index": segment,
                "predecessor_segment_chain_head": chains[arm],
            }))
            if (attempt != expected or row["input_key_sha256"] != expected_key
                    or row["predecessor_segment_chain_head"] != chains[arm]
                    or segment != completed[arm] or pair in passes
                    or any(open_key[0] == arm for open_key in open_keys)):
                raise VerificationError("primary ledger START binding changed")
            attempts[pair] = attempt; open_keys.add(key); starts[key] = row
        elif row.get("event") in ("PASS", "FAIL"):
            base = {"schema", "sequence", "event", "execution_label", "arm_id",
                    "segment_index", "attempt_index", "return_code"}
            passed = base | {"segment_chain_head"}
            recovered_pass = passed | {"recovered_committed_attempt_at_resume"}
            failed = base | {
                "failure_class", "fail_event_sha256", "failure_receipt_filename",
                "failure_receipt_sha256",
                "complete_uncommitted_attempt_semantic_sha256",
                "complete_uncommitted_attempt_decoded_state_sha256",
            }
            if set(row) not in (failed, passed, recovered_pass):
                raise VerificationError("primary ledger terminal fields changed")
            if key not in open_keys:
                raise VerificationError("primary ledger terminal pairing changed")
            if row["event"] == "PASS":
                if (set(row) not in (passed, recovered_pass)
                        or type(row["return_code"]) is not int or row["return_code"] != 0
                        or not lower_sha256(row["segment_chain_head"]) or pair in passes):
                    raise VerificationError("primary ledger PASS semantics changed")
                if set(row) == recovered_pass and row[
                    "recovered_committed_attempt_at_resume"
                ] is not True:
                    raise VerificationError("primary recovered PASS marker changed")
                passes[pair] = row; completed[arm] += 1; chains[arm] = row["segment_chain_head"]
            else:
                failure_class = row.get("failure_class")
                return_code = row["return_code"]
                coherent = (
                    (failure_class == "RECOVERED_UNCOMMITTED" and return_code is None)
                    or (failure_class == "CHILD_EXIT_NONZERO" and type(return_code) is int
                        and return_code > 0)
                    or (failure_class == "CHILD_SIGNAL" and type(return_code) is int
                        and return_code < 0)
                    or (failure_class in ("SEGMENT_TIMEOUT", "CHILD_RSS_LIMIT")
                        and type(return_code) is int and return_code == -9)
                )
                receipt_name = f"failure_{arm}_segment_{segment:02d}_attempt_{attempt:02d}.json"
                event_core = {
                    "schema": PRIMARY_FAILURE_SCHEMA, "event": "FAIL",
                    "execution_label": label, "arm_id": arm,
                    "start_sequence": starts[key]["sequence"],
                    "segment_index": segment, "attempt_index": attempt,
                    "return_code": return_code, "failure_class": failure_class,
                    "complete_uncommitted_attempt_semantic_sha256": row[
                        "complete_uncommitted_attempt_semantic_sha256"
                    ],
                    "complete_uncommitted_attempt_decoded_state_sha256": row[
                        "complete_uncommitted_attempt_decoded_state_sha256"
                    ],
                }
                if (set(row) != failed or failure_class not in PRIMARY_FAILURE_CLASSES
                        or not coherent or row["failure_receipt_filename"] != receipt_name
                        or not lower_sha256(row["failure_receipt_sha256"])
                        or ((row["complete_uncommitted_attempt_semantic_sha256"] is None)
                            != (row[
                                "complete_uncommitted_attempt_decoded_state_sha256"
                            ] is None))
                        or (row["complete_uncommitted_attempt_semantic_sha256"] is not None
                            and (not lower_sha256(row[
                                "complete_uncommitted_attempt_semantic_sha256"
                            ]) or not lower_sha256(row[
                                "complete_uncommitted_attempt_decoded_state_sha256"
                            ])))
                        or row["fail_event_sha256"] != digest_bytes(canonical(event_core))):
                    raise VerificationError("primary ledger FAIL semantics changed")
            open_keys.remove(key)
        else:
            raise VerificationError("primary ledger event changed")
        rows.append(row)
    if open_keys or any(value != 20 for value in completed.values()) or len(passes) != 1000:
        raise VerificationError("complete primary output has an open attempt")
    return rows, passes


def primary_quarantine_names(arm: str, segment: int, attempt: int) -> set[str]:
    return {
        f"{arm}_attempt_{attempt:02d}_segment_{segment:02d}_attempt_{attempt:02d}_state.bin",
        f"{arm}_attempt_{attempt:02d}_segment_{segment:02d}_attempt_{attempt:02d}_receipt.json",
        f"{arm}_attempt_{attempt:02d}_segment_{segment:02d}_attempt_{attempt:02d}_state.bin.pending",
        f"{arm}_attempt_{attempt:02d}_segment_{segment:02d}_attempt_{attempt:02d}_receipt.json.pending",
        f"{arm}_attempt_{attempt:02d}_segment_{segment:02d}_commit.json.pending",
        f"{arm}_attempt_{attempt:02d}_failure_segment_{segment:02d}_receipt_pending_bytes.bin",
    }


def validate_primary_complete_attempt_evidence(value: Any) -> None:
    exact_keys(value, {
        "semantic_segment_payload_sha256", "segment_chain_head",
        "decoded_integrator_state_sha256", "sampled_state_stream_sha256",
        "raw_checkpoint_sha256", "raw_checkpoint_size_bytes",
        "attempt_receipt_sha256",
    }, "complete uncommitted primary attempt evidence")
    if (any(not lower_sha256(value[key]) for key in (
            "semantic_segment_payload_sha256", "segment_chain_head",
            "decoded_integrator_state_sha256", "sampled_state_stream_sha256",
            "raw_checkpoint_sha256", "attempt_receipt_sha256",
        )) or type(value["raw_checkpoint_size_bytes"]) is not int
            or value["raw_checkpoint_size_bytes"] <= 0):
        raise VerificationError("complete primary attempt evidence changed")


def verify_quarantined_complete_attempt(
    output_root: Path, failure_dir: Path, fail: dict[str, Any],
    evidence: dict[str, Any], contract: dict[str, Any],
    expanded: dict[str, list[list[Any]]], inventory_names: set[str],
) -> None:
    validate_primary_complete_attempt_evidence(evidence)
    arm = fail["arm_id"]; segment = fail["segment_index"]
    attempt = fail["attempt_index"]
    logical_state = f"segment_{segment:02d}_attempt_{attempt:02d}_state.bin"
    logical_receipt = f"segment_{segment:02d}_attempt_{attempt:02d}_receipt.json"
    state_name = f"{arm}_attempt_{attempt:02d}_{logical_state}"
    receipt_name = f"{arm}_attempt_{attempt:02d}_{logical_receipt}"
    if state_name not in inventory_names or receipt_name not in inventory_names:
        raise VerificationError("complete primary attempt lacks its quarantine pair")
    state_path = failure_dir / state_name
    receipt_path = failure_dir / receipt_name
    receipt = read_object(receipt_path)
    exact_keys(receipt, set(PRIMARY_SEGMENT_SEMANTIC_FIELDS) | {
        "schema", "experiment_id", "previous_segment_chain_head", "segment_chain_head",
        "checkpoint_filename", "checkpoint_sha256", "checkpoint_size_bytes",
        "raw_checkpoint_integrity_only", "provenance",
    }, "quarantined complete primary receipt")
    configuration, arm_class, dt_years = primary_arm_identity(arm, contract)
    provenance = receipt["provenance"]
    exact_keys(provenance, {"attempt_index", "wall_seconds", "peak_rss_bytes"},
               "quarantined complete primary provenance")
    first = segment * 1000 + (0 if segment == 0 else 1)
    last = (segment + 1) * 1000
    if (receipt["schema"] != RECEIPT_SCHEMA or receipt["experiment_id"] != EXPERIMENT_ID
            or receipt["arm_id"] != arm or receipt["segment_index"] != segment
            or receipt["configuration_id"] != configuration
            or receipt["arm_class"] != arm_class or receipt["dt_years"] != dt_years
            or receipt["checkpoint_filename"] != logical_state
            or receipt["raw_checkpoint_integrity_only"] is not True
            or provenance["attempt_index"] != attempt
            or not finite_number(provenance["wall_seconds"])
            or provenance["wall_seconds"] < 0.0
            or provenance["wall_seconds"] >= contract["resource_caps_per_execution"][
                "max_wall_seconds_per_segment_attempt"
            ]
            or type(provenance["peak_rss_bytes"]) is not int
            or not 0 <= provenance["peak_rss_bytes"] <= contract[
                "resource_caps_per_execution"
            ]["max_peak_rss_bytes_per_process"]
            or receipt["first_sample_index"] != first
            or receipt["last_sample_index"] != last
            or receipt["new_sample_count"] != last - first + 1
            or receipt["sample_count_total"] != last + 1
            or receipt["start_years"] != segment * 50_000.0
            or receipt["end_years"] != (segment + 1) * 50_000.0):
        raise VerificationError("quarantined complete primary identity changed")
    if (digest_file(state_path) != receipt["checkpoint_sha256"]
            or state_path.stat().st_size != receipt["checkpoint_size_bytes"]
            or digest_file(receipt_path) != evidence["attempt_receipt_sha256"]
            or receipt["checkpoint_sha256"] != evidence["raw_checkpoint_sha256"]
            or receipt["checkpoint_size_bytes"] != evidence["raw_checkpoint_size_bytes"]
            or receipt["decoded_integrator_state_sha256"]
            != evidence["decoded_integrator_state_sha256"]
            or receipt["sampled_state_stream_sha256"]
            != evidence["sampled_state_stream_sha256"]):
        raise VerificationError("quarantined complete primary raw/evidence binding changed")
    previous = SEGMENT_GENESIS
    if segment:
        commit = read_object(
            output_root / "arms" / arm / "segments"
            / f"segment_{segment - 1:02d}_commit.json"
        )
        prior = read_object(
            output_root / "arms" / arm / "segments" / commit["attempt_receipt_filename"]
        )
        previous = prior["segment_chain_head"]
    semantic = primary_segment_semantic_payload(receipt)
    semantic_sha256 = digest_bytes(canonical(semantic))
    chain = digest_bytes(SEGMENT_DOMAIN + bytes.fromhex(previous) + canonical(semantic))
    if (receipt["previous_segment_chain_head"] != previous
            or receipt["segment_chain_head"] != chain
            or chain != evidence["segment_chain_head"]
            or semantic_sha256 != evidence["semantic_segment_payload_sha256"]):
        raise VerificationError("quarantined complete primary semantics changed")
    ids = validate_primary_tracker(receipt["tracker"], receipt["end_years"])
    expected_initial = primary_active_invariants(
        expanded[configuration],
        contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"],
    )
    if receipt["initial_active_invariants"] != expected_initial:
        raise VerificationError("quarantined primary invariant baseline changed")
    validate_primary_drifts(
        receipt["maximum_active_invariant_drifts"],
        "quarantined primary maximum drifts",
    )
    expected_landmarks = {
        str(int(year)) for year in HORIZONS if year <= receipt["end_years"]
    }
    if set(receipt["landmarks"]) != expected_landmarks:
        raise VerificationError("quarantined primary landmark ownership changed")
    for key, landmark in receipt["landmarks"].items():
        validate_primary_landmark(landmark, float(key), ids)
    verify_primary_checkpoint_endpoint(
        state_path, receipt, contract, expanded[configuration]
    )


def verify_primary_failures(
    output_root: Path, ledger: Sequence[dict[str, Any]], label: str,
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    failure_dir = output_root / "failures"
    starts = {
        (row["arm_id"], row["segment_index"], row["attempt_index"]): row
        for row in ledger if row["event"] == "START"
    }
    failed = [row for row in ledger if row["event"] == "FAIL"]
    expected_receipts = {row["failure_receipt_filename"] for row in failed}
    actual_receipts: set[str] = set()
    actual_quarantine: set[str] = set()
    for entry in failure_dir.iterdir():
        safe_regular(entry, "primary failure/quarantine artifact")
        if entry.name.startswith("."):
            raise VerificationError("primary failure directory has a pending artifact")
        if entry.name.startswith("failure_") and entry.name.endswith(".json"):
            actual_receipts.add(entry.name)
        else:
            actual_quarantine.add(entry.name)
    if actual_receipts != expected_receipts:
        raise VerificationError("primary FAIL/receipt sets are not bijective")
    bound_quarantine: set[str] = set()
    complete_by_segment: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for fail in failed:
        key = (fail["arm_id"], fail["segment_index"], fail["attempt_index"])
        start = starts.get(key)
        if start is None:
            raise VerificationError("primary failure receipt lacks its exact START")
        filename = fail["failure_receipt_filename"]
        path = failure_dir / filename
        receipt = read_object(path)
        expected_receipt_bytes = (
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        if path.read_bytes() != expected_receipt_bytes:
            raise VerificationError("primary failure receipt is noncanonical")
        exact_keys(receipt, {
            "schema", "experiment_id", "execution_label", "arm_id", "segment_index",
            "attempt_index", "failure_class", "return_code", "start_sequence",
            "predecessor_segment_chain_head", "input_key_sha256", "fail_event_sha256",
            "complete_uncommitted_attempt", "quarantined_artifacts", "message",
            "authorizes_analysis",
        }, "primary failure receipt")
        complete = receipt["complete_uncommitted_attempt"]
        if complete is not None:
            validate_primary_complete_attempt_evidence(complete)
        complete_semantic = None if complete is None else complete.get(
            "semantic_segment_payload_sha256"
        )
        complete_decoded = None if complete is None else complete.get(
            "decoded_integrator_state_sha256"
        )
        event_core = {
            "schema": PRIMARY_FAILURE_SCHEMA, "event": "FAIL",
            "execution_label": label, "arm_id": fail["arm_id"],
            "start_sequence": start["sequence"], "segment_index": fail["segment_index"],
            "attempt_index": fail["attempt_index"], "return_code": fail["return_code"],
            "failure_class": fail["failure_class"],
            "complete_uncommitted_attempt_semantic_sha256": complete_semantic,
            "complete_uncommitted_attempt_decoded_state_sha256": complete_decoded,
        }
        if (
            receipt["schema"] != PRIMARY_FAILURE_SCHEMA
            or receipt["experiment_id"] != EXPERIMENT_ID
            or receipt["execution_label"] != label
            or receipt["arm_id"] != fail["arm_id"]
            or type(receipt["segment_index"]) is not int
            or receipt["segment_index"] != fail["segment_index"]
            or type(receipt["attempt_index"]) is not int
            or receipt["attempt_index"] != fail["attempt_index"]
            or receipt["failure_class"] != fail["failure_class"]
            or (receipt["return_code"] is not None
                and type(receipt["return_code"]) is not int)
            or receipt["return_code"] != fail["return_code"]
            or type(receipt["start_sequence"]) is not int
            or receipt["start_sequence"] != start["sequence"]
            or receipt["predecessor_segment_chain_head"]
            != start["predecessor_segment_chain_head"]
            or receipt["input_key_sha256"] != start["input_key_sha256"]
            or receipt["fail_event_sha256"] != digest_bytes(canonical(event_core))
            or receipt["fail_event_sha256"] != fail["fail_event_sha256"]
            or fail["complete_uncommitted_attempt_semantic_sha256"] != complete_semantic
            or fail["complete_uncommitted_attempt_decoded_state_sha256"] != complete_decoded
            or digest_file(path) != fail["failure_receipt_sha256"]
            or receipt["message"] != "REDACTED_NON_SEMANTIC_FAILURE_DETAIL"
            or receipt["authorizes_analysis"] is not False
        ):
            raise VerificationError("primary failure receipt binding changed")
        inventory = receipt["quarantined_artifacts"]
        if not isinstance(inventory, list):
            raise VerificationError("primary quarantine inventory changed")
        allowed = primary_quarantine_names(*key)
        previous = ""
        names: set[str] = set()
        for item in inventory:
            exact_keys(item, {"filename", "size_bytes", "sha256"}, "quarantine item")
            name = item["filename"]
            artifact = failure_dir / name
            if (not isinstance(name, str) or name not in allowed or name <= previous
                    or not isinstance(item["size_bytes"], int)
                    or isinstance(item["size_bytes"], bool) or item["size_bytes"] < 0
                    or not lower_sha256(item["sha256"])):
                raise VerificationError("primary quarantine row changed")
            safe_regular(artifact, "primary quarantined artifact")
            if (artifact.stat().st_size != item["size_bytes"]
                    or digest_file(artifact) != item["sha256"]):
                raise VerificationError("primary quarantine artifact binding changed")
            names.add(name); previous = name
        if bound_quarantine & names:
            raise VerificationError("primary quarantine artifact is multiply bound")
        bound_quarantine.update(names)
        if complete is not None:
            verify_quarantined_complete_attempt(
                output_root, failure_dir, fail, complete, contract, expanded, names
            )
            complete_by_segment.setdefault(
                (fail["arm_id"], fail["segment_index"]), []
            ).append(complete)
    if bound_quarantine != actual_quarantine:
        raise VerificationError("primary quarantine inventory has an omission or extra")
    return complete_by_segment


def verify_primary_output(
    output_root: Path, label: str, contract: dict[str, Any], initial: dict[str, Any],
    expanded: dict[str, list[list[Any]]], package_paths: dict[str, Path],
    expected_a_prerequisite: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    safe_dir(output_root, "primary output root")
    exact_root = {
        "run_manifest.json", "attempt_ledger.jsonl", "arms", "failures",
        "execution.lock", "result_v1.json",
    }
    if {entry.name for entry in output_root.iterdir()} != exact_root:
        raise VerificationError("primary output root inventory changed")
    safe_dir(output_root / "arms", "primary arms root")
    safe_dir(output_root / "failures", "primary failure root")
    verify_unlocked_execution_lock(
        output_root / "execution.lock", "primary execution lock"
    )
    arm_entries = list((output_root / "arms").iterdir())
    if ({entry.name for entry in arm_entries} != set(ALL_ARMS)
            or any(entry.is_symlink() or not entry.is_dir() for entry in arm_entries)):
        raise VerificationError("primary arm inventory changed")
    result_path = output_root / "result_v1.json"
    result = read_object(result_path)
    registration_hash = digest_file(package_paths["registration_v1.json"])
    bindings = {
        "registration_sha256": registration_hash,
        "contract_sha256": digest_file(package_paths["contract_v1.json"]),
        "seed_manifest_sha256": digest_file(package_paths["seed_manifest_v1.json"]),
        "initial_states_sha256": digest_file(package_paths["initial_states_v1.json"]),
    }
    exact_keys(result, {
        "schema", "experiment_id", "artifact_class", "execution_label", "input_bindings",
        "runtime", "semantic", "semantic_sha256", "raw_artifact_integrity",
        "resource_provenance",
        "scientific_classification_emitted", "mandatory_nonclaim",
    }, "primary result")
    expected_runtime = {key: contract["runtime_lock"][key] for key in (
        "python_version", "python_executable_sha256", "rebound_version", "rebound_build",
        "rebound_binary_sha256", "rebound_python_source_file_count",
        "rebound_python_source_sha256",
    )}
    if (result["schema"] != PRIMARY_RESULT_SCHEMA or result["experiment_id"] != EXPERIMENT_ID
            or result["artifact_class"]
            != "COMPLETE_PRIMARY_NUMERICAL_OUTPUT_AWAITING_INDEPENDENT_VERIFICATION"
            or result["execution_label"] != label or result["input_bindings"] != bindings
            or result["runtime"] != expected_runtime
            or result["scientific_classification_emitted"] is not False
            or result["mandatory_nonclaim"] != contract["mandatory_nonclaim"]
            or result["semantic_sha256"] != digest_bytes(canonical(result["semantic"]))):
        raise VerificationError("primary result identity/digest changed")
    manifest = read_object(output_root / "run_manifest.json")
    exact_keys(manifest, {
        "schema", "experiment_id", "execution_label", "registration_sha256",
        "contract_sha256", "seed_manifest_sha256", "initial_states_sha256",
        "a_prerequisite", "arm_order", "workers", "segment_count_per_arm",
        "segment_years",
    }, "primary run manifest")
    if (manifest["schema"] != PRIMARY_RUN_MANIFEST_SCHEMA
            or manifest["experiment_id"] != EXPERIMENT_ID
            or manifest["execution_label"] != label
            or manifest["registration_sha256"] != bindings["registration_sha256"]
            or manifest["contract_sha256"] != bindings["contract_sha256"]
            or manifest["seed_manifest_sha256"] != bindings["seed_manifest_sha256"]
            or manifest["initial_states_sha256"] != bindings["initial_states_sha256"]
            or manifest["a_prerequisite"] != expected_a_prerequisite
            or manifest["arm_order"] != list(ALL_ARMS) or manifest["workers"] != 4
            or manifest["segment_count_per_arm"] != 20 or manifest["segment_years"] != 50_000.0):
        raise VerificationError("primary run manifest changed")
    ledger, terminal_passes = read_primary_ledger(
        output_root / "attempt_ledger.jsonl", label, registration_hash
    )
    complete_failed_attempts = verify_primary_failures(
        output_root, ledger, label, contract, expanded
    )
    provenance = result["resource_provenance"]
    exact_keys(provenance, {
        "wall_seconds", "coordinator_peak_rss_bytes", "output_bytes_before_result",
        "attempt_ledger_sha256",
    }, "primary resource provenance")
    caps = contract["resource_caps_per_execution"]
    actual_before_result = sum(
        path.stat().st_size for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != result_path
    )
    if (not finite_number(provenance["wall_seconds"]) or provenance["wall_seconds"] < 0.0
            or provenance["wall_seconds"] >= caps["max_wall_seconds_total"]
            or not isinstance(provenance["coordinator_peak_rss_bytes"], int)
            or isinstance(provenance["coordinator_peak_rss_bytes"], bool)
            or not 0 <= provenance["coordinator_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or not isinstance(provenance["output_bytes_before_result"], int)
            or isinstance(provenance["output_bytes_before_result"], bool)
            or not 0 <= provenance["output_bytes_before_result"] <= caps["max_output_bytes"]
            or provenance["output_bytes_before_result"] != actual_before_result
            or provenance["attempt_ledger_sha256"]
            != digest_file(output_root / "attempt_ledger.jsonl")):
        raise VerificationError("primary resource provenance changed")
    finals = {
        arm_id: verify_segment_tree(output_root, arm_id, contract, expanded)
        for arm_id in ALL_ARMS
    }
    if result["raw_artifact_integrity"] != primary_raw_artifact_integrity_inventory(
        output_root
    ):
        raise VerificationError("primary raw artifact integrity root/inventory changed")
    for arm_id in ALL_ARMS:
        for segment in range(20):
            commit = read_object(output_root / "arms" / arm_id / "segments"
                                 / f"segment_{segment:02d}_commit.json")
            receipt = read_object(output_root / "arms" / arm_id / "segments"
                                  / commit["attempt_receipt_filename"])
            attempt = receipt["provenance"]["attempt_index"]
            passed = terminal_passes.get((arm_id, segment))
            if (passed is None or passed["attempt_index"] != attempt
                    or passed["segment_chain_head"] != receipt["segment_chain_head"]):
                raise VerificationError("committed primary segment lacks a matching PASS")
            semantic_sha256 = digest_bytes(canonical(
                primary_segment_semantic_payload(receipt)
            ))
            for evidence in complete_failed_attempts.get((arm_id, segment), []):
                if (semantic_sha256 != evidence["semantic_segment_payload_sha256"]
                        or receipt["segment_chain_head"] != evidence["segment_chain_head"]
                        or receipt["decoded_integrator_state_sha256"]
                        != evidence["decoded_integrator_state_sha256"]
                        or receipt["sampled_state_stream_sha256"]
                        != evidence["sampled_state_stream_sha256"]):
                    raise VerificationError(
                        "primary retry changed decoded/scientific semantics"
                    )
    computed = recompute_analysis(finals, contract)
    if result["semantic"]["analysis"] != computed:
        raise VerificationError("stored primary analysis differs from independent recomputation")
    digest_index = {row[0]: row[-1] for row in initial["configuration_states"]}
    expected_arms = []
    for arm_id in ALL_ARMS:
        receipt = finals[arm_id]
        expected_arms.append({
            "arm_id": arm_id, "configuration_id": receipt["configuration_id"],
            "arm_class": receipt["arm_class"], "dt_years": receipt["dt_years"],
            "registered_expanded_initial_state_sha256": digest_index[receipt["configuration_id"]],
            "sample_count": receipt["sample_count_total"],
            "segment_chain_head_sha256": receipt["segment_chain_head"],
            "maximum_active_invariant_drifts": receipt["maximum_active_invariant_drifts"],
            "landmarks": receipt["landmarks"],
        })
    expected_semantic = {
        "schema": "jx-xp2-primary-semantic/v3", "experiment_id": EXPERIMENT_ID,
        "arm_order": list(ALL_ARMS), "arms": expected_arms, "analysis": computed,
        "claim_ceiling": contract["claim_ceiling"], "official_classification": None,
        "classification_requires_verified_A_B_and_DOP853": True,
    }
    if result["semantic"] != expected_semantic:
        raise VerificationError("primary semantic payload differs from committed artifacts")
    return result, finals, computed


DOP_CHAIN_DOMAIN = b"jx-xp2-dop853-segment-chain/v1\0"
DOP_PAYLOAD_DOMAIN = b"jx-xp2-dop853-segment-payload/v1\0"
DOP_GENESIS = hashlib.sha256(DOP_CHAIN_DOMAIN + b"GENESIS").hexdigest()
DOP_SUBSET_DOMAIN = b"jx-xp2-dop853-selected-initial-state/v1\0"
DOP_FAILURE_EVENT_DOMAIN = b"jx-xp2-dop853-failure-event/v2\0"
DOP_FAILURE_CLASSES = {
    "InterruptedAttempt", "IntegrityError", "NumericalError",
    "ResourceLimitError", "UnexpectedFailure",
}


def expected_dop_initial(
    expanded: dict[str, list[list[Any]]], ids: Sequence[str], arm_id: str
) -> dict[str, Any]:
    rows = expanded[arm_id]
    active_count = len(rows) - 128
    by_id = {row[0]: row for row in rows[active_count:]}
    if active_count not in (5, 6) or set(ids) - set(by_id) or len(set(ids)) != 32:
        raise VerificationError("DOP853 registered subset is malformed")
    selected = rows[:active_count] + [by_id[logical_id] for logical_id in ids]
    states = [unpack6(row[3]) for row in selected]
    return {
        "arm_id": arm_id,
        "active_count": active_count,
        "logical_ids": [row[0] for row in selected],
        "masses": [f64(row[2]) for row in selected],
        "initial_state": [value for row in states for value in row[:3]]
        + [value for row in states for value in row[3:]],
        "tracer_block_indices": [int(logical_id[5:7]) for logical_id in ids],
        "initial_state_sha256": digest_bytes(DOP_SUBSET_DOMAIN + canonical(selected)),
        "registered_expanded_initial_state_sha256": digest_bytes(
            EXPANDED_DOMAIN + canonical(rows)
        ),
    }


def dop_norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def dop_cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def dop_state_from_hex(value: Any, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise VerificationError("DOP853 endpoint state encoding changed")
    output = []
    for word in value:
        if not isinstance(word, str):
            raise VerificationError("DOP853 endpoint state word is not text")
        try:
            number = float.fromhex(word)
        except ValueError as exc:
            raise VerificationError("DOP853 endpoint state word is not binary64 hex") from exc
        if not math.isfinite(number) or number.hex() != word:
            raise VerificationError("DOP853 endpoint state word is noncanonical/nonfinite")
        output.append(number)
    return output


def dop_matrix(state: Sequence[float], count: int) -> tuple[list[list[float]], list[list[float]]]:
    if len(state) != count * 6 or not all(math.isfinite(value) for value in state):
        raise VerificationError("DOP853 Cartesian state shape/finiteness changed")
    positions = [list(state[index * 3:(index + 1) * 3]) for index in range(count)]
    offset = count * 3
    velocities = [list(state[offset + index * 3:offset + (index + 1) * 3])
                  for index in range(count)]
    return positions, velocities


def dop_orbital_metric(
    position: Sequence[float], velocity: Sequence[float], sun_position: Sequence[float],
    sun_velocity: Sequence[float], G: float, sun_mass: float,
) -> dict[str, Any]:
    relative_position = [position[index] - sun_position[index] for index in range(3)]
    relative_velocity = [velocity[index] - sun_velocity[index] for index in range(3)]
    radius = dop_norm(relative_position)
    speed2 = math.fsum(value * value for value in relative_velocity)
    mu = G * sun_mass
    if not (radius > 0.0 and mu > 0.0 and math.isfinite(speed2)):
        raise VerificationError("DOP853 endpoint has invalid Sun-relative state")
    angular = dop_cross(relative_position, relative_velocity)
    angular_norm = dop_norm(angular)
    if not angular_norm > 0.0:
        raise VerificationError("DOP853 endpoint has degenerate angular momentum")
    velocity_cross_angular = dop_cross(relative_velocity, angular)
    eccentricity = dop_norm([
        velocity_cross_angular[index] / mu - relative_position[index] / radius
        for index in range(3)
    ])
    q = angular_norm * angular_norm / (mu * (1.0 + eccentricity))
    energy = 0.5 * speed2 - mu / radius
    if energy == 0.0:
        raise VerificationError("DOP853 endpoint has parabolic nonfinite semimajor axis")
    semimajor = -mu / (2.0 * energy)
    inclination = math.degrees(math.acos(max(-1.0, min(1.0, angular[2] / angular_norm))))
    values = (radius, eccentricity, q, energy, semimajor, inclination)
    if not all(math.isfinite(value) for value in values) or q < 0.0:
        raise VerificationError("DOP853 endpoint osculating metric is invalid")
    return {
        "a_AU": semimajor, "e": eccentricity, "i_deg": inclination,
        "q_AU": q, "distance_AU": radius, "finite": True,
        "bound": energy < 0.0 and eccentricity < 1.0,
    }


def dop_active_invariants(
    positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]],
    masses: Sequence[float], G: float,
) -> dict[str, Any]:
    if len(positions) != len(velocities) or len(positions) != len(masses):
        raise VerificationError("DOP853 active invariant array shape changed")
    total_mass = math.fsum(masses)
    if not total_mass > 0.0:
        raise VerificationError("DOP853 active system has invalid total mass")
    momentum = [math.fsum(mass * velocities[index][axis]
                          for index, mass in enumerate(masses)) for axis in range(3)]
    center_position = [math.fsum(mass * positions[index][axis]
                                 for index, mass in enumerate(masses)) / total_mass
                       for axis in range(3)]
    center_velocity = [value / total_mass for value in momentum]
    kinetic: list[float] = []
    angular_terms: list[list[float]] = [[], [], []]
    momentum_scale: list[float] = []
    for index, mass in enumerate(masses):
        position = [positions[index][axis] - center_position[axis] for axis in range(3)]
        velocity = [velocities[index][axis] - center_velocity[axis] for axis in range(3)]
        kinetic.append(0.5 * mass * math.fsum(value * value for value in velocity))
        cross = dop_cross(position, velocity)
        for axis in range(3):
            angular_terms[axis].append(mass * cross[axis])
        momentum_scale.append(mass * dop_norm(velocity))
    potential: list[float] = []
    for left in range(len(masses)):
        for right in range(left + 1, len(masses)):
            separation = dop_norm([positions[right][axis] - positions[left][axis]
                                   for axis in range(3)])
            if not separation > 0.0:
                raise VerificationError("DOP853 active-body endpoint collision")
            potential.append(-G * masses[left] * masses[right] / separation)
    result = {
        "intrinsic_energy": math.fsum(kinetic + potential),
        "com_angular_momentum": [math.fsum(values) for values in angular_terms],
        "linear_momentum": momentum,
        "linear_momentum_scale": math.fsum(momentum_scale),
    }
    flat = [result["intrinsic_energy"], result["linear_momentum_scale"],
            *result["com_angular_momentum"], *result["linear_momentum"]]
    if not all(math.isfinite(value) for value in flat):
        raise VerificationError("DOP853 active invariant is nonfinite")
    return result


def dop_invariant_drifts(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    energy_scale = abs(baseline["intrinsic_energy"])
    angular_scale = dop_norm(baseline["com_angular_momentum"])
    momentum_scale = baseline["linear_momentum_scale"]
    if not (energy_scale > 0.0 and angular_scale > 0.0 and momentum_scale > 0.0):
        raise VerificationError("DOP853 invariant normalization changed")
    result = {
        "relative_compensated_intrinsic_energy_drift": abs(
            current["intrinsic_energy"] - baseline["intrinsic_energy"]
        ) / energy_scale,
        "relative_com_angular_momentum_vector_drift": dop_norm([
            current["com_angular_momentum"][index]
            - baseline["com_angular_momentum"][index] for index in range(3)
        ]) / angular_scale,
        "scale_normalized_linear_momentum_residual": dop_norm([
            current["linear_momentum"][index] - baseline["linear_momentum"][index]
            for index in range(3)
        ]) / momentum_scale,
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in result.values()):
        raise VerificationError("DOP853 endpoint invariant drift is invalid")
    return result


def dop_endpoint(
    state: Sequence[float], arm: dict[str, Any], G: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    positions, velocities = dop_matrix(state, len(arm["logical_ids"]))
    active_count = arm["active_count"]
    invariants = dop_active_invariants(
        positions[:active_count], velocities[:active_count],
        arm["masses"][:active_count], G,
    )
    particles = {
        logical_id: dop_orbital_metric(
            positions[index], velocities[index], positions[0], velocities[0], G,
            arm["masses"][0],
        )
        for index, logical_id in enumerate(
            arm["logical_ids"][active_count:], start=active_count
        )
    }
    return particles, invariants


def dop_sample_chain(previous: str, year: float, state: Sequence[float]) -> str:
    if not lower_sha256(previous):
        raise VerificationError("DOP853 sampled-state chain head changed")
    payload = bytearray(DOP_CHAIN_DOMAIN)
    payload.extend(b"SAMPLE"); payload.extend(bytes.fromhex(previous))
    payload.extend(struct.pack(">dQ", float(year), len(state)))
    payload.extend(struct.pack(f">{len(state)}d", *state))
    return digest_bytes(bytes(payload))


def initial_dop_accumulator(arm: dict[str, Any], G: float) -> dict[str, Any]:
    metrics, baseline = dop_endpoint(arm["initial_state"], arm, G)
    particles = {}
    for logical_id, block in zip(
        arm["logical_ids"][arm["active_count"]:], arm["tracer_block_indices"], strict=True
    ):
        metric = metrics[logical_id]
        particles[logical_id] = {
            "block_index": block,
            "minimum_sampled_q_AU": metric["q_AU"],
            "first_sampled_below_year": {
                str(threshold): 0.0 if metric["q_AU"] < threshold else None
                for threshold in (30, 35, 40)
            },
            "current": metric,
        }
    return {
        "sample_count": 1,
        "sample_state_chain_head": dop_sample_chain(
            DOP_GENESIS, 0.0, arm["initial_state"]
        ),
        "baseline_invariants": baseline,
        "maximum_active_drifts": {
            "relative_compensated_intrinsic_energy_drift": 0.0,
            "relative_com_angular_momentum_vector_drift": 0.0,
            "scale_normalized_linear_momentum_residual": 0.0,
        },
        "particles": particles, "landmarks": {},
    }


def validate_dop_metric(metric: Any, label: str) -> None:
    exact_keys(metric, {"a_AU", "e", "i_deg", "q_AU", "distance_AU", "finite", "bound"}, label)
    if (any(not finite_number(metric[key]) for key in
            ("a_AU", "e", "i_deg", "q_AU", "distance_AU"))
            or metric["a_AU"] == 0.0 or metric["e"] < 0.0
            or not 0.0 <= metric["i_deg"] <= 180.0 or metric["q_AU"] < 0.0
            or metric["distance_AU"] <= 0.0 or metric["finite"] is not True
            or type(metric["bound"]) is not bool
            or metric["bound"] != (metric["a_AU"] > 0.0 and metric["e"] < 1.0)):
        raise VerificationError(f"{label} violates DOP853 metric semantics")


def validate_dop_first(first: Any, minimum_q: float, horizon: float, label: str) -> None:
    exact_keys(first, {"30", "35", "40"}, label)
    for threshold in (30, 35, 40):
        value = first[str(threshold)]
        if (value is not None and (not finite_number(value) or value < 0.0
                                   or value > horizon or value % 50.0 != 0.0)):
            raise VerificationError(f"{label} is off the frozen sample grid")
        if (value is not None) != (minimum_q < threshold):
            raise VerificationError(f"{label} contradicts prefix minimum")
    first30, first35, first40 = first["30"], first["35"], first["40"]
    if ((first30 is not None and (first35 is None or first40 is None
                                  or first40 > first35 or first35 > first30))
            or (first35 is not None and (first40 is None or first40 > first35))):
        raise VerificationError(f"{label} threshold crossings are not nested")


def validate_dop_accumulator(
    accumulator: Any, arm: dict[str, Any], segment: int, contract: dict[str, Any],
    endpoint_state: Sequence[float],
) -> None:
    exact_keys(accumulator, {
        "sample_count", "sample_state_chain_head", "baseline_invariants",
        "maximum_active_drifts", "particles", "landmarks",
    }, "DOP853 accumulator")
    if (type(accumulator["sample_count"]) is not int
            or accumulator["sample_count"] != 1 + (segment + 1) * 1000
            or not lower_sha256(accumulator["sample_state_chain_head"])):
        raise VerificationError("DOP853 accumulator sample ownership changed")
    baseline = accumulator["baseline_invariants"]
    exact_keys(baseline, {
        "intrinsic_energy", "com_angular_momentum", "linear_momentum",
        "linear_momentum_scale",
    }, "DOP853 baseline invariants")
    if (not finite_number(baseline["intrinsic_energy"])
            or baseline["intrinsic_energy"] == 0.0
            or not isinstance(baseline["com_angular_momentum"], list)
            or len(baseline["com_angular_momentum"]) != 3
            or not all(finite_number(value) for value in baseline["com_angular_momentum"])
            or dop_norm(baseline["com_angular_momentum"]) <= 0.0
            or not isinstance(baseline["linear_momentum"], list)
            or len(baseline["linear_momentum"]) != 3
            or not all(finite_number(value) for value in baseline["linear_momentum"])
            or not finite_number(baseline["linear_momentum_scale"])
            or baseline["linear_momentum_scale"] <= 0.0):
        raise VerificationError("DOP853 baseline invariant domain changed")
    G = contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
    _initial_metrics, expected_baseline = dop_endpoint(arm["initial_state"], arm, G)
    if baseline != expected_baseline:
        raise VerificationError("DOP853 baseline differs from registered initial state")
    drifts = accumulator["maximum_active_drifts"]
    exact_keys(drifts, {
        "relative_compensated_intrinsic_energy_drift",
        "relative_com_angular_momentum_vector_drift",
        "scale_normalized_linear_momentum_residual",
    }, "DOP853 maximum drift map")
    if any(not finite_number(value) or value < 0.0 for value in drifts.values()):
        raise VerificationError("DOP853 maximum drift domain changed")
    endpoint, endpoint_invariants = dop_endpoint(endpoint_state, arm, G)
    endpoint_drifts = dop_invariant_drifts(endpoint_invariants, baseline)
    if any(drifts[key] < value for key, value in endpoint_drifts.items()):
        raise VerificationError("DOP853 maximum drift is below retained endpoint")
    tracer_ids = arm["logical_ids"][arm["active_count"]:]
    block_by_id = dict(zip(tracer_ids, arm["tracer_block_indices"], strict=True))
    particles = accumulator["particles"]
    if not isinstance(particles, dict) or set(particles) != set(tracer_ids):
        raise VerificationError("DOP853 accumulator tracer identity changed")
    horizon = (segment + 1) * 50_000.0
    for logical_id in tracer_ids:
        particle = particles[logical_id]
        exact_keys(particle, {
            "block_index", "minimum_sampled_q_AU", "first_sampled_below_year", "current"
        }, "DOP853 accumulator particle")
        minimum = particle["minimum_sampled_q_AU"]
        if (particle["block_index"] != block_by_id[logical_id]
                or not finite_number(minimum) or minimum < 0.0):
            raise VerificationError("DOP853 accumulator particle identity/domain changed")
        validate_dop_metric(particle["current"], "DOP853 current metric")
        if particle["current"] != endpoint[logical_id] or minimum > particle["current"]["q_AU"]:
            raise VerificationError("DOP853 current metric differs from retained endpoint")
        validate_dop_first(particle["first_sampled_below_year"], minimum, horizon,
                           "DOP853 first passage")
    expected_landmarks = {str(int(year)) for year in HORIZONS if year <= horizon}
    landmarks = accumulator["landmarks"]
    if not isinstance(landmarks, dict) or set(landmarks) != expected_landmarks:
        raise VerificationError("DOP853 landmark ownership changed")
    sorted_ids = sorted(tracer_ids)
    for key in sorted(expected_landmarks, key=int):
        landmark = landmarks[key]; landmark_year = float(int(key))
        exact_keys(landmark, {"landmark_year", "event_counts", "bound_count", "particles"},
                   "DOP853 landmark")
        if landmark["landmark_year"] != landmark_year:
            raise VerificationError("DOP853 landmark year/key changed")
        exact_keys(landmark["event_counts"], {"30", "35", "40"},
                   "DOP853 landmark event counts")
        if (any(type(value) is not int or not 0 <= value <= 32
                for value in landmark["event_counts"].values())
                or type(landmark["bound_count"]) is not int
                or not 0 <= landmark["bound_count"] <= 32
                or not isinstance(landmark["particles"], list)
                or len(landmark["particles"]) != 32
                or [row.get("logical_id") if isinstance(row, dict) else None
                    for row in landmark["particles"]] != sorted_ids):
            raise VerificationError("DOP853 landmark aggregate/cardinality changed")
        counts = {"30": 0, "35": 0, "40": 0}; bound_count = 0
        for row in landmark["particles"]:
            exact_keys(row, {
                "logical_id", "block_index", "minimum_sampled_q_AU", "q_AU", "a_AU",
                "e", "i_deg", "distance_AU", "finite", "bound", "ever_sampled_below",
                "first_sampled_below_year", "censored_first_below_divided_by_landmark",
            }, "DOP853 landmark particle")
            logical_id = row["logical_id"]; minimum = row["minimum_sampled_q_AU"]
            metric = {field: row[field] for field in
                      ("a_AU", "e", "i_deg", "q_AU", "distance_AU", "finite", "bound")}
            if (row["block_index"] != block_by_id[logical_id]
                    or not finite_number(minimum) or minimum < 0.0 or minimum > row["q_AU"]):
                raise VerificationError("DOP853 landmark particle identity/domain changed")
            validate_dop_metric(metric, "DOP853 landmark metric")
            validate_dop_first(row["first_sampled_below_year"], minimum, landmark_year,
                               "DOP853 landmark first passage")
            exact_keys(row["ever_sampled_below"], {"30", "35", "40"},
                       "DOP853 landmark indicator map")
            exact_keys(row["censored_first_below_divided_by_landmark"], {"30", "35", "40"},
                       "DOP853 landmark censor map")
            for threshold in (30, 35, 40):
                threshold_key = str(threshold)
                first = row["first_sampled_below_year"][threshold_key]
                indicator = first is not None
                censored = (landmark_year if first is None else first) / landmark_year
                if (type(row["ever_sampled_below"][threshold_key]) is not bool
                        or row["ever_sampled_below"][threshold_key] != indicator
                        or not finite_number(row["censored_first_below_divided_by_landmark"][threshold_key])
                        or row["censored_first_below_divided_by_landmark"][threshold_key] != censored
                        or not 0.0 <= censored <= 1.0):
                    raise VerificationError("DOP853 landmark indicator/censor changed")
                counts[threshold_key] += int(indicator)
            bound_count += int(row["bound"])
            final = particles[logical_id]
            if minimum < final["minimum_sampled_q_AU"]:
                raise VerificationError("DOP853 historical minimum contradicts later prefix")
            for threshold_key in ("30", "35", "40"):
                final_first = final["first_sampled_below_year"][threshold_key]
                expected_first = final_first if final_first is not None and final_first <= landmark_year else None
                if row["first_sampled_below_year"][threshold_key] != expected_first:
                    raise VerificationError("DOP853 landmark crossing contradicts later prefix")
            if landmark_year == horizon and (
                minimum != final["minimum_sampled_q_AU"]
                or row["first_sampled_below_year"] != final["first_sampled_below_year"]
                or metric != final["current"]
            ):
                raise VerificationError("DOP853 boundary landmark differs from endpoint")
        if landmark["event_counts"] != counts or landmark["bound_count"] != bound_count:
            raise VerificationError("DOP853 landmark summary differs from particles")


def validate_dop_transition(
    previous: dict[str, Any], current: dict[str, Any], arm: dict[str, Any], segment: int,
) -> None:
    if (current["sample_count"] != previous["sample_count"] + 1000
            or current["sample_state_chain_head"] == previous["sample_state_chain_head"]
            or current["baseline_invariants"] != previous["baseline_invariants"]):
        raise VerificationError("DOP853 segment accumulator did not extend its predecessor")
    for key, value in previous["maximum_active_drifts"].items():
        if current["maximum_active_drifts"][key] < value:
            raise VerificationError("DOP853 segment reduced a historical maximum drift")
    start = segment * 50_000.0
    for logical_id in arm["logical_ids"][arm["active_count"]:]:
        old = previous["particles"][logical_id]; new = current["particles"][logical_id]
        if (new["block_index"] != old["block_index"]
                or new["minimum_sampled_q_AU"] > old["minimum_sampled_q_AU"]):
            raise VerificationError("DOP853 segment rewrote tracer prefix history")
        for key in ("30", "35", "40"):
            old_first = old["first_sampled_below_year"][key]
            new_first = new["first_sampled_below_year"][key]
            if (old_first is not None and new_first != old_first) or (
                old_first is None and new_first is not None and new_first <= start
            ):
                raise VerificationError("DOP853 segment rewrote/backdated first passage")
    if not set(previous["landmarks"]).issubset(current["landmarks"]):
        raise VerificationError("DOP853 segment removed a landmark")
    for key, value in previous["landmarks"].items():
        if current["landmarks"][key] != value:
            raise VerificationError("DOP853 segment rewrote a landmark")
    end = (segment + 1) * 50_000.0
    expected_added = {str(int(year)) for year in HORIZONS if start < year <= end}
    if set(current["landmarks"]) - set(previous["landmarks"]) != expected_added:
        raise VerificationError("DOP853 segment added the wrong landmark")


def dop_checkpoint_path(output_root: Path, arm_id: str, segment: int) -> Path:
    return output_root / "checkpoints" / (
        f"checkpoint_{arm_id.replace('-', '_')}_segment_{segment:02d}.json"
    )


def dop_receipt_path(output_root: Path, arm_id: str, segment: int) -> Path:
    return output_root / "receipts" / (
        f"receipt_{arm_id.replace('-', '_')}_segment_{segment:02d}.json"
    )


def verify_dop_checkpoint(
    output_root: Path, arm: dict[str, Any], segment: int, bindings: dict[str, str],
    contract: dict[str, Any], previous_accumulator: dict[str, Any],
    previous_commitments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = dop_checkpoint_path(output_root, arm["arm_id"], segment)
    receipt_path = dop_receipt_path(output_root, arm["arm_id"], segment)
    safe_regular(checkpoint_path, "DOP853 immutable checkpoint")
    safe_regular(receipt_path, "DOP853 immutable parent receipt")
    checkpoint = read_object(checkpoint_path)
    exact_keys(checkpoint, {
        "schema", "experiment_id", "execution_label", "arm_id", "input_bindings",
        "completed_segment_index", "end_year", "end_state_hex", "accumulator",
        "segment_commitments", "segment_chain_head_sha256",
    }, "DOP853 immutable checkpoint")
    if (checkpoint["schema"] != DOP_CHECKPOINT_SCHEMA
            or checkpoint["experiment_id"] != EXPERIMENT_ID
            or checkpoint["execution_label"] != "DOP853-SENTINEL"
            or checkpoint["arm_id"] != arm["arm_id"] or checkpoint["input_bindings"] != bindings
            or type(checkpoint["completed_segment_index"]) is not int
            or checkpoint["completed_segment_index"] != segment
            or checkpoint["end_year"] != (segment + 1) * 50_000.0):
        raise VerificationError("DOP853 checkpoint identity/boundary changed")
    state = dop_state_from_hex(checkpoint["end_state_hex"], len(arm["logical_ids"]) * 6)
    validate_dop_accumulator(checkpoint["accumulator"], arm, segment, contract, state)
    validate_dop_transition(previous_accumulator, checkpoint["accumulator"], arm, segment)
    commitments = checkpoint["segment_commitments"]
    if not isinstance(commitments, list) or len(commitments) != segment + 1:
        raise VerificationError("DOP853 checkpoint commitment count changed")
    previous_chain = DOP_GENESIS
    for index, row in enumerate(commitments):
        exact_keys(row, {"segment_index", "end_year", "segment_payload_sha256",
                         "chain_head_sha256"}, "DOP853 segment commitment")
        if (type(row["segment_index"]) is not int or row["segment_index"] != index
                or row["end_year"] != (index + 1) * 50_000.0
                or not lower_sha256(row["segment_payload_sha256"])
                or not lower_sha256(row["chain_head_sha256"])):
            raise VerificationError("DOP853 segment commitment identity changed")
        expected_chain = digest_bytes(
            DOP_CHAIN_DOMAIN + bytes.fromhex(previous_chain)
            + bytes.fromhex(row["segment_payload_sha256"])
        )
        if row["chain_head_sha256"] != expected_chain:
            raise VerificationError("DOP853 segment commitment chain broke")
        previous_chain = expected_chain
    if (commitments[:-1] != previous_commitments
            or checkpoint["segment_chain_head_sha256"] != previous_chain):
        raise VerificationError("DOP853 immutable checkpoint rewrote its prefix")
    last_payload = {
        "arm_id": arm["arm_id"], "segment_index": segment,
        "start_year": segment * 50_000.0, "end_year": (segment + 1) * 50_000.0,
        "end_state_hex": checkpoint["end_state_hex"],
        "accumulator_sha256": digest_bytes(canonical(checkpoint["accumulator"])),
    }
    if commitments[-1]["segment_payload_sha256"] != digest_bytes(
        DOP_PAYLOAD_DOMAIN + canonical(last_payload)
    ):
        raise VerificationError("DOP853 retained boundary payload digest changed")
    receipt = read_object(receipt_path)
    exact_keys(receipt, {
        "schema", "experiment_id", "execution_label", "arm_id", "segment_index",
        "input_bindings", "checkpoint_filename", "checkpoint_sha256",
        "checkpoint_size_bytes", "segment_payload_sha256", "segment_chain_head_sha256",
        "parent_terminal_validation", "parent_resource_validation",
    }, "DOP853 immutable parent receipt")
    if (receipt["schema"] != DOP_SEGMENT_RECEIPT_SCHEMA
            or receipt["experiment_id"] != EXPERIMENT_ID
            or receipt["execution_label"] != "DOP853-SENTINEL"
            or receipt["arm_id"] != arm["arm_id"] or receipt["segment_index"] != segment
            or receipt["input_bindings"] != bindings
            or receipt["checkpoint_filename"] != checkpoint_path.name
            or receipt["checkpoint_sha256"] != digest_file(checkpoint_path)
            or type(receipt["checkpoint_size_bytes"]) is not int
            or receipt["checkpoint_size_bytes"] != checkpoint_path.stat().st_size
            or receipt["segment_payload_sha256"] != commitments[-1]["segment_payload_sha256"]
            or receipt["segment_chain_head_sha256"] != commitments[-1]["chain_head_sha256"]
            or receipt["parent_terminal_validation"]
            != "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS"):
        raise VerificationError("DOP853 immutable parent receipt binding changed")
    resources = receipt["parent_resource_validation"]
    exact_keys(resources, {
        "segment_elapsed_seconds", "terminal_child_peak_rss_bytes",
        "coordinator_peak_rss_bytes", "total_elapsed_seconds_before_publication",
        "output_bytes_projected", "free_disk_bytes_before_publication",
    }, "DOP853 segment resource validation")
    caps = contract["resource_caps_per_execution"]
    if (not finite_number(resources["segment_elapsed_seconds"])
            or not 0.0 <= resources["segment_elapsed_seconds"]
            < caps["max_wall_seconds_per_segment_attempt"]
            or type(resources["terminal_child_peak_rss_bytes"]) is not int
            or not 0 <= resources["terminal_child_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or type(resources["coordinator_peak_rss_bytes"]) is not int
            or not 0 <= resources["coordinator_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or not finite_number(resources["total_elapsed_seconds_before_publication"])
            or not 0.0 <= resources["total_elapsed_seconds_before_publication"]
            < caps["max_wall_seconds_total"]
            or type(resources["output_bytes_projected"]) is not int
            or not checkpoint_path.stat().st_size + receipt_path.stat().st_size
            <= resources["output_bytes_projected"] <= caps["max_output_bytes"]
            or type(resources["free_disk_bytes_before_publication"]) is not int
            or resources["free_disk_bytes_before_publication"] < caps["minimum_free_disk_bytes"]):
        raise VerificationError("DOP853 segment terminal resource proof changed")
    return checkpoint, receipt


def read_dop_ledger(
    path: Path, contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    safe_regular(path, "DOP853 attempt ledger")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise VerificationError("DOP853 attempt ledger lacks its final newline")
    rows: list[dict[str, Any]] = []
    commits: dict[tuple[str, int], dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    next_segment = {arm: 0 for arm in DOP_ARMS}; attempts: dict[tuple[str, int], int] = {}
    pending: dict[str, Any] | None = None
    for sequence, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            raise VerificationError("DOP853 attempt ledger contains a blank row")
        row = json.loads(raw, object_pairs_hook=unique_pairs, parse_float=finite_float,
                         parse_constant=reject_constant)
        if not isinstance(row, dict):
            raise VerificationError("DOP853 attempt ledger row is not an object")
        if raw + b"\n" != canonical(row) + b"\n":
            raise VerificationError("DOP853 attempt ledger row is not canonical JSONL")
        common = {"schema", "sequence", "event", "arm_id", "segment_index",
                  "attempt_number_for_segment"}
        event = row.get("event")
        if event == "SEGMENT_ATTEMPT_STARTED":
            fields = common
        elif event == "SEGMENT_ATTEMPT_FAILED":
            fields = common | {
                "failure_class", "fail_event_sha256",
                "failure_receipt_filename", "failure_receipt_sha256",
            }
        elif event == "SEGMENT_ATTEMPT_COMMITTED":
            fields = common | {"elapsed_seconds", "terminal_peak_rss_bytes",
                               "checkpoint_sha256", "segment_receipt_sha256"}
        else:
            raise VerificationError("DOP853 attempt ledger event changed")
        if "recovery" in row:
            if event == "SEGMENT_ATTEMPT_STARTED":
                raise VerificationError("DOP853 START cannot be a recovery row")
            fields |= {"recovery"}
        exact_keys(row, fields, "DOP853 attempt ledger row")
        arm = row["arm_id"]; segment = row["segment_index"]
        attempt = row["attempt_number_for_segment"]
        if (row["schema"] != DOP_ATTEMPT_SCHEMA
                or type(row["sequence"]) is not int or row["sequence"] != sequence
                or arm not in DOP_ARMS or type(segment) is not int or not 0 <= segment < 20
                or type(attempt) is not int or not 1 <= attempt <= 3):
            raise VerificationError("DOP853 attempt ledger identity/counter changed")
        if "recovery" in row and row["recovery"] not in {
            "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION",
            "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION",
        }:
            raise VerificationError("DOP853 attempt recovery marker changed")
        if (row.get("recovery") == "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION"
                and event != "SEGMENT_ATTEMPT_FAILED") or (
            row.get("recovery") == "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION"
                and event != "SEGMENT_ATTEMPT_COMMITTED"
        ):
            raise VerificationError("DOP853 recovery marker/event pairing changed")
        if event == "SEGMENT_ATTEMPT_STARTED":
            key = (arm, segment)
            position = DOP_ARMS.index(arm)
            if (pending is not None or segment != next_segment[arm]
                    or any(next_segment[prior] != 20 for prior in DOP_ARMS[:position])
                    or any(next_segment[later] != 0 for later in DOP_ARMS[position + 1:])
                    or attempt != attempts.get(key, 0) + 1 or key in commits):
                raise VerificationError("DOP853 attempt ledger START state changed")
            attempts[key] = attempt; pending = row
        else:
            if (pending is None or (arm, segment, attempt) != (
                pending["arm_id"], pending["segment_index"],
                pending["attempt_number_for_segment"],
            )):
                raise VerificationError("DOP853 attempt terminal row is unpaired")
            if event == "SEGMENT_ATTEMPT_FAILED":
                if (not isinstance(row["failure_class"], str)
                        or row["failure_class"] not in DOP_FAILURE_CLASSES
                        or not lower_sha256(row["fail_event_sha256"])
                        or row["failure_receipt_filename"]
                        != (f"failure_{arm}_segment_{segment:02d}_"
                            f"attempt_{attempt:02d}.json")
                        or not lower_sha256(row["failure_receipt_sha256"])):
                    raise VerificationError("DOP853 attempt failure class changed")
                if (row.get("recovery")
                        == "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION"
                        and row["failure_class"] != "InterruptedAttempt"):
                    raise VerificationError("DOP853 interrupted recovery class changed")
                failed.append(row)
            else:
                caps = contract["resource_caps_per_execution"]
                if (type(row["elapsed_seconds"]) not in (int, float)
                        or not finite_number(row["elapsed_seconds"])
                        or row["elapsed_seconds"] < 0.0
                        or row["elapsed_seconds"] >= caps["max_wall_seconds_per_segment_attempt"]
                        or type(row["terminal_peak_rss_bytes"]) is not int
                        or not 0 <= row["terminal_peak_rss_bytes"]
                        <= caps["max_peak_rss_bytes_per_process"]
                        or not lower_sha256(row["checkpoint_sha256"])
                        or not lower_sha256(row["segment_receipt_sha256"])):
                    raise VerificationError("DOP853 COMMITTED provenance changed")
                commits[(arm, segment)] = row; next_segment[arm] += 1
            pending = None
        rows.append(row)
    if pending is not None or any(value != 20 for value in next_segment.values()) \
            or len(commits) != 140:
        raise VerificationError("DOP853 complete ledger is incomplete")
    return rows, commits, failed


def verify_dop_failures(directory: Path, rows: Sequence[dict[str, Any]]) -> None:
    safe_dir(directory, "DOP853 failures directory")
    starts: dict[tuple[str, int, int], dict[str, Any]] = {}
    failed_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row["arm_id"], row["segment_index"], row["attempt_number_for_segment"])
        if row["event"] == "SEGMENT_ATTEMPT_STARTED":
            starts[key] = row
        elif row["event"] == "SEGMENT_ATTEMPT_FAILED":
            failed_rows.append(row)
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    expected_names = {row["failure_receipt_filename"] for row in failed_rows}
    if {path.name for path in entries} != expected_names or len(entries) != len(expected_names):
        raise VerificationError("DOP853 failure receipts do not match failed attempts")
    for ledger in failed_rows:
        path = directory / ledger["failure_receipt_filename"]
        receipt = read_object(path)
        exact_keys(receipt, {
            "schema", "experiment_id", "execution_label", "event", "arm_id",
            "segment_index", "attempt_number", "start_sequence", "failure_class",
            "recovery", "fail_event_sha256", "failure_message", "result_emitted",
            "scientific_classification_emitted", "mandatory_nonclaim",
        }, "DOP853 failure receipt")
        key = (ledger["arm_id"], ledger["segment_index"],
               ledger["attempt_number_for_segment"])
        start = starts.get(key)
        if start is None:
            raise VerificationError("DOP853 failure receipt lacks exact START")
        if (type(receipt["segment_index"]) is not int
                or type(receipt["attempt_number"]) is not int
                or type(receipt["start_sequence"]) is not int
                or not isinstance(receipt["failure_class"], str)
                or receipt["failure_class"] not in DOP_FAILURE_CLASSES
                or (receipt["recovery"] is not None and (
                    not isinstance(receipt["recovery"], str)
                    or receipt["recovery"]
                    != "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION"
                ))
                or ((receipt["recovery"] is not None)
                    != (receipt["failure_class"] == "InterruptedAttempt"))):
            raise VerificationError("DOP853 failure receipt types changed")
        core = {
            "schema": "jx-xp2-dop853-failure-event/v2",
            "experiment_id": EXPERIMENT_ID,
            "execution_label": "DOP853-SENTINEL",
            "event": "SEGMENT_ATTEMPT_FAILED",
            "arm_id": start["arm_id"],
            "segment_index": start["segment_index"],
            "attempt_number": start["attempt_number_for_segment"],
            "start_sequence": start["sequence"],
            "failure_class": receipt["failure_class"],
            "recovery": receipt["recovery"],
        }
        expected_event_sha = digest_bytes(DOP_FAILURE_EVENT_DOMAIN + canonical(core))
        expected_bytes = (
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode()
        if (receipt["schema"] != DOP_FAILURE_SCHEMA
                or receipt["experiment_id"] != EXPERIMENT_ID
                or receipt["execution_label"] != "DOP853-SENTINEL"
                or receipt["event"] != "SEGMENT_ATTEMPT_FAILED"
                or receipt["arm_id"] != ledger["arm_id"]
                or receipt["segment_index"] != ledger["segment_index"]
                or receipt["attempt_number"] != ledger["attempt_number_for_segment"]
                or receipt["start_sequence"] != start["sequence"]
                or receipt["failure_class"] != ledger["failure_class"]
                or receipt["recovery"] != ledger.get("recovery")
                or receipt["fail_event_sha256"] != expected_event_sha
                or ledger["fail_event_sha256"] != expected_event_sha
                or ledger["failure_receipt_filename"] != path.name
                or ledger["failure_receipt_sha256"] != digest_file(path)
                or path.read_bytes() != expected_bytes
                or receipt["failure_message"] != "REDACTED_NON_SEMANTIC_FAILURE_DETAIL"
                or receipt["result_emitted"] is not False
                or receipt["scientific_classification_emitted"] is not False
                or receipt["mandatory_nonclaim"]
                != "An incomplete or failed numerical sentinel is not a scientific result."):
            raise VerificationError("DOP853 failure receipt binding changed")


def dop_inventory_digest(output_root: Path) -> str:
    rows = []
    for directory_name in ("checkpoints", "receipts"):
        for path in sorted((output_root / directory_name).iterdir(), key=lambda value: value.name):
            rows.append([f"{directory_name}/{path.name}", digest_file(path), path.stat().st_size])
    return digest_bytes(canonical(rows))


def verify_dop_output(
    output_root: Path, contract: dict[str, Any], selected: Sequence[str],
    expanded: dict[str, list[list[Any]]], package_paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bool]:
    safe_dir(output_root, "DOP853 output root")
    expected_root = {"run_manifest.json", "attempt_ledger.jsonl", "checkpoints",
                     "receipts", "failures", "execution.lock", "result_v1.json"}
    if {path.name for path in output_root.iterdir()} != expected_root:
        raise VerificationError("DOP853 output root inventory changed")
    for name in (
        "run_manifest.json", "attempt_ledger.jsonl", "execution.lock", "result_v1.json"
    ):
        safe_regular(output_root / name, f"DOP853 {name}")
    verify_unlocked_execution_lock(
        output_root / "execution.lock", "DOP853 execution lock"
    )
    for name in ("checkpoints", "receipts", "failures"):
        safe_dir(output_root / name, f"DOP853 {name} directory")
    checkpoint_names = {
        dop_checkpoint_path(output_root, arm, segment).name
        for arm in DOP_ARMS for segment in range(20)
    }
    receipt_names = {
        dop_receipt_path(output_root, arm, segment).name
        for arm in DOP_ARMS for segment in range(20)
    }
    for directory, names in ((output_root / "checkpoints", checkpoint_names),
                             (output_root / "receipts", receipt_names)):
        entries = list(directory.iterdir())
        if ({path.name for path in entries} != names
                or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                       for path in entries)):
            raise VerificationError("DOP853 immutable segment inventory changed")
    bindings = {
        name: digest_file(package_paths[name]) for name in (
            "contract_v1.json", "seed_manifest_v1.json", "selection_manifest_v1.json",
            "initial_states_v1.json", "registration_v1.json", "run_independent.py",
        )
    }
    runtime = {key: contract["runtime_lock"][key] for key in (
        "python_version", "python_executable_sha256", "numpy_version",
        "numpy_multiarray_binary_sha256", "scipy_version", "scipy_rk_source_sha256",
        "scipy_dop853_coefficients_sha256", "native_thread_environment",
    )}
    manifest = read_object(output_root / "run_manifest.json")
    expected_manifest = {
        "schema": DOP_RUN_MANIFEST_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "execution_label": "DOP853-SENTINEL", "input_bindings": bindings,
        "runtime": runtime,
        "method": "CUSTOM_NEWTONIAN_RHS_VIA_SCIPY_SOLVE_IVP_DOP853_NO_REBOUND",
        "arm_ids": list(DOP_ARMS), "sentinel_tracer_count": 32,
        "duration_years": 1_000_000.0, "sample_cadence_years": 50.0,
        "segment_years": 50_000.0, "official_execution_count": 1,
        "raw_trajectories_retained": False, "observed_or_private_data_used": False,
        "gpu_used": False,
    }
    if manifest != expected_manifest:
        raise VerificationError("DOP853 run manifest changed")
    ledger_rows, committed, failed = read_dop_ledger(
        output_root / "attempt_ledger.jsonl", contract
    )
    verify_dop_failures(output_root / "failures", ledger_rows)
    arms = {arm_id: expected_dop_initial(expanded, selected, arm_id) for arm_id in DOP_ARMS}
    final_checkpoints: dict[str, dict[str, Any]] = {}
    for arm_id in DOP_ARMS:
        arm = arms[arm_id]
        previous_accumulator = initial_dop_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        previous_commitments: list[dict[str, Any]] = []
        for segment in range(20):
            checkpoint, receipt = verify_dop_checkpoint(
                output_root, arm, segment, bindings, contract,
                previous_accumulator, previous_commitments,
            )
            ledger = committed.get((arm_id, segment))
            if (ledger is None
                    or ledger["checkpoint_sha256"]
                    != digest_file(dop_checkpoint_path(output_root, arm_id, segment))
                    or ledger["segment_receipt_sha256"]
                    != digest_file(dop_receipt_path(output_root, arm_id, segment))):
                raise VerificationError("DOP853 COMMITTED ledger/artifact binding changed")
            recovered = ledger.get("recovery") \
                == "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION"
            if ((recovered and (ledger["elapsed_seconds"] != 0.0
                                or ledger["terminal_peak_rss_bytes"] != 0))
                    or (not recovered and ledger["terminal_peak_rss_bytes"]
                        != receipt["parent_resource_validation"][
                            "terminal_child_peak_rss_bytes"
                        ])):
                raise VerificationError("DOP853 COMMITTED terminal provenance changed")
            previous_accumulator = checkpoint["accumulator"]
            previous_commitments = checkpoint["segment_commitments"]
            final_checkpoints[arm_id] = checkpoint
    result_path = output_root / "result_v1.json"
    result = read_object(result_path)
    exact_keys(result, {
        "schema", "experiment_id", "artifact_class", "execution_label", "input_bindings",
        "runtime", "semantic", "semantic_sha256", "resource_provenance",
        "scientific_classification_emitted", "physical_validation_claim", "mandatory_nonclaim",
    }, "DOP853 result")
    if (result["schema"] != DOP_RESULT_SCHEMA or result["experiment_id"] != EXPERIMENT_ID
            or result["artifact_class"]
            != "LOCAL_SYNTHETIC_INDEPENDENT_NUMERICAL_SENTINEL_RESULT"
            or result["execution_label"] != "DOP853-SENTINEL"
            or result["input_bindings"] != bindings or result["runtime"] != runtime
            or result["scientific_classification_emitted"] is not False
            or result["physical_validation_claim"] is not False
            or result["mandatory_nonclaim"] != contract["mandatory_nonclaim"]
            or result["semantic_sha256"] != digest_bytes(canonical(result["semantic"]))):
        raise VerificationError("DOP853 result identity/digest changed")
    by_arm: dict[str, dict[str, Any]] = {}
    active_all = True
    limits = contract["independent_sentinel"]["active_gates"]
    expected_arm_rows = []
    for arm_id in DOP_ARMS:
        arm = arms[arm_id]; checkpoint = final_checkpoints[arm_id]
        accumulator = checkpoint["accumulator"]
        expected_gate = {
            "relative_compensated_intrinsic_energy_drift": accumulator["maximum_active_drifts"][
                "relative_compensated_intrinsic_energy_drift"
            ] <= limits["max_relative_compensated_intrinsic_energy_drift"],
            "relative_com_angular_momentum_vector_drift": accumulator["maximum_active_drifts"][
                "relative_com_angular_momentum_vector_drift"
            ] <= limits["max_relative_com_angular_momentum_vector_drift"],
            "scale_normalized_linear_momentum_residual": accumulator["maximum_active_drifts"][
                "scale_normalized_linear_momentum_residual"
            ] <= limits["max_scale_normalized_linear_momentum_residual"],
        }
        landmarks = [accumulator["landmarks"][key] for key in ("250000", "500000", "1000000")]
        arm_result = {
            "arm_id": arm_id, "active_body_count": arm["active_count"],
            "sentinel_tracer_count": 32,
            "initial_state_sha256": arm["initial_state_sha256"],
            "registered_expanded_initial_state_sha256": arm[
                "registered_expanded_initial_state_sha256"
            ],
            "sample_count": accumulator["sample_count"],
            "sample_state_chain_head_sha256": accumulator["sample_state_chain_head"],
            "segment_chain_head_sha256": checkpoint["segment_chain_head_sha256"],
            "maximum_active_drifts": accumulator["maximum_active_drifts"],
            "active_gate_pass": expected_gate,
            "all_active_gates_pass": all(expected_gate.values()), "landmarks": landmarks,
        }
        expected_arm_rows.append(arm_result)
        active_all = active_all and all(expected_gate.values())
        by_arm[arm_id] = arm_result
    semantic = {
        "schema": "jx-xp2-dop853-semantic/v1", "experiment_id": EXPERIMENT_ID,
        "execution_label": "DOP853-SENTINEL", "selected_logical_ids": list(selected),
        "arms": expected_arm_rows, "all_active_gates_pass": active_all,
        "cross_method_gate_status": "PENDING_INDEPENDENT_REPLAY_VERIFIER",
    }
    if result["semantic"] != semantic:
        raise VerificationError("DOP853 result semantic differs from immutable artifacts")
    provenance = result["resource_provenance"]
    exact_keys(provenance, {
        "elapsed_seconds", "coordinator_peak_rss_bytes",
        "maximum_terminal_child_peak_rss_bytes", "output_bytes_before_result",
        "attempt_ledger_sha256", "immutable_segment_inventory_sha256",
    }, "DOP853 resource provenance")
    caps = contract["resource_caps_per_execution"]
    actual_before_result = sum(
        path.stat().st_size for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != result_path
    )
    if (not finite_number(provenance["elapsed_seconds"])
            or not 0.0 <= provenance["elapsed_seconds"] < caps["max_wall_seconds_total"]
            or type(provenance["coordinator_peak_rss_bytes"]) is not int
            or not 0 <= provenance["coordinator_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or type(provenance["maximum_terminal_child_peak_rss_bytes"]) is not int
            or not 0 <= provenance["maximum_terminal_child_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or type(provenance["output_bytes_before_result"]) is not int
            or provenance["output_bytes_before_result"] != actual_before_result
            or actual_before_result > caps["max_output_bytes"]
            or provenance["attempt_ledger_sha256"]
            != digest_file(output_root / "attempt_ledger.jsonl")
            or provenance["immutable_segment_inventory_sha256"]
            != dop_inventory_digest(output_root)):
        raise VerificationError("DOP853 resource provenance changed")
    return result, by_arm, active_all


def primary_censored(row: dict[str, Any], threshold: int, horizon: float) -> float:
    value = row[f"first_sampled_q_below_{threshold}_time_year"]
    return (horizon if value is None else value) / horizon


def cross_method_rows(
    primary_finals: dict[str, dict[str, Any]], dop_arms: dict[str, dict[str, Any]],
    selected: Sequence[str], contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    limits = contract["independent_sentinel"]["cross_method_gates_against_each_mercurius_resolution"]
    results = []
    for arm_id in DOP_ARMS:
        dop_landmarks = {str(int(row["landmark_year"])): row for row in dop_arms[arm_id]["landmarks"]}
        for resolution, primary_arm in (
            ("dt_0p125", arm_id), ("dt_0p0625", f"AUDIT-{arm_id}")
        ):
            for horizon in HORIZONS:
                key = str(int(horizon)); dop = dop_landmarks[key]
                primary = {row["logical_id"]: row
                           for row in primary_finals[primary_arm]["landmarks"][key]["particles"]}
                dop_rows = {row["logical_id"]: row for row in dop["particles"]}
                if set(dop_rows) != set(selected) or not set(selected).issubset(primary):
                    raise VerificationError("cross-method tracer identity changed")
                checks: dict[str, bool] = {}; metrics: dict[str, Any] = {
                    "arm_id": arm_id, "mercurius_resolution": resolution,
                    "horizon_years": horizon,
                }
                p_list = [primary[logical_id] for logical_id in selected]
                d_list = [dop_rows[logical_id] for logical_id in selected]
                for threshold in (30, 35, 40):
                    keyq = str(threshold)
                    pi = [indicator(row, float(threshold)) for row in p_list]
                    di = [int(row["ever_sampled_below"][keyq]) for row in d_list]
                    count = abs(sum(pi) - sum(di)); discordance = sum(
                        left != right for left, right in zip(pi, di, strict=True)
                    )
                    metrics[f"q{threshold}_event_count_absolute_difference"] = count
                    metrics[f"q{threshold}_indicator_discordance"] = discordance
                    checks[f"q{threshold}_count"] = count <= limits[
                        "max_event_count_difference_each_q30_q35_q40"
                    ]
                    checks[f"q{threshold}_discordance"] = discordance <= limits[
                        "max_indicator_discordance_each_q30_q35_q40"
                    ]
                bound = abs(sum(row["final_finite_and_bound"] for row in p_list)
                            - sum(row["bound"] for row in d_list))
                min_q = w1([row["minimum_sampled_q_AU"] for row in p_list],
                           [row["minimum_sampled_q_AU"] for row in d_list])
                final_q = w1([row["final_q_AU"] for row in p_list], [row["q_AU"] for row in d_list])
                final_i = w1([row["final_i_deg"] for row in p_list], [row["i_deg"] for row in d_list])
                first30 = w1([primary_censored(row, 30, horizon) for row in p_list],
                             [row["censored_first_below_divided_by_landmark"]["30"] for row in d_list])
                first35 = w1([primary_censored(row, 35, horizon) for row in p_list],
                             [row["censored_first_below_divided_by_landmark"]["35"] for row in d_list])
                metrics.update({
                    "bound_count_absolute_difference": bound,
                    "w1_minimum_sampled_q_AU": min_q, "w1_final_q_AU": final_q,
                    "w1_final_i_deg": final_i,
                    "w1_censored_first_q30_divided_by_horizon": first30,
                    "w1_censored_first_q35_divided_by_horizon": first35,
                })
                checks.update({
                    "bound_count": bound <= limits["max_bound_count_difference"],
                    "minimum_q": min_q <= limits["max_w1_minimum_sampled_q_AU"],
                    "final_q": final_q <= limits["max_w1_final_q_AU"],
                    "final_i": final_i <= limits["max_w1_final_i_deg"],
                    "first_q30": first30 <= limits[
                        "max_w1_censored_first_q30_divided_by_horizon"
                    ],
                    "first_q35": first35 <= limits[
                        "max_w1_censored_first_q35_divided_by_horizon"
                    ],
                })
                metrics["checks"] = checks; metrics["passes"] = all(checks.values())
                results.append(metrics)
    return results, all(row["passes"] for row in results)


def overlaps(left: Path, right: Path) -> bool:
    left = left.resolve(); right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def reject_symlink_path(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise VerificationError(f"{label} component is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise VerificationError(f"{label} contains a symlink component")
        if index < len(absolute.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise VerificationError(f"{label} ancestor is not a directory")
    return absolute


def validate_receipt_destination(path: Path, label: str) -> Path:
    """Validate an existing safe parent while allowing one absent output leaf."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name or absolute.name in {".", ".."}:
        raise VerificationError(f"{label} leaf is invalid")
    parent = reject_symlink_path(absolute.parent, f"{label} parent")
    safe_dir(parent, f"{label} parent")
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError:
        return absolute
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VerificationError(f"{label} leaf is unsafe")
    if metadata.st_nlink == 2:
        pending = absolute.with_name(f".{absolute.name}.pending")
        try:
            pending_metadata = os.lstat(pending)
        except OSError as exc:
            raise VerificationError(f"{label} leaf link count is unsafe") from exc
        if (not stat.S_ISREG(pending_metadata.st_mode)
                or pending_metadata.st_nlink != 2
                or pending_metadata.st_dev != metadata.st_dev
                or pending_metadata.st_ino != metadata.st_ino):
            raise VerificationError(f"{label} leaf link count is unsafe")
    elif metadata.st_nlink != 1:
        raise VerificationError(f"{label} leaf link count is unsafe")
    return absolute


def held_verification_tree_fingerprint(
    root: Path, lock_path: Path, label: str,
) -> str:
    lexical_root = reject_symlink_path(root, label)
    lexical_lock = reject_symlink_path(lock_path, f"{label} execution lock")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fds = [os.open(lexical_root.anchor, directory_flags)]
    directory_bindings: list[tuple[int, str, int, os.stat_result]] = []
    file_bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    listed_names: list[tuple[int, list[str]]] = []
    try:
        for component in lexical_root.parts[1:]:
            parent_fd = directory_fds[-1]
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened = os.fstat(child_fd)
            if (not stat.S_ISDIR(before.st_mode) or not stat.S_ISDIR(opened.st_mode)
                    or before.st_dev != opened.st_dev or before.st_ino != opened.st_ino):
                os.close(child_fd)
                raise VerificationError(f"{label} component binding changed")
            directory_fds.append(child_fd)
            directory_bindings.append((parent_fd, component, child_fd, opened))
        root_fd = directory_fds[-1]
        root_metadata = os.fstat(root_fd)
        lock_on_disk = os.stat("execution.lock", dir_fd=root_fd, follow_symlinks=False)
        lock_identity = (lock_on_disk.st_dev, lock_on_disk.st_ino)
        lock_fd = _VERIFICATION_LOCK_FDS.get(lock_identity)
        if lock_fd is None:
            raise VerificationError(f"{label} scan lacks its held execution lock")
        lock_metadata = os.fstat(lock_fd)
        lock_in_root = os.stat("execution.lock", dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(root_metadata.st_mode)
                or not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1
                or lock_metadata.st_size != 0
                or lock_metadata.st_dev != lock_in_root.st_dev
                or lock_metadata.st_ino != lock_in_root.st_ino):
            raise VerificationError(f"{label} root/lock binding changed")
        rows: list[list[Any]] = []

        def digest_at(directory_fd: int, name: str, before: os.stat_result) -> str:
            flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino
                    or opened.st_size != before.st_size):
                os.close(descriptor)
                raise VerificationError(f"{label} file binding changed")
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            result = digest.hexdigest()
            file_bindings.append((directory_fd, name, descriptor, opened, result))
            return result

        def scan(directory_fd: int, prefix: str) -> None:
            names = sorted(os.listdir(directory_fd))
            listed_names.append((directory_fd, names))
            for name in names:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative = f"{prefix}/{name}" if prefix else name
                if stat.S_ISLNK(before.st_mode):
                    raise VerificationError(f"{label} contains a symlink")
                if stat.S_ISDIR(before.st_mode):
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                    opened = os.fstat(child_fd)
                    if (not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != before.st_dev
                            or opened.st_ino != before.st_ino):
                        os.close(child_fd)
                        raise VerificationError(f"{label} directory binding changed")
                    directory_fds.append(child_fd)
                    directory_bindings.append((directory_fd, name, child_fd, opened))
                    rows.append([relative, "D"])
                    scan(child_fd, relative)
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    rows.append([relative, "F", before.st_size,
                                 digest_at(directory_fd, name, before)])
                else:
                    raise VerificationError(f"{label} contains a hardlink or special file")
        scan(root_fd, "")
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for directory_fd, names in listed_names:
            if sorted(os.listdir(directory_fd)) != names:
                raise VerificationError(f"{label} entries changed during scan")
        for parent_fd, name, descriptor, opened in directory_bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if any(getattr(opened, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise VerificationError(f"{label} directory changed during final binding check")
        for parent_fd, name, descriptor, opened, expected_digest in file_bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if any(getattr(opened, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise VerificationError(f"{label} file changed during final binding check")
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise VerificationError(f"{label} file content changed during final check")
        return digest_bytes(canonical(sorted(rows, key=lambda row: row[0])))
    finally:
        for _parent_fd, _name, descriptor, _opened, _digest in reversed(file_bindings):
            os.close(descriptor)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def tree_fingerprint(root: Path) -> str:
    safe_dir(root, "fingerprinted tree")
    rows: list[list[Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise VerificationError("fingerprinted tree contains a symlink")
        if path.is_dir():
            rows.append([relative, "D"])
        elif path.is_file() and path.stat().st_nlink == 1:
            rows.append([relative, "F", path.stat().st_size, digest_file(path)])
        else:
            raise VerificationError("fingerprinted tree contains an unsafe entry")
    return digest_bytes(canonical(rows))


def validate_external_paths(
    paths: Sequence[Path], contract: dict[str, Any], package_root: Path, receipt: Path
) -> None:
    protected = [package_root.resolve()] + [
        (package_root / value).resolve()
        for value in contract["xp1_historical_binding"]["protected_read_only_trees"]
    ]
    protected.extend(
        (package_root / value).resolve()
        for value in contract["xp2_v1_invalid_protocol_lineage"][
            "protected_read_only_trees"
        ]
    )
    protected.extend(
        (package_root / value).resolve()
        for value in contract["xp2_v2_invalid_replay_lineage"][
            "protected_read_only_trees"
        ]
    )
    protected.extend(
        (package_root / value).resolve()
        for value in contract["xp2_v3_failed_startup_lineage"][
            "protected_read_only_trees"
        ]
    )
    gate = contract["engineering_boundary_gate_v1"]
    protected.extend((package_root / gate[key]).resolve() for key in (
        "engineering_output_root", "engineering_verifier_scratch_root",
        "engineering_verifier_start_path", "engineering_verifier_terminal_path",
        "engineering_verification_receipt_path",
    ))
    resolved = [path.resolve() for path in paths]
    if (any(overlaps(resolved[left], resolved[right])
            for left in range(len(resolved)) for right in range(left + 1, len(resolved)))
            or any(overlaps(path, root) for path in resolved for root in protected)):
        raise VerificationError("output trees overlap each other or a protected tree")
    receipt_parent = Path(os.path.abspath(os.fspath(receipt))).parent
    if any(overlaps(receipt_parent, path) for path in (*resolved, *protected)):
        raise VerificationError("verification receipt destination overlaps an input/protected tree")


def verify_a_receipt(path: Path, result_path: Path, semantic_hash: str) -> dict[str, Any]:
    receipt = read_object(path)
    exact_keys(receipt, {
        "schema", "experiment_id", "artifact_class", "execution_label", "result_sha256",
        "semantic_sha256", "verified_for_b", "analysis_state", "official_classification",
        "checks", "mandatory_nonclaim",
    }, "A verification receipt")
    exact_keys(receipt["checks"], {
        "registered_inputs_and_exact_initial_states",
        "all_50_arms_20_parent_commits_and_hash_chains",
        "independent_metrics_effects_gates_and_labels",
        "no_scientific_classification_emitted",
    }, "A verification receipt checks")
    if (receipt["schema"] != A_RECEIPT_SCHEMA or receipt["experiment_id"] != EXPERIMENT_ID
            or receipt["artifact_class"] != "INDEPENDENT_STORED_ARTIFACT_VERIFICATION_FOR_B_GATE"
            or receipt["execution_label"] != "A" or receipt["verified_for_b"] is not True
            or receipt["result_sha256"] != digest_file(result_path)
            or receipt["semantic_sha256"] != semantic_hash
            or receipt["official_classification"] is not None
            or any(value is not True for value in receipt["checks"].values())):
        raise VerificationError("A verification receipt changed")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--initial-states", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-a", action="store_true")
    mode.add_argument("--verify-final", action="store_true")
    parser.add_argument("--a-verification-receipt", type=Path)
    parser.add_argument("--output-b", type=Path)
    parser.add_argument("--dop-output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_paths = [
        args.contract, args.seed_manifest, args.selection_manifest, args.initial_states,
        args.registration, args.output_a,
        *(value for value in (
            args.a_verification_receipt, args.output_b, args.dop_output
        ) if value is not None),
    ]
    for raw in raw_paths:
        reject_symlink_path(raw, "verifier CLI path")
    receipt_path = validate_receipt_destination(
        args.receipt, "verification receipt destination",
    )
    contract_path = args.contract.resolve(); seed_path = args.seed_manifest.resolve()
    selection_path = args.selection_manifest.resolve(); initial_path = args.initial_states.resolve()
    registration_path = args.registration.resolve(); output_a = args.output_a.resolve()
    contract, seed_manifest, selection, initial = validate_package(
        contract_path, seed_path, selection_path, initial_path, registration_path
    )
    expanded = reconstruct_states(initial, contract, seed_manifest)
    sentinel_ids = selected_ids(selection, initial)
    package_root = registration_path.parent
    package_paths = {
        "contract_v1.json": contract_path, "seed_manifest_v1.json": seed_path,
        "selection_manifest_v1.json": selection_path,
        "initial_states_v1.json": initial_path, "registration_v1.json": registration_path,
        "run_primary.py": package_root / "run_primary.py",
        "run_independent.py": package_root / "run_independent.py",
        "verify_replay.py": package_root / "verify_replay.py",
    }
    if args.verify_a:
        if any(value is not None for value in (
            args.a_verification_receipt, args.output_b, args.dop_output
        )):
            raise VerificationError("A-only verification received final-mode inputs")
        validate_external_paths([output_a], contract, package_root, receipt_path)
        start_fingerprints = {
            package_root: tree_fingerprint(package_root),
            output_a: tree_fingerprint(output_a),
        }
        result_a, _finals_a, analysis_a = verify_primary_output(
            output_a, "A", contract, initial, expanded, package_paths, None
        )
        receipt = {
            "schema": A_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
            "artifact_class": "INDEPENDENT_STORED_ARTIFACT_VERIFICATION_FOR_B_GATE",
            "execution_label": "A", "result_sha256": digest_file(output_a / "result_v1.json"),
            "semantic_sha256": result_a["semantic_sha256"], "verified_for_b": True,
            "analysis_state": analysis_a["analysis_state"], "official_classification": None,
            "checks": {
                "registered_inputs_and_exact_initial_states": True,
                "all_50_arms_20_parent_commits_and_hash_chains": True,
                "independent_metrics_effects_gates_and_labels": True,
                "no_scientific_classification_emitted": True,
            },
            "mandatory_nonclaim": contract["mandatory_nonclaim"],
        }
        if any(tree_fingerprint(path) != fingerprint
               for path, fingerprint in start_fingerprints.items()):
            raise VerificationError("an input tree changed during A verification")
        revalidate_final_engineering_evidence()
        publish(receipt_path, receipt)
        revalidate_final_engineering_evidence()
        return 0

    if args.output_b is None or args.dop_output is None or args.a_verification_receipt is None:
        raise VerificationError("final verification requires B, DOP853, and verified-A receipt")
    output_b = args.output_b.resolve(); dop_output = args.dop_output.resolve()
    a_receipt_path = args.a_verification_receipt.resolve()
    validate_external_paths(
        [output_a, output_b, dop_output, a_receipt_path],
        contract, package_root, receipt_path,
    )
    start_fingerprints = {
        package_root: tree_fingerprint(package_root), output_a: tree_fingerprint(output_a),
        output_b: tree_fingerprint(output_b), dop_output: tree_fingerprint(dop_output),
    }
    a_receipt_start_hash = digest_file(a_receipt_path)
    result_a, finals_a, analysis_a = verify_primary_output(
        output_a, "A", contract, initial, expanded, package_paths, None
    )
    a_receipt = verify_a_receipt(
        a_receipt_path, output_a / "result_v1.json", result_a["semantic_sha256"]
    )
    if (a_receipt["analysis_state"] != analysis_a["analysis_state"]
            or a_receipt["mandatory_nonclaim"] != contract["mandatory_nonclaim"]):
        raise VerificationError("A verification receipt analysis binding changed")
    b_prerequisite = {
        "a_result_sha256": digest_file(output_a / "result_v1.json"),
        "a_verification_receipt_sha256": digest_file(a_receipt_path),
    }
    result_b, _finals_b, analysis_b = verify_primary_output(
        output_b, "B", contract, initial, expanded, package_paths, b_prerequisite
    )
    dop_result, dop_arms, active_pass = verify_dop_output(
        dop_output, contract, sentinel_ids, expanded, package_paths
    )
    cross_rows, cross_pass = cross_method_rows(finals_a, dop_arms, sentinel_ids, contract)
    replay_equal = (result_a["semantic_sha256"] == result_b["semantic_sha256"]
                    and result_a["semantic"] == result_b["semantic"]
                    and analysis_a == analysis_b)
    label: str | None = None
    if not replay_equal:
        state = "NONDETERMINISTIC_REPLAY"
    elif analysis_a["analysis_state"] != "PRIMARY_NUMERICS_COMPLETE_AWAITING_REPLAY_AND_DOP853":
        state = analysis_a["analysis_state"]
    elif not active_pass or not cross_pass:
        state = "INTEGRATOR_SENSITIVE"
    else:
        state = "VERIFIED_COMPLETE"
        label = analysis_a["primary_screen_label"]
    receipt = {
        "schema": FINAL_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "artifact_class": "FINAL_INDEPENDENT_STORED_ARTIFACT_VERIFICATION",
        "verification_state": state, "official_classification": label,
        "claim_ceiling": contract["claim_ceiling"],
        "primary_A_result_sha256": digest_file(output_a / "result_v1.json"),
        "primary_B_result_sha256": digest_file(output_b / "result_v1.json"),
        "primary_semantic_sha256": result_a["semantic_sha256"] if replay_equal else None,
        "dop853_result_sha256": digest_file(dop_output / "result_v1.json"),
        "dop853_semantic_sha256": dop_result["semantic_sha256"],
        "exact_primary_replay_equal": replay_equal,
        "dop853_active_gates_pass": active_pass,
        "cross_method_gates_pass": cross_pass,
        "cross_method_gate_rows": cross_rows,
        "scientific_classification_emitted": label is not None,
        "independent_numerical_implementation_not_independent_physics": True,
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }
    if (digest_file(a_receipt_path) != a_receipt_start_hash
            or any(tree_fingerprint(path) != fingerprint
                   for path, fingerprint in start_fingerprints.items())):
        raise VerificationError("an input/result tree changed during final verification")
    revalidate_final_engineering_evidence()
    publish(receipt_path, receipt)
    revalidate_final_engineering_evidence()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"verification failed: {type(error).__name__}", file=__import__("sys").stderr)
        raise SystemExit(2)
    finally:
        release_verification_locks()
