#!/usr/bin/env python3
"""Frozen independent SciPy DOP853 sentinel for JX-XP2.

This module intentionally contains no REBOUND import or call.  It consumes the
same preregistered binary64 barycentric Cartesian rows as the primary method and
integrates a deterministic 32-tracer sentinel with a custom Newtonian RHS.
"""

from __future__ import annotations

import argparse
import decimal
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import select
import shutil
import signal
import stat
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v4"
SCIENTIFIC_DESIGN_EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v1"
CONTRACT_SCHEMA = "jx-xp2-robustness-contract/v3"
INITIAL_STATES_SCHEMA = "jx-xp2-barycentric-initial-states/v1"
SELECTION_SCHEMA = "jx-xp2-dop853-selection-manifest/v1"
REGISTRATION_SCHEMA = "jx-xp2-local-registration/v1"
ENGINEERING_REGISTRATION_SCHEMA = "jx-xp2-v4-engineering-registration/v1"
ENGINEERING_RECEIPT_SCHEMA = "jx-xp2-v4-engineering-boundary-verification/v1"
RUN_MANIFEST_SCHEMA = "jx-xp2-dop853-run-manifest/v1"
CHECKPOINT_SCHEMA = "jx-xp2-dop853-segment-checkpoint/v1"
SEGMENT_RECEIPT_SCHEMA = "jx-xp2-dop853-segment-parent-receipt/v1"
RESULT_SCHEMA = "jx-xp2-dop853-result/v1"
ATTEMPT_SCHEMA = "jx-xp2-dop853-attempt-ledger-row/v2"
FAILURE_SCHEMA = "jx-xp2-dop853-failure/v2"
PRIMARY_SEGMENT_SEMANTIC_FIELD_ORDER = (
    "arm_id", "configuration_id", "arm_class", "dt_years", "segment_index",
    "start_years", "end_years", "first_sample_index", "last_sample_index",
    "new_sample_count", "sample_count_total", "sampled_state_stream_sha256",
    "decoded_integrator_state_sha256", "tracker", "initial_active_invariants",
    "maximum_active_invariant_drifts", "landmarks",
)
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
V2_DEFECT_EVIDENCE_SHA256 = "7cd515610718eaa9fac3159f988ef924c6df030cc8828719818b5b461789ff47"
V2_DEFECT_EVIDENCE_SIZE_BYTES = 5626
V3_FAILED_STARTUP_EVIDENCE_SHA256 = "eeb5ed87e05aab1ac0fa3cad68391bae1c850090dc48d4113b8b71c58c1dd473"
V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES = 4064
_V3_A_GUARD_FD: int | None = None
_ENGINEERING_RUNNER_GUARD_FD: int | None = None
_ENGINEERING_SCRATCH_GUARD_FD: int | None = None
_FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS: list[Any] = []
MAX_REBOUND_ALLOCATION_CAPACITY = 4096
ENDPOINT_DIGEST_DOMAIN = b"jx-xp2-mercurius-live-archive-endpoint/v1\0"

OFFICIAL_EXECUTION_LABEL = "DOP853-SENTINEL"
ARM_IDS = (
    "M0",
    "CI01-P0", "CI01-P4",
    "CI05-P1", "CI05-P5",
    "CI09-P2", "CI09-P6",
)
THRESHOLDS_AU = (30.0, 35.0, 40.0)
LANDMARK_YEARS = (250_000.0, 500_000.0, 1_000_000.0)
SENTINEL_DOMAIN = b"jx-xp2-dop853-sentinel/v1\0"
CHAIN_DOMAIN = b"jx-xp2-dop853-segment-chain/v1\0"
SEGMENT_PAYLOAD_DOMAIN = b"jx-xp2-dop853-segment-payload/v1\0"
INITIAL_CHAIN = hashlib.sha256(CHAIN_DOMAIN + b"GENESIS").hexdigest()
NANOSECONDS_PER_SECOND = 1_000_000_000
FRAME_HEADER = struct.Struct(">Q")
REDACTED_FAILURE_MESSAGE = "REDACTED_NON_SEMANTIC_FAILURE_DETAIL"
DOP_FAILURE_EVENT_DOMAIN = b"jx-xp2-dop853-failure-event/v2\0"
DOP_FAILURE_CLASSES = {
    "InterruptedAttempt", "IntegrityError", "NumericalError",
    "ResourceLimitError", "UnexpectedFailure",
}
NATIVE_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# This happens before the intentionally lazy NumPy/SciPy import.  It both avoids
# oversubscription and makes a POSIX child boundary safe and reproducible.
for _thread_key, _thread_value in NATIVE_THREAD_ENVIRONMENT.items():
    os.environ[_thread_key] = _thread_value


class IntegrityError(RuntimeError):
    """A frozen input or generated artifact failed an integrity condition."""


class ResourceLimitError(RuntimeError):
    """A predeclared local resource limit was reached."""


class NumericalError(RuntimeError):
    """The independent integration did not produce its required finite state."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def serialized_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_symlink_components(path: Path, label: str) -> Path:
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


def held_tree_inventory(root: Path, lock_fd: int, label: str) -> list[list[Any]]:
    lexical_root = reject_symlink_components(root, label)
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
                raise IntegrityError(f"{label} component binding changed")
            directory_fds.append(child_fd)
            directory_bindings.append((parent_fd, component, child_fd, opened))
        root_fd = directory_fds[-1]
        root_metadata = os.fstat(root_fd)
        lock_metadata = os.fstat(lock_fd)
        lock_on_disk = os.stat("execution.lock", dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(root_metadata.st_mode)
                or not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1
                or lock_metadata.st_size != 0
                or lock_metadata.st_dev != lock_on_disk.st_dev
                or lock_metadata.st_ino != lock_on_disk.st_ino):
            raise IntegrityError(f"{label} root/lock binding changed")
        rows: list[list[Any]] = []

        def digest_at(directory_fd: int, name: str, before: os.stat_result) -> str:
            flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino
                    or opened.st_size != before.st_size):
                os.close(descriptor)
                raise IntegrityError(f"{label} file binding changed")
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
                    raise IntegrityError(f"{label} contains a symlink")
                if stat.S_ISDIR(before.st_mode):
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                    opened = os.fstat(child_fd)
                    if (not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != before.st_dev
                            or opened.st_ino != before.st_ino):
                        os.close(child_fd)
                        raise IntegrityError(f"{label} directory binding changed")
                    directory_fds.append(child_fd)
                    directory_bindings.append((directory_fd, name, child_fd, opened))
                    rows.append([relative, "D"])
                    scan(child_fd, relative)
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    rows.append([relative, "F", before.st_size,
                                 digest_at(directory_fd, name, before)])
                else:
                    raise IntegrityError(f"{label} contains a hardlink or special file")
        scan(root_fd, "")
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        for directory_fd, names in listed_names:
            if sorted(os.listdir(directory_fd)) != names:
                raise IntegrityError(f"{label} entries changed during scan")
        for parent_fd, name, descriptor, opened in directory_bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if any(getattr(opened, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise IntegrityError(f"{label} directory changed during final binding check")
        for parent_fd, name, descriptor, opened, expected_digest in file_bindings:
            after = os.fstat(descriptor)
            on_disk = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if any(getattr(opened, key) != getattr(after, key) for key in stable) \
                    or any(getattr(after, key) != getattr(on_disk, key) for key in stable):
                raise IntegrityError(f"{label} file changed during final binding check")
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise IntegrityError(f"{label} file content changed during final check")
        return sorted(rows, key=lambda row: row[0])
    finally:
        for _parent_fd, _name, descriptor, _opened, _digest in reversed(file_bindings):
            os.close(descriptor)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    exact = decimal.Decimal(value)
    if not math.isfinite(parsed) or not exact.is_finite() or (
        parsed == 0.0 and exact != 0
    ):
        raise ValueError("non-finite or underflowed JSON number")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("JSON input must be a singly-linked regular non-symlink file")
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def strict_json_bytes(payload: bytes, label: str = "held JSON") -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_float=_finite_float, parse_constant=_reject_constant,
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



def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


def atomic_create_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite an artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise FileExistsError("stale pending artifact exists")
    payload = serialized_json(value)
    try:
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if pending.stat().st_size != len(payload) or sha256_file(pending) != sha256_bytes(payload):
            raise IntegrityError("pending JSON hash or size verification failed")
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_replace_checkpoint(path: Path, value: Any) -> None:
    """Atomically replace only a runner-owned regular checkpoint."""
    if path.is_symlink() or (path.exists() and (
        not path.is_file() or path.stat().st_nlink != 1
    )):
        raise IntegrityError("checkpoint target is not a safe regular file")
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise IntegrityError("stale pending checkpoint exists")
    payload = serialized_json(value)
    try:
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if pending.stat().st_size != len(payload) or sha256_file(pending) != sha256_bytes(payload):
            raise IntegrityError("pending checkpoint hash or size verification failed")
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def peak_rss_bytes_self() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def child_peak_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def proc_thread_count() -> int:
    return len(list(Path("/proc/self/task").iterdir()))


def directory_bytes(root: Path) -> int:
    """Account a live tree without ever following a replaced path component."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise IntegrityError("output byte root is missing or unsafe") from exc
    root_metadata = os.fstat(root_fd)
    if not stat.S_ISDIR(root_metadata.st_mode):
        os.close(root_fd)
        raise IntegrityError("output byte root is not a directory")
    def scan(directory_fd: int) -> int:
        subtotal = 0
        entries = {entry.name: entry for entry in os.scandir(directory_fd)}
        original_kinds: dict[str, tuple[str, int | None, int | None]] = {}
        for name in sorted(entries):
            entry = entries[name]
            hinted_symlink = entry.is_symlink()
            hinted_directory = entry.is_dir(follow_symlinks=False)
            hinted_file = entry.is_file(follow_symlinks=False)
            if hinted_symlink:
                raise IntegrityError("output byte scan encountered a symlink")
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                if hinted_file:
                    original_kinds[name] = ("file", None, None)
                    continue
                raise IntegrityError(
                    "output directory or unknown entry disappeared during byte scan"
                ) from exc
            if stat.S_ISREG(metadata.st_mode):
                if not hinted_file:
                    raise IntegrityError("output directory changed to a regular file")
                if metadata.st_nlink != 1:
                    raise IntegrityError("output byte scan encountered a hard link")
                original_kinds[name] = ("file", metadata.st_dev, metadata.st_ino)
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise IntegrityError(
                        "output directory disappeared or changed type during byte scan"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if (not hinted_directory or not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != metadata.st_dev
                            or opened.st_ino != metadata.st_ino):
                        raise IntegrityError("output child directory binding changed")
                    child_total = scan(child_fd)
                    try:
                        after_child = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                    except FileNotFoundError as exc:
                        raise IntegrityError(
                            "output directory disappeared during byte scan"
                        ) from exc
                    if (not stat.S_ISDIR(after_child.st_mode)
                            or after_child.st_dev != opened.st_dev
                            or after_child.st_ino != opened.st_ino):
                        raise IntegrityError("output directory was replaced during scan")
                    original_kinds[name] = ("directory", opened.st_dev, opened.st_ino)
                    subtotal += child_total
                finally:
                    os.close(child_fd)
            else:
                raise IntegrityError("output byte scan encountered a special file")
        final_names = set(os.listdir(directory_fd))
        for original_name, (kind, device, inode) in original_kinds.items():
            if original_name not in final_names:
                if kind == "directory":
                    raise IntegrityError("output directory disappeared during final scan")
                continue
            try:
                final_entry = os.stat(
                    original_name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError as exc:
                if kind == "file":
                    continue
                raise IntegrityError(
                    "output directory disappeared during final scan"
                ) from exc
            if kind == "directory":
                if (not stat.S_ISDIR(final_entry.st_mode)
                        or final_entry.st_dev != device or final_entry.st_ino != inode):
                    raise IntegrityError("output directory changed after traversal")
            else:
                if (not stat.S_ISREG(final_entry.st_mode) or final_entry.st_nlink != 1):
                    raise IntegrityError("output file changed to an unsafe final type")
                subtotal += final_entry.st_size
        for added_name in sorted(final_names - set(entries)):
            try:
                added = os.stat(
                    added_name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(added.st_mode):
                raise IntegrityError("output directory was added during byte scan")
            if not stat.S_ISREG(added.st_mode) or added.st_nlink != 1:
                raise IntegrityError("output byte scan encountered an unsafe new entry")
            subtotal += added.st_size
        return subtotal

    try:
        total = scan(root_fd)
        try:
            final_root = os.stat(root, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise IntegrityError("output byte root disappeared during scan") from exc
        if (not stat.S_ISDIR(final_root.st_mode)
                or final_root.st_dev != root_metadata.st_dev
                or final_root.st_ino != root_metadata.st_ino):
            raise IntegrityError("output byte root changed during scan")
        return total
    finally:
        os.close(root_fd)


def is_lower_hex(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise IntegrityError(f"{label} shape changed")


def binary64_from_hex(value: Any, label: str = "binary64") -> float:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} must be a binary64 hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise IntegrityError(f"{label} is not a binary64 hex string") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise IntegrityError(f"{label} is noncanonical or nonfinite")
    return result


def float_hex(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise NumericalError("nonfinite binary64 value")
    return value.hex()


def trees_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_output_root(
    raw_output: Path, package_root: Path, contract: dict[str, Any], resume: bool
) -> Path:
    if raw_output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if raw_output.parent.is_symlink():
        raise ValueError("output directory parent must not be a symlink")
    output = raw_output.resolve()
    protected = [package_root.resolve()]
    historical = contract["xp1_historical_binding"]
    if historical.get("output_receipt_or_checkpoint_tree_overlap") != "FORBIDDEN":
        raise IntegrityError("historical output-overlap policy changed")
    for relative in historical.get("protected_read_only_trees", []):
        candidate = package_root / relative
        if candidate.is_symlink() or not candidate.resolve().is_dir():
            raise IntegrityError("protected historical tree is missing or unsafe")
        protected.append(candidate.resolve())
    v1_lineage = contract["xp2_v1_invalid_protocol_lineage"]
    if v1_lineage.get("output_receipt_or_checkpoint_tree_overlap") != "FORBIDDEN":
        raise IntegrityError("XP2-v1 output-overlap policy changed")
    for relative in v1_lineage.get("protected_read_only_trees", []):
        candidate = package_root / relative
        if candidate.is_symlink() or not candidate.resolve().is_dir():
            raise IntegrityError("protected XP2-v1 tree is missing or unsafe")
        protected.append(candidate.resolve())
    v2_lineage = contract["xp2_v2_invalid_replay_lineage"]
    if v2_lineage.get("output_receipt_or_checkpoint_tree_overlap") != "FORBIDDEN":
        raise IntegrityError("XP2-v2 output-overlap policy changed")
    for relative in v2_lineage.get("protected_read_only_trees", []):
        candidate = package_root / relative
        if candidate.is_symlink() or not candidate.resolve().is_dir():
            raise IntegrityError("protected XP2-v2 tree is missing or unsafe")
        protected.append(candidate.resolve())
    v3_lineage = contract["xp2_v3_failed_startup_lineage"]
    if v3_lineage.get("output_receipt_or_checkpoint_tree_overlap") != "FORBIDDEN":
        raise IntegrityError("XP2-v3 output-overlap policy changed")
    for relative in v3_lineage.get("protected_read_only_trees", []):
        candidate = package_root / relative
        if candidate.is_symlink() or not candidate.resolve().is_dir():
            raise IntegrityError("protected XP2-v3 tree is missing or unsafe")
        protected.append(candidate.resolve())
    gate = contract["engineering_boundary_gate_v1"]
    protected.extend((package_root / gate[key]).resolve() for key in (
        "engineering_output_root", "engineering_verifier_scratch_root",
        "engineering_verifier_start_path", "engineering_verifier_terminal_path",
        "engineering_verification_receipt_path",
    ))
    if any(trees_overlap(output, root) for root in protected):
        raise ValueError("output directory overlaps a protected package or historical tree")
    if resume:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("resume output must be an existing regular directory")
    else:
        if output.exists():
            raise FileExistsError("new output directory must be absent")
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise ValueError("output parent must be an existing non-symlink directory")
    return output


def _vector_norm3(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in values))


def _cross3(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def orbital_metrics(
    tracer_position: Sequence[float],
    tracer_velocity: Sequence[float],
    sun_position: Sequence[float],
    sun_velocity: Sequence[float],
    gravitational_constant: float,
    sun_mass: float,
) -> dict[str, Any]:
    """Return finite Sun-relative osculating metrics for a massless tracer."""
    position = tuple(tracer_position[index] - sun_position[index] for index in range(3))
    velocity = tuple(tracer_velocity[index] - sun_velocity[index] for index in range(3))
    radius = _vector_norm3(position)
    speed2 = math.fsum(component * component for component in velocity)
    mu = gravitational_constant * sun_mass
    if not (radius > 0.0 and mu > 0.0 and math.isfinite(speed2)):
        raise NumericalError("invalid Sun-relative state")
    angular = _cross3(position, velocity)
    angular_norm = _vector_norm3(angular)
    if not angular_norm > 0.0:
        raise NumericalError("degenerate osculating angular momentum")
    velocity_cross_angular = _cross3(velocity, angular)
    eccentricity_vector = tuple(
        velocity_cross_angular[index] / mu - position[index] / radius
        for index in range(3)
    )
    eccentricity = _vector_norm3(eccentricity_vector)
    denominator = mu * (1.0 + eccentricity)
    perihelion = angular_norm * angular_norm / denominator
    specific_energy = 0.5 * speed2 - mu / radius
    semimajor = math.inf if specific_energy == 0.0 else -mu / (2.0 * specific_energy)
    cosine_i = max(-1.0, min(1.0, angular[2] / angular_norm))
    inclination_deg = math.degrees(math.acos(cosine_i))
    finite = all(math.isfinite(item) for item in (
        radius, eccentricity, perihelion, inclination_deg, specific_energy, semimajor
    ))
    if not finite or perihelion < 0.0:
        raise NumericalError("nonfinite osculating metric")
    bound = specific_energy < 0.0 and eccentricity < 1.0
    return {
        "a_AU": semimajor,
        "e": eccentricity,
        "i_deg": inclination_deg,
        "q_AU": perihelion,
        "distance_AU": radius,
        "finite": True,
        "bound": bound,
    }


def active_invariants(
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    masses: Sequence[float],
    gravitational_constant: float,
) -> dict[str, Any]:
    """Compensated intrinsic invariants of active bodies only."""
    if len(positions) != len(velocities) or len(positions) != len(masses):
        raise ValueError("active invariant array lengths differ")
    total_mass = math.fsum(masses)
    if not total_mass > 0.0:
        raise NumericalError("active system has no positive mass")
    momentum = tuple(
        math.fsum(mass * velocities[index][axis] for index, mass in enumerate(masses))
        for axis in range(3)
    )
    center_position = tuple(
        math.fsum(mass * positions[index][axis] for index, mass in enumerate(masses))
        / total_mass
        for axis in range(3)
    )
    center_velocity = tuple(component / total_mass for component in momentum)
    kinetic_terms: list[float] = []
    angular_terms: list[list[float]] = [[], [], []]
    momentum_scale_terms: list[float] = []
    for index, mass in enumerate(masses):
        relative_position = tuple(
            positions[index][axis] - center_position[axis] for axis in range(3)
        )
        relative_velocity = tuple(
            velocities[index][axis] - center_velocity[axis] for axis in range(3)
        )
        kinetic_terms.append(
            0.5 * mass * math.fsum(value * value for value in relative_velocity)
        )
        contribution = _cross3(relative_position, relative_velocity)
        for axis in range(3):
            angular_terms[axis].append(mass * contribution[axis])
        momentum_scale_terms.append(mass * _vector_norm3(relative_velocity))
    potential_terms: list[float] = []
    for left in range(len(masses)):
        for right in range(left + 1, len(masses)):
            separation = _vector_norm3(tuple(
                positions[right][axis] - positions[left][axis] for axis in range(3)
            ))
            if not separation > 0.0:
                raise NumericalError("active-body collision or duplicate position")
            potential_terms.append(
                -gravitational_constant * masses[left] * masses[right] / separation
            )
    result = {
        "intrinsic_energy": math.fsum(kinetic_terms + potential_terms),
        "com_angular_momentum": tuple(math.fsum(axis) for axis in angular_terms),
        "linear_momentum": momentum,
        "linear_momentum_scale": math.fsum(momentum_scale_terms),
    }
    flat = [
        result["intrinsic_energy"], result["linear_momentum_scale"],
        *result["com_angular_momentum"], *result["linear_momentum"],
    ]
    if not all(math.isfinite(item) for item in flat):
        raise NumericalError("nonfinite active invariant")
    return result


def invariant_drifts(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    energy_denominator = abs(float(baseline["intrinsic_energy"]))
    angular_denominator = _vector_norm3(baseline["com_angular_momentum"])
    momentum_denominator = float(baseline["linear_momentum_scale"])
    if not (energy_denominator > 0.0 and angular_denominator > 0.0 and momentum_denominator > 0.0):
        raise NumericalError("invalid invariant normalization")
    return {
        "relative_compensated_intrinsic_energy_drift": abs(
            current["intrinsic_energy"] - baseline["intrinsic_energy"]
        ) / energy_denominator,
        "relative_com_angular_momentum_vector_drift": _vector_norm3(tuple(
            current["com_angular_momentum"][axis]
            - baseline["com_angular_momentum"][axis]
            for axis in range(3)
        )) / angular_denominator,
        "scale_normalized_linear_momentum_residual": _vector_norm3(tuple(
            current["linear_momentum"][axis] - baseline["linear_momentum"][axis]
            for axis in range(3)
        )) / momentum_denominator,
    }


def newtonian_rhs_factory(
    masses: Sequence[float], active_count: int, gravitational_constant: float
) -> Callable[[float, Any], Any]:
    """Build the independent point-mass RHS; tracers exert no force."""
    numpy, _solve_ivp = scipy_runtime()
    if not (1 <= active_count <= len(masses)):
        raise ValueError("invalid active-body count")
    if any(mass <= 0.0 for mass in masses[:active_count]) or any(
        mass != 0.0 for mass in masses[active_count:]
    ):
        raise ValueError("active/massless particle ordering or masses changed")

    def rhs(_time: float, state: Any) -> Any:
        particle_count = len(masses)
        matrix = numpy.asarray(state, dtype=numpy.float64).reshape(2, particle_count, 3)
        positions = matrix[0]
        velocities = matrix[1]
        derivatives = numpy.empty_like(matrix)
        derivatives[0] = velocities
        accelerations = derivatives[1]
        active_positions = positions[:active_count]
        active_masses = numpy.asarray(masses[:active_count], dtype=numpy.float64)
        # Shape: target particle x active source x Cartesian component.  This is
        # still the explicit custom Newtonian law, but avoids a Python loop in the
        # million-step RHS hot path.
        displacement = active_positions[numpy.newaxis, :, :] - positions[:, numpy.newaxis, :]
        distance_squared = numpy.einsum("tsa,tsa->ts", displacement, displacement)
        active_diagonal = numpy.arange(active_count)
        distance_squared[active_diagonal, active_diagonal] = numpy.inf
        if numpy.any(distance_squared <= 0.0) or not numpy.all(
            numpy.isfinite(distance_squared) | numpy.isinf(distance_squared)
        ):
            raise NumericalError("singular or nonfinite Newtonian separation")
        inverse_cube = distance_squared ** -1.5
        accelerations[:] = gravitational_constant * numpy.einsum(
            "tsa,ts,s->ta", displacement, inverse_cube, active_masses,
            optimize=False,
        )
        if not numpy.all(numpy.isfinite(derivatives)):
            raise NumericalError("nonfinite Newtonian derivative")
        return derivatives.reshape(-1)

    return rhs


_SCIPY_CACHE: tuple[Any, Any] | None = None


def scipy_runtime() -> tuple[Any, Any]:
    """Lazily load only NumPy and SciPy's solve_ivp implementation."""
    global _SCIPY_CACHE
    if "rebound" in sys.modules:
        raise IntegrityError("REBOUND must not be loaded by the independent runner")
    if _SCIPY_CACHE is None:
        numpy = importlib.import_module("numpy")
        integrate = importlib.import_module("scipy.integrate")
        if proc_thread_count() != 1:
            raise IntegrityError("NumPy/SciPy native thread lock did not hold")
        _SCIPY_CACHE = numpy, integrate.solve_ivp
    return _SCIPY_CACHE


def validate_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    numpy, _solve_ivp = scipy_runtime()
    scipy = importlib.import_module("scipy")
    multiarray = importlib.import_module("numpy._core._multiarray_umath")
    rk = importlib.import_module("scipy.integrate._ivp.rk")
    coefficients = importlib.import_module("scipy.integrate._ivp.dop853_coefficients")
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "numpy_version": numpy.__version__,
        "numpy_multiarray_binary_sha256": sha256_file(Path(multiarray.__file__).resolve()),
        "scipy_version": scipy.__version__,
        "scipy_rk_source_sha256": sha256_file(Path(rk.__file__).resolve()),
        "scipy_dop853_coefficients_sha256": sha256_file(
            Path(coefficients.__file__).resolve()
        ),
        "native_thread_environment": {
            key: os.environ.get(key) for key in NATIVE_THREAD_ENVIRONMENT
        },
    }
    locked = contract["runtime_lock"]
    expected = {key: locked[key] for key in actual}
    if actual != expected:
        raise IntegrityError("independent Python/NumPy/SciPy runtime lock mismatch")
    if "rebound" in sys.modules:
        raise IntegrityError("independent runtime loaded REBOUND")
    return actual


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
        raise IntegrityError("XP2-v2 invalid-replay lineage declaration changed")
    evidence_path = package_root / expected["defect_evidence_path"]
    if (evidence_path.is_symlink() or not evidence_path.is_file()
            or evidence_path.stat().st_nlink != 1
            or evidence_path.stat().st_size != V2_DEFECT_EVIDENCE_SIZE_BYTES
            or sha256_file(evidence_path) != V2_DEFECT_EVIDENCE_SHA256):
        raise IntegrityError("XP2-v2 replay-defect evidence changed")
    evidence = strict_json(evidence_path)
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
        or evidence.get("comparison", {}).get("v3_expected_segment_chain_head_both_runs")
        != "5cc01d89db885889ae8dc0c8ed2cc1de2f36969d941f9da849709e40063133bf"
    ):
        raise IntegrityError("XP2-v2 replay-defect diagnosis changed")
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
            raise IntegrityError("XP2-v2 defect-evidence binding shape changed")
        bound = package_root / binding["path"]
        if (bound.is_symlink() or not bound.is_file() or bound.stat().st_nlink != 1
                or bound.stat().st_size != binding["size_bytes"]
                or sha256_file(bound) != binding["sha256"]):
            raise IntegrityError("XP2-v2 defect-evidence external artifact changed")
    lock_path = package_root / expected["v2_b_execution_lock_path"]
    if (lock_path.is_symlink() or not lock_path.is_file()
            or lock_path.stat().st_nlink != 1 or lock_path.stat().st_size != 0):
        raise IntegrityError("XP2-v2 B execution lock is unsafe")
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
        raise IntegrityError("XP2-v3 failed-startup lineage declaration changed")
    evidence_path = package_root / expected["evidence_path"]
    if (evidence_path.is_symlink() or not evidence_path.is_file()
            or evidence_path.stat().st_nlink != 1
            or evidence_path.stat().st_size != V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES
            or sha256_file(evidence_path) != V3_FAILED_STARTUP_EVIDENCE_SHA256):
        raise IntegrityError("XP2-v3 failed-startup evidence changed")
    evidence = strict_json(evidence_path)
    if (evidence.get("schema") != "jx-xp2-v3-failed-startup-evidence/v1"
            or evidence.get("experiment_id") != EXPERIMENT_ID
            or evidence.get("role")
            != "IMMUTABLE_V3_PROTOCOL_FAILURE_EVIDENCE_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT"
            or evidence.get("ledger_counts")
            != {"START": 12, "PASS": 0, "FAIL": 9, "OPEN": 3}
            or evidence.get("absence_assertions") != {
                "segment_PASS_rows": 0, "segment_commits": 0, "checkpoints": 0,
                "primary_results": 0, "verification_receipts": 0,
            }
            or evidence.get("v3_output_policy")
            != "FAILED_CLOSED_STARTUP_DIAGNOSTIC_ONLY; NO_PROMOTION_NO_RESUME_NO_SCIENTIFIC_CONSUMPTION"
            or evidence.get("v3_scientific_outcomes_gates_or_classification_consumed") is not False
            or evidence.get("protected_read_only_trees") != expected["protected_read_only_trees"]):
        raise IntegrityError("XP2-v3 failed-startup diagnosis changed")
    bindings = [
        evidence["v3_registration"], evidence["v3_run_manifest"],
        evidence["v3_attempt_ledger"], evidence["v3_execution_lock"],
        *evidence["failure_receipts"],
    ]
    for binding in bindings:
        if set(binding) != {"path", "size_bytes", "sha256"}:
            raise IntegrityError("XP2-v3 evidence binding shape changed")
        bound = package_root / binding["path"]
        if (bound.is_symlink() or not bound.is_file() or bound.stat().st_nlink != 1
                or bound.stat().st_size != binding["size_bytes"]
                or sha256_file(bound) != binding["sha256"]):
            raise IntegrityError("XP2-v3 evidence external artifact changed")
    output_root = package_root / evidence["tree_fingerprint"]["root_path"]
    if _V3_A_GUARD_FD is None:
        raise IntegrityError("XP2-v3 failed tree scan lacks its held execution guard")
    entries = held_tree_inventory(
        output_root, _V3_A_GUARD_FD, "XP2-v3 failed output tree"
    )
    tree = evidence.get("tree_fingerprint", {})
    fingerprint = sha256_bytes(canonical_bytes(entries))
    if (tree != {
            "root_path": "../jx_xp2_runs_v3/output_a",
            "schema": "jx-xp2-v3-verifier-tree-fingerprint/v1",
            "digest_formula": "SHA256_CANONICAL_JSON_OF_ROWS",
            "root_entry_included": False, "entry_count": 22,
            "entry_order": "RELATIVE_POSIX_PATH_ASCENDING",
            "directory_row": ["relative_posix_path", "D"],
            "file_row": ["relative_posix_path", "F", "size_bytes", "sha256"],
            "symlinks_hardlinks_and_special_files_allowed": False,
            "sha256": fingerprint,
        } or len(entries) != 22):
        raise IntegrityError("XP2-v3 failed-startup tree fingerprint changed")
    ledger_path = package_root / evidence["v3_attempt_ledger"]["path"]
    rows: list[dict[str, Any]] = []
    for line in ledger_path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise IntegrityError("XP2-v3 ledger framing changed")
        row = json.loads(line)
        if canonical_bytes(row) + b"\n" != line:
            raise IntegrityError("XP2-v3 ledger canonical bytes changed")
        rows.append(row)
    starts = {(row["arm_id"], row["segment_index"], row["attempt_index"])
              for row in rows if row.get("event") == "START"}
    terminals = {(row["arm_id"], row["segment_index"], row["attempt_index"])
                 for row in rows if row.get("event") in {"PASS", "FAIL"}}
    if (len(rows) != 21 or len(starts) != 12 or len(terminals) != 9
            or any(row.get("event") == "PASS" for row in rows)
            or starts - terminals != {
                ("CI01-P0", 0, 3), ("CI01-P1", 0, 3), ("CI01-P2", 0, 3),
            }):
        raise IntegrityError("XP2-v3 failed-startup ledger state changed")
    start_rows = {
        (row["arm_id"], row["segment_index"], row["attempt_index"]): row
        for row in rows if row.get("event") == "START"
    }
    fail_rows = [row for row in rows if row.get("event") == "FAIL"]
    if ([row.get("sequence") for row in rows] != list(range(1, 22))
            or any(row.get("schema") != "jx-xp2-mercurius-segment-attempt/v3"
                   or row.get("execution_label") != "A" for row in rows)):
        raise IntegrityError("XP2-v3 failed-startup ledger identity changed")
    receipt_bindings = {
        Path(binding["path"]).name: binding for binding in evidence["failure_receipts"]
    }
    if len(receipt_bindings) != 9:
        raise IntegrityError("XP2-v3 failure-receipt binding cardinality changed")
    for row in fail_rows:
        key = (row["arm_id"], row["segment_index"], row["attempt_index"])
        name = row.get("failure_receipt_filename")
        if (key not in start_rows or row.get("failure_class") != "CHILD_EXIT_NONZERO"
                or row.get("return_code") != 2 or name not in receipt_bindings
                or row.get("failure_receipt_sha256") != receipt_bindings[name]["sha256"]):
            raise IntegrityError("XP2-v3 FAIL row is not bound to one startup receipt")
        receipt = strict_json(package_root / receipt_bindings[name]["path"])
        if (receipt.get("schema") != "jx-xp2-primary-failure/v3"
                or receipt.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v3"
                or receipt.get("execution_label") != "A"
                or receipt.get("arm_id") != key[0] or receipt.get("segment_index") != key[1]
                or receipt.get("attempt_index") != key[2]
                or receipt.get("start_sequence") != start_rows[key]["sequence"]
                or receipt.get("failure_class") != "CHILD_EXIT_NONZERO"
                or receipt.get("return_code") != 2
                or receipt.get("fail_event_sha256") != row.get("fail_event_sha256")
                or receipt.get("input_key_sha256") != start_rows[key].get("input_key_sha256")
                or receipt.get("complete_uncommitted_attempt") is not None
                or receipt.get("authorizes_analysis") is not False):
            raise IntegrityError("XP2-v3 failure receipt does not match its FAIL/START")
    manifest = strict_json(package_root / evidence["v3_run_manifest"]["path"])
    if (manifest.get("schema") != "jx-xp2-primary-run-manifest/v2"
            or manifest.get("experiment_id") != "jx-xp2-public-synthetic-robustness-v3"
            or manifest.get("execution_label") != "A"
            or manifest.get("registration_sha256")
            != evidence["v3_registration"]["sha256"]):
        raise IntegrityError("XP2-v3 failed-startup run manifest binding changed")
    return evidence


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("experiment_id") != EXPERIMENT_ID:
        raise IntegrityError("contract identity changed")
    permissions = contract.get("permissions", {})
    if permissions != {
        "local_cpu_execution_authorized": True,
        "network_access_authorized": False,
        "gpu_execution_authorized": False,
        "observed_data_access_authorized": False,
        "survey_adapter_execution_authorized": False,
        "jx_o2_execution_or_g0_evidence_authorized": False,
        "planet_x_detection_exclusion_constraint_or_preference_claim_authorized": False,
    }:
        raise IntegrityError("contract permission boundary changed")
    core = contract["design_core"]
    dynamics = core["dynamics"]
    if (
        dynamics["duration_years"] != 1_000_000.0
        or dynamics["sample_cadence_years"] != 50.0
        or dynamics["sample_count_including_t0"] != 20_001
        or dynamics["segment_years"] != 50_000.0
        or dynamics["segment_count"] != 20
        or tuple(dynamics["landmark_years"]) != LANDMARK_YEARS
    ):
        raise IntegrityError("independent horizon or output grid changed")
    if core["tracer_design"]["block_count"] != 8 or core["tracer_design"]["tracers_per_block"] != 16:
        raise IntegrityError("tracer block design changed")
    sentinel = contract["independent_sentinel"]
    solver = sentinel["solver"]
    expected_solver = {
        "method": "scipy.integrate.DOP853",
        "rtol": 1e-11,
        "component_atol_position_AU": 1e-12,
        "component_atol_velocity_AU_per_year": 1e-12,
        "first_step_years": 0.125,
        "max_step_years": 1.0,
        "sample_cadence_years": 50.0,
        "segment_years": 50_000.0,
        "fixed_restart_at_every_segment_boundary_is_part_of_method": True,
    }
    if (
        sentinel["implementation"]
        != "SEPARATE_SCIPY_DOP853_NEWTONIAN_FORCE_AND_METRIC_PATH_WITH_NO_REBOUND_IMPORT"
        or sentinel["official_execution_count"] != 1
        or sentinel["tracers"] != 32
        or tuple(sentinel["arm_ids"]) != ARM_IDS
        or solver != expected_solver
    ):
        raise IntegrityError("independent integrator design changed")
    active_gates = sentinel["active_gates"]
    if active_gates != {
        "max_relative_compensated_intrinsic_energy_drift": 1e-8,
        "max_relative_com_angular_momentum_vector_drift": 1e-10,
        "max_scale_normalized_linear_momentum_residual": 1e-10,
    }:
        raise IntegrityError("independent active-invariant gates changed")
    cross = sentinel["cross_method_gates_against_each_mercurius_resolution"]
    if cross != {
        "landmarks_year": [250_000.0, 500_000.0, 1_000_000.0],
        "max_event_count_difference_each_q30_q35_q40": 1,
        "max_indicator_discordance_each_q30_q35_q40": 2,
        "max_bound_count_difference": 1,
        "max_w1_minimum_sampled_q_AU": 1.0,
        "max_w1_final_q_AU": 2.0,
        "max_w1_final_i_deg": 1.0,
        "max_w1_censored_first_q30_divided_by_horizon": 0.05,
        "max_w1_censored_first_q35_divided_by_horizon": 0.05,
    }:
        raise IntegrityError("cross-method numerical gates changed")
    checkpoint = contract["checkpoint_and_resume"]
    if (
        contract.get("registration_status")
        != "ENGINEERING_BOUNDARY_PASS_AND_FINAL_LOCAL_HASH_LOCK_REQUIRED_BEFORE_FIRST_OFFICIAL_V4_SCIENTIFIC_OUTPUT"
        or contract.get("v4_fresh_execution_and_operational_repair")
        != expected_v4_fresh_repair_declaration()
        or contract.get("engineering_boundary_gate_v1")
        != expected_engineering_boundary_gate_v1()
        or checkpoint["segment_years"] != 50_000.0
        or checkpoint["segments_per_arm"] != 20
        or checkpoint["expected_total_samples"] != 20_001
        or checkpoint["dop853_storage"]
        != "IMMUTABLE_FULL_BOUNDARY_CARTESIAN_STATE_AND_ACCUMULATOR_FOR_EACH_OF_20_SEGMENTS_PER_ARM_WITH_FIXED_SOLVER_RESTART_BY_DESIGN"
        or checkpoint["hash_chain_required"] is not True
        or checkpoint["maximum_attempts_per_segment"] != 3
        or checkpoint["sample_boundary_ownership"]
        != "SEGMENT_0_OWNS_T0_EACH_LATER_SEGMENT_OMITS_DUPLICATE_START_SAMPLE"
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
        or checkpoint.get("max_primary_checkpoint_bytes") != 1_048_576
        or checkpoint.get("primary_failure_receipt_v4")
        != "DETERMINISTIC_ATTEMPT_KEY_FILENAME; RECEIPT_FSYNCED_BEFORE_FAIL_LEDGER_PUBLICATION; STABLE_START_BOUND_EVENT_DIGEST; EXACT_CLASS_RETURN_CODE_AND_QUARANTINE_BINDING; EVERY_FULLY_PUBLISHED_ATTEMPT_ACROSS_SOURCE_PENDING_SPLIT_OR_QUARANTINE_CRASH_CUTS_BINDS_CANONICAL_DECODED_AND_SAMPLED_SEMANTIC_EVIDENCE_IN_RECEIPT_AND_FAIL; LATER_PASS_MUST_MATCH_EVERY_COMPLETE_FAILED_ATTEMPT_WHILE_RAW_ARCHIVE_BYTES_MAY_DIFFER; TORN_UNPUBLISHED_FAILURE_RECEIPT_BYTES_ARE_PRESERVED_AND_HASH_BOUND_IN_RECOVERED_QUARANTINE; RECEIPT_WITHOUT_FAIL_RECONCILED_ON_RESUME; FAIL_WITHOUT_RECEIPT_OR_ANY_EXTRA_REJECTED"
        or checkpoint.get("retry_identical_inputs_decoded_state_scientific_payload_and_seed_only")
        is not True
        or checkpoint.get("retry_raw_archive_byte_equality_required") is not False
        or checkpoint.get("dop853_failure_receipt_v2")
        != "DETERMINISTIC_ARM_SEGMENT_ATTEMPT_FILENAME; RECEIPT_FSYNCED_BEFORE_SEGMENT_ATTEMPT_FAILED_LEDGER_PUBLICATION; START_SEQUENCE_BOUND_EVENT_DIGEST; EXACT_CLOSED_FAILURE_TO_RECEIPT_FILENAME_SHA256_AND_CLASS_BIJECTION; COMPLETE_ORPHAN_RECEIPT_RECONCILED; PARTIAL_UNPUBLISHED_RECEIPT_BECOMES_INTERRUPTED_ATTEMPT; MISSING_EXTRA_DUPLICATE_OR_TAMPERED_RECEIPT_REJECTED"
        or checkpoint.get("dop853_failure_classes_v2") != [
            "InterruptedAttempt", "IntegrityError", "NumericalError",
            "ResourceLimitError", "UnexpectedFailure",
        ]
    ):
        raise IntegrityError("checkpoint design changed")
    caps = contract["resource_caps_per_execution"]
    if caps != {
        "workers": 4,
        "max_wall_seconds_total": 86400.0,
        "max_wall_seconds_per_segment_attempt": 14400.0,
        "max_peak_rss_bytes_per_process": 1073741824,
        "max_aggregate_child_rss_bytes": 4294967296,
        "max_output_bytes": 2147483648,
        "minimum_free_disk_bytes": 21474836480,
        "gpu_used": False,
        "native_dynamics_watchdog": "PARENT_MONOTONIC_DEADLINES_WITH_ONE_FRESH_SEPARATE_PROCESS_GROUP_PER_ARM_SEGMENT_ATTEMPT_AND_SIGKILL_ON_CAP_OR_FAILURE",
        "watchdog_poll_seconds": 0.25,
    }:
        raise IntegrityError("execution resource or watchdog lock changed")
    state_policy = contract["initial_state_policy"]
    if (
        state_policy["path"] != "initial_states_v1.json"
        or state_policy["schema"] != INITIAL_STATES_SCHEMA
        or state_policy["shared_input_rule"]
        != "PRIMARY_MERCURIUS_AND_INDEPENDENT_DOP853_MUST_CONSUME_THE_SAME_REGISTERED_EXPANDED_BINARY64_BARYCENTRIC_CARTESIAN_ROWS"
        or state_policy["runner_reconstruction_or_tolerance_substitution"] != "FORBIDDEN"
        or state_policy["post_registration_regeneration"] != "FORBIDDEN"
    ):
        raise IntegrityError("shared initial-state policy changed")
    if core["units_and_frame"]["G_AU3_Msun_yr2"] != 39.47841760435743:
        raise IntegrityError("gravitational constant changed")
    if contract["result_policy"]["dop853_official_execution_label"] != OFFICIAL_EXECUTION_LABEL:
        raise IntegrityError("independent execution label changed")
    if (
        contract["result_policy"]["no_raw_sampled_trajectories"] is not True
        or contract["result_policy"]["output_overwrite"] != "FORBIDDEN"
        or contract["result_policy"].get("dop853_failure_receipt_bijection_required")
        is not True
        or contract["result_policy"].get("dop853_attempt_ledger_schema")
        != ATTEMPT_SCHEMA
        or contract["result_policy"].get("dop853_failure_receipt_schema")
        != FAILURE_SCHEMA
        or contract["result_policy"].get(
            "primary_semantic_payload_excludes_raw_rebound_archive_filename_size_and_sha256"
        ) is not True
        or contract["result_policy"].get(
            "primary_raw_rebound_archive_integrity_validation_required"
        ) is not True
        or contract["result_policy"].get("verification_receipt_publication_v4")
        != "HELD_INPUT_AND_LINEAGE_LOCKS_THROUGH_FSYNCED_ATOMIC_PUBLICATION; EXACT_FINAL_IS_IDEMPOTENT; COMPLETE_PENDING_IS_PROMOTED; EXACT_PAYLOAD_PREFIX_IS_DISCARDED_AND_REBUILT; DIVERGENT_PENDING_OR_FINAL_FAILS_CLOSED"
        or contract["result_policy"].get("primary_raw_artifact_integrity_v1")
        != expected_raw_artifact_declaration_v1()
        or contract["result_policy"].get("primary_failure_receipt_schema")
        != "jx-xp2-primary-failure/v4"
        or contract["result_policy"].get("primary_attempt_ledger_schema")
        != "jx-xp2-mercurius-segment-attempt/v4"
        or contract["result_policy"].get("primary_checkpoint_receipt_schema")
        != "jx-xp2-mercurius-segment-receipt/v3"
        or contract["result_policy"].get("primary_segment_commit_schema")
        != "jx-xp2-mercurius-segment-parent-commit/v3"
        or contract["result_policy"].get("primary_result_schema")
        != "jx-xp2-primary-result/v3"
        or contract["result_policy"].get("primary_semantic_schema")
        != "jx-xp2-primary-semantic/v3"
    ):
        raise IntegrityError("independent output policy changed")


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


def validate_registration(
    registration_path: Path,
    contract_path: Path,
    seed_manifest_path: Path,
    initial_states_path: Path,
    selection_path: Path,
    runner_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    registration = strict_json(registration_path)
    if (
        registration.get("schema") != REGISTRATION_SCHEMA
        or registration.get("experiment_id") != EXPERIMENT_ID
        or registration.get("outcomes_generated") is not False
        or registration.get("scientific_evidence_artifact") is not False
    ):
        raise IntegrityError("registration identity or preoutput status changed")
    locked = registration.get("locked_files")
    if not isinstance(locked, dict) or not all(
        isinstance(name, str) and is_lower_hex(digest) for name, digest in locked.items()
    ):
        raise IntegrityError("registration locked-files map is invalid")
    package_root = registration_path.resolve().parent
    inventory = set(strict_json(contract_path)["result_policy"]["registered_package_inventory"])
    if inventory != set(locked) | {"registration_v1.json"}:
        raise IntegrityError("registration and contract inventories differ")
    actual_inventory = {candidate.name for candidate in package_root.iterdir()}
    if actual_inventory != inventory:
        raise IntegrityError("registered package inventory has extras or omissions")
    for name, expected_digest in locked.items():
        candidate = package_root / name
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_nlink != 1:
            raise IntegrityError("registered file is not a safe regular file")
        if sha256_file(candidate) != expected_digest:
            raise IntegrityError("registered file byte hash mismatch")
    expected_paths = {
        "contract_v1.json": contract_path,
        "seed_manifest_v1.json": seed_manifest_path,
        "initial_states_v1.json": initial_states_path,
        "selection_manifest_v1.json": selection_path,
        "run_independent.py": runner_path,
    }
    for name, supplied in expected_paths.items():
        if supplied.resolve() != package_root / name:
            raise IntegrityError("noncanonical registered input path")
    validate_final_engineering_authorization(
        registration, strict_json(contract_path), package_root,
    )
    return registration, locked


def validate_selection(
    selection: dict[str, Any], all_tracer_ids: Sequence[str]
) -> tuple[str, ...]:
    required_keys = {
        "schema", "experiment_id", "artifact_class", "tracer_rows_sha256",
        "selection_domain_ascii_without_terminator", "selection_domain_terminator_hex",
        "rank_formula", "selection_rule", "selection_core_canonicalization",
        "sentinels_by_block", "ordered_logical_ids", "selection_core_sha256",
        "selected_tracer_count", "selected_arm_ids",
        "outcome_or_prior_trajectory_used", "override_allowed", "mandatory_nonclaim",
    }
    require_exact_keys(selection, required_keys, "selection manifest")
    if (
        selection["schema"] != SELECTION_SCHEMA
        or selection["experiment_id"] != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or selection["selection_domain_ascii_without_terminator"]
        != "jx-xp2-dop853-sentinel/v1"
        or selection["selection_domain_terminator_hex"] != "00"
        or selection["rank_formula"]
        != "SHA256(ASCII_SELECTION_DOMAIN || BYTE_00 || ASCII_LOGICAL_ID)"
        or selection["selection_rule"]
        != "WITHIN_EACH_BLOCK_RANK_ALL_16_LOGICAL_IDS_BY_RANK_DIGEST_ASCENDING_THEN_LOGICAL_ID_ASCENDING_SELECT_FIRST_4"
        or selection["selected_tracer_count"] != 32
        or tuple(selection["selected_arm_ids"]) != ARM_IDS
        or selection["outcome_or_prior_trajectory_used"] is not False
        or selection["override_allowed"] is not False
    ):
        raise IntegrityError("selection manifest policy changed")
    ids = list(all_tracer_ids)
    if len(ids) != 128 or len(set(ids)) != 128:
        raise IntegrityError("initial-state artifact must contain 128 unique tracer IDs")
    expected_order: list[str] = []
    blocks = selection["sentinels_by_block"]
    if not isinstance(blocks, dict) or set(blocks) != {str(index) for index in range(8)}:
        raise IntegrityError("selection block map changed")
    for block_index in range(8):
        prefix = f"XP2-B{block_index:02d}-T"
        block_ids = [logical_id for logical_id in ids if logical_id.startswith(prefix)]
        if len(block_ids) != 16:
            raise IntegrityError("tracer logical IDs do not encode eight blocks of sixteen")
        ranked = sorted(
            (sha256_bytes(SENTINEL_DOMAIN + logical_id.encode("ascii")), logical_id)
            for logical_id in block_ids
        )[:4]
        row = blocks[str(block_index)]
        require_exact_keys(row, {"ordered_logical_ids", "rank_sha256"}, "selection block")
        if row["ordered_logical_ids"] != [logical_id for _digest, logical_id in ranked]:
            raise IntegrityError("sentinel IDs are not the locked four-lowest hashes")
        if row["rank_sha256"] != [digest for digest, _logical_id in ranked]:
            raise IntegrityError("sentinel rank digests changed")
        expected_order.extend(row["ordered_logical_ids"])
    if selection["ordered_logical_ids"] != expected_order:
        raise IntegrityError("global sentinel order changed")
    core = {
        "selection_domain_ascii_without_terminator": selection[
            "selection_domain_ascii_without_terminator"
        ],
        "selection_domain_terminator_hex": selection["selection_domain_terminator_hex"],
        "rank_formula": selection["rank_formula"],
        "selection_rule": selection["selection_rule"],
        "sentinels_by_block": selection["sentinels_by_block"],
        "ordered_logical_ids": expected_order,
    }
    if sha256_bytes(canonical_bytes(core)) != selection["selection_core_sha256"]:
        raise IntegrityError("selection-core digest mismatch")
    return tuple(expected_order)


def unpack_binary64_be(value: Any, label: str) -> float:
    if not isinstance(value, str) or len(value) != 16 or not is_lower_hex(value, 16):
        raise IntegrityError(f"{label} is not one packed binary64 word")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise IntegrityError(f"{label} is nonfinite")
    return result


def pack_binary64_be(value: float) -> str:
    if not math.isfinite(value):
        raise IntegrityError("cannot pack a nonfinite initial-state value")
    return struct.pack(">d", float(value)).hex()


def unpack_vector_be(value: Any, components: int, label: str) -> list[float]:
    if not isinstance(value, str) or len(value) != components * 16 or not is_lower_hex(
        value, components * 16
    ):
        raise IntegrityError(f"{label} has invalid packed length or encoding")
    return [
        unpack_binary64_be(value[index * 16 : (index + 1) * 16], label)
        for index in range(components)
    ]


def parse_factorized_row(row: Any, expected_role: str | None = None) -> dict[str, Any]:
    if not isinstance(row, list) or len(row) != 4:
        raise IntegrityError("factorized state row shape changed")
    logical_id, role, mass_hex, state_hex = row
    if (
        not isinstance(logical_id, str)
        or not logical_id.isascii()
        or role not in {"A", "T"}
        or (expected_role is not None and role != expected_role)
    ):
        raise IntegrityError("factorized state identity or role changed")
    mass = unpack_binary64_be(mass_hex, "particle mass")
    state = unpack_vector_be(state_hex, 6, "particle Cartesian state")
    if (role == "A" and mass <= 0.0) or (role == "T" and mass != 0.0):
        raise IntegrityError("active or tracer mass rule changed")
    return {
        "logical_id": logical_id,
        "role": role,
        "mass_hex": mass_hex,
        "mass": mass,
        "state": state,
    }


def validate_and_expand_initial_states(
    initial_states: dict[str, Any],
    initial_states_path: Path,
    contract: dict[str, Any],
    selected_ids: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    required_keys = {
        "schema", "experiment_id", "artifact_class", "canonicalization",
        "representation", "expansion_rule", "row_order", "row_tuple_fields",
        "state_component_order", "common_active_sun_centered_rows",
        "tracer_sun_centered_rows", "configuration_tuple_fields",
        "configuration_states", "expanded_digest", "configuration_digest_index_sha256",
        "configuration_digest_index",
        "independent_element_to_cartesian_recomputation_required",
        "dynamics_or_outcomes_generated", "mandatory_nonclaim",
    }
    require_exact_keys(initial_states, required_keys, "initial-state artifact")
    policy = contract["initial_state_policy"]
    if (
        initial_states["schema"] != INITIAL_STATES_SCHEMA
        or initial_states["experiment_id"] != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or initial_states["artifact_class"]
        != "PREOUTPUT_EXACT_BINARY64_INITIAL_STATE_FACTORIZATION"
        or initial_states["representation"] != policy["representation"]
        or initial_states["row_order"] != policy["row_order"]
        or sha256_file(initial_states_path) != policy["artifact_sha256"]
        or initial_states["dynamics_or_outcomes_generated"] is not False
        or initial_states["independent_element_to_cartesian_recomputation_required"] is not True
    ):
        raise IntegrityError("initial-state artifact identity or policy changed")
    if initial_states["row_tuple_fields"] != [
        "logical_id", "role_A_OR_T", "mass_binary64_be_hex", "state_6x_binary64_be_hex"
    ] or initial_states["state_component_order"] != [
        "x_AU", "y_AU", "z_AU", "vx_AU_per_year", "vy_AU_per_year", "vz_AU_per_year"
    ] or initial_states["configuration_tuple_fields"] != [
        "arm_id", "active_count", "added_body_sun_centered_row_or_null",
        "active_com_position_3x_binary64_be_hex",
        "active_com_velocity_3x_binary64_be_hex",
        "expanded_barycentric_rows_sha256",
    ]:
        raise IntegrityError("initial-state row layouts changed")
    common = [
        parse_factorized_row(row, "A")
        for row in initial_states["common_active_sun_centered_rows"]
    ]
    tracers = [
        parse_factorized_row(row, "T")
        for row in initial_states["tracer_sun_centered_rows"]
    ]
    if [row["logical_id"] for row in common] != [
        "Sun", "Jupiter", "Saturn", "Uranus", "Neptune"
    ]:
        raise IntegrityError("common active-body row order changed")
    all_tracer_ids = tuple(row["logical_id"] for row in tracers)
    if len(all_tracer_ids) != 128 or len(set(all_tracer_ids)) != 128:
        raise IntegrityError("initial-state tracer rows changed cardinality or identity")
    selected_set = set(all_tracer_ids if selected_ids is None else selected_ids)
    if not selected_set.issubset(all_tracer_ids):
        raise IntegrityError("selection contains a tracer absent from initial states")
    configurations = initial_states["configuration_states"]
    if not isinstance(configurations, list) or len(configurations) != 25:
        raise IntegrityError("initial-state configuration count changed")
    if [row[0] for row in configurations] != contract["design_core"]["primary_arm_ids"]:
        raise IntegrityError("initial-state configuration order changed")
    expanded_domain = (
        policy["expanded_digest_domain_ascii_without_terminator"].encode("ascii")
        + bytes.fromhex(policy["expanded_digest_domain_terminator_hex"])
    )
    arms: dict[str, dict[str, Any]] = {}
    digest_index: list[list[str]] = []
    for configuration in configurations:
        if not isinstance(configuration, list) or len(configuration) != 6:
            raise IntegrityError("initial-state configuration tuple shape changed")
        arm_id, active_count, added_raw, com_position_hex, com_velocity_hex, expected_digest = configuration
        if not isinstance(active_count, int) or active_count not in {5, 6}:
            raise IntegrityError("configuration active count changed")
        if arm_id == "M0":
            if added_raw is not None or active_count != 5:
                raise IntegrityError("M0 active-system factorization changed")
            active_rows = list(common)
        else:
            added = parse_factorized_row(added_raw, "A")
            if added["logical_id"] != f"XP2-{arm_id}" or active_count != 6:
                raise IntegrityError("M1 added-body row changed identity")
            active_rows = list(common) + [added]
        com_position = unpack_vector_be(com_position_hex, 3, "active COM position")
        com_velocity = unpack_vector_be(com_velocity_hex, 3, "active COM velocity")
        expanded_rows: list[list[str]] = []
        decoded: list[dict[str, Any]] = []
        for source in active_rows + tracers:
            translated = [
                source["state"][axis] - com_position[axis] for axis in range(3)
            ] + [
                source["state"][axis + 3] - com_velocity[axis] for axis in range(3)
            ]
            packed_state = "".join(pack_binary64_be(value) for value in translated)
            expanded_rows.append([
                source["logical_id"], source["role"], source["mass_hex"], packed_state
            ])
            decoded.append({
                "logical_id": source["logical_id"],
                "role": source["role"],
                "mass": source["mass"],
                "state": translated,
            })
        actual_digest = sha256_bytes(expanded_domain + canonical_bytes(expanded_rows))
        if not is_lower_hex(expected_digest) or actual_digest != expected_digest:
            raise IntegrityError("expanded barycentric initial-state digest mismatch")
        digest_index.append([arm_id, actual_digest])
        if arm_id not in ARM_IDS:
            continue
        selected_rows = decoded[:active_count] + [
            row for row in decoded[active_count:] if row["logical_id"] in selected_set
        ]
        selected_order = [row["logical_id"] for row in selected_rows[active_count:]]
        expected_selected_order = [
            logical_id for logical_id in (selected_ids or all_tracer_ids)
            if logical_id in selected_set
        ]
        by_id = {row["logical_id"]: row for row in selected_rows[active_count:]}
        selected_rows = selected_rows[:active_count] + [by_id[item] for item in expected_selected_order]
        if selected_order != expected_selected_order:
            # Reordering to the registered selection is intentional, but duplicates or
            # omissions are not.
            if set(selected_order) != set(expected_selected_order):
                raise IntegrityError("expanded state omitted a selected tracer")
        logical_ids = [row["logical_id"] for row in selected_rows]
        masses = [row["mass"] for row in selected_rows]
        positions = [row["state"][:3] for row in selected_rows]
        velocities = [row["state"][3:] for row in selected_rows]
        flat_state = [component for row in positions for component in row] + [
            component for row in velocities for component in row
        ]
        selected_canonical = [
            [
                row["logical_id"], row["role"], pack_binary64_be(row["mass"]),
                "".join(pack_binary64_be(value) for value in row["state"]),
            ]
            for row in selected_rows
        ]
        arms[arm_id] = {
            "arm_id": arm_id,
            "active_count": active_count,
            "sun_index": 0,
            "logical_ids": logical_ids,
            "masses": masses,
            "initial_state": flat_state,
            "tracer_block_indices": [int(item[5:7]) for item in logical_ids[active_count:]],
            "initial_state_sha256": sha256_bytes(
                b"jx-xp2-dop853-selected-initial-state/v1\0"
                + canonical_bytes(selected_canonical)
            ),
            "registered_expanded_initial_state_sha256": actual_digest,
        }
    if set(arms) != set(ARM_IDS):
        raise IntegrityError("independent arm initial states are incomplete")
    index_digest = sha256_bytes(
        b"jx-xp2-configuration-digest-index/v1\0" + canonical_bytes(digest_index)
    )
    if (
        initial_states["configuration_digest_index"]
        != "SHA256(ASCII_jx-xp2-configuration-digest-index/v1 || BYTE_00 || CANONICAL_LIST_OF_ARM_ID_AND_EXPANDED_DIGEST_PAIRS_IN_CONFIGURATION_ORDER)"
        or initial_states["configuration_digest_index_sha256"] != index_digest
    ):
        raise IntegrityError("configuration digest index is invalid")
    return arms, all_tracer_ids



def state_to_hex(state: Sequence[float]) -> list[str]:
    return [float_hex(value) for value in state]


def state_from_hex(values: Any, expected_length: int) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_length:
        raise IntegrityError("checkpoint state vector shape changed")
    return [binary64_from_hex(value, "checkpoint state component") for value in values]


def state_sample_chain(previous: str, sample_year: float, state: Sequence[float]) -> str:
    if not is_lower_hex(previous):
        raise IntegrityError("sample-state chain head is invalid")
    payload = bytearray(CHAIN_DOMAIN)
    payload.extend(b"SAMPLE")
    payload.extend(bytes.fromhex(previous))
    payload.extend(struct.pack(">dQ", float(sample_year), len(state)))
    payload.extend(struct.pack(f">{len(state)}d", *(float(value) for value in state)))
    return sha256_bytes(bytes(payload))


def matrix_from_state(state: Sequence[float], particle_count: int) -> tuple[Any, Any]:
    numpy, _solve_ivp = scipy_runtime()
    vector = numpy.asarray(state, dtype=numpy.float64)
    if vector.shape != (particle_count * 6,) or not numpy.all(numpy.isfinite(vector)):
        raise NumericalError("Cartesian state vector is nonfinite or has the wrong shape")
    matrix = vector.reshape(2, particle_count, 3)
    return matrix[0], matrix[1]


def _sample_metrics(
    state: Sequence[float],
    arm: dict[str, Any],
    gravitational_constant: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    particle_count = len(arm["logical_ids"])
    positions, velocities = matrix_from_state(state, particle_count)
    active_count = arm["active_count"]
    invariants = active_invariants(
        positions[:active_count].tolist(),
        velocities[:active_count].tolist(),
        arm["masses"][:active_count],
        gravitational_constant,
    )
    sun_index = arm["sun_index"]
    sun_mass = arm["masses"][sun_index]
    particles: list[dict[str, Any]] = []
    for index in range(active_count, particle_count):
        particles.append(orbital_metrics(
            positions[index], velocities[index], positions[sun_index], velocities[sun_index],
            gravitational_constant, sun_mass,
        ))
    return particles, invariants


def initialize_accumulator(
    arm: dict[str, Any], gravitational_constant: float
) -> dict[str, Any]:
    state = arm["initial_state"]
    metrics, baseline = _sample_metrics(state, arm, gravitational_constant)
    tracer_ids = arm["logical_ids"][arm["active_count"]:]
    rows: dict[str, Any] = {}
    for logical_id, block_index, metric in zip(
        tracer_ids, arm["tracer_block_indices"], metrics, strict=True
    ):
        first = {
            str(int(threshold)): 0.0 if metric["q_AU"] < threshold else None
            for threshold in THRESHOLDS_AU
        }
        rows[logical_id] = {
            "block_index": block_index,
            "minimum_sampled_q_AU": metric["q_AU"],
            "first_sampled_below_year": first,
            "current": metric,
        }
    accumulator = {
        "sample_count": 1,
        "sample_state_chain_head": state_sample_chain(INITIAL_CHAIN, 0.0, state),
        "baseline_invariants": baseline,
        "maximum_active_drifts": {
            "relative_compensated_intrinsic_energy_drift": 0.0,
            "relative_com_angular_momentum_vector_drift": 0.0,
            "scale_normalized_linear_momentum_residual": 0.0,
        },
        "particles": rows,
        "landmarks": {},
    }
    return accumulator


def _landmark_key(year: float) -> str:
    if not year.is_integer():
        raise ValueError("landmark year must be integral")
    return str(int(year))


def landmark_snapshot(accumulator: dict[str, Any], landmark_year: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts = {str(int(threshold)): 0 for threshold in THRESHOLDS_AU}
    bound_count = 0
    for logical_id, particle in accumulator["particles"].items():
        current = particle["current"]
        first = particle["first_sampled_below_year"]
        indicators: dict[str, bool] = {}
        censored: dict[str, float] = {}
        for threshold in THRESHOLDS_AU:
            key = str(int(threshold))
            indicators[key] = first[key] is not None
            if indicators[key]:
                counts[key] += 1
            crossing = landmark_year if first[key] is None else float(first[key])
            censored[key] = crossing / landmark_year
        if current["bound"]:
            bound_count += 1
        rows.append({
            "logical_id": logical_id,
            "block_index": particle["block_index"],
            "minimum_sampled_q_AU": particle["minimum_sampled_q_AU"],
            "q_AU": current["q_AU"],
            "a_AU": current["a_AU"],
            "e": current["e"],
            "i_deg": current["i_deg"],
            "distance_AU": current["distance_AU"],
            "finite": current["finite"],
            "bound": current["bound"],
            "ever_sampled_below": indicators,
            "first_sampled_below_year": dict(first),
            "censored_first_below_divided_by_landmark": censored,
        })
    rows.sort(key=lambda row: row["logical_id"])
    return {
        "landmark_year": landmark_year,
        "event_counts": counts,
        "bound_count": bound_count,
        "particles": rows,
    }


def accumulate_sample(
    accumulator: dict[str, Any],
    arm: dict[str, Any],
    sample_year: float,
    state: Sequence[float],
    gravitational_constant: float,
) -> None:
    metrics, current_invariants = _sample_metrics(state, arm, gravitational_constant)
    drift = invariant_drifts(current_invariants, accumulator["baseline_invariants"])
    for key, value in drift.items():
        if not math.isfinite(value):
            raise NumericalError("nonfinite invariant drift")
        accumulator["maximum_active_drifts"][key] = max(
            accumulator["maximum_active_drifts"][key], value
        )
    tracer_ids = arm["logical_ids"][arm["active_count"]:]
    for logical_id, metric in zip(tracer_ids, metrics, strict=True):
        particle = accumulator["particles"][logical_id]
        particle["minimum_sampled_q_AU"] = min(
            particle["minimum_sampled_q_AU"], metric["q_AU"]
        )
        for threshold in THRESHOLDS_AU:
            key = str(int(threshold))
            if particle["first_sampled_below_year"][key] is None and metric["q_AU"] < threshold:
                particle["first_sampled_below_year"][key] = sample_year
        particle["current"] = metric
    accumulator["sample_count"] += 1
    accumulator["sample_state_chain_head"] = state_sample_chain(
        accumulator["sample_state_chain_head"], sample_year, state
    )
    if sample_year in LANDMARK_YEARS:
        key = _landmark_key(sample_year)
        if key in accumulator["landmarks"]:
            raise IntegrityError("landmark sample was processed twice")
        accumulator["landmarks"][key] = landmark_snapshot(accumulator, sample_year)


def integrate_segment(
    arm: dict[str, Any],
    initial_state: Sequence[float],
    accumulator: dict[str, Any],
    segment_index: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Run one fixed 50-kyr DOP853 segment and return no raw trajectory."""
    numpy, solve_ivp = scipy_runtime()
    sentinel = contract["independent_sentinel"]
    solver = sentinel["solver"]
    dynamics = contract["design_core"]["dynamics"]
    segment_years = float(dynamics["segment_years"])
    cadence = float(dynamics["sample_cadence_years"])
    start_year = segment_index * segment_years
    end_year = (segment_index + 1) * segment_years
    expected_samples = int(segment_years / cadence) + 1
    output_times = start_year + numpy.arange(expected_samples, dtype=numpy.float64) * cadence
    output_times[0] = start_year
    output_times[-1] = end_year
    particle_count = len(arm["logical_ids"])
    vector = numpy.asarray(initial_state, dtype=numpy.float64)
    if vector.shape != (particle_count * 6,) or not numpy.all(numpy.isfinite(vector)):
        raise NumericalError("segment initial state is invalid")
    atol = numpy.empty_like(vector)
    atol[: particle_count * 3] = solver["component_atol_position_AU"]
    atol[particle_count * 3 :] = solver["component_atol_velocity_AU_per_year"]
    rhs = newtonian_rhs_factory(
        arm["masses"], arm["active_count"],
        contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"],
    )
    solution = solve_ivp(
        rhs,
        (start_year, end_year),
        vector,
        method="DOP853",
        t_eval=output_times,
        rtol=solver["rtol"],
        atol=atol,
        first_step=solver["first_step_years"],
        max_step=solver["max_step_years"],
        dense_output=False,
        vectorized=False,
    )
    if (
        not solution.success
        or solution.t.shape != (expected_samples,)
        or solution.y.shape != (particle_count * 6, expected_samples)
        or not numpy.array_equal(solution.t, output_times)
        or not numpy.all(numpy.isfinite(solution.y))
    ):
        raise NumericalError("DOP853 did not return the exact finite output grid")
    # The checkpoint start was already accumulated by segment 0 initialization or
    # the preceding segment, so every segment omits output column zero.
    gravitational_constant = contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
    for column in range(1, expected_samples):
        accumulate_sample(
            accumulator,
            arm,
            float(output_times[column]),
            solution.y[:, column],
            gravitational_constant,
        )
    end_state = solution.y[:, -1].tolist()
    del solution
    return {
        "segment_index": segment_index,
        "start_year": start_year,
        "end_year": end_year,
        "end_state_hex": state_to_hex(end_state),
        "accumulator": accumulator,
    }


def check_active_gates(accumulator: dict[str, Any], contract: dict[str, Any]) -> dict[str, bool]:
    drift = accumulator["maximum_active_drifts"]
    gates = contract["independent_sentinel"]["active_gates"]
    result = {
        "relative_compensated_intrinsic_energy_drift": drift[
            "relative_compensated_intrinsic_energy_drift"
        ] <= gates["max_relative_compensated_intrinsic_energy_drift"],
        "relative_com_angular_momentum_vector_drift": drift[
            "relative_com_angular_momentum_vector_drift"
        ] <= gates["max_relative_com_angular_momentum_vector_drift"],
        "scale_normalized_linear_momentum_residual": drift[
            "scale_normalized_linear_momentum_residual"
        ] <= gates["max_scale_normalized_linear_momentum_residual"],
    }
    return result


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_metric(metric: Any, label: str) -> None:
    if not isinstance(metric, dict):
        raise IntegrityError(f"{label} metric is not an object")
    require_exact_keys(metric, {
        "a_AU", "e", "i_deg", "q_AU", "distance_AU", "finite", "bound"
    }, f"{label} metric")
    numeric = ("a_AU", "e", "i_deg", "q_AU", "distance_AU")
    if not all(_finite_number(metric[key]) for key in numeric):
        raise IntegrityError(f"{label} metric contains a nonfinite number")
    if (
        metric["a_AU"] == 0.0
        or metric["e"] < 0.0
        or not (0.0 <= metric["i_deg"] <= 180.0)
        or metric["q_AU"] < 0.0
        or metric["distance_AU"] <= 0.0
        or metric["finite"] is not True
        or type(metric["bound"]) is not bool
        or metric["bound"] != (metric["a_AU"] > 0.0 and metric["e"] < 1.0)
    ):
        raise IntegrityError(f"{label} metric violates osculating semantics")


def _validate_first_passages(
    first: Any, minimum_q: float, horizon_year: float, label: str
) -> None:
    if not isinstance(first, dict):
        raise IntegrityError(f"{label} first-passage map is not an object")
    require_exact_keys(first, {"30", "35", "40"}, f"{label} first-passage map")
    for threshold in THRESHOLDS_AU:
        key = str(int(threshold))
        value = first[key]
        if value is not None and (
            not _finite_number(value)
            or value < 0.0
            or value > horizon_year
            or float(value) % 50.0 != 0.0
        ):
            raise IntegrityError(f"{label} first passage is off the exact sample grid")
        if (value is not None) != (minimum_q < threshold):
            raise IntegrityError(f"{label} first passage and prefix minimum disagree")
    first_30, first_35, first_40 = first["30"], first["35"], first["40"]
    if (
        (first_30 is not None and (
            first_35 is None or first_40 is None
            or first_40 > first_35 or first_35 > first_30
        ))
        or (first_35 is not None and (
            first_40 is None or first_40 > first_35
        ))
    ):
        raise IntegrityError(f"{label} threshold first passages are not chronologically nested")


def _invariants_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["intrinsic_energy"] == right["intrinsic_energy"]
        and list(left["com_angular_momentum"]) == list(right["com_angular_momentum"])
        and list(left["linear_momentum"]) == list(right["linear_momentum"])
        and left["linear_momentum_scale"] == right["linear_momentum_scale"]
    )


def validate_accumulator(
    accumulator: Any,
    arm: dict[str, Any],
    expected_completed_segment: int,
    contract: dict[str, Any],
    endpoint_state: Sequence[float],
) -> None:
    """Exhaustively validate the compact history before it can be published."""
    if not isinstance(accumulator, dict):
        raise IntegrityError("checkpoint accumulator is not an object")
    require_exact_keys(accumulator, {
        "sample_count", "sample_state_chain_head", "baseline_invariants",
        "maximum_active_drifts", "particles", "landmarks",
    }, "checkpoint accumulator")
    if not isinstance(expected_completed_segment, int) or not (0 <= expected_completed_segment < 20):
        raise IntegrityError("accumulator segment context is invalid")
    expected_sample_count = 1 + (expected_completed_segment + 1) * 1000
    if accumulator["sample_count"] != expected_sample_count:
        raise IntegrityError("accumulator sample count violates boundary ownership")
    if not is_lower_hex(accumulator["sample_state_chain_head"]):
        raise IntegrityError("accumulator sample-state chain head is invalid")
    baseline = accumulator["baseline_invariants"]
    if not isinstance(baseline, dict):
        raise IntegrityError("baseline invariants are not an object")
    require_exact_keys(baseline, {
        "intrinsic_energy", "com_angular_momentum", "linear_momentum",
        "linear_momentum_scale",
    }, "baseline invariants")
    if (
        not _finite_number(baseline["intrinsic_energy"])
        or baseline["intrinsic_energy"] == 0.0
        or not isinstance(baseline["com_angular_momentum"], (list, tuple))
        or len(baseline["com_angular_momentum"]) != 3
        or not all(_finite_number(value) for value in baseline["com_angular_momentum"])
        or _vector_norm3(baseline["com_angular_momentum"]) <= 0.0
        or not isinstance(baseline["linear_momentum"], (list, tuple))
        or len(baseline["linear_momentum"]) != 3
        or not all(_finite_number(value) for value in baseline["linear_momentum"])
        or not _finite_number(baseline["linear_momentum_scale"])
        or baseline["linear_momentum_scale"] <= 0.0
    ):
        raise IntegrityError("baseline invariant values are invalid")
    initial_positions, initial_velocities = matrix_from_state(
        arm["initial_state"], len(arm["logical_ids"])
    )
    expected_baseline = active_invariants(
        initial_positions[: arm["active_count"]].tolist(),
        initial_velocities[: arm["active_count"]].tolist(),
        arm["masses"][: arm["active_count"]],
        contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"],
    )
    if not _invariants_equal(baseline, expected_baseline):
        raise IntegrityError("baseline invariants do not match registered initial state")
    drifts = accumulator["maximum_active_drifts"]
    if not isinstance(drifts, dict):
        raise IntegrityError("maximum drift map is not an object")
    require_exact_keys(drifts, {
        "relative_compensated_intrinsic_energy_drift",
        "relative_com_angular_momentum_vector_drift",
        "scale_normalized_linear_momentum_residual",
    }, "maximum drift map")
    if not all(_finite_number(value) and value >= 0.0 for value in drifts.values()):
        raise IntegrityError("maximum drift map contains an invalid value")

    tracer_ids = arm["logical_ids"][arm["active_count"]:]
    block_by_id = dict(zip(tracer_ids, arm["tracer_block_indices"], strict=True))
    particles = accumulator["particles"]
    if not isinstance(particles, dict) or set(particles) != set(tracer_ids):
        raise IntegrityError("accumulator tracer identity changed")
    horizon_year = (expected_completed_segment + 1) * 50_000.0
    endpoint_metrics, endpoint_invariants = _sample_metrics(
        endpoint_state,
        arm,
        contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"],
    )
    endpoint_by_id = dict(zip(tracer_ids, endpoint_metrics, strict=True))
    endpoint_drift = invariant_drifts(endpoint_invariants, baseline)
    for key, value in endpoint_drift.items():
        if drifts[key] < value:
            raise IntegrityError("maximum drift is below the retained endpoint drift")
    for logical_id in tracer_ids:
        particle = particles[logical_id]
        if not isinstance(particle, dict):
            raise IntegrityError("accumulator particle row is not an object")
        require_exact_keys(particle, {
            "block_index", "minimum_sampled_q_AU", "first_sampled_below_year", "current"
        }, "accumulator particle")
        if particle["block_index"] != block_by_id[logical_id]:
            raise IntegrityError("accumulator particle block changed")
        minimum_q = particle["minimum_sampled_q_AU"]
        if not _finite_number(minimum_q) or minimum_q < 0.0:
            raise IntegrityError("accumulator prefix minimum is invalid")
        _validate_metric(particle["current"], "accumulator current")
        if particle["current"] != endpoint_by_id[logical_id]:
            raise IntegrityError("accumulator current metric disagrees with endpoint state")
        if minimum_q > particle["current"]["q_AU"]:
            raise IntegrityError("accumulator prefix minimum exceeds current q")
        _validate_first_passages(
            particle["first_sampled_below_year"], minimum_q, horizon_year,
            "accumulator particle",
        )

    expected_landmark_keys = {
        _landmark_key(year) for year in LANDMARK_YEARS if year <= horizon_year
    }
    landmarks = accumulator["landmarks"]
    if not isinstance(landmarks, dict) or set(landmarks) != expected_landmark_keys:
        raise IntegrityError("accumulator landmark ownership changed")
    sorted_ids = sorted(tracer_ids)
    for landmark_key in sorted(expected_landmark_keys, key=int):
        snapshot = landmarks[landmark_key]
        if not isinstance(snapshot, dict):
            raise IntegrityError("landmark snapshot is not an object")
        require_exact_keys(snapshot, {
            "landmark_year", "event_counts", "bound_count", "particles"
        }, "landmark snapshot")
        landmark_year = float(int(landmark_key))
        if snapshot["landmark_year"] != landmark_year:
            raise IntegrityError("landmark key and year disagree")
        counts = snapshot["event_counts"]
        if not isinstance(counts, dict):
            raise IntegrityError("landmark event counts are not an object")
        require_exact_keys(counts, {"30", "35", "40"}, "landmark event counts")
        if not all(type(value) is int and 0 <= value <= 32 for value in counts.values()):
            raise IntegrityError("landmark event count is invalid")
        if type(snapshot["bound_count"]) is not int or not (0 <= snapshot["bound_count"] <= 32):
            raise IntegrityError("landmark bound count is invalid")
        rows = snapshot["particles"]
        if not isinstance(rows, list) or len(rows) != 32:
            raise IntegrityError("landmark particle cardinality changed")
        if [row.get("logical_id") if isinstance(row, dict) else None for row in rows] != sorted_ids:
            raise IntegrityError("landmark particle identity/order changed")
        recomputed_counts = {"30": 0, "35": 0, "40": 0}
        recomputed_bound = 0
        for row in rows:
            require_exact_keys(row, {
                "logical_id", "block_index", "minimum_sampled_q_AU", "q_AU",
                "a_AU", "e", "i_deg", "distance_AU", "finite", "bound",
                "ever_sampled_below", "first_sampled_below_year",
                "censored_first_below_divided_by_landmark",
            }, "landmark particle")
            logical_id = row["logical_id"]
            if row["block_index"] != block_by_id[logical_id]:
                raise IntegrityError("landmark particle block changed")
            landmark_metric = {
                key: row[key] for key in (
                    "a_AU", "e", "i_deg", "q_AU", "distance_AU", "finite", "bound"
                )
            }
            _validate_metric(landmark_metric, "landmark")
            minimum_q = row["minimum_sampled_q_AU"]
            if not _finite_number(minimum_q) or minimum_q < 0.0 or minimum_q > row["q_AU"]:
                raise IntegrityError("landmark prefix minimum is invalid")
            _validate_first_passages(
                row["first_sampled_below_year"], minimum_q, landmark_year,
                "landmark particle",
            )
            indicators = row["ever_sampled_below"]
            censored = row["censored_first_below_divided_by_landmark"]
            if not isinstance(indicators, dict) or not isinstance(censored, dict):
                raise IntegrityError("landmark indicator or censor map is invalid")
            require_exact_keys(indicators, {"30", "35", "40"}, "landmark indicators")
            require_exact_keys(censored, {"30", "35", "40"}, "landmark censor map")
            for threshold in THRESHOLDS_AU:
                key = str(int(threshold))
                first = row["first_sampled_below_year"][key]
                indicator = first is not None
                expected_censored = (landmark_year if first is None else first) / landmark_year
                if (
                    type(indicators[key]) is not bool
                    or indicators[key] != indicator
                    or not _finite_number(censored[key])
                    or censored[key] != expected_censored
                    or not (0.0 <= censored[key] <= 1.0)
                ):
                    raise IntegrityError("landmark indicator/censor consistency failed")
                recomputed_counts[key] += int(indicator)
            recomputed_bound += int(row["bound"])
            final_particle = particles[logical_id]
            if minimum_q < final_particle["minimum_sampled_q_AU"]:
                raise IntegrityError("landmark prefix minimum is below later prefix minimum")
            for key in ("30", "35", "40"):
                first = row["first_sampled_below_year"][key]
                final_first = final_particle["first_sampled_below_year"][key]
                expected_landmark_first = (
                    final_first
                    if final_first is not None and final_first <= landmark_year
                    else None
                )
                if first != expected_landmark_first:
                    raise IntegrityError("landmark and final first passage disagree")
            if landmark_year == horizon_year:
                if (
                    minimum_q != final_particle["minimum_sampled_q_AU"]
                    or row["first_sampled_below_year"]
                    != final_particle["first_sampled_below_year"]
                    or landmark_metric != final_particle["current"]
                ):
                    raise IntegrityError("terminal landmark and accumulator disagree")
        if counts != recomputed_counts or snapshot["bound_count"] != recomputed_bound:
            raise IntegrityError("landmark aggregate counts disagree with particle rows")


def validate_segment_payload(
    payload: Any,
    arm: dict[str, Any],
    expected_segment_index: int,
    contract: dict[str, Any],
) -> None:
    if not isinstance(payload, dict):
        raise IntegrityError("segment payload is not an object")
    require_exact_keys(payload, {
        "segment_index", "start_year", "end_year", "end_state_hex", "accumulator"
    }, "segment payload")
    segment_years = contract["checkpoint_and_resume"]["segment_years"]
    if (
        payload["segment_index"] != expected_segment_index
        or payload["start_year"] != expected_segment_index * segment_years
        or payload["end_year"] != (expected_segment_index + 1) * segment_years
    ):
        raise IntegrityError("segment payload is not the exact next boundary")
    endpoint_state = state_from_hex(
        payload["end_state_hex"], len(arm["logical_ids"]) * 6
    )
    validate_accumulator(
        payload["accumulator"], arm, expected_segment_index, contract, endpoint_state
    )


def validate_accumulator_transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    arm: dict[str, Any],
    segment_index: int,
) -> None:
    """Ensure one payload extends, rather than rewrites, compact prior history."""
    if current["sample_count"] != previous["sample_count"] + 1000:
        raise IntegrityError("segment did not add exactly 1000 owned samples")
    if current["sample_state_chain_head"] == previous["sample_state_chain_head"]:
        raise IntegrityError("segment did not advance the sampled-state chain")
    if canonical_bytes(current["baseline_invariants"]) != canonical_bytes(
        previous["baseline_invariants"]
    ):
        raise IntegrityError("segment rewrote baseline invariants")
    for key, previous_value in previous["maximum_active_drifts"].items():
        if current["maximum_active_drifts"][key] < previous_value:
            raise IntegrityError("segment reduced a historical maximum drift")
    start_year = segment_index * 50_000.0
    for logical_id in arm["logical_ids"][arm["active_count"]:]:
        old = previous["particles"][logical_id]
        new = current["particles"][logical_id]
        if new["block_index"] != old["block_index"]:
            raise IntegrityError("segment rewrote a tracer block")
        if new["minimum_sampled_q_AU"] > old["minimum_sampled_q_AU"]:
            raise IntegrityError("segment increased a historical prefix minimum")
        for key in ("30", "35", "40"):
            old_first = old["first_sampled_below_year"][key]
            new_first = new["first_sampled_below_year"][key]
            if old_first is not None and new_first != old_first:
                raise IntegrityError("segment rewrote an existing first passage")
            if old_first is None and new_first is not None and new_first <= start_year:
                raise IntegrityError("segment inserted a first passage into prior history")
    old_landmarks = previous["landmarks"]
    new_landmarks = current["landmarks"]
    if not set(old_landmarks).issubset(new_landmarks):
        raise IntegrityError("segment removed a historical landmark")
    for key, old_snapshot in old_landmarks.items():
        if canonical_bytes(new_landmarks[key]) != canonical_bytes(old_snapshot):
            raise IntegrityError("segment rewrote a historical landmark")
    end_year = (segment_index + 1) * 50_000.0
    expected_added = {
        _landmark_key(year) for year in LANDMARK_YEARS
        if start_year < year <= end_year
    }
    if set(new_landmarks) - set(old_landmarks) != expected_added:
        raise IntegrityError("segment added the wrong landmark boundary")


def segment_commitment(
    arm_id: str,
    segment_payload: dict[str, Any],
    previous_chain_head: str,
) -> dict[str, Any]:
    payload_core = {
        "arm_id": arm_id,
        "segment_index": segment_payload["segment_index"],
        "start_year": segment_payload["start_year"],
        "end_year": segment_payload["end_year"],
        "end_state_hex": segment_payload["end_state_hex"],
        "accumulator_sha256": sha256_bytes(canonical_bytes(segment_payload["accumulator"])),
    }
    payload_sha = sha256_bytes(SEGMENT_PAYLOAD_DOMAIN + canonical_bytes(payload_core))
    chain_head = sha256_bytes(
        CHAIN_DOMAIN + bytes.fromhex(previous_chain_head) + bytes.fromhex(payload_sha)
    )
    return {
        "segment_index": segment_payload["segment_index"],
        "end_year": segment_payload["end_year"],
        "segment_payload_sha256": payload_sha,
        "chain_head_sha256": chain_head,
    }


def checkpoint_path(output_dir: Path, arm_id: str, segment_index: int) -> Path:
    if arm_id not in ARM_IDS or not isinstance(segment_index, int) or not 0 <= segment_index < 20:
        raise IntegrityError("checkpoint path identity is invalid")
    safe_id = arm_id.replace("-", "_")
    return output_dir / "checkpoints" / f"checkpoint_{safe_id}_segment_{segment_index:02d}.json"


def segment_receipt_path(output_dir: Path, arm_id: str, segment_index: int) -> Path:
    if arm_id not in ARM_IDS or not isinstance(segment_index, int) or not 0 <= segment_index < 20:
        raise IntegrityError("segment receipt path identity is invalid")
    safe_id = arm_id.replace("-", "_")
    return output_dir / "receipts" / f"receipt_{safe_id}_segment_{segment_index:02d}.json"


def validate_segment_receipt(
    output_dir: Path, arm: dict[str, Any], segment_index: int,
    input_bindings: dict[str, str], contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_file = checkpoint_path(output_dir, arm["arm_id"], segment_index)
    receipt_file = segment_receipt_path(output_dir, arm["arm_id"], segment_index)
    if (checkpoint_file.is_symlink() or receipt_file.is_symlink()
            or not checkpoint_file.is_file() or checkpoint_file.stat().st_nlink != 1
            or not receipt_file.is_file() or receipt_file.stat().st_nlink != 1):
        raise IntegrityError("immutable segment artifact is unsafe or incomplete")
    checkpoint = strict_json(checkpoint_file)
    next_segment, _state, _accumulator, _commitments = validate_checkpoint(
        checkpoint, arm, input_bindings, contract
    )
    if next_segment != segment_index + 1:
        raise IntegrityError("immutable checkpoint segment identity changed")
    receipt = strict_json(receipt_file)
    require_exact_keys(receipt, {
        "schema", "experiment_id", "execution_label", "arm_id", "segment_index",
        "input_bindings", "checkpoint_filename", "checkpoint_sha256",
        "checkpoint_size_bytes", "segment_payload_sha256",
        "segment_chain_head_sha256", "parent_terminal_validation",
        "parent_resource_validation",
    }, "immutable segment receipt")
    commitment = checkpoint["segment_commitments"][-1]
    if (
        receipt["schema"] != SEGMENT_RECEIPT_SCHEMA
        or receipt["experiment_id"] != EXPERIMENT_ID
        or receipt["execution_label"] != OFFICIAL_EXECUTION_LABEL
        or receipt["arm_id"] != arm["arm_id"]
        or receipt["segment_index"] != segment_index
        or receipt["input_bindings"] != input_bindings
        or receipt["checkpoint_filename"] != checkpoint_file.name
        or receipt["checkpoint_sha256"] != sha256_file(checkpoint_file)
        or receipt["checkpoint_size_bytes"] != checkpoint_file.stat().st_size
        or receipt["segment_payload_sha256"] != commitment["segment_payload_sha256"]
        or receipt["segment_chain_head_sha256"] != commitment["chain_head_sha256"]
        or receipt["parent_terminal_validation"]
        != "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS"
    ):
        raise IntegrityError("immutable segment receipt binding changed")
    resources = receipt["parent_resource_validation"]
    require_exact_keys(resources, {
        "segment_elapsed_seconds", "terminal_child_peak_rss_bytes",
        "coordinator_peak_rss_bytes", "total_elapsed_seconds_before_publication",
        "output_bytes_projected", "free_disk_bytes_before_publication",
    }, "immutable segment parent resource validation")
    caps = contract["resource_caps_per_execution"]
    if (
        not _finite_number(resources["segment_elapsed_seconds"])
        or resources["segment_elapsed_seconds"] >= caps["max_wall_seconds_per_segment_attempt"]
        or not isinstance(resources["terminal_child_peak_rss_bytes"], int)
        or isinstance(resources["terminal_child_peak_rss_bytes"], bool)
        or not 0 <= resources["terminal_child_peak_rss_bytes"]
        <= caps["max_peak_rss_bytes_per_process"]
        or not isinstance(resources["coordinator_peak_rss_bytes"], int)
        or isinstance(resources["coordinator_peak_rss_bytes"], bool)
        or not 0 <= resources["coordinator_peak_rss_bytes"]
        <= caps["max_peak_rss_bytes_per_process"]
        or not _finite_number(resources["total_elapsed_seconds_before_publication"])
        or not 0.0 <= resources["total_elapsed_seconds_before_publication"]
        < caps["max_wall_seconds_total"]
        or not isinstance(resources["output_bytes_projected"], int)
        or isinstance(resources["output_bytes_projected"], bool)
        or not checkpoint_file.stat().st_size + receipt_file.stat().st_size
        <= resources["output_bytes_projected"] <= caps["max_output_bytes"]
        or not isinstance(resources["free_disk_bytes_before_publication"], int)
        or isinstance(resources["free_disk_bytes_before_publication"], bool)
        or resources["free_disk_bytes_before_publication"]
        < caps["minimum_free_disk_bytes"]
    ):
        raise IntegrityError("immutable segment resource validation changed")
    return checkpoint, receipt


def make_checkpoint(
    arm: dict[str, Any],
    segment_payload: dict[str, Any],
    commitments: list[dict[str, Any]],
    input_bindings: dict[str, str],
) -> dict[str, Any]:
    previous = INITIAL_CHAIN if not commitments else commitments[-1]["chain_head_sha256"]
    commitment = segment_commitment(arm["arm_id"], segment_payload, previous)
    updated = commitments + [commitment]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "arm_id": arm["arm_id"],
        "input_bindings": input_bindings,
        "completed_segment_index": segment_payload["segment_index"],
        "end_year": segment_payload["end_year"],
        "end_state_hex": segment_payload["end_state_hex"],
        "accumulator": segment_payload["accumulator"],
        "segment_commitments": updated,
        "segment_chain_head_sha256": commitment["chain_head_sha256"],
    }


def validate_checkpoint(
    checkpoint: dict[str, Any],
    arm: dict[str, Any],
    input_bindings: dict[str, str],
    contract: dict[str, Any],
) -> tuple[int, list[float], dict[str, Any], list[dict[str, Any]]]:
    require_exact_keys(checkpoint, {
        "schema", "experiment_id", "execution_label", "arm_id", "input_bindings",
        "completed_segment_index", "end_year", "end_state_hex", "accumulator",
        "segment_commitments", "segment_chain_head_sha256",
    }, "checkpoint")
    if (
        checkpoint["schema"] != CHECKPOINT_SCHEMA
        or checkpoint["experiment_id"] != EXPERIMENT_ID
        or checkpoint["execution_label"] != OFFICIAL_EXECUTION_LABEL
        or checkpoint["arm_id"] != arm["arm_id"]
        or checkpoint["input_bindings"] != input_bindings
    ):
        raise IntegrityError("checkpoint identity or input binding changed")
    segment_index = checkpoint["completed_segment_index"]
    segment_count = contract["checkpoint_and_resume"]["segments_per_arm"]
    if not isinstance(segment_index, int) or not (0 <= segment_index < segment_count):
        raise IntegrityError("checkpoint segment index is invalid")
    expected_end = (segment_index + 1) * contract["checkpoint_and_resume"]["segment_years"]
    if checkpoint["end_year"] != expected_end:
        raise IntegrityError("checkpoint end time changed")
    particle_count = len(arm["logical_ids"])
    state = state_from_hex(checkpoint["end_state_hex"], particle_count * 6)
    accumulator = checkpoint["accumulator"]
    validate_accumulator(accumulator, arm, segment_index, contract, state)
    commitments = checkpoint["segment_commitments"]
    if not isinstance(commitments, list) or len(commitments) != segment_index + 1:
        raise IntegrityError("checkpoint commitment count changed")
    previous = INITIAL_CHAIN
    for expected_index, row in enumerate(commitments):
        require_exact_keys(row, {
            "segment_index", "end_year", "segment_payload_sha256", "chain_head_sha256"
        }, "segment commitment")
        if (
            row["segment_index"] != expected_index
            or row["end_year"] != (expected_index + 1) * 50_000.0
            or not is_lower_hex(row["segment_payload_sha256"])
            or not is_lower_hex(row["chain_head_sha256"])
        ):
            raise IntegrityError("segment commitment order or shape changed")
        expected_chain = sha256_bytes(
            CHAIN_DOMAIN
            + bytes.fromhex(previous)
            + bytes.fromhex(row["segment_payload_sha256"])
        )
        if row["chain_head_sha256"] != expected_chain:
            raise IntegrityError("segment hash chain is broken")
        previous = expected_chain
    if checkpoint["segment_chain_head_sha256"] != previous:
        raise IntegrityError("checkpoint chain head mismatch")
    # The last retained boundary state and accumulator must commit to the last
    # segment payload, while earlier payloads remain represented by the chain.
    last_payload = {
        "arm_id": arm["arm_id"],
        "segment_index": segment_index,
        "start_year": segment_index * 50_000.0,
        "end_year": expected_end,
        "end_state_hex": checkpoint["end_state_hex"],
        "accumulator_sha256": sha256_bytes(canonical_bytes(accumulator)),
    }
    expected_payload_sha = sha256_bytes(
        SEGMENT_PAYLOAD_DOMAIN + canonical_bytes(last_payload)
    )
    if commitments[-1]["segment_payload_sha256"] != expected_payload_sha:
        raise IntegrityError("checkpoint retained payload does not match chain commitment")
    return segment_index + 1, state, accumulator, commitments


def finalize_arm_result(
    arm: dict[str, Any], checkpoint: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    accumulator = checkpoint["accumulator"]
    if accumulator["sample_count"] != 20_001:
        raise IntegrityError("completed arm has the wrong sample count")
    if set(accumulator["landmarks"]) != {"250000", "500000", "1000000"}:
        raise IntegrityError("completed arm is missing a locked landmark")
    gate_pass = check_active_gates(accumulator, contract)
    return {
        "arm_id": arm["arm_id"],
        "active_body_count": arm["active_count"],
        "sentinel_tracer_count": 32,
        "initial_state_sha256": arm["initial_state_sha256"],
        "registered_expanded_initial_state_sha256": arm[
            "registered_expanded_initial_state_sha256"
        ],
        "sample_count": accumulator["sample_count"],
        "sample_state_chain_head_sha256": accumulator["sample_state_chain_head"],
        "segment_chain_head_sha256": checkpoint["segment_chain_head_sha256"],
        "maximum_active_drifts": accumulator["maximum_active_drifts"],
        "active_gate_pass": gate_pass,
        "all_active_gates_pass": all(gate_pass.values()),
        "landmarks": [accumulator["landmarks"][key] for key in (
            "250000", "500000", "1000000"
        )],
    }


def decode_attempt_ledger_bytes(payload: bytes) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise IntegrityError("attempt ledger lacks its canonical final newline")
    rows: list[dict[str, Any]] = []
    for expected_sequence, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            raise IntegrityError("attempt ledger contains a blank line")
        row = strict_json_bytes(raw)
        if raw + b"\n" != canonical_bytes(row) + b"\n":
            raise IntegrityError("attempt ledger row is not canonical JSONL")
        common = {
            "schema", "sequence", "event", "arm_id", "segment_index",
            "attempt_number_for_segment",
        }
        event = row.get("event")
        if event == "SEGMENT_ATTEMPT_STARTED":
            expected_keys = common
        elif event == "SEGMENT_ATTEMPT_FAILED":
            expected_keys = common | {
                "failure_class", "fail_event_sha256",
                "failure_receipt_filename", "failure_receipt_sha256",
            }
        elif event == "SEGMENT_ATTEMPT_COMMITTED":
            expected_keys = common | {
                "elapsed_seconds", "terminal_peak_rss_bytes", "checkpoint_sha256",
                "segment_receipt_sha256",
            }
        else:
            raise IntegrityError("attempt ledger event is unknown")
        if "recovery" in row:
            if event == "SEGMENT_ATTEMPT_STARTED":
                raise IntegrityError("attempt start cannot be a recovery row")
            expected_keys = expected_keys | {"recovery"}
        require_exact_keys(row, expected_keys, "attempt ledger row")
        if (
            row["schema"] != ATTEMPT_SCHEMA
            or type(row["sequence"]) is not int
            or row["sequence"] != expected_sequence
            or row["arm_id"] not in ARM_IDS
            or type(row["segment_index"]) is not int
            or not (0 <= row["segment_index"] < 20)
            or type(row["attempt_number_for_segment"]) is not int
            or not (1 <= row["attempt_number_for_segment"] <= 3)
        ):
            raise IntegrityError("attempt ledger identity or counter changed")
        if "recovery" in row and row["recovery"] not in {
            "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION",
            "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION",
        }:
            raise IntegrityError("attempt ledger recovery marker is invalid")
        if event == "SEGMENT_ATTEMPT_FAILED" and (
            not isinstance(row["failure_class"], str)
            or row["failure_class"] not in DOP_FAILURE_CLASSES
            or not is_lower_hex(row["fail_event_sha256"])
            or row["failure_receipt_filename"]
            != failure_receipt_filename(
                row["arm_id"], row["segment_index"],
                row["attempt_number_for_segment"],
            )
            or not is_lower_hex(row["failure_receipt_sha256"])
        ):
            raise IntegrityError("attempt failure class is invalid")
        if ((row.get("recovery")
                == "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION"
                and (event != "SEGMENT_ATTEMPT_FAILED"
                     or row["failure_class"] != "InterruptedAttempt"))
                or (row.get("recovery")
                    == "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION"
                    and event != "SEGMENT_ATTEMPT_COMMITTED")):
            raise IntegrityError("attempt recovery marker/event pairing changed")
        if event == "SEGMENT_ATTEMPT_COMMITTED" and (
            type(row["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(row["elapsed_seconds"])
            or row["elapsed_seconds"] < 0.0
            or type(row["terminal_peak_rss_bytes"]) is not int
            or row["terminal_peak_rss_bytes"] < 0
            or not is_lower_hex(row["checkpoint_sha256"])
            or not is_lower_hex(row["segment_receipt_sha256"])
        ):
            raise IntegrityError("attempt completion provenance is invalid")
        rows.append(row)
    return rows


def read_attempt_ledger(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "attempt_ledger.jsonl"
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("attempt ledger is not a safe regular file")
    return decode_attempt_ledger_bytes(path.read_bytes())


def append_attempt_ledger(output_dir: Path, row: dict[str, Any]) -> None:
    """Publish one canonical row by atomic whole-ledger replacement."""
    path = output_dir / "attempt_ledger.jsonl"
    pending = output_dir / ".attempt_ledger.jsonl.pending"
    rows = read_attempt_ledger(output_dir)
    if pending.exists() or pending.is_symlink():
        raise IntegrityError("stale pending attempt ledger exists")
    candidate_rows = [*rows, row]
    if type(row.get("sequence")) is not int or row["sequence"] != len(candidate_rows):
        raise IntegrityError("attempt ledger append sequence changed")
    payload = b"".join(canonical_bytes(item) + b"\n" for item in candidate_rows)
    decoded_rows = decode_attempt_ledger_bytes(payload)
    if decoded_rows != candidate_rows:
        raise IntegrityError("attempt ledger append changed during serialization")
    replay_attempt_ledger(decoded_rows)
    descriptor = os.open(
        pending,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if pending.stat().st_size != len(payload) or sha256_file(pending) != sha256_bytes(payload):
        raise IntegrityError("pending attempt ledger hash or size changed")
    os.replace(pending, path)
    _fsync_directory(output_dir)


def replay_attempt_ledger(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    dict[str, int], dict[tuple[str, int], int],
    dict[tuple[str, int], tuple[str, str]], dict[str, Any] | None,
]:
    """Validate exact START -> FAIL/COMMIT state-machine ordering."""
    next_segment = {arm_id: 0 for arm_id in ARM_IDS}
    attempts: dict[tuple[str, int], int] = {}
    committed_hashes: dict[tuple[str, int], tuple[str, str]] = {}
    pending: dict[str, Any] | None = None
    for row in rows:
        event = row["event"]
        arm_id = row["arm_id"]
        segment_index = row["segment_index"]
        attempt_number = row["attempt_number_for_segment"]
        if event == "SEGMENT_ATTEMPT_STARTED":
            if pending is not None:
                raise IntegrityError("attempt ledger has overlapping START rows")
            if segment_index != next_segment[arm_id]:
                raise IntegrityError("attempt ledger starts a skipped or completed segment")
            arm_position = ARM_IDS.index(arm_id)
            if any(next_segment[prior] != 20 for prior in ARM_IDS[:arm_position]):
                raise IntegrityError("attempt ledger starts an arm before its predecessor completed")
            if any(next_segment[later] != 0 for later in ARM_IDS[arm_position + 1:]):
                raise IntegrityError("attempt ledger arm order moved backwards")
            key = (arm_id, segment_index)
            expected_attempt = attempts.get(key, 0) + 1
            if attempt_number != expected_attempt:
                raise IntegrityError("attempt number is not contiguous for its exact segment")
            attempts[key] = attempt_number
            pending = row
            continue
        if pending is None:
            raise IntegrityError("attempt terminal row has no preceding START")
        if (
            arm_id != pending["arm_id"]
            or segment_index != pending["segment_index"]
            or attempt_number != pending["attempt_number_for_segment"]
        ):
            raise IntegrityError("attempt terminal row does not match its START")
        if event == "SEGMENT_ATTEMPT_COMMITTED":
            next_segment[arm_id] += 1
            committed_hashes[(arm_id, segment_index)] = (
                row["checkpoint_sha256"], row["segment_receipt_sha256"]
            )
        pending = None
    return next_segment, attempts, committed_hashes, pending


def recover_pending_attempt_ledger(output_dir: Path) -> None:
    """Promote only an exact complete one-row extension; discard torn bytes."""
    path = output_dir / "attempt_ledger.jsonl"
    pending = output_dir / ".attempt_ledger.jsonl.pending"
    if not pending.exists() and not pending.is_symlink():
        return
    if (pending.is_symlink() or not pending.is_file()
            or pending.stat().st_nlink != 1):
        raise IntegrityError("pending attempt ledger is unsafe")
    old_rows = read_attempt_ledger(output_dir)
    old_payload = (output_dir / "attempt_ledger.jsonl").read_bytes() \
        if (output_dir / "attempt_ledger.jsonl").exists() else b""
    pending_payload = pending.read_bytes()
    if old_payload.startswith(pending_payload):
        pending.unlink()
        _fsync_directory(output_dir)
        return
    if not pending_payload.startswith(old_payload):
        raise IntegrityError("pending attempt ledger diverges from published bytes")
    suffix = pending_payload[len(old_payload):]
    if not suffix.endswith(b"\n"):
        pending.unlink()
        _fsync_directory(output_dir)
        return
    try:
        candidate_rows = decode_attempt_ledger_bytes(pending_payload)
        replay_attempt_ledger(candidate_rows)
    except (ValueError, IntegrityError, OSError):
        raise IntegrityError("complete pending attempt ledger extension is invalid")
    if len(candidate_rows) == len(old_rows) + 1 and candidate_rows[:-1] == old_rows:
        os.replace(pending, path)
        _fsync_directory(output_dir)
        return
    raise IntegrityError("pending attempt ledger is not one exact extension")


def validate_ledger_against_checkpoints(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    arms: dict[str, dict[str, Any]],
    input_bindings: dict[str, str],
    contract: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    ledger_next, _attempts, committed_hashes, pending = replay_attempt_ledger(rows)
    pending_checkpoint_advanced = False
    checkpoint_names: set[str] = set()
    receipt_names: set[str] = set()
    for arm_id in ARM_IDS:
        previous_accumulator = initialize_accumulator(
            arms[arm_id], contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        previous_commitments: list[dict[str, Any]] = []
        for segment_index in range(20):
            checkpoint_file = checkpoint_path(output_dir, arm_id, segment_index)
            receipt_file = segment_receipt_path(output_dir, arm_id, segment_index)
            checkpoint_exists = checkpoint_file.exists() or checkpoint_file.is_symlink()
            receipt_exists = receipt_file.exists() or receipt_file.is_symlink()
            if checkpoint_exists:
                checkpoint_names.add(checkpoint_file.name)
            if receipt_exists:
                receipt_names.add(receipt_file.name)
            committed = segment_index < ledger_next[arm_id]
            is_pending = (pending is not None and pending["arm_id"] == arm_id
                          and pending["segment_index"] == segment_index)
            if committed:
                if not checkpoint_exists or not receipt_exists:
                    raise IntegrityError("committed immutable segment artifact is missing")
                checkpoint, _receipt = validate_segment_receipt(
                    output_dir, arms[arm_id], segment_index, input_bindings, contract
                )
                validate_accumulator_transition(
                    previous_accumulator, checkpoint["accumulator"],
                    arms[arm_id], segment_index,
                )
                if checkpoint["segment_commitments"][:-1] != previous_commitments:
                    raise IntegrityError("immutable checkpoint rewrote prior commitments")
                previous_accumulator = checkpoint["accumulator"]
                previous_commitments = checkpoint["segment_commitments"]
                expected_hashes = committed_hashes.get((arm_id, segment_index))
                actual_hashes = (sha256_file(checkpoint_file), sha256_file(receipt_file))
                if expected_hashes != actual_hashes:
                    raise IntegrityError("COMMITTED ledger hashes differ from immutable artifacts")
            elif is_pending:
                if checkpoint_exists and receipt_exists:
                    checkpoint, _receipt = validate_segment_receipt(
                        output_dir, arms[arm_id], segment_index, input_bindings, contract
                    )
                    validate_accumulator_transition(
                        previous_accumulator, checkpoint["accumulator"],
                        arms[arm_id], segment_index,
                    )
                    if checkpoint["segment_commitments"][:-1] != previous_commitments:
                        raise IntegrityError("pending checkpoint rewrote prior commitments")
                    pending_checkpoint_advanced = True
                # A one-file partial publication is ineligible and is removed only
                # by the coordinator's deterministic recovery path below.
            elif checkpoint_exists or receipt_exists:
                raise IntegrityError("uncommitted immutable segment artifact exists")
    checkpoint_dir = output_dir / "checkpoints"
    receipt_dir = output_dir / "receipts"
    for directory, expected in ((checkpoint_dir, checkpoint_names), (receipt_dir, receipt_names)):
        if directory.is_symlink() or not directory.is_dir():
            raise IntegrityError("immutable artifact directory is unsafe")
        entries = list(directory.iterdir())
        if (any(entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1
                for entry in entries) or {entry.name for entry in entries} != expected):
            raise IntegrityError("immutable artifact directory contains an extra entry")
    return pending, pending_checkpoint_advanced


def reconcile_attempt_ledger(
    output_dir: Path,
    arms: dict[str, dict[str, Any]],
    input_bindings: dict[str, str],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Close the single crash-window START using validated checkpoint truth."""
    rows = read_attempt_ledger(output_dir)
    _next, _attempts, _hashes, raw_pending = replay_attempt_ledger(rows)
    validate_failure_receipt_bindings(
        rows, output_dir, open_start=raw_pending
    )
    if raw_pending is not None:
        for final_path in (
            checkpoint_path(
                output_dir, raw_pending["arm_id"], raw_pending["segment_index"]
            ),
            segment_receipt_path(
                output_dir, raw_pending["arm_id"], raw_pending["segment_index"]
            ),
        ):
            pending_path = final_path.with_name(f".{final_path.name}.pending")
            if pending_path.exists() or pending_path.is_symlink():
                if (pending_path.is_symlink() or not pending_path.is_file()
                        or pending_path.stat().st_nlink != 1):
                    raise IntegrityError("interrupted pending immutable artifact is unsafe")
                pending_path.unlink()
                _fsync_directory(pending_path.parent)
    pending, checkpoint_advanced = validate_ledger_against_checkpoints(
        rows, output_dir, arms, input_bindings, contract
    )
    if pending is not None:
        common = {
            "schema": ATTEMPT_SCHEMA,
            "sequence": len(rows) + 1,
            "arm_id": pending["arm_id"],
            "segment_index": pending["segment_index"],
            "attempt_number_for_segment": pending["attempt_number_for_segment"],
        }
        if checkpoint_advanced:
            if recover_or_read_orphan_failure_receipt(output_dir, pending) is not None:
                raise IntegrityError("pending attempt has both checkpoint and failure receipt")
            path = checkpoint_path(output_dir, pending["arm_id"], pending["segment_index"])
            receipt_path = segment_receipt_path(
                output_dir, pending["arm_id"], pending["segment_index"]
            )
            recovery = common | {
                "event": "SEGMENT_ATTEMPT_COMMITTED",
                "elapsed_seconds": 0.0,
                "terminal_peak_rss_bytes": 0,
                "checkpoint_sha256": sha256_file(path),
                "segment_receipt_sha256": sha256_file(receipt_path),
                "recovery": "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION",
            }
        else:
            for partial in (
                checkpoint_path(output_dir, pending["arm_id"], pending["segment_index"]),
                segment_receipt_path(output_dir, pending["arm_id"], pending["segment_index"]),
            ):
                if partial.exists() or partial.is_symlink():
                    if partial.is_symlink() or not partial.is_file() or partial.stat().st_nlink != 1:
                        raise IntegrityError("partial immutable segment artifact is unsafe")
                    partial.unlink()
                    _fsync_directory(partial.parent)
            orphan = recover_or_read_orphan_failure_receipt(output_dir, pending)
            if orphan is None:
                receipt_path, receipt = publish_failure_receipt(
                    output_dir, pending, "InterruptedAttempt",
                    "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION",
                )
            else:
                receipt_path, receipt = orphan
            recovery = common | {
                "event": "SEGMENT_ATTEMPT_FAILED",
                "failure_class": receipt["failure_class"],
                "fail_event_sha256": receipt["fail_event_sha256"],
                "failure_receipt_filename": receipt_path.name,
                "failure_receipt_sha256": sha256_file(receipt_path),
            }
            if receipt["recovery"] is not None:
                recovery["recovery"] = receipt["recovery"]
        append_attempt_ledger(output_dir, recovery)
        rows = read_attempt_ledger(output_dir)
        final_pending, _advanced = validate_ledger_against_checkpoints(
            rows, output_dir, arms, input_bindings, contract
        )
        if final_pending is not None:
            raise AssertionError("attempt-ledger reconciliation did not close START")
    validate_failure_receipt_bindings(rows, output_dir)
    return rows


def next_attempt_number(ledger: Sequence[dict[str, Any]], arm_id: str, segment_index: int) -> int:
    starts = sum(
        row.get("event") == "SEGMENT_ATTEMPT_STARTED"
        and row.get("arm_id") == arm_id
        and row.get("segment_index") == segment_index
        for row in ledger
    )
    return starts + 1


def failure_receipt_filename(arm_id: str, segment_index: int, attempt_number: int) -> str:
    if (arm_id not in ARM_IDS or type(segment_index) is not int
            or not 0 <= segment_index < 20 or type(attempt_number) is not int
            or not 1 <= attempt_number <= 3):
        raise IntegrityError("failure receipt identity is invalid")
    return f"failure_{arm_id}_segment_{segment_index:02d}_attempt_{attempt_number:02d}.json"


def failure_event_core(
    start: dict[str, Any], failure_class: str, recovery: str | None,
) -> dict[str, Any]:
    if (start.get("schema") != ATTEMPT_SCHEMA
            or start.get("event") != "SEGMENT_ATTEMPT_STARTED"
            or type(start.get("sequence")) is not int
            or start.get("arm_id") not in ARM_IDS
            or type(start.get("segment_index")) is not int
            or type(start.get("attempt_number_for_segment")) is not int
            or not isinstance(failure_class, str)
            or failure_class not in DOP_FAILURE_CLASSES
            or (recovery is not None and (
                not isinstance(recovery, str)
                or recovery != "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION"
            ))
            or ((recovery is not None) != (failure_class == "InterruptedAttempt"))):
        raise IntegrityError("failure event identity or class is invalid")
    return {
        "schema": "jx-xp2-dop853-failure-event/v2",
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "event": "SEGMENT_ATTEMPT_FAILED",
        "arm_id": start["arm_id"],
        "segment_index": start["segment_index"],
        "attempt_number": start["attempt_number_for_segment"],
        "start_sequence": start["sequence"],
        "failure_class": failure_class,
        "recovery": recovery,
    }


def failure_receipt_payload(
    start: dict[str, Any], failure_class: str, recovery: str | None = None,
) -> dict[str, Any]:
    core = failure_event_core(start, failure_class, recovery)
    return {
        "schema": FAILURE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "event": "SEGMENT_ATTEMPT_FAILED",
        "arm_id": core["arm_id"],
        "segment_index": core["segment_index"],
        "attempt_number": core["attempt_number"],
        "start_sequence": core["start_sequence"],
        "failure_class": failure_class,
        "recovery": recovery,
        "fail_event_sha256": sha256_bytes(
            DOP_FAILURE_EVENT_DOMAIN + canonical_bytes(core)
        ),
        "failure_message": REDACTED_FAILURE_MESSAGE,
        "result_emitted": False,
        "scientific_classification_emitted": False,
        "mandatory_nonclaim": (
            "An incomplete or failed numerical sentinel is not a scientific result."
        ),
    }


def validate_failure_receipt_payload(
    receipt: dict[str, Any], start: dict[str, Any],
) -> None:
    require_exact_keys(receipt, {
        "schema", "experiment_id", "execution_label", "event", "arm_id",
        "segment_index", "attempt_number", "start_sequence", "failure_class",
        "recovery", "fail_event_sha256", "failure_message", "result_emitted",
        "scientific_classification_emitted", "mandatory_nonclaim",
    }, "DOP853 failure receipt")
    if (type(receipt["segment_index"]) is not int
            or type(receipt["attempt_number"]) is not int
            or type(receipt["start_sequence"]) is not int):
        raise IntegrityError("DOP853 failure receipt integer identity changed")
    expected = failure_receipt_payload(
        start, receipt["failure_class"], receipt["recovery"]
    )
    if receipt != expected:
        raise IntegrityError("DOP853 failure receipt semantics changed")


def publish_failure_receipt(
    output_dir: Path, start: dict[str, Any], failure_class: str,
    recovery: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    directory = output_dir / "failures"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise IntegrityError("failure receipt directory is unsafe")
    directory.mkdir(exist_ok=True)
    filename = failure_receipt_filename(
        start["arm_id"], start["segment_index"],
        start["attempt_number_for_segment"],
    )
    path = directory / filename
    pending = directory / f".{filename}.pending"
    receipt = failure_receipt_payload(start, failure_class, recovery)
    expected_bytes = serialized_json(receipt)
    if path.exists() or path.is_symlink():
        if pending.exists() or pending.is_symlink():
            raise IntegrityError("published failure receipt conflicts with pending bytes")
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.read_bytes() != expected_bytes):
            raise IntegrityError("published failure receipt changed")
        return path, receipt
    if pending.exists() or pending.is_symlink():
        if (pending.is_symlink() or not pending.is_file()
                or pending.stat().st_nlink != 1):
            raise IntegrityError("pending failure receipt is unsafe")
        pending_bytes = pending.read_bytes()
        if pending_bytes == expected_bytes:
            os.replace(pending, path)
            _fsync_directory(directory)
            return path, receipt
        if expected_bytes.startswith(pending_bytes):
            pending.unlink()
            _fsync_directory(directory)
        else:
            raise IntegrityError("pending failure receipt diverges from expected bytes")
    atomic_create_json(path, receipt)
    return path, receipt


def failure_class_for_error(error: BaseException) -> str:
    name = type(error).__name__
    return name if name in DOP_FAILURE_CLASSES else "UnexpectedFailure"


def recover_or_read_orphan_failure_receipt(
    output_dir: Path, start: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    directory = output_dir / "failures"
    filename = failure_receipt_filename(
        start["arm_id"], start["segment_index"],
        start["attempt_number_for_segment"],
    )
    path = directory / filename
    pending = directory / f".{filename}.pending"
    if path.exists() or path.is_symlink():
        if pending.exists() or pending.is_symlink():
            raise IntegrityError("orphan receipt has conflicting pending bytes")
        receipt = strict_json(path)
        validate_failure_receipt_payload(receipt, start)
        if path.read_bytes() != serialized_json(receipt):
            raise IntegrityError("orphan failure receipt is not canonical")
        return path, receipt
    if not pending.exists() and not pending.is_symlink():
        return None
    if (pending.is_symlink() or not pending.is_file()
            or pending.stat().st_nlink != 1):
        raise IntegrityError("pending orphan failure receipt is unsafe")
    pending_bytes = pending.read_bytes()
    try:
        receipt = strict_json(pending)
        validate_failure_receipt_payload(receipt, start)
        complete = pending_bytes == serialized_json(receipt)
    except (ValueError, IntegrityError, OSError):
        complete = False
    if not complete:
        candidates = [
            failure_receipt_payload(start, failure_class)
            for failure_class in sorted(DOP_FAILURE_CLASSES - {"InterruptedAttempt"})
        ] + [failure_receipt_payload(
            start, "InterruptedAttempt",
            "COORDINATOR_INTERRUPTED_BEFORE_CHECKPOINT_PUBLICATION",
        )]
        if any(serialized_json(candidate).startswith(pending_bytes)
               for candidate in candidates):
            pending.unlink()
            _fsync_directory(directory)
            return None
        raise IntegrityError("pending orphan failure receipt diverges from every valid receipt")
    os.replace(pending, path)
    _fsync_directory(directory)
    return path, receipt


def validate_failure_receipt_bindings(
    rows: Sequence[dict[str, Any]], output_dir: Path,
    *, open_start: dict[str, Any] | None = None,
) -> None:
    directory = output_dir / "failures"
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityError("DOP853 failure directory is unsafe")
    starts: dict[tuple[str, int, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        key = (row["arm_id"], row["segment_index"], row["attempt_number_for_segment"])
        if row["event"] == "SEGMENT_ATTEMPT_STARTED":
            starts[key] = row
        elif row["event"] == "SEGMENT_ATTEMPT_FAILED":
            failures.append(row)
    expected_names = {row["failure_receipt_filename"] for row in failures}
    entries = list(directory.iterdir())
    actual_names = {entry.name for entry in entries}
    allowed_orphan_names: set[str] = set()
    if open_start is not None:
        orphan_name = failure_receipt_filename(
            open_start["arm_id"], open_start["segment_index"],
            open_start["attempt_number_for_segment"],
        )
        allowed_orphan_names = {orphan_name, f".{orphan_name}.pending"}
    orphan_entries = actual_names - expected_names
    if (any(entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1
            for entry in entries) or not expected_names <= actual_names
            or not orphan_entries <= allowed_orphan_names
            or len(orphan_entries) > 1):
        raise IntegrityError("DOP853 FAIL/receipt inventory is not an exact bijection")
    for row in failures:
        key = (row["arm_id"], row["segment_index"], row["attempt_number_for_segment"])
        start = starts.get(key)
        if start is None:
            raise IntegrityError("DOP853 FAIL receipt lacks its exact START")
        path = directory / row["failure_receipt_filename"]
        receipt = strict_json(path)
        validate_failure_receipt_payload(receipt, start)
        if (path.read_bytes() != serialized_json(receipt)
                or sha256_file(path) != row["failure_receipt_sha256"]
                or receipt["fail_event_sha256"] != row["fail_event_sha256"]
                or receipt["failure_class"] != row["failure_class"]
                or receipt["recovery"] != row.get("recovery")):
            raise IntegrityError("DOP853 FAIL/receipt binding changed")


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _worker_send(write_fd: int, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    frame = FRAME_HEADER.pack(len(payload)) + payload
    offset = 0
    while offset < len(frame):
        offset += os.write(write_fd, frame[offset:])


def supervise_worker(
    worker: Callable[[], dict[str, Any]],
    wall_seconds: float,
    rss_cap_bytes: int,
    output_dir: Path,
    output_cap_bytes: int,
    total_deadline_ns: int,
    poll_seconds: float,
) -> tuple[dict[str, Any], int]:
    """Run one arm attempt behind a hard POSIX process/RSS/wall boundary."""
    if proc_thread_count() != 1:
        raise ResourceLimitError("coordinator must have exactly one thread before fork")
    read_fd, write_fd = os.pipe()
    started_ns = time.monotonic_ns()
    arm_deadline_ns = started_ns + int(wall_seconds * NANOSECONDS_PER_SECOND)
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            os.setpgid(0, 0)
            try:
                payload = worker()
                response = {"ok": True, "payload": payload}
            except BaseException as error:  # child sends no path-bearing message
                response = {"ok": False, "error_type": type(error).__name__}
            _worker_send(write_fd, response)
        except BaseException:
            pass
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass
            os._exit(0)
    os.close(write_fd)
    status: int | None = None
    try:
        try:
            os.setpgid(pid, pid)
        except (ProcessLookupError, PermissionError):
            pass
        fcntl.fcntl(read_fd, fcntl.F_SETFL, fcntl.fcntl(read_fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        received = bytearray()
        terminal_rss = 0
        while status is None:
            now = time.monotonic_ns()
            live_rss = child_peak_rss_bytes(pid)
            if now >= arm_deadline_ns or now >= total_deadline_ns:
                _kill_process_group(pid)
                _wait_pid, status, usage = os.wait4(pid, 0)
                terminal_rss = int(usage.ru_maxrss) * 1024
                raise ResourceLimitError("worker wall deadline reached")
            if live_rss > rss_cap_bytes or peak_rss_bytes_self() > rss_cap_bytes:
                _kill_process_group(pid)
                _wait_pid, status, usage = os.wait4(pid, 0)
                terminal_rss = int(usage.ru_maxrss) * 1024
                raise ResourceLimitError("worker or coordinator RSS cap reached")
            if directory_bytes(output_dir) > output_cap_bytes:
                _kill_process_group(pid)
                _wait_pid, status, usage = os.wait4(pid, 0)
                terminal_rss = int(usage.ru_maxrss) * 1024
                raise ResourceLimitError("output cap reached")
            while True:
                try:
                    chunk = os.read(read_fd, 1024 * 1024)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                received.extend(chunk)
                if len(received) > output_cap_bytes:
                    _kill_process_group(pid)
                    _wait_pid, status, usage = os.wait4(pid, 0)
                    terminal_rss = int(usage.ru_maxrss) * 1024
                    raise ResourceLimitError("worker response cap reached")
            waited_pid, child_status, usage = os.wait4(pid, os.WNOHANG)
            if waited_pid == pid:
                status = child_status
                terminal_rss = int(usage.ru_maxrss) * 1024
                break
            select.select([read_fd], [], [], poll_seconds)
        while True:
            try:
                chunk = os.read(read_fd, 1024 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            received.extend(chunk)
        if status != 0:
            raise NumericalError("arm worker terminated abnormally")
        if terminal_rss > rss_cap_bytes:
            raise ResourceLimitError("terminal worker RSS cap reached")
        if len(received) < FRAME_HEADER.size:
            raise NumericalError("arm worker response is missing")
        (payload_length,) = FRAME_HEADER.unpack(received[: FRAME_HEADER.size])
        if payload_length != len(received) - FRAME_HEADER.size:
            raise NumericalError("arm worker response framing failed")
        response = strict_json_bytes(bytes(received[FRAME_HEADER.size:]))
        if set(response) not in ({"ok", "payload"}, {"ok", "error_type"}):
            raise NumericalError("arm worker response shape changed")
        if response["ok"] is not True:
            raise NumericalError("arm worker reported a sanitized failure")
        if not isinstance(response["payload"], dict):
            raise NumericalError("arm worker payload is invalid")
        return response["payload"], terminal_rss
    except BaseException:
        if status is None:
            _kill_process_group(pid)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        raise
    finally:
        os.close(read_fd)


def load_arm_progress(
    arm: dict[str, Any],
    contract: dict[str, Any],
    output_dir: Path,
    input_bindings: dict[str, str],
) -> tuple[int, list[float], dict[str, Any], list[dict[str, Any]]]:
    checkpoint_dir = output_dir / "checkpoints"
    receipt_dir = output_dir / "receipts"
    checkpoint_dir.mkdir(exist_ok=True); receipt_dir.mkdir(exist_ok=True)
    if (checkpoint_dir.is_symlink() or receipt_dir.is_symlink()
            or not checkpoint_dir.is_dir() or not receipt_dir.is_dir()):
        raise IntegrityError("immutable segment artifact directory is unsafe")
    latest: tuple[int, list[float], dict[str, Any], list[dict[str, Any]]] | None = None
    previous_accumulator = initialize_accumulator(
        arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
    )
    previous_commitments: list[dict[str, Any]] = []
    for segment_index in range(20):
        checkpoint_file = checkpoint_path(output_dir, arm["arm_id"], segment_index)
        receipt_file = segment_receipt_path(output_dir, arm["arm_id"], segment_index)
        if not checkpoint_file.exists() and not receipt_file.exists():
            break
        checkpoint, _receipt = validate_segment_receipt(
            output_dir, arm, segment_index, input_bindings, contract
        )
        latest = validate_checkpoint(checkpoint, arm, input_bindings, contract)
        _next, _state, current_accumulator, current_commitments = latest
        validate_accumulator_transition(
            previous_accumulator, current_accumulator, arm, segment_index
        )
        if current_commitments[:-1] != previous_commitments:
            raise IntegrityError("immutable checkpoint rewrote prior segment commitments")
        previous_accumulator = current_accumulator
        previous_commitments = current_commitments
    if latest is not None:
        return latest
    return (
        0,
        list(arm["initial_state"]),
        initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        ),
        [],
    )


def commit_segment_payload(
    arm: dict[str, Any],
    segment_payload: dict[str, Any],
    commitments: list[dict[str, Any]],
    input_bindings: dict[str, str],
    contract: dict[str, Any],
    output_dir: Path,
    segment_elapsed_seconds: float,
    terminal_child_peak_rss_bytes: int,
    total_started_ns: int,
    total_deadline_ns: int,
) -> dict[str, Any]:
    """Commit only a payload whose supervised child exited within every cap."""
    expected_segment_index = len(commitments)
    path = checkpoint_path(output_dir, arm["arm_id"], expected_segment_index)
    receipt_path = segment_receipt_path(
        output_dir, arm["arm_id"], expected_segment_index
    )
    if expected_segment_index == 0:
        if (commitments or path.exists() or path.is_symlink()
                or receipt_path.exists() or receipt_path.is_symlink()):
            raise IntegrityError("first segment has a predecessor checkpoint")
        previous_accumulator = initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
    else:
        predecessor_path = checkpoint_path(
            output_dir, arm["arm_id"], expected_segment_index - 1
        )
        if predecessor_path.is_symlink() or not predecessor_path.is_file():
            raise IntegrityError("next segment is missing its predecessor checkpoint")
        predecessor, _predecessor_receipt = validate_segment_receipt(
            output_dir, arm, expected_segment_index - 1, input_bindings, contract
        )
        predecessor_next, _state, previous_accumulator, predecessor_commitments = (
            validate_checkpoint(predecessor, arm, input_bindings, contract)
        )
        if (
            predecessor_next != expected_segment_index
            or predecessor_commitments != commitments
        ):
            raise IntegrityError("segment commitments disagree with retained predecessor")
    validate_segment_payload(
        segment_payload, arm, expected_segment_index, contract
    )
    validate_accumulator_transition(
        previous_accumulator, segment_payload["accumulator"], arm,
        expected_segment_index,
    )
    checkpoint = make_checkpoint(arm, segment_payload, commitments, input_bindings)
    # Verify the full candidate while the last committed checkpoint is still
    # untouched.  A malformed-but-framed worker payload can never destroy the
    # resumable boundary.
    validate_checkpoint(checkpoint, arm, input_bindings, contract)
    checkpoint_bytes = serialized_json(checkpoint)
    checkpoint_digest = sha256_bytes(checkpoint_bytes)
    checkpoint_size = len(checkpoint_bytes)
    commitment = checkpoint["segment_commitments"][-1]
    caps = contract["resource_caps_per_execution"]
    coordinator_rss = peak_rss_bytes_self()
    now_ns = time.monotonic_ns()
    if (not _finite_number(segment_elapsed_seconds)
            or segment_elapsed_seconds >= caps["max_wall_seconds_per_segment_attempt"]
            or not isinstance(terminal_child_peak_rss_bytes, int)
            or isinstance(terminal_child_peak_rss_bytes, bool)
            or not 0 <= terminal_child_peak_rss_bytes <= caps["max_peak_rss_bytes_per_process"]
            or coordinator_rss > caps["max_peak_rss_bytes_per_process"]
            or now_ns >= total_deadline_ns):
        raise ResourceLimitError("terminal segment wall or RSS cap failed before publication")
    # The checkpoint alone is ineligible.  It is published first, then a fully
    # resource-validated immutable parent receipt makes the pair eligible.
    if (directory_bytes(output_dir) + checkpoint_size > caps["max_output_bytes"]
            or shutil.disk_usage(output_dir).free
            < caps["minimum_free_disk_bytes"] + checkpoint_size):
        raise ResourceLimitError("checkpoint would violate projected output or disk reserve")
    atomic_create_json(path, checkpoint)
    now_ns = time.monotonic_ns()
    coordinator_rss = peak_rss_bytes_self()
    free_before = shutil.disk_usage(output_dir).free
    resource_validation = {
        "segment_elapsed_seconds": segment_elapsed_seconds,
        "terminal_child_peak_rss_bytes": terminal_child_peak_rss_bytes,
        "coordinator_peak_rss_bytes": coordinator_rss,
        "total_elapsed_seconds_before_publication": (
            now_ns - total_started_ns
        ) / NANOSECONDS_PER_SECOND,
        "output_bytes_projected": 0,
        "free_disk_bytes_before_publication": free_before,
    }
    receipt = {
        "schema": SEGMENT_RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "arm_id": arm["arm_id"],
        "segment_index": expected_segment_index,
        "input_bindings": input_bindings,
        "checkpoint_filename": path.name,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_size_bytes": checkpoint_size,
        "segment_payload_sha256": commitment["segment_payload_sha256"],
        "segment_chain_head_sha256": commitment["chain_head_sha256"],
        "parent_terminal_validation": "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS",
        "parent_resource_validation": resource_validation,
    }
    for _ in range(4):
        projected_bytes = directory_bytes(output_dir) + len(serialized_json(receipt))
        if resource_validation["output_bytes_projected"] == projected_bytes:
            break
        resource_validation["output_bytes_projected"] = projected_bytes
    receipt_size = len(serialized_json(receipt))
    if (now_ns >= total_deadline_ns
            or coordinator_rss > caps["max_peak_rss_bytes_per_process"]
            or projected_bytes > caps["max_output_bytes"]
            or free_before < caps["minimum_free_disk_bytes"] + receipt_size):
        raise ResourceLimitError("parent receipt terminal resource validation failed")
    atomic_create_json(receipt_path, receipt)
    validate_segment_receipt(
        output_dir, arm, expected_segment_index, input_bindings, contract
    )
    return checkpoint


def make_run_manifest(
    input_bindings: dict[str, str], runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "input_bindings": input_bindings,
        "runtime": runtime,
        "method": "CUSTOM_NEWTONIAN_RHS_VIA_SCIPY_SOLVE_IVP_DOP853_NO_REBOUND",
        "arm_ids": list(ARM_IDS),
        "sentinel_tracer_count": 32,
        "duration_years": 1_000_000.0,
        "sample_cadence_years": 50.0,
        "segment_years": 50_000.0,
        "official_execution_count": 1,
        "raw_trajectories_retained": False,
        "observed_or_private_data_used": False,
        "gpu_used": False,
    }


def validate_output_root_inventory(output_dir: Path, *, allow_result: bool) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise IntegrityError("DOP853 output root is unsafe")
    allowed = {
        "run_manifest.json", "checkpoints", "receipts", "failures", "execution.lock",
    }
    if (output_dir / "attempt_ledger.jsonl").exists():
        allowed.add("attempt_ledger.jsonl")
    if allow_result:
        allowed.add("result_v1.json")
    entries = list(output_dir.iterdir())
    if {entry.name for entry in entries} != allowed:
        raise IntegrityError("DOP853 output root contains an unexpected entry")
    for name in ("run_manifest.json", "attempt_ledger.jsonl", "result_v1.json"):
        path = output_dir / name
        if path.exists() and (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1):
            raise IntegrityError("DOP853 output file is unsafe")
    lock_path = output_dir / "execution.lock"
    if (lock_path.is_symlink() or not lock_path.is_file()
            or lock_path.stat().st_nlink != 1 or lock_path.stat().st_size != 0):
        raise IntegrityError("DOP853 execution lock file is unsafe")
    for name in ("checkpoints", "receipts", "failures"):
        path = output_dir / name
        if path.is_symlink() or not path.is_dir():
            raise IntegrityError("DOP853 output subdirectory is unsafe")
    failure_dir = output_dir / "failures"
    allowed_failure_names = {
        failure_receipt_filename(arm_id, segment_index, attempt_number)
        for arm_id in ARM_IDS for segment_index in range(20)
        for attempt_number in range(1, 4)
    }
    allowed_failure_names |= {f".{name}.pending" for name in allowed_failure_names}
    for path in failure_dir.iterdir():
        if (path.name not in allowed_failure_names or path.is_symlink()
                or not path.is_file() or path.stat().st_nlink != 1):
            raise IntegrityError("DOP853 failure receipt inventory changed")


def immutable_segment_inventory_sha256(output_dir: Path) -> str:
    rows: list[list[Any]] = []
    for directory_name in ("checkpoints", "receipts"):
        directory = output_dir / directory_name
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                raise IntegrityError("immutable segment inventory entry is unsafe")
            rows.append([
                f"{directory_name}/{path.name}", sha256_file(path), path.stat().st_size,
            ])
    return sha256_bytes(canonical_bytes(rows))


def validate_run_manifest(path: Path, expected: dict[str, Any]) -> None:
    if strict_json(path) != expected:
        raise IntegrityError("resumed run manifest differs from locked invocation")


def recover_or_validate_run_manifest(
    output_dir: Path, expected: dict[str, Any],
) -> None:
    path = output_dir / "run_manifest.json"
    pending = output_dir / ".run_manifest.json.pending"
    if path.exists() or path.is_symlink():
        if pending.exists() or pending.is_symlink() or strict_json(path) != expected:
            raise IntegrityError("DOP853 run manifest conflicts with its invocation")
        return
    numerical_absent = not any(
        candidate.exists() or candidate.is_symlink() for candidate in (
            output_dir / "attempt_ledger.jsonl", output_dir / "result_v1.json",
            output_dir / ".attempt_ledger.jsonl.pending",
            output_dir / ".result_v1.json.pending",
        )
    )
    for directory_name in ("checkpoints", "receipts", "failures"):
        directory = output_dir / directory_name
        if directory.exists() or directory.is_symlink():
            numerical_absent = numerical_absent and not directory.is_symlink() \
                and directory.is_dir() and not any(directory.iterdir())
    if pending.exists() or pending.is_symlink():
        if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
            raise IntegrityError("pending DOP853 run manifest is unsafe")
        try:
            candidate = strict_json(pending)
        except (ValueError, IntegrityError, OSError):
            candidate = None
        if candidate == expected:
            os.replace(pending, path); _fsync_directory(output_dir); return
        if not numerical_absent:
            raise IntegrityError("invalid DOP853 run manifest accompanies numerical state")
        pending.unlink(); _fsync_directory(output_dir)
    elif not numerical_absent:
        raise IntegrityError("DOP853 numerical state lacks its run manifest")
    atomic_create_json(path, expected)


def recover_unpublished_result_pending(output_dir: Path) -> dict[str, Any] | None:
    pending = output_dir / ".result_v1.json.pending"
    if not pending.exists() and not pending.is_symlink():
        return None
    if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
        raise IntegrityError("pending DOP853 result is unsafe")
    try:
        candidate = strict_json(pending)
    except (ValueError, IntegrityError, OSError):
        candidate = None
    pending.unlink(); _fsync_directory(output_dir)
    return candidate


def publish_or_recover_result(
    output_dir: Path, expected: dict[str, Any], candidate: dict[str, Any] | None,
    contract: dict[str, Any],
) -> None:
    if candidate is None:
        atomic_create_json(output_dir / "result_v1.json", expected); return
    candidate_core = dict(candidate); expected_core = dict(expected)
    provenance = candidate_core.pop("resource_provenance", None)
    expected_core.pop("resource_provenance", None)
    required = {
        "elapsed_seconds", "coordinator_peak_rss_bytes",
        "maximum_terminal_child_peak_rss_bytes", "output_bytes_before_result",
        "attempt_ledger_sha256", "immutable_segment_inventory_sha256",
    }
    caps = contract["resource_caps_per_execution"]
    if (candidate_core != expected_core or not isinstance(provenance, dict)
            or set(provenance) != required
            or not isinstance(provenance["elapsed_seconds"], (int, float))
            or isinstance(provenance["elapsed_seconds"], bool)
            or not math.isfinite(provenance["elapsed_seconds"])
            or not 0.0 <= provenance["elapsed_seconds"] < caps["max_wall_seconds_total"]
            or type(provenance["coordinator_peak_rss_bytes"]) is not int
            or not 0 <= provenance["coordinator_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or type(provenance["maximum_terminal_child_peak_rss_bytes"]) is not int
            or not 0 <= provenance["maximum_terminal_child_peak_rss_bytes"]
            <= caps["max_peak_rss_bytes_per_process"]
            or provenance["output_bytes_before_result"] != directory_bytes(output_dir)
            or provenance["attempt_ledger_sha256"]
            != sha256_file(output_dir / "attempt_ledger.jsonl")
            or provenance["immutable_segment_inventory_sha256"]
            != immutable_segment_inventory_sha256(output_dir)):
        raise IntegrityError("complete pending DOP853 result differs from recomputed output")
    atomic_create_json(output_dir / "result_v1.json", candidate)


def preflight_resources(output_parent: Path, contract: dict[str, Any]) -> None:
    caps = contract["resource_caps_per_execution"]
    free = shutil.disk_usage(output_parent).free
    if free < caps["minimum_free_disk_bytes"]:
        raise ResourceLimitError("minimum free-disk reserve is unavailable")
    if peak_rss_bytes_self() > caps["max_peak_rss_bytes_per_process"]:
        raise ResourceLimitError("coordinator already exceeds its RSS cap")


def input_binding_map(
    contract_path: Path,
    seed_manifest_path: Path,
    selection_path: Path,
    initial_states_path: Path,
    registration_path: Path,
    runner_path: Path,
) -> dict[str, str]:
    return {
        "contract_v1.json": sha256_file(contract_path),
        "seed_manifest_v1.json": sha256_file(seed_manifest_path),
        "selection_manifest_v1.json": sha256_file(selection_path),
        "initial_states_v1.json": sha256_file(initial_states_path),
        "registration_v1.json": sha256_file(registration_path),
        "run_independent.py": sha256_file(runner_path),
    }


def validate_inputs(
    contract_path: Path,
    seed_manifest_path: Path,
    selection_path: Path,
    initial_states_path: Path,
    registration_path: Path,
    runner_path: Path,
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], tuple[str, ...],
    dict[str, str], dict[str, Any]
]:
    paths = (
        contract_path, seed_manifest_path, selection_path,
        initial_states_path, registration_path, runner_path,
    )
    if any(path.is_symlink() for path in paths):
        raise IntegrityError("registered input path must not be a symlink")
    contract = strict_json(contract_path)
    validate_contract(contract)
    historical = contract["xp1_historical_binding"]
    historical_registration = contract_path.parent / historical["registration_path"]
    if (
        historical["role"]
        != "MOTIVATION_AND_DESIGN_LINEAGE_ONLY_NOT_A_TRAJECTORY_INPUT_OR_PRIOR"
        or historical["prior_trajectory_or_particle_metric_consumed"] is not False
        or historical_registration.is_symlink()
        or not historical_registration.is_file()
        or historical_registration.stat().st_nlink != 1
        or sha256_file(historical_registration) != historical["registration_sha256"]
    ):
        raise IntegrityError("XP1 read-only lineage binding changed")
    lineage = contract["xp2_v1_invalid_protocol_lineage"]
    v1_registration = contract_path.parent / lineage["v1_registration_path"]
    v1_diagnostic = contract_path.parent / lineage["v1_final_invalid_diagnostic_path"]
    if (
        lineage["role"]
        != "INVALID_PROTOCOL_DIAGNOSTIC_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT"
        or lineage["v1_output_or_outcome_consumed"] is not False
        or lineage["v1_scientific_classification_consumed"] is not False
        or lineage["protocol_repair_only_scientific_design_unchanged"] is not True
        or v1_registration.is_symlink() or not v1_registration.is_file()
        or v1_registration.stat().st_nlink != 1
        or sha256_file(v1_registration) != lineage["v1_registration_sha256"]
        or v1_diagnostic.is_symlink() or not v1_diagnostic.is_file()
        or v1_diagnostic.stat().st_nlink != 1
        or v1_diagnostic.stat().st_size != lineage["v1_final_invalid_diagnostic_size_bytes"]
        or sha256_file(v1_diagnostic) != lineage["v1_final_invalid_diagnostic_sha256"]
    ):
        raise IntegrityError("XP2-v1 invalid-protocol lineage binding changed")
    validate_v2_replay_lineage(contract, contract_path.parent)
    validate_v3_failed_startup_lineage(contract, contract_path.parent)
    science = contract["frozen_v1_scientific_design_inputs"]
    if (
        science["role"]
        != "EXACT_SCIENTIFIC_DESIGN_INPUT_REUSE_WITH_V4_EXECUTION_PROTOCOL_ONLY"
        or science["scientific_design_experiment_id"]
        != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or science["tracer_or_state_bytes_regenerated_for_v4"] is not False
        or science["v1_dynamics_or_outcomes_consumed"] is not False
        or science["seed_manifest_path"] != seed_manifest_path.name
        or science["selection_manifest_path"] != selection_path.name
        or science["initial_states_path"] != initial_states_path.name
        or sha256_file(seed_manifest_path) != science["seed_manifest_sha256"]
        or sha256_file(selection_path) != science["selection_manifest_sha256"]
        or sha256_file(initial_states_path) != science["initial_states_sha256"]
    ):
        raise IntegrityError("frozen v1 scientific-design input lineage changed")
    validate_registration(
        registration_path, contract_path, seed_manifest_path,
        initial_states_path, selection_path, runner_path,
    )
    # The seed manifest is consumed as a registered preoutput lineage binding;
    # exact tracer Cartesian rows are consumed only from initial_states_v1.json.
    seed_manifest = strict_json(seed_manifest_path)
    if (
        seed_manifest.get("schema") != "jx-xp2-local-seed-manifest/v1"
        or seed_manifest.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or seed_manifest.get("external_randomness_used") is not False
        or seed_manifest.get("outcome_or_prior_trajectory_used") is not False
    ):
        raise IntegrityError("seed-manifest lineage changed")
    selection = strict_json(selection_path)
    design_core_sha256 = sha256_bytes(canonical_bytes(contract["design_core"]))
    if (
        seed_manifest.get("design_core_sha256") != design_core_sha256
        or selection.get("tracer_rows_sha256")
        != seed_manifest.get("canonical_tracer_rows_sha256")
    ):
        raise IntegrityError("design, seed, tracer, and selection bindings disagree")
    initial_states = strict_json(initial_states_path)
    _all_arms, all_tracer_ids = validate_and_expand_initial_states(
        initial_states, initial_states_path, contract, None
    )
    selected_ids = validate_selection(selection, all_tracer_ids)
    arms, repeated_ids = validate_and_expand_initial_states(
        initial_states, initial_states_path, contract, selected_ids
    )
    if repeated_ids != all_tracer_ids:
        raise AssertionError("initial-state validation is not deterministic")
    bindings = input_binding_map(
        contract_path, seed_manifest_path, selection_path,
        initial_states_path, registration_path, runner_path,
    )
    runtime = validate_runtime(contract)
    return contract, arms, selected_ids, bindings, runtime


def capability_test() -> dict[str, Any]:
    """Protocol-only response; numerical capability requires final authority."""
    return {
        "capability": "NOT_RUN_REGISTRATION_REQUIRED",
        "method": "DOP853_CUSTOM_NEWTONIAN_RHS",
        "duration_years": 0.0,
        "dynamics_executed": False,
        "rebound_loaded": "rebound" in sys.modules,
    }


def _execute_locked(
    contract: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    selected_ids: Sequence[str],
    bindings: dict[str, str],
    runtime: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    caps = contract["resource_caps_per_execution"]
    preflight_resources(output_dir.parent, contract)
    run_manifest = make_run_manifest(bindings, runtime)
    pending_result_candidate: dict[str, Any] | None = None
    if resume:
        recover_or_validate_run_manifest(output_dir, run_manifest)
        if (output_dir / "result_v1.json").exists() or (output_dir / "result_v1.json").is_symlink():
            raise FileExistsError("independent execution already has a final result")
        pending_result_candidate = recover_unpublished_result_pending(output_dir)
        recover_pending_attempt_ledger(output_dir)
    else:
        atomic_create_json(output_dir / "run_manifest.json", run_manifest)
    for directory_name in ("checkpoints", "receipts", "failures"):
        directory = output_dir / directory_name
        if directory.is_symlink():
            raise IntegrityError("runner-owned output subdirectory is a symlink")
        directory.mkdir(exist_ok=True)
    validate_output_root_inventory(output_dir, allow_result=False)
    reconcile_attempt_ledger(output_dir, arms, bindings, contract)
    total_started_ns = time.monotonic_ns()
    total_deadline_ns = total_started_ns + int(
        caps["max_wall_seconds_total"] * NANOSECONDS_PER_SECOND
    )
    arm_results: list[dict[str, Any]] = []
    peak_child_rss = 0
    for arm_index, arm_id in enumerate(ARM_IDS, start=1):
        arm = arms[arm_id]
        next_segment_index, state, accumulator, commitments = load_arm_progress(
            arm, contract, output_dir, bindings
        )
        segment_count = contract["checkpoint_and_resume"]["segments_per_arm"]
        while next_segment_index < segment_count:
            preflight_resources(output_dir.parent, contract)
            if time.monotonic_ns() >= total_deadline_ns:
                raise ResourceLimitError("total execution wall deadline reached")
            if directory_bytes(output_dir) > caps["max_output_bytes"]:
                raise ResourceLimitError("output cap reached before segment attempt")
            ledger = read_attempt_ledger(output_dir)
            pending, _advanced = validate_ledger_against_checkpoints(
                ledger, output_dir, arms, bindings, contract
            )
            if pending is not None:
                raise IntegrityError("unreconciled attempt exists before segment start")
            attempt_number = next_attempt_number(ledger, arm_id, next_segment_index)
            maximum_attempts = contract["checkpoint_and_resume"][
                "maximum_attempts_per_segment"
            ]
            if attempt_number > maximum_attempts:
                error = ResourceLimitError("maximum attempts for segment exhausted")
                raise error
            sequence = len(ledger) + 1
            start_row = {
                "schema": ATTEMPT_SCHEMA,
                "sequence": sequence,
                "event": "SEGMENT_ATTEMPT_STARTED",
                "arm_id": arm_id,
                "segment_index": next_segment_index,
                "attempt_number_for_segment": attempt_number,
            }
            append_attempt_ledger(output_dir, start_row)
            segment_started_ns = time.monotonic_ns()
            try:
                segment_payload, child_rss = supervise_worker(
                    lambda arm=arm, state=state, accumulator=accumulator,
                    segment_index=next_segment_index: integrate_segment(
                        arm, state, accumulator, segment_index, contract
                    ),
                    caps["max_wall_seconds_per_segment_attempt"],
                    caps["max_peak_rss_bytes_per_process"],
                    output_dir,
                    caps["max_output_bytes"],
                    total_deadline_ns,
                    caps["watchdog_poll_seconds"],
                )
            except BaseException as error:
                failure_class = failure_class_for_error(error)
                receipt_path, receipt = publish_failure_receipt(
                    output_dir, start_row, failure_class
                )
                append_attempt_ledger(output_dir, {
                    "schema": ATTEMPT_SCHEMA,
                    "sequence": sequence + 1,
                    "event": "SEGMENT_ATTEMPT_FAILED",
                    "arm_id": arm_id,
                    "segment_index": next_segment_index,
                    "attempt_number_for_segment": attempt_number,
                    "failure_class": failure_class,
                    "fail_event_sha256": receipt["fail_event_sha256"],
                    "failure_receipt_filename": receipt_path.name,
                    "failure_receipt_sha256": sha256_file(receipt_path),
                })
                raise
            # Publication occurs in the coordinator only after the child has
            # exited successfully within wall/RSS/response/output caps.
            checkpoint = commit_segment_payload(
                arm, segment_payload, commitments, bindings, contract, output_dir,
                (time.monotonic_ns() - segment_started_ns) / NANOSECONDS_PER_SECOND,
                child_rss, total_started_ns, total_deadline_ns,
            )
            preflight_resources(output_dir.parent, contract)
            if time.monotonic_ns() >= total_deadline_ns:
                raise ResourceLimitError("total execution wall deadline reached after segment")
            if directory_bytes(output_dir) > caps["max_output_bytes"]:
                raise ResourceLimitError("output cap reached after checkpoint publication")
            peak_child_rss = max(peak_child_rss, child_rss)
            append_attempt_ledger(output_dir, {
                "schema": ATTEMPT_SCHEMA,
                "sequence": sequence + 1,
                "event": "SEGMENT_ATTEMPT_COMMITTED",
                "arm_id": arm_id,
                "segment_index": next_segment_index,
                "attempt_number_for_segment": attempt_number,
                "elapsed_seconds": (
                    time.monotonic_ns() - segment_started_ns
                ) / NANOSECONDS_PER_SECOND,
                "terminal_peak_rss_bytes": child_rss,
                "checkpoint_sha256": sha256_file(checkpoint_path(
                    output_dir, arm_id, next_segment_index
                )),
                "segment_receipt_sha256": sha256_file(segment_receipt_path(
                    output_dir, arm_id, next_segment_index
                )),
            })
            next_segment_index += 1
            state = state_from_hex(
                checkpoint["end_state_hex"], len(arm["logical_ids"]) * 6
            )
            accumulator = checkpoint["accumulator"]
            commitments = checkpoint["segment_commitments"]
            print(
                f"[{arm_id}] committed segment {next_segment_index}/{segment_count}",
                flush=True,
            )
        # A completed arm is finalized directly from its validated checkpoint;
        # it never consumes a fake worker attempt or retry allowance.
        final_checkpoint = strict_json(checkpoint_path(output_dir, arm_id, 19))
        validate_checkpoint(final_checkpoint, arm, bindings, contract)
        arm_results.append(finalize_arm_result(arm, final_checkpoint, contract))
        print(f"[arm {arm_index}/{len(ARM_IDS)}] {arm_id} complete", flush=True)
    final_ledger = read_attempt_ledger(output_dir)
    final_pending, _final_advanced = validate_ledger_against_checkpoints(
        final_ledger, output_dir, arms, bindings, contract
    )
    if final_pending is not None:
        raise IntegrityError("finalization found an unterminated attempt")
    validate_failure_receipt_bindings(final_ledger, output_dir)
    validate_output_root_inventory(output_dir, allow_result=False)
    active_pass = all(row["all_active_gates_pass"] for row in arm_results)
    preflight_resources(output_dir.parent, contract)
    if time.monotonic_ns() >= total_deadline_ns:
        raise ResourceLimitError("total execution wall deadline reached before finalization")
    semantic = {
        "schema": "jx-xp2-dop853-semantic/v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "selected_logical_ids": list(selected_ids),
        "arms": arm_results,
        "all_active_gates_pass": active_pass,
        "cross_method_gate_status": "PENDING_INDEPENDENT_REPLAY_VERIFIER",
    }
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "artifact_class": "LOCAL_SYNTHETIC_INDEPENDENT_NUMERICAL_SENTINEL_RESULT",
        "execution_label": OFFICIAL_EXECUTION_LABEL,
        "input_bindings": bindings,
        "runtime": runtime,
        "semantic": semantic,
        "semantic_sha256": sha256_bytes(canonical_bytes(semantic)),
        "resource_provenance": {
            "elapsed_seconds": (
                time.monotonic_ns() - total_started_ns
            ) / NANOSECONDS_PER_SECOND,
            "coordinator_peak_rss_bytes": peak_rss_bytes_self(),
            "maximum_terminal_child_peak_rss_bytes": peak_child_rss,
            "output_bytes_before_result": directory_bytes(output_dir),
            "attempt_ledger_sha256": sha256_file(output_dir / "attempt_ledger.jsonl"),
            "immutable_segment_inventory_sha256": immutable_segment_inventory_sha256(
                output_dir
            ),
        },
        "scientific_classification_emitted": False,
        "physical_validation_claim": False,
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }
    if directory_bytes(output_dir) + len(serialized_json(result)) > caps["max_output_bytes"]:
        raise ResourceLimitError("final result would exceed the output cap")
    revalidate_final_engineering_evidence()
    publish_or_recover_result(
        output_dir, result, pending_result_candidate, contract
    )
    revalidate_final_engineering_evidence()
    validate_output_root_inventory(output_dir, allow_result=True)
    return result


def acquire_output_execution_lock(output_dir: Path, *, create: bool) -> int:
    path = output_dir / "execution.lock"
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError:
        if create:
            raise
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    metadata = os.fstat(descriptor)
    try:
        on_disk = os.stat(path, follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        raise
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or not stat.S_ISREG(on_disk.st_mode)
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise IntegrityError("DOP853 execution lock is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError(
            "DOP853 output is still owned by a coordinator or inherited child"
        ) from None
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != metadata.st_dev
            or after_lock.st_ino != metadata.st_ino):
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)
        raise IntegrityError("DOP853 execution lock path changed during acquisition")
    return descriptor


def acquire_v2_b_guard(contract: dict[str, Any], package_root: Path) -> int:
    path = reject_symlink_components(
        package_root / contract["xp2_v2_invalid_replay_lineage"][
            "v2_b_execution_lock_path"
        ],
        "XP2-v2 B execution guard",
    )
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or path.is_symlink()
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise IntegrityError("XP2-v2 B execution guard is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError(
            "XP2-v2 B is still executing; v4 numerical output is forbidden"
        ) from None
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != metadata.st_dev
            or after_lock.st_ino != metadata.st_ino):
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)
        raise IntegrityError("XP2-v2 B guard path changed during acquisition")
    return descriptor


def acquire_v3_a_guard(contract: dict[str, Any], package_root: Path) -> int:
    path = reject_symlink_components(
        package_root / contract["xp2_v3_failed_startup_lineage"][
            "v3_a_execution_lock_path"
        ],
        "XP2-v3 A execution guard",
    )
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or path.is_symlink()
            or metadata.st_dev != on_disk.st_dev or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise IntegrityError("XP2-v3 A execution guard is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError(
            "XP2-v3 A is still executing; v4 numerical output is forbidden"
        ) from None
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != metadata.st_dev
            or after_lock.st_ino != metadata.st_ino):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise IntegrityError("XP2-v3 A guard path changed during acquisition")
    return descriptor


def acquire_engineering_evidence_guard(
    contract: dict[str, Any], package_root: Path, root_key: str, label: str,
) -> int:
    expected = expected_engineering_boundary_gate_v1().get(root_key)
    if (contract.get("engineering_boundary_gate_v1", {}).get(root_key) != expected
            or not isinstance(expected, str)):
        raise IntegrityError(f"{label} path changed before lock acquisition")
    root = reject_symlink_components(package_root / expected, label)
    path = reject_symlink_components(root / "execution.lock", f"{label} lock")
    if not root.is_dir():
        raise IntegrityError(f"{label} root is missing")
    descriptor = os.open(
        path, os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    )
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or metadata.st_dev != on_disk.st_dev
            or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor); raise IntegrityError(f"{label} lock binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError(f"{label} is still being produced") from None
    return descriptor


def execute(
    contract: dict[str, Any], arms: dict[str, dict[str, Any]],
    selected_ids: Sequence[str], bindings: dict[str, str], runtime: dict[str, Any],
    output_dir: Path, resume: bool,
) -> dict[str, Any]:
    if not resume:
        output_dir.mkdir()
    lock_fd = acquire_output_execution_lock(output_dir, create=not resume)
    try:
        return _execute_locked(
            contract, arms, selected_ids, bindings, runtime, output_dir, resume
        )
    finally:
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--initial-states", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capability-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _V3_A_GUARD_FD, _ENGINEERING_RUNNER_GUARD_FD
    global _ENGINEERING_SCRATCH_GUARD_FD
    arguments = build_parser().parse_args(argv)
    if arguments.capability_test:
        if any(value is not None for value in (
            arguments.contract, arguments.seed_manifest, arguments.selection_manifest,
            arguments.initial_states, arguments.registration, arguments.output_dir,
        )) or arguments.validate_only or arguments.resume:
            raise SystemExit("--capability-test must be used alone")
        print(json.dumps(capability_test(), sort_keys=True, allow_nan=False))
        return 0
    required = {
        "--contract": arguments.contract,
        "--seed-manifest": arguments.seed_manifest,
        "--selection-manifest": arguments.selection_manifest,
        "--initial-states": arguments.initial_states,
        "--registration": arguments.registration,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    runner_path = Path(__file__).resolve()
    raw_contract = strict_json(arguments.contract)
    if (raw_contract.get("schema") != CONTRACT_SCHEMA
            or raw_contract.get("experiment_id") != EXPERIMENT_ID):
        raise IntegrityError("contract identity changed before lineage locking")
    package_root = arguments.registration.resolve().parent
    v2_guard_fd = acquire_v2_b_guard(raw_contract, package_root)
    try:
        v3_guard_fd = acquire_v3_a_guard(raw_contract, package_root)
        _V3_A_GUARD_FD = v3_guard_fd
        try:
            _ENGINEERING_RUNNER_GUARD_FD = acquire_engineering_evidence_guard(
                raw_contract, package_root, "engineering_output_root",
                "engineering runner evidence",
            )
            _ENGINEERING_SCRATCH_GUARD_FD = acquire_engineering_evidence_guard(
                raw_contract, package_root, "engineering_verifier_scratch_root",
                "engineering scratch evidence",
            )
            contract, arms, selected_ids, bindings, runtime = validate_inputs(
                arguments.contract, arguments.seed_manifest, arguments.selection_manifest,
                arguments.initial_states, arguments.registration, runner_path,
            )
            if arguments.validate_only:
                if arguments.output_dir is not None or arguments.resume:
                    raise SystemExit("--validate-only does not accept output or resume")
                print(json.dumps({
                    "validation": "PASS",
                    "experiment_id": EXPERIMENT_ID,
                    "execution_label": OFFICIAL_EXECUTION_LABEL,
                    "arm_count": len(arms),
                    "sentinel_tracer_count": len(selected_ids),
                    "long_dynamics_executed": False,
                    "rebound_loaded": "rebound" in sys.modules,
                }, sort_keys=True, allow_nan=False))
                return 0
            if arguments.output_dir is None:
                raise SystemExit("--output-dir is required for execution")
            output = validate_output_root(
                arguments.output_dir, package_root, contract, arguments.resume
            )
            execute(contract, arms, selected_ids, bindings, runtime, output, arguments.resume)
        finally:
            if _ENGINEERING_RUNNER_GUARD_FD is not None:
                os.close(_ENGINEERING_RUNNER_GUARD_FD)
                _ENGINEERING_RUNNER_GUARD_FD = None
            if _ENGINEERING_SCRATCH_GUARD_FD is not None:
                os.close(_ENGINEERING_SCRATCH_GUARD_FD)
                _ENGINEERING_SCRATCH_GUARD_FD = None
            os.close(v3_guard_fd)
            _V3_A_GUARD_FD = None
    finally:
        os.close(v2_guard_fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IntegrityError, ResourceLimitError, NumericalError, ValueError, OSError) as error:
        print(f"independent sentinel failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(2)
