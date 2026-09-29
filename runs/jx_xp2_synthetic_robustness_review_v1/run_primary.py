#!/usr/bin/env python3
"""Registered, segmented MERCURIUS runner for the JX-XP2 synthetic screen.

The executable is deliberately self contained.  It consumes registered exact
binary64 Cartesian rows; it never imports XP1, the DOP853 implementation, or a
classification helper from the verifier.  ``--validate-only`` performs no
integration.  Numerical output is impossible until the complete package has a
valid pre-output registration.
"""

from __future__ import annotations

import argparse
import ctypes
import decimal
import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v4"
SCIENTIFIC_DESIGN_EXPERIMENT_ID = "jx-xp2-public-synthetic-robustness-v1"
CONTRACT_SCHEMA = "jx-xp2-robustness-contract/v3"
INITIAL_SCHEMA = "jx-xp2-barycentric-initial-states/v1"
REGISTRATION_SCHEMA = "jx-xp2-local-registration/v1"
ENGINEERING_REGISTRATION_SCHEMA = "jx-xp2-v4-engineering-registration/v1"
ENGINEERING_RECEIPT_SCHEMA = "jx-xp2-v4-engineering-boundary-verification/v1"
RUN_MANIFEST_SCHEMA = "jx-xp2-primary-run-manifest/v3"
ARM_RESULT_SCHEMA = "jx-xp2-mercurius-arm-result/v1"
RESULT_SCHEMA = "jx-xp2-primary-result/v3"
SEMANTIC_SCHEMA = "jx-xp2-primary-semantic/v3"
FAILURE_SCHEMA = "jx-xp2-primary-failure/v4"
CHECKPOINT_SCHEMA = "jx-xp2-mercurius-segment-receipt/v3"
SEGMENT_COMMIT_SCHEMA = "jx-xp2-mercurius-segment-parent-commit/v3"
ATTEMPT_SCHEMA = "jx-xp2-mercurius-segment-attempt/v4"

PRIMARY_ARM_IDS = (
    "M0",
    *(f"{case}-P{probe}" for case in ("CI01", "CI05", "CI09") for probe in range(8)),
)
AUDIT_ARM_IDS = tuple(f"AUDIT-{arm}" for arm in PRIMARY_ARM_IDS)
ALL_ARM_IDS = PRIMARY_ARM_IDS + AUDIT_ARM_IDS
AUDIT_TO_PRIMARY = {f"AUDIT-{arm}": arm for arm in PRIMARY_ARM_IDS}
LANDMARKS = (250_000.0, 500_000.0, 1_000_000.0)
CLASSIFICATION_HORIZONS = (500_000.0, 1_000_000.0)
THRESHOLDS = (30.0, 35.0, 40.0)
STREAM_SUFFIXES = ("LOG_A", "Q", "COS_I", "OMEGA", "OMEGA_ARGUMENT", "MEAN_ANOMALY")
LHS_DOMAIN = b"jx-xp2-lhs-u64/v1\0"
TRACER_DOMAIN = b"jx-xp2-canonical-tracer-design/v1\0"
EXPANDED_STATE_DOMAIN = b"jx-xp2-expanded-barycentric-state/v1\0"
CONFIG_INDEX_DOMAIN = b"jx-xp2-configuration-digest-index/v1\0"
STATE_DIGEST_DOMAIN = b"jx-xp2-mercurius-decoded-continuation-state/v3\0"
CONTINUATION_ARRAY_DOMAIN = b"jx-xp2-mercurius-decoded-array/v3\0"
ENDPOINT_DIGEST_DOMAIN = b"jx-xp2-mercurius-live-archive-endpoint/v1\0"
SAMPLE_STREAM_DOMAIN = b"jx-xp2-mercurius-sampled-state/v1\0"
SEGMENT_CHAIN_DOMAIN = b"jx-xp2-mercurius-semantic-segment-chain/v3\0"
RAW_ARTIFACT_INTEGRITY_DOMAIN = b"jx-xp2-mercurius-raw-artifact-integrity/v1\0"
INITIAL_SEGMENT_CHAIN = hashlib.sha256(SEGMENT_CHAIN_DOMAIN + b"GENESIS").hexdigest()
SEGMENT_SEMANTIC_FIELD_ORDER = (
    "arm_id", "configuration_id", "arm_class", "dt_years", "segment_index",
    "start_years", "end_years", "first_sample_index", "last_sample_index",
    "new_sample_count", "sample_count_total", "sampled_state_stream_sha256",
    "decoded_integrator_state_sha256", "tracker", "initial_active_invariants",
    "maximum_active_invariant_drifts", "landmarks",
)
SEGMENT_SEMANTIC_FIELDS = frozenset(SEGMENT_SEMANTIC_FIELD_ORDER)
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
RAW_CHECKPOINT_INTEGRITY_FIELDS = frozenset({
    "checkpoint_sha256", "checkpoint_size_bytes", "raw_checkpoint_integrity_only",
})
REBOUND_TREE_DOMAIN = b"jx-e2-rebound-python-sources/v1\0"
REDACTED_FAILURE_MESSAGE = "REDACTED_NON_SEMANTIC_FAILURE_DETAIL"
NANOSECONDS_PER_SECOND = 1_000_000_000
FAILURE_CLASSES = frozenset({
    "CHILD_EXIT_NONZERO", "CHILD_SIGNAL", "SEGMENT_TIMEOUT",
    "CHILD_RSS_LIMIT", "RECOVERED_UNCOMMITTED",
})
V2_DEFECT_EVIDENCE_SHA256 = "7cd515610718eaa9fac3159f988ef924c6df030cc8828719818b5b461789ff47"
V2_DEFECT_EVIDENCE_SIZE_BYTES = 5626
V3_FAILED_STARTUP_EVIDENCE_SHA256 = "eeb5ed87e05aab1ac0fa3cad68391bae1c850090dc48d4113b8b71c58c1dd473"
V3_FAILED_STARTUP_EVIDENCE_SIZE_BYTES = 4064
MAX_PRIMARY_CHECKPOINT_BYTES = 1_048_576
MAX_REBOUND_ALLOCATION_CAPACITY = 4096

_REBOUND_CACHE: tempfile.TemporaryDirectory[str] | None = None
_FAILURE_CONTEXT: dict[str, Any] | None = None
_EXECUTION_LOCK_FD: int | None = None
_V2_B_GUARD_FD: int | None = None
_V3_A_GUARD_FD: int | None = None
_ENGINEERING_RUNNER_GUARD_FD: int | None = None
_ENGINEERING_SCRATCH_GUARD_FD: int | None = None
_FINAL_ENGINEERING_EVIDENCE_SNAPSHOTS: list[Any] = []


class IntegrityError(RuntimeError):
    """A frozen input or stored artifact failed an integrity check."""


class NumericalError(RuntimeError):
    """Required finite numerical state was not available."""


class ResourceLimitError(RuntimeError):
    """A preregistered local resource cap was reached."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def serialized_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def held_tree_inventory(root: Path, lock_fd: int, label: str) -> list[list[Any]]:
    """Inventory one lock-owned tree through bound no-follow directory descriptors."""
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
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise IntegrityError(f"{label} root binding changed")
        lock_metadata = os.fstat(lock_fd)
        lock_on_disk = os.stat("execution.lock", dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1
                or lock_metadata.st_size != 0
                or lock_metadata.st_dev != lock_on_disk.st_dev
                or lock_metadata.st_ino != lock_on_disk.st_ino):
            raise IntegrityError(f"{label} is not owned by its held execution lock")
        rows: list[list[Any]] = []

        def digest_file_at(directory_fd: int, name: str, before: os.stat_result) -> str:
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
                if not name or "/" in name or name in {".", ".."}:
                    raise IntegrityError(f"{label} contains an unsafe entry name")
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
                                 digest_file_at(directory_fd, name, before)])
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    parsed = float(text)
    exact = decimal.Decimal(text)
    if not math.isfinite(parsed) or not exact.is_finite() or (parsed == 0.0 and exact != 0):
        raise ValueError("non-finite or underflowed JSON number")
    return parsed


def _reject_constant(text: str) -> None:
    raise ValueError(f"non-standard JSON constant: {text}")


def strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError(f"unsafe JSON file: {path}")
    parsed = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
        parse_float=_finite_float, parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise IntegrityError("JSON root must be an object")
    return parsed


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
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


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise FileExistsError(f"stale pending artifact: {pending}")
    payload = serialized_json(value)
    try:
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256_file(pending) != sha256_bytes(payload) or pending.stat().st_size != len(payload):
            raise IntegrityError("pending JSON verification failed")
        os.replace(pending, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        raise


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    """Publish one logical append as an atomic whole-ledger replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1):
        raise IntegrityError("attempt ledger is unsafe")
    previous = path.read_bytes() if path.exists() else b""
    if previous and not previous.endswith(b"\n"):
        raise IntegrityError("attempt ledger has a torn tail")
    payload = previous + canonical_bytes(row) + b"\n"
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise IntegrityError("stale pending attempt ledger requires recovery")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write to pending attempt ledger")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if pending.read_bytes() != payload:
            raise IntegrityError("pending attempt ledger verification failed")
        os.replace(pending, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Preserve a complete pending extension for deterministic resume; an
        # incomplete pending file is also preserved as fail-closed evidence.
        raise


def recover_pending_ledger(
    path: Path, *, execution_label: str, registration_sha256: str,
    output_root: Path,
) -> None:
    pending = path.with_name(f".{path.name}.pending")
    if not pending.exists() and not pending.is_symlink():
        return
    if (pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1
            or (path.exists() and (path.is_symlink() or not path.is_file()
                                   or path.stat().st_nlink != 1))):
        raise IntegrityError("pending attempt ledger is unsafe")
    previous = path.read_bytes() if path.exists() else b""
    candidate = pending.read_bytes()
    if previous and not previous.endswith(b"\n"):
        raise IntegrityError("published attempt ledger has a torn tail")
    def discard_incomplete() -> None:
        pending.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    # Atomic replacement writes the old bytes first.  A kill during that copy
    # may leave any byte prefix of the still-authoritative old ledger.
    if previous.startswith(candidate):
        discard_incomplete(); return
    if not candidate.startswith(previous):
        raise IntegrityError("pending attempt ledger diverges from published bytes")
    suffix = candidate[len(previous):]
    if not suffix.endswith(b"\n"):
        discard_incomplete(); return
    if suffix.count(b"\n") != 1:
        raise IntegrityError("pending attempt ledger contains multiple complete rows")
    try:
        row = json.loads(suffix[:-1], object_pairs_hook=_unique_object,
                         parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError):
        raise IntegrityError("complete pending attempt ledger row is invalid")
    expected_sequence = len(previous.splitlines()) + 1
    if (not isinstance(row, dict) or row.get("schema") != ATTEMPT_SCHEMA
            or type(row.get("sequence")) is not int
            or row.get("sequence") != expected_sequence
            or row.get("event") not in ("START", "PASS", "FAIL")
            or canonical_bytes(row) + b"\n" != suffix):
        raise IntegrityError("complete pending attempt ledger row diverges from protocol")
    previous_rows = read_jsonl(path)
    candidate_rows = [*previous_rows, row]
    validate_attempt_ledger(candidate_rows, execution_label, registration_sha256)
    if row["event"] == "PASS":
        receipt = load_completed_segment(
            output_root / "arms" / row["arm_id"], row["segment_index"]
        )
        if (row["attempt_index"] != receipt["provenance"]["attempt_index"]
                or row["segment_chain_head"] != receipt["segment_chain_head"]):
            raise IntegrityError("pending PASS does not bind its selected parent commit")
    elif row["event"] == "FAIL":
        start = next(
            (candidate for candidate in previous_rows
             if candidate["event"] == "START"
             and candidate["arm_id"] == row["arm_id"]
             and candidate["segment_index"] == row["segment_index"]
             and candidate["attempt_index"] == row["attempt_index"]),
            None,
        )
        if start is None:
            raise IntegrityError("pending FAIL lacks its exact START")
        failure_dir = output_root / "failures"
        receipt_path = failure_dir / row["failure_receipt_filename"]
        receipt = strict_json(receipt_path)
        if receipt_path.read_bytes() != serialized_json(receipt):
            raise IntegrityError("pending FAIL receipt is noncanonical")
        validate_failure_receipt_payload(
            receipt, start, row["failure_receipt_filename"], failure_dir
        )
        if row != failure_terminal_row(
            receipt, row["failure_receipt_filename"], sha256_file(receipt_path),
            row["sequence"],
        ):
            raise IntegrityError("pending FAIL does not bind its permanent receipt")
    os.replace(pending, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def acquire_execution_lock(output_root: Path, *, create: bool) -> int:
    path = output_root / "execution.lock"
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError:
        if create:
            raise
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise IntegrityError("new execution lock already exists") from None
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
        raise IntegrityError("execution lock file is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError("execution tree is still owned by a coordinator or child") from None
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != metadata.st_dev
            or after_lock.st_ino != metadata.st_ino):
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)
        raise IntegrityError("execution lock path changed during acquisition")
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


def validate_inherited_v2_b_guard(
    descriptor: int, contract: dict[str, Any], package_root: Path,
) -> None:
    path = package_root / contract["xp2_v2_invalid_replay_lineage"][
        "v2_b_execution_lock_path"
    ]
    try:
        inherited = os.fstat(descriptor)
        on_disk = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise IntegrityError("internal worker did not inherit the XP2-v2 B guard") from exc
    if (not stat.S_ISREG(inherited.st_mode) or inherited.st_nlink != 1
            or inherited.st_size != 0 or path.is_symlink()
            or inherited.st_dev != on_disk.st_dev or inherited.st_ino != on_disk.st_ino):
        raise IntegrityError("internal worker XP2-v2 B guard binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise IntegrityError("internal worker XP2-v2 B guard is not inherited") from exc
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != inherited.st_dev
            or after_lock.st_ino != inherited.st_ino):
        raise IntegrityError("internal worker XP2-v2 B guard path changed")


def acquire_v3_a_guard(contract: dict[str, Any], package_root: Path) -> int:
    path = reject_symlink_components(
        package_root / contract["xp2_v3_failed_startup_lineage"][
            "v3_a_execution_lock_path"
        ],
        "XP2-v3 A execution guard",
    )
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
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
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)
        raise IntegrityError("XP2-v3 A guard path changed during acquisition")
    return descriptor


def validate_inherited_v3_a_guard(
    descriptor: int, contract: dict[str, Any], package_root: Path,
) -> None:
    path = package_root / contract["xp2_v3_failed_startup_lineage"][
        "v3_a_execution_lock_path"
    ]
    try:
        inherited = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise IntegrityError("internal worker did not inherit the XP2-v3 A guard") from exc
    if (not stat.S_ISREG(inherited.st_mode) or inherited.st_nlink != 1
            or inherited.st_size != 0 or path.is_symlink()
            or inherited.st_dev != on_disk.st_dev or inherited.st_ino != on_disk.st_ino):
        raise IntegrityError("internal worker XP2-v3 A guard binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise IntegrityError("internal worker XP2-v3 A guard is not inherited") from exc
    after_lock = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0 or after_lock.st_dev != inherited.st_dev
            or after_lock.st_ino != inherited.st_ino):
        raise IntegrityError("internal worker XP2-v3 A guard path changed")


def acquire_engineering_evidence_guard(
    contract: dict[str, Any], package_root: Path, root_key: str, label: str,
) -> int:
    gate = contract.get("engineering_boundary_gate_v1", {})
    expected = expected_engineering_boundary_gate_v1().get(root_key)
    if gate.get(root_key) != expected or not isinstance(expected, str):
        raise IntegrityError(f"{label} path changed before lock acquisition")
    root = reject_symlink_components(package_root / expected, label)
    if not root.is_dir():
        raise IntegrityError(f"{label} root is missing")
    path = reject_symlink_components(root / "execution.lock", f"{label} lock")
    flags = os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or metadata.st_dev != on_disk.st_dev
            or metadata.st_ino != on_disk.st_ino):
        os.close(descriptor)
        raise IntegrityError(f"{label} lock binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ResourceLimitError(f"{label} is still being produced") from None
    return descriptor


def validate_inherited_engineering_evidence_guard(
    descriptor: int, contract: dict[str, Any], package_root: Path,
    root_key: str, label: str,
) -> None:
    expected = expected_engineering_boundary_gate_v1()[root_key]
    path = reject_symlink_components(
        package_root / expected / "execution.lock", f"{label} inherited lock",
    )
    metadata = os.fstat(descriptor); on_disk = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size != 0 or metadata.st_dev != on_disk.st_dev
            or metadata.st_ino != on_disk.st_ino):
        raise IntegrityError(f"{label} inherited lock binding changed")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise IntegrityError(f"{label} inherited lock is not held") from exc


def write_failure_receipt(_error: BaseException) -> None:
    """Deliberately do nothing outside the coordinator terminal protocol.

    V1 let arbitrary processes create timestamp-named receipts and therefore
    could neither prove that each FAIL had a receipt nor bind a receipt back to
    one ledger event.  V2 permits only ``append_failure_terminal`` to publish a
    primary failure receipt.  A coordinator-level exception leaves an
    incomplete run; any open START is classified and receipted on resume.
    """


def failure_receipt_filename(arm_id: str, segment_index: int, attempt_index: int) -> str:
    if arm_id not in ALL_ARM_IDS:
        raise IntegrityError("failure receipt arm changed")
    if not 0 <= segment_index < 20 or not 1 <= attempt_index <= 3:
        raise IntegrityError("failure receipt attempt identity changed")
    return f"failure_{arm_id}_segment_{segment_index:02d}_attempt_{attempt_index:02d}.json"


def failure_event_core(
    *, execution_label: str, arm_id: str, segment_index: int,
    attempt_index: int, start_sequence: int, return_code: int | None,
    failure_class: str, complete_uncommitted_attempt: dict[str, Any] | None = None,
    complete_semantic_sha256: str | None = None,
    complete_decoded_state_sha256: str | None = None,
) -> dict[str, Any]:
    if failure_class not in FAILURE_CLASSES:
        raise IntegrityError("unknown terminal failure class")
    coherent = (
        (failure_class == "RECOVERED_UNCOMMITTED" and return_code is None)
        or (failure_class == "CHILD_EXIT_NONZERO" and type(return_code) is int
            and return_code > 0)
        or (failure_class == "CHILD_SIGNAL" and type(return_code) is int
            and return_code < 0)
        or (failure_class in ("SEGMENT_TIMEOUT", "CHILD_RSS_LIMIT")
            and type(return_code) is int and return_code == -signal.SIGKILL)
    )
    if not coherent:
        raise IntegrityError("terminal failure class/return-code mismatch")
    if complete_uncommitted_attempt is not None:
        validate_complete_attempt_evidence_shape(complete_uncommitted_attempt)
        if complete_semantic_sha256 is not None or complete_decoded_state_sha256 is not None:
            raise IntegrityError("duplicate complete-attempt event binding")
        complete_semantic_sha256 = complete_uncommitted_attempt[
            "semantic_segment_payload_sha256"
        ]
        complete_decoded_state_sha256 = complete_uncommitted_attempt[
            "decoded_integrator_state_sha256"
        ]
    if ((complete_semantic_sha256 is None) != (complete_decoded_state_sha256 is None)
            or (complete_semantic_sha256 is not None and (
                not valid_sha256(complete_semantic_sha256)
                or not valid_sha256(complete_decoded_state_sha256)
            ))):
        raise IntegrityError("complete-attempt event digests changed")
    return {
        "schema": FAILURE_SCHEMA, "event": "FAIL", "execution_label": execution_label,
        "arm_id": arm_id, "start_sequence": start_sequence,
        "segment_index": segment_index, "attempt_index": attempt_index,
        "return_code": return_code, "failure_class": failure_class,
        "complete_uncommitted_attempt_semantic_sha256": (
            complete_semantic_sha256
        ),
        "complete_uncommitted_attempt_decoded_state_sha256": (
            complete_decoded_state_sha256
        ),
    }


def failure_event_sha256(core: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(core))


def failure_quarantine_inventory(failure_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in sorted(names):
        path = failure_dir / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.resolve().parent != failure_dir.resolve()):
            raise IntegrityError("failure quarantine artifact is unsafe")
        inventory.append({
            "filename": name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return inventory


def build_failure_receipt(
    start_row: dict[str, Any], core: dict[str, Any],
    quarantined_artifacts: Sequence[dict[str, Any]],
    complete_uncommitted_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": FAILURE_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "execution_label": core["execution_label"], "arm_id": core["arm_id"],
        "segment_index": core["segment_index"],
        "attempt_index": core["attempt_index"],
        "failure_class": core["failure_class"], "return_code": core["return_code"],
        "start_sequence": start_row["sequence"],
        "predecessor_segment_chain_head": start_row["predecessor_segment_chain_head"],
        "input_key_sha256": start_row["input_key_sha256"],
        "fail_event_sha256": failure_event_sha256(core),
        "complete_uncommitted_attempt": complete_uncommitted_attempt,
        "quarantined_artifacts": list(quarantined_artifacts),
        "message": REDACTED_FAILURE_MESSAGE, "authorizes_analysis": False,
    }


def failure_terminal_row(
    receipt: dict[str, Any], filename: str, receipt_sha256: str, sequence: int,
) -> dict[str, Any]:
    core = failure_event_core(
        execution_label=receipt["execution_label"], arm_id=receipt["arm_id"],
        segment_index=receipt["segment_index"], attempt_index=receipt["attempt_index"],
        start_sequence=receipt["start_sequence"],
        return_code=receipt["return_code"], failure_class=receipt["failure_class"],
        complete_uncommitted_attempt=receipt["complete_uncommitted_attempt"],
    )
    if receipt["fail_event_sha256"] != failure_event_sha256(core):
        raise IntegrityError("failure receipt event digest changed")
    return {
        "schema": ATTEMPT_SCHEMA, "sequence": sequence, "event": "FAIL",
        "execution_label": receipt["execution_label"], "arm_id": receipt["arm_id"],
        "segment_index": receipt["segment_index"],
        "attempt_index": receipt["attempt_index"], "return_code": receipt["return_code"],
        "failure_class": receipt["failure_class"],
        "complete_uncommitted_attempt_semantic_sha256": core[
            "complete_uncommitted_attempt_semantic_sha256"
        ],
        "complete_uncommitted_attempt_decoded_state_sha256": core[
            "complete_uncommitted_attempt_decoded_state_sha256"
        ],
        "fail_event_sha256": receipt["fail_event_sha256"],
        "failure_receipt_filename": filename,
        "failure_receipt_sha256": receipt_sha256,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("attempt ledger is unsafe")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise IntegrityError("attempt ledger has a nonterminated tail")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(line, object_pairs_hook=_unique_object,
                             parse_float=_finite_float, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid attempt ledger row {line_number}") from exc
        if not isinstance(row, dict):
            raise IntegrityError("attempt ledger row is not an object")
        if (row.get("schema") != ATTEMPT_SCHEMA
                or type(row.get("sequence")) is not int
                or row.get("sequence") != line_number):
            raise IntegrityError("attempt ledger schema or sequence changed")
        if row.get("event") not in ("START", "PASS", "FAIL"):
            raise IntegrityError("attempt ledger event changed")
        if canonical_bytes(row) != line:
            raise IntegrityError("attempt ledger row is noncanonical")
        rows.append(row)
    return rows


def read_jsonl_prefix(path: Path, stop_sequence: int) -> list[dict[str, Any]]:
    """Read only complete ledger records through an immutable sequence."""
    if (not isinstance(stop_sequence, int) or isinstance(stop_sequence, bool)
            or stop_sequence < 1 or path.is_symlink() or not path.is_file()
            or path.stat().st_nlink != 1):
        raise IntegrityError("attempt ledger prefix request is unsafe")
    payload = path.read_bytes()
    complete = payload.split(b"\n")
    if complete and complete[-1] == b"":
        complete.pop()
    elif complete:
        # A concurrent append can expose an incomplete trailing record.  It is
        # deliberately outside the immutable prefix and must never be parsed.
        complete.pop()
    if len(complete) < stop_sequence:
        raise IntegrityError("attempt ledger immutable prefix is incomplete")
    rows: list[dict[str, Any]] = []
    for sequence, raw in enumerate(complete[:stop_sequence], start=1):
        try:
            row = json.loads(raw, object_pairs_hook=_unique_object,
                             parse_float=_finite_float, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid attempt ledger prefix row {sequence}") from exc
        if (not isinstance(row, dict) or row.get("schema") != ATTEMPT_SCHEMA
                or type(row.get("sequence")) is not int
                or row.get("sequence") != sequence
                or row.get("event") not in ("START", "PASS", "FAIL")):
            raise IntegrityError("attempt ledger prefix identity changed")
        if canonical_bytes(row) != raw:
            raise IntegrityError("attempt ledger prefix row is noncanonical")
        rows.append(row)
    return rows


def binary64_from_hex(value: str) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise IntegrityError("binary64 hex width changed")
    try:
        parsed = struct.unpack(">d", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as exc:
        raise IntegrityError("invalid binary64 hex") from exc
    if not math.isfinite(parsed):
        raise IntegrityError("non-finite binary64 input")
    return parsed


def binary64_hex(value: float) -> str:
    if not math.isfinite(value):
        raise NumericalError("cannot encode non-finite binary64")
    return struct.pack(">d", float(value)).hex()


def unpack_state(value: str) -> list[float]:
    if not isinstance(value, str) or len(value) != 96:
        raise IntegrityError("packed Cartesian state width changed")
    return [binary64_from_hex(value[index:index + 16]) for index in range(0, 96, 16)]


def pack_state(value: Sequence[float]) -> str:
    if len(value) != 6:
        raise IntegrityError("Cartesian state must have six components")
    return "".join(binary64_hex(component) for component in value)


def require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise IntegrityError(f"{label} fields changed")


def derive_seed(domain: str, design_digest: str, label: str, counter: int) -> bytes:
    domain_bytes = domain.encode("ascii")
    label_bytes = label.encode("ascii")
    payload = (
        len(domain_bytes).to_bytes(4, "big") + domain_bytes + bytes.fromhex(design_digest)
        + len(label_bytes).to_bytes(4, "big") + label_bytes + counter.to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()


def lhs_values(seed: bytes) -> tuple[list[int], list[float]]:
    if len(seed) != 16:
        raise IntegrityError("LHS seed length changed")
    counter = 0

    def next_u64() -> int:
        nonlocal counter
        word = int.from_bytes(
            hashlib.sha256(LHS_DOMAIN + seed + counter.to_bytes(8, "big")).digest()[:8],
            "big",
        )
        counter += 1
        return word

    permutation = list(range(16))
    modulus = 1 << 64
    for index in range(15, 0, -1):
        divisor = index + 1
        limit = modulus - modulus % divisor
        while True:
            word = next_u64()
            if word < limit:
                break
        other = word % divisor
        permutation[index], permutation[other] = permutation[other], permutation[index]
    values = [(stratum + (next_u64() >> 11) / float(1 << 53)) / 16.0
              for stratum in permutation]
    return permutation, values


def validate_seed_manifest(contract: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest = strict_json(path)
    policy = contract["seed_policy"]
    if (
        manifest.get("schema") != "jx-xp2-local-seed-manifest/v1"
        or manifest.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or manifest.get("design_core_sha256") != policy["design_core_sha256"]
        or manifest.get("domain_ascii") != policy["domain_ascii"]
        or manifest.get("derivation") != policy["stream_formula"]
        or manifest.get("seed_bytes_used") != 16
        or manifest.get("encoding") != "LOWERCASE_HEX_BIG_ENDIAN"
        or manifest.get("external_randomness_used") is not False
        or manifest.get("outcome_or_prior_trajectory_used") is not False
        or manifest.get("override_allowed") is not False
    ):
        raise IntegrityError("seed manifest identity or policy changed")
    labels = [f"LHS_BLOCK_{block}_{suffix}" for block in range(8) for suffix in STREAM_SUFFIXES]
    rows = manifest.get("streams")
    if not isinstance(rows, list) or len(rows) != 48:
        raise IntegrityError("seed stream cardinality changed")
    seeds: dict[str, bytes] = {}
    for label, row in zip(labels, rows, strict=True):
        require_keys(row, {"stream_label", "counter", "seed_hex_128"}, "seed row")
        expected = derive_seed(manifest["domain_ascii"], manifest["design_core_sha256"], label, 0)[:16]
        if row["stream_label"] != label or row["counter"] != 0 or row["seed_hex_128"] != expected.hex():
            raise IntegrityError("seed derivation mismatch")
        seeds[label] = expected
    return manifest, seeds


def make_tracers(contract: dict[str, Any], seeds: dict[str, bytes]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    test_vector: tuple[list[int], list[float]] | None = None
    for block in range(8):
        dimensions: dict[str, list[float]] = {}
        for suffix in STREAM_SUFFIXES:
            permutation, values = lhs_values(seeds[f"LHS_BLOCK_{block}_{suffix}"])
            dimensions[suffix] = values
            if block == 0 and suffix == "LOG_A":
                test_vector = permutation, values
        for index in range(16):
            a_au = math.exp(math.log(150.0) + dimensions["LOG_A"][index]
                            * (math.log(800.0) - math.log(150.0)))
            q_au = 35.0 + 45.0 * dimensions["Q"][index]
            eccentricity = 1.0 - q_au / a_au
            inclination = math.acos(math.cos(math.radians(40.0))
                                    + dimensions["COS_I"][index]
                                    * (1.0 - math.cos(math.radians(40.0))))
            row = {
                "logical_id": f"XP2-B{block:02d}-T{index:02d}",
                "block_index": block, "index_within_block": index,
                "a_AU": a_au, "q_AU": q_au, "e": eccentricity,
                "i_rad": inclination,
                "Omega_rad": 2.0 * math.pi * dimensions["OMEGA"][index],
                "omega_rad": 2.0 * math.pi * dimensions["OMEGA_ARGUMENT"][index],
                "M_rad": 2.0 * math.pi * dimensions["MEAN_ANOMALY"][index],
            }
            rows.append(row)
            canonical.append({
                "logical_id": row["logical_id"], "block_index": block,
                "index_within_block": index, "a_AU_hex": a_au.hex(),
                "q_AU_hex": q_au.hex(), "e_hex": eccentricity.hex(),
                "i_rad_hex": inclination.hex(), "Omega_rad_hex": row["Omega_rad"].hex(),
                "omega_rad_hex": row["omega_rad"].hex(), "M_rad_hex": row["M_rad"].hex(),
            })
    policy = contract["seed_policy"]
    if test_vector is None or test_vector[0] != policy["block_0_log_a_permutation_test_vector"]:
        raise IntegrityError("LHS permutation test vector mismatch")
    if [value.hex() for value in test_vector[1][:4]] != policy["block_0_log_a_first_four_lhs_float_hex"]:
        raise IntegrityError("LHS jitter test vector mismatch")
    digest = sha256_bytes(TRACER_DOMAIN + canonical_bytes(canonical))
    if digest != policy["canonical_rows_sha256"]:
        raise IntegrityError("canonical tracer digest mismatch")
    return rows, digest


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
    lineage = contract.get("xp2_v3_failed_startup_lineage")
    if lineage != expected:
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
    raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
    rows = []
    for line in raw_lines:
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


def validate_contract(contract: dict[str, Any], path: Path) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("experiment_id") != EXPERIMENT_ID:
        raise IntegrityError("contract identity changed")
    if contract.get("claim_ceiling") != "SYNTHETIC_1MYR_8_ORIENTATION_RESPONSE_WITH_DOP853_SENTINEL_ONLY":
        raise IntegrityError("claim ceiling changed")
    design_digest = sha256_bytes(canonical_bytes(contract["design_core"]))
    if design_digest != contract["seed_policy"]["design_core_sha256"]:
        raise IntegrityError("design-core digest mismatch")
    core = contract["design_core"]
    if tuple(core["primary_arm_ids"]) != PRIMARY_ARM_IDS or tuple(core["audit_arm_ids"]) != AUDIT_ARM_IDS:
        raise IntegrityError("arm matrix changed")
    tracer = core["tracer_design"]
    dynamics = core["dynamics"]
    if (tracer["block_count"], tracer["tracers_per_block"], tracer["total_tracers"]) != (8, 16, 128):
        raise IntegrityError("tracer design cardinality changed")
    if (
        dynamics["duration_years"] != 1_000_000.0
        or tuple(dynamics["landmark_years"]) != LANDMARKS
        or dynamics["sample_cadence_years"] != 50.0
        or dynamics["sample_count_including_t0"] != 20_001
        or dynamics["segment_years"] != 50_000.0
        or dynamics["segment_count"] != 20
        or dynamics["primary_dt_years"] != 0.125
        or dynamics["audit_dt_years"] != 0.0625
    ):
        raise IntegrityError("dynamics lock changed")
    if contract["resource_caps_per_execution"]["workers"] != 4:
        raise IntegrityError("worker count changed")
    if contract["initial_state_policy"]["path"] != "initial_states_v1.json":
        raise IntegrityError("initial-state path changed")
    checkpoint = contract["checkpoint_and_resume"]
    result_policy = contract["result_policy"]
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
        != list(SEGMENT_SEMANTIC_FIELD_ORDER)
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
        or result_policy.get("primary_failure_receipt_schema") != FAILURE_SCHEMA
        or result_policy.get("primary_attempt_ledger_schema") != ATTEMPT_SCHEMA
        or result_policy.get("primary_checkpoint_receipt_schema") != CHECKPOINT_SCHEMA
        or result_policy.get("primary_segment_commit_schema") != SEGMENT_COMMIT_SCHEMA
        or result_policy.get("primary_result_schema") != RESULT_SCHEMA
        or result_policy.get("primary_semantic_schema") != SEMANTIC_SCHEMA
        or
        checkpoint.get("dop853_failure_receipt_v2")
        != "DETERMINISTIC_ARM_SEGMENT_ATTEMPT_FILENAME; RECEIPT_FSYNCED_BEFORE_SEGMENT_ATTEMPT_FAILED_LEDGER_PUBLICATION; START_SEQUENCE_BOUND_EVENT_DIGEST; EXACT_CLOSED_FAILURE_TO_RECEIPT_FILENAME_SHA256_AND_CLASS_BIJECTION; COMPLETE_ORPHAN_RECEIPT_RECONCILED; PARTIAL_UNPUBLISHED_RECEIPT_BECOMES_INTERRUPTED_ATTEMPT; MISSING_EXTRA_DUPLICATE_OR_TAMPERED_RECEIPT_REJECTED"
        or checkpoint.get("dop853_failure_classes_v2") != [
            "InterruptedAttempt", "IntegrityError", "NumericalError",
            "ResourceLimitError", "UnexpectedFailure",
        ]
        or result_policy.get("dop853_failure_receipt_bijection_required") is not True
        or result_policy.get("dop853_attempt_ledger_schema")
        != "jx-xp2-dop853-attempt-ledger-row/v2"
        or result_policy.get("dop853_failure_receipt_schema")
        != "jx-xp2-dop853-failure/v2"
    ):
        raise IntegrityError("v4 semantic/raw or DOP853 failure protocol declaration changed")
    historical = contract["xp1_historical_binding"]
    registration = (path.parent / historical["registration_path"]).resolve()
    if registration.is_symlink() or not registration.is_file() or registration.stat().st_nlink != 1:
        raise IntegrityError("XP1 historical registration missing")
    if sha256_file(registration) != historical["registration_sha256"]:
        raise IntegrityError("XP1 historical registration digest mismatch")
    lineage = contract.get("xp2_v1_invalid_protocol_lineage")
    if not isinstance(lineage, dict) or lineage != {
        "role": "INVALID_PROTOCOL_DIAGNOSTIC_ONLY_NOT_A_TRAJECTORY_INPUT_OR_SCIENTIFIC_RESULT",
        "v1_registration_path": "../jx_xp2_robustness_v1/registration_v1.json",
        "v1_registration_sha256": "7ab012862513df6d597d58270bdd1abab33cbea25a0f63e018279f8705aaafba",
        "v1_final_invalid_diagnostic_path": "../jx_xp2_runs_v1/v1_final_invalid_protocol_diagnostic_receipt.json",
        "v1_final_invalid_diagnostic_sha256": "231ccc7170bd7418eb258d648fd243beaf21f296bd9ed3f9ed8981e78f2520d6",
        "v1_final_invalid_diagnostic_size_bytes": 3375,
        "v1_output_or_outcome_consumed": False,
        "v1_scientific_classification_consumed": False,
        "protocol_repair_only_scientific_design_unchanged": True,
        "protected_read_only_trees": ["../jx_xp2_robustness_v1", "../jx_xp2_runs_v1"],
        "output_receipt_or_checkpoint_tree_overlap": "FORBIDDEN",
    }:
        raise IntegrityError("XP2-v1 invalid-protocol lineage declaration changed")
    for path_key, digest_key, size_key in (
        ("v1_registration_path", "v1_registration_sha256", None),
        ("v1_final_invalid_diagnostic_path", "v1_final_invalid_diagnostic_sha256",
         "v1_final_invalid_diagnostic_size_bytes"),
    ):
        bound = path.parent / lineage[path_key]
        if (bound.is_symlink() or not bound.is_file() or bound.stat().st_nlink != 1
                or sha256_file(bound) != lineage[digest_key]
                or (size_key is not None and bound.stat().st_size != lineage[size_key])):
            raise IntegrityError("XP2-v1 invalid-protocol lineage file changed")
    validate_v2_replay_lineage(contract, path.parent)
    validate_v3_failed_startup_lineage(contract, path.parent)
    science = contract.get("frozen_v1_scientific_design_inputs")
    if not isinstance(science, dict) or (
        science.get("role")
        != "EXACT_SCIENTIFIC_DESIGN_INPUT_REUSE_WITH_V4_EXECUTION_PROTOCOL_ONLY"
        or science.get("scientific_design_experiment_id")
        != SCIENTIFIC_DESIGN_EXPERIMENT_ID
        or science.get("tracer_or_state_bytes_regenerated_for_v4") is not False
        or science.get("v1_dynamics_or_outcomes_consumed") is not False
    ):
        raise IntegrityError("frozen v1 scientific-design lineage changed")
    for name in ("seed_manifest", "selection_manifest", "initial_states"):
        frozen = path.parent / science[f"{name}_path"]
        if (frozen.is_symlink() or not frozen.is_file() or frozen.stat().st_nlink != 1
                or sha256_file(frozen) != science[f"{name}_sha256"]):
            raise IntegrityError("frozen v1 scientific-design input bytes changed")


def expand_initial_states(contract: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, list[list[Any]]]]:
    policy = contract["initial_state_policy"]
    if sha256_file(path) != policy["artifact_sha256"]:
        raise IntegrityError("initial-state artifact byte digest mismatch")
    artifact = strict_json(path)
    if (artifact.get("schema") != INITIAL_SCHEMA
            or artifact.get("experiment_id") != SCIENTIFIC_DESIGN_EXPERIMENT_ID):
        raise IntegrityError("initial-state artifact identity changed")
    common = artifact.get("common_active_sun_centered_rows")
    tracers = artifact.get("tracer_sun_centered_rows")
    configurations = artifact.get("configuration_states")
    if not isinstance(common, list) or len(common) != 5 or not isinstance(tracers, list) or len(tracers) != 128:
        raise IntegrityError("initial-state row cardinality changed")
    if not isinstance(configurations, list) or len(configurations) != 25:
        raise IntegrityError("initial-state configuration cardinality changed")
    expected_ids = set(PRIMARY_ARM_IDS)
    expanded: dict[str, list[list[Any]]] = {}
    index_rows: list[list[str]] = []
    for config in configurations:
        if not isinstance(config, list) or len(config) != 6:
            raise IntegrityError("configuration tuple shape changed")
        arm_id, active_count, added, position_hex, velocity_hex, expected_digest = config
        if arm_id not in expected_ids or arm_id in expanded:
            raise IntegrityError("configuration identity changed")
        source_rows = list(common) + ([] if added is None else [added]) + list(tracers)
        if active_count != (5 if added is None else 6) or len(source_rows) != active_count + 128:
            raise IntegrityError("configuration active count changed")
        position = unpack_state(position_hex + "0" * 48)[:3]
        velocity = unpack_state(velocity_hex + "0" * 48)[:3]
        output: list[list[Any]] = []
        for row in source_rows:
            if not isinstance(row, list) or len(row) != 4:
                raise IntegrityError("initial row shape changed")
            logical_id, role, mass_hex, state_hex = row
            if role not in ("A", "T"):
                raise IntegrityError("particle role changed")
            binary64_from_hex(mass_hex)
            state = unpack_state(state_hex)
            translated = [state[index] - position[index] for index in range(3)] + [
                state[index + 3] - velocity[index] for index in range(3)
            ]
            output.append([logical_id, role, mass_hex, pack_state(translated)])
        actual_digest = sha256_bytes(EXPANDED_STATE_DOMAIN + canonical_bytes(output))
        if actual_digest != expected_digest:
            raise IntegrityError("expanded initial-state digest mismatch")
        expanded[arm_id] = output
        index_rows.append([arm_id, actual_digest])
    if set(expanded) != expected_ids:
        raise IntegrityError("initial-state configuration set changed")
    index_digest = sha256_bytes(CONFIG_INDEX_DOMAIN + canonical_bytes(index_rows))
    if index_digest != artifact.get("configuration_digest_index_sha256"):
        raise IntegrityError("configuration digest index mismatch")
    return artifact, expanded


def validate_initial_pairing(
    tracers: Sequence[dict[str, Any]], artifact: dict[str, Any], expanded: dict[str, list[list[Any]]]
) -> None:
    expected_ids = [row["logical_id"] for row in tracers]
    artifact_ids = [row[0] for row in artifact["tracer_sun_centered_rows"]]
    if artifact_ids != expected_ids:
        raise IntegrityError("seed and Cartesian tracer orders differ")
    for arm_id, rows in expanded.items():
        active_count = len(rows) - 128
        if [row[0] for row in rows[active_count:]] != expected_ids:
            raise IntegrityError(f"expanded tracer order changed in {arm_id}")


def trees_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def protected_roots(contract: dict[str, Any], package_root: Path) -> list[Path]:
    roots = [package_root.resolve()]
    roots.extend((package_root / item).resolve()
                 for item in contract["xp1_historical_binding"]["protected_read_only_trees"])
    roots.extend((package_root / item).resolve()
                 for item in contract["xp2_v1_invalid_protocol_lineage"][
                     "protected_read_only_trees"
                 ])
    roots.extend((package_root / item).resolve()
                 for item in contract["xp2_v2_invalid_replay_lineage"][
                     "protected_read_only_trees"
                 ])
    roots.extend((package_root / item).resolve()
                 for item in contract["xp2_v3_failed_startup_lineage"][
                     "protected_read_only_trees"
                 ])
    gate = contract["engineering_boundary_gate_v1"]
    roots.extend((package_root / gate[key]).resolve() for key in (
        "engineering_output_root", "engineering_verifier_scratch_root",
        "engineering_verifier_start_path", "engineering_verifier_terminal_path",
        "engineering_verification_receipt_path",
    ))
    return roots


def validate_output_root(path: Path, contract: dict[str, Any], package_root: Path, resume: bool) -> Path:
    if path.is_symlink():
        raise IntegrityError("output root must not be a symlink")
    result = path.resolve()
    if any(trees_overlap(result, protected) for protected in protected_roots(contract, package_root)):
        raise IntegrityError("output root overlaps a protected tree")
    if resume:
        if not result.is_dir():
            raise IntegrityError("resume output root does not exist")
    elif result.exists():
        raise FileExistsError("clean output root must be absent")
    elif not result.parent.is_dir():
        raise IntegrityError("output parent does not exist")
    return result


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
    registration_path: Path, contract_path: Path, seed_path: Path,
    initial_path: Path, runner_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    registration = strict_json(registration_path)
    if (
        registration.get("schema") != REGISTRATION_SCHEMA
        or registration.get("experiment_id") != EXPERIMENT_ID
        or registration.get("outcomes_generated") is not False
        or registration.get("scientific_evidence_artifact") is not False
    ):
        raise IntegrityError("registration identity or pre-output state changed")
    locked = registration.get("locked_files")
    if not isinstance(locked, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
        for name, digest in locked.items()
    ):
        raise IntegrityError("registration locked-files map is invalid")
    root = registration_path.resolve().parent
    inventory = set(strict_json(contract_path)["result_policy"]["registered_package_inventory"])
    if inventory != set(locked) | {"registration_v1.json"}:
        raise IntegrityError("registration and contract inventories differ")
    entries = list(root.iterdir())
    if ({candidate.name for candidate in entries} != inventory
            or any(candidate.is_symlink() or not candidate.is_file()
                   or candidate.stat().st_nlink != 1 for candidate in entries)):
        raise IntegrityError("registered package has extras or omissions")
    for name, expected in locked.items():
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_nlink != 1:
            raise IntegrityError("registered artifact is unsafe")
        if sha256_file(candidate) != expected:
            raise IntegrityError(f"registered artifact digest mismatch: {name}")
    expected_paths = {
        "contract_v1.json": contract_path, "seed_manifest_v1.json": seed_path,
        "initial_states_v1.json": initial_path, "run_primary.py": runner_path,
    }
    for name, supplied in expected_paths.items():
        if supplied.resolve() != root / name:
            raise IntegrityError("noncanonical registered input path")
    validate_final_engineering_authorization(registration, strict_json(contract_path), root)
    return registration, locked


def rebound_python_tree(root: Path) -> tuple[int, str]:
    paths = sorted(root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix())
    digest = hashlib.sha256(REBOUND_TREE_DOMAIN)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return len(paths), digest.hexdigest()


def get_rebound(contract: dict[str, Any]) -> Any:
    global _REBOUND_CACHE
    existing = sys.modules.get("rebound")
    runtime = contract["runtime_lock"]
    if existing is None:
        specification = importlib.util.find_spec("rebound")
        if specification is None or not specification.submodule_search_locations:
            raise IntegrityError("REBOUND is unavailable")
        source_root = Path(next(iter(specification.submodule_search_locations))).resolve()
        if rebound_python_tree(source_root) != (
            runtime["rebound_python_source_file_count"], runtime["rebound_python_source_sha256"]
        ):
            raise IntegrityError("REBOUND source-tree digest mismatch")
        binaries = sorted(source_root.parent.glob("librebound*.so"))
        if len(binaries) != 1 or sha256_file(binaries[0]) != runtime["rebound_binary_sha256"]:
            raise IntegrityError("REBOUND native library digest mismatch")
        _REBOUND_CACHE = tempfile.TemporaryDirectory(prefix="jx-xp2-rebound-import-")
        sys.pycache_prefix = _REBOUND_CACHE.name
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        existing = importlib.import_module("rebound")
        setattr(existing, "_jx_xp2_hash_locked_import", True)
    elif getattr(existing, "_jx_xp2_hash_locked_import", False) is not True:
        raise IntegrityError("REBOUND imported before hash guard")
    return existing


def validate_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    rebound = get_rebound(contract)
    root = Path(rebound.__file__).resolve().parent
    binary = Path(rebound.clibrebound._name).resolve()
    count, source_digest = rebound_python_tree(root)
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "rebound_version": rebound.__version__, "rebound_build": rebound.__build__,
        "rebound_binary_sha256": sha256_file(binary),
        "rebound_python_source_file_count": count,
        "rebound_python_source_sha256": source_digest,
    }
    expected = {key: contract["runtime_lock"][key] for key in actual}
    if actual != expected:
        raise IntegrityError("primary runtime lock mismatch")
    return actual


def empirical_w1(left: Sequence[float], right: Sequence[float]) -> float:
    """Exact equal-mass empirical one-dimensional Wasserstein distance."""
    if len(left) != len(right) or not left:
        raise IntegrityError("W1 requires equally sized nonempty samples")
    a, b = sorted(map(float, left)), sorted(map(float, right))
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise NumericalError("W1 input is non-finite")
    return math.fsum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def hit(row: dict[str, Any], threshold: float) -> int:
    value = row["minimum_sampled_q_AU"]
    return int(value is not None and value < threshold)


def particle_index(arm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = arm["particles"]
    if len(rows) != 128:
        raise IntegrityError("particle summary cardinality changed")
    result = {row["logical_id"]: row for row in rows}
    if len(result) != 128:
        raise IntegrityError("particle summary IDs are not unique")
    return result


def structural_effects(arms: dict[str, dict[str, Any]], threshold: float = 35.0) -> dict[str, Any]:
    if set(arms) != set(PRIMARY_ARM_IDS):
        raise IntegrityError("structural effects require all 25 configurations")
    indexes = {arm_id: particle_index(row) for arm_id, row in arms.items()}
    ids = sorted(indexes["M0"])
    if any(set(indexes[arm_id]) != set(ids) for arm_id in PRIMARY_ARM_IDS):
        raise IntegrityError("matched tracer IDs differ across configurations")
    m1_ids = PRIMARY_ARM_IDS[1:]
    mixture = 0
    by_block = {block: 0 for block in range(8)}
    by_case = {case: 0 for case in ("CI01", "CI05", "CI09")}
    by_orientation = {f"P{probe}": 0 for probe in range(8)}
    for logical_id in ids:
        control = hit(indexes["M0"][logical_id], threshold)
        block = indexes["M0"][logical_id].get("block_index")
        if not isinstance(block, int) or not 0 <= block < 8:
            raise IntegrityError("particle block metadata is invalid")
        if any(indexes[arm_id][logical_id].get("block_index") != block for arm_id in m1_ids):
            raise IntegrityError("matched particle block metadata differs across configurations")
        tracer_total = sum(hit(indexes[arm_id][logical_id], threshold) for arm_id in m1_ids)
        delta = tracer_total - 24 * control
        mixture += delta
        by_block[block] += delta
        for case in by_case:
            by_case[case] += sum(hit(indexes[f"{case}-P{probe}"][logical_id], threshold)
                                 for probe in range(8)) - 8 * control
        for probe in range(8):
            orientation = f"P{probe}"
            by_orientation[orientation] += sum(
                hit(indexes[f"{case}-{orientation}"][logical_id], threshold)
                for case in by_case
            ) - 3 * control
    return {
        "threshold_AU": threshold, "mixture_numerator": mixture,
        "mixture_denominator": 3072,
        "block_numerators": {str(key): value for key, value in by_block.items()},
        "block_denominator_each": 384, "physical_case_numerators": by_case,
        "physical_case_denominator_each": 1024,
        "orientation_numerators": by_orientation, "orientation_denominator_each": 384,
    }


def structural_label(effect: dict[str, Any]) -> str:
    n = effect["mixture_numerator"]
    blocks = list(effect["block_numerators"].values())
    cases = list(effect["physical_case_numerators"].values())
    orientations = list(effect["orientation_numerators"].values())
    if (n >= 154 and all(value > 0 for value in blocks) and all(value > 0 for value in cases)
            and sum(value > 0 for value in orientations) >= 6
            and not any(value < 0 for value in orientations)):
        return "DESIGN_GRID_DIRECTIONALLY_ROBUST_INCREASE"
    if (n <= -154 and all(value < 0 for value in blocks) and all(value < 0 for value in cases)
            and sum(value < 0 for value in orientations) >= 6
            and not any(value > 0 for value in orientations)):
        return "DESIGN_GRID_DIRECTIONALLY_ROBUST_DECREASE"
    if (abs(n) <= 61 and max(map(abs, blocks)) <= 19 and max(map(abs, cases)) <= 51
            and max(map(abs, orientations)) <= 19):
        return "DESIGN_GRID_Q35_PRACTICALLY_SMALL_WITH_EVENT_SUPPORT"
    return "INCONCLUSIVE"


def event_support(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    indexes = {arm_id: particle_index(row) for arm_id, row in arms.items()}
    ids = sorted(indexes["M0"])
    control = {logical_id for logical_id in ids if hit(indexes["M0"][logical_id], 35.0)}
    any_m1 = {logical_id for logical_id in ids if any(
        hit(indexes[arm_id][logical_id], 35.0) for arm_id in PRIMARY_ARM_IDS[1:]
    )}
    discordant = {logical_id for logical_id in ids if any(
        hit(indexes[arm_id][logical_id], 35.0) != hit(indexes["M0"][logical_id], 35.0)
        for arm_id in PRIMARY_ARM_IDS[1:]
    )}
    return {
        "unique_M0_q35_hitter_count": len(control),
        "unique_any_M1_q35_hitter_count": len(any_m1),
        "unique_union_q35_hitter_count": len(control | any_m1),
        "unique_discordant_q35_tracer_count": len(discordant),
        "M0_logical_ids_sha256": sha256_bytes(canonical_bytes(sorted(control))),
        "any_M1_logical_ids_sha256": sha256_bytes(canonical_bytes(sorted(any_m1))),
        "union_logical_ids_sha256": sha256_bytes(canonical_bytes(sorted(control | any_m1))),
        "discordant_logical_ids_sha256": sha256_bytes(canonical_bytes(sorted(discordant))),
    }


def apply_event_floor(label: str, support: dict[str, Any]) -> str:
    if label == "DESIGN_GRID_Q35_PRACTICALLY_SMALL_WITH_EVENT_SUPPORT":
        passes = (support["unique_M0_q35_hitter_count"] >= 16
                  and support["unique_any_M1_q35_hitter_count"] >= 16)
        return label if passes else "ENDPOINT_FLOOR_LIMITED"
    if label in ("DESIGN_GRID_DIRECTIONALLY_ROBUST_INCREASE",
                 "DESIGN_GRID_DIRECTIONALLY_ROBUST_DECREASE"):
        passes = (support["unique_union_q35_hitter_count"] >= 24
                  and support["unique_discordant_q35_tracer_count"] >= 16)
        return label if passes else "ENDPOINT_FLOOR_LIMITED"
    return label


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def vector_subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a - b for a, b in zip(left, right, strict=True)]


def active_snapshot(simulation: Any) -> dict[str, Any]:
    particles = [simulation.particles[index] for index in range(simulation.N_active)]
    masses = [float(particle.m) for particle in particles]
    total_mass = math.fsum(masses)
    if total_mass <= 0.0:
        raise NumericalError("active mass is not positive")
    positions = [[float(p.x), float(p.y), float(p.z)] for p in particles]
    velocities = [[float(p.vx), float(p.vy), float(p.vz)] for p in particles]
    if not all(math.isfinite(value) for row in (*positions, *velocities) for value in row):
        raise NumericalError("non-finite active Cartesian state")
    momentum = [math.fsum(mass * velocities[index][axis] for index, mass in enumerate(masses))
                for axis in range(3)]
    r_com = [math.fsum(mass * positions[index][axis] for index, mass in enumerate(masses))
             / total_mass for axis in range(3)]
    v_com = [component / total_mass for component in momentum]
    angular_terms: list[list[float]] = []
    kinetic_terms: list[float] = []
    linear_scale_terms: list[float] = []
    for mass, position, velocity in zip(masses, positions, velocities, strict=True):
        relative_position = vector_subtract(position, r_com)
        relative_velocity = vector_subtract(velocity, v_com)
        angular_terms.append([
            mass * (relative_position[1] * relative_velocity[2]
                    - relative_position[2] * relative_velocity[1]),
            mass * (relative_position[2] * relative_velocity[0]
                    - relative_position[0] * relative_velocity[2]),
            mass * (relative_position[0] * relative_velocity[1]
                    - relative_position[1] * relative_velocity[0]),
        ])
        kinetic_terms.append(0.5 * mass * math.fsum(value * value for value in relative_velocity))
        linear_scale_terms.append(mass * vector_norm(relative_velocity))
    com_angular = [math.fsum(row[axis] for row in angular_terms) for axis in range(3)]
    potential_terms: list[float] = []
    for left in range(len(particles)):
        for right in range(left + 1, len(particles)):
            separation = vector_norm(vector_subtract(positions[left], positions[right]))
            if separation <= 0.0:
                raise NumericalError("zero active-body separation")
            potential_terms.append(-float(simulation.G) * masses[left] * masses[right] / separation)
    intrinsic_energy = math.fsum((math.fsum(kinetic_terms), math.fsum(potential_terms)))
    linear_scale = math.fsum(linear_scale_terms)
    if (not all(math.isfinite(value) for value in (*momentum, *r_com, *v_com, *com_angular,
                                                    intrinsic_energy, linear_scale))
            or abs(intrinsic_energy) == 0.0 or vector_norm(com_angular) == 0.0
            or linear_scale == 0.0):
        raise NumericalError("invalid active invariant denominator")
    return {
        "momentum": momentum, "r_com": r_com, "v_com": v_com,
        "com_angular": com_angular, "linear_internal_scale": linear_scale,
        "intrinsic_energy": intrinsic_energy,
    }


def update_invariant_maximum(maximum: dict[str, float], initial: dict[str, Any], current: dict[str, Any]) -> None:
    values = {
        "relative_compensated_active_energy_drift": abs(
            current["intrinsic_energy"] - initial["intrinsic_energy"]
        ) / abs(initial["intrinsic_energy"]),
        "relative_active_com_angular_momentum_vector_drift": vector_norm(
            vector_subtract(current["com_angular"], initial["com_angular"])
        ) / vector_norm(initial["com_angular"]),
        "scale_normalized_active_linear_momentum_residual": vector_norm(
            vector_subtract(current["momentum"], initial["momentum"])
        ) / initial["linear_internal_scale"],
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise NumericalError("non-finite invariant drift")
    for key, value in values.items():
        maximum[key] = max(maximum[key], value)


def decoded_continuation_projection(
    simulation: Any, *, source_mode: str = "ARCHIVE",
) -> dict[str, Any]:
    """Canonical state decoded from an archive, excluding allocator capacities."""
    if source_mode not in {"ARCHIVE", "LIVE_BOUNDARY"}:
        raise IntegrityError("unknown continuation projection source mode")
    mercurius = simulation.ri_mercurius
    whfast = simulation.ri_whfast
    ias15 = simulation.ri_ias15
    box = simulation.boxsize
    particle_count = int(simulation.N)
    particle_capacity = int(simulation.N_allocated)
    if (particle_count not in {133, 134}
            or not particle_count <= particle_capacity <= MAX_REBOUND_ALLOCATION_CAPACITY
            or not bool(simulation._particles)):
        raise IntegrityError("decoded particle allocation/count is unsafe")
    dcrit_capacity = int(mercurius._N_allocated_dcrit)
    dcrit_present = bool(mercurius._dcrit)
    if ((dcrit_capacity == 0) != (not dcrit_present)
            or dcrit_capacity < 0 or dcrit_capacity > MAX_REBOUND_ALLOCATION_CAPACITY
            or (dcrit_present and dcrit_capacity < particle_count)):
        raise IntegrityError("MERCURIUS dcrit allocation/state is unsafe")
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
        raise IntegrityError("MERCURIUS transient cache topology is unsafe")
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
        raise IntegrityError("IAS15 decoded array allocation/state is unsafe")
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
        raise IntegrityError("WHFast decoded internal arrays are unexpectedly live")

    def active_interval(
        pointer: Any, count: int, item_size: int, alignment: int,
    ) -> tuple[int, int]:
        if count <= 0 or not bool(pointer) or item_size <= 0:
            raise IntegrityError("decoded active-memory interval is unsafe")
        start = ctypes.cast(pointer, ctypes.c_void_p).value
        if type(start) is not int:
            raise IntegrityError("decoded active-memory pointer is null")
        end = start + count * item_size
        max_address = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
        if (start <= 0 or end <= start or end - 1 > max_address
                or alignment <= 0 or start % alignment != 0):
            raise IntegrityError("decoded active-memory interval overflowed")
        return start, end

    simulation_start = ctypes.addressof(simulation)
    simulation_size = ctypes.sizeof(simulation)
    if (simulation_start <= 0 or simulation_size <= 0
            or simulation_start % ctypes.alignment(type(simulation)) != 0):
        raise IntegrityError("decoded Simulation memory range is unsafe")
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
        raise IntegrityError("decoded active-memory allocations overlap")
    dcrit = [] if not dcrit_present else [
        binary64_hex(float(mercurius._dcrit[index]))
        for index in range(particle_count)
    ]
    particles = []
    for index, particle in enumerate(simulation.particles):
        parent_bound = bool(particle._sim) and (
            ctypes.cast(particle._sim, ctypes.c_void_p).value == simulation_start
        )
        if not parent_bound:
            raise IntegrityError("decoded particle simulation reference is not parent-bound")
        particles.append({
            "index": index, "hash": int(particle.hash.value),
            "simulation_reference_bound_to_parent": True,
            "m_hex": binary64_hex(float(particle.m)),
            "r_hex": binary64_hex(float(particle.r)),
            "x_hex": binary64_hex(float(particle.x)),
            "y_hex": binary64_hex(float(particle.y)),
            "z_hex": binary64_hex(float(particle.z)),
            "vx_hex": binary64_hex(float(particle.vx)),
            "vy_hex": binary64_hex(float(particle.vy)),
            "vz_hex": binary64_hex(float(particle.vz)),
            "ax_hex": binary64_hex(float(particle.ax)),
            "ay_hex": binary64_hex(float(particle.ay)),
            "az_hex": binary64_hex(float(particle.az)),
            "last_collision_hex": binary64_hex(float(particle.last_collision)),
            "collision_cell_present": particle.c is not None,
            "additional_properties_present": particle.ap is not None,
        })
    ias_direct = {
        name.removeprefix("_"): decoded_double_array_sha256(
            getattr(ias15, name), ias_count
        ) if source_mode == "ARCHIVE" else None
        for name in ("_at", "_x0", "_v0", "_a0", "_csx", "_csv", "_csa0")
    }
    ias_coefficients = {
        group.removeprefix("_"): {
            f"p{index}": decoded_double_array_sha256(
                getattr(getattr(ias15, group), f"p{index}"), ias_count
            ) if source_mode == "ARCHIVE" else None for index in range(7)
        } for group in ("_g", "_b", "_csb", "_e", "_br", "_er")
    }
    map_sha256 = None if source_mode == "LIVE_BOUNDARY" or not bool(ias15._map) else sha256_bytes(
        CONTINUATION_ARRAY_DOMAIN + b"IAS15_MAP\0" + b"".join(
            struct.pack(">i", int(ias15._map[index])) for index in range(map_count)
        )
    )
    return {
        "schema": "jx-xp2-mercurius-decoded-continuation-state/v3",
        "simulation": {
            "t_hex": binary64_hex(float(simulation.t)),
            "G_hex": binary64_hex(float(simulation.G)),
            "softening_hex": binary64_hex(float(simulation.softening)),
            "dt_hex": binary64_hex(float(simulation.dt)),
            "dt_last_done_hex": binary64_hex(float(simulation.dt_last_done)),
            "steps_done": int(simulation.steps_done),
            "usleep_hex": binary64_hex(float(simulation.usleep)),
            "save_messages": int(simulation.save_messages),
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
            "exit_max_distance_hex": binary64_hex(float(simulation.exit_max_distance)),
            "exit_min_distance_hex": binary64_hex(float(simulation.exit_min_distance)),
            "track_energy_offset": int(simulation.track_energy_offset),
            "energy_offset_hex": binary64_hex(float(simulation.energy_offset)),
            "opening_angle2_hex": binary64_hex(float(simulation.opening_angle2)),
            "boxsize_hex": [binary64_hex(float(value)) for value in (box.x, box.y, box.z)],
            "boxsize_max_hex": binary64_hex(float(simulation.boxsize_max)),
            "root_size_hex": binary64_hex(float(simulation.root_size)),
            "N_root": int(simulation.N_root),
            "N_root_xyz": [int(simulation.N_root_x), int(simulation.N_root_y),
                           int(simulation.N_root_z)],
            "N_ghost_xyz": [int(simulation.N_ghost_x), int(simulation.N_ghost_y),
                            int(simulation.N_ghost_z)],
            "collision_resolve_keep_sorted": int(simulation.collision_resolve_keep_sorted),
            "collisions_N": int(simulation.collisions_N),
            "minimum_collision_velocity_hex": binary64_hex(
                float(simulation.minimum_collision_velocity)
            ),
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
            "collisions_plog_hex": binary64_hex(float(simulation.collisions_plog)),
            "collisions_log_n": int(simulation.collisions_log_n),
            "calculate_megno": int(simulation._calculate_megno),
            "megno_Ys_hex": binary64_hex(float(simulation._megno_Ys)),
            "megno_Yss_hex": binary64_hex(float(simulation._megno_Yss)),
            "megno_cov_Yt_hex": binary64_hex(float(simulation._megno_cov_Yt)),
            "megno_var_t_hex": binary64_hex(float(simulation._megno_var_t)),
            "megno_mean_t_hex": binary64_hex(float(simulation._megno_mean_t)),
            "megno_mean_Y_hex": binary64_hex(float(simulation._megno_mean_Y)),
            "megno_initial_t_hex": binary64_hex(float(simulation._megno_initial_t)),
            "megno_n": int(simulation._megno_n),
            "N_odes": int(simulation._N_odes),
            "odes_allocation_count": int(simulation._N_allocated_odes),
            "odes_warnings": int(simulation._odes_warnings),
            "odes_present": bool(simulation._odes),
            "extras_present": simulation.extras is not None,
            "simulationarchive_auto_interval_hex": binary64_hex(
                float(simulation.simulationarchive_auto_interval)
            ),
            "simulationarchive_auto_walltime_hex": binary64_hex(
                float(simulation.simulationarchive_auto_walltime)
            ),
            "simulationarchive_auto_step": int(simulation.simulationarchive_auto_step),
            "simulationarchive_next_hex": binary64_hex(
                float(simulation.simulationarchive_next)
            ),
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
            "r_crit_hill_hex": binary64_hex(float(mercurius.r_crit_hill)),
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
            "com_position_hex": [binary64_hex(float(value)) for value in (
                mercurius._com_pos.x, mercurius._com_pos.y, mercurius._com_pos.z
            )],
            "com_velocity_hex": [binary64_hex(float(value)) for value in (
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
            "epsilon_hex": binary64_hex(float(ias15.epsilon)),
            "min_dt_hex": binary64_hex(float(ias15.min_dt)),
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


def live_archive_endpoint_projection(simulation: Any) -> dict[str, Any]:
    """Full selected continuation state minus proven save/load cache normalization."""
    projection = decoded_continuation_projection(simulation, source_mode="LIVE_BOUNDARY")
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


def live_archive_endpoint_sha256(simulation: Any) -> str:
    return sha256_bytes(
        ENDPOINT_DIGEST_DOMAIN + canonical_bytes(live_archive_endpoint_projection(simulation))
    )


def decoded_double_array_sha256(pointer: Any, count: int) -> str | None:
    if count < 0:
        raise IntegrityError("negative decoded array length")
    if not bool(pointer):
        return None
    digest = hashlib.sha256(CONTINUATION_ARRAY_DOMAIN)
    for index in range(count):
        value = float(pointer[index])
        if not math.isfinite(value):
            raise IntegrityError("decoded continuation array contains a non-finite value")
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def decoded_state_sha256(simulation: Any) -> str:
    return sha256_bytes(
        STATE_DIGEST_DOMAIN + canonical_bytes(decoded_continuation_projection(simulation))
    )


def valid_ias15_continuation(value: dict[str, Any]) -> bool:
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
            or (count > 0 and any(not valid_sha256(digest) for digest in digests)):
        return False
    map_count = value["map_count"]
    return (type(map_count) is int and map_count == 0
            and value["map_sha256"] is None)


def validate_decoded_continuation_settings(
    projection: dict[str, Any], *, end_years: float, dt_years: float,
    particle_count: int,
) -> None:
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
        raise IntegrityError("decoded continuation projection schema changed")
    settings = projection["simulation"]
    mercurius = projection["mercurius"]
    whfast = projection["whfast"]
    ias15 = projection["ias15"]
    h = binary64_hex
    if (
        settings["softening_hex"] != h(0.0)
        or settings["dt_last_done_hex"] != h(dt_years)
        or settings["steps_done"] != int(round(end_years / dt_years))
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
        or len(mercurius["dcrit_hex"]) != particle_count
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
        or not valid_ias15_continuation(ias15)
        or any(row["simulation_reference_bound_to_parent"] is not True
               or row["last_collision_hex"] != h(0.0)
               or row["collision_cell_present"]
               or row["additional_properties_present"]
               for row in projection["particles"])
    ):
        raise IntegrityError("decoded continuation settings changed")


def arm_specification(contract: dict[str, Any], arm_id: str) -> dict[str, Any]:
    if arm_id not in ALL_ARM_IDS:
        raise IntegrityError("unknown arm ID")
    if arm_id.startswith("AUDIT-"):
        base = AUDIT_TO_PRIMARY[arm_id]
        arm_class = "HALF_TIMESTEP"
        dt = contract["design_core"]["dynamics"]["audit_dt_years"]
    else:
        base = arm_id
        arm_class = "PRIMARY_TIMESTEP"
        dt = contract["design_core"]["dynamics"]["primary_dt_years"]
    return {"arm_id": arm_id, "configuration_id": base, "arm_class": arm_class,
            "dt_years": float(dt)}


def build_simulation(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]], spec: dict[str, Any]
) -> Any:
    rebound = get_rebound(contract)
    rows = expanded[spec["configuration_id"]]
    simulation = rebound.Simulation()
    simulation.G = float(contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"])
    active_count = sum(row[1] == "A" for row in rows)
    if active_count not in (5, 6) or any(row[1] != "T" for row in rows[active_count:]):
        raise IntegrityError("initial-state roles/order changed")
    for logical_id, _role, mass_hex, state_hex in rows:
        state = unpack_state(state_hex)
        simulation.add(
            m=binary64_from_hex(mass_hex), x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5], hash=logical_id,
        )
    simulation.N_active = active_count
    simulation.integrator = "mercurius"
    simulation.dt = float(spec["dt_years"])
    simulation.testparticle_type = int(contract["design_core"]["dynamics"]["testparticle_type"])
    simulation.ri_mercurius.r_crit_hill = float(
        contract["design_core"]["dynamics"]["r_crit_hill"]
    )
    simulation.ri_mercurius.safe_mode = int(contract["design_core"]["dynamics"]["safe_mode"])
    simulation.collision = "none"
    if (
        simulation.N != active_count + 128 or simulation.N_active != active_count
        or simulation.integrator != "mercurius" or simulation.dt != spec["dt_years"]
        or simulation.testparticle_type != 0
        or simulation.ri_mercurius.r_crit_hill != 3.0
        or int(simulation.ri_mercurius.safe_mode) != 1
    ):
        raise IntegrityError("MERCURIUS configuration readback mismatch")
    return simulation


def blank_tracker(tracers: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "logical_id": row["logical_id"], "block_index": row["block_index"],
        "index_within_block": row["index_within_block"],
        "minimum_sampled_q_AU": None,
        "first_sampled_q_below_30_time_year": None,
        "first_sampled_q_below_35_time_year": None,
        "first_sampled_q_below_40_time_year": None,
        "all_samples_finite_cartesian_and_osculating": True,
    } for row in tracers]


def update_sample_stream(digest: Any, simulation: Any, sample_index: int, sample_year: float) -> None:
    digest.update(struct.pack(">QdII", sample_index, sample_year, simulation.N, simulation.N_active))
    for index, particle in enumerate(simulation.particles):
        digest.update(struct.pack(
            ">II8d", index, int(particle.hash.value), float(particle.m), float(particle.r),
            float(particle.x), float(particle.y), float(particle.z),
            float(particle.vx), float(particle.vy), float(particle.vz),
        ))


def sample_tracers(
    simulation: Any, tracker: list[dict[str, Any]], time_year: float, landmark: bool
) -> list[dict[str, Any]] | None:
    if not all(math.isfinite(float(getattr(particle, field)))
               for particle in simulation.particles
               for field in ("x", "y", "z", "vx", "vy", "vz")):
        raise NumericalError("non-finite Cartesian sample")
    sun = simulation.particles[0]
    final_rows: list[dict[str, Any]] = []
    for offset, row in enumerate(tracker):
        particle = simulation.particles[simulation.N_active + offset]
        try:
            orbit = particle.orbit(primary=sun)
            a_au = float(orbit.a); eccentricity = float(orbit.e)
            inclination_deg = math.degrees(float(orbit.inc))
            q_au = a_au * (1.0 - eccentricity)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            row["all_samples_finite_cartesian_and_osculating"] = False
            raise NumericalError("osculating conversion failed") from exc
        if (not all(math.isfinite(value) for value in (a_au, eccentricity, inclination_deg, q_au))
                or eccentricity < 0.0 or q_au < 0.0):
            row["all_samples_finite_cartesian_and_osculating"] = False
            raise NumericalError("non-finite or invalid osculating sample")
        previous = row["minimum_sampled_q_AU"]
        if previous is None or q_au < previous:
            row["minimum_sampled_q_AU"] = q_au
        for threshold in THRESHOLDS:
            key = f"first_sampled_q_below_{int(threshold)}_time_year"
            if q_au < threshold and row[key] is None:
                row[key] = time_year
        if landmark:
            distance = math.sqrt(
                (float(particle.x) - float(sun.x)) ** 2
                + (float(particle.y) - float(sun.y)) ** 2
                + (float(particle.z) - float(sun.z)) ** 2
            )
            if not math.isfinite(distance):
                raise NumericalError("non-finite final distance")
            final_rows.append({
                **row, "final_a_AU": a_au, "final_e": eccentricity,
                "final_i_deg": inclination_deg, "final_q_AU": q_au,
                "final_distance_AU": distance,
                "final_finite_and_bound": bool(a_au > 0.0 and eccentricity < 1.0),
            })
    return final_rows if landmark else None


def summarize_particles(rows: list[dict[str, Any]], horizon: float) -> dict[str, Any]:
    if len(rows) != 128:
        raise IntegrityError("landmark particle count changed")
    summary: dict[str, Any] = {
        "particle_count": 128,
        "all_particles_all_samples_finite_cartesian_and_osculating": all(
            row["all_samples_finite_cartesian_and_osculating"] for row in rows
        ),
        "final_finite_bound_count": sum(row["final_finite_and_bound"] for row in rows),
    }
    for threshold in THRESHOLDS:
        count = sum(hit(row, threshold) for row in rows)
        summary[f"q_below_{int(threshold)}_hit_count"] = count
        summary[f"q_below_{int(threshold)}_fraction"] = count / 128.0
        key = f"first_sampled_q_below_{int(threshold)}_time_year"
        summary[f"restricted_mean_censored_first_q{int(threshold)}_years"] = math.fsum(
            horizon if row[key] is None else float(row[key]) for row in rows
        ) / 128.0
    return summary


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


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
                    # An unpublished regular file may be atomically renamed/unlinked.
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


def enforce_worker_resources(contract: dict[str, Any], output_root: Path, started_ns: int) -> None:
    caps = contract["resource_caps_per_execution"]
    elapsed = (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
    if elapsed >= float(caps["max_wall_seconds_per_segment_attempt"]):
        raise ResourceLimitError("segment-attempt wall cap reached")
    if peak_rss_bytes() > int(caps["max_peak_rss_bytes_per_process"]):
        raise ResourceLimitError("process RSS cap reached")
    if directory_bytes(output_root) > int(caps["max_output_bytes"]):
        raise ResourceLimitError("output byte cap reached")
    if shutil.disk_usage(output_root).free < int(caps["minimum_free_disk_bytes"]):
        raise ResourceLimitError("free-disk floor reached")


def semantic_segment_chain(previous: str, payload: dict[str, Any]) -> str:
    if len(previous) != 64:
        raise IntegrityError("segment chain predecessor invalid")
    return sha256_bytes(SEGMENT_CHAIN_DOMAIN + bytes.fromhex(previous) + canonical_bytes(payload))


def semantic_segment_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Select deterministic decoded/scientific state; exclude raw archive bytes."""
    if not SEGMENT_SEMANTIC_FIELDS <= set(value):
        raise IntegrityError("segment semantic payload is incomplete")
    return {key: value[key] for key in SEGMENT_SEMANTIC_FIELDS}


def complete_attempt_evidence(receipt: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    evidence = {
        "semantic_segment_payload_sha256": sha256_bytes(
            canonical_bytes(semantic_segment_payload(receipt))
        ),
        "segment_chain_head": receipt["segment_chain_head"],
        "decoded_integrator_state_sha256": receipt["decoded_integrator_state_sha256"],
        "sampled_state_stream_sha256": receipt["sampled_state_stream_sha256"],
        "raw_checkpoint_sha256": receipt["checkpoint_sha256"],
        "raw_checkpoint_size_bytes": receipt["checkpoint_size_bytes"],
        "attempt_receipt_sha256": sha256_file(receipt_path),
    }
    validate_complete_attempt_evidence_shape(evidence)
    return evidence


def validate_complete_attempt_evidence_shape(value: dict[str, Any]) -> None:
    require_keys(value, {
        "semantic_segment_payload_sha256", "segment_chain_head",
        "decoded_integrator_state_sha256", "sampled_state_stream_sha256",
        "raw_checkpoint_sha256", "raw_checkpoint_size_bytes",
        "attempt_receipt_sha256",
    }, "complete uncommitted attempt evidence")
    if (any(not valid_sha256(value[key]) for key in (
            "semantic_segment_payload_sha256", "segment_chain_head",
            "decoded_integrator_state_sha256", "sampled_state_stream_sha256",
            "raw_checkpoint_sha256", "attempt_receipt_sha256",
        )) or type(value["raw_checkpoint_size_bytes"]) is not int
            or value["raw_checkpoint_size_bytes"] <= 0):
        raise IntegrityError("complete uncommitted attempt evidence changed")


def require_complete_attempt_semantic_match(
    receipt: dict[str, Any], evidence: dict[str, Any],
) -> None:
    validate_complete_attempt_evidence_shape(evidence)
    current_semantic = sha256_bytes(canonical_bytes(semantic_segment_payload(receipt)))
    if (current_semantic != evidence["semantic_segment_payload_sha256"]
            or receipt.get("segment_chain_head") != evidence["segment_chain_head"]
            or receipt.get("decoded_integrator_state_sha256")
            != evidence["decoded_integrator_state_sha256"]
            or receipt.get("sampled_state_stream_sha256")
            != evidence["sampled_state_stream_sha256"]):
        raise IntegrityError(
            "NONDETERMINISTIC_RESUME: retry changed decoded/scientific semantics"
        )


def checkpoint_paths(arm_dir: Path, segment_index: int, attempt_index: int = 1) -> tuple[Path, Path]:
    segment_dir = arm_dir / "segments"
    return (segment_dir / f"segment_{segment_index:02d}_attempt_{attempt_index:02d}_state.bin",
            segment_dir / f"segment_{segment_index:02d}_attempt_{attempt_index:02d}_receipt.json")


def segment_commit_path(arm_dir: Path, segment_index: int) -> Path:
    return arm_dir / "segments" / f"segment_{segment_index:02d}_commit.json"


def load_completed_segment(arm_dir: Path, segment_index: int) -> dict[str, Any]:
    segment_dir = arm_dir / "segments"
    if (arm_dir.is_symlink() or not arm_dir.is_dir() or segment_dir.is_symlink()
            or not segment_dir.is_dir()):
        raise IntegrityError("segment directory ancestry is unsafe")
    commit_path = segment_commit_path(arm_dir, segment_index)
    commit = strict_json(commit_path)
    require_keys(commit, {
        "schema", "experiment_id", "arm_id", "segment_index",
        "attempt_receipt_filename", "attempt_receipt_sha256", "checkpoint_filename",
        "checkpoint_sha256", "raw_checkpoint_integrity_only",
        "decoded_integrator_state_sha256", "segment_chain_head",
        "parent_terminal_validation",
    }, "segment parent commit")
    if (commit["schema"] != SEGMENT_COMMIT_SCHEMA
            or commit["experiment_id"] != EXPERIMENT_ID
            or commit["arm_id"] not in ALL_ARM_IDS
            or commit["segment_index"] != segment_index
            or commit["parent_terminal_validation"]
            != "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS"):
        raise IntegrityError("segment parent commit identity changed")
    receipt_name = commit["attempt_receipt_filename"]
    if (not isinstance(receipt_name, str) or Path(receipt_name).name != receipt_name
            or not receipt_name.startswith(f"segment_{segment_index:02d}_attempt_")
            or not receipt_name.endswith("_receipt.json")):
        raise IntegrityError("segment attempt receipt filename changed")
    receipt_path = commit_path.parent / receipt_name
    if receipt_path.parent.resolve() != commit_path.parent.resolve() or receipt_path.name != commit.get(
        "attempt_receipt_filename"
    ):
        raise IntegrityError("segment attempt receipt path escaped its directory")
    receipt = strict_json(receipt_path)
    state_name = receipt.get("checkpoint_filename", "")
    if (not isinstance(state_name, str) or Path(state_name).name != state_name
            or not state_name.startswith(f"segment_{segment_index:02d}_attempt_")
            or not state_name.endswith("_state.bin")
            or commit["checkpoint_filename"] != state_name):
        raise IntegrityError("segment checkpoint filename changed")
    state_path = receipt_path.parent / state_name
    if (state_path.parent.resolve() != receipt_path.parent.resolve()
            or state_path.name != receipt.get("checkpoint_filename")
            or state_path.is_symlink() or not state_path.is_file() or state_path.stat().st_nlink != 1):
        raise IntegrityError("segment checkpoint path is unsafe or escaped")
    if (
        receipt.get("schema") != CHECKPOINT_SCHEMA
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("segment_index") != segment_index
        or receipt.get("checkpoint_filename") != state_path.name
        or receipt.get("checkpoint_sha256") != sha256_file(state_path)
        or receipt.get("checkpoint_size_bytes") != state_path.stat().st_size
        or commit.get("attempt_receipt_sha256") != sha256_file(receipt_path)
        or commit.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
        or commit.get("raw_checkpoint_integrity_only") is not True
        or receipt.get("raw_checkpoint_integrity_only") is not True
        or commit.get("decoded_integrator_state_sha256")
        != receipt.get("decoded_integrator_state_sha256")
        or commit.get("segment_chain_head") != receipt.get("segment_chain_head")
    ):
        raise IntegrityError("stored segment receipt mismatch")
    return receipt


def save_simulation_checkpoint(
    simulation: Any, contract: dict[str, Any], state_path: Path, *,
    boundary_mode: str = "OFFICIAL_SEGMENT", end_years: float | None = None,
    dt_years: float | None = None, particle_count: int | None = None,
) -> tuple[str, int, str]:
    if boundary_mode not in {
        "OFFICIAL_SEGMENT", "ENGINEERING_FULL_SEGMENT",
        "ENGINEERING_CONTINUATION_PROBE", "NO_DYNAMICS_ENGINEERING",
    }:
        raise IntegrityError("unknown checkpoint boundary mode")
    expected = (end_years, dt_years, particle_count)
    if ((boundary_mode in {
            "OFFICIAL_SEGMENT", "ENGINEERING_FULL_SEGMENT",
            "ENGINEERING_CONTINUATION_PROBE",
         }
         and any(value is None for value in expected))
            or (boundary_mode == "NO_DYNAMICS_ENGINEERING"
                and any(value is not None for value in expected))):
        raise IntegrityError("checkpoint boundary expectations are incomplete")
    if (boundary_mode == "ENGINEERING_FULL_SEGMENT"
            and (end_years != 50_000.0 or dt_years not in {0.125, 0.0625}
                 or particle_count not in {133, 134})):
        raise IntegrityError("engineering full-segment checkpoint boundary changed")
    if (boundary_mode == "ENGINEERING_CONTINUATION_PROBE"
            and (end_years != 50_050.0 or dt_years not in {0.125, 0.0625}
                 or particle_count not in {133, 134})):
        raise IntegrityError("engineering continuation checkpoint boundary changed")
    if state_path.exists() or state_path.is_symlink():
        raise FileExistsError("checkpoint final path already exists")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pending = state_path.with_name(f".{state_path.name}.pending")
    if pending.exists() or pending.is_symlink():
        raise FileExistsError("stale checkpoint pending path")
    simulation.save_to_file(str(pending), delete_file=True)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    digest = sha256_file(pending); size = pending.stat().st_size
    if not 0 < size <= MAX_PRIMARY_CHECKPOINT_BYTES:
        raise ResourceLimitError("primary checkpoint exceeds its registered byte bound")
    rebound = get_rebound(contract)
    decoded = rebound.Simulation(str(pending))
    decoded_projection = decoded_continuation_projection(decoded)
    decoded_digest = sha256_bytes(
        STATE_DIGEST_DOMAIN + canonical_bytes(decoded_projection)
    )
    if boundary_mode in {
        "OFFICIAL_SEGMENT", "ENGINEERING_FULL_SEGMENT",
        "ENGINEERING_CONTINUATION_PROBE",
    }:
        validate_decoded_continuation_settings(
            decoded_projection, end_years=float(end_years), dt_years=float(dt_years),
            particle_count=int(particle_count),
        )
    live_endpoint = live_archive_endpoint_projection(simulation)
    decoded_endpoint = live_archive_endpoint_projection(decoded)
    if live_endpoint != decoded_endpoint:
        raise IntegrityError("checkpoint scientific endpoint changed across save/load")
    del decoded
    gc.collect()
    os.replace(pending, state_path)
    descriptor = os.open(state_path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest, size, decoded_digest


def initial_execution_state(
    contract: dict[str, Any], tracers: list[dict[str, Any]],
    expanded: dict[str, list[list[Any]]], spec: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, float], dict[str, Any], str, int]:
    simulation = build_simulation(contract, expanded, spec)
    tracker = blank_tracker(tracers)
    initial = active_snapshot(simulation)
    maximum = {
        "relative_compensated_active_energy_drift": 0.0,
        "relative_active_com_angular_momentum_vector_drift": 0.0,
        "scale_normalized_active_linear_momentum_residual": 0.0,
    }
    return simulation, tracker, initial, maximum, {}, INITIAL_SEGMENT_CHAIN, 0


def resume_execution_state(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]], arm_dir: Path,
    segment_index: int, spec: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, float], dict[str, Any], str, int]:
    previous = load_completed_segment(arm_dir, segment_index - 1)
    previous_attempt = previous.get("provenance", {}).get("attempt_index")
    if not isinstance(previous_attempt, int) or validate_attempt_payload(
        contract, expanded, arm_dir, spec["arm_id"], segment_index - 1, previous_attempt
    ) != previous:
        raise IntegrityError("previous committed segment failed full payload validation")
    if previous["arm_id"] != spec["arm_id"] or previous["configuration_id"] != spec["configuration_id"]:
        raise IntegrityError("previous segment arm identity changed")
    commit = strict_json(segment_commit_path(arm_dir, segment_index - 1))
    receipt_path = segment_commit_path(arm_dir, segment_index - 1).parent / commit[
        "attempt_receipt_filename"
    ]
    state_path = receipt_path.parent / previous["checkpoint_filename"]
    simulation = get_rebound(contract).Simulation(str(state_path))
    if decoded_state_sha256(simulation) != previous["decoded_integrator_state_sha256"]:
        raise IntegrityError("resumed checkpoint decoded digest mismatch")
    return (
        simulation, previous["tracker"], previous["initial_active_invariants"],
        previous["maximum_active_invariant_drifts"], previous["landmarks"],
        previous["segment_chain_head"], previous["sample_count_total"],
    )


def run_one_segment(
    contract: dict[str, Any], tracers: list[dict[str, Any]],
    expanded: dict[str, list[list[Any]]], spec: dict[str, Any],
    arm_dir: Path, segment_index: int, attempt_index: int, output_root: Path,
) -> dict[str, Any]:
    if not 0 <= segment_index < 20:
        raise IntegrityError("segment index outside frozen range")
    segment_dir = arm_dir / "segments"
    if (arm_dir.is_symlink() or not arm_dir.is_dir() or segment_dir.is_symlink()
            or not segment_dir.is_dir() or segment_dir.resolve().parent != arm_dir.resolve()
            or arm_dir.resolve().parent != (output_root / "arms").resolve()):
        raise IntegrityError("worker segment directory ancestry is unsafe")
    started_ns = time.monotonic_ns()
    if segment_index == 0:
        state = initial_execution_state(contract, tracers, expanded, spec)
    else:
        state = resume_execution_state(contract, expanded, arm_dir, segment_index, spec)
    simulation, tracker, initial, maximum, landmarks, previous_chain, sample_total = state
    expected_start = segment_index * 50_000.0
    if float(simulation.t) != expected_start:
        raise IntegrityError("segment start time readback mismatch")
    stream = hashlib.sha256(SAMPLE_STREAM_DOMAIN + spec["arm_id"].encode("ascii")
                            + struct.pack(">I", segment_index))
    first_sample_index = segment_index * 1000 + (0 if segment_index == 0 else 1)
    last_sample_index = (segment_index + 1) * 1000
    for sample_index in range(first_sample_index, last_sample_index + 1):
        target = sample_index * 50.0
        if target != float(simulation.t):
            simulation.integrate(target, exact_finish_time=1)
        if float(simulation.t) != target:
            raise NumericalError("sample-time readback mismatch")
        update_sample_stream(stream, simulation, sample_index, target)
        rows = sample_tracers(simulation, tracker, target, target in LANDMARKS)
        update_invariant_maximum(maximum, initial, active_snapshot(simulation))
        if rows is not None:
            landmarks[str(int(target))] = {
                "horizon_years": target, "particles": rows,
                "summary": summarize_particles(rows, target),
                "maximum_active_invariant_drifts": dict(maximum),
            }
        enforce_worker_resources(contract, output_root, started_ns)
    new_samples = last_sample_index - first_sample_index + 1
    sample_total += new_samples
    expected_total = last_sample_index + 1
    if sample_total != expected_total:
        raise IntegrityError("sample boundary ownership mismatch")
    state_path, receipt_path = checkpoint_paths(arm_dir, segment_index, attempt_index)
    checkpoint_digest, checkpoint_size, decoded_digest = save_simulation_checkpoint(
        simulation, contract, state_path, boundary_mode="OFFICIAL_SEGMENT",
        end_years=(segment_index + 1) * 50_000.0,
        dt_years=float(spec["dt_years"]), particle_count=int(simulation.N),
    )
    semantic = {
        "arm_id": spec["arm_id"], "configuration_id": spec["configuration_id"],
        "arm_class": spec["arm_class"], "dt_years": spec["dt_years"],
        "segment_index": segment_index, "start_years": expected_start,
        "end_years": (segment_index + 1) * 50_000.0,
        "first_sample_index": first_sample_index, "last_sample_index": last_sample_index,
        "new_sample_count": new_samples, "sample_count_total": sample_total,
        "sampled_state_stream_sha256": stream.hexdigest(),
        "decoded_integrator_state_sha256": decoded_digest,
        "tracker": tracker, "initial_active_invariants": initial,
        "maximum_active_invariant_drifts": maximum, "landmarks": landmarks,
    }
    raw_checkpoint_integrity = {
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_size_bytes": checkpoint_size,
        "raw_checkpoint_integrity_only": True,
    }
    chain_head = semantic_segment_chain(
        previous_chain, semantic_segment_payload(semantic)
    )
    receipt = {
        "schema": CHECKPOINT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        **semantic, **raw_checkpoint_integrity,
        "previous_segment_chain_head": previous_chain,
        "segment_chain_head": chain_head, "checkpoint_filename": state_path.name,
        "provenance": {"attempt_index": attempt_index,
                       "wall_seconds": (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND,
                       "peak_rss_bytes": peak_rss_bytes()},
    }
    enforce_worker_resources(contract, output_root, started_ns)
    atomic_json(receipt_path, receipt)
    return receipt


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_crossings(row: dict[str, Any], minimum_q: float, horizon: float, label: str) -> None:
    crossings: list[float | None] = []
    for threshold in (40, 35, 30):
        value = row[f"first_sampled_q_below_{threshold}_time_year"]
        if (value is not None) != (minimum_q < float(threshold)):
            raise IntegrityError(f"{label} first-passage presence contradicts its prefix minimum")
        if value is not None:
            if (not finite_number(value) or value < 0.0 or value > horizon
                    or value % 50.0 != 0.0):
                raise IntegrityError(f"{label} first-passage value is outside the frozen grid")
            crossings.append(float(value))
        else:
            crossings.append(None)
    # Crossing q<40 must exist before q<35, which must exist before q<30.
    if ((crossings[1] is not None and crossings[0] is None)
            or (crossings[2] is not None and crossings[1] is None)):
        raise IntegrityError(f"{label} first-passage threshold nesting changed")
    finite = [value for value in crossings if value is not None]
    if finite != sorted(finite):
        raise IntegrityError(f"{label} first-passage time ordering changed")


def validate_tracker_shape(tracker: Any, horizon: float) -> list[str]:
    if not isinstance(tracker, list) or len(tracker) != 128:
        raise IntegrityError("segment tracker cardinality changed")
    expected_ids = [f"XP2-B{block:02d}-T{index:02d}" for block in range(8) for index in range(16)]
    fields = {
        "logical_id", "block_index", "index_within_block", "minimum_sampled_q_AU",
        "first_sampled_q_below_30_time_year", "first_sampled_q_below_35_time_year",
        "first_sampled_q_below_40_time_year",
        "all_samples_finite_cartesian_and_osculating",
    }
    for expected_id, row in zip(expected_ids, tracker, strict=True):
        require_keys(row, fields, "segment tracker row")
        block = int(expected_id[5:7]); index = int(expected_id[9:11])
        if (row["logical_id"] != expected_id or row["block_index"] != block
                or row["index_within_block"] != index
                or row["all_samples_finite_cartesian_and_osculating"] is not True
                or not finite_number(row["minimum_sampled_q_AU"])
                or row["minimum_sampled_q_AU"] < 0.0):
            raise IntegrityError("segment tracker row identity/value changed")
        validate_crossings(
            row, float(row["minimum_sampled_q_AU"]), horizon, "segment tracker"
        )
    return expected_ids


def validate_initial_invariants(value: Any) -> None:
    if not isinstance(value, dict):
        raise IntegrityError("initial invariants are not an object")
    require_keys(value, {
        "momentum", "r_com", "v_com", "com_angular", "linear_internal_scale",
        "intrinsic_energy",
    }, "initial invariants")
    for key in ("momentum", "r_com", "v_com", "com_angular"):
        vector = value[key]
        if (not isinstance(vector, list) or len(vector) != 3
                or not all(finite_number(component) for component in vector)):
            raise IntegrityError("initial invariant vector shape/value changed")
    if (not finite_number(value["linear_internal_scale"])
            or value["linear_internal_scale"] <= 0.0
            or not finite_number(value["intrinsic_energy"])
            or value["intrinsic_energy"] == 0.0
            or vector_norm(value["com_angular"]) == 0.0):
        raise IntegrityError("initial invariant denominator changed")


def validate_maximum_drifts(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not an object")
    require_keys(value, {
        "relative_compensated_active_energy_drift",
        "relative_active_com_angular_momentum_vector_drift",
        "scale_normalized_active_linear_momentum_residual",
    }, label)
    if any(not finite_number(number) or number < 0.0 for number in value.values()):
        raise IntegrityError(f"{label} contains an invalid value")


def validate_landmark_particle(
    row: dict[str, Any], expected_id: str, horizon: float,
) -> None:
    landmark_fields = {
        "logical_id", "block_index", "index_within_block", "minimum_sampled_q_AU",
        "first_sampled_q_below_30_time_year", "first_sampled_q_below_35_time_year",
        "first_sampled_q_below_40_time_year",
        "all_samples_finite_cartesian_and_osculating", "final_a_AU", "final_e",
        "final_i_deg", "final_q_AU", "final_distance_AU", "final_finite_and_bound",
    }
    require_keys(row, landmark_fields, "attempt landmark particle")
    block = int(expected_id[5:7]); index = int(expected_id[9:11])
    if (row["logical_id"] != expected_id or row["block_index"] != block
            or row["index_within_block"] != index
            or row["all_samples_finite_cartesian_and_osculating"] is not True
            or type(row["final_finite_and_bound"]) is not bool):
        raise IntegrityError("attempt landmark particle identity/boolean changed")
    for field in (
        "minimum_sampled_q_AU", "final_a_AU", "final_e", "final_i_deg",
        "final_q_AU", "final_distance_AU",
    ):
        if not finite_number(row[field]):
            raise IntegrityError("attempt landmark particle contains non-finite metric")
    if (row["minimum_sampled_q_AU"] < 0.0 or row["final_e"] < 0.0
            or not 0.0 <= row["final_i_deg"] <= 180.0 or row["final_q_AU"] < 0.0
            or row["final_distance_AU"] <= 0.0
            or row["final_finite_and_bound"] is not (
                row["final_a_AU"] > 0.0 and row["final_e"] < 1.0
            )):
        raise IntegrityError("attempt landmark particle metric domain changed")
    validate_crossings(
        row, float(row["minimum_sampled_q_AU"]), horizon, "attempt landmark particle"
    )


def validate_attempt_payload(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]], arm_dir: Path,
    arm_id: str, segment_index: int, attempt_index: int, *,
    state_path_override: Path | None = None,
    receipt_path_override: Path | None = None,
) -> dict[str, Any]:
    logical_state_path, logical_receipt_path = checkpoint_paths(
        arm_dir, segment_index, attempt_index
    )
    state_path = state_path_override or logical_state_path
    receipt_path = receipt_path_override or logical_receipt_path
    if (state_path_override is None) != (receipt_path_override is None):
        raise IntegrityError("attempt validation overrides are incomplete")
    if state_path_override is None and (
        state_path.parent.resolve() != (arm_dir / "segments").resolve()
        or receipt_path.parent.resolve() != (arm_dir / "segments").resolve()
    ):
        raise IntegrityError("attempt artifact path escaped its segment directory")
    if (state_path.is_symlink() or not state_path.is_file() or state_path.stat().st_nlink != 1
            or receipt_path.is_symlink() or not receipt_path.is_file()
            or receipt_path.stat().st_nlink != 1):
        raise IntegrityError("attempt artifact path is unsafe")
    if not 0 < state_path.stat().st_size <= MAX_PRIMARY_CHECKPOINT_BYTES:
        raise IntegrityError("attempt checkpoint exceeds its registered byte bound")
    receipt = strict_json(receipt_path)
    semantic_fields = set(SEGMENT_SEMANTIC_FIELDS)
    require_keys(receipt, {
        "schema", "experiment_id", *semantic_fields, *RAW_CHECKPOINT_INTEGRITY_FIELDS,
        "previous_segment_chain_head",
        "segment_chain_head", "checkpoint_filename", "provenance",
    }, "attempt segment receipt")
    spec = arm_specification(contract, arm_id)
    if (
        receipt["schema"] != CHECKPOINT_SCHEMA or receipt["experiment_id"] != EXPERIMENT_ID
        or receipt["arm_id"] != arm_id or receipt["configuration_id"] != spec["configuration_id"]
        or receipt["arm_class"] != spec["arm_class"] or receipt["dt_years"] != spec["dt_years"]
        or receipt["segment_index"] != segment_index
        or receipt["checkpoint_filename"] != logical_state_path.name
        or receipt["checkpoint_sha256"] != sha256_file(state_path)
        or receipt["checkpoint_size_bytes"] != state_path.stat().st_size
        or receipt["raw_checkpoint_integrity_only"] is not True
    ):
        raise IntegrityError("attempt segment identity/hash changed")
    expected_first = segment_index * 1000 + (0 if segment_index == 0 else 1)
    expected_last = (segment_index + 1) * 1000
    if (
        receipt["start_years"] != segment_index * 50_000.0
        or receipt["end_years"] != (segment_index + 1) * 50_000.0
        or receipt["first_sample_index"] != expected_first
        or receipt["last_sample_index"] != expected_last
        or receipt["new_sample_count"] != expected_last - expected_first + 1
        or receipt["sample_count_total"] != expected_last + 1
        or not valid_sha256(receipt["sampled_state_stream_sha256"])
    ):
        raise IntegrityError("attempt segment sampling ownership changed")
    expected_ids = validate_tracker_shape(receipt["tracker"], float(receipt["end_years"]))
    validate_initial_invariants(receipt["initial_active_invariants"])
    validate_maximum_drifts(receipt["maximum_active_invariant_drifts"], "maximum drifts")
    registered_initial = active_snapshot(build_simulation(contract, expanded, spec))
    if receipt["initial_active_invariants"] != registered_initial:
        raise IntegrityError("initial invariant baseline differs from registered initial rows")
    expected_landmarks = {str(int(value)) for value in LANDMARKS if value <= receipt["end_years"]}
    if set(receipt["landmarks"]) != expected_landmarks:
        raise IntegrityError("attempt landmark set changed")
    for key, landmark in receipt["landmarks"].items():
        require_keys(landmark, {
            "horizon_years", "particles", "summary", "maximum_active_invariant_drifts"
        }, "attempt landmark")
        if landmark["horizon_years"] != float(key):
            raise IntegrityError("attempt landmark horizon changed")
        if [row.get("logical_id") for row in landmark["particles"]] != expected_ids:
            raise IntegrityError("attempt landmark tracer order changed")
        for expected_id, row in zip(expected_ids, landmark["particles"], strict=True):
            validate_landmark_particle(row, expected_id, float(key))
        if landmark["summary"] != summarize_particles(landmark["particles"], float(key)):
            raise IntegrityError("attempt landmark summary changed")
        validate_maximum_drifts(landmark["maximum_active_invariant_drifts"], "landmark drifts")
    if segment_index == 0:
        previous = INITIAL_SEGMENT_CHAIN
    else:
        prior_receipt = load_completed_segment(arm_dir, segment_index - 1)
        previous = prior_receipt["segment_chain_head"]
        if receipt["initial_active_invariants"] != prior_receipt["initial_active_invariants"]:
            raise IntegrityError("initial invariant baseline changed across segments")
        if any(receipt["maximum_active_invariant_drifts"][key]
               < prior_receipt["maximum_active_invariant_drifts"][key]
               for key in receipt["maximum_active_invariant_drifts"]):
            raise IntegrityError("maximum invariant drift decreased across segments")
        old_landmarks = prior_receipt["landmarks"]
        if any(receipt["landmarks"].get(key) != value for key, value in old_landmarks.items()):
            raise IntegrityError("historical landmark changed across segments")
        for old, new in zip(prior_receipt["tracker"], receipt["tracker"], strict=True):
            if (new["logical_id"] != old["logical_id"]
                    or new["minimum_sampled_q_AU"] > old["minimum_sampled_q_AU"]):
                raise IntegrityError("tracker identity/minimum regressed across segments")
            for threshold in (30, 35, 40):
                field = f"first_sampled_q_below_{threshold}_time_year"
                if old[field] is not None and new[field] != old[field]:
                    raise IntegrityError("first-passage time changed across segments")
                if old[field] is None and new[field] is not None and not (
                    receipt["start_years"] < new[field] <= receipt["end_years"]
                ):
                    raise IntegrityError("new first passage is outside its segment")
    new_landmark_key = str(int(receipt["end_years"]))
    if new_landmark_key in receipt["landmarks"] and (
        segment_index == 0 or new_landmark_key not in prior_receipt["landmarks"]
    ):
        landmark_rows = receipt["landmarks"][new_landmark_key]["particles"]
        for tracker_row, landmark_row in zip(receipt["tracker"], landmark_rows, strict=True):
            for field in (
                "logical_id", "block_index", "index_within_block", "minimum_sampled_q_AU",
                "first_sampled_q_below_30_time_year", "first_sampled_q_below_35_time_year",
                "first_sampled_q_below_40_time_year",
                "all_samples_finite_cartesian_and_osculating",
            ):
                if tracker_row[field] != landmark_row[field]:
                    raise IntegrityError("new landmark is inconsistent with tracker endpoint")
        if (receipt["landmarks"][new_landmark_key]["maximum_active_invariant_drifts"]
                != receipt["maximum_active_invariant_drifts"]):
            raise IntegrityError("new landmark drift maxima differ from segment endpoint")
    semantic = semantic_segment_payload(receipt)
    expected_chain = semantic_segment_chain(previous, semantic)
    if receipt["previous_segment_chain_head"] != previous or receipt["segment_chain_head"] != expected_chain:
        raise IntegrityError("attempt predecessor or semantic chain changed")
    provenance = receipt["provenance"]
    require_keys(provenance, {"attempt_index", "wall_seconds", "peak_rss_bytes"}, "attempt provenance")
    if (provenance["attempt_index"] != attempt_index
            or not finite_number(provenance["wall_seconds"])
            or provenance["wall_seconds"] < 0.0
            or not isinstance(provenance["peak_rss_bytes"], int)
            or isinstance(provenance["peak_rss_bytes"], bool)
            or provenance["peak_rss_bytes"] < 0
            or any(not valid_sha256(receipt[key]) for key in (
                "sampled_state_stream_sha256", "decoded_integrator_state_sha256",
                "checkpoint_sha256", "previous_segment_chain_head", "segment_chain_head",
            ))):
        raise IntegrityError("attempt provenance index changed")
    simulation = get_rebound(contract).Simulation(str(state_path))
    expected_active = 5 if spec["configuration_id"] == "M0" else 6
    continuation = decoded_continuation_projection(simulation)
    if (
        decoded_state_sha256(simulation) != receipt["decoded_integrator_state_sha256"]
        or float(simulation.t) != receipt["end_years"]
        or float(simulation.G) != contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        or simulation.N != expected_active + 128 or simulation.N_active != expected_active
        or simulation.integrator != "mercurius" or simulation.dt != spec["dt_years"]
        or simulation.testparticle_type != 0 or simulation.ri_mercurius.r_crit_hill != 3.0
        or int(simulation.ri_mercurius.safe_mode) != 1
        or str(simulation.collision) != "none"
    ):
        raise IntegrityError("attempt checkpoint decoded integrator state/settings changed")
    validate_decoded_continuation_settings(
        continuation, end_years=float(receipt["end_years"]),
        dt_years=float(spec["dt_years"]), particle_count=expected_active + 128,
    )
    expected_particle_ids = ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"]
    expected_masses = [
        contract["design_core"]["common_active_system"]["sun_mass_Msun"],
        *(body["mass_Msun"] for body in contract["design_core"]["common_active_system"]["giants"]),
    ]
    if spec["configuration_id"] != "M0":
        case_id = spec["configuration_id"].split("-")[0]
        model = next(item for item in contract["design_core"]["m1_physical_cases"]
                     if item["id"] == case_id)
        expected_particle_ids.append(f"XP2-{spec['configuration_id']}")
        expected_masses.append(model["mass_Mearth"] * contract["design_core"][
            "common_active_system"
        ]["earth_to_sun_mass_ratio"])
    expected_particle_ids.extend(expected_ids); expected_masses.extend([0.0] * 128)
    rebound = get_rebound(contract)
    for index, (logical_id, mass) in enumerate(zip(expected_particle_ids, expected_masses, strict=True)):
        particle = simulation.particles[index]
        if (int(particle.hash.value) != int(rebound.hash(logical_id).value)
                or float(particle.m) != mass or float(particle.r) != 0.0):
            raise IntegrityError("checkpoint particle order/hash/mass/role changed")
    endpoint_drift = {
        "relative_compensated_active_energy_drift": 0.0,
        "relative_active_com_angular_momentum_vector_drift": 0.0,
        "scale_normalized_active_linear_momentum_residual": 0.0,
    }
    update_invariant_maximum(
        endpoint_drift, receipt["initial_active_invariants"], active_snapshot(simulation)
    )
    if any(receipt["maximum_active_invariant_drifts"][key] < value
           for key, value in endpoint_drift.items()):
        raise IntegrityError("maximum invariant drift is below decoded endpoint drift")
    decoded_tracker = [dict(row) for row in receipt["tracker"]]
    endpoint_rows = sample_tracers(
        simulation, decoded_tracker, float(receipt["end_years"]),
        new_landmark_key in receipt["landmarks"],
    )
    if decoded_tracker != receipt["tracker"]:
        raise IntegrityError("tracker prefix omits or contradicts the decoded endpoint sample")
    if new_landmark_key in receipt["landmarks"]:
        if endpoint_rows is None:
            raise IntegrityError("decoded landmark endpoint reconstruction failed")
        stored_rows = receipt["landmarks"][new_landmark_key]["particles"]
        if stored_rows != endpoint_rows:
            raise IntegrityError("landmark row differs from the decoded checkpoint endpoint")
    del particle, simulation
    gc.collect()
    return receipt


def commit_segment_attempt(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]], arm_dir: Path,
    arm_id: str, segment_index: int, attempt_index: int,
    parent_elapsed_seconds: float, output_root: Path,
    execution_elapsed_seconds: float,
) -> dict[str, Any]:
    state_path, receipt_path = checkpoint_paths(arm_dir, segment_index, attempt_index)
    receipt = validate_attempt_payload(
        contract, expanded, arm_dir, arm_id, segment_index, attempt_index
    )
    ledger_rows = read_jsonl(output_root / "attempt_ledger.jsonl")
    failure_dir = output_root / "failures"
    for row in ledger_rows:
        if (row.get("event") != "FAIL" or row.get("arm_id") != arm_id
                or row.get("segment_index") != segment_index):
            continue
        failed_receipt = strict_json(failure_dir / row["failure_receipt_filename"])
        complete = failed_receipt["complete_uncommitted_attempt"]
        if complete is not None:
            require_complete_attempt_semantic_match(receipt, complete)
    caps = contract["resource_caps_per_execution"]
    if (parent_elapsed_seconds >= float(caps["max_wall_seconds_per_segment_attempt"])
            or float(receipt["provenance"]["wall_seconds"])
            >= float(caps["max_wall_seconds_per_segment_attempt"])
            or int(receipt["provenance"]["peak_rss_bytes"])
            > int(caps["max_peak_rss_bytes_per_process"])
            or execution_elapsed_seconds >= float(caps["max_wall_seconds_total"])
            or directory_bytes(output_root) > int(caps["max_output_bytes"])
            or shutil.disk_usage(output_root).free < int(caps["minimum_free_disk_bytes"])
            or peak_rss_bytes() > int(caps["max_peak_rss_bytes_per_process"])):
        raise ResourceLimitError("completed attempt exceeded a terminal resource cap")
    prior_receipts = sorted(receipt_path.parent.glob(
        f"segment_{segment_index:02d}_attempt_*_receipt.json"
    ))
    for prior_path in prior_receipts:
        if prior_path == receipt_path:
            continue
        try:
            prior_attempt = int(prior_path.stem.split("_")[3])
        except (IndexError, ValueError) as exc:
            raise IntegrityError("attempt receipt filename changed") from exc
        prior = validate_attempt_payload(
            contract, expanded, arm_dir, arm_id, segment_index, prior_attempt
        )
        if (prior.get("segment_chain_head") != receipt["segment_chain_head"]
                or prior.get("decoded_integrator_state_sha256")
                != receipt["decoded_integrator_state_sha256"]):
            raise IntegrityError("NONDETERMINISTIC_RESUME: conflicting complete attempt artifact")
    commit = {
        "schema": SEGMENT_COMMIT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "arm_id": arm_id, "segment_index": segment_index,
        "attempt_receipt_filename": receipt_path.name,
        "attempt_receipt_sha256": sha256_file(receipt_path),
        "checkpoint_filename": state_path.name,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "raw_checkpoint_integrity_only": True,
        "decoded_integrator_state_sha256": receipt["decoded_integrator_state_sha256"],
        "segment_chain_head": receipt["segment_chain_head"],
        "parent_terminal_validation": "CLEAN_EXIT_AND_WITHIN_WALL_RSS_OUTPUT_AND_DISK_CAPS",
    }
    commit_payload_size = len(serialized_json(commit))
    if (directory_bytes(output_root) + commit_payload_size > int(caps["max_output_bytes"])
            or shutil.disk_usage(output_root).free
            < int(caps["minimum_free_disk_bytes"]) + commit_payload_size):
        raise ResourceLimitError("parent commit would exceed an output or disk cap")
    atomic_json(segment_commit_path(arm_dir, segment_index), commit)
    reread = load_completed_segment(arm_dir, segment_index)
    if reread != receipt:
        raise IntegrityError("published parent commit did not re-read the validated receipt")
    validate_attempt_payload(
        contract, expanded, arm_dir, arm_id, segment_index, attempt_index
    )
    if (directory_bytes(output_root) > int(caps["max_output_bytes"])
            or shutil.disk_usage(output_root).free < int(caps["minimum_free_disk_bytes"])):
        raise ResourceLimitError("published parent commit exceeded a terminal resource cap")
    return reread


def censored_values(rows: Sequence[dict[str, Any]], threshold: int, horizon: float) -> list[float]:
    key = f"first_sampled_q_below_{threshold}_time_year"
    return [(horizon if row[key] is None else float(row[key])) / horizon for row in rows]


def compare_timestep_pair(
    primary: dict[str, Any], audit: dict[str, Any], horizon: float,
    contract: dict[str, Any], configuration_id: str,
) -> dict[str, Any]:
    primary_rows = particle_index(primary)
    audit_rows = particle_index(audit)
    if set(primary_rows) != set(audit_rows):
        raise IntegrityError("timestep pair tracer IDs differ")
    ids = sorted(primary_rows)
    gates = contract["numerical_gates"]["timestep_pairs"]
    metrics: dict[str, Any] = {"configuration_id": configuration_id, "horizon_years": horizon}
    checks: dict[str, bool] = {}
    for threshold in (30, 35, 40):
        left_hits = [hit(primary_rows[logical_id], float(threshold)) for logical_id in ids]
        right_hits = [hit(audit_rows[logical_id], float(threshold)) for logical_id in ids]
        difference = abs(sum(left_hits) - sum(right_hits))
        discordance = sum(left != right for left, right in zip(left_hits, right_hits, strict=True))
        metrics[f"q{threshold}_hit_count_absolute_difference"] = difference
        metrics[f"q{threshold}_paired_indicator_discordance"] = discordance
        checks[f"q{threshold}_count_within_gate"] = difference <= int(
            gates[f"max_q{threshold}_count_difference"]
        )
        checks[f"q{threshold}_discordance_within_gate"] = discordance <= int(
            gates[f"max_q{threshold}_indicator_discordance"]
        )
    primary_list = [primary_rows[logical_id] for logical_id in ids]
    audit_list = [audit_rows[logical_id] for logical_id in ids]
    bound_difference = abs(sum(row["final_finite_and_bound"] for row in primary_list)
                           - sum(row["final_finite_and_bound"] for row in audit_list))
    w1_minimum = empirical_w1(
        [row["minimum_sampled_q_AU"] for row in primary_list],
        [row["minimum_sampled_q_AU"] for row in audit_list],
    )
    w1_final_q = empirical_w1([row["final_q_AU"] for row in primary_list],
                              [row["final_q_AU"] for row in audit_list])
    w1_final_i = empirical_w1([row["final_i_deg"] for row in primary_list],
                              [row["final_i_deg"] for row in audit_list])
    w1_first_q30 = empirical_w1(censored_values(primary_list, 30, horizon),
                                censored_values(audit_list, 30, horizon))
    w1_first_q35 = empirical_w1(censored_values(primary_list, 35, horizon),
                                censored_values(audit_list, 35, horizon))
    metrics.update({
        "final_bound_count_absolute_difference": bound_difference,
        "w1_minimum_sampled_q_AU": w1_minimum, "w1_final_q_AU": w1_final_q,
        "w1_final_i_deg": w1_final_i,
        "w1_censored_first_q30_divided_by_horizon": w1_first_q30,
        "w1_censored_first_q35_divided_by_horizon": w1_first_q35,
    })
    checks.update({
        "bound_count_within_gate": bound_difference <= int(gates["max_bound_count_difference"]),
        "w1_minimum_q_within_gate": w1_minimum <= float(gates["max_w1_minimum_sampled_q_AU"]),
        "w1_final_q_within_gate": w1_final_q <= float(gates["max_w1_final_q_AU"]),
        "w1_final_i_within_gate": w1_final_i <= float(gates["max_w1_final_i_deg"]),
        "w1_first_q30_within_gate": w1_first_q30 <= float(
            gates["max_w1_censored_first_q30_divided_by_horizon"]
        ),
        "w1_first_q35_within_gate": w1_first_q35 <= float(
            gates["max_w1_censored_first_q35_divided_by_horizon"]
        ),
    })
    metrics["checks"] = checks
    metrics["passes"] = all(checks.values())
    return metrics


def active_gate_checks(landmark: dict[str, Any], contract: dict[str, Any]) -> dict[str, bool]:
    maximum = landmark["maximum_active_invariant_drifts"]
    gates = contract["numerical_gates"]
    return {
        "all_finite": bool(
            landmark["summary"]["all_particles_all_samples_finite_cartesian_and_osculating"]
        ),
        "energy": maximum["relative_compensated_active_energy_drift"]
        <= float(gates["max_relative_compensated_active_energy_drift"]),
        "angular": maximum["relative_active_com_angular_momentum_vector_drift"]
        <= float(gates["max_relative_active_com_angular_momentum_vector_drift"]),
        "linear": maximum["scale_normalized_active_linear_momentum_residual"]
        <= float(gates["max_scale_normalized_active_linear_momentum_residual"]),
    }


def analyze_primary(
    completed: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    if set(completed) != set(ALL_ARM_IDS):
        raise IntegrityError("analysis requires all 50 arms")
    horizon_arms: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "primary": {}, "audit": {}
    }
    invariant_checks: list[dict[str, Any]] = []
    for horizon in LANDMARKS:
        horizon_key = str(int(horizon))
        primary_rows: dict[str, dict[str, Any]] = {}
        audit_rows: dict[str, dict[str, Any]] = {}
        for arm_id in ALL_ARM_IDS:
            receipt = completed[arm_id]
            landmark = receipt["landmarks"].get(horizon_key)
            if not isinstance(landmark, dict):
                raise IntegrityError("required landmark missing")
            checks = active_gate_checks(landmark, contract)
            invariant_checks.append({"arm_id": arm_id, "horizon_years": horizon,
                                     "checks": checks, "passes": all(checks.values())})
            if arm_id.startswith("AUDIT-"):
                audit_rows[AUDIT_TO_PRIMARY[arm_id]] = landmark
            else:
                primary_rows[arm_id] = landmark
        horizon_arms["primary"][horizon_key] = primary_rows
        horizon_arms["audit"][horizon_key] = audit_rows

    pair_rows: list[dict[str, Any]] = []
    for horizon in LANDMARKS:
        key = str(int(horizon))
        for configuration_id in PRIMARY_ARM_IDS:
            pair_rows.append(compare_timestep_pair(
                horizon_arms["primary"][key][configuration_id],
                horizon_arms["audit"][key][configuration_id],
                horizon, contract, configuration_id,
            ))
    numerical_passes = all(row["passes"] for row in pair_rows) and all(
        row["passes"] for row in invariant_checks
    )

    effects: dict[str, dict[str, Any]] = {"primary": {}, "audit": {}}
    raw_labels: dict[str, dict[str, str]] = {"primary": {}, "audit": {}}
    for resolution in ("primary", "audit"):
        for horizon in CLASSIFICATION_HORIZONS:
            key = str(int(horizon))
            arms = {arm_id: horizon_arms[resolution][key][arm_id] for arm_id in PRIMARY_ARM_IDS}
            effect = structural_effects(arms, 35.0)
            effects[resolution][key] = effect
            raw_labels[resolution][key] = structural_label(effect)

    analysis_state: str
    primary_screen_label: str | None = None
    support_rows: dict[str, Any] = {}
    promoted: dict[str, str] = {}
    if not numerical_passes:
        analysis_state = "NUMERICALLY_UNRESOLVED"
    elif any(raw_labels["primary"][str(int(horizon))]
             != raw_labels["audit"][str(int(horizon))]
             for horizon in CLASSIFICATION_HORIZONS):
        analysis_state = "TIMESTEP_SENSITIVE"
    elif raw_labels["primary"]["500000"] != raw_labels["primary"]["1000000"]:
        analysis_state = "HORIZON_SENSITIVE"
    else:
        for resolution in ("primary", "audit"):
            arms = {arm_id: horizon_arms[resolution]["1000000"][arm_id]
                    for arm_id in PRIMARY_ARM_IDS}
            support_rows[resolution] = event_support(arms)
            promoted[resolution] = apply_event_floor(
                raw_labels[resolution]["1000000"], support_rows[resolution]
            )
        if promoted["primary"] != promoted["audit"]:
            analysis_state = "TIMESTEP_SENSITIVE"
        else:
            analysis_state = "PRIMARY_NUMERICS_COMPLETE_AWAITING_REPLAY_AND_DOP853"
            primary_screen_label = promoted["primary"]

    bridge_effects = {
        resolution: structural_effects(
            {arm_id: horizon_arms[resolution]["250000"][arm_id]
             for arm_id in PRIMARY_ARM_IDS}, 35.0
        ) for resolution in ("primary", "audit")
    }
    return {
        "analysis_state": analysis_state,
        "official_classification": None,
        "primary_screen_label": primary_screen_label,
        "classification_suppressed_until_replay_and_dop853": True,
        "structural_effects_q35": effects, "structural_raw_labels_q35": raw_labels,
        "event_support_1M_by_resolution": support_rows,
        "event_floor_promoted_labels_by_resolution": promoted,
        "bridge_250k_effects_q35_descriptive_only": bridge_effects,
        "timestep_pair_gates": pair_rows, "active_invariant_gates": invariant_checks,
        "all_primary_numerical_gates_pass": numerical_passes,
    }


def validate_segment_chain(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]],
    arm_dir: Path, arm_id: str,
) -> tuple[int, dict[str, Any] | None]:
    segment_dir = arm_dir / "segments"
    if (arm_dir.is_symlink() or not arm_dir.is_dir() or segment_dir.is_symlink()
            or not segment_dir.is_dir() or segment_dir.resolve().parent != arm_dir.resolve()):
        raise IntegrityError("segment chain directory ancestry is unsafe")
    previous = INITIAL_SEGMENT_CHAIN
    last: dict[str, Any] | None = None
    completed = 0
    for segment_index in range(20):
        commit_path = segment_commit_path(arm_dir, segment_index)
        if not commit_path.exists():
            break
        receipt = load_completed_segment(arm_dir, segment_index)
        attempt_index = receipt.get("provenance", {}).get("attempt_index")
        if not isinstance(attempt_index, int) or validate_attempt_payload(
            contract, expanded, arm_dir, arm_id, segment_index, attempt_index
        ) != receipt:
            raise IntegrityError("committed segment failed full payload validation")
        if (receipt["arm_id"] != arm_id or receipt["previous_segment_chain_head"] != previous
                or receipt["segment_index"] != segment_index):
            raise IntegrityError("segment chain identity/order changed")
        semantic = semantic_segment_payload(receipt)
        expected_chain = semantic_segment_chain(previous, semantic)
        if receipt["segment_chain_head"] != expected_chain:
            raise IntegrityError("segment semantic hash-chain mismatch")
        previous = expected_chain; last = receipt; completed += 1
    if segment_dir.exists():
        commit_indices = sorted(
            int(path.stem.split("_")[1]) for path in segment_dir.glob("segment_*_commit.json")
        )
        if commit_indices != list(range(completed)):
            raise IntegrityError("segment commits contain a gap or extra segment")
        expected_names: set[str] = set()
        for segment_index in range(completed):
            commit_path = segment_commit_path(arm_dir, segment_index)
            commit = strict_json(commit_path)
            expected_names.update({
                commit_path.name, commit["attempt_receipt_filename"], commit["checkpoint_filename"],
            })
        entries = list(segment_dir.iterdir())
        if (any(entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1
                for entry in entries) or {entry.name for entry in entries} != expected_names):
            raise IntegrityError("segment directory contains an uncommitted or extra artifact")
    return completed, last


def attempt_artifact_source_names(segment_index: int, attempt_index: int) -> tuple[str, ...]:
    return (
        f"segment_{segment_index:02d}_attempt_{attempt_index:02d}_state.bin",
        f"segment_{segment_index:02d}_attempt_{attempt_index:02d}_receipt.json",
        f".segment_{segment_index:02d}_attempt_{attempt_index:02d}_state.bin.pending",
        f".segment_{segment_index:02d}_attempt_{attempt_index:02d}_receipt.json.pending",
        f".segment_{segment_index:02d}_commit.json.pending",
    )


def attempt_quarantine_names(
    arm_id: str, segment_index: int, attempt_index: int,
) -> set[str]:
    return {
        f"{arm_id}_attempt_{attempt_index:02d}_{name.lstrip('.')}"
        for name in attempt_artifact_source_names(segment_index, attempt_index)
    } | {torn_failure_receipt_quarantine_name(
        arm_id, segment_index, attempt_index
    )}


def torn_failure_receipt_quarantine_name(
    arm_id: str, segment_index: int, attempt_index: int,
) -> str:
    return (
        f"{arm_id}_attempt_{attempt_index:02d}_failure_segment_"
        f"{segment_index:02d}_receipt_pending_bytes.bin"
    )


def validated_complete_uncommitted_attempt(
    contract: dict[str, Any], expanded: dict[str, list[list[Any]]],
    output_root: Path, arm_id: str, segment_index: int, attempt_index: int,
) -> dict[str, Any] | None:
    arm_dir = output_root / "arms" / arm_id
    source_state, source_receipt = checkpoint_paths(
        arm_dir, segment_index, attempt_index
    )
    failure_dir = output_root / "failures"
    target_state = failure_dir / (
        f"{arm_id}_attempt_{attempt_index:02d}_{source_state.name}"
    )
    target_receipt = failure_dir / (
        f"{arm_id}_attempt_{attempt_index:02d}_{source_receipt.name}"
    )
    source_receipt_pending = source_receipt.with_name(f".{source_receipt.name}.pending")
    target_receipt_pending = failure_dir / (
        f"{arm_id}_attempt_{attempt_index:02d}_{source_receipt.name}.pending"
    )

    def promote_complete_pending(pending: Path, final: Path) -> None:
        if not pending.exists() and not pending.is_symlink():
            return
        if final.exists() or final.is_symlink():
            raise IntegrityError("complete attempt has final and pending receipts")
        if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
            raise IntegrityError("complete attempt pending receipt is unsafe")
        try:
            candidate = strict_json(pending)
        except (IntegrityError, ValueError, OSError):
            return
        if pending.read_bytes() != serialized_json(candidate):
            raise IntegrityError("complete attempt pending receipt is noncanonical")
        os.replace(pending, final)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    promote_complete_pending(source_receipt_pending, source_receipt)
    promote_complete_pending(target_receipt_pending, target_receipt)
    states = [path for path in (source_state, target_state)
              if path.exists() or path.is_symlink()]
    receipts = [path for path in (source_receipt, target_receipt)
                if path.exists() or path.is_symlink()]
    if len(states) > 1 or len(receipts) > 1:
        raise IntegrityError("complete attempt has duplicate source/quarantine artifacts")
    if not states or not receipts:
        return None
    state_path = states[0]; receipt_path = receipts[0]
    if ((state_path == target_state and state_path.parent.resolve() != failure_dir.resolve())
            or (receipt_path == target_receipt
                and receipt_path.parent.resolve() != failure_dir.resolve())):
        raise IntegrityError("quarantined complete attempt escaped failure root")
    receipt = validate_attempt_payload(
        contract, expanded, arm_dir, arm_id, segment_index, attempt_index,
        state_path_override=state_path, receipt_path_override=receipt_path,
    )
    return complete_attempt_evidence(receipt, receipt_path)


def quarantine_attempt_artifacts(
    output_root: Path, arm_id: str, segment_index: int, attempt_index: int,
) -> list[dict[str, Any]]:
    arm_dir = output_root / "arms" / arm_id
    if segment_commit_path(arm_dir, segment_index).exists():
        raise IntegrityError("refusing to quarantine an attempt with a parent commit")
    segment_dir = arm_dir / "segments"; failure_dir = output_root / "failures"
    if (arm_dir.is_symlink() or not arm_dir.is_dir() or segment_dir.is_symlink()
            or not segment_dir.is_dir() or segment_dir.resolve().parent != arm_dir.resolve()):
        raise IntegrityError("failed attempt segment directory is unsafe")
    failure_dir.mkdir(exist_ok=True)
    if (failure_dir.is_symlink() or not failure_dir.is_dir()
            or failure_dir.resolve().parent != output_root.resolve()):
        raise IntegrityError("failure quarantine directory is unsafe")
    stems = attempt_artifact_source_names(segment_index, attempt_index)
    target_names: list[str] = []
    torn_name = torn_failure_receipt_quarantine_name(
        arm_id, segment_index, attempt_index
    )
    torn_path = failure_dir / torn_name
    if torn_path.exists() or torn_path.is_symlink():
        if torn_path.is_symlink() or not torn_path.is_file() \
                or torn_path.stat().st_nlink != 1:
            raise IntegrityError("torn failure-receipt evidence is unsafe")
        target_names.append(torn_name)
    for name in stems:
        source = segment_dir / name
        target_name = f"{arm_id}_attempt_{attempt_index:02d}_{name.lstrip('.')}"
        target = failure_dir / target_name
        if not source.exists() and not source.is_symlink():
            if target.exists() or target.is_symlink():
                target_names.append(target_name)
            continue
        if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
            raise IntegrityError("failed attempt artifact is unsafe")
        if target.exists() or target.is_symlink():
            raise IntegrityError("both failed source and quarantine target exist")
        os.replace(source, target)
        target_names.append(target_name)
    for directory in (segment_dir, failure_dir):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return failure_quarantine_inventory(failure_dir, target_names)


def validate_failure_receipt_payload(
    receipt: dict[str, Any], start_row: dict[str, Any], filename: str,
    failure_dir: Path,
) -> None:
    published_path = failure_dir / filename
    if (published_path.exists() or published_path.is_symlink()) and (
        published_path.is_symlink() or not published_path.is_file()
        or published_path.stat().st_nlink != 1
        or published_path.read_bytes() != serialized_json(receipt)
    ):
        raise IntegrityError("primary failure receipt bytes are noncanonical")
    require_keys(receipt, {
        "schema", "experiment_id", "execution_label", "arm_id", "segment_index",
        "attempt_index", "failure_class", "return_code",
        "predecessor_segment_chain_head", "input_key_sha256",
        "start_sequence", "fail_event_sha256", "quarantined_artifacts",
        "complete_uncommitted_attempt", "message", "authorizes_analysis",
    }, "primary failure receipt")
    if (
        receipt["schema"] != FAILURE_SCHEMA or receipt["experiment_id"] != EXPERIMENT_ID
        or receipt["execution_label"] != start_row["execution_label"]
        or receipt["arm_id"] != start_row["arm_id"]
        or type(receipt["segment_index"]) is not int
        or receipt["segment_index"] != start_row["segment_index"]
        or type(receipt["attempt_index"]) is not int
        or receipt["attempt_index"] != start_row["attempt_index"]
        or receipt["predecessor_segment_chain_head"]
        != start_row["predecessor_segment_chain_head"]
        or receipt["input_key_sha256"] != start_row["input_key_sha256"]
        or type(receipt["start_sequence"]) is not int
        or receipt["start_sequence"] != start_row["sequence"]
        or receipt["failure_class"] not in FAILURE_CLASSES
        or receipt["return_code"] == 0
        or (receipt["return_code"] is not None
            and (not isinstance(receipt["return_code"], int)
                 or isinstance(receipt["return_code"], bool)))
        or receipt["message"] != REDACTED_FAILURE_MESSAGE
        or receipt["authorizes_analysis"] is not False
        or filename != failure_receipt_filename(
            receipt["arm_id"], receipt["segment_index"], receipt["attempt_index"]
        )
    ):
        raise IntegrityError("primary failure receipt binding changed")
    core = failure_event_core(
        execution_label=receipt["execution_label"], arm_id=receipt["arm_id"],
        segment_index=receipt["segment_index"],
        attempt_index=receipt["attempt_index"],
        start_sequence=receipt["start_sequence"], return_code=receipt["return_code"],
        failure_class=receipt["failure_class"],
        complete_uncommitted_attempt=receipt["complete_uncommitted_attempt"],
    )
    if receipt["fail_event_sha256"] != failure_event_sha256(core):
        raise IntegrityError("primary failure receipt event digest changed")
    inventory = receipt["quarantined_artifacts"]
    if not isinstance(inventory, list):
        raise IntegrityError("failure quarantine inventory changed")
    expected_names: list[str] = []
    previous = ""
    allowed_names = attempt_quarantine_names(
        receipt["arm_id"], receipt["segment_index"], receipt["attempt_index"]
    )
    for row in inventory:
        require_keys(row, {"filename", "size_bytes", "sha256"}, "quarantine row")
        name = row["filename"]
        if (not isinstance(name, str) or name not in allowed_names or name <= previous
                or not isinstance(row["size_bytes"], int)
                or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 0
                or not valid_sha256(row["sha256"])):
            raise IntegrityError("failure quarantine row changed")
        path = failure_dir / name
        if (path.resolve().parent != failure_dir.resolve() or path.is_symlink()
                or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_size != row["size_bytes"]
                or sha256_file(path) != row["sha256"]):
            raise IntegrityError("failure quarantine artifact binding changed")
        expected_names.append(name); previous = name
    complete = receipt["complete_uncommitted_attempt"]
    if complete is not None:
        validate_complete_attempt_evidence_shape(complete)
        logical_state, logical_receipt = checkpoint_paths(
            failure_dir.parent / "arms" / receipt["arm_id"],
            receipt["segment_index"], receipt["attempt_index"],
        )
        state_name = (
            f"{receipt['arm_id']}_attempt_{receipt['attempt_index']:02d}_{logical_state.name}"
        )
        receipt_name = (
            f"{receipt['arm_id']}_attempt_{receipt['attempt_index']:02d}_{logical_receipt.name}"
        )
        inventory_by_name = {row["filename"]: row for row in inventory}
        if state_name not in inventory_by_name or receipt_name not in inventory_by_name:
            raise IntegrityError("complete attempt evidence lacks its quarantine pair")
        state_path = failure_dir / state_name
        attempt_receipt_path = failure_dir / receipt_name
        attempt_receipt = strict_json(attempt_receipt_path)
        if (sha256_file(state_path) != complete["raw_checkpoint_sha256"]
                or state_path.stat().st_size != complete["raw_checkpoint_size_bytes"]
                or sha256_file(attempt_receipt_path) != complete["attempt_receipt_sha256"]
                or attempt_receipt.get("checkpoint_sha256")
                != complete["raw_checkpoint_sha256"]
                or attempt_receipt.get("checkpoint_size_bytes")
                != complete["raw_checkpoint_size_bytes"]
                or attempt_receipt.get("decoded_integrator_state_sha256")
                != complete["decoded_integrator_state_sha256"]
                or attempt_receipt.get("sampled_state_stream_sha256")
                != complete["sampled_state_stream_sha256"]
                or attempt_receipt.get("segment_chain_head")
                != complete["segment_chain_head"]
                or sha256_bytes(canonical_bytes(
                    semantic_segment_payload(attempt_receipt)
                )) != complete["semantic_segment_payload_sha256"]):
            raise IntegrityError("complete quarantined attempt evidence changed")


def append_failure_terminal(
    ledger_path: Path, ledger_rows: list[dict[str, Any]], output_root: Path,
    start_row: dict[str, Any], *, failure_class: str, return_code: int | None,
    quarantined_artifacts: Sequence[dict[str, Any]],
    complete_uncommitted_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core = failure_event_core(
        execution_label=start_row["execution_label"], arm_id=start_row["arm_id"],
        segment_index=start_row["segment_index"],
        attempt_index=start_row["attempt_index"], start_sequence=start_row["sequence"],
        return_code=return_code,
        failure_class=failure_class,
        complete_uncommitted_attempt=complete_uncommitted_attempt,
    )
    receipt = build_failure_receipt(
        start_row, core, quarantined_artifacts, complete_uncommitted_attempt
    )
    failure_dir = output_root / "failures"
    filename = failure_receipt_filename(
        start_row["arm_id"], start_row["segment_index"], start_row["attempt_index"]
    )
    receipt_path = failure_dir / filename
    pending_path = receipt_path.with_name(f".{receipt_path.name}.pending")
    if pending_path.exists() or pending_path.is_symlink():
        if receipt_path.exists() or receipt_path.is_symlink() \
                or strict_json(pending_path) != receipt \
                or pending_path.read_bytes() != serialized_json(receipt):
            raise IntegrityError("pending deterministic failure receipt conflicts")
        os.replace(pending_path, receipt_path)
        descriptor = os.open(failure_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if receipt_path.exists() or receipt_path.is_symlink():
        if (strict_json(receipt_path) != receipt
                or receipt_path.read_bytes() != serialized_json(receipt)):
            raise IntegrityError("existing deterministic failure receipt conflicts")
    else:
        atomic_json(receipt_path, receipt)
    row = failure_terminal_row(
        receipt, filename, sha256_file(receipt_path), len(ledger_rows) + 1
    )
    append_ledger(ledger_path, row)
    ledger_rows.append(row)
    return row


def reconcile_orphan_failure_receipt(
    ledger_path: Path, ledger_rows: list[dict[str, Any]], output_root: Path,
    start_row: dict[str, Any],
) -> bool:
    failure_dir = output_root / "failures"
    filename = failure_receipt_filename(
        start_row["arm_id"], start_row["segment_index"], start_row["attempt_index"]
    )
    path = failure_dir / filename
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or pending.is_symlink():
        if path.exists() or path.is_symlink():
            raise IntegrityError("final and pending failure receipts coexist")
        # Promote only a complete strict JSON object; its exact attempt,
        # quarantine, class, and event bindings are checked immediately below.
        if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
            raise IntegrityError("pending failure receipt is unsafe")
        try:
            pending_receipt = strict_json(pending)
        except (IntegrityError, ValueError, OSError):
            torn_name = torn_failure_receipt_quarantine_name(
                start_row["arm_id"], start_row["segment_index"],
                start_row["attempt_index"],
            )
            torn_path = failure_dir / torn_name
            if torn_path.exists() or torn_path.is_symlink():
                raise IntegrityError("duplicate torn failure-receipt evidence")
            os.replace(pending, torn_path)
            descriptor = os.open(failure_dir, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return False
        if pending.read_bytes() != serialized_json(pending_receipt):
            raise IntegrityError("complete pending failure receipt is noncanonical")
        validate_failure_receipt_payload(
            pending_receipt, start_row, filename, failure_dir
        )
        os.replace(pending, path)
        descriptor = os.open(failure_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if not path.exists() and not path.is_symlink():
        return False
    receipt = strict_json(path)
    validate_failure_receipt_payload(receipt, start_row, filename, failure_dir)
    row = failure_terminal_row(
        receipt, filename, sha256_file(path), len(ledger_rows) + 1
    )
    append_ledger(ledger_path, row); ledger_rows.append(row)
    return True


def validate_failure_receipt_bindings(
    rows: Sequence[dict[str, Any]], output_root: Path,
) -> None:
    failure_dir = output_root / "failures"
    if (failure_dir.is_symlink() or not failure_dir.is_dir()
            or failure_dir.resolve().parent != output_root.resolve()):
        raise IntegrityError("primary failure directory is unsafe")
    starts = {
        (row["arm_id"], row["segment_index"], row["attempt_index"]): row
        for row in rows if row["event"] == "START"
    }
    failed = [row for row in rows if row["event"] == "FAIL"]
    receipt_names = {
        row["failure_receipt_filename"] for row in failed
    }
    actual_receipts: set[str] = set()
    actual_quarantine: set[str] = set()
    for entry in failure_dir.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1:
            raise IntegrityError("failure directory contains an unsafe entry")
        if entry.name.startswith("failure_") and entry.name.endswith(".json"):
            actual_receipts.add(entry.name)
        else:
            actual_quarantine.add(entry.name)
    if actual_receipts != receipt_names:
        raise IntegrityError("FAIL rows and permanent failure receipts are not bijective")
    bound_quarantine: set[str] = set()
    for row in failed:
        key = (row["arm_id"], row["segment_index"], row["attempt_index"])
        start = starts.get(key)
        if start is None:
            raise IntegrityError("FAIL receipt lacks its START")
        filename = row["failure_receipt_filename"]
        receipt = strict_json(failure_dir / filename)
        validate_failure_receipt_payload(receipt, start, filename, failure_dir)
        expected = failure_terminal_row(
            receipt, filename, sha256_file(failure_dir / filename), row["sequence"]
        )
        if row != expected:
            raise IntegrityError("FAIL ledger row does not exactly bind its receipt")
        names = {item["filename"] for item in receipt["quarantined_artifacts"]}
        if bound_quarantine & names:
            raise IntegrityError("quarantine artifact is bound by multiple receipts")
        bound_quarantine.update(names)
    if actual_quarantine != bound_quarantine:
        raise IntegrityError("quarantine inventory has an omission or extra artifact")


def proc_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def child_environment(contract: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    for key, value in contract["runtime_lock"]["native_thread_environment"].items():
        environment[key] = value
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def verify_child_environment(contract: dict[str, Any]) -> None:
    expected = contract["runtime_lock"]["native_thread_environment"]
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise IntegrityError("native thread environment lock mismatch")


def attempts_for(rows: Sequence[dict[str, Any]], label: str, arm_id: str, segment: int) -> int:
    return sum(
        row.get("event") == "START" and row.get("execution_label") == label
        and row.get("arm_id") == arm_id and row.get("segment_index") == segment
        for row in rows
    )


def validate_attempt_ledger(
    rows: Sequence[dict[str, Any]], execution_label: str, registration_sha256: str
) -> set[tuple[str, int, int]]:
    open_keys: set[tuple[str, int, int]] = set()
    start_counts: dict[tuple[str, int], int] = {}
    completed_segments = {arm_id: 0 for arm_id in ALL_ARM_IDS}
    predecessor_chains = {arm_id: INITIAL_SEGMENT_CHAIN for arm_id in ALL_ARM_IDS}
    passed_pairs: set[tuple[str, int]] = set()
    start_rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("execution_label") != execution_label:
            raise IntegrityError("attempt ledger execution label changed")
        arm_id = row.get("arm_id"); segment = row.get("segment_index")
        attempt = row.get("attempt_index")
        if arm_id not in ALL_ARM_IDS or type(segment) is not int or not 0 <= segment < 20:
            raise IntegrityError("attempt ledger segment identity changed")
        if type(attempt) is not int or not 1 <= attempt <= 3:
            raise IntegrityError("attempt ledger retry number changed")
        key = (arm_id, segment, attempt)
        pair = (arm_id, segment)
        if row["event"] == "START":
            require_keys(row, {
                "schema", "sequence", "event", "execution_label", "arm_id",
                "segment_index", "attempt_index", "predecessor_segment_chain_head",
                "input_key_sha256",
            }, "attempt ledger START")
            expected_attempt = start_counts.get(pair, 0) + 1
            if (attempt != expected_attempt or key in open_keys or pair in passed_pairs
                    or segment != completed_segments[arm_id]
                    or any(open_key[0] == arm_id for open_key in open_keys)
                    or row["predecessor_segment_chain_head"] != predecessor_chains[arm_id]):
                raise IntegrityError("attempt ledger START order changed")
            expected_input_key = sha256_bytes(canonical_bytes({
                "registration_sha256": registration_sha256,
                "execution_label": execution_label, "arm_id": arm_id,
                "segment_index": segment,
                "predecessor_segment_chain_head": predecessor_chains[arm_id],
            }))
            if row["input_key_sha256"] != expected_input_key:
                raise IntegrityError("attempt ledger START input binding changed")
            start_counts[pair] = attempt; open_keys.add(key); start_rows[key] = row
        else:
            normal_fields = {
                "schema", "sequence", "event", "execution_label", "arm_id",
                "segment_index", "attempt_index", "return_code",
            }
            pass_fields = normal_fields | {"segment_chain_head"}
            recovered_pass_fields = pass_fields | {"recovered_committed_attempt_at_resume"}
            fail_fields = normal_fields | {
                "failure_class", "fail_event_sha256", "failure_receipt_filename",
                "failure_receipt_sha256",
                "complete_uncommitted_attempt_semantic_sha256",
                "complete_uncommitted_attempt_decoded_state_sha256",
            }
            if set(row) not in (fail_fields, pass_fields, recovered_pass_fields):
                raise IntegrityError("attempt ledger terminal shape changed")
            if set(row) == recovered_pass_fields and (
                row["event"] != "PASS" or row["return_code"] != 0
                or row["recovered_committed_attempt_at_resume"] is not True
            ):
                raise IntegrityError("recovered committed attempt row changed")
            if key not in open_keys:
                raise IntegrityError("attempt ledger terminal row lacks one open START")
            if row["event"] == "PASS" and (
                set(row) not in (pass_fields, recovered_pass_fields)
                or row.get("return_code") != 0
                or not valid_sha256(row.get("segment_chain_head"))
            ):
                raise IntegrityError("attempt PASS return code changed")
            if row["event"] == "FAIL" and (
                set(row) != fail_fields
                or row.get("return_code") == 0
                or (row.get("return_code") is not None
                    and (not isinstance(row["return_code"], int)
                         or isinstance(row["return_code"], bool)))
                or row.get("failure_class") not in FAILURE_CLASSES
                or not valid_sha256(row.get("fail_event_sha256"))
                or not valid_sha256(row.get("failure_receipt_sha256"))
                or row.get("failure_receipt_filename") != failure_receipt_filename(
                    arm_id, segment, attempt
                )
                or ((row.get("complete_uncommitted_attempt_semantic_sha256") is None)
                    != (row.get("complete_uncommitted_attempt_decoded_state_sha256") is None))
                or (row.get("complete_uncommitted_attempt_semantic_sha256") is not None
                    and (not valid_sha256(
                        row["complete_uncommitted_attempt_semantic_sha256"]
                    ) or not valid_sha256(
                        row["complete_uncommitted_attempt_decoded_state_sha256"]
                    )))
            ):
                raise IntegrityError("attempt FAIL return code changed")
            if row["event"] == "FAIL":
                core = failure_event_core(
                    execution_label=execution_label,
                    arm_id=arm_id, segment_index=segment, attempt_index=attempt,
                    start_sequence=start_rows[key]["sequence"],
                    return_code=row["return_code"], failure_class=row["failure_class"],
                    complete_semantic_sha256=row[
                        "complete_uncommitted_attempt_semantic_sha256"
                    ],
                    complete_decoded_state_sha256=row[
                        "complete_uncommitted_attempt_decoded_state_sha256"
                    ],
                )
                if row["fail_event_sha256"] != failure_event_sha256(core):
                    raise IntegrityError("attempt FAIL event digest changed")
            open_keys.remove(key)
            if row["event"] == "PASS":
                if pair in passed_pairs:
                    raise IntegrityError("attempt ledger has a second PASS for one segment")
                passed_pairs.add(pair)
                completed_segments[arm_id] += 1
                predecessor_chains[arm_id] = row["segment_chain_head"]
    return open_keys


def validate_ledger_commit_bindings(
    rows: Sequence[dict[str, Any]], output_root: Path,
) -> None:
    passes: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "PASS":
            continue
        pair = (row["arm_id"], row["segment_index"])
        if pair in passes:
            raise IntegrityError("attempt ledger contains duplicate PASS rows")
        passes[pair] = row
    committed: set[tuple[str, int]] = set()
    arms_root = output_root / "arms"
    if not arms_root.is_dir() or arms_root.is_symlink():
        raise IntegrityError("attempt ledger arms root is unsafe")
    for arm_id in ALL_ARM_IDS:
        arm_dir = arms_root / arm_id
        if not arm_dir.exists():
            continue
        if arm_dir.is_symlink() or not arm_dir.is_dir():
            raise IntegrityError("attempt ledger arm directory is unsafe")
        segment_dir = arm_dir / "segments"
        if not segment_dir.exists():
            continue
        if segment_dir.is_symlink() or not segment_dir.is_dir():
            raise IntegrityError("attempt ledger segment directory is unsafe")
        for commit_path in segment_dir.glob("segment_*_commit.json"):
            try:
                segment = int(commit_path.stem.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise IntegrityError("segment commit filename changed") from exc
            if not 0 <= segment < 20 or commit_path.name != f"segment_{segment:02d}_commit.json":
                raise IntegrityError("segment commit filename changed")
            receipt = load_completed_segment(arm_dir, segment)
            pair = (arm_id, segment); committed.add(pair)
            passed = passes.get(pair)
            if (passed is None
                    or passed["attempt_index"] != receipt["provenance"]["attempt_index"]
                    or passed["segment_chain_head"] != receipt["segment_chain_head"]):
                raise IntegrityError("ledger PASS does not bind the selected parent commit")
    if set(passes) != committed:
        raise IntegrityError("attempt ledger PASS/commit sets differ")


def recover_or_validate_run_manifest(
    output_root: Path, expected: dict[str, Any],
) -> None:
    path = output_root / "run_manifest.json"
    pending = output_root / ".run_manifest.json.pending"
    if path.exists() or path.is_symlink():
        if pending.exists() or pending.is_symlink() or strict_json(path) != expected:
            raise IntegrityError("resumed run manifest differs or has a conflicting pending copy")
        return
    ledger = output_root / "attempt_ledger.jsonl"
    pending_ledger = output_root / ".attempt_ledger.jsonl.pending"
    result = output_root / "result_v1.json"
    pending_result = output_root / ".result_v1.json.pending"
    safe_prestart = not any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (ledger, pending_ledger, result, pending_result)
    )
    for directory_name in ("arms", "failures"):
        directory = output_root / directory_name
        if directory.exists() or directory.is_symlink():
            safe_prestart = safe_prestart and not directory.is_symlink() \
                and directory.is_dir() and not any(directory.iterdir())
    if pending.exists() or pending.is_symlink():
        if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
            raise IntegrityError("pending run manifest is unsafe")
        try:
            candidate = strict_json(pending)
        except (IntegrityError, ValueError, OSError):
            candidate = None
        if candidate == expected:
            os.replace(pending, path)
            descriptor = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return
        if not safe_prestart:
            raise IntegrityError("invalid pending run manifest accompanies numerical state")
        pending.unlink()
    elif not safe_prestart:
        raise IntegrityError("numerical state exists without its run manifest")
    for directory_name in ("arms", "failures"):
        (output_root / directory_name).mkdir(exist_ok=True)
    atomic_json(path, expected)


def recover_empty_arm_skeletons(
    output_root: Path, ledger_rows: Sequence[dict[str, Any]],
) -> None:
    """Repair only the mkdir-before-START crash window."""
    arms_root = output_root / "arms"
    for arm_dir in arms_root.iterdir():
        segment_dir = arm_dir / "segments"
        if segment_dir.exists() or segment_dir.is_symlink():
            continue
        if (arm_dir.is_symlink() or not arm_dir.is_dir() or any(arm_dir.iterdir())
                or any(row.get("arm_id") == arm_dir.name for row in ledger_rows)):
            raise IntegrityError("missing segment directory is not an empty pre-START skeleton")
        segment_dir.mkdir()
        descriptor = os.open(arm_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def run_supervisor(
    contract: dict[str, Any], contract_path: Path, seed_path: Path,
    initial_path: Path, registration_path: Path,
    expanded: dict[str, list[list[Any]]], output_root: Path,
    execution_label: str, resume: bool, a_binding: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    global _EXECUTION_LOCK_FD
    caps = contract["resource_caps_per_execution"]
    ledger_path = output_root / "attempt_ledger.jsonl"
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "execution_label": execution_label,
        "registration_sha256": sha256_file(registration_path),
        "contract_sha256": sha256_file(contract_path),
        "seed_manifest_sha256": sha256_file(seed_path),
        "initial_states_sha256": sha256_file(initial_path),
        "a_prerequisite": a_binding,
        "arm_order": list(ALL_ARM_IDS), "workers": 4,
        "segment_count_per_arm": 20, "segment_years": 50_000.0,
    }
    if resume:
        execution_lock_fd = acquire_execution_lock(output_root, create=False)
        _EXECUTION_LOCK_FD = execution_lock_fd
        recover_or_validate_run_manifest(output_root, manifest)
        recover_pending_ledger(
            ledger_path, execution_label=execution_label,
            registration_sha256=manifest["registration_sha256"],
            output_root=output_root,
        )
        if not (output_root / "arms").is_dir():
            raise IntegrityError("resumed arms directory is missing")
        if (output_root / "result_v1.json").exists():
            raise IntegrityError("a published result cannot be resumed")
        allowed_root = {
            "run_manifest.json", "attempt_ledger.jsonl", "arms", "failures",
            "execution.lock", ".result_v1.json.pending",
        }
        if any(item.name not in allowed_root or item.is_symlink() for item in output_root.iterdir()):
            raise IntegrityError("resumed output root contains an unsafe extra entry")
        if any(item.name not in set(ALL_ARM_IDS) or item.is_symlink() or not item.is_dir()
               for item in (output_root / "arms").iterdir()):
            raise IntegrityError("resumed arms tree contains an unsafe entry")
    else:
        output_root.mkdir(parents=False, exist_ok=False)
        execution_lock_fd = acquire_execution_lock(output_root, create=True)
        _EXECUTION_LOCK_FD = execution_lock_fd
        (output_root / "arms").mkdir()
        (output_root / "failures").mkdir()
        atomic_json(output_root / "run_manifest.json", manifest)
    started_ns = time.monotonic_ns()
    running: dict[int, dict[str, Any]] = {}
    next_segment = {arm_id: 0 for arm_id in ALL_ARM_IDS}
    completed: dict[str, dict[str, Any]] = {}
    ledger_rows: list[dict[str, Any]] = read_jsonl(ledger_path)
    if resume:
        recover_empty_arm_skeletons(output_root, ledger_rows)
    open_attempts = validate_attempt_ledger(
        ledger_rows, execution_label, manifest["registration_sha256"]
    )
    for arm_id, segment_index, attempt_index in sorted(open_attempts):
        start_row = next(
            row for row in ledger_rows if row["event"] == "START"
            and row["arm_id"] == arm_id and row["segment_index"] == segment_index
            and row["attempt_index"] == attempt_index
        )
        reconcile_orphan_failure_receipt(
            ledger_path, ledger_rows, output_root, start_row
        )
    open_attempts = validate_attempt_ledger(
        ledger_rows, execution_label, manifest["registration_sha256"]
    )
    for arm_id, segment_index, attempt_index in sorted(open_attempts):
        start_row = next(
            row for row in ledger_rows if row["event"] == "START"
            and row["arm_id"] == arm_id and row["segment_index"] == segment_index
            and row["attempt_index"] == attempt_index
        )
        arm_dir = output_root / "arms" / arm_id
        commit_path = segment_commit_path(arm_dir, segment_index)
        recovered_pass = False
        if commit_path.exists():
            receipt = load_completed_segment(arm_dir, segment_index)
            recovered_pass = (
                receipt.get("provenance", {}).get("attempt_index") == attempt_index
                and validate_attempt_payload(
                    contract, expanded, arm_dir, arm_id, segment_index, attempt_index
                ) == receipt
            )
            if not recovered_pass:
                raise IntegrityError("open attempt conflicts with its existing parent commit")
        if recovered_pass:
            recovered = {
                "schema": ATTEMPT_SCHEMA, "sequence": len(ledger_rows) + 1,
                "event": "PASS", "execution_label": execution_label,
                "arm_id": arm_id, "segment_index": segment_index,
                "attempt_index": attempt_index, "return_code": 0,
                "recovered_committed_attempt_at_resume": True,
            }
            recovered["segment_chain_head"] = receipt["segment_chain_head"]
            append_ledger(ledger_path, recovered); ledger_rows.append(recovered)
        else:
            complete = validated_complete_uncommitted_attempt(
                contract, expanded, output_root, arm_id, segment_index, attempt_index
            )
            quarantine = quarantine_attempt_artifacts(
                output_root, arm_id, segment_index, attempt_index
            )
            append_failure_terminal(
                ledger_path, ledger_rows, output_root, start_row,
                failure_class="RECOVERED_UNCOMMITTED", return_code=None,
                quarantined_artifacts=quarantine,
                complete_uncommitted_attempt=complete,
            )
    open_attempts = validate_attempt_ledger(
        ledger_rows, execution_label, manifest["registration_sha256"]
    )
    if open_attempts:
        raise IntegrityError("recovered attempt ledger still contains an open START")
    validate_failure_receipt_bindings(ledger_rows, output_root)
    if resume:
        for arm_id in ALL_ARM_IDS:
            arm_dir = output_root / "arms" / arm_id
            if not arm_dir.exists():
                continue
            count, last = validate_segment_chain(contract, expanded, arm_dir, arm_id)
            next_segment[arm_id] = count
            if count == 20:
                if last is None:
                    raise IntegrityError("complete resumed arm lacks final receipt")
                completed[arm_id] = last
    validate_ledger_commit_bindings(ledger_rows, output_root)
    environment = child_environment(contract)

    def kill_and_reap_all() -> None:
        for job in running.values():
            try:
                os.killpg(job["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for job in running.values():
            job["process"].wait()

    def launch(arm_id: str, segment_index: int) -> None:
        nonlocal ledger_rows
        attempt = attempts_for(ledger_rows, execution_label, arm_id, segment_index) + 1
        if attempt > int(contract["checkpoint_and_resume"]["maximum_attempts_per_segment"]):
            raise ResourceLimitError("segment attempt limit exhausted")
        if segment_index == 0:
            predecessor = INITIAL_SEGMENT_CHAIN
        else:
            predecessor = load_completed_segment(
                output_root / "arms" / arm_id, segment_index - 1
            )["segment_chain_head"]
        row = {
            "schema": ATTEMPT_SCHEMA, "sequence": len(ledger_rows) + 1,
            "event": "START", "execution_label": execution_label,
            "arm_id": arm_id, "segment_index": segment_index, "attempt_index": attempt,
            "predecessor_segment_chain_head": predecessor,
            "input_key_sha256": sha256_bytes(canonical_bytes({
                "registration_sha256": manifest["registration_sha256"],
                "execution_label": execution_label, "arm_id": arm_id,
                "segment_index": segment_index,
                "predecessor_segment_chain_head": predecessor,
            })),
        }
        arm_dir = output_root / "arms" / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        segment_dir = arm_dir / "segments"
        segment_dir.mkdir(exist_ok=True)
        if (arm_dir.is_symlink() or not arm_dir.is_dir() or segment_dir.is_symlink()
                or not segment_dir.is_dir()):
            raise IntegrityError("coordinator arm/segment directory is unsafe")
        append_ledger(ledger_path, row); ledger_rows.append(row)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--internal-segment",
            "--contract", str(contract_path), "--seed-manifest", str(seed_path),
            "--initial-states", str(initial_path), "--registration", str(registration_path),
            "--output-dir", str(output_root), "--execution-label", execution_label,
            "--arm-id", arm_id, "--segment-index", str(segment_index),
            "--attempt-index", str(attempt), "--start-sequence", str(row["sequence"]),
            "--lock-fd", str(execution_lock_fd),
            "--lineage-lock-fd", str(_V2_B_GUARD_FD),
            "--v3-lineage-lock-fd", str(_V3_A_GUARD_FD),
            "--engineering-runner-lock-fd", str(_ENGINEERING_RUNNER_GUARD_FD),
            "--engineering-scratch-lock-fd", str(_ENGINEERING_SCRATCH_GUARD_FD),
        ]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=environment,
                                   start_new_session=True, close_fds=True,
                                   pass_fds=(
                                       execution_lock_fd, _V2_B_GUARD_FD, _V3_A_GUARD_FD,
                                       _ENGINEERING_RUNNER_GUARD_FD,
                                       _ENGINEERING_SCRATCH_GUARD_FD,
                                   ))
        running[process.pid] = {
            "process": process, "arm_id": arm_id, "segment_index": segment_index,
            "attempt_index": attempt, "started_ns": time.monotonic_ns(),
        }

    try:
        while len(completed) < len(ALL_ARM_IDS):
            now_ns = time.monotonic_ns()
            if peak_rss_bytes() > int(caps["max_peak_rss_bytes_per_process"]):
                kill_and_reap_all()
                raise ResourceLimitError("coordinator RSS cap reached")
            if (now_ns - started_ns) / NANOSECONDS_PER_SECOND >= float(caps["max_wall_seconds_total"]):
                kill_and_reap_all()
                raise ResourceLimitError("execution wall cap reached")
            if directory_bytes(output_root) > int(caps["max_output_bytes"]):
                kill_and_reap_all()
                raise ResourceLimitError("execution output cap reached")
            if shutil.disk_usage(output_root).free < int(caps["minimum_free_disk_bytes"]):
                kill_and_reap_all()
                raise ResourceLimitError("free-disk floor reached")
            live_rss = {pid: proc_rss_bytes(pid) for pid in running}
            for pid, rss in live_rss.items():
                if rss > int(caps["max_peak_rss_bytes_per_process"]):
                    running[pid]["failure_override"] = "CHILD_RSS_LIMIT"
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            aggregate_rss = sum(live_rss.values())
            if aggregate_rss > int(caps["max_aggregate_child_rss_bytes"]):
                kill_and_reap_all()
                raise ResourceLimitError("aggregate child RSS cap reached")
    
            for pid, job in list(running.items()):
                process = job["process"]
                elapsed = (now_ns - job["started_ns"]) / NANOSECONDS_PER_SECOND
                if elapsed >= float(caps["max_wall_seconds_per_segment_attempt"]):
                    os.killpg(pid, signal.SIGKILL); process.wait()
                    return_code = -signal.SIGKILL
                    failure_class = "SEGMENT_TIMEOUT"
                else:
                    return_code = process.poll()
                    failure_class = job.get("failure_override")
                if return_code is None:
                    continue
                del running[pid]
                if return_code == 0:
                    arm_dir = output_root / "arms" / job["arm_id"]
                    commit_segment_attempt(
                        contract, expanded, arm_dir, job["arm_id"], job["segment_index"],
                        job["attempt_index"], elapsed, output_root,
                        (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND,
                    )
                    count, last = validate_segment_chain(
                        contract, expanded, arm_dir, job["arm_id"]
                    )
                    if count != job["segment_index"] + 1 or last is None:
                        raise IntegrityError("successful child did not publish exactly one segment")
                    next_segment[job["arm_id"]] = count
                    if count == 20:
                        completed[job["arm_id"]] = last
                if return_code == 0:
                    terminal = {
                        "schema": ATTEMPT_SCHEMA, "sequence": len(ledger_rows) + 1,
                        "event": "PASS", "execution_label": execution_label,
                        "arm_id": job["arm_id"], "segment_index": job["segment_index"],
                        "attempt_index": job["attempt_index"], "return_code": 0,
                    }
                    terminal["segment_chain_head"] = last["segment_chain_head"]
                    append_ledger(ledger_path, terminal); ledger_rows.append(terminal)
                else:
                    complete = validated_complete_uncommitted_attempt(
                        contract, expanded, output_root, job["arm_id"],
                        job["segment_index"], job["attempt_index"],
                    )
                    quarantine = quarantine_attempt_artifacts(
                        output_root, job["arm_id"], job["segment_index"], job["attempt_index"]
                    )
                    if failure_class is None:
                        failure_class = (
                            "CHILD_SIGNAL" if return_code < 0 else "CHILD_EXIT_NONZERO"
                        )
                    start_row = next(
                        row for row in ledger_rows if row["event"] == "START"
                        and row["arm_id"] == job["arm_id"]
                        and row["segment_index"] == job["segment_index"]
                        and row["attempt_index"] == job["attempt_index"]
                    )
                    append_failure_terminal(
                        ledger_path, ledger_rows, output_root, start_row,
                        failure_class=failure_class, return_code=int(return_code),
                        quarantined_artifacts=quarantine,
                        complete_uncommitted_attempt=complete,
                    )
                validate_attempt_ledger(
                    ledger_rows, execution_label, manifest["registration_sha256"]
                )
                validate_failure_receipt_bindings(ledger_rows, output_root)
                validate_ledger_commit_bindings(ledger_rows, output_root)
                if return_code != 0 and attempts_for(
                    ledger_rows, execution_label, job["arm_id"], job["segment_index"]
                ) >= 3:
                    raise ResourceLimitError("segment failed three identical attempts")
    
            busy_arms = {job["arm_id"] for job in running.values()}
            for arm_id in ALL_ARM_IDS:
                if len(running) >= int(caps["workers"]):
                    break
                if arm_id in completed or arm_id in busy_arms:
                    continue
                launch(arm_id, next_segment[arm_id])
                busy_arms.add(arm_id)
            if running:
                time.sleep(float(caps["watchdog_poll_seconds"]))
        validate_ledger_commit_bindings(ledger_rows, output_root)
        validate_failure_receipt_bindings(ledger_rows, output_root)
        return completed
    finally:
        if running:
            kill_and_reap_all()


def initial_digest_map(artifact: dict[str, Any]) -> dict[str, str]:
    return {row[0]: row[-1] for row in artifact["configuration_states"]}


def raw_artifact_integrity_inventory(output_root: Path) -> dict[str, Any]:
    """Bind raw archives and their pointer receipts outside scientific semantics."""
    entries: list[dict[str, Any]] = []
    for arm_id in ALL_ARM_IDS:
        segment_dir = output_root / "arms" / arm_id / "segments"
        for segment_index in range(20):
            commit_path = segment_dir / f"segment_{segment_index:02d}_commit.json"
            commit = strict_json(commit_path)
            receipt_path = segment_dir / commit["attempt_receipt_filename"]
            receipt = strict_json(receipt_path)
            state_path = segment_dir / commit["checkpoint_filename"]
            if (state_path.name != receipt.get("checkpoint_filename")
                    or state_path.is_symlink() or not state_path.is_file()
                    or state_path.stat().st_nlink != 1):
                raise IntegrityError("raw artifact integrity inventory path changed")
            entries.append({
                "arm_id": arm_id,
                "segment_index": segment_index,
                "commit_filename": commit_path.name,
                "commit_size_bytes": commit_path.stat().st_size,
                "commit_sha256": sha256_file(commit_path),
                "receipt_filename": receipt_path.name,
                "receipt_size_bytes": receipt_path.stat().st_size,
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_filename": state_path.name,
                "checkpoint_size_bytes": state_path.stat().st_size,
                "checkpoint_sha256": sha256_file(state_path),
            })
    if len(entries) != len(ALL_ARM_IDS) * 20:
        raise IntegrityError("raw artifact integrity inventory cardinality changed")
    return {
        "schema": "jx-xp2-mercurius-raw-artifact-integrity/v1",
        "entry_count": len(entries),
        "entries": entries,
        "root_sha256": sha256_bytes(
            RAW_ARTIFACT_INTEGRITY_DOMAIN + canonical_bytes(entries)
        ),
        "scientific_semantic_input": False,
    }


def build_primary_result(
    contract: dict[str, Any], contract_path: Path, seed_path: Path,
    initial_path: Path, registration_path: Path, runtime: dict[str, Any],
    output_root: Path, execution_label: str,
    completed: dict[str, dict[str, Any]], artifact: dict[str, Any],
    started_ns: int,
) -> dict[str, Any]:
    analysis = analyze_primary(completed, contract)
    digests = initial_digest_map(artifact)
    arms = []
    for arm_id in ALL_ARM_IDS:
        receipt = completed[arm_id]
        if receipt["sample_count_total"] != 20_001 or set(receipt["landmarks"]) != {
            "250000", "500000", "1000000"
        }:
            raise IntegrityError("complete arm sample/landmark cardinality changed")
        arms.append({
            "arm_id": arm_id, "configuration_id": receipt["configuration_id"],
            "arm_class": receipt["arm_class"], "dt_years": receipt["dt_years"],
            "registered_expanded_initial_state_sha256": digests[receipt["configuration_id"]],
            "sample_count": receipt["sample_count_total"],
            "segment_chain_head_sha256": receipt["segment_chain_head"],
            "maximum_active_invariant_drifts": receipt["maximum_active_invariant_drifts"],
            "landmarks": receipt["landmarks"],
        })
    semantic = {
        "schema": SEMANTIC_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "arm_order": list(ALL_ARM_IDS),
        "arms": arms, "analysis": analysis,
        "claim_ceiling": contract["claim_ceiling"],
        "official_classification": None,
        "classification_requires_verified_A_B_and_DOP853": True,
    }
    semantic_digest = sha256_bytes(canonical_bytes(semantic))
    return {
        "schema": RESULT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "artifact_class": "COMPLETE_PRIMARY_NUMERICAL_OUTPUT_AWAITING_INDEPENDENT_VERIFICATION",
        "execution_label": execution_label,
        "input_bindings": {
            "registration_sha256": sha256_file(registration_path),
            "contract_sha256": sha256_file(contract_path),
            "seed_manifest_sha256": sha256_file(seed_path),
            "initial_states_sha256": sha256_file(initial_path),
        },
        "runtime": runtime, "semantic": semantic, "semantic_sha256": semantic_digest,
        "raw_artifact_integrity": raw_artifact_integrity_inventory(output_root),
        "resource_provenance": {
            "wall_seconds": (time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND,
            "coordinator_peak_rss_bytes": peak_rss_bytes(),
            "output_bytes_before_result": directory_bytes(output_root),
            "attempt_ledger_sha256": sha256_file(output_root / "attempt_ledger.jsonl"),
        },
        "scientific_classification_emitted": False,
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }


def publish_or_recover_primary_result(
    output_root: Path, expected: dict[str, Any], contract: dict[str, Any],
) -> None:
    result_path = output_root / "result_v1.json"
    pending = output_root / ".result_v1.json.pending"
    if not pending.exists() and not pending.is_symlink():
        atomic_json(result_path, expected)
        return
    if pending.is_symlink() or not pending.is_file() or pending.stat().st_nlink != 1:
        raise IntegrityError("pending primary result is unsafe")
    try:
        candidate = strict_json(pending)
    except (IntegrityError, ValueError, OSError):
        pending.unlink()
        descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        expected["resource_provenance"]["output_bytes_before_result"] = directory_bytes(
            output_root
        )
        atomic_json(result_path, expected)
        return
    candidate_semantic = dict(candidate)
    expected_semantic = dict(expected)
    provenance = candidate_semantic.pop("resource_provenance", None)
    expected_semantic.pop("resource_provenance", None)
    if candidate_semantic != expected_semantic or not isinstance(provenance, dict) or set(
        provenance
    ) != {
        "wall_seconds", "coordinator_peak_rss_bytes", "output_bytes_before_result",
        "attempt_ledger_sha256",
    }:
        raise IntegrityError("complete pending primary result differs from recomputed semantics")
    before_result = sum(
        path.stat().st_size for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != pending
    )
    caps = contract["resource_caps_per_execution"]
    if (
        not isinstance(provenance["wall_seconds"], (int, float))
        or isinstance(provenance["wall_seconds"], bool)
        or not math.isfinite(provenance["wall_seconds"])
        or not 0.0 <= provenance["wall_seconds"] < caps["max_wall_seconds_total"]
        or type(provenance["coordinator_peak_rss_bytes"]) is not int
        or not 0 <= provenance["coordinator_peak_rss_bytes"]
        <= caps["max_peak_rss_bytes_per_process"]
        or type(provenance["output_bytes_before_result"]) is not int
        or provenance["output_bytes_before_result"] != before_result
        or provenance["attempt_ledger_sha256"]
        != sha256_file(output_root / "attempt_ledger.jsonl")
    ):
        raise IntegrityError("pending primary result resource provenance changed")
    os.replace(pending, result_path)
    descriptor = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_a_prerequisite(
    result_path: Path, receipt_path: Path, registration_digest: str, output_b: Path
) -> None:
    result = strict_json(result_path)
    receipt = strict_json(receipt_path)
    if (
        result.get("schema") != RESULT_SCHEMA or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("execution_label") != "A"
        or result.get("input_bindings", {}).get("registration_sha256") != registration_digest
        or result.get("semantic_sha256") != sha256_bytes(canonical_bytes(result.get("semantic")))
        or result.get("scientific_classification_emitted") is not False
    ):
        raise IntegrityError("A result is not a valid complete primary execution")
    if (
        receipt.get("schema") != "jx-xp2-primary-a-verification/v3"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("execution_label") != "A"
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("semantic_sha256") != result["semantic_sha256"]
        or receipt.get("verified_for_b") is not True
    ):
        raise IntegrityError("A verification receipt does not authorize B")
    if trees_overlap(output_b, result_path.resolve().parent):
        raise IntegrityError("A and B output trees overlap")


def common_preflight(
    contract_path: Path, seed_path: Path, initial_path: Path, registration_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], dict[str, Any],
           dict[str, list[list[Any]]], dict[str, Any]]:
    contract = strict_json(contract_path)
    validate_contract(contract, contract_path)
    _registration, _locked = validate_registration(
        registration_path, contract_path, seed_path, initial_path, Path(__file__).resolve()
    )
    _manifest, seeds = validate_seed_manifest(contract, seed_path)
    tracers, _tracer_digest = make_tracers(contract, seeds)
    artifact, expanded = expand_initial_states(contract, initial_path)
    validate_initial_pairing(tracers, artifact, expanded)
    runtime = validate_runtime(contract)
    return contract, seeds, tracers, artifact, expanded, runtime


def validate_worker_start(
    ledger: Sequence[dict[str, Any]], *, execution_label: str,
    registration_sha256: str, arm_id: str, segment_index: int,
    attempt_index: int, predecessor_segment_chain_head: str,
) -> dict[str, Any]:
    """Validate only this worker's immutable START, never sibling live state.

    The coordinator is the sole owner of global ledger/commit consistency.  A
    worker may observe a sibling commit during the short interval before the
    coordinator appends that sibling's PASS; that legal interleaving must not
    invalidate this worker.
    """
    key_rows = [
        row for row in ledger if row.get("arm_id") == arm_id
        and row.get("segment_index") == segment_index
        and row.get("attempt_index") == attempt_index
    ]
    expected_input_key = sha256_bytes(canonical_bytes({
        "registration_sha256": registration_sha256,
        "execution_label": execution_label, "arm_id": arm_id,
        "segment_index": segment_index,
        "predecessor_segment_chain_head": predecessor_segment_chain_head,
    }))
    if len(key_rows) != 1:
        raise IntegrityError("internal worker attempt key is not unique and open")
    start = key_rows[0]
    require_keys(start, {
        "schema", "sequence", "event", "execution_label", "arm_id",
        "segment_index", "attempt_index", "predecessor_segment_chain_head",
        "input_key_sha256",
    }, "internal worker START")
    if (
        start["schema"] != ATTEMPT_SCHEMA or start["event"] != "START"
        or start["sequence"] != len(ledger)
        or start["execution_label"] != execution_label
        or start["arm_id"] != arm_id or start["segment_index"] != segment_index
        or start["attempt_index"] != attempt_index
        or start["predecessor_segment_chain_head"] != predecessor_segment_chain_head
        or start["input_key_sha256"] != expected_input_key
    ):
        raise IntegrityError("internal worker lacks one exact coordinator START")
    return start


def internal_segment(args: argparse.Namespace) -> int:
    global _FAILURE_CONTEXT, _V2_B_GUARD_FD, _V3_A_GUARD_FD
    global _ENGINEERING_RUNNER_GUARD_FD, _ENGINEERING_SCRATCH_GUARD_FD
    raw_contract = strict_json(args.contract)
    verify_child_environment(raw_contract)
    validate_inherited_v2_b_guard(
        args.lineage_lock_fd, raw_contract, args.contract.parent
    )
    validate_inherited_v3_a_guard(
        args.v3_lineage_lock_fd, raw_contract, args.contract.parent
    )
    validate_inherited_engineering_evidence_guard(
        args.engineering_runner_lock_fd, raw_contract, args.contract.parent,
        "engineering_output_root", "engineering runner evidence",
    )
    validate_inherited_engineering_evidence_guard(
        args.engineering_scratch_lock_fd, raw_contract, args.contract.parent,
        "engineering_verifier_scratch_root", "engineering scratch evidence",
    )
    _V2_B_GUARD_FD = args.lineage_lock_fd
    _V3_A_GUARD_FD = args.v3_lineage_lock_fd
    _ENGINEERING_RUNNER_GUARD_FD = args.engineering_runner_lock_fd
    _ENGINEERING_SCRATCH_GUARD_FD = args.engineering_scratch_lock_fd
    contract, _seeds, tracers, _artifact, expanded, _runtime = common_preflight(
        args.contract, args.seed_manifest, args.initial_states, args.registration
    )
    output_root = args.output_dir.resolve()
    if not output_root.is_dir() or any(
        trees_overlap(output_root, root) for root in protected_roots(contract, args.contract.parent)
    ):
        raise IntegrityError("internal output root is missing or protected")
    lock_path = output_root / "execution.lock"
    try:
        inherited_lock = os.fstat(args.lock_fd)
        on_disk_lock = os.stat(lock_path, follow_symlinks=False)
    except (OSError, FileNotFoundError) as exc:
        raise IntegrityError("internal worker did not inherit the execution lock") from exc
    if (not stat.S_ISREG(inherited_lock.st_mode) or inherited_lock.st_nlink != 1
            or inherited_lock.st_size != 0 or lock_path.is_symlink()
            or inherited_lock.st_dev != on_disk_lock.st_dev
            or inherited_lock.st_ino != on_disk_lock.st_ino):
        raise IntegrityError("internal worker execution lock binding changed")
    try:
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise IntegrityError("internal worker lock fd is not inherited") from exc
    after_lock = os.stat(lock_path, follow_symlinks=False)
    if (not stat.S_ISREG(after_lock.st_mode) or after_lock.st_nlink != 1
            or after_lock.st_size != 0
            or after_lock.st_dev != inherited_lock.st_dev
            or after_lock.st_ino != inherited_lock.st_ino):
        raise IntegrityError("internal worker execution lock path changed")
    if args.execution_label not in ("A", "B"):
        raise IntegrityError("internal execution label changed")
    manifest = strict_json(output_root / "run_manifest.json")
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("execution_label") != args.execution_label
        or manifest.get("registration_sha256") != sha256_file(args.registration)
        or manifest.get("contract_sha256") != sha256_file(args.contract)
        or manifest.get("seed_manifest_sha256") != sha256_file(args.seed_manifest)
        or manifest.get("initial_states_sha256") != sha256_file(args.initial_states)
        or manifest.get("arm_order") != list(ALL_ARM_IDS)
        or manifest.get("workers") != 4
        or (args.execution_label == "B" and not isinstance(manifest.get("a_prerequisite"), dict))
        or (args.execution_label == "A" and manifest.get("a_prerequisite") is not None)
    ):
        raise IntegrityError("internal worker run-manifest binding changed")
    if (output_root / "result_v1.json").exists() or segment_commit_path(
        output_root / "arms" / args.arm_id, args.segment_index
    ).exists():
        raise IntegrityError("internal worker target is already committed or finalized")
    arm_dir = output_root / "arms" / args.arm_id
    if not arm_dir.is_dir() or arm_dir.is_symlink():
        raise IntegrityError("internal worker arm directory is not coordinator-owned")
    ledger = read_jsonl_prefix(
        output_root / "attempt_ledger.jsonl", args.start_sequence
    )
    predecessor = (INITIAL_SEGMENT_CHAIN if args.segment_index == 0 else load_completed_segment(
        arm_dir, args.segment_index - 1
    )["segment_chain_head"])
    validate_worker_start(
        ledger, execution_label=args.execution_label,
        registration_sha256=manifest["registration_sha256"], arm_id=args.arm_id,
        segment_index=args.segment_index, attempt_index=args.attempt_index,
        predecessor_segment_chain_head=predecessor,
    )
    _FAILURE_CONTEXT = {
        "output_root": str(output_root), "execution_label": args.execution_label,
        "arm_id": args.arm_id, "segment_index": args.segment_index,
        "attempt_index": args.attempt_index,
    }
    spec = arm_specification(contract, args.arm_id)
    run_one_segment(
        contract, tracers, expanded, spec, arm_dir, args.segment_index,
        args.attempt_index, output_root,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--seed-manifest", type=Path, required=True)
    result.add_argument("--initial-states", type=Path, required=True)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--execution-label", choices=("A", "B"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--a-result", type=Path)
    result.add_argument("--a-verification-receipt", type=Path)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--internal-segment", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--arm-id", choices=ALL_ARM_IDS, help=argparse.SUPPRESS)
    result.add_argument("--segment-index", type=int, help=argparse.SUPPRESS)
    result.add_argument("--attempt-index", type=int, help=argparse.SUPPRESS)
    result.add_argument("--start-sequence", type=int, help=argparse.SUPPRESS)
    result.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--lineage-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--v3-lineage-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--engineering-runner-lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--engineering-scratch-lock-fd", type=int, help=argparse.SUPPRESS)
    return result


def guarded_main(args: argparse.Namespace) -> int:
    global _FAILURE_CONTEXT
    contract, _seeds, _tracers, artifact, expanded, runtime = common_preflight(
        args.contract, args.seed_manifest, args.initial_states, args.registration
    )
    # Both predecessor execution locks are held while their exact evidence trees
    # are scanned, closing the preflight-to-lock race.
    validate_v2_replay_lineage(contract, args.contract.parent)
    validate_v3_failed_startup_lineage(contract, args.contract.parent)
    if args.validate_only:
        if any(value is not None for value in (
            args.output_dir, args.execution_label, args.a_result, args.a_verification_receipt
        )) or args.resume:
            raise IntegrityError("validate-only accepts no execution arguments")
        print(json.dumps({
            "schema": "jx-xp2-primary-validation/v1", "experiment_id": EXPERIMENT_ID,
            "status": "VALID_NO_DYNAMICS_RUN", "runtime": runtime,
            "registration_sha256": sha256_file(args.registration),
            "initial_states_sha256": sha256_file(args.initial_states),
        }, sort_keys=True))
        return 0
    if args.output_dir is None or args.execution_label is None:
        raise IntegrityError("execution requires output directory and label")
    output_root = validate_output_root(
        args.output_dir, contract, args.contract.parent, args.resume
    )
    _FAILURE_CONTEXT = {
        "output_root": str(output_root), "execution_label": args.execution_label,
        "arm_id": None, "segment_index": None, "attempt_index": None,
    }
    a_binding: dict[str, str] | None = None
    if args.execution_label == "B":
        if args.a_result is None or args.a_verification_receipt is None:
            raise IntegrityError("B requires a verified complete A")
        validate_a_prerequisite(
            args.a_result.resolve(), args.a_verification_receipt.resolve(),
            sha256_file(args.registration), output_root,
        )
        a_binding = {
            "a_result_sha256": sha256_file(args.a_result.resolve()),
            "a_verification_receipt_sha256": sha256_file(
                args.a_verification_receipt.resolve()
            ),
        }
    elif args.a_result is not None or args.a_verification_receipt is not None:
        raise IntegrityError("A must not consume A-prerequisite arguments")
    started_ns = time.monotonic_ns()
    completed = run_supervisor(
        contract, args.contract, args.seed_manifest, args.initial_states,
        args.registration, expanded, output_root, args.execution_label, args.resume, a_binding,
    )
    result = build_primary_result(
        contract, args.contract, args.seed_manifest, args.initial_states,
        args.registration, runtime, output_root, args.execution_label,
        completed, artifact, started_ns,
    )
    caps = contract["resource_caps_per_execution"]
    result_size = len(serialized_json(result))
    pending_result = output_root / ".result_v1.json.pending"
    pending_size = (
        pending_result.stat().st_size
        if pending_result.exists() and not pending_result.is_symlink() else 0
    )
    bytes_without_pending = directory_bytes(output_root) - pending_size
    final_result_size = max(result_size, pending_size)
    if ((time.monotonic_ns() - started_ns) / NANOSECONDS_PER_SECOND
            >= float(caps["max_wall_seconds_total"])
            or peak_rss_bytes() > int(caps["max_peak_rss_bytes_per_process"])
            or bytes_without_pending + final_result_size > int(caps["max_output_bytes"])
            or shutil.disk_usage(output_root).free
            < int(caps["minimum_free_disk_bytes"]) + final_result_size):
        raise ResourceLimitError("final result would exceed an output resource cap")
    revalidate_final_engineering_evidence()
    publish_or_recover_primary_result(output_root, result, contract)
    revalidate_final_engineering_evidence()
    return 0


def main() -> int:
    global _FAILURE_CONTEXT, _EXECUTION_LOCK_FD, _V2_B_GUARD_FD, _V3_A_GUARD_FD
    global _ENGINEERING_RUNNER_GUARD_FD, _ENGINEERING_SCRATCH_GUARD_FD
    args = parser().parse_args()
    args.contract = args.contract.resolve(); args.seed_manifest = args.seed_manifest.resolve()
    args.initial_states = args.initial_states.resolve(); args.registration = args.registration.resolve()
    if args.internal_segment:
        if (args.output_dir is None or args.execution_label is None or args.arm_id is None
                or args.segment_index is None or args.attempt_index is None
                or args.start_sequence is None or args.lock_fd is None
                or args.lineage_lock_fd is None or args.v3_lineage_lock_fd is None
                or args.engineering_runner_lock_fd is None
                or args.engineering_scratch_lock_fd is None):
            raise IntegrityError("internal segment arguments incomplete")
        args.output_dir = args.output_dir.resolve()
        return internal_segment(args)
    raw_contract = strict_json(args.contract)
    if (raw_contract.get("schema") != CONTRACT_SCHEMA
            or raw_contract.get("experiment_id") != EXPERIMENT_ID):
        raise IntegrityError("contract identity changed before lineage lock acquisition")
    _V2_B_GUARD_FD = acquire_v2_b_guard(raw_contract, args.contract.parent)
    try:
        _V3_A_GUARD_FD = acquire_v3_a_guard(raw_contract, args.contract.parent)
        _ENGINEERING_RUNNER_GUARD_FD = acquire_engineering_evidence_guard(
            raw_contract, args.contract.parent, "engineering_output_root",
            "engineering runner evidence",
        )
        _ENGINEERING_SCRATCH_GUARD_FD = acquire_engineering_evidence_guard(
            raw_contract, args.contract.parent, "engineering_verifier_scratch_root",
            "engineering scratch evidence",
        )
        return guarded_main(args)
    finally:
        if _EXECUTION_LOCK_FD is not None:
            os.close(_EXECUTION_LOCK_FD)
            _EXECUTION_LOCK_FD = None
        if _V2_B_GUARD_FD is not None:
            os.close(_V2_B_GUARD_FD)
            _V2_B_GUARD_FD = None
        if _V3_A_GUARD_FD is not None:
            os.close(_V3_A_GUARD_FD)
            _V3_A_GUARD_FD = None
        if _ENGINEERING_RUNNER_GUARD_FD is not None:
            os.close(_ENGINEERING_RUNNER_GUARD_FD)
            _ENGINEERING_RUNNER_GUARD_FD = None
        if _ENGINEERING_SCRATCH_GUARD_FD is not None:
            os.close(_ENGINEERING_SCRATCH_GUARD_FD)
            _ENGINEERING_SCRATCH_GUARD_FD = None


if __name__ == "__main__":
    try:
        _EXIT_CODE = main()
    except Exception as exc:
        write_failure_receipt(exc)
        sys.stderr.write(f"{REDACTED_FAILURE_MESSAGE}\n")
        _EXIT_CODE = 2
    raise SystemExit(_EXIT_CODE)
