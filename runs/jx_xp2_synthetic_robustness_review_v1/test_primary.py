#!/usr/bin/env python3
"""No-dynamics regression tests for the frozen JX-XP2 primary design."""

from __future__ import annotations

import ast
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import subprocess
import struct
import sys
import tempfile
import os
import shutil
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primary = load_module("jx_xp2_primary_static_tests", "run_primary.py")
builder = load_module("jx_xp2_builder_static_tests", "build_design.py")
verifier = load_module("jx_xp2_verifier_static_tests", "verify_replay.py")
independent = load_module("jx_xp2_independent_static_tests", "run_independent.py")


def binary64_hex(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def rebound_string_hash(value: str) -> int:
    """Pure MurmurHash3-x86-32 equivalent to REBOUND's string hash."""
    data = value.encode("utf-8")
    mask = 0xFFFFFFFF

    def rotate_left(item: int, count: int) -> int:
        return ((item << count) | (item >> (32 - count))) & mask

    result = 1983
    for offset in range(0, len(data) - len(data) % 4, 4):
        block = int.from_bytes(data[offset:offset + 4], "little")
        block = (block * 0xCC9E2D51) & mask
        block = rotate_left(block, 15)
        block = (block * 0x1B873593) & mask
        result ^= block
        result = rotate_left(result, 13)
        result = (result * 5 + 0xE6546B64) & mask
    tail = data[len(data) - len(data) % 4:]
    block = 0
    if len(tail) == 3:
        block ^= tail[2] << 16
    if len(tail) >= 2:
        block ^= tail[1] << 8
    if tail:
        block ^= tail[0]
        block = (block * 0xCC9E2D51) & mask
        block = rotate_left(block, 15)
        block = (block * 0x1B873593) & mask
        result ^= block
    result ^= len(data)
    result ^= result >> 16
    result = (result * 0x85EBCA6B) & mask
    result ^= result >> 13
    result = (result * 0xC2B2AE35) & mask
    result ^= result >> 16
    return result & mask


def registered_particle_vector(configuration_id: str) -> list[list[object]]:
    initial = json.loads((ROOT / "initial_states_v1.json").read_text(encoding="utf-8"))
    rows = list(initial["common_active_sun_centered_rows"])
    if configuration_id != "M0":
        configuration = next(
            item for item in initial["configuration_states"]
            if item[0] == configuration_id
        )
        rows.append(configuration[2])
    rows.extend(initial["tracer_sun_centered_rows"])
    return [
        [rebound_string_hash(row[0]), row[2], binary64_hex(0.0)]
        for row in rows
    ]


def synthetic_engineering_endpoint(
    configuration_id: str = "M0", *, dt_years: float = 0.125,
    end_years: float = 50_000.0,
) -> dict:
    vector = registered_particle_vector(configuration_id)
    particle_count = len(vector)
    active_count = particle_count - 128
    zero = binary64_hex(0.0)
    simulation = {
        field: (zero if field.endswith("_hex") else 0)
        for field in primary.CONTINUATION_SIMULATION_FIELDS
    }
    boolean_fields = {
        "particle_capacity_covers_logical_count", "particle_storage_present",
        "active_memory_ranges_pairwise_disjoint", "variation_config_present",
        "particle_lookup_present", "gravity_compensated_sums_present",
        "tree_root_present", "messages_present", "display_view_present",
        "display_data_present", "server_data_present", "collision_storage_present",
        "odes_present", "extras_present", "simulationarchive_filename_present",
    }
    simulation.update({field: False for field in boolean_fields})
    simulation.update({
        "t_hex": binary64_hex(end_years),
        "G_hex": binary64_hex(39.47841760435743),
        "dt_hex": binary64_hex(dt_years),
        "dt_last_done_hex": binary64_hex(dt_years),
        "steps_done": int(round(end_years / dt_years)),
        "save_messages": 1, "N": particle_count, "N_active": active_count,
        "particle_capacity_covers_logical_count": True,
        "particle_storage_present": True,
        "active_memory_ranges_pairwise_disjoint": True,
        "integrator": "mercurius", "gravity": "mercurius",
        "boundary": "none", "collision": "none", "exact_finish_time": 1,
        "opening_angle2_hex": binary64_hex(0.25),
        "boxsize_hex": [zero, zero, zero], "root_size_hex": binary64_hex(-1.0),
        "N_root": 1, "N_root_xyz": [1, 1, 1], "N_ghost_xyz": [0, 0, 0],
        "callbacks_present": {
            field: False for field in primary.CONTINUATION_CALLBACK_FIELDS
        },
    })
    normalized_mercurius = {
        "encounter_N", "encounter_N_active", "tponly_encounter",
        "allocated_particle_backup_count",
        "allocated_additional_forces_backup_count", "particles_backup_present",
        "additional_forces_backup_present", "encounter_map_present",
    }
    mercurius = {
        field: (zero if field.endswith("_hex") else 0)
        for field in primary.CONTINUATION_MERCURIUS_FIELDS
        if field not in normalized_mercurius
    }
    mercurius.update({
        "r_crit_hill_hex": binary64_hex(3.0), "safe_mode": 1, "mode": 0,
        "is_synchronized": 1, "recalculate_coordinates_this_timestep": 1,
        "recalculate_r_crit_this_timestep": 0,
        "dcrit_storage_present": True,
        "dcrit_capacity_covers_logical_count": True,
        "dcrit_hex": [zero] * particle_count,
        "com_position_hex": [zero, zero, zero],
        "com_velocity_hex": [zero, zero, zero], "L_callback_present": False,
    })
    particles = [{
        "index": index, "hash": particle_hash,
        "simulation_reference_bound_to_parent": True,
        "m_hex": mass, "r_hex": radius,
        "x_hex": zero, "y_hex": zero, "z_hex": zero,
        "vx_hex": zero, "vy_hex": zero, "vz_hex": zero,
        "ax_hex": zero, "ay_hex": zero, "az_hex": zero,
        "last_collision_hex": zero, "collision_cell_present": False,
        "additional_properties_present": False,
    } for index, (particle_hash, mass, radius) in enumerate(vector)]
    return {
        "schema": "jx-xp2-mercurius-live-archive-endpoint/v1",
        "simulation": simulation, "mercurius": mercurius,
        "whfast": {
            "coordinates": "jacobi", "kernel": "default", "corrector": 0,
            "corrector2": 0, "recalculate_coordinates_this_timestep": 0,
            "safe_mode": 1, "keep_unsynchronized": 0, "is_synchronized": 1,
            "timestep_warning": 0, "unsynchronized_recalculation_warning": 0,
        },
        "ias15": {
            "epsilon_hex": binary64_hex(1e-9), "min_dt_hex": zero,
            "adaptive_mode": "prs23", "iterations_max_exceeded": 0,
        },
        "particles": particles,
        "save_load_normalized_mercurius_fields": [
            "encounter_N", "encounter_N_active", "tponly_encounter",
            "allocated_particle_backup_count",
            "allocated_additional_forces_backup_count",
            "particles_backup_present", "additional_forces_backup_present",
            "encounter_map_present",
        ],
        "save_load_normalized_whfast_fields": ["internal_particle_arrays_present"],
        "save_load_normalized_ias15_fields": [
            "stored_coordinate_count", "direct_array_sha256",
            "coefficient_array_sha256", "map_count", "map_sha256",
        ],
        "excluded_noncontinuation_fields": list(primary.CONTINUATION_EXCLUDED_FIELDS),
    }


def synthetic_live_topology(endpoint: dict) -> dict:
    particle_count = endpoint["simulation"]["N"]
    direct = {name: True for name in ("at", "x0", "v0", "a0", "csx", "csv", "csa0")}
    coefficients = {
        name: [True] * 7 for name in ("g", "b", "csb", "e", "br", "er")
    }
    digest = "0" * 64
    endpoint_sha = primary.sha256_bytes(
        primary.ENDPOINT_DIGEST_DOMAIN + primary.canonical_bytes(endpoint)
    )
    return {
        "source_mode": "LIVE_BOUNDARY",
        "structural_projection_validation_passed": True,
        "simulation": {"N": particle_count, "N_allocated": 256,
                       "particles_present": True},
        "mercurius": {
            "dcrit_count": 256, "dcrit_present": True,
            "allocated_particle_backup_count": 256,
            "particles_backup_present": True, "encounter_map_present": True,
            "encounter_N": 0, "encounter_N_active": 0, "tponly_encounter": 0,
            "allocated_additional_forces_backup_count": 0,
            "additional_forces_backup_present": False,
        },
        "whfast": {
            "particle_count": 256, "particle_present": True,
            "temporary_count": 0, "temporary_present": False,
            "internal_particle_arrays_present": True,
        },
        "ias15": {
            "stored_coordinate_count": 9, "map_count": 0, "map_present": False,
            "direct_pointer_presence": direct,
            "coefficient_pointer_presence": coefficients,
            "direct_array_sha256": {name: digest for name in direct},
            "coefficient_array_sha256": {
                name: [digest] * 7 for name in coefficients
            },
            "map_sha256": None,
        },
        "normalized_endpoint_projection": endpoint,
        "normalized_endpoint_sha256": endpoint_sha,
        "strict_projection_sha256": None,
    }


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((ROOT / "contract_v1.json").read_text(encoding="utf-8"))
        cls.seed = json.loads((ROOT / "seed_manifest_v1.json").read_text(encoding="utf-8"))

    def test_builder_reproduces_registered_preoutput_bytes(self) -> None:
        rebuilt = builder.canonical(builder.generate(self.contract, self.seed)) + b"\n"
        artifact = (ROOT / "initial_states_v1.json").read_bytes()
        self.assertEqual(rebuilt, artifact)
        self.assertEqual(
            primary.sha256_bytes(artifact),
            self.contract["initial_state_policy"]["artifact_sha256"],
        )

    def test_verifier_does_not_import_either_runner(self) -> None:
        tree = ast.parse((ROOT / "verify_replay.py").read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name.endswith("run_primary") for name in imported))
        self.assertFalse(any(name.endswith("run_independent") for name in imported))

    def test_registered_inventory_has_no_missing_source(self) -> None:
        expected = set(self.contract["result_policy"]["registered_package_inventory"])
        self.assertIn("test_primary.py", expected)
        self.assertTrue(expected - {
            "engineering_registration_v1.json", "registration_v1.json",
        } <= {
            path.name for path in ROOT.iterdir() if path.is_file() and not path.is_symlink()
        })

    def test_cli_help_success_is_not_converted_to_failure(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run_primary.py"), "--help"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)


class MetricAndClassificationTests(unittest.TestCase):
    def tracker(self) -> list[dict]:
        return [{
            "logical_id": f"XP2-B{block:02d}-T{index:02d}",
            "block_index": block,
            "index_within_block": index,
            "minimum_sampled_q_AU": 50.0,
            "first_sampled_q_below_30_time_year": None,
            "first_sampled_q_below_35_time_year": None,
            "first_sampled_q_below_40_time_year": None,
            "all_samples_finite_cartesian_and_osculating": True,
        } for block in range(8) for index in range(16)]

    def test_prefix_minimum_and_crossing_presence_are_equivalent(self) -> None:
        rows = self.tracker()
        primary.validate_tracker_shape(rows, 50_000.0)
        rows[0]["minimum_sampled_q_AU"] = 34.0
        with self.assertRaises(primary.IntegrityError):
            primary.validate_tracker_shape(rows, 50_000.0)
        rows[0]["first_sampled_q_below_40_time_year"] = 50.0
        rows[0]["first_sampled_q_below_35_time_year"] = 50.0
        primary.validate_tracker_shape(rows, 50_000.0)

    def test_bool_is_not_accepted_as_a_number(self) -> None:
        rows = self.tracker()
        rows[0]["minimum_sampled_q_AU"] = True
        with self.assertRaises(primary.IntegrityError):
            primary.validate_tracker_shape(rows, 50_000.0)

    def test_effects_use_registered_block_metadata(self) -> None:
        rows = self.tracker()
        arms = {
            arm: {"particles": copy.deepcopy(rows)} for arm in primary.PRIMARY_ARM_IDS
        }
        result = primary.structural_effects(arms)
        self.assertEqual(result["mixture_numerator"], 0)
        self.assertEqual(result["block_numerators"], {str(i): 0 for i in range(8)})
        arms["CI01-P0"]["particles"][0]["block_index"] = 7
        with self.assertRaises(primary.IntegrityError):
            primary.structural_effects(arms)


class IndependentVerifierSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((ROOT / "contract_v1.json").read_text(encoding="utf-8"))
        cls.seed = json.loads((ROOT / "seed_manifest_v1.json").read_text(encoding="utf-8"))
        cls.initial = json.loads((ROOT / "initial_states_v1.json").read_text(encoding="utf-8"))

    def test_independent_source_and_com_reconstruction_reject_mutations(self) -> None:
        verifier.verify_source_rows(self.contract, self.seed, self.initial)
        verifier.reconstruct_states(self.initial, self.contract, self.seed)
        mutated = copy.deepcopy(self.initial)
        value = bytearray.fromhex(mutated["tracer_sun_centered_rows"][0][3])
        value[-1] ^= 1
        mutated["tracer_sun_centered_rows"][0][3] = value.hex()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_source_rows(self.contract, self.seed, mutated)
        mutated = copy.deepcopy(self.initial)
        value = bytearray.fromhex(mutated["configuration_states"][0][3])
        value[-1] ^= 1
        mutated["configuration_states"][0][3] = value.hex()
        with self.assertRaises(verifier.VerificationError):
            verifier.reconstruct_states(mutated, self.contract, self.seed)

    def test_nested_output_trees_and_raw_symlinks_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / "package"; package.mkdir()
            outer = base / "out"; outer.mkdir()
            nested = outer / "nested"; nested.mkdir()
            receipt = base / "receipts" / "final.json"
            with self.assertRaises(verifier.VerificationError):
                verifier.validate_external_paths(
                    [outer, nested], self.contract, package, receipt
                )
            target = base / "target"; target.mkdir()
            link = base / "link"; link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(verifier.VerificationError):
                verifier.reject_symlink_path(link / "child", "test path")

    def test_primary_arm_identity_locks_configuration_class_and_timestep(self) -> None:
        self.assertEqual(
            verifier.primary_arm_identity("CI05-P3", self.contract),
            ("CI05-P3", "PRIMARY_TIMESTEP", 0.125),
        )
        self.assertEqual(
            verifier.primary_arm_identity("AUDIT-CI05-P3", self.contract),
            ("CI05-P3", "HALF_TIMESTEP", 0.0625),
        )

    def test_dop_endpoint_hex_and_compact_history_are_independently_checked(self) -> None:
        expanded = verifier.reconstruct_states(self.initial, self.contract, self.seed)
        selection = json.loads((ROOT / "selection_manifest_v1.json").read_text())
        ids = verifier.selected_ids(selection, self.initial)
        arm = verifier.expected_dop_initial(expanded, ids, "M0")
        canonical_state = [value.hex() for value in arm["initial_state"]]
        self.assertEqual(
            verifier.dop_state_from_hex(canonical_state, len(canonical_state)),
            arm["initial_state"],
        )
        noncanonical = list(canonical_state)
        noncanonical[0] = verifier.pack6([arm["initial_state"][0]] * 6)[:16]
        with self.assertRaises(verifier.VerificationError):
            verifier.dop_state_from_hex(noncanonical, len(noncanonical))
        accumulator = verifier.initial_dop_accumulator(
            arm, self.contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        segment_zero = copy.deepcopy(accumulator)
        segment_zero["sample_count"] = 1001
        segment_zero["sample_state_chain_head"] = "a" * 64
        verifier.validate_dop_accumulator(
            segment_zero, arm, 0, self.contract, arm["initial_state"]
        )
        tampered = copy.deepcopy(segment_zero)
        first_id = arm["logical_ids"][arm["active_count"]]
        tampered["particles"][first_id]["current"]["q_AU"] += 1.0
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_dop_accumulator(
                tampered, arm, 0, self.contract, arm["initial_state"]
            )
        rewritten = copy.deepcopy(segment_zero)
        rewritten["particles"][first_id]["minimum_sampled_q_AU"] = (
            accumulator["particles"][first_id]["minimum_sampled_q_AU"] + 1.0
        )
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_dop_transition(accumulator, rewritten, arm, 0)


class LedgerStateMachineTests(unittest.TestCase):
    registration = "1" * 64
    label = "A"
    arm = "M0"

    def start(self, sequence: int, segment: int, attempt: int, predecessor: str) -> dict:
        payload = {
            "registration_sha256": self.registration,
            "execution_label": self.label,
            "arm_id": self.arm,
            "segment_index": segment,
            "predecessor_segment_chain_head": predecessor,
        }
        return {
            "schema": primary.ATTEMPT_SCHEMA,
            "sequence": sequence,
            "event": "START",
            "execution_label": self.label,
            "arm_id": self.arm,
            "segment_index": segment,
            "attempt_index": attempt,
            "predecessor_segment_chain_head": predecessor,
            "input_key_sha256": primary.sha256_bytes(primary.canonical_bytes(payload)),
        }

    def passed(self, sequence: int, segment: int, attempt: int, chain: str) -> dict:
        return {
            "schema": primary.ATTEMPT_SCHEMA,
            "sequence": sequence,
            "event": "PASS",
            "execution_label": self.label,
            "arm_id": self.arm,
            "segment_index": segment,
            "attempt_index": attempt,
            "return_code": 0,
            "segment_chain_head": chain,
        }

    def test_exact_per_arm_progression(self) -> None:
        chain0 = "a" * 64
        rows = [
            self.start(1, 0, 1, primary.INITIAL_SEGMENT_CHAIN),
            self.passed(2, 0, 1, chain0),
            self.start(3, 1, 1, chain0),
        ]
        self.assertEqual(
            primary.validate_attempt_ledger(rows, self.label, self.registration),
            {(self.arm, 1, 1)},
        )

    def test_second_start_after_pass_rejects(self) -> None:
        chain0 = "a" * 64
        rows = [
            self.start(1, 0, 1, primary.INITIAL_SEGMENT_CHAIN),
            self.passed(2, 0, 1, chain0),
            self.start(3, 0, 2, chain0),
        ]
        with self.assertRaises(primary.IntegrityError):
            primary.validate_attempt_ledger(rows, self.label, self.registration)

    def test_skipped_segment_rejects(self) -> None:
        rows = [self.start(1, 1, 1, primary.INITIAL_SEGMENT_CHAIN)]
        with self.assertRaises(primary.IntegrityError):
            primary.validate_attempt_ledger(rows, self.label, self.registration)

    def test_pass_requires_chain_binding(self) -> None:
        row = self.passed(2, 0, 1, "a" * 64)
        row.pop("segment_chain_head")
        with self.assertRaises(primary.IntegrityError):
            primary.validate_attempt_ledger(
                [self.start(1, 0, 1, primary.INITIAL_SEGMENT_CHAIN), row],
                self.label, self.registration,
            )


class V3FailureProtocolTests(unittest.TestCase):
    registration = "1" * 64
    contract = json.loads((ROOT / "contract_v1.json").read_text(encoding="utf-8"))

    def make_tree(self, base: Path, *, arm: str = "M0", segment: int = 0,
                  attempt: int = 1) -> tuple[Path, Path, dict]:
        root = base / "output"; root.mkdir()
        (root / "failures").mkdir()
        (root / "arms" / arm / "segments").mkdir(parents=True)
        ledger = root / "attempt_ledger.jsonl"
        predecessor = primary.INITIAL_SEGMENT_CHAIN
        start = {
            "schema": primary.ATTEMPT_SCHEMA, "sequence": 1, "event": "START",
            "execution_label": "A", "arm_id": arm, "segment_index": segment,
            "attempt_index": attempt, "predecessor_segment_chain_head": predecessor,
            "input_key_sha256": primary.sha256_bytes(primary.canonical_bytes({
                "registration_sha256": self.registration, "execution_label": "A",
                "arm_id": arm, "segment_index": segment,
                "predecessor_segment_chain_head": predecessor,
            })),
        }
        primary.append_ledger(ledger, start)
        return root, ledger, start

    def publish(self, root: Path, ledger: Path, start: dict, failure_class: str,
                return_code: int | None,
                quarantined_artifacts: list[dict] | None = None) -> list[dict]:
        rows = [start]
        primary.append_failure_terminal(
            ledger, rows, root, start, failure_class=failure_class,
            return_code=return_code,
            quarantined_artifacts=quarantined_artifacts or [],
        )
        primary.validate_attempt_ledger(rows, "A", self.registration)
        primary.validate_failure_receipt_bindings(rows, root)
        verifier.verify_primary_failures(root, rows, "A", self.contract, {})
        return rows

    def test_worker_uses_own_start_prefix_not_sibling_commit_pass_window(self) -> None:
        sibling = {
            "schema": primary.ATTEMPT_SCHEMA, "sequence": 1, "event": "START",
            "execution_label": "A", "arm_id": "CI01-P0", "segment_index": 0,
            "attempt_index": 1,
            "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
            "input_key_sha256": primary.sha256_bytes(primary.canonical_bytes({
                "registration_sha256": self.registration, "execution_label": "A",
                "arm_id": "CI01-P0", "segment_index": 0,
                "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
            })),
        }
        target = copy.deepcopy(sibling)
        target.update({"sequence": 2, "arm_id": "M0"})
        target["input_key_sha256"] = primary.sha256_bytes(primary.canonical_bytes({
            "registration_sha256": self.registration, "execution_label": "A",
            "arm_id": "M0", "segment_index": 0,
            "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
        }))
        self.assertEqual(
            primary.validate_worker_start(
                [sibling, target], execution_label="A",
                registration_sha256=self.registration, arm_id="M0",
                segment_index=0, attempt_index=1,
                predecessor_segment_chain_head=primary.INITIAL_SEGMENT_CHAIN,
            ), target,
        )

    def test_precontext_exit_timeout_signal_rss_and_recovery_are_receipted(self) -> None:
        cases = (
            ("CHILD_EXIT_NONZERO", 2), ("CHILD_SIGNAL", -15),
            ("SEGMENT_TIMEOUT", -9), ("CHILD_RSS_LIMIT", -9),
            ("RECOVERED_UNCOMMITTED", None),
        )
        for failure_class, return_code in cases:
            with self.subTest(failure_class=failure_class), tempfile.TemporaryDirectory() as d:
                root, ledger, start = self.make_tree(Path(d))
                rows = self.publish(root, ledger, start, failure_class, return_code)
                self.assertEqual(rows[-1]["failure_class"], failure_class)
                self.assertEqual(len(list((root / "failures").glob("failure_*.json"))), 1)

    def test_receipt_before_fail_crash_is_reconciled_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root, ledger, start = self.make_tree(Path(d))
            core = primary.failure_event_core(
                execution_label="A", arm_id="M0", segment_index=0, attempt_index=1,
                start_sequence=1, return_code=2, failure_class="CHILD_EXIT_NONZERO",
            )
            receipt = primary.build_failure_receipt(start, core, [])
            name = primary.failure_receipt_filename("M0", 0, 1)
            primary.atomic_json(root / "failures" / name, receipt)
            rows = [start]
            self.assertTrue(primary.reconcile_orphan_failure_receipt(
                ledger, rows, root, start
            ))
            primary.validate_failure_receipt_bindings(rows, root)

    def test_complete_and_partial_pending_publications_recover_safely(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root, ledger, start = self.make_tree(Path(d))
            core = primary.failure_event_core(
                execution_label="A", arm_id="M0", segment_index=0, attempt_index=1,
                start_sequence=1, return_code=2, failure_class="CHILD_EXIT_NONZERO",
            )
            receipt = primary.build_failure_receipt(start, core, [])
            name = primary.failure_receipt_filename("M0", 0, 1)
            pending = root / "failures" / f".{name}.pending"
            pending.write_bytes(primary.serialized_json(receipt))
            rows = [start]
            self.assertTrue(primary.reconcile_orphan_failure_receipt(
                ledger, rows, root, start
            ))
            self.assertFalse(pending.exists())
            primary.validate_failure_receipt_bindings(rows, root)
        with tempfile.TemporaryDirectory() as d:
            root, ledger, start = self.make_tree(Path(d))
            name = primary.failure_receipt_filename("M0", 0, 1)
            pending = root / "failures" / f".{name}.pending"
            pending.write_bytes(b"{")
            rows = [start]
            self.assertFalse(primary.reconcile_orphan_failure_receipt(
                ledger, rows, root, start
            ))
            self.assertFalse(pending.exists())
            quarantine = primary.quarantine_attempt_artifacts(root, "M0", 0, 1)
            self.publish(
                root, ledger, start, "RECOVERED_UNCOMMITTED", None, quarantine
            )

    def test_atomic_ledger_pending_recovery_and_canonical_framing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ledger = root / "attempt_ledger.jsonl"
            row = {
                "schema": primary.ATTEMPT_SCHEMA, "sequence": 1, "event": "START",
                "execution_label": "A", "arm_id": "M0", "segment_index": 0,
                "attempt_index": 1,
                "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
                "input_key_sha256": primary.sha256_bytes(primary.canonical_bytes({
                    "registration_sha256": self.registration,
                    "execution_label": "A", "arm_id": "M0", "segment_index": 0,
                    "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
                })),
            }
            pending = root / ".attempt_ledger.jsonl.pending"
            pending.write_bytes(primary.canonical_bytes(row) + b"\n")
            primary.recover_pending_ledger(
                ledger, execution_label="A", registration_sha256=self.registration,
                output_root=root,
            )
            self.assertEqual(primary.read_jsonl(ledger), [row])
            ledger.write_bytes(ledger.read_bytes()[:-1])
            with self.assertRaises(primary.IntegrityError):
                primary.read_jsonl(ledger)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ledger = root / "attempt_ledger.jsonl"
            pending = root / ".attempt_ledger.jsonl.pending"
            pending.write_bytes(b"{")
            primary.recover_pending_ledger(
                ledger, execution_label="A", registration_sha256=self.registration,
                output_root=root,
            )
            self.assertFalse(pending.exists())
            self.assertFalse(ledger.exists())
        for cut in (
            "mid_old", "exact_old", "mid_new", "complete", "divergent",
            "wrong_sequence", "wrong_event",
        ):
            with self.subTest(cut=cut), tempfile.TemporaryDirectory() as d:
                root = Path(d); ledger = root / "attempt_ledger.jsonl"
                first = {
                    "schema": primary.ATTEMPT_SCHEMA, "sequence": 1,
                    "event": "START", "execution_label": "A", "arm_id": "M0",
                    "segment_index": 0, "attempt_index": 1,
                    "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
                    "input_key_sha256": primary.sha256_bytes(primary.canonical_bytes({
                        "registration_sha256": self.registration,
                        "execution_label": "A", "arm_id": "M0", "segment_index": 0,
                        "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
                    })),
                }
                second = dict(first, sequence=2, arm_id="CI01-P0")
                second["input_key_sha256"] = primary.sha256_bytes(
                    primary.canonical_bytes({
                        "registration_sha256": self.registration,
                        "execution_label": "A", "arm_id": "CI01-P0",
                        "segment_index": 0,
                        "predecessor_segment_chain_head": primary.INITIAL_SEGMENT_CHAIN,
                    })
                )
                old = primary.canonical_bytes(first) + b"\n"
                extension = old + primary.canonical_bytes(second) + b"\n"
                ledger.write_bytes(old)
                pending = root / ".attempt_ledger.jsonl.pending"
                if cut == "mid_old":
                    candidate = old[:len(old) // 2]
                elif cut == "exact_old":
                    candidate = old
                elif cut == "mid_new":
                    candidate = old + primary.canonical_bytes(second)[:17]
                elif cut == "complete":
                    candidate = extension
                elif cut == "wrong_sequence":
                    candidate = old + primary.canonical_bytes(
                        dict(second, sequence=3)
                    ) + b"\n"
                elif cut == "wrong_event":
                    candidate = old + primary.canonical_bytes(
                        dict(second, event="UNKNOWN")
                    ) + b"\n"
                else:
                    candidate = b"[" + old[1:]
                pending.write_bytes(candidate)
                if cut in {"divergent", "wrong_sequence", "wrong_event"}:
                    with self.assertRaises(primary.IntegrityError):
                        primary.recover_pending_ledger(
                            ledger, execution_label="A",
                            registration_sha256=self.registration, output_root=root,
                        )
                    continue
                primary.recover_pending_ledger(
                    ledger, execution_label="A",
                    registration_sha256=self.registration, output_root=root,
                )
                self.assertFalse(pending.exists())
                self.assertEqual(ledger.read_bytes(), extension if cut == "complete" else old)

    def test_top_level_manifest_and_result_pending_recovery(self) -> None:
        for partial in (False, True):
            with self.subTest(manifest_partial=partial), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "output"; root.mkdir()
                (root / "arms").mkdir(); (root / "failures").mkdir()
                expected = {"schema": "manifest", "execution_label": "A"}
                pending = root / ".run_manifest.json.pending"
                pending.write_bytes(b"{" if partial else primary.serialized_json(expected))
                primary.recover_or_validate_run_manifest(root, expected)
                self.assertEqual(primary.strict_json(root / "run_manifest.json"), expected)
                self.assertFalse(pending.exists())
        for partial in (False, True):
            with self.subTest(result_partial=partial), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "output"; root.mkdir()
                ledger = root / "attempt_ledger.jsonl"; ledger.write_bytes(b"ledger\n")
                expected = {
                    "schema": "result", "semantic": {"value": 1},
                    "resource_provenance": {
                        "wall_seconds": 1.0, "coordinator_peak_rss_bytes": 1,
                        "output_bytes_before_result": ledger.stat().st_size,
                        "attempt_ledger_sha256": primary.sha256_file(ledger),
                    },
                }
                pending = root / ".result_v1.json.pending"
                pending.write_bytes(b"{" if partial else primary.serialized_json(expected))
                contract = {"resource_caps_per_execution": {
                    "max_wall_seconds_total": 10.0,
                    "max_peak_rss_bytes_per_process": 1024,
                }}
                primary.publish_or_recover_primary_result(root, expected, contract)
                self.assertEqual(primary.strict_json(root / "result_v1.json"), expected)
                self.assertFalse(pending.exists())

    def test_empty_arm_mkdir_before_start_crash_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "output"; (root / "arms" / "M0").mkdir(parents=True)
            primary.recover_empty_arm_skeletons(root, [])
            self.assertTrue((root / "arms" / "M0" / "segments").is_dir())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "output"; arm = root / "arms" / "M0"; arm.mkdir(parents=True)
            (arm / "unexpected").write_bytes(b"x")
            with self.assertRaises(primary.IntegrityError):
                primary.recover_empty_arm_skeletons(root, [])

    def test_missing_extra_and_tampered_receipts_are_rejected(self) -> None:
        for mutation in ("missing", "extra", "tampered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root, ledger, start = self.make_tree(Path(d))
                rows = self.publish(root, ledger, start, "CHILD_EXIT_NONZERO", 2)
                receipt_path = root / "failures" / rows[-1]["failure_receipt_filename"]
                if mutation == "missing":
                    receipt_path.unlink()
                elif mutation == "extra":
                    (root / "failures" / "failure_extra.json").write_text("{}\n")
                else:
                    receipt = json.loads(receipt_path.read_text())
                    receipt["return_code"] = 3
                    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
                with self.assertRaises(primary.IntegrityError):
                    primary.validate_failure_receipt_bindings(rows, root)
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_primary_failures(
                        root, rows, "A", self.contract, {}
                    )

    def test_partial_quarantine_is_idempotently_bound(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root, ledger, start = self.make_tree(Path(d))
            segment_dir = root / "arms" / "M0" / "segments"
            source = segment_dir / "segment_00_attempt_01_state.bin"
            source.write_bytes(b"state")
            existing_name = "M0_attempt_01_segment_00_attempt_01_receipt.json"
            (root / "failures" / existing_name).write_bytes(b"receipt")
            inventory = primary.quarantine_attempt_artifacts(root, "M0", 0, 1)
            self.assertEqual({row["filename"] for row in inventory}, {
                "M0_attempt_01_segment_00_attempt_01_state.bin", existing_name,
            })
            rows = [start]
            primary.append_failure_terminal(
                ledger, rows, root, start, failure_class="RECOVERED_UNCOMMITTED",
                return_code=None, quarantined_artifacts=inventory,
            )
            primary.validate_failure_receipt_bindings(rows, root)


@unittest.skip("historical numerical archive fixtures are not run before v4 registration")
class V3SemanticRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((ROOT / "contract_v1.json").read_text())
        cls.v2 = ROOT.parent / "jx_xp2_robustness_v2"
        cls.runs = ROOT.parent / "jx_xp2_runs_v2"
        cls.a_state = cls.runs / (
            "output_a/arms/M0/segments/segment_00_attempt_01_state.bin"
        )
        cls.b_state = cls.runs / (
            "output_b/arms/M0/segments/segment_00_attempt_01_state.bin"
        )
        cls.ci_state = cls.runs / (
            "output_a/arms/CI01-P0/segments/segment_00_attempt_01_state.bin"
        )
        cls.rebound = primary.get_rebound(cls.contract)

    def load(self, path: Path | None = None):
        return self.rebound.Simulation(str(path or self.a_state))

    def test_v2_science_bytes_and_numerical_functions_are_exact(self) -> None:
        for filename in (
            "seed_manifest_v1.json", "selection_manifest_v1.json",
            "initial_states_v1.json", "build_design.py",
        ):
            self.assertEqual((ROOT / filename).read_bytes(), (self.v2 / filename).read_bytes())
        old = json.loads((self.v2 / "contract_v1.json").read_text())
        for key in (
            "claim_ceiling", "permissions", "design_core", "seed_policy",
            "initial_state_policy", "classification", "numerical_gates",
            "independent_sentinel", "resource_caps_per_execution", "runtime_lock",
            "mandatory_nonclaim",
        ):
            self.assertEqual(self.contract[key], old[key], key)
        for filename, names in {
            "run_primary.py": (
                "active_snapshot", "update_invariant_maximum", "build_simulation",
                "update_sample_stream", "sample_tracers", "summarize_particles",
            ),
            "run_independent.py": ("active_invariants", "integrate_segment"),
        }.items():
            def functions(path: Path) -> dict[str, str]:
                tree = ast.parse(path.read_text())
                return {
                    node.name: ast.dump(node, include_attributes=False)
                    for node in tree.body if isinstance(node, ast.FunctionDef)
                }
            before = functions(self.v2 / filename); after = functions(ROOT / filename)
            for name in names:
                self.assertEqual(after[name], before[name], f"{filename}:{name}")

    def test_every_runner_ast_change_is_in_the_protocol_only_allowlist(self) -> None:
        allowed = {
            "run_primary.py": {
                "acquire_execution_lock", "acquire_v2_b_guard",
                "append_failure_terminal", "attempt_quarantine_names",
                "build_failure_receipt", "build_primary_result",
                "commit_segment_attempt", "complete_attempt_evidence",
                "decoded_continuation_projection", "decoded_double_array_sha256",
                "decoded_state_sha256", "directory_bytes",
                "expected_continuation_declaration_v3",
                "expected_raw_artifact_declaration_v1",
                "expected_v4_fresh_repair_declaration", "failure_event_core",
                "failure_terminal_row", "internal_segment", "load_completed_segment",
                "main", "parser", "protected_roots", "quarantine_attempt_artifacts",
                "raw_artifact_integrity_inventory", "reconcile_orphan_failure_receipt",
                "recover_pending_ledger", "require_complete_attempt_semantic_match",
                "run_one_segment", "run_supervisor", "save_simulation_checkpoint",
                "semantic_segment_payload", "torn_failure_receipt_quarantine_name",
                "valid_ias15_continuation", "validate_a_prerequisite",
                "validate_attempt_ledger", "validate_attempt_payload",
                "validate_complete_attempt_evidence_shape", "validate_contract",
                "validate_decoded_continuation_settings",
                "validate_failure_receipt_payload", "validate_inherited_v2_b_guard",
                "validate_segment_chain", "validate_v2_replay_lineage",
                "validated_complete_uncommitted_attempt",
            },
            "run_independent.py": {
                "_kill_process_group", "acquire_output_execution_lock",
                "acquire_v2_b_guard",
                "directory_bytes", "expected_continuation_declaration_v3",
                "expected_raw_artifact_declaration_v1",
                "expected_v4_fresh_repair_declaration", "execute", "main",
                "supervise_worker", "validate_contract",
                "validate_inputs", "validate_output_root", "validate_v2_replay_lineage",
            },
            "verify_replay.py": {
                "decoded_primary_continuation_projection",
                "decoded_primary_double_array_sha256", "decoded_primary_state_sha256",
                "expected_continuation_declaration_v3",
                "expected_raw_artifact_declaration_v1",
                "expected_v4_fresh_repair_declaration", "primary_quarantine_names",
                "primary_raw_artifact_integrity_inventory",
                "primary_segment_semantic_payload", "publish", "read_primary_ledger",
                "release_verification_locks", "segment_semantic",
                "valid_primary_ias15_continuation",
                "validate_external_paths", "validate_package",
                "validate_primary_complete_attempt_evidence",
                "validate_v2_replay_lineage", "verify_primary_checkpoint_endpoint",
                "verify_primary_failures", "verify_primary_output",
                "verify_quarantined_complete_attempt", "verify_segment_tree",
                "verify_unlocked_execution_lock",
            },
        }
        for filename, expected_changed in allowed.items():
            def functions(path: Path) -> dict[str, str]:
                tree = ast.parse(path.read_text())
                return {
                    node.name: ast.dump(node, include_attributes=False)
                    for node in tree.body if isinstance(node, ast.FunctionDef)
                }
            before = functions(self.v2 / filename); after = functions(ROOT / filename)
            self.assertFalse(set(before) - set(after), filename)
            changed = {name for name, value in after.items() if before.get(name) != value}
            self.assertEqual(changed, expected_changed, filename)

    def test_contract_runner_and_verifier_freeze_one_continuation_domain(self) -> None:
        declaration = self.contract["checkpoint_and_resume"][
            "decoded_continuation_state_v3"
        ]
        self.assertEqual(primary.expected_continuation_declaration_v3(), declaration)
        self.assertEqual(independent.expected_continuation_declaration_v3(), declaration)
        self.assertEqual(verifier.expected_continuation_declaration_v3(), declaration)
        self.assertNotIn("checkpoint_sha256", primary.SEGMENT_SEMANTIC_FIELDS)
        self.assertNotIn("checkpoint_size_bytes", primary.SEGMENT_SEMANTIC_FIELDS)

    def test_real_v2_raw_archives_differ_but_v3_continuation_is_exact(self) -> None:
        self.assertNotEqual(primary.sha256_file(self.a_state), primary.sha256_file(self.b_state))
        a_runner = primary.decoded_continuation_projection(self.load(self.a_state))
        b_runner = primary.decoded_continuation_projection(self.load(self.b_state))
        a_verifier = verifier.decoded_primary_continuation_projection(self.load(self.a_state))
        b_verifier = verifier.decoded_primary_continuation_projection(self.load(self.b_state))
        self.assertEqual(a_runner, b_runner)
        self.assertEqual(a_runner, a_verifier)
        self.assertEqual(b_runner, b_verifier)

    def test_independent_v2_forensic_proof_accepts_exact_and_rejects_chain_mutations(self) -> None:
        evidence_source = ROOT / "v2_replay_defect_evidence_v1.json"
        evidence = json.loads(evidence_source.read_text())
        bindings = [
            evidence["v2_registration"], evidence["v2_completed_a_result"],
            evidence["v2_a_verification_receipt"], evidence["v2_a_run_manifest"],
            evidence["v2_b_run_manifest"],
        ]
        for segment_key in ("v2_a_m0_segment_00", "v2_b_m0_segment_00"):
            bindings.extend(
                evidence[segment_key][kind] for kind in ("commit", "receipt", "state")
            )
        with tempfile.TemporaryDirectory() as d:
            package = Path(d) / "jx_xp2_robustness_v3"; package.mkdir()
            evidence_path = package / evidence_source.name
            shutil.copyfile(evidence_source, evidence_path)
            for binding in bindings:
                source = (ROOT / binding["path"]).resolve()
                target = (package / binding["path"]).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            lock_path = (package / self.contract["xp2_v2_invalid_replay_lineage"][
                "v2_b_execution_lock_path"
            ]).resolve()
            lock_path.parent.mkdir(parents=True, exist_ok=True); lock_path.write_bytes(b"")
            try:
                accepted = verifier.validate_v2_replay_lineage(self.contract, package)
                self.assertEqual(accepted["comparison"][
                    "v3_expected_segment_chain_head_both_runs"
                ], "5cc01d89db885889ae8dc0c8ed2cc1de2f36969d941f9da849709e40063133bf")
            finally:
                verifier.release_verification_locks()
            original_sha = verifier.V2_DEFECT_EVIDENCE_SHA256
            original_size = verifier.V2_DEFECT_EVIDENCE_SIZE_BYTES
            try:
                for mutation in ("missing", "null", "wrong"):
                    with self.subTest(mutation=mutation):
                        changed = copy.deepcopy(evidence)
                        if mutation == "missing":
                            changed["comparison"].pop(
                                "v3_expected_segment_chain_head_both_runs"
                            )
                        elif mutation == "null":
                            changed["comparison"][
                                "v3_expected_segment_chain_head_both_runs"
                            ] = None
                        else:
                            changed["comparison"][
                                "v3_expected_segment_chain_head_both_runs"
                            ] = "0" * 64
                        payload = (json.dumps(changed, indent=2) + "\n").encode()
                        evidence_path.write_bytes(payload)
                        mutated_contract = copy.deepcopy(self.contract)
                        lineage = mutated_contract["xp2_v2_invalid_replay_lineage"]
                        lineage["defect_evidence_sha256"] = verifier.digest_bytes(payload)
                        lineage["defect_evidence_size_bytes"] = len(payload)
                        verifier.V2_DEFECT_EVIDENCE_SHA256 = lineage[
                            "defect_evidence_sha256"
                        ]
                        verifier.V2_DEFECT_EVIDENCE_SIZE_BYTES = len(payload)
                        try:
                            with self.assertRaises(verifier.VerificationError):
                                verifier.validate_v2_replay_lineage(mutated_contract, package)
                        finally:
                            verifier.release_verification_locks()
            finally:
                verifier.V2_DEFECT_EVIDENCE_SHA256 = original_sha
                verifier.V2_DEFECT_EVIDENCE_SIZE_BYTES = original_size

    def test_raw_identity_is_integrity_only_for_semantic_chain_and_retry(self) -> None:
        a_receipt_path = self.a_state.with_name("segment_00_attempt_01_receipt.json")
        b_receipt_path = self.b_state.with_name("segment_00_attempt_01_receipt.json")
        a = json.loads(a_receipt_path.read_text()); b = json.loads(b_receipt_path.read_text())
        digest = primary.decoded_state_sha256(self.load(self.a_state))
        a["decoded_integrator_state_sha256"] = digest
        b["decoded_integrator_state_sha256"] = digest
        self.assertNotEqual(a["checkpoint_sha256"], b["checkpoint_sha256"])
        self.assertEqual(primary.semantic_segment_payload(a), primary.semantic_segment_payload(b))
        self.assertEqual(
            primary.semantic_segment_chain(
                primary.INITIAL_SEGMENT_CHAIN, primary.semantic_segment_payload(a)
            ),
            primary.semantic_segment_chain(
                primary.INITIAL_SEGMENT_CHAIN, primary.semantic_segment_payload(b)
            ),
        )
        evidence = primary.complete_attempt_evidence(a, a_receipt_path)
        raw_only = copy.deepcopy(a)
        raw_only.update({
            "checkpoint_filename": "different_archive.bin",
            "checkpoint_sha256": "f" * 64,
            "checkpoint_size_bytes": a["checkpoint_size_bytes"] + 17,
        })
        primary.require_complete_attempt_semantic_match(raw_only, evidence)
        for field in ("decoded_integrator_state_sha256", "sampled_state_stream_sha256"):
            changed = copy.deepcopy(raw_only); changed[field] = "e" * 64
            with self.assertRaises(primary.IntegrityError):
                primary.require_complete_attempt_semantic_match(changed, evidence)
        changed = copy.deepcopy(raw_only)
        changed["tracker"][0]["minimum_sampled_q_AU"] += 1.0
        with self.assertRaises(primary.IntegrityError):
            primary.require_complete_attempt_semantic_match(changed, evidence)

    def test_continuation_mutations_change_digest_or_fail_frozen_settings(self) -> None:
        baseline = primary.decoded_state_sha256(self.load())
        for label, mutation in (
            ("gravity", lambda s: setattr(s, "gravity", "none")),
            ("status", lambda s: setattr(s, "_status", -3)),
            ("usleep", lambda s: setattr(s, "usleep", 1.0)),
            ("archive", lambda s: setattr(s, "simulationarchive_auto_step", 1)),
            ("mercurius_allocator", lambda s: setattr(s.ri_mercurius, "_N_allocated", 133)),
        ):
            with self.subTest(label=label):
                simulation = self.load(); mutation(simulation)
                projection = primary.decoded_continuation_projection(simulation)
                self.assertNotEqual(primary.decoded_state_sha256(simulation), baseline)
                with self.assertRaises(primary.IntegrityError):
                    primary.validate_decoded_continuation_settings(
                        projection, end_years=50_000.0, dt_years=0.125,
                        particle_count=133,
                    )
        inflated = self.load(); inflated.N = 135
        with self.assertRaises(primary.IntegrityError):
            primary.decoded_continuation_projection(inflated)
        dcrit = self.load(); dcrit.ri_mercurius._dcrit[0] += 1.0
        self.assertNotEqual(primary.decoded_state_sha256(dcrit), baseline)
        acceleration = self.load(); acceleration.particles[-1].ax += 1.0
        self.assertNotEqual(primary.decoded_state_sha256(acceleration), baseline)
        ias = self.load(self.ci_state)
        self.assertEqual(int(ias.ri_ias15._N_allocated), 9)
        ias_baseline = primary.decoded_state_sha256(ias)
        ias.ri_ias15._g.p0[0] += 1.0
        self.assertNotEqual(primary.decoded_state_sha256(ias), ias_baseline)

    def test_excluded_wall_clock_and_unused_rand_seed_do_not_change_semantics(self) -> None:
        simulation = self.load(); baseline = primary.decoded_state_sha256(simulation)
        simulation.walltime += 10.0
        simulation.walltime_last_steps += 1.0
        simulation.rand_seed += 1
        self.assertEqual(primary.decoded_state_sha256(simulation), baseline)

    def test_particle_parent_and_active_pointer_topology_fail_closed(self) -> None:
        simulation = self.load()
        simulation.particles[-1]._sim = type(simulation.particles[-1]._sim)()
        with self.assertRaises(primary.IntegrityError):
            primary.decoded_continuation_projection(simulation)
        simulation = self.load(); other = self.rebound.Simulation()
        simulation.particles[-1]._sim = ctypes.pointer(other)
        with self.assertRaises(primary.IntegrityError):
            primary.decoded_continuation_projection(simulation)
        for module_name, function_name, error_name in (
            ("run_primary.py", "decoded_continuation_projection", "IntegrityError"),
            ("verify_replay.py", "decoded_primary_continuation_projection", "VerificationError"),
        ):
            for mutation in ("alias", "misalign"):
                with self.subTest(module=module_name, mutation=mutation):
                    script = f'''\nimport ctypes, importlib.util, os, rebound\nspec=importlib.util.spec_from_file_location("target", {str(ROOT / module_name)!r})\nm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\ns=rebound.Simulation({str(self.ci_state)!r}); i=s.ri_ias15\nif {mutation!r} == "alias": i._csb.p6=i._csa0\nelse:\n p=i._csa0; address=ctypes.cast(p,ctypes.c_void_p).value\n i._csa0=ctypes.cast(ctypes.c_void_p(address+1),type(p))\ntry: getattr(m,{function_name!r})(s)\nexcept getattr(m,{error_name!r}): os._exit(0)\nos._exit(9)\n'''
                    completed = subprocess.run(
                        [sys.executable, "-c", script], stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                    self.assertEqual(completed.returncode, 0)

    def test_directory_accounting_and_receipt_publication_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "tree"; root.mkdir(); (root / "stable").write_bytes(b"abc")
            self.assertEqual(primary.directory_bytes(root), 3)
            with self.assertRaises(primary.IntegrityError):
                primary.directory_bytes(root / "missing")
            outside = Path(d) / "outside"; outside.mkdir()
            link = Path(d) / "link"; link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(primary.IntegrityError):
                primary.directory_bytes(link)
        for scanner, error in (
            (primary.directory_bytes, primary.IntegrityError),
            (independent.directory_bytes, independent.IntegrityError),
        ):
            with self.subTest(scanner=scanner.__module__), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "tree"; child = root / "child"; child.mkdir(parents=True)
                (child / "old").write_bytes(b"x")
                real_listdir = os.listdir
                raced = False

                def replace_child(directory_fd):
                    nonlocal raced
                    names = real_listdir(directory_fd)
                    if not raced:
                        raced = True
                        child.rename(root / "oldchild")
                        child.mkdir(); (child / "new").write_bytes(b"x" * 10_000)
                    return names

                with mock.patch.object(scanner.__globals__["os"], "listdir",
                                       side_effect=replace_child):
                    with self.assertRaises(error):
                        scanner(root)
            for race_kind in ("file_to_directory", "late_directory_replacement"):
                with (self.subTest(scanner=scanner.__module__, race=race_kind),
                      tempfile.TemporaryDirectory() as d):
                    root = Path(d) / "tree"; root.mkdir()
                    if race_kind == "file_to_directory":
                        victim = root / "victim"; victim.write_bytes(b"x")
                    else:
                        victim = root / "child"; victim.mkdir()
                        (victim / "old").write_bytes(b"x")
                    root_inode = root.stat().st_ino
                    real_listdir = os.listdir; raced = False

                    def replace_at_root_final(directory_fd):
                        nonlocal raced
                        names = real_listdir(directory_fd)
                        if not raced and os.fstat(directory_fd).st_ino == root_inode:
                            raced = True
                            if race_kind == "file_to_directory":
                                victim.unlink()
                            else:
                                victim.rename(root / "oldchild")
                            victim.mkdir(); (victim / "hidden").write_bytes(b"x" * 10_000)
                        return names

                    with mock.patch.object(scanner.__globals__["os"], "listdir",
                                           side_effect=replace_at_root_final):
                        with self.assertRaises(error):
                            scanner(root)
            with (self.subTest(scanner=scanner.__module__, race="final_file_unlink"),
                  tempfile.TemporaryDirectory() as d):
                root = Path(d) / "tree"; root.mkdir()
                victim = root / "victim"; victim.write_bytes(b"unpublished")
                real_listdir = os.listdir; raced = False

                def unlink_after_final_list(directory_fd):
                    nonlocal raced
                    names = real_listdir(directory_fd)
                    if not raced:
                        raced = True; victim.unlink()
                    return names

                with mock.patch.object(scanner.__globals__["os"], "listdir",
                                       side_effect=unlink_after_final_list):
                    self.assertEqual(scanner(root), 0)
            with (self.subTest(scanner=scanner.__module__, race="dir_to_file_first_stat"),
                  tempfile.TemporaryDirectory() as d):
                root = Path(d) / "tree"; child = root / "child"; child.mkdir(parents=True)
                (child / "old").write_bytes(b"x")
                real_stat = os.stat; raced = False

                def replace_directory_before_stat(path, *args, **kwargs):
                    nonlocal raced
                    if not raced and path == "child" and kwargs.get("dir_fd") is not None:
                        raced = True
                        child.rename(root / "oldchild")
                        child.write_bytes(b"x")
                    return real_stat(path, *args, **kwargs)

                with mock.patch.object(scanner.__globals__["os"], "stat",
                                       side_effect=replace_directory_before_stat):
                    with self.assertRaises(error):
                        scanner(root)
        with tempfile.TemporaryDirectory() as d:
            receipt = Path(d) / "receipt.json"; value = {"schema": "test", "ok": True}
            payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            pending = receipt.with_name(".receipt.json.pending")
            pending.write_bytes(payload[: len(payload) // 2])
            verifier.publish(receipt, value)
            self.assertEqual(receipt.read_bytes(), payload)
            verifier.publish(receipt, value)
            receipt.unlink(); pending.write_bytes(b"divergent")
            with self.assertRaises(verifier.VerificationError):
                verifier.publish(receipt, value)

    def test_inherited_execution_lock_blocks_resume_until_child_exits(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "output"; root.mkdir()
            lock_fd = primary.acquire_execution_lock(root, create=True)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(write_fd); os.read(read_fd, 1); os.close(read_fd); os._exit(0)
            os.close(read_fd)
            os.close(lock_fd)  # simulate coordinator death without LOCK_UN
            with self.assertRaises(primary.ResourceLimitError):
                primary.acquire_execution_lock(root, create=False)
            os.write(write_fd, b"x"); os.close(write_fd); os.waitpid(pid, 0)
            recovered = primary.acquire_execution_lock(root, create=False)
            os.close(recovered)


class V4ProtocolOnlyStaticTests(unittest.TestCase):
    def test_registered_v3_science_function_asts_are_exact(self) -> None:
        v3 = ROOT.parent / "jx_xp2_robustness_v3"

        def functions(path: Path) -> dict[str, str]:
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef)
            }

        for filename, names in {
            "run_primary.py": (
                "active_snapshot", "update_invariant_maximum", "build_simulation",
                "update_sample_stream", "sample_tracers", "summarize_particles",
            ),
            "run_independent.py": ("active_invariants", "integrate_segment"),
        }.items():
            previous = functions(v3 / filename)
            current = functions(ROOT / filename)
            for name in names:
                self.assertEqual(current[name], previous[name], f"{filename}:{name}")

    def test_registered_v3_science_artifacts_are_byte_exact(self) -> None:
        v3 = ROOT.parent / "jx_xp2_robustness_v3"
        for filename in (
            "build_design.py", "seed_manifest_v1.json", "selection_manifest_v1.json",
            "initial_states_v1.json",
        ):
            self.assertEqual((ROOT / filename).read_bytes(), (v3 / filename).read_bytes())
        previous = json.loads((v3 / "contract_v1.json").read_text())
        current = json.loads((ROOT / "contract_v1.json").read_text())
        for key in (
            "claim_ceiling", "permissions", "design_core", "seed_policy",
            "initial_state_policy", "classification", "numerical_gates",
            "independent_sentinel", "resource_caps_per_execution", "runtime_lock",
            "mandatory_nonclaim",
        ):
            self.assertEqual(current[key], previous[key], key)

    def test_engineering_gate_and_registered_particle_anchors_are_exact(self) -> None:
        contract = json.loads((ROOT / "contract_v1.json").read_text(encoding="utf-8"))
        expected = primary.expected_engineering_boundary_gate_v1()
        self.assertEqual(contract["engineering_boundary_gate_v1"], expected)
        self.assertEqual(expected, independent.expected_engineering_boundary_gate_v1())
        self.assertEqual(expected, verifier.expected_engineering_boundary_gate_v1())
        for configuration_id, expected_digest in (
            primary.ENGINEERING_PARTICLE_VECTOR_SHA256.items()
        ):
            vector = registered_particle_vector(configuration_id)
            digest = hashlib.sha256(
                primary.ENGINEERING_PARTICLE_VECTOR_DIGEST_DOMAIN
                + primary.canonical_bytes(vector)
            ).hexdigest()
            self.assertEqual(digest, expected_digest, configuration_id)
        self.assertEqual(
            expected["registered_particle_vector_sha256_by_configuration"],
            primary.ENGINEERING_PARTICLE_VECTOR_SHA256,
        )

    def test_deep_engineering_endpoint_semantics_reject_mutations_in_all_roles(self) -> None:
        endpoint = synthetic_engineering_endpoint()
        validators = (primary, independent, verifier)
        for module in validators:
            module.validate_engineering_endpoint(
                copy.deepcopy(endpoint), configuration_id="M0", particle_count=133,
                dt_years=0.125, end_years=50_000.0,
            )
        mutations = {
            "extra-field": lambda item: item.update({"extra": None}),
            "missing-field": lambda item: item.pop("particles"),
            "logical-N": lambda item: item["simulation"].update({"N": 0}),
            "integrator": lambda item: item["simulation"].update(
                {"integrator": "bogus"}
            ),
            "softening": lambda item: item["simulation"].update(
                {"softening_hex": binary64_hex(1.0)}
            ),
            "status": lambda item: item["simulation"].update({"status": 9}),
            "variation": lambda item: item["simulation"].update(
                {"N_var": 1, "variation_config_present": True}
            ),
            "extras": lambda item: item["simulation"].update(
                {"extras_present": True}
            ),
            "exact-finish": lambda item: item["simulation"].update(
                {"exact_finish_time": 0}
            ),
            "callback": lambda item: item["simulation"]["callbacks_present"].update(
                {"heartbeat": True}
            ),
            "active-mass": lambda item: item["particles"][0].update(
                {"m_hex": binary64_hex(2.0)}
            ),
            "hash-range": lambda item: item["particles"][0].update(
                {"hash": 0x1_0000_0000}
            ),
            "dcrit-negative": lambda item: item["mercurius"]["dcrit_hex"].__setitem__(
                0, binary64_hex(-1.0)
            ),
            "raw-parser": lambda item: item["particles"][0].update(
                {"m_hex": "0x1.0000000000000p+0"}
            ),
        }
        used_hashes = {item["hash"] for item in endpoint["particles"]}
        changed_hash = (endpoint["particles"][0]["hash"] + 1) & 0xFFFFFFFF
        while changed_hash in used_hashes:
            changed_hash = (changed_hash + 1) & 0xFFFFFFFF
        mutations["registered-anchor"] = lambda item: item["particles"][0].update(
            {"hash": changed_hash}
        )
        for label, mutate in mutations.items():
            candidate = copy.deepcopy(endpoint)
            mutate(candidate)
            for module in validators:
                with self.subTest(label=label, role=module.__name__):
                    with self.assertRaises(module.IntegrityError):
                        module.validate_engineering_endpoint(
                            candidate, configuration_id="M0", particle_count=133,
                            dt_years=0.125, end_years=50_000.0,
                        )

    def test_live_nalloc_256_topology_is_admissible_but_incoherence_is_not(self) -> None:
        endpoint = synthetic_engineering_endpoint()
        topology = synthetic_live_topology(endpoint)
        validators = (primary, independent, verifier)
        for module in validators:
            module.validate_engineering_topology(
                copy.deepcopy(topology), source_mode="LIVE_BOUNDARY",
                configuration_id="M0", particle_count=133,
                dt_years=0.125, end_years=50_000.0,
            )
        mutations = {
            "extra-field": lambda item: item.update({"extra": None}),
            "missing-field": lambda item: item.pop("source_mode"),
            "outer-N": lambda item: item["simulation"].update({"N": 134}),
            "dcrit-short": lambda item: item["mercurius"].update(
                {"dcrit_count": 132}
            ),
            "backup-short": lambda item: item["mercurius"].update(
                {"allocated_particle_backup_count": 132}
            ),
            "encounter-negative": lambda item: item["mercurius"].update(
                {"encounter_N": -1}
            ),
            "encounter-order": lambda item: item["mercurius"].update(
                {"encounter_N": 1, "encounter_N_active": 2}
            ),
            "whfast-short": lambda item: item["whfast"].update(
                {"particle_count": 132}
            ),
            "ias-digest": lambda item: item["ias15"]["direct_array_sha256"].update(
                {"at": None}
            ),
        }
        for label, mutate in mutations.items():
            candidate = copy.deepcopy(topology)
            mutate(candidate)
            for module in validators:
                with self.subTest(label=label, role=module.__name__):
                    with self.assertRaises(module.IntegrityError):
                        module.validate_engineering_topology(
                            candidate, source_mode="LIVE_BOUNDARY",
                            configuration_id="M0", particle_count=133,
                            dt_years=0.125, end_years=50_000.0,
                        )

    def test_held_engineering_snapshot_binds_selected_root_and_inventory(self) -> None:
        for module in (primary, independent, verifier):
            with self.subTest(role=module.__name__), tempfile.TemporaryDirectory() as d:
                base = Path(d)
                root_a = base / "root-a"; root_a.mkdir()
                root_b = base / "root-b"; root_b.mkdir()
                (root_a / "payload.json").write_bytes(b"{}\n")
                (root_b / "payload.json").write_bytes(b"{}\n")
                snapshot = module.HeldEngineeringEvidence(root_a, "test evidence")
                try:
                    snapshot.require_selected_root(root_a)
                    with self.assertRaises(module.IntegrityError):
                        snapshot.require_selected_root(root_b)
                    (root_a / "extra.json").write_bytes(b"{}\n")
                    with self.assertRaises(module.IntegrityError):
                        snapshot.revalidate()
                finally:
                    snapshot.close()
            with self.subTest(role=module.__name__, mutation="replacement"), \
                    tempfile.TemporaryDirectory() as d:
                base = Path(d)
                selected = base / "selected"; selected.mkdir()
                (selected / "payload.json").write_bytes(b"{}\n")
                snapshot = module.HeldEngineeringEvidence(selected, "test evidence")
                held = base / "held"
                selected.rename(held); selected.mkdir()
                (selected / "payload.json").write_bytes(b"{}\n")
                try:
                    with self.assertRaises(module.IntegrityError):
                        snapshot.require_selected_root(selected)
                finally:
                    snapshot.close()

    def test_all_official_entrypoints_reject_missing_or_tampered_final_authority_before_numerics(
        self,
    ) -> None:
        common = [
            "--contract", str(ROOT / "contract_v1.json"),
            "--seed-manifest", str(ROOT / "seed_manifest_v1.json"),
            "--initial-states", str(ROOT / "initial_states_v1.json"),
        ]

        def bomb(*_args, **_kwargs):
            raise AssertionError("numerical or decoder entrypoint was reached")

        with tempfile.TemporaryDirectory() as d:
            temp = Path(d)
            (temp / "verification").mkdir()
            tampered = temp / "registration_v1.json"
            tampered.write_text("{}\n", encoding="utf-8")
            for label, registration in (
                ("missing", ROOT / "registration_v1.json"),
                ("tampered", tampered),
            ):
                opened: list[int] = []

                def fresh_guard(*_args, **_kwargs) -> int:
                    descriptor = os.open(os.devnull, os.O_RDONLY)
                    opened.append(descriptor)
                    return descriptor

                primary_argv = [
                    "run_primary.py", *common, "--registration", str(registration),
                    "--validate-only",
                ]
                with self.subTest(role="primary", authority=label), \
                        mock.patch.object(sys, "argv", primary_argv), \
                        mock.patch.object(primary, "acquire_v2_b_guard", fresh_guard), \
                        mock.patch.object(primary, "acquire_v3_a_guard", fresh_guard), \
                        mock.patch.object(
                            primary, "acquire_engineering_evidence_guard", fresh_guard,
                        ), \
                        mock.patch.object(primary, "get_rebound", bomb), \
                        mock.patch.object(primary, "validate_runtime", bomb), \
                        mock.patch.object(primary, "run_supervisor", bomb):
                    with self.assertRaises((OSError, primary.IntegrityError)):
                        primary.main()

                independent_argv = [
                    *common,
                    "--selection-manifest", str(ROOT / "selection_manifest_v1.json"),
                    "--registration", str(registration), "--validate-only",
                ]
                with self.subTest(role="DOP853", authority=label), \
                        mock.patch.object(independent, "acquire_v2_b_guard", fresh_guard), \
                        mock.patch.object(independent, "acquire_v3_a_guard", fresh_guard), \
                        mock.patch.object(
                            independent, "acquire_engineering_evidence_guard", fresh_guard,
                        ), \
                        mock.patch.object(independent, "scipy_runtime", bomb), \
                        mock.patch.object(independent, "integrate_segment", bomb), \
                        mock.patch.object(independent, "execute", bomb):
                    with self.assertRaises((OSError, independent.IntegrityError)):
                        independent.main(independent_argv)

                verifier_argv = [
                    "verify_replay.py", *common,
                    "--selection-manifest", str(ROOT / "selection_manifest_v1.json"),
                    "--registration", str(registration),
                    "--output-a", str(temp / "output-a"), "--verify-a",
                    "--receipt", str(temp / "verification" / "a.json"),
                ]
                verifier_fds: list[int] = []

                def verifier_guard(*_args, **_kwargs) -> int:
                    descriptor = os.open(os.devnull, os.O_RDONLY)
                    verifier_fds.append(descriptor)
                    return descriptor

                try:
                    with self.subTest(role="verifier", authority=label), \
                            mock.patch.object(sys, "argv", verifier_argv), \
                            mock.patch.object(
                                verifier, "verify_unlocked_execution_lock", verifier_guard,
                            ), \
                            mock.patch.object(verifier, "reconstruct_states", bomb), \
                            mock.patch.object(verifier, "verify_primary_output", bomb), \
                            mock.patch.object(
                                verifier, "decoded_primary_continuation_projection", bomb,
                            ):
                        with self.assertRaises((OSError, verifier.VerificationError)):
                            verifier.main()
                finally:
                    verifier._ENGINEERING_RUNNER_GUARD_FD = None
                    verifier._ENGINEERING_SCRATCH_GUARD_FD = None
                    for descriptor in verifier_fds:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                    for descriptor in opened:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass

    def test_verification_receipt_destination_and_publication_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parent = root / "receipts"; parent.mkdir()
            receipt = parent / "receipt.json"
            self.assertEqual(
                verifier.validate_receipt_destination(receipt, "test receipt"),
                receipt.absolute(),
            )
            payload = {"schema": "test/v1", "status": "PASS"}
            verifier.publish(receipt, payload)
            before = (receipt.stat().st_dev, receipt.stat().st_ino, receipt.read_bytes())
            verifier.publish(receipt, payload)
            after = (receipt.stat().st_dev, receipt.stat().st_ino, receipt.read_bytes())
            self.assertEqual(after, before)
            with self.assertRaises(verifier.VerificationError):
                verifier.validate_receipt_destination(
                    root / "missing-parent" / "receipt.json", "test receipt",
                )
            real_parent = root / "real-parent"; real_parent.mkdir()
            symlink_parent = root / "symlink-parent"
            symlink_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(verifier.VerificationError):
                verifier.validate_receipt_destination(
                    symlink_parent / "receipt.json", "test receipt",
                )
            symlink_leaf = parent / "symlink.json"
            symlink_leaf.symlink_to(receipt)
            with self.assertRaises(verifier.VerificationError):
                verifier.validate_receipt_destination(symlink_leaf, "test receipt")

    def test_verification_receipt_publication_never_clobbers_a_raced_final(self) -> None:
        value = {"schema": "test/v1", "status": "PASS"}
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        for mode in ("fresh", "complete-pending"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                parent = Path(d)
                final = parent / "receipt.json"
                pending = parent / ".receipt.json.pending"
                if mode == "complete-pending":
                    pending.write_bytes(payload)
                real_link = os.link

                def race_link(source, destination, **kwargs):
                    directory_fd = kwargs["dst_dir_fd"]
                    descriptor = os.open(
                        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600, dir_fd=directory_fd,
                    )
                    try:
                        os.write(descriptor, b"DIVERGENT\n")
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    return real_link(source, destination, **kwargs)

                with mock.patch.object(verifier.os, "link", side_effect=race_link):
                    with self.assertRaises(verifier.VerificationError):
                        verifier.publish(final, value)
                self.assertEqual(final.read_bytes(), b"DIVERGENT\n")
                self.assertEqual(pending.read_bytes(), payload)

    def test_verification_receipt_recovers_link_before_unlink_crash_cut(self) -> None:
        value = {"schema": "test/v1", "status": "PASS"}
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            final = parent / "receipt.json"
            pending = parent / ".receipt.json.pending"
            pending.write_bytes(payload)
            os.link(pending, final)
            self.assertEqual(final.stat().st_nlink, 2)
            verifier.publish(final, value)
            self.assertFalse(pending.exists())
            self.assertEqual(final.read_bytes(), payload)
            self.assertEqual(final.stat().st_nlink, 1)

    def test_final_receipt_uses_a_distinct_parent_from_the_a_receipt(self) -> None:
        contract = json.loads((ROOT / "contract_v1.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = [
                root / "output-a", root / "output-b", root / "dop-output",
                root / "a-receipts" / "a.json",
            ]
            with self.assertRaises(verifier.VerificationError):
                verifier.validate_external_paths(
                    inputs, contract, ROOT, root / "a-receipts" / "final.json",
                )
            verifier.validate_external_paths(
                inputs, contract, ROOT, root / "final-receipts" / "final.json",
            )
    def test_registration_stage_is_ordered_and_exact_when_present(self) -> None:
        engineering_path = ROOT / "engineering_registration_v1.json"
        final_path = ROOT / "registration_v1.json"
        self.assertFalse(final_path.exists() and not engineering_path.exists())
        if engineering_path.exists():
            engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
            self.assertEqual(
                engineering.get("schema"), primary.ENGINEERING_REGISTRATION_SCHEMA,
            )
            self.assertEqual(engineering.get("experiment_id"), primary.EXPERIMENT_ID)
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(final.get("schema"), primary.REGISTRATION_SCHEMA)
            self.assertEqual(final.get("experiment_id"), primary.EXPERIMENT_ID)
            self.assertIn("engineering_boundary_authorization", final)

    def test_save_boundary_order_is_pending_decode_compare_then_publish(self) -> None:
        source = (ROOT / "run_primary.py").read_text()
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "save_simulation_checkpoint"
        )
        segment = ast.get_source_segment(source, function)
        self.assertLess(segment.index("save_to_file"), segment.index("Simulation("))
        self.assertLess(segment.index("Simulation("), segment.index("os.replace"))
        self.assertLess(segment.index("live_archive_endpoint_projection"),
                        segment.index("os.replace"))
        self.assertIn("validate_decoded_continuation_settings", segment)

    def test_normalized_endpoint_keeps_logical_dcrit_and_drops_only_declared_caches(self) -> None:
        projection = {
            "schema": "jx-xp2-mercurius-decoded-continuation-state/v3",
            "simulation": {
                "N": 133, "particle_capacity_covers_logical_count": True,
                "active_memory_ranges_pairwise_disjoint": True,
            },
            "mercurius": {
                "dcrit_storage_present": True,
                "dcrit_capacity_covers_logical_count": True,
                "dcrit_hex": ["0x0.0p+0"] * 133,
                "encounter_N": 1, "encounter_N_active": 1, "tponly_encounter": 0,
                "allocated_particle_backup_count": 256,
                "allocated_additional_forces_backup_count": 0,
                "particles_backup_present": True,
                "additional_forces_backup_present": False,
                "encounter_map_present": True,
            },
            "whfast": {"coordinates": "democraticheliocentric",
                       "internal_particle_arrays_present": True},
            "ias15": {"epsilon_hex": "0x1.0p-32", "min_dt_hex": "0x0.0p+0",
                      "adaptive_mode": "global", "iterations_max_exceeded": 0,
                      "stored_coordinate_count": 256, "direct_array_sha256": {},
                      "coefficient_array_sha256": {}, "map_count": 0,
                      "map_sha256": None},
            "particles": [{"index": index} for index in range(133)],
            "excluded_noncontinuation_fields": [],
        }
        with mock.patch.object(
            primary, "decoded_continuation_projection", return_value=projection,
        ):
            endpoint = primary.live_archive_endpoint_projection(object())
        self.assertEqual(len(endpoint["mercurius"]["dcrit_hex"]), 133)
        self.assertTrue(endpoint["mercurius"]["dcrit_capacity_covers_logical_count"])
        self.assertNotIn("allocated_particle_backup_count", endpoint["mercurius"])
        self.assertNotIn("internal_particle_arrays_present", endpoint["whfast"])
        self.assertEqual(set(endpoint["ias15"]), {
            "epsilon_hex", "min_dt_hex", "adaptive_mode", "iterations_max_exceeded",
        })

    def test_engineering_harness_has_exact_registered_dynamics_sites(self) -> None:
        runner_tree = ast.parse((ROOT / "run_engineering_boundary.py").read_text())
        verifier_tree = ast.parse((ROOT / "verify_engineering_boundary.py").read_text())
        runner_integrates = [node for node in ast.walk(runner_tree)
                             if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute)
                             and node.func.attr == "integrate"]
        verifier_integrates = [node for node in ast.walk(verifier_tree)
                               if isinstance(node, ast.Call)
                               and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "integrate"]
        self.assertEqual(len(runner_integrates), 4)
        self.assertEqual(len(verifier_integrates), 1)


if __name__ == "__main__":
    unittest.main()
