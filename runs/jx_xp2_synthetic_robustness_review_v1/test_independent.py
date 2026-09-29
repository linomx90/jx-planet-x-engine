#!/usr/bin/env python3
"""No-long-dynamics tests for the frozen JX-XP2 DOP853 sentinel."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
import tempfile
import time
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
RUNNER_PATH = ROOT / "run_independent.py"
SPEC = importlib.util.spec_from_file_location("jx_xp2_run_independent_tests", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load independent runner")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
VERIFIER_PATH = ROOT / "verify_replay.py"
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "jx_xp2_verify_replay_independent_tests", VERIFIER_PATH
)
if VERIFY_SPEC is None or VERIFY_SPEC.loader is None:
    raise RuntimeError("cannot load independent verifier")
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)


def load_registered_design() -> tuple[dict, dict, tuple[str, ...]]:
    contract = runner.strict_json(ROOT / "contract_v1.json")
    initial = runner.strict_json(ROOT / "initial_states_v1.json")
    selection = runner.strict_json(ROOT / "selection_manifest_v1.json")
    _full, all_ids = runner.validate_and_expand_initial_states(
        initial, ROOT / "initial_states_v1.json", contract, None
    )
    selected = runner.validate_selection(selection, all_ids)
    arms, repeated = runner.validate_and_expand_initial_states(
        initial, ROOT / "initial_states_v1.json", contract, selected
    )
    if repeated != all_ids:
        raise AssertionError("initial-state validation changed tracer identity")
    return contract, arms, selected


class IndependentStaticAndDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = runner.strict_json(ROOT / "contract_v1.json")
        cls.initial = runner.strict_json(ROOT / "initial_states_v1.json")
        cls.selection = runner.strict_json(ROOT / "selection_manifest_v1.json")
        cls.all_arms, cls.all_ids = runner.validate_and_expand_initial_states(
            cls.initial, ROOT / "initial_states_v1.json", cls.contract, None
        )
        cls.selected_ids = runner.validate_selection(cls.selection, cls.all_ids)
        cls.arms, repeated = runner.validate_and_expand_initial_states(
            cls.initial, ROOT / "initial_states_v1.json", cls.contract, cls.selected_ids
        )
        if repeated != cls.all_ids:
            raise AssertionError("initial-state validation changed tracer identity")

    def test_source_has_no_rebound_import_or_call(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name == "rebound" or name.startswith("rebound.") for name in imported))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "rebound"
            for call in calls
        ))

    def test_contract_and_runtime_locks(self) -> None:
        runner.validate_contract(self.contract)
        try:
            runtime = runner.validate_runtime(self.contract)
        except runner.IntegrityError:
            self.skipTest("this shell is not the frozen registered Python/NumPy/SciPy runtime")
        self.assertEqual(runtime["scipy_version"], "1.17.0")
        self.assertEqual(runtime["numpy_version"], "2.3.5")
        self.assertNotIn("rebound", __import__("sys").modules)

    def test_registered_initial_state_artifact_and_all_digests(self) -> None:
        self.assertEqual(len(self.all_ids), 128)
        self.assertEqual(set(self.all_arms), set(runner.ARM_IDS))
        self.assertEqual(len(self.all_arms["M0"]["logical_ids"]), 133)
        self.assertEqual(len(self.all_arms["CI01-P0"]["logical_ids"]), 134)
        self.assertEqual(
            self.all_arms["M0"]["registered_expanded_initial_state_sha256"],
            self.initial["configuration_states"][0][5],
        )
        self.assertEqual(
            runner.sha256_file(ROOT / "initial_states_v1.json"),
            self.contract["initial_state_policy"]["artifact_sha256"],
        )

    def test_selection_is_exact_logical_id_hash_rule(self) -> None:
        self.assertEqual(len(self.selected_ids), 32)
        self.assertEqual(
            self.selected_ids[:4],
            ("XP2-B00-T02", "XP2-B00-T11", "XP2-B00-T10", "XP2-B00-T07"),
        )
        for block in range(8):
            self.assertEqual(sum(item.startswith(f"XP2-B{block:02d}-T") for item in self.selected_ids), 4)

    def test_selection_rejects_one_substituted_id(self) -> None:
        changed = copy.deepcopy(self.selection)
        changed["sentinels_by_block"]["0"]["ordered_logical_ids"][0] = "XP2-B00-T00"
        with self.assertRaises(runner.IntegrityError):
            runner.validate_selection(changed, self.all_ids)

    def test_selected_arm_shapes_and_order(self) -> None:
        for arm_id, arm in self.arms.items():
            self.assertEqual(arm["logical_ids"][arm["active_count"]:], list(self.selected_ids))
            self.assertEqual(len(arm["masses"]), arm["active_count"] + 32)
            self.assertEqual(len(arm["initial_state"]), 6 * (arm["active_count"] + 32))
            self.assertTrue(runner.is_lower_hex(arm["initial_state_sha256"]))
            self.assertEqual(arm["arm_id"], arm_id)

    def test_packed_binary64_round_trip_and_rejections(self) -> None:
        for value in (-1.25, -0.0, 0.0, math.pi, 1e300):
            packed = runner.pack_binary64_be(value)
            decoded = runner.unpack_binary64_be(packed, "test")
            self.assertEqual(runner.pack_binary64_be(decoded), packed)
        with self.assertRaises(runner.IntegrityError):
            runner.unpack_binary64_be("7ff0000000000000", "infinity")

    def test_strict_json_rejects_duplicate_underflow_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, payload in enumerate((
                '{"a":1,"a":2}', '{"a":1e-9999}', '{"a":NaN}'
            )):
                path = root / f"bad{index}.json"
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.strict_json(path)


class IndependentNumericsTests(unittest.TestCase):
    def test_rhs_two_active_bodies_obeys_force_balance(self) -> None:
        numpy, _solve_ivp = runner.scipy_runtime()
        masses = [1.0, 0.5]
        rhs = runner.newtonian_rhs_factory(masses, 2, 1.0)
        state = numpy.asarray([
            -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        acceleration = rhs(0.0, state).reshape(2, 2, 3)[1]
        self.assertTrue(numpy.allclose(acceleration[0], [0.125, 0.0, 0.0]))
        self.assertTrue(numpy.allclose(acceleration[1], [-0.25, 0.0, 0.0]))
        self.assertTrue(numpy.allclose(masses[0] * acceleration[0] + masses[1] * acceleration[1], 0.0))

    def test_massless_tracer_has_no_backreaction(self) -> None:
        numpy, _solve_ivp = runner.scipy_runtime()
        active_state = numpy.asarray([
            -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        active_rhs = runner.newtonian_rhs_factory([1.0, 0.5], 2, 1.0)
        active_acceleration = active_rhs(0.0, active_state).reshape(2, 2, 3)[1]
        full_state = numpy.asarray([
            -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 50.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, -1.0, 3.0,
        ])
        full_rhs = runner.newtonian_rhs_factory([1.0, 0.5, 0.0], 2, 1.0)
        full_acceleration = full_rhs(0.0, full_state).reshape(2, 3, 3)[1]
        self.assertTrue(numpy.array_equal(active_acceleration, full_acceleration[:2]))

    def test_orbital_metrics_circular_and_hyperbolic(self) -> None:
        circular = runner.orbital_metrics(
            [1.0, 0.0, 0.0], [0.0, 2.0 * math.pi, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 4.0 * math.pi * math.pi, 1.0,
        )
        self.assertAlmostEqual(circular["q_AU"], 1.0, places=14)
        self.assertAlmostEqual(circular["i_deg"], 0.0, places=14)
        self.assertTrue(circular["bound"])
        hyperbolic = runner.orbital_metrics(
            [1.0, 0.0, 0.0], [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 4.0 * math.pi * math.pi, 1.0,
        )
        self.assertLess(hyperbolic["a_AU"], 0.0)
        self.assertFalse(hyperbolic["bound"])
        self.assertTrue(math.isfinite(hyperbolic["q_AU"]))

    def test_exact_parabolic_semimajor_axis_fails_finite_gate(self) -> None:
        escape = math.sqrt(8.0 * math.pi * math.pi)
        with self.assertRaises(runner.NumericalError):
            runner.orbital_metrics(
                [1.0, 0.0, 0.0], [0.0, escape, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 4.0 * math.pi * math.pi, 1.0,
            )

    def test_invariants_are_translation_and_boost_intrinsic(self) -> None:
        positions = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        velocities = [[0.0, -0.25, 0.0], [0.0, 0.5, 0.0]]
        masses = [2.0, 1.0]
        baseline = runner.active_invariants(positions, velocities, masses, 1.0)
        shifted_positions = [[x + 17.0, y - 8.0, z + 3.0] for x, y, z in positions]
        shifted_velocities = [[x + 4.0, y - 2.0, z + 1.0] for x, y, z in velocities]
        shifted = runner.active_invariants(shifted_positions, shifted_velocities, masses, 1.0)
        self.assertAlmostEqual(baseline["intrinsic_energy"], shifted["intrinsic_energy"], places=14)
        for left, right in zip(baseline["com_angular_momentum"], shifted["com_angular_momentum"], strict=True):
            self.assertAlmostEqual(left, right, places=14)

    def test_state_and_segment_chains_are_deterministic_and_sensitive(self) -> None:
        first = runner.state_sample_chain(runner.INITIAL_CHAIN, 0.0, [1.0, 2.0])
        self.assertEqual(first, runner.state_sample_chain(runner.INITIAL_CHAIN, 0.0, [1.0, 2.0]))
        self.assertNotEqual(first, runner.state_sample_chain(runner.INITIAL_CHAIN, 0.0, [1.0, 2.1]))
        payload = {
            "segment_index": 0, "start_year": 0.0, "end_year": 50_000.0,
            "end_state_hex": [1.0.hex()], "accumulator": {"sample_count": 1001},
        }
        commitment = runner.segment_commitment("M0", payload, runner.INITIAL_CHAIN)
        changed = copy.deepcopy(payload)
        changed["end_state_hex"] = [2.0.hex()]
        self.assertNotEqual(
            commitment["chain_head_sha256"],
            runner.segment_commitment("M0", changed, runner.INITIAL_CHAIN)["chain_head_sha256"],
        )

    def test_threshold_first_passages_require_chronological_nesting(self) -> None:
        with self.assertRaises(runner.IntegrityError):
            runner._validate_first_passages(
                {"30": 150.0, "35": 50.0, "40": 100.0},
                20.0, 1_000.0, "malicious",
            )

    def test_short_capability_integration_only(self) -> None:
        result = runner.capability_test()
        self.assertEqual(result["capability"], "NOT_RUN_REGISTRATION_REQUIRED")
        self.assertFalse(result["dynamics_executed"])
        self.assertEqual(result["duration_years"], 0.0)
        self.assertFalse(result["rebound_loaded"])

    @unittest.skip("requires the final registered official DOP853 authority")
    def test_one_year_real_state_segment_capability(self) -> None:
        loaded_contract, arms, _selected = load_registered_design()
        contract = copy.deepcopy(loaded_contract)
        contract["design_core"]["dynamics"]["segment_years"] = 1.0
        contract["design_core"]["dynamics"]["sample_cadence_years"] = 0.5
        arm = arms["M0"]
        accumulator = runner.initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        payload = runner.integrate_segment(
            arm, arm["initial_state"], accumulator, 0, contract
        )
        self.assertEqual(payload["end_year"], 1.0)
        self.assertEqual(payload["accumulator"]["sample_count"], 3)
        self.assertEqual(len(payload["end_state_hex"]), len(arm["initial_state"]))
        self.assertEqual(payload["accumulator"]["landmarks"], {})


class IndependentSafetyTests(unittest.TestCase):
    def test_segment_transition_rejects_historical_minimum_reset(self) -> None:
        contract, arms, _selected = load_registered_design()
        arm = arms["M0"]
        previous = runner.initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        current = copy.deepcopy(previous)
        current["sample_count"] += 1000
        current["sample_state_chain_head"] = runner.state_sample_chain(
            current["sample_state_chain_head"], 50_000.0, arm["initial_state"]
        )
        victim = next(iter(current["particles"]))
        current["particles"][victim]["minimum_sampled_q_AU"] += 1.0
        with self.assertRaises(runner.IntegrityError):
            runner.validate_accumulator_transition(previous, current, arm, 0)

    def test_malformed_next_payload_cannot_replace_good_checkpoint(self) -> None:
        contract, arms, _selected = load_registered_design()
        arm = arms["M0"]
        gravitational_constant = contract["design_core"]["units_and_frame"][
            "G_AU3_Msun_yr2"
        ]
        accumulator = runner.initialize_accumulator(arm, gravitational_constant)
        accumulator["sample_count"] = 1001
        accumulator["sample_state_chain_head"] = runner.state_sample_chain(
            accumulator["sample_state_chain_head"], 50_000.0, arm["initial_state"]
        )
        payload = {
            "segment_index": 0,
            "start_year": 0.0,
            "end_year": 50_000.0,
            "end_state_hex": runner.state_to_hex(arm["initial_state"]),
            "accumulator": accumulator,
        }
        bindings = {"test": "0" * 64}
        contract = copy.deepcopy(contract)
        contract["resource_caps_per_execution"]["minimum_free_disk_bytes"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "receipts").mkdir()
            started = time.monotonic_ns()
            good = runner.commit_segment_payload(
                arm, payload, [], bindings, contract, root,
                0.01, 0, started, started + 2_000_000_000,
            )
            path = runner.checkpoint_path(root, "M0", 0)
            before = path.read_bytes()
            malformed = copy.deepcopy(payload)
            malformed["segment_index"] = 1
            malformed["start_year"] = 50_000.0
            malformed["end_year"] = 100_000.0
            malformed["accumulator"]["sample_count"] = 2001
            victim = next(iter(malformed["accumulator"]["particles"]))
            del malformed["accumulator"]["particles"][victim]["current"]["q_AU"]
            now = time.monotonic_ns()
            framed_payload, _rss = runner.supervise_worker(
                lambda: malformed, 2.0, 512 * 1024 * 1024, root,
                1024 * 1024, now + 2_000_000_000, 0.01,
            )
            with self.assertRaises(runner.IntegrityError):
                runner.commit_segment_payload(
                    arm, framed_payload, good["segment_commitments"], bindings,
                    contract, root, 0.01, 0, now, now + 2_000_000_000,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_retry_accounting_is_keyed_to_exact_segment(self) -> None:
        ledger = [
            {"event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0", "segment_index": 0},
            {"event": "SEGMENT_ATTEMPT_FAILED", "arm_id": "M0", "segment_index": 0},
            {"event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0", "segment_index": 0},
            {"event": "SEGMENT_ATTEMPT_COMMITTED", "arm_id": "M0", "segment_index": 0},
            {"event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0", "segment_index": 1},
            {"event": "SEGMENT_ATTEMPT_FAILED", "arm_id": "M0", "segment_index": 1},
        ]
        self.assertEqual(runner.next_attempt_number(ledger, "M0", 0), 3)
        self.assertEqual(runner.next_attempt_number(ledger, "M0", 1), 2)
        self.assertEqual(runner.next_attempt_number(ledger, "CI01-P0", 0), 1)

    def test_attempt_ledger_rejects_skip_overlap_and_noncontiguous_retry(self) -> None:
        start = {
            "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
            "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
            "segment_index": 0, "attempt_number_for_segment": 1,
        }
        failed = {
            "schema": runner.ATTEMPT_SCHEMA, "sequence": 2,
            "event": "SEGMENT_ATTEMPT_FAILED", "arm_id": "M0",
            "segment_index": 0, "attempt_number_for_segment": 1,
            "failure_class": "NumericalError",
        }
        with self.assertRaises(runner.IntegrityError):
            runner.replay_attempt_ledger([start, copy.deepcopy(start)])
        skipped = copy.deepcopy(start)
        skipped["segment_index"] = 1
        with self.assertRaises(runner.IntegrityError):
            runner.replay_attempt_ledger([skipped])
        retry = copy.deepcopy(start)
        retry["sequence"] = 3
        retry["attempt_number_for_segment"] = 3
        with self.assertRaises(runner.IntegrityError):
            runner.replay_attempt_ledger([start, failed, retry])

    def test_interrupted_postpublication_attempt_reconciles_to_checkpoint_hash(self) -> None:
        contract, arms, _selected = load_registered_design()
        arm = arms["M0"]
        accumulator = runner.initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        accumulator["sample_count"] = 1001
        accumulator["sample_state_chain_head"] = runner.state_sample_chain(
            accumulator["sample_state_chain_head"], 50_000.0, arm["initial_state"]
        )
        payload = {
            "segment_index": 0, "start_year": 0.0, "end_year": 50_000.0,
            "end_state_hex": runner.state_to_hex(arm["initial_state"]),
            "accumulator": accumulator,
        }
        bindings = {"test": "0" * 64}
        contract = copy.deepcopy(contract)
        contract["resource_caps_per_execution"]["minimum_free_disk_bytes"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "receipts").mkdir()
            (root / "failures").mkdir()
            started = time.monotonic_ns()
            runner.commit_segment_payload(
                arm, payload, [], bindings, contract, root,
                0.01, 0, started, started + 2_000_000_000,
            )
            runner.append_attempt_ledger(root, {
                "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                "segment_index": 0, "attempt_number_for_segment": 1,
            })
            rows = runner.reconcile_attempt_ledger(root, arms, bindings, contract)
            self.assertEqual(rows[-1]["event"], "SEGMENT_ATTEMPT_COMMITTED")
            self.assertEqual(
                rows[-1]["recovery"],
                "COORDINATOR_INTERRUPTED_AFTER_CHECKPOINT_PUBLICATION",
            )
            self.assertEqual(
                rows[-1]["checkpoint_sha256"],
                runner.sha256_file(runner.checkpoint_path(root, "M0", 0)),
            )
            self.assertEqual(
                rows[-1]["segment_receipt_sha256"],
                runner.sha256_file(runner.segment_receipt_path(root, "M0", 0)),
            )
            pending, advanced = runner.validate_ledger_against_checkpoints(
                rows, root, arms, bindings, contract
            )
            self.assertIsNone(pending)
            self.assertFalse(advanced)

    def test_terminal_resource_failure_cannot_publish_eligible_pair(self) -> None:
        contract, arms, _selected = load_registered_design()
        arm = arms["M0"]
        accumulator = runner.initialize_accumulator(
            arm, contract["design_core"]["units_and_frame"]["G_AU3_Msun_yr2"]
        )
        accumulator["sample_count"] = 1001
        accumulator["sample_state_chain_head"] = runner.state_sample_chain(
            accumulator["sample_state_chain_head"], 50_000.0, arm["initial_state"]
        )
        payload = {
            "segment_index": 0, "start_year": 0.0, "end_year": 50_000.0,
            "end_state_hex": runner.state_to_hex(arm["initial_state"]),
            "accumulator": accumulator,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "checkpoints").mkdir(); (root / "receipts").mkdir()
            started = time.monotonic_ns()
            with self.assertRaises(runner.ResourceLimitError):
                runner.commit_segment_payload(
                    arm, payload, [], {"test": "0" * 64}, contract, root,
                    0.01,
                    contract["resource_caps_per_execution"]["max_peak_rss_bytes_per_process"] + 1,
                    started, started + 2_000_000_000,
                )
            self.assertEqual(list((root / "checkpoints").iterdir()), [])
            self.assertEqual(list((root / "receipts").iterdir()), [])

    def test_failed_supervised_segment_cannot_publish_checkpoint(self) -> None:
        def fail_before_response() -> dict[str, int]:
            raise RuntimeError("synthetic crash")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = time.monotonic_ns()
            with self.assertRaises(runner.NumericalError):
                runner.supervise_worker(
                    fail_before_response, 2.0, 512 * 1024 * 1024, root,
                    1024 * 1024, now + 2_000_000_000, 0.01,
                )
            self.assertFalse((root / "checkpoints").exists())

    def test_atomic_create_refuses_overwrite_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "value.json"
            runner.atomic_create_json(target, {"a": 1})
            with self.assertRaises(FileExistsError):
                runner.atomic_create_json(target, {"a": 2})
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                runner.atomic_create_json(link, {"a": 3})

    def test_failure_receipt_redacts_exception_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "failures").mkdir()
            secret = "/private/absolute/secret/input.json"
            start = {
                "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                "segment_index": 2, "attempt_number_for_segment": 1,
            }
            path, _receipt = runner.publish_failure_receipt(
                root, start, runner.failure_class_for_error(ValueError(secret))
            )
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn(secret, payload)
            self.assertIn(runner.REDACTED_FAILURE_MESSAGE, payload)

    def test_dop_failure_receipt_first_orphan_reconciles_exactly(self) -> None:
        contract, arms, _selected = load_registered_design()
        bindings = {"test": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoints", "receipts", "failures"):
                (root / name).mkdir()
            start = {
                "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                "segment_index": 0, "attempt_number_for_segment": 1,
            }
            runner.append_attempt_ledger(root, start)
            receipt_path, receipt = runner.publish_failure_receipt(
                root, start, "NumericalError"
            )
            rows = runner.reconcile_attempt_ledger(root, arms, bindings, contract)
            self.assertEqual(rows[-1]["event"], "SEGMENT_ATTEMPT_FAILED")
            self.assertEqual(rows[-1]["failure_receipt_filename"], receipt_path.name)
            self.assertEqual(rows[-1]["failure_receipt_sha256"], runner.sha256_file(receipt_path))
            self.assertEqual(rows[-1]["fail_event_sha256"], receipt["fail_event_sha256"])
            runner.validate_failure_receipt_bindings(rows, root)
            verifier.verify_dop_failures(root / "failures", rows)

    def test_dop_complete_and_partial_pending_failure_receipt_recover(self) -> None:
        contract, arms, _selected = load_registered_design()
        bindings = {"test": "0" * 64}
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("checkpoints", "receipts", "failures"):
                    (root / name).mkdir()
                start = {
                    "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                    "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                    "segment_index": 0, "attempt_number_for_segment": 1,
                }
                runner.append_attempt_ledger(root, start)
                filename = runner.failure_receipt_filename("M0", 0, 1)
                pending = root / "failures" / f".{filename}.pending"
                expected = runner.failure_receipt_payload(start, "NumericalError")
                pending.write_bytes(b"{" if partial else runner.serialized_json(expected))
                rows = runner.reconcile_attempt_ledger(root, arms, bindings, contract)
                self.assertFalse(pending.exists())
                self.assertEqual(
                    rows[-1]["failure_class"],
                    "InterruptedAttempt" if partial else "NumericalError",
                )
                runner.validate_failure_receipt_bindings(rows, root)
                verifier.verify_dop_failures(root / "failures", rows)

    def test_dop_missing_extra_and_tampered_failure_receipts_rejected(self) -> None:
        contract, arms, _selected = load_registered_design()
        bindings = {"test": "0" * 64}
        for mutation in ("missing", "extra", "tampered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("checkpoints", "receipts", "failures"):
                    (root / name).mkdir()
                start = {
                    "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                    "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                    "segment_index": 0, "attempt_number_for_segment": 1,
                }
                runner.append_attempt_ledger(root, start)
                path, _receipt = runner.publish_failure_receipt(root, start, "NumericalError")
                rows = runner.reconcile_attempt_ledger(root, arms, bindings, contract)
                if mutation == "missing":
                    path.unlink()
                elif mutation == "extra":
                    (root / "failures" / "failure_M0_segment_01_attempt_01.json").write_bytes(b"{}\n")
                else:
                    changed = runner.strict_json(path)
                    changed["failure_message"] = "MUTATED"
                    path.write_bytes(runner.serialized_json(changed))
                with self.assertRaises(runner.IntegrityError):
                    runner.validate_failure_receipt_bindings(rows, root)
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_dop_failures(root / "failures", rows)

    def test_dop_atomic_ledger_pending_crash_cuts(self) -> None:
        cases = (
            "mid_old", "exact_old", "mid_new", "complete", "divergent",
            "wrong_sequence", "wrong_event",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                start = {
                    "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                    "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                    "segment_index": 0, "attempt_number_for_segment": 1,
                }
                runner.append_attempt_ledger(root, start)
                old = (root / "attempt_ledger.jsonl").read_bytes()
                receipt = runner.failure_receipt_payload(start, "NumericalError")
                terminal = {
                    "schema": runner.ATTEMPT_SCHEMA, "sequence": 2,
                    "event": "SEGMENT_ATTEMPT_FAILED", "arm_id": "M0",
                    "segment_index": 0, "attempt_number_for_segment": 1,
                    "failure_class": "NumericalError",
                    "fail_event_sha256": receipt["fail_event_sha256"],
                    "failure_receipt_filename": runner.failure_receipt_filename("M0", 0, 1),
                    "failure_receipt_sha256": "0" * 64,
                }
                extension = old + runner.canonical_bytes(terminal) + b"\n"
                pending = root / ".attempt_ledger.jsonl.pending"
                if case == "mid_old":
                    candidate = old[:len(old) // 2]
                elif case == "exact_old":
                    candidate = old
                elif case == "mid_new":
                    candidate = old + (runner.canonical_bytes(terminal) + b"\n")[:19]
                elif case == "complete":
                    candidate = extension
                elif case == "wrong_sequence":
                    wrong = dict(terminal, sequence=3)
                    candidate = old + runner.canonical_bytes(wrong) + b"\n"
                elif case == "wrong_event":
                    wrong = dict(terminal, event="UNKNOWN")
                    candidate = old + runner.canonical_bytes(wrong) + b"\n"
                else:
                    candidate = b"[" + old[1:]
                pending.write_bytes(candidate)
                if case in {"divergent", "wrong_sequence", "wrong_event"}:
                    with self.assertRaises(runner.IntegrityError):
                        runner.recover_pending_attempt_ledger(root)
                    continue
                runner.recover_pending_attempt_ledger(root)
                self.assertFalse(pending.exists())
                self.assertEqual(
                    (root / "attempt_ledger.jsonl").read_bytes(),
                    extension if case == "complete" else old,
                )

    def test_dop_divergent_complete_pending_failure_receipt_rejects(self) -> None:
        contract, arms, _selected = load_registered_design()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoints", "receipts", "failures"):
                (root / name).mkdir()
            start = {
                "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                "segment_index": 0, "attempt_number_for_segment": 1,
            }
            runner.append_attempt_ledger(root, start)
            wrong = runner.failure_receipt_payload(start, "NumericalError")
            wrong["failure_message"] = "DIVERGENT"
            filename = runner.failure_receipt_filename("M0", 0, 1)
            (root / "failures" / f".{filename}.pending").write_bytes(
                runner.serialized_json(wrong)
            )
            with self.assertRaises(runner.IntegrityError):
                runner.reconcile_attempt_ledger(
                    root, arms, {"test": "0" * 64}, contract
                )

    def test_dop_ledger_requires_canonical_framing_and_exact_integer_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "schema": runner.ATTEMPT_SCHEMA, "sequence": 1,
                "event": "SEGMENT_ATTEMPT_STARTED", "arm_id": "M0",
                "segment_index": 0, "attempt_number_for_segment": 1,
            }
            path = root / "attempt_ledger.jsonl"
            path.write_bytes(runner.canonical_bytes(row))
            with self.assertRaises(runner.IntegrityError):
                runner.read_attempt_ledger(root)
            row["sequence"] = True
            path.write_bytes(runner.canonical_bytes(row) + b"\n")
            with self.assertRaises(runner.IntegrityError):
                runner.read_attempt_ledger(root)

    def test_supervisor_returns_small_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = time.monotonic_ns()
            payload, rss = runner.supervise_worker(
                lambda: {"answer": 42}, 2.0, 512 * 1024 * 1024, root,
                1024 * 1024, now + 2_000_000_000, 0.01,
            )
            self.assertEqual(payload, {"answer": 42})
            self.assertGreaterEqual(rss, 0)

    def test_supervisor_enforces_hard_wall_deadline(self) -> None:
        def spin() -> dict[str, int]:
            while True:
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = time.monotonic_ns()
            with self.assertRaises(runner.ResourceLimitError):
                runner.supervise_worker(
                    spin, 0.05, 512 * 1024 * 1024, root,
                    1024 * 1024, now + 2_000_000_000, 0.01,
                )

    def test_output_guard_rejects_package_and_xp1_trees(self) -> None:
        contract = runner.strict_json(ROOT / "contract_v1.json")
        with self.assertRaises(ValueError):
            runner.validate_output_root(ROOT / "nested", ROOT, contract, False)
        xp1 = (ROOT / "../jx_xp1_runs_v1").resolve()
        with self.assertRaises(ValueError):
            runner.validate_output_root(xp1 / "nested", ROOT, contract, False)

    def test_dop_lock_survives_coordinator_death_while_child_lives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"; root.mkdir()
            lock_fd = runner.acquire_output_execution_lock(root, create=True)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(write_fd); os.read(read_fd, 1); os.close(read_fd); os._exit(0)
            os.close(read_fd); os.close(lock_fd)
            with self.assertRaises(runner.ResourceLimitError):
                runner.acquire_output_execution_lock(root, create=False)
            os.write(write_fd, b"x"); os.close(write_fd); os.waitpid(pid, 0)
            recovered = runner.acquire_output_execution_lock(root, create=False)
            os.close(recovered)

    def test_parent_side_exception_kills_and_reaps_worker_before_lock_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base / "output"; guard = base / "guard"
            root.mkdir(); guard.mkdir()
            output_fd = runner.acquire_output_execution_lock(root, create=True)
            guard_fd = runner.acquire_output_execution_lock(guard, create=True)
            pid_path = root / "worker.pid"

            def worker() -> dict:
                pid_path.write_text(str(os.getpid()), encoding="ascii")
                while True:
                    time.sleep(0.01)

            original_directory_bytes = runner.directory_bytes

            def injected_parent_failure(_root: Path) -> int:
                deadline = time.monotonic() + 2.0
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.001)
                raise runner.IntegrityError("injected parent-side accounting failure")

            runner.directory_bytes = injected_parent_failure
            try:
                with self.assertRaises(runner.IntegrityError):
                    runner.supervise_worker(
                        worker, 5.0, 512 * 1024 * 1024, root,
                        1024 * 1024, time.monotonic_ns() + 5_000_000_000, 0.01,
                    )
            finally:
                runner.directory_bytes = original_directory_bytes
            child_pid = int(pid_path.read_text(encoding="ascii"))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            os.close(output_fd); os.close(guard_fd)
            recovered_output = runner.acquire_output_execution_lock(root, create=False)
            recovered_guard = runner.acquire_output_execution_lock(guard, create=False)
            os.close(recovered_output); os.close(recovered_guard)

    def test_dop_top_level_pending_manifest_and_result_recover(self) -> None:
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "output"; root.mkdir()
                expected = {"schema": "manifest", "execution_label": "DOP853-SENTINEL"}
                pending = root / ".run_manifest.json.pending"
                pending.write_bytes(b"{" if partial else runner.serialized_json(expected))
                runner.recover_or_validate_run_manifest(root, expected)
                self.assertEqual(runner.strict_json(root / "run_manifest.json"), expected)
                self.assertFalse(pending.exists())
                result_pending = root / ".result_v1.json.pending"
                result_pending.write_bytes(b"partial-unpublished-result")
                self.assertIsNone(runner.recover_unpublished_result_pending(root))
                self.assertFalse(result_pending.exists())

    def test_complete_dop_pending_result_is_semantically_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"; root.mkdir()
            (root / "checkpoints").mkdir(); (root / "receipts").mkdir()
            ledger = root / "attempt_ledger.jsonl"; ledger.write_bytes(b"ledger\n")
            core = {"schema": "result", "semantic": {"value": 1}}
            provenance = {
                "elapsed_seconds": 1.0, "coordinator_peak_rss_bytes": 1,
                "maximum_terminal_child_peak_rss_bytes": 1,
                "output_bytes_before_result": runner.directory_bytes(root),
                "attempt_ledger_sha256": runner.sha256_file(ledger),
                "immutable_segment_inventory_sha256":
                runner.immutable_segment_inventory_sha256(root),
            }
            candidate = {**core, "resource_provenance": provenance}
            pending = root / ".result_v1.json.pending"
            pending.write_bytes(runner.serialized_json(candidate))
            recovered = runner.recover_unpublished_result_pending(root)
            self.assertEqual(recovered, candidate)
            contract = {"resource_caps_per_execution": {
                "max_wall_seconds_total": 10.0,
                "max_peak_rss_bytes_per_process": 1024,
            }}
            runner.publish_or_recover_result(root, candidate, recovered, contract)
            self.assertEqual(runner.strict_json(root / "result_v1.json"), candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
