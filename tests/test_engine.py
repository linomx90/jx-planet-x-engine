import json
import tempfile
import unittest
from decimal import localcontext
from pathlib import Path

from jxplanetx.claims import Decision, LOCKED_OBSERVATION_GATES, assess_claim
from jxplanetx.decimal_math import D, precision_context, vec
from jxplanetx.decimal_bs import validate_bs_oscillator
from jxplanetx.dynamics import Body, State
from jxplanetx.gates import GateResult, convergence_gate, run_core_gates
from jxplanetx.ias15_gate import IAS15_GATES, POPULATION_GATES, SELECTED_TRACERS, SOURCE_EFFECT_GATES
from jxplanetx.provenance import (
    package_source_manifest,
    runtime_source_manifest,
    sha256_data,
    source_manifest,
    write_run_record,
)
from jxplanetx.production_benchmark import EXPECTED_CHECKSUM_MANIFEST_SHA256, verify_bundle
from jxplanetx.yoshida6 import StepStats, step


class NumericalTests(unittest.TestCase):
    def test_yoshida6_convergence(self):
        gate = convergence_gate(80)
        self.assertTrue(gate.passed, gate)

    def test_eight_force_evaluations(self):
        with localcontext(precision_context(50)):
            state = State(
                (Body("Sun", D(1)), Body("test", D(0), False)),
                [vec((0, 0, 0)), vec((1, 0, 0))],
                [vec((0, 0, 0)), vec((0, 1, 0))],
            )
            stats = StepStats()
            step(state, D("0.01"), stats)
            self.assertEqual(stats.force_evaluations, 8)

    def test_all_core_gates_pass(self):
        failures = [g for g in run_core_gates(80) if not g.passed]
        self.assertEqual(failures, [])

    def test_independent_bs_oscillator(self):
        result = validate_bs_oscillator(60)
        self.assertTrue(result["passed"], result)

    def test_ias15_stratification_and_locked_gates(self):
        self.assertEqual(len(SELECTED_TRACERS), 15)
        self.assertEqual(len({int(name[1:]) // 9 for name in SELECTED_TRACERS}), 5)
        self.assertEqual(IAS15_GATES["bound_mismatches"], 0)
        self.assertEqual(POPULATION_GATES["max_q_lt30_count_difference"], 0)
        self.assertEqual(SOURCE_EFFECT_GATES["max_count_effect_disagreement"], 0)


class ClaimControlTests(unittest.TestCase):
    def test_numerical_success_remains_screening_only(self):
        decision = assess_claim(run_core_gates(80), observational=False)
        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)

    def test_observation_claim_blocked_when_gates_missing(self):
        decision = assess_claim([], observational=True)
        self.assertEqual(decision.decision, Decision.SCREENING_ONLY)
        self.assertEqual(set(decision.missing_gates), set(LOCKED_OBSERVATION_GATES))

    def test_failed_gate_is_invalid(self):
        failed = GateResult("rank", False, "rank", "4", "6")
        decision = assess_claim([failed], observational=True)
        self.assertEqual(decision.decision, Decision.INVALID)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_ignores_dictionary_order(self):
        self.assertEqual(sha256_data({"a": 1, "b": 2}), sha256_data({"b": 2, "a": 1}))

    def test_atomic_record_contains_payload_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            record = write_run_record(path, {"result": "SCREENING_ONLY"})
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["payload_sha256"], record["payload_sha256"])

    def test_source_manifest_hashes_engine(self):
        root = Path(__file__).resolve().parents[1]
        manifest = source_manifest(root)
        self.assertIn("src/jxplanetx/yoshida6.py", manifest["files"])
        self.assertEqual(manifest["scope"], "repository")
        self.assertEqual(len(manifest["tree_sha256"]), 64)

    def test_installed_package_manifest_hashes_executable_source(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder) / "jxplanetx"
            package.mkdir()
            (package / "__init__.py").write_text('__version__ = "test"\n')
            (package / "engine.py").write_text("VALUE = 1\n")
            manifest = package_source_manifest(package)
            self.assertEqual(manifest["scope"], "installed_package")
            self.assertEqual(
                set(manifest["files"]),
                {"src/jxplanetx/__init__.py", "src/jxplanetx/engine.py"},
            )
            self.assertEqual(len(manifest["tree_sha256"]), 64)

    def test_runtime_manifest_is_never_empty(self):
        manifest = runtime_source_manifest()
        self.assertTrue(manifest["files"])
        self.assertIn(manifest["scope"], {"repository", "installed_package"})

    def test_empty_repository_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                source_manifest(folder)

    def test_locked_manifest_hash_is_sha256(self):
        self.assertEqual(len(EXPECTED_CHECKSUM_MANIFEST_SHA256), 64)

    def test_benchmark_refuses_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "checksums.sha256").write_text("not the benchmark\n")
            with self.assertRaises(ValueError):
                verify_bundle(folder)


if __name__ == "__main__":
    unittest.main()
