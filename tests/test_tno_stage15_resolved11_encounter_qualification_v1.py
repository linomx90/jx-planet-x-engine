from __future__ import annotations

import builtins
from contextlib import redirect_stderr
import copy
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_stage15_resolved11_encounter_qualification_v1.py"
)


def _load_source(name: str = "jx_tno_stage15_resolved11_encounter_qualification_v1_test"):
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(
            payload,
            str(SOURCE),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class Stage15Resolved11EncounterProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_source()
        cls.protocol = cls.module.build_protocol()

    def test_source_only_load_and_build_never_import_dynamics_modules(self) -> None:
        module = _load_source("jx_tno_stage15_source_only_probe")
        forbidden = ("numpy", "scipy", "cupy", "cupyx", "jxplanetx", "rebound")
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in forbidden or name.startswith(tuple(item + "." for item in forbidden)):
                raise AssertionError(f"forbidden dynamics/runtime import: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            rebuilt = module.build_protocol()
        self.assertEqual(rebuilt["content_sha256"], self.protocol["content_sha256"])
        self.assertNotIn("numpy", module.__dict__)
        self.assertNotIn("cupy", module.__dict__)

    def test_scope_selection_lanes_and_claim_ceiling_are_exact(self) -> None:
        value = self.protocol
        self.assertEqual(
            value["schema"],
            "jx.tno.stage15-resolved11-encounter-qualification.protocol.v1",
        )
        self.assertEqual(value["status"], "PREPARED_FAIL_CLOSED_EXECUTION_BLOCKED")
        scope = value["scope"]
        self.assertEqual(scope["ephemeris"], "DE441")
        self.assertEqual(scope["native_epoch_batch_count"], 12)
        self.assertEqual(scope["tracers_per_batch"], 9)
        self.assertEqual(scope["total_tracer_count"], 108)
        self.assertEqual(scope["holdout_clone_indices"], [0, 18, 36, 54, 73, 91, 109, 127])
        self.assertEqual(scope["checkpoint_count_per_lane"], 1001)
        self.assertEqual(scope["total_lane_count"], 24)
        self.assertEqual(
            [(row["steps_per_julian_year"], row["total_fixed_steps"]) for row in scope["lanes"]],
            [(64, 64_000), (128, 128_000)],
        )
        self.assertFalse(value["execution_authorized"])
        self.assertFalse(value["outcomes_generated"])
        self.assertTrue(value["preparation_only"])
        for key in (
            "encounter_engine_qualified",
            "adapter_implemented",
            "execution_authorized",
            "dynamics_executed",
            "result_artifacts_published",
            "qualification_established",
            "dynamical_classification_generated",
            "resonance_classification_generated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            self.assertIs(value["claims"][key], False)

    def test_inputs_oracle_slices_and_screening_boundary_are_pinned(self) -> None:
        value = self.protocol
        inputs = value["inputs"]
        roles = [row["role"] for row in inputs["files"]]
        self.assertEqual(len(roles), len(set(roles)))
        self.assertIn("stage1_v4_arrays", roles)
        self.assertIn("encounter_engine", roles)
        oracle = inputs["stage1_v4_de441_resolved11_ias15_oracle_lanes"]
        self.assertEqual(len(oracle), 12)
        self.assertEqual([row["batch_order"] for row in oracle], list(range(12)))
        self.assertEqual([row["object_id"] for row in oracle], list(self.module.OBJECT_IDS))
        self.assertTrue(all(row["checkpoint_count"] == 1001 for row in oracle))
        self.assertTrue(all(row["record_bytes"] == 960 for row in oracle))
        self.assertEqual(value["engine_adapter_contract"]["current_scientific_classification"], "MODEL_OUTPUT")
        self.assertEqual(value["engine_adapter_contract"]["current_project_scope"], "SCREENING_ONLY")
        self.assertFalse(value["engine_adapter_contract"]["public_result_exposes_full_source_node_stream"])
        self.assertFalse(value["engine_adapter_contract"]["adapter_implemented"])

    def test_route_restart_runtime_and_evidence_contracts_are_frozen(self) -> None:
        value = self.protocol
        route = value["encounter_route_contract"]
        self.assertEqual(route["maximum_routes_per_tracer"], 16_384)
        self.assertEqual(route["maximum_total_routes"], 100_000)
        self.assertEqual(route["method"], "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2")
        self.assertFalse(route["materializer_implemented"])
        restart = value["restart_contract"]
        self.assertEqual(restart["midpoint_year"], 500)
        self.assertEqual(restart["midpoint_lane_count"], 24)
        self.assertEqual(len(restart["selected_fresh_process_cases"]), 4)
        self.assertFalse(restart["restart_oracle_implemented"])
        runtime = value["runtime_custody_contract"]
        self.assertTrue(runtime["fresh_project_venv_child_required"])
        self.assertEqual(runtime["reject_preloaded_prefixes"], ["jxplanetx", "numpy", "scipy", "cupy", "cupyx"])
        self.assertFalse(runtime["runtime_boundary_implemented"])
        evidence = value["retained_evidence_contract"]
        self.assertEqual(evidence["arrays"]["exact_bytes"], 23_063_040)
        self.assertEqual(evidence["source_nodes"]["combined_exact_bytes"], 1_216_524_672)
        self.assertEqual(evidence["restart_streams_exact_bytes"], 1_731_456)

    def test_stage1_gate_budgets_and_decision_rules_are_inherited_exactly(self) -> None:
        value = self.protocol
        budgets = value["exact_gate_budgets"]
        expected = {
            "preliminary_scattering_diagnostic": (3, 80),
            "numerical_richardson_semimajor_axis_tail": (3, 8000),
            "fine_vs_ias15_semimajor_axis": (3, 4000),
            "fine_vs_ias15_scattering_diagnostic": (3, 4000),
            "cartesian_position_AU": (1, 500),
            "cartesian_velocity_AU_per_julian_year": (1, 50000),
            "jx_resolved11_max_relative_energy_drift": (1, 100000000),
            "jx_resolved11_max_relative_angular_momentum_drift": (1, 10000000000),
            "energy_contraction_floor": (1, 2**46),
        }
        for key, (numerator, denominator) in expected.items():
            self.assertEqual((budgets[key]["numerator"], budgets[key]["denominator"]), (numerator, denominator))
            self.assertEqual(float.fromhex(budgets[key]["binary64_hex"]), numerator / denominator)
        gates = value["predeclared_gates"]
        self.assertTrue(gates["all_individual_predicates_must_pass"])
        self.assertFalse(gates["preliminary_scattering_bit_used_for_acceptance"])
        self.assertIn("EXACT_SQUARED_FRACTIONS", gates["decision_arithmetic"])

    def test_protocol_validation_rejects_claim_blocker_and_type_tampering(self) -> None:
        self.assertEqual(
            self.module.validate_protocol(self.protocol),
            self.protocol["content_sha256"],
        )
        mutations = []
        claim = copy.deepcopy(self.protocol)
        claim["claims"]["science_result"] = True
        mutations.append(claim)
        blocker = copy.deepcopy(self.protocol)
        blocker["execution_gate"]["blockers"] = blocker["execution_gate"]["blockers"][1:]
        mutations.append(blocker)
        typed = copy.deepcopy(self.protocol)
        typed["scope"]["native_epoch_batch_count"] = True
        mutations.append(typed)
        digest = copy.deepcopy(self.protocol)
        digest["content_sha256"] = "0" * 64
        mutations.append(digest)
        for value in mutations:
            with self.subTest(value=value.get("content_sha256")):
                with self.assertRaises(self.module.Stage15ProtocolError):
                    self.module.validate_protocol(value)
        with self.assertRaises(self.module.Stage15ProtocolError):
            self.module._require_exact(True, 1, "bool-int collision")

    def test_bound_input_drift_fails_before_protocol_construction(self) -> None:
        original = self.module._read_regular_bytes

        def altered(path: Path) -> bytes:
            payload = original(path)
            if path.name == "cuda_massless_tracer.py":
                return payload + b"\n"
            return payload

        with mock.patch.object(self.module, "_read_regular_bytes", side_effect=altered):
            with self.assertRaisesRegex(self.module.Stage15ProtocolError, "size differs"):
                self.module.build_protocol()

    def test_strict_json_and_canonical_encoding_fail_closed(self) -> None:
        with self.assertRaises(self.module.Stage15ProtocolError):
            self.module._strict_json(b'{"a":1,"a":2}\n')
        with self.assertRaises(self.module.Stage15ProtocolError):
            self.module._strict_json(b'{"a":NaN}\n')
        with self.assertRaises(self.module.Stage15ProtocolError):
            self.module._load_canonical_payload(b'{ "a": 1 }\n', "pretty")
        canonical = self.module._canonical_json({"z": 1, "a": 2})
        self.assertEqual(canonical, b'{"a":2,"z":1}\n')

    def test_prepare_is_create_only_and_validate_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage15" / "PROTOCOL.json"
            prepared = self.module.prepare(output)
            self.assertEqual(output.read_bytes(), self.module._canonical_json(prepared))
            retained = self.module.validate_file(output)
            self.assertEqual(retained["content_sha256"], prepared["content_sha256"])
            with self.assertRaises(self.module.Stage15ProtocolError):
                self.module.prepare(output)

    def test_cli_has_no_execute_mode(self) -> None:
        parser = self.module._parser()
        self.assertFalse(hasattr(self.module, "execute"))
        parsed = parser.parse_args(["--validate-protocol", "--output", "/tmp/PROTOCOL.json"])
        self.assertTrue(parsed.validate_protocol)
        self.assertFalse(parsed.prepare_only)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args(["--execute"])
        self.assertIn("--prepare-only --validate-protocol is required", stderr.getvalue())
        self.assertNotIn("execute", self.protocol["implementation"]["cli_surface"].lower().replace("no_execute", ""))


if __name__ == "__main__":
    unittest.main()
