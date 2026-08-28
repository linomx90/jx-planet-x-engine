from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_verified_source_module() -> types.ModuleType:
    path = ROOT / "verify_preregistration.py"
    source = path.read_bytes()
    module = types.ModuleType("verify_preregistration_source_only")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


verifier = load_verified_source_module()


class OpenRegenerationPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registration_sha256 = verifier.sha256_file(ROOT / "registration_v1.json")
        cls.result = verifier.verify(ROOT, cls.registration_sha256)
        cls.artifacts = {
            name: verifier.strict_json(ROOT / filename)
            for name, filename in verifier.ARTIFACT_FILES.items()
        }

    def test_package_verifies_and_remains_blocked(self) -> None:
        self.assertEqual(
            self.result["status"],
            "JX_O2_OPEN_REGEN_PREREG_VERIFIED_DESIGN_ONLY_BLOCKED",
        )
        self.assertEqual(self.result["g0_requirement_count"], 31)
        self.assertEqual(self.result["g0_requirements_semantically_verified"], 0)
        self.assertFalse(self.result["g0_complete"])
        self.assertFalse(self.result["eligible_for_g1"])
        self.assertFalse(self.result["execution_authorized"])
        self.assertFalse(self.result["observed_execution_authorized"])
        self.assertFalse(self.result["gpu_run_authorized"])

    def test_external_registration_anchor_is_required(self) -> None:
        with self.assertRaises(RuntimeError):
            verifier.verify(ROOT, "0" * 64)

    def test_exact_flat_inventory_has_no_payload_or_runner(self) -> None:
        self.assertEqual({path.name for path in ROOT.iterdir()}, verifier.EXPECTED_FILES)
        prohibited = ("checkpoint", "state", "seed", "output", "result", "runner", "deck")
        for name in verifier.EXPECTED_FILES:
            lowered = name.lower()
            self.assertFalse(any(token in lowered for token in prohibited))

    def test_recursive_shapes_are_closed(self) -> None:
        for name, artifact in self.artifacts.items():
            self.assertEqual(
                verifier.json_shape_sha256(artifact),
                verifier.EXPECTED_SHAPE_SHA256[name],
            )
        registration = verifier.strict_json(ROOT / "registration_v1.json")
        self.assertEqual(
            verifier.json_shape_sha256(registration),
            verifier.EXPECTED_SHAPE_SHA256["registration"],
        )

    def test_no_g0_requirement_is_accepted(self) -> None:
        trace = self.artifacts["analysis"]["g0_requirement_trace"]
        self.assertEqual(len(trace), 31)
        self.assertEqual({item["requirement_id"] for item in trace}, set(verifier.EXPECTED_REQUIREMENT_STATUSES))
        self.assertEqual(
            {item["requirement_id"]: item["g0_acceptance_status"] for item in trace},
            verifier.EXPECTED_REQUIREMENT_STATUSES,
        )
        self.assertNotIn("SEMANTICALLY_VERIFIED", {item["g0_acceptance_status"] for item in trace})

    def test_every_state_changing_permission_is_false(self) -> None:
        main = self.artifacts["main"]
        permissions = main["current_permissions"]
        self.assertTrue(permissions["local_hash_verification"])
        self.assertFalse(any(value for key, value in permissions.items() if key != "local_hash_verification"))
        self.assertEqual(
            main["permission_ontology"]["state_changing_actions"],
            verifier.EXPECTED_PERMISSION_ACTIONS,
        )
        self.assertEqual(main["permission_ontology"]["missing_or_unmapped_action"], "DENY")

    def test_finite_physical_and_angular_support_is_exact(self) -> None:
        model = self.artifacts["model"]
        rows = model["physical_support"]["rows"]
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(Fraction(row["weight"]) for row in rows), 1)
        self.assertEqual(model["angular_support"]["sample_count"], 128)
        self.assertEqual(Fraction(model["angular_support"]["weight_per_realized_angle"]), Fraction(1, 128))
        self.assertEqual(model["joint_support"]["count_after_seed_realization"], 1152)
        self.assertEqual(Fraction(model["joint_support"]["each_weight"]), Fraction(1, 1152))
        self.assertFalse(model["angular_support"]["angle_values_realized"])
        self.assertFalse(model["joint_support"]["adaptive_member_selection"])

    def test_pairing_and_inference_unit_are_locked(self) -> None:
        main = self.artifacts["main"]
        history = self.artifacts["model"]["history_ensemble_and_assignment"]
        self.assertTrue(all(main["pairing_invariants"].values()))
        self.assertEqual(history["final_history_count"], 128)
        self.assertEqual(history["convergence_block_count"], 4)
        self.assertEqual(history["independent_joint_history_count_per_block"], 32)
        self.assertEqual(history["independent_disk_draw_count"], 128)
        self.assertEqual(history["independent_cluster_and_field_star_schedule_count"], 128)
        self.assertIn("NO_CROSS_REUSE", history["assignment_rule"])
        self.assertEqual(history["inference_unit"], "HISTORY")
        self.assertFalse(history["tracers_and_epochs_are_nested_independent_units"])
        self.assertFalse(history["m0_reuse_across_nine_rows_in_same_history_counts_as_nine_independent_controls"])

    def test_future_beacon_formula_realizes_no_seed(self) -> None:
        randomization = self.artifacts["randomization"]
        self.assertFalse(randomization["seed_values_realized"])
        self.assertEqual(randomization["seed_protocol_common"], verifier.EXPECTED_SEED_PROTOCOL_COMMON)
        self.assertEqual(randomization["input_seed_derivation"], verifier.EXPECTED_INPUT_SEED_DERIVATION)
        self.assertEqual(randomization["analysis_seed_derivation"], verifier.EXPECTED_ANALYSIS_SEED_DERIVATION)
        self.assertIn("INPUT_GENERATION_CONTRACT_SHA256", randomization["input_seed_derivation"]["master_formula"])
        self.assertIn("G1_EXECUTION_CONTRACT_SHA256", randomization["analysis_seed_derivation"]["master_formula"])

    def test_retired_candidate_is_governing_reference_only(self) -> None:
        verifier.validate_retirement_nonuse(self.artifacts)
        mutant = copy.deepcopy(self.artifacts)
        mutant["model"]["m1_definition"]["candidate_index"] = "9.118e3"
        with self.assertRaises(ValueError):
            verifier.validate_retirement_nonuse(mutant)
        mutant = copy.deepcopy(self.artifacts)
        mutant["model"]["m1_definition"]["source_state_sha256"] = (
            "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf"
        )
        with self.assertRaises(ValueError):
            verifier.validate_retirement_nonuse(mutant)

    def test_analysis_and_power_gates_are_fail_closed(self) -> None:
        analysis = self.artifacts["analysis"]
        power = analysis["power_and_calibration_policy"]
        self.assertEqual(power["fixed_history_count"], 128)
        self.assertEqual(power["threshold_selection_calibration_pseudocatalog_count_under_m0"], 100000)
        self.assertEqual(power["absolute_adequacy_calibration_pseudocatalog_count_per_family"], 100000)
        self.assertEqual(power["per_physical_row_power_pseudocatalog_count"], 100000)
        self.assertEqual(power["m0_audit_pseudocatalog_count"], 100000)
        self.assertEqual(power["m1_equal_weight_mixture_audit_pseudocatalog_count"], 100000)
        self.assertEqual(power["failed_power_gate_action"], "STOP_NO_OBSERVED_EXECUTION")
        self.assertFalse(analysis["activation_sequence"]["observed_execution_allowed"])
        self.assertEqual(
            analysis["future_closed_outcome_vocabulary"]["previously_inspected_data_disposition"],
            "PREREGISTERED_REANALYSIS_EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION",
        )

    def test_semantic_mutants_are_rejected(self) -> None:
        main = copy.deepcopy(self.artifacts["main"])
        main["current_permissions"]["run_pilot_or_dry_run"] = True
        with self.assertRaises(ValueError):
            verifier.validate_main(main)
        model = copy.deepcopy(self.artifacts["model"])
        model["physical_support"]["rows"][0]["weight"] = "2/9"
        with self.assertRaises(ValueError):
            verifier.validate_model(model)
        analysis = copy.deepcopy(self.artifacts["analysis"])
        analysis["g0_requirement_trace"][0]["g0_acceptance_status"] = "SEMANTICALLY_VERIFIED"
        with self.assertRaises(ValueError):
            verifier.validate_analysis(analysis)
        analysis = copy.deepcopy(self.artifacts["analysis"])
        analysis["analysis_policy"]["primary_joint_score_formula"] = "T_EQUALS_ARBITRARY"
        with self.assertRaises(ValueError):
            verifier.validate_analysis(analysis)
        analysis = copy.deepcopy(self.artifacts["analysis"])
        analysis["claim_firewall"]["allowed_current_statement"] = "PLANET_X_DETECTED"
        with self.assertRaises(ValueError):
            verifier.validate_analysis(analysis)
        randomization = copy.deepcopy(self.artifacts["randomization"])
        randomization["input_seed_derivation"]["master_formula"] = "SHA256(ARBITRARY)"
        with self.assertRaises(ValueError):
            verifier.validate_randomization(randomization)

    def test_strict_json_rejects_duplicates_constants_and_all_floats(self) -> None:
        cases = (
            '{"x": 1, "x": 2}',
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": 1.5}',
            '{"x": 1e400}',
            '{"x": 1e-400}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for payload in cases:
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError, msg=payload):
                    verifier.strict_json(path)

    def test_no_unresolved_marker_text_in_design_artifacts(self) -> None:
        for name in (*verifier.ARTIFACT_FILES.values(), "README.md"):
            text = (ROOT / name).read_text(encoding="utf-8").upper()
            self.assertNotIn("TODO", text)
            self.assertNotIn("TBD", text)
            self.assertNotIn("PLACEHOLDER", text)

    def test_prior_sources_are_exact_and_e2_is_excluded(self) -> None:
        records = {record["binding_id"]: record for record in self.artifacts["priors"]["bindings"]}
        self.assertEqual(set(records), set(verifier.EXPECTED_PRIOR_BINDINGS))
        self.assertEqual(records["JX_E2_LOCAL_CLOSURE"]["role"], "EXCLUDED_ENGINEERING_CONTEXT_NOT_MODEL_INPUT")
        self.assertEqual(
            records["CANDIDATE_9118_RETIREMENT_REGISTRATION"]["role"],
            "HIGHEST_PRECEDENCE_RETIREMENT_REGISTRATION",
        )


if __name__ == "__main__":
    unittest.main()
