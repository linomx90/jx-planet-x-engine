from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "planet_x_survey_model_comparison_v1"
DESIGN_PATH = RUN_DIR / "design_contract_v1.json"
SURVEY_PATH = RUN_DIR / "survey_inputs_manifest_v1.json"
MODELS_PATH = RUN_DIR / "source_models_manifest_v1.json"
REGISTRATION_PATH = RUN_DIR / "registration_design_v1.json"

EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
DESIGN_NONCLAIM = (
    "This design authorizes no execution and contains no observational outcome. "
    "A future result may compare only the specified generative model families "
    "within locked data and assumptions; it cannot by itself detect or exclude "
    "Planet X. JX-O1 V4 validates only the survey-adapter calibration prerequisite."
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


class PlanetXSurveyModelDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = load_json(DESIGN_PATH)
        cls.surveys = load_json(SURVEY_PATH)
        cls.models = load_json(MODELS_PATH)
        cls.registration = load_json(REGISTRATION_PATH)

    def test_design_is_permanently_non_executable_and_contains_no_outcome(self) -> None:
        self.assertEqual(
            self.design["schema"], "jx-observational-model-comparison-design/v1"
        )
        self.assertEqual(self.design["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.design["artifact_class"], "DESIGN_ONLY_PREREGISTRATION")
        self.assertEqual(
            self.design["registration_status"],
            "REGISTERED_BEFORE_ANY_JX_O2_ANALYSIS_EXECUTION_OR_COMPUTED_RESULT",
        )
        self.assertFalse(self.design["pre_data_registration"])
        self.assertTrue(self.design["prior_public_observational_literature_inspected"])
        self.assertFalse(self.design["outcomes_generated_at_registration"])
        self.assertFalse(self.design["execution_authorized"])
        self.assertEqual(
            self.design["design_state"], "DESIGN_REGISTERED_NOT_EXECUTABLE"
        )
        self.assertEqual(self.design["claim_ceiling"], "MODEL_FAMILY_COMPARISON_ONLY")
        self.assertEqual(
            self.design["scientific_evidence_role"], "PROTOCOL_NOT_OUTCOME_EVIDENCE"
        )
        self.assertEqual(self.design["mandatory_nonclaim"], DESIGN_NONCLAIM)
        self.assertTrue(self.design["unresolved_execution_blockers"])
        self.assertFalse(self.design["lifecycle"]["design_artifact_may_authorize_execution"])
        self.assertTrue(
            self.design["lifecycle"][
                "design_artifact_execution_authorized_must_remain_false"
            ]
        )

    def test_all_design_artifacts_fail_closed(self) -> None:
        for artifact in (
            self.design,
            self.surveys,
            self.models,
            self.registration,
        ):
            self.assertEqual(artifact["experiment_id"], EXPERIMENT_ID)
            self.assertFalse(artifact["execution_authorized"])

        self.assertFalse(self.registration["outcomes_generated_at_registration"])
        self.assertFalse(self.registration["pre_data_registration"])
        self.assertTrue(
            self.registration["prior_public_observational_literature_inspected"]
        )
        self.assertEqual(
            self.registration["design_state"], "DESIGN_REGISTERED_NOT_EXECUTABLE"
        )
        self.assertEqual(
            self.registration["claim_ceiling"], "MODEL_FAMILY_COMPARISON_ONLY"
        )
        self.assertTrue(self.registration["unresolved_execution_blockers"])

        prohibited_permissions = (
            "run_synthetic_calibration",
            "access_or_unblind_untouched_holdout_outcomes",
            "compute_jx_o2_observed_score",
            "run_observed_model_comparison",
            "start_large_gpu_job",
            "issue_planet_x_detection_or_exclusion_claim",
        )
        for permission in prohibited_permissions:
            self.assertFalse(self.registration["current_permissions"][permission])
        self.assertFalse(
            self.registration["activation_status"]["observed_execution_authorized"]
        )

        for artifact in (
            self.design,
            self.surveys,
            self.models,
            self.registration,
        ):
            for item in walk_json(artifact):
                if isinstance(item, dict) and "execution_authorized" in item:
                    self.assertFalse(item["execution_authorized"])

    def test_claim_firewall_has_no_observational_decision_state(self) -> None:
        forbidden_keys = set(self.design["claim_firewall"]["forbidden_result_keys"])
        actual_keys: set[str] = set()
        for artifact in (
            self.design,
            self.surveys,
            self.models,
            self.registration,
        ):
            for item in walk_json(artifact):
                if isinstance(item, dict):
                    actual_keys.update(item)
        self.assertTrue(forbidden_keys.isdisjoint(actual_keys))

        allowed = set(self.design["allowed_future_observation_states"])
        forbidden_values = set(
            self.design["claim_firewall"]["forbidden_claim_enum_values"]
        )
        self.assertTrue(allowed.isdisjoint(forbidden_values))
        self.assertNotIn("SCREENING_ONLY", allowed)
        self.assertEqual(
            self.design["claim_firewall"]["mandatory_future_result_nonclaim"],
            "This compares specified generative model families within the locked "
            "data and assumptions; it does not detect or exclude Planet X.",
        )

    def test_neutral_hypotheses_are_required_but_not_falsely_completed(self) -> None:
        hypotheses = self.design["hypotheses"]
        self.assertTrue(hypotheses["labels_correct_wrong_forbidden"])
        self.assertEqual(
            hypotheses["exact_model_manifest_status"],
            "UNRESOLVED_EXECUTION_BLOCKER",
        )
        self.assertEqual(
            hypotheses["candidate_selection_lineage_status"],
            "UNRESOLVED_EXECUTION_BLOCKER",
        )
        self.assertIn("M0_LOCKED_BASELINE", hypotheses)
        self.assertIn("M1_LOCKED_COMPACT_BODY_FAMILY", hypotheses)

        model_families = self.models["required_model_families"]
        self.assertEqual(
            {entry["model_id"] for entry in model_families},
            {"M0_LOCKED_BASELINE", "M1_LOCKED_COMPACT_BODY_FAMILY"},
        )
        for entry in model_families:
            self.assertEqual(
                entry["exact_manifest_status"], "UNRESOLVED_EXECUTION_BLOCKER"
            )
            self.assertFalse(entry["eligible_for_execution"])
        self.assertEqual(self.models["acquired_model_artifacts"], [])

    def test_survey_scope_and_exposure_are_explicit(self) -> None:
        self.assertEqual(
            self.surveys["manifest_status"], "DESIGN_ONLY_INPUTS_NOT_ACQUIRED"
        )
        self.assertEqual(self.surveys["acquired_artifacts"], [])
        self.assertEqual(
            self.surveys["global_data_exposure_status"], "PREVIOUSLY_INSPECTED"
        )
        self.assertEqual(
            self.surveys["confirmatory_holdout_status"],
            "UNRESOLVED_EXECUTION_BLOCKER",
        )
        strata = self.surveys["required_survey_strata"]
        self.assertEqual(
            {entry["survey_id"] for entry in strata},
            {
                "OSSOS_AFFILIATED_CHARACTERIZED_ENSEMBLE",
                "DES_Y6_CHARACTERIZED_TNO_SAMPLE",
            },
        )
        self.assertEqual(
            self.surveys["partition_policy"]["unit"],
            "SURVEY_BLOCK_OR_CHRONOLOGY_NOT_RANDOM_OBJECT",
        )
        self.assertEqual(
            self.surveys["partition_policy"]["unknown_exposure_action"], "BLOCK"
        )
        self.assertTrue(self.surveys["unresolved_execution_blockers"])

    def test_statistical_safety_thresholds_are_locked(self) -> None:
        analysis = self.design["analysis_policy"]
        self.assertEqual(analysis["global_alpha"], 0.05)
        self.assertFalse(analysis["secondary_endpoints_govern_verdict"])
        self.assertFalse(analysis["optional_stopping"])
        self.assertTrue(analysis["full_search_repeated_in_every_null_simulation"])
        self.assertTrue(
            analysis["full_nuisance_refit_repeated_in_every_calibration_replicate"]
        )
        for key in (
            "primary_joint_statistic_status",
            "absolute_model_adequacy_rule_status",
            "nuisance_fit_rule_status",
            "global_threshold_rule_status",
        ):
            self.assertEqual(analysis[key], "UNRESOLVED_EXECUTION_BLOCKER")

        multiplicity = self.design["multiplicity_policy"]
        self.assertTrue(multiplicity["familywise_error_control_required"])
        self.assertEqual(multiplicity["maximum_familywise_error_rate"], 0.05)
        self.assertGreaterEqual(len(multiplicity["global_null_calibration_must_repeat"]), 7)

        power = self.design["power_policy"]
        self.assertEqual(power["minimum_power"], 0.9)
        self.assertEqual(power["confidence_level"], 0.95)
        self.assertEqual(power["maximum_global_type1_error"], 0.05)
        self.assertTrue(power["require_simultaneous_power_lower_bound"])
        self.assertTrue(power["require_simultaneous_type1_upper_bound"])
        self.assertFalse(power["point_estimate_alone_satisfies_gate"])
        self.assertFalse(power["jx_o1_v4_power_satisfies_gate"])
        self.assertEqual(
            power["fixed_replicate_counts_status"],
            "UNRESOLVED_EXECUTION_BLOCKER",
        )
        outside = self.design["failure_mapping"]["failed_model_outside_powered_region"]
        self.assertIn(
            outside["observation_state"],
            self.design["allowed_future_observation_states"],
        )
        self.assertEqual(
            outside["required_annotation"], power["outside_powered_region_label"]
        )

    def test_activation_requires_every_gate_without_waiver(self) -> None:
        activation = self.design["activation_gates"]
        self.assertEqual(
            activation["logic"],
            "ALL_REQUIRED_WITH_EXPLICIT_HOLDOUT_CLAIM_BRANCH_AND_NO_WAIVERS",
        )
        self.assertEqual(activation["gate_status"], "NOT_SATISFIED")
        self.assertFalse(activation["waivers_allowed"])
        self.assertTrue(
            activation["only_separate_activation_receipt_may_authorize_observed_execution"]
        )
        gates = activation["gates"]
        self.assertEqual(
            [gate["gate_id"] for gate in gates],
            [f"A{index:02d}" for index in range(1, 17)],
        )
        self.assertTrue(all(gate["status"] == "NOT_SATISFIED" for gate in gates))
        self.assertEqual(
            [stage["stage"] for stage in self.design["lifecycle"]["stop_go_sequence"]],
            [f"G{index}" for index in range(7)],
        )

    def test_design_links_exact_manifest_and_prior_calibration_hashes(self) -> None:
        for linked in self.design["linked_design_manifests"].values():
            path = (RUN_DIR / linked["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), linked["sha256"])

        prior = self.design["prior_evidence"]["jx_o1_v4"]
        for path_key, hash_key in (
            ("contract_path", "contract_sha256"),
            ("result_path", "result_sha256"),
        ):
            path = (RUN_DIR / prior[path_key]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), prior[hash_key])
        self.assertEqual(prior["allowed_role"], "SURVEY_ADAPTER_CALIBRATION_ONLY")
        self.assertFalse(prior["may_satisfy_observational_gate"])
        self.assertFalse(prior["may_establish_realistic_planet_x_power"])

    def test_registration_binds_every_design_file(self) -> None:
        expected = {
            "README.md",
            "design_contract_v1.json",
            "survey_inputs_manifest_v1.json",
            "source_models_manifest_v1.json",
            "../../tests/test_planet_x_survey_model_design.py",
        }
        locked = self.registration["locked_files"]
        self.assertEqual({entry["path"] for entry in locked.values()}, expected)
        for entry in locked.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(entry["sha256"], "0" * 64)
            path = (RUN_DIR / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), entry["sha256"])

        self.assertEqual(
            self.registration["design_contract_sha256"], sha256_file(DESIGN_PATH)
        )
        self.assertEqual(self.registration["mandatory_nonclaim"], DESIGN_NONCLAIM)

    def test_no_runner_result_or_execution_artifact_exists(self) -> None:
        expected_files = {
            "README.md",
            "design_contract_v1.json",
            "registration_design_v1.json",
            "source_models_manifest_v1.json",
            "survey_inputs_manifest_v1.json",
        }
        self.assertEqual(
            {
                path.relative_to(RUN_DIR).as_posix()
                for path in RUN_DIR.rglob("*")
                if path.is_file()
            },
            expected_files,
        )
        self.assertFalse(any(path.is_dir() for path in RUN_DIR.rglob("*")))

    def test_registered_text_has_no_unresolved_token_disguise(self) -> None:
        forbidden_tokens = ("TO" + "DO", "T" + "BD", "PLACE" + "HOLDER")
        paths = [
            RUN_DIR / "README.md",
            DESIGN_PATH,
            SURVEY_PATH,
            MODELS_PATH,
            REGISTRATION_PATH,
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8").upper()
            for token in forbidden_tokens:
                self.assertNotIn(token, text, f"{path} contains {token}")


if __name__ == "__main__":
    unittest.main()
