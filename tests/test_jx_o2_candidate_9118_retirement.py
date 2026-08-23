from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_DIR = ROOT / "audits" / "jx_o2_g0_candidate_9118_retirement_v1"
RETIREMENT_PATH = RETIREMENT_DIR / "candidate_9118_retirement_v1.json"
REGISTRATION_PATH = RETIREMENT_DIR / "registration_retirement_v1.json"
G0_DIR = ROOT / "audits" / "jx_o2_g0_input_audit_v1"
JX_O2_RUN_DIR = ROOT / "runs" / "planet_x_survey_model_comparison_v1"

EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
SOURCE_AUDIT_ID = "jx-o2-g0-input-audit-v1"
RESOLUTION_ID = "jx-o2-g0-candidate-9118-retirement-v1"
NONCLAIM = (
    "This receipt records only a provenance disposition. Candidate 9118 "
    "remains a historical exploratory screening assumption and is not evidence "
    "for or against Planet X. JX-O2 remains blocked and no execution is authorized."
)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
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


def normalize_token(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def numeric_values(value: Any) -> set[float]:
    result: set[float] = set()
    for item in walk_json(value):
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            result.add(float(item))
        elif isinstance(item, str):
            try:
                result.add(float(item))
            except ValueError:
                pass
    return result


def candidate_signal_present(value: Any) -> bool:
    aliases = {
        "candidate9118",
        "p9bb21idx9118",
        "p9bb21index9118",
        "bb219118",
        "bb21index9118",
        "brownbatygincatalogindex9118",
        "de441source9118state",
        "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf",
    }
    contextual_key_terms = {
        "candidate",
        "catalog",
        "source",
        "body",
        "planet",
        "grid",
        "model",
        "row",
        "index",
    }
    orbit_fingerprint = {5.06, 495.19, 0.236, 20.28, 284.17, 96.87, 126.21}
    if orbit_fingerprint.issubset(numeric_values(value)):
        return True
    for item in walk_json(value):
        if isinstance(item, str):
            normalized = normalize_token(item)
            if any(alias in normalized for alias in aliases):
                return True
        if not isinstance(item, dict):
            continue
        for key, child in item.items():
            key_normalized = normalize_token(key)
            if any(alias in key_normalized for alias in aliases):
                return True
            child_normalized = normalize_token(child)
            if child_normalized == "9118" and any(
                term in key_normalized for term in contextual_key_terms
            ):
                return True
    return False


class JXO2Candidate9118RetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retirement = load_json(RETIREMENT_PATH)
        cls.registration = load_json(REGISTRATION_PATH)

    def test_identity_and_fail_closed_state(self) -> None:
        receipt = self.retirement
        self.assertEqual(receipt["schema"], "jx-o2-g0-candidate-retirement/v1")
        self.assertEqual(receipt["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(receipt["source_audit_id"], SOURCE_AUDIT_ID)
        self.assertEqual(receipt["resolution_id"], RESOLUTION_ID)
        self.assertEqual(
            receipt["resolution_status"],
            "CANDIDATE_REMOVED_FROM_JX_O2_SCOPE_WITHOUT_LINEAGE_RECOVERY",
        )
        self.assertEqual(receipt["resolution_method"], "PERMANENT_SCOPE_REMOVAL")
        self.assertEqual(
            receipt["artifact_class"], "SCOPED_G0_CANDIDATE_RETIREMENT"
        )
        self.assertEqual(
            receipt["base_repository_commit"],
            "2d5e3aa001d12178bd7c264d378cc41d6072fef9",
        )
        self.assertEqual(receipt["issue"]["number"], 11)
        self.assertEqual(
            receipt["issue"]["url"],
            "https://github.com/linomx90/jx-planet-x-engine/issues/11",
        )
        self.assertEqual(receipt["claim_ceiling"], "PROVENANCE_DISPOSITION_ONLY")
        self.assertFalse(receipt["execution_authorized"])
        self.assertFalse(receipt["observed_execution_authorized"])
        self.assertFalse(receipt["gpu_run_authorized"])
        self.assertFalse(receipt["outcomes_generated"])
        self.assertFalse(receipt["observed_jx_o2_score_computed"])
        self.assertEqual(receipt["mandatory_nonclaim"], NONCLAIM)

    def test_search_records_bounded_nonrecovery_without_overclaim(self) -> None:
        search = self.retirement["search_record"]
        self.assertEqual(
            search["result"], "NOT_RECOVERED_WITHIN_DOCUMENTED_SEARCH_SCOPE"
        )
        self.assertEqual(
            [entry["scope_id"] for entry in search["documented_search_scope"]],
            [f"S{index:02d}" for index in range(1, 6)],
        )
        self.assertGreaterEqual(len(search["limitations"]), 3)
        self.assertTrue(
            any("not that the record never existed anywhere" in item for item in search["limitations"])
        )
        metadata = search["metadata_blob"]
        self.assertEqual(
            metadata["expected_sha256"],
            "509917e0093107464d9ee45ed2c8e9f403403b2bb0e94455fa3614825917f8b0",
        )
        self.assertEqual(
            metadata["status"], "NOT_RECOVERED_WITHIN_DOCUMENTED_SEARCH_SCOPE"
        )
        self.assertFalse(metadata["was_parsed_by_historical_state_builder"])
        self.assertEqual(
            set(search["selection_decision_fields"]),
            {
                "selector",
                "selection_time",
                "selection_rule",
                "candidate_universe",
                "ranking",
                "prior_inspection_record",
            },
        )
        self.assertTrue(
            all(
                value == "NOT_RECOVERED"
                for value in search["selection_decision_fields"].values()
            )
        )
        replay = search["deterministic_selection_replay"]
        self.assertEqual(
            replay["status"], "NOT_RECONSTRUCTABLE_FROM_DOCUMENTED_EVIDENCE"
        )
        self.assertFalse(
            replay["retrospective_reconstruction_may_be_presented_as_original_provenance"]
        )

        evidence = search["search_evidence"]
        git_evidence = evidence["git_repository"]
        self.assertFalse(git_evidence["is_shallow"])
        self.assertEqual(git_evidence["retained_ref_count"], 12)
        self.assertEqual(
            git_evidence["object_database_counts"],
            {"commit": 63, "tree": 162, "blob": 181},
        )
        self.assertEqual(git_evidence["unreachable_object_count"], 0)
        self.assertGreaterEqual(len(git_evidence["commands"]), 9)

        github_evidence = evidence["public_github"]
        self.assertEqual(len(github_evidence["public_branch_heads_examined"]), 8)
        self.assertEqual(github_evidence["plausible_metadata_and_selection_paths_examined"], 40)
        self.assertEqual(github_evidence["pull_request_2_review_comment_count"], 0)
        self.assertFalse(github_evidence["code_and_commit_search_treated_as_authoritative"])
        self.assertIn("candidate_metadata.txt", github_evidence["query_terms"])

        workspace = evidence["accessible_workspace"]
        self.assertEqual(workspace["expected_sha256_scan_max_bytes_exclusive"], 1048576)
        self.assertEqual(workspace["expected_sha256_match_count"], 0)
        self.assertFalse(workspace["recorded_scratch_path_present"])

    def test_tracked_history_does_not_invent_selection_provenance(self) -> None:
        history = self.retirement["tracked_history"]
        self.assertEqual(
            history["first_candidate_appearance_commit"],
            "269c68adb823ac445e72c48bdea92d9da2207f22",
        )
        self.assertEqual(
            history["parent_commit"],
            "e6ff286abbf31f216e21deca2b83ee65e61a0487",
        )
        self.assertFalse(history["parent_contains_candidate_reference"])
        self.assertTrue(history["candidate_tuple_hard_coded_in_builder"])
        self.assertFalse(history["metadata_blob_used_to_derive_candidate_tuple"])
        self.assertFalse(history["selection_time_inferred_from_import_commit"])
        self.assertFalse(history["selection_rationale_present_in_import_commit"])

    def test_candidate_is_permanently_retired_from_this_experiment(self) -> None:
        policy = self.retirement["retirement_policy"]
        self.assertEqual(policy["lifecycle_state"], "RETIRED_FROM_JX_O2_M1")
        self.assertEqual(policy["classification"], "EXPLORATORY_SCREENING_ONLY")
        self.assertTrue(policy["permanent_within_experiment"])
        self.assertFalse(policy["reinstatement_within_this_experiment"])
        self.assertEqual(
            set(policy["candidate_identifiers"]),
            {
                "candidate_9118",
                "P9_BB21_idx9118",
                "Brown-Batygin catalog index 9118",
            },
        )
        self.assertTrue(policy["historical_screening_artifacts_preserved_unchanged"])
        self.assertFalse(policy["historical_numerical_results_reinterpreted"])
        self.assertTrue(policy["eligibility"])
        self.assertTrue(all(value is False for value in policy["eligibility"].values()))

        precedence = self.retirement["policy_precedence"]
        self.assertEqual(
            precedence["governing_rule"],
            "STRICTER_RETIREMENT_POLICY_CONTROLS_CANDIDATE_9118_FOR_THIS_EXPERIMENT",
        )
        self.assertEqual(
            precedence["prior_conditional_rule"]["json_pointer"],
            "/candidate_selection_lineage/candidate_9118_rule",
        )
        prior_path = (
            RETIREMENT_DIR / precedence["prior_conditional_rule"]["path"]
        ).resolve()
        self.assertEqual(
            sha256_file(prior_path), precedence["prior_conditional_rule"]["sha256"]
        )
        self.assertEqual(
            precedence["policy_effect"],
            "REPLACES_CONDITIONAL_ELIGIBILITY_WITH_PERMANENT_INELIGIBILITY",
        )
        self.assertEqual(precedence["scope"], "THIS_EXPERIMENT_ID_ONLY")
        self.assertIn("NO_LATER_RECEIPT_OR_VERSION_MAY_REINSTATE", precedence["non_supersession_rule"])

    def test_epoch_defect_is_preserved_without_invented_repair(self) -> None:
        catalog = self.retirement["catalog_identity"]
        epoch = self.retirement["epoch_disposition"]
        self.assertEqual(catalog["catalog_index"], 9118)
        self.assertEqual(catalog["historical_jx_name"], "P9_BB21_idx9118")
        self.assertEqual(catalog["official_catalog_epoch_jd"], 2458270.0)
        self.assertEqual(epoch["official_catalog_epoch_jd"], 2458270.0)
        self.assertEqual(epoch["historical_jx_epoch_jd"], 2461200.5)
        self.assertEqual(
            epoch["historical_treatment"],
            "CATALOG_ELEMENTS_REDECLARED_AT_JX_EPOCH_WITHOUT_PROPAGATION",
        )
        self.assertFalse(epoch["physical_propagation_or_refit_recovered"])
        self.assertFalse(epoch["historical_epoch_defect_repaired"])
        self.assertFalse(epoch["eligible_as_jx_o2_physical_state"])
        self.assertTrue(epoch["historical_reproduction_only"])

    def test_scope_exclusion_does_not_claim_recovery_or_complete_g0(self) -> None:
        blocker = self.retirement["g0_blocker_disposition"]
        state = self.retirement["overall_state"]
        self.assertEqual(blocker["original_blocker_id"], "G0-B07")
        self.assertEqual(blocker["original_recovery_requirement_status"], "NOT_SATISFIED")
        self.assertEqual(blocker["primary_recovery_path_status"], "NOT_SATISFIED")
        self.assertEqual(blocker["fallback_scope_removal_status"], "SATISFIED")
        self.assertEqual(blocker["disposition"], "SCOPE_EXCLUSION_NOT_RECOVERY")
        self.assertTrue(blocker["scope_limited_to_candidate_9118"])
        self.assertTrue(blocker["does_not_complete_g0"])
        self.assertEqual(state["g0_audit_status"], "BLOCKED")
        self.assertFalse(state["g0_complete"])
        self.assertEqual(state["activation_gate_a03_status"], "NOT_SATISFIED")
        self.assertFalse(state["eligible_for_g1"])
        self.assertFalse(state["matched_m0_m1_family_acquired"])
        self.assertFalse(state["generic_candidate_selection_lineage_complete"])

        exposure = self.retirement["outcome_exposure_and_decision_independence"]
        self.assertTrue(
            exposure["historical_candidate_9118_numerical_outcomes_previously_inspected"]
        )
        self.assertFalse(exposure["retirement_decision_based_on_numerical_outcome"])
        self.assertEqual(
            set(exposure["retirement_basis"]),
            {"MISSING_SELECTION_PROVENANCE", "INVALID_JX_O2_EPOCH_LINEAGE"},
        )
        self.assertFalse(
            exposure[
                "future_m1_family_grid_or_weights_may_be_tuned_using_historical_9118_outcomes"
            ]
        )

    def test_permissions_never_expand(self) -> None:
        false_keys = {
            "execution_authorized",
            "observed_execution_authorized",
            "gpu_run_authorized",
            "outcomes_generated",
            "observed_jx_o2_score_computed",
            "promote_candidate_9118",
            "create_execution_contract",
            "create_activation_receipt",
            "run_synthetic_calibration",
            "run_observed_model_comparison",
            "access_or_unblind_untouched_holdout",
            "start_large_gpu_job",
            "g0_complete",
            "eligible_for_g1",
            "lineage_recovered",
            "epoch_defect_repaired",
            "activation_gate_a03_satisfied",
            "matched_m0_m1_family_acquired",
        }
        for artifact in (self.retirement, self.registration):
            for item in walk_json(artifact):
                if not isinstance(item, dict):
                    continue
                for key in false_keys:
                    if key in item:
                        self.assertIs(item[key], False, f"{key} must remain false")

    def test_prior_artifacts_remain_byte_identical(self) -> None:
        expected_hashes = {
            "g0_registration": "21634c974ffdff0f59b1f7b33c80bdabdfe9fc337f3476bf6f20be3ede00bd03",
            "g0_audit": "5f7a5f9cef4f6bd73f7c36235e610d96c8475bd646bf5807aa591c58fbf002b0",
            "g0_source_model_inventory": "0ac5ef9bc1fbf84313a9ee5a8650a3685cba3d2d06eea1d29a547c7eda59da9f",
            "jx_o2_design_contract": "ccd9631097a403d374ca1d6954ef32751027b1b3b755d5de7065ccce8017c971",
            "jx_o2_source_model_manifest": "d723c2dede9dee4bd490751ca4b5ad334397ecf8e913094039ddb8d624b6ac73",
            "historical_state_manifest": "ec6e2ba8877d206838172dbb573318a2122977e263c054994d4c82a409a43d36",
            "historical_source_state": "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf",
            "historical_control_state": "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14",
            "historical_state_builder": "1a10ead88bf2f8cd5ff872207328c9c1d574c26750295736125dc16ed49c983f",
            "historical_release_manifest": "e3bf310509fc0b2f33e3dea273f5baff3272f0da6bf82ee4363e9087c733d325",
            "historical_population_result": "24b7572cf130c683acd66a4677dac62d0d30b15b24f9bf5997b612a8045d7efd",
            "historical_independent_result": "2fe7c4a5f9c26b76ff0f4c2aaa02c584171a33739150801d6e4dce072771b6ff",
            "historical_independent_audit": "f58df8ac8f5394ba0c83c6872788b0f976b7bd878c647c2ed6441cd9ea549b5b",
        }
        bindings = self.retirement["bound_prior_artifacts"]
        self.assertEqual(set(bindings), set(expected_hashes))
        for artifact_id, expected_hash in expected_hashes.items():
            binding = bindings[artifact_id]
            self.assertEqual(binding["sha256"], expected_hash)
            path = (RETIREMENT_DIR / binding["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), expected_hash)

    def test_registration_hashes_every_new_substantive_file(self) -> None:
        self.assertEqual(
            self.registration["schema"],
            "jx-o2-g0-candidate-retirement-registration/v1",
        )
        self.assertEqual(
            self.registration["artifact_class"],
            "IMMUTABLE_SCOPED_G0_RETIREMENT_REGISTRATION",
        )
        self.assertEqual(
            self.registration["base_repository_commit"],
            "2d5e3aa001d12178bd7c264d378cc41d6072fef9",
        )
        self.assertEqual(self.registration["resolution_id"], RESOLUTION_ID)
        self.assertEqual(self.registration["g0_audit_status"], "BLOCKED")
        self.assertFalse(self.registration["g0_complete"])
        self.assertEqual(self.registration["mandatory_nonclaim"], NONCLAIM)
        self.assertEqual(
            self.registration["timestamp_authority"],
            "The Git object and pull-request history that first publish this immutable receipt.",
        )
        self.assertIn(
            "No later receipt or version may reinstate candidate 9118",
            self.registration["non_supersession_rule"],
        )
        search_completed = datetime.fromisoformat(
            self.retirement["search_record"]["search_completed_at_utc"].replace(
                "Z", "+00:00"
            )
        )
        receipt_recorded = datetime.fromisoformat(
            self.retirement["recorded_at_utc"].replace("Z", "+00:00")
        )
        registration_recorded = datetime.fromisoformat(
            self.registration["registered_at_utc"].replace("Z", "+00:00")
        )
        self.assertLessEqual(search_completed, receipt_recorded)
        self.assertLessEqual(receipt_recorded, registration_recorded)
        expected = {
            "README.md",
            "candidate_9118_retirement_v1.json",
            "../../tests/test_jx_o2_candidate_9118_retirement.py",
        }
        locked = self.registration["locked_files"]
        self.assertEqual({entry["path"] for entry in locked.values()}, expected)
        for entry in locked.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            path = (RETIREMENT_DIR / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), entry["sha256"])

    def test_compact_tree_contains_no_runner_result_or_state_payload(self) -> None:
        expected_files = {
            "README.md",
            "candidate_9118_retirement_v1.json",
            "registration_retirement_v1.json",
        }
        self.assertEqual(
            {
                path.relative_to(RETIREMENT_DIR).as_posix()
                for path in RETIREMENT_DIR.rglob("*")
                if path.is_file()
            },
            expected_files,
        )
        self.assertFalse(any(path.is_dir() for path in RETIREMENT_DIR.rglob("*")))
        self.assertLess(
            sum(path.stat().st_size for path in RETIREMENT_DIR.iterdir()), 100000
        )
        forbidden_names = ("runner", "result", "execution", "state.csv", "archive")
        for path in RETIREMENT_DIR.iterdir():
            lowered = path.name.lower()
            self.assertTrue(all(token not in lowered for token in forbidden_names))

    def test_future_jx_o2_model_artifacts_cannot_promote_candidate(self) -> None:
        immutable_historical_candidate_artifacts = {
            (
                JX_O2_RUN_DIR / "source_models_manifest_v1.json"
            ).resolve(): "d723c2dede9dee4bd490751ca4b5ad334397ecf8e913094039ddb8d624b6ac73",
            (
                G0_DIR / "g0_audit_v1.json"
            ).resolve(): "5f7a5f9cef4f6bd73f7c36235e610d96c8475bd646bf5807aa591c58fbf002b0",
            (
                G0_DIR / "source_model_inventory_v1.json"
            ).resolve(): "0ac5ef9bc1fbf84313a9ee5a8650a3685cba3d2d06eea1d29a547c7eda59da9f",
            (
                G0_DIR / "registration_g0_v1.json"
            ).resolve(): "21634c974ffdff0f59b1f7b33c80bdabdfe9fc337f3476bf6f20be3ede00bd03",
            RETIREMENT_PATH.resolve(): (
                "01a041382aa716293db9f61ae8917d44bd80c62347cc2aaa7c03d0442f6009cd"
            ),
        }
        for path, expected_hash in immutable_historical_candidate_artifacts.items():
            self.assertEqual(sha256_file(path), expected_hash, path)
        self.assertNotIn(
            (JX_O2_RUN_DIR / "future_execution.json").resolve(),
            immutable_historical_candidate_artifacts,
        )

        expected_binding = {
            "resolution_id": RESOLUTION_ID,
            "registration_sha256": sha256_file(REGISTRATION_PATH),
            "use": "HISTORICAL_REFERENCE_ONLY",
            "eligible_for_jx_o2_use": False,
        }

        def assert_future_artifact_safe(artifact: dict[str, Any], label: Any) -> None:
            binding = artifact.get("candidate_9118_retirement_binding")
            remainder = dict(artifact)
            remainder.pop("candidate_9118_retirement_binding", None)
            if binding is not None:
                self.assertEqual(binding, expected_binding, label)
            self.assertFalse(
                candidate_signal_present(remainder),
                f"retired candidate signal outside exact historical binding: {label}",
            )

        for path in [*(ROOT / "runs").rglob("*.json"), *(ROOT / "audits").rglob("*.json")]:
            resolved = path.resolve()
            if resolved in immutable_historical_candidate_artifacts:
                continue
            artifact = load_json(path)
            if resolved == REGISTRATION_PATH.resolve():
                self.assertEqual(
                    artifact["schema"],
                    "jx-o2-g0-candidate-retirement-registration/v1",
                )
                continue
            schema = artifact.get("schema", "")
            is_jx_o2_artifact = (
                artifact.get("experiment_id") == EXPERIMENT_ID
                or path.parent.name.startswith("planet_x_survey_model_comparison")
                or (isinstance(schema, str) and schema.startswith("jx-o2"))
            )
            if is_jx_o2_artifact:
                assert_future_artifact_safe(artifact, path)

        valid_reference_only = {
            "schema": "jx-o2-future-provenance/v1",
            "experiment_id": EXPERIMENT_ID,
            "candidate_9118_retirement_binding": expected_binding,
        }
        assert_future_artifact_safe(valid_reference_only, "valid reference-only binding")
        promotion_attempts = (
            {
                **valid_reference_only,
                "candidate": {
                    "source_sha256": (
                        "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf"
                    ),
                    "included": True,
                },
            },
            {
                **valid_reference_only,
                "model_input": {
                    "catalog_index": 9118,
                    "selected": True,
                    "weight": 1.0,
                    "family_role": "M1",
                },
            },
        )
        for attempt in promotion_attempts:
            with self.assertRaises(AssertionError):
                assert_future_artifact_safe(attempt, attempt)

    def test_candidate_signal_guard_catches_common_evasions(self) -> None:
        orbit_as_strings = {
            "m": "5.06",
            "semi_major": "495.19",
            "eccentricity": "0.236",
            "inclination": "20.28",
            "longitude_perihelion": "284.17",
            "node": "96.87",
            "mean_anomaly": "126.21",
        }
        bypass_attempts = (
            {"catalog_index": "9118"},
            {"source_index": 9118},
            {"source_path": "../../runs/de441_population_100k/states/de441_source_9118_state.csv"},
            {
                "renamed_input_sha256":
                    "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf"
            },
            {"catalog_row": 9118},
            {"candidate_id": "P9-BB21-index-9118"},
            {"renamed_candidate_parameters": orbit_as_strings},
        )
        for attempt in bypass_attempts:
            self.assertTrue(candidate_signal_present(attempt), attempt)
        self.assertFalse(
            candidate_signal_present(
                {"candidate_id": "SYNTHETIC_CONTROL_42", "catalog_index": 42}
            )
        )

    def test_claim_firewall_and_strict_json_parser(self) -> None:
        forbidden_keys = {
            "observed_jx_o2_score",
            "observed_test_statistic",
            "observed_p_value",
            "observed_bayes_factor",
            "observed_model_preference",
            "model_preference",
            "preferred_model",
            "planet_x_mass_estimate",
            "planet_x_orbit_estimate",
            "planet_x_detected",
            "planet_x_excluded",
        }
        forbidden_tokens = {
            "DE" + "TECTED",
            "EX" + "CLUDED",
            "CON" + "FIRMED",
            "RULED" + "OUT",
            "PA" + "SSED",
            "PRE" + "FERRED",
        }
        for artifact in (self.retirement, self.registration):
            for item in walk_json(artifact):
                if isinstance(item, dict):
                    self.assertTrue(forbidden_keys.isdisjoint(item))
                elif isinstance(item, str):
                    tokens = {token for token in item.upper().split("_") if token}
                    self.assertTrue(forbidden_tokens.isdisjoint(tokens), item)

        invalid_documents = (
            '{"execution_authorized": true, "execution_authorized": false}',
            '{"execution_authorized": false, "value": NaN}',
            '{"execution_authorized": false, "value": Infinity}',
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, document in enumerate(invalid_documents):
                path = Path(directory) / f"invalid-{index}.json"
                path.write_text(document, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_json(path)

        forbidden_placeholders = ("FINAL_" + "TIMESTAMP", "TO" + "DO", "T" + "BD")
        for path in [*RETIREMENT_DIR.iterdir(), Path(__file__)]:
            content = path.read_text(encoding="utf-8").upper()
            for placeholder in forbidden_placeholders:
                self.assertNotIn(placeholder, content, path)


if __name__ == "__main__":
    unittest.main()
