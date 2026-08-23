from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = ROOT / "audits" / "jx_o2_g0_public_reconstruction_design_v1"
DESIGN_PATH = DESIGN_DIR / "reconstruction_design_v1.json"
LITERATURE_PATH = DESIGN_DIR / "literature_prior_grid_v1.json"
REGISTRATION_PATH = DESIGN_DIR / "registration_reconstruction_design_v1.json"

EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
ARTIFACT_ID = "jx-o2-g0-public-reconstruction-design-v1"
BASE_COMMIT = "bffeeae889f6403fe087ebe0e6d6f3f08efe4e5c"
NONCLAIM = (
    "This design records public reconstruction constraints only. It does not "
    "recover or reproduce the authors' original simulation, authorize any run, "
    "compare models against observations, or provide evidence for or against "
    "Planet X."
)
LITERATURE_NONCLAIM = (
    "This inventory records public literature inputs and bounded nonrecovery "
    "findings only. It does not recover the authors' original checkpoint, "
    "angles, seeds, code, or machine-readable trajectory or footprint outputs; "
    "it authorizes no run and provides no evidence for or against Planet X."
)
RETIREMENT_BINDING = {
    "resolution_id": "jx-o2-g0-candidate-9118-retirement-v1",
    "registration_sha256": (
        "037c22ea0d1a9cb4c86eadc9056b8e1e605a7a5790cba60871daaa75aad1bbc5"
    ),
    "use": "HISTORICAL_REFERENCE_ONLY",
    "eligible_for_jx_o2_use": False,
}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def parse_finite_float(token: str) -> float:
    exact = Decimal(token)
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"out-of-range JSON number: {token}")
    if exact != 0 and value == 0:
        raise ValueError(f"underflowed JSON number: {token}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
            parse_float=parse_finite_float,
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


def walk_json(value: Any, path: tuple[str, ...] = ()):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_json(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_json(child, (*path, str(index)))


def json_structure_sha256(value: Any) -> str:
    """Bind every object key, list position, and scalar type without binding values."""
    entries: list[list[Any]] = []

    def record(child: Any, path: tuple[str, ...]) -> None:
        if isinstance(child, dict):
            entries.append(["object", list(path), sorted(child)])
            for key in sorted(child):
                record(child[key], (*path, key))
        elif isinstance(child, list):
            entries.append(["array", list(path), len(child)])
            for index, item in enumerate(child):
                record(item, (*path, str(index)))
        elif child is None:
            entries.append(["null", list(path)])
        elif isinstance(child, bool):
            entries.append(["boolean", list(path)])
        elif isinstance(child, int):
            entries.append(["integer", list(path)])
        elif isinstance(child, float):
            entries.append(["number", list(path)])
        elif isinstance(child, str):
            entries.append(["string", list(path)])
        else:
            raise TypeError(f"unsupported JSON value at {path}: {type(child).__name__}")

    record(value, ())
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def retired_candidate_signal_present(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "9118" in re.sub(r"[^a-z0-9]", "", str(key).lower())
            or retired_candidate_signal_present(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(retired_candidate_signal_present(child) for child in value)
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, Decimal)):
        try:
            candidate_number = Decimal(str(value))
        except InvalidOperation:
            return False
        return candidate_number.is_finite() and candidate_number == Decimal(9118)
    if isinstance(value, str):
        if "9118" in re.sub(r"[^a-z0-9]", "", value.lower()):
            return True
        try:
            candidate_number = Decimal(value.strip())
        except InvalidOperation:
            return False
        return candidate_number.is_finite() and candidate_number == Decimal(9118)
    return False


class JXO2PublicReconstructionDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = load_json(DESIGN_PATH)
        cls.literature = load_json(LITERATURE_PATH)
        cls.registration = load_json(REGISTRATION_PATH)
        cls.artifacts = (cls.design, cls.literature, cls.registration)

    def test_design_is_blocked_and_non_executable(self) -> None:
        self.assertEqual(
            self.design["schema"], "jx-o2-g0-public-reconstruction-design/v1"
        )
        self.assertEqual(self.design["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.design["artifact_id"], ARTIFACT_ID)
        self.assertEqual(
            self.design["milestone"],
            "JX_O2_G0_PUBLIC_MODEL_RECONSTRUCTION_DESIGN",
        )
        self.assertEqual(
            self.design["artifact_class"], "DESIGN_ONLY_G0_REMEDIATION_PROTOCOL"
        )
        self.assertEqual(
            self.design["registration_status"],
            "REGISTERED_BEFORE_ANY_PUBLIC_RECONSTRUCTION_EXECUTION_OR_OUTPUT",
        )
        self.assertEqual(self.design["base_repository_commit"], BASE_COMMIT)
        self.assertEqual(self.design["design_state"], "DESIGN_REGISTERED_NOT_EXECUTABLE")
        self.assertEqual(self.design["audit_status"], "BLOCKED")
        self.assertFalse(self.design["g0_complete"])
        self.assertFalse(self.design["eligible_for_g1"])
        self.assertFalse(self.design["execution_authorized"])
        self.assertFalse(self.design["gpu_run_authorized"])
        self.assertFalse(self.design["outcomes_generated"])
        self.assertEqual(
            self.design["outcomes_generated_scope"],
            "NO_JX_O2_OR_PUBLIC_RECONSTRUCTION_OUTPUT_WAS_GENERATED_BY_THIS_ARTIFACT",
        )
        self.assertEqual(
            self.design["claim_ceiling"], "PUBLIC_RECONSTRUCTION_DESIGN_ONLY"
        )
        self.assertEqual(
            self.design["scientific_evidence_role"],
            "PROTOCOL_AND_SOURCE_CONSTRAINTS_NOT_OUTCOME_EVIDENCE",
        )
        self.assertEqual(self.design["mandatory_nonclaim"], NONCLAIM)
        self.assertTrue(self.design["unresolved_execution_blockers"])

    def test_literature_inventory_identity_is_exact_and_non_executable(self) -> None:
        expected = {
            "schema": "jx-o2-g0-public-literature-grid/v1",
            "experiment_id": EXPERIMENT_ID,
            "artifact_id": "jx-o2-g0-public-literature-grid-v1",
            "artifact_class": "DESIGN_ONLY_LITERATURE_INPUT_INVENTORY",
            "artifact_status": "BLOCKED_NOT_EXECUTABLE",
            "base_repository_commit": BASE_COMMIT,
            "outcomes_generated_scope": (
                "NO_JX_O2_OR_PUBLIC_RECONSTRUCTION_OUTPUT_WAS_GENERATED_BY_THIS_ARTIFACT"
            ),
            "claim_ceiling": "PUBLIC_SOURCE_INVENTORY_ONLY",
            "scientific_evidence_role": (
                "LITERATURE_INPUT_PROVENANCE_NOT_OUTCOME_EVIDENCE"
            ),
            "mandatory_nonclaim": LITERATURE_NONCLAIM,
        }
        for key, value in expected.items():
            self.assertEqual(self.literature[key], value, key)

    def test_every_artifact_fails_closed_recursively(self) -> None:
        false_keys = {
            "execution_authorized",
            "observed_execution_authorized",
            "gpu_run_authorized",
            "outcomes_generated",
            "observed_jx_o2_score_computed",
            "g0_complete",
            "eligible_for_g1",
            "generate_or_select_checkpoint",
            "realize_angle_grid_or_prior",
            "realize_compact_body_emplacement",
            "realize_random_seeds",
            "run_dynamics",
            "run_synthetic_calibration",
            "compute_jx_o2_observed_score",
            "run_observed_model_comparison",
            "access_or_unblind_untouched_holdout",
            "start_large_gpu_job",
            "create_execution_contract",
            "create_activation_receipt",
            "checkpoint_generation_authorized",
            "seed_realization_authorized",
            "emplacement_realization_authorized",
            "send_external_author_contact",
        }
        for artifact in self.artifacts:
            self.assertEqual(artifact["experiment_id"], EXPERIMENT_ID)
            for _, item in walk_json(artifact):
                if not isinstance(item, dict):
                    continue
                for key in false_keys:
                    if key in item:
                        self.assertIs(item[key], False, f"{key} must remain false")

    def test_top_level_schemas_are_closed(self) -> None:
        expected_design_keys = {
            "schema", "experiment_id", "artifact_id", "milestone", "artifact_class",
            "registration_status", "base_repository_commit", "design_state",
            "audit_status", "g0_complete", "eligible_for_g1", "execution_authorized",
            "observed_execution_authorized", "gpu_run_authorized", "outcomes_generated",
            "outcomes_generated_scope", "observed_jx_o2_score_computed", "claim_ceiling",
            "scientific_evidence_role", "candidate_9118_retirement_binding",
            "retired_candidate_nonuse_policy", "bound_prior_artifacts",
            "linked_literature_inventory", "target_exact_replication",
            "independent_public_analogue", "checkpoint_policy",
            "compact_body_emplacement_policy", "angular_state_policy", "seed_policy",
            "paired_model_invariants", "literature_grid_policy", "g0_disposition",
            "policy_precedence", "permission_ontology", "current_permissions", "claim_firewall",
            "unresolved_execution_blockers", "mandatory_nonclaim",
        }
        expected_literature_keys = {
            "schema", "experiment_id", "artifact_id", "artifact_class", "artifact_status",
            "base_repository_commit", "execution_authorized",
            "observed_execution_authorized", "gpu_run_authorized", "outcomes_generated",
            "outcomes_generated_scope", "observed_jx_o2_score_computed", "g0_complete",
            "eligible_for_g1", "claim_ceiling", "scientific_evidence_role",
            "candidate_9118_retirement_binding", "evidence_tiers",
            "archive_audit_method", "source_archives", "published_2024_benchmark",
            "published_2026_cluster_influenced_input_grid",
            "published_2026_cluster_free_rows_policy", "public_search_boundary",
            "literature_exposure", "unresolved_execution_blockers", "mandatory_nonclaim",
        }
        expected_registration_keys = {
            "schema", "experiment_id", "artifact_id", "artifact_class",
            "registered_at_utc", "timestamp_authority", "base_repository_commit",
            "registration_status", "audit_status", "g0_complete", "eligible_for_g1",
            "execution_authorized", "observed_execution_authorized", "gpu_run_authorized",
            "outcomes_generated", "outcomes_generated_scope",
            "observed_jx_o2_score_computed", "claim_ceiling", "scientific_evidence_role",
            "candidate_9118_retirement_binding", "locked_files", "design_sha256",
            "literature_inventory_sha256", "registration_invariants",
            "immutability_enforcement", "g0_disposition", "policy_precedence",
            "current_permissions", "unresolved_execution_blockers", "mandatory_nonclaim",
        }
        self.assertEqual(set(self.design), expected_design_keys)
        self.assertEqual(set(self.literature), expected_literature_keys)
        self.assertEqual(set(self.registration), expected_registration_keys)

    def test_recursive_json_structures_are_closed(self) -> None:
        expected_structure_hashes = {
            "design": "d8f8dd474c6d107d42f3bf7f55a654ca4e45cddff82b1ef6315dad8697c9056d",
            "literature": "86b4e6bcdf6928101fce65efbe5023c28066f874333cc6c8dd3f4bc60feaf5c7",
            "registration": "907cc34ae2fc50c47ba1a53e82749690667958fb9f099d09cc808277493d0ba6",
        }
        self.assertEqual(json_structure_sha256(self.design), expected_structure_hashes["design"])
        self.assertEqual(
            json_structure_sha256(self.literature), expected_structure_hashes["literature"]
        )
        self.assertEqual(
            json_structure_sha256(self.registration), expected_structure_hashes["registration"]
        )

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            duplicate = Path(temporary_directory) / "duplicate.json"
            duplicate.write_text('{"execution_authorized": false, "execution_authorized": true}')
            with self.assertRaises(ValueError):
                load_json(duplicate)

            for token in ("NaN", "Infinity", "-Infinity"):
                nonfinite = Path(temporary_directory) / "nonfinite.json"
                nonfinite.write_text('{"value": ' + token + "}")
                with self.assertRaises(ValueError):
                    load_json(nonfinite)

            for token in ("1e400", "-1e400"):
                overflow = Path(temporary_directory) / "overflow.json"
                overflow.write_text('{"value": ' + token + "}")
                with self.assertRaises(ValueError):
                    load_json(overflow)

            for token in ("1e-400", "-1e-400"):
                underflow = Path(temporary_directory) / "underflow.json"
                underflow.write_text('{"value": ' + token + "}")
                with self.assertRaises(ValueError):
                    load_json(underflow)

    def test_critical_nested_schemas_and_missing_value_aliases_are_closed(self) -> None:
        expected_key_sets = {
            "target_exact_replication": {
                "status", "exact_replication_possible", "author_checkpoint_recovered",
                "complete_compact_body_state_recovered", "original_seeds_recovered",
                "modified_code_and_decks_recovered",
                "machine_readable_trajectory_or_footprint_outputs_recovered",
                "eligible_as_jx_o2_matched_pair",
                "published_comparator_uses_same_integrator_and_cadence",
                "summary_statistic_agreement_may_be_called_exact_replication",
                "minimum_recovery_condition",
            },
            "independent_public_analogue": {
                "status", "may_be_called_original_cluster_checkpoint",
                "may_be_called_reproduction_of_published_run",
                "may_become_a_new_registered_model_family_after_all_gates",
                "required_label", "required_claim_boundary",
            },
            "checkpoint_policy": {
                "status", "common_checkpoint_hash_required_within_each_paired_realization",
                "prelocked_ensemble_manifest_required_if_multiple_cluster_histories_are_modeled",
                "required_checkpoint_fields", "resolution_lanes",
                "checkpoint_generation_authorized",
            },
            "compact_body_emplacement_policy": {
                "status", "published_parent_checkpoint_generated_without_additional_compact_body",
                "emplacement_time_and_complete_state_must_be_locked_before_outputs",
                "same_emplacement_operator_required_for_every_m1_member",
                "post_checkpoint_lane_pairing_rule", "full_history_lane_pairing_rule",
                "post_checkpoint_insertion_estimand",
                "post_checkpoint_insertion_may_represent_self_consistent_primordial_formation_and_survival",
                "full_history_alternative_requires_separate_preregistered_primordial_m0_m1_initial_conditions",
                "required_fields", "emplacement_realization_authorized",
            },
            "angular_state_policy": {
                "status", "required_fields_before_any_model_output",
                "missing_values_default_to_zero", "single_favorable_imputation_allowed",
                "post_outcome_angle_selection_allowed", "adaptive_angle_grid_expansion_allowed",
                "allowed_later_resolution_methods",
                "same_angle_rule_required_at_every_physical_grid_point",
                "maximization_over_angles_repeated_in_every_null_replicate",
                "future_angle_state_weight_source_provenance_required",
                "provenance_must_bind_source_artifact_and_transformation_hash",
                "retired_candidate_or_derivative_provenance_allowed",
                "separate_2021_zero_phase_convention_inherited_as_2024_fact",
                "rotational_symmetry_assumed_with_tides_stars_and_survey_footprint",
                "observation_conditioned_angle_catalog_status",
            },
            "seed_policy": {
                "status", "original_author_seeds_status", "new_seed_derivation_formula",
                "new_seeds_may_be_called_recovered_author_seeds", "seed_shopping_allowed",
                "outcome_dependent_restarts_allowed", "caller_supplied_seed_override_allowed",
                "common_random_numbers_required_for_matched_arms", "fields_to_lock_later",
                "seed_realization_authorized",
            },
            "paired_model_invariants": {
                "same_checkpoint_bytes_within_each_paired_realization",
                "same_prelocked_cluster_history_ensemble_and_weights_across_families",
                "same_particle_ids_order_and_weights", "same_known_body_initial_states",
                "same_non_compact_body_forces", "same_stellar_and_tidal_history",
                "same_nuisance_and_survey_random_streams",
                "same_integrator_tolerances_cadence_and_loss_rules",
                "same_analysis_and_absolute_adequacy_rules",
                "only_registered_additional_compact_body_may_differ",
                "published_2024_comparator_is_a_valid_jx_o2_matched_pair",
                "published_2026_cluster_free_rows_are_a_valid_jx_o2_m0",
            },
            "literature_grid_policy": {
                "nine_physical_input_rows_recorded", "status",
                "probability_weights_status", "angular_expansion_status",
                "selection_lineage_status",
                "published_outputs_may_influence_future_weights_or_pruning",
                "equal_weighting_inferred_from_table", "adaptive_pruning_after_outputs",
            },
            "g0_disposition": {
                "source_blocker_id", "source_blocker_category",
                "source_blocker_status", "this_design_resolves_source_blocker",
                "g0_audit_status", "g0_complete", "eligible_for_g1",
            },
            "policy_precedence": {
                "effective_permission_rule", "canonical_permission_ontology",
                "this_artifact_may_expand_prior_permissions",
                "static_drafting_is_not_execution_contract_publication_or_activation",
                "future_external_contact_requires_case_specific_user_authorization",
            },
            "permission_ontology": {
                "missing_bound_permission_action", "unmapped_alias_action",
                "read_only_design_work_scope", "state_changing_effective_permissions",
                "all_state_changing_effective_permissions_false",
            },
        }
        for section, expected_keys in expected_key_sets.items():
            self.assertEqual(set(self.design[section]), expected_keys, section)

        orientation_key_forms = {
            "omega", "omegadeg", "ascendingnode", "ascendingnodedeg",
            "argumentofperihelion", "argumentofperiheliondeg",
            "longitudeofperihelion", "longitudeofperiheliondeg",
            "meananomaly", "meananomalydeg", "trueanomaly", "trueanomalydeg",
            "phase", "phasedeg",
        }
        seed_key_forms = {"seed", "randomseed", "rngseed", "seedvalue", "seedvalues"}
        checkpoint_payload_key_forms = {
            "checkpointpath", "checkpointsha256", "checkpointbytes", "checkpointuri"
        }
        authorization_aliases = {
            "dynamicsallowed", "runallowed", "simulationauthorized"
        }
        for artifact in self.artifacts:
            for path, item in walk_json(artifact):
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    canonical = re.sub(r"[^a-z0-9]", "", str(key).lower())
                    if canonical in orientation_key_forms:
                        self.assertEqual(path, ("published_2024_benchmark",))
                        self.assertEqual(value, "NOT_RECOVERED_IN_AUDITED_PUBLIC_SOURCES")
                    self.assertNotIn(canonical, seed_key_forms)
                    self.assertNotIn(canonical, checkpoint_payload_key_forms)
                    self.assertNotIn(canonical, authorization_aliases)

    def test_exact_replication_and_independent_analogue_are_not_conflated(self) -> None:
        exact = self.design["target_exact_replication"]
        analogue = self.design["independent_public_analogue"]
        self.assertEqual(exact["status"], "BLOCKED_SOURCE_ARTIFACTS_NOT_RECOVERED")
        self.assertFalse(exact["exact_replication_possible"])
        self.assertFalse(exact["author_checkpoint_recovered"])
        self.assertFalse(exact["complete_compact_body_state_recovered"])
        self.assertFalse(exact["original_seeds_recovered"])
        self.assertFalse(
            exact["machine_readable_trajectory_or_footprint_outputs_recovered"]
        )
        self.assertFalse(exact["eligible_as_jx_o2_matched_pair"])
        self.assertFalse(exact["published_comparator_uses_same_integrator_and_cadence"])
        self.assertFalse(exact["summary_statistic_agreement_may_be_called_exact_replication"])
        self.assertEqual(analogue["status"], "DESIGN_ONLY_NOT_INSTANTIATED")
        self.assertFalse(analogue["may_be_called_original_cluster_checkpoint"])
        self.assertFalse(analogue["may_be_called_reproduction_of_published_run"])
        self.assertEqual(analogue["required_label"], "INDEPENDENT_PUBLIC_MODEL_FAMILY")

    def test_literature_grid_is_exact_and_input_only(self) -> None:
        grid = self.literature["published_2026_cluster_influenced_input_grid"]
        self.assertEqual(grid["status"], "LITERATURE_SEED_GRID_NOT_EXECUTABLE")
        expected_rows = {
            ("B2026_CI_01", 5.0, 367, 0.2, 20),
            ("B2026_CI_02", 5.0, 420, 0.35, 20),
            ("B2026_CI_03", 5.0, 480, 0.5, 20),
            ("B2026_CI_04", 7.07, 356, 0.2, 20),
            ("B2026_CI_05", 7.07, 433, 0.35, 20),
            ("B2026_CI_06", 7.07, 497, 0.5, 20),
            ("B2026_CI_07", 10.0, 356, 0.2, 20),
            ("B2026_CI_08", 10.0, 433, 0.35, 20),
            ("B2026_CI_09", 10.0, 540, 0.5, 20),
        }
        actual_rows = {
            (
                row["model_id"],
                row["mass_earth"],
                row["a_au"],
                row["e"],
                row["i_deg"],
            )
            for row in grid["rows"]
        }
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(len(grid["rows"]), 9)
        required_keys = {"model_id", "mass_earth", "a_au", "e", "i_deg"}
        self.assertEqual(set(grid["row_schema"]), required_keys)
        self.assertTrue(all(set(row) == required_keys for row in grid["rows"]))
        self.assertFalse(grid["published_probability_weights_supplied"])
        self.assertFalse(grid["equal_weights_assumed"])
        self.assertFalse(grid["published_output_columns_included"])
        self.assertFalse(grid["grid_may_be_filtered_or_weighted_using_published_outputs"])
        serialized_rows = json.dumps(grid["rows"]).lower()
        self.assertNotIn('"w"', serialized_rows)
        self.assertNotIn("kappa", serialized_rows)

    def test_2026_cluster_free_rows_are_not_misused_as_m0(self) -> None:
        policy = self.literature["published_2026_cluster_free_rows_policy"]
        self.assertEqual(
            policy,
            {
                "recorded_as_m0_grid": False,
                "contain_additional_compact_body": True,
                "use_different_initial_population": True,
                "eligible_as_physically_matched_no_compact_body_control": False,
                "reason": (
                    "The eight cluster-free rows are compact-body runs from a distinct "
                    "initial population, not a no-compact-body arm paired to the nine "
                    "cluster-influenced rows."
                ),
            },
        )
        self.assertFalse(
            self.design["paired_model_invariants"][
                "published_2026_cluster_free_rows_are_a_valid_jx_o2_m0"
            ]
        )

    def test_missing_angular_state_is_not_zero_filled(self) -> None:
        benchmark = self.literature["published_2024_benchmark"]
        missing_value = "NOT_RECOVERED_IN_AUDITED_PUBLIC_SOURCES"
        self.assertEqual(
            benchmark,
            {
                "status": "PARTIAL_ORBIT_NOT_EXECUTABLE",
                "mass_earth": 5.0,
                "a_au": 500,
                "e": 0.25,
                "i_deg": 20,
                "ascending_node_deg": missing_value,
                "argument_of_perihelion_deg": missing_value,
                "longitude_of_perihelion_deg": missing_value,
                "mean_anomaly_deg": missing_value,
                "true_anomaly_deg": missing_value,
                "epoch": missing_value,
                "frame_origin_time_standard": missing_value,
                "eligible_as_exact_replication_state": False,
                "eligible_as_jx_o2_model_member": False,
            },
        )
        self.assertEqual(
            (benchmark["mass_earth"], benchmark["a_au"], benchmark["e"], benchmark["i_deg"]),
            (5.0, 500, 0.25, 20),
        )
        missing_keys = {
            "ascending_node_deg",
            "argument_of_perihelion_deg",
            "longitude_of_perihelion_deg",
            "mean_anomaly_deg",
            "true_anomaly_deg",
            "epoch",
            "frame_origin_time_standard",
        }
        self.assertTrue(
            all(
                benchmark[key] == "NOT_RECOVERED_IN_AUDITED_PUBLIC_SOURCES"
                for key in missing_keys
            )
        )
        self.assertFalse(benchmark["eligible_as_exact_replication_state"])
        self.assertFalse(benchmark["eligible_as_jx_o2_model_member"])

        angles = self.design["angular_state_policy"]
        self.assertEqual(angles["status"], "UNRESOLVED_EXECUTION_BLOCKER")
        self.assertFalse(angles["missing_values_default_to_zero"])
        self.assertFalse(angles["single_favorable_imputation_allowed"])
        self.assertFalse(angles["post_outcome_angle_selection_allowed"])
        self.assertFalse(angles["adaptive_angle_grid_expansion_allowed"])
        self.assertTrue(angles["same_angle_rule_required_at_every_physical_grid_point"])
        self.assertTrue(angles["maximization_over_angles_repeated_in_every_null_replicate"])
        self.assertTrue(angles["future_angle_state_weight_source_provenance_required"])
        self.assertTrue(
            angles["provenance_must_bind_source_artifact_and_transformation_hash"]
        )
        self.assertFalse(angles["retired_candidate_or_derivative_provenance_allowed"])
        self.assertFalse(angles["separate_2021_zero_phase_convention_inherited_as_2024_fact"])

    def test_seed_policy_is_deterministic_but_not_realized(self) -> None:
        seeds = self.design["seed_policy"]
        self.assertEqual(seeds["status"], "UNRESOLVED_EXECUTION_BLOCKER")
        self.assertEqual(
            seeds["original_author_seeds_status"],
            "NOT_RECOVERED_IN_AUDITED_PUBLIC_SOURCES",
        )
        self.assertEqual(
            seeds["new_seed_derivation_formula"],
            "SHA256(execution_contract_hash || post_registration_public_randomness_beacon || counter)",
        )
        self.assertFalse(seeds["new_seeds_may_be_called_recovered_author_seeds"])
        self.assertFalse(seeds["seed_shopping_allowed"])
        self.assertFalse(seeds["outcome_dependent_restarts_allowed"])
        self.assertFalse(seeds["caller_supplied_seed_override_allowed"])
        self.assertTrue(seeds["common_random_numbers_required_for_matched_arms"])
        self.assertFalse(seeds["seed_realization_authorized"])
        self.assertGreaterEqual(len(seeds["fields_to_lock_later"]), 6)

    def test_checkpoint_lanes_and_pairing_fail_closed(self) -> None:
        checkpoint = self.design["checkpoint_policy"]
        self.assertEqual(checkpoint["status"], "UNRESOLVED_EXECUTION_BLOCKER")
        self.assertTrue(
            checkpoint["common_checkpoint_hash_required_within_each_paired_realization"]
        )
        self.assertTrue(
            checkpoint[
                "prelocked_ensemble_manifest_required_if_multiple_cluster_histories_are_modeled"
            ]
        )
        self.assertGreaterEqual(len(checkpoint["required_checkpoint_fields"]), 8)
        self.assertEqual(len(checkpoint["resolution_lanes"]), 3)
        self.assertEqual(
            len({lane["lane_id"] for lane in checkpoint["resolution_lanes"]}), 3
        )
        lanes = {lane["lane_id"]: lane for lane in checkpoint["resolution_lanes"]}
        self.assertEqual(set(lanes), {"AUTHOR_EXACT", "OPEN_REGENERATION", "ANALYTIC_OR_FIGURE_SURROGATE"})
        self.assertFalse(lanes["OPEN_REGENERATION"]["eligible_to_claim_exact_replication"])
        self.assertFalse(lanes["ANALYTIC_OR_FIGURE_SURROGATE"]["eligible_to_satisfy_checkpoint_gate"])
        self.assertFalse(lanes["ANALYTIC_OR_FIGURE_SURROGATE"]["eligible_as_observational_evidence"])
        self.assertFalse(checkpoint["checkpoint_generation_authorized"])

        pair = self.design["paired_model_invariants"]
        self.assertEqual(
            pair,
            {
                "same_checkpoint_bytes_within_each_paired_realization": True,
                "same_prelocked_cluster_history_ensemble_and_weights_across_families": True,
                "same_particle_ids_order_and_weights": True,
                "same_known_body_initial_states": True,
                "same_non_compact_body_forces": True,
                "same_stellar_and_tidal_history": True,
                "same_nuisance_and_survey_random_streams": True,
                "same_integrator_tolerances_cadence_and_loss_rules": True,
                "same_analysis_and_absolute_adequacy_rules": True,
                "only_registered_additional_compact_body_may_differ": True,
                "published_2024_comparator_is_a_valid_jx_o2_matched_pair": False,
                "published_2026_cluster_free_rows_are_a_valid_jx_o2_m0": False,
            },
        )

        emplacement = self.design["compact_body_emplacement_policy"]
        self.assertEqual(emplacement["status"], "UNRESOLVED_EXECUTION_BLOCKER")
        self.assertTrue(
            emplacement["published_parent_checkpoint_generated_without_additional_compact_body"]
        )
        self.assertTrue(
            emplacement["emplacement_time_and_complete_state_must_be_locked_before_outputs"]
        )
        self.assertFalse(
            emplacement[
                "post_checkpoint_insertion_may_represent_self_consistent_primordial_formation_and_survival"
            ]
        )
        self.assertTrue(
            emplacement[
                "full_history_alternative_requires_separate_preregistered_primordial_m0_m1_initial_conditions"
            ]
        )
        self.assertFalse(emplacement["emplacement_realization_authorized"])
        self.assertGreaterEqual(len(emplacement["required_fields"]), 6)

        self.assertEqual(
            self.design["literature_grid_policy"],
            {
                "nine_physical_input_rows_recorded": True,
                "status": "LITERATURE_SEED_GRID_NOT_EXECUTABLE",
                "probability_weights_status": "UNRESOLVED_EXECUTION_BLOCKER",
                "angular_expansion_status": "UNRESOLVED_EXECUTION_BLOCKER",
                "selection_lineage_status": "UNRESOLVED_EXECUTION_BLOCKER",
                "published_outputs_may_influence_future_weights_or_pruning": False,
                "equal_weighting_inferred_from_table": False,
                "adaptive_pruning_after_outputs": "INVALID",
            },
        )
        self.assertEqual(
            self.design["g0_disposition"],
            {
                "source_blocker_id": "G0-B06",
                "source_blocker_category": "MATCHED_PHYSICAL_MODELS",
                "source_blocker_status": "UNRESOLVED_EXECUTION_BLOCKER",
                "this_design_resolves_source_blocker": False,
                "g0_audit_status": "BLOCKED",
                "g0_complete": False,
                "eligible_for_g1": False,
            },
        )

    def test_external_source_archive_identities_are_exact(self) -> None:
        archive_audit = self.literature["archive_audit_method"]
        self.assertEqual(archive_audit["audited_at_utc"], "2026-08-23T05:20:19Z")
        self.assertFalse(archive_audit["nested_archive_members_found"])
        self.assertFalse(archive_audit["binary_figures_ocr_exhaustively_searched"])
        self.assertFalse(
            archive_audit[
                "rendered_figures_count_as_machine_readable_trajectory_or_footprint_outputs"
            ]
        )
        self.assertEqual(len(archive_audit["method"]), 5)
        expected = {
            "NESVORNY_2023_CLUSTER_MODEL": (
                "2308.11059v1",
                2263011,
                "55ae68c68379b30e28be3dfb49a28d6ece8553ef84aa24fed81c3dbf3d3f2f90",
                18,
                "manuscript.tex",
                "60f507b485173cf19007079a735f43985a81be5ff519658172b972a4d1212f33",
            ),
            "BATYGIN_2024_LOW_INCLINATION_MODEL": (
                "2404.11594v1",
                4373401,
                "1078523dc00bb509c5763474d45cdc3c5b60362c18e3196d15a24f1fcbb6b9e0",
                7,
                "ms.tex",
                "b72c34d671216b9a22abe374ef7a49c217a5590e74edd0cc6658456c51472fa2",
            ),
            "BANSAL_2026_CLUSTER_INFLUENCED_GRID": (
                "2607.15646v1",
                799451,
                "84a9b3d08cfc155b7aa4723de8699104f54c544e562d9ec31b494bea7d5c9719",
                11,
                "main_paper.tex",
                "eec6d961b4b0f367ef13732ac5b73098525509ab9c7be685dc3ff818c0038803",
            ),
            "BROWN_BATYGIN_2021_ANGLE_CONVENTION": (
                "2108.09868v2",
                1573271,
                "4acf8448c98e58cfb0e92d3d58fc0e705852ef2befb010ca7bdbab76ecb5fa0e",
                12,
                "p9orbit.tex",
                "4d3df6d74d6fd5d981cc9eaa948b4743b265383b53365e9e4f8c8ff428639cff",
            ),
        }
        sources = {source["source_id"]: source for source in self.literature["source_archives"]}
        self.assertEqual(len(self.literature["source_archives"]), 4)
        self.assertEqual(
            len({source["source_id"] for source in self.literature["source_archives"]}),
            4,
        )
        self.assertEqual(set(sources), set(expected))
        for source_id, (
            arxiv_id,
            byte_count,
            digest,
            member_count,
            primary_tex_member,
            primary_tex_sha256,
        ) in expected.items():
            source = sources[source_id]
            self.assertEqual(source["arxiv_id"], arxiv_id)
            self.assertEqual(source["source_archive_bytes"], byte_count)
            self.assertEqual(source["source_archive_sha256"], digest)
            self.assertEqual(source["regular_archive_member_count"], member_count)
            self.assertEqual(source["primary_tex_member"], primary_tex_member)
            self.assertEqual(source["primary_tex_sha256"], primary_tex_sha256)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(source["archive_custody"], "HASHED_REMOTE_SOURCE_NOT_REDISTRIBUTED")

        exposure = self.literature["literature_exposure"]
        self.assertEqual(
            exposure,
            {
                "published_model_inputs_and_outcomes_previously_inspected": True,
                "parent_cluster_model_was_developed_and_validated_against_des_sdo_detections": True,
                "future_des_reuse_may_be_called_independent_confirmation": False,
                "future_grid_or_weights_may_be_tuned_using_published_outcomes": False,
                "same_observations_may_be_called_independent_confirmation": False,
            },
        )
        self.assertEqual(
            self.literature["public_search_boundary"],
            {
                "original_checkpoint_recovered": False,
                "original_full_compact_body_state_recovered": False,
                "original_seeds_recovered": False,
                "original_modified_code_and_decks_recovered": False,
                "original_machine_readable_trajectory_or_footprint_outputs_recovered": False,
                "exact_reproduction_possible_from_recorded_public_sources": False,
                "absence_claim_scope": (
                    "DOCUMENTED_PUBLIC_SOURCES_AUDITED_NOT_GLOBAL_NONEXISTENCE"
                ),
            },
        )

    def test_prior_bindings_and_linked_inventory_are_byte_exact(self) -> None:
        expected_hashes = {
            "jx_o2_design_registration": "d2978c56c492285d7e9f5cef6c9a8e4bdb927dca7c7a08699b4b45d1e8b2aa01",
            "jx_o2_design_contract": "ccd9631097a403d374ca1d6954ef32751027b1b3b755d5de7065ccce8017c971",
            "jx_o2_source_model_manifest": "d723c2dede9dee4bd490751ca4b5ad334397ecf8e913094039ddb8d624b6ac73",
            "g0_registration": "21634c974ffdff0f59b1f7b33c80bdabdfe9fc337f3476bf6f20be3ede00bd03",
            "g0_audit": "5f7a5f9cef4f6bd73f7c36235e610d96c8475bd646bf5807aa591c58fbf002b0",
            "g0_source_model_inventory": "0ac5ef9bc1fbf84313a9ee5a8650a3685cba3d2d06eea1d29a547c7eda59da9f",
        }
        self.assertEqual(set(self.design["bound_prior_artifacts"]), set(expected_hashes))
        for artifact_id, expected_hash in expected_hashes.items():
            binding = self.design["bound_prior_artifacts"][artifact_id]
            self.assertEqual(binding["sha256"], expected_hash)
            path = (DESIGN_DIR / binding["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), expected_hash)

        linked = self.design["linked_literature_inventory"]
        self.assertEqual(linked["path"], LITERATURE_PATH.name)
        self.assertEqual(linked["sha256"], sha256_file(LITERATURE_PATH))

    def test_permission_ontology_is_explicit_and_all_state_changes_are_denied(self) -> None:
        ontology = self.design["permission_ontology"]
        self.assertEqual(ontology["missing_bound_permission_action"], "DENY")
        self.assertEqual(ontology["unmapped_alias_action"], "DENY")
        self.assertEqual(
            ontology["read_only_design_work_scope"],
            [
                "PUBLIC_SOURCE_ACQUISITION",
                "DRAFT_AUTHOR_CONTACT_WITHOUT_SENDING",
                "STATIC_PROTOCOL_DRAFTING",
                "NONEXECUTING_INTEGRITY_TESTS",
            ],
        )
        expected_mappings = {
            "SEND_EXTERNAL_AUTHOR_CONTACT": (
                "contact_data_and_model_maintainers",
                "send_external_author_contact",
            ),
            "GENERATE_OR_SELECT_CHECKPOINT": ("generate_or_select_checkpoint",),
            "REALIZE_ANGLE_GRID_OR_PRIOR": ("realize_angle_grid_or_prior",),
            "REALIZE_COMPACT_BODY_EMPLACEMENT": (
                "realize_compact_body_emplacement",
            ),
            "REALIZE_RANDOM_SEEDS": ("realize_random_seeds",),
            "RUN_DYNAMICS_OR_SYNTHETIC_CALIBRATION": (
                "run_dynamics",
                "run_synthetic_calibration",
            ),
            "RUN_OBSERVED_MODEL_COMPARISON_OR_UNBLIND": (
                "compute_jx_o2_observed_score",
                "run_observed_model_comparison",
                "access_or_unblind_untouched_holdout",
                "access_or_unblind_untouched_holdout_outcomes",
            ),
            "START_LARGE_GPU_JOB": ("start_large_gpu_job",),
            "CREATE_OR_ACTIVATE_EXECUTION_CONTRACT": (
                "draft_separate_execution_contract",
                "create_execution_contract",
                "create_activation_receipt",
            ),
        }
        entries = ontology["state_changing_effective_permissions"]
        self.assertEqual(len(entries), len(expected_mappings))
        self.assertEqual(
            len({entry["canonical_action"] for entry in entries}), len(entries)
        )
        actual_mappings: dict[str, tuple[str, ...]] = {}
        for entry in entries:
            self.assertEqual(
                set(entry), {"canonical_action", "bound_permission_keys", "effective"}
            )
            self.assertIs(entry["effective"], False)
            self.assertTrue(entry["bound_permission_keys"])
            self.assertEqual(
                len(set(entry["bound_permission_keys"])),
                len(entry["bound_permission_keys"]),
            )
            actual_mappings[entry["canonical_action"]] = tuple(
                entry["bound_permission_keys"]
            )
        self.assertEqual(actual_mappings, expected_mappings)
        self.assertIs(ontology["all_state_changing_effective_permissions_false"], True)

    def test_retirement_binding_is_reference_only(self) -> None:
        for artifact in self.artifacts:
            self.assertEqual(artifact["candidate_9118_retirement_binding"], RETIREMENT_BINDING)
            remainder = dict(artifact)
            remainder.pop("candidate_9118_retirement_binding")
            self.assertFalse(retired_candidate_signal_present(remainder))

        nonuse = self.design["retired_candidate_nonuse_policy"]
        self.assertEqual(
            nonuse,
            {
                "governing_binding_is_historical_reference_only": True,
                "bound_retired_candidate_may_supply_angles_phase_state_weights_or_prior_influence": False,
                "partial_parameter_reuse_from_bound_retired_candidate_allowed": False,
                "numeric_or_alias_reencoding_changes_nonuse_policy": False,
            },
        )
        for attempted_value in (9118, 9118.0, "9118", "9118.0", "9.118e3"):
            self.assertTrue(
                retired_candidate_signal_present({"candidate-index": attempted_value})
            )

    def test_registration_hashes_every_substantive_file(self) -> None:
        self.assertEqual(
            self.registration["schema"], "jx-o2-g0-public-reconstruction-registration/v1"
        )
        self.assertEqual(self.registration["artifact_id"], ARTIFACT_ID)
        self.assertEqual(self.registration["base_repository_commit"], BASE_COMMIT)
        self.assertEqual(
            self.registration["artifact_class"],
            "IMMUTABLE_DESIGN_ONLY_G0_REMEDIATION_REGISTRATION",
        )
        self.assertEqual(
            self.registration["registration_status"],
            "DESIGN_REGISTERED_NOT_EXECUTABLE",
        )
        self.assertEqual(self.registration["audit_status"], "BLOCKED")
        self.assertFalse(self.registration["g0_complete"])
        self.assertFalse(self.registration["eligible_for_g1"])
        self.assertEqual(
            self.registration["claim_ceiling"], "PUBLIC_RECONSTRUCTION_DESIGN_ONLY"
        )
        self.assertEqual(
            self.registration["outcomes_generated_scope"],
            "NO_JX_O2_OR_PUBLIC_RECONSTRUCTION_OUTPUT_WAS_GENERATED_BY_THIS_ARTIFACT",
        )
        self.assertEqual(
            self.registration["scientific_evidence_role"],
            "IMMUTABLE_PROTOCOL_REGISTRATION_NOT_OUTCOME_EVIDENCE",
        )
        self.assertEqual(self.registration["mandatory_nonclaim"], NONCLAIM)
        self.assertEqual(
            self.registration["timestamp_authority"],
            "The Git object and pull-request history that first publish this immutable design package.",
        )
        registered_at = datetime.fromisoformat(
            self.registration["registered_at_utc"].replace("Z", "+00:00")
        )
        self.assertEqual(registered_at.tzinfo, timezone.utc)
        self.assertGreaterEqual(
            registered_at, datetime(2026, 8, 23, 5, 20, 19, tzinfo=timezone.utc)
        )
        expected = {
            "README.md",
            "literature_prior_grid_v1.json",
            "reconstruction_design_v1.json",
            "../../tests/test_jx_o2_public_reconstruction_design.py",
            "../../.github/workflows/jx-o2-immutability-v1.yml",
        }
        locked = self.registration["locked_files"]
        self.assertEqual({entry["path"] for entry in locked.values()}, expected)
        for entry in locked.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(entry["sha256"], "0" * 64)
            path = (DESIGN_DIR / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), entry["sha256"])

        self.assertEqual(
            self.registration["design_sha256"],
            self.registration["locked_files"]["reconstruction_design"]["sha256"],
        )
        self.assertEqual(self.registration["design_sha256"], sha256_file(DESIGN_PATH))
        self.assertEqual(
            self.registration["literature_inventory_sha256"],
            self.registration["locked_files"]["literature_prior_grid"]["sha256"],
        )
        self.assertEqual(
            self.registration["literature_inventory_sha256"], sha256_file(LITERATURE_PATH)
        )

        self.assertEqual(
            self.registration["registration_invariants"],
            {
                "registration_file_is_not_self_hashed": True,
                "locked_files_may_be_edited_in_place": False,
                "changes_require_new_version_and_preservation_of_this_record": True,
                "design_may_authorize_execution": False,
                "all_state_changing_effective_permissions_false": True,
                "analogue_may_be_relabelled_as_original_simulation": False,
                "missing_angles_seeds_or_checkpoint_may_be_zero_filled_or_invented": False,
            },
        )
        self.assertEqual(
            self.registration["g0_disposition"],
            {
                "source_blocker_id": "G0-B06",
                "source_blocker_status": "UNRESOLVED_EXECUTION_BLOCKER",
                "this_registration_resolves_source_blocker": False,
                "next_state": "CONTINUE_DESIGN_AND_PUBLIC_INPUT_ACQUISITION_ONLY",
            },
        )
        expected_precedence = {
            "effective_permission_rule": "STATE_CHANGING_ACTIONS_REQUIRE_ALL_MAPPED_PERMISSIONS_TRUE_AND_ANY_MISSING_MAPPING_DENIES",
            "canonical_permission_ontology": "reconstruction_design_v1.json#/permission_ontology",
            "this_artifact_may_expand_prior_permissions": False,
            "static_drafting_is_not_execution_contract_publication_or_activation": True,
            "future_external_contact_requires_case_specific_user_authorization": True,
        }
        self.assertEqual(self.registration["policy_precedence"], expected_precedence)
        self.assertEqual(self.design["policy_precedence"], expected_precedence)
        self.assertEqual(
            self.registration["immutability_enforcement"],
            {
                "pull_request_ci_step": "Verify immutable registered JX-O2 v1 artifacts",
                "workflow_path": ".github/workflows/jx-o2-immutability-v1.yml",
                "trusted_event": "pull_request_target",
                "trusted_base_workflow_compares_head_objects_without_executing_head_code": True,
                "rule": "IF_A_REGISTERED_V1_PATH_EXISTS_AT_THE_PULL_REQUEST_BASE_SHA_ITS_BYTES_MUST_NOT_CHANGE",
                "new_version_paths_remain_additive": True,
                "bootstrap_note": "This workflow first becomes trusted base-side enforcement after this registration is merged.",
                "direct_push_and_required_check_enforcement_requires_external_ruleset": True,
            },
        )
        expected_permissions = {
            "continue_public_source_acquisition": True,
            "draft_author_contact_request": True,
            "send_external_author_contact": False,
            "draft_static_reconstruction_protocols": True,
            "add_nonexecuting_integrity_tests": True,
            "generate_or_select_checkpoint": False,
            "realize_angle_grid_or_prior": False,
            "realize_compact_body_emplacement": False,
            "realize_random_seeds": False,
            "run_dynamics": False,
            "run_synthetic_calibration": False,
            "compute_jx_o2_observed_score": False,
            "run_observed_model_comparison": False,
            "access_or_unblind_untouched_holdout": False,
            "start_large_gpu_job": False,
            "create_execution_contract": False,
            "create_activation_receipt": False,
        }
        self.assertEqual(self.registration["current_permissions"], expected_permissions)
        self.assertEqual(self.design["current_permissions"], expected_permissions)

        workflow_text = (
            ROOT / ".github" / "workflows" / "jx-o2-immutability-v1.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("Verify immutable registered JX-O2 v1 artifacts", workflow_text)
        self.assertIn("pull_request_target:", workflow_text)
        self.assertIn("permissions:\n  contents: read", workflow_text)
        self.assertEqual(workflow_text.count("uses: actions/checkout@v4"), 1)
        self.assertIn("ref: ${{ github.event.pull_request.base.sha }}", workflow_text)
        self.assertIn(
            "EXPECTED_HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
            workflow_text,
        )
        self.assertIn(
            "+refs/pull/${PR_NUMBER}/head:refs/remotes/origin/jx-o2-pr-head",
            workflow_text,
        )
        self.assertIn('--no-ext-diff', workflow_text)
        self.assertIn('--no-textconv', workflow_text)
        self.assertIn('--no-renames', workflow_text)
        self.assertIn('--ignore-submodules=none', workflow_text)
        self.assertNotIn("git checkout", workflow_text)
        self.assertNotIn("git switch", workflow_text)
        self.assertNotIn("actions/setup-python", workflow_text)
        self.assertNotIn("pip install", workflow_text)
        self.assertNotIn("PYTHONPATH", workflow_text)
        expected_registered_paths = {
            ".github/workflows/jx-o2-immutability-v1.yml",
            "runs/planet_x_survey_model_comparison_v1",
            "audits/jx_o2_g0_input_audit_v1",
            "audits/jx_o2_g0_candidate_9118_retirement_v1",
            "audits/jx_o2_g0_public_reconstruction_design_v1",
            "tests/test_planet_x_survey_model_design.py",
            "tests/test_jx_o2_g0_audit.py",
            "tests/test_jx_o2_candidate_9118_retirement.py",
            "tests/test_jx_o2_public_reconstruction_design.py",
        }
        for registered_path in expected_registered_paths:
            self.assertIn(f"            {registered_path}\n", workflow_text)
        test_workflow_text = (ROOT / ".github" / "workflows" / "test.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("pull_request_target:", test_workflow_text)

    def test_compact_tree_has_no_run_payload(self) -> None:
        expected_files = {
            "README.md",
            "literature_prior_grid_v1.json",
            "reconstruction_design_v1.json",
            "registration_reconstruction_design_v1.json",
        }
        self.assertEqual(
            {
                path.relative_to(DESIGN_DIR).as_posix()
                for path in DESIGN_DIR.rglob("*")
                if path.is_file()
            },
            expected_files,
        )
        self.assertFalse(any(path.is_dir() for path in DESIGN_DIR.rglob("*")))
        self.assertLess(sum(path.stat().st_size for path in DESIGN_DIR.iterdir()), 120000)
        forbidden_names = ("runner", "result", "state", "checkpoint", "archive")
        for path in DESIGN_DIR.iterdir():
            lowered = path.name.lower()
            self.assertTrue(all(token not in lowered for token in forbidden_names))

    def test_claim_firewall_and_unresolved_marker_guard(self) -> None:
        forbidden_keys = set(self.design["claim_firewall"]["forbidden_outcome_keys"])
        forbidden_tokens = set(self.design["claim_firewall"]["forbidden_decision_tokens"])
        forbidden_authorization_keys = set(
            self.design["claim_firewall"]["forbidden_authorization_keys"]
        )
        self.assertEqual(
            forbidden_keys,
            {
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
                "verdict",
                "decision",
                "result",
                "result_state",
                "run_authorized",
                "p_value",
                "p-value",
                "bayes_factor",
                "model_score",
                "candidate_index",
                "candidate-index",
            },
        )
        self.assertEqual(
            forbidden_tokens,
            {
                "DETECTED", "EXCLUDED", "CONFIRMED", "RULED_OUT", "RULEDOUT",
                "PASSED", "PREFERRED",
            },
        )
        self.assertEqual(
            forbidden_authorization_keys,
            {
                "execution_permitted", "run_permitted", "simulation_permitted",
                "dynamics_allowed",
            },
        )
        self.assertEqual(
            set(self.design["claim_firewall"]),
            {
                "forbidden_outcome_keys",
                "forbidden_decision_tokens",
                "forbidden_authorization_keys",
                "analogue_may_be_relabelled_as_original_simulation",
                "summary_agreement_may_repair_missing_provenance",
                "more_gpu_objects_may_repair_missing_provenance",
            },
        )
        forbidden_key_forms = {
            re.sub(r"[^a-z0-9]", "", key.lower())
            for key in forbidden_keys | forbidden_authorization_keys
        }
        forbidden_word_tokens = {
            "DETECTED",
            "EXCLUDED",
            "CONFIRMED",
            "RULEDOUT",
            "PASSED",
            "PREFERRED",
        }
        for artifact in self.artifacts:
            for path, item in walk_json(artifact):
                is_declaration = path[:2] in {
                    ("claim_firewall", "forbidden_outcome_keys"),
                    ("claim_firewall", "forbidden_decision_tokens"),
                    ("claim_firewall", "forbidden_authorization_keys"),
                }
                if is_declaration:
                    continue
                if isinstance(item, dict):
                    canonical_keys = {
                        re.sub(r"[^a-z0-9]", "", str(key).lower()) for key in item
                    }
                    self.assertTrue(forbidden_key_forms.isdisjoint(canonical_keys), path)
                if isinstance(item, str):
                    words = re.findall(r"[A-Z]+", item.upper())
                    self.assertTrue(forbidden_word_tokens.isdisjoint(words), path)
                    self.assertNotIn(("RULED", "OUT"), set(zip(words, words[1:])), path)

        readme_words = re.findall(
            r"[A-Z]+", (DESIGN_DIR / "README.md").read_text(encoding="utf-8").upper()
        )
        self.assertTrue(forbidden_word_tokens.isdisjoint(readme_words))
        self.assertNotIn(("RULED", "OUT"), set(zip(readme_words, readme_words[1:])))

        attempted_decisions = ("RULEDOUT", "RULED_OUT", "RULED OUT", "PASSED")
        for attempted in attempted_decisions:
            words = re.findall(r"[A-Z]+", attempted.upper())
            detected = bool(forbidden_word_tokens.intersection(words)) or (
                ("RULED", "OUT") in set(zip(words, words[1:]))
            )
            self.assertTrue(detected, attempted)

        forbidden_markers = ("TO" + "DO", "T" + "BD", "PLACE" + "HOLDER")
        for path in (
            DESIGN_DIR / "README.md",
            DESIGN_PATH,
            LITERATURE_PATH,
            REGISTRATION_PATH,
            ROOT / "tests" / "test_jx_o2_public_reconstruction_design.py",
        ):
            text = path.read_text(encoding="utf-8").upper()
            for token in forbidden_markers:
                self.assertNotIn(token, text, f"{path} contains {token}")


if __name__ == "__main__":
    unittest.main()
