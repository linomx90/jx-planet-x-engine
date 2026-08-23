from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audits" / "jx_o2_g0_input_audit_v1"
AUDIT_PATH = AUDIT_DIR / "g0_audit_v1.json"
OSSOS_PATH = AUDIT_DIR / "ossos_inventory_v1.json"
DES_PATH = AUDIT_DIR / "des_y6_inventory_v1.json"
MODELS_PATH = AUDIT_DIR / "source_model_inventory_v1.json"
REGISTRATION_PATH = AUDIT_DIR / "registration_g0_v1.json"
DESIGN_PATH = ROOT / "runs" / "planet_x_survey_model_comparison_v1" / "design_contract_v1.json"

AUDIT_ID = "jx-o2-g0-input-audit-v1"
EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
NONCLAIM = (
    "This audit contains no JX-O2 observational statistic or model preference "
    "and authorizes no execution. It does not detect, confirm, exclude, or rule "
    "out Planet X."
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


class JXO2G0InputAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load_json(AUDIT_PATH)
        cls.ossos = load_json(OSSOS_PATH)
        cls.des = load_json(DES_PATH)
        cls.models = load_json(MODELS_PATH)
        cls.registration = load_json(REGISTRATION_PATH)
        cls.artifacts = (
            cls.audit,
            cls.ossos,
            cls.des,
            cls.models,
            cls.registration,
        )

    def test_audit_is_blocked_and_permanently_non_executable(self) -> None:
        self.assertEqual(self.audit["schema"], "jx-o2-g0-input-audit/v1")
        self.assertEqual(self.audit["audit_id"], AUDIT_ID)
        self.assertEqual(self.audit["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.audit["audit_status"], "BLOCKED")
        self.assertFalse(self.audit["g0_complete"])
        self.assertFalse(self.audit["execution_authorized"])
        self.assertFalse(self.audit["observed_execution_authorized"])
        self.assertFalse(self.audit["gpu_run_authorized"])
        self.assertFalse(self.audit["outcomes_generated"])
        self.assertFalse(self.audit["observed_jx_o2_score_computed"])
        self.assertEqual(self.audit["claim_ceiling"], "PROVENANCE_AUDIT_ONLY")
        self.assertEqual(self.audit["mandatory_nonclaim"], NONCLAIM)
        self.assertEqual(self.audit["next_gate"]["stage"], "G0_REMEDIATION")
        self.assertEqual(
            self.audit["next_gate"]["execution_state"],
            "DESIGN_AND_ACQUISITION_ONLY",
        )

    def test_every_artifact_fails_closed_recursively(self) -> None:
        for artifact in self.artifacts:
            self.assertEqual(artifact["audit_id"], AUDIT_ID)
            self.assertEqual(artifact["experiment_id"], EXPERIMENT_ID)
            self.assertFalse(artifact["execution_authorized"])
            self.assertFalse(artifact["outcomes_generated"])
            if "g0_complete" in artifact:
                self.assertFalse(artifact["g0_complete"])
            if "audit_status" in artifact:
                self.assertEqual(artifact["audit_status"], "BLOCKED")
            if "claim_ceiling" in artifact:
                self.assertEqual(artifact["claim_ceiling"], "PROVENANCE_AUDIT_ONLY")
            if "eligible_for_g1" in artifact:
                self.assertFalse(artifact["eligible_for_g1"])
            for item in walk_json(artifact):
                if not isinstance(item, dict):
                    continue
                for key in (
                    "execution_authorized",
                    "observed_execution_authorized",
                    "gpu_run_authorized",
                    "run_synthetic_calibration",
                    "compute_jx_o2_observed_score",
                    "run_observed_model_comparison",
                    "access_or_unblind_untouched_holdout",
                    "start_large_gpu_job",
                    "create_execution_contract",
                    "create_activation_receipt",
                    "g0_complete",
                    "outcomes_generated",
                    "observed_jx_o2_score_computed",
                    "eligible_for_g1",
                ):
                    if key in item:
                        self.assertFalse(item[key], f"{key} must remain false")

        self.assertEqual(self.registration["audit_status"], "BLOCKED")
        self.assertFalse(self.registration["g0_complete"])
        self.assertFalse(self.registration["outcomes_generated"])
        self.assertEqual(
            self.registration["claim_ceiling"], "PROVENANCE_AUDIT_ONLY"
        )
        incomplete_receipt_fields = (
            "complete_ossos_selection_function_acquired",
            "des_selection_lineage_execution_ready",
            "physically_matched_m0_m1_pair_acquired",
            "candidate_9118_selection_lineage_complete",
            "untouched_confirmatory_holdout_identified",
            "all_licenses_resolved",
        )
        for key in incomplete_receipt_fields:
            self.assertFalse(self.registration["evidence_summary"][key])

    def test_json_parser_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
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

    def test_design_binding_and_registration_hash_every_compact_file(self) -> None:
        design_binding = self.audit["design_binding"]
        bound_design = (AUDIT_DIR / design_binding["path"]).resolve()
        self.assertEqual(bound_design, DESIGN_PATH.resolve())
        self.assertEqual(sha256_file(bound_design), design_binding["sha256"])
        self.assertEqual(
            design_binding["sha256"],
            "ccd9631097a403d374ca1d6954ef32751027b1b3b755d5de7065ccce8017c971",
        )

        expected = {
            "README.md",
            "g0_audit_v1.json",
            "ossos_inventory_v1.json",
            "des_y6_inventory_v1.json",
            "source_model_inventory_v1.json",
            "../../tests/test_jx_o2_g0_audit.py",
        }
        locked = self.registration["locked_files"]
        self.assertEqual({entry["path"] for entry in locked.values()}, expected)
        for entry in locked.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(entry["sha256"], "0" * 64)
            path = (AUDIT_DIR / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), entry["sha256"])

    def test_compact_audit_tree_contains_no_runner_result_or_raw_archive(self) -> None:
        expected_files = {
            "README.md",
            "g0_audit_v1.json",
            "ossos_inventory_v1.json",
            "des_y6_inventory_v1.json",
            "source_model_inventory_v1.json",
            "registration_g0_v1.json",
        }
        self.assertEqual(
            {
                path.relative_to(AUDIT_DIR).as_posix()
                for path in AUDIT_DIR.rglob("*")
                if path.is_file()
            },
            expected_files,
        )
        self.assertFalse(any(path.is_dir() for path in AUDIT_DIR.rglob("*")))
        self.assertTrue(
            all(path.suffix in {".json", ".md"} for path in AUDIT_DIR.iterdir())
        )
        self.assertLess(sum(path.stat().st_size for path in AUDIT_DIR.iterdir()), 200000)

    def test_g0_gate_assessment_is_fail_closed_and_complete(self) -> None:
        gates = self.audit["g0_subgates"]
        self.assertEqual(
            [gate["gate_id"] for gate in gates],
            [f"G0-{index:02d}" for index in range(1, 8)],
        )
        self.assertEqual(gates[0]["status"], "PARTIALLY_SATISFIED")
        self.assertTrue(all(gate["status"] == "BLOCKED" for gate in gates[1:]))
        activation = self.audit["design_activation_gates_assessed"]
        self.assertEqual(
            [gate["gate_id"] for gate in activation],
            [f"A{index:02d}" for index in range(1, 6)],
        )
        self.assertTrue(all(gate["status"] == "NOT_SATISFIED" for gate in activation))
        blockers = self.audit["blockers"]
        self.assertEqual(
            [blocker["blocker_id"] for blocker in blockers],
            [f"G0-B{index:02d}" for index in range(1, 10)],
        )
        self.assertTrue(all(blocker["resolution_required"] for blocker in blockers))

    def test_byte_verified_external_artifacts_are_content_addressed(self) -> None:
        byte_verified_count = 0
        for artifact in (self.ossos, self.des, self.models):
            for item in walk_json(artifact):
                if not isinstance(item, dict):
                    continue
                level = item.get("verification_level", "")
                if not level.startswith("BYTE_VERIFIED"):
                    continue
                byte_verified_count += 1
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
                self.assertNotEqual(item["sha256"], "0" * 64)
                self.assertGreater(item["byte_count"], 0)
                self.assertTrue(item["url"].startswith("https://"))
                self.assertRegex(
                    item["retrieved_at_utc"], r"^2026-08-23T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
                )
        self.assertGreaterEqual(byte_verified_count, 16)

    def test_ossos_catalog_is_hashed_but_full_selection_is_blocked(self) -> None:
        catalog_by_id = {
            item["artifact_id"]: item
            for item in self.ossos["complete_catalog_artifacts"]
        }
        self.assertEqual(catalog_by_id["OSSOS_CHARACTERIZED_CATALOG"]["record_count"], 840)
        ensemble = catalog_by_id["OSSOS_PLUS_ENSEMBLE_CATALOG"]
        self.assertEqual(ensemble["record_count"], 1148)
        self.assertEqual(ensemble["distinct_physical_object_count"], 1142)

        scope = self.ossos["catalog_semantics"]
        self.assertEqual(scope["published_sky_region_block_count"], 8)
        self.assertEqual(len(scope["required_characterization_components"]), 10)
        self.assertEqual(scope["published_table1_footprint_row_count"], 11)
        release_scope = self.ossos["official_simulator_repository"][
            "public_survsimb_mapping"
        ]["native_scope"]
        self.assertEqual(release_scope["ossos_components"], ["13AE", "13AO", "13BL", "14BH"])
        self.assertFalse(release_scope["includes_later_ossos_components"])

        withdrawn = self.ossos["withdrawn_later_ossos_selection"]
        self.assertEqual(withdrawn["status"], "QUARANTINED_NOT_ELIGIBLE")
        self.assertEqual(
            withdrawn["delete_commit"],
            "9a84f67f279851bbb4231b8d2ce633968f412561",
        )
        self.assertEqual(withdrawn["delete_message"], "Removing bad directory.")
        self.assertEqual(
            withdrawn["files"]["later_block_detection_catalog"]["record_count"],
            607,
        )
        self.assertEqual(withdrawn["published_later_block_characterized_count"], 605)
        self.assertEqual(withdrawn["withdrawn_only_object_ids"], ["o5c114", "o5t58"])
        self.assertEqual(len(self.ossos["deduplication"]["rediscovery_alias_pairs"]), 6)
        self.assertFalse(self.ossos["eligible_for_g1"])

    def test_des_catalog_and_selection_discrepancies_are_explicit(self) -> None:
        journal = self.des["journal_catalog"]["fits_member"]
        enhanced = self.des["enhanced_author_catalog"]["catalog"]
        self.assertEqual(journal["record_count"], 814)
        self.assertEqual(enhanced["record_count"], 814)
        self.assertEqual(
            self.des["enhanced_author_catalog"]["journal_parity"]["hr_rows_changed"],
            814,
        )
        scope = self.des["catalog_scope_and_orbits"]
        self.assertEqual(
            set(scope["noncharacterized_flagged_ids"]),
            {"2013 TA188", "2014 VC41", "C/2014 UN271"},
        )
        self.assertEqual(scope["rows_without_nc_flag"], 811)
        self.assertEqual(scope["strict_characterized_tno_only_count_after_excluding_centaur"], 810)
        self.assertEqual(scope["earlier_manuscript_prose_total_records"], 817)
        self.assertEqual(scope["earlier_manuscript_prose_nominal_tnos"], 815)
        self.assertTrue(scope["final_revision1_and_cds_release_is_canonical_for_this_audit"])
        self.assertEqual(
            self.des["native_simulator"]["declared_versions"],
            {
                "setup_py": "1.3.4",
                "destnosim_init_py": "1.3.3",
                "status": "CONFLICT_NOT_RESOLVED",
            },
        )

        payloads = {
            item["artifact_id"]: item for item in self.des["native_selection_payloads"]
        }
        self.assertEqual(
            payloads["DES_Y6_EXPOSURE_POSITIONS"]["sha256"],
            "514dcf72a55f9cf37b168dc59bdd37bdd35141bb7efe0dbe9986579674883b56",
        )
        self.assertEqual(
            payloads["DES_Y6_CCD_CORNERS"]["sha256"],
            "45a95d48eff422347c68ae2cf46186b541455c16c523e10cee07e11539b1c4e7",
        )
        self.assertEqual(payloads["DES_Y6_EXPOSURE_POSITIONS"]["active_hdu_unique_exposure_count"], 76162)
        self.assertEqual(payloads["DES_Y6_CCD_CORNERS"]["unique_exposure_count"], 76226)
        self.assertEqual(
            [entry["id"] for entry in self.des["blocking_discrepancies"]],
            [f"DES-D{index:02d}" for index in range(1, 9)],
        )
        self.assertFalse(self.des["eligible_for_g1"])

    def test_no_matched_model_pair_and_candidate_9118_is_not_promoted(self) -> None:
        pair = self.models["required_model_pair"]
        self.assertFalse(pair["public_execution_ready_pair_found"])
        self.assertEqual(pair["status"], "NOT_ACQUIRED")

        catalogs = self.models["brown_batygin_reference_planet_orbit_catalogs"]
        self.assertEqual(catalogs["record_count_each"], 100000)
        self.assertFalse(catalogs["eligible_as_tno_source_population"])
        self.assertEqual(
            catalogs["versions"][-1]["sha256"],
            "050b68182ecbf7fd76f280d8cc43c0683d207499aba1488a68092b834420a422",
        )

        candidate = self.models["candidate_9118"]
        self.assertEqual(candidate["catalog_index"], 9118)
        self.assertEqual(candidate["official_epoch_jd"], 2458270.0)
        self.assertEqual(candidate["jx_epoch_jd"], 2461200.5)
        self.assertEqual(
            candidate["epoch_treatment"],
            "CATALOG_ELEMENTS_REDECLARED_AT_JX_EPOCH_WITHOUT_PROPAGATION",
        )
        self.assertEqual(candidate["selection_lineage"]["status"], "MISSING")
        self.assertFalse(candidate["eligible_as_m1_family"])

        for entry in candidate["jx_local_artifacts"].values():
            path = (AUDIT_DIR / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertEqual(sha256_file(path), entry["sha256"])
        self.assertTrue(
            all(not lead["eligible_for_jx_o2_execution"] for lead in self.models["physical_model_leads"])
        )
        self.assertFalse(self.models["eligible_for_g1"])

    def test_claim_firewall_and_no_disguised_unresolved_tokens(self) -> None:
        design_forbidden_keys = {
            "planet_x_detected",
            "planet_x_excluded",
            "planet_x_present",
            "planet_x_absent",
            "detection_claim",
            "exclusion_claim",
        }
        firewall = self.audit["claim_firewall"]
        expected_outcome_keys = {
            "observed_jx_o2_score",
            "observed_test_statistic",
            "observed_p_value",
            "observed_bayes_factor",
            "observed_model_preference",
            "model_preference",
            "preferred_model",
            "planet_x_mass_estimate",
            "planet_x_orbit_estimate",
        }
        expected_decision_tokens = {
            "DETECTED",
            "EXCLUDED",
            "CONFIRMED",
            "RULED_OUT",
            "PASSED",
            "PREFERRED",
        }
        self.assertEqual(set(firewall["forbidden_outcome_keys"]), expected_outcome_keys)
        self.assertEqual(
            set(firewall["forbidden_decision_tokens"]), expected_decision_tokens
        )
        forbidden_keys = design_forbidden_keys | expected_outcome_keys
        forbidden_tokens = expected_decision_tokens
        actual_keys: set[str] = set()
        actual_string_values: set[str] = set()
        audit_without_firewall = {
            key: value for key, value in self.audit.items() if key != "claim_firewall"
        }
        for artifact in (
            audit_without_firewall,
            self.ossos,
            self.des,
            self.models,
            self.registration,
        ):
            for item in walk_json(artifact):
                if isinstance(item, dict):
                    actual_keys.update(item)
                elif isinstance(item, str):
                    actual_string_values.add(item)
        self.assertTrue(forbidden_keys.isdisjoint(actual_keys))
        for regression_value in (
            "RESULT_PASSED_WITH_EVIDENCE",
            "MODEL_PREFERRED_RESULT",
        ):
            self.assertFalse(
                forbidden_tokens.isdisjoint(regression_value.split("_"))
            )
        for value in actual_string_values:
            tokens = {token for token in value.upper().split("_") if token}
            self.assertTrue(
                forbidden_tokens.isdisjoint(tokens),
                f"forbidden decision token in {value!r}",
            )

        forbidden_unresolved_tokens = ("TO" + "DO", "T" + "BD", "PLACE" + "HOLDER")
        for path in [*AUDIT_DIR.iterdir(), Path(__file__)]:
            text = path.read_text(encoding="utf-8").upper()
            for token in forbidden_unresolved_tokens:
                self.assertNotIn(token, text, f"{path} contains {token}")


if __name__ == "__main__":
    unittest.main()
