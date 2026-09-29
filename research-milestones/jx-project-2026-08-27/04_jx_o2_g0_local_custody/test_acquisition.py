from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import verify_acquisition as verifier


ROOT = Path(__file__).resolve().parent


class LocalAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checklist = verifier.strict_json(ROOT / "acquisition_checklist_v1.json")
        cls.custody = verifier.strict_json(ROOT / "local_custody_manifest_v1.json")
        cls.registration_sha256 = verifier.sha256_file(ROOT / "registration_v1.json")
        cls.verified = verifier.verify(ROOT, cls.registration_sha256)

    def test_package_verifies_and_remains_blocked(self) -> None:
        self.assertEqual(
            self.verified["status"], "JX_O2_G0_LOCAL_CUSTODY_VERIFIED_BLOCKED"
        )
        self.assertFalse(self.verified["g0_complete"])
        self.assertFalse(self.verified["eligible_for_g1"])
        self.assertFalse(self.verified["execution_authorized"])
        self.assertEqual(self.verified["evidence_file_count"], 25)
        self.assertEqual(self.verified["acceptance_requirement_count"], 31)

    def test_no_requirement_is_semantically_accepted(self) -> None:
        requirements = (
            self.checklist["survey_acceptance_requirements"]["ossos"]
            + self.checklist["survey_acceptance_requirements"]["des"]
            + self.checklist["matched_model_acceptance_requirements"]
        )
        self.assertNotIn("SEMANTICALLY_VERIFIED", {item["status"] for item in requirements})
        self.assertTrue(
            self.checklist["acceptance_policy"][
                "every_requirement_must_be_semantically_verified"
            ]
        )
        model_requirements = {
            item["requirement_id"]: item["required_evidence"]
            for item in self.checklist["matched_model_acceptance_requirements"]
        }
        self.assertEqual(
            set(model_requirements),
            {f"MODEL-A{index:02d}" for index in range(1, 16)},
        )
        required_terms = {
            "MODEL-A10": ("integration duration", "numerical tolerances"),
            "MODEL-A11": ("survival", "resonance"),
            "MODEL-A12": ("absolute-magnitude", "nuisance model"),
            "MODEL-A13": ("model-to-survey forward-adapter", "parity fixtures"),
            "MODEL-A14": ("phase and orientation uncertainty", "outcome-conditioned"),
            "MODEL-A15": ("hardware architecture", "reduction order"),
        }
        for requirement_id, terms in required_terms.items():
            for term in terms:
                self.assertIn(term, model_requirements[requirement_id])

    def test_compute_and_claim_permissions_are_false(self) -> None:
        permissions = self.checklist["permissions"]
        for key in (
            "activate_quarantined_input",
            "generate_or_select_physical_checkpoint",
            "realize_angle_grid_or_prior",
            "realize_random_seeds",
            "run_dynamics",
            "run_synthetic_calibration",
            "access_or_unblind_untouched_holdout",
            "compute_observed_statistic",
            "run_observed_model_comparison",
            "start_large_gpu_job",
            "create_g1_execution_contract",
            "create_activation_receipt",
        ):
            self.assertIs(permissions[key], False)
        self.assertFalse(any(self.checklist["claim_firewall"].values()))

    def test_ineligible_material_is_explicit(self) -> None:
        assertions = self.custody["custody_assertions"]
        self.assertFalse(assertions["complete_authoritative_ossos_selection_bundle_present"])
        self.assertFalse(assertions["withdrawn_2019_deep_surveys_payload_preserved_or_eligible"])
        self.assertFalse(assertions["physical_cluster_checkpoint_present"])
        self.assertFalse(assertions["paired_physical_m0_m1_decks_present"])
        self.assertFalse(assertions["candidate_9118_eligible_for_jx_o2"])
        records = {
            item["artifact_id"]: item for item in verifier.artifact_records(self.custody)
        }
        for artifact_id in (
            "A16_CDS_README",
            "HILAT_PUBLICATION_SOURCE_1608_02873",
        ):
            self.assertEqual(
                records[artifact_id]["activation_status"],
                "REFERENCE_ONLY_UNBOUND_LOCAL_BYTES",
            )
            self.assertEqual(
                records[artifact_id]["provenance_status"],
                "UNBOUND_LOCAL_COPY_REQUIRES_REACQUISITION",
            )

    def test_external_registration_anchor_is_required(self) -> None:
        with self.assertRaises(RuntimeError):
            verifier.verify(ROOT, "0" * 64)

    def test_every_custody_artifact_matches_bytes(self) -> None:
        for record in verifier.artifact_records(self.custody):
            path = ROOT / record["path"]
            self.assertEqual(path.stat().st_size, record["byte_count"])
            self.assertEqual(verifier.sha256_file(path), record["sha256"])

    def test_withdrawn_deep_survey_payload_is_absent(self) -> None:
        relative_paths = {
            record["path"] for record in verifier.artifact_records(self.custody)
        }
        self.assertFalse(any("Deep_Surveys" in path for path in relative_paths))
        self.assertFalse(any("2015AM.eff" in path for path in relative_paths))

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        cases = (
            '{"x": 1, "x": 2}',
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": 1e400}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for payload in cases:
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    verifier.strict_json(path)


if __name__ == "__main__":
    unittest.main()
