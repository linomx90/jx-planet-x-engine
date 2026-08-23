from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jxplanetx.survey_selection_v4 import (
    EVIDENCE_RELATIONSHIP,
    EXPERIMENT_ID,
    OFFICIAL_BACKEND,
    POPULATION_SEED,
    RESAMPLING_SEED,
    SurveySelectionError,
    V2_POOL_SEMANTIC_SHA256,
    V2_POPULATION_SEED,
    V2_RESAMPLING_SEED,
    _require_independent_contract,
    independent_contract_audit,
    independent_pool_audit,
)


def contract(population_seed: str = POPULATION_SEED, resampling_seed: str = RESAMPLING_SEED) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "population": {"seed_key": population_seed},
        "statistics": {"resampling_seed": resampling_seed},
        "execution": {
            "official_backend_name": OFFICIAL_BACKEND,
            "checkpoint_restart_required": True,
        },
        "gates": {"exact_replay_required": True},
    }


class IndependentConfirmationTests(unittest.TestCase):
    def test_predeclared_v4_keys_pass(self) -> None:
        audit = independent_contract_audit(contract())
        self.assertTrue(audit["all_passed"])
        self.assertEqual(EVIDENCE_RELATIONSHIP, "INDEPENDENT_CONFIRMATION_WITH_FRESH_OFFICIAL_POOLS")

    def test_v2_population_seed_is_rejected(self) -> None:
        with self.assertRaises(SurveySelectionError) as context:
            _require_independent_contract(contract(population_seed=V2_POPULATION_SEED))
        self.assertEqual(context.exception.code, "not_independent_confirmation")

    def test_v2_resampling_seed_is_rejected(self) -> None:
        with self.assertRaises(SurveySelectionError) as context:
            _require_independent_contract(contract(resampling_seed=V2_RESAMPLING_SEED))
        self.assertEqual(context.exception.code, "not_independent_confirmation")

    def test_prior_semantic_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = root / "correct.json"
            wrong = root / "wrong.json"
            correct.write_text(
                json.dumps({"detection_semantic_sha256": V2_POOL_SEMANTIC_SHA256["correct"]}),
                encoding="utf-8",
            )
            wrong.write_text(
                json.dumps({"detection_semantic_sha256": "1" * 64}),
                encoding="utf-8",
            )
            with self.assertRaises(SurveySelectionError) as context:
                independent_pool_audit(correct, wrong)
            self.assertEqual(context.exception.code, "reused_official_pool")

    def test_fresh_distinct_pool_manifests_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = root / "correct.json"
            wrong = root / "wrong.json"
            correct.write_text(
                json.dumps({"detection_semantic_sha256": "2" * 64}),
                encoding="utf-8",
            )
            wrong.write_text(
                json.dumps({"detection_semantic_sha256": "3" * 64}),
                encoding="utf-8",
            )
            audit = independent_pool_audit(correct, wrong)
            self.assertTrue(audit["all_passed"])


if __name__ == "__main__":
    unittest.main()
