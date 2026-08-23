from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path


VERIFIER = (
    Path(__file__).parents[1]
    / "runs/survey_selection_o1/verify_independent_confirmation_v4.py"
)
VERIFIER_NAMESPACE = runpy.run_path(
    str(VERIFIER),
    run_name="jxplanetx_v4_replay_test",
)
_verify_existing_result = VERIFIER_NAMESPACE["_verify_existing_result"]
verify_v4_result = VERIFIER_NAMESPACE["verify_v4_result"]


class ExistingResultReplayTests(unittest.TestCase):
    def test_matching_existing_result_is_returned(self) -> None:
        expected = {"schema": "test", "verdict": "PASSED"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(_verify_existing_result(output, expected), expected)

    def test_mismatched_existing_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text(json.dumps({"verdict": "PASSED"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match deterministic replay"):
                _verify_existing_result(output, {"verdict": "INVALID"})

    def test_malformed_existing_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot verify immutable result"):
                _verify_existing_result(output, {"verdict": "PASSED"})

    def test_non_utf8_existing_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_bytes(b"\xff")
            with self.assertRaisesRegex(RuntimeError, "cannot verify immutable result"):
                _verify_existing_result(output, {"verdict": "PASSED"})

    def test_verifier_replays_manifests_before_accepting_result(self) -> None:
        expected = {"schema": "test", "verdict": "PASSED"}
        contract = Path("contract.json")
        correct = Path("correct.json")
        wrong = Path("wrong.json")
        calls: list[tuple[Path, Path, Path]] = []
        original = verify_v4_result.__globals__["replay_survey_selection"]

        def replay(
            contract_path: Path,
            correct_manifest_path: Path,
            wrong_manifest_path: Path,
        ) -> dict:
            calls.append((contract_path, correct_manifest_path, wrong_manifest_path))
            return expected

        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps(expected), encoding="utf-8")
            try:
                verify_v4_result.__globals__["replay_survey_selection"] = replay
                self.assertEqual(
                    verify_v4_result(contract, correct, wrong, result),
                    expected,
                )
            finally:
                verify_v4_result.__globals__["replay_survey_selection"] = original
        self.assertEqual(calls, [(contract, correct, wrong)])


if __name__ == "__main__":
    unittest.main()
