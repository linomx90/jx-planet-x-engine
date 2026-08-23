from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from jxplanetx.provenance import sha256_data
from jxplanetx.survey_selection import (
    CONTRACT_SCHEMA,
    SurveySelectionError,
    anderson_darling_uniform,
    empirical_pit,
    finalize_survey_selection,
    generate_intrinsic_population,
    load_detection_csv,
    parse_ossos_tracked_file,
    validate_survey_contract,
    write_detection_csv,
    write_ossos_model_file,
    write_pool_manifest,
)


def contract_fixture(*, seed_blocks: int = 2, catalog_size: int = 5) -> dict:
    return {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": "jx-o1-test",
        "registration_status": "PRELOCKED_BEFORE_ANY_JX_O1_OUTCOMES",
        "milestone": "JX-O1_TELESCOPE_SELECTION_VALIDATION",
        "external_simulator": {
            "repository_url": "https://github.com/OSSOS/SurveySimulator",
            "commit": "1" * 40,
            "license": "EUPL-1.1",
            "execution_mode": "external_process_not_vendored",
            "source_subdirectory": "F95",
            "characterization_path": "Surveys/OSSOS",
            "characterization_scope": "test fixture",
            "required_files": {"README.md": "2" * 64},
        },
        "population": {
            "epoch_jd": 2453157.5,
            "seed_key": "jx-o1-test-seed",
            "seed_blocks": seed_blocks,
            "minimum_intrinsic_draws_per_model": 100,
            "minimum_intrinsic_draws_per_seed_block": 50,
            "minimum_tracked_detections_per_model": 100,
            "minimum_tracked_detections_per_seed_block": 50,
            "catalog_size": catalog_size,
            "correct_model": {
                "q_distribution": "uniform",
                "q_min_AU": 15.0,
                "q_max_AU": 30.0,
            },
            "wrong_model": {
                "q_distribution": "beta",
                "beta_alpha": 5.0,
                "beta_beta": 1.0,
                "q_min_AU": 15.0,
                "q_max_AU": 30.0,
                "interpretation": "phenomenological Neptune-barrier alternative",
            },
            "shared": {
                "a_distribution": "power_law",
                "a_power": -1.5,
                "a_min_AU": 100.0,
                "a_max_AU": 1000.0,
                "inclination_distribution": "truncated_half_normal",
                "inclination_sigma_deg": 15.0,
                "inclination_min_deg": 0.0,
                "inclination_max_deg": 40.0,
                "angular_distribution": "independent_uniform_0_360_deg",
                "H_distribution": "lawler_2018_divot",
                "H_min": 5.0,
                "H_break": 8.3,
                "H_max": 12.0,
                "alpha_bright": 0.9,
                "alpha_faint": 0.5,
                "contrast": 3.2,
                "colors_r_reference": [0.7, 0.0, -0.5, -1.0, 1.5, 0.4, 0.8, -0.4, -0.8, 0.0],
            },
        },
        "statistics": {
            "variables": ["a_AU", "q_AU", "i_deg", "H_r", "r_AU", "m_r"],
            "primary_pit_variable": "q_AU",
            "mock_catalogs": 40,
            "bootstrap_reference_catalogs": 40,
            "seed_stability_catalogs": 40,
            "alpha": 0.05,
            "zeta_expected_mean": -catalog_size / math.log(10.0),
            "zeta_expected_sd": math.sqrt(catalog_size) / math.log(10.0),
            "resampling_seed": "jx-o1-test-resampling",
        },
        "gates": {
            "correct_model_false_rejection_rate_min": 0.0,
            "correct_model_false_rejection_rate_max": 1.0,
            "zeta_mean_absolute_tolerance": 100.0,
            "zeta_sd_absolute_tolerance": 100.0,
            "wrong_model_rejection_power_min": 0.0,
            "seed_block_verdict_stability_required": True,
            "adapter_identity_required": True,
            "exact_replay_required": True,
            "max_missing_records": 0,
            "max_nonfinite_records": 0,
        },
        "execution": {
            "checkpoint_restart_required": True,
            "immutable_result": True,
            "pilot_seeds_excluded_from_final": True,
            "official_backend_name": "OSSOS_SURVEY_SIMULATOR_F95",
            "pilot_backend_name": "JX_ANALYTIC_SELECTION_PILOT",
        },
        "scientific_scope": "test telescope-selection calibration",
        "limitations": ["fixture only"],
        "nonclaim": "not a Planet X detection",
        "allowed_verdicts": {
            "PASSED": "all gates pass",
            "BLOCKED": "required scale or evidence is incomplete",
            "INVALID": "integrity or calibration failed",
        },
        "locked_files": {"module": {"path": "locked.py", "sha256": "3" * 64}},
    }


def detection_rows(model_id: str, seed_blocks: int = 2, per_block: int = 50) -> list[dict]:
    rows = []
    prefix = "c" if model_id == "correct" else "w"
    total = seed_blocks * per_block
    for block in range(seed_blocks):
        for index in range(per_block):
            rank = block * per_block + index
            fraction = (rank + 0.5) / total
            q_fraction = fraction if model_id == "correct" else fraction ** 0.2
            q = 15.0 + 15.0 * q_fraction
            rows.append(
                {
                    "object_id": f"{prefix}{block:02d}{index:08d}",
                    "model_id": model_id,
                    "seed_block": block,
                    "a_AU": 100.0 + rank,
                    "q_AU": q,
                    "i_deg": 1.0 + 0.1 * rank,
                    "H_r": 6.0 + 0.01 * rank,
                    "r_AU": q + 1.0,
                    "m_r": 21.0 + 0.02 * rank,
                }
            )
    return rows


class ContractAndPopulationTests(unittest.TestCase):
    def test_contract_accepts_only_prelocked_schema(self) -> None:
        locked = validate_survey_contract(contract_fixture())
        self.assertEqual(locked["schema"], CONTRACT_SCHEMA)
        broken = contract_fixture()
        broken["unexpected"] = True
        with self.assertRaisesRegex(SurveySelectionError, "unknown fields"):
            validate_survey_contract(broken)

    def test_population_is_reproducible_and_paired_except_q_and_identity(self) -> None:
        contract = contract_fixture()
        first = generate_intrinsic_population(contract, "correct", 0, 20)
        replay = generate_intrinsic_population(contract, "correct", 0, 20)
        wrong = generate_intrinsic_population(contract, "wrong", 0, 20)
        self.assertEqual(first, replay)
        for correct_row, wrong_row in zip(first, wrong, strict=True):
            self.assertNotEqual(correct_row["object_id"], wrong_row["object_id"])
            self.assertLess(correct_row["q_AU"], wrong_row["q_AU"])
            for key in ("a_AU", "i_deg", "node_deg", "peri_deg", "mean_anomaly_deg", "H_r"):
                self.assertEqual(correct_row[key], wrong_row[key])

    def test_generated_support_and_bound_orbits(self) -> None:
        rows = generate_intrinsic_population(contract_fixture(), "correct", 1, 500)
        for row in rows:
            self.assertLessEqual(15.0, row["q_AU"])
            self.assertLessEqual(row["q_AU"], 30.0)
            self.assertLessEqual(100.0, row["a_AU"])
            self.assertLessEqual(row["a_AU"], 1000.0)
            self.assertLessEqual(0.0, row["e"])
            self.assertLess(row["e"], 1.0)
            self.assertLessEqual(0.0, row["i_deg"])
            self.assertLessEqual(row["i_deg"], 40.0)
            self.assertLessEqual(5.0, row["H_r"])
            self.assertLessEqual(row["H_r"], 12.0)

    def test_ossos_model_writer_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first.dat"
            second = root / "second.dat"
            write_ossos_model_file(first, contract_fixture(), "correct", 0, 10)
            write_ossos_model_file(second, contract_fixture(), "correct", 0, 10)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            lines = first.read_text().splitlines()
            self.assertTrue(lines[0].startswith("# Epoch of elements: JD ="))
            self.assertEqual(len(lines[3].split()), 8)


class AdapterTests(unittest.TestCase):
    def test_detection_csv_roundtrip_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detections.csv"
            rows = detection_rows("correct", per_block=3)
            metadata = write_detection_csv(path, reversed(rows))
            loaded = load_detection_csv(path)
            self.assertEqual(metadata["semantic_sha256"], sha256_data(loaded))
            self.assertEqual(loaded, sorted(loaded, key=lambda row: row["object_id"]))

    def test_duplicate_detection_identity_is_rejected(self) -> None:
        rows = detection_rows("correct", per_block=2)
        rows.append(dict(rows[0]))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SurveySelectionError, "duplicate detection ID"):
                write_detection_csv(Path(folder) / "duplicate.csv", rows)

    def test_official_tracked_output_parser_maps_documented_columns(self) -> None:
        line = (
            "120.000 0.800 15.000 24.000 25.000 2.000 3.000 4.000 "
            "23.10 7.20 0.00 4 24.0 23.0 7.1 0.8 10.0 -5.0 2013AE c0000000001\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracked.dat"
            path.write_text("# header\n" + line, encoding="utf-8")
            rows = parse_ossos_tracked_file(path, "correct", 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_id"], "c0000000001")
            self.assertEqual(rows[0]["q_AU"], 24.0)
            self.assertEqual(rows[0]["H_r"], 7.2)

    def test_untracked_official_row_is_rejected(self) -> None:
        line = (
            "120 0.8 15 24 25 2 3 4 23.1 7.2 0 3 24 23 7.1 0.8 10 -5 2013AE c0\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracked.dat"
            path.write_text(line, encoding="utf-8")
            with self.assertRaisesRegex(SurveySelectionError, "not a tracked detection"):
                parse_ossos_tracked_file(path, "correct", 0)


class StatisticsTests(unittest.TestCase):
    def test_empirical_pit_has_exact_midrank_grid(self) -> None:
        model = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual([empirical_pit(value, model) for value in model], [0.125, 0.375, 0.625, 0.875])

    def test_tied_empirical_pit_uses_midrank(self) -> None:
        model = [1.0, 2.0, 2.0, 4.0]
        self.assertEqual(empirical_pit(2.0, model), 0.5)

    def test_ad_statistic_is_permutation_invariant_and_tail_sensitive(self) -> None:
        centered = [0.1, 0.3, 0.5, 0.7, 0.9]
        tail = [0.001, 0.002, 0.003, 0.004, 0.005]
        self.assertEqual(anderson_darling_uniform(centered), anderson_darling_uniform(reversed(centered)))
        self.assertGreater(anderson_darling_uniform(tail), anderson_darling_uniform(centered))


class FinalizationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        contract = contract_fixture()
        locked = root / "locked.py"
        locked.write_text("# locked test fixture\n", encoding="utf-8")
        from jxplanetx.provenance import sha256_file

        contract["locked_files"]["module"]["sha256"] = sha256_file(locked)
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests = []
        for model_id in ("correct", "wrong"):
            detections = root / f"{model_id}.csv"
            manifest = root / f"{model_id}.json"
            write_detection_csv(detections, detection_rows(model_id))
            write_pool_manifest(
                manifest,
                model_id=model_id,
                backend=contract["execution"]["pilot_backend_name"],
                simulator_commit=None,
                detections_path=detections,
                intrinsic_draws_by_block={0: 50, 1: 50},
                raw_tracked_files=[],
                checkpoint_replay_passed=False,
            )
            manifests.append(manifest)
        return contract_path, manifests[0], manifests[1]

    def test_complete_analytic_pipeline_is_blocked_not_passed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, correct, wrong = self._fixture(root)
            result = finalize_survey_selection(contract, correct, wrong, root / "result.json")
            self.assertEqual(result["verdict"], "BLOCKED")
            self.assertTrue(result["statistics"]["calibration_passed"])
            self.assertTrue(result["exact_replay_passed"])
            codes = {item["code"] for item in result["blocked_reasons"]}
            self.assertIn("correct_official_backend_missing", codes)
            self.assertIn("wrong_adapter_identity_unproven", codes)

    def test_repeated_finalization_is_semantically_exact(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, correct, wrong = self._fixture(root)
            first = finalize_survey_selection(contract, correct, wrong, root / "first.json")
            second = finalize_survey_selection(contract, correct, wrong, root / "second.json")
            self.assertEqual(first, second)
            self.assertEqual(first["replay_sha256"], second["replay_sha256"])

    def test_tampered_detection_file_returns_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, correct, wrong = self._fixture(root)
            detection_path = root / "correct.csv"
            detection_path.write_text(detection_path.read_text() + "\n", encoding="utf-8")
            result = finalize_survey_selection(contract, correct, wrong, root / "invalid.json")
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["invalid_reasons"][0]["code"], "detection_hash_mismatch")

    def test_result_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, correct, wrong = self._fixture(root)
            output = root / "result.json"
            finalize_survey_selection(contract, correct, wrong, output)
            with self.assertRaisesRegex(SurveySelectionError, "refusing to overwrite"):
                finalize_survey_selection(contract, correct, wrong, output)


if __name__ == "__main__":
    unittest.main()
