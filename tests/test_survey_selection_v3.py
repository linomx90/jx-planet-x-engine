from __future__ import annotations

import math
import unittest

from jxplanetx.survey_selection_v3 import (
    _evaluate_statistics,
    exact_zeta_moments,
)


def statistics_contract(correct_rows: list[dict], catalog_size: int = 2) -> dict:
    mean, sd = exact_zeta_moments(
        correct_rows,
        {
            "statistics": {"primary_pit_variable": "q_AU"},
            "population": {"catalog_size": catalog_size},
        },
    )
    return {
        "population": {"catalog_size": catalog_size},
        "statistics": {
            "variables": ["q_AU"],
            "primary_pit_variable": "q_AU",
            "mock_catalogs": 40,
            "bootstrap_reference_catalogs": 40,
            "alpha": 0.05,
            "zeta_expected_mean": mean,
            "zeta_expected_sd": sd,
            "resampling_seed": "v3-test-seed",
        },
        "gates": {
            "correct_model_false_rejection_rate_min": 0.0,
            "correct_model_false_rejection_rate_max": 1.0,
            "zeta_mean_absolute_tolerance": 1e-14,
            "zeta_sd_absolute_tolerance": 1e-14,
            "wrong_model_rejection_power_min": 0.0,
        },
    }


class ExactZetaTests(unittest.TestCase):
    def test_exact_finite_pool_replacement_formula(self) -> None:
        rows = [{"q_AU": value} for value in (1.0, 2.0, 3.0, 4.0)]
        contract = {
            "statistics": {"primary_pit_variable": "q_AU"},
            "population": {"catalog_size": 2},
        }
        mean, sd = exact_zeta_moments(rows, contract)
        logs = [math.log10(value) for value in (0.125, 0.375, 0.625, 0.875)]
        expected_mean = 2.0 * sum(logs) / 4.0
        expected_variance = sum((value - sum(logs) / 4.0) ** 2 for value in logs) / 4.0
        self.assertAlmostEqual(mean, expected_mean, places=15)
        self.assertAlmostEqual(sd, math.sqrt(2.0 * expected_variance), places=15)

    def test_exact_moments_are_permutation_invariant(self) -> None:
        rows = [{"q_AU": value} for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
        contract = {
            "statistics": {"primary_pit_variable": "q_AU"},
            "population": {"catalog_size": 3},
        }
        self.assertEqual(
            exact_zeta_moments(rows, contract),
            exact_zeta_moments(list(reversed(rows)), contract),
        )

    def test_evaluator_uses_exact_moments_for_locked_gates(self) -> None:
        correct = [{"q_AU": float(value)} for value in range(1, 21)]
        wrong = [{"q_AU": float(value)} for value in range(11, 31)]
        contract = statistics_contract(correct)
        result = _evaluate_statistics(correct, wrong, contract, label="exact-test")
        self.assertEqual(
            result["zeta_estimator"],
            "exact_finite_empirical_pool_with_replacement",
        )
        self.assertTrue(result["gate_results"]["zeta_mean"])
        self.assertTrue(result["gate_results"]["zeta_sd"])
        self.assertIn("zeta_monte_carlo_diagnostic_sd", result)

    def test_exact_evaluator_replay_is_identical(self) -> None:
        correct = [{"q_AU": float(value)} for value in range(1, 21)]
        wrong = [{"q_AU": float(value)} for value in range(11, 31)]
        contract = statistics_contract(correct)
        first = _evaluate_statistics(correct, wrong, contract, label="replay")
        second = _evaluate_statistics(correct, wrong, contract, label="replay")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

