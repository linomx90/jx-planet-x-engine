from __future__ import annotations

from pathlib import Path
import unittest

from benchmarks import jx_gr15_eih_1pn_de440_attribution as attribution


ROOT = Path(__file__).resolve().parents[1]


class GR15EIH1PNDE440AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = attribution.run(ROOT)

    def test_locked_de440_attribution_gates_pass(self) -> None:
        self.assertEqual(self.report["status"], "PASS_SCREENING_ONLY")
        for assessment in self.report["acceptance"].values():
            self.assertTrue(assessment["passed"])
        self.assertEqual(
            [row["julian_year"] for row in self.report["checkpoint_results"]],
            [10, 25, 50, 100],
        )
        for row in self.report["checkpoint_results"]:
            self.assertLessEqual(
                row["aggregate_rms"]["ratio_to_newtonian"],
                0.25,
            )
            self.assertEqual(len(row["bodies"]), 10)

    def test_claim_ceiling_and_weak_field_diagnostics(self) -> None:
        claims = self.report["claim_controls"]
        self.assertFalse(claims["de440_equivalence_claimed"])
        self.assertFalse(claims["exact_general_relativity_claimed"])
        self.assertFalse(claims["general_superiority_claimed"])
        self.assertFalse(claims["navigation_or_production_authorized"])
        accounting = self.report["accounting"]["eih_1pn"]
        self.assertLess(accounting["observed_maximum_compactness"], 1.0e-4)
        self.assertLess(
            accounting["observed_maximum_speed_fraction_squared"],
            1.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
