from __future__ import annotations

import json
import unittest
from pathlib import Path

from jxplanetx.de441_horizons import (
    compare_state_sets,
    load_gm,
    load_reference,
    verdict_from_gates,
    verify_locked_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "de441_horizons_10yr"


class De441HorizonsTests(unittest.TestCase):
    def test_locked_reference_has_complete_grid(self) -> None:
        epochs, reference = load_reference(RUN_ROOT / "reference" / "horizons_de441_vectors.csv")
        self.assertEqual(len(epochs), 11)
        self.assertEqual(epochs[0], 2461200.5)
        self.assertEqual(epochs[-1], 2464853.0)
        self.assertTrue(all(tuple(sorted(reference[epoch])) == tuple(range(1, 11)) for epoch in epochs))

    def test_gm_table_has_all_major_barycenters(self) -> None:
        gm = load_gm(RUN_ROOT / "gm_de440_major_barycenters.csv")
        self.assertEqual(tuple(sorted(gm)), tuple(range(1, 11)))
        self.assertGreater(gm[10]["gm_au3_day2"], gm[5]["gm_au3_day2"])

    def test_identical_states_have_zero_heliocentric_residual(self) -> None:
        epochs, reference = load_reference(RUN_ROOT / "reference" / "horizons_de441_vectors.csv")
        rows, summary = compare_state_sets(reference, reference, epochs)
        self.assertTrue(all(row["position_residual_au"] == 0.0 for row in rows))
        self.assertTrue(all(row["velocity_residual_au_per_day"] == 0.0 for row in rows))
        self.assertTrue(all(record["max_position_residual_au"] == 0.0 for record in summary.values()))

    def test_contract_and_all_locked_inputs_verify(self) -> None:
        contract = json.loads((RUN_ROOT / "contract_v1.json").read_text(encoding="utf-8"))
        verified = verify_locked_inputs(contract, PROJECT_ROOT)
        self.assertEqual(verified["epochs"][0], 2461200.5)
        self.assertEqual(verified["manifest"]["ephemeris_source_required"], "DE441")

    def test_verdict_is_fail_closed(self) -> None:
        self.assertEqual(verdict_from_gates({"a": {"passed": True}}), "PASSED")
        self.assertEqual(verdict_from_gates({"a": {"passed": False}}), "INVALID")
        self.assertEqual(verdict_from_gates({}), "INVALID")


if __name__ == "__main__":
    unittest.main()
