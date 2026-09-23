from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from benchmarks import jx_gr15_v3_adversarial as qualification


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "jx_gr15_v3_adversarial_protocol.json"
PROTOCOL_SHA256 = (
    "13304170185aeda698d2af5c784c1aa9a478164c3ee72c8e19a9efa5bad3bf1d"
)


class GR15V3AdversarialQualificationTests(unittest.TestCase):
    def test_protocol_identity_scope_and_source_binding(self) -> None:
        self.assertEqual(
            hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), PROTOCOL_SHA256
        )
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(tuple(protocol["case_order"]), qualification.REQUIRED_CASES)
        self.assertEqual(protocol["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(
            protocol["claim_controls"]["general_superiority_claim_authorized"]
        )
        self.assertFalse(
            protocol["claim_controls"]["performance_claim_authorized"]
        )
        self.assertFalse(
            protocol["claim_controls"]["production_ephemeris_claim_authorized"]
        )

    def test_locked_fixtures_cover_declared_body_domain(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        observed_counts = []
        for name in protocol["case_order"]:
            with self.subTest(case=name):
                positions, velocities, gm, epochs, _ = qualification._fixture(
                    protocol["cases"][name]
                )
                observed_counts.append(len(gm))
                self.assertEqual(positions.shape, (len(gm), 3))
                self.assertEqual(velocities.shape, positions.shape)
                self.assertTrue(np.all(np.isfinite(positions)))
                self.assertTrue(np.all(np.isfinite(velocities)))
                self.assertTrue(np.all(np.isfinite(gm)))
                self.assertTrue(np.all(gm > 0.0))
                self.assertGreater(len(epochs), 1)
                self.assertTrue(
                    all(right > left for left, right in zip(epochs, epochs[1:]))
                )
        self.assertEqual(observed_counts, [2, 2, 3, 11, 32])

    def test_rc8_development_protocol_rejects_the_rc9_tree(self) -> None:
        with self.assertRaisesRegex(
            qualification.AdversarialQualificationError, "source binding changed"
        ):
            qualification._read_protocol(PROTOCOL, ROOT)

    def test_rc10_regression_portfolio_passes_without_elevating_claims(self) -> None:
        try:
            import rebound
        except ModuleNotFoundError:
            self.skipTest("REBOUND 5.1.1 is not installed")
        if (
            rebound.__version__ != "5.1.1"
            or rebound.__githash__
            != "33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f"
        ):
            self.skipTest("requires the exact REBOUND 5.1.1 tagged runtime")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["package_version"] = "0.6.0rc10"
        report = qualification.run(protocol)
        self.assertEqual(report["status"], "PASS_SCREENING_ONLY")
        self.assertEqual(report["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(
            report["claim_controls"]["release_promotion_authorized_by_this_report_alone"]
        )
        for name in qualification.REQUIRED_CASES:
            case = report["cases"][name]
            self.assertEqual(case["status"], "PASS_FIXED_GATES")
            self.assertTrue(all(case["gate_results"].values()))
            self.assertEqual(case["jx_digest"], case["jx_repeat_digest"])
        self.assertEqual(report["failure_domain"]["status"], "PASS_FIXED_GATES")
        self.assertTrue(all(report["failure_domain"]["gates"].values()))


if __name__ == "__main__":
    unittest.main()
