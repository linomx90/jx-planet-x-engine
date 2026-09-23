from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from benchmarks import jx_gr15_ias15 as comparison


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "jx_gr15_v3_ias15_protocol.json"
PROTOCOL_SHA256 = "ed7cfb0d42114c2a679a91d3e4cd634b9583837771b6a1a61cd9f561009afe83"


class PublicGR15IAS15ComparisonTests(unittest.TestCase):
    def test_locked_protocol_and_claim_boundary(self) -> None:
        self.assertEqual(
            hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), PROTOCOL_SHA256
        )
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(
            protocol["claim_controls"]["general_superiority_claim_authorized"]
        )
        self.assertFalse(
            protocol["claim_controls"]["production_qualification_authorized"]
        )
        self.assertEqual(
            set(protocol["workloads"]),
            {
                "circular_binary_100_periods_endpoint",
                "circular_binary_100_periods_101_outputs",
                "eccentric_binary_e_0_9_one_period_17_outputs",
            },
        )

    def test_analytic_reference_is_barycentric_and_periodic(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        workload = protocol["workloads"][
            "eccentric_binary_e_0_9_one_period_17_outputs"
        ]
        positions, velocities, gm, epochs = comparison._fixture(workload)
        reference_positions, reference_velocities = comparison._analytic_binary(
            epochs, gm, float(workload["eccentricity"])
        )
        np.testing.assert_array_equal(positions, reference_positions[0])
        np.testing.assert_array_equal(velocities, reference_velocities[0])
        np.testing.assert_allclose(
            np.sum(gm[:, None] * reference_positions, axis=1), 0.0, atol=1.0e-20
        )
        np.testing.assert_allclose(
            np.sum(gm[:, None] * reference_velocities, axis=1), 0.0, atol=1.0e-20
        )
        self.assertLessEqual(
            float(np.max(np.abs(reference_positions[-1] - reference_positions[0]))),
            2.0e-15,
        )
        self.assertLessEqual(
            float(np.max(np.abs(reference_velocities[-1] - reference_velocities[0]))),
            3.0e-15,
        )
        self.assertAlmostEqual(epochs[-1], 2.0 * math.pi / math.sqrt(float(sum(gm))))

    def test_rc8_development_protocol_rejects_the_rc9_tree(self) -> None:
        with self.assertRaisesRegex(comparison.ComparisonError, "source binding changed"):
            comparison._read_protocol(PROTOCOL, ROOT)


if __name__ == "__main__":
    unittest.main()
