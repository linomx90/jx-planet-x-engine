from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import jxplanetx
from benchmarks import jx_gr15_rc9_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "jx_gr15_rc9_acceptance_protocol.json"
PROTOCOL_SHA256 = (
    "87cc7fbc0b990321d5eed24a9abb6c76f202cda23d21dd7122badb0a06a9e940"
)


class GR15RC9ArtifactAcceptanceTests(unittest.TestCase):
    def test_unpublished_protocol_is_immutable_and_rejects_current_source(self) -> None:
        self.assertEqual(
            hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), PROTOCOL_SHA256
        )
        self.assertEqual(jxplanetx.__version__, "0.6.0rc16")
        with self.assertRaisesRegex(
            acceptance.RC9AcceptanceError, "source binding changed"
        ):
            acceptance._read_protocol(PROTOCOL, ROOT)

    def test_claim_ceiling_and_base_protocol_custody(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(
            protocol["claim_controls"]["general_superiority_claim_authorized"]
        )
        self.assertFalse(
            protocol["claim_controls"]["production_ephemeris_claim_authorized"]
        )
        self.assertFalse(
            protocol["claim_controls"]["publication_authorized_by_this_report"]
        )
        self.assertEqual(
            protocol["base_protocols"]["direct"]["sha256"],
            "ed7cfb0d42114c2a679a91d3e4cd634b9583837771b6a1a61cd9f561009afe83",
        )
        self.assertEqual(
            protocol["base_protocols"]["adversarial"]["sha256"],
            "13304170185aeda698d2af5c784c1aa9a478164c3ee72c8e19a9efa5bad3bf1d",
        )


if __name__ == "__main__":
    unittest.main()
