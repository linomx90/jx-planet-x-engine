from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import jxplanetx
from benchmarks import jx_gr15_rc10_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "jx_gr15_rc10_acceptance_protocol.json"
PROTOCOL_SHA256 = (
    "cd3fa44f20779a2ef06d823a2cae0b20755e31b04ea9ea2edcbfe994d6645b7b"
)


class GR15RC10ArtifactAcceptanceTests(unittest.TestCase):
    def test_protocol_is_source_and_artifact_bound(self) -> None:
        self.assertEqual(
            hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), PROTOCOL_SHA256
        )
        protocol, identity = acceptance._read_protocol(PROTOCOL, ROOT)
        self.assertEqual(identity["sha256"], PROTOCOL_SHA256)
        self.assertEqual(jxplanetx.__version__, "0.6.0rc10")
        self.assertEqual(protocol["package_version"], jxplanetx.__version__)
        self.assertEqual(
            protocol["artifacts"]["wheel"]["sha256"],
            "01bff6e4f418268a0acd0954e91f201397c8e62b3ef84852d332282aed13fa7e",
        )
        self.assertEqual(
            protocol["artifacts"]["sdist"]["sha256"],
            "af390c558aa46eefa869edf2062a81869ef0d0147588467ce5af2f61c7b07e5a",
        )

    def test_claim_ceiling_and_base_protocol_custody(self) -> None:
        protocol, _ = acceptance._read_protocol(PROTOCOL, ROOT)
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
