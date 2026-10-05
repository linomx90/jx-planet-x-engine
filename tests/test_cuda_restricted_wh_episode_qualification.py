from __future__ import annotations

import unittest

from benchmarks import jx_cuda_restricted_wh_episode_qualification as qualification


class CUDARestrictedWHEpisodeQualificationContractTests(unittest.TestCase):
    def test_profile_is_bounded_and_uses_six_physical_workers(self):
        self.assertEqual(qualification.ROUTED_LANES, 48)
        self.assertEqual(qualification.QUIET_LANES, 16)
        self.assertEqual(qualification.STEP_COUNT, 8)
        self.assertEqual(qualification.ROUTE_WORKER_COUNT, 6)
        self.assertEqual(qualification.PHYSICAL_CPU_AFFINITY, (0, 1, 2, 3, 4, 5))
        self.assertEqual(qualification.REPEATS, 3)
        self.assertEqual(qualification.EXPECTED_ROUTE_COUNT, 384)
        self.assertEqual(qualification.EXPECTED_EPISODE_COUNT, 48)
        self.assertGreaterEqual(
            qualification.MINIMUM_EPISODE_VS_REFERENCE_SPEEDUP, 1.0
        )

    def test_every_claim_is_structurally_false(self):
        self.assertTrue(qualification.CLAIMS)
        self.assertTrue(all(value is False for value in qualification.CLAIMS.values()))

    def test_content_hash_is_domain_separated_and_order_independent(self):
        first = qualification._content_record(
            {"schema": "test", "b": 2, "a": 1}, b"domain-a\0"
        )
        second = qualification._content_record(
            {"a": 1, "schema": "test", "b": 2}, b"domain-a\0"
        )
        third = qualification._content_record(
            {"a": 1, "schema": "test", "b": 2}, b"domain-b\0"
        )
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        self.assertNotEqual(first["content_sha256"], third["content_sha256"])

    def test_cv_is_sample_standard_deviation_over_mean(self):
        self.assertEqual(qualification._cv([1.0]), 0.0)
        self.assertAlmostEqual(qualification._cv([1.0, 2.0, 3.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
