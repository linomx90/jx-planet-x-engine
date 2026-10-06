from __future__ import annotations

import hashlib
import unittest

from benchmarks import jx_cuda_restricted_wh_episode_owner_qualification as q


class CUDARestrictedWHOwnedEpisodeQualificationContractTests(unittest.TestCase):
    def test_profile_is_fixed_to_long_observed_episode_scale(self):
        self.assertEqual(q.ROUTED_LANES, 6)
        self.assertEqual(q.QUIET_LANES, 6)
        self.assertEqual(q.STEP_COUNT, 196)
        self.assertEqual(q.ROUTE_WORKER_COUNT, 6)
        self.assertEqual(q.ROUTE_LOOKAHEAD_STEPS, 32)
        self.assertEqual(q.PHYSICAL_CPU_AFFINITY, (0, 1, 2, 3, 4, 5))
        self.assertEqual(q.EXPECTED_ROUTE_COUNT, 1176)
        self.assertEqual(q.EXPECTED_EPISODE_COUNT, 6)
        self.assertEqual(q.EXPECTED_CHAIN_SUBMISSIONS_PER_LANE, 8)
        self.assertEqual(q.EXPECTED_CHAIN_SUBMISSIONS, 48)
        self.assertEqual(q.EXPECTED_CACHE_HITS, 1128)

    def test_claim_ceiling_is_all_false(self):
        self.assertTrue(q.CLAIMS)
        self.assertFalse(any(q.CLAIMS.values()))

    def test_content_record_hash_is_domain_separated_and_stable(self):
        record = {"schema": "test", "value": [1, 2, 3]}
        first = q._content_record(record, q.RESULT_DOMAIN)
        second = q._content_record(record, q.RESULT_DOMAIN)
        self.assertEqual(first, second)
        self.assertEqual(
            first["content_sha256"],
            hashlib.sha256(
                q.RESULT_DOMAIN
                + q._canonical_bytes({"schema": "test", "value": [1, 2, 3]})
            ).hexdigest(),
        )
        self.assertNotEqual(
            first["content_sha256"],
            q._content_record(record, q.PROTOCOL_DOMAIN)["content_sha256"],
        )

    def test_sample_cv_uses_sample_standard_deviation(self):
        self.assertEqual(q._cv([2.0]), 0.0)
        self.assertAlmostEqual(q._cv([1.0, 2.0, 3.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
