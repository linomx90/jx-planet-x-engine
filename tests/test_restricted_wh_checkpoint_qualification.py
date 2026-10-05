from __future__ import annotations

import hashlib
import unittest

from benchmarks import jx_restricted_wh_checkpoint_qualification as qualification


class RestrictedWHCheckpointQualificationContractTests(unittest.TestCase):
    def test_profile_is_the_fixed_bounded_six_worker_screen(self):
        self.assertEqual(qualification.ROUTED_LANES, 256)
        self.assertEqual(qualification.SEGMENT_STEPS, 2)
        self.assertEqual(qualification.SEGMENT_COUNT, 2)
        self.assertEqual(qualification.TOTAL_STEPS, 4)
        self.assertEqual(qualification.ROUTE_WORKER_COUNT, 6)
        self.assertEqual(qualification.PHYSICAL_CPU_AFFINITY, (0, 1, 2, 3, 4, 5))
        self.assertEqual(qualification.SCALING_REPEATS, 3)
        self.assertEqual(qualification.MINIMUM_SIX_WORKER_SPEEDUP, 2.0)
        self.assertTrue(
            all(
                qualification.os.environ[name] == "1"
                for name in qualification.NUMERICAL_THREAD_ENVIRONMENT
            )
        )
        self.assertEqual(qualification.EXPECTED_ROUTE_COUNT, 1024)
        self.assertEqual(qualification.EXPECTED_ROUTES_PER_SEGMENT, 512)
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
        self.assertEqual(len(bytes.fromhex(first["content_sha256"])), 32)

    def test_route_digest_is_canonical_and_order_sensitive(self):
        rows = [{"global_step": 0, "tracer": 1}, {"global_step": 1, "tracer": 1}]
        expected = qualification._route_digest(rows)
        self.assertEqual(expected, qualification._route_digest(list(rows)))
        self.assertNotEqual(expected, qualification._route_digest(list(reversed(rows))))
        self.assertEqual(len(bytes.fromhex(expected)), hashlib.sha256().digest_size)


if __name__ == "__main__":
    unittest.main()
