from __future__ import annotations

import unittest

import benchmarks.jx_restricted_wh_episode_delta_checkpoint_qualification as q


class OwnedEpisodeDeltaQualificationContractTests(unittest.TestCase):
    def test_frozen_route_heavy_profile_uses_six_physical_cores(self):
        self.assertEqual(q.PHYSICAL_CPU_AFFINITY, (0, 1, 2, 3, 4, 5))
        self.assertEqual(q.ROUTE_WORKER_COUNT, 6)
        self.assertEqual(q.ROUTE_LOOKAHEAD_STEPS, 16)
        self.assertEqual(q.LANE_COUNT, 6)
        self.assertEqual(q.SEGMENT_STEP_COUNT, 16)
        self.assertEqual(q.SEGMENT_COUNT, 2)
        self.assertEqual(q.EXPECTED_ROUTE_DELTA, 96)
        self.assertEqual(q.EXPECTED_ROUTE_TOTAL, 192)

    def test_claim_ceiling_is_all_false(self):
        self.assertTrue(q.CLAIMS)
        self.assertFalse(any(q.CLAIMS.values()))

    def test_content_hash_is_domain_separated_and_stable(self):
        source = {"schema": q.SCHEMA, "value": [False, 1, "x"]}
        first = q._content_record(source, q.PROTOCOL_DOMAIN)
        second = q._content_record(first, q.PROTOCOL_DOMAIN)
        other = q._content_record(source, q.RESULT_DOMAIN)
        self.assertEqual(first, second)
        self.assertNotEqual(first["content_sha256"], other["content_sha256"])

    def test_protocol_requires_delta_replay_gates(self):
        protocol = q._protocol()
        self.assertEqual(protocol["schema"], q.SCHEMA)
        self.assertFalse(any(protocol["claims"].values()))
        self.assertTrue(all(protocol["required_gates"].values()))
        self.assertEqual(
            protocol["fixture"]["expected_route_total"], q.EXPECTED_ROUTE_TOTAL
        )

    def test_imported_numerical_sources_match_current_disk_bytes(self):
        imported = q._in_memory_source_hashes()
        disk = q._current_source_hashes()
        self.assertEqual(
            imported, {key: disk[key] for key in imported}
        )


if __name__ == "__main__":
    unittest.main()
