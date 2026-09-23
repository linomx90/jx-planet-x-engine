from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from jxplanetx import __version__
from jxplanetx import fast_wisdom_holman as fast_wh

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JXRC6ReleaseCandidateTests(unittest.TestCase):
    def test_exact_rc6_protocol_is_source_bound(self) -> None:
        path = (
            ROOT
            / "benchmarks/jx_fast_wisdom_holman_rc6_rebound_race_protocol.json"
        )
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["required_jx_version"], "0.6.0rc6")
        self.assertEqual(protocol["periods"], 100)
        self.assertEqual(protocol["timed_repetitions"], 9)
        self.assertEqual(
            _sha256(path),
            "788241795b57433cd0e4d3282bfffa4a6e079eb9bf755f0aeeb38dec4b48ffe2",
        )

    def test_rc6_manifest_binds_api_artifacts_and_claim_limits(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc6.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["version"], "0.6.0rc6")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["fast_wisdom_holman"]["api_id"],
            "jx.fast-wisdom-holman.screening-api.v1",
        )
        self.assertTrue(
            manifest["fast_wisdom_holman"]["default_exact_replay"]
        )
        self.assertFalse(
            manifest["fast_wisdom_holman"]["production_authorized"]
        )
        race_result = manifest["verification"]["exact_wheel_cpu_rebound_race"]
        self.assertEqual(
            race_result["required_wheel_sha256"],
            manifest["reproducible_packaging"]["artifacts"]["wheel"]["sha256"],
        )
        self.assertTrue(race_result["public_jx_100_period_bitwise_parity"])
        aggregate = manifest["verification"]["cuda_two_machine_aggregate"]
        self.assertEqual(aggregate["machine_count"], 2)
        self.assertTrue(
            aggregate["accuracy_results_bitwise_identical_across_machines"]
        )
        self.assertFalse(aggregate["organizational_independence_claimed"])
        self.assertFalse(manifest["release_actions"]["publication_performed"])
        self.assertFalse(manifest["release_actions"]["rc5_republished_or_relabelled"])

    def test_rc7_manifest_binds_supported_v3_and_keeps_v4_private(self) -> None:
        protocol = (
            ROOT
            / "benchmarks/jx_fast_wisdom_holman_rc7_rebound_race_protocol.json"
        )
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc7.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["version"], "0.6.0rc7")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["verification"]["exact_wheel_cpu_rebound_race"][
                "protocol_sha256"
            ],
            _sha256(protocol),
        )
        supported = manifest["fast_wisdom_holman"]
        self.assertEqual(supported["api_id"], fast_wh.API_ID)
        self.assertEqual(supported["kernel_id"], fast_wh.KERNEL_ID)
        self.assertEqual(
            supported["public_wrapper_sha256"],
            _sha256(ROOT / "src/jxplanetx/fast_wisdom_holman.py"),
        )
        self.assertFalse(supported["default_exact_replay"])
        self.assertTrue(supported["exact_replay_required_for_release_qualification"])
        private = manifest["private_kepler_v4_prototype"]
        self.assertFalse(private["supported_api"])
        self.assertFalse(private["promoted_for_rc7"])
        self.assertEqual(private["recorded_fallback_count"], 0)
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_rc8_manifest_binds_release_evidence_without_promoting_prototypes(
        self,
    ) -> None:
        protocol = (
            ROOT
            / "benchmarks/jx_fast_wisdom_holman_rc8_rebound_race_protocol.json"
        )
        catalog = ROOT / "catalog/JX_EXPERIMENT_CATALOG.json"
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc8.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(__version__, "0.6.0rc10")
        self.assertEqual(manifest["version"], "0.6.0rc8")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(manifest["release_scope"]["general_rebound_superiority_claimed"])

        race = manifest["verification"]["exact_wheel_cpu_rebound_race"]
        self.assertEqual(race["protocol_sha256"], _sha256(protocol))
        self.assertEqual(
            race["required_wheel_sha256"],
            manifest["reproducible_packaging"]["artifacts"]["wheel"]["sha256"],
        )
        self.assertTrue(race["public_jx_100_period_bitwise_parity"])
        self.assertTrue(race["accuracy_and_conservation_gate_passed"])
        self.assertFalse(race["portable_or_general_claim"])

        catalog_record = manifest["experiment_catalog"]
        self.assertEqual(catalog_record["catalog"]["sha256"], _sha256(catalog))
        self.assertEqual(
            catalog_record["library_authority_catalog"]["sha256"],
            "02290fb5d71ff2b1fa961bd72544ba47cc76fd7d086abe578600308f98016b34",
        )
        self.assertFalse(
            catalog_record["active_workspace_hygiene"][
                "flags_are_scientific_validity_judgements"
            ]
        )

        hybrid = manifest["development_hybrid_performance"]
        self.assertFalse(hybrid["supported_api"])
        self.assertFalse(hybrid["promoted_for_rc8"])
        self.assertTrue(hybrid["stateful_near"]["timing_report"]["overall_pass"])
        self.assertFalse(hybrid["endpoint_sync_negative"]["overall_pass"])

        actions = manifest["release_actions"]
        self.assertFalse(actions["push_performed"])
        self.assertFalse(actions["publication_performed"])
        self.assertFalse(actions["frozen_runs_or_archives_modified"])

    def test_rc9_manifest_binds_exact_gr15_artifacts_and_claim_ceiling(self) -> None:
        protocol = ROOT / "benchmarks/jx_gr15_rc9_acceptance_protocol.json"
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc9.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(__version__, "0.6.0rc10")
        self.assertEqual(manifest["version"], "0.6.0rc9")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertFalse(
            manifest["release_scope"]["general_rebound_superiority_claimed"]
        )
        self.assertFalse(
            manifest["release_scope"]["production_or_scientific_qualification_claimed"]
        )
        self.assertEqual(
            manifest["verification"]["exact_rc9_wheel_gr15_acceptance"][
                "protocol_sha256"
            ],
            _sha256(protocol),
        )
        self.assertEqual(
            manifest["verification"]["exact_rc9_wheel_gr15_acceptance"][
                "required_wheel_sha256"
            ],
            "b5db276d9e4293e3a17a380ecbefb614b7ecea0671fb876736e090112dbde842",
        )
        self.assertTrue(
            manifest["verification"]["exact_rc9_wheel_gr15_acceptance"][
                "repeat_non_timing_content_equal"
            ]
        )
        self.assertTrue(
            manifest["verification"]["exact_rc9_wheel_gr15_acceptance"][
                "adversarial_portfolio"
            ]["all_fixed_gates_passed"]
        )
        self.assertFalse(manifest["release_actions"]["push_performed"])
        self.assertFalse(manifest["release_actions"]["publication_performed"])
        self.assertFalse(manifest["release_actions"]["frozen_runs_or_archives_modified"])

    def test_rc10_manifest_binds_reproducible_and_two_host_gates(self) -> None:
        protocol = ROOT / "benchmarks/jx_gr15_rc10_acceptance_protocol.json"
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc10.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(__version__, "0.6.0rc10")
        self.assertEqual(manifest["version"], __version__)
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        packaging = manifest["reproducible_packaging"]
        self.assertTrue(packaging["stage_modes_canonicalized"])
        self.assertEqual(packaging["independent_checkout_contexts"], 2)
        self.assertTrue(packaging["pypi_wheel"]["byte_identical"])
        acceptance = manifest["verification"]["exact_rc10_wheel_gr15_acceptance"]
        self.assertEqual(acceptance["protocol_sha256"], _sha256(protocol))
        self.assertEqual(
            acceptance["required_wheel_sha256"],
            packaging["artifacts"]["wheel"]["sha256"],
        )
        self.assertTrue(acceptance["repeat_non_timing_content_equal"])
        self.assertEqual(acceptance["second_host"]["status"], "PASS_SCREENING_ONLY")
        self.assertFalse(
            acceptance["second_host"]["desktop_timing_classification_portable"]
        )
        self.assertFalse(
            manifest["release_scope"]["general_rebound_superiority_claimed"]
        )
        self.assertFalse(
            manifest["release_scope"]["production_or_scientific_qualification_claimed"]
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])


if __name__ == "__main__":
    unittest.main()
