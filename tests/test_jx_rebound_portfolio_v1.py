from __future__ import annotations

import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from benchmarks import jx_rebound_portfolio_v1 as portfolio


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "jx_rebound_portfolio_v1_protocol.json"
RECOVERED_ROOTS = (
    Path(
        "/home/linoavila/Documents/Codex/2026-08-22/"
        "github-plugin-github-openai-curated-remote-3/work/jx-planet-x-engine"
    ),
    Path(
        "/home/linoavila/Documents/Codex/2026-09-29/c/work/"
        "jx-rc17-equations-candidate-20260929"
    ),
)
RECOVERED_RC13_SDIST = Path(
    "/home/linoavila/Documents/JX-Active-Experiments-Backup-2026-09-24/"
    "tmp-experiments/jxplanetx-0.6.0rc13-final-a/"
    "jxplanetx-0.6.0rc13.tar.gz"
)
RECOVERED_P8_BUNDLE = (
    RECOVERED_ROOTS[0]
    / "runs"
    / "jx_gpu_rebound_global_symplectic_bridge_100yr_v1.tar.gz"
)
RECOVERED_REPORT_ROOT = Path(
    "/home/linoavila/Documents/JX-Active-Experiments-Backup-2026-09-24/"
    "tmp-experiments"
)
RECOVERED_REPORT_INVENTORY = (
    ROOT / "benchmarks" / "jx_rebound_portfolio_v1_recovered_report_inventory.json"
)


class ReboundPortfolioV1Tests(unittest.TestCase):
    def test_checked_protocol_is_exact_deterministic_build(self) -> None:
        receipt = portfolio.validate_protocol(PROTOCOL, ROOT)
        self.assertEqual(receipt["status"], "VALID_PORTFOLIO_CHARTER")
        self.assertEqual(receipt["lanes"], 10)
        self.assertEqual(receipt["rows"], 196)
        self.assertEqual(
            receipt["sha256"],
            "0d2534bdd76bf3aec1c1a6ab1ee747e9695053540490d36835835a5b4af7be5e",
        )
        raw = PROTOCOL.read_bytes()
        parsed = json.loads(raw, parse_constant=portfolio._reject_nonfinite_json_constant)
        self.assertEqual(raw, portfolio.canonical_bytes(parsed))

    def test_charter_claim_ceiling_and_current_comparator(self) -> None:
        protocol = portfolio.build_protocol(ROOT)
        stack = [protocol]
        while stack:
            value = stack.pop()
            self.assertNotIsInstance(value, tuple)
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        self.assertEqual(protocol["status"], portfolio.STATUS)
        comparator = protocol["comparator"]
        self.assertEqual(comparator["version"], "5.2.2")
        self.assertEqual(comparator["license"], "GPL-3.0-only")
        self.assertEqual(
            comparator["wheel"]["sha256"],
            "6f971600864e7c2df3cc00df7df8c1d5a399e08d522789a6a9724d12358945b8",
        )
        runtime = comparator["loaded_wheel_runtime"]
        self.assertEqual(
            runtime["librebound_sha256"],
            "574c062c4dacf8c912c5821d92c0b1b361d31801998f3248cc55e5e742670f04",
        )
        self.assertEqual(runtime["wheel_metadata_bytes"], 9389)
        self.assertEqual(
            runtime["wheel_metadata_sha256"],
            "ce67696222e092693a9b0d93c7b57f7cc3c138a466ca02f48811b0bf2ef78661",
        )
        claims = protocol["claims"]
        self.assertTrue(claims["portfolio_charter_prepared"])
        self.assertTrue(claims["finite_row_roster_defined"])
        self.assertFalse(claims["portfolio_preregistered"])
        self.assertFalse(claims["all_row_physics_inputs_and_gates_frozen"])
        self.assertFalse(claims["dependency_closure_inventory_complete"])
        for name in (
            "benchmark_executed",
            "jx_beats_rebound_generally",
            "jx_beats_rebound_on_any_5_2_2_lane",
            "portable_performance",
            "scientific_qualification",
            "production_authorization",
        ):
            self.assertFalse(claims[name], name)

    def test_finite_row_roster_is_unique_nonvacuous_and_not_run(self) -> None:
        protocol = portfolio.build_protocol(ROOT)
        rows = protocol["row_roster"]
        self.assertEqual(len(rows), 196)
        self.assertEqual(len({row["id"] for row in rows}), 196)
        self.assertEqual(sum(row["required_for_portfolio"] for row in rows), 151)
        self.assertTrue(all(row["outcome"] == "NOT_RUN" for row in rows))
        self.assertNotIn("scoring_status", {key for row in rows for key in row})
        counts = Counter(row["lane"] for row in rows)
        self.assertEqual(
            counts,
            Counter(
                {
                    "P1_CPU_SMALLN": 6,
                    "P2_BATCH_SMALLN": 72,
                    "P3_WH_EQUAL_AND_ENDPOINT": 6,
                    "P4_GR15_TWO_BODY": 12,
                    "P5_ENCOUNTER": 28,
                    "P6_EIH_BATCH": 48,
                    "P7_LONG_ANTI_CHERRY_PICK": 4,
                    "P8_REPRO_ACCURACY": 1,
                    "P9_MASSLESS_POPULATION": 16,
                    "P10_ARCHIVE_CHECKPOINT_SERVICE": 3,
                }
            ),
        )
        aggregate = protocol["aggregate_rule"]
        self.assertEqual(aggregate["total_row_count"], 196)
        self.assertEqual(aggregate["required_row_count"], 151)
        self.assertTrue(aggregate["required_row_count_must_be_nonzero"])
        self.assertTrue(aggregate["terminal_result_row_ids_must_equal_charter_row_ids_exactly"])
        self.assertTrue(aggregate["terminal_result_requires_each_row_exactly_once"])
        self.assertTrue(aggregate["missing_duplicate_or_unexpected_row_invalidates_result"])
        self.assertTrue(aggregate["required_flags_must_match_charter_exactly"])
        self.assertEqual(
            aggregate["finite_portfolio_beaten"],
            "TRUE_IFF_REQUIRED_ROW_COUNT_GT_ZERO_AND_EVERY_REQUIRED_ROW_OUTCOME_IS_"
            "EXACTLY_JX_WIN",
        )

    def test_primary_diagnostic_and_capability_mappings_are_predeclared(self) -> None:
        rows = portfolio.build_protocol(ROOT)["row_roster"]
        p3 = [row for row in rows if row["lane"] == "P3_WH_EQUAL_AND_ENDPOINT"]
        self.assertTrue(
            all(
                row["required_for_portfolio"]
                for row in p3
                if row["jx_variant"] == "RECOVERED_WH_V8"
            )
        )
        self.assertTrue(
            all(
                not row["required_for_portfolio"]
                for row in p3
                if row["jx_variant"] == "SUPPORTED_WH"
            )
        )
        self.assertEqual(
            {row["timing_scope"] for row in p3},
            {
                "EQUAL_OUTPUT_SINGLE_PASS",
                "MANDATORY_REPLAY",
                "MATCHED_ENDPOINT_ONLY_BOTH_ENGINES",
            },
        )
        p6 = [row for row in rows if row["lane"] == "P6_EIH_BATCH"]
        self.assertTrue(all(row["required_for_portfolio"] for row in p6))
        self.assertTrue(
            all(
                row["preparation_status"] == "BLOCKED_MODEL_CONCORDANCE"
                for row in p6
            )
        )
        p9_primary = [
            row
            for row in rows
            if row["lane"] == "P9_MASSLESS_POPULATION"
            and row["required_for_portfolio"]
        ]
        self.assertEqual(len(p9_primary), 8)
        self.assertTrue(
            all(
                (
                    row["workload"] == "QUIET"
                    and row["rebound_variant"] == "WHFAST_TEST_PARTICLES"
                )
                or (
                    row["workload"] == "ROUTED_ENCOUNTER"
                    and row["rebound_variant"] == "MERCURIUS_TEST_PARTICLES"
                )
                for row in p9_primary
            )
        )
        self.assertTrue(
            all(
                row["preparation_status"] == "UNAVAILABLE_CAPABILITY"
                for row in p9_primary
            )
        )
        p10 = [
            row for row in rows if row["lane"] == "P10_ARCHIVE_CHECKPOINT_SERVICE"
        ]
        self.assertEqual(len(p10), 3)
        self.assertTrue(all(not row["required_for_portfolio"] for row in p10))
        self.assertTrue(all("UNSCORED" in row["claim_kind"] for row in p10))

    def test_gr15_gates_preserve_exact_workload_specific_thresholds(self) -> None:
        lane = next(
            lane
            for lane in portfolio.build_protocol(ROOT)["lanes"]
            if lane["id"] == "P4_GR15_TWO_BODY"
        )
        gates = lane["accuracy_gates"]
        self.assertEqual(gates["circular_position_velocity_l2_max"], float(2e-9).hex())
        self.assertEqual(gates["eccentric_position_velocity_l2_max"], float(1e-9).hex())
        self.assertEqual(gates["circular_relative_total_energy_max"], float(2e-11).hex())
        self.assertEqual(
            gates["circular_relative_angular_momentum_max"], float(2e-11).hex()
        )
        self.assertEqual(gates["eccentric_relative_total_energy_max"], float(1e-11).hex())
        self.assertEqual(
            gates["eccentric_relative_angular_momentum_max"], float(1e-11).hex()
        )
        qualification = lane["qualification_only_gates"]
        self.assertTrue(qualification["reverse_probe_is_not_a_timed_row"])

    def test_timing_statistic_and_classification_are_exact(self) -> None:
        protocol = portfolio.build_protocol(ROOT)
        contract = protocol["global_fairness_contract"]
        self.assertEqual(contract["warmups_each_implementation"], 5)
        self.assertEqual(
            contract["timed_repetitions_by_lane"],
            {
                "P1_CPU_SMALLN": 31,
                "P2_BATCH_SMALLN": 31,
                "P3_WH_EQUAL_AND_ENDPOINT": 31,
                "P4_GR15_TWO_BODY": 31,
                "P5_ENCOUNTER": 31,
                "P6_EIH_BATCH": 15,
                "P7_LONG_ANTI_CHERRY_PICK": 7,
                "P8_REPRO_ACCURACY": 0,
                "P9_MASSLESS_POPULATION": 15,
                "P10_ARCHIVE_CHECKPOINT_SERVICE": 15,
            },
        )
        self.assertEqual(
            contract["paired_ratio_definition"],
            "R_I_EQUALS_REBOUND_SECONDS_I_DIVIDED_BY_JX_SECONDS_I",
        )
        bootstrap = contract["paired_speed_ratio_ci"]
        self.assertEqual(
            bootstrap["method"], "PAIRED_BLOCK_PERCENTILE_BOOTSTRAP_OF_MEDIAN_R_I"
        )
        self.assertEqual(bootstrap["resamples"], 100000)
        self.assertEqual(bootstrap["seed"], 2026101502)
        self.assertTrue(contract["outlier_deletion_forbidden"])
        self.assertTrue(contract["early_stopping_forbidden"])
        self.assertTrue(contract["post_result_row_or_lane_removal_forbidden"])
        thresholds = protocol["row_classification_thresholds_decimal"]
        self.assertEqual(thresholds["jx_win_median_minimum"], "1.05")
        self.assertEqual(
            thresholds["rebound_win_median_maximum"], "0.9523809523809523"
        )
        outcomes = protocol["row_outcome_enum"]
        self.assertEqual(
            set(outcomes),
            {
                "NOT_RUN",
                "JX_WIN",
                "REBOUND_WIN",
                "TIE",
                "INVALID",
                "UNAVAILABLE",
                "UNSCORED",
                "SKIPPED",
            },
        )

    def test_massless_contract_is_native_and_not_frozen_source_replay(self) -> None:
        lane = next(
            lane
            for lane in portfolio.build_protocol(ROOT)["lanes"]
            if lane["id"] == "P9_MASSLESS_POPULATION"
        )
        self.assertIn(
            "IDENTICAL_INITIAL_ACTIVE_AND_MASSLESS_BYTES",
            lane["active_source_contract"],
        )
        self.assertIn(
            "EACH_ENGINE_EVOLVES_ITS_OWN_ACTIVE_HISTORY",
            lane["active_source_contract"],
        )
        self.assertTrue(lane["frozen_source_replay_is_not_a_native_rebound_comparison"])

    def test_historical_results_are_disclosures_not_current_proof(self) -> None:
        audit = portfolio.build_protocol(ROOT)["evidence_audit"]
        self.assertEqual(audit["current_speed_proof_count"], 0)
        self.assertEqual(
            audit["historical_direct_speed_comparator_versions"],
            ["5.1.1"],
        )
        self.assertEqual(audit["historical_non_speed_comparator_versions"], ["4.4.11"])
        self.assertTrue(audit["original_temporary_locators_are_not_custody"])
        self.assertEqual(audit["selected_recovered_historical_report_count"], 34)
        self.assertEqual(
            audit["selected_external_direct_comparative_raw_sample_report_count"],
            13,
        )
        inventory = audit["recovered_report_inventory"]
        self.assertEqual(
            inventory["path"],
            "benchmarks/jx_rebound_portfolio_v1_recovered_report_inventory.json",
        )
        self.assertEqual(inventory["schema"], "jx-recovered-report-inventory-v3")
        self.assertTrue(inventory["nonportable_local_paths_omitted"])
        self.assertEqual(
            inventory["counts"],
            {
                "referenced_basenames": 84,
                "basename_present": 75,
                "basename_absent": 9,
                "retained_artifact_present_and_not_known_wrong_bytes": 74,
                "prior_hash_bound_expected_artifact_present": 69,
                "prior_hash_bound_expected_artifact_absent_or_wrong_bytes": 6,
                "no_prior_cryptographic_report_binder": 9,
                "present_without_prior_cryptographic_report_binder": 5,
                "absent_without_prior_cryptographic_report_binder": 4,
                "present_with_raw_timing_sample_arrays": 24,
                "present_without_raw_timing_sample_arrays": 51,
                "prior_hash_bound_expected_artifacts_with_raw_timing_sample_arrays": 21,
            },
        )
        self.assertEqual(
            audit["bound_absent_report_receipts"]["private_gr15_v6"]["sha256"],
            "30b5251c7a4fb575e5c5164b1b947805f8a33d00fa5d8aa6b7efe8291f741b53",
        )
        self.assertFalse(audit["private_wh_v8_confirmatory_report_present"])
        self.assertEqual(
            audit["bound_absent_report_receipts"]["private_wh_v8"]["sha256"],
            "d37eda3c83d8bc5e3360ee74c2fb1616318ee5c385ddd8854bf14306fb03d3aa",
        )
        self.assertEqual(len(audit["bound_absent_report_receipts"]), 6)
        self.assertEqual(audit["unreceipted_missing_historical_output_name_count"], 4)
        self.assertEqual(
            audit["private_wh_v8_bound_pilot_throughput_relative_to_rebound"],
            "1.198506411",
        )
        self.assertIn(
            "JX_DEFENSIVE_OUTPUT_COPY_EXCLUDED",
            audit["private_gr15_v6_timing_scope_disclosure"],
        )
        self.assertTrue(
            all(
                receipt["role"] != "CURRENT_SPEED_PROOF"
                for receipt in audit["public"]
            )
        )

    def test_portable_recovered_report_inventory_is_exact_and_self_consistent(self) -> None:
        expected = portfolio.RECOVERED_REPORT_INVENTORY
        raw = RECOVERED_REPORT_INVENTORY.read_bytes()
        self.assertEqual(len(raw), expected["bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
        self.assertNotIn(b"/home/", raw)
        self.assertNotIn(b"/tmp/", raw)
        parsed = json.loads(
            raw,
            parse_constant=portfolio._reject_nonfinite_json_constant,
        )
        self.assertEqual(parsed["schema"], "jx-recovered-report-inventory-v3")
        self.assertEqual(raw, portfolio.canonical_bytes(parsed))
        self.assertTrue(parsed["nonportable_local_paths_omitted"])
        self.assertEqual(len(parsed["reports"]), 84)
        basenames = [report["basename"] for report in parsed["reports"]]
        self.assertEqual(basenames, sorted(basenames))
        self.assertEqual(len(basenames), len(set(basenames)))
        self.assertFalse(
            any(
                "nonportable" in key.lower()
                for report in parsed["reports"]
                for record in report.get("binding_or_reference_records", ())
                for key in record
            )
        )
        wrong = next(
            report
            for report in parsed["reports"]
            if report["basename"] == "jx_native_stateful_near_hybrid_timing_v1.json"
        )
        self.assertEqual(wrong["custody_status"], "SAME_BASENAME_WRONG_BYTES")
        self.assertFalse(wrong["exact_expected_artifact_present"])
        self.assertEqual(wrong["expected_bytes"], 23005)
        self.assertEqual(
            wrong["expected_sha256"],
            "96b3d0b5021195cb83d04d2c62e96dd7f2872b075696de777dc326d31562aca3",
        )
        self.assertEqual(wrong["bytes"], 22981)
        self.assertEqual(
            wrong["sha256"],
            "cc41151b1fc0907ad33d08326e9ce16aaf8ba344e91d8fe4077afcf11fc0d30c",
        )
        verifications = {
            record["verification"]
            for report in parsed["reports"]
            for record in report.get("binding_or_reference_records", ())
        }
        self.assertEqual(
            verifications,
            {
                "VERIFIED_BINDER_IDENTITY",
                "RECORDED_REFERENCE_NAME",
                "RETAINED_SOURCE_RELATIONSHIP_ONLY",
            },
        )
        one_shot = next(
            report
            for report in parsed["reports"]
            if report["basename"] == "jx_current_100y_same_model_report.json"
        )
        self.assertIn("NOT_SOLVER_SPEED_EVIDENCE", one_shot["role"])
        self.assertIn("asymmetric Python", one_shot["current_claim_limit"])
        receipt = portfolio._load_recovered_report_inventory(ROOT)
        self.assertEqual(receipt["counts"], parsed["counts"])
        self.assertEqual(receipt["sha256"], expected["sha256"])

    def test_strict_json_rejects_nonfinite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            portfolio.canonical_bytes({"value": float("nan")})
        with tempfile.TemporaryDirectory() as temp_dir:
            bad = Path(temp_dir) / "bad.json"
            bad.write_bytes(b'{"value":NaN}\n')
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                portfolio.validate_protocol(bad, ROOT)

    def test_source_is_stdlib_only_and_has_no_execution_surface(self) -> None:
        source_path = ROOT / "benchmarks" / "jx_rebound_portfolio_v1.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        called_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
        self.assertTrue(
            imports
            <= {
                "__future__",
                "argparse",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "stat",
                "tarfile",
                "typing",
            }
        )
        self.assertTrue(
            {
                "rebound",
                "reboundx",
                "numpy",
                "cupy",
                "scipy",
                "socket",
                "subprocess",
            }.isdisjoint(imports)
        )
        self.assertTrue(
            {"integrate", "benchmark", "execute", "run"}.isdisjoint(called_names)
        )
        parser_source = source_path.read_text(encoding="utf-8")
        self.assertNotIn('add_parser("execute")', parser_source)
        self.assertNotIn('add_parser("run")', parser_source)

    def test_recovered_roster_is_portable_selected_and_hash_bound(self) -> None:
        self.assertEqual(len(portfolio.RECOVERED_RESEARCH), 57)
        paths = [item[0] for item in portfolio.RECOVERED_RESEARCH]
        hashes = [item[1] for item in portfolio.RECOVERED_RESEARCH]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))
        self.assertTrue(all(len(digest) == 64 for digest in hashes))
        ordered = hashlib.sha256(
            b"\n".join(
                f"{path}\0{digest}\0{role}".encode("utf-8")
                for path, digest, role in portfolio.RECOVERED_RESEARCH
            )
        ).hexdigest()
        self.assertEqual(
            ordered,
            "cc85b0153f20d0ffb088b1fad1f0efe01d7a82afea0d43af3b461133857fccb6",
        )
        audit = portfolio.build_protocol(ROOT)["evidence_audit"]
        self.assertTrue(
            audit["selected_recovered_research_roster_bytes_match_across_copies"]
        )
        self.assertEqual(audit["selected_recovered_research_roster_file_count"], 57)
        self.assertTrue(audit["rc13_eih_release_source_archive_recovered"])
        self.assertFalse(
            audit["all_candidate_build_configurations_exact_bytes_recovered"]
        )

    def test_selected_historical_report_roster_is_exact_and_portable(self) -> None:
        reports = portfolio.RECOVERED_HISTORICAL_REPORTS
        self.assertEqual(len(reports), 34)
        paths = [entry[0] for entry in reports]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))
        self.assertTrue(all(".." not in Path(path).parts for path in paths))
        self.assertTrue(all(entry[1] > 0 and len(entry[2]) == 64 for entry in reports))
        protocol = portfolio.build_protocol(ROOT)
        audit = protocol["evidence_audit"]
        self.assertEqual(
            audit["selected_recovered_historical_report_ordered_roster_sha256"],
            portfolio._historical_report_roster_sha256(),
        )
        self.assertEqual(
            len(audit["selected_recovered_historical_reports"]), len(reports)
        )

    @unittest.skipUnless(
        RECOVERED_REPORT_ROOT.is_dir(),
        "retained historical report backup absent",
    )
    def test_selected_external_historical_reports_match_exactly(self) -> None:
        receipt = portfolio.audit_recovered_report_root(RECOVERED_REPORT_ROOT)
        self.assertEqual(
            receipt["status"], "SELECTED_RECOVERED_HISTORICAL_REPORTS_MATCH"
        )
        self.assertEqual(receipt["files"], 34)
        self.assertEqual(len(receipt["reports"]), 34)
        self.assertTrue(receipt["absolute_paths_omitted"])
        self.assertEqual(receipt["current_speed_proof_count"], 0)
        self.assertNotIn(str(RECOVERED_REPORT_ROOT), json.dumps(receipt))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = portfolio.RECOVERED_HISTORICAL_REPORTS[0]
            (root / first[0]).symlink_to(RECOVERED_REPORT_ROOT / first[0])
            with self.assertRaisesRegex(ValueError, "without symlinks"):
                portfolio.audit_recovered_report_root(root)

    @unittest.skipUnless(
        all(root.is_dir() for root in RECOVERED_ROOTS),
        "preserved research snapshots absent",
    )
    def test_both_local_preserved_snapshots_match_selected_roster(self) -> None:
        receipt = portfolio.audit_recovered_roots(RECOVERED_ROOTS)
        self.assertEqual(receipt["status"], "RECOVERED_RESEARCH_COPIES_MATCH")
        self.assertEqual(receipt["copies"], 2)
        self.assertEqual(receipt["files_per_copy"], 57)
        self.assertTrue(receipt["absolute_paths_omitted"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            portfolio.audit_recovered_roots((RECOVERED_ROOTS[0], RECOVERED_ROOTS[0]))

    def test_historical_eih_wrapper_custody_is_exact_and_claim_limited(self) -> None:
        archive = portfolio.RECOVERED_RC13_SDIST
        self.assertEqual(archive["release"], "0.6.0rc13")
        self.assertEqual(
            archive["artifact"]["sha256"],
            "9285a3ab7136ba76ff41d0e0b001462e626cb6e6499fac7237845a4996a1b555",
        )
        self.assertEqual(archive["artifact"]["bytes"], 766889)
        self.assertEqual(
            archive["binding_manifest"]["sha256"],
            "8547485e410b880910700c5b7e1563fbca06e32e57eb06c623e35e8b2c55dd95",
        )
        self.assertTrue(archive["binding_manifest"]["base_repository_tracked"])
        self.assertEqual(len(archive["members"]), 2)
        self.assertEqual(
            {member["sha256"] for member in archive["members"]},
            {
                "aa694614e55feb827db5075c46d1489bcdca580dad7d0843957de948d2dddc85",
                "7765a3ed6e628b63031841feab91e05e21edb190f57290eb8089d1ff1b1d3f53",
            },
        )
        lane = next(
            item
            for item in portfolio.build_protocol(ROOT)["lanes"]
            if item["id"] == "P6_EIH_BATCH"
        )
        closure = lane["historical_source_closure"]
        self.assertEqual(
            closure["historical_protocol_bound_wrappers_recovered_from_rc13_sdist"],
            2,
        )
        self.assertIn("RECOVERED_FULL_RAW_REBOUND_5_1_1", lane["historical_prior"])
        self.assertIn("CONCORDANCE_UNESTABLISHED", lane["equation_relationship"])

    @unittest.skipUnless(
        RECOVERED_P8_BUNDLE.is_file(),
        "retained P8 accuracy bundle absent",
    )
    def test_p8_accuracy_bundle_is_exact_and_claim_limited(self) -> None:
        receipt = portfolio.audit_recovered_p8_accuracy_bundle(
            RECOVERED_P8_BUNDLE
        )
        self.assertEqual(receipt["status"], "RECOVERED_P8_ACCURACY_BUNDLE_MATCHES")
        self.assertEqual(
            receipt["artifact"]["sha256"],
            "45123f7b2ec860d6b50d00c84e09b89e9a4c04618b1194d3148dab93342a9155",
        )
        self.assertEqual(receipt["artifact"]["member_count"], 42)
        self.assertEqual(receipt["artifact"]["regular_file_count"], 29)
        self.assertEqual(receipt["artifact"]["directory_count"], 13)
        self.assertEqual(len(receipt["key_members"]), 8)
        self.assertEqual(receipt["sha256sums"]["covered_regular_files"], 28)
        self.assertTrue(receipt["absolute_path_omitted"])
        self.assertFalse(receipt["speed_claim_authorized"])
        self.assertIn("EXPLORATORY_TRANSCRIPT_NOT_RETAINED", receipt["calibration_disclosure"])
        self.assertIn("NO_TIMING_OR_SPEED_CLAIM", receipt["claim_limit"])
        second = (
            RECOVERED_ROOTS[1]
            / portfolio.RECOVERED_P8_ACCURACY_BUNDLE["artifact"]["path"]
        )
        if second.is_file():
            second_receipt = portfolio.audit_recovered_p8_accuracy_bundle(second)
            self.assertEqual(second_receipt, receipt)
        protocol_bundle = portfolio.build_protocol(ROOT)["evidence_audit"][
            "recovered_p8_historical_accuracy_bundle"
        ]
        self.assertEqual(protocol_bundle["artifact"], receipt["artifact"])
        with tempfile.TemporaryDirectory() as temp_dir:
            symlink = Path(temp_dir) / "bundle.tar.gz"
            symlink.symlink_to(RECOVERED_P8_BUNDLE)
            with self.assertRaisesRegex(ValueError, "opened safely"):
                portfolio.audit_recovered_p8_accuracy_bundle(symlink)

    def test_p8_bundle_hostile_tar_shapes_fail_closed(self) -> None:
        root_name = "bundle"

        def json_bytes(value: dict[str, object]) -> bytes:
            return portfolio.canonical_bytes(value)

        def base_payloads(*, optimized_status: str = "PASS") -> dict[str, bytes]:
            payloads = {
                "PROTOCOL.json": json_bytes(
                    {
                        "calibration_disclosure": {
                            "configuration_selected_after_exploratory_same_state_run": True,
                            "exploratory_raw_transcript_retained": False,
                            "gates_are_confirmation_thresholds_not_blind_predictions": True,
                        },
                        "claim_controls": {
                            "gpu_speed_superiority_claimed": False,
                            "same_state_calibration_is_independent_validation_claimed": False,
                        },
                    }
                ),
                "evidence/normal/report.json": json_bytes({"status": "PASS"}),
                "evidence/optimized/report.json": json_bytes(
                    {"status": optimized_status}
                ),
                "SUMMARY.json": json_bytes({}),
                "inputs/ELEVEN_BODY_INPUT.json": json_bytes({}),
                "EXECUTION_RECORD.json": json_bytes({}),
                "verify.py": b"# fixture\n",
            }
            rows = [
                f"{hashlib.sha256(payload).hexdigest()}  {name}"
                for name, payload in sorted(payloads.items())
            ]
            payloads["SHA256SUMS"] = ("\n".join(rows) + "\n").encode("ascii")
            return payloads

        def write_archive(
            path: Path,
            payloads: dict[str, bytes],
            *,
            duplicate_name: str | None = None,
            unsafe_extra: bool = False,
        ) -> None:
            with tarfile.open(path, "w:gz") as archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(f"{root_name}/{name}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                if duplicate_name is not None:
                    payload = payloads[duplicate_name]
                    info = tarfile.TarInfo(f"{root_name}/{duplicate_name}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                if unsafe_extra:
                    payload = b"outside\n"
                    info = tarfile.TarInfo("outside.txt")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

        def spec_for(
            path: Path,
            payloads: dict[str, bytes],
            *,
            member_count: int,
        ) -> dict[str, object]:
            raw = path.read_bytes()
            members = tuple(
                {
                    "path": f"{root_name}/{name}",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "role": "TEST",
                }
                for name, payload in payloads.items()
            )
            return {
                "artifact": {
                    "path": path.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "embedded_root": root_name,
                    "member_count": member_count,
                    "regular_file_count": member_count,
                    "directory_count": 0,
                },
                "members": members,
                "sha256sums": {
                    "path": f"{root_name}/SHA256SUMS",
                    "bytes": len(payloads["SHA256SUMS"]),
                    "sha256": hashlib.sha256(payloads["SHA256SUMS"]).hexdigest(),
                    "covered_regular_files": 7,
                },
                "result_status": "PASS",
                "calibration_disclosure": "TEST",
                "claim_limit": "TEST",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            duplicate = root / "duplicate.tar.gz"
            payloads = base_payloads()
            write_archive(duplicate, payloads, duplicate_name="PROTOCOL.json")
            with mock.patch.object(
                portfolio,
                "RECOVERED_P8_ACCURACY_BUNDLE",
                spec_for(duplicate, payloads, member_count=9),
            ):
                with self.assertRaisesRegex(ValueError, "duplicate member names"):
                    portfolio.audit_recovered_p8_accuracy_bundle(duplicate)

            unsafe = root / "unsafe.tar.gz"
            write_archive(unsafe, payloads, unsafe_extra=True)
            with mock.patch.object(
                portfolio,
                "RECOVERED_P8_ACCURACY_BUNDLE",
                spec_for(unsafe, payloads, member_count=9),
            ):
                with self.assertRaisesRegex(ValueError, "unsafe member path"):
                    portfolio.audit_recovered_p8_accuracy_bundle(unsafe)

            incomplete = root / "incomplete.tar.gz"
            payloads = base_payloads()
            lines = payloads["SHA256SUMS"].splitlines()
            payloads["SHA256SUMS"] = b"\n".join(lines[:-1]) + b"\n"
            write_archive(incomplete, payloads)
            with mock.patch.object(
                portfolio,
                "RECOVERED_P8_ACCURACY_BUNDLE",
                spec_for(incomplete, payloads, member_count=8),
            ):
                with self.assertRaisesRegex(ValueError, "checksum roster differs"):
                    portfolio.audit_recovered_p8_accuracy_bundle(incomplete)

            diverged = root / "diverged.tar.gz"
            payloads = base_payloads(optimized_status="DIFFERENT")
            write_archive(diverged, payloads)
            with mock.patch.object(
                portfolio,
                "RECOVERED_P8_ACCURACY_BUNDLE",
                spec_for(diverged, payloads, member_count=8),
            ):
                with self.assertRaisesRegex(ValueError, "normal and optimized"):
                    portfolio.audit_recovered_p8_accuracy_bundle(diverged)

    @unittest.skipUnless(
        RECOVERED_RC13_SDIST.is_file(),
        "retained rc13 source distribution absent",
    )
    def test_retained_rc13_sdist_and_eih_members_match_exactly(self) -> None:
        receipt = portfolio.audit_recovered_rc13_sdist(RECOVERED_RC13_SDIST)
        self.assertEqual(
            receipt["status"], "RECOVERED_RC13_EIH_SOURCE_ARCHIVE_MATCHES"
        )
        self.assertEqual(len(receipt["members"]), 2)
        self.assertTrue(receipt["absolute_path_omitted"])
        self.assertFalse(receipt["raw_timing_report_embedded_in_this_source_archive"])
        self.assertFalse(receipt["model_concordance_established"])
        with tempfile.TemporaryDirectory() as temp_dir:
            symlink = Path(temp_dir) / "archive.tar.gz"
            symlink.symlink_to(RECOVERED_RC13_SDIST)
            with self.assertRaisesRegex(ValueError, "opened safely"):
                portfolio.audit_recovered_rc13_sdist(symlink)

    def test_rc13_archive_member_failures_are_closed(self) -> None:
        member_path = "pkg/src/jxplanetx/gr15_eih_1pn.py"

        def expected_for(path: Path, member: dict[str, object]) -> dict[str, object]:
            raw = path.read_bytes()
            return {
                "release": "fixture",
                "artifact": {
                    "filename": path.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "binding_manifest": {},
                "members": (member,),
                "claim_limit": "TEST_ONLY",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            duplicate = root / "duplicate.tar.gz"
            with tarfile.open(duplicate, "w:gz") as archive:
                for payload in (b"a", b"b"):
                    info = tarfile.TarInfo(member_path)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            fixture = expected_for(
                duplicate,
                {
                    "path": member_path,
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                    "role": "TEST",
                },
            )
            with mock.patch.object(portfolio, "RECOVERED_RC13_SDIST", fixture):
                with self.assertRaisesRegex(ValueError, "duplicate member names"):
                    portfolio.audit_recovered_rc13_sdist(duplicate)

            symlink_member = root / "symlink-member.tar.gz"
            with tarfile.open(symlink_member, "w:gz") as archive:
                info = tarfile.TarInfo(member_path)
                info.type = tarfile.SYMTYPE
                info.linkname = "elsewhere"
                archive.addfile(info)
            fixture = expected_for(
                symlink_member,
                {
                    "path": member_path,
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "role": "TEST",
                },
            )
            with mock.patch.object(portfolio, "RECOVERED_RC13_SDIST", fixture):
                with self.assertRaisesRegex(ValueError, "member shape differs"):
                    portfolio.audit_recovered_rc13_sdist(symlink_member)

            wrong_bytes = root / "wrong-bytes.tar.gz"
            with tarfile.open(wrong_bytes, "w:gz") as archive:
                info = tarfile.TarInfo(member_path)
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            fixture = expected_for(
                wrong_bytes,
                {
                    "path": member_path,
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"y").hexdigest(),
                    "role": "TEST",
                },
            )
            with mock.patch.object(portfolio, "RECOVERED_RC13_SDIST", fixture):
                with self.assertRaisesRegex(ValueError, "member hash differs"):
                    portfolio.audit_recovered_rc13_sdist(wrong_bytes)


if __name__ == "__main__":
    unittest.main()
