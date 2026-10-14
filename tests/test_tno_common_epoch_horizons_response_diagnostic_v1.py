from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "tools"
    / "jx_tno_common_epoch_horizons_response_diagnostic_v1.py"
)


def load_source() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType("_jx_tno_common_epoch_horizons_response_diagnostic_v1_test")
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    import sys

    sys.modules[module.__name__] = module
    try:
        exec(
            compile(payload, str(SOURCE), "exec", optimize=sys.flags.optimize),
            module.__dict__,
        )
    finally:
        sys.modules.pop(module.__name__, None)
    return module


M = load_source()


class HorizonsResponseDiagnosticV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = M.audit_prerequisites()
        cls.protocol = M.build_protocol()

    def test_sealed_parent_and_exact_input_bindings_validate(self) -> None:
        self.assertEqual(
            self.audit["status"],
            "VALIDATED_SEALED_V1_FAILURE_CONTEXT_WITH_NO_NETWORK",
        )
        self.assertEqual(self.audit["head"], M.PARENT_COMMIT)
        self.assertEqual(self.audit["tree"], M.PARENT_TREE)
        self.assertEqual(
            self.audit["v1_protocol_content_sha256"],
            "06aee564cdc8c41be671129da96ed9a531423f68238de972c78533c32d1036cc",
        )
        self.assertEqual(
            {record["role"] for record in self.audit["bindings"]},
            {binding.role for binding in M.V1_BINDINGS},
        )
        for binding in M.V1_BINDINGS:
            payload = (PROJECT_ROOT / binding.relative_path).read_bytes()
            self.assertEqual(len(payload), binding.bytes)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), binding.sha256)

    def test_forensic_conclusion_is_narrow_and_missing_body_is_admitted(self) -> None:
        context = self.protocol["forensic_context"]
        self.assertFalse(context["exact_live_response_retained_by_v1"])
        self.assertFalse(context["exact_live_element_epoch_known"])
        self.assertTrue(context["hypothesis_is_not_accepted_without_the_NEW_RETAINED_BODY"])
        self.assertEqual(
            context["sealed_same_solution_context"],
            {
                "common_standard_orbit_epoch_jd_tdb": "2461200.5",
                "native_covariance_epoch_jd_tdb": "2458411.5",
                "same_solution_native_horizons_header_epoch_jd_tdb": "2458411.5",
            },
        )
        self.assertIn("TDB", context["reason_scale_is_excluded"])

    def test_query_is_byte_for_byte_first_v1_request(self) -> None:
        request = self.protocol["frozen_request"]
        self.assertEqual(M.QUERY_URL, M.EXPECTED_QUERY_URL)
        self.assertEqual(request["url"], M.EXPECTED_QUERY_URL)
        self.assertEqual(request["ordered_parameter_pairs"], [list(pair) for pair in M.QUERY_PAIRS])
        self.assertEqual(request["headers"], [list(pair) for pair in M.REQUEST_HEADERS])
        self.assertEqual(request["target_order"], 0)
        self.assertEqual(request["object_id"], "jpl-20090377")
        self.assertEqual(request["spk_id"], 20090377)
        self.assertEqual(request["solution_id"], 51)
        self.assertEqual(request["request_count"], 1)
        self.assertEqual(request["requests_per_url"], 1)
        self.assertEqual(request["retries_for_any_reason"], 0)
        self.assertEqual(request["parallel_requests"], 0)

    def test_transport_is_bounded_direct_and_terminal(self) -> None:
        transport = self.protocol["future_transport"]
        self.assertTrue(transport["direct_connection_only"])
        self.assertTrue(transport["redirects_forbidden"])
        self.assertTrue(transport["proxy_cookies_authorization_forbidden"])
        self.assertEqual(transport["http_status_required"], 200)
        self.assertEqual(
            transport["streaming"],
            {
                "maximum_chunk_bytes": 65536,
                "maximum_complete_entity_bytes_inclusive": 1048576,
                "maximum_buffered_bytes": 1048577,
                "oversize_action": "STOP_AT_BYTE_1048577_TRANSPORT_FAILURE_NO_COMPLETE_BODY_RESULT",
            },
        )
        self.assertIn("FORBID_ANY_SECOND_REQUEST", transport["transport_failure"])

    def test_attempt_marker_enforces_one_request_across_restarts(self) -> None:
        budget = self.protocol["durable_one_request_budget"]
        self.assertTrue(budget["atomic_create_only_noreplace"])
        self.assertTrue(budget["created_and_parent_fsynced_before_request_bytes_can_be_sent"])
        self.assertTrue(budget["marker_never_removed_or_reused"])
        self.assertEqual(budget["maximum_request_invocations_across_PROCESS_RESTARTS"], 1)
        self.assertEqual(budget["existing_marker_action"], "REFUSE_BEFORE_DNS_OR_SOCKET_CREATION")
        self.assertIn("REMAINS_CONSUMED", budget["crash_after_reservation"])
        attempt = budget["attempt_custody_contract"]
        self.assertEqual(
            attempt["exact_top_level_keys_sorted"],
            sorted(attempt["exact_top_level_keys_sorted"]),
        )
        self.assertEqual(attempt["request_budget_exact_values"]["maximum_request_invocations"], 1)
        self.assertEqual(attempt["request_budget_exact_values"]["retries"], 0)
        self.assertEqual(
            attempt["preregistration_identity_exact_keys_sorted"],
            list(M.PREREGISTRATION_IDENTITY_KEYS),
        )
        self.assertEqual(
            attempt["preregistration_identity_exact_types"],
            M.PREREGISTRATION_IDENTITY_TYPES,
        )
        self.assertEqual(
            attempt["preregistration_identity_fixed_values"],
            {"tag": M.FUTURE_TAG},
        )
        self.assertEqual(
            attempt["top_level_fixed_values"],
            {
                "schema": M.ATTEMPT_SCHEMA,
                "status": "ONE_REQUEST_BUDGET_IRREVERSIBLY_RESERVED",
            },
        )
        publication = budget["attempt_publication_contract"]
        self.assertEqual(publication["exact_root_roster_sorted"], ["ATTEMPT.json"])
        self.assertEqual(publication["directory_mode_octal"], "0700")
        self.assertEqual(publication["file_mode_octal"], "0600")
        self.assertTrue(publication["file_must_be_owned_regular_single_link"])
        self.assertIn("RENAME_NOREPLACE", publication["algorithm"])
        self.assertFalse(budget["offline_validation"]["network_access"])
        self.assertTrue(budget["offline_validation"]["read_only"])

    def test_raw_body_is_fsynced_before_total_semantic_diagnosis(self) -> None:
        order = self.protocol["response_retention_and_diagnosis_order"]
        stage_index = next(index for index, value in enumerate(order) if "PRIVATE_RESULT_STAGING" in value)
        marker_index = next(index for index, value in enumerate(order) if "ATTEMPT_ROOT" in value)
        request_index = next(index for index, value in enumerate(order) if "HTTPS_GET" in value)
        raw_index = next(index for index, value in enumerate(order) if "WRITE_EXACT_ENTITY_BYTES" in value)
        diagnosis_index = next(index for index, value in enumerate(order) if "TOTAL_NONAUTHORITATIVE_DIAGNOSIS" in value)
        custody_index = next(index for index, value in enumerate(order) if "CANONICAL_CUSTODY" in value)
        self.assertLess(stage_index, marker_index)
        self.assertLess(marker_index, request_index)
        self.assertLess(request_index, raw_index)
        self.assertLess(raw_index, diagnosis_index)
        self.assertLess(diagnosis_index, custody_index)
        policy = self.protocol["semantic_policy"]
        self.assertFalse(policy["semantic_gate_controls_retention"])
        self.assertTrue(policy["v1_validator_rejection_is_a_successful_diagnostic_outcome"])
        self.assertFalse(policy["surprising_v1_gate_pass_still_authorizes_oracle"])

    def test_diagnostic_custody_schema_is_fixed_and_never_oracle_authority(self) -> None:
        custody = self.protocol["future_custody_contract"]
        self.assertEqual(custody["schema"], M.CUSTODY_SCHEMA)
        self.assertEqual(custody["status_exact"], M.CUSTODY_STATUS)
        self.assertEqual(
            custody["top_level_fixed_values"],
            {"schema": M.CUSTODY_SCHEMA, "status": M.CUSTODY_STATUS},
        )
        self.assertEqual(custody["maximum_canonical_custody_bytes_inclusive"], 8 * 1048576)
        self.assertEqual(
            custody["exact_top_level_keys_sorted"],
            sorted(custody["exact_top_level_keys_sorted"]),
        )
        self.assertEqual(custody["response_fixed_values"]["raw_relative_path"], M.RAW_RELATIVE)
        self.assertEqual(custody["exact_top_level_keys_sorted"][0], "attempt")
        self.assertEqual(
            custody["attempt_binding_fixed_values"]["relative_path"],
            f"{M.ATTEMPT_ROOT_RELATIVE}/ATTEMPT.json",
        )
        self.assertEqual(
            custody["response_dynamic_types"]["raw_bytes"],
            "INTEGER_NOT_BOOLEAN_0_THROUGH_1048576",
        )
        self.assertEqual(
            custody["preregistration_identity_exact_types"],
            M.PREREGISTRATION_IDENTITY_TYPES,
        )
        self.assertEqual(
            custody["preregistration_identity_fixed_values"],
            {"tag": M.FUTURE_TAG},
        )
        self.assertEqual(
            custody["diagnosis_exact_types"],
            {
                "element_epoch_line_candidates": "ARRAY_OF_STRINGS",
                "json_envelope_status": "EXACT_ENUM_STRING",
                "result_string_sha256": "LOWERCASE_HEX_STRING_LENGTH_64_OR_NULL",
                "state_vector_epoch_token_candidates": "ARRAY_OF_STRINGS",
                "v1_gate_error_message": "STRING_OR_NULL",
                "v1_gate_outcome": "EXACT_ENUM_STRING",
            },
        )
        self.assertIn("EVERY_COMPLETE_BODY", custody["diagnosis_contract"]["totality"])
        for key, value in custody["claims_exact"].items():
            with self.subTest(key=key):
                self.assertFalse(value)
        for key, value in custody["nonclaims_exact"].items():
            with self.subTest(key=key):
                self.assertFalse(value)
        self.assertEqual(
            custody["identity_scope"]["embedded_identity"],
            "PREPARE_PREREGISTRATION_ONLY",
        )
        self.assertFalse(custody["identity_scope"]["executor_identity_embedded"])
        self.assertIn("DOES_NOT_CRYPTOGRAPHICALLY_ATTEST", custody["identity_scope"]["limitation"])

    def test_all_oracle_bridge_dynamics_and_science_claims_are_false(self) -> None:
        claims = self.protocol["claims"]
        self.assertTrue(claims)
        self.assertTrue(all(value is False for value in claims.values()))
        for fragment in ("NO_ORACLE", "NO_BRIDGE", "NO_SCIENCE"):
            self.assertIn(fragment, self.protocol["claim_ceiling"])
        self.assertFalse(self.protocol["authority"]["network_access"])
        self.assertFalse(self.protocol["authority"]["provider_contacted"])
        self.assertFalse(self.protocol["authority"]["execution_in_this_artifact"])

    def test_result_publication_is_create_only_atomic_and_exact(self) -> None:
        publication = self.protocol["future_result_publication"]
        self.assertTrue(publication["result_root_must_not_exist"])
        self.assertEqual(publication["exact_root_roster_sorted"], ["CUSTODY.json", "raw"])
        self.assertEqual(publication["exact_raw_roster_sorted"], ["20090377.json"])
        self.assertEqual(publication["directory_mode_octal"], "0700")
        self.assertEqual(publication["file_mode_octal"], "0600")
        self.assertEqual(publication["atomic_final_rename"], "RENAMEAT2_RENAME_NOREPLACE")
        self.assertTrue(publication["custody_last_staged_file_and_commit_marker"])
        self.assertIn("UNPREDICTABLE_EXCLUSIVE", publication["private_staging_creation"])
        self.assertIn("SUCCESSFUL_RENAMEAT2", publication["publication_commit_point"])
        self.assertIn("REMOVE_ONLY_THE_PRIVATE_STAGING_TREE", publication["handled_failure_before_final_rename"])
        self.assertIn("PRIVATE_STAGING_MAY_REMAIN", publication["crash_before_final_rename"])
        self.assertIn("DURABILITY_INDETERMINATE", publication["failure_after_final_rename_before_parent_fsync"])
        self.assertIn("NEVER_DELETE_REPLACE_REPAIR_OR_RETRY", publication["failure_after_parent_fsync_or_postpublication_recheck"])
        self.assertIn("BYPASS_HANDLED_FAILURE_CLEANUP", publication["process_control_baseexceptions"])
        self.assertIn("NEVER_PATH_RECURSIVELY_REMOVE", publication["cleanup_identity_constraints"])
        self.assertIn("REFUSE_WITHOUT_NETWORK", publication["pre_request_existing_attempt_or_result"])
        self.assertIn("WITHOUT_REPLACEMENT", publication["pre_final_rename_existing_result"])
        self.assertIn("MUST_REPLAY", publication["required_attempt_at_final_rename"])
        self.assertIn("PUBLISH_RAW", publication["semantic_failure_after_complete_body"])
        self.assertIn("KEEP_ATTEMPT_MARKER", publication["transport_failure_before_complete_body"])

    def test_offline_replay_is_read_only_and_network_free(self) -> None:
        replay = self.protocol["future_offline_replay"]
        self.assertFalse(replay["network_access"])
        self.assertTrue(replay["read_only"])
        self.assertTrue(replay["rebuild_exact_custody_from_retained_raw"])
        self.assertTrue(replay["recompute_diagnosis_from_retained_raw"])
        self.assertTrue(replay["verify_attempt_marker_binding"])
        self.assertTrue(replay["enforce_raw_and_custody_byte_caps_before_decoding"])
        self.assertIn("NO_REPAIR", replay["mismatch_action"])
        self.assertEqual(replay["required_cli"], ["validate-attempt", "validate-result"])

    def test_diagnostic_observations_are_total_for_hostile_bounded_bodies(self) -> None:
        surrogate = b'{"result":"\\ud800","signature":{}}'
        observed = M.diagnostic_observations(surrogate)
        self.assertEqual(observed["json_envelope_status"], "STRICT_OBJECT")
        result = json.loads(surrogate)["result"]
        canonical_token = json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
        self.assertEqual(
            observed["result_string_sha256"],
            hashlib.sha256(canonical_token).hexdigest(),
        )

        too_deep = b"[" * 5000 + b"0"
        self.assertEqual(
            M.diagnostic_observations(too_deep)["json_envelope_status"],
            "STRICT_JSON_REJECTED",
        )
        self.assertEqual(
            M.diagnostic_observations(b'{"result":NaN}')["json_envelope_status"],
            "STRICT_JSON_REJECTED",
        )
        self.assertEqual(
            M.diagnostic_observations(b'{"x":1,"x":2}')["json_envelope_status"],
            "STRICT_JSON_REJECTED",
        )
        self.assertEqual(
            M.diagnostic_observations(b"[]")["json_envelope_status"],
            "STRICT_NON_OBJECT",
        )

        reversed_delimiters = json.dumps(
            {"result": "$$EOE x $$SOE"},
            separators=(",", ":"),
        ).encode("ascii")
        reversed_observation = M.diagnostic_observations(reversed_delimiters)
        reversed_token = json.dumps(
            "$$EOE x $$SOE",
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
        self.assertEqual(reversed_observation["json_envelope_status"], "STRICT_OBJECT")
        self.assertEqual(
            reversed_observation["result_string_sha256"],
            hashlib.sha256(reversed_token).hexdigest(),
        )
        self.assertEqual(reversed_observation["state_vector_epoch_token_candidates"], [])

        result_text = " EPOCH= 2458411.5 ! date (TDB)\n$$SOE\n2461200.5, date, 1,2,3,4,5,6,\n$$EOE"
        payload = json.dumps({"result": result_text}, separators=(",", ":")).encode("ascii")
        parsed = M.diagnostic_observations(payload)
        self.assertEqual(parsed["element_epoch_line_candidates"], [" EPOCH= 2458411.5 ! date (TDB)"])
        self.assertEqual(parsed["state_vector_epoch_token_candidates"], ["2461200.5"])

        with self.assertRaises(M.DiagnosticProtocolError):
            M.diagnostic_observations(b"x" * (M.MAX_RESPONSE_BYTES + 1))

    def test_successor_order_and_allowed_diff_are_frozen(self) -> None:
        future = self.protocol["future_preregistration_and_execution"]
        self.assertEqual(future["prepare_tag"], M.FUTURE_TAG)
        self.assertEqual(future["executor_tag"], M.FUTURE_EXECUTOR_TAG)
        self.assertTrue(future["annotated_tags_required"])
        self.assertTrue(future["successor_must_be_one_clean_direct_child_of_prepare_commit"])
        self.assertEqual(future["request_count_before_all_bindings_and_attempt_reservation_pass"], 0)
        self.assertEqual(
            future["successor_allowed_diff"],
            [
                "tests/test_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py",
                "tools/jx_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py",
                "tools/test_matrix_manifest.json",
            ],
        )
        self.assertTrue(future["v1_result_path_must_remain_absent_and_unmodified"])
        self.assertTrue(future["custody_embeds_prepare_identity_not_executor_identity"])
        self.assertIn(
            "CUSTODY_DOES_NOT_CLAIM_EXECUTOR_CRYPTOGRAPHIC_ATTESTATION",
            future["executor_identity_binding_is_repository_external_to_custody"],
        )

    def test_protocol_is_deterministic_domain_hashed_canonical_json(self) -> None:
        first = M.build_protocol()
        second = M.build_protocol()
        self.assertEqual(first, second)
        self.assertRegex(first["content_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["content_sha256"], M._domain_hash(M.PROTOCOL_DOMAIN, first))
        payload = M._canonical_json(first)
        self.assertEqual(payload, M._canonical_json(M._strict_json(payload, "protocol")))
        self.assertTrue(payload.endswith(b"\n"))
        self.assertNotIn(b"NaN", payload)
        self.assertNotIn(b"Infinity", payload)

    def test_strict_json_rejects_duplicates_nonfinite_and_nonobjects(self) -> None:
        cases = (
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'[]',
            b'\xff',
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(M.DiagnosticProtocolError):
                    M._strict_json(payload, "hostile")

    def test_prepare_is_create_only_private_and_replay_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            value = M.prepare(root)
            self.assertEqual(value, self.protocol)
            self.assertEqual(M.validate_prepared(root), self.protocol)
            self.assertEqual(sorted(item.name for item in root.iterdir()), ["PROTOCOL.json"])
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            path = root / "PROTOCOL.json"
            metadata = path.stat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(metadata.st_uid, os.geteuid())
            with self.assertRaisesRegex(M.DiagnosticProtocolError, "already exists"):
                M.prepare(root)

    def test_validate_prepared_rejects_tamper_mode_and_extra_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            M.prepare(root)
            path = root / "PROTOCOL.json"
            original = path.read_bytes()
            path.write_bytes(original[:-1] + b" \n")
            with self.assertRaises(M.DiagnosticProtocolError):
                M.validate_prepared(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            M.prepare(root)
            os.chmod(root / "PROTOCOL.json", 0o644)
            with self.assertRaisesRegex(M.DiagnosticProtocolError, "identity differs"):
                M.validate_prepared(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            M.prepare(root)
            (root / "extra").write_bytes(b"x")
            with self.assertRaisesRegex(M.DiagnosticProtocolError, "roster differs"):
                M.validate_prepared(root)

    def test_prepare_refuses_existing_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "target"
            target.mkdir()
            root = parent / "prepared"
            root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(M.DiagnosticProtocolError, "already exists"):
                M.prepare(root)
            self.assertEqual(list(target.iterdir()), [])

    def test_prepare_cleanup_on_write_failure_has_no_partial_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(
                M,
                "_write_all",
                side_effect=M.DiagnosticProtocolError("injected write failure"),
            ):
                with self.assertRaisesRegex(M.DiagnosticProtocolError, "injected"):
                    M.prepare(root)
            self.assertFalse(root.exists())

    def test_committed_protocol_equals_build_exactly(self) -> None:
        payload = M.PROTOCOL_PATH.read_bytes()
        self.assertEqual(payload, M._canonical_json(self.protocol))
        self.assertEqual(json.loads(payload), self.protocol)

    def test_manifest_has_one_sorted_candidate_entry(self) -> None:
        manifest = json.loads((PROJECT_ROOT / "tools/test_matrix_manifest.json").read_bytes())
        candidate = "tests/test_tno_common_epoch_horizons_response_diagnostic_v1.py"
        assignments = []
        for profile in manifest["profiles"]:
            self.assertEqual(profile["files"], sorted(profile["files"]))
            assignments.extend(
                profile["id"] for path in profile["files"] if path == candidate
            )
        self.assertEqual(assignments, ["performance-local"])

    def test_prepare_source_has_no_network_client_or_acquisition_cli(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_roots = {"requests", "socket", "http", "urllib.request"}
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertTrue(imported.isdisjoint(forbidden_roots))
        self.assertNotIn("HTTPSConnection", source)
        self.assertNotIn("urlopen", source)
        choices = M._parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices),
            {"audit-prerequisites", "build", "prepare", "validate-prepared"},
        )


if __name__ == "__main__":
    unittest.main()
