from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "jx_tno_common_epoch_horizons_oracle_v1.py"
PROTOCOL = ROOT / "runs" / "tno_common_epoch_horizons_oracle_v1" / "PROTOCOL.json"


def _load_module() -> types.ModuleType:
    name = "jx_tno_common_epoch_horizons_oracle_v1_test_subject"
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    exec(compile(payload, str(SOURCE), "exec"), module.__dict__)
    return module


M = _load_module()


def _response_payload(
    target,
    *,
    full_name: str | None = None,
    solution_id: int | None = None,
    planetary_ephemeris: str = "DE441",
    small_body_perturbers: str = "SB441-N16",
    element_tokens: tuple[str, ...] | None = None,
    vector_tokens: tuple[str, ...] = ("1", "2", "3", "4e-3", "5e-3", "6e-3"),
    second_vector: bool = False,
) -> bytes:
    elements = target.element_tokens if element_tokens is None else element_tokens
    vector_row = (
        "2461200.500000000, A.D. 2026-Dec-08 00:00:00.0000, "
        + ", ".join(vector_tokens)
        + ","
    )
    rows = vector_row if not second_vector else vector_row + "\n" + vector_row
    result = "\n".join(
        [
            f"Target body name: {target.full_name if full_name is None else full_name} {{source: JPL#{target.solution_id if solution_id is None else solution_id}}}",
            f"Center body name: Sun (10) {{source: {planetary_ephemeris}}}",
            f"Small perturbers: Yes {{source: {small_body_perturbers}}}",
            "Output units    : AU-D",
            "Output type     : GEOMETRIC cartesian states",
            "Reference frame : Ecliptic of J2000.0",
            "Initial IAU76/J2000 heliocentric ecliptic osculating elements",
            "  EPOCH=  2461200.5 ! 2026-Dec-08.00 (TDB) Residual RMS= 0",
            f"   EC= {elements[0]}   QR= {elements[1]}   TP= {elements[2]}",
            f"   OM= {elements[3]}   W= {elements[4]}   IN= {elements[5]}",
            '  Barycentric Dynamical Time ("TDB" or T_eph) output was requested.',
            " Geometric state vectors have NO corrections or aberrations applied.",
            "$$SOE",
            rows,
            "$$EOE",
        ]
    )
    return M._canonical_json(
        {
            "signature": {"source": "NASA/JPL Horizons API", "version": "1.2"},
            "result": result,
        }
    )


class CommonEpochHorizonsOracleV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = M.build_protocol()

    def test_prepare_only_claim_ceiling_and_no_residual_tolerance(self) -> None:
        self.assertEqual(
            self.protocol["status"],
            "PREPARED_COMMON_EPOCH_HORIZONS_NOMINAL_ORACLE_PREREGISTRATION_NO_ACQUISITION",
        )
        authority = self.protocol["authority"]
        self.assertEqual(
            authority["public_cli"],
            ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        )
        self.assertEqual(
            authority["explicitly_absent_public_cli"],
            ["acquire", "execute", "execute-worker", "resume", "validate-result"],
        )
        self.assertIs(authority["network_access"], False)
        self.assertIs(authority["acquisition_in_this_artifact"], False)
        self.assertIs(authority["execution_in_this_artifact"], False)
        claims = self.protocol["claims"]
        for name in (
            "provider_contacted",
            "responses_acquired",
            "common_epoch_nominal_vectors_retained",
            "common_epoch_nominal_oracle_bound",
            "parent_bridge_oracle_blocker_resolved",
            "residual_tolerance_defined",
            "dynamics_executed",
            "common_epoch_bridge_executed",
            "common_epoch_bridge_qualified",
            "classification_generated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            with self.subTest(name=name):
                self.assertIs(claims[name], False)
        diagnostic = self.protocol["diagnostic_role"]
        self.assertIn("DIAGNOSTIC_ONLY", diagnostic["classification"])
        self.assertIs(diagnostic["numerical_bridge_acceptance_gate"], False)
        self.assertIsNone(diagnostic["residual_pass_tolerance"])
        self.assertIs(diagnostic["residual_pass_fail_decision_authorized"], False)

    def test_parent_tag_commit_tree_payload_and_bridge_bytes_are_exact(self) -> None:
        parent = self.protocol["authority"]["parent_preregistration"]
        self.assertEqual(parent["tag_object"], "4620bf745540da9569d9f94bc3677ab176bca8ef")
        self.assertEqual(parent["commit"], "fca1d7b9ea3a97a8127665dcd0b87c600971a803")
        self.assertEqual(parent["tree"], "300507c69d9c66996da51bb01ce4b7b7bd909979")
        self.assertEqual(parent["tag_payload_bytes"], 327)
        self.assertEqual(
            parent["tag_payload_sha256"],
            "671d24cf40b4cc29ef7e9c5126880b3c8d0f645e2246c697e2cec3b9ab6ab863",
        )
        bridge = self.protocol["parent_bridge"]
        self.assertEqual(bridge["source_sha256"], M.BRIDGE_SOURCE_SHA256)
        self.assertEqual(bridge["test_sha256"], M.BRIDGE_TEST_SHA256)
        self.assertEqual(bridge["protocol_file_sha256"], M.BRIDGE_PROTOCOL_SHA256)
        self.assertEqual(bridge["protocol_content_sha256"], M.BRIDGE_PROTOCOL_CONTENT_SHA256)
        self.assertIs(bridge["blocker_resolved_by_this_prepare_only_artifact"], False)

    def test_future_preregistration_tag_and_successor_phase_order_are_exact(self) -> None:
        future = self.protocol["future_preregistration"]
        self.assertEqual(
            future["exact_annotated_tag_name"],
            "tno-common-epoch-horizons-oracle-v1-preregistered-20261014",
        )
        self.assertIs(future["lightweight_tag_forbidden"], True)
        self.assertEqual(
            future["ordered_release_and_successor_phases"],
            [
                "COMMIT_EXACT_PREPARE_ONLY_SOURCE_TEST_PROTOCOL_AND_MANIFEST_ENTRY",
                "CREATE_ANNOTATED_TAG_WITH_EXACT_FROZEN_NAME_POINTING_TO_THAT_PREPARE_COMMIT",
                "PUSH_PREPARE_BRANCH_AND_ANNOTATED_TAG",
                "REFETCH_BRANCH_AND_TAG_FROM_ORIGIN_WITH_AN_EXPLICIT_TAG_REFSPEC",
                "VERIFY_REFETCHED_TAG_OBJECT_PEELED_COMMIT_TREE_TAG_PAYLOAD_BYTE_LENGTH_AND_TAG_PAYLOAD_SHA256",
                "ONLY_THEN_CREATE_THE_ACQUISITION_SUCCESSOR_COMMIT",
                "SUCCESSOR_VERIFIES_EVERY_EXACT_BINDING_BEFORE_REQUEST_ZERO",
            ],
        )
        self.assertIs(future["refetch_verification_must_complete_before_successor_exists"], True)
        self.assertEqual(future["request_count_allowed_before_all_bindings_pass"], 0)
        self.assertEqual(future["source_binding_frozen_here"], self.protocol["implementation"]["source"])
        self.assertEqual(future["test_binding_frozen_here"], self.protocol["implementation"]["test"])
        required = future["successor_exact_literal_bindings_required_before_request_zero"]
        self.assertIn("TAG_OBJECT_SHA", required)
        self.assertIn("PEELED_COMMIT_SHA", required)
        self.assertIn("PEELED_TREE_SHA", required)
        self.assertIn("TAG_PAYLOAD_BYTE_LENGTH", required)
        self.assertIn("TAG_PAYLOAD_SHA256", required)
        self.assertIn("SOURCE_RELATIVE_PATH_BYTE_LENGTH_SHA256", required)
        self.assertIn("TEST_RELATIVE_PATH_BYTE_LENGTH_SHA256", required)
        self.assertIn("PROTOCOL_RELATIVE_PATH_BYTE_LENGTH_FILE_SHA256_CONTENT_SHA256", required)

    def test_all_twelve_sbdb_raw_records_solution_names_and_tokens_are_bound(self) -> None:
        inputs = self.protocol["inputs"]
        self.assertEqual(inputs["target_count"], 12)
        self.assertEqual(inputs["frozen_solution_ids_in_object_order"], [51, 12, 3, 7, 11, 5, 14, 29, 25, 28, 38, 34])
        self.assertEqual(len(inputs["targets"]), 12)
        for target, record in zip(M.TARGETS, inputs["targets"], strict=True):
            self.assertEqual(record["order"], target.order)
            self.assertEqual(record["object_id"], target.object_id)
            self.assertEqual(record["spk_id"], target.spk_id)
            self.assertEqual(record["primary_designation"], target.primary_designation)
            self.assertEqual(record["full_name"], target.full_name)
            self.assertEqual(record["frozen_solution_id"], target.solution_id)
            self.assertEqual(record["sbdb_models"], {"planetary_ephemeris": "DE441", "small_body_perturbers": "SB441-N16"})
            elements = record["standard_osculating_elements"]
            self.assertEqual(elements["labels"], ["e", "q", "tp", "node", "peri", "i"])
            self.assertEqual(elements["decimal_tokens"], list(target.element_tokens))
            self.assertEqual(elements["binary64_hex"], [float(token).hex() for token in target.element_tokens])
            raw_path = ROOT / record["sbdb_raw"]["relative_path"]
            payload = raw_path.read_bytes()
            self.assertEqual(len(payload), record["sbdb_raw"]["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sbdb_raw"]["sha256"])

    def test_exact_designation_only_query_contract_is_frozen_per_target(self) -> None:
        expected_names = [
            "format", "COMMAND", "OBJ_DATA", "MAKE_EPHEM", "EPHEM_TYPE", "CENTER",
            "TLIST", "TLIST_TYPE", "TIME_TYPE", "REF_PLANE", "REF_SYSTEM", "OUT_UNITS", "VEC_TABLE",
            "VEC_CORR", "CSV_FORMAT", "VEC_LABELS", "TIME_DIGITS",
        ]
        for target, record in zip(M.TARGETS, self.protocol["inputs"]["targets"], strict=True):
            pairs = record["horizons_query"]["ordered_parameter_pairs"]
            self.assertEqual([pair[0] for pair in pairs], expected_names)
            values = dict(pairs)
            self.assertEqual(values["COMMAND"], f"'DES={target.spk_id};'")
            self.assertEqual(values["OBJ_DATA"], "'NO'")
            self.assertEqual(values["CENTER"], "'500@10'")
            self.assertEqual(values["TLIST"], "'2461200.5'")
            self.assertEqual(values["TLIST_TYPE"], "'JD'")
            self.assertEqual(values["TIME_TYPE"], "'TDB'")
            self.assertEqual(values["REF_PLANE"], "'ECLIPTIC'")
            self.assertEqual(values["REF_SYSTEM"], "'J2000'")
            self.assertEqual(values["OUT_UNITS"], "'AU-D'")
            self.assertEqual(values["VEC_TABLE"], "'2'")
            self.assertEqual(values["VEC_CORR"], "'NONE'")
            self.assertEqual(values["CSV_FORMAT"], "'YES'")
            self.assertEqual(values["VEC_LABELS"], "'YES'")
            self.assertEqual(values["TIME_DIGITS"], "'FRACSEC'")
            self.assertEqual(record["horizons_query"]["url"], target.query_url)
            self.assertIn(f"COMMAND=%27DES%3D{target.spk_id}%3B%27", target.query_url)
            self.assertNotIn("EC=", target.query_url)
            self.assertNotIn("QR=", target.query_url)

    def test_future_transport_is_single_pass_bounded_and_byte_preserving(self) -> None:
        transport = self.protocol["future_transport_contract"]
        self.assertIs(transport["implemented_in_this_artifact"], False)
        self.assertEqual(transport["method"], "GET")
        self.assertEqual(transport["ordered_exact_urls"], [target.query_url for target in M.TARGETS])
        self.assertEqual(transport["request_order"], list(range(12)))
        self.assertEqual(transport["request_count"], 12)
        self.assertEqual(transport["requests_per_url"], 1)
        self.assertEqual(transport["retries_for_any_reason"], 0)
        self.assertIn("FORBIDDEN", transport["redirect_policy"])
        self.assertEqual(transport["http_success_policy"], "STATUS_CODE_EXACTLY_200_REQUIRED")
        self.assertEqual(transport["application_request_headers"]["Accept-Encoding"], "identity")
        self.assertEqual(
            transport["timeouts_seconds_per_request"],
            {"connect": 30, "read_idle": 30, "total_wall_clock": 120, "timeout_action": "FAIL_CLOSED_NO_RETRY"},
        )
        streaming = transport["streaming_read"]
        self.assertEqual(streaming["maximum_chunk_bytes"], 65_536)
        self.assertEqual(streaming["maximum_buffered_bytes"], 1_048_577)
        self.assertIn("BYTE_1048577", streaming["algorithm"])
        self.assertIn("NO_DECOMPRESSION", transport["custody_raw_byte_semantics"])
        self.assertIn("PUBLISH_NO_FINAL_RESULT", transport["failure_action"])

    def test_pure_response_gate_accepts_every_exact_identity_and_one_ulp_tp(self) -> None:
        for target in M.TARGETS:
            with self.subTest(target=target.object_id):
                record = M.validate_horizons_response(_response_payload(target), target)
                self.assertEqual(record["full_name"], target.full_name)
                self.assertEqual(record["solution_id"], target.solution_id)
                self.assertEqual(record["planetary_ephemeris"], "DE441")
                self.assertEqual(record["small_body_perturbers"], "SB441-N16")
                self.assertEqual(record["tp_binary64_ulp_distance"], 0)
                self.assertEqual(len(record["position_AU_binary64_hex"]), 3)
                self.assertEqual(len(record["velocity_AU_per_day_binary64_hex"]), 3)
        target = M.TARGETS[0]
        elements = list(target.element_tokens)
        elements[2] = repr(math.nextafter(float(elements[2]), math.inf))
        accepted = M.validate_horizons_response(
            _response_payload(target, element_tokens=tuple(elements)),
            target,
        )
        self.assertEqual(accepted["tp_binary64_ulp_distance"], 1)

    def test_response_gate_rejects_identity_lineage_model_and_vector_faults(self) -> None:
        target = M.TARGETS[0]
        exact = list(target.element_tokens)
        alternate_e = list(exact)
        alternate_e[0] = "0" + alternate_e[0]
        distant_tp = list(exact)
        value = float(distant_tp[2])
        value = math.nextafter(math.nextafter(value, math.inf), math.inf)
        distant_tp[2] = repr(value)
        cases = (
            ("full name", {"full_name": target.full_name + " altered"}),
            ("solution ID", {"solution_id": target.solution_id + 1}),
            ("small-perturber", {"small_body_perturbers": "SB441-N15"}),
            ("exact e token", {"element_tokens": tuple(alternate_e)}),
            ("one binary64 ULP", {"element_tokens": tuple(distant_tp)}),
            ("non-finite", {"vector_tokens": ("1", "2", "3", "4", "5", "1e999")}),
            ("exactly one vector", {"second_vector": True}),
        )
        for message, changes in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(M.HorizonsOracleProtocolError, message):
                    M.validate_horizons_response(_response_payload(target, **changes), target)

    def test_response_gate_rejects_extra_keys_and_enforces_inclusive_byte_ceiling(self) -> None:
        target = M.TARGETS[0]
        envelope = self.protocol["response_acceptance"]["raw_json_envelope"]
        self.assertEqual(envelope["exact_top_level_keys"], ["signature", "result"])
        self.assertIs(envelope["additional_top_level_keys_forbidden"], True)
        self.assertEqual(envelope["maximum_bytes_inclusive"], 1_048_576)
        self.assertIs(envelope["byte_ceiling_enforced_before_utf8_json_decode"], True)
        self.assertEqual(
            self.protocol["future_acquisition_contract"]["maximum_raw_response_bytes_each_inclusive"],
            1_048_576,
        )
        value = json.loads(_response_payload(target))
        value["hostile_extra_key"] = "must be rejected"
        with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "top-level key roster"):
            M.validate_horizons_response(M._canonical_json(value), target)

        payload = _response_payload(target)
        boundary = payload + b" " * (M.MAX_RESPONSE_BYTES - len(payload))
        self.assertEqual(len(boundary), 1_048_576)
        accepted = M.validate_horizons_response(boundary, target)
        self.assertEqual(accepted["raw_bytes"], 1_048_576)
        with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "1048576-byte ceiling"):
            M.validate_horizons_response(boundary + b" ", target)

    def test_future_custody_schema_rosters_types_claims_and_publication_are_frozen(self) -> None:
        custody = self.protocol["future_custody_contract"]
        self.assertEqual(custody["schema"], "jx.tno.common-epoch-horizons-oracle.custody.v1")
        self.assertEqual(
            custody["content_hash"]["domain_separator_hex"],
            b"jx.tno.common-epoch-horizons-oracle.custody.v1\0".hex(),
        )
        self.assertEqual(
            custody["exact_top_level_key_roster_in_lexicographic_canonical_order"],
            list(M.CUSTODY_TOP_LEVEL_KEYS),
        )
        self.assertEqual(list(M.CUSTODY_TOP_LEVEL_KEYS), sorted(M.CUSTODY_TOP_LEVEL_KEYS))
        identity = custody["preregistration_identity"]
        self.assertEqual(identity["exact_key_roster_in_lexicographic_canonical_order"], list(M.CUSTODY_PREREGISTRATION_KEYS))
        self.assertEqual(identity["tag_exact"], M.FUTURE_PREREGISTRATION_TAG)
        record = custody["record_contract"]
        self.assertEqual(record["exact_key_roster_in_lexicographic_canonical_order"], list(M.CUSTODY_RECORD_KEYS))
        self.assertEqual(list(M.CUSTODY_RECORD_KEYS), sorted(M.CUSTODY_RECORD_KEYS))
        self.assertEqual(record["record_array_order_spk_ids"], [target.spk_id for target in M.TARGETS])
        self.assertEqual(record["printed_element_order"], list(M.ELEMENT_LABELS))
        self.assertEqual(record["vector_component_order"], list(M.VECTOR_COMPONENT_LABELS))
        self.assertEqual(custody["claims"]["exact_values"], M.CUSTODY_CLAIMS)
        self.assertEqual(custody["nonclaims"]["exact_values"], M.CUSTODY_NONCLAIMS)

        publication = self.protocol["future_result_publication_contract"]
        self.assertEqual(publication["exact_result_root_roster"], ["CUSTODY.json", "raw"])
        self.assertEqual(publication["exact_raw_directory_roster"], list(M.RESULT_RAW_ROSTER))
        self.assertEqual(publication["directory_mode_octal"], "0700")
        self.assertEqual(publication["file_mode_octal"], "0600")
        self.assertIs(publication["every_file_must_be_regular_owned_by_effective_uid_with_nlink_1"], True)
        self.assertIs(publication["custody_is_last_file_and_commit_marker"], True)
        phases = publication["ordered_publication_phases"]
        self.assertIn("ATOMIC_RENAME_NOREPLACE_PARTIAL_TO_CUSTODY_JSON_AS_THE_LAST_STAGED_FILE_AND_COMMIT_MARKER", phases)
        self.assertEqual(phases[-1], "FSYNC_FINAL_PARENT_DIRECTORY")
        self.assertIn("RENAME_NOREPLACE", publication["final_root_publication"])
        self.assertIn("FINAL_ROOT_ABSENT", publication["prepublication_failure"])

        replay = self.protocol["future_offline_replay_validator"]
        self.assertEqual(replay["required_successor_cli"], "validate-result")
        self.assertIs(replay["network_access"], False)
        self.assertIs(replay["read_only"], True)
        self.assertTrue(any("REPLAY_VALIDATE_HORIZONS_RESPONSE" in phase for phase in replay["validation_order"]))
        self.assertEqual(replay["provider_contact_or_dns_tls_http_use"], "FORBIDDEN")

    def test_ordered_vector_digest_is_domain_separated_little_endian_12x6(self) -> None:
        records = [M.validate_horizons_response(_response_payload(target), target) for target in M.TARGETS]
        for target, record in zip(M.TARGETS, records, strict=True):
            self.assertEqual(tuple(record), M.CUSTODY_RECORD_KEYS)
            self.assertEqual(record["order"], target.order)
            self.assertEqual(record["retained_relative_path"], f"raw/{target.spk_id}.json")
        encoded = b"".join(
            struct.pack(
                "<6d",
                *[
                    float.fromhex(token)
                    for token in record["position_AU_binary64_hex"] + record["velocity_AU_per_day_binary64_hex"]
                ],
            )
            for record in records
        )
        self.assertEqual(len(encoded), 576)
        expected = hashlib.sha256(M.VECTOR_DOMAIN + encoded).hexdigest()
        self.assertEqual(M.ordered_vector_sha256(records), expected)
        digest_contract = self.protocol["future_custody_contract"]["ordered_vector_digest"]
        self.assertEqual(digest_contract["shape"], [12, 6])
        self.assertEqual(digest_contract["payload_bytes"], 576)
        self.assertEqual(digest_contract["byte_encoding"], "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER_NO_PADDING")
        tampered = json.loads(json.dumps(records))
        tampered[0]["position_AU_decimal_tokens"][0] = "2"
        with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "decimal/hex binding"):
            M.ordered_vector_sha256(tampered)

    def test_every_declared_prerequisite_is_exactly_audited(self) -> None:
        audit = M.audit_prerequisites()
        groups = audit["input_bindings"]
        self.assertEqual(sum(len(records) for records in groups.values()), len(M.INPUT_BINDINGS))
        self.assertEqual(len(groups["parent_bridge"]), 3)
        self.assertEqual(len(groups["sbdb"]), 2)
        self.assertEqual(len(groups["sbdb_raw"]), 12)
        self.assertEqual(len(audit["target_roster"]), 12)
        self.assertEqual(audit["parent_preregistration"]["tag_object"], M.PARENT_TAG_OBJECT)
        self.assertEqual(audit["parent_preregistration"]["commit"], M.PARENT_COMMIT)
        self.assertEqual(audit["parent_preregistration"]["tree"], M.PARENT_TREE)

    def test_retained_protocol_is_exact_canonical_domain_hashed_reconstruction(self) -> None:
        payload = PROTOCOL.read_bytes()
        self.assertEqual(payload, M._canonical_json(self.protocol))
        parsed = M._strict_json(payload)
        self.assertEqual(parsed, self.protocol)
        self.assertEqual(parsed["content_sha256"], M._content_sha256(parsed))
        self.assertEqual(M.build_protocol(), self.protocol)
        source_payload = SOURCE.read_bytes()
        test_payload = Path(__file__).read_bytes()
        self.assertEqual(self.protocol["implementation"]["source"]["bytes"], len(source_payload))
        self.assertEqual(self.protocol["implementation"]["source"]["sha256"], hashlib.sha256(source_payload).hexdigest())
        self.assertEqual(self.protocol["implementation"]["test"]["bytes"], len(test_payload))
        self.assertEqual(self.protocol["implementation"]["test"]["sha256"], hashlib.sha256(test_payload).hexdigest())

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "duplicate"):
            M._strict_json(b'{"a":1,"a":2}')
        with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "non-finite"):
            M._strict_json(b'{"a":NaN}')

    def test_source_has_no_network_dynamics_or_forbidden_public_operation(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
        self.assertTrue(
            {"urllib", "requests", "socket", "http", "numpy", "scipy", "rebound", "cupy", "spiceypy"}.isdisjoint(imported)
        )
        self.assertTrue({"acquire", "execute", "execute_worker", "resume", "validate_result"}.isdisjoint(functions))

    def test_create_only_atomic_prepare_modes_and_read_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            created = M.prepare(root)
            self.assertEqual(created, self.protocol)
            self.assertEqual(set(os.listdir(root)), {"PROTOCOL.json"})
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "PROTOCOL.json").stat().st_mode), 0o600)
            before = (root.stat().st_mtime_ns, (root / "PROTOCOL.json").stat().st_mtime_ns)
            self.assertEqual(M.validate_prepared(root), self.protocol)
            after = (root.stat().st_mtime_ns, (root / "PROTOCOL.json").stat().st_mtime_ns)
            self.assertEqual(after, before)
            with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "already exists"):
                M.prepare(root)

    def test_prepared_validation_rejects_tampering_modes_and_extra_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tampered = base / "tampered"
            M.prepare(tampered)
            path = tampered / "PROTOCOL.json"
            payload = path.read_bytes()
            path.write_bytes(payload.replace(b'"science_result":false', b'"science_result":true'))
            os.chmod(path, 0o600)
            with self.assertRaises(M.HorizonsOracleProtocolError):
                M.validate_prepared(tampered)

            wrong_mode = base / "wrong-mode"
            M.prepare(wrong_mode)
            os.chmod(wrong_mode / "PROTOCOL.json", 0o644)
            with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "identity"):
                M.validate_prepared(wrong_mode)

            extra = base / "extra"
            M.prepare(extra)
            (extra / "unexpected").write_bytes(b"")
            with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "roster"):
                M.validate_prepared(extra)

    def test_fault_injection_leaves_no_root_or_partial_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "write-failure"
            with mock.patch.object(M, "_write_all", side_effect=OSError("injected write failure")):
                with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "cannot create prepared protocol"):
                    M.prepare(first)
            self.assertFalse(first.exists())

            second = Path(directory) / "publish-failure"
            with mock.patch.object(
                M,
                "_rename_noreplace",
                side_effect=M.HorizonsOracleProtocolError("injected publication failure"),
            ):
                with self.assertRaisesRegex(M.HorizonsOracleProtocolError, "injected publication failure"):
                    M.prepare(second)
            self.assertFalse(second.exists())

    def test_optimized_build_and_audit_match_normal_mode(self) -> None:
        build = subprocess.run(
            [sys.executable, "-OO", str(SOURCE), "build"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertEqual(json.loads(build.stdout), self.protocol)
        audit = subprocess.run(
            [sys.executable, "-OO", str(SOURCE), "audit-prerequisites"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertEqual(json.loads(audit.stdout)["binding_count"], len(M.INPUT_BINDINGS))


if __name__ == "__main__":
    unittest.main()
