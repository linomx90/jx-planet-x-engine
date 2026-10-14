from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_part2_acquisition_v1.py"
MATRIX_PATH = PROJECT_ROOT / "tools" / "test_matrix_manifest.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_de441_part2_acquisition_v1", SOURCE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DE441 part-2 acquisition protocol module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_strings(value: object):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _all_strings(item)
    elif type(value) is list:
        for item in value:
            yield from _all_strings(item)


class TnoDe441Part2AcquisitionV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = ACQ.build_protocol()

    def test_protocol_is_prepare_only_and_exactly_sealed(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], ACQ.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["classification"], ACQ.CLASSIFICATION)
        self.assertEqual(protocol["claim_ceiling"], ACQ.CLAIM_CEILING)
        self.assertEqual(
            protocol["content_sha256"], ACQ._content_sha256(protocol)
        )
        ACQ._validate_protocol(protocol)
        self.assertEqual(
            protocol["status"],
            "PREPARED_DE441_PART2_SOURCE_ACQUISITION_PREREGISTRATION_NO_BYTES",
        )
        self.assertIs(protocol["authority"]["network_access"], False)
        self.assertIs(
            protocol["authority"]["kernel_bytes_read_or_written"], False
        )
        self.assertIs(protocol["claims"]["kernel_downloaded"], False)
        self.assertIs(protocol["claims"]["kernel_bytes_read"], False)
        self.assertIs(
            protocol["claims"]["kernel_custody_established"], False
        )
        self.assertIs(
            protocol["claims"]["scientific_execution_authorized"], False
        )
        self.assertIs(protocol["claims"]["science_result"], False)

    def test_official_source_size_md5_metadata_and_coverage_are_exact(self) -> None:
        source = self.protocol["official_source"]
        self.assertEqual(source["canonical_filename"], "de441_part-2.bsp")
        self.assertEqual(
            source["official_url"],
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
            "de441_part-2.bsp",
        )
        self.assertEqual(source["expected_byte_length"], 1_656_830_976)
        self.assertEqual(
            source["official_md5"], "ad8dfa4e505ef0e3a5d587a5b4705632"
        )
        self.assertIsNone(source["local_sha256"])
        metadata = {
            row["role"]: (row["byte_length"], row["sha256"])
            for row in source["metadata_evidence"]
        }
        self.assertEqual(
            metadata,
            {
                "official_md5_roster": (
                    559,
                    "3c618bed7d51a48ee059269097eeeab1f69f3b2641ec2aeb96132757418c845b",
                ),
                "official_spk_summary_roster": (
                    12_911,
                    "2b34f045b63731a6a315a83b1e2f240dd208a4b5a19ee3419be0573485c15f87",
                ),
                "official_de441_technical_comments": (
                    59_453,
                    "eda144896f7a09fd810c6261036d05d767ce8c77cb2b3795bc824fb623881007",
                ),
            },
        )
        semantics = self.protocol["kernel_semantics"]
        self.assertEqual(
            semantics["coverage_jd_tdb"], ["2440400.50", "8000016.50"]
        )
        self.assertEqual(
            semantics["required_bridge_interval_jd_tdb"],
            ["2452568.5", "2461200.5"],
        )
        self.assertIs(
            semantics["coverage_contains_required_bridge_interval"], True
        )

    def test_exact_spk_roster_and_resolved11_chains_are_frozen(self) -> None:
        semantics = self.protocol["kernel_semantics"]
        observed = [
            (row["target_naif_id"], row["center_naif_id"])
            for row in semantics["stored_target_center_roster"]
        ]
        self.assertEqual(
            observed,
            [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 0),
                (9, 0),
                (10, 0),
                (199, 1),
                (299, 2),
                (301, 3),
                (399, 3),
            ],
        )
        chains = semantics["resolved11_ssb_evaluation_order"]
        self.assertEqual(
            [row["target_naif_id"] for row in chains],
            [10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9],
        )
        self.assertEqual(
            [row["target_to_ssb_chain"] for row in chains],
            [
                [10, 0],
                [199, 1, 0],
                [299, 2, 0],
                [399, 3, 0],
                [301, 3, 0],
                [4, 0],
                [5, 0],
                [6, 0],
                [7, 0],
                [8, 0],
                [9, 0],
            ],
        )

    def test_storage_is_external_portable_private_and_two_copy_sized(self) -> None:
        storage = self.protocol["storage_contract"]
        self.assertEqual(
            storage["asset_root_input_form"],
            "CALLER_SUPPLIED_ABSOLUTE_REAL_DIRECTORY",
        )
        self.assertIn("SIBLING", storage["asset_root_relationship"])
        self.assertIs(storage["reject_asset_root_inside_git_worktree"], True)
        self.assertIs(storage["reject_symlink_root_or_ancestors"], True)
        self.assertIs(
            storage["absolute_asset_root_recorded_in_protocol_or_custody"],
            False,
        )
        self.assertEqual(
            storage["portable_logical_locator_template"],
            "objects/sha256/<computed_lowercase_sha256>/de441_part-2.bsp",
        )
        self.assertEqual(storage["directory_mode_octal"], "0700")
        self.assertEqual(storage["kernel_mode_octal"], "0600")
        self.assertEqual(storage["kernel_link_count"], 1)
        self.assertEqual(
            storage["minimum_two_copy_free_bytes_before_publication"],
            3_313_661_952,
        )
        self.assertIs(storage["kernel_never_stored_in_git_worktree"], True)
        self.assertIs(storage["kernel_never_committed_to_git"], True)
        self.assertIs(storage["multi_file_atomicity_claimed"], False)
        serialized = ACQ._canonical_json(self.protocol).decode("ascii")
        self.assertNotIn(str(PROJECT_ROOT), serialized)
        self.assertFalse(
            any(text.startswith("/") for text in _all_strings(self.protocol))
        )

    def test_future_custody_schema_and_offline_validator_are_complete(self) -> None:
        custody = self.protocol["future_custody_artifact"]
        self.assertEqual(
            custody["schema"], "jx.tno.de441-part2-acquisition.custody.v1"
        )
        self.assertEqual(
            bytes.fromhex(custody["content_hash_domain_hex"]),
            b"jx.tno.de441-part2-acquisition.custody.v1\0",
        )
        self.assertEqual(custody["tracked_post_acquisition_roster"], ["CUSTODY.json"])
        self.assertIs(custody["external_kernel_only"], True)
        self.assertIs(
            custody["official_metadata_source_bytes_in_tracked_roster"], False
        )
        self.assertIs(custody["receipt_created_last"], True)
        self.assertIn("sha256", custody["artifact_required_keys"])
        self.assertIn("ABSOLUTE_LOCAL_PATH", custody["forbidden_values"])
        validator = self.protocol["offline_validator_contract"]
        self.assertIs(validator["network_access"], False)
        self.assertIs(validator["full_kernel_rehash_required_on_every_validation"], True)
        self.assertIs(validator["held_fd_and_stable_path_identity_required"], True)
        self.assertIs(validator["exact_size_and_eof_required"], True)
        self.assertEqual(
            validator["official_metadata_offline_validation"],
            "COMPARE_CUSTODY_RECEIPT_FIELDS_TO_EXACT_FROZEN_PREREGISTRATION_"
            "VALUES_ONLY_NO_SOURCE_BYTE_REEXTRACTION",
        )
        self.assertIs(
            validator[
                "official_metadata_source_bytes_retained_or_revalidated_offline"
            ],
            False,
        )
        self.assertIs(
            validator["spk_semantics_recomputed_from_external_held_kernel"], True
        )
        self.assertNotIn(
            "metadata_and_spk_semantic_receipts_revalidated", validator
        )
        self.assertIs(validator["read_only"], True)
        gates = self.protocol["future_successor_gates"]
        self.assertIn(
            "COMPLETE_STREAMING_MD5_EQUALS_OFFICIAL_MD5", gates["staging"]
        )
        self.assertIn(
            "ONE_HELD_PUBLISHED_FD_FOR_INDEPENDENT_COMPLETE_SIZE_MD5_AND_SHA256_REHASH",
            gates["publication"],
        )
        self.assertIn(
            "CONFIRM_DAF_SPK_STRUCTURE_AND_SEGMENT_DESCRIPTOR_INTEGRITY",
            gates["spk_semantics"],
        )
        self.assertIn("NO_CUSTODY_JSON", gates["failure_semantics"])

    def test_resume_is_forbidden_and_persistent_evaluator_remains_blocked(self) -> None:
        transport = self.protocol["transport_contract"]
        self.assertIs(transport["resumable_transfer_permitted"], False)
        self.assertIs(transport["staging_must_be_one_complete_response_body"], True)
        semantics = self.protocol["kernel_semantics"]
        self.assertIs(
            semantics["persistent_force_callback_evaluator_qualified"], False
        )
        self.assertIn("NOT_A_PERSISTENT_FORCE_CALLBACK", semantics["generic_jpl_ephemeris_role"])
        self.assertIn(
            "DE441_PERSISTENT_SOURCE_EVALUATOR_AND_SOURCE_NODE_GATES_NOT_BOUND",
            self.protocol["execution_blockers_after_this_protocol"],
        )

    def test_parent_bridge_bytes_semantics_and_annotated_tag_are_bound(self) -> None:
        audit = ACQ.audit_prerequisites()
        self.assertEqual(audit["binding_count"], 3)
        self.assertIs(audit["coverage_contains_required_interval"], True)
        self.assertEqual(
            audit["resolved11_source_order"],
            [10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9],
        )
        ancestry = audit["parent_ancestry"]
        self.assertEqual(ancestry["tag"], ACQ.PARENT_TAG)
        self.assertEqual(ancestry["tag_object"], ACQ.PARENT_TAG_OBJECT)
        self.assertEqual(ancestry["commit"], ACQ.PARENT_COMMIT)
        self.assertEqual(ancestry["tree"], ACQ.PARENT_TREE)
        self.assertEqual(
            self.protocol["parent_bridge"]["protocol_content_sha256"],
            ACQ.PARENT_PROTOCOL_CONTENT_SHA256,
        )
        self.assertIs(
            self.protocol["parent_bridge"]["kernel_blocker_closed_by_this_protocol"],
            False,
        )
        self.assertEqual(
            self.protocol["authority"]["git_custody_sequence"],
            [
                "COMMIT_SOURCE_TEST_PROTOCOL_AND_SORTED_MATRIX_ENTRY_ONLY",
                "CREATE_ANNOTATED_PREREGISTRATION_TAG_ON_THE_EXACT_PREREGISTRATION_COMMIT",
                "SUCCESSOR_BINDS_TAG_OBJECT_PEELED_COMMIT_TREE_AND_EXACT_BYTES_BEFORE_"
                "STAGED_KERNEL_READ",
                "AFTER_OFFLINE_VALIDATION_COMMIT_ONLY_PORTABLE_CUSTODY_JSON_NEVER_THE_"
                "EXTERNAL_BSP",
                "CREATE_ANNOTATED_CUSTODY_TAG_ONLY_AFTER_THE_CUSTODY_COMMIT",
            ],
        )

    def test_source_and_test_bytes_are_bound_without_network_or_spice_imports(self) -> None:
        implementation = self.protocol["implementation"]
        self.assertEqual(implementation["source"]["bytes"], SOURCE_PATH.stat().st_size)
        self.assertEqual(implementation["source"]["sha256"], _sha256(SOURCE_PATH))
        self.assertEqual(
            implementation["test"]["bytes"], Path(__file__).stat().st_size
        )
        self.assertEqual(implementation["test"]["sha256"], _sha256(Path(__file__)))
        self.assertIs(
            implementation["prepare_imports_network_or_spice_runtime"], False
        )
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        imports: set[str] = set()
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
        self.assertTrue(
            {"urllib", "requests", "httpx", "socket", "spiceypy"}.isdisjoint(imports)
        )
        self.assertTrue(
            {
                "download",
                "acquire",
                "publish",
                "execute",
                "execute_worker",
                "resume",
                "validate_custody",
            }.isdisjoint(functions)
        )

    def test_cli_surface_is_exact_and_forbidden_commands_do_not_parse(self) -> None:
        parser = ACQ._parser()
        subparser_action = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(subparser_action.choices),
            {"build", "audit-prerequisites", "prepare", "validate-prepared"},
        )
        for command in ("download", "acquire", "publish", "execute", "resume"):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                with redirect_stderr(io.StringIO()):
                    parser.parse_args([command])

    def test_semantic_mutations_fail_even_with_recomputed_content_hash(self) -> None:
        cases = []
        changed = json.loads(json.dumps(self.protocol))
        changed["claims"]["kernel_downloaded"] = True
        cases.append(changed)
        changed = json.loads(json.dumps(self.protocol))
        changed["authority"]["network_access"] = True
        cases.append(changed)
        changed = json.loads(json.dumps(self.protocol))
        changed["storage_contract"]["kernel_never_stored_in_git_worktree"] = False
        cases.append(changed)
        for value in cases:
            value["content_sha256"] = ACQ._content_sha256(value)
            with self.assertRaises(ACQ.KernelProtocolError):
                ACQ._validate_protocol(value)

    def test_prepare_is_private_create_only_and_validate_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            value = ACQ.prepare(root)
            self.assertEqual(value, self.protocol)
            root_info = root.stat()
            protocol_path = root / "PROTOCOL.json"
            file_info = protocol_path.stat()
            before_hash = _sha256(protocol_path)
            self.assertEqual(stat.S_IMODE(root_info.st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(file_info.st_mode), 0o600)
            self.assertEqual(file_info.st_nlink, 1)
            self.assertEqual({item.name for item in root.iterdir()}, {"PROTOCOL.json"})
            self.assertEqual(
                protocol_path.read_bytes(), ACQ._canonical_json(self.protocol)
            )
            self.assertEqual(ACQ.validate_prepared(root), self.protocol)
            self.assertEqual(_sha256(protocol_path), before_hash)
            self.assertEqual(protocol_path.stat().st_size, file_info.st_size)
            with self.assertRaisesRegex(
                ACQ.KernelProtocolError, "already exists"
            ):
                ACQ.prepare(root)

    def test_tamper_wrong_modes_rosters_and_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tampered = base / "tampered"
            ACQ.prepare(tampered)
            path = tampered / "PROTOCOL.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ACQ.KernelProtocolError):
                ACQ.validate_prepared(tampered)

            wrong_mode = base / "wrong-mode"
            ACQ.prepare(wrong_mode)
            os.chmod(wrong_mode / "PROTOCOL.json", 0o644)
            with self.assertRaisesRegex(ACQ.KernelProtocolError, "identity"):
                ACQ.validate_prepared(wrong_mode)

            extra = base / "extra"
            ACQ.prepare(extra)
            (extra / "EXTRA").write_bytes(b"x")
            with self.assertRaisesRegex(ACQ.KernelProtocolError, "roster"):
                ACQ.validate_prepared(extra)

            real_parent = base / "real-parent"
            real_parent.mkdir()
            linked_parent = base / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ACQ.KernelProtocolError, "symlink"):
                ACQ.prepare(linked_parent / "prepared")

    def test_prepublication_faults_leave_no_output_root(self) -> None:
        injections = (
            ("_write_all", ACQ.KernelProtocolError("injected write failure")),
            (
                "_rename_noreplace",
                ACQ.KernelProtocolError("injected rename failure"),
            ),
            (
                "_validate_prepared_fd",
                ACQ.KernelProtocolError("injected validation failure"),
            ),
            ("_write_all", KeyboardInterrupt("injected interrupt")),
        )
        for attribute, failure in injections:
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "prepared"
                expected_error = (
                    KeyboardInterrupt
                    if isinstance(failure, KeyboardInterrupt)
                    else ACQ.KernelProtocolError
                )
                with mock.patch.object(ACQ, attribute, side_effect=failure):
                    with self.assertRaisesRegex(expected_error, "injected"):
                        ACQ.prepare(root)
                self.assertFalse(root.exists())
                self.assertFalse(root.is_symlink())

    def test_wrong_branch_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(ACQ, "_current_branch", return_value="wrong"):
                with self.assertRaisesRegex(
                    ACQ.KernelProtocolError, "branch differs"
                ):
                    ACQ.prepare(root)
            self.assertFalse(root.exists())

    def test_cli_build_audit_prepare_validate_and_optimized_mode(self) -> None:
        build = subprocess.run(
            [sys.executable, str(SOURCE_PATH), "build"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertEqual(json.loads(build.stdout), self.protocol)
        audit = subprocess.run(
            [sys.executable, "-OO", str(SOURCE_PATH), "audit-prerequisites"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertEqual(json.loads(audit.stdout)["status"], "PASSED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            prepared = subprocess.run(
                [
                    sys.executable,
                    "-OO",
                    str(SOURCE_PATH),
                    "prepare",
                    "--output-root",
                    str(root),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            validated = subprocess.run(
                [
                    sys.executable,
                    "-OO",
                    str(SOURCE_PATH),
                    "validate-prepared",
                    "--output-root",
                    str(root),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertEqual(json.loads(validated.stdout)["status"], "VALID")

    def test_matrix_assignment_is_single_sorted_performance_local_entry(self) -> None:
        manifest = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        target = "tests/test_tno_de441_part2_acquisition_v1.py"
        assignments = [
            profile["id"]
            for profile in manifest["profiles"]
            if target in profile["files"]
        ]
        self.assertEqual(assignments, ["performance-local"])
        profile = next(
            row for row in manifest["profiles"] if row["id"] == "performance-local"
        )
        self.assertEqual(profile["files"], sorted(profile["files"]))


if __name__ == "__main__":
    unittest.main()
