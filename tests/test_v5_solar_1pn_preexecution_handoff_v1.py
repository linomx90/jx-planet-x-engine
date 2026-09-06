"""Adversarial metadata-only tests for the V5 pre-execution handoff request."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_preexecution_handoff_v1")
MANIFEST_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_handoff_v1.py"
SCHEMA_RELATIVE = Path("schemas/jx-v5-solar-1pn-preexecution-handoff-v1.schema.json")
TEST_RELATIVE = Path("tests/test_v5_solar_1pn_preexecution_handoff_v1.py")

PREDECESSOR_PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_qualification_successor_v2")
PREDECESSOR_MANIFEST_RELATIVE = PREDECESSOR_PACKAGE_RELATIVE / "qualification_successor_v2.json"
PREDECESSOR_README_RELATIVE = PREDECESSOR_PACKAGE_RELATIVE / "README.md"
PREDECESSOR_VERIFIER_RELATIVE = PREDECESSOR_PACKAGE_RELATIVE / "verify_qualification_successor_v2.py"
PREDECESSOR_SCHEMA_RELATIVE = Path("schemas/jx-v5-solar-1pn-qualification-successor-v2.schema.json")
PREDECESSOR_TEST_RELATIVE = Path("tests/test_v5_solar_1pn_qualification_successor_v2.py")

REQUEST_ID_PREFIX = "jx.v5.solar_1pn.preexecution_handoff_request."
IDENTITY_DOMAIN = "jx-v5-solar-1pn-preexecution-handoff-request-identity/v1"
IDENTITY_SENTINEL = "CONTENT_DIGEST_SENTINEL"
SOURCE_SLOT_DOMAIN = "jx-v5-solar-1pn-preexecution-handoff-source-slot-binding/v1"


def _load_verifier(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Load exact source bytes without consulting or creating bytecode caches."""
    path = root / VERIFIER_RELATIVE
    namespace: dict[str, object] = {
        "__name__": "jx_v5_preexecution_handoff_verifier_source",
        "__file__": str(path),
    }
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


VERIFIER = _load_verifier()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_identity_sha256(document: dict) -> str:
    normalized = copy.deepcopy(document)
    normalized["handoff_request_id"] = IDENTITY_SENTINEL
    envelope = {
        "schema": IDENTITY_DOMAIN,
        "sentinel": IDENTITY_SENTINEL,
        "handoff": normalized,
    }
    return hashlib.sha256(_canonical_bytes(envelope)).hexdigest()


def _write_with_fresh_identity(path: Path, document: dict) -> tuple[str, str]:
    old_id = document["handoff_request_id"]
    document["handoff_request_id"] = REQUEST_ID_PREFIX + _independent_identity_sha256(document)
    _write_json(path, document)
    return old_id, document["handoff_request_id"]


def _copy_file(source_root: Path, destination_root: Path, relative: Path) -> None:
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / relative, destination)


@contextmanager
def _clone():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "repository"
        for relative in (
            README_RELATIVE,
            MANIFEST_RELATIVE,
            VERIFIER_RELATIVE,
            SCHEMA_RELATIVE,
            TEST_RELATIVE,
            PREDECESSOR_README_RELATIVE,
            PREDECESSOR_MANIFEST_RELATIVE,
            PREDECESSOR_VERIFIER_RELATIVE,
            PREDECESSOR_SCHEMA_RELATIVE,
            PREDECESSOR_TEST_RELATIVE,
        ):
            _copy_file(PROJECT_ROOT, root, relative)
        yield root


def _rejection_code(
    root: Path,
    *,
    manifest_path: str | Path = MANIFEST_RELATIVE,
    verifier: dict[str, object] | None = None,
) -> str:
    namespace = _load_verifier(root) if verifier is None else verifier
    error_type = namespace["PreexecutionHandoffError"]
    try:
        namespace["verify_package"](root, manifest_path)
    except error_type as exc:  # type: ignore[misc]
        return exc.code
    raise AssertionError("mutation unexpectedly verified")


class PreexecutionHandoffV1Tests(unittest.TestCase):
    def assertSemanticMutationRejected(self, root: Path, base: dict, mutator) -> str:
        document = copy.deepcopy(base)
        mutator(document)
        old_id, new_id = _write_with_fresh_identity(root / MANIFEST_RELATIVE, document)
        self.assertNotEqual(old_id, new_id)
        code = _rejection_code(root)
        self.assertNotEqual(code, "handoff_request_id_digest")
        return code

    def test_positive_inspection_is_read_only_blocked_and_content_derived(self):
        watched = (
            MANIFEST_RELATIVE,
            README_RELATIVE,
            VERIFIER_RELATIVE,
            SCHEMA_RELATIVE,
            TEST_RELATIVE,
            PREDECESSOR_MANIFEST_RELATIVE,
            PREDECESSOR_README_RELATIVE,
            PREDECESSOR_VERIFIER_RELATIVE,
            PREDECESSOR_SCHEMA_RELATIVE,
            PREDECESSOR_TEST_RELATIVE,
        )
        before = {path: _file_sha256(PROJECT_ROOT / path) for path in watched}
        inspection = VERIFIER["verify_package"](PROJECT_ROOT)
        after = {path: _file_sha256(PROJECT_ROOT / path) for path in watched}
        self.assertEqual(before, after)
        self.assertEqual(inspection["status"], "HANDOFF_REQUEST_ONLY_BLOCKED")
        self.assertEqual(inspection["requested_role_count"], "14")
        self.assertEqual(inspection["resolved_successor_slots"], "0")
        self.assertIsNone(inspection["execution_qualification_id"])
        for field in (
            "scientific_evidence_artifact",
            "outcomes_generated",
            "execution_authorized",
            "registry_authorized",
            "qualification_authorized",
            "adjudication_authorized",
            "ready_for_holdout_execution",
            "unblinding_occurred",
            "case_branch_selected",
            "sealed_expectation_commitment_present",
            "external_responses_present",
            "root_trust_accepted",
        ):
            self.assertFalse(inspection[field])
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        self.assertEqual(
            manifest["handoff_request_id"],
            REQUEST_ID_PREFIX + _independent_identity_sha256(manifest),
        )

    def test_identity_is_independent_acyclic_and_not_embedded_in_locked_files(self):
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        identity = manifest["handoff_request_id"]
        self.assertRegex(identity, rf"^{REQUEST_ID_PREFIX}[0-9a-f]{{64}}$")
        self.assertNotEqual(identity, REQUEST_ID_PREFIX + "0" * 64)
        self.assertEqual(
            VERIFIER["handoff_request_identity_sha256"](manifest),
            _independent_identity_sha256(manifest),
        )
        locked_paths = {item["path"] for item in manifest["locked_files"]}
        self.assertNotIn(MANIFEST_RELATIVE.as_posix(), locked_paths)
        for relative in (README_RELATIVE, VERIFIER_RELATIVE, SCHEMA_RELATIVE, TEST_RELATIVE):
            self.assertNotIn(identity, (PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        duplicated_id = copy.deepcopy(manifest)
        duplicated_id["blocked_reasons"][0] = identity
        with self.assertRaises(VERIFIER["PreexecutionHandoffError"]) as caught:
            VERIFIER["handoff_request_identity_sha256"](duplicated_id)
        self.assertEqual(caught.exception.code, "identity_self_reference")
        duplicated_sentinel = copy.deepcopy(manifest)
        duplicated_sentinel["blocked_reasons"][0] = IDENTITY_SENTINEL
        with self.assertRaises(VERIFIER["PreexecutionHandoffError"]) as caught:
            VERIFIER["handoff_request_identity_sha256"](duplicated_sentinel)
        self.assertEqual(caught.exception.code, "identity_sentinel")

    def test_source_loader_is_cache_free_and_verifier_is_metadata_only(self):
        source = (PROJECT_ROOT / VERIFIER_RELATIVE).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        allowed = {"__future__", "copy", "hashlib", "json", "re", "pathlib", "typing"}
        self.assertTrue(imports.issubset(allowed), imports)
        self.assertNotIn("__main__", source)
        with _clone() as root:
            self.assertFalse((root / PACKAGE_RELATIVE / "__pycache__").exists())
            _load_verifier(root)["verify_package"](root)
            self.assertFalse((root / PACKAGE_RELATIVE / "__pycache__").exists())

    def test_normal_loader_cache_and_manual_cache_are_rejected(self):
        with _clone() as root:
            code = (
                "import importlib.util,pathlib;"
                f"p=pathlib.Path({VERIFIER_RELATIVE.as_posix()!r});"
                "s=importlib.util.spec_from_file_location('cached_handoff_verifier',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)"
            )
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPYCACHEPREFIX", None)
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / PACKAGE_RELATIVE / "__pycache__").is_dir())
            self.assertEqual(_rejection_code(root), "package_roster")
        with _clone() as root:
            cache = root / PREDECESSOR_PACKAGE_RELATIVE / "__pycache__"
            cache.mkdir()
            (cache / "unbound.pyc").write_bytes(b"unbound")
            self.assertEqual(_rejection_code(root), "package_roster")

    def test_strict_manifest_parser_rejects_duplicate_keys_numbers_and_nonfinite(self):
        mutations = (
            ('{"status":"HANDOFF_REQUEST_ONLY_BLOCKED",', "duplicate_json_key"),
            ('{"rogue":1,', "json_number_forbidden"),
            ('{"rogue":NaN,', "nonfinite_json_constant"),
        )
        with _clone() as root:
            path = root / MANIFEST_RELATIVE
            original = path.read_text(encoding="utf-8")
            for prefix, expected in mutations:
                with self.subTest(expected=expected):
                    path.write_text(prefix + original.lstrip()[1:], encoding="utf-8")
                    self.assertEqual(_rejection_code(root), expected)
            path.write_text(original, encoding="utf-8")

    def test_fixed_manifest_path_cannot_be_substituted(self):
        with _clone() as root:
            alternate = root / "tests/valid_handoff_copy.json"
            shutil.copy2(root / MANIFEST_RELATIVE, alternate)
            fixed = _read_json(root / MANIFEST_RELATIVE)
            fixed["handoff_request_id"] = REQUEST_ID_PREFIX + "f" * 64
            _write_json(root / MANIFEST_RELATIVE, fixed)
            self.assertEqual(
                _rejection_code(root, manifest_path="tests/valid_handoff_copy.json"),
                "manifest_path",
            )

    def test_unchanged_identity_rejects_manifest_drift(self):
        with _clone() as root:
            manifest = _read_json(root / MANIFEST_RELATIVE)
            manifest["locked_files"][0]["size_bytes"] = "999"
            _write_json(root / MANIFEST_RELATIVE, manifest)
            self.assertEqual(_rejection_code(root), "handoff_request_id_digest")

    def test_final_recheck_rejects_concurrent_manifest_mutation(self):
        with _clone() as root:
            namespace = _load_verifier(root)
            original = namespace["_stable_read"]
            target = (root / MANIFEST_RELATIVE).resolve()
            mutated = False

            def racing_read(path, context):
                nonlocal mutated
                data = original(path, context)
                if path == target and context == "handoff request manifest" and not mutated:
                    path.write_bytes(data + b"\n")
                    mutated = True
                return data

            namespace["_stable_read"] = racing_read
            error_type = namespace["PreexecutionHandoffError"]
            with self.assertRaises(error_type) as caught:
                namespace["verify_package"](root)
            self.assertTrue(mutated)
            self.assertEqual(caught.exception.code, "concurrent_mutation")

    def test_all_fourteen_source_slots_and_requirements_are_exact_disk_derived(self):
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        predecessor = _read_json(PROJECT_ROOT / PREDECESSOR_MANIFEST_RELATIVE)
        self.assertEqual(
            [item["role"] for item in manifest["artifact_requests"]],
            [item["role"] for item in predecessor["unresolved_artifact_slots"][:14]],
        )
        for index, (request, slot) in enumerate(
            zip(manifest["artifact_requests"], predecessor["unresolved_artifact_slots"][:14])
        ):
            envelope = {
                "schema": SOURCE_SLOT_DOMAIN,
                "successor_spec_id": predecessor["successor_spec_id"],
                "slot": slot,
            }
            self.assertEqual(request["request_ordinal"], f"{index + 1:02}")
            self.assertEqual(request["requirements"], slot["requirements"])
            self.assertEqual(
                request["source_slot_binding_sha256"],
                hashlib.sha256(_canonical_bytes(envelope)).hexdigest(),
            )
        with _clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)
            for index, request in enumerate(base["artifact_requests"]):
                for requirement in tuple(request["requirements"]):
                    with self.subTest(role=request["role"], requirement=requirement):
                        def remove(document, index=index, requirement=requirement):
                            document["artifact_requests"][index]["requirements"].remove(requirement)

                        self.assertSemanticMutationRejected(root, base, remove)

    def test_role_slot_digest_party_confidentiality_and_dependency_mutations(self):
        with _clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)
            mutators = (
                lambda d: d["artifact_requests"].reverse(),
                lambda d: d["artifact_requests"][0].update({"source_selector": "unresolved_artifact_slots[1]"}),
                lambda d: d["artifact_requests"][0].update({"source_slot_binding_sha256": "a" * 64}),
                lambda d: d["artifact_requests"][0].update({"responsible_party_class": "JX_SELF_ATTESTED"}),
                lambda d: d["artifact_requests"][13].update({"confidentiality_class": "PUBLIC_PLAINTEXT"}),
                lambda d: d["artifact_requests"][12]["dependency_roles"].pop(),
                lambda d: d["artifact_requests"][13]["dependency_roles"].pop(),
            )
            for index, mutator in enumerate(mutators):
                with self.subTest(mutation=index):
                    self.assertSemanticMutationRejected(root, base, mutator)

    def test_dependency_cycle_and_trust_bootstrap_cycles_are_rejected(self):
        with _clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)

            def cycle(document):
                document["artifact_requests"][2]["dependency_roles"] = [
                    "fresh_external_case_custody_and_claim_scope"
                ]

            self.assertEqual(
                self.assertSemanticMutationRejected(root, base, cycle),
                "dependency_cycle",
            )
            mutators = (
                lambda d: d["trust_bootstrap"].update({"response_may_self_authorize": True}),
                lambda d: d["trust_bootstrap"].update({"response_may_self_sign_for_acceptance": True}),
                lambda d: d["trust_bootstrap"].update({"proposed_key_may_be_sole_acceptance_root": True}),
                lambda d: d["trust_bootstrap"].update({"accepted": True}),
                lambda d: d["trust_bootstrap"]["acceptance"].update({"root_trust_policy_id": "jx.root.fake"}),
                lambda d: d["trust_bootstrap"]["acceptance"].update({"proposed_trust_profile_response_id": "jx.response.fake"}),
            )
            for index, mutator in enumerate(mutators):
                with self.subTest(mutation=index):
                    self.assertSemanticMutationRejected(root, base, mutator)

    def test_response_null_authority_claim_blocker_and_nonclaim_mutations(self):
        with _clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)
            mutators = [
                lambda d: d["artifact_requests"][0].update({"selection": "FRESH_INPUT_BLIND"}),
                lambda d: d["artifact_requests"][0]["response"].update({"path": "secret.json"}),
                lambda d: d["artifact_requests"][0]["response"].update({"signature": "self-signed"}),
                lambda d: d["artifact_requests"][0].update({"slot_resolved": True}),
                lambda d: d["artifact_requests"][0].update({"authority_granted": True}),
                lambda d: d["common_future_response_contract"].update({"request_mutation_permitted": True}),
                lambda d: d["common_future_response_contract"]["signed_payload_replacement_map"].update({"signature": ""}),
                lambda d: d["blocked_reasons"].__setitem__(0, "READY"),
                lambda d: d.update({"nonclaim": "Execution is scientifically qualified."}),
                lambda d: d["claim_control"].update({"claim_emitted": True, "claim_text": "qualified"}),
                lambda d: d.update({"execution_qualification_id": "jx.execution.fake"}),
                lambda d: d.update({"sealed_expectation_commitment_present": True}),
            ]
            for flag in (
                "scientific_evidence_artifact",
                "outcomes_generated",
                "execution_authorized",
                "registry_authorized",
                "qualification_authorized",
                "adjudication_authorized",
                "ready_for_holdout_execution",
                "unblinding_occurred",
                "case_branch_selected",
                "external_responses_present",
                "root_trust_accepted",
            ):
                mutators.append(lambda d, flag=flag: d.update({flag: True}))
            for index, mutator in enumerate(mutators):
                with self.subTest(mutation=index):
                    self.assertSemanticMutationRejected(root, base, mutator)

    def test_schema_cannot_be_weakened_and_rehashed_into_new_request_identity(self):
        with _clone() as root:
            schema_path = root / SCHEMA_RELATIVE
            schema = _read_json(schema_path)
            schema["title"] += " weakened"
            _write_json(schema_path, schema)
            manifest = _read_json(root / MANIFEST_RELATIVE)
            binding = next(item for item in manifest["locked_files"] if item["role"] == "handoff_schema")
            binding["sha256"] = _file_sha256(schema_path)
            binding["size_bytes"] = str(schema_path.stat().st_size)
            binding["canonical_sha256"] = _canonical_sha256(schema)
            _write_with_fresh_identity(root / MANIFEST_RELATIVE, manifest)
            self.assertEqual(_rejection_code(root), "schema_binding")

    def test_current_locked_file_deletion_drift_path_extra_link_and_hardlink_rejected(self):
        cases = (README_RELATIVE, SCHEMA_RELATIVE, VERIFIER_RELATIVE, TEST_RELATIVE)
        for relative in cases:
            with self.subTest(relative=relative, operation="delete"), _clone() as root:
                (root / relative).unlink()
                self.assertIn(
                    _rejection_code(root, verifier=VERIFIER),
                    {"unsafe_path", "package_roster"},
                )
            with self.subTest(relative=relative, operation="drift"), _clone() as root:
                path = root / relative
                path.write_bytes(path.read_bytes() + b"\n")
                self.assertIn(
                    _rejection_code(root, verifier=VERIFIER),
                    {"size_mismatch", "raw_digest_mismatch", "schema_binding"},
                )
        with _clone() as root:
            (root / PACKAGE_RELATIVE / "extra.txt").write_text("extra\n", encoding="utf-8")
            self.assertEqual(_rejection_code(root), "package_roster")
        with _clone() as root:
            link = root / PACKAGE_RELATIVE / "extra-link"
            link.symlink_to(root / README_RELATIVE)
            self.assertEqual(_rejection_code(root), "symlink_forbidden")
        with _clone() as root:
            readme = root / README_RELATIVE
            outside = root / "outside.md"
            shutil.copy2(readme, outside)
            readme.unlink()
            os.link(outside, readme)
            self.assertEqual(_rejection_code(root), "hardlink_forbidden")
        with _clone() as root:
            readme = root / README_RELATIVE
            outside = root / "outside.md"
            shutil.copy2(readme, outside)
            readme.unlink()
            readme.symlink_to(outside)
            self.assertEqual(_rejection_code(root), "symlink_forbidden")

    def test_every_predecessor_closure_file_deletion_and_drift_is_rejected(self):
        cases = (
            PREDECESSOR_MANIFEST_RELATIVE,
            PREDECESSOR_README_RELATIVE,
            PREDECESSOR_SCHEMA_RELATIVE,
            PREDECESSOR_VERIFIER_RELATIVE,
            PREDECESSOR_TEST_RELATIVE,
        )
        for relative in cases:
            with self.subTest(relative=relative, operation="delete"), _clone() as root:
                (root / relative).unlink()
                self.assertIn(_rejection_code(root), {"unsafe_path", "package_roster"})
            with self.subTest(relative=relative, operation="drift"), _clone() as root:
                path = root / relative
                path.write_bytes(path.read_bytes() + b"\n")
                self.assertIn(
                    _rejection_code(root),
                    {"size_mismatch", "raw_digest_mismatch", "predecessor_binding"},
                )
        with _clone() as root:
            (root / PREDECESSOR_PACKAGE_RELATIVE / "extra.txt").write_text("extra\n", encoding="utf-8")
            self.assertEqual(_rejection_code(root), "package_roster")
        with _clone() as root:
            link = root / PREDECESSOR_PACKAGE_RELATIVE / "extra-link"
            link.symlink_to(root / PREDECESSOR_README_RELATIVE)
            self.assertEqual(_rejection_code(root), "symlink_forbidden")
        with _clone() as root:
            readme = root / PREDECESSOR_README_RELATIVE
            outside = root / "predecessor-outside.md"
            shutil.copy2(readme, outside)
            readme.unlink()
            readme.symlink_to(outside)
            self.assertEqual(_rejection_code(root), "symlink_forbidden")
        with _clone() as root:
            readme = root / PREDECESSOR_README_RELATIVE
            outside = root / "predecessor-outside.md"
            shutil.copy2(readme, outside)
            readme.unlink()
            os.link(outside, readme)
            self.assertEqual(_rejection_code(root), "hardlink_forbidden")

    def test_locked_roster_and_manifest_identity_cycle_are_rejected(self):
        with _clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)

            def substitute(document):
                item = document["locked_files"][0]
                replacement = root / VERIFIER_RELATIVE
                item["path"] = VERIFIER_RELATIVE.as_posix()
                item["sha256"] = _file_sha256(replacement)
                item["size_bytes"] = str(replacement.stat().st_size)

            self.assertEqual(
                self.assertSemanticMutationRejected(root, base, substitute),
                "locked_roster",
            )

            def self_lock(document):
                item = document["locked_files"][0]
                item["path"] = MANIFEST_RELATIVE.as_posix()

            self.assertSemanticMutationRejected(root, base, self_lock)


if __name__ == "__main__":
    unittest.main()
