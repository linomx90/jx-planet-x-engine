"""Adversarial tests for the blocked V5 qualification successor spec."""

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
PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_qualification_successor_v2")
MANIFEST_RELATIVE = PACKAGE_RELATIVE / "qualification_successor_v2.json"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_qualification_successor_v2.py"
SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-qualification-successor-v2.schema.json"
)
TEST_RELATIVE = Path("tests/test_v5_solar_1pn_qualification_successor_v2.py")

SPEC_ID_PREFIX = "jx.v5.solar_1pn.qualification_successor_spec."
IDENTITY_DOMAIN = "jx-v5-solar-1pn-qualification-successor-spec-identity/v2"
IDENTITY_SENTINEL = "CONTENT_DIGEST_SENTINEL"


def _load_verifier(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Load exact source bytes without consulting or creating bytecode caches."""
    path = root / VERIFIER_RELATIVE
    namespace: dict[str, object] = {
        "__name__": "jx_v5_successor_verifier_source",
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
    normalized["successor_spec_id"] = IDENTITY_SENTINEL
    envelope = {
        "schema": IDENTITY_DOMAIN,
        "sentinel": IDENTITY_SENTINEL,
        "successor": normalized,
    }
    return hashlib.sha256(_canonical_bytes(envelope)).hexdigest()


def _write_with_fresh_identity(path: Path, document: dict) -> tuple[str, str]:
    old_id = document["successor_spec_id"]
    document["successor_spec_id"] = SPEC_ID_PREFIX + _independent_identity_sha256(
        document
    )
    _write_json(path, document)
    return old_id, document["successor_spec_id"]


def _copy_ignoring_caches(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"
        ),
    )


@contextmanager
def _minimal_clone():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "repository"
        _copy_ignoring_caches(PROJECT_ROOT / PACKAGE_RELATIVE, root / PACKAGE_RELATIVE)
        for relative in (SCHEMA_RELATIVE, TEST_RELATIVE):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, destination)
        yield root


@contextmanager
def _full_clone():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "repository"
        _copy_ignoring_caches(PROJECT_ROOT, root)
        yield root


def _rejection_code(
    root: Path, *, manifest_path: str | Path = MANIFEST_RELATIVE
) -> str:
    namespace = _load_verifier(root)
    error_type = namespace["QualificationSuccessorError"]
    try:
        namespace["verify_package"](root, manifest_path)
    except error_type as exc:  # type: ignore[misc]
        return exc.code
    raise AssertionError("mutation unexpectedly verified")


def _slot(document: dict, role: str) -> dict:
    return next(item for item in document["unresolved_artifact_slots"] if item["role"] == role)


class QualificationSuccessorV2Tests(unittest.TestCase):
    def assertSemanticMutationRejected(self, root: Path, base: dict, mutator) -> str:
        document = copy.deepcopy(base)
        mutator(document)
        old_id, new_id = _write_with_fresh_identity(root / MANIFEST_RELATIVE, document)
        self.assertNotEqual(old_id, new_id)
        code = _rejection_code(root)
        self.assertNotEqual(code, "successor_spec_id_digest")
        return code

    def test_positive_inspection_is_read_only_blocked_and_content_derived(self):
        watched = (
            MANIFEST_RELATIVE,
            README_RELATIVE,
            VERIFIER_RELATIVE,
            SCHEMA_RELATIVE,
            TEST_RELATIVE,
        )
        before = {path: _file_sha256(PROJECT_ROOT / path) for path in watched}
        inspection = VERIFIER["verify_package"](PROJECT_ROOT)
        after = {path: _file_sha256(PROJECT_ROOT / path) for path in watched}
        self.assertEqual(before, after)
        self.assertEqual(inspection["status"], "SPECIFICATION_ONLY_BLOCKED")
        self.assertIsNone(inspection["execution_qualification_id"])
        for field in (
            "scientific_evidence_artifact",
            "outcomes_generated",
            "execution_authorized",
            "registry_authorized",
            "ready_for_holdout_execution",
            "unblinding_occurred",
        ):
            self.assertFalse(inspection[field])
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        self.assertEqual(
            manifest["successor_spec_id"],
            SPEC_ID_PREFIX + _independent_identity_sha256(manifest),
        )

    def test_identity_envelope_is_independent_and_has_no_locked_file_cycle(self):
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        identity = manifest["successor_spec_id"]
        self.assertRegex(identity, rf"^{SPEC_ID_PREFIX}[0-9a-f]{{64}}$")
        self.assertNotEqual(identity, SPEC_ID_PREFIX + "0" * 64)
        self.assertEqual(
            VERIFIER["successor_spec_identity_sha256"](manifest),
            _independent_identity_sha256(manifest),
        )
        self.assertNotIn(MANIFEST_RELATIVE.as_posix(), {
            item["path"] for item in manifest["locked_files"]
        })
        for relative in (README_RELATIVE, VERIFIER_RELATIVE, SCHEMA_RELATIVE, TEST_RELATIVE):
            self.assertNotIn(identity, (PROJECT_ROOT / relative).read_text(encoding="utf-8"))

    def test_schema_and_requirement_arrays_are_closed_syntactic_and_unique(self):
        for relative in (MANIFEST_RELATIVE, SCHEMA_RELATIVE):
            raw = (PROJECT_ROOT / relative).read_text(encoding="utf-8")

            def reject_duplicates(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise AssertionError(f"duplicate key {key!r} in {relative}")
                    result[key] = value
                return result

            json.loads(raw, object_pairs_hook=reject_duplicates)

        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        schema = _read_json(PROJECT_ROOT / SCHEMA_RELATIVE)
        for document in (manifest, schema):
            stack = [document]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in {"required", "requirements"} and isinstance(child, list):
                            encoded = [_canonical_bytes(item) for item in child]
                            self.assertEqual(len(encoded), len(set(encoded)))
                        stack.append(child)
                elif isinstance(value, list):
                    stack.extend(value)

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
        with _full_clone() as root:
            cache = root / PACKAGE_RELATIVE / "__pycache__"
            self.assertFalse(cache.exists())
            _load_verifier(root)["verify_package"](root)
            self.assertFalse(cache.exists())

    def test_existing_or_normally_created_bytecode_cache_is_rejected(self):
        with _full_clone() as root:
            code = (
                "import importlib.util,pathlib;"
                f"p=pathlib.Path({VERIFIER_RELATIVE.as_posix()!r});"
                "s=importlib.util.spec_from_file_location('cached_successor_verifier',p);"
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

    def test_strict_manifest_parser_rejects_duplicate_keys_numbers_and_nonfinite(self):
        mutations = (
            ('{"status":"SPECIFICATION_ONLY_BLOCKED",', "duplicate_json_key"),
            ('{"rogue":1,', "json_number_forbidden"),
            ('{"rogue":NaN,', "nonfinite_json_constant"),
        )
        with _minimal_clone() as root:
            path = root / MANIFEST_RELATIVE
            original = path.read_text(encoding="utf-8")
            for prefix, expected in mutations:
                with self.subTest(expected=expected):
                    path.write_text(prefix + original.lstrip()[1:], encoding="utf-8")
                    self.assertEqual(_rejection_code(root), expected)
            path.write_text(original, encoding="utf-8")

    def test_schema_cannot_be_weakened_and_rehashed_into_a_new_identity(self):
        with _minimal_clone() as root:
            schema_path = root / SCHEMA_RELATIVE
            schema = _read_json(schema_path)
            schema["title"] = schema["title"] + " weakened"
            _write_json(schema_path, schema)
            manifest = _read_json(root / MANIFEST_RELATIVE)
            binding = next(
                item for item in manifest["locked_files"] if item["role"] == "successor_schema"
            )
            binding["sha256"] = _file_sha256(schema_path)
            binding["size_bytes"] = str(schema_path.stat().st_size)
            binding["canonical_sha256"] = _canonical_sha256(schema)
            _write_with_fresh_identity(root / MANIFEST_RELATIVE, manifest)
            self.assertEqual(_rejection_code(root), "schema_binding")

    def test_fixed_manifest_path_cannot_be_substituted(self):
        with _minimal_clone() as root:
            alternate = root / "tests/valid_successor_copy.json"
            shutil.copy2(root / MANIFEST_RELATIVE, alternate)
            fixed = _read_json(root / MANIFEST_RELATIVE)
            fixed["successor_spec_id"] = SPEC_ID_PREFIX + "f" * 64
            _write_json(root / MANIFEST_RELATIVE, fixed)
            self.assertEqual(
                _rejection_code(root, manifest_path="tests/valid_successor_copy.json"),
                "manifest_path",
            )

    def test_unchanged_identity_rejects_manifest_drift(self):
        with _minimal_clone() as root:
            manifest = _read_json(root / MANIFEST_RELATIVE)
            manifest["locked_files"][0]["size_bytes"] = "999"
            _write_json(root / MANIFEST_RELATIVE, manifest)
            self.assertEqual(_rejection_code(root), "successor_spec_id_digest")

    def test_chronology_eligibility_map_phase_slot_and_authority_mutations(self):
        with _minimal_clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)

            mutators = []
            mutators.append(lambda d: d["effective_protocol"]["chronology_corrections"][0]["frozen_source"]["expected_fields"].update({"required_before_execution": False}))
            mutators.append(lambda d: d["effective_protocol"]["chronology_corrections"][0]["effective_rule"].update({"phase_required": "BEFORE_EXECUTION"}))
            mutators.append(lambda d: d["effective_protocol"]["chronology_corrections"][0]["effective_rule"].update({"valid_unblinding_ordinal": "2"}))
            mutators.append(lambda d: d["effective_protocol"]["chronology_corrections"][0]["effective_rule"].update({"invalid_or_aborted_requests": "MAY_REVEAL"}))
            mutators.append(lambda d: d["effective_protocol"]["chronology_corrections"][0]["effective_rule"].update({"post_unblinding_tuning_allowed": True}))
            mutators.append(lambda d: d["effective_protocol"]["eligibility_and_claim_corrections"][0]["expected_values"].pop())
            mutators.append(lambda d: d["effective_protocol"]["eligibility_and_claim_corrections"][0].update({"effective_rule": "INPUT_BLIND_HOLDOUTS"}))
            mutators.append(lambda d: d["effective_protocol"].update({"legacy_sealed_expectation_schema_execution_acceptable": True}))
            mutators.append(lambda d: d["effective_protocol"]["frozen_preexecution_blocker_resolution_map"].pop())
            mutators.append(lambda d: d["effective_protocol"]["frozen_preexecution_blocker_resolution_map"].reverse())
            mutators.append(lambda d: d["effective_protocol"]["frozen_preexecution_blocker_resolution_map"][0].update({"artifact_kind": "OTHER"}))
            mutators.append(lambda d: d["effective_protocol"]["frozen_preexecution_blocker_resolution_map"][1].update({"placeholder_id": d["effective_protocol"]["frozen_preexecution_blocker_resolution_map"][0]["placeholder_id"]}))

            def add_q8(mapping_document):
                item = copy.deepcopy(mapping_document["effective_protocol"]["frozen_preexecution_blocker_resolution_map"][-1])
                item.update({"placeholder_id": "blocker.q8.unblinding_record", "artifact_kind": "UNBLINDING_RECORD"})
                mapping_document["effective_protocol"]["frozen_preexecution_blocker_resolution_map"].append(item)

            mutators.append(add_q8)
            mutators.append(lambda d: d["effective_protocol"]["phase_requirements"].reverse())
            mutators.append(lambda d: d["unresolved_artifact_slots"].pop())
            mutators.append(lambda d: d["unresolved_artifact_slots"].reverse())
            mutators.append(lambda d: d["unresolved_artifact_slots"][-2].update({"content_derived_artifact_id": "jx.child." + "a" * 64}))
            mutators.append(lambda d: d["external_identity_policy"]["future_execution_identity_preexecution_role_roster"].pop())
            mutators.append(lambda d: d["external_identity_policy"].update({"future_execution_identity_acyclicity": "EXECUTION_ID_FIRST"}))
            mutators.append(lambda d: d["identity_contract"]["envelope_keys"].reverse())
            mutators.append(lambda d: d["execution_qualification"].update({"execution_qualification_id": "jx.v5.solar_1pn.execution_qualification." + "a" * 64}))
            mutators.append(lambda d: d["claim_control"].update({"claim_emitted": True, "claim_text": "claim"}))
            mutators.append(lambda d: d.update({"nonclaim": "claim"}))
            for flag in (
                "scientific_evidence_artifact",
                "outcomes_generated",
                "execution_authorized",
                "registry_authorized",
                "ready_for_holdout_execution",
                "unblinding_occurred",
            ):
                mutators.append(lambda d, flag=flag: d.update({flag: True}))

            for index, mutator in enumerate(mutators):
                with self.subTest(mutation=index):
                    self.assertSemanticMutationRejected(root, base, mutator)

    def test_every_q2_q3_q4_and_budget_requirement_is_exact_locked(self):
        roles = (
            "q2_execution_and_adjudication_contract",
            "q2_planarity_identifiability_and_remainder_policy",
            "q3_independent_oracle_execution_contract",
            "q4_external_eih_execution_contract",
            "qualification_error_budget",
        )
        with _minimal_clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)
            for role in roles:
                for requirement in tuple(_slot(base, role)["requirements"]):
                    with self.subTest(role=role, requirement=requirement):
                        def remove(document, role=role, requirement=requirement):
                            _slot(document, role)["requirements"].remove(requirement)

                        self.assertSemanticMutationRejected(root, base, remove)

    def test_secrecy_case_output_unblind_and_final_lineage_requirements_are_exact(self):
        roles = (
            "fresh_external_case_custody_and_claim_scope",
            "typed_expectation_content_schema",
            "secrecy_preserving_expectation_commitment_schema",
            "sealed_expectation_commitment",
            "jx_execution_implementation_output_authentication_serialization_and_attempt_contract",
            "output_manifest_commitment",
            "unblinding_record",
            "final_adjudication_and_result_record",
        )
        with _minimal_clone() as root:
            base = _read_json(root / MANIFEST_RELATIVE)
            for role in roles:
                for requirement in tuple(_slot(base, role)["requirements"]):
                    with self.subTest(role=role, requirement=requirement):
                        def remove(document, role=role, requirement=requirement):
                            _slot(document, role)["requirements"].remove(requirement)

                        self.assertSemanticMutationRejected(root, base, remove)

    def test_locked_role_path_and_binding_roster_is_exact(self):
        with _minimal_clone() as root:
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

    def test_frozen_blocker_map_is_disk_derived_exact_25_of_26(self):
        manifest = _read_json(PROJECT_ROOT / MANIFEST_RELATIVE)
        inputs = _read_json(
            PROJECT_ROOT / "runs/v5_solar_1pn_qualification/qualification_inputs_v1.json"
        )
        frozen = [
            (item["placeholder_id"], item["artifact_kind"])
            for item in inputs["blocked_artifacts"]
            if item["required_before_execution"] is True
        ]
        mapped = [
            (item["placeholder_id"], item["artifact_kind"])
            for item in manifest["effective_protocol"]["frozen_preexecution_blocker_resolution_map"]
        ]
        self.assertEqual(len(frozen), 26)
        self.assertEqual(len(mapped), 25)
        self.assertEqual(
            mapped,
            [item for item in frozen if item[0] != "blocker.q8.unblinding_record"],
        )

    def test_frozen_extra_outcome_cache_symlink_and_hardlink_are_rejected(self):
        with _full_clone() as root:
            path = root / "runs/v5_solar_1pn_qualification/outcomes.json"
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(_rejection_code(root), "package_roster")
        with _full_clone() as root:
            cache = root / PACKAGE_RELATIVE / "__pycache__"
            cache.mkdir()
            (cache / "rogue.pyc").write_bytes(b"unbound")
            self.assertEqual(_rejection_code(root), "package_roster")
        with _full_clone() as root:
            link = root / "runs/v5_solar_1pn_qualification/rogue-link"
            link.symlink_to(root / "runs/v5_solar_1pn_qualification/README.md")
            self.assertEqual(_rejection_code(root), "symlink_forbidden")
        with _full_clone() as root:
            readme = root / README_RELATIVE
            outside = root / "outside-hardlink-target.md"
            shutil.copy2(readme, outside)
            readme.unlink()
            os.link(outside, readme)
            self.assertEqual(_rejection_code(root), "hardlink_forbidden")

    def test_complete_predecessor_closures_detect_deletion_and_drift(self):
        cases = (
            (
                "runs/v5_solar_1pn_qualification/prior/equation_level_perihelion_4096_fingerprints_v1.json",
                "delete",
            ),
            (
                "runs/v5_solar_1pn_qualification/sources/README.md",
                "drift",
            ),
            (
                "runs/v5_solar_1pn_execution_prerequisites_v1/q3_oracle_registration_v1.json",
                "delete",
            ),
            (
                "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/error_budget_template_v1.json",
                "delete",
            ),
            ("observer/v5_solar_1pn/candidate.py", "drift"),
        )
        for relative, operation in cases:
            with self.subTest(relative=relative, operation=operation), _full_clone() as root:
                path = root / relative
                if operation == "delete":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b"\n")
                code = _rejection_code(root)
                self.assertIn(
                    code,
                    {
                        "package_roster",
                        "unsafe_path",
                        "size_mismatch",
                        "raw_digest_mismatch",
                        "frozen_source_readme",
                    },
                )

    def test_repository_lineage_independently_confirms_disclosure_precedes_observer(self):
        completed = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                "975b1b7002e358a980457acb18c9beaa90aa1c9f",
                "d6344c49e7c6a734a337e84bd209ca5c11c23916",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
