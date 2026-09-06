from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from decimal import Decimal
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1")
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_prerequisites_v1.py"
REGISTRATION_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
SLOTS_RELATIVE = PACKAGE_RELATIVE / "external_artifact_slots_v1.json"
BUDGET_RELATIVE = PACKAGE_RELATIVE / "error_budget_template_v1.json"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "jx_v5_q8q9_prerequisite_verifier_tests",
        PROJECT_ROOT / VERIFIER_RELATIVE,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load Q8/Q9 prerequisite verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected object in {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _refresh_artifact_binding(root: Path, role: str) -> None:
    registration_path = root / REGISTRATION_RELATIVE
    registration = _read_json(registration_path)
    binding = next(item for item in registration["artifacts"] if item["role"] == role)
    path = root / binding["path"]
    binding["sha256"] = _file_sha256(path)
    binding["size_bytes"] = str(path.stat().st_size)
    binding["canonical_sha256"] = verifier.sha256_data(_read_json(path))
    _write_json(registration_path, registration)


def _refresh_schema_binding(root: Path, role: str) -> None:
    registration_path = root / REGISTRATION_RELATIVE
    registration = _read_json(registration_path)
    binding = next(item for item in registration["schemas"] if item["role"] == role)
    path = root / binding["path"]
    binding["sha256"] = _file_sha256(path)
    binding["size_bytes"] = str(path.stat().st_size)
    binding["canonical_sha256"] = verifier.sha256_data(_read_json(path))
    _write_json(registration_path, registration)


@contextmanager
def _minimal_clone():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "repository"
        required = [
            Path("runs/v5_solar_1pn_qualification/qualification_plan_v1.json"),
            Path("runs/v5_solar_1pn_qualification/qualification_inputs_v1.json"),
            Path("runs/v5_solar_1pn_qualification/registration_v1.json"),
            *[
                Path(path)
                for _, path in verifier.SCHEMA_SPECS.values()
            ],
        ]
        for relative in required:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, target)
        shutil.copytree(PROJECT_ROOT / PACKAGE_RELATIVE, root / PACKAGE_RELATIVE)
        yield root


def _external_artifact_binding() -> dict:
    return {
        "status": "EXTERNAL_ATTESTATION_ARTIFACT_BOUND",
        "path": "external/attestation.json",
        "size_bytes": "100",
        "sha256": "a" * 64,
        "canonical_sha256": "b" * 64,
        "signed_utc": "2026-08-29T10:00:00Z",
        "signature": "detached-signature",
    }


def _custody_attestation() -> dict:
    return {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "qualification_id": verifier.QUALIFICATION_ID,
        "predecessor_package_sha256": verifier.PREDECESSOR_PACKAGE_SHA256,
        "attestation_role": "EXTERNAL_HOLDOUT_CUSTODIAN",
        "status": "EXTERNAL_CUSTODY_ACCEPTED",
        "phase_required": "BEFORE_EXECUTION",
        "custodian_identity": "external-custodian",
        "custodian_organization": "external-organization",
        "signing_key_id": "external-key",
        "independence_statement": "No implementation-team role.",
        "conflicts_of_interest_statement": "No conflict disclosed.",
        "no_prior_expectation_disclosure_attested": True,
        "holdout_freshness_attested": True,
        "disclosed_prior_manifest_binding": copy.deepcopy(
            verifier.PREDECESSOR_PRIOR_MANIFEST_BINDING
        ),
        "holdout_manifest_binding": {
            "path": "external/holdout-manifest.json",
            "size_bytes": "1000",
            "sha256": "6" * 64,
            "canonical_sha256": "7" * 64,
        },
        "required_case_roles": list(verifier.CASE_ROLES),
        "covered_case_roles": list(verifier.CASE_ROLES),
        "first_valid_unblinding_only_attested": True,
        "no_selective_retry_or_deletion_attested": True,
        "attested_utc": "2026-08-29T10:00:00Z",
        "signature": "detached-signature",
        "external_attestation_artifact": _external_artifact_binding(),
        "nonclaim": "External custody attestation fixture for schema testing only.",
    }


def _independence_attestation() -> dict:
    return {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-independence-attestation-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-independence-attestation/v1",
        "qualification_id": verifier.QUALIFICATION_ID,
        "predecessor_package_sha256": verifier.PREDECESSOR_PACKAGE_SHA256,
        "attestation_role": "EXTERNAL_SCIENTIFIC_INDEPENDENCE_REVIEW",
        "status": "EXTERNAL_INDEPENDENCE_ACCEPTED",
        "phase_required": "BEFORE_EXECUTION",
        "reviewer_identity": "external-reviewer",
        "reviewer_organization": "external-organization",
        "signing_key_id": "external-key",
        "relationship_to_implementation_team": "None.",
        "conflicts_of_interest_statement": "No conflict disclosed.",
        "oracle_independence_accepted": True,
        "eih_independence_accepted": True,
        "budget_review_independence_accepted": True,
        "reviewed_source_bindings": ["c" * 64],
        "attested_utc": "2026-08-29T10:00:00Z",
        "signature": "detached-signature",
        "external_attestation_artifact": _external_artifact_binding(),
        "nonclaim": "External independence attestation fixture for schema testing only.",
    }


def _commitment_record() -> dict:
    return {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-sealed-expectation-commitment-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "qualification_id": verifier.QUALIFICATION_ID,
        "predecessor_package_sha256": verifier.PREDECESSOR_PACKAGE_SHA256,
        "status": "EXPECTATIONS_COMMITTED_SEALED",
        "phase_required": "BEFORE_EXECUTION",
        "expectation_visibility": "UNAVAILABLE_TO_IMPLEMENTATION_TEAM_BEFORE_UNBLINDING",
        "hash_algorithm": "SHA-256",
        "domain_separator": verifier.DOMAIN_SEPARATOR.decode("ascii"),
        "commitment_formula": "SHA256(domain_separator_utf8 || nonce_bytes || canonical_expectation_bytes)",
        "nonce_policy": {
            "minimum_entropy_bits": 256,
            "encoding": "LOWERCASE_HEX_ON_REVEAL",
            "source": "EXTERNAL_CUSTODIAN_CSPRNG",
            "preunblind_visibility": "SECRET_EXTERNAL",
            "nonce_stored_in_this_artifact": False,
        },
        "required_case_roles": list(verifier.CASE_ROLES),
        "required_observable_ids": list(verifier.OBSERVABLES),
        "expectation_package_path": "external/expectations.json",
        "expectation_package_size_bytes": "1000",
        "expectation_package_sha256": "d" * 64,
        "expectation_package_canonical_sha256": "e" * 64,
        "commitment_sha256": "f" * 64,
        "custodian_signing_key_id": "external-key",
        "committed_utc": "2026-08-29T10:00:00Z",
        "signature": "detached-signature",
        "reveal_status": "EXPECTATIONS_STILL_SEALED",
        "nonclaim": "External commitment fixture for schema testing only.",
    }


def _unblinding_record() -> dict:
    return {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-unblinding-record-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-unblinding-record/v1",
        "qualification_id": verifier.QUALIFICATION_ID,
        "predecessor_package_sha256": verifier.PREDECESSOR_PACKAGE_SHA256,
        "status": "UNBLINDED_FIRST_VALID",
        "phase_required": "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
        "custody_state_required": "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
        "first_valid_unblinding_only": True,
        "post_unblinding_tuning_allowed": False,
        "selective_retry_allowed": False,
        "failed_record_retention_required": True,
        "outcomes_generated": True,
        "output_manifest_status": "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
        "output_manifest_path": "external/output-manifest.json",
        "output_manifest_size_bytes": "1000",
        "output_manifest_sha256": "1" * 64,
        "output_manifest_canonical_sha256": "2" * 64,
        "output_committed_utc": "2026-08-29T12:00:00Z",
        "unblind_request_utc": "2026-08-29T12:01:00Z",
        "custodian_authorization_signature": "detached-signature",
        "sealed_expectation_commitment_sha256": "8" * 64,
        "revealed_nonce_hex": "3" * 64,
        "revealed_expectation_package_path": "external/expectations.json",
        "revealed_expectation_package_sha256": "4" * 64,
        "revealed_expectation_package_canonical_sha256": "9" * 64,
        "commitment_verified": True,
        "valid_unblinding_ordinal": 1,
        "second_unblinding_attempted": False,
        "adjudication_status": "AWAITING_ADJUDICATION",
        "adjudication_verdict": None,
        "adjudicated_utc": None,
        "nonclaim": "External unblinding fixture for schema testing only.",
    }


def _bind_unblinding_files(root: Path, record: dict) -> None:
    output_manifest = {
        "schema": "synthetic-output-manifest/v1",
        "status": "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
    }
    output_path = root / record["output_manifest_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = verifier.canonical_json(output_manifest)
    output_path.write_bytes(output_bytes)
    record["output_manifest_size_bytes"] = str(len(output_bytes))
    record["output_manifest_sha256"] = sha256(output_bytes).hexdigest()
    record["output_manifest_canonical_sha256"] = verifier.sha256_data(output_manifest)

    expectations = {
        "schema": "synthetic-expectations/v1",
        "observable": "observable.q3.differential_signal",
        "expected_relation": "WITHIN_FROZEN_BUDGET",
    }
    expectation_path = root / record["revealed_expectation_package_path"]
    expectation_path.parent.mkdir(parents=True, exist_ok=True)
    expectation_bytes = verifier.canonical_json(expectations)
    expectation_path.write_bytes(expectation_bytes)
    record["revealed_expectation_package_sha256"] = sha256(
        expectation_bytes
    ).hexdigest()
    record["revealed_expectation_package_canonical_sha256"] = verifier.sha256_data(
        expectations
    )
    record["sealed_expectation_commitment_sha256"] = verifier.compute_sealed_commitment(
        expectations, record["revealed_nonce_hex"]
    )


def _frozen_budget() -> dict:
    budget = copy.deepcopy(_read_json(PROJECT_ROOT / BUDGET_RELATIVE))
    budget["status"] = "INDEPENDENTLY_REVIEWED_AND_FROZEN"
    for observable in budget["observables"]:
        observable.update(
            {
                "status": "INDEPENDENTLY_REVIEWED_AND_FROZEN",
                "norm_status": "FROZEN",
                "norm": "MAX_COMPONENT",
                "scope_status": "FROZEN",
                "scope": "SYNTHETIC_MODEL_MATCHED",
                "discrimination_margin_status": "FROZEN",
                "discrimination_margin": "1",
                "total_status": "CONSERVATIVE_SUM_FROZEN",
                "total_allocation": "9",
            }
        )
        for component in observable["components"]:
            component.update(
                {
                    "status": "INDEPENDENTLY_REVIEWED_AND_FROZEN",
                    "allocation": "1",
                    "derivation_class": "ANALYTIC_BOUND",
                    "evidence_bindings": [
                        {"path": "evidence/bound.json", "sha256": "5" * 64}
                    ],
                    "review_status": "EXTERNAL_REVIEW_ACCEPTED",
                }
            )
    budget["external_scientific_review"] = {
        "status": "EXTERNAL_BUDGET_REVIEW_ACCEPTED",
        "reviewer_identity": "external-reviewer",
        "reviewer_organization": "external-organization",
        "signing_key_id": "external-key",
        "scientific_acceptance_statement": "Accepted as conservative.",
        "reviewed_utc": "2026-08-29T10:00:00Z",
        "signature": "detached-signature",
    }
    return budget


def _budget_at_status(status: str) -> dict:
    budget = _frozen_budget()
    budget["status"] = status
    if status in {"DRAFT_UNREVIEWED", "FROZEN_AWAITING_EXTERNAL_REVIEW"}:
        component_status = (
            "ALLOCATED_UNREVIEWED"
            if status == "DRAFT_UNREVIEWED"
            else "FROZEN_AWAITING_EXTERNAL_REVIEW"
        )
        for observable in budget["observables"]:
            observable["status"] = status
            for component in observable["components"]:
                component["status"] = component_status
                component["review_status"] = "AWAITING_EXTERNAL_REVIEW"
        budget["external_scientific_review"] = {
            "status": "AWAITING_EXTERNAL_BUDGET_REVIEW",
            "reviewer_identity": None,
            "reviewer_organization": None,
            "signing_key_id": None,
            "scientific_acceptance_statement": None,
            "reviewed_utc": None,
            "signature": None,
        }
    elif status == "INVALIDATED":
        for observable in budget["observables"]:
            observable.update(
                {
                    "status": "INVALIDATED",
                    "norm_status": "INVALIDATED",
                    "scope_status": "INVALIDATED",
                    "discrimination_margin_status": "INVALIDATED",
                    "total_status": "INVALIDATED",
                }
            )
            for component in observable["components"]:
                component["status"] = "INVALIDATED"
                component["review_status"] = "INVALIDATED"
        budget["external_scientific_review"]["status"] = "INVALIDATED"
        budget["postrun_realized_errors_status"] = "INVALIDATED"
    return budget


class V5Solar1PNQ8Q9PrerequisiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.slots, cls.budget, cls.registration, cls.inspection = (
            verifier.inspect_q8q9_prerequisite_package(project_root=PROJECT_ROOT)
        )

    def assertPrerequisiteError(self, code: str, callable_, *args, **kwargs) -> None:
        with self.assertRaises(verifier.Q8Q9PrerequisiteError) as raised:
            callable_(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def test_package_is_design_only_blocked_and_nonauthorizing(self) -> None:
        self.assertEqual(self.inspection.status, "DESIGN_ONLY_BLOCKED")
        self.assertFalse(self.inspection.scientific_evidence_artifact)
        self.assertFalse(self.inspection.outcomes_generated)
        self.assertFalse(self.inspection.execution_authorized)
        self.assertFalse(self.inspection.ready_for_holdout_execution)
        self.assertFalse(self.inspection.unblinding_occurred)
        self.assertEqual(self.inspection.predecessor_commit, verifier.PREDECESSOR_COMMIT)
        self.assertEqual(
            self.inspection.predecessor_package_sha256,
            verifier.PREDECESSOR_PACKAGE_SHA256,
        )
        self.assertRegex(self.inspection.package_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(self.registration["claim_control"]["claim_text"], None)

    def test_no_external_instance_runner_cli_or_trajectory_code_exists(self) -> None:
        package = PROJECT_ROOT / PACKAGE_RELATIVE
        for name in verifier.FORBIDDEN_INSTANCE_NAMES:
            self.assertFalse((package / name).exists(), name)
        module = ast.parse((PROJECT_ROOT / VERIFIER_RELATIVE).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("jxplanetx.solar_1pn", imported)
        self.assertNotIn("jxplanetx.v5_reference_dynamics", imported)
        self.assertNotIn("jxplanetx.v5_implicit_midpoint", imported)
        self.assertFalse(
            any(
                name == "jxplanetx" or name.startswith("jxplanetx.")
                for name in imported
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(
                    isinstance(item, ast.Constant) and item.value == "__main__"
                    for item in ast.walk(node.test)
                )
                for node in ast.walk(module)
            )
        )

    def test_verifier_runs_directly_from_root_without_pythonpath(self) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-B", str(PROJECT_ROOT / VERIFIER_RELATIVE)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

        module_probe = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "from runs.v5_solar_1pn_qualification_q8q9_prerequisites_v1."
                    "verify_prerequisites_v1 import inspect_q8q9_prerequisite_package;"
                    "inspection=inspect_q8q9_prerequisite_package(project_root='.')[3];"
                    "assert inspection.status=='DESIGN_ONLY_BLOCKED'"
                ),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(module_probe.returncode, 0, module_probe.stderr)
        self.assertEqual(module_probe.stdout, "")
        self.assertEqual(module_probe.stderr, "")

    def test_external_slots_are_exactly_null_and_budget_is_unresolved(self) -> None:
        self.assertEqual(self.slots["status"], "DESIGN_ONLY_BLOCKED")
        self.assertEqual(
            {slot["role"] for slot in self.slots["slots"]},
            set(verifier.EXTERNAL_SLOT_SPECS),
        )
        for slot in self.slots["slots"]:
            for field in (
                "path",
                "size_bytes",
                "sha256",
                "canonical_sha256",
                "signer_identity",
                "signing_key_id",
                "signed_utc",
                "signature",
            ):
                self.assertIsNone(slot[field])
        unblinding_slot = next(
            slot for slot in self.slots["slots"] if slot["role"] == "unblinding_record"
        )
        self.assertEqual(unblinding_slot["status"], "NOT_APPLICABLE_PREEXECUTION")
        self.assertEqual(
            unblinding_slot["phase_required"],
            "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
        )
        self.assertNotIn(
            "AWAITING_EXTERNAL_POSTOUTPUT_UNBLINDING_RECORD",
            self.registration["blocked_reasons"],
        )
        verifier._validate_error_budget_semantics(self.budget)

    def test_duplicate_key_float_nonfinite_and_unknown_field_fail_closed(self) -> None:
        with _minimal_clone() as root:
            path = root / REGISTRATION_RELATIVE
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("{", '{\n  "schema": "duplicate",', 1), encoding="utf-8")
            self.assertPrerequisiteError(
                "duplicate_json_key",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            path = root / BUDGET_RELATIVE
            budget = _read_json(path)
            budget["observables"][0]["components"][0]["allocation"] = 1.5
            path.write_text(json.dumps(budget, sort_keys=True), encoding="utf-8")
            registration = _read_json(root / REGISTRATION_RELATIVE)
            binding = next(item for item in registration["artifacts"] if item["role"] == "error_budget_template")
            binding["sha256"] = _file_sha256(path)
            binding["size_bytes"] = str(path.stat().st_size)
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "binary_float_forbidden",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        self.assertPrerequisiteError(
            "nonfinite_json", verifier._load_json_bytes, b'{"x":NaN}', "test"
        )

        with _minimal_clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["silent_default"] = False
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "schema_unknown_field",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_closed_schema_subset_rejects_maxitems_unknown_and_weakened_schemas(self) -> None:
        bounded_array_schema = {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        }
        verifier._validate_schema(["a", "b"], bounded_array_schema, "bounded-array")
        self.assertPrerequisiteError(
            "schema_max_items",
            verifier._validate_schema,
            ["a", "b", "c"],
            bounded_array_schema,
            "bounded-array",
        )

        conditional_schema = {
            "if": {"const": None},
            "then": {"const": None},
            "else": {
                "oneOf": [
                    {"const": "PASS"},
                    {"const": "FAIL"},
                ]
            },
        }
        verifier._validate_schema(None, conditional_schema, "conditional")
        verifier._validate_schema("PASS", conditional_schema, "conditional")
        self.assertPrerequisiteError(
            "schema_one_of",
            verifier._validate_schema,
            "INCONCLUSIVE",
            conditional_schema,
            "conditional",
        )

        unknown_keyword_schema = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "description": "silently ignored before"},
        }
        self.assertPrerequisiteError(
            "unsupported_schema_keyword",
            verifier._validate_schema,
            ["x"],
            unknown_keyword_schema,
            "unknown-keyword",
        )

        open_object_schema = {
            "type": "object",
            "additionalProperties": True,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        }
        self.assertPrerequisiteError(
            "schema_not_closed",
            verifier._validate_schema,
            {"value": "x"},
            open_object_schema,
            "open-object",
        )

        optional_field_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["required_value"],
            "properties": {
                "required_value": {"type": "string"},
                "silently_optional": {"type": "string"},
            },
        }
        self.assertPrerequisiteError(
            "schema_not_closed",
            verifier._validate_schema,
            {"required_value": "x"},
            optional_field_schema,
            "optional-field",
        )

        with _minimal_clone() as root:
            schema_path = (
                root / "schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json"
            )
            weakened = _read_json(schema_path)
            weakened["additionalProperties"] = True
            _write_json(schema_path, weakened)
            _refresh_schema_binding(root, "custody_attestation_schema")
            self.assertPrerequisiteError(
                "schema_binding_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_every_public_validation_path_pins_schema_identity_before_semantics(self) -> None:
        external_documents = {
            "custody_attestation_schema": _custody_attestation,
            "independence_attestation_schema": _independence_attestation,
            "sealed_expectation_commitment_schema": _commitment_record,
            "error_budget_schema": _frozen_budget,
            "unblinding_record_schema": _unblinding_record,
        }
        self.assertEqual(
            set(external_documents) | {"prerequisite_registration_schema"},
            set(verifier.SCHEMA_SPECS),
        )

        for role, document_factory in external_documents.items():
            with self.subTest(role=role), _minimal_clone() as root:
                _, relative_schema_path = verifier.SCHEMA_SPECS[role]
                schema_path = root / relative_schema_path
                altered_schema = _read_json(schema_path)
                altered_schema["properties"]["phase_required"]["const"] = (
                    "ATTACKER_CONTROLLED_PHASE"
                )
                _write_json(schema_path, altered_schema)
                _refresh_schema_binding(root, role)
                matching_document = document_factory()
                matching_document["phase_required"] = "ATTACKER_CONTROLLED_PHASE"
                self.assertPrerequisiteError(
                    "schema_identity_mismatch",
                    verifier.validate_external_artifact,
                    matching_document,
                    schema_role=role,
                    project_root=root,
                    signature_verifier=lambda _: True,
                )

        with _minimal_clone() as root:
            role = "prerequisite_registration_schema"
            _, relative_schema_path = verifier.SCHEMA_SPECS[role]
            schema_path = root / relative_schema_path
            altered_schema = _read_json(schema_path)
            altered_schema["properties"]["schema"]["const"] = (
                "attacker-controlled-registration/v1"
            )
            _write_json(schema_path, altered_schema)
            _refresh_schema_binding(root, role)
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["schema"] = "attacker-controlled-registration/v1"
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "schema_identity_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_external_constants_and_public_budget_surface_fail_closed(self) -> None:
        wrong_phase = _custody_attestation()
        wrong_phase["phase_required"] = "ATTACKER_CONTROLLED_PHASE"
        self.assertPrerequisiteError(
            "external_constant_mismatch",
            verifier.validate_external_artifact,
            wrong_phase,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )

        missing_nonclaim = _independence_attestation()
        missing_nonclaim["nonclaim"] = "   "
        self.assertPrerequisiteError(
            "external_nonclaim_missing",
            verifier.validate_external_artifact,
            missing_nonclaim,
            schema_role="independence_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )

        self.assertFalse(hasattr(verifier, "validate_error_budget_document"))
        self.assertNotIn("validate_error_budget_document", verifier.__all__)

        with _minimal_clone() as root:
            schema_path = (
                root / "schemas/jx-v5-solar-1pn-q8q9-independence-attestation-v1.schema.json"
            )
            unknown = _read_json(schema_path)
            unknown["properties"]["reviewed_source_bindings"]["silently_ignored"] = True
            _write_json(schema_path, unknown)
            _refresh_schema_binding(root, "independence_attestation_schema")
            self.assertPrerequisiteError(
                "schema_binding_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            schema_path = (
                root / "schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json"
            )
            weakened = _read_json(schema_path)
            weakened["$defs"]["caseRoleArray"]["maxItems"] = 10
            _write_json(schema_path, weakened)
            _refresh_schema_binding(root, "custody_attestation_schema")
            self.assertPrerequisiteError(
                "schema_binding_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_hash_size_path_escape_and_symlink_fail_closed(self) -> None:
        with _minimal_clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["artifacts"][0]["sha256"] = "0" * 64
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "raw_digest_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["artifacts"][0]["size_bytes"] = "1"
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "size_mismatch",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["artifacts"][0]["path"] = "../escape.json"
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "schema_pattern",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            target = root / SLOTS_RELATIVE
            backup = target.with_name("slots-real.json")
            target.rename(backup)
            target.symlink_to(backup)
            self.assertPrerequisiteError(
                "symlink_forbidden",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_fabricated_slot_instance_authorization_and_early_claim_are_rejected(self) -> None:
        with _minimal_clone() as root:
            slots = _read_json(root / SLOTS_RELATIVE)
            slots["slots"][0]["signer_identity"] = "invented-custodian"
            _write_json(root / SLOTS_RELATIVE, slots)
            _refresh_artifact_binding(root, "external_artifact_slots")
            self.assertPrerequisiteError(
                "fabricated_external_artifact",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        with _minimal_clone() as root:
            forbidden = root / PACKAGE_RELATIVE / verifier.FORBIDDEN_INSTANCE_NAMES[0]
            forbidden.write_text("{}\n", encoding="utf-8")
            self.assertPrerequisiteError(
                "fabricated_external_instance",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

        for field in (
            "scientific_evidence_artifact",
            "outcomes_generated",
            "execution_authorized",
            "ready_for_holdout_execution",
            "unblinding_occurred",
        ):
            with self.subTest(field=field), _minimal_clone() as root:
                registration = _read_json(root / REGISTRATION_RELATIVE)
                registration[field] = True
                _write_json(root / REGISTRATION_RELATIVE, registration)
                self.assertPrerequisiteError(
                    "premature_authorization",
                    verifier.inspect_q8q9_prerequisite_package,
                    project_root=root,
                )

        with _minimal_clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["claim_control"]["claim_emitted"] = True
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertPrerequisiteError(
                "early_claim",
                verifier.inspect_q8q9_prerequisite_package,
                project_root=root,
            )

    def test_unsigned_late_incomplete_and_nonindependent_attestations_fail(self) -> None:
        custody = _custody_attestation()
        verifier.validate_external_artifact(
            custody,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
            execution_started_utc="2026-08-29T11:00:00Z",
            signature_verifier=lambda _: True,
        )
        self.assertPrerequisiteError(
            "signature_verification_required",
            verifier.validate_external_artifact,
            custody,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
        )
        self.assertPrerequisiteError(
            "invalid_external_signature",
            verifier.validate_external_artifact,
            custody,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: False,
        )
        unsigned = copy.deepcopy(custody)
        unsigned["signature"] = None
        self.assertPrerequisiteError(
            "unsigned_custody_attestation",
            verifier.validate_external_artifact,
            unsigned,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
        )
        late = copy.deepcopy(custody)
        late["attested_utc"] = "2026-08-29T12:00:00Z"
        self.assertPrerequisiteError(
            "late_attestation",
            verifier.validate_external_artifact,
            late,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
            execution_started_utc="2026-08-29T11:00:00Z",
        )
        gap = copy.deepcopy(custody)
        gap["covered_case_roles"].remove("EIH_LIMIT")
        self.assertPrerequisiteError(
            "schema_min_items",
            verifier.validate_external_artifact,
            gap,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
        )
        wrong_required_order = copy.deepcopy(custody)
        wrong_required_order["required_case_roles"][0:2] = reversed(
            wrong_required_order["required_case_roles"][0:2]
        )
        self.assertPrerequisiteError(
            "holdout_role_gap",
            verifier.validate_external_artifact,
            wrong_required_order,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
        )
        reordered_coverage = copy.deepcopy(custody)
        reordered_coverage["covered_case_roles"].reverse()
        verifier.validate_external_artifact(
            reordered_coverage,
            schema_role="custody_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )

        independence = _independence_attestation()
        verifier.validate_external_artifact(
            independence,
            schema_role="independence_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )
        independence["oracle_independence_accepted"] = False
        self.assertPrerequisiteError(
            "independence_not_accepted",
            verifier.validate_external_artifact,
            independence,
            schema_role="independence_attestation_schema",
            project_root=PROJECT_ROOT,
        )

        empty_sources = _independence_attestation()
        empty_sources["reviewed_source_bindings"] = []
        self.assertPrerequisiteError(
            "schema_min_items",
            verifier.validate_external_artifact,
            empty_sources,
            schema_role="independence_attestation_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )
        independence_schema = _read_json(
            PROJECT_ROOT
            / "schemas/jx-v5-solar-1pn-q8q9-independence-attestation-v1.schema.json"
        )
        self.assertEqual(
            independence_schema["properties"]["reviewed_source_bindings"]["minItems"],
            1,
        )

    def test_commitment_requires_secret_nonce_canonical_bytes_and_matching_reveal(self) -> None:
        expectations = {"schema": "synthetic-expectations/v1", "value": "1"}
        nonce = "ab" * 32
        commitment = verifier.compute_sealed_commitment(expectations, nonce)
        revealed = verifier.verify_sealed_reveal(
            commitment, nonce, verifier.canonical_json(expectations)
        )
        self.assertEqual(revealed, expectations)
        self.assertPrerequisiteError(
            "nonce_policy_violation",
            verifier.compute_sealed_commitment,
            expectations,
            "",
        )
        self.assertPrerequisiteError(
            "binary_float_forbidden",
            verifier.compute_sealed_commitment,
            {"value": 1.5},
            nonce,
        )
        self.assertPrerequisiteError(
            "commitment_reveal_mismatch",
            verifier.verify_sealed_reveal,
            commitment,
            nonce,
            verifier.canonical_json({"schema": "synthetic-expectations/v1", "value": "2"}),
        )
        self.assertPrerequisiteError(
            "noncanonical_expectation_bytes",
            verifier.verify_sealed_reveal,
            commitment,
            nonce,
            b'{"schema": "synthetic-expectations/v1", "value": "1"}',
        )

        record = _commitment_record()
        verifier.validate_external_artifact(
            record,
            schema_role="sealed_expectation_commitment_schema",
            project_root=PROJECT_ROOT,
            execution_started_utc="2026-08-29T11:00:00Z",
            signature_verifier=lambda _: True,
        )
        record["committed_utc"] = "2026-08-29T12:00:00Z"
        self.assertPrerequisiteError(
            "late_expectation_commitment",
            verifier.validate_external_artifact,
            record,
            schema_role="sealed_expectation_commitment_schema",
            project_root=PROJECT_ROOT,
            execution_started_utc="2026-08-29T11:00:00Z",
        )

    def test_holdout_roster_gaps_prior_collisions_and_duplicates_are_rejected(self) -> None:
        cases = [
            {
                "case_roles": [role],
                "scientific_fingerprint_sha256": sha256(role.encode()).hexdigest(),
            }
            for role in verifier.CASE_ROLES
        ]
        verifier.validate_holdout_roster(cases, [])
        self.assertPrerequisiteError(
            "holdout_role_gap", verifier.validate_holdout_roster, cases[:-1], []
        )
        self.assertPrerequisiteError(
            "prior_case_collision",
            verifier.validate_holdout_roster,
            cases,
            [cases[0]["scientific_fingerprint_sha256"]],
        )
        duplicate = copy.deepcopy(cases)
        duplicate[-1]["scientific_fingerprint_sha256"] = duplicate[0][
            "scientific_fingerprint_sha256"
        ]
        self.assertPrerequisiteError(
            "duplicate_holdout", verifier.validate_holdout_roster, duplicate, []
        )

    def test_illegal_transitions_early_or_repeated_unblinding_and_mutation_invalidate(self) -> None:
        self.assertEqual(
            verifier.transition_state(
                "custody",
                "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
                "UNBLIND_REQUESTED",
            ),
            "UNBLIND_REQUESTED",
        )
        self.assertEqual(
            verifier.transition_state(
                "custody", "EXPECTATIONS_COMMITTED_SEALED", "UNBLIND_REQUESTED"
            ),
            "INVALIDATED",
        )
        self.assertEqual(
            verifier.transition_state(
                "custody", "UNBLIND_REQUESTED", "UNBLIND_REQUESTED"
            ),
            "INVALIDATED",
        )
        self.assertEqual(
            verifier.transition_state(
                "custody", "UNBLINDED_FIRST_VALID", "UNBLIND_REQUESTED"
            ),
            "INVALIDATED",
        )
        self.assertPrerequisiteError(
            "unblinding_before_output_commit",
            verifier.validate_unblinding_attempt,
            "RUNNING_EXPECTATIONS_SEALED",
            1,
        )
        self.assertPrerequisiteError(
            "repeated_unblinding",
            verifier.validate_unblinding_attempt,
            "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
            2,
        )
        self.assertPrerequisiteError(
            "output_mutation_after_commit",
            verifier.enforce_output_immutability,
            "a" * 64,
            "b" * 64,
            "UNBLIND_REQUESTED",
        )
        self.assertPrerequisiteError(
            "post_outcome_budget_edit",
            verifier.enforce_budget_immutability,
            "a" * 64,
            "b" * 64,
            outcomes_generated=True,
        )

        with _minimal_clone() as root:
            record = _unblinding_record()
            _bind_unblinding_files(root, record)
            verifier.validate_external_artifact(
                record,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )
            record["second_unblinding_attempted"] = True
            self.assertPrerequisiteError(
                "repeated_unblinding",
                verifier.validate_external_artifact,
                record,
                schema_role="unblinding_record_schema",
                project_root=root,
            )

    def test_unblinding_recomputes_commitment_and_closes_adjudication(self) -> None:
        with _minimal_clone() as root:
            record = _unblinding_record()
            _bind_unblinding_files(root, record)
            verifier.validate_external_artifact(
                record,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

            forged_boolean = copy.deepcopy(record)
            forged_boolean["sealed_expectation_commitment_sha256"] = "0" * 64
            self.assertTrue(forged_boolean["commitment_verified"])
            self.assertPrerequisiteError(
                "commitment_reveal_mismatch",
                verifier.validate_external_artifact,
                forged_boolean,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

            incomplete_adjudication = copy.deepcopy(record)
            incomplete_adjudication["status"] = "ADJUDICATED_RETAINED"
            self.assertPrerequisiteError(
                "adjudication_state_mismatch",
                verifier.validate_external_artifact,
                incomplete_adjudication,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

            adjudicated = copy.deepcopy(record)
            adjudicated.update(
                {
                    "status": "ADJUDICATED_RETAINED",
                    "adjudication_status": "ADJUDICATED_RETAINED",
                    "adjudication_verdict": "PASS",
                    "adjudicated_utc": "2026-08-29T12:02:00Z",
                }
            )
            verifier.validate_external_artifact(
                adjudicated,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

            bad_verdict = copy.deepcopy(adjudicated)
            bad_verdict["adjudication_verdict"] = "CUSTOM_VERDICT"
            self.assertPrerequisiteError(
                "schema_enum",
                verifier.validate_external_artifact,
                bad_verdict,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

            output_path = root / adjudicated["output_manifest_path"]
            output_path.write_bytes(output_path.read_bytes() + b"\n")
            self.assertPrerequisiteError(
                "size_mismatch",
                verifier.validate_external_artifact,
                adjudicated,
                schema_role="unblinding_record_schema",
                project_root=root,
                signature_verifier=lambda _: True,
            )

    def test_error_budget_lifecycle_states_are_exact_and_not_ready_early(self) -> None:
        for status in (
            "DRAFT_UNREVIEWED",
            "FROZEN_AWAITING_EXTERNAL_REVIEW",
            "INDEPENDENTLY_REVIEWED_AND_FROZEN",
            "INVALIDATED",
        ):
            with self.subTest(status=status):
                verifier._validate_error_budget_semantics(_budget_at_status(status))

        for mislabeled_status in (
            "DRAFT_UNREVIEWED",
            "FROZEN_AWAITING_EXTERNAL_REVIEW",
            "INVALIDATED",
        ):
            with self.subTest(mislabeled_status=mislabeled_status):
                mislabeled = _frozen_budget()
                mislabeled["status"] = mislabeled_status
                self.assertPrerequisiteError(
                    "budget_state_mismatch",
                    verifier._validate_error_budget_semantics,
                    mislabeled,
                )

        incoherent_postrun = _frozen_budget()
        incoherent_postrun["postrun_realized_errors_status"] = (
            "BOUND_SEPARATELY_AFTER_EXECUTION"
        )
        self.assertPrerequisiteError(
            "postrun_status_mismatch",
            verifier._validate_error_budget_semantics,
            incoherent_postrun,
        )

        draft = _budget_at_status("DRAFT_UNREVIEWED")
        self.assertPrerequisiteError(
            "budget_not_ready",
            verifier.validate_external_artifact,
            draft,
            schema_role="error_budget_schema",
            project_root=PROJECT_ROOT,
            signature_verifier=lambda _: True,
        )

    def test_error_budget_missing_component_zero_wrong_sum_units_and_norm_fail(self) -> None:
        budget = _frozen_budget()
        verifier._validate_error_budget_semantics(budget)

        missing = copy.deepcopy(budget)
        missing["observables"][0]["components"].pop()
        self.assertPrerequisiteError(
            "incomplete_roster", verifier._validate_error_budget_semantics, missing
        )

        zero = copy.deepcopy(budget)
        zero["observables"][0]["components"][0]["allocation"] = "0"
        zero["observables"][0]["total_allocation"] = "8"
        self.assertPrerequisiteError(
            "unjustified_zero", verifier._validate_error_budget_semantics, zero
        )

        wrong_total = copy.deepcopy(budget)
        wrong_total["observables"][0]["total_allocation"] = "8"
        self.assertPrerequisiteError(
            "conservative_sum_mismatch",
            verifier._validate_error_budget_semantics,
            wrong_total,
        )

        unit = copy.deepcopy(budget)
        unit["observables"][0]["components"][0]["unit_ref"] = "unit.au"
        self.assertPrerequisiteError(
            "unit_mismatch", verifier._validate_error_budget_semantics, unit
        )

        observable = budget["observables"][0]
        realized = {
            "unit_ref": observable["unit_ref"],
            "norm": observable["norm"],
            "components": {name: "0" for name in verifier.COMPONENTS},
            "total_error": "0",
        }
        verifier.validate_realized_error_record(observable, realized)
        wrong_norm = copy.deepcopy(realized)
        wrong_norm["norm"] = "EUCLIDEAN"
        self.assertPrerequisiteError(
            "norm_mismatch",
            verifier.validate_realized_error_record,
            observable,
            wrong_norm,
        )

        tiny = Decimal("0." + "0" * 999 + "1")
        self.assertEqual(
            verifier._exact_decimal_sum((Decimal("1"), tiny)),
            Decimal("1." + "0" * 999 + "1"),
        )

    def test_component_overflow_cannot_be_hidden_by_total_budget(self) -> None:
        budget = _frozen_budget()
        observable = budget["observables"][0]
        realized = {
            "unit_ref": observable["unit_ref"],
            "norm": observable["norm"],
            "components": {name: "0" for name in verifier.COMPONENTS},
            "total_error": "2",
        }
        realized["components"]["oracle"] = "2"
        self.assertLessEqual(int(realized["total_error"]), int(observable["total_allocation"]))
        self.assertPrerequisiteError(
            "component_overflow",
            verifier.validate_realized_error_record,
            observable,
            realized,
        )
