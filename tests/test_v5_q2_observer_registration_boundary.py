"""Adversarial tests for the blocked Q2 observer registration boundary."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
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
PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_q2_observer_registration_v1")
REGISTRATION_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
TEMPLATE_RELATIVE = PACKAGE_RELATIVE / "external_registration_template_v1.json"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_registration_v1.py"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-q2-observer-registration-boundary-v1.schema.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "v5_q2_observer_registration_verifier", PROJECT_ROOT / VERIFIER_RELATIVE
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load Q2 registration verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


ALL_BOUND_RELATIVES = (
    REGISTRATION_RELATIVE,
    TEMPLATE_RELATIVE,
    VERIFIER_RELATIVE,
    README_RELATIVE,
    SCHEMA_RELATIVE,
    *(Path(item["path"]) for item in VERIFIER.CANDIDATE_FILES),
    *(Path(item["registration"]["path"]) for item in VERIFIER.PREDECESSORS.values()),
    *(Path(item["path"]) for item in VERIFIER.Q8Q9_SCHEMA_BINDINGS),
)


@contextmanager
def _clone():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in ALL_BOUND_RELATIVES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, destination)
        yield root


def _refresh_template_binding(root: Path) -> None:
    template_path = root / TEMPLATE_RELATIVE
    registration = _read_json(root / REGISTRATION_RELATIVE)
    binding = registration["external_registration_template"]
    binding["sha256"] = _sha256(template_path)
    binding["size_bytes"] = str(template_path.stat().st_size)
    binding["canonical_sha256"] = _canonical_sha256(_read_json(template_path))
    _write_json(root / REGISTRATION_RELATIVE, registration)


def _configuration(identifier: str, tolerance: str):
    fixed = VERIFIER.FIXED_CONFIGURATION
    configuration = {
        "configuration_id": identifier,
        "root_bracket_width_tolerance": tolerance,
        "maximum_relative_off_plane": "0",
        "maximum_root_iterations": fixed["maximum_root_iterations"],
        "certificate_precision": fixed["certificate_precision"],
        "context": {
            "context_id": fixed["context_id"],
            "precision": fixed["precision"],
            "rounding": fixed["rounding"],
            "emin": fixed["emin"],
            "emax": fixed["emax"],
            "capitals": fixed["capitals"],
            "clamp": fixed["clamp"],
        },
        "expected_event_count": fixed["expected_event_count"],
        "atan_guard_digits": fixed["atan_guard_digits"],
        "maximum_atan_series_iterations": fixed["maximum_atan_series_iterations"],
        "epoch_unit_id": fixed["epoch_unit_id"],
        "root_tolerance_unit_id": fixed["root_tolerance_unit_id"],
        "angle_unit_id": fixed["angle_unit_id"],
        "observer_id": fixed["observer_id"],
        "dense_state_method_id": fixed["dense_state_method_id"],
        "root_method_id": fixed["root_method_id"],
        "root_isolation_method_id": fixed["root_isolation_method_id"],
        "angle_method_id": fixed["angle_method_id"],
        "unwrap_method_id": fixed["unwrap_method_id"],
        "regression_method_id": fixed["regression_method_id"],
        "observer_configuration_sha256": "0" * 64,
    }
    configuration["observer_configuration_sha256"] = VERIFIER._configuration_hash(
        configuration
    )
    return configuration


def _completed_registration():
    document = _read_json(PROJECT_ROOT / TEMPLATE_RELATIVE)
    document["status"] = "EXTERNAL_Q2_OBSERVER_REGISTERED_NONAUTHORIZING"
    document["evidence_class"] = "EXTERNAL_ATTESTATION_METADATA"
    document["nonclaim"] = VERIFIER.EXTERNAL_NONCLAIM
    document["runtime_registration"] = {
        "status": "EXTERNAL_RUNTIME_BOUND",
        "runtime_id": "external.cpython.q2.v1",
        "python_implementation": "CPython",
        "python_version": "3.12.11",
        "platform_id": "linux.x86_64.glibc",
        "locale_id": "locale.C_UTF_8",
        "runtime_artifact_kind": "OCI_IMAGE",
        "runtime_artifact_locator": "oci://independent.example/q2@sha256:" + "1" * 64,
        "runtime_artifact_size_bytes": "123456",
        "runtime_artifact_sha256": "1" * 64,
        "python_executable_sha256": "2" * 64,
        "stdlib_archive_sha256": "3" * 64,
        "decimal_backend_id": "cpython._decimal.libmpdec",
        "decimal_module_sha256": "4" * 64,
        "libmpdec_version": "2.5.1",
        "invocation_manifest_sha256": "5" * 64,
        "environment_manifest_sha256": "6" * 64,
    }
    document["dependency_registration"] = {
        "status": "EXTERNAL_DEPENDENCY_MANIFEST_BOUND",
        "manifest_id": "external.q2.dependencies.v1",
        "manifest_locator": "oci://independent.example/q2-dependencies@sha256:" + "7" * 64,
        "manifest_size_bytes": "789",
        "manifest_sha256": "7" * 64,
        "manifest_canonical_sha256": "8" * 64,
        "direct_stdlib_imports": list(VERIFIER.DIRECT_STDLIB_IMPORTS),
        "third_party_distributions": [],
    }
    document["configuration_registration"] = {
        "status": "EXTERNAL_CONFIGURATION_ROSTER_BOUND",
        "configurations": [
            _configuration("external.q2.observer.t20.v1", "1E-20"),
            _configuration("external.q2.observer.t30.v1", "1E-30"),
        ],
    }
    document["independence_attestation"] = {
        "status": "EXTERNAL_Q2_INDEPENDENCE_ATTESTED",
        "reviewer_identity": "Independent Reviewer",
        "reviewer_organization": "Independent Organization",
        "relationship_to_implementation_team": "No employment or authorship relationship",
        "conflicts_of_interest_statement": "No conflicts of interest identified",
        "reviewer_independence_accepted": True,
        "conflicts_of_interest_absent_attested": True,
        "source_isolation_accepted": True,
        "runtime_binding_reviewed": True,
        "dependency_closure_reviewed": True,
        "configuration_roster_reviewed": True,
        "no_holdout_access_attested": True,
        "no_execution_attested": True,
        "attested_utc": "2026-08-30T12:00:00Z",
        "signature_profile_id": "external.signature.profile.v1",
        "signing_key_id": "external.key.q2.v1",
        "signed_payload_sha256": None,
        "signature": None,
    }
    payload = VERIFIER.external_signed_payload(document)
    document["independence_attestation"]["signed_payload_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    document["independence_attestation"]["signature"] = "synthetic-test-signature"
    return document


def _resign(document) -> None:
    document["independence_attestation"]["signed_payload_sha256"] = None
    document["independence_attestation"]["signature"] = None
    payload = VERIFIER.external_signed_payload(document)
    document["independence_attestation"]["signed_payload_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    document["independence_attestation"]["signature"] = "synthetic-test-signature"


class Q2ObserverBoundaryPackageTests(unittest.TestCase):
    def assertRejected(self, code, function, *args, **kwargs):
        with self.assertRaises(VERIFIER.Q2ObserverRegistrationError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_package_verifies_read_only_and_remains_blocked(self):
        before = {relative: _sha256(PROJECT_ROOT / relative) for relative in ALL_BOUND_RELATIVES}
        inspection = VERIFIER.verify_package(PROJECT_ROOT)
        self.assertEqual(inspection["status"], "DESIGN_ONLY_BLOCKED")
        self.assertEqual(inspection["source_tuple_sha256"], VERIFIER.OBSERVER_SOURCE_TUPLE_SHA256)
        for field in (
            "scientific_evidence_artifact", "outcomes_generated",
            "execution_authorized", "registry_authorized", "ready",
        ):
            self.assertFalse(inspection[field])
        after = {relative: _sha256(PROJECT_ROOT / relative) for relative in ALL_BOUND_RELATIVES}
        self.assertEqual(before, after)

    def test_verifier_is_self_contained_metadata_only_and_has_no_cli(self):
        source = (PROJECT_ROOT / VERIFIER_RELATIVE).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any(name.startswith("jxplanetx") for name in imported))
        self.assertFalse(any(name.startswith("observer") for name in imported))
        self.assertFalse(any(name.startswith("oracle") for name in imported))
        self.assertNotIn("argparse", imported)
        self.assertNotIn("__main__", source)
        self.assertNotIn("subprocess", imported)

    def test_verifier_runs_from_root_without_pythonpath(self):
        code = (
            "import importlib.util,pathlib,sys;"
            "p=pathlib.Path('runs/v5_solar_1pn_q2_observer_registration_v1/verify_registration_v1.py');"
            "s=importlib.util.spec_from_file_location('direct_q2reg',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "assert m.verify_package(pathlib.Path.cwd())['status']=='DESIGN_ONLY_BLOCKED'"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
            env=environment, check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_schema_is_closed_and_pinned(self):
        schema = _read_json(PROJECT_ROOT / SCHEMA_RELATIVE)

        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        inspect(schema)
        self.assertEqual(_sha256(PROJECT_ROOT / SCHEMA_RELATIVE), VERIFIER.SCHEMA_RAW_SHA256)
        self.assertEqual(_canonical_sha256(schema), VERIFIER.SCHEMA_CANONICAL_SHA256)

    def test_exact_source_tuple_and_predecessor_identity_kinds(self):
        registration = _read_json(PROJECT_ROOT / REGISTRATION_RELATIVE)
        self.assertEqual(registration["observer_candidate"]["files"], list(VERIFIER.CANDIDATE_FILES))
        self.assertEqual(
            registration["predecessors"]["execution_prerequisites"]["identity_kind"],
            "REGISTRATION_CANONICAL_SHA256",
        )
        self.assertNotIn("9b16", json.dumps(registration))
        self.assertEqual(
            registration["predecessors"]["q8q9_prerequisites"]["identity_sha256"],
            "1d9008ae6a3c3a79edd06c3b01f6ab2a662246633188339c0d7f608f5712080b",
        )

    def test_template_is_fully_null_and_full_budget_role_is_not_duplicated(self):
        template = _read_json(PROJECT_ROOT / TEMPLATE_RELATIVE)
        VERIFIER._template_semantics(template)
        roles = [item["role"] for item in template["downstream_q8q9_slots"]]
        self.assertEqual(
            roles,
            ["custody_attestation", "sealed_expectation_commitment", "qualification_error_budget"],
        )
        self.assertNotIn("q2_error_budget", json.dumps(template))

    def test_implemented_and_unresolved_scopes_are_exact(self):
        registration = _read_json(PROJECT_ROOT / REGISTRATION_RELATIVE)
        self.assertEqual(tuple(registration["implemented_scope"]), VERIFIER.IMPLEMENTED_SCOPE)
        self.assertEqual(tuple(registration["unresolved_scope"]), VERIFIER.UNRESOLVED_SCOPE)
        self.assertIn("ANALYTIC_6PI_COMPARISON", registration["unresolved_scope"])
        self.assertIn("DECIMAL_OLS_SLOPE", registration["implemented_scope"])

    def test_successor_and_q8_phase_contradiction_are_explicit_blockers(self):
        blockers = set(VERIFIER.BLOCKED_REASONS)
        self.assertIn("FROZEN_QUALIFICATION_OMITS_CONCRETE_Q2_OBSERVER_IDENTITY", blockers)
        self.assertIn(
            "FROZEN_QUALIFICATION_Q8_PHASE_CONTRADICTION_REQUIRES_CONTENT_DERIVED_SUCCESSOR",
            blockers,
        )
        template = _read_json(PROJECT_ROOT / TEMPLATE_RELATIVE)
        self.assertEqual(
            template["unresolved_prerequisites"]["qualification_successor"]["value"], None
        )

    def test_duplicate_float_and_nonfinite_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for payload, code in (
                ('{"x":1,"x":2}', "duplicate_json_key"),
                ('{"x":1.5}', "binary_float_json"),
                ('{"x":NaN}', "nonfinite_json"),
            ):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    self.assertRejected(code, VERIFIER._load_json, path, "bad")

    def test_unsafe_paths_symlinks_and_byte_drift_fail_closed(self):
        for path in ("/tmp/x", "../x", "runs/../x", "runs//x"):
            self.assertRejected(
                "invalid_repository_path", VERIFIER._safe_file, PROJECT_ROOT, path, "bad"
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "real/file").write_text("x", encoding="utf-8")
            os.symlink(root / "real", root / "link")
            self.assertRejected("symlink_forbidden", VERIFIER._safe_file, root, "link/file", "bad")
        with _clone() as root:
            with (root / TEMPLATE_RELATIVE).open("a", encoding="utf-8") as stream:
                stream.write(" ")
            self.assertRejected("size_mismatch", VERIFIER.verify_package, root)

    def test_weakened_schema_is_rejected_even_if_registration_is_rehashed(self):
        with _clone() as root:
            schema = _read_json(root / SCHEMA_RELATIVE)
            schema["properties"]["nonclaim"] = {"type": "string"}
            _write_json(root / SCHEMA_RELATIVE, schema)
            registration = _read_json(root / REGISTRATION_RELATIVE)
            binding = registration["schema_binding"]
            binding["sha256"] = _sha256(root / SCHEMA_RELATIVE)
            binding["size_bytes"] = str((root / SCHEMA_RELATIVE).stat().st_size)
            binding["canonical_sha256"] = _canonical_sha256(schema)
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertRejected("schema_identity_mismatch", VERIFIER.verify_package, root)

    def test_unknown_schema_keyword_and_maxitems_are_enforced(self):
        schema = _read_json(PROJECT_ROOT / SCHEMA_RELATIVE)
        unknown = copy.deepcopy(schema)
        unknown["inventedKeyword"] = True
        self.assertRejected(
            "unsupported_schema_keyword", VERIFIER._validate_schema,
            _read_json(PROJECT_ROOT / REGISTRATION_RELATIVE), unknown, "unknown",
        )
        wrapper = {"$ref": "#/$defs/externalRegistrationDocument", "$defs": schema["$defs"]}
        template = _read_json(PROJECT_ROOT / TEMPLATE_RELATIVE)
        template["configuration_registration"]["configurations"] = [
            _configuration("x.a", "1E-20"),
            _configuration("x.b", "1E-30"),
            _configuration("x.c", "1E-20"),
        ]
        self.assertRejected(
            "schema_max_items", VERIFIER._validate_schema, template, wrapper, "too many"
        )

    def test_partial_template_fill_and_authority_claim_fail_closed(self):
        with _clone() as root:
            template = _read_json(root / TEMPLATE_RELATIVE)
            template["runtime_registration"]["runtime_id"] = "fabricated.runtime"
            _write_json(root / TEMPLATE_RELATIVE, template)
            _refresh_template_binding(root)
            self.assertRejected("fabricated_external_runtime", VERIFIER.verify_package, root)
        with _clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["execution_authorized"] = True
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertRejected("schema_const", VERIFIER.verify_package, root)

    def test_extra_file_directory_and_symlink_package_entries_fail(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind), _clone() as root:
                extra = root / PACKAGE_RELATIVE / "unexpected"
                if kind == "file":
                    extra.write_text("x", encoding="utf-8")
                elif kind == "directory":
                    extra.mkdir()
                else:
                    os.symlink(root / README_RELATIVE, extra)
                self.assertRejected("package_roster_mismatch", VERIFIER.verify_package, root)


class Q2ExternalRegistrationTests(unittest.TestCase):
    def assertRejected(
        self,
        code,
        document,
        *,
        verifier=lambda _payload, _attestation: True,
        project_root=PROJECT_ROOT,
    ):
        with self.assertRaises(VERIFIER.Q2ObserverRegistrationError) as caught:
            VERIFIER.validate_external_registration(
                document,
                project_root=project_root,
                signature_verifier=verifier,
            )
        self.assertEqual(caught.exception.code, code)

    def test_completed_record_validates_but_remains_nonauthorizing(self):
        document = _completed_registration()
        inspection = VERIFIER.validate_external_registration(
            document,
            project_root=PROJECT_ROOT,
            signature_verifier=lambda payload, attestation: (
                hashlib.sha256(payload).hexdigest()
                == attestation["signed_payload_sha256"]
            ),
        )
        self.assertEqual(inspection["status"], "EXTERNAL_Q2_OBSERVER_REGISTERED_NONAUTHORIZING")
        self.assertRegex(inspection["boundary_package_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(inspection["configuration_sha256s"]), 2)
        for field in (
            "scientific_evidence_artifact", "outcomes_generated",
            "execution_authorized", "registry_authorized", "ready",
        ):
            self.assertFalse(inspection[field])

    def test_signature_callback_is_required_and_false_or_error_fails(self):
        document = _completed_registration()
        with self.assertRaises(VERIFIER.Q2ObserverRegistrationError) as caught:
            VERIFIER.validate_external_registration(document, project_root=PROJECT_ROOT)
        self.assertEqual(caught.exception.code, "signature_verification_required")
        self.assertRejected("invalid_external_signature", document, verifier=lambda _p, _a: False)

        def broken(_payload, _attestation):
            raise RuntimeError("broken trust service")

        self.assertRejected("signature_verification_failed", document, verifier=broken)

    def test_mutating_signature_callback_cannot_change_validated_snapshot(self):
        document = _completed_registration()
        original_runtime_id = document["runtime_registration"]["runtime_id"]
        original_configuration_sha256s = tuple(
            item["observer_configuration_sha256"]
            for item in document["configuration_registration"]["configurations"]
        )

        def mutating(_payload, attestation):
            attestation["signing_key_id"] = "mutated.callback.key"
            document["runtime_registration"]["runtime_id"] = "mutated.callback.runtime"
            document["configuration_registration"]["configurations"][0][
                "observer_configuration_sha256"
            ] = "f" * 64
            return True

        inspection = VERIFIER.validate_external_registration(
            document,
            project_root=PROJECT_ROOT,
            signature_verifier=mutating,
        )
        self.assertEqual(inspection["runtime_id"], original_runtime_id)
        self.assertEqual(
            tuple(inspection["configuration_sha256s"]),
            original_configuration_sha256s,
        )

    def test_signature_callback_cannot_mutate_bound_package_during_validation(self):
        document = _completed_registration()
        with _clone() as root:
            def mutating_package(_payload, _attestation):
                with (root / README_RELATIVE).open("a", encoding="utf-8") as stream:
                    stream.write(" ")
                return True

            self.assertRejected(
                "size_mismatch",
                document,
                verifier=mutating_package,
                project_root=root,
            )

    def test_external_validation_requires_present_undrifted_package_and_candidate(self):
        document = _completed_registration()
        candidate = Path(VERIFIER.CANDIDATE_FILES[2]["path"])
        with _clone() as root:
            (root / candidate).unlink()
            self.assertRejected("path_escape", document, project_root=root)
        with _clone() as root:
            with (root / candidate).open("a", encoding="utf-8") as stream:
                stream.write(" ")
            self.assertRejected("size_mismatch", document, project_root=root)
        with _clone() as root:
            (root / REGISTRATION_RELATIVE).unlink()
            self.assertRejected("path_escape", document, project_root=root)
        with _clone() as root:
            registration = _read_json(root / REGISTRATION_RELATIVE)
            registration["ready"] = True
            _write_json(root / REGISTRATION_RELATIVE, registration)
            self.assertRejected("schema_const", document, project_root=root)

    def test_source_runtime_locale_invocation_and_environment_are_mandatory(self):
        for path, value, expected in (
            (("observer_source_binding", "commit"), "0" * 40, "schema_const"),
            (("runtime_registration", "runtime_artifact_sha256"), None, "runtime_unbound"),
            (("runtime_registration", "locale_id"), None, "runtime_unbound"),
            (("runtime_registration", "invocation_manifest_sha256"), None, "runtime_unbound"),
            (("runtime_registration", "environment_manifest_sha256"), None, "runtime_unbound"),
        ):
            with self.subTest(path=path):
                document = _completed_registration()
                document[path[0]][path[1]] = value
                _resign(document)
                self.assertRejected(expected, document)

    def test_runtime_must_be_cpython_and_dependencies_are_exact_stdlib_only(self):
        document = _completed_registration()
        document["runtime_registration"]["python_implementation"] = "PyPy"
        _resign(document)
        self.assertRejected("runtime_implementation_mismatch", document)
        document = _completed_registration()
        document["dependency_registration"]["direct_stdlib_imports"].reverse()
        _resign(document)
        self.assertRejected("dependency_roster_mismatch", document)
        document = _completed_registration()
        document["dependency_registration"]["third_party_distributions"] = ["numpy"]
        _resign(document)
        self.assertRejected("schema_max_items", document)

    def test_exactly_two_distinct_tolerance_configurations_are_required(self):
        document = _completed_registration()
        document["configuration_registration"]["configurations"].pop()
        _resign(document)
        self.assertRejected("configuration_roster_unbound", document)
        document = _completed_registration()
        document["configuration_registration"]["configurations"][1]["configuration_id"] = (
            document["configuration_registration"]["configurations"][0]["configuration_id"]
        )
        document["configuration_registration"]["configurations"][1][
            "observer_configuration_sha256"
        ] = VERIFIER._configuration_hash(
            document["configuration_registration"]["configurations"][1]
        )
        _resign(document)
        self.assertRejected("duplicate_configuration_id", document)
        document = _completed_registration()
        document["configuration_registration"]["configurations"].reverse()
        _resign(document)
        self.assertRejected("configuration_tolerance_mismatch", document)
        document = _completed_registration()
        configuration = document["configuration_registration"]["configurations"][1]
        configuration["maximum_relative_off_plane"] = "0.25"
        configuration["observer_configuration_sha256"] = VERIFIER._configuration_hash(configuration)
        _resign(document)
        self.assertRejected("configuration_comparison_confounded", document)

    def test_external_values_are_not_invented_and_configuration_hash_is_exact(self):
        template = _read_json(PROJECT_ROOT / TEMPLATE_RELATIVE)
        self.assertEqual(template["configuration_registration"]["configurations"], [])
        document = _completed_registration()
        document["configuration_registration"]["configurations"][0][
            "maximum_relative_off_plane"
        ] = "0.25"
        _resign(document)
        self.assertRejected("configuration_digest_mismatch", document)
        document = _completed_registration()
        for configuration in document["configuration_registration"]["configurations"]:
            configuration["maximum_relative_off_plane"] = "0.25"
            configuration["observer_configuration_sha256"] = VERIFIER._configuration_hash(
                configuration
            )
        _resign(document)
        VERIFIER.validate_external_registration(
            document, project_root=PROJECT_ROOT, signature_verifier=lambda _p, _a: True
        )
        document = _completed_registration()
        configuration = document["configuration_registration"]["configurations"][0]
        configuration["maximum_relative_off_plane"] = "1"
        configuration["observer_configuration_sha256"] = VERIFIER._configuration_hash(configuration)
        _resign(document)
        self.assertRejected("schema_pattern", document)

    def test_independence_conflict_and_review_assertions_must_be_true_and_signed(self):
        for field in (
            "reviewer_independence_accepted",
            "conflicts_of_interest_absent_attested",
            "source_isolation_accepted",
            "runtime_binding_reviewed",
            "dependency_closure_reviewed",
            "configuration_roster_reviewed",
            "no_holdout_access_attested",
            "no_execution_attested",
        ):
            with self.subTest(field=field):
                document = _completed_registration()
                document["independence_attestation"][field] = False
                _resign(document)
                self.assertRejected("independence_not_accepted", document)
        document = _completed_registration()
        document["independence_attestation"]["conflicts_of_interest_statement"] = None
        _resign(document)
        self.assertRejected("independence_unattested", document)

    def test_attestation_timestamp_must_be_a_real_canonical_utc_instant(self):
        document = _completed_registration()
        document["independence_attestation"]["attested_utc"] = "2026-99-99T99:99:99Z"
        _resign(document)
        self.assertRejected("invalid_timestamp", document)

    def test_signed_payload_detects_post_signature_mutation(self):
        document = _completed_registration()
        document["runtime_registration"]["python_version"] = "3.12.12"
        self.assertRejected("signed_payload_mismatch", document)

    def test_downstream_q8q9_and_unresolved_slots_cannot_claim_completion(self):
        document = _completed_registration()
        document["downstream_q8q9_slots"][0]["status"] = "AWAITING_EXTERNAL_OTHER"
        _resign(document)
        self.assertRejected("external_registration_mismatch", document)
        document = _completed_registration()
        document["unresolved_prerequisites"]["qualification_successor"]["status"] = (
            "AWAITING_EXTERNAL_OTHER"
        )
        _resign(document)
        self.assertRejected("external_registration_mismatch", document)

    def test_completed_record_cannot_claim_execution_result_or_readiness(self):
        for field in (
            "scientific_evidence_artifact", "outcomes_generated",
            "execution_authorized", "registry_authorized", "ready",
        ):
            with self.subTest(field=field):
                document = _completed_registration()
                document[field] = True
                _resign(document)
                self.assertRejected("schema_const", document)


if __name__ == "__main__":
    unittest.main()
