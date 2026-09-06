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
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_execution_prerequisites_v1")
REGISTRATION_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_execution_prerequisites_v1.py"
Q2_RELATIVE = PACKAGE_RELATIVE / "q2_observer_registration_v1.json"
Q3_RELATIVE = PACKAGE_RELATIVE / "q3_oracle_registration_v1.json"
Q4_RELATIVE = PACKAGE_RELATIVE / "q4_eih_registration_v1.json"
SCHEMA_RELATIVES = (
    Path("schemas/jx-v5-solar-1pn-execution-prerequisite-registration-v1.schema.json"),
    Path("schemas/jx-v5-solar-1pn-q2-observer-registration-v1.schema.json"),
    Path("schemas/jx-v5-solar-1pn-q3-oracle-registration-v1.schema.json"),
    Path("schemas/jx-v5-solar-1pn-q4-eih-registration-v1.schema.json"),
)
PREDECESSOR_RELATIVES = (
    Path("runs/v5_solar_1pn_qualification/qualification_plan_v1.json"),
    Path("runs/v5_solar_1pn_qualification/qualification_inputs_v1.json"),
    Path("runs/v5_solar_1pn_qualification/registration_v1.json"),
)


def _load_module():
    path = PROJECT_ROOT / VERIFIER_RELATIVE
    spec = importlib.util.spec_from_file_location("v5_execution_prerequisite_verifier", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load prerequisite verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module()


def _json(relative: Path, root: Path = PROJECT_ROOT):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _canonical_sha256(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ALL_COPIED_RELATIVES = (
    REGISTRATION_RELATIVE,
    Q2_RELATIVE,
    Q3_RELATIVE,
    Q4_RELATIVE,
    VERIFIER_RELATIVE,
    PACKAGE_RELATIVE / "README.md",
    *SCHEMA_RELATIVES,
    *PREDECESSOR_RELATIVES,
)


@contextmanager
def _copied_repository():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in ALL_COPIED_RELATIVES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, destination)
        yield root


def _rewrite_bound_json(root: Path, relative: Path, mutate, *, canonical: bool = True):
    document = _json(relative, root)
    mutate(document)
    path = root / relative
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    registration = _json(REGISTRATION_RELATIVE, root)
    matches = [
        binding
        for binding in (*registration["artifacts"], *registration["schemas"])
        if binding["path"] == relative.as_posix()
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one binding for {relative}")
    binding = matches[0]
    binding["sha256"] = _file_sha256(path)
    binding["size_bytes"] = str(path.stat().st_size)
    if canonical:
        binding["canonical_sha256"] = _canonical_sha256(document)
    (root / REGISTRATION_RELATIVE).write_text(
        json.dumps(registration, indent=2) + "\n", encoding="utf-8"
    )


class V5Solar1PNExecutionPrerequisiteTests(unittest.TestCase):
    def assertRejected(self, root: Path):
        with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
            VERIFIER.verify_package(root)

    def test_registered_package_verifies_and_is_read_only(self):
        before = {
            relative: _file_sha256(PROJECT_ROOT / relative)
            for relative in ALL_COPIED_RELATIVES
        }
        inspection = VERIFIER.verify_package(PROJECT_ROOT)
        self.assertEqual(inspection["status"], "DESIGN_ONLY_BLOCKED")
        self.assertFalse(inspection["outcomes_generated"])
        self.assertFalse(inspection["registry_authorized"])
        self.assertFalse(inspection["execution_authorized"])
        self.assertFalse(inspection["ready"])
        self.assertFalse(inspection["scientific_evidence_artifact"])
        after = {
            relative: _file_sha256(PROJECT_ROOT / relative)
            for relative in ALL_COPIED_RELATIVES
        }
        self.assertEqual(after, before)

    def test_self_contained_schema_validator_is_applied_to_all_four_documents(self):
        validator = VERIFIER._validate_schema
        with mock.patch.object(
            VERIFIER,
            "_validate_schema",
            wraps=validator,
        ) as observed:
            VERIFIER.verify_package(PROJECT_ROOT)
        self.assertEqual(observed.call_count, 4)

    def test_verifier_runs_from_repository_root_without_pythonpath(self):
        code = """
from pathlib import Path
import importlib.util
p = Path('runs/v5_solar_1pn_execution_prerequisites_v1/verify_execution_prerequisites_v1.py')
s = importlib.util.spec_from_file_location('direct_execution_prerequisite_verifier', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
assert m.verify_package(Path.cwd())['status'] == 'DESIGN_ONLY_BLOCKED'
"""
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_predecessor_binding_is_exact(self):
        predecessor = _json(REGISTRATION_RELATIVE)["predecessor"]
        self.assertEqual(
            predecessor["qualification_id"],
            "jx.v5.solar_1pn.qualification.78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781",
        )
        self.assertEqual(
            predecessor["package_sha256"],
            "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237",
        )
        self.assertEqual(
            predecessor["commit"], "975b1b7002e358a980457acb18c9beaa90aa1c9f"
        )
        for name, relative in zip(
            ("plan", "inputs", "registration"), PREDECESSOR_RELATIVES
        ):
            binding = predecessor[name]
            document = _json(relative)
            self.assertEqual(binding["sha256"], _file_sha256(PROJECT_ROOT / relative))
            self.assertEqual(binding["size_bytes"], str((PROJECT_ROOT / relative).stat().st_size))
            self.assertEqual(binding["canonical_sha256"], _canonical_sha256(document))

    def test_all_four_schemas_are_closed(self):
        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        for relative in SCHEMA_RELATIVES:
            schema = _json(relative)
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            inspect(schema)

    def test_q2_required_machine_fields_are_explicit_and_null(self):
        document = _json(Q2_RELATIVE)
        expected = {
            "initial_event_exclusion_rule", "event_bracket_acceptance_rule",
            "root_method_id", "root_termination_rule", "orbital_orientation_rule",
            "angle_unwrap_rule", "orbit_index_rule", "secular_regression_rule",
            "step_extrapolation_rule", "event_extrapolation_rule",
            "inverse_c_squared_extrapolation_rule", "analytic_target_expression",
            "high_precision_pi_rule",
        }
        self.assertEqual(set(document["observer_rules"]), expected)
        self.assertTrue(all(item["value"] is None for item in document["observer_rules"].values()))
        self.assertEqual(document["protocol_constants"]["analytic_leading_coefficient"], "6*pi")
        self.assertFalse(document["protocol_constants"]["binary_float_trigonometry_allowed"])
        self.assertFalse(document["protocol_constants"]["linear_event_interpolation_allowed"])

    def test_q3_required_machine_fields_and_checkpoint_slots_are_null(self):
        document = _json(Q3_RELATIVE)
        required_method = {
            "method_id", "exact_order", "source_citation", "method_coefficients_binding",
            "rhs_transcription_binding", "force_ledger_binding", "error_control_policy",
            "dense_output_or_exact_checkpoint_policy", "event_handling", "controller_binding",
        }
        self.assertEqual(set(document["method_requirements"]), required_method)
        self.assertTrue(all(item["value"] is None for item in document["method_requirements"].values()))
        self.assertEqual(document["protocol_constants"]["minimum_method_order"], 8)
        self.assertTrue(document["protocol_constants"]["comparison_prohibited_until_self_convergence_passed"])
        self.assertEqual(len(document["protocol_constants"]["checkpoint_schedule_refs"]), 5)
        self.assertEqual(
            document["protocol_constants"]["per_checkpoint_required_fields"],
            ["checkpoint_id", "checkpoint_epoch", "length_scale_lk", "velocity_scale_vk", "differential_signal_position_scale", "differential_signal_velocity_scale", "signal_floor", "component_treatment", "total_state_norm", "differential_signal_norm"],
        )
        self.assertEqual(
            document["checkpoint_requirements"],
            {
                "checkpoint_roster": {"status": "AWAITING_EXTERNAL_CHECKPOINT_ROSTER", "value": None},
                "per_checkpoint_metric_rows": {"status": "AWAITING_EXTERNAL_PER_CHECKPOINT_METRICS", "value": None},
            },
        )

    def test_q4_identity_conventions_and_fixture_roster_are_explicit(self):
        document = _json(Q4_RELATIVE)
        constants = document["protocol_constants"]
        self.assertEqual(constants["coordinate_gauge"], "HARMONIC")
        self.assertEqual(constants["retained_order_policy"], "NEWTONIAN_SUBSTITUTION_INSIDE_C_MINUS_2_TERMS")
        self.assertTrue(constants["fixed_total_mu"])
        self.assertTrue(constants["correction_only_comparison"])
        self.assertIn("a_rel(nu)", constants["finite_mass_relative_identity"])
        self.assertIn("a_restricted", constants["restricted_difference_identity"])
        self.assertEqual(
            [slot["role"] for slot in document["fixture_slots"]],
            ["GENERIC", "RADIAL", "TRANSVERSE", "ROTATED", "BODY_EXCHANGED"],
        )
        self.assertTrue(all(slot["binding"] is None for slot in document["fixture_slots"]))

    def test_every_awaiting_external_value_is_null_and_no_status_claims_pass(self):
        count = 0

        def inspect(value):
            nonlocal count
            if isinstance(value, dict):
                status = value.get("status")
                if isinstance(status, str) and status.startswith("AWAITING_EXTERNAL_"):
                    count += 1
                    if "value" in value:
                        self.assertIsNone(value["value"])
                    if "binding" in value:
                        self.assertIsNone(value["binding"])
                    self.assertNotIn("PASSED", status)
                    self.assertNotIn("ACCEPTED", status)
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        for relative in (Q2_RELATIVE, Q3_RELATIVE, Q4_RELATIVE):
            inspect(_json(relative))
        self.assertGreater(count, 50)

    def test_package_contains_no_runner_cli_trajectory_or_outcome(self):
        files = {path.name for path in (PROJECT_ROOT / PACKAGE_RELATIVE).iterdir() if path.is_file()}
        self.assertEqual(
            files,
            {"README.md", "registration_v1.json", "q2_observer_registration_v1.json", "q3_oracle_registration_v1.json", "q4_eih_registration_v1.json", "verify_execution_prerequisites_v1.py"},
        )
        source = (PROJECT_ROOT / VERIFIER_RELATIVE).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and "__name__" in ast.unparse(node.test) for node in ast.walk(tree)))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertEqual(
            {name for name in imports if name.startswith("jxplanetx")},
            set(),
        )
        self.assertFalse(
            {
                "jxplanetx.solar_1pn",
                "jxplanetx.v5_reference_dynamics",
                "jxplanetx.v5_implicit_midpoint",
            }
            & imports
        )

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"status":"A","status":"B"}\n', encoding="utf-8")
            with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
                VERIFIER._strict_json(path)

    def test_binary_float_and_nonfinite_json_are_rejected(self):
        for payload in ('{"value":1.5}\n', '{"value":NaN}\n', '{"value":Infinity}\n'):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
                        VERIFIER._strict_json(path)

    def test_absolute_parent_and_empty_segment_paths_are_rejected(self):
        for path in ("/tmp/x", "../x", "runs/../x", "runs//x"):
            with self.subTest(path=path):
                with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
                    VERIFIER._safe_file(PROJECT_ROOT, path, "adversarial path")

    def test_symlink_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "file.json").write_text("{}\n", encoding="utf-8")
            os.symlink(target, root / "link")
            with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
                VERIFIER._safe_file(root, "link/file.json", "symlink")

    def test_raw_artifact_hash_drift_is_rejected(self):
        with _copied_repository() as root:
            with (root / Q2_RELATIVE).open("a", encoding="utf-8") as stream:
                stream.write(" \n")
            self.assertRejected(root)

    def test_canonical_artifact_hash_drift_is_rejected(self):
        with _copied_repository() as root:
            _rewrite_bound_json(root, Q2_RELATIVE, lambda doc: doc.update({"unknown": None}), canonical=False)
            self.assertRejected(root)

    def test_non_null_q2_rule_is_rejected_even_when_rehashed(self):
        with _copied_repository() as root:
            def mutate(document):
                document["observer_rules"]["initial_event_exclusion_rule"]["value"] = "skip t=0"
            _rewrite_bound_json(root, Q2_RELATIVE, mutate)
            self.assertRejected(root)

    def test_fake_package_readiness_is_rejected(self):
        with _copied_repository() as root:
            registration = _json(REGISTRATION_RELATIVE, root)
            registration["ready"] = True
            (root / REGISTRATION_RELATIVE).write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
            self.assertRejected(root)

    def test_package_identity_nonclaim_claim_control_and_blocked_reasons_are_exact(self):
        mutations = {
            "$schema": lambda document: document.__setitem__("$schema", "other-schema.json"),
            "schema": lambda document: document.__setitem__("schema", "other/v1"),
            "package_id": lambda document: document.__setitem__(
                "prerequisite_package_id", "jx.v5.solar_1pn.other.v1"
            ),
            "nonclaim": lambda document: document.__setitem__("nonclaim", "weakened"),
            "permitted_claim": lambda document: document["claim_control"].__setitem__(
                "permitted_claim", "QUALIFIED"
            ),
            "prohibited_claims": lambda document: document["claim_control"][
                "prohibited_claims"
            ].pop(),
            "blocked_reasons": lambda document: document["blocked_reasons"].pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), _copied_repository() as root:
                registration = _json(REGISTRATION_RELATIVE, root)
                mutate(registration)
                (root / REGISTRATION_RELATIVE).write_text(
                    json.dumps(registration, indent=2) + "\n",
                    encoding="utf-8",
                )
                self.assertRejected(root)

    def test_subordinate_identity_and_nonclaim_mutations_are_rejected_when_rehashed(self):
        for relative in (Q2_RELATIVE, Q3_RELATIVE, Q4_RELATIVE):
            for field, value in (
                ("$schema", "../../schemas/wrong.schema.json"),
                ("schema", "wrong/v1"),
                ("registration_id", "jx.v5.solar_1pn.wrong.v1"),
                ("nonclaim", "weakened"),
            ):
                with self.subTest(relative=relative, field=field), _copied_repository() as root:
                    _rewrite_bound_json(
                        root,
                        relative,
                        lambda document, field=field, value=value: document.__setitem__(
                            field, value
                        ),
                    )
                    self.assertRejected(root)

    def test_each_weakened_rehashed_schema_is_rejected_by_frozen_schema_binding(self):
        def weaken_package(schema):
            schema["properties"]["nonclaim"] = {"type": "string"}

        def weaken_q2(schema):
            schema["$defs"]["observerRules"]["properties"][
                "initial_event_exclusion_rule"
            ] = {"$ref": "#/$defs/awaiting"}

        def weaken_subordinate(schema):
            schema["properties"]["nonclaim"] = {"type": "string"}

        mutations = (
            (SCHEMA_RELATIVES[0], weaken_package),
            (SCHEMA_RELATIVES[1], weaken_q2),
            (SCHEMA_RELATIVES[2], weaken_subordinate),
            (SCHEMA_RELATIVES[3], weaken_subordinate),
        )
        for relative, mutate in mutations:
            with self.subTest(relative=relative), _copied_repository() as root:
                _rewrite_bound_json(root, relative, mutate)
                self.assertRejected(root)

    def test_schema_validator_rejects_unknown_keywords_and_enforces_max_items(self):
        package_schema = _json(SCHEMA_RELATIVES[0])
        registration = _json(REGISTRATION_RELATIVE)

        unknown = copy.deepcopy(package_schema)
        unknown["unregisteredKeyword"] = True
        with self.assertRaisesRegex(
            VERIFIER.PrerequisiteVerificationError,
            "unsupported schema keywords",
        ):
            VERIFIER._validate_schema(registration, unknown, "unknown-keyword")

        too_many = copy.deepcopy(registration)
        extra = copy.deepcopy(too_many["artifacts"][0])
        extra["role"] = "extra_artifact"
        extra["schema"] = "extra/v1"
        extra["path"] = "extra/artifact.json"
        too_many["artifacts"].append(extra)
        with self.assertRaisesRegex(
            VERIFIER.PrerequisiteVerificationError,
            "maxItems 3",
        ):
            VERIFIER._validate_schema(too_many, package_schema, "max-items")

        with _copied_repository() as root:
            _rewrite_bound_json(
                root,
                SCHEMA_RELATIVES[0],
                lambda schema: schema.__setitem__("unregisteredKeyword", True),
            )
            self.assertRejected(root)

    def test_every_awaiting_slot_has_a_role_specific_schema_status(self):
        documents_and_schemas = (
            (Q2_RELATIVE, SCHEMA_RELATIVES[1]),
            (Q3_RELATIVE, SCHEMA_RELATIVES[2]),
            (Q4_RELATIVE, SCHEMA_RELATIVES[3]),
        )

        def slot_paths(value, path=()):
            paths = []
            if isinstance(value, dict):
                status = value.get("status")
                if (
                    isinstance(status, str)
                    and status.startswith("AWAITING_EXTERNAL_")
                ):
                    paths.append(path)
                for key, child in value.items():
                    paths.extend(slot_paths(child, (*path, key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    paths.extend(slot_paths(child, (*path, index)))
            return paths

        observed = 0
        for document_relative, schema_relative in documents_and_schemas:
            original = _json(document_relative)
            schema = _json(schema_relative)
            for path in slot_paths(original):
                observed += 1
                mutated = copy.deepcopy(original)
                slot = mutated
                for part in path:
                    slot = slot[part]
                replacement = (
                    "AWAITING_EXTERNAL_SOURCE"
                    if slot["status"] != "AWAITING_EXTERNAL_SOURCE"
                    else "AWAITING_EXTERNAL_RUNTIME"
                )
                slot["status"] = replacement
                with self.subTest(document=document_relative, path=path):
                    with self.assertRaises(VERIFIER.PrerequisiteVerificationError):
                        VERIFIER._validate_schema(mutated, schema, "role-specific status")
        self.assertEqual(observed, 62)

    def test_reordered_q3_force_ledger_is_rejected_even_when_rehashed(self):
        with _copied_repository() as root:
            def mutate(document):
                document["protocol_constants"]["force_ledger"].reverse()
            _rewrite_bound_json(root, Q3_RELATIVE, mutate)
            self.assertRejected(root)

    def test_filled_q3_scale_is_rejected_even_when_rehashed(self):
        with _copied_repository() as root:
            def mutate(document):
                document["checkpoint_requirements"]["per_checkpoint_metric_rows"]["value"] = [
                    {"checkpoint_epoch": "8", "length_scale_lk": "1"}
                ]
            _rewrite_bound_json(root, Q3_RELATIVE, mutate)
            self.assertRejected(root)

    def test_supplied_q4_fixture_is_rejected_even_when_rehashed(self):
        with _copied_repository() as root:
            def mutate(document):
                document["fixture_slots"][0]["binding"] = {"path": "external.json"}
            _rewrite_bound_json(root, Q4_RELATIVE, mutate)
            self.assertRejected(root)

    def test_unknown_q4_key_is_rejected_even_when_rehashed(self):
        with _copied_repository() as root:
            _rewrite_bound_json(root, Q4_RELATIVE, lambda doc: doc.update({"outcome": "PASSED"}))
            self.assertRejected(root)

    def test_predecessor_commit_drift_is_rejected(self):
        with _copied_repository() as root:
            registration = _json(REGISTRATION_RELATIVE, root)
            registration["predecessor"]["commit"] = "0" * 40
            (root / REGISTRATION_RELATIVE).write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
            self.assertRejected(root)

    def test_relocated_but_byte_identical_artifact_is_rejected(self):
        with _copied_repository() as root:
            alias = PACKAGE_RELATIVE / "q2_alias.json"
            shutil.copy2(root / Q2_RELATIVE, root / alias)
            registration = _json(REGISTRATION_RELATIVE, root)
            registration["artifacts"][0]["path"] = alias.as_posix()
            (root / REGISTRATION_RELATIVE).write_text(
                json.dumps(registration, indent=2) + "\n", encoding="utf-8"
            )
            self.assertRejected(root)


if __name__ == "__main__":
    unittest.main()
