from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from jxplanetx.cli import parser
from jxplanetx.force_registry_v5 import (
    ForceRegistryError,
    inspect_registry_file,
    load_executable_registry,
    validate_against_bundled_schema,
    validate_registry_semantics,
)
from jxplanetx.provenance import sha256_data, source_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "registries" / "jx_force_parameters_v5.json"
SCHEMA = PROJECT_ROOT / "schemas" / "jx-force-parameter-registry-v5.schema.json"


class ForceRegistryV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_checked_in_registry_is_valid_but_nonauthorizing(self) -> None:
        normalized, inspection = inspect_registry_file(REGISTRY, project_root=PROJECT_ROOT)
        self.assertEqual(normalized["state"], "DRAFT_NONEXECUTABLE")
        self.assertFalse(inspection.execution_authorized)
        self.assertTrue(inspection.unresolved_parameter_ids)
        self.assertTrue(inspection.blocked_force_model_ids)
        self.assertEqual(inspection.registry_sha256, sha256_data(normalized))

    def test_registry_declares_every_required_force_family(self) -> None:
        _, inspection = inspect_registry_file(REGISTRY, project_root=PROJECT_ROOT)
        self.assertTrue(
            {"NEWTONIAN_POINT_MASS", "RELATIVITY", "GRAVITY_HARMONICS", "NONGRAVITATIONAL"}
            <= set(inspection.force_families)
        )

    def test_draft_cannot_be_loaded_for_execution(self) -> None:
        with self.assertRaisesRegex(ForceRegistryError, "grants no execution authority") as raised:
            load_executable_registry(REGISTRY, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "registry_not_executable")

    def test_schema_rejects_unknown_fields(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["silent_default"] = True
        with self.assertRaises(ForceRegistryError) as raised:
            validate_against_bundled_schema(mutated, self.schema)
        self.assertEqual(raised.exception.code, "schema_unknown_field")

    def test_duplicate_model_identifiers_fail(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["force_models"].append(copy.deepcopy(mutated["force_models"][0]))
        with self.assertRaises(ForceRegistryError) as raised:
            validate_registry_semantics(mutated, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "duplicate_identifier")

    def test_missing_force_family_fails(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["force_models"] = [
            row for row in mutated["force_models"] if row["family"] != "RELATIVITY"
        ]
        mutated["applicability"] = [
            row
            for row in mutated["applicability"]
            if row["model_id"] in {model["model_id"] for model in mutated["force_models"]}
        ]
        with self.assertRaises(ForceRegistryError) as raised:
            validate_registry_semantics(mutated, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "missing_force_family")

    def test_unknown_parameter_reference_fails(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["force_models"][0]["parameter_refs"] = ["parameter.does-not-exist"]
        with self.assertRaises(ForceRegistryError) as raised:
            validate_registry_semantics(mutated, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "unknown_reference")

    def test_binary_float_scientific_values_fail(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["parameters"][0]["resolution"]["value"] = 1.0
        with self.assertRaises(ForceRegistryError) as raised:
            validate_registry_semantics(mutated, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "binary_float_forbidden")

    def test_scientific_contract_binding_is_live(self) -> None:
        mutated = copy.deepcopy(self.registry)
        mutated["framework"]["scientific_contract_sha256"] = "1" * 64
        with self.assertRaises(ForceRegistryError) as raised:
            validate_registry_semantics(mutated, project_root=PROJECT_ROOT)
        self.assertEqual(raised.exception.code, "scientific_contract_hash_mismatch")

    def test_duplicate_json_keys_are_rejected_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            duplicate = Path(folder) / "duplicate.json"
            duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
            with self.assertRaises(ForceRegistryError) as raised:
                inspect_registry_file(
                    duplicate,
                    schema_path=SCHEMA,
                    project_root=PROJECT_ROOT,
                )
        self.assertEqual(raised.exception.code, "duplicate_json_key")

    def test_schema_closes_every_object_record(self) -> None:
        self.assertFalse(self.schema["additionalProperties"])
        for name, definition in self.schema["$defs"].items():
            if definition.get("type") == "object":
                self.assertIs(
                    definition.get("additionalProperties"),
                    False,
                    f"schema object {name} is not fail-closed",
                )

    def test_v4_is_explicitly_immutable_and_not_consumed(self) -> None:
        lineage = self.registry["lineage"]
        self.assertTrue(lineage["v4_immutable"])
        self.assertFalse(lineage["v4_may_be_read"])
        self.assertFalse(lineage["v4_may_be_ingested"])
        self.assertFalse(lineage["v4_may_be_modified"])
        self.assertEqual(lineage["v4_artifact_references"], [])

    def test_cli_exposes_inspection_not_execution(self) -> None:
        command = parser().parse_args(
            [
                "inspect-force-registry",
                "--registry",
                str(REGISTRY),
                "--output",
                "inspection.json",
            ]
        )
        self.assertEqual(command.func.__name__, "inspect_force_registry_cli")
        self.assertFalse(hasattr(command, "execute"))

    def test_source_manifest_binds_registry_and_schema(self) -> None:
        manifest = source_manifest(PROJECT_ROOT)
        self.assertIn("registries/jx_force_parameters_v5.json", manifest["files"])
        self.assertIn("schemas/jx-force-parameter-registry-v5.schema.json", manifest["files"])


if __name__ == "__main__":
    unittest.main()
