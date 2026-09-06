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
from jxplanetx.solar_1pn import SOLAR_SCHWARZSCHILD_1PN_MODEL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "registries" / "jx_force_parameters_v5.json"
SCHEMA = PROJECT_ROOT / "schemas" / "jx-force-parameter-registry-v5.schema.json"

EXPECTED_MODEL_IDS = frozenset(
    {
        "force.newtonian.point_mass",
        "relativity.solar_schwarzschild_test_particle_1pn",
        "force.relativity.eih_1pn_gr",
        "force.relativity.restricted_ppn_beta_gamma",
        "force.relativity.solar_lense_thirring",
        "force.harmonics.solar_j2_j4",
        "force.harmonics.planetary",
        "force.tides.constant_time_lag",
        "force.nongrav.srp_cannonball",
        "force.nongrav.srp_pr_burns_1979",
        "force.nongrav.yarkovsky.empirical_a2",
        "force.nongrav.yarkovsky.linear_sphere",
        "force.nongrav.yarkovsky.facet_thermophysical",
        "force.nongrav.comet.marsden_esm",
        "force.nongrav.comet.rotating_jet",
        "force.encounter.hybrid_switching",
        "force.regularization.algorithmic",
    }
)


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

    def test_model_roster_and_solar_kernel_boundary_are_exact(self) -> None:
        models = {row["model_id"]: row for row in self.registry["force_models"]}
        self.assertEqual(frozenset(models), EXPECTED_MODEL_IDS)
        self.assertEqual(SOLAR_SCHWARZSCHILD_1PN_MODEL_ID, "relativity.solar_schwarzschild_test_particle_1pn")

        solar = models[SOLAR_SCHWARZSCHILD_1PN_MODEL_ID]
        self.assertEqual(solar["implementation_ref"], "src/jxplanetx/solar_1pn.py")
        self.assertEqual(solar["implementation_status"], "NOT_IMPLEMENTED")
        self.assertEqual(solar["qualification_status"], "UNQUALIFIED")
        self.assertEqual(solar["treatment"], "BLOCKED")
        self.assertEqual(solar["applicability_status"], "TBD_BLOCKED")
        blockers = " ".join(solar["blocking_reasons"])
        self.assertIn("equation-level Decimal kernel", blockers)
        self.assertIn("velocity-dependent production integrator", blockers)
        self.assertIn("precision and rounding", blockers)
        self.assertIn("isolated reference integrator", blockers)
        self.assertIn("per-reference-run binding grants no authority", blockers)
        permitted = self.registry["claim_boundaries"]["permitted_claims"]
        self.assertTrue(any("implicit-midpoint reference path" in claim for claim in permitted))
        self.assertTrue(any("no registry authority" in claim for claim in permitted))

    def test_discovery_links_do_not_masquerade_as_retained_provenance(self) -> None:
        for source in self.registry["provenance_sources"]:
            self.assertEqual(source["status"], "TBD_BLOCKED")
            self.assertEqual(source["sha256"], "TBD_BLOCKED")
            self.assertEqual(source["size_bytes"], "TBD_BLOCKED")
        self.assertFalse(self.registry["execution_policy"]["executable"])

    def test_solar_kernel_registers_domain_and_tdb_compatible_inputs(self) -> None:
        models = {row["model_id"]: row for row in self.registry["force_models"]}
        parameters = {row["parameter_id"]: row for row in self.registry["parameters"]}
        solar = models[SOLAR_SCHWARZSCHILD_1PN_MODEL_ID]
        required = {
            "parameter.sun.gm_tdb_compatible",
            "parameter.speed_of_light",
            "parameter.solar_1pn.maximum_compactness",
            "parameter.solar_1pn.maximum_speed_fraction_squared",
            "parameter.decimal_context",
            "parameter.omitted_force.error_budget",
        }
        self.assertEqual(set(solar["parameter_refs"]), required)

        compatible_gm = parameters["parameter.sun.gm_tdb_compatible"]
        self.assertEqual(compatible_gm["frame_ref"], "frame.sun_relative_icrs_tdb_restricted")
        self.assertIn("source.iau.tdb2006", compatible_gm["provenance_refs"])

        for parameter_id in (
            "parameter.solar_1pn.maximum_compactness",
            "parameter.solar_1pn.maximum_speed_fraction_squared",
        ):
            threshold = parameters[parameter_id]
            self.assertEqual(threshold["unit_id"], "unit.dimensionless")
            self.assertEqual(threshold["resolution"]["state"], "TBD_BLOCKED")
            self.assertTrue(threshold["required_for_execution"])

    def test_every_declared_model_has_discovery_provenance_but_no_verified_source(self) -> None:
        self.assertTrue(all(model["provenance_refs"] for model in self.registry["force_models"]))
        self.assertFalse(any(source["status"] == "VERIFIED" for source in self.registry["provenance_sources"]))

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

    def test_source_manifest_binds_registry_schema_kernel_and_contracts(self) -> None:
        manifest = source_manifest(PROJECT_ROOT)
        self.assertIn("registries/jx_force_parameters_v5.json", manifest["files"])
        self.assertIn("schemas/jx-force-parameter-registry-v5.schema.json", manifest["files"])
        self.assertIn("src/jxplanetx/force_registry_v5.py", manifest["files"])
        self.assertIn("src/jxplanetx/solar_1pn.py", manifest["files"])
        self.assertIn("src/jxplanetx/v5_reference_dynamics.py", manifest["files"])
        self.assertIn("src/jxplanetx/v5_implicit_midpoint.py", manifest["files"])
        self.assertIn("src/jxplanetx/v5_solar_1pn_qualification.py", manifest["files"])
        self.assertIn("tests/test_v5_reference_integrator.py", manifest["files"])
        self.assertIn("tests/test_v5_solar_1pn_qualification.py", manifest["files"])
        self.assertIn("schemas/jx-v5-solar-1pn-qualification-plan-v1.schema.json", manifest["files"])
        self.assertIn("schemas/jx-v5-solar-1pn-qualification-inputs-v1.schema.json", manifest["files"])
        self.assertIn("schemas/jx-v5-solar-1pn-qualification-registration-v1.schema.json", manifest["files"])
        self.assertIn("schemas/jx-v5-solar-1pn-prior-development-cases-v1.schema.json", manifest["files"])
        self.assertIn("docs/SCIENTIFIC_CONTRACT.md", manifest["files"])
        self.assertIn("docs/FORCE_PARAMETER_REGISTRY_V5.md", manifest["files"])
        self.assertIn("docs/V5_REFERENCE_INTEGRATOR.md", manifest["files"])


if __name__ == "__main__":
    unittest.main()
