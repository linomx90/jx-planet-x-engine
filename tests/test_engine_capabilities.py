import unittest
import re
from pathlib import Path
from types import MappingProxyType

from jxplanetx.engine import (
    ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
    CAPABILITY_CATALOG,
    COUPLED_LUNAR_RKF78_METHOD_ID,
    HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
    LUNAR_EPHEMERIS_V1_METHOD_ID,
    AccelerationSemantics,
    CapabilityUnavailableError,
    CodeStatus,
    ContractError,
    DeclaredModelConfig,
    Maturity,
    ParameterBinding,
    ParameterMetadata,
    Provenance,
    get_capability,
    list_capabilities,
)


EXPECTED_IMPLEMENTED = {
    "force.newtonian.point_mass",
    "relativity.solar_schwarzschild_test_particle_1pn",
    "force.relativity.eih_1pn_gr",
    "force.nongrav.srp_cannonball",
    "solar-system.force.earth-zonal-j2-j5-axisymmetric-pair",
    "solar-system.force.lunar-static-degree2-principal-axis-pair",
    "solar-system.force.lunar-static-degree3-principal-axis-pair",
    "backend.numpy.cpu",
    "backend.cupy.cuda",
    "precision.float64",
    "determinism.same_runtime_device",
    "integrator.adaptive.rkf78.fehlberg_1968",
    COUPLED_LUNAR_RKF78_METHOD_ID,
    LUNAR_EPHEMERIS_V1_METHOD_ID,
    "integrator.adaptive.rkf78.encounter_segment_newtonian_v1",
    "integrator.symplectic.kdk_leapfrog_2",
    "integrator.symplectic.wisdom_holman_jacobi_kdk_2",
    "integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1",
}

EXPECTED_FAMILIES = {
    "BACKEND",
    "CLOSE_ENCOUNTER",
    "COLLISION",
    "DETERMINISM",
    "DRAG",
    "EPHEMERIS",
    "GRAVITY",
    "GRAVITY_HARMONICS",
    "INTEGRATOR",
    "MEASUREMENT",
    "NONGRAVITATIONAL",
    "ORBIT_DETERMINATION",
    "OUTGASSING",
    "PRECISION",
    "REGULARIZATION",
    "RELATIVITY",
    "SENSITIVITY",
    "THRUST",
    "TIDES",
}


def provenance() -> Provenance:
    return Provenance("fixture.catalog", "Synthetic catalog fixture", "1", "b" * 64)


def metadata(
    parameter_id: str,
    units: str,
    *,
    validity_start: float = -1.0,
    validity_end: float = 1.0,
) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id,
        units,
        provenance(),
        None,
        None,
        validity_start,
        validity_end,
    )


def sample_value(value_kind: str) -> object:
    if value_kind == "scalar":
        return 0.0
    if value_kind == "positive_scalar":
        return 1.0
    if value_kind == "nonnegative_scalar":
        return 0.0
    if value_kind == "integer":
        return 1
    if value_kind == "boolean":
        return False
    if value_kind in {"string", "identifier"}:
        return "FIXTURE"
    if value_kind == "identifier_tuple":
        return ("FIXTURE_BODY",)
    if value_kind == "vector3":
        return (1.0, 0.0, 0.0)
    if value_kind in {"array", "record"}:
        return object()
    if value_kind == "callable":
        return lambda *args: None
    raise AssertionError(f"test fixture does not recognize value_kind={value_kind!r}")


def bindings_for(model_id: str) -> tuple[ParameterBinding, ...]:
    capability = get_capability(model_id)
    return tuple(
        ParameterBinding(
            parameter.parameter_id,
            sample_value(parameter.value_kind),
            metadata(parameter.parameter_id, parameter.unit_dimension),
        )
        for parameter in capability.parameters
        if parameter.required
    )


def declared_config(model_id: str, **changes: object) -> DeclaredModelConfig:
    values: dict[str, object] = {
        "config_id": f"fixture.{model_id}",
        "model_id": model_id,
        "epoch": 0.0,
        "unit_system_id": "fixture.units",
        "parameters": bindings_for(model_id),
    }
    values.update(changes)
    return DeclaredModelConfig(**values)  # type: ignore[arg-type]


class CatalogClosureTests(unittest.TestCase):
    def test_public_docs_bind_exact_catalog_order_counts_and_hybrid_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        engine_api_text = (root / "docs" / "ENGINE_API.md").read_text(
            encoding="utf-8"
        )
        catalog_section = engine_api_text.split("## Capability catalog", 1)[1]
        documented_rows = tuple(
            match.group(2)
            for match in re.finditer(
                r"^\| (Implemented|Declared) \| `([^`]+)` \|",
                catalog_section,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(
            documented_rows,
            tuple(row.model_id for row in list_capabilities()),
        )
        normalized_engine_api = " ".join(engine_api_text.split())
        self.assertIn(
            "stable 51-row catalog spanning 19 families", normalized_engine_api
        )
        self.assertIn(
            "Exactly eighteen rows are marked `IMPLEMENTED`", normalized_engine_api
        )
        for required in (
            HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
            "discards every candidate array and all provisional state",
            "exact Newtonian local IVP issuing from that node",
            "There is exactly one outer replay and no nested public Wisdom--Holman or encounter replay",
            "65,536 outer records",
            "not globally symplectic",
            "external process isolation, timeout, memory, and concurrency limits",
        ):
            self.assertIn(required, normalized_engine_api)
        self.assertIn(
            "| Declared | `force.encounter.hybrid_switching` |",
            engine_api_text,
        )
        self.assertIn(
            "| Declared | `integrator.hybrid.close_encounter` |",
            engine_api_text,
        )

        expected_release_phrases = {
            "README.md": (
                "development catalog contains exactly eighteen `IMPLEMENTED`",
                HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID.split(".v1", 1)[0],
            ),
            "CHANGELOG.md": (
                "46-capability catalog with exactly twelve `IMPLEMENTED`",
                "Specific whole-system Wisdom--Holman/RKF78 hybrid integration",
            ),
            "RELEASE_NOTES.md": (
                "ordered 46-capability catalog",
                HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
            ),
        }
        for relative_path, phrases in expected_release_phrases.items():
            text = (root / relative_path).read_text(encoding="utf-8")
            for phrase in phrases:
                with self.subTest(path=relative_path, phrase=phrase):
                    self.assertIn(phrase, text)

    def test_catalog_has_exact_stable_scope_and_no_duplicate_identifiers(self):
        rows = list_capabilities()
        self.assertEqual(len(rows), 51)
        self.assertEqual(len(CAPABILITY_CATALOG), 51)
        self.assertIsInstance(CAPABILITY_CATALOG, MappingProxyType)
        self.assertEqual(tuple(CAPABILITY_CATALOG), tuple(row.model_id for row in rows))
        self.assertEqual(len({row.model_id for row in rows}), len(rows))
        self.assertEqual({row.family for row in rows}, EXPECTED_FAMILIES)
        with self.assertRaises(TypeError):
            CAPABILITY_CATALOG["invented"] = rows[0]  # type: ignore[index]

    def test_only_exact_alpha_code_paths_are_marked_implemented(self):
        rows = list_capabilities()
        implemented = {
            row.model_id for row in rows if row.code_status is CodeStatus.IMPLEMENTED
        }
        self.assertEqual(implemented, EXPECTED_IMPLEMENTED)
        self.assertEqual(sum(row.code_status is CodeStatus.DECLARED for row in rows), 33)
        self.assertTrue(all(row.maturity is Maturity.UNQUALIFIED for row in rows))
        self.assertTrue(
            all(row.execution_available == (row.model_id in EXPECTED_IMPLEMENTED) for row in rows)
        )

    def test_dependency_and_exclusion_graph_is_closed_and_symmetric(self):
        ids = set(CAPABILITY_CATALOG)
        for row in list_capabilities():
            with self.subTest(model_id=row.model_id):
                self.assertNotIn(row.model_id, row.dependencies)
                self.assertTrue(set(row.dependencies).issubset(ids))
                self.assertTrue(set(row.mutually_exclusive_with).issubset(ids))
                self.assertEqual(
                    len({parameter.parameter_id for parameter in row.parameters}),
                    len(row.parameters),
                )
                for excluded in row.mutually_exclusive_with:
                    self.assertIn(
                        row.model_id,
                        get_capability(excluded).mutually_exclusive_with,
                    )

    def test_force_integrator_compatibility_is_explicit_and_conservative(self):
        newtonian = get_capability("force.newtonian.point_mass")
        one_pn = get_capability("relativity.solar_schwarzschild_test_particle_1pn")
        srp = get_capability("force.nongrav.srp_cannonball")
        self.assertIs(newtonian.semantics, AccelerationSemantics.BASE)
        self.assertFalse(newtonian.velocity_dependent)
        self.assertIn("POSITION_FORCE_KICK_DRIFT", newtonian.compatible_integrators)
        self.assertIs(one_pn.semantics, AccelerationSemantics.CORRECTION)
        self.assertTrue(one_pn.velocity_dependent)
        self.assertIn("GENERAL_FIRST_ORDER", one_pn.compatible_integrators)
        self.assertIn("IMPLICIT_VELOCITY_DEPENDENT", one_pn.compatible_integrators)
        self.assertNotIn("POSITION_FORCE_KICK_DRIFT", one_pn.compatible_integrators)
        self.assertFalse(srp.velocity_dependent)
        self.assertIn("GENERAL_FIRST_ORDER", srp.compatible_integrators)
        self.assertEqual(srp.dependencies, ("force.newtonian.point_mass",))

        integrators = tuple(row for row in list_capabilities() if row.family == "INTEGRATOR")
        self.assertEqual(len(integrators), 10)
        implemented = tuple(
            row.model_id
            for row in integrators
            if row.code_status is CodeStatus.IMPLEMENTED
        )
        self.assertEqual(
            implemented,
            (
                "integrator.adaptive.rkf78.fehlberg_1968",
                COUPLED_LUNAR_RKF78_METHOD_ID,
                LUNAR_EPHEMERIS_V1_METHOD_ID,
                ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
                "integrator.symplectic.kdk_leapfrog_2",
                "integrator.symplectic.wisdom_holman_jacobi_kdk_2",
                HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
            ),
        )
        self.assertTrue(get_capability(implemented[0]).execution_available)
        self.assertTrue(
            all(
                not row.execution_available
                for row in integrators
                if row.code_status is CodeStatus.DECLARED
            )
        )
        adaptive = get_capability("integrator.adaptive.rkf78.fehlberg_1968")
        self.assertIs(adaptive.semantics, AccelerationSemantics.SERVICE)
        self.assertTrue(adaptive.velocity_dependent)
        self.assertTrue(adaptive.global_snapshot_required)
        self.assertIn("GENERAL_FIRST_ORDER", adaptive.compatible_integrators)
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in adaptive.parameters),
            (
                "checkpoint_epochs",
                "initial_step",
                "minimum_step",
                "maximum_step",
                "position_atol",
                "position_rtol",
                "velocity_atol",
                "velocity_rtol",
                "maximum_steps",
                "maximum_rejections",
                "safety_factor",
                "minimum_scale_factor",
                "maximum_scale_factor",
            ),
        )
        coupled = get_capability(COUPLED_LUNAR_RKF78_METHOD_ID)
        self.assertIs(coupled.semantics, AccelerationSemantics.SERVICE)
        self.assertTrue(coupled.velocity_dependent)
        self.assertTrue(coupled.global_snapshot_required)
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in coupled.parameters),
            (
                "initial_state",
                "parameters",
                "integration_spec",
                "prehistory_provider",
            ),
        )
        self.assertEqual(
            coupled.dependencies,
            ("backend.numpy.cpu", "precision.float64"),
        )
        self.assertTrue(
            any("not yet dispatched through force ABI v1" in item for item in coupled.restrictions)
        )
        lunar_ephemeris = get_capability(LUNAR_EPHEMERIS_V1_METHOD_ID)
        self.assertTrue(lunar_ephemeris.velocity_dependent)
        self.assertTrue(lunar_ephemeris.global_snapshot_required)
        self.assertEqual(
            tuple(
                parameter.parameter_id
                for parameter in lunar_ephemeris.parameters
            ),
            ("initial_state", "parameters", "integration_spec"),
        )
        self.assertEqual(
            lunar_ephemeris.dependencies,
            (
                "backend.numpy.cpu",
                "precision.float64",
                "force.relativity.eih_1pn_gr",
            ),
        )
        self.assertTrue(
            any(
                "not a production ephemeris" in item
                for item in lunar_ephemeris.restrictions
            )
        )
        encounter = get_capability(ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID)
        self.assertIs(encounter.semantics, AccelerationSemantics.SERVICE)
        self.assertFalse(encounter.velocity_dependent)
        self.assertTrue(encounter.global_snapshot_required)
        self.assertIn("GENERAL_FIRST_ORDER", encounter.compatible_integrators)
        self.assertEqual(
            encounter.dependencies,
            (
                "force.newtonian.point_mass",
                "backend.numpy.cpu",
                "precision.float64",
            ),
        )
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in encounter.parameters),
            (
                "body_order",
                "initial_epoch",
                "endpoint_epoch",
                "duration",
                "initial_step_magnitude",
                "minimum_step_magnitude",
                "maximum_step_magnitude",
                "pair_certification_floors",
                "pair_position_atols",
                "pair_position_rtol",
                "pair_velocity_atols",
                "pair_velocity_rtol",
                "gm_centroid_position_atol",
                "gm_centroid_position_rtol",
                "gm_centroid_velocity_atol",
                "gm_centroid_velocity_rtol",
                "maximum_substep_proposals",
                "maximum_accepted_substeps",
                "maximum_rejected_substeps",
                "maximum_consecutive_rejections",
                "maximum_force_evaluations",
                "safety_factor",
                "minimum_scale_factor",
                "maximum_scale_factor",
                "exact_rational_resources",
            ),
        )
        for required_restriction in (
            "local IVP issuing from each accepted numerical node",
            "independent externally supplied provenance label",
            "mandatory deterministic semantic replay",
            "never a collision or event determination",
            "no dense output, event detection or location, collision response",
            "unqualified MODEL_OUTPUT",
        ):
            self.assertTrue(
                any(required_restriction in item for item in encounter.restrictions),
                required_restriction,
            )
        kdk = get_capability("integrator.symplectic.kdk_leapfrog_2")
        self.assertIs(kdk.semantics, AccelerationSemantics.SERVICE)
        self.assertFalse(kdk.velocity_dependent)
        self.assertTrue(kdk.global_snapshot_required)
        self.assertIn("POSITION_FORCE_KICK_DRIFT", kdk.compatible_integrators)
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in kdk.parameters),
            (
                "checkpoint_step_indices",
                "fixed_step",
                "maximum_steps",
                "minimum_swept_pair_separation",
                "maximum_pair_frequency_step",
            ),
        )
        self.assertTrue(
            any("passive/massless tracers unavailable" in item for item in kdk.restrictions)
        )
        wh = get_capability(
            "integrator.symplectic.wisdom_holman_jacobi_kdk_2"
        )
        self.assertIs(wh.semantics, AccelerationSemantics.SERVICE)
        self.assertFalse(wh.velocity_dependent)
        self.assertTrue(wh.global_snapshot_required)
        self.assertIn("POSITION_FORCE_KICK_DRIFT", wh.compatible_integrators)
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in wh.parameters),
            (
                "checkpoint_step_indices",
                "fixed_step",
                "maximum_steps",
                "jacobi_body_order",
                "minimum_encounter_pair_separation",
                "minimum_jacobi_periapse",
                "maximum_initial_barycenter_position_norm",
                "maximum_initial_barycenter_velocity_norm",
                "kepler_solver",
            ),
        )
        self.assertFalse(wh.parameters[-1].required)
        self.assertTrue(
            any("central body first" in item for item in wh.restrictions)
        )
        self.assertTrue(
            any("mandatory deterministic semantic validation replay" in item for item in wh.restrictions)
        )
        hybrid = get_capability(HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID)
        self.assertIs(hybrid.semantics, AccelerationSemantics.SERVICE)
        self.assertFalse(hybrid.velocity_dependent)
        self.assertTrue(hybrid.global_snapshot_required)
        self.assertIn("POSITION_FORCE_KICK_DRIFT", hybrid.compatible_integrators)
        self.assertEqual(
            tuple(parameter.parameter_id for parameter in hybrid.parameters),
            ("wisdom_holman_spec", "encounter_control"),
        )
        self.assertEqual(
            hybrid.dependencies,
            (
                "force.newtonian.point_mass",
                "backend.numpy.cpu",
                "precision.float64",
            ),
        )
        for required_restriction in (
            "discards the entire provisional candidate",
            "exact local IVP issuing from each accepted numerical node",
            "no public child wrapper or nested public child replay",
            "65536 outer records, 4096 near macrosteps",
            "all-far execution preserves the frozen public WH",
            "static, numerical, resource, custody, and replay failures are fatal",
            "not globally symplectic",
            "external process isolation, timeout, memory, and concurrency limits",
            "unqualified MODEL_OUTPUT",
        ):
            self.assertTrue(
                any(required_restriction in item for item in hybrid.restrictions),
                required_restriction,
            )
        self.assertIs(
            get_capability("integrator.symplectic.split").code_status,
            CodeStatus.DECLARED,
        )
        self.assertIs(
            get_capability("force.encounter.hybrid_switching").code_status,
            CodeStatus.DECLARED,
        )
        self.assertIs(
            get_capability("integrator.hybrid.close_encounter").code_status,
            CodeStatus.DECLARED,
        )

    def test_unknown_or_ambiguous_model_identifiers_fail_closed(self):
        for model_id in ("", " invented", "invented", None):
            with self.subTest(model_id=model_id), self.assertRaises(ContractError):
                get_capability(model_id)  # type: ignore[arg-type]


class DeclaredCapabilityTests(unittest.TestCase):
    def test_every_declared_capability_validates_then_refuses_execution(self):
        declared = tuple(
            row for row in list_capabilities() if row.code_status is CodeStatus.DECLARED
        )
        self.assertEqual(len(declared), 33)
        for row in declared:
            with self.subTest(model_id=row.model_id):
                config = declared_config(row.model_id)
                self.assertIs(config.code_status, CodeStatus.DECLARED)
                self.assertIs(config.maturity, Maturity.UNQUALIFIED)
                with self.assertRaisesRegex(
                    CapabilityUnavailableError,
                    "declared but has no executable JX implementation",
                ):
                    config.require_executable()

    def test_declared_wrapper_cannot_be_used_for_implemented_models(self):
        for model_id in EXPECTED_IMPLEMENTED:
            with self.subTest(model_id=model_id), self.assertRaises(ContractError):
                DeclaredModelConfig(
                    f"fixture.{model_id}",
                    model_id,
                    0.0,
                    "fixture.units",
                    (),
                )

    def test_missing_unknown_duplicate_units_and_validity_fail_closed(self):
        model_id = "force.newtonian.softened_point_mass"
        valid = bindings_for(model_id)
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=valid[:-1])

        unknown = ParameterBinding(
            "unknown_parameter",
            1.0,
            metadata("unknown_parameter", "1"),
        )
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=valid + (unknown,))
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=valid + (valid[0],))

        wrong_units = ParameterBinding(
            valid[0].parameter_id,
            valid[0].value,
            metadata(valid[0].parameter_id, "WRONG_UNITS"),
        )
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=(wrong_units,) + valid[1:])

        expired = ParameterBinding(
            valid[0].parameter_id,
            valid[0].value,
            metadata(valid[0].parameter_id, valid[0].metadata.units, validity_start=-2.0, validity_end=-1.0),
        )
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=(expired,) + valid[1:])

    def test_binding_identifiers_and_known_value_kinds_are_exact(self):
        with self.assertRaises(ContractError):
            ParameterBinding("one", 1.0, metadata("two", "1"))

        model_id = "force.newtonian.softened_point_mass"
        valid = bindings_for(model_id)
        source_binding = valid[0]
        invalid_source = ParameterBinding(
            source_binding.parameter_id,
            ["BODY"],
            source_binding.metadata,
        )
        with self.assertRaises(ContractError):
            declared_config(model_id, parameters=(invalid_source,) + valid[1:])


if __name__ == "__main__":
    unittest.main()
