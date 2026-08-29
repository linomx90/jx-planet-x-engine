from __future__ import annotations

import ast
import copy
import difflib
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path
from unittest import mock

import jxplanetx.v5_solar_1pn_qualification as qualification_module
from jxplanetx.force_registry_v5 import inspect_registry_file
from jxplanetx.provenance import sha256_data, sha256_file
from jxplanetx.v5_solar_1pn_qualification import (
    ALLOWED_VERDICTS,
    ARM_GATE_MATRIX,
    ARM_SPECS,
    BLOCKED_REASONS,
    BLOCKER_SPECS,
    CASE_ROLES,
    DERIVATION_IDS,
    EXECUTION_IMPLEMENTATION_STATE,
    FOUNDATION_COMMIT as MODULE_FOUNDATION_COMMIT,
    INPUT_SCHEMA,
    INPUT_STATE,
    KNOWN_FILE_ROLE_PATHS,
    KNOWN_STATE_RECORD_SPECS,
    LOCKED_FILE_ROLE_PATHS,
    NONCLAIM,
    OBSERVABLE_IDS,
    PASS_EFFECT,
    PERMITTED_CLAIM_CODE,
    PERMITTED_CLAIM_TEXT,
    PLAN_SCHEMA,
    PLAN_STATE,
    PROHIBITED_CLAIM_CODES,
    PROVENANCE_SOURCE_PATHS,
    PRIOR_DEVELOPMENT_COUNTS,
    PRIOR_DEVELOPMENT_DISCLOSURE_SHA256,
    PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256,
    PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256,
    PRIOR_DEVELOPMENT_MANIFEST_ID,
    PRIOR_DEVELOPMENT_MANIFEST_PATH,
    PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES,
    PRIOR_DEVELOPMENT_MANIFEST_STATE,
    PRIOR_DEVELOPMENT_NONCLAIM,
    PRIOR_DEVELOPMENT_SCHEMA as MODULE_PRIOR_DEVELOPMENT_SCHEMA,
    PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256,
    PRIOR_FINGERPRINT_LIST_FILE_SHA256,
    PRIOR_FINGERPRINT_LIST_PATH,
    PRIOR_FINGERPRINT_LIST_SIZE_BYTES,
    PRIOR_FIRST_MEMBER_FINGERPRINT_SHA256,
    PRIOR_FOUNDATION_SOURCE_PATHS,
    PRIOR_LAST_MEMBER_FINGERPRINT_SHA256,
    PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256,
    QUALIFICATION_ID_PREFIX,
    GATE_SPECS,
    REGISTRATION_SCHEMA,
    REGISTRATION_STATE,
    SOURCE_LITERAL_SPECS,
    Solar1PNQualificationError,
    inspect_solar_1pn_qualification_package,
    qualification_fixture_sha256,
    qualification_plan_identity_sha256,
    qualification_scientific_fingerprint_sha256,
    validate_prior_development_manifest_semantics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORY = PROJECT_ROOT / "runs" / "v5_solar_1pn_qualification"
REGISTRATION_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification/registration_v1.json"
)
INPUTS_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification/qualification_inputs_v1.json"
)
PLAN_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification/qualification_plan_v1.json"
)
PRIOR_MANIFEST_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification/prior_development_cases_v1.json"
)
PRIOR_DEVELOPMENT_SCHEMA = "jx-v5-solar-1pn-prior-development-cases/v1"
FOUNDATION_COMMIT = "58b57094d519205f3ef398f3c738af96107c59e5"
EXPECTED_MANIFEST_RAW_SHA256 = (
    "c7897e82143cf155933910103d3dd9cca3166b2033a36bbd0d72c3109fe3c8ce"
)
EXPECTED_MANIFEST_CANONICAL_SHA256 = (
    "3444af9afbc39556148a7757e4e5cbf1a99d7be14335b126d6378b8ecae25140"
)
EXPECTED_MANIFEST_DISCLOSURE_SHA256 = (
    "7797cebb12c85ccc8c950a39dc1ed2e40cb3d59468a4aed9f6676489c55c2975"
)
EXPECTED_INPUTS_RAW_SHA256 = (
    "30ae263fc402d2d0d0bf6318c77285be13f8c2a5daa9c8e5667c789a20ab1b7c"
)
EXPECTED_INPUTS_CANONICAL_SHA256 = (
    "57b64944d65d930d4dfcf0a60fd02e5945788a14483c3daa30040257433c4f79"
)
EXPECTED_INPUTS_SIZE_BYTES = "186336"
EXPECTED_ORDERED_FAMILY_FINGERPRINTS_SHA256 = (
    "afebc490ed48337727210b857b4a1491c6647ede071439ec99071421c0a52dd1"
)
EXPECTED_FAMILY_LIST_RAW_SHA256 = (
    "20d23a93c426baa3818c05f5a57616edd7844905ef805d4ad87fb7948f3cd205"
)
EXPECTED_FAMILY_LIST_CANONICAL_SHA256 = (
    "c4f1c03ff7aa9a1127caa2c8924ffb638cc84229191961449a2239378a45c1d2"
)
EXPECTED_FAMILY_LIST_SIZE_BYTES = "1762116"
EXPECTED_FIRST_FAMILY_FINGERPRINT_SHA256 = (
    "09a1ab95feeb6fe4750202e351a8ed8157730c632a1d252e25a2dabb71cce88d"
)
EXPECTED_LAST_FAMILY_FINGERPRINT_SHA256 = (
    "d37787b6b6c2ea49e08375aa395a3be1abe0867939366cb7ba2468092afce6c7"
)
EXPECTED_PRIOR_SCHEMA_RAW_SHA256 = (
    "e4f8cf8e752c63dd74b470ecd273c62a015695d3e5b612356984d1e891d25d0c"
)
EXPECTED_MANIFEST_COUNTS = {
    "foundation_source_count": 2,
    "method_count": 43,
    "known_fixture_id_count": 79,
    "known_state_record_count": 26,
    "generated_state_family_count": 1,
    "generated_state_member_count": 4096,
    "invalid_case_record_count": 46,
    "generic_rhs_record_count": 6,
    "known_case_descriptor_count": 96,
}

EXPECTED_CONTEXT_IDS = frozenset(
    {
        "decimal.context.precision_50",
        "decimal.context.precision_70",
        "decimal.context.precision_90",
    }
)
EXPECTED_FIXTURE_IDS = frozenset(
    {
        "fixture.holdout.synthetic_perihelion_3d",
        "fixture.holdout.synthetic_perihelion_scaled_3d",
        "fixture.holdout.synthetic_perihelion_e047_3d",
        "fixture.holdout.synthetic_perihelion_e047_scaled_3d",
        "fixture.holdout.generic_equation_3d",
        "fixture.holdout.domain_boundary_3d",
        "fixture.holdout.domain_outside_3d",
        "fixture.holdout.domain_speed_outside_3d",
        "fixture.holdout.tdb_km_s_transform",
        "fixture.holdout.tdb_au_day_transform",
        "fixture.holdout.tcb_km_s_transform",
        "fixture.holdout.eih_mass_ratio",
    }
)
EXPECTED_FIXTURE_CASE_ROLES = {
    "fixture.holdout.synthetic_perihelion_3d": frozenset(
        {"ECCENTRIC", "LONG_ARC", "SIGNAL_SCALE", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.synthetic_perihelion_scaled_3d": frozenset(
        {"ECCENTRIC", "LONG_ARC", "SIGNAL_SCALE", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.synthetic_perihelion_e047_3d": frozenset(
        {"ECCENTRIC", "LONG_ARC", "SIGNAL_SCALE", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.synthetic_perihelion_e047_scaled_3d": frozenset(
        {"ECCENTRIC", "LONG_ARC", "SIGNAL_SCALE", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.generic_equation_3d": frozenset(
        {"GENERIC_3D", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.domain_boundary_3d": frozenset({"DOMAIN_BOUNDARY"}),
    "fixture.holdout.domain_outside_3d": frozenset({"DOMAIN_OUTSIDE"}),
    "fixture.holdout.domain_speed_outside_3d": frozenset({"DOMAIN_OUTSIDE"}),
    "fixture.holdout.tdb_km_s_transform": frozenset(
        {"TRANSFORM_TWIN", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.tdb_au_day_transform": frozenset(
        {"TRANSFORM_TWIN", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.tcb_km_s_transform": frozenset(
        {"TRANSFORM_TWIN", "DOMAIN_INSIDE"}
    ),
    "fixture.holdout.eih_mass_ratio": frozenset(
        {"EIH_LIMIT", "DOMAIN_INSIDE"}
    ),
}
EXPECTED_ARM_IDS = frozenset(
    {
        "arm.q0_integrity",
        "arm.q1_equation",
        "arm.q2_perihelion",
        "arm.q3_high_order_oracle",
        "arm.q4_restricted_eih",
        "arm.q5_step_solver",
        "arm.q6_precision",
        "arm.q7_transform",
        "arm.q8_holdout",
        "arm.q9_claim_audit",
    }
)
EXPECTED_GATE_IDS = frozenset(
    {
        "gate.q0_package_integrity",
        "gate.q1_equation_discrimination",
        "gate.q2_perihelion_coefficient",
        "gate.q3_high_order_oracle",
        "gate.q4_restricted_eih_limit",
        "gate.q5_step_solver_convergence",
        "gate.q6_decimal_precision",
        "gate.q7_unit_time_transform",
        "gate.q8_first_unblinding",
        "gate.q9_claim_audit",
    }
)
EXPECTED_ARM_GATE_MATRIX = (
    ("arm.q0_integrity", "gate.q0_package_integrity"),
    ("arm.q1_equation", "gate.q1_equation_discrimination"),
    ("arm.q2_perihelion", "gate.q2_perihelion_coefficient"),
    ("arm.q3_high_order_oracle", "gate.q3_high_order_oracle"),
    ("arm.q4_restricted_eih", "gate.q4_restricted_eih_limit"),
    ("arm.q5_step_solver", "gate.q5_step_solver_convergence"),
    ("arm.q6_precision", "gate.q6_decimal_precision"),
    ("arm.q7_transform", "gate.q7_unit_time_transform"),
    ("arm.q8_holdout", "gate.q8_first_unblinding"),
    ("arm.q9_claim_audit", "gate.q9_claim_audit"),
)
EXPECTED_SOURCE_IDS = frozenset(
    {
        "source.bipm.si_brochure_9_v4_01",
        "source.iau.resolution_b2_2012",
        "source.iau.resolution_b2_2006",
        "source.iau.resolution_b3_2006",
        "source.iers.tn36.chapter10",
        "source.jpl.astrodynamic_parameters.de440",
        "source.jpl.de440.export_info",
        "source.naif.gm_de440",
    }
)
EXPECTED_SOURCE_LITERAL_SPECS = {
    "derivation.source.c_km_s": {
        "result_name": "c_km_s",
        "value": "299792.458",
        "unit_id": "unit.kilometre_per_second",
        "source_id": "source.bipm.si_brochure_9_v4_01",
        "source_locator": (
            "SI base units, exact speed of light, second and kilometre definitions"
        ),
    },
    "derivation.source.gm_de440_km3_s2": {
        "result_name": "gm_km3_s2",
        "value": "132712440041.279419",
        "unit_id": "unit.kilometre3_per_second2",
        "source_id": "source.jpl.astrodynamic_parameters.de440",
        "source_locator": (
            "heliocentric gravitational constant GM_sun row: "
            "1.32712440041279419 x 10^20 m^3 s^-2"
        ),
    },
    "derivation.source.lb": {
        "result_name": "l_b",
        "value": "0.00000001550519768",
        "unit_id": "unit.dimensionless",
        "source_id": "source.iau.resolution_b3_2006",
        "source_locator": "TDB definition and L_B-compatible scaling",
    },
    "derivation.source.au_km": {
        "result_name": "km_per_au",
        "value": "149597870.7",
        "unit_id": "unit.dimensionless",
        "source_id": "source.iau.resolution_b2_2012",
        "source_locator": "Resolution B2 exact astronomical-unit definition",
    },
    "derivation.source.seconds_per_day": {
        "result_name": "seconds_per_day",
        "value": "86400",
        "unit_id": "unit.dimensionless",
        "source_id": "source.bipm.si_brochure_9_v4_01",
        "source_locator": (
            "SI base units, exact speed of light, second and kilometre definitions"
        ),
    },
}
EXPECTED_DERIVATION_ID_ROSTER = (
    "derivation.source.c_km_s",
    "derivation.source.gm_de440_km3_s2",
    "derivation.source.lb",
    "derivation.source.au_km",
    "derivation.source.seconds_per_day",
    "derivation.c_au_day",
    "derivation.gm_au3_day2",
    "derivation.tcb_tdb_scaling",
    "derivation.synthetic_perihelion_state",
    "derivation.synthetic_perihelion_e047_state",
    "derivation.synthetic_generic_state",
    "derivation.domain_bounds",
)
EXPECTED_DERIVATION_IDS = frozenset(EXPECTED_DERIVATION_ID_ROSTER)
EXPECTED_UNIT_IDS = frozenset(
    {
        "unit.au",
        "unit.day",
        "unit.au_per_day",
        "unit.au3_per_day2",
        "unit.kilometre",
        "unit.second",
        "unit.kilometre_per_second",
        "unit.kilometre3_per_second2",
        "unit.dimensionless",
    }
)
EXPECTED_COORDINATE_UNIT_REFS = {
    "AU_DAY": {
        "length_unit_ref": "unit.au",
        "time_unit_ref": "unit.day",
        "velocity_unit_ref": "unit.au_per_day",
        "gravitational_parameter_unit_ref": "unit.au3_per_day2",
    },
    "KILOMETRE_SECOND": {
        "length_unit_ref": "unit.kilometre",
        "time_unit_ref": "unit.second",
        "velocity_unit_ref": "unit.kilometre_per_second",
        "gravitational_parameter_unit_ref": "unit.kilometre3_per_second2",
    },
}
EXPECTED_BLOCKED_REASONS = (
    "EXECUTION_IMPLEMENTATION_NOT_REGISTERED",
    "Q0_Q9_NOT_EXECUTED",
    "PROVENANCE_UNRESOLVED",
    "INDEPENDENT_ORACLE_UNRESOLVED",
    "EIH_EVALUATOR_UNRESOLVED",
    "ERROR_BUDGETS_UNRESOLVED",
    "HOLDOUT_CUSTODY_UNRESOLVED",
)
EXPECTED_BLOCKER_SPECS = {
    "blocker.execution_implementation": (
        "EXECUTION_SOURCE",
        tuple(gate for _, gate in EXPECTED_ARM_GATE_MATRIX),
    ),
    "blocker.q0.provenance_review": (
        "PROVENANCE_REVIEW",
        ("gate.q0_package_integrity",),
    ),
    "blocker.q1.independent_equation_oracle": (
        "INDEPENDENT_EQUATION_ORACLE",
        ("gate.q1_equation_discrimination",),
    ),
    "blocker.q2.event_observer_source": (
        "EVENT_OBSERVER_SOURCE",
        ("gate.q2_perihelion_coefficient",),
    ),
    "blocker.q2.event_observer_runtime": (
        "EVENT_OBSERVER_RUNTIME",
        ("gate.q2_perihelion_coefficient",),
    ),
    "blocker.q2.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q2_perihelion_coefficient",),
    ),
    "blocker.q3.oracle_source": (
        "INDEPENDENT_ORACLE_SOURCE",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.oracle_runtime": (
        "INDEPENDENT_ORACLE_RUNTIME",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.dependency_lock": (
        "DEPENDENCY_LOCK",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.method_coefficients": (
        "METHOD_COEFFICIENTS",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.configuration": (
        "CONFIGURATION",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.self_convergence_outputs": (
        "SELF_CONVERGENCE_OUTPUTS",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q3.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q3_high_order_oracle",),
    ),
    "blocker.q4.eih_source": (
        "EIH_SOURCE",
        ("gate.q4_restricted_eih_limit",),
    ),
    "blocker.q4.eih_runtime": (
        "EIH_RUNTIME",
        ("gate.q4_restricted_eih_limit",),
    ),
    "blocker.q4.dependency_lock": (
        "DEPENDENCY_LOCK",
        ("gate.q4_restricted_eih_limit",),
    ),
    "blocker.q4.configuration": (
        "CONFIGURATION",
        ("gate.q4_restricted_eih_limit",),
    ),
    "blocker.q4.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q4_restricted_eih_limit",),
    ),
    "blocker.q5.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q5_step_solver_convergence",),
    ),
    "blocker.q6.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q6_decimal_precision",),
    ),
    "blocker.q7.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q7_unit_time_transform",),
    ),
    "blocker.q8.custodian_attestation": (
        "CUSTODIAN_ATTESTATION",
        ("gate.q8_first_unblinding",),
    ),
    "blocker.q8.sealed_expectations": (
        "SEALED_EXPECTATIONS",
        ("gate.q8_first_unblinding",),
    ),
    "blocker.q8.unblinding_record": (
        "UNBLINDING_RECORD",
        ("gate.q8_first_unblinding",),
    ),
    "blocker.q9.budget_justification": (
        "ERROR_BUDGET_JUSTIFICATION",
        ("gate.q9_claim_audit",),
    ),
    "blocker.q9.expected_outputs": (
        "EXPECTED_OUTPUTS",
        ("gate.q9_claim_audit",),
    ),
}
EXPECTED_PROHIBITED_CLAIMS = (
    "NO_REGISTRY_AUTHORITY",
    "NO_PRODUCTION_INTEGRATOR",
    "NO_OBSERVATIONAL_EPHEMERIS",
    "NO_FULL_EIH_IMPLEMENTATION",
    "NO_DE440_HORIZONS_REPRODUCTION",
    "NO_DETECTION_EXCLUSION_CONSTRAINT",
    "NO_PHYSICAL_COMPLETENESS",
    "NO_SYMPLECTICITY_OR_EXACT_REVERSIBILITY",
    "NO_PERFORMANCE_SUPERIORITY",
)
EXPECTED_RESULT_SEMANTICS = {
    "missing_result_verdict": "BLOCKED",
    "skipped_result_verdict": "BLOCKED",
    "nonfinite_result_verdict": "FAIL",
    "gate_failure_verdict": "FAIL",
    "conflict_verdict": "CONFLICT",
}
EXPECTED_ALLOWED_VERDICTS = (
    "NOT_RUN",
    "BLOCKED",
    "INVALID",
    "FAIL",
    "CONFLICT",
    "ELIGIBLE_FOR_REVIEW_NONAUTHORIZING",
)
EXPECTED_KNOWN_FILE_ROLE_PATHS = {
    "development_solar_1pn_tests": "tests/test_solar_1pn.py",
    "development_reference_integrator_tests": (
        "tests/test_v5_reference_integrator.py"
    ),
}
EXPECTED_LOCKED_FILE_ROLE_PATHS = {
    "qualification_protocol": "runs/v5_solar_1pn_qualification/README.md",
    "solar_1pn_kernel": "src/jxplanetx/solar_1pn.py",
    "reference_force_composition": "src/jxplanetx/v5_reference_dynamics.py",
    "implicit_midpoint_integrator": "src/jxplanetx/v5_implicit_midpoint.py",
    "qualification_validator": "src/jxplanetx/v5_solar_1pn_qualification.py",
    "force_registry_validator": "src/jxplanetx/force_registry_v5.py",
    "provenance_hashing": "src/jxplanetx/provenance.py",
    "decimal_vector_math": "src/jxplanetx/decimal_math.py",
}
EXPECTED_OBSERVABLE_IDS = frozenset(
    {
        "observable.q1.acceleration",
        "observable.q2.perihelion_coefficient",
        "observable.q3.total_state",
        "observable.q3.differential_signal",
        "observable.q4.eih_restricted_limit",
        "observable.q5.order_solver",
        "observable.q6.precision_plateau",
        "observable.q7.transform",
    }
)
EXPECTED_CASE_ROLES = (
    "GENERIC_3D",
    "ECCENTRIC",
    "LONG_ARC",
    "DOMAIN_INSIDE",
    "DOMAIN_BOUNDARY",
    "DOMAIN_OUTSIDE",
    "TRANSFORM_TWIN",
    "SIGNAL_SCALE",
    "EIH_LIMIT",
)
EXPECTED_PERMITTED_CLAIM_CODE = (
    "FROZEN_RESTRICTED_SOLAR_1PN_REFERENCE_REPRODUCTION_"
    "ELIGIBLE_FOR_EXTERNAL_REVIEW_NONAUTHORIZING"
)
EXPECTED_PERMITTED_CLAIM_TEXT = (
    "For the frozen cases and declared domain in qualification package "
    "<qualification_id>, the nonauthorizing JX reference path reproduced the declared "
    "restricted static-Sun, massless-target Schwarzschild 1PN equation within "
    "the preregistered numerical and model-matched error budgets. The resulting "
    "trajectories are MODEL_OUTPUT and are eligible only for external scientific "
    "review."
)
EXPECTED_KNOWN_STATE_RECORD_IDS = frozenset(
    {
        "development.state.solar_fraction_oracle",
        "development.state.solar_locked_component",
        "development.state.solar_zero_velocity",
        "development.state.solar_radial_inward",
        "development.state.solar_circular",
        "development.state.solar_rotation",
        "development.state.solar_rotation_twin",
        "development.state.solar_unit_transform_base",
        "development.state.solar_unit_transform_km_s",
        "development.state.solar_inverse_c_squared_c200",
        "development.state.solar_large_c",
        "development.state.solar_physical_scale",
        "development.state.solar_zero_separation",
        "development.state.solar_domain_compactness_excess",
        "development.state.solar_domain_speed_excess",
        "development.state.reference_force_composition",
        "development.state.reference_default_p60",
        "development.state.reference_default_p70",
        "development.state.reference_large_epoch_p34",
        "development.state.reference_midpoint_domain_candidate",
        "development.state.reference_domain_boundary",
        "development.state.reference_domain_compactness_excess",
        "development.state.reference_domain_speed_excess",
        "development.state.reference_large_c_p70",
        "development.state.reference_default_p60_km_s",
        "development.state.reference_default_p60_rotated",
    }
)
EXPECTED_SCHEMA_PATHS = {
    INPUT_SCHEMA: "schemas/jx-v5-solar-1pn-qualification-inputs-v1.schema.json",
    PLAN_SCHEMA: "schemas/jx-v5-solar-1pn-qualification-plan-v1.schema.json",
    REGISTRATION_SCHEMA: (
        "schemas/jx-v5-solar-1pn-qualification-registration-v1.schema.json"
    ),
    PRIOR_DEVELOPMENT_SCHEMA: (
        "schemas/jx-v5-solar-1pn-prior-development-cases-v1.schema.json"
    ),
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected an object in {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_binding(root: Path, binding: dict, document: dict | None = None) -> None:
    path = root / binding["path"]
    binding["sha256"] = sha256_file(path)
    binding["size_bytes"] = str(path.stat().st_size)
    if "canonical_sha256" in binding:
        if document is None:
            document = _read_json(path)
        binding["canonical_sha256"] = sha256_data(document)


def _rewrite_plan(
    root: Path,
    plan: dict,
    *,
    refresh_identity: bool = True,
) -> None:
    plan_path = root / PLAN_RELATIVE
    registration_path = root / REGISTRATION_RELATIVE
    registration = _read_json(registration_path)
    if refresh_identity:
        identity = qualification_plan_identity_sha256(plan)
        plan["qualification_id"] = QUALIFICATION_ID_PREFIX + identity
        registration["qualification_id"] = plan["qualification_id"]
        registration["plan_identity_sha256"] = identity
    _write_json(plan_path, plan)
    _refresh_binding(root, registration["plan"], plan)
    _write_json(registration_path, registration)


def _rewrite_inputs(root: Path, inputs: dict) -> None:
    input_path = root / INPUTS_RELATIVE
    plan_path = root / PLAN_RELATIVE
    registration_path = root / REGISTRATION_RELATIVE
    plan = _read_json(plan_path)
    registration = _read_json(registration_path)

    _write_json(input_path, inputs)
    _refresh_binding(root, plan["bindings"]["inputs"], inputs)
    identity = qualification_plan_identity_sha256(plan)
    plan["qualification_id"] = QUALIFICATION_ID_PREFIX + identity
    _write_json(plan_path, plan)
    _refresh_binding(root, registration["inputs"], inputs)
    _refresh_binding(root, registration["plan"], plan)
    registration["qualification_id"] = plan["qualification_id"]
    registration["plan_identity_sha256"] = identity
    _write_json(registration_path, registration)


def _manifest_disclosure_sha256(manifest: dict) -> str:
    disclosed = copy.deepcopy(manifest)
    disclosed.pop("manifest_disclosure_sha256", None)
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-prior-development-disclosure/v1",
            "manifest": disclosed,
        }
    )


def _rewrite_prior_manifest(root: Path, manifest: dict) -> None:
    manifest_path = root / PRIOR_MANIFEST_RELATIVE
    plan_path = root / PLAN_RELATIVE
    registration_path = root / REGISTRATION_RELATIVE
    plan = _read_json(plan_path)
    registration = _read_json(registration_path)

    manifest["manifest_disclosure_sha256"] = _manifest_disclosure_sha256(manifest)
    _write_json(manifest_path, manifest)
    manifest_binding = plan["bindings"]["prior_development_manifest"]
    _refresh_binding(root, manifest_binding, manifest)

    prior = plan["prior_development_evidence"]
    prior["known_fixture_ids"] = copy.deepcopy(manifest["known_fixture_ids"])
    prior["known_state_records"] = copy.deepcopy(manifest["known_state_records"])
    prior["known_state_record_count"] = len(manifest["known_state_records"])
    for field in (
        "manifest_disclosure_sha256",
        "known_case_descriptor_sha256s",
        "generated_family_fingerprint_list_sha256s",
    ):
        prior[field] = copy.deepcopy(manifest[field])

    identity = qualification_plan_identity_sha256(plan)
    plan["qualification_id"] = QUALIFICATION_ID_PREFIX + identity
    _write_json(plan_path, plan)

    registration["qualification_id"] = plan["qualification_id"]
    registration["plan_identity_sha256"] = identity
    registration["prior_development_manifest"] = copy.deepcopy(manifest_binding)
    registered_prior = registration["known_prior_evidence"]
    registered_prior["known_fixture_ids"] = copy.deepcopy(
        manifest["known_fixture_ids"]
    )
    registered_prior["known_state_fingerprints"] = sorted(
        {
            record["scientific_fingerprint_sha256"]
            for record in manifest["known_state_records"]
        }
    )
    for field in (
        "manifest_disclosure_sha256",
        "known_case_descriptor_sha256s",
        "generated_family_fingerprint_list_sha256s",
    ):
        registered_prior[field] = copy.deepcopy(manifest[field])
    _refresh_binding(root, registration["plan"], plan)
    _write_json(registration_path, registration)


@contextmanager
def _cloned_project():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "repository"
        shutil.copytree(
            PROJECT_ROOT,
            root,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
        )
        yield root


def _result_values(derivation: dict) -> dict[str, str]:
    return {record["name"]: record["value"] for record in derivation["results"]}


def _result_records(derivation: dict) -> dict[str, dict]:
    return {record["name"]: record for record in derivation["results"]}


def _known_state_fingerprint(record: dict) -> str:
    return qualification_scientific_fingerprint_sha256(
        record,
        record["coefficient_contract"],
        record["coordinate_contract"],
    )


def _blockers_for_gate(gate_id: str) -> set[str]:
    return {
        blocker_id
        for blocker_id, (_, gate_refs) in EXPECTED_BLOCKER_SPECS.items()
        if gate_id in gate_refs
    }


def _schedule_endpoint(schedule: dict, fixtures: dict[str, dict]) -> Decimal:
    return Decimal(fixtures[schedule["fixture_ref"]]["epoch"]) + (
        Decimal(schedule["step_size"]) * schedule["step_count"]
    )


def _plain_decimal(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _holdout_numeric_fingerprint(fixture: dict, coordinate: dict) -> str:
    with localcontext() as arithmetic:
        arithmetic.prec = 90
        if coordinate["units"] == "AU_DAY":
            length_to_au = Decimal(1)
            time_to_day = Decimal(1)
        else:
            length_to_au = Decimal(1) / Decimal("149597870.7")
            time_to_day = Decimal(1) / Decimal("86400")
        return sha256_data(
            {
                "schema": (
                    "jx-v5-solar-1pn-prior-development-"
                    "numeric-state-fingerprint/v1"
                ),
                "synthetic_unit_convention": (
                    "AU_DAY_NUMERIC_ONLY_NO_PHYSICAL_SOLAR_CONTRACT"
                ),
                "epoch": _plain_decimal(Decimal(fixture["epoch"]) * time_to_day),
                "position": [
                    _plain_decimal(Decimal(value) * length_to_au)
                    for value in fixture["position"]
                ],
                "velocity": [
                    _plain_decimal(
                        Decimal(value) * length_to_au / time_to_day
                    )
                    for value in fixture["velocity"]
                ],
            }
        )


def _test_method_locators(path: Path) -> tuple[str, ...]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return tuple(
        f"{class_node.name}.{method.name}"
        for class_node in module.body
        if isinstance(class_node, ast.ClassDef)
        for method in class_node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name.startswith("test_")
    )


def _prior_schema(root: Path) -> dict:
    return _read_json(root / EXPECTED_SCHEMA_PATHS[PRIOR_DEVELOPMENT_SCHEMA])


def _rewrite_family_list(
    root: Path,
    manifest: dict,
    family_document: dict,
) -> dict[str, str]:
    binding = manifest["generated_state_families"][0]["fingerprint_list"]
    family_path = root / binding["path"]
    _write_json(family_path, family_document)
    _refresh_binding(root, binding, family_document)
    _write_json(root / PRIOR_MANIFEST_RELATIVE, manifest)
    return {
        "PRIOR_FINGERPRINT_LIST_FILE_SHA256": binding["sha256"],
        "PRIOR_FINGERPRINT_LIST_SIZE_BYTES": binding["size_bytes"],
        "PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256": binding[
            "canonical_sha256"
        ],
    }


class V5Solar1PNQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prior_manifest = _read_json(PROJECT_ROOT / PRIOR_MANIFEST_RELATIVE)
        family_binding = cls.prior_manifest["generated_state_families"][0][
            "fingerprint_list"
        ]
        cls.prior_family_fingerprints = _read_json(
            PROJECT_ROOT / family_binding["path"]
        )
        cls.plan, cls.inputs, cls.registration, cls.inspection = (
            inspect_solar_1pn_qualification_package(
                REGISTRATION_RELATIVE,
                project_root=PROJECT_ROOT,
            )
        )

    def assertQualificationError(
        self,
        expected_code: str,
        root: Path,
        registration: str | Path = REGISTRATION_RELATIVE,
    ) -> None:
        with self.assertRaises(Solar1PNQualificationError) as raised:
            inspect_solar_1pn_qualification_package(
                registration,
                project_root=root,
            )
        self.assertEqual(raised.exception.code, expected_code)

    def assertPriorManifestError(
        self,
        expected_code: str,
        root: Path,
        manifest: dict,
    ) -> None:
        with self.assertRaises(Solar1PNQualificationError) as raised:
            validate_prior_development_manifest_semantics(
                manifest,
                _prior_schema(root),
                manifest_path=root / PRIOR_MANIFEST_RELATIVE,
                project_root=root,
            )
        self.assertEqual(raised.exception.code, expected_code)

    def assertArtifactBinding(
        self,
        binding: dict,
        *,
        canonical: bool = False,
    ) -> None:
        path = PROJECT_ROOT / binding["path"]
        self.assertTrue(path.is_file(), binding["path"])
        self.assertEqual(binding["sha256"], sha256_file(path))
        self.assertEqual(binding["size_bytes"], str(path.stat().st_size))
        if canonical:
            self.assertEqual(binding["canonical_sha256"], sha256_data(_read_json(path)))

    def test_valid_inspection_is_nonauthorizing_and_never_runs_trajectories(self) -> None:
        refusal = AssertionError("qualification inspection evaluated dynamics")
        with mock.patch(
            "jxplanetx.v5_implicit_midpoint.integrate_reference_trajectory",
            side_effect=refusal,
        ) as trajectory_runner, mock.patch(
            "jxplanetx.v5_implicit_midpoint.implicit_midpoint_step",
            side_effect=refusal,
        ) as trajectory_step, mock.patch(
            "jxplanetx.v5_reference_dynamics.evaluate_reference_rhs",
            side_effect=refusal,
        ) as rhs_evaluator:
            plan, inputs, registration, inspection = (
                inspect_solar_1pn_qualification_package(
                    REGISTRATION_RELATIVE,
                    project_root=PROJECT_ROOT,
                )
            )

        trajectory_runner.assert_not_called()
        trajectory_step.assert_not_called()
        rhs_evaluator.assert_not_called()
        self.assertEqual(plan, self.plan)
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(registration, self.registration)
        self.assertEqual(inspection, self.inspection)
        self.assertFalse(inspection.registry_authorized)
        self.assertFalse(inspection.outcomes_generated)
        self.assertFalse(inspection.ready_for_holdout_execution)
        self.assertEqual(inspection.blocked_reasons, EXPECTED_BLOCKED_REASONS)
        self.assertRegex(inspection.package_sha256, r"^[0-9a-f]{64}$")

    def test_exact_states_verdicts_arms_and_gates_are_frozen(self) -> None:
        self.assertEqual(self.plan["state"], PLAN_STATE)
        self.assertEqual(self.inputs["state"], INPUT_STATE)
        self.assertEqual(self.registration["state"], REGISTRATION_STATE)
        self.assertEqual(
            self.plan["execution_registration_policy"]["current_state"],
            EXECUTION_IMPLEMENTATION_STATE,
        )
        self.assertEqual(
            self.registration["execution_implementation_state"],
            EXECUTION_IMPLEMENTATION_STATE,
        )
        self.assertEqual(
            set(self.plan["verdict_policy"]["allowed_verdicts"]),
            set(EXPECTED_ALLOWED_VERDICTS),
        )
        self.assertEqual(ALLOWED_VERDICTS, EXPECTED_ALLOWED_VERDICTS)
        self.assertEqual(self.plan["verdict_policy"]["pass_effect"], PASS_EFFECT)
        self.assertEqual(
            {record["arm_id"] for record in self.plan["test_arms"]},
            EXPECTED_ARM_IDS,
        )
        self.assertEqual(
            {record["gate_id"] for record in self.plan["gates"]},
            EXPECTED_GATE_IDS,
        )
        self.assertEqual(frozenset(self.inspection.arm_ids), EXPECTED_ARM_IDS)
        self.assertEqual(frozenset(self.inspection.gate_ids), EXPECTED_GATE_IDS)

    def test_q0_q9_matrix_result_semantics_and_readiness_are_exact(self) -> None:
        self.assertEqual(ARM_GATE_MATRIX, EXPECTED_ARM_GATE_MATRIX)
        self.assertEqual(tuple(CASE_ROLES), EXPECTED_CASE_ROLES)
        self.assertTrue(self.plan["all_gates_conjunctive"])
        self.assertEqual(self.plan["result_semantics"], EXPECTED_RESULT_SEMANTICS)
        arms = {record["arm_id"]: record for record in self.plan["test_arms"]}
        gates = {record["gate_id"]: record for record in self.plan["gates"]}
        for arm_id, gate_id in EXPECTED_ARM_GATE_MATRIX:
            arm = arms[arm_id]
            gate = gates[gate_id]
            expected_blockers = _blockers_for_gate(gate_id)
            self.assertEqual(arm["gate_refs"], [gate_id])
            self.assertEqual(gate["arm_refs"], [arm_id])
            self.assertEqual(arm["readiness_state"], "BLOCKED")
            self.assertEqual(gate["readiness_state"], "BLOCKED")
            self.assertEqual(set(arm["blocked_artifact_refs"]), expected_blockers)
            self.assertEqual(set(gate["blocked_artifact_refs"]), expected_blockers)
            for field in (
                "purpose_code",
                "purpose",
                "implementation_independence",
                "required_case_roles",
                "observable_refs",
                "budget_refs",
            ):
                expected = ARM_SPECS[arm_id][field]
                observed = arm[field]
                self.assertEqual(tuple(observed), tuple(expected)) if isinstance(
                    expected, tuple
                ) else self.assertEqual(observed, expected)
            for field in (
                "metric_code",
                "metric",
                "value_kind",
                "operator",
                "threshold",
                "unit",
                "failure_verdict",
                "observable_refs",
                "budget_refs",
            ):
                expected = GATE_SPECS[gate_id][field]
                observed = gate[field]
                self.assertEqual(tuple(observed), tuple(expected)) if isinstance(
                    expected, tuple
                ) else self.assertEqual(observed, expected)

        self.assertEqual(self.plan["verdict_policy"]["pass_effect"], PASS_EFFECT)
        self.assertEqual(
            self.plan["verdict_policy"]["registry_status_after_pass"],
            "UNCHANGED_DRAFT_NONEXECUTABLE_UNQUALIFIED",
        )

    def test_plan_identity_is_derived_from_sentinelized_content(self) -> None:
        identity = qualification_plan_identity_sha256(self.plan)
        self.assertEqual(self.plan["qualification_id"], QUALIFICATION_ID_PREFIX + identity)
        self.assertEqual(self.registration["plan_identity_sha256"], identity)
        self.assertEqual(self.registration["qualification_id"], self.plan["qualification_id"])
        self.assertEqual(self.inspection.qualification_id, self.plan["qualification_id"])

        with _cloned_project() as root:
            plan = _read_json(root / PLAN_RELATIVE)
            plan["limitations"][0] += " Content mutation."
            _rewrite_plan(root, plan, refresh_identity=False)
            self.assertQualificationError("qualification_id_digest", root)

    def test_each_scientific_arm_binds_its_exact_fixture_section(self) -> None:
        arms = {record["arm_id"]: record for record in self.plan["test_arms"]}
        schedules = {
            record["schedule_id"]: record
            for record in self.inputs["integration_schedules"]
        }
        q2 = self.inputs["q2_perihelion_observer"]
        q3 = self.inputs["q3_independent_oracle"]
        q4 = self.inputs["q4_eih_evaluator"]
        q5 = self.inputs["q5_convergence_grid"]
        q6 = self.inputs["q6_precision_cross_grid"]
        q7 = self.inputs["q7_transform_contract"]
        expected = {
            "arm.q0_integrity": set(),
            "arm.q1_equation": {
                "fixture.holdout.generic_equation_3d",
                "fixture.holdout.domain_boundary_3d",
                "fixture.holdout.domain_outside_3d",
                "fixture.holdout.domain_speed_outside_3d",
            },
            "arm.q2_perihelion": {
                cell["fixture_ref"] for cell in q2["case_cells"]
            },
            "arm.q3_high_order_oracle": {
                cell["fixture_ref"] for cell in q3["case_cells"]
            },
            "arm.q4_restricted_eih": {q4["fixture_ref"]},
            "arm.q5_step_solver": {
                schedules[reference]["fixture_ref"]
                for reference in q5["step_schedule_refs"]
            },
            "arm.q6_precision": {q6["fixture_ref"]},
            "arm.q7_transform": {
                pair[field]
                for pair in q7["transform_pairs"]
                for field in ("source_fixture_ref", "transformed_fixture_ref")
            },
            "arm.q8_holdout": set(EXPECTED_FIXTURE_IDS),
            "arm.q9_claim_audit": set(EXPECTED_FIXTURE_IDS),
        }
        self.assertEqual(set(arms), set(expected))
        for arm_id, expected_fixture_refs in expected.items():
            self.assertEqual(
                set(arms[arm_id]["fixture_refs"]),
                expected_fixture_refs,
            )

    def test_exact_foundation_and_file_role_path_rosters_are_frozen(self) -> None:
        self.assertEqual(MODULE_FOUNDATION_COMMIT, FOUNDATION_COMMIT)
        self.assertEqual(KNOWN_FILE_ROLE_PATHS, EXPECTED_KNOWN_FILE_ROLE_PATHS)
        self.assertEqual(LOCKED_FILE_ROLE_PATHS, EXPECTED_LOCKED_FILE_ROLE_PATHS)
        known = {
            record["role"]: record["path"]
            for record in self.plan["prior_development_evidence"]["known_files"]
        }
        locked = {
            record["role"]: record["path"]
            for record in self.plan["bindings"]["locked_files"]
        }
        self.assertEqual(known, EXPECTED_KNOWN_FILE_ROLE_PATHS)
        self.assertEqual(locked, EXPECTED_LOCKED_FILE_ROLE_PATHS)
        manifest_records = {
            record["record_id"]: record
            for record in self.prior_manifest["known_state_records"]
        }
        manifest_specs = {
            record_id: (record["source_role"], record["source_locator"])
            for record_id, record in manifest_records.items()
        }
        self.assertEqual(set(manifest_records), EXPECTED_KNOWN_STATE_RECORD_IDS)
        self.assertEqual(KNOWN_STATE_RECORD_SPECS, manifest_specs)
        prior = self.plan["prior_development_evidence"]
        records = {record["record_id"]: record for record in prior["known_state_records"]}
        self.assertEqual(prior["known_state_record_count"], len(EXPECTED_KNOWN_STATE_RECORD_IDS))
        self.assertEqual(set(records), EXPECTED_KNOWN_STATE_RECORD_IDS)
        self.assertEqual(prior["known_state_records"], self.prior_manifest["known_state_records"])
        for record_id, (source_role, source_locator) in manifest_specs.items():
            self.assertEqual(records[record_id]["source_role"], source_role)
            self.assertEqual(records[record_id]["source_locator"], source_locator)

    def test_prior_development_manifest_is_exhaustive_bound_and_nonauthorizing(self) -> None:
        manifest = self.prior_manifest
        family_document = self.prior_family_fingerprints
        plan_binding = self.plan["bindings"]["prior_development_manifest"]
        registration_binding = self.registration["prior_development_manifest"]

        self.assertEqual(MODULE_PRIOR_DEVELOPMENT_SCHEMA, PRIOR_DEVELOPMENT_SCHEMA)
        self.assertEqual(
            PRIOR_DEVELOPMENT_MANIFEST_PATH,
            PRIOR_MANIFEST_RELATIVE.as_posix(),
        )
        self.assertEqual(
            PRIOR_DEVELOPMENT_MANIFEST_ID,
            "jx.v5.solar_1pn.prior_development_cases.v1",
        )
        self.assertEqual(
            PRIOR_DEVELOPMENT_MANIFEST_STATE,
            "FROZEN_DISCLOSED_PRIOR_DEVELOPMENT_CASES_NONAUTHORIZING",
        )
        self.assertEqual(
            PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256,
            EXPECTED_MANIFEST_RAW_SHA256,
        )
        self.assertEqual(
            PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256,
            EXPECTED_MANIFEST_CANONICAL_SHA256,
        )
        self.assertEqual(PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES, "132803")
        self.assertEqual(
            PRIOR_DEVELOPMENT_DISCLOSURE_SHA256,
            EXPECTED_MANIFEST_DISCLOSURE_SHA256,
        )
        self.assertEqual(PRIOR_DEVELOPMENT_COUNTS, EXPECTED_MANIFEST_COUNTS)
        self.assertEqual(
            PRIOR_FOUNDATION_SOURCE_PATHS,
            {
                "development_solar_1pn_tests": (
                    "runs/v5_solar_1pn_qualification/prior/"
                    "foundation_test_solar_1pn.py",
                    "tests/test_solar_1pn.py",
                ),
                "development_reference_integrator_tests": (
                    "runs/v5_solar_1pn_qualification/prior/"
                    "foundation_test_v5_reference_integrator.py",
                    "tests/test_v5_reference_integrator.py",
                ),
            },
        )

        self.assertEqual(plan_binding, registration_binding)
        self.assertEqual(plan_binding["path"], PRIOR_MANIFEST_RELATIVE.as_posix())
        self.assertEqual(plan_binding["schema"], PRIOR_DEVELOPMENT_SCHEMA)
        self.assertEqual(
            plan_binding["manifest_id"],
            "jx.v5.solar_1pn.prior_development_cases.v1",
        )
        self.assertEqual(plan_binding["sha256"], EXPECTED_MANIFEST_RAW_SHA256)
        self.assertEqual(
            plan_binding["canonical_sha256"],
            EXPECTED_MANIFEST_CANONICAL_SHA256,
        )
        self.assertArtifactBinding(plan_binding, canonical=True)

        self.assertEqual(manifest["schema"], PRIOR_DEVELOPMENT_SCHEMA)
        self.assertEqual(
            manifest["manifest_id"],
            "jx.v5.solar_1pn.prior_development_cases.v1",
        )
        self.assertEqual(
            manifest["state"],
            "FROZEN_DISCLOSED_PRIOR_DEVELOPMENT_CASES_NONAUTHORIZING",
        )
        self.assertEqual(manifest["foundation_commit"], FOUNDATION_COMMIT)
        self.assertEqual(manifest["counts"], EXPECTED_MANIFEST_COUNTS)
        self.assertEqual(
            manifest["manifest_disclosure_sha256"],
            EXPECTED_MANIFEST_DISCLOSURE_SHA256,
        )
        disclosed = copy.deepcopy(manifest)
        disclosed.pop("manifest_disclosure_sha256")
        self.assertEqual(
            sha256_data(
                {
                    "schema": "jx-v5-solar-1pn-prior-development-disclosure/v1",
                    "manifest": disclosed,
                }
            ),
            EXPECTED_MANIFEST_DISCLOSURE_SHA256,
        )
        self.assertFalse(manifest["outcomes_generated"])
        self.assertFalse(manifest["registry_authorized"])
        self.assertTrue(manifest["excluded_from_decisive_gates"])
        self.assertEqual(manifest["nonclaim"], PRIOR_DEVELOPMENT_NONCLAIM)

        sections = {
            "method_count": manifest["method_roster"],
            "known_fixture_id_count": manifest["known_fixture_ids"],
            "known_state_record_count": manifest["known_state_records"],
            "generated_state_family_count": manifest["generated_state_families"],
            "invalid_case_record_count": manifest["invalid_case_records"],
            "generic_rhs_record_count": manifest["generic_rhs_records"],
            "known_case_descriptor_count": manifest[
                "known_case_descriptor_sha256s"
            ],
        }
        for count_name, records in sections.items():
            self.assertEqual(len(records), EXPECTED_MANIFEST_COUNTS[count_name])

        source_paths = {
            "development_solar_1pn_tests": Path("tests/test_solar_1pn.py"),
            "development_reference_integrator_tests": Path(
                "tests/test_v5_reference_integrator.py"
            ),
        }
        expected_method_locators = set().union(
            *(
                _test_method_locators(PROJECT_ROOT / relative)
                for relative in source_paths.values()
            )
        )
        method_records = {
            record["source_locator"]: record
            for record in manifest["method_roster"]
        }
        self.assertEqual(set(method_records), expected_method_locators)
        self.assertEqual(len(method_records), 43)
        for locator, record in method_records.items():
            self.assertEqual(record["record_id"], f"development.method.{locator}")
            self.assertTrue(record["excluded_from_decisive_gates"])

        state_records = {
            record["record_id"]: record
            for record in manifest["known_state_records"]
        }
        self.assertEqual(set(state_records), EXPECTED_KNOWN_STATE_RECORD_IDS)
        for record in state_records.values():
            self.assertIn(record["source_locator"].split(":", 1)[0], method_records)
            self.assertEqual(
                _known_state_fingerprint(record),
                record["scientific_fingerprint_sha256"],
            )

        invalid_records = {
            record["record_id"]: record
            for record in manifest["invalid_case_records"]
        }
        rhs_records = {
            record["record_id"]: record
            for record in manifest["generic_rhs_records"]
        }
        self.assertEqual(len(invalid_records), 46)
        self.assertEqual(len(rhs_records), 6)
        for record in (*invalid_records.values(), *rhs_records.values()):
            self.assertIn(record["source_locator"].split(":", 1)[0], method_records)
            self.assertTrue(record["excluded_from_decisive_gates"])

        family = manifest["generated_state_families"][0]
        self.assertEqual(
            family["family_id"],
            "development.family.solar_equation_level_perihelion_4096",
        )
        self.assertEqual(family["sample_count"], 4096)
        self.assertTrue(family["excluded_from_decisive_gates"])
        expected_known_fixture_ids = (
            set(state_records)
            | set(invalid_records)
            | set(rhs_records)
            | {family["family_id"]}
        )
        self.assertEqual(set(manifest["known_fixture_ids"]), expected_known_fixture_ids)

        source_raw_sha256s = {
            record["source_role"]: record["foundation_copy"]["sha256"]
            for record in manifest["foundation_sources"]
        }
        descriptor_field_rosters = {
            "METHOD": (
                "case_category",
                "descriptor_code",
                "descriptor",
                "expected_semantics",
            ),
            "INVALID_CASE": (
                "descriptor_code",
                "descriptor",
                "expected_exception",
                "expected_semantics",
            ),
            "GENERIC_RHS": (
                "rhs_kind",
                "initial_epoch",
                "initial_six_vector",
                "synthetic_unit_convention",
                "state_only_numeric_fingerprint_sha256",
                "rhs_descriptor",
                "expected_semantics",
            ),
            "GENERATED_STATE_FAMILY": (
                "generator_descriptor",
                "expected_semantics",
                "sample_count",
                "first_scientific_fingerprint_sha256",
                "last_scientific_fingerprint_sha256",
                "ordered_member_fingerprints_sha256",
                "collision_policy",
            ),
        }

        def descriptor_digest(
            record: dict,
            *,
            record_kind: str,
            record_id_field: str,
        ) -> str:
            return sha256_data(
                {
                    "schema": (
                        "jx-v5-solar-1pn-prior-development-case-descriptor/v1"
                    ),
                    "foundation_commit": FOUNDATION_COMMIT,
                    "source_role": record["source_role"],
                    "source_raw_sha256": source_raw_sha256s[
                        record["source_role"]
                    ],
                    "record_kind": record_kind,
                    "record_id": record[record_id_field],
                    "source_locator": record["source_locator"],
                    "alias_locators": record.get("alias_locators", []),
                    "descriptor": {
                        field: record[field]
                        for field in descriptor_field_rosters[record_kind]
                    },
                }
            )

        descriptor_records = (
            *((record, "METHOD", "record_id") for record in manifest["method_roster"]),
            (family, "GENERATED_STATE_FAMILY", "family_id"),
            *((record, "INVALID_CASE", "record_id") for record in manifest["invalid_case_records"]),
            *((record, "GENERIC_RHS", "record_id") for record in manifest["generic_rhs_records"]),
        )
        descriptor_hashes = set()
        for record, record_kind, record_id_field in descriptor_records:
            observed = descriptor_digest(
                record,
                record_kind=record_kind,
                record_id_field=record_id_field,
            )
            self.assertEqual(record["descriptor_sha256"], observed)
            descriptor_hashes.add(observed)
        self.assertEqual(len(descriptor_hashes), 96)
        self.assertEqual(
            set(manifest["known_case_descriptor_sha256s"]),
            descriptor_hashes,
        )

        family_binding = family["fingerprint_list"]
        self.assertArtifactBinding(family_binding, canonical=True)
        self.assertEqual(family_binding["path"], PRIOR_FINGERPRINT_LIST_PATH)
        self.assertEqual(
            family_binding["sha256"], EXPECTED_FAMILY_LIST_RAW_SHA256
        )
        self.assertEqual(
            family_binding["canonical_sha256"],
            EXPECTED_FAMILY_LIST_CANONICAL_SHA256,
        )
        self.assertEqual(
            family_binding["size_bytes"], EXPECTED_FAMILY_LIST_SIZE_BYTES
        )
        self.assertEqual(
            PRIOR_FINGERPRINT_LIST_FILE_SHA256,
            EXPECTED_FAMILY_LIST_RAW_SHA256,
        )
        self.assertEqual(
            PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256,
            EXPECTED_FAMILY_LIST_CANONICAL_SHA256,
        )
        self.assertEqual(
            PRIOR_FINGERPRINT_LIST_SIZE_BYTES,
            EXPECTED_FAMILY_LIST_SIZE_BYTES,
        )
        self.assertEqual(
            family_binding["schema"],
            "jx-v5-solar-1pn-prior-development-fingerprint-list/v1",
        )
        self.assertEqual(family_document["sample_count"], 4096)
        members = family_document["members"]
        self.assertEqual([member["index"] for member in members], list(range(4096)))
        self.assertEqual(
            [member["anomaly_fraction_numerator"] for member in members],
            list(range(4096)),
        )
        self.assertTrue(
            all(member["anomaly_fraction_denominator"] == 4096 for member in members)
        )
        recomputed_fingerprints = []
        for member in members:
            fingerprint = qualification_scientific_fingerprint_sha256(
                {
                    "epoch": family_document["epoch"],
                    "position": member["position"],
                    "velocity": member["velocity"],
                },
                family_document["coefficient_contract"],
                family_document["coordinate_contract"],
            )
            self.assertEqual(fingerprint, member["scientific_fingerprint_sha256"])
            recomputed_fingerprints.append(fingerprint)
        self.assertEqual(len(set(recomputed_fingerprints)), 4096)
        self.assertEqual(
            family["first_scientific_fingerprint_sha256"],
            recomputed_fingerprints[0],
        )
        self.assertEqual(
            family["last_scientific_fingerprint_sha256"],
            recomputed_fingerprints[-1],
        )
        self.assertEqual(
            recomputed_fingerprints[0],
            EXPECTED_FIRST_FAMILY_FINGERPRINT_SHA256,
        )
        self.assertEqual(
            recomputed_fingerprints[-1],
            EXPECTED_LAST_FAMILY_FINGERPRINT_SHA256,
        )
        self.assertEqual(
            PRIOR_FIRST_MEMBER_FINGERPRINT_SHA256,
            EXPECTED_FIRST_FAMILY_FINGERPRINT_SHA256,
        )
        self.assertEqual(
            PRIOR_LAST_MEMBER_FINGERPRINT_SHA256,
            EXPECTED_LAST_FAMILY_FINGERPRINT_SHA256,
        )
        ordered_digest = sha256_data(
            {
                "schema": (
                    "jx-v5-solar-1pn-prior-development-ordered-fingerprints/v1"
                ),
                "family_id": family["family_id"],
                "fingerprints": recomputed_fingerprints,
            }
        )
        self.assertEqual(ordered_digest, EXPECTED_ORDERED_FAMILY_FINGERPRINTS_SHA256)
        self.assertEqual(
            PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256,
            EXPECTED_ORDERED_FAMILY_FINGERPRINTS_SHA256,
        )
        self.assertEqual(family["ordered_member_fingerprints_sha256"], ordered_digest)
        self.assertEqual(
            family_document["ordered_member_fingerprints_sha256"], ordered_digest
        )
        self.assertEqual(
            manifest["generated_family_fingerprint_list_sha256s"],
            [ordered_digest],
        )
        explicit_fingerprints = {
            record["scientific_fingerprint_sha256"]
            for record in state_records.values()
        }
        self.assertEqual(len(explicit_fingerprints), 25)
        self.assertEqual(
            len(explicit_fingerprints | set(recomputed_fingerprints)),
            4121,
        )

        generic_numeric_fingerprints = set()
        for record in rhs_records.values():
            initial = record["initial_six_vector"]
            numeric_fingerprint = sha256_data(
                {
                    "schema": (
                        "jx-v5-solar-1pn-prior-development-"
                        "numeric-state-fingerprint/v1"
                    ),
                    "synthetic_unit_convention": record[
                        "synthetic_unit_convention"
                    ],
                    "epoch": record["initial_epoch"],
                    "position": initial[:3],
                    "velocity": initial[3:],
                }
            )
            self.assertEqual(
                record["state_only_numeric_fingerprint_sha256"],
                numeric_fingerprint,
            )
            generic_numeric_fingerprints.add(numeric_fingerprint)
        self.assertEqual(len(generic_numeric_fingerprints), 5)

        prior = self.plan["prior_development_evidence"]
        registered = self.registration["known_prior_evidence"]
        self.assertEqual(prior["known_fixture_ids"], manifest["known_fixture_ids"])
        self.assertEqual(prior["known_state_records"], manifest["known_state_records"])
        for field in (
            "manifest_disclosure_sha256",
            "known_case_descriptor_sha256s",
            "generated_family_fingerprint_list_sha256s",
        ):
            self.assertEqual(prior[field], manifest[field])
            self.assertEqual(registered[field], manifest[field])

        foundation_sources = {
            record["source_role"]: record
            for record in manifest["foundation_sources"]
        }
        self.assertEqual(set(foundation_sources), set(source_paths))
        for source_role, source in foundation_sources.items():
            self.assertEqual(source["foundation_commit"], FOUNDATION_COMMIT)
            self.assertEqual(
                source["current_file"]["path"], source_paths[source_role].as_posix()
            )
            self.assertArtifactBinding(source["foundation_copy"])
            self.assertArtifactBinding(source["current_file"])
            proof = source["allowed_delta_proof"]
            self.assertTrue(proof["method_roster_unchanged"])
            self.assertEqual(proof["changed_scientific_case_count"], 0)
            foundation_path = PROJECT_ROOT / source["foundation_copy"]["path"]
            current_path = PROJECT_ROOT / source["current_file"]["path"]
            foundation_locators = _test_method_locators(foundation_path)
            current_locators = _test_method_locators(current_path)
            self.assertEqual(foundation_locators, current_locators)
            roster_digest = sha256_data(
                {
                    "schema": (
                        "jx-v5-solar-1pn-prior-development-method-roster/v1"
                    ),
                    "source_role": source_role,
                    "locators": list(foundation_locators),
                }
            )
            self.assertEqual(
                proof["foundation_method_roster_sha256"],
                roster_digest,
            )
            self.assertEqual(proof["current_method_roster_sha256"], roster_digest)
            foundation_text = foundation_path.read_text(encoding="utf-8")
            current_text = current_path.read_text(encoding="utf-8")
            unified_diff = "".join(
                difflib.unified_diff(
                    foundation_text.splitlines(keepends=True),
                    current_text.splitlines(keepends=True),
                    fromfile=source["foundation_copy"]["path"],
                    tofile=source["current_file"]["path"],
                    n=3,
                )
            )
            self.assertEqual(
                proof["unified_diff_sha256"],
                sha256(unified_diff.encode("utf-8")).hexdigest(),
            )

        solar_source = foundation_sources["development_solar_1pn_tests"]
        self.assertEqual(
            solar_source["allowed_delta_proof"]["policy"],
            "NO_DELTA_BYTE_IDENTICAL",
        )
        self.assertEqual(
            solar_source["allowed_delta_proof"]["changed_test_method_locators"],
            [],
        )
        self.assertEqual(
            solar_source["allowed_delta_proof"]["added_line_sha256"],
            sha256(b"").hexdigest(),
        )
        self.assertEqual(
            (PROJECT_ROOT / solar_source["foundation_copy"]["path"]).read_bytes(),
            (PROJECT_ROOT / solar_source["current_file"]["path"]).read_bytes(),
        )

        reference_source = foundation_sources[
            "development_reference_integrator_tests"
        ]
        reference_proof = reference_source["allowed_delta_proof"]
        self.assertEqual(
            reference_proof["policy"],
            "EXACT_SINGLE_NONSCIENTIFIC_ISOLATION_ALLOWLIST_INSERTION",
        )
        self.assertEqual(
            reference_proof["changed_test_method_locators"],
            [
                "V5IsolationTests."
                "test_legacy_modules_and_cli_do_not_reverse_import_v5_reference_path"
            ],
        )
        added_line = '            "v5_solar_1pn_qualification.py",'
        self.assertEqual(
            reference_proof["added_line_sha256"],
            sha256(added_line.encode("utf-8")).hexdigest(),
        )
        foundation_lines = (
            PROJECT_ROOT / reference_source["foundation_copy"]["path"]
        ).read_text(encoding="utf-8").splitlines()
        current_lines = (
            PROJECT_ROOT / reference_source["current_file"]["path"]
        ).read_text(encoding="utf-8").splitlines()
        insertion_index = foundation_lines.index(
            '            "v5_implicit_midpoint.py",'
        ) + 1
        expected_current_lines = list(foundation_lines)
        expected_current_lines.insert(insertion_index, added_line)
        self.assertEqual(current_lines, expected_current_lines)

    def test_prior_manifest_delta_descriptor_and_schema_mutations_fail_closed(self) -> None:
        with self.subTest(mutation="allowed delta proof"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            reference_source = next(
                record
                for record in manifest["foundation_sources"]
                if record["source_role"]
                == "development_reference_integrator_tests"
            )
            reference_source["allowed_delta_proof"]["unified_diff_sha256"] = (
                "0" * 64
            )
            self.assertPriorManifestError("prior_allowed_delta", root, manifest)

        with self.subTest(mutation="descriptor text"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["method_roster"][0]["descriptor"] += " Mutated."
            self.assertPriorManifestError("prior_descriptor_digest", root, manifest)

        with self.subTest(mutation="known-state roster"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["known_state_records"][0]["record_id"] = (
                "development.state.mutated_manifest_state"
            )
            self.assertPriorManifestError("known_state_roster", root, manifest)

        with self.subTest(mutation="known-state locator"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["known_state_records"][0]["source_locator"] = (
                "TestSolarSchwarzschild1PN.mutated_locator:state"
            )
            self.assertPriorManifestError("known_state_locator", root, manifest)

        with self.subTest(mutation="family binding"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["generated_state_families"][0]["fingerprint_list"][
                "canonical_sha256"
            ] = "0" * 64
            self.assertPriorManifestError("prior_family_binding", root, manifest)

        family_mutations = (
            (
                "family order",
                "prior_family_order",
                lambda document: document["members"][0].update({"index": 1}),
            ),
            (
                "family fingerprint",
                "prior_family_fingerprint",
                lambda document: document["members"][0].update(
                    {"scientific_fingerprint_sha256": "0" * 64}
                ),
            ),
            (
                "family ordered digest",
                "prior_family_digest",
                lambda document: document.update(
                    {"ordered_member_fingerprints_sha256": "0" * 64}
                ),
            ),
        )
        for label, code, mutate in family_mutations:
            with self.subTest(mutation=label), _cloned_project() as root:
                manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
                family_document = _read_json(
                    root
                    / manifest["generated_state_families"][0][
                        "fingerprint_list"
                    ]["path"]
                )
                mutate(family_document)
                patched_constants = _rewrite_family_list(
                    root,
                    manifest,
                    family_document,
                )
                with mock.patch.multiple(
                    qualification_module,
                    **patched_constants,
                ):
                    self.assertPriorManifestError(code, root, manifest)

        with self.subTest(mutation="family collision"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            family_document = _read_json(
                root
                / manifest["generated_state_families"][0]["fingerprint_list"][
                    "path"
                ]
            )
            duplicate = copy.deepcopy(family_document["members"][0])
            duplicate["index"] = 1
            duplicate["anomaly_fraction_numerator"] = 1
            family_document["members"][1] = duplicate
            patched_constants = _rewrite_family_list(
                root,
                manifest,
                family_document,
            )
            with mock.patch.multiple(
                qualification_module,
                **patched_constants,
            ):
                self.assertPriorManifestError(
                    "prior_family_uniqueness",
                    root,
                    manifest,
                )

        with self.subTest(mutation="numeric fingerprint"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["generic_rhs_records"][0][
                "state_only_numeric_fingerprint_sha256"
            ] = "0" * 64
            self.assertPriorManifestError("prior_numeric_fingerprint", root, manifest)

        with self.subTest(mutation="disclosure digest"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["manifest_disclosure_sha256"] = "0" * 64
            self.assertPriorManifestError("prior_disclosure_digest", root, manifest)

        with self.subTest(mutation="unknown manifest field"), _cloned_project() as root:
            manifest = _read_json(root / PRIOR_MANIFEST_RELATIVE)
            manifest["silent_default"] = False
            self.assertPriorManifestError("schema_unknown_field", root, manifest)

        with self.subTest(mutation="registration manifest binding"), _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["prior_development_manifest"]["canonical_sha256"] = (
                "0" * 64
            )
            _write_json(registration_path, registration)
            self.assertQualificationError("prior_manifest_binding", root)

        with self.subTest(mutation="registration disclosure mismatch"), _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["known_prior_evidence"][
                "manifest_disclosure_sha256"
            ] = "0" * 64
            _write_json(registration_path, registration)
            self.assertQualificationError("prior_evidence_mismatch", root)

    def test_exact_blocker_and_claim_rosters_are_consistent_everywhere(self) -> None:
        self.assertEqual(BLOCKED_REASONS, EXPECTED_BLOCKED_REASONS)
        self.assertEqual(BLOCKER_SPECS, EXPECTED_BLOCKER_SPECS)
        self.assertEqual(PROHIBITED_CLAIM_CODES, EXPECTED_PROHIBITED_CLAIMS)
        self.assertEqual(PERMITTED_CLAIM_CODE, EXPECTED_PERMITTED_CLAIM_CODE)
        self.assertEqual(PERMITTED_CLAIM_TEXT, EXPECTED_PERMITTED_CLAIM_TEXT)
        blockers = {
            record["placeholder_id"]: record
            for record in self.inputs["blocked_artifacts"]
        }
        self.assertEqual(set(blockers), set(EXPECTED_BLOCKER_SPECS))
        for blocker_id, (artifact_kind, gate_refs) in EXPECTED_BLOCKER_SPECS.items():
            blocker = blockers[blocker_id]
            self.assertEqual(blocker["status"], "BLOCKED")
            self.assertEqual(blocker["artifact_kind"], artifact_kind)
            self.assertEqual(set(blocker["gate_refs"]), set(gate_refs))
            self.assertTrue(blocker["required_before_execution"])
            self.assertTrue(blocker["resolution_requirement"])

        self.assertEqual(
            tuple(self.registration["blocking_status"]["unresolved_blocker_codes"]),
            EXPECTED_BLOCKED_REASONS,
        )
        self.assertEqual(
            tuple(self.registration["prohibited_claim_codes"]),
            EXPECTED_PROHIBITED_CLAIMS,
        )
        self.assertEqual(
            tuple(self.plan["verdict_policy"]["prohibited_claim_codes"]),
            EXPECTED_PROHIBITED_CLAIMS,
        )
        self.assertEqual(
            tuple(self.inputs["claim_boundary"]["prohibited_claims"]),
            EXPECTED_PROHIBITED_CLAIMS,
        )
        self.assertEqual(self.registration["nonclaim"], NONCLAIM)
        self.assertEqual(self.plan["nonclaim"], NONCLAIM)
        self.assertEqual(self.inputs["claim_boundary"]["nonclaim"], NONCLAIM)
        for document in (
            self.registration,
            self.plan["verdict_policy"],
            self.inputs["claim_boundary"],
        ):
            self.assertEqual(document["permitted_claim_code"], EXPECTED_PERMITTED_CLAIM_CODE)
            self.assertEqual(document["permitted_claim_text"], EXPECTED_PERMITTED_CLAIM_TEXT)

    def test_all_source_hash_size_and_schema_bindings_are_exact(self) -> None:
        self.assertArtifactBinding(self.registration["plan"], canonical=True)
        self.assertArtifactBinding(self.registration["inputs"], canonical=True)
        self.assertArtifactBinding(self.plan["bindings"]["inputs"], canonical=True)
        for binding in (
            self.registration["inputs"],
            self.plan["bindings"]["inputs"],
        ):
            self.assertEqual(binding["sha256"], EXPECTED_INPUTS_RAW_SHA256)
            self.assertEqual(
                binding["canonical_sha256"],
                EXPECTED_INPUTS_CANONICAL_SHA256,
            )
            self.assertEqual(binding["size_bytes"], EXPECTED_INPUTS_SIZE_BYTES)
        self.assertArtifactBinding(
            self.plan["bindings"]["prior_development_manifest"],
            canonical=True,
        )
        self.assertEqual(
            self.registration["prior_development_manifest"],
            self.plan["bindings"]["prior_development_manifest"],
        )
        self.assertArtifactBinding(self.plan["bindings"]["registry"], canonical=True)
        self.assertArtifactBinding(self.plan["bindings"]["scientific_contract"])
        self.assertArtifactBinding(
            self.prior_manifest["generated_state_families"][0][
                "fingerprint_list"
            ],
            canonical=True,
        )
        for source in self.prior_manifest["foundation_sources"]:
            self.assertArtifactBinding(source["foundation_copy"])
            self.assertArtifactBinding(source["current_file"])
        for binding in self.plan["bindings"]["locked_files"]:
            self.assertArtifactBinding(binding)
        for binding in self.plan["prior_development_evidence"]["known_files"]:
            self.assertArtifactBinding(binding)
        for binding in self.inputs["provenance_sources"]:
            self.assertArtifactBinding(binding)

        schemas = {binding["schema"]: binding for binding in self.registration["schemas"]}
        self.assertEqual(set(schemas), set(EXPECTED_SCHEMA_PATHS))
        for schema, path in EXPECTED_SCHEMA_PATHS.items():
            self.assertEqual(schemas[schema]["path"], path)
            self.assertArtifactBinding(schemas[schema], canonical=True)
        self.assertEqual(
            sha256_file(
                PROJECT_ROOT / EXPECTED_SCHEMA_PATHS[PRIOR_DEVELOPMENT_SCHEMA]
            ),
            EXPECTED_PRIOR_SCHEMA_RAW_SHA256,
        )

    def test_disclosed_commit_and_known_fixture_exclusion_are_explicit(self) -> None:
        prior = self.plan["prior_development_evidence"]
        registered_prior = self.registration["known_prior_evidence"]
        self.assertEqual(self.plan["foundation_commit"], FOUNDATION_COMMIT)
        self.assertEqual(prior["development_commit"], FOUNDATION_COMMIT)
        self.assertEqual(registered_prior["development_commit"], FOUNDATION_COMMIT)
        self.assertEqual(prior["disclosure_status"], "DISCLOSED_BEFORE_HOLDOUT_EXECUTION")
        self.assertTrue(prior["excluded_from_decisive_gates"])
        self.assertEqual(registered_prior["known_fixture_ids"], prior["known_fixture_ids"])
        self.assertTrue(prior["known_fixture_ids"])
        self.assertTrue(set(prior["known_fixture_ids"]).isdisjoint(EXPECTED_FIXTURE_IDS))
        known_paths = {record["path"] for record in prior["known_files"]}
        self.assertIn("tests/test_v5_reference_integrator.py", known_paths)

    def test_exact_primary_source_and_derivation_rosters_are_frozen(self) -> None:
        self.assertEqual(SOURCE_LITERAL_SPECS, EXPECTED_SOURCE_LITERAL_SPECS)
        source_paths = {
            record["provenance_id"]: record["path"]
            for record in self.inputs["provenance_sources"]
        }
        self.assertEqual(set(source_paths), EXPECTED_SOURCE_IDS)
        self.assertEqual(source_paths, PROVENANCE_SOURCE_PATHS)
        self.assertEqual(
            {record["derivation_id"] for record in self.inputs["derivations"]},
            EXPECTED_DERIVATION_IDS,
        )
        self.assertEqual(tuple(DERIVATION_IDS), EXPECTED_DERIVATION_ID_ROSTER)
        self.assertNotIn("derivation.offaxis_basis", EXPECTED_DERIVATION_IDS)
        self.assertEqual(len(self.inputs["derivations"]), 12)
        self.assertEqual(
            {record["unit_id"] for record in self.inputs["unit_definitions"]},
            EXPECTED_UNIT_IDS,
        )
        for source in self.inputs["provenance_sources"]:
            self.assertEqual(source["status"], "RETAINED_HASHED")
            self.assertTrue(source["path"].startswith(
                "runs/v5_solar_1pn_qualification/sources/"
            ))
        sources = {
            record["provenance_id"]: record
            for record in self.inputs["provenance_sources"]
        }
        derivations = {
            record["derivation_id"]: record
            for record in self.inputs["derivations"]
        }
        for derivation_id, spec in EXPECTED_SOURCE_LITERAL_SPECS.items():
            derivation = derivations[derivation_id]
            self.assertEqual(derivation["formula"], "SOURCE_LITERAL")
            self.assertEqual(derivation["derivation_role"], "SOURCE_CONSTANT")
            self.assertEqual(derivation["operands"], [])
            self.assertEqual(derivation["provenance_refs"], [spec["source_id"]])
            self.assertEqual(
                derivation["results"],
                [
                    {
                        "name": spec["result_name"],
                        "value": spec["value"],
                        "unit_id": spec["unit_id"],
                    }
                ],
            )
            self.assertEqual(
                sources[spec["source_id"]]["locator"],
                spec["source_locator"],
            )

    def test_exact_c_au_day_lb_de440_gm_and_derived_values(self) -> None:
        units = {record["unit_id"]: record for record in self.inputs["unit_definitions"]}
        derivations = {
            record["derivation_id"]: record for record in self.inputs["derivations"]
        }
        formula_dimensions = {
            "GM_KM3_S2_TO_AU3_DAY2": {
                "OPERAND": {
                    "gm_km3_s2": "LENGTH^3/TIME^2",
                    "seconds_per_day": "DIMENSIONLESS",
                    "km_per_au": "DIMENSIONLESS",
                },
                "RESULT": {"gm_au3_day2": "LENGTH^3/TIME^2"},
            },
            "C_KM_S_TO_AU_DAY": {
                "OPERAND": {
                    "c_km_s": "LENGTH/TIME",
                    "seconds_per_day": "DIMENSIONLESS",
                    "km_per_au": "DIMENSIONLESS",
                },
                "RESULT": {"c_au_day": "LENGTH/TIME"},
            },
            "NEWTONIAN_PERIAPSIS_STATE": {
                "OPERAND": {
                    "mu": "LENGTH^3/TIME^2",
                    "semimajor_axis": "LENGTH",
                    "eccentricity": "DIMENSIONLESS",
                },
                "RESULT": {
                    "periapsis_radius": "LENGTH",
                    "periapsis_speed": "LENGTH/TIME",
                },
            },
            "TCB_TDB_LINEAR_SCALING": {
                "OPERAND": {
                    "l_b": "DIMENSIONLESS",
                    "seconds_per_day": "DIMENSIONLESS",
                    "t0": "TIME",
                    "tdb0": "TIME",
                    "tcb_position": "LENGTH",
                    "tcb_time": "TIME",
                    "tcb_mu": "LENGTH^3/TIME^2",
                    "tcb_velocity": "LENGTH/TIME",
                    "tcb_c": "LENGTH/TIME",
                },
                "RESULT": {
                    "scale_factor": "DIMENSIONLESS",
                    "t0": "TIME",
                    "tdb0": "TIME",
                    "tcb_mu": "LENGTH^3/TIME^2",
                    "tdb_position": "LENGTH",
                    "tdb_time": "TIME",
                    "tdb_mu": "LENGTH^3/TIME^2",
                    "tdb_velocity": "LENGTH/TIME",
                    "tdb_c": "LENGTH/TIME",
                },
            },
            "RATIONAL_ORTHONORMAL_BASIS": {
                "OPERAND": {
                    name: "DIMENSIONLESS"
                    for name in ("p0", "p1", "p2", "q0", "q1", "q2", "n0", "n1", "n2")
                },
                "RESULT": {
                    name: "DIMENSIONLESS"
                    for name in (
                        "p_dot_p",
                        "q_dot_q",
                        "n_dot_n",
                        "p_dot_q",
                        "p_dot_n",
                        "q_dot_n",
                        "orientation_determinant",
                    )
                },
            },
        }
        for derivation in derivations.values():
            self.assertTrue(derivation["derivation_role"])
            signature = {
                (item["value_role"], item["name"]): (
                    item["unit_id"],
                    item["quantity_dimension"],
                )
                for item in derivation["expected_signature"]
            }
            actual = {
                (role, item["name"]): (
                    item["unit_id"],
                    units[item["unit_id"]]["quantity_dimension"],
                )
                for role, records in (
                    ("OPERAND", derivation["operands"]),
                    ("RESULT", derivation["results"]),
                )
                for item in records
            }
            self.assertEqual(signature, actual)
            if derivation["formula"] in formula_dimensions:
                observed_dimensions = {
                    role: {
                        name: dimension
                        for (item_role, name), (_, dimension) in signature.items()
                        if item_role == role
                    }
                    for role in ("OPERAND", "RESULT")
                }
                self.assertEqual(
                    observed_dimensions,
                    formula_dimensions[derivation["formula"]],
                )
        source_results = {
            identifier: _result_values(derivations[identifier])
            for identifier in (
                "derivation.source.c_km_s",
                "derivation.source.gm_de440_km3_s2",
                "derivation.source.lb",
                "derivation.source.au_km",
                "derivation.source.seconds_per_day",
            )
        }
        self.assertIn("299792.458", source_results["derivation.source.c_km_s"].values())
        self.assertIn(
            "132712440041.279419",
            source_results["derivation.source.gm_de440_km3_s2"].values(),
        )
        gm_source = derivations["derivation.source.gm_de440_km3_s2"]
        sources = {
            record["provenance_id"]: record
            for record in self.inputs["provenance_sources"]
        }
        jpl_gm_source = sources["source.jpl.astrodynamic_parameters.de440"]
        self.assertEqual(gm_source["formula"], "SOURCE_LITERAL")
        self.assertEqual(
            gm_source["provenance_refs"],
            ["source.jpl.astrodynamic_parameters.de440"],
        )
        self.assertNotIn("source.naif.gm_de440", gm_source["provenance_refs"])
        self.assertEqual(
            jpl_gm_source["path"],
            "runs/v5_solar_1pn_qualification/sources/"
            "jpl_astrodynamic_parameters_2026-08-29.html",
        )
        self.assertEqual(
            jpl_gm_source["sha256"],
            "168f48b94ae3a65f31c96f081d08e7ca91ecc9e65678fdc57353821b492b313e",
        )
        self.assertEqual(
            jpl_gm_source["locator"],
            "heliocentric gravitational constant GM_sun row: "
            "1.32712440041279419 x 10^20 m^3 s^-2",
        )
        self.assertEqual(
            set(derivations["derivation.gm_au3_day2"]["provenance_refs"]),
            {
                "source.jpl.astrodynamic_parameters.de440",
                "source.iau.resolution_b2_2012",
                "source.bipm.si_brochure_9_v4_01",
            },
        )
        self.assertIn(
            "0.00000001550519768",
            source_results["derivation.source.lb"].values(),
        )
        self.assertIn("149597870.7", source_results["derivation.source.au_km"].values())
        self.assertIn("86400", source_results["derivation.source.seconds_per_day"].values())

        with localcontext() as arithmetic:
            arithmetic.prec = 90
            c_au_day = Decimal("299792.458") * Decimal("86400") / Decimal("149597870.7")
            gm_au3_day2 = (
                Decimal("132712440041.279419")
                * Decimal("86400") ** 2
                / Decimal("149597870.7") ** 3
            )
        self.assertEqual(
            Decimal(_result_values(derivations["derivation.c_au_day"])["c_au_day"]),
            c_au_day,
        )
        self.assertEqual(
            Decimal(
                _result_values(derivations["derivation.gm_au3_day2"])[
                    "gm_au3_day2"
                ]
            ),
            gm_au3_day2,
        )
        scaling = derivations["derivation.tcb_tdb_scaling"]
        scale_values = {
            record["name"]: record["value"] for record in scaling["operands"]
        }
        self.assertIn("0.00000001550519768", scale_values.values())
        self.assertEqual(
            _result_values(scaling)["scale_factor"],
            "0.99999998449480232",
        )

    def test_coefficients_and_state_components_bind_derivation_results_units_and_sources(self) -> None:
        units = {record["unit_id"]: record for record in self.inputs["unit_definitions"]}
        coordinates = {
            record["coordinate_contract_id"]: record
            for record in self.inputs["coordinate_contracts"]
        }
        self.assertEqual(
            {record["units"] for record in coordinates.values()},
            set(EXPECTED_COORDINATE_UNIT_REFS),
        )
        for coordinate in coordinates.values():
            for field, expected_unit_ref in EXPECTED_COORDINATE_UNIT_REFS[
                coordinate["units"]
            ].items():
                self.assertEqual(coordinate[field], expected_unit_ref)
        derivations = {
            record["derivation_id"]: record for record in self.inputs["derivations"]
        }
        expected_dimensions = {
            "gravitational_parameter": "LENGTH^3/TIME^2",
            "speed_of_light": "LENGTH/TIME",
            "maximum_compactness": "DIMENSIONLESS",
            "maximum_speed_fraction_squared": "DIMENSIONLESS",
        }
        coefficients = {
            record["coefficient_set_id"]: record
            for record in self.inputs["coefficient_sets"]
        }
        for coefficient in coefficients.values():
            coordinate = coordinates[coefficient["coordinate_contract_ref"]]
            for field, dimension in expected_dimensions.items():
                value = coefficient[field]
                derivation = derivations[value["derivation_ref"]]
                result = _result_records(derivation)[value["derivation_result_name"]]
                self.assertEqual(value["value"], result["value"])
                self.assertEqual(value["unit_id"], result["unit_id"])
                self.assertEqual(units[value["unit_id"]]["quantity_dimension"], dimension)
                self.assertEqual(set(value["provenance_refs"]), set(derivation["provenance_refs"]))
            self.assertEqual(
                coefficient["gravitational_parameter"]["unit_id"],
                coordinate["gravitational_parameter_unit_ref"],
            )
            self.assertEqual(
                coefficient["speed_of_light"]["unit_id"],
                coordinate["velocity_unit_ref"],
            )

        for fixture in self.inputs["fixtures"]:
            coordinate = coordinates[fixture["coordinate_contract_ref"]]
            bindings = fixture["state_derivation_bindings"]
            groups = (
                ((fixture["epoch"],), (bindings["epoch"],), coordinate["time_unit_ref"]),
                (fixture["position"], bindings["position"], coordinate["length_unit_ref"]),
                (fixture["velocity"], bindings["velocity"], coordinate["velocity_unit_ref"]),
            )
            used = set()
            for values, records, unit_id in groups:
                for value, binding in zip(values, records):
                    used.add(binding["derivation_ref"])
                    result = _result_records(
                        derivations[binding["derivation_ref"]]
                    )[binding["result_name"]]
                    self.assertEqual((value, unit_id), (result["value"], result["unit_id"]))
            self.assertEqual(set(fixture["derivation_refs"]), used)

        with _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            inputs["coefficient_sets"][0]["speed_of_light"][
                "derivation_result_name"
            ] = "missing.result"
            _rewrite_inputs(root, inputs)
            self.assertQualificationError("derivation_result_reference", root)

        with _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            item = inputs["derivations"][0]["expected_signature"][0]
            item["quantity_dimension"] = (
                "TIME" if item["quantity_dimension"] != "TIME" else "LENGTH"
            )
            _rewrite_inputs(root, inputs)
            self.assertQualificationError("derivation_signature", root)

    def test_decimal_contexts_are_exactly_50_70_and_90_digit_half_even(self) -> None:
        contexts = {
            record["decimal_context_id"]: record
            for record in self.inputs["decimal_contexts"]
        }
        self.assertEqual(set(contexts), EXPECTED_CONTEXT_IDS)
        self.assertEqual(
            {record["precision"] for record in contexts.values()},
            {50, 70, 90},
        )
        for context in contexts.values():
            self.assertEqual(context["rounding"], "ROUND_HALF_EVEN")
            self.assertEqual(
                context["traps"],
                [
                    "DivisionByZero",
                    "FloatOperation",
                    "InvalidOperation",
                    "Overflow",
                    "Underflow",
                ],
            )
            self.assertRegex(context["expected_sha256"], r"^[0-9a-f]{64}$")

    def test_q2_q3_and_q4_requirements_are_structured_and_explicitly_blocked(self) -> None:
        q2 = self.inputs["q2_perihelion_observer"]
        self.assertEqual(q2["status"], "BLOCKED")
        self.assertEqual(q2["source_placeholder_ref"], "blocker.q2.event_observer_source")
        self.assertEqual(q2["runtime_placeholder_ref"], "blocker.q2.event_observer_runtime")
        self.assertEqual(q2["expected_outputs_placeholder_ref"], "blocker.q2.expected_outputs")
        self.assertEqual(q2["event_function"], "R_DOT_V_ZERO_OUTWARD_PERIHELION")
        self.assertEqual(q2["event_direction"], "NEGATIVE_TO_POSITIVE")
        self.assertEqual(q2["event_method"], "INDEPENDENT_ARBITRARY_PRECISION_BRACKETED_ROOT")
        self.assertEqual(q2["angle_method"], "HIGH_PRECISION_ORIENTED_ATAN2")
        self.assertFalse(q2["binary_float_trigonometry_allowed"])
        self.assertFalse(q2["linear_event_interpolation_allowed"])
        self.assertEqual(q2["scaling_parameter_kind"], "INVERSE_C_SQUARED")
        self.assertTrue(q2["complete_cartesian_product"])
        self.assertTrue(q2["paired_force_execution_required"])
        self.assertGreaterEqual(q2["orbit_count"], 2)
        for field in (
            "step_grid",
            "event_tolerance_grid",
            "inverse_c_squared_ladder",
            "eccentricity_roster",
        ):
            self.assertGreaterEqual(len(q2[field]), 2)
        expected_q2_cells = {
            (eccentricity, inverse_c_squared, step, event_tolerance)
            for eccentricity in q2["eccentricity_roster"]
            for inverse_c_squared in q2["inverse_c_squared_ladder"]
            for step in q2["step_grid"]
            for event_tolerance in q2["event_tolerance_grid"]
        }
        observed_q2_cells = {
            (
                cell["eccentricity"],
                cell["inverse_c_squared_value"],
                cell["step_size"],
                cell["event_tolerance"],
            )
            for cell in q2["case_cells"]
        }
        self.assertEqual(observed_q2_cells, expected_q2_cells)
        self.assertEqual(len(q2["case_cells"]), 24)
        schedules = {
            record["schedule_id"]: record
            for record in self.inputs["integration_schedules"]
        }
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        derivations = {
            record["derivation_id"]: record for record in self.inputs["derivations"]
        }
        coefficients = {
            record["coefficient_set_id"]: record
            for record in self.inputs["coefficient_sets"]
        }
        orbit_span_bindings = {
            record["fixture_ref"]: record for record in q2["orbit_span_bindings"]
        }
        self.assertEqual(
            set(orbit_span_bindings),
            {cell["fixture_ref"] for cell in q2["case_cells"]},
        )
        for fixture_ref, binding in orbit_span_bindings.items():
            fixture = fixtures[fixture_ref]
            derivation = derivations[binding["periapsis_derivation_ref"]]
            operands = {
                record["name"]: record for record in derivation["operands"]
            }
            coefficient = coefficients[fixture["coefficient_set_ref"]]
            self.assertEqual(derivation["formula"], "NEWTONIAN_PERIAPSIS_STATE")
            self.assertIn(derivation["derivation_id"], fixture["derivation_refs"])
            self.assertEqual(binding["orbit_count"], q2["orbit_count"])
            self.assertEqual(
                (binding["tau_upper_numerator"], binding["tau_upper_denominator"]),
                ("44", "7"),
            )
            self.assertEqual(
                binding["semantics"],
                "CONSERVATIVE_COVERAGE_BOUND_NOT_ANALYTIC_GATE_COEFFICIENT",
            )
            self.assertEqual(
                (
                    operands["mu"]["value"],
                    operands["mu"]["unit_id"],
                ),
                (
                    coefficient["gravitational_parameter"]["value"],
                    coefficient["gravitational_parameter"]["unit_id"],
                ),
            )
            with localcontext() as arithmetic:
                arithmetic.prec = 90
                expected_span = q2["orbit_count"] * (
                    Decimal(44)
                    / Decimal(7)
                    * (
                        Decimal(operands["semimajor_axis"]["value"]) ** 3
                        / Decimal(operands["mu"]["value"])
                    ).sqrt()
                )
            self.assertEqual(Decimal(binding["minimum_span"]), expected_span)
        for cell in q2["case_cells"]:
            fixture = fixtures[cell["fixture_ref"]]
            self.assertTrue({"ECCENTRIC", "LONG_ARC"} <= set(fixture["case_roles"]))
            perihelion_derivations = [
                derivations[reference]
                for reference in fixture["derivation_refs"]
                if derivations[reference]["formula"] == "NEWTONIAN_PERIAPSIS_STATE"
            ]
            self.assertEqual(len(perihelion_derivations), 1)
            perihelion_operands = {
                record["name"]: record["value"]
                for record in perihelion_derivations[0]["operands"]
            }
            self.assertEqual(
                Decimal(perihelion_operands["eccentricity"]),
                Decimal(cell["eccentricity"]),
            )
            speed_of_light = Decimal(
                coefficients[fixture["coefficient_set_ref"]]["speed_of_light"][
                    "value"
                ]
            )
            with localcontext() as arithmetic:
                arithmetic.prec = 90
                self.assertEqual(
                    Decimal(cell["inverse_c_squared_value"]),
                    Decimal(1) / speed_of_light**2,
                )
            newtonian = schedules[cell["newtonian_schedule_ref"]]
            one_pn = schedules[cell["one_pn_schedule_ref"]]
            self.assertEqual(newtonian["fixture_ref"], cell["fixture_ref"])
            self.assertEqual(one_pn["fixture_ref"], cell["fixture_ref"])
            self.assertEqual(newtonian["force_model_set"], "NEWTONIAN_ONLY")
            self.assertEqual(one_pn["force_model_set"], "NEWTONIAN_PLUS_SOLAR_1PN")
            minimum_span = Decimal(
                orbit_span_bindings[cell["fixture_ref"]]["minimum_span"]
            )
            for schedule in (newtonian, one_pn):
                self.assertGreaterEqual(
                    abs(Decimal(schedule["step_size"]) * schedule["step_count"]),
                    minimum_span,
                )
        q2_arm = next(
            record for record in self.plan["test_arms"] if record["arm_id"] == "arm.q2_perihelion"
        )
        self.assertEqual(
            set(q2_arm["fixture_refs"]),
            {cell["fixture_ref"] for cell in q2["case_cells"]},
        )

        q3 = self.inputs["q3_independent_oracle"]
        self.assertEqual(q3["status"], "BLOCKED")
        self.assertGreaterEqual(q3["minimum_method_order"], 8)
        self.assertEqual(q3["arithmetic"], "ARBITRARY_PRECISION")
        self.assertTrue(q3["adaptive"])
        self.assertTrue(q3["no_jx_imports_required"])
        self.assertEqual(q3["independence_attestation_status"], "BLOCKED")
        self.assertTrue(q3["complete_fixture_checkpoint_cross_product"])
        self.assertEqual(
            {
                q3["source_placeholder_ref"],
                q3["runtime_placeholder_ref"],
                q3["dependency_lock_placeholder_ref"],
                q3["method_coefficients_placeholder_ref"],
                q3["configuration_placeholder_ref"],
                q3["self_convergence_outputs_placeholder_ref"],
                q3["expected_outputs_placeholder_ref"],
            },
            {
                "blocker.q3.oracle_source",
                "blocker.q3.oracle_runtime",
                "blocker.q3.dependency_lock",
                "blocker.q3.method_coefficients",
                "blocker.q3.configuration",
                "blocker.q3.self_convergence_outputs",
                "blocker.q3.expected_outputs",
            },
        )
        self.assertTrue(q3["self_convergence"]["required"])
        self.assertEqual(q3["self_convergence"]["result_status"], "BLOCKED")
        self.assertGreaterEqual(len(q3["self_convergence"]["tolerance_levels"]), 2)
        self.assertGreaterEqual(len(q3["self_convergence"]["precision_context_refs"]), 2)
        metrics = {record["metric_id"]: record for record in q3["metrics"]}
        self.assertEqual(set(metrics), {"TOTAL_STATE", "DIFFERENTIAL_SIGNAL"})
        for metric in metrics.values():
            self.assertEqual(metric["acceptance_status"], "BLOCKED")
            self.assertTrue(metric["checkpoint_schedule_refs"])
            self.assertTrue(metric["scale_unit_refs"])
        self.assertEqual(
            metrics["TOTAL_STATE"]["checkpoint_schedule_refs"],
            metrics["DIFFERENTIAL_SIGNAL"]["checkpoint_schedule_refs"],
        )
        convergence = q3["self_convergence"]
        checkpoint_refs = {
            reference
            for metric in metrics.values()
            for reference in metric["checkpoint_schedule_refs"]
        }
        q3_fixture_refs = {cell["fixture_ref"] for cell in q3["case_cells"]}
        q3_schedule_pairs = {
            (
                cell["fixture_ref"],
                cell["checkpoint_schedule_ref"],
                cell["paired_newtonian_schedule_ref"],
            )
            for cell in q3["case_cells"]
        }
        expected_q3_cells = {
            (fixture_ref, schedule_ref, newtonian_ref, tolerance, precision)
            for fixture_ref, schedule_ref, newtonian_ref in q3_schedule_pairs
            for tolerance in convergence["tolerance_levels"]
            for precision in convergence["precision_context_refs"]
        }
        observed_q3_cells = {
            (
                cell["fixture_ref"],
                cell["checkpoint_schedule_ref"],
                cell["paired_newtonian_schedule_ref"],
                cell["self_convergence_tolerance"],
                cell["precision_context_ref"],
            )
            for cell in q3["case_cells"]
        }
        self.assertEqual(observed_q3_cells, expected_q3_cells)
        self.assertEqual(len(q3["case_cells"]), 30)
        self.assertEqual(
            {schedule_ref for _, schedule_ref, _ in q3_schedule_pairs},
            checkpoint_refs,
        )
        for cell in q3["case_cells"]:
            one_pn_schedule = schedules[cell["checkpoint_schedule_ref"]]
            newtonian_schedule = schedules[
                cell["paired_newtonian_schedule_ref"]
            ]
            self.assertEqual(
                one_pn_schedule["fixture_ref"],
                cell["fixture_ref"],
            )
            self.assertEqual(newtonian_schedule["fixture_ref"], cell["fixture_ref"])
            self.assertEqual(
                one_pn_schedule["force_model_set"],
                "NEWTONIAN_PLUS_SOLAR_1PN",
            )
            self.assertEqual(
                newtonian_schedule["force_model_set"],
                "NEWTONIAN_ONLY",
            )
            for field in (
                "decimal_context_ref",
                "step_size",
                "step_count",
                "position_atol",
                "velocity_atol",
                "relative_tolerance",
                "maximum_iterations",
                "initial_guess_policy",
                "method_id",
            ):
                self.assertEqual(one_pn_schedule[field], newtonian_schedule[field])
            self.assertEqual(cell["total_state_metric_id"], "TOTAL_STATE")
            self.assertEqual(cell["differential_signal_metric_id"], "DIFFERENTIAL_SIGNAL")
        q3_arm = next(
            record
            for record in self.plan["test_arms"]
            if record["arm_id"] == "arm.q3_high_order_oracle"
        )
        self.assertEqual(set(q3_arm["fixture_refs"]), q3_fixture_refs)

        q4 = self.inputs["q4_eih_evaluator"]
        self.assertEqual(q4["status"], "BLOCKED")
        self.assertEqual(
            {
                q4["source_placeholder_ref"],
                q4["runtime_placeholder_ref"],
                q4["dependency_lock_placeholder_ref"],
                q4["configuration_placeholder_ref"],
                q4["expected_outputs_placeholder_ref"],
            },
            {
                "blocker.q4.eih_source",
                "blocker.q4.eih_runtime",
                "blocker.q4.dependency_lock",
                "blocker.q4.configuration",
                "blocker.q4.expected_outputs",
            },
        )
        self.assertEqual(q4["coordinate_gauge"], "HARMONIC")
        self.assertEqual((q4["beta"], q4["gamma"]), ("1", "1"))
        self.assertEqual(
            q4["retained_order_policy"],
            "NEWTONIAN_SUBSTITUTION_INSIDE_C_MINUS_2_TERMS",
        )
        self.assertTrue(q4["fixed_total_mu"])
        self.assertTrue(q4["complete_nu_ladder"])
        self.assertEqual(q4["identity_result_status"], "BLOCKED")
        nu = [Decimal(value) for value in q4["nu_ladder"]]
        self.assertGreaterEqual(len(nu), 3)
        self.assertTrue(all(Decimal(0) < value <= Decimal("0.25") for value in nu))
        self.assertEqual(q4["fixture_ref"], "fixture.holdout.eih_mass_ratio")
        eih_fixture = fixtures[q4["fixture_ref"]]
        self.assertIn("EIH_LIMIT", eih_fixture["case_roles"])
        eih_coefficient = coefficients[eih_fixture["coefficient_set_ref"]]
        self.assertEqual(
            (q4["total_mu_value"], q4["total_mu_unit_ref"]),
            (
                eih_coefficient["gravitational_parameter"]["value"],
                eih_coefficient["gravitational_parameter"]["unit_id"],
            ),
        )
        q4_cells = {record["cell_id"]: record for record in q4["case_cells"]}
        self.assertEqual(
            {Decimal(record["nu"]) for record in q4_cells.values()},
            set(nu),
        )
        self.assertEqual(len(q4_cells), len(nu))
        for cell in q4_cells.values():
            self.assertEqual(cell["fixture_ref"], q4["fixture_ref"])
            self.assertEqual(
                (cell["total_mu_value"], cell["total_mu_unit_ref"]),
                (q4["total_mu_value"], q4["total_mu_unit_ref"]),
            )
            self.assertEqual(cell["result_status"], "BLOCKED")
        q4_arm = next(
            record
            for record in self.plan["test_arms"]
            if record["arm_id"] == "arm.q4_restricted_eih"
        )
        self.assertEqual(set(q4_arm["fixture_refs"]), {q4["fixture_ref"]})

    def test_q5_q6_and_all_fixture_schedule_grids_are_complete(self) -> None:
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        schedules = {
            record["schedule_id"]: record
            for record in self.inputs["integration_schedules"]
        }
        self.assertEqual(len(fixtures), 12)
        self.assertEqual(len(schedules), 43)
        covered = {record["fixture_ref"] for record in schedules.values()}
        self.assertEqual(covered, EXPECTED_FIXTURE_IDS)

        q2_steps = {abs(Decimal(value)) for value in self.inputs["q2_perihelion_observer"]["step_grid"]}
        schedule_steps = {abs(Decimal(record["step_size"])) for record in schedules.values()}
        self.assertTrue(q2_steps <= schedule_steps)
        for metric in self.inputs["q3_independent_oracle"]["metrics"]:
            self.assertTrue(set(metric["checkpoint_schedule_refs"]) <= set(schedules))

        q5 = self.inputs["q5_convergence_grid"]
        self.assertEqual(q5["status"], "BLOCKED")
        self.assertEqual(
            q5["expected_outputs_placeholder_ref"],
            "blocker.q5.expected_outputs",
        )
        self.assertTrue(q5["richardson_required"])
        self.assertTrue(q5["signed_time_diagnostic_required"])
        q5_schedules = [schedules[reference] for reference in q5["step_schedule_refs"]]
        self.assertGreaterEqual(len(q5_schedules), 3)
        self.assertEqual(len({record["fixture_ref"] for record in q5_schedules}), 1)
        q5_reference_schedule = q5_schedules[0]
        self.assertEqual(
            q5_reference_schedule["force_model_set"],
            "NEWTONIAN_PLUS_SOLAR_1PN",
        )
        for schedule in q5_schedules[1:]:
            for field in (
                "fixture_ref",
                "force_model_set",
                "decimal_context_ref",
                "position_atol",
                "velocity_atol",
                "relative_tolerance",
                "maximum_iterations",
                "initial_guess_policy",
                "method_id",
            ):
                self.assertEqual(schedule[field], q5_reference_schedule[field])
        self.assertTrue(
            all(
                _schedule_endpoint(record, fixtures) == Decimal(q5["common_endpoint"])
                for record in q5_schedules
            )
        )
        steps = sorted({abs(Decimal(record["step_size"])) for record in q5_schedules}, reverse=True)
        self.assertTrue(
            all(steps[index] == Decimal(2) * steps[index + 1] for index in range(len(steps) - 1))
        )
        self.assertIn(q5["common_endpoint"], q5["checkpoint_epochs"])
        self.assertLessEqual(Decimal(q5["minimum_order"]), Decimal(2))
        self.assertGreaterEqual(Decimal(q5["maximum_order"]), Decimal(2))
        baseline_refs = []
        for pair in q5["solver_tightening_pairs"]:
            baseline = schedules[pair["baseline_schedule_ref"]]
            tightened = schedules[pair["tightened_schedule_ref"]]
            baseline_refs.append(pair["baseline_schedule_ref"])
            self.assertIn(pair["baseline_schedule_ref"], q5["step_schedule_refs"])
            self.assertEqual(
                _schedule_endpoint(tightened, fixtures),
                Decimal(q5["common_endpoint"]),
            )
            for field in ("fixture_ref", "decimal_context_ref", "step_size", "step_count"):
                self.assertEqual(baseline[field], tightened[field])
            self.assertTrue(
                any(
                    Decimal(tightened[field]) < Decimal(baseline[field])
                    for field in ("position_atol", "velocity_atol", "relative_tolerance")
                )
            )
        self.assertEqual(set(baseline_refs), set(q5["step_schedule_refs"]))
        self.assertEqual(len(baseline_refs), len(set(baseline_refs)))

        q6 = self.inputs["q6_precision_cross_grid"]
        self.assertEqual(q6["status"], "BLOCKED")
        self.assertEqual(
            q6["expected_outputs_placeholder_ref"],
            "blocker.q6.expected_outputs",
        )
        self.assertTrue(q6["plateau_rule"])
        self.assertTrue(q6["complete_cartesian_product"])
        self.assertEqual(set(q6["precision_context_refs"]), EXPECTED_CONTEXT_IDS)
        expected_cells = {
            (step, context)
            for step in q6["step_sizes"]
            for context in q6["precision_context_refs"]
        }
        observed_cells = {
            (cell["step_size"], cell["decimal_context_ref"])
            for cell in q6["cross_grid_cells"]
        }
        self.assertEqual(observed_cells, expected_cells)
        q5_by_step = {
            abs(Decimal(schedule["step_size"])): schedule
            for schedule in q5_schedules
        }
        self.assertEqual(q6["fixture_ref"], q5_reference_schedule["fixture_ref"])
        self.assertEqual(q6["common_endpoint"], q5["common_endpoint"])
        self.assertEqual(
            {abs(Decimal(value)) for value in q6["step_sizes"]},
            set(q5_by_step),
        )
        for cell in q6["cross_grid_cells"]:
            schedule = schedules[cell["schedule_ref"]]
            self.assertEqual(schedule["fixture_ref"], q6["fixture_ref"])
            self.assertEqual(schedule["step_size"], cell["step_size"])
            self.assertEqual(schedule["decimal_context_ref"], cell["decimal_context_ref"])
            q5_schedule = q5_by_step[abs(Decimal(cell["step_size"]))]
            for field in (
                "fixture_ref",
                "force_model_set",
                "step_count",
                "position_atol",
                "velocity_atol",
                "relative_tolerance",
                "maximum_iterations",
                "initial_guess_policy",
                "method_id",
                "expected_spec_sha256",
            ):
                self.assertEqual(schedule[field], q5_schedule[field])

    def test_q7_binds_exact_affine_constants_scaling_identities_and_transform_pairs(self) -> None:
        q7 = self.inputs["q7_transform_contract"]
        derivations = {
            record["derivation_id"]: record for record in self.inputs["derivations"]
        }
        self.assertEqual(q7["status"], "BLOCKED")
        self.assertEqual(
            q7["expected_outputs_placeholder_ref"],
            "blocker.q7.expected_outputs",
        )
        self.assertEqual(q7["lb_derivation_ref"], "derivation.source.lb")
        self.assertEqual(q7["scale_factor_derivation_ref"], "derivation.tcb_tdb_scaling")
        lb = _result_records(derivations[q7["lb_derivation_ref"]])["l_b"]
        scaling = derivations[q7["scale_factor_derivation_ref"]]
        operands = {record["name"]: record for record in scaling["operands"]}
        results = _result_records(scaling)
        self.assertEqual(lb["value"], "0.00000001550519768")
        self.assertEqual(operands["l_b"], lb)
        factor = Decimal(1) - Decimal(lb["value"])
        self.assertEqual(Decimal(results["scale_factor"]["value"]), factor)
        self.assertEqual(factor, Decimal("0.99999998449480232"))
        self.assertEqual(
            (operands["tcb_time"]["value"], operands["tcb_time"]["unit_id"]),
            ("211813488000", "unit.second"),
        )
        self.assertEqual(results["tdb_time"]["unit_id"], "unit.second")
        with localcontext() as arithmetic:
            arithmetic.prec = 90
            exact_affine_tdb_time = (
                factor * Decimal(operands["tcb_time"]["value"])
                + Decimal(operands["l_b"]["value"])
                * Decimal(operands["t0"]["value"])
                * Decimal(operands["seconds_per_day"]["value"])
                + Decimal(operands["tdb0"]["value"])
            )
        self.assertEqual(
            exact_affine_tdb_time,
            Decimal("211813487988.746212906242706133120000"),
        )
        self.assertEqual(
            Decimal(results["tdb_time"]["value"]),
            exact_affine_tdb_time,
        )

        self.assertEqual(q7["seconds_per_day"]["value"], "86400")
        self.assertEqual(q7["t0"]["value"], "2443144.5003725")
        self.assertEqual(q7["tdb0"]["value"], "-0.0000655")
        self.assertEqual(q7["seconds_per_day"]["unit_id"], "unit.dimensionless")
        self.assertEqual(q7["t0"]["unit_id"], "unit.day")
        self.assertEqual(q7["tdb0"]["unit_id"], "unit.second")
        for binding in (q7["seconds_per_day"], q7["t0"], q7["tdb0"]):
            result = _result_records(derivations[binding["derivation_ref"]])[
                binding["derivation_result_name"]
            ]
            self.assertEqual((binding["value"], binding["unit_id"]), (result["value"], result["unit_id"]))

        self.assertEqual(
            q7["unit_transform_rules"],
            [
                "POSITION=s_L*POSITION",
                "TIME=s_T*TIME",
                "VELOCITY=(s_L/s_T)*VELOCITY",
                "MU=(s_L^3/s_T^2)*MU",
                "C=(s_L/s_T)*C",
                "ACCELERATION=(s_L/s_T^2)*ACCELERATION",
                "STEP=s_T*STEP",
                "POSITION_ATOL=s_L*POSITION_ATOL",
                "VELOCITY_ATOL=(s_L/s_T)*VELOCITY_ATOL",
            ],
        )
        self.assertEqual(
            q7["tcb_tdb_scaling_rules"],
            [
                "F=1-L_B",
                "DELTA_T_D=F*DELTA_T_B",
                "POSITION_D=F*POSITION_B",
                "MU_D=F*MU_B",
                "VELOCITY_D=VELOCITY_B",
                "C_D=C_B",
                "ACCELERATION_D=ACCELERATION_B/F",
                "STEP_D=F*STEP_B",
                "POSITION_ATOL_D=F*POSITION_ATOL_B",
                "VELOCITY_ATOL_D=VELOCITY_ATOL_B",
            ],
        )
        kinds = [pair["transform_kind"] for pair in q7["transform_pairs"]]
        self.assertEqual(
            kinds.count("EXACT_UNIT_CHANGE"),
            1,
        )
        self.assertEqual(kinds.count("TCB_TO_TDB_COMPATIBLE_SCALING"), 1)
        self.assertEqual(len(kinds), 2)
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        coordinates = {
            record["coordinate_contract_id"]: record
            for record in self.inputs["coordinate_contracts"]
        }
        coefficients = {
            record["coefficient_set_id"]: record
            for record in self.inputs["coefficient_sets"]
        }
        schedules = {
            record["schedule_id"]: record
            for record in self.inputs["integration_schedules"]
        }
        affine_epoch_checked = False
        for pair in q7["transform_pairs"]:
            length_scale = Decimal(pair["length_scale"])
            time_scale = Decimal(pair["time_scale"])
            source = fixtures[pair["source_fixture_ref"]]
            transformed = fixtures[pair["transformed_fixture_ref"]]
            source_coordinate = coordinates[source["coordinate_contract_ref"]]
            transformed_coordinate = coordinates[
                transformed["coordinate_contract_ref"]
            ]
            source_coefficient = coefficients[source["coefficient_set_ref"]]
            transformed_coefficient = coefficients[
                transformed["coefficient_set_ref"]
            ]
            source_schedule = schedules[pair["source_schedule_ref"]]
            transformed_schedule = schedules[pair["transformed_schedule_ref"]]
            self.assertEqual(
                source_schedule["force_model_set"],
                "NEWTONIAN_PLUS_SOLAR_1PN",
            )
            for field in (
                "step_count",
                "relative_tolerance",
                "maximum_iterations",
                "decimal_context_ref",
                "force_model_set",
                "initial_guess_policy",
                "method_id",
            ):
                self.assertEqual(source_schedule[field], transformed_schedule[field])
            with localcontext() as arithmetic:
                arithmetic.prec = 90
                self.assertEqual(
                    Decimal(pair["acceleration_scale"]),
                    length_scale / time_scale**2,
                )
            if pair["transform_kind"] == "EXACT_UNIT_CHANGE":
                self.assertEqual(source_coordinate["units"], "KILOMETRE_SECOND")
                self.assertEqual(transformed_coordinate["units"], "AU_DAY")
                with localcontext() as arithmetic:
                    arithmetic.prec = 90
                    au_km = Decimal("149597870.7")
                    seconds_per_day = Decimal("86400")
                    self.assertEqual(length_scale, Decimal(1) / au_km)
                    self.assertEqual(time_scale, Decimal(1) / seconds_per_day)
                    self.assertEqual(
                        Decimal(transformed["epoch"]),
                        Decimal(source["epoch"]) / seconds_per_day,
                    )
                    self.assertEqual(
                        [Decimal(value) for value in transformed["position"]],
                        [
                            Decimal(value) / au_km
                            for value in source["position"]
                        ],
                    )
                    self.assertEqual(
                        [Decimal(value) for value in transformed["velocity"]],
                        [
                            Decimal(value) * seconds_per_day / au_km
                            for value in source["velocity"]
                        ],
                    )
                    self.assertEqual(
                        Decimal(
                            transformed_coefficient["gravitational_parameter"][
                                "value"
                            ]
                        ),
                        Decimal(
                            source_coefficient["gravitational_parameter"]["value"]
                        )
                        * seconds_per_day**2
                        / au_km**3,
                    )
                    self.assertEqual(
                        Decimal(
                            transformed_coefficient["speed_of_light"]["value"]
                        ),
                        Decimal(source_coefficient["speed_of_light"]["value"])
                        * seconds_per_day
                        / au_km,
                    )
                    self.assertEqual(
                        Decimal(transformed_schedule["step_size"]),
                        Decimal(source_schedule["step_size"])
                        / seconds_per_day,
                    )
                    self.assertEqual(
                        Decimal(transformed_schedule["position_atol"]),
                        Decimal(source_schedule["position_atol"]) / au_km,
                    )
                    self.assertEqual(
                        Decimal(transformed_schedule["velocity_atol"]),
                        Decimal(source_schedule["velocity_atol"])
                        * seconds_per_day
                        / au_km,
                    )
            if pair["transform_kind"] == "TCB_TO_TDB_COMPATIBLE_SCALING":
                self.assertEqual(length_scale, factor)
                self.assertEqual(time_scale, factor)
                if source_coordinate["epoch_kind"] == "DECLARED_COORDINATE_EPOCH":
                    affine_epoch_checked = True
                    with localcontext() as arithmetic:
                        arithmetic.prec = 90
                        time_to_day = (
                            Decimal(1)
                            if source_coordinate["units"] == "AU_DAY"
                            else Decimal(1) / Decimal(86400)
                        )
                        expected_epoch = (
                            factor * Decimal(source["epoch"])
                            + (
                                Decimal(lb["value"])
                                * Decimal(q7["t0"]["value"])
                                + Decimal(q7["tdb0"]["value"])
                                / Decimal(q7["seconds_per_day"]["value"])
                            )
                            / time_to_day
                        )
                    self.assertEqual(Decimal(transformed["epoch"]), expected_epoch)
        self.assertTrue(affine_epoch_checked)

    def test_q8_custody_and_q9_error_budgets_are_structured_blockers(self) -> None:
        q8 = self.inputs["q8_holdout_custody"]
        self.assertEqual(q8["status"], "BLOCKED")
        self.assertEqual(set(q8["holdout_fixture_refs"]), EXPECTED_FIXTURE_IDS)
        self.assertEqual(tuple(q8["required_case_roles"]), EXPECTED_CASE_ROLES)
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        self.assertEqual(
            {
                role
                for fixture_ref in q8["holdout_fixture_refs"]
                for role in fixtures[fixture_ref]["case_roles"]
            },
            set(EXPECTED_CASE_ROLES),
        )
        self.assertEqual(
            q8["expectation_visibility"],
            "UNAVAILABLE_TO_IMPLEMENTATION_TEAM_BEFORE_UNBLINDING",
        )
        self.assertTrue(q8["first_valid_unblinding_only"])
        self.assertFalse(q8["post_unblinding_tuning_allowed"])
        self.assertEqual(q8["sealed_expectations_status"], "BLOCKED")
        self.assertEqual(q8["unblinding_status"], "BLOCKED")
        self.assertEqual(
            {
                q8["custodian_attestation_placeholder_ref"],
                q8["sealed_expectations_placeholder_ref"],
                q8["unblinding_record_placeholder_ref"],
            },
            {
                "blocker.q8.custodian_attestation",
                "blocker.q8.sealed_expectations",
                "blocker.q8.unblinding_record",
            },
        )

        q9 = self.inputs["q9_error_budgets"]
        self.assertEqual(q9["status"], "BLOCKED")
        self.assertEqual(q9["combination_rule"], "CONSERVATIVE_SUM")
        self.assertEqual(q9["justification_placeholder_ref"], "blocker.q9.budget_justification")
        self.assertEqual(q9["expected_outputs_placeholder_ref"], "blocker.q9.expected_outputs")
        observables = {record["observable_id"]: record for record in q9["observables"]}
        self.assertEqual(set(observables), EXPECTED_OBSERVABLE_IDS)
        self.assertEqual(set(OBSERVABLE_IDS), EXPECTED_OBSERVABLE_IDS)
        expected_components = {
            "oracle",
            "step",
            "solve",
            "decimal",
            "event",
            "transform",
            "parameter",
            "analytic",
            "model",
        }
        for observable in observables.values():
            self.assertEqual(observable["combination_rule"], "CONSERVATIVE_SUM")
            self.assertTrue(observable["component_gates_required"])
            self.assertTrue(observable["total_gate_required"])
            self.assertEqual(observable["total_status"], "BLOCKED")
            self.assertIsNone(observable["total_value"])
            self.assertEqual(set(observable["components"]), expected_components)
            for component in observable["components"].values():
                self.assertEqual(component["status"], "BLOCKED")
                self.assertIsNone(component["value"])
                self.assertTrue(component["reason"])

    def test_q2_through_q9_semantic_mutations_fail_closed(self) -> None:
        def q2(inputs: dict) -> None:
            inputs["q2_perihelion_observer"]["orbit_count"] = 1

        def q3(inputs: dict) -> None:
            inputs["q3_independent_oracle"]["minimum_method_order"] = 7

        def q4(inputs: dict) -> None:
            inputs["q4_eih_evaluator"]["nu_ladder"] = ["0.1", "0.2"]

        def q5(inputs: dict) -> None:
            inputs["q5_convergence_grid"]["common_endpoint"] = "999"

        def q6(inputs: dict) -> None:
            section = inputs["q6_precision_cross_grid"]
            section["cross_grid_cells"][0]["schedule_ref"] = section[
                "cross_grid_cells"
            ][-1]["schedule_ref"]

        def q7(inputs: dict) -> None:
            pair = next(
                record
                for record in inputs["q7_transform_contract"]["transform_pairs"]
                if record["transform_kind"] == "TCB_TO_TDB_COMPATIBLE_SCALING"
            )
            pair["length_scale"] = "1"

        def q8(inputs: dict) -> None:
            inputs["q8_holdout_custody"]["holdout_fixture_refs"].pop()

        def q9(inputs: dict) -> None:
            inputs["q9_error_budgets"]["observables"][0]["unit_ref"] = "unit.unknown"

        cases = (
            ("Q2", "q2_orbit_count", q2),
            ("Q3", "q3_method_order", q3),
            ("Q4", "q4_nu_ladder", q4),
            ("Q5", "q5_common_endpoint", q5),
            ("Q6", "q6_cell_binding", q6),
            ("Q7", "q7_tcb_tdb_contract", q7),
            ("Q8", "q8_holdout_roster", q8),
            ("Q9", "unknown_reference", q9),
        )
        for label, code, mutate in cases:
            with self.subTest(section=label), _cloned_project() as root:
                inputs = _read_json(root / INPUTS_RELATIVE)
                mutate(inputs)
                _rewrite_inputs(root, inputs)
                self.assertQualificationError(code, root)

    def test_final_matrix_grid_role_and_claim_closures_fail_closed(self) -> None:
        def swap_q2_eccentricity_fixtures(inputs: dict) -> None:
            cells = inputs["q2_perihelion_observer"]["case_cells"]
            left = next(
                cell
                for cell in cells
                if any(
                    other["eccentricity"] != cell["eccentricity"]
                    and other["inverse_c_squared_value"]
                    == cell["inverse_c_squared_value"]
                    and other["step_size"] == cell["step_size"]
                    and other["event_tolerance"] == cell["event_tolerance"]
                    for other in cells
                )
            )
            right = next(
                cell
                for cell in cells
                if cell["eccentricity"] != left["eccentricity"]
                and cell["inverse_c_squared_value"]
                == left["inverse_c_squared_value"]
                and cell["step_size"] == left["step_size"]
                and cell["event_tolerance"] == left["event_tolerance"]
            )
            fields = (
                "fixture_ref",
                "newtonian_schedule_ref",
                "one_pn_schedule_ref",
            )
            left_values = {field: left[field] for field in fields}
            for field in fields:
                left[field], right[field] = right[field], left_values[field]

        def inflate_q2_orbit_requirement(inputs: dict) -> None:
            q2 = inputs["q2_perihelion_observer"]
            derivations = {
                record["derivation_id"]: record
                for record in inputs["derivations"]
            }
            q2["orbit_count"] = 1000000
            for binding in q2["orbit_span_bindings"]:
                derivation = derivations[binding["periapsis_derivation_ref"]]
                operands = {
                    record["name"]: Decimal(record["value"])
                    for record in derivation["operands"]
                }
                binding["orbit_count"] = q2["orbit_count"]
                with localcontext() as arithmetic:
                    arithmetic.prec = 90
                    minimum_span = q2["orbit_count"] * (
                        Decimal(44)
                        / Decimal(7)
                        * (
                            operands["semimajor_axis"] ** 3 / operands["mu"]
                        ).sqrt()
                    )
                binding["minimum_span"] = format(minimum_span, "f")

        def break_coordinate_unit_mapping(inputs: dict) -> None:
            coordinate = next(
                record
                for record in inputs["coordinate_contracts"]
                if record["units"] == "AU_DAY"
            )
            coordinate["length_unit_ref"] = "unit.kilometre"

        def strip_eih_domain_role(inputs: dict) -> None:
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == "fixture.holdout.eih_mass_ratio"
            )
            fixture["case_roles"].remove("DOMAIN_INSIDE")
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        def add_unpaired_signal_role(inputs: dict) -> None:
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == "fixture.holdout.generic_equation_3d"
            )
            fixture["case_roles"].append("SIGNAL_SCALE")
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        def add_false_generic_role(inputs: dict) -> None:
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"]
                == "fixture.holdout.synthetic_perihelion_3d"
            )
            fixture["case_roles"].append("GENERIC_3D")
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        def break_q3_force_pair(inputs: dict) -> None:
            q3 = inputs["q3_independent_oracle"]
            cell = q3["case_cells"][0]
            replacement = next(
                candidate
                for candidate in inputs["integration_schedules"]
                if candidate["force_model_set"] == "NEWTONIAN_ONLY"
                and candidate["fixture_ref"] != cell["fixture_ref"]
            )
            cell["paired_newtonian_schedule_ref"] = replacement["schedule_id"]

        def break_q5_schedule_policy(inputs: dict) -> None:
            q5 = inputs["q5_convergence_grid"]
            schedules = {
                record["schedule_id"]: record
                for record in inputs["integration_schedules"]
            }
            q5_context = schedules[q5["step_schedule_refs"][0]][
                "decimal_context_ref"
            ]
            replacement = next(
                schedules[cell["schedule_ref"]]
                for cell in inputs["q6_precision_cross_grid"]["cross_grid_cells"]
                if schedules[cell["schedule_ref"]]["decimal_context_ref"]
                != q5_context
            )
            matching_index = next(
                index
                for index, reference in enumerate(q5["step_schedule_refs"])
                if abs(Decimal(schedules[reference]["step_size"]))
                == abs(Decimal(replacement["step_size"]))
            )
            q5["step_schedule_refs"][matching_index] = replacement["schedule_id"]

        def break_q6_cell_policy(inputs: dict) -> None:
            q6 = inputs["q6_precision_cross_grid"]
            cell = q6["cross_grid_cells"][0]
            schedule = copy.deepcopy(next(
                record
                for record in inputs["integration_schedules"]
                if record["schedule_id"] == cell["schedule_ref"]
            ))
            schedule["schedule_id"] = "schedule.mutant.q6_force_policy"
            schedule["force_model_set"] = "NEWTONIAN_ONLY"
            inputs["integration_schedules"].append(schedule)
            cell["schedule_ref"] = schedule["schedule_id"]

        def break_q7_schedule_policy(inputs: dict) -> None:
            pair = inputs["q7_transform_contract"]["transform_pairs"][0]
            schedule = copy.deepcopy(next(
                record
                for record in inputs["integration_schedules"]
                if record["schedule_id"] == pair["transformed_schedule_ref"]
            ))
            schedule["schedule_id"] = "schedule.mutant.q7_force_policy"
            schedule["force_model_set"] = "NEWTONIAN_ONLY"
            inputs["integration_schedules"].append(schedule)
            pair["transformed_schedule_ref"] = schedule["schedule_id"]

        def break_q7_exact_unit_scale(inputs: dict) -> None:
            pair = next(
                record
                for record in inputs["q7_transform_contract"]["transform_pairs"]
                if record["transform_kind"] == "EXACT_UNIT_CHANGE"
            )
            pair["length_scale"] = "1"

        def perturb_q7_exact_unit_position(inputs: dict) -> None:
            pair = next(
                record
                for record in inputs["q7_transform_contract"]["transform_pairs"]
                if record["transform_kind"] == "EXACT_UNIT_CHANGE"
            )
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == pair["transformed_fixture_ref"]
            )
            original = Decimal(fixture["position"][0])
            last_place = Decimal(1).scaleb(original.as_tuple().exponent)
            mutated = format(original + last_place, "f")
            fixture["position"][0] = mutated
            binding = fixture["state_derivation_bindings"]["position"][0]
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"] == binding["derivation_ref"]
            )
            next(
                result
                for result in derivation["results"]
                if result["name"] == binding["result_name"]
            )["value"] = mutated
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        def perturb_q7_exact_unit_schedule(inputs: dict) -> None:
            pair = next(
                record
                for record in inputs["q7_transform_contract"]["transform_pairs"]
                if record["transform_kind"] == "EXACT_UNIT_CHANGE"
            )
            schedule = next(
                record
                for record in inputs["integration_schedules"]
                if record["schedule_id"] == pair["transformed_schedule_ref"]
            )
            with localcontext() as arithmetic:
                arithmetic.prec = 120
                schedule["velocity_atol"] = format(
                    Decimal(schedule["velocity_atol"]) * Decimal(2),
                    "f",
                )
            schedule["expected_spec_sha256"] = qualification_module.ImplicitMidpointSpec(
                step_size=Decimal(schedule["step_size"]),
                step_count=schedule["step_count"],
                position_atol=Decimal(schedule["position_atol"]),
                velocity_atol=Decimal(schedule["velocity_atol"]),
                relative_tolerance=Decimal(schedule["relative_tolerance"]),
                maximum_iterations=schedule["maximum_iterations"],
                initial_guess_policy=schedule["initial_guess_policy"],
                method_id=schedule["method_id"],
                registry_authorized=False,
            ).sha256

        def substitute_rounded_naif_gm_provenance(inputs: dict) -> None:
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"]
                == "derivation.source.gm_de440_km3_s2"
            )
            derivation["provenance_refs"] = ["source.naif.gm_de440"]

        def mutate_source_literal_value(
            inputs: dict,
            derivation_id: str,
        ) -> None:
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"] == derivation_id
            )
            derivation["results"][0]["value"] = format(
                Decimal(derivation["results"][0]["value"])
                + Decimal("0.000000000000000001"),
                "f",
            )

        def mutate_lb_source_literal_unit(inputs: dict) -> None:
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"] == "derivation.source.lb"
            )
            derivation["results"][0]["unit_id"] = "unit.second"
            signature = next(
                record
                for record in derivation["expected_signature"]
                if record["value_role"] == "RESULT"
            )
            signature["unit_id"] = "unit.second"
            signature["quantity_dimension"] = "TIME"

        def mutate_au_source_literal_provenance(inputs: dict) -> None:
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"] == "derivation.source.au_km"
            )
            derivation["provenance_refs"] = [
                "source.bipm.si_brochure_9_v4_01"
            ]

        def mutate_bipm_source_literal_locator(inputs: dict) -> None:
            source = next(
                record
                for record in inputs["provenance_sources"]
                if record["provenance_id"]
                == "source.bipm.si_brochure_9_v4_01"
            )
            source["locator"] = "mutated locator"

        def perturb_affine_tdb_time(inputs: dict) -> None:
            derivation = next(
                record
                for record in inputs["derivations"]
                if record["derivation_id"] == "derivation.tcb_tdb_scaling"
            )
            result = next(
                record
                for record in derivation["results"]
                if record["name"] == "tdb_time"
            )
            result["value"] = format(
                Decimal(result["value"])
                + Decimal("0.000000000000000000000001"),
                "f",
            )

        def delete_domain_axis_fixture(inputs: dict, fixture_id: str) -> None:
            inputs["fixtures"] = [
                fixture
                for fixture in inputs["fixtures"]
                if fixture["fixture_id"] != fixture_id
            ]
            inputs["integration_schedules"] = [
                schedule
                for schedule in inputs["integration_schedules"]
                if schedule["fixture_ref"] != fixture_id
            ]
            inputs["q8_holdout_custody"]["holdout_fixture_refs"].remove(
                fixture_id
            )

        def relabel_domain_axis_fixture(inputs: dict, fixture_id: str) -> None:
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == fixture_id
            )
            fixture["case_roles"] = ["DOMAIN_INSIDE"]
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        def rebind_domain_axis_fixture(
            inputs: dict,
            fixture_id: str,
            replacement_fixture_id: str,
        ) -> None:
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == fixture_id
            )
            replacement = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"] == replacement_fixture_id
            )
            for field in (
                "epoch",
                "position",
                "velocity",
                "coordinate_contract_ref",
                "coefficient_set_ref",
                "derivation_refs",
                "state_derivation_bindings",
            ):
                fixture[field] = copy.deepcopy(replacement[field])
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )

        input_cases = (
            (
                "missing frozen derivation",
                "derivation_roster",
                lambda inputs: inputs["derivations"].pop(),
            ),
            (
                "extra frozen derivation",
                "derivation_roster",
                lambda inputs: inputs["derivations"].append(
                    {
                        **copy.deepcopy(inputs["derivations"][-1]),
                        "derivation_id": "derivation.unregistered_extra",
                    }
                ),
            ),
            (
                "relabeled frozen derivation",
                "derivation_roster",
                lambda inputs: inputs["derivations"][0].update(
                    {"derivation_id": "derivation.relabelled_source"}
                ),
            ),
            (
                "Q2 Cartesian case matrix",
                "q2_cartesian_grid",
                lambda inputs: inputs["q2_perihelion_observer"]["case_cells"].pop(),
            ),
            (
                "Q2 fixture eccentricity binding",
                "q2_eccentricity_binding",
                swap_q2_eccentricity_fixtures,
            ),
            (
                "Q2 conservative orbit span",
                "q2_orbit_span",
                inflate_q2_orbit_requirement,
            ),
            (
                "Q3 fixture-checkpoint case matrix",
                "q3_checkpoint_matrix",
                lambda inputs: inputs["q3_independent_oracle"]["case_cells"].pop(),
            ),
            (
                "Q3 paired Newtonian schedule",
                "q3_force_pair",
                break_q3_force_pair,
            ),
            (
                "Q4 fixed-total-mu fixture binding",
                "q4_total_mu_binding",
                lambda inputs: inputs["q4_eih_evaluator"].update(
                    {"total_mu_value": "1.1"}
                ),
            ),
            (
                "Q4 complete nu case roster",
                "q4_nu_roster",
                lambda inputs: inputs["q4_eih_evaluator"]["case_cells"][0].update(
                    {"nu": "0.0000000001"}
                ),
            ),
            (
                "Q5 solver-pair coverage",
                "q5_solver_coverage",
                lambda inputs: inputs["q5_convergence_grid"][
                    "solver_tightening_pairs"
                ].pop(),
            ),
            (
                "Q5 common schedule policy",
                "q5_schedule_policy",
                break_q5_schedule_policy,
            ),
            (
                "Q6 precision-step matrix",
                "q6_cartesian_grid",
                lambda inputs: inputs["q6_precision_cross_grid"][
                    "cross_grid_cells"
                ].pop(),
            ),
            (
                "Q5/Q6 cell policy binding",
                "q5_q6_cell_policy",
                break_q6_cell_policy,
            ),
            (
                "Q7 exact transform-kind roster",
                "q7_transform_roster",
                lambda inputs: inputs["q7_transform_contract"]["transform_pairs"][
                    0
                ].update(
                    {
                        "transform_kind": inputs["q7_transform_contract"][
                            "transform_pairs"
                        ][1]["transform_kind"]
                    }
                ),
            ),
            (
                "Q7 same force and context policy",
                "q7_schedule_policy",
                break_q7_schedule_policy,
            ),
            (
                "Q7 exact kilometre-to-AU scale",
                "q7_unit_contract",
                break_q7_exact_unit_scale,
            ),
            (
                "Q7 exact transformed position",
                "q7_position_transform",
                perturb_q7_exact_unit_position,
            ),
            (
                "Q7 exact transformed schedule",
                "q7_schedule_transform",
                perturb_q7_exact_unit_schedule,
            ),
            (
                "Q7 affine constant unit binding",
                "derivation_result_binding",
                lambda inputs: inputs["q7_transform_contract"]["tdb0"].update(
                    {"unit_id": "unit.day"}
                ),
            ),
            (
                "exact DE440 GM source provenance",
                "gm_source_provenance",
                substitute_rounded_naif_gm_provenance,
            ),
            (
                "exact DE440 GM literal",
                "gm_source_provenance",
                lambda inputs: next(
                    record
                    for record in inputs["derivations"]
                    if record["derivation_id"]
                    == "derivation.source.gm_de440_km3_s2"
                )["results"][0].update(
                    {"value": "132712440041.27942"}
                ),
            ),
            (
                "exact speed-of-light literal",
                "source_literal_binding",
                lambda inputs: mutate_source_literal_value(
                    inputs,
                    "derivation.source.c_km_s",
                ),
            ),
            (
                "exact L_B literal unit",
                "source_literal_binding",
                mutate_lb_source_literal_unit,
            ),
            (
                "exact AU literal provenance",
                "source_literal_binding",
                mutate_au_source_literal_provenance,
            ),
            (
                "exact seconds-per-day source locator",
                "source_literal_binding",
                mutate_bipm_source_literal_locator,
            ),
            (
                "exact affine TCB-to-TDB epoch result",
                "derivation_mismatch",
                perturb_affine_tdb_time,
            ),
            (
                "Q9 observable roster",
                "schema_min_items",
                lambda inputs: inputs["q9_error_budgets"]["observables"].pop(),
            ),
            (
                "compactness-only outside fixture deletion",
                "domain_outside_axis_coverage",
                lambda inputs: delete_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_outside_3d",
                ),
            ),
            (
                "speed-only outside fixture deletion",
                "domain_outside_axis_coverage",
                lambda inputs: delete_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_speed_outside_3d",
                ),
            ),
            (
                "compactness-only outside fixture relabeling",
                "derived_domain_role",
                lambda inputs: relabel_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_outside_3d",
                ),
            ),
            (
                "speed-only outside fixture relabeling",
                "derived_domain_role",
                lambda inputs: relabel_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_speed_outside_3d",
                ),
            ),
            (
                "compactness-only outside fixture rebinding",
                "domain_outside_axis_coverage",
                lambda inputs: rebind_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_outside_3d",
                    "fixture.holdout.domain_speed_outside_3d",
                ),
            ),
            (
                "speed-only outside fixture rebinding",
                "domain_outside_axis_coverage",
                lambda inputs: rebind_domain_axis_fixture(
                    inputs,
                    "fixture.holdout.domain_speed_outside_3d",
                    "fixture.holdout.domain_outside_3d",
                ),
            ),
            (
                "exact coordinate unit mapping",
                "coordinate_unit_roster",
                break_coordinate_unit_mapping,
            ),
            (
                "derived EIH domain role",
                "derived_domain_role",
                strip_eih_domain_role,
            ),
            (
                "derived generic 3D role",
                "derived_generic_3d_role",
                add_false_generic_role,
            ),
            (
                "derived signal-scale family role",
                "derived_case_role",
                add_unpaired_signal_role,
            ),
        )
        for label, code, mutate in input_cases:
            with self.subTest(closure=label), _cloned_project() as root:
                inputs = _read_json(root / INPUTS_RELATIVE)
                mutate(inputs)
                _rewrite_inputs(root, inputs)
                self.assertQualificationError(code, root)

        with self.subTest(closure="derived generic role removal"), _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            for fixture in inputs["fixtures"]:
                if "GENERIC_3D" in fixture["case_roles"]:
                    fixture["case_roles"].remove("GENERIC_3D")
                    fixture["expected_state_record_sha256"] = (
                        qualification_fixture_sha256(fixture)
                    )
            _rewrite_inputs(root, inputs)
            self.assertQualificationError("q8_case_role_coverage", root)

        def set_arm_fixtures(
            plan: dict,
            arm_id: str,
            fixture_refs: list[str],
        ) -> None:
            next(
                arm
                for arm in plan["test_arms"]
                if arm["arm_id"] == arm_id
            )["fixture_refs"] = fixture_refs

        plan_cases = (
            (
                "arm purpose code",
                "arm_semantic_spec",
                lambda plan: plan["test_arms"][0].update(
                    {"purpose_code": "MUTATED_PURPOSE"}
                ),
            ),
            (
                "gate metric code",
                "gate_semantic_spec",
                lambda plan: plan["gates"][0].update(
                    {"metric_code": "MUTATED_METRIC"}
                ),
            ),
            (
                "known-state source locator",
                "prior_manifest_disclosure",
                lambda plan: plan["prior_development_evidence"][
                    "known_state_records"
                ][0].update({"source_locator": "mutated.source.locator"}),
            ),
            (
                "Q4 arm fixture binding",
                "q4_arm_binding",
                lambda plan: next(
                    arm
                    for arm in plan["test_arms"]
                    if arm["arm_id"] == "arm.q4_restricted_eih"
                )["fixture_refs"].append(
                    "fixture.holdout.generic_equation_3d"
                ),
            ),
            (
                "Q1 equation fixture roster",
                "q1_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q1_equation",
                    [
                        "fixture.holdout.generic_equation_3d",
                        "fixture.holdout.domain_boundary_3d",
                        "fixture.holdout.domain_outside_3d",
                    ],
                ),
            ),
            (
                "Q5 convergence fixture roster",
                "q5_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q5_step_solver",
                    ["fixture.holdout.synthetic_perihelion_scaled_3d"],
                ),
            ),
            (
                "Q6 precision fixture roster",
                "q6_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q6_precision",
                    ["fixture.holdout.synthetic_perihelion_scaled_3d"],
                ),
            ),
            (
                "Q7 transform fixture roster",
                "q7_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q7_transform",
                    [
                        "fixture.holdout.tdb_km_s_transform",
                        "fixture.holdout.tdb_au_day_transform",
                    ],
                ),
            ),
            (
                "Q8 custody fixture roster",
                "q8_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q8_holdout",
                    sorted(EXPECTED_FIXTURE_IDS)[:-1],
                ),
            ),
            (
                "Q9 audit fixture roster",
                "q9_arm_binding",
                lambda plan: set_arm_fixtures(
                    plan,
                    "arm.q9_claim_audit",
                    sorted(EXPECTED_FIXTURE_IDS)[:-1],
                ),
            ),
        )
        for label, code, mutate in plan_cases:
            with self.subTest(closure=label), _cloned_project() as root:
                plan = _read_json(root / PLAN_RELATIVE)
                mutate(plan)
                _rewrite_plan(root, plan)
                self.assertQualificationError(code, root)

        claim_locations = (
            ("registration", REGISTRATION_RELATIVE, ("permitted_claim_text",)),
            (
                "plan",
                PLAN_RELATIVE,
                ("verdict_policy", "permitted_claim_text"),
            ),
            (
                "inputs",
                INPUTS_RELATIVE,
                ("claim_boundary", "permitted_claim_text"),
            ),
        )
        for label, relative, keys in claim_locations:
            with self.subTest(claim_document=label), _cloned_project() as root:
                document = _read_json(root / relative)
                target = document
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = "A broader claim is authorized."
                if relative == PLAN_RELATIVE:
                    _rewrite_plan(root, document)
                elif relative == INPUTS_RELATIVE:
                    _rewrite_inputs(root, document)
                else:
                    _write_json(root / relative, document)
                self.assertQualificationError("schema_const", root)

    def test_every_fixture_is_a_fresh_hashed_holdout_used_by_an_arm(self) -> None:
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        coefficients = {
            record["coefficient_set_id"]: record
            for record in self.inputs["coefficient_sets"]
        }
        self.assertEqual(set(fixtures), EXPECTED_FIXTURE_IDS)
        known = set(self.plan["prior_development_evidence"]["known_fixture_ids"])
        used = {
            fixture
            for arm in self.plan["test_arms"]
            for fixture in arm["fixture_refs"]
        }
        self.assertEqual(used, EXPECTED_FIXTURE_IDS)
        self.assertTrue(known.isdisjoint(fixtures))
        observed_roles = set()
        compactness_only_outside = set()
        speed_only_outside = set()
        for fixture in fixtures.values():
            self.assertEqual(fixture["status"], "HOLDOUT")
            self.assertEqual(fixture["evidence_class"], "ASSUMPTION")
            self.assertEqual(
                frozenset(fixture["case_roles"]),
                EXPECTED_FIXTURE_CASE_ROLES[fixture["fixture_id"]],
            )
            observed_roles.update(fixture["case_roles"])
            self.assertEqual(
                fixture["expected_state_record_sha256"],
                qualification_fixture_sha256(fixture),
            )
            coefficient = coefficients[fixture["coefficient_set_ref"]]
            with localcontext() as arithmetic:
                arithmetic.prec = 90
                position = tuple(Decimal(value) for value in fixture["position"])
                velocity = tuple(Decimal(value) for value in fixture["velocity"])
                mu = Decimal(coefficient["gravitational_parameter"]["value"])
                c = Decimal(coefficient["speed_of_light"]["value"])
                radius_squared = sum(
                    (value * value for value in position),
                    Decimal(0),
                )
                speed_squared = sum(
                    (value * value for value in velocity),
                    Decimal(0),
                )
                if radius_squared == 0:
                    derived_domain_role = "DOMAIN_OUTSIDE"
                    generic_3d = False
                else:
                    radius = radius_squared.sqrt()
                    compactness = mu / (radius * c * c)
                    speed_fraction_squared = speed_squared / (c * c)
                    maximum_compactness = Decimal(
                        coefficient["maximum_compactness"]["value"]
                    )
                    maximum_speed = Decimal(
                        coefficient["maximum_speed_fraction_squared"]["value"]
                    )
                    if (
                        compactness <= maximum_compactness
                        and speed_fraction_squared <= maximum_speed
                    ):
                        derived_domain_role = (
                            "DOMAIN_BOUNDARY"
                            if compactness == maximum_compactness
                            or speed_fraction_squared == maximum_speed
                            else "DOMAIN_INSIDE"
                        )
                    else:
                        derived_domain_role = "DOMAIN_OUTSIDE"
                    if (
                        compactness > maximum_compactness
                        and speed_fraction_squared < maximum_speed
                    ):
                        compactness_only_outside.add(fixture["fixture_id"])
                    if (
                        speed_fraction_squared > maximum_speed
                        and compactness < maximum_compactness
                    ):
                        speed_only_outside.add(fixture["fixture_id"])
                    radius_dot_velocity = sum(
                        (
                            position[index] * velocity[index]
                            for index in range(3)
                        ),
                        Decimal(0),
                    )
                    radial = Decimal(4) * mu / radius - speed_squared
                    numerators = tuple(
                        radial * position[index]
                        + Decimal(4) * radius_dot_velocity * velocity[index]
                        for index in range(3)
                    )
                    generic_3d = (
                        all(value != 0 for value in position)
                        and all(value != 0 for value in velocity)
                        and radius_dot_velocity != 0
                        and radial != 0
                        and all(value != 0 for value in numerators)
                    )
            declared_domain_roles = set(fixture["case_roles"]) & {
                "DOMAIN_INSIDE",
                "DOMAIN_BOUNDARY",
                "DOMAIN_OUTSIDE",
            }
            self.assertEqual(declared_domain_roles, {derived_domain_role})
            if "GENERIC_3D" in fixture["case_roles"]:
                self.assertTrue(generic_3d)
        self.assertEqual(
            compactness_only_outside,
            {"fixture.holdout.domain_outside_3d"},
        )
        self.assertEqual(
            speed_only_outside,
            {"fixture.holdout.domain_speed_outside_3d"},
        )
        self.assertEqual(
            fixtures["fixture.holdout.domain_outside_3d"]["velocity"],
            ["0.72", "-0.54", "0"],
        )
        speed_outside = fixtures["fixture.holdout.domain_speed_outside_3d"]
        self.assertEqual(speed_outside["position"], ["1.2", "1.6", "0"])
        self.assertEqual(speed_outside["velocity"], ["0", "1.000001", "0"])
        self.assertEqual(
            speed_outside["coefficient_set_ref"],
            "coefficient.domain.outside",
        )
        self.assertEqual(
            speed_outside["derivation_refs"],
            ["derivation.synthetic_generic_state"],
        )
        self.assertIn(
            ("schedule.coverage.speed_outside", speed_outside["fixture_id"]),
            {
                (schedule["schedule_id"], schedule["fixture_ref"])
                for schedule in self.inputs["integration_schedules"]
            },
        )
        self.assertEqual(observed_roles, set(EXPECTED_CASE_ROLES))
        signal_families = (
            (
                "fixture.holdout.synthetic_perihelion_3d",
                "fixture.holdout.synthetic_perihelion_scaled_3d",
            ),
            (
                "fixture.holdout.synthetic_perihelion_e047_3d",
                "fixture.holdout.synthetic_perihelion_e047_scaled_3d",
            ),
        )
        self.assertEqual(
            {
                fixture_id
                for fixture_id, fixture in fixtures.items()
                if "SIGNAL_SCALE" in fixture["case_roles"]
            },
            {fixture_id for family in signal_families for fixture_id in family},
        )
        for left_id, right_id in signal_families:
            left = fixtures[left_id]
            right = fixtures[right_id]
            for field in ("epoch", "position", "velocity", "coordinate_contract_ref"):
                self.assertEqual(left[field], right[field])
            left_coefficient = coefficients[left["coefficient_set_ref"]]
            right_coefficient = coefficients[right["coefficient_set_ref"]]
            for field in (
                "central_source",
                "gravitational_parameter",
                "maximum_compactness",
                "maximum_speed_fraction_squared",
            ):
                self.assertEqual(left_coefficient[field], right_coefficient[field])
            self.assertNotEqual(
                left_coefficient["speed_of_light"]["value"],
                right_coefficient["speed_of_light"]["value"],
            )
        for schedule in self.inputs["integration_schedules"]:
            self.assertIn(
                schedule["force_model_set"],
                {"NEWTONIAN_ONLY", "NEWTONIAN_PLUS_SOLAR_1PN"},
            )

    def test_scientific_fingerprints_are_id_independent_and_do_not_reuse_prior_values(self) -> None:
        fixtures = {record["fixture_id"]: record for record in self.inputs["fixtures"]}
        coefficients = {
            record["coefficient_set_id"]: record
            for record in self.inputs["coefficient_sets"]
        }
        coordinates = {
            record["coordinate_contract_id"]: record
            for record in self.inputs["coordinate_contracts"]
        }
        holdout_manifest_fingerprints = {
            fixture_id: qualification_scientific_fingerprint_sha256(
                fixture,
                coefficients[fixture["coefficient_set_ref"]],
                coordinates[fixture["coordinate_contract_ref"]],
            )
            for fixture_id, fixture in fixtures.items()
        }
        holdout_fingerprints = {
            fixture_id: qualification_module._transform_invariant_scientific_fingerprint_sha256(
                fixture,
                coefficients[fixture["coefficient_set_ref"]],
                coordinates[fixture["coordinate_contract_ref"]],
            )
            for fixture_id, fixture in fixtures.items()
        }
        de440_transform_fixture_ids = {
            "fixture.holdout.tdb_km_s_transform",
            "fixture.holdout.tdb_au_day_transform",
            "fixture.holdout.tcb_km_s_transform",
        }
        self.assertEqual(
            len(
                {
                    holdout_fingerprints[fixture_id]
                    for fixture_id in de440_transform_fixture_ids
                }
            ),
            1,
        )
        with localcontext() as ambient:
            ambient.prec = 7
            ambient_fingerprints = {
                fixture_id: qualification_module._transform_invariant_scientific_fingerprint_sha256(
                    fixtures[fixture_id],
                    coefficients[
                        fixtures[fixture_id]["coefficient_set_ref"]
                    ],
                    coordinates[
                        fixtures[fixture_id]["coordinate_contract_ref"]
                    ],
                )
                for fixture_id in de440_transform_fixture_ids
            }
        self.assertEqual(
            ambient_fingerprints,
            {
                fixture_id: holdout_fingerprints[fixture_id]
                for fixture_id in de440_transform_fixture_ids
            },
        )
        last_digit_mutant = copy.deepcopy(
            fixtures["fixture.holdout.tdb_km_s_transform"]
        )
        component_index = next(
            index
            for index, value in enumerate(last_digit_mutant["position"])
            if Decimal(value) != 0
        )
        original_component = Decimal(last_digit_mutant["position"][component_index])
        last_place = Decimal(1).scaleb(original_component.as_tuple().exponent)
        last_digit_mutant["position"][component_index] = format(
            original_component + last_place,
            "f",
        )
        self.assertNotEqual(
            qualification_module._transform_invariant_scientific_fingerprint_sha256(
                last_digit_mutant,
                coefficients[last_digit_mutant["coefficient_set_ref"]],
                coordinates[last_digit_mutant["coordinate_contract_ref"]],
            ),
            holdout_fingerprints[last_digit_mutant["fixture_id"]],
        )
        declared_pairs = {
            frozenset((pair["source_fixture_ref"], pair["transformed_fixture_ref"]))
            for pair in self.inputs["q7_transform_contract"]["transform_pairs"]
        }
        for pair in declared_pairs:
            left, right = tuple(pair)
            self.assertEqual(holdout_fingerprints[left], holdout_fingerprints[right])

        transform_graph = {fixture_id: set() for fixture_id in fixtures}
        for pair in declared_pairs:
            left, right = tuple(pair)
            transform_graph[left].add(right)
            transform_graph[right].add(left)

        fingerprint_classes: dict[str, set[str]] = {}
        for fixture_id, fingerprint in holdout_fingerprints.items():
            fingerprint_classes.setdefault(fingerprint, set()).add(fixture_id)
        collision_classes = {
            fingerprint: fixture_ids
            for fingerprint, fixture_ids in fingerprint_classes.items()
            if len(fixture_ids) > 1
        }

        declared_transform_fixtures = set().union(*declared_pairs)
        self.assertEqual(
            set().union(*collision_classes.values()),
            declared_transform_fixtures,
        )
        for fixture_ids in collision_classes.values():
            self.assertTrue(
                all(
                    "TRANSFORM_TWIN" in fixtures[fixture_id]["case_roles"]
                    for fixture_id in fixture_ids
                )
            )
            pending = [next(iter(fixture_ids))]
            visited: set[str] = set()
            while pending:
                fixture_id = pending.pop()
                if fixture_id in visited:
                    continue
                visited.add(fixture_id)
                pending.extend(transform_graph[fixture_id] - visited)
            self.assertTrue(fixture_ids <= visited)

        nonpaired_fixture_ids = set(fixtures) - declared_transform_fixtures
        self.assertEqual(
            len({holdout_fingerprints[fixture_id] for fixture_id in nonpaired_fixture_ids}),
            len(nonpaired_fixture_ids),
        )

        renamed = copy.deepcopy(next(iter(fixtures.values())))
        renamed["target"] = "RENAMED_TARGET_LABEL"
        self.assertEqual(
            qualification_module._transform_invariant_scientific_fingerprint_sha256(
                renamed,
                coefficients[renamed["coefficient_set_ref"]],
                coordinates[renamed["coordinate_contract_ref"]],
            ),
            holdout_fingerprints[next(iter(fixtures))],
        )
        self.assertEqual(
            qualification_scientific_fingerprint_sha256(
                renamed,
                coefficients[renamed["coefficient_set_ref"]],
                coordinates[renamed["coordinate_contract_ref"]],
            ),
            holdout_manifest_fingerprints[next(iter(fixtures))],
        )
        known_records = self.plan["prior_development_evidence"]["known_state_records"]
        known_by_fingerprint: dict[str, set[str]] = {}
        for record in known_records:
            observed = _known_state_fingerprint(record)
            self.assertEqual(observed, record["scientific_fingerprint_sha256"])
            known_by_fingerprint.setdefault(observed, set()).add(record["record_id"])
        known_fingerprints = set(known_by_fingerprint)
        self.assertEqual(
            {
                frozenset(record_ids)
                for record_ids in known_by_fingerprint.values()
                if len(record_ids) > 1
            },
            {
                frozenset(
                    {
                        "development.state.solar_locked_component",
                        "development.state.reference_force_composition",
                    }
                )
            },
        )
        self.assertEqual(
            sum(len(record_ids) for record_ids in known_by_fingerprint.values()),
            len(EXPECTED_KNOWN_STATE_RECORD_IDS),
        )
        family_fingerprints = {
            member["scientific_fingerprint_sha256"]
            for member in self.prior_family_fingerprints["members"]
        }
        known_transform_fingerprints = set(
            qualification_module._prior_transform_invariant_fingerprints(
                self.prior_manifest,
                project_root=PROJECT_ROOT,
            )
        )
        self.assertEqual(len(known_fingerprints | family_fingerprints), 4121)
        self.assertTrue(
            set(holdout_manifest_fingerprints.values()).isdisjoint(
                known_fingerprints | family_fingerprints
            )
        )
        self.assertTrue(
            set(holdout_fingerprints.values()).isdisjoint(
                known_transform_fingerprints
            )
        )
        prior_numeric_fingerprints = {
            record["state_only_numeric_fingerprint_sha256"]
            for record in self.prior_manifest["generic_rhs_records"]
        }
        self.assertEqual(len(prior_numeric_fingerprints), 5)
        holdout_numeric_fingerprints = {
            _holdout_numeric_fingerprint(
                fixture,
                coordinates[fixture["coordinate_contract_ref"]],
            )
            for fixture in fixtures.values()
        }
        self.assertTrue(
            holdout_numeric_fingerprints.isdisjoint(prior_numeric_fingerprints)
        )
        self.assertEqual(
            self.registration["known_prior_evidence"]["known_state_fingerprints"],
            sorted(known_fingerprints),
        )

        with _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            fixture = next(
                record
                for record in inputs["fixtures"]
                if record["fixture_id"]
                == "fixture.holdout.tdb_km_s_transform"
            )
            component_index = next(
                index
                for index, value in enumerate(fixture["position"])
                if Decimal(value) != 0
            )
            original_component = Decimal(fixture["position"][component_index])
            last_place = Decimal(1).scaleb(original_component.as_tuple().exponent)
            fixture["position"][component_index] = format(
                original_component + last_place,
                "f",
            )
            fixture["expected_state_record_sha256"] = qualification_fixture_sha256(
                fixture
            )
            _rewrite_inputs(root, inputs)
            self.assertQualificationError("derivation_result_binding", root)

        value_level_collision = next(
            iter(holdout_manifest_fingerprints.values())
        )
        with mock.patch(
            "jxplanetx.v5_solar_1pn_qualification."
            "validate_prior_development_manifest_semantics",
            return_value=(
                frozenset(
                    known_fingerprints
                    | family_fingerprints
                    | {value_level_collision}
                ),
                frozenset(prior_numeric_fingerprints),
            ),
        ):
            self.assertQualificationError("holdout_state_reuse", PROJECT_ROOT)

        transform_value_level_collision = next(iter(holdout_fingerprints.values()))
        with mock.patch(
            "jxplanetx.v5_solar_1pn_qualification."
            "_prior_transform_invariant_fingerprints",
            return_value=frozenset(
                known_transform_fingerprints
                | {transform_value_level_collision}
            ),
        ):
            self.assertQualificationError("holdout_state_reuse", PROJECT_ROOT)

        numeric_value_level_collision = next(iter(holdout_numeric_fingerprints))
        with mock.patch(
            "jxplanetx.v5_solar_1pn_qualification."
            "validate_prior_development_manifest_semantics",
            return_value=(
                frozenset(known_fingerprints | family_fingerprints),
                frozenset(
                    prior_numeric_fingerprints
                    | {numeric_value_level_collision}
                ),
            ),
        ):
            self.assertQualificationError(
                "holdout_numeric_state_reuse",
                PROJECT_ROOT,
            )

    def test_transform_fingerprint_equivalence_classes_are_order_independent_and_closed(self) -> None:
        with _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            twin_ids = {
                "fixture.holdout.tdb_km_s_transform",
                "fixture.holdout.tdb_au_day_transform",
                "fixture.holdout.tcb_km_s_transform",
            }
            twins = [
                fixture for fixture in inputs["fixtures"]
                if fixture["fixture_id"] in twin_ids
            ]
            self.assertEqual({fixture["fixture_id"] for fixture in twins}, twin_ids)
            rotated_twins = twins[1:] + twins[:1]
            rotated = iter(rotated_twins)
            inputs["fixtures"] = [
                next(rotated) if fixture["fixture_id"] in twin_ids else fixture
                for fixture in inputs["fixtures"]
            ]
            _rewrite_inputs(root, inputs)
            _, _, _, inspection = inspect_solar_1pn_qualification_package(
                REGISTRATION_RELATIVE,
                project_root=root,
            )
            self.assertFalse(inspection.ready_for_holdout_execution)

        duplicate_cases = (
            (True, "derived_case_role"),
            (False, "duplicate_holdout_state"),
        )
        for transform_labeled, expected_code in duplicate_cases:
            with self.subTest(
                unconnected_transform_label=transform_labeled
            ), _cloned_project() as root:
                inputs = _read_json(root / INPUTS_RELATIVE)
                declared_pair = inputs["q7_transform_contract"]["transform_pairs"][0]
                source = next(
                    record
                    for record in inputs["fixtures"]
                    if record["fixture_id"] == declared_pair["source_fixture_ref"]
                )
                duplicate = copy.deepcopy(source)
                duplicate["fixture_id"] = (
                    "fixture.holdout.unconnected_transform_twin"
                )
                duplicate["target"] = "UNCONNECTED_RENAMED_TARGET"
                duplicate["case_roles"] = copy.deepcopy(source["case_roles"])
                if not transform_labeled:
                    duplicate["case_roles"].remove("TRANSFORM_TWIN")
                duplicate["expected_state_record_sha256"] = (
                    qualification_fixture_sha256(duplicate)
                )
                inputs["fixtures"].append(duplicate)

                source_schedule = next(
                    record
                    for record in inputs["integration_schedules"]
                    if record["fixture_ref"] == source["fixture_id"]
                )
                duplicate_schedule = copy.deepcopy(source_schedule)
                duplicate_schedule["schedule_id"] = (
                    "schedule.unconnected_transform_twin"
                )
                duplicate_schedule["fixture_ref"] = duplicate["fixture_id"]
                inputs["integration_schedules"].append(duplicate_schedule)
                inputs["q8_holdout_custody"]["holdout_fixture_refs"].append(
                    duplicate["fixture_id"]
                )
                _rewrite_inputs(root, inputs)

                plan = _read_json(root / PLAN_RELATIVE)
                arm_id = (
                    "arm.q7_transform"
                    if transform_labeled
                    else "arm.q8_holdout"
                )
                next(
                    arm
                    for arm in plan["test_arms"]
                    if arm["arm_id"] == arm_id
                )["fixture_refs"].append(duplicate["fixture_id"])
                _rewrite_plan(root, plan)
                self.assertQualificationError(expected_code, root)

    def test_registry_and_solar_row_remain_blocked_after_any_future_pass(self) -> None:
        registry_path = PROJECT_ROOT / self.plan["bindings"]["registry"]["path"]
        registry, inspection = inspect_registry_file(
            registry_path,
            project_root=PROJECT_ROOT,
        )
        solar = {
            model["model_id"]: model for model in registry["force_models"]
        }["relativity.solar_schwarzschild_test_particle_1pn"]
        self.assertEqual(registry["state"], "DRAFT_NONEXECUTABLE")
        self.assertFalse(inspection.execution_authorized)
        self.assertEqual(solar["implementation_status"], "NOT_IMPLEMENTED")
        self.assertEqual(solar["qualification_status"], "UNQUALIFIED")
        self.assertEqual(solar["treatment"], "BLOCKED")
        self.assertEqual(solar["qualification_refs"], [])
        self.assertEqual(
            self.plan["verdict_policy"]["registry_status_after_pass"],
            "UNCHANGED_DRAFT_NONEXECUTABLE_UNQUALIFIED",
        )

    def test_duplicate_json_key_is_rejected_before_schema_validation(self) -> None:
        with _cloned_project() as root:
            path = root / REGISTRATION_RELATIVE
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("{", '{\n  "schema": "duplicate",', 1),
                encoding="utf-8",
            )
            self.assertQualificationError("duplicate_json_key", root)

    def test_binary_float_and_unknown_field_are_rejected(self) -> None:
        with _cloned_project() as root:
            inputs = _read_json(root / INPUTS_RELATIVE)
            inputs["coefficient_sets"][0]["speed_of_light"]["value"] = 1.5
            _rewrite_inputs(root, inputs)
            self.assertQualificationError("binary_float_forbidden", root)

        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["silent_default"] = False
            _write_json(registration_path, registration)
            self.assertQualificationError("schema_unknown_field", root)

    def test_symlink_and_path_escape_are_rejected(self) -> None:
        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            link = root / "runs" / "v5_solar_1pn_qualification" / "plan-link.json"
            link.symlink_to(root / PLAN_RELATIVE)
            registration["plan"]["path"] = link.relative_to(root).as_posix()
            _write_json(registration_path, registration)
            self.assertQualificationError("symlink_forbidden", root)

        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["plan"]["path"] = "../qualification_plan_v1.json"
            _write_json(registration_path, registration)
            self.assertQualificationError("schema_pattern", root)

    def test_input_source_plan_and_schema_byte_tampering_are_rejected(self) -> None:
        targets = (
            INPUTS_RELATIVE,
            PLAN_RELATIVE,
            PRIOR_MANIFEST_RELATIVE,
            Path(
                self.prior_manifest["generated_state_families"][0][
                    "fingerprint_list"
                ]["path"]
            ),
            Path(self.inputs["provenance_sources"][0]["path"]),
            Path(EXPECTED_SCHEMA_PATHS[PLAN_SCHEMA]),
            Path(EXPECTED_SCHEMA_PATHS[PRIOR_DEVELOPMENT_SCHEMA]),
        )
        for target in targets:
            with self.subTest(target=target), _cloned_project() as root:
                path = root / target
                path.write_bytes(path.read_bytes() + b"\n")
                self.assertQualificationError("artifact_hash_mismatch", root)

    def test_raw_size_and_canonical_hash_binding_tampering_are_rejected(self) -> None:
        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["plan"]["size_bytes"] = str(
                int(registration["plan"]["size_bytes"]) + 1
            )
            _write_json(registration_path, registration)
            self.assertQualificationError("artifact_size_mismatch", root)

        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration = _read_json(registration_path)
            registration["plan"]["canonical_sha256"] = "0" * 64
            _write_json(registration_path, registration)
            self.assertQualificationError("canonical_digest_mismatch", root)

    def test_package_digest_is_sensitive_to_raw_registration_bytes(self) -> None:
        with _cloned_project() as root:
            registration_path = root / REGISTRATION_RELATIVE
            registration_path.write_bytes(registration_path.read_bytes() + b"\n")
            _, _, registration, inspection = inspect_solar_1pn_qualification_package(
                REGISTRATION_RELATIVE,
                project_root=root,
            )
        self.assertEqual(registration, self.registration)
        self.assertEqual(
            inspection.registration_canonical_sha256,
            self.inspection.registration_canonical_sha256,
        )
        self.assertNotEqual(
            inspection.registration_file_sha256,
            self.inspection.registration_file_sha256,
        )
        self.assertNotEqual(
            inspection.registration_size_bytes,
            self.inspection.registration_size_bytes,
        )
        self.assertNotEqual(inspection.package_sha256, self.inspection.package_sha256)

    def test_known_fixture_roster_mutation_is_rejected_by_frozen_disclosure(self) -> None:
        with _cloned_project() as root:
            plan = _read_json(root / PLAN_RELATIVE)
            plan["prior_development_evidence"]["known_fixture_ids"].append(
                next(iter(sorted(EXPECTED_FIXTURE_IDS)))
            )
            _rewrite_plan(root, plan)
            self.assertQualificationError("prior_manifest_disclosure", root)

    def test_asymmetric_arm_gate_reference_is_rejected_after_rebinding(self) -> None:
        with _cloned_project() as root:
            plan = _read_json(root / PLAN_RELATIVE)
            expected_arm, expected_gate = EXPECTED_ARM_GATE_MATRIX[0]
            replacement_arm, _ = EXPECTED_ARM_GATE_MATRIX[1]
            gate = next(
                record for record in plan["gates"]
                if record["gate_id"] == expected_gate
            )
            self.assertEqual(gate["arm_refs"], [expected_arm])
            gate["arm_refs"] = [replacement_arm]
            _rewrite_plan(root, plan)
            self.assertQualificationError("asymmetric_gate_reference", root)

    def test_outcome_execution_and_authorization_mutations_are_rejected(self) -> None:
        mutations = (
            ("outcomes_generated", True),
            ("execution_implementation_state", "REGISTERED"),
            ("registry_authorized", True),
        )
        for key, value in mutations:
            with self.subTest(key=key), _cloned_project() as root:
                registration_path = root / REGISTRATION_RELATIVE
                registration = _read_json(registration_path)
                registration[key] = value
                _write_json(registration_path, registration)
                self.assertQualificationError("schema_const", root)

    def test_missing_execution_registration_keeps_package_blocked(self) -> None:
        policy = self.plan["execution_registration_policy"]
        self.assertTrue(policy["required"])
        self.assertTrue(policy["outcomes_prohibited_until_registered"])
        self.assertTrue(policy["required_bindings"])
        self.assertEqual(policy["current_state"], "NOT_REGISTERED")
        self.assertFalse(self.registration["outcomes_generated"])
        self.assertFalse(self.inspection.ready_for_holdout_execution)

        with _cloned_project() as root:
            plan = _read_json(root / PLAN_RELATIVE)
            del plan["execution_registration_policy"]
            _rewrite_plan(root, plan)
            self.assertQualificationError("schema_required", root)

    def test_all_qualification_schema_objects_are_fail_closed(self) -> None:
        for relative in EXPECTED_SCHEMA_PATHS.values():
            schema = _read_json(PROJECT_ROOT / relative)
            self.assertIs(schema["additionalProperties"], False)
            for name, definition in schema["$defs"].items():
                if definition.get("type") == "object":
                    self.assertIs(
                        definition.get("additionalProperties"),
                        False,
                        f"{relative} definition {name} is not fail-closed",
                    )


if __name__ == "__main__":
    unittest.main()
