"""Fail-closed inspection of the frozen V5 Solar 1PN qualification package.

This module validates a pre-execution plan, its frozen inputs, and the
registration that binds their exact bytes.  It deliberately exposes no
trajectory runner and never converts the draft V5 registry into execution
authority.  A valid inspection means only that the holdout package is intact
and still awaits a separately hash-locked execution implementation.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .force_registry_v5 import (
    ForceRegistryError,
    inspect_registry_file,
    validate_against_bundled_schema,
)
from .provenance import canonical_json, sha256_data, sha256_file
from .solar_1pn import (
    AccelerationSemantics,
    CentralSourceModel,
    CoherentUnitSystem,
    CoordinateTimeScale,
    InertialFrame,
    SolarSchwarzschild1PNContract,
    TargetTreatment,
)
from .v5_implicit_midpoint import ImplicitMidpointSpec
from .v5_reference_dynamics import (
    DecimalContextSpec,
    inspect_reference_force_plan,
)


PLAN_SCHEMA = "jx-v5-solar-1pn-qualification-plan/v1"
INPUT_SCHEMA = "jx-v5-solar-1pn-qualification-inputs/v1"
REGISTRATION_SCHEMA = "jx-v5-solar-1pn-qualification-registration/v1"
PRIOR_DEVELOPMENT_SCHEMA = "jx-v5-solar-1pn-prior-development-cases/v1"
INSPECTION_SCHEMA = "jx-v5-solar-1pn-qualification-inspection/v1"

PLAN_STATE = (
    "PRELOCKED_AFTER_DISCLOSED_DEVELOPMENT_TESTS_BEFORE_HOLDOUT_OUTCOMES"
)
INPUT_STATE = "FROZEN_NONAUTHORIZING_INPUTS"
REGISTRATION_STATE = "REGISTERED_BEFORE_HOLDOUT_OUTCOMES"
EXECUTION_IMPLEMENTATION_STATE = "NOT_REGISTERED"
PASS_EFFECT = "ELIGIBLE_FOR_REVIEW_NONAUTHORIZING_ONLY"
ALLOWED_VERDICTS = (
    "NOT_RUN",
    "BLOCKED",
    "INVALID",
    "FAIL",
    "CONFLICT",
    "ELIGIBLE_FOR_REVIEW_NONAUTHORIZING",
)
FOUNDATION_COMMIT = "58b57094d519205f3ef398f3c738af96107c59e5"
QUALIFICATION_ID_PREFIX = "jx.v5.solar_1pn.qualification."
QUALIFICATION_ID_SENTINEL = "jx.v5.solar_1pn.qualification.CONTENT_DIGEST_SENTINEL"
NONCLAIM = (
    "This pre-execution package is MODEL_OUTPUT only and grants no registry "
    "or scientific-propagation authority."
)
PRIOR_DEVELOPMENT_NONCLAIM = (
    "This manifest discloses prior development evidence only. It contains no "
    "qualification outcome, grants no registry authority, and every listed case "
    "is excluded from decisive Q0-Q9 evidence."
)
PRIOR_DEVELOPMENT_MANIFEST_PATH = (
    "runs/v5_solar_1pn_qualification/prior_development_cases_v1.json"
)
PRIOR_DEVELOPMENT_MANIFEST_ID = (
    "jx.v5.solar_1pn.prior_development_cases.v1"
)
PRIOR_DEVELOPMENT_MANIFEST_STATE = (
    "FROZEN_DISCLOSED_PRIOR_DEVELOPMENT_CASES_NONAUTHORIZING"
)
PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256 = (
    "c7897e82143cf155933910103d3dd9cca3166b2033a36bbd0d72c3109fe3c8ce"
)
PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES = "132803"
PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256 = (
    "3444af9afbc39556148a7757e4e5cbf1a99d7be14335b126d6378b8ecae25140"
)
PRIOR_DEVELOPMENT_DISCLOSURE_SHA256 = (
    "7797cebb12c85ccc8c950a39dc1ed2e40cb3d59468a4aed9f6676489c55c2975"
)
PRIOR_FINGERPRINT_LIST_PATH = (
    "runs/v5_solar_1pn_qualification/prior/"
    "equation_level_perihelion_4096_fingerprints_v1.json"
)
PRIOR_FINGERPRINT_LIST_FILE_SHA256 = (
    "20d23a93c426baa3818c05f5a57616edd7844905ef805d4ad87fb7948f3cd205"
)
PRIOR_FINGERPRINT_LIST_SIZE_BYTES = "1762116"
PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256 = (
    "c4f1c03ff7aa9a1127caa2c8924ffb638cc84229191961449a2239378a45c1d2"
)
PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256 = (
    "afebc490ed48337727210b857b4a1491c6647ede071439ec99071421c0a52dd1"
)
PRIOR_FIRST_MEMBER_FINGERPRINT_SHA256 = (
    "09a1ab95feeb6fe4750202e351a8ed8157730c632a1d252e25a2dabb71cce88d"
)
PRIOR_LAST_MEMBER_FINGERPRINT_SHA256 = (
    "d37787b6b6c2ea49e08375aa395a3be1abe0867939366cb7ba2468092afce6c7"
)
PRIOR_DEVELOPMENT_COUNTS = {
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
PRIOR_FOUNDATION_SOURCE_PATHS = {
    "development_solar_1pn_tests": (
        "runs/v5_solar_1pn_qualification/prior/foundation_test_solar_1pn.py",
        "tests/test_solar_1pn.py",
    ),
    "development_reference_integrator_tests": (
        "runs/v5_solar_1pn_qualification/prior/"
        "foundation_test_v5_reference_integrator.py",
        "tests/test_v5_reference_integrator.py",
    ),
}
PERMITTED_CLAIM_CODE = (
    "FROZEN_RESTRICTED_SOLAR_1PN_REFERENCE_REPRODUCTION_"
    "ELIGIBLE_FOR_EXTERNAL_REVIEW_NONAUTHORIZING"
)
PERMITTED_CLAIM_TEXT = (
    "For the frozen cases and declared domain in qualification package "
    "<qualification_id>, the nonauthorizing JX reference path reproduced the declared "
    "restricted static-Sun, massless-target Schwarzschild 1PN equation within "
    "the preregistered numerical and model-matched error budgets. The resulting "
    "trajectories are MODEL_OUTPUT and are eligible only for external scientific "
    "review."
)
PROHIBITED_CLAIM_CODES = (
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
ARM_GATE_MATRIX = (
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
ARM_IDS = tuple(arm for arm, _ in ARM_GATE_MATRIX)
GATE_IDS = tuple(gate for _, gate in ARM_GATE_MATRIX)
BLOCKED_REASONS = (
    "EXECUTION_IMPLEMENTATION_NOT_REGISTERED",
    "Q0_Q9_NOT_EXECUTED",
    "PROVENANCE_UNRESOLVED",
    "INDEPENDENT_ORACLE_UNRESOLVED",
    "EIH_EVALUATOR_UNRESOLVED",
    "ERROR_BUDGETS_UNRESOLVED",
    "HOLDOUT_CUSTODY_UNRESOLVED",
)
KNOWN_FILE_ROLE_PATHS = {
    "development_solar_1pn_tests": "tests/test_solar_1pn.py",
    "development_reference_integrator_tests": (
        "tests/test_v5_reference_integrator.py"
    ),
}
LOCKED_FILE_ROLE_PATHS = {
    "qualification_protocol": "runs/v5_solar_1pn_qualification/README.md",
    "solar_1pn_kernel": "src/jxplanetx/solar_1pn.py",
    "reference_force_composition": "src/jxplanetx/v5_reference_dynamics.py",
    "implicit_midpoint_integrator": "src/jxplanetx/v5_implicit_midpoint.py",
    "qualification_validator": "src/jxplanetx/v5_solar_1pn_qualification.py",
    "force_registry_validator": "src/jxplanetx/force_registry_v5.py",
    "provenance_hashing": "src/jxplanetx/provenance.py",
    "decimal_vector_math": "src/jxplanetx/decimal_math.py",
}
PROVENANCE_SOURCE_PATHS = {
    "source.bipm.si_brochure_9_v4_01": (
        "runs/v5_solar_1pn_qualification/sources/bipm_si_brochure_9_v4_01.pdf"
    ),
    "source.iau.resolution_b2_2012": (
        "runs/v5_solar_1pn_qualification/sources/iau_2012_resolutions_en.pdf"
    ),
    "source.iau.resolution_b2_2006": (
        "runs/v5_solar_1pn_qualification/sources/iau_2006_resolution_b2.pdf"
    ),
    "source.iau.resolution_b3_2006": (
        "runs/v5_solar_1pn_qualification/sources/iau_2006_resolution_b3.pdf"
    ),
    "source.iers.tn36.chapter10": (
        "runs/v5_solar_1pn_qualification/sources/iers_tn36_chapter_10.pdf"
    ),
    "source.jpl.astrodynamic_parameters.de440": (
        "runs/v5_solar_1pn_qualification/sources/"
        "jpl_astrodynamic_parameters_2026-08-29.html"
    ),
    "source.jpl.de440.export_info": (
        "runs/v5_solar_1pn_qualification/sources/"
        "jpl_de440_export_2026-08-29.html"
    ),
    "source.naif.gm_de440": (
        "runs/v5_solar_1pn_qualification/sources/gm_de440.tpc"
    ),
}
OBSERVABLE_IDS = (
    "observable.q1.acceleration",
    "observable.q2.perihelion_coefficient",
    "observable.q3.total_state",
    "observable.q3.differential_signal",
    "observable.q4.eih_restricted_limit",
    "observable.q5.order_solver",
    "observable.q6.precision_plateau",
    "observable.q7.transform",
)
CASE_ROLES = (
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
_ALL_OBSERVABLES = OBSERVABLE_IDS
ARM_SPECS = {
    "arm.q0_integrity": {
        "purpose_code": "PACKAGE_INTEGRITY_AND_PROVENANCE",
        "purpose": "Verify package bytes, schemas, provenance, frozen identities, and unresolved blockers before any execution.",
        "implementation_independence": "PROVENANCE_ONLY",
        "required_case_roles": (),
        "observable_refs": (),
        "budget_refs": (),
    },
    "arm.q1_equation": {
        "purpose_code": "EQUATION_LEVEL_DISCRIMINATION",
        "purpose": "Discriminate the declared restricted Solar Schwarzschild 1PN acceleration from preregistered equation mutants.",
        "implementation_independence": "ANALYTIC_OR_EXACT",
        "required_case_roles": ("GENERIC_3D", "DOMAIN_INSIDE", "DOMAIN_BOUNDARY", "DOMAIN_OUTSIDE"),
        "observable_refs": ("observable.q1.acceleration",),
        "budget_refs": ("observable.q1.acceleration",),
    },
    "arm.q2_perihelion": {
        "purpose_code": "PERIHELION_COEFFICIENT_SCALING",
        "purpose": "Recover the analytic perihelion coefficient across the frozen eccentricity, inverse-c-squared, step, and event-tolerance Cartesian matrix.",
        "implementation_independence": "SEPARATELY_CODED_NO_JX_DYNAMICS_IMPORTS",
        "required_case_roles": ("ECCENTRIC", "LONG_ARC"),
        "observable_refs": ("observable.q2.perihelion_coefficient",),
        "budget_refs": ("observable.q2.perihelion_coefficient",),
    },
    "arm.q3_high_order_oracle": {
        "purpose_code": "INDEPENDENT_HIGH_ORDER_ORACLE",
        "purpose": "Compare total state and isolated 1PN differential signal against a separately coded arbitrary-precision high-order oracle with self-convergence.",
        "implementation_independence": "SEPARATELY_CODED_NO_JX_DYNAMICS_IMPORTS",
        "required_case_roles": ("GENERIC_3D", "SIGNAL_SCALE"),
        "observable_refs": ("observable.q3.total_state", "observable.q3.differential_signal"),
        "budget_refs": ("observable.q3.total_state", "observable.q3.differential_signal"),
    },
    "arm.q4_restricted_eih": {
        "purpose_code": "RESTRICTED_EIH_LIMIT",
        "purpose": "Recover the declared static-Sun massless-target 1PN equation from an independent harmonic-gauge EIH evaluator at fixed total mu over the frozen nu ladder.",
        "implementation_independence": "SEPARATELY_CODED_NO_JX_DYNAMICS_IMPORTS",
        "required_case_roles": ("EIH_LIMIT",),
        "observable_refs": ("observable.q4.eih_restricted_limit",),
        "budget_refs": ("observable.q4.eih_restricted_limit",),
    },
    "arm.q5_step_solver": {
        "purpose_code": "STEP_AND_SOLVER_CONVERGENCE",
        "purpose": "Demonstrate frozen common-endpoint second-order step convergence and solver-tolerance independence.",
        "implementation_independence": "JX_REFERENCE_IMPLEMENTATION",
        "required_case_roles": ("ECCENTRIC", "LONG_ARC"),
        "observable_refs": ("observable.q5.order_solver",),
        "budget_refs": ("observable.q5.order_solver",),
    },
    "arm.q6_precision": {
        "purpose_code": "DECIMAL_PRECISION_CROSS_GRID",
        "purpose": "Separate step error from Decimal precision error on the complete frozen h-by-precision Cartesian grid.",
        "implementation_independence": "JX_REFERENCE_IMPLEMENTATION",
        "required_case_roles": ("SIGNAL_SCALE",),
        "observable_refs": ("observable.q6.precision_plateau",),
        "budget_refs": ("observable.q6.precision_plateau",),
    },
    "arm.q7_transform": {
        "purpose_code": "UNIT_AND_TIME_SCALE_COVARIANCE",
        "purpose": "Verify exact coherent-unit changes and the retained L_B-derived TCB-to-TDB compatible affine scaling, including accelerations and tolerances.",
        "implementation_independence": "ANALYTIC_OR_EXACT",
        "required_case_roles": ("TRANSFORM_TWIN",),
        "observable_refs": ("observable.q7.transform",),
        "budget_refs": ("observable.q7.transform",),
    },
    "arm.q8_holdout": {
        "purpose_code": "FIRST_VALID_UNBLINDING",
        "purpose": "Enforce sealed expectations, complete scientific-case custody, and the first-valid-unblinding-only policy.",
        "implementation_independence": "PROVENANCE_ONLY",
        "required_case_roles": CASE_ROLES,
        "observable_refs": _ALL_OBSERVABLES,
        "budget_refs": _ALL_OBSERVABLES,
    },
    "arm.q9_claim_audit": {
        "purpose_code": "CLAIM_AND_ERROR_BUDGET_AUDIT",
        "purpose": "Require all component and total error-budget gates and enforce the exact nonauthorizing claim ceiling.",
        "implementation_independence": "PROVENANCE_ONLY",
        "required_case_roles": CASE_ROLES,
        "observable_refs": _ALL_OBSERVABLES,
        "budget_refs": _ALL_OBSERVABLES,
    },
}
GATE_SPECS = {
    gate_id: {
        "metric_code": metric_code,
        "metric": metric,
        "value_kind": "BOOLEAN",
        "operator": "EQUAL",
        "threshold": "true",
        "unit": "unit.dimensionless",
        "failure_verdict": "FAIL",
        "observable_refs": ARM_SPECS[arm_id]["observable_refs"],
        "budget_refs": ARM_SPECS[arm_id]["budget_refs"],
    }
    for arm_id, gate_id, metric_code, metric in (
        ("arm.q0_integrity", "gate.q0_package_integrity", "PACKAGE_INTEGRITY_COMPLETE", "All package, schema, provenance, identity, and blocker checks are satisfied."),
        ("arm.q1_equation", "gate.q1_equation_discrimination", "ALL_EQUATION_MUTANTS_REJECTED", "All preregistered equation mutants are rejected and exact analytic identities pass."),
        ("arm.q2_perihelion", "gate.q2_perihelion_coefficient", "PERIHELION_MATRIX_WITHIN_BUDGET", "Every frozen paired perihelion case cell meets its coefficient error budget."),
        ("arm.q3_high_order_oracle", "gate.q3_high_order_oracle", "ORACLE_MATRIX_WITHIN_BUDGET", "Every frozen total-state and differential-signal oracle cell meets its budget after oracle self-convergence."),
        ("arm.q4_restricted_eih", "gate.q4_restricted_eih_limit", "EIH_LIMIT_WITHIN_BUDGET", "The fixed-total-mu nu ladder recovers the declared restricted 1PN equation within budget."),
        ("arm.q5_step_solver", "gate.q5_step_solver_convergence", "STEP_SOLVER_CONVERGENCE_ACCEPTED", "The frozen common-endpoint step and solver-tightening diagnostics satisfy the declared acceptance rule."),
        ("arm.q6_precision", "gate.q6_decimal_precision", "PRECISION_CROSS_GRID_ACCEPTED", "The complete h-by-precision grid demonstrates the declared Decimal precision plateau."),
        ("arm.q7_transform", "gate.q7_unit_time_transform", "ALL_TRANSFORM_PAIRS_COVARIANT", "Both exact-unit and TCB-to-TDB compatible transform pairs satisfy state, acceleration, and tolerance covariance."),
        ("arm.q8_holdout", "gate.q8_first_unblinding", "FIRST_VALID_UNBLINDING_COMPLIANT", "Custody, sealed expectations, and the first valid unblinding record cover every required scientific case role."),
        ("arm.q9_claim_audit", "gate.q9_claim_audit", "ERROR_BUDGET_AND_CLAIM_CEILING_COMPLIANT", "All nine-component and total error-budget gates pass and only the exact permitted nonauthorizing claim is emitted."),
    )
}
KNOWN_STATE_RECORD_SPECS = {
    "development.state.solar_fraction_oracle": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_exact_fixture_matches_independently_coded_fraction_oracle"),
    "development.state.solar_locked_component": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_locked_sign_and_component_fixture_is_correction_only"),
    "development.state.solar_zero_velocity": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_zero_velocity_is_radially_outward"),
    "development.state.solar_radial_inward": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_radial_motion_generates_tangential_term_with_expected_sign:inward"),
    "development.state.solar_circular": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_circular_orbit_analytic_component"),
    "development.state.solar_rotation": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_rotation_covariance:original"),
    "development.state.solar_rotation_twin": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_rotation_covariance:rotated"),
    "development.state.solar_unit_transform_base": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_coherent_length_time_rescaling_and_inverse_c_squared:base"),
    "development.state.solar_unit_transform_km_s": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_coherent_length_time_rescaling_and_inverse_c_squared:kilometre_second"),
    "development.state.solar_inverse_c_squared_c200": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_coherent_length_time_rescaling_and_inverse_c_squared:doubled_c"),
    "development.state.solar_large_c": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_large_finite_c_limit_recovers_newtonian_total"),
    "development.state.solar_physical_scale": ("development_solar_1pn_tests", "SolarSchwarzschild1PNEquationTests.test_solar_system_fixture_is_small_relative_to_newtonian"),
    "development.state.solar_zero_separation": ("development_solar_1pn_tests", "SolarSchwarzschild1PNFailureTests.test_rejects_duplicate_application_before_evaluation:zero_separation"),
    "development.state.solar_domain_compactness_excess": ("development_solar_1pn_tests", "SolarSchwarzschild1PNFailureTests.test_rejects_states_outside_declared_1pn_bounds:compactness"),
    "development.state.solar_domain_speed_excess": ("development_solar_1pn_tests", "SolarSchwarzschild1PNFailureTests.test_rejects_states_outside_declared_1pn_bounds:speed"),
    "development.state.reference_force_composition": ("development_reference_integrator_tests", "ReferenceForceCompositionTests.test_exact_newtonian_plus_1pn_fixture_and_ledger"),
    "development.state.reference_default_p60": ("development_reference_integrator_tests", "SolarReferenceTrajectoryTests.test_short_solar_arc_converges_at_second_order:initial"),
    "development.state.reference_default_p70": ("development_reference_integrator_tests", "ImplicitMidpointMethodTests.test_forward_negative_step_roundtrip_is_solver_tolerance_limited:initial"),
    "development.state.reference_large_epoch_p34": ("development_reference_integrator_tests", "ImplicitMidpointMethodTests.test_nonzero_step_must_advance_large_epoch_exactly:initial"),
    "development.state.reference_midpoint_domain_candidate": ("development_reference_integrator_tests", "ReferenceForceCompositionTests.test_midpoint_domain_failure_returns_no_candidate"),
    "development.state.reference_domain_boundary": ("development_reference_integrator_tests", "ReferenceForceCompositionTests.test_declared_domain_threshold_is_inclusive_and_excess_is_rejected:boundary"),
    "development.state.reference_domain_compactness_excess": ("development_reference_integrator_tests", "ReferenceForceCompositionTests.test_declared_domain_threshold_is_inclusive_and_excess_is_rejected:compactness_excess"),
    "development.state.reference_domain_speed_excess": ("development_reference_integrator_tests", "ReferenceForceCompositionTests.test_declared_domain_threshold_is_inclusive_and_excess_is_rejected:speed_excess"),
    "development.state.reference_large_c_p70": ("development_reference_integrator_tests", "SolarReferenceTrajectoryTests.test_large_c_trajectory_recovers_independent_newtonian_midpoint:initial"),
    "development.state.reference_default_p60_km_s": ("development_reference_integrator_tests", "SolarReferenceTrajectoryTests.test_coherent_au_day_and_kilometre_second_trajectories_match:converted_initial"),
    "development.state.reference_default_p60_rotated": ("development_reference_integrator_tests", "SolarReferenceTrajectoryTests.test_rotation_covariance_for_signed_axis_permutation:rotated_initial"),
}
UNIT_DIMENSIONS = {
    "length": "LENGTH",
    "time": "TIME",
    "velocity": "LENGTH/TIME",
    "gravitational_parameter": "LENGTH^3/TIME^2",
    "dimensionless": "DIMENSIONLESS",
}
COORDINATE_UNIT_REF_SPECS = {
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
SOURCE_LITERAL_SPECS = {
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
DERIVATION_IDS = (
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
FORMULA_DIMENSION_SIGNATURES = {
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
        "OPERAND": {name: "DIMENSIONLESS" for name in (
            "p0", "p1", "p2", "q0", "q1", "q2", "n0", "n1", "n2"
        )},
        "RESULT": {name: "DIMENSIONLESS" for name in (
            "p_dot_p", "q_dot_q", "n_dot_n", "p_dot_q", "p_dot_n",
            "q_dot_n", "orientation_determinant"
        )},
    },
}
BLOCKER_SPECS = {
    "blocker.execution_implementation": ("EXECUTION_SOURCE", GATE_IDS),
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

_SCHEMA_PATHS = {
    PLAN_SCHEMA: "schemas/jx-v5-solar-1pn-qualification-plan-v1.schema.json",
    INPUT_SCHEMA: "schemas/jx-v5-solar-1pn-qualification-inputs-v1.schema.json",
    REGISTRATION_SCHEMA: (
        "schemas/jx-v5-solar-1pn-qualification-registration-v1.schema.json"
    ),
    PRIOR_DEVELOPMENT_SCHEMA: (
        "schemas/jx-v5-solar-1pn-prior-development-cases-v1.schema.json"
    ),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_CANONICAL_DECIMAL = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")
_INTEGER_TEXT = re.compile(r"^-?(0|[1-9][0-9]*)$")


class Solar1PNQualificationError(ValueError):
    """Stable, machine-readable qualification-package validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise Solar1PNQualificationError(code, message)


def _reject_json_constant(value: str) -> None:
    _fail("nonfinite_json", f"JSON constant {value!r} is forbidden")


def _reject_json_float(value: str) -> None:
    _fail(
        "binary_float_forbidden",
        f"JSON scientific numbers must be canonical strings, not {value!r}",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except Solar1PNQualificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail("invalid_json", f"cannot load {context} {path}: {exc}")
    if not isinstance(value, dict):
        _fail("invalid_json_root", f"{context} root must be an object")
    return value


def _contains_binary_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_binary_float(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_binary_float(item) for item in value)
    return False


def _reject_binary_float(value: Any, context: str) -> None:
    if _contains_binary_float(value):
        _fail("binary_float_forbidden", f"{context} contains a JSON binary float")


def _normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json(value).decode("utf-8"))


def _require_identifier(value: Any, context: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        _fail("invalid_identifier", f"{context} must be a stable identifier")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail("invalid_sha256", f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_git_commit(value: Any, context: str) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        _fail("invalid_git_commit", f"{context} must be a full lowercase commit ID")
    return value


def _verify_foundation_git_object(root: Path) -> None:
    """Verify the frozen foundation when repository metadata is available."""

    if not (root / ".git").exists():
        return
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{FOUNDATION_COMMIT}^{{commit}}"],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        _fail(
            "foundation_git_verification_unavailable",
            f"repository metadata exists but git verification failed: {exc}",
        )
    if completed.returncode != 0:
        _fail(
            "foundation_git_object_missing",
            f"frozen foundation commit {FOUNDATION_COMMIT} is not present",
        )


def qualification_plan_identity_sha256(plan: Mapping[str, Any]) -> str:
    """Digest a plan after replacing its self-identifying field by a sentinel."""

    _reject_binary_float(plan, "qualification plan identity")
    normalized = _normalized(plan)
    if "qualification_id" not in normalized:
        _fail("plan_identity", "qualification plan has no qualification_id")
    normalized["qualification_id"] = QUALIFICATION_ID_SENTINEL
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-plan-identity/v1",
            "sentinel": QUALIFICATION_ID_SENTINEL,
            "plan": normalized,
        }
    )


_AU_IN_KILOMETRES = Decimal("149597870.7")
_SECONDS_PER_DAY = Decimal("86400")
_IAU_L_B = Decimal("0.00000001550519768")
_TCB_TO_TDB_FACTOR = Decimal(1) - _IAU_L_B
_IAU_T0_JULIAN_DATE = Decimal("2443144.5003725")
_IAU_TDB0_SECONDS = Decimal("-0.0000655")
_FINGERPRINT_CONTEXT_SPEC = DecimalContextSpec(precision=90)
_TAU_CONSERVATIVE_NUMERATOR = Decimal(44)
_TAU_CONSERVATIVE_DENOMINATOR = Decimal(7)


def _plain_decimal(value: Decimal) -> str:
    """Return a canonical non-exponent Decimal spelling for fingerprints."""

    if not value.is_finite():
        _fail("nonfinite_fingerprint", "scientific fingerprint value is nonfinite")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _unit_scales_to_au_day(units: str) -> tuple[Decimal, Decimal]:
    if units == "AU_DAY":
        return Decimal(1), Decimal(1)
    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        if units == "KILOMETRE_SECOND":
            return Decimal(1) / _AU_IN_KILOMETRES, Decimal(1) / _SECONDS_PER_DAY
    _fail("unsupported_fingerprint_units", f"cannot normalize unit system {units!r}")


def _numeric_unit_transform_scales(source_units: str, target_units: str) -> tuple[Decimal, Decimal]:
    physical_scales = {
        "AU_DAY": (_AU_IN_KILOMETRES, _SECONDS_PER_DAY),
        "KILOMETRE_SECOND": (Decimal(1), Decimal(1)),
    }
    try:
        source_length, source_time = physical_scales[source_units]
        target_length, target_time = physical_scales[target_units]
    except KeyError:
        _fail("unsupported_transform_units", "Q7 uses an unsupported unit system")
    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        return source_length / target_length, source_time / target_time


def _exact_unit_transform_value(
    value: Decimal,
    *,
    source_units: str,
    target_units: str,
    quantity: str,
) -> Decimal:
    """Convert a value with one frozen precision-90 arithmetic order."""

    physical_scales = {
        "AU_DAY": (_AU_IN_KILOMETRES, _SECONDS_PER_DAY),
        "KILOMETRE_SECOND": (Decimal(1), Decimal(1)),
    }
    try:
        source_length, source_time = physical_scales[source_units]
        target_length, target_time = physical_scales[target_units]
    except KeyError:
        _fail("unsupported_transform_units", "Q7 uses an unsupported unit system")
    if quantity == "LENGTH":
        return value * source_length / target_length
    if quantity == "TIME":
        return value * source_time / target_time
    if quantity == "VELOCITY":
        return value * source_length * target_time / source_time / target_length
    if quantity == "GRAVITATIONAL_PARAMETER":
        return (
            value
            * source_length ** 3
            * target_time ** 2
            / source_time ** 2
            / target_length ** 3
        )
    if quantity == "ACCELERATION":
        return (
            value
            * source_length
            * target_time ** 2
            / source_time ** 2
            / target_length
        )
    _fail("unsupported_transform_quantity", f"unknown exact-unit quantity {quantity!r}")


def _tcb_to_tdb_declared_epoch(
    source_epoch: Decimal,
    *,
    units: str,
    factor: Decimal,
    l_b: Decimal,
    t0_julian_date: Decimal,
    tdb0_seconds: Decimal,
) -> Decimal:
    """Apply the IAU affine epoch relation in the contract's time unit."""

    _, time_to_day = _unit_scales_to_au_day(units)
    affine_days = l_b * t0_julian_date + tdb0_seconds / _SECONDS_PER_DAY
    return factor * source_epoch + affine_days / time_to_day


def _normalized_scientific_payload(
    *,
    epoch: Any,
    position: Sequence[Any],
    velocity: Sequence[Any],
    coefficient_set: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize exact unit and TCB/TDB aliases to one AU/day/TCB payload.

    Display identifiers, including target labels and unit IDs, are deliberately
    absent.  The retained IAU constants make exact AU/day<->km/s and
    TCB-compatible<->TDB-compatible twins collide by scientific value.
    """

    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        length_to_au, time_to_day = _unit_scales_to_au_day(
            coordinate_contract["units"]
        )
        velocity_to_au_day = length_to_au / time_to_day
        mu_to_au3_day2 = length_to_au ** 3 / time_to_day ** 2
        normalized_epoch = _canonical_decimal(epoch, "fingerprint epoch") * time_to_day
        normalized_position = [
            _canonical_decimal(value, "fingerprint position") * length_to_au
            for value in position
        ]
        normalized_velocity = [
            _canonical_decimal(value, "fingerprint velocity") * velocity_to_au_day
            for value in velocity
        ]
        normalized_mu = _canonical_decimal(
            coefficient_set["gravitational_parameter"]["value"],
            "fingerprint gravitational parameter",
        ) * mu_to_au3_day2
        normalized_c = _canonical_decimal(
            coefficient_set["speed_of_light"]["value"],
            "fingerprint speed of light",
        ) * velocity_to_au_day
        if coordinate_contract["time_scale"] == "TDB_COMPATIBLE":
            normalized_position = [
                value / _TCB_TO_TDB_FACTOR for value in normalized_position
            ]
            normalized_mu /= _TCB_TO_TDB_FACTOR
            if coordinate_contract["epoch_kind"] == "DECLARED_COORDINATE_EPOCH":
                normalized_epoch = (
                    normalized_epoch
                    - _IAU_L_B * _IAU_T0_JULIAN_DATE
                    - _IAU_TDB0_SECONDS / _SECONDS_PER_DAY
                ) / _TCB_TO_TDB_FACTOR
            else:
                normalized_epoch /= _TCB_TO_TDB_FACTOR
        elif coordinate_contract["time_scale"] != "TCB_COMPATIBLE":
            _fail(
                "unsupported_fingerprint_time_scale",
                f"cannot normalize time scale {coordinate_contract['time_scale']!r}",
            )
        invariant_coordinate_fields = (
            "frame",
            "origin",
            "orientation",
            "epoch_kind",
            "central_source_model",
            "target_treatment",
            "output_semantics",
        )
        return {
            "normal_form": "AU_DAY_TCB_COMPATIBLE_V1",
            "epoch": _plain_decimal(+normalized_epoch),
            "position": [_plain_decimal(+value) for value in normalized_position],
            "velocity": [_plain_decimal(+value) for value in normalized_velocity],
            "coordinate_contract": {
                **{
                    field: coordinate_contract[field]
                    for field in invariant_coordinate_fields
                },
                "units": "AU_DAY",
                "time_scale": "TCB_COMPATIBLE",
            },
            "coefficient_contract": {
                "central_source": coefficient_set["central_source"],
                "gravitational_parameter": {
                    "value": _plain_decimal(+normalized_mu),
                    "quantity_dimension": "LENGTH^3/TIME^2",
                },
                "speed_of_light": {
                    "value": _plain_decimal(+normalized_c),
                    "quantity_dimension": "LENGTH/TIME",
                },
                "maximum_compactness": {
                    "value": coefficient_set["maximum_compactness"]["value"],
                    "quantity_dimension": "DIMENSIONLESS",
                },
                "maximum_speed_fraction_squared": {
                    "value": coefficient_set["maximum_speed_fraction_squared"]["value"],
                    "quantity_dimension": "DIMENSIONLESS",
                },
            },
        }


def qualification_scientific_fingerprint_sha256(
    fixture: Mapping[str, Any],
    coefficient_set: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> str:
    """Hash scientific state semantics without IDs or holdout labels."""

    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-scientific-state-fingerprint/v2",
            **_normalized_scientific_payload(
                epoch=fixture["epoch"],
                position=fixture["position"],
                velocity=fixture["velocity"],
                coefficient_set=coefficient_set,
                coordinate_contract=coordinate_contract,
            ),
        }
    )


def _transform_invariant_scientific_payload(
    *,
    epoch: Any,
    position: Sequence[Any],
    velocity: Sequence[Any],
    coefficient_set: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize exact aliases to TDB-compatible kilometre/second values.

    The operation order is part of this identity primitive: AU/day values are
    converted by multiplication before division, then TCB-compatible values
    are mapped forward to TDB-compatible values.  This avoids treating two
    separately rounded reciprocal scale factors as an exact quotient.
    """

    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        normalized_epoch = _canonical_decimal(epoch, "transform fingerprint epoch")
        normalized_position = [
            _canonical_decimal(value, "transform fingerprint position")
            for value in position
        ]
        normalized_velocity = [
            _canonical_decimal(value, "transform fingerprint velocity")
            for value in velocity
        ]
        normalized_mu = _canonical_decimal(
            coefficient_set["gravitational_parameter"]["value"],
            "transform fingerprint gravitational parameter",
        )
        normalized_c = _canonical_decimal(
            coefficient_set["speed_of_light"]["value"],
            "transform fingerprint speed of light",
        )
        if coordinate_contract["units"] == "AU_DAY":
            normalized_epoch *= _SECONDS_PER_DAY
            normalized_position = [
                value * _AU_IN_KILOMETRES for value in normalized_position
            ]
            normalized_velocity = [
                value * _AU_IN_KILOMETRES / _SECONDS_PER_DAY
                for value in normalized_velocity
            ]
            normalized_mu = (
                normalized_mu
                * _AU_IN_KILOMETRES ** 3
                / _SECONDS_PER_DAY ** 2
            )
            normalized_c = (
                normalized_c * _AU_IN_KILOMETRES / _SECONDS_PER_DAY
            )
        elif coordinate_contract["units"] != "KILOMETRE_SECOND":
            _fail(
                "unsupported_fingerprint_units",
                f"cannot normalize unit system {coordinate_contract['units']!r}",
            )
        if coordinate_contract["time_scale"] == "TCB_COMPATIBLE":
            normalized_epoch = (
                _TCB_TO_TDB_FACTOR * normalized_epoch
                + _IAU_L_B * _IAU_T0_JULIAN_DATE * _SECONDS_PER_DAY
                + _IAU_TDB0_SECONDS
            )
            normalized_position = [
                _TCB_TO_TDB_FACTOR * value for value in normalized_position
            ]
            normalized_mu = _TCB_TO_TDB_FACTOR * normalized_mu
        elif coordinate_contract["time_scale"] != "TDB_COMPATIBLE":
            _fail(
                "unsupported_fingerprint_time_scale",
                "scientific fingerprint requires a TCB- or TDB-compatible scale",
            )
        return {
            "epoch_kind": coordinate_contract["epoch_kind"],
            "central_source_model": coordinate_contract["central_source_model"],
            "target_treatment": coordinate_contract["target_treatment"],
            "output_semantics": coordinate_contract["output_semantics"],
            "epoch": _plain_decimal(+normalized_epoch),
            "position": [_plain_decimal(+value) for value in normalized_position],
            "velocity": [_plain_decimal(+value) for value in normalized_velocity],
            "coefficient_contract": {
                "central_source": coefficient_set["central_source"],
                "gravitational_parameter": {
                    "value": _plain_decimal(+normalized_mu),
                    "quantity_dimension": "LENGTH^3/TIME^2",
                },
                "speed_of_light": {
                    "value": _plain_decimal(+normalized_c),
                    "quantity_dimension": "LENGTH/TIME",
                },
                "maximum_compactness": {
                    "value": coefficient_set["maximum_compactness"]["value"],
                    "quantity_dimension": "DIMENSIONLESS",
                },
                "maximum_speed_fraction_squared": {
                    "value": coefficient_set["maximum_speed_fraction_squared"]["value"],
                    "quantity_dimension": "DIMENSIONLESS",
                },
            },
        }


def _transform_invariant_scientific_fingerprint_sha256(
    fixture_or_record: Mapping[str, Any],
    coefficient_set: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> str:
    return sha256_data(
        {
            "schema": (
                "jx-v5-solar-1pn-transform-invariant-"
                "scientific-state-fingerprint/v1"
            ),
            **_transform_invariant_scientific_payload(
                epoch=fixture_or_record["epoch"],
                position=fixture_or_record["position"],
                velocity=fixture_or_record["velocity"],
                coefficient_set=coefficient_set,
                coordinate_contract=coordinate_contract,
            ),
        }
    )


def _known_state_fingerprint(record: Mapping[str, Any]) -> str:
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-scientific-state-fingerprint/v2",
            **_normalized_scientific_payload(
                epoch=record["epoch"],
                position=record["position"],
                velocity=record["velocity"],
                coefficient_set=record["coefficient_contract"],
                coordinate_contract=record["coordinate_contract"],
            ),
        }
    )


def _prior_numeric_state_fingerprint(
    epoch: Any,
    position: Sequence[Any],
    velocity: Sequence[Any],
) -> str:
    """Hash a disclosed generic solver state under its explicit numeric convention."""

    return sha256_data(
        {
            "schema": (
                "jx-v5-solar-1pn-prior-development-"
                "numeric-state-fingerprint/v1"
            ),
            "synthetic_unit_convention": (
                "AU_DAY_NUMERIC_ONLY_NO_PHYSICAL_SOLAR_CONTRACT"
            ),
            "epoch": _plain_decimal(_canonical_decimal(epoch, "numeric epoch")),
            "position": [
                _plain_decimal(_canonical_decimal(value, "numeric position"))
                for value in position
            ],
            "velocity": [
                _plain_decimal(_canonical_decimal(value, "numeric velocity"))
                for value in velocity
            ],
        }
    )


def _holdout_numeric_state_fingerprint(
    fixture: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> str:
    """Normalize coherent units only for generic-solver input novelty checks."""

    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        length_to_au, time_to_day = _unit_scales_to_au_day(
            coordinate_contract["units"]
        )
        return _prior_numeric_state_fingerprint(
            _plain_decimal(
                _canonical_decimal(fixture["epoch"], "holdout numeric epoch")
                * time_to_day
            ),
            [
                _plain_decimal(
                    _canonical_decimal(value, "holdout numeric position")
                    * length_to_au
                )
                for value in fixture["position"]
            ],
            [
                _plain_decimal(
                    _canonical_decimal(value, "holdout numeric velocity")
                    * length_to_au
                    / time_to_day
                )
                for value in fixture["velocity"]
            ],
        )


def _prior_disclosure_sha256(manifest: Mapping[str, Any]) -> str:
    disclosed = dict(manifest)
    disclosed.pop("manifest_disclosure_sha256", None)
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-prior-development-disclosure/v1",
            "manifest": disclosed,
        }
    )


def _test_method_locators(source_text: str, context: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError, TypeError) as exc:
        _fail("prior_source_syntax", f"cannot parse {context}: {exc}")
    return tuple(
        f"{class_node.name}.{method.name}"
        for class_node in tree.body
        if isinstance(class_node, ast.ClassDef)
        for method in class_node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name.startswith("test_")
    )


def _method_roster_sha256(source_role: str, locators: Sequence[str]) -> str:
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-prior-development-method-roster/v1",
            "source_role": source_role,
            "locators": list(locators),
        }
    )


def _prior_case_descriptor_sha256(
    record: Mapping[str, Any],
    *,
    record_kind: str,
    record_id_field: str,
    source_raw_sha256: str,
) -> str:
    fields = {
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
    }[record_kind]
    return sha256_data(
        {
            "schema": (
                "jx-v5-solar-1pn-prior-development-case-descriptor/v1"
            ),
            "foundation_commit": FOUNDATION_COMMIT,
            "source_role": record["source_role"],
            "source_raw_sha256": source_raw_sha256,
            "record_kind": record_kind,
            "record_id": record[record_id_field],
            "source_locator": record["source_locator"],
            "alias_locators": record.get("alias_locators", []),
            "descriptor": {field: record[field] for field in fields},
        }
    )


def _connected_by_declared_transform_pairs(
    fixture_ids: set[str],
    graph: Mapping[str, set[str]],
) -> bool:
    """Return whether one fingerprint class is connected, order-independently."""

    if len(fixture_ids) < 2:
        return True
    pending = [next(iter(fixture_ids))]
    visited: set[str] = set()
    while pending:
        fixture_id = pending.pop()
        if fixture_id in visited:
            continue
        visited.add(fixture_id)
        pending.extend(graph.get(fixture_id, set()) - visited)
    return fixture_ids.issubset(visited)


def _fixture_contract_diagnostics(
    fixture: Mapping[str, Any],
    coefficient_set: Mapping[str, Any],
) -> tuple[
    str,
    bool,
    bool,
    Decimal | None,
    Decimal,
    Decimal,
    Decimal,
]:
    """Derive domain, generic-3D, and nonzero-correction predicates."""

    with localcontext(_FINGERPRINT_CONTEXT_SPEC.make_context()):
        position = tuple(
            _canonical_decimal(value, "case-role position")
            for value in fixture["position"]
        )
        velocity = tuple(
            _canonical_decimal(value, "case-role velocity")
            for value in fixture["velocity"]
        )
        mu = _canonical_decimal(
            coefficient_set["gravitational_parameter"]["value"],
            "case-role gravitational parameter",
        )
        c = _canonical_decimal(
            coefficient_set["speed_of_light"]["value"],
            "case-role speed of light",
        )
        maximum_compactness = _canonical_decimal(
            coefficient_set["maximum_compactness"]["value"],
            "case-role maximum compactness",
        )
        maximum_speed_fraction_squared = _canonical_decimal(
            coefficient_set["maximum_speed_fraction_squared"]["value"],
            "case-role maximum speed fraction squared",
        )
        radius_squared = sum((value * value for value in position), Decimal(0))
        speed_squared = sum((value * value for value in velocity), Decimal(0))
        speed_fraction_squared = speed_squared / (c * c)
        if radius_squared == 0:
            return (
                "DOMAIN_OUTSIDE",
                False,
                False,
                None,
                speed_fraction_squared,
                maximum_compactness,
                maximum_speed_fraction_squared,
            )
        radius = radius_squared.sqrt()
        compactness = mu / (radius * c * c)
        if (
            compactness <= maximum_compactness
            and speed_fraction_squared <= maximum_speed_fraction_squared
        ):
            domain_role = (
                "DOMAIN_BOUNDARY"
                if compactness == maximum_compactness
                or speed_fraction_squared == maximum_speed_fraction_squared
                else "DOMAIN_INSIDE"
            )
        else:
            domain_role = "DOMAIN_OUTSIDE"
        radius_dot_velocity = sum(
            (position[index] * velocity[index] for index in range(3)),
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
        correction_nonzero = any(value != 0 for value in numerators)
        return (
            domain_role,
            generic_3d,
            correction_nonzero,
            compactness,
            speed_fraction_squared,
            maximum_compactness,
            maximum_speed_fraction_squared,
        )


def _signal_scale_group_sha256(
    fixture: Mapping[str, Any],
    coefficient_set: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
) -> tuple[str, str]:
    """Return a state/contract digest with c removed, plus normalized c."""

    payload = _normalized_scientific_payload(
        epoch=fixture["epoch"],
        position=fixture["position"],
        velocity=fixture["velocity"],
        coefficient_set=coefficient_set,
        coordinate_contract=coordinate_contract,
    )
    normalized_c = payload["coefficient_contract"].pop("speed_of_light")["value"]
    return (
        sha256_data(
            {
                "schema": "jx-v5-solar-1pn-signal-scale-group/v1",
                **payload,
            }
        ),
        normalized_c,
    )


def _require_utc(value: Any, context: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        _fail("invalid_timestamp", f"{context} must be an explicit UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        _fail("invalid_timestamp", f"{context} is not a valid UTC timestamp: {exc}")
    return parsed


def _canonical_decimal(value: Any, context: str) -> Decimal:
    if type(value) is not str or _CANONICAL_DECIMAL.fullmatch(value) is None:
        _fail("noncanonical_decimal", f"{context} must be a canonical decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        _fail("invalid_decimal", f"{context} is not a Decimal")
    if not number.is_finite() or number.is_zero() and value.startswith("-"):
        _fail("invalid_decimal", f"{context} must be finite and not negative zero")
    return number


def _string_tuple(value: Any, context: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        _fail("invalid_string_list", f"{context} must contain nonempty strings")
    if not allow_empty and not value:
        _fail("empty_string_list", f"{context} must not be empty")
    if len(set(value)) != len(value):
        _fail("duplicate_list_item", f"{context} contains duplicate values")
    return tuple(value)


def _index(
    records: Any,
    id_field: str,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        _fail("empty_section", f"{context} must be a nonempty list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("invalid_record", f"{context}[{position}] must be an object")
        identifier = _require_identifier(
            record.get(id_field), f"{context}[{position}].{id_field}"
        )
        if identifier in indexed:
            _fail("duplicate_identifier", f"duplicate {context} ID {identifier!r}")
        indexed[identifier] = record
    return indexed


def _require_role_path_map(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, str],
    context: str,
) -> None:
    observed: dict[str, str] = {}
    for position, record in enumerate(records):
        role = _require_identifier(record.get("role"), f"{context}[{position}].role")
        if role in observed:
            _fail("duplicate_role", f"{context} repeats role {role!r}")
        path = record.get("path")
        if type(path) is not str:
            _fail("unsafe_path", f"{context}[{position}].path must be a string")
        observed[role] = path
    if observed != dict(expected):
        _fail(
            "role_path_roster",
            f"{context} role/path map differs from the frozen roster",
        )


def _blockers_for_gate(gate_id: str) -> set[str]:
    return {
        blocker_id
        for blocker_id, (_, gate_refs) in BLOCKER_SPECS.items()
        if gate_id in gate_refs
    }


def _check_refs(
    references: Any,
    available: Mapping[str, Any],
    context: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    checked = _string_tuple(references, context, allow_empty=allow_empty)
    for reference in checked:
        _require_identifier(reference, context)
        if reference not in available:
            _fail("unknown_reference", f"{context} references unknown ID {reference!r}")
    return checked


def _safe_repository_file(
    root: Path,
    relative_value: Any,
    context: str,
) -> Path:
    if type(relative_value) is not str or not relative_value:
        _fail("unsafe_path", f"{context} must be a nonempty repository-relative path")
    if "\\" in relative_value:
        _fail("unsafe_path", f"{context} must use POSIX separators")
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail("unsafe_path", f"{context} is not a normalized repository-relative path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail("symlink_forbidden", f"{context} traverses a symlink: {current}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError):
        _fail("missing_or_unsafe_file", f"{context} is missing or escapes the repository")
    if not candidate.is_file():
        _fail("missing_or_unsafe_file", f"{context} is not a regular file")
    return candidate


def _registration_file(root: Path, value: str | Path) -> Path:
    source = Path(value)
    if source.is_absolute():
        try:
            relative = source.relative_to(root)
        except ValueError:
            _fail(
                "unsafe_registration_path",
                "registration path must be a file inside the repository",
            )
    else:
        relative = source
    return _safe_repository_file(root, relative.as_posix(), "registration path")


def _verify_artifact(
    root: Path,
    binding: Mapping[str, Any],
    context: str,
) -> tuple[Path, dict[str, str]]:
    path = _safe_repository_file(root, binding.get("path"), f"{context}.path")
    expected_hash = _require_sha256(binding.get("sha256"), f"{context}.sha256")
    size = binding.get("size_bytes")
    if type(size) is not str or not size.isdigit() or int(size) <= 0:
        _fail("invalid_size", f"{context}.size_bytes must be a positive integer string")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        _fail("artifact_hash_mismatch", f"{context} hash differs from the frozen binding")
    if path.stat().st_size != int(size):
        _fail("artifact_size_mismatch", f"{context} size differs from the frozen binding")
    return path, {
        "path": path.relative_to(root).as_posix(),
        "sha256": observed_hash,
        "size_bytes": size,
    }


def _verify_canonical_digest(
    document: Mapping[str, Any],
    binding: Mapping[str, Any],
    context: str,
) -> str:
    expected = _require_sha256(
        binding.get("canonical_sha256"), f"{context}.canonical_sha256"
    )
    observed = sha256_data(_normalized(document))
    if observed != expected:
        _fail(
            "canonical_digest_mismatch",
            f"{context} canonical JSON digest differs from the frozen binding",
        )
    return observed


def _validate_with_schema(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    context: str,
) -> None:
    try:
        validate_against_bundled_schema(document, schema)
    except ForceRegistryError as exc:
        _fail(exc.code, f"{context}: {exc.message}")


def _context_from_record(record: Mapping[str, Any]) -> DecimalContextSpec:
    try:
        return DecimalContextSpec(
            precision=record["precision"],
            rounding=record["rounding"],
            emin=record["emin"],
            emax=record["emax"],
            capitals=record["capitals"],
            clamp=record["clamp"],
            traps=tuple(record["traps"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail("invalid_decimal_context", f"invalid frozen Decimal context: {exc}")


def _derivation_values(
    records: Sequence[Mapping[str, Any]],
    units: Mapping[str, Any],
    context: str,
) -> dict[str, Decimal]:
    values: dict[str, Decimal] = {}
    for position, record in enumerate(records):
        name = _require_identifier(record.get("name"), f"{context}[{position}].name")
        if name in values:
            _fail("duplicate_derivation_value", f"{context} repeats {name!r}")
        unit = _require_identifier(record.get("unit_id"), f"{context}[{position}].unit_id")
        if unit not in units:
            _fail("unknown_reference", f"{context} references unknown unit {unit!r}")
        values[name] = _canonical_decimal(
            record.get("value"), f"{context}[{position}].value"
        )
    return values


def _derivation_result_records(
    derivation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return _index(
        derivation["results"],
        "name",
        f"derivation {derivation['derivation_id']}.results",
    )


def _validate_result_binding(
    *,
    value: Any,
    unit_id: str,
    derivation_ref: str,
    result_name: str,
    derivations: Mapping[str, Mapping[str, Any]],
    context: str,
) -> None:
    derivation = derivations.get(derivation_ref)
    if derivation is None:
        _fail("unknown_reference", f"{context} references unknown derivation")
    result = _derivation_result_records(derivation).get(result_name)
    if result is None:
        _fail(
            "derivation_result_reference",
            f"{context} references missing result {result_name!r}",
        )
    if result["value"] != value or result["unit_id"] != unit_id:
        _fail(
            "derivation_result_binding",
            f"{context} value or unit differs from its derivation result",
        )


def _fixture_periapsis_eccentricity(
    fixture: Mapping[str, Any],
    derivations: Mapping[str, Mapping[str, Any]],
) -> Decimal:
    """Return the eccentricity from the one state-bound periapsis derivation."""

    state_bindings = fixture["state_derivation_bindings"]
    position_bindings = state_bindings["position"]
    velocity_bindings = state_bindings["velocity"]
    state_derivation_refs = {
        binding["derivation_ref"]
        for binding in (*position_bindings, *velocity_bindings)
    }
    periapsis_refs = {
        derivation_ref
        for derivation_ref in state_derivation_refs
        if derivation_ref in derivations
        and derivations[derivation_ref]["formula"] == "NEWTONIAN_PERIAPSIS_STATE"
    }
    if len(periapsis_refs) != 1:
        _fail(
            "q2_eccentricity_binding",
            "Q2 fixture state must use exactly one NEWTONIAN_PERIAPSIS_STATE derivation",
        )
    derivation_ref = next(iter(periapsis_refs))
    if not any(
        binding["derivation_ref"] == derivation_ref
        and binding["result_name"] == "periapsis_radius"
        for binding in position_bindings
    ) or not any(
        binding["derivation_ref"] == derivation_ref
        and binding["result_name"] == "periapsis_speed"
        for binding in velocity_bindings
    ):
        _fail(
            "q2_eccentricity_binding",
            "Q2 fixture position and velocity must bind periapsis radius and speed from the same derivation",
        )
    operands = _index(
        derivations[derivation_ref]["operands"],
        "name",
        f"derivation {derivation_ref}.operands",
    )
    eccentricity = operands.get("eccentricity")
    if eccentricity is None:
        _fail(
            "q2_eccentricity_binding",
            "Q2 periapsis derivation has no eccentricity operand",
        )
    return _canonical_decimal(
        eccentricity["value"], f"derivation {derivation_ref}.eccentricity"
    )


def _validate_q2_cell_eccentricity(
    cell: Mapping[str, Any],
    fixture: Mapping[str, Any],
    derivations: Mapping[str, Mapping[str, Any]],
) -> None:
    if _canonical_decimal(cell["eccentricity"], "Q2 cell eccentricity") != (
        _fixture_periapsis_eccentricity(fixture, derivations)
    ):
        _fail(
            "q2_eccentricity_binding",
            "Q2 cell eccentricity differs from its fixture's state-bound periapsis derivation",
        )


def _require_named_values(
    values: Mapping[str, Decimal], required: set[str], context: str
) -> None:
    if set(values) != required:
        _fail(
            "derivation_keyset",
            f"{context} requires exactly {sorted(required)}, observed {sorted(values)}",
        )


def _compare_derivation_result(
    results: Mapping[str, Decimal],
    name: str,
    expected: Decimal,
    context: str,
) -> None:
    if results.get(name) != expected:
        _fail("derivation_mismatch", f"{context} result {name!r} was not reproduced")


def _validate_derivation(
    record: Mapping[str, Any],
    contexts: Mapping[str, DecimalContextSpec],
    units: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    identifier = record["derivation_id"]
    _require_identifier(record["derivation_role"], f"derivation {identifier}.derivation_role")
    context_ref = record["decimal_context_ref"]
    if context_ref not in contexts:
        _fail("unknown_reference", f"derivation {identifier} uses unknown Decimal context")
    _check_refs(
        record["provenance_refs"],
        sources,
        f"derivation {identifier}.provenance_refs",
    )
    operands = _derivation_values(
        record["operands"], units, f"derivation {identifier}.operands"
    )
    results = _derivation_values(
        record["results"], units, f"derivation {identifier}.results"
    )
    formula = record["formula"]
    signature: dict[tuple[str, str], tuple[str, str]] = {}
    for item in record["expected_signature"]:
        key = (item["value_role"], item["name"])
        if key in signature:
            _fail("derivation_signature", f"derivation {identifier} repeats signature item {key}")
        if item["unit_id"] not in units:
            _fail("unknown_reference", f"derivation {identifier} signature uses unknown unit")
        if units[item["unit_id"]]["quantity_dimension"] != item["quantity_dimension"]:
            _fail("derivation_signature", f"derivation {identifier} signature dimension differs from its unit")
        signature[key] = (item["unit_id"], item["quantity_dimension"])
    actual_signature = {
        (role, item["name"]): (
            item["unit_id"], units[item["unit_id"]]["quantity_dimension"]
        )
        for role, records in (("OPERAND", record["operands"]), ("RESULT", record["results"]))
        for item in records
    }
    if signature != actual_signature:
        _fail("derivation_signature", f"derivation {identifier} values differ from its declared exact signature")
    expected_dimensions = FORMULA_DIMENSION_SIGNATURES.get(formula)
    if expected_dimensions is not None:
        observed_dimensions = {
            role: {
                name: dimension
                for (item_role, name), (_, dimension) in signature.items()
                if item_role == role
            }
            for role in ("OPERAND", "RESULT")
        }
        if observed_dimensions != expected_dimensions:
            _fail("derivation_signature", f"derivation {identifier} arithmetic formula signature differs")
    if formula in {"SOURCE_LITERAL", "FIXED_SCENARIO"}:
        if not results:
            _fail("empty_derivation_result", f"derivation {identifier} has no result")
        if formula == "SOURCE_LITERAL" and not record["provenance_refs"]:
            _fail(
                "unprovenanced_source_literal",
                f"derivation {identifier} requires retained provenance",
            )
        literal_spec = SOURCE_LITERAL_SPECS.get(identifier)
        if literal_spec is not None:
            source_id = literal_spec["source_id"]
            result_records = {
                result["name"]: result for result in record["results"]
            }
            expected_result = {
                "name": literal_spec["result_name"],
                "value": literal_spec["value"],
                "unit_id": literal_spec["unit_id"],
            }
            if (
                tuple(record["provenance_refs"]) != (source_id,)
                or formula != "SOURCE_LITERAL"
                or record["derivation_role"] != "SOURCE_CONSTANT"
                or record["operands"] != []
                or result_records != {
                    literal_spec["result_name"]: expected_result
                }
                or sources[source_id]["path"]
                != PROVENANCE_SOURCE_PATHS[source_id]
                or sources[source_id]["locator"]
                != literal_spec["source_locator"]
            ):
                _fail(
                    (
                        "gm_source_provenance"
                        if identifier == "derivation.source.gm_de440_km3_s2"
                        else "source_literal_binding"
                    ),
                    f"source literal {identifier} differs from its exact retained binding",
                )
        return

    arithmetic = contexts[context_ref]
    with localcontext(arithmetic.make_context()):
        if formula == "GM_KM3_S2_TO_AU3_DAY2":
            required = {"gm_km3_s2", "seconds_per_day", "km_per_au"}
            _require_named_values(operands, required, f"derivation {identifier}.operands")
            _require_named_values(results, {"gm_au3_day2"}, f"derivation {identifier}.results")
            expected = (
                operands["gm_km3_s2"]
                * operands["seconds_per_day"] ** 2
                / operands["km_per_au"] ** 3
            )
            _compare_derivation_result(results, "gm_au3_day2", expected, identifier)
        elif formula == "C_KM_S_TO_AU_DAY":
            required = {"c_km_s", "seconds_per_day", "km_per_au"}
            _require_named_values(operands, required, f"derivation {identifier}.operands")
            _require_named_values(results, {"c_au_day"}, f"derivation {identifier}.results")
            expected = (
                operands["c_km_s"]
                * operands["seconds_per_day"]
                / operands["km_per_au"]
            )
            _compare_derivation_result(results, "c_au_day", expected, identifier)
        elif formula == "NEWTONIAN_PERIAPSIS_STATE":
            required = {"mu", "semimajor_axis", "eccentricity"}
            _require_named_values(operands, required, f"derivation {identifier}.operands")
            _require_named_values(
                results,
                {"periapsis_radius", "periapsis_speed"},
                f"derivation {identifier}.results",
            )
            one = Decimal(1)
            a = operands["semimajor_axis"]
            eccentricity = operands["eccentricity"]
            if a <= 0 or not (0 <= eccentricity < 1) or operands["mu"] <= 0:
                _fail("derivation_domain", f"derivation {identifier} has invalid orbital inputs")
            radius = a * (one - eccentricity)
            speed = (
                operands["mu"] * (one + eccentricity) / radius
            ).sqrt()
            _compare_derivation_result(results, "periapsis_radius", radius, identifier)
            _compare_derivation_result(results, "periapsis_speed", speed, identifier)
        elif formula == "TCB_TDB_LINEAR_SCALING":
            required = {
                "l_b",
                "seconds_per_day",
                "t0",
                "tdb0",
                "tcb_position",
                "tcb_time",
                "tcb_mu",
                "tcb_velocity",
                "tcb_c",
            }
            _require_named_values(operands, required, f"derivation {identifier}.operands")
            expected_names = {
                "scale_factor",
                "t0",
                "tdb0",
                "tcb_mu",
                "tdb_position",
                "tdb_time",
                "tdb_mu",
                "tdb_velocity",
                "tdb_c",
            }
            _require_named_values(results, expected_names, f"derivation {identifier}.results")
            factor = Decimal(1) - operands["l_b"]
            if not (0 < factor < 1):
                _fail("derivation_domain", f"derivation {identifier} has invalid scale factor")
            _compare_derivation_result(
                results, "scale_factor", factor, identifier
            )
            _compare_derivation_result(results, "t0", operands["t0"], identifier)
            _compare_derivation_result(results, "tdb0", operands["tdb0"], identifier)
            _compare_derivation_result(
                results, "tcb_mu", operands["tcb_mu"], identifier
            )
            for source_name, result_name in (
                ("tcb_position", "tdb_position"),
                ("tcb_mu", "tdb_mu"),
            ):
                _compare_derivation_result(
                    results, result_name, factor * operands[source_name], identifier
                )
            affine_tdb_time = (
                factor * operands["tcb_time"]
                + operands["l_b"]
                * operands["t0"]
                * operands["seconds_per_day"]
                + operands["tdb0"]
            )
            _compare_derivation_result(
                results, "tdb_time", affine_tdb_time, identifier
            )
            _compare_derivation_result(
                results, "tdb_velocity", operands["tcb_velocity"], identifier
            )
            _compare_derivation_result(results, "tdb_c", operands["tcb_c"], identifier)
        elif formula == "RATIONAL_ORTHONORMAL_BASIS":
            required = {
                "p0", "p1", "p2", "q0", "q1", "q2", "n0", "n1", "n2"
            }
            _require_named_values(operands, required, f"derivation {identifier}.operands")
            result_names = {
                "p_dot_p", "q_dot_q", "n_dot_n", "p_dot_q", "p_dot_n",
                "q_dot_n", "orientation_determinant"
            }
            _require_named_values(results, result_names, f"derivation {identifier}.results")
            p = tuple(operands[f"p{i}"] for i in range(3))
            q = tuple(operands[f"q{i}"] for i in range(3))
            n = tuple(operands[f"n{i}"] for i in range(3))
            dot = lambda left, right: sum(
                (left[index] * right[index] for index in range(3)), Decimal(0)
            )
            determinant = (
                p[0] * (q[1] * n[2] - q[2] * n[1])
                - p[1] * (q[0] * n[2] - q[2] * n[0])
                + p[2] * (q[0] * n[1] - q[1] * n[0])
            )
            for name, expected in (
                ("p_dot_p", dot(p, p)),
                ("q_dot_q", dot(q, q)),
                ("n_dot_n", dot(n, n)),
                ("p_dot_q", dot(p, q)),
                ("p_dot_n", dot(p, n)),
                ("q_dot_n", dot(q, n)),
                ("orientation_determinant", determinant),
            ):
                _compare_derivation_result(results, name, expected, identifier)
            if any(results[name] != Decimal(1) for name in (
                "p_dot_p", "q_dot_q", "n_dot_n", "orientation_determinant"
            )) or any(results[name] != Decimal(0) for name in (
                "p_dot_q", "p_dot_n", "q_dot_n"
            )):
                _fail("nonorthonormal_basis", f"derivation {identifier} basis is not right-handed orthonormal")
        else:
            _fail("unsupported_derivation", f"derivation {identifier} has unsupported formula")


def qualification_fixture_sha256(fixture: Mapping[str, Any]) -> str:
    """Return the stable declared-input digest for one holdout fixture."""

    required = (
        "fixture_id",
        "status",
        "case_roles",
        "target",
        "epoch",
        "coefficient_set_ref",
        "coordinate_contract_ref",
        "position",
        "velocity",
        "derivation_refs",
        "state_derivation_bindings",
        "evidence_class",
    )
    try:
        record = {key: fixture[key] for key in required}
    except KeyError as exc:
        _fail("fixture_record", f"fixture is missing {exc.args[0]!r}")
    return sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-fixture-record/v1",
            **record,
        }
    )


def validate_prior_development_manifest_semantics(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    project_root: str | Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate byte-anchored disclosed development cases without execution.

    Returns the complete coherent-state fingerprint set (including all 4,096
    generated equation-level inputs) and the generic numeric-state fingerprint
    set used by the holdout freshness check.
    """

    _reject_binary_float(manifest, "prior-development manifest")
    _validate_with_schema(manifest, schema, "prior-development manifest")
    if (
        manifest.get("schema") != PRIOR_DEVELOPMENT_SCHEMA
        or manifest.get("manifest_id") != PRIOR_DEVELOPMENT_MANIFEST_ID
        or manifest.get("state") != PRIOR_DEVELOPMENT_MANIFEST_STATE
        or manifest.get("foundation_commit") != FOUNDATION_COMMIT
        or manifest.get("outcomes_generated") is not False
        or manifest.get("registry_authorized") is not False
        or manifest.get("excluded_from_decisive_gates") is not True
        or manifest.get("nonclaim") != PRIOR_DEVELOPMENT_NONCLAIM
    ):
        _fail(
            "prior_manifest_identity",
            "prior-development manifest identity, authority, or nonclaim differs",
        )
    if manifest["counts"] != PRIOR_DEVELOPMENT_COUNTS:
        _fail("prior_manifest_counts", "prior-development exact counts differ")
    root = Path(project_root).resolve()
    manifest_source = Path(manifest_path)
    if not manifest_source.is_absolute():
        manifest_source = _safe_repository_file(
            root, manifest_source.as_posix(), "prior-development manifest"
        )
    else:
        try:
            relative = manifest_source.resolve(strict=True).relative_to(root).as_posix()
        except (FileNotFoundError, RuntimeError, ValueError):
            _fail("unsafe_path", "prior-development manifest must be inside the repository")
        manifest_source = _safe_repository_file(
            root, relative, "prior-development manifest"
        )
    if manifest_source.relative_to(root).as_posix() != PRIOR_DEVELOPMENT_MANIFEST_PATH:
        _fail("prior_manifest_path", "prior-development manifest path differs")

    foundation_specs = {
        "development_solar_1pn_tests": {
            "foundation_sha256": "c99f77627fc351294fcdbe6d81748ff1c1ab0e0e4f40309747e02ed75969d7d0",
            "foundation_size": "16028",
            "current_sha256": "c99f77627fc351294fcdbe6d81748ff1c1ab0e0e4f40309747e02ed75969d7d0",
            "current_size": "16028",
            "policy": "NO_DELTA_BYTE_IDENTICAL",
            "changed_locators": (),
        },
        "development_reference_integrator_tests": {
            "foundation_sha256": "07ad98669347f4a10df363e41ca863f0bde65016f51cf66594c60aff7a2f195a",
            "foundation_size": "30065",
            "current_sha256": "97016ac020e2e3da0d05928d23bd3f5d4cb4263611934cd9ca37283978e42fd2",
            "current_size": "30110",
            "policy": "EXACT_SINGLE_NONSCIENTIFIC_ISOLATION_ALLOWLIST_INSERTION",
            "changed_locators": (
                "V5IsolationTests.test_legacy_modules_and_cli_do_not_reverse_import_v5_reference_path",
            ),
        },
    }
    foundation_sources = _index(
        manifest["foundation_sources"], "source_role", "foundation sources"
    )
    if set(foundation_sources) != set(PRIOR_FOUNDATION_SOURCE_PATHS):
        _fail("prior_foundation_roster", "foundation source-role roster differs")
    source_raw_sha256s: dict[str, str] = {}
    source_method_locators: dict[str, tuple[str, ...]] = {}
    for source_role, source in foundation_sources.items():
        foundation_path_expected, current_path_expected = PRIOR_FOUNDATION_SOURCE_PATHS[
            source_role
        ]
        if (
            source["foundation_commit"] != FOUNDATION_COMMIT
            or source["foundation_copy"]["path"] != foundation_path_expected
            or source["current_file"]["path"] != current_path_expected
        ):
            _fail(
                "prior_foundation_path",
                f"foundation source {source_role} path or commit differs",
            )
        foundation_path, foundation_binding = _verify_artifact(
            root, source["foundation_copy"], f"foundation copy {source_role}"
        )
        current_path, current_binding = _verify_artifact(
            root, source["current_file"], f"current development file {source_role}"
        )
        expected = foundation_specs[source_role]
        if (
            foundation_binding["sha256"] != expected["foundation_sha256"]
            or foundation_binding["size_bytes"] != expected["foundation_size"]
            or current_binding["sha256"] != expected["current_sha256"]
            or current_binding["size_bytes"] != expected["current_size"]
        ):
            _fail(
                "prior_foundation_bytes",
                f"foundation or current bytes for {source_role} differ",
            )
        try:
            foundation_text = foundation_path.read_text(encoding="utf-8")
            current_text = current_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _fail("prior_foundation_bytes", f"cannot read disclosed source: {exc}")
        proof = source["allowed_delta_proof"]
        if proof["policy"] != expected["policy"]:
            _fail("prior_allowed_delta", "allowed-delta policy differs")
        if source_role == "development_solar_1pn_tests":
            added_line = ""
            if foundation_text != current_text:
                _fail("prior_allowed_delta", "byte-identical solar source has a delta")
        else:
            added_line = '            "v5_solar_1pn_qualification.py",'
            line_with_newline = added_line + "\n"
            if current_text.count(line_with_newline) != 1:
                _fail("prior_allowed_delta", "qualification whitelist insertion differs")
            if current_text.replace(line_with_newline, "", 1) != foundation_text:
                _fail(
                    "prior_allowed_delta",
                    "current reference tests contain more than the one allowed non-scientific delta",
                )
        unified_diff = "".join(
            difflib.unified_diff(
                foundation_text.splitlines(keepends=True),
                current_text.splitlines(keepends=True),
                fromfile=foundation_path_expected,
                tofile=current_path_expected,
                n=3,
            )
        )
        foundation_locators = _test_method_locators(
            foundation_text, f"foundation {source_role}"
        )
        current_locators = _test_method_locators(
            current_text, f"current {source_role}"
        )
        expected_proof = {
            "policy": expected["policy"],
            "unified_diff_sha256": hashlib.sha256(
                unified_diff.encode("utf-8")
            ).hexdigest(),
            "added_line_sha256": hashlib.sha256(
                added_line.encode("utf-8")
            ).hexdigest(),
            "foundation_method_roster_sha256": _method_roster_sha256(
                source_role, foundation_locators
            ),
            "current_method_roster_sha256": _method_roster_sha256(
                source_role, current_locators
            ),
            "method_roster_unchanged": True,
            "changed_test_method_locators": list(expected["changed_locators"]),
            "changed_scientific_case_count": 0,
        }
        if foundation_locators != current_locators or proof != expected_proof:
            _fail(
                "prior_allowed_delta",
                f"allowed-delta or method-roster proof for {source_role} differs",
            )
        source_raw_sha256s[source_role] = foundation_binding["sha256"]
        source_method_locators[source_role] = foundation_locators

    methods = _index(manifest["method_roster"], "record_id", "prior method roster")
    method_records_by_source = {
        source_role: [
            record["source_locator"]
            for record in manifest["method_roster"]
            if record["source_role"] == source_role
        ]
        for source_role in PRIOR_FOUNDATION_SOURCE_PATHS
    }
    if any(
        tuple(method_records_by_source[source_role])
        != source_method_locators[source_role]
        for source_role in PRIOR_FOUNDATION_SOURCE_PATHS
    ):
        _fail("prior_method_roster", "manifest method roster differs from retained AST")

    descriptor_digests: list[str] = []
    for record in methods.values():
        expected_digest = _prior_case_descriptor_sha256(
            record,
            record_kind="METHOD",
            record_id_field="record_id",
            source_raw_sha256=source_raw_sha256s[record["source_role"]],
        )
        if record["descriptor_sha256"] != expected_digest:
            _fail("prior_descriptor_digest", "prior method descriptor digest differs")
        descriptor_digests.append(expected_digest)

    state_records = _index(
        manifest["known_state_records"], "record_id", "known state records"
    )
    if set(state_records) != set(KNOWN_STATE_RECORD_SPECS):
        _fail("known_state_roster", "known state record roster differs")
    known_fingerprints: set[str] = set()
    for record_id, record in state_records.items():
        if (record["source_role"], record["source_locator"]) != (
            KNOWN_STATE_RECORD_SPECS[record_id]
        ):
            _fail("known_state_locator", f"known state {record_id} locator differs")
        expected_units = COORDINATE_UNIT_REF_SPECS.get(
            record["coordinate_contract"]["units"]
        )
        if expected_units is None or any(
            record["coordinate_contract"][field] != value
            for field, value in expected_units.items()
        ):
            _fail("known_state_unit_contract", "known-state conventional units differ")
        observed = _known_state_fingerprint(record)
        if observed != record["scientific_fingerprint_sha256"]:
            _fail("known_state_fingerprint", f"known state {record_id} fingerprint differs")
        known_fingerprints.add(observed)

    invalid_records = _index(
        manifest["invalid_case_records"], "record_id", "invalid case records"
    )
    for record in invalid_records.values():
        expected_digest = _prior_case_descriptor_sha256(
            record,
            record_kind="INVALID_CASE",
            record_id_field="record_id",
            source_raw_sha256=source_raw_sha256s[record["source_role"]],
        )
        if record["descriptor_sha256"] != expected_digest:
            _fail("prior_descriptor_digest", "invalid-case descriptor digest differs")
        descriptor_digests.append(expected_digest)

    generic_records = _index(
        manifest["generic_rhs_records"], "record_id", "generic RHS records"
    )
    generic_numeric_fingerprints: set[str] = set()
    for record in generic_records.values():
        expected_numeric = _prior_numeric_state_fingerprint(
            record["initial_epoch"],
            record["initial_six_vector"][:3],
            record["initial_six_vector"][3:],
        )
        if record["state_only_numeric_fingerprint_sha256"] != expected_numeric:
            _fail("prior_numeric_fingerprint", "generic RHS numeric fingerprint differs")
        expected_digest = _prior_case_descriptor_sha256(
            record,
            record_kind="GENERIC_RHS",
            record_id_field="record_id",
            source_raw_sha256=source_raw_sha256s[record["source_role"]],
        )
        if record["descriptor_sha256"] != expected_digest:
            _fail("prior_descriptor_digest", "generic-RHS descriptor digest differs")
        generic_numeric_fingerprints.add(expected_numeric)
        descriptor_digests.append(expected_digest)

    families = _index(
        manifest["generated_state_families"],
        "family_id",
        "generated state families",
    )
    if set(families) != {"development.family.solar_equation_level_perihelion_4096"}:
        _fail("prior_family_roster", "generated state-family roster differs")
    family = next(iter(families.values()))
    family_digest = _prior_case_descriptor_sha256(
        family,
        record_kind="GENERATED_STATE_FAMILY",
        record_id_field="family_id",
        source_raw_sha256=source_raw_sha256s[family["source_role"]],
    )
    if family["descriptor_sha256"] != family_digest:
        _fail("prior_descriptor_digest", "generated-family descriptor digest differs")
    descriptor_digests.append(family_digest)
    list_binding = family["fingerprint_list"]
    if (
        list_binding["path"] != PRIOR_FINGERPRINT_LIST_PATH
        or list_binding["sha256"] != PRIOR_FINGERPRINT_LIST_FILE_SHA256
        or list_binding["size_bytes"] != PRIOR_FINGERPRINT_LIST_SIZE_BYTES
        or list_binding["canonical_sha256"]
        != PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256
    ):
        _fail("prior_family_binding", "generated fingerprint-list binding differs")
    list_path, _ = _verify_artifact(root, list_binding, "prior fingerprint list")
    fingerprint_list = _load_json(list_path, "prior fingerprint list")
    list_schema = dict(schema)
    list_schema.update(schema["$defs"]["generatedFingerprintListDocument"])
    _validate_with_schema(
        fingerprint_list, list_schema, "prior fingerprint-list artifact"
    )
    _verify_canonical_digest(
        fingerprint_list, list_binding, "prior fingerprint-list artifact"
    )
    member_fingerprints: list[str] = []
    for index, member in enumerate(fingerprint_list["members"]):
        if (
            member["index"] != index
            or member["anomaly_fraction_numerator"] != index
            or member["anomaly_fraction_denominator"] != 4096
        ):
            _fail("prior_family_order", "generated family index order differs")
        member_record = {
            "epoch": fingerprint_list["epoch"],
            "position": member["position"],
            "velocity": member["velocity"],
            "coefficient_contract": fingerprint_list["coefficient_contract"],
            "coordinate_contract": fingerprint_list["coordinate_contract"],
        }
        observed = _known_state_fingerprint(member_record)
        if observed != member["scientific_fingerprint_sha256"]:
            _fail("prior_family_fingerprint", f"generated member {index} fingerprint differs")
        member_fingerprints.append(observed)
    if len(set(member_fingerprints)) != 4096:
        _fail("prior_family_uniqueness", "generated family fingerprints are not unique")
    ordered_digest = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-prior-development-ordered-fingerprints/v1",
            "family_id": family["family_id"],
            "fingerprints": member_fingerprints,
        }
    )
    if (
        ordered_digest != PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256
        or fingerprint_list["ordered_member_fingerprints_sha256"] != ordered_digest
        or family["ordered_member_fingerprints_sha256"] != ordered_digest
        or member_fingerprints[0] != PRIOR_FIRST_MEMBER_FINGERPRINT_SHA256
        or member_fingerprints[-1] != PRIOR_LAST_MEMBER_FINGERPRINT_SHA256
        or family["first_scientific_fingerprint_sha256"] != member_fingerprints[0]
        or family["last_scientific_fingerprint_sha256"] != member_fingerprints[-1]
    ):
        _fail("prior_family_digest", "generated fingerprint-list roster digest differs")
    known_fingerprints.update(member_fingerprints)

    expected_known_ids = (
        set(state_records) | set(families) | set(invalid_records) | set(generic_records)
    )
    if set(manifest["known_fixture_ids"]) != expected_known_ids:
        _fail("prior_known_id_roster", "known prior-case ID roster differs")
    if (
        len(descriptor_digests) != PRIOR_DEVELOPMENT_COUNTS["known_case_descriptor_count"]
        or set(manifest["known_case_descriptor_sha256s"])
        != set(descriptor_digests)
        or manifest["generated_family_fingerprint_list_sha256s"]
        != [PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256]
    ):
        _fail("prior_descriptor_roster", "prior descriptor or family digest roster differs")
    disclosure_digest = _prior_disclosure_sha256(manifest)
    if (
        manifest["manifest_disclosure_sha256"] != disclosure_digest
        or disclosure_digest != PRIOR_DEVELOPMENT_DISCLOSURE_SHA256
    ):
        _fail("prior_disclosure_digest", "prior manifest disclosure digest differs")
    if (
        sha256_file(manifest_source) != PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256
        or str(manifest_source.stat().st_size)
        != PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES
        or sha256_data(_normalized(manifest))
        != PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256
    ):
        _fail("prior_manifest_bytes", "prior manifest frozen raw or canonical bytes differ")
    return frozenset(known_fingerprints), frozenset(generic_numeric_fingerprints)


def _prior_transform_invariant_fingerprints(
    manifest: Mapping[str, Any], *, project_root: Path
) -> frozenset[str]:
    """Recompute the secondary exact-alias fingerprint set from retained values."""

    fingerprints = {
        _transform_invariant_scientific_fingerprint_sha256(
            record,
            record["coefficient_contract"],
            record["coordinate_contract"],
        )
        for record in manifest["known_state_records"]
    }
    for family in manifest["generated_state_families"]:
        list_path = _safe_repository_file(
            project_root,
            family["fingerprint_list"]["path"],
            "prior transform-invariant fingerprint list",
        )
        fingerprint_list = _load_json(
            list_path, "prior transform-invariant fingerprint list"
        )
        for member in fingerprint_list["members"]:
            fingerprints.add(
                _transform_invariant_scientific_fingerprint_sha256(
                    {
                        "epoch": fingerprint_list["epoch"],
                        "position": member["position"],
                        "velocity": member["velocity"],
                    },
                    fingerprint_list["coefficient_contract"],
                    fingerprint_list["coordinate_contract"],
                )
            )
    return frozenset(fingerprints)


def _solar_contract(
    coefficient_set: Mapping[str, Any],
    coordinate: Mapping[str, Any],
) -> SolarSchwarzschild1PNContract:
    try:
        return SolarSchwarzschild1PNContract(
            central_source=coefficient_set["central_source"],
            gravitational_parameter=_canonical_decimal(
                coefficient_set["gravitational_parameter"]["value"],
                "gravitational_parameter",
            ),
            speed_of_light=_canonical_decimal(
                coefficient_set["speed_of_light"]["value"], "speed_of_light"
            ),
            units=CoherentUnitSystem[coordinate["units"]],
            frame=InertialFrame[coordinate["frame"]],
            time_scale=CoordinateTimeScale[coordinate["time_scale"]],
            central_source_model=CentralSourceModel[coordinate["central_source_model"]],
            target_treatment=TargetTreatment[coordinate["target_treatment"]],
            output_semantics=AccelerationSemantics[coordinate["output_semantics"]],
            maximum_compactness=_canonical_decimal(
                coefficient_set["maximum_compactness"]["value"],
                "maximum_compactness",
            ),
            maximum_speed_fraction_squared=_canonical_decimal(
                coefficient_set["maximum_speed_fraction_squared"]["value"],
                "maximum_speed_fraction_squared",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail("invalid_solar_contract", f"cannot construct the frozen Solar contract: {exc}")


def _validate_gate_threshold(gate: Mapping[str, Any]) -> None:
    identifier = gate["gate_id"]
    kind = gate["value_kind"]
    operator = gate["operator"]
    threshold = gate["threshold"]
    if kind == "BOOLEAN":
        if operator != "EQUAL" or threshold not in {"true", "false"}:
            _fail("invalid_gate_threshold", f"gate {identifier} has an invalid Boolean threshold")
    elif kind == "INTEGER":
        if operator not in {
            "EQUAL",
            "LESS_THAN",
            "LESS_THAN_OR_EQUAL",
            "GREATER_THAN",
            "GREATER_THAN_OR_EQUAL",
        }:
            _fail("invalid_gate_operator", f"gate {identifier} has an invalid integer operator")
        if _INTEGER_TEXT.fullmatch(threshold) is None:
            _fail("invalid_gate_threshold", f"gate {identifier} has an invalid integer threshold")
    elif kind == "DECIMAL":
        if operator == "IN_CLOSED_INTERVAL":
            pieces = threshold.split(",")
            if len(pieces) != 2:
                _fail("invalid_gate_threshold", f"gate {identifier} interval requires two decimals")
            low = _canonical_decimal(pieces[0], f"gate {identifier} lower threshold")
            high = _canonical_decimal(pieces[1], f"gate {identifier} upper threshold")
            if low > high:
                _fail("invalid_gate_threshold", f"gate {identifier} interval is reversed")
        else:
            _canonical_decimal(threshold, f"gate {identifier} threshold")
    elif kind == "ENUM":
        if operator != "EQUAL" or type(threshold) is not str or not threshold:
            _fail("invalid_gate_threshold", f"gate {identifier} has an invalid enum threshold")
    else:
        _fail("invalid_gate_threshold", f"gate {identifier} has an unknown value kind")


def _schedule_endpoint(
    schedule: Mapping[str, Any],
    fixtures: Mapping[str, Mapping[str, Any]],
) -> Decimal:
    fixture = fixtures[schedule["fixture_ref"]]
    return _canonical_decimal(fixture["epoch"], "fixture epoch") + (
        _canonical_decimal(schedule["step_size"], "schedule step_size")
        * schedule["step_count"]
    )


def validate_qualification_inputs_semantics(
    inputs: Mapping[str, Any],
    *,
    registry_path: str | Path,
    project_root: str | Path,
) -> None:
    """Validate frozen input references, derivations, and JX contract digests."""

    _reject_binary_float(inputs, "qualification inputs")
    if inputs.get("schema") != INPUT_SCHEMA or inputs.get("state") != INPUT_STATE:
        _fail("input_identity", "qualification input schema or state differs")
    if inputs["evidence_policy"]["registry_authorized"] is not False:
        _fail("authorization_boundary", "qualification inputs cannot authorize a registry")
    if inputs["claim_boundary"]["registry_authorized"] is not False:
        _fail("authorization_boundary", "qualification claims cannot authorize a registry")
    claim_boundary = inputs["claim_boundary"]
    if (
        claim_boundary["maximum_claim_after_pass"] != PASS_EFFECT
        or claim_boundary["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or claim_boundary["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
        or tuple(claim_boundary["prohibited_claims"]) != PROHIBITED_CLAIM_CODES
        or claim_boundary["nonclaim"] != NONCLAIM
    ):
        _fail("claim_boundary", "qualification input claim boundary differs")

    root = Path(project_root).resolve()
    units = _index(inputs["unit_definitions"], "unit_id", "unit_definitions")
    coordinates = _index(
        inputs["coordinate_contracts"],
        "coordinate_contract_id",
        "coordinate_contracts",
    )
    coefficients = _index(
        inputs["coefficient_sets"], "coefficient_set_id", "coefficient_sets"
    )
    context_records = _index(
        inputs["decimal_contexts"], "decimal_context_id", "decimal_contexts"
    )
    fixtures = _index(inputs["fixtures"], "fixture_id", "fixtures")
    schedules = _index(
        inputs["integration_schedules"], "schedule_id", "integration_schedules"
    )
    sources = _index(inputs["provenance_sources"], "provenance_id", "provenance_sources")
    derivations = _index(inputs["derivations"], "derivation_id", "derivations")
    blockers = _index(
        inputs["blocked_artifacts"],
        "placeholder_id",
        "blocked_artifacts",
    )

    if set(derivations) != set(DERIVATION_IDS):
        _fail("derivation_roster", "qualification derivation roster differs")

    if set(blockers) != set(BLOCKER_SPECS):
        _fail("blocked_artifact_roster", "blocked artifact roster differs")
    for blocker_id, blocker in blockers.items():
        expected_kind, expected_gates = BLOCKER_SPECS[blocker_id]
        if (
            blocker["status"] != "BLOCKED"
            or blocker["artifact_kind"] != expected_kind
            or set(blocker["gate_refs"]) != set(expected_gates)
            or blocker["required_before_execution"] is not True
        ):
            _fail(
                "blocked_artifact_binding",
                f"blocked artifact {blocker_id!r} differs from its frozen role",
            )

    observed_source_paths = {
        source_id: source["path"] for source_id, source in sources.items()
    }
    if observed_source_paths != PROVENANCE_SOURCE_PATHS:
        _fail("provenance_source_roster", "retained source ID/path roster differs")

    for source_id, source in sources.items():
        _verify_artifact(root, source, f"provenance source {source_id}")

    for unit_id, unit in units.items():
        _check_refs(unit["provenance_refs"], sources, f"unit {unit_id}.provenance_refs")
        if unit["definition_kind"] != "FIXED_SCENARIO" and not unit["provenance_refs"]:
            _fail("unprovenanced_unit", f"unit {unit_id} requires retained provenance")

    coordinate_dimension_fields = {
        "length_unit_ref": UNIT_DIMENSIONS["length"],
        "time_unit_ref": UNIT_DIMENSIONS["time"],
        "velocity_unit_ref": UNIT_DIMENSIONS["velocity"],
        "gravitational_parameter_unit_ref": UNIT_DIMENSIONS[
            "gravitational_parameter"
        ],
    }
    for coordinate_id, coordinate in coordinates.items():
        expected_unit_refs = COORDINATE_UNIT_REF_SPECS.get(coordinate["units"])
        if expected_unit_refs is None:
            _fail(
                "unsupported_coordinate_units",
                f"coordinate contract {coordinate_id} uses unsupported units",
            )
        if any(
            coordinate[field] != expected_unit_ref
            for field, expected_unit_ref in expected_unit_refs.items()
        ):
            _fail(
                "coordinate_unit_roster",
                f"coordinate contract {coordinate_id} does not use the exact conventional unit IDs",
            )
        for field, expected_dimension in coordinate_dimension_fields.items():
            unit_ref = coordinate[field]
            if unit_ref not in units:
                _fail(
                    "unknown_reference",
                    f"coordinate contract {coordinate_id}.{field} has unknown unit",
                )
            if units[unit_ref]["quantity_dimension"] != expected_dimension:
                _fail(
                    "unit_dimension",
                    f"coordinate contract {coordinate_id}.{field} has wrong dimension",
                )

    contexts: dict[str, DecimalContextSpec] = {}
    for context_id, record in context_records.items():
        arithmetic = _context_from_record(record)
        if arithmetic.sha256 != record["expected_sha256"]:
            _fail("decimal_context_digest", f"Decimal context {context_id} digest differs")
        contexts[context_id] = arithmetic
    expected_contexts = {
        "decimal.context.precision_50": 50,
        "decimal.context.precision_70": 70,
        "decimal.context.precision_90": 90,
    }
    if set(contexts) != set(expected_contexts) or any(
        contexts[context_id].precision != precision
        for context_id, precision in expected_contexts.items()
    ):
        _fail("precision_roster", "qualification inputs require exactly the frozen 50/70/90 contexts")

    for derivation in derivations.values():
        _validate_derivation(derivation, contexts, units, sources)

    registry_source = Path(registry_path)
    if not registry_source.is_absolute():
        registry_source = root / registry_source
    try:
        registry_relative = registry_source.resolve(strict=True).relative_to(root).as_posix()
    except (FileNotFoundError, RuntimeError, ValueError):
        _fail("unsafe_registry_path", "registry path must be inside the repository")
    registry_source = _safe_repository_file(root, registry_relative, "registry path")

    for coefficient_id, coefficient_set in coefficients.items():
        coordinate_ref = coefficient_set["coordinate_contract_ref"]
        if coordinate_ref not in coordinates:
            _fail("unknown_reference", f"coefficient set {coefficient_id} has unknown coordinates")
        coordinate = coordinates[coordinate_ref]
        expected_dimensions = {
            "gravitational_parameter": UNIT_DIMENSIONS["gravitational_parameter"],
            "speed_of_light": UNIT_DIMENSIONS["velocity"],
            "maximum_compactness": UNIT_DIMENSIONS["dimensionless"],
            "maximum_speed_fraction_squared": UNIT_DIMENSIONS["dimensionless"],
        }
        for field in (
            "gravitational_parameter",
            "speed_of_light",
            "maximum_compactness",
            "maximum_speed_fraction_squared",
        ):
            value = coefficient_set[field]
            unit_ref = value["unit_id"]
            if unit_ref not in units:
                _fail("unknown_reference", f"coefficient set {coefficient_id}.{field} has unknown unit")
            _check_refs(
                value["provenance_refs"],
                sources,
                f"coefficient set {coefficient_id}.{field}.provenance_refs",
            )
            derivation_ref = value["derivation_ref"]
            if derivation_ref not in derivations:
                _fail("unknown_reference", f"coefficient set {coefficient_id}.{field} has unknown derivation")
            _canonical_decimal(value["value"], f"coefficient set {coefficient_id}.{field}.value")
            if units[unit_ref]["quantity_dimension"] != expected_dimensions[field]:
                _fail(
                    "coefficient_dimension",
                    f"coefficient set {coefficient_id}.{field} has wrong dimension",
                )
            if field == "gravitational_parameter" and unit_ref != coordinate[
                "gravitational_parameter_unit_ref"
            ]:
                _fail("coefficient_unit", "GM unit differs from coordinate contract")
            if field == "speed_of_light" and unit_ref != coordinate["velocity_unit_ref"]:
                _fail("coefficient_unit", "c unit differs from coordinate contract")
            derivation = derivations[derivation_ref]
            if set(value["provenance_refs"]) != set(derivation["provenance_refs"]):
                _fail(
                    "coefficient_provenance_binding",
                    f"coefficient set {coefficient_id}.{field} provenance differs from derivation",
                )
            _validate_result_binding(
                value=value["value"],
                unit_id=unit_ref,
                derivation_ref=derivation_ref,
                result_name=value["derivation_result_name"],
                derivations=derivations,
                context=f"coefficient set {coefficient_id}.{field}",
            )
        contract = _solar_contract(coefficient_set, coordinate)
        try:
            force_plan = inspect_reference_force_plan(
                registry_source,
                contract,
                project_root=root,
            )
        except ValueError as exc:
            _fail("force_plan_binding", f"coefficient set {coefficient_id} cannot bind a reference plan: {exc}")
        if force_plan.sha256 != coefficient_set["expected_force_plan_sha256"]:
            _fail("force_plan_digest", f"coefficient set {coefficient_id} force-plan digest differs")

    fixture_diagnostics: dict[
        str,
        tuple[
            str,
            bool,
            bool,
            Decimal | None,
            Decimal,
            Decimal,
            Decimal,
        ],
    ] = {}
    domain_roles = {"DOMAIN_INSIDE", "DOMAIN_BOUNDARY", "DOMAIN_OUTSIDE"}
    compactness_only_outside: set[str] = set()
    speed_only_outside: set[str] = set()
    for fixture_id, fixture in fixtures.items():
        case_roles = _string_tuple(
            fixture["case_roles"], f"fixture {fixture_id}.case_roles", allow_empty=False
        )
        if not set(case_roles).issubset(CASE_ROLES):
            _fail("fixture_case_role", f"fixture {fixture_id} has an unknown case role")
        coefficient_ref = fixture["coefficient_set_ref"]
        coordinate_ref = fixture["coordinate_contract_ref"]
        if coefficient_ref not in coefficients or coordinate_ref not in coordinates:
            _fail("unknown_reference", f"fixture {fixture_id} has an unknown contract reference")
        if coefficients[coefficient_ref]["coordinate_contract_ref"] != coordinate_ref:
            _fail("coordinate_mismatch", f"fixture {fixture_id} coordinate bindings differ")
        if len(fixture["position"]) != 3 or len(fixture["velocity"]) != 3:
            _fail("state_shape", f"fixture {fixture_id} requires three position and velocity components")
        _canonical_decimal(fixture["epoch"], f"fixture {fixture_id}.epoch")
        for index, value in enumerate(fixture["position"]):
            _canonical_decimal(value, f"fixture {fixture_id}.position[{index}]")
        for index, value in enumerate(fixture["velocity"]):
            _canonical_decimal(value, f"fixture {fixture_id}.velocity[{index}]")
        _check_refs(
            fixture["derivation_refs"],
            derivations,
            f"fixture {fixture_id}.derivation_refs",
        )
        coordinate = coordinates[coordinate_ref]
        state_bindings = fixture["state_derivation_bindings"]
        used_derivations: set[str] = set()
        binding_groups = (
            (
                (fixture["epoch"],),
                (state_bindings["epoch"],),
                coordinate["time_unit_ref"],
                "epoch",
            ),
            (
                fixture["position"],
                state_bindings["position"],
                coordinate["length_unit_ref"],
                "position",
            ),
            (
                fixture["velocity"],
                state_bindings["velocity"],
                coordinate["velocity_unit_ref"],
                "velocity",
            ),
        )
        for values, bindings, expected_unit, label in binding_groups:
            if len(values) != len(bindings):
                _fail("state_derivation_shape", f"fixture {fixture_id}.{label} bindings differ")
            for index, (value, binding) in enumerate(zip(values, bindings)):
                derivation_ref = binding["derivation_ref"]
                used_derivations.add(derivation_ref)
                _validate_result_binding(
                    value=value,
                    unit_id=expected_unit,
                    derivation_ref=derivation_ref,
                    result_name=binding["result_name"],
                    derivations=derivations,
                    context=f"fixture {fixture_id}.{label}[{index}]",
                )
        if set(fixture["derivation_refs"]) != used_derivations:
            _fail(
                "state_derivation_roster",
                f"fixture {fixture_id} derivation roster differs from state bindings",
            )
        if qualification_fixture_sha256(fixture) != fixture["expected_state_record_sha256"]:
            _fail("fixture_digest", f"fixture {fixture_id} declared-state digest differs")
        diagnostics = _fixture_contract_diagnostics(
            fixture, coefficients[coefficient_ref]
        )
        fixture_diagnostics[fixture_id] = diagnostics
        (
            derived_domain_role,
            generic_3d,
            _,
            compactness,
            speed_fraction_squared,
            maximum_compactness,
            maximum_speed_fraction_squared,
        ) = diagnostics
        declared_domain_roles = set(case_roles) & domain_roles
        if declared_domain_roles != {derived_domain_role}:
            _fail(
                "derived_domain_role",
                f"fixture {fixture_id} must declare exactly its computed {derived_domain_role} role",
            )
        if "GENERIC_3D" in case_roles and not generic_3d:
            _fail(
                "derived_generic_3d_role",
                f"fixture {fixture_id} declares GENERIC_3D without satisfying its algebraic component test",
            )
        if derived_domain_role == "DOMAIN_OUTSIDE" and compactness is not None:
            if (
                compactness > maximum_compactness
                and speed_fraction_squared < maximum_speed_fraction_squared
            ):
                compactness_only_outside.add(fixture_id)
            if (
                speed_fraction_squared > maximum_speed_fraction_squared
                and compactness < maximum_compactness
            ):
                speed_only_outside.add(fixture_id)

    if not compactness_only_outside or not speed_only_outside:
        _fail(
            "domain_outside_axis_coverage",
            "holdouts must contain distinct compactness-only and speed-only domain violations",
        )

    schedule_coverage = {fixture_id: 0 for fixture_id in fixtures}
    for schedule_id, schedule in schedules.items():
        if schedule["fixture_ref"] not in fixtures:
            _fail("unknown_reference", f"schedule {schedule_id} references unknown fixture")
        schedule_coverage[schedule["fixture_ref"]] += 1
        if schedule["decimal_context_ref"] not in contexts:
            _fail("unknown_reference", f"schedule {schedule_id} references unknown context")
        try:
            specification = ImplicitMidpointSpec(
                step_size=_canonical_decimal(schedule["step_size"], f"schedule {schedule_id}.step_size"),
                step_count=schedule["step_count"],
                position_atol=_canonical_decimal(schedule["position_atol"], f"schedule {schedule_id}.position_atol"),
                velocity_atol=_canonical_decimal(schedule["velocity_atol"], f"schedule {schedule_id}.velocity_atol"),
                relative_tolerance=_canonical_decimal(
                    schedule["relative_tolerance"], f"schedule {schedule_id}.relative_tolerance"
                ),
                maximum_iterations=schedule["maximum_iterations"],
                initial_guess_policy=schedule["initial_guess_policy"],
                method_id=schedule["method_id"],
                registry_authorized=False,
            )
        except (TypeError, ValueError) as exc:
            _fail("invalid_integration_schedule", f"schedule {schedule_id} is invalid: {exc}")
        if specification.sha256 != schedule["expected_spec_sha256"]:
            _fail("integration_spec_digest", f"schedule {schedule_id} spec digest differs")
    missing_schedule_fixtures = sorted(
        fixture_id for fixture_id, count in schedule_coverage.items() if count == 0
    )
    if missing_schedule_fixtures:
        _fail(
            "fixture_schedule_coverage",
            f"fixtures lack schedules: {missing_schedule_fixtures}",
        )

    q2 = inputs["q2_perihelion_observer"]
    if q2["decimal_context_ref"] not in contexts:
        _fail("unknown_reference", "Q2 observer uses unknown Decimal context")
    for field in (
        "step_grid",
        "event_tolerance_grid",
        "inverse_c_squared_ladder",
    ):
        values = [_canonical_decimal(value, f"Q2 {field}") for value in q2[field]]
        if any(value <= 0 for value in values):
            _fail("q2_grid_domain", f"Q2 {field} must be strictly positive")
    eccentricities = [
        _canonical_decimal(value, "Q2 eccentricity")
        for value in q2["eccentricity_roster"]
    ]
    if any(not (0 < value < 1) for value in eccentricities):
        _fail("q2_eccentricity_domain", "Q2 eccentricities must lie in (0, 1)")
    if type(q2["orbit_count"]) is not int or q2["orbit_count"] < 2:
        _fail("q2_orbit_count", "Q2 requires at least two complete orbits")
    orbit_span_bindings = _index(
        q2["orbit_span_bindings"],
        "fixture_ref",
        "Q2 orbit-span bindings",
    )
    for fixture_ref, binding in orbit_span_bindings.items():
        if fixture_ref not in fixtures:
            _fail("unknown_reference", "Q2 orbit-span binding uses unknown fixture")
        derivation_ref = binding["periapsis_derivation_ref"]
        if (
            derivation_ref not in derivations
            or derivations[derivation_ref]["formula"] != "NEWTONIAN_PERIAPSIS_STATE"
            or derivation_ref not in fixtures[fixture_ref]["derivation_refs"]
        ):
            _fail(
                "q2_orbit_binding",
                f"Q2 fixture {fixture_ref} does not bind one declared periapsis derivation",
            )
        if binding["orbit_count"] != q2["orbit_count"]:
            _fail("q2_orbit_binding", "Q2 orbit-span orbit count differs")
        orbit_operands = _derivation_values(
            derivations[derivation_ref]["operands"],
            units,
            f"Q2 fixture {fixture_ref} orbit operands",
        )
        coefficient = coefficients[fixtures[fixture_ref]["coefficient_set_ref"]]
        mu_value = _canonical_decimal(
            coefficient["gravitational_parameter"]["value"],
            "Q2 coefficient mu",
        )
        if orbit_operands["mu"] != mu_value:
            _fail("q2_orbit_binding", "Q2 orbit-span mu differs from fixture GM")
        with localcontext(contexts["decimal.context.precision_90"].make_context()):
            expected_minimum_span = q2["orbit_count"] * (
                _TAU_CONSERVATIVE_NUMERATOR
                / _TAU_CONSERVATIVE_DENOMINATOR
                * (
                    orbit_operands["semimajor_axis"] ** 3
                    / orbit_operands["mu"]
                ).sqrt()
            )
        if _canonical_decimal(
            binding["minimum_span"], "Q2 minimum span"
        ) != expected_minimum_span:
            _fail(
                "q2_orbit_span_binding",
                "Q2 minimum span is not the exact precision-90 44/7 conservative bound",
            )
    schedule_steps = {
        abs(_canonical_decimal(schedule["step_size"], "schedule step"))
        for schedule in schedules.values()
    }
    if not {
        _canonical_decimal(value, "Q2 step grid") for value in q2["step_grid"]
    }.issubset(schedule_steps):
        _fail("q2_schedule_grid", "Q2 step grid is not represented by schedules")
    expected_q2_cells = {
        (eccentricity, inverse_c_squared, step, event_tolerance)
        for eccentricity in q2["eccentricity_roster"]
        for inverse_c_squared in q2["inverse_c_squared_ladder"]
        for step in q2["step_grid"]
        for event_tolerance in q2["event_tolerance_grid"]
    }
    observed_q2_cells: dict[tuple[str, str, str, str], str] = {}
    q2_cell_ids: set[str] = set()
    for cell in q2["case_cells"]:
        cell_id = _require_identifier(cell["cell_id"], "Q2 cell_id")
        if cell_id in q2_cell_ids:
            _fail("q2_duplicate_cell", f"Q2 repeats cell ID {cell_id!r}")
        q2_cell_ids.add(cell_id)
        key = (
            cell["eccentricity"],
            cell["inverse_c_squared_value"],
            cell["step_size"],
            cell["event_tolerance"],
        )
        if key in observed_q2_cells:
            _fail("q2_duplicate_cell", f"Q2 repeats Cartesian cell {key}")
        if cell["fixture_ref"] not in fixtures:
            _fail("unknown_reference", f"Q2 cell {cell_id} references an unknown fixture")
        fixture = fixtures[cell["fixture_ref"]]
        if not {"ECCENTRIC", "LONG_ARC"}.issubset(set(fixture["case_roles"])):
            _fail("q2_case_role", f"Q2 cell {cell_id} lacks ECCENTRIC and LONG_ARC roles")
        _validate_q2_cell_eccentricity(cell, fixture, derivations)
        newtonian_ref = cell["newtonian_schedule_ref"]
        one_pn_ref = cell["one_pn_schedule_ref"]
        if newtonian_ref not in schedules or one_pn_ref not in schedules:
            _fail("unknown_reference", f"Q2 cell {cell_id} references an unknown schedule")
        newtonian = schedules[newtonian_ref]
        one_pn = schedules[one_pn_ref]
        if (
            newtonian["fixture_ref"] != cell["fixture_ref"]
            or one_pn["fixture_ref"] != cell["fixture_ref"]
            or newtonian["force_model_set"] != "NEWTONIAN_ONLY"
            or one_pn["force_model_set"] != "NEWTONIAN_PLUS_SOLAR_1PN"
        ):
            _fail("q2_force_pair", f"Q2 cell {cell_id} is not a paired Newtonian/1PN execution")
        paired_fields = (
            "decimal_context_ref", "step_size", "step_count", "position_atol",
            "velocity_atol", "relative_tolerance", "maximum_iterations",
            "initial_guess_policy", "method_id",
        )
        if any(newtonian[field] != one_pn[field] for field in paired_fields):
            _fail("q2_force_pair", f"Q2 cell {cell_id} paired schedules differ numerically")
        if abs(_canonical_decimal(newtonian["step_size"], "Q2 schedule step")) != _canonical_decimal(cell["step_size"], "Q2 cell step"):
            _fail("q2_cell_binding", f"Q2 cell {cell_id} step differs from its schedule")
        coefficient = coefficients[fixture["coefficient_set_ref"]]
        c_value = _canonical_decimal(coefficient["speed_of_light"]["value"], "Q2 c")
        with localcontext(contexts["decimal.context.precision_90"].make_context()):
            if _canonical_decimal(cell["inverse_c_squared_value"], "Q2 inverse c squared") != Decimal(1) / c_value ** 2:
                _fail("q2_scaling_binding", f"Q2 cell {cell_id} does not bind its fixture c^-2")
            periapsis_derivations = [
                derivations[reference]
                for reference in fixture["derivation_refs"]
                if derivations[reference]["formula"] == "NEWTONIAN_PERIAPSIS_STATE"
            ]
            if len(periapsis_derivations) != 1:
                _fail("q2_orbit_binding", f"Q2 cell {cell_id} has an ambiguous orbit derivation")
            orbit_operands = _derivation_values(
                periapsis_derivations[0]["operands"],
                units,
                f"Q2 cell {cell_id} orbit operands",
            )
            mu_value = _canonical_decimal(
                coefficient["gravitational_parameter"]["value"], "Q2 coefficient mu"
            )
            if orbit_operands["mu"] != mu_value:
                _fail("q2_orbit_binding", f"Q2 cell {cell_id} orbit mu differs from its coefficient")
            conservative_period = (
                _TAU_CONSERVATIVE_NUMERATOR
                / _TAU_CONSERVATIVE_DENOMINATOR
                * (
                    orbit_operands["semimajor_axis"] ** 3
                    / orbit_operands["mu"]
                ).sqrt()
            )
            required_span = q2["orbit_count"] * conservative_period
            span_binding = orbit_span_bindings.get(cell["fixture_ref"])
            if span_binding is None:
                _fail(
                    "q2_orbit_span_binding",
                    f"Q2 cell {cell_id} has no frozen orbit-span binding",
                )
            if (
                span_binding["periapsis_derivation_ref"]
                != periapsis_derivations[0]["derivation_id"]
                or _canonical_decimal(
                    span_binding["minimum_span"], "Q2 bound minimum span"
                )
                != required_span
            ):
                _fail(
                    "q2_orbit_span_binding",
                    f"Q2 cell {cell_id} differs from its frozen orbit-span binding",
                )
            for schedule in (newtonian, one_pn):
                schedule_span = abs(
                    _canonical_decimal(schedule["step_size"], "Q2 schedule span")
                    * schedule["step_count"]
                )
                if schedule_span < required_span:
                    _fail(
                        "q2_orbit_span",
                        f"Q2 cell {cell_id} does not cover orbit_count under the conservative period bound",
                    )
        observed_q2_cells[key] = cell_id
    if set(observed_q2_cells) != expected_q2_cells:
        _fail("q2_cartesian_grid", "Q2 case cells do not form the exact frozen Cartesian product")
    q2_fixture_ids = {
        cell["fixture_ref"] for cell in q2["case_cells"]
    }
    if set(orbit_span_bindings) != q2_fixture_ids:
        _fail(
            "q2_orbit_span_roster",
            "Q2 orbit-span bindings must cover every and only Q2 fixture exactly once",
        )

    q3 = inputs["q3_independent_oracle"]
    if type(q3["minimum_method_order"]) is not int or q3["minimum_method_order"] < 8:
        _fail("q3_method_order", "Q3 oracle method order must be at least eight")
    self_convergence = q3["self_convergence"]
    _check_refs(
        self_convergence["precision_context_refs"],
        contexts,
        "Q3 self-convergence precision contexts",
        allow_empty=False,
    )
    if any(
        _canonical_decimal(value, "Q3 tolerance level") <= 0
        for value in self_convergence["tolerance_levels"]
    ):
        _fail("q3_tolerance_grid", "Q3 tolerance levels must be positive")
    metrics = _index(q3["metrics"], "metric_id", "Q3 metrics")
    if set(metrics) != {"TOTAL_STATE", "DIFFERENTIAL_SIGNAL"}:
        _fail("q3_metric_roster", "Q3 requires total-state and differential-signal metrics")
    for metric_id, metric in metrics.items():
        _check_refs(metric["scale_unit_refs"], units, f"Q3 {metric_id} units", allow_empty=False)
        _check_refs(
            metric["checkpoint_schedule_refs"],
            schedules,
            f"Q3 {metric_id} schedules",
            allow_empty=False,
        )
    if (
        metrics["TOTAL_STATE"]["checkpoint_schedule_refs"]
        != metrics["DIFFERENTIAL_SIGNAL"]["checkpoint_schedule_refs"]
    ):
        _fail(
            "q3_metric_checkpoint_roster",
            "Q3 total-state and differential-signal checkpoint rosters must be identical",
        )
    checkpoint_schedule_refs = set(
        metrics["TOTAL_STATE"]["checkpoint_schedule_refs"]
    )
    observed_q3_cells: dict[tuple[str, str, str, str, str], str] = {}
    q3_schedule_pairs: set[tuple[str, str, str]] = set()
    q3_fixture_ids: set[str] = set()
    q3_cell_ids: set[str] = set()
    for cell in q3["case_cells"]:
        cell_id = _require_identifier(cell["cell_id"], "Q3 cell_id")
        if cell_id in q3_cell_ids:
            _fail("q3_duplicate_cell", f"Q3 repeats cell ID {cell_id!r}")
        q3_cell_ids.add(cell_id)
        fixture_ref = cell["fixture_ref"]
        schedule_ref = cell["checkpoint_schedule_ref"]
        newtonian_ref = cell["paired_newtonian_schedule_ref"]
        if (
            fixture_ref not in fixtures
            or schedule_ref not in schedules
            or newtonian_ref not in schedules
        ):
            _fail("unknown_reference", f"Q3 cell {cell_id} has an unknown fixture or schedule")
        one_pn_schedule = schedules[schedule_ref]
        newtonian_schedule = schedules[newtonian_ref]
        q3_pair_fields = (
            "fixture_ref",
            "decimal_context_ref",
            "step_size",
            "step_count",
            "position_atol",
            "velocity_atol",
            "relative_tolerance",
            "maximum_iterations",
            "initial_guess_policy",
            "method_id",
        )
        if (
            one_pn_schedule["fixture_ref"] != fixture_ref
            or newtonian_schedule["fixture_ref"] != fixture_ref
            or one_pn_schedule["force_model_set"]
            != "NEWTONIAN_PLUS_SOLAR_1PN"
            or newtonian_schedule["force_model_set"] != "NEWTONIAN_ONLY"
            or any(
                one_pn_schedule[field] != newtonian_schedule[field]
                for field in q3_pair_fields
            )
        ):
            _fail(
                "q3_force_pair",
                f"Q3 cell {cell_id} is not an otherwise-identical Newtonian/1PN pair",
            )
        if schedule_ref not in checkpoint_schedule_refs:
            _fail("q3_cell_binding", f"Q3 cell {cell_id} schedule is absent from both metric rosters")
        if cell["self_convergence_tolerance"] not in self_convergence["tolerance_levels"]:
            _fail("q3_cell_binding", f"Q3 cell {cell_id} uses an unfrozen tolerance")
        if cell["precision_context_ref"] not in self_convergence["precision_context_refs"]:
            _fail("q3_cell_binding", f"Q3 cell {cell_id} uses an unfrozen precision")
        pair_key = (fixture_ref, schedule_ref, newtonian_ref)
        q3_schedule_pairs.add(pair_key)
        key = (
            fixture_ref,
            schedule_ref,
            newtonian_ref,
            cell["self_convergence_tolerance"],
            cell["precision_context_ref"],
        )
        if key in observed_q3_cells:
            _fail("q3_duplicate_cell", f"Q3 repeats oracle cell {key}")
        observed_q3_cells[key] = cell_id
        q3_fixture_ids.add(fixture_ref)
    expected_q3_cells = {
        (fixture_ref, schedule_ref, newtonian_ref, tolerance, precision)
        for fixture_ref, schedule_ref, newtonian_ref in q3_schedule_pairs
        for tolerance in self_convergence["tolerance_levels"]
        for precision in self_convergence["precision_context_refs"]
    }
    if set(observed_q3_cells) != expected_q3_cells or not q3_fixture_ids:
        _fail("q3_checkpoint_matrix", "Q3 cells do not cover every bound fixture/checkpoint/tolerance/precision case")
    if {schedule_ref for _, schedule_ref, _ in q3_schedule_pairs} != checkpoint_schedule_refs:
        _fail(
            "q3_checkpoint_roster",
            "Q3 cells and both metric checkpoint rosters differ",
        )

    q4 = inputs["q4_eih_evaluator"]
    nu_ladder = [_canonical_decimal(value, "Q4 nu") for value in q4["nu_ladder"]]
    if len(nu_ladder) < 3 or any(not (0 < value <= Decimal("0.25")) for value in nu_ladder):
        _fail("q4_nu_ladder", "Q4 requires at least three nu values in (0, 0.25]")
    q4_fixture_ref = q4["fixture_ref"]
    if q4_fixture_ref not in fixtures:
        _fail("unknown_reference", "Q4 references an unknown fixture")
    q4_fixture = fixtures[q4_fixture_ref]
    q4_coefficient = coefficients[q4_fixture["coefficient_set_ref"]]
    q4_mu = q4_coefficient["gravitational_parameter"]
    if (
        _canonical_decimal(q4["total_mu_value"], "Q4 total mu")
        != _canonical_decimal(q4_mu["value"], "Q4 fixture mu")
        or q4["total_mu_unit_ref"] != q4_mu["unit_id"]
    ):
        _fail("q4_total_mu_binding", "Q4 fixed total mu differs from its fixture GM")
    q4_radius_squared = sum(
        (
            _canonical_decimal(value, "Q4 position") ** 2
            for value in q4_fixture["position"]
        ),
        Decimal(0),
    )
    if q4_radius_squared <= 0:
        _fail("q4_fixture_domain", "Q4 EIH-limit fixture requires positive radius")
    observed_q4_cells: dict[str, str] = {}
    q4_cell_ids: set[str] = set()
    for cell in q4["case_cells"]:
        cell_id = _require_identifier(cell["cell_id"], "Q4 cell_id")
        if cell_id in q4_cell_ids:
            _fail("q4_duplicate_cell", f"Q4 repeats cell ID {cell_id!r}")
        q4_cell_ids.add(cell_id)
        if cell["nu"] in observed_q4_cells:
            _fail("q4_duplicate_cell", f"Q4 repeats nu {cell['nu']}")
        if (
            cell["fixture_ref"] != q4_fixture_ref
            or cell["total_mu_value"] != q4["total_mu_value"]
            or cell["total_mu_unit_ref"] != q4["total_mu_unit_ref"]
        ):
            _fail("q4_cell_binding", "Q4 case cell differs from its fixed-total-mu fixture")
        observed_q4_cells[cell["nu"]] = cell_id
    if set(observed_q4_cells) != set(q4["nu_ladder"]):
        _fail("q4_nu_roster", "Q4 case cells must cover the exact nu ladder once")

    q5 = inputs["q5_convergence_grid"]
    q5_schedules = _check_refs(
        q5["step_schedule_refs"],
        schedules,
        "Q5 step schedules",
        allow_empty=False,
    )
    common_endpoint = _canonical_decimal(q5["common_endpoint"], "Q5 common endpoint")
    q5_fixture_ids = {schedules[reference]["fixture_ref"] for reference in q5_schedules}
    if len(q5_fixture_ids) != 1:
        _fail("q5_fixture_grid", "Q5 step grid must use one fixture")
    q5_common_policy_fields = (
        "fixture_ref",
        "force_model_set",
        "decimal_context_ref",
        "position_atol",
        "velocity_atol",
        "relative_tolerance",
        "maximum_iterations",
        "initial_guess_policy",
        "method_id",
    )
    q5_reference_schedule = schedules[q5_schedules[0]]
    if any(
        schedules[reference][field] != q5_reference_schedule[field]
        for reference in q5_schedules[1:]
        for field in q5_common_policy_fields
    ):
        _fail(
            "q5_schedule_policy",
            "Q5 step schedules must share one force, context, solver, and tolerance policy",
        )
    if any(_schedule_endpoint(schedules[reference], fixtures) != common_endpoint for reference in q5_schedules):
        _fail("q5_common_endpoint", "Q5 step schedules do not share the frozen endpoint")
    step_magnitudes = sorted(
        {
            abs(_canonical_decimal(schedules[reference]["step_size"], "Q5 step"))
            for reference in q5_schedules
        },
        reverse=True,
    )
    if len(step_magnitudes) < 3 or any(
        step_magnitudes[index] != Decimal(2) * step_magnitudes[index + 1]
        for index in range(len(step_magnitudes) - 1)
    ):
        _fail("q5_step_halving", "Q5 requires an exact step-halving grid")
    checkpoints = [
        _canonical_decimal(value, "Q5 checkpoint")
        for value in q5["checkpoint_epochs"]
    ]
    if common_endpoint not in checkpoints:
        _fail("q5_checkpoint_endpoint", "Q5 checkpoints must include the common endpoint")
    minimum_order = _canonical_decimal(q5["minimum_order"], "Q5 minimum order")
    maximum_order = _canonical_decimal(q5["maximum_order"], "Q5 maximum order")
    if not (minimum_order <= Decimal(2) <= maximum_order and minimum_order < maximum_order):
        _fail("q5_order_interval", "Q5 order interval must contain the theoretical order two")
    for position, pair in enumerate(q5["solver_tightening_pairs"]):
        baseline_ref = pair["baseline_schedule_ref"]
        tightened_ref = pair["tightened_schedule_ref"]
        if baseline_ref not in schedules or tightened_ref not in schedules:
            _fail("unknown_reference", f"Q5 solver pair {position} uses unknown schedule")
        baseline = schedules[baseline_ref]
        tightened = schedules[tightened_ref]
        fixed_fields = (
            "fixture_ref",
            "force_model_set",
            "decimal_context_ref",
            "step_size",
            "step_count",
            "initial_guess_policy",
            "method_id",
        )
        if any(baseline[field] != tightened[field] for field in fixed_fields):
            _fail("q5_solver_pair", "Q5 solver tightening changes its state grid")
        tolerance_fields = ("position_atol", "velocity_atol", "relative_tolerance")
        baseline_tolerances = [
            _canonical_decimal(baseline[field], f"Q5 baseline {field}")
            for field in tolerance_fields
        ]
        tightened_tolerances = [
            _canonical_decimal(tightened[field], f"Q5 tightened {field}")
            for field in tolerance_fields
        ]
        if not all(
            tightened_value <= baseline_value
            for tightened_value, baseline_value in zip(
                tightened_tolerances, baseline_tolerances
            )
        ) or tightened_tolerances == baseline_tolerances:
            _fail("q5_solver_tightening", "Q5 tightened tolerances are not strictly tighter")
        if tightened["maximum_iterations"] < baseline["maximum_iterations"]:
            _fail("q5_solver_tightening", "Q5 tightening lowers maximum iterations")
        if (
            baseline_ref not in q5_schedules
            or _schedule_endpoint(tightened, fixtures) != common_endpoint
        ):
            _fail("q5_solver_coverage", "Q5 solver pair is outside the frozen grid or common endpoint")
    q5_baselines = [
        pair["baseline_schedule_ref"] for pair in q5["solver_tightening_pairs"]
    ]
    if set(q5_baselines) != set(q5_schedules) or len(q5_baselines) != len(set(q5_baselines)):
        _fail("q5_solver_coverage", "Q5 solver-tightening pairs must cover every step-grid schedule exactly")

    q6 = inputs["q6_precision_cross_grid"]
    if q6["fixture_ref"] not in fixtures:
        _fail("unknown_reference", "Q6 references unknown fixture")
    q6_endpoint = _canonical_decimal(q6["common_endpoint"], "Q6 common endpoint")
    q6_steps = tuple(q6["step_sizes"])
    q6_contexts = tuple(q6["precision_context_refs"])
    expected_cells = {(step, context) for step in q6_steps for context in q6_contexts}
    observed_cells: dict[tuple[str, str], str] = {}
    for cell in q6["cross_grid_cells"]:
        key = (cell["step_size"], cell["decimal_context_ref"])
        if key in observed_cells:
            _fail("q6_duplicate_cell", f"Q6 repeats cross-grid cell {key}")
        schedule_ref = cell["schedule_ref"]
        if schedule_ref not in schedules:
            _fail("unknown_reference", "Q6 cross-grid cell uses unknown schedule")
        schedule = schedules[schedule_ref]
        if (
            schedule["fixture_ref"] != q6["fixture_ref"]
            or schedule["step_size"] != cell["step_size"]
            or schedule["decimal_context_ref"] != cell["decimal_context_ref"]
            or _schedule_endpoint(schedule, fixtures) != q6_endpoint
        ):
            _fail("q6_cell_binding", "Q6 cross-grid cell differs from its schedule")
        observed_cells[key] = schedule_ref
    if set(observed_cells) != expected_cells:
        _fail("q6_cartesian_grid", "Q6 h-by-precision Cartesian grid is incomplete")
    q5_step_to_schedule = {
        abs(_canonical_decimal(schedules[reference]["step_size"], "Q5/Q6 step")): schedules[reference]
        for reference in q5_schedules
    }
    q6_step_values = {
        abs(_canonical_decimal(value, "Q6 step")) for value in q6_steps
    }
    if (
        q6["fixture_ref"] not in q5_fixture_ids
        or q6_endpoint != common_endpoint
        or q6_step_values != set(q5_step_to_schedule)
    ):
        _fail(
            "q5_q6_shared_grid",
            "Q6 must use the exact Q5 fixture, endpoint, and step grid",
        )
    q5_q6_fixed_fields = (
        "fixture_ref",
        "force_model_set",
        "step_count",
        "position_atol",
        "velocity_atol",
        "relative_tolerance",
        "maximum_iterations",
        "initial_guess_policy",
        "method_id",
    )
    for cell in q6["cross_grid_cells"]:
        schedule = schedules[cell["schedule_ref"]]
        q5_schedule = q5_step_to_schedule[
            abs(_canonical_decimal(cell["step_size"], "Q5/Q6 cell step"))
        ]
        if any(schedule[field] != q5_schedule[field] for field in q5_q6_fixed_fields):
            _fail(
                "q5_q6_cell_policy",
                "Q6 cell differs from its Q5 baseline by more than Decimal precision",
            )

    q7 = inputs["q7_transform_contract"]
    lb_ref = q7["lb_derivation_ref"]
    scaling_ref = q7["scale_factor_derivation_ref"]
    if lb_ref not in derivations or scaling_ref not in derivations:
        _fail("unknown_reference", "Q7 scaling uses unknown derivation")
    lb_results = _derivation_result_records(derivations[lb_ref])
    scaling = derivations[scaling_ref]
    if "l_b" not in lb_results or scaling["formula"] != "TCB_TDB_LINEAR_SCALING":
        _fail("q7_scaling_derivation", "Q7 does not bind a retained L_B scaling derivation")
    scaling_operands = {record["name"]: record for record in scaling["operands"]}
    scaling_results = _derivation_result_records(scaling)
    if (
        "l_b" not in scaling_operands
        or scaling_operands["l_b"]["value"] != lb_results["l_b"]["value"]
        or scaling_operands["l_b"]["unit_id"] != lb_results["l_b"]["unit_id"]
        or _canonical_decimal(lb_results["l_b"]["value"], "Q7 retained L_B") != _IAU_L_B
        or _canonical_decimal(scaling_results["scale_factor"]["value"], "Q7 F")
        != Decimal(1) - _canonical_decimal(lb_results["l_b"]["value"], "Q7 L_B")
    ):
        _fail("q7_scale_factor", "Q7 scale factor is not exactly F=1-L_B")
    for label in ("seconds_per_day", "t0", "tdb0"):
        binding = q7[label]
        _validate_result_binding(
            value=binding["value"],
            unit_id=binding["unit_id"],
            derivation_ref=binding["derivation_ref"],
            result_name=binding["derivation_result_name"],
            derivations=derivations,
            context=f"Q7 {label}",
        )
    if _canonical_decimal(q7["seconds_per_day"]["value"], "Q7 seconds_per_day") != _SECONDS_PER_DAY:
        _fail("q7_seconds_per_day", "Q7 must bind the exact retained 86400 seconds per day")
    if (
        _canonical_decimal(q7["t0"]["value"], "Q7 T0") != _IAU_T0_JULIAN_DATE
        or _canonical_decimal(q7["tdb0"]["value"], "Q7 TDB0") != _IAU_TDB0_SECONDS
    ):
        _fail("q7_affine_constants", "Q7 must bind the exact retained IAU T0 and TDB0 constants")
    transform_kinds = [pair["transform_kind"] for pair in q7["transform_pairs"]]
    if transform_kinds.count("EXACT_UNIT_CHANGE") != 1 or transform_kinds.count("TCB_TO_TDB_COMPATIBLE_SCALING") != 1 or len(transform_kinds) != 2:
        _fail("q7_transform_roster", "Q7 requires exactly one exact-unit and one TCB/TDB transform pair")
    q7_fixture_ids: set[str] = set()
    for position, pair in enumerate(q7["transform_pairs"]):
        for field in ("source_fixture_ref", "transformed_fixture_ref"):
            if pair[field] not in fixtures:
                _fail("unknown_reference", f"Q7 transform pair {position} uses unknown fixture")
        for field in ("source_schedule_ref", "transformed_schedule_ref"):
            if pair[field] not in schedules:
                _fail("unknown_reference", f"Q7 transform pair {position} uses unknown schedule")
        source_fixture = fixtures[pair["source_fixture_ref"]]
        transformed_fixture = fixtures[pair["transformed_fixture_ref"]]
        q7_fixture_ids.update(
            (pair["source_fixture_ref"], pair["transformed_fixture_ref"])
        )
        source_schedule = schedules[pair["source_schedule_ref"]]
        transformed_schedule = schedules[pair["transformed_schedule_ref"]]
        if (
            source_schedule["fixture_ref"] != source_fixture["fixture_id"]
            or transformed_schedule["fixture_ref"] != transformed_fixture["fixture_id"]
        ):
            _fail("q7_schedule_binding", "Q7 schedule does not bind its transform fixture")
        source_coordinate = coordinates[source_fixture["coordinate_contract_ref"]]
        transformed_coordinate = coordinates[
            transformed_fixture["coordinate_contract_ref"]
        ]
        source_coefficient = coefficients[source_fixture["coefficient_set_ref"]]
        transformed_coefficient = coefficients[
            transformed_fixture["coefficient_set_ref"]
        ]
        length_scale = _canonical_decimal(pair["length_scale"], "Q7 length scale")
        time_scale = _canonical_decimal(pair["time_scale"], "Q7 time scale")
        acceleration_scale = _canonical_decimal(
            pair["acceleration_scale"], "Q7 acceleration scale"
        )
        if length_scale <= 0 or time_scale <= 0:
            _fail("q7_transform_scale", "Q7 transform scales must be positive")
        factor = _canonical_decimal(
            scaling_results["scale_factor"]["value"], "Q7 F"
        )
        if pair["transform_kind"] == "TCB_TO_TDB_COMPATIBLE_SCALING":
            if (
                source_coordinate["time_scale"] != "TCB_COMPATIBLE"
                or transformed_coordinate["time_scale"] != "TDB_COMPATIBLE"
                or source_coordinate["units"] != transformed_coordinate["units"]
                or source_coordinate["epoch_kind"]
                != "DECLARED_COORDINATE_EPOCH"
                or transformed_coordinate["epoch_kind"]
                != "DECLARED_COORDINATE_EPOCH"
                or length_scale != factor
                or time_scale != factor
            ):
                _fail("q7_tcb_tdb_contract", "Q7 TCB/TDB transform does not use F=1-L_B")
        else:
            expected_length_scale, expected_time_scale = _numeric_unit_transform_scales(
                source_coordinate["units"], transformed_coordinate["units"]
            )
            if (
                source_coordinate["time_scale"] != transformed_coordinate["time_scale"]
                or source_coordinate["units"] == transformed_coordinate["units"]
                or length_scale != expected_length_scale
                or time_scale != expected_time_scale
            ):
                _fail("q7_unit_contract", "exact unit transform scale or coordinate-time binding differs")
        if (
            source_coordinate["epoch_kind"] != transformed_coordinate["epoch_kind"]
            or "TRANSFORM_TWIN" not in source_fixture["case_roles"]
            or "TRANSFORM_TWIN" not in transformed_fixture["case_roles"]
        ):
            _fail("q7_transform_role", "Q7 transform pair is not an epoch-compatible transform twin")
        with localcontext(contexts["decimal.context.precision_90"].make_context()):
            velocity_scale = length_scale / time_scale
            mu_scale = length_scale ** 3 / time_scale ** 2
            expected_acceleration_scale = length_scale / time_scale ** 2
            if acceleration_scale != expected_acceleration_scale:
                _fail("q7_acceleration_transform", "Q7 acceleration scale differs from s_L/s_T^2")
            for source_value, transformed_value in zip(
                source_fixture["position"], transformed_fixture["position"]
            ):
                source_number = _canonical_decimal(
                    source_value, "Q7 source position"
                )
                expected_position = (
                    _exact_unit_transform_value(
                        source_number,
                        source_units=source_coordinate["units"],
                        target_units=transformed_coordinate["units"],
                        quantity="LENGTH",
                    )
                    if pair["transform_kind"] == "EXACT_UNIT_CHANGE"
                    else length_scale * source_number
                )
                if _canonical_decimal(
                    transformed_value, "Q7 transformed position"
                ) != expected_position:
                    _fail("q7_position_transform", "Q7 position transform differs")
            for source_value, transformed_value in zip(
                source_fixture["velocity"], transformed_fixture["velocity"]
            ):
                source_number = _canonical_decimal(
                    source_value, "Q7 source velocity"
                )
                expected_velocity = (
                    _exact_unit_transform_value(
                        source_number,
                        source_units=source_coordinate["units"],
                        target_units=transformed_coordinate["units"],
                        quantity="VELOCITY",
                    )
                    if pair["transform_kind"] == "EXACT_UNIT_CHANGE"
                    else velocity_scale * source_number
                )
                if _canonical_decimal(
                    transformed_value, "Q7 transformed velocity"
                ) != expected_velocity:
                    _fail("q7_velocity_transform", "Q7 velocity transform differs")
            transformed_epoch = _canonical_decimal(transformed_fixture["epoch"], "Q7 transformed epoch")
            source_epoch = _canonical_decimal(source_fixture["epoch"], "Q7 source epoch")
            if pair["transform_kind"] == "EXACT_UNIT_CHANGE":
                expected_epoch = _exact_unit_transform_value(
                    source_epoch,
                    source_units=source_coordinate["units"],
                    target_units=transformed_coordinate["units"],
                    quantity="TIME",
                )
            elif source_coordinate["epoch_kind"] == "SYNTHETIC_COORDINATE_OFFSET":
                expected_epoch = time_scale * source_epoch
            elif pair["transform_kind"] == "TCB_TO_TDB_COMPATIBLE_SCALING":
                lb_value = _canonical_decimal(lb_results["l_b"]["value"], "Q7 L_B")
                t0_value = _canonical_decimal(q7["t0"]["value"], "Q7 T0")
                tdb0_value = _canonical_decimal(q7["tdb0"]["value"], "Q7 TDB0")
                expected_epoch = _tcb_to_tdb_declared_epoch(
                    source_epoch,
                    units=source_coordinate["units"],
                    factor=factor,
                    l_b=lb_value,
                    t0_julian_date=t0_value,
                    tdb0_seconds=tdb0_value,
                )
            else:
                expected_epoch = time_scale * source_epoch
            if transformed_epoch != expected_epoch:
                _fail("q7_epoch_transform", "Q7 affine or exact-unit epoch transform differs")
            expected_coefficient_quantities = {
                "gravitational_parameter": (mu_scale, "GRAVITATIONAL_PARAMETER"),
                "speed_of_light": (velocity_scale, "VELOCITY"),
                "maximum_compactness": (Decimal(1), None),
                "maximum_speed_fraction_squared": (Decimal(1), None),
            }
            for field, (scale, quantity) in expected_coefficient_quantities.items():
                source_number = _canonical_decimal(
                    source_coefficient[field]["value"], f"Q7 source {field}"
                )
                expected_value = (
                    _exact_unit_transform_value(
                        source_number,
                        source_units=source_coordinate["units"],
                        target_units=transformed_coordinate["units"],
                        quantity=quantity,
                    )
                    if pair["transform_kind"] == "EXACT_UNIT_CHANGE"
                    and quantity is not None
                    else scale * source_number
                )
                if _canonical_decimal(
                    transformed_coefficient[field]["value"],
                    f"Q7 transformed {field}",
                ) != expected_value:
                    _fail("q7_coefficient_transform", f"Q7 {field} transform differs")
            schedule_scales = {
                "step_size": (time_scale, "TIME"),
                "position_atol": (length_scale, "LENGTH"),
                "velocity_atol": (velocity_scale, "VELOCITY"),
            }
            for field, (scale, quantity) in schedule_scales.items():
                source_number = _canonical_decimal(
                    source_schedule[field], f"Q7 source schedule {field}"
                )
                expected_value = (
                    _exact_unit_transform_value(
                        source_number,
                        source_units=source_coordinate["units"],
                        target_units=transformed_coordinate["units"],
                        quantity=quantity,
                    )
                    if pair["transform_kind"] == "EXACT_UNIT_CHANGE"
                    else scale * source_number
                )
                if _canonical_decimal(
                    transformed_schedule[field], f"Q7 transformed schedule {field}"
                ) != expected_value:
                    _fail("q7_schedule_transform", f"Q7 schedule {field} transform differs")
        for field in (
            "step_count",
            "relative_tolerance",
            "maximum_iterations",
            "decimal_context_ref",
            "force_model_set",
            "initial_guess_policy",
            "method_id",
        ):
            if source_schedule[field] != transformed_schedule[field]:
                _fail("q7_schedule_policy", f"Q7 schedule policy changes {field}")

    signal_groups: dict[str, list[tuple[str, str]]] = {}
    for fixture_id, fixture in fixtures.items():
        group_digest, normalized_c = _signal_scale_group_sha256(
            fixture,
            coefficients[fixture["coefficient_set_ref"]],
            coordinates[fixture["coordinate_contract_ref"]],
        )
        signal_groups.setdefault(group_digest, []).append(
            (fixture_id, normalized_c)
        )
    eligible_signal_fixture_ids: set[str] = set()
    for members in signal_groups.values():
        if len({normalized_c for _, normalized_c in members}) < 2:
            continue
        member_ids = {fixture_id for fixture_id, _ in members}
        if all(
            fixture_diagnostics[fixture_id][0] == "DOMAIN_INSIDE"
            and fixture_diagnostics[fixture_id][2]
            for fixture_id in member_ids
        ):
            eligible_signal_fixture_ids.update(member_ids)
    q3_q6_fixture_ids = q3_fixture_ids | {q6["fixture_ref"]}
    if not eligible_signal_fixture_ids:
        _fail(
            "signal_scale_roster",
            "qualification inputs require at least one inside-domain nonzero signal family at two or more c values",
        )
    if not eligible_signal_fixture_ids.issubset(q3_q6_fixture_ids):
        _fail(
            "signal_scale_binding",
            "every derived signal-scale fixture must be bound to Q3 or Q6",
        )
    for fixture_id, fixture in fixtures.items():
        roles = set(fixture["case_roles"])
        exact_role_memberships = {
            "ECCENTRIC": fixture_id in q2_fixture_ids,
            "LONG_ARC": fixture_id in q2_fixture_ids,
            "TRANSFORM_TWIN": fixture_id in q7_fixture_ids,
            "SIGNAL_SCALE": fixture_id in eligible_signal_fixture_ids,
            "EIH_LIMIT": fixture_id == q4_fixture_ref,
        }
        for role, expected in exact_role_memberships.items():
            if (role in roles) != expected:
                _fail(
                    "derived_case_role",
                    f"fixture {fixture_id} role {role} differs from its frozen computational binding",
                )

    q8 = inputs["q8_holdout_custody"]
    holdout_refs = _check_refs(
        q8["holdout_fixture_refs"], fixtures, "Q8 holdout fixtures", allow_empty=False
    )
    if set(holdout_refs) != set(fixtures):
        _fail("q8_holdout_roster", "Q8 custody must cover every holdout fixture")
    if tuple(q8["required_case_roles"]) != CASE_ROLES:
        _fail("q8_case_role_roster", "Q8 required scientific case-role roster differs")
    observed_holdout_roles = {
        role for fixture_ref in holdout_refs for role in fixtures[fixture_ref]["case_roles"]
    }
    if observed_holdout_roles != set(CASE_ROLES):
        _fail("q8_case_role_coverage", "Q8 holdouts do not cover the exact required scientific case roles")

    q9 = inputs["q9_error_budgets"]
    observable_records = _index(q9["observables"], "observable_id", "Q9 observables")
    if set(observable_records) != set(OBSERVABLE_IDS):
        _fail("q9_observable_roster", "Q9 observable budget roster differs")
    for observable_id, observable in observable_records.items():
        if observable["unit_ref"] not in units:
            _fail("unknown_reference", f"Q9 observable {observable_id} has unknown unit")


def validate_qualification_plan_semantics(
    plan: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> None:
    """Validate plan bindings, disclosed prior evidence, arms, and gates."""

    _reject_binary_float(plan, "qualification plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("state") != PLAN_STATE:
        _fail("plan_identity", "qualification plan schema or state differs")
    root = Path(project_root).resolve()
    expected_identity_sha256 = qualification_plan_identity_sha256(plan)
    expected_qualification_id = QUALIFICATION_ID_PREFIX + expected_identity_sha256
    if plan["qualification_id"] != expected_qualification_id:
        _fail("qualification_id_digest", "qualification_id does not bind plan content")
    _require_utc(plan["registered_utc"], "plan.registered_utc")
    foundation = _require_git_commit(plan["foundation_commit"], "plan.foundation_commit")
    if foundation != FOUNDATION_COMMIT:
        _fail("foundation_commit_mismatch", "plan foundation commit differs from the frozen commit")
    _verify_foundation_git_object(root)
    prior = plan["prior_development_evidence"]
    if _require_git_commit(prior["development_commit"], "prior development commit") != foundation:
        _fail("foundation_commit_mismatch", "plan and disclosed development commits differ")
    if prior["excluded_from_decisive_gates"] is not True:
        _fail("prior_evidence_boundary", "known development fixtures must be excluded")
    if plan["scope"]["registry_authorized"] is not False:
        _fail("authorization_boundary", "the qualification scope cannot authorize a registry")
    if plan["all_gates_conjunctive"] is not True:
        _fail("gate_policy", "all Q0-Q9 gates must remain conjunctive")

    input_binding = plan["bindings"]["inputs"]
    input_path, _ = _verify_artifact(root, input_binding, "plan input binding")
    if input_binding["schema"] != inputs["schema"] or input_binding["input_id"] != inputs["input_id"]:
        _fail("input_binding", "plan input identity differs from the frozen input document")
    if _normalized(_load_json(input_path, "plan-bound inputs")) != _normalized(inputs):
        _fail("input_binding", "plan-bound input bytes do not contain the inspected inputs")
    _verify_canonical_digest(inputs, input_binding, "plan input binding")

    manifest_binding = plan["bindings"]["prior_development_manifest"]
    manifest_path, manifest_record = _verify_artifact(
        root, manifest_binding, "plan prior-development manifest binding"
    )
    if (
        manifest_binding["role"] != "prior_development_cases_manifest"
        or manifest_record["path"] != PRIOR_DEVELOPMENT_MANIFEST_PATH
        or manifest_binding["schema"] != PRIOR_DEVELOPMENT_SCHEMA
        or manifest_binding["manifest_id"] != PRIOR_DEVELOPMENT_MANIFEST_ID
        or manifest_binding["sha256"] != PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256
        or manifest_binding["size_bytes"] != PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES
        or manifest_binding["canonical_sha256"]
        != PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256
    ):
        _fail(
            "prior_manifest_binding",
            "plan prior-development manifest identity or frozen bytes differ",
        )
    manifest = _load_json(manifest_path, "plan-bound prior-development manifest")
    prior_schema_path = _safe_repository_file(
        root,
        _SCHEMA_PATHS[PRIOR_DEVELOPMENT_SCHEMA],
        "prior-development manifest schema",
    )
    prior_schema = _load_json(
        prior_schema_path, "prior-development manifest schema"
    )
    known_fingerprints, generic_numeric_fingerprints = (
        validate_prior_development_manifest_semantics(
            manifest,
            prior_schema,
            manifest_path=manifest_path,
            project_root=root,
        )
    )
    known_transform_fingerprints = _prior_transform_invariant_fingerprints(
        manifest, project_root=root
    )
    _verify_canonical_digest(
        manifest, manifest_binding, "plan prior-development manifest binding"
    )

    registry_binding = plan["bindings"]["registry"]
    registry_path, _ = _verify_artifact(root, registry_binding, "plan registry binding")
    registry, inspection = inspect_registry_file(registry_path, project_root=root)
    if (
        inspection.registry_sha256 != registry_binding["canonical_sha256"]
        or inspection.registry_id != registry_binding["registry_id"]
        or inspection.registry_state != registry_binding["state"]
        or inspection.execution_authorized
    ):
        _fail("registry_binding", "plan registry identity, digest, state, or authority differs")
    models = {item["model_id"]: item for item in registry["force_models"]}
    solar = models.get("relativity.solar_schwarzschild_test_particle_1pn")
    if solar is None or (
        solar["implementation_status"] != "NOT_IMPLEMENTED"
        or solar["qualification_status"] != "UNQUALIFIED"
        or solar["treatment"] != "BLOCKED"
        or solar["qualification_refs"]
    ):
        _fail("registry_model_boundary", "Solar 1PN registry row is no longer blocked and unqualified")

    contract_path, _ = _verify_artifact(
        root, plan["bindings"]["scientific_contract"], "scientific contract binding"
    )
    framework = registry["framework"]
    if (
        sha256_file(contract_path) != framework["scientific_contract_sha256"]
        or str(contract_path.stat().st_size) != framework["scientific_contract_size_bytes"]
    ):
        _fail("scientific_contract_binding", "plan and registry scientific contracts differ")

    locked_roles: set[str] = set()
    locked_paths: set[str] = set()
    _require_role_path_map(
        plan["bindings"]["locked_files"],
        LOCKED_FILE_ROLE_PATHS,
        "locked_files",
    )
    for position, binding in enumerate(plan["bindings"]["locked_files"]):
        role = _require_identifier(binding["role"], f"locked_files[{position}].role")
        if role in locked_roles or binding["path"] in locked_paths:
            _fail("duplicate_locked_file", "locked file roles and paths must be unique")
        locked_roles.add(role)
        locked_paths.add(binding["path"])
        _verify_artifact(root, binding, f"locked file {role}")
    expected_known_files = [
        {"role": source["source_role"], **source["current_file"]}
        for source in manifest["foundation_sources"]
    ]
    if prior["known_files"] != expected_known_files:
        _fail(
            "prior_manifest_disclosure",
            "plan known-file disclosures differ from the byte-anchored manifest",
        )
    _require_role_path_map(prior["known_files"], KNOWN_FILE_ROLE_PATHS, "known_files")
    known_paths: set[str] = set()
    for position, binding in enumerate(prior["known_files"]):
        role = _require_identifier(binding["role"], f"known_files[{position}].role")
        if binding["path"] in known_paths:
            _fail("duplicate_prior_file", "known prior-evidence paths must be unique")
        known_paths.add(binding["path"])
        _verify_artifact(root, binding, f"known development file {role}")

    fixtures = _index(inputs["fixtures"], "fixture_id", "fixtures")
    if (
        prior["known_fixture_ids"] != manifest["known_fixture_ids"]
        or prior["known_state_records"] != manifest["known_state_records"]
        or prior["known_state_record_count"]
        != PRIOR_DEVELOPMENT_COUNTS["known_state_record_count"]
        or prior["manifest_disclosure_sha256"]
        != manifest["manifest_disclosure_sha256"]
        or prior["known_case_descriptor_sha256s"]
        != manifest["known_case_descriptor_sha256s"]
        or prior["generated_family_fingerprint_list_sha256s"]
        != manifest["generated_family_fingerprint_list_sha256s"]
    ):
        _fail(
            "prior_manifest_disclosure",
            "plan prior-evidence rosters or digests differ from the frozen manifest",
        )
    known_fixture_ids = set(
        _string_tuple(prior["known_fixture_ids"], "known fixture IDs", allow_empty=False)
    )
    overlap = known_fixture_ids & set(fixtures)
    if overlap:
        _fail("holdout_reuse", f"holdout fixture IDs reuse known development IDs: {sorted(overlap)}")

    known_state_records = _index(
        prior["known_state_records"], "record_id", "known_state_records"
    )
    if (
        set(known_state_records) != set(KNOWN_STATE_RECORD_SPECS)
        or prior["known_state_record_count"] != len(KNOWN_STATE_RECORD_SPECS)
        or len(known_state_records) != prior["known_state_record_count"]
    ):
        _fail(
            "known_state_roster",
            "known development state record IDs and exact frozen count differ",
        )
    for record_id, record in known_state_records.items():
        if (record["source_role"], record["source_locator"]) != (
            KNOWN_STATE_RECORD_SPECS[record_id]
        ):
            _fail(
                "known_state_locator",
                f"known state {record_id} does not bind its exact test-byte locator",
            )
        if len(record["position"]) != 3 or len(record["velocity"]) != 3:
            _fail("known_state_shape", f"known state {record_id} is not a six-vector")
        _canonical_decimal(record["epoch"], f"known state {record_id}.epoch")
        for field in ("position", "velocity"):
            for index, value in enumerate(record[field]):
                _canonical_decimal(value, f"known state {record_id}.{field}[{index}]")
        observed = _known_state_fingerprint(record)
        if observed != record["scientific_fingerprint_sha256"]:
            _fail("known_state_fingerprint", f"known state {record_id} fingerprint differs")
    input_coordinates = _index(
        inputs["coordinate_contracts"],
        "coordinate_contract_id",
        "coordinate_contracts",
    )
    input_coefficients = _index(
        inputs["coefficient_sets"], "coefficient_set_id", "coefficient_sets"
    )
    transform_graph: dict[str, set[str]] = {fixture_id: set() for fixture_id in fixtures}
    for pair in inputs["q7_transform_contract"]["transform_pairs"]:
        source_fixture_ref = pair["source_fixture_ref"]
        transformed_fixture_ref = pair["transformed_fixture_ref"]
        transform_graph.setdefault(source_fixture_ref, set()).add(
            transformed_fixture_ref
        )
        transform_graph.setdefault(transformed_fixture_ref, set()).add(
            source_fixture_ref
        )
    holdout_fingerprints: dict[str, set[str]] = {}
    holdout_prior_fingerprints: set[str] = set()
    for fixture_id, fixture in fixtures.items():
        coefficient = input_coefficients[fixture["coefficient_set_ref"]]
        coordinate = input_coordinates[fixture["coordinate_contract_ref"]]
        holdout_prior_fingerprints.add(
            qualification_scientific_fingerprint_sha256(
                fixture, coefficient, coordinate
            )
        )
        fingerprint = _transform_invariant_scientific_fingerprint_sha256(
            fixture,
            coefficient,
            coordinate,
        )
        holdout_fingerprints.setdefault(fingerprint, set()).add(fixture_id)
    for equivalent_fixture_ids in holdout_fingerprints.values():
        if len(equivalent_fixture_ids) < 2:
            continue
        if (
            any(
                "TRANSFORM_TWIN" not in fixtures[fixture_id]["case_roles"]
                for fixture_id in equivalent_fixture_ids
            )
            or not _connected_by_declared_transform_pairs(
                equivalent_fixture_ids, transform_graph
            )
        ):
            _fail(
                "duplicate_holdout_state",
                "scientifically equivalent holdouts must all be TRANSFORM_TWIN "
                "fixtures connected by the declared Q7 transform graph",
            )
    reused = sorted(
        (holdout_prior_fingerprints & known_fingerprints)
        | (set(holdout_fingerprints) & known_transform_fingerprints)
    )
    if reused:
        _fail("holdout_state_reuse", "a holdout reuses a disclosed scientific state")
    reused_numeric = sorted(
        {
            _holdout_numeric_state_fingerprint(
                fixture,
                input_coordinates[fixture["coordinate_contract_ref"]],
            )
            for fixture in fixtures.values()
        }
        & generic_numeric_fingerprints
    )
    if reused_numeric:
        _fail(
            "holdout_numeric_state_reuse",
            "a holdout reuses a disclosed generic-solver numeric state",
        )

    arms = _index(plan["test_arms"], "arm_id", "test_arms")
    gates = _index(plan["gates"], "gate_id", "gates")
    if set(arms) != set(ARM_IDS) or set(gates) != set(GATE_IDS):
        _fail("qualification_matrix_roster", "plan must contain exactly Q0-Q9 arms and gates")
    arm_to_gate = dict(ARM_GATE_MATRIX)
    gate_to_arm = {gate: arm for arm, gate in ARM_GATE_MATRIX}
    input_units = _index(inputs["unit_definitions"], "unit_id", "unit_definitions")
    input_blockers = _index(
        inputs["blocked_artifacts"], "placeholder_id", "blocked_artifacts"
    )
    used_fixtures: set[str] = set()
    for arm_id, arm in arms.items():
        arm_spec = ARM_SPECS[arm_id]
        for field in (
            "purpose_code",
            "purpose",
            "implementation_independence",
        ):
            if arm[field] != arm_spec[field]:
                _fail(
                    "arm_semantic_spec",
                    f"arm {arm_id}.{field} differs from its exact frozen specification",
                )
        for field in ("required_case_roles", "observable_refs", "budget_refs"):
            if tuple(arm[field]) != tuple(arm_spec[field]):
                _fail(
                    "arm_semantic_spec",
                    f"arm {arm_id}.{field} differs from its exact frozen roster",
                )
        fixture_refs = _check_refs(
            arm["fixture_refs"], fixtures, f"arm {arm_id}.fixture_refs"
        )
        if arm["implementation_independence"] != "PROVENANCE_ONLY" and not fixture_refs:
            _fail("empty_arm_fixture", f"arm {arm_id} requires a holdout fixture")
        used_fixtures.update(fixture_refs)
        observed_roles = {
            role for fixture_ref in fixture_refs for role in fixtures[fixture_ref]["case_roles"]
        }
        if not set(arm_spec["required_case_roles"]).issubset(observed_roles):
            _fail(
                "arm_case_role_coverage",
                f"arm {arm_id} does not cover every required scientific case role",
            )
        gate_refs = _check_refs(
            arm["gate_refs"], gates, f"arm {arm_id}.gate_refs", allow_empty=False
        )
        expected_gate = arm_to_gate[arm_id]
        if set(gate_refs) != {expected_gate}:
            _fail("qualification_matrix", f"arm {arm_id} does not bind only {expected_gate}")
        blocked_refs = _check_refs(
            arm["blocked_artifact_refs"],
            input_blockers,
            f"arm {arm_id}.blocked_artifact_refs",
            allow_empty=False,
        )
        if set(blocked_refs) != _blockers_for_gate(expected_gate):
            _fail("arm_blocker_roster", f"arm {arm_id} blocker roster differs")
        if arm["readiness_state"] != "BLOCKED":
            _fail("arm_readiness", f"arm {arm_id} must remain blocked")
        for gate_id in gate_refs:
            if arm_id not in gates[gate_id]["arm_refs"]:
                _fail("asymmetric_gate_reference", f"arm {arm_id} and gate {gate_id} differ")
    for gate_id, gate in gates.items():
        gate_spec = GATE_SPECS[gate_id]
        for field in (
            "metric_code",
            "metric",
            "value_kind",
            "operator",
            "threshold",
            "unit",
            "failure_verdict",
        ):
            if gate[field] != gate_spec[field]:
                _fail(
                    "gate_semantic_spec",
                    f"gate {gate_id}.{field} differs from its exact frozen specification",
                )
        for field in ("observable_refs", "budget_refs"):
            if tuple(gate[field]) != tuple(gate_spec[field]):
                _fail(
                    "gate_semantic_spec",
                    f"gate {gate_id}.{field} differs from its exact frozen roster",
                )
        _validate_gate_threshold(gate)
        if gate["unit"] not in input_units:
            _fail("unknown_reference", f"gate {gate_id} uses unknown unit")
        if gate["value_kind"] in {"BOOLEAN", "INTEGER", "ENUM"} and input_units[
            gate["unit"]
        ]["quantity_dimension"] != UNIT_DIMENSIONS["dimensionless"]:
            _fail("gate_unit_dimension", f"gate {gate_id} requires a dimensionless unit")
        arm_refs = _check_refs(
            gate["arm_refs"], arms, f"gate {gate_id}.arm_refs", allow_empty=False
        )
        expected_arm = gate_to_arm[gate_id]
        if set(arm_refs) != {expected_arm}:
            _fail("qualification_matrix", f"gate {gate_id} does not bind only {expected_arm}")
        blocked_refs = _check_refs(
            gate["blocked_artifact_refs"],
            input_blockers,
            f"gate {gate_id}.blocked_artifact_refs",
            allow_empty=False,
        )
        if set(blocked_refs) != _blockers_for_gate(gate_id):
            _fail("gate_blocker_roster", f"gate {gate_id} blocker roster differs")
        if gate["readiness_state"] != "BLOCKED":
            _fail("gate_readiness", f"gate {gate_id} must remain blocked")
        for arm_id in arm_refs:
            if gate_id not in arms[arm_id]["gate_refs"]:
                _fail("asymmetric_gate_reference", f"gate {gate_id} and arm {arm_id} differ")
    if used_fixtures != set(fixtures):
        _fail("orphan_fixture", "every frozen holdout fixture must belong to a test arm")
    q2_cell_fixtures = {
        cell["fixture_ref"] for cell in inputs["q2_perihelion_observer"]["case_cells"]
    }
    q1_fixture_roster = {
        fixture_id
        for fixture_id, fixture in fixtures.items()
        if set(fixture["case_roles"])
        & {"GENERIC_3D", "DOMAIN_BOUNDARY", "DOMAIN_OUTSIDE"}
    }
    if set(arms["arm.q1_equation"]["fixture_refs"]) != q1_fixture_roster:
        _fail(
            "q1_arm_binding",
            "Q1 arm must bind every generic, boundary, and independent outside-domain fixture",
        )
    if set(arms["arm.q2_perihelion"]["fixture_refs"]) != q2_cell_fixtures:
        _fail("q2_arm_binding", "Q2 arm fixtures differ from the frozen Q2 case-cell roster")
    q3_cell_fixtures = {
        cell["fixture_ref"] for cell in inputs["q3_independent_oracle"]["case_cells"]
    }
    if set(arms["arm.q3_high_order_oracle"]["fixture_refs"]) != q3_cell_fixtures:
        _fail("q3_arm_binding", "Q3 arm fixtures differ from the frozen oracle-cell roster")
    if set(arms["arm.q4_restricted_eih"]["fixture_refs"]) != {
        inputs["q4_eih_evaluator"]["fixture_ref"]
    }:
        _fail(
            "q4_arm_binding",
            "Q4 arm must reference exactly the frozen EIH-limit fixture",
        )
    input_schedules = _index(
        inputs["integration_schedules"], "schedule_id", "integration_schedules"
    )
    q5_fixture_roster = {
        input_schedules[reference]["fixture_ref"]
        for reference in inputs["q5_convergence_grid"]["step_schedule_refs"]
    }
    if set(arms["arm.q5_step_solver"]["fixture_refs"]) != q5_fixture_roster:
        _fail("q5_arm_binding", "Q5 arm differs from its frozen schedule fixture")
    if set(arms["arm.q6_precision"]["fixture_refs"]) != {
        inputs["q6_precision_cross_grid"]["fixture_ref"]
    }:
        _fail("q6_arm_binding", "Q6 arm differs from its frozen cross-grid fixture")
    q7_fixture_roster = {
        fixture_ref
        for pair in inputs["q7_transform_contract"]["transform_pairs"]
        for fixture_ref in (
            pair["source_fixture_ref"],
            pair["transformed_fixture_ref"],
        )
    }
    if set(arms["arm.q7_transform"]["fixture_refs"]) != q7_fixture_roster:
        _fail("q7_arm_binding", "Q7 arm differs from its frozen transform fixtures")
    q8_fixture_roster = set(inputs["q8_holdout_custody"]["holdout_fixture_refs"])
    if set(arms["arm.q8_holdout"]["fixture_refs"]) != q8_fixture_roster:
        _fail("q8_arm_binding", "Q8 arm must bind the complete custody roster")
    if set(arms["arm.q9_claim_audit"]["fixture_refs"]) != q8_fixture_roster:
        _fail("q9_arm_binding", "Q9 arm must bind the complete holdout roster")
    if plan["execution_registration_policy"]["current_state"] != EXECUTION_IMPLEMENTATION_STATE:
        _fail("execution_state", "execution implementation must remain unregistered")
    if plan["result_semantics"] != {
        "missing_result_verdict": "BLOCKED",
        "skipped_result_verdict": "BLOCKED",
        "nonfinite_result_verdict": "FAIL",
        "gate_failure_verdict": "FAIL",
        "conflict_verdict": "CONFLICT",
    }:
        _fail("result_semantics", "missing, skipped, nonfinite, or failed result semantics differ")
    verdict_policy = plan["verdict_policy"]
    if set(verdict_policy["allowed_verdicts"]) != set(ALLOWED_VERDICTS):
        _fail("verdict_roster", "qualification verdict vocabulary differs")
    if verdict_policy["pass_effect"] != PASS_EFFECT:
        _fail("pass_effect", "a complete pass may grant review eligibility only")
    if verdict_policy["registry_status_after_pass"] != "UNCHANGED_DRAFT_NONEXECUTABLE_UNQUALIFIED":
        _fail("pass_authority", "a complete pass cannot change registry authority")
    if (
        verdict_policy["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or verdict_policy["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
        or inputs["claim_boundary"]["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or inputs["claim_boundary"]["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
    ):
        _fail("permitted_claim", "plan and inputs must bind the exact permitted claim")
    if tuple(verdict_policy["prohibited_claim_codes"]) != PROHIBITED_CLAIM_CODES:
        _fail("prohibited_claim_roster", "plan prohibited-claim code roster differs")
    if plan["nonclaim"] != NONCLAIM or inputs["claim_boundary"]["nonclaim"] != NONCLAIM:
        _fail("nonclaim", "plan and input nonclaim text differs")


def validate_qualification_registration_semantics(
    registration: Mapping[str, Any],
    plan: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> None:
    """Validate the byte-level registration that freezes plan and inputs."""

    _reject_binary_float(registration, "qualification registration")
    if (
        registration.get("schema") != REGISTRATION_SCHEMA
        or registration.get("state") != REGISTRATION_STATE
    ):
        _fail("registration_identity", "qualification registration schema or state differs")
    if registration["outcomes_generated"] is not False:
        _fail("outcomes_already_generated", "pre-execution registration cannot contain outcomes")
    if registration["execution_implementation_state"] != EXECUTION_IMPLEMENTATION_STATE:
        _fail("execution_state", "execution implementation is not in the required blocked state")
    if registration["registry_authorized"] is not False:
        _fail("authorization_boundary", "registration cannot authorize the registry")

    if registration["qualification_id"] != plan["qualification_id"]:
        _fail("qualification_id_mismatch", "registration and plan qualification IDs differ")
    plan_identity = qualification_plan_identity_sha256(plan)
    if (
        registration["plan_identity_sha256"] != plan_identity
        or registration["qualification_id"] != QUALIFICATION_ID_PREFIX + plan_identity
    ):
        _fail("qualification_id_digest", "registration does not bind the sentinelized plan identity")
    foundation = _require_git_commit(registration["foundation_commit"], "registration foundation commit")
    if foundation != FOUNDATION_COMMIT or foundation != plan["foundation_commit"]:
        _fail("foundation_commit_mismatch", "registration and plan commits differ")
    registered = _require_utc(registration["registered_utc"], "registration.registered_utc")
    planned = _require_utc(plan["registered_utc"], "plan.registered_utc")
    if registered < planned:
        _fail("registration_time", "registration predates its plan")

    prior = registration["known_prior_evidence"]
    plan_prior = plan["prior_development_evidence"]
    plan_known_fingerprints = sorted(
        {
            record["scientific_fingerprint_sha256"]
            for record in plan_prior["known_state_records"]
        }
    )
    if (
        prior["development_commit"] != plan_prior["development_commit"]
        or prior["disclosure_status"] != plan_prior["disclosure_status"]
        or prior["known_fixture_ids"] != plan_prior["known_fixture_ids"]
        or prior["known_state_fingerprints"] != plan_known_fingerprints
        or prior["manifest_disclosure_sha256"]
        != plan_prior["manifest_disclosure_sha256"]
        or prior["known_case_descriptor_sha256s"]
        != plan_prior["known_case_descriptor_sha256s"]
        or prior["generated_family_fingerprint_list_sha256s"]
        != plan_prior["generated_family_fingerprint_list_sha256s"]
        or prior["excluded_from_decisive_gates"] is not True
    ):
        _fail("prior_evidence_mismatch", "registration and plan prior-evidence disclosures differ")
    blocking_status = registration["blocking_status"]
    if (
        blocking_status["q0_q9_execution_status"] != "NOT_EXECUTED"
        or blocking_status["ready_for_holdout_execution"] is not False
        or tuple(blocking_status["unresolved_blocker_codes"]) != BLOCKED_REASONS
    ):
        _fail("blocking_status", "registration blocker status differs")
    if (
        tuple(registration["prohibited_claim_codes"]) != PROHIBITED_CLAIM_CODES
        or tuple(plan["verdict_policy"]["prohibited_claim_codes"])
        != PROHIBITED_CLAIM_CODES
        or tuple(inputs["claim_boundary"]["prohibited_claims"])
        != PROHIBITED_CLAIM_CODES
        or registration["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or registration["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
        or plan["verdict_policy"]["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or plan["verdict_policy"]["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
        or inputs["claim_boundary"]["permitted_claim_code"] != PERMITTED_CLAIM_CODE
        or inputs["claim_boundary"]["permitted_claim_text"] != PERMITTED_CLAIM_TEXT
        or registration["nonclaim"] != NONCLAIM
        or plan["nonclaim"] != NONCLAIM
        or inputs["claim_boundary"]["nonclaim"] != NONCLAIM
    ):
        _fail("claim_boundary", "registration, plan, and input claim boundaries differ")

    root = Path(project_root).resolve()
    plan_path, _ = _verify_artifact(root, registration["plan"], "registered plan")
    input_path, _ = _verify_artifact(root, registration["inputs"], "registered inputs")
    if registration["plan"]["schema"] != PLAN_SCHEMA:
        _fail("document_schema_binding", "registered plan schema differs")
    if registration["inputs"]["schema"] != INPUT_SCHEMA:
        _fail("document_schema_binding", "registered input schema differs")
    if _normalized(_load_json(plan_path, "registered plan")) != _normalized(plan):
        _fail("document_binding", "registered plan bytes differ from inspected plan")
    if _normalized(_load_json(input_path, "registered inputs")) != _normalized(inputs):
        _fail("document_binding", "registered input bytes differ from inspected inputs")
    _verify_canonical_digest(plan, registration["plan"], "registered plan")
    _verify_canonical_digest(inputs, registration["inputs"], "registered inputs")
    plan_input = plan["bindings"]["inputs"]
    for key in ("path", "sha256", "size_bytes", "canonical_sha256", "schema"):
        if plan_input[key] != registration["inputs"][key]:
            _fail("input_binding", "plan and registration input bindings differ")

    plan_manifest = plan["bindings"]["prior_development_manifest"]
    registered_manifest = registration["prior_development_manifest"]
    if registered_manifest != plan_manifest:
        _fail(
            "prior_manifest_binding",
            "plan and registration prior-development manifest bindings differ",
        )
    manifest_path, _ = _verify_artifact(
        root, registered_manifest, "registered prior-development manifest"
    )
    manifest = _load_json(manifest_path, "registered prior-development manifest")
    _verify_canonical_digest(
        manifest,
        registered_manifest,
        "registered prior-development manifest",
    )
    if (
        manifest["manifest_disclosure_sha256"]
        != prior["manifest_disclosure_sha256"]
        or manifest["known_fixture_ids"] != prior["known_fixture_ids"]
        or manifest["known_case_descriptor_sha256s"]
        != prior["known_case_descriptor_sha256s"]
        or manifest["generated_family_fingerprint_list_sha256s"]
        != prior["generated_family_fingerprint_list_sha256s"]
    ):
        _fail(
            "prior_manifest_disclosure",
            "registration prior-evidence rosters differ from the frozen manifest",
        )

    schema_bindings = registration["schemas"]
    observed_schemas: dict[str, str] = {}
    for position, binding in enumerate(schema_bindings):
        schema_name = binding["schema"]
        if schema_name in observed_schemas:
            _fail("duplicate_schema_binding", f"registration repeats schema {schema_name!r}")
        schema_path, record = _verify_artifact(
            root, binding, f"registered schema[{position}]"
        )
        schema_document = _load_json(schema_path, f"registered schema[{position}]")
        _verify_canonical_digest(
            schema_document, binding, f"registered schema[{position}]"
        )
        observed_schemas[schema_name] = record["path"]
    if set(observed_schemas) != set(_SCHEMA_PATHS):
        _fail("schema_roster", "registration must bind exactly the four qualification schemas")
    for schema_name, expected_path in _SCHEMA_PATHS.items():
        if observed_schemas[schema_name] != expected_path:
            _fail("schema_path_binding", f"schema {schema_name!r} is bound to the wrong path")


@dataclass(frozen=True)
class Solar1PNQualificationInspection:
    schema: str
    qualification_id: str
    plan_state: str
    input_state: str
    registration_state: str
    plan_file_sha256: str
    plan_canonical_sha256: str
    input_file_sha256: str
    input_canonical_sha256: str
    registration_file_sha256: str
    registration_size_bytes: str
    registration_canonical_sha256: str
    package_sha256: str
    registry_canonical_sha256: str
    registry_authorized: bool
    outcomes_generated: bool
    ready_for_holdout_execution: bool
    blocked_reasons: tuple[str, ...]
    arm_ids: tuple[str, ...]
    gate_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_solar_1pn_qualification_package(
    registration_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Solar1PNQualificationInspection,
]:
    """Inspect a frozen package without running or authorizing dynamics.

    The returned documents are canonicalized copies in the order plan, inputs,
    registration.  A valid package remains blocked from holdout execution until
    a later, separately hash-locked execution implementation is registered.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail("invalid_project_root", "project_root must be an existing directory")
    registration_source = _registration_file(root, registration_path)
    registration = _load_json(registration_source, "qualification registration")

    schemas: dict[str, dict[str, Any]] = {}
    for name, relative in _SCHEMA_PATHS.items():
        schema_path = _safe_repository_file(root, relative, f"schema {name}")
        schemas[name] = _load_json(schema_path, f"schema {name}")
    _validate_with_schema(
        registration, schemas[REGISTRATION_SCHEMA], "qualification registration"
    )

    plan_path, _ = _verify_artifact(root, registration["plan"], "registered plan")
    input_path, _ = _verify_artifact(root, registration["inputs"], "registered inputs")
    manifest_path, _ = _verify_artifact(
        root,
        registration["prior_development_manifest"],
        "registered prior-development manifest",
    )
    plan = _load_json(plan_path, "qualification plan")
    inputs = _load_json(input_path, "qualification inputs")
    manifest = _load_json(manifest_path, "prior-development manifest")
    _validate_with_schema(plan, schemas[PLAN_SCHEMA], "qualification plan")
    _validate_with_schema(inputs, schemas[INPUT_SCHEMA], "qualification inputs")
    _validate_with_schema(
        manifest,
        schemas[PRIOR_DEVELOPMENT_SCHEMA],
        "prior-development manifest",
    )

    validate_qualification_inputs_semantics(
        inputs,
        registry_path=plan["bindings"]["registry"]["path"],
        project_root=root,
    )
    validate_qualification_plan_semantics(plan, inputs, project_root=root)
    validate_qualification_registration_semantics(
        registration, plan, inputs, project_root=root
    )

    plan_normalized = _normalized(plan)
    input_normalized = _normalized(inputs)
    registration_normalized = _normalized(registration)
    locked: dict[str, dict[str, str]] = {}

    def retain(binding: Mapping[str, Any], context: str) -> None:
        _, record = _verify_artifact(root, binding, context)
        prior = locked.get(record["path"])
        if prior is not None and prior != record:
            _fail("inconsistent_artifact_binding", f"artifact {record['path']!r} has inconsistent bindings")
        locked[record["path"]] = record

    retain(registration["plan"], "registered plan")
    retain(registration["inputs"], "registered inputs")
    retain(
        registration["prior_development_manifest"],
        "registered prior-development manifest",
    )
    for binding in registration["schemas"]:
        retain(binding, f"registered schema {binding['schema']}")
    retain(plan["bindings"]["registry"], "plan registry")
    retain(plan["bindings"]["scientific_contract"], "scientific contract")
    for binding in plan["bindings"]["locked_files"]:
        retain(binding, f"locked file {binding['role']}")
    for binding in plan["prior_development_evidence"]["known_files"]:
        retain(binding, f"known development file {binding['role']}")
    for source in manifest["foundation_sources"]:
        retain(
            source["foundation_copy"],
            f"foundation development file {source['source_role']}",
        )
        retain(
            source["current_file"],
            f"current development file {source['source_role']}",
        )
    for family in manifest["generated_state_families"]:
        retain(
            family["fingerprint_list"],
            f"generated prior-state family {family['family_id']}",
        )
    for source in inputs["provenance_sources"]:
        retain(source, f"provenance source {source['provenance_id']}")

    registry_path = _safe_repository_file(
        root, plan["bindings"]["registry"]["path"], "registry path"
    )
    _, registry_inspection = inspect_registry_file(registry_path, project_root=root)
    package_digest = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-package-digest/v1",
            "qualification_id": plan["qualification_id"],
            "plan_canonical_sha256": sha256_data(plan_normalized),
            "input_canonical_sha256": sha256_data(input_normalized),
            "registration_file_sha256": sha256_file(registration_source),
            "registration_size_bytes": str(registration_source.stat().st_size),
            "registration_canonical_sha256": sha256_data(registration_normalized),
            "locked_files": [locked[path] for path in sorted(locked)],
        }
    )
    inspection = Solar1PNQualificationInspection(
        schema=INSPECTION_SCHEMA,
        qualification_id=plan["qualification_id"],
        plan_state=plan["state"],
        input_state=inputs["state"],
        registration_state=registration["state"],
        plan_file_sha256=sha256_file(plan_path),
        plan_canonical_sha256=sha256_data(plan_normalized),
        input_file_sha256=sha256_file(input_path),
        input_canonical_sha256=sha256_data(input_normalized),
        registration_file_sha256=sha256_file(registration_source),
        registration_size_bytes=str(registration_source.stat().st_size),
        registration_canonical_sha256=sha256_data(registration_normalized),
        package_sha256=package_digest,
        registry_canonical_sha256=registry_inspection.registry_sha256,
        registry_authorized=False,
        outcomes_generated=False,
        ready_for_holdout_execution=False,
        blocked_reasons=BLOCKED_REASONS,
        arm_ids=tuple(sorted(item["arm_id"] for item in plan["test_arms"])),
        gate_ids=tuple(sorted(item["gate_id"] for item in plan["gates"])),
    )
    return plan_normalized, input_normalized, registration_normalized, inspection


__all__ = [
    "ALLOWED_VERDICTS",
    "ARM_GATE_MATRIX",
    "ARM_IDS",
    "ARM_SPECS",
    "BLOCKED_REASONS",
    "BLOCKER_SPECS",
    "CASE_ROLES",
    "DERIVATION_IDS",
    "EXECUTION_IMPLEMENTATION_STATE",
    "FOUNDATION_COMMIT",
    "INPUT_SCHEMA",
    "INPUT_STATE",
    "INSPECTION_SCHEMA",
    "GATE_IDS",
    "KNOWN_FILE_ROLE_PATHS",
    "KNOWN_STATE_RECORD_SPECS",
    "LOCKED_FILE_ROLE_PATHS",
    "NONCLAIM",
    "OBSERVABLE_IDS",
    "GATE_SPECS",
    "PLAN_SCHEMA",
    "PLAN_STATE",
    "PASS_EFFECT",
    "PERMITTED_CLAIM_CODE",
    "PERMITTED_CLAIM_TEXT",
    "PROHIBITED_CLAIM_CODES",
    "PROVENANCE_SOURCE_PATHS",
    "PRIOR_DEVELOPMENT_COUNTS",
    "PRIOR_DEVELOPMENT_DISCLOSURE_SHA256",
    "PRIOR_DEVELOPMENT_MANIFEST_CANONICAL_SHA256",
    "PRIOR_DEVELOPMENT_MANIFEST_FILE_SHA256",
    "PRIOR_DEVELOPMENT_MANIFEST_ID",
    "PRIOR_DEVELOPMENT_MANIFEST_PATH",
    "PRIOR_DEVELOPMENT_MANIFEST_SIZE_BYTES",
    "PRIOR_DEVELOPMENT_MANIFEST_STATE",
    "PRIOR_DEVELOPMENT_NONCLAIM",
    "PRIOR_DEVELOPMENT_SCHEMA",
    "PRIOR_FINGERPRINT_LIST_CANONICAL_SHA256",
    "PRIOR_FINGERPRINT_LIST_FILE_SHA256",
    "PRIOR_FINGERPRINT_LIST_PATH",
    "PRIOR_FINGERPRINT_LIST_SIZE_BYTES",
    "PRIOR_FIRST_MEMBER_FINGERPRINT_SHA256",
    "PRIOR_FOUNDATION_SOURCE_PATHS",
    "PRIOR_LAST_MEMBER_FINGERPRINT_SHA256",
    "PRIOR_ORDERED_MEMBER_FINGERPRINTS_SHA256",
    "QUALIFICATION_ID_PREFIX",
    "REGISTRATION_SCHEMA",
    "REGISTRATION_STATE",
    "SOURCE_LITERAL_SPECS",
    "Solar1PNQualificationError",
    "Solar1PNQualificationInspection",
    "inspect_solar_1pn_qualification_package",
    "qualification_fixture_sha256",
    "qualification_plan_identity_sha256",
    "qualification_scientific_fingerprint_sha256",
    "validate_prior_development_manifest_semantics",
    "validate_qualification_inputs_semantics",
    "validate_qualification_plan_semantics",
    "validate_qualification_registration_semantics",
]
