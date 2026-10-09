"""Fail-closed contracts for the WH--RKF78 whole-macrostep hybrid.

This module contains immutable request and policy types only.  It does not
publish or implement a hybrid integrator.  The future runtime is deliberately
restricted to a whole-system, whole-outer-step choice between the frozen
ordered-Jacobi Wisdom--Holman map and the frozen full-Cartesian Newtonian
encounter segment.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields
from typing import TypeAlias

from .contracts import ContractError
from .encounter_contracts import (
    ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
    ENCOUNTER_HARD_MAXIMUM_BODY_COUNT,
    ENCOUNTER_HARD_MAXIMUM_FORCE_EVALUATIONS,
    ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
    ENCOUNTER_HARD_MAXIMUM_SUBSTEP_PROPOSALS,
    ENCOUNTER_STAGE_COUNT,
    AdaptiveEncounterSegmentSpec,
    EncounterExactRationalResourceSpec,
)
from .wisdom_holman_contracts import (
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS,
    WH_KEPLER_MAXIMUM_ITERATIONS,
    WH_KEPLER_SERIES_TERM_COUNT,
    FixedStepWisdomHolmanSpec,
)


HybridFarProbeLegalMappingRow: TypeAlias = tuple[
    str, str, tuple[str, ...], str
]
HybridFarProbeWorkFormulaRow: TypeAlias = tuple[str, str]
HybridFarProbePhaseFloorRow: TypeAlias = tuple[
    str, tuple[HybridFarProbeWorkFormulaRow, ...]
]
HybridKeplerProbeFailureStatusRow: TypeAlias = tuple[str, str]
HybridPrivateEncounterDigestDomainRow: TypeAlias = tuple[str, str, str]
HybridPrivateEncounterPreimageSchemaRow: TypeAlias = tuple[
    str, str, tuple[str, ...], str
]
HybridPrivateEncounterSummarySourceRow: TypeAlias = tuple[str, str]


HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID = (
    "integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1"
)
HYBRID_METHOD_CLASS = (
    "WHOLE_MACROSTEP_ORDERED_JACOBI_WISDOM_HOLMAN_OR_FULL_CARTESIAN_RKF78"
)
HYBRID_COMPOSITION = (
    "PROBE_ONE_COMPLETE_SIGNED_WISDOM_HOLMAN_MACROSTEP;COMMIT_IFF_ALL_"
    "DYNAMIC_FAR_GUARDS_PASS;OTHERWISE_DISCARD_THE_PROBE_AND_REDO_THE_"
    "ENTIRE_ORIGINAL_INTERVAL_WITH_THE_GUARDED_CARTESIAN_ENCOUNTER_SOLVER"
)
HYBRID_BASE_TYPE_COMPOSITION_POLICY = (
    "the request retains one exact FixedStepWisdomHolmanSpec and one exact "
    "HybridEncounterControlProfile; it never copies or weakens frozen WH or "
    "encounter policy fields; each near interval is bound to a fresh exact "
    "AdaptiveEncounterSegmentSpec and one exact bounded private execution record "
    "is retained without constructing a standalone public child result"
)

HYBRID_ENCOUNTER_INTERVAL_BINDING_FIELDS = (
    "body_order",
    "initial_epoch",
    "endpoint_epoch",
    "duration",
)
HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER = (
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
)
HYBRID_ENCOUNTER_CONTROL_SCHEMA_POLICY = (
    "HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER must equal, in frozen "
    "dataclass order, every caller-required AdaptiveEncounterSegmentSpec field "
    "except body_order,initial_epoch,endpoint_epoch,and duration; both the "
    "profile constructor and top-level hybrid constructor check this identity "
    "from dataclasses.fields and fail closed if the frozen child schema drifts"
)
HYBRID_ENCOUNTER_BINDING_POLICY = (
    "derive body_order only from wisdom_holman_spec.jacobi_body_order; derive "
    "duration only from the bit-identical signed wisdom_holman_spec.fixed_step; "
    "for outer macrostep k>=0 derive explicit child labels t_k=float(t0+k*h) "
    "and t_k1=float(t0+(k+1)*h) from the original runtime snapshot epoch t0 "
    "using the global integer lattice,then construct an exact frozen encounter "
    "segment with initial_epoch=t_k,endpoint_epoch=t_k1,duration=h and the 21 "
    "profile fields unchanged; never derive h from t_k1-t_k; retain and hash "
    "every derived child spec and bounded private execution record; initialize fresh exact positive-"
    "zero Kahan carries for every full-macrostep encounter replacement"
)
HYBRID_CANONICAL_PROFILE_VALIDATION_POLICY = (
    "the standalone profile performs exhaustive exact built-in scalar,tuple-"
    "entry,equal pair-cardinality,finite-positive,controller,cap-relation,and "
    "resource-type validation for every interval-invariant field because it has "
    "no authoritative body order or h; "
    "the one and only complete validation path is the top-level constructor,"
    "which binds the profile to an exact encounter segment "
    "at canonical public labels 0.0 and h with duration exactly h and WH body "
    "order; construction of that exact frozen child delegates pair-table,"
    "controller,resource,reservation,and hard-cap validation without treating "
    "the canonical labels as runtime provenance"
)

HYBRID_OUTER_LATTICE_POLICY = (
    "the authoritative outer advance is the fixed signed binary64 h from the "
    "frozen WH spec on integer indices k=0..completed_steps; every public node "
    "label is independently evaluated as float(t0+k*h) from the original exact "
    "runtime snapshot epoch and never by recursive addition; preflight requires "
    "every label finite and strictly monotone in sign(h),retains the complete "
    "explicit label tuple,and requires force-parameter metadata validity across "
    "the closed label interval; each macrostep advances mathematical duration h "
    "regardless of rounded public-label differences"
)
HYBRID_CHECKPOINT_POLICY = (
    "emit only the frozen WH checkpoint_step_indices on the same global signed "
    "integer outer lattice; there is no clipping,interpolation,dense output,"
    "near-substep checkpoint,or event checkpoint; accepted_steps at checkpoint "
    "k remains the outer integer step index and rejected_steps remains zero "
    "because a discarded probe is accounted as provisional work,not as an "
    "accepted-lattice rejection"
)

HYBRID_STEP_MODES = (
    "WISDOM_HOLMAN_FAR",
    "CARTESIAN_RKF78_NEAR_FULL_INTERVAL",
)
HYBRID_FAR_PROBE_OUTCOMES = (
    "FAR_PASS",
    "NEAR_SWITCH",
    "FATAL_FAILURE",
)
HYBRID_ALL_FAR_REASON = "ALL_DYNAMIC_FAR_GUARDS_PASSED"
HYBRID_FAR_PROBE_PHASES = (
    "STATIC_PREFLIGHT",
    "ACCEPTED_START_REBIND",
    "ACCEPTED_START_NODE",
    "FIRST_HALF_KICK",
    "POST_FIRST_KICK_PREDRIFT_NODE",
    "KEPLER_DRIFT_FLOW",
    "POST_DRIFT_CARTESIAN_RECONSTRUCTION",
    "POST_DRIFT_PRE_FORCE_PATH_SCREEN",
    "POST_DRIFT_CANDIDATE_FORCE",
    "SECOND_HALF_KICK",
    "POST_SECOND_KICK_CARTESIAN_RECONSTRUCTION",
    "COMPLETED_CANDIDATE_NODE",
    "ALL_GUARDS_COMPLETED",
)
HYBRID_DYNAMIC_GUARD_PHASES = (
    "ACCEPTED_START_NODE",
    "POST_FIRST_KICK_PREDRIFT_NODE",
    "POST_DRIFT_PRE_FORCE_PATH_SCREEN",
    "COMPLETED_CANDIDATE_NODE",
)
HYBRID_DYNAMIC_SWITCH_REASONS = (
    "FINITE_JACOBI_STATE_OUTSIDE_BOUND_ELLIPTIC_FAR_DOMAIN",
    "FINITE_JACOBI_ECCENTRICITY_ABOVE_FAR_LIMIT",
    "FINITE_JACOBI_PERIAPSE_BELOW_FAR_FLOOR",
    "FINITE_JACOBI_SEMIMAJOR_AXES_NOT_STRICTLY_INCREASING",
    "FINITE_INTERACTION_TO_KEPLER_RATIO_ABOVE_FAR_LIMIT",
    "FINITE_ORBIT_STEP_FRACTION_ABOVE_FAR_LIMIT",
    "FINITE_PERIAPSE_STEP_FRACTION_ABOVE_FAR_LIMIT",
    "FINITE_NODE_PAIR_CLEARANCE_AT_OR_BELOW_FAR_FLOOR",
    "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN",
)
HYBRID_PAIR_SWITCH_REASONS = (
    "FINITE_NODE_PAIR_CLEARANCE_AT_OR_BELOW_FAR_FLOOR",
    "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN",
)
HYBRID_BODY_SWITCH_REASONS = HYBRID_DYNAMIC_SWITCH_REASONS[:-2]
HYBRID_FAR_PROBE_FATAL_REASONS = (
    "STATIC_CONTRACT_INPUT_BACKEND_FORCE_OR_METADATA_FAILURE",
    "OUTER_INTEGER_LATTICE_OR_PUBLIC_LABEL_FAILURE",
    "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
    "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
    "FORCE_EVALUATION_OR_IDENTITY_FAILURE",
    "TRANSLATION_FORCE_RESIDUAL_FAILURE",
    "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
)
HYBRID_TRIGGER_RULES = (
    "EMPTY",
    "BODY_INDICES_NONEMPTY_PAIR_INDICES_EMPTY",
    "BODY_INDICES_EMPTY_PAIR_INDICES_NONEMPTY",
)
_HYBRID_NODE_GUARD_PHASES = (
    "ACCEPTED_START_NODE",
    "POST_FIRST_KICK_PREDRIFT_NODE",
    "COMPLETED_CANDIDATE_NODE",
)
_HYBRID_BODY_NUMERICAL_PHASES = (
    "ACCEPTED_START_REBIND",
    "ACCEPTED_START_NODE",
    "FIRST_HALF_KICK",
    "POST_FIRST_KICK_PREDRIFT_NODE",
    "KEPLER_DRIFT_FLOW",
    "POST_DRIFT_CARTESIAN_RECONSTRUCTION",
    "POST_DRIFT_CANDIDATE_FORCE",
    "SECOND_HALF_KICK",
    "POST_SECOND_KICK_CARTESIAN_RECONSTRUCTION",
    "COMPLETED_CANDIDATE_NODE",
)
HYBRID_FAR_PROBE_LEGAL_MAPPING: tuple[HybridFarProbeLegalMappingRow, ...] = (
    (
        "FAR_PASS",
        HYBRID_ALL_FAR_REASON,
        ("ALL_GUARDS_COMPLETED",),
        "EMPTY",
    ),
    *tuple(
        (
            "NEAR_SWITCH",
            reason,
            _HYBRID_NODE_GUARD_PHASES,
            "BODY_INDICES_NONEMPTY_PAIR_INDICES_EMPTY",
        )
        for reason in HYBRID_BODY_SWITCH_REASONS
    ),
    (
        "NEAR_SWITCH",
        "FINITE_NODE_PAIR_CLEARANCE_AT_OR_BELOW_FAR_FLOOR",
        _HYBRID_NODE_GUARD_PHASES,
        "BODY_INDICES_EMPTY_PAIR_INDICES_NONEMPTY",
    ),
    (
        "NEAR_SWITCH",
        "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN",
        ("POST_DRIFT_PRE_FORCE_PATH_SCREEN",),
        "BODY_INDICES_EMPTY_PAIR_INDICES_NONEMPTY",
    ),
    (
        "FATAL_FAILURE",
        "STATIC_CONTRACT_INPUT_BACKEND_FORCE_OR_METADATA_FAILURE",
        ("STATIC_PREFLIGHT",),
        "EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "OUTER_INTEGER_LATTICE_OR_PUBLIC_LABEL_FAILURE",
        ("STATIC_PREFLIGHT",),
        "EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
        _HYBRID_BODY_NUMERICAL_PHASES,
        "BODY_INDICES_NONEMPTY_PAIR_INDICES_EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
        _HYBRID_BODY_NUMERICAL_PHASES + ("POST_DRIFT_PRE_FORCE_PATH_SCREEN",),
        "EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "FORCE_EVALUATION_OR_IDENTITY_FAILURE",
        ("ACCEPTED_START_NODE", "POST_DRIFT_CANDIDATE_FORCE"),
        "EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "TRANSLATION_FORCE_RESIDUAL_FAILURE",
        ("ACCEPTED_START_NODE", "POST_DRIFT_CANDIDATE_FORCE"),
        "EMPTY",
    ),
    (
        "FATAL_FAILURE",
        "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
        ("KEPLER_DRIFT_FLOW",),
        "BODY_INDICES_NONEMPTY_PAIR_INDICES_EMPTY",
    ),
)
HYBRID_KEPLER_PROBE_SOURCE = (
    "FROZEN_PRIVATE_WISDOM_HOLMAN_ELLIPTIC_UNIVERSAL_KEPLER_SUBFLOW"
)
HYBRID_KEPLER_PROBE_TERMINAL_STATUSES = (
    "COMPLETED",
    "INCOMPLETE_NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
    "INCOMPLETE_NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
    "INCOMPLETE_KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
)
HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON: tuple[
    HybridKeplerProbeFailureStatusRow, ...
] = (
    (
        "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
        "INCOMPLETE_NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
    ),
    (
        "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
        "INCOMPLETE_NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
    ),
    (
        "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
        "INCOMPLETE_KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
    ),
)
HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER = (
    "secondary_index",
    "call_completed",
    "iterations_entered",
    "bracket_expansions_entered",
    "g_bundle_calls_entered",
    "g_bundle_calls_completed",
    "completed_series_bundle_count",
    "interrupted_series_terms",
    "terminal_status",
    "source",
)
HYBRID_KEPLER_PROBE_RECORD_POLICY = (
    "retain one exact frozen HybridKeplerProbeRecord for every entered secondary "
    "subflow,with tuple length checked at most body_count-1 and at most 15 before "
    "traversal,indices exactly "
    "1..length,and only the final record permitted incomplete; let I,X,GE,GC be "
    "that record's entered iterations,entered bracket expansions,and entered/"
    "completed G bundles,and seed=GE-I-X; require seed in {0,1},seed=1 whenever "
    "I+X>0 or the call completes,I<=96,X<=32,and GE-GC in {0,1}; a completed "
    "call requires I>=1,GE=GC=1+I+X,zero interrupted terms,and COMPLETED status; "
    "an incomplete call with seed=0 is wholly zero,otherwise GE=1+I+X and "
    "GC in {GE-1,GE}; completed_series_bundle_count<=GC; every completed series "
    "bundle contributes exactly 256 terms,every completed trigonometric bundle "
    "contributes zero; increment iterations at entry to the frozen for-iteration "
    "body,bracket expansions at the frozen pre-evaluation increment,and partial "
    "terms for every attempted term including the term whose finite check fails; "
    "interrupted terms are zero unless GE=GC+1,when their "
    "exact allowed set is {0} union [2,64] union [66,128] union [130,192] union "
    "[194,256]; retain the exact frozen private source and closed terminal "
    "failure status; derive every aggregate Kepler,I,X,G,and series counter only "
    "as the exact sum of this tuple and reject an independently rewritten total"
)
HYBRID_FAR_PROBE_WORK_FIELD_ROSTER = (
    "force_calls_entered",
    "force_evaluations_completed",
    "interaction_force_assemblies",
    "kepler_subflow_calls_entered",
    "kepler_subflow_solves_completed",
    "universal_solver_iterations",
    "universal_solver_bracket_expansions",
    "universal_g_bundle_calls_entered",
    "universal_g_bundle_calls_completed",
    "universal_series_terms_evaluated",
    "kepler_probe_records",
    "cartesian_to_jacobi_calls_entered",
    "cartesian_to_jacobi_transforms_completed",
    "jacobi_to_cartesian_calls_entered",
    "jacobi_to_cartesian_transforms_completed",
    "node_guard_evaluations",
    "path_guard_evaluations",
    "first_half_kicks_completed",
    "center_of_mass_drifts_completed",
    "second_half_kicks_completed",
)
HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER = tuple(
    name for name in HYBRID_FAR_PROBE_WORK_FIELD_ROSTER if name != "kepler_probe_records"
)
HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS: tuple[
    HybridFarProbeWorkFormulaRow, ...
] = (
    ("force_calls_entered", "2"),
    ("force_evaluations_completed", "2"),
    ("interaction_force_assemblies", "2"),
    ("kepler_subflow_calls_entered", "body_count-1"),
    ("kepler_subflow_solves_completed", "body_count-1"),
    (
        "universal_solver_iterations",
        f"{WH_KEPLER_MAXIMUM_ITERATIONS}*(body_count-1)",
    ),
    (
        "universal_solver_bracket_expansions",
        f"{WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS}*(body_count-1)",
    ),
    (
        "universal_g_bundle_calls_entered",
        f"{1 + WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS + WH_KEPLER_MAXIMUM_ITERATIONS}*(body_count-1)",
    ),
    (
        "universal_g_bundle_calls_completed",
        f"{1 + WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS + WH_KEPLER_MAXIMUM_ITERATIONS}*(body_count-1)",
    ),
    (
        "universal_series_terms_evaluated",
        f"4*{WH_KEPLER_SERIES_TERM_COUNT}*universal_g_bundle_calls_entered",
    ),
    ("cartesian_to_jacobi_calls_entered", "1"),
    ("cartesian_to_jacobi_transforms_completed", "1"),
    ("jacobi_to_cartesian_calls_entered", "2"),
    ("jacobi_to_cartesian_transforms_completed", "2"),
    ("node_guard_evaluations", "3"),
    ("path_guard_evaluations", "1"),
    ("first_half_kicks_completed", "1"),
    ("center_of_mass_drifts_completed", "1"),
    ("second_half_kicks_completed", "1"),
)
HYBRID_FAR_PROBE_PHASE_WORK_FLOORS: tuple[
    HybridFarProbePhaseFloorRow, ...
] = (
    ("STATIC_PREFLIGHT", ()),
    (
        "ACCEPTED_START_REBIND",
        (("cartesian_to_jacobi_calls_entered", "1"),),
    ),
    (
        "ACCEPTED_START_NODE",
        (
            ("cartesian_to_jacobi_transforms_completed", "1"),
        ),
    ),
    ("FIRST_HALF_KICK", (("force_evaluations_completed", "R"),)),
    (
        "POST_FIRST_KICK_PREDRIFT_NODE",
        (
            ("first_half_kicks_completed", "1"),
            ("node_guard_evaluations", "1+R"),
        ),
    ),
    (
        "KEPLER_DRIFT_FLOW",
        (
            ("first_half_kicks_completed", "1"),
        ),
    ),
    (
        "POST_DRIFT_CARTESIAN_RECONSTRUCTION",
        (
            ("kepler_subflow_solves_completed", "body_count-1"),
            ("jacobi_to_cartesian_calls_entered", "1"),
            ("node_guard_evaluations", "1+R"),
        ),
    ),
    (
        "POST_DRIFT_PRE_FORCE_PATH_SCREEN",
        (
            ("first_half_kicks_completed", "1"),
            ("center_of_mass_drifts_completed", "1"),
            ("kepler_subflow_solves_completed", "body_count-1"),
            ("jacobi_to_cartesian_calls_entered", "1"),
            ("jacobi_to_cartesian_transforms_completed", "1"),
            ("force_calls_entered", "R"),
            ("force_evaluations_completed", "R"),
            ("interaction_force_assemblies", "R"),
            ("node_guard_evaluations", "1+R"),
            ("path_guard_evaluations", "1"),
        ),
    ),
    (
        "POST_DRIFT_CANDIDATE_FORCE",
        (
            ("first_half_kicks_completed", "1"),
            ("center_of_mass_drifts_completed", "1"),
            ("kepler_subflow_solves_completed", "body_count-1"),
            ("jacobi_to_cartesian_transforms_completed", "1"),
            ("path_guard_evaluations", "1"),
            ("force_calls_entered", "R+1"),
            ("force_evaluations_completed", "R"),
            ("interaction_force_assemblies", "R"),
        ),
    ),
    (
        "SECOND_HALF_KICK",
        (
            ("path_guard_evaluations", "1"),
            ("second_half_kicks_completed", "0"),
        ),
    ),
    (
        "POST_SECOND_KICK_CARTESIAN_RECONSTRUCTION",
        (
            ("second_half_kicks_completed", "1"),
            ("jacobi_to_cartesian_calls_entered", "2"),
        ),
    ),
    (
        "COMPLETED_CANDIDATE_NODE",
        (
            ("second_half_kicks_completed", "1"),
            ("jacobi_to_cartesian_transforms_completed", "2"),
            ("node_guard_evaluations", "2+R"),
        ),
    ),
    (
        "ALL_GUARDS_COMPLETED",
        (
            ("force_evaluations_completed", "R+1"),
            ("kepler_subflow_solves_completed", "body_count-1"),
            ("path_guard_evaluations", "1"),
            ("second_half_kicks_completed", "1"),
            ("node_guard_evaluations", "2+R"),
        ),
    ),
)
HYBRID_ACCEPTED_START_REBIND_POLICY = (
    "let R=cartesian_to_jacobi_transforms_completed for one probe; R is exactly one only "
    "for the first outer probe and the first probe after every committed near "
    "step,and is zero on every other probe in a continuous far streak; R=1 "
    "carries the one completed initial/rebind force evaluation,interaction "
    "assembly,and accepted-start node guard; the outer ledger validates this "
    "cross-record sequence so accepted-start work is never charged every step"
)
HYBRID_FAR_PROBE_WORK_VALIDATION_POLICY = (
    "all 19 counters are exact nonnegative built-in integers prospectively "
    "bounded by HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS; completed force calls "
    "and interaction assemblies cannot exceed entered/completed force calls; "
    "the exact bounded per-secondary HybridKeplerProbeRecord tuple is the sole "
    "source for entered/completed Kepler solves,iterations,bracket expansions,"
    "G-bundle entered/completed calls,and actual series terms; every aggregate "
    "counter is the exact tuple reduction and at most its final record or G bundle "
    "is incomplete; both "
    "coordinate transforms retain entered and completed calls; candidate force "
    "and assembly require the completed path screen,and second kick<=path guard"
    "<=first inverse transform and second kick<=center "
    "drift<=first kick; every terminal phase obeys the retained causal phase "
    "floor table,with fatal work retained only through its exact failure point"
)
HYBRID_FAR_PROBE_REASON_WORK_POLICY = (
    "STATIC_PREFLIGHT has all 19 counters zero; ACCEPTED_START_REBIND has only "
    "C2J-entered=1,C2J-completed=0; at ACCEPTED_START_NODE force/identity failure "
    "has C2J E=C=1,force-entered=1,force-completed in {0,1},assembly=node=0,"
    "translation failure additionally has force-completed=assembly=1,node=0,"
    "interaction-assembly numerical failure has force E=C=1,assembly=node=0,and "
    "a guard outcome has force E=C=assembly=node=1; after a completed drift the "
    "first inverse-transform fatal has J2C E=1,C=0 and only baseline R force; "
    "the sole pre-force path screen has J2C E=C=1,node=1+R,path=1,force E=C="
    "assembly=R,and second-kick=0; candidate force/identity occurs only after that "
    "screen and has J2C E=C=1,path=1,force-entered=R+1,force-completed in "
    "{R,R+1},assembly=R; candidate assembly numerical failure has force E=C=R+1,"
    "assembly=R; translation failure has force E=C=assembly=R+1; final inverse-transform fatal "
    "has second-kick=1,J2C-entered=2,J2C-completed=1,node=1+R,whereas completed-"
    "node guard entry has J2C E=C=2,node=2+R; first inverse reconstruction has "
    "node exactly 1+R; a dedicated Kepler failure has one final incomplete "
    "per-secondary record and its sole body trigger is that record's exact "
    "secondary index; its terminal status is bound to the outer failure reason; "
    "a generic numerical Kepler-flow fatal may precede the COM-drift completion,"
    "occur between returned subflows,follow a completed prefix or complete A "
    "flow,or end inside one final incomplete subflow; only an actually incomplete "
    "record carries the corresponding numerical failure status; every body-local "
    "in-solver or post-return failure requires a record and names its exact "
    "secondary; a no-record q0/COM failure is global,and a global failure with "
    "only completed records is post-A and therefore requires all B-1 records; "
    "global failures have no body trigger"
)
HYBRID_FAR_PROBE_DECISION_INTERFACE_POLICY = (
    "the future private WH macrostep kernel returns one closed typed outcome "
    "from HYBRID_FAR_PROBE_OUTCOMES plus an exact reason enum,guard phase,"
    "canonical built-in body-index tuple,canonical pair-index tuple,and actual "
    "provisional-work record; hybrid control flow must never infer an outcome "
    "from a generic exception class,exception message,substring,or mutable "
    "dictionary; only the explicit NEAR_SWITCH variant carrying a member of "
    "HYBRID_DYNAMIC_SWITCH_REASONS may invoke the encounter child; any "
    "nonfinite operand,evaluation failure,or fatal mapping reached through the "
    "terminal phase takes precedence over a finite switch candidate at that phase "
    "and can never be relabeled as a finite far-envelope exit; a completed finite "
    "pre-force path-screen switch terminates before candidate force is entered,so "
    "no hypothetical unentered later failure is inferred"
)
HYBRID_DYNAMIC_SWITCH_REASON_POLICY = (
    "only a finite evaluated predicate named in HYBRID_DYNAMIC_SWITCH_REASONS "
    "may select near mode; use the retained probe cadence with the sole pure post-"
    "drift path-screen hoist,then the exact "
    "reason-tuple order,and immutable body/canonical-pair order as deterministic "
    "precedence; retain one phase and one first reason; body reasons retain every "
    "offending immutable WH/Jacobi body index in increasing order; pair reasons retain every "
    "pair failing that selected predicate at that phase in canonical i<j order,"
    "while the all-far reason retains empty body and pair tuples; a "
    "failed far screen means only that WH is not committed and is not evidence "
    "of contact,collision,an event time,or physical close approach"
)
HYBRID_SWITCH_POLICY = (
    "make a fresh independent decision at every outer step from its accepted "
    "start state; there is no hysteresis,latch,cooldown,memory of a previous "
    "mode,predictive partial prefix,prefix/suffix split,recursive refinement,"
    "body subset,encounter group,or event location; the only choices are the "
    "whole-system WH macrostep or a whole-system Cartesian redo over the same "
    "complete original signed interval"
)
HYBRID_PROBE_TRANSACTION_POLICY = (
    "assemble WH work only in provisional private buffers; when its domain "
    "permits,probe the first kick,predrift node guard,COM plus Kepler drift,"
    "candidate Cartesian reconstruction,the sole pure pre-force path screen,"
    "candidate force evaluation,interaction assembly and translation residual,"
    "second kick,and "
    "completed-node guard; initial/rebind force and translation work belongs to "
    "the one accepted-start initialization lane selected by R and is not recurring "
    "accepted-start work on a continuous far streak; a dynamic failure at its "
    "retained cadence aborts the candidate and "
    "retains actual work through that point; commit no state,private WH cache,"
    "checkpoint,or accepted-WH accounting until every dynamic far guard passes; "
    "on near selection discard all provisional state and start the encounter "
    "child from the untouched accepted outer-node Cartesian state for duration h"
)
HYBRID_NEAR_REPLACEMENT_POLICY = (
    "a near redo covers exactly the original full outer interval and all bodies "
    "through the frozen autonomous all-active positive-GM unsoftened Newtonian "
    "encounter solver; its local first-crossing certificate retains only the "
    "frozen per-accepted-node local-IVP noncollision scope; any encounter "
    "certificate rejection is an adaptive child proposal decision and never a "
    "hybrid collision or event assertion"
)
HYBRID_ORIGINAL_NODE_ADMISSIBILITY_POLICY = (
    "before a typed near outcome may start RKF78,run the frozen encounter child "
    "preflight at the untouched committed original node and require its exact-"
    "dyadic initial squared separations strictly above every retained child "
    "certification floor and radii sums; a violation already present at that "
    "committed node is fatal because redoing the same interval cannot repair its "
    "initial condition; a finite stricter WH scalar or mutual-Hill far-screen "
    "exit from an encounter-admissible original node remains switchable,as does "
    "a violation found only in discarded provisional path/end state; retain the "
    "child initialization work on success or fatal failure"
)

HYBRID_FATAL_FAILURE_CLASSES = (
    "STATIC_CONTRACT_INPUT_BACKEND_FORCE_OR_METADATA_FAILURE",
    "STATIC_WH_MASS_RATIO_OR_INITIAL_BARYCENTER_FAILURE",
    "OUTER_INTEGER_LATTICE_OR_PUBLIC_LABEL_FAILURE",
    "NONFINITE_OR_SINGULAR_NUMERICAL_FAILURE",
    "FORCE_EVALUATION_IDENTITY_OR_TRANSLATION_RESIDUAL_FAILURE",
    "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
    "ENCOUNTER_ACCEPTED_NODE_DOMAIN_OR_INITIAL_FLOOR_FAILURE",
    "ENCOUNTER_STEP_REJECTION_FORCE_OR_EXACT_RESOURCE_CAP_FAILURE",
    "CUSTODY_CHECKSUM_ACCOUNTING_OR_REPLAY_FAILURE",
)
HYBRID_FATAL_FAILURE_POLICY = (
    "every failure not produced by a finite named dynamic far-guard predicate is "
    "fatal and returns no result; in particular invalid types or schemas,body/"
    "force/backend/metadata mismatch,static WH preflight limits,nonfinite or "
    "singular arithmetic,force evaluation failure,Kepler solver failure,child "
    "initial-floor or accepted-node failure,child step/force/exact-resource "
    "exhaustion,and any custody/checksum/accounting/replay mismatch never trigger "
    "a mode switch and never expose partial state; a nonfinite or evaluation "
    "failure encountered anywhere in the entered pre-force path screen is fatal "
    "and supersedes finite switch predicates from that screen,while a completed "
    "finite near decision returns before any candidate-force work is entered"
)

HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE = 65_536
HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE = 4_096
HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE = 524_288
HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE = 917_504
HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL = 131_072
HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL = 8_192
HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL = 1_048_576
HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL = 1_835_008
HYBRID_AGGREGATE_RESOURCE_POLICY = (
    "primary and mandatory hybrid-replay lanes each independently obey hard "
    "library ceilings of 65536 outer records,4096 near macrosteps,524288 "
    "retained accepted near substeps,and 917504 raw retained SHA-256 digest "
    "bytes; public-total ceilings are exactly twice those values and neither lane "
    "borrows capacity; construction requires both WH maximum_steps and executed "
    "completed_steps no greater than the outer cap and prospectively proves with "
    "division before multiplication that min(completed_steps,4096) times the "
    "child maximum_accepted_substeps fits the retained-substep cap; runtime "
    "checks every append before allocation and fails without partial output when "
    "a near,substep,record,or digest ceiling would be exceeded; these hard "
    "qualification ceilings are not recommended workload targets"
)
HYBRID_AGGREGATE_RESOURCE_NONCLAIM_POLICY = (
    "the fixed library ceilings and digest-only custody bound retained logical "
    "records but do not guarantee wall time,resident memory,allocator behavior,"
    "concurrency safety,or denial-of-service resistance; exact child arithmetic "
    "and streamed transcripts can still be expensive,so an external service must "
    "enforce process isolation,timeout,memory,and concurrency limits"
)

HYBRID_ALL_FAR_COMPATIBILITY_POLICY = (
    "conditional on the future runtime using a continuous private WH state and "
    "preserving every frozen WH arithmetic expression,loop,and commit/diagnostic "
    "merge order except for the sole pure post-drift path-screen hoist before "
    "candidate force,an execution whose every dynamic guard passes must project "
    "to exactly the frozen public WH result: bit-identical accepted state,"
    "Cartesian checkpoint arrays and scalar diagnostics,identical "
    "checkpoint labels,force ledger,schedule and result checksums,and identical "
    "primary/replay/public WH accounting; repeated public one-step WH calls or "
    "Cartesian-to-Jacobi rebinding at every far step are forbidden because they "
    "do not satisfy this compatibility requirement; hybrid-only mode ledgers and "
    "their checksums are additional content and do not alter the WH projection"
)
HYBRID_PROVISIONAL_WORK_ACCOUNTING_POLICY = (
    "for every outer step retain exactly the 19 nonnegative built-in counters in "
    "HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER plus the exact bounded "
    "per-secondary HybridKeplerProbeRecord tuple: entered and completed force calls,"
    "interaction-force assemblies,entered and completed Kepler solves,solver "
    "iterations,bracket expansions,entered/completed G bundles,actual series "
    "terms derived only from those records,entered/completed calls in both "
    "coordinate-transform directions,node/"
    "path guards,and completed first-kick,"
    "center-drift,and second-kick flows; increment entered-call and guard counters "
    "before invocation and every completed counter only after successful "
    "completion; the retained maxima and causal phase-floor tables reject "
    "impossible zero,out-of-order,or oversized terminal records; "
    "far-probe work is also accepted WH work,while near-probe work is separately "
    "marked discarded and must not enter accepted-WH counters; work attempted "
    "before a dynamic guard or fatal failure is never rounded up to a complete "
    "step and is never hidden"
)
HYBRID_NEAR_LEDGER_POLICY = (
    "each near step executes the frozen encounter algorithm once through its "
    "private primary executor and never constructs or fabricates a public "
    "EncounterSegmentResult,whose frozen constructor requires its own replay; "
    "retain one exact HybridPrivateEncounterRecord containing the exact derived "
    "AdaptiveEncounterSegmentSpec,a bounded accepted-signed-step ledger,bounded "
    "primary summary counters mapped exactly to frozen child sources,and seven "
    "distinct domain-separated content digests streamed over the final state,"
    "initialization/proposal/force ledgers,and complete private primary counts; "
    "do not retain child exact transcripts or full proposal/force ledgers; the "
    "aggregate digest excludes itself and binds the child index,derived spec,"
    "summary,and six component digests; the outer record binds the private child "
    "record to its index/labels"
)
HYBRID_PRIVATE_ENCOUNTER_ACCEPTED_STEP_POLICY = (
    "the private accepted-step tuple is nonempty,exact built-in finite binary64,"
    "signed in duration direction,and exact-rationally sums to duration; its first "
    "magnitude is no greater than initial_step_magnitude; every magnitude is no "
    "greater than maximum_step_magnitude,and for j>0 is no greater than the "
    "binary64 min(maximum_step_magnitude,maximum_scale_factor*abs(step[j-1])); "
    "every nonterminal magnitude is at least minimum_step_magnitude and only the "
    "last exact endpoint remainder may be smaller; accepted_count equals tuple "
    "length and 13*accepted_count<=force_evaluations<=13*proposal_count"
)
HYBRID_STEP_LEDGER_POLICY = (
    "retain exactly one immutable record for every completed outer index in "
    "increasing order; each record binds start/end label hex,signed h hex,mode,"
    "typed probe outcome,guard phase,reason,canonical trigger bodies and pairs,"
    "actual provisional WH work including every ordered per-secondary Kepler "
    "probe record and "
    "whether it was committed or discarded,and either no child or one exact "
    "private near-child record; tuple lengths and aggregate products are checked "
    "against the fixed per-lane hard caps before traversal or allocation"
)
HYBRID_NESTED_REPLAY_ACCOUNTING_POLICY = (
    "the hybrid never calls either public child wrapper and performs no nested "
    "child semantic replay; one primary hybrid lane performs one private far "
    "probe per outer step and exactly one private encounter execution per switched "
    "step; the one mandatory outer hybrid replay independently repeats those "
    "private executions and recomputes every private child record; all child work "
    "is counted once in its enclosing lane and public totals are the exact sum of "
    "the two enclosing lanes"
)
HYBRID_EXECUTION_ACCOUNTING_POLICY = (
    "in each lane let O be completed outer steps,F far commits,N near commits,"
    "A the componentwise sum of the 19-counter vectors for far-accepted probe "
    "work,D the componentwise sum of the 19-counter vectors for far-discarded "
    "probe work,and C the componentwise sum of private encounter "
    "primary counts; require O=F+N,far_probe_count=O,far_accepted_count=F,"
    "far_discarded_count=N,private_encounter_execution_count=N,and total probe "
    "work=A+D; ordered per-secondary Kepler records remain per-outer-step custody "
    "and are never arithmetically aggregated; retain accepted-WH,discarded-WH,"
    "private-encounter,coordinate-"
    "transform,near accepted-substep,and checkpoint counters separately; for a "
    "successful deterministic replay every replay scalar/record equals primary "
    "bit-for-bit and each public-total counter equals primary plus replay exactly"
)
HYBRID_PUBLIC_EXECUTION_ACCOUNTING_SCOPE = (
    "PRIMARY_HYBRID_EXECUTION_PLUS_ONE_MANDATORY_FULL_HYBRID_SEMANTIC_REPLAY;"
    "REPORT_PRIMARY_REPLAY_AND_PUBLIC_TOTALS_WITH_ACCEPTED_WH,DISCARDED_WH_"
    "PROBE,AND_PRIVATE_ENCOUNTER_LANES_SEPARATE"
)
HYBRID_VALIDATION_REPLAY_POLICY = (
    "one full deterministic replay from owned input recomputes the global label "
    "lattice,every independent mode decision,reason and pair tuple,all actual "
    "provisional-work counters,every derived near spec and private execution "
    "record,all accepted states,checkpoints,ledgers,and checksums; release only "
    "after exact built-in scalar and canonical-byte comparison plus bitwise "
    "array identity; replay accounting is retained separately and public totals "
    "are exact primary-plus-replay sums"
)
HYBRID_VALIDATION_REPLAY_COUNT = 1

HYBRID_CANONICAL_SERIALIZATION_POLICY = (
    "use the frozen encounter canonical encoder exactly: None and exact built-in "
    "bool,int,str remain JSON primitives; finite exact built-in float becomes "
    "{'float_hex':value.hex()}; exact tuple/list becomes a JSON array in order; "
    "exact str-key dict becomes a JSON object; an exact jxplanetx dataclass "
    "becomes {'dataclass':'<module>.<qualname>','fields':{each declared dataclass "
    "field name:recursive value}} with every declared field present; exact NumPy "
    "float64/bool ndarray becomes {'dtype':str(dtype),'shape':[built-in integer "
    "dimensions],'values':[C-order flat float.hex strings or built-in booleans]}; "
    "serialize the recursive object with json.dumps(sort_keys=True,ensure_ascii="
    "True,separators=(',',':'),allow_nan=False).encode('utf-8'); reject every "
    "other type and nonfinite float; nested HybridKeplerProbeRecord tuples are "
    "encoded recursively as declared dataclasses in exact secondary order and "
    "future step-ledger literal known-answer tests must bind and mutate every "
    "record field; domain preimages begin domain UTF-8,NUL; "
    "this is deterministic content integrity and replay provenance,not authentication"
)
HYBRID_STREAMING_SEQUENCE_FRAMING_POLICY = (
    "a streamed sequence preimage after domain_utf8+NUL is schema_id_utf8+NUL,"
    "then the exact item count as one unsigned 64-bit big-endian integer,then for "
    "each item in retained order its canonical-JSON byte length as unsigned "
    "64-bit big-endian followed by exactly those bytes; prospectively reject a "
    "count,length,or cumulative byte total above its retained cap before encoding "
    "or hashing the item; no delimiter,implicit concatenation,or alternate framing"
)
HYBRID_RESULT_CUSTODY_POLICY = (
    "copy validated inputs before work; caller arrays are never mutated; all "
    "retained states,checkpoints,step records,provisional-work records,near child "
    "bindings,bounded summaries,accepted-step ledgers,and digests are exact-type "
    "owned read-only nonaliasing copies; full private child transcripts and "
    "proposal/force ledgers are streamed into digests and not retained; no "
    "provisional WH array or child work buffer may alias an accepted or public "
    "array; validate tuple bounds and exact nested types before traversal"
)
HYBRID_CHECKSUM_BINDING_POLICY = (
    "the schedule checksum binds the complete exact HybridWisdomHolmanRKF78Spec "
    "including every retained mode,outcome,reason,phase and its sole pre-force "
    "path-screen ordering,trigger,fatal,work,"
    "per-secondary Kepler record/source/status/cadence,child-"
    "record,binder,cap,and mapping roster,the exact two nested request types,the "
    "initial snapshot epoch,global explicit label lattice,signed h "
    "and checkpoint indices; the step-ledger checksum binds every mode,typed "
    "probe outcome,phase,reason,body-index tuple,pair tuple,actual provisional-"
    "work value for every one of the 19 counters,the complete ordered per-"
    "secondary Kepler record tuple,commit/discard flag,and "
    "near private-child spec/record digest; the result checksum binds owned input "
    "and force provenance,all checkpoints and bounded child records,all component digests,"
    "and separate primary,replay,and public-total accounting"
)
HYBRID_SCHEDULE_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
HYBRID_SCHEDULE_CHECKSUM_DOMAIN = (
    "jxplanetx.hybrid-wh-rkf78-schedule.content-integrity.v1"
)
HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN = (
    "jxplanetx.hybrid-wh-rkf78-step-ledger.content-integrity.v1"
)
HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_DOMAIN = (
    "jxplanetx.hybrid-private-encounter-execution.content-integrity.v1"
)
HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER = (
    "final_state_content_sha256",
    "initialization_content_sha256",
    "proposal_ledger_content_sha256",
    "force_ledger_content_sha256",
    "primary_counts_content_sha256",
    "accepted_steps_content_sha256",
    "private_execution_content_sha256",
)
HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER: tuple[
    HybridPrivateEncounterDigestDomainRow, ...
] = (
    (
        "final_state_content_sha256",
        "jxplanetx.hybrid-private-encounter-final-state.content-integrity.v1",
        "owned final Cartesian positions and velocities in exact body order",
    ),
    (
        "initialization_content_sha256",
        "jxplanetx.hybrid-private-encounter-initialization.content-integrity.v1",
        "the exact frozen EncounterInitializationRecord",
    ),
    (
        "proposal_ledger_content_sha256",
        "jxplanetx.hybrid-private-encounter-proposal-ledger.content-integrity.v1",
        "the complete ordered streamed EncounterProposalRecord sequence",
    ),
    (
        "force_ledger_content_sha256",
        "jxplanetx.hybrid-private-encounter-force-ledger.content-integrity.v1",
        "the complete ordered frozen ForceLedgerEntry tuple in force-plan ledger order",
    ),
    (
        "primary_counts_content_sha256",
        "jxplanetx.hybrid-private-encounter-primary-counts.content-integrity.v1",
        "the exact frozen EncounterExecutionCounts from the private primary run",
    ),
    (
        "accepted_steps_content_sha256",
        "jxplanetx.hybrid-private-encounter-accepted-steps.content-integrity.v1",
        "the complete ordered accepted signed binary64 substep tuple",
    ),
    (
        "private_execution_content_sha256",
        HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_DOMAIN,
        "outer index,derived exact segment spec,all bounded summary fields,and the six preceding labeled component digests; excludes itself",
    ),
)
HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER: tuple[
    HybridPrivateEncounterPreimageSchemaRow, ...
] = (
    (
        "final_state_content_sha256",
        "jxplanetx.hybrid-private-encounter-final-state.payload.v1",
        ("schema", "positions", "velocities"),
        "one canonical JSON dict with exactly these keys; positions and velocities are owned read-only C-contiguous float64 arrays of shape (body_count,3)",
    ),
    (
        "initialization_content_sha256",
        "jxplanetx.hybrid-private-encounter-initialization.payload.v1",
        ("schema", "record"),
        "one canonical JSON dict with exactly these keys; record is the exact complete frozen EncounterInitializationRecord dataclass encoding",
    ),
    (
        "proposal_ledger_content_sha256",
        "jxplanetx.hybrid-private-encounter-proposal-ledger.sequence.v1",
        ("sequence",),
        "length-prefixed streaming sequence of every exact complete jxplanetx.engine.encounter.EncounterProposalRecord dataclass encoding in proposal-index order",
    ),
    (
        "force_ledger_content_sha256",
        "jxplanetx.hybrid-private-encounter-force-ledger.sequence.v1",
        ("sequence",),
        "length-prefixed streaming sequence of every exact complete jxplanetx.engine.evaluator.ForceLedgerEntry dataclass encoding in frozen force-plan ledger order",
    ),
    (
        "primary_counts_content_sha256",
        "jxplanetx.hybrid-private-encounter-primary-counts.payload.v1",
        ("schema", "counts"),
        "one canonical JSON dict with exactly these keys; counts is the exact complete frozen EncounterExecutionCounts dataclass encoding",
    ),
    (
        "accepted_steps_content_sha256",
        "jxplanetx.hybrid-private-encounter-accepted-steps.sequence.v1",
        ("sequence",),
        "length-prefixed streaming sequence of each accepted exact built-in binary64 signed step in acceptance order",
    ),
    (
        "private_execution_content_sha256",
        "jxplanetx.hybrid-private-encounter-execution-manifest.payload.v1",
        (
            "schema",
            "method_id",
            "outer_step_index",
            "segment_spec",
            "summary",
            "component_digests",
        ),
        "one canonical JSON dict with exactly these keys; method_id is the exact encounter method string,outer_step_index is a positive bounded built-in integer,segment_spec is the exact complete AdaptiveEncounterSegmentSpec dataclass encoding; summary is an exact str-key dict containing exactly HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER names with nonnegative built-in integer values; component_digests is an exact str-key dict containing exactly the preceding six digest-field names with lowercase SHA-256 values in digest-roster order; excludes private_execution_content_sha256",
    ),
)
HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER: tuple[
    HybridPrivateEncounterSummarySourceRow, ...
] = (
    ("proposal_count", "EncounterExecutionCounts.substep_proposals"),
    ("accepted_substep_count", "EncounterExecutionCounts.accepted_substeps"),
    ("rejected_substep_count", "EncounterExecutionCounts.rejected_substeps"),
    ("force_evaluations", "EncounterExecutionCounts.force_evaluations"),
    (
        "general_rational_operation_count",
        "EncounterExecutionCounts.total_rational_operations",
    ),
    ("dyadic_operation_count", "EncounterExecutionCounts.total_dyadic_operations"),
    ("gcd_iteration_count", "EncounterExecutionCounts.total_gcd_iterations"),
    ("transcript_byte_count", "EncounterExecutionCounts.total_transcript_bytes"),
    (
        "maximum_integer_bits",
        "max(initialization.maximum_integer_bits,proposal[].maximum_integer_bits;empty proposal default initialization)",
    ),
    (
        "maximum_rational_exponent_magnitude",
        "max(initialization.maximum_rational_exponent_magnitude,proposal[].maximum_rational_exponent_magnitude;empty proposal default initialization)",
    ),
    (
        "streamed_witness_ledger_bytes",
        "canonical_json_bytes(initialization)+sum(canonical_json_bytes(proposal[]) in order)",
    ),
)
HYBRID_PRIVATE_ENCOUNTER_DIGEST_POLICY = (
    "HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER is ordered one-to-one with "
    "HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER and every component uses its "
    "own fixed domain plus its exactly aligned retained preimage schema; every "
    "ordinary payload schema key and every sequence framing schema ID equals the "
    "second field of its aligned preimage row exactly; ordinary "
    "payloads use the canonical serialization policy and sequences use the exact "
    "length-prefixed streaming policy; the aggregate "
    "private_execution_content_sha256 preimage binds method_id exactly equal to "
    "ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,outer "
    "step index,complete derived AdaptiveEncounterSegmentSpec,all summary fields "
    "mapped by HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER,and the preceding "
    "six labeled component digests in roster order,and explicitly excludes the "
    "aggregate digest field itself; these hashes provide content integrity and "
    "replay provenance only,not authentication; the future serializer must pin "
    "independent literal preimage-byte,length,and SHA-256 known-answer tests for "
    "all seven schemas before publication"
)
HYBRID_PRIVATE_ENCOUNTER_LITERAL_KAT_POLICY = (
    "before any runtime or public API publication,tests must hard-code for each "
    "of the seven private schemas one independent literal complete preimage byte "
    "string,its exact byte length,and lowercase SHA-256 result,and must reject "
    "mutations of domain,schema,key roster,dataclass identity or field,array "
    "dtype/shape/C-order,signed-zero float hex,sequence count,item length,item "
    "order,summary source,component label,and aggregate self-exclusion"
)
HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_POLICY = (
    "each private summary value is copied exactly from or reduced exactly over "
    "the frozen private encounter primary executor sources named in the retained "
    "source roster; maxima use the stated initialization default and the witness "
    "byte count is the exact ordered canonical-byte sum; the aggregate digest "
    "binds both the summary and its full streamed component digests"
)
HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN = (
    "jxplanetx.hybrid-wh-rkf78-result.content-integrity.v1"
)

HYBRID_CLAIM_SCOPE = (
    "DETERMINISTIC_WHOLE_OUTER_STEP_MODE_COMPOSITION_OF_THE_FROZEN_WH_MAP_AND_"
    "FROZEN_LOCAL_IVP_ENCOUNTER_SEGMENT_WITH_EXACT_REPLAY_PROVENANCE"
)
HYBRID_NONCLAIMS = (
    "NO_GLOBAL_SYMPLECTICITY;NO_EXACT_OR_FORMAL_GLOBAL_REVERSIBILITY;NO_DENSE_"
    "OUTPUT;NO_EVENT_LOCATION;NO_COLLISION_DETECTION_OR_RESPONSE;NO_"
    "REGULARIZATION;NO_GLOBAL_CLEARANCE;NO_GLOBAL_CONVERGENCE_ORDER_CLAIM;NO_"
    "QUALIFICATION;NO_ACCURACY_OR_PERFORMANCE_SUPERIORITY"
)
HYBRID_EVIDENCE_CLASS = "MODEL_OUTPUT"


def _required_caller_fields() -> tuple[str, ...]:
    return tuple(
        descriptor.name
        for descriptor in fields(AdaptiveEncounterSegmentSpec)
        if descriptor.default is MISSING and descriptor.default_factory is MISSING
    )


def _validate_control_schema() -> None:
    expected = (
        HYBRID_ENCOUNTER_INTERVAL_BINDING_FIELDS
        + HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER
    )
    if _required_caller_fields() != expected:
        raise ContractError(
            "frozen AdaptiveEncounterSegmentSpec caller-field schema changed"
        )
    if tuple(
        descriptor.name for descriptor in fields(HybridEncounterControlProfile)
    ) != HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER:
        raise ContractError("HybridEncounterControlProfile field roster changed")


def _validate_far_probe_mapping() -> None:
    expected_keys = (
        (("FAR_PASS", HYBRID_ALL_FAR_REASON),)
        + tuple(("NEAR_SWITCH", reason) for reason in HYBRID_DYNAMIC_SWITCH_REASONS)
        + tuple(("FATAL_FAILURE", reason) for reason in HYBRID_FAR_PROBE_FATAL_REASONS)
    )
    actual_keys: list[tuple[str, str]] = []
    for entry in HYBRID_FAR_PROBE_LEGAL_MAPPING:
        if (
            type(entry) is not tuple
            or len(entry) != 4
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            or type(entry[2]) is not tuple
            or not entry[2]
            or any(type(phase) is not str for phase in entry[2])
            or type(entry[3]) is not str
        ):
            raise ContractError("far-probe legal mapping entry has an invalid schema")
        if any(phase not in HYBRID_FAR_PROBE_PHASES for phase in entry[2]):
            raise ContractError("far-probe legal mapping names an unknown phase")
        if entry[3] not in HYBRID_TRIGGER_RULES:
            raise ContractError("far-probe legal mapping names an unknown trigger rule")
        actual_keys.append((entry[0], entry[1]))
    if tuple(actual_keys) != expected_keys:
        raise ContractError("far-probe legal mapping is not closed and exhaustive")


def _validate_hybrid_policy_rosters() -> None:
    maximum_names: list[str] = []
    for row in HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise ContractError("far-probe maximum formula row has an invalid schema")
        maximum_names.append(row[0])
    if tuple(maximum_names) != HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER:
        raise ContractError("far-probe maximum formulas do not close the counter roster")

    if tuple(
        descriptor.name for descriptor in fields(HybridKeplerProbeRecord)
    ) != HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER:
        raise ContractError("HybridKeplerProbeRecord field roster changed")
    if (
        type(HYBRID_KEPLER_PROBE_TERMINAL_STATUSES) is not tuple
        or len(HYBRID_KEPLER_PROBE_TERMINAL_STATUSES) != 4
        or any(
            type(status) is not str or not status
            for status in HYBRID_KEPLER_PROBE_TERMINAL_STATUSES
        )
        or len(set(HYBRID_KEPLER_PROBE_TERMINAL_STATUSES))
        != len(HYBRID_KEPLER_PROBE_TERMINAL_STATUSES)
    ):
        raise ContractError("Kepler probe terminal-status roster is invalid")
    failure_rows = HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON
    expected_failure_reasons = (
        "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
        "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
        "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
    )
    if type(failure_rows) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
            or row[1] not in HYBRID_KEPLER_PROBE_TERMINAL_STATUSES[1:]
            for row in failure_rows
        ):
        raise ContractError("Kepler failure-status mapping has an invalid schema")
    if tuple(row[0] for row in failure_rows) != expected_failure_reasons:
        raise ContractError("Kepler failure-status mapping is not exact and closed")

    floor_phases: list[str] = []
    for row in HYBRID_FAR_PROBE_PHASE_WORK_FLOORS:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not tuple
        ):
            raise ContractError("far-probe phase-floor row has an invalid schema")
        floor_phases.append(row[0])
        for item in row[1]:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or item[0] not in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
                or type(item[1]) is not str
            ):
                raise ContractError("far-probe phase-floor item has an invalid schema")
    if tuple(floor_phases) != HYBRID_FAR_PROBE_PHASES:
        raise ContractError("far-probe phase floors do not close the phase roster")

    digest_names: list[str] = []
    digest_domains: list[str] = []
    for row in HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER:
        if (
            type(row) is not tuple
            or len(row) != 3
            or any(type(item) is not str or not item for item in row)
        ):
            raise ContractError("private encounter digest-domain row is invalid")
        digest_names.append(row[0])
        digest_domains.append(row[1])
    if tuple(digest_names) != HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER:
        raise ContractError("private digest domains do not close the digest roster")
    if len(set(digest_domains)) != len(digest_domains):
        raise ContractError("private encounter digest domains must be distinct")

    preimage_names: list[str] = []
    preimage_schema_ids: list[str] = []
    for row in HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER:
        if (
            type(row) is not tuple
            or len(row) != 4
            or type(row[0]) is not str
            or type(row[1]) is not str
            or type(row[2]) is not tuple
            or not row[2]
            or any(type(key) is not str or not key for key in row[2])
            or type(row[3]) is not str
            or not row[3]
        ):
            raise ContractError("private encounter preimage-schema row is invalid")
        preimage_names.append(row[0])
        preimage_schema_ids.append(row[1])
    if tuple(preimage_names) != HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER:
        raise ContractError("private preimage schemas do not close the digest roster")
    if len(set(preimage_schema_ids)) != len(preimage_schema_ids):
        raise ContractError("private encounter preimage schema IDs must be distinct")
    if HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER[-1][2] != (
        "schema",
        "method_id",
        "outer_step_index",
        "segment_spec",
        "summary",
        "component_digests",
    ):
        raise ContractError("private aggregate manifest key roster changed")

    expected_summary_names = (
        "proposal_count",
        "accepted_substep_count",
        "rejected_substep_count",
        "force_evaluations",
        "general_rational_operation_count",
        "dyadic_operation_count",
        "gcd_iteration_count",
        "transcript_byte_count",
        "maximum_integer_bits",
        "maximum_rational_exponent_magnitude",
        "streamed_witness_ledger_bytes",
    )
    summary_names: list[str] = []
    for row in HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
            or not row[1]
        ):
            raise ContractError("private encounter summary-source row is invalid")
        summary_names.append(row[0])
    if tuple(summary_names) != expected_summary_names:
        raise ContractError("private summary sources do not close the summary roster")


def _profile_body_count(profile: HybridEncounterControlProfile) -> int:
    for name in (
        "pair_certification_floors",
        "pair_position_atols",
        "pair_velocity_atols",
    ):
        value = getattr(profile, name)
        if type(value) is not tuple:
            raise ContractError(f"{name} must be an exact tuple")
    pair_count = len(profile.pair_certification_floors)
    if pair_count < 1 or pair_count > ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT:
        raise ContractError("pair tables have an invalid hard-bounded length")
    discriminant = 1 + 8 * pair_count
    root = math.isqrt(discriminant)
    if root * root != discriminant or (1 + root) % 2 != 0:
        raise ContractError("pair-table length is not a canonical N*(N-1)/2")
    body_count = (1 + root) // 2
    if body_count < 2 or body_count > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise ContractError("pair-table length implies an invalid body count")
    return body_count


def _bind_encounter_segment(
    *,
    profile: HybridEncounterControlProfile,
    body_order: tuple[str, ...],
    initial_epoch: float,
    endpoint_epoch: float,
    duration: float,
) -> AdaptiveEncounterSegmentSpec:
    values = {
        name: getattr(profile, name)
        for name in HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER
    }
    return AdaptiveEncounterSegmentSpec(
        body_order=body_order,
        initial_epoch=initial_epoch,
        endpoint_epoch=endpoint_epoch,
        duration=duration,
        **values,
    )


HYBRID_FAR_PROBE_DECISION_FIELD_ROSTER = (
    "body_count",
    "outcome",
    "reason",
    "phase",
    "body_indices",
    "pair_indices",
    "work",
)
HYBRID_PRIVATE_ENCOUNTER_RECORD_FIELD_ROSTER = (
    "outer_step_index",
    "segment_spec",
    "accepted_signed_steps",
    "proposal_count",
    "accepted_substep_count",
    "rejected_substep_count",
    "force_evaluations",
    "general_rational_operation_count",
    "dyadic_operation_count",
    "gcd_iteration_count",
    "transcript_byte_count",
    "maximum_integer_bits",
    "maximum_rational_exponent_magnitude",
    "streamed_witness_ledger_bytes",
    *HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER,
)


def _exact_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a nonnegative built-in integer")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    checked = _exact_nonnegative_int(value, label)
    if checked == 0:
        raise ContractError(f"{label} must be positive")
    return checked


def _exact_positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ContractError(f"{label} must be a positive finite built-in float")
    return value


def _exact_positive_float_table(
    value: object, label: str, expected_length: int
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != expected_length:
        raise ContractError(
            f"{label} must be an exact tuple of {expected_length} built-in floats"
        )
    for index, item in enumerate(value):
        _exact_positive_float(item, f"{label}[{index}]")
    return value


def _validate_profile_binary64_rational_envelope(
    value: float,
    resources: EncounterExactRationalResourceSpec,
    label: str,
) -> None:
    numerator, denominator = value.as_integer_ratio()
    if (
        abs(numerator).bit_length() > resources.maximum_integer_bits
        or denominator.bit_length() > resources.maximum_integer_bits
    ):
        raise ContractError(f"{label} exceeds maximum_integer_bits as a ratio")
    if numerator == 0:
        exponent = 0
    else:
        absolute_numerator = abs(numerator)
        exponent = absolute_numerator.bit_length() - denominator.bit_length()
        if exponent >= 0:
            if absolute_numerator < (denominator << exponent):
                exponent -= 1
        elif (absolute_numerator << (-exponent)) < denominator:
            exponent -= 1
    if abs(exponent) > resources.maximum_rational_exponent_magnitude:
        raise ContractError(f"{label} exceeds maximum_rational_exponent_magnitude")


def _exact_builtin_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_builtin_equal(left, right)
            for left, right in zip(value, expected)  # type: ignore[arg-type]
        )
    return bool(value == expected)


def _validate_body_indices(value: object, body_count: int) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ContractError("body_indices must be an exact tuple")
    if len(value) > body_count:
        raise ContractError("body_indices exceeds its body-count bound")
    if any(type(index) is not int or not 0 <= index < body_count for index in value):
        raise ContractError("body_indices entries are outside the exact body order")
    if any(right <= left for left, right in zip(value, value[1:])):
        raise ContractError("body_indices must be strictly increasing and unique")
    return value


def _validate_pair_indices(
    value: object, body_count: int
) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple:
        raise ContractError("pair_indices must be an exact tuple")
    if len(value) > body_count * (body_count - 1) // 2:
        raise ContractError("pair_indices exceeds its canonical pair-count bound")
    for pair in value:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
            or not 0 <= pair[0] < pair[1] < body_count
        ):
            raise ContractError("pair_indices entries must be canonical exact (i,j) pairs")
    if any(right <= left for left, right in zip(value, value[1:])):
        raise ContractError("pair_indices must be lexicographically sorted and unique")
    return value


def _require_zero_work(work: HybridFarProbeWork, names: tuple[str, ...]) -> None:
    if any(getattr(work, name) != 0 for name in names):
        raise ContractError("far-probe work is nonzero beyond its terminal phase")


def _work_formula_value(
    formula: str, work: HybridFarProbeWork, body_count: int
) -> int:
    secondary_count = body_count - 1
    rebind = work.cartesian_to_jacobi_transforms_completed
    fixed = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "R": rebind,
        "R+1": rebind + 1,
        "1+R": 1 + rebind,
        "2+R": 2 + rebind,
        "body_count-1": secondary_count,
        f"{WH_KEPLER_MAXIMUM_ITERATIONS}*(body_count-1)": (
            WH_KEPLER_MAXIMUM_ITERATIONS * secondary_count
        ),
        f"{WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS}*(body_count-1)": (
            WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS * secondary_count
        ),
        f"{1 + WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS + WH_KEPLER_MAXIMUM_ITERATIONS}*(body_count-1)": (
            (
                1
                + WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS
                + WH_KEPLER_MAXIMUM_ITERATIONS
            )
            * secondary_count
        ),
        f"4*{WH_KEPLER_SERIES_TERM_COUNT}*universal_g_bundle_calls_entered": (
            4
            * WH_KEPLER_SERIES_TERM_COUNT
            * work.universal_g_bundle_calls_entered
        ),
    }
    try:
        return fixed[formula]
    except KeyError as exc:
        raise ContractError("retained far-probe work formula is not recognized") from exc


def _enforce_declarative_probe_tables(
    work: HybridFarProbeWork, body_count: int, phase: str
) -> None:
    for name, formula in HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS:
        if getattr(work, name) > _work_formula_value(formula, work, body_count):
            raise ContractError(f"{name} exceeds its retained per-probe maximum")
    matching = tuple(
        requirements
        for retained_phase, requirements in HYBRID_FAR_PROBE_PHASE_WORK_FLOORS
        if retained_phase == phase
    )
    if len(matching) != 1:
        raise ContractError("terminal phase has no unique retained work-floor row")
    for name, formula in matching[0]:
        if getattr(work, name) < _work_formula_value(formula, work, body_count):
            raise ContractError(f"{name} is below its retained terminal-phase floor")


def _validate_probe_work_for_phase(
    work: HybridFarProbeWork, body_count: int, phase: str, reason: str
) -> None:
    secondary_count = body_count - 1
    _enforce_declarative_probe_tables(work, body_count, phase)
    if not (
        work.interaction_force_assemblies
        <= work.force_evaluations_completed
        <= work.force_calls_entered
    ):
        raise ContractError("far-probe force work violates entered/completed order")
    entered = work.kepler_subflow_calls_entered
    completed = work.kepler_subflow_solves_completed
    if not (
        work.cartesian_to_jacobi_transforms_completed
        <= work.cartesian_to_jacobi_calls_entered
        <= work.cartesian_to_jacobi_transforms_completed + 1
        and work.jacobi_to_cartesian_transforms_completed
        <= work.jacobi_to_cartesian_calls_entered
        <= work.jacobi_to_cartesian_transforms_completed + 1
    ):
        raise ContractError("coordinate-transform entered/completed accounting is invalid")
    if not (
        work.second_half_kicks_completed
        <= work.path_guard_evaluations
        <= work.jacobi_to_cartesian_transforms_completed
        and work.second_half_kicks_completed
        <= work.center_of_mass_drifts_completed
        <= work.first_half_kicks_completed
    ):
        raise ContractError("far-probe flow counters violate causal order")

    rebind = work.cartesian_to_jacobi_transforms_completed
    base_force = rebind
    base_node_guards = rebind
    later_flow = (
        "kepler_subflow_calls_entered",
        "kepler_subflow_solves_completed",
        "universal_solver_iterations",
        "universal_solver_bracket_expansions",
        "universal_g_bundle_calls_entered",
        "universal_g_bundle_calls_completed",
        "universal_series_terms_evaluated",
        "jacobi_to_cartesian_calls_entered",
        "jacobi_to_cartesian_transforms_completed",
        "path_guard_evaluations",
        "center_of_mass_drifts_completed",
        "second_half_kicks_completed",
    )
    after_drift = (
        "jacobi_to_cartesian_calls_entered",
        "jacobi_to_cartesian_transforms_completed",
        "path_guard_evaluations",
        "second_half_kicks_completed",
    )

    if phase == "STATIC_PREFLIGHT":
        _require_zero_work(work, HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER)
        return

    if phase == "ACCEPTED_START_REBIND":
        if not (
            work.cartesian_to_jacobi_calls_entered == 1
            and work.cartesian_to_jacobi_transforms_completed == 0
        ):
            raise ContractError("initial/rebind transform failure has impossible work")
        _require_zero_work(
            work,
            tuple(
                name
                for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
                if name != "cartesian_to_jacobi_calls_entered"
            ),
        )
        return

    if phase == "ACCEPTED_START_NODE":
        later_names = tuple(
            name
            for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
            if name
            not in (
                "force_calls_entered",
                "force_evaluations_completed",
                "interaction_force_assemblies",
                "cartesian_to_jacobi_calls_entered",
                "cartesian_to_jacobi_transforms_completed",
                "node_guard_evaluations",
            )
        )
        _require_zero_work(work, later_names)
        common = (
            rebind == 1
            and work.cartesian_to_jacobi_calls_entered == 1
        )
        if reason == "FORCE_EVALUATION_OR_IDENTITY_FAILURE":
            valid = (
                common
                and work.force_calls_entered == 1
                and work.force_evaluations_completed in (0, 1)
                and work.interaction_force_assemblies == 0
                and work.node_guard_evaluations == 0
            )
        elif reason == "TRANSLATION_FORCE_RESIDUAL_FAILURE":
            valid = (
                common
                and work.force_calls_entered == 1
                and work.force_evaluations_completed == 1
                and work.interaction_force_assemblies == 1
                and work.node_guard_evaluations == 0
            )
        elif reason in (
            "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE",
        ):
            valid = common and (
                (
                    work.force_calls_entered == 1
                    and work.force_evaluations_completed == 1
                    and work.interaction_force_assemblies == 0
                    and work.node_guard_evaluations == 0
                )
                or (
                    work.force_calls_entered == 1
                    and work.force_evaluations_completed == 1
                    and work.interaction_force_assemblies == 1
                    and work.node_guard_evaluations == 1
                )
            )
        else:
            valid = (
                common
                and work.force_calls_entered == 1
                and work.force_evaluations_completed == 1
                and work.interaction_force_assemblies == 1
                and work.node_guard_evaluations == 1
            )
        if not valid:
            raise ContractError("accepted-start decision has impossible work")
        return

    if not (
        work.cartesian_to_jacobi_calls_entered == rebind
        and work.force_calls_entered >= base_force
        and work.force_evaluations_completed >= base_force
        and work.interaction_force_assemblies >= base_force
        and work.node_guard_evaluations >= base_node_guards
    ):
        raise ContractError("probe omitted required initial/rebind work")

    if phase == "FIRST_HALF_KICK":
        if not (
            work.force_calls_entered == base_force
            and work.force_evaluations_completed == base_force
            and work.interaction_force_assemblies == base_force
            and work.node_guard_evaluations == base_node_guards
            and work.first_half_kicks_completed == 0
        ):
            raise ContractError("first-half-kick failure has impossible work")
        _require_zero_work(work, later_flow)
        return

    if phase == "POST_FIRST_KICK_PREDRIFT_NODE":
        if not (
            work.force_calls_entered == base_force
            and work.force_evaluations_completed == base_force
            and work.interaction_force_assemblies == base_force
            and work.first_half_kicks_completed == 1
            and work.node_guard_evaluations == 1 + base_node_guards
        ):
            raise ContractError("predrift-node decision has impossible work")
        _require_zero_work(work, later_flow)
        return

    if phase == "KEPLER_DRIFT_FLOW":
        common = (
            work.force_calls_entered == base_force
            and work.force_evaluations_completed == base_force
            and work.interaction_force_assemblies == base_force
            and work.first_half_kicks_completed == 1
            and work.node_guard_evaluations == 1 + base_node_guards
        )
        if reason == (
            "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE"
        ):
            valid_flow = (
                work.center_of_mass_drifts_completed == 1
                and 1 <= entered <= secondary_count
                and entered == completed + 1
            )
        else:
            valid_flow = (
                work.center_of_mass_drifts_completed in (0, 1)
                and 0 <= entered <= secondary_count
                and (
                    work.center_of_mass_drifts_completed == 1 or entered == 0
                )
            )
        if not (common and valid_flow):
            raise ContractError("Kepler-flow failure has impossible work")
        _require_zero_work(work, after_drift)
        return

    common_complete_drift = (
        work.first_half_kicks_completed == 1
        and work.center_of_mass_drifts_completed == 1
        and work.kepler_subflow_calls_entered == secondary_count
        and work.kepler_subflow_solves_completed == secondary_count
        and work.node_guard_evaluations >= 1 + base_node_guards
    )
    if not common_complete_drift:
        raise ContractError("post-drift decision omitted completed drift work")

    if phase == "POST_DRIFT_CARTESIAN_RECONSTRUCTION":
        if not (
            work.jacobi_to_cartesian_calls_entered == 1
            and work.jacobi_to_cartesian_transforms_completed == 0
            and work.force_calls_entered == base_force
            and work.force_evaluations_completed == base_force
            and work.interaction_force_assemblies == base_force
            and work.node_guard_evaluations == 1 + base_node_guards
            and work.path_guard_evaluations == 0
            and work.second_half_kicks_completed == 0
        ):
            raise ContractError("post-drift reconstruction failure has impossible work")
        return

    if phase == "POST_DRIFT_PRE_FORCE_PATH_SCREEN":
        if not (
            work.jacobi_to_cartesian_calls_entered == 1
            and work.jacobi_to_cartesian_transforms_completed == 1
            and work.force_calls_entered == base_force
            and work.force_evaluations_completed == base_force
            and work.interaction_force_assemblies == base_force
            and work.node_guard_evaluations == 1 + base_node_guards
            and work.path_guard_evaluations == 1
            and work.second_half_kicks_completed == 0
        ):
            raise ContractError("pre-force path-screen decision has impossible work")
        return

    if phase == "POST_DRIFT_CANDIDATE_FORCE":
        common = (
            work.jacobi_to_cartesian_calls_entered == 1
            and work.jacobi_to_cartesian_transforms_completed == 1
            and work.force_calls_entered == base_force + 1
            and work.node_guard_evaluations == 1 + base_node_guards
            and work.path_guard_evaluations == 1
            and work.second_half_kicks_completed == 0
        )
        if reason == "FORCE_EVALUATION_OR_IDENTITY_FAILURE":
            valid_force = (
                base_force
                <= work.force_evaluations_completed
                <= base_force + 1
                and work.interaction_force_assemblies == base_force
            )
        elif reason == "TRANSLATION_FORCE_RESIDUAL_FAILURE":
            valid_force = (
                work.force_evaluations_completed == base_force + 1
                and work.interaction_force_assemblies == base_force + 1
            )
        else:
            valid_force = (
                work.force_evaluations_completed == base_force + 1
                and work.interaction_force_assemblies == base_force
            )
        if not (
            common and valid_force
        ):
            raise ContractError("candidate-force failure has impossible work")
        return

    common_complete_candidate_force = (
        work.force_calls_entered == base_force + 1
        and work.force_evaluations_completed == base_force + 1
        and work.interaction_force_assemblies == base_force + 1
        and work.path_guard_evaluations == 1
    )
    if not common_complete_candidate_force:
        raise ContractError("post-force decision omitted completed candidate-force work")

    if phase == "SECOND_HALF_KICK":
        if not (
            work.jacobi_to_cartesian_calls_entered == 1
            and work.jacobi_to_cartesian_transforms_completed == 1
            and work.node_guard_evaluations == 1 + base_node_guards
            and work.second_half_kicks_completed == 0
        ):
            raise ContractError("second-kick decision has impossible work")
        return

    if phase == "POST_SECOND_KICK_CARTESIAN_RECONSTRUCTION":
        if not (
            work.jacobi_to_cartesian_calls_entered == 2
            and work.jacobi_to_cartesian_transforms_completed == 1
            and work.node_guard_evaluations == 1 + base_node_guards
            and work.second_half_kicks_completed == 1
        ):
            raise ContractError("final reconstruction failure has impossible work")
        return

    if phase in ("COMPLETED_CANDIDATE_NODE", "ALL_GUARDS_COMPLETED"):
        if not (
            work.jacobi_to_cartesian_calls_entered == 2
            and work.jacobi_to_cartesian_transforms_completed == 2
            and work.node_guard_evaluations == 2 + base_node_guards
            and work.second_half_kicks_completed == 1
        ):
            raise ContractError("completed-node decision has impossible work")
        return
    raise ContractError("far-probe work names an unknown terminal phase")


def _exact_binary64_sum(values: tuple[float, ...], target: float) -> bool:
    ratios = tuple(value.as_integer_ratio() for value in values)
    common_denominator = max((ratio[1] for ratio in ratios), default=1)
    numerator = sum(
        item_numerator * (common_denominator // item_denominator)
        for item_numerator, item_denominator in ratios
    )
    target_numerator, target_denominator = target.as_integer_ratio()
    return numerator * target_denominator == target_numerator * common_denominator


@dataclass(frozen=True, eq=False)
class HybridKeplerProbeRecord:
    """Exact attempted work for one ordered secondary Kepler subflow."""

    secondary_index: int
    call_completed: bool
    iterations_entered: int
    bracket_expansions_entered: int
    g_bundle_calls_entered: int
    g_bundle_calls_completed: int
    completed_series_bundle_count: int
    interrupted_series_terms: int
    terminal_status: str
    source: str = HYBRID_KEPLER_PROBE_SOURCE

    def __post_init__(self) -> None:
        if type(self) is not HybridKeplerProbeRecord:
            raise ContractError(
                "Kepler probe record must be an exact HybridKeplerProbeRecord"
            )
        if tuple(descriptor.name for descriptor in fields(self)) != (
            HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER
        ):
            raise ContractError("HybridKeplerProbeRecord field roster changed")
        if (
            type(self.secondary_index) is not int
            or not 1
            <= self.secondary_index
            < ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
        ):
            raise ContractError("Kepler secondary index is outside its hard cap")
        if type(self.call_completed) is not bool:
            raise ContractError("call_completed must be an exact built-in bool")
        for name in (
            "iterations_entered",
            "bracket_expansions_entered",
            "g_bundle_calls_entered",
            "g_bundle_calls_completed",
            "completed_series_bundle_count",
            "interrupted_series_terms",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        if self.iterations_entered > WH_KEPLER_MAXIMUM_ITERATIONS:
            raise ContractError("Kepler record iteration count exceeds its hard cap")
        if (
            self.bracket_expansions_entered
            > WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS
        ):
            raise ContractError("Kepler record expansion count exceeds its hard cap")
        if self.source != HYBRID_KEPLER_PROBE_SOURCE or type(self.source) is not str:
            raise ContractError("Kepler probe source is not the exact frozen source")
        if (
            type(self.terminal_status) is not str
            or self.terminal_status not in HYBRID_KEPLER_PROBE_TERMINAL_STATUSES
        ):
            raise ContractError("Kepler probe terminal status is outside its roster")
        if self.call_completed != (self.terminal_status == "COMPLETED"):
            raise ContractError("Kepler completion flag and terminal status disagree")

        iterations = self.iterations_entered
        expansions = self.bracket_expansions_entered
        g_entered = self.g_bundle_calls_entered
        g_completed = self.g_bundle_calls_completed
        seed = g_entered - iterations - expansions
        if seed not in (0, 1):
            raise ContractError("Kepler record has an impossible seed-G cadence")
        if (iterations + expansions > 0 or self.call_completed) and seed != 1:
            raise ContractError("advanced Kepler work requires one entered seed G")
        if not 0 <= g_entered - g_completed <= 1:
            raise ContractError("Kepler record G entered/completed cadence is invalid")
        if self.call_completed:
            if not (
                iterations >= 1
                and g_entered == g_completed == 1 + iterations + expansions
                and self.interrupted_series_terms == 0
            ):
                raise ContractError("completed Kepler record has impossible work")
        elif seed == 0:
            if any((iterations, expansions, g_entered, g_completed)):
                raise ContractError("a pre-seed Kepler failure must have zero G work")
        elif not (
            g_entered == 1 + iterations + expansions
            and g_completed in (g_entered - 1, g_entered)
        ):
            raise ContractError("incomplete Kepler record has impossible G work")

        if self.completed_series_bundle_count > g_completed:
            raise ContractError("completed series bundles exceed completed G bundles")
        partial = self.interrupted_series_terms
        if g_entered == g_completed:
            if partial != 0:
                raise ContractError("partial series work requires one interrupted G")
        else:
            valid_partial = partial == 0 or any(
                64 * order + 2 <= partial <= 64 * order + 64
                for order in range(4)
            )
            if not valid_partial:
                raise ContractError("interrupted series work has impossible cadence")

    @property
    def series_terms_evaluated(self) -> int:
        return (
            4 * WH_KEPLER_SERIES_TERM_COUNT * self.completed_series_bundle_count
            + self.interrupted_series_terms
        )


@dataclass(frozen=True, eq=False)
class HybridFarProbeWork:
    """Actual high-level frozen-WH work attempted by one private probe."""

    force_calls_entered: int
    force_evaluations_completed: int
    interaction_force_assemblies: int
    kepler_subflow_calls_entered: int
    kepler_subflow_solves_completed: int
    universal_solver_iterations: int
    universal_solver_bracket_expansions: int
    universal_g_bundle_calls_entered: int
    universal_g_bundle_calls_completed: int
    universal_series_terms_evaluated: int
    kepler_probe_records: tuple[HybridKeplerProbeRecord, ...]
    cartesian_to_jacobi_calls_entered: int
    cartesian_to_jacobi_transforms_completed: int
    jacobi_to_cartesian_calls_entered: int
    jacobi_to_cartesian_transforms_completed: int
    node_guard_evaluations: int
    path_guard_evaluations: int
    first_half_kicks_completed: int
    center_of_mass_drifts_completed: int
    second_half_kicks_completed: int

    def __post_init__(self) -> None:
        if type(self) is not HybridFarProbeWork:
            raise ContractError("probe work must be an exact HybridFarProbeWork")
        if tuple(descriptor.name for descriptor in fields(self)) != (
            HYBRID_FAR_PROBE_WORK_FIELD_ROSTER
        ):
            raise ContractError("HybridFarProbeWork field roster changed")
        records = self.kepler_probe_records
        if type(records) is not tuple:
            raise ContractError("kepler_probe_records must be an exact tuple")
        if len(records) > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT - 1:
            raise ContractError("Kepler probe record tuple exceeds its hard cap")
        for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER:
            _exact_nonnegative_int(getattr(self, name), name)
        if self.force_evaluations_completed > self.force_calls_entered:
            raise ContractError("completed force evaluations exceed entered calls")
        if self.interaction_force_assemblies > self.force_evaluations_completed:
            raise ContractError("interaction assemblies exceed completed force calls")
        if self.kepler_subflow_solves_completed > self.kepler_subflow_calls_entered:
            raise ContractError("completed Kepler solves exceed entered calls")
        for name, formula in HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS:
            maximum = _work_formula_value(
                formula, self, ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
            )
            if getattr(self, name) > maximum:
                raise ContractError(f"{name} exceeds its absolute hybrid probe cap")
        entered = self.kepler_subflow_calls_entered
        completed = self.kepler_subflow_solves_completed
        if len(records) != entered:
            raise ContractError("Kepler record count differs from entered subflows")
        for expected_index, record in enumerate(records, start=1):
            if type(record) is not HybridKeplerProbeRecord:
                raise ContractError(
                    "Kepler probe tuple entries must be exact records"
                )
            record.__post_init__()
            if record.secondary_index != expected_index:
                raise ContractError("Kepler probe records are not in canonical order")
        if any(not record.call_completed for record in records[:-1]):
            raise ContractError("only the final Kepler probe record may be incomplete")
        if sum(record.call_completed for record in records) != completed:
            raise ContractError("completed Kepler aggregate differs from its records")
        derived_counts = (
            (
                "universal_solver_iterations",
                sum(record.iterations_entered for record in records),
            ),
            (
                "universal_solver_bracket_expansions",
                sum(record.bracket_expansions_entered for record in records),
            ),
            (
                "universal_g_bundle_calls_entered",
                sum(record.g_bundle_calls_entered for record in records),
            ),
            (
                "universal_g_bundle_calls_completed",
                sum(record.g_bundle_calls_completed for record in records),
            ),
            (
                "universal_series_terms_evaluated",
                sum(record.series_terms_evaluated for record in records),
            ),
        )
        for name, derived in derived_counts:
            if getattr(self, name) != derived:
                raise ContractError(f"{name} differs from its Kepler record sum")
        if not (
            self.cartesian_to_jacobi_transforms_completed
            <= self.cartesian_to_jacobi_calls_entered
            <= self.cartesian_to_jacobi_transforms_completed + 1
            and self.jacobi_to_cartesian_transforms_completed
            <= self.jacobi_to_cartesian_calls_entered
            <= self.jacobi_to_cartesian_transforms_completed + 1
        ):
            raise ContractError("coordinate-transform work violates exact cadence")
        if not (
            self.second_half_kicks_completed
            <= self.path_guard_evaluations
            <= self.jacobi_to_cartesian_transforms_completed
            and self.second_half_kicks_completed
            <= self.center_of_mass_drifts_completed
            <= self.first_half_kicks_completed
        ):
            raise ContractError("far-probe flow counters violate causal order")
        if self.kepler_subflow_calls_entered and not (
            self.first_half_kicks_completed == 1
            and self.center_of_mass_drifts_completed == 1
        ):
            raise ContractError("Kepler work requires completed prior kick and COM drift")
        if (
            self.second_half_kicks_completed
            and self.kepler_subflow_solves_completed == 0
        ):
            raise ContractError("second-half completion requires completed Kepler work")


@dataclass(frozen=True, eq=False)
class HybridFarProbeDecision:
    """Closed typed terminal outcome from one private WH far probe."""

    body_count: int
    outcome: str
    reason: str
    phase: str
    body_indices: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    work: HybridFarProbeWork

    def __post_init__(self) -> None:
        if type(self) is not HybridFarProbeDecision:
            raise ContractError("probe decision must be exact HybridFarProbeDecision")
        if tuple(descriptor.name for descriptor in fields(self)) != (
            HYBRID_FAR_PROBE_DECISION_FIELD_ROSTER
        ):
            raise ContractError("HybridFarProbeDecision field roster changed")
        _validate_far_probe_mapping()
        _validate_hybrid_policy_rosters()
        if (
            type(self.body_count) is not int
            or not 2 <= self.body_count <= ENCOUNTER_HARD_MAXIMUM_BODY_COUNT
        ):
            raise ContractError("decision body_count is outside the hybrid hard cap")
        for name in ("outcome", "reason", "phase"):
            if type(getattr(self, name)) is not str:
                raise ContractError(f"{name} must be an exact built-in string")
        if type(self.work) is not HybridFarProbeWork:
            raise ContractError("work must be an exact HybridFarProbeWork")
        records = self.work.kepler_probe_records
        if type(records) is not tuple:
            raise ContractError("kepler_probe_records must be an exact tuple")
        if len(records) > self.body_count - 1:
            raise ContractError(
                "Kepler probe record tuple exceeds the decision body bound"
            )
        self.work.__post_init__()
        bodies = _validate_body_indices(self.body_indices, self.body_count)
        pairs = _validate_pair_indices(self.pair_indices, self.body_count)
        matching = tuple(
            entry
            for entry in HYBRID_FAR_PROBE_LEGAL_MAPPING
            if entry[0] == self.outcome and entry[1] == self.reason
        )
        if len(matching) != 1:
            raise ContractError("probe outcome/reason is outside the closed mapping")
        _outcome, _reason, allowed_phases, trigger_rule = matching[0]
        if self.phase not in allowed_phases:
            raise ContractError("probe phase is illegal for its outcome/reason")
        if trigger_rule == "EMPTY":
            valid_triggers = not bodies and not pairs
        elif trigger_rule == "BODY_INDICES_NONEMPTY_PAIR_INDICES_EMPTY":
            valid_triggers = bool(bodies) and not pairs
        elif trigger_rule == "BODY_INDICES_EMPTY_PAIR_INDICES_NONEMPTY":
            valid_triggers = not bodies and bool(pairs)
        else:
            raise ContractError("closed trigger rule is not recognized")
        if not valid_triggers:
            raise ContractError("probe trigger tuples violate their closed mapping")
        if self.reason in HYBRID_BODY_SWITCH_REASONS and any(
            index == 0 for index in bodies
        ):
            raise ContractError("WH/Jacobi body-envelope triggers exclude body zero")
        if self.phase == "KEPLER_DRIFT_FLOW":
            incomplete = bool(records and not records[-1].call_completed)
            dedicated = self.reason == (
                "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE"
            )
            if dedicated and not incomplete:
                raise ContractError(
                    "dedicated Kepler failure needs one incomplete record"
                )
            if incomplete:
                expected_status = dict(
                    HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON
                ).get(self.reason)
                if (
                    expected_status is None
                    or records[-1].terminal_status != expected_status
                ):
                    raise ContractError(
                        "incomplete Kepler status does not match the outer reason"
                    )
            if self.reason in (
                "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
                "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
            ) and (
                not records or bodies != (records[-1].secondary_index,)
            ):
                raise ContractError(
                    "body-local Kepler failure must name its exact final record"
                )
            if (
                self.reason == "NONFINITE_OR_SINGULAR_GLOBAL_NUMERICAL_FAILURE"
                and records
                and not incomplete
                and len(records) != self.body_count - 1
            ):
                raise ContractError(
                    "global post-A failure requires the complete secondary tuple"
                )
        _validate_probe_work_for_phase(
            self.work, self.body_count, self.phase, self.reason
        )


@dataclass(frozen=True, eq=False)
class HybridPrivateEncounterRecord:
    """Bounded digest-only custody for one private near execution."""

    outer_step_index: int
    segment_spec: AdaptiveEncounterSegmentSpec
    accepted_signed_steps: tuple[float, ...]
    proposal_count: int
    accepted_substep_count: int
    rejected_substep_count: int
    force_evaluations: int
    general_rational_operation_count: int
    dyadic_operation_count: int
    gcd_iteration_count: int
    transcript_byte_count: int
    maximum_integer_bits: int
    maximum_rational_exponent_magnitude: int
    streamed_witness_ledger_bytes: int
    final_state_content_sha256: str
    initialization_content_sha256: str
    proposal_ledger_content_sha256: str
    force_ledger_content_sha256: str
    primary_counts_content_sha256: str
    accepted_steps_content_sha256: str
    private_execution_content_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not HybridPrivateEncounterRecord:
            raise ContractError(
                "private child record must be exact HybridPrivateEncounterRecord"
            )
        if tuple(descriptor.name for descriptor in fields(self)) != (
            HYBRID_PRIVATE_ENCOUNTER_RECORD_FIELD_ROSTER
        ):
            raise ContractError("HybridPrivateEncounterRecord field roster changed")
        _validate_hybrid_policy_rosters()
        if (
            type(self.outer_step_index) is not int
            or not 1
            <= self.outer_step_index
            <= HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
        ):
            raise ContractError("outer_step_index is outside the hybrid outer cap")
        if type(self.segment_spec) is not AdaptiveEncounterSegmentSpec:
            raise ContractError("segment_spec must be exact AdaptiveEncounterSegmentSpec")
        self.segment_spec.__post_init__()
        steps = self.accepted_signed_steps
        if (
            type(steps) is not tuple
            or not steps
            or len(steps) > self.segment_spec.maximum_accepted_substeps
        ):
            raise ContractError("accepted_signed_steps has an invalid bounded length")
        direction = 1.0 if self.segment_spec.duration > 0.0 else -1.0
        if any(
            type(step) is not float
            or not math.isfinite(step)
            or direction * step <= 0.0
            for step in steps
        ):
            raise ContractError("accepted_signed_steps contain an invalid signed step")
        maximum_step = self.segment_spec.maximum_step_magnitude
        minimum_step = self.segment_spec.minimum_step_magnitude
        if abs(steps[0]) > self.segment_spec.initial_step_magnitude:
            raise ContractError("the first accepted step exceeds initial_step_magnitude")
        if any(abs(step) > maximum_step for step in steps):
            raise ContractError("an accepted signed step exceeds maximum_step_magnitude")
        for previous, current in zip(steps, steps[1:]):
            allowed_growth = min(
                maximum_step,
                self.segment_spec.maximum_scale_factor * abs(previous),
            )
            if abs(current) > allowed_growth:
                raise ContractError(
                    "an accepted step exceeds the controller growth envelope"
                )
        if any(abs(step) < minimum_step for step in steps[:-1]):
            raise ContractError(
                "only the final deterministic endpoint remainder may be below minimum"
            )
        if not _exact_binary64_sum(steps, self.segment_spec.duration):
            raise ContractError("accepted_signed_steps do not sum exactly to duration")
        count_names = (
            "proposal_count",
            "accepted_substep_count",
            "rejected_substep_count",
            "force_evaluations",
            "general_rational_operation_count",
            "dyadic_operation_count",
            "gcd_iteration_count",
            "transcript_byte_count",
            "maximum_integer_bits",
            "maximum_rational_exponent_magnitude",
            "streamed_witness_ledger_bytes",
        )
        for name in count_names:
            _exact_nonnegative_int(getattr(self, name), name)
        if self.accepted_substep_count != len(steps):
            raise ContractError("accepted_substep_count differs from its ledger")
        if self.proposal_count != (
            self.accepted_substep_count + self.rejected_substep_count
        ):
            raise ContractError("private child proposal accounting is inconsistent")
        if not (
            ENCOUNTER_STAGE_COUNT * self.accepted_substep_count
            <= self.force_evaluations
            <= ENCOUNTER_STAGE_COUNT * self.proposal_count
        ):
            raise ContractError(
                "private child force calls are inconsistent with RKF78 proposals"
            )
        child = self.segment_spec
        resources = child.exact_rational_resources
        if (
            self.proposal_count > child.maximum_substep_proposals
            or self.rejected_substep_count > child.maximum_rejected_substeps
            or self.force_evaluations > child.maximum_force_evaluations
            or self.general_rational_operation_count + self.dyadic_operation_count
            > resources.maximum_operations_per_segment
            or self.gcd_iteration_count > resources.maximum_gcd_iterations_per_segment
            or self.transcript_byte_count
            > resources.maximum_witness_transcript_bytes_per_segment
            or self.maximum_integer_bits > resources.maximum_integer_bits
            or self.maximum_rational_exponent_magnitude
            > resources.maximum_rational_exponent_magnitude
            or self.streamed_witness_ledger_bytes > resources.maximum_witness_ledger_bytes
        ):
            raise ContractError("private child summary exceeds a frozen child cap")
        for name in HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER:
            _exact_sha256(getattr(self, name), name)


@dataclass(frozen=True, eq=False)
class HybridEncounterControlProfile:
    """The exact 21 interval-invariant controls, completed by a hybrid spec."""

    initial_step_magnitude: float
    minimum_step_magnitude: float
    maximum_step_magnitude: float
    pair_certification_floors: tuple[float, ...]
    pair_position_atols: tuple[float, ...]
    pair_position_rtol: float
    pair_velocity_atols: tuple[float, ...]
    pair_velocity_rtol: float
    gm_centroid_position_atol: float
    gm_centroid_position_rtol: float
    gm_centroid_velocity_atol: float
    gm_centroid_velocity_rtol: float
    maximum_substep_proposals: int
    maximum_accepted_substeps: int
    maximum_rejected_substeps: int
    maximum_consecutive_rejections: int
    maximum_force_evaluations: int
    safety_factor: float
    minimum_scale_factor: float
    maximum_scale_factor: float
    exact_rational_resources: EncounterExactRationalResourceSpec

    def __post_init__(self) -> None:
        if type(self) is not HybridEncounterControlProfile:
            raise ContractError(
                "encounter control must be an exact HybridEncounterControlProfile"
            )
        _validate_control_schema()
        pair_tables = (
            self.pair_certification_floors,
            self.pair_position_atols,
            self.pair_velocity_atols,
        )
        if any(type(table) is not tuple for table in pair_tables):
            raise ContractError("all three pair tables must be exact tuples")
        pair_count = len(self.pair_certification_floors)
        if not 1 <= pair_count <= ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT:
            raise ContractError("pair tables exceed their hard bounded length")
        if any(len(table) != pair_count for table in pair_tables[1:]):
            raise ContractError("all three pair tables must have equal cardinality")
        _exact_positive_float_table(
            self.pair_certification_floors,
            "pair_certification_floors",
            pair_count,
        )
        _exact_positive_float_table(
            self.pair_position_atols,
            "pair_position_atols",
            pair_count,
        )
        _exact_positive_float_table(
            self.pair_velocity_atols,
            "pair_velocity_atols",
            pair_count,
        )
        initial_step = _exact_positive_float(
            self.initial_step_magnitude, "initial_step_magnitude"
        )
        minimum_step = _exact_positive_float(
            self.minimum_step_magnitude, "minimum_step_magnitude"
        )
        maximum_step = _exact_positive_float(
            self.maximum_step_magnitude, "maximum_step_magnitude"
        )
        if not minimum_step <= initial_step <= maximum_step:
            raise ContractError(
                "step magnitudes must satisfy minimum <= initial <= maximum"
            )
        for name in (
            "pair_position_rtol",
            "pair_velocity_rtol",
            "gm_centroid_position_atol",
            "gm_centroid_position_rtol",
            "gm_centroid_velocity_atol",
            "gm_centroid_velocity_rtol",
            "safety_factor",
            "minimum_scale_factor",
            "maximum_scale_factor",
        ):
            _exact_positive_float(getattr(self, name), name)
        if self.safety_factor >= 1.0:
            raise ContractError("safety_factor must be below one")
        if self.minimum_scale_factor > 1.0:
            raise ContractError("minimum_scale_factor cannot exceed one")
        if self.maximum_scale_factor < 1.0:
            raise ContractError("maximum_scale_factor cannot be below one")
        if self.minimum_scale_factor > self.maximum_scale_factor:
            raise ContractError(
                "minimum_scale_factor cannot exceed maximum_scale_factor"
            )
        if self.pair_position_rtol > 1.0 or self.pair_velocity_rtol > 1.0:
            raise ContractError("pair relative tolerances cannot exceed one")
        if (
            self.gm_centroid_position_rtol > 1.0
            or self.gm_centroid_velocity_rtol > 1.0
        ):
            raise ContractError("GM-centroid relative tolerances cannot exceed one")
        for name in (
            "maximum_substep_proposals",
            "maximum_accepted_substeps",
            "maximum_force_evaluations",
        ):
            _exact_positive_int(getattr(self, name), name)
        for name in (
            "maximum_rejected_substeps",
            "maximum_consecutive_rejections",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        if self.maximum_accepted_substeps > self.maximum_substep_proposals:
            raise ContractError(
                "maximum_accepted_substeps cannot exceed maximum_substep_proposals"
            )
        if self.maximum_rejected_substeps > self.maximum_substep_proposals:
            raise ContractError(
                "maximum_rejected_substeps cannot exceed maximum_substep_proposals"
            )
        if self.maximum_consecutive_rejections > self.maximum_rejected_substeps:
            raise ContractError(
                "maximum_consecutive_rejections cannot exceed maximum_rejected_substeps"
            )
        if self.maximum_substep_proposals > ENCOUNTER_HARD_MAXIMUM_SUBSTEP_PROPOSALS:
            raise ContractError("maximum_substep_proposals exceeds its fixed hard cap")
        if self.maximum_force_evaluations < ENCOUNTER_STAGE_COUNT:
            raise ContractError(
                "maximum_force_evaluations must permit one completed RKF78 attempt"
            )
        if self.maximum_force_evaluations > ENCOUNTER_HARD_MAXIMUM_FORCE_EVALUATIONS:
            raise ContractError("maximum_force_evaluations exceeds its fixed hard cap")
        body_count = _profile_body_count(self)
        if type(self.exact_rational_resources) is not EncounterExactRationalResourceSpec:
            raise ContractError(
                "exact_rational_resources must be an exact "
                "EncounterExactRationalResourceSpec"
            )
        if body_count > self.exact_rational_resources.maximum_body_count:
            raise ContractError("profile pair tables exceed maximum_body_count")
        canonical_pair_count = body_count * (body_count - 1) // 2
        if canonical_pair_count > self.exact_rational_resources.maximum_pair_count:
            raise ContractError("profile pair tables exceed maximum_pair_count")
        self.exact_rational_resources.__post_init__()
        resources = self.exact_rational_resources
        rational_binary64_inputs = (
            (self.initial_step_magnitude, "initial_step_magnitude"),
            (self.minimum_step_magnitude, "minimum_step_magnitude"),
            (self.maximum_step_magnitude, "maximum_step_magnitude"),
            *tuple(
                (value, f"pair_certification_floors[{index}]")
                for index, value in enumerate(self.pair_certification_floors)
            ),
        )
        for value, label in rational_binary64_inputs:
            _validate_profile_binary64_rational_envelope(value, resources, label)
        if (
            (self.maximum_substep_proposals + 1)
            * resources.maximum_witness_diagnostic_bytes_per_proposal
            > resources.maximum_witness_ledger_bytes
        ):
            raise ContractError(
                "the witness ledger cap cannot hold every prospective diagnostic"
            )
        if (
            resources.maximum_initialization_operations
            + self.maximum_substep_proposals
            * resources.maximum_operations_per_proposal
            > resources.maximum_operations_per_segment
        ):
            raise ContractError(
                "the segment work cap cannot reserve every allowed proposal"
            )
        if (
            resources.maximum_initialization_gcd_iterations
            + self.maximum_substep_proposals
            * resources.maximum_gcd_iterations_per_proposal
            > resources.maximum_gcd_iterations_per_segment
        ):
            raise ContractError(
                "the segment GCD cap cannot reserve every allowed proposal"
            )
        if (
            resources.maximum_initialization_transcript_bytes
            + self.maximum_substep_proposals
            * resources.maximum_witness_transcript_bytes_per_proposal
            > resources.maximum_witness_transcript_bytes_per_segment
        ):
            raise ContractError(
                "the segment transcript cap cannot reserve every allowed proposal"
            )


@dataclass(frozen=True, eq=False)
class HybridWisdomHolmanRKF78Spec:
    """Whole-system, whole-macrostep WH/RKF78 hybrid request contract."""

    wisdom_holman_spec: FixedStepWisdomHolmanSpec
    encounter_control: HybridEncounterControlProfile

    method_id: str = HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID
    method_class: str = HYBRID_METHOD_CLASS
    composition: str = HYBRID_COMPOSITION
    wisdom_holman_method_id: str = FIXED_STEP_WISDOM_HOLMAN_METHOD_ID
    encounter_method_id: str = ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID
    encounter_interval_binding_fields: tuple[str, ...] = (
        HYBRID_ENCOUNTER_INTERVAL_BINDING_FIELDS
    )
    encounter_control_binder_field_roster: tuple[str, ...] = (
        HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER
    )
    step_modes: tuple[str, ...] = HYBRID_STEP_MODES
    far_probe_outcomes: tuple[str, ...] = HYBRID_FAR_PROBE_OUTCOMES
    far_probe_phases: tuple[str, ...] = HYBRID_FAR_PROBE_PHASES
    dynamic_guard_phases: tuple[str, ...] = HYBRID_DYNAMIC_GUARD_PHASES
    dynamic_switch_reasons: tuple[str, ...] = HYBRID_DYNAMIC_SWITCH_REASONS
    body_switch_reasons: tuple[str, ...] = HYBRID_BODY_SWITCH_REASONS
    pair_switch_reasons: tuple[str, ...] = HYBRID_PAIR_SWITCH_REASONS
    far_probe_fatal_reasons: tuple[str, ...] = HYBRID_FAR_PROBE_FATAL_REASONS
    trigger_rules: tuple[str, ...] = HYBRID_TRIGGER_RULES
    far_probe_legal_mapping: tuple[HybridFarProbeLegalMappingRow, ...] = (
        HYBRID_FAR_PROBE_LEGAL_MAPPING
    )
    fatal_failure_classes: tuple[str, ...] = HYBRID_FATAL_FAILURE_CLASSES
    far_probe_work_field_roster: tuple[str, ...] = (
        HYBRID_FAR_PROBE_WORK_FIELD_ROSTER
    )
    far_probe_work_counter_field_roster: tuple[str, ...] = (
        HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
    )
    far_probe_work_maximum_formulas: tuple[
        HybridFarProbeWorkFormulaRow, ...
    ] = HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS
    far_probe_phase_work_floors: tuple[
        HybridFarProbePhaseFloorRow, ...
    ] = HYBRID_FAR_PROBE_PHASE_WORK_FLOORS
    far_probe_decision_field_roster: tuple[str, ...] = (
        HYBRID_FAR_PROBE_DECISION_FIELD_ROSTER
    )
    kepler_probe_record_field_roster: tuple[str, ...] = (
        HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER
    )
    kepler_probe_source: str = HYBRID_KEPLER_PROBE_SOURCE
    kepler_probe_terminal_statuses: tuple[str, ...] = (
        HYBRID_KEPLER_PROBE_TERMINAL_STATUSES
    )
    kepler_probe_failure_status_by_reason: tuple[
        HybridKeplerProbeFailureStatusRow, ...
    ] = HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON
    private_encounter_record_field_roster: tuple[str, ...] = (
        HYBRID_PRIVATE_ENCOUNTER_RECORD_FIELD_ROSTER
    )
    private_encounter_digest_field_roster: tuple[str, ...] = (
        HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER
    )
    private_encounter_digest_domain_roster: tuple[
        HybridPrivateEncounterDigestDomainRow, ...
    ] = HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER
    private_encounter_preimage_schema_roster: tuple[
        HybridPrivateEncounterPreimageSchemaRow, ...
    ] = HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER
    private_encounter_summary_source_roster: tuple[
        HybridPrivateEncounterSummarySourceRow, ...
    ] = HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER
    maximum_outer_records_per_lane: int = HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
    maximum_near_macrosteps_per_lane: int = (
        HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE
    )
    maximum_retained_near_accepted_substeps_per_lane: int = (
        HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE
    )
    maximum_retained_near_digest_bytes_per_lane: int = (
        HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE
    )
    maximum_outer_records_public_total: int = (
        HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL
    )
    maximum_near_macrosteps_public_total: int = (
        HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL
    )
    maximum_retained_near_accepted_substeps_public_total: int = (
        HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL
    )
    maximum_retained_near_digest_bytes_public_total: int = (
        HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL
    )
    base_type_composition_policy: str = HYBRID_BASE_TYPE_COMPOSITION_POLICY
    encounter_control_schema_policy: str = HYBRID_ENCOUNTER_CONTROL_SCHEMA_POLICY
    encounter_binding_policy: str = HYBRID_ENCOUNTER_BINDING_POLICY
    canonical_profile_validation_policy: str = (
        HYBRID_CANONICAL_PROFILE_VALIDATION_POLICY
    )
    outer_lattice_policy: str = HYBRID_OUTER_LATTICE_POLICY
    checkpoint_policy: str = HYBRID_CHECKPOINT_POLICY
    switch_policy: str = HYBRID_SWITCH_POLICY
    dynamic_switch_reason_policy: str = HYBRID_DYNAMIC_SWITCH_REASON_POLICY
    probe_transaction_policy: str = HYBRID_PROBE_TRANSACTION_POLICY
    near_replacement_policy: str = HYBRID_NEAR_REPLACEMENT_POLICY
    original_node_admissibility_policy: str = HYBRID_ORIGINAL_NODE_ADMISSIBILITY_POLICY
    far_probe_decision_interface_policy: str = (
        HYBRID_FAR_PROBE_DECISION_INTERFACE_POLICY
    )
    accepted_start_rebind_policy: str = HYBRID_ACCEPTED_START_REBIND_POLICY
    far_probe_work_validation_policy: str = (
        HYBRID_FAR_PROBE_WORK_VALIDATION_POLICY
    )
    far_probe_reason_work_policy: str = HYBRID_FAR_PROBE_REASON_WORK_POLICY
    kepler_probe_record_policy: str = HYBRID_KEPLER_PROBE_RECORD_POLICY
    fatal_failure_policy: str = HYBRID_FATAL_FAILURE_POLICY
    all_far_compatibility_policy: str = HYBRID_ALL_FAR_COMPATIBILITY_POLICY
    provisional_work_accounting_policy: str = (
        HYBRID_PROVISIONAL_WORK_ACCOUNTING_POLICY
    )
    near_ledger_policy: str = HYBRID_NEAR_LEDGER_POLICY
    private_encounter_accepted_step_policy: str = (
        HYBRID_PRIVATE_ENCOUNTER_ACCEPTED_STEP_POLICY
    )
    private_encounter_digest_policy: str = HYBRID_PRIVATE_ENCOUNTER_DIGEST_POLICY
    private_encounter_literal_kat_policy: str = (
        HYBRID_PRIVATE_ENCOUNTER_LITERAL_KAT_POLICY
    )
    private_encounter_summary_source_policy: str = (
        HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_POLICY
    )
    step_ledger_policy: str = HYBRID_STEP_LEDGER_POLICY
    nested_replay_accounting_policy: str = HYBRID_NESTED_REPLAY_ACCOUNTING_POLICY
    execution_accounting_policy: str = HYBRID_EXECUTION_ACCOUNTING_POLICY
    aggregate_resource_policy: str = HYBRID_AGGREGATE_RESOURCE_POLICY
    aggregate_resource_nonclaim_policy: str = (
        HYBRID_AGGREGATE_RESOURCE_NONCLAIM_POLICY
    )
    public_execution_accounting_scope: str = (
        HYBRID_PUBLIC_EXECUTION_ACCOUNTING_SCOPE
    )
    validation_replay_policy: str = HYBRID_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = HYBRID_VALIDATION_REPLAY_COUNT
    canonical_serialization_policy: str = HYBRID_CANONICAL_SERIALIZATION_POLICY
    streaming_sequence_framing_policy: str = (
        HYBRID_STREAMING_SEQUENCE_FRAMING_POLICY
    )
    result_custody_policy: str = HYBRID_RESULT_CUSTODY_POLICY
    checksum_binding_policy: str = HYBRID_CHECKSUM_BINDING_POLICY
    schedule_checksum_algorithm: str = HYBRID_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = HYBRID_SCHEDULE_CHECKSUM_DOMAIN
    step_ledger_checksum_algorithm: str = HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM
    step_ledger_checksum_domain: str = HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN
    private_encounter_checksum_algorithm: str = (
        HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM
    )
    private_encounter_checksum_domain: str = HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_DOMAIN
    result_content_checksum_algorithm: str = HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM
    result_content_checksum_domain: str = HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN
    claim_scope: str = HYBRID_CLAIM_SCOPE
    nonclaims: str = HYBRID_NONCLAIMS
    outer_step_adaptive: bool = False
    encounter_substeps_adaptive: bool = True
    symplectic: bool = False
    formally_symplectic: bool = False
    floating_point_symplectic: bool = False
    globally_symplectic: bool = False
    time_reversible: bool = False
    formally_time_reversible: bool = False
    floating_point_exactly_reversible: bool = False
    exactly_time_reversible: bool = False
    dense_output: bool = False
    event_detection: bool = False
    collision_detection: bool = False
    collision_response: bool = False
    regularized: bool = False
    global_clearance_claimed: bool = False
    global_order_claimed: bool = False
    superiority_claimed: bool = False
    evidence_class: str = HYBRID_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self) is not HybridWisdomHolmanRKF78Spec:
            raise ContractError(
                "hybrid spec must be an exact HybridWisdomHolmanRKF78Spec"
            )
        if type(self.wisdom_holman_spec) is not FixedStepWisdomHolmanSpec:
            raise ContractError(
                "wisdom_holman_spec must be an exact FixedStepWisdomHolmanSpec"
            )
        if type(self.encounter_control) is not HybridEncounterControlProfile:
            raise ContractError(
                "encounter_control must be an exact HybridEncounterControlProfile"
            )
        self.wisdom_holman_spec.__post_init__()
        self.encounter_control.__post_init__()
        _validate_control_schema()
        _validate_far_probe_mapping()
        _validate_hybrid_policy_rosters()

        signed_step = self.wisdom_holman_spec.fixed_step
        canonical_child = _bind_encounter_segment(
            profile=self.encounter_control,
            body_order=self.wisdom_holman_spec.jacobi_body_order,
            initial_epoch=0.0,
            endpoint_epoch=signed_step,
            duration=signed_step,
        )
        outer_steps = self.wisdom_holman_spec.completed_steps
        if (
            self.wisdom_holman_spec.maximum_steps
            > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
            or outer_steps > HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE
        ):
            raise ContractError("WH maximum or completed steps exceed the hybrid outer cap")
        prospective_near_steps = min(
            outer_steps, HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE
        )
        maximum_child_steps = canonical_child.maximum_accepted_substeps
        if prospective_near_steps > (
            HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE
            // maximum_child_steps
        ):
            raise ContractError(
                "prospective near-step product exceeds the retained-substep cap"
            )
        digest_bytes_per_near = 32 * len(
            HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER
        )
        if prospective_near_steps > (
            HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE
            // digest_bytes_per_near
        ):
            raise ContractError(
                "prospective near digest product exceeds the retained-digest cap"
            )
        public_cap_pairs = (
            (
                HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE,
                HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL,
            ),
            (
                HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE,
                HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL,
            ),
            (
                HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE,
                HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL,
            ),
            (
                HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE,
                HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL,
            ),
        )
        if any(public != 2 * lane for lane, public in public_cap_pairs):
            raise ContractError("hybrid public resource caps must be twice one lane")

        exact = {
            "method_id": HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
            "method_class": HYBRID_METHOD_CLASS,
            "composition": HYBRID_COMPOSITION,
            "wisdom_holman_method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
            "encounter_method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "encounter_interval_binding_fields": HYBRID_ENCOUNTER_INTERVAL_BINDING_FIELDS,
            "encounter_control_binder_field_roster": HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER,
            "step_modes": HYBRID_STEP_MODES,
            "far_probe_outcomes": HYBRID_FAR_PROBE_OUTCOMES,
            "far_probe_phases": HYBRID_FAR_PROBE_PHASES,
            "dynamic_guard_phases": HYBRID_DYNAMIC_GUARD_PHASES,
            "dynamic_switch_reasons": HYBRID_DYNAMIC_SWITCH_REASONS,
            "body_switch_reasons": HYBRID_BODY_SWITCH_REASONS,
            "pair_switch_reasons": HYBRID_PAIR_SWITCH_REASONS,
            "far_probe_fatal_reasons": HYBRID_FAR_PROBE_FATAL_REASONS,
            "trigger_rules": HYBRID_TRIGGER_RULES,
            "far_probe_legal_mapping": HYBRID_FAR_PROBE_LEGAL_MAPPING,
            "fatal_failure_classes": HYBRID_FATAL_FAILURE_CLASSES,
            "far_probe_work_field_roster": HYBRID_FAR_PROBE_WORK_FIELD_ROSTER,
            "far_probe_work_counter_field_roster": HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER,
            "far_probe_work_maximum_formulas": HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS,
            "far_probe_phase_work_floors": HYBRID_FAR_PROBE_PHASE_WORK_FLOORS,
            "far_probe_decision_field_roster": HYBRID_FAR_PROBE_DECISION_FIELD_ROSTER,
            "kepler_probe_record_field_roster": HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER,
            "kepler_probe_source": HYBRID_KEPLER_PROBE_SOURCE,
            "kepler_probe_terminal_statuses": HYBRID_KEPLER_PROBE_TERMINAL_STATUSES,
            "kepler_probe_failure_status_by_reason": HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON,
            "private_encounter_record_field_roster": HYBRID_PRIVATE_ENCOUNTER_RECORD_FIELD_ROSTER,
            "private_encounter_digest_field_roster": HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER,
            "private_encounter_digest_domain_roster": HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER,
            "private_encounter_preimage_schema_roster": HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER,
            "private_encounter_summary_source_roster": HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER,
            "maximum_outer_records_per_lane": HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE,
            "maximum_near_macrosteps_per_lane": HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE,
            "maximum_retained_near_accepted_substeps_per_lane": HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE,
            "maximum_retained_near_digest_bytes_per_lane": HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE,
            "maximum_outer_records_public_total": HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL,
            "maximum_near_macrosteps_public_total": HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL,
            "maximum_retained_near_accepted_substeps_public_total": HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL,
            "maximum_retained_near_digest_bytes_public_total": HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL,
            "base_type_composition_policy": HYBRID_BASE_TYPE_COMPOSITION_POLICY,
            "encounter_control_schema_policy": HYBRID_ENCOUNTER_CONTROL_SCHEMA_POLICY,
            "encounter_binding_policy": HYBRID_ENCOUNTER_BINDING_POLICY,
            "canonical_profile_validation_policy": HYBRID_CANONICAL_PROFILE_VALIDATION_POLICY,
            "outer_lattice_policy": HYBRID_OUTER_LATTICE_POLICY,
            "checkpoint_policy": HYBRID_CHECKPOINT_POLICY,
            "switch_policy": HYBRID_SWITCH_POLICY,
            "dynamic_switch_reason_policy": HYBRID_DYNAMIC_SWITCH_REASON_POLICY,
            "probe_transaction_policy": HYBRID_PROBE_TRANSACTION_POLICY,
            "near_replacement_policy": HYBRID_NEAR_REPLACEMENT_POLICY,
            "original_node_admissibility_policy": HYBRID_ORIGINAL_NODE_ADMISSIBILITY_POLICY,
            "far_probe_decision_interface_policy": HYBRID_FAR_PROBE_DECISION_INTERFACE_POLICY,
            "accepted_start_rebind_policy": HYBRID_ACCEPTED_START_REBIND_POLICY,
            "far_probe_work_validation_policy": HYBRID_FAR_PROBE_WORK_VALIDATION_POLICY,
            "far_probe_reason_work_policy": HYBRID_FAR_PROBE_REASON_WORK_POLICY,
            "kepler_probe_record_policy": HYBRID_KEPLER_PROBE_RECORD_POLICY,
            "fatal_failure_policy": HYBRID_FATAL_FAILURE_POLICY,
            "all_far_compatibility_policy": HYBRID_ALL_FAR_COMPATIBILITY_POLICY,
            "provisional_work_accounting_policy": HYBRID_PROVISIONAL_WORK_ACCOUNTING_POLICY,
            "near_ledger_policy": HYBRID_NEAR_LEDGER_POLICY,
            "private_encounter_accepted_step_policy": HYBRID_PRIVATE_ENCOUNTER_ACCEPTED_STEP_POLICY,
            "private_encounter_digest_policy": HYBRID_PRIVATE_ENCOUNTER_DIGEST_POLICY,
            "private_encounter_literal_kat_policy": HYBRID_PRIVATE_ENCOUNTER_LITERAL_KAT_POLICY,
            "private_encounter_summary_source_policy": HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_POLICY,
            "step_ledger_policy": HYBRID_STEP_LEDGER_POLICY,
            "nested_replay_accounting_policy": HYBRID_NESTED_REPLAY_ACCOUNTING_POLICY,
            "execution_accounting_policy": HYBRID_EXECUTION_ACCOUNTING_POLICY,
            "aggregate_resource_policy": HYBRID_AGGREGATE_RESOURCE_POLICY,
            "aggregate_resource_nonclaim_policy": HYBRID_AGGREGATE_RESOURCE_NONCLAIM_POLICY,
            "public_execution_accounting_scope": HYBRID_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
            "validation_replay_policy": HYBRID_VALIDATION_REPLAY_POLICY,
            "validation_replay_count": HYBRID_VALIDATION_REPLAY_COUNT,
            "canonical_serialization_policy": HYBRID_CANONICAL_SERIALIZATION_POLICY,
            "streaming_sequence_framing_policy": HYBRID_STREAMING_SEQUENCE_FRAMING_POLICY,
            "result_custody_policy": HYBRID_RESULT_CUSTODY_POLICY,
            "checksum_binding_policy": HYBRID_CHECKSUM_BINDING_POLICY,
            "schedule_checksum_algorithm": HYBRID_SCHEDULE_CHECKSUM_ALGORITHM,
            "schedule_checksum_domain": HYBRID_SCHEDULE_CHECKSUM_DOMAIN,
            "step_ledger_checksum_algorithm": HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM,
            "step_ledger_checksum_domain": HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN,
            "private_encounter_checksum_algorithm": HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM,
            "private_encounter_checksum_domain": HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_DOMAIN,
            "result_content_checksum_algorithm": HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            "result_content_checksum_domain": HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "claim_scope": HYBRID_CLAIM_SCOPE,
            "nonclaims": HYBRID_NONCLAIMS,
            "outer_step_adaptive": False,
            "encounter_substeps_adaptive": True,
            "symplectic": False,
            "formally_symplectic": False,
            "floating_point_symplectic": False,
            "globally_symplectic": False,
            "time_reversible": False,
            "formally_time_reversible": False,
            "floating_point_exactly_reversible": False,
            "exactly_time_reversible": False,
            "dense_output": False,
            "event_detection": False,
            "collision_detection": False,
            "collision_response": False,
            "regularized": False,
            "global_clearance_claimed": False,
            "global_order_claimed": False,
            "superiority_claimed": False,
            "evidence_class": HYBRID_EVIDENCE_CLASS,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if not _exact_builtin_equal(value, expected):
                raise ContractError(
                    f"{name} must equal the fixed hybrid value {expected!r}"
                )

    @property
    def direction(self) -> str:
        return self.wisdom_holman_spec.direction

    @property
    def completed_steps(self) -> int:
        return self.wisdom_holman_spec.completed_steps

    @property
    def body_order(self) -> tuple[str, ...]:
        return self.wisdom_holman_spec.jacobi_body_order

    @property
    def canonical_validation_segment_spec(self) -> AdaptiveEncounterSegmentSpec:
        signed_step = self.wisdom_holman_spec.fixed_step
        return _bind_encounter_segment(
            profile=self.encounter_control,
            body_order=self.body_order,
            initial_epoch=0.0,
            endpoint_epoch=signed_step,
            duration=signed_step,
        )


__all__ = [
    "HYBRID_ACCEPTED_START_REBIND_POLICY",
    "HYBRID_AGGREGATE_RESOURCE_NONCLAIM_POLICY",
    "HYBRID_AGGREGATE_RESOURCE_POLICY",
    "HYBRID_ALL_FAR_COMPATIBILITY_POLICY",
    "HYBRID_ALL_FAR_REASON",
    "HYBRID_BASE_TYPE_COMPOSITION_POLICY",
    "HYBRID_BODY_SWITCH_REASONS",
    "HYBRID_CANONICAL_PROFILE_VALIDATION_POLICY",
    "HYBRID_CANONICAL_SERIALIZATION_POLICY",
    "HYBRID_CHECKPOINT_POLICY",
    "HYBRID_CHECKSUM_BINDING_POLICY",
    "HYBRID_CLAIM_SCOPE",
    "HYBRID_COMPOSITION",
    "HYBRID_DYNAMIC_GUARD_PHASES",
    "HYBRID_DYNAMIC_SWITCH_REASONS",
    "HYBRID_DYNAMIC_SWITCH_REASON_POLICY",
    "HYBRID_ENCOUNTER_BINDING_POLICY",
    "HYBRID_ENCOUNTER_CONTROL_BINDER_FIELD_ROSTER",
    "HYBRID_ENCOUNTER_CONTROL_SCHEMA_POLICY",
    "HYBRID_ENCOUNTER_INTERVAL_BINDING_FIELDS",
    "HYBRID_EVIDENCE_CLASS",
    "HYBRID_EXECUTION_ACCOUNTING_POLICY",
    "HYBRID_FAR_PROBE_DECISION_FIELD_ROSTER",
    "HYBRID_FAR_PROBE_DECISION_INTERFACE_POLICY",
    "HYBRID_FAR_PROBE_FATAL_REASONS",
    "HYBRID_FAR_PROBE_LEGAL_MAPPING",
    "HYBRID_FAR_PROBE_OUTCOMES",
    "HYBRID_FAR_PROBE_PHASES",
    "HYBRID_FAR_PROBE_PHASE_WORK_FLOORS",
    "HYBRID_FAR_PROBE_REASON_WORK_POLICY",
    "HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER",
    "HYBRID_FAR_PROBE_WORK_FIELD_ROSTER",
    "HYBRID_FAR_PROBE_WORK_MAXIMUM_FORMULAS",
    "HYBRID_FAR_PROBE_WORK_VALIDATION_POLICY",
    "HYBRID_FATAL_FAILURE_CLASSES",
    "HYBRID_FATAL_FAILURE_POLICY",
    "HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PER_LANE",
    "HYBRID_HARD_MAXIMUM_NEAR_MACROSTEPS_PUBLIC_TOTAL",
    "HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PER_LANE",
    "HYBRID_HARD_MAXIMUM_OUTER_RECORDS_PUBLIC_TOTAL",
    "HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PER_LANE",
    "HYBRID_HARD_MAXIMUM_RETAINED_NEAR_ACCEPTED_SUBSTEPS_PUBLIC_TOTAL",
    "HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PER_LANE",
    "HYBRID_HARD_MAXIMUM_RETAINED_NEAR_DIGEST_BYTES_PUBLIC_TOTAL",
    "HYBRID_KEPLER_PROBE_FAILURE_STATUS_BY_REASON",
    "HYBRID_KEPLER_PROBE_RECORD_FIELD_ROSTER",
    "HYBRID_KEPLER_PROBE_RECORD_POLICY",
    "HYBRID_KEPLER_PROBE_SOURCE",
    "HYBRID_KEPLER_PROBE_TERMINAL_STATUSES",
    "HYBRID_METHOD_CLASS",
    "HYBRID_NEAR_LEDGER_POLICY",
    "HYBRID_NEAR_REPLACEMENT_POLICY",
    "HYBRID_NESTED_REPLAY_ACCOUNTING_POLICY",
    "HYBRID_NONCLAIMS",
    "HYBRID_ORIGINAL_NODE_ADMISSIBILITY_POLICY",
    "HYBRID_OUTER_LATTICE_POLICY",
    "HYBRID_PAIR_SWITCH_REASONS",
    "HYBRID_PRIVATE_ENCOUNTER_ACCEPTED_STEP_POLICY",
    "HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_ALGORITHM",
    "HYBRID_PRIVATE_ENCOUNTER_CHECKSUM_DOMAIN",
    "HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER",
    "HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER",
    "HYBRID_PRIVATE_ENCOUNTER_DIGEST_POLICY",
    "HYBRID_PRIVATE_ENCOUNTER_LITERAL_KAT_POLICY",
    "HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER",
    "HYBRID_PRIVATE_ENCOUNTER_RECORD_FIELD_ROSTER",
    "HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_POLICY",
    "HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER",
    "HYBRID_PROBE_TRANSACTION_POLICY",
    "HYBRID_PROVISIONAL_WORK_ACCOUNTING_POLICY",
    "HYBRID_PUBLIC_EXECUTION_ACCOUNTING_SCOPE",
    "HYBRID_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "HYBRID_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "HYBRID_RESULT_CUSTODY_POLICY",
    "HYBRID_SCHEDULE_CHECKSUM_ALGORITHM",
    "HYBRID_SCHEDULE_CHECKSUM_DOMAIN",
    "HYBRID_STEP_LEDGER_CHECKSUM_ALGORITHM",
    "HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN",
    "HYBRID_STEP_LEDGER_POLICY",
    "HYBRID_STEP_MODES",
    "HYBRID_STREAMING_SEQUENCE_FRAMING_POLICY",
    "HYBRID_SWITCH_POLICY",
    "HYBRID_TRIGGER_RULES",
    "HYBRID_VALIDATION_REPLAY_COUNT",
    "HYBRID_VALIDATION_REPLAY_POLICY",
    "HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID",
    "HybridEncounterControlProfile",
    "HybridFarProbeDecision",
    "HybridFarProbeLegalMappingRow",
    "HybridFarProbePhaseFloorRow",
    "HybridFarProbeWork",
    "HybridFarProbeWorkFormulaRow",
    "HybridKeplerProbeFailureStatusRow",
    "HybridKeplerProbeRecord",
    "HybridPrivateEncounterDigestDomainRow",
    "HybridPrivateEncounterPreimageSchemaRow",
    "HybridPrivateEncounterRecord",
    "HybridPrivateEncounterSummarySourceRow",
    "HybridWisdomHolmanRKF78Spec",
]
