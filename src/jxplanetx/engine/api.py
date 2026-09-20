"""Small public API for JX's experimental force and trajectory runtime."""

from __future__ import annotations

from . import encounter as _encounter_runtime
from . import encounter_contracts as _encounter_contracts
from . import hybrid as _hybrid_runtime
from . import hybrid_contracts as _hybrid_contracts
from .backends import (
    BackendArrayError,
    BackendError,
    BackendUnavailableError,
)
from .contracts import (
    BackendSpec,
    CannonballSRP,
    ForcePlan,
    NewtonianPointMass,
    RestrictedStaticCentral1PN,
    StateSnapshot,
)
from .evaluator import (
    DuplicateForceError,
    EvaluationError,
    ForceContribution,
    ForceEvaluationResult,
    ForceLedgerEntry,
    UnsupportedForceError,
    evaluate_force_plan,
)
from .forces import (
    ForceCollisionError,
    ForceContractError,
    ForceDomainError,
    ForceSingularityError,
)
from .rkf78 import (
    RKF78_ACCEPTED_ORDER,
    RKF78_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_ORDER,
    RKF78_METHOD_ID,
    RKF78_STAGE_COUNT,
    RKF78_TABLEAU_ID,
    RKF78_TABLEAU_SOURCE,
)
from .scenario import (
    DYNAMICS_SCENARIO_SCOPE,
    MATCHED_SCENARIO_SCOPE,
    SCENARIO_COMPARISON_SEMANTICS,
    SCENARIO_EVIDENCE_CLASS,
    SCENARIO_EXECUTION_ORDER,
    DynamicsScenario,
    DynamicsScenarioResult,
    MatchedScenarioComparison,
    MatchedScenarioResult,
    ScenarioCheckpointDelta,
    ScenarioContractError,
    run_dynamics_scenario,
    run_matched_scenario_comparison,
)
from .scenario_manifest import (
    PORTABLE_SCENARIO_ARRAY_ORDER,
    PORTABLE_SCENARIO_FLOAT_ENCODING,
    PORTABLE_SCENARIO_HASH_ALGORITHM,
    PORTABLE_SCENARIO_MANIFEST_MAX_BYTES,
    PORTABLE_SCENARIO_MANIFEST_SCHEMA,
    PORTABLE_SCENARIO_MANIFEST_SCOPE,
    ScenarioManifestError,
    dump_dynamics_scenario_manifest,
    dynamics_scenario_manifest_sha256,
    load_dynamics_scenario_manifest,
)
from .symplectic import (
    KDK_TRAJECTORY_SCOPE,
    KDKTrajectoryResult,
    integrate_kdk_trajectory,
)
from .symplectic_contracts import (
    FIXED_STEP_KDK_METHOD_ID,
    FixedStepKDKSpec,
    KDK_BACKEND_SCOPE,
    KDK_CHECKPOINT_POLICY,
    KDK_COMPOSITION,
    KDK_ENCOUNTER_GUARD,
    KDK_FORCE_EVALUATION_ACCOUNTING,
    KDK_FORCE_PLAN_SCOPE,
    KDK_PAIR_FREQUENCY_GUARD,
    KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
    KDK_SCHEDULE_CHECKSUM_ALGORITHM,
    KDK_SCHEDULE_CHECKSUM_DOMAIN,
    KDK_STEP_REPRESENTATION,
    KDK_TIME_SEMANTICS,
)
from .wisdom_holman import (
    JacobiCoordinateBinding,
    WH_TRAJECTORY_SCOPE,
    WisdomHolmanTrajectoryResult,
    integrate_wisdom_holman_trajectory,
)
from .wisdom_holman_contracts import (
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    FixedStepWisdomHolmanSpec,
    UniversalKeplerSolverSpec,
    WH_ALLOWED_AXES,
    WH_ALLOWED_TIME_SCALES,
    WH_BACKEND_SCOPE,
    WH_BINARY64_EPSILON_POLICY,
    WH_BARYCENTER_ROUNDOFF_FACTOR,
    WH_BARYCENTER_TOLERANCE_POLICY,
    WH_CANONICAL_MOMENTUM_CONVENTION,
    WH_CANONICAL_WEIGHT_SOURCE,
    WH_CHECKPOINT_POLICY,
    WH_COMPOSITION,
    WH_ENCOUNTER_GUARD,
    WH_ENCOUNTER_ROUNDOFF_FACTOR,
    WH_EVIDENCE_CLASS,
    WH_FORCE_EVALUATION_ACCOUNTING,
    WH_FORCE_PLAN_SCOPE,
    WH_GUARD_CADENCE_POLICY,
    WH_HAMILTONIAN_SPLIT_POLICY,
    WH_HIERARCHY_GUARD,
    WH_INTERACTION_FORCE_POLICY,
    WH_INTERACTION_KICK_COEFFICIENTS,
    WH_INTERACTION_RATIO_POLICY,
    WH_JACOBI_BINDING_CHECKSUM_ALGORITHM,
    WH_JACOBI_BINDING_CHECKSUM_DOMAIN,
    WH_JACOBI_COORDINATE_SYSTEM,
    WH_JACOBI_MATRIX_POLICY,
    WH_JACOBI_SYMBOL_POLICY,
    WH_JACOBI_TRANSFORM_POLICY,
    WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR,
    WH_KEPLER_BRACKET_ULP_FACTOR,
    WH_KEPLER_CONIC_SCOPE,
    WH_KEPLER_CONVERGENCE_POLICY,
    WH_KEPLER_DRIFT_COEFFICIENTS,
    WH_KEPLER_ENERGY_ROUNDOFF_FACTOR,
    WH_KEPLER_G_FUNCTION_POLICY,
    WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR,
    WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS,
    WH_KEPLER_MAXIMUM_ITERATIONS,
    WH_KEPLER_POSTCONDITION_POLICY,
    WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR,
    WH_KEPLER_RESIDUAL_ULP_FACTOR,
    WH_KEPLER_ROOT_POLICY,
    WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT,
    WH_KEPLER_SERIES_TERM_COUNT,
    WH_KEPLER_SOLVER_ID,
    WH_KEPLER_SYMBOL_POLICY,
    WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO,
    WH_MAXIMUM_JACOBI_ECCENTRICITY,
    WH_MAXIMUM_ORBIT_STEP_FRACTION,
    WH_MAXIMUM_PERIAPSE_STEP_FRACTION,
    WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO,
    WH_METHOD_CLASS,
    WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE,
    WH_MUTUAL_HILL_POLICY,
    WH_OSCULATING_JACOBI_POLICY,
    WH_PHYSICAL_MASS_POLICY,
    WH_PRINCIPAL_ORDER,
    WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
    WH_REQUIRED_FRAME,
    WH_REQUIRED_ORIGIN,
    WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
    WH_SCHEDULE_CHECKSUM_ALGORITHM,
    WH_SCHEDULE_CHECKSUM_DOMAIN,
    WH_STEP_ENVELOPE_POLICY,
    WH_STEP_REPRESENTATION,
    WH_TIME_SEMANTICS,
    WH_VALIDATION_REPLAY_COUNT,
    WH_VALIDATION_REPLAY_POLICY,
)
from .trajectory import (
    TRAJECTORY_SCOPE,
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryError,
    TrajectoryResult,
    TrajectoryStepLimitError,
    integrate_trajectory,
    validate_trajectory_result_integrity,
)
from .trajectory_contracts import (
    AdaptiveRKF78Spec,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    RKF78_CHECKPOINT_PROPOSAL_POLICY,
    RKF78_TIME_STEP_REPRESENTATION,
)


def evaluate(
    snapshot: StateSnapshot,
    plan: ForcePlan,
) -> ForceEvaluationResult:
    """Return acceleration contributions without integrating a trajectory."""

    return evaluate_force_plan(snapshot, plan)


def evaluate_forces(
    snapshot: StateSnapshot,
    plan: ForcePlan,
) -> ForceEvaluationResult:
    """Descriptive alias for :func:`evaluate`."""

    return evaluate_force_plan(snapshot, plan)


__all__ = [
    "AdaptiveRKF78Spec",
    "BackendArrayError",
    "BackendError",
    "BackendSpec",
    "BackendUnavailableError",
    "CannonballSRP",
    "DuplicateForceError",
    "DYNAMICS_SCENARIO_SCOPE",
    "DynamicsScenario",
    "DynamicsScenarioResult",
    "EvaluationError",
    "FIXED_STEP_KDK_METHOD_ID",
    "FIXED_STEP_WISDOM_HOLMAN_METHOD_ID",
    "FixedStepKDKSpec",
    "FixedStepWisdomHolmanSpec",
    "ForceCollisionError",
    "ForceContractError",
    "ForceContribution",
    "ForceDomainError",
    "ForceEvaluationResult",
    "ForceLedgerEntry",
    "ForcePlan",
    "ForceSingularityError",
    "NewtonianPointMass",
    "KDK_BACKEND_SCOPE",
    "KDK_CHECKPOINT_POLICY",
    "KDK_COMPOSITION",
    "KDK_ENCOUNTER_GUARD",
    "KDK_FORCE_EVALUATION_ACCOUNTING",
    "KDK_FORCE_PLAN_SCOPE",
    "KDK_PAIR_FREQUENCY_GUARD",
    "KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "KDK_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "KDK_SCHEDULE_CHECKSUM_ALGORITHM",
    "KDK_SCHEDULE_CHECKSUM_DOMAIN",
    "KDK_STEP_REPRESENTATION",
    "KDK_TIME_SEMANTICS",
    "KDK_TRAJECTORY_SCOPE",
    "KDKTrajectoryResult",
    "JacobiCoordinateBinding",
    "MATCHED_SCENARIO_SCOPE",
    "MatchedScenarioComparison",
    "MatchedScenarioResult",
    "PORTABLE_SCENARIO_ARRAY_ORDER",
    "PORTABLE_SCENARIO_FLOAT_ENCODING",
    "PORTABLE_SCENARIO_HASH_ALGORITHM",
    "PORTABLE_SCENARIO_MANIFEST_MAX_BYTES",
    "PORTABLE_SCENARIO_MANIFEST_SCHEMA",
    "PORTABLE_SCENARIO_MANIFEST_SCOPE",
    "RKF78_ACCEPTED_ORDER",
    "RKF78_ACCEPTED_STATE_ACCUMULATION",
    "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM",
    "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN",
    "RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE",
    "RKF78_CHECKPOINT_PROPOSAL_POLICY",
    "RKF78_DEFECT_ORIENTATION",
    "RKF78_EMBEDDED_ORDER",
    "RKF78_METHOD_ID",
    "RKF78_STAGE_COUNT",
    "RKF78_TABLEAU_ID",
    "RKF78_TABLEAU_SOURCE",
    "RKF78_TIME_STEP_REPRESENTATION",
    "RestrictedStaticCentral1PN",
    "SCENARIO_COMPARISON_SEMANTICS",
    "SCENARIO_EVIDENCE_CLASS",
    "SCENARIO_EXECUTION_ORDER",
    "ScenarioCheckpointDelta",
    "ScenarioContractError",
    "ScenarioManifestError",
    "StateSnapshot",
    "TRAJECTORY_SCOPE",
    "TrajectoryCheckpoint",
    "TrajectoryContractError",
    "TrajectoryDomainError",
    "TrajectoryError",
    "TrajectoryResult",
    "TrajectoryStepLimitError",
    "UniversalKeplerSolverSpec",
    "UnsupportedForceError",
    "WH_ALLOWED_AXES",
    "WH_ALLOWED_TIME_SCALES",
    "WH_BACKEND_SCOPE",
    "WH_BINARY64_EPSILON_POLICY",
    "WH_BARYCENTER_ROUNDOFF_FACTOR",
    "WH_BARYCENTER_TOLERANCE_POLICY",
    "WH_CANONICAL_MOMENTUM_CONVENTION",
    "WH_CANONICAL_WEIGHT_SOURCE",
    "WH_CHECKPOINT_POLICY",
    "WH_COMPOSITION",
    "WH_ENCOUNTER_GUARD",
    "WH_ENCOUNTER_ROUNDOFF_FACTOR",
    "WH_EVIDENCE_CLASS",
    "WH_FORCE_EVALUATION_ACCOUNTING",
    "WH_FORCE_PLAN_SCOPE",
    "WH_GUARD_CADENCE_POLICY",
    "WH_HAMILTONIAN_SPLIT_POLICY",
    "WH_HIERARCHY_GUARD",
    "WH_INTERACTION_FORCE_POLICY",
    "WH_INTERACTION_KICK_COEFFICIENTS",
    "WH_INTERACTION_RATIO_POLICY",
    "WH_JACOBI_BINDING_CHECKSUM_ALGORITHM",
    "WH_JACOBI_BINDING_CHECKSUM_DOMAIN",
    "WH_JACOBI_COORDINATE_SYSTEM",
    "WH_JACOBI_MATRIX_POLICY",
    "WH_JACOBI_SYMBOL_POLICY",
    "WH_JACOBI_TRANSFORM_POLICY",
    "WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR",
    "WH_KEPLER_BRACKET_ULP_FACTOR",
    "WH_KEPLER_CONIC_SCOPE",
    "WH_KEPLER_CONVERGENCE_POLICY",
    "WH_KEPLER_DRIFT_COEFFICIENTS",
    "WH_KEPLER_ENERGY_ROUNDOFF_FACTOR",
    "WH_KEPLER_G_FUNCTION_POLICY",
    "WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR",
    "WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS",
    "WH_KEPLER_MAXIMUM_ITERATIONS",
    "WH_KEPLER_POSTCONDITION_POLICY",
    "WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR",
    "WH_KEPLER_RESIDUAL_ULP_FACTOR",
    "WH_KEPLER_ROOT_POLICY",
    "WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT",
    "WH_KEPLER_SERIES_TERM_COUNT",
    "WH_KEPLER_SOLVER_ID",
    "WH_KEPLER_SYMBOL_POLICY",
    "WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO",
    "WH_MAXIMUM_JACOBI_ECCENTRICITY",
    "WH_MAXIMUM_ORBIT_STEP_FRACTION",
    "WH_MAXIMUM_PERIAPSE_STEP_FRACTION",
    "WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO",
    "WH_METHOD_CLASS",
    "WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE",
    "WH_MUTUAL_HILL_POLICY",
    "WH_OSCULATING_JACOBI_POLICY",
    "WH_PHYSICAL_MASS_POLICY",
    "WH_PRINCIPAL_ORDER",
    "WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE",
    "WH_REQUIRED_FRAME",
    "WH_REQUIRED_ORIGIN",
    "WH_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "WH_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "WH_SCHEDULE_CHECKSUM_ALGORITHM",
    "WH_SCHEDULE_CHECKSUM_DOMAIN",
    "WH_STEP_ENVELOPE_POLICY",
    "WH_STEP_REPRESENTATION",
    "WH_TIME_SEMANTICS",
    "WH_TRAJECTORY_SCOPE",
    "WH_VALIDATION_REPLAY_COUNT",
    "WH_VALIDATION_REPLAY_POLICY",
    "WisdomHolmanTrajectoryResult",
    "evaluate",
    "evaluate_forces",
    "dump_dynamics_scenario_manifest",
    "dynamics_scenario_manifest_sha256",
    "integrate_trajectory",
    "validate_trajectory_result_integrity",
    "integrate_kdk_trajectory",
    "integrate_wisdom_holman_trajectory",
    "load_dynamics_scenario_manifest",
    "run_dynamics_scenario",
    "run_matched_scenario_comparison",
]


_ENCOUNTER_PUBLIC_ROSTER = tuple(
    dict.fromkeys(
        (*_encounter_contracts.__all__, *_encounter_runtime.__all__)
    )
)
for _public_name in _ENCOUNTER_PUBLIC_ROSTER:
    if _public_name in _encounter_contracts.__all__:
        globals()[_public_name] = getattr(_encounter_contracts, _public_name)
    else:
        globals()[_public_name] = getattr(_encounter_runtime, _public_name)
__all__.extend(_ENCOUNTER_PUBLIC_ROSTER)
del _public_name


_HYBRID_PUBLIC_ROSTER = tuple(
    dict.fromkeys(
        (*_hybrid_contracts.__all__, *_hybrid_runtime.__all__)
    )
)
for _public_name in _HYBRID_PUBLIC_ROSTER:
    if _public_name in _hybrid_contracts.__all__:
        globals()[_public_name] = getattr(_hybrid_contracts, _public_name)
    else:
        globals()[_public_name] = getattr(_hybrid_runtime, _public_name)
__all__.extend(_HYBRID_PUBLIC_ROSTER)
del _public_name
