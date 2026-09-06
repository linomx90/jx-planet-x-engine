"""Fail-closed contracts for the ordered-Jacobi Wisdom--Holman map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .contracts import ContractError


FIXED_STEP_WISDOM_HOLMAN_METHOD_ID = (
    "integrator.symplectic.wisdom_holman_jacobi_kdk_2"
)
WH_METHOD_CLASS = "EXPLICIT_SYMMETRIC_JACOBI_KEPLER_INTERACTION_SPLITTING"
WH_PRINCIPAL_ORDER = 2
WH_INTERACTION_KICK_COEFFICIENTS = (0.5, 0.5)
WH_KEPLER_DRIFT_COEFFICIENTS = (1.0,)
WH_COMPOSITION = (
    "INTERACTION_KICK_HALF_KEPLER_AND_CENTER_OF_MASS_DRIFT_FULL_"
    "INTERACTION_KICK_HALF"
)
WH_HAMILTONIAN_SPLIT_POLICY = (
    "H_A=dot(P_0,P_0)/(2*K_total)+sum_{i>=1}(dot(P_i,P_i)/"
    "(2*lambda_i)-lambda_i*K_i/norm(Q_i)); H_B=-sum_{a<b}(mu_a*mu_b/"
    "norm(x_a-x_b))+sum_{i>=1}(lambda_i*K_i/norm(Q_i)); "
    "compose B(h/2) then A(h) then B(h/2)"
)
WH_STEP_REPRESENTATION = "CONSTANT_SIGNED_BINARY64_MAP_STEP"
WH_CHECKPOINT_POLICY = "INTEGER_STEP_LATTICE_NO_CLIPPING_NO_INTERPOLATION"
WH_FORCE_PLAN_SCOPE = "MUTUAL_ALL_BODY_NEWTONIAN_POINT_MASS_ONLY"
WH_BACKEND_SCOPE = "NUMPY_CPU_ONLY"
WH_TIME_SEMANTICS = "CONTINUOUS_COORDINATE_TIME"
WH_ALLOWED_TIME_SCALES = ("TDB", "SYNTHETIC")
WH_REQUIRED_FRAME = "BARYCENTRIC_INERTIAL"
WH_REQUIRED_ORIGIN = "BARYCENTER"
WH_ALLOWED_AXES = ("ICRS_ALIGNED", "CARTESIAN_RIGHT_HANDED")
WH_JACOBI_COORDINATE_SYSTEM = "ORDERED_JACOBI_BARYCENTRIC_V1"
WH_CANONICAL_WEIGHT_SOURCE = "STATE_GRAVITATIONAL_PARAMETERS"
WH_CANONICAL_MOMENTUM_CONVENTION = (
    "CARTESIAN_CANONICAL_MOMENTUM_I_EQUALS_GM_I_TIMES_VELOCITY_I"
)
WH_PHYSICAL_MASS_POLICY = "RETAINED_BUT_UNUSED_BY_GM_SCALED_MAP"
WH_JACOBI_SYMBOL_POLICY = (
    "mu_i=GM_i; K_i=sum_{j=0}^i(mu_j); K_previous_i=K_{i-1}; "
    "lambda_i=mu_i*K_previous_i/K_i"
)
WH_JACOBI_TRANSFORM_POLICY = (
    "R_i=sum_{j=0}^i(mu_j*x_j)/K_i; "
    "V_i=sum_{j=0}^i(mu_j*v_j)/K_i; Q_0=R_{N-1}; "
    "P_0=K_{N-1}*V_{N-1}=sum_j(mu_j*v_j); "
    "Q_i=x_i-R_{i-1}; P_i=lambda_i*(v_i-V_{i-1}) for i>=1; "
    "inverse starts R_{N-1}=Q_0,V_{N-1}=P_0/K_{N-1} and for "
    "i=N-1..1 sets R_{i-1}=R_i-(mu_i/K_i)*Q_i and "
    "V_{i-1}=V_i-P_i/K_{i-1}; x_0=R_0,v_0=V_0; "
    "x_i=R_{i-1}+Q_i,v_i=V_{i-1}+P_i/lambda_i"
)
WH_JACOBI_MATRIX_POLICY = (
    "B[0,j]=mu_j/K_{N-1}; B[i,j]=-mu_j/K_{i-1} for j<i; "
    "B[i,i]=1; B[i,j]=0 for j>i; A[j,0]=1; "
    "A[j,i]=-mu_i/K_i for j<i; A[i,i]=K_{i-1}/K_i; "
    "A[j,i]=0 for j>i; Q=B*x and x=A*Q"
)
WH_FORCE_EVALUATION_ACCOUNTING = (
    "PRIMARY_MAP_ONE_INITIAL_PLUS_ONE_PER_COMPLETED_MAP_STEP"
)
WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE = (
    "PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY"
)
WH_VALIDATION_REPLAY_POLICY = (
    "ONE_FULL_DETERMINISTIC_SEMANTIC_REPLAY_DURING_RESULT_VALIDATION"
)
WH_VALIDATION_REPLAY_COUNT = 1
WH_ENCOUNTER_GUARD = (
    "for each physical pair a,b set d0=norm(x_b_start-x_a_start),"
    "d1=norm(x_b_end-x_a_end),v_peri_i=sqrt(K_i*(1+e_i)/"
    "(a_i*(1-e_i))),"
    "D_ab=abs(h)*sum_{i>=1}(abs(A[a,i]-A[b,i])*v_peri_i),and "
    "lower_ab=max(d0,d1)-D_ab; floor_ab=max(radii_a+radii_b,caller encounter "
    "floor,secondary-pair 3*R_Hill_ab); margin_ab=256*N*eps*"
    "max(d0,d1,D_ab,floor_ab); require d0-margin_ab>floor_ab,"
    "d1-margin_ab>floor_ab,and lower_ab-margin_ab>floor_ab; the path-length "
    "bound is conservative in exact arithmetic and outward-margined in "
    "binary64 but is not directed interval arithmetic or a continuum proof"
)
WH_HIERARCHY_GUARD = (
    "INITIAL_AND_EACH_COMPLETED_NODE_OSCULATING_JACOBI_SEMIMAJOR_AXES_"
    "POSITIVE_AND_STRICTLY_INCREASING_IN_IMMUTABLE_BODY_ORDER"
)
WH_OSCULATING_JACOBI_POLICY = (
    "U_i=P_i/lambda_i; epsilon_i=0.5*dot(U_i,U_i)-K_i/norm(Q_i)<0; "
    "a_i=-K_i/(2*epsilon_i); eccentricity_vector_i=((dot(U_i,U_i)-"
    "K_i/norm(Q_i))*Q_i-dot(Q_i,U_i)*U_i)/K_i; "
    "e_i=norm(eccentricity_vector_i); q_peri_i=a_i*(1-e_i)"
)
WH_GUARD_CADENCE_POLICY = (
    "at the static initial preflight require sum_{i>=1}(mu_i)/mu_0<=0.01; "
    "at the initial state and every completed map node require positive "
    "strictly increasing a_i in immutable order,e_i<=0.9,q_peri_i>="
    "minimum_jacobi_periapse,interaction ratio<=0.1,step envelopes,and "
    "secondary mutual-Hill separation; apply conservative swept/contact "
    "screen to each full Kepler drift; after each first interaction half-kick "
    "and before its Kepler drift recheck finite b>0,a,e<=0.9,q_peri floor,"
    "step envelopes,hierarchy,and interaction-force ratio"
)
WH_INTERACTION_RATIO_POLICY = (
    "max_{i>=1}[(norm(F_interaction_i)/lambda_i)/"
    "(K_i/(norm(Q_i)^2))] <= 0.1"
)
WH_INTERACTION_FORCE_POLICY = (
    "f_cartesian_i=mu_i*a_full_i; F_full_Q=A_transpose*f_cartesian; "
    "F_Kepler_i=-lambda_i*K_i*Q_i/(norm(Q_i)^3) for i>=1; "
    "F_interaction_i=F_full_Q_i-F_Kepler_i; closed mutual gravity requires "
    "and sets F_interaction_0 exactly zero"
)
WH_MUTUAL_HILL_POLICY = (
    "secondary pairs only: R_Hill_ij=0.5*(a_i+a_j)*"
    "((mu_i+mu_j)/(3*mu_0))^(1/3); separation must exceed 3*R_Hill_ij"
)
WH_STEP_ENVELOPE_POLICY = (
    "P_i=2*pi*sqrt(a_i^3/K_i); "
    "tau_peri_i=2*pi*sqrt((a_i^3/K_i)*((1-e_i)^3/(1+e_i))); "
    "abs(h)/P_i<=1/20 and abs(h)/tau_peri_i<=1/16"
)
WH_BARYCENTER_TOLERANCE_POLICY = (
    "cap_x=256*N*eps*max_{i<j}(norm(x_i-x_j)); "
    "cap_v=256*N*eps*max_{i<j}(norm(v_i-v_j)); require norm(R_{N-1})"
    "<=min(caller_position_cap,cap_x) and norm(V_{N-1})"
    "<=min(caller_velocity_cap,cap_v); zero derived scale requires exact zero"
)
WH_EVIDENCE_CLASS = "MODEL_OUTPUT"

WH_KEPLER_SOLVER_ID = "ELLIPTIC_UNIVERSAL_G_FUNCTION_SAFEGUARDED_V1"
WH_KEPLER_CONIC_SCOPE = "BOUND_ELLIPTIC_B_POSITIVE_ONLY"
WH_KEPLER_SYMBOL_POLICY = (
    "r0=norm(q0); b=2*k/r0-dot(u0,u0)>0; eta=dot(q0,u0); "
    "E(q,u;k)=0.5*dot(u,u)-k/norm(q); L(q,u)=q cross u"
)
WH_KEPLER_ROOT_POLICY = (
    "seed s=h/r0 with R(0)=-h; order low,high from 0 and nonzero seed; "
    "double only the nonzero outer endpoint with finite representable progress "
    "at most 32 times until R(low)<=0<=R(high); current and backup midpoint "
    "are evaluated as low+0.5*(high-low); accept Newton current-R(current)/"
    "r(current) only when finite and strictly inside low,high, otherwise use "
    "the midpoint; replace low when residual<0 and high when residual>0; "
    "an exactly zero residual sets low=high=current and settles the bracket; "
    "every iteration must make representable bracket or candidate progress; "
    "at most 96 iterations; nonfinite values, nonpositive root derivative "
    "radius, missing bracket, unaccepted stagnation, cap exhaustion, or failed "
    "postcondition raises and returns no partial output"
)
WH_KEPLER_G_FUNCTION_POLICY = (
    "G_n(b,s)=sum_{m=0}^infinity((-b)^m*s^(n+2*m)/(n+2*m)!); "
    "if abs(sqrt(b)*s)<=0.5 set t_0=s^n/n!,rho=-b*s*s,"
    "t_{m+1}=t_m*rho/((n+2*m+1)*(n+2*m+2)) and evaluate exactly "
    "t_0 through t_63 by naive ascending-m left-to-right binary64 sum; "
    "otherwise theta=sqrt(b)*s then G0=cos(theta),"
    "G1=sin(theta)/sqrt(b),G2=(1-cos(theta))/b,G3=(s-G1)/b; "
    "fixed written operation order and no series early exit"
)
WH_KEPLER_CONVERGENCE_POLICY = (
    "R(s)=r0*s+eta*G2+(k-b*r0)*G3-h; "
    "R'(s)=r0+eta*G1+(k-b*r0)*G2=r(s)>0; "
    "tol=max(8*ulp(abs(h)),32*eps*(abs(r0*s)+abs(eta*G2)+"
    "abs((k-b*r0)*G3)+abs(h))); converge iff abs(R)<=tol and "
    "(bracket endpoints are adjacent or high-low<=8*max(ulp(abs(low)),"
    "ulp(abs(high)),ulp(abs(current))) or current exactly equals the previous "
    "or two-iterations-ago candidate)"
)
WH_BINARY64_EPSILON_POLICY = (
    "eps=numpy.finfo(numpy.float64).eps; "
    "ulp(x)=math.ulp(float(abs(x))) for finite binary64 x; ulp(0)=2^-1074"
)
WH_KEPLER_POSTCONDITION_POLICY = (
    "f=1-(k/r0)*G2; g=h-k*G3; fdot=-(k/(r0*r1))*G1; "
    "gdot=1-(k/r1)*G2; q1=f*q0+g*u0; "
    "u1=fdot*q0+gdot*u0; all relevant scalar and vector inputs and outputs "
    "must be finite; only k,b,r0,r1 and root derivative radius are required "
    "strictly positive; signed h,eta,f,g,fdot,gdot may be negative; "
    "abs(f*gdot-fdot*g-1)<=512*eps*max(1,abs(f*gdot)+abs(fdot*g)); "
    "abs(E(q1,u1;k)-E(q0,u0;k))<=1024*eps*max(k/r0,k/r1,"
    "0.5*dot(u0,u0),0.5*dot(u1,u1)); norm(L(q1,u1)-L(q0,u0))"
    "<=1024*eps*max(r0*norm(u0),r1*norm(u1))"
)
WH_KEPLER_MAXIMUM_ITERATIONS = 96
WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS = 32
WH_KEPLER_SERIES_TERM_COUNT = 64
WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT = 0.5
WH_KEPLER_RESIDUAL_ULP_FACTOR = 8
WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR = 32
WH_KEPLER_BRACKET_ULP_FACTOR = 8
WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR = 512
WH_KEPLER_ENERGY_ROUNDOFF_FACTOR = 1024
WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR = 1024

WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO = 0.01
WH_MAXIMUM_JACOBI_ECCENTRICITY = 0.9
WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO = 0.1
WH_MAXIMUM_ORBIT_STEP_FRACTION = 0.05
WH_MAXIMUM_PERIAPSE_STEP_FRACTION = 0.0625
WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE = 3.0
WH_BARYCENTER_ROUNDOFF_FACTOR = 256
WH_ENCOUNTER_ROUNDOFF_FACTOR = 256

WH_SCHEDULE_CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
WH_SCHEDULE_CHECKSUM_DOMAIN = "jxplanetx.wisdom-holman-schedule.content-integrity.v1"
WH_JACOBI_BINDING_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
WH_JACOBI_BINDING_CHECKSUM_DOMAIN = (
    "jxplanetx.wisdom-holman-jacobi-binding.content-integrity.v1"
)
WH_RESULT_CONTENT_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
WH_RESULT_CONTENT_CHECKSUM_DOMAIN = (
    "jxplanetx.wisdom-holman-result.content-integrity.v1"
)


def _positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ContractError(f"{label} must be a finite positive built-in float")
    return value


def _exact_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) < 2:
        raise ContractError(f"{label} must be an exact tuple containing at least two IDs")
    if any(
        type(item) is not str or not item or item.strip() != item for item in value
    ):
        raise ContractError(f"{label} entries must be nonempty trimmed built-in strings")
    if len(set(value)) != len(value):
        raise ContractError(f"{label} must not repeat a body ID")
    return value


@dataclass(frozen=True, eq=False)
class UniversalKeplerSolverSpec:
    """Fixed deterministic numerical contract for each elliptic Kepler flow."""

    solver_id: str = WH_KEPLER_SOLVER_ID
    conic_scope: str = WH_KEPLER_CONIC_SCOPE
    symbol_policy: str = WH_KEPLER_SYMBOL_POLICY
    root_policy: str = WH_KEPLER_ROOT_POLICY
    g_function_policy: str = WH_KEPLER_G_FUNCTION_POLICY
    convergence_policy: str = WH_KEPLER_CONVERGENCE_POLICY
    binary64_epsilon_policy: str = WH_BINARY64_EPSILON_POLICY
    postcondition_policy: str = WH_KEPLER_POSTCONDITION_POLICY
    maximum_iterations: int = WH_KEPLER_MAXIMUM_ITERATIONS
    maximum_bracket_expansions: int = WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS
    series_term_count: int = WH_KEPLER_SERIES_TERM_COUNT
    series_switch_abs_argument: float = WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT
    residual_ulp_factor: int = WH_KEPLER_RESIDUAL_ULP_FACTOR
    residual_roundoff_factor: int = WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR
    bracket_ulp_factor: int = WH_KEPLER_BRACKET_ULP_FACTOR
    lagrange_identity_roundoff_factor: int = (
        WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR
    )
    energy_roundoff_factor: int = WH_KEPLER_ENERGY_ROUNDOFF_FACTOR
    angular_momentum_roundoff_factor: int = (
        WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR
    )

    def __post_init__(self) -> None:
        exact = {
            "solver_id": WH_KEPLER_SOLVER_ID,
            "conic_scope": WH_KEPLER_CONIC_SCOPE,
            "symbol_policy": WH_KEPLER_SYMBOL_POLICY,
            "root_policy": WH_KEPLER_ROOT_POLICY,
            "g_function_policy": WH_KEPLER_G_FUNCTION_POLICY,
            "convergence_policy": WH_KEPLER_CONVERGENCE_POLICY,
            "binary64_epsilon_policy": WH_BINARY64_EPSILON_POLICY,
            "postcondition_policy": WH_KEPLER_POSTCONDITION_POLICY,
            "maximum_iterations": WH_KEPLER_MAXIMUM_ITERATIONS,
            "maximum_bracket_expansions": WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS,
            "series_term_count": WH_KEPLER_SERIES_TERM_COUNT,
            "series_switch_abs_argument": WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT,
            "residual_ulp_factor": WH_KEPLER_RESIDUAL_ULP_FACTOR,
            "residual_roundoff_factor": WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR,
            "bracket_ulp_factor": WH_KEPLER_BRACKET_ULP_FACTOR,
            "lagrange_identity_roundoff_factor": WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR,
            "energy_roundoff_factor": WH_KEPLER_ENERGY_ROUNDOFF_FACTOR,
            "angular_momentum_roundoff_factor": WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(
                    f"{name} must equal the fixed Kepler-solver value {expected!r}"
                )


@dataclass(frozen=True, eq=False)
class FixedStepWisdomHolmanSpec:
    """One ordered-Jacobi map request on an integer output lattice."""

    checkpoint_step_indices: tuple[int, ...]
    fixed_step: float
    maximum_steps: int
    jacobi_body_order: tuple[str, ...]
    minimum_encounter_pair_separation: float
    minimum_jacobi_periapse: float
    maximum_initial_barycenter_position_norm: float
    maximum_initial_barycenter_velocity_norm: float
    kepler_solver: UniversalKeplerSolverSpec = field(
        default_factory=UniversalKeplerSolverSpec
    )

    method_id: str = FIXED_STEP_WISDOM_HOLMAN_METHOD_ID
    method_class: str = WH_METHOD_CLASS
    principal_order: int = WH_PRINCIPAL_ORDER
    interaction_kick_coefficients: tuple[float, ...] = (
        WH_INTERACTION_KICK_COEFFICIENTS
    )
    kepler_drift_coefficients: tuple[float, ...] = WH_KEPLER_DRIFT_COEFFICIENTS
    composition: str = WH_COMPOSITION
    hamiltonian_split_policy: str = WH_HAMILTONIAN_SPLIT_POLICY
    step_representation: str = WH_STEP_REPRESENTATION
    checkpoint_policy: str = WH_CHECKPOINT_POLICY
    force_plan_scope: str = WH_FORCE_PLAN_SCOPE
    backend_scope: str = WH_BACKEND_SCOPE
    time_semantics: str = WH_TIME_SEMANTICS
    jacobi_coordinate_system: str = WH_JACOBI_COORDINATE_SYSTEM
    canonical_weight_source: str = WH_CANONICAL_WEIGHT_SOURCE
    canonical_momentum_convention: str = WH_CANONICAL_MOMENTUM_CONVENTION
    physical_mass_policy: str = WH_PHYSICAL_MASS_POLICY
    jacobi_symbol_policy: str = WH_JACOBI_SYMBOL_POLICY
    jacobi_transform_policy: str = WH_JACOBI_TRANSFORM_POLICY
    jacobi_matrix_policy: str = WH_JACOBI_MATRIX_POLICY
    encounter_guard: str = WH_ENCOUNTER_GUARD
    hierarchy_guard: str = WH_HIERARCHY_GUARD
    osculating_jacobi_policy: str = WH_OSCULATING_JACOBI_POLICY
    guard_cadence_policy: str = WH_GUARD_CADENCE_POLICY
    interaction_ratio_policy: str = WH_INTERACTION_RATIO_POLICY
    interaction_force_policy: str = WH_INTERACTION_FORCE_POLICY
    mutual_hill_policy: str = WH_MUTUAL_HILL_POLICY
    step_envelope_policy: str = WH_STEP_ENVELOPE_POLICY
    barycenter_tolerance_policy: str = WH_BARYCENTER_TOLERANCE_POLICY
    force_evaluation_accounting: str = WH_FORCE_EVALUATION_ACCOUNTING
    public_execution_accounting_scope: str = (
        WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE
    )
    validation_replay_policy: str = WH_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = WH_VALIDATION_REPLAY_COUNT
    maximum_total_secondary_to_primary_gm_ratio: float = (
        WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
    )
    maximum_jacobi_eccentricity: float = WH_MAXIMUM_JACOBI_ECCENTRICITY
    maximum_interaction_to_kepler_force_ratio: float = (
        WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO
    )
    maximum_orbit_step_fraction: float = WH_MAXIMUM_ORBIT_STEP_FRACTION
    maximum_periapse_step_fraction: float = WH_MAXIMUM_PERIAPSE_STEP_FRACTION
    minimum_mutual_hill_separation_multiple: float = (
        WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE
    )
    barycenter_roundoff_factor: int = WH_BARYCENTER_ROUNDOFF_FACTOR
    encounter_roundoff_factor: int = WH_ENCOUNTER_ROUNDOFF_FACTOR
    formal_exact_kepler_subflow_symplectic: bool = True
    floating_point_symplectic: bool = False
    formal_exact_kepler_subflow_time_reversible: bool = True
    floating_point_exactly_reversible: bool = False
    dense_output: bool = False
    adaptive: bool = False
    evidence_class: str = WH_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        indices = self.checkpoint_step_indices
        if type(indices) is not tuple or len(indices) < 2:
            raise ContractError(
                "checkpoint_step_indices must be an exact tuple containing zero "
                "and at least one later map index"
            )
        if any(type(index) is not int or index < 0 for index in indices):
            raise ContractError(
                "checkpoint_step_indices entries must be nonnegative built-in integers"
            )
        if indices[0] != 0 or any(
            right <= left for left, right in zip(indices, indices[1:])
        ):
            raise ContractError(
                "checkpoint_step_indices must begin at zero and be strictly increasing"
            )
        if type(self.fixed_step) is not float or not math.isfinite(self.fixed_step):
            raise ContractError("fixed_step must be a finite nonzero built-in float")
        if self.fixed_step == 0.0:
            raise ContractError("fixed_step must be nonzero")
        if type(self.maximum_steps) is not int or self.maximum_steps <= 0:
            raise ContractError("maximum_steps must be a positive built-in integer")
        if indices[-1] > self.maximum_steps:
            raise ContractError(
                "the final checkpoint step index cannot exceed maximum_steps"
            )
        _exact_string_tuple(self.jacobi_body_order, "jacobi_body_order")
        for name in (
            "minimum_encounter_pair_separation",
            "minimum_jacobi_periapse",
            "maximum_initial_barycenter_position_norm",
            "maximum_initial_barycenter_velocity_norm",
        ):
            _positive_float(getattr(self, name), name)
        if type(self.kepler_solver) is not UniversalKeplerSolverSpec:
            raise ContractError(
                "kepler_solver must be an exact UniversalKeplerSolverSpec"
            )
        self.kepler_solver.__post_init__()
        for name in (
            "interaction_kick_coefficients",
            "kepler_drift_coefficients",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not float for value in values
            ):
                raise ContractError(
                    f"{name} must be an exact tuple of built-in floats"
                )

        exact = {
            "method_id": FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
            "method_class": WH_METHOD_CLASS,
            "principal_order": WH_PRINCIPAL_ORDER,
            "interaction_kick_coefficients": WH_INTERACTION_KICK_COEFFICIENTS,
            "kepler_drift_coefficients": WH_KEPLER_DRIFT_COEFFICIENTS,
            "composition": WH_COMPOSITION,
            "hamiltonian_split_policy": WH_HAMILTONIAN_SPLIT_POLICY,
            "step_representation": WH_STEP_REPRESENTATION,
            "checkpoint_policy": WH_CHECKPOINT_POLICY,
            "force_plan_scope": WH_FORCE_PLAN_SCOPE,
            "backend_scope": WH_BACKEND_SCOPE,
            "time_semantics": WH_TIME_SEMANTICS,
            "jacobi_coordinate_system": WH_JACOBI_COORDINATE_SYSTEM,
            "canonical_weight_source": WH_CANONICAL_WEIGHT_SOURCE,
            "canonical_momentum_convention": WH_CANONICAL_MOMENTUM_CONVENTION,
            "physical_mass_policy": WH_PHYSICAL_MASS_POLICY,
            "jacobi_symbol_policy": WH_JACOBI_SYMBOL_POLICY,
            "jacobi_transform_policy": WH_JACOBI_TRANSFORM_POLICY,
            "jacobi_matrix_policy": WH_JACOBI_MATRIX_POLICY,
            "encounter_guard": WH_ENCOUNTER_GUARD,
            "hierarchy_guard": WH_HIERARCHY_GUARD,
            "osculating_jacobi_policy": WH_OSCULATING_JACOBI_POLICY,
            "guard_cadence_policy": WH_GUARD_CADENCE_POLICY,
            "interaction_ratio_policy": WH_INTERACTION_RATIO_POLICY,
            "interaction_force_policy": WH_INTERACTION_FORCE_POLICY,
            "mutual_hill_policy": WH_MUTUAL_HILL_POLICY,
            "step_envelope_policy": WH_STEP_ENVELOPE_POLICY,
            "barycenter_tolerance_policy": WH_BARYCENTER_TOLERANCE_POLICY,
            "force_evaluation_accounting": WH_FORCE_EVALUATION_ACCOUNTING,
            "public_execution_accounting_scope": WH_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
            "validation_replay_policy": WH_VALIDATION_REPLAY_POLICY,
            "validation_replay_count": WH_VALIDATION_REPLAY_COUNT,
            "maximum_total_secondary_to_primary_gm_ratio": WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO,
            "maximum_jacobi_eccentricity": WH_MAXIMUM_JACOBI_ECCENTRICITY,
            "maximum_interaction_to_kepler_force_ratio": WH_MAXIMUM_INTERACTION_TO_KEPLER_FORCE_RATIO,
            "maximum_orbit_step_fraction": WH_MAXIMUM_ORBIT_STEP_FRACTION,
            "maximum_periapse_step_fraction": WH_MAXIMUM_PERIAPSE_STEP_FRACTION,
            "minimum_mutual_hill_separation_multiple": WH_MINIMUM_MUTUAL_HILL_SEPARATION_MULTIPLE,
            "barycenter_roundoff_factor": WH_BARYCENTER_ROUNDOFF_FACTOR,
            "encounter_roundoff_factor": WH_ENCOUNTER_ROUNDOFF_FACTOR,
            "formal_exact_kepler_subflow_symplectic": True,
            "floating_point_symplectic": False,
            "formal_exact_kepler_subflow_time_reversible": True,
            "floating_point_exactly_reversible": False,
            "dense_output": False,
            "adaptive": False,
            "evidence_class": WH_EVIDENCE_CLASS,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(
                    f"{name} must equal the fixed Wisdom--Holman value {expected!r}"
                )

    @property
    def primary_body_id(self) -> str:
        return self.jacobi_body_order[0]

    @property
    def direction(self) -> str:
        return "FORWARD" if self.fixed_step > 0.0 else "BACKWARD"

    @property
    def completed_steps(self) -> int:
        return self.checkpoint_step_indices[-1]


__all__ = [
    "FIXED_STEP_WISDOM_HOLMAN_METHOD_ID",
    "FixedStepWisdomHolmanSpec",
    "UniversalKeplerSolverSpec",
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
    "WH_HIERARCHY_GUARD",
    "WH_HAMILTONIAN_SPLIT_POLICY",
    "WH_INTERACTION_RATIO_POLICY",
    "WH_INTERACTION_FORCE_POLICY",
    "WH_INTERACTION_KICK_COEFFICIENTS",
    "WH_JACOBI_BINDING_CHECKSUM_ALGORITHM",
    "WH_JACOBI_BINDING_CHECKSUM_DOMAIN",
    "WH_JACOBI_COORDINATE_SYSTEM",
    "WH_JACOBI_MATRIX_POLICY",
    "WH_JACOBI_SYMBOL_POLICY",
    "WH_JACOBI_TRANSFORM_POLICY",
    "WH_KEPLER_BRACKET_ULP_FACTOR",
    "WH_KEPLER_CONIC_SCOPE",
    "WH_KEPLER_CONVERGENCE_POLICY",
    "WH_KEPLER_DRIFT_COEFFICIENTS",
    "WH_KEPLER_G_FUNCTION_POLICY",
    "WH_KEPLER_ANGULAR_MOMENTUM_ROUNDOFF_FACTOR",
    "WH_KEPLER_ENERGY_ROUNDOFF_FACTOR",
    "WH_KEPLER_LAGRANGE_IDENTITY_ROUNDOFF_FACTOR",
    "WH_KEPLER_MAXIMUM_BRACKET_EXPANSIONS",
    "WH_KEPLER_MAXIMUM_ITERATIONS",
    "WH_KEPLER_RESIDUAL_ROUNDOFF_FACTOR",
    "WH_KEPLER_RESIDUAL_ULP_FACTOR",
    "WH_KEPLER_ROOT_POLICY",
    "WH_KEPLER_POSTCONDITION_POLICY",
    "WH_KEPLER_SERIES_TERM_COUNT",
    "WH_KEPLER_SERIES_SWITCH_ABS_ARGUMENT",
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
    "WH_STEP_REPRESENTATION",
    "WH_STEP_ENVELOPE_POLICY",
    "WH_TIME_SEMANTICS",
    "WH_VALIDATION_REPLAY_COUNT",
    "WH_VALIDATION_REPLAY_POLICY",
]
