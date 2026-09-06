"""Fail-closed contracts for one adaptive Newtonian encounter segment.

This module owns only immutable request and policy types.  It deliberately
does not expose a runtime integrator.  The first implementation is restricted
to an autonomous, all-active, unsoftened Newtonian IVP on NumPy CPU float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ContractError


ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID = (
    "integrator.adaptive.rkf78.encounter_segment_newtonian_v1"
)
ENCOUNTER_METHOD_CLASS = "EXPLICIT_EMBEDDED_RUNGE_KUTTA_LOCAL_ENCOUNTER_IVP"
ENCOUNTER_PRINCIPAL_ORDER = 8
ENCOUNTER_ACCEPTED_ORDER = 8
ENCOUNTER_EMBEDDED_ORDER = 7
ENCOUNTER_STAGE_COUNT = 13
ENCOUNTER_TABLEAU_ID = "NASA_TR_R_287_FEHLBERG_7_8_13_STAGE"
ENCOUNTER_SOURCE_REPORT = "NASA-TR-R-287"
ENCOUNTER_SOURCE_DOCUMENT_ID = "19680027281"
ENCOUNTER_SOURCE_TITLE = (
    "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta "
    "Formulas with Stepsize Control"
)
ENCOUNTER_SOURCE_AUTHOR = "Erwin Fehlberg"
ENCOUNTER_SOURCE_PUBLICATION_DATE = "1968-10-01"
ENCOUNTER_SOURCE_URL = "https://ntrs.nasa.gov/citations/19680027281"
ENCOUNTER_ACCEPTED_SOLUTION = "ORDER_8_HATTED"
ENCOUNTER_EMBEDDED_SOLUTION = "ORDER_7_ORDINARY"
ENCOUNTER_DEFECT_ORIENTATION = "ACCEPTED_ORDER_8_MINUS_EMBEDDED_ORDER_7"
ENCOUNTER_CONTROLLER_EXPONENT = 0.125
ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR = 0.5

ENCOUNTER_BACKEND_SCOPE = "EXACT_NUMPY_CPU_FLOAT64_ONLY"
ENCOUNTER_FORCE_PLAN_SCOPE = (
    "AUTONOMOUS_MUTUAL_ALL_ACTIVE_POSITIVE_GM_UNSOFTENED_NEWTONIAN_ONLY"
)
ENCOUNTER_REQUIRED_FRAME = "BARYCENTRIC_INERTIAL"
ENCOUNTER_REQUIRED_ORIGIN = "BARYCENTER"
ENCOUNTER_ALLOWED_AXES = ("ICRS_ALIGNED", "CARTESIAN_RIGHT_HANDED")
ENCOUNTER_ALLOWED_TIME_SCALES = ("TDB", "SYNTHETIC")
ENCOUNTER_DTYPE = "float64"
ENCOUNTER_INPUT_BINDING_POLICY = (
    "snapshot body_ids must exactly equal body_order,snapshot.epoch must be an "
    "exact built-in float bit-identical to initial_epoch,and metadata must name "
    "the required frame,origin,allowed axes,and allowed time scale; positions,"
    "velocities,GM,masses,and radii must be exact CPU numpy.ndarray objects "
    "with float64 dtype,required shapes,C-contiguous finite storage,and no "
    "subclass or device transfer; massive must be an exact CPU NumPy boolean "
    "array containing only true; every GM is strictly positive,every radius "
    "is nonnegative,and physical masses are retained but unused by the GM-"
    "scaled dynamics; the force plan is exactly one NewtonianPointMass with "
    "source_ids=target_ids=body_order on numpy.cpu and contains no other force; "
    "all parameter-metadata validity intervals must cover the closed public "
    "label interval between initial_epoch and endpoint_epoch"
)
ENCOUNTER_BODY_ORDER_POLICY = (
    "IMMUTABLE_STATE_BODY_ORDER_WITH_CANONICAL_PAIRS_I_LESS_THAN_J"
)
ENCOUNTER_PAIR_TABLE_POLICY = (
    "for N bodies the canonical table has exactly N*(N-1)/2 entries ordered "
    "(0,1),(0,2),...,(0,N-1),(1,2),...,(N-2,N-1); each supplied "
    "binary64 certification floor is positive, retained verbatim, and must "
    "be exact-dyadically greater than or equal to the exact sum of the two "
    "retained nonnegative binary64 radii; no scalar or hidden default floor"
)

ENCOUNTER_TIME_POLICY = (
    "retain initial_epoch,an externally supplied integrity-bound but "
    "unauthenticated provenance-only endpoint_epoch label,and one nonzero signed "
    "binary64 mathematical duration h; only finiteness and strict monotonicity "
    "from initial_epoch in sign(h) are internally checkable,so endpoint_epoch "
    "need not equal float(initial_epoch+h) and is not evidence of outer-lattice "
    "construction; advance state on an authoritative "
    "local offset lattice from exact zero to exact h and never recover any RK "
    "step or state duration from endpoint_epoch-initial_epoch; accepted signed "
    "binary64 substeps are exact-rationally accounted and their mathematical "
    "sum must equal h exactly"
)
ENCOUNTER_STAGE_EPOCH_POLICY = (
    "CONSTANT_INITIAL_PUBLIC_LABEL_FOR_ALL_AUTONOMOUS_NEWTONIAN_STAGE_"
    "EVALUATIONS; every derivative evaluation presents the bit-identical "
    "initial_epoch public label to force metadata and ignores the RK stage-time "
    "label; exact local offsets,accepted signed substeps,proposal signed steps,"
    "and the fixed RKF78 c-table identify mathematical stage times separately; "
    "the full closed public label interval between initial_epoch and "
    "endpoint_epoch remains subject to parameter-metadata preflight validation; "
    "no public label alters or supplies a retained RK substep or force"
)
ENCOUNTER_STEP_SCHEDULER_POLICY = (
    "proposals have sign(h); set exact positive target to min(exact remaining "
    "duration,exact proposed positive binary64 magnitude,exact maximum-step "
    "magnitude); convert target once by correctly-rounded nearest-ties-to-even "
    "binary64,then if that value is greater than target by exact-dyadic "
    "comparison apply math.nextafter(value,0.0) once; require the result finite "
    "and positive,which is the unique largest positive binary64 no greater than "
    "target; exact-subtract each accepted magnitude from remaining and repeat "
    "until exact zero; every chunk remains subject to the certificate,stage "
    "guards,and error test; a clipped chunk strictly greater than minimum may "
    "reject and retry/"
    "split normally,while a terminal chunk <minimum is acceptance-only and any "
    "rejection is fatal; fail closed if conversion,progress,exact subtraction,"
    "or endpoint completion violates a step or resource cap"
)
ENCOUNTER_ACCEPTED_STATE_ACCUMULATION = (
    "KAHAN_NUMPY_FLOAT64_COMPONENTWISE_POSITIONS_AND_VELOCITIES"
)
ENCOUNTER_CARRY_ENTRY_POLICY = (
    "initialize position_carry and velocity_carry as owned exact positive-zero "
    "NumPy float64 arrays at the start of every standalone encounter segment "
    "and every hybrid full-macrostep replacement; never import a carry from a "
    "discarded Wisdom-Holman candidate or a preceding outer segment"
)

ENCOUNTER_EXACT_RATIONAL_REPRESENTATION = (
    "two disjoint custom exact lanes: general values used by the simultaneous "
    "first-crossing certificate are canonical built-in-integer pairs (p,q) "
    "with q>0,gcd(abs(p),q)=1,and zero (0,1); binary64 duration scheduling and "
    "all radius-sum,start,stage,and candidate squared-distance guards use "
    "canonical exact dyadics (m,e) denoting m*2^e,with zero (0,0) and every "
    "nonzero m odd; finite binary64 enters either lane only through a "
    "prospectively authorized float.as_integer_ratio conversion; Fraction,"
    "Decimal,third-party exact code,and floating comparison fallback are forbidden"
)
ENCOUNTER_EXACT_RATIONAL_OPERATION_POLICY = (
    "general p/q primitives are FROM_BINARY64,ADD,SUBTRACT,MULTIPLY,DIVIDE,"
    "SQUARE,NEGATE,and COMPARE with sign normalization and Euclidean reduction "
    "after construction; dyadic primitives are FROM_BINARY64,ADD,SUBTRACT,"
    "MULTIPLY,SQUARE,NEGATE,and COMPARE,align ADD/SUBTRACT to the smaller "
    "exponent,and canonicalize by shifting every factor of two from m into e; "
    "for each dyadic distance guard convert the 3N coordinates once,form each "
    "component difference,and accumulate (dx^2+dy^2)+dz^2 left-to-right before "
    "one strict comparison with the cached dyadic rho^2; fixed vector-component "
    "and canonical-pair order is mandatory in both lanes"
)
ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID = (
    "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1"
)
ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE = (
    ("GENERAL_RATIONAL", "INTEGER_WIDTH_OBSERVATION", 2),
    ("GENERAL_RATIONAL", "RATIONAL_EXPONENT_DIAGNOSTIC", 1),
    ("GENERAL_RATIONAL", "INTEGER_COMPARE", 1),
    ("GENERAL_RATIONAL", "INTEGER_ABSOLUTE", 1),
    ("GENERAL_RATIONAL", "INTEGER_NEGATE", 1),
    ("GENERAL_RATIONAL", "INTEGER_SHIFT_LEFT", 1),
    ("GENERAL_RATIONAL", "INTEGER_ADD", 1),
    ("GENERAL_RATIONAL", "INTEGER_SUBTRACT", 1),
    ("GENERAL_RATIONAL", "INTEGER_MULTIPLY", 1),
    ("GENERAL_RATIONAL", "INTEGER_FLOOR_DIVIDE", 1),
    ("GENERAL_RATIONAL", "BINARY64_ENVELOPE_BASE", 8),
    ("GENERAL_RATIONAL", "BINARY64_ENVELOPE_NONZERO", 4),
    ("GENERAL_RATIONAL", "RATIONAL_CONSTRUCTION", 1),
    ("GENERAL_RATIONAL", "FROM_BINARY64_RATIO_EXTRACTION", 1),
    ("GENERAL_RATIONAL", "SIGNED_STEP_ABSOLUTE", 1),
    ("DYADIC", "INTEGER_WIDTH_OBSERVATION", 2),
    ("DYADIC", "BINARY64_ENVELOPE_BASE", 8),
    ("DYADIC", "BINARY64_ENVELOPE_NONZERO", 4),
    ("DYADIC", "DYADIC_CONSTRUCTION", 1),
    ("DYADIC", "INPUT_EXPONENT_CHECK", 1),
    ("DYADIC", "NONZERO_ODDNESS_BRANCH", 1),
    ("DYADIC", "EVEN_LOWBIT_PATH", 3),
    ("DYADIC", "EVEN_CANONICALIZE_PATH", 3),
    ("DYADIC", "CANONICAL_EXPONENT_DIAGNOSTIC", 4),
    ("DYADIC", "FROM_BINARY64_RATIO_EXTRACTION", 1),
    ("DYADIC", "FROM_BINARY64_POWER_OF_TWO_CHECK", 2),
    ("DYADIC", "ADD_OR_SUBTRACT_ALIGN", 3),
    ("DYADIC", "ADD_OR_SUBTRACT_SHIFT_PAIR", 2),
    ("DYADIC", "ADD_OR_SUBTRACT_COMBINE", 1),
    ("DYADIC", "MULTIPLY_MANTISSA_AND_EXPONENT", 2),
    ("DYADIC", "NEGATE", 1),
    ("DYADIC", "COMPARE_ALIGN_AND_SHIFT_PAIR", 5),
    ("DYADIC", "COMPARE_RESULT", 1),
    ("DYADIC", "TO_BINARY64_RATIO_BUILD", 1),
    ("DYADIC", "TO_BINARY64_ROUND", 1),
    ("DYADIC", "DURATION_ABSOLUTE", 1),
    ("DYADIC", "SCHEDULER_NEXTAFTER_DOWN", 1),
    ("DYADIC", "SCHEDULER_SUCCESSOR_PROBE", 1),
    ("GCD", "EUCLIDEAN_ITERATION", 1),
)
ENCOUNTER_EXACT_RATIONAL_RESOURCE_POLICY = (
    "report general-rational and dyadic deterministic abstract exact-work units "
    "separately and their exact sum; the fixed named path weights are exactly "
    "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1 and are retained verbatim in the "
    "resource specification; a selected named path charges its table weight once "
    "at its explicit runtime charge site,including conditional nonzero,even-"
    "normalization,nextafter,and conversion paths,while unselected conditional "
    "paths charge zero; these are qualification weights for this algorithm,not "
    "counts of Python,C,CPU,big-integer,or machine instructions or primitives "
    "and not estimates or guarantees of elapsed time,wall time,or memory; each "
    "charge site prospectively authorizes its complete table weight against the "
    "initialization-plus-segment ledgers during preflight or proposal-plus-"
    "segment ledgers during a proposal before the selected path produces its "
    "result; explicit integer-width and exponent checks additionally authorize "
    "potentially enlarged shift,add,subtract,multiply,and conversion results "
    "before those result integers are formed; reduction uses "
    "a custom interruptible Euclidean loop only in the general p/q lane and "
    "charges exactly one separately reported GCD.EUCLIDEAN_ITERATION unit per "
    "selected loop iteration,without claiming a count of underlying modulus or "
    "integer primitives; this unit is charged to separate "
    "initialization/proposal and segment gcd ledgers without double-charging "
    "exact-work units; "
    "it also obeys the per-reduction iteration cap; "
    "body,pair,and identifier caps are checked before numeric conversion or set "
    "allocation,and full initialization and per-proposal operation reservations "
    "are required from the remaining segment budget; every normalized rational "
    "exponent and integer intermediate is capped; any failed prospective check is fatal "
    "and never falls back to opaque math.gcd or floating arithmetic; the dyadic "
    "lane performs no GCD"
)
ENCOUNTER_RATIONAL_EXPONENT_POLICY = (
    "for nonzero p/q define e=floor(log2(abs(p/q))) by integer bit lengths "
    "plus one exact shifted-integer comparison; define e=0 for zero; require "
    "abs(e) no greater than maximum_rational_exponent_magnitude; for every "
    "retained nonzero canonical dyadic (m,e),the diagnostic observes both abs(e) "
    "and abs(e+bit_length(abs(m))-1),the latter being the value floor-log2,and "
    "zero contributes zero; maximum_rational_exponent_magnitude in each usage "
    "record is the maximum of those general-rational and canonical-dyadic "
    "observations only and deliberately excludes a larger pre-normalization "
    "constructor exponent,alignment exponent or shift count,and any intermediate "
    "that canonicalizes or cancels to zero,although all such paths remain "
    "prospectively subject to the fixed exponent and integer-bit caps"
)
ENCOUNTER_RESOURCE_LANE_POLICY = (
    "primary execution and mandatory semantic replay each receive separate "
    "identical proposal,acceptance,rejection,force,body,pair,integer,exponent,"
    "gcd,operation,transcript,and witness caps,including a separate initialization "
    "ledger in each lane; report initialization,proposal,lane-total,primary,"
    "replay,and public-total usage separately; the public-total force,exact-"
    "operation,GCD,and transcript ceilings are exactly twice the corresponding "
    "one-lane ceilings; neither lane may borrow unused budget from the other"
)
ENCOUNTER_RESOURCE_NONCLAIM_POLICY = (
    "fixed logical caps make arithmetic and custody finite but are not a "
    "wall-clock or resident-memory denial-of-service guarantee; an external "
    "service must additionally enforce process isolation,time,and memory limits"
)

ENCOUNTER_EFFECTIVE_FLOOR_POLICY = (
    "rho_ij is exactly the retained canonical pair_certification_floors entry; "
    "runtime preflight converts rho_ij and both retained radii to canonical exact "
    "dyadics and requires rho_ij>=radius_i+radius_j before any force call"
)
ENCOUNTER_INITIAL_DOMAIN_POLICY = (
    "before any force call require the accepted initial numerical node finite "
    "and,for every canonical pair,the exact-dyadic squared separation "
    "strictly greater than rho_ij^2; failure is fatal and returns no result"
)
ENCOUNTER_INITIALIZATION_RESOURCE_POLICY = (
    "before proposal one,charge every exact GM,radius,floor,and initial-position "
    "conversion,every rho_ij>=radius_i+radius_j proof,"
    "every initial squared-separation>rho_ij^2 proof,and construction of all "
    "cached A_i=sum_{k!=i}(GM_k/rho_ik^2) values to a separately capped "
    "initialization dyadic-work,general-rational-work,gcd,transcript ledger and also to the enclosing "
    "lane's whole-segment ledgers; retain a domain-separated initialization "
    "transcript digest plus diagnostics bounded by "
    "maximum_witness_diagnostic_bytes_per_proposal; successful work is never "
    "hidden in a segment-only pool or charged to proposal one,while any "
    "initialization failure is fatal and releases no partial result; mandatory "
    "replay recomputes initialization under an identical separate ledger"
)
ENCOUNTER_ACCELERATION_BOUND_POLICY = (
    "under the simultaneous pre-crossing hypothesis d_ik(t)>rho_ik, set "
    "A_i=sum_{k!=i}(GM_k/rho_ik^2) in immutable k order and "
    "A_ij=A_i+A_j; all operands and sums are custom exact rationals"
)
ENCOUNTER_LINEAR_MINIMUM_POLICY = (
    "for one proposed signed step h_s set Delta=abs(h_s), "
    "r_ij=x_j-x_i, w_ij=sign(h_s)*(v_j-v_i), b=dot(r_ij,w_ij), "
    "c=dot(w_ij,w_ij),and q=dot(r_ij,r_ij), all exactly; the exact minimum "
    "ell^2 of norm(r_ij+u*w_ij)^2 on 0<=u<=Delta is q when c=0 or b>=0, "
    "norm(r_ij+Delta*w_ij)^2 when b<0 and -b>=c*Delta, and q-b^2/c "
    "otherwise; branch comparisons and ell^2 are exact rational"
)
ENCOUNTER_FIRST_CROSSING_CERTIFICATE = (
    "for every canonical pair simultaneously require the strict exact-"
    "rational inequality ell_ij^2 > "
    "(rho_ij+(1/2)*(A_i+A_j)*Delta^2)^2; if an earliest first crossing "
    "existed, all pair floors would hold beforehand, the acceleration bounds "
    "would bound the relative Taylor remainder by "
    "(1/2)*(A_i+A_j)*u^2, and the strict inequality would contradict contact "
    "at that crossing; equality is uncertified and is rejected"
)
ENCOUNTER_CERTIFICATE_CADENCE_POLICY = (
    "evaluate one all-pair simultaneous certificate from the currently "
    "accepted state before any RK stage force call for every proposed "
    "substep; a false strict inequality is a recoverable certificate rejection "
    "reduced by the fixed domain-rejection factor unless a step/rejection cap "
    "then requires fatal fail-closed termination; an uncertified proposal is "
    "not evidence that a collision or floor crossing occurs"
)
ENCOUNTER_CERTIFICATE_CLAIM_SCOPE = (
    "EXACT_NEWTONIAN_LOCAL_IVP_ISSUING_FROM_EACH_ACCEPTED_NUMERICAL_NODE_IS_"
    "NONCOLLIDING_WITH_RESPECT_TO_RETAINED_PAIR_FLOORS_ON_THAT_SUBSTEP_ONLY"
)
ENCOUNTER_CERTIFICATE_NONCLAIMS = (
    "NO_EVENT_TIME_OR_MINIMUM_DISTANCE_LOCALIZATION;NO_COLLISION_RESPONSE;"
    "NO_CLEARANCE_BEFORE_THE_INITIAL_STATE_OR_AFTER_THE_SEGMENT_ENDPOINT;"
    "NO_GLOBAL_OR_FUTURE_CLEARANCE;NO_TRAJECTORY_ACCURACY_FROM_THE_CLEARANCE_"
    "CERTIFICATE;NO_EXISTENCE_CLAIM_BEYOND_THE_ACCEPTED_LOCAL_IVP"
)

ENCOUNTER_STAGE_GUARD_POLICY = (
    "before each of the 13 derivative calls require every trial-stage "
    "position and velocity finite and exact-dyadic squared pair separation strictly "
    "greater than rho_ij^2; after a completed 13-stage attempt apply the same "
    "finite strict guard to the eighth-order candidate before error acceptance; "
    "a trial-local nonfinite stage,candidate,defect,derivative,or a stage/"
    "candidate at or below a floor is not an event and is a recoverable fixed-"
    "factor domain rejection; accepted-state nonfiniteness,initial floor "
    "failure,or exact-resource failure is fatal"
)
ENCOUNTER_COMPLETED_TRIAL_ARRAY_POLICY = (
    "after thirteen finite derivatives require accepted_positions,"
    "accepted_velocities,embedded_positions,embedded_velocities,position_defect,"
    "velocity_defect,accepted_position_carry,and accepted_velocity_carry all be "
    "exact NumPy float64 arrays of the state shape and entirely finite before "
    "candidate-domain or normalized-error evaluation"
)
ENCOUNTER_STAGE_ABORT_POLICY = (
    "a failed pre-derivative stage guard aborts that RK attempt immediately "
    "and retains the actual number from zero through twelve of derivative/"
    "force calls already completed; a nonfinite derivative is a distinct "
    "derivative-domain abort retaining the actual number from one through "
    "thirteen of calls made; only thirteen finite derivatives constitute a "
    "completed RK attempt,and candidate-domain and error rejections exist only "
    "after such a completed attempt"
)

ENCOUNTER_PAIR_ERROR_NORM = (
    "NORMALIZED_MAX_OF_ALL_CANONICAL_PAIR_RELATIVE_COMPONENTS_AND_GM_CENTROID_"
    "COMPONENTS"
)
ENCOUNTER_PAIR_ERROR_SCALE = (
    "for pair i<j and component k, defect is "
    "(trial.position_defect_j-trial.position_defect_i)_k or "
    "(trial.velocity_defect_j-trial.velocity_defect_i)_k directly from the "
    "frozen Fehlberg tableau result,never recomputed from Kahan-updated accepted "
    "minus embedded states; scale is "
    "pair_atol_ij+pair_rtol*max(abs(current_j-current_i)_k,"
    "abs(accepted_j-accepted_i)_k); apply separately to position and velocity"
)
ENCOUNTER_GM_CENTROID_ERROR_SCALE = (
    "C_x=sum_i(GM_i*x_i)/sum_i(GM_i) and C_v analogously in immutable body "
    "order; centroid defect is sum_i(GM_i*trial.position_defect_i)/sum_i(GM_i) "
    "or the velocity-defect analogue directly from the frozen tableau and "
    "scale is centroid_atol+centroid_rtol*max(abs(current_centroid_component),"
    "abs(accepted_centroid_component)); position and velocity controls are "
    "separate and all reductions use fixed left-to-right NumPy float64 order"
)
ENCOUNTER_ACCEPTANCE_POLICY = (
    "after a completed finite candidate-domain-safe 13-stage attempt accept "
    "iff the normalized maximum pair-and-centroid error is <=1; error=0 uses "
    "maximum_scale_factor,otherwise clamp safety_factor*error^(-1/8) between "
    "minimum_scale_factor and maximum_scale_factor"
)
ENCOUNTER_MINIMUM_STEP_FAILURE_POLICY = (
    "a proposal rejected at minimum_step_magnitude,an exact remaining chunk "
    "below that magnitude except a deterministic endpoint-remainder chunk,or "
    "a rejected terminal endpoint-remainder chunk strictly below minimum-step "
    "magnitude,or any exhausted proposal,acceptance,rejection,force,or exact-resource "
    "cap is fatal; an ordinary clipped remainder strictly greater than minimum "
    "may reject and retry or split,while equality is fatal on rejection"
)

ENCOUNTER_TRANSACTION_POLICY = (
    "copy validated inputs to owned work buffers before evaluation; never "
    "mutate caller buffers; keep accepted state,carries,local offset,and exact "
    "duration ledger unchanged until certificate,all stages,candidate guard,"
    "and error acceptance succeed; discard every rejected trial completely; "
    "on any fatal failure return no partial result"
)
ENCOUNTER_FORCE_EVALUATION_ACCOUNTING = (
    "record per proposal its disposition and actual derivative/force calls; "
    "certificate rejection has zero; stage-guard abort has zero through twelve; "
    "derivative-domain abort has one through thirteen; completed RK attempt has "
    "exactly thirteen finite derivative calls; total force evaluations equal "
    "the exact sum of per-proposal calls,not thirteen times all proposals"
)
ENCOUNTER_PROPOSAL_ACCOUNTING = (
    "proposals=certificate_rejections+stage_guard_aborts+derivative_domain_"
    "aborts+completed_rk_attempts; "
    "completed_rk_attempts=candidate_domain_rejections+error_rejections+"
    "accepted_substeps; rejected_substeps=certificate_rejections+stage_guard_"
    "aborts+derivative_domain_aborts+candidate_domain_rejections+error_"
    "rejections; total and consecutive rejection caps apply to this exact "
    "five-category sum; all counts are exact nonnegative built-in integers"
)
ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE = (
    "PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY_WITH_SEPARATE_AND_TOTAL_COUNTERS"
)
ENCOUNTER_VALIDATION_REPLAY_POLICY = (
    "one full deterministic replay from owned input recomputes adaptive "
    "decisions,exact certificates,stage guards,states,carries,local offsets,"
    "proposal dispositions,actual force counts,and witness hashes; require "
    "bitwise-identical arrays and identical scalar ledgers before release; "
    "resource caps and usage obey the separate-lane resource policy"
)
ENCOUNTER_VALIDATION_REPLAY_COUNT = 1
ENCOUNTER_WITNESS_CUSTODY_POLICY = (
    "retain per proposal only canonical pair/disposition identifiers,actual "
    "force-call count,bounded binary64 diagnostics,bounded integer bit-length "
    "diagnostics,and a domain-separated SHA-256 of the canonical exact "
    "comparison transcript; never retain unbounded rational numerators or "
    "denominators; replay recomputes every exact inequality"
)
ENCOUNTER_WITNESS_SERIALIZATION_POLICY = (
    "for initialization and each proposal serialize UTF-8 JSON with sort_keys=True,"
    "ensure_ascii=True,separators=(',',':'),allow_nan=False; payload contains "
    "method/version,phase,proposal index or null,signed-step float.hex or null,"
    "disposition,actual "
    "force calls,canonical pair indices,linear-minimum branch,and every exact "
    "comparison operand as either canonical base-10 numerator and positive "
    "denominator strings or canonical base-10 dyadic mantissa and built-in "
    "integer exponent; charge and prospectively cap the decimal and JSON byte "
    "envelope before conversion,serialization,retention,or hashing,and then charge "
    "every actual UTF-8 transcript byte before "
    "streaming domain+'\\0'+payload into SHA-256; retain only the digest and "
    "bounded diagnostics,not the serialized exact operands"
)
ENCOUNTER_RESULT_CUSTODY_POLICY = (
    "result arrays and ledgers are owned immutable copies bound to input "
    "provenance,body order,pair tables,tolerances,resources,duration,schedule,"
    "primary accounting,and replay accounting by domain-separated checksums"
)
ENCOUNTER_CHECKSUM_BINDING_POLICY = (
    "pair-table checksum binds body order,floors,pair position/velocity atols "
    "and rtols,and GM-centroid tolerances; schedule and result checksums bind "
    "initial_epoch,the separate explicit endpoint_epoch,the retained signed "
    "duration,all accepted signed substeps,proposal dispositions,actual force "
    "counts,initialization and proposal witness digests,resource specifications,"
    "input provenance,primary accounting,and replay accounting"
)

ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN = (
    "jxplanetx.encounter-pair-table.content-integrity.v1"
)
ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_JSON_V1"
)
ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN = (
    "jxplanetx.encounter-initialization.content-integrity.v1"
)
ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN = (
    "jxplanetx.encounter-segment-schedule.content-integrity.v1"
)
ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_JSON_V1"
ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN = (
    "jxplanetx.encounter-exact-clearance-witness.content-integrity.v1"
)
ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM = (
    "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
)
ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN = (
    "jxplanetx.encounter-segment-result.content-integrity.v1"
)
ENCOUNTER_EVIDENCE_CLASS = "MODEL_OUTPUT"

ENCOUNTER_HARD_MAXIMUM_BODY_COUNT = 16
ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES = 128
ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT = 120
ENCOUNTER_HARD_MAXIMUM_INTEGER_BITS = 8192
ENCOUNTER_HARD_MAXIMUM_RATIONAL_EXPONENT_MAGNITUDE = 4096
ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_REDUCTION = 16384
ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_GCD_ITERATIONS = 500_000
ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_OPERATIONS = 250_000
ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_TRANSCRIPT_BYTES = 1_048_576
ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_PROPOSAL = 500_000
ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_SEGMENT = 64_500_000
ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_PROPOSAL = 650_000
ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_SEGMENT = 83_450_000
ENCOUNTER_HARD_MAXIMUM_WITNESS_DIAGNOSTIC_BYTES_PER_PROPOSAL = 4096
ENCOUNTER_HARD_MAXIMUM_WITNESS_LEDGER_BYTES = 16_777_216
ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_PROPOSAL = 1_310_720
ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_SEGMENT = 168_820_736
ENCOUNTER_HARD_MAXIMUM_SUBSTEP_PROPOSALS = 128
ENCOUNTER_HARD_MAXIMUM_FORCE_EVALUATIONS = 1664


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite built-in float")
    return value


def _positive_float(value: object, label: str) -> float:
    checked = _finite_float(value, label)
    if checked <= 0.0:
        raise ContractError(f"{label} must be positive")
    return checked


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{label} must be a positive built-in integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a nonnegative built-in integer")
    return value


def _body_order(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) < 2:
        raise ContractError("body_order must be an exact tuple of at least two IDs")
    if len(value) > ENCOUNTER_HARD_MAXIMUM_BODY_COUNT:
        raise ContractError("body_order exceeds the fixed hard body-count cap")
    for item in value:
        if type(item) is not str:
            raise ContractError("body_order entries must be built-in strings")
        if len(item) > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES:
            raise ContractError("body_order entry exceeds the UTF-8 byte cap")
        try:
            encoded = item.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ContractError("body_order entries must be valid UTF-8") from exc
        if len(encoded) > ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES:
            raise ContractError("body_order entry exceeds the UTF-8 byte cap")
        if not item or item.strip() != item:
            raise ContractError(
                "body_order entries must be nonempty trimmed built-in strings"
            )
    if len(set(value)) != len(value):
        raise ContractError("body_order must not repeat a body ID")
    return value


def _positive_float_table(
    value: object,
    label: str,
    expected_length: int,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != expected_length:
        raise ContractError(
            f"{label} must be an exact tuple of {expected_length} built-in floats"
        )
    for index, item in enumerate(value):
        _positive_float(item, f"{label}[{index}]")
    return value


def _validate_binary64_rational_envelope(
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
        raise ContractError(
            f"{label} exceeds maximum_rational_exponent_magnitude"
        )


@dataclass(frozen=True, eq=False)
class EncounterExactRationalResourceSpec:
    """Caller-selected hard bounds for custom exact encounter arithmetic."""

    maximum_body_count: int
    maximum_pair_count: int
    maximum_integer_bits: int
    maximum_rational_exponent_magnitude: int
    maximum_gcd_iterations_per_reduction: int
    maximum_initialization_operations: int
    maximum_initialization_gcd_iterations: int
    maximum_initialization_transcript_bytes: int
    maximum_gcd_iterations_per_proposal: int
    maximum_gcd_iterations_per_segment: int
    maximum_operations_per_proposal: int
    maximum_operations_per_segment: int
    maximum_witness_transcript_bytes_per_proposal: int
    maximum_witness_transcript_bytes_per_segment: int
    maximum_witness_diagnostic_bytes_per_proposal: int
    maximum_witness_ledger_bytes: int

    representation: str = ENCOUNTER_EXACT_RATIONAL_REPRESENTATION
    operation_policy: str = ENCOUNTER_EXACT_RATIONAL_OPERATION_POLICY
    resource_policy: str = ENCOUNTER_EXACT_RATIONAL_RESOURCE_POLICY
    exponent_policy: str = ENCOUNTER_RATIONAL_EXPONENT_POLICY
    work_weight_table_id: str = ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID
    work_weight_table: tuple[tuple[str, str, int], ...] = (
        ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE
    )
    no_floating_fallback: bool = True

    def __post_init__(self) -> None:
        hard_caps = {
            "maximum_body_count": ENCOUNTER_HARD_MAXIMUM_BODY_COUNT,
            "maximum_pair_count": ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT,
            "maximum_integer_bits": ENCOUNTER_HARD_MAXIMUM_INTEGER_BITS,
            "maximum_rational_exponent_magnitude": ENCOUNTER_HARD_MAXIMUM_RATIONAL_EXPONENT_MAGNITUDE,
            "maximum_gcd_iterations_per_reduction": ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_REDUCTION,
            "maximum_initialization_operations": ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_OPERATIONS,
            "maximum_initialization_gcd_iterations": ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_GCD_ITERATIONS,
            "maximum_initialization_transcript_bytes": ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_TRANSCRIPT_BYTES,
            "maximum_gcd_iterations_per_proposal": ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_PROPOSAL,
            "maximum_gcd_iterations_per_segment": ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_SEGMENT,
            "maximum_operations_per_proposal": ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_PROPOSAL,
            "maximum_operations_per_segment": ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_SEGMENT,
            "maximum_witness_transcript_bytes_per_proposal": ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_PROPOSAL,
            "maximum_witness_transcript_bytes_per_segment": ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_SEGMENT,
            "maximum_witness_diagnostic_bytes_per_proposal": ENCOUNTER_HARD_MAXIMUM_WITNESS_DIAGNOSTIC_BYTES_PER_PROPOSAL,
            "maximum_witness_ledger_bytes": ENCOUNTER_HARD_MAXIMUM_WITNESS_LEDGER_BYTES,
        }
        for name, hard_cap in hard_caps.items():
            checked = _positive_int(getattr(self, name), name)
            if checked > hard_cap:
                raise ContractError(f"{name} cannot exceed hard cap {hard_cap}")
        if self.maximum_operations_per_segment < self.maximum_operations_per_proposal:
            raise ContractError(
                "maximum_operations_per_segment cannot be below "
                "maximum_operations_per_proposal"
            )
        if self.maximum_operations_per_segment < self.maximum_initialization_operations:
            raise ContractError(
                "maximum_operations_per_segment cannot be below the "
                "initialization operation cap"
            )
        if (
            self.maximum_gcd_iterations_per_segment
            < self.maximum_gcd_iterations_per_proposal
        ):
            raise ContractError(
                "maximum_gcd_iterations_per_segment cannot be below the "
                "per-proposal GCD cap"
            )
        if (
            self.maximum_gcd_iterations_per_segment
            < self.maximum_initialization_gcd_iterations
        ):
            raise ContractError(
                "maximum_gcd_iterations_per_segment cannot be below the "
                "initialization GCD cap"
            )
        if (
            self.maximum_witness_transcript_bytes_per_segment
            < self.maximum_witness_transcript_bytes_per_proposal
        ):
            raise ContractError(
                "maximum_witness_transcript_bytes_per_segment cannot be below "
                "the per-proposal transcript cap"
            )
        if (
            self.maximum_witness_transcript_bytes_per_segment
            < self.maximum_initialization_transcript_bytes
        ):
            raise ContractError(
                "maximum_witness_transcript_bytes_per_segment cannot be below "
                "the initialization transcript cap"
            )
        if (
            self.maximum_witness_transcript_bytes_per_segment
            < self.maximum_initialization_transcript_bytes
            + self.maximum_witness_transcript_bytes_per_proposal
        ):
            raise ContractError(
                "the segment transcript cap must reserve initialization and at "
                "least one proposal transcript"
            )
        if (
            self.maximum_witness_ledger_bytes
            < self.maximum_witness_diagnostic_bytes_per_proposal
        ):
            raise ContractError(
                "maximum_witness_ledger_bytes cannot be below the per-proposal cap"
            )

        if (
            type(self.work_weight_table) is not tuple
            or len(self.work_weight_table)
            != len(ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE)
        ):
            raise ContractError(
                "work_weight_table must be an exact fixed-length tuple"
            )
        for entry in self.work_weight_table:
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or type(entry[0]) is not str
                or type(entry[1]) is not str
                or type(entry[2]) is not int
            ):
                raise ContractError(
                    "work_weight_table entries must be exact (str,str,int) tuples"
                )

        exact = {
            "representation": ENCOUNTER_EXACT_RATIONAL_REPRESENTATION,
            "operation_policy": ENCOUNTER_EXACT_RATIONAL_OPERATION_POLICY,
            "resource_policy": ENCOUNTER_EXACT_RATIONAL_RESOURCE_POLICY,
            "exponent_policy": ENCOUNTER_RATIONAL_EXPONENT_POLICY,
            "work_weight_table_id": ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID,
            "work_weight_table": ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE,
            "no_floating_fallback": True,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(
                    f"{name} must equal the fixed exact-rational value {expected!r}"
                )


@dataclass(frozen=True, eq=False)
class AdaptiveEncounterSegmentSpec:
    """One full-Cartesian RKF78 local encounter IVP from offset zero to h."""

    body_order: tuple[str, ...]
    initial_epoch: float
    endpoint_epoch: float
    duration: float
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

    method_id: str = ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID
    method_class: str = ENCOUNTER_METHOD_CLASS
    principal_order: int = ENCOUNTER_PRINCIPAL_ORDER
    accepted_order: int = ENCOUNTER_ACCEPTED_ORDER
    embedded_order: int = ENCOUNTER_EMBEDDED_ORDER
    stage_count: int = ENCOUNTER_STAGE_COUNT
    tableau_id: str = ENCOUNTER_TABLEAU_ID
    source_report: str = ENCOUNTER_SOURCE_REPORT
    source_document_id: str = ENCOUNTER_SOURCE_DOCUMENT_ID
    source_title: str = ENCOUNTER_SOURCE_TITLE
    source_author: str = ENCOUNTER_SOURCE_AUTHOR
    source_publication_date: str = ENCOUNTER_SOURCE_PUBLICATION_DATE
    source_url: str = ENCOUNTER_SOURCE_URL
    accepted_solution: str = ENCOUNTER_ACCEPTED_SOLUTION
    embedded_solution: str = ENCOUNTER_EMBEDDED_SOLUTION
    defect_orientation: str = ENCOUNTER_DEFECT_ORIENTATION
    controller_exponent: float = ENCOUNTER_CONTROLLER_EXPONENT
    domain_rejection_scale_factor: float = ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR
    backend_scope: str = ENCOUNTER_BACKEND_SCOPE
    force_plan_scope: str = ENCOUNTER_FORCE_PLAN_SCOPE
    required_frame: str = ENCOUNTER_REQUIRED_FRAME
    required_origin: str = ENCOUNTER_REQUIRED_ORIGIN
    allowed_axes: tuple[str, ...] = ENCOUNTER_ALLOWED_AXES
    allowed_time_scales: tuple[str, ...] = ENCOUNTER_ALLOWED_TIME_SCALES
    dtype: str = ENCOUNTER_DTYPE
    body_order_policy: str = ENCOUNTER_BODY_ORDER_POLICY
    pair_table_policy: str = ENCOUNTER_PAIR_TABLE_POLICY
    input_binding_policy: str = ENCOUNTER_INPUT_BINDING_POLICY
    time_policy: str = ENCOUNTER_TIME_POLICY
    stage_epoch_policy: str = ENCOUNTER_STAGE_EPOCH_POLICY
    step_scheduler_policy: str = ENCOUNTER_STEP_SCHEDULER_POLICY
    accepted_state_accumulation: str = ENCOUNTER_ACCEPTED_STATE_ACCUMULATION
    carry_entry_policy: str = ENCOUNTER_CARRY_ENTRY_POLICY
    effective_floor_policy: str = ENCOUNTER_EFFECTIVE_FLOOR_POLICY
    initial_domain_policy: str = ENCOUNTER_INITIAL_DOMAIN_POLICY
    initialization_resource_policy: str = ENCOUNTER_INITIALIZATION_RESOURCE_POLICY
    acceleration_bound_policy: str = ENCOUNTER_ACCELERATION_BOUND_POLICY
    linear_minimum_policy: str = ENCOUNTER_LINEAR_MINIMUM_POLICY
    first_crossing_certificate: str = ENCOUNTER_FIRST_CROSSING_CERTIFICATE
    certificate_cadence_policy: str = ENCOUNTER_CERTIFICATE_CADENCE_POLICY
    certificate_claim_scope: str = ENCOUNTER_CERTIFICATE_CLAIM_SCOPE
    certificate_nonclaims: str = ENCOUNTER_CERTIFICATE_NONCLAIMS
    stage_guard_policy: str = ENCOUNTER_STAGE_GUARD_POLICY
    stage_abort_policy: str = ENCOUNTER_STAGE_ABORT_POLICY
    completed_trial_array_policy: str = ENCOUNTER_COMPLETED_TRIAL_ARRAY_POLICY
    error_norm: str = ENCOUNTER_PAIR_ERROR_NORM
    pair_error_scale: str = ENCOUNTER_PAIR_ERROR_SCALE
    gm_centroid_error_scale: str = ENCOUNTER_GM_CENTROID_ERROR_SCALE
    acceptance_policy: str = ENCOUNTER_ACCEPTANCE_POLICY
    minimum_step_failure_policy: str = ENCOUNTER_MINIMUM_STEP_FAILURE_POLICY
    transaction_policy: str = ENCOUNTER_TRANSACTION_POLICY
    force_evaluation_accounting: str = ENCOUNTER_FORCE_EVALUATION_ACCOUNTING
    proposal_accounting: str = ENCOUNTER_PROPOSAL_ACCOUNTING
    public_execution_accounting_scope: str = (
        ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE
    )
    validation_replay_policy: str = ENCOUNTER_VALIDATION_REPLAY_POLICY
    validation_replay_count: int = ENCOUNTER_VALIDATION_REPLAY_COUNT
    resource_lane_policy: str = ENCOUNTER_RESOURCE_LANE_POLICY
    resource_nonclaim_policy: str = ENCOUNTER_RESOURCE_NONCLAIM_POLICY
    witness_custody_policy: str = ENCOUNTER_WITNESS_CUSTODY_POLICY
    witness_serialization_policy: str = ENCOUNTER_WITNESS_SERIALIZATION_POLICY
    result_custody_policy: str = ENCOUNTER_RESULT_CUSTODY_POLICY
    checksum_binding_policy: str = ENCOUNTER_CHECKSUM_BINDING_POLICY
    pair_table_checksum_algorithm: str = ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM
    pair_table_checksum_domain: str = ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN
    initialization_checksum_algorithm: str = ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM
    initialization_checksum_domain: str = ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN
    schedule_checksum_algorithm: str = ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM
    schedule_checksum_domain: str = ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN
    exact_witness_checksum_algorithm: str = (
        ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM
    )
    exact_witness_checksum_domain: str = ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN
    result_content_checksum_algorithm: str = (
        ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM
    )
    result_content_checksum_domain: str = ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN
    adaptive: bool = True
    dense_output: bool = False
    event_detection: bool = False
    collision_response: bool = False
    globally_symplectic: bool = False
    exactly_time_reversible: bool = False
    evidence_class: str = ENCOUNTER_EVIDENCE_CLASS
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        body_order = _body_order(self.body_order)
        pair_count = len(body_order) * (len(body_order) - 1) // 2

        initial_epoch = _finite_float(self.initial_epoch, "initial_epoch")
        endpoint_epoch = _finite_float(self.endpoint_epoch, "endpoint_epoch")
        duration = _finite_float(self.duration, "duration")
        if duration == 0.0:
            raise ContractError("duration must be nonzero")
        if (duration > 0.0 and endpoint_epoch <= initial_epoch) or (
            duration < 0.0 and endpoint_epoch >= initial_epoch
        ):
            raise ContractError(
                "endpoint_epoch must be strictly monotone in sign(duration)"
            )

        initial_step = _positive_float(
            self.initial_step_magnitude, "initial_step_magnitude"
        )
        minimum_step = _positive_float(
            self.minimum_step_magnitude, "minimum_step_magnitude"
        )
        maximum_step = _positive_float(
            self.maximum_step_magnitude, "maximum_step_magnitude"
        )
        if not minimum_step <= initial_step <= maximum_step:
            raise ContractError(
                "step magnitudes must satisfy minimum <= initial <= maximum"
            )
        if initial_step > abs(duration):
            raise ContractError("initial_step_magnitude cannot exceed abs(duration)")

        _positive_float_table(
            self.pair_certification_floors,
            "pair_certification_floors",
            pair_count,
        )
        _positive_float_table(
            self.pair_position_atols,
            "pair_position_atols",
            pair_count,
        )
        _positive_float_table(
            self.pair_velocity_atols,
            "pair_velocity_atols",
            pair_count,
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
            _positive_float(getattr(self, name), name)
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
            _positive_int(getattr(self, name), name)
        for name in (
            "maximum_rejected_substeps",
            "maximum_consecutive_rejections",
        ):
            _nonnegative_int(getattr(self, name), name)
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
                "maximum_consecutive_rejections cannot exceed "
                "maximum_rejected_substeps"
            )
        if self.maximum_substep_proposals > ENCOUNTER_HARD_MAXIMUM_SUBSTEP_PROPOSALS:
            raise ContractError(
                "maximum_substep_proposals exceeds its fixed hard cap"
            )
        if self.maximum_force_evaluations < ENCOUNTER_STAGE_COUNT:
            raise ContractError(
                "maximum_force_evaluations must permit one completed RKF78 attempt"
            )
        if self.maximum_force_evaluations > ENCOUNTER_HARD_MAXIMUM_FORCE_EVALUATIONS:
            raise ContractError(
                "maximum_force_evaluations exceeds its fixed hard cap"
            )

        if type(self.exact_rational_resources) is not EncounterExactRationalResourceSpec:
            raise ContractError(
                "exact_rational_resources must be an exact "
                "EncounterExactRationalResourceSpec"
            )
        self.exact_rational_resources.__post_init__()
        resources = self.exact_rational_resources
        if len(body_order) > resources.maximum_body_count:
            raise ContractError("body_order exceeds maximum_body_count")
        if pair_count > resources.maximum_pair_count:
            raise ContractError("canonical pair count exceeds maximum_pair_count")
        rational_binary64_inputs = (
            (self.duration, "duration"),
            (self.initial_step_magnitude, "initial_step_magnitude"),
            (self.minimum_step_magnitude, "minimum_step_magnitude"),
            (self.maximum_step_magnitude, "maximum_step_magnitude"),
            *tuple(
                (value, f"pair_certification_floors[{index}]")
                for index, value in enumerate(self.pair_certification_floors)
            ),
        )
        for value, label in rational_binary64_inputs:
            _validate_binary64_rational_envelope(value, resources, label)
        if (
            (self.maximum_substep_proposals + 1)
            * resources.maximum_witness_diagnostic_bytes_per_proposal
            > resources.maximum_witness_ledger_bytes
        ):
            raise ContractError(
                "the witness ledger cap cannot hold the prospective per-proposal "
                "diagnostic maximum"
            )
        if (
            resources.maximum_initialization_operations
            + self.maximum_substep_proposals
            * resources.maximum_operations_per_proposal
            > resources.maximum_operations_per_segment
        ):
            raise ContractError(
                "the segment abstract exact-work cap cannot reserve the "
                "per-proposal maximum for every allowed proposal"
            )
        if (
            resources.maximum_initialization_gcd_iterations
            + self.maximum_substep_proposals
            * resources.maximum_gcd_iterations_per_proposal
            > resources.maximum_gcd_iterations_per_segment
        ):
            raise ContractError(
                "the segment GCD-iteration cap cannot reserve the per-proposal "
                "maximum for every allowed proposal"
            )
        if (
            resources.maximum_initialization_transcript_bytes
            + self.maximum_substep_proposals
            * resources.maximum_witness_transcript_bytes_per_proposal
            > resources.maximum_witness_transcript_bytes_per_segment
        ):
            raise ContractError(
                "the segment transcript cap cannot reserve the per-proposal "
                "maximum for every allowed proposal"
            )

        exact = {
            "method_id": ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID,
            "method_class": ENCOUNTER_METHOD_CLASS,
            "principal_order": ENCOUNTER_PRINCIPAL_ORDER,
            "accepted_order": ENCOUNTER_ACCEPTED_ORDER,
            "embedded_order": ENCOUNTER_EMBEDDED_ORDER,
            "stage_count": ENCOUNTER_STAGE_COUNT,
            "tableau_id": ENCOUNTER_TABLEAU_ID,
            "source_report": ENCOUNTER_SOURCE_REPORT,
            "source_document_id": ENCOUNTER_SOURCE_DOCUMENT_ID,
            "source_title": ENCOUNTER_SOURCE_TITLE,
            "source_author": ENCOUNTER_SOURCE_AUTHOR,
            "source_publication_date": ENCOUNTER_SOURCE_PUBLICATION_DATE,
            "source_url": ENCOUNTER_SOURCE_URL,
            "accepted_solution": ENCOUNTER_ACCEPTED_SOLUTION,
            "embedded_solution": ENCOUNTER_EMBEDDED_SOLUTION,
            "defect_orientation": ENCOUNTER_DEFECT_ORIENTATION,
            "controller_exponent": ENCOUNTER_CONTROLLER_EXPONENT,
            "domain_rejection_scale_factor": ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR,
            "backend_scope": ENCOUNTER_BACKEND_SCOPE,
            "force_plan_scope": ENCOUNTER_FORCE_PLAN_SCOPE,
            "required_frame": ENCOUNTER_REQUIRED_FRAME,
            "required_origin": ENCOUNTER_REQUIRED_ORIGIN,
            "allowed_axes": ENCOUNTER_ALLOWED_AXES,
            "allowed_time_scales": ENCOUNTER_ALLOWED_TIME_SCALES,
            "dtype": ENCOUNTER_DTYPE,
            "body_order_policy": ENCOUNTER_BODY_ORDER_POLICY,
            "pair_table_policy": ENCOUNTER_PAIR_TABLE_POLICY,
            "input_binding_policy": ENCOUNTER_INPUT_BINDING_POLICY,
            "time_policy": ENCOUNTER_TIME_POLICY,
            "stage_epoch_policy": ENCOUNTER_STAGE_EPOCH_POLICY,
            "step_scheduler_policy": ENCOUNTER_STEP_SCHEDULER_POLICY,
            "accepted_state_accumulation": ENCOUNTER_ACCEPTED_STATE_ACCUMULATION,
            "carry_entry_policy": ENCOUNTER_CARRY_ENTRY_POLICY,
            "effective_floor_policy": ENCOUNTER_EFFECTIVE_FLOOR_POLICY,
            "initial_domain_policy": ENCOUNTER_INITIAL_DOMAIN_POLICY,
            "initialization_resource_policy": ENCOUNTER_INITIALIZATION_RESOURCE_POLICY,
            "acceleration_bound_policy": ENCOUNTER_ACCELERATION_BOUND_POLICY,
            "linear_minimum_policy": ENCOUNTER_LINEAR_MINIMUM_POLICY,
            "first_crossing_certificate": ENCOUNTER_FIRST_CROSSING_CERTIFICATE,
            "certificate_cadence_policy": ENCOUNTER_CERTIFICATE_CADENCE_POLICY,
            "certificate_claim_scope": ENCOUNTER_CERTIFICATE_CLAIM_SCOPE,
            "certificate_nonclaims": ENCOUNTER_CERTIFICATE_NONCLAIMS,
            "stage_guard_policy": ENCOUNTER_STAGE_GUARD_POLICY,
            "stage_abort_policy": ENCOUNTER_STAGE_ABORT_POLICY,
            "completed_trial_array_policy": ENCOUNTER_COMPLETED_TRIAL_ARRAY_POLICY,
            "error_norm": ENCOUNTER_PAIR_ERROR_NORM,
            "pair_error_scale": ENCOUNTER_PAIR_ERROR_SCALE,
            "gm_centroid_error_scale": ENCOUNTER_GM_CENTROID_ERROR_SCALE,
            "acceptance_policy": ENCOUNTER_ACCEPTANCE_POLICY,
            "minimum_step_failure_policy": ENCOUNTER_MINIMUM_STEP_FAILURE_POLICY,
            "transaction_policy": ENCOUNTER_TRANSACTION_POLICY,
            "force_evaluation_accounting": ENCOUNTER_FORCE_EVALUATION_ACCOUNTING,
            "proposal_accounting": ENCOUNTER_PROPOSAL_ACCOUNTING,
            "public_execution_accounting_scope": ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE,
            "validation_replay_policy": ENCOUNTER_VALIDATION_REPLAY_POLICY,
            "validation_replay_count": ENCOUNTER_VALIDATION_REPLAY_COUNT,
            "resource_lane_policy": ENCOUNTER_RESOURCE_LANE_POLICY,
            "resource_nonclaim_policy": ENCOUNTER_RESOURCE_NONCLAIM_POLICY,
            "witness_custody_policy": ENCOUNTER_WITNESS_CUSTODY_POLICY,
            "witness_serialization_policy": ENCOUNTER_WITNESS_SERIALIZATION_POLICY,
            "result_custody_policy": ENCOUNTER_RESULT_CUSTODY_POLICY,
            "checksum_binding_policy": ENCOUNTER_CHECKSUM_BINDING_POLICY,
            "pair_table_checksum_algorithm": ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM,
            "pair_table_checksum_domain": ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN,
            "initialization_checksum_algorithm": ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM,
            "initialization_checksum_domain": ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN,
            "schedule_checksum_algorithm": ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM,
            "schedule_checksum_domain": ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN,
            "exact_witness_checksum_algorithm": ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM,
            "exact_witness_checksum_domain": ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN,
            "result_content_checksum_algorithm": ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            "result_content_checksum_domain": ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "adaptive": True,
            "dense_output": False,
            "event_detection": False,
            "collision_response": False,
            "globally_symplectic": False,
            "exactly_time_reversible": False,
            "evidence_class": ENCOUNTER_EVIDENCE_CLASS,
            "registry_authorized": False,
            "qualification_authorized": False,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(
                    f"{name} must equal the fixed encounter value {expected!r}"
                )

    @property
    def direction(self) -> str:
        return "FORWARD" if self.duration > 0.0 else "BACKWARD"

    @property
    def canonical_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.body_order[left], self.body_order[right])
            for left in range(len(self.body_order) - 1)
            for right in range(left + 1, len(self.body_order))
        )


__all__ = [
    "ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID",
    "AdaptiveEncounterSegmentSpec",
    "ENCOUNTER_ACCELERATION_BOUND_POLICY",
    "ENCOUNTER_ACCEPTANCE_POLICY",
    "ENCOUNTER_ACCEPTED_ORDER",
    "ENCOUNTER_ACCEPTED_SOLUTION",
    "ENCOUNTER_ACCEPTED_STATE_ACCUMULATION",
    "ENCOUNTER_ALLOWED_AXES",
    "ENCOUNTER_ALLOWED_TIME_SCALES",
    "ENCOUNTER_BACKEND_SCOPE",
    "ENCOUNTER_BODY_ORDER_POLICY",
    "ENCOUNTER_CARRY_ENTRY_POLICY",
    "ENCOUNTER_CERTIFICATE_CADENCE_POLICY",
    "ENCOUNTER_CERTIFICATE_CLAIM_SCOPE",
    "ENCOUNTER_CERTIFICATE_NONCLAIMS",
    "ENCOUNTER_CHECKSUM_BINDING_POLICY",
    "ENCOUNTER_COMPLETED_TRIAL_ARRAY_POLICY",
    "ENCOUNTER_CONTROLLER_EXPONENT",
    "ENCOUNTER_DEFECT_ORIENTATION",
    "ENCOUNTER_DOMAIN_REJECTION_SCALE_FACTOR",
    "ENCOUNTER_DTYPE",
    "ENCOUNTER_EFFECTIVE_FLOOR_POLICY",
    "ENCOUNTER_EMBEDDED_ORDER",
    "ENCOUNTER_EMBEDDED_SOLUTION",
    "ENCOUNTER_PAIR_ERROR_NORM",
    "ENCOUNTER_EVIDENCE_CLASS",
    "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE",
    "ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHT_TABLE_ID",
    "ENCOUNTER_EXACT_RATIONAL_OPERATION_POLICY",
    "ENCOUNTER_EXACT_RATIONAL_REPRESENTATION",
    "ENCOUNTER_EXACT_RATIONAL_RESOURCE_POLICY",
    "ENCOUNTER_EXACT_WITNESS_CHECKSUM_ALGORITHM",
    "ENCOUNTER_EXACT_WITNESS_CHECKSUM_DOMAIN",
    "ENCOUNTER_FIRST_CROSSING_CERTIFICATE",
    "ENCOUNTER_FORCE_EVALUATION_ACCOUNTING",
    "ENCOUNTER_FORCE_PLAN_SCOPE",
    "ENCOUNTER_GM_CENTROID_ERROR_SCALE",
    "ENCOUNTER_HARD_MAXIMUM_BODY_COUNT",
    "ENCOUNTER_HARD_MAXIMUM_BODY_ID_UTF8_BYTES",
    "ENCOUNTER_HARD_MAXIMUM_FORCE_EVALUATIONS",
    "ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_PROPOSAL",
    "ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_REDUCTION",
    "ENCOUNTER_HARD_MAXIMUM_GCD_ITERATIONS_PER_SEGMENT",
    "ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_GCD_ITERATIONS",
    "ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_OPERATIONS",
    "ENCOUNTER_HARD_MAXIMUM_INITIALIZATION_TRANSCRIPT_BYTES",
    "ENCOUNTER_HARD_MAXIMUM_INTEGER_BITS",
    "ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_PROPOSAL",
    "ENCOUNTER_HARD_MAXIMUM_OPERATIONS_PER_SEGMENT",
    "ENCOUNTER_HARD_MAXIMUM_PAIR_COUNT",
    "ENCOUNTER_HARD_MAXIMUM_RATIONAL_EXPONENT_MAGNITUDE",
    "ENCOUNTER_HARD_MAXIMUM_SUBSTEP_PROPOSALS",
    "ENCOUNTER_HARD_MAXIMUM_WITNESS_DIAGNOSTIC_BYTES_PER_PROPOSAL",
    "ENCOUNTER_HARD_MAXIMUM_WITNESS_LEDGER_BYTES",
    "ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_PROPOSAL",
    "ENCOUNTER_HARD_MAXIMUM_WITNESS_TRANSCRIPT_BYTES_PER_SEGMENT",
    "ENCOUNTER_INITIAL_DOMAIN_POLICY",
    "ENCOUNTER_INITIALIZATION_CHECKSUM_ALGORITHM",
    "ENCOUNTER_INITIALIZATION_CHECKSUM_DOMAIN",
    "ENCOUNTER_INITIALIZATION_RESOURCE_POLICY",
    "ENCOUNTER_INPUT_BINDING_POLICY",
    "ENCOUNTER_LINEAR_MINIMUM_POLICY",
    "ENCOUNTER_METHOD_CLASS",
    "ENCOUNTER_MINIMUM_STEP_FAILURE_POLICY",
    "ENCOUNTER_PAIR_ERROR_SCALE",
    "ENCOUNTER_PAIR_TABLE_CHECKSUM_ALGORITHM",
    "ENCOUNTER_PAIR_TABLE_CHECKSUM_DOMAIN",
    "ENCOUNTER_PAIR_TABLE_POLICY",
    "ENCOUNTER_PRINCIPAL_ORDER",
    "ENCOUNTER_PROPOSAL_ACCOUNTING",
    "ENCOUNTER_PUBLIC_EXECUTION_ACCOUNTING_SCOPE",
    "ENCOUNTER_RATIONAL_EXPONENT_POLICY",
    "ENCOUNTER_REQUIRED_FRAME",
    "ENCOUNTER_REQUIRED_ORIGIN",
    "ENCOUNTER_RESOURCE_LANE_POLICY",
    "ENCOUNTER_RESOURCE_NONCLAIM_POLICY",
    "ENCOUNTER_RESULT_CONTENT_CHECKSUM_ALGORITHM",
    "ENCOUNTER_RESULT_CONTENT_CHECKSUM_DOMAIN",
    "ENCOUNTER_RESULT_CUSTODY_POLICY",
    "ENCOUNTER_SCHEDULE_CHECKSUM_ALGORITHM",
    "ENCOUNTER_SCHEDULE_CHECKSUM_DOMAIN",
    "ENCOUNTER_STAGE_ABORT_POLICY",
    "ENCOUNTER_STAGE_COUNT",
    "ENCOUNTER_STAGE_EPOCH_POLICY",
    "ENCOUNTER_STAGE_GUARD_POLICY",
    "ENCOUNTER_STEP_SCHEDULER_POLICY",
    "ENCOUNTER_SOURCE_AUTHOR",
    "ENCOUNTER_SOURCE_DOCUMENT_ID",
    "ENCOUNTER_SOURCE_PUBLICATION_DATE",
    "ENCOUNTER_SOURCE_REPORT",
    "ENCOUNTER_SOURCE_TITLE",
    "ENCOUNTER_SOURCE_URL",
    "ENCOUNTER_TABLEAU_ID",
    "ENCOUNTER_TIME_POLICY",
    "ENCOUNTER_TRANSACTION_POLICY",
    "ENCOUNTER_VALIDATION_REPLAY_COUNT",
    "ENCOUNTER_VALIDATION_REPLAY_POLICY",
    "ENCOUNTER_WITNESS_CUSTODY_POLICY",
    "ENCOUNTER_WITNESS_SERIALIZATION_POLICY",
    "EncounterExactRationalResourceSpec",
]
