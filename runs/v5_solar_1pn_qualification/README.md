# JX V5 Solar 1PN qualification preregistration

**Broad protocol phase:** `SPECIFICATION_ONLY_BLOCKED`

**Machine plan state:**
`PRELOCKED_AFTER_DISCLOSED_DEVELOPMENT_TESTS_BEFORE_HOLDOUT_OUTCOMES`

**Run verdict:** `NOT_RUN`

**Evidence class of every future numerical result:** `MODEL_OUTPUT`

**Registry authority:** `false`

**Model under test:**
`relativity.solar_schwarzschild_test_particle_1pn`

## Purpose and nonauthority

This directory defines the qualification that must be completed before JX may
ask for scientific review of its restricted Solar Schwarzschild test-particle
1PN reference path. It deliberately contains no trajectory, oracle output,
qualification verdict, physical-parameter default, or authorization record.

The subject is the static, spherical, nonspinning Solar monopole correction for
a massless target with no backreaction:

\[
\Delta\mathbf a=
\frac{\mu}{c^2r^3}
\left[
\left(\frac{4\mu}{r}-v^2\right)\mathbf r
+4(\mathbf r\mathbin{\cdot}\mathbf v)\mathbf v
\right].
\]

It is a correction only. A qualifying evaluation must apply exactly one
Newtonian point-mass base and exactly one Solar 1PN correction. It must not
silently substitute a full EIH, restricted PPN, Solar-spin, multipole, or other
relativistic model.

Even complete passage of this protocol would establish only that the frozen JX
reference path reproduced this declared differential equation within frozen
budgets for the named cases. It would not make the V5 registry executable,
qualify a production propagator, validate a physical ephemeris, or authorize a
scientific result.

The registry therefore remains `DRAFT_NONEXECUTABLE`, and the model remains
`NOT_IMPLEMENTED`, `UNQUALIFIED`, and `BLOCKED`. No lower qualification rung
waives a missing higher rung.

## Disclosed prior development evidence

Commit
`58b57094d519205f3ef398f3c738af96107c59e5`
(`Add nonauthorizing V5 1PN reference integrator`) is disclosed before this
preregistration. Its tests and all cases derived from them are development
evidence, not blind, held-out, externally independent, or authorizing evidence.

The disclosed tests already exercise:

- exact Newtonian-plus-1PN force composition and ledger closure;
- a soluble linear velocity-dependent midpoint fixture;
- second-order behavior on a harmonic oscillator and a short Solar arc;
- negative-step recovery, forced nonconvergence, and domain failure;
- a separately written in-test Decimal RK4 comparison;
- the large-\(c\) Newtonian limit;
- AU/day to kilometre/second conversion and signed-axis rotation;
- a synthetic perihelion case with \(\mu=1\), \(c=100\),
  \(a=1\), \(e=0.2\), step sizes `0.02` and `0.01`, linear event
  interpolation, binary-float `atan2`, and Richardson extrapolation; and
- nonauthorizing output metadata and isolation from the legacy integrators.

Those checks are valuable regression evidence, but their inputs, outputs, and
acceptance criteria have already been seen. The in-test RK4 arm shares the JX
repository, runtime, test process, and development team; it is not the
independent high-order oracle required by Q3. The disclosed perihelion observer
is not the observer required by Q2.

No disclosed case may be relabeled as a qualification holdout.

## Immutable qualification identity

A future run must have one immutable qualification identifier binding, by
canonical SHA-256 manifests:

- this protocol and the scientific contract;
- the draft-registry bytes inspected by the reference path;
- all JX source and test bytes used by the run;
- every equation source and retained provenance artifact;
- all physical and synthetic parameter values;
- frames, origins, axes, epochs, time scales, and unit definitions;
- applicability limits and omitted-term budgets;
- Decimal context, solver policy, step and precision grids, checkpoints,
  norms, event rules, and pass/fail thresholds;
- independent-oracle source, runtime, dependencies, method coefficients,
  configuration, and outputs;
- EIH comparator source, conventions, software, and outputs; and
- holdout generation, custody, sealed expectations, and unblinding records.

Any change to a bound byte, value, threshold, case, observer, or executable
environment creates a new qualification identifier. A rerun must not overwrite
or reinterpret an earlier result.

The machine identifier is
`jx.v5.solar_1pn.qualification.<sha256>`, where the digest is computed from the
canonical plan after replacing `qualification_id` with the fixed
`CONTENT_DIGEST_SENTINEL`. This avoids a self-hash cycle while making the
identifier depend on every plan field, including the frozen input and artifact
bindings. The separately reported `package_sha256` additionally binds both the
raw registration bytes and its canonical JSON digest. Registration formatting
therefore cannot change invisibly, and registration is not incorrectly folded
into its own pre-execution plan identity.

## States and verdicts

Protocol state and scientific verdict are distinct from registry authority.
The machine plan state above records that this exact package was frozen after
disclosed development tests and before holdout outcomes; it does not mean the
broader protocol is ready to execute. The allowed broad run states are:

- `SPECIFICATION_ONLY_BLOCKED` — one or more required inputs are unresolved;
- `READY_NONAUTHORIZING` — the complete manifest has passed Q0, but grants no
  propagation authority;
- `RUNNING_NONAUTHORIZING` — the immutable package is being evaluated;
- `COMPLETE_NONAUTHORIZING` — every planned output has been retained; and
- `INVALIDATED` — execution departed from the immutable package or the blind
  protocol was breached.

The allowed verdicts are:

- `NOT_RUN` — no qualification execution has occurred;
- `BLOCKED` — a prerequisite, source, value, budget, or independent arm is
  missing;
- `INVALID` — a hash, schema, execution, retention, or unblinding rule was
  violated;
- `FAIL` — at least one conjunctive scientific gate failed;
- `CONFLICT` — individually valid arms disagree outside their frozen budgets;
  and
- `ELIGIBLE_FOR_REVIEW_NONAUTHORIZING` — Q0 through Q9 all passed and the
  complete package may be submitted for external scientific review.

There is no weighted score, majority vote, partial pass, ignored outlier, or
automatic `DETECTED` state. A missing, nonfinite, malformed, skipped, or
unbudgeted result blocks or fails the candidate. A conflict cannot be averaged
away.

## Qualification matrix

All ten gates are conjunctive.

| Gate | Discriminating question | Required outcome |
| --- | --- | --- |
| Q0 | Is the exact package complete, immutable, reproducible, and nonauthorizing? | Every required byte, value, convention, threshold, and digest is present and verified before arithmetic. |
| Q1 | Does the implementation evaluate the declared acceleration rather than a nearby or incomplete equation? | Independent arbitrary-precision fixtures, covariance identities, limits, ledger checks, and declared equation mutants all discriminate correctly. |
| Q2 | Does the integrated equation recover the analytic leading 1PN perihelion coefficient? | Paired Newtonian/1PN, step-refined, event-refined measurements converge to the \(6\pi\) coefficient with the correct sign and \(c^{-2}\) scaling. |
| Q3 | Does a separately implemented high-order trajectory oracle reproduce both the orbit and the small relativistic signal? | The oracle self-qualifies, then total-state and differential-signal discrepancies remain inside their separate budgets at every frozen checkpoint. |
| Q4 | Is the restricted model the correct massless limit of the retained-order finite-mass EIH equation? | A complete independent EIH evaluator passes the finite-mass identity and its fixed-total-\(\mu\), \(\nu\rightarrow0\) restricted limit. |
| Q5 | Is the fixed-step midpoint trajectory in its second-order regime, with nonlinear-solve error separated from truncation error? | Oracle-based order, Richardson behavior, solver tightening, and signed-time diagnostics pass on a frozen step grid. |
| Q6 | Are the reported digits stable under an independently frozen Decimal-precision grid? | The crossed step/precision matrix reaches a documented arithmetic plateau without concealing loss of step convergence. |
| Q7 | Are coherent units and TCB/TDB-compatible scalings applied as physical transformations rather than metadata relabeling? | Exact unit and compatible-scaling twins agree after inverse transformation, and deliberately mixed contracts fail closed. |
| Q8 | Do preregistered, independently held cases pass without tuning or refitting? | Every sealed holdout passes on first valid unblinding, including required inside-domain, outside-domain, long-arc, transform, and EIH-limit cases. |
| Q9 | Are all discrepancies covered by a defensible error budget and all resulting words below the claim ceiling? | Every component and total budget passes, all conflicts are resolved by a new preregistration, and emitted claims equal the permitted text. |

### Q0 — manifest, provenance, and preflight

Before any qualification arithmetic, Q0 must verify:

1. canonical parsing, schema validation, exact file sizes, and SHA-256 digests;
2. no unknown keys, implicit defaults, environment-derived values, binary-float
   physical inputs, or unresolved required fields;
3. exact software commits or source archives, dependency locks, interpreter and
   arithmetic-library identities, locale, and platform record;
4. exact model, force order, acceleration semantics, target treatment, source
   roster, units, frame, time scale, epoch, and validity interval;
5. exact case values, applicability limits, numerical grids, observers,
   checkpoints, norms, error allocations, and acceptance thresholds;
6. absence of result files before a new run and an append-only output policy;
7. `MODEL_OUTPUT` and `registry_authorized=false` throughout the planned
   package; and
8. separation of the JX implementation from each claimed independent oracle.

Qualification execution remains blocked while any required field is
`TBD_BLOCKED`. A synthetic development run may still be performed outside this
qualification, but it cannot receive a Q0 pass or be promoted afterward.

### Q1 — independent equation and acceleration qualification

An arbitrary-precision oracle must transcribe the equation independently; it
must not call or translate the JX force kernel. The frozen fixture roster must
include, at minimum:

- a generic three-dimensional state for which every scalar and vector term is
  nonzero;
- purely radial and purely transverse velocity cases;
- multiple radii, speeds, and coordinate orientations;
- proper-rotation and signed-axis transformation twins; and
- cases inside, exactly on, and just outside each declared compactness and
  speed-domain boundary.

The gate compares correction-only components before adding the Newtonian base.
It then verifies the total force ledger contains, in order, exactly
`force.newtonian.point_mass` and
`relativity.solar_schwarzschild_test_particle_1pn`.

Metamorphic checks must include
\(\Delta\mathbf a(\mathbf r,-\mathbf v)=
\Delta\mathbf a(\mathbf r,\mathbf v)\), proper-rotation covariance,
\(c^{-2}\) scaling at fixed state, and Newtonian recovery as
\(c^{-2}\rightarrow0\). Domain failures must occur before a candidate state or
partial result is returned.

The fixture set must also reject every preregistered mutant, including:

- omission or coefficient corruption of
  \(4(\mathbf r\mathbin{\cdot}\mathbf v)\mathbf v\);
- reversal or omission of the \(-v^2\mathbf r\) term;
- an incorrect radial power;
- an incorrect power or unit conversion of \(c\);
- double application of the Newtonian base or relativistic correction; and
- use of absolute rather than simultaneous target-minus-Sun velocity.

If a mutant survives all fixtures, Q1 fails because the matrix has not shown
that it can distinguish the declared equation.

### Q2 — analytic perihelion qualification

For each frozen synthetic case, define

\[
p=a(1-e^2),\qquad
\lambda=\frac{\mu}{pc^2},\qquad
\Delta\varpi_{1\mathrm{PN}}=6\pi\lambda.
\]

The initial state, its Newtonian osculating \(a,e,p\), the eccentricity roster,
the \(\lambda\) or \(c^{-2}\) ladder, orbit count, step grid, event grid, and
all angle-unwrapping rules must be frozen before execution. Circular cases are
not perihelion cases and must not be made well-defined by an implicit
convention.

Every paired schedule must span at least the frozen orbit count times

\[
T_{upper}=\frac{44}{7}\sqrt{\frac{a^3}{\mu}}.
\]

Here (44/7>2\pi) is used only as a conservative machine-checkable coverage
bound. It is not the analytic (6\pi) gate coefficient and is not evidence for
the perihelion result.

Every case uses matched Newtonian and Newtonian-plus-1PN runs from the same
initial state. The measured quantity is their paired secular advance, not the
raw angle from only the relativistic run. A fixed roster of successive
perihelia is used to estimate advance per orbit.

The observer must:

1. bracket an outward perihelion crossing by
   \(\mathbf r\mathbin{\cdot}\mathbf v=0\) with increasing radius derivative,
   excluding the initial event by a frozen rule;
2. locate the event using a separately implemented arbitrary-precision method;
3. define the signed angle about the frozen orbital angular-momentum direction
   with a high-precision oriented `atan2` construction;
4. use a frozen unwrap, orbit-index, and regression rule; and
5. repeat with a tighter event tolerance so event error is measured separately
   from trajectory error.

Binary-float trigonometry and linear event interpolation from the disclosed
development test are forbidden in this gate.

At each eccentricity, step and event extrapolation occur before the
\(\lambda\rightarrow0\) comparison. The gate tests the intercept of

\[
\frac{\Delta\varpi_{1\mathrm{PN\ run}}
      -\Delta\varpi_{Newtonian\ run}}{\lambda}
\]

against \(6\pi\), within the frozen analytic, numerical, and event budget. It
also requires prograde sign under the declared orientation and the expected
linear \(c^{-2}\) asymptote. This construction separates the leading analytic
coefficient from finite-\(\lambda\) terms of order \(\lambda^2\).

A single synthetic orbit, a Mercury headline value such as arcseconds per
century, or improvement under one step halving cannot pass Q2.

### Q3 — independently implemented high-order oracle

The trajectory oracle must be operationally independent of JX:

- separate source and package ownership or an explicitly independent
  implementation review;
- no import, copy, translation, generated binding, or shared helper from the JX
  force, integrator, event, or test code;
- an independently transcribed right-hand side and force ledger;
- a pinned arbitrary-precision arithmetic implementation;
- a pinned adaptive method of order at least eight, including exact method
  identity and coefficient provenance;
- independent error control, dense output or exact-checkpoint policy, and event
  handling; and
- retained source, dependency, runtime, configuration, log, and output bytes.

Before comparison with JX, the oracle must self-converge under both tighter
tolerances and higher precision. Its self-discrepancy must fit wholly inside
the frozen oracle allocation. Failure to self-converge blocks the comparison;
JX cannot pass by agreeing with an unresolved oracle.

At exact frozen epochs, Q3 separately compares:

\[
E_{state}=\max_k\left(
\frac{\|\mathbf r_k^{JX}-\mathbf r_k^{O}\|}{L_k},
\frac{\|\mathbf v_k^{JX}-\mathbf v_k^{O}\|}{V_k}
\right)
\]

and the paired relativistic signal

\[
\delta\mathbf y=\mathbf y_{Newtonian+1PN}-\mathbf y_{Newtonian},
\qquad
E_{signal}=\max_k\|\delta\mathbf y_k^{JX}-
\delta\mathbf y_k^{O}\|_{scaled}.
\]

The length scales \(L_k\), velocity scales \(V_k\), signal floor, norm,
component treatment, and checkpoint roster are immutable inputs. Passing only
the total-state comparison is insufficient because the total orbit can look
accurate while the much smaller relativistic signal is numerically lost.

The in-repository RK4 development test cannot satisfy Q3.

### Q4 — restricted full-EIH limit

This gate begins from an independently implemented complete, simultaneous
finite-mass EIH 1PN evaluator in the declared harmonic-coordinate convention,
with general relativity fixed to \(\beta=\gamma=1\). Any acceleration that
appears inside a term already multiplied by \(c^{-2}\) must be evaluated
Newtonianly or explicitly re-expanded to retained 1PN order. Iterating an
untruncated acceleration there would introduce uncontrolled \(c^{-4}\) terms.

Let the total gravitational parameter \(\mu\) be fixed, let
\(q=\mu_t/\mu_s\), and define

\[
\nu=\frac{q}{(1+q)^2},\qquad
\mathbf n=\frac{\mathbf r}{r},\qquad
\dot r=\mathbf n\mathbin{\cdot}\mathbf v.
\]

Matched barycentric two-body states use

\[
\mathbf x_s=-\frac{q}{1+q}\mathbf r,\qquad
\mathbf x_t=\frac{1}{1+q}\mathbf r,
\]

with the same mass fractions applied to \(\mathbf v\). For a fixed total
\(\mu\), the retained-order relative equation is

\[
\mathbf a_{rel}(\nu)=
-\frac{\mu}{r^2}\mathbf n
+\frac{\mu}{c^2r^2}
\left\{
\left[
(4+2\nu)\frac{\mu}{r}
-(1+3\nu)v^2
+\frac{3}{2}\nu\dot r^2
\right]\mathbf n
+(4-2\nu)\dot r\,\mathbf v
\right\}.
\]

Consequently, at the same relative state,

\[
\mathbf a_{rel}(\nu)-\mathbf a_{restricted}=
\nu\frac{\mu}{c^2r^2}
\left\{
\left[
2\frac{\mu}{r}-3v^2+\frac{3}{2}\dot r^2
\right]\mathbf n
-2\dot r\,\mathbf v
\right\}.
\]

Q4 requires the complete EIH evaluator to reproduce the finite-mass equation
on generic, radial, transverse, rotated, and body-exchanged fixtures. It then
uses a frozen, previously unseen mass-ratio ladder and requires the exact
linear-in-\(\nu\) retained-order identity above, convergence to the restricted
JX acceleration as \(\nu\rightarrow0\), and source backreaction that vanishes
with target mass.

Total \(\mu\) is held fixed so a changing Newtonian parameter cannot masquerade
as mass-ratio convergence. Correction-only quantities are compared before the
Newtonian base is added once.

Passing Q4 would validate only the restricted-limit correspondence. It would
not establish that JX implements or qualifies full EIH dynamics.

### Q5 — step, nonlinear-solver, and signed-time convergence

Each trajectory case uses an exact step-halving grid whose members end at the
same exact coordinate epochs. The theoretical global order is two. The
predeclared asymptotic window, norm, fit, minimum and maximum acceptable order,
Richardson rule, and rejection behavior are qualification inputs; none may be
selected after results are seen.

The primary order estimate uses error against the self-qualified Q3 oracle.
Successive-difference ratios are a secondary Richardson diagnostic, not a
substitute for an external reference. The accepted window must show
second-order behavior in both state and relativistic-signal errors before an
arithmetic plateau is reached.

Nonlinear-solve error is separated by repeating each selected grid point with
tighter residual tolerances and adequate iteration limits. The resulting
trajectory change must remain inside its solver allocation and be negligible
under the frozen solver-to-truncation margin. A local scaled residual of at
most one is necessary but is not, by itself, a global error bound.

Forward/negative-step recovery is retained as a diagnostic. Finite nonlinear
termination and Decimal rounding prohibit claims of exact reversibility.
Nothing in Q5 creates a symplecticity claim for coordinate \((\mathbf r,
\mathbf v)\), which has not been established as a canonical 1PN pair.

### Q6 — Decimal-precision convergence

Q6 crosses the Q5 step grid with a frozen Decimal-precision grid. Every context
binds precision, `ROUND_HALF_EVEN`, exponent limits, clamp and capitals policy,
and the complete trap roster. Position and velocity absolute tolerances remain
dimensioned and explicit.

For each selected step, the highest-precision repetitions must agree within
the arithmetic allocation. Increasing precision must leave the Q5 order
conclusion stable until truncation error dominates. Roundoff need not decrease
monotonically, so the gate tests a frozen plateau bound rather than choosing
the visually smoothest sequence.

Changing precision, step, nonlinear tolerance, and event tolerance all at once
is prohibited because it prevents attribution of the discrepancy.

### Q7 — coherent units and compatible time scaling

For an exact change from numerical unit system A to B, define

\[
s_L=\frac{L_A}{L_B},\qquad s_T=\frac{T_A}{T_B}.
\]

The complete transformation is

\[
\begin{aligned}
\mathbf r_B&=s_L\mathbf r_A,&
t_B&=s_Tt_A,\\
\mathbf v_B&=\frac{s_L}{s_T}\mathbf v_A,&
\mu_B&=\frac{s_L^3}{s_T^2}\mu_A,\\
c_B&=\frac{s_L}{s_T}c_A,&
\mathbf a_B&=\frac{s_L}{s_T^2}\mathbf a_A.
\end{aligned}
\]

Step size and all epoch offsets scale by \(s_T\), position tolerances by
\(s_L\), and velocity tolerances by \(s_L/s_T\). The exact SI second, speed of
light, astronomical unit, day, and kilometre relationships must be obtained
from retained primary sources rather than copied from the disclosed test.

For TCB/TDB-compatible quantities, define \(F=1-L_B\). The bound IAU affine
definition is

\[
\mathrm{TDB}=\mathrm{TCB}
-L_B(\mathrm{JD}_{TCB}-T_0)86400+\mathrm{TDB}_0.
\]

Matched coordinate differences and compatible quantities obey

\[
\Delta t_D=F\Delta t_B,\qquad
\mathbf x_D=F\mathbf x_B,\qquad
\mu_D=F\mu_B,
\]

and therefore

\[
\mathbf v_D=\mathbf v_B,\qquad
c_D=c_B,\qquad
\mathbf a_D=\frac{1}{F}\mathbf a_B.
\]

The transformed step and position absolute tolerance scale by \(F\); velocity
absolute tolerance does not. The affine epoch label, \(T_0\), \(L_B\), and
\(\mathrm{TDB}_0\) must be bound exactly. Corresponding checkpoints must refer
to the same transformed coordinate event, not the same unscaled numeric time.

Q7 compares inverse-transformed trajectories and the paired relativistic
signal. It also requires deliberate mixes of TCB state, TDB-compatible
\(\mu\), unscaled step, mislabeled epoch, and relabeled units to fail closed.

Because the restricted static-Sun equation is autonomous, this scaling test
cannot validate the affine offset dynamically. It does not qualify UTC, UT1,
TT, TCG, TDB, or TCB realization; Earth orientation; leap seconds; moving
source states; or an observational time-scale chain.

### Q8 — held-out qualification

Holdout inputs, expected outputs, tolerances, and classification rules must be
generated or custodied independently of the implementation team. Before a
qualifying run, the canonical bytes and expected-output package are sealed by
hash, with the latter unavailable to the implementation team.

The holdout roster must include previously unseen numerical values for:

- a generic off-axis state with all relativistic terms active;
- multiple eccentricities and a start phase not located at perihelion;
- a long, multi-orbit secular case;
- an inside-domain case near each declared applicability boundary;
- an outside-domain counterpart that must reject before arithmetic;
- exact unit-system and TCB/TDB-compatible twins;
- a signal-scale case testing a small but budget-resolvable 1PN difference;
  and
- a finite-mass EIH mass ratio not used to develop Q4.

Any observational or physical-scale holdout additionally requires frozen
parameters, covariance, frame/time realization, force roster, initial-state
provenance, and omitted-force budget.

There is no refitting, state adjustment, selective checkpoint removal,
tolerance relaxation, or case reclassification after unblinding. A protocol,
code, input, or threshold change after a failed unblinding creates a new
qualification version and requires a new independent holdout. The failed
record remains retained.

### Q9 — error-budget and claim audit

For every scalar, vector, event, and trajectory observable, the preregistration
must provide compatible units, a frozen norm, and the conservative budget

\[
B_{total}=
B_{oracle}+B_{step}+B_{solve}+B_{decimal}+B_{event}
+B_{transform}+B_{parameter}+B_{analytic}+B_{model}.
\]

The terms mean:

- \(B_{oracle}\): independently demonstrated oracle and dense-output error;
- \(B_{step}\): retained midpoint discretization error;
- \(B_{solve}\): nonlinear iteration and accepted-residual contribution;
- \(B_{decimal}\): Decimal rounding and representability contribution;
- \(B_{event}\): perihelion location, angle, unwrap, and regression error;
- \(B_{transform}\): unit and compatible-scale transformation error;
- \(B_{parameter}\): input-state, parameter, uncertainty, and covariance
  propagation;
- \(B_{analytic}\): finite-\(\lambda\) remainder in an analytic comparison;
  and
- \(B_{model}\): omitted 2PN, source-motion, planetary, multipole, spin, and
  other physical-model contributions relevant to the comparison.

The default combination is the conservative sum shown above. Another
combination rule requires a retained derivation establishing the necessary
dependence assumptions. No term may be set to zero merely because it is
inconvenient. A synthetic same-equation fixture may set physical-parameter or
omitted-physics terms to exact zero only when that synthetic scope is explicit
and the expected equation truly contains no such term.

For a pass, the observed discrepancy must be no larger than \(B_{total}\),
every realized component must remain inside its own allocation, and the oracle
uncertainty must be small enough under a frozen margin to discriminate the JX
acceptance threshold. Passing by a large total budget while one component has
overflowed fails Q9.

All numerical allocations, margins, order intervals, signal floors, and
observable tolerances are currently unresolved and therefore block execution.
They must be justified and frozen before results exist.

## Required source and parameter provenance

Discovery links are not qualification provenance. For every normative source,
the package must retain exact local bytes, SHA-256, byte count, retrieval time,
provider, title, version, stable URI, equation or table locator, extraction or
derivation, and independent transcription review. At minimum, the source
manifest must cover:

- IERS Conventions, Technical Note 36, Chapter 10, for BCRS equations and
  reference-system conventions;
- the applicable IAU 2000 resolutions for BCRS and compatible spatial and
  gravitational-parameter scaling;
- IAU 2006 Resolution B3 for the exact TDB affine definition and defining
  constants;
- the IAU astronomical-unit definition and the current SI definitions needed
  for exact unit conversions and \(c\);
- the DE440/DE441 model paper and the applicable JPL equations-of-motion
  source, including the exact retained-order EIH convention;
- a primary finite-mass harmonic-coordinate 1PN source for the Q4 identity;
- primary numerical-method sources for implicit midpoint and the selected
  independent high-order method; and
- the complete independent-oracle and EIH software artifacts.

The physical-input manifest must separately bind:

- Solar \(GM\) in its native coordinate-time scaling, value role, units,
  uncertainty, covariance, epoch, validity, and source solution;
- the derived TDB-compatible Solar \(GM\) and its exact scaling derivation;
- the exact speed of light in each selected coherent numerical system;
- \(L_B\), \(T_0\), and \(\mathrm{TDB}_0\);
- target identity and massless/no-backreaction scope;
- frame origin and axes, simultaneous target-minus-Sun state definition, and
  coordinate epoch;
- maximum \(\mu/(rc^2)\) and \(v^2/c^2\), with a derivation connecting each
  threshold to a bounded weak-field/slow-motion error;
- minimum separation and event-domain policy; and
- covariance and an omitted-force budget appropriate to every physical case.

The current draft registry leaves these values or artifacts unresolved. No
value may be copied from a convenient library, DE file, Horizons response,
paper abstract, or development test without retaining and validating its exact
provenance and compatible scaling.

## Prohibited DE440 and Horizons use

DE440/DE441 and JPL equations are valid primary sources for model definitions,
parameter provenance, and the full EIH comparator when their exact artifacts
and conventions are bound. Their ordinary trajectory output is not, by
default, a restricted Solar 1PN oracle.

DE440 and Horizons trajectories include moving bodies, simultaneous N-body
relativity, fitted initial conditions, and additional forces and conventions
absent from the static-Sun test-particle equation. A raw residual against them
cannot isolate whether JX implemented the restricted correction correctly.
Agreement may reflect fitted or cancelling omitted terms; disagreement may be
caused by physics outside the restricted model.

No DE440, DE441, SPK, or Horizons state may gate this qualification unless the
comparison preregisters and binds:

- an identical force and source roster or a quantified differential
  observable that isolates the tested term;
- identical initial state, epoch, frame, origin, axes, time scale, and units;
- no post-comparison refit or state alignment;
- interpolation and source-ephemeris errors; and
- a complete omitted-term and parameter-covariance budget.

Without those conditions, such a comparison is contextual `MODEL_OUTPUT` only
and must be labeled `NOT_MODEL_MATCHED`; it cannot pass Q3 or Q8 and cannot be
used to claim ephemeris accuracy.

## Current blockers

This preregistration is intentionally not executable. At least the following
remain blocking:

- retained and independently checked normative source artifacts;
- frozen physical constants, compatible scalings, uncertainty, covariance,
  epoch, and validity;
- justified compactness, speed, separation, and omitted-force bounds;
- exact development and holdout input manifests not reusing disclosed cases;
- all numerical grids, norms, checkpoints, event rules, tolerances, margins,
  and error allocations;
- a separate high-order arbitrary-precision oracle and proof of its
  independence and self-convergence;
- a complete independent retained-order EIH evaluator and convention audit;
- a sealed external holdout package and unblinding custodian; and
- independent scientific review of the completed evidence.

Resolving one blocker does not authorize execution while another remains.
The machine-readable blocker codes remain, in order:
`EXECUTION_IMPLEMENTATION_NOT_REGISTERED`, `Q0_Q9_NOT_EXECUTED`,
`PROVENANCE_UNRESOLVED`, `INDEPENDENT_ORACLE_UNRESOLVED`,
`EIH_EVALUATOR_UNRESOLVED`, `ERROR_BUDGETS_UNRESOLVED`, and
`HOLDOUT_CUSTODY_UNRESOLVED`.

## Claim ceiling

Machine claim code:
`FROZEN_RESTRICTED_SOLAR_1PN_REFERENCE_REPRODUCTION_ELIGIBLE_FOR_EXTERNAL_REVIEW_NONAUTHORIZING`.
The machine claim text is exactly the paragraph below after removing Markdown
code formatting; `<qualification_id>` is replaced only by the frozen plan ID.

If and only if Q0 through Q9 all pass, the maximum permitted result is:

> For the frozen cases and declared domain in qualification package
> `<qualification_id>`, the nonauthorizing JX reference path reproduced the
> declared restricted static-Sun, massless-target Schwarzschild 1PN equation
> within the preregistered numerical and model-matched error budgets. The
> resulting trajectories are `MODEL_OUTPUT` and are eligible only for external
> scientific review.

That statement must name the qualification identifier and domain. It must not
be shortened into a broader claim.

This protocol prohibits claims that:

- the V5 registry or force model is authorized for scientific propagation;
- the reference path is a production integrator or observational ephemeris;
- JX implements or validates full EIH relativity;
- JX reproduces DE440, DE441, or Horizons;
- a fixed-initial-state trajectory difference is a fitted residual,
  observation, detection, exclusion, constraint, or model preference;
- relativity alone makes JX physically complete;
- symplecticity or exact reversibility has been proved for this state and
  nonlinear solve; or
- JX is faster, more accurate, or otherwise superior to another system.

External review is required after an eligible result. It cannot retroactively
repair a failed, conflicted, invalid, incomplete, or unblinded qualification.

For holdout freshness, the machine inspector removes target/display labels and
uses two versioned scientific fingerprints. The primary fingerprint preserves
the already frozen prior-development manifest. A secondary transform-invariant
fingerprint converts AU/day and kilometre/second values to one
TDB-compatible kilometre/second representation using an explicit
precision-90, round-half-even operation order and the retained IAU constants.
Renaming a target or presenting an exact declared unit/time transform twin
therefore cannot evade the disclosed-development state check. A repeated
holdout fingerprint is permitted only when every member is an explicit Q7
transform-twin fixture and the full equivalence class is connected by the
frozen undirected transform-pair graph. This permits a declared transitive
unit/time transform chain without making acceptance depend on JSON order; any
disconnected duplicate remains invalid.
