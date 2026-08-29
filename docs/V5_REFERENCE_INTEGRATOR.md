# JX V5 velocity-dependent reference integrator

**State:** `REFERENCE_ONLY_NONAUTHORIZING`

**Evidence class:** `MODEL_OUTPUT`
**Registry authority:** `false`

## Purpose and boundary

This layer tests whether JX can integrate the already implemented restricted
Solar Schwarzschild 1PN equation without misusing its position-only Yoshida
kick. It is a numerical-method and equation-composition experiment, not an
ephemeris, orbit-determination system, full EIH model, or production propagator.
It has no CLI entry point and is not imported by the legacy dynamics,
Yoshida, Bulirsch–Stoer, or independent DOP853 paths.

The draft V5 registry remains nonexecutable. A successful reference trajectory
cannot qualify a force, fill a `TBD_BLOCKED` parameter, validate an astronomical
result, or promote a claim.

## Integrated equation

For the simultaneous target-minus-Sun state

\[
\mathbf y=(\mathbf r,\mathbf v),\qquad
\dot{\mathbf y}=\left(\mathbf v,\mathbf a\right),
\]

the reference ledger constructs exactly

\[
\mathbf a=-\frac{\mu\mathbf r}{r^3}
+\frac{\mu}{c^2r^3}
\left[
\left(\frac{4\mu}{r}-v^2\right)\mathbf r
+4(\mathbf r\cdot\mathbf v)\mathbf v
\right].
\]

The first term is tagged `force.newtonian.point_mass`; the second is tagged
`relativity.solar_schwarzschild_test_particle_1pn` and remains
`CORRECTION_ONLY`. Missing, repeated, reordered, or alternative relativistic
model identifiers are rejected before force arithmetic. The physical scope is
still a static spherical Solar monopole and a massless target with no
backreaction. The relativistic equation follows the Schwarzschild line of
[IERS Conventions, Chapter 10](https://iers-conventions.obspm.fr/content/chapter10/tn36_c10.pdf);
the full Solar-System BCRS model is materially broader.

## Numerical method

JX applies the implicit midpoint rule to all six components at once:

\[
\mathbf y_{n+1}=\mathbf y_n+h\,f\!\left(
t_n+\frac h2,\frac{\mathbf y_n+\mathbf y_{n+1}}2
\right).
\]

It is the one-stage Gauss collocation method and has global order two. The
method is symmetric when the nonlinear equation is solved accurately. Those
properties and the distinction between a general Runge–Kutta state and
canonical Hamiltonian variables are described in
[Hairer's geometric-integration notes](https://unige.ch/~hairer/poly_geoint/week2.pdf)
and the implicit Runge–Kutta foundation in
[Butcher (1964)](https://doi.org/10.1090/S0025-5718-1964-0159424-9).

JX does **not** claim symplecticity here. The stored coordinate velocity is not
established as the canonical 1PN momentum, and a finite nonlinear stopping
rule also prevents a claim of exact reversibility. The canonical-coordinate
scope of symplectic Runge–Kutta claims is treated by
[Sanz-Serna (1988)](https://doi.org/10.1007/BF01954907). The use of implicit
midpoint ideas for astronomical velocity-dependent forces is discussed by
[Mikkola and Merritt (2006)](https://doi.org/10.1111/j.1365-2966.2006.10854.x).

The bounded reference uses deterministic fixed-point iteration from one fresh
explicit-Euler guess. Every candidate is checked by recomputing the midpoint
right-hand side and the full implicit residual. For component \(i\), acceptance
requires the maximum of

\[
\frac{|r_{r,i}|}{a_r+r_{tol}\max(|r_{0,i}|,|r_{1,i}|,|h\dot r_{m,i}|)},
\qquad
\frac{|r_{v,i}|}{a_v+r_{tol}\max(|v_{0,i}|,|v_{1,i}|,|h\dot v_{m,i}|)}
\]

to be at most one. Position and velocity have separate dimensional absolute
tolerances. Nonconvergence, a nonfinite value, a trapped Decimal signal, zero
separation, applicability-bound failure, metadata drift, or a ledger mismatch
raises without returning a trajectory.

## Bound numerical inputs

Together, the force plan, phase states, trajectory specification, and accepted
step diagnostics bind:

- the canonical draft-registry digest and scientific-contract digest;
- the exact Newtonian-plus-Solar force-plan digest;
- coordinate epoch, target, units, frame, time scale, and target treatment;
- Decimal precision, `ROUND_HALF_EVEN`, exponent range, clamp/capital policy,
  and the trapped-signal roster;
- fixed step, step count, dimensional absolute tolerances, relative tolerance,
  iteration limit, method identifier, and initial-guess policy.

The midpoint and final epochs must equal the exact rational offsets
\(t+h/2\) and \(t+h\) under the declared context. A step too small to advance
the coordinate-time value is rejected before it can produce a mislabeled
state.

No binary float enters reference coefficients, force arithmetic, iteration,
residuals, or step control. Binary trigonometry is used only by a test observer
after a completed Decimal trajectory to measure a perihelion angle.

## Validation performed

The checked-in tests establish only the bounded reference behavior:

- exact Newtonian-plus-1PN component and force-ledger closure;
- an independently soluble linear velocity-dependent midpoint fixture;
- second-order convergence for a harmonic oscillator and a Solar arc;
- forward/negative-step recovery at the nonlinear-solver tolerance scale;
- forced nonconvergence and an internal zero-separation failure;
- the \(c\rightarrow\infty\) Newtonian trajectory limit;
- agreement of a step-extrapolated midpoint endpoint with a separately coded
  Decimal RK4 equation and step implementation;
- coherent AU/day and kilometre/second trajectory equivalence;
- proper signed-axis rotation covariance;
- prograde 1PN perihelion advance after step-size Richardson extrapolation;
- unchanged legacy `dynamics.py` and `yoshida6.py` bytes and no legacy imports;
- `MODEL_OUTPUT` and `registry_authorized=false` on every retained state and
  diagnostic.

The perihelion comparison uses

\[
\Delta\varpi=\frac{6\pi\mu}{a(1-e^2)c^2}
\]

as a restricted 1PN validation relation. It is not a fit to observations or an
accuracy claim against a planetary ephemeris.

## Frozen qualification package

The checked-in
[Solar 1PN qualification package](../runs/v5_solar_1pn_qualification/README.md)
separates known development evidence from fresh holdout fixtures and defines
ten conjunctive gates, Q0 through Q9. It retains and hashes the governing IERS,
IAU, BIPM, JPL, and NAIF source artifacts; binds the Decimal contexts, force
plans, coordinate conventions, coefficient derivations, integration schedules,
and fixture records; and predeclares equation, perihelion, independent-oracle,
restricted-EIH, convergence, precision, transform, holdout, and claim-audit
checks.

This is a preregistered input and inspection layer, not a qualification result.
No outcomes have been generated, no holdout runner is registered, and raw
DE440 or Horizons trajectories are explicitly ineligible as a correctness
oracle for the restricted static-Sun equation unless their broader force model
is removed or quantitatively budgeted. A future all-gates pass would remain
`MODEL_OUTPUT` and only `ELIGIBLE_FOR_REVIEW_NONAUTHORIZING`.

## Still blocked

Production use requires, at minimum, frozen physical parameter values and
covariance, retained and hashed source artifacts, explicit validity and epoch,
a physical error budget, a robust solver policy outside the tested contraction
domain, a separately packaged and provenance-bound higher-order production arm,
retained-order invariant tests, TCB/TDB-compatible scaling qualification,
held-out external references, independently implemented qualification, and
external scientific review. The in-test Decimal RK4 oracle is not an external
qualification arm. Full EIH relativity, source motion,
multipoles/spin, observations, estimation, encounters, and collisions remain
separate capabilities.
