# JX Celestial Dynamics Framework 0.4.0a1

Release date: 6 September 2026
Release stage: alpha
Scientific claim state: `SCREENING_ONLY`

## General-purpose engine surface

- Reframed JX as a general celestial-dynamics framework while retaining the
  `jxplanetx` package name, existing command, and Planet X workflows.
- Added the public `jxplanetx.engine` API with immutable state, backend,
  parameter, force-plan, contribution, checkpoint, and result contracts.
- Implemented direct unsoftened Newtonian point-mass gravity, restricted
  static-central Schwarzschild test-particle 1PN correction, and unshadowed
  isotropic cannonball solar-radiation pressure.
- Implemented an explicit 13-stage adaptive Fehlberg RK7(8) trajectory path
  with a hatted order-eight accepted solution, an ordinary order-seven defect,
  normalized maximum per-component error control, exact checkpoint clipping,
  forward/backward propagation, and velocity-dependent-force compatibility.
- Implemented a standalone full-Cartesian adaptive RKF78 encounter segment for
  NumPy/CPU binary64, all-active positive-GM mutual unsoftened Newtonian
  states. It uses authoritative signed local-duration accounting, independent
  endpoint provenance labels, caller-supplied pair floors, exact simultaneous
  local-IVP clearance certificates, pair-relative and GM-centroid defect
  control, Kahan accepted updates, bounded witness/checksum custody, and one
  separately accounted mandatory semantic replay.
- Implemented an experimental, unqualified ordered-Jacobi second-order
  Wisdom--Holman KDK map for NumPy/CPU, fully mutual active positive-GM
  Newtonian hierarchies. It uses fixed integer map nodes, elliptic
  universal-variable Kepler subflows, strict hierarchy/periapse/interaction/
  step/Hill/path guards, and one mandatory deterministic semantic replay.
- Implemented the specific, unqualified whole-system
  `integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1` method. On every
  fixed outer interval it transactionally commits a complete typed far probe
  or discards it and redoes the untouched original full interval through the
  private guarded Cartesian encounter execution. One outer replay recomputes
  all mode decisions, work, states, private-child digests, and accounting;
  public child wrappers and nested child replay are not used.
- Added an ordered 46-capability catalog spanning physics, integrators,
  backends, precision, determinism, sensitivity, orbit determination, and
  measurements: exactly twelve rows are `IMPLEMENTED` and 34 are `DECLARED`.
  The implemented set includes the standalone guarded encounter segment and
  the experimental, unqualified Cartesian KDK and ordered-Jacobi
  Wisdom--Holman maps plus the specific whole-step hybrid; broad event-driven
  encounter switching, generic hybrid integration, regularization, and the
  generic split stay declared.
  Every row is `UNQUALIFIED`; declared rows validate and then refuse execution.
- Kept all outputs at `MODEL_OUTPUT` with registry and qualification authority
  fixed false. Units, frame, origin, epoch, parameter validity, provenance,
  float64 dtype, backend/device identity, and summation grouping are explicit.

## Backend and packaging boundary

- Added a NumPy CPU extra, `engine`, constrained to NumPy 2.x.
- Added mutually exclusive `gpu-cuda12` and `gpu-cuda13` extras using the
  corresponding CuPy 14 `[ctk]` wheel bundles; a compatible driver remains
  required.
- Backend selection is explicit. The API performs no implicit host/device
  transfer and never falls back from a requested GPU to CPU.
- The CuPy code path is present but has not been run on project-owned GPU
  hardware or independently qualified. Version 0.4.0a1 makes no GPU
  correctness, cross-device reproducibility, performance, or speedup claim.
- The RKF78 path is unqualified and deliberately narrow: float64 only, no dense
  output, event location, collision response, close-encounter switching, or
  regularization. All other cataloged integrator families remain declared and
  fail closed, including the generic split, implicit velocity-dependent, and
  hybrid close-encounter rows.
- The standalone encounter screen makes only a retained-floor noncollision
  claim for the exact Newtonian local IVP issuing from each accepted numerical
  node on the certified substep. Failure is uncertified rather than collision
  evidence. It does not locate events or minima, respond to collisions,
  regularize, switch integrators, certify trajectory accuracy, or establish
  global/future physical-trajectory clearance. Exact resources and actual
  force calls are capped and reported separately for primary execution and
  the mandatory replay, plus their public totals.
- The Wisdom--Holman path is CPU/all-active/elliptic/non-encounter only. Its
  symplectic and reversible statements concern formal exact subflows, not
  floating-point execution. Sampled energy behavior is not phase accuracy or
  a general long-term stability claim; registry, qualification, production,
  and superiority authority remain false. The implementation and tests were
  independently written from published equations; no external REBOUND source
  was copied or vendored.
- The specific hybrid is NumPy/CPU binary64, all-active positive-GM mutual
  Newtonian, central-first, and limited to 16 bodies on a fixed signed outer
  lattice. A typed near choice discards the full provisional far candidate and
  redoes the original whole interval. There is no hysteresis, latch, partial
  prefix, grouping, dense output, event location, collision response, or
  regularization. Its exact encounter screen proves only accepted-node local-
  IVP retained-floor noncollision, not global trajectory clearance. The
  complete method is not globally symplectic, formally or exactly reversible,
  globally order-qualified, superior, production-ready, or qualified. Hard
  logical-record caps do not replace external process, timeout, memory, and
  concurrency isolation.

## Step 5 Solar-System preparation

- Added an opt-in, unpublished preparation boundary from isolated
  CSPICE/DE440s geometric-state evidence and retained DE440 GM parameters to an
  exact resolved-Earth/Moon eleven-body Newtonian engine input.
- The boundary performs explicit binary64-to-SI projection, operational-GM
  model-barycenter recentering, primary/replay semantic checks, and construction
  of fresh owned read-only NumPy arrays, an engine snapshot, and one direct
  mutual unsoftened Newtonian force plan.
- This is initial-input preparation only. It runs no trajectory and does not
  claim that the reduced Newtonian model is DE440, authenticate or authorize
  source artifacts, establish continuous custody, qualify the roster for
  Wisdom--Holman, or authorize scientific or production use. Every resulting
  engine object remains unqualified `MODEL_OUTPUT`.

## Challenger V1 and post-V5 precision evidence

- Added Challenger V1, a subprocess orchestrator that retains the native
  reports for locked analytic-binary, weak-hierarchy, and close-scatter smoke
  fixtures. Optional comparison lanes require exact REBOUND 5.1.1; the bundle
  defines no cross-study score, ranking, or engine-equivalence rule.
- Added an equal-mass circular-binary precision probe at fixed steps P/256,
  P/512, and P/1024. Its smoke and full profiles cover one and 100 periods,
  respectively, and record analytic-oracle state/phase errors, sampled
  invariants, complete lane replays, and fixture-specific refinement gates for
  JX KDK and optional exact REBOUND 5.1.1 leapfrog.
- These results are narrow numerical regression evidence. They do not establish
  general or theoretical convergence, continuous-time energy bounds,
  equivalence, REBOUND accuracy or speed superiority, timing comparability,
  long-term stability, scientific qualification, or production fitness. All
  reports remain unauthenticated, unqualified `MODEL_OUTPUT`.

## Compatibility and preserved records

The legacy Decimal/Newtonian core, Yoshida and reference integrators, CLI,
previously frozen V4/V5 assets, and prior scientific result sections are
unchanged. The Step 5, Challenger V1, and post-V5 precision assets are
additive. The 0.3.0 release manifest and checksums remain historical records
and were not rewritten for this alpha.

---

# JX N-Body Engine 0.3.0

Release date: 22 August 2026  
Claim state: `SCREENING_ONLY`

## Independent population replication

- Added an independent Newtonian force and SciPy DOP853 population runner that
  does not import or call REBOUND.
- Added independent Kepler element conversion and recovery, annual
  classification, segmented exact-binary64 checkpoint/restart, source
  stability audits, population comparisons, deterministic paired bootstrap,
  and fail-closed `PASSED`, `CONFLICT`, or `INVALID` verdicts.
- Locked Python, NumPy, SciPy, solver-source, coefficient-table, binary,
  initial-state, population, selection, reference-result, and runner hashes.
- Added an `independent` installation extra pinned to NumPy 2.3.5 and SciPy
  1.17.0.
- Expanded the test suite from 70 to 76 tests.

## Independent scientific record

The release includes a compact record of an independent replication of ten
outcome-blind hash-selected 1,000-tracer blocks from the DE441-backed
100,000-tracer experiment.

The original independent attempt is preserved as `INVALID`. It exceeded only
the active endpoint-position consistency gate: 1.27532×10⁻⁶ AU observed
against 1×10⁻⁶ AU locked. All population comparisons passed. A separate locked
diagnostic confirmed adaptive-resolution dependence; no v1 gate was relaxed or
retroactively changed.

A corrective v2 was registered with the same population, physical model,
statistics, and acceptance thresholds. Only DOP853 resolution changed:

- relative tolerance: 1×10⁻¹³;
- absolute tolerance: 1×10⁻¹⁵;
- maximum step: 0.125 year.

V2 returned `PASSED`:

- 10,000 tracers per arm over 10,000 years;
- 433/10,000 sampled injections in both independent arms;
- the same 433 identities in each corresponding REBOUND arm;
- zero injection-identity disagreement;
- 100% final survival in both arms;
- source-minus-control injection fraction 0.0;
- paired-block 95% bootstrap interval `[0.0, 0.0]`;
- maximum active energy drift 8.35832×10⁻¹³;
- maximum active angular-momentum-vector drift 3.02089×10⁻¹³; and
- maximum active endpoint-position disagreement 2.86934×10⁻⁹ AU.

An independent stored-artifact audit returned `AUDIT_PASSED`. It verified 19
locked files, 20 summaries, 20 independent tracer tables, 20 REBOUND reference
tables, and 100 checkpoint state pairs; reconstructed final orbital elements;
and recomputed every statistic, gate, and verdict.

## Scientific boundary

This result strengthens the numerical robustness of one candidate-9118 screen,
but remains `SCREENING_ONLY`. It does not detect or exclude Planet X, validate
candidate 9118, cover the wider candidate space, or substitute for an observed
TNO population and survey selection function. The v2 solver was selected after
the v1 numerical failure and is transparently labeled as a corrective run, not
the original preregistration.

The next scientific gate is an observed-population model plus an explicit
survey-selection likelihood. A longer-horizon hierarchical experiment should
follow only after that gate is defined.

## 0.2.0 foundation

Version 0.3.0 retains the 0.2.0 deterministic ensemble-validation framework,
strict trajectory registration, distribution metrics, claim-control state
machine, ten-year Horizons/DE441 compatibility record, and locked
100,000-tracer-per-arm REBOUND result. It also retains the earlier provenance
corrections that reject non-standard `NaN` and `Infinity` values.

This remains an engine-focused release. Bulk trajectory/checkpoint archives,
observational data, and candidate-search catalogs are excluded; compact
contracts, manifests, source, audits, and scientific reports are included.
