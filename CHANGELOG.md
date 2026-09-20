# Changelog

All notable package changes are recorded here. Scientific outcomes and their
claim boundaries remain documented in the corresponding locked run reports.

## 0.6.0rc1 — 2026-09-19

### Added

- General Dynamics registry v17 with explicit top-level `SCREENING_ONLY`,
  package-version binding, and preserved capability-specific claim ceilings.
- A create-only release builder that performs two independent clean builds,
  canonicalizes source-archive metadata, and rejects non-identical artifacts.
- A fail-closed CUDA release gate with an exact five-test GPU core roster and
  a create-only machine-readable hardware/runtime report.
- Modern MIT license metadata, the full license text, packaging regressions,
  and independent wheel/source-install smoke gates.

### Changed

- Aligned package, CLI, citation, documentation, and registry identity at
  `0.6.0rc1`.
- Corrected CuPy trajectory documentation to match its actual no-host-hash,
  device-resident ownership contract. No numerical behavior changed.
- Preserved explicitly selected virtual-environment Python paths in the test
  matrix runner instead of resolving them to the base interpreter.

### Boundaries

- The release remains `SCREENING_ONLY`, is not production-ready, and is not a
  unified multiphysics execution qualification.
- Benchmark qualifications remain limited to their exact frozen workloads.
- Current CPU tests pass locally. The current CUDA core gate also passes on the
  exact recorded RTX 5060 Ti/CUDA 13.2/CuPy 14.2.0 runtime: 233 engine tests
  with no skips plus the five GPU-specific tests under optimized Python. This
  does not generalize across devices or authorize performance/scientific claims.
- Registry v16, frozen runs, evidence, archives, and earlier manifests remain
  immutable.

## 0.4.0a2 — 2026-09-12

### Added

- Portable NumPy/CPU dynamics scenarios with canonical binary64 manifests and
  same-state, same-integrator control/candidate comparisons.
- Reproducible portable-scenario equal-binary, narrow CPU/CUDA parity, and GPU
  scientific-ladder packages with offline standard-library verifiers.
- A project-owned CUDA qualification on an NVIDIA GeForce RTX 5060 Ti: the
  frozen package's then-current 228-test engine scope passed in normal and
  optimized Python, the final 231-test checkpoint scope also passes in both
  modes, and a
  one-day reduced eleven-body RKF78 endpoint reproduced bit-for-bit across
  NumPy CPU, repeated CuPy GPU runs, and the preserved CPU reference.
- A descriptive large mutual-Newtonian CUDA scale lane covering 4,096 through
  65,536 bodies, with device residency, finite-output digests, memory guards,
  and sampled hardware telemetry.

### Changed

- Force-plan body identifiers are resolved through a linear lookup map instead
  of repeated tuple scans. A non-timing regression test prevents restoration
  of the quadratic lookup.
- Optional CuPy trajectory residency tests now inspect each checkpoint array
  rather than mistakenly treating the result's array tuples as device arrays.
- Root documentation now separates recorded-machine GPU evidence from general
  GPU, performance, production, and scientific claims.

### Boundaries

- GPU evidence is fixture-, device-, runtime-, dtype-, force-order-, body-order-,
  and tile-size-specific. It is not a portable speedup or general production
  qualification.
- The eleven-body fixture is a reduced Newtonian model initialized from
  DE440-derived data, not a reproduction of the DE440 ephemeris.
- Lunar rotation, fluid-core physics, Planet X, and all observational outcomes
  remain outside this engineering checkpoint and undecided.

## 0.4.0a1 — 2026-09-06

### Added

- Public `jxplanetx.engine` force and trajectory API with explicit state,
  backend, parameter, force-plan, contribution, checkpoint, and result
  contracts.
- Float64 NumPy and optional CuPy array backends with explicit device selection,
  no implicit host/device transfers, and no backend fallback.
- Three executable, unqualified force models: direct unsoftened Newtonian point
  masses, restricted static-central test-particle Schwarzschild 1PN correction,
  and unshadowed isotropic cannonball radiation pressure.
- Explicit adaptive Fehlberg RK7(8) trajectory integration with all controller
  limits exposed, exact clipped checkpoints, and no dense-output interpolation.
- Standalone full-Cartesian adaptive RKF78 Newtonian encounter-segment API for
  exact NumPy/CPU binary64, all-active positive-GM barycentric states. It uses
  caller-supplied pair floors, exact local-duration accounting, pair-relative
  and GM-centroid error control, capped exact local-IVP clearance witnesses,
  transactional rejection, content checksums, and one separately accounted
  mandatory semantic replay.
- Ordered-Jacobi second-order Wisdom--Holman KDK integration for a guarded,
  all-active, positive-GM, hierarchical elliptic Newtonian domain. It uses
  universal-variable Kepler subflows, integer checkpoints, mandatory
  hierarchy/encounter guards, and an independently checked semantic replay.
- Specific whole-system Wisdom--Holman/RKF78 hybrid integration on a fixed
  signed integer outer lattice. Each step transactionally commits one complete
  far probe or discards it and redoes the untouched original full interval
  through the private guarded Cartesian encounter execution. Typed finite
  near reasons are distinct from fatal failures; one outer replay validates
  the complete state, decision, private-child digest, and accounting record.
- Fail-closed 46-capability catalog with exactly twelve `IMPLEMENTED` and 34
  `DECLARED` rows covering the general engine roadmap, including the
  standalone encounter segment and the experimental, unqualified Cartesian
  KDK and ordered-Jacobi Wisdom--Holman maps plus the specific whole-step
  hybrid; broad event-driven encounter switching, generic hybrid integration,
  regularization, and the generic symplectic-split row remain declared.
- `engine`, `gpu-cuda12`, and `gpu-cuda13` optional dependency groups; GPU
  extras use the matching CuPy 14 `[ctk]` toolkit bundle.
- Opt-in full three-force and RKF78 trajectory benchmark that reports elapsed
  time and interactions per second without a comparative speedup claim.
- Opt-in Step 5 Solar-System preparation boundary that verifies isolated
  CSPICE/DE440s geometric-state evidence and the retained DE440 GM artifact,
  resolves Earth and Moon into an exact eleven-body roster, converts to SI,
  recenters at the selected model's operational-GM barycenter, and constructs
  fresh read-only engine state plus a direct mutual Newtonian force plan.
- Challenger V1 subprocess orchestrator for three locked smoke fixtures:
  analytic binary, weak hierarchy, and close scatter. It preserves each JX
  lane's native report and can run optional exact REBOUND 5.1.1 comparison
  lanes without adding a cross-study score or ranking.
- Additive post-V5 precision probe for the analytic equal-mass circular binary.
  It records JX KDK and optional exact REBOUND 5.1.1 leapfrog results at
  P/256, P/512, and P/1024 over one- and 100-period profiles, with a specified
  analytic oracle, replay, state/phase refinement, and sampled-invariant gates.

### Changed

- Project positioning now describes JX as a general celestial-dynamics
  framework, with outer-Solar-System and Planet X research as applications.
- Package version advanced to the 0.4.0 alpha series. The distribution name
  remains `jxplanetx` for compatibility.

### Boundaries

- All new engine capabilities remain `UNQUALIFIED` and emit `MODEL_OUTPUT`.
- The RKF78 trajectory path remains float64, explicit, nonstiff, and
  unqualified, with no dense output, event handling, collision response,
  close-encounter switching, or regularization.
- The encounter certificate is only a retained-floor noncollision proof for
  the exact Newtonian local IVP issuing from each accepted numerical node on
  one substep. It is not event/collision detection, minimum-distance
  localization, collision response, trajectory-accuracy evidence, global or
  future clearance, hybrid switching, or a qualification claim. Its endpoint
  epoch is an independent provenance label; signed duration and local offsets
  are authoritative for state advance. Primary and mandatory replay resources
  are capped and reported separately and in public totals.
- The Wisdom--Holman path remains NumPy/CPU-only, all-active, elliptic, and
  non-encounter; its exact-subflow symplectic statement is formal, not a
  floating-point, phase-accuracy, long-term-boundedness, or qualification
  claim. Its public-call accounting includes one mandatory full replay.
- The specific hybrid is NumPy/CPU binary64, all-active, positive-GM, fully
  mutual unsoftened Newtonian, central-first, and limited to 16 bodies. The
  outer lattice is fixed; a near decision discards and redoes the entire
  original interval, with no latch, hysteresis, grouping, partial prefix, or
  event location. Its encounter proof retains only accepted-node local-IVP
  scope. It is not globally symplectic, formally or exactly reversible,
  collision-detecting or responding, regularized, globally order-qualified,
  superior, or qualified. Per-lane record/substep/digest ceilings do not
  guarantee wall time, memory use, concurrency safety, or denial-of-service
  resistance; external isolation and resource limits remain required.
- Step 5 prepares one reduced eleven-body Newtonian initial state and force
  plan only. It does not execute a trajectory, reproduce the DE440 ephemeris,
  authorize its parameter sources, preserve continuous custody, or qualify the
  resolved model for Wisdom--Holman, production, or scientific inference.
- Challenger V1 is a short-fixture smoke protocol. The post-V5 probe supports
  only fixture-specific empirical state/phase convergence on its declared
  equal-binary grids and sampled invariant measurements. Both remain
  unauthenticated, unqualified `MODEL_OUTPUT`; neither establishes general
  convergence, equivalence, accuracy or speed superiority over REBOUND, timing
  comparability, long-horizon behavior, or production fitness.
- The optional CuPy path has not been validated on project-owned GPU hardware;
  no GPU correctness, cross-device reproducibility, performance, or speedup
  claim is made.
- The legacy core, CLI, previously frozen V4/V5 assets, prior scientific
  results, and the 0.3.0 release manifest/checksums are unchanged; the Step 5,
  Challenger V1, and post-V5 precision assets are additive.

## 0.3.0 — 2026-08-22

- Added the independently implemented SciPy DOP853 population-replication path,
  checkpoint and provenance controls, corrective v2 numerical result, and
  stored-artifact audit.
- Retained the `SCREENING_ONLY` scientific claim ceiling.

## 0.2.0

- Established deterministic ensemble validation, claim control, locked
  trajectory registration, and the DE441/Horizons compatibility record.
