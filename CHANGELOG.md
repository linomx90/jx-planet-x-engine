# Changelog

All notable package changes are recorded here. Scientific outcomes and their
claim boundaries remain documented in the corresponding locked run reports.

## Unreleased

### Added

- Added the screening-only
  `orbit_determination.multi_arc_linearized_weighted_svd_step` component with
  shared global parameters, disjoint arc-local parameters, multiple full-
  covariance observation blocks per arc, optional global-only Gaussian prior,
  full-rank/condition gates, per-block diagnostics, immutable results, and
  provenance-bound digests.
- Added a lunar sensitivity adapter that enforces physical force parameters as
  global and GM-reacted initial-state parameters as arc-local. No saved lunar
  scientific outcome or release version was promoted.

- Added a reacting, multi-target solar-J2 correction using the retained DE440
  `ASUN` and `J2SUN` defaults, an explicit J2000 pole, GM-weighted reaction on
  the resolved Sun, and a provenance-bound NumPy force-ABI adapter.
- Added a supported, provenance-bound NumPy/CPU weighted-SVD component for one
  linearized estimation step. It owns its inputs and outputs, supports optional
  positive-definite Gaussian priors, reports covariance/residual/rank/condition
  diagnostics (including separate observation and augmented ranks), binds
  results to SHA-256 content digests, and fails closed on
  rank deficiency or a caller-defined condition gate.
- Added a lunar-ephemeris-v1 sensitivity component that applies GM-reacted
  pair-relative initial position/velocity perturbations, reruns the identical
  public force roster at two symmetric scales, rejects representation,
  asymmetry, or derivative-agreement failures, and emits a Richardson design
  matrix that binds directly into the weighted-SVD fit component.
- Extended that sensitivity component with synchronized Sun, Earth, and Moon
  GM plus Earth-J2 effective-force parameters. Added a reproducible split-mode
  multi-epoch fit/holdout runner and its source-bound day-2718 protocol.

### Screening result

- A nonauthoritative matched 90-day probe found that adding solar J2 alone to
  the rc16 coupled lunar equations worsened all four DE440 endpoint metrics;
  the Sun--Earth position and velocity errors increased by 12.35% and 13.17%.
  Solar J2 is therefore not promoted into the lunar ephemeris component.
- The locked day-2718 effective-force experiment passed its fresh screening
  holdout: all eight Earth--Moon and Sun--Earth position/velocity metrics
  improved at 90 and 365 days, and aggregate normalized error improved by
  25.55%. DE440 is the fitted reference, so this is not independent validation
  or production-ephemeris qualification.
- A subsequently locked three-epoch replication failed: only start day 3037
  passed, the combined ratio was 1.2651, and 20 of 24 individual metrics
  improved. A four-epoch external INPOP21a screen also failed its strict
  all-metrics gate, although aggregate error improved by 15.44% and all 16
  Earth--Moon metrics improved. The inconsistent Sun--Earth response rejects
  the fitted effective-force deltas as stable global parameters.

### Boundary

- The solar-J2 kernel is a general screening-only engine component, not a new
  ephemeris version. A fitted coupled model is required before another native
  ephemeris promotion attempt.
- The weighted-SVD step consumes a caller-supplied design matrix. It does not
  yet supply force-consistent variational equations, measurement modeling,
  nonlinear iteration, outlier handling, or production orbit determination.
- The new sensitivity path is finite-difference and CPU-only. Its GM/J2 deltas
  are effective bias absorbers, not physical-constant measurements. It does
  not implement general analytic variational equations or authorize a fitted
  production ephemeris.

## 0.6.0rc16 — 2026-09-24

### Added

- Added force ABI v1: one public exact-type registry for canonical force
  ordering, backend support, integrator-class compatibility, and acceleration
  semantics.
- Promoted the retained Earth J2--J5 and static lunar degree-two and
  degree-three pair kernels into provenance-bound `ForcePlan` components used
  by public force evaluation, adaptive RKF78, and `JXSimulation`.
- Added explicit NumPy/CPU continuation with parent/result/endpoint state
  digests and deterministic, non-pickle restart archives that require an exact
  external archive SHA-256 and caller-supplied force-plan digest on load.
- Added coupled state ABI v1 and a public engine adapter over the retained v3
  simultaneous Sun--Earth--Moon translation, lunar mantle attitude/rate,
  delayed deformation, and fluid-core-rate integrator. The common result owns
  typed checkpoints and binds all five state blocks to a content SHA-256.
- Added the public resolved-eleven lunar ephemeris v1 component. It advances
  Newtonian translation, mutual EIH 1PN, reacting static lunar quadrupoles,
  fixed-axis Earth J2, lunar mantle attitude/rate, and fluid-core rate through
  one fixed-step RKF78 checkpoint contract with deterministic accounting.
- Added an outcome-blind, create-only 90/365-day DE440 screening protocol for
  the new public component. Its result remains screening evidence and cannot
  authorize model promotion or production use.

### Verification

- The frozen 90/365-day screen passed every gate. At 365 days the public EIH
  component measured 0.061568 km Earth--Moon and 0.029261 km Sun--Earth
  position error, versus 3.729707 km and 56.094548 km for the matched no-EIH
  control. The aggregate candidate/control RMS ratio was 0.012199 and the
  worst individual ratio was 0.018686.
- The run accepted exactly 11,680 steps and performed 151,840 RKF78 force
  evaluations. The maximum quaternion norm error before projection was one
  binary64 epsilon.

### Boundaries

- The new physical-harmonic adapters and restart archives are NumPy/CPU-only;
  the harmonic terms require metre-second J2000
  pair corrections. Specialized GR15 and CUDA routes reject them without
  fallback. The coupled lunar route remains a retained native physics bundle,
  not force ABI v1, and has no CUDA, coupled archive, geodetic transport,
  event, ephemeris, or qualification claim. Events, fitting, independent
  ephemeris validation, and production qualification remain open.

## 0.6.0rc15 — 2026-09-24

### Added

- Added `integrate_verified_gr15_eih_1pn`, an operational CPU propagation
  boundary requiring explicit J2000/TDB/unit conventions, unique body
  identities, source-provenance hashes, distinct primary/confirmation
  numerical contracts, and full-trajectory position/velocity agreement gates.
- Added immutable context, gate, verified-result, named contract-error, and
  named verification-failure types plus stable input and verification digests.

### Verification

- The prospective locked portfolio passed forward and backward finite-mass
  binaries, an eccentric ten-period binary, and the DE440-derived eleven-body
  state for ten Julian years. Every case repeated exactly and preserved the
  existing public primary trajectory bit-for-bit.
- The ten-year eleven-body primary/confirmation maxima were 0.002441 km and
  6.913e-9 km/s, inside the locked 1 km and 1e-7 km/s gates.
- Reran the external REBOUND 5.1.1/REBOUNDx 5.1.0 equation and 100-year
  Sun--Earth tests plus the 10/25/50/100-year DE440 attribution gates.

### Boundaries

- Confirmation uses a second execution of the same packaged implementation;
  it is a per-request consistency gate, not independent validation.
- Results remain `SCREENING_ONLY`. Passing does not authorize navigation,
  production ephemerides, DE440 equivalence, or exact-general-relativity use.

## 0.6.0rc14 — 2026-09-23

### Added

- Added `integrate_gr15_eih_1pn_cpu_batch`, a supported NumPy batch boundary
  that runs independent native GR15-EIH1PN systems concurrently with threads.
  The native extension releases the GIL for each complete integration, so the
  component requires no subprocess lifecycle or state serialization.
- Added `integrate_gr15_eih_1pn_batch`, which returns one owned, read-only
  NumPy result contract for CPU and CUDA execution and records its backend
  decision, calibration identity, worker/kernel accounting, and transfer time.
- Added an explicit dispatch-policy type and a single machine-bound crossover
  profile for the Ryzen 5 5500 and RTX 5060 Ti ten-year, eleven-body workload.

### Verification

- The locked ten-year crossover passed at 1, 2, 4, 8, 16, 32, 64, and 128
  independent systems. Eight-worker JX CPU was 2.17--2.30x faster than
  eight-worker REBOUND IAS15 plus REBOUNDx `gr_full` at every tested count.
- In the final supported NumPy API, CUDA was 3.8% faster than eight-thread JX
  CPU at 96 systems and 9.9% faster at 128, while CPU was 17.5% faster at 64.
  The automatic policy retains a conservative 128-system CUDA cutoff. All
  accuracy, automatic-route identity, and CPU/CUDA agreement gates passed.
- Added CPU batch parity, determinism, backward, per-system-GM, routing, claim
  ceiling, contract, and fail-closed tests plus a real-device CUDA parity test.

### Boundaries

- Automatic CUDA routing is a performance policy, not a scientific claim. It
  applies only to its exact registered hardware/workload scope. Unmatched
  cases stay on CPU, while explicit CUDA requests remain available.
- System counts above the largest measured 128-system point are labelled as
  threshold extrapolations. The dispatcher never claims portable crossover
  behavior or general superiority over REBOUND.
- The physical model remains screening-only mutual point-mass Newtonian plus
  EIH 1PN and is not a production ephemeris or navigation system.

## 0.6.0rc13 — 2026-09-23

### Added

- Added the supported `integrate_gr15_eih_1pn_cuda_batch` component. One
  persistent CUDA block per independent system now executes the complete GR15
  predictor, eight-stage corrector, convergence gate, error estimate,
  rejection controller, exact checkpoint schedule, and adaptive step control.
- Added device-resident batched trajectory outputs, shared or per-system GM,
  host-audited controller accounting, named fail-closed statuses, exact CUDA
  source/runtime identity, and a reproducible CPU/CUDA trajectory benchmark.

### Verification

- Passed exact binary CPU/CUDA checkpoint and controller-accounting parity,
  multi-system and per-system-GM parity, backward integration, bitwise CUDA
  replay, the 32-body boundary, and input, singularity, weak-field, step-limit,
  and rejection-limit failure gates on an RTX 5060 Ti.
- Passed all 220 isolated live CPU, optional-CUDA, and CUDA-hardware lanes;
  reproduced the wheel and source archive across four clean builds; and passed
  fresh installed-artifact CPU/CUDA smoke plus Twine metadata validation.
- In the bounded 11-body trajectory screen, one and eight systems remained
  CPU-faster. CUDA was 3.46x the sequential CPU throughput at 64 systems and
  6.19x at 512 systems; maximum checkpoint differences were 3.56e-15 km and
  1.18e-17 km/s.

### Boundaries

- CUDA accelerates independent trajectory ensembles; it is not the preferred
  route for one small system, and the recorded CPU comparison is sequential.
- The EIH model still omits extended-body figures, tides, spins, frame
  dragging, collisions, signal propagation, and observation reduction.
- Results remain `SCREENING_ONLY`; no production-ephemeris, exact-GR,
  navigation, portable speed, or general REBOUND-superiority claim is made.

## 0.6.0rc12 — 2026-09-23

### Added

- Added the supported `evaluate_eih_1pn_total_acceleration_cuda` component for
  device-resident float64 batches of 2--32 mutual finite-mass bodies.
- Added `evaluate_gr15_eih_1pn_stages_cuda`, which evaluates all eight GR15
  corrector-stage forces for one or more systems in a single CUDA launch.
- Added per-lane weak-field diagnostics, exact runtime/source identity, shared
  and lane-specific GM support, named fail-closed statuses, and a reproducible
  finite-host CPU/CUDA crossover benchmark.

### Verification

- Passed strict CUDA parity against both the package-owned native C core and
  the independent Python equation, including the 32-body boundary and a full
  eight-stage/two-system GR15 sweep.
- Passed bitwise repeated-launch determinism, input ownership, source-identity,
  and input, nonfinite, singularity, and weak-field failure-domain tests on an
  NVIDIA GeForce RTX 5060 Ti with CUDA runtime/driver 13.2 and CuPy 14.2.0.
- In the bounded component screen, CUDA was 5.28x the current native wrapper
  throughput at 4,096 eleven-body lanes and 19.74x at 4,096 thirty-two-body
  lanes. It remained slower for one eight-stage eleven-body system.

### Boundaries

- This is a supported force/stage component, not a complete CUDA GR15
  trajectory integrator. Prediction, corrector updates, convergence, error
  estimation, rejection, checkpointing, and adaptive control remain CPU-only.
- Timings exclude input transfer, output transfer, allocation, and the complete
  GR15 controller. They are one-host engineering observations, not portable or
  general superiority claims.
- Results remain `SCREENING_ONLY`; no exact-GR, production-ephemeris,
  navigation, DE440-equivalence, or general REBOUND claim is made.

## 0.6.0rc11 — 2026-09-23

### Added

- Added the packaged `integrate_gr15_eih_1pn` CPU component for 2--32
  finite-mass bodies under mutual Newtonian plus EIH 1PN point-mass gravity.
- Added native source identity, reusable workspaces, replay digests,
  weak-field fail-closed gates, and explicit screening-only claim controls.
- Added exact-version external comparison lanes for REBOUND 5.1.1, REBOUNDx
  5.1.0, and SpiceyPy 8.2.0.

### Verification

- Preserved the rc10 Newtonian GR15 V3 core byte-for-byte.
- Passed native/Python force parity, Newtonian-limit identity, permutation,
  replay, failure-domain, analytic periapsis, reversal, and convergence gates.
- Passed short same-model IAS15 parity, the REBOUNDx asymptotic comparison,
  and a physical Sun--Earth 100-year same-model IAS15 gate.
- Passed body-resolved DE440 attribution at 10, 25, 50, and 100 Julian years;
  the aggregate residual ratios to the Newtonian control were 0.0882, 0.1282,
  0.0987, and 0.1545 respectively.

### Boundaries

- The new component is CPU-only mutual point-mass EIH 1PN. It does not include
  figures, tides, spins, frame dragging, collisions, signal propagation, or
  observation reduction.
- Results remain `SCREENING_ONLY`; no exact-GR, DE440-equivalence, production,
  navigation, or general-superiority claim is made.

## 0.6.0rc10 — 2026-09-23

### Fixed

- Canonicalized release-stage directory and file modes before wheel and source
  construction. Wheel ZIP metadata is now independent of checkout group-write
  bits and umask policy.
- Rebuilt the unchanged GR15 V3 package under a new candidate version rather
  than replacing the already-qualified, unpublished rc9 artifact identity.

### Verification

- Reproduced the exact wheel and source archive across the active tree and a
  clean GitHub-main overlay.
- Passed the exact-wheel IAS15 and five-case adversarial portfolios twice on
  the desktop and once on the RTX 4050 laptop.
- Passed the five-test CUDA core gate on both the RTX 5060 Ti and RTX 4050.
- Produced a byte-reproducible `manylinux_2_31_x86_64` PyPI wheel with strict
  Auditwheel/Twine and fresh-install checks.

### Boundaries

- Numerical GR15 behavior is unchanged from rc9. All results remain
  `SCREENING_ONLY`; no production, ephemeris, lunar, general GPU, or general
  REBOUND-superiority claim is made.

## 0.6.0rc9 — 2026-09-23

### Added

- Promoted the validated Gauss--Radau order-15 V2 numerical core from a
  benchmark-private experiment into the packaged `jxplanetx.gr15` CPU
  interface. The wheel now carries the native extension; normal execution no
  longer compiles temporary C code or imports the benchmark tree.
- Added a stable specification, reusable workspace, immutable successful
  result, named failure statuses, native build identity, tableau export,
  deterministic replay digest, public documentation, and installed-component
  acceptance tests.
- Added a prospectively locked public-GR15/REBOUND-5.1.1-IAS15 comparison
  using closed-form two-body truth, method-specific tolerance selection from a
  common fixed grid, shared accuracy/conservation gates, balanced timing, and
  separate endpoint-only and retained-output workloads.
- Added and promoted the preregistered GR15 V3 native corrector. V3 uses a
  tolerance-aware stopping threshold and residual-gated terminal-force reuse,
  reports both paths explicitly, and retains the exact V2 package path as a
  bitwise-tested private reference.
- Added a sealed correctness-only V3 adversarial portfolio spanning analytic
  high-eccentricity and extreme-mass-ratio binaries, a retained three-body
  close scatter, a long 11-body hierarchy, and the supported 32-body boundary.
  It uses strict IAS15 primary/sensitivity references and requires replay,
  conservation, reverse-time, residual, accounting, and failure-domain gates.

### Development results

- Both methods selected epsilon `1e-6` and passed every shared analytic gate.
  On this one CPU host, IAS15 was 1.540x faster for the 100-period endpoint
  workload, 1.293x faster with 101 retained outputs, and 1.657x faster for the
  eccentricity-0.9 workload with 17 outputs. These are bounded screening
  results, not a general ranking.
- V3 passed all three sealed holdouts, reduced V2 force evaluations by
  24.6%--26.7%, and was 1.234x--1.292x faster than V2. In the subsequent
  direct IAS15 rerun, the 101-output row was an unresolved tie (JX throughput
  1.016x, below the 5% resolution threshold); IAS15 remained 1.189x faster
  endpoint-only and 1.393x faster on the eccentric row. The eccentric
  rejection count remained unchanged at 20.
- Every sealed adversarial gate passed, and the complete report repeated
  byte-for-byte. The largest scaled JX/IAS15 state difference was 1.36e-8 for
  the eccentricity-0.99 velocity trajectory; the 32-body row was within
  4.19e-13. These are finite synthetic correctness screens, not production or
  general-accuracy qualification.

### Boundaries

- GR15 remains `SCREENING_ONLY` and supports only 2--32 positive-GM, fully
  mutual, unsoftened Newtonian point masses in binary64. Promotion changes the
  software support boundary, not scientific or production authorization.
- The exact local rc9 wheel/source pair, installed-wheel qualification,
  installed-source smoke, normal/optimized engine matrices, 191-lane current
  matrix, four-lane REBOUND profile, and five-test CUDA core gate pass. No push
  or publication was performed; modified source must never be republished as
  rc8.

## 0.6.0rc8 — 2026-09-22

### Added

- Added a deterministic, non-destructive experiment catalog spanning the 99
  immutable authority packages, active run/result records, benchmark
  definitions, linked archives, family lineages, and explicit metadata gaps.
- Added source-bound equal-scope raw-C JX/TRACE reports and the preregistered
  stateful-NEAR RKF78 development candidate.
- Added exact runtime profiles for the local performance evidence and library
  catalog, restoring complete non-overlapping ownership of all 258 test files.

### Results and boundaries

- Stateful NEAR controller continuity reduced attempted adaptive substeps from
  192 to 66 on the close-scatter fixture and from 354 to 264 on the pericenter
  fixture. Its internal parent-over-candidate speedups were 1.524x and 1.246x.
- In the equal-output raw-C screen, the candidate remained 2.158x, 1.415x, and
  2.412x slower than TRACE on the all-far, close-scatter, and pericenter
  fixtures. The comparison is not accuracy-matched and establishes no general
  ranking.
- Endpoint synchronization preserved exact output but achieved only a 1.005x
  speedup, below its preregistered 1.02x threshold, so it is retained as a
  negative result.
- The exact rc8 wheel passed the locked 100-period accuracy, conservation,
  determinism, public-parity, and solver-prerequisite gates. Its supported
  single-pass median was 9.128 ms versus 10.952 ms for REBOUND WHFast, or
  1.200x JX throughput; REBOUND remained 1.684x faster than the replay lane.
- The supported rc7 numerical APIs and General Dynamics registry v18 remain
  unchanged. Stateful NEAR remains an unpromoted development prototype;
  `SCREENING_ONLY`, production, lunar, and general solver claim ceilings do not
  change.

## 0.6.0rc7 — 2026-09-22

### Added

- Advanced the supported `jxplanetx.fast_wisdom_holman` CPU screen to the
  checkpoint-synchronized v3 native map. The default single execution now
  returns a complete-map postcondition certificate; exact full replay remains
  explicitly available and is required during release qualification.
- Added the private endpoint-seeded Kepler v4 prototype, exact-v2 fallback
  accounting, and a source-bound 100-period JX/REBOUND WHFast v2 development
  race that fails closed if any optimized v4 solve uses the fallback.

### Results and boundaries

- The locked v4 race used zero fallbacks, reduced universal-G evaluations from
  70,211 to 53,014, and was 5.23% faster than v3 for the recorded numerical
  scope. Its equal-output result was within 1.99% of tuned REBOUND WHFast,
  below the predeclared five-percent resolution and therefore an unresolved
  tie. Endpoint-only WHFast remained 4.42x faster with unequal output cadence.
- V4 remains private and `SCREENING_ONLY`; it is not the supported rc7 backend.
  No portable performance, scientific qualification, production fitness, or
  general superiority over REBOUND is claimed. General Dynamics registry v18,
  frozen runs, prior artifacts, and earlier manifests remain unchanged.
- The exact reproducible rc7 wheel passed public-v3 bitwise parity, exact
  replay, deterministic-output, universal-G, IAS15-envelope, and conservation
  gates. Its single-pass end-to-end median was 9.142 ms versus 10.037 ms for
  REBOUND WHFast on the locked weak-hierarchy workload; the replay lane was
  18.222 ms. These are single-host, single-workload results only.
- The current CUDA core gate passed all five non-skippable backend/trajectory
  tests with zero skips on the RTX 5060 Ti. This is current-host core
  validation, not a new scaling or general GPU qualification result.

## 0.6.0rc6 — 2026-09-22

### Added

- Added the opt-in supported `jxplanetx.fast_wisdom_holman` CPU screening API.
  It runs the complete guarded fixed-step map in compiled C, retains read-only
  input/output custody and deterministic replay, and preserves the rc5 solver
  as its exact universal-G fallback.
- Added the source-locked v2 100-period JX/REBOUND WHFast race, dense universal-
  G equivalence and failure-domain gates, conservation gates, and nine balanced
  timing repetitions.

### Results and boundaries

- The fixed 64-term universal-G series dominated 93.92% of the v1 native
  profile. Binary64 no-change termination reduced the 100-period series count
  from 17,974,016 to 1,589,999 while remaining bitwise equal to v1.
- In the recorded single-host weak-hierarchy workload, v2 raw end-to-end median
  latency was 9.358 ms versus 11.013 ms for REBOUND WHFast, or 1.177x WHFast
  throughput. The mandatory-replay lane was 18.748 ms, so WHFast was 1.702x
  faster than that safer lane. This is `SCREENING_ONLY`, workload-specific
  evidence—not a portable or general REBOUND superiority result.
- The rc5 release, manifests, and source-bound timing reports remain immutable;
  the changed package is identified only as rc6.

## 0.6.0rc5 — 2026-09-21

### Changed

- Reduced supported small-N CPU wrapper latency by freezing workspaces and
  caching immutable RKF78 tableau, checkpoint-epoch, and read-only checkpoint-
  view metadata. The audited native C numerical core is unchanged.
- Converted the accuracy-matched hybrid benchmark from the historical CPU
  prototype to the supported `jxplanetx.smalln` API and added explicit machine
  identity to the crossover report.
- Revised the measured desktop route so CPU handles eight-lane workloads for
  bodies 2--31; CUDA remains selected for 32 bodies at eight lanes and for all
  supported body counts at 16 or more lanes.

### Results and boundaries

- The installed-wheel public CPU path passed the shared `1e-10` accuracy gate
  and had lower one-system latency than exact REBOUND 5.1.1 IAS15 for the
  sampled 2-, 11-, and 32-body synthetic workloads on the recorded desktop.
- The exact final wheel passed the five-repetition, 256-system fused-CUDA
  matrix against eight REBOUND IAS15 process workers. JX throughput was 2.01x,
  3.56x, and 3.60x the eight-worker integration critical path for the sampled
  2-, 11-, and 32-body rows. Sequential REBOUND retained lower one-lane CUDA
  latency, and no portable or general speed ranking is claimed.
- The rc5 10/30/100-year eleven-body mutual-Newtonian point-mass screen passed
  against REBOUND IAS15, reaching a 1.911314 m maximum body-position difference
  at 100 years. Timing is explicitly non-comparable, and this is not a coupled
  lunar-physics, ephemeris, or long-term-stability qualification.
- The rc5 Challenger v2 full portfolio passed the Leapfrog equal-binary,
  WHFast/IAS15 weak-hierarchy, and MERCURIUS/TRACE/IAS15 close-scatter screens,
  including their limitation witnesses. It authorizes no cross-study timing
  comparison or solver-superiority claim.
- Added a source- and input-bound prospective long single-trajectory timing
  protocol with common accuracy gates, equal host-output cadence, three fixed
  repetitions, one CPU thread, and a predeclared five-percent classification
  rule. The stopped v1 preflight produced no measurements; a v2 overlay binds
  its field-name-only correction without rewriting v1.
- All long-race accuracy gates passed. On the recorded host, REBOUND IAS15,
  WHFast, MERCURIUS, and TRACE were respectively 23.62x, 3,331.94x, 2,650.18x,
  and 18,819.64x faster than the matching current JX implementations. JX public
  timings include mandatory semantic replay. This is workload-specific
  screening evidence, not a portable or general solver ranking, and it does
  not negate JX's separate short independent-system CUDA throughput result.
- The full 2--32-body, 1--256-lane crossover passed its accuracy gates. Its
  noise-sensitive five-repetition rows disagreed with focused 29--31-body
  results; a 101-repetition boundary audit selected CPU for bodies 28--31 and
  retained the 32-body/eight-lane CUDA choice by only a 1.27% median margin.
  Both reports are retained, and no portable threshold is claimed.
- All results remain `SCREENING_ONLY`. No general REBOUND superiority,
  production fitness, scientific qualification, or portable routing claim is
  made, and General Dynamics registry v18 remains unchanged.

## 0.6.0rc4 — 2026-09-20

### Added

- Added the supported `jxplanetx.smalln` screening interface for measured
  2--32-body fully mutual Newtonian workloads. It dispatches low-lane NumPy
  inputs to the packaged compiled CPU core and measured batched CuPy inputs to
  the fused CUDA RKF78 core without implicit host/device transfer.
- Added exact public route, workspace, checkpoint, accepted-step-ledger, and
  native-build identity contracts while retaining `SCREENING_ONLY`,
  `production_authorized: false`, and
  `scientific_qualification_claimed: false`.
- Added a deterministic rc4 second-machine kit for repeating the locked v8
  accuracy-matched JX/REBOUND matrix on the separately available RTX 4050.

### Results and boundaries

- The rc4 v8 matrix passed on both the RTX 5060 Ti and RTX 4050 Laptop GPU
  against exact REBOUND 5.1.1; the fail-closed two-machine aggregate passed.
- The first remote-kit attempt exposed a missing benchmark import before any
  timing began. The deterministic kit now binds that complete import closure,
  passes isolated local and remote import checks, and builds byte-identically.
- REBOUND IAS15 remains faster for one individual system. JX is faster only
  for the tested 256-independent-system throughput rows on this host, including
  the measured eight-worker REBOUND comparison.
- Registry v18 remains immutable historical evidence bound to `0.6.0rc3` and
  is carried by rc4 without a capability or lunar-claim promotion.
- Both tested machines have common ownership, so the result expands hardware
  coverage but is not an independent replication. No general REBOUND
  superiority is claimed.

## 0.6.0rc3 — 2026-09-20

### Added

- Added an opt-in complete delayed tidal-plus-spin mantle-deformation package
  with analytic inertia derivatives and common-potential tensor force/torque.
- Added source-fixed lunar geodetic-precession transport in the mantle and
  fluid-core relative-frame gyroscopic terms.
- Added preregistered 90/365-day deformation and geodetic trajectory screens,
  both with exact delay-lattice histories and coarse/fine numerical controls.
- Added General Dynamics registry v18 and a create-only lunar evidence summary.

### Results and boundaries

- The complete-deformation candidate improved all four registered endpoint
  metrics, but only one cleared the frozen one-percent materiality gate.
- The subsequent geodetic candidate improved none of the four metrics. Both
  promotion branches therefore stop, and the static mantle model remains the
  accepted default.
- Registry v18 keeps lunar mantle/core rotation `INCONCLUSIVE` and the whole
  project `SCREENING_ONLY`; no raw LLR, exact-DE440, production, or unified-
  multiphysics claim is made.
- The next lunar gate is simultaneous translation/rotation with complete
  delayed deformation and common-potential reactions, preregistered before a
  new long outcome run.

## 0.6.0rc2 — 2026-09-20

### Changed

- Changed the active JX project license prospectively from MIT to a
  proprietary, all-rights-reserved license held by Lino Avila.
- Added an explicit human-authorship record and repository-wide CODEOWNERS
  assignment to `@linomx90`.
- Kept General Dynamics registry v17 and all scientific claim ceilings
  unchanged; v17 remains historically bound to `0.6.0rc1`.

### Boundaries

- This license change applies to the new `0.6.0rc2` source and does not
  retroactively alter licenses attached to earlier public releases, frozen
  artifacts, or third-party material.
- External contributions require a separate written contribution and
  licensing agreement before incorporation.

## 0.6.0rc1 — 2026-09-20

### Added

- General Dynamics registry v17 with explicit top-level `SCREENING_ONLY`,
  package-version binding, and preserved capability-specific claim ceilings.
- A create-only release builder that performs two independent clean builds,
  canonicalizes source-archive metadata, and rejects non-identical artifacts.
- A fail-closed CUDA release gate with an exact five-test GPU core roster and
  a create-only machine-readable hardware/runtime report.
- Modern MIT license metadata, the full license text, packaging regressions,
  and independent wheel/source-install smoke gates.
- A reproducibly packaged private CPython extension for the audited 2--32-body
  point-mass RKF78 C core, used by the small-N prototype when installed.
- An unregistered `SCREENING_ONLY` simultaneous Earth--Moon translation,
  lunar-mantle quaternion/rate, and lunar fluid-core-rate RKF78 model with a
  common static degree-two orbit/spin potential and paired CMB torque.
- A hash-bound retained-DE440 SPK/PCK screen that separates numerical
  convergence from physical agreement and explicitly refuses DE440 or raw-LLR
  qualification claims.
- An additive v2 Sun--Earth--Moon coupled lunar boundary with simultaneous
  solar forcing, common-potential solar/lunar-figure torque, and a
  reaction-balanced fixed-pole Earth-J2 orbital correction.

### Changed

- Aligned package, CLI, citation, documentation, and registry identity at
  `0.6.0rc1`.
- Corrected CuPy trajectory documentation to match its actual no-host-hash,
  device-resident ownership contract. No numerical behavior changed.
- Preserved explicitly selected virtual-environment Python paths in the test
  matrix runner instead of resolving them to the base interpreter.
- Native release wheels are now CPython/platform specific; source-only
  benchmark execution retains a temporary GCC fallback.

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
- The v1 isolated coupled lunar model differs from retained DE440 by about
  95.99 km after one day. The converged v2 Sun--Earth--Moon/J2 screen reduces
  that difference to about 1.251 m, but remains a descriptive comparison to a
  fitted ephemeris. Other planets, delayed deformation, Earth spin/pole
  dynamics, extended-Earth rotational torque, relativity, and the LLR
  observation model remain absent; neither model is registered or qualified.

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
