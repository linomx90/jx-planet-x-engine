# JX Long-Horizon Engine

## Milestone M1: precision-first restricted prototype

The **JX Celestial Dynamics Framework** now contains the first additive
prototype for a future **JX Long-Horizon Engine**.  Milestone M1 is a narrow
engineering component for testing numerical contracts.  It is not a finished
long-horizon integrator and does not change the qualified surface of
`jxplanetx.engine`.

The prototype is available only by direct import from:

- `jxplanetx._all_conic_kepler`, a private CPU/CUDA all-conic two-body drift;
- `jxplanetx.cuda_restricted_wh`, an experimental CUDA restricted
  Wisdom--Holman-style map;
- `jxplanetx.restricted_wh_route`, a primary-relative adapter for continuous
  DOP853 guarded-step redo and first-event location;
- `jxplanetx.cuda_restricted_wh_hybrid`, a transactional reference scheduler
  that keeps quiet lanes on CUDA and redoes guarded lanes on the CPU;
- `jxplanetx.restricted_wh_lifecycle`, a fixed-lane terminal lifecycle with an
  immutable in-memory removal ledger;
- `jxplanetx.restricted_wh_source`, an experimental replay-verified massive
  source-path provider using JX's compiled ordered-Jacobi Wisdom--Holman map.

M1 uses binary64 throughout its numerical state and CUDA arithmetic.  Its
declared result classification is `MODEL_OUTPUT / SCREENING_ONLY`.  The goal of
this milestone is to make precision failures visible and transactional before
adding production interfaces, disk restart, a complete encounter lifecycle,
or accepting performance optimizations as qualified behavior.

## Numerical map

Tracers are exactly massless and are represented in nonrotating,
primary-relative Cartesian coordinates:

```text
q = x_tracer - x_primary
u = v_tracer - v_primary
```

For every fixed map step, the CUDA component applies:

```text
perturbing half kick
    -> exact all-conic primary drift
    -> perturbing half kick
```

The drift solves the central two-body problem for elliptic, exactly parabolic,
near-parabolic, and hyperbolic states with a universal-variable formulation.
The kick contains only secondary-source perturbations.  It includes the
indirect acceleration of the primary, so the acceleration remains consistent
with the primary-relative coordinates.  The primary monopole appears only in
the exact Kepler drift.  For a source much farther away than the tracer, the
two nearly equal source accelerations are combined with a cancellation-safe
tidal identity rather than subtracted directly.

The source bodies follow an externally prescribed path.  That path may now be
built by the dedicated source provider, which advances only the fully
interacting massive subsystem and performs a second complete bitwise replay
before exposing a tile.  Tracer state is never accepted by the provider and is
never an input to source evolution, so back-reaction is structurally absent.
Because the resulting source path is still prescribed to the tracer map rather
than advanced as part of one autonomous Hamiltonian map, M1 does **not** claim
that the complete map is globally symplectic.

## Immutable source tiles and lattice time

The caller supplies a `FrozenSourcePathTile` containing fine source nodes,
source GMs, encounter radii, certified relative-acceleration bounds, coordinate
metadata, and provenance.  Construction accepts exact NumPy `float64` arrays,
copies them into immutable bytes-backed storage, and computes a canonical
SHA-256 digest over both metadata and array contents.  A supplied expected
digest must match.  The digest is rechecked before execution, so mutation or
transport corruption fails closed.

Source slot zero is the primary and must be exactly zero in relative position
and velocity at every node.  M1 accepts only the explicit `AU-YEAR` unit system
and the `PRIMARY_RELATIVE_NONROTATING` coordinate system.

Time is indexed by an integer `global_source_step`.  The epoch is derived from
the origin, this integer index, and `source_step_years`; a recursively updated
floating epoch is never the authoritative clock.  A coarse map step consumes
an exact positive integer number of fine source intervals.  This permits an
in-memory result to feed a following segment on the same source lattice.

That continuation is only an in-memory split.  M1 has no checkpoint writer,
checkpoint reader, predecessor chain, or disk-restart qualification.

The source provider converts a primary-relative initial massive state to its
GM-weighted barycentric representation, runs the complete compiled
Wisdom--Holman map and all of its guards, requires an exact replay, and converts
every retained node back to primary-relative coordinates.  The provider binds
its own source hash, the native source-map binary and compiler identity, input
and result digests, verification digest, numerical metrics, and final tile
digest.  A source-source guard failure returns no partial tile.  Each tile is
limited to one million source intervals so memory use remains explicit and
bounded.

The combined experimental entry point
`advance_cuda_restricted_wh_with_wisdom_holman_source` performs source-tile
construction and fused CUDA tracer advancement as one linked operation.  Its
result retains both custody records and requires the tracer result's tile hash
to equal the source build's immutable tile hash.  This is the preferred M1
entry point; the lower-level builder and tracer functions remain available for
precision comparisons.

The guarded variant
`advance_cuda_restricted_wh_hybrid_with_wisdom_holman_source` retains that
same link while using the transactional CUDA/DOP853 scheduler.  Guarded lanes
are collected into a canonical wave, evaluated by spawned CPU processes, and
merged in exact input-lane order.  Six workers are the frozen default on the
target Ryzen 5 5500 because that processor has six physical cores.  The worker
count and spawn policy are recorded in every result runtime identity.  This is
the correctness reference for guarded-step work; it is not yet the production
encounter-episode path.

The lifecycle variant
`advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source` adds stable
per-lane lifecycle codes.  A continuously located source-radius contact or
outward primary-boundary crossing freezes that lane at the event while active
neighbors continue.  Lanes are never compacted.  Each terminal transition
adds one canonical removal record whose source-segment hash depends on the
exact global lattice interval and source-node bytes, not on arbitrary outer
tile chunking.  This makes the removal record identical across an in-memory
split, but it is still not a disk-checkpoint chain.

## Guards and fail-closed behavior

M1 is intended only for quiet restricted trajectories.  Before the drift it
checks endpoint source distances and a conservative swept-distance certificate.
For every source, the caller must provide a positive upper bound on the
magnitude of tracer--source relative acceleration throughout the coarse map
interval.  The device combines that assertion with the current separation,
relative velocity, and step duration to compute a conservative lower bound on
separation.  It also limits the primary dynamical phase using that certified
lower radius.  The post-kick endpoint is checked again.

These are **caller-certified swept guards**, not by themselves an encounter
detector.  In the quiet-only entry points, an insufficient certificate,
equality with a guard surface, a nonfinite value, a singular force, or an
excessive primary phase causes the complete requested segment to fail.

The additive hybrid reference can redo a guarded coarse step with SciPy
DOP853.  It deterministically reconstructs a GM-barycentric realization of
the primary-relative source nodes, consumes every fine cubic-Hermite source
interval, continuously searches every source surface (including tangencies),
and returns to primary-relative coordinates only at a full-step endpoint or
the first terminal event.  Quiet lanes remain bitwise equal to the fused CUDA
map.  Rejoined lanes are merged in original lane order.  A terminal event,
ambiguous location, phase-policy violation, or non-routable device failure
aborts the whole public transaction and exposes the unchanged input state
with a route witness.  Terminal lifecycle/removal, an outer-boundary guard on
the quiet CUDA path, and an append-only in-memory removal ledger are now
implemented by the lifecycle wrapper.  Removal at the outer surface is only an
operational domain exit; it is not called an ejection.  Disk persistence and a
whole-episode worker remain intentionally unimplemented.  The current parallel
wavefront routes one guarded map step at a time.

Execution is transactional.  Numerical work occurs in private device buffers.
On a guard, Kepler-solver, or CUDA failure, the raised handoff refers to the
exact unchanged input state and reports zero committed map steps.  A successful
result is committed only after the full segment completes.

## Precision gates in M1

Precision is the first acceptance criterion.  Throughput is not an acceptance
criterion until these gates remain satisfied.

1. **Binary64-only numerical path.** Tracer/source arrays and the universal
   drift use binary64.  CUDA compilation disables fused multiply-add contraction
   for the drift so its operation order is explicit within the documented
   same-runtime scope.
2. **All-conic root contract.** The signed universal anomaly is bracketed, the
   bracket is preserved, and iteration and expansion counts have fixed caps.
   Near zero universal argument uses a series with a fixed 64-term ceiling.  It
   may stop only once all four next additions are bitwise inert; a regression
   compares that result exactly with the complete 64-term evaluation.  Exact
   parabolic motion has its own finite limit; nonfinite hyperbolic growth is
   rejected.
3. **Per-drift postconditions.** A drift is accepted only when the time residual
   is within its scale- and ULP-aware tolerance and the reconstructed state
   satisfies roundoff-scaled Lagrange-identity, specific-energy, and angular-
   momentum checks.
4. **Independent known answers.** The focused tests include 100-decimal-digit
   oracle vectors rounded once to binary64 across eccentric elliptic, exactly
   parabolic, near-parabolic on both sides, and hyperbolic cases.  They also
   check signed forward/reverse drift, three-dimensional rotation covariance,
   continuity at the parabolic boundary, and overlap with the existing
   elliptic solver.
5. **CPU/CUDA agreement and repeatability.** Mixed-conic CUDA lanes are compared
   with the independent vectors, repeated on the same device/runtime, and
   reversed.  Invalid lanes must return an explicit status and no usable state.
6. **Source and segment invariants.** The restricted-map tests require source
   tile digest stability, exact rejection of tampering, integer source-node
   selection, transactional guard failure, and bitwise equality between an
   uninterrupted primary-only segment and the equivalent in-memory split on
   the same runtime and device.  They also cover signed-time splits, empty
   tracer batches, lane permutation, CUDA block-size changes, and a distant
   secondary whose physical tidal kick would disappear under naive floating-
   point subtraction.
7. **Massive-source replay and independent refinement.** The source-provider
   tests require immutable input custody, exact source-map replay, deterministic
   tile hashes, fixed all-false claims, and fail-closed source-source guards.
   Separate refinement compares the massive endpoint with JX GR15 at two GR15
   controller tolerances before the tile is accepted for tracer experiments.
8. **Guarded-step coordinate and transaction closure.** The DOP853 adapter is
   checked against the exact all-conic two-body solution in both time
   directions, uses every retained fine source interval, locates source and
   outward-boundary surfaces continuously, and binds the complete numerical
   route policy.  Hybrid GPU tests require quiet bitwise parity, deterministic
   multi-lane route order, bitwise split continuity across consecutive routes,
   quiet-prefix recovery, and unchanged-input handoff on a terminal event.
9. **Deterministic route-wave parallelism.** Fixed one-, six-, and twelve-worker
   runs must produce the same endpoint digest and canonical route order.  On
   the target six-core/twelve-thread Ryzen 5 5500, a 96-route synthetic wave
   measured medians of `8.704`, `2.676`, and `3.139` seconds respectively.
   Six physical-core workers were therefore 3.25 times faster than serial and
   1.17 times faster than twelve SMT workers without changing a bit of the
   accepted state.
10. **Fixed-lane terminal continuity.** A terminal source contact and outward
    boundary crossing must freeze only the located lane, keep every other lane
    advancing, preserve the terminal position and velocity bitwise on later
    calls, and reject any attempted frozen-state mutation.  Split and
    uninterrupted runs must have bitwise-identical state/lifecycle arrays and
    the same canonical removal-record digest.

Successful results bind the implementation source, all-conic source, CUDA
kick source and compiler options, CuPy/CUDA/NVRTC versions, and the physical
GPU UUID/compute capability in their runtime identity.  This is evidence for
the recorded run; it is not yet a persistent checkpoint lineage.

Passing these focused gates establishes only the behavior they test.  It does
not extrapolate accuracy or stability to astronomical durations.

## Fused CUDA execution and measured bounds

M1 now has two execution paths for the identical map:

- `advance_cuda_restricted_wh`, the stepwise precision reference;
- `advance_cuda_restricted_wh_fused`, which applies every map step in one
  per-tracer CUDA loop and performs one synchronized segment launch.

The fused kernel calls the same binary64 kick and all-conic device functions.
It records the first lane failure and the host applies the same canonical
`(map step, phase, lane)` selection as the reference.  Tests require bitwise
equality of positions, velocities, tracer IDs, minimum-distance diagnostics,
and maximum Kepler iterations for primary-only and multisource tiles, integer
source-step ratios, signed time, empty populations, guard handoffs, CUDA block
sizes, and the cancellation-sensitive distant-source case.

One exact-result optimization was material: the small-argument G series no
longer executes terms that cannot alter any binary64 accumulator.  The
optimized and earlier implementations produced the same endpoint SHA-256
`8543854738ba40809d727436740b979a43964b3b181c9c5fd8662ba1c6587d53`
for the bounded 16,800-tracer, 8,000-map-step fixture.

On the recorded RTX 5060 Ti / CuPy 14.2 / CUDA 13.2-runtime, 13.4-NVRTC
environment, the synthetic six-source throughput fixture measured:

| Source steps per map step | Map step | Median fused wall time for 1,000 modeled years | Fine-lattice-equivalent rate |
|---:|---:|---:|---:|
| 1 | 0.125 year | 2.780 s | 48.35 million tracer-steps/s |
| 4 | 0.5 year | 0.829 s | 162.20 million tracer-steps/s |
| 8 | 1.0 year | 0.436 s | 308.55 million tracer-steps/s |

The ratio-1 fused path was 1.72 times faster than the optimized stepwise path
and about 10.4 times faster than the first fixed-64-series fused prototype.
These are bounded engineering timings.  The distant static-secondary fixture
is designed to measure the numerical kernel and is not a Solar-System model.
The fine-lattice-equivalent rates at ratios 4 and 8 do not imply that those
coarser map steps are sufficiently accurate.

The hybrid scheduler adds less than one percent measured cost on this quiet
segment because it does not create worker processes unless a guard is
encountered.  Five repeated 16,800-tracer, 8,000-step quiet runs had the same
endpoint SHA-256
`8543854738ba40809d727436740b979a43964b3b181c9c5fd8662ba1c6587d53`;
the hybrid median was `2.794` seconds, or 48.10 million tracer-steps per second,
versus `2.777` seconds for the direct fused path.  The measured overhead was
0.61 percent.

A separate refinement used the tracked current-epoch six-source control state
and transformed tracer population.  It selected the first 2,048 frozen tracer
IDs with initial primary-relative two-body perihelion at least 60 AU, built one
common 0.125-year massive-source KDK path, and compared restricted-map steps
of 0.125, 0.25, 0.5, and 1.0 years over 1,000 years.  Relative to the same-model
0.125-year endpoint, the 95th-percentile absolute perihelion differences were
approximately 0.00478, 0.0239, and 0.1005 AU.  Error ratios followed the
expected second-order pattern closely.  The finest run repeated bitwise.

That refinement is not independent accuracy validation: all ratios share the
same 0.125-year source path, and the finest tracer map is only a same-method
reference.  It supports the implementation's observed second-order behavior;
it does not yet select a production timestep.

That experiment also exposed the dominant M1 error: the old direct Cartesian
KDK source path, not the fused tracer kernel.  A new source-only comparison used
the tracked Sun--Jupiter--Saturn--Uranus--Neptune--Pluto state and an independent
GR15 reference.  Over 1,000 years, the two GR15 controller settings agreed to a
maximum source-position difference of about `7.69e-13` AU.  At a 0.125-year
source step, the ordered-Jacobi Wisdom--Holman path differed from GR15 by
`5.44e-5` AU, while direct Cartesian KDK differed by `3.96` AU.  The new path
was therefore about 72,800 times closer at the same step.  It was also about
295 times closer than KDK at the much smaller 0.0078125-year step.  The
Wisdom--Holman 0.25/0.125/0.0625/0.03125-year lanes showed the expected
second-order factor of approximately four; the 0.5-year lane failed its path
guard over this interval and was retained as a fail-closed result.

The complete source-provider-to-CUDA pipeline then advanced all 6,945 frozen
tracers satisfying the outcome-blind initial `q >= 60 AU` selection for 1,000
years at a 0.125-year tracer step.  Relative to the identical tracer map driven
by the GR15 source path, the replay-verified Wisdom--Holman provider produced a
median absolute perihelion difference of `8.49e-6` AU and a 95th percentile of
`3.85e-5` AU.  The old KDK source path produced `0.864` AU and `3.93` AU,
respectively.  Median endpoint position difference improved by about 5,608
times, and median perihelion difference by about 101,800 times.  These are
bounded engineering comparisons over one millennium, not a 4.5-Gyr accuracy
claim, but they select the replay-verified Wisdom--Holman provider for the next
M1 rung and retire direct Cartesian KDK as the intended long-horizon source
provider.

A coupled source-and-tracer refinement then used all 6,945 tracers in that
same frozen, outcome-blind selection for 1,000 years.  Each candidate used a
replay-verified Wisdom--Holman source tile and the same map step for the
restricted tracer map.  The comparison endpoint used a GR15 source path and a
0.03125-year tracer map.  The 95th-percentile absolute perihelion differences
at 0.25, 0.125, and 0.0625 year were `1.034e-2`, `2.460e-3`, and `4.903e-4`
AU; the corresponding median values were `2.308e-3`, `5.494e-4`, and
`1.095e-4` AU.  The 0.25-to-0.125 and 0.125-to-0.0625 95th-percentile ratios
were 4.20 and 5.02, while the latter median endpoint-position ratio was 5.00.
The finest Wisdom--Holman-source lane at 0.03125 year differed from the
GR15-source comparison by only `2.40e-6` AU at the 95th percentile in
perihelion and `8.79e-8` AU at the 95th percentile in endpoint position.

This finite-reference ladder selects 0.0625 year as the precision-first
candidate for the next bounded validation rung.  It does **not** select a
production timestep: the reference retains the same restricted tracer map,
the population excludes encounter-prone initial conditions by construction,
and neither close encounters nor persistent restart are implemented here.

## Explicitly false claims

For M1, all result claim controls are fixed to false.  In particular, this
prototype does not establish or provide:

- production qualification;
- 4.5-billion-year accuracy, stability, or completion;
- disk checkpoint or restart qualification;
- production-qualified close-encounter episodes, collision response,
  physical collision/ejection inference, or regularization (M1 contains the
  bounded guarded-step redo, operational terminal lifecycle, and first-event
  locator described above);
- a globally symplectic map when driven by the external source path;
- scientific inference, including detection, exclusion, orbit recovery, or a
  stellar or rogue-body flyby conclusion;
- independent reproduction or cross-runtime/cross-device bitwise
  reproducibility.

The name “Long-Horizon Engine” describes the target subsystem, not the present
qualification level.

## Next milestones

Work should advance only through evidence-backed rungs:

1. **Close the M1 precision audit.** Retain independent high-precision oracle
   coverage over a wider conic/step domain; stress the parabolic boundary,
   reversibility, invariant budgets, source immutability, deterministic lane
   layouts, and every fail-closed status on the target GPU.
2. **Qualify the restricted map.** Use the replay-verified ordered-Jacobi source
   provider for every new rung.  Carry the 0.0625-year candidate into
   representative encounter-adjacent subsets and an additional independent
   implementation, while retaining 0.03125 year as the refinement lane.
   Define bounded distribution-level error budgets before selecting a
   production timestep or increasing duration.
3. **Specify persistent evidence.** Design content-addressed checkpoints with
   complete source-schedule identity, tracer genesis, predecessor linkage,
   runtime identity, and replay validation.  Only then add disk restart and
   test every split point against uninterrupted execution.
4. **Complete encounter episodes deliberately.** Extend the now-tested
   guarded-step handoff/rejoin and fixed-lane terminal lifecycle with
   persistent route/removal ledgers.  Replace iterative guarded-lane discovery
   and one-step CPU jobs with GPU compaction plus whole-episode worker
   ownership, while requiring exact agreement with the deterministic
   six-worker reference wavefront.  A guard hit must never be silently
   interpreted as a collision or flyby.
5. **Add multirate execution without changing the mathematics.** The fixed
   integer-ratio fused path now has exact parity with the stepwise reference.
   A future adaptive hierarchy still requires a reversible transition policy,
   correctors, conservative eligibility relative to every massive source, and
   independent refinement evidence before its speedup can be accepted.
6. **Climb duration and population gates.** Authorize progressively larger
   validation windows only after precision, restart, custody, comparison, and
   resource gates pass.  A throughput estimate cannot authorize a 4.5-Gyr
   scientific run.

Performance work follows these gates.  The GPU is valuable because independent
tracer lanes can be advanced in parallel against one shared immutable source
path, but using more hardware does not reduce model error or replace numerical
qualification.
