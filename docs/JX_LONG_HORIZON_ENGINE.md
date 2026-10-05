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
- `jxplanetx.cuda_restricted_wh_episode`, an experimental episode-aware
  scheduler that classifies a complete guarded wave in one CUDA launch and
  groups consecutive routed macrosteps without changing the route solution;
- `jxplanetx.cuda_restricted_wh_episode_owner`, an experimental scheduler that
  assigns each consecutive routed episode to one dedicated CPU worker and
  caches only CUDA-certified, bitwise-start-matched prospective routes;
- `jxplanetx.restricted_wh_episode_checkpoint`, an experimental delta
  checkpoint chain for persistent-owner segments that stores only each
  segment's new route and episode evidence while replay-verifying the complete
  predecessor chain;
- `jxplanetx.restricted_wh_lifecycle`, a fixed-lane terminal lifecycle with an
  immutable in-memory removal ledger;
- `jxplanetx.restricted_wh_checkpoint`, a create-only, content-addressed,
  predecessor-linked lifecycle checkpoint chain whose segments are replayed
  before publication and again before restart;
- `jxplanetx.restricted_wh_source`, an experimental replay-verified massive
  source-path provider using JX's compiled ordered-Jacobi Wisdom--Holman map.

M1 uses binary64 throughout its numerical state and CUDA arithmetic.  Its
declared result classification is `MODEL_OUTPUT / SCREENING_ONLY`.  The goal of
this milestone is to make precision failures visible and transactional before
adding production interfaces, a complete encounter lifecycle, or accepting
performance optimizations as qualified behavior.  Disk restart now exists as
an experimental replay-verified component, but it is not production-qualified.

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

The direct numerical result remains an in-memory segment.  The additive
checkpoint module can now persist complete lifecycle segments without changing
that result contract.  Every checkpoint stores the exact segment input and
output state, the immutable source tile, integer lattice endpoints, route
policy, route/removal deltas, runtime identity, code/kernel identity, and a
required externally supplied genesis-state commitment.  Records are namespaced
by run and protocol digest, arrays are content-addressed, JSON publication is
create-only, and every successor names the latest predecessor digest.  A
namespace-scoped operating-system lock serializes the complete verify-and-
publish transaction so concurrent writers cannot create a fork.  Loading also
requires an external expected head index and content digest; a deleted or
rolled-back tail is therefore an integrity failure rather than a valid older
restart.  The first segment's reconstructed input bytes must hash to the
external genesis commitment; retaining only a rewritten genesis label cannot
substitute a different starting state.

Checkpoint loading verifies every artifact and predecessor, reconstructs every
input and tile, then deterministically replays the complete chain on the exact
recorded Python/NumPy/SciPy/CuPy/CUDA/GPU/CPU runtime before releasing the
latest device state.  Appending first performs the same full predecessor
verification.  Only per-segment route and removal evidence is repeated; the
chain does not copy an ever-growing cumulative route ledger into each record.
An array orphaned before its JSON record is tolerated, while a committed JSON
record with missing, changed, or incompatible bytes fails closed.  This proves
same-runtime restart integrity for the recorded chain; it is not a production
or cross-runtime reproducibility claim.

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
the correctness reference for guarded-step work.

The additive episode-aware variant keeps that reference mathematics and route
solver unchanged.  At the first guard, one fused CUDA classification launch
exposes every lane's first failure for the current map step.  Quiet lanes use
their exact fused endpoint; guarded lanes are redone by the same DOP853 worker
function and merged in the same canonical lane order.  The scheduler remains
in episode mode until one full map step has no routes, and it groups each
tracer's maximal sequence of consecutive routed macrosteps into an immutable
operational episode.  “Episode” here means consecutive guard activity, not a
physical encounter classification.  CPU route jobs are still one macrostep at
a time, and this direct-module component has no checkpoint or production
contract.

The additive persistent-owner variant keeps the same CUDA classification and
DOP853 step solution.  The first routed step is deliberately non-speculative.
If CUDA requests the same tracer again on the next map step, one of six
dedicated spawned workers computes a bounded prospective chain.  Each later
result is released only when CUDA independently requests a route at that exact
source node and its binary64 input position and velocity match bit-for-bit.
A quiet step immediately releases the owner and discards every unused result.
Thus ownership and lookahead reduce process submissions without deciding which
steps are routed or altering any committed endpoint.  The default lookahead is
32 map steps, the hard limit is 64, and both values are recorded.  This
direct-module component remains in-memory, experimental, and non-production.

The additive persistent-owner checkpoint layer deliberately restarts at an
episode-segment boundary rather than attempting to serialize live worker
processes or speculative cache entries.  Each record stores only the current
segment's canonical route rows, worker assignments, and derived episode
summaries.  A rolling route-chain digest and exact cumulative counters bind
those deltas across records without copying all earlier route rows into every
checkpoint.  The standalone append API loads and completely replays the
existing head before publishing a successor.  The locked append-session API
instead performs that full replay once, retains a private process-and-thread
bound authorization capability, rehashes every committed JSON and NPZ before
each append, and replays only the new segment.  It advances the capability only
from newly published bytes that were read back and checked.  Any error closes
the session.  Loading, reopening, and the final evidence gate still require an
externally anchored complete replay from the genesis commitment.  This makes
numerical append replay linear in the number of segments inside one session;
artifact hashing remains cumulative and the current component remains bounded
engineering evidence, not a production persistence design.

The GPU encounter surface and terminal source surface are separate policies.
The larger encounter radius decides when the quiet CUDA map must be discarded
and rerouted.  An optional smaller terminal-radius roster decides whether the
continuous DOP853 route freezes the lane; if omitted, legacy behavior uses the
encounter radii for both.  Every explicit terminal radius must cover the exact
source roster and cannot exceed its encounter radius.  This distinction lets a
real close pass rejoin without being mislabeled as a physical contact.

The lifecycle variant
`advance_cuda_restricted_wh_lifecycle_with_wisdom_holman_source` adds stable
per-lane lifecycle codes.  A continuously located source-radius contact or
outward primary-boundary crossing freezes that lane at the event while active
neighbors continue.  Lanes are never compacted.  Each terminal transition
adds one canonical removal record whose source-segment hash depends on the
exact global lattice interval and source-node bytes, not on arbitrary outer
tile chunking.  This makes the removal record identical across an in-memory
split.  The removal record alone is not a disk-checkpoint chain; the separate
checkpoint module binds it to the exact state, source tile, segment, and
predecessor replay.

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
operational domain exit; it is not called an ejection.  Replay-verified disk
persistence is implemented by the separate lifecycle checkpoint layer and the
persistent-owner delta checkpoint layer.  A checkpoint boundary intentionally
discards live ownership and unused speculative results; after restart the GPU
guard and bitwise route-start comparison must again authorize every committed
map step.

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
11. **Replay-verified persistent lineage.** A checkpoint can be published only
    after a fresh deterministic segment replay matches the supplied endpoint,
    diagnostics, routes, removals, and runtime evidence.  Loading rehashes the
    exact NPZ roster, verifies the genesis and predecessor chain, reconstructs
    the source tile and both endpoint states, and replays every segment before
    returning a synchronized CUDA state.  Tests cover quiet and terminal
    round trips, a two-segment chain whose numeric latest index is 1,000,
    wrong input/predecessor rejection, serialized concurrent appenders,
    create-only behavior, namespace type failures, crash-orphan tolerance,
    explicit tail-rollback rejection, predecessor corruption before append,
    exact source-boundary/GM continuity, canonical index bounds, exact
    binary64 diagnostic arrays, and rehashed claim/runtime tampering.
12. **Delta episode lineage.** Persistent-owner checkpoints retain only the
    current segment's route rows and episode summaries, bind them with a
    rolling route-chain digest and exact cumulative counts, and reset live
    worker ownership at the boundary.  Standalone append authorization fully
    reloads and replays the predecessor under the namespace lock.  A locked
    session may replay the predecessor once, but must rehash all committed
    artifacts on every append, use a private PID/thread/runtime-bound head,
    serialize the independently replayed endpoint, read it back, and poison
    itself after any failure.  Tests require exact direct-versus-restart state
    and result evidence, six-worker episode ownership, source-node continuity,
    externally anchored heads, strict binary64 arrays, caller-buffer isolation,
    and rejection of modified NPZs, cross-thread/process use, runtime drift,
    and rehashed worker, source, claim, implementation, genesis, index, and
    predecessor forgeries.

Successful results bind the numerical implementation sources, all-conic
source, CUDA kick source and compiler options, CuPy/CUDA/NVRTC versions, and
the physical GPU UUID/compute capability in their runtime identity.  The
checkpoint layer additionally binds its local backend, contract, and
Wisdom--Holman contract modules, the host runtime/build evidence, the
protocol/genesis commitment, and hash-linked persistent lineage, while
retaining the same conservative claim ceiling.

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
and neither whole close-encounter episodes nor long-duration restart
qualification has been established.  The new checkpoint component establishes
exact bounded same-runtime segment restart behavior only.

### Bounded checkpoint and six-worker qualification

The replay-verified checkpoint component has now passed a deliberately bounded
route-heavy engineering qualification on the RTX 5060 Ti and Ryzen 5 5500.
The fixed fixture used 256 guarded tracers for four map steps, split into two
two-step checkpoints.  It produced 1,024 nonterminal route records.  Six
spawned route workers were restricted to physical CPUs 0--5, and the numerical
thread environment was fixed to one thread per worker.

All 1,024 normalized route rows, the final state, and both loaded endpoints
were bitwise identical to the uninterrupted four-step result.  Two independent
full-chain loads returned the same state and checkpoint head.  The canonical
route-ledger SHA-256 is
`6e1ca67cbf5fbf7b6eabee6ab2fec074853e13022b774093aad4ffcc11b866d4`;
the final lifecycle-state SHA-256 is
`ec81b6a18159b829aa6103d824cea9ff521086580842ff94b4dc3dac831b92f6`.
The two JSON records occupied about 2.69 MB each, or about 5.13 KiB per
route row.  Full-chain replay took about 19 seconds on this exact runtime.

A separate three-repeat scaling screen used 12 guarded tracers for eight
steps.  One worker took about 8.6 seconds and six workers about 2.3 seconds:
an approximately 3.8-times speedup.  Endpoint and complete
route-ledger hashes were identical between worker counts and across repeats.
This selects six physical-core workers for the current deterministic CPU route
path; it is not a cross-machine performance claim.

Process-interruption tests kill a separate writer with `os._exit` while
appending a successor checkpoint at all four publication boundaries: before
and after the content-addressed NPZ hard link, and before and after the JSON
hard link.  They verify kernel-released namespace locking, exact predecessor
preservation, stale temporary/orphan handling, retry behavior, old-head
rejection when a new JSON head is visible, and explicit replay-verified head
adoption.  These tests cover process death on the local filesystem.  They do
not establish controller, kernel, filesystem, or power-loss durability.

The retained evidence is under
`runs/restricted_wh_checkpoint_qualification_v1/`.  Its result status is
`PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE`, and every production,
long-horizon, cross-runtime, power-loss, and scientific claim remains false.

### Episode-wave engineering qualification

The episode-aware scheduler has passed a separate bounded exact-parity and
throughput screen on the same RTX 5060 Ti and six physical Ryzen 5 5500 cores.
The synthetic fixture contained 48 routed lanes and 16 quiet lanes for eight
map steps: 384 DOP853 route records grouped into 48 eight-step operational
episodes.  It is deliberately route-heavy and primary-only; it is not a model
of a physical encounter-rate distribution.

Three modes were each repeated three times.  Every repeat produced the same
state SHA-256
`e39ce956d8864a5a20b6d874911f8ea75352de01ca18598d26a172a1d1bf1f5a`
and the same complete route-ledger SHA-256
`144bb5373071319665f02b82c6cedf78619b6238f6f98f4ff0d9048bd180c9ca`.
Median wall times were:

| Mode | Median seconds | Speedup versus one-worker reference |
|---|---:|---:|
| Reference scheduler, one CPU worker | 34.266 | 1.00× |
| Reference scheduler, six CPU workers | 7.118 | 4.81× |
| Episode-aware scheduler, six CPU workers | 6.504 | 5.27× |

The episode-aware classification is 1.094 times faster than the already
parallel six-worker reference on this bounded fixture.  Sample coefficients
of variation were below 0.51 percent, the process was pinned to physical CPUs
0--5, and each numerical library was limited to one thread per worker.  This
selects the all-lane CUDA classification wave as the next experimental route
path.  It does not qualify long-horizon accuracy, completion, collision
response, physical encounter interpretation, disk restart, or production use.

The retained create-only evidence is under
`runs/restricted_wh_episode_qualification_v1/`.  Its protocol SHA-256 is
`5d0fdd54614aabb3c2fdf40962fca7b6544b8b36c14b3924c1e6c5b26e356179`;
its result content SHA-256 is
`8a508bab7ee11b728bdcac341402c57aafcf73a8c0eb33af65eeb4647823dce8`.
The status is `PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE`, with every claim
control false.

### Persistent-owner engineering qualification

The persistent-owner scheduler passed a second create-only exact-parity and
throughput screen on the RTX 5060 Ti and physical Ryzen 5 5500 CPUs 0--5.  The
fixture used six routed and six quiet lanes for 196 map steps, matching the
longest operational route episode observed in the earlier bounded mechanism
screen.  It produced 1,176 canonical DOP853 route records in six episodes.
Four balanced-order repetitions compared the qualified episode-wave scheduler
with six persistent owners and a 64-step hard-bounded lookahead.

Every repetition produced the same state SHA-256
`a85db82fbdcc59fb900a538a29abb91ae1ee2ad19d1ab7f706b1663105666d72`
and the same complete route-ledger SHA-256
`491a02956247d14a09d36409512b43ee05e461ffc5be5a6351de6338c977910d`.
All six workers were used, every episode remained on one worker, and 1,176
route records required 30 worker submissions; 1,146 later steps were released
from the exact prospective cache only after their CUDA and start-state gates.

| Mode | Median seconds | Relative speed |
|---|---:|---:|
| Episode-wave scheduler, six CPU workers | 19.873 | 1.000× |
| Persistent-owner scheduler, six CPU workers | 18.911 | 1.051× |

The persistent-owner path was 5.09 percent faster on this long, route-heavy
fixture.  Its sample coefficient of variation was 0.36 percent, compared with
1.67 percent for the reference.  Focused tests also require exact parity for
isolated one-step guards, a deferred terminal event, a moving secondary with a
3:1 fine-source ratio, negative time, mixed quiet/routed waves, and
transactional non-routable failure.  Infrastructure failure while constructing
a prospective chain still fails the complete call immediately; only a
successfully serialized future numerical result can be deferred.

The retained evidence is under
`runs/restricted_wh_owned_episode_qualification_v1/`.  Its protocol SHA-256 is
`4706f07ca061de03186e0baff0b44967afb67f3dfbf0b1c0c3e8e1f3e02d544a`;
its result content SHA-256 is
`200ccf5d1906756c5053cc84120c4641c3980564f804bcd97f73eff731c6c95b`.
The status is `PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE`.  This bounded
speed result does not qualify checkpoint/restart, physical encounter rates,
long-horizon accuracy or completion, collision response, or science use.

### Persistent-owner delta-checkpoint qualification

The additive delta-checkpoint layer passed a separate create-only engineering
qualification on the RTX 5060 Ti and physical Ryzen 5 5500 CPUs 0--5.  The
synthetic primary-only fixture used six continuously routed lanes for two
16-step segments.  Each segment retained exactly 96 new route records and six
new operational episode summaries; the complete replayed chain contained 192
routes and 12 segment-local episodes.

The uninterrupted second segment, the segment resumed from the first loaded
checkpoint, and a fresh load of the final chain produced the same exact state
SHA-256:
`18451651ab50d8a0ecf60908a7adb2a1a08cc782780287f2c5a53dad6c181e24`.
The direct and resumed second-segment result evidence also matched exactly at
`fe45c9f9b3b1c936895a587563f6bf78946f1ce98ced9e3e36dc367200f0d1af`.
The two checkpoint JSON records were 515,569 and 515,764 bytes, a size ratio of
1.00038 despite the cumulative route count doubling.  This is the intended
delta-storage property; it is not a claim that replay cost is constant.

The final externally anchored checkpoint head is
`638da7b2e474255bb7618220cc7a55bdd6820781303d6bee49665e451bcbf2ec`,
and the final rolling route-chain SHA-256 is
`827a49f742c80011a3f9bd303a7f957ffdfe39859e9e98660ef37f590832b4da`.
A separate fresh process, using the exact recorded six-core affinity and
runtime, reloaded and replayed the official chain to the same head, state, 192
routes, and 12 episode summaries.  Starting that verifier with a different
runtime identity failed closed before releasing state.

The retained evidence is under
`runs/restricted_wh_episode_delta_checkpoint_qualification_v1/`.  Its protocol
SHA-256 is
`9b75d14ca48e0b4348309033c562481a0bee3b0819c51a01e9aa0a6c9bf85860`;
its result content SHA-256 is
`38501887afc9b4b03ce5e0d2e3a941f911b2b3db5dfa8ea59c1a1682f96e981b`.
The status is `PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE`, with every claim
control false.  The fixture covers only two synthetic segments on one exact
runtime.  It does not qualify owner persistence across a checkpoint, physical
encounter rates, long-duration replay cost, production use, or science.

## Explicitly false claims

For M1, all result claim controls are fixed to false.  In particular, this
prototype does not establish or provide:

- production qualification;
- 4.5-billion-year accuracy, stability, or completion;
- production-qualified or cross-runtime disk checkpoint/restart behavior (an
  experimental same-runtime replay-verified chain is implemented and tested);
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
3. **Qualify persistent evidence at larger scale.** The content-addressed
   checkpoint format now binds complete source-schedule identity, tracer
   genesis, predecessor linkage, runtime identity, and deterministic replay.
   The bounded 1,024-route chain and all four process-publication crash points
   pass.  Extend this to representative long segments, every operational split
   point, cumulative-ledger and disk-capacity budgets, and independent archive
   reconstruction before promoting restart to a production interface.
4. **Persist episode evidence deliberately.** All-lane CUDA classification,
   exact episode grouping, and persistent six-worker ownership with bounded
   prospective caching now pass exact-parity gates.  The delta checkpoint
   schema now binds per-segment worker/episode evidence without duplicating the
   historical route ledger, and its bounded split/restart qualification passes.
   The next rung is representative non-synthetic episode-duration, cumulative
   replay-time, memory, and disk-resource testing, followed by a linear-time
   verifier receipt or equivalent if repeated full-chain replay becomes the
   limiting cost.  Exact endpoint/route parity and transactional first-event
   behavior remain mandatory.  A guard hit must never be silently interpreted
   as a collision or flyby.
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
