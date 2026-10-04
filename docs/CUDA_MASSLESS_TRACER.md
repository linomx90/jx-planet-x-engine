# CUDA massless-tracer backend

## Current outcome

JX now contains an additive, package-owned experimental backend for advancing a
large population of exactly massless tracers under a small mutually interacting
Newtonian source system.  It runs on the explicitly selected NVIDIA/CuPy device
for the shared source path and quiet tracer lanes.  A guarded lane deliberately
uses the separately identified CPU SciPy DOP853 route described below; there is
no silent whole-run CPU substitution.

The implementation is `MODEL_OUTPUT / SCREENING_ONLY`.  It is software needed
for the proposed outer-Solar-System experiments; it is not yet a qualified
4.5-Gyr science result.

The current M2 contract adds continuous operational removal surfaces,
fixed-lane lifecycle state, a central-periapse phase guard, scale-aware event
tolerances, closed force-schedule identities, and checkpoint-v4 event lineage.
These additions have focused CPU/GPU coverage.  The four-arm runner now keeps
the 1,000-year mechanics qualification separate from a required 100,000-year,
16,800-tracer route-capacity pilot.  Only a passing pilot receipt with the same
physical initial bytes, 50,000-year passage geometry, runtime, implementation,
and frozen population can authorize a 1-Myr attempt.  The 1-Myr run starts
separately and repeats the first 100,000 years; pilot checkpoints are not
relabelled or reused.

The final v3 RTX acceptance used the Sun and four giant planets with 16,800
deterministic synthetic tracers for 5,000 years at a 0.5-year step.  The source
was first frozen at commit `2cb655f04df3a8057c53d3d64575813e2680c500`.
It then completed 168,000,000 tracer-steps in 5.223346 seconds (32.163291
million tracer-steps/second) on the RTX 5060 Ti.  All endpoint arrays were
finite, all 522 guarded steps were transactionally resolved, the embedded
focused suite passed 21/21 tests, and every evidence checksum verified.  The
result status is `PASSED_ENGINEERING_ACCEPTANCE_NOT_SCIENCE`.  The preserved v1
and v2 measurements remain historical pre-audit evidence and do not qualify the
current source.  V3 includes the exact protocol, test transcript, raw input and
endpoint states, route ledger, environment, result, and run-wide checksums.

## Numerical architecture

For each bounded chunk, the backend:

1. Advances the active point-mass sources once with a barycentric inertial,
   fixed-step kick--drift--kick map.
2. Retains immutable source position, velocity, and acceleration nodes for the
   chunk.
3. Advances independent CUDA tracer lanes against that one shared source path.
4. Stops a lane before committing any step whose swept-distance, operational
   surface, acceleration, or central-periapse certificate is unavailable.
5. Redoes the complete untouched outer step with SciPy DOP853 against the same
   cubic-Hermite source segment and continuously searches that accepted dense
   solution for source-radius or outward model-boundary events.
6. Either rejoins the CUDA path at the exact next lattice node or freezes the
   fixed tracer lane at the first located operational removal event.

Tracers never accelerate one another and never affect a source.  Source bytes
therefore do not depend on tracer count, lane order, CUDA block partitioning,
or encounter routing.  The focused tests exercise 0, 1, 16, 65, and 129 tracer
layouts and compare shared source endpoints bit for bit.

The route is intentionally called an `encounter_route`, not a backend fallback.
The configured CuPy backend remains the integration backend.  A guard switch
means only that the quiet CUDA step was not certified; it is not evidence of a
physical encounter, contact, collision, or ejection.

Route evidence is also resource-bounded.  The current engine contract permits
at most 16,384 cumulative routes for one tracer and 100,000 cumulative route
records for one arm.  The four-arm runner additionally requires at least
16 GiB of free host storage before every segment and checkpoint write.  These
are fail-closed engineering limits that protect recoverable evidence; they are
not physical encounter limits or completion guarantees.

## Guard contract

Each source has three caller-bound device arrays:

- an encounter radius in AU;
- a nonnegative operational source-removal radius in AU, no larger than its
  encounter radius;
- a positive upper bound on relative acceleration in AU/year² over one outer
  step.

The state may also declare one primary-relative outer model-domain boundary.
The quiet route conservatively routes any possible source-surface crossing,
outer-boundary crossing, or primary-periapse interval whose maximum declared
dynamical phase cannot be certified below the fixed ceiling.  Equality routes.
The CPU route uses exhaustive stationary-point analysis of the degree-seven
dense relative trajectory, including grazing contacts, and fails closed when a
root or tolerance is numerically ambiguous.

The quiet route requires a positive full-step separation lower bound after the
relative-speed and acceleration envelopes, and also checks the minimum of the
linear swept segment.  Equality routes.  If the instantaneous relative
acceleration already exceeds the declared bound, the lane routes rather than
committing.

The acceleration envelope remains a caller assertion and must be derived and
audited for each protocol.  Source-radius and outer-boundary events are
operational model exits, not claims of collision or definitive ejection.  The
current route is not a regularized close-encounter integrator.

## State and units

`CUDAMasslessTracerState` requires:

- explicit frame, origin, axes, time scale, unit-system ID, and provenance;
- `length_unit="AU"` and `time_unit="year"`;
- one explicit primary source in slot zero;
- native C-contiguous CuPy float64 source/tracer arrays on the selected device;
- positive source GMs, nonnegative encounter radii, and positive acceleration
  bounds;
- stable unique uint64 tracer identities.

Tracer arrays never compact.  Each lane has an immutable identity and a
lifecycle code: active, removed at a source-radius surface, or removed at the
primary outer boundary.  A removed lane retains its continuously located event
position/velocity and is skipped forever.  Its ledger stores the exact source
segment preimage, event state, effective tolerance, and primary-relative
diagnostics; reload recomputes the source-segment hash, Hermite event state,
surface geometry, and frozen lane identity.

Every CPU-route row likewise retains its source and tracer pre/post numerical
preimage plus the locator policy.  Continuation and checkpoint validation
recompute the CUDA guard decision and deterministically replay the DOP853 event
locator.  The four-arm runner adds two stronger, experiment-level gates: every
routed source segment must occur byte-for-byte in a regenerated all-horizon
source path, and every routed tracer is rerun from its frozen initial state over
the complete stage.  These are same-code lineage audits, not independent
reproduction.

The generic checkpoint API alone verifies each route row's internal preimage,
guard decision, and locator output; it does not prove that an otherwise valid
row belongs to the named tracer's quiet-path history.  A scientific runner must
therefore supply the full-stage routed-tracer lineage gate used here (or an
equivalent predecessor/genesis replay).  A bare engine checkpoint is not a
standalone scientific evidence package.

Finite positive-energy tracer states remain valid and are recorded by the
`ever_unbound` diagnostic.  This flag is a primary-relative two-body energy
check at committed lattice nodes, including CPU-route endpoints.  It is not a
multi-source total-energy test and can miss a within-step sign change.  Such
states are not numerical failures and are not removed automatically.

For a rejoined route, the cumulative minimum combines GPU lattice nodes with
all DOP853 stages.  For a terminal route it includes only the committed prefix
through the located event; discarded post-event solver work is retained only
as audit telemetry.  It is not a continuously located periapse except at an
explicitly located terminal surface.

## Checkpoint and restart

Checkpoints are explicit device-to-host operations.  Every checkpoint is an
immutable NPZ/JSON pair with:

- full source and tracer endpoint state;
- stable tracer identities;
- cumulative minimum-distance, unbound, and encounter-route diagnostics;
- fixed tracer lifecycle plus complete encounter-route and removal-event
  ledgers;
- state, file, protocol, CUDA-kernel, Python implementation, NumPy/SciPy,
  CPU-platform, CUDA runtime, local toolkit, NVRTC compiler, driver,
  physical-GPU UUID, and device identities;
- the exact semantic hash of the state from which the segment began and an
  immutable system/population lineage digest;
- a mandatory predecessor content digest after the first checkpoint.

Checkpoint numbers use 20 decimal digits and are also selected numerically, so
restart remains correct beyond index 999.  The complete predecessor-record
chain and every historical NPZ size/hash are checked on append and load.  Every
segment's input-state commitment must equal the preceding endpoint.  Immutable
source/population identity must remain fixed; route ledgers must be exact prefix
extensions; route counts, unbound flags, and minimum-distance diagnostics must
remain cumulative.  Every historical record must use the same method, runtime,
compiler, implementation, and all-false claim controls.  Corruption,
truncation, false linkage, protocol mismatch, history reset, claim widening,
diagnostic-ledger mismatch, or runtime/device/implementation mismatch fails
closed.  Final file publication is create-only, so concurrent writers cannot
overwrite a committed checkpoint.  Each run/protocol pair has a hash-derived
namespace, and arrays use content-addressed names.  If a process dies after
publishing an array but before the JSON commit marker, the orphan is ignored
and the same checkpoint slot can be retried without deleting evidence.

On the same runtime and device, split and uninterrupted runs have bitwise-
identical numerical source/tracer buffers and cumulative diagnostics in the
focused test.  The convenience floating `epoch` field is not claimed bitwise
partition-invariant; integer `global_step` is the authoritative lattice clock.

## Minimal use

```python
from jxplanetx.cuda_massless_tracer import (
    CUDAMasslessTracerState,
    FixedStepCUDAMasslessTracerSpec,
    advance_cuda_massless_tracers,
)
from jxplanetx.engine import BackendSpec

spec = FixedStepCUDAMasslessTracerSpec(
    step_years=0.125,
    step_count=20_000,
    backend=BackendSpec("cupy", "cuda:0", 128),
)
result = advance_cuda_massless_tracers(initial_state, spec)
next_state = result.state
```

The caller owns construction and provenance of `initial_state`.  No current
Rubin orbit, retired candidate-9118 state, present-epoch synthetic population,
or old H1 workspace is silently selected by this API.

## Path to the 4.5-Gyr experiment

The following work remains before authorizing the intended 4.5-Gyr study:

1. Replace the synthetic current-epoch engineering population with a
   self-consistent, content-addressed initial population appropriate to the
   scientific question.
2. Qualify the 0.125-year direct-KDK screen against a 0.0625-year subset or
   replace the source map; the present method is not a long-horizon Wisdom--
   Holman map.
3. Add Galactic tide and stellar-history terms where flyby or very distant
   trajectories require them.
4. Replace the circular distant-planet sensitivity point with a predeclared
   eccentric six-element family before testing canonical angular confinement.
5. Add the required population/angular observables and an independent
   REBOUND/MERCURIUS comparison subset.
6. Advance through 1-Myr, 10-Myr, 100-Myr, and 1-Gyr authorization rungs before
   4.5 Gyr.  No rung may be inferred from throughput alone.

At the measured 16,800-tracer rate, a simple 4.5-Gyr, 0.5-year-step arithmetic
projection is about 54.4 GPU-days per arm before checkpoint, output, route,
force-schedule, and qualification costs.  Four such arms would therefore be a
months-long local campaign.  This projection is planning information only;
small populations and encounter-rich histories do not scale linearly.

## Explicit nonclaims

The current backend and acceptance result do not establish:

- accuracy or stability over 4.5 Gyr;
- a real, primordial, Rubin-selected, or survey-debiased TNO population;
- Planet Nine detection, exclusion, orbit, mass, or model preference;
- a stellar flyby or rogue-planet history;
- a physical collision, definitive ejection, or complete encounter history;
- a globally symplectic hybrid method;
- independent reproduction or cross-device bitwise reproducibility.
