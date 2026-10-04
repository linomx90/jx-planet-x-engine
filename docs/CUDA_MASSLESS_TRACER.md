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

The preliminary retained RTX measurements used the Sun and four giant planets with
16,800 deterministic synthetic tracers for 5,000 years at a 0.5-year step.  It
completed 168,000,000 tracer-steps in 5.425 seconds (30.97 million
tracer-steps/second) on the RTX 5060 Ti.  All endpoint states were finite and
all 522 guarded steps were transactionally resolved.  Those numbers are an
engineering measurement for superseded pre-audit bytes, not current-code
acceptance, a portable speed guarantee, or scientific validation.  The final
v3 evidence bundle is generated only after source freeze and includes the exact
protocol, test transcript, raw input and endpoint states, route ledger,
environment, result, and run-wide checksums.

## Numerical architecture

For each bounded chunk, the backend:

1. Advances the active point-mass sources once with a barycentric inertial,
   fixed-step kick--drift--kick map.
2. Retains immutable source position, velocity, and acceleration nodes for the
   chunk.
3. Advances independent CUDA tracer lanes against that one shared source path.
4. Stops a lane before committing any step whose swept-distance certificate is
   unavailable.
5. Redoes the complete untouched outer step with SciPy DOP853 against the same
   cubic-Hermite source segment, then rejoins the CUDA path at the exact next
   lattice node.

Tracers never accelerate one another and never affect a source.  Source bytes
therefore do not depend on tracer count, lane order, CUDA block partitioning,
or encounter routing.  The focused tests exercise 0, 1, 16, 65, and 129 tracer
layouts and compare shared source endpoints bit for bit.

The route is intentionally called an `encounter_route`, not a backend fallback.
The configured CuPy backend remains the integration backend.  A guard switch
means only that the quiet CUDA step was not certified; it is not evidence of a
physical encounter, contact, collision, or ejection.

## Guard contract

Each source has two caller-bound device arrays:

- an encounter radius in AU;
- a positive upper bound on relative acceleration in AU/year² over one outer
  step.

The quiet route requires a positive full-step separation lower bound after the
relative-speed and acceleration envelopes, and also checks the minimum of the
linear swept segment.  Equality routes.  If the instantaneous relative
acceleration already exceeds the declared bound, the lane routes rather than
committing.

The acceleration envelope is a caller assertion.  It must be derived and
audited for each scientific protocol.  V1 does not yet prove a global bound
from the force model, perform continuous collision root finding, or impose the
planned central-periapse resolution guard.  These are authorization blockers
for the 4.5-Gyr science run, not hidden assumptions.

## State and units

`CUDAMasslessTracerState` requires:

- explicit frame, origin, axes, time scale, unit-system ID, and provenance;
- `length_unit="AU"` and `time_unit="year"`;
- one explicit primary source in slot zero;
- native C-contiguous CuPy float64 source/tracer arrays on the selected device;
- positive source GMs, nonnegative encounter radii, and positive acceleration
  bounds;
- stable unique uint64 tracer identities.

Finite positive-energy tracer states remain valid and are recorded by the
`ever_unbound` diagnostic.  This flag is a primary-relative two-body energy
check at committed lattice nodes, including CPU-route endpoints.  It is not a
multi-source total-energy test and can miss a within-step sign change.  Such
states are not numerical failures and are not removed automatically.

The cumulative minimum distance combines GPU lattice-node evaluations with
CPU DOP853 derivative-evaluation stages.  It is an evaluated-point diagnostic,
not a continuously located periapse, encounter distance, or collision event.

## Checkpoint and restart

Checkpoints are explicit device-to-host operations.  Every checkpoint is an
immutable NPZ/JSON pair with:

- full source and tracer endpoint state;
- stable tracer identities;
- cumulative minimum-distance, unbound, and encounter-route diagnostics;
- the complete encounter-route ledger;
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
    step_years=0.5,
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

The following work remains before authorizing the intended mechanism study:

1. Add and test the central-periapse step guard and removal/event surfaces.
2. Freeze one self-consistent primordial source state and tracer population.
3. Implement closed, hashed force schedules rather than arbitrary callbacks:
   persistent Planet Nine, a whole-system stellar-flyby window, and an active
   temporary-rogue lifecycle.
4. Add Galactic tide and stellar-history terms where the interpretation needs
   them.
5. Qualify 0.5-year versus 0.25-year population observables and an
   encounter-rich REBOUND/MERCURIUS subset.
6. Advance through 1-Myr, 10-Myr, 100-Myr, and 1-Gyr authorization rungs before
   4.5 Gyr.  No rung may be inferred from throughput alone.

At the measured 16,800-tracer rate, a simple 4.5-Gyr, 0.5-year-step arithmetic
projection is about 56.5 GPU-days per arm before checkpoint, output, route,
force-schedule, and qualification costs.  Four such arms would therefore be a
months-long local campaign.  This projection is planning information only;
small populations and encounter-rich histories do not scale linearly.

## Explicit nonclaims

The current backend and acceptance result do not establish:

- accuracy or stability over 4.5 Gyr;
- a real, primordial, Rubin-selected, or survey-debiased TNO population;
- Planet Nine detection, exclusion, orbit, mass, or model preference;
- a stellar flyby or temporary rogue;
- a collision, ejection, or continuously resolved encounter;
- a globally symplectic hybrid method;
- independent reproduction or cross-device bitwise reproducibility.
