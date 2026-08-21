# JX scientific contract

## Objective

Identify which physical mass configuration, if any, can reproduce the inferred
outer-Solar-System gravitational and orbital effects. Candidate classes include
a compact planet, compact swarm, eccentric disk, distributed disk, transient
source, and the no-additional-source model.

## Evidence classes

Every value entering or leaving the engine is classified as one of:

1. `MEASURED` — directly observed with a cited instrument and uncertainty.
2. `RECONSTRUCTED` — derived from observations through a documented reduction.
3. `MODEL_OUTPUT` — produced by numerical or statistical computation.
4. `ASSUMPTION` — fixed by the analyst rather than established by the data.
5. `FORECAST` — projected sensitivity of a proposed measurement design.
6. `SPECULATION` — a hypothesis not yet supported by a qualifying test.

Simulation output is never silently promoted to measurement.

## Locked observation gates

A source direction or candidate signal is ineligible for unblinding unless all
of the following pass:

- zero-signal recovery;
- injection-density and amplitude recovery;
- numerical rank and covariance stability;
- chronological and planet holdouts;
- independent ephemeris replication;
- residual and source-precision thresholds;
- independent implementation recovery;
- global look-elsewhere correction;
- complete known-force and nuisance-parameter fit.

The production observation model must include full Solar-System dynamics, DSN
station-time round-trip observables, uplink/downlink and ramp handling, media
corrections, relativistic light time, time-scale conversion, and simultaneous
global nuisance fitting. Record count or algebraic full rank is not Fisher
information.

## Numerical gates

- deterministic initial conditions and precision;
- explicit frame, origin, epoch, and units;
- expected integrator order on an independent analytic problem;
- state convergence across step size and precision;
- energy and angular-momentum checks where applicable;
- independent integrator comparison for production claims;
- preserved raw states, configuration, code identity, and hashes.

Failed convergence invalidates the associated result.

## Claim states

- `SCREENING_ONLY`: useful for eliminating or prioritizing models only.
- `INVALID`: one or more required validity gates failed.
- `CONFLICT`: individually valid tests disagree.
- `ELIGIBLE_FOR_REVIEW`: all encoded gates passed; external scientific review
  is still required.

There is deliberately no automatic `DETECTED` state.

## Known exclusions

Do not treat a printed-digit precision ceiling as physical sensitivity. Do not
derive a sky direction from ephemeris differences. Do not use Neptune-only
astrometry, public reduced-data shortcuts, or superseded OF201 resonance
calculations as validated evidence. Do not infer mass and distance separately
from a tidal tensor alone: the leading signal constrains approximately
`M/R^3`; a gradient or wide-baseline parallax is required to break that
degeneracy.

