# JX Planet X Scientific Engine

**Version:** 0.1.1  
**Release date:** 21 Aug 2026

JX Planet X is a falsification-first numerical and inference framework for
testing physical sources of outer-Solar-System gravity. A planet is one
candidate source, not the assumed answer.

**Current scientific status:** the numerical foundation passed its encoded
production gate, but the latest 100-kyr compact-source population did not
converge. Its governing result is
`BLOCKED_SOURCE_POPULATION_NONCONVERGENCE`; this release does not claim a
Planet X detection or sky localization.

**0.1.1 provenance hotfix:** installed-package runs now hash the executing
packaged Python sources instead of allowing an empty software manifest.
Numerical dynamics are unchanged.

This repository begins with the validated numerical foundation:

- deterministic arbitrary-precision arithmetic using Python's `decimal`;
- a sixth-order symmetric Yoshida composition of seven KDK maps;
- eight force evaluations per macro-step;
- convergence and conservation gates;
- cryptographic run manifests and evidence-class labels;
- claim controls that cannot automatically declare a detection.

The current engine is **not** a production orbit-determination system. It does
not yet implement complete DSN uplink/downlink observables, ramp tables,
troposphere/ionosphere corrections, full light time, or a simultaneous global
ephemeris fit. Results from the present core must therefore remain
`SCREENING_ONLY` unless an external observation module satisfies the locked
gates in `docs/SCIENTIFIC_CONTRACT.md`.

## Run

No third-party Python packages are required.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m jxplanetx.cli validate --output runs/validation.json
PYTHONPATH=src python3 -m jxplanetx.cli reproduce-yoshida6 \
  --bundle-dir imports/yoshida6_gate \
  --output runs/yoshida6_reproduction.json
```

Run these commands from the repository root with `PYTHONPATH=src`, as shown
above. The tested baseline is Python 3.12. Version 0.1.1 passes all 16 unit
tests.

The validation command runs a sixth-order harmonic-oscillator convergence
test, a two-body conservation test, and writes a hashed JSON record.

`reproduce-yoshida6` accepts only the preserved synthetic benchmark whose checksum
manifest is locked in `benchmarks/yoshida6_2026_08_20.lock.json`. It reruns the
160- and 224-bit executable, requires byte-exact dynamical state fields plus
numerically gated derived angles, audits performance/invariants/force counts,
and independently compares the result to the preserved 224-bit Bulirsch-Stoer
trajectory.

This benchmark is barycentric and J2000-ecliptic-like, but it is not a DE441
state. A true DE441 state import must be implemented and validated separately.

The `import-de441-anchor` command performs that separate import. It selects
the five preserved DE441 massive-body states at TDB JD 2461200.5 and a declared
15-tracer synthetic subset.

`run-de441-anchor-gate` executes the locked Yoshida-6 binary at 160 and 224
bits on that state. Passing cross-precision and invariant gates is recorded as
`PARTIAL_PASS_INDEPENDENT_REFERENCE_MISSING` until the independent gate runs.

`run-bs-block-reference` supplies a separate 78-decimal-digit adaptive
Bulirsch–Stoer implementation. Because the tracers are exactly massless, it
integrates five massive bodies plus one tracer per block, then verifies the
independently repeated massive trajectories before merging. The preserved
100-year reference is compared with the 224-bit Yoshida trajectory by
`audit-de441-independent-reference`. Passing this closes the numerical gate as
`PRODUCTION_NUMERICAL_GATE_PASSED`; it does not qualify an observation model or
constitute evidence for a physical source.

## 100-kyr equation gate

The adaptive `run-ias15-member` and population-comparison commands resume the
anomaly-zone family calculation from the preserved DE441 Cartesian states.
They compare REBOUND 4.4.11 IAS15 runs at `epsilon=1e-12` and `1e-14` using
pointwise and population equations.

The untouched phase-2 no-source population converged, but the middle-family
source population did not: maximum mean-perihelion and Wasserstein-perihelion
disagreements were 0.145672 AU and 0.155080 AU against locked 0.1-AU gates.
The governing result is therefore
`BLOCKED_SOURCE_POPULATION_NONCONVERGENCE`. The resolved source-minus-control
sensitivity is not scientifically usable because its prerequisite source-run
convergence failed. See `docs/JX_IAS15_EQUATION_GATE_2026-08-20.md`.

The optional preserved dependency can be installed locally with
`python3 -m pip install --target .vendor imports/Planetx/rebound-4.4.11-cp312-cp312-linux_x86_64.whl`;
then run IAS15 commands with `PYTHONPATH=.vendor:src`.

## Architecture

```text
src/jxplanetx/
  decimal_math.py   precision and vector primitives
  dynamics.py       deterministic N-body acceleration and invariants
  yoshida6.py       optimized sixth-order symplectic stepper
  decimal_bs.py     independent adaptive Bulirsch–Stoer reference
  ias15_gate.py     adaptive close-encounter and population equation gates
  gates.py          scientific validation gates
  claims.py         evidence labels and claim-control state machine
  provenance.py     canonical JSON, hashes, and atomic run records
  cli.py            reproducible validation entry point
```

## Scientific status

The software encodes the project's present conclusion: no public-data branch
has passed the physically realistic blinded six-mode measurement gate. A
compact distant source remains a candidate class, not a detection or sky
localization.
