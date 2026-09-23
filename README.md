# JX Celestial Dynamics Framework

**Version:** 0.6.0rc10

**Release date:** 2026-09-23

**License:** Proprietary — all rights reserved
**Scientific claim state:** `SCREENING_ONLY`

JX is a falsification-first, general celestial-dynamics framework for explicit
physical models, reproducible numerical experiments, and scientific claim
control. Outer-Solar-System and Planet X studies are important applications,
not assumptions built into the engine. Model output is not astronomical
evidence.

This repository contains the engine source, packaging metadata, unit tests,
locked protocols, and compact result summaries. It intentionally excludes the
project's large observational inputs, bulk execution archives, and
candidate-search catalogs.

The [experiment data index](docs/JX_EXPERIMENT_DATA_INDEX.md) provides the
non-destructive map of the local immutable library, active runs, benchmark
definitions, result bundles, sibling archives, and metadata gaps. It preserves
the library authority catalog as the source of truth and does not promote
workspace results.

The installable distribution and command remain `jxplanetx` for backward
compatibility. Version 0.6.0rc10 carries the immutable General Dynamics
registry v18 evidence historically bound to rc3, retains the supported
measured small-N CPU/CUDA interfaces, and advances the supported fast
Wisdom--Holman CPU screen to a checkpoint-synchronized native map. Its default
single execution produces a complete-map postcondition certificate; an exact
second execution remains opt-in and is required by release qualification.
The packaged endpoint-seeded v4 solver remains an implementation-private,
fallback-audited research prototype. Byte-reproducible wheel and source-archive
construction and the existing force and trajectory surface are preserved. Rc10
carries the prospectively qualified GR15 V3 CPU component while retaining
the exact V2 implementation as a private fallback/reference.
Registry v18
states `SCREENING_ONLY` explicitly and does not promote any capability beyond
its exact evidence and claim ceiling. The legacy propagation core, CLI, locked
experiments, and scientific records remain unchanged.

## General engine

The additive `jxplanetx.engine` API evaluates ordered acceleration terms on
caller-owned, backend-native arrays and advances explicitly requested
checkpoints with an adaptive RKF78 method. It implements exactly three
unqualified physics models:

- direct, unsoftened Newtonian point-mass gravity;
- restricted static-central Schwarzschild test-particle 1PN, as a
  correction-only term; and
- unshadowed isotropic cannonball solar-radiation pressure.

`integrate_trajectory` implements the 13-stage Fehlberg RK7(8) pair from
NASA-TR-R-287. It advances the hatted order-eight solution, uses the ordinary
order-seven solution for the defect, applies a normalized maximum
per-component error test, and reaches each forward or backward checkpoint by
step clipping without interpolation. Every force and trajectory result remains
`MODEL_OUTPUT`, with registry and qualification authority fixed false. Inputs
bind units, frame, origin, epoch, provenance, parameter validity, backend,
device, binary64 dtype, deterministic reduction scope, summation tile size,
error tolerances, and every controller limit. Unsupported combinations fail
before arithmetic; backend selection never silently falls back or transfers
arrays between host and GPU.

The rc10 candidate packages the qualified V3 Gauss--Radau order-15
numerical core behind the supported `jxplanetx.gr15` CPU interface and retains
the bitwise-validated V2 implementation as an exact private reference.
`integrate_gr15` advances 2--32 fully mutual, positive-GM, unsoftened
Newtonian point masses in binary64, exposes complete controller and predictor
accounting, and fails with named statuses. It is built into the wheel and does
not compile code at runtime or import the benchmark tree. Its scope remains
`SCREENING_ONLY`, and no production, ephemeris, collision-handling, broad
force-model, or general IAS15 superiority claim is authorized. See the
[GR15 user and contract guide](docs/JX_GR15.md).

The preregistered V3 correctness portfolio now passes analytic eccentric and
extreme-mass-ratio binaries, a three-body close scatter, a 20-period 11-body
hierarchy, and the 32-body boundary against strict REBOUND 5.1.1 IAS15
primary/sensitivity references. The whole report repeats byte-for-byte. This
extends screening coverage; it does not make GR15 a production ephemeris or
authorize a general accuracy or speed claim.

The additive [Dynamics scenarios V1](docs/DYNAMICS_SCENARIOS.md) layer now
assembles one state, force plan, and adaptive-RKF78 request as an explicit
scenario. Its matched mode runs a same-state, same-solver control followed by a
candidate that appends named force terms, retaining both trajectories and
per-body checkpoint differences. The candidate starts from the control run's
retained baseline copies. Those differences are model-to-model diagnostics,
not accuracy, improvement, or physics claims; every result remains
unqualified `MODEL_OUTPUT`. A canonical portable manifest records NumPy/CPU
binary64 scenarios with hexadecimal floats and an external SHA-256 identity;
loading reconstructs read-only arrays and does not authenticate or qualify the
scenario.

`integrate_encounter_segment` adds a standalone full-Cartesian adaptive
RKF78 segment for exact NumPy/CPU binary64, barycentric-inertial, fully mutual,
all-active positive-GM unsoftened Newtonian states. It advances an
authoritative signed mathematical duration on exact local offsets; its
separately supplied endpoint epoch is an independent provenance label, not the
source of the state duration. Pair-relative and GM-centroid defect control,
Kahan accepted updates, exact pair floors, bounded exact-arithmetic witnesses,
transactional rejection, and one separately accounted mandatory semantic
replay are explicit public contracts.

The encounter certificate proves only that the exact Newtonian local IVP
issuing from each accepted numerical node does not cross the retained pair
floors during that accepted substep. A failed certificate is merely
uncertified; it is not collision evidence. The solver does not locate events
or minimum distance, respond to collisions, regularize, switch integrators,
or claim clearance of a global physical trajectory, and every result remains
unqualified `MODEL_OUTPUT`.

Two additive fixed-step NumPy/CPU maps are also public and unqualified:
fully mutual Cartesian KDK, and an ordered-Jacobi Wisdom--Holman KDK map for
all-active, positive-GM, low-secondary-mass hierarchical elliptic systems.
The Wisdom--Holman path uses analytic universal-variable Kepler subflows,
integer map-index checkpoints, mandatory hierarchy/periapse/step/Hill/path
guards, and one full semantic validation replay. It has no passive tracers,
close-encounter switching, adaptive fallback, or production claim. Its formal
exact-subflow symplectic statement does not apply to floating-point execution;
small sampled energy error does not imply small phase or trajectory error.

The opt-in `jxplanetx.fast_wisdom_holman` interface accepts the same validated
state, force-plan, and fixed-step specification contracts, then executes the
complete map and numerical guards in one compiled CPU call. Its default
single execution returns a complete-map postcondition certificate; callers may
explicitly request an independent exact replay, and release qualification does
so. The faster universal-G evaluation stops only when the next binary64 term
cannot change the accumulated value and keeps the fixed 64-term rc5 evaluator
as its exact fallback. This API remains
`SCREENING_ONLY`, supports at most 32 bodies inside the existing narrow
hierarchical elliptic domain, and adds no close-encounter switching,
qualification, production authorization, or general speed claim.

`integrate_hybrid_wisdom_holman_rkf78_trajectory` implements the narrowly
defined
`integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1` method for at most
16 bodies. Every fixed outer interval first
runs one typed, provisional Wisdom--Holman far probe. A far pass commits that
candidate. A finite named near exit discards the complete candidate and reruns
the untouched original node over the full interval through the private guarded
Cartesian encounter execution; static, numerical, resource, custody, and
replay failures are fatal. The outer integer lattice never adapts, and there
is no hysteresis, latch, partial-prefix commit, grouping, or event location.

The near certificate keeps exactly the standalone solver's local-IVP scope;
it is not collision or event detection and gives no global clearance claim.
One full hybrid semantic replay, with no nested public child replay, recomputes
all states, decisions, ledgers, digests, and accounting. All-far runs preserve
the frozen public Wisdom--Holman state/checkpoint/diagnostic/accounting
projection bit-for-bit. The complete hybrid is not globally symplectic,
formally or exactly reversible, regularized, globally order-qualified, or
scientifically qualified.

The accompanying 46-capability catalog contains exactly twelve `IMPLEMENTED`
and 34 `DECLARED` rows and covers the alpha surface and broader JX roadmap.
The implemented set includes the standalone guarded encounter segment, the
two experimental unqualified fixed-step Newtonian maps, and the one specific
whole-step hybrid. The broader roster spans
gravity, relativity, harmonics, tides,
nongravitational forces, encounters, integrators, accelerator backends,
precision modes, sensitivities, orbit determination, and measurements. Every
entry is `UNQUALIFIED`. A `DECLARED` entry validates its known parameter roster
and then refuses execution.

NumPy CPU and optional CuPy CUDA 12/13 code paths are available. The
[GPU scientific ladder V1](runs/jx_gpu_scientific_ladder_v1/README.md) records
fixture-specific CPU/CUDA parity, the complete live-engine test scope, a
reduced eleven-body replay, descriptive interleaved timings, and a large
mutual-gravity CUDA scale observation on one project-owned RTX 5060 Ti. That
package does not establish reproducibility across devices, a portable speedup,
general GPU qualification, production fitness, or physical accuracy.

For `0.6.0rc1`, a fail-closed current-core gate was also run on the same model
of project-owned GPU: an RTX 5060 Ti with driver 595.91.07, CUDA 13.2, NumPy
2.3.5, and CuPy 14.2.0. All 13 isolated engine modules passed with the hardware
tests active (233 tests, zero skips), and the five GPU-specific backend and
trajectory tests also passed under optimized Python. This verifies the exact
recorded runtime/device core contracts only, including CPU/GPU numerical
parity, device residency, no implicit transfer, mixed-input refusal, float32
refusal, and CuPy trajectory custody. It is not cross-device qualification, a
performance or speedup claim, production fitness, or scientific validation.

The additive [eleven-body portable evidence V1](runs/jx_eleven_body_portable_v1/README.md)
provides the compact Ubuntu workflow: offline integrity verification, exact
dependency records, a host-readiness doctor, a fresh bounded CPU/GPU replay,
and the saved 100-year REBOUND comparator. It remains `SCREENING_ONLY`; the
historical 100-year result is offline-verifiable but its fresh standalone
rerun is explicitly blocked by incomplete archived JX source closure.

The additive [R0 source-closed 100-year confirmation package](runs/jx_reph_v1_r0_source_closed_100y/README.md)
resolves that archived source closure from a provenance-bound Git bundle plus
the four accepted comparator overlays. Its bounded construction preflight
passes, but the 100-year pair has not yet run. Any future run remains
`SCREENING_ONLY`, requires a fresh monitored preflight and short-lived
authorization, and must pass the separate offline execution verifier before
its result is described as verified.

The additive [cosmology particle-mesh foundation](runs/jx_cosmology_pm_foundation_v1/README.md)
extends the same project with a native NumPy/CuPy periodic collisionless-gravity
solver in a prescribed flat expanding background. Its frozen analytic
growing-mode benchmarks pass at 128^3 and 256^3 particles, with
near-second-order spatial and temporal convergence, CPU/GPU parity, and an
independently coded planar NumPy reference. The
[256^3 validation package](runs/jx_cosmology_pm_256_independent_r2/README.md)
retains the complete cross-code evidence. The first 256^3 attempt is retained
as a stopped accumulator failure; the repaired run keeps its thresholds and
workload unchanged. This is a qualification of one controlled
foundation workload only. It is not a realistic structure-formation run and
does not include gas, radiation, plasma, nuclear physics, general relativity,
an observational fit, or a production cosmology claim.

The [R4 dark-matter plus baryon benchmark](runs/jx_cosmology_baryons_3d_r4/README.md)
adds a native 128^3 adiabatic gas mesh coupled to 2,097,152 dark-matter particles
through the same spectral gravity field. It validates three-direction linear
and complex phase, periodic conservation, gas positivity, sound-wave
convergence, a Sod shock, exact GPU repetition, and NumPy/CuPy full-state
agreement. The stopped R2 and R3 records preserve discovery of a half-cell
particle/gas phase error, an unsafe multidimensional CFL bound, and the
corrected 64^3 accuracy failure. R4 keeps the accuracy threshold unchanged and
passes at the convergence-selected resolution. This is a controlled
first-order supercomoving validation, not yet an externally validated nonlinear
galaxy-formation or production cosmology solver.

## Scientific boundary

The code does **not** claim a Planet X detection, sky position, mass, or
distance. Numerical simulation is not astronomical measurement. The engine's
claim-control logic keeps ordinary numerical output at `SCREENING_ONLY` and
blocks observational claims when required evidence gates are absent or fail.

The source includes:

- deterministic arbitrary-precision arithmetic using Python `decimal`;
- N-body acceleration, invariants, and state objects;
- a sixth-order symmetric Yoshida integrator;
- an independent Decimal Bulirsch–Stoer reference integrator;
- optional REBOUND trajectory, IAS15, and large massless-population scale gates;
- deterministic uncertainty/phase ensemble plans with locked contracts;
- paired source/control population validation across numerical methods;
- perihelion, injection, survival, inclination-width, and Wasserstein metrics;
- fail-closed `PASSED`, `BLOCKED`, and `INVALID` ensemble verdicts;
- convergence, conservation, provenance, and claim-control utilities;
- a prelocked ten-year JPL Horizons/DE441 outer-planet compatibility test;
- a real-epoch, checkpointed, matched 100,000-tracer-per-arm population screen;
- an independent SciPy DOP853 force, integration, checkpoint, and replication path;
- a pinned official OSSOS telescope-selection adapter with deterministic paired
  populations, checkpointed execution, calibration/power tests, and fail-closed verdicts;
- a command-line interface for reproducible validation workflows.

The most important present limitation is that no general engine model or
integrator is scientifically qualified. Exact benchmark qualifications do not
promote the wider engine. The RKF78 path is explicit and
nonstiff, with no dense output, event location, collision response,
or regularization. The specific hybrid adds guarded whole-step mode selection,
not a collision/event system or a general close-encounter framework. The
unchanged native propagation core remains Newtonian point-mass dynamics. JX has no qualified built-in
relativity, oblateness, nongravitational-force, collision-regularization,
orbit-determination, or observational-ephemeris capability.

## JX V5 physics foundation

The V5 foundation begins the controlled expansion beyond gravity-only
propagation. Its force-parameter registry covers relativity, gravity harmonics,
nongravitational forces, physical-body properties, collision capability, and
future measurement modeling. Every declaration is bound to units, frames,
epochs, provenance, uncertainty, covariance, applicability, and validation
requirements.

This foundation is deliberately `DRAFT_NONEXECUTABLE`. Unresolved scientific
values are recorded as `TBD_BLOCKED`; they are never replaced by guessed
defaults. Declared-but-unimplemented models cannot authorize a run or increase
the scientific claim state. See the
[V5 force-parameter registry](docs/FORCE_PARAMETER_REGISTRY_V5.md).

V5 now includes a JX-owned Decimal reference kernel for the
restricted Solar Schwarzschild 1PN correction. It is equation-level,
correction-only, and permanently reports `registry_authorized=false`. It is not
wired into the existing Yoshida integrator because the relativistic term is
velocity-dependent and that integrator assumes a separable, position-only
force. A separate fixed-step Decimal implicit-midpoint path now tests the
six-component restricted Newtonian-plus-1PN first-order equation with an explicit precision, rounding,
force-ledger, and nonlinear-residual contract. That path is reference-only,
retains `MODEL_OUTPUT`, and cannot make the draft registry executable. Full
N-body EIH relativity, production propagation, and scientific claim promotion
remain blocked. See the
[V5 reference-integrator protocol](docs/V5_REFERENCE_INTEGRATOR.md).

A separate [Solar 1PN qualification package](runs/v5_solar_1pn_qualification/README.md)
now freezes the source artifacts, unit and time-scale transformations, fresh
holdout fixtures, Q0--Q9 gate definitions, prior-development disclosure, and
claim ceiling needed for a later qualification attempt. The package contains
no outcomes and exposes no trajectory runner: its execution implementation is
`NOT_REGISTERED`, its evidence ceiling is `MODEL_OUTPUT`, and even a future
all-gates pass could make the reference path only
`ELIGIBLE_FOR_REVIEW_NONAUTHORIZING`. It cannot change the draft registry or
authorize physical Solar-System propagation.

### Step 5 Solar-System preparation

The opt-in, unpublished Step 5 boundary verifies isolated CSPICE/DE440s
geometric-state evidence and the retained DE440 GM parameter artifact, resolves
Earth and Moon into an exact eleven-body roster, converts the state and
parameters to SI, and recenters them at the selected model's operational-GM
barycenter. It then creates fresh owned read-only NumPy arrays, an engine state,
and one direct mutual unsoftened Newtonian force plan with a checked primary and
preparation replay.

This boundary prepares initial inputs only; it does not run a trajectory. The
reduced Newtonian model is not DE440 itself, the model-barycenter origin is not
NAIF body 0, and the frame crosswalk is nonauthorizing. The artifacts and
receipts provide unauthenticated content integrity, not source authority,
continuous custody, physical accuracy, Wisdom--Holman domain qualification, or
scientific/production authorization. Outputs remain unqualified `MODEL_OUTPUT`.

### Challenger V1 and post-V5 precision probe

[JX Challenger V1](docs/JX_CHALLENGER_V1.md) runs three locked smoke fixtures in
separate subprocesses: an analytic binary, a weak hierarchy, and a close
scatter. It retains each solver's native metrics and can add optional exact
REBOUND 5.1.1 lanes. It creates no common score or ranking and makes no
equivalence, accuracy-superiority, speed-superiority, or timing-comparability
claim. Its reports remain unauthenticated, unqualified `MODEL_OUTPUT`.

The additive post-V5 precision probe evaluates only the exact equal-mass
circular-binary fixture at P/256, P/512, and P/1024. It records one- and
100-period JX KDK results, optional exact REBOUND 5.1.1 leapfrog results, a
specified analytic oracle, complete lane replays, sampled invariants, and
fixture-specific empirical state/phase refinement. It does not establish a
general or theoretical convergence order, continuous-time energy bounds,
long-term stability, REBOUND superiority or equivalence, scientific
qualification, or production fitness; its reports remain unauthenticated,
unqualified `MODEL_OUTPUT`.

### Packaged native orbital benchmark

The [portable-scenario equal-binary qualification](runs/jx_dynamics_scenario_equal_binary_v1/README.md)
freezes three content-addressed public `DynamicsScenario` inputs, the complete
engine source used to execute them, a prewritten analytic oracle and 17 gates,
three byte-identical accepted reports, and an independent offline verifier.
On that locked two-body fixture, RKF78 passes the absolute, refinement,
conservation, symmetry, accounting, manifest, and reproducibility checks. This
is a fixture-specific numerical qualification, not a general N-body,
eleven-body, lunar-rotation, or production claim.

The [100-period equal-binary reproducibility package](runs/jx_equal_binary_100_period_repro_v1/README.md)
turns one existing benchmark into a self-contained, offline artifact. It
includes the exact JX source snapshot, frozen inputs and gates, the complete
path-normalized reference result, an independent checksum/result verifier, and a
create-only reproduction runner. Native KDK and RKF78 both pass their declared
analytic-workload gates; the reproduced scientific fingerprint matches the
reference exactly on the recorded runtime. REBOUND is deliberately disabled,
timings are excluded, and the result makes no general N-body, superiority, or
production-qualification claim.

The [orbital validation ladder v4](runs/jx_orbital_validation_ladder_v4/README.md)
combines that analytic foundation with a fresh one-day, DE440-initialized
resolved-eleven-body Earth–Moon J2 fixture and an accepted broader J2 study.
The third tier covers three start epochs, 1-, 7-, and 30-day horizons, and a
900-to-450-second step-size sensitivity check. The top-level runner executes
the first two trajectories from fresh state and labels the multi-epoch tier as
preserved accepted evidence. All three components retain separate claim
boundaries: there is no cross-tier score, no long-term qualification, and no
lunar fluid-core result. Its machine-readable scientific acceptance matrix
marks engineering reproducibility `PASS` while retaining scientific
qualification as `NOT_QUALIFIED` and the project claim state as
`SCREENING_ONLY`.

The [GPU scientific ladder V1](runs/jx_gpu_scientific_ladder_v1/README.md)
extends the engineering evidence across the live NumPy and CuPy paths. On its
recorded runtime, all 228 engine tests pass in normal and optimized Python, and
the frozen one-day reduced eleven-body RKF78 endpoint is bitwise identical on
CPU, repeated GPU runs, and the preserved CPU reference. Its timing and large
CUDA rows are descriptive only, and it does not decide lunar rotation or any
other physical model.

## Requirements

- Python 3.12 or newer
- No third-party dependency for the legacy core and its legacy unit tests
- Optional: `numpy>=2,<3` for the public general force and trajectory API
- Optional: CuPy 14 toolkit bundles (`cupy-cuda12x[ctk]` or
  `cupy-cuda13x[ctk]`) for the explicit CUDA path
- Optional: `rebound==4.4.11` for IAS15 and population-scale commands
- Optional: an isolated `rebound==5.1.1` environment for Challenger V1 and the
  post-V5 precision probe's external comparison lanes
- Optional: `numpy==2.3.5` and `scipy==1.17.0` for the independent DOP853 runner
- Optional: `numpy==2.3.5`, `pyerfa==2.0.1.5`, and `spiceypy==8.2.0` for the
  screening-only lunar laser-ranging transport runner

## Install and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python tools/run_local_test_matrix.py --profile engine
```

Install the pinned REBOUND backend with
`python -m pip install -e '.[rebound]'`. The legacy `ias15` extra remains an
alias for compatibility.

Install the independent population-replication backend with
`python -m pip install -e '.[independent]'`.

Install the exact LLR screening stack with
`python -m pip install -e '.[llr-screen]'`. This does not provide ephemeris,
Earth-orientation, or observation data; the runner requires separately
supplied, hash-bound read-only inputs.

Install the public CPU dynamics API with
`python -m pip install -e '.[engine]'`. For an NVIDIA GPU, install exactly one
matching wheel extra: `.[gpu-cuda12]` for CUDA 12.x or `.[gpu-cuda13]` for
CUDA 13.x. These extras request CuPy's `[ctk]` bundles; a compatible NVIDIA
driver is still required. Do not install both CuPy wheel families in one
environment. JX will not fall back to CPU if the requested GPU backend is
unavailable.

Without installing the package:

```bash
python3 tools/run_local_test_matrix.py --profile engine
PYTHONPATH=src python3 -m jxplanetx.cli validate --output validation.json
```

Build the release artifacts twice from the declared package-only input boundary
and require byte identity with:

```bash
python3 tools/build_release.py --output-dir /tmp/jxplanetx-0.6.0rc10
```

The builder requires the exact backend pinned in `pyproject.toml`, normalizes
staged source modes plus source-archive timestamps, ownership, modes, ordering,
and gzip metadata, and
writes artifact hashes to `ARTIFACTS.json`. Research evidence and frozen runs
are not distribution inputs. The distribution does include the exact audited
small-N RKF78 C source and builds it as the implementation-private
`jxplanetx._smalln_cpu` CPython extension. The supported
`jxplanetx.smalln` interface exposes only the measured 2--32-body
fully-mutual Newtonian CPU/CUDA routes, refuses implicit host/device transfers,
and remains `SCREENING_ONLY`. Consequently, wheels are tied to a
Python ABI and platform rather than tagged `py3-none-any`; building from the
source archive requires a C11 compiler and Python development headers. The
interface is not registered as a scientifically qualified or production
backend.

The rc10 source tree builds `jxplanetx._gr15_v3_cpu` from the package-owned
GR15 V3 core and exposes it through `jxplanetx.gr15`; `jxplanetx._gr15_cpu`
retains V2 for exact fallback/reference checks. The public wrapper verifies the
compiled V3 identity before execution. Source and wheel installations therefore
use the same native implementation; the old benchmark-private runtime
compilation path is not used by the supported API.

In rc5 the public CPU workspace freezes and reuses immutable RKF78 tableau,
checkpoint-epoch, and checkpoint-view metadata instead of rebuilding it for
every integration. On the recorded RTX 5060 Ti desktop host, the
accuracy-gated public CPU path had lower one-system latency than REBOUND IAS15
for the sampled 2-, 11-, and 32-body synthetic workloads. The same-host
crossover screen routes 2--31 bodies through CPU at eight or fewer independent
systems, retains CUDA for the measured 32-body/eight-system point, and routes
all measured body counts through CUDA from 16 systems. These are finite-host,
finite-workload engineering observations, not a portable speed ranking or a
general superiority claim.

The source-closed evidence tests intentionally span mutually incompatible
runtime contracts, so do not use one monolithic Python process to judge the
expanded local evidence tree. The local matrix runner isolates the current
engine tests, REBOUND 5.1.1 bridge, retained-wheel lunar checks, and exact R2
launcher runtime:

```bash
python3 tools/run_local_test_matrix.py --profile engine
python3 tools/run_local_test_matrix.py --profile evidence
```

The evidence profile verifies the retained wheel and shared-library hashes
before importing them. If absent, it reconstructs the frozen ephemeral
`/tmp/jx-reference-core-runtime` with the system Python; it never installs
packages or changes frozen run artifacts. Use `--library-root PATH` when the
read-only library mirror is mounted elsewhere.

The complete 260-file ownership and runtime policy is documented in
[JX test profiles](docs/TEST_PROFILES.md). Validate it with
`python3 tools/run_local_test_matrix.py --validate-only`, list every assignment
with `--list-files`, or run all current-source profiles with `--profile live`.

The synthetic dynamics probe is opt-in and is not part of validation:

```bash
python benchmarks/engine_force_benchmark.py --backend numpy --device cpu
python benchmarks/engine_force_benchmark.py --backend cupy --device cuda:0
```

It runs the complete three-force plan and an RKF78 trajectory, reporting
elapsed time and explicitly counted model interactions per second for one
backend. It does not compare backends or claim a GPU speedup.

The test suite covers integrator behavior,
convergence gates, force-evaluation accounting, independent-reference logic,
installed-package provenance, deterministic ensemble generation, strict
trajectory registration, distribution metrics, official OSSOS tracked-output
normalization, exact finite-pool statistics, and fail-closed verdicts.

## Command line

```bash
jxplanetx --help
jxplanetx validate --output validation.json
jxplanetx write-ensemble-contract --output ensemble-contract.json
jxplanetx prepare-ensemble --contract ensemble-contract.json --output plan.lock.json
```

The general ensemble workflow validates externally computed trajectories. The
project-specific DOP853 module now supplies an independent 10,000-year
population replication, but not a general physical state builder or complete
100,000-year source/control backend. See
[the ensemble validation guide](docs/ENSEMBLE_VALIDATION.md).

`run-population-scale-gate` is a narrower execution backend for locked,
paired, massless-tracer scalability tests. It does not turn the preserved
15-orbit template set into a physical TNO population model. See
[the population scale-gate guide](docs/POPULATION_SCALE_GATE.md).

`run-encounter-tail-pilot` runs the checkpointed 10,000-tracer controlled-
synthetic encounter-tail experiment and its prelocked timestep-halving audit.
See [the encounter-tail pilot guide](docs/ENCOUNTER_TAIL_PILOT.md).

The JX-O1 survey-selection workflow generates paired calibration populations,
executes a separately installed and hash-locked official OSSOS SurveySimulator,
normalizes its real 14-field tracked output, and evaluates calibration, power,
adapter, replay, scale, independence, and seed-stability gates. A public
preregistration and fresh official V4 execution now provide independent
computational confirmation. See
[the survey-selection validation report](docs/SURVEY_SELECTION_VALIDATION.md).

Some CLI subcommands reproduce project-specific DE441, benchmark, or IAS15
experiments. Those commands require external input bundles that are not part of
this engine-only repository. They fail closed when required manifests or inputs
are unavailable or inconsistent.

## DE441/Horizons compatibility result

The locked ten-year external-reference validation passed. With all ten major
Solar-System barycenters active, the maximum annual heliocentric residual among
Jupiter, Saturn, Uranus, and Neptune was 33.6013 km in position and
0.000510045 m/s in velocity. All predeclared convergence, conservation, and
completion gates also passed. This validates only the stated short-arc
Newtonian compatibility scope; it is not a full DE441 reconstruction or an
observational Planet X result. See the
[complete protocol and report](runs/de441_horizons_10yr/README.md).

## DE441-backed 100,000-tracer result

The locked 10,000-year source/control screen completed 100,000 matched
massless tracers per arm plus 40,000 audit trajectories. All numerical gates
passed. The control arm produced 4,377 sampled low-perihelion injections and
the candidate-9118 source arm produced 4,374. The source-minus-control fraction
was −0.00003 with a paired-block 95% bootstrap interval of
[−0.00009, +0.00003], entirely inside the predeclared ±0.001 equivalence
margin. The result is therefore `EQUIVALENT_WITHIN_LOCKED_MARGIN` and remains
`SCREENING_ONLY`—it is not a Planet X detection or exclusion. See the
[complete protocol, audit, and interpretation](runs/de441_population_100k/README.md).

## Independent DOP853 replication result

An outcome-blind SHA-256 selection of ten 1,000-tracer blocks from the
100,000-tracer experiment was independently rerun with a separate Newtonian
force implementation and SciPy DOP853. The corrective high-resolution run
passed every unchanged numerical and cross-software gate. DOP853 and REBOUND
identified exactly the same 433 injections in control and the same 433 in
source, with zero identity disagreement, 100% survival, and a paired source-
minus-control effect of 0.0 with bootstrap interval `[0.0, 0.0]`.

The first independent attempt is preserved as `INVALID`: it missed one strict
endpoint-position gate by 27.5% while all population gates passed. A locked
diagnostic attributed the miss to adaptive resolution; v2 doubled temporal
resolution without relaxing any threshold and passed. A separate artifact
audit then rehashed 100 checkpoints, reconstructed final orbital elements, and
recomputed the `PASSED` verdict. The conclusion remains `SCREENING_ONLY`. See
the [full independent replication report](runs/independent_dop853_10k/README.md).

## JX-O1 telescope-selection result

V4 independently repeated the locked calibration with fresh intrinsic
populations, official-driver seeds, resampling streams, raw outputs, and pool
hashes. It processed 33,500,335 intrinsic objects and produced 212 correct-model
and 205 deliberately wrong-model tracked detections. Every unchanged gate
passed: 4.65% false rejection, 100% wrong-model power, exact finite-pool zeta
moments, exact replay, raw-adapter identity, checkpoint replay, and stable
verdicts for all ten leave-one-block-out evaluations.

The design was published and CI-validated before V4 execution. The original V2
result remains `INVALID`, and the V3 corrective replay remains a non-independent
`PASSED` record. V4 is independent computational confirmation of the locked
telescope-selection calibration workflow—not a Planet X detection, exclusion,
or validation of a physical distant-source model. See the
[complete JX-O1 report](runs/survey_selection_o1/README.md).

## Package map

```text
src/jxplanetx/
  engine/                   public alpha force/trajectory contracts, catalog,
                            backends, kernels, evaluator, RKF78, and standalone
                            encounter-segment runtime
  decimal_math.py          precision and vector primitives
  dynamics.py              N-body acceleration, state, and invariants
  force_registry_v5.py     fail-closed V5 physics-registry inspection
  solar_1pn.py             nonauthorizing Decimal Solar 1PN reference kernel
  v5_reference_dynamics.py nonauthorizing Newtonian-plus-1PN force ledger
  v5_implicit_midpoint.py  fixed-step Decimal velocity-dependent reference
  v5_solar_1pn_qualification.py fail-closed frozen-package inspector only
  solar_system/            unpublished opt-in CSPICE/DE440s contracts and
                           resolved-eleven Newtonian input preparation
  yoshida6.py              sixth-order symmetric integrator
  decimal_bs.py            independent Bulirsch–Stoer reference
  ias15_gate.py            IAS15 and population comparison gates
  ensemble_validation.py  locked chaotic-population ensemble validation
  population_scale.py     paired large-population execution scale gate
  encounter_tail.py       checkpointed synthetic encounter-tail pilot
  de441_anchor.py          declared DE441-anchor import workflow
  de441_population.py      real-epoch paired population execution and gates
  independent_dop853.py    independent DOP853 population replication backend
  survey_selection.py      frozen v1 survey-selection audit implementation
  survey_selection_v2.py   corrected official 14-field OSSOS adapter
  survey_selection_v3.py   exact-zeta corrective replay evaluator
  survey_selection_v4.py   fresh-pool independent confirmation evaluator
  production_benchmark.py locked benchmark verification
  gates.py                 numerical validation gates
  claims.py                scientific claim-control state machine
  provenance.py            canonical hashes and atomic run records
  cli.py                   command-line interface
```

## Runtime independence

ChatGPT/JX helped develop and organize the project, but the released engine does
not require ChatGPT, an API key, or an internet connection to run its core tests
and validation command.

## Citation and license

Citation metadata and human authorship are recorded in `CITATION.cff` and
`AUTHORS.md`. Original JX material in version `0.6.0rc10` is proprietary and
all rights are reserved by Lino Avila; copying, modification, distribution,
commercial use, and derivative works require prior written authorization.
Optional third-party packages retain their own licenses. Earlier public JX
releases and frozen artifacts retain the notices that accompanied them.
The rc6, rc7, rc8, and unpublished rc9 manifests remain historical records.
The rc10 candidate receives its own manifest after final package verification;
it never overwrites or relabels earlier artifacts.

The candidate also contains opt-in, unregistered coupled lunar screening
models. The v2 boundary advances Sun--Earth--Moon translation, lunar mantle
attitude/rate, and fluid-core rate simultaneously, with solar figure torque
and a fixed-pole Earth-J2 orbital correction. Its converged one-day screen is
within 1.251 m of retained DE440 in Earth--Moon position, but this fitted-
ephemeris comparison is not independent validation and performs no raw LLR
reduction. Scope, equations, retained-input screens, omissions, and next gates
are documented in
[the coupled lunar model note](docs/JX_COUPLED_LUNAR_MODEL.md).

The rc5 source also contains separate opt-in complete delayed mantle-
deformation and source-fixed geodetic-transport components. Their frozen
90/365-day screens did not pass the registered promotion rules, so the static
mantle remains the accepted default and registry v18 keeps lunar rotation
`INCONCLUSIVE`. These reference-orbit screens are not simultaneous orbit/
rotation validation. See
[the deformation result](docs/JX_LUNAR_MANTLE_DEFORMATION_90_365_HOLDOUT.md)
and
[the geodetic result](docs/JX_LUNAR_GEODETIC_TRANSPORT_90_365_CONFIRMATION.md).

Post-rc3 research now integrates that complete delayed deformation package
inside the simultaneous orbit/rotation state as an additive v3 boundary. Its
new preregistered 180/730-day holdout passed every structural and real-CUDA
parity gate but none of eight physical promotion cells: rotational errors
improved by less than 1%, while Earth--Moon translation became slightly worse.
The frozen decision retains static v2 and stops v3 promotion. Registry v18 and
the sealed rc3 artifacts remain unchanged. See
[the simultaneous v3 result](docs/JX_LUNAR_COUPLED_DEFORMABLE_180_730_HOLDOUT.md).
