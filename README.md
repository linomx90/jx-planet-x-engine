# JX N-Body Engine

**Version:** 0.3.0  
**License:** MIT  
**Scientific claim state:** `SCREENING_ONLY`

JX N-Body Engine is a high-precision numerical toolkit for testing gravitational
source hypotheses in the outer Solar System. It was developed inside the JX
Planet X research project, but it is a general N-body engine: a planet is one
possible source, not an assumed answer.

This repository contains the engine source, packaging metadata, unit tests,
locked protocols, and compact result summaries. It intentionally excludes the
project's large observational inputs, bulk execution archives, and
candidate-search catalogs.

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

The software is not a complete orbit-determination or global ephemeris-fit
system. Observation-level use would additionally require validated light-time,
station, media, clock, calibration, and simultaneous-fit models.

## Requirements

- Python 3.12 or newer
- No third-party dependency for the core engine and unit tests
- Optional: `rebound==4.4.11` for IAS15 and population-scale commands
- Optional: `numpy==2.3.5` and `scipy==1.17.0` for the independent DOP853 runner

## Install and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Install the pinned REBOUND backend with
`python -m pip install -e '.[rebound]'`. The legacy `ias15` extra remains an
alias for compatibility.

Install the independent population-replication backend with
`python -m pip install -e '.[independent]'`.

Without installing the package:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m jxplanetx.cli validate --output validation.json
```

The test suite contains 102 tests covering integrator behavior,
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
  decimal_math.py          precision and vector primitives
  dynamics.py              N-body acceleration, state, and invariants
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

Citation metadata is provided in `CITATION.cff`. Original JX N-Body Engine code
is released under the MIT License. Optional third-party packages retain their
own licenses. `RELEASE_MANIFEST_v0.3.0.json` records the portable release and
scientific-artifact hashes.
