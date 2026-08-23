# DE441-backed 100,000-tracer source/control screen

## Outcome

The locked experiment **passed every contracted numerical gate**. Within this
specific 10,000-year model, adding Brown–Batygin candidate 9118 produced an
injection fraction **equivalent within the predeclared ±0.001 margin**.

This is a numerical source/control result, not a detection or exclusion of
Planet X. The result remains `SCREENING_ONLY`.

## Question tested

Does adding one declared distant source model—candidate 9118, 5.06 Earth
masses—materially change the fraction of a broad, outcome-blind test-particle
population whose sampled osculating perihelion falls below 29.999999 AU?

The control and source arms use exactly matched initial tracer states. The
source arm differs only by the added candidate. The planetary backbone begins
at JD 2461200.5 TDB and is derived from the already-passed JPL
Horizons/DE441 compatibility state. Sun, Mercury, Venus, Earth–Moon barycenter,
and Mars are collapsed into one inner-system monopole at the initial epoch;
Jupiter, Saturn, Uranus, Neptune, and Pluto remain active bodies.

Candidate 9118's catalog elements are declared at the backbone epoch. That
epoch assignment is a model assumption, not a JPL source state or a new
observational fit.

## Frozen design

| Item | Locked value |
|---|---:|
| Primary tracers per arm | 100,000 |
| Matched primary trajectories | 200,000 |
| Independent phase/design blocks | 100 × 1,000 tracers |
| Duration | 10,000 years |
| Primary integrator | REBOUND 4.4.11 MERCURIUS |
| Primary step | 0.0625 year |
| Primary sample cadence | 1 year |
| Half-step audit | 10,000 tracers/arm at 0.03125 year |
| Fine-sample audit | 10,000 tracers/arm at 0.25-year sampling |
| Total propagated tracer trajectories | 240,000 |
| Injection boundary | sampled q < 29.999999 AU |
| Effect unit | source minus control injection fraction |
| Equivalence margin | ±0.001 |
| Bootstrap | paired block bootstrap, 9,999 repetitions |

The locked population is an outcome-blind stratified proposal, not an observed
TNO prior, posterior, or survey selection function. It draws semimajor axis
log-uniformly from 100–1,000 AU, perihelion uniformly from 31–80 AU,
inclination isotropically in cos(i) from 0–40 degrees, and all three angular
variables uniformly over a full turn.

The final contract was registered after the operational qualification passed
but before any final-run integration. Population draws, effect rules, margins,
and all numerical thresholds were already frozen.

## Primary population result

| Metric | Control | Source | Source − control |
|---|---:|---:|---:|
| Sampled injections | 4,377 | 4,374 | −3 |
| Sampled injection fraction | 0.04377 | 0.04374 | −0.00003 |
| Bound at 10,000 years | 100,000 | 100,000 | 0 |
| Survival fraction | 1.00000 | 1.00000 | 0 |
| Mean minimum sampled q (AU) | 51.578373 | 51.578109 | — |
| Minimum sampled q (AU) | 19.289961 | 19.292071 | — |
| Final inclination width (deg) | 9.471714 | 9.471684 | — |

The paired-block 95% bootstrap interval for the injection-fraction effect is
**[−0.00009, +0.00003]**. The entire interval lies within the locked ±0.001
equivalence margin, giving `EQUIVALENT_WITHIN_LOCKED_MARGIN`.

Population-distribution distances were small:

- minimum sampled q Wasserstein distance: 0.00115140 AU;
- final bound q Wasserstein distance: 0.00157002 AU;
- final bound inclination Wasserstein distance: 0.000640356 degrees.

The annual snapshots recorded 31 source-arm tracers within one initial source
Hill radius and 282 within three. These are sampled proximity counts, not
continuous close-encounter detections.

## Numerical gates

| Gate | Observed | Threshold | Status |
|---|---:|---:|---:|
| Maximum every-step massive-body energy drift | 4.81251×10⁻⁸ | ≤ 1×10⁻⁷ | PASS |
| Maximum every-step angular-momentum-vector drift | 8.13361×10⁻¹² | ≤ 1×10⁻¹⁰ | PASS |
| Source minimum perihelion | 352.262 AU | ≥ 200 AU | PASS |
| Source maximum fractional semimajor-axis excursion | 0.0363324 | ≤ 0.1 | PASS |
| Half-step population convergence | all metrics | locked limits | PASS |
| Fine-sample population convergence | all metrics | locked limits | PASS |
| Massless active-twin endpoints | exact hashes | exact | PASS |
| Checkpoint/restart replay | exact hashes | exact | PASS |
| Finite, complete block outputs | 240/240 | all | PASS |

The half-step audit had zero injection-identity disagreement in both arms. The
fine-sample audit changed 2/10,000 control classifications and 3/10,000 source
classifications; the induced source/control-effect difference was 0.0001,
inside its locked 0.001 limit.

The run used eight workers and completed in 4,921.89 seconds (82.03 minutes).

## Independent artifact readback

`final_result_v1_audit.json` has verdict `AUDIT_PASSED`. The audit independently
re-read and verified:

- 240 block summaries and their hashes;
- 240 tracer CSVs containing 240,000 rows;
- exact paired initial metadata;
- all 1,200 checkpoint archive hashes and reconstructed REBOUND state digests;
- primary injection and survival counts;
- convergence metrics;
- the deterministic paired bootstrap interval and effect classification.

This protects against a damaged or internally inconsistent stored result. It
does not provide independent dynamical-software replication.

## Scientific interpretation

Under this exact candidate, epoch assumption, population proposal, force
model, 10,000-year duration, and sampling rule, the candidate did **not**
produce a material change in low-perihelion injection rate. The measured
difference is three fewer sampled injections per 100,000 tracers, with an
interval spanning zero and fully inside the predeclared equivalence region.

This does **not** mean:

- Planet X does not exist;
- candidate 9118 is observationally excluded;
- all source masses or orbits are dynamically equivalent to no source;
- a 10,000-year screen predicts a 100-million-year or Solar-System-age result;
- the broad proposal represents the real observed TNO population;
- sampled perihelion minima are continuous encounter histories.

An independent-software DOP853 replication has now passed on ten outcome-blind
hash-selected blocks, with exact injection-identity agreement against REBOUND.
See [its complete report](../independent_dop853_10k/README.md). The next
scientific gate is now a population model tied to observed objects and an
explicit survey selection function. A longer-horizon hierarchical experiment
would then test whether the short-screen equivalence persists.

## Locked artifacts

| Artifact | SHA-256 |
|---|---|
| `final_contract_v1.json` | `1f3ccf6f29d4ec9890b9d66d19858c148760b00167281ce386afdfd0c8801725` |
| `final_result_v1.json` | `24b7572cf130c683acd66a4677dac62d0d30b15b24f9bf5997b612a8045d7efd` |
| `final_result_v1_audit.json` | `57922e53e0c9a0bb021c659c5abecff85212792dac88241d62b9c8c4423acf12` |
| `population_elements_v1.csv` | `2e4def4aa9fbbfaf451dc563480d7010c3afd8e39b7ffeceb91a4f6a055442d1` |
| `states/de441_control_state.csv` | `edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14` |
| `states/de441_source_9118_state.csv` | `81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf` |
| DE441 validation result | `277c006e4c6474b2d52c1f002daf6b156067398685dca8cc1efa1e138c753a4e` |

## Reproduce and audit

With the pinned REBOUND wheel and locked inputs available:

```bash
PYTHONPATH=../.vendor:src python3 -m unittest discover -s tests -v

PYTHONPATH=../.vendor:src python3 \
  runs/de441_population_100k/run_population.py \
  --contract runs/de441_population_100k/final_contract_v1.json \
  --run-dir runs/de441_population_100k/final_execution_v1 \
  --output runs/de441_population_100k/final_result_v1.json \
  --workers 8

PYTHONPATH=../.vendor:src python3 \
  runs/de441_population_100k/audit_final_result.py \
  --contract runs/de441_population_100k/final_contract_v1.json \
  --result runs/de441_population_100k/final_result_v1.json \
  --execution runs/de441_population_100k/final_execution_v1 \
  --population runs/de441_population_100k/population_elements_v1.csv \
  --output runs/de441_population_100k/final_result_v1_audit.json
```

The runners refuse to overwrite existing locked result files. Move the
existing execution/result artifacts to a separate archival location before an
intentional clean replication; do not delete the original scientific record.
