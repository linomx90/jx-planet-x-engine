# Independent DOP853 replication of the 10,000-year population screen

## Outcome

The corrective high-resolution replication, `final_result_v2.json`, returned
`PASSED`, and its stored-artifact audit returned `AUDIT_PASSED`.

For the 10,000 hash-selected tracers in each arm, independent SciPy DOP853 and
the locked REBOUND MERCURIUS reference classified exactly the same 433 tracer
identities as sampled low-perihelion injections in control and exactly the same
433 identities in source. Both methods therefore measured a source-minus-
control injection-fraction effect of exactly 0.0 on this selected population.

The scientific state remains `SCREENING_ONLY`. This result is an independent-
software replication of one numerical screen; it is not observational evidence
for or against Planet X.

## Question tested

Does the candidate-9118 source/control conclusion from the locked 100,000-
tracer REBOUND experiment survive a genuinely separate trajectory code and a
different integration method?

The replication intentionally reuses the reference experiment's initial-state
files, physical force assumptions, tracer population, 10,000-year duration,
annual sampling rule, and injection boundary. It changes the software and
numerical method:

| Component | Reference | Independent replication |
|---|---|---|
| Integration software | REBOUND 4.4.11 | SciPy 1.17.0 plus JX force code |
| Method | MERCURIUS | DOP853 |
| Force implementation | REBOUND | independent NumPy Newtonian evaluator |
| Step control | hybrid fixed/adaptive | adaptive, maximum 0.125 year |
| Relative tolerance | method-specific | 1×10⁻¹³ |
| Absolute tolerance | method-specific | 1×10⁻¹⁵ |
| Arithmetic | binary64 | binary64 |

`src/jxplanetx/independent_dop853.py` does not import REBOUND. It independently
implements the force evaluation, Kepler conversion, orbital-element recovery,
annual sampling, checkpoint/restart path, population comparison, paired
bootstrap, and fail-closed verdict logic.

## Frozen selection and design

Ten 1,000-tracer blocks were selected from the 100 reference blocks by ranking
SHA-256 digests of a public seed and integer block IDs. The selection function
does not read outcomes:

`[3, 9, 33, 47, 49, 64, 65, 74, 83, 98]`

This selection was frozen after the original REBOUND run, so it is outcome-
blind but not equivalent to a preregistration made before the reference
experiment. That limitation is stated in the selection manifest and both final
contracts.

| Item | Locked value |
|---|---:|
| Tracers per arm | 10,000 |
| Matched trajectories | 20,000 |
| Blocks | 10 × 1,000 |
| Duration | 10,000 years |
| Sample cadence | 1 year |
| Checkpoint cadence | 2,500 years |
| Injection boundary | sampled q < 29.999999 AU |
| Effect unit | source minus control injection fraction |
| Equivalence margin | ±0.001 |
| Bootstrap | paired block bootstrap, 9,999 repetitions |
| Workers | 8 |

Before the final replication, the independent path passed:

- six dedicated conversion, two-body, checkpoint, distribution, selection,
  and no-REBOUND-import tests;
- the complete 76-test project suite;
- a separate ten-year Horizons/DE441 qualification with a 33.6013 km maximum
  outer-planet position residual and 0.000510045 m/s velocity residual; and
- a 2,000-tracer-per-arm, 100-year operational population qualification with
  exact injection-identity agreement against REBOUND.

## Population result

| Metric | Independent control | Independent source | REBOUND control | REBOUND source |
|---|---:|---:|---:|---:|
| Tracers | 10,000 | 10,000 | 10,000 | 10,000 |
| Sampled injections | 433 | 433 | 433 | 433 |
| Injection fraction | 0.0433 | 0.0433 | 0.0433 | 0.0433 |
| Bound at 10,000 years | 10,000 | 10,000 | 10,000 | 10,000 |
| Survival fraction | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

For every selected block, the source-minus-control injection fraction was
0.0. The paired-block 95% bootstrap interval was therefore `[0.0, 0.0]`, fully
inside the locked ±0.001 equivalence margin. Independent DOP853 and reference
REBOUND both classified the selected-population effect as
`EQUIVALENT_WITHIN_LOCKED_MARGIN`.

Cross-software population distances were small:

| Arm | Injection identity disagreement | W₁ minimum q (AU) | W₁ final q (AU) | W₁ final i (deg) |
|---|---:|---:|---:|---:|
| Control | 0 | 1.08709×10⁻⁵ | 1.07288×10⁻⁴ | 1.61792×10⁻⁵ |
| Source | 0 | 1.07960×10⁻⁵ | 1.06215×10⁻⁴ | 1.60329×10⁻⁵ |

All are far inside their frozen gates. Injection and survival fractions agreed
exactly, so the absolute independent-versus-reference source/control-effect
difference was also 0.0.

## Numerical gates

| Gate | V2 observed | Locked threshold | Status |
|---|---:|---:|---:|
| Initial element roundtrip | 3.71756×10⁻¹¹ | ≤ 1×10⁻¹⁰ | PASS |
| Active-body relative energy drift | 8.35832×10⁻¹³ | ≤ 1×10⁻⁹ | PASS |
| Active angular-momentum-vector drift | 3.02089×10⁻¹³ | ≤ 1×10⁻⁹ | PASS |
| Active endpoint position disagreement | 2.86934×10⁻⁹ AU | ≤ 1×10⁻⁶ AU | PASS |
| Active endpoint velocity disagreement | 1.52002×10⁻⁹ AU/yr | ≤ 1×10⁻⁶ AU/yr | PASS |
| Source minimum perihelion | 352.261 AU | ≥ 200 AU | PASS |
| Source maximum fractional a excursion | 0.0363345 | ≤ 0.1 | PASS |
| Checkpoint replay | exact binary64 | exact | PASS |
| Finite, complete outputs | 20/20 blocks | all | PASS |
| Cross-software population comparison | all metrics | locked limits | PASS |

V2 completed in 1,935.11 seconds (32.25 minutes) with eight workers.

## Why there are v1 and v2 records

The original independent contract was executed and preserved. It returned
`INVALID` because one numerical gate missed: the maximum active-only versus
tracer-loaded endpoint position difference was 1.27532×10⁻⁶ AU against a
frozen 1×10⁻⁶ AU threshold. All other numerical gates and every cross-software
population gate passed; injection identities and source/control effects already
agreed exactly.

The failure was not erased or relabeled. A post-failure diagnostic found:

- the ten tracer-loaded active endpoints agreed pairwise within
  5.57401×10⁻⁹ AU;
- a half-step, tighter-tolerance active-only reference improved relative energy
  conservation from about 2×10⁻¹⁰ to 7.07899×10⁻¹³; and
- the failure was consistent with adaptive error-norm resolution dependence.

V2 was then registered as a corrective run. It changed only DOP853 resolution:
`rtol` 1×10⁻¹² → 1×10⁻¹³, `atol` 1×10⁻¹⁴ → 1×10⁻¹⁵, and maximum step
0.25 → 0.125 year. Selection, physical model, statistics, and every acceptance
threshold remained unchanged. V2 explicitly records that its settings were
chosen after v1 outcomes were available; it is not presented as the original
preregistered attempt.

## Independent artifact audit

`final_result_v2_audit.json` returned `AUDIT_PASSED`. Without calling the
trajectory runner's comparison functions, it verified and recomputed:

- all 19 contract-locked files and runtime hashes;
- 20 block summaries and 20 tracer CSVs containing 20,000 rows;
- all 100 checkpoint JSON manifests and all 100 binary64 NPZ state arrays;
- checkpoint array byte hashes, shapes, epochs, trackers, and segment records;
- final q, inclination, and bound state directly from each final checkpoint;
- all 20 locked REBOUND comparison CSVs;
- injection identities, survival, Wasserstein distances, block effects, and
  the deterministic 9,999-draw bootstrap interval;
- active endpoint disagreement, numerical extrema, every gate decision, and
  the final `PASSED` verdict.

The audit protects the stored result against corruption or internal
inconsistency. It does not constitute a third trajectory integrator.

## Scientific interpretation

The earlier REBOUND conclusion is robust to this independent-software check on
the selected 10% of its population. Under this exact candidate, initial epoch,
population proposal, Newtonian model, 10,000-year duration, and annual sampling
rule, adding candidate 9118 caused no measured change in sampled low-perihelion
injection rate in these 10,000 paired tracers.

This does **not** mean:

- Planet X does not exist;
- candidate 9118 is observationally excluded or validated;
- all possible distant-planet masses and orbits are equivalent to no source;
- a 10,000-year result predicts Solar-System-age behavior;
- the broad synthetic proposal represents the observed TNO population; or
- annual sampled minima are continuous encounter histories.

The next scientific gate is no longer another arithmetic-precision repeat. It
is a population model anchored to observed objects together with an explicit
survey selection function. Only after that should the project spend heavily on
a longer-horizon hierarchical integration.

## Locked artifacts

| Artifact | SHA-256 |
|---|---|
| `selection_v1.json` | `6cf177087444c109d8e83ee805d7c7319c396396f371f923c82612e1111a287d` |
| `final_contract_v1.json` | `cb3c03a8d62f3ce621d184e0603020e7f32dcdf527eb57b58459c58626d42210` |
| `final_result_v1.json` | `63dca00d4ef86df374d8ff4beafb6f86c8259177bac077cc1b7b1c4f0d1ba4d7` |
| `final_v1_diagnostic_contract.json` | `592c9ea6cdee09ad325f13e84afce3f84b0fdbf3ce16d85f4a42dc97ba165d72` |
| `final_v1_diagnostic_result.json` | `79fb86d3d99c8fba14d1229080abe40a29ead63c1fcab17fcdbbabb29d4cbcf8` |
| `final_contract_v2.json` | `1e7421c96bc401c44edb371079f7c9b590a84bfe98df9ed4de3d910ede4ed834` |
| `final_result_v2.json` | `2fe7c4a5f9c26b76ff0f4c2aaa02c584171a33739150801d6e4dce072771b6ff` |
| `final_result_v2_audit.json` | `f58df8ac8f5394ba0c83c6872788b0f976b7bd878c647c2ed6441cd9ea549b5b` |

## Reproduce and audit

Install the optional independent backend and run the tests:

```bash
python -m pip install -e '.[independent]'
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run a clean v2 replication only after archiving the existing immutable result:

```bash
PYTHONPATH=src python3 \
  runs/independent_dop853_10k/run_replication.py \
  --contract runs/independent_dop853_10k/final_contract_v2.json \
  --run-dir runs/independent_dop853_10k/final_execution_v2 \
  --output runs/independent_dop853_10k/final_result_v2.json \
  --workers 8
```

Audit the stored result:

```bash
PYTHONPATH=src python3 \
  runs/independent_dop853_10k/audit_final_v2.py \
  --contract runs/independent_dop853_10k/final_contract_v2.json \
  --result runs/independent_dop853_10k/final_result_v2.json \
  --execution-root runs/independent_dop853_10k/final_execution_v2 \
  --output runs/independent_dop853_10k/final_result_v2_audit.json
```

The runners and audit refuse to overwrite their locked outputs. The bulky
execution directories are intentionally excluded from the engine release;
contracts, source, compact result JSON, audit code, and this report are the
portable scientific record.
