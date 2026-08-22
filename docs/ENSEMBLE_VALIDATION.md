# Ensemble validation for chaotic N-body populations

## Purpose and boundary

The ensemble validator tests whether a paired source/control population result
is reproducible across predeclared phase or uncertainty draws, numerical
settings, and genuinely independent numerical methods. It compares population
distributions after individual chaotic trajectories cease to be meaningfully
reproducible.

The module validates **precomputed trajectory files**. It does not integrate an
N-body system, generate Cartesian initial states from the draws, or provide the
independent backend required by the scientific contract. A production result
therefore still requires a genuinely independent algorithm and code path to run
the complete locked 100,000-year source/control ensemble. Two tolerances of
REBOUND IAS15 are useful precision repeats, but they are not independent
methods.

Every accepted value remains `MODEL_OUTPUT`. Even a `PASSED` result remains
`SCREENING_ONLY`; it is not an astronomical detection or measurement.

## Workflow overview

The workflow has four stages:

1. Write and review a contract before inspecting ensemble outcomes.
2. Lock the contract and deterministically generate the ensemble plan.
3. Run every planned source/control trajectory externally and register each
   trajectory with its backend validity record.
4. Finalize the complete ensemble and inspect its validity, convergence,
   repeatability, and source/control-effect classification.

The locked plan is immutable. The commands refuse to overwrite plan and member
artifacts. If a scientific design changes, create a new experiment identifier
and a new plan rather than editing a completed plan.

## Contract and registration

The contract uses schema `jx-ensemble-contract/v1`. It fixes:

- experiment identifier, purpose, evidence class, and registration status;
- SHA-256 identities for the governing dynamics, initial-state builder, and
  source/control model artifacts;
- seed blocks, outer replicates, tracers, phase factors, and uncertainty
  distributions;
- frame, origin, units, duration, and exact output epochs;
- source/control methods, versions, settings, and independence groups;
- minimum sample sizes and minimum surviving bound samples;
- method-equivalence and repeat-equivalence thresholds, including the
  full-horizon minimum-perihelion Wasserstein gate;
- maximum primary-effect disagreement across methods and across seed blocks;
- primary endpoint, confidence level, bootstrap repetitions, null-equivalence
  margin, and minimum material effect; and
- a written power justification.

`CONFIRMATORY` requires a nonempty `registration_reference`, such as a Git
commit or external registration identifier. The software records that
reference but cannot prove that it existed before the results, validate its
timestamp, or judge its scientific adequacy. Without credible external
registration, the honest classification is `EXPLORATORY`.

The `power_plan` field is also a recorded scientific commitment, not an
implemented power calculator. A nonempty description satisfies the file
schema; investigators must independently justify the effective sample size,
clustering assumptions, target effect, power, and absence of optional stopping.

The included contract template is an engineering example, not a universal set
of astronomical thresholds. Its three model hashes identify placeholder text
and must be replaced with hashes of the actual governing-dynamics,
initial-state, and source/control specifications. Thresholds must be justified
and locked before the new trajectories are examined. They must not be tuned to
make a completed run pass.

## Ensemble hierarchy and pairing

The ensemble has three sampling levels:

```text
seed block
  outer replicate / member
    tracer
```

A seed block is an independently seeded repetition of the design. An outer
replicate, represented by one `member_id`, carries replicate-scope draws such
as source phase and contains all tracer-scope draws. Multiple tracers inside a
member share the replicate environment and are not treated as independent
bootstrap units.

For every member, the run matrix contains:

```text
control arm x every locked method
source arm  x every locked method
```

All arms and methods refer to the same deterministic member manifest and
`initial_draw_sha256`. The hardened validity record also binds each run to a
relative-initial-state digest and a full-initial-state digest. Finalization
requires one relative-state digest across all arms and methods for a member,
and one full-state digest across methods within each arm. The control and source
full-state digests may differ because adding the declared source changes the
full system.

These checks establish record-level pairing and detect inconsistent inputs.
The external trajectory producer remains responsible for applying the draws
correctly and hashing the intended states. Because the validator receives
digests rather than full Cartesian states, it cannot reconstruct the states or
prove that a dishonest or defective producer hashed the correct physical
content.

The plan generator uses deterministic hash-keyed draws. Supported independent
factors are uniform, normal, and periodic phase distributions. Correlated
Gaussian blocks use a declared mean and positive-definite covariance matrix;
the validator rejects a non-positive-definite matrix rather than silently
repairing it. The contract hash, member manifest hash, and plan hash preserve
the exact generated design.

## Precomputed trajectory and validity inputs

Each trajectory is a CSV file with at least these columns:

| Column | Meaning | Required units or values |
| --- | --- | --- |
| `time_year` | Locked output epoch | integer years |
| `name` | Planned tracer identifier | for example `t0000` |
| `q` | Osculating perihelion for a bound tracer | AU |
| `i_deg` | Osculating inclination for a bound tracer | degrees in `[0, 180]` |
| `bound` | Bound-state flag | `0`/`1` or `false`/`true` |

Every planned tracer must appear exactly once at every locked epoch. Duplicate,
missing, or unexpected tracer rows fail closed. A bound tracer requires finite,
nonnegative `q` and finite `i_deg`. For an unbound tracer, `q` and `i_deg` are
treated as unavailable rather than inserted into the conditional bound
distribution. Extra massive-body rows may be present, but they are not ensemble
samples.

The validator does not convert units. The producer must ensure that time is in
years, perihelion is in AU, and inclination is in degrees. The frame, origin,
and general units string in the contract are provenance declarations, not a
coordinate transformation performed by this module.

The ensemble analysis layer parses finite observables as Python binary64
floats. Arbitrary-precision integration remains the responsibility of the
backend; its exported observables are rounded when they enter these population
statistics. The engineering template's margins are many orders of magnitude
larger than binary64 roundoff, but a different study must justify its own
locked margins accordingly.

Each trajectory also requires a strict `jx-integrator-validity/v1` JSON record:

```json
{
  "schema": "jx-integrator-validity/v1",
  "plan_sha256": "<locked-plan-sha256>",
  "member_id": "b00-r0000",
  "arm": "control",
  "method_id": "ias15-primary",
  "method_spec_sha256": "<locked-method-spec-sha256>",
  "initial_draw_sha256": "<locked-member-draw-sha256>",
  "dynamics_model_sha256": "<locked-dynamics-model-sha256>",
  "initial_state_model_sha256": "<locked-initial-state-model-sha256>",
  "source_model_sha256": "<locked-source-model-sha256>",
  "relative_initial_state_sha256": "<paired-relative-state-sha256>",
  "full_initial_state_sha256": "<full-control-state-sha256>",
  "trajectory_sha256": "<raw-trajectory-sha256>",
  "duration_years": 100000,
  "epochs_year": [0, 1000, 10000, 100000],
  "frame": "declared frame",
  "origin": "declared origin",
  "units": "AU, yr, solar mass",
  "runner_source_manifest": {
    "schema": "jx-source-manifest/v2",
    "scope": "repository",
    "files": {
      "src/backend.py": "<backend-file-sha256>"
    },
    "tree_sha256": "<canonical-file-map-sha256>"
  },
  "passed": true,
  "checks": {
    "finite_state": {"passed": true},
    "energy_gate": {"passed": true}
  }
}
```

Digest labels in this schematic example are placeholders. A real validity
record requires lowercase 64-character SHA-256 digests, and the state digests
may not use the all-zero placeholder.

The validator requires exact agreement with the locked plan, method
specification, draw, duration, epoch support, frame, origin, units, and raw
trajectory hash. State digests must be valid nonzero SHA-256 values. The runner
source manifest must contain a nonempty file map with valid SHA-256 values and
a matching canonical tree digest. The registered member record embeds the
validity record and retains both raw-file and semantic hashes.

The registered summary also retains each tracer's bound/perihelion history at
every locked epoch plus its derived event outcome: initial and final state,
minimum sampled bound perihelion, and first sampled threshold-crossing epoch
with its from/to perihelia. Finalization re-derives transitions from that
history and cross-checks the histories against every epoch's perihelion
distribution, the locked tracer IDs, injection and survival totals, endpoint
distributions, and minimum-q distribution. This prevents rehashed aggregate or
transition fields from standing without matching epoch-by-epoch evidence
inside the member record.

All declared checks must pass. A failed hard backend-validity check makes the
ensemble `INVALID`. This record is still an assertion supplied by the backend
workflow: the ensemble module verifies its binding and internal consistency,
but it does not recalculate energy conservation, rehash the runner's source
files from their filesystem, or determine whether the chosen validity checks
are scientifically sufficient.

## Population metrics

The following metrics are evaluated in the locked units and at the locked
epochs:

- `low_q_fraction`: bound tracers with `q < q_threshold_AU`, divided by all
  planned tracers. The default template uses 30 AU.
- `injection_fraction`: tracers with at least one sampled transition from
  `q >= q_threshold_AU` to `q < q_threshold_AU`, divided by all planned
  tracers.
- `survival_fraction`: tracers marked bound at the final locked epoch, divided
  by all planned tracers.
- `mean_q_AU`: arithmetic mean perihelion among bound tracers at an epoch.
- `inclination_width_deg`: conditional bound-population width
  `Q84(i) - Q16(i)` in degrees, using linear sample-quantile interpolation.
- `wasserstein_q_AU`: one-dimensional Wasserstein distance between conditional
  bound-perihelion distributions, in AU.
- `wasserstein_i_deg`: one-dimensional Wasserstein distance between conditional
  bound-inclination distributions, in degrees.
- `wasserstein_min_q_AU`: Wasserstein distance between distributions of each
  tracer's minimum sampled bound perihelion over the full locked horizon, in
  AU.

Each tracer contributes its minimum `q` over epochs at which it is bound. A
tracer with no bound epoch has no minimum-q value. Minimum-q Wasserstein is a
method- and repeat-convergence metric and is also reported descriptively for
source versus control; it is not currently a selectable governing primary
endpoint. If `wasserstein_min_q_AU` is omitted from an older contract's gate
set, normalization uses the corresponding `wasserstein_q_AU` threshold.

### Exact meaning of `q < 30 AU`

`q` is an osculating orbital perihelion inferred from the state at a stored
epoch. It is not the instantaneous heliocentric radius. Therefore
`q < 30 AU` must not be described as a measured or event-detected physical
crossing of 30 AU.

The comparison is strict: `q == 30 AU` is not low-q when the threshold is
30 AU. A tracer already below the threshold at time zero contributes to low-q
occupancy but is not counted as a new injection. An injection is counted only
when two consecutive stored epochs are both bound and show a transition from
at-or-above to below the threshold. A transition that occurs and reverses
between stored epochs, or occurs across an unbound epoch, is not detected by
this summary. Output cadence is consequently part of the locked scientific
definition.

### Wasserstein distance is not a p-value

For one-dimensional empirical distributions, the module computes

```text
W1(F, G) = integral |F(x) - G(x)| dx
```

over the merged sorted support. It supports unequal sample sizes and optional
positive weights. It compares distributions rather than particle identities;
permuting tracer labels does not change the result.

`W1` has the units of the variable being compared. A value of `0.05 AU` is an
average distributional displacement in epoch-specific or minimum-perihelion
space. It is not a probability, significance level, likelihood, Bayes factor,
or confidence that
the two populations differ. Its meaning comes only from the locked numerical
equivalence threshold and the scientific scale of the endpoint.

If either bound conditional sample is empty or smaller than the locked minimum,
the conditional distributional inference is blocked. Ejections are not
silently dropped from the experiment: their effect remains visible through the
survival fraction, while `q` and inclination metrics remain explicitly
conditional on being bound.

## Numerical methods and independence

Each method declares a `method_id`, implementation, version, settings, and
`independence_group`.

Two IAS15 runs with different epsilon values belong to the same independence
group. They can establish within-method precision sensitivity, especially when
`require_within_group_repeat` is true, but they do not satisfy the independent
implementation requirement.

A genuine independent method must use a materially different integration
algorithm and code path, run the same locked members and both arms, cover the
same epochs and full 100,000-year horizon, and produce its own validity records.
Declaring a different group name does not make two implementations
scientifically independent; the validator checks the declared run matrix but
cannot audit source-code independence by itself.
It does reject two methods with identical implementation, version, and settings
even when they are assigned different method IDs or independence-group labels.

The example contract contains an `independent-method` placeholder. That
placeholder must be replaced with a real implementation and completed runs.
The ensemble validator itself supplies no such backend. Until this genuine
independent 100-kyr calculation exists, the production ensemble requirement is
not closed and the scientific result must remain `BLOCKED`.

## Repeatability and bootstrap inference

Method equivalence compares each pair of locked methods separately for source
and control. It takes the maximum disagreement across locked epochs for
epoch-dependent quantities, also compares the full-horizon minimum-q
distribution, and applies the predeclared method-equivalence thresholds.

The meaning of a failure depends on numerical independence:

- disagreement between methods in the same `independence_group` is failed
  precision convergence and makes the associated ensemble `INVALID`;
- disagreement between different independence groups is an independent-method
  conflict and yields `BLOCKED` with `claim_decision: CONFLICT`.

Repeat equivalence compares seed blocks within each method and arm using the
separately locked repeat thresholds. Seed-block instability yields `BLOCKED`
but remains `claim_decision: SCREENING_ONLY`; it is inadequate repeatability,
not a conflict between independently valid numerical methods. These
comparisons use normalized fractions, not exact-count equality.

The primary source/control effect is calculated inside every member as

```text
member effect = source primary endpoint - control primary endpoint
```

The supported primary endpoints are:

- `injection_fraction`;
- `survival_fraction`;
- `final_low_q_fraction`; and
- `final_mean_q_AU`.

Confidence intervals use a deterministic percentile bootstrap. The resampling
unit is the **paired outer replicate/member**, stratified by seed block. Every
tracer inside the selected member remains together, and source/control pairing
is preserved. Treating individual tracers or repeated epochs as independent
bootstrap observations would understate dependence and is not the implemented
analysis.

For a confidence interval `[L, U]`, null-equivalence margin `d0`, and minimum
material effect `d1`:

- `MATERIAL_POSITIVE` when `L >= d1`;
- `MATERIAL_NEGATIVE` when `U <= -d1`;
- `PRACTICALLY_EQUIVALENT` when `L >= -d0` and `U <= d0`; and
- `INCONCLUSIVE` otherwise.

All methods must agree on the primary-effect classification and remain within
the locked magnitude-disagreement threshold. Within-group disagreement in
primary-effect magnitude or classification is precision nonconvergence and is
`INVALID`. Cross-group disagreement is `BLOCKED` with
`claim_decision: CONFLICT`.

The validator also applies a paired-effect repeat gate. For each method it
averages the paired member effects separately inside each seed block, compares
every block pair, and requires the absolute difference to be at most
`max_primary_effect_repeat_disagreement`. If that field is omitted, contract
normalization uses `max_primary_effect_method_disagreement`. A failed paired
effect repeat gate is `BLOCKED` with `claim_decision: SCREENING_ONLY`.

A bootstrap interval is conditional on the specified ensemble and sampling
hierarchy; it does not incorporate omitted physical-model uncertainty or
observational selection.

## Verdicts

The governing precedence is `INVALID`, then `BLOCKED`, then `PASSED`.

### `INVALID`

The data or protocol cannot support inference. Examples include a corrupted or
modified plan, inconsistent plan/method/draw/state/trajectory/scope bindings,
an invalid runner source manifest, duplicate records, unexpected member
identity, nonfinite bound values, a failed hard integrator-validity record, or
failed convergence between precision variants in the same independence group.
Within-group disagreement on primary-effect magnitude or classification is
also `INVALID`. The associated `claim_decision` is `INVALID`.

### `BLOCKED`

The available records are structurally valid but do not close the locked
scientific gates. Examples include missing member runs, insufficient sample or
bound population, absence of a precision repeat, seed-block population or
paired-effect instability, cross-group independent-method disagreement, or an
inconclusive primary effect.

Cross-group population, primary-effect magnitude, or effect-classification
disagreement is `BLOCKED` with `claim_decision: CONFLICT`. Seed-block
instability, including `repeat_block_disagreement` and
`repeat_effect_disagreement`, is `BLOCKED` with
`claim_decision: SCREENING_ONLY`.

`BLOCKED` is not evidence that a source exists or does not exist. It means the
planned inference is not currently justified.

### `PASSED`

All encoded integrity, completeness, sample, bound-population, method,
repeatability, and primary-effect gates passed. A valid practically equivalent
result may pass; a nonzero source effect is not required.

`PASSED` means only that the modeled population comparison is numerically and
statistically resolved under the locked contract. Its `claim_decision` remains
`SCREENING_ONLY`.

## Commands

Create the editable template once:

```bash
PYTHONPATH=src python3 -m jxplanetx.cli write-ensemble-contract \
  --output runs/ensemble/contract.json
```

Edit and scientifically review that contract before running the ensemble. Then
lock it and generate the deterministic members:

```bash
PYTHONPATH=src python3 -m jxplanetx.cli prepare-ensemble \
  --contract runs/ensemble/contract.json \
  --output runs/ensemble/plan.lock.json
```

For each planned member, arm, and method, run the external backend and register
the resulting trajectory and validity record:

```bash
PYTHONPATH=src python3 -m jxplanetx.cli register-ensemble-member \
  --plan runs/ensemble/plan.lock.json \
  --member-id b00-r0000 \
  --arm control \
  --method-id ias15-primary \
  --trajectory external/b00-r0000-control-ias15-primary.csv \
  --validity external/b00-r0000-control-ias15-primary.validity.json \
  --output runs/ensemble/records/b00-r0000/control/ias15-primary.json
```

Repeat registration for both arms and every locked method. Finalize only after
the complete run matrix is available. Relative to `--run-root`, each member
record path is part of its validated identity and must be exactly
`<member_id>/<arm>/<method_id>.json`.

```bash
PYTHONPATH=src python3 -m jxplanetx.cli finalize-ensemble-validation \
  --plan runs/ensemble/plan.lock.json \
  --run-root runs/ensemble/records \
  --output runs/ensemble/final.json
```

Exit status is `0` for `PASSED`, `3` for `BLOCKED`, and `2` for `INVALID`.
Operational parsing or filesystem failures may also stop a command before a
scientific result is written.

## Output and interpretation

The final CLI artifact is a provenance run record. The ensemble result is under
`payload.result`. A complete finalization includes:

- `verdict`, `claim_decision`, and `effect_classification`;
- plan and contract hashes;
- registration status;
- invalid and blocked reason codes;
- completeness and sample-size gates;
- minimum-bound-population gates;
- descriptive `population_summaries` for every method and arm, including
  bound/low-q counts and fractions, injection and survival fractions,
  distribution summaries, and the minimum-q distribution;
- descriptive `source_control_distributions` for every method, including
  signed source-minus-control fractions and moments plus epoch-specific and
  minimum-q Wasserstein distances;
- arm-specific method comparisons;
- seed-block repeat comparisons;
- per-method primary source/control effects and bootstrap intervals; and
- cross-method primary-effect comparisons and the paired primary-effect repeat
  comparisons.

An early `INVALID` or incomplete `BLOCKED` result may contain only the fields
that could be established safely before finalization stopped.

The source/control distribution tables are descriptive model output. They do
not independently govern the primary-effect classification, and their
Wasserstein values are not p-values. The locked primary endpoint and paired
bootstrap remain the governing source/control inference.

The record also carries the active engine source manifest through the CLI
wrapper. Member records retain raw and semantic trajectory hashes, validity
hashes, deterministic draw identity, and summary hashes. These hashes establish
which artifacts were analyzed; they do not establish that the physical model,
priors, thresholds, backend implementation, or observational interpretation is
correct.

## Scientific non-claims

This module does not establish:

- detection or nonexistence of Planet X or any other source;
- measured mass, distance, orbit, or sky direction;
- an observational likelihood, survey completeness, or detectability result;
- explanatory sufficiency for the observed outer Solar System;
- validity outside the locked priors, bodies, equations, duration, epochs,
  endpoints, and thresholds; or
- independence merely because two methods have different declared labels.

A practically equivalent result means that the modeled effect was resolved
inside the locked equivalence margin for this ensemble. It does not prove that
the physical source is absent. A material effect means that the model changes
the selected simulated endpoint; it does not show that nature contains the
modeled source.
