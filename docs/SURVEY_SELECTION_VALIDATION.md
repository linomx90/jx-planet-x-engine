# JX-O1 telescope-selection validation

## Outcome

The locked JX-O1 workflow reached a corrective milestone verdict of `PASSED`.
That verdict validates a reproducible telescope-selection calibration workflow
on the existing official v2 pools. It is explicitly **not** an independent
replication and **not** a Planet X detection or exclusion.

The audit trail is intentionally preserved:

| Stage | Backend | Verdict | Meaning |
|---|---|---:|---|
| v1 analytic pilot | JX analytic approximation | `BLOCKED` | Statistical software path worked, but no official survey evidence was present. |
| v1 official qualification | Pinned OSSOS F95 | failed before data | Absolute paths exceeded legacy fixed-length fields; the v1 adapter also modeled detected rather than tracked rows. |
| v2 analytic pilot | JX analytic approximation | `BLOCKED` | Calibration and power passed; official-backend gates correctly blocked acceptance. |
| v2 official run | Pinned OSSOS F95 | `INVALID` | All physical and provenance gates passed, but one Monte Carlo leave-one-block-out zeta-SD estimate missed tolerance by 0.00157. |
| v3 corrective replay | Same immutable v2 pools | `PASSED` | Replaced only the noisy zeta moment estimates with exact finite-pool formulas; thresholds and all other tests were unchanged. |

V2 remains `INVALID`. V3 does not overwrite or retroactively reclassify it.

## External simulator boundary

JX executes the official [OSSOS SurveySimulator](https://github.com/OSSOS/SurveySimulator)
as a separate process. The source and characterization inputs are locked to
commit `86ce0936c0585ea1120830831c66276e05b076aa`, and every required file is
SHA-256 verified before execution. The external code remains under EUPL-1.1
and is not copied into the MIT-licensed JX package.

The pinned repository state contains the historical four-block OSSOS-B
characterization (`2013AE`, `2013AO`, `2013BL`, and `2014BH`). It is not the
complete eight-block OSSOS release or the full affiliated-survey ensemble.

## Frozen populations

The two arms share semimajor axis, inclination, angular elements, and absolute
magnitude draws. Only perihelion and object identity differ.

The shared semimajor-axis density is

\[
p(a) \propto a^{-3/2}, \qquad 100 \le a \le 1000\;\mathrm{AU}.
\]

The correct calibration arm uses

\[
q = 15 + 15U, \qquad U\sim\mathrm{Uniform}(0,1),
\]

while the deliberately wrong phenomenological arm uses

\[
q = 15 + 15U^{1/5}.
\]

Inclinations follow a half-normal distribution with width (15^\circ),
truncated to (0^\circ\le i\le40^\circ). Angular elements are independent and
uniform. The finite (H_r\) distribution is the locked divot with bright slope
0.9, faint slope 0.5, break 8.3, contrast 3.2, and support (5\le H_r\le12).
These are calibration distributions, not physical formation models.

## Official execution scale

Ten deterministic seed blocks were run per model. Each official batch contains
100,001 intrinsic objects. The non-round size is required because the pinned
`ReadModelFromFile` EOF path fails when its final internal 100-object chunk is
exactly full.

| Model | Intrinsic draws | Tracked detections | Minimum per block |
|---|---:|---:|---:|
| Correct | 9,700,097 | 206 | 20 |
| Wrong | 18,900,189 | 206 | 20 |
| **Total** | **28,600,286** | **412** | — |

Every raw tracked file was reparsed during finalization. Its semantic hash had
to equal the normalized CSV exactly. All checkpoint artifacts, driver inputs,
model files, outputs, and the compiled driver were hash verified on replay.

## Statistics and gates

For each variable

\[
(a,q,i,H_r,r,m_r),
\]

JX maps mock detections through a tie-aware empirical probability-integral
transform and sums six one-sample Anderson-Darling statistics. The reference
distribution uses 2,000 deterministic bootstrap catalogs, each containing 20
detections. Another 2,000 catalogs estimate false rejection and wrong-model
power.

The primary perihelion statistic is

\[
\zeta=\sum_{k=1}^{20}\log_{10}\!\left(\operatorname{PIT}(q_k)\right).
\]

For a finite correct-model pool sampled with replacement, define

\[
x_j=\log_{10}\!\left(\operatorname{PIT}(q_j)\right).
\]

V3 evaluates its two moments exactly:

\[
\mathbb E[\zeta]=20\,\bar{x},\qquad
\operatorname{SD}(\zeta)=
\sqrt{20\left(\frac{1}{N}\sum_{j=1}^{N}(x_j-\bar{x})^2\right)}.
\]

This removes Monte Carlo error from quantities that are analytically known;
the summed-AD calibration and power calculation remain bootstrapped.

The unchanged acceptance gates were:

- false rejection between 3.5% and 6.5%;
- absolute zeta mean and SD error at most 0.1;
- wrong-model rejection power at least 80%;
- exact statistical replay;
- exact raw-to-normalized adapter identity;
- checkpoint replay;
- the same verdict after excluding each of ten seed blocks;
- required intrinsic and tracked sample scales in every block.

The corrective v3 result was:

| Metric | Result | Gate |
|---|---:|---:|
| Correct-model false rejection | 5.45% | 3.5%–6.5% |
| Wrong-model rejection power | 100% | ≥80% |
| Exact zeta mean | −8.6712851 | within 0.1 of −8.6858896 |
| Exact zeta SD | 1.9208218 | within 0.1 of 1.9422240 |
| Leave-one-block-out verdicts | 10/10 stable | 10/10 |
| Adapter identity | passed for both arms | required |
| Checkpoint replay | passed for both arms | required |
| Exact replay | passed | required |

## Reproduction

The official simulator must be cloned separately at the locked commit and a
Fortran compiler must be available.

```bash
PYTHONPATH=src python3 runs/survey_selection_o1/run_experiment_v2.py preflight \
  --contract runs/survey_selection_o1/contract_v2.json \
  --simulator-root /path/to/OSSOS-SurveySimulator

PYTHONPATH=src python3 runs/survey_selection_o1/run_experiment_v2.py run \
  --contract runs/survey_selection_o1/contract_v2.json \
  --simulator-root /path/to/OSSOS-SurveySimulator \
  --run-dir runs/survey_selection_o1/final_execution_v2 \
  --output runs/survey_selection_o1/final_result_v2.json

PYTHONPATH=src python3 runs/survey_selection_o1/run_corrective_replay_v3.py \
  --contract runs/survey_selection_o1/contract_v3_exact_zeta.json \
  --v2-result runs/survey_selection_o1/final_result_v2.json \
  --correct-manifest runs/survey_selection_o1/final_execution_v2/correct_pool.json \
  --wrong-manifest runs/survey_selection_o1/final_execution_v2/wrong_pool.json \
  --output runs/survey_selection_o1/corrective_result_v3.json
```

The multi-gigabyte execution archive is intentionally excluded from the code
release. Contracts, compact verdicts, qualification records, per-block counts,
and cryptographic bindings are retained.

## Scientific boundary

A `PASSED` JX-O1 corrective verdict says the locked code can generate a
reproducible calibration population, forward-bias it through a pinned real
telescope-selection function, recover nominal calibration, and reject one
deliberately strong wrong population. It does not establish sensitivity to a
realistic Planet X model, describe the observed Solar System, or provide a
source mass, orbit, distance, or sky location.

