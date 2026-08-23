# JX-O1 survey-selection milestone

**Independent V4 verdict:** `PASSED`  
**Evidence class:** fresh-pool independent computational confirmation  
**Claim state:** `SCREENING_ONLY`  
**Planet X claim:** none

V4 independently forward-biased two fresh outer-Solar-System calibration
populations through the official OSSOS F95 SurveySimulator at commit
`86ce0936c0585ea1120830831c66276e05b076aa`. Ten new seed blocks per arm
produced 212 tracked detections from 10,800,108 correct-model draws and 205
tracked detections from 22,700,227 deliberately wrong-model draws.

The official v2 run is preserved as `INVALID`: one leave-one-block-out Monte
Carlo estimate of the zeta SD was 0.00157 beyond its frozen tolerance. A
post-run exact diagnostic showed that all finite-pool zeta moments satisfy the
unchanged gate. V3 therefore performed a prelocked corrective replay using the
exact finite-pool formulas. It changed no data, threshold, summed-AD bootstrap,
or power calculation and is cryptographically bound to the v2 result and pool
manifests. V3 passed all gates, but it is not independent confirmation.

V4 was publicly preregistered before execution. It kept every V2 gate,
population law, sample target, and external simulator file unchanged while
using new intrinsic, official-driver, and resampling random domains. Its
finalizer also rejects any manifest or normalized detection-pool hash matching
V2. The separate result replay was byte-identical.

## Result summary

| Gate | Result |
|---|---:|
| False rejection | 4.65% |
| Wrong-model rejection power | 100% |
| Exact zeta mean | −8.6716982 |
| Exact zeta SD | 1.9213388 |
| Stable leave-one-block-out verdicts | 10/10 |
| Raw adapter identity | passed, both arms |
| Checkpoint replay | passed, both arms |
| Exact replay | passed |
| Contract and pool independence | passed |

## Compact artifacts

- `contract_v1.json` and `pilot_result_v1.json`: frozen first design and blocked pilot.
- `contract_v2.json`: corrected official-run contract.
- `official_qualification_v2_correct.json` and `official_qualification_v2_wrong.json`: 100,001-object official adapter qualifications.
- `pilot_result_v2.json`: blocked analytic control.
- `final_execution_summary_v2.json`: public per-block counts and cryptographic bindings.
- `final_result_v2.json`: immutable official `INVALID` record.
- `contract_v3_exact_zeta.json`: corrective replay contract.
- `corrective_result_v3.json`: immutable corrective `PASSED` record.
- `contract_v4_independent.json`: fresh-pool independent-confirmation contract.
- `registration_v4_independent.json`: public pre-outcome design registration.
- `final_execution_summary_v4_independent.json`: compact counts, hashes, and replay proof.
- `final_result_v4_independent.json`: immutable independent `PASSED` record.
- `run_experiment_v2.py`: checkpointed official runner.
- `run_corrective_replay_v3.py`: hash-bound corrective runner.
- `run_independent_confirmation_v4.py`: fresh-domain checkpointed V4 runner.

The V2 and V4 raw model/output/checkpoint archives are approximately 3.9 GB and
4.9 GB respectively and are excluded from the public code release. See
[`docs/SURVEY_SELECTION_VALIDATION.md`](../../docs/SURVEY_SELECTION_VALIDATION.md)
for equations, gates, limitations, and reproduction commands.
