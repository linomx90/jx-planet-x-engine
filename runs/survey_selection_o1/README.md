# JX-O1 survey-selection milestone

**Corrective verdict:** `PASSED`  
**Evidence class:** telescope-selection calibration on existing official v2 pools  
**Claim state:** `SCREENING_ONLY`  
**Planet X claim:** none

JX forward-biased two locked outer-Solar-System calibration populations through
the official OSSOS F95 SurveySimulator at commit
`86ce0936c0585ea1120830831c66276e05b076aa`. Ten seed blocks per arm produced
206 tracked detections from 9,700,097 correct-model draws and 206 tracked
detections from 18,900,189 deliberately wrong-model draws.

The official v2 run is preserved as `INVALID`: one leave-one-block-out Monte
Carlo estimate of the zeta SD was 0.00157 beyond its frozen tolerance. A
post-run exact diagnostic showed that all finite-pool zeta moments satisfy the
unchanged gate. V3 therefore performed a prelocked corrective replay using the
exact finite-pool formulas. It changed no data, threshold, summed-AD bootstrap,
or power calculation and is cryptographically bound to the v2 result and pool
manifests. V3 passed all gates, but it is not independent confirmation.

## Result summary

| Gate | Result |
|---|---:|
| False rejection | 5.45% |
| Wrong-model rejection power | 100% |
| Exact zeta mean | −8.6712851 |
| Exact zeta SD | 1.9208218 |
| Stable leave-one-block-out verdicts | 10/10 |
| Raw adapter identity | passed, both arms |
| Checkpoint replay | passed, both arms |
| Exact replay | passed |

## Compact artifacts

- `contract_v1.json` and `pilot_result_v1.json`: frozen first design and blocked pilot.
- `contract_v2.json`: corrected official-run contract.
- `official_qualification_v2_correct.json` and `official_qualification_v2_wrong.json`: 100,001-object official adapter qualifications.
- `pilot_result_v2.json`: blocked analytic control.
- `final_execution_summary_v2.json`: public per-block counts and cryptographic bindings.
- `final_result_v2.json`: immutable official `INVALID` record.
- `contract_v3_exact_zeta.json`: corrective replay contract.
- `corrective_result_v3.json`: immutable corrective `PASSED` record.
- `run_experiment_v2.py`: checkpointed official runner.
- `run_corrective_replay_v3.py`: hash-bound corrective runner.

The raw model/output/checkpoint archive is approximately 3.9 GB and is excluded
from the public code release. See
[`docs/SURVEY_SELECTION_VALIDATION.md`](../../docs/SURVEY_SELECTION_VALIDATION.md)
for equations, gates, limitations, and reproduction commands.

