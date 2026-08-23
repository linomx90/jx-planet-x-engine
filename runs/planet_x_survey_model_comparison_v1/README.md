# JX-O2 characterized-survey model-comparison design

**DESIGN ONLY — NO OUTCOMES GENERATED — EXECUTION NOT AUTHORIZED**

This directory publicly registers the safety and scientific structure before
any JX-O2 analysis execution or computed result. It is not a pre-data
registration: the named public catalogs and their literature outcomes were
already inspected. It does not contain an observed catalog, a runner, a
result, an execution contract, or permission to use the GPU.

## Scientific question

Can a later, fully locked model family containing an additional compact body
make better survey-forward predictions than a physically matched family
without that body, while each family also passes an absolute predictive
adequacy test?

The neutral labels are `M0_LOCKED_BASELINE` and
`M1_LOCKED_COMPACT_BODY_FAMILY`. They do not mean “correct” and “wrong.” A
failure of M0 can arise from population, migration, nuisance, or survey-model
misspecification. A failure of the registered M1 family cannot exclude other
Planet X configurations.

The maximum possible claim is a comparison of the exact registered model
families within the exact registered data and assumptions. No state in this
design means `DETECTED`, `EXCLUDED`, `CONFIRMED`, `RULED_OUT`, or scientific
`PASSED`.

## Evidence boundary

JX-O1 V4 validated a telescope-selection adapter using historical four-block
OSSOS-B characterization and two phenomenological calibration populations.
It did not use the complete characterized survey ensemble, an observed Solar
System catalog, or a realistic Planet X alternative. JX-O1 V4 therefore
satisfies no JX-O2 observational or realistic-power gate.

The intended survey inventory is broader and must be acquired and hashed
before any execution contract can exist:

- the complete characterized OSSOS release and its affiliated
  CFEPS/HiLat/Alexandersen ensemble, through its pinned native survey
  simulator;
- DES Y6 through its independently pinned simulator, retained as a separate
  survey stratum; and
- optionally, DEEP B1 as a separately declared robustness stratum.

Uncharacterized detections and heterogeneous MPC discoveries are excluded
from confirmatory inference because their selection and non-detection
histories are not equivalently characterized. Raw detections from different
surveys must never be pooled as if they shared one selection function.

All named catalogs and their published scientific outcomes are public and
have already been inspected. They are recorded as `PREVIOUSLY_INSPECTED`, not
`UNTOUCHED`. A future use of a survey-level holdout can support a
preregistered held-out prediction, but not a blinded-discovery claim. A truly
confirmatory claim requires a genuinely independent future survey or
chronological holdout whose outcomes did not influence the model, grid,
statistic, or candidate selection.

## What this registration freezes

The design freezes the claim ceiling, neutral hypothesis semantics,
survey-level partition rule, family-wide error rate, simultaneous type-I and
power requirements, prohibition on optional stopping, provenance and archive
requirements, blinding rules, allowed future states, and fail-closed
activation sequence.

It deliberately does **not** invent an exact Planet X grid, nuisance model,
primary score, catalog hash, or replicate count. Those unresolved scientific
choices are explicit execution blockers. A later immutable execution contract
must resolve every one of them before synthetic calibration. An independent
activation receipt may authorize one observed-data run only after calibration,
power, provenance, and independent-replay gates pass.

## Stop/go sequence

1. `G0`: audit the complete data inventory, prior exposure, candidate
   selection lineage, and model inputs.
2. `G1`: externally register an immutable execution contract with no
   unresolved fields.
3. `G2`: pass full-pipeline synthetic null calibration, realistic power,
   robustness, and numerical convergence.
4. `G3`: reproduce calibration with a materially independent implementation.
5. `G4`: have an independent custodian verify and unlock the committed
   holdout, if an untouched holdout exists.
6. `G5`: perform one fixed observed-data run and exact immutable replay.
7. `G6`: submit any `ELIGIBLE_FOR_EXTERNAL_REVIEW` result to external
   scientific review before making a broader scientific interpretation.

More intrinsic simulated objects can reduce Monte Carlo noise. They cannot
create observational information, repair prior data exposure, or substitute
for independent survey detections.

## Primary sources defining the design inventory

- [OSSOS complete data release](https://doi.org/10.3847/1538-4365/aab77a)
- [Official OSSOS SurveySimulator](https://github.com/OSSOS/SurveySimulator)
- [DES Y6 TNO catalog and characterization](https://arxiv.org/abs/2109.03758)
- [DESTNOSIM](https://github.com/bernardinelli/DESTNOSIM)
- [DEEP B1 characterization](https://arxiv.org/abs/2310.03671)
- [Selection-aware survey-simulator methodology](https://doi.org/10.3389/fspas.2018.00014)
- [Planet Nine model-family reference](https://arxiv.org/abs/2108.09868)

## Mandatory nonclaim

This design authorizes no execution and contains no observational outcome. A
future result may compare only the specified generative model families within
locked data and assumptions; it cannot by itself detect or exclude Planet X.
JX-O1 V4 validates only the survey-adapter calibration prerequisite.
