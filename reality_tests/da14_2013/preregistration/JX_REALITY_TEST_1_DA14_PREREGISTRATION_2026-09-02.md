# JX Reality Test 1 — Sealed Historical Holdout Preregistration

**Status:** FROZEN BEFORE OBJECT-SPECIFIC OBSERVATION RETRIEVAL  
**Preregistration date:** 2026-09-02 UTC  
**Target:** 367943 Duende (2012 DA14)  
**Cutoff:** 2013-02-01T00:00:00 UTC  
**Holdout window:** 2013-02-01T00:00:00 through 2013-03-01T00:00:00 UTC  
**Evidence class:** historical sealed-measurement holdout; **not** analyst-naïve blind, because the approximate encounter date and scale are already public and known.

## Question and immutable claim ceiling

Can a JX orbit-determination and propagation pipeline, using only target observations timestamped before the cutoff, predict withheld optical/radar measurements through the February 2013 Earth encounter and the resulting post-encounter motion? A pass applies only to this object, cutoff, observation set, model contract, and holdout window. It is not universal validation of JX, proof of superiority over JPL/REBOUND, or a Planet-X result.

## Permitted information before sealing predictions

1. **Target measurements:** MPC ADES-2022 observations with observation time strictly before the cutoff; JPL small-body radar astrometry only when its measurement epoch is before the cutoff.
2. **Observer/environment data:** MPC observatory-code metadata; frozen leap-second and IERS Earth-orientation tables; a frozen JPL DE441 major-body ephemeris and declared constants/kernels, all identified by filename and SHA-256.
3. **Documentation and generic algorithms:** published orbit-determination, astrometry, relativity, covariance, and numerical-integration literature.
4. **Schedule-only holdout metadata:** after automated curation, the solver may receive observation time, station code, measurement type, and required transmitter/receiver metadata, but not the withheld measured value or residual.

**Prohibited before prediction seal:** post-cutoff target astrometry or radar values; current JPL SBDB orbit/covariance; target-specific Horizons vectors, elements, SPKs, observer ephemerides, close-approach tables, B-plane values, or current target solutions; any manual use of published encounter numbers as fit constraints. Discovery identity and the already-known approximate encounter date are acknowledged background, not numerical inputs.

## Frozen model structure

The six-dimensional target state is estimated at a declared TDB epoch using weighted nonlinear least squares on pre-cutoff observations only. The covariance comes from the training-only normal system, with at most one preregistered robust variance-inflation factor estimated from training residuals. Outlier handling, station weighting, debiasing, and any model-selection rule must be specified and hashed before the first fit; no rule may depend on holdout residuals.

The observation model must explicitly implement and test: observatory geometry; UTC→TAI→TT→TDB conversion; Earth orientation; topocentric parallax; iterative light time; the coordinate conventions declared by each ADES record; and applicable aberration/relativistic light-deflection terms without double correction.

The force model is: Sun, eight planets, Moon separately, Pluto system, declared massive-asteroid perturbers, solar first-order post-Newtonian gravity, and Earth J2 during the encounter. Nongravitational target parameters are forbidden unless a training-only, prospectively specified significance gate activates them before prediction sealing. Every included or excluded force receives either implementation tests or a training-independent upper-bound justification.

## Predictions and metrics

Before unblinding, JX must seal: fitted state/covariance; training diagnostics; predicted topocentric RA/Dec at every optical holdout schedule row; predicted radar delay/Doppler for radar schedule rows; closest-approach epoch/distance; incoming/outgoing geocentric velocity; B-plane coordinates/covariance; and post-encounter state and osculating period. An independent adaptive implementation must propagate the same fitted state and force model. Numerical disagreement must be reported separately from observation residuals.

Primary metrics are tangent-plane optical residuals \(\Delta\alpha\cos\delta\) and \(\Delta\delta\), radar delay/Doppler residuals, weighted RMS, Mahalanobis-distance distribution, and covariance coverage. Secondary metrics are errors in closest-approach time/distance, B-plane coordinates, outgoing state, and post-encounter period. A **Threshold Addendum** must set all numeric pass/fail limits using only training data and declared measurement uncertainties, then be hashed before unblinding. Unblinding without that addendum yields `INVALID_BLIND`.

## Curation, sealing, and one-time unblinding

A deterministic curator downloads the complete records without printing post-cutoff values, writes `training.*`, encrypted/read-protected `holdout.*`, and a value-redacted `holdout_schedule.csv`, then hashes all outputs. Before unblinding, the following are frozen: source commit, environment lock, data/kernel hashes, model contract, weighting/outlier rules, Threshold Addendum, fitted state/covariance, complete prediction files, and SHA-256 manifest. Unblinding occurs once through a logged command that verifies the manifest. No refit, threshold change, force addition, observation deletion, or code modification is permitted afterward. Any later modification is a separate exploratory analysis and cannot replace the preregistered verdict.

**Allowed verdicts:** `PHYSICALLY_VALIDATED_ON_THIS_HOLDOUT`, `NUMERICALLY_VALID_BUT_PHYSICALLY_FAILED`, `ORBIT_FIT_FAILED`, `OBSERVATION_MODEL_FAILED`, `INVALID_BLIND`, or `BLOCKED`.

**Immediate next action:** implement and audit only the curator/seal mechanism. Do not download object-specific observations until that code can prove that withheld values are never exposed before the prediction manifest is frozen.
