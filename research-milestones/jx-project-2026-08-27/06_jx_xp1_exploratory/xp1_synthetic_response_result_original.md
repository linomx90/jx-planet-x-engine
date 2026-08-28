# XP1 synthetic response result

Date: 2026-08-23  
Experiment: `jx-xp1-public-synthetic-response-v1`  
Claim ceiling: `SYNTHETIC_250KYR_RESPONSE_ONLY`

## Result

In this frozen, idealized 250,000-year experiment, adding any of the six declared distant-body configurations produced **no change in the sampled q < 30 AU event fraction** relative to the control. The preregistered endpoint classification was:

`PRACTICALLY_SMALL`

This classification applies only to the locked sampled-q<30 count endpoint. It does not mean that the trajectories were identical or that a distant planet has no dynamical effect.

## Exact endpoint counts

The experiment used 64 deterministic synthetic tracers, one control, six added-body configurations, and a complete half-timestep repeat of all seven arms.

| Endpoint | Control and every added-body arm | Mixture effect |
|---|---:|---:|
| Ever sampled q < 30 AU | 0 / 64 | 0 / 384 = 0 |
| Ever sampled q < 35 AU | 4 / 64 | 0 / 384 = 0 |
| Finite and bound at 250 kyr | 64 / 64 | 0 / 384 = 0 |

The same four tracers crossed q<35 in every arm and at both timesteps. Every one of the four predeclared 16-tracer block effects was also exactly zero. The nearest sampled perihelion anywhere in the experiment was about 30.50 AU, so the q<30 result may be floor-limited.

## Nonzero distribution response

The added-body trajectories were not identical to the control. The unsigned mixture Wasserstein-1 distances were:

| Quantity | Primary timestep | Half timestep |
|---|---:|---:|
| Minimum sampled q | 0.05775 AU | 0.05927 AU |
| Final q | 0.18201 AU | 0.17633 AU |
| Final inclination | 0.07275 deg | 0.07330 deg |

These are descriptive distribution distances. No physical-smallness threshold was registered for them.

## Numerical validation

All integrity, conservation, and all-seven-arm timestep gates passed.

- Largest primary-versus-half-step W1 differences: 0.00487 AU for minimum q, 0.05968 AU for final q, and 0.01410 degrees for final inclination.
- Worst active-system energy drift: 1.7850e-7, below the 1e-6 gate.
- Worst COM angular-momentum drift: 2.5962e-13, below the 1e-10 gate.
- Worst scale-normalized linear-momentum residual: 3.2621e-15, below the 1e-10 gate.
- Every arm retained 5,001 samples and a complete finite osculating history for every tracer.

## Repeatability and provenance

Two clean executions reproduced the complete semantic result exactly:

- Execution A: 674.095 seconds; peak RSS 51.35 MB.
- Execution B: 674.428 seconds; peak RSS 50.72 MB.
- Shared semantic SHA-256: `1a04e928287e19558a621f09af128da76a69233c1e8c339306ef8b9c42499c9a`
- Replay verdict: `XP1_SEMANTIC_REPLAY_EXACT`
- Pre-output registration SHA-256: `9b8fb748ce3850cbf908c6b1dc8d22c0ffb7f94a2223d0d3f1fbfedb1e9dca81`

The independent verifier rebuilt the 64 tracers and all 14 initial states, then recomputed particle summaries, integer effects, block effects, Wasserstein distances, timestep comparisons, gates, and classification. This establishes deterministic repeatability within the same locked Python/REBOUND build, not an independent physical or cross-code replication.

## Scope and limitations

This experiment used an idealized Sun plus four circular, coplanar giant planets; 64 deterministic synthetic tracers; three physical cases; two fixed orientations; instantaneous insertion; 50-year sampling; and a 250-kyr horizon. It omitted observed catalogs, survey selection, inner planets, general relativity, migration, cluster history, Galactic tide, stellar encounters, collisions, and tracer backreaction.

Therefore this result:

- is not evidence for or against Planet X;
- is not a detection, exclusion, mass estimate, or orbital constraint;
- is not an author reproduction or independent physical validation;
- is not a formation-history or four-billion-year stability result;
- does not unblock or advance JX-O2 G0, which remains blocked pending audited source and survey inputs.

## Best next experiment

The most useful follow-up is a separately frozen XP2 robustness screen with broader deterministic angular coverage, more tracer blocks, multiple longer horizons, richer minimum-q and first-crossing-time endpoints, full timestep checks, and an independent integrator comparison on a preselected subset. That would test whether XP1's zero q<30 endpoint was caused by its two orientations and short horizon. Observational inference remains a separate later stage requiring independently audited and licensed characterized-survey inputs.
