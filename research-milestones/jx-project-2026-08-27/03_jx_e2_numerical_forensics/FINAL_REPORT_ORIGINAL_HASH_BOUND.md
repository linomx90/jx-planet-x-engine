# JX-E2 final numerical-method forensics report

## Outcome

JX-E2 completed two clean, independently instantiated CPU executions and a
separate replay verification.

- Replay verdict: `JX_E2_SEMANTIC_REPLAY_EXACT`
- Numerical classification: `MIXED_OR_INCONCLUSIVE`
- Claim ceiling: `NUMERICAL_METHOD_FORENSICS_ONLY`
- Shared semantic SHA-256:
  `4cf3a6cbd1cb051e8404d7242190820e1fb07ec7a26ca099d06cb4df837f2b2d`

This locked numerical-method diagnostic reproduced exactly at the semantic
level in two executions of the same content-addressed build. It is not evidence
for or against Planet X, not an independent scientific replication, and not a
repair or rehabilitation of JX-E1. JX-E1 remains
`ENGINEERING_LONG_INVALID`; JX-O2 remains blocked.

## What was run

The locked matrix retained all eight predeclared E1 audit configurations:
M0, CI01-A, CI03-C, CI05-C, CI06-C, CI07-B, CI09-A, and CI09-D. Each was run
in the original frame and an active-body barycentric frame with three
MERCURIUS timesteps and three IAS15 tolerances, for 50,000 synthetic years.

Per clean execution, the output contains:

- 8 configuration bundles;
- 96 numerical arms;
- 160 within-E2 comparisons;
- 160 preserved E1 checkpoint-context comparisons;
- 960 checkpoints;
- 970 files in the exact output inventory.

Execution A took 1,028.881 seconds and execution B took 1,029.737 seconds.
They had distinct execution-instance identifiers but produced the identical
semantic hash. All declared integrity checks were true, no IAS15
iteration-limit event occurred, and neither output has a failure receipt.

## Numerical finding

All eight configurations are `MIXED_OR_INCONCLUSIVE`.

The preregistered IAS15 reference gate failed in both frames for every
configuration. The discrepancy between the two tightest IAS15 tolerances was
5.90 to 18.46 times the absolute ceiling, and its tightening ratio was 0.622 to
3.928 rather than the required value of at most 0.25. Therefore IAS15 at the
tightest setting was not established as a reference. The protocol consequently
does not license a positive accuracy, frame-equivalence, instability, or
mechanism classification.

Two descriptive patterns are nevertheless useful for future engineering:

1. Across all configurations and both frames, MERCURIUS discrepancy relative
   to the tightest IAS15 run decreased by almost exactly four when the timestep
   was halved. This is smooth, second-order-like sampled refinement, with no
   observed two-step divergence. It is not demonstrated accuracy because the
   IAS15 reference gate remained closed.
2. Only the three configurations that exceeded E1's momentum gates—CI07-B,
   CI09-A, and CI09-D—showed the predeclared linear-momentum step-count pattern
   in the original frame. The barycentric-frame runs did not satisfy that
   predeclared signature. This exactly replayed frame-specific correlation is a
   useful forensic observation, but because the IAS15-reference and registered
   both-frame prerequisites failed, it cannot be classified as evidence for
   frame-sensitive accumulation or any unique cause.

`frame_state_equivalent=false` is a fail-closed consequence of the missing
IAS15 reference, not evidence that the frames are physically inequivalent.
Likewise, the true MERCURIUS refinement sub-indicator must not be promoted
to an accuracy or adequacy claim.

## Locked provenance

- Contract SHA-256:
  `d4edc6e17df40c3eeb6a72c7c55ad3bb530e6c79a40017f6dffe9ec553bc3d8f`
- Registration SHA-256:
  `326f7e51dcdb51c477041b9c941b67743a0e6a92c8d04fa42bd8a0ae9eeb8486`
- Runner SHA-256:
  `223735e8256ae86ee00bc0105a33b88a6a225ba556414201b27cacb0a7ab7f9d`
- Verifier SHA-256:
  `75fff9cd167da6e12b628a8d2960024f4979d023e387cfe146a1524c30ccf444`
- Execution A result SHA-256:
  `102c54a849ba3e03266e648e2b3346d2a235989a34bb789c378eb3c95dccc310`
- Execution B result SHA-256:
  `eb704ade084c20c002b69c50b166d830703dca25cd748032397a250e97ef9912`
- Execution A output-manifest SHA-256:
  `0eba12ba737ce63d4cd9ac8e46eae536184acd5c54c800b41e46c0841acebbda`
- Execution B output-manifest SHA-256:
  `3da6ac80bf8cfbec2da53197e1bb529515254795fbb1cf87170e29eb3d5041e3`
- Replay-receipt SHA-256:
  `8989e8ba31dd0e0bfe15180fab760b80056e021944c08f73f191f702029056cb`

The verifier independently checked locked identities, the exact file and
checkpoint inventory, decoded checkpoint states, initial-state reconstruction,
schemas, endpoint arithmetic, classification arithmetic, and A/B semantic
equality. Raw 10-year trajectories were intentionally not retained, so sampled
maxima are content-bound by exact A/B replay but cannot be reconstructed
independently from stored trajectories.

## Next step

Freeze JX-E2 as inconclusive; do not widen its thresholds and do not launch a
post-outcome rescue run. Return to JX-O2 G0 input readiness: obtain and hash the
authoritative complete characterized survey inputs and acquire or define
physically matched, finite M0/M1 model families with full state/orbit,
selection lineage, nuisance model, weights, covariance/epoch/count rules, and
deterministic seeds. After G0 is complete, register the immutable G1 execution
contract. Calibration may begin only under that contract, and observed-data
execution remains unauthorized until the later registered calibration,
independent-reproduction, and holdout gates are satisfied.
