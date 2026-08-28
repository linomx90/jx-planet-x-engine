# JX-E1 50,000-Year Local Result Report

## Bottom line

The locked local experiment completed all 90 arms but correctly returned `ENGINEERING_LONG_INVALID`. Execution B was not started, no threshold was changed, and no adaptive rerun was performed.

This is a valid numerical-methods no-go result. It is not evidence for or against Planet X and does not modify or unblock JX-O2.

## Locked experiment

- 74 primary arms: 2 synthetic controls plus 9 published `(mass, a, e, i)` tuples × 4 arbitrary orientation probes × 2 massless-tracer blocks.
- 16 predeclared half-step audit arms covering 8 active configurations in both tracer blocks.
- Synthetic analytic Sun + four-giant benchmark; no observed catalog, survey selection, author checkpoint, recovered angles, or recovered seeds.
- 50,000 synthetic years; REBOUND/MERCURIUS; primary step 0.125 yr and audit step 0.0625 yr.
- Runtime: 865.05 cumulative seconds; peak RSS 45,989,888 bytes; CPU only.

## Result

All 74 primary arms passed. Six of the 16 half-step records failed the locked (10^{-10}) angular- and linear-momentum drift limits. Because the tracers are massless, these are three distinct active configurations duplicated across the two tracer blocks.

| Fine-step configuration | Angular drift / gate | Linear drift / gate |
|---|---:|---:|
| CI07-B | 1.6784× | 1.0948× |
| CI09-A | 1.00966× | 1.09992× |
| CI09-D | 1.40456× | 1.09185× |

Everything else passed:

- Global maximum energy drift: `1.8410e-7` against a `1e-6` gate.
- Exact particle identity and fixed particle counts.
- Finite sampled states and invariants.
- Exact decoded checkpoint save/reload.
- Exact direct-versus-chained-restart state at every 10-year sample.
- All 16 timestep comparisons retained 32 paired bound tracers; minimum required was 16.
- Sampled event identities were exact at the two timesteps.
- Maximum final-perihelion discrepancy: `0.004068 AU` against `0.1 AU`.
- Maximum final-inclination discrepancy: `0.000704°` against `0.1°`.

## Integrity and diagnosis

- Contract SHA-256: `8ac4b0a54418e2ce1d4ce13e9c986177086ec433c900b6060537f15bf97a1187`
- Runner SHA-256: `a5f4d74e0a9c03c28fdb1a9a54d8c4be676df2c5caca7325570e9049e73500d9`
- Result-file SHA-256: `d121033d412a8cd3739b79827e506e895a616d735dde8eb4769d1617e05fc559`
- Result semantic SHA-256: `beb1a2d5756b3c4ca565009b2e24922541565504cb6dad52276392088532b479`
- All 90 arm records and all 900 checkpoint containers/decoded states verified.

A post-failure audit used no new dynamics. It verified all 16 fine-step records and their 160 unique stored checkpoints. The duplicate tracer blocks had identical checkpointed active-body states.

For the three failing configurations, center-of-mass-relative angular drift at the stored checkpoints was about `2.1e-13` to `2.8e-13`, while origin-based angular drift reached `1.01e-10` to `1.68e-10`. This supports a numerical-floor/frame-coupling diagnosis. Small linear-momentum drift remains in the stored double-precision states. That diagnosis cannot replace the locked gate or rehabilitate the result.

- Post-failure audit script SHA-256: `612f628eea7e25ec4096c1e69e9ef8031cab725dbdc7fba4508245b428127478`
- Post-failure audit receipt SHA-256: `eeb3843af50c1deb901bdd4a26abe88e4a9bccc1ef6378b684482b3e11d13257`

## Claim boundary

The only defensible conclusion is:

> The current locked 50,000-year synthetic engineering configuration did not satisfy its locally pre-output hash-locked conservation standard. The lock had no independent external timestamp.

It does not detect, confirm, prefer, exclude, or constrain Planet X. It is not an author-simulation reproduction, observational comparison, population inference, formation history, four-billion-year stability result, angle marginalization, or continuous closest-approach analysis.

## Next scientific requirement

Further local synthetic runs can improve numerical methods but cannot create Planet X evidence. A real scientific comparison requires a fully specified physical initial population/state, epoch and frame, complete six-dimensional compact-body states or a locked family, observational orbital uncertainties, and a complete characterized survey selection function with a preregistered statistic. Those inputs remain unavailable or unresolved, so JX-O2 remains blocked.
