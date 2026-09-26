# JX public lunar ephemeris v1: 90/365-day holdout

Date: 2026-09-24

Package version: `0.6.0rc16`

Decision: `PASS_SCREENING_ONLY`

## Question and frozen gate

This screen asks whether the mutual EIH 1PN increment in the public
simultaneous resolved-eleven coupled lunar component improves untouched DE440
translational endpoints at nominal 90 and 365 days. The matched control uses
the same initial state, fixed RKF78 step, Newtonian point masses, reacting
static lunar quadrupoles, reacting fixed-axis Earth J2, mantle rotation, and
fluid-core subsystem, but removes mutual EIH 1PN.

The protocol was frozen before either long endpoint was executed. Its semantic
SHA-256 is
`27b1d225d4773b96763407a9acccae5a99bc57a3d7f7b6fbc72545a182ea73b2`.
The create-only preregistration is
`/tmp/jx_public_lunar_ephemeris_90_365_prereg_v1.json`; its file SHA-256 is
`aafc027092d39bd442617502d276c3354d6e8502ecb4c8738746fd059328588a`.

The fixed gate requires every candidate endpoint metric to improve over the
matched control, an aggregate normalized RMS candidate/control ratio below
`0.25`, every individual ratio below `0.5`, finite output, and exact step and
force-evaluation accounting. No parameter was fitted.

## Results

| Endpoint | Metric | Public EIH candidate | No-EIH control | Ratio |
|---|---|---:|---:|---:|
| 90 days | Earth–Moon position | 0.013980 km | 0.874538 km | 0.015986 |
| 90 days | Earth–Moon velocity | 4.0902e-8 km/s | 2.3367e-6 km/s | 0.017504 |
| 90 days | Sun–Earth position | 0.012435 km | 6.653770 km | 0.001869 |
| 90 days | Sun–Earth velocity | 3.3708e-9 km/s | 2.1075e-6 km/s | 0.001599 |
| 365 days | Earth–Moon position | 0.061568 km | 3.729707 km | 0.016507 |
| 365 days | Earth–Moon velocity | 1.8471e-7 km/s | 9.8849e-6 km/s | 0.018686 |
| 365 days | Sun–Earth position | 0.029261 km | 56.094548 km | 0.000522 |
| 365 days | Sun–Earth velocity | 8.1235e-9 km/s | 1.1074e-5 km/s | 0.000734 |

The aggregate normalized RMS ratio is `0.012198720845768172`; the largest
individual ratio is `0.018685875377906735`. Every registered gate passed.

The candidate accepted exactly 11,680 fixed steps and performed 151,840 force
evaluations. The largest quaternion norm error before projection was one
binary64 epsilon. The accepted-step ledger SHA-256 is
`3b440e03396fa41bc3274b9373ce84f1895f34e843cba89f272ced71698f1005`;
the engine result-content SHA-256 is
`a3d00e079ddf3f507cf6fa7cbba50549a097799f9cc7ab6b45615a03a938a25f`.

The create-only report is
`/tmp/jx_public_lunar_ephemeris_90_365_holdout_v1.json`. Its file SHA-256 is
`acebb17381812e3b73c0d1f3f77ba01c946f1a963b2cf59b7bdfa46ef91665bb`;
its internal semantic digest is
`39a2c4bb8e0b752710d8e3465d1b79b3c3feb0e981ba77520f89530934b6d279`.

## Interpretation and limits

The result establishes that the public implementation remains coherent over
one year and that its mutual EIH 1PN term materially improves these four DE440
translation metrics relative to the otherwise matched no-EIH model. It does
not independently validate DE440, score lunar attitude against a held-out
orientation reference, validate raw LLR, or qualify an astronomical ephemeris.

The public v1 model still omits solar J2, delayed tides, time-variable lunar deformation,
lunar degree three and higher, Earth higher zonals and tesserals, minor bodies,
planetary satellites, observation reduction, and parameter fitting. DE440 is
a fitted reference and the retained inputs are project-held, so organizational
independence is false. The report authorizes no navigation, production,
publication, or general superiority claim.

Solar J2 was identified after this protocol and report were frozen. This
documentation records the omission without rewriting the programmatic v1
roster, source hash, report, or preregistration.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python \
  benchmarks/jx_public_lunar_ephemeris_90_365_holdout.py \
  --validate-report /tmp/jx_public_lunar_ephemeris_90_365_holdout_v1.json

PYTHONPATH=src:. .venv/bin/python -m unittest -v \
  tests.test_engine_lunar_ephemeris \
  tests.test_jx_public_lunar_ephemeris_90_365_holdout
```
