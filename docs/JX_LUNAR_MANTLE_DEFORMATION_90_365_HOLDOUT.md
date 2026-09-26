# JX complete mantle-deformation implementation and long holdout

Date: 2026-09-20

Status: `SCREENING_ONLY`

Decision: `JX_RETAIN_STATIC_MANTLE_CONTROL_AND_STOP_DEFORMATION_PROMOTION`

## What was implemented

The opt-in engine component
`src/jxplanetx/solar_system/lunar_mantle_deformation.py` now evaluates one
coherent time-variable lunar deformation package:

- the TAUM-delayed Earth-raised tidal inertia and its analytic derivative;
- the TAUM-delayed spin-distortion inertia and its analytic derivative;
- undistorted total-minus-core inertia on the mantle angular-momentum side;
- the undistorted total-Moon figure plus both deformation increments on the
  exterior gravity side; and
- arbitrary symmetric-tensor point-source force and torque from one common
  quadrupole potential.

The caller owns the delay history. The component does not infer prehistory,
query an ephemeris, select a core state, or claim to reproduce the private JPL
generator.

Focused tests require the new tidal and spin tensors and derivatives to match
independent forms of the retained diagnostic equations. They also require the
mantle/figure ownership distinction, positive inertia, rotational balance,
pair reaction, and orbit-spin torque exchange to close.

## Outcome-blind protocol

The protocol was frozen in
`benchmarks/jx_lunar_mantle_deformation_90_365_holdout.py` before outcome
execution. Its semantic SHA-256 is
`cffcc10fd8fd235ac65a8aefe2049e306604dccea512ae7e3e87dc4e47f3d4e6`.
The create-only preregistration is
`/tmp/jx_lunar_mantle_deformation_90_365_prereg_v1.json`, 2,564 bytes, SHA-256
`0d3bf06936b2738647849762cb45177567ecdb1ff00cbf92a8c97e5f4c2243a0`.

The registered nominal 90- and 365-day endpoints are the nearest exact TAUM
lattice points: 620 TAUM, or 90.0202072035 days, and 2,514 TAUM, or
365.0174208221 days. This preserves exact matched-stage delay history without
introducing a post-hoc interpolation rule.

Control and candidate share the historical header.440 mantle/core initial
state, fluid-core subsystem, DE440 Earth/Sun reference orbit, RKF78 stage
lattice, and scoring epochs. The only candidate increment is the complete
time-variable deformation package. Divisors 8 and 16 form the registered
coarse/fine pair. The fine lane is scored, and the largest control or candidate
coarse-to-fine disagreement is the numerical uncertainty.

Every metric at every endpoint had to improve by at least 1% and by more than
ten times its numerical uncertainty. No parameter was fit.

## Result

| Nominal endpoint | Metric | Static control | Complete deformation | Improvement | Gate |
|---|---|---:|---:|---:|---|
| 90 days | orientation angle | 5.344013e-5 rad | 5.268870e-5 rad | 1.4061% | pass |
| 90 days | inertial mantle-rate error | 2.368783e-11 rad/s | 2.352189e-11 rad/s | 0.7005% | fail |
| 365 days | orientation angle | 4.943898e-4 rad | 4.913890e-4 rad | 0.6070% | fail |
| 365 days | inertial mantle-rate error | 6.677116e-11 rad/s | 6.658027e-11 rad/s | 0.2859% | fail |

All four candidate errors are smaller than their matched controls. The rate
improvements are millions of times larger than the measured coarse-to-fine
disagreement. The orientation coarse/fine principal-angle differences round
to zero at binary64 resolution. The predeclared materiality threshold, not
unresolved numerical noise, rejects three of four cells.

Both lanes passed every validity gate. The divisor-8 and divisor-16 lanes ran
20,112 and 40,224 accepted steps, respectively, for 784,368 total RKF78 stage
evaluations. Their largest rotational-balance residuals were approximately
`7.08e-16` and `7.50e-16` relative. The maximum accepted-step quaternion norm
error before projection was one binary64 epsilon in each lane. No post-start
PCK state entered the dynamics, the mantle inertia stayed positive, and
viscous work never became positive.

The create-only report is
`/tmp/jx_lunar_mantle_deformation_90_365_holdout_v1.json`, 10,918 bytes,
file SHA-256
`3785330dce2e689dc4583467fd6b0b95f8cecdece6dc0858f50c15d6f1da1736`.
Its internal semantic digest is
`2be1ea3c1693d3945a4ccd9b35f873a6e4084751d4662ed301aafbea30fa40eb`.

## Interpretation and next boundary

The implementation is a meaningful engineering advance: the active source
tree now has the complete deformation accounting that the v2 coupled
trajectory lacked, and the long screen shows a consistent beneficial sign.
It is not enough to promote the model. Under the frozen rule, the static
mantle control remains the accepted control and deformation promotion stops.

The result does not authorize a smaller timestep, a looser gate, parameter
tuning, or declaring the Moon solved. It also does not test simultaneous
translation/rotation feedback: DE440 supplies the matched Earth/Sun orbit.
The next scientifically distinct candidate must address omitted physics with
an independently preregistered increment, most plausibly geodetic transport
and extended-Earth/planetary torque, before another untouched long holdout.

The source-fixed geodetic increment has now been implemented and run under a
separate frozen confirmation. It improved none of its four registered metrics
and is closed for promotion; see
`docs/JX_LUNAR_GEODETIC_TRANSPORT_90_365_CONFIRMATION.md`. The older sealed
extended-Earth input screen also remains a negative stop result, so it is not
authorized for an outcome-informed long trajectory.

This report is not raw LLR validation, independent physical validation, exact
DE440 reproduction, registry evidence, production qualification, or release
authority.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest -v \
  tests.test_solar_system_lunar_mantle_deformation \
  tests.test_jx_lunar_mantle_deformation_90_365_holdout

.venv/bin/python tools/run_local_test_matrix.py --validate-only --list
```

The original focused suite passes 9 tests. The existing lunar tide, delayed-tide,
spin-distortion, and coupled-Solar suite passes 26 tests. The exact test matrix
now contains 218 files assigned once across ten profiles. The retained historical
complete-balance diagnostic still requires its optional `erfa` runtime, which
is absent from the active virtual environment; that import gap is not a
failure of the new component.
