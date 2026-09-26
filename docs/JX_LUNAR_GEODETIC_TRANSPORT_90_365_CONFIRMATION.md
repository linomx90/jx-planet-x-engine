# JX Lunar Geodetic-Transport 90/365-Day Confirmation

Status: completed negative incremental confirmation. Scientific claim state:
`SCREENING_ONLY`.

## Question and lineage

This comparison asked whether the source-fixed DE440 geodetic-precession rate,
inserted in the relative-frame gyroscopic terms of the complete delayed
mantle-deformation and fluid-core equations, improves the retained lunar
orientation trajectory.

The matched control and candidate both use delayed tidal deformation, delayed
spin deformation, the analytic combined inertia derivative, deformed
Earth/Sun gravity-gradient torques, and the same fluid-core subsystem. The
candidate adds only geodetic transport. `CLIGHT`, PPN `GAMMA`, Earth GM, and
Sun GM come from the retained bound DE440 inputs; there are no fitted
parameters.

The control reproduced the prior deformation candidate endpoints bit-for-bit
in both numerical lanes. The prior report remains bound by file SHA-256
`3785330dce2e689dc4583467fd6b0b95f8cecdece6dc0858f50c15d6f1da1736`
and internal report digest
`2be1ea3c1693d3945a4ccd9b35f873a6e4084751d4662ed301aafbea30fa40eb`.

## Registration

The frozen protocol SHA-256 is
`e70eac9fe4e3d4789e8d0255dec1ba4fc88867f6e0257a8f6a9ddcad9efd6271`.
The create-only preregistration is
`/tmp/jx_lunar_geodetic_transport_90_365_prereg_v1.json`, 3,128 bytes,
SHA-256
`af0ddeb3cd68c643e9ce0fcbf0aa6bd0c6d946361fd24148477c2087f752c49d`.

The protocol retained the previous one-percent materiality threshold and the
strict improvement-greater-than-ten-times-numerical-uncertainty rule for both
orientation angle and inertial mantle-rate error at both horizons. The 90- and
365-day endpoint roster was reused after the deformation holdout. The
geodetic candidate was unseen when registered, but this is therefore an
incremental confirmation and not a pristine endpoint-selection holdout.

## Execution validity

The two lanes used exact response-delay fractions of 1/8 and 1/16. They
completed 20,112 and 40,224 accepted RKF78 steps, respectively, for 784,368
total stage evaluations. All registered validity checks passed.

The coarse/fine maximum rotational-balance residuals were
`7.348739428785807e-16` and `7.605173434063519e-16`. The maximum quaternion
preprojection norm error was `2.220446049250313e-16` in both lanes. Mantle
inertia stayed positive, viscous power was never positive, no post-start PCK
state leaked into delayed dynamics, and the maximum geodetic-precession rate
was `1.624964354240232e-16` rad/s.

## Registered results

| Horizon | Metric | Control error | Candidate error | Improvement fraction | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| nominal 90 d | orientation angle | `5.268870230475851e-05` rad | `5.268870230475851e-05` rad | `0` | fail |
| nominal 90 d | inertial mantle rate | `2.352188861685924e-11` rad/s | `2.3521982825302932e-11` rad/s | `-4.005139435257622e-06` | fail |
| nominal 365 d | orientation angle | `4.913890479355609e-04` rad | `4.913890687216402e-04` rad | `-4.2300656544521885e-08` | fail |
| nominal 365 d | inertial mantle rate | `6.658026977225119e-11` rad/s | `6.658069559349158e-11` rad/s | `-6.395607014597985e-06` | fail |

The 90-day orientation change was below binary64 principal-angle resolution.
The other three changes were resolved in state/rate space but harmful or far
below the registered one-percent materiality requirement. No registered
metric passed.

Decision:
`JX_CLOSE_GEODETIC_TRANSPORT_PROMOTION_AND_RETAIN_PRIOR_STATIC_DEFAULT`.

The create-only report is
`/tmp/jx_lunar_geodetic_transport_90_365_confirmation_v1.json`, 12,154 bytes,
file SHA-256
`eba398b9fcefb7e3b6f7db04261aff10892a6fadafad6d87d0278107aeac8991`,
and internal report digest
`98db6222494de631498cced8f46cc165f686773aa187cc29797081ff7acff31e`.

## Interpretation and next boundary

Geodetic transport is now implemented as a reusable, source-fixed research
primitive and is correctly placed in both mantle and core relative-frame
terms. It is not supported for promotion by this trajectory comparison and
does not change the engine default.

The previously sealed extended-Earth input screen also failed its registered
advance gate: its aggregate closure did not improve and its median scale was
below one percent. That closed branch must not be revived by silently moving
it into a long trajectory after seeing the present outcome. A future direct
planetary-torque or simultaneous orbit/rotation increment requires its own
outcome-blind materiality screen and preregistration. None of these results is
raw LLR validation, an exact DE440 generator, production qualification, or a
claim that the lunar model is solved.
