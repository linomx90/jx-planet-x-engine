# JX simultaneous deformable lunar-v3 180/730-day holdout

Date: 2026-09-20

Status: `SCREENING_ONLY`

Decision: `JX_RETAIN_STATIC_SIMULTANEOUS_V2_AND_STOP_V3_PROMOTION`

## Implemented boundary

The additive v3 model advances Sun, Earth, and Moon translation together with
the lunar mantle quaternion, mantle rate, and fluid-core rate. Its only
physical increment over the v2 control is the complete TAUM-delayed lunar
deformation balance:

- delayed Earth-raised tidal and spin-distortion inertia;
- analytic mantle-inertia derivative;
- time-variable mantle and exterior-figure tensors; and
- Earth/Sun force reactions and lunar torques from the same tensor potential.

The caller supplies PCK rotation and DE440 Earth--Moon prehistory only at or
before the start. After the start, delayed rotation, angular acceleration,
and Earth--Moon translation come only from matched RKF78 stage history.
Geodetic transport is excluded because its separate preregistered gate failed.

## Frozen protocol

The new endpoints were fixed at nominal 180 and 730 days after the earlier
90/365-day outcomes were known. Their nearest exact TAUM points are 1,240 and
5,028 delays. Divisors 8 and 16 provide the registered convergence lanes.

Every one of four metrics at both endpoints had to improve by at least 1% and
by more than ten times its matched coarse/fine uncertainty. No parameter was
fit. A 32-step real-CUDA prefix also had to match the CPU reference and repeat
an identical lane bit-for-bit.

- Protocol semantic SHA-256:
  `242696de66c64d57e6904ab54ee20f0ce82577cd434d5497873e4f7f4779aa08`
- Preregistration file SHA-256:
  `4b2de97c81f10824bdf1644a8bd469353d4fd4f869f0c23a70159b7841f8cfff`

## Result

| Endpoint | Metric | Static v2 | Deformable v3 | Improvement | Gate |
|---|---|---:|---:|---:|---|
| 180 d | Earth--Moon position | 2.024282 km | 2.024731 km | -0.02219% | fail |
| 180 d | Earth--Moon velocity | 4.770278e-6 km/s | 4.771333e-6 km/s | -0.02213% | fail |
| 180 d | orientation | 1.795556e-4 rad | 1.778321e-4 rad | +0.95986% | fail |
| 180 d | inertial mantle rate | 3.925635e-11 rad/s | 3.896392e-11 rad/s | +0.74492% | fail |
| 730 d | Earth--Moon position | 18.172820 km | 18.174693 km | -0.01031% | fail |
| 730 d | Earth--Moon velocity | 4.497543e-5 km/s | 4.497996e-5 km/s | -0.01008% | fail |
| 730 d | orientation | 3.831105e-4 rad | 3.821201e-4 rad | +0.25853% | fail |
| 730 d | inertial mantle rate | 1.074313e-10 rad/s | 1.070616e-10 rad/s | +0.34415% | fail |

All eight promotion cells failed. The rotational changes were beneficial and
resolved far above numerical uncertainty, but they were smaller than the
frozen materiality threshold. The translational errors became slightly worse.
The result therefore does not authorize threshold changes, parameter tuning,
or model promotion.

Both lanes passed all structural checks. The v3 candidate completed 40,224
and 80,448 accepted steps (1,568,736 RKF78 stage evaluations total). The
largest pair-reaction residual was `2.92e-16`, orbit--spin torque-balance
residual `3.86e-12`, and rotational-balance residual `7.44e-16`. Mantle
inertia stayed positive, viscous power never became positive, and no
post-start external history entered the dynamics.

The v2 control required one extra clipped endpoint step in the divisor-8 lane
and two in the divisor-16 lane because its older absolute-epoch loop does not
use integer delay indices. This is disclosed in the report. The measured
control coarse/fine differences remain far below the physical changes, and
the stop decision does not depend on relaxing this discrepancy.

## CUDA parity

The actual retained-input prefix passed on the NVIDIA GeForce RTX 5060 Ti
(compute capability 12.0, CuPy 14.2.0). Two identical lanes were bit-identical.
CPU/CUDA maximum differences were zero for position, velocity, quaternion,
and core rate; mantle rate differed by about `1.40e-25 rad/s`.

This is a vectorized CuPy trajectory path, explicitly not a fused kernel and
not general CUDA qualification.

## Evidence and next gate

The create-only report is
`/tmp/jx_lunar_coupled_deformable_180_730_holdout_v1.json`, 16,264 bytes,
file SHA-256
`168b0d4dac0101630894dc1c0cf3a3923170263e4203d53154591393df276832`.
Its internal semantic digest is
`bd4d55c5f0967b3d7cc0ebbca7c64b24ca56596fde20cb0909070eb06a2f2cb7`.

Deformation tuning stops. The next defensible lunar step is a separately
preregistered translational model that adds the major missing planetary,
relativistic, and Earth-orientation physics before attempting raw LLR
observation reduction. This report is not DE440 reproduction, raw LLR
validation, independent validation, registry evidence, or production proof.
