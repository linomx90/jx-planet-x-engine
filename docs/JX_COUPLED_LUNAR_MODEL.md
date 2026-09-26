# Coupled lunar mantle/core screening models

JX `0.6.0rc3` contains two additive, unregistered simultaneous lunar research
boundaries plus separate complete-deformation and geodetic screening
components. Post-rc3 research adds a third, delayed-deformation simultaneous
boundary without rebuilding the sealed release artifacts. All are explicitly
`SCREENING_ONLY`; none changes General Dynamics v18's `INCONCLUSIVE` lunar
mantle/core status.

- `jxplanetx.solar_system.lunar_coupled` is the v1 isolated Earth–Moon
  simultaneous translation/rotation model.
- `jxplanetx.solar_system.lunar_coupled_solar` is the v2 Sun–Earth–Moon model
  with solar forcing/torque and a fixed-pole Earth-J2 orbital correction.
- `jxplanetx.solar_system.lunar_coupled_deformable` is the additive v3 model
  with exact-lattice delayed deformation and common-potential reactions.

## Shared state and numerical contract

All three modules advance a scalar-first lunar mantle body-to-inertial quaternion,
mantle angular velocity, and fluid-core angular velocity in the same fixed-step
RKF78 state as the orbit. Static lunar degree-two force and gravity-gradient
torque come from common potentials, so translational reactions and rotational
torques are applied together. Core–mantle boundary pressure and viscous torque
are equal and opposite.

All provide exact checkpoint clipping, bounded work, accepted-step ledgers,
quaternion projection, owned read-only outputs, and conservation/structure
diagnostics. Caller-supplied parameters remain explicit and immutable.

## v1 isolated Earth–Moon boundary

The v1 state contains Earth and Moon inertial positions/velocities. It includes
mutual monopole gravity, a static lunar degree-two common-potential
interaction, static mantle and oblate-core inertia, and CMB coupling.

Its retained one-day DE440 screen at 12-, 6-, and 3-hour steps produced:

| Metric | Result |
| --- | ---: |
| Successive-grid ratio | `251.5744818254179` |
| Finest total-angular-momentum relative drift | `2.8066486280905437e-16` |
| DE440 Earth–Moon relative-position difference | `95.98697457082537 km` |
| DE440 Earth–Moon relative-velocity difference | `0.0022598922759548535 km/s` |
| Mantle-orientation difference | `5.142126818725099e-08 rad` |

This result established a converged simultaneous chassis and rejected any
claim that the isolated model reproduced DE440.

## v2 Sun–Earth–Moon and Earth-J2 boundary

The v2 state integrates Sun, Earth, and Moon positions/velocities. It adds:

- simultaneous mutual point gravity among all three bodies;
- Sun–Moon and Earth–Moon static lunar degree-two common-potential force and
  mantle torque, including the source reaction for each interaction; and
- an axisymmetric Earth-J2 correction on the Earth–Moon orbit, with the
  GM-weighted Earth reaction.

The Earth symmetry pole is held at J2000 positive Z. Earth spin is not a state,
so full angular-momentum-vector conservation is not claimed for the J2 term.
The valid symmetry diagnostic is total angular momentum projected onto the
held pole. The full-vector change is still reported rather than hidden.

`benchmarks/jx_lunar_coupled_solar_de440_screen.py` starts from exact retained
DE440 SPK and lunar PCK states at JD TDB 2440400.5 and propagates one day at
12-, 6-, and 3-hour steps. The final report is
`/tmp/jx_lunar_coupled_solar_de440_screen_final.json` (file SHA-256
`cdab35965e8f369a1dc9b466c971d2d2d07c3c50208b7aa76d0a277b53b27baa`,
embedded report digest
`a45f2e3a81b5bb3cb213d2a3a4d5a0d39bf2709f406aceca77abd9ac3ef9050a`).

| Metric | Finest-grid result |
| --- | ---: |
| Successive-grid ratio | `52.80901735181783` |
| Fixed-pole angular-momentum relative drift | `3.0316608496718316e-16` |
| Pair-reaction relative residual | `2.4073515871891595e-16` |
| Lunar-figure orbit/spin torque residual | `3.149160757865758e-13` |
| DE440 Earth–Moon relative-position difference | `0.0012509274203315898 km` |
| DE440 Earth–Moon relative-velocity difference | `2.8002684322311473e-08 km/s` |
| DE440 Sun–Earth relative-position difference | `0.255626729264441 km` |
| Mantle-orientation difference | `4.832219920209389e-08 rad` |
| Mantle-rate difference | `1.1622715691427302e-12 rad/s` |

The v2 Earth–Moon position difference is about 1.251 m, a measured
`76732.6489`-fold reduction from v1 on this exact one-day screen. A same-model
ablation with Earth J2 set to exact zero produced 5.170 m; enabling J2 reduced
that error by `4.13298` times. These are descriptive comparisons to the fitted
DE440 ephemeris, not independent validation.

## Verification and scientific boundary

Focused tests establish source/Moon linear reaction, common-potential
orbit/spin torque balance, agreement of the J2 acceleration with the retained
Earth-J2 implementation under unit conversion, zero J2 torque about the fixed
symmetry axis, simultaneous state advancement, nonpositive viscous relative
power, immutable false authorization flags, and fail-closed schedules/work
caps. Report validation rejects DE440 and raw-LLR overclaims.

DE440 integrates the lunar orbit and physical librations together and includes
a fluid lunar core, as described by JPL's
[DE440/DE441 documentation](https://ssd.jpl.nasa.gov/doc/de440_de441.html) and
[Park et al. (2021)](https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf).
Those sources support the architecture, but comparison to the fitted ephemeris
is not independent physical validation.

The v2 model still omits other planets and small bodies, delayed lunar tides
and spin-distortion inertia, Earth spin reaction and a time-varying pole,
extended-Earth-figure rotational torque, relativity, precession/nutation, and
geodetic precession. A post-rc3 LLR layer now parses public ILRS CRD v2 normal
points and screens them with observed IERS EOP, IAU 2006/2000A rotation, the
DE440 lunar principal-axis frame, FCUL atmosphere, and first-order Sun/Earth/
Moon propagation delay. It now also applies IERS solid-Earth tide, source-
bound FES2014B ocean loading, and solid-Earth pole tide. On five public Apollo
15 points the descriptive incomplete-correction RMS is `0.1973876517 m`, down
from `0.3289155079 m` before those displacement increments. This is not
validation. A source-closed one-pass fit to all 3,390 accepted 2006--2017 APOL
points now freezes a six-parameter empirical station offset/velocity state and
reduces nonlinear calibration replay RMS from `0.2622065258 m` to
`0.1590267959 m`. The no-refit result on 899 already-inspected 2018--2020
points is `0.1934077606 m`; it is not a fresh holdout. Authoritative station
discontinuities, ocean pole/other loading, the complete relativistic transfer
function, atmospheric gradients/ray bending, and a source-closed holdout
runner remain absent. The preregistered 2023 holdout remains sealed and has
not been accessed.
See `docs/JX_LLR_OBSERVATION_FOUNDATION.md`.

The complete-deformation and subsequent geodetic 90/365-day promotion gates
both failed under their frozen rules. The v3 simultaneous implementation then
passed all structural and CUDA-parity checks but failed all eight registered
180/730-day physical promotion cells: rotational errors improved by less than
1%, and Earth--Moon translation became slightly worse. Static v2 therefore
remains the accepted control and deformation tuning stops. See
`docs/JX_LUNAR_COUPLED_DEFORMABLE_180_730_HOLDOUT.md`.

The next clean physical gate was not a looser deformation threshold. The
retained eleven-body planetary control already supplies the major planets.
The separately preregistered EIH 1PN and dynamic-Earth-pole J2 translational
screen described in `docs/JX_LUNAR_TRANSLATIONAL_PHYSICS_NEXT_GATE.md` passed
its 14/60/240-day gate. A second, independently frozen 90/365-day confirmation
then reproduced the EIH-only gain at three fresh start epochs: normalized RMS
error was `0.007864801108094175` relative to control `1.0`, and every start
passed its registered replication bound. The dynamic-pole and combined arms
remained diagnostic. This confirms a robust DE440 screening improvement but
does not by itself promote the lunar model, registry, or production claims.

## Reproduction

With the exact retained input root and local SpiceyPy runtime available:

```bash
PYTHONPATH=src:. .venv/bin/python \
  benchmarks/jx_lunar_coupled_solar_de440_screen.py \
  --input-root /path/to/retained/checkpoint/inputs \
  --output /tmp/jx_lunar_coupled_solar_de440_screen.json
```

The output must be an absolute new path with an existing parent. Input identity
changes, dependency absence, numerical nonconvergence, and scientific
overclaims fail closed. Protected retained inputs are only read.
