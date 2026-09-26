# JX General Dynamics 0.6.0rc16

Release date: 24 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc16 turns the retained coupled lunar equations into a supported public engine
component. `integrate_lunar_ephemeris_v1` advances the resolved eleven-body
Solar-System roster, lunar mantle quaternion and angular velocity, and lunar
fluid-core angular velocity through one fixed-step RKF78 checkpoint contract.
Every stage evaluates fully mutual Newtonian gravity, mutual EIH 1PN, reacting
Sun--Moon and Earth--Moon static lunar quadrupoles, reacting fixed-axis Earth
J2, and mantle/core pressure plus viscous coupling. Inputs and outputs are
owned, read-only, provenance-bound, and protected by deterministic ledgers and
content hashes.

The outcome-blind 90/365-day protocol passed every frozen long-arc gate. At
365 days, the candidate's Earth--Moon position error was 0.061568 km versus
3.729707 km for the no-EIH control; Sun--Earth error was 0.029261 km versus
56.094548 km. The aggregate candidate/control RMS ratio was 0.012199 and the
worst individual ratio was 0.018686.

Two independent release-builder invocations, each performing two clean
package-only builds, produced byte-identical artifacts. Fresh installations of
both artifacts imported rc16 and exposed the lunar ephemeris capability as
`IMPLEMENTED`.

## Artifacts

- `jxplanetx-0.6.0rc16-cp314-cp314-linux_x86_64.whl`
  - SHA-256: `f54ac18fb074b25430155dde2bcdd9fdf2d3dc2c96765ba191ed955112fd799f`
- `jxplanetx-0.6.0rc16.tar.gz`
  - SHA-256: `8f0a9964fca26acf0ba2cc7ab887b63c5544cc45be850fd43bdca8d41722e4b0`

## Claim limits

This component remains NumPy/CPU screening code. It omits delayed tides,
time-variable lunar deformation, lunar degree three and higher, terrestrial
tesserals and higher zonals, minor bodies and planetary satellites,
observation reduction, and parameter fitting. Passing a DE440 screen does not
establish an independent ephemeris, navigation fitness, raw-LLR validity, or
general superiority over REBOUND.

The General Dynamics registry v18 remains immutable historical evidence bound
to `0.6.0rc3`; this package release does not alter that registry.
