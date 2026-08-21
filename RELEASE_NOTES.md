# Release notes

## 0.1.1 — 2026-08-21

Provenance hotfix:

- Fixed a runtime source-manifest bug that could record an empty software
  manifest when JX was executed from an installed package.
- Installed-wheel validation hashes all 12 packaged Python source files and
  passes all 5 provenance gates.
- The source-checkout test suite passes 16/16 tests.
- Numerical dynamics are unchanged.
- Scientific claim state remains `SCREENING_ONLY`; this release does not claim
  a Planet X detection.

## 0.1.0 — 2026-08-20

Initial public release of the falsification-first JX Planet X Scientific Engine:

- deterministic arbitrary-precision N-body dynamics;
- sixth-order Yoshida integration and an independent Decimal Bulirsch–Stoer
  reference;
- convergence, conservation, provenance, and claim-control gates;
- preserved DE441 and numerical benchmark records; and
- the governing compact-source result
  `BLOCKED_SOURCE_POPULATION_NONCONVERGENCE`.

Version 0.1.0 did not claim a Planet X detection or sky localization.
