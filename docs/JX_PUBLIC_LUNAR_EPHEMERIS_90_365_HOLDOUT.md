# JX public lunar ephemeris v1: historical 90/365-day report

Original report date: 2026-09-24

Repository audit date: 2026-10-05

Original package version: `0.6.0rc16`

Current evidence status:
`HISTORICAL_SCREENING_REPORT_NOT_REPRODUCIBLE_FROM_RETAINED_ARTIFACTS`

## Evidence warning

This file preserves numbers recorded by the original screening report. It is
not a reproducible qualification record and it does not authorize a scientific
or production ephemeris.

The original report named a preregistration and result under `/tmp`. Those
files are not retained in this repository. The benchmark and holdout-test
modules named by the original reproduction commands are also absent. The
repository therefore cannot verify the preregistration digest, report digest,
input state, control implementation, endpoint calculations, or claimed gate
decision from retained evidence.

The public lunar run contract has since been hardened. Its result and
accepted-step ledger use v2 digest domains, explicitly interpret every epoch
as SI seconds on the TDB scale since J2000 TDB (JD 2451545.0), and reject
inconsistent step, checkpoint, provenance, parameter, or digest records. The
legacy hashes below predate that contract and are not hashes of a current
`LunarEphemerisV1Run`.

## Historical question and reported gate

The historical screen reportedly asked whether the mutual EIH 1PN increment
improved untouched DE440 translational endpoints at nominal 90 and 365 days
relative to a matched no-EIH control. Both arms reportedly used the same
initial state, fixed RKF78 step, Newtonian point masses, reacting static lunar
quadrupoles, reacting fixed-axis Earth J2, mantle rotation, and fluid-core
subsystem.

The report recorded semantic protocol SHA-256
`27b1d225d4773b96763407a9acccae5a99bc57a3d7f7b6fbc72545a182ea73b2` and
preregistration-file SHA-256
`aafc027092d39bd442617502d276c3354d6e8502ecb4c8738746fd059328588a`.
Neither can currently be checked because the corresponding bytes are absent.

The reported gate required every candidate metric to improve, an aggregate
normalized RMS candidate/control ratio below `0.25`, every individual ratio
below `0.5`, finite output, and exact work accounting. No parameter was
reported as fitted.

## Preserved reported numbers (unverified)

| Endpoint | Metric | Reported EIH candidate | Reported no-EIH control | Reported ratio |
|---|---|---:|---:|---:|
| 90 days | Earth–Moon position | 0.013980 km | 0.874538 km | 0.015986 |
| 90 days | Earth–Moon velocity | 4.0902e-8 km/s | 2.3367e-6 km/s | 0.017504 |
| 90 days | Sun–Earth position | 0.012435 km | 6.653770 km | 0.001869 |
| 90 days | Sun–Earth velocity | 3.3708e-9 km/s | 2.1075e-6 km/s | 0.001599 |
| 365 days | Earth–Moon position | 0.061568 km | 3.729707 km | 0.016507 |
| 365 days | Earth–Moon velocity | 1.8471e-7 km/s | 9.8849e-6 km/s | 0.018686 |
| 365 days | Sun–Earth position | 0.029261 km | 56.094548 km | 0.000522 |
| 365 days | Sun–Earth velocity | 8.1235e-9 km/s | 1.1074e-5 km/s | 0.000734 |

The historical report also recorded an aggregate ratio of
`0.012198720845768172`, largest individual ratio of
`0.018685875377906735`, 11,680 accepted steps, 151,840 force evaluations,
accepted-step ledger SHA-256
`3b440e03396fa41bc3274b9373ce84f1895f34e843cba89f272ced71698f1005`, and
result-content SHA-256
`a3d00e079ddf3f507cf6fa7cbba50549a097799f9cc7ab6b45615a03a938a25f`.
These are preserved as historical assertions, not revalidated results.

## Interpretation and limits

Even if independently reproduced, this comparison would be a
`SCREENING_ONLY` test against a fitted reference. It would not independently
validate DE440, score lunar attitude against a held-out orientation reference,
validate raw LLR, or qualify an astronomical ephemeris.

The v1 model omits solar J2, delayed tides, time-variable lunar deformation,
lunar degree three and higher, Earth higher zonals and tesserals, minor
bodies, planetary satellites, observation reduction, and parameter fitting.
Organizational independence was reported false. No navigation, production,
publication, ephemeris-qualification, or general-superiority claim follows.

## Requirements for a new reproducible holdout

A replacement run must retain, inside an immutable evidence package:

1. the preregistration created before execution;
2. exact initial state, constants, provenance, epoch origin, and reference
   endpoint bytes;
3. candidate and independently identified control source/runtime identities;
4. the executable benchmark and adversarial validator;
5. complete candidate/control outputs and current v2 run records;
6. a manifest binding every file by size and SHA-256; and
7. CI tests that validate the retained package without consulting `/tmp`.

Until such a package exists and passes, the historical decision must not be
presented as a reproducible current pass.
