# JX-XP2 Synthetic 1-Myr Robustness Screen

> **Status: FROZEN EXPLORATORY REVIEW SNAPSHOT — NO SCIENTIFIC RESULT.**
> No authorized V4 XP2 scientific execution has been completed, and this
> directory contains no XP2 outcome classification or evidence for or against
> Planet Nine.

`contract_v1.json` is a byte-preserved historical V4 science/protocol artifact,
not an active public configuration. Its registration-status and local-CPU
permission fields are nonauthorizing historical data. Publishing these files
grants no execution, registration, activation, or result authority.

## What this snapshot contains

This directory preserves the smallest coherent scientific core of an
exploratory synthetic N-body benchmark. It includes the exact protocol, initial
states, deterministic selection manifests, source under review, tests, and two
diagnostic-lineage records. The 14 preserved core files are byte-identical to
the private frozen source set; `SHA256SUMS` records their identities and the
identities of the public wrapper documents.

This is an archival review snapshot, not a self-contained reproduction package.
The preserved contract and programs refer to omitted historical registrations,
runtime locks, receipts, and sibling trees. Do not infer that those dependencies
exist at the public paths, and do not treat this directory as ready to run.

## Scientific question

If a future, separately authorized execution were completed successfully, the
benchmark could measure the change over 1 Myr in the fraction of a deterministic
128-tracer design ever *sampled* at perihelion distance `q < 35 au`, comparing
three fixed distant-perturber cases across eight fixed orientations with one
algebraically shared no-added-body control.

The exact claim ceiling encoded in the historical contract is:

```text
SYNTHETIC_1MYR_8_ORIENTATION_RESPONSE_WITH_DOP853_SENTINEL_ONLY
```

Numerical checks include deterministic replay, a halved MERCURIUS timestep,
comparison of 0.5 Myr and 1 Myr landmarks, and a separate SciPy DOP853 path with
a custom Newtonian right-hand side.

## Design at a glance

- One shared no-added-body control.
- Three fixed added-body cases:
  - 5 Earth masses, 367 au, eccentricity 0.20, inclination 20 degrees;
  - 7.07 Earth masses, 433 au, eccentricity 0.35, inclination 20 degrees;
  - 10 Earth masses, 540 au, eccentricity 0.50, inclination 20 degrees.
- Eight fixed orientations per added-body case: 24 added-body cells.
- One deterministic 128-tracer Latin-hypercube design in eight 16-tracer
  blocks; these are design blocks, not independent population samples.
- MERCURIUS at `dt = 0.125 yr` and `dt = 0.0625 yr` for all 25 configurations.
- A 1 Myr horizon sampled every 50 yr, with 0.25, 0.5, and 1 Myr landmarks.
- A separate 32-tracer, seven-arm SciPy DOP853 numerical sentinel.

The three-case subset and equal design weights are project choices. They are not
a posterior distribution, a parameter-space average, or a claim that these
cases are representative of the real Solar System.

## Exploratory provenance and execution history

The preserved contract records that XP2 design choices were informed by a
reported floor-limited XP1 `q < 30 au` endpoint. XP1 artifacts and results are
not included or independently reassessed in this release. The broader tracer
design, longer horizon, additional orientations, `q < 35 au` endpoint, richer
crossing summaries, timestep audit, and solver sentinel must therefore be
treated as outcome-informed exploratory choices.

Earlier V1/V2/V3 XP2 diagnostic executions or startup attempts existed. The
contract excludes every such artifact from scientific inputs, gates, labels,
classification, and claims. No scientifically admissible XP2 result under this
V4 protocol was generated. Freezing the protocol does not make it an
outcome-blind confirmation.

## Scientific boundary

This snapshot does **not** provide:

- evidence for or against Planet Nine;
- a detection, exclusion, orbit inference, or model preference;
- an observed-TNO population estimate or survey-selection analysis;
- a posterior, p-value, statistical-significance, or power claim;
- a continuous first-passage measurement—`ever q < 35 au` is evaluated only
  on the registered 50 yr sampling cadence;
- Solar-System-age stability, secular-equilibrium, formation-history, or
  present-day population inference;
- independent physical replication from DOP853, which is only a limited
  numerical cross-check.

The model is intentionally idealized. The giant planets begin circular and
coplanar, and the model omits inner planets, general relativity, migration and
gas, cluster evolution, Galactic tide, stellar passages, collisions/removal,
and tracer backreaction. One million years is short compared with the Gyr-scale
integrations used for many Planet Nine studies.

## File groups

- `contract_v1.json`, `initial_states_v1.json`, `seed_manifest_v1.json`, and
  `selection_manifest_v1.json`: frozen scientific design and exact inputs.
- `build_design.py`, `run_primary.py`, `run_independent.py`,
  `run_engineering_boundary.py`, `verify_engineering_boundary.py`, and
  `verify_replay.py`: preserved source under review.
- `test_primary.py` and `test_independent.py`: no-long-dynamics tests from the
  same source set.
- `v2_replay_defect_evidence_v1.json` and
  `v3_failed_startup_evidence_v1.json`: diagnostic lineage required by the
  historical contract.
- `SCIENTIFIC_SCOPE_AUDIT.md`: dated internal, non-peer-reviewed assessment.
- `PROVENANCE.md`, `REFERENCES.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`,
  `LICENSE`, and `SHA256SUMS`: public-release metadata.

No result files, raw trajectories, observed catalogs, private machine paths,
runtime wheels, binaries, credentials, activation records, or ceremony drafts
are included.

## Integrity check

The only supported command for this archival release is the checksum check:

```bash
sha256sum -c SHA256SUMS
```

The checksum file is self-free: it lists every other file in this directory and
does not list itself.

## License and citation

Original project files and release documentation are distributed under the MIT
license in `LICENSE`. External software is not vendored; dependency licenses
and scientific citations are listed in `THIRD_PARTY_NOTICES.md` and
`REFERENCES.md`. Citation metadata is provided in `CITATION.cff`.

