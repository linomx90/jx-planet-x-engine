# JX Planet X project milestone ledger

> **Status: FULL-TRANSPARENCY WORK-IN-PROGRESS CHECKPOINT THROUGH 2026-08-27
> — MIXED VALIDITY — NO OBSERVATIONAL PLANET NINE RESULT.**

This dated checkpoint publishes the project's meaningful milestones so far,
including failures,
invalid attempts, inconclusive diagnostics, blocked gates, and the one completed
exploratory synthetic response experiment. It is a retrospective record, not a
claim that every milestone succeeded or that the JX project is closed. Work may
continue after this checkpoint through newly versioned protocols and artifacts.

As of 2026-08-27, the project contains software validation, engineering
experiments, numerical diagnostics, an exploratory synthetic XP1 result,
invalid XP2 attempts, and blocked observational-design work. It contains no
observational Planet Nine result and no scientifically admissible XP2 result.
Nothing here detects, confirms, excludes, constrains, or prefers Planet Nine.

## Status matrix

| Milestone | Canonical status | What may be concluded |
|---|---|---|
| Existing public engine foundation | `SCREENING_ONLY` | Numerical/software screening within each published contract. |
| JX-E1 hardened 100-year smoke | `ENGINEERING_SMOKE_VALID` | The hardened synthetic machinery and replay checks completed. |
| JX-E1 50-kyr engineering run | `ENGINEERING_LONG_INVALID` | The locked configuration failed its conservation standard; execution B was not run. |
| JX-E2 numerical forensics | `MIXED_OR_INCONCLUSIVE` | Exact same-build replay; descriptive numerical patterns only. |
| JX-O2 local custody | `JX_O2_G0_LOCAL_CUSTODY_VERIFIED_BLOCKED` | Public-input bytes were inventoried locally; 31 acceptance requirements remain unresolved. |
| JX-O2 open regeneration | `LOCAL_DESIGN_HASH_LOCKED_NOT_EXTERNALLY_PREREGISTERED` | A public-model design exists; no model, seed, checkpoint, simulation, or result exists. |
| JX-XP1 synthetic response | `PRACTICALLY_SMALL` on the locked sampled q<30 endpoint | The exact 250-kyr synthetic design had zero q<30 count contrast; the endpoint was floor-limited and distribution responses were nonzero. |
| JX-XP2 V1–V3 | `HISTORICAL_INVALID_NONAUTHORIZING` | Failure and defect lineage only; no admissible XP2 result. |
| JX-XP2 V4 public snapshot | `FROZEN_EXPLORATORY_REVIEW_SNAPSHOT` | Protocol/source review only; no execution or result. |
| JX-XP2 V5 engineering | `NONAUTHORITY_REVIEW_ONLY` | Static schema/source review milestones only; no registration or simulation. |
| Scientific reset | `PRESERVATION_ONLY` | The project state was preserved before a scientific-validity review. |

## Archive layout

- `01_jx_e1_smoke/`: final hardened engineering-smoke protocol and replay
  receipt.
- `02_jx_e1_50k_invalid/`: the failed 50-kyr engineering milestone, report,
  contract, runner, verifier, and post-failure audit.
- `03_jx_e2_numerical_forensics/`: the mixed/inconclusive numerical-method
  diagnostic and its replay/closure records.
- `04_jx_o2_g0_local_custody/`: metadata, checklist, and verifier for the
  blocked local-input custody package; third-party payload bytes are absent.
- `05_jx_o2_open_regeneration/`: the design-only public-regeneration protocol.
- `06_jx_xp1_exploratory/`: the complete compact XP1 protocol, A/B results,
  receipts, and public result report.
- `07_jx_xp2_invalid_lineage/`: compact V1–V3 protocol-failure and replay-defect
  lineage. These files are historical and nonauthorizing.
- `08_jx_xp2_v5_scientific_reset/`: the preservation checkpoint status and a
  public record of later review-only engineering, including its accepted static
  integration gate, blocked independence evidence, incomplete packages, and
  scientific reset.
- `inventories/`: checksums for retained raw execution and engineering trees
  that are not duplicated in this Git repository.

The separately published XP2 V4 review snapshot is in
`runs/jx_xp2_synthetic_robustness_review_v1/` on the same release branch.

## Evidence classes

- **Narrow result:** XP1 only, limited to its exact deterministic synthetic
  design and sampled endpoint.
- **Engineering result:** E1 smoke and E1 50-kyr, with the latter invalid.
- **Numerical diagnostic:** E2, explicitly mixed/inconclusive.
- **Design or readiness record:** JX-O2 and XP2 V4/V5.
- **Invalid diagnostic lineage:** XP2 V1–V3.
- **Integrity-only GO/PASS:** checksum, replay, schema, registration, and static
  review labels. These are not scientific evidence.

## Retrospective publication

Several local registrations were content-hash locks without an independent
external timestamp. Publishing them now does not retroactively make them
externally preregistered or outcome-blind. XP2 was informed by the reported XP1
floor-limited q<30 endpoint and must remain exploratory.

## Data availability

See `DATA_AVAILABILITY.md` and `PUBLICATION_DISPOSITION.json`. Our generated
scientific data are represented by reports, compact results, receipts, and full
tree checksum inventories. Bulky binary checkpoints are not duplicated in Git.
Downloaded survey/catalog/source archives are third-party material with
unresolved redistribution terms and are described by exact custody metadata
rather than republished.

## Integrity

`SHA256SUMS` is self-free: it covers every other file in this archive and does
not list itself.

## License

JX-owned source, reports, metadata, and release documentation are provided under
the MIT license in `LICENSE`. Third-party software and data are not vendored;
see `THIRD_PARTY_NOTICES.md` and `DATA_AVAILABILITY.md`.
