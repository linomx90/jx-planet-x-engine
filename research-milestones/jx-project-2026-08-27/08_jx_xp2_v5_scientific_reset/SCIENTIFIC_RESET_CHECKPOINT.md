# XP2 scientific-reset checkpoint — 2026-08-27

This checkpoint preserves the complete durable project evidence and the active
review-package sources at the moment the work was reset for a scientific-validity
audit. It is a preservation artifact only. It authorizes no compilation,
registration, dynamics, simulation, publication, or scientific claim.

## Status at capture

- Package A source-successor review package: frozen dual GO.
- Accepted Package A/Package B/149-document integration evidence: preserved in
  the power-loss checkpoint tree.
- Package B Ubuntu ceremony review package: moving, invalid, and NO-GO; all edits
  stopped before this capture.
- Ubuntu host-discovery review package: incomplete, 3 of 8 planned leaves.
- Scientific execution: not started.
- Scientific result or evidence for Planet Nine: none.
- Active next step: read-only audit of hypothesis, provenance, controls,
  ensemble, horizon, statistics, numerical validity, and publication novelty.

## Immutable archive

- File: `XP2_SCIENTIFIC_RESET_SNAPSHOT_V1.tar`
- Raw size: 23,336,960 bytes
- SHA-256: `6fba35697f5d60c8a5035d0e1f1ff93f3b0e5374f19313489861be3009315fe3`
- Mode at verification: `0444`
- Link count at verification: `1`
- Tar entries: `984`
- Creation time recorded by the coordinator: `2026-08-27T22:25:37-04:00`

The archive contains these exact workspace-relative trees:

| Tree | Regular files | Raw bytes | Symlinks | Leaf-list digest |
|---|---:|---:|---:|---|
| `outputs/jx_xp2_powerloss_checkpoint_v1` | 783 | 15,386,017 | 0 | `a76d25b9a86ff72be67664359cb79ec7450566490517fef1bfbbf231a0cc7e3c` |
| `work/jx_xp2_robustness_v5` | 18 | 2,366,612 | 0 | `4036f9ba72f83396295cc1c16e38a819039ddfa7d85958d76a8728349ad622b9` |
| `work/jx_xp2_v5_operational_adapter_v1` | 31 | 70,384 | 0 | `4c6b7ebf71ce16d96669a693f19d495f338b9f2a4cc9355426558972a5d22fcd` |
| `work/jx_xp2_v5_source_successor_review_v1` | 40 | 2,344,285 | 0 | `e5082c2ea14122ba87ef80b6236b462956d7daea45a072a2983ea9ec20fed871` |
| `work/jx_xp2_v5_offline_ceremony_review_v1` | 49 | 1,456,776 | 0 | `116d482f5ca79d889518c31cedf37e6b5cf289435c4feac1de54cc7cd8aa6426` |
| `work/jx_xp2_v5_ubuntu_host_discovery_review_v1` | 3 | 44,159 | 0 | `c42bfc61748825122884562da8e4f11f1ff60e2be063efd98b48860e2cf9da07` |

Each leaf-list digest is SHA-256 over the ordinary `sha256sum` rows for all
regular files in bytewise-sorted workspace-relative path order. The archive
SHA-256 is the authoritative preservation identity.

## Recovery rule

Never overwrite an accepted or historical tree with this archive. Recovery
must extract into a fresh private directory, verify the archive SHA-256 first,
and compare the desired tree byte-for-byte before any reviewed successor work.
The moving Package B and discovery trees remain nonauthority evidence even
after recovery.
