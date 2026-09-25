# Data availability and withheld-artifact ledger

## What is published here

This archive contains the decisive milestone reports, compact protocols,
source, registrations, receipts, failure evidence, and the complete compact
XP1 A/B result record. Full-tree checksum inventories bind larger retained
execution trees without placing thousands of checkpoint containers in Git.
For the four engineering inventories named `FULL_TREE`, the published counts
and rows deliberately exclude generated `__pycache__` directories and `.pyc`
files under the archive-wide release policy; the three XP2 run inventories
include every retained file in their declared run roots.

## Generated data retained outside Git

| Retained tree | Files | Bytes | Public inventory | Reason not duplicated |
|---|---:|---:|---|---|
| E1 smoke | 403 | 4,456,641 | `inventories/E1_SMOKE_FULL_TREE_SHA256SUMS` | Mostly binary checkpoints; decisive compact record is published. |
| E1 50-kyr | 998 | 13,108,831 | `inventories/E1_LONG_FULL_TREE_SHA256SUMS` | 900 checkpoints and path-bearing raw output; report and audit are published. |
| E2 numerical forensics | 1,949 | 17,268,418 | `inventories/E2_FULL_TREE_SHA256SUMS` | 1,920 checkpoints and path-bearing raw results; report, manifests, closure, and replay receipt are published. |
| XP2 V1 run tree | 518 | 32,086,376 | `inventories/XP2_V1_RUN_FULL_TREE_SHA256SUMS` | Permanently invalid protocol; compact failure receipts are published. |
| XP2 V2 run tree | 6,022 | 420,305,056 | `inventories/XP2_V2_RUN_FULL_TREE_SHA256SUMS` | Unclassified, replay-defective diagnostics; compact defect evidence is published. |
| XP2 V3 run tree | 12 | 17,167 | `inventories/XP2_V3_RUN_FULL_TREE_SHA256SUMS` | Failed startup; manifest, ledger, and failure proof are published. |
| XP2 protocol-engineering checkpoint | 771 | 14,351,952 | `inventories/XP2_PROTOCOL_ENGINEERING_FULL_TREE_SHA256SUMS` | Rejected/intermediate schemas and host-bound review paths; status is summarized publicly. |

The 23,336,960-byte scientific-reset TAR remains a private preservation copy.
Its SHA-256 is
`6fba35697f5d60c8a5035d0e1f1ff93f3b0e5374f19313489861be3009315fe3`.
It contains duplicated recovery material rather than a new scientific result.

The reset checkpoint also binds four review-only V5 trees: the 31-file
operational-adapter review, 40-file source-successor review, 49-file moving
offline-ceremony review, and incomplete 3-file host-discovery review. Their
counts, bytes, and leaf-list digests are recorded in
`PUBLICATION_DISPOSITION.json` and `08_jx_xp2_v5_scientific_reset/`. They are
not silently omitted: six compact primary status/evidence records are included,
while the remaining bytes are bound by inventory because they are nonauthority
or incomplete engineering material, not scientific data or results.

## Third-party data not redistributed

The local JX-O2 custody package records 25 downloaded files totaling
680,533,216 bytes. These include OSSOS and DES catalog/survey material,
literature-source archives, simulator material, and a published model catalog.
The directory was explicitly quarantined as local-only, its redistribution
terms are not fully audited, and two files exceed GitHub's ordinary 100 MB file
limit. The bytes are therefore not uploaded.

Their filenames, sources where known, sizes, SHA-256 identities, and unresolved
acceptance/license states are published in
`04_jx_o2_g0_local_custody/local_custody_manifest_v1.json` and
`acquisition_checklist_v1.json`. A matching hash proves byte identity only; it
does not prove completeness, eligibility, or permission to redistribute.

## Runtime limitation

This archive is not a self-contained exact rerun environment. The original
temporary Python/REBOUND environment is not retained as a portable public
artifact. A later V5 REBOUND wheel has different native bytes and cannot be
substituted for the historical runtime. The wheel and all compiled binaries are
excluded.

## Future raw-data release

JX-generated checkpoint trees can be packaged separately in a path-neutral
data repository if needed. Any such release must preserve these inventories,
distinguish invalid diagnostics from scientific results, and use a storage
service appropriate for hundreds of megabytes rather than ordinary Git blobs.
