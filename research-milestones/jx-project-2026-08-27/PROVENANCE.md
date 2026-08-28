# Publication provenance

The milestone files were selected from the preserved local project state on
2026-08-27. Their release is retrospective. It does not alter their original
status, create an earlier public timestamp, or supply missing authority.

All selected files were copied byte-for-byte except the six documents listed
below. Obsolete private temporary-environment command prefixes were replaced by
the neutral command `python3`; the E1 command's filenames were also corrected
to the published V3 runner and contract. Three retrospective result documents
were changed only to replace an unqualified `preregistered` label with the
truthful local pre-output content-hash-lock status and lack of an independent
external timestamp. No scientific value, threshold, canonical status, numeric
result, or hash-bearing machine evidence record was changed.

For the three corrected result documents, the exact original bytes are also
published beside the reading copy as `RESULT_REPORT_ORIGINAL.md`,
`FINAL_REPORT_ORIGINAL_HASH_BOUND.md`, and
`xp1_synthetic_response_result_original.md`. Those originals are retained for
historical byte identity, not as the archive's authoritative interpretation of
timestamp or preregistration status. The registration-bound E2 README itself is
not republished because it contains a private ephemeral runtime path; its exact
identity remains recorded by the original SHA-256 below and the E2 full-tree
inventory.

| Public file | Original SHA-256 | Public SHA-256 | Change |
|---|---|---|---|
| `01_jx_e1_smoke/README.md` | `98aa894cee4331a17bdfabcf1fe39e6d9c95444a1193118b9425670a1d0cf526` | `d05e8909ff235d0e64e92cf7e4bda2d6258c8a06d560413024eeaeb43deb4ec8` | Removed obsolete private `/tmp` interpreter prefix, corrected the command to `run_pilot_v3.py --contract contract_v3.json`, and marked it as historical/nonportable. |
| `03_jx_e2_numerical_forensics/README.md` | `d4d097841adf09ce1f57636a8129185602d5ddd1f8c0148e07376de463888f8f` | `432211a9fa1cb08c4d77442481404c107e064f9d7722d6fcbf0913bb73faee83` | Removed obsolete private `/tmp` interpreter prefix and added the public-reading-copy/runtime boundary. |
| `02_jx_e1_50k_invalid/RESULT_REPORT.md` | `35a20733b2d42e9ba259de3820fcd090a382ce2d23c03d9e06a5fab7b56404b8` | `a5c001a3229e03d5bd647031ad55ece6255f6ba681ed7c1e80543588424fbb05` | Qualified the local content-hash lock and absence of an external timestamp. |
| `03_jx_e2_numerical_forensics/FINAL_REPORT.md` | `9da2903bfca951ead02bc441bdab8e07ae7ac434ce1d32e184b11e90f25f9804` | `81d798b283292e7aa7c2b8862448332ea393cee8d9dd626e3f1a0e8c0abf34b2` | Qualified the local content-hash lock and absence of an external timestamp. |
| `06_jx_xp1_exploratory/README.md` | `6a56df4efa8c2c002f298523f5fb9deffa517978703e415c53e8440f63944804` | `1430088214c4725c7853bf9db87c70534668133755f2466fbdccaabf3dd386cb` | Added the reading-copy versus immutable-original publication note. |
| `06_jx_xp1_exploratory/xp1_synthetic_response_result_v1.md` | `34d313e874db8ec469d3fa60e65750cd613779936e26e4c07c21d22f771f8abc` | `bd2f46f2aef830b4152f99b896bb93cf26bfff3f94e41bc8f913412398d8df4c` | Replaced the unqualified preregistration label with the exact local pre-output hash-lock status and lack of an external timestamp. |

The five local Git worktrees were not copied. Their substantive engine,
JX-O2 audit, candidate-retirement, preregistration, and public-reconstruction
milestones are already present in repository history. Duplicate worktrees and
Git metadata are not new research data.

The earlier 12-file XP2 publication candidate is superseded and rejected. The
23-file V4 review snapshot in the same pull request is the accepted public XP2
protocol/source-review milestone.

The literal `/private/absolute/secret/input.json` present in historical test
source is a synthetic redaction fixture, not a credential or real path.
