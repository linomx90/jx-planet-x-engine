# JX-O2 candidate 9118 retirement receipt

This compact receipt permanently removes Brown–Batygin catalog index 9118 from
the JX-O2 model-family comparison. It records a scope decision, not recovered
provenance and not a scientific outcome.

## Why retirement is required

The retained project evidence contains a byte-audit record and tuple matching
the public catalog row, historical JX state files, and a checksum expected for
`candidate_metadata.txt`. It does not contain that metadata file or a
contemporaneous record of who selected index 9118, when it was selected, the
candidate universe, selection rule, ranking, or prior inspection history. The
historical state builder also redeclared the catalog elements from JD 2458270.0
at JD 2461200.5 without physical propagation. The retained checksum repairs
neither defect: the builder only hash-checked the external file and used a
separately hard-coded tuple. Any recovered blob would still require content
audit, and the epoch defect would still require propagation or refit.

The search covered the complete retained non-shallow Git object database,
refs, reflogs, stashes, notes, tracked and untracked project files, linked work
copies, the recorded scratch path, exact-hash matches among accessible regular
files smaller than 1 MiB, and the public repository's branches, commits,
issues, pull requests, and plausible paths. Large files and archive interiors
were not exhaustively hash-scanned. The conclusion is limited to retained and
accessible evidence; it does not claim that the file never existed elsewhere.

## Binding decision

Within experiment `jx-o2-characterized-survey-model-comparison-design-v1`,
candidate 9118 is permanently `EXPLORATORY_SCREENING_ONLY`. It cannot be an M1
member, family anchor, grid point, prior influence, calibration target, power
case, physical state, or observational input. The historical screening files
and results remain pre-existing records and are not reinterpreted by this
receipt. Their release manifest and principal result/audit files are bound by
hash.

The historical candidate-9118 numerical outcomes were already public and had
been inspected before this retirement decision. Retirement is based only on
the missing selection provenance and invalid JX-O2 epoch lineage, not on those
numerical outcomes. Those outcomes cannot be used to tune the future M1
family, grid, or weights.

This receipt is the stricter governing policy for candidate 9118 in this JX-O2
experiment and replaces the design manifest's earlier conditional possibility
of using it after a lineage audit. No later receipt or version may reinstate it
under the same experiment ID.

This disposes of the candidate-specific dependency by scope exclusion. It does
not satisfy the original recovery requirement, repair the epoch reassignment,
provide the still-missing matched physical M0/M1 family, satisfy activation
gate A03, complete G0, or authorize any run. A different future experiment
could reconsider the catalog row only through a new preregistration and fresh,
independently audited provenance.

## Files

- `candidate_9118_retirement_v1.json` — substantive search record, retirement
  policy, immutable prior-artifact bindings, and fail-closed state.
- `registration_retirement_v1.json` — timestamped hashes for this README, the
  substantive receipt, and its same-change integrity tests.
- `../../tests/test_jx_o2_candidate_9118_retirement.py` — same-change integrity,
  immutability, scope, claim-control, and future-promotion guards.

The original JX-O2 design, G0 audit, and historical numerical artifacts remain
byte-for-byte unchanged.
