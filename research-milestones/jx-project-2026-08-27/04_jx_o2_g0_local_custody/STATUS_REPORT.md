# JX-O2 G0 local acquisition status

## Outcome

The useful public inputs already present on this computer are now preserved in
a content-addressed, fail-closed local custody package.

- Status: `JX_O2_G0_LOCAL_CUSTODY_VERIFIED_BLOCKED`
- Preserved files: 25
- Preserved bytes: 680,533,216 (about 649 MiB)
- Explicit unresolved acceptance requirements: 31
- G0 complete: `false`
- Eligible for G1: `false`
- Dynamics, GPU, and observed-data execution authorized: `false`
- Claim ceiling: `ACQUISITION_CHECKLIST_ONLY`
- Registration SHA-256:
  `ad94637de5b2062213b8d86726b15f256ae3a14522382ad0194e95b28ae337d7`

The package passed its byte, schema, policy, path, and fail-closed verification
and all 8 focused tests. Verification requires the registration hash above as
an external anchor; an incorrect anchor is rejected.

## What is now safely preserved

The quarantine contains the available OSSOS publication catalogs and partial
simulator material, the DES catalog and selection payloads, four relevant
paper-source archives, and the Brown--Batygin 100,000-row orbit catalog. Every
listed local file has a verified size and SHA-256.

Byte custody does not make a file scientifically eligible. In particular:

- no withdrawn 2019 OSSOS `Deep_Surveys` payload was copied;
- the A16 metadata ReadMe and HiLat source archive are explicitly marked as
  unbound local copies requiring authoritative reacquisition and license audit;
- candidate 9118 remains retired and cannot enter JX-O2;
- engineering runs JX-E1 and JX-E2 cannot substitute for the missing physical
  model inputs.

## Why JX-O2 remains blocked

The complete, authoritative later OSSOS characterization bundle is still
missing, including corrected detection/efficiency/pointing files, count and
alias reconciliation, exact historical simulator/runtime bindings, and clear
software and data licenses.

DES still requires a locked release choice, eligibility/count rules, epoch and
covariance policy, exposure reconciliation, full selection lineage, dependency
and RNG locks, qualification, and license resolution.

The physical-model side still lacks an execution-ready matched M0/M1 pair: no
common physical `cluster_2` checkpoint, full finite state family, paired decks,
weights, selection lineage, deterministic seeds, duration/timestep/tolerances,
survival rules, intrinsic population nuisance model, model-to-survey adapter,
phase/orientation policy, or hardware-determinism contract is complete.

## Correct next sequence

1. Acquire authoritative survey and physical-model artifacts from their
   custodians or a preregistered open-regeneration route.
2. Preserve exact URLs, versions, licenses, byte counts, and SHA-256 values.
3. Independently audit every one of the 31 requirements. A link, comment, hash
   match, partial bundle, or software license alone is insufficient.
4. Only after every G0 requirement is satisfied, issue a separate immutable G0
   resolution receipt and draft the G1 execution contract.
5. Run calibration and independent reproduction under that later contract.
   Observed-data execution remains unauthorized until its later registered
   gates pass.

More particles or a GPU run cannot repair missing provenance. This package
contains no Planet X result and provides no evidence for or against Planet X.

## E2 closure

The separate JX-E2 numerical-method investigation is frozen locally as
`MIXED_OR_INCONCLUSIVE`. Its local closure SHA-256 is
`4cec842a5f5b6f22a97f0b1b5dfaaf4bcddfe7efbe5c6836bf3772ef7724d901`.
It was a same-build numerical diagnostic, not a scientific Planet X test, and
does not unblock JX-O2.
