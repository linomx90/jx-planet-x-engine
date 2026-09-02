# JX Reality Test 1 — First Real-Retrieval Attempt: Blocked Checkpoint

**Date:** 2026-09-02 UTC  
**Classification:** `WORKFLOW / CURATION ATTEMPT ONLY`  
**Status:** `REAL_RETRIEVAL_BLOCKED_BEFORE_NETWORK_REQUEST`  
**Target:** 367943 Duende (2012 DA14)  
**Workflow run:** `33686072499`  
**Workflow commit:** `10875b3c731243e5cc1853167841042328104991`  
**Artifact ID:** `9868145137`

## What happened

The one-time retrieval workflow passed all frozen-input, runtime-source, archive-member, public-key-only, and ephemeral-environment preflight checks. It then stopped before the adapter process began because the runner shell could not create the redirected stdout file inside `/tmp/jx-da14-export-parent`:

```text
/tmp/jx-da14-export-parent/retrieval_stdout.json: Permission denied
```

The failure was in workflow output plumbing, not in the MPC/JPL source schemas, network transport, normalization, encryption, or curation algorithms.

## Evidence state

- No adapter network request was initiated.
- No DA14 optical or radar records were downloaded.
- No raw HTTP response was retained.
- No complete normalized plaintext dataset was written.
- No training, holdout, or quarantine record files were produced.
- The private key and password were absent from the retrieval environment.
- The holdout was not unblinded.
- Orbit fitting was not performed.

The generated run manifest recorded:

```text
status = REAL_RETRIEVAL_BLOCKED
retrieval_exit_code = 1
verification_exit_code = 127
source_requests = []
curated_files = []
training_count = null
holdout_count = null
quarantine_count = null
```

## Integrity

The blocked artifact's internal `DELIVERABLE_SHA256SUMS.txt` verified every contained file. The GitHub Actions artifact ZIP SHA-256 is:

```text
3e72b5a0dc977bd32d62753f23777770e88617407ae1ca6441149187697bf891
```

## Scientific interpretation

This run is preserved as a blocked execution. It is not a failed physical prediction, not a failed source-schema test, and not evidence about JX orbital accuracy. Because no network request occurred, the real curation stage has not begun.

The original single-attempt authorization is treated as consumed administratively. A second execution requires explicit human authorization and a prospectively frozen plumbing-only correction. The correction must not change the source-adapter code, endpoints, cutoff, holdout window, data model, or scientific rules. Because the original private custody key was not archived before the temporary execution environment reset, a replacement keypair would also have to be generated, frozen, and backed up before any second retrieval; no real holdout was encrypted under the lost key.
