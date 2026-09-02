# JX Reality Test 1 — Real DA14 Retrieval Attempt 2

**Date:** 2026-09-02 UTC  
**Classification:** `REAL-SOURCE CURATION ATTEMPT / NO PHYSICAL RESULT`  
**Status:** `SOURCE_SCHEMA_UNSUPPORTED`  
**Workflow run:** `33689123745`  
**Workflow commit:** `7262a0ea0c22fc8b9f6a4d92cce0f80630f613b0`  
**Frozen input commit:** `9a30ff44861cfa711315aad82bb546e4ffab23f8`

The prospectively authorized second attempt executed once. The replacement RSA-3072 keypair was synthetically verified, archived in persistent Library custody, read back, and verified before execution. The permission-plumbing correction worked and the unprivileged adapter process launched.

The live adapter then stopped fail-closed with exit code `22` and status:

```text
SOURCE_SCHEMA_UNSUPPORTED
```

At least one live official-source response did not match the exact frozen schema accepted by the audited normalizer. The adapter did not retain raw responses or print protected values, so this artifact does not identify the failing source or field.

No curated training file, encrypted holdout, or encrypted quarantine was produced. The private key and password were absent from the retrieval environment, the holdout was not unblinded, and no orbit fitting occurred. Because no source-response receipt was created, the number of completed requests is not established; real responses may have existed transiently in process memory but were not persisted.

The artifact's internal SHA-256 manifest verified. GitHub Actions artifact ID: `9869307779`; ZIP SHA-256:

```text
6e75c33d55e2f6fe6901e209401ef3cd1378ac6e842a05df887be438c85c2854
```

Replacement public-key fingerprint:

```text
d995cd53e4bb9769d0112b868def61191d7b74ce7544f4e5141932d78b8803ee
```

This is not an orbit-fit failure and not a physical-prediction failure. The authorized second attempt is consumed. Any future live retrieval requires new explicit authorization after a prospective live-schema diagnostic and a newly audited adapter revision. The holdout remains unblinded.
