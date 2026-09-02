# JX Reality Test 1 — Schema-Probe Preflight Hash Corrections

**Date:** 2026-09-02 UTC  
**Blocked workflow runs:** `33691866126`, `33692319546`  
**Independent hash-inspection run:** `33692457730`  
**Status of both probe runs:** blocked during frozen-input verification, before the synthetic audit and before any live network request

The initial hash manifest contained the SHA-256 of a local working copy rather than the exact GitHub-checked-out bytes. A first prospective estimate corrected a visible nonfunctional comment difference but still did not equal the checked-out file. No diagnostic code or contract was changed.

A separate hash-only GitHub Actions workflow then measured the exact checked-out files without running the schema probe or making a live request. It reported:

```text
schema_probe.py
f18957c23c2c2a696d13d67dc19de6aad0bf672bf93a4df28e87204bbdec9f68

schema-probe contract
11cc3252316de74f6e55a7bb9f8f7035c0f1ee33a028c4f75f0fa552c13d9870

README
f22323d19d5673c2be7694887d353a468e0eb43e9d42bc09daa642a808b23f69
```

The manifest is now frozen to those runner-measured hashes. No target request, source response, observation value, synthetic audit result, normalizer result, orbit fit, or unblinding occurred in either blocked probe workflow. The contract, diagnostic code, permitted outputs, prohibited outputs, endpoints, and scientific rules remain unchanged.

A new one-shot workflow is required because the earlier workflow files were intentionally single-use. The original authorization remains limited to the schema-only diagnostic; it does not authorize a complete real-data retrieval.
