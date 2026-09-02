# JX Reality Test 1 — Value-Redacted Live Schema Diagnostic

**Date:** 2026-09-02 UTC  
**Classification:** `LIVE API STRUCTURE DIAGNOSTIC / NO OBSERVATION DATASET`  
**Status:** `LIVE_SCHEMA_PROBE_COMPLETE`  
**Workflow run:** `33692648520`  
**Execution commit:** `98fb738505c3594156ce84b13966f9654fd85db2`  
**Artifact ID:** `9870611284`

## Privacy and execution gates

The embedded synthetic no-value-leak audit passed **19/19** checks before the live diagnostic. The live process emitted no measurement values, retained no raw HTTP response, wrote no complete normalized plaintext dataset, contained no private key, did not unblind the holdout, and did not perform orbit fitting.

## Live-source result

All three approved official endpoints returned HTTP 200 JSON responses and were held in memory only.

| Source | Structural result | Existing v0.1 normalizer |
|---|---|---|
| MPC observations | Array with one object envelope plus one integer | **Incompatible** |
| MPC observatory codes | Object map with 2,718 station records | Compatible |
| JPL radar | API signature 1.1; 8 records; 3 station records | Compatible |

The MPC observations response had:

```text
root type: array
envelope items: 2
envelope item types: 1 object + 1 integer
object-envelope keys: ADES_DF, OBS80, OBS_DF, XML
ADES_DF rows: 1,071 objects
measurement-key signatures: 1,071 optical
```

The v0.1 normalizer rejected this response only because it required the outer array length to equal exactly one. The diagnostic did not expose or preserve the integer's value or any optical/radar measurement value.

## Diagnosis

```text
MPC observations: INCOMPATIBLE
reason: overly strict outer-envelope cardinality check
MPC observatory codes: COMPATIBLE
JPL radar: COMPATIBLE
```

This isolates the Attempt-2 `SOURCE_SCHEMA_UNSUPPORTED` result to one narrow MPC transport-envelope condition. It does not establish that a complete retrieval will succeed after correction, and it is not an orbit-fit or physical-prediction result.

## Provenance

```text
live probe SHA-256:
e4ccf666acef8a0728cd8744b83603ffd778887bfebae2d40611c991edfd60c3

workflow artifact SHA-256:
bec68ea95b9af71dfeae168bafa0940d58c2c221dac85d11ac83ddfd511ef798

first preflight-blocked artifact SHA-256:
1b4ccf435f64d1e0e017dd51a7bbd5397d67bcf361d3c47dd476a4e4ff445229

second preflight-blocked artifact SHA-256:
0fd6cca5a6948bbf1971b72e8dab651253a3537c90442c8c2fd926d5a703edd1
```

The two earlier schema-probe workflows are preserved as blocked preflight executions. Neither reached the synthetic audit nor made a live request.
