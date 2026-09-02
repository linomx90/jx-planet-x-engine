# JX Reality Test 1 — Source Adapter v0.2 Revision Audit

**Date:** 2026-09-02 UTC  
**Classification:** `PROSPECTIVE SYNTHETIC NORMALIZER REVISION`  
**Status:** `SOURCE_ADAPTER_V2_AUDIT_PASSED`  
**Workflow run:** `33694352872`  
**Execution commit:** `1f7e1e05a7aebef6b357571993a70941f13c7287`  
**Artifact ID:** `9871211121`

## Authorized correction

The exact previously audited v0.1 runtime was reconstructed and all original source hashes were verified. One file changed:

```text
src/jx_source_adapter/normalize.py
before: 32e7d20ab7ef8e488bba9299c78a1eabcb47146c28d88a580a99b4cd0d35e2fd
after:  b9b9d49e576786f845779fff8ca33a56ddc424f1a17c6fac0a912819abf0d236
```

The revised parser accepts only:

1. an outer array containing exactly one object envelope; or
2. an outer array containing exactly one object envelope and exactly one non-boolean integer, in either order.

The integer is opaque: its value is not interpreted, retained, emitted, or allowed to alter normalized records. All unregistered layouts still fail closed.

## Executed audit

Two complete synthetic no-network executions produced the identical result SHA-256:

```text
0f9daa39657514d51deb7b79108635079d332df37e54df6c8dea7e09388a0ea5
```

Each execution passed **21/21** checks. Those checks covered the unchanged production endpoint allowlist, blocked arbitrary hosts, legacy-envelope compatibility, the newly registered live envelope, invalid-layout rejection, full encrypted curation, output verification, protected-value non-leakage, private-key absence, successful custodian recovery, and invariance of training/schedule outputs when the opaque integer changed.

The inherited v0.1 security baseline remains the previously recorded **29/29** pass with result hash:

```text
794fafe5b33b0091e6846ca263a772baa90ff4191009b892652d81b7236d2821
```

The audit used no external network, made no real DA14 query, did not unblind a holdout, and performed no orbit fit.

## Result

```text
SOURCE_ADAPTER_V2_AUDIT_PASSED
checks: 21/21
repeat executions: 2
external network: no
real target query: no
complete real retrieval authorized: no
```

This establishes only that the narrow parser correction passed the declared synthetic gates. It does not itself prove that the next real retrieval will complete, nor does it produce scientific evidence about the asteroid.

## Provenance

```text
workflow artifact SHA-256:
bba96cef39d6a54c2678cd6c9984c6683bd536b86377daf893729d03c5687ff1

v0.2 result SHA-256:
0f9daa39657514d51deb7b79108635079d332df37e54df6c8dea7e09388a0ea5

workflow run manifest SHA-256:
e9f9a861d8aa0d0d55f42a704be0a40e7bd5b3459731f461d75e2f0d41a3f6e7
```

A complete real retrieval remains blocked until a new explicit human authorization is recorded.
