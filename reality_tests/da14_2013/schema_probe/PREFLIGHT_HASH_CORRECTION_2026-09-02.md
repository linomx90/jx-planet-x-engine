# JX Reality Test 1 — Schema-Probe Preflight Hash Correction

**Date:** 2026-09-02 UTC  
**Prior workflow run:** `33691866126`  
**Prior execution commit:** `90e75bc9691dbbff98d54e768e74f92ef8c601a9`  
**Prior status:** blocked during frozen-input verification

The first schema-probe workflow stopped before the synthetic audit and before any live network request because `SCHEMA_PROBE_INPUT_SHA256SUMS.txt` contained the hash of a local working copy whose `do_GET` line included the nonfunctional comment `# noqa: N802`. The committed source omitted that comment. The committed source itself was not changed.

Prospective correction:

```text
schema_probe.py expected SHA-256
old: ab3df042d699e97ac284187e6e7bec516ecd5a2297fa509a0b40fbb724d88616
new: d2d5d501a967ef96e32cea46de0b6155720d31d86181694b15af2eccb7047655
```

No target request, source response, observation value, synthetic audit result, normalizer result, orbit fit, or unblinding occurred in the blocked run. The contract, diagnostic code, permitted outputs, prohibited outputs, endpoints, and scientific rules remain unchanged.

A second workflow file is required because the original workflow was explicitly one-shot. This correction authorizes only execution of the already-authorized schema-only diagnostic; it does not authorize a complete real-data retrieval.
