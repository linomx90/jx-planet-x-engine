# JX Reality Test 1 — Real DA14 Retrieval Attempt 3 (Source Adapter v0.2)

This directory freezes one authorized production curation attempt for
367943 Duende (2012 DA14).

The execution reconstructs the exact previously audited v0.1 runtime, verifies
all inherited source hashes, applies only the prospectively audited v0.2 MPC
envelope patch, verifies the patched normalizer hash, and runs a 21/21
synthetic no-network audit before any production request.

The production process receives only the public RSA key. It may export
pre-cutoff training data, a redacted holdout schedule, encrypted holdout and
quarantine files, observer metadata, safe source receipts, manifests, and
checksums. It may not receive the private key or password, unblind data, fit an
orbit, generate predictions, persist raw HTTP responses, or write the complete
normalized plaintext dataset.

Any outcome consumes this authorization. A later production retrieval requires
new explicit human authorization.
