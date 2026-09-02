# JX Reality Test 1 — Source Adapter v0.2 prospective revision

The value-redacted live schema diagnostic completed successfully and isolated the Attempt-2 failure to one narrow condition: the MPC observations endpoint returned an outer JSON array containing one requested-object envelope plus one ancillary integer, while the frozen v0.1 normalizer required the array length to equal exactly one.

This package does not contain observation values or a complete dataset. It reconstructs the exact previously audited v0.1 runtime, verifies all original source hashes, and applies one deterministic patch to `src/jx_source_adapter/normalize.py`.

The v0.2 parser accepts only:

1. one object envelope; or
2. one object envelope plus one non-boolean integer, in either order.

The integer is treated as opaque transport metadata. Its value is not interpreted, retained, emitted, or allowed to affect normalized records. All other new layouts fail closed.

The new synthetic audit uses a loopback mock server and verifies the old endpoint allowlist, fail-closed layouts, full training/holdout/quarantine curation, output integrity, encryption recovery, absence of protected values from the pre-unblind workspace, and invariance of normalized outputs when the ancillary integer changes.

No external network request and no DA14 retrieval are authorized by this revision. A later complete real retrieval requires separate explicit human authorization after the v0.2 audit passes and is preserved.
