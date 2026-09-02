# JX Reality Test 1 — value-redacted live schema diagnostic

This gate diagnoses the live MPC/JPL response structure that caused DA14 real-retrieval Attempt 2 to stop with `SOURCE_SCHEMA_UNSUPPORTED`.

It does **not** retrieve a usable observation dataset. The diagnostic keeps each live response only in process memory, emits structural metadata and hashes, and writes no observation values, sample records, raw HTTP body, or complete normalized plaintext.

The workflow first runs the loopback synthetic audit embedded in `schema_probe.py`. The live query is allowed only if every audit check passes. The probe then uses the exact previously audited request policy and HTTP client, queries the three frozen official endpoints sequentially, and tests the existing normalizers in memory.

The live artifact may contain field names, JSON types, record/array counts, string-pattern classes, response hashes, and a sanitized compatibility error. It must not contain right ascension, declination, radar measurements, uncertainties, or any unblinded holdout values.

A successful diagnostic authorizes only a prospective normalizer revision and a new synthetic audit. It does not authorize another complete real retrieval.
