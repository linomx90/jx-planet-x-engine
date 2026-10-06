# DE441 anchor: historical screening record only

This directory is retained for provenance, but it is **not a replayable
ephemeris qualification package**.

The checkout contains four compact JSON records. It does not contain the raw
DE441 source archive, normalized epoch-state table, Y6-160/Y6-224 trajectory
files and summaries, binary trajectory, or independent Bulirsch–Stoer
trajectory whose digests are named by those records. No retained file in this
checkout matches those missing trajectory digests.

`metadata.json` and `gate.json` record that the independent reference was
missing. The later `independent_reference_gate.json` records a numerical gate
result, but the underlying arrays are absent and therefore cannot be audited.
Current JX validators intentionally treat this old schema as
`NO_CROSS_METHOD_CLAIM`.

Allowed use: historical engineering context for a five-massive-body plus
15-synthetic-tracer screen.

Not allowed: scientific validation, DE441 reproduction, cross-method
agreement, precision authorization, or long-horizon authorization.

Use [`../de441_horizons_10yr`](../de441_horizons_10yr) for the retained,
hash-verified DE441/Horizons ten-year comparison.
