# CUDA four-arm mechanism screen evidence

This directory preserves two executions of the bounded 1,000-year qualification.

- `qualification/` is the authoritative execution under runner commit `6bc31dbbdc2238f1a672942151f0f7e8dfc1dce6`. It binds one exact runtime identity across all arms and binds/reloads the complete checkpoint and segment evidence roster. Its internal `SHA256SUMS` verifies every retained artifact.
- `qualification_superseded_fd25033/` is the earlier execution under runner commit `fd250331777b80e70ad800be02f1aecbd5f13c63`. It is retained as historical/negative process evidence because its runner did not yet fail closed on cross-runtime arm reuse or require raw checkpoint/segment artifacts during cached-arm acceptance. Its own checksums verify, and its four endpoint NPZ files are byte-identical to the authoritative execution, but it is not the accepted qualification package.

The authoritative result is `MODEL_OUTPUT / SCREENING_ONLY`. Its 1,000-year execution and integrity gates passed, while its separate predeclared one-million-year route-capacity authorization failed. No one-million-year or 4.5-billion-year run was launched from this result.
