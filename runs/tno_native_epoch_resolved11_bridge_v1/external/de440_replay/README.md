# Eleven-body fresh-replay kit

This directory contains the exact scientific inputs and clean JX source
snapshot needed by the orbital validation ladder v3 to rerun the accepted
one-day eleven-body benchmark.

Included inputs:

- DE440s planetary ephemeris kernel;
- DE440 GM parameter kernel;
- the retained CSPICE native library and provenance records;
- the NumPy and SpiceyPy wheels used by the frozen input contract;
- the JX source snapshot at commit
  `750206382bd6bf323bd72b3b779cd61ddc79b3a0`.

The top-level `run_validation.py` checks every input and source byte before
starting the replay. It then accepts a fresh result only when all state, step,
work, claim, and numerical-KAT checks pass and the runtime-independent
scientific fingerprint matches the preserved reference.

The replay still requires a compatible local interpreter with NumPy 2.3.5 and
PyERFA 2.0.1.5 importable. The included native artifacts target Linux x86-64
and CPython 3.14. A different Python executable build may legitimately change
the raw report's runtime section and semantic checksum while leaving the
scientific fingerprint exact; v3 reports both facts separately.

This kit does not include or test the lunar mantle or fluid core.
