# TNO Stage 1.5 V7 two-lane continuation smoke

This directory retains the post-preregistration outcome of the bounded V7
two-lane continuation smoke. It is setup evidence only, not a numerical
qualification, dynamical classification, Planet Nine result, or scientific
result.

- Preregistered commit: `7861eda581d093da83959412885864b207de806c`
- Annotated tag: `tno-stage15-resolved11-encounter-v7-preregistered-20261010`
- Receipt file SHA-256: `7386f3470c75dfdfe3a035206c9f7079eaef69ab3e7e6fb0ecb72ec9efaeeff7`
- Receipt content SHA-256: `ee66ee4e35d6ae5f983054948ddaab45ee6c5feddcaf59e7911921fd550fbcd4`
- Receipt size: `1,877,869` bytes

The smoke used the real DE441 batch-0 nominal orbit and eight fixed holdouts.
It executed one year at h64 and one year at h128 after independently deriving
the source-specific acceleration-bound certificate from each precomputed
source path. Both lanes completed with zero routes, zero removals, nine active
tracers, bitwise-equal public/prepass source endpoints, and successful next
continuation validation. No second year was executed and the qualification
output root remained protocol-only.
