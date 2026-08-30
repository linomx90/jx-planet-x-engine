# V5 Solar 1PN isolated oracle candidate

This directory contains a separately transcribed, pure-standard-library
`Decimal` implementation candidate for independent review. It is outside the
`jxplanetx` package and imports no JX force, integrator, event, test, registry,
or helper code. That source isolation does **not** establish operational
independence: the candidate remains
`CANDIDATE_PENDING_INDEPENDENT_REVIEW`, produces only `MODEL_OUTPUT`, and is
never registry-authorized.

The method is Fehlberg RK7(8), transcribed from NASA TR R-287 Table X (printed
page 65; PDF page 72). Equation (105), printed page 52/PDF page 59, identifies
the hatted formula as eighth order. Accordingly, the accepted weights use
zero-based stages 11 and 12 while the embedded seventh-order weights use
stages 0 and 10. Equation (134) gives the retained defect orientation
`embedded_seventh - accepted_eighth`. The controller uses that order-seven
defect with exponent `1/8`, an immutable per-component absolute-tolerance
roster plus scalar relative tolerance, and clips every step to each requested
checkpoint; there is no interpolation or dense output.

Every integration result retains the complete immutable controller record and
its canonical-JSON SHA-256, so different tolerances or step bounds cannot share
only a generic controller label. Generic results retain their exact initial
epoch/vector; restricted-model results additionally retain the complete
physical contract and typed initial state.

Pinned source identity:

- SHA-256: `5553a2a3eb53785a461762cc2b29428015f1b32c3ad0a5cb57f85a421256a0c8`
- Size: `2625098` bytes

No CLI, file writer, frozen holdout loader, event observer, qualification
runner, sealed expectation, or unblinding operation is provided here. The
candidate cannot remove the frozen Q3 independence/review blocker by itself.
