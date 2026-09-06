# V5 Solar 1PN execution prerequisites (design only)

**State:** `DESIGN_ONLY_BLOCKED`

This additive package records the still-missing execution prerequisites for
qualification gates Q2, Q3, and Q4. It does not modify or supersede any byte
in `runs/v5_solar_1pn_qualification/`. It contains no dynamics, runner, CLI,
trajectory, event, oracle result, EIH result, qualification outcome, budget,
sealed expectation, independence attestation, or execution authorization.

The three subordinate registrations distinguish protocol constants already
frozen in V1 from values that must be supplied and reviewed externally. Every
external value remains `null` with an `AWAITING_EXTERNAL_*` status. Filling any
field requires a new, separately reviewed registration; this design package
must not be edited in place and then represented as the frozen qualification.
In particular, the Q3 schedule references do not supply a per-checkpoint epoch
roster. The Q5 checkpoint grid is not silently reused for Q3; the Q3 roster and
all per-checkpoint scales, floors, component rules, and norms remain null.

The package is bound to predecessor qualification
`jx.v5.solar_1pn.qualification.78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781`,
package SHA-256
`80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237`,
and freeze commit `975b1b7002e358a980457acb18c9beaa90aa1c9f`.

`verify_execution_prerequisites_v1.py` is read-only. It rejects duplicate JSON
keys, binary-float JSON numbers, nonfinite constants, unknown keys, malformed
or escaping paths, symlinks, digest/size drift, predecessor drift, non-null
external fields, and any claim that the package is ready or authorizing.
Its self-contained schema checker implements and audits only the exact keyword
subset used by the four frozen schemas, including `maxItems`; any unrecognized
schema keyword fails closed. It is not represented as a general-purpose JSON
Schema implementation and imports no mutable project validator.

All content remains `MODEL_OUTPUT`-class design metadata. It makes no scientific
claim and grants no registry, propagation, holdout, or unblinding authority.
