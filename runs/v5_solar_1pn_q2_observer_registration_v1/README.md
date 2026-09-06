# V5 Solar 1PN Q2 observer registration boundary (design only)

**State:** `DESIGN_ONLY_BLOCKED`

This additive JX package is a handoff boundary for external registration of
the nonauthorizing Q2 Decimal perihelion observer frozen by commit
`d6344c49e7c6a734a337e84bd209ca5c11c23916`. It binds that commit's exact
observer, package, README, and disclosed-test bytes. It does not modify or
supersede the frozen qualification, execution-prerequisite, Q8/Q9, observer,
oracle, or V4 artifacts.

`external_registration_template_v1.json` is an empty template. An external
reviewer must copy it to a separately named, separately reviewed artifact;
the template and this package registration must never be filled in place. A
completed copy must bind one exact CPython runtime artifact, an exact
stdlib/dependency manifest (with no third-party distributions), and exactly
two complete observer configurations, one for each frozen root tolerance.
The runtime capsule also binds its locale plus exact invocation and environment
manifest digests.
`configuration_id` and `maximum_relative_off_plane` are intentionally not
invented here and must be selected and signed externally.

The Q2-specific external document also records reviewer identity,
organization, relationship to the implementation team, conflicts of
interest, review assertions, time, signing key, signature profile, signed
payload digest, and signature. The verifier requires a caller-supplied
signature-verification callback. This package does not select or trust a
signature profile or keyring, so even a structurally valid signed record
remains nonauthorizing.

## Bound and unbound scope

The frozen candidate implements only the observation layer:

- the initial-event outward-then-inward arming state machine;
- cubic-Hermite dense state and exact degree-five Bernstein root isolation;
- negative-to-positive `r dot v` bracketing, refinement, and termination;
- initial-position/angular-momentum orientation with an off-plane check;
- Decimal `atan2` and pi, exact-pi-tie-rejecting unwrap;
- orbit indices exactly 1 and 2; and
- Decimal ordinary-least-squares slope and signed `1PN - Newtonian` pairing.

It does not implement step extrapolation, event-tolerance extrapolation,
inverse-`c^2` extrapolation, the analytic `6*pi` comparison, error budgets,
or gate adjudication. Those items remain explicit blockers, as do wrapper
authentication/serialization, expectation-content schema, Q3, Q4, holdout
custody, and sealed expectations. No holdout or official trajectory may be
run from this package.

The pinned Q8/Q9 custody, sealed-expectation, and error-budget schemas are
referenced only as downstream interfaces. They bind the older qualification
identity and do not identify this concrete observer. They therefore cannot
satisfy Q2 registration and are not incorporated as evidence here.

There are two qualification-level blockers that this package cannot repair:

1. the frozen qualification identity predates and omits the concrete Q2
   source/runtime/configuration tuple, while its Q8/Q9 records could admit
   multiple observers; and
2. the frozen qualification requires `q8.unblinding_record` before execution,
   while the later Q8/Q9 lifecycle correctly places unblinding after sealed
   output commitment.

Frozen bytes are not patched and no overlay is claimed. Official execution
requires a separately reviewed, content-derived qualification successor with
a new identity that binds the completed Q2 registration and resolves the Q8
phase contradiction.

`verify_registration_v1.py` is read-only, standard-library-only, and imports
no JX, observer, oracle, dynamics, trajectory, or registry module. It rejects
duplicate JSON keys, binary-float JSON numbers, nonfinite constants, unknown
schema keywords, open object schemas, unsafe paths, symlinks, byte or
canonical-digest drift, partial external slots, configuration drift, missing
conflict disclosures, unsigned records, and any authority or outcome claim.

All package-authored content is `MODEL_OUTPUT`. This boundary produces no
scientific result and grants no execution, registry, qualification,
unblinding, or claim authority.
