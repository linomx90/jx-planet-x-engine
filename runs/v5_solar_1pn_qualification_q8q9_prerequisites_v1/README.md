# JX V5 Solar 1PN Q8/Q9 prerequisite design

This additive package defines the fail-closed records required before the
frozen Solar 1PN qualification may produce holdout outcomes. It is bound to
qualification
`jx.v5.solar_1pn.qualification.78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781`,
package digest
`80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237`,
and predecessor commit `975b1b7002e358a980457acb18c9beaa90aa1c9f`.

The package state is `DESIGN_ONLY_BLOCKED`. It is not scientific evidence,
does not authorize execution, contains no trajectory code or outcome, and
records no unblinding. It also contains no named custody, independence,
expectation, or unblinding instance. `external_artifact_slots_v1.json` contains
only null slots showing which independently supplied pre-execution artifacts
remain absent. The post-execution unblinding slot is explicitly
`NOT_APPLICABLE_PREEXECUTION`.

## Required sequence

The custody sequence is strictly:

1. `DESIGN_ONLY_BLOCKED`
2. `AWAITING_EXTERNAL_CUSTODIAN`
3. `CUSTODY_ACCEPTED_EXPECTATIONS_NOT_COMMITTED`
4. `EXPECTATIONS_COMMITTED_SEALED`
5. `EXECUTION_PACKAGE_LOCKED_SEALED`
6. `RUNNING_EXPECTATIONS_SEALED`
7. `OUTPUTS_COMMITTED_EXPECTATIONS_SEALED`
8. `UNBLIND_REQUESTED`
9. `UNBLINDED_FIRST_VALID`
10. `ADJUDICATED_RETAINED`

An illegal transition, a repeated unblinding, mutation of committed outputs,
or post-outcome budget change produces `INVALIDATED`. The unblinding artifact
belongs only to `AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION`; it cannot be filled
as a pre-execution placeholder.

Before execution, independent signed custody and independence attestations,
the sealed-expectation commitment, and independently reviewed frozen budgets
must exist. After execution but while expectations remain sealed, the exact
output manifest must be committed. Only then may a first-valid unblinding be
requested.

## Commitment rule

The external custodian must use a secret, independently generated 256-bit
nonce. The commitment is exactly:

`SHA256(domain_separator_utf8 || nonce_bytes || canonical_expectation_bytes)`

where the domain separator is
`JX-V5-SOLAR-1PN-Q8Q9-SEALED-EXPECTATIONS-V1|`. The nonce and canonical
expectation bytes remain unavailable to the implementation team before
unblinding. An unsalted commitment, noncanonical reveal, or reveal mismatch is
invalid.

## Error budgets

`error_budget_template_v1.json` is deliberately unresolved. For each of the
eight frozen observables it leaves the norm, scope, discrimination margin, all
nine component allocations, their derivations and evidence, the exact total,
and external scientific review empty.

The only permitted default combination is the exact Decimal conservative sum

`oracle + step + solve + decimal + event + transform + parameter + analytic + model`.

Every component and the total must use the observable unit and frozen norm.
Each realized component must remain inside its own allocation in addition to
the total discrepancy remaining inside the total allocation. Zero is accepted
only with an `EXACT_ZERO_PROVED` derivation, retained evidence, and an explicit
synthetic same-equation scope proof.

## Holdout coverage

External custody and sealed expectations must cover exactly the frozen
scientific roles: generic three-dimensional, eccentric, long-arc,
inside-domain, domain-boundary, outside-domain, transform-twin, signal-scale,
and EIH-limit cases. Holdout scientific fingerprints must be fresh relative to
all disclosed development fingerprints.

## Verification boundary

`verify_prerequisites_v1.py` is read-only and has no command-line entry point.
It is self-contained inside this JX package and has no production-module or
`PYTHONPATH` dependency. Its schema checks implement only the explicitly
enumerated closed subset used by the six bound schemas; they do not claim to
be a general Draft 2020-12 implementation. Unsupported schema keywords and
weakened/open object schemas fail closed, and the six canonical schema
definitions are pinned by the verifier as well as the registration. Every
public external-artifact validation checks the schema path, size, raw digest,
and canonical digest before interpreting the artifact. Error-budget semantics
remain private to those pinned validation paths. It rejects duplicate JSON
keys, binary JSON floats, nonfinite constants, unknown schema fields, path
escapes,
symlinks, hash or size mismatches, fabricated external fields, missing roles,
invalid lifecycle transitions, premature unblinding, incorrect Decimal sums,
component overflow hidden by a larger total, and early claims. It never
imports or invokes a dynamics or trajectory function.

This design package does not resolve Q8 or Q9 and cannot support any scientific
or registry claim.
