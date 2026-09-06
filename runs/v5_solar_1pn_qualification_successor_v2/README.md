# V5 Solar 1PN content-derived qualification successor specification

**State:** `SPECIFICATION_ONLY_BLOCKED`

This additive JX package introduces a content-derived successor identity,
repairs one chronology contradiction, and records two execution-eligibility
and claim-boundary corrections without modifying any frozen byte. It binds the
frozen V5 qualification, the execution and
Q8/Q9 prerequisite packages, and the exact Q2 observer-registration boundary
at commit `3102c53616e3b7134092191ca94c77077128ae64`. The boundary in turn binds
the observer at commit `d6344c49e7c6a734a337e84bd209ca5c11c23916`.

The package mints a content-derived `successor_spec_id`. That identifier names
only this blocked specification. It is deliberately not a runnable or final
`qualification_id`: `execution_qualification_id` remains null until a new,
content-derived execution envelope binds every required external artifact.

## Exact chronology correction

The frozen inputs say both that `blocker.q8.unblinding_record` is required
before execution and that the record is created only after execution. This
successor replaces that contradictory pre-execution flag. This is the sole
chronology replacement. Two separately enumerated eligibility/claim-boundary
corrections also prevent publicly disclosed cases and a hash-leaking legacy
commitment schema from being treated as execution-eligible; they do not edit
their frozen source bytes. The effective lifecycle is:

1. custody, independent review, sealed expectations, error budgets, and all
   implementation registrations are complete before execution;
2. outputs are committed while expectations remain sealed; and
3. the first-valid unblinding record is created in
   `AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION`.

The Q8 gate remains an after-unblinding adjudication gate. The correction does
not waive custody, sealing, output commitment, first-valid-only, or no-tuning
requirements.

## Identity and attachment rule

`qualification_successor_v2.json` is the sole identity manifest. Its ID is
computed by strict-parsing the document with duplicate keys, every JSON number,
and nonfinite constants rejected; replacing only `successor_spec_id` with
`CONTENT_DIGEST_SENTINEL`; and constructing exactly this three-key envelope:
`{"schema": <declared domain>, "sentinel": "CONTENT_DIGEST_SENTINEL",
"successor": <sentinelized manifest>}`. The envelope is encoded as UTF-8 JSON
with keys sorted, `ensure_ascii=false`, `allow_nan=false`, and separators
`(',', ':')`, then hashed with SHA-256.

The manifest contains a closed, ordered roster of null external-artifact
slots. A slot must never be filled in this file. Resolving a pre-execution slot
requires a separately named artifact with its own content-derived ID and a new
successor specification that binds it. Once all pre-execution inputs are
resolved, a new content-derived execution envelope must mint a distinct
`execution_qualification_id`.

That future execution identity may not float free of this specification. Its
identity preimage must be exactly the declared three-key envelope containing
the then-current `successor_spec_id` and the ordered, complete 14-role
pre-execution roster. Every roster entry must bind its role, content-derived
artifact ID, path, size, raw SHA-256, and canonical SHA-256. Omission,
substitution, or reordering creates a different or invalid execution identity.
Each pre-execution artifact ID is derived independently from this successor
specification ID and must not depend on the not-yet-minted execution ID. The
single `execution_qualification_id` is derived last from the successor ID and
the complete ordered roster; there is no second envelope identifier.

The static template label
`jx.v5.solar_1pn.q2_observer.external_registration.v1` is not a content
identity and is insufficient for execution. Any completed external Q2 record
must derive its own ID from the exact observer source, runtime artifact,
dependency closure, configurations, reviewer statement, signature profile,
key, signed payload, and signature.

Post-execution output, unblinding, and final-adjudication records instead mint
content-derived child IDs that bind the already-stable execution identity.
They must not mutate or recompute either the successor specification ID or the
execution ID, and they cannot be attached to this blocked specification as
evidence of completion. Their slots remain null forever in this immutable
specification; fulfillment exists only in separately content-derived child
records.

The frozen `fixture.holdout.*` records remain byte-for-byte unchanged, but
their public disclosure before the observer existed makes them ineligible as
input-blind holdouts for a new execution. They may be treated as validation
cases. A future protocol must either use fresh externally custodied cases or
state an explicitly narrower outcome-blinding-only claim. Likewise, the
legacy sealed-expectation schema remains frozen but is not execution-eligible:
it exposes unsalted raw and canonical expectation hashes before reveal. A new
schema must expose only a domain-separated salted commitment plus schema and
roster bindings until unblinding.

A future case artifact must select exactly one branch. `FRESH_INPUT_BLIND`
requires an exact secret roster locked before execution by a domain-separated
hiding commitment, authenticated-encrypted package, or equivalent custodian-run
profile; no unsalted plaintext case hash may be public, injection occurs only
after executor/configuration lock, and outputs bind the commitment or
ciphertext. `DISCLOSED_VALIDATION_OUTCOME_BLIND_ONLY` binds the exact public
validation roster and permanently narrows the claim. The branch code is part
of the future execution identity and final adjudication.

## Nonauthorization

The exact Q2 boundary package digest is
`e4abdcf4b83b9a01b8b060ea8095b229d3ee8191eb73d2ca93ef9e4fbd57ba55`.
It still has no external runtime, dependency manifest, configuration,
independence signature, trusted signature profile/keyring, expectation-content
schema, authenticated wrapper format, Q2 extrapolation/analytic adjudication,
reviewed error budget, resolved Q3 oracle, resolved Q4 EIH evaluator, custody,
or sealed expectation.

Accordingly this package is `MODEL_OUTPUT`, produces no outcome, and grants no
execution, registry, qualification, unblinding, adjudication, or scientific
claim authority. No holdout or official trajectory may be run from it.

`verify_qualification_successor_v2.py` is read-only and standard-library-only.
It imports no JX, observer, oracle, dynamics, registry, trajectory, or holdout
module. It verifies the content-derived identity; exact predecessor and Q2
bindings; the old contradictory source field; the one allowed chronology
replacement; null slots; false authority flags; schema closure; safe paths;
and the exact package roster.

Verification must be cache-free. Preflight and quarantine any `__pycache__`
directory, then load the verifier's exact source bytes with
`compile(path.read_bytes(), str(path), "exec")` and `exec` them into a fresh
namespace before calling `verify_package`. `python -B` or
`PYTHONDONTWRITEBYTECODE=1` must also be used for any surrounding interpreter,
but those switches alone do not neutralize a pre-existing bytecode cache. The
exact recursive package rosters deliberately reject every cache, extra result,
extra directory, symlink, and multiply linked regular file.

The verifier deliberately does not invoke Git. It pins commit/tree strings and
recomputes the complete content-hash closures, but the statement that the
frozen qualification commit precedes the observer is independently audited
repository-lineage evidence, not a fact proved by this metadata verifier.
