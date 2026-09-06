# V5 Solar 1PN external pre-execution handoff request

**State:** `HANDOFF_REQUEST_ONLY_BLOCKED`

This additive JX package begins the external handoff required by successor
specification
`jx.v5.solar_1pn.qualification_successor_spec.f402f13c924543633925a99734d800911b01a8812ca7b293f7426fea6e07030c`.
It binds and later verifies the entire successor five-file content closure,
not merely a manifest literal. It resolves no successor slot, contains no external response, chooses no case
or claim branch, and grants no execution, registry, qualification,
unblinding, adjudication, or scientific-claim authority.

## Minimal scope

`registration_v1.json` is the sole request identity manifest. It binds the
entire five-file successor content closure—the manifest plus its four locked
files—and its ordered fourteen-role pre-execution roster. Each request copies
the source role, phase, and requirements exactly and binds the domain-separated
canonical digest of the exact `{schema, successor_spec_id, slot}` envelope.
All selection, evidence, response identity, path, size,
digest, signer, key, signature, and trust acceptance fields remain null.

The request is immutable. An external response is always a separately named,
content-derived artifact that binds this `handoff_request_id`, the frozen
successor ID, one exact role, the source-slot digest, and its own exact public
payload and signature metadata. Fresh-case and sealed-expectation roles expose
only a hiding commitment, authenticated-encrypted artifact, or custodian-safe
record—never plaintext secret material or its unsalted raw/canonical digest. A
response is never written into this manifest.
Accepting any response requires a new content-derived successor specification;
the execution qualification ID is minted only after the complete ordered
pre-execution roster is bound.

## Trust bootstrap

The `signature_trust_profile_and_keyring` response cannot authorize itself.
Its admission requires a separate sponsor/root-trust acceptance record whose
identity, path, hashes, accepting authority, signed payload, key material, and
signature are all absent and null here. That bootstrap record must bind this
request and the proposed trust-profile response, prohibit self-signing and
self-authorization, and be accepted through an independently pinned
out-of-band root-trust policy. Acceptance cannot verify under a key whose sole
provenance is the proposed keyring. Only after that acceptance may other response
signatures chain to the admitted trust profile and keyring.

## Secrecy and branches

The scientific sponsor must choose exactly one future branch, but this request
does not choose it:

- `FRESH_INPUT_BLIND` requires externally custodied secret cases. Public
  metadata may contain only a domain-separated hiding commitment,
  authenticated ciphertext, or equivalent custodian-safe reference. It may
  never contain plaintext case or expectation IDs or values, or unsalted raw
  or canonical hashes of low-entropy secret material.
- `DISCLOSED_VALIDATION_OUTCOME_BLIND_ONLY` binds the exact public validation
  roster and permanently narrows the claim; it may never be described as an
  input-blind holdout.

Fresh-branch plaintext cases, high-entropy salts, encryption keys, and typed
expectation values remain outside this package and unavailable to implementers;
disclosed validation case content is public but remains outside this request.
Expectation values remain unavailable in both branches before the valid
post-output-commit unblinding. The sealed-expectation response is committed
only after the branch, schemas, custody, trust, and executor/configuration
locks exist.

## Unavoidable external inputs

JX cannot invent or self-attest the following:

1. sponsor selection of the case/claim branch;
2. sponsor selection of a larger Q2 event and inverse-c-squared grid, or a
   rigorous remainder budget with a narrower claim;
3. named independent custodians, reviewers, affiliations, conflicts, and
   delegations;
4. an out-of-band root-trust decision, signature profile, keyring, revocation
   policy, keys, signed payloads, and signatures;
5. fresh secret cases and custodied salts/keys/ciphertext when the fresh branch
   is selected, or explicit acceptance of the disclosed validation roster and
   narrowed claim otherwise;
6. an externally reviewed registration of the bound Q2 source, plus
   independently implemented Q1, Q3, and Q4 sources and the required runtimes,
   configurations, dependency closures, retained evidence, and reviews;
7. a reviewed nine-component error budget with separately budgeted Hermite
   event interpolation; and
8. typed sealed expectation values and their high-entropy hiding commitment.

## Lifecycle

1. `REQUEST_FROZEN_BLOCKED`: this five-file package is content-derived and all
   response and trust-acceptance fields are null.
2. `EXTERNAL_DECISIONS_AND_ROOT_TRUST`: branch, claim scope, Q2 policy,
   custodians, reviewers, and root-trust acceptance are supplied as separate
   content-derived records.
3. `EXTERNAL_ARTIFACT_PRODUCTION`: assigned parties produce the fourteen role
   artifacts, with independent review, implementation, and signatures wherever
   each frozen role requires them; secret payloads stay with the custodian.
4. `NEW_SUCCESSOR_BINDING`: a new successor specification binds accepted
   responses. This request remains unchanged and still resolves zero slots.
5. `EXECUTION_ID_LAST`: only a complete accepted fourteen-role roster may feed
   the separate execution-qualification identity. Execution, outputs,
   unblinding, and adjudication are outside this package.

## Verification

`verify_handoff_v1.py` is read-only and standard-library-only. It imports no
JX, observer, oracle, dynamics, registry, trajectory, or holdout module and has
no CLI. Load its exact source bytes cache-free with `compile(..., "exec")` and
`exec`, then call `verify_package`. Exact package rosters reject caches,
results, links, and extra files. Verification requires a stable worktree and
performs a final byte-and-roster recheck to reject concurrent mutation. No
trajectory or holdout may be run from this request.
