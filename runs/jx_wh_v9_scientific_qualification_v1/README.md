# Wisdom–Holman V9 scientific qualification v1

Status: `LOCAL_FORMAL_PASS_ELIGIBLE_FOR_INDEPENDENT_REVIEW_EXTERNAL_APPROVAL_WITHHELD`

This directory registers a prospective independent-oracle execution for the
exact rc17 V9 implementation.  It does not rename existing screening results
as scientific evidence and it does not make V9 the default route.

The formal execution must compare the public V9 route with a standalone
100-digit analytic two-body oracle, SciPy 1.17.0 DOP853 using an independently
written Newtonian right-hand side, and REBOUND 5.1.1 IAS15.  Every numeric,
custody, replay, and platform gate in `protocol_v1.json` is conjunctive.

The candidate and weak-three-body workload were inspected during development.
The protocol therefore records the pilot values and explicitly says the study
is not outcome blind.  Passing locally can reach only
`ELIGIBLE_FOR_INDEPENDENT_REVIEW`.  Final scientific approval requires a
second independently administered supported host and a named external
reviewer/signature.  Those requirements cannot be satisfied by the engine or
by its authors grading their own output.

Protocol revision 2 preserves the registered revision-1 SHA-256 and records
one pre-execution constructibility amendment.  A dry-run showed that
reconstructing the nominal binary64 `e = 0.9` boundary can round just above the
engine's inclusive eccentricity guard.  Before any formal scientific result
was generated, that boundary cell was moved to `e = 0.899`.  No metric,
threshold, oracle, horizon, or decision rule changed.

Protocol revision 3 also preserves both earlier hashes and discloses a second
pre-execution feasibility correction.  A six-case stress dry-run, not a formal
qualification result, showed up to about `1.7e-8` normalized state error after
100 periods at the largest scale and highest eccentricity.  The original
`5e-10` analytic state budget was below accumulated binary64 solver roundoff.
Before the formal 240-case execution, the analytic position/velocity budget
was fixed at `5e-8`, the energy/angular budgets at `5e-10`/`5e-11`, and the
round-trip budget at `1e-7`.  This is explicitly nonblind feasibility tuning;
it is why local passage cannot count as final scientific approval.

The claim ceiling covers only the frozen non-encountering Newtonian two-body
portfolio and weak hierarchical three-body benchmark on the declared Linux
x86-64 GCC runtime.  It does not authorize production ephemerides,
observational inference, close encounters, general N-body dynamics,
cross-platform equivalence, long-term stability, superiority, or a Planet-X
claim.  V9 remains explicit opt-in and the project remains `SCREENING_ONLY`.

## Smoke contract versus formal execution

The repository test matrix runs the qualification harness only in its reduced
`smoke` profile, without IAS15.  That profile checks the analytic oracle,
candidate replay, report serialization, custody rules, and anti-forgery
validation, but it deliberately fails the formal-profile gate.  A smoke report
is diagnostic output, is not a qualification result, and must not be promoted
or retained as formal evidence.

The `formal` profile is a separate, explicitly acknowledged execution.  It
requires the exact registered dependency versions and a bound engineering
receipt before it can evaluate the full locked workload.  Even a conjunctive
formal pass can emit only `ELIGIBLE_FOR_INDEPENDENT_REVIEW`; it cannot grant
scientific approval.  The source archive carries this runner, its smoke test,
the complete declared non-package source closure, the locked protocol, and
this README for reconstruction, but no generated formal result.

Formal execution starts from the committed, clean source checkout with
`PYTHONPATH=src:.`.  The runner then loads the candidate science route from
the separately validated build-A wheel and verifies that its native extension
matches the registered runtime identity.  A wheel-only launch environment is
not the registered execution boundary.  Generate the initial result outside
the checkout so that the worktree remains clean through the mandatory live
verification; retain the verified result afterward.

`--verify` is an intrinsic schema, digest, and gate-consistency check.  Because
its SHA-256 digests are unkeyed, it is not evidence of authorship or an
independent reproduction.  Every formal result must also pass
`--verify-live`, with all six registered engineering paths supplied, so the
runner repeats the complete profile and compares the canonical report.

The registered `next_gate` also requires a separately preregistered,
outcome-blind hierarchical holdout.  A second independently administered host
and a named, cryptographically signing external reviewer are necessary but
not sufficient for final approval until that holdout is completed.  Any later
review or approval must be an additive signed artifact; the local formal
report's `external_review_completed` and `scientific_approval_granted` fields
remain false permanently.

## Retained local result

The registered formal execution at Git commit
`d1248279a9054d428d4ce9e59dc12937a35009e5` completed with all 93 conjunctive
gates passing and status `ELIGIBLE_FOR_INDEPENDENT_REVIEW`.  The retained
`result_v1.json` is 20,988,601 bytes, has raw SHA-256
`807a8d40dff32e229876f47818578fd0e6a0aa305f1780e8a8f40aebad55b5d9`,
and has domain-separated content SHA-256
`68e8301760f5cb74b32427097e84be1a31ce110a0ee36fa86bb520e84c130e86`.
It covers all 240 registered analytic cases and the complete registered weak
three-body workload.  Intrinsic verification and a full same-host
`--verify-live` reproduction both passed before the result was retained.

The engineering logs and manifests used by that execution are retained below
`engineering/`; `independent_review_handoff_v1.json` binds those files, the
registration, and the result.  To repeat live verification, check out the
recorded execution commit above and use the separately registered build-A and
build-B artifacts.  The later evidence-retention commit is not the execution
checkout.  This local result is not an external reproduction, an outcome-blind
holdout, or scientific approval.
