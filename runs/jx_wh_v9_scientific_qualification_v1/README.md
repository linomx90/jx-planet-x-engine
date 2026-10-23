# Wisdom–Holman V9 scientific qualification v1

Status: `REGISTERED_AWAITING_EXECUTION_EXTERNAL_REVIEW_REQUIRED`

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
