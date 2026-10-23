# Wisdom–Holman V9 hierarchical holdout v1

Status: `REGISTERED_DESIGN_ONLY_AWAITING_SOURCE_BOUND_RUNNER_AND_INDEPENDENT_EXECUTION`

This directory preregisters an outcome-blind 32-case hierarchical Newtonian
holdout for the exact frozen rc17 V9 candidate. No V9, DOP853, or IAS15
integration has been run on any holdout case, and no holdout scientific result
exists. The protocol is therefore a design artifact, not a pass, approval, or
new product claim.

The protocol must be committed before the NIST Randomness Beacon 2.0 chain-2
pulse at `2026-10-23T04:00:00.000Z`. After that pulse, the exact raw response
and `outputValue` will be registered additively. The public case seed is
SHA-256 over the frozen canonical protocol bytes and those future beacon bytes,
so the author cannot know or choose the case draws at registration. Twenty
interior cases put four cases at each of
`N = 3, 4, 8, 16, 32`. Twelve additional mandatory confirmatory cases stress
mass, eccentricity, step-envelope, Hill clearance, interaction force, and the
maximum body count with dynamic safety margin. Five separate
inside/outside twin pairs (10 controls total) probe eccentricity, step, mass,
initial-node Hill, and maximum-body-count guard semantics.
The mass and body-count pairs stop at public workspace preparation; the other
three pairs use the source-required one-map-step schedule, with the OUT member
required to reject at the native initial-node guard before that step. No oracle
is run for a control. These mandatory contract checks are never counted in the
32-case oracle-accuracy denominator. Every
draw has a domain-separated SHA-256 definition. There is no rejection sampling,
replacement case, discretionary seed, or outcome-dependent subset.

Protocol revision 2 transparently corrects one constructibility defect found
after the first preregistration commit but before the registered pulse was
available or accessed: the maximum-body-count rejection control has 33 bodies
and therefore needs the deterministic ID `secondary_32`, while revision 1 had
described IDs only through `secondary_31`. Revision 2 extends only that ID rule,
records the superseded commit and both superseded protocol hashes, and changes
no scientific case, gate, candidate, oracle, or randomness choice. No holdout
candidate or oracle execution occurred before this correction.

Protocol revision 3 records a second input-only feasibility correction made
before the pulse was available or accessed. Exact binary64 reconstruction after
the frozen Hill bisection placed the nominal `0.800` target at
`0.8000000000000002`, just outside the inclusive `[0.798, 0.800]` band.
Revision 3 moves only the confirmatory Hill-stress and HILL-IN calibration
target to `0.799`, safely inside that unchanged band. It records the superseded
revision-2 commit and hashes; it changes no roster, gate, candidate, oracle, or
randomness choice, and no candidate or oracle execution preceded the fix.

Protocol revision 4 records a third input-only feasibility correction made
before the pulse was available or accessed. With 32 bodies, converting a
high-precision equal-share remainder left the public sequential binary64 mass
sum three ulps below its target. Revision 4 instead makes the highest-index
secondary the deterministic recurrence-closing body: its GM is the positive
binary64 result of `target - S_previous`, after which the final sequential sum
must equal the target bit-for-bit. The same exact closure applies to science
cases, the unequal interaction stress, and guard controls. No tolerance was
widened, and no roster, gate, candidate, oracle, or randomness choice changed.

Protocol revision 5 records a fourth input-only feasibility correction before
the pulse was available or accessed. The earlier, underspecified ascending
ordered-Jacobi inverse produced precision-dependent tiny residues where outer
Hill-calibration components are algebraically zero. Revision 5 freezes the
equivalent backward inverse from an exact zero total barycenter, with the exact
descending recurrence and subsequent ascending body reconstruction written in
the protocol. A complete synthetic-seed feasibility sweep then produced
identical 160/220-digit records and branch traces for all 32 cases and all 10
controls. No threshold or quantization was added, and no scientific outcome
was generated.

The confirmatory interaction stress targets a reconstructed force ratio in
`[0.0145, 0.0155]` with a frozen unequal-mass construction, at least 3.5
mutual-Hill radii of initial adjacent separation, and all other guards inside.
A mandatory near-`0.1` interaction twin is intentionally excluded:
the simultaneous total-mass and three-mutual-Hill domain does not guarantee a
constructible isolated `0.099` positive control. Existing source-bound native
adversarial tests cover that fail-closed boundary, but they do not count toward
this holdout.

The input constructor is allowed to evaluate only standalone input-domain guard
formulas. It must freeze a create-only binary64 input manifest before importing
or executing V9 or either oracle. The exact input-manifest hash and the exact
source-bound runner must then be registered additively before scientific
execution. If a boundary input cannot be constructed under the frozen rules,
the result is `INVALID_INPUT_CONSTRUCTION`; another case may not be substituted.
Construction uses 160 decimal digits and must reproduce every final binary64
hex value and branch decision in a complete 220-digit rerun. The immutable
ordered-Jacobi body order and all four public Wisdom–Holman spec guard inputs
are frozen in the protocol and retained in each manifest record.

Each case uses `h = P_inner/{D0,2D0,4D0}`, where `D0` is the smallest power of
two satisfying the frozen reconstructed-binary64 periapse-timescale rule and is
at least 64. The rule keeps the coarse nominal periapse-step utilization at or
below 50% of the registered limit. All cases run for ten inner periods.
Independent baseline and tighter runs of SciPy 1.17.0 DOP853 and REBOUND 5.1.1
IAS15 define the oracle floor.
All 32 cases must clear that floor. An unresolved oracle case makes the study
`INCONCLUSIVE_ORACLE_RESOLUTION`; it is not discarded and does not count as a
candidate pass.

For every case, both position and velocity errors must decrease strictly at
second order, with both adjacent ratios in `[3.2, 4.8]`, and the fine `4D0`
nondimensional state error must not exceed `2e-4`. The coarse `D0`, middle
`2D0`, and fine `4D0` relative-energy limits are respectively `1e-5`,
`2.5e-6`, and `6.25e-7`.
Angular momentum, barycenter position, and total momentum have `2e-12` limits;
the middle-resolution forward/backward state limit is `1e-9`; and every exact-solver fallback
count must be zero. All gates and cases are conjunctive.

Even a complete pass can claim only the exact frozen 32-case, non-encountering,
hierarchical trajectories that remain inside the registered domain for ten
inner periods on the declared Linux x86-64 GCC binary64 runtime. It
does not qualify arbitrary N-body dynamics, close encounters, long-term
stability, production ephemerides, observations, or Planet X. It cannot make V9
the default route. Scientific approval still requires execution on an
independently administered supported host and a named external reviewer's
detached cryptographic signature. The current project and V9 claim state remain
`SCREENING_ONLY`.
