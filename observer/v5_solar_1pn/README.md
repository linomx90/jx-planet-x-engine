# V5 Q2 Decimal perihelion observer candidate

This directory contains the smallest separately coded observer needed to
measure a paired perihelion signal from already-produced checkpoint series.
It is outside `jxplanetx`, imports no JX dynamics, integrator, event, oracle,
test, or registry helper, and performs no trajectory integration or artifact
I/O.

The public path is deterministic and self-contained:

- all inputs and outputs are finite `Decimal` values. State interpolation,
  radial values, orientation, angles, and slopes use the frozen p90 context;
  a separately frozen p4096 exact-or-fail layer constructs Bernstein
  coefficients, performs de Casteljau subdivision, and retains dyadic
  unit-parameter brackets and event epochs. Any inexact certificate/time
  arithmetic is rejected;
- each checkpoint interval uses a cubic Hermite position fixed by endpoint
  `(r,v)` values, with velocity defined as the derivative of that same
  polynomial;
- every interval's degree-five Hermite `r dot v` polynomial is certified with
  exact Decimal Bernstein coefficients and sign variation before event
  scanning; hidden, multiple, tangent, plateau, or unresolved roots fail
  closed;
- perihelia are confirmed negative-to-positive crossings of `r dot v`, and
  Decimal bisection stops only on coordinate-time bracket width;
- every p90 endpoint, bisection midpoint, and reported-event radial sign must
  agree with the exact certificate. A deeply bisected unit parameter may be
  rounded once for p90 state evaluation only when the induced coordinate-time
  displacement is no larger than the frozen root tolerance;
- the exact initial `r dot v = 0` event is excluded by an explicit state
  machine that first requires outward motion and then inward motion, without
  an arbitrary time cutoff;
- the full supplied series must contain exactly two postinitial events;
- the oriented basis is frozen from the shared initial `r` and `r cross v`,
  off-plane residual is bounded and retained, and angles use a Decimal
  `atan2` in radians;
- each arm is unwrapped independently by the unique nearest principal branch;
  an exact pi tie fails closed; and
- the paired value is the 1PN secular slope minus the Newtonian secular slope
  on the same checkpoint schedule, initial input, observer configuration, and
  common numerical source-configuration binding (apart from force mode).

The arctangent kernel applies the argument reduction in
[NIST DLMF 4.45.E8](https://dlmf.nist.gov/4.45.E8), followed by the ascending
alternating series in
[NIST DLMF 4.24.E3](https://dlmf.nist.gov/4.24.E3).  The first omitted term
bounds the reduced alternating-series remainder; its tolerance is divided by
the reconstruction factor. Pi is evaluated with Machin's identity using the
same bounded kernel and guard digits. Python's explicit local Decimal context
and signal behavior are documented in the
[Python `decimal` documentation](https://docs.python.org/3/library/decimal.html).

The root certificate uses the Bernstein sign-variation theorem: the number of
interior real roots is bounded by the coefficient sign variations and differs
from that count by an even number. Together with endpoint signs, zero
variations certify a root-free interval and one variation plus opposite
endpoint signs certifies exactly one root. The basis properties and theorem
are described by Mourrain, Rouillier, and Roy in
[“The Bernstein Basis and Real Root Isolation”](https://library.slmath.org/books/Book52/files/24roy.pdf),
MSRI Publications 52 (2005), DOI 10.1017/9781009701259.025. Endpoint roots,
zero coefficients, and higher variation counts use stricter fail-closed rules.
Both arithmetic-layer precisions and all dense-state, root, isolation, and
angle method identifiers are bound into the observer-configuration identity.

Every result retains its complete immutable trajectory series, schedule,
initial input, observer configuration, domain-separated SHA-256 identities,
event brackets, orientation residuals, and nonauthorization metadata.

This source remains `CANDIDATE_PENDING_INDEPENDENT_REVIEW` and produces only
`MODEL_OUTPUT`. It cannot fill the frozen external Q2 registration or
authorize execution, registry promotion, qualification outcomes, or claims.
Step extrapolation, event-grid comparison, inverse-c-squared extrapolation,
the `6*pi` comparison, expectation handling, error budgets, and gate
adjudication remain explicitly out of scope and unresolved.
