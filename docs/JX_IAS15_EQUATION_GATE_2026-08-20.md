# JX adaptive IAS15 equation gate

## Question

Can the preserved anomaly-zone dynamics support a reproducible 100,000-year
source-minus-control conclusion when integrated with adaptive IAS15 at
`epsilon=1e-12` and `epsilon=1e-14`?

## Equations

The integrated Newtonian system is

`r_i_ddot = G sum_(j!=i) m_j (r_j-r_i)/|r_j-r_i|^3`.

After individual close-encounter trajectories became chaotic, the locked
population observables were

`f_q<30(t) = (1/N) sum_i I[q_i(t)<30 AU]`

and the one-dimensional Wasserstein distance between the source and matched
control perihelion samples.

## Result

The pointwise trajectory gate failed in every tested phase. The untouched
phase-2 no-source population passed, with maximum mean-q and Wasserstein-q
disagreement of `0.0295453 AU`.

The untouched phase-2 middle-family source population failed its locked
`0.1 AU` convergence gates:

- maximum mean-q disagreement: `0.145672 AU`;
- maximum Wasserstein-q disagreement: `0.155080 AU`;
- low-q count mismatch: `0`;
- bound mismatch: `0`.

The source-minus-control effect itself exceeded the numerical-resolution floor
at both tolerances, but this cannot override failure of the source population's
prerequisite convergence gate.

## Governing verdict

`BLOCKED_SOURCE_POPULATION_NONCONVERGENCE`

The associated source inference is invalid. This is not evidence for or
against the existence of a compact source. The next valid method is a
preregistered phase/uncertainty ensemble whose distributional convergence is
tested directly. Repeating the same deterministic chaotic trajectories is not
the next test.
