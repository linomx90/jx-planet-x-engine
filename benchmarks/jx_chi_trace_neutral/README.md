# Neutral REBOUND TRACE fixture

This public harness contains no private JX-CHI or V5 source. It freezes only a
Cartesian N-body fixture, output schedule, REBOUND 5.1.1 configuration, and
artifact format. The private JX-CHI reference is executed separately against
the same contract and compared after both sides are frozen.

The initial Cartesian state is transcribed from REBOUND 5.1.1's
`chaotic_exchange_sim()` regression fixture. The authoritative contract is
`contract.json`.
