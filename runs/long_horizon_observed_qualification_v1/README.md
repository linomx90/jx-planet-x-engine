# Observed-input long-horizon qualification v1

This directory retains the create-only preparation receipt for JX's bounded
1,000-year observed-input qualification.  It freezes the 108-state observed
population, six-body DE441-derived source model, candidate and refinement
timesteps, independent REBOUND comparator, restart checks, numerical error
limits, resource ceilings, and 50 implementation-source hashes.

`PROTOCOL.json` has protocol content digest
`3a566fb5a98a3d3709cd7dbda9983ab6a4670b15fbcb74300eb0a3f78377fc35`.
Its status is `PREPARED_NOT_EXECUTED` and `execution_authorized` is `false`.
No dynamics integration was run to create this directory.

Before an execution tool may be added or any staged qualification may start,
it must bind exact installed Python, NumPy, SciPy, CuPy, CUDA, REBOUND, native
extension, CPU, and GPU identities; materialize and verify the Cartesian
inventory; and satisfy every `required_before_execution` item in the receipt.
A later passing qualification would remain engineering validation only, not a
science result or authorization for a 4.5-Gyr production run.
