# TNO Stage 1.5 V6 continuation smoke

This directory retains the post-preregistration outcome of the bounded V6
continuation smoke. It is setup evidence only, not a numerical qualification,
dynamical classification, Planet Nine result, or scientific result.

- Preregistered commit: `e74567bffebd51ef553707d276b9c546a8558cea`
- Annotated tag: `tno-stage15-resolved11-encounter-v6-preregistered-20261010`
- Receipt file SHA-256: `476c7ffe8b6297dfc81276a743a06c5657d4e6e589a4d636faa43bb9852963ae`
- Receipt content SHA-256: `8b746a630e29960611b90b7c9ea083fa7c0c21a42b0f405fa3255ece520e16be`
- Receipt size: `4,079,552` bytes

The smoke executed one DE441 batch-0 H64 annual segment (64 fixed steps),
confirmed bitwise source endpoints and the host-binary64 active endpoint-minimum
policy, and successfully validated the next continuation. It executed no second
segment and did not mutate the V6 qualification output root.

The receipt also records 576 encounter-route events: every one of the nine
tracers routed at each of the 64 steps for `ACCELERATION_BOUND` against Mercury.
That observation triggered a separate route-capacity and acceleration-bound
review before any 1,000-year qualification run.
