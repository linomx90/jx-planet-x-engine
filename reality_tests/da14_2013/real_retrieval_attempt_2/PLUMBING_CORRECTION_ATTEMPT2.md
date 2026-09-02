# JX Reality Test 1 — Attempt 2 Plumbing Correction

The first real-retrieval workflow stopped before launching the adapter because the runner shell attempted to open stdout/stderr files inside a mode-0700 directory owned by the unprivileged `jxretrieve` identity.

Attempt 2 changes only command plumbing: the shell that opens stdout/stderr now executes as `jxretrieve` inside the same owned directory. The source adapter, approved endpoints, target, cutoff, holdout window, normalization, encryption format, and scientific rules are unchanged.

No network request occurred in attempt 1. This correction has no scientific effect.
