# JX-O2 open-regeneration design — completion status

## Outcome

The second path is complete at the design stage. We now have a content-locked
protocol for creating a new, fully public Planet X comparison model rather than
pretending to recover the authors' unavailable private checkpoint, angles, or
seeds.

This is intentionally an independent public model family. It is not the
original `cluster_2` simulation and cannot be described as a reproduction of
it.

## What is locked

- The scientific question is the post-300-Myr effect of inserting one compact
  body into a newly generated compact-body-free checkpoint.
- M0 and M1 use byte-identical parent particles, known-body states, stellar and
  tidal histories, numerical rules, nuisance streams, and survey rules. Only
  the registered additional body may physically differ.
- The M1 physical support contains all nine audited 2026 `(mass, a, e, i)`
  rows, each with design weight `1/9`; none may be pruned or reweighted after
  seeing outcomes.
- The ensemble contains 128 independent joint disk-and-stellar histories,
  partitioned into four fixed 32-history convergence blocks. No disk draw or
  stellar schedule is reused across histories.
- One observation-unconditioned angular Latin-hypercube point is assigned to
  each history and shared across all nine physical rows as a common nuisance
  control. Each history-angle point has weight `1/128`.
- The future M1 support therefore has 1,152 equally weighted
  physical-history-angle members. M0 is run once per history, not nine times
  and not counted as replicated evidence.
- Input-generation randomness and later G1 analysis randomness use separate
  contract hashes, future public-beacon events, and disjoint stream namespaces.
  No seed has been realized.
- The planned post-insertion span is 4 Gyr, with the final 1 Gyr analyzed at a
  2.5-Myr cadence. This remains a proposed design, not an executed model.
- The primary future statistic is explicitly
  `2 × [log p(data | fixed M1 mixture) − log p(data | M0)]`; it is one-sided,
  and row or angle maximization cannot affect it.
- Calibration, adequacy, audit, and each-row power samples are fixed at 100,000
  pseudo-catalogs in their respective disjoint roles, with strict tie rules and
  exact Clopper–Pearson gates.
- Candidate 9118 remains permanently excluded from this experiment, including
  transformed or derivative use, subject to an independent lineage audit.

## Verification

- Local registration SHA-256:
  `10d2b2f15c4f3db040d92fd95f6ea4679c6deb9bdc6a9c3862eabbadaecd3d72`
- Exact package inventory: nine regular files; no runner, checkpoint, state,
  realized seed, deck, output, result, or bytecode payload.
- New verification tests: 15/15 passed on Python 3.12 and the current local
  Python runtime.
- Existing JX-O2 reconstruction, retirement, and G0 tests: 44/44 passed.
- Existing local acquisition tests: 8/8 passed.
- Independent final reviews: scientific design PASS, adversarial safety PASS,
  and byte/provenance integrity PASS.
- No dynamics, survey adapter, observed-data analysis, GPU job, or network/GitHub
  action was performed.

## Current scientific status

JX-O2 remains **BLOCKED**. This protocol accepts zero of the 31 G0 requirements,
does not complete G0, is not eligible for G1, and is not externally timestamped.
It provides no evidence for or against Planet X.

The remaining work is to externally register an exact input-generation
contract, finish the unresolved public generator details and software/license
bindings, generate and independently audit the new input checkpoints, and
resolve the OSSOS and DES characterization requirements. Only after every G0
requirement passes may a separate G1 calibration contract be created. Observed
data remains unauthorized until later calibration, independent-reproduction,
and holdout gates pass.
