# JX Challenger V1

## Purpose and boundary

`benchmarks/jx_challenger_v1.py` is the smallest reproducible comparison
bundle for the current JX Newtonian solvers. It launches three existing
benchmark programs in separate Python subprocesses and publishes their reports
with one aggregate manifest. It adds no force model, integrator, score, or
ranking rule.

The orchestrator itself uses only the Python standard library. The numerical
lanes are not “pure Python”: JX uses NumPy/CPU binary64 arrays and REBOUND uses
its native extension when enabled.
REBOUND is an optional external comparator; it is not copied, vendored, or
linked into the JX engine. Challenger V1 is a smoke protocol, not a calibrated
full study. Every result remains unauthenticated `MODEL_OUTPUT` and does not
establish scientific qualification, production fitness, speed superiority,
accuracy superiority, or equivalence to REBOUND.

## Fixed study roster

V1 always runs these three smoke studies:

| Study | Locked workload | JX lanes | Optional exact REBOUND 5.1.1 lanes |
| --- | --- | --- | --- |
| Analytic binary | Equal-mass circular binary; one period with `samples_per_period=16` | Fixed-grid Cartesian KDK and adaptive RKF78 reference | Leapfrog |
| Weak hierarchy | Locked weak hierarchical three-body smoke profile | Ordered-Jacobi Wisdom--Holman | WHFast and IAS15 reference |
| Close scatter | Locked all-active three-body close-scatter smoke profile, without the separate limitation witnesses | Transactional Wisdom--Holman/RKF78 hybrid P/256 lane | MERCURIUS, TRACE, and IAS15 reference |

The subprocess boundary is part of the protocol. It keeps each mature
comparator's module state, provenance checks, serialization, and failure path
isolated from the other comparators. This is not a process-independence or
security-sandbox attestation. The runner does not reconstruct states, import result objects from
one study into another, or create a new cross-study numerical score.

The analytic binary can expose conservation and phase error against an exact
fixture solution. The hierarchical and close-scatter fixtures exercise
different solver domains and retain their comparator-specific metrics and
gates. Agreement on these three short, selected fixtures is useful regression
evidence only; it does not demonstrate general N-body accuracy, long-horizon
stability, collision detection, event location, regularization, or a mapped
chaos horizon.

The compact `relative_total_energy_max` and angular-momentum fields are maximum
absolute errors at retained samples, not signed secular drift or continuous
envelopes. The binary uses the analytic fixture's initial invariants; the other
studies use each lane's first retained state. Phase fields are deliberately not
homogenized: binary phase is unwrapped candidate-minus-analytic angle, WHFast
phase (when available) is wrapped per-body candidate-minus-IAS15 angle, and
hybrid phase is an unwrapped reference-minus-candidate relative-vector proxy.
A null phase field means no numerical reference was executed, not zero error.

## Run the JX-only protocol

From the repository root, choose a destination directory that does not already
exist:

```bash
python3 -B benchmarks/jx_challenger_v1.py \
  --rebound-mode disabled \
  --output-dir ../jx-challenger-v1-smoke
```

`--output-dir` is required. The runner fails rather than overwrite an existing
path. It prepares the complete bundle privately and publishes the directory
only after all three children and aggregate validation succeed. A successful
directory contains exactly `leapfrog_smoke.json`, `whfast_smoke.json`,
`hybrid_smoke.json`, `challenger_v1.json`, and `SHA256SUMS`. Keeping the
destination outside the Git worktree prevents generated diagnostics from
becoming accidental source inputs or checkpoint payloads.

`disabled` is the default and does not require REBOUND. It is the appropriate
mode for checking the native JX lanes, report construction, provenance, and
repeatable scientific bookkeeping.

V1 freezes the exact current engine-source-tree identity and the exact three
comparator source files. Changing those bytes requires a new Challenger
version; a self-consistent report from altered source is rejected as V1.

## Run the optional external comparison

The external lanes require the exact REBOUND 5.1.1 distribution. The comparator
scripts reject another version. Create a separate environment and do not use
the repository's general-purpose `rebound` or `ias15` optional extras, which
remain pinned for older workflows:

```bash
python3 -m venv ../jx-rebound-5-1-1-venv
. ../jx-rebound-5-1-1-venv/bin/activate
python -m pip install -e ".[engine]"
python -m pip install "rebound==5.1.1"

python3 -B benchmarks/jx_challenger_v1.py \
  --rebound-mode required \
  --output-dir ../jx-challenger-v1-rebound-smoke

JX_RUN_CHALLENGER_REBOUND_5_1_1=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest -v tests.test_jx_challenger_v1
```

Use `--rebound-mode auto` when an all-or-none availability probe is useful. It
runs all external lanes if the exact dependency is available and otherwise
records their unavailability; disagreement among the three child environments
fails closed. `required` is the audit mode: it fails closed if any study cannot
execute its exact REBOUND lanes. `disabled` always selects the JX-only path.

## Reading and preserving the bundle

Treat the output directory as one artifact. Keep all five files together.
`SHA256SUMS` binds the raw bytes of all four JSON payloads, while
`challenger_v1.json` binds the fixed roster and the child results; the child
reports retain the detailed state, invariant, phase, work, runtime, provenance,
claim-control, and local content-integrity records defined by their original
comparators. Content hashes detect changed bytes or semantics under their
declared recipes, but they are not signatures and confer no scientific
authority. Aggregate digests use a fully type-tagged JSON tree and hexadecimal
binary64 values, so distinct JSON types and signed zero remain distinct.

For a meaningful repeatability check, run the same command twice into two new
directories with the same interpreter, NumPy build, machine environment, source
tree, and REBOUND mode. Compare the scientific/state digests and declared
metrics. Raw wall-clock values and filesystem paths are diagnostics and must not
be used as cross-method speed evidence or expected to match between runs. Some
native child provenance digests intentionally include the exact source
location, so aggregate repeatability is scoped to the same source location and
runtime; path-independent trajectory/component digests remain available in the
raw reports for scientific comparison after relocation.

Publication uses a private stage and a whole-directory rename. It prevents
ordinary partial publication and refuses an existing destination, but it is
not a hostile same-user filesystem or concurrent no-clobber security proof.

V1 deliberately stops at a smoke bundle. Full 100-period binary,
10/100-period hierarchical, calibrated hybrid refinement, limitation probes,
larger fixture ensembles, hardware replication, and independent peer review
remain separate work. Any claim beyond the exact fixture, method settings,
runtime identity, and metrics recorded in these reports requires those later
protocols.
