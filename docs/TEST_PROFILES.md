# JX test profiles

The repository test tree is intentionally heterogeneous. Current engine tests,
CUDA hardware checks, external GR15/EIH comparisons, historical frozen probes,
and two incompatible REBOUND releases cannot share one Python process or one
source identity. `tools/test_matrix_manifest.json` is the authoritative
inventory and `tools/run_local_test_matrix.py` enforces it.

## Coverage contract

Every regular `tests/test_*.py` file must occur exactly once in the manifest.
Duplicate assignments, stale paths, and unassigned files are fatal before any
test starts. Each executable file runs in a separate child process. Adding or
renaming a test therefore requires updating the manifest in the same change.

The recovered baseline contains 73 repository-bound test files:

| Profile | Files | Disposition |
| --- | ---: | --- |
| `live-current` | 52 | Run against the current source tree |
| `cuda-hardware-only` | 2 | Run only when a compatible CUDA device is available |
| `rebound-4.4.11` | 3 | Run only with exact REBOUND 4.4.11 |
| `rebound-5.1.1` | 4 | Run only with exact REBOUND 5.1.1 |
| `gr15-eih-external` | 2 | Run against the exact external GR15/EIH comparison runtime |
| `frozen-history` | 10 | Replay only from the matching frozen source/runtime custody |

## Commands

Validate or inspect the inventory without running tests:

```bash
python3 tools/run_local_test_matrix.py --validate-only
python3 tools/run_local_test_matrix.py --list
python3 tools/run_local_test_matrix.py --list-files
```

Run the current engine scope, all current-source profiles, or the active
external-comparison evidence profiles:

```bash
python3 tools/run_local_test_matrix.py --profile engine
python3 tools/run_local_test_matrix.py --profile engine --optimized
python3 tools/run_local_test_matrix.py --profile live
python3 tools/run_local_test_matrix.py --profile evidence
```

Run exact REBOUND profiles from their corresponding environments:

```bash
python tools/run_local_test_matrix.py --profile rebound-4.4.11
python tools/run_local_test_matrix.py --profile rebound-5.1.1
```

`--runtime-python PATH` selects an explicit interpreter for the REBOUND and
GR15/EIH external profiles.

## Baseline recovery boundary

The rc16 Git tree originally contained a 284-path manifest but only 73 tracked
test files. The omitted paths came from a larger, uncommitted development
workspace and several require source modules that are not in rc16. They are
therefore not active baseline tests. Importing those files alone makes the
engine profile fail at import time and would falsely mix different source
lineages.

Two omitted test files have individual hashes in the historical rc16 release
receipt, but their complete dependency closures are not bound to the release
tree. They remain custody evidence outside this active inventory. The
historical rc16 receipt is preserved unchanged; this manifest describes only
the tests that can be traced to and executed from the current Git baseline.

Two tracked tests are also frozen rather than live: one asserts old rc10
version state, and one requires retained population inputs absent from the Git
tree. Their code remains inventoried, but neither is treated as a current-source
release gate.

## Frozen and superseded evidence

Frozen-history probes deliberately reject the evolved live engine. A refusal
such as “the frozen engine source roster changed” is a custody success, not a
live-engine regression. Those files must be replayed only after reconstructing
their matching historical source and dependency boundary. The runner refuses a
direct live execution request for this profile. This profile also contains
historical integrator/comparison probes and obsolete qualification harnesses;
running those against the live tree is neither a release gate nor a valid
current-source test.

## CI policy

CI validates exact manifest coverage in a dedicated prerequisite job. The
current workflow then runs the 13 `test_engine*.py` files under normal and
optimized Python, runs the supported GR15 subset, and verifies reproducible
wheel/source packaging with fresh-install smoke tests. Other manifest profiles
remain explicit local lanes; they are not described as CI-covered until a
corresponding workflow job exists.
