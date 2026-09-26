# JX test profiles

The repository test tree is intentionally heterogeneous. Current engine tests,
historical frozen probes, source-closed packages, optional CUDA checks, and two
incompatible REBOUND releases cannot share one Python process or one source
identity. `tools/test_matrix_manifest.json` is the authoritative inventory and
`tools/run_local_test_matrix.py` enforces it.

## Coverage contract

Every regular `tests/test_*.py` file must occur exactly once in the manifest.
Duplicate assignments, stale paths, and unassigned files are fatal before any
test starts. Each executable file runs in a separate child process. Adding or
renaming a test therefore requires updating the manifest in the same change.

The current 278-file inventory is:

| Profile | Files | Disposition |
| --- | ---: | --- |
| `live-current` | 199 | Run against the current source tree |
| `cuda-optional` | 23 | Run CPU-safe checks; hardware-only cases self-skip |
| `cuda-hardware-only` | 5 | Run only when a compatible CUDA device is available |
| `rebound-4.4.11` | 3 | Run only with exact REBOUND 4.4.11 |
| `rebound-5.1.1` | 5 | Run only with exact REBOUND 5.1.1 |
| `gr15-eih-external` | 2 | Run against the exact external GR15/EIH comparison runtime |
| `performance-local` | 3 | Run source-bound native/TRACE evidence with exact REBOUND 5.1.1 |
| `library-catalog-local` | 1 | Verify the catalog against the read-only local library |
| `source-closed-reph` | 1 | Run from a virtual environment with its source-closed package |
| `retained-lunar-coupled` | 1 | Run with its lineage-specific retained mpmath wheel |
| `retained-lunar-torque-spice` | 1 | Run with retained mpmath, spiceypy, and CSPICE binaries |
| `exact-lunar-launcher` | 1 | Run with the exact reconstructed R2 launcher runtime |
| `frozen-history` | 31 | Replay only from the matching frozen source/runtime custody |
| `superseded-preregistration` | 2 | Accounted for, but not executed after later outputs exist |

## Commands

Validate or inspect the inventory without running tests:

```bash
python3 tools/run_local_test_matrix.py --validate-only
python3 tools/run_local_test_matrix.py --list
python3 tools/run_local_test_matrix.py --list-files
```

Run the current engine scope, all current-source profiles, or the local
source-closed evidence profiles:

```bash
python3 tools/run_local_test_matrix.py --profile engine
python3 tools/run_local_test_matrix.py --profile live
python3 tools/run_local_test_matrix.py --profile evidence
```

Run exact REBOUND profiles from their corresponding environments:

```bash
python tools/run_local_test_matrix.py --profile rebound-4.4.11
python tools/run_local_test_matrix.py --profile rebound-5.1.1
python tools/run_local_test_matrix.py --profile performance-local
python tools/run_local_test_matrix.py --profile library-catalog-local
```

`--runtime-python PATH` selects an explicit interpreter for REBOUND and REPH
profiles. `--library-root PATH` relocates the read-only retained library used by
the lunar profiles.

## Frozen and superseded evidence

Frozen-history probes deliberately reject the evolved live engine. A refusal
such as “the frozen engine source roster changed” is a custody success, not a
live-engine regression. Those files must be replayed only after reconstructing
their matching historical source and dependency boundary. The runner refuses a
direct live execution request for this profile. This profile also contains
historical lunar diagnostics bound to pinned PyERFA and obsolete qualification
harnesses that clone the full evidence tree for each mutation; running those
against the live tree is neither a release gate nor a valid current-source test.

The superseded preregistration tests assert that later execution artifacts are
absent. Those artifacts now exist as preserved evidence, so deleting them to
make an old state assertion pass would be destructive and scientifically
incorrect. The manifest records these files without executing that obsolete
state assertion.

## CI policy

CI validates exact manifest coverage first. It then runs current-source and
CUDA-optional files with per-file process isolation, and runs REBOUND 4.4.11
and 5.1.1 in separate exact-version jobs. The Ubuntu 26.04 post-V5 job verifies
the frozen numerical runtime and confirms that the evolved live source is
rejected by the historical custody boundary. Retained-library lunar lanes stay
local because their immutable inputs are not part of the Git checkout. A
separate packaging job requires byte-identical wheel and source artifacts from
two independent clean builds and smoke-tests fresh installs of both formats.
