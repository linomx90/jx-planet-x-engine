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

The current 100-file inventory is:

| Profile | Files | Disposition |
| --- | ---: | --- |
| `live-current` | 65 | Run against the current source tree |
| `cuda-optional` | 18 | Run CPU-safe checks; hardware-only cases self-skip |
| `cuda-hardware-only` | 0 | Reserved for cases without a no-device skip path |
| `rebound-4.4.11` | 3 | Run only with exact REBOUND 4.4.11 |
| `rebound-5.1.1` | 4 | Run only with exact REBOUND 5.1.1 |
| `gr15-eih-external` | 2 | Run against the exact external GR15/EIH comparison runtime |
| `performance-local` | 0 | Reserved for source-bound local performance evidence |
| `library-catalog-local` | 0 | Reserved for read-only catalog evidence |
| `source-closed-reph` | 0 | Reserved for source-closed REPH evidence |
| `retained-lunar-coupled` | 0 | Reserved for retained coupled-lunar evidence |
| `retained-lunar-torque-spice` | 0 | Reserved for retained lunar torque/SPICE evidence |
| `exact-lunar-launcher` | 0 | Reserved for exact-environment lunar launchers |
| `frozen-history` | 8 | Replay only from the matching frozen source/runtime custody |
| `superseded-preregistration` | 0 | Reserved for superseded preregistration states |

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
direct live execution request for this profile. Its eight-file roster is the
legacy challenger/post-V5, REBOUND comparison, and V5 solar-1PN qualification
set recorded in the executable manifest; running those files against the live
tree is neither a release gate nor a valid current-source test.

The superseded preregistration tests assert that later execution artifacts are
absent. Those artifacts now exist as preserved evidence, so deleting them to
make an old state assertion pass would be destructive and scientifically
incorrect. The manifest currently keeps an empty reserved profile for that
superseded state assertion; no live test file is assigned to it.

## CI policy

CI validates exact manifest coverage first. It runs the CPU-safe restricted-WH
precision and checkpoint contracts with pinned NumPy/SciPy, the current engine
modules in isolated lanes (including optimized-Python checks), the generic
SPICE/lunar/science-input ephemeris contracts with pinned SpiceyPy, the frozen
population profile under exact REBOUND 4.4.11, and the GR15 comparison gates in
their declared dependency job. CUDA device cases remain self-skipping on
generic runners and are exercised on the qualified local RTX. Retained or
source-closed evidence lanes stay local because their immutable inputs are not
part of the Git checkout. A separate packaging job requires byte-identical
wheel and source artifacts from two independent clean builds and smoke-tests
fresh installs of both formats.
