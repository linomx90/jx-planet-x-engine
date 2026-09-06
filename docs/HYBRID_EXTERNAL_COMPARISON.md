# Optional hybrid external comparison

`benchmarks/rebound_hybrid_comparison.py` is a measurement-only comparison of
the JX transactional Wisdom--Holman/RKF78 hybrid with REBOUND 5.1.1
MERCURIUS and TRACE. REBOUND is optional, loaded lazily, and used only as an
external comparator and numerical-reference provider. No REBOUND code is
copied, vendored, or linked into the JX engine.

The study reports fixture-specific numerical-reference errors, empirical
refinement orders, and descriptive regression thresholds. It does **not**
establish general or theoretical order, superiority, qualification, production
fitness, collision handling, event location, regularization, dense output,
global clearance, long-term boundedness, floating-point symplecticity, or
floating-point reversibility.

## Running it

The ordinary smoke profile is intentionally short and does not require
REBOUND:

```bash
PYTHONPATH=src python3 benchmarks/rebound_hybrid_comparison.py \
  --profile smoke --rebound-mode disabled
```

The calibrated profile requires the exact pinned REBOUND 5.1.1 distribution
and `OMP_NUM_THREADS=1`:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 benchmarks/rebound_hybrid_comparison.py \
  --profile full --rebound-mode required --output hybrid-report.json
```

`--include-limitations` additionally executes the endpoint-tunneling and
central-periapse limitation witnesses. `auto` executes the external lanes only
when the exact runtime is available; `disabled` always produces a JX-only
report. In a full JX-only report, reference-dependent accuracy, order, and
envelope gates are explicitly unevaluated.

Ordinary unit tests stay fast. The complete calibrated run is opt-in:

```bash
OMP_NUM_THREADS=1 JX_RUN_REBOUND_HYBRID_FULL=1 PYTHONPATH=src \
  python3 -m unittest -v tests.test_rebound_hybrid_comparison
```

The full JX study includes the public hybrid solver's mandatory semantic replay
and one additional complete report replay. On the calibration host this takes
several minutes. The reported `summed_primary_lane_timing_seconds` is exactly
the sum of selected lane integration intervals; it excludes setup, controls,
limitation probes, and the report replay. All raw timings are diagnostics and
are not cross-method speed comparisons.

## Locked physical input

The main fixture contains three active Newtonian bodies in the fixed order
`STAR`, `INNER`, `OUTER`, with `G=1`. Its 24 row-major little-endian float64
values `(m,r,x,y,z,vx,vy,vz)` have SHA-256:

```text
c2c5eb2a44a58fe1d8f61838cf1adb35030ed816f4b9038bc6b8a68b1b661f8c
```

The complete float-hex rows are constants in the benchmark and are checked
before execution. Every REBOUND lane is constructed afresh from those bytes;
particle mass, radius, Cartesian state, body count, active count, test-particle
mode, gravity, collision, boundary, softening, and initial clock readbacks are
checked. The benchmark never reconstructs orbital elements, calls
`move_to_com`, transfers state between lanes, interpolates output, or relabels
an observed state to a nominal epoch.

The period scale is the exact binary64 value
`0x1.921fb54442d18p+2`. The named full profile uses:

| Lane family | Divisors | Outer steps |
| --- | ---: | ---: |
| JX hybrid | 256, 512, 1024 | 1630, 3260, 6520 |
| MERCURIUS | 128, 256, 512, 1024 | 815, 1630, 3260, 6520 |
| TRACE | 128, 256, 512, 1024 | 815, 1630, 3260, 6520 |

All headline lanes retain the same 103 **declared epoch labels**: every `P/16`
plus the exact final label, reached through divisor-specific outer-step indices.
MERCURIUS and TRACE still execute and synchronize every outer step. Their full
observed-clock and state arrays are materialized transiently, hashed, and used
for every-node diagnostics; only the 103 headline nodes are retained in the
study result and report. External observed-clock vectors differ by divisor;
each candidate is compared to IAS15 at that candidate's exact observed clock.
Cross-divisor empirical orders therefore compare maxima measured on
divisor-specific observed-clock vectors, not identical observed epochs.

## Method settings and clock protocols

MERCURIUS requests `r_crit_hill=3` and `safe_mode=1`. TRACE requests
`r_crit_hill=3`, `peri_crit_eta=1`, and `peri_mode=FULL_BS`, with no custom
callbacks. Both request `gravity=basic`; the pinned runtime reads back
`gravity=custom` after stepping. Requested settings, pristine integrator
readbacks, configured pre-step readbacks, and genuine post-step readbacks are
retained separately. The TRACE 5.1.1 pristine `r_crit_hill=3` readback is also
kept distinct from the documentation default of 4.

Two main IAS15 protocols answer different alignment questions; a third,
optional protocol is confined to the central-periapse limitation:

- The authoritative JX reference uses epsilon `1e-14`, `PRS23`, `min_dt=0`,
  initial `dt=P/128`, and sequential exact-finish stops only at the common JX
  headline labels. A second run with initial `dt=P/1024` is a sensitivity check.
- Each external reference uses epsilon `1e-12`, `PRS23`, `min_dt=0`, initial
  `dt` equal to that fixed lane's `h`, and sequential exact-finish stops at
  every observed outer node before headline downsampling.
- The optional central-periapse limitation uses a separate IAS15 reference with
  epsilon `1e-12`, `PRS23`, `min_dt=0`, initial `dt=P/64`, and one exact-finish
  stop at the common observed MERCURIUS/TRACE clock. Its target and observed
  clock, requested/initial/post settings, 429 adaptive steps, and zero
  `iterations_max_exceeded` are retained and checked independently.

Requested, initial-effective, and post-run IAS15 epsilon, minimum step, mode,
`dt`, `dt_last_done`, clock targets, adaptive `steps_done`, and synchronization
counts are retained. Adaptive step counts and timings are not compared across
these distinct protocols.

## JX hybrid control and accounting

The JX lane uses a fixed integer outer lattice. A complete typed far probe is
attempted on every outer step. A far pass commits the Wisdom--Holman candidate.
A near decision discards all provisional work and integrates the entire original
macrostep from its committed Cartesian start node with the guarded RKF78 local
IVP solver. There is no latch, hysteresis, partial-prefix commit, event
location, nested public child call, or nested public child replay.

The calibrated control uses a Wisdom--Holman pair floor of `0.01`, a Jacobi
periapse floor of `0.05`, encounter certificate floors `(0.25,0.25,0.01)`,
pair position/velocity absolute tolerances `1e-10`, relative tolerances
`1e-12`, centroid tolerances `1e-12`, and exact controller/resource ceilings
retained in each lane's settings.

The full regression gates require:

- exact FAR/NEAR totals `(1022,608)`, `(2068,1192)`, `(4153,2367)`;
- FAR + NEAR equal to the outer-step count;
- every near child has one proposal, one accepted substep, no rejection, and
  13 force evaluations;
- zero certificate rejections;
- at least one real
  `FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN` decision at
  `POST_DRIFT_PRE_FORCE_PATH_SCREEN`, with phase totals consistent with the
  complete near ledger;
- primary accounting equals the mandatory hybrid replay and public totals are
  exactly their sum.

The result retains all observed reason and phase histograms. The encounter
certificate applies only to the local full-interval IVP under its explicit
assumptions; it is not a collision detector or a global-clearance proof.

## Metrics and descriptive gates

Position and velocity `vector_l2_max` are maxima of Euclidean three-vector
errors over retained nodes and bodies. `vector_l2_rms` is the square root of the
mean squared vector norm over node-by-body samples. The reported phase proxy is
the signed angle of the inner-to-outer relative-position vector in the xy plane,
with sign `reference - candidate`; it is not mean longitude or a general
orbital phase. Phase maximum/RMS are taken over retained nodes.

Invariant drift uses the first retained state as its baseline. Absolute center
of mass and momentum are Euclidean norms, while fields explicitly named
`drift` are norms of the difference from that baseline. A small invariant or
energy error does not imply a small state or phase error.

For JX, all three levels must improve monotonically. Only the fine
`P/512 -> P/1024` pair has an order floor: at least 1.5 for position, velocity,
and phase proxy, and at least 1.7 for energy. The finest envelopes are:

| Metric | Maximum |
| --- | ---: |
| Position vector max / RMS | `4e-4` / `8e-5` |
| Velocity vector max / RMS | `1e-4` / `2.5e-5` |
| Relative-vector phase proxy | `3e-4` |
| Relative energy | `5e-8` |
| Relative angular momentum | `2e-14` |
| Absolute center of mass / momentum | `1e-12` / `1e-12` |

The authoritative/sensitivity IAS15 state difference must remain below `5e-12`.
MERCURIUS's fixture-specific order window is checked across all adjacent levels;
TRACE excludes its non-asymptotic `P/128 -> P/256` interval and checks only
`P/256 -> P/512 -> P/1024`. These are regression observations for this fixture,
not method-wide order or accuracy claims.

## Separate limitation witnesses

The optional limitation section is deliberately outside the main success gates:

- An original-node encounter-floor violation is fatal and releases no partial
  hybrid result.
- A locked symmetric two-tracer IVP with `h=0.5` crosses continuously inside a
  distance threshold but has separated endpoints. In this configured witness,
  the MERCURIUS and TRACE runs with `exit_min_distance=0.1` return without an
  exception, while their retained signed relative-y brackets change strictly
  from positive to negative and a separate REBOUND line-collision run detects a
  collision.
  The reported `0.25` epoch is only the unforced straight-line prediction, not
  an exact Newtonian event location. This demonstrates the fixture's tunneling
  limitation; it does not make a general claim about either integrator's checks
  or qualify any collision response.
- A locked high-eccentricity central-periapse case shows that the MERCURIUS
  setting used here does not protect that central-periapse workload. No common
  MERCURIUS success gate is imposed on that out-of-domain witness. Exact final
  `(m,r,x,y,z,vx,vy,vz)` little-endian float64 fingerprints bind the MERCURIUS,
  TRACE, and IAS15 endpoint states independently of the displayed metrics.
- TRACE is measured forward only. Collisions and exit thresholds are disabled
  in the main all-active science lane.
- A separate unequal-mass binary control checks three-step forward/backward,
  sparse/dense all-FAR state, epoch, diagnostic, checksum, and 19-counter
  projections against the frozen public Wisdom--Holman implementation. It is a
  short projection/custody test, not an analytic-accuracy or broad finite-step
  equivalence claim.

## Provenance, replay, and digests

The exact REBOUND version, git hash, build, module, native library, complete
Python-source tree, distribution metadata, license file, import path, and
`OMP_NUM_THREADS` are bound using the existing comparator provenance support.
The report links the official [MERCURIUS documentation](https://rebound.hanno-rein.de/integrators/mercurius/),
[TRACE documentation](https://rebound.hanno-rein.de/integrators/trace/),
[IAS15 documentation](https://rebound.hanno-rein.de/integrators/ias15/),
[5.1.1 tag](https://github.com/hannorein/rebound/tree/5.1.1), and
[license text](https://github.com/hannorein/rebound/blob/5.1.1/LICENSE).
License metadata inconsistencies are reported without making a legal or
redistribution conclusion.

An independent outer semantic replay reconstructs every lane from `Profile` and
compares settings, accounting, state/checkpoint bytes, clocks, limitations, and
controls. Timings are checked only for finite nonnegative shape and are excluded
from the semantic digest. Canonical serialization has exact type, depth, node,
integer-bit, array, string, and cumulative-byte limits before conversion.
Component and report SHA-256 values are domain-separated and bind signed-zero
bits, array dtype/shape/order, lane manifests, ordered lane IDs/digests, and the
content-integrity manifest. They are explicitly **unauthenticated local content
integrity**, never signatures or scientific authority.
