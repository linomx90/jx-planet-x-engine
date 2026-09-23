# Why REBOUND is faster than JX

Date: 2026-09-22
Scope: current JX post-`0.6.0rc5` source and the recorded same-host REBOUND
5.1.1 comparisons
Claim state: `SCREENING_ONLY`

## Conclusion

REBOUND is faster on the recorded long, individual-trajectory workloads
primarily because its entire numerical step loop is a specialized, mutable C
runtime. JX currently runs most Wisdom--Holman orchestration in Python and does
far more than advance the numerical state: on every step it validates private
capabilities, validates force custody, constructs immutable candidates and
caches, copies arrays, evaluates several physical guards, serializes retained
content, and computes multiple SHA-256 commitments. A public JX call then runs
the complete trajectory a second time as a mandatory semantic replay.

This is not mainly a difference in integrator order. On the smooth-hierarchy
race, both candidates use a second-order Wisdom--Holman/Jacobi map. It is also
not a float-versus-arbitrary-precision comparison: this JX lane is binary64.

The post-rc5 native universal-G/fused-commit prototype is real progress, but
it only moves one small kernel into C. It improves the 100-period JX median
from `39.7208881 s` to `23.8857250 s` (`1.66296x`) while preserving the exact
public result. Compared only as scale with the previously recorded WHFast
median of `0.0120271540 s`, the remaining inferred gap is about `1,985.98x`.
That last ratio is a cross-report inference, not a new paired race.

## What was compared

The prospective rc5 timing report used one process and `OMP_NUM_THREADS=1`.
The smooth-hierarchy workload had three active bodies, 6,400 fixed steps over
100 inner periods, and 401 retained host checkpoints. Both candidates passed
the predeclared accuracy envelope.

| Candidate | Median | Relative result |
|---|---:|---:|
| JX rc5 public Wisdom--Holman | 40.0737703 s | WHFast 3,331.94x faster |
| REBOUND 5.1.1 WHFast | 0.0120271540 s | reference |
| JX post-rc5 native/fused prototype | 23.8857250 s | inferred WHFast gap 1,985.98x |

The REBOUND lane is not using its fastest risky settings. The benchmark fixes
Jacobi coordinates, the default second-order kernel, no corrector,
`safe_mode=1`, and `keep_unsynchronized=0`. It calls
`Simulation.steps(integer_delta)` and checks synchronized state at every
retained checkpoint. REBOUND's own documentation says `safe_mode=0` can
improve speed and accuracy when the caller manages synchronization, so a
best-tuned WHFast race could widen rather than close the recorded gap.

Accuracy was not traded away to obtain the WHFast timing. Relative to the
same IAS15 reference, JX and WHFast both passed all locked gates. On this one
fixture WHFast also had the smaller maximum phase proxy
(`1.92934e-4 rad` versus `3.62666e-4 rad`) and smaller maximum relative-energy
error (`1.06925e-6` versus `2.15341e-6`). These observations do not establish
general superiority.

## Measured root causes in JX

A fresh `cProfile` run used the post-rc5 native/fused prototype for a reduced,
structurally identical two-period workload: 128 requested outer steps and 256
executed map steps after the mandatory replay. It recorded 1,616,167 Python
calls in 0.895 s. The important cumulative costs were:

| Operation | Calls | Cumulative time |
|---|---:|---:|
| Complete private map execution | 2 | 0.903 s |
| Per-step proposal path | 256 | 0.836 s |
| Public result validation, including replay | 1 | 0.455 s |
| Domain-separated JSON SHA-256 | 1,838 | 0.286 s |
| Recursive canonical-content conversion | 106,246 recursive calls | 0.221 s |
| Continuous-cache validation | 258 | 0.188 s |
| Force-custody validation | 260 | 0.125 s |
| Elliptic universal Kepler solve | 512 | 0.110 s |
| Generic force-plan evaluation | 258 | 0.110 s |

These are cumulative profile times and overlap; they must not be added. They
do establish that the numerical Kepler solve is not the dominant cost after
the first native optimization. Hashing, canonicalization, validation, Python
object construction, and the replay dominate the public operation.

The profile also exposes the scale of the orchestration: approximately 6,313
Python calls per executed map step, about 7.2 domain-separated hashes per map
step, one full cache validation per map step, and roughly two cache-content
hashes per map step. The profile artifacts are:

- `/tmp/jx_wh_profile_research.py`, 573 bytes, SHA-256
  `406d0812d78fa73912d68d3902a980f1f8d8bb2f7ce8752398572eb9d11667d2`;
- `/tmp/jx_wh_profile_research.pstats`, 54,813 bytes, SHA-256
  `df21e27be7158f619176c3e23ec167f9e46df0438162172979e42ee65b0a1551`.

### 1. REBOUND keeps the hot loop in C

REBOUND states that its computationally intensive code is written in C99.
In the exact 5.1.1 source, `reb_simulation_steps()` loops over steps in C and
calls the integrator callback without returning to Python. WHFast mutates a
preallocated Jacobi particle buffer and the simulation particle array in
place. The Python benchmark crosses the language boundary only once per
checkpoint span, not once per step or subflow.

JX's current C extension implements only the four fixed universal-G series.
Every macrostep still returns to Python for kicks, Kepler-control logic,
coordinate transforms, force-plan evaluation, guard evidence, custody,
candidate construction, commit, accounting, and hashes.

### 2. WHFast is a highly optimized specialist

WHFast is explicitly designed for a dominant central object with small
perturbations. Its authors report that speed comes from an improved Kepler
solver, efficient c/G functions, combining adjacent drift operations when
synchronization permits, and avoiding unnecessary Jacobi transformations.
The 5.1.1 source uses compact Stumpff-function recurrences, inverse-factorial
lookup tables, a fast Newton path, a higher-order difficult-case path, and a
bisection fallback.

The exact-result gate forced JX's native prototype to preserve its Python
algorithm and operation order, including a fixed 64-term series for each of
four G functions. That was the right gate for proving a safe mechanical
optimization, but it also prevents a more efficient Kepler algorithm from
showing its full potential. A new numerical algorithm needs accuracy,
convergence, conservation, and long-horizon gates; it cannot honestly claim
bitwise identity to the old algorithm.

### 3. JX performs certification work inside the timed integration

JX intentionally treats each step as a checked transaction. The current
public path includes:

- a full source-cache and force-custody validation;
- two node guards and one curved-path/encounter guard per step;
- immutable, owned, read-only array copies for candidates and committed
  caches;
- retained work/accounting records and guard extrema;
- candidate, cache, force-context, schedule, and result content hashes; and
- a full second execution followed by exact semantic comparison.

REBOUND's timed numerical API does not promise or perform that JX-specific
custody protocol. This makes the existing timing honest for public-call
latency, but it is not a pure numerical-kernel comparison.

Replay alone cannot explain the result. The prototype averages
`3,732.14 microseconds` per requested outer step including two executions, or
about `1,866.07 microseconds` per executed map step. WHFast averages about
`1.87924 microseconds` per requested step. Even a hypothetical free removal
of replay would leave an approximately `993x` inferred gap.

### 4. Allocation and serialization are expensive at three bodies

At three bodies the arithmetic is tiny. Rebuilding Python dataclasses,
walking their fields, converting hundreds of binary64 values with
`float.hex()`, JSON encoding, SHA-256 hashing, NumPy shape/finite checks, and
copying five small arrays can cost much more than the actual force and Kepler
math. REBOUND mutates compact C structs instead.

### 5. The generic JX force boundary costs more than a specialized map

JX evaluates the public force plan and retains its ledger, then constructs the
interaction force and validates translation residuals. REBOUND's WHFast path
directly configures the gravity exclusions implied by its Hamiltonian split
and calls the specialized C routines. Both approaches are valid, but the JX
abstraction boundary is currently inside the hot loop.

## Why CUDA does not solve this small-trajectory problem

A single three- or eleven-body trajectory exposes little parallel work within
one time step, and each next step depends on the previous state. A GPU is most
effective when JX can batch many independent systems or many bodies. The rc5
results reflect that distinction:

- for one 100-year, eleven-body trajectory, JX fused CUDA RKF78 took
  `44.1870 s` and REBOUND IAS15 took `1.87047 s` (`23.62x` faster);
- for 256 independent short systems, JX fused CUDA throughput was `2.013x`,
  `3.557x`, and `3.604x` the eight-worker REBOUND integration critical path
  for 2, 11, and 32 bodies respectively.

IAS15 is a 15th-order adaptive Gauss--Radau method with automatic timestep
selection. The recorded JX CUDA trajectory executed 584,400 accepted fixed
steps. Launch fusion reduces GPU launch overhead, but it does not remove the
large fixed-step work count or create enough same-step parallelism at eleven
bodies. Therefore CPU-native execution is the correct priority for small,
single-system latency; CUDA remains valuable for ensemble throughput.

The same specialization issue appears in close encounters. MERCURIUS uses a
Wisdom--Holman method away from encounters and switches to IAS15 during them;
TRACE is a purpose-built time-reversible close-encounter method. JX's current
hybrid path additionally constructs witnesses, performs Python custody work,
and replays/validates its decisions. The recorded factors of `2,650x` versus
MERCURIUS and `18,820x` versus TRACE are workload-specific, but they point to
the same architectural bottleneck.

## What is not causing the WHFast gap

- **Not decimal or arbitrary precision:** the compared JX WH lane is NumPy
  binary64.
- **Not REBOUND parallel workers:** the race used one process and one OpenMP
  thread for each single trajectory.
- **Not mismatched retained output cadence:** both lanes retained 401 host
  checkpoints.
- **Not REBOUND's unsafe mode:** the recorded lane used `safe_mode=1`.
- **Not merely the mandatory replay:** removing a factor of two cannot erase a
  roughly three-order-of-magnitude per-execution gap.
- **Not evidence that REBOUND is always faster:** JX already wins the separate
  batched-CUDA throughput workload. Latency, throughput, body count, method,
  and output contract must be reported separately.

## Recommended implementation order

1. **Build one complete native CPU WH map loop.** Keep positions, velocities,
   Jacobi state, forces, counters, and guard extrema in a reusable C workspace.
   Cross Python only for setup, retained checkpoints, terminal failures, and
   final result construction. This is the only current change with plausible
   `100x-1000x` leverage.
2. **Keep guards native, but remove serialization from each numerical step.**
   Native code can evaluate finite/domain/encounter guards and return compact
   typed failure records. Hash the final retained evidence once. Do not use
   SHA-256 as a substitute for numerical validation.
3. **Version the execution contract.** Expose a clearly named numerical-core
   timing lane and a separate certified/replayed public lane. Report both.
   This preserves JX's evidence strength without comparing certification
   overhead to REBOUND's numerical kernel as if they were identical services.
4. **Create a new Kepler-solver v2 under prospective accuracy gates.** Test an
   optimized Stumpff recurrence, fast Newton convergence checks, and robust
   difficult-case fallback. Require convergence sweeps and long-horizon
   conservation/phase gates, not bitwise equality to the 64-term reference.
5. **Run a new paired race only after the native loop exists.** Measure
   standard safe WHFast exactly as now, then add a separately labelled tuned
   `safe_mode=0` lane. Include setup, integration, checkpoint materialization,
   and certification as separate timing columns.
6. **Use CUDA where it has enough width.** Continue ensemble and many-body
   throughput work; do not make a small-N, single-trajectory GPU win a release
   requirement.

The immediate engineering target is therefore not another Python
micro-optimization and not another GPU kernel. It is a complete native CPU
Wisdom--Holman loop with native guard evaluation and one boundary crossing per
checkpoint span.

## Complete native-loop result (2026-09-22)

That immediate target is now implemented as a benchmark-private prototype.
The complete fixed-step loop, Jacobi transforms, mutual gravity, interaction
force assembly, elliptic Kepler solves, physical guards, accounting, and
checkpoint writes execute in C. Python performs the validated boundary,
retained-output custody, and optional second complete native replay. The public
backend is unchanged and the claim ceiling remains `SCREENING_ONLY`.

The strongest parity gate passed: all 401 retained positions, velocities, and
epochs over 100 periods are bitwise identical to the existing public JX map.
An independent native replay is also bitwise exact. The paired, source-locked
three-repetition medians on `Galaxy-PC` are:

| 100-period, 6,400-step lane | End-to-end median |
|---|---:|
| JX complete native map, one pass | `0.079649746 s` |
| JX complete native map plus one replay | `0.159790266 s` |
| REBOUND 5.1.1 WHFast, safe mode | `0.011262349 s` |

Both JX and WHFast passed the same pre-existing IAS15-based descriptive
envelope. Relative to the fresh `39.830334 s` public-JX parity execution, the
new lane is about `500.07x` faster raw and `249.27x` faster with replay. The
paired WHFast gap is now `7.07x` raw or `14.19x` with replay, down from the
historical public-path `3,331.94x` gap. This is major implementation progress,
but not a REBOUND win.

The remaining raw gap is now inside native numerical work rather than Python
per-step orchestration. The next prospective optimization should therefore be
a Kepler-solver v2 and leaner native guard/transform implementation under the
same long-horizon accuracy and failure-domain gates. CUDA remains the correct
lane for wide ensembles and many-body throughput, not this dependency-bound
three-body trajectory. The project has separate RTX 5060 and RTX 4050
hardware; neither GPU participates in this CPU protocol.

Locked protocol:
`benchmarks/jx_native_wisdom_holman_rebound_race_v1_protocol.json`, 5,652
bytes, SHA-256
`a9d08b1014d9713d881785f205878020dbaba4086df6dcaa2419b4d6b19a6759`.
Report: `/tmp/jx_native_wh_rebound_race_v1.json`, 30,138 bytes, SHA-256
`2beb7944f9e885473fb3a23cc840d4fcc79ca789028a0afe4064b9f62ad4f290`,
semantic SHA-256
`0d2d50c3edd84056eeb615246242996722cc0d6c4dd315fceb1742f98a9c9769`.

## Primary external sources

- REBOUND 5.1.1 repository and implementation statement:
  <https://github.com/hannorein/rebound/tree/5.1.1>
- REBOUND 5.1.1 WHFast C source:
  <https://raw.githubusercontent.com/hannorein/rebound/5.1.1/src/integrator_whfast.c>
- REBOUND 5.1.1 simulation step loop:
  <https://raw.githubusercontent.com/hannorein/rebound/5.1.1/src/simulation.c>
- Official WHFast settings documentation:
  <https://rebound.hanno-rein.de/integrators/whfast/>
- Official advanced-WHFast performance tutorial:
  <https://rebound.hanno-rein.de/ipython_examples/AdvWHFast/>
- Rein and Tamayo, WHFast implementation paper:
  <https://doi.org/10.1093/mnras/stv1257>
- Rein and Spiegel, IAS15 paper:
  <https://arxiv.org/abs/1409.4779>
- Rein et al., hybrid symplectic integrators/MERCURIUS:
  <https://arxiv.org/abs/1903.04972>
- Lu, Hernandez, and Rein, TRACE:
  <https://arxiv.org/abs/2405.03800>

## Local evidence

- `benchmarks/rebound_whfast_comparison.py`
- `src/jxplanetx/engine/wisdom_holman.py`
- `src/jxplanetx/_wisdom_holman_cpu_module.c`
- `src/jxplanetx/_wisdom_holman_loop.c`
- `benchmarks/jx_native_wisdom_holman_prototype.py`
- `benchmarks/jx_native_wisdom_holman_long_v1_protocol.json`
- `benchmarks/jx_native_wisdom_holman_loop_prototype.py`
- `benchmarks/jx_native_wisdom_holman_rebound_race.py`
- `benchmarks/jx_native_wisdom_holman_rebound_race_v1_protocol.json`
- `/tmp/jx_rebound_long_timing_race_rc5_v2.json`
- `/tmp/jx_native_wh_long_100p_v1.json`
- `/tmp/jx_native_wh_rebound_race_v1.json`
