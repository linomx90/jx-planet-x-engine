# JX versus REBOUND: research-backed competition plan

Date: 2026-09-22
Scope: active post-`0.6.0rc6` source
Scientific claim state: `SCREENING_ONLY`

## Decision

JX cannot credibly compete with every REBOUND workload by tuning one solver.
REBOUND is a portfolio: specialized symplectic methods for smooth planetary
systems, IAS15 for high-accuracy adaptive integration and arbitrary forces,
TRACE/MERCURIUS for close encounters, direct and tree gravity for different
body-count regimes, plus collision, archive, variational, and custom-force
facilities. JX needs the same method-dispatch strategy while keeping CPU and
CUDA roles distinct.

The smallest next release remains `0.6.0rc7`. Its scope is the supported fast
CPU Wisdom--Holman path, a generic single-execution postcondition certificate,
checkpoint-synchronized native execution, and an honest benchmark against
optimized WHFast. Native IAS15-class, TRACE-class, and large-N solvers are
later independently gated work; they must not be implied by rc7.

## Benchmark correction

The sealed rc6 report compared JX with REBOUND WHFast configured with
`safe_mode=1`, `keep_unsynchronized=0`, and 401 retained checkpoints. That is a
valid safe-default comparison, but it is not REBOUND's optimized WHFast path.
The official WHFast documentation says `safe_mode=0` improves speed and
accuracy when the caller handles synchronization, and
`keep_unsynchronized=1` synchronizes a copy so continued integration retains
its internal state and bitwise reproducibility.

The rc6 result also needs two timing interpretations. Its end-to-end single-
pass median was `0.009473917 s` versus `0.011029525 s` for safe-default
WHFast. Its native numerical median was `0.009173697 s` versus
`0.008958657 s` for WHFast. The apparent end-to-end JX lead therefore came
from wrapper/evidence construction, not a faster numerical core.

A disclosed post-rc6 development pilot on the same 3-body, 6,400-step,
401-checkpoint workload measured these medians over 15 repetitions:

| Lane | Median seconds | Interpretation |
|---|---:|---|
| JX native v2 | 0.009252416 | rc6 arithmetic/reference kernel |
| JX native v3 | 0.008831262 | checkpoint-synchronized candidate |
| WHFast safe mode, 401 outputs | 0.008638152 | existing safe baseline |
| WHFast optimized, 401 outputs | 0.008240442 | fair tuned retained-output baseline |
| WHFast optimized, endpoint only | 0.001892894 | maximum-throughput output lane |

This pilot is retrospective development evidence, not a preregistered release
result. JX v3 remained bitwise equal to v2 for every retained position,
velocity, and metric. It reduced counted inverse transformations from 12,800
to 6,800 and improved the native median by about 4.55%. Tuned retained-output
WHFast remained about 7.17% faster; endpoint-only WHFast remained about 4.67x
faster. The release benchmark must rerun these lanes prospectively against the
final rc7 artifact.

Linux sampling profiles were unavailable because the host retains the secure
`perf_event_paranoid=4` policy. A temporary `/tmp` build therefore used
selective function instrumentation without changing the production source or
the system policy. Over 100 identical 100-period executions, approximate
inclusive shares of the instrumented v3 loop were:

| Native phase | Approximate inclusive share |
|---|---:|
| Kepler subflows | 55.9% |
| Loop/glue outside selected children | 16.2% |
| Node guards | 14.9% |
| Gravity plus interaction assembly | 6.5% |
| Path guards | 4.4% |
| Position/full inverse transforms | 2.2% |

Function instrumentation changes runtime and is especially sensitive to call
counts, so these are diagnostic proportions rather than release performance
evidence. They are sufficient to reject more coordinate-transform work as the
primary next optimization and to prioritize the Kepler solver. The raw selected
profile is `/tmp/jx_v3_instrumented_profile_selected.txt`, 1,704 bytes,
SHA-256 `87ae999f49bdbf47df5cbdf96572577f5d41996010c7074bd9765e845d89e9a7`.

## Kepler v4 prototype result

The next solver experiment was preregistered before implementation or timing
in `benchmarks/jx_native_kepler_v4_preregistration.json` (SHA-256
`ffb05cf5db4778dec2e6ab8a9a0ab4adac1289d72f4c34e6ddfddae715034d00`).
It is a separate private v4 extension and does not replace supported v3. V4
retains the v2 G functions, bracketing cap, safeguarded Newton/midpoint rule,
acceptance condition, invariant postconditions, and exact v2 fallback. It
changes the outer seed to use the larger magnitude of `h/r` and `h/a`, begins
at the already evaluated bracket endpoint, and reuses that evaluation.

The four focused v4 gates pass. They cover the locked protocol/identity, 480
requested two-body elliptic/direction cases with v3 failure parity and
forward/backward checks, explicit failure domains, deterministic 100-period
execution, and the pre-existing IAS15 accuracy/conservation envelope. Combined
v3/v4 focused coverage currently passes 14 tests.

On the disclosed 15-repetition development timing, v4 reduced the v3 solver
work and latency as follows:

| Quantity | v3 | v4 |
|---|---:|---:|
| Bracket expansions | 6,391 | 3,131 |
| G evaluations | 70,211 | 53,014 |
| Series terms | 1,589,999 | 1,206,231 |
| Solver iterations | 51,020 | 49,883 |
| Native median | 0.008860492 s | 0.008407328 s |

The largest retained v4/v3 Cartesian differences were
`1.7046808409304504e-11` in position and
`8.03063865406628e-12` in velocity; v4 independently passed the IAS15-based
gate. In the same run, safe-default WHFast was `0.008781474 s`, optimized
retained-output WHFast was `0.008249093 s`, and optimized endpoint-only WHFast
was `0.001918193 s`. V4 and optimized WHFast at equal output differed by only
about 1.9%, below the existing 5% resolution threshold. Endpoint-only WHFast
remained about 4.38x faster. This is prototype progress, not release evidence
or a general superiority result.

The subsequent source-bound, 15-repetition recorded development screen passed.
Its protocol is
`benchmarks/jx_native_kepler_v4_rebound_race_protocol.json` (6,029 bytes,
SHA-256 `de3239507c1203bbbf69b6ca1eaceadcfef16cf6740ac0a9ac28a381d59383dc`).
The report is `/tmp/jx_native_kepler_v4_rebound_race_v1.json` (38,370 bytes,
SHA-256 `2efab17598fb4b9b70779b561367c42086e8e46c3199992cffc538da08f7b9bf`,
semantic SHA-256
`aff02e5098541246d98c147f9f2ee1bbeb4dd2a38d73768165c317959bb3ef55`).

| Recorded lane | Numerical call/readback median | End-to-end lane median |
|---|---:|---:|
| JX v3 retained | 0.008810118 s | 0.008880259 s |
| JX v4 retained | 0.008432915 s | 0.008512303 s |
| WHFast safe retained | 0.008796613 s | 0.009486868 s |
| WHFast tuned retained | 0.008212144 s | 0.008794599 s |
| WHFast tuned endpoint | 0.001891603 s | 0.002134785 s |

In the numerical-call/readback scope, tuned retained WHFast was 2.69% faster
than v4. In the end-to-end lane scope, v4 was about 3.3% faster than tuned
retained WHFast. Both differences are smaller than the established 5%
resolution rule and must be classified as unresolved. The endpoint-only lane
was 4.46x faster than v4. Native JX output-buffer allocation is excluded from
the numerical-call scope while REBOUND Python particle readback is included;
the report marks that asymmetry and does not claim raw-C-core equivalence.
Every retained lane passed the same existing IAS15-based envelope.

Fallback use was then made directly observable without changing v4 numerical
arithmetic. A separate v2 lock disclosed the full v1 result before execution
and required zero fallback in every timed v4 repetition. Protocol:
`benchmarks/jx_native_kepler_v4_rebound_race_protocol_v2.json` (6,748 bytes,
SHA-256
`b8a2b021e7ada64a897da49c688c0928738aee4410f1ebd075f5e8cfd7d062d3`).
Report: `/tmp/jx_native_kepler_v4_rebound_race_v2.json` (39,047 bytes,
SHA-256
`a6c64b6d1d7ed45791a30f0d016818bd89050731f840d12864dc44198a032a3a`,
semantic SHA-256
`8ce5c8c916016e5aaecd1e19deeec35e7c9a792dd6a39ad71da3c5b408c799ae`).
It passed with zero fallbacks. V4 was 5.23% faster than v3; tuned retained
WHFast was 1.99% faster than v4, still unresolved by the five-percent rule;
endpoint-only WHFast was 4.42x faster.

Validation after that recorded screen was clean: the combined v3/v4 focused
suite passed 16 tests; the exact test inventory covered 244 files in ten
non-overlapping profiles; and both normal and `PYTHONOPTIMIZE=2` engine
matrices passed 13 isolated lanes / 233 tests. That report remains historical
pre-rc7 development evidence.

## Exact rc7 artifact result

The conservative release decision keeps checkpoint-synchronized v3 as the
supported backend and packages v4 only as a private screening prototype. The
rc7 wheel and source archive were built twice from the declared package-only
boundary and were byte-identical. Fresh wheel and sdist installs produced the
same one-period exact-replay result digest.

The prospective exact-wheel protocol is
`benchmarks/jx_fast_wisdom_holman_rc7_rebound_race_protocol.json` (7,015
bytes, SHA-256
`b701ebe7d3dd3f1463ca7d7f478c1bf6eb8c334c495beba4643f87ef65048098`).
The final report is `/tmp/jx_fast_wh_rc7_rebound_race_v1.json` (32,178 bytes,
SHA-256
`7dec5cd0d247abed56522d0c9e98f8b3d2068496ba7b88f10ab33f1dc2c16fad`,
semantic SHA-256
`fc3ca65f3fc20adf9f592d872e551103bc79f93d397ce0b3f8768e5b4bf32c92`).

| Exact rc7 wheel lane | End-to-end median | Numerical median |
|---|---:|---:|
| JX v3 single pass | 0.009141534 s | 0.008821377 s |
| JX v3 exact replay | 0.018222235 s | 0.017628779 s |
| REBOUND WHFast | 0.010037142 s | 0.008007922 s |

The supported single-pass call had 1.098x WHFast throughput in this exact
end-to-end scope, while WHFast had 1.815x the replay lane's throughput. All
accuracy, conservation, public-JX bitwise-parity, exact-replay, solver,
failure-domain, and deterministic-output gates passed. The result is one-host,
one-workload `SCREENING_ONLY` evidence. It does not contradict the much faster
WHFast endpoint-only lane, the historical long-workload gaps, or the need for
native adaptive and encounter solvers.

The rc7 CUDA core gate passed five tests with zero skips on the RTX 5060 Ti
using CuPy 14.2.0 and CUDA runtime/driver 13.2. Report:
`/tmp/jx-cuda-core-0.6.0rc7-device-v1.json` (813 bytes, SHA-256
`cd5991dd5d6066a7bc411e56df50960989f1b97f2138140bc9f4bab7dea43dff`).
An initial sandboxed preflight could see the GPU through `nvidia-smi` but not
through the CUDA runtime; it ran no tests. The device-visible report is the
authoritative current-host gate. This is core parity/residency validation, not
a new CUDA scaling or multi-machine result.

## Native adaptive Gauss--Radau results

An independent benchmark-private eight-node left Gauss--Radau collocation
prototype now covers the adaptive order-15 lane. It was derived from published
algorithm descriptions without consulting or copying REBOUND implementation
source. It remains `SCREENING_ONLY`, supports only 2--32 fully mutual,
unsoftened Newtonian point masses in binary64, and is not a public backend.

The v1 source-bound report is
`/tmp/jx_native_gauss_radau15_timing_v1.json` (11,870 bytes, SHA-256
`d1138e03a385b6d7f6f4b40172df4113d5c5cec239f262fd1230ba266dee18aa`,
semantic SHA-256
`b0889e9abcd72f74dd8af1fc0727a26f57accf47fc085dad8797e5b581e39ea9`).
All analytic, conservation, eccentric-IAS15, reverse-time, deterministic, and
failure-domain gates passed. The v1 constant-acceleration stage predictor was
the main defect: it required about nine fixed-point corrections per circular
step.

V2 retains the v1 corrector and controller but seeds each safe adjacent step
by analytically integrating the preceding converged seventh-degree
acceleration polynomial. Rejected trials never commit predictor history. The
source-bound V2 report is
`/tmp/jx_native_gauss_radau15_v2_timing_v1.json` (15,725 bytes, SHA-256
`73e4fe10fbe3e3b533014f41d4706f154530e8330944b1762824f40efbe68ef4`,
semantic SHA-256
`284d8850fbd2c5f9c9ff803f20af0b58da4452ea073528fcaac0c269874d2b5c`).

| Accuracy-gated workload | GR15 V2 | GR15 V1 | native RKF78 | REBOUND IAS15 |
|---|---:|---:|---:|---:|
| Circular binary, 100 periods | 0.017961662 s | 0.030848667 s | 0.007897818 s | 0.014422246 s |
| Eccentric binary e=0.9, one period | 0.001152407 s | 0.001680291 s | 0.000224298 s | 0.000668656 s |

V2 is 1.717x and 1.458x faster than v1, satisfying its prospective 1.2x
per-workload gate. It cut circular force evaluations from 321,692 to 133,684
and eccentric force evaluations from 17,113 to 7,946. It remains 1.245x and
1.723x slower than IAS15 on these two workloads. The existing native RKF78
lane is 1.826x and 2.981x faster than IAS15 inside the same broad acceptance
envelopes, but with larger numerical error; that is a bounded one-host result,
not a general solver ranking.

## Reversible encounter-control foundation

Primary TRACE and reversible-switching research favors a stateless symmetric
selection rule over ordinary hysteresis: select from the accepted start node,
inspect the attempted path, and redo the untouched step if the required
integration level increased. The existing public JX hybrid already has
transactional whole-step rollback but explicitly forbids hysteresis, latching,
and prior-mode memory.

The benchmark-private `jx_reversible_encounter_state_machine_prototype.py`
first isolated and tested that binary FAR/NEAR transaction. Its seven locked
gates pass for FAR commit, NEAR-at-start, discard-and-redo from the original
state, reversed forward/backward mode sequence, canonical triggers,
deterministic custody, and fail-closed inputs.

The next binding gate is now implemented in
`jx_native_reversible_hybrid_prototype.py`. Each macrostep uses the native
Wisdom--Holman V2 kernel for a FAR attempt and the native CPU RKF78 kernel for
the complete NEAR interval. The controller independently expresses the
published modified-Hill and PRS23 Eq. 16 mathematical criteria in vector form,
evaluates both endpoints, applies a symmetric swept-pair screen, and discards
every escalated FAR candidate. Native far statuses for the documented orbital,
node-pair, and path guards route to NEAR; numerical, solver, input, and
postcondition failures remain fatal. Ten focused tests pass in normal and
optimized Python.

The three-fixture REBOUND 5.1.1 report is
`/tmp/jx_native_reversible_hybrid_rebound_screen_v1.json` (17,118 bytes,
SHA-256
`e56e24904aec38b8cc772e48fd0d65e2ab845dc602f060bd4ffa8dfedcdca189`,
semantic SHA-256
`5c3a752324e7777d171bf526b2e124ed57fd027c50824c652a0672fdaba0d17b`).
Median times are diagnostic end-to-end Python-call times; errors are final
maximum body-vector errors against REBOUND IAS15 with epsilon `1e-13`.

| Locked fixture | JX position error | TRACE position error | MERCURIUS position error | JX median | TRACE median | MERCURIUS median |
|---|---:|---:|---:|---:|---:|---:|
| all-far binary, half period | 1.278e-14 | 1.001e-7 | 1.001e-7 | 0.024814 s | 0.000148 s | 0.000113 s |
| three-body close scatter, half period | 3.899e-6 | 5.183e-6 | 2.837e-5 | 0.031222 s | 0.000256 s | 0.002607 s |
| central pericenter, one period | 9.979e-12 | 1.338e-5 | 4.756 | 0.011726 s | 0.000214 s | 0.000511 s |

The screen establishes working classifier/rollback/solver composition and
shows useful fixture-specific accuracy. It also exposes the next dominant
defect: JX re-enters Python and recreates native-call buffers at every
macrostep, while TRACE and MERCURIUS retain the whole state machine in C. JX
is therefore roughly 55--168 times slower than TRACE on these lanes despite
the favorable errors. This is not TRACE/MERCURIUS equivalence, collision or
regularization support, a long-horizon result, or a general ranking.

## Persistent native encounter loop result

That Python orchestration defect is now removed in a preregistered,
benchmark-private C11 prototype. The preregistration is
`benchmarks/jx_native_reversible_hybrid_loop_preregistration.json` (SHA-256
`f321047ee28a9a524e9a70ea9ea640a9d98cdf1c16c4ca05ccc0bebed23bee19`).
The implementation keeps the complete macrostep loop, modified-Hill and PRS23
classification, symmetric swept-path test, FAR rollback decision, NEAR redo,
transition ledger, and counters inside one native call. Caller-owned
checkpoint and ledger buffers are reused. The existing WH V2 and RKF78
arithmetic kernels are linked without changing their numerical algorithms.

All three locked fixtures reproduce the former Python controller bit-for-bit
at every position and velocity checkpoint. Their mode sequences, transition
reasons, FAR statuses, classifier counts, and aggregate work counters also
match exactly. Reusing the same workspace and using a fresh workspace produce
the same content digest, invalid workspaces and inputs fail closed, and the
close-scatter forward/backward envelope remains below `2e-10`.

The locked 9-repetition, 2-warmup timing report is
`/tmp/jx_native_reversible_hybrid_loop_timing_v1.json` (22,333 bytes,
SHA-256
`f1557ca4778ebd8aeec5477cd9d21f5d2b608089324bdb4c070e9d60293d3a6c`,
semantic SHA-256 excluding timing
`988e3395e6baeed967accbfc85e8780d89fcc4492e93cc4e0476d74ce87bbf33`).

| Locked fixture | Native C call | Native wrapper | Python controller | TRACE end-to-end | Native/Python speedup | Native/TRACE ratio |
|---|---:|---:|---:|---:|---:|---:|
| all-far binary | 0.000173443 s | 0.000323553 s | 0.024393927 s | 0.000116177 s | 140.65x | 1.49x |
| three-body close scatter | 0.000414291 s | 0.000565173 s | 0.029831462 s | 0.000228225 s | 72.01x | 1.82x |
| central pericenter | 0.000334022 s | 0.000435120 s | 0.011101082 s | 0.000174204 s | 33.23x | 1.92x |

The native/Python columns satisfy the prospective minimum 5x speedup gate on
every fixture and show that the measured 55--168x orchestration defect has
been removed. The native-call column is prevalidated core execution with a
reused workspace, whereas the REBOUND columns construct and read back a
simulation end-to-end. Their ratios are therefore diagnostic separate-scope
observations, not evidence of TRACE timing equivalence. Final errors against
IAS15 are unchanged from the Python controller: `1.278e-14`, `3.899e-6`, and
`9.979e-12` for the three fixtures. The result remains one-host,
short-trajectory, `SCREENING_ONLY` evidence; no public API, release version,
registry row, or superiority claim changed.

## Why REBOUND is difficult to beat

### Smooth hierarchical planetary systems

WHFast keeps persistent Jacobi state, avoids needless coordinate conversions,
combines work across steps when output synchronization permits, and uses a
purpose-built Kepler solver. The WHFast paper emphasizes better initial
guesses, early series termination, stable transformations, and long-run
roundoff behavior. The advanced official tutorial recommends `safe_mode=0`
for suitable immutable systems and explains that correctors are economical
only when many steps occur between outputs.

WHFast512 raises the target again on compatible AVX512 hardware by retaining
state in vector registers across many concatenated steps. It is restricted to
small planetary systems and specific hardware, so it belongs in a separate
capability lane rather than the portable WHFast baseline.

### High-accuracy and arbitrary-force integration

IAS15 is a 15th-order adaptive Gauss--Radau integrator designed for conservative
and non-conservative forces, high eccentricity, and close encounters. Its
current PRS23 timestep criterion uses acceleration, jerk, and snap to recover
orbital/pericenter timescales robustly. Optimizing JX's fixed-step CUDA RKF78
cannot substitute for this algorithmic advantage on a single long trajectory.

### Close encounters

MERCURIUS uses WHFast away from encounters and IAS15 within a smooth encounter
switch. TRACE uses WHFast away from encounters and Bulirsch--Stoer or IAS15 in
the encounter region, including a central-pericenter switch. JX's current
Python hybrid orchestration and replay cannot be competitive with those fully
native paths.

### Large body counts and feature breadth

REBOUND provides direct and tree gravity, OpenMP, limited MPI tree use,
collisions, test particles, variational equations, simulation archives,
arbitrary ODEs, and custom forces. A claim about competing "on everything"
requires capability and accuracy parity in addition to runtime.

## JX implementation portfolio

| Workload | JX target | Hardware | Comparator |
|---|---|---|---|
| Smooth, dominant central mass | persistent native WH v3/v4 | CPU | WHFast and SABA |
| Very small SIMD-compatible systems | vectorized multi-system WH | CPU SIMD | WHFast512 |
| High accuracy, arbitrary forces | native 15th-order adaptive Gauss--Radau-class solver | CPU first | IAS15 |
| Occasional close encounters | reversible native switch plus BS/adaptive tail | CPU | TRACE, then MERCURIUS |
| One large self-gravitating system | direct tiled and tree/FMM gravity | CPU/CUDA | REBOUND direct/tree |
| Many independent systems | fused batched integration | CUDA | parallel REBOUND workers |

CUDA remains JX's strongest current advantage for independent ensembles and
larger same-step workloads. CPU remains the right target for one small,
dependency-bound trajectory. Two-GPU reproduction on the owned RTX 5060 Ti
and RTX 4050 is valuable portability evidence, but it is not independent
replication.

## Immediate engineering order

1. Finish v3 release correctness: irregular checkpoints, backward time,
   two-body sentinel handling, failure domains, exact replay, source custody,
   packaging, and final-artifact tests.
2. Build an optimized WHFast comparator with equal retained-output cadence,
   `safe_mode=0`, `keep_unsynchronized=1`, explicit copied-state
   synchronization, and a separate endpoint-only lane. Keep the old safe lane.
3. Treat the passing v4 Kepler experiment as a prototype until a final-artifact
   rerun reproduces the now source-bound accuracy and timing result.
4. Use the completed phase profile for the next optimization. Full every-step
   certification is now the largest architectural difference from endpoint-
   only WHFast; expose separate fully certified and explicitly reduced-check
   numerical-core scopes without silently weakening the default.
5. Expose two honest timing scopes: numerical core and certified supported
   call. Never compare JX certification overhead with REBOUND's raw core
   without labeling the difference.
6. Begin a separate native adaptive high-order design for the IAS15 workload;
   do not force the WH kernel or CUDA RKF78 to cover it. V1 and predictor V2
   now satisfy their private correctness and performance gates; neither is a
   supported backend.
7. Preserve the completed persistent native state-machine result as a private
   screening lane; it removes the measured Python macrostep bottleneck with
   bitwise controller parity.
8. Make the FAR path itself stateful across macrosteps, package the native loop
   behind a supported extension boundary, and then run equal-scope raw-C and
   supported-call comparisons plus long-horizon encounter and failure-domain
   gates before considering promotion.
9. Add direct/tree gravity and feature-parity tracks if large-N and full API
   competition remain release objectives.

## Release and claim gates

- Pinned REBOUND 5.1.1 remains the reproducible baseline; a current exact
  commit is an additional moving-target lane.
- Equal initial bytes, force model, horizon, requested accuracy, retained
  output cadence, thread/process budget, and setup inclusion are required for
  a fair timing classification.
- Endpoint-only, retained-output, single-trajectory latency, and ensemble
  throughput are separate results.
- The final artifact—not an editable checkout—must pass the benchmark.
- Accuracy, conservation, failure-domain, deterministic-replay, and package
  reproducibility gates precede performance classification.
- No result authorizes a general claim that JX defeats REBOUND.
- JX's proprietary source must not copy or derive implementation code from
  REBOUND's GPL source. Papers and public algorithm descriptions may guide an
  independently implemented design; REBOUND remains an external comparator.

## Primary sources

- [REBOUND repository and feature architecture](https://github.com/hannorein/rebound)
- [Official WHFast documentation](https://rebound.hanno-rein.de/integrators/whfast/)
- [Official advanced WHFast tutorial](https://rebound.hanno-rein.de/ipython_examples/AdvWHFast/)
- [WHFast algorithm paper](https://arxiv.org/abs/1506.01084)
- [WHFast512 paper](https://arxiv.org/abs/2307.05683)
- [Current WHFast512 implementation](https://github.com/hannorein/rebound/blob/main/src/integrator_whfast512.c)
- [Official IAS15 documentation](https://rebound.hanno-rein.de/integrators/ias15/)
- [IAS15 paper](https://arxiv.org/abs/1409.4779)
- [PRS23 timestep criterion](https://arxiv.org/abs/2401.02849)
- [Official MERCURIUS documentation](https://rebound.hanno-rein.de/integrators/mercurius/)
- [MERCURIUS paper](https://arxiv.org/abs/1903.04972)
- [Official TRACE documentation](https://rebound.hanno-rein.de/integrators/trace/)
- [TRACE paper](https://arxiv.org/abs/2405.03800)
- [REBOUND 5.1.1 TRACE source used only as an external formula/readback cross-check](https://github.com/hannorein/rebound/blob/5.1.1/src/integrator_trace.c)
- [Reversible integrator switching](https://arxiv.org/abs/2301.06253)
- [Multiple-timestep reversible close-encounter methods](https://arxiv.org/abs/2401.07113)
- [REBOUND gravity choices in the public header](https://github.com/hannorein/rebound/blob/main/src/rebound.h)
