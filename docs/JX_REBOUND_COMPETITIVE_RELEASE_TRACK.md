# JX versus REBOUND release track

## Claim boundary

This track measures specific workloads. It does not authorize a general
statement that JX is faster or more accurate than REBOUND. Every result remains
`SCREENING_ONLY`; REBOUND remains an independent external comparator.

## Package promotion

The fused CUDA RKF78 implementation is now part of the installed `jxplanetx`
package rather than existing only below `benchmarks/`:

- `jxplanetx.fused_cuda_rkf78` owns the fused 13-stage step;
- `jxplanetx.fused_cuda_adaptive_rkf78` owns the persistent adaptive controller;
- `jxplanetx.fused_cuda_trajectory` binds the restricted CUDA path to the public
  trajectory contracts.

The old benchmark module names are compatibility entry points. The promoted
path is still experimental, supports only 2--32 fully massive bodies with
unsoftened Newtonian gravity, and is not a generally qualified CUDA backend.

Two clean release builds produced byte-identical `0.6.0rc4` artifacts:

| Artifact | SHA-256 |
|---|---|
| `jxplanetx-0.6.0rc4.tar.gz` | `75e80143f45d3d2897fad75363b3b213d4ea1669f937d4a2e9407eabed32a1f8` |
| `jxplanetx-0.6.0rc4-cp314-cp314-linux_x86_64.whl` | `53258c18a70c7eda765e52b8a95f10dcd3483c670cce1631987c5262a81456b5` |

The wheel was installed into a clean temporary environment. Its public
`jxplanetx.smalln` interface passed compiled-CPU forward, rejection, backward,
ledger, and workspace-reuse checks, and its CUDA route passed on the recorded
RTX 5060 Ti. The optional `rebound-competition` extra pins REBOUND 5.1.1
separately from the legacy REBOUND 4.4.11 extra.

## Accuracy-gated rc4 matrix with parallel REBOUND

The v8 matrix records JX `0.6.0rc4`, exact source hashes, a nonsecret machine
label, CUDA identity, REBOUND 5.1.1 identity, and two eight-worker REBOUND timing
views. The integration critical path excludes process startup and simulation
construction. The end-to-end view includes dispatch, per-repetition simulation
construction, checkpoint extraction, and result transport.

All candidates passed the same `1e-10` maximum componentwise checkpoint-error
gate against strict JX and REBOUND references. On the labeled RTX 5060 Ti host:

| Bodies | JX / sequential REBOUND throughput | JX / 8-worker REBOUND integration throughput | JX / 8-worker REBOUND end-to-end throughput | JX / REBOUND single latency |
|---:|---:|---:|---:|---:|
| 2 | 10.14x | 2.02x | 12.36x | 5.17x slower |
| 11 | 17.63x | 3.43x | 7.46x | 2.05x slower |
| 32 | 19.61x | 3.82x | 4.76x | 1.068x slower |

The create-new report is
`/tmp/jx_fused_rebound_matrix_rc4_v8_rtx5060ti.json`, SHA-256
`5bb1fc6566d9d7dd5b76279da414f4331e321b3a4e0f5d7b6d988ca20f32fbc4`.
Temporary output is execution evidence, not a committed release artifact.

The same locked workload then passed on the RTX 4050 Laptop GPU:

| Bodies | JX / sequential REBOUND throughput | JX / 8-worker REBOUND integration throughput | JX / 8-worker REBOUND end-to-end throughput | JX / REBOUND single latency |
|---:|---:|---:|---:|---:|
| 2 | 4.17x | 1.023x | 13.11x | 8.83x slower |
| 11 | 4.58x | 1.56x | 2.85x | 4.33x slower |
| 32 | 5.00x | 1.27x | 1.76x | 2.50x slower |

The laptop report is
`/tmp/jx_fused_rebound_matrix_rc4_v8_rtx4050.json`, SHA-256
`34b3b7d98bc3c5ef564d70ad1618618ff8e86b4bca373cdea7cd165d89444c40`.
The 2-body advantage over the eight-worker integration critical path is only
about 2.3%, so it is a narrow observed result rather than a robust margin.

## Long-horizon and solver-portfolio coverage

The exact final rc5 wheel reran the package-owned long-horizon screen using the
promoted adaptive fused CUDA RKF78 core and the read-only DE440-derived
eleven-body state. Earth and Moon are separately resolved, and exact REBOUND
5.1.1 IAS15 supplies the checkpoint comparator at 10, 30, and 100 years. Every
predeclared screening gate passed. At 100 years, the maximum body position and
velocity differences were `1.9113140004626685 m` and
`1.4308693188075322e-6 m/s`; the Earth--Moon relative differences were
`0.4761816084612515 m` and `1.323299230858037e-6 m/s`. JX accepted 584,400
steps with zero rejects in 143 persistent-kernel launches. The report is
`/tmp/jx_fused_package_solar_lunar_100yr_rc5.json`, 6,271 bytes, SHA-256
`f39a919ff1f4d3561d7c8c99c2da1b1afd10224a410bd99f73af7941b3cc3cd6`.
Its raw timing diagnostics are explicitly non-comparable; they do not support
a long-horizon speed claim.

This long-horizon workload is mutual Newtonian point-mass dynamics. It does
not exercise the coupled lunar mantle rotation, fluid core, tides, relativity,
or figure forces and therefore is not an ephemeris or lunar-physics
qualification.

Challenger v2 was also rerun against current rc5 source. It binds the active
20-file engine roster, leaves frozen v1 unchanged, and executes each existing
self-validating comparator in a separate CPython process. Its full profile
passed the 100-period equal-binary Leapfrog workload, the 10/100-period
weak-hierarchy WHFast/IAS15 workload, and the full calibrated close-scatter
MERCURIUS/TRACE/IAS15 ladders including limitation witnesses. The aggregate is
`/tmp/jx_challenger_v2_rc5_full/challenger_v2.json`, 8,232 bytes, SHA-256
`97bf6b1691d8ff1e7c6e2e17e4e1f797f2f00c60ea895a11368b1d38ec40291a`,
with semantic-content SHA-256
`4aaddd1bb246b4f0196bf9db6a0ef869786bb4438f71d56065338bedefa510ee`.
It ranks no methods, treats child wall times as non-comparable diagnostics, and
authorizes no performance-superiority claim.

The current lunar coupled-eleven roundoff holdout remains a separate 1-, 7-,
and 30-day screening result. It does not compare with REBOUND and does not
extend the lunar physical-validation horizon beyond 30 days.

## Multi-machine protocol

Run the v8 matrix on each machine with the same arguments and a unique,
nonsecret label:

```bash
PYTHONPATH=src:. python benchmarks/jx_fused_accuracy_matched_rebound.py \
  --body-counts 2,11,32 --batch-lanes 256 --horizon 1 \
  --target 1e-10 --controls 1e-5,1e-8,1e-11 \
  --strict-control 1e-13 --reference-limit 1e-11 \
  --repetitions 3 --rebound-workers 8 \
  --machine-label HOST-LABEL --output REPORT.json
```

Aggregate reports only after copying them to one review host:

```bash
python tools/aggregate_fused_rebound_reports.py \
  host-a.json host-b.json --output multi-machine.json
```

The aggregator fails closed on duplicate labels, failed rows, source drift,
version drift, or workload drift. Even a passing aggregate does not create a
portable speed ranking; it reports a finite hardware sample.

A deterministic corrected rc4 remote kit is available at
`/tmp/jx-fused-rebound-remote-0.6.0rc4-v3-a.tar.gz`, SHA-256
`7c97f28a407741bb25335738a563d2cc770c12fed0e71ff436602ca841ab5e79`.
The initial v2 kit failed before timing because it omitted one benchmark import;
v3 includes that dependency as a checksummed member and passed isolated import
checks on both machines. See
`docs/JX_REMOTE_CUDA_REBOUND_RUN.md` for the second-host procedure. Two builds
were byte-identical and the final hash is recorded in the rc4 release
manifest. Packaging does not establish machine independence. Because the
laptop remains under the same ownership, its successful run expands the
hardware sample but is not an independent replication.

The fail-closed aggregate is
`/tmp/jx-fused-rebound-multimachine-rc4.json`, 4,620 bytes, SHA-256
`14e0e1bd3df5f8b1c33716e0296f133c81e569cdf41ddc5b69217c353375883f`.
Its status is `PASS_MULTI_MACHINE_SCREENING_ONLY`; it explicitly keeps portable
ranking, comprehensive hardware sampling, production, and scientific
qualification false.

## Rc5 supported-CPU latency and crossover screen

Rc5 changes no native arithmetic. It caches only immutable public-wrapper
invariants: the RKF78 tableau, exact checkpoint epochs, and read-only
checkpoint views in a frozen workspace. The accuracy-matched hybrid matrix now
calls the installed supported `jxplanetx.smalln` API rather than the historical
CPU prototype and binds a nonsecret machine label.

On the recorded RTX 5060 Ti desktop, every sampled public-CPU row passed the
same `1e-10` checkpoint-error gate. For one 2-, 11-, and 32-body system, public
CPU latency was lower than REBOUND IAS15; at 256 independent systems fused
CUDA retained higher JX throughput. The full discrete 2--32-body and
1--256-lane screen passed. Its five-repetition 29--31-body/eight-lane rows
flipped relative to focused runs with small margins, so they are explicitly
treated as timing-noise-sensitive rather than stable routing evidence. A
separate 101-repetition boundary audit found CPU faster at eight lanes for
bodies 28--31 and CUDA faster from 16 lanes. The 32-body/eight-lane median was
effectively tied, with CUDA ahead by about 1.27%, so the dispatcher preserves
CUDA for that measured point and makes no interpolation or portability claim.

The exact final-wheel reports are:

| Report | Bytes | SHA-256 |
|---|---:|---|
| `/tmp/jx_smalln_hybrid_matrix_rc5_final_v2.json` | 31,557 | `88ddce1554e8f7fbc0f8828cd457a01a6688ab1327e1775fdd0850681c1fa17c` |
| `/tmp/jx_smalln_hybrid_crossover_rc5_final_v2.json` | 248,792 | `37df8cce436c012911738973613dd20372f770d7039dd0bf15b8f5de780248bc` |
| `/tmp/jx_smalln_hybrid_crossover_rc5_boundary101_a.json` | 11,969 | `2e8a93e61c6f79a81e3135c9552e16454bf750db8337ba4defd837f8dcb79ca8` |

All three bind JX `0.6.0rc5`, the labeled runtime, native-core identity, and
the exact public wrapper and benchmark source hashes. They are temporary local
execution evidence rather than committed scientific artifacts.

## Rc5 eight-worker fair race

The exact final rc5 wheel was rerun at five timed repetitions against REBOUND
5.1.1 IAS15 using eight process workers. All methods passed the shared `1e-10`
maximum componentwise checkpoint-error gate for 256 independent short
synthetic systems. The integration-critical-path view excludes process startup
and simulation construction; the end-to-end view includes dispatch,
construction, checkpoint extraction, and result transport.

| Bodies | JX systems/s | REBOUND 8-worker integration systems/s | JX / 8-worker integration | JX / 8-worker end-to-end |
|---:|---:|---:|---:|---:|
| 2 | 132,715.86 | 65,942.18 | 2.013x | 12.901x |
| 11 | 44,258.31 | 12,443.90 | 3.557x | 5.347x |
| 32 | 10,678.28 | 2,962.85 | 3.604x | 4.458x |

The report is `/tmp/jx_fused_rebound_matrix_rc5_v8_rtx5060ti.json`, 25,248
bytes, SHA-256
`e18ea8fec04410ab9584cec7000a5f77068fcd3958c96b6dda5d6a3d1a34c0dc`.
JX wins these recorded batch-throughput rows, while sequential REBOUND IAS15
still has lower single-system latency than one-lane fused CUDA. This is a
finite workload result on one machine, not a general JX-versus-REBOUND rank.

## Prospectively locked long single-trajectory race

A new timing protocol fixes the source roster, workloads, output cadence,
accuracy gates, thread count, warmups, three-repetition order, timing scope,
median statistic, and five-percent classification rule before the measured
execution. It honestly discloses that the earlier screening accuracy and
diagnostic timings were already known; the new timing sample is prospective,
but the scientific workloads were not unseen.

The first protocol stopped in preflight because the runner used incorrect
field names for the already-valid REBOUND provenance object. It created no
output and ran no measured lane. The unchanged v1 protocol is retained at
`benchmarks/jx_rebound_long_timing_race_v1_protocol.json`, SHA-256
`a137ce85d676b53c066ea31e5b80c8cb2307f27b4d41a19d247f63acfae479b0`.
The v2 overlay binds that stopped protocol and the field-name-only runner
correction; its SHA-256 is
`5d43f9ade52cabfe6ca99f33c80d0d046927945fe6f53d465f3264a4fbcc047c`.

Every candidate passed its predeclared accuracy envelope. Median results on the
recorded AMD Ryzen 5 5500 / RTX 5060 Ti desktop were:

| Intended workload | JX median | REBOUND median | Recorded classification |
|---|---:|---:|---|
| 100-year eleven-body point-mass: fused CUDA RKF78 / IAS15 | 44.1870 s | 1.87047 s | IAS15 23.62x faster |
| 100-period smooth hierarchy: JX Wisdom--Holman / WHFast | 40.0738 s | 0.0120272 s | WHFast 3,331.94x faster |
| Close scatter: JX hybrid / MERCURIUS | 110.393 s | 0.0416548 s | MERCURIUS 2,650.18x faster |
| Close scatter: JX hybrid / TRACE | 110.393 s | 0.00586582 s | TRACE 18,819.64x faster |

The report is `/tmp/jx_rebound_long_timing_race_rc5_v2.json`, 43,660 bytes,
SHA-256
`dbd7f46b0cecc95b5f81a69b57e201855fbe8165b3adb3e83b072d2ae2531eba`,
with semantic-content SHA-256
`cb104eb7f8e161fce1a80d080045a9bb5de000dec95f3722a4f74faabdda6bdd`.

These timings include JX's mandatory public semantic replay and equal retained
host-output cadence while excluding candidate setup and IAS15 reference
analysis. REBOUND uses one process because each lane is one trajectory; eight
process workers would measure independent-trajectory throughput instead. The
result shows that current JX long single-trajectory implementations are not
competitive with the appropriate REBOUND solvers. It does not invalidate the
separate JX win for 256 short independent CUDA systems and is not a portable
hardware or general algorithm ranking.

## Post-rc5 native/fused Wisdom--Holman prototype

The first post-rc5 long-trajectory optimization is benchmark-private and does
not change the supported public engine backend. It moves the fixed 64-term
universal-G series bundle into a separately identified C extension and fuses
the local propose/commit handoff. Every outer proposal still fully validates
its source cache; every local commit checks source-capability identity; the
final committed cache is fully validated; every physical guard remains active;
and the public result still performs its mandatory deterministic full replay.

Before the first measured 100-period execution, protocol v1 locked the exact
source roster, compiler/runtime, workload, three balanced repetitions,
bitwise-result gate, timing scope, and minimum `1.20x` speedup gate. It also
disclosed the known two-period development timing and the earlier public JX /
REBOUND result. The locked protocol is
`benchmarks/jx_native_wisdom_holman_long_v1_protocol.json`, 3,728 bytes,
SHA-256
`80445a5eeb5a0431183d3bb9999988054eae68eb2d40b52ffccecd3a5be10b08`.

The prospective result passed both gates:

| 100-period weak hierarchy | Median seconds |
|---|---:|
| Unchanged public Python WH path | 39.7208881 |
| Native/fused benchmark prototype | 23.8857250 |

The prototype is `1.66296x` faster and retained the exact public result-content
SHA-256
`c62e9feb3c5a61f07ba48aeacbac42b81fb12dc682e28151794497d95683c62e`.
The report is `/tmp/jx_native_wh_long_100p_v1.json`, 4,614 bytes, SHA-256
`9618426bae1b7c98d550b8ab23556949989f61b4fe8ab9458318672efd5ee534`,
with verified semantic-content SHA-256
`8e96bc10e5ecc2afd2fce0c5a2593981c5088c4e887be0bb95b621052c5963c2`.

For scale only, comparing the prototype median with the previously locked
same-host WHFast median reduces the observed implementation gap from about
`3,332x` to about `1,986x`. That cross-report calculation is an inference, not
a new paired JX/REBOUND race. JX therefore still does not compete with WHFast
for this long individual trajectory. The next implementation target is a
native complete map loop rather than another Python orchestration micro-
optimization.

## Post-rc5 complete native Wisdom--Holman loop

The next target is now implemented as a separate benchmark-private C lane.
It keeps the public JX boundary validation, then performs the entire fixed-step
map and all numerical guards in one native call. It supports an independently
timed second complete native replay. It does not silently replace the public
engine backend.

The source-locked protocol honestly records that full-workload pilot timings
and the outcome were known before lock; it is prospective only for the final
balanced samples and bound report. The 100-period public parity gate passed
bitwise for retained positions, velocities, and epochs. All timed native
repetitions and all WHFast repetitions retained stable content hashes. Both
methods passed the same existing IAS15-based accuracy envelope.

| 100-period weak hierarchy | End-to-end median | Relative to WHFast |
|---|---:|---:|
| Complete native JX, one pass | `0.079649746 s` | `7.072x` slower |
| Complete native JX, one replay | `0.159790266 s` | `14.188x` slower |
| REBOUND 5.1.1 WHFast, safe mode | `0.011262349 s` | baseline |

The one-pass lane is about `500.07x` faster than the fresh `39.830334 s`
public-JX parity execution while preserving all retained trajectory bits. The
historical public-path gap to WHFast fell from about `3,331.94x` to `7.07x`, a
roughly `471x` reduction in the gap. JX has made decisive progress, but WHFast
still wins this recorded individual-trajectory latency workload.

Protocol:
`benchmarks/jx_native_wisdom_holman_rebound_race_v1_protocol.json`, 5,652
bytes, SHA-256
`a9d08b1014d9713d881785f205878020dbaba4086df6dcaa2419b4d6b19a6759`.
Report: `/tmp/jx_native_wh_rebound_race_v1.json`, 30,138 bytes, SHA-256
`2beb7944f9e885473fb3a23cc840d4fcc79ca789028a0afe4064b9f62ad4f290`,
semantic SHA-256
`0d2d50c3edd84056eeb615246242996722cc0d6c4dd315fceb1742f98a9c9769`.

## Remaining gates

1. Use an independently administered host before claiming independent
   replication; the two current hosts have common ownership.
2. The complete native map loop now preserves the numerical guards, exact
   replay, accounting, and bitwise/accuracy gates in benchmark-private form.
   Both 233-test engine matrices and two clean byte-identical verification
   builds passed. Before public-backend promotion it still needs a public
   failure-record and result-custody design, a separately reviewed supported
   API, and a new release version/protocol. Do not reinterpret or tune the
   completed protocol.
3. Define a REBOUND-comparable coupled-lunar orbital subset before attempting a lunar
   speed statement; REBOUND does not implement the complete JX mantle/core
   model used by the current holdout.
4. Reduce the remaining `7.07x` raw WHFast gap under a new protocol, beginning
   with the universal-Kepler solver and native guard/transform cost. Keep the
   RTX 5060 and RTX 4050 CUDA work scoped to batch throughput and two-machine
   reproduction.
