# JX native BM6 lane

This directory contains the native C++20 replay lane for **JX Benchmark A**.
It implements the same direct mutual Newtonian force and the same experimental
Blanes–Moan sixth-order S10/BM6 map used by the preserved Python benchmark.

## Numerical contract

- IEEE-754 binary64 only.
- `-fno-fast-math` is mandatory.
- floating-point contraction is disabled with `-ffp-contract=off`.
- 11 drift stages and 10 kick/force stages per BM6 macro-step.
- the published decimal coefficients are suitable for the binary64 challenge,
  not for the 160/224-bit production lane.
- existing KDK, Yoshida-6, Python BM6, DE441 inputs, and historical results are
  not modified.

## Build

From the repository root:

```bash
make -C native release
make -C native self-test
make -C native sanitizer
```

The release binary is written to `build/native/jx_bm6_native`.

## Frozen primary replay

```bash
build/native/jx_bm6_native \
  --state runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv \
  --gm runs/de441_horizons_10yr/gm_de440_major_barycenters.csv \
  --trajectory runs/jx_benchmark_a_native_cpp/trajectory_native_bm6.csv \
  --result runs/jx_benchmark_a_native_cpp/native_run.json \
  --contest equal_force_budget \
  --dt-days 1.2423469387755102 \
  --steps 2940 \
  --output-every-steps 294 \
  --timing-repeats 31
```

The authoritative CI harness is
`benchmarks/jx_benchmark_a_native_validate.py`. It requires two byte-identical
native trajectory replays, compares the parsed native state against the Python
BM6 state, reruns the valid DOP853 reference gate, reruns REBOUND 5.1.1
Leapfrog, and reports separate median integration-only timings.

## Claim boundary

This lane can establish only a scoped native replay and benchmark result on the
frozen smooth ten-body Newtonian workload. It does not validate close
encounters, arbitrary precision, a complete ephemeris model, universal
superiority over REBOUND, or any Planet-X claim.
