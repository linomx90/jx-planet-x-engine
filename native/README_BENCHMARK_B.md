# JX Benchmark B native encounter runner

`jx_benchmark_b_native.cpp` is an isolated close-encounter qualification tool.
It does not replace the frozen Benchmark A BM6 implementation.

It provides four lanes:

- `bm6`: native binary64 Blanes–Moan S10/BM6;
- `kdk`: cached-acceleration kick–drift–kick control;
- `bs160`: 160-bit adaptive Bulirsch–Stoer reference;
- `bs224`: 224-bit adaptive Bulirsch–Stoer reference.

The Bulirsch–Stoer lanes use modified-midpoint integrations and polynomial
extrapolation in squared substep size. They are independent of the fixed-step
BM6 implementation. The Python harness adds closed-form two-body checks,
REBOUND 5.1.1 IAS15/TRACE/MERCURIUS lanes, and a SciPy DOP853 full-system
cross-check.

Build:

```bash
make -C native -f Makefile.benchmark_b release
make -C native -f Makefile.benchmark_b sanitizer
```

The strict release command disables fast-math and floating-point contraction.
All authoritative schedules and gates are defined in
`benchmarks/jx_benchmark_b_contract.json`.
