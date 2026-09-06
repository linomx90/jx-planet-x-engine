# JX Benchmark A — Native C++ BM6 Result

Date: 2026-09-01  
Branch: `experiment/bm6-rebound-5.1.1`  
Native code commit: `5f55c3d16c14b3caaab49851d2d76d0a8dc54fd1`  
Workflow run: `33570885036`  
Classification: `MODEL_OUTPUT / NUMERICAL ENGINEERING ONLY`  
Verdict: `JX_NATIVE_BM6_WIN`

## Locked result

The native C++20 BM6 lane passed the primary Benchmark A contest against REBOUND 5.1.1 Leapfrog.

Both lanes used the frozen ten-body Newtonian DE441/Horizons state, identical GM values, a ten-year duration, annual outputs, and 29,400 modeled gravitational force evaluations. The independent DOP853 reference gate passed.

| Metric | Native JX BM6 | REBOUND 5.1.1 Leapfrog | Native BM6 advantage |
|---|---:|---:|---:|
| Steps | 2,940 | 29,400 | method-dependent |
| Force evaluations | 29,400 | 29,400 | matched |
| Maximum position error | `1.5009665938e-07 AU` (22.454141 km) | `2.9449334074e-03 AU` (440,555.767 km) | 19,620.2× lower |
| RMS position error | `2.7322635250e-08 AU` (4.087408 km) | `6.0237158791e-04 AU` (90,113.507 km) | 22,046.6× lower |
| Maximum velocity error | `7.8725100122e-09 AU/day` (0.013630911 m/s) | `2.0232043958e-04 AU/day` (350.309108 m/s) | 25,699.6× lower |
| RMS velocity error | `1.4777032468e-09 AU/day` (0.002558579 m/s) | `3.8861685145e-05 AU/day` (67.287330 m/s) | 26,298.7× lower |
| Maximum relative energy error | `7.7476917777e-13` | `1.2158039277e-08` | 15,692.5× lower |
| Maximum relative angular-momentum-vector error | `7.4539344749e-15` | `9.8183812493e-15` | native BM6 lower |
| Median integration-only time, 31 runs | `0.006660755 s` | `0.010079628 s` | native BM6 1.513× faster |

## Passed gates

- AddressSanitizer and UndefinedBehaviorSanitizer.
- Forward/backward self-test: maximum component return error `1.7919151405754175e-16` after 20,000 force evaluations.
- Deterministic replay: two byte-identical trajectories, SHA-256 `9ab582a9ba4f6a13a3d1a8635aaeda838f77aac22cce957f53c92e99c8797fb9`.
- Exact replay against the preserved valid Benchmark A v1 BM6 trajectory: all 11 snapshot hashes and the complete state hash matched exactly.
- Independent DOP853 reference gate.
- Locked trajectory, invariant, and accuracy win rule.

The earlier live-NumPy replay failure remains preserved as workflow run `33570374629`; it was not erased or relabeled. The final hard replay oracle is the exact SHA-256 state sequence from the already-preserved valid Benchmark A v1 BM6 output. The same-run NumPy comparison remains an explicit processor/library-dispatch diagnostic.

## Native code

- `native/jx_bm6_types.hpp`
- `native/jx_bm6_integrator.hpp`
- `native/jx_bm6_io.hpp`
- `native/jx_bm6_native.cpp`
- `native/Makefile`
- `native/README.md`

Strict release build:

```bash
g++ -std=c++20 -fno-fast-math -ffp-contract=off \
  -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG \
  native/jx_bm6_native.cpp -o build/native/jx_bm6_native
```

The implementation rejects fast-math, requires IEEE-754 binary64, validates coefficient symmetry and closure, validates the frozen input roster, counts force evaluations directly, and writes deterministic CSV/JSON artifacts.

## Provenance

- GitHub artifact ID: `9824909036`
- Artifact ZIP SHA-256: `67065838cecf215b339aeb8202056a58abfdd82a3d54571826f10995dd4ef3db`
- Result JSON SHA-256: `e53ad3e2404e6d071977a5dcfefcae5f4a1ae798943895952cb7a5ab55f40c38`
- Native executable SHA-256: `172ba517333ece69bb57cd199e2062aed6d87dbc6ab6cf61494f54f21f151575`
- Frozen normalized state SHA-256: `43e6f0655447286e74a146b3b07ae24c9daeed77a0aa14bfc98f76dbd29704c4`
- Golden replay state SHA-256: `84a6f400b565ab7769329f8eb0099d74d16e02bbb277a8e1a4974417c10f06b0`

## Claim boundary

This is a scoped native C++ BM6 accuracy and speed win over REBOUND 5.1.1 Leapfrog at equal modeled force evaluations on this frozen smooth ten-body workload. It is not universal superiority over REBOUND, close-encounter validation, arbitrary-precision qualification, a complete DE441 reconstruction, or a Planet-X result.

## Engineering decision

Native BM6 is now the leading binary64 smooth/far-field lane. KDK remains the transparent control and cached Yoshida-6 remains the trusted high-order fallback. The next independent qualification is the close-encounter/hybrid gate; the 160/224-bit lane still requires higher-precision BM6 coefficients.
