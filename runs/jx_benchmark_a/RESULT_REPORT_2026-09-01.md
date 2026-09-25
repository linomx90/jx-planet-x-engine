# JX Benchmark A — result checkpoint

Date: 2026-09-01  
Branch: `experiment/bm6-rebound-5.1.1`  
Classification: `MODEL_OUTPUT / NUMERICAL_ENGINEERING_ONLY`

## Scoped result

**Primary equal-force-evaluation verdict: `JX_BM6_WIN`.**

This result applies only to the frozen ten-body Newtonian DE441/Horizons workload in `benchmarks/jx_benchmark_a_contract.json`: identical initial Cartesian state, GM table, frame, epoch, ten-year duration, annual output grid, and Newtonian point-mass force model.

The primary contest assigned approximately 29,400 propagation force evaluations to each lane. REBOUND 5.1.1 Leapfrog uses one force evaluation per second-order step; JX KDK, cached Yoshida-6, and BM6 counts were recorded by the benchmark harness.

| Metric | JX BM6 | REBOUND 5.1.1 Leapfrog | BM6 improvement |
|---|---:|---:|---:|
| Force evaluations | 29,400 | 29,400 | matched |
| All-body max position error | `1.5009542e-7 AU` | `2.9449334e-3 AU` | `19,620.4x` lower |
| All-body RMS position error | `2.7322386e-8 AU` | `6.0237159e-4 AU` | `22,046.8x` lower |
| All-body max velocity error | `7.8724051e-9 AU/day` | `2.0232044e-4 AU/day` | `25,700.0x` lower |
| All-body RMS velocity error | `1.4776849e-9 AU/day` | `3.8861685e-5 AU/day` | `26,299.0x` lower |
| Max relative energy error | `7.7460486e-13` | `1.2158039e-8` | `15,695.8x` lower |
| Max relative angular-momentum-vector error | `7.4539345e-15` | `9.8183812e-15` | BM6 lower |
| Wall time in Python-vs-C workflow | `9.09135 s` | `0.0764382 s` | REBOUND `118.9x` faster |

The independent DOP853 loose/tight gate passed for this primary contest. The v2 reference disagreement was `7.4708911e-13 AU` RMS position and `4.2102593e-14 AU/day` RMS velocity, approximately 366x and 351x inside the locked limits.

## Equal-timestep status

**Verdict: `INVALID_REFERENCE`.**

A prospectively locked v2 increased only the independent DOP853 reference resolution; candidate code, state, budgets, metrics, and win rules remained immutable. The equal-timestep candidate errors became too small for the binary64 DOP853 pair to satisfy the one-percent referee gate. The reference disagreement was `7.4708911e-13 AU` RMS position while the required limit was `1.1271947e-14 AU`. The tighter member used 701,327 RHS evaluations and showed larger invariant error, consistent with a binary64 accumulation/roundoff floor.

No equal-timestep winner is claimed. A 160/224-bit or otherwise genuinely higher-precision referee is required.

## Preservation

Benchmark v1:

- workflow run `33565234437`
- candidate commit `847433b819836cf71cf55f2b5fac9f7c566a4243`
- result SHA-256 `063de521c66bc4671dc6fa28cffcadfd4d17cc7b0cd2ea2fbdd6946eed563786`
- artifact ZIP SHA-256 `95a2064b4524f37c3ff5e8324a2961e166fe8463cab531600b806d35a25f8378`

Reference-resolution v2:

- workflow run `33566265896`
- workflow commit `91e259772a82649460c51c0c66276f968ac3651e`
- result SHA-256 `aeee0cee19474113278b6865722a8033fc5932aa5d7a36844684d0786286cf7a`
- artifact ZIP SHA-256 `abd638fb493f64b686060a21dfc9fe720d69b87c2d6cf4d33fe4d9ebe4bf5f30`
- frozen normalized state SHA-256 `43e6f0655447286e74a146b3b07ae24c9daeed77a0aa14bfc98f76dbd29704c4`

The complete bundles, trajectories, contracts, dependency lists, summaries, and SHA-256 manifests are also preserved in the ChatGPT Library under:

`/JX General Dynamics Engine/Benchmarks/JX Benchmark A/2026-09-01/`

## Engineering decision

1. Keep KDK as the transparent baseline.
2. Keep cached Yoshida-6 as the trusted high-order fallback.
3. Advance BM6 as the leading binary64 smooth/far-field lane.
4. Do not use the current approximate BM6 coefficients for 160/224-bit production qualification.
5. Implement BM6 in native C++ and repeat the frozen contest to address the wall-time deficit.
6. Build a genuinely higher-precision referee before reopening the equal-timestep verdict.
7. Treat close encounters and hybrid switching as a separate validation regime.

## Claim boundary

This is a scoped accuracy-per-force-evaluation win over REBOUND 5.1.1 Leapfrog on one workload. It is not a universal REBOUND ranking, a speed win, a complete ephemeris reconstruction, or a Planet-X result.
