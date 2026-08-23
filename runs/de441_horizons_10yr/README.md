# JX / DE441–Horizons ten-year validation

**Machine verdict:** `PASSED`  
**Completed:** 2026-08-22  
**Claim scope:** external-reference compatibility test; `SCREENING_ONLY`

## Question tested

Starting from authoritative JPL Horizons states at TDB JD 2461200.5, can the
JX Newtonian point-mass engine propagate the ten major Solar-System
barycenters for ten years while remaining compatible with annual DE441 states
for Jupiter, Saturn, Uranus, and Neptune?

The reference data, source hashes, active bodies, epochs, integrator settings,
step sizes, comparison coordinates, and acceptance limits were locked before
JX was executed. Limits were not changed after seeing the output.

## Locked design

- Reference: NASA/JPL Horizons, source DE441.
- State: geometric Cartesian vectors, Solar-System-barycentric ICRF, TDB,
  AU and AU/day.
- Epochs: 11 annual samples from JD 2461200.5 through JD 2464853.0.
- Active model: Sun plus Mercury through Pluto system barycenters.
- Science gates: heliocentric Jupiter, Saturn, Uranus, and Neptune residuals.
- Backend: REBOUND 4.4.11 IAS15 through JX, fixed-step mode.
- Independent numerical settings: 0.125 day and 0.0625 day.

## Results

| Locked gate | Observed | Limit | Result |
|---|---:|---:|:---:|
| Maximum outer-planet DE441 position residual | 33.6013 km | 1,495.9787 km | PASS |
| Maximum outer-planet DE441 velocity residual | 0.000510045 m/s | 0.0173146 m/s | PASS |
| Coarse–tight outer position difference | 4.44089e-16 AU | 1e-10 AU | PASS |
| Coarse–tight outer velocity difference | 5.42101e-20 AU/day | 1e-11 AU/day | PASS |
| Maximum relative energy drift | 4.92960e-16 | 1e-12 | PASS |
| Maximum relative angular-momentum drift | 3.79038e-16 | 1e-12 | PASS |
| IAS15 iteration-limit events | 0 | 0 | PASS |
| Exact requested step counts | 29,220 and 58,440 | exact | PASS |

Maximum residual over the 11 locked epochs by science-gate body:

| Body | Position | Velocity |
|---|---:|---:|
| Jupiter barycenter | 33.6013 km | 0.000510045 m/s |
| Saturn barycenter | 7.38075 km | 0.0000560207 m/s |
| Uranus barycenter | 3.43145 km | 0.0000243859 m/s |
| Neptune barycenter | 9.64109 km | 0.0000522757 m/s |

The global maximum occurred for Jupiter at the final epoch, TDB JD 2464853.0.

## Interpretation

Within the exact predeclared scope, JX numerically reproduces the short-arc
outer-planet behavior of DE441 to tens of kilometres across a decade. The
coarse/tight agreement and invariant drift are far smaller than the
external-reference mismatch, so the remaining difference is consistent with
the deliberately simplified force model rather than timestep error.

This does **not** reproduce the full DE441 force model. JX omitted
post-Newtonian relativity, resolved moons and satellite systems, asteroids and
trans-Neptunian perturbers, solar oblateness, and non-gravitational forces.
The result is not an observational fit and neither detects nor excludes Planet
X. It also does not validate long-term chaotic population conclusions.

## Reproduce and audit

From the repository root, with the locked REBOUND 4.4.11 binary available:

```bash
python3 runs/de441_horizons_10yr/run_validation.py
```

Primary artifacts:

- `contract_v1.json` — immutable pre-execution protocol and thresholds.
- `reference/horizons_reference_manifest.json` — queries and hashes for all
  ten raw Horizons responses.
- `reference/horizons_de441_vectors.csv` — normalized 110-row reference.
- `result_v1.json` — machine verdict, every gate, provenance, and summaries.
- `jx_states_v1.csv` — both complete JX output grids.
- `residuals_v1.csv` — all numerical and DE441 residuals.

Locked contract SHA-256:
`5112c90e71317a5cae54d17ea82296296af68812f18c663429dc773ac161053f`

First completed result SHA-256:
`277c006e4c6474b2d52c1f002daf6b156067398685dca8cc1efa1e138c753a4e`

## Primary references

- [JPL Horizons API documentation](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
- [JPL Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html)
- [JPL planetary ephemeris export information](https://ssd.jpl.nasa.gov/planets/eph_export.html)
- [Park et al. (2021), DE440 and DE441](https://doi.org/10.3847/1538-3881/abd414)
