# JX General Dynamics Engine — Benchmark B Result Checkpoint

**Date:** 2026-09-02  
**Classification:** `MODEL_OUTPUT / NUMERICAL ENGINEERING ONLY`  
**Verdict:** `JX_BM6_ENCOUNTER_BOUNDARY_MEASURED`  
**Branch:** `experiment/jx-benchmark-b-close-encounter`  
**Successful workflow run:** `33575199573`

## Scope

Benchmark B measured the fixed-step encounter operating boundary of the native C++20 BM6/S10 lane. It covered 25 cases:

- 3 exact hyperbolic two-body flybys;
- 16 normalized restricted three-body encounters with Jupiter- and Neptune-mass ratios, `rho = d_min/R_H` near 3, 1, 0.3, and 0.1, and planar/vertical geometry;
- 6 injected Jupiter/Neptune encounters in the frozen ten-body Solar-System workload.

All 25 reference constructions passed. All 25 deterministic state/event replays passed. The six full ten-body portability screens passed.

## Boundary

The controlling dimensionless step is

\[
\eta = \frac{\Delta t}{\tau_{\rm enc}},
\qquad
\tau_{\rm enc}=\frac{d_{\min}}{v_{\rm rel,min}}.
\]

For each case, `eta_safe_max` is the largest achieved value in the uninterrupted fine-to-coarse passing prefix. A later isolated coarse-step pass cannot reopen the boundary after a failure.

The strict shared primary boundary was

\[
\boxed{\eta_{\rm measured}=0.1952545635089222}.
\]

The limiting case was `cr3bp_p5_rho_3_planar`, with `rho_reference=3.03877994935888`. This is about 5.1215 BM6 macrosteps per local crossing time.

Family ranges:

| Family | Cases | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Exact analytic two-body | 3 | 0.199602628 | 0.398136395 | 0.398269639 |
| Restricted three-body | 16 | 0.195254564 | 0.494564185 | 0.577495591 |
| Frozen ten-body screen | 6 | 0.581542643 | 0.594413927 | 0.751060867 |

Every case passed the predeclared `eta_target=0.2` grid point. Integer step rounding made achieved eta range from 0.187764690 to 0.213548861, so the strict shared achieved boundary is 0.195254564 rather than exactly 0.2. At `eta_target=0.4`, only 22/25 cases passed.

## Prospective Benchmark-C guard

Use

\[
\boxed{\Delta t_{\rm BM6}\le 0.1\,\tau_{\rm enc}}
\]

as the conservative guard for the next deterministic hybrid experiment. That gives approximately ten BM6 macrosteps and 100 BM6 force evaluations per local crossing time. It is about 1.95 times inside the limiting measured boundary.

All 25 cases passed at `eta_target=0.1`. The weakest margin was minimum-separation accuracy, still 27.73 times inside its gate. Time-of-minimum accuracy was 47.41 times inside, and reversibility was 83.32 times inside.

This `0.1` value is a prospectively chosen Benchmark-C engineering guard, not a universal physical or production constant.

## Phase-sensitivity warning

The vertical `rho≈3 R_H` Jupiter and Neptune restricted-three-body cases were non-monotonic. An intermediate coarse schedule failed, while a still coarser schedule passed because of numerical phase cancellation. Their continuous safe prefixes ended at eta 0.350484440 and 0.350564888. The benchmark correctly did not treat the isolated coarse pass as evidence of safety.

## Reference stack

- 224-bit Bulirsch–Stoer versus exact analytic hyperbola: maximum final position error `1.1235e-14` of scale; velocity error `1.3343e-16`.
- Restricted three-body 160-bit versus 224-bit Bulirsch–Stoer: maximum minimum-separation disagreement `4.9366e-9 R_H`; time disagreement `9.2721e-7 tau`.
- REBOUND 5.1.1 IAS15 versus 224-bit reference: final position below `1.63e-15 R_H`; velocity below `6.09e-15 v_H`.
- Full ten-body DOP853 tolerance pair: final position below `6.3500e-13 R_H`; velocity below `1.5142e-11 v_H`.
- IAS15 versus tight DOP853: below `1.21e-13 R_H` and `1.75e-13 v_H`.
- TRACE and MERCURIUS completed all 16 restricted-three-body screening cases but did not set the BM6 boundary.

## KDK control

Across 225 fixed-step points per method, BM6 passed 144 and KDK passed 10 under the locked gates. This is a diagnostic, not an equal-force-budget ranking: BM6 uses ten force evaluations per macrostep, whereas cached KDK uses approximately one new force solve per step.

KDK remains the transparent control and disposable scout. It is not the authoritative encounter trajectory.

## Preserved failures

Two blocked workflows remain preserved without relabeling:

1. Run `33574905285`, commit `2b72b440a8a983687cb65f09b491711d388ee866`: Boost headers absent from the runner. Artifact SHA-256 `212d527bcca53bcad161ac673710719eff7aaab0ce362ac5f9c8018a3b7a809a`.
2. Run `33574980680`, commit `f5342001fc01f9ec4b780f9b947e9958fc04dd79`: obsolete REBOUND 5.1.1 IAS15 configuration attribute. Artifact SHA-256 `542981a9582e5264669137a8b2a5f161143f9d05950fe23ae3a1570ac306724b`.

Both corrections were prospective.

## Provenance

- Successful commit: `c43b9a2ebdeab40792c2eeb48d41986e5e2d1be7`
- Workflow run: `33575199573`
- Artifact ID: `9826387815`
- Artifact ZIP SHA-256: `3d981cc6a28e68d7cb8366890b274f53190bef98b84d7605d9b7335534187f67`
- Result JSON SHA-256: `d1ec4d029bb8d52053e01ed353e9f2225f2f9ea84ca58d1d333e9003a6891b6c`
- Native executable SHA-256: `b658a06cb6822a5715c40c54d1aa061730f4a9c1763057b3dd70d36249475cb5`

## Engineering decision

The next architecture is Benchmark C:

\[
\boxed{
\text{KDK scout}
\rightarrow
\text{frozen hashed encounter schedule}
\rightarrow
\text{BM6 where }\eta_{\rm predicted}\le0.1
+
\text{adaptive solver inside flagged windows}
}
\]

Benchmark C must test missed-event rate, handoff continuity, forward/backward return across both transitions, checkpoint/restart identity, and convergence against the 224-bit reference.

## Claim boundary

This result establishes a case-specific fixed-step BM6 encounter boundary for the tested geometries and gates. It does not establish universal close-encounter safety, hybrid correctness, arbitrary-precision BM6 qualification, a complete Solar-System ephemeris model, or any Planet-X result.
