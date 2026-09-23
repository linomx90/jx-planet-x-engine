# JX GR15 CPU backend

## Status

The 0.6.0rc10 candidate packages the qualified V3 Gauss--Radau
order-15 solver as the supported `jxplanetx.gr15` CPU interface. V3 adds a
tolerance-aware corrector threshold and residual-gated terminal-force reuse;
the exact V2 implementation remains packaged as a private reference. The
numerical scope is unchanged:

- 2--32 fully mutual bodies;
- unsoftened Newtonian point-mass gravity;
- positive gravitational parameters;
- NumPy binary64, C-contiguous state arrays;
- forward or backward integration; and
- exact output checkpoints reached by step clipping.

The software interface is supported. Its scientific claim state remains
`SCREENING_ONLY`; it is not a production ephemeris, navigation product,
collision solver, or assertion of superiority over another integrator.

## Installation

Install the engine extra so NumPy and the packaged native extension are
available:

```bash
python -m pip install 'jxplanetx[engine]'
```

Platform wheels include the public backend `jxplanetx._gr15_v3_cpu` and the
retained V2 reference `jxplanetx._gr15_cpu`. Building from the source archive
requires a C11 compiler and Python development headers. GR15 never invokes a
compiler at runtime and never imports files from `benchmarks/`.

## Minimal example

```python
import math
import numpy as np

from jxplanetx import GR15Spec, integrate_gr15

positions = np.array(
    [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    dtype=np.float64,
)
velocities = np.array(
    [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
    dtype=np.float64,
)
gm = np.array([0.5, 0.5], dtype=np.float64)

spec = GR15Spec(
    initial_epoch=0.0,
    final_epoch=2.0 * math.pi,
    intermediate_epochs=(math.pi,),
    initial_step=0.05,
    minimum_step=1.0e-14,
    maximum_step=0.5,
    epsilon=1.0e-9,
)

result = integrate_gr15(positions, velocities, gm, spec)
print(result.positions)
print(result.accepted_steps, result.rejected_steps)
print(result.replay_digest)
```

## Unit contract

GR15 does not assume SI, astronomical, or canonical units. Inputs must form
one self-consistent system:

- epochs and steps: time;
- positions: length;
- velocities: length/time; and
- gravitational parameters: length³/time².

Positions and velocities must describe the same Cartesian frame and origin.
GR15 does not transform frames, remove barycentric drift, or obtain physical
constants for the caller.

## Controller semantics

`epsilon` is the acceptance threshold for the highest coefficient of the
degree-seven acceleration polynomial, normalized by the largest sampled
acceleration. It is not a conventional componentwise `rtol` or `atol`.

Each proposed step solves the eight-node implicit collocation equations by
fixed-point correction. After an accepted step, the converged acceleration
polynomial seeds the next step when the adjacent step ratio lies in
`[0.25, 2.0]`. Other trials use the constant-acceleration fallback. Rejected
trials never commit predictor history.

V3 stops correction at the larger of `convergence_factor * binary64 epsilon`
and `epsilon * 2^-20`. It reuses the already aligned final force vector only
when the stricter first threshold is satisfied; otherwise it performs the V2
terminal force sweep. The default `convergence_factor` is 256, selected before
the locked holdout run.

The result exposes attempted, accepted, rejected, force-evaluation,
corrector-iteration, predictor, history-commit, terminal-reuse, and terminal-
sweep counts. It also returns the maximum corrector residual and threshold,
maximum observed controller ratio, and accepted-step range.

## Failure behavior

Invalid Python inputs raise `GR15ContractError`. Native failures raise
`GR15IntegrationError` with `status` and `status_name`. Named native failures
cover:

- invalid input domain;
- non-finite arithmetic;
- exact force singularity;
- step limit;
- rejection limit;
- minimum-step exhaustion; and
- invalid checkpoint schedule.

JX does not return a partial trajectory as success and does not silently
switch to another integrator.

## Reusable workspaces

`prepare_gr15_workspace(body_count, spec)` avoids repeated output allocation.
Results created with an explicit workspace borrow read-only views and remain
valid only until that workspace is reused. Call `.copy()` when a trajectory
must outlive the next workspace call.

## Current limitations

- no massless particles or zero/negative gravitational parameters;
- no softening, collision response, regularization, or event location;
- no user-defined, relativistic, figure, tide, or nongravitational forces;
- no dense-output interpolator; checkpoints shorten steps;
- no GPU GR15 implementation;
- no general accuracy or performance superiority claim; and
- no production or independent scientific qualification.

The first preregistered multi-body and adversarial matrix has passed. Remaining
release work is final versioned-wheel qualification and broader stress beyond
the finite cases described below. Performance changes must preserve this
public interface and pass the installed-wheel acceptance suite.

## V2 supported-component IAS15 baseline

The prospective protocol in `benchmarks/jx_gr15_ias15_protocol.json` compares
the public GR15 API with exact REBOUND 5.1.1 IAS15. It uses closed-form Kepler
truth for both methods, chooses the loosest passing epsilon from the locked
`1e-6` through `1e-12` grid, and balances 16 timing repetitions.

Both methods selected `1e-6` and passed the shared gates. Recorded median-time
ratios (JX/IAS15) were 1.540 for a 100-period endpoint-only circular binary,
1.293 for the same integration with 101 outputs, and 1.657 for an
eccentricity-0.9 orbit with 17 outputs. IAS15 was faster in all three rows.

The report is `/tmp/jx_gr15_ias15_v1.json`, SHA-256
`67ae4818f3874960aa0a46185a01190e8756bd1f0b70d5008d784ac45ae2ec8f`;
its semantic-content SHA-256 is
`56968d8389c538e0be3a7ac28e7bff2a7dc1014a9e69384612f5fbd50f86361d`.
This is one-host, two-body, screening-only evidence. It does not qualify
multi-body dynamics, close encounters, arbitrary forces, or production use.

## V3 corrector qualification and IAS15 rerun

V3 was preregistered before calibration and timing. On the three sealed
holdouts it reduced V2 force evaluations by 24.6%--26.7% and produced median
V2/V3 speedups of 1.288, 1.292, and 1.234. An exact repeat preserved every
non-timing field and measured 1.280, 1.279, and 1.211. All analytic accuracy,
conservation, reverse-time, residual, determinism, failure-domain, and
32-body execution gates passed. The eccentric rejection count remained 20;
the rejection controller has not improved yet.

The direct V3/IAS15 rerun found JX 1.016x faster with 101 outputs, below the
pre-existing 5% timing-resolution threshold and therefore an unresolved tie.
IAS15 remained 1.189x faster endpoint-only and 1.393x faster on the eccentric
workload. The repeat measured 1.014x JX throughput on the output-rich row and
IAS15 advantages of 1.184x and 1.387x on the other rows.

Qualification report `/tmp/jx_gr15_v3_qualification_v1.json` has SHA-256
`1226492348fe8a4ac0607afad793083bf049e69f34f8e326811ca5b3e7f8d9fa`.
The direct report `/tmp/jx_gr15_v3_ias15_v1.json` has SHA-256
`008d8bf76e4361256e7d71db0da3f68d911080fc9ee2a46238b1de7f0bfe9b67`.
These remain one-host, two-body, screening-only results.

## V3 long and adversarial qualification

The correctness-only protocol
`benchmarks/jx_gr15_v3_adversarial_protocol.json` was locked before its first
execution. It covers:

- an eccentricity-0.99 binary for 10 periods;
- a 1e12:1 mass-ratio binary for 25 periods;
- the retained three-body close-scatter geometry with dense encounter outputs;
- an 11-body planetary hierarchy for 20 inner periods; and
- the supported 32-body boundary for two inner periods.

Each case requires two bitwise-identical JX executions, agreement with strict
REBOUND 5.1.1 IAS15 primary and sensitivity references, conservation,
forward/backward consistency, residual and terminal-force accounting, and
positive retained pair separation. The binary cases additionally require
agreement with closed-form Kepler truth. Coincident input, minimum-step
exhaustion, and a body count above 32 must fail closed.

All gates passed. The largest scaled JX/IAS15 state difference was
`1.3564007860417082e-8` in the eccentricity-0.99 velocity trajectory; the
32-body case was within `4.189276020611027e-13`. The complete run repeated
byte-for-byte. Protocol SHA-256:
`13304170185aeda698d2af5c784c1aa9a478164c3ee72c8e19a9efa5bad3bf1d`.
Report `/tmp/jx_gr15_v3_adversarial_v1.json` SHA-256:
`b0e53c293efbc0cc27e9a246724569183c4cac0b700c53fe04236cf4da18154d`;
semantic SHA-256:
`2abd199741f03200c077dd4670a85a53594a7878547eca8fbbbbf723d43673d6`.

This is still a finite, synthetic, one-host screen. Retained minimum distance
is not encounter-time localization, IAS15 is an external numerical reference
rather than physical truth, and the result does not qualify collisions,
arbitrary forces, production ephemerides, or general superiority.

## Exact rc9 artifact acceptance (unpublished predecessor)

The final rc9 protocol binds the byte-reproducible wheel and source archive,
the current package sources, and both earlier numerical protocols. A fresh
wheel installation verified all 125 hashed files against the wheel `RECORD`,
confirmed that `jxplanetx` loaded outside the repository, and reran both the
direct IAS15 matrix and full adversarial portfolio. A fresh source-archive
installation separately passed the public V3 smoke test.

All fixed gates passed. The immediate repeat preserved every non-timing field
exactly. The output-rich direct row remained a timing tie: JX throughput was
1.020x and 1.015x IAS15 across the two runs. IAS15 remained 1.194x/1.186x
faster endpoint-only and 1.386x/1.390x faster on the eccentric row.

- protocol SHA-256:
  `87cc7fbc0b990321d5eed24a9abb6c76f202cda23d21dd7122badb0a06a9e940`;
- wheel SHA-256:
  `b5db276d9e4293e3a17a380ecbefb614b7ecea0671fb876736e090112dbde842`;
- source archive SHA-256:
  `e65c56f5aecc435df5356d4969d9ec755d6f01ff92ffdef17a5b7305e3284efb`;
- first report SHA-256:
  `005ab24e9e0189c233c48a13874c4064f8087a08c6a20ae6490d81d50d63076a`;
- repeat non-timing content SHA-256:
  `592ae43793e19af4595ba4673fe41f2c02b6d4c8680f8e66489199e0c5a9e674`.

These hashes identify the unpublished rc9 predecessor artifacts and reports.
Rc10 preserves the numerical implementation while correcting checkout-mode
normalization in the release builder; it receives a separate artifact identity
and acceptance record. Neither record authorizes production use or a general
solver claim.

## Exact rc10 artifact acceptance

Rc10 leaves the GR15 V3 numerical source unchanged and fixes the release
builder's checkout-mode sensitivity. Its local CPython 3.14 Linux wheel is
SHA-256
`01bff6e4f418268a0acd0954e91f201397c8e62b3ef84852d332282aed13fa7e`;
the source archive is
`af390c558aa46eefa869edf2062a81869ef0d0147588467ce5af2f61c7b07e5a`.
The locked acceptance protocol is
`cd3fa44f20779a2ef06d823a2cae0b20755e31b04ea9ea2edcbfe994d6645b7b`.

The installed-wheel direct IAS15 and adversarial portfolios passed twice on
the desktop with exact non-timing replay, then passed from the same artifact
on the RTX 4050 laptop. The laptop did not reproduce the desktop output-rich
timing tie, so timing remains explicitly host-dependent. CUDA core gates also
passed on the RTX 5060 Ti and RTX 4050, but GR15 itself remains CPU-only and
the two machines share ownership.
