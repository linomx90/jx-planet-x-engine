# JX General Dynamics

JX General Dynamics is one research platform with several evidence tracks.  A
track can belong to the platform without sharing a numerical solver or being
scientifically qualified.  This distinction prevents a successful benchmark in
one domain from silently certifying another.

The machine-readable registry lives in
`src/jxplanetx/general_dynamics.py`.  It binds each active track to an exact
evidence artifact, status, integration boundary, claim ceiling, and next gate.
The current frozen export is stored at
`runs/jx_general_dynamics_registry_v17/REGISTRY.json`. Version 17 binds the
registry to package `0.6.0rc1` and makes the project-wide
`SCREENING_ONLY` claim ceiling explicit without changing any capability row,
evidence binding, or scientific status. Version 16 and all earlier versions
remain immutable historical evidence. Regenerate with
`benchmarks/jx_general_dynamics_registry.py` into a new, empty destination.

## Current capability map

| Capability | Evidence state | Integration boundary |
|---|---|---|
| Equal-mass binary orbit | Qualified benchmark | Native JX benchmark |
| Reduced DE440 eleven-body orbit | Reproducible screening | Native JX benchmark |
| 4.5-Gyr linear Solar-System secular baseline | Inconclusive | Native JX benchmark |
| Lunar mantle/core rotation | Inconclusive | Research binding only |
| Collisionless cosmology particle mesh | Qualified foundation benchmark | Native JX benchmark |
| 3-D dark-matter plus adiabatic baryons | Qualified benchmark | Native JX benchmark |
| Ideal-gas Sod and Sedov shocks | Qualified benchmark | Pinned Castro solver |
| Nuclear reacting detonation | Inconclusive | Pinned Castro/Microphysics solver |
| AIR-5 thermochemistry | Supporting reference | Pinned Mutation++/SU2 solvers |
| Radiation-reacting hydrodynamics | Planned | No implementation |

“Qualified benchmark” applies only to the exact frozen problem.  It does not
mean the entire JX platform is scientifically qualified or production-ready.

`JX-COSMO-PM-01` established the first native cosmological-dynamics foundation. It
evolves collisionless particles in a prescribed expanding background with CIC
mass assignment/gathering, a periodic spectral Poisson solve, and second-order
kick-drift-kick integration. The frozen analytic growing-mode case passes all
10 gates, including near-fourfold spatial and temporal error contraction,
CPU/GPU parity, bitwise GPU repetition, and a 128^3-particle result with
0.01334% relative growth error. Normal and optimized reports are byte-identical.
This qualification applies only to that analytic periodic workload: it is not
a realistic structure-formation run and includes no gas, radiation, plasma,
nuclear physics, general relativity, observational fit, or production claim.
`JX-COSMO-PM-02` now extends that exact analytic case to 256^3 particles and a
separately coded symmetry-reduced NumPy CPU particle-mesh reference. The first
frozen attempt stopped on GPU segmented-accumulation noise; recovery 2 changed
only that numerical reduction and retained every physical input, threshold,
and claim limit. The repaired normal and optimized reports are byte-identical,
all 10 gates pass, final sheet positions match exactly, and growth differs by
8.22e-15. The reference is independent code for this planar symmetry case, not
an external general 3-D cosmology package. The next gate is an arbitrary 3-D
perturbation comparison against an independently maintained external
particle-mesh code.

`JX-COSMO-BARYON-01` adds the first native coupled gas milestone. It evolves
262,144 collisionless dark-matter particles and a 64^3 monatomic ideal-gas
baryon mesh through the same periodic gravity field in supercomoving
variables. Three unequal axis-aligned modes form a multidirectional separable
linear state. Independent review invalidated the frozen R2 pass: its particle
gravity mesh was displaced by half a cell from the gas mesh, and its old 3-D
CFL estimate did not bound six independently fast incident faces. Both defects
are fixed and covered by adversarial regressions. The corrected 64^3 R3 then
honestly failed the unchanged growth gate; a separate spatial/time ladder
showed near-second-order spatial contraction, remained above the gate at 64^3
through 48 sampled steps, and selected 128^3 with 16 steps.
The preregistered R4 kept every accuracy threshold, added phase-sensitive
complex-growth checks, and passed all 15 gates in byte-identical normal and
optimized runs. This remains a small, linear, first-order validation problem—
not an external-code validation or a production simulation of nonlinear
structure, galaxies, cooling, chemistry, magnetic fields, stars, black holes,
radiation, or feedback.

The 4.5-Gyr GPU result is an orbit-averaged Laplace-Lagrange baseline, not a
direct eleven-body trajectory. A refined nonlinear GPU bridge connects the
same source state to that baseline over 100 years. An external REBOUND 5.1.1
reproduction now independently confirms the GPU endpoint with IAS15 to a
maximum component difference of 0.7043 m and 2.524e-6 m/s. A global
drift-kick-drift symplectic ladder also shows the expected approximately 4x
second-order refinement; its disclosed Richardson endpoint is 33.44 km and
0.07956 m/s from IAS15. Normal and optimized reports are byte-identical.

A second validation starts from the one-day endpoint already frozen in the
source data but not used to select that symplectic configuration. The GPU
RKF78 endpoint agrees with tight IAS15 after another 100 years to 0.8030 m and
2.243e-6 m/s, while the global leapfrog ladder again shows approximately 4x
refinement. Direct elements are also averaged over each body's first complete
initial osculating period: this reduces the linear-secular mismatch for eight
of nine eccentricity records and seven of nine inclination records. Because
the state comes from the same frozen source, this is a within-source holdout,
not blind external validation. The body-period averages are descriptive, not
canonical mean elements or pass/fail evidence for the linear secular model.

The next external-epoch gate now also passes. A state queried directly from
the retained JPL DE440s kernel at exactly 1,000,000,000 TDB seconds after
J2000 was selected before its numerical outcome was calculated. All previous
100-year settings and thresholds were retained unchanged. JX GPU RKF78 agrees
with tight REBOUND IAS15 to 0.5383 m and 1.985e-6 m/s, and the independent
global leapfrog ladder again refines by approximately 4x in both position and
velocity. This is stronger cross-epoch numerical validation, but all solvers
propagate the same reduced Newtonian model after initialization; it is not a
100-year reproduction of DE440 or evidence of navigation accuracy.

The new multi-epoch stress test broadens that numerical evidence to 63 direct
DE440s initial epochs spanning roughly 1873--2126. Eight repetitions per epoch
produce 504 concurrent CUDA lanes; every 100-year kernel, repeatability,
embedded-error, conservation, and barycentre gate passes. Both recorded runs
reached 100% peak and median GPU utilization, and normal/optimized scientific
reports are byte-identical. A separate 1,000-year ladder at three preselected
epochs is deliberately stopped: leapfrog refinement remains approximately 4x,
but the Richardson-extrapolated velocity differs from tight IAS15 by
132--156 m/s, above the frozen 100 m/s limit. JX therefore records broader GPU
screening without promoting the long-horizon orbit track. The next clean gate
is a separately preregistered 600-second fourth symplectic level, not a post-hoc
relaxation of the failed limit.

The first attempted WHFast/WHDS branch is retained as stopped evidence because
its refinement was non-monotonic for the separately represented Earth-Moon
subsystem. The successful confirmation configuration was selected after a
same-state exploratory run, so it is reproducible numerical confirmation, not
blind validation. The model still reduces eleven bodies to nine nodes for its
secular comparison, collapses Earth and Moon into their barycentre there, and
omits nonlinear long-horizon qualification, close encounters, tides, solar
evolution, migration, and stellar encounters. It is not evidence that the
real Solar System is stable for 4.5 billion years.

## Architecture

The orbital Python engine remains the native JX dynamics implementation.
Specialist continuum-physics solvers are bound as reproducible external
backends until JX owns an independently validated implementation.  Evidence
packages provide the bridge:

```text
JX General Dynamics
  orbital dynamics                 native JX CPU/GPU kernels
  rotational/interior dynamics     native research probes, unresolved physics
  compressible hydrodynamics       pinned Castro/AMReX CPU/GPU backend
  reaction microphysics            pinned Microphysics or Mutation++ data/code
  radiation/plasma/materials       planned, separately qualified later
```

`JX-GD-REACT-02` advances the small readiness case to a six-lane,
one-dimensional helium-detonation qualification attempt. All lanes were
physically admissible, the published-speed consistency, timestep,
shock-treatment, and CPU/GPU speed gates passed, and the GPU reached 100% peak
utilization. The attempt remains reproducible screening rather than a qualified
benchmark because spatial differences did not contract and the unaligned
CPU/GPU final-profile gate failed. The next milestone must be separately
preregistered: a finer spatial ladder plus a front-aligned or transport-based
profile comparison, while retaining the original unaligned metric. No
multidimensional explosion, supernova, nucleosynthesis, radiation, or
production claim is authorized.

`JX-GD-REACT-03` adds an external-physics reality test using a steady
Chapman–Jouguet calculation from the same fuel state. The reference solver
repeated exactly and satisfied its Hugoniot, sonic, composition, and nonlinear
residual gates, but predicted 15,416 km/s. The matched CPU and GPU simulations
were both about 10,240 km/s, 33.6% below the reference and outside the frozen
10% gate. The reacting track is therefore `INCONCLUSIVE`. This comparison
shares the Helmholtz EOS and aprox19 nuclear masses and assumes complete Ni-56
ash, so it is a model-based reality check rather than laboratory or
astronomical observation. Front-observable and steady-state diagnostics must
precede a larger multidimensional run.
