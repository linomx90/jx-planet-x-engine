# JX V5 physical-model registry

**State:** `DRAFT_NONEXECUTABLE`

**Claim ceiling:** `SCREENING_ONLY`
**Purpose:** define what JX must know and prove before any added physical model can authorize scientific propagation.

The V5 registry is a fail-closed scientific contract. It is not a list of
checkboxes and it does not imply that a declared force has been implemented.
Each model is tied to equations, units, frames, time scales, bodies,
applicability, parameters, uncertainty, covariance, provenance, software
identity, and independent validation. An unresolved value is recorded as
`TBD_BLOCKED`; JX supplies no scientific default.

The registry and its schema are:

- `registries/jx_force_parameters_v5.json`
- `schemas/jx-force-parameter-registry-v5.schema.json`

V4 remains immutable external lineage. V5 must not read, ingest, reinterpret,
or modify V4 execution artifacts.

## Scientific layers

JX separates five capabilities that are often incorrectly combined:

1. **Propagation:** calculate acceleration from a declared physical model.
2. **Encounter and collision handling:** choose a numerical method and a
   physical event policy.
3. **Observation synthesis:** transform a geometric trajectory into a modeled
   angle, delay, Doppler, range, or other observable.
4. **Estimation:** fit states and physical/nuisance parameters to measurements
   while preserving covariance and rank diagnostics.
5. **Claim control:** determine what the combined evidence can support.

A force implementation does not create an ephemeris or orbit-determination
system. A successful propagation test remains `MODEL_OUTPUT`.

## Reference systems and time

The ephemeris-grade relativistic target is the IAU 2000 Barycentric Celestial
Reference System (BCRS), in harmonic coordinates, with axes aligned to the
ICRS. Its coordinate time is TCB. A TDB-compatible implementation must declare
the linear scaling of time, spatial coordinates, and gravitational parameters;
mixing TCB and TDB-compatible quantities is forbidden.

The IAU definition is

\[
\mathrm{TDB}=\mathrm{TCB}
-L_B(\mathrm{JD}_{\mathrm{TCB}}-T_0)86400+\mathrm{TDB}_0,
\]

and compatible ephemeris quantities obey

\[
\mathbf x_{\mathrm{TDB}}=(1-L_B)\mathbf x_{\mathrm{TCB}},
\qquad
\mu_{\mathrm{TDB}}=(1-L_B)\mu_{\mathrm{TCB}}.
\]

The registry therefore requires an explicit frame, origin, axes, time scale,
epoch, compatible unit system, and validity interval for every state and
parameter. The normative references are the [IAU 2000
resolutions](https://iauarchive.eso.org/static/resolutions/IAU2000_French.pdf),
the [IAU 2006 TDB definition](https://iauarchive.eso.org/static/resolutions/IAU2006_Resol3.pdf),
and [IERS Conventions Chapter 10](https://iers-conventions.obspm.fr/content/chapter10/tn36_c10.pdf).

## Relativity

Relativistic models are distinct and mutually exclusive where they represent
alternative approximations to the same contribution.

### Solar Schwarzschild 1PN

For a negligible target relative to a stationary, spherical, nonspinning
central source, the GR correction is

\[
\Delta\mathbf a=
\frac{\mu}{c^2r^3}
\left[
\left(\frac{4\mu}{r}-v^2\right)\mathbf r
+4(\mathbf r\cdot\mathbf v)\mathbf v
\right].
\]

Here \(\mathbf r=\mathbf x_{target}-\mathbf x_{source}\) and
\(\mathbf v=\mathbf v_{target}-\mathbf v_{source}\). This function returns a
correction only; Newtonian acceleration must be supplied exactly once. It is
not the finite-mass N-body Einstein–Infeld–Hoffmann (EIH) equation and supplies
no source backreaction.

The analytic GR perihelion advance used for qualification is

\[
\Delta\varpi=\frac{6\pi\mu}{a(1-e^2)c^2}
\]

per orbit. Equation-level tests precede any trajectory test.

### N-body EIH 1PN

The full first-post-Newtonian point-mass model is global, barycentric,
velocity-dependent, and non-pairwise. It requires a simultaneous snapshot of
all source positions, velocities, gravitational parameters, interbody
separations, cross-potentials, and Newtonian source accelerations. At retained
1PN order, right-hand-side source accelerations are evaluated Newtonianly.

The complete model must be implemented from the modern BCRS/JPL equations,
not by summing independent Solar corrections. Relevant references are
[DE440/DE441](https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf) and the
[JPL equations of motion](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/Standish-Chap-8.pdf).

### Restricted PPN and Lense–Thirring

A model exposing only \(\beta\) and \(\gamma\) is named
`restricted-isotropic-PPN-beta-gamma`; it is not called full PPN. General
relativity fixes both to one. Solar spin/frame dragging is a separate
Lense–Thirring contribution requiring the source angular-momentum vector,
pole frame and epoch, rotation model, moment coefficient, and a backreaction
policy.

### Relativity qualification

Before authorization, each implementation must pass:

- an independently coded arbitrary-precision acceleration oracle;
- fixed two-, three-, and N-body vector fixtures as applicable;
- rotation, translation, and body-permutation covariance;
- \(c\rightarrow\infty\) recovery of Newtonian dynamics;
- massless-target nonbackreaction;
- central-model convergence from the matched EIH mass-ratio limit;
- TCB/TDB-compatible fixture equivalence;
- analytic perihelion advance and finite-mass binary periastron tests;
- held-out, hash-pinned ephemeris residual tests without refitting.

The current JX Yoshida method is a separable kick–drift integrator. Because 1PN
acceleration depends on velocity, the correction must not be silently inserted
into that method. An appropriate velocity-dependent integration method and its
own validation contract are required first.

JX therefore keeps its first velocity-dependent experiment in an isolated
reference layer. The fixed-step Decimal implicit-midpoint implementation solves
the full six-component first-order equation, recomputes the accepted candidate's
midpoint right-hand side and scaled residual before accepting every step, and
verifies exactly one Newtonian base plus one Solar 1PN correction at every
right-hand-side call while recording the closed ledger per accepted step. Its
precision, rounding, Decimal traps, step, tolerances, iteration limit, registry
digest, and force-plan digest are immutable inputs. It is not imported by the
legacy dynamics or Yoshida modules and exposes no command-line execution mode.

This numerical reference does not change the Solar row from `NOT_IMPLEMENTED`,
`UNQUALIFIED`, and `BLOCKED`: the registry still lacks frozen physical inputs,
validity, covariance, retained provenance bytes, an error budget, a production
integrator, and independent external qualification. Details and validation
boundaries are in [V5_REFERENCE_INTEGRATOR.md](V5_REFERENCE_INTEGRATOR.md).

The companion
[frozen Solar 1PN qualification package](../runs/v5_solar_1pn_qualification/README.md)
retains the first source and transformation inputs and preregisters Q0--Q9,
including fresh holdouts and the restricted finite-mass EIH correspondence.
It deliberately has no execution implementation or outcomes. Package validity
therefore means only that the future experiment was frozen consistently; it
does not alter this registry row, resolve a parameter, or authorize a force.

## Gravity harmonics and oblateness

The external potential of body \(b\) is declared using one coefficient
normalization and sign convention:

\[
V_b=\frac{\mu_b}{r}\left[
1+\sum_{\ell=2}^{L}\left(\frac{R_b}{r}\right)^\ell
\sum_{m=0}^{\ell}P_{\ell m}(\sin\phi)
(C_{\ell m}\cos m\lambda+S_{\ell m}\sin m\lambda)
\right].
\]

For unnormalized zonals, \(C_{\ell0}=-J_\ell\). Every coefficient artifact must
bind its gravitational parameter, reference radius, maximum degree/order,
normalization, associated-Legendre convention, potential sign, longitude and
latitude conventions, tide system, body-fixed orientation, epoch, coverage,
uncertainty/covariance, and byte hash. Changing the reference radius requires
rescaling degree \(\ell\) coefficients by \((R/R')^\ell\).

The model must distinguish solar \(J_2/J_4\), planetary zonals, and general
tesseral coefficients. It may be used only outside the declared convergence
domain of the external expansion. Body orientation follows the [NAIF PCK
contract](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html); data
metadata should preserve the [ICGEM format
conventions](https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf).

Validation includes direct Cartesian-gradient fixtures, pole rotations,
coefficient-normalization conversions, radius rescaling, and the first-order
\(J_2\) secular rates

\[
\dot\Omega=-\frac{3nJ_2}{2}\left(\frac{R}{p}\right)^2\cos i,
\qquad
\dot\omega=\frac{3nJ_2}{4}\left(\frac{R}{p}\right)^2(5\cos^2i-1).
\]

## Tides

JX keeps time-variable gravity coefficients separate from mutual equilibrium
tides and spin evolution. A constant-time-lag model is not interchangeable
with constant-\(Q\), frequency-dependent complex Love numbers, dynamical tides,
or disruption/contact physics.

Required parameters include mass, physical radius, Love-number convention,
spin vector, moment of inertia when spin evolves, time lag or frequency law,
tide-raiser/responder roster, backreaction, valid frequency/separation domain,
and collision/Roche cutoff. Missing spin or lag blocks execution instead of
defaulting to zero.

Tests include zero-lag conservative behavior, the circular aligned synchronous
fixed point, angular-momentum conservation with spin backreaction, mechanical
energy loss for positive dissipation, pseudo-synchronous equilibrium, and
radius/separation scaling. References include [Hut
1981](https://ui.adsabs.harvard.edu/abs/1981A%26A....99..126H/abstract) and
[Mignard 1979](https://doi.org/10.1007/BF00907581).

## Nongravitational forces

### Radiation pressure and Poynting–Robertson drag

For target–Sun relative state \((\mathbf r,\mathbf u)\), the first-order Burns,
Lamy and Soter model is

\[
\mathbf a_{rad}=
\frac{S(r)AQ_{pr}}{mc}
\left[
\left(1-\frac{u_r}{c}\right)\hat{\mathbf r}
-\frac{\mathbf u}{c}
\right],
\qquad S(r)=\frac{L_\odot}{4\pi r^2}.
\]

JX registers the cannonball SRP approximation separately from the combined
SRP/PR model. Required inputs include luminous source and luminosity,
area-to-mass ratio, \(Q_{pr}\) or an explicitly defined reflectivity
convention, projected-area/attitude model, occulting radii, shadow model,
units, frame, and validity. JPL Horizons `AMRAT` assumes total absorption and
must not be silently equated to another \(C_R\) or \(Q_{pr}\) convention.
Sources: [NASA Burns–Lamy–Soter record](https://ntrs.nasa.gov/citations/19790070469)
and the [JPL Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html).

### Yarkovsky effect

Three models remain separate:

1. empirical orbit-fit acceleration
   \(\mathbf a=A_2(r/1\,\mathrm{au})^{-d}\hat{\mathbf T}\);
2. a linear spherical thermophysical model; and
3. a nonlinear facet/shape thermophysical model.

The physical models require size/shape, mass and density, spin vector/period/
phase/epoch, conductivity, heat capacity, thermal inertia, emissivity, albedo,
roughness, layered-regolith policy, self-heating, shadowing, and orientation
history. An empirical \(A_2\) detection is not itself proof of a thermophysical
interpretation. State and \(A_2\) covariance must be preserved together.

Validation covers zero thermal lag, spin reversal, obliquity limits, the
published spherical solution, facet-to-sphere limits, and a preregistered
reproduction of a documented optical/radar solution such as the
[Bennu analysis](https://echo.jpl.nasa.gov/asteroids/chesley_etal_2014_Bennu.pdf).
The three model tiers and their physical inputs follow the distinctions in the
[Yarkovsky/YORP review](https://arxiv.org/abs/1502.01249); that discovery link
does not become verified provenance until its bytes are retained and hashed.

### Comet outgassing

The Marsden extended standard model is

\[
\mathbf a_{ng}=g(r')
(A_1\hat{\mathbf R}+A_2\hat{\mathbf T}+A_3\hat{\mathbf N}),
\]

\[
g(r)=\alpha(r/r_0)^{-m}[1+(r/r_0)^n]^{-k}.
\]

The registry stores every coefficient, the RTN basis convention, activity and
apparition interval, volatile species, and an explicit lag-sign convention.
The familiar Horizons constants are a semi-empirical water-ice model, not a
universal comet law. Rotating-jet/body-fixed outgassing is a different model
with pole, rotation, jet, illumination, thermal-lag, shape, and mass-loss
parameters. See the [JPL Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html)
and [Marsden–Sekanina–Yeomans](https://doi.org/10.1086/111402).

## Close encounters, collisions, and regularization

Hybrid encounter switching, collision detection, collision outcome, and
mathematical regularization are four separate contracts.

- MERCURIUS-like switching changes from a declared base map to a high-order
  encounter solver. TRACE-like switching adds time-reversible pair and central
  periapse criteria. Threshold equations, coordinates, step, solver tolerances,
  pair eligibility, synchronization, restart state, and exact software version
  are mandatory. See the [MERCURIUS paper](https://arxiv.org/abs/1903.04972)
  and [TRACE paper](https://arxiv.org/abs/2405.03800).
- Collision detection declares endpoint or swept geometry, event time, body
  radii/shapes, and simultaneous-event ordering. Resolution separately declares
  halt, bounce, merge, fragmentation, survivor identity, momentum/spin policy,
  restitution, and an energy-loss ledger.
- Levi–Civita, KS, chain, logarithmic-Hamiltonian, and time-transformed
  regularizations are distinct methods. Numerical continuation through
  \(r=0\) is not a physical collision outcome, and softening is a different
  force law. The algorithmic-regularization registry starts from
  [Mikkola and Tanikawa (1999)](https://doi.org/10.1046/j.1365-8711.1999.02982.x)
  as discovery metadata, without treating it as an implemented method.

Tests cover switch boundaries, analytic hyperbolic scattering, central
periapse cases, forward–reverse recovery where claimed, restart inside an
encounter, swept tunneling, collision-order permutations, head-on and oblique
momentum accounting, exact Kepler conics, and regularized/unregularized
convergence away from singularities.

## Observation and orbit determination

The initial frame chain is ITRS → GCRS → BCRS/ICRS. UTC, UT1, TT, TCG, TDB and
TCB conversions are explicit. Station coordinates and velocities, Earth
orientation, polar motion, precession/nutation, site displacement, clocks, and
media corrections are registered artifacts.

The first-order relativistic signal equation includes geometric light time and
gravitational delay:

\[
t_2-t_1=\frac{\|\mathbf x_2(t_2)-\mathbf x_1(t_1)\|}{c}
+\sum_J\frac{2GM_J}{c^3}
\ln\left(\frac{r_{J1}+r_{J2}+\rho}
{r_{J1}+r_{J2}-\rho}\right).
\]

Reception, transmission, and round-trip observables solve participant states
at different epochs. Light time, stellar aberration, gravitational deflection,
and gravitational delay are distinct corrections. NAIF/SPICE ordinary light
time and stellar aberration explicitly omit relativistic bending and delay;
they are validation components, not the whole observation model. See the
[NAIF aberration specification](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/abcorr.html)
and [JPL DSN formulation](https://descanso.jpl.nasa.gov/monograph/series2_section.html).

The first observable roster includes optical angles, radar round-trip delay and
Doppler, topocentric range and range rate. Estimation requires the state
transition matrix, parameter sensitivities, observation covariance and
correlations, priors/consider parameters, explicit outlier policy, rank and
condition diagnostics, residual analysis, and held-out prediction.

Covariance artifacts must state epoch, ordered parameter list, units, frame,
perturbation model, and positive-definiteness evidence, following the
[CCSDS Orbit Data Message requirements](https://ccsds.org/Pubs/502x0b3e1.pdf).

## Parameter and provenance rules

Every scientific value has:

- stable identifier and physical role;
- value kind and canonical decimal or immutable artifact representation;
- unit, frame, origin, epoch, and validity interval;
- source and target bodies;
- evidence class and locally retained source artifact;
- source byte size and SHA-256;
- uncertainty type, confidence meaning, and covariance membership;
- fixed, estimated, solved-for, or model-selection role;
- applicability and omitted-term error budget.

Internet links are discovery metadata only. A source becomes `VERIFIED` only
after its exact bytes are retained, hashed, and its relevant equations or data
locations are recorded. A value copied from a webpage without that binding
remains `TBD_BLOCKED`.

## Validation ladder

1. **Schema:** exact keys, canonical JSON, unique identifiers, and closed
   cross-references.
2. **Equation:** independent arbitrary-precision acceleration and limiting-case
   fixtures.
3. **Component:** analytic secular effects, conservation or dissipation laws,
   frame covariance, and domain boundaries.
4. **Numerical method:** convergence, restart identity, close-encounter and
   velocity-dependent-force compatibility.
5. **External reference:** held-out, hash-pinned SOFA, SPICE, Horizons, JPL
   ephemeris, or published case fixtures without refitting.
6. **Observation:** synthetic parameter recovery, covariance coverage, real
   residuals, held-out prediction, and model-subset sensitivity.
7. **Independent review:** separate implementation and external scientific
   reproduction.

Passing a lower rung never waives a higher one.

## Current execution boundary

The checked-in V5 registry is intentionally nonexecuting. It may be inspected
and validated, but it cannot authorize a scientific run. The following remain
blocked:

- locally pinned parameter and coefficient artifacts with uncertainty;
- exact frame/time and compatible-state selection;
- a declared Decimal precision and rounding context for every authorized run;
- a registry-authorizing production integrator for velocity-dependent
  relativistic forces and an independently qualified external reference arm;
- N-body EIH and independent acceleration fixtures;
- body orientation and harmonic coverage;
- per-body tide laws and spin backreaction;
- nongravitational target applicability and joint covariance;
- collision outcomes and regularization compatibility;
- observation synthesis, estimation, and external residual validation.

Until those gates pass, JX may say that a model contract, equation-level
implementation, and bounded nonauthorizing numerical reference exist. It may
not claim authorized physical propagation, DE440/DE441 reproduction,
ephemeris-grade accuracy, orbit determination, a physical detection or
exclusion, or superiority over another system.
