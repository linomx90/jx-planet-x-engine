# JX Public Research Study: Requirements for a Top-Tier Celestial-Dynamics Engine

Status: research baseline, not a capability claim  
Scope: public papers, official documentation, engineering requirements, and a staged path for JX  
Date: 2026-08-30

## 1. Executive conclusion

There is no single "best" celestial-dynamics engine. Different engines lead different problem classes:

- REBOUND leads as a flexible research N-body framework with several complementary integrators.
- IAS15 is a strong reference for adaptive, high-accuracy integration, including close encounters and non-conservative forces.
- WHFast and SABA-type schemes are designed for fast, long-term integration of hierarchical planetary systems.
- MERCURIUS and TRACE address systems that are mostly hierarchical but sometimes experience close encounters.
- ASSIST adds ephemeris-quality Solar System forces for test particles.
- GENGA targets large GPU workloads involving planet formation, planetesimals, and many related systems.
- Orekit, Tudat, GMAT, and Basilisk cover spacecraft propagation, frames, measurements, estimation, events, and mission operations.
- Bonsai and NBODY6++GPU address large star-cluster or galaxy simulations, a different numerical regime from precision planetary dynamics.

JX should therefore become a portfolio, not one oversized solver. Its foundation should be a strict scientific contract shared by multiple integrators and CPU/GPU backends. Every run must declare its state, units, time scale, frame, force models, numerical controls, source data, and validation status.

The correct order is: learn, specify, build a trusted CPU reference, validate, then accelerate. A GPU does not repair an incomplete physical model or an unverified integrator.

## 2. What "top-tier" means

A top-tier engine is not defined only by speed or a long force list. It must be:

1. **Correct within a declared domain:** equations, approximations, and exclusions are explicit.
2. **Numerically appropriate:** the integrator matches the system rather than being used universally.
3. **Reproducible:** inputs, software version, kernels, constants, hardware, and numerical settings are recorded.
4. **Validated by components:** forces, integrators, events, derivatives, and backends have independent tests.
5. **Honest about uncertainty:** truncation, floating-point, ephemeris, measurement, and model errors are separated.
6. **Interoperable:** time, frames, units, ephemerides, and output formats have unambiguous metadata.
7. **Performant at the right scale:** CPU, SIMD, GPU, and distributed modes are chosen from measured workload thresholds.
8. **Legally reusable:** implementation provenance and third-party licenses are tracked.

Recommended public status labels for every JX capability:

- **Research:** equations or design studied; no implementation claim.
- **Experimental:** implemented, but validation is incomplete or the interface may change.
- **Validated:** predefined tests and error limits pass within a stated domain.
- **Qualified:** independently reproduced on specified platforms and data.
- **Operational:** release, documentation, monitoring, and backward-compatibility commitments exist.

No global "top-tier" label should hide a weak subsystem. Status belongs to a specific model, integrator, backend, and problem domain.

## 3. Engine landscape

| Engine or method | Best-fit problem | Main lesson for JX | Important boundary |
|---|---|---|---|
| REBOUND | General research N-body work | Clean simulation model plus interchangeable integrators | GPL implementation cannot be copied into MIT JX |
| IAS15 | High accuracy, eccentric motion, close encounters, velocity-dependent forces | Adaptive reference lane and careful roundoff control | More expensive than a symplectic method for quiet long runs |
| WHFast | Long-term hierarchical planetary systems | Fast fixed-step symplectic lane with unbiased substeps | Not the universal answer for strong encounters or arbitrary forces |
| SABA family | High-order planetary symplectic integration | Problem-specific splitting can outperform generic solvers | Assumptions about Hamiltonian structure must remain true |
| MERCURIUS | Mostly hierarchical systems with occasional encounters | Hybrid switching between fast and accurate modes | Switching parameters and transition behavior require validation |
| TRACE | Reversible hybrid close-encounter work | Treat reversibility and central-body encounters deliberately | Newer method with a narrower public evidence base |
| MERCURY | Classic hybrid planetary integration | Historical benchmark and encounter design reference | Later work reports reproducibility and implementation inconsistencies |
| SWIFT / Swiftest | Planetary symplectic and encounter studies | Multiple coordinate choices, OpenMP, structured output | Swiftest is GPL-3.0 |
| JANUS | Bit-wise reversible long-term experiments | Exact replay can be a design goal | Specialized; velocity-dependent forces are restricted |
| AR-CHAIN | Few-body extreme encounters and mass ratios | Regularization is its own solver family | Not a general large-N replacement |
| ASSIST | Ephemeris-quality small-body test particles | Combine a trusted integrator, ephemeris, harmonics, GR, and nongravitational forces | GPLv3; focused on test particles |
| GENGA | GPU planet formation and large planetesimal workloads | Keep work resident on GPU and design encounter handling for parallelism | GPU value depends strongly on problem size |
| Orekit / Tudat | Spacecraft propagation and orbit determination | State, frames, events, forces, derivatives, and measurements need first-class interfaces | Different focus from massive N-body planet formation |
| GMAT / Basilisk | Mission design, navigation, spacecraft dynamics, and GNC | Operational workflows need provenance, events, estimation, and subsystem tests | Not intended as a single universal astrophysical engine |
| Bonsai / NBODY6++GPU | Galaxies and star clusters | Tree/direct GPU algorithms serve large-N regimes | Approximation and error goals differ from Solar System ephemerides |

## 4. Integrator portfolio

JX should expose several solver families behind one state-and-force contract:

- **Adaptive high-accuracy reference:** Use an IAS15-like high-order adaptive lane, or another independently implemented and validated high-order method, for close encounters, high eccentricity, velocity-dependent forces, and reference comparisons. Controls include absolute/relative tolerance, step bounds, rejection limits, dense output, iteration convergence, compensated summation, and explicit failures.
- **Long-term symplectic lane:** Use a WHFast/SABA-like fixed-step lane for hierarchical systems whose Hamiltonian can be split correctly. Controls include coordinates, splitting, step, corrector order, synchronization, Kepler-solver tolerance, safe mode, and compatibility warnings.
- **Hybrid close-encounter lane:** Use a MERCURIUS/TRACE-like lane when normally hierarchical systems sometimes encounter. Controls include encounter-radius definition, Hill multiplier, switching function, transition width, encounter grouping, central-body treatment, sub-integrator, collision rules, and switch diagnostics.
- **Regularized few-body lane:** Reserve chain or algorithmic regularization for compact subsystems, extreme mass ratios, binaries, and near-singular passages. Define entry/exit and how energy, momentum, time, and derivatives cross the boundary.
- **Independent oracle lane:** Maintain a different algorithm and preferably different software lineage. High or arbitrary precision is valuable for small cases. Two wrappers around the same solver are not independent confirmations.

Integrator selection must be explicit. JX should reject incompatible combinations rather than silently producing plausible numbers.

## 5. Complete parameter architecture

Every run should serialize the following parameter families.

### 5.1 Identity and provenance

- run identifier, scenario identifier, creation time, operator, and purpose;
- JX version, commit, build options, dependency versions, and license manifest;
- CPU/GPU model, driver/runtime, compiler, precision mode, and deterministic-mode flag;
- random algorithm and seeds;
- input-file hashes, ephemeris/kernel hashes, constants version, and configuration hash.

### 5.2 State definition

- body identifier, role, and provenance;
- active massive body, passive test particle, spacecraft, field source, or massless marker;
- epoch and time scale;
- reference frame, origin, orientation, and center convention;
- position and velocity, with units;
- mass and gravitational parameter, with a declared precedence rule if both are supplied;
- radius, shape, density, inertia tensor, attitude, angular velocity, and spin axis when used;
- covariance and parameter correlations when the state is estimated;
- physical, optical, thermal, atmospheric, and material properties required by enabled forces.

### 5.3 Numerical controls

- integrator name and version;
- step or tolerance controls and error norm;
- precision and summation method;
- coordinate/splitting choice;
- iteration and convergence limits;
- dense-output and interpolation order;
- event-location tolerance and ordering;
- close-encounter and regularization controls;
- parallel reduction policy, device partitioning, checkpoint cadence, and restart policy.

### 5.4 Force configuration

Each force instance needs a unique identifier, model citation, model version, source bodies, affected bodies, parameters with units, valid epoch/range, reference frame, approximation flags, exclusions, differentiability status, and validation status.

### 5.5 Output and acceptance

- requested epochs or cadence, state representation, frame, time scale, and units;
- events, osculating elements, invariants, residuals, and uncertainty products;
- tolerances for conservation, reference differences, derivatives, and replay;
- reason for termination and whether the run is scientifically acceptable;
- warnings, extrapolation flags, missing-data flags, and fallback decisions.

## 6. Physics-model requirements and parameters

### 6.1 Newtonian gravity

Required parameters are gravitational constant or gravitational parameters, masses, source/target roles, softening policy, collision radii, and summation policy. Direct summation, tree approximation, multipole approximation, and test-particle optimization must be different named models; none may be silently substituted.

### 6.2 Relativity

JX should distinguish:

- central-body Schwarzschild or post-Newtonian approximation;
- full first post-Newtonian N-body/EIH terms;
- higher-order conservative PN terms where justified;
- 2.5PN radiation reaction for compact systems;
- Lense-Thirring/frame-dragging terms;
- relativistic time or light-time models used by observations.

Parameters may include speed of light, gravitational parameters and masses, PPN coefficients, source spin vector, moment of inertia, approximation order, coordinate convention, included body pairs, and iteration tolerance for velocity-dependent terms. A simple precession potential must not be labeled full general relativity.

### 6.3 Gravity harmonics and figure effects

Parameters include reference radius, normalized or unnormalized coefficients, degree/order, normalization convention, body-fixed frame, orientation model, spin state, and coefficient epoch. Simpler models may use J2/J4 and a pole vector; higher fidelity requires spherical harmonics and time-dependent orientation.

### 6.4 Tides and rotational coupling

Parameters include radius, Love numbers, time lag or quality-factor convention, spin state, inertia, tidal source/target, frequency dependence, dissipation model, and dynamical-mode state if used. Constant-time-lag and constant-Q models are not interchangeable.

### 6.5 Radiation forces

Solar-radiation pressure needs source luminosity or flux, occultation model, area, mass, reflectivity coefficient, surface/attitude model, and reference distance. Poynting-Robertson drag additionally needs the speed of light and the radiation-to-gravity ratio or equivalent optical properties.

### 6.6 Thermal recoil and Yarkovsky

Parameters can include source luminosity, radius, density, albedo, emissivity, thermal conductivity or inertia, heat capacity, rotation period/rate, spin-axis vector, surface model, speed of light, and Stefan-Boltzmann constant. Diurnal and seasonal components must be identified separately.

### 6.7 Atmosphere, gas, and drag

Parameters include density model and provenance, reference density/altitude, scale height, temperature, composition, winds or gas velocity, sound speed, body area, mass, drag coefficient, attitude, disk radial slopes, aspect ratio, and Coulomb-log terms where dynamical friction is modeled.

### 6.8 Outgassing, mass change, and propulsion

Cometary acceleration requires an activity law, heliocentric-distance dependence, radial/transverse/normal coefficients, reference epoch, delay/asymmetry, body-fixed orientation if used, and parameter units. Propulsion requires thrust vector/profile, mass-flow or specific impulse, attitude/pointing, start/stop events, interpolation, and saturation. Variable mass must be integrated consistently with momentum.

### 6.9 Migration and empirical accelerations

Parameters include semimajor-axis, eccentricity, and inclination damping timescales; coordinate convention; disk properties; activation windows; and empirical radial/transverse/normal coefficients. These models must be labeled phenomenological when they are not derived from a resolved environment.

## 7. Frames, time, units, and ephemerides

An engine state is incomplete without target, observer/origin, epoch, time scale, axes, orientation, and units.

- Use explicit inertial, rotating, body-fixed, topocentric, or barycentric frames.
- Record the orientation realization and data version, not just a short frame name.
- Distinguish UTC, TAI, TT, TDB/ephemeris time, and other scales. Leap seconds are data, not an arithmetic constant.
- Keep internal dynamics time separate from human-readable civil time.
- Record transformations and Earth-orientation data used by ground-based observations.
- Treat nominal conversion constants separately from measured physical parameters.
- Version and hash SPICE kernels or JPL ephemeris files; record covered epochs and extrapolation behavior.
- Do not mix heliocentric, barycentric, planet-centered, osculating, and mean elements without explicit conversion metadata.

DE440 and DE441 serve different spans and accuracy goals. Ephemeris choice is part of the physical model, not an invisible lookup detail. Standard exchange should support CCSDS Orbit Data Messages where appropriate.

## 8. Collisions, encounters, and events

Required event types include collision/contact, close approach, periapsis/apoapsis, plane or surface crossing, eclipse/occultation, sphere-of-influence entry, escape/ejection, user-defined scalar roots, and propagation termination.

Every event needs a direction, activation window, root tolerance, priority, simultaneous-event rule, and terminal/nonterminal flag. Collision handling must name the model: stop, merge, bounce, fragment, erode, or callback. It must define radii/shape, restitution, friction, mass/momentum transfer, spin change, fragment distribution, minimum resolved mass, and conservation accounting.

Encounter logic must not depend only on output cadence. Detection, root finding, and collision resolution need tests for steps that cross an event without sampling it directly.

## 9. Orbit determination and variational equations

A complete exploration engine eventually needs sensitivities, not only trajectories.

- Propagate a state-transition matrix and parameter partial derivatives.
- Define solve-for parameters, fixed parameters, scaling, priors, bounds, and correlations.
- Support observation epochs, station states, measurement types, reference frames, media/light-time corrections, biases, weights, and covariance.
- Provide batch least squares and, later, sequential filtering/smoothing with process noise.
- Support single-arc, multi-arc, and estimated maneuver/discontinuity handling.
- Report residuals, normalized residuals, covariance, condition indicators, rejected observations, and convergence history.
- Verify analytic/automatic derivatives against finite differences or another independent method.

Variational equations must use exactly the same enabled force models and switching logic as the nominal trajectory. A force without validated derivatives should declare that limitation.

## 10. GPU and HPC requirements

GPU support should be a backend, not a separate scientific definition.

1. Keep state and intermediate data resident on the device across many steps.
2. Use structure-of-arrays layouts, coalesced access, batched systems, and minimized host-device synchronization.
3. Separate massive-active interactions from passive test-particle work.
4. Benchmark direct O(N^2), neighbor/encounter methods, and tree/multipole methods in their valid error regimes.
5. Define floating-point precision by operation; mixed precision requires an error budget.
6. Use stable reductions and document whether results are deterministic, statistically reproducible, or neither.
7. Match CPU and GPU force contracts and compare both against the independent oracle.
8. Detect unsupported hardware and fail clearly; never claim GPU execution after silent CPU fallback.
9. Record driver, runtime, device, kernel/build, launch parameters, and multi-device decomposition.
10. Measure transfer time, initialization, throughput, latency, memory, energy if available, and accuracy—not only kernel speed.

Small planetary systems may be faster on a CPU because launch and synchronization overhead dominate. SIMD methods such as WHFast512 show that CPU vectorization remains important. GPU priority should begin with many independent systems, large test-particle populations, or sufficiently large active-body counts.

## 11. Validation and benchmark matrix

JX should publish predefined cases and pass/fail thresholds for:

| Layer | Minimum evidence |
|---|---|
| Units, time, frames | Round trips, authoritative reference vectors, leap-second and boundary cases |
| Individual forces | Analytic limits, hand calculations, published cases, symmetry and dimension checks |
| Integrator order | Step refinement and observed convergence order |
| Long-term behavior | Energy, angular momentum, Jacobi constant, phase, and secular-frequency errors |
| Encounters | Two-body scattering, central encounter, collision crossing, repeated switching, extreme mass ratios |
| Relativity | Perihelion/precession and independent 1PN comparisons within the declared approximation |
| Ephemerides | State comparisons at sampled epochs against the exact kernel/version used |
| Variational equations | Finite-difference or independent derivative comparisons and covariance consistency |
| CPU/GPU parity | Force, event, and trajectory differences under an explicit error budget |
| Reproducibility | Checkpoint/restart, archive replay, thread/device-count policy, environment manifest |
| Performance | Reproducible accuracy-versus-cost curves, not isolated steps per second |

Energy conservation alone does not prove a correct trajectory, especially in chaotic systems. Validation must examine phase, invariants, events, statistical ensemble behavior, and comparisons with an independent method. IAS15's separation of scheme, roundoff-floor, random-walk, and biased error is a useful model for JX error budgets.

Release evidence should contain machine-readable inputs, expected outputs, tolerances, data hashes, software/hardware metadata, plots or tables, and the exact command or API call. Failed cases remain visible.

## 12. Licensing and clean-room reuse

JX declares an MIT license. Public availability does not mean code may be copied into MIT software.

- REBOUND, REBOUNDx/ASSIST, and Swiftest use GPL-family licenses; study their papers and public interfaces, but do not copy implementation into MIT JX.
- Orekit and GMAT use Apache-2.0; Basilisk uses ISC; heyoka uses MPL-2.0. Each still needs notice, compatibility, and file-level review before reuse.
- A project described only as "free to use" must be treated as license-unclear until an exact license is located.
- Mathematical ideas and published equations can inform an independent implementation, but comments, tests, constants tables, control flow, and code structure can also carry provenance concerns.
- Keep a design notebook linking each equation to a paper/standard, record who implemented it, and require reviewers to attest that GPL source was not copied.
- Prefer black-box comparison, published formulas, and independently written tests. Optional interoperability with a GPL program should use a clearly separated process and receive legal review.

This is an engineering safeguard, not legal advice. `THIRD_PARTY_NOTICES`, dependency locks, source-data licenses, and SPDX identifiers should be release gates.

## 13. Staged JX roadmap

1. **Freeze claims and finish the scientific contract:** Publish problem domains, state schema, units/time/frame rules, capability labels, force registry, provenance format, and acceptance metrics. Exit when a scenario can be reconstructed without private knowledge.
2. **Trusted CPU reference:** Build or select an independently licensed adaptive reference path with Newtonian gravity, event location, checkpoints, and full manifests. Pass two-body, restricted three-body, encounter, convergence, and restart tests.
3. **Long-term planetary portfolio:** Add a clean-room symplectic path and later a hybrid encounter path. Compare accuracy versus cost across quiet, eccentric, resonant, and encounter cases.
4. **Physical-model library:** Add harmonics, 1PN variants, radiation/P-R drag, tides, Yarkovsky, drag, outgassing, and thrust one at a time. Each needs parameters, units, domain, source, derivatives, tests, and incompatibility rules.
5. **Frames, ephemerides, and observations:** Integrate versioned SPICE/JPL data, Earth orientation where needed, standard outputs, observation models, and light-time corrections. Pass authoritative state/frame/time comparisons.
6. **Variational and orbit-determination layer:** Add state-transition and sensitivity propagation, batch estimation, covariance, residuals, multi-arc handling, and independent derivative tests.
7. **GPU backend:** Port only measured bottlenecks behind the same force contract, beginning with batched systems and passive particles. Pass CPU/GPU error budgets, deterministic-policy tests, and crossover measurements.
8. **Encounters, collisions, and population scale:** Add regularization, collision/fragmentation, parallel encounter grouping, and large-N approximations only for declared domains. Validate conservation and population statistics.
9. **Public release qualification:** Publish documentation, examples, benchmark archives, known limitations, license provenance, reproducible builds, and third-party reproduction instructions. Evidence—not a version number—establishes qualification.

## 14. Public primary-source reading list

**Core engines and integrators:** [REBOUND repository/license](https://github.com/hannorein/rebound); [REBOUND paper](https://arxiv.org/abs/1110.4876); [IAS15](https://arxiv.org/abs/1409.4779); [WHFast](https://arxiv.org/abs/1506.01084); [high-order symplectic methods](https://arxiv.org/abs/1907.11335); [MERCURIUS documentation](https://rebound.hanno-rein.de/integrators/mercurius/) and [paper](https://academic.oup.com/mnras/article/485/4/5490/5380811); [TRACE](https://arxiv.org/abs/2405.03800); [MERCURY](https://academic.oup.com/mnras/article/304/4/793/1047461); [SWIFT](https://www2.boulder.swri.edu/~hal/swift.html); [Swiftest](https://github.com/carlislewishard/swiftest); [JANUS](https://academic.oup.com/mnras/article/473/3/3351/4243609); [AR-CHAIN](https://arxiv.org/abs/0709.3367); [Brutus](https://arxiv.org/abs/1411.6671); [heyoka](https://arxiv.org/abs/2105.00800); [NbodyGradient.jl](https://arxiv.org/abs/2106.02188).

**Extended physics and ephemerides:** [REBOUNDx effects](https://reboundx.readthedocs.io/en/latest/effects.html) and [paper](https://arxiv.org/abs/1908.05634); [ASSIST paper](https://arxiv.org/abs/2303.16246) and [documentation](https://assist.readthedocs.io/en/latest/); [JPL DE440/DE441 description](https://ssd.jpl.nasa.gov/doc/de440_de441.html) and [paper](https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf); [Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html); [SPICE concepts](https://naif.jpl.nasa.gov/naif/spiceconcept.html), [frames](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/frames.html), and [time](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html); [IAU SOFA](https://www.iausofa.org/); [IERS Conventions](https://www.iers.org/iers/en/dataproducts/conventions/conventions); [IAU nominal constants](https://www.iau.org/static/resolutions/IAU2015_English.pdf); [CCSDS Orbit Data Messages](https://ccsds.org/Pubs/502x0b3e1.pdf).

**GPU, spacecraft, and verification:** [GENGA](https://arxiv.org/abs/1404.2324); [GENGA II](https://arxiv.org/abs/2201.10058); [Bonsai](https://arxiv.org/abs/1204.2280); [NBODY6++GPU](https://arxiv.org/abs/1504.03687); [WHFast512](https://arxiv.org/abs/2307.05683); [Orekit propagation](https://www.orekit.org/site-orekit-13.1/architecture/propagation.html); [Tudat propagation](https://docs.tudat.space/en/stable/user-guide/state-propagation/propagation-setup.html); [NASA GMAT](https://github.com/nasa/GMAT); [Basilisk](https://github.com/AVSLab/basilisk) and its [validation checklist](https://avslab.github.io/basilisk/Support/Developer/bskModuleCheckoutList.html); [AMUSE](https://github.com/amusecode/amuse); [REBOUND SimulationArchive](https://academic.oup.com/mnras/article/467/2/2377/2961801).

## 15. Immediate decision for JX

Do not start by promising every force on every GPU. First approve this research baseline, define JX's first supported problem class, and choose one reference benchmark suite. The recommended first target is high-accuracy Solar System and small-body propagation on CPU with explicit frames, time, ephemerides, Newtonian gravity, a clearly named 1PN model, events, trajectory output, and reproducible archives. Long-term symplectic and GPU lanes should follow only after the reference contract is stable.
