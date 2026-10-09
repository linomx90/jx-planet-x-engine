# Trans-Neptunian Objects: research dossier and JX roadmap

**Evidence date:** 2026-10-08

**Scope:** trans-Neptunian objects (TNOs), the Kuiper belt, scattered and detached populations, physical properties, formation, surveys, open questions, and the implications for the JX project.

**Claim boundary:** this is a literature and workspace synthesis, not a new astronomical result. Rapidly changing orbit solutions and discovery counts should be read from the Minor Planet Center (MPC) and JPL at the time of analysis.

## Executive findings

1. **A TNO is an orbital category, not one kind of world.** Operationally it is usually a minor body whose semimajor axis is beyond Neptune's, about 30.1 AU. The strict population contains dynamically cold and hot classical objects, Neptune resonators, actively scattering objects, detached objects, and very distant high-perihelion bodies. Neptune Trojans are a closely related co-orbital population that the MPC classifies separately; Centaurs and the Oort cloud are related reservoirs, but are not interchangeable with the strict TNO population.

2. **TNOs are the best surviving record of outer Solar System formation.** The cold classical belt is probably a comparatively undisturbed, in-situ remnant. Much of the dynamically hot, resonant, and scattering population was implanted or reorganized while Neptune migrated. The mixture of primordial and implanted bodies is a central conclusion of the modern review literature, while the cold belt's sharp outer edge and the origin of high-perihelion detached objects remain unresolved ([Gladman & Volk 2021](https://doi.org/10.1146/annurev-astro-120920-010005)).

3. **The detected catalog is a strongly filtered sample.** A TNO's reflected flux falls approximately as `r^-4` near opposition, so surveys strongly favor objects near perihelion, at smaller distance, in their sky footprint, and within their rate and magnitude cuts. Raw MPC histograms cannot establish an intrinsic population or angular clustering. A characterized survey and its forward selection simulator are required ([Lawler et al. 2018](https://doi.org/10.3389/fspas.2018.00014)).

4. **Orbits are classifications with uncertainty, not permanent labels.** Resonance requires libration of a critical angle, and scattering is normally diagnosed from numerical evolution. A nominal semimajor axis alone is insufficient. Classification should propagate the full orbital covariance through clones and integrate the known planetary system over a declared interval.

5. **TNOs are physically diverse.** Their surfaces contain water ice, carbon dioxide, carbon monoxide, methanol, methane or nitrogen on some large bodies, and complex carbon-, oxygen-, and nitrogen-bearing material. JWST spectra resolve at least three broad compositional groups rather than one universal “dirty ice” surface ([Pinilla-Alonso et al. 2025](https://doi.org/10.1038/s41550-024-02433-2)). Binaries, contact binaries, rings, atmospheres, and active geology all occur.

6. **Planet Nine is only one TNO hypothesis.** It is unconfirmed. Selection-aware samples have found no significant angular clustering in some tests, while other dynamical studies report model-dependent anomalies. A defensible TNO program must first establish the known-planets baseline, orbit covariances, survey selection, and alternative formation mechanisms; it must not treat Planet Nine as the default explanation.

7. **The strongest immediate JX project is a TNO atlas and dynamical classifier.** The workspace already has byte-verified OSSOS and DES inventories and digests, a current detached-object inventory, strong integrators, provenance controls, and a qualified historical survey adapter. The large OSSOS/DES catalog payloads themselves are not retained locally and must be reacquired and hash-verified. The workspace does **not** yet have an execution-ready full-survey selection bundle, a general TNO covariance-clone classifier, or a physical-properties database. Those are the shortest path to useful, Planet-Nine-independent astronomy, subject to a new TNO-specific scientific contract.

## 1. What “trans-Neptunian” means

The common operational definition is a minor planet with semimajor axis `a` greater than Neptune's. The MPC's current osculating-element classifier uses `a >= 30.178 AU` ([MPC orbit-type definition](https://docs.minorplanetcenter.net/mpc-ops-docs/orbits/orbit-types/)). This definition concerns the **average scale of the orbit**, not the body's instantaneous position. An eccentric TNO can temporarily be inside Neptune's orbital distance, and a comet from the Oort cloud can be far outside Neptune without belonging to the dynamical populations normally meant by “the trans-Neptunian region.”

“Kuiper belt object” and “TNO” are often used loosely as synonyms, but useful precision is:

- **TNO:** broad orbital category beyond Neptune.
- **Kuiper belt:** the structured disk dominated by classical and resonant populations, with its main region roughly 30–50 AU.
- **Scattered disk:** dynamically active population coupled to Neptune, extending to much larger aphelia.
- **Detached population:** high-perihelion objects not presently scattering strongly from Neptune.
- **Centaurs:** transient giant-planet-crossing or giant-planet-region bodies, probably supplied largely from trans-Neptunian reservoirs; most strict definitions do not count them as TNOs.
- **Oort cloud:** much more distant, approximately spherical comet reservoir; it is not simply the outer Kuiper belt.

NASA's broad overview places the main belt from Neptune's orbit to about 50 AU and the overlapping scattered disk outward to nearly 1,000 AU, with some orbits extending farther ([NASA Kuiper Belt facts](https://science.nasa.gov/solar-system/kuiper-belt/facts/)). The boundary between “detached,” “extreme TNO,” “sednoid,” and “inner Oort cloud” is not universal, so every analysis must publish its exact cuts.

### Dwarf planets are a different classification

“TNO” describes an orbit; “dwarf planet” describes a body's physical/gravitational status under the IAU definition. The IAU requires a dwarf planet to orbit the Sun, be massive enough to be nearly round, not have cleared its orbital neighborhood, and not be a satellite. Pluto is the prototype of a trans-Neptunian dwarf-planet category ([IAU 2006 Resolution 5A/6A](https://www.iau.org/IAU/Iau/News/PR2006/iau-2006-general-assembly-resolution-votes.aspx)). Pluto, Eris, Haumea, and Makemake are IAU-recognized trans-Neptunian dwarf planets. Large objects such as Quaoar, Gonggong, Orcus, and Sedna are often called dwarf-planet candidates in the literature, but that wording should not be upgraded to formal IAU status without an explicit source.

## 2. Orbital language and useful scale relations

For a heliocentric or barycentric Keplerian orbit:

- semimajor axis: `a`
- eccentricity: `e`
- perihelion: `q = a(1-e)`
- aphelion: `Q = a(1+e)`
- inclination: `i`
- longitude of ascending node: `Omega`
- argument of perihelion: `omega`
- longitude of perihelion: `varpi = Omega + omega`
- mean anomaly: `M`, which specifies orbital phase at a stated epoch

With `a` in AU and period `P` in years, `P ≈ a^(3/2)`. Thus a circular orbit at 40 AU takes about 253 years, one at 50 AU about 354 years, and an `a = 500 AU` detached orbit about 11,200 years. A few years of astrometry therefore observe only a tiny fraction of a distant orbit; phase, covariance, reference frame, epoch, and time scale are essential parts of an orbit record.

An approximate zero-albedo, globally reradiating (isothermal) equilibrium temperature is `T ≈ 278 K / sqrt(r/AU)`: about 51 K at 30 AU, 44 K at 40 AU, and 39 K at 50 AU. Albedo, spin, thermal inertia, latitude, and volatile transport can change the actual surface temperature; the subsolar equilibrium temperature is higher.

Near opposition, reflected flux is approximately proportional to `p D^2/(r^2 Delta^2)`, where `p` is albedo, `D` diameter, `r` heliocentric distance, and `Delta` observer distance. Because `r ≈ Delta` for a distant TNO, brightness is roughly proportional to distance to the negative fourth power. This creates a powerful perihelion and distance bias.

## 3. Dynamical populations

No single table of element cuts replaces numerical classification. The following is the conceptual map.

| Population | Operational dynamical signature | Scientific meaning and cautions |
|---|---|---|
| Neptune Trojans (closely related; separate MPC class) | 1:1 mean-motion resonance with Neptune; resonant angle librates around a stable Lagrange region | Co-orbitals often studied alongside TNOs that constrain capture and migration. Stability must be integrated. |
| Resonant TNOs | A Neptune mean-motion-resonant critical angle librates | Includes 3:2 plutinos, 2:1 twotinos, and many more distant resonances. Proximity to the nominal resonance in `a` is not proof. |
| Classical belt | Nonresonant and nonscattering over the classification integration | Commonly divided into inner, main, and outer regions, then dynamically cold and hot components. Exact cuts vary. |
| Cold classical | Low dynamical excitation, concentrated roughly from 42.4 AU to Neptune's 2:1 resonance at 47.7 AU | “Cold” means low inclination/eccentricity, not low temperature. A hard `i < 5 deg` cut is only a proxy because hot and cold distributions overlap. Generally redder and richer in wide, similar-size binaries; likely largely in-situ. |
| Hot classical | Broader inclination/eccentricity distribution | Likely substantially implanted during migration; “hot” is dynamical. |
| Scattering | Orbit remains coupled to Neptune; a common Gladman-style test is a semimajor-axis change of at least 1.5 AU over 10 Myr | Dynamically evolving source for Centaurs and short-period comets. The interval and threshold must be stated. |
| Detached | Nonresonant, nonscattering, and sufficiently high perihelion to be weakly coupled to Neptune | Definitions differ. Origins can include early migration, resonance/Kozai evolution, stellar encounters, cluster tides, rogue planets, or an additional surviving planet. |
| Extreme TNO / sednoid | Informal high-`a`, high-`q` subsets of detached space | Cuts are author-dependent. Sedna-like objects are especially valuable but extremely selection-biased. |
| Centaur | Giant-planet-region, typically planet-crossing or with `a`/`q` between Jupiter and Neptune | A related transient population rather than a strict TNO class. Definitions vary between catalogs. |
| Inner Oort / Oort cloud | Much more weakly bound orbits whose evolution can be dominated by Galactic tide and stellar passages | A separate reservoir. The boundary with the most distant detached TNOs is model-dependent. |

The standard Gladman classification philosophy uses present short-term dynamics and a 10 Myr numerical integration rather than assuming an object's origin or future ([Gladman, Marsden & VanLaerhoven 2008](https://web.gps.caltech.edu/~mbrown/out/kbbook/Chapters/Gladman_Nomenclature.pdf)). A 10 Myr label is not proof of 4.5 Gyr stability: resonance borders can be chaotic, scattering bodies can temporarily “stick” in resonances, and weak Neptune kicks can diffuse even very large-`a` orbits. A robust implementation should:

1. start from a barycentric Cartesian state at a declared TDB epoch;
2. draw clones from the full six-dimensional covariance, not independent element error bars;
3. integrate the nominal state and clones with the same known-planet model;
4. test close encounters and change in semimajor axis;
5. search appropriate resonance angles for sustained libration;
6. report secure, insecure, or clone-dependent classification rather than forcing one label.

Important Neptune resonance locations are approximately 1:1 at 30.1 AU, 3:2 at 39.4 AU, 5:3 at 42.3 AU, 7:4 at 43.7 AU, 2:1 at 47.7 AU, 5:2 at 55.4 AU, 3:1 at 62.6 AU, 4:1 at 75.8 AU, and 9:1 near 130 AU. These are navigation markers in semimajor-axis space, not classifications: a resonant angle must librate, and temporary resonance sticking can mimic long residence.

## 4. How TNOs are discovered and measured

### Discovery and orbit determination

Wide-field surveys take repeated images, identify moving point sources, link detections into tracks, fit an orbit, and recover the object over later nights and oppositions. A short arc can fit radically different distant orbits. Multi-opposition recovery is especially important for apparent extreme-`a` objects.

The MPC is the authoritative clearinghouse for designations and observations. JPL's Small-Body Database provides machine-readable orbit solutions, uncertainties, solution epochs, data arcs, observation counts, and optional covariance matrices; JPL explicitly ties covariance to its solution epoch ([SBDB API documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html)). Neither database is itself a characterized survey selection function.

### What each observational technique measures

| Technique | Direct information | Principal degeneracy or limitation |
|---|---|---|
| Astrometry over time | Orbit, covariance, dynamical class | Distant periods are centuries to millennia; short arcs can give misleading `a` and `e`. |
| Multiband photometry | Brightness, color, phase behavior | Size and albedo are degenerate; color is not a unique mineral identification. |
| Thermal infrared/mm | Thermal flux, radiometric diameter, albedo with optical data, thermal inertia constraints | Requires a thermal model; faint and distance-limited. |
| Reflectance spectroscopy | Surface molecular/chemical absorption bands | Samples the optically active surface, not the bulk interior; mixtures and irradiation complicate interpretation. |
| Rotational lightcurve | Rotation period, elongation, albedo variegation, contact-binary clues | Pole orientation and shape/albedo effects are degenerate. |
| Stellar occultation | Kilometer-scale chords, size/shape, rings, atmosphere limits | Rare, prediction-sensitive, and often chord-limited. |
| Resolved binary orbit | System mass | Individual masses require mass ratio; density also needs component volumes. |
| Spacecraft flyby | Geology, morphology, local composition, craters, atmosphere/plasma | Only Pluto/Charon and Arrokoth have close spacecraft reconnaissance. |

### Survey characterization is mandatory for population inference

All surveys select by sky position, time, limiting magnitude, apparent rate, crowding, weather, linking, follow-up, and tracking success. The OSSOS methodology states the consequence directly: intrinsic models cannot be compared to discoveries until the same survey biases have been imposed on the model. Forward simulation also uses non-detections and survey blind spots, which inverse corrections cannot recover reliably ([OSSOS Survey Simulator methodology](https://doi.org/10.3389/fspas.2018.00014)).

For reproducible science, keep three data layers separate:

1. **observed detections:** the catalog actually measured;
2. **characterized survey response:** pointings, efficiencies, rate/color/tracking rules, and non-detections;
3. **intrinsic population model:** a physical or phenomenological distribution that is forward-biased into predicted detections.

## 5. Physical nature of TNOs

### Sizes, albedos, masses, and densities

Optical absolute magnitude alone does not give diameter. A small bright/high-albedo body can resemble a larger dark body. Thermal measurements or occultations break much of this degeneracy. Mass normally comes from a satellite orbit; density then requires reliable sizes and shapes. Consequently, the best-characterized objects are not a random subset: large, bright, binary, occulting, or spacecraft-visited systems are overrepresented.

TNOs span from kilometer-scale bodies to Pluto/Eris-scale worlds. Their albedos range from very dark irradiated surfaces to bright volatile frost. Densities range from porous, ice-rich small bodies to rock-rich large dwarf planets, but sample selection and shape uncertainties matter.

### Surface composition and color

TNO colors range from nearly neutral to very red. Cold classical objects tend to occupy a distinctive red color sequence, while dynamically excited populations span multiple color/compositional classes. Color is useful population evidence, but cannot by itself identify a molecule.

JWST's DiSCo-TNO program detected H2O, CO2, 13CO2, CO, CH3OH, and complex or refractory material carrying C-H, C≡N, O-H, and N-H bonds. Independent clustering of the spectra yields three broad compositional groups ([JWST/DiSCo-TNOs portrait](https://doi.org/10.1038/s41550-024-02433-2)). This supports formation-zone and volatile-retention information surviving on present surfaces, while irradiation, impacts, seasonal transport, and differentiation can modify individual bodies.

A 2026 coordinated HST/JWST study extended optical-to-near-infrared colors down to bodies no larger than about 40 km. Thirteen JWST detections were recovered with HST; the small cold-classical sample remained on the narrow color sequence seen for larger cold classicals, whereas excited objects retained a broader color spread ([Morgan et al. 2026](https://doi.org/10.3847/1538-3881/ae9084)). This is recent evidence and should be tested with larger characterized samples.

### Binaries and contact binaries

Binaries are unusually important in the Kuiper belt because they preserve formation and dynamical-survival information and provide masses. Wide, near-equal binaries are common in the cold classical population; a representative study reports a cold-classical binary fraction around 30%, compared with roughly 10% among dynamically excited populations ([Fraser et al. 2017](https://doi.org/10.1038/s41550-017-0088)). Exact fractions depend on resolution and detection efficiency.

New Horizons revealed Arrokoth as a flattened, bilobate contact binary. Its morphology and composition favor a gentle, low-speed merger of two planetesimals rather than a destructive collision ([McKinnon et al. 2020](https://pubmed.ncbi.nlm.nih.gov/32054695/)). This is powerful evidence for gentle planetesimal formation in the cold classical environment, not proof that every TNO formed identically.

### Rings, atmospheres, and geology

- A 2017 multi-chord occultation measured Haumea's elongated figure and discovered a ring coplanar with its equator and the orbit of Hiʻiaka ([Ortiz et al. 2017](https://doi.org/10.1038/nature24051)).
- Quaoar has a dense ring well outside its classical Roche limit, showing that the simple “rings inside, moons outside” rule is incomplete ([Morgado et al. 2023](https://doi.org/10.1038/s41586-022-05629-6)); a published correction did not change that primary conclusion.
- New Horizons found Pluto to have diverse, geologically young terrain, mobile nitrogen-rich ices, a global haze, and an extended thin atmosphere. Small icy worlds can remain active over billions of years ([New Horizons initial Pluto results](https://pubmed.ncbi.nlm.nih.gov/26472913/)).
- Volatile atmospheres are easiest for the largest bodies; occultations place strong upper limits on atmospheres around many smaller TNOs.
- A 2026 *Nature Astronomy* paper reports a roughly 100–200 nanobar atmosphere around the approximately 500 km plutino (612533) 2002 XV93, which would be the first reported TNO atmosphere beyond Pluto. Its composition and origin remain unknown, so this new result still warrants independent confirmation ([Arimatsu et al. 2026](https://doi.org/10.1038/s41550-026-02846-1)).

## 6. Formation and evolution

### What is comparatively well supported

- The present belt is a depleted remnant of a much more massive early planetesimal disk.
- Neptune migrated outward and reorganized the trans-Neptunian region, capturing resonant objects, implanting dynamically hot populations, and feeding scattering reservoirs.
- The cold classical population formed largely in or near its current region and remained comparatively undisturbed.
- The modern population mixes implanted and primordial bodies.
- Collisions modified the small-size distribution, surfaces, spin states, and families, but did not erase every primordial signature.

### What remains model-dependent

The timing, smoothness, jumps, and duration of giant-planet migration are inferred by comparing simulations to the belt's inclination distribution, resonances, kernel near 44 AU, binary survival, and outer edge. One influential model family finds Neptune migration timescales of at least about 10 Myr are needed to reproduce the excited inclination distribution while keeping Neptune's eccentricity and inclination low enough to preserve the cold belt ([Nesvorný 2015](https://arxiv.org/abs/1504.06021)). This is a constraint within those simulations, not a unique reconstruction of Solar System history.

The origin of detached objects is less settled. Relevant mechanisms include:

- resonance capture plus Kozai-like perihelion lifting during migration;
- an early stellar cluster and close stellar passages;
- the Galactic tide for the most distant orbits;
- a temporary rogue or ejected planet;
- an additional surviving distant planet;
- combinations of these processes.

The sharp outer edge of the cold belt and the number/orbits of large-`a`, high-`q` bodies remain priority constraints precisely because observations there are sparse and strongly biased ([Gladman & Volk 2021](https://doi.org/10.1146/annurev-astro-120920-010005)).

## 7. Landmark objects and systems

| Object/system | Population or role | Why it matters |
|---|---|---|
| Pluto–Charon | 3:2 Neptune resonance; dwarf planet and binary system | First discovered TNO; active geology, volatile cycles, atmosphere, complex satellite system, spacecraft ground truth. |
| Eris–Dysnomia | Scattered/detached large dwarf planet system | Eris's discovery triggered the modern planet definition; satellite orbit constrains system mass. |
| Haumea system | Excited TNO/dwarf planet, collisional family | Rapid rotation, elongated shape, moons, water-ice-rich family, and ring constrain collisions and internal structure. |
| Makemake | Large classical-belt dwarf planet | Bright methane-bearing surface and a moon; important volatile-retention comparison. |
| Quaoar–Weywot | Large classical-belt system | Ring(s) outside the classical Roche limit challenge ring-accretion theory. |
| Gonggong–Xiangliu | Large eccentric resonant/excited system | Example of a large body's resonance dynamics and mass/density constraints from a moon. |
| Orcus–Vanth | 3:2 plutino binary | Comparative “anti-Pluto” resonant system; binary orbit provides mass. |
| Arrokoth | Cold classical contact binary | Most primitive body visited closely; records gentle planetesimal formation. |
| Triton (related, no longer a TNO) | Captured moon of Neptune | Its retrograde orbit and composition strongly suggest trans-Neptunian origin; Voyager 2 provides a rare close look at a captured member of the source population. |
| Sedna | Very high-perihelion detached object | Orbit is decoupled from present close Neptune scattering; key constraint on early environment and distant perturbers. |
| 2012 VP113 | High-`q` detached/sednoid-like body | Expanded the known high-perihelion population and motivated clustering debate. |
| Leleākūhonua (2015 TG387) | Very distant high-`q` object | Long-period orbit probes the distant detached/inner-Oort transition. |
| 2023 KQ14 (“Ammonite”) | High-perihelion detached object | Recent well-followed addition; its longitude is not aligned with the original small group, illustrating how new objects alter clustering narratives ([Chen et al. 2025](https://doi.org/10.1038/s41550-025-02595-7)). |
| 2017 OF201 | Extreme, large-`a` candidate | Recent orbit adds leverage on the distant population but should be handled with its actual solution covariance and survey selection ([discovery/orbit paper](https://arxiv.org/abs/2505.15806)). |

The class of a particular object can change with an improved orbit solution or a different declared classifier. The table is a scientific guide, not a replacement for a versioned dynamical classification.

## 8. Census and major surveys

Discovery counts change too rapidly for a static document to be authoritative. Unarchived spot queries made on 2026-10-08 returned about **7,300** JPL SBDB TNO-class objects (1,049 numbered) and **7,215** entries from the MPC List API. Treat these as approximate snapshots only: the query parameters, responses, and hashes were not retained with this dossier. The database-specific totals differ because orbit solutions, update times, and inclusion rules differ; they are not estimates of the intrinsic population. A scientific analysis must freeze the exact request, UTC retrieval time, response bytes, and hash from the [MPC List API](https://docs.minorplanetcenter.net/mpc-ops-docs/apis/list/) and [JPL SBDB query service](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html). An official Rubin update in April 2026 described roughly 5,000 objects before its large early submission and approximately 380 Rubin TNO candidates found in less than two months ([Rubin Observatory](https://rubinobservatory.org/news/11000-new-asteroids)). “Known” can mean candidate, designated object, numbered object, or securely classified orbit; those are not the same count.

### OSSOS and affiliated surveys

The Outer Solar System Origins Survey covered 155 square degrees to approximately `m_r = 24.1–25.2`, reported 838 well-tracked outer Solar System discoveries, and obtained 20–60 astrometric measurements over 2–5 oppositions. It included 313 resonant TNOs, 132 of them plutinos, and 436 nonresonant belt objects. Its survey simulator quantifies the selection function ([Bannister et al. 2018](https://doi.org/10.3847/1538-4365/aab77a)).

OSSOS's scientific importance is not just sample size; it minimizes ephemeris bias and records where, when, how deep, at what rates, and with what tracking efficiency it searched.

### Dark Energy Survey Year 6

DES searched roughly 5,000 square degrees of southern sky across six years and supplied software for simulating observational biases. The earlier manuscript reported **817 total objects**: 815 nominal TNOs, one Centaur, and one Oort-cloud comet, with 461 first reported there. The canonical final journal/CDS release contains **814 total objects**: 812 nominal TNOs, one Centaur, and one Oort-cloud comet, with 458 first reported. The declared sample is complete to roughly `r = 23.8` under its stated conditions ([Bernardinelli et al. 2022](https://doi.org/10.3847/1538-4365/ac3914)). This release discrepancy is exactly why scientific pipelines must bind a particular file and hash.

DES's 16 extreme TNOs were consistent with azimuthal isotropy in its selection-aware test. The catalog also provides barycentric Cartesian states and full six-dimensional state covariances useful for clone-based dynamical classification.

### Vera C. Rubin Observatory / LSST

Rubin is changing the scale of TNO science now. A high-fidelity LSST simulation predicts a final catalog of approximately 37,000 TNOs and expects about 70% of more distant small-body populations to be discovered in the first two survey years ([Kurlander et al. 2025](https://doi.org/10.3847/1538-3881/add685)). That is a prediction, not a final observed count.

Early Rubin optimization data already yielded roughly 380 TNO candidates. Short initial arcs still require recovery: a 2026 follow-up recovered two initially very large-`a` Rubin candidates and reduced their fitted semimajor axes by factors of roughly 3–10, making them ordinary 3:2 plutinos ([Hemmann et al. 2026](https://arxiv.org/abs/2608.04287)). Therefore early Rubin discoveries should be stored with solution date, arc, covariance, and a provisional-status flag rather than immediately treated as stable extreme-TNO evidence.

### Other essential facilities

- **Subaru/Hyper Suprime-Cam** and other deep wide-field programs probe faint and distant populations.
- **DEEP** uses digital tracking to reach roughly magnitude 26–27 and publishes an injection-calibrated selection function ([DEEP B1 characterization](https://arxiv.org/abs/2310.03671)).
- **HST and JWST** provide deep discovery/recovery, colors, lightcurves, and spectroscopy beyond ground-based atmospheric limits.
- **ALMA and thermal facilities** constrain size, albedo, emissivity, and thermal behavior for selected targets.
- **Gaia-derived star catalogs** make precise occultation predictions possible.
- **New Horizons** remains the only spacecraft to have explored TNOs close-up.

## 9. The Planet Nine question in proper context

The Planet Nine hypothesis proposes that an unseen several-Earth-mass planet organizes some distant TNO orbits. A 2021 fit reported a representative favored region around 6 Earth masses, semimajor axis about 380 AU, moderate eccentricity, and inclination around 16 degrees, but these values are posterior/model dependent and do not constitute a detection ([Brown & Batygin 2021](https://arxiv.org/abs/2108.09868)).

The central difficulty is that extreme TNO discoveries are highly nonuniform in sky position, depth, season, orbital phase, and follow-up. A selection-aware analysis of 14 ETNOs from DES, OSSOS, and the Sheppard–Trujillo survey found their perihelion longitudes and orbital poles compatible with a uniform parent population and reported no clustering evidence in that sample ([Napier et al. 2021](https://arxiv.org/abs/2102.05601)). DES Y6 likewise found its 16 extreme objects consistent with azimuthal isotropy. A 2026 conditional-likelihood update using 25 stable or metastable distant TNOs reported only 1.9-sigma three-dimensional apsidal clustering and a poorly constrained preferred direction, reduced from the same group's earlier approximately 3-sigma result after new objects entered the sample ([Siraj, Chyba & Tremaine 2026](https://doi.org/10.3847/1538-4357/ae6586)). Other papers report dynamical features that are difficult for particular no-Planet-Nine models. These statements are not logically inconsistent: they test different samples, features, model families, and selection functions.

Direct searches are also constraining rather than confirming the hypothesis: a combined ZTF, DES, and Pan-STARRS1 analysis excluded about 78% of one Brown–Batygin reference population, while leaving fainter solutions and difficult sky near the Galactic plane ([Brown, Holman & Batygin 2024](https://arxiv.org/abs/2401.17977)). That is not exclusion of every distant-planet orbit.

The scientifically correct status is:

- no Planet Nine object has been directly detected;
- the clustering evidence is disputed and selection-sensitive;
- no single known mechanism yet explains every distant-orbit feature uniquely;
- new TNOs can strengthen, weaken, or redirect the hypothesis;
- Rubin will provide a much larger, more homogeneous test sample;
- a null result for one finite Planet Nine grid cannot exclude all distant planets.

For JX, Planet Nine should be an **optional matched hypothesis arm** after a validated TNO baseline, not the definition of the project.

## 10. Major open questions

1. What produced the cold classical belt's sharp outer edge and 44 AU kernel?
2. How smooth, slow, or “jumping” was Neptune's migration, and when did giant-planet instability occur?
3. How much of each population formed in place versus being implanted?
4. What is the intrinsic size distribution below current direct-detection limits?
5. How did wide binaries and contact binaries form and survive?
6. Which surface compositions trace formation location, and which reflect later irradiation or volatile loss?
7. How common are rings and satellites, and how do rings persist outside classical Roche limits?
8. What creates high-perihelion detached and sednoid orbits: migration, stellar encounters, cluster tides, rogue planets, a surviving distant planet, or a mixture?
9. Is any angular clustering present after survey selection and upstream object-selection choices are fully modeled?
10. How does the scattering disk feed Centaurs and Jupiter-family comets?
11. What fraction of large TNOs are differentiated or maintain subsurface oceans?
12. Which apparent “extreme” objects remain extreme after multi-opposition recovery?

## 11. Reproducible TNO research standard

A claim-grade TNO analysis should satisfy all of the following.

### Catalog custody

- Freeze the exact release, URL/DOI, retrieval time, file bytes, and SHA-256.
- Preserve MPC/JPL designation crosswalks and a documented deduplication rule.
- Record orbit solution ID/date, observation arc, observation count, epoch, time scale, frame, origin, and GM convention.
- Retain a full covariance where available and verify its element order, units, symmetry, positive semidefiniteness, and epoch.
- Never silently propagate old osculating elements to a new epoch by changing the label.

### Dynamics

- Convert all inputs to one explicit barycentric Cartesian frame and TDB epoch.
- Sample the joint covariance reproducibly; never draw six independent one-sigma element errors unless that is truly the supplied covariance model.
- Use the same known-planet states and force model for every clone and hypothesis arm.
- Define resonance, scattering, detachment, ejection, collision, and close-encounter events before inspecting outcomes.
- Publish nominal and clone classifications, not just one label.
- Demonstrate time-step/integrator convergence on population statistics and event classifications.

### Population inference

- Keep each survey as its own selection-function stratum.
- Forward-bias model populations through actual pointings, depths, rate cuts, colors, tracking, and linking rules.
- Do not use a heterogeneous MPC compilation for confirmatory angular statistics.
- Include every model search, cut, nuisance fit, and candidate-selection step in the null calibration.
- Distinguish descriptive detected-sample plots from debiased intrinsic-population claims.

### Physical properties

- Trace each diameter to thermal or occultation assumptions and each mass to a binary/orbit solution.
- Propagate albedo, shape, pole, phase, and thermal-model uncertainty.
- Label colors as colors; do not turn broadband color into a unique composition.
- Keep system properties separate from individual component properties.

## 12. What the JX workspace already has

The project has unusually strong engineering and provenance foundations, but they must be reframed around TNO science.

### Real observational assets already present

- A byte-verified **OSSOS complete-release inventory and digest record**: 840 characterized catalog rows corresponding to 838 securely tracked discoveries, plus 1,148 ensemble detection rows representing 1,142 distinct objects. The downloaded catalog payloads are not committed here. See [`ossos_inventory_v1.json`](../audits/jx_o2_g0_input_audit_v1/ossos_inventory_v1.json).
- A byte-verified **DES Y6 inventory and digest record**: the final 814-row journal/CDS release, the enhanced author catalog, state-covariance fields, exposure positions, CCD corners, and exact discrepancies. The downloaded catalog and large selection payloads are not committed here. See [`des_y6_inventory_v1.json`](../audits/jx_o2_g0_input_audit_v1/des_y6_inventory_v1.json).
- A current, frozen **JPL detached-object query** containing 26 high-perihelion/distant objects at the selected query epoch. It is useful as an inventory, but its listed one-dimensional uncertainties are not a substitute for full covariance clones. See [`jpl_sbdb_detached_query_20261005.json`](../runs/long_horizon_science_inputs_v1/raw/jpl_sbdb_detached_query_20261005.json).
- A qualified historical **four-block OSSOS-B survey-adapter** calibration. It proves adapter behavior for that limited characterization, not the complete eight-block OSSOS survey or a real intrinsic TNO model. See [`SURVEY_SELECTION_VALIDATION.md`](SURVEY_SELECTION_VALIDATION.md).

### Numerical assets already present

- high-throughput massless-tracer and restricted-Wisdom–Holman infrastructure;
- independent adaptive-integrator comparisons;
- exact state, checkpoint, replay, runtime-identity, and artifact hashing machinery;
- matched control/intervention construction and strong claim firewalls;
- preparation work for Galactic tide, stellar encounters, solar evolution, and direct encounters.

These are valuable tools. Existing long integrations primarily use synthetic tracer proposals or narrow Planet Nine fixtures, so they are engineering evidence rather than a measured model of the real TNO population.

### Current blockers for general TNO science

- The public OSSOS simulator bundle in the workspace covers only the earlier E/O/L/H portion; later selection files found in history were removed as bad. The full eight-block selection function is not safely activatable.
- DES has unresolved release, eligibility, exposure-count, bandpass, linking, efficiency, dependency, and raw-selection-lineage differences.
- The DES FITS header does not encode the Cartesian state epoch's time scale; the local audit records the TDB interpretation as `NOT_YET_LOCKED`. That must be resolved before covariance sampling or integration.
- There is no unified, versioned TNO master catalog with cross-survey deduplication and common state/covariance semantics.
- There is no production clone-based dynamical classifier covering resonance, scattering, and detachment.
- There is no joined physical-properties table with measurement provenance.
- No current result supports detecting, excluding, or ranking Planet Nine.

The existing G0 audit correctly says `BLOCKED` and permits only continued acquisition, maintainer contact, reconstruction planning, license clarification, and design work. It does **not** authorize catalog construction, descriptive science, classifier execution, observational scoring, a large GPU job, or unblinding. Those activities require a separate TNO-specific contract and claim boundary. See [`JX-O2 G0 audit`](../audits/jx_o2_g0_input_audit_v1/README.md).

## 13. Recommended TNO-first JX program

This sequence follows Occam's razor: obtain the broadest defensible astronomy from the simplest sufficient model before adding a hypothetical planet or billion-year environment.

### Stage A — TNO master catalog v1

First establish a new TNO-specific authorization/claim contract, then reacquire the final DES Y6 and OSSOS characterized payloads from their canonical sources and verify their byte counts and hashes against the retained inventories. Only then build a content-addressed catalog from those releases and a dated, archived MPC/JPL crossmatch. Preserve each native record and add, without overwriting it:

- canonical designation and aliases;
- survey and characterization status;
- state/element epoch, time scale, frame, origin, and solution ID;
- full covariance when supplied;
- observation arc and quality fields;
- provisional/multi-opposition status;
- published dynamical class and its source;
- physical-property references, not guessed values.

This is useful even before any dynamics and does not require a GPU, but it is not authorized by the older blocked G0 contract on its own.

### Stage B — known-planets dynamical atlas

After locking the DES state epoch's time scale, start with DES because it supplies homogeneous barycentric states and full 6D covariance. Reproduce its declared clone approach as a baseline, then independently classify:

- nominal orbit plus a fixed, preregistered covariance-clone ensemble;
- 10 Myr under Sun + giant planets, with terrestrial masses treated explicitly and consistently;
- resonance search, scattering test, close-encounter history, and clone agreement;
- no extra planet, no Galactic environment, and no population inference in the first pass.

This creates a real-object TNO product while keeping the force model simple and scientifically conventional. Ambiguous or rare detached cases can later receive longer or more complete integrations.

### Stage C — physical-property atlas

Join published, provenance-tagged colors, absolute magnitudes, thermal/occultation diameters, albedos, rotations, binaries, masses, densities, spectra, rings, and atmosphere limits. Analyze relationships by **dynamical class with measurement selection disclosed**, rather than mixing every TNO.

### Stage D — characterized-survey population tests

Resolve the OSSOS and DES selection blockers or limit the claim to the exact characterized subset that is genuinely available. Forward-bias intrinsic models separately through each survey. First questions should be Planet-Nine-independent:

- classical-belt inclination/eccentricity structure;
- 44 AU kernel and outer edge;
- resonance occupancy;
- detached/scattering ratio;
- color and binary fractions by securely classified population;
- size-distribution alternatives.

### Stage E — distant-perturber hypotheses

Only after Stages A–D should JX compare a known-planets baseline with Planet Nine, rogue-planet, stellar-cluster, or migration alternatives. Use identical TNO clones and survey forward models. Report a finite model-family comparison, never “Planet Nine detected/excluded.”

### Stage F — Rubin-era updates

Ingest Rubin releases chronologically, preserve provisional orbit history, and reserve genuinely future data for prospective tests. Do not tune cuts on early Rubin objects and then call later analysis blind unless the temporal holdout and analysis contract were frozen first.

## 14. The most efficient next deliverable

The best immediate deliverable is **not** another billion-year synthetic run. It is a small TNO-specific authorization and claim contract plus a create-only reacquisition manifest for the OSSOS and DES payloads. That first step is design/provenance work allowed by the current G0 boundary. Once independently approved and the DES time scale is locked, the next implementation should be a read-only, content-addressed `TNO_MASTER_CATALOG_V1` plus a CPU pilot that classifies a small, preregistered set of high-quality DES objects and covariance clones under the known planets for 10 Myr.

Success criteria for the pilot:

1. exact recovery of the source catalog rows and covariance ordering;
2. deterministic clone bytes from a frozen sampler and seed derivation;
3. agreement with published DES classes for a representative classical, resonant, scattering, and detached subset;
4. secure/insecure classification based on clone agreement;
5. integrator and step-size crosscheck on the same clones;
6. no survey-population, Planet Nine, or formation claim.

If that passes, scale to the full DES catalog and then crossmatch OSSOS. This produces a general astronomy asset, exercises the best existing JX numerical work, and avoids spending compute on hypotheses before the real-object baseline exists.

## Primary source map

- Gladman & Volk, [*Transneptunian Space*](https://doi.org/10.1146/annurev-astro-120920-010005), Annual Review of Astronomy and Astrophysics 59 (2021).
- Gladman, Marsden & VanLaerhoven, [*Nomenclature in the Outer Solar System*](https://web.gps.caltech.edu/~mbrown/out/kbbook/Chapters/Gladman_Nomenclature.pdf) (dynamical classification framework).
- Bannister et al., [OSSOS complete data release](https://doi.org/10.3847/1538-4365/aab77a) (2018).
- Lawler et al., [OSSOS Survey Simulator methodology](https://doi.org/10.3389/fspas.2018.00014) (2018).
- Bernardinelli et al., [DES Year 6 outer Solar System search](https://doi.org/10.3847/1538-4365/ac3914) (2022).
- JPL, [Small-Body Database API documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html).
- IAU, [2006 planet and dwarf-planet resolutions](https://www.iau.org/IAU/Iau/News/PR2006/iau-2006-general-assembly-resolution-votes.aspx).
- Nesvorný, [slow Neptune migration constraint](https://arxiv.org/abs/1504.06021) (2015).
- Pinilla-Alonso et al., [JWST/DiSCo-TNO compositional portrait](https://doi.org/10.1038/s41550-024-02433-2) (2025).
- McKinnon et al., [Arrokoth's solar-nebula origin](https://pubmed.ncbi.nlm.nih.gov/32054695/) (2020).
- Stern et al., [New Horizons initial Pluto-system results](https://pubmed.ncbi.nlm.nih.gov/26472913/) (2015).
- Ortiz et al., [Haumea occultation and ring](https://doi.org/10.1038/nature24051) (2017).
- Morgado et al., [Quaoar ring outside the Roche limit](https://doi.org/10.1038/s41586-022-05629-6) (2023).
- Napier et al., [selection-aware ETNO clustering test](https://arxiv.org/abs/2102.05601) (2021).
- Brown & Batygin, [Planet Nine orbit/population inference](https://arxiv.org/abs/2108.09868) (2021).
- Kurlander et al., [LSST Solar System yield prediction](https://doi.org/10.3847/1538-3881/add685) (2025).
- Rubin Observatory, [early 2026 asteroid/TNO submission](https://rubinobservatory.org/news/11000-new-asteroids).
- Morgan et al., [small TNO optical–NIR colors with HST/JWST](https://doi.org/10.3847/1538-3881/ae9084) (2026).
