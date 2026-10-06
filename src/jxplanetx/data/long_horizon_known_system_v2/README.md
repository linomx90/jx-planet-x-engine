# JX full known-system source model v2

This directory retains the ten source rows needed for the full known Solar
System foundation at TDB JD 2461200.5. The source roster is the physical Sun,
then the Mercury, Venus, Earth-Moon, Mars, Jupiter, Saturn, Uranus, Neptune,
and Pluto system barycenters. It does not use the earlier collapsed
inner-system monopole.

`horizons_de441_barycentric_icrf_jd2461200p5.csv` is an exact ordered
extraction from the retained 110-row NASA/JPL Horizons DE441 reference.
`gm_de440_major_barycenters.csv` is the retained NAIF DE440 GM extraction.
`known_system_primary_relative_ecliptic_float_hex.csv` preserves the exact
binary64 result of rotating each SSB/ICRF barycentric state by the frozen IAU
1976 J2000 mean-obliquity sine/cosine, then subtracting the rotated Sun state
to form primary-relative coordinates; velocities are subsequently multiplied
by exactly 365.25. This rotation-before-subtraction order preserves the
historical retained DE441 conversion exactly; the source provider separately
discloses its bounded, idempotent barycentric coordinate canonicalization.

The Python loader verifies every packaged byte digest, independently repeats
all conversions, and requires bitwise equality with the derived float-hex
artifact. This is an initial-state physical model only. It is not a retained
SPK evaluator, a complete DE441 force model, a long-duration qualification,
an observational fit, or evidence for detection or exclusion of any body.
