# JX long-horizon source model v1

This package directory retains the exact six-row control source state used by
the experimental restricted-Wisdom--Holman long-horizon path. `MANIFEST.json`
defines its epoch, frame, origin, units, ordered source roster, collapsed
inner-system primary, upstream digests, and nonclaims.

The CSV is a reduced, reconstructed DE441/Horizons-derived model. It is not a
retained DE441 SPK kernel, a direct ephemeris evaluator, or evidence that a
long-duration integration is accurate. Row 0 is the effective inner-system
monopole used by the JX force model; it must not be silently reinterpreted as
the standalone physical Sun.
