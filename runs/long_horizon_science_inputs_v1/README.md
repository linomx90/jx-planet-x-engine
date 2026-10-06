# Long-horizon observed-object science inputs v1

This package replaces the earlier arbitrary orbital sentinels with two
content-addressed NASA/JPL SBDB benchmark rosters at the exact frozen DE441
source epoch, JD TDB `2461200.5`.

- The detached query contains all 26 SBDB rows satisfying the locked query.
- The control query contains all 329 rows satisfying its locked query.
- Six nominal objects per arm are selected only by source-data properties,
  before any JX dynamical outcome: complete element uncertainties, `ORB`
  provenance, longest data arc, observation count, then SPK-ID.
- The control additionally requires the JPL `name` field to be null.
- Eight deterministic perturbations per nominal object use independent offsets
  within three source-reported standard errors for each element.

The nominal rosters are observed-object benchmarks, but they are not complete,
debiased, likelihood-weighted, or suitable for population inference. The
independent perturbations do not reproduce the missing covariance matrices and
must not be called posterior samples.

The raw API responses are retained byte-for-byte under `raw/`; `MANIFEST.json`
binds their identities, selection rules, epoch, frame, and claim ceiling.
