# Canonical observed-TNO numerical cross-check v1

This is the single compact TNO engineering experiment selected for the current
JX consolidation pass. It does not run new dynamics. The verifier reopens and
hashes the retained M3 ten-year experiment, validates every retained array,
and independently recomputes its direct Cartesian comparisons.

The source experiment contains:

- 12 nominal NASA/JPL SBDB TNO states;
- 96 deterministic bounded sensitivity states (not covariance or posterior
  samples);
- the Sun, eight planetary-system barycenters, and Pluto as ten active massive
  sources;
- a JX candidate lane with a 1/512-year tracer step;
- a JX refinement lane with a 1/1024-year tracer step; and
- a REBOUND 5.1.1 IAS15 comparator sampled on the refinement grid.

The result is numerical engineering evidence only. Ten years cannot establish
a TNO dynamical class, long-term stability, resonance, population structure,
survey bias, or evidence for or against Planet Nine. The original acceptance
thresholds were calibrated from a retained calibration run, and the
calibration and official state archives are byte-identical. The compact result
therefore describes deterministic agreement and replay, not a blinded
independent accuracy qualification.

Generate or verify the create-only result from the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python tools/jx_tno_observed_crosscheck_v1.py --write
PYTHONPATH=src:. .venv/bin/python tools/jx_tno_observed_crosscheck_v1.py --verify
```

`RESULT.json` is the canonical concise entry point. The much larger retained
source experiment remains immutable under
`runs/long_horizon_m3_tracer_preflight_v1/attempt-20261006d/`.
