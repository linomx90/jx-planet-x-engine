# JX-E1 local paired-dynamics engineering pilot

This directory is deliberately outside the JX-O2 registered experiment. It is
a local, synthetic engineering exercise and does not alter JX-O2's blocked
state.

The locked smoke stage instantiates all nine published 2026 `(mass, a, e, i)`
rows under the same four arbitrary, predeclared angular completions. Every M1
run uses the same synthetic particles and analytic known-body state as its M0
counterpart; the only inserted object is the engineering-surrogate compact
body. The particles are a two-block Latin hypercube and are not an observed TNO
population or survey sample.

The 100-year horizon is intentionally too short for a scientific Planet Nine
test. It checks only:

- exact paired initial-state construction;
- finite integration of the complete 9 x 4 matrix;
- exact REBOUND checkpoint serialization and resumed continuation;
- active-body energy and angular-momentum drift;
- a preselected `dt/2` audit; and
- compact descriptive sensitivity diagnostics that cannot rank models.

The original checkpoint, missing angles, original seeds, modified author code,
and raw trajectories were not recovered. No observed catalog, survey adapter,
or published outcome column is read. A valid result therefore means only that
this synthetic machinery operated within its locked numerical gates.

The contract forbids automatic scaling. A 50,000-year engineering stage would
need a new contract locked after this smoke result is assessed using only
integrity, numerical-stability, and resource-budget criteria—not a favorable
effect direction or model row.

Historical invocation (shown for provenance only):

```text
python3 run_pilot_v3.py \
  --contract contract_v3.json --output-dir output_a
```

The exact historical runtime is not bundled in this checkpoint, and this
command is not a claim of portable reproducibility or current project
execution authority.

Mandatory boundary: this is not a reproduction, observation, detection,
exclusion, mass/orbit constraint, realistic cluster population, or 4-Gyr
survival result. It does not authorize JX-O2 execution.
