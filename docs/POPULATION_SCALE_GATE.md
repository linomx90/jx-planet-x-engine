# Population scale gate

`run-population-scale-gate` measures whether the current machine and pinned
REBOUND build can propagate a locked, paired source/control population of
massless tracers. It is an execution and integrity gate, not a Planet X
detection test and not an observational population inference.

Install the optional backend:

```bash
python -m pip install -e '.[rebound]'
```

Run an immutable contract:

```bash
jxplanetx run-population-scale-gate \
  --contract runs/population_100k/scale_gate_contract.json \
  --output runs/population_100k/scale_gate_result.json
```

The runner requires source and control CSV states whose common bodies have
exactly equal heliocentric Cartesian states. It verifies the prelocked state,
JX source-tree, REBOUND version/build/binary/wheel, deterministic phase
generator, and population counts before integrating.

Tracers are massless (`N_active` contains only the massive bodies and
`testparticle_type=0`). They are addressed by stable particle index, not a
REBOUND string hash; 100,000 ordinary `tNNNNN` labels are not collision-free
in REBOUND's 32-bit hash space. Collision handling and particle removal remain
disabled so indices cannot change.

Perihelion classification uses the contract's locked hysteresis band. Values
inside the band are reported as boundary values, never silently assigned to a
side. Injection fractions normalized by eligible initial tracers are `null`
when no template begins above the upper band.

Massive-body energy drift is sampled with an active-only twin at the locked
cadence. The runner requires its final active state to be bit-identical to the
full test-particle run. This verifies that test particles did not back-react
while avoiding an all-particle energy pass at every nominal step.

A short-horizon linear runtime projection is only an estimate. It cannot close
the projection-evidence gate until repeated timings and a locked long-horizon
encounter-tail pilot have been completed. A scientific ensemble additionally
requires a physical population prior, timestep convergence, close-encounter
audits, and an independent integration algorithm.
