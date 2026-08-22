# Encounter-tail pilot

The encounter-tail runner measures a controlled numerical question: can JX
propagate a physically injection-eligible, chaotic test-particle population
with stable population conclusions at two timesteps?

It uses 72 synthetic orbit cells from the Cartesian product
`a=[100,150,225,325,450,600,800,1000] AU`, `q0=[35,45,60] AU`, and
`i=[5,20,35] deg`. Ten independently phased blocks contain 1,000 massless
tracers per arm. Candidate 9118 is present only in the source arm; the common
Sun and giant-planet states are required to be bit-identical after canonical
heliocentric conversion.

MERCURIUS runs for 10,000 years at `dt=0.0625 yr`. The observer samples
osculating perihelion and distances every 0.25 years, records aggregate
population history every 10 years, and creates a verified restart checkpoint
every 250 years. A prelocked 1,000-tracer subset repeats the full interval at
`dt=0.03125 yr`.

An injection means an initially eligible tracer (`q0>30.000001 AU`) has at
least one sampled bound osculating perihelion below `29.999999 AU`. Hill-sphere
outputs are sampled outside-to-inside entries relative to fixed initial Hill
radii. They are not complete continuous encounter detections.

The inference unit is the independent phase block, never an individual
tracer. Zero or null source effects may pass the numerical gate.

## Required limitations

The archived planetary control is a synthetic J2000-ecliptic-like benchmark
generated from approximate elements. It is not DE441, a real common epoch, or
ephemeris validation. The 72-cell population is a nearly equal synthetic grid,
not an observational likelihood. A 10-kyr run does not test 100-Myr steady
state or 4-Gyr survival. Candidate 9118 does not represent the full posterior.
Massless tracers omit collective gravity. No result is a Planet X detection,
exclusion, sky location, composition, or survey-completeness claim.

After this pilot, a tail-enriched IAS15 epsilon-pair audit is required. Because
MERCURIUS and IAS15 are both inside REBOUND, that remains a cross-integrator
audit rather than independent-software validation.
