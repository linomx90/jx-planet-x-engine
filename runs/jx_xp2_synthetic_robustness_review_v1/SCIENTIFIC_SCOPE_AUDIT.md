# XP2 scientific-scope assessment

**Status:** internal, non-peer-reviewed assessment  
**Assessment date:** 2026-08-27  
**Method:** static review of the preserved contract, input manifests, source,
diagnostic lineage, and the primary-literature map in `REFERENCES.md`. No
dynamics, compilation, registration, or outcome analysis was performed for
this assessment.

## Verdict

XP2 is defensible as an exploratory synthetic numerical-response benchmark
without an authorized scientifically admissible result. It is not a
confirmatory Planet Nine test and cannot support an observational or
population-level claim.

## Exact question the protocol could answer

If a future authorized execution passed all encoded gates, XP2 could report how
the exact deterministic tracer design responds over 1 Myr to the exact 24-cell
added-body grid relative to the shared no-added-body control, and whether its
qualitative classification is stable across replay, a halved MERCURIUS
timestep, the 0.5 versus 1 Myr landmark, and a limited SciPy DOP853 sentinel.

That conclusion would apply only to the encoded synthetic design and model.

## Why it is exploratory

The preserved contract's `xp1_historical_binding.endpoint_change_disclosure`
records that XP2 uses `q < 35 au` at 1 Myr after a reported floor-limited XP1
`q < 30 au` endpoint. The underlying XP1 artifacts and results are not included
or independently reassessed here.

Consequently, the current endpoint and expanded design are outcome-responsive.
Freezing XP2 before a scientifically admissible V4 result prevents later
within-XP2 adaptation, but it does not turn the design into an outcome-blind
confirmation.

## Useful numerical controls already specified

- A matched no-added-body control using identical tracer initial states.
- Two MERCURIUS timestep resolutions.
- Exact replay requirements.
- A 0.5 Myr versus 1 Myr landmark-stability check.
- A separate SciPy DOP853 path with a custom Newtonian right-hand side over a
  preselected subset.
- Conservation, completion, restart, and cross-method numerical gates.

These are numerical controls. They are not observational replication or
independent physical validation.

## Scientific limitations

1. **No observations or survey model.** The tracer design is synthetic and is
   not an observed or inferential trans-Neptunian population.
2. **Short horizon.** One million years cannot establish Gyr-scale confinement,
   formation, or survival.
3. **No uncertainty ensemble.** The design uses one deterministic tracer set,
   one giant-planet phase configuration, and eight fixed orientations.
4. **Correlated physical grid.** Mass, semimajor axis, and eccentricity change
   together in three tuples, so their effects cannot be separated.
5. **No inferential calibration.** Effect-size cutoffs and event-support floors
   are deterministic classification conventions, not thresholds derived from a
   power or error-rate model.
6. **Sampled crossings.** A 50 yr cadence can miss excursions between samples.
7. **Idealized physics.** Several known Solar-System and environmental effects
   are intentionally absent.
8. **Limited solver sentinel.** DOP853 covers 32 tracers and seven arms and is
   not independent astrophysical evidence.
9. **Incomplete public reproducibility closure.** Historical registrations,
   runtime locks, receipts, and sibling trees referenced by the contract are
   intentionally omitted from this snapshot.

## Minimum path to a confirmatory successor

A later confirmatory synthetic experiment should be frozen before its outputs
are seen and use fresh independently generated realizations. It should declare
one scientifically justified estimand, use a factorized or space-filling
perturber design, pair each realization with a control, justify sample size and
thresholds, quantify uncertainty across independent realizations, test cadence
and horizon convergence, retain reanalysis-grade events or trajectories, and
preselect broader cross-code validation.

Any real Planet Nine inference is a separate observational project requiring
audited ephemerides and uncertainties, characterized survey catalogs and
selection functions, matched physical models, forward survey simulation,
absolute-adequacy checks, multiplicity control, and held-out or independent
validation.

## Current evidence state

- Authorized V4 XP2 scientific execution: not completed.
- Scientifically admissible XP2 result: none.
- Planet Nine evidence from XP2: none.
- Public value of this release: protocol transparency, source review, and
  documentation of an exploratory numerical benchmark.
