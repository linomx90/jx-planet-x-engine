# JX N-Body Engine 0.3.0

Release date: 22 August 2026  
Claim state: `SCREENING_ONLY`

## Independent population replication

- Added an independent Newtonian force and SciPy DOP853 population runner that
  does not import or call REBOUND.
- Added independent Kepler element conversion and recovery, annual
  classification, segmented exact-binary64 checkpoint/restart, source
  stability audits, population comparisons, deterministic paired bootstrap,
  and fail-closed `PASSED`, `CONFLICT`, or `INVALID` verdicts.
- Locked Python, NumPy, SciPy, solver-source, coefficient-table, binary,
  initial-state, population, selection, reference-result, and runner hashes.
- Added an `independent` installation extra pinned to NumPy 2.3.5 and SciPy
  1.17.0.
- Expanded the test suite from 70 to 76 tests.

## Independent scientific record

The release includes a compact record of an independent replication of ten
outcome-blind hash-selected 1,000-tracer blocks from the DE441-backed
100,000-tracer experiment.

The original independent attempt is preserved as `INVALID`. It exceeded only
the active endpoint-position consistency gate: 1.27532×10⁻⁶ AU observed
against 1×10⁻⁶ AU locked. All population comparisons passed. A separate locked
diagnostic confirmed adaptive-resolution dependence; no v1 gate was relaxed or
retroactively changed.

A corrective v2 was registered with the same population, physical model,
statistics, and acceptance thresholds. Only DOP853 resolution changed:

- relative tolerance: 1×10⁻¹³;
- absolute tolerance: 1×10⁻¹⁵;
- maximum step: 0.125 year.

V2 returned `PASSED`:

- 10,000 tracers per arm over 10,000 years;
- 433/10,000 sampled injections in both independent arms;
- the same 433 identities in each corresponding REBOUND arm;
- zero injection-identity disagreement;
- 100% final survival in both arms;
- source-minus-control injection fraction 0.0;
- paired-block 95% bootstrap interval `[0.0, 0.0]`;
- maximum active energy drift 8.35832×10⁻¹³;
- maximum active angular-momentum-vector drift 3.02089×10⁻¹³; and
- maximum active endpoint-position disagreement 2.86934×10⁻⁹ AU.

An independent stored-artifact audit returned `AUDIT_PASSED`. It verified 19
locked files, 20 summaries, 20 independent tracer tables, 20 REBOUND reference
tables, and 100 checkpoint state pairs; reconstructed final orbital elements;
and recomputed every statistic, gate, and verdict.

## Scientific boundary

This result strengthens the numerical robustness of one candidate-9118 screen,
but remains `SCREENING_ONLY`. It does not detect or exclude Planet X, validate
candidate 9118, cover the wider candidate space, or substitute for an observed
TNO population and survey selection function. The v2 solver was selected after
the v1 numerical failure and is transparently labeled as a corrective run, not
the original preregistration.

The next scientific gate is an observed-population model plus an explicit
survey-selection likelihood. A longer-horizon hierarchical experiment should
follow only after that gate is defined.

## 0.2.0 foundation

Version 0.3.0 retains the 0.2.0 deterministic ensemble-validation framework,
strict trajectory registration, distribution metrics, claim-control state
machine, ten-year Horizons/DE441 compatibility record, and locked
100,000-tracer-per-arm REBOUND result. It also retains the earlier provenance
corrections that reject non-standard `NaN` and `Infinity` values.

This remains an engine-focused release. Bulk trajectory/checkpoint archives,
observational data, and candidate-search catalogs are excluded; compact
contracts, manifests, source, audits, and scientific reports are included.
