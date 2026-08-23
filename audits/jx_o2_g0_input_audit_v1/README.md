# JX-O2 G0 input audit v1

This directory records the first evidence audit required by the immutable JX-O2 design. It inventories public survey catalogs, survey-selection artifacts, physical-model leads, licenses, data exposure, and the local candidate-9118 lineage.

## Verdict

`BLOCKED` / no-go.

Useful public bytes were acquired and independently hashed, including the complete OSSOS catalog, the DES Y6 journal table, the two large DES selection payloads, and several exact source snapshots. They do not form an execution-ready input bundle:

- the official public OSSOS simulator release contains only the E/O/L/H half of OSSOS, while the later selection directory was explicitly removed from Git history as bad and disagrees with the published catalog;
- DES catalog, eligibility, efficiency, exposure-count, backend, dependency, and raw-selection-lineage discrepancies remain unresolved;
- no public, physically matched baseline/compact-body TNO source-population pair was found;
- candidate 9118 has no reproducible JX selection record and its published 2018 elements were reassigned to the JX 2026 epoch without propagation;
- the named observed catalogs and model-fitting samples were previously inspected, and no untouched confirmatory holdout was identified; and
- several data or software licenses are absent or ambiguous.

## What this audit permits

Only continued acquisition, maintainer contact, reconstruction planning, license clarification, and design work. It does not authorize synthetic calibration, an observed-data score, an observational model comparison, a large GPU job, or unblinding.

The downloaded catalogs and large selection files are not committed here. This compact package records canonical URLs, commits, byte counts, SHA-256 digests, scope, and discrepancies. A later activation-ready bundle would also require a content-addressed public archive and independently validated semantic manifest.

## Files

- `g0_audit_v1.json`: fail-closed verdict, gate assessment, and blocker register.
- `ossos_inventory_v1.json`: OSSOS, CFEPS, HiLat, and Alexandersen survey artifacts.
- `des_y6_inventory_v1.json`: DES Y6 catalog, simulator, selection payloads, and discrepancies.
- `source_model_inventory_v1.json`: physical-model leads and candidate-9118 lineage.
- `registration_g0_v1.json`: hash binding for this compact audit package.

This is a provenance audit, not a Planet X result. It contains no JX-O2 observational statistic or model preference and authorizes no execution.
