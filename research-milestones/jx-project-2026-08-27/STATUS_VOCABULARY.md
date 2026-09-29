# Status vocabulary

This vocabulary prevents an integrity or engineering label from being promoted
into a scientific claim.

In each `STATUS.json`, `canonical_publication_status` is this archive's
normalized, evidence-backed label for the milestone. It may aggregate several
historical source states and is not represented as a verbatim field from one
original artifact.

- `VALID` or `PASS`: the exact gate named by that artifact passed. It says
  nothing about gates that were not named.
- `REPLAY_EXACT`: two executions or records matched under the stated semantic
  projection. It is not independent physical replication.
- `ENGINEERING_SMOKE_VALID`: the short machinery test passed its engineering
  contract. It is not an astrophysical test.
- `ENGINEERING_LONG_INVALID`: the locked long engineering run failed a required
  gate and cannot be rehabilitated post hoc.
- `MIXED_OR_INCONCLUSIVE`: the registered diagnostic did not establish its
  reference conditions; descriptive patterns cannot be called accuracy or a
  mechanism finding.
- `BLOCKED`: prerequisites are unresolved. Blocked is not a null result.
- `DESIGN_REGISTERED_NOT_EXECUTABLE`: a design exists, but required inputs or
  activation conditions do not.
- `PRACTICALLY_SMALL`: an exact decision label for one locked synthetic
  endpoint. It does not mean no physical effect.
- `HISTORICAL_INVALID_NONAUTHORIZING`: retained for transparent failure lineage;
  publication does not activate or validate it.
- `NONAUTHORITY_REVIEW_ONLY`: static review bytes only; no permission to build,
  register, execute, or claim a result.
- `PRESERVATION_ONLY`: a recovery checkpoint, not an experiment or result.

No label in this archive means evidence for or against Planet Nine.
