# JX General Dynamics registry v17

Status: `PARTIALLY_QUALIFIED_RESEARCH_PLATFORM`

Scientific claim state: `SCREENING_ONLY`

This create-only export binds registry generation 17 to package release
candidate `0.6.0rc1`. It adds an explicit machine-readable top-level
`scientific_claim_state` and changes no capability status, scientific result,
evidence binding, or claim ceiling.

`production_ready` and `unified_multiphysics_execution` remain false. A
`QUALIFIED_BENCHMARK` row applies only to its exact frozen workload and does
not qualify the wider platform. Registry v16 and all earlier exports remain
unchanged historical records.

Regenerate the live registry only into a new, empty destination with
`benchmarks/jx_general_dynamics_registry.py`.
