# JX Scientific Engine union

This document is the navigation and authority map for the scientific engine.
It joins the executable components into one system without copying bulk
research data, rewriting historical evidence, or treating every experiment as
qualified science.

This map is authoritative only for this scientific-engine branch; it does not
supersede other JX layers. The 2026-10-12 master registry's JX-3 row selects the
older engine commit `985249492cbbbfc0d2f4f0953b2b52150f6e25b5` and is
historical context for this recovered branch, not its current source selector.

The union is a chain of explicit contracts:

```text
portable scenario manifest
        |
        v
owned state + ordered force plan
        |
        v
integrator-route registry -> JXSimulation
        |
        v
common run envelope
        |
        +--> deterministic trajectory artifact
        +--> deterministic restart archive
        +--> canonical nonauthorizing receipt
                    |
                    v
             one .jxjob bundle
                    |
                    v
          exact-byte structural verifier
```

Every executable engine and job layer remains `SCREENING_ONLY`. Referenced
evidence retains its own recorded claim ceiling and gains no authority from the
union. A verified bundle is deterministically encoded `MODEL_OUTPUT` with
exact-byte content integrity for the supplied bytes. It is not authenticated
execution, astronomical evidence, a production ephemeris, or proof of a Planet
X object.

## Selection and verification order

These objects have different roles; the list is not a ranking of scientific
truth. When selecting among records that claim the same role, resolve and
verify them in this order rather than by filename age or directory name:

1. The exact Git commit containing the source and contracts.
2. Machine-readable force and integrator registries.
3. An externally SHA-256-bound portable scenario manifest.
4. An externally SHA-256-bound `.jxjob` bundle and its verified members.
5. Immutable evidence packages referenced by the experiment-data index.
6. Active workspace results, exploratory runs, and historical copies.

Each object must satisfy its own governing contract and cannot promote another
object. In particular, a filename containing `final`, a passing local test, or
a successful experiment does not override a registry claim ceiling or become
immutable-library authority.

## Identity anchors

- Release base: commit `aa36f8f1cb5852b8b5bb34c7ac476a8e35025e84`,
  tree `35af12ed2593979d45dd6148cdc20171063979dc`.
- Recovered test-custody baseline: commit
  `dcce239b7bfa59b297ed501a74761acd42141c2f`, tree
  `12796cea1a259838043dd8169fe89e2aa98da5a9`.
- Unified runtime-route baseline: commit
  `b77aad089c39a7331ea5f9f85fabe2e1e7ac6b79`, tree
  `ee744d975800c799827b5d461912a2c300d0bc3b`.
- Engine Job V1 implementation baseline: commit
  `7f04b0d7632ca143abea16486c10b763430327fd`, tree
  `9815e23145d589fce345f53f800e5836addf7df5`.
- This union-map seal is identified externally by the Git commit that contains
  it; a file cannot embed its own containing commit without a successor seal.
- Repository experiment navigation catalog:
  `catalog/JX_EXPERIMENT_CATALOG.json`, SHA-256
  `923a06bcf06b472813df32f709b49ae12495561292b5848c1b71c47314feaba4`.
- External immutable-library authority catalog: SHA-256
  `02290fb5d71ff2b1fa961bd72544ba47cc76fd7d086abe578600308f98016b34`.

These anchors identify source lineage or exact catalog bytes only. They are
not signatures, execution authentication, or evidence promotion.

## What is connected now

| Component | Current union boundary |
|---|---|
| State, units, coordinates, provenance | Immutable engine contracts and owned arrays |
| Force execution | Seven exact Python Force ABI v1 configurations |
| Integrator selection | Four exact routes in the integrator-route registry |
| Common execution | `JXSimulation` and `JXSimulationRun` |
| Portable execution | NumPy/CPU adaptive RKF78 through Engine Job V1 |
| Output packaging | Trajectory artifact, restart archive, receipt, and one `.jxjob` |
| Verification | Exact caller-supplied hashes, canonical bytes, and cross-bindings; no numerical replay |
| Test inventory | Exact non-overlapping profile manifest and isolated engine lanes |
| Research navigation | Non-destructive experiment-data index; authority catalog remains external |

The main implementation boundaries are:

- `src/jxplanetx/engine/force_abi.py`
- `src/jxplanetx/engine/integrator_registry.py`
- `src/jxplanetx/engine/simulation.py`
- `src/jxplanetx/engine/run_artifact.py`
- `src/jxplanetx/engine/simulation_archive.py`
- `src/jxplanetx/engine/runner.py`
- `src/jxplanetx/engine_job_cli.py`

The public command boundary is:

```text
jxplanetx engine list
jxplanetx engine run
jxplanetx engine verify
```

See [Dynamics scenarios](DYNAMICS_SCENARIOS.md) for the exact byte, resource,
filesystem, and recovery contracts, and [Engine API](ENGINE_API.md) for the
in-process interfaces.

## Connected but not yet portable through Engine Job V1

The unified runtime also exposes native GR15 Newtonian CPU, GR15 EIH 1PN CPU,
and GR15 EIH 1PN CUDA routes. Earth zonal and static lunar degree-2/3 forces
are registered in the generic force evaluator. These components are real
engine code, but their complete request and result contracts are not yet part
of the portable Engine Job V1 codec.

The coupled lunar state/physics bundle remains a separate typed execution
boundary. It has not yet been decomposed onto the common Force ABI and has no
shared restart or portable job contract.

The broader General Dynamics tracks include separately bounded cosmology,
fluid, reaction, and research-backend evidence. They belong to the research
platform but do not inherit the celestial engine's claim state or validation
scope, and they do not become engine routes merely by appearing in the same
index.

## Evidence and storage boundary

The union references evidence; it does not absorb raw payloads.

- [Experiment data index](JX_EXPERIMENT_DATA_INDEX.md) is the navigation layer.
- The immutable library authority catalog remains the source of truth for its
  retained packages.
- Active runs and results remain discovery records until separately promoted
  by an explicit protocol and authority record.
- Failed, stopped, and inconclusive runs remain evidence and must not be
  silently deleted or relabeled.
- Large arrays, checkpoints, virtual environments, toolchains, and duplicate
  worktrees stay outside the source commit.

This keeps the engine reviewable while preserving the expensive research that
led to it.

## Reasoning-assistant boundary

Janetx/JX assistant software is not part of the numerical authority. It may
list or explain a checksum-verified registered result record, but it must
consume a fixed result interface and preserve the record's claim ceiling. It
cannot select hidden physics, rewrite evidence, certify execution, or promote
model output into a scientific detection.

The current external assistant-bridge custody anchors are receipt SHA-256
`5309f1fea89b76e4ee3b31ac8e06d15c56979c088f32c68d91ca4649600e2f7e`
and catalog SHA-256
`35ecc4d28ffc7f21b08023d62527277f99b84690eb3934cabb94f09e540035fa`.
They identify bridge records only and grant no scientific authority.

## Future integration sequence (not implemented or authorized)

This is a candidate dependency order, not execution authority or an approved
roadmap.

1. Freeze the current portable NumPy/CPU RKF78 Engine Job baseline.
2. Add portable codecs and job receipts for the native GR15 CPU routes.
3. Define the compiled Force ABI shared by RKF78, GR15, and CUDA.
4. Add explicit cross-process resume lineage for long-running jobs.
5. Move coupled lunar physics onto the shared ABI and add its restart contract.
6. Bind separately governed evidence packages to engine releases without
   copying their bulk payloads into Git.

Each step requires its own tests, exact inventory update, claim boundary, and
independent audit. Later steps do not retroactively qualify earlier research.
