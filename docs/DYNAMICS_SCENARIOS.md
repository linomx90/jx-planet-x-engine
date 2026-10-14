# JX dynamics scenarios V1

## Purpose

`DynamicsScenario` is the first public assembly layer above the JX force and
trajectory APIs. It makes one experiment legible as a single object:

- the exact initial `StateSnapshot`;
- the ordered `ForcePlan`;
- the adaptive `AdaptiveRKF78Spec`;
- a scenario identity, role, and description; and
- fixed scientific claim controls.

`MatchedScenarioComparison` binds a control and candidate into one narrow
ablation experiment. V1 answers only this question:

> Starting from the same retained state, with the same adaptive solver request
> and baseline forces, what checkpoint-state difference appears when named
> force terms are appended?

It does not answer which model is more accurate, whether the added physics is
real, or whether the engine matches an observational ephemeris.

## Public entry points

```python
from jxplanetx.engine import (
    DynamicsScenario,
    MatchedScenarioComparison,
    dump_dynamics_scenario_manifest,
    dynamics_scenario_manifest_sha256,
    load_dynamics_scenario_manifest,
    run_dynamics_scenario,
    run_matched_scenario_comparison,
)
```

The full executable synthetic example is
[`tests/test_engine_scenario.py`](../tests/test_engine_scenario.py). It compares
a Newtonian control with a candidate that appends the public cannonball SRP
term.

## Portable scenario manifests

`dump_dynamics_scenario_manifest()` converts a standalone or comparison-arm
scenario into canonical, newline-terminated ASCII JSON. Every binary64 scalar
uses Python's canonical hexadecimal representation, array values use explicit
C row-major order, and the document records the complete fixed RKF78 contract.
The V1 format supports NumPy/CPU binary64 and all three current public force
types: Newtonian point mass, restricted static-central 1PN, and cannonball SRP.

Loading is content addressed rather than name addressed:

```python
manifest = dump_dynamics_scenario_manifest(scenario)
identity = dynamics_scenario_manifest_sha256(manifest)
restored = load_dynamics_scenario_manifest(manifest, identity)
```

The loader rejects a wrong external SHA-256 identity, duplicate JSON keys,
noncanonical bytes, ordinary JSON floating tokens, noncanonical hexadecimal
floats, unsupported backends or methods, incomplete fixed contracts, and
incorrect JSON types. Rebuilt NumPy arrays are read-only. A caller that
deliberately changes valid values and recomputes the external hash has created
a new scenario identity; the format is content-addressed, not signed or
authenticated.

The manifest preserves exact public inputs. It does not preserve outputs,
certify code or dependency versions, authenticate provenance, or confer
accuracy, improvement, physics, registry, or production qualification.

## Engine Job V1: manifest to one bundle

Engine Job V1 is the narrow executable boundary for one portable scenario
manifest. It accepts only the portable NumPy/CPU binary64 adaptive-RKF78 route,
runs that scenario, and emits one deterministic `.jxjob` ZIP file. The outer
bundle has exactly these members:

- `trajectory.zip`, the deterministic trajectory artifact;
- `restart.zip`, the deterministic continuation archive; and
- `receipt.json`, the canonical record binding the manifest identity, force
  plan, run identity, member hashes, continuation lineage, and claim ceiling.

Install `jxplanetx[engine]` to use this numerical route. The base package CLI
and help remain importable without NumPy and report a controlled missing-extra
error if an `engine` command is invoked without it.

The manifest remains a separate input; it is not embedded in the `.jxjob`.
Run and verification therefore require a trusted manifest file and its
caller-supplied SHA-256. Verification additionally requires the caller-supplied
SHA-256 of the complete `.jxjob`:

```bash
jxplanetx engine list

jxplanetx engine run \
  --manifest scenario.json \
  --manifest-sha256 MANIFEST_SHA256 \
  --output result.jxjob

jxplanetx engine verify \
  --manifest scenario.json \
  --manifest-sha256 MANIFEST_SHA256 \
  --job result.jxjob \
  --job-sha256 JOB_SHA256
```

`engine list` reports all unified integrator routes and marks the one route
accepted by Engine Job V1. With an observable stdout, `engine run` prints the
completed job hash and creates exactly one output file. `engine verify`
verifies the supplied manifest and job without running the dynamics again.

### What verification establishes

Verification fails closed unless all of the following agree:

- the external manifest and job SHA-256 values and their exact input bytes;
- the canonical outer ZIP member roster, ordering, metadata, sizes, and
  member hashes;
- the canonical receipt reconstructed from the independently loaded members;
- the manifest, trajectory artifact, force plan, run identity, initial state,
  checkpoint schedule, static body data, and controller accounting; and
- the trajectory endpoint, continuation metadata, and restart-state arrays.

This is exact-byte content integrity and structural cross-binding. It does
not numerically replay RKF78, recompute the declared native-result digest,
authenticate who executed the run, reproduce or identify the source/runtime,
establish cross-platform replay, or prove accuracy or physical correctness.
The receipt and CLI report these controls explicitly as false. A party that
can alter valid content and supply new external hashes can create a different
internally consistent identity; SHA-256 is not a signature. The expected
manifest and job hashes must come from a trusted out-of-band channel when
identity matters.

Every artifact and verification result remains `SCREENING_ONLY`
`MODEL_OUTPUT`. Accuracy, improvement, physics, superiority, registry,
qualification, and production claims remain false.

### Resource and filesystem boundary

The shared runner/verifier policy places deterministic ceilings on manifest,
member, and bundle bytes; bodies; checkpoints; force models; steps;
rejections; encoded identity text; and a body-pair/model/stage work proxy.
These are contract and admission bounds, not guarantees of wall-clock time,
resident memory, allocator behavior, concurrency safety, or denial-of-service
resistance. A service still needs process isolation, timeouts, memory limits,
and concurrency limits.

The path-based CLI is Linux-specific. It pins each regular input inode through
`O_PATH` and `/proc/self/fd`, refuses symbolic or changing inputs, and checks
the byte ceiling while the descriptor remains held. The output parent must be
owned by the effective user and must not be group- or world-writable. Output
publication stages and synchronizes the complete bytes, then uses a
create-only hard link, requires staging-link removal, and synchronizes the
directory; an existing destination is never replaced.

A process failure after the hard-link publication point can leave a complete
final file—even if the success report was not observed—and a crash before
staging-link cleanup can also leave the hidden staging name. In that case the
exit status alone is ambiguous: inspect the final path and compare its exact
SHA-256 with the expected job hash before accepting it. Do not retry by
overwriting the path; publication is intentionally create-only. If no trusted
expected job hash is available, the file's presence alone is not verification.

## One standalone scenario

```python
scenario = DynamicsScenario(
    scenario_id="study.baseline",
    role="STANDALONE",
    description="Explicit Newtonian baseline",
    comparison_id=None,
    initial_snapshot=snapshot,
    force_plan=force_plan,
    integration_spec=rkf78_spec,
)

result = run_dynamics_scenario(scenario)
```

The result retains the scenario and the complete public `TrajectoryResult`.
It does not reduce the trajectory to a score.

## One matched comparison

The control and candidate are ordinary scenarios with roles `CONTROL` and
`CANDIDATE`. They use different plan IDs, but must share the exact same
snapshot and integration-spec objects. The candidate force tuple must contain
the exact control force objects as an unchanged prefix, followed by at least
one additional force:

```python
control = DynamicsScenario(
    scenario_id="study.control",
    role="CONTROL",
    description="Newtonian point-mass control",
    comparison_id="study.srp_ablation",
    initial_snapshot=snapshot,
    force_plan=control_plan,
    integration_spec=rkf78_spec,
)

candidate = DynamicsScenario(
    scenario_id="study.candidate",
    role="CANDIDATE",
    description="Control plus cannonball SRP",
    comparison_id="study.srp_ablation",
    initial_snapshot=snapshot,
    force_plan=candidate_plan,
    integration_spec=rkf78_spec,
)

comparison = MatchedScenarioComparison(
    comparison_id="study.srp_ablation",
    scientific_question="What state difference follows from adding SRP?",
    control=control,
    candidate=candidate,
    compared_body_ids=("SPACECRAFT",),
    added_force_model_ids=("force.nongrav.srp_cannonball",),
)

result = run_matched_scenario_comparison(comparison)
```

The result contains both complete trajectories and one
`ScenarioCheckpointDelta` for each selected body at each requested checkpoint.
Each delta holds the Euclidean norm of candidate-minus-control position and
velocity in the snapshot's declared units.

## Matching guarantees

V1 fails before integration unless all of these are true:

| Binding | Required condition |
| --- | --- |
| Roles | Exactly one `CONTROL` and one `CANDIDATE` |
| Comparison identity | Both scenario bindings equal the comparison ID |
| Initial state | The same exact `StateSnapshot` object |
| Integrator | The same exact `AdaptiveRKF78Spec` object |
| Backend | Equal explicit backend contracts |
| Baseline forces | Candidate preserves the control's exact ordered force-object prefix |
| Change | Candidate appends at least one force, named exactly by `added_force_model_ids` |
| Bodies | Every compared body exists in the shared ordered roster |
| Plan identity | Control and candidate have distinct plan IDs |

Execution order is fixed as control then candidate. After the control finishes,
the candidate starts from the control trajectory's retained internal copies of
the initial state, integration request, backend, and force prefix. This makes
the executed baseline identical rather than relying only on two descriptors
that point at caller-owned mutable arrays. Candidate-only force buffers remain
caller-owned and must stay stable for the complete comparison call.

The complete control and candidate trajectories still retain their own copied
inputs and step/force ledgers. The comparison does not hide or replace those
records.

## Difference arithmetic

For each selected body and checkpoint, V1 forms candidate minus control for
the three Cartesian position components and separately for velocity. It reads
those six backend scalars explicitly and evaluates each norm with Python's
overflow-resistant `math.hypot` in fixed component order. Full trajectory
arrays remain on the selected backend; a GPU comparison necessarily
synchronizes the named scalar components used in the public delta records.

These are binary64 model-to-model distances. V1 does not attach endpoint
uncertainty, truncation bounds, accumulated-roundoff bounds, dense-output
error, covariance propagation, or reference residuals.

## Fixed claim ceiling

Every scenario, result, comparison, and delta fixes:

```text
evidence_class = MODEL_OUTPUT
registry_authorized = false        # where present
qualification_authorized = false   # where present
accuracy_claimed = false
improvement_claimed = false
physics_claimed = false
qualified = false                  # where present
```

The comparison semantics are fixed as:

```text
MODEL_TO_MODEL_CHECKPOINT_DIFFERENCE_NOT_ACCURACY_OR_IMPROVEMENT
```

A smaller candidate-control difference is not a better orbit. A nonzero
difference is not evidence that the added force is physically correct. A
candidate that resembles an external reference has not been evaluated unless
that reference, metric, uncertainty model, and acceptance rule are separately
preregistered and executed.

## Deliberately excluded from V1

- full N-body EIH relativity;
- public spherical-harmonic or J2 forces;
- tidal, spin, mantle, or fluid-core state evolution;
- parameter replacement or fitting;
- changing initial conditions or tolerances between comparison arms;
- cross-integrator ranking;
- ephemeris residual scoring;
- uncertainty or covariance propagation;
- dense output, event location, or encounter handling; and
- scientific or production qualification.

The current eleven-body and lunar studies remain research packages outside
this public scenario surface. They can become public scenarios only after
their required forces and state contracts are promoted through separate
equation, implementation, convergence, and independent-validation gates.

## Next qualification steps

The scenario layer makes future qualification cleaner but does not by itself
perform it. The next engine-wide work should be:

1. retain and independently reproduce the
   [locked equal-binary qualification](../runs/jx_dynamics_scenario_equal_binary_v1/README.md)
   of the portable manifest plus public RKF78 path;
2. add broader independent analytic and external-engine suites that exercise
   scenario pairs without introducing a universal score;
3. promote a full, sourced EIH implementation and a general harmonics model
   through separate contracts and tests;
4. qualify dense output and event detection as independent numerical
   capabilities; and
5. only then assemble the eleven-body and lunar-rotation layers from those
   qualified pieces.
