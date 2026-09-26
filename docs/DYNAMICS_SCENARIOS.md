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
