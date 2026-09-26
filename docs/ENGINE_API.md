# JX experimental engine API

## Status and claim boundary

The `jxplanetx.engine` package is a **general-purpose experimental
force/right-hand-side and trajectory alpha**. It is not specific to Planet X. The current
release provides instantaneous acceleration evaluation, an adaptive
checkpoint trajectory integrator, an adaptive-RKF78 scenario and matched
additive-force comparison layer, a narrow fixed-step Newtonian KDK map, and
an ordered-Jacobi Wisdom--Holman KDK map for a strictly guarded hierarchical
Newtonian domain. It also provides one standalone adaptive full-Cartesian
Newtonian encounter-segment solver with an exact local-IVP clearance
certificate and bounded replay/accounting contract, plus one specific
whole-system Wisdom--Holman/RKF78 whole-step hybrid built from those frozen
lanes.
Force evaluation and RKF78 support NumPy CPU arrays and, when a compatible
CuPy installation and GPU are present, CuPy device arrays. KDK v1 is
NumPy/CPU-only, as are the standalone encounter segment, Wisdom--Holman map,
hybrid, Earth J2--J5, and static lunar degree-two/degree-three figure terms.

This alpha is not a top-tier-qualified celestial-dynamics system, an
ephemeris, an orbit-determination system, or an observation model. It has not
established accuracy, throughput, or GPU-speedup claims against external
engines. Every result is `MODEL_OUTPUT`, `registry_authorized=False`,
`qualification_authorized=False`, and `qualified=False`.

`evaluate()` and `evaluate_forces()` return accelerations only.
`integrate_trajectory()`, `run_dynamics_scenario()`,
`run_matched_scenario_comparison()`, `integrate_encounter_segment()`,
`integrate_kdk_trajectory()`,
`integrate_wisdom_holman_trajectory()`, and
`integrate_hybrid_wisdom_holman_rkf78_trajectory()` are separate, explicit
operations.
Their presence does not make this alpha a complete
orbit-propagation system: there is no dense output, event location, collision
response, regularization, or general event-driven close-encounter framework.
The hybrid is one exact whole-step state machine, not the broader declared
generic hybrid or encounter-switching capability. The KDK path is an
intentionally narrow map, not a general symplectic framework.

## Implemented alpha surface

| Area | Implemented behavior | Important boundary |
| --- | --- | --- |
| Gravity | Direct, tiled, unsoftened Newtonian point masses | Distinct coincident bodies and finite-radius contact fail closed |
| Relativity | Restricted static-central Schwarzschild 1PN correction | Central-body inertial frame, central source at coordinate zero and exactly static, massless targets, correction only |
| Radiation | Unshadowed isotropic cannonball SRP | Massive radiation source, massless targets, no eclipse, attitude, Poynting-Robertson, or thermal model |
| Earth figure | Axisymmetric unnormalized J2--J5 pair correction with a prepared pole policy | NumPy CPU, metre-second J2000 state, one Earth-target pair, no tesserals or tides |
| Lunar figure | Static unnormalized degree-two and degree-three principal-axis pair corrections | NumPy CPU, metre-second J2000 state, caller-bound orientation provider, no deformation or rotation integration |
| CPU | NumPy native arrays | CPU device only |
| GPU | Optional CuPy native arrays on an explicitly selected CUDA device | Code path exists, but it is not locally GPU-qualified and never falls back to CPU |
| Precision | Binary64 storage, compute, and accumulation | `float32`, mixed precision, and dtype aliases are rejected |
| Repeatability | Same runtime, device, software stack, force order, body order, and tile size | No cross-device or cross-backend bitwise guarantee |
| Composition | Exact built-in types in canonical order: Newtonian, one 1PN model, physical harmonics, SRP | Unknown types, duplicate model IDs, reversed order, unsupported backends, and missing dependencies fail closed |
| Integration | Fehlberg's 13-stage adaptive RK7(8), advancing the hatted eighth-order solution | One global adaptive step, exact checkpoints by clipping, no interpolation or event system |
| Scenario assembly | One RKF78 state/force/integrator request and same-state additive-force control/candidate comparisons | Model-to-model checkpoint differences only; no error, improvement, or physics claim |
| Standalone encounter segment | Full-Cartesian adaptive RK7(8), exact signed-duration accounting, pair/centroid defect control, and exact all-pair local-IVP clearance certificates | NumPy CPU only; mutual all-active positive-GM Newtonian scope; no event, collision, global-clearance, or hybrid-switching claim |
| Fixed-step mapping | Second-order kick-drift-kick for fully mutual Newtonian active bodies | NumPy CPU only; integer step-index outputs, mandatory encounter and resolution guards, no passive tracers |
| Hierarchical mapping | Second-order ordered-Jacobi interaction-kick/Kepler-drift/interaction-kick | NumPy CPU only; elliptic, low-secondary-mass, non-encounter domain; mandatory semantic replay; no passive tracers |
| Specific whole-step hybrid | Typed provisional Wisdom--Holman far probe with full-interval guarded Cartesian RKF78 replacement | NumPy CPU binary64, all-active mutual Newtonian, at most 16 bodies; no event/collision/global-clearance/general-switching claim |

The CuPy row means the code path is implemented, not that CuPy or a usable GPU
is installed. Runtime availability is checked on every explicit GPU request.

## Installation

CPU force evaluation and trajectory integration require NumPy:

```bash
python -m pip install -e '.[engine]'
```

For NVIDIA GPUs, install exactly one wheel family matching the local CUDA
runtime:

```bash
python -m pip install -e '.[gpu-cuda12]'
# or
python -m pip install -e '.[gpu-cuda13]'
```

Do not install both CuPy wheel families in one environment. A missing CuPy
package, missing GPU, inaccessible device, or invalid device index raises
`BackendUnavailableError`; JX does not substitute NumPy.

## Public force API

```python
from jxplanetx.engine import evaluate, evaluate_forces

result = evaluate(snapshot, force_plan)
```

`evaluate_forces(snapshot, force_plan)` is a descriptive alias. Neither
function accepts a tile-size override. Reduction geometry belongs to
`BackendSpec.tile_size`, is used by the evaluator, and is recorded in the
result and every ledger entry.

### `BackendSpec`

Required fields and fixed alpha policies:

| Field | Contract |
| --- | --- |
| `backend_id` | Exact lowercase `"numpy"` or `"cupy"` |
| `device` | `"cpu"` for NumPy; `"cuda"`, `"gpu"`, `"cuda:N"`, or `"gpu:N"` for CuPy |
| `tile_size` | Positive built-in integer; booleans are rejected |
| `dtype` | Exact built-in string `"float64"` |
| `allow_fallback` | Must be `False` |
| `deterministic_reductions` | Must be `True` |
| `fast_math` | Must be `False` |
| `determinism_scope` | Must be `"SAME_RUNTIME_DEVICE"` |

Tile size affects memory use, execution geometry, and floating-point grouping.
It is not a physical softening parameter. Runs with different tile sizes are
not claimed to be bitwise identical.

### `StateSnapshot`

`StateSnapshot` binds one Cartesian state to its scientific context:

- `snapshot_id`, `epoch`, `time_scale`, `frame`, `origin`, and `axes`;
- `length_unit`, `time_unit`, `mass_unit`, and `unit_system_id`;
- a unique ordered `body_ids` tuple;
- backend-native `positions` and `velocities` arrays of shape `(N, 3)`;
- backend-native `gravitational_parameters`, `masses`, and `radii` arrays of
  shape `(N,)`;
- a backend-native Boolean `massive` array of shape `(N,)`; and
- non-placeholder `Provenance`.

Every numerical state array must be native to the requested backend. Numeric
arrays are binary64; `massive` is Boolean. Values must be finite, masses,
radii, and gravitational parameters must be nonnegative, and every body marked
massive must have positive GM.

The snapshot descriptor is frozen and compares by identity, but JX does not
copy or freeze caller-owned arrays. The caller must keep them stable for the
entire evaluation. JX validates them in place and does not mutate them.

### Provenance and parameter metadata

`Provenance` requires a source ID, citation, version, and a lowercase,
nonzero SHA-256 digest. Every implemented force parameter roster requires
ordered `ParameterMetadata` entries. Each entry includes:

- the exact parameter ID and exact units;
- provenance;
- explicit uncertainty or `None`;
- explicit covariance-group ID or `None`; and
- a finite closed validity interval containing the snapshot epoch.

All model `unit_system_id` values must equal the snapshot value. JX does no
unit conversion. Callers must convert values before constructing a request.
At evaluation, units are bound exactly to the snapshot's unit strings:

| Parameter | Required units |
| --- | --- |
| State gravitational parameters | `<length_unit>^3/<time_unit>^2` |
| Speed of light | `<length_unit>/<time_unit>` |
| 1PN compactness and speed-fraction bounds | `1` |
| SRP reference pressure | `<mass_unit>/(<length_unit>*<time_unit>^2)` |
| SRP reference distance | `<length_unit>` |
| SRP area to mass | `<length_unit>^2/<mass_unit>` |
| SRP coefficient | `1` |

### Implemented force configurations

`FORCE_ABI_REGISTRY` and `list_force_abis()` expose force ABI v1. Each row
binds an exact configuration type to its model ID, canonical accumulation
rank, supported backend IDs, compatible integrator classes, and acceleration
semantics. This is the evaluator's real dispatch roster, but it is not yet a
compiled callback ABI shared by specialized GR15 and CUDA kernels.

`NewtonianPointMass` requires:

- nonempty, unique `source_ids` and `target_ids`;
- `unit_system_id`; and
- exactly one metadata entry, `state.gravitational_parameters`.

It is direct and unsoftened. There is no hidden `softening_length`. Selected
sources must be marked massive and have positive GM. Bodies omitted from
`source_ids` can be massless targets: they feel selected sources but exert no
backreaction. The only public singularity and collision policies are `error`.

`RestrictedStaticCentral1PN` requires:

- `central_source_id` and nonempty massless `target_ids`;
- caller-supplied positive `speed_of_light`;
- positive `maximum_compactness < 1`;
- positive `maximum_speed_fraction_squared < 1`;
- `unit_system_id`; and
- exact metadata in this order: `speed_of_light`,
  `maximum_compactness`, `maximum_speed_fraction_squared`.

The state must declare `origin == central_source_id` and
`frame == "CENTRAL_BODY_INERTIAL"`. The central position and velocity must
both be exactly zero. The kernel returns only the restricted Schwarzschild
correction and requires a compatible Newtonian base in the force plan. It is
not EIH N-body relativity, PPN beta/gamma, frame dragging, or an ephemeris
relativity model.

`CannonballSRP` requires:

- `radiation_source_id` and nonempty massless `target_ids`;
- positive `reference_pressure` and `reference_distance`;
- backend-native, target-aligned, binary64 `area_to_mass` and
  `radiation_pressure_coefficient` arrays;
- explicit `coefficient_convention` equal to `QPR` or `CR`;
- `attitude_model == "ISOTROPIC_CANNONBALL"`;
- `shadow_model == "NONE"`;
- `unit_system_id`; and
- exact metadata in this order: `reference_pressure`, `reference_distance`,
  `area_to_mass`, `radiation_pressure_coefficient`.

Area-to-mass and coefficient values may be zero, yielding an exact zero SRP
term for that target; negative or nonfinite values are rejected. Acceleration
is radial and away from the radiation source, with inverse-square distance
scaling.

`SolarJ2Force`, `EarthZonalJ2J5Force`, `LunarStaticDegree2Force`, and
`LunarStaticDegree3Force` bind retained Solar-System kernels to a `ForcePlan`.
The solar component accepts a resolved target roster and applies the
GM-weighted reaction to the Sun; the Earth and lunar components are pair
forces. Construct the retained physical force first, then call the
corresponding `bind_*` function with the exact `unit_system_id`, ordered
metadata, and an explicit pole/orientation provider ID. Reference-radius
metadata uses the snapshot length unit; coefficient and provider metadata use
unit `1`. The current adapters require `length_unit="M"`, `time_unit="S"`,
`axes="J2000"`, and the NumPy backend. They are correction-only, require the
source and target to participate in the Newtonian base plan, and never make a
coupled-rotation, deformable-Moon, ephemeris, or qualification claim.

### Coupled lunar state blocks

`CoupledLunarStateSnapshot` is the public engine contract for five simultaneous
blocks: three-body translational position and velocity, lunar mantle attitude,
mantle angular velocity, and fluid-core angular velocity. State ABI v1 requires
exact `SUN`, `EARTH`, `MOON` order, NumPy/CPU binary64, kilometres and seconds,
and explicit TDB, barycentric-inertial, system-barycenter, J2000 context. The
snapshot owns read-only copies and binds them to a `Provenance` record.

`integrate_deformable_coupled_lunar_state()` calls the retained v3 simultaneous
solver rather than duplicating its equations. It advances translation,
quaternion attitude, mantle and core rates on one fixed RKF78 delay lattice;
the delayed tide/spin deformation, changing mantle inertia, Earth/Sun lunar-
figure reactions and torques, Earth J2, and CMB pressure/viscous coupling are
evaluated together. The result retains typed checkpoints, the full native
diagnostic result, and a domain-separated content SHA-256 over every state
block.

```python
from jxplanetx.engine import (
    bind_coupled_lunar_state,
    integrate_deformable_coupled_lunar_state,
)

initial = bind_coupled_lunar_state(
    "lunar-screen", native_initial_state, provenance
)
run = integrate_deformable_coupled_lunar_state(
    initial, deformable_parameters, fixed_delay_spec, prehistory_provider
)
print(run.final_state.core_angular_velocity_body_s)
```

This component records
`NATIVE_COUPLED_PHYSICS_BUNDLE_NOT_FORCE_ABI_V1`: the coupled equations are a
real public execution path, but have not yet been decomposed through force ABI
v1 or made available to GR15/CUDA. It includes no geodetic transport, event or
collision system, coupled restart archive, ephemeris qualification, or claim
of superiority. Every result remains `SCREENING_ONLY`.

### Resolved-eleven coupled lunar EIH component

`LunarEphemerisV1State` and `integrate_lunar_ephemeris_v1()` are the first
public whole-system physical-model component built from the accepted lunar EIH
equations. The state contains the exact resolved-eleven translation roster plus
the lunar mantle quaternion, mantle angular velocity, and fluid-core angular
velocity. Every fixed RKF78 stage evaluates mutual Newtonian and EIH 1PN
gravity, reacting Sun/Moon and Earth/Moon static quadrupole interactions,
reacting fixed-axis Earth J2, and mantle/core pressure and viscous coupling.

```python
from jxplanetx.engine import (
    LunarEphemerisV1IntegrationSpec,
    LunarEphemerisV1Parameters,
    LunarEphemerisV1State,
    integrate_lunar_ephemeris_v1,
)

run = integrate_lunar_ephemeris_v1(initial, parameters, integration_spec)
print(run.final_state.positions_km)
print(run.result_content_sha256)
```

The route owns its inputs, fixes TDB/barycentric-inertial/J2000 context,
records its exact force roster and omitted-physics roster, accounts for all 13
force evaluations per accepted step, and produces domain-separated state and
step-ledger hashes. It is CPU/NumPy and screening-only. It is not named or
qualified as a production ephemeris: solar J2, Earth J3--J5, lunar degree
three and higher, time-variable deformation, delayed tides, minor bodies, observation
reduction, fitted parameters, continuation archives, events, and CUDA remain
outside this v1 component.

### Force plan and result ledger

`ForcePlan` contains a plan ID, one `BackendSpec`, and an ordered tuple of
models. It can emit only nonauthorizing `MODEL_OUTPUT`. Evaluation requires:

1. exactly one Newtonian base;
2. optional restricted or mutual EIH 1PN next;
3. optional Earth/lunar physical harmonics next;
4. optional cannonball SRP last;
5. correction sources contained in Newtonian sources; and
6. correction targets contained in Newtonian targets.

`ForceEvaluationResult` contains a backend-native total acceleration, each
backend-native contribution, an exact ledger, the backend/device/dtype and
tile size, fixed singularity/collision policy, determinism scope, and a
`StateMetadataBinding` with the snapshot's epoch, coordinate context, units,
body order, and provenance. Every ledger entry states its role, sources,
targets, assumptions, and accumulation order.

Results containing arrays compare by identity. JX does not define content
equality or a cross-device result hash in this alpha.

### Explicit continuation and restart archives

`JXSimulation.integrate()` always starts from its owned initial snapshot.
Continuation is a separate operation: `prepare_continuation(run_index)` binds
an exact successful NumPy/CPU endpoint to its parent result digest and returns
a read-only `JXSimulationContinuation`; `continue_from(run_index,
integrator_id, spec)` requires the new initial epoch to equal that endpoint and
records the parent and continuation-state identities in the resulting run.
No last-run or implicit continuation behavior exists.

`dump_simulation_archive(simulation, run_index)` returns deterministic restart
archive bytes. Archive v1 contains canonical JSON and fixed-order, uncompressed,
non-pickle NPY arrays. The caller records `simulation_archive_sha256(bytes)`.
Loading requires both that expected digest and the exact live `ForcePlan`:

```python
payload = dump_simulation_archive(simulation, run_index=0)
digest = simulation_archive_sha256(payload)
restart = load_simulation_archive(
    payload,
    expected_sha256=digest,
    force_plan=force_plan,
)
run = restart.simulation.integrate(integrator_id, next_spec)
```

The loader verifies the archive/member/state/force-plan/provenance identities
before constructing a simulation rooted at the archived endpoint. It never
unpickles code or reconstructs executable providers. CUDA archive creation is
refused because current device results have no host-verifiable content digest.
The archive is a numerical restart mechanism, not ephemeris qualification or
long-term physical validation.

## Public trajectory API

```python
from jxplanetx.engine import AdaptiveRKF78Spec, integrate_trajectory

result = integrate_trajectory(snapshot, force_plan, integration_spec)
```

The implemented method ID is
`integrator.adaptive.rkf78.fehlberg_1968`. The tableau is the classic
13-stage Fehlberg pair from *NASA-TR-R-287*, NTRS document 19680027281. JX
advances the hatted order-eight formula, whose distinguishing zero-based
stages are 11 and 12. The embedded ordinary order-seven formula uses stages 0
and 10. The signed local defect is always accepted order eight minus embedded
order seven. This orientation and the complete tableau are locked by tests;
they are not inferred from the ambiguous shorthand “RKF78.”

### `AdaptiveRKF78Spec`

The request supplies:

- a strictly monotone tuple of `checkpoint_epochs`, beginning exactly at the
  snapshot epoch;
- positive controller-proposal magnitudes `initial_step`, `minimum_step`, and
  `maximum_step` satisfying
  `minimum_step <= initial_step <= maximum_step`;
- separate backend-native, binary64, strictly positive componentwise
  `position_atol` and `velocity_atol` arrays of shape `(body_count, 3)`;
- separate positive scalar `position_rtol` and `velocity_rtol`;
- positive `maximum_steps` and nonnegative `maximum_rejections` budgets; and
- a safety factor below one, a minimum scale factor in `(0, 1]`, and a maximum
  scale factor at least one.

The normalized error is the maximum over all position and velocity
components. Each component uses
`atol + rtol * max(abs(current), abs(candidate))`. Accepted steps advance the
order-eight solution. The controller exponent is `1/8`.

Accepted position and velocity increments use backend-native, componentwise
Kahan accumulation, recorded by the fixed provenance value
`KAHAN_BACKEND_NATIVE_COMPONENTWISE`. Position and velocity carries begin as
positive binary64 zero, survive checkpoint copies, and change only when a
trial is accepted. A rejected trial's candidate state and carries are both
discarded. Tableau rows, the embedded solution, the signed defect, and the
adaptive controller remain the published fixed-order RKF78 operations.

The fixed time-step representation is `REPRESENTABLE_ENDPOINT_DELTA`. For a
nonclipped trial, the runtime first forms the binary64 endpoint from the
controller proposal and then passes exactly `endpoint - current_epoch` to
RKF78. If rounding would make that actual magnitude exceed the proposal, the
endpoint moves one `nextafter` value toward the current epoch and the delta is
recomputed. A clipped trial similarly uses exactly
`checkpoint_epoch - current_epoch`. If no bounded endpoint can advance in the
requested direction, execution fails closed.

`initial_step` and `minimum_step` govern controller proposals. An unavoidable
representable endpoint delta may be slightly smaller than its proposal, so
that quantization alone is not a minimum-step violation. `maximum_step` is
also a hard bound on the actual signed argument supplied to RKF78. Rejection
handling continues to enforce the proposal floor and requires a retry to
reduce the representable actual magnitude. An accepted checkpoint-clipped
residual preserves the pre-clip controller proposal exactly for the next
interval; forced output clipping neither grows nor shrinks the adaptive
proposal, and a tiny scheduling residual cannot create a microstep-growth
sequence. The fixed machine-readable policy is
`PRESERVE_PRECLIP_PROPOSAL_AFTER_ACCEPTED_CLIP`.

Each step is clipped to the next requested checkpoint. A checkpoint is
therefore an accepted RK endpoint represented by `TrajectoryCheckpoint`, not
an interpolated sample. Integration works in either monotone time direction.
A clipped checkpoint step may be shorter than `minimum_step`; if that trial
is rejected and cannot be reduced within policy, execution fails closed.
`maximum_steps` counts all attempted steps, and `maximum_rejections` is a
whole-trajectory rejection budget.

### Trajectory result and retained custody

`TrajectoryResult` contains exact checkpoint records, force-model identity
and ledger, attempted/accepted/rejected counts, 13 force evaluations per
attempt, backend/device/dtype identity, method and tableau identity, and
nonauthorization fields. Its complete `accepted_step_magnitudes` tuple stores
the absolute value of the signed step argument supplied to every accepted
RKF78 call. The fixed source label is
`ABS_SIGNED_RKF78_STEP_ARGUMENT`; accepted-step extrema and the maximum-step
cap are derived from that authoritative ledger. Because the RKF78 argument is
itself the representable endpoint delta, each stored magnitude must
bit-exactly equal `abs(endpoint - previous_endpoint)` using the same binary64
operands and order. The endpoint ledger remains strictly monotone and hits
every checkpoint exactly.

`accepted_step_ledger_content_sha256` binds the two complete ledgers, exact
`float.hex()` extrema, direction, step/evaluation counters, magnitude source,
and checkpoint counter-to-epoch bindings. Its fixed algorithm label is
`SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1`; the canonical sorted JSON payload
is prefixed by `jxplanetx.accepted-step-ledger.content-integrity.v1` and a NUL
byte. This is an unauthenticated content-integrity checksum, not an authority
claim, digital signature, or proof of origin. A caller that changes content
and coherently recomputes the checksum has defined a different unqualified
`MODEL_OUTPUT`; the checksum does not make such rewriting impossible. All
registry and qualification authorization fields remain false.

#### Accepted-step ledger checksum V1 byte recipe

The checksum preimage is exactly these bytes:

```text
UTF8("jxplanetx.accepted-step-ledger.content-integrity.v1")
+ 0x00
+ UTF8(canonical_json_payload)
```

`canonical_json_payload` is produced with exactly
`json.dumps(payload, sort_keys=True, separators=(",", ":"),
ensure_ascii=True, allow_nan=False)`. Dictionary keys are therefore sorted
lexicographically at every level and no insignificant whitespace is emitted.
Array order remains significant: accepted-step arrays retain integration
order, and checkpoint bindings retain checkpoint order. Because
`ensure_ascii=True` is fixed and every V1 key and value is ASCII, the UTF-8
payload bytes are also literal ASCII bytes.

The payload has exactly these keys and JSON types:

- `accepted_step_epochs_hex`: array of strings, one Python binary64
  `float.hex()` result per accepted endpoint;
- `accepted_step_magnitudes_hex`: array of strings, one `float.hex()` result
  per accepted signed-step magnitude;
- `checkpoint_endpoint_bindings`: array of objects with integer
  `accepted_steps`, string `epoch_hex`, integer `index`, and integer
  `rejected_steps`;
- `checksum_algorithm`: the string
  `SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1`;
- `counters`: an object containing integer `accepted_steps`,
  `attempted_steps`, `force_evaluations`, and `rejected_steps`;
- `direction`: the string `FORWARD` or `BACKWARD`;
- `extrema_hex`: an object containing string `last`, `maximum`, and `minimum`
  binary64 `float.hex()` values; and
- `magnitude_source`: the string `ABS_SIGNED_RKF78_STEP_ARGUMENT`.

V1 applies Python `float.hex()` directly, with no case conversion or numeric
normalization. It therefore preserves the sign of zero: positive zero is
`0x0.0p+0` and negative zero is `-0x0.0p+0`. All ledger floats are validated
as finite before serialization, and accepted magnitudes are additionally
strictly positive.

SHA-256 covers only the named step-ledger metadata above. It deliberately
excludes checkpoint state arrays, the initial snapshot, force plan, force
ledger, integration specification, tolerances, backend buffers, and every
other result field. In particular, the locked V1 payload does not include the
separately fixed `time_step_representation` or `checkpoint_proposal_policy`
fields. It is an unauthenticated integrity checksum over that limited
content—not a signature, authority token, or complete trajectory digest.

The result also retains independent backend-native copies of the initial
snapshot, force plan, SRP parameter buffers, and integration tolerances.
Caller-owned inputs are not mutated and cannot later rewrite those retained
result bindings. Checkpoints expose copies of the accepted high-word state;
the internal compensation carries continue across checkpoints but are not a
public restart state. No checkpoint-restart equivalence is claimed.

For NumPy, every retained array is owned, pairwise disjoint, C-contiguous,
read-only, and covered with the other retained result fields by
`result_content_sha256`. This unauthenticated content checksum detects a stale
or incoherently replaced result when
`validate_trajectory_result_integrity(result)` is called; it is not a signature
or proof of origin. CuPy exposes no equivalent read-only flag,
and JX does not copy numerical results to the host merely to hash them. A CuPy
result therefore has `result_content_sha256=None` and reports
`DEVICE_RESIDENT_OWNED_DISJOINT_NO_HOST_CONTENT_HASH`; its buffers are owned
and disjoint but remain directly mutable by the caller. This is an explicit
GPU custody limitation, not a hidden device-to-host synchronization.

Checkpoint position and velocity arrays stay on the selected backend and
device. The adaptive controller synchronizes only scalar decisions. There is
no implicit host/device conversion, CPU fallback, or float32 trajectory path.

This is a nonstiff, explicit, globally stepped integrator. A single rejected
trial governs all bodies. It has no dense output, root/event location,
collision response, close-encounter switching, regularization, symplectic
mapping, multirate stepping, or implicit solve. Coincidence and finite-radius
contact continue to fail in the force layer; the trajectory runtime does not
step through or resolve them.

## Public dynamics scenario API

`DynamicsScenario` assembles one existing snapshot, force plan, and adaptive
RKF78 request under an explicit role and description. A standalone scenario
runs through `run_dynamics_scenario()` and retains the complete underlying
`TrajectoryResult`.

`MatchedScenarioComparison` supports one deliberately narrow experiment: an
explicit control and a candidate that appends named force terms. The two arms
must share the exact snapshot and integration-spec objects, equal backend
contracts, and the exact ordered control force-object prefix. The candidate
is executed from the control result's retained copies of the state,
integration request, backend, and force prefix. This prevents the executed
baseline from silently changing between arms.

`run_matched_scenario_comparison()` runs control then candidate and returns
both trajectories plus fixed-order per-body checkpoint norms of candidate
minus control position and velocity. Those scalar records are explicitly
`MODEL_TO_MODEL_CHECKPOINT_DIFFERENCE_NOT_ACCURACY_OR_IMPROVEMENT`. Every
accuracy, improvement, physics, registry, and qualification control remains
false.

`dump_dynamics_scenario_manifest()` and
`load_dynamics_scenario_manifest()` provide the V1 portable boundary for these
scenarios. The canonical ASCII JSON uses exact hexadecimal binary64 values,
explicit C-order arrays, the complete fixed RKF78 contract, and an external
SHA-256 identity. Loading currently supports NumPy/CPU only and returns
read-only arrays. The hash detects changed bytes but is not a signature or a
scientific qualification.

Dynamics-scenario V1 is RKF78-only and additive-force-only. That scenario
format does not expose J2, full EIH, eleven-body, or coupled lunar state paths.
The complete contract, example, arithmetic policy, and next qualification
steps are in [Dynamics scenarios V1](DYNAMICS_SCENARIOS.md).

## Public standalone adaptive encounter-segment API

```python
from jxplanetx.engine import (
    AdaptiveEncounterSegmentSpec,
    EncounterExactRationalResourceSpec,
    integrate_encounter_segment,
)

result = integrate_encounter_segment(snapshot, force_plan, encounter_spec)
```

The method ID is
`integrator.adaptive.rkf78.encounter_segment_newtonian_v1`. This is a
standalone exact-duration segment solver, not the broader declared
`force.encounter.hybrid_switching` or
`integrator.hybrid.close_encounter` capability. Both `jxplanetx.engine.api`
and the `jxplanetx.engine` package root expose the exact ordered union of the
public `encounter_contracts` and `encounter` rosters, with identical object
identity.

### Closed encounter v1 domain and request

`AdaptiveEncounterSegmentSpec` requires this caller-supplied roster, with no
hidden floor or tolerance defaults:

- an immutable `body_order`, `initial_epoch`, independent `endpoint_epoch`
  label, and nonzero signed mathematical `duration`;
- positive `initial_step_magnitude`, `minimum_step_magnitude`, and
  `maximum_step_magnitude`;
- canonical-pair `pair_certification_floors`, `pair_position_atols`, and
  `pair_velocity_atols`, plus the corresponding positive relative
  tolerances;
- positive GM-centroid position and velocity absolute/relative tolerances;
- proposal, accepted-substep, rejection, consecutive-rejection, and actual
  force-evaluation ceilings;
- controller safety, minimum-scale, and maximum-scale factors; and
- one explicit `EncounterExactRationalResourceSpec`.

V1 accepts only exact built-in contract types, exact NumPy CPU C-contiguous
binary64 state arrays, and one exact NumPy/CPU `NewtonianPointMass` plan whose
sources and targets both equal `body_order`. Every body must be active, have
strictly positive GM, and participate in fully mutual direct unsoftened
Newtonian dynamics. The state must be barycentric inertial, with the required
origin, allowed axes/time scales, nonnegative radii, and force metadata valid
over the entire closed public-label interval. Passive or massless tracers,
softening, other forces, GPU arrays, float32, hidden transfers, and fallback
are rejected.

For each canonical pair `i < j`, the supplied positive binary64 floor
`rho_ij` must be exact-dyadically at least the sum of the two retained radii.
The accepted initial state, every trial stage, and every completed
eighth-order candidate must have exact-dyadic squared separation strictly
greater than `rho_ij**2`. A trial stage or candidate at or below a floor is a
recoverable shrink/reject condition when budgets permit; it is not an event
or collision determination.

### Authoritative time and endpoint label

State advances on a local offset lattice from exact zero to the retained
signed binary64 `duration`. The exact-rational sum of all accepted signed
binary64 substeps must equal that duration. The separately supplied
`endpoint_epoch` is an integrity-bound but unauthenticated external provenance
label: it must be finite and strictly monotone from `initial_epoch` in the
duration's direction, but it need not equal `float(initial_epoch + duration)`.
No state step is ever recovered from `endpoint_epoch - initial_epoch`.

Because the supported force is autonomous, every derivative evaluation uses
the bit-identical `initial_epoch` as its public metadata label. Exact local
offsets, signed proposals, accepted substeps, and the fixed RKF78 stage
coefficients retain the mathematical stage-time provenance separately. This
constant stage label does not weaken the preflight requirement that parameter
metadata cover the full public-label interval.

### Exact local-IVP clearance certificate

Before any force call for each proposal, V1 computes a simultaneous all-pair
certificate. With retained floor `rho_ik` and GM `mu_k`, it forms

```text
A_i = sum(k != i) mu_k / rho_ik^2
```

in immutable body order using custom exact rationals. For pair `i,j`, it
computes the exact minimum `ell_ij^2` of the signed relative linear path over
the proposed duration `Delta`, including the start, endpoint, and interior
minimum branches. The proposal is certified only when every canonical pair
satisfies the strict exact-rational inequality

```text
ell_ij^2 > (rho_ij + 0.5 * (A_i + A_j) * Delta^2)^2.
```

Equality is uncertified. The theorem says only that the exact Newtonian local
IVP issuing from the current accepted numerical node cannot cross any retained
pair floor during that proposed substep. It does not certify trajectory
accuracy, locate a minimum distance or event time, prove that a failed screen
will collide, establish clearance before the initial state or after the
segment endpoint, establish global or future physical-trajectory clearance,
or establish an IVP solution beyond the accepted local interval.

The accepted state uses the hatted order-eight Fehlberg solution and
componentwise NumPy binary64 Kahan accumulation. Error control is the maximum
of all canonical-pair relative position/velocity components and GM-centroid
position/velocity components. Defects come directly from the frozen RKF78
tableau's returned position and velocity defect arrays; they are never
recomputed as Kahan-updated accepted state minus embedded state. A rejection
commits no state, carry, local-offset, or duration-ledger change.

### Exact resources, accounting, replay, and custody

General certificate arithmetic uses canonical custom integer ratios. Duration
scheduling and radius/start/stage/candidate distance guards use canonical
exact dyadics. There is no `Fraction`, `Decimal`, third-party exact package,
opaque `gcd`, or floating-comparison fallback. Deterministic named abstract
work weights are qualification units for this algorithm; they are not Python,
C, CPU, machine-instruction, wall-time, or memory measurements.

The fixed v1 upper envelope is 16 bodies, 120 canonical pairs, 8,192-bit
integers, rational-exponent magnitude 4,096, 16,384 Euclidean iterations per
reduction, 128 proposals, and 1,664 force evaluations per execution lane.
Initialization is separately capped at 250,000 abstract work units, 500,000
GCD iterations, and 1,048,576 transcript bytes. Each proposal is capped at
650,000 work units, 500,000 GCD iterations, and 1,310,720 transcript bytes;
each lane is capped at 83,450,000 work units, 64,500,000 GCD iterations, and
168,820,736 transcript bytes. The caller may select smaller internally
consistent limits, never larger ones. These deterministic finite-work and
custody ceilings are not a denial-of-service, elapsed-time, or resident-memory
guarantee; an external service still needs process, time, and memory limits.

Per-proposal accounting records the exact disposition and actual derivative/
force calls. Certificate rejection uses zero calls; a stage-guard abort can
use zero through twelve; a derivative-domain abort can use one through
thirteen; only a completed RK attempt uses thirteen. Counts are never inferred
as 13 times all proposals.

One mandatory deterministic semantic replay starts again from owned input and
recomputes initialization, adaptive decisions, exact certificates, stage
guards, accepted states and carries, local offsets, dispositions, force
counts, and witness hashes. Primary and replay receive separate identical
caps and ledgers and cannot borrow resources from one another. Results expose
primary, replay, and exact public-total counts; at the hard ceiling public
totals are therefore at most 3,328 force calls, 166,900,000 abstract work
units, 129,000,000 GCD iterations, and 337,641,472 transcript bytes.

The result retains owned read-only state arrays, pair/schedule/result hashes,
bounded initialization and proposal records, force identity/ledger, the exact
accepted signed-substep ledger, and all three accounting views. Its
domain-separated SHA-256 values are unauthenticated content-integrity
checksums, not signatures, authority tokens, qualification, or proof of
physical correctness. Fatal contract, domain, step, or exact-resource failure
returns no partial result.

V1 has no dense output, interpolation, event detection or location, collision
response, merging, fragmentation, regularization, hybrid switching,
symplecticity, exact reversibility, global-clearance claim, registry authority,
or qualification authority. Every successful result remains unqualified
`MODEL_OUTPUT`.

## Public fixed-step KDK API

```python
from jxplanetx.engine import FixedStepKDKSpec, integrate_kdk_trajectory

result = integrate_kdk_trajectory(snapshot, force_plan, kdk_spec)
```

The additive method ID is `integrator.symplectic.kdk_leapfrog_2`. It applies
the second-order composition kick `0.5`, drift `1.0`, kick `0.5`, caching the
initial acceleration and evaluating acceleration once after each drift.
Exactly `completed_steps + 1` force evaluations are therefore required. It
is an adaptive-free KDK map, not a new force model, a variable-step method, a
Wisdom-Holman splitting, or the broader declared
`integrator.symplectic.split` capability.

The exact canonical convention is GM-scaled. For each active body let
`mu_i = GM_i > 0`, `p_i = mu_i v_i`, and
`H = sum_i |p_i|^2/(2 mu_i) - sum_{i<j} mu_i mu_j/r_ij`. The first kick maps
`p_i <- p_i - (h/2) grad_i V(q)`, equivalently
`v_i <- v_i + (h/2) a_i(q)`; the drift maps
`q_i <- q_i + h p_i/mu_i`; the final kick repeats the first form at the new
positions. This is the precise scaled Hamiltonian for the exact-arithmetic
symplectic statement. `StateSnapshot.masses` remains retained provenance but
is not used by KDK v1 arithmetic; GM supplies the canonical weights.

The official [REBOUND LEAPFROG documentation](https://rebound.hanno-rein.de/integrators/leapfrog/)
is an external behavior oracle for the public expectations “second order,”
“symplectic,” and one new force evaluation per step. REBOUND itself is
published in the [official repository](https://github.com/hannorein/rebound)
under its [GPL license](https://github.com/hannorein/rebound/blob/main/LICENSE).
JX's independently licensed KDK equations, implementation, and tests were
written as a clean-room vertical slice; no REBOUND source code was
copied. This provenance statement makes no broader license or equivalence
claim.

### Closed KDK v1 scope

KDK v1 accepts only:

- `BackendSpec(backend_id="numpy", device="cpu", ...)` and exact base-class
  binary64 NumPy arrays (array subclasses fail preflight);
- exactly one `NewtonianPointMass` model;
- `source_ids == target_ids == snapshot.body_ids` in exact order;
- at least two bodies, every one marked active/massive and having positive
  GM;
- `frame == "BARYCENTRIC_INERTIAL"`, `origin == "BARYCENTER"`, axes equal
  `"ICRS_ALIGNED"` or `"CARTESIAN_RIGHT_HANDED"`, and a continuous
  coordinate-time label of `"TDB"` or `"SYNTHETIC"`; and
- metadata validity covering the entire fixed-step schedule.

Passive or massless tracers, central-source/test-particle plans, 1PN, SRP,
velocity-dependent forces, CuPy, collisions, and close-encounter switching
are explicitly deferred. In particular, v1 has no Sun-plus-massless-particle
validation claim. The existing RKF78 path remains the public choice for
currently implemented non-Newtonian or passive-target plans.

### `FixedStepKDKSpec` and schedule semantics

The request contains exactly these caller controls:

- a strictly increasing immutable `checkpoint_step_indices` tuple beginning
  at zero and containing at least one later integer map index;
- a finite nonzero built-in binary64 `fixed_step`; its sign determines
  `FORWARD` or `BACKWARD`;
- a positive `maximum_steps` budget no smaller than the final output index;
- a positive `minimum_swept_pair_separation` in the snapshot length unit; and
- a dimensionless positive `maximum_pair_frequency_step` no greater than
  one.

The identical signed binary64 `fixed_step` is used in every map. Checkpoint
epochs are derived before state arithmetic as
`float(initial_epoch + fixed_step * step_index)`. Every adjacent binary64
epoch must advance in the requested direction. Outputs exist only at the
requested integer indices: KDK never clips a map step and provides no dense
output or interpolation.

Both safety controls are mandatory. For each pair, the runtime minimizes
separation along the complete linear relative drift segment, not merely at
its endpoints. That swept distance must be strictly greater than both the
caller floor and the pair's retained radius sum. It must also satisfy
`abs(h) * sqrt((GM_i + GM_j) / r_min^3) <= maximum_pair_frequency_step`.
The initial state is checked with the same rules. A violation raises
`TrajectoryDomainError`; KDK does not soften, shorten, retry, switch methods,
or step through the encounter.

### KDK result, claims, and custody

`KDKTrajectoryResult` is separate from the adaptive `TrajectoryResult`. It
retains exact base-class NumPy buffers that own their memory, are read-only,
and are pairwise nonoverlapping for the initial state and every checkpoint
state; a copied force plan and spec; the initial force
ledger; the exact integer/epoch lattice; the exact force count; and
per-pair swept minimum separations in deterministic `(i,j)` order. The scalar
minimum separation and maximum pair-frequency observation are derived from
that per-pair ledger, so contact radii from unrelated pairs are never mixed.

The result records `exact_arithmetic_symplectic=True` and
`exact_arithmetic_time_reversible=True`. Those are mathematical statements
about the fixed-coefficient composition. It also fixes
`floating_point_symplectic=False`; no bitwise reversibility, long-arc
qualification, registry authority, or scientific superiority follows.
Every result remains `MODEL_OUTPUT`, `registry_authorized=False`,
`qualification_authorized=False`, and `qualified=False`.

Two independent unauthenticated checksums are retained:

- `schedule_content_sha256` uses algorithm
  `SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1` and domain
  `jxplanetx.kdk-schedule.content-integrity.v1`. Its sorted JSON payload binds
  the checkpoint indices and binary64 epoch hex values, fixed-step hex,
  direction, method ID, completed-step count, and force-evaluation count.
- `result_content_sha256` uses the same JSON/float-hex algorithm and the
  distinct domain `jxplanetx.kdk-result.content-integrity.v1`. Its sorted JSON
  payload binds the retained snapshot metadata, provenance, body order and
  every numerical state buffer; backend and force-plan/spec controls;
  complete force-ledger and state-metadata fields; every checkpoint's index,
  epoch, counters, and position/velocity components; per-pair guard evidence;
  schedule checksum; runtime accounting; method identity; and every result
  claim/authority control.

For both V1 recipes the preimage is `UTF8(domain) + 0x00 + UTF8(payload)` and
the payload is serialized exactly with
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`. Every binary64 value uses Python `float.hex()` without
normalization; array and tuple order is significant. The two domains prevent
substitution between the narrow schedule checksum and full retained-content
checksum.

These are content-integrity checks, not authentication, signatures, authority
tokens, or proofs of origin. `KDKTrajectoryResult.__post_init__` recomputes
both and rejects an old digest paired with altered content. A caller who
coherently rewrites content and recomputes the applicable digest has created
a different, still-unqualified `MODEL_OUTPUT`; the checksums do not make that
impossible.

## Public ordered-Jacobi Wisdom--Holman API

```python
from jxplanetx.engine import (
    FixedStepWisdomHolmanSpec,
    integrate_wisdom_holman_trajectory,
)

result = integrate_wisdom_holman_trajectory(snapshot, force_plan, wh_spec)
```

The concrete method ID is
`integrator.symplectic.wisdom_holman_jacobi_kdk_2`. It is an additive,
second-order ordered-Jacobi map, not an implementation of the generic
`integrator.symplectic.split` declaration, WHFast, a close-encounter method,
or a replacement for RKF78. The exact composition is interaction kick
`B(h/2)`, analytic Kepler and center-of-mass drift `A(h)`, then
interaction kick `B(h/2)`. The same signed built-in binary64 `h` is used for
every map step; output nodes are immutable integer step indices with no
clipping, interpolation, subdivision, adaptive fallback, or restart claim.

### Canonical convention and exact split

Body order is scientific input and immutable. Body zero is the dominant
central body. For `mu_i = GM_i > 0`, define

- `K_i = sum_{j=0}^i mu_j` and
  `lambda_i = mu_i K_{i-1}/K_i` for `i >= 1`;
- prefix barycenters `R_i = sum_{j=0}^i mu_j x_j/K_i` and
  `V_i = sum_{j=0}^i mu_j v_j/K_i`;
- `Q_0 = R_{N-1}`, `P_0 = K_{N-1} V_{N-1}`; and
- `Q_i = x_i - R_{i-1}` and
  `P_i = lambda_i (v_i - V_{i-1})` for `i >= 1`.

Cartesian canonical momentum is GM-scaled, `p_i = mu_i v_i`.
`StateSnapshot.masses` is retained provenance and is unused by the map. The
Hamiltonian split is exactly

`H_A = |P_0|^2/(2 K_total) + sum_{i>=1}
[|P_i|^2/(2 lambda_i) - lambda_i K_i/|Q_i|]`

and

`H_B = -sum_{a<b} mu_a mu_b/|x_a-x_b|
+ sum_{i>=1} lambda_i K_i/|Q_i|`.

The transform and its analytic inverse use fixed scalar/body/component order.
`JacobiCoordinateBinding` retains the exact GM roster, cumulative `K_i`,
`lambda_i`, both transform matrices, body order, and a content checksum.
Returned nodes are synchronized barycentric Cartesian states, not exposed
unsynchronized map coordinates.

The residual kick is also fixed: with full Cartesian acceleration `a_i`,
`f_i = mu_i a_i`, `F_full,Q = A^T f`,
`F_K,i = -lambda_i K_i Q_i/|Q_i|^3`, and
`F_interaction,i = F_full,Q,i - F_K,i`; the closed-system translation
component is validated and set exactly to zero. Each `A` flow solves the
bound elliptic two-body problem independently and drifts the center of mass.

### Universal-variable Kepler contract

`UniversalKeplerSolverSpec` is fixed, not a tolerance-tuning surface. For a
relative state `(q_0,u_0)` and `k=K_i`, it requires
`b = 2k/|q_0| - |u_0|^2 > 0`, `eta=q_0 dot u_0`, and solves

`R(s)=|q_0|s + eta G_2 + (k-b|q_0|)G_3 - h = 0`,

with positive derivative
`R'(s)=|q_0| + eta G_1 + (k-b|q_0|)G_2`.
The signed bracket expands at most 32 times; at most 96 safeguarded
Newton/bisection iterations are permitted. The normalized functions are
`G_n(b,s)=sum_{m>=0}(-b)^m s^(n+2m)/(n+2m)!`. At
`|sqrt(b)s| <= 0.5`, exactly 64 ascending binary64 series terms are summed;
the larger-argument branch uses the fixed trigonometric identities. A result
is accepted only when its residual and settled-bracket/two-cycle conditions
both hold, followed by finite radius, Lagrange identity, two-body energy, and
angular-momentum postconditions. Any nonfinite value, missing bracket,
stagnation, cap exhaustion, nonpositive radius, or failed postcondition aborts
transactionally; no partial trajectory is returned.

### Closed v1 domain and guards

The v1 call accepts only exact base-class, owned-on-result NumPy binary64 CPU
arrays, one fully mutual `NewtonianPointMass` plan, and
`source_ids == target_ids == snapshot.body_ids == jacobi_body_order` in exact
central-first order. Every body must be active and have positive GM. The
snapshot must be barycentric inertial Cartesian with origin `BARYCENTER`, axes
`ICRS_ALIGNED` or `CARTESIAN_RIGHT_HANDED`, and time scale `TDB` or
`SYNTHETIC`. Physical masses are retained but unused. CuPy, massless/passive
test particles, 1PN, SRP, velocity-dependent forces, parabolic/hyperbolic
Kepler flows, coordinate reordering, and encounter switching are unavailable.

Before arithmetic, total secondary GM divided by primary GM must be at most
`0.01`, and the GM-weighted barycenter position and velocity must satisfy both
caller caps and a fixed `256*N*epsilon` pair-extent cap. At the initial state,
after every first half-kick before its Kepler drift, and at every completed
map node as specified by the retained cadence policy, the runtime requires:

- negative Jacobi specific energy; positive, strictly increasing osculating
  semimajor axes in immutable order; eccentricity at most `0.9`; and positive
  periapse no smaller than the caller floor;
- residual-interaction acceleration relative to Kepler acceleration at most
  `0.1`;
- `|h|/P_i <= 1/20` and `|h|/tau_peri,i <= 1/16`, where
  `P_i=2*pi*sqrt(a_i^3/K_i)` and
  `tau_peri,i=2*pi*sqrt((a_i^3/K_i)(1-e_i)^3/(1+e_i))`; and
- every secondary pair farther apart than three mutual Hill radii, in
  addition to the radius-sum and caller encounter floors.

Each Kepler drift also applies the retained curved-path displacement bound
and an outward `256*N*epsilon` margin to its start, end, and lower-bound
separations. The exact-arithmetic path-length bound is conservative; its
binary64 evaluation is not directed interval arithmetic and is not a
continuum-proof claim. Violations fail closed. There is no automatic
regularization, step reduction, hybrid switch, or rescue method.
V1 also has no automatic timestep-resonance detector: the `h/P` and periapse
screens do not exclude symplectic step-size resonances. Scientific use
therefore requires a preregistered `h`, `h/2`, `h/4` convergence scan and,
where appropriate, phase-shifted or small-neighborhood step checks.

### Accounting, custody, and claim boundary

The primary map evaluates the full Newtonian force exactly `completed_steps+1`
times. Result construction then performs one mandatory, full deterministic
semantic replay and exact-compares the retained state, guard evidence,
accounting, and checksums. Accordingly the public call performs another
`completed_steps+1` force evaluations: the total is
`2*(completed_steps+1)`. The result names primary-map, validation-replay, and
total-public-call counters separately for force evaluations, interaction
assemblies, Kepler solves, solver/bracket/G-function work, coordinate
transforms, and node/path guards. Per-call timing or performance analysis must
include both passes.

`WisdomHolmanTrajectoryResult` retains exact owned, read-only, nonoverlapping
buffers and three domain-separated, unauthenticated SHA-256 content-integrity
bindings: the coordinate binding, the static integer schedule, and the full
result. Dynamic solver work belongs to the full result digest, not the
schedule digest. The payload uses sorted finite JSON and binary64
`float.hex()` values. Validation recomputes the digests and the scientific
semantics. These hashes are not authentication, signatures, authority, or
proof of origin; coherent rewriting plus recomputation creates a different,
still-unqualified `MODEL_OUTPUT`.

The result fixes `formal_exact_kepler_subflow_symplectic=True` and
`formal_exact_kepler_subflow_time_reversible=True` only for the mathematical
exact subflows and symmetric composition. It also fixes
`floating_point_symplectic=False` and
`floating_point_exactly_reversible=False`. Sampled bounded energy error does
not establish small phase or Cartesian trajectory error, and one sampled
long-arc run does not establish general long-term boundedness, stability,
accuracy, production suitability, or superiority. Authority, qualification,
production, reference-truth, and superiority flags remain false;
`integrated=True` means only that the trajectory was materialized.

The equations and tests were independently implemented for JX
from the primary [Wisdom--Holman map paper](https://web.mit.edu/wisdom/www/nbodymap.pdf),
the [WHFast Jacobi-coordinate analysis](https://arxiv.org/abs/1506.01084), and
the [universal-variable formulation](https://arxiv.org/abs/1508.02699).
Official [REBOUND WHFast documentation](https://rebound.hanno-rein.de/integrators/whfast/)
is an external behavior oracle only. No REBOUND source was copied, linked, or
vendored. REBOUND remains a separate GPL-v3-family project; its tagged
[5.1.1 license text](https://github.com/hannorein/rebound/blob/5.1.1/LICENSE)
does not alter JX's proprietary license. This statement makes no legal conclusion or
finite-step map-equivalence claim.

## Public whole-step Wisdom--Holman/RKF78 hybrid API

```python
from jxplanetx.engine import (
    HybridEncounterControlProfile,
    HybridWisdomHolmanRKF78Spec,
    integrate_hybrid_wisdom_holman_rkf78_trajectory,
)

hybrid_spec = HybridWisdomHolmanRKF78Spec(
    wisdom_holman_spec=wh_spec,
    encounter_control=encounter_control,
)
result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
    snapshot,
    force_plan,
    hybrid_spec,
)
```

The exact method ID is
`integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1`. Both
`jxplanetx.engine.api` and the package root expose, with identical object
identity and order, the complete public union of `hybrid_contracts` and
`hybrid`. This concrete method is not the declared generic
`integrator.hybrid.close_encounter`, the declared event-oriented
`force.encounter.hybrid_switching`, a regularizer, or a collision handler.

`HybridWisdomHolmanRKF78Spec` composes one exact frozen
`FixedStepWisdomHolmanSpec` with one exact
`HybridEncounterControlProfile`. The profile is precisely the 21
interval-invariant caller fields of `AdaptiveEncounterSegmentSpec`; body order,
signed duration, and public start/end labels are derived for each near child.
There are no hidden switching floors, tolerances, or controller/resource
defaults beyond the frozen child contracts.

### Fixed outer lattice and transactional mode decision

The signed Wisdom--Holman `fixed_step` is the authoritative mathematical
duration `h` for every outer interval. Public node labels are independently
computed as `float(t0 + k*h)` from the original snapshot epoch and integer
outer index; state advance is never recovered from a rounded difference of
two labels. Checkpoints remain exactly the immutable outer indices requested
by the nested Wisdom--Holman spec. Near substeps produce no public checkpoint,
and there is no clipping, interpolation, dense output, or event node.

At each outer step, the runtime performs exactly one typed, complete,
provisional Wisdom--Holman probe in this order: first kick, predrift node
screen, center-of-mass and ordered Kepler drift, first Cartesian
reconstruction, one pure pre-force swept-path screen, candidate force and
interaction assembly, second kick, final reconstruction, and completed-node
screen. The first probe and the first probe after a near interval perform one
exact accepted-start rebind; a continuous far streak reuses its accepted
private cache.

- `FAR_PASS` commits the complete candidate atomically.
- `NEAR_SWITCH` discards every candidate array and all provisional state, then
  runs one private guarded encounter execution from the untouched committed
  original Cartesian node for the full signed duration `h` and explicit next
  global-lattice label.
- `FATAL_FAILURE` returns no partial result. Static contract/backend/force or
  metadata failures, nonfinite/singular arithmetic, Kepler/root/transform
  failures, initial encounter-floor violations, child step or exact-resource
  exhaustion, custody/checksum failures, and replay mismatches never switch.

Only finite named dynamic Wisdom--Holman envelope exits may select near mode.
Reasons, detection phase, canonical body indices, and canonical body pairs are
closed typed records; exception message text is never parsed. A contact or
certification-floor violation already present at the original committed node
is fatal because rerunning the same initial condition cannot repair it. There
is no hysteresis, latch, mode memory, group subdivision, partial-prefix commit,
event location, or minimum-distance search. After a near commit, the next
outer probe performs exactly one fresh Wisdom--Holman rebind.

### Near certificate, custody, and accounting

Each near replacement derives a complete exact
`AdaptiveEncounterSegmentSpec` with the Wisdom--Holman body order, duration
exactly `h`, and the independently computed global-lattice endpoint label.
Fresh positive-zero Kahan carries are used for the whole replacement. The
private execution uses the frozen full-Cartesian, all-active positive-GM,
mutual unsoftened Newtonian algorithm once. It never constructs a public
`EncounterSegmentResult` and therefore performs no nested public child replay.

The exact clearance theorem remains strictly local: for each accepted
numerical node and accepted substep, it proves retained-floor noncrossing for
the exact Newtonian local IVP issuing from that node. It does not identify a
collision or event, prove that an uncertified proposal will collide, locate a
minimum separation, certify numerical trajectory accuracy, or establish
global or future physical-trajectory clearance.

Every outer record retains its mode, typed decision, reason and triggers, all
19 provisional-work counters, ordered per-secondary Kepler work records, and
whether that work was committed or discarded. A near record additionally
binds its exact derived child spec, accepted signed-step ledger, bounded child
summary, and seven domain-separated digests over streamed private execution
content. Accepted far work, discarded far-probe work, private near work,
coordinate transforms, near accepted substeps, and checkpoints remain
separate accounting lanes.

Result construction performs one mandatory full hybrid semantic replay from
owned input. That replay recomputes the integer label lattice, every far/near
decision and work record, every near child, all accepted states and
checkpoints, all digests, and all accounting. Primary and replay lanes receive
separate budgets and public totals are their exact sum. There is exactly one
outer replay and no nested public Wisdom--Holman or encounter replay.

Hard library ceilings apply independently to primary and replay: 65,536 outer
records, 4,096 near macrosteps, 524,288 retained near accepted substeps, and
917,504 retained raw digest bytes per lane. Public-total ceilings are exactly
131,072, 8,192, 1,048,576, and 1,835,008 respectively. Runtime checks are
prospective and fail without partial output. These logical ceilings do not
guarantee elapsed time, resident memory, allocator behavior, concurrency
safety, or denial-of-service resistance. A service must impose external
process isolation, timeout, memory, and concurrency limits.

When every dynamic guard passes, the accepted-state, Cartesian checkpoint,
scalar-diagnostic, force-ledger, checksum, and primary/replay/public
Wisdom--Holman accounting projection is bit-identical to the frozen public
Wisdom--Holman run; the only cadence change is the pure swept-path screen being
performed before candidate force so a near path can return without entering a
singular endpoint force. Hybrid-only mode and discarded-work ledgers are
additional content.

The complete hybrid fixes every symplectic and reversibility claim false. It
is not globally symplectic, formally or exactly reversible, or globally
order-qualified. It has no dense output, event detection or location, collision detection or
response, merging, fragmentation, regularization, global-clearance claim,
superiority claim, registry authority, production
authority, or qualification authority. The all-far projection does not turn a
run containing near replacements into a symplectic map, and every result
remains unqualified `MODEL_OUTPUT`.

## Array residency and GPU rules

The selected backend owns every numerical array boundary:

- NumPy requests accept only NumPy arrays.
- CuPy requests accept only CuPy arrays on the selected device.
- Mixed host/device inputs are errors.
- JX never calls `numpy.asarray`, `cupy.asarray`, `cupy.asnumpy`, or `.get()`
  to rescue a mismatched request.
- GPU scalar domain checks may synchronize a zero-dimensional Boolean with
  `.item()`; user arrays and numerical results remain device resident.
- CuPy allocations occur inside a scoped device context, which is restored on
  exit.

Transfers are therefore caller-controlled and explicit. Optional CPU/GPU
tests compare all implemented terms when a usable CuPy runtime is present,
but they do not assert cross-backend bitwise identity.

## Errors and fail-closed behavior

| Error | Meaning |
| --- | --- |
| `ContractError` | A typed descriptor is incomplete or ambiguous |
| `BackendError` | Backend ID, device, precision, or arithmetic policy is invalid |
| `BackendUnavailableError` | The explicitly requested runtime/device cannot be used |
| `BackendArrayError` | An input is not native to the selected backend/device |
| `EvaluationError` | State, model order, dependencies, units, validity, or scope are inconsistent |
| `UnsupportedForceError` | A force object has no exact built-in implementation |
| `DuplicateForceError` | A model ID is applied twice |
| `ForceDomainError` | Finite inputs violate a physical/numerical domain |
| `ForceSingularityError` | Distinct selected source and target positions coincide |
| `ForceCollisionError` | Selected finite-radius bodies overlap or touch |
| `TrajectoryContractError` | A checkpoint lattice/schedule, tolerance, retained binding, force compatibility rule, or result accounting field is inconsistent |
| `TrajectoryDomainError` | An RK stage/controller value, KDK state/encounter/frequency guard, or Wisdom--Holman hierarchy/Kepler/encounter guard leaves its permitted finite domain |
| `TrajectoryStepLimitError` | A binary64 epoch cannot advance or a declared step/rejection budget is exhausted |
| `EncounterContractError` | A standalone encounter request, retained binding, exact schema, or result/replay accounting field is inconsistent |
| `EncounterDomainError` | An accepted local state or unrecoverable trial leaves the finite guarded Newtonian domain |
| `EncounterResourceError` | A prospective exact-work, GCD, integer/exponent, transcript, witness, body, or pair ceiling is exhausted |
| `EncounterStepLimitError` | A proposal, acceptance, rejection, force-call, minimum-step, or exact-duration progress ceiling is exhausted |
| `HybridContractError` | A hybrid request, fixed outer lattice, typed record, custody binding, checksum, or replay/accounting relation is inconsistent |
| `HybridDomainError` | A fatal static or numerical domain failure prevents either a valid far commit or an admissible full-interval near replacement |
| `HybridStepLimitError` | An outer-record, near-macrostep, retained-substep, retained-digest, or child execution ceiling is exhausted |
| `LinearizedFitContractError` | A linearized fit request has invalid identities, units, shapes, scales, weights, provenance, limits, or prior |
| `LinearizedFitRankError` | The scaled and whitened design matrix is not full column rank at the caller's threshold |
| `LinearizedFitConditionError` | A full-rank fit exceeds the caller's maximum condition number |
| `LinearizedFitNumericalError` | The NumPy linear algebra runtime fails or returns a non-finite result |

JX does not soften, merge, regularize, skip, transfer, downgrade precision, or
change backend as an error-recovery side effect.

## Capability catalog

`list_capabilities()` returns a stable 51-row catalog spanning 19 families.
Catalog membership is not execution authority. Every row has maturity
`UNQUALIFIED`. Exactly eighteen rows are marked `IMPLEMENTED`: seven force
models, NumPy, the optional CuPy code path, binary64, same-runtime/device
repeatability, the adaptive Fehlberg RK7(8) trajectory integrator, the
standalone guarded Newtonian encounter segment, and the narrow NumPy/CPU
mutual-Newtonian KDK and ordered-Jacobi Wisdom--Holman maps, plus the one
specific whole-step Wisdom--Holman/RKF78 hybrid and the coupled lunar fixed-
delay route, plus the resolved-eleven coupled lunar EIH component. The force
roster includes the full mutual, correction-only EIH 1PN model and
provenance-bound NumPy/CPU adapters for Earth J2--J5 and the static lunar degree-two
and degree-three figures. The physical-harmonic adapters require metre-second J2000 states and
remain screening-only.
It is a finite-difference service, not general variational propagation.
The remaining 33 rows are `DECLARED` and fail closed through
`DeclaredModelConfig.require_executable()`.

Each declared configuration requires `config_id`, `model_id`, `epoch`,
`unit_system_id`, and the catalog's required parameter bindings. Every binding
contains its exact parameter ID, value, unit dimension, closed validity, and
provenance. Validating a declaration does not make it executable.

### Force and physical-model roadmap

| Status | Capability ID | Required parameter roster |
| --- | --- | --- |
| Implemented | `force.newtonian.point_mass` | `source_ids`, `target_ids`, `unit_system_id`, `state.gravitational_parameters` |
| Implemented | `relativity.solar_schwarzschild_test_particle_1pn` | `central_source_id`, `target_ids`, `unit_system_id`, `speed_of_light`, `maximum_compactness`, `maximum_speed_fraction_squared` |
| Implemented | `force.nongrav.srp_cannonball` | `radiation_source_id`, `target_ids`, `unit_system_id`, `reference_pressure`, `reference_distance`, `area_to_mass`, `radiation_pressure_coefficient`, `coefficient_convention`, `attitude_model`, `shadow_model` |
| Declared | `force.newtonian.softened_point_mass` | `source_ids`, `target_ids`, `gravitational_parameters`, `softening_kernel`, `softening_lengths` |
| Declared | `force.newtonian.barnes_hut_tree` | `body_ids`, `gravitational_parameters`, `opening_angle`, `opening_criterion`, `leaf_capacity`, `multipole_order`, `singularity_policy` |
| Declared | `force.harmonics.solar_j2_j4` | `central_source_id`, `target_ids`, `reference_radius`, `j2`, `j4`, `pole_vector`, `orientation_frame` |
| Implemented | `solar-system.force.earth-zonal-j2-j5-axisymmetric-pair` | `source_id`, `target_id`, `unit_system_id`, `reference_radius`, `zonal_coefficients`, `earth_pole_model` |
| Implemented | `solar-system.force.lunar-static-degree2-principal-axis-pair` | `source_id`, `target_id`, `unit_system_id`, `reference_radius`, `degree2_coefficients`, `lunar_orientation_model` |
| Implemented | `solar-system.force.lunar-static-degree3-principal-axis-pair` | `source_id`, `target_id`, `unit_system_id`, `reference_radius`, `degree3_coefficients`, `lunar_orientation_model` |
| Declared | `force.harmonics.planetary` | `source_ids`, `target_ids`, `reference_radii`, `coefficient_sets`, `orientation_model`, `maximum_degree`, `maximum_order` |
| Implemented | `force.relativity.eih_1pn_gr` | `body_ids`, `unit_system_id`, `speed_of_light`, `maximum_compactness`, `maximum_speed_fraction_squared` |
| Declared | `force.relativity.restricted_ppn_beta_gamma` | `central_source_id`, `target_ids`, `speed_of_light`, `beta`, `gamma`, `maximum_compactness`, `maximum_speed_fraction_squared` |
| Declared | `force.relativity.solar_lense_thirring` | `central_source_id`, `target_ids`, `speed_of_light`, `spin_angular_momentum`, `orientation_frame` |
| Declared | `force.tides.constant_time_lag` | `interacting_pairs`, `love_numbers`, `time_lags`, `radii`, `spin_states`, `dissipation_convention` |
| Declared | `force.nongrav.srp_pr_burns_1979` | `radiation_source_id`, `target_ids`, `speed_of_light`, `reference_pressure`, `reference_distance`, `area_to_mass`, `radiation_pressure_coefficient`, `attitude_model`, `shadow_model` |
| Declared | `force.nongrav.yarkovsky.empirical_a2` | `target_ids`, `a2`, `reference_distance`, `radial_exponent`, `transverse_convention` |
| Declared | `force.nongrav.yarkovsky.linear_sphere` | `target_ids`, `radii`, `densities`, `thermal_conductivities`, `heat_capacities`, `emissivities`, `albedos`, `spin_states`, `pole_vectors`, `radiation_source` |
| Declared | `force.nongrav.yarkovsky.facet_thermophysical` | `target_ids`, `shape_meshes`, `facet_materials`, `spin_states`, `attitude_model`, `thermal_solver`, `shadow_model` |
| Declared | `force.nongrav.comet.marsden_esm` | `target_ids`, `a1_a2_a3`, `g_law_parameters`, `time_shifts`, `orbital_frame_convention` |
| Declared | `force.nongrav.comet.rotating_jet` | `target_ids`, `jet_geometry`, `mass_flow_model`, `exhaust_velocity`, `spin_states`, `pole_vectors`, `attitude_model` |
| Declared | `force.drag.gas` | `target_ids`, `gas_density_model`, `gas_velocity_model`, `drag_coefficients`, `area_to_mass`, `flow_regime_model` |
| Declared | `force.drag.atmospheric` | `central_body_id`, `target_ids`, `atmosphere_model`, `atmosphere_rotation`, `drag_coefficients`, `area_to_mass`, `space_weather_inputs` |
| Declared | `force.thrust.prescribed` | `target_ids`, `burn_windows`, `force_or_acceleration_profile`, `direction_law`, `mass_flow_profile`, `attitude_model` |
| Declared | `ephemeris.external.interpolated` | `source_ids`, `kernel_or_table`, `frame`, `origin`, `time_scale`, `validity_interval`, `interpolation_method`, `interpolation_tolerance` |
| Declared | `force.collision.hard_sphere` | `body_ids`, `radii`, `collision_response`, `restitution_coefficients`, optional `fragmentation_model`, `event_tolerance` |
| Declared | `force.encounter.hybrid_switching` | `body_ids`, `switch_distance`, `switch_hysteresis`, `far_integrator`, `near_integrator`, `event_tolerance` |
| Declared | `force.regularization.algorithmic` | `body_ids`, `regularization_method`, `activation_distance`, `termination_distance`, `error_tolerance` |

### Backend, precision, determinism, and integration roadmap

| Status | Capability ID | Required parameter roster |
| --- | --- | --- |
| Implemented | `backend.numpy.cpu` | `device`, `dtype`, `tile_size` |
| Implemented | `backend.cupy.cuda` | `device`, `dtype`, `tile_size` |
| Declared | `backend.hip` | `device`, `runtime`, `dtype`, `tile_size` |
| Declared | `backend.sycl` | `device`, `runtime`, `dtype`, `tile_size` |
| Declared | `backend.metal` | `device`, `runtime`, `dtype`, `tile_size` |
| Implemented | `precision.float64` | `dtype` |
| Declared | `precision.float32` | `dtype`, `error_budget` |
| Declared | `precision.mixed` | `storage_dtype`, `compute_dtype`, `accumulation_dtype`, `error_budget` |
| Declared | `precision.decimal_reference` | `decimal_precision`, `rounding`, `trap_policy` |
| Implemented | `determinism.same_runtime_device` | `runtime_fingerprint`, `device_fingerprint`, `tile_size`, `fast_math` |
| Declared | `determinism.cross_device_bitwise` | `backend_matrix`, `reduction_policy`, `compiler_policy` |
| Implemented | `integrator.adaptive.rkf78.fehlberg_1968` | `checkpoint_epochs`, `initial_step`, `minimum_step`, `maximum_step`, `position_atol`, `position_rtol`, `velocity_atol`, `velocity_rtol`, `maximum_steps`, `maximum_rejections`, `safety_factor`, `minimum_scale_factor`, `maximum_scale_factor` |
| Implemented | `integrator.rkf78.fixed_delay.deformable_lunar_mantle_core.v3` | exact coupled `initial_state`, deformable `parameters`, fixed-delay `integration_spec`, exact pre-start `prehistory_provider` |
| Implemented | `jx.integrator.rkf78.fixed.coupled-lunar.v1` | resolved-eleven coupled `initial_state`, bound physical `parameters`, fixed-step `integration_spec` |
| Implemented | `integrator.adaptive.rkf78.encounter_segment_newtonian_v1` | `body_order`, `initial_epoch`, independent `endpoint_epoch`, signed `duration`, initial/minimum/maximum step magnitudes, canonical-pair clearance floors and pair position/velocity tolerances, GM-centroid position/velocity tolerances, proposal/acceptance/rejection/consecutive-rejection/force caps, controller scale factors, exact-rational resource record |
| Implemented | `integrator.symplectic.kdk_leapfrog_2` | `checkpoint_step_indices`, signed `fixed_step`, `maximum_steps`, `minimum_swept_pair_separation`, `maximum_pair_frequency_step` |
| Implemented | `integrator.symplectic.wisdom_holman_jacobi_kdk_2` | `checkpoint_step_indices`, signed `fixed_step`, `maximum_steps`, `jacobi_body_order`, `minimum_encounter_pair_separation`, `minimum_jacobi_periapse`, initial barycenter position/velocity caps, optional fixed `kepler_solver` record |
| Implemented | `integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1` | exact `wisdom_holman_spec`, exact interval-invariant `encounter_control` profile |
| Declared | `integrator.symplectic.split` | `method`, `fixed_step`, `splitting`, optional `corrector`, `output_schedule` |
| Declared | `integrator.implicit.velocity_dependent` | `method`, `step`, `nonlinear_tolerance`, `maximum_iterations`, `failure_policy` |
| Declared | `integrator.hybrid.close_encounter` | `far_integrator`, `near_integrator`, `switch_policy`, `regularization`, `event_tolerance` |

### Analysis and observation roadmap

| Status | Capability ID | Required parameter roster |
| --- | --- | --- |
| Declared | `analysis.variational_equations` | `parameter_ids`, `initial_state_transition`, `jacobian_method`, optional `differentiation_step` |
| Declared | `orbit_determination.batch_least_squares` | `estimated_parameters`, `prior_state`, `prior_covariance`, `measurement_set`, `weight_model`, `convergence_policy` |
| Declared | `measurement.astrometry` | `observer_ephemeris`, `time_scale`, `frame`, `light_time_model`, `aberration_model`, `bias_model`, `covariance_model` |

## Qualification and performance roadmap

The following work is necessary before describing this engine as top-tier:

- independent equation and implementation review for every force model;
- published analytic, Decimal/reference, conservation, convergence, encounter,
  and long-arc ephemeris validation suites;
- independently qualified trajectory methods, plus dense output, event
  location, collision response, close-encounter switching, regularization,
  and symplectic/hybrid choices;
- compiled CPU, HIP, SYCL, Metal, multi-GPU, and distributed implementations;
- explicit runtime/device/compiler fingerprints and reproducibility matrices;
- observation modeling, light time, aberration, time-scale transformations,
  station geometry, variational equations, and orbit fitting;
- preregistered public comparisons using identical initial states, force
  models, tolerances, output epochs, and hardware; and
- external researchers reproducing both correctness and performance results.

Performance acceptance criteria must be preregistered per workload. At a
minimum they must report body count, source/target mix, enabled force graph,
backend/device/compiler/runtime, tile size, warm-up and repetitions, wall time,
interactions or RHS evaluations per second, memory footprint, transfer volume,
trajectory error at common epochs, conservation drift where applicable, and
failure/rejection counts. This document makes no numerical performance claim.

The Decimal engine and frozen V4/V5 qualification artifacts remain separate
reference/evidence systems. This additive alpha does not modify their bytes or
inherit their authority.
