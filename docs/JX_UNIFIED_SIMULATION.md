# JX unified simulation runtime

## Status

This is post-`0.6.0rc15` development. It unifies existing components without
changing their numerical algorithms or scientific claim ceilings. Every run
remains `SCREENING_ONLY`; the new coordinator is not a production ephemeris or
a general claim of superiority over another engine.

## Implemented vertical slices

`JXSimulation` owns one explicit `StateSnapshot` and `ForcePlan`, serializes
execution, and records only successful runs. Callers select the integrator by
its exact identifier. No force, body, backend, collision policy, or numerical
control is selected implicitly.

The first four routes are:

| Route | Force-plan scope | Backend |
|---|---|---|
| `integrator.adaptive.rkf78.fehlberg_1968` | Every compatible force composition currently implemented by the generic engine | Explicit NumPy CPU or CuPy CUDA |
| `JX_GAUSS_RADAU15_V3` | Exactly one fully mutual Newtonian point-mass term | NumPy CPU |
| `JX_GR15_EIH1PN_V1` | Fully mutual Newtonian point masses followed by one full mutual EIH 1PN correction | NumPy CPU |
| `JX_GR15_EIH1PN_CUDA_V1` | The identical Newtonian plus mutual EIH 1PN plan, wrapped as one device-resident system | CuPy CUDA |

The GR15 adapter requires every body to be a source and target in state order,
positive GM and mass, all bodies marked massive, zero physical radii, matching
unit identities and metadata, and complete parameter-validity coverage. These
requirements preserve the native solver's exact semantics. A request with an
extra force, passive target, nonzero collision radius, CUDA backend, or expired
parameter set fails rather than silently changing the model.

`MutualEIH1PN` is now a typed force configuration rather than a detached solar-
system helper. Its correction-only kernel uses the same `ForcePlan` ledger as
the other engine forces and supports both NumPy and CuPy through the general
RKF78 route. The force contract requires every state body, positive GM, a TDB
`BARYCENTRIC_INERTIAL` snapshot with `SYSTEM_BARYCENTER` origin, explicit
weak-field and slow-motion ceilings, and beta=gamma=1. It is mutually exclusive
with the restricted static-central test-particle 1PN term.

The specialized native GR15-EIH CPU and CUDA adapters consume that identical
force plan. Both additionally require kilometres and seconds because their
native public boundaries are unit-specific and 2--32 bodies because that is
their compiled domain. Backend selection is exact: the CPU route requires
NumPy, the CUDA route requires CuPy device arrays, and neither falls back. The
CUDA adapter adds a one-system batch dimension without transferring state to
the host, calls the existing persistent batched kernel, and retains its
device-resident trajectory in the common result envelope.

The post-rc15 physical-force slice also exposes three exact, provenance-bound
`ForcePlan` types: `EarthZonalJ2J5Force`, `LunarStaticDegree2Force`, and
`LunarStaticDegree3Force`. They adapt the already tested Solar-System kernels
without reimplementing their equations. The generic RKF78 route evaluates
them at every stage epoch and records each correction in the normal force
ledger. Their current contract is deliberately narrow: NumPy/CPU, metre and
second units, J2000 axes, and one configured source-target pair. The lunar
terms use a caller-bound orientation provider identity; the Earth term binds
its prepared pole policy. All constants and provider identities require
ordered `ParameterMetadata` with closed validity and non-placeholder
provenance.

`FORCE_ABI_REGISTRY` is now the single exact-type, canonical-order, backend,
semantics, and integrator-class roster used by the unified evaluator. ABI v1
contains seven executable force configurations. It is an engine-native Python
dispatch contract, not yet a general C callback ABI: native GR15 and CUDA
kernels still accept only their explicitly specialized force combinations and
reject these three additions without recording a run.

State ABI v1 now adds a separate typed coupled-lunar route. The five-block
`CoupledLunarStateSnapshot` owns Sun/Earth/Moon translation, lunar mantle
attitude, mantle angular velocity, and fluid-core angular velocity with exact
TDB/barycentric/J2000 context and provenance. The public
`integrate_deformable_coupled_lunar_state()` adapter calls the retained v3
fixed-delay RKF78 solver and returns all five blocks at every checkpoint plus
the native structural diagnostics and a content SHA-256. Its simultaneous
physics include delayed tide/spin deformation, changing mantle inertia,
Earth/Sun figure reactions and torques, fixed-pole Earth J2, and mantle-core
boundary coupling.

This state route is intentionally marked
`NATIVE_COUPLED_PHYSICS_BUNDLE_NOT_FORCE_ABI_V1`. It is public executable
engine code, but its coupled physics have not yet been decomposed into the
shared force ABI, and it has no CUDA, coupled restart-archive, geodetic-
transport, event, collision, or ephemeris-qualification claim.

## Development verification observation

On 2026-09-24, the permanent unified CUDA regression test passed on both an
RTX 5060 Ti (16,311 MiB, driver 595.91.07) and an RTX 4050 Laptop GPU (6,141
MiB, driver 595.71.05). For the retained two-body EIH fixture, the CPU and CUDA
routes used the same ordered force-model IDs, accepted 7 steps with 0
rejections, and produced exact-zero final position and velocity differences in
binary64. This is a development observation, not release evidence, a broad
performance claim, or ephemeris qualification.

```python
from jxplanetx import GR15Spec, JXSimulation, GR15_NEWTONIAN_METHOD_ID
from jxplanetx.engine import StateSnapshot, ForcePlan

simulation = JXSimulation("solar-system-screen", snapshot, force_plan)
run = simulation.integrate(
    GR15_NEWTONIAN_METHOD_ID,
    GR15Spec(
        initial_epoch=0.0,
        final_epoch=86400.0,
        initial_step=60.0,
        maximum_step=3600.0,
    ),
)
print(run.final_positions)
print(run.native_result)
```

Every `JXSimulationRun` provides one common envelope containing the simulation
identity, integrator identity, ordered force-model identities, exact checkpoint
schedule, checkpoint states, controller accounting, backend/device identity,
native result and native result digest. The complete solver-specific result is
retained rather than flattened or weakened.

`integrate()` always starts from the retained initial snapshot; the runtime
never infers continuation from call order. `prepare_continuation(run_index)`
instead copies an exact NumPy/CPU endpoint and binds its bytes, state constants,
force roster, parent integrator, parent result digest, coordinate context, and
units into a domain-separated SHA-256 identity. `continue_from(...)` requires
that exact parent and a new spec whose initial epoch equals the endpoint. A
failed preparation or integration appends no run. CUDA continuation is refused
until device results have a retained content digest.

`dump_simulation_archive()` writes that explicit continuation as a deterministic
ZIP byte stream containing canonical ASCII JSON and six non-pickle NPY arrays.
`load_simulation_archive()` requires the expected archive SHA-256 plus a live
caller-supplied `ForcePlan` whose canonical digest exactly matches the archive.
It validates member roster/order/type/size, CRC, every member digest, array
dtype/shape/order, the continuation state digest, provenance, force roster, and
claim ceiling before returning a new `JXSimulation` rooted at the endpoint.
Executable orientation providers are deliberately not deserialized; their
public IDs, parameters, and provenance are bound through the supplied plan.

## Remaining unification sequence

1. Extend force ABI v1 into a compiled force-evaluation ABI usable by GR15,
   RKF78 and CUDA without
   Python callbacks inside a stage loop.
2. Move the retained coupled lunar physics bundle onto that shared ABI, then
   add coupled-state restart archives and a qualified CUDA route.
3. Add collision/event interfaces separately; collision detection must not be
   confused with collision resolution.
