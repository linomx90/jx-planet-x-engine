from __future__ import annotations

import io
import json
import unittest
from unittest import mock
import zipfile

import numpy as np

import jxplanetx
import jxplanetx.engine as engine
import jxplanetx.engine.api as engine_api
from jxplanetx.engine.simulation import _continuation_state_identity_sha256
from jxplanetx import GR15Spec
from jxplanetx.engine import (
    ADAPTIVE_RKF78_METHOD_ID,
    AdaptiveRKF78Spec,
    BackendSpec,
    ContractError,
    FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE,
    FORCE_ABI_REGISTRY,
    FORCE_ABI_VERSION,
    ForcePlan,
    GR15_EIH1PN_CUDA_METHOD_ID,
    GR15_EIH1PN_METHOD_ID,
    GR15_NEWTONIAN_METHOD_ID,
    INTEGRATOR_ROUTE_REGISTRY,
    INTEGRATOR_ROUTE_REGISTRY_VERSION,
    INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE,
    IntegratorRouteSpec,
    JXSimulation,
    JXSimulationArchiveLoad,
    JXSimulationCompatibilityError,
    JXSimulationContinuation,
    JXSimulationContractError,
    JXSimulationRun,
    MutualEIH1PN,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    SimulationArchiveError,
    StateSnapshot,
    dump_simulation_archive,
    get_force_abi,
    get_integrator_route,
    list_force_abis,
    list_integrator_routes,
    load_simulation_archive,
    simulation_archive_sha256,
    simulation_force_plan_sha256,
    bind_lunar_static_degree2_force,
    bind_lunar_static_degree3_force,
)
from jxplanetx.solar_system.lunar_figure import (
    prepare_lunar_static_degree2_pair_force,
)
from jxplanetx.solar_system.lunar_figure_degree3 import (
    prepare_lunar_static_degree3_pair_force,
)


UNIT_SYSTEM_ID = "fixture.unified-runtime.canonical"
EIH_UNIT_SYSTEM_ID = "fixture.unified-runtime.km-s"
BODY_IDS = ("PRIMARY", "SECONDARY")


def provenance() -> Provenance:
    return Provenance(
        source_id="fixture.unified-runtime",
        citation="Synthetic two-body unified-runtime fixture",
        version="1",
        sha256="a" * 64,
    )


def parameter_metadata(
    *,
    validity_start: float = -1.0,
    validity_end: float = 1.0,
) -> tuple[ParameterMetadata, ...]:
    return (
        ParameterMetadata(
            parameter_id="state.gravitational_parameters",
            units="CANONICAL_LENGTH^3/CANONICAL_TIME^2",
            provenance=provenance(),
            uncertainty=None,
            covariance_group=None,
            validity_start=validity_start,
            validity_end=validity_end,
        ),
    )


def state_snapshot() -> StateSnapshot:
    return StateSnapshot(
        snapshot_id="fixture.unified-runtime.initial",
        epoch=0.0,
        time_scale="TDB",
        frame="BARYCENTRIC_INERTIAL",
        origin="SYSTEM_BARYCENTER",
        axes="ICRS_ALIGNED",
        length_unit="CANONICAL_LENGTH",
        time_unit="CANONICAL_TIME",
        mass_unit="CANONICAL_MASS",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=BODY_IDS,
        positions=np.array(
            ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
            dtype=np.float64,
        ),
        velocities=np.array(
            ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)),
            dtype=np.float64,
        ),
        gravitational_parameters=np.array((0.5, 0.5), dtype=np.float64),
        masses=np.array((0.5, 0.5), dtype=np.float64),
        radii=np.zeros(2, dtype=np.float64),
        massive=np.ones(2, dtype=np.bool_),
        provenance=provenance(),
    )


def single_body_state_snapshot() -> StateSnapshot:
    return StateSnapshot(
        snapshot_id="fixture.unified-runtime.single-body",
        epoch=0.0,
        time_scale="TDB",
        frame="BARYCENTRIC_INERTIAL",
        origin="SYSTEM_BARYCENTER",
        axes="ICRS_ALIGNED",
        length_unit="CANONICAL_LENGTH",
        time_unit="CANONICAL_TIME",
        mass_unit="CANONICAL_MASS",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=("PRIMARY",),
        positions=np.zeros((1, 3), dtype=np.float64),
        velocities=np.zeros((1, 3), dtype=np.float64),
        gravitational_parameters=np.ones(1, dtype=np.float64),
        masses=np.ones(1, dtype=np.float64),
        radii=np.zeros(1, dtype=np.float64),
        massive=np.ones(1, dtype=np.bool_),
        provenance=provenance(),
    )


def eih_force_plan(*, speed_of_light: float = 100.0) -> ForcePlan:
    def metadata(parameter_id: str, units: str) -> ParameterMetadata:
        return ParameterMetadata(
            parameter_id=parameter_id,
            units=units,
            provenance=provenance(),
            uncertainty=None,
            covariance_group=None,
            validity_start=-1.0,
            validity_end=1.0,
        )

    return ForcePlan(
        plan_id="fixture.unified-runtime.eih-plan",
        backend=BackendSpec("numpy", "cpu", 2),
        models=(
            NewtonianPointMass(
                source_ids=BODY_IDS,
                target_ids=BODY_IDS,
                unit_system_id=EIH_UNIT_SYSTEM_ID,
                parameter_metadata=(
                    metadata("state.gravitational_parameters", "KM^3/S^2"),
                ),
            ),
            MutualEIH1PN(
                body_ids=BODY_IDS,
                speed_of_light=speed_of_light,
                maximum_compactness=0.5,
                maximum_speed_fraction_squared=0.5,
                unit_system_id=EIH_UNIT_SYSTEM_ID,
                parameter_metadata=(
                    metadata("speed_of_light", "KM/S"),
                    metadata("maximum_compactness", "1"),
                    metadata("maximum_speed_fraction_squared", "1"),
                ),
            ),
        ),
    )


def lunar_state_snapshot(
    *,
    aux_id: str = "AUX",
    massless_targets: bool = True,
) -> StateSnapshot:
    target_gm = 0.0 if massless_targets else 1.0
    return StateSnapshot(
        snapshot_id="fixture.unified-runtime.lunar",
        epoch=0.0,
        time_scale="TDB",
        frame="BARYCENTRIC_INERTIAL",
        origin="SYSTEM_BARYCENTER",
        axes="J2000",
        length_unit="M",
        time_unit="S",
        mass_unit="KG",
        unit_system_id="fixture.unified-runtime.m-s",
        body_ids=("MOON", "TARGET", aux_id),
        positions=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
            dtype=np.float64,
        ),
        velocities=np.zeros((3, 3), dtype=np.float64),
        gravitational_parameters=np.array(
            (1.0, target_gm, target_gm),
            dtype=np.float64,
        ),
        masses=np.array((1.0, target_gm, target_gm), dtype=np.float64),
        radii=np.zeros(3, dtype=np.float64),
        massive=np.array(
            (True, not massless_targets, not massless_targets),
            dtype=np.bool_,
        ),
        provenance=provenance(),
    )


def _physical_metadata(parameter_ids: tuple[str, ...]) -> tuple[ParameterMetadata, ...]:
    units = {
        "state.gravitational_parameters": "M^3/S^2",
        "reference_radius": "M",
        "degree2_coefficients": "1",
        "degree3_coefficients": "1",
        "lunar_orientation_model": "1",
    }
    return tuple(
        ParameterMetadata(
            parameter_id=parameter_id,
            units=units[parameter_id],
            provenance=provenance(),
            uncertainty=None,
            covariance_group=None,
            validity_start=-1.0,
            validity_end=1.0,
        )
        for parameter_id in parameter_ids
    )


def _identity_lunar_orientation(_absolute_et: float) -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def lunar_force_plan(degree: int | None, *, aux_id: str = "AUX") -> ForcePlan:
    base = NewtonianPointMass(
        source_ids=("MOON",),
        target_ids=("MOON", "TARGET", aux_id),
        unit_system_id="fixture.unified-runtime.m-s",
        parameter_metadata=_physical_metadata(("state.gravitational_parameters",)),
    )
    models: tuple[object, ...]
    if degree is None:
        models = (base,)
    elif degree == 2:
        pair = prepare_lunar_static_degree2_pair_force(
            source_id="MOON",
            target_id="TARGET",
            source_index=0,
            target_index=1,
            absolute_start_et=0.0,
            orientation_provider=_identity_lunar_orientation,
        )
        models = (
            base,
            bind_lunar_static_degree2_force(
                pair,
                unit_system_id="fixture.unified-runtime.m-s",
                parameter_metadata=_physical_metadata(
                    (
                        "reference_radius",
                        "degree2_coefficients",
                        "lunar_orientation_model",
                    )
                ),
                orientation_provider_id="fixture.identity-lunar-orientation",
            ),
        )
    elif degree == 3:
        pair = prepare_lunar_static_degree3_pair_force(
            source_id="MOON",
            target_id="TARGET",
            source_index=0,
            target_index=1,
            absolute_start_et=0.0,
            orientation_provider=_identity_lunar_orientation,
        )
        models = (
            base,
            bind_lunar_static_degree3_force(
                pair,
                unit_system_id="fixture.unified-runtime.m-s",
                parameter_metadata=_physical_metadata(
                    (
                        "reference_radius",
                        "degree3_coefficients",
                        "lunar_orientation_model",
                    )
                ),
                orientation_provider_id="fixture.identity-lunar-orientation",
            ),
        )
    else:
        raise ValueError("degree must be None, 2, or 3")
    return ForcePlan(
        plan_id=f"fixture.unified-runtime.lunar-{degree}",
        backend=BackendSpec("numpy", "cpu", 3),
        models=models,  # type: ignore[arg-type]
    )


def newtonian_force(
    *,
    source_ids: tuple[str, ...] = BODY_IDS,
    target_ids: tuple[str, ...] = BODY_IDS,
    validity_start: float = -1.0,
    validity_end: float = 1.0,
) -> NewtonianPointMass:
    return NewtonianPointMass(
        source_ids=source_ids,
        target_ids=target_ids,
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=parameter_metadata(
            validity_start=validity_start,
            validity_end=validity_end,
        ),
    )


def force_plan(
    *,
    tile_size: int = 2,
    source_ids: tuple[str, ...] = BODY_IDS,
    target_ids: tuple[str, ...] = BODY_IDS,
    validity_start: float = -1.0,
    validity_end: float = 1.0,
) -> ForcePlan:
    return ForcePlan(
        plan_id="fixture.unified-runtime.plan",
        backend=BackendSpec("numpy", "cpu", tile_size),
        models=(
            newtonian_force(
                source_ids=source_ids,
                target_ids=target_ids,
                validity_start=validity_start,
                validity_end=validity_end,
            ),
        ),
    )


def restricted_missing_central_plan() -> ForcePlan:
    return ForcePlan(
        plan_id="fixture.unified-runtime.missing-central",
        backend=BackendSpec("numpy", "cpu", 2),
        models=(
            newtonian_force(
                source_ids=("MISSING",),
                target_ids=("SECONDARY",),
            ),
            RestrictedStaticCentral1PN(
                central_source_id="MISSING",
                target_ids=("SECONDARY",),
                speed_of_light=100.0,
                maximum_compactness=0.5,
                maximum_speed_fraction_squared=0.5,
                unit_system_id=UNIT_SYSTEM_ID,
                parameter_metadata=tuple(
                    ParameterMetadata(
                        parameter_id=parameter_id,
                        units=units,
                        provenance=provenance(),
                        uncertainty=None,
                        covariance_group=None,
                        validity_start=-1.0,
                        validity_end=1.0,
                    )
                    for parameter_id, units in (
                        (
                            "speed_of_light",
                            "CANONICAL_LENGTH/CANONICAL_TIME",
                        ),
                        ("maximum_compactness", "1"),
                        ("maximum_speed_fraction_squared", "1"),
                    )
                ),
            ),
        ),
    )


def rkf78_spec(
    checkpoint_epochs: tuple[float, ...],
    *,
    body_count: int = 2,
) -> AdaptiveRKF78Spec:
    return AdaptiveRKF78Spec(
        checkpoint_epochs=checkpoint_epochs,
        initial_step=0.005,
        minimum_step=1.0e-12,
        maximum_step=0.01,
        position_atol=np.full((body_count, 3), 1.0e-13, dtype=np.float64),
        position_rtol=1.0e-12,
        velocity_atol=np.full((body_count, 3), 1.0e-13, dtype=np.float64),
        velocity_rtol=1.0e-12,
        maximum_steps=1_000,
        maximum_rejections=100,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=4.0,
    )


def gr15_spec() -> GR15Spec:
    return GR15Spec(
        initial_epoch=0.0,
        final_epoch=0.02,
        intermediate_epochs=(0.01,),
        initial_step=0.005,
        minimum_step=1.0e-14,
        maximum_step=0.01,
        epsilon=1.0e-11,
        maximum_steps=1_000,
        maximum_rejections=100,
    )


def relabel_archive_parent(
    archive_bytes: bytes,
    continuation: JXSimulationContinuation,
    parent_integrator_id: str,
) -> bytes:
    """Produce a canonically rehashed archive with an impossible parent label."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        infos = archive.infolist()
        members = {info.filename: archive.read(info.filename) for info in infos}
    manifest = json.loads(members["manifest.json"].decode("ascii"))
    state_sha256 = _continuation_state_identity_sha256(
        simulation_id=continuation.simulation_id,
        parent_run_index=continuation.parent_run_index,
        parent_integrator_id=parent_integrator_id,
        parent_native_result_digest=continuation.parent_native_result_digest,
        force_model_ids=continuation.force_model_ids,
        device="cpu",
        snapshot=continuation.snapshot,
    )
    manifest["continuation"]["parent_integrator_id"] = parent_integrator_id
    manifest["continuation"]["state_sha256"] = state_sha256
    manifest["snapshot"]["provenance"]["sha256"] = state_sha256
    members["manifest.json"] = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


def rebind_archive_force_plan(
    archive_bytes: bytes,
    plan: ForcePlan,
) -> bytes:
    """Canonically rehash an archive against a different same-roster plan."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        infos = archive.infolist()
        members = {info.filename: archive.read(info.filename) for info in infos}
    manifest = json.loads(members["manifest.json"].decode("ascii"))
    manifest["force_plan"]["sha256"] = simulation_force_plan_sha256(plan)
    members["manifest.json"] = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


def rebind_archive_force_plan_and_roster(
    archive_bytes: bytes,
    continuation: JXSimulationContinuation,
    plan: ForcePlan,
) -> bytes:
    """Canonically rewrite plan and lineage rosters for a hostile archive."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        infos = archive.infolist()
        members = {info.filename: archive.read(info.filename) for info in infos}
    manifest = json.loads(members["manifest.json"].decode("ascii"))
    force_model_ids = tuple(model.model_id for model in plan.models)
    state_sha256 = _continuation_state_identity_sha256(
        simulation_id=continuation.simulation_id,
        parent_run_index=continuation.parent_run_index,
        parent_integrator_id=continuation.parent_integrator_id,
        parent_native_result_digest=continuation.parent_native_result_digest,
        force_model_ids=force_model_ids,
        device="cpu",
        snapshot=continuation.snapshot,
    )
    manifest["force_plan"]["sha256"] = simulation_force_plan_sha256(plan)
    manifest["continuation"]["force_model_ids"] = list(force_model_ids)
    manifest["continuation"]["state_sha256"] = state_sha256
    manifest["snapshot"]["provenance"]["sha256"] = state_sha256
    members["manifest.json"] = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


class UnifiedRuntimeTests(unittest.TestCase):
    def test_public_route_and_force_abi_rosters_are_exact(self) -> None:
        expected_route_ids = (
            "integrator.adaptive.rkf78.fehlberg_1968",
            "JX_GAUSS_RADAU15_V3",
            "JX_GR15_EIH1PN_V1",
            "JX_GR15_EIH1PN_CUDA_V1",
        )
        self.assertEqual(
            tuple(INTEGRATOR_ROUTE_REGISTRY),
            expected_route_ids,
        )
        self.assertEqual(INTEGRATOR_ROUTE_REGISTRY_VERSION, 1)
        self.assertEqual(
            INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE,
            "SCREENING_ONLY",
        )
        routes = list_integrator_routes()
        self.assertEqual(
            tuple(route.integrator_id for route in routes),
            expected_route_ids,
        )
        self.assertTrue(all(type(route) is IntegratorRouteSpec for route in routes))
        self.assertTrue(all(not route.production_authorized for route in routes))
        self.assertTrue(all(not route.qualification_authorized for route in routes))
        self.assertTrue(all(not route.general_superiority_claimed for route in routes))
        self.assertEqual(routes[0].spec_type, AdaptiveRKF78Spec)
        self.assertEqual(
            routes[0].force_plan_scope,
            FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE,
        )
        self.assertEqual(
            routes[0].required_force_abi_integrator_class,
            "GENERAL_FIRST_ORDER",
        )
        self.assertTrue(
            routes[0].supports_portable_manifest(
                "numpy",
                "cpu",
                (NewtonianPointMass.MODEL_ID,),
            )
        )
        self.assertFalse(
            routes[0].supports_portable_manifest(
                "numpy",
                "cpu",
                (
                    NewtonianPointMass.MODEL_ID,
                    "force.relativity.eih_1pn_gr",
                ),
            )
        )
        self.assertFalse(routes[0].supports_backend("numpy", "cpu:bogus"))
        self.assertFalse(routes[-1].supports_backend("cupy", "cuda-not-a-device"))
        self.assertTrue(routes[1].supports_archive("numpy", "cpu"))
        self.assertEqual(routes[-1].continuation_backend_ids, ())
        self.assertIs(get_integrator_route(ADAPTIVE_RKF78_METHOD_ID), routes[0])
        with self.assertRaises(KeyError):
            get_integrator_route(True)  # type: ignore[arg-type]
        expected_force_ids = (
            "force.newtonian.point_mass",
            "relativity.solar_schwarzschild_test_particle_1pn",
            "force.relativity.eih_1pn_gr",
            "solar-system.force.earth-zonal-j2-j5-axisymmetric-pair",
            "solar-system.force.lunar-static-degree2-principal-axis-pair",
            "solar-system.force.lunar-static-degree3-principal-axis-pair",
            "force.nongrav.srp_cannonball",
        )
        self.assertEqual(FORCE_ABI_VERSION, 1)
        self.assertEqual(tuple(FORCE_ABI_REGISTRY), expected_force_ids)
        self.assertEqual(
            tuple(spec.model_id for spec in list_force_abis()),
            expected_force_ids,
        )
        self.assertIs(get_force_abi(NewtonianPointMass), list_force_abis()[0])

        public_names = (
            "FORCE_ABI_REGISTRY",
            "INTEGRATOR_ROUTE_REGISTRY",
            "INTEGRATOR_ROUTE_REGISTRY_VERSION",
            "INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE",
            "IntegratorRouteSpec",
            "GR15_EIH1PN_CUDA_METHOD_ID",
            "GR15_EIH1PN_METHOD_ID",
            "GR15_NEWTONIAN_METHOD_ID",
            "JXSimulation",
            "JXSimulationArchiveLoad",
            "JXSimulationCompatibilityError",
            "JXSimulationContractError",
            "JXSimulationRun",
            "dump_simulation_archive",
            "get_integrator_route",
            "list_integrator_routes",
            "load_simulation_archive",
            "simulation_archive_sha256",
        )
        for name in public_names:
            with self.subTest(name=name):
                value = getattr(engine, name)
                self.assertIs(getattr(engine_api, name), value)
                self.assertIs(getattr(jxplanetx, name), value)
                self.assertIn(name, engine.__all__)
                self.assertIn(name, engine_api.__all__)
                self.assertIn(name, jxplanetx.__all__)

        plan = force_plan()
        self.assertEqual(plan.evidence_class, "MODEL_OUTPUT")
        self.assertFalse(plan.registry_authorized)
        self.assertFalse(plan.qualification_authorized)
        with self.assertRaisesRegex(ContractError, "squared must be finite"):
            eih_force_plan(speed_of_light=1.0e308)

    def test_same_state_and_plan_use_rkf78_and_native_gr15_envelopes(self) -> None:
        initial = state_snapshot()
        simulation = JXSimulation(
            "fixture.unified-runtime.routes",
            initial,
            force_plan(),
        )
        self.assertIsNot(simulation.snapshot.positions, initial.positions)
        self.assertFalse(simulation.snapshot.positions.flags.writeable)

        rkf78 = simulation.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.01, 0.02)),
        )
        gr15 = simulation.integrate(GR15_NEWTONIAN_METHOD_ID, gr15_spec())

        self.assertEqual(simulation.run_count, 2)
        self.assertEqual(simulation.runs, (rkf78, gr15))
        self.assertIs(simulation.last_run, gr15)
        for index, run, method_id, digest_scope in (
            (
                0,
                rkf78,
                ADAPTIVE_RKF78_METHOD_ID,
                "FULL_RETAINED_RESULT_CONTENT",
            ),
            (1, gr15, GR15_NEWTONIAN_METHOD_ID, "GR15_NATIVE_REPLAY_CONTENT"),
        ):
            with self.subTest(method_id=method_id):
                self.assertIs(type(run), JXSimulationRun)
                self.assertEqual(run.simulation_id, simulation.simulation_id)
                self.assertEqual(run.run_index, index)
                self.assertEqual(run.integrator_id, method_id)
                self.assertEqual(
                    run.force_model_ids,
                    ("force.newtonian.point_mass",),
                )
                self.assertEqual(run.checkpoint_epochs, (0.0, 0.01, 0.02))
                self.assertEqual(run.backend_id, "numpy")
                self.assertEqual(run.device, "cpu")
                self.assertEqual(run.native_result_digest_scope, digest_scope)
                self.assertRegex(run.native_result_digest or "", r"^[0-9a-f]{64}$")
                self.assertEqual(run.scientific_claim_state, "SCREENING_ONLY")
                self.assertFalse(run.production_authorized)
                self.assertFalse(run.general_superiority_claimed)
                self.assertFalse(run.is_continuation)
                np.testing.assert_array_equal(
                    run.checkpoint_positions[0],
                    simulation.snapshot.positions,
                )
                np.testing.assert_array_equal(
                    run.checkpoint_velocities[0],
                    simulation.snapshot.velocities,
                )

    def test_incompatible_route_and_wrong_spec_leave_no_partial_run(self) -> None:
        simulation = JXSimulation(
            "fixture.unified-runtime.atomic-failure",
            state_snapshot(),
            force_plan(),
        )
        with self.assertRaisesRegex(
            JXSimulationContractError,
            "AdaptiveRKF78Spec",
        ):
            simulation.integrate(ADAPTIVE_RKF78_METHOD_ID, gr15_spec())
        self.assertEqual(simulation.runs, ())
        self.assertIsNone(simulation.last_run)

        with self.assertRaisesRegex(
            JXSimulationCompatibilityError,
            "NewtonianPointMass followed by MutualEIH1PN",
        ):
            simulation.integrate(GR15_EIH1PN_METHOD_ID, gr15_spec())
        self.assertEqual(simulation.run_count, 0)

        recovered = simulation.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.01, 0.02)),
        )
        self.assertEqual(recovered.run_index, 0)
        self.assertEqual(simulation.runs, (recovered,))

    def test_lunar_archive_rejects_zero_gm_before_kernel_entry(self) -> None:
        base_plan = lunar_force_plan(None)
        source = JXSimulation(
            "fixture.unified-runtime.lunar-archive",
            lunar_state_snapshot(),
            base_plan,
        )
        source.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.001, 0.002), body_count=3),
        )
        continuation = source.prepare_continuation(0)
        archive = dump_simulation_archive(source, 0)

        for degree in (2, 3):
            with self.subTest(degree=degree):
                plan = lunar_force_plan(degree)
                forged = rebind_archive_force_plan_and_roster(
                    archive,
                    continuation,
                    plan,
                )
                with self.assertRaisesRegex(
                    SimulationArchiveError,
                    "incompatible with the force plan",
                ):
                    load_simulation_archive(
                        forged,
                        expected_sha256=simulation_archive_sha256(forged),
                        force_plan=plan,
                    )

        invalid_id = "A UX"
        identifier_base_plan = lunar_force_plan(None, aux_id=invalid_id)
        identifier_source = JXSimulation(
            "fixture.unified-runtime.lunar-identifier-archive",
            lunar_state_snapshot(aux_id=invalid_id, massless_targets=False),
            identifier_base_plan,
        )
        identifier_source.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.001, 0.002), body_count=3),
        )
        identifier_continuation = identifier_source.prepare_continuation(0)
        identifier_archive = dump_simulation_archive(identifier_source, 0)
        identifier_plan = lunar_force_plan(2, aux_id=invalid_id)
        forged_identifier = rebind_archive_force_plan_and_roster(
            identifier_archive,
            identifier_continuation,
            identifier_plan,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_identifier,
                expected_sha256=simulation_archive_sha256(forged_identifier),
                force_plan=identifier_plan,
            )

    def test_cuda_native_call_stays_inside_selected_device_context(self) -> None:
        simulation = JXSimulation(
            "fixture.unified-runtime.cuda-context",
            state_snapshot(),
            force_plan(),
        )
        events: list[str] = []

        class FakeBackend:
            active = False

            def activate(self):
                backend = self

                class Activation:
                    def __enter__(self):
                        self_backend = backend
                        self_backend.active = True
                        events.append("enter")

                    def __exit__(self, exc_type, exc, traceback):
                        backend.active = False
                        events.append("exit")

                return Activation()

        backend = FakeBackend()
        native = object()
        wrapped = object()

        def inputs(*_args):
            self.assertTrue(backend.active)
            events.append("inputs")
            return object(), object(), object(), object()

        def integrate(*_args):
            self.assertTrue(backend.active)
            events.append("integrate")
            return native

        def wrap(*args):
            self.assertFalse(backend.active)
            self.assertIs(args[1], native)
            events.append("wrap")
            return wrapped

        cuda_route = get_integrator_route(GR15_EIH1PN_CUDA_METHOD_ID)
        with (
            mock.patch(
                "jxplanetx.engine.simulation._preflight_integrator_route",
                return_value=cuda_route,
            ),
            mock.patch(
                "jxplanetx.engine.simulation.resolve_backend",
                return_value=backend,
            ),
            mock.patch(
                "jxplanetx.engine.simulation._gr15_eih_cuda_inputs",
                side_effect=inputs,
            ),
            mock.patch(
                "jxplanetx.gr15_eih_1pn_cuda.integrate_gr15_eih_1pn_cuda_batch",
                side_effect=integrate,
            ),
            mock.patch.object(
                JXSimulation,
                "_wrap_gr15_cuda",
                side_effect=wrap,
            ),
        ):
            result = simulation._integrate_locked(
                simulation.snapshot,
                GR15_EIH1PN_CUDA_METHOD_ID,
                object(),
                continuation=None,
            )
        self.assertIs(result, wrapped)
        self.assertEqual(events, ["enter", "inputs", "integrate", "exit", "wrap"])

    def test_continuation_archive_is_deterministic_verified_and_resumable(self) -> None:
        plan = force_plan()
        source = JXSimulation(
            "fixture.unified-runtime.archive",
            state_snapshot(),
            plan,
        )
        parent = source.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.01, 0.02)),
        )
        continuation = source.prepare_continuation(0)
        archive = dump_simulation_archive(source, 0)
        self.assertEqual(archive, dump_simulation_archive(source, 0))
        archive_sha256 = simulation_archive_sha256(archive)

        loaded = load_simulation_archive(
            archive,
            expected_sha256=archive_sha256,
            force_plan=plan,
        )
        self.assertIs(type(loaded), JXSimulationArchiveLoad)
        self.assertEqual(loaded.archive_sha256, archive_sha256)
        self.assertEqual(loaded.scientific_claim_state, "SCREENING_ONLY")
        self.assertFalse(loaded.production_authorized)
        self.assertEqual(
            loaded.continuation.continuation_state_sha256,
            continuation.continuation_state_sha256,
        )
        self.assertEqual(
            loaded.continuation.parent_native_result_digest,
            parent.native_result_digest,
        )
        np.testing.assert_array_equal(
            loaded.simulation.snapshot.positions,
            parent.final_positions,
        )
        self.assertFalse(loaded.simulation.snapshot.positions.flags.writeable)

        resumed_spec = rkf78_spec((0.02, 0.03, 0.04))
        direct = source.continue_from(0, ADAPTIVE_RKF78_METHOD_ID, resumed_spec)
        resumed = loaded.simulation.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            resumed_spec,
        )
        np.testing.assert_array_equal(resumed.final_positions, direct.final_positions)
        np.testing.assert_array_equal(resumed.final_velocities, direct.final_velocities)
        self.assertEqual(resumed.native_result_digest, direct.native_result_digest)

        with self.assertRaisesRegex(SimulationArchiveError, "SHA-256"):
            load_simulation_archive(
                archive,
                expected_sha256="0" * 64,
                force_plan=plan,
            )
        tampered = archive[:-1] + bytes((archive[-1] ^ 1,))
        with self.assertRaisesRegex(SimulationArchiveError, "SHA-256"):
            load_simulation_archive(
                tampered,
                expected_sha256=archive_sha256,
                force_plan=plan,
            )
        with self.assertRaisesRegex(SimulationArchiveError, "force plan"):
            load_simulation_archive(
                archive,
                expected_sha256=archive_sha256,
                force_plan=force_plan(tile_size=3),
            )

        absent_source_plan = force_plan(source_ids=("ABSENT",))
        forged_plan_binding = rebind_archive_force_plan(
            archive,
            absent_source_plan,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_plan_binding,
                expected_sha256=simulation_archive_sha256(forged_plan_binding),
                force_plan=absent_source_plan,
            )

        restricted_plan = restricted_missing_central_plan()
        forged_restricted = rebind_archive_force_plan_and_roster(
            archive,
            continuation,
            restricted_plan,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_restricted,
                expected_sha256=simulation_archive_sha256(forged_restricted),
                force_plan=restricted_plan,
            )

        forged_lineage = relabel_archive_parent(
            archive,
            continuation,
            GR15_EIH1PN_METHOD_ID,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_lineage,
                expected_sha256=simulation_archive_sha256(forged_lineage),
                force_plan=plan,
            )

        partial_plan = force_plan(source_ids=("PRIMARY",))
        partial_source = JXSimulation(
            "fixture.unified-runtime.partial-newtonian",
            state_snapshot(),
            partial_plan,
        )
        partial_source.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.01, 0.02)),
        )
        partial_continuation = partial_source.prepare_continuation(0)
        partial_archive = dump_simulation_archive(partial_source, 0)
        forged_configuration = relabel_archive_parent(
            partial_archive,
            partial_continuation,
            GR15_NEWTONIAN_METHOD_ID,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_configuration,
                expected_sha256=simulation_archive_sha256(
                    forged_configuration
                ),
                force_plan=partial_plan,
            )

        single_plan = force_plan(
            source_ids=("PRIMARY",),
            target_ids=("PRIMARY",),
        )
        single_source = JXSimulation(
            "fixture.unified-runtime.single-body",
            single_body_state_snapshot(),
            single_plan,
        )
        single_source.integrate(
            ADAPTIVE_RKF78_METHOD_ID,
            rkf78_spec((0.0, 0.01, 0.02), body_count=1),
        )
        single_continuation = single_source.prepare_continuation(0)
        single_archive = dump_simulation_archive(single_source, 0)
        forged_body_count = relabel_archive_parent(
            single_archive,
            single_continuation,
            GR15_NEWTONIAN_METHOD_ID,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_body_count,
                expected_sha256=simulation_archive_sha256(forged_body_count),
                force_plan=single_plan,
            )

        expired_plan = force_plan(validity_end=0.0)
        expired_binding = rebind_archive_force_plan(archive, expired_plan)
        forged_expired_endpoint = relabel_archive_parent(
            expired_binding,
            continuation,
            GR15_NEWTONIAN_METHOD_ID,
        )
        with self.assertRaisesRegex(
            SimulationArchiveError,
            "incompatible with the force plan",
        ):
            load_simulation_archive(
                forged_expired_endpoint,
                expected_sha256=simulation_archive_sha256(
                    forged_expired_endpoint
                ),
                force_plan=expired_plan,
            )

if __name__ == "__main__":
    unittest.main()
