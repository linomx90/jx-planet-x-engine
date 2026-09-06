import dataclasses
import inspect
import unittest

import numpy as np

import jxplanetx.engine as engine
import jxplanetx.engine.api as engine_api
import jxplanetx.engine.encounter as encounter_runtime
import jxplanetx.engine.encounter_contracts as encounter_contracts
import jxplanetx.engine.hybrid as hybrid_runtime
import jxplanetx.engine.hybrid_contracts as hybrid_contracts
import jxplanetx.engine.wisdom_holman as wisdom_holman_runtime
import jxplanetx.engine.wisdom_holman_contracts as wisdom_holman_contracts
from jxplanetx.engine import (
    BackendArrayError,
    BackendSpec,
    CannonballSRP,
    DuplicateForceError,
    EvaluationError,
    FIXED_STEP_KDK_METHOD_ID,
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    FixedStepKDKSpec,
    FixedStepWisdomHolmanSpec,
    ForceDomainError,
    ForcePlan,
    KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
    KDK_SCHEDULE_CHECKSUM_ALGORITHM,
    KDK_SCHEDULE_CHECKSUM_DOMAIN,
    KDKTrajectoryResult,
    JacobiCoordinateBinding,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    RKF78_CHECKPOINT_PROPOSAL_POLICY,
    RKF78_TIME_STEP_REPRESENTATION,
    StateSnapshot,
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryError,
    TrajectoryResult,
    TrajectoryStepLimitError,
    UniversalKeplerSolverSpec,
    UnsupportedForceError,
    WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
    WH_SCHEDULE_CHECKSUM_ALGORITHM,
    WH_SCHEDULE_CHECKSUM_DOMAIN,
    WisdomHolmanTrajectoryResult,
    evaluate,
    evaluate_forces,
    integrate_trajectory,
    integrate_kdk_trajectory,
    integrate_wisdom_holman_trajectory,
)
from jxplanetx.engine.evaluator import evaluate_force_plan


UNIT_SYSTEM_ID = "fixture.au_day_solar_mass"


def provenance() -> Provenance:
    return Provenance("fixture.state", "Synthetic API fixture", "1", "c" * 64)


def metadata(
    parameter_id: str,
    units: str,
    *,
    validity_start: float = -1.0,
    validity_end: float = 1.0,
) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id,
        units,
        provenance(),
        None,
        None,
        validity_start,
        validity_end,
    )


def newtonian_metadata() -> tuple[ParameterMetadata, ...]:
    return (metadata("state.gravitational_parameters", "AU^3/DAY^2"),)


def one_pn_metadata() -> tuple[ParameterMetadata, ...]:
    return (
        metadata("speed_of_light", "AU/DAY"),
        metadata("maximum_compactness", "1"),
        metadata("maximum_speed_fraction_squared", "1"),
    )


def srp_metadata() -> tuple[ParameterMetadata, ...]:
    return (
        metadata("reference_pressure", "SOLAR_MASS/(AU*DAY^2)"),
        metadata("reference_distance", "AU"),
        metadata("area_to_mass", "AU^2/SOLAR_MASS"),
        metadata("radiation_pressure_coefficient", "1"),
    )


def snapshot(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "fixture.snapshot",
        "epoch": 0.0,
        "time_scale": "TDB",
        "frame": "CENTRAL_BODY_INERTIAL",
        "origin": "SUN",
        "axes": "ICRS_ALIGNED",
        "length_unit": "AU",
        "time_unit": "DAY",
        "mass_unit": "SOLAR_MASS",
        "unit_system_id": UNIT_SYSTEM_ID,
        "body_ids": ("SUN", "REL", "SRP"),
        "positions": np.array(
            ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        "velocities": np.array(
            ((0.0, 0.0, 0.0), (2.0, -1.0, 1.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        "gravitational_parameters": np.array((7.0, 0.0, 0.0), dtype=np.float64),
        "masses": np.array((1.0, 0.0, 0.0), dtype=np.float64),
        "radii": np.zeros(3, dtype=np.float64),
        "massive": np.array((True, False, False), dtype=np.bool_),
        "provenance": provenance(),
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def backend(**changes: object) -> BackendSpec:
    values: dict[str, object] = {
        "backend_id": "numpy",
        "device": "cpu",
        "tile_size": 2,
        "dtype": "float64",
        "allow_fallback": False,
        "deterministic_reductions": True,
        "fast_math": False,
        "determinism_scope": "SAME_RUNTIME_DEVICE",
    }
    values.update(changes)
    return BackendSpec(**values)  # type: ignore[arg-type]


def newtonian(**changes: object) -> NewtonianPointMass:
    values: dict[str, object] = {
        "source_ids": ("SUN",),
        "target_ids": ("SUN", "REL", "SRP"),
        "unit_system_id": UNIT_SYSTEM_ID,
        "parameter_metadata": newtonian_metadata(),
    }
    values.update(changes)
    return NewtonianPointMass(**values)  # type: ignore[arg-type]


def one_pn(**changes: object) -> RestrictedStaticCentral1PN:
    values: dict[str, object] = {
        "central_source_id": "SUN",
        "target_ids": ("REL",),
        "speed_of_light": 100.0,
        "maximum_compactness": 0.1,
        "maximum_speed_fraction_squared": 0.1,
        "unit_system_id": UNIT_SYSTEM_ID,
        "parameter_metadata": one_pn_metadata(),
    }
    values.update(changes)
    return RestrictedStaticCentral1PN(**values)  # type: ignore[arg-type]


def srp(**changes: object) -> CannonballSRP:
    values: dict[str, object] = {
        "radiation_source_id": "SUN",
        "target_ids": ("SRP",),
        "reference_pressure": 5.0,
        "reference_distance": 2.0,
        "area_to_mass": np.array((3.0,), dtype=np.float64),
        "radiation_pressure_coefficient": np.array((2.0,), dtype=np.float64),
        "coefficient_convention": "QPR",
        "attitude_model": "ISOTROPIC_CANNONBALL",
        "shadow_model": "NONE",
        "unit_system_id": UNIT_SYSTEM_ID,
        "parameter_metadata": srp_metadata(),
    }
    values.update(changes)
    return CannonballSRP(**values)  # type: ignore[arg-type]


def force_plan(*models: object, backend_spec: BackendSpec | None = None) -> ForcePlan:
    return ForcePlan(
        "fixture.plan",
        backend_spec or backend(),
        tuple(models) if models else (newtonian(), one_pn(), srp()),
    )


class PublicEvaluationTests(unittest.TestCase):
    def test_new_rkf78_provenance_constants_have_the_same_public_roster(self):
        expected = (
            "RKF78_ACCEPTED_STATE_ACCUMULATION",
            "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM",
            "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN",
            "RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE",
            "RKF78_CHECKPOINT_PROPOSAL_POLICY",
            "RKF78_TIME_STEP_REPRESENTATION",
        )
        expected_values = {
            "RKF78_ACCEPTED_STATE_ACCUMULATION": (
                "KAHAN_BACKEND_NATIVE_COMPONENTWISE"
            ),
            "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM": (
                "SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1"
            ),
            "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN": (
                "jxplanetx.accepted-step-ledger.content-integrity.v1"
            ),
            "RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE": (
                "ABS_SIGNED_RKF78_STEP_ARGUMENT"
            ),
            "RKF78_CHECKPOINT_PROPOSAL_POLICY": (
                "PRESERVE_PRECLIP_PROPOSAL_AFTER_ACCEPTED_CLIP"
            ),
            "RKF78_TIME_STEP_REPRESENTATION": (
                "REPRESENTABLE_ENDPOINT_DELTA"
            ),
        }
        for module in (engine, engine_api):
            with self.subTest(module=module.__name__):
                roster = tuple(name for name in module.__all__ if name in expected)
                self.assertEqual(roster, expected)
                self.assertEqual(
                    {name: getattr(module, name) for name in expected},
                    expected_values,
                )

    def test_kdk_public_roster_is_identical_at_api_and_package_root(self):
        expected = (
            "FIXED_STEP_KDK_METHOD_ID",
            "FixedStepKDKSpec",
            "KDK_BACKEND_SCOPE",
            "KDK_CHECKPOINT_POLICY",
            "KDK_COMPOSITION",
            "KDK_ENCOUNTER_GUARD",
            "KDK_FORCE_EVALUATION_ACCOUNTING",
            "KDK_FORCE_PLAN_SCOPE",
            "KDK_PAIR_FREQUENCY_GUARD",
            "KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM",
            "KDK_RESULT_CONTENT_CHECKSUM_DOMAIN",
            "KDK_SCHEDULE_CHECKSUM_ALGORITHM",
            "KDK_SCHEDULE_CHECKSUM_DOMAIN",
            "KDK_STEP_REPRESENTATION",
            "KDK_TIME_SEMANTICS",
            "KDK_TRAJECTORY_SCOPE",
            "KDKTrajectoryResult",
            "integrate_kdk_trajectory",
        )
        for module in (engine, engine_api):
            with self.subTest(module=module.__name__):
                roster = tuple(
                    name
                    for name in module.__all__
                    if name.startswith("KDK")
                    or name
                    in {
                        "FIXED_STEP_KDK_METHOD_ID",
                        "FixedStepKDKSpec",
                        "integrate_kdk_trajectory",
                    }
                )
                self.assertEqual(roster, expected)
                self.assertEqual(
                    {name: getattr(module, name) for name in expected},
                    {name: getattr(engine, name) for name in expected},
                )

    def test_wisdom_holman_public_roster_is_complete_and_identical(self):
        expected_names = frozenset(wisdom_holman_contracts.__all__) | frozenset(
            wisdom_holman_runtime.__all__
        )
        self.assertIn("FIXED_STEP_WISDOM_HOLMAN_METHOD_ID", expected_names)
        self.assertIn("FixedStepWisdomHolmanSpec", expected_names)
        self.assertIn("UniversalKeplerSolverSpec", expected_names)
        self.assertIn("JacobiCoordinateBinding", expected_names)
        self.assertIn("WisdomHolmanTrajectoryResult", expected_names)
        self.assertIn("integrate_wisdom_holman_trajectory", expected_names)
        api_roster = tuple(name for name in engine_api.__all__ if name in expected_names)
        root_roster = tuple(name for name in engine.__all__ if name in expected_names)
        self.assertEqual(api_roster, root_roster)
        self.assertEqual(frozenset(api_roster), expected_names)
        self.assertEqual(len(api_roster), len(expected_names))
        for name in api_roster:
            with self.subTest(name=name):
                self.assertIs(getattr(engine_api, name), getattr(engine, name))

    def test_encounter_public_roster_is_complete_exact_and_identical(self):
        expected = tuple(
            dict.fromkeys(
                (*encounter_contracts.__all__, *encounter_runtime.__all__)
            )
        )
        expected_names = frozenset(expected)
        self.assertEqual(len(expected), len(expected_names))
        for required in (
            "ADAPTIVE_ENCOUNTER_SEGMENT_METHOD_ID",
            "AdaptiveEncounterSegmentSpec",
            "EncounterExactRationalResourceSpec",
            "EncounterSegmentResult",
            "EncounterExecutionCounts",
            "EncounterResourceError",
            "integrate_encounter_segment",
        ):
            self.assertIn(required, expected_names)

        api_roster = tuple(
            name for name in engine_api.__all__ if name in expected_names
        )
        root_roster = tuple(
            name for name in engine.__all__ if name in expected_names
        )
        self.assertEqual(api_roster, expected)
        self.assertEqual(root_roster, expected)
        for name in expected:
            with self.subTest(name=name):
                source = (
                    encounter_contracts
                    if name in encounter_contracts.__all__
                    else encounter_runtime
                )
                self.assertIs(getattr(engine_api, name), getattr(source, name))
                self.assertIs(getattr(engine, name), getattr(source, name))

    def test_hybrid_public_roster_is_complete_exact_and_identical(self):
        expected = tuple(
            dict.fromkeys(
                (*hybrid_contracts.__all__, *hybrid_runtime.__all__)
            )
        )
        expected_names = frozenset(expected)
        self.assertEqual(len(expected), 109)
        self.assertEqual(len(expected), len(expected_names))
        for required in (
            "HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID",
            "HybridEncounterControlProfile",
            "HybridWisdomHolmanRKF78Spec",
            "HybridFarProbeDecision",
            "HybridFarProbeWork",
            "HybridKeplerProbeRecord",
            "HybridPrivateEncounterRecord",
            "HybridLaneExecutionCounts",
            "HybridOuterStepRecord",
            "HybridWisdomHolmanRKF78Result",
            "HybridContractError",
            "integrate_hybrid_wisdom_holman_rkf78_trajectory",
        ):
            self.assertIn(required, expected_names)

        api_roster = tuple(
            name for name in engine_api.__all__ if name in expected_names
        )
        root_roster = tuple(
            name for name in engine.__all__ if name in expected_names
        )
        self.assertEqual(api_roster, expected)
        self.assertEqual(root_roster, expected)
        for name in expected:
            with self.subTest(name=name):
                source = (
                    hybrid_contracts
                    if name in hybrid_contracts.__all__
                    else hybrid_runtime
                )
                self.assertIs(getattr(engine_api, name), getattr(source, name))
                self.assertIs(getattr(engine, name), getattr(source, name))

    def test_three_model_plan_has_exact_ledger_and_total(self):
        state = snapshot()
        plan = force_plan()
        state_buffers = tuple(
            getattr(state, name).copy()
            for name in (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            )
        )
        srp_buffers = tuple(
            getattr(plan.models[2], name).copy()
            for name in ("area_to_mass", "radiation_pressure_coefficient")
        )

        result = evaluate(state, plan)
        self.assertEqual(result.total_acceleration.shape, (3, 3))
        self.assertEqual(result.total_acceleration.dtype, np.dtype("float64"))
        self.assertEqual(result.backend_id, "numpy")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.tile_size, 2)
        self.assertIs(result.backend_spec, plan.backend)
        self.assertEqual(result.scope, "FORCE_EVALUATION_ONLY")
        self.assertEqual(result.evidence_class, "MODEL_OUTPUT")
        self.assertFalse(result.integrated)
        self.assertFalse(result.qualified)
        self.assertFalse(result.registry_authorized)
        self.assertFalse(result.qualification_authorized)
        self.assertEqual(result.determinism_scope, "SAME_RUNTIME_DEVICE")

        expected_ids = (
            "force.newtonian.point_mass",
            "relativity.solar_schwarzschild_test_particle_1pn",
            "force.nongrav.srp_cannonball",
        )
        self.assertEqual(result.applied_model_ids, expected_ids)
        self.assertEqual(tuple(entry.order for entry in result.ledger), (0, 1, 2))
        self.assertEqual(
            tuple(entry.role for entry in result.ledger),
            ("NEWTONIAN_BASE", "CORRECTION_ONLY", "CORRECTION_ONLY"),
        )
        self.assertEqual(tuple(item.model_id for item in result.contributions), expected_ids)
        for entry in result.ledger:
            self.assertEqual(entry.evidence_class, "MODEL_OUTPUT")
            self.assertFalse(entry.qualified)
            self.assertFalse(entry.registry_authorized)
            self.assertFalse(entry.qualification_authorized)
            self.assertEqual(entry.tile_size, plan.backend.tile_size)
            self.assertEqual(entry.determinism_scope, "SAME_RUNTIME_DEVICE")
            self.assertTrue(entry.assumptions)
            self.assertEqual(entry.state_metadata, result.state_metadata)

        summed = np.zeros_like(result.total_acceleration)
        for contribution in result.contributions:
            summed = summed + contribution.acceleration
        np.testing.assert_array_equal(result.total_acceleration, summed)
        np.testing.assert_allclose(
            result.contributions[1].acceleration[1],
            np.array((8.288e-5, -5.376e-5, 4.48e-5)),
            rtol=2e-15,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            result.contributions[2].acceleration[2],
            np.array((30.0, 0.0, 0.0)),
        )

        for name, original in zip(
            (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            ),
            state_buffers,
        ):
            np.testing.assert_array_equal(getattr(state, name), original)
        for name, original in zip(
            ("area_to_mass", "radiation_pressure_coefficient"), srp_buffers
        ):
            np.testing.assert_array_equal(getattr(plan.models[2], name), original)

    def test_result_binds_complete_nonarray_state_context(self):
        state = snapshot()
        result = evaluate(state, force_plan(newtonian()))
        binding = result.state_metadata
        self.assertEqual(binding.snapshot_id, state.snapshot_id)
        self.assertEqual(binding.epoch, state.epoch)
        self.assertEqual(binding.unit_system_id, state.unit_system_id)
        self.assertEqual(binding.time_scale, state.time_scale)
        self.assertEqual(binding.frame, state.frame)
        self.assertEqual(binding.origin, state.origin)
        self.assertEqual(binding.axes, state.axes)
        self.assertEqual(binding.length_unit, state.length_unit)
        self.assertEqual(binding.time_unit, state.time_unit)
        self.assertEqual(binding.mass_unit, state.mass_unit)
        self.assertEqual(binding.body_ids, state.body_ids)
        self.assertEqual(binding.provenance_sha256, state.provenance.sha256)

    def test_force_api_has_no_tile_override_and_trajectory_api_is_explicit(self):
        for function in (evaluate, evaluate_forces, evaluate_force_plan):
            with self.subTest(function=function.__name__):
                self.assertEqual(tuple(inspect.signature(function).parameters), ("snapshot", "plan"))
                with self.assertRaises(TypeError):
                    function(snapshot(), force_plan(newtonian()), tile_size=1)  # type: ignore[call-arg]
        self.assertEqual(
            tuple(inspect.signature(integrate_trajectory).parameters),
            ("snapshot", "plan", "spec"),
        )
        self.assertEqual(
            tuple(inspect.signature(integrate_kdk_trajectory).parameters),
            ("snapshot", "plan", "spec"),
        )
        self.assertEqual(
            tuple(inspect.signature(integrate_wisdom_holman_trajectory).parameters),
            ("snapshot", "plan", "spec"),
        )
        self.assertEqual(
            tuple(inspect.signature(engine.integrate_encounter_segment).parameters),
            ("snapshot", "plan", "spec"),
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    engine.integrate_hybrid_wisdom_holman_rkf78_trajectory
                ).parameters
            ),
            ("snapshot", "plan", "spec"),
        )
        for name, value in (
            ("integrate_trajectory", integrate_trajectory),
            ("TrajectoryCheckpoint", TrajectoryCheckpoint),
            ("TrajectoryResult", TrajectoryResult),
            ("TrajectoryError", TrajectoryError),
            ("TrajectoryContractError", TrajectoryContractError),
            ("TrajectoryDomainError", TrajectoryDomainError),
            ("TrajectoryStepLimitError", TrajectoryStepLimitError),
            (
                "RKF78_ACCEPTED_STATE_ACCUMULATION",
                RKF78_ACCEPTED_STATE_ACCUMULATION,
            ),
            (
                "RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE",
                RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
            ),
            (
                "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM",
                RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
            ),
            (
                "RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN",
                RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
            ),
            ("integrate_kdk_trajectory", integrate_kdk_trajectory),
            ("FixedStepKDKSpec", FixedStepKDKSpec),
            ("KDKTrajectoryResult", KDKTrajectoryResult),
            ("FIXED_STEP_KDK_METHOD_ID", FIXED_STEP_KDK_METHOD_ID),
            (
                "KDK_SCHEDULE_CHECKSUM_ALGORITHM",
                KDK_SCHEDULE_CHECKSUM_ALGORITHM,
            ),
            ("KDK_SCHEDULE_CHECKSUM_DOMAIN", KDK_SCHEDULE_CHECKSUM_DOMAIN),
            (
                "KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM",
                KDK_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            ),
            (
                "KDK_RESULT_CONTENT_CHECKSUM_DOMAIN",
                KDK_RESULT_CONTENT_CHECKSUM_DOMAIN,
            ),
            (
                "integrate_wisdom_holman_trajectory",
                integrate_wisdom_holman_trajectory,
            ),
            ("FixedStepWisdomHolmanSpec", FixedStepWisdomHolmanSpec),
            ("UniversalKeplerSolverSpec", UniversalKeplerSolverSpec),
            ("JacobiCoordinateBinding", JacobiCoordinateBinding),
            ("WisdomHolmanTrajectoryResult", WisdomHolmanTrajectoryResult),
            (
                "FIXED_STEP_WISDOM_HOLMAN_METHOD_ID",
                FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
            ),
            (
                "WH_SCHEDULE_CHECKSUM_ALGORITHM",
                WH_SCHEDULE_CHECKSUM_ALGORITHM,
            ),
            ("WH_SCHEDULE_CHECKSUM_DOMAIN", WH_SCHEDULE_CHECKSUM_DOMAIN),
            (
                "WH_RESULT_CONTENT_CHECKSUM_ALGORITHM",
                WH_RESULT_CONTENT_CHECKSUM_ALGORITHM,
            ),
            (
                "WH_RESULT_CONTENT_CHECKSUM_DOMAIN",
                WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
            ),
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(engine, name), value)
                self.assertIn(name, engine.__all__)
        self.assertFalse(hasattr(engine, "integrate"))
        self.assertFalse(hasattr(engine, "propagate"))

    def test_evaluate_aliases_produce_same_acceleration_without_result_equality_traps(self):
        state = snapshot()
        plan = force_plan(newtonian())
        first = evaluate(state, plan)
        second = evaluate_forces(state, plan)
        np.testing.assert_array_equal(first.total_acceleration, second.total_acceleration)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)
        self.assertIsInstance(hash(first), int)


class UnitFrameAndValidityTests(unittest.TestCase):
    def test_unit_system_mismatch_fails_before_arithmetic(self):
        with self.assertRaisesRegex(EvaluationError, "unit_system_id"):
            evaluate(snapshot(), force_plan(newtonian(unit_system_id="other.units")))

    def test_wrong_parameter_units_and_expired_metadata_fail_closed(self):
        model = newtonian()
        wrong_units = dataclasses.replace(model.parameter_metadata[0], units="KM^3/S^2")
        with self.assertRaisesRegex(EvaluationError, "units must be exactly"):
            evaluate(
                snapshot(),
                force_plan(dataclasses.replace(model, parameter_metadata=(wrong_units,))),
            )

        expired = dataclasses.replace(
            model.parameter_metadata[0], validity_start=-2.0, validity_end=-1.0
        )
        with self.assertRaisesRegex(EvaluationError, "validity interval"):
            evaluate(
                snapshot(),
                force_plan(dataclasses.replace(model, parameter_metadata=(expired,))),
            )

    def test_restricted_1pn_requires_exact_central_rest_frame_and_origin(self):
        plan = force_plan(newtonian(), one_pn())
        with self.assertRaisesRegex(EvaluationError, "origin"):
            evaluate(snapshot(origin="SOLAR_SYSTEM_BARYCENTER"), plan)
        with self.assertRaisesRegex(EvaluationError, "CENTRAL_BODY_INERTIAL"):
            evaluate(snapshot(frame="ICRS"), plan)

        displaced = snapshot().positions.copy()
        displaced[0] = np.array((1.0, 0.0, 0.0))
        with self.assertRaisesRegex(EvaluationError, "coordinate zero"):
            evaluate(snapshot(positions=displaced), plan)


class EvaluationValidationTests(unittest.TestCase):
    def test_all_state_buffers_validate_native_dtype_shape_finiteness_and_domain(self):
        cases: tuple[tuple[str, object], ...] = (
            ("positions", [[0.0, 0.0, 0.0]] * 3),
            ("positions", np.zeros((3, 3), dtype=np.float32)),
            ("positions", np.zeros((2, 3), dtype=np.float64)),
            ("positions", np.full((3, 3), np.inf, dtype=np.float64)),
            ("velocities", np.zeros((3, 2), dtype=np.float64)),
            ("velocities", np.full((3, 3), np.nan, dtype=np.float64)),
            ("gravitational_parameters", np.zeros(3, dtype=np.float32)),
            ("gravitational_parameters", np.array((7.0, -1.0, 0.0))),
            ("masses", np.array((1.0, -1.0, 0.0))),
            ("radii", np.array((0.0, np.nan, 0.0))),
            ("massive", np.array((1, 0, 0), dtype=np.int64)),
            ("massive", np.array((True, True, False), dtype=np.bool_)),
        )
        expected_errors = (BackendArrayError, EvaluationError, ForceDomainError)
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(expected_errors):
                evaluate(snapshot(**{field: value}), force_plan(newtonian()))

    def test_srp_per_target_buffers_validate_device_dtype_shape_and_values(self):
        cases: tuple[dict[str, object], ...] = (
            {"area_to_mass": [3.0]},
            {"area_to_mass": np.array((3.0,), dtype=np.float32)},
            {"area_to_mass": np.array((3.0, 4.0), dtype=np.float64)},
            {"area_to_mass": np.array((-1.0,), dtype=np.float64)},
            {"radiation_pressure_coefficient": np.array((np.nan,), dtype=np.float64)},
        )
        expected_errors = (BackendArrayError, EvaluationError, ForceDomainError)
        for change in cases:
            with self.subTest(change=change), self.assertRaises(expected_errors):
                evaluate(snapshot(), force_plan(newtonian(), srp(**change)))

    def test_unknown_order_duplicate_and_dependency_mutations_fail_closed(self):
        class UnknownForce:
            model_id = "force.fixture.unknown"

        with self.assertRaises(UnsupportedForceError):
            evaluate(snapshot(), force_plan(UnknownForce()))

        with self.assertRaisesRegex(EvaluationError, "canonical order"):
            evaluate(snapshot(), force_plan(srp(), newtonian()))

        duplicate_plan = force_plan(newtonian())
        object.__setattr__(
            duplicate_plan,
            "models",
            (duplicate_plan.models[0], duplicate_plan.models[0]),
        )
        with self.assertRaises(DuplicateForceError):
            evaluate(snapshot(), duplicate_plan)

        with self.assertRaisesRegex(EvaluationError, "Newtonian base"):
            evaluate(snapshot(), force_plan(one_pn()))

        with self.assertRaisesRegex(EvaluationError, "Newtonian targets"):
            evaluate(
                snapshot(),
                force_plan(
                    newtonian(target_ids=("SUN", "SRP")),
                    one_pn(),
                ),
            )


if __name__ == "__main__":
    unittest.main()
