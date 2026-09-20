import dataclasses
import math
import unittest
from unittest import mock

import numpy as np

import jxplanetx.engine as engine
import jxplanetx.engine.api as engine_api
import jxplanetx.engine.scenario as scenario_runtime
from jxplanetx.engine import (
    AdaptiveRKF78Spec,
    BackendSpec,
    CannonballSRP,
    DynamicsScenario,
    ForcePlan,
    MatchedScenarioComparison,
    MatchedScenarioResult,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    ScenarioCheckpointDelta,
    ScenarioContractError,
    StateSnapshot,
    run_dynamics_scenario,
    run_matched_scenario_comparison,
)


UNIT_SYSTEM_ID = "fixture.scenario.au_day_solar_mass"


def provenance() -> Provenance:
    return Provenance(
        "fixture.scenario",
        "Synthetic scenario fixture",
        "1",
        "9" * 64,
    )


def metadata(parameter_id: str, units: str) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id,
        units,
        provenance(),
        None,
        None,
        0.0,
        1.0,
    )


def backend() -> BackendSpec:
    return BackendSpec(
        backend_id="numpy",
        device="cpu",
        tile_size=2,
        dtype="float64",
        allow_fallback=False,
        deterministic_reductions=True,
        fast_math=False,
        determinism_scope="SAME_RUNTIME_DEVICE",
    )


def snapshot() -> StateSnapshot:
    return StateSnapshot(
        snapshot_id="fixture.scenario.snapshot",
        epoch=0.0,
        time_scale="TDB",
        frame="CENTRAL_BODY_INERTIAL",
        origin="SUN",
        axes="ICRS_ALIGNED",
        length_unit="AU",
        time_unit="DAY",
        mass_unit="SOLAR_MASS",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=("SUN", "ORBITER", "SAIL"),
        positions=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        velocities=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, math.sqrt(0.5), 0.0),
            ),
            dtype=np.float64,
        ),
        gravitational_parameters=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        masses=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        radii=np.zeros(3, dtype=np.float64),
        massive=np.asarray((True, False, False), dtype=np.bool_),
        provenance=provenance(),
    )


def gravity() -> NewtonianPointMass:
    return NewtonianPointMass(
        source_ids=("SUN",),
        target_ids=("ORBITER", "SAIL"),
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            metadata("state.gravitational_parameters", "AU^3/DAY^2"),
        ),
    )


def srp() -> CannonballSRP:
    return CannonballSRP(
        radiation_source_id="SUN",
        target_ids=("SAIL",),
        reference_pressure=1.0e-8,
        reference_distance=1.0,
        area_to_mass=np.asarray((1.0,), dtype=np.float64),
        radiation_pressure_coefficient=np.asarray((1.0,), dtype=np.float64),
        coefficient_convention="QPR",
        attitude_model="ISOTROPIC_CANNONBALL",
        shadow_model="NONE",
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            metadata("reference_pressure", "SOLAR_MASS/(AU*DAY^2)"),
            metadata("reference_distance", "AU"),
            metadata("area_to_mass", "AU^2/SOLAR_MASS"),
            metadata("radiation_pressure_coefficient", "1"),
        ),
    )


def integration_spec() -> AdaptiveRKF78Spec:
    tolerance = np.full((3, 3), 1.0e-12, dtype=np.float64)
    return AdaptiveRKF78Spec(
        checkpoint_epochs=(0.0, 0.01, 0.02),
        initial_step=0.01,
        minimum_step=1.0e-12,
        maximum_step=0.01,
        position_atol=tolerance,
        position_rtol=1.0e-10,
        velocity_atol=tolerance.copy(),
        velocity_rtol=1.0e-10,
        maximum_steps=1_000,
        maximum_rejections=100,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
    )


def scenario_pair() -> tuple[
    DynamicsScenario,
    DynamicsScenario,
    MatchedScenarioComparison,
]:
    state = snapshot()
    request = integration_spec()
    backend_spec = backend()
    base_gravity = gravity()
    control = DynamicsScenario(
        scenario_id="fixture.scenario.control",
        role="CONTROL",
        description="Newtonian point-mass control",
        comparison_id="fixture.scenario.comparison",
        initial_snapshot=state,
        force_plan=ForcePlan(
            "fixture.scenario.control.plan",
            backend_spec,
            (base_gravity,),
        ),
        integration_spec=request,
    )
    candidate = DynamicsScenario(
        scenario_id="fixture.scenario.candidate",
        role="CANDIDATE",
        description="Control plus an explicit SRP term",
        comparison_id="fixture.scenario.comparison",
        initial_snapshot=state,
        force_plan=ForcePlan(
            "fixture.scenario.candidate.plan",
            backend_spec,
            (base_gravity, srp()),
        ),
        integration_spec=request,
    )
    comparison = MatchedScenarioComparison(
        comparison_id="fixture.scenario.comparison",
        scientific_question="What state difference is induced by the appended SRP term?",
        control=control,
        candidate=candidate,
        compared_body_ids=("SUN", "ORBITER", "SAIL"),
        added_force_model_ids=("force.nongrav.srp_cannonball",),
    )
    return control, candidate, comparison


class DynamicsScenarioContractTests(unittest.TestCase):
    def test_standalone_and_matched_roles_are_explicit_and_nonauthorizing(self):
        state = snapshot()
        request = integration_spec()
        standalone = DynamicsScenario(
            "fixture.scenario.standalone",
            "STANDALONE",
            "One explicit baseline run",
            None,
            state,
            ForcePlan("fixture.scenario.standalone.plan", backend(), (gravity(),)),
            request,
        )
        self.assertEqual(standalone.force_model_ids, ("force.newtonian.point_mass",))
        self.assertFalse(standalone.qualified)
        self.assertFalse(standalone.accuracy_claimed)
        self.assertFalse(standalone.improvement_claimed)
        self.assertFalse(standalone.physics_claimed)

        control, candidate, comparison = scenario_pair()
        self.assertEqual(control.role, "CONTROL")
        self.assertEqual(candidate.role, "CANDIDATE")
        self.assertEqual(comparison.added_force_model_ids, candidate.force_model_ids[-1:])
        self.assertFalse(comparison.qualified)

    def test_scenario_rejects_ambiguous_roles_epochs_and_claim_promotion(self):
        control, _, _ = scenario_pair()
        mutations = (
            ("role", "BASELINE", "role must be exactly"),
            ("comparison_id", None, "comparison_id must be"),
            ("accuracy_claimed", True, "accuracy_claimed"),
            ("qualification_authorized", 0, "qualification_authorized"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ScenarioContractError, message):
                    dataclasses.replace(control, **{field: value})

        wrong_epoch_spec = dataclasses.replace(
            control.integration_spec,
            checkpoint_epochs=(0.25, 0.5),
        )
        with self.assertRaisesRegex(ScenarioContractError, "first checkpoint epoch"):
            dataclasses.replace(control, integration_spec=wrong_epoch_spec)

    def test_matched_contract_rejects_hidden_state_solver_or_force_changes(self):
        control, candidate, comparison = scenario_pair()

        different_state_candidate = dataclasses.replace(
            candidate,
            initial_snapshot=snapshot(),
        )
        with self.assertRaisesRegex(ScenarioContractError, "exact initial_snapshot"):
            dataclasses.replace(comparison, candidate=different_state_candidate)

        different_spec_candidate = dataclasses.replace(
            candidate,
            integration_spec=dataclasses.replace(candidate.integration_spec),
        )
        with self.assertRaisesRegex(ScenarioContractError, "exact integration_spec"):
            dataclasses.replace(comparison, candidate=different_spec_candidate)

        changed_prefix_plan = ForcePlan(
            candidate.force_plan.plan_id,
            candidate.force_plan.backend,
            (dataclasses.replace(control.force_plan.models[0]), candidate.force_plan.models[1]),
        )
        changed_prefix_candidate = dataclasses.replace(
            candidate,
            force_plan=changed_prefix_plan,
        )
        with self.assertRaisesRegex(ScenarioContractError, "exact ordered control force prefix"):
            dataclasses.replace(comparison, candidate=changed_prefix_candidate)

        with self.assertRaisesRegex(ScenarioContractError, "exactly name"):
            dataclasses.replace(
                comparison,
                added_force_model_ids=("force.newtonian.point_mass",),
            )
        with self.assertRaisesRegex(ScenarioContractError, "unknown bodies"):
            dataclasses.replace(comparison, compared_body_ids=("UNKNOWN",))


class DynamicsScenarioRuntimeTests(unittest.TestCase):
    def test_standalone_execution_retains_scenario_and_trajectory_claim_ceiling(self):
        control, _, _ = scenario_pair()
        result = run_dynamics_scenario(control)
        self.assertIs(result.scenario, control)
        self.assertEqual(result.trajectory.snapshot_id, control.initial_snapshot.snapshot_id)
        self.assertEqual(result.trajectory.plan_id, control.force_plan.plan_id)
        self.assertEqual(result.trajectory.checkpoint_epochs, (0.0, 0.01, 0.02))
        self.assertEqual(result.trajectory.force_model_ids, control.force_model_ids)
        self.assertIs(result.final_positions, result.trajectory.final_positions)
        self.assertIs(result.final_velocities, result.trajectory.final_velocities)
        self.assertFalse(result.qualified)
        self.assertFalse(result.accuracy_claimed)
        self.assertFalse(result.improvement_claimed)
        self.assertFalse(result.physics_claimed)

    def test_matched_execution_reports_only_fixed_order_model_differences(self):
        control, candidate, comparison = scenario_pair()
        result = run_matched_scenario_comparison(comparison)

        self.assertIsInstance(result, MatchedScenarioResult)
        self.assertIs(result.control_result.scenario, control)
        self.assertIs(result.candidate_result.scenario, candidate)
        self.assertEqual(len(result.checkpoint_deltas), 9)
        self.assertEqual(
            tuple((row.checkpoint_index, row.body_id) for row in result.checkpoint_deltas),
            tuple(
                (checkpoint_index, body_id)
                for checkpoint_index in range(3)
                for body_id in ("SUN", "ORBITER", "SAIL")
            ),
        )
        self.assertTrue(
            all(
                row.position_difference_norm == 0.0
                and row.velocity_difference_norm == 0.0
                for row in result.checkpoint_deltas[:3]
            )
        )
        unaffected = tuple(
            row
            for row in result.checkpoint_deltas
            if row.body_id in {"SUN", "ORBITER"}
        )
        self.assertTrue(
            all(
                row.position_difference_norm == 0.0
                and row.velocity_difference_norm == 0.0
                for row in unaffected
            )
        )
        sail_final = result.checkpoint_deltas[-1]
        self.assertEqual(sail_final.body_id, "SAIL")
        self.assertGreater(sail_final.position_difference_norm, 0.0)
        self.assertGreater(sail_final.velocity_difference_norm, 0.0)
        self.assertEqual(sail_final.position_unit, "AU")
        self.assertEqual(sail_final.velocity_unit, "AU/DAY")
        self.assertEqual(
            result.comparison_semantics,
            "MODEL_TO_MODEL_CHECKPOINT_DIFFERENCE_NOT_ACCURACY_OR_IMPROVEMENT",
        )
        self.assertFalse(result.accuracy_claimed)
        self.assertFalse(result.improvement_claimed)
        self.assertFalse(result.physics_claimed)

    def test_candidate_executes_from_the_controls_retained_baseline_copies(self):
        control, _, comparison = scenario_pair()
        caller_positions = control.initial_snapshot.positions
        real_integrate = scenario_runtime.integrate_trajectory
        calls: list[tuple[StateSnapshot, ForcePlan, AdaptiveRKF78Spec]] = []

        def observed_integrate(
            state: StateSnapshot,
            plan: ForcePlan,
            request: AdaptiveRKF78Spec,
        ):
            calls.append((state, plan, request))
            trajectory = real_integrate(state, plan, request)
            if len(calls) == 1:
                caller_positions[1, 0] = 9.0
            return trajectory

        with mock.patch.object(
            scenario_runtime,
            "integrate_trajectory",
            side_effect=observed_integrate,
        ):
            result = run_matched_scenario_comparison(comparison)

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], control.initial_snapshot)
        self.assertIs(
            calls[1][0],
            result.control_result.trajectory.initial_snapshot,
        )
        self.assertIs(
            calls[1][2],
            result.control_result.trajectory.integration_spec,
        )
        self.assertIsNot(calls[1][0].positions, caller_positions)
        self.assertEqual(float(calls[1][0].positions[1, 0]), 1.0)
        self.assertEqual(float(caller_positions[1, 0]), 9.0)
        self.assertEqual(
            calls[1][1].models[0].model_id,
            result.control_result.trajectory.force_plan.models[0].model_id,
        )

    def test_result_recomputes_and_rejects_tampered_delta_records(self):
        _, _, comparison = scenario_pair()
        result = run_matched_scenario_comparison(comparison)
        rows = list(result.checkpoint_deltas)
        rows[-1] = dataclasses.replace(
            rows[-1],
            position_difference_norm=rows[-1].position_difference_norm * 2.0,
        )
        with self.assertRaisesRegex(ScenarioContractError, "does not match"):
            dataclasses.replace(result, checkpoint_deltas=tuple(rows))

        with self.assertRaisesRegex(ScenarioContractError, "finite and nonnegative"):
            ScenarioCheckpointDelta(
                checkpoint_index=0,
                epoch=0.0,
                body_id="SAIL",
                position_difference_norm=-1.0,
                velocity_difference_norm=0.0,
                position_unit="AU",
                velocity_unit="AU/DAY",
            )

    def test_runtime_entry_points_require_exact_public_contracts(self):
        with self.assertRaisesRegex(ScenarioContractError, "exact DynamicsScenario"):
            run_dynamics_scenario(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            ScenarioContractError,
            "exact MatchedScenarioComparison",
        ):
            run_matched_scenario_comparison(object())  # type: ignore[arg-type]


class DynamicsScenarioPublicApiTests(unittest.TestCase):
    def test_api_and_package_root_export_the_same_scenario_surface(self):
        expected = (
            "DYNAMICS_SCENARIO_SCOPE",
            "DynamicsScenario",
            "DynamicsScenarioResult",
            "MATCHED_SCENARIO_SCOPE",
            "MatchedScenarioComparison",
            "MatchedScenarioResult",
            "SCENARIO_COMPARISON_SEMANTICS",
            "SCENARIO_EVIDENCE_CLASS",
            "SCENARIO_EXECUTION_ORDER",
            "ScenarioCheckpointDelta",
            "ScenarioContractError",
            "run_dynamics_scenario",
            "run_matched_scenario_comparison",
        )
        for module in (engine, engine_api):
            with self.subTest(module=module.__name__):
                self.assertTrue(all(name in module.__all__ for name in expected))
                self.assertEqual(
                    tuple(getattr(module, name) for name in expected),
                    tuple(getattr(engine, name) for name in expected),
                )


if __name__ == "__main__":
    unittest.main()
