import ast
import hashlib
import math
import unittest
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import jxplanetx.v5_implicit_midpoint as midpoint_module
from jxplanetx.solar_1pn import (
    AccelerationSemantics,
    CentralSourceModel,
    CoherentUnitSystem,
    CoordinateTimeScale,
    InertialFrame,
    SolarSchwarzschild1PNContract,
    TargetTreatment,
)
from jxplanetx.v5_implicit_midpoint import (
    ImplicitMidpointSpec,
    NonlinearSolveError,
    ReferenceIntegrationError,
    ReferenceTrajectoryResult,
    implicit_midpoint_step,
    integrate_reference_trajectory,
)
from jxplanetx.v5_reference_dynamics import (
    DecimalContextSpec,
    NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID,
    REFERENCE_FORCE_MODEL_IDS,
    REFERENCE_TRAPS,
    ReferenceDynamicsDomainError,
    ReferenceDynamicsError,
    ReferenceForceCompositionError,
    bind_reference_state,
    evaluate_reference_rhs,
    inspect_reference_force_plan,
)


D = Decimal
ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registries" / "jx_force_parameters_v5.json"
FRAME = InertialFrame.BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED
SCOPE = TargetTreatment.MASSLESS_NO_BACKREACTION


def solar_contract(
    mu: str = "1",
    c: str = "100",
    *,
    units: CoherentUnitSystem = CoherentUnitSystem.AU_DAY,
    time_scale: CoordinateTimeScale = CoordinateTimeScale.TDB_COMPATIBLE,
) -> SolarSchwarzschild1PNContract:
    return SolarSchwarzschild1PNContract(
        central_source="SUN",
        gravitational_parameter=D(mu),
        speed_of_light=D(c),
        units=units,
        frame=FRAME,
        time_scale=time_scale,
        central_source_model=CentralSourceModel.STATIC_SPHERICAL_SOLAR_MONOPOLE,
        target_treatment=SCOPE,
        output_semantics=AccelerationSemantics.CORRECTION_ONLY,
        maximum_compactness=D("0.1"),
        maximum_speed_fraction_squared=D("0.1"),
    )


def plan_for(contract: SolarSchwarzschild1PNContract):
    return inspect_reference_force_plan(REGISTRY, contract, project_root=ROOT)


def spec(
    step: str,
    count: int,
    *,
    tolerance: str = "1e-40",
    maximum_iterations: int = 100,
) -> ImplicitMidpointSpec:
    return ImplicitMidpointSpec(
        step_size=D(step),
        step_count=count,
        position_atol=D(tolerance),
        velocity_atol=D(tolerance),
        relative_tolerance=D(tolerance),
        maximum_iterations=maximum_iterations,
    )


def initial_state(plan, arithmetic, *, position=None, velocity=None, epoch="0"):
    if velocity is None:
        with localcontext(arithmetic.make_context()):
            velocity = (D(0), D("1.5").sqrt(), D(0))
    return bind_reference_state(
        epoch=D(epoch),
        target="REFERENCE_TEST_PARTICLE",
        position=position or (D("0.8"), D(0), D(0)),
        velocity=velocity,
        plan=plan,
        arithmetic=arithmetic,
    )


def endpoint_distance(left, right) -> Decimal:
    return max(
        abs(a - b)
        for a, b in zip(
            left.position + left.velocity,
            right.position + right.velocity,
        )
    )


class ReferenceForceCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arithmetic = DecimalContextSpec(60)

    def test_exact_newtonian_plus_1pn_fixture_and_ledger(self):
        plan = plan_for(solar_contract("3", "10"))
        state = initial_state(
            plan,
            self.arithmetic,
            position=(D(2), D(0), D(0)),
            velocity=(D(1), D(1), D(0)),
        )
        result = evaluate_reference_rhs(state, plan, self.arithmetic)
        self.assertEqual(result.position_rate, (D(1), D(1), D(0)))
        self.assertEqual(result.contributions[0].acceleration, (D("-0.75"), D(0), D(0)))
        self.assertEqual(result.contributions[1].acceleration, (D("0.06"), D("0.03"), D(0)))
        self.assertEqual(result.velocity_rate, (D("-0.69"), D("0.03"), D(0)))
        self.assertEqual(result.applied_model_ids, REFERENCE_FORCE_MODEL_IDS)
        self.assertEqual(result.contributions[0].model_id, NEWTONIAN_SOLAR_MONOPOLE_MODEL_ID)
        self.assertFalse(result.registry_authorized)
        self.assertEqual(result.evidence_class, "MODEL_OUTPUT")

    def test_plan_is_bound_to_inspected_nonexecutable_registry(self):
        plan = plan_for(solar_contract())
        self.assertEqual(plan.registry_state, "DRAFT_NONEXECUTABLE")
        self.assertFalse(plan.registry_authorized)
        self.assertEqual(plan.model_ids, REFERENCE_FORCE_MODEL_IDS)
        self.assertRegex(plan.registry_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(plan.sha256, r"^[0-9a-f]{64}$")

    def test_duplicate_missing_reordered_and_unknown_models_fail_before_rhs(self):
        plan = plan_for(solar_contract())
        bad_model_sets = (
            (REFERENCE_FORCE_MODEL_IDS[0], REFERENCE_FORCE_MODEL_IDS[0]),
            (REFERENCE_FORCE_MODEL_IDS[0],),
            tuple(reversed(REFERENCE_FORCE_MODEL_IDS)),
            (REFERENCE_FORCE_MODEL_IDS[0], "force.relativity.eih_1pn_gr"),
        )
        for model_ids in bad_model_sets:
            with self.subTest(model_ids=model_ids):
                with self.assertRaises(ReferenceForceCompositionError):
                    replace(plan, model_ids=model_ids)

    def test_state_plan_context_and_time_scale_mismatch_fail(self):
        plan = plan_for(solar_contract())
        state = initial_state(plan, self.arithmetic)
        other_arithmetic = DecimalContextSpec(61)
        with self.assertRaises(ReferenceDynamicsError):
            evaluate_reference_rhs(state, plan, other_arithmetic)
        with self.assertRaises(ReferenceDynamicsError):
            evaluate_reference_rhs(
                replace(state, time_scale=CoordinateTimeScale.TCB_COMPATIBLE),
                plan,
                self.arithmetic,
            )
        with self.assertRaises(ReferenceIntegrationError):
            implicit_midpoint_step(
                replace(
                    state,
                    units=CoherentUnitSystem.KILOMETRE_SECOND,
                    time_scale=CoordinateTimeScale.TCB_COMPATIBLE,
                ),
                plan,
                self.arithmetic,
                spec("0.01", 1),
            )
        with self.assertRaises(ReferenceDynamicsError):
            evaluate_reference_rhs(
                replace(state, force_plan_sha256="0" * 64),
                plan,
                self.arithmetic,
            )

    def test_midpoint_domain_failure_returns_no_candidate(self):
        plan = plan_for(solar_contract())
        state = initial_state(
            plan,
            self.arithmetic,
            position=(D(1), D(0), D(0)),
            velocity=(D(-2), D(0), D(0)),
        )
        original = state
        with self.assertRaises(ReferenceDynamicsDomainError):
            implicit_midpoint_step(state, plan, self.arithmetic, spec("1", 1))
        self.assertIs(state, original)

    def test_declared_domain_threshold_is_inclusive_and_excess_is_rejected(self):
        contract = replace(
            solar_contract(mu="1", c="10"),
            maximum_compactness=D("0.01"),
            maximum_speed_fraction_squared=D("0.01"),
        )
        plan = plan_for(contract)
        boundary = initial_state(
            plan,
            self.arithmetic,
            position=(D(1), D(0), D(0)),
            velocity=(D(0), D(1), D(0)),
        )
        self.assertEqual(evaluate_reference_rhs(boundary, plan, self.arithmetic).applied_model_ids,
                         REFERENCE_FORCE_MODEL_IDS)
        with self.assertRaises(ReferenceDynamicsDomainError):
            evaluate_reference_rhs(
                replace(boundary, position=(D("0.999999"), D(0), D(0))),
                plan,
                self.arithmetic,
            )
        with self.assertRaises(ReferenceDynamicsDomainError):
            evaluate_reference_rhs(
                replace(boundary, velocity=(D(0), D("1.000001"), D(0))),
                plan,
                self.arithmetic,
            )


class ImplicitMidpointMethodTests(unittest.TestCase):
    def setUp(self):
        self.arithmetic = DecimalContextSpec(70)

    @staticmethod
    def solve_generic(initial, h, rhs, *, iterations=100, tolerance="1e-55"):
        with localcontext(DecimalContextSpec(70).make_context()):
            return midpoint_module._solve_implicit_midpoint_six(
                initial_epoch=D(0),
                initial=initial,
                step_size=D(h),
                position_atol=D(tolerance),
                velocity_atol=D(tolerance),
                relative_tolerance=D(tolerance),
                maximum_iterations=iterations,
                rhs=rhs,
            )

    def test_exact_linear_velocity_dependent_midpoint_solution(self):
        beta = D("0.2")
        h = D("0.5")

        def rhs(_epoch, y):
            return (y[3], y[4], y[5], beta * y[3], beta * y[4], beta * y[5])

        initial = (D(1), D(2), D(3), D(4), D(5), D(6))
        result, iterations, calls, residual = self.solve_generic(initial, str(h), rhs)
        with localcontext(self.arithmetic.make_context()):
            amplification = (D(1) + h * beta / 2) / (D(1) - h * beta / 2)
            expected_velocity = tuple(initial[index] * amplification for index in range(3, 6))
            expected_position = tuple(
                initial[index] + h * (initial[index + 3] + expected_velocity[index]) / 2
                for index in range(3)
            )
        self.assertLess(max(abs(result[i] - expected_position[i]) for i in range(3)), D("1e-53"))
        self.assertLess(max(abs(result[i + 3] - expected_velocity[i]) for i in range(3)), D("1e-53"))
        self.assertGreater(iterations, 1)
        self.assertGreaterEqual(calls, 3)
        self.assertLessEqual(residual, 1)

    def test_harmonic_oscillator_has_second_order_convergence(self):
        def rhs(_epoch, y):
            return (y[3], y[4], y[5], -y[0], -y[1], -y[2])

        def integrate(step_text):
            h = D(step_text)
            y = (D(1), D(0), D(0), D(0), D(0), D(0))
            epoch = D(0)
            for _ in range(int(D(1) / h)):
                with localcontext(self.arithmetic.make_context()):
                    y, _, _, _ = midpoint_module._solve_implicit_midpoint_six(
                        initial_epoch=epoch,
                        initial=y,
                        step_size=h,
                        position_atol=D("1e-58"),
                        velocity_atol=D("1e-58"),
                        relative_tolerance=D("1e-56"),
                        maximum_iterations=100,
                        rhs=rhs,
                    )
                    epoch += h
            return y

        with localcontext(self.arithmetic.make_context()):
            sine, cosine = _decimal_sine_cosine_one()
            exact = (cosine, D(0), D(0), -sine, D(0), D(0))
            errors = []
            for step_text in ("0.1", "0.05", "0.025"):
                numerical = integrate(step_text)
                errors.append(max(abs(numerical[i] - exact[i]) for i in range(6)))
        self.assertGreater(errors[0] / errors[1], D("3.8"))
        self.assertLess(errors[0] / errors[1], D("4.2"))
        self.assertGreater(errors[1] / errors[2], D("3.8"))
        self.assertLess(errors[1] / errors[2], D("4.2"))

    def test_nonautonomous_rhs_uses_exact_midpoint_epoch(self):
        def rhs(epoch, _state):
            return (epoch,) * 6

        initial = (D(0),) * 6
        result, _, _, residual = midpoint_module._solve_implicit_midpoint_six(
            initial_epoch=D(2),
            initial=initial,
            step_size=D("0.5"),
            position_atol=D("1e-60"),
            velocity_atol=D("1e-60"),
            relative_tolerance=D("1e-60"),
            maximum_iterations=4,
            rhs=rhs,
        )
        self.assertEqual(result, (D("1.125"),) * 6)
        self.assertEqual(residual, 0)

    def test_forward_negative_step_roundtrip_is_solver_tolerance_limited(self):
        plan = plan_for(solar_contract())
        state = initial_state(plan, self.arithmetic)
        forward = integrate_reference_trajectory(
            state, plan, self.arithmetic, spec("0.05", 20, tolerance="1e-48")
        ).final_state
        backward = integrate_reference_trajectory(
            forward, plan, self.arithmetic, spec("-0.05", 20, tolerance="1e-48")
        ).final_state
        self.assertEqual(backward.epoch, state.epoch)
        self.assertLess(endpoint_distance(state, backward), D("5e-47"))

    def test_forced_nonlinear_nonconvergence_is_fail_closed(self):
        def rhs(_epoch, y):
            return tuple(D(100) * value for value in y)

        initial = (D(1),) * 6
        with self.assertRaises(NonlinearSolveError):
            self.solve_generic(initial, "1", rhs, iterations=2, tolerance="1e-60")
        self.assertEqual(initial, (D(1),) * 6)

    def test_nonzero_step_must_advance_large_epoch_exactly(self):
        arithmetic = DecimalContextSpec(34)
        plan = plan_for(solar_contract())
        state = initial_state(plan, arithmetic, epoch="2451545")
        for step_size in ("1e-28", "-1e-28"):
            with self.subTest(step_size=step_size):
                with self.assertRaisesRegex(
                    ReferenceIntegrationError, "coordinate-time offset"
                ):
                    implicit_midpoint_step(
                        state,
                        plan,
                        arithmetic,
                        spec(step_size, 1, tolerance="1e-30"),
                    )
        self.assertEqual(state.epoch, D("2451545"))

    def test_epoch_offset_is_bounded_decimal_only_and_signed(self):
        with localcontext(DecimalContextSpec(34).make_context()):
            self.assertEqual(
                midpoint_module._exact_epoch_offset(D("2451545"), D("0.125"), 1, 2),
                D("2451545.0625"),
            )
            self.assertEqual(
                midpoint_module._exact_epoch_offset(D("2451545"), D("-0.125"), 1, 1),
                D("2451544.875"),
            )
        source = (ROOT / "src" / "jxplanetx" / "v5_implicit_midpoint.py").read_text()
        imports = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("fractions", imports)
        self.assertNotIn("Fraction", source)

    def test_contract_rejects_implicit_or_binary_numeric_policy(self):
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(33)
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(10**30)
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(50, emin=-(10**30))
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(50, emax=10**30)
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(50, rounding="ROUND_DOWN")
        with self.assertRaises(ReferenceDynamicsError):
            DecimalContextSpec(50, traps=tuple(reversed(REFERENCE_TRAPS)))
        with self.assertRaises(ReferenceIntegrationError):
            spec("0", 1)
        with self.assertRaises(ReferenceIntegrationError):
            ImplicitMidpointSpec(  # type: ignore[arg-type]
                step_size=0.1,
                step_count=1,
                position_atol=D("1e-20"),
                velocity_atol=D("1e-20"),
                relative_tolerance=D("1e-20"),
                maximum_iterations=10,
            )
        with self.assertRaises(ReferenceIntegrationError):
            spec("0.1", 1, tolerance="0")


class SolarReferenceTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arithmetic = DecimalContextSpec(60)
        cls.plan = plan_for(solar_contract())
        cls.initial = initial_state(cls.plan, cls.arithmetic)

    def integrate(self, step_text, total="1", *, plan=None, state=None, tolerance="1e-40"):
        h = D(step_text)
        return integrate_reference_trajectory(
            state or self.initial,
            plan or self.plan,
            self.arithmetic,
            spec(step_text, int(D(total) / abs(h)), tolerance=tolerance),
        )

    def test_short_solar_arc_converges_at_second_order(self):
        endpoints = [
            self.integrate(step_text).final_state
            for step_text in ("0.1", "0.05", "0.025")
        ]
        coarse = endpoint_distance(endpoints[0], endpoints[1])
        fine = endpoint_distance(endpoints[1], endpoints[2])
        ratio = coarse / fine
        self.assertGreater(ratio, D("3.8"))
        self.assertLess(ratio, D("4.3"))

    def test_richardson_endpoint_agrees_with_independently_coded_decimal_rk4(self):
        arithmetic = DecimalContextSpec(70)
        plan = plan_for(solar_contract())
        state = initial_state(plan, arithmetic)

        def independent_rhs(vector):
            position = vector[:3]
            velocity = vector[3:]
            radius_squared = sum(component * component for component in position)
            radius = radius_squared.sqrt()
            speed_squared = sum(component * component for component in velocity)
            radial_velocity = sum(position[i] * velocity[i] for i in range(3))
            mu = D(1)
            c = D(100)
            correction_scale = mu / (c * c * radius_squared * radius)
            radial_coefficient = 4 * mu / radius - speed_squared
            acceleration = tuple(
                -mu * position[i] / (radius_squared * radius)
                + correction_scale
                * (radial_coefficient * position[i] + 4 * radial_velocity * velocity[i])
                for i in range(3)
            )
            return velocity + acceleration

        def independent_rk4(step_text):
            step_size = D(step_text)
            vector = state.position + state.velocity
            for _ in range(int(D(1) / step_size)):
                k1 = independent_rhs(vector)
                k2 = independent_rhs(tuple(vector[i] + step_size * k1[i] / 2 for i in range(6)))
                k3 = independent_rhs(tuple(vector[i] + step_size * k2[i] / 2 for i in range(6)))
                k4 = independent_rhs(tuple(vector[i] + step_size * k3[i] for i in range(6)))
                vector = tuple(
                    vector[i]
                    + step_size * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
                    for i in range(6)
                )
            return vector

        with localcontext(arithmetic.make_context()):
            rk4_coarse = independent_rk4("0.002")
            rk4_fine = independent_rk4("0.001")
            self.assertLess(
                max(abs(a - b) for a, b in zip(rk4_coarse, rk4_fine)),
                D("2e-12"),
            )
            midpoint_coarse = integrate_reference_trajectory(
                state, plan, arithmetic, spec("0.025", 40, tolerance="1e-50")
            ).final_state
            midpoint_fine = integrate_reference_trajectory(
                state, plan, arithmetic, spec("0.0125", 80, tolerance="1e-50")
            ).final_state
            extrapolated = tuple(
                (4 * fine - coarse) / 3
                for coarse, fine in zip(
                    midpoint_coarse.position + midpoint_coarse.velocity,
                    midpoint_fine.position + midpoint_fine.velocity,
                )
            )
            self.assertLess(
                max(abs(a - b) for a, b in zip(extrapolated, rk4_fine)),
                D("2e-7"),
            )

    def test_large_c_trajectory_recovers_independent_newtonian_midpoint(self):
        arithmetic = DecimalContextSpec(70)
        plan = plan_for(solar_contract(c="1e30"))
        state = initial_state(plan, arithmetic)
        integration_spec = spec("0.05", 20, tolerance="1e-52")
        relativistic = integrate_reference_trajectory(
            state, plan, arithmetic, integration_spec
        ).final_state

        mu = D(1)
        vector = state.position + state.velocity
        epoch = D(0)

        def newtonian_rhs(_epoch, y):
            radius2 = sum(y[index] * y[index] for index in range(3))
            radius = radius2.sqrt()
            scale = -mu / (radius2 * radius)
            return y[3:] + tuple(scale * y[index] for index in range(3))

        with localcontext(arithmetic.make_context()):
            for _ in range(20):
                vector, _, _, _ = midpoint_module._solve_implicit_midpoint_six(
                    initial_epoch=epoch,
                    initial=vector,
                    step_size=D("0.05"),
                    position_atol=D("1e-52"),
                    velocity_atol=D("1e-52"),
                    relative_tolerance=D("1e-52"),
                    maximum_iterations=100,
                    rhs=newtonian_rhs,
                )
                epoch += D("0.05")
        self.assertLess(
            max(abs(a - b) for a, b in zip(relativistic.position + relativistic.velocity, vector)),
            D("1e-55"),
        )

    def test_coherent_au_day_and_kilometre_second_trajectories_match(self):
        with localcontext(self.arithmetic.make_context()):
            length = D("149597870.7")
            time = D(86400)
            base_plan = self.plan
            converted_contract = solar_contract(
                mu=str(D(1) * length**3 / time**2),
                c=str(D(100) * length / time),
                units=CoherentUnitSystem.KILOMETRE_SECOND,
            )
            converted_plan = plan_for(converted_contract)
            converted_state = initial_state(
                converted_plan,
                self.arithmetic,
                position=tuple(length * item for item in self.initial.position),  # type: ignore[arg-type]
                velocity=tuple(length / time * item for item in self.initial.velocity),  # type: ignore[arg-type]
            )
        base = integrate_reference_trajectory(
            self.initial,
            base_plan,
            self.arithmetic,
            spec("0.01", 20, tolerance="1e-42"),
        ).final_state
        converted_spec = ImplicitMidpointSpec(
            step_size=D("0.01") * time,
            step_count=20,
            position_atol=D("1e-42") * length,
            velocity_atol=D("1e-42") * length / time,
            relative_tolerance=D("1e-42"),
            maximum_iterations=100,
        )
        converted = integrate_reference_trajectory(
            converted_state,
            converted_plan,
            self.arithmetic,
            converted_spec,
        ).final_state
        with localcontext(self.arithmetic.make_context()):
            converted_back = tuple(item / length for item in converted.position) + tuple(
                item * time / length for item in converted.velocity
            )
        self.assertLess(
            max(abs(a - b) for a, b in zip(base.position + base.velocity, converted_back)),
            D("1e-38"),
        )

    def test_rotation_covariance_for_signed_axis_permutation(self):
        rotate = lambda vector: (vector[1].copy_negate(), vector[0], vector[2])
        rotated_initial = initial_state(
            self.plan,
            self.arithmetic,
            position=rotate(self.initial.position),
            velocity=rotate(self.initial.velocity),
        )
        base = self.integrate("0.02", total="0.2").final_state
        rotated = self.integrate(
            "0.02", total="0.2", state=rotated_initial
        ).final_state
        self.assertLess(
            max(abs(a - b) for a, b in zip(rotate(base.position), rotated.position)),
            D("1e-55"),
        )
        self.assertLess(
            max(abs(a - b) for a, b in zip(rotate(base.velocity), rotated.velocity)),
            D("1e-55"),
        )

    def test_trajectory_perihelion_advance_after_step_richardson_extrapolation(self):
        def first_perihelion_angle(step_text):
            trajectory = self.integrate(step_text, total="7")
            radial_products = [
                sum(state.position[i] * state.velocity[i] for i in range(3))
                for state in trajectory.states
            ]
            crossing = next(
                index
                for index in range(1, len(radial_products))
                if radial_products[index - 1] < 0 <= radial_products[index]
            )
            left = trajectory.states[crossing - 1]
            right = trajectory.states[crossing]
            fraction = radial_products[crossing - 1] / (
                radial_products[crossing - 1] - radial_products[crossing]
            )
            x = left.position[0] + fraction * (right.position[0] - left.position[0])
            y = left.position[1] + fraction * (right.position[1] - left.position[1])
            return math.atan2(float(y), float(x))

        coarse = first_perihelion_angle("0.02")
        fine = first_perihelion_angle("0.01")
        extrapolated = (4.0 * fine - coarse) / 3.0
        analytic = 6.0 * math.pi / (0.8 * 1.2 * 100.0**2)
        self.assertGreater(extrapolated, 0.0)
        self.assertLess(abs(extrapolated - analytic) / analytic, 0.002)
        self.assertLess(abs(fine - analytic), abs(coarse - analytic))

    def test_every_step_remains_nonauthorizing_and_ledger_closed(self):
        trajectory = self.integrate("0.05", total="0.2")
        self.assertFalse(trajectory.registry_authorized)
        self.assertEqual(trajectory.evidence_class, "MODEL_OUTPUT")
        self.assertTrue(all(not state.registry_authorized for state in trajectory.states))
        self.assertTrue(
            all(item.applied_model_ids == REFERENCE_FORCE_MODEL_IDS for item in trajectory.diagnostics)
        )
        self.assertTrue(all(item.scaled_residual <= 1 for item in trajectory.diagnostics))

    def test_trajectory_result_rejects_mixed_spec_or_epoch_chain(self):
        trajectory = self.integrate("0.05", total="0.05")
        bad_spec_diagnostic = replace(
            trajectory.diagnostics[0],
            spec_sha256="0" * 64,
        )
        with self.assertRaises(ReferenceIntegrationError):
            replace(trajectory, diagnostics=(bad_spec_diagnostic,))
        bad_epoch_diagnostic = replace(
            trajectory.diagnostics[0],
            end_epoch=trajectory.diagnostics[0].end_epoch + D("0.01"),
        )
        with self.assertRaises(ReferenceIntegrationError):
            ReferenceTrajectoryResult(
                method_id=trajectory.method_id,
                states=trajectory.states,
                diagnostics=(bad_epoch_diagnostic,),
                force_plan_sha256=trajectory.force_plan_sha256,
                decimal_context_sha256=trajectory.decimal_context_sha256,
                spec_sha256=trajectory.spec_sha256,
            )
        with self.assertRaises(ReferenceIntegrationError):
            replace(trajectory, diagnostics=(object(),))  # type: ignore[arg-type]


class V5IsolationTests(unittest.TestCase):
    def test_legacy_integrator_and_dynamics_bytes_are_unchanged(self):
        expected = {
            ROOT / "src" / "jxplanetx" / "dynamics.py":
                "c83073d78844f5875af3c48dd083be2f81d073441ec0277ab3bcb38d2a7dbf51",
            ROOT / "src" / "jxplanetx" / "yoshida6.py":
                "562c2fff76a8a5e7838029ac94728147d84fb055bac84eadfed18d8a36e840f3",
        }
        for path, digest in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_reference_modules_do_not_import_legacy_integrators_or_dynamics(self):
        forbidden = {"dynamics", "yoshida6", "decimal_bs", "independent_dop853"}
        for filename in ("v5_reference_dynamics.py", "v5_implicit_midpoint.py"):
            source = (ROOT / "src" / "jxplanetx" / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.rsplit(".", 1)[-1])
            self.assertTrue(forbidden.isdisjoint(imported), (filename, imported & forbidden))

    def test_legacy_modules_and_cli_do_not_reverse_import_v5_reference_path(self):
        package = ROOT / "src" / "jxplanetx"
        allowed = {
            "v5_reference_dynamics.py",
            "v5_implicit_midpoint.py",
            "v5_solar_1pn_qualification.py",
        }
        forbidden_names = {"v5_reference_dynamics", "v5_implicit_midpoint"}
        for path in package.glob("*.py"):
            if path.name in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.rsplit(".", 1)[-1])
            self.assertTrue(forbidden_names.isdisjoint(imported), path)


def _decimal_sine_cosine_one():
    """Independent Decimal Taylor oracle at x=1 for method-order tests."""
    x = D(1)
    x2 = x * x
    sine_term = x
    cosine_term = D(1)
    sine = sine_term
    cosine = cosine_term
    epsilon = D(10) ** -65
    for index in range(1, 1000):
        sine_term *= -x2 / D((2 * index) * (2 * index + 1))
        cosine_term *= -x2 / D((2 * index - 1) * (2 * index))
        sine += sine_term
        cosine += cosine_term
        if abs(sine_term) < epsilon and abs(cosine_term) < epsilon:
            return sine, cosine
    raise AssertionError("Decimal trigonometric oracle did not converge")


if __name__ == "__main__":
    unittest.main()
