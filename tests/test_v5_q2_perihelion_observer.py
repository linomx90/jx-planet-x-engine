"""Disclosed toy/development tests for the nonauthorizing V5 Q2 observer."""

from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from decimal import ROUND_DOWN, Decimal, FloatOperation, getcontext, localcontext
from pathlib import Path

import observer.v5_solar_1pn as observer_package
import observer.v5_solar_1pn.candidate as candidate_module
from observer.v5_solar_1pn.candidate import (
    ANGLE_METHOD_ID,
    DENSE_STATE_METHOD_ID,
    EXECUTION_AUTHORIZED,
    OBSERVER_ID,
    OUTPUT_CLASS,
    QUALIFICATION_OUTCOMES_GENERATED,
    REGISTRY_AUTHORIZED,
    REVIEW_STATUS,
    ROOT_ISOLATION_METHOD_ID,
    ROOT_METHOD_ID,
    CartesianState,
    DecimalContextSpec,
    ForceMode,
    InitialInput,
    ObserverArithmeticError,
    ObserverConfiguration,
    ObserverContractError,
    ObserverDomainError,
    ObserverMetadata,
    StateSample,
    TrajectoryObservation,
    TrajectorySeries,
    cubic_hermite_state,
    decimal_atan2,
    decimal_pi,
    initial_input_sha256,
    observe_trajectory,
    observer_configuration_sha256,
    oriented_position_angle,
    pair_perihelion_observations,
    radial_product,
    sampling_schedule_sha256,
    trajectory_series_sha256,
    unwrap_oriented_angles,
)
from observer.v5_solar_1pn.candidate import __all__ as CANDIDATE_EXPORTS


D = Decimal
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "observer/v5_solar_1pn/candidate.py"
ZERO = D(0)
ONE = D(1)


def state(position, velocity) -> CartesianState:
    return CartesianState(tuple(D(str(x)) for x in position), tuple(D(str(x)) for x in velocity))


INITIAL_STATE = state((1, 0, 0), (0, 1, 0))


def initial_input(*, initial_state: CartesianState = INITIAL_STATE, target: str = "TOY_TARGET"):
    return InitialInput(
        input_id="development.q2.toy.initial.v1",
        initial_epoch=D("0"),
        initial_state=initial_state,
        length_unit_id="unit.au",
        time_unit_id="unit.day",
        velocity_unit_id="unit.au_per_day",
        frame_id="BCRS_DERIVED_SUN_RELATIVE_ICRS_ALIGNED_RESTRICTED",
        orientation_id="ICRS_ALIGNED_RESTRICTED",
        time_scale_id="TDB_COMPATIBLE",
        origin_id="SUN",
        target_id=target,
    )


def configuration(*, width: str = "1e-30", off_plane: str = "0"):
    return ObserverConfiguration(
        configuration_id="development.q2.observer.toy.v1",
        root_bracket_width_tolerance=D(width),
        maximum_relative_off_plane=D(off_plane),
    )


def exact_event_series(
    mode: ForceMode,
    first_y: str,
    second_y: str,
    *,
    schedule_pair_id: str = "development.q2.schedule.toy.v1",
    rotate=None,
) -> TrajectorySeries:
    def s(position, velocity):
        result = state(position, velocity)
        if rotate is None:
            return result
        return CartesianState(rotate(result.position), rotate(result.velocity))

    a1 = D(first_y)
    a2 = D(second_y)
    # Piecewise cubic development curve with alternating radial extrema.
    # Each half-segment checkpoint is derived from the same Hermite polynomial,
    # so the observer can certify its complete dense root roster.
    keys = (
        StateSample(D("0"), s((1, 0, 0), (0, "0.1", 0))),
        StateSample(D("1"), s((2, 0, 0), (0, 0, 0))),
        StateSample(D("2"), s((1, a1, 0), (0, 0, 0))),
        StateSample(D("3"), s((2, 2 * a1, 0), (0, 0, 0))),
        StateSample(D("4"), s((1, a2, 0), (0, 0, 0))),
        StateSample(D("5"), s((2, 2 * a2, 0), (0, 0, 0))),
    )
    toy_config = configuration()
    retained = []
    for index in range(len(keys) - 1):
        if not retained:
            retained.append(keys[index])
        midpoint = (keys[index].epoch + keys[index + 1].epoch) / 2
        retained.append(
            StateSample(
                midpoint,
                cubic_hermite_state(keys[index], keys[index + 1], midpoint, toy_config),
            )
        )
        if index < len(keys) - 2:
            retained.append(keys[index + 1])
    samples = tuple(retained)
    bound_initial = initial_input(initial_state=samples[0].state)
    return TrajectorySeries(
        trajectory_id=f"development.q2.{mode.name.lower()}.toy.v1",
        schedule_pair_id=schedule_pair_id,
        force_mode=mode,
        source_configuration_id="development.source.common_numeric_spec.v1",
        source_configuration_sha256="1" * 64,
        initial_input=bound_initial,
        samples=samples,
    )


def nonexact_event_series() -> TrajectorySeries:
    base = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
    samples_by_epoch = {item.epoch: item for item in base.samples}

    def between(left_epoch: str, right_epoch: str, epoch: str) -> StateSample:
        left = samples_by_epoch[D(left_epoch)]
        right = samples_by_epoch[D(right_epoch)]
        value = D(epoch)
        return StateSample(value, cubic_hermite_state(left, right, value, configuration()))

    samples = (
        samples_by_epoch[D("0")],
        samples_by_epoch[D("0.5")],
        samples_by_epoch[D("1")],
        between("1", "1.5", "1.2"),
        between("2", "2.5", "2.1"),
        samples_by_epoch[D("3")],
        between("3", "3.5", "3.2"),
        between("4", "4.5", "4.1"),
        samples_by_epoch[D("4.5")],
    )
    return TrajectorySeries(
        trajectory_id="development.q2.nonexact.toy.v1",
        schedule_pair_id="development.q2.schedule.nonexact.v1",
        force_mode=ForceMode.NEWTONIAN,
        source_configuration_id="development.source.nonexact.v1",
        source_configuration_sha256="3" * 64,
        initial_input=base.initial_input,
        samples=samples,
    )


class Q2ObserverContractTests(unittest.TestCase):
    def test_source_is_isolated_decimal_only_and_has_no_io_or_cli(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        from_imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any(name.startswith("jxplanetx") for name in imports | from_imports))
        self.assertFalse(any(name.startswith("oracle") for name in imports | from_imports))
        self.assertNotIn("math", imports | from_imports)
        self.assertNotIn("numpy", imports | from_imports)
        self.assertNotIn("scipy", imports | from_imports)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue({"open", "print", "input"}.isdisjoint(called_names))
        self.assertNotIn("argparse", source)
        self.assertNotIn("__main__", source)
        test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                isinstance(node, ast.Constant) and type(node.value) is float
                for parsed in (tree, test_tree)
                for node in ast.walk(parsed)
            )
        )
        self.assertEqual(tuple(inspect.signature(observe_trajectory).parameters), ("series", "configuration"))

    def test_method_and_nonauthorization_constants_are_closed(self):
        self.assertEqual(OUTPUT_CLASS, "MODEL_OUTPUT")
        self.assertEqual(REVIEW_STATUS, "CANDIDATE_PENDING_INDEPENDENT_REVIEW")
        self.assertFalse(REGISTRY_AUTHORIZED)
        self.assertFalse(EXECUTION_AUTHORIZED)
        self.assertFalse(QUALIFICATION_OUTCOMES_GENERATED)
        self.assertIn("q2.perihelion_observer_candidate", OBSERVER_ID)
        self.assertIn("cubic-hermite", DENSE_STATE_METHOD_ID)
        self.assertIn("bisection", ROOT_METHOD_ID)
        self.assertIn("bernstein-degree5", ROOT_ISOLATION_METHOD_ID)
        self.assertIn("atan2", ANGLE_METHOD_ID)
        with self.assertRaises(ObserverContractError):
            ObserverMetadata(registry_authorized=True)

    def test_package_root_exports_exact_candidate_api_only(self):
        self.assertEqual(observer_package.__all__, CANDIDATE_EXPORTS)
        self.assertNotIn("candidate", observer_package.__all__)

    def test_context_and_coordinate_contract_are_exact(self):
        context = DecimalContextSpec()
        built = context.build()
        self.assertEqual(built.prec, 90)
        self.assertTrue(built.traps[FloatOperation])
        with self.assertRaises(ObserverContractError):
            DecimalContextSpec(precision=89)
        with self.assertRaises(ObserverContractError):
            configuration(width="1e-25")
        base = initial_input()
        for field_name, bad in (
            ("length_unit_id", "unit.km"),
            ("time_unit_id", "unit.second"),
            ("velocity_unit_id", "unit.km_per_second"),
            ("frame_id", "OTHER_FRAME"),
            ("orientation_id", "OTHER_ORIENTATION"),
            ("time_scale_id", "TCB_COMPATIBLE"),
            ("origin_id", "BARYCENTER"),
        ):
            with self.subTest(field=field_name), self.assertRaises(ObserverContractError):
                replace(base, **{field_name: bad})

    def test_binary_float_bool_nonfinite_and_mutable_inputs_fail_closed(self):
        with self.assertRaises(ObserverContractError):
            CartesianState((float("1"), ZERO, ZERO), (ZERO, ONE, ZERO))
        with self.assertRaises(ObserverContractError):
            CartesianState([ONE, ZERO, ZERO], (ZERO, ONE, ZERO))
        with self.assertRaises(ObserverContractError):
            StateSample(True, INITIAL_STATE)
        with self.assertRaises(ObserverDomainError):
            StateSample(D("NaN"), INITIAL_STATE)
        with self.assertRaises(FrozenInstanceError):
            configuration().expected_event_count = 3

    def test_domain_separated_hashes_bind_every_identity(self):
        series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        config = configuration()
        for digest in (
            initial_input_sha256(series.initial_input),
            sampling_schedule_sha256(series),
            trajectory_series_sha256(series),
            observer_configuration_sha256(config),
        ):
            self.assertEqual(len(digest), 64)
        self.assertNotEqual(
            trajectory_series_sha256(series),
            trajectory_series_sha256(replace(series, trajectory_id="development.q2.changed.v1")),
        )
        changed_epoch = replace(series.samples[1], epoch=D("0.11"))
        changed = replace(series, samples=(series.samples[0], changed_epoch, *series.samples[2:]))
        self.assertNotEqual(sampling_schedule_sha256(series), sampling_schedule_sha256(changed))


class Q2HermiteAndEventTests(unittest.TestCase):
    def test_cubic_hermite_velocity_is_derivative_of_position_polynomial(self):
        config = configuration()
        left = StateSample(D("0"), state((0, 1, 0), (0, 0, 0)))
        right = StateSample(D("1"), state((1, 1, 0), (3, 0, 0)))
        observed = cubic_hermite_state(left, right, D("0.25"), config)
        self.assertEqual(observed.position, (D("0.015625"), ONE, ZERO))
        self.assertEqual(observed.velocity, (D("0.1875"), ZERO, ZERO))
        self.assertEqual(cubic_hermite_state(left, right, left.epoch, config), left.state)
        self.assertEqual(cubic_hermite_state(left, right, right.epoch, config), right.state)

    def test_initial_event_is_excluded_and_exact_endpoint_is_emitted_once(self):
        observed = observe_trajectory(
            exact_event_series(ForceMode.NEWTONIAN, "0", "0"), configuration()
        )
        self.assertEqual(tuple(item.orbit_index for item in observed.events), (1, 2))
        self.assertEqual(tuple(item.epoch for item in observed.events), (D("2.0"), D("4.0")))
        self.assertTrue(all(item.exact_root for item in observed.events))
        self.assertTrue(all(item.source_left_value < 0 for item in observed.events))
        self.assertTrue(all(item.direction_confirmation_value > 0 for item in observed.events))
        self.assertNotIn(D("0"), tuple(item.epoch for item in observed.events))

    def test_nonexact_events_use_width_terminated_bisection(self):
        for width in ("1e-20", "1e-30"):
            with self.subTest(width=width):
                config = configuration(width=width)
                observed = observe_trajectory(nonexact_event_series(), config)
                self.assertTrue(all(not item.exact_root for item in observed.events))
                self.assertTrue(all(item.bisection_iterations > 0 for item in observed.events))
                with localcontext(config.context.build()):
                    widths = tuple(
                        item.bracket_right_epoch - item.bracket_left_epoch
                        for item in observed.events
                    )
                self.assertTrue(
                    all(value <= config.root_bracket_width_tolerance for value in widths)
                )
                self.assertTrue(
                    all(
                        item.bracket_left_value < 0 < item.bracket_right_value
                        for item in observed.events
                    )
                )

    def test_initial_arming_rejects_near_initial_inward_artifact(self):
        series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        bad_second = replace(series.samples[1], state=state((1, 0, 0), (-1, 0, 0)))
        bad = replace(series, samples=(series.samples[0], bad_second, *series.samples[2:]))
        with self.assertRaises(ObserverDomainError):
            observe_trajectory(bad, configuration())

    def test_initial_state_must_be_exact_event(self):
        series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        changed_state = state((1, 0, 0), (D("0.1"), 1, 0))
        changed_initial = initial_input(initial_state=changed_state)
        bad = replace(
            series,
            initial_input=changed_initial,
            samples=(StateSample(D("0"), changed_state), *series.samples[1:]),
        )
        with self.assertRaises(ObserverDomainError):
            observe_trajectory(bad, configuration())

    def test_reversed_crossings_and_extra_perihelion_fail_closed(self):
        series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        truncated = replace(series, samples=series.samples[:-3])
        with self.assertRaises(ObserverDomainError):
            observe_trajectory(truncated, configuration())
        config = configuration()
        apo = StateSample(D("5"), state((2, 0, 0), (0, 0, 0)))
        peri = StateSample(D("6"), state((1, 0, 0), (0, 0, 0)))
        next_apo = StateSample(D("7"), state((2, 0, 0), (0, 0, 0)))
        inward = StateSample(D("5.5"), cubic_hermite_state(apo, peri, D("5.5"), config))
        outward = StateSample(
            D("6.5"), cubic_hermite_state(peri, next_apo, D("6.5"), config)
        )
        extra = series.samples + (apo, inward, peri, outward)
        with self.assertRaises(ObserverDomainError):
            observe_trajectory(replace(series, samples=extra), configuration())

    def test_radial_product_ignores_hostile_ambient_context(self):
        config = configuration()
        sample = state((D("1.234567890123"), 2, 3), (4, 5, 6))
        expected = radial_product(sample, config)
        with localcontext() as ambient:
            ambient.prec = 6
            ambient.rounding = ROUND_DOWN
            actual = radial_product(sample, config)
        self.assertEqual(actual, expected)

    def test_hidden_same_sign_dense_roots_are_rejected(self):
        config = configuration()
        left = StateSample(D("0"), state((1, 0, 0), (10, 0, 0)))
        right = StateSample(D("0.1"), state((1, 0, 0), (10, 0, 0)))
        midpoint = cubic_hermite_state(left, right, D("0.05"), config)
        self.assertGreater(radial_product(left.state, configuration()), 0)
        self.assertGreater(radial_product(right.state, configuration()), 0)
        self.assertLess(radial_product(midpoint, configuration()), 0)
        coefficients = candidate_module._bernstein_radial_coefficients(left, right, config)
        signs = tuple(1 if value > 0 else -1 if value < 0 else 0 for value in coefficients)
        self.assertEqual((signs[0], signs[-1]), (1, 1))
        self.assertIn(-1, signs)
        samples = (
            left,
            right,
            StateSample(D("0.2"), state((2, 0, 0), (10, 0, 0))),
            StateSample(D("0.3"), state((3, 0, 0), (10, 0, 0))),
            StateSample(D("0.4"), state((4, 0, 0), (10, 0, 0))),
        )
        series = TrajectorySeries(
            "development.q2.hidden_roots.v1",
            "development.q2.hidden_roots.schedule.v1",
            ForceMode.NEWTONIAN,
            "development.source.common_numeric_spec.v1",
            "6" * 64,
            initial_input(initial_state=left.state),
            samples,
        )
        with self.assertRaisesRegex(
            ObserverDomainError, "same-sign Hermite interval is not certified root-free"
        ):
            observe_trajectory(series, config)

    def test_sign_changing_multiple_root_interval_is_rejected(self):
        config = configuration()
        # The owned Hermite is exactly x=(u-0.5)^2, hence
        # g=x*dx/du=2*(u-0.5)^3 has one root of multiplicity three.
        left = StateSample(D("0"), state(("0.25", 1, 0), (-1, 0, 0)))
        right = StateSample(D("1"), state(("0.25", 1, 0), (1, 0, 0)))
        self.assertLess(radial_product(left.state, configuration()), 0)
        self.assertGreater(radial_product(right.state, configuration()), 0)
        coefficients = candidate_module._bernstein_radial_coefficients(
            left, right, config
        )
        signs = tuple(1 if value > 0 else -1 if value < 0 else 0 for value in coefficients)
        self.assertEqual(signs, (-1, 1, 1, -1, -1, 1))
        self.assertEqual(
            sum(signs[index] != signs[index - 1] for index in range(1, len(signs))),
            3,
        )
        samples = (
            left,
            right,
            StateSample(D("2"), right.state),
            StateSample(D("3"), right.state),
            StateSample(D("4"), right.state),
        )
        series = TrajectorySeries(
            "development.q2.multiple_roots.v1",
            "development.q2.multiple_roots.schedule.v1",
            ForceMode.NEWTONIAN,
            "development.source.common_numeric_spec.v1",
            "7" * 64,
            initial_input(initial_state=left.state),
            samples,
        )
        with self.assertRaisesRegex(
            ObserverDomainError,
            "sign-changing Hermite interval is not certified to one simple root",
        ):
            observe_trajectory(series, config)

    def test_plateau_and_checkpoint_tangent_roots_are_rejected(self):
        series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        repeated_initial = StateSample(D("0.25"), series.samples[0].state)
        plateau = (series.samples[0], repeated_initial, *series.samples[1:])
        with self.assertRaises(ObserverDomainError):
            observe_trajectory(replace(series, samples=plateau), configuration())

        tangent_samples = (
            StateSample(D("0"), state((1, 0, 0), (0, 0, 0))),
            StateSample(D("0.1"), state(("1.01", 0, 0), ("0.2", 0, 0))),
            StateSample(D("0.2"), state(("1.02", 0, 0), (0, 0, 0))),
            StateSample(D("0.3"), state(("1.03", 0, 0), ("0.2", 0, 0))),
            StateSample(D("0.4"), state(("1.06", 0, 0), ("0.4", 0, 0))),
        )
        tangent_series = TrajectorySeries(
            "development.q2.checkpoint_tangent.v1",
            "development.q2.checkpoint_tangent.schedule.v1",
            ForceMode.NEWTONIAN,
            "development.source.common_numeric_spec.v1",
            "8" * 64,
            initial_input(initial_state=tangent_samples[0].state),
            tangent_samples,
        )
        tangent_values = tuple(
            radial_product(item.state, configuration()) for item in tangent_samples
        )
        self.assertGreater(tangent_values[1], ZERO)
        self.assertEqual(tangent_values[2], ZERO)
        self.assertGreater(tangent_values[3], ZERO)
        with self.assertRaisesRegex(
            ObserverDomainError,
            "interior endpoint zero is a plateau, tangent, or unresolved root",
        ):
            observe_trajectory(tangent_series, configuration())

    def test_p90_endpoint_cancellation_cannot_disagree_with_exact_certificate(self):
        config = configuration()
        with localcontext() as exact:
            exact.prec = 220
            magnitude = D("1e89")
            cancellation_state = CartesianState(
                (magnitude, magnitude - 1, ZERO),
                (magnitude, -(magnitude + 1), ZERO),
            )
        self.assertEqual(radial_product(cancellation_state, config), ZERO)
        samples = (
            StateSample(D("0"), cancellation_state),
            StateSample(D("1"), state((1, 0, 0), (1, 1, 0))),
            StateSample(D("2"), state((1, 0, 0), (-1, 0, 0))),
            StateSample(D("3"), state((1, 0, 0), (0, 1, 0))),
            StateSample(D("4"), state((1, 0, 0), (1, 1, 0))),
        )
        bound = initial_input(initial_state=cancellation_state)
        series = TrajectorySeries(
            "development.q2.cancellation.v1",
            "development.q2.cancellation.schedule.v1",
            ForceMode.NEWTONIAN,
            "development.source.common_numeric_spec.v1",
            "4" * 64,
            bound,
            samples,
        )
        with self.assertRaisesRegex(
            ObserverArithmeticError,
            "p90 endpoint sign disagrees with the exact root certificate",
        ):
            observe_trajectory(series, config)

    def test_exact_certificate_and_p90_midpoint_sign_mismatch_is_rejected(self):
        config = configuration()
        with localcontext() as exact:
            exact.prec = 220
            base = D("1e100")
            scale = D("1e12")
            duration = D("0.04")
            offset = D("0.51")

            def hostile_state(unit):
                displacement = scale * (D(unit) - offset)
                speed = scale / duration
                return CartesianState(
                    (base + displacement, base - displacement, ZERO),
                    (speed, -speed, ZERO),
                )

            left = StateSample(D("0"), hostile_state("0"))
            right = StateSample(duration, hostile_state("1"))
        fillers = tuple(
            StateSample(D(str(index)), right.state) for index in (1, 2, 3)
        )
        bound = initial_input(initial_state=left.state)
        series = TrajectorySeries(
            "development.q2.midpoint_cancellation.v1",
            "development.q2.midpoint_cancellation.schedule.v1",
            ForceMode.NEWTONIAN,
            "development.source.common_numeric_spec.v1",
            "5" * 64,
            bound,
            (left, right, *fillers),
        )
        with localcontext(config.context.build()):
            with self.assertRaisesRegex(
                ObserverArithmeticError,
                "p90 midpoint sign disagrees with the exact root certificate",
            ):
                candidate_module._refine_root(series, 0, 1, 1, config)


class Q2AngleTests(unittest.TestCase):
    def test_pi_and_all_atan2_axes_and_quadrants(self):
        config = configuration()
        pi = decimal_pi(config)
        reference = D(
            "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803483"
        )
        self.assertEqual(pi, reference)
        self.assertEqual(decimal_atan2(ZERO, ONE, config), ZERO)
        self.assertEqual(decimal_atan2(ZERO, -ONE, config), pi)
        with localcontext(config.context.build()):
            half = pi / 2
            quarter = pi / 4
        self.assertLess(abs(decimal_atan2(ONE, ZERO, config) - half), D("1e-88"))
        self.assertLess(abs(decimal_atan2(-ONE, ZERO, config) + half), D("1e-88"))
        with localcontext(config.context.build()):
            targets = (quarter, 3 * quarter, -(3 * quarter), -quarter)
        for actual, target in zip(
            (
                decimal_atan2(ONE, ONE, config),
                decimal_atan2(ONE, -ONE, config),
                decimal_atan2(-ONE, -ONE, config),
                decimal_atan2(-ONE, ONE, config),
            ),
            targets,
        ):
            self.assertLess(abs(actual - target), D("1e-88"))

    def test_atan2_handles_extreme_dynamic_range_without_overflow_or_underflow(self):
        config = configuration()
        pi = decimal_pi(config)
        huge = D("1e999999")
        tiny = D("1e-999999")
        with localcontext(config.context.build()):
            half = pi / 2
        self.assertLess(abs(decimal_atan2(huge, tiny, config) - half), D("1e-88"))
        self.assertEqual(decimal_atan2(tiny, huge, config), ZERO)

    def test_branch_boundary_unwraps_and_exact_pi_tie_fails(self):
        config = configuration()
        pi = decimal_pi(config)
        epsilon = D("1e-40")
        with localcontext(config.context.build()):
            result = unwrap_oriented_angles((pi - epsilon, -pi + epsilon), config)
        self.assertEqual(result[1] - result[0], 2 * epsilon)
        with self.assertRaises(ObserverDomainError):
            unwrap_oriented_angles((ZERO, pi), config)

    def test_oriented_3d_basis_is_prograde_and_rotation_covariant(self):
        config = configuration()
        event = (ZERO, ONE, ZERO)
        base = oriented_position_angle(event, INITIAL_STATE, config)
        rotate = lambda vector: (-vector[1], vector[0], vector[2])
        rotated_initial = CartesianState(rotate(INITIAL_STATE.position), rotate(INITIAL_STATE.velocity))
        rotated = oriented_position_angle(rotate(event), rotated_initial, config)
        with localcontext(config.context.build()):
            half = decimal_pi(config) / 2
        self.assertLess(abs(base - half), D("1e-88"))
        self.assertEqual(rotated, base)

    def test_material_off_plane_position_is_rejected(self):
        with self.assertRaises(ObserverDomainError):
            oriented_position_angle((ONE, ZERO, D("1e-5")), INITIAL_STATE, configuration())

    def test_angle_results_ignore_hostile_ambient_context(self):
        config = configuration()
        expected = decimal_atan2(D("0.123456789"), D("-7.5"), config)
        global_context = getcontext()
        old_precision = global_context.prec
        old_rounding = global_context.rounding
        try:
            global_context.prec = 5
            global_context.rounding = ROUND_DOWN
            actual = decimal_atan2(D("0.123456789"), D("-7.5"), config)
        finally:
            global_context.prec = old_precision
            global_context.rounding = old_rounding
        self.assertEqual(actual, expected)


class Q2ResultClosureAndPairingTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.newtonian_series = exact_event_series(ForceMode.NEWTONIAN, "0", "0")
        self.one_pn_series = exact_event_series(
            ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, "0.01", "0.02"
        )
        self.newtonian = observe_trajectory(self.newtonian_series, self.config)
        self.one_pn = observe_trajectory(self.one_pn_series, self.config)

    def test_observation_is_fully_bound_and_nonauthorizing(self):
        observed = self.one_pn
        self.assertEqual(observed.trajectory_sha256, trajectory_series_sha256(observed.series))
        self.assertEqual(observed.sampling_schedule_sha256, sampling_schedule_sha256(observed.series))
        self.assertEqual(observed.initial_input_sha256, initial_input_sha256(observed.series.initial_input))
        self.assertEqual(
            observed.observer_configuration_sha256,
            observer_configuration_sha256(observed.observer_configuration),
        )
        self.assertEqual(observed.metadata.output_class, "MODEL_OUTPUT")
        self.assertFalse(observed.metadata.registry_authorized)
        self.assertFalse(observed.metadata.execution_authorized)
        self.assertFalse(observed.metadata.qualification_outcomes_generated)
        with localcontext(self.config.context.build()):
            direct = observed.events[1].unwrapped_angle - observed.events[0].unwrapped_angle
        self.assertEqual(observed.secular_advance_per_orbit, direct)

    def test_pair_subtracts_newtonian_drift_and_retains_prograde_sign(self):
        paired = pair_perihelion_observations(self.newtonian, self.one_pn)
        with localcontext(self.config.context.build()):
            direct = (
                self.one_pn.secular_advance_per_orbit
                - self.newtonian.secular_advance_per_orbit
            )
        self.assertEqual(paired.paired_advance_per_orbit, direct)
        self.assertTrue(paired.prograde)
        self.assertEqual(paired.sampling_schedule_sha256, self.newtonian.sampling_schedule_sha256)
        with self.assertRaises(ObserverContractError):
            pair_perihelion_observations(self.one_pn, self.newtonian)

    def test_null_or_retrograde_pair_is_retained_not_censored(self):
        null_one_pn = observe_trajectory(
            exact_event_series(ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, "0", "0"),
            self.config,
        )
        paired = pair_perihelion_observations(self.newtonian, null_one_pn)
        self.assertEqual(paired.paired_advance_per_orbit, ZERO)
        self.assertFalse(paired.prograde)

        retrograde_one_pn = observe_trajectory(
            exact_event_series(ForceMode.NEWTONIAN_PLUS_SOLAR_1PN, "0.02", "0.01"),
            self.config,
        )
        retrograde = pair_perihelion_observations(self.newtonian, retrograde_one_pn)
        self.assertLess(retrograde.paired_advance_per_orbit, ZERO)
        self.assertFalse(retrograde.prograde)

    def test_mismatched_schedule_and_initial_input_are_rejected(self):
        changed_schedule_series = replace(
            self.one_pn_series,
            schedule_pair_id="development.q2.schedule.other.v1",
        )
        changed_schedule = observe_trajectory(changed_schedule_series, self.config)
        with self.assertRaises(ObserverContractError):
            pair_perihelion_observations(self.newtonian, changed_schedule)
        changed_target_input = replace(self.one_pn_series.initial_input, target_id="OTHER_TARGET")
        changed_target_series = replace(self.one_pn_series, initial_input=changed_target_input)
        changed_target = observe_trajectory(changed_target_series, self.config)
        with self.assertRaises(ObserverContractError):
            pair_perihelion_observations(self.newtonian, changed_target)

    def test_mismatched_source_and_decimal_encoded_observer_configs_are_rejected(self):
        changed_source_series = replace(
            self.one_pn_series,
            source_configuration_sha256="9" * 64,
        )
        changed_source = observe_trajectory(changed_source_series, self.config)
        with self.assertRaises(ObserverContractError):
            pair_perihelion_observations(self.newtonian, changed_source)

        representation_variant = configuration(off_plane="0.0")
        self.assertEqual(representation_variant, self.config)
        self.assertNotEqual(
            observer_configuration_sha256(representation_variant),
            observer_configuration_sha256(self.config),
        )
        changed_observer = observe_trajectory(self.one_pn_series, representation_variant)
        with self.assertRaises(ObserverContractError):
            pair_perihelion_observations(self.newtonian, changed_observer)

    def test_observation_rejects_mutated_hash_event_state_value_angle_and_slope(self):
        observed = self.one_pn
        mutations = (
            {"trajectory_sha256": "0" * 64},
            {"sampling_schedule_sha256": "0" * 64},
            {"initial_input_sha256": "0" * 64},
            {"observer_configuration_sha256": "0" * 64},
            {"secular_advance_per_orbit": observed.secular_advance_per_orbit + D("1e-30")},
        )
        for mutation in mutations:
            with self.subTest(mutation=tuple(mutation)), self.assertRaises(ObserverContractError):
                replace(observed, **mutation)
        first = observed.events[0]
        event_mutants = (
            replace(first, event_value=first.event_value + ONE),
            replace(first, principal_angle=first.principal_angle + D("1e-20")),
            replace(first, unwrapped_angle=first.unwrapped_angle + D("1e-20")),
            replace(first, state=state((2, 0, 0), (0, 1, 0))),
            replace(first, bracket_left_epoch=first.bracket_left_epoch - D("1e-20")),
        )
        for mutant in event_mutants:
            with self.subTest(field=mutant), self.assertRaises(ObserverContractError):
                replace(observed, events=(mutant, observed.events[1]))

    def test_rotation_of_complete_series_preserves_angles_and_slope(self):
        rotate = lambda vector: (-vector[1], vector[0], vector[2])
        rotated = observe_trajectory(
            exact_event_series(
                ForceMode.NEWTONIAN_PLUS_SOLAR_1PN,
                "0.01",
                "0.02",
                rotate=rotate,
            ),
            self.config,
        )
        self.assertEqual(
            tuple(item.principal_angle for item in rotated.events),
            tuple(item.principal_angle for item in self.one_pn.events),
        )
        self.assertEqual(rotated.secular_advance_per_orbit, self.one_pn.secular_advance_per_orbit)

    def test_tilted_proper_3d_rotation_preserves_angles_and_slope(self):
        # Exact determinant-positive rotation from unit quaternion (1,2,2,4)/5.
        def rotate(vector):
            x, y, z = vector
            return (
                -D("0.6") * x + D("0.8") * z,
                D("0.64") * x - D("0.6") * y + D("0.48") * z,
                D("0.48") * x + D("0.8") * y + D("0.36") * z,
            )

        rotated_series = exact_event_series(
            ForceMode.NEWTONIAN_PLUS_SOLAR_1PN,
            "0.01",
            "0.02",
            rotate=rotate,
        )
        self.assertNotEqual(rotated_series.samples[0].state.position[2], ZERO)
        self.assertNotEqual(rotated_series.samples[0].state.velocity[2], ZERO)
        rotated = observe_trajectory(rotated_series, self.config)
        self.assertEqual(
            tuple(item.principal_angle for item in rotated.events),
            tuple(item.principal_angle for item in self.one_pn.events),
        )
        self.assertEqual(
            rotated.secular_advance_per_orbit,
            self.one_pn.secular_advance_per_orbit,
        )

    def test_uniform_state_scaling_preserves_observable_but_changes_identity(self):
        factor = D("10")
        scaled_samples = tuple(
            StateSample(
                item.epoch,
                CartesianState(
                    tuple(factor * value for value in item.state.position),
                    tuple(factor * value for value in item.state.velocity),
                ),
            )
            for item in self.one_pn_series.samples
        )
        scaled_input = replace(
            self.one_pn_series.initial_input,
            initial_state=scaled_samples[0].state,
        )
        scaled_series = replace(
            self.one_pn_series,
            trajectory_id="development.q2.newtonian_plus_solar_1pn.scaled.toy.v1",
            initial_input=scaled_input,
            samples=scaled_samples,
        )
        scaled = observe_trajectory(scaled_series, self.config)
        self.assertEqual(
            tuple(item.epoch for item in scaled.events),
            tuple(item.epoch for item in self.one_pn.events),
        )
        self.assertEqual(
            tuple(item.principal_angle for item in scaled.events),
            tuple(item.principal_angle for item in self.one_pn.events),
        )
        self.assertEqual(
            scaled.secular_advance_per_orbit,
            self.one_pn.secular_advance_per_orbit,
        )
        self.assertNotEqual(scaled.initial_input_sha256, self.one_pn.initial_input_sha256)
        self.assertNotEqual(scaled.trajectory_sha256, self.one_pn.trajectory_sha256)


if __name__ == "__main__":
    unittest.main()
