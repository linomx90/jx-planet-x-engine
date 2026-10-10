from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
import unittest
from unittest.mock import patch

import numpy as np

import jxplanetx.cuda_tracer_events as event_module
from jxplanetx.cuda_tracer_events import (
    DOP853EventLocatorSpec,
    EVENT_PRIMARY_BOUNDARY_OUTWARD,
    EVENT_SOURCE_RADIUS_CONTACT,
    LOCATION_DENSE_MINIMUM,
    TracerEventCandidatePolicy,
    TracerEventContractError,
    TracerEventIntegrationError,
    cubic_hermite_source_state,
    locate_tracer_step_events,
)
from jxplanetx.cuda_tracer_events import _locate_surface_roots


TINY_GM = 1.0e-30


def _vector(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def _stationary_sources(*positions: tuple[float, float, float]):
    source_position = np.array(positions, dtype=np.float64)
    source_velocity = np.zeros_like(source_position)
    source_gm = np.full(len(positions), TINY_GM, dtype=np.float64)
    return (
        source_position,
        source_velocity,
        source_position.copy(),
        source_velocity.copy(),
        source_gm,
    )


def _assert_bitwise_equal(
    test: unittest.TestCase,
    left: object,
    right: object,
) -> None:
    test.assertIs(type(left), type(right))
    if isinstance(left, np.ndarray):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertEqual(left.tobytes(order="C"), right.tobytes(order="C"))
    elif isinstance(left, float):
        test.assertEqual(np.float64(left).tobytes(), np.float64(right).tobytes())
    elif is_dataclass(left):
        for field in fields(left):
            _assert_bitwise_equal(
                test,
                getattr(left, field.name),
                getattr(right, field.name),
            )
    elif isinstance(left, tuple):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_bitwise_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


def _route(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    positions: tuple[tuple[float, float, float], ...],
    radii: tuple[float, ...],
    *,
    step: float = 1.0,
    boundary: float | None = None,
    primary: int = 0,
    candidate_policy: TracerEventCandidatePolicy | None = None,
    spec: DOP853EventLocatorSpec | None = None,
):
    p0, v0, p1, v1, gm = _stationary_sources(*positions)
    return locate_tracer_step_events(
        tracer_position,
        tracer_velocity,
        p0,
        v0,
        p1,
        v1,
        gm,
        np.array(radii, dtype=np.float64),
        step_years=step,
        primary_source_index=primary,
        primary_boundary_radius_AU=boundary,
        candidate_policy=candidate_policy,
        spec=spec,
    )


class HermiteSourceTests(unittest.TestCase):
    def test_endpoint_states_and_signed_step_velocity_are_exact(self):
        p0 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        v0 = np.array([[0.5, -0.25, 1.0]], dtype=np.float64)
        p1 = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        v1 = np.array([[-1.0, 2.0, 0.25]], dtype=np.float64)
        for step in (2.0, -2.0):
            start_p, start_v = cubic_hermite_source_state(p0, v0, p1, v1, 0.0, step)
            end_p, end_v = cubic_hermite_source_state(p0, v0, p1, v1, 1.0, step)
            np.testing.assert_array_equal(start_p, p0)
            np.testing.assert_array_equal(start_v, v0)
            np.testing.assert_array_equal(end_p, p1)
            np.testing.assert_array_equal(end_v, v1)


class DensePolynomialExhaustivenessTests(unittest.TestCase):
    def test_two_tangencies_inside_one_legacy_scan_bin_are_both_found(self):
        # Degree-four relative motion is within the DOP853 dense degree bound.
        # The two tangencies are only 0.005 apart, well inside 1/32, and have a
        # genuine above-surface maximum between them.  Exhausting stationary
        # roots, rather than assuming a bin is unimodal, must recover both.
        from scipy.optimize import brentq, minimize_scalar

        first = 0.505
        second = 0.510
        scale = 1_000.0

        class DenseOutput:
            def __call__(self, local_time: float) -> np.ndarray:
                coordinate = 1.0 + scale * (
                    (local_time - first) * (local_time - second)
                ) ** 2
                return np.array(
                    [coordinate, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )

        zero = np.zeros(3, dtype=np.float64)
        roots = _locate_surface_roots(
            DenseOutput(),
            zero,
            zero,
            zero,
            zero,
            step=1.0,
            fraction_0=0.0,
            fraction_1=1.0,
            threshold=1.0,
            spec=DOP853EventLocatorSpec(
                event_distance_tolerance_AU=1.0e-10
            ),
            brentq=brentq,
            minimize_scalar=minimize_scalar,
        )
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[0].fraction, first, places=8)
        self.assertAlmostEqual(roots[1].fraction, second, places=8)
        self.assertTrue(
            all(root.method == LOCATION_DENSE_MINIMUM for root in roots)
        )

    def test_large_scale_tangent_uses_scale_aware_surface_tolerance(self):
        from numpy.polynomial import Chebyshev
        from scipy.optimize import brentq, minimize_scalar

        coefficients = (
            (
                426413.6495765046, -172027.2385584681, 41889.134759519366,
                22558.755398047244, -9524.637099305597, 24120.06753861088,
                -4480.048945362631, 4860.776508057739,
            ),
            (
                459352.42556156317, -23657.16397398243, 66548.26133740526,
                -13441.722830897052, -525.7405069793039, 1712.3518946570894,
                8851.480593595272, 12917.4851837537,
            ),
            (
                -440625.3191406737, -199850.74255739411, 42235.799977572286,
                -75796.85888213094, 1049.2237580599417, 9205.432502351783,
                -20710.855241849105, -7619.881139246548,
            ),
        )
        polynomials = tuple(
            Chebyshev(np.asarray(item) * 0.1, domain=(0.0, 1.0))
            for item in coefficients
        )

        class DenseOutput:
            def __call__(self, local_time: float) -> np.ndarray:
                position = [item(local_time) for item in polynomials]
                velocity = [item.deriv()(local_time) for item in polynomials]
                return np.asarray(position + velocity, dtype=np.float64)

        zero = np.zeros(3, dtype=np.float64)
        dense = DenseOutput()
        threshold = 69080.12730951629
        roots = _locate_surface_roots(
            dense,
            zero,
            zero,
            zero,
            zero,
            step=1.0,
            fraction_0=0.0,
            fraction_1=1.0,
            threshold=threshold,
            spec=DOP853EventLocatorSpec(),
            brentq=brentq,
            minimize_scalar=minimize_scalar,
        )
        self.assertGreaterEqual(len(roots), 1)
        for root in roots:
            distance = float(np.linalg.norm(dense(root.fraction)[:3]))
            self.assertLessEqual(
                abs(distance - threshold), root.effective_tolerance_AU
            )


class ContinuousEventLocationTests(unittest.TestCase):
    def test_filtered_event_free_result_is_bitwise_identical_to_default(self):
        positions = tuple(
            (float(index * 10), float((index % 3) * 7), 0.0)
            for index in range(11)
        )
        radii = tuple(0.01 for _ in positions)
        exhaustive = _route(
            _vector(3.0, 4.0),
            _vector(0.1, -0.05),
            positions,
            radii,
            boundary=100_000.0,
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=0.25),
        )
        filtered = _route(
            _vector(3.0, 4.0),
            _vector(0.1, -0.05),
            positions,
            radii,
            boundary=100_000.0,
            candidate_policy=TracerEventCandidatePolicy(
                source_radius_indices=(),
                include_primary_boundary=False,
            ),
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=0.25),
        )

        _assert_bitwise_equal(self, exhaustive, filtered)

    def test_outer_only_event_is_bitwise_identical_to_exhaustive(self):
        arguments = (
            _vector(0.5),
            _vector(1.0),
            ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)),
            (0.01, 1.0),
        )
        exhaustive = _route(*arguments, boundary=1.0)
        outer_only = _route(
            *arguments,
            boundary=1.0,
            candidate_policy=TracerEventCandidatePolicy(
                source_radius_indices=(),
                include_primary_boundary=True,
            ),
        )

        self.assertEqual(
            outer_only.earliest_event.event_type,
            EVENT_PRIMARY_BOUNDARY_OUTWARD,
        )
        _assert_bitwise_equal(self, exhaustive, outer_only)

    def test_policy_does_not_narrow_initial_surface_or_inside_checks(self):
        no_continuous_surfaces = TracerEventCandidatePolicy(
            source_radius_indices=(),
            include_primary_boundary=False,
        )
        contact = _route(
            _vector(1.0),
            _vector(0.0, 1.0),
            ((0.0, 0.0, 0.0),),
            (1.0,),
            candidate_policy=no_continuous_surfaces,
        )
        self.assertEqual(contact.completed_fraction, 0.0)
        self.assertEqual(
            contact.earliest_event.event_type,
            EVENT_SOURCE_RADIUS_CONTACT,
        )
        with self.assertRaisesRegex(TracerEventIntegrationError, "already inside"):
            _route(
                _vector(0.5),
                _vector(0.0, 1.0),
                ((0.0, 0.0, 0.0),),
                (1.0,),
                candidate_policy=no_continuous_surfaces,
            )

    def test_selected_surface_retains_grazing_stationary_point_search(self):
        result = _route(
            _vector(-2.0, 1.0),
            _vector(4.0, 0.0),
            ((-100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.0, 1.0),
            candidate_policy=TracerEventCandidatePolicy(
                source_radius_indices=(1,),
                include_primary_boundary=False,
            ),
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=1.0),
        )

        self.assertEqual(result.earliest_event.source_index, 1)
        self.assertEqual(
            result.earliest_event.location_method,
            LOCATION_DENSE_MINIMUM,
        )
        self.assertAlmostEqual(result.earliest_event.step_fraction, 0.5, places=10)

    def test_primary_dense_analysis_is_reused_within_each_accepted_interval(self):
        with patch.object(
            event_module,
            "_dense_distance_critical_fractions",
            wraps=event_module._dense_distance_critical_fractions,
        ) as critical_fractions:
            result = _route(
                _vector(2.0, 3.0),
                _vector(0.2, 0.0),
                ((0.0, 0.0, 0.0),),
                (0.1,),
                boundary=100.0,
                spec=DOP853EventLocatorSpec(maximum_substep_fraction=0.25),
            )

        self.assertFalse(result.terminated_at_event)
        self.assertEqual(
            critical_fractions.call_count,
            result.accepted_step_count,
        )

    def test_event_free_route_releases_only_full_step_endpoint_and_diagnostics(self):
        result = _route(
            _vector(2.0, 3.0),
            _vector(0.2, 0.0),
            ((0.0, 0.0, 0.0),),
            (0.1,),
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=0.25),
        )
        self.assertFalse(result.terminated_at_event)
        self.assertIsNone(result.earliest_event)
        self.assertIsNotNone(result.endpoint_position_AU)
        self.assertEqual(result.completed_fraction, 1.0)
        self.assertGreaterEqual(result.accepted_step_count, 4)
        self.assertGreater(result.derivative_evaluation_count, result.accepted_step_count)
        self.assertGreater(result.minimum_evaluated_stage_distance_AU, 0.0)
        self.assertGreaterEqual(
            result.maximum_accepted_primary_dynamical_phase_radians, 0.0
        )
        self.assertFalse(result.endpoint_position_AU.flags.writeable)
        np.testing.assert_allclose(result.endpoint_position_AU, _vector(2.2, 3.0), atol=1e-12)

    def test_source_entry_is_located_between_dop853_endpoints(self):
        # The primary is far away.  The second source's unit sphere is crossed
        # along y=0.5 at x=-sqrt(3)/2, at the analytic fraction below.
        result = _route(
            _vector(-2.0, 0.5),
            _vector(4.0, 0.0),
            ((-100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.0, 1.0),
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=1.0),
        )
        event = result.earliest_event
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EVENT_SOURCE_RADIUS_CONTACT)
        self.assertEqual(event.source_index, 1)
        expected = (2.0 - math.sqrt(0.75)) / 4.0
        self.assertAlmostEqual(event.step_fraction, expected, places=10)
        self.assertAlmostEqual(event.surface_distance_AU, 1.0, places=11)
        self.assertLessEqual(abs(event.surface_residual_AU), 1.0e-11)
        self.assertIsNone(result.endpoint_position_AU)
        self.assertFalse(event.position_AU.flags.writeable)

    def test_grazing_contact_without_sign_change_uses_dense_minimization(self):
        result = _route(
            _vector(-2.0, 1.0),
            _vector(4.0, 0.0),
            ((-100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.0, 1.0),
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=1.0),
        )
        event = result.earliest_event
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EVENT_SOURCE_RADIUS_CONTACT)
        self.assertEqual(event.location_method, LOCATION_DENSE_MINIMUM)
        self.assertAlmostEqual(event.step_fraction, 0.5, places=10)
        self.assertAlmostEqual(event.surface_distance_AU, 1.0, places=11)
        self.assertAlmostEqual(event.surface_radial_velocity_AU_per_year, 0.0, places=10)

    def test_contact_uses_continuously_moving_hermite_source(self):
        p0 = np.array([[-100.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        p1 = np.array([[-100.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        velocity = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        result = locate_tracer_step_events(
            _vector(-1.0),
            _vector(4.0),
            p0,
            velocity,
            p1,
            velocity.copy(),
            np.array([TINY_GM, TINY_GM], dtype=np.float64),
            np.array([0.0, 0.5], dtype=np.float64),
            step_years=1.0,
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=1.0),
        )
        event = result.earliest_event
        self.assertEqual(event.source_index, 1)
        self.assertAlmostEqual(event.step_fraction, 0.5, places=10)
        self.assertAlmostEqual(event.surface_distance_AU, 0.5, places=11)
        self.assertAlmostEqual(event.surface_radial_velocity_AU_per_year, -3.0, places=10)

    def test_outward_primary_boundary_reports_complete_event_state(self):
        result = _route(
            _vector(0.5),
            _vector(1.0),
            ((0.0, 0.0, 0.0),),
            (0.0,),
            boundary=1.0,
        )
        event = result.earliest_event
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EVENT_PRIMARY_BOUNDARY_OUTWARD)
        self.assertEqual(event.source_index, 0)
        self.assertAlmostEqual(event.step_fraction, 0.5, places=11)
        self.assertAlmostEqual(event.surface_threshold_AU, 1.0)
        self.assertAlmostEqual(event.primary_radial_velocity_AU_per_year, 1.0, places=11)
        self.assertAlmostEqual(event.primary_specific_energy_AU2_per_year2, 0.5, places=11)
        np.testing.assert_allclose(result.terminal_position_AU, _vector(1.0), atol=1e-11)

    def test_large_outer_boundary_uses_scale_aware_residual_tolerance(self):
        direction = np.random.default_rng(1).normal(size=3)
        direction /= np.linalg.norm(direction)
        zero = np.zeros((1, 3), dtype=np.float64)
        result = locate_tracer_step_events(
            np.ascontiguousarray(direction * 99_999.5, dtype=np.float64),
            np.ascontiguousarray(direction, dtype=np.float64),
            zero,
            zero.copy(),
            zero.copy(),
            zero.copy(),
            np.array([TINY_GM], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            step_years=1.0,
            primary_boundary_radius_AU=100_000.0,
            spec=DOP853EventLocatorSpec(maximum_substep_fraction=1.0),
        )
        event = result.earliest_event
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EVENT_PRIMARY_BOUNDARY_OUTWARD)
        self.assertLessEqual(
            abs(event.surface_residual_AU), event.surface_tolerance_AU
        )
        self.assertGreater(event.surface_tolerance_AU, 1.0e-11)

    def test_inward_boundary_crossing_is_not_an_outward_event(self):
        result = _route(
            _vector(1.5),
            _vector(-1.0),
            ((0.0, 0.0, 0.0),),
            (0.0,),
            boundary=1.0,
        )
        self.assertFalse(result.terminated_at_event)
        np.testing.assert_allclose(result.endpoint_position_AU, _vector(0.5), atol=1e-11)

    def test_negative_step_locates_physical_outward_crossing(self):
        # Reverse-time integration traverses the surface from outside to
        # inside, but the physical radial velocity is still outward (+1).
        result = _route(
            _vector(1.5),
            _vector(1.0),
            ((0.0, 0.0, 0.0),),
            (0.0,),
            boundary=1.0,
            step=-1.0,
        )
        event = result.earliest_event
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EVENT_PRIMARY_BOUNDARY_OUTWARD)
        self.assertAlmostEqual(event.step_fraction, 0.5, places=11)
        self.assertAlmostEqual(event.local_time_years, -0.5, places=11)
        self.assertGreater(event.primary_radial_velocity_AU_per_year, 0.0)

    def test_simultaneous_source_contacts_have_stable_source_index_order(self):
        result = _route(
            _vector(-2.0, 0.5),
            _vector(4.0, 0.0),
            (
                (-100.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (0.0, 1.0, 1.0),
        )
        self.assertEqual(len(result.events), 2)
        self.assertEqual(tuple(event.source_index for event in result.events), (1, 2))
        self.assertEqual(
            tuple(event.event_type for event in result.events),
            (EVENT_SOURCE_RADIUS_CONTACT, EVENT_SOURCE_RADIUS_CONTACT),
        )
        self.assertAlmostEqual(
            result.events[0].step_fraction,
            result.events[1].step_fraction,
            places=14,
        )

    def test_earliest_surface_wins_even_when_its_source_index_is_later(self):
        result = _route(
            _vector(-3.0, 0.5),
            _vector(5.0, 0.0),
            (
                (-100.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
            ),
            (0.0, 0.75, 0.75),
        )
        self.assertEqual(result.earliest_event.source_index, 2)
        self.assertLess(result.earliest_event.step_fraction, 0.5)


class EventContractAndFailureTests(unittest.TestCase):
    def test_initial_surface_is_reported_and_initial_interior_fails_closed(self):
        contact = _route(
            _vector(1.0),
            _vector(0.0, 1.0),
            ((0.0, 0.0, 0.0),),
            (1.0,),
        )
        self.assertTrue(contact.terminated_at_event)
        self.assertEqual(contact.completed_fraction, 0.0)
        self.assertEqual(contact.accepted_step_count, 0)
        with self.assertRaisesRegex(
            TracerEventIntegrationError, "already inside"
        ):
            _route(
                _vector(0.5),
                _vector(0.0, 1.0),
                ((0.0, 0.0, 0.0),),
                (1.0,),
            )

    def test_initial_contact_precedes_any_dop853_trial_derivative(self):
        zero = np.zeros((1, 3), dtype=np.float64)
        result = locate_tracer_step_events(
            _vector(1.0e-6),
            _vector(0.0, 1.0),
            zero,
            zero.copy(),
            zero.copy(),
            zero.copy(),
            np.array([1.0e300], dtype=np.float64),
            np.array([1.0e-6], dtype=np.float64),
            step_years=1.0,
        )
        self.assertEqual(result.earliest_event.event_type, EVENT_SOURCE_RADIUS_CONTACT)
        self.assertEqual(result.completed_fraction, 0.0)
        self.assertEqual(result.accepted_step_count, 0)
        self.assertEqual(result.derivative_evaluation_count, 0)

    def test_point_mass_singularity_and_nonfinite_inputs_fail_closed(self):
        with self.assertRaisesRegex(TracerEventIntegrationError, "singularity"):
            _route(
                _vector(0.0),
                _vector(1.0),
                ((0.0, 0.0, 0.0),),
                (0.0,),
            )
        p0, v0, p1, v1, gm = _stationary_sources((0.0, 0.0, 0.0))
        bad = _vector(1.0)
        bad[1] = math.nan
        with self.assertRaises(TracerEventContractError):
            locate_tracer_step_events(
                bad,
                _vector(0.0, 1.0),
                p0,
                v0,
                p1,
                v1,
                gm,
                np.array([0.0], dtype=np.float64),
                step_years=1.0,
            )

    def test_contract_rejects_ambiguous_types_and_invalid_policy(self):
        with self.assertRaises(TracerEventContractError):
            DOP853EventLocatorSpec(dense_polynomial_validation_points=4)
        with self.assertRaises(TracerEventContractError):
            TracerEventCandidatePolicy(
                source_radius_indices=[0]  # type: ignore[arg-type]
            )
        with self.assertRaises(TracerEventContractError):
            TracerEventCandidatePolicy(source_radius_indices=(1, 1))
        p0, v0, p1, v1, gm = _stationary_sources((0.0, 0.0, 0.0))
        with self.assertRaisesRegex(
            TracerEventContractError, "outside the source table"
        ):
            locate_tracer_step_events(
                _vector(1.0),
                _vector(0.0, 1.0),
                p0,
                v0,
                p1,
                v1,
                gm,
                np.array([0.0], dtype=np.float64),
                step_years=1.0,
                candidate_policy=TracerEventCandidatePolicy(
                    source_radius_indices=(1,)
                ),
            )
        with self.assertRaisesRegex(
            TracerEventContractError, "candidate_policy"
        ):
            locate_tracer_step_events(
                _vector(1.0),
                _vector(0.0, 1.0),
                p0,
                v0,
                p1,
                v1,
                gm,
                np.array([0.0], dtype=np.float64),
                step_years=1.0,
                candidate_policy=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(TracerEventContractError):
            locate_tracer_step_events(
                [1.0, 0.0, 0.0],  # type: ignore[arg-type]
                _vector(0.0, 1.0),
                p0,
                v0,
                p1,
                v1,
                gm,
                np.array([0.0], dtype=np.float64),
                step_years=1.0,
            )
        with self.assertRaises(TracerEventContractError):
            locate_tracer_step_events(
                _vector(1.0),
                _vector(0.0, 1.0),
                p0,
                v0,
                p1,
                v1,
                gm,
                np.array([0.0], dtype=np.float64),
                step_years=0.0,
            )

    def test_accepted_step_resource_ceiling_fails_closed(self):
        with self.assertRaisesRegex(
            TracerEventIntegrationError, "resource ceiling"
        ):
            _route(
                _vector(2.0, 3.0),
                _vector(0.2, 0.0),
                ((0.0, 0.0, 0.0),),
                (0.0,),
                spec=DOP853EventLocatorSpec(
                    maximum_substep_fraction=0.1,
                    maximum_accepted_steps=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
