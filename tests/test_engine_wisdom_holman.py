import dataclasses
import hashlib
import math
import unittest
from unittest import mock

import numpy as np
import jxplanetx.engine.wisdom_holman as wh_runtime

from jxplanetx.engine.contracts import (
    BackendSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
from jxplanetx.engine.trajectory import (
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryStepLimitError,
    integrate_trajectory,
)
from jxplanetx.engine.trajectory_contracts import AdaptiveRKF78Spec

from jxplanetx.engine.wisdom_holman import (
    JacobiCoordinateBinding,
    WisdomHolmanTrajectoryResult,
    _build_jacobi_coordinate_binding,
    _cartesian_to_jacobi,
    _domain_separated_json_sha256,
    _interaction_force,
    _jacobi_to_cartesian,
    _result_content_sha256,
    _run_from_result,
    _schedule_content_sha256,
    _solve_elliptic_universal_kepler,
    _universal_g_values,
    integrate_wisdom_holman_trajectory,
)
from jxplanetx.engine.wisdom_holman_contracts import (
    FIXED_STEP_WISDOM_HOLMAN_METHOD_ID,
    FixedStepWisdomHolmanSpec,
    UniversalKeplerSolverSpec,
    WH_JACOBI_BINDING_CHECKSUM_DOMAIN,
    WH_KEPLER_SERIES_TERM_COUNT,
    WH_RESULT_CONTENT_CHECKSUM_DOMAIN,
    WH_SCHEDULE_CHECKSUM_DOMAIN,
)


def wh_spec(**changes: object) -> FixedStepWisdomHolmanSpec:
    values: dict[str, object] = {
        "checkpoint_step_indices": (0, 16, 32),
        "fixed_step": 2.0 * math.pi / 64.0,
        "maximum_steps": 32,
        "jacobi_body_order": ("STAR", "INNER", "OUTER"),
        "minimum_encounter_pair_separation": 0.01,
        "minimum_jacobi_periapse": 0.1,
        "maximum_initial_barycenter_position_norm": 1.0e-12,
        "maximum_initial_barycenter_velocity_norm": 1.0e-12,
        "kepler_solver": UniversalKeplerSolverSpec(),
    }
    values.update(changes)
    return FixedStepWisdomHolmanSpec(**values)  # type: ignore[arg-type]


def runtime_provenance() -> Provenance:
    return Provenance(
        "fixture.wh",
        "Synthetic Wisdom--Holman fixture",
        "1",
        "e" * 64,
    )


def runtime_backend() -> BackendSpec:
    return BackendSpec("numpy", "cpu", 2)


def runtime_metadata() -> ParameterMetadata:
    return ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        runtime_provenance(),
        None,
        None,
        -1.0e6,
        1.0e6,
    )


def weak_hierarchy_state(**changes: object) -> StateSnapshot:
    rows = (
        (
            "0x1.56f40f71a182dp-9",
            "-0x1.3a5c50c2d6a40p-11",
            "0x0p+0",
            "0x1.3d399c60c981cp-11",
            "0x1.2b6efeed14806p-11",
            "0x0p+0",
        ),
        (
            "0x1.ad1b090b71629p-1",
            "0x1.d2c3e5372a878p-2",
            "0x0p+0",
            "-0x1.f55cc646a3929p-2",
            "0x1.db8934f99b592p-1",
            "0x0p+0",
        ),
        (
            "-0x1.ba311957d4125p+0",
            "0x1.2670d10ac1618p-4",
            "0x0p+0",
            "-0x1.d8defc9997fb1p-5",
            "-0x1.7ff9ccf690af4p-1",
            "0x0p+0",
        ),
    )
    cartesian = np.array(
        [[float.fromhex(value) for value in row] for row in rows],
        dtype=np.float64,
    )
    values: dict[str, object] = {
        "snapshot_id": "fixture.wh.weak-hierarchy",
        "epoch": 0.0,
        "time_scale": "TDB",
        "frame": "BARYCENTRIC_INERTIAL",
        "origin": "BARYCENTER",
        "axes": "ICRS_ALIGNED",
        "length_unit": "L",
        "time_unit": "T",
        "mass_unit": "M",
        "unit_system_id": "fixture.wh.units",
        "body_ids": ("STAR", "INNER", "OUTER"),
        "positions": cartesian[:, :3].copy(),
        "velocities": cartesian[:, 3:].copy(),
        "gravitational_parameters": np.array((1.0, 0.001, 0.002)),
        "masses": np.array((1.0, 0.001, 0.002)),
        "radii": np.zeros(3, dtype=np.float64),
        "massive": np.ones(3, dtype=np.bool_),
        "provenance": runtime_provenance(),
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def circular_planet_state(**changes: object) -> StateSnapshot:
    gm = np.array((1.0, 0.001), dtype=np.float64)
    total = float(np.sum(gm))
    speed = math.sqrt(total)
    values: dict[str, object] = {
        "snapshot_id": "fixture.wh.circular-planet",
        "epoch": 0.0,
        "time_scale": "SYNTHETIC",
        "frame": "BARYCENTRIC_INERTIAL",
        "origin": "BARYCENTER",
        "axes": "CARTESIAN_RIGHT_HANDED",
        "length_unit": "L",
        "time_unit": "T",
        "mass_unit": "M",
        "unit_system_id": "fixture.wh.units",
        "body_ids": ("STAR", "PLANET"),
        "positions": np.array(
            ((-gm[1] / total, 0.0, 0.0), (gm[0] / total, 0.0, 0.0)),
            dtype=np.float64,
        ),
        "velocities": np.array(
            (
                (0.0, -gm[1] * speed / total, 0.0),
                (0.0, gm[0] * speed / total, 0.0),
            ),
            dtype=np.float64,
        ),
        "gravitational_parameters": gm,
        "masses": gm.copy(),
        "radii": np.zeros(2, dtype=np.float64),
        "massive": np.ones(2, dtype=np.bool_),
        "provenance": runtime_provenance(),
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def runtime_plan(state: StateSnapshot) -> ForcePlan:
    return ForcePlan(
        "fixture.wh.plan",
        runtime_backend(),
        (
            NewtonianPointMass(
                state.body_ids,
                state.body_ids,
                state.unit_system_id,
                (runtime_metadata(),),
            ),
        ),
    )


def runtime_spec(
    state: StateSnapshot,
    *,
    final_step: int,
    fixed_step: float,
    checkpoints: tuple[int, ...] | None = None,
    **changes: object,
) -> FixedStepWisdomHolmanSpec:
    values: dict[str, object] = {
        "checkpoint_step_indices": checkpoints or (0, final_step),
        "fixed_step": fixed_step,
        "maximum_steps": final_step,
        "jacobi_body_order": state.body_ids,
        "minimum_encounter_pair_separation": 0.01,
        "minimum_jacobi_periapse": 0.05,
        "maximum_initial_barycenter_position_norm": 1.0e-12,
        "maximum_initial_barycenter_velocity_norm": 1.0e-12,
    }
    values.update(changes)
    return FixedStepWisdomHolmanSpec(**values)  # type: ignore[arg-type]


class WisdomHolmanContractTests(unittest.TestCase):
    def test_spec_is_closed_derived_and_exact_typed(self):
        spec = wh_spec()
        self.assertEqual(
            spec.method_id, "integrator.symplectic.wisdom_holman_jacobi_kdk_2"
        )
        self.assertEqual(spec.method_id, FIXED_STEP_WISDOM_HOLMAN_METHOD_ID)
        self.assertEqual(spec.primary_body_id, "STAR")
        self.assertEqual(spec.direction, "FORWARD")
        self.assertEqual(spec.completed_steps, 32)
        self.assertEqual(spec.barycenter_roundoff_factor, 256)
        self.assertEqual(spec.encounter_roundoff_factor, 256)
        self.assertEqual(
            spec.public_execution_accounting_scope,
            "PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY",
        )
        self.assertEqual(spec.validation_replay_count, 1)
        self.assertFalse(spec.floating_point_symplectic)
        self.assertFalse(spec.floating_point_exactly_reversible)
        self.assertFalse(spec.registry_authorized)
        self.assertFalse(spec.qualification_authorized)

        class FloatSubclass(float):
            pass

        class TupleSubclass(tuple):
            pass

        cases = (
            {"checkpoint_step_indices": [0, 1]},
            {"checkpoint_step_indices": (1, 2)},
            {"checkpoint_step_indices": (0, 1, 1)},
            {"checkpoint_step_indices": (0, True)},
            {"fixed_step": 0.0},
            {"fixed_step": 1},
            {"maximum_steps": 31},
            {"jacobi_body_order": ("STAR",)},
            {"jacobi_body_order": ("STAR", "STAR")},
            {"jacobi_body_order": TupleSubclass(("STAR", "INNER"))},
            {"minimum_encounter_pair_separation": 0.0},
            {"minimum_jacobi_periapse": -1.0},
            {"maximum_initial_barycenter_position_norm": 1},
            {"kepler_solver": object()},
            {
                "interaction_kick_coefficients": (
                    FloatSubclass(0.5),
                    0.5,
                )
            },
            {"kepler_drift_coefficients": (FloatSubclass(1.0),)},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                wh_spec(**changes)
        with self.assertRaises(ValueError):
            dataclasses.replace(spec, adaptive=True)
        with self.assertRaises(ValueError):
            dataclasses.replace(spec, validation_replay_count=0)

    def test_solver_spec_has_fixed_nonloosable_numerical_policy(self):
        spec = UniversalKeplerSolverSpec()
        self.assertEqual(spec.maximum_iterations, 96)
        self.assertEqual(spec.maximum_bracket_expansions, 32)
        self.assertEqual(spec.series_term_count, 64)
        self.assertEqual(spec.series_switch_abs_argument, 0.5)
        self.assertIn("R(s)=", spec.convergence_policy)
        self.assertIn("G_n", spec.g_function_policy)
        self.assertIn("q1=", spec.postcondition_policy)
        for changes in (
            {"maximum_iterations": 97},
            {"maximum_iterations": True},
            {"series_term_count": 63},
            {"series_switch_abs_argument": 0.5000000000000001},
            {"residual_ulp_factor": 9},
            {"root_policy": "NEWTON"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(spec, **changes)


class JacobiCoordinateTests(unittest.TestCase):
    def test_binding_retains_exact_analytic_coefficients_and_owned_buffers(self):
        mu = np.array((2.0, 0.01, 0.02), dtype=np.float64)
        binding = _build_jacobi_coordinate_binding(
            ("STAR", "INNER", "OUTER"), mu
        )
        self.assertIs(type(binding), JacobiCoordinateBinding)
        self.assertEqual(binding.primary_body_id, "STAR")
        expected_cumulative = np.array((2.0, 2.01, 2.03), dtype=np.float64)
        expected_inertias = np.array(
            (
                2.03,
                0.01 * 2.0 / 2.01,
                0.02 * 2.01 / 2.03,
            ),
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            binding.cumulative_gravitational_parameters, expected_cumulative
        )
        np.testing.assert_array_equal(binding.canonical_inertias, expected_inertias)

        expected_b = np.array(
            (
                (2.0 / 2.03, 0.01 / 2.03, 0.02 / 2.03),
                (-1.0, 1.0, 0.0),
                (-2.0 / 2.01, -0.01 / 2.01, 1.0),
            ),
            dtype=np.float64,
        )
        expected_a = np.array(
            (
                (1.0, -0.01 / 2.01, -0.02 / 2.03),
                (1.0, 2.0 / 2.01, -0.02 / 2.03),
                (1.0, 0.0, 2.01 / 2.03),
            ),
            dtype=np.float64,
        )
        np.testing.assert_array_equal(binding.jacobi_from_cartesian_matrix, expected_b)
        np.testing.assert_array_equal(binding.cartesian_from_jacobi_matrix, expected_a)
        np.testing.assert_allclose(expected_b @ expected_a, np.eye(3), atol=3e-16)

        arrays = (
            binding.gravitational_parameters,
            binding.cumulative_gravitational_parameters,
            binding.canonical_inertias,
            binding.jacobi_from_cartesian_matrix,
            binding.cartesian_from_jacobi_matrix,
        )
        for array in arrays:
            self.assertIs(type(array), np.ndarray)
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
        for left in range(len(arrays) - 1):
            for right in range(left + 1, len(arrays)):
                self.assertFalse(np.shares_memory(arrays[left], arrays[right]))
        mu[0] = 9.0
        self.assertEqual(binding.gravitational_parameters[0], 2.0)

    def test_forward_inverse_recurrences_preserve_state_com_and_momentum(self):
        mu = np.array((1.0, 0.003, 0.007), dtype=np.float64)
        binding = _build_jacobi_coordinate_binding(("S", "I", "O"), mu)
        positions = np.array(
            ((-0.02, 0.01, 0.0), (0.9, 0.2, 0.0), (-1.4, 0.7, 0.1)),
            dtype=np.float64,
        )
        velocities = np.array(
            ((0.001, -0.002, 0.0), (-0.3, 1.0, 0.01), (-0.2, -0.7, 0.02)),
            dtype=np.float64,
        )
        coordinates, momenta = _cartesian_to_jacobi(
            binding, positions, velocities
        )
        recovered_positions, recovered_velocities = _jacobi_to_cartesian(
            binding, coordinates, momenta
        )
        np.testing.assert_allclose(recovered_positions, positions, rtol=0.0, atol=3e-16)
        np.testing.assert_allclose(recovered_velocities, velocities, rtol=0.0, atol=3e-16)

        total = float(np.sum(mu))
        expected_com = np.sum(mu[:, None] * positions, axis=0) / total
        expected_momentum = np.sum(mu[:, None] * velocities, axis=0)
        np.testing.assert_allclose(coordinates[0], expected_com, atol=3e-18)
        np.testing.assert_allclose(momenta[0], expected_momentum, atol=3e-18)
        matrix_coordinates = binding.jacobi_from_cartesian_matrix @ positions
        cartesian_momenta = mu[:, None] * velocities
        matrix_momenta = binding.cartesian_from_jacobi_matrix.T @ cartesian_momenta
        np.testing.assert_allclose(coordinates, matrix_coordinates, atol=3e-16)
        np.testing.assert_allclose(momenta, matrix_momenta, atol=3e-18)

    def test_binding_checksum_has_independent_literal_known_answer(self):
        binding = _build_jacobi_coordinate_binding(
            ("A", "B"), np.array((1.0, 1.0), dtype=np.float64)
        )
        literal_payload = (
            b'{"body_ids":["A","B"],"canonical_inertias_hex":["0x1.0000000000000p+1","0x1.0000000000000p-1"],'
            b'"canonical_momentum_convention":"CARTESIAN_CANONICAL_MOMENTUM_I_EQUALS_GM_I_TIMES_VELOCITY_I",'
            b'"canonical_weight_source":"STATE_GRAVITATIONAL_PARAMETERS","cartesian_from_jacobi_matrix_hex":'
            b'[["0x1.0000000000000p+0","-0x1.0000000000000p-1"],["0x1.0000000000000p+0","0x1.0000000000000p-1"]],'
            b'"coordinate_system":"ORDERED_JACOBI_BARYCENTRIC_V1","cumulative_gravitational_parameters_hex":'
            b'["0x1.0000000000000p+0","0x1.0000000000000p+1"],"gravitational_parameters_hex":'
            b'["0x1.0000000000000p+0","0x1.0000000000000p+0"],"jacobi_from_cartesian_matrix_hex":'
            b'[["0x1.0000000000000p-1","0x1.0000000000000p-1"],["-0x1.0000000000000p+0","0x1.0000000000000p+0"]],'
            b'"matrix_policy":"B[0,j]=mu_j/K_{N-1}; B[i,j]=-mu_j/K_{i-1} for j<i; B[i,i]=1; B[i,j]=0 for j>i; '
            b'A[j,0]=1; A[j,i]=-mu_i/K_i for j<i; A[i,i]=K_{i-1}/K_i; A[j,i]=0 for j>i; Q=B*x and x=A*Q",'
            b'"physical_mass_policy":"RETAINED_BUT_UNUSED_BY_GM_SCALED_MAP","primary_body_id":"A",'
            b'"schema":"jxplanetx.wisdom-holman-jacobi-binding.v1","symbol_policy":"mu_i=GM_i; '
            b'K_i=sum_{j=0}^i(mu_j); K_previous_i=K_{i-1}; lambda_i=mu_i*K_previous_i/K_i",'
            b'"transform_policy":"R_i=sum_{j=0}^i(mu_j*x_j)/K_i; V_i=sum_{j=0}^i(mu_j*v_j)/K_i; '
            b'Q_0=R_{N-1}; P_0=K_{N-1}*V_{N-1}=sum_j(mu_j*v_j); Q_i=x_i-R_{i-1}; '
            b'P_i=lambda_i*(v_i-V_{i-1}) for i>=1; inverse starts R_{N-1}=Q_0,V_{N-1}=P_0/K_{N-1} '
            b'and for i=N-1..1 sets R_{i-1}=R_i-(mu_i/K_i)*Q_i and V_{i-1}=V_i-P_i/K_{i-1}; '
            b'x_0=R_0,v_0=V_0; x_i=R_{i-1}+Q_i,v_i=V_{i-1}+P_i/lambda_i"}'
        )
        literal_preimage = (
            WH_JACOBI_BINDING_CHECKSUM_DOMAIN.encode("ascii")
            + b"\x00"
            + literal_payload
        )
        expected = hashlib.sha256(literal_preimage).hexdigest()
        self.assertEqual(
            expected,
            "872fc17e684933ced56e823f1991ff189220c223caad78dcaf1f98e5aebfbacc",
        )
        self.assertEqual(binding.binding_content_sha256, expected)

    def test_binding_rejects_old_digest_mutation_alias_and_subtypes(self):
        binding = _build_jacobi_coordinate_binding(
            ("A", "B"), np.array((1.0, 0.01), dtype=np.float64)
        )
        changed = binding.cartesian_from_jacobi_matrix.copy()
        changed[0, 1] = np.nextafter(changed[0, 1], math.inf)
        changed.setflags(write=False)
        with self.assertRaises(ValueError):
            dataclasses.replace(binding, cartesian_from_jacobi_matrix=changed)
        with self.assertRaises(ValueError):
            dataclasses.replace(binding, binding_content_sha256="0" * 64)

        owner = binding.gravitational_parameters.copy()
        external_view = owner.view()
        external_view.setflags(write=False)
        with self.assertRaises(ValueError):
            dataclasses.replace(binding, gravitational_parameters=external_view)

        class TupleSubclass(tuple):
            pass

        with self.assertRaises(ValueError):
            dataclasses.replace(binding, body_ids=TupleSubclass(binding.body_ids))


class UniversalKeplerMathTests(unittest.TestCase):
    def test_parabolic_boundary_is_a_typed_finite_far_envelope_exit(self):
        with self.assertRaises(wh_runtime._FiniteFarEnvelopeExit) as captured:
            wh_runtime._orbital_elements(
                np.array((1.0, 0.0, 0.0), dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                1.0,
            )
        self.assertEqual(
            captured.exception.reason,
            "FINITE_JACOBI_ECCENTRICITY_ABOVE_FAR_LIMIT",
        )

    def test_g_function_branch_boundary_and_parity(self):
        spec = UniversalKeplerSolverSpec()
        below = math.nextafter(0.5, 0.0)
        at = 0.5
        above = math.nextafter(0.5, math.inf)
        below_values = _universal_g_values(1.0, below, spec)
        at_values = _universal_g_values(1.0, at, spec)
        above_values = _universal_g_values(1.0, above, spec)
        self.assertTrue(below_values.used_series)
        self.assertTrue(at_values.used_series)
        self.assertFalse(above_values.used_series)
        self.assertEqual(at_values.term_count_per_function, WH_KEPLER_SERIES_TERM_COUNT)
        self.assertAlmostEqual(at_values.g0, math.cos(at), delta=2e-16)
        self.assertAlmostEqual(at_values.g1, math.sin(at), delta=2e-16)
        self.assertAlmostEqual(at_values.g2, 1.0 - math.cos(at), delta=2e-16)
        self.assertAlmostEqual(at_values.g3, at - math.sin(at), delta=2e-16)
        negative = _universal_g_values(1.0, -at, spec)
        self.assertEqual(negative.g0, at_values.g0)
        self.assertEqual(negative.g2, at_values.g2)
        self.assertEqual(negative.g1, -at_values.g1)
        self.assertEqual(negative.g3, -at_values.g3)

    def test_circular_forward_and_backward_match_independent_rotation(self):
        spec = UniversalKeplerSolverSpec()
        position = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        velocity = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        for signed_step in (0.1, -0.1, 0.5, -0.5):
            with self.subTest(signed_step=signed_step):
                result = _solve_elliptic_universal_kepler(
                    position, velocity, 1.0, signed_step, spec
                )
                expected_position = np.array(
                    (math.cos(signed_step), math.sin(signed_step), 0.0)
                )
                expected_velocity = np.array(
                    (-math.sin(signed_step), math.cos(signed_step), 0.0)
                )
                np.testing.assert_allclose(
                    result.position, expected_position, rtol=0.0, atol=3e-16
                )
                np.testing.assert_allclose(
                    result.velocity, expected_velocity, rtol=0.0, atol=3e-16
                )
                self.assertLessEqual(result.time_residual, result.residual_tolerance)
                self.assertLessEqual(result.lagrange_identity_error, 2e-15)
                self.assertLessEqual(result.energy_error, 2e-15)
                self.assertLessEqual(result.angular_momentum_error, 2e-15)

    def test_eccentric_forward_backward_roundtrip_and_accounting(self):
        eccentricity = 0.7
        position = np.array((1.0 - eccentricity, 0.0, 0.0), dtype=np.float64)
        velocity = np.array(
            (0.0, math.sqrt((1.0 + eccentricity) / (1.0 - eccentricity)), 0.0),
            dtype=np.float64,
        )
        spec = UniversalKeplerSolverSpec()
        forward = _solve_elliptic_universal_kepler(
            position, velocity, 1.0, 0.1, spec
        )
        backward = _solve_elliptic_universal_kepler(
            forward.position, forward.velocity, 1.0, -0.1, spec
        )
        np.testing.assert_allclose(backward.position, position, rtol=0.0, atol=3e-15)
        np.testing.assert_allclose(backward.velocity, velocity, rtol=0.0, atol=7e-15)
        self.assertGreater(forward.iterations, 0)
        self.assertLessEqual(forward.iterations, spec.maximum_iterations)
        self.assertLessEqual(
            forward.bracket_expansions, spec.maximum_bracket_expansions
        )
        self.assertGreater(forward.g_function_evaluations, 0)
        self.assertEqual(
            forward.series_terms_evaluated
            % (4 * WH_KEPLER_SERIES_TERM_COUNT),
            0,
        )
        self.assertAlmostEqual(forward.eccentricity, eccentricity, delta=2e-15)

    def test_whole_solver_exercises_trigonometric_g_branch(self):
        result = _solve_elliptic_universal_kepler(
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
            np.array((0.0, 1.0, 0.0), dtype=np.float64),
            1.0,
            0.6,
            UniversalKeplerSolverSpec(),
        )
        self.assertEqual(result.series_terms_evaluated, 0)
        np.testing.assert_allclose(
            result.position,
            np.array((math.cos(0.6), math.sin(0.6), 0.0)),
            rtol=0.0,
            atol=3e-16,
        )

    def test_nonfinite_derived_postcondition_values_fail_closed(self):
        position = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        velocity = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        target = "jxplanetx.engine.wisdom_holman"
        with mock.patch(
            f"{target}.WH_KEPLER_ENERGY_ROUNDOFF_FACTOR", math.inf
        ), self.assertRaisesRegex(ValueError, "specific-energy postcondition"):
            _solve_elliptic_universal_kepler(
                position, velocity, 1.0, 0.1, UniversalKeplerSolverSpec()
            )
        with mock.patch(
            f"{target}._cross3",
            return_value=np.array((math.inf, 0.0, 0.0), dtype=np.float64),
        ), np.errstate(invalid="ignore"), self.assertRaisesRegex(
            ValueError, "norm became nonfinite"
        ):
            _solve_elliptic_universal_kepler(
                position, velocity, 1.0, 0.1, UniversalKeplerSolverSpec()
            )

    def test_solver_fails_closed_outside_exact_types_and_elliptic_envelope(self):
        spec = UniversalKeplerSolverSpec()
        position = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        unbound_velocity = np.array((0.0, 2.0, 0.0), dtype=np.float64)
        high_e_velocity = np.array((0.0, math.sqrt(1.95), 0.0), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "bound elliptic"):
            _solve_elliptic_universal_kepler(
                position, unbound_velocity, 1.0, 0.01, spec
            )
        with self.assertRaisesRegex(ValueError, "eccentricity exceeds"):
            _solve_elliptic_universal_kepler(
                position, high_e_velocity, 1.0, 0.01, spec
            )
        for bad_gm in (1, 0.0, math.inf):
            with self.subTest(bad_gm=bad_gm), self.assertRaises(ValueError):
                _solve_elliptic_universal_kepler(
                    position,
                    np.array((0.0, 1.0, 0.0), dtype=np.float64),
                    bad_gm,  # type: ignore[arg-type]
                    0.01,
                    spec,
                )

        class ArraySubclass(np.ndarray):
            pass

        with self.assertRaisesRegex(ValueError, "exact NumPy ndarray"):
            _solve_elliptic_universal_kepler(
                position.view(ArraySubclass),
                np.array((0.0, 1.0, 0.0), dtype=np.float64),
                1.0,
                0.01,
                spec,
            )
        with self.assertRaises(ValueError):
            _solve_elliptic_universal_kepler(
                position,
                np.array((0.0, 1.0, 0.0), dtype=np.float64),
                1.0,
                0.0,
                spec,
            )
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "norm became nonfinite"):
                _solve_elliptic_universal_kepler(
                    np.array((1.0e308, 0.0, 0.0), dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    1.0,
                    0.01,
                    spec,
                )


class WisdomHolmanTransactionalProbeTests(unittest.TestCase):
    def initialized_circular_probe(
        self,
        *,
        fixed_step: float = 0.125,
        final_step: int = 2,
        minimum_encounter_pair_separation: float = 0.01,
    ):
        state = circular_planet_state()
        plan = runtime_plan(state)
        spec = runtime_spec(
            state,
            final_step=final_step,
            fixed_step=fixed_step,
            minimum_jacobi_periapse=0.5,
            minimum_encounter_pair_separation=(
                minimum_encounter_pair_separation
            ),
        )
        binding = _build_jacobi_coordinate_binding(
            state.body_ids, state.gravitational_parameters
        )
        prepared = wh_runtime._initialize_wisdom_holman_continuous_cache(
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
        )
        self.assertIsNone(prepared.terminal)
        self.assertIsNotNone(prepared.cache)
        return state, plan, spec, binding, prepared.cache

    def propose(
        self,
        *,
        cache,
        initial_snapshot,
        force_plan,
        integration_spec,
        binding,
        candidate_epoch,
    ):
        return wh_runtime._propose_wisdom_holman_macrostep(
            cache=cache,
            initial_snapshot=initial_snapshot,
            force_plan=force_plan,
            integration_spec=integration_spec,
            binding=binding,
            expected_cache_epoch=cache.accepted_epoch,
            expected_rebind_work_pending=cache.rebind_work_pending,
            candidate_epoch=candidate_epoch,
            candidate_outer_step_index=cache.accepted_outer_step_index + 1,
        )

    def commit(self, *, cache, proposal):
        self.assertIsNotNone(proposal.candidate)
        return wh_runtime._commit_wisdom_holman_macrostep(
            cache=cache,
            proposal=proposal,
            initial_snapshot=cache.authoritative_snapshot,
            force_plan=cache.force_plan,
            integration_spec=cache.integration_spec,
            binding=cache.binding,
            expected_cache_epoch=cache.accepted_epoch,
            expected_rebind_work_pending=cache.rebind_work_pending,
            expected_candidate_epoch=proposal.candidate.epoch,
            expected_outer_step_index=proposal.candidate.outer_step_index,
        )

    def reseal_cache(self, cache):
        digest = wh_runtime._cache_content_sha256(
            positions=cache.positions,
            velocities=cache.velocities,
            coordinates=cache.coordinates,
            momenta=cache.momenta,
            interaction_force=cache.interaction_force,
            accepted_outer_step_index=cache.accepted_outer_step_index,
            accepted_epoch=cache.accepted_epoch,
            force_custody=cache.force_custody,
            rebind_work_pending=cache.rebind_work_pending,
            accepted_start_guard=cache.accepted_start_guard,
            accepted_start_translation_residual=(
                cache.accepted_start_translation_residual
            ),
            authoritative_snapshot=cache.authoritative_snapshot,
            force_plan=cache.force_plan,
            integration_spec=cache.integration_spec,
            binding=cache.binding,
        )
        object.__setattr__(cache, "content_sha256", digest)

    def reseal_candidate(self, candidate):
        digest = wh_runtime._candidate_content_sha256(
            positions=candidate.positions,
            velocities=candidate.velocities,
            coordinates=candidate.coordinates,
            momenta=candidate.momenta,
            interaction_force=candidate.interaction_force,
            outer_step_index=candidate.outer_step_index,
            epoch=candidate.epoch,
            drift_guard=candidate.drift_guard,
            completed_guard=candidate.completed_guard,
            path_evidence=candidate.path_evidence,
            solver_records=candidate.solver_records,
            translation_residual=candidate.translation_residual,
            source_cache=candidate.source_cache,
            signed_fixed_step=candidate.signed_fixed_step,
        )
        object.__setattr__(candidate, "content_sha256", digest)

    def reseal_force_custody(self, custody):
        digest = wh_runtime._force_custody_content_sha256(
            force_model_ids=custody.force_model_ids,
            force_ledger=custody.force_ledger,
            origin_snapshot=custody.origin_snapshot,
            origin_snapshot_id=custody.origin_snapshot_id,
            origin_snapshot_epoch=custody.origin_snapshot_epoch,
            origin_snapshot_context_sha256=(
                custody.origin_snapshot_context_sha256
            ),
            origin_snapshot_content_sha256=(
                custody.origin_snapshot_content_sha256
            ),
            origin_force_plan=custody.origin_force_plan,
            origin_integration_spec=custody.origin_integration_spec,
            origin_binding=custody.origin_binding,
        )
        object.__setattr__(custody, "content_sha256", digest)

    def test_transactional_path_screen_precedes_endpoint_force(self):
        original = wh_runtime.evaluate_force_plan
        with mock.patch.object(
            wh_runtime, "evaluate_force_plan", wraps=original
        ) as evaluator:
            state, plan, spec, binding, cache = self.initialized_circular_probe(
                final_step=1,
                minimum_encounter_pair_separation=0.9,
            )
            proposal = self.propose(
                cache=cache,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                candidate_epoch=0.125,
            )
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(proposal.decision.outcome, "NEAR_SWITCH")
        self.assertEqual(
            proposal.decision.reason,
            "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN",
        )
        self.assertEqual(
            proposal.decision.phase, "POST_DRIFT_PRE_FORCE_PATH_SCREEN"
        )
        self.assertEqual(proposal.decision.pair_indices, ((0, 1),))
        self.assertIsNone(proposal.candidate)
        work = proposal.decision.work
        self.assertEqual(
            (
                work.force_calls_entered,
                work.force_evaluations_completed,
                work.interaction_force_assemblies,
                work.kepler_subflow_calls_entered,
                work.kepler_subflow_solves_completed,
                work.cartesian_to_jacobi_calls_entered,
                work.cartesian_to_jacobi_transforms_completed,
                work.jacobi_to_cartesian_calls_entered,
                work.jacobi_to_cartesian_transforms_completed,
                work.node_guard_evaluations,
                work.path_guard_evaluations,
                work.first_half_kicks_completed,
                work.center_of_mass_drifts_completed,
                work.second_half_kicks_completed,
            ),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0),
        )

    def test_transactional_pass_then_candidate_force_failure_is_typed(self):
        decisions = []
        for message in (
            "this text claims a path encounter",
            "this completely different text claims a solver bracket",
        ):
            original = wh_runtime.evaluate_force_plan
            entered = 0

            def fail_second(snapshot, plan):
                nonlocal entered
                entered += 1
                if entered == 2:
                    raise TrajectoryDomainError(message)
                return original(snapshot, plan)

            with mock.patch.object(
                wh_runtime, "evaluate_force_plan", side_effect=fail_second
            ):
                state, plan, spec, binding, cache = (
                    self.initialized_circular_probe(final_step=1)
                )
                proposal = self.propose(
                    cache=cache,
                    initial_snapshot=state,
                    force_plan=plan,
                    integration_spec=spec,
                    binding=binding,
                    candidate_epoch=0.125,
                )
            self.assertEqual(entered, 2)
            self.assertEqual(proposal.decision.outcome, "FATAL_FAILURE")
            self.assertEqual(
                proposal.decision.reason,
                "FORCE_EVALUATION_OR_IDENTITY_FAILURE",
            )
            self.assertEqual(
                proposal.decision.phase, "POST_DRIFT_CANDIDATE_FORCE"
            )
            self.assertEqual(proposal.decision.body_indices, ())
            self.assertEqual(proposal.decision.pair_indices, ())
            self.assertIsNone(proposal.candidate)
            self.assertEqual(str(proposal.public_error), message)
            work = proposal.decision.work
            self.assertEqual(
                (
                    work.force_calls_entered,
                    work.force_evaluations_completed,
                    work.interaction_force_assemblies,
                    work.path_guard_evaluations,
                    work.second_half_kicks_completed,
                ),
                (2, 1, 1, 1, 0),
            )
            decisions.append(dataclasses.asdict(proposal.decision))
        self.assertEqual(decisions[0], decisions[1])

    def test_transactional_continuous_far_streak_charges_rebind_once(self):
        state, plan, spec, binding, cache = self.initialized_circular_probe()
        first = self.propose(
            cache=cache,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.125,
        )
        self.assertEqual(first.decision.outcome, "FAR_PASS")
        self.assertIsNotNone(first.candidate)
        first_work = first.decision.work
        self.assertEqual(
            (
                first_work.cartesian_to_jacobi_transforms_completed,
                first_work.force_evaluations_completed,
                first_work.interaction_force_assemblies,
                first_work.node_guard_evaluations,
            ),
            (1, 2, 2, 3),
        )
        committed = self.commit(
            cache=cache, proposal=first
        )
        second = self.propose(
            cache=committed,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.25,
        )
        self.assertEqual(second.decision.outcome, "FAR_PASS")
        self.assertIsNotNone(second.candidate)
        second_work = second.decision.work
        self.assertEqual(
            (
                second_work.cartesian_to_jacobi_transforms_completed,
                second_work.force_evaluations_completed,
                second_work.interaction_force_assemblies,
                second_work.node_guard_evaluations,
            ),
            (0, 1, 1, 2),
        )

        rebound_snapshot = dataclasses.replace(
            state,
            epoch=0.125,
            positions=committed.positions,
            velocities=committed.velocities,
        )
        rebound = wh_runtime._initialize_wisdom_holman_continuous_cache(
            initial_snapshot=rebound_snapshot,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            accepted_outer_step_index=1,
            validate_initial_barycenter=False,
            retained_force_custody=committed.force_custody,
        )
        self.assertIsNone(rebound.terminal)
        self.assertIsNotNone(rebound.cache)
        rebound_proposal = self.propose(
            cache=rebound.cache,
            initial_snapshot=rebound_snapshot,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.25,
        )
        self.assertEqual(rebound_proposal.decision.outcome, "FAR_PASS")
        rebound_work = rebound_proposal.decision.work
        self.assertEqual(
            (
                rebound_work.cartesian_to_jacobi_transforms_completed,
                rebound_work.force_evaluations_completed,
                rebound_work.interaction_force_assemblies,
                rebound_work.node_guard_evaluations,
            ),
            (1, 2, 2, 3),
        )

    def test_kepler_probe_records_series_trig_and_both_directions(self):
        position = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        velocity = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        for signed_step, expected_series_bundles in (
            (0.125, 2),
            (-0.125, 2),
            (0.6, 0),
            (-0.6, 0),
        ):
            with self.subTest(signed_step=signed_step):
                recorder = wh_runtime._KeplerProbeAccumulator(1)
                subflow = _solve_elliptic_universal_kepler(
                    position,
                    velocity,
                    1.0,
                    signed_step,
                    UniversalKeplerSolverSpec(),
                    recorder,
                )
                recorder.call_completed = True
                retained = recorder.freeze("COMPLETED")
                self.assertTrue(retained.call_completed)
                self.assertEqual(retained.iterations_entered, 1)
                self.assertEqual(retained.bracket_expansions_entered, 0)
                self.assertEqual(retained.g_bundle_calls_entered, 2)
                self.assertEqual(retained.g_bundle_calls_completed, 2)
                self.assertEqual(
                    retained.completed_series_bundle_count,
                    expected_series_bundles,
                )
                self.assertEqual(
                    retained.series_terms_evaluated,
                    expected_series_bundles * 4 * WH_KEPLER_SERIES_TERM_COUNT,
                )
                self.assertEqual(
                    (
                        retained.iterations_entered,
                        retained.bracket_expansions_entered,
                        retained.g_bundle_calls_completed,
                        retained.series_terms_evaluated,
                    ),
                    (
                        subflow.iterations,
                        subflow.bracket_expansions,
                        subflow.g_function_evaluations,
                        subflow.series_terms_evaluated,
                    ),
                )

    def test_partial_kepler_work_and_reason_ignore_exception_text(self):
        state, plan, spec, binding, cache = self.initialized_circular_probe(
            final_step=1
        )
        decisions = []
        for message in (
            "pretend this is a force identity failure",
            "pretend this is a path clearance exit",
        ):
            def fail_g_bundle(_beta, _anomaly, _spec, recorder):
                recorder.record_series_term()
                recorder.record_series_term()
                raise TrajectoryDomainError(message)

            with mock.patch.object(
                wh_runtime, "_universal_g_values", side_effect=fail_g_bundle
            ):
                proposal = self.propose(
                    cache=cache,
                    initial_snapshot=state,
                    force_plan=plan,
                    integration_spec=spec,
                    binding=binding,
                    candidate_epoch=0.125,
                )
            self.assertEqual(proposal.decision.outcome, "FATAL_FAILURE")
            self.assertEqual(
                proposal.decision.reason,
                "NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            )
            self.assertEqual(proposal.decision.phase, "KEPLER_DRIFT_FLOW")
            self.assertEqual(proposal.decision.body_indices, (1,))
            self.assertIsNone(proposal.candidate)
            record = proposal.decision.work.kepler_probe_records[-1]
            self.assertFalse(record.call_completed)
            self.assertEqual(record.g_bundle_calls_entered, 1)
            self.assertEqual(record.g_bundle_calls_completed, 0)
            self.assertEqual(record.interrupted_series_terms, 2)
            self.assertEqual(record.series_terms_evaluated, 2)
            self.assertEqual(
                record.terminal_status,
                "INCOMPLETE_NONFINITE_OR_SINGULAR_BODY_NUMERICAL_FAILURE",
            )
            decisions.append(dataclasses.asdict(proposal.decision))
        self.assertEqual(decisions[0], decisions[1])

    def test_kepler_postcondition_failure_has_dedicated_typed_reason(self):
        state, plan, spec, binding, cache = self.initialized_circular_probe(
            final_step=1
        )
        with mock.patch.object(
            wh_runtime, "WH_KEPLER_ENERGY_ROUNDOFF_FACTOR", math.inf
        ):
            proposal = self.propose(
                cache=cache,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                candidate_epoch=0.125,
            )
        self.assertEqual(proposal.decision.outcome, "FATAL_FAILURE")
        self.assertEqual(
            proposal.decision.reason,
            "KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
        )
        self.assertEqual(proposal.decision.phase, "KEPLER_DRIFT_FLOW")
        self.assertEqual(proposal.decision.body_indices, (1,))
        self.assertIsNone(proposal.candidate)
        self.assertIsInstance(proposal.public_error, TrajectoryDomainError)
        record = proposal.decision.work.kepler_probe_records[-1]
        self.assertFalse(record.call_completed)
        self.assertEqual(
            record.g_bundle_calls_entered, record.g_bundle_calls_completed
        )
        self.assertEqual(
            record.terminal_status,
            "INCOMPLETE_KEPLER_SOLVER_BRACKET_ITERATION_STAGNATION_OR_POSTCONDITION_FAILURE",
        )

    def test_transactional_candidate_and_commit_have_independent_custody(self):
        state, plan, spec, binding, cache = self.initialized_circular_probe(
            final_step=1
        )
        snapshot_bytes = (
            state.positions.tobytes(order="C"),
            state.velocities.tobytes(order="C"),
        )
        cache_bytes = tuple(
            getattr(cache, name).tobytes(order="C")
            for name in (
                "positions",
                "velocities",
                "coordinates",
                "momenta",
                "interaction_force",
            )
        )
        proposal = self.propose(
            cache=cache,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.125,
        )
        self.assertEqual(proposal.decision.outcome, "FAR_PASS")
        self.assertIsNotNone(proposal.candidate)
        candidate = proposal.candidate
        committed = self.commit(
            cache=cache, proposal=proposal
        )
        self.assertEqual(
            snapshot_bytes,
            (
                state.positions.tobytes(order="C"),
                state.velocities.tobytes(order="C"),
            ),
        )
        self.assertEqual(
            cache_bytes,
            tuple(
                getattr(cache, name).tobytes(order="C")
                for name in (
                    "positions",
                    "velocities",
                    "coordinates",
                    "momenta",
                    "interaction_force",
                )
            ),
        )
        for name in (
            "positions",
            "velocities",
            "coordinates",
            "momenta",
            "interaction_force",
        ):
            initial_array = getattr(cache, name)
            candidate_array = getattr(candidate, name)
            committed_array = getattr(committed, name)
            for value in (initial_array, candidate_array, committed_array):
                self.assertIs(type(value), np.ndarray)
                self.assertEqual(value.dtype, np.dtype(np.float64))
                self.assertTrue(value.flags.owndata)
                self.assertFalse(value.flags.writeable)
            self.assertFalse(np.shares_memory(initial_array, candidate_array))
            self.assertFalse(np.shares_memory(initial_array, committed_array))
            self.assertFalse(np.shares_memory(candidate_array, committed_array))
            np.testing.assert_array_equal(candidate_array, committed_array)
        for subflow in candidate.solver_records:
            for value in (subflow.position, subflow.velocity):
                self.assertIs(type(value), np.ndarray)
                self.assertEqual(value.dtype, np.dtype(np.float64))
                self.assertTrue(value.flags.owndata)
                self.assertFalse(value.flags.writeable)
                for retained in (
                    cache.positions,
                    cache.velocities,
                    candidate.positions,
                    candidate.velocities,
                    committed.positions,
                    committed.velocities,
                ):
                    self.assertFalse(np.shares_memory(value, retained))
        for state_array in (state.positions, state.velocities):
            for retained in (
                cache.positions,
                cache.velocities,
                candidate.positions,
                candidate.velocities,
                committed.positions,
                committed.velocities,
            ):
                self.assertFalse(np.shares_memory(state_array, retained))

    def test_transactional_commit_rejects_stale_and_cross_cache_proposals(self):
        state, plan, spec, binding, initial_cache = (
            self.initialized_circular_probe()
        )
        first = self.propose(
            cache=initial_cache,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.125,
        )
        committed = self.commit(
            cache=initial_cache, proposal=first
        )
        second = self.propose(
            cache=committed,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            candidate_epoch=0.25,
        )
        with self.assertRaises(TrajectoryContractError):
            self.commit(
                cache=committed, proposal=first
            )
        with self.assertRaises(TrajectoryContractError):
            self.commit(
                cache=initial_cache, proposal=second
            )

        sibling_preparation = (
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
            )
        )
        self.assertIsNone(sibling_preparation.terminal)
        self.assertIsNotNone(sibling_preparation.cache)
        with self.assertRaises(TrajectoryContractError):
            self.commit(
                cache=sibling_preparation.cache, proposal=first
            )

    def test_transactional_boundaries_reject_coherent_cache_mutations(self):
        def fresh():
            return self.initialized_circular_probe(final_step=1)

        mutations = (
            (
                "wrong-shape momenta",
                lambda cache: object.__setattr__(
                    cache,
                    "momenta",
                    wh_runtime._readonly_copy(
                        np.zeros(1, dtype=np.float64)
                    ),
                ),
            ),
            (
                "altered interaction force",
                lambda cache: object.__setattr__(
                    cache,
                    "interaction_force",
                    wh_runtime._readonly_copy(
                        np.zeros_like(cache.interaction_force)
                    ),
                ),
            ),
            (
                "forged rebind state",
                lambda cache: object.__setattr__(
                    cache, "rebind_work_pending", False
                ),
            ),
            (
                "forged accepted epoch",
                lambda cache: object.__setattr__(
                    cache, "accepted_epoch", -100.0
                ),
            ),
            (
                "bool accepted index with coherent seal",
                lambda cache: object.__setattr__(
                    cache, "accepted_outer_step_index", False
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                state, plan, spec, binding, cache = fresh()
                with self.assertRaises(TypeError):
                    dataclasses.replace(cache, accepted_epoch=-100.0)
                mutate(cache)
                if "coherent seal" in label:
                    self.reseal_cache(cache)
                with self.assertRaises(
                    (TrajectoryContractError, TrajectoryDomainError)
                ):
                    wh_runtime._propose_wisdom_holman_macrostep(
                        cache=cache,
                        initial_snapshot=state,
                        force_plan=plan,
                        integration_spec=spec,
                        binding=binding,
                        expected_cache_epoch=0.0,
                        expected_rebind_work_pending=True,
                        candidate_epoch=0.125,
                        candidate_outer_step_index=1,
                    )

    def test_transactional_commit_rejects_candidate_and_work_mutations(self):
        def fresh_far():
            state, plan, spec, binding, cache = (
                self.initialized_circular_probe(final_step=1)
            )
            proposal = self.propose(
                cache=cache,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                candidate_epoch=0.125,
            )
            self.assertIsNotNone(proposal.candidate)
            return state, plan, spec, binding, cache, proposal

        def commit(state, plan, spec, binding, cache, proposal):
            return wh_runtime._commit_wisdom_holman_macrostep(
                cache=cache,
                proposal=proposal,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                expected_cache_epoch=0.0,
                expected_rebind_work_pending=True,
                expected_candidate_epoch=0.125,
                expected_outer_step_index=1,
            )

        candidate_mutations = (
            (
                "nonfinite positions",
                lambda candidate: object.__setattr__(
                    candidate,
                    "positions",
                    wh_runtime._readonly_copy(
                        np.full_like(candidate.positions, math.nan)
                    ),
                ),
            ),
            (
                "wrong-shape positions",
                lambda candidate: object.__setattr__(
                    candidate,
                    "positions",
                    wh_runtime._readonly_copy(
                        np.zeros(1, dtype=np.float64)
                    ),
                ),
            ),
            (
                "inconsistent momenta",
                lambda candidate: object.__setattr__(
                    candidate,
                    "momenta",
                    wh_runtime._readonly_copy(
                        np.zeros_like(candidate.momenta)
                    ),
                ),
            ),
            (
                "forged epoch and step",
                lambda candidate: (
                    object.__setattr__(candidate, "epoch", 999.0),
                    object.__setattr__(
                        candidate, "signed_fixed_step", 998.0
                    ),
                ),
            ),
            (
                "bool outer index with coherent seal",
                lambda candidate: object.__setattr__(
                    candidate, "outer_step_index", True
                ),
            ),
        )
        for label, mutate in candidate_mutations:
            with self.subTest(label=label):
                state, plan, spec, binding, cache, proposal = fresh_far()
                with self.assertRaises(TypeError):
                    dataclasses.replace(proposal.candidate, epoch=999.0)
                mutate(proposal.candidate)
                if "coherent seal" in label:
                    self.reseal_candidate(proposal.candidate)
                with self.assertRaises(
                    (TrajectoryContractError, TrajectoryDomainError)
                ):
                    commit(state, plan, spec, binding, cache, proposal)

        state, plan, spec, binding, cache, proposal = fresh_far()
        work = dataclasses.replace(
            proposal.decision.work,
            force_calls_entered=1,
            force_evaluations_completed=1,
            interaction_force_assemblies=1,
            cartesian_to_jacobi_calls_entered=0,
            cartesian_to_jacobi_transforms_completed=0,
            node_guard_evaluations=2,
        )
        altered = dataclasses.replace(
            proposal,
            decision=dataclasses.replace(proposal.decision, work=work),
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "rebind cadence"
        ):
            commit(state, plan, spec, binding, cache, altered)

        malformed_decision = dataclasses.replace(
            proposal, decision=object()
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "wrong exact schema"
        ):
            commit(
                state,
                plan,
                spec,
                binding,
                cache,
                malformed_decision,
            )

    def test_rebind_rejects_raw_or_forged_force_ledger_custody(self):
        state, plan, spec, binding, cache = self.initialized_circular_probe(
            final_step=1
        )
        forged_entry = dataclasses.replace(
            cache.force_ledger[0], model_id="forged.model"
        )
        for label, ids, ledger in (
            ("empty", cache.force_model_ids, ()),
            ("mutable", cache.force_model_ids, list(cache.force_ledger)),
            ("forged", cache.force_model_ids, (forged_entry,)),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                TrajectoryContractError, "raw rebind"
            ):
                wh_runtime._initialize_wisdom_holman_continuous_cache(
                    initial_snapshot=state,
                    force_plan=plan,
                    integration_spec=spec,
                    binding=binding,
                    retained_force_model_ids=ids,
                    retained_force_ledger=ledger,
                )
        with self.assertRaisesRegex(
            TrajectoryContractError, "exact private type"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                retained_force_custody=object(),
            )

        other_state = dataclasses.replace(
            state, snapshot_id="fixture.wh.other-run"
        )
        other_plan = dataclasses.replace(
            plan, plan_id="fixture.wh.other-plan"
        )
        other_spec = dataclasses.replace(spec)
        other_binding = _build_jacobi_coordinate_binding(
            other_state.body_ids, other_state.gravitational_parameters
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "different execution context"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=other_state,
                force_plan=other_plan,
                integration_spec=other_spec,
                binding=other_binding,
                retained_force_custody=cache.force_custody,
            )

        (
            sealed_state,
            sealed_plan,
            sealed_spec,
            sealed_binding,
            sealed_cache,
        ) = self.initialized_circular_probe(final_step=1)
        sealed_custody = sealed_cache.force_custody
        seal_forgery = dataclasses.replace(
            sealed_custody.force_ledger[0],
            assumptions=sealed_custody.force_ledger[0].assumptions
            + ("coherent but unauthorized assumption",),
        )
        object.__setattr__(
            sealed_custody, "force_ledger", (seal_forgery,)
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "sealed content"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=sealed_state,
                force_plan=sealed_plan,
                integration_spec=sealed_spec,
                binding=sealed_binding,
                retained_force_custody=sealed_custody,
            )

        (
            coherent_state,
            coherent_plan,
            coherent_spec,
            coherent_binding,
            coherent_cache,
        ) = self.initialized_circular_probe(final_step=1)
        coherent_custody = coherent_cache.force_custody
        coherent_forgery = dataclasses.replace(
            coherent_custody.force_ledger[0],
            assumptions=coherent_custody.force_ledger[0].assumptions
            + ("coherent resealed forgery",),
        )
        object.__setattr__(
            coherent_custody, "force_ledger", (coherent_forgery,)
        )
        self.reseal_force_custody(coherent_custody)
        with self.assertRaisesRegex(
            TrajectoryContractError, "authoritative rebind evaluation"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=coherent_state,
                force_plan=coherent_plan,
                integration_spec=coherent_spec,
                binding=coherent_binding,
                retained_force_custody=coherent_custody,
            )

        (
            metadata_state,
            metadata_plan,
            metadata_spec,
            metadata_binding,
            metadata_cache,
        ) = self.initialized_circular_probe(final_step=1)
        metadata_custody = metadata_cache.force_custody
        forged_metadata = dataclasses.replace(
            metadata_custody.force_ledger[0].state_metadata,
            snapshot_id="coherent.forged.snapshot",
        )
        forged_metadata_entry = dataclasses.replace(
            metadata_custody.force_ledger[0],
            state_metadata=forged_metadata,
        )
        object.__setattr__(
            metadata_custody, "force_ledger", (forged_metadata_entry,)
        )
        self.reseal_force_custody(metadata_custody)
        with self.assertRaises(TrajectoryContractError):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=metadata_state,
                force_plan=metadata_plan,
                integration_spec=metadata_spec,
                binding=metadata_binding,
                retained_force_custody=metadata_custody,
            )

        (
            epoch_state,
            epoch_plan,
            epoch_spec,
            epoch_binding,
            epoch_cache,
        ) = self.initialized_circular_probe(final_step=1)
        epoch_custody = epoch_cache.force_custody
        forged_epoch = 123.0
        forged_epoch_ledger = tuple(
            dataclasses.replace(
                entry,
                state_metadata=dataclasses.replace(
                    entry.state_metadata, epoch=forged_epoch
                ),
            )
            for entry in epoch_custody.force_ledger
        )
        object.__setattr__(
            epoch_custody, "origin_snapshot_epoch", forged_epoch
        )
        object.__setattr__(
            epoch_custody, "force_ledger", forged_epoch_ledger
        )
        self.reseal_force_custody(epoch_custody)
        with self.assertRaisesRegex(
            TrajectoryContractError, "originating snapshot label"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=epoch_state,
                force_plan=epoch_plan,
                integration_spec=epoch_spec,
                binding=epoch_binding,
                retained_force_custody=epoch_custody,
            )

    def test_public_wh_remains_unbounded_by_hybrid_body_cap(self):
        body_count = 17
        body_ids = ("STAR",) + tuple(
            f"P{index}" for index in range(1, body_count)
        )
        gravitational_parameters = np.array(
            (1.0,) + tuple(1.0e-10 for _ in range(body_count - 1)),
            dtype=np.float64,
        )
        binding = _build_jacobi_coordinate_binding(
            body_ids, gravitational_parameters
        )
        coordinates = np.zeros((body_count, 3), dtype=np.float64)
        momenta = np.zeros((body_count, 3), dtype=np.float64)
        for index in range(1, body_count):
            radius = 1.0 + 0.5 * index
            coordinates[index, 0] = radius
            momenta[index, 1] = float(binding.canonical_inertias[index]) * math.sqrt(
                float(binding.cumulative_gravitational_parameters[index]) / radius
            )
        positions, velocities = _jacobi_to_cartesian(
            binding, coordinates, momenta
        )
        state = StateSnapshot(
            "fixture.wh.seventeen-body",
            0.0,
            "SYNTHETIC",
            "BARYCENTRIC_INERTIAL",
            "BARYCENTER",
            "CARTESIAN_RIGHT_HANDED",
            "L",
            "T",
            "M",
            "fixture.wh.units",
            body_ids,
            positions,
            velocities,
            gravitational_parameters,
            gravitational_parameters.copy(),
            np.zeros(body_count, dtype=np.float64),
            np.ones(body_count, dtype=np.bool_),
            runtime_provenance(),
        )
        plan = runtime_plan(state)
        spec = runtime_spec(
            state,
            final_step=1,
            fixed_step=0.01,
            minimum_encounter_pair_separation=1.0e-4,
            minimum_jacobi_periapse=0.5,
        )
        result = integrate_wisdom_holman_trajectory(state, plan, spec)
        self.assertEqual(result.primary_map_kepler_subflow_solves, 16)
        with mock.patch.object(
            wh_runtime,
            "evaluate_force_plan",
            wraps=wh_runtime.evaluate_force_plan,
        ) as evaluator, self.assertRaisesRegex(
            TrajectoryContractError, "at most sixteen"
        ):
            wh_runtime._initialize_wisdom_holman_continuous_cache(
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                require_hybrid_decision=True,
            )
        self.assertEqual(evaluator.call_count, 0)

        prepared = wh_runtime._initialize_wisdom_holman_continuous_cache(
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
        )
        self.assertIsNotNone(prepared.cache)
        public_proposal = wh_runtime._propose_wisdom_holman_macrostep(
            cache=prepared.cache,
            initial_snapshot=state,
            force_plan=plan,
            integration_spec=spec,
            binding=binding,
            expected_cache_epoch=0.0,
            expected_rebind_work_pending=True,
            candidate_epoch=0.01,
            candidate_outer_step_index=1,
        )
        self.assertEqual(public_proposal.decision.outcome, "FAR_PASS")
        self.assertIsNotNone(public_proposal.candidate)

        def commit_strict(proposal):
            return wh_runtime._commit_wisdom_holman_macrostep(
                cache=prepared.cache,
                proposal=proposal,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                expected_cache_epoch=0.0,
                expected_rebind_work_pending=True,
                expected_candidate_epoch=0.01,
                expected_outer_step_index=1,
            )

        bool_work = dataclasses.replace(
            public_proposal.decision.work,
            first_half_kicks_completed=True,
        )
        bool_proposal = dataclasses.replace(
            public_proposal,
            decision=dataclasses.replace(
                public_proposal.decision, work=bool_work
            ),
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "nonnegative built-in integers"
        ):
            commit_strict(bool_proposal)

        class TextSubclass(str):
            pass

        text_proposal = dataclasses.replace(
            public_proposal,
            decision=dataclasses.replace(
                public_proposal.decision,
                outcome=TextSubclass("FAR_PASS"),
            ),
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "built-in string"
        ):
            commit_strict(text_proposal)

        with mock.patch.object(
            wh_runtime,
            "evaluate_force_plan",
            wraps=wh_runtime.evaluate_force_plan,
        ) as evaluator, self.assertRaisesRegex(
            TrajectoryContractError, "at most sixteen"
        ):
            wh_runtime._propose_wisdom_holman_macrostep(
                cache=prepared.cache,
                initial_snapshot=state,
                force_plan=plan,
                integration_spec=spec,
                binding=binding,
                expected_cache_epoch=0.0,
                expected_rebind_work_pending=True,
                candidate_epoch=0.01,
                candidate_outer_step_index=1,
                require_hybrid_decision=True,
            )
        self.assertEqual(evaluator.call_count, 0)


class WisdomHolmanRuntimeTests(unittest.TestCase):
    def test_weak_hierarchy_residual_force_sign_and_mass_known_answer(self):
        state = weak_hierarchy_state()
        binding = _build_jacobi_coordinate_binding(
            state.body_ids, state.gravitational_parameters
        )
        coordinates, _ = _cartesian_to_jacobi(
            binding, state.positions, state.velocities
        )
        evaluation = wh_runtime.evaluate_force_plan(state, runtime_plan(state))
        interaction, translation_residual = _interaction_force(
            binding=binding,
            coordinates=coordinates,
            cartesian_acceleration=evaluation.total_acceleration,
        )
        self.assertEqual(
            tuple(tuple(float(value).hex() for value in row) for row in interaction),
            (
                ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0"),
                (
                    "0x1.8faa0d0530000p-22",
                    "-0x1.3498c01e62000p-24",
                    "0x0.0p+0",
                ),
                (
                    "0x1.0959c80498800p-22",
                    "-0x1.335ceba0d5180p-23",
                    "0x0.0p+0",
                ),
            ),
        )
        self.assertEqual(translation_residual.hex(), "0x1.09eeacab398f3p-63")

    def test_public_call_counts_primary_map_and_mandatory_replay(self):
        state = circular_planet_state()
        original = wh_runtime.evaluate_force_plan
        with mock.patch(
            "jxplanetx.engine.wisdom_holman.evaluate_force_plan",
            wraps=original,
        ) as observed:
            result = integrate_wisdom_holman_trajectory(
                state,
                runtime_plan(state),
                runtime_spec(
                    state,
                    final_step=3,
                    fixed_step=0.125,
                    minimum_jacobi_periapse=0.5,
                ),
            )
        self.assertEqual(observed.call_count, 8)
        self.assertEqual(
            tuple(call.args[0].epoch for call in observed.call_args_list),
            (0.0, 0.125, 0.25, 0.375, 0.0, 0.125, 0.25, 0.375),
        )
        self.assertEqual(result.primary_map_force_evaluations, 4)
        self.assertEqual(result.validation_replay_force_evaluations, 4)
        self.assertEqual(result.total_public_call_force_evaluations, 8)

    def test_public_circular_two_body_map_and_exact_accounting(self):
        state = circular_planet_state()
        period = 2.0 * math.pi / math.sqrt(1.001)
        result = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(
                state,
                final_step=64,
                fixed_step=period / 64.0,
                minimum_jacobi_periapse=0.5,
            ),
        )
        self.assertIs(type(result), WisdomHolmanTrajectoryResult)
        np.testing.assert_allclose(
            result.final_positions, state.positions, rtol=0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            result.final_velocities, state.velocities, rtol=0.0, atol=2.0e-14
        )
        self.assertEqual(result.primary_map_force_evaluations, 65)
        self.assertEqual(result.validation_replay_force_evaluations, 65)
        self.assertEqual(result.total_public_call_force_evaluations, 130)
        self.assertEqual(result.primary_map_interaction_force_assemblies, 65)
        self.assertEqual(result.primary_map_kepler_subflow_solves, 64)
        self.assertEqual(result.total_public_call_kepler_subflow_solves, 128)
        self.assertEqual(result.primary_map_coordinate_forward_transforms, 1)
        self.assertEqual(result.total_public_call_coordinate_forward_transforms, 2)
        self.assertEqual(result.primary_map_coordinate_inverse_transforms, 128)
        self.assertEqual(result.total_public_call_coordinate_inverse_transforms, 256)
        self.assertEqual(result.primary_map_node_guard_evaluations, 129)
        self.assertEqual(result.primary_map_path_guard_evaluations, 64)
        self.assertEqual(
            result.public_execution_accounting_scope,
            "PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY",
        )
        self.assertLess(result.maximum_interaction_force_ratios[0], 1.0e-14)
        self.assertFalse(result.floating_point_symplectic)
        self.assertFalse(result.floating_point_exactly_reversible)
        self.assertFalse(result.registry_authorized)
        self.assertFalse(result.qualification_authorized)

    def test_weak_hierarchy_retains_guards_solver_and_owned_buffers(self):
        state = weak_hierarchy_state()
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        result = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(
                state,
                final_step=16,
                fixed_step=period / 64.0,
                checkpoints=(0, 4, 8, 12, 16),
            ),
        )
        self.assertEqual(result.checkpoint_count, 5)
        self.assertEqual(result.primary_map_force_evaluations, 17)
        self.assertEqual(result.total_public_call_force_evaluations, 34)
        self.assertEqual(result.primary_map_kepler_subflow_solves, 32)
        self.assertEqual(result.total_public_call_kepler_subflow_solves, 64)
        self.assertEqual(len(result.minimum_jacobi_periapses), 2)
        self.assertEqual(len(result.minimum_pair_path_lower_bounds), 3)
        self.assertEqual(len(result.minimum_secondary_hill_floor_ratios), 1)
        self.assertGreater(result.minimum_secondary_hill_floor_ratios[0], 1.0)
        self.assertLessEqual(
            max(result.maximum_interaction_force_ratios), 0.1
        )
        retained = [
            result.initial_snapshot.positions,
            result.initial_snapshot.velocities,
            result.coordinate_binding.gravitational_parameters,
            result.coordinate_binding.canonical_inertias,
        ]
        retained.extend(result.positions)
        retained.extend(result.velocities)
        for array in retained:
            self.assertIs(type(array), np.ndarray)
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
        for left in range(len(retained) - 1):
            for right in range(left + 1, len(retained)):
                self.assertFalse(np.shares_memory(retained[left], retained[right]))

    def test_checkpoint_cadence_does_not_change_map_or_force_count(self):
        state = weak_hierarchy_state()
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        sparse = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(state, final_step=64, fixed_step=period / 64.0),
        )
        dense = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(
                state,
                final_step=64,
                fixed_step=period / 64.0,
                checkpoints=(0, 16, 32, 48, 64),
            ),
        )
        np.testing.assert_array_equal(sparse.final_positions, dense.final_positions)
        np.testing.assert_array_equal(sparse.final_velocities, dense.final_velocities)
        self.assertEqual(
            sparse.primary_map_force_evaluations,
            dense.primary_map_force_evaluations,
        )
        self.assertEqual(
            sparse.primary_map_kepler_subflow_solves,
            dense.primary_map_kepler_subflow_solves,
        )

    def test_weak_hierarchy_absolute_rkf_oracle_and_second_order_gate(self):
        state = weak_hierarchy_state()
        plan = runtime_plan(state)
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        tolerance = np.full((3, 3), 1.0e-14, dtype=np.float64)
        reference = integrate_trajectory(
            state,
            plan,
            AdaptiveRKF78Spec(
                (0.0, period),
                period / 256.0,
                1.0e-15,
                period / 64.0,
                tolerance,
                1.0e-13,
                tolerance.copy(),
                1.0e-13,
                100_000,
                10_000,
                0.9,
                0.2,
                5.0,
            ),
        )
        states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for divisor in (64, 128, 256):
            result = integrate_wisdom_holman_trajectory(
                state,
                plan,
                runtime_spec(
                    state,
                    final_step=divisor,
                    fixed_step=period / divisor,
                ),
            )
            states[divisor] = (result.final_positions, result.final_velocities)
        position_errors = {
            divisor: float(
                np.max(
                    np.linalg.norm(
                        states[divisor][0] - reference.final_positions, axis=1
                    )
                )
            )
            for divisor in (64, 128, 256)
        }
        velocity_errors = {
            divisor: float(
                np.max(
                    np.linalg.norm(
                        states[divisor][1] - reference.final_velocities, axis=1
                    )
                )
            )
            for divisor in (64, 128, 256)
        }
        expected_position = {
            64: 3.4390517820644245e-6,
            128: 8.59605437202303e-7,
            256: 2.148915090357654e-7,
        }
        expected_velocity = {
            64: 3.8688854839660905e-6,
            128: 9.671250474486729e-7,
            256: 2.417752218497894e-7,
        }
        for divisor in (64, 128, 256):
            self.assertAlmostEqual(
                position_errors[divisor], expected_position[divisor], delta=2.0e-12
            )
            self.assertAlmostEqual(
                velocity_errors[divisor], expected_velocity[divisor], delta=2.0e-12
            )
        self.assertGreater(position_errors[64] / position_errors[128], 3.7)
        self.assertLess(position_errors[64] / position_errors[128], 4.3)
        self.assertGreater(velocity_errors[64] / velocity_errors[128], 3.7)
        self.assertLess(velocity_errors[64] / velocity_errors[128], 4.3)

    def test_forward_backward_is_roundoff_bounded_not_claimed_exact(self):
        state = circular_planet_state()
        period = 2.0 * math.pi / math.sqrt(1.001)
        forward = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(
                state,
                final_step=64,
                fixed_step=period / 64.0,
                minimum_jacobi_periapse=0.5,
            ),
        )
        backward_state = dataclasses.replace(
            state,
            snapshot_id="fixture.wh.circular-planet.backward",
            epoch=forward.final_epoch,
            positions=forward.final_positions.copy(),
            velocities=forward.final_velocities.copy(),
        )
        backward = integrate_wisdom_holman_trajectory(
            backward_state,
            runtime_plan(backward_state),
            runtime_spec(
                backward_state,
                final_step=64,
                fixed_step=-period / 64.0,
                minimum_jacobi_periapse=0.5,
            ),
        )
        np.testing.assert_allclose(
            backward.final_positions, state.positions, rtol=0.0, atol=5.0e-14
        )
        np.testing.assert_allclose(
            backward.final_velocities, state.velocities, rtol=0.0, atol=5.0e-14
        )
        self.assertFalse(backward.floating_point_exactly_reversible)

    def test_weak_hierarchy_forward_backward_one_period_is_roundoff_bounded(self):
        state = weak_hierarchy_state()
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        forward = integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(state, final_step=128, fixed_step=period / 128.0),
        )
        backward_state = dataclasses.replace(
            state,
            snapshot_id="fixture.wh.weak-hierarchy.backward",
            epoch=forward.final_epoch,
            positions=forward.final_positions.copy(),
            velocities=forward.final_velocities.copy(),
        )
        backward = integrate_wisdom_holman_trajectory(
            backward_state,
            runtime_plan(backward_state),
            runtime_spec(
                backward_state, final_step=128, fixed_step=-period / 128.0
            ),
        )
        position_error = float(
            np.max(np.linalg.norm(backward.final_positions - state.positions, axis=1))
        )
        velocity_error = float(
            np.max(np.linalg.norm(backward.final_velocities - state.velocities, axis=1))
        )
        self.assertLessEqual(position_error, 1.0e-12)
        self.assertLessEqual(velocity_error, 1.0e-12)
        self.assertFalse(backward.floating_point_exactly_reversible)

    def test_twenty_period_sampled_energy_envelope_improves_quadratically(self):
        state = weak_hierarchy_state()
        plan = runtime_plan(state)
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        gm = state.gravitational_parameters

        def invariants(
            positions: np.ndarray, velocities: np.ndarray
        ) -> tuple[float, np.ndarray]:
            energy = 0.0
            angular = np.zeros(3, dtype=np.float64)
            for body_index in range(3):
                energy += 0.5 * float(gm[body_index]) * float(
                    velocities[body_index] @ velocities[body_index]
                )
                angular += float(gm[body_index]) * np.cross(
                    positions[body_index], velocities[body_index]
                )
            for left in range(2):
                for right in range(left + 1, 3):
                    energy -= float(gm[left] * gm[right]) / float(
                        np.linalg.norm(positions[right] - positions[left])
                    )
            return energy, angular

        initial_energy, initial_angular = invariants(
            state.positions, state.velocities
        )
        energy_envelopes: dict[int, float] = {}
        final_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for divisor in (64, 128):
            final_step = 20 * divisor
            checkpoints = tuple(range(0, final_step + 1, divisor // 4))
            result = integrate_wisdom_holman_trajectory(
                state,
                plan,
                runtime_spec(
                    state,
                    final_step=final_step,
                    fixed_step=period / divisor,
                    checkpoints=checkpoints,
                ),
            )
            energy_errors: list[float] = []
            angular_errors: list[float] = []
            for positions, velocities in zip(result.positions, result.velocities):
                energy, angular = invariants(positions, velocities)
                energy_errors.append(abs((energy - initial_energy) / initial_energy))
                angular_errors.append(
                    float(
                        np.linalg.norm(angular - initial_angular)
                        / np.linalg.norm(initial_angular)
                    )
                )
            energy_envelopes[divisor] = max(energy_errors)
            final_states[divisor] = (
                result.final_positions,
                result.final_velocities,
            )
            self.assertLessEqual(max(angular_errors), 1.0e-13)
        self.assertLessEqual(energy_envelopes[64], 2.1e-6)
        self.assertLessEqual(energy_envelopes[128], 5.2e-7)
        self.assertGreater(
            energy_envelopes[64] / energy_envelopes[128], 3.7
        )
        self.assertLess(
            energy_envelopes[64] / energy_envelopes[128], 4.3
        )
        # The separately retained endpoint state disagreement makes clear that
        # a sampled invariant envelope is not a phase/trajectory guarantee.
        self.assertGreater(
            float(
                np.max(
                    np.linalg.norm(
                        final_states[64][0] - final_states[128][0], axis=1
                    )
                )
            ),
            1.0e-5,
        )

    def test_preflight_and_runtime_guards_fail_without_mutating_inputs(self):
        state = weak_hierarchy_state()
        before_positions = state.positions.copy()
        before_velocities = state.velocities.copy()
        excessive_gm = state.gravitational_parameters.copy()
        excessive_gm[2] = 0.02
        excessive_state = dataclasses.replace(
            state, gravitational_parameters=excessive_gm
        )
        period = float.fromhex("0x1.91ec4658ac4a3p+2")
        with self.assertRaisesRegex(TrajectoryDomainError, "secondary-to-primary"):
            integrate_wisdom_holman_trajectory(
                excessive_state,
                runtime_plan(excessive_state),
                runtime_spec(
                    excessive_state, final_step=1, fixed_step=period / 64.0
                ),
            )
        with self.assertRaisesRegex(TrajectoryDomainError, "Kepler drift reaches"):
            integrate_wisdom_holman_trajectory(
                state,
                runtime_plan(state),
                runtime_spec(state, final_step=32, fixed_step=period / 32.0),
            )
        np.testing.assert_array_equal(state.positions, before_positions)
        np.testing.assert_array_equal(state.velocities, before_velocities)

        class ArraySubclass(np.ndarray):
            pass

        subclass_state = dataclasses.replace(
            state, positions=state.positions.view(ArraySubclass)
        )
        with self.assertRaisesRegex(
            TrajectoryContractError, "input positions must be an exact NumPy ndarray"
        ):
            integrate_wisdom_holman_trajectory(
                subclass_state,
                runtime_plan(subclass_state),
                runtime_spec(
                    subclass_state, final_step=1, fixed_step=period / 64.0
                ),
            )

    def test_schedule_must_make_binary64_progress(self):
        state = circular_planet_state(epoch=1.0e20)
        wide_metadata = dataclasses.replace(
            runtime_metadata(), validity_start=-1.0e30, validity_end=1.0e30
        )
        plan = ForcePlan(
            "fixture.wh.large-epoch.plan",
            runtime_backend(),
            (
                NewtonianPointMass(
                    state.body_ids,
                    state.body_ids,
                    state.unit_system_id,
                    (wide_metadata,),
                ),
            ),
        )
        with self.assertRaises(TrajectoryStepLimitError):
            integrate_wisdom_holman_trajectory(
                state,
                plan,
                runtime_spec(
                    state,
                    final_step=1,
                    fixed_step=0.1,
                    minimum_jacobi_periapse=0.5,
                ),
            )


class WisdomHolmanResultIntegrityTests(unittest.TestCase):
    def one_step_result(self) -> WisdomHolmanTrajectoryResult:
        state = circular_planet_state()
        return integrate_wisdom_holman_trajectory(
            state,
            runtime_plan(state),
            runtime_spec(
                state,
                final_step=1,
                fixed_step=0.125,
                minimum_jacobi_periapse=0.5,
            ),
        )

    def test_schedule_checksum_has_independent_literal_known_answer(self):
        payload = (
            b'{"body_count":2,"checkpoint_epochs_hex":["0x0.0p+0","0x1.0000000000000p-2"],'
            b'"checkpoint_step_indices":[0,2],'
            b'"checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1",'
            b'"completed_steps":2,"direction":"FORWARD",'
            b'"fixed_step_hex":"0x1.0000000000000p-3",'
            b'"method_id":"integrator.symplectic.wisdom_holman_jacobi_kdk_2",'
            b'"public_execution_accounting_scope":"PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY",'
            b'"static_expected_counts":{'
            b'"coordinate_forward_transforms":{"primary_map":1,"total_public_call":2,"validation_replay":1},'
            b'"coordinate_inverse_transforms":{"primary_map":4,"total_public_call":8,"validation_replay":4},'
            b'"force_evaluations":{"primary_map":3,"total_public_call":6,"validation_replay":3},'
            b'"interaction_force_assemblies":{"primary_map":3,"total_public_call":6,"validation_replay":3},'
            b'"kepler_subflow_solves":{"primary_map":2,"total_public_call":4,"validation_replay":2},'
            b'"node_guard_evaluations":{"primary_map":5,"total_public_call":10,"validation_replay":5},'
            b'"path_guard_evaluations":{"primary_map":2,"total_public_call":4,"validation_replay":2}},'
            b'"validation_replay_count":1,'
            b'"validation_replay_policy":"ONE_FULL_DETERMINISTIC_SEMANTIC_REPLAY_DURING_RESULT_VALIDATION"}'
        )
        preimage = WH_SCHEDULE_CHECKSUM_DOMAIN.encode("ascii") + b"\x00" + payload
        expected = hashlib.sha256(preimage).hexdigest()
        self.assertEqual(
            expected,
            "d90c5edc26d9ef1d7fd2ea3bbe33e97b3a4fa1739ddc1e63ec8b50368a48c57d",
        )
        self.assertEqual(
            _schedule_content_sha256(
                checkpoint_step_indices=(0, 2),
                checkpoint_epochs=(0.0, 0.25),
                fixed_step=0.125,
                direction="FORWARD",
                completed_steps=2,
                body_count=2,
            ),
            expected,
        )

    def test_result_rejects_old_digest_alias_subtype_and_coherent_state_rewrite(self):
        result = self.one_step_result()
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(result, result_content_sha256="0" * 64)
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(result, completed_steps=True)
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(
                result,
                validation_replay_force_evaluations=(
                    result.validation_replay_force_evaluations - 1
                ),
            )

        external_owner = result.final_positions.copy()
        external_view = external_owner.view()
        external_view.setflags(write=False)
        altered_checkpoint = dataclasses.replace(
            result.checkpoints[-1], positions=external_view
        )
        with self.assertRaisesRegex(TrajectoryContractError, "owned read-only"):
            dataclasses.replace(
                result,
                checkpoints=result.checkpoints[:-1] + (altered_checkpoint,),
            )

        altered_positions = result.final_positions.copy()
        altered_positions[0, 0] = np.nextafter(altered_positions[0, 0], math.inf)
        altered_positions.setflags(write=False)
        altered_checkpoint = dataclasses.replace(
            result.checkpoints[-1], positions=altered_positions
        )
        altered_checkpoints = result.checkpoints[:-1] + (altered_checkpoint,)
        altered_run = dataclasses.replace(
            _run_from_result(result), checkpoints=altered_checkpoints
        )
        coherent_digest = _result_content_sha256(
            initial_snapshot=result.initial_snapshot,
            force_plan=result.force_plan,
            integration_spec=result.integration_spec,
            coordinate_binding=result.coordinate_binding,
            checkpoint_step_indices=result.checkpoint_step_indices,
            checkpoint_epochs=result.checkpoint_epochs,
            run=altered_run,
            direction=result.direction,
            schedule_content_sha256=result.schedule_content_sha256,
        )
        with self.assertRaisesRegex(TrajectoryContractError, "semantic replay"):
            dataclasses.replace(
                result,
                checkpoints=altered_checkpoints,
                result_content_sha256=coherent_digest,
            )

    def test_result_digest_is_stable_for_locked_one_step_fixture(self):
        literal_payload = (
            b'{"checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1",'
            b'"component_hashes":{"control":"cbb649801bffe0e71277e3607ff52037c964e1cc1476b9dddaa111d01ddb198b",'
            b'"coordinate_binding":"c3f925446d8a607e9cf1f97382769b02cd33fabad44c7a3dfb200021940358ce",'
            b'"force_plan":"85c7a67bf9cc8af88cad2e169ff3191b8f4aa1d9aa6fc1731403d8f1dc783e5b",'
            b'"initial_snapshot":"2a637200defcff0f465c81195177bad826b16a0e69fe3b72f77b0409b0338951",'
            b'"integration_spec":"1da4a4c1ef3d1b45dddcdf04cb45903d385d4f106d9b2e28688b47601b874ef9",'
            b'"public_execution_accounting":"e204496988727050cbccf6f62c54480c3eade039e8e253422f88740554fbdf64",'
            b'"runtime":"53af4ba82594c4d9ff64668e46ab1a309c06d6c48d8573a7da4c6e8aabdd624e",'
            b'"schedule_binding":"58c7936aef54caf0c3e980cefab9fb9bf7a5b4ed600b0cef65f5aed366e9217e"},'
            b'"component_order":["initial_snapshot","force_plan","integration_spec",'
            b'"coordinate_binding","schedule_binding","runtime",'
            b'"public_execution_accounting","control"],'
            b'"schema":"jxplanetx.wisdom-holman-result.v1"}'
        )
        literal_preimage = (
            WH_RESULT_CONTENT_CHECKSUM_DOMAIN.encode("ascii")
            + b"\x00"
            + literal_payload
        )
        expected = "862a2f1bf23acea0a4bc571203ddacd645e4c412531d020834d0094a95cba6a1"
        self.assertEqual(hashlib.sha256(literal_preimage).hexdigest(), expected)
        result = self.one_step_result()
        self.assertEqual(result.result_content_sha256, expected)

    def test_component_serializer_has_independent_float_hex_known_answer(self):
        domain = (
            WH_RESULT_CONTENT_CHECKSUM_DOMAIN + ".component.serializer-test"
        )
        literal_payload = (
            b'{"array":{"dtype":"float64","shape":[2],"values":'
            b'["0x0.0p+0","-0x0.0p+0"]},"flag":false,'
            b'"scalar":{"binary64_hex":"0x1.0000000000000p-2"},'
            b'"tuple":[3,"X"]}'
        )
        preimage = domain.encode("ascii") + b"\x00" + literal_payload
        expected = hashlib.sha256(preimage).hexdigest()
        self.assertEqual(
            expected,
            "6a60595fd84c26d7259d51704d529ce5a62ce2fccecf928058f1b7b704b16314",
        )
        self.assertEqual(
            _domain_separated_json_sha256(
                domain,
                {
                    "array": np.array((0.0, -0.0), dtype=np.float64),
                    "flag": False,
                    "scalar": 0.25,
                    "tuple": (3, "X"),
                },
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
