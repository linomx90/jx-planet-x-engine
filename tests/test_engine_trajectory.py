import dataclasses
import hashlib
import importlib
import math
import sys
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

import jxplanetx.engine as engine
from jxplanetx.engine import (
    AdaptiveRKF78Spec,
    BackendArrayError,
    BackendSpec,
    BackendUnavailableError,
    CannonballSRP,
    EvaluationError,
    ForceCollisionError,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    StateSnapshot,
    TrajectoryCheckpoint,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryResult,
    TrajectoryStepLimitError,
    integrate_trajectory,
)
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.contracts import ContractError
from jxplanetx.engine.rkf78 import (
    RKF78_ACCEPTED_ORDER,
    RKF78_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_ORDER,
    RKF78_STAGE_COUNT,
    RKF78_TABLEAU_SOURCE,
    _A,
    _B_DEFECT,
    _B_EIGHTH,
    _B_SEVENTH,
    _C,
    _compensated_add,
    _rkf78_step,
)
from jxplanetx.engine.trajectory_contracts import (
    ADAPTIVE_RKF78_METHOD_ID,
    RKF78_ACCEPTED_SOLUTION,
    RKF78_ACCEPTED_STATE_ACCUMULATION,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
    RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
    RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
    RKF78_CHECKPOINT_POLICY,
    RKF78_CHECKPOINT_PROPOSAL_POLICY,
    RKF78_CONTROLLER_EXPONENT,
    RKF78_DEFECT_ORIENTATION as CONTRACT_DEFECT_ORIENTATION,
    RKF78_EMBEDDED_SOLUTION,
    RKF78_ERROR_NORM,
    RKF78_ERROR_SCALE,
    RKF78_SOURCE_DOCUMENT_ID,
    RKF78_SOURCE_REPORT,
    RKF78_TIME_STEP_REPRESENTATION,
)
from jxplanetx.engine.trajectory import (
    _accepted_step_ledger_content_sha256,
    _normalized_max_error,
    _trial_endpoint,
)


UNIT_SYSTEM_ID = "fixture.trajectory.au_day_solar_mass"


def provenance() -> Provenance:
    return Provenance(
        "fixture.trajectory",
        "Synthetic trajectory fixture",
        "1",
        "e" * 64,
    )


def metadata(
    parameter_id: str,
    units: str,
    *,
    validity_start: float = -100.0,
    validity_end: float = 100.0,
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


def backend_spec(
    backend_id: str = "numpy",
    device: str = "cpu",
) -> BackendSpec:
    return BackendSpec(
        backend_id=backend_id,
        device=device,
        tile_size=2,
        dtype="float64",
        allow_fallback=False,
        deterministic_reductions=True,
        fast_math=False,
        determinism_scope="SAME_RUNTIME_DEVICE",
    )


def trajectory_state(
    xp=np,
    *,
    epoch: float = 0.0,
    **changes: object,
) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "fixture.trajectory.snapshot",
        "epoch": epoch,
        "time_scale": "TDB",
        "frame": "CENTRAL_BODY_INERTIAL",
        "origin": "SUN",
        "axes": "ICRS_ALIGNED",
        "length_unit": "AU",
        "time_unit": "DAY",
        "mass_unit": "SOLAR_MASS",
        "unit_system_id": UNIT_SYSTEM_ID,
        "body_ids": ("SUN", "ORBITER", "SAIL"),
        "positions": xp.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=xp.float64,
        ),
        "velocities": xp.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, math.sqrt(0.5), 0.0),
            ),
            dtype=xp.float64,
        ),
        "gravitational_parameters": xp.asarray((1.0, 0.0, 0.0), dtype=xp.float64),
        "masses": xp.asarray((1.0, 0.0, 0.0), dtype=xp.float64),
        "radii": xp.zeros(3, dtype=xp.float64),
        "massive": xp.asarray((True, False, False), dtype=xp.bool_),
        "provenance": provenance(),
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def newtonian(
    *,
    target_ids: tuple[str, ...] = ("ORBITER", "SAIL"),
    parameter_metadata: tuple[ParameterMetadata, ...] | None = None,
) -> NewtonianPointMass:
    return NewtonianPointMass(
        source_ids=("SUN",),
        target_ids=target_ids,
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=parameter_metadata
        or (metadata("state.gravitational_parameters", "AU^3/DAY^2"),),
    )


def one_pn() -> RestrictedStaticCentral1PN:
    return RestrictedStaticCentral1PN(
        central_source_id="SUN",
        target_ids=("ORBITER",),
        speed_of_light=100.0,
        maximum_compactness=0.1,
        maximum_speed_fraction_squared=0.1,
        unit_system_id=UNIT_SYSTEM_ID,
        parameter_metadata=(
            metadata("speed_of_light", "AU/DAY"),
            metadata("maximum_compactness", "1"),
            metadata("maximum_speed_fraction_squared", "1"),
        ),
    )


def srp(xp=np) -> CannonballSRP:
    return CannonballSRP(
        radiation_source_id="SUN",
        target_ids=("SAIL",),
        reference_pressure=1.0e-8,
        reference_distance=1.0,
        area_to_mass=xp.asarray((1.0,), dtype=xp.float64),
        radiation_pressure_coefficient=xp.asarray((1.0,), dtype=xp.float64),
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


def trajectory_plan(
    *models: object,
    backend: BackendSpec | None = None,
) -> ForcePlan:
    return ForcePlan(
        "fixture.trajectory.plan",
        backend or backend_spec(),
        tuple(models) if models else (newtonian(),),
    )


def runtime_spec(
    xp=np,
    *,
    body_count: int = 3,
    **changes: object,
) -> AdaptiveRKF78Spec:
    values: dict[str, object] = {
        "checkpoint_epochs": (0.0, 0.03, 0.1),
        "initial_step": 0.07,
        "minimum_step": 1e-12,
        "maximum_step": 0.1,
        "position_atol": xp.full((body_count, 3), 1e-12, dtype=xp.float64),
        "position_rtol": 1e-10,
        "velocity_atol": xp.full((body_count, 3), 1e-12, dtype=xp.float64),
        "velocity_rtol": 1e-10,
        "maximum_steps": 10_000,
        "maximum_rejections": 1_000,
        "safety_factor": 0.9,
        "minimum_scale_factor": 0.2,
        "maximum_scale_factor": 5.0,
    }
    values.update(changes)
    return AdaptiveRKF78Spec(**values)  # type: ignore[arg-type]


def spec(**changes: object) -> AdaptiveRKF78Spec:
    values: dict[str, object] = {
        "checkpoint_epochs": (0.0, 0.5, 1.0),
        "initial_step": 0.1,
        "minimum_step": 1e-8,
        "maximum_step": 0.25,
        "position_atol": np.full((2, 3), 1e-12, dtype=np.float64),
        "position_rtol": 1e-10,
        "velocity_atol": np.full((2, 3), 1e-12, dtype=np.float64),
        "velocity_rtol": 1e-10,
        "maximum_steps": 10_000,
        "maximum_rejections": 1_000,
        "safety_factor": 0.9,
        "minimum_scale_factor": 0.2,
        "maximum_scale_factor": 5.0,
    }
    values.update(changes)
    return AdaptiveRKF78Spec(**values)  # type: ignore[arg-type]


class RKF78ContractTests(unittest.TestCase):
    def test_fixed_method_identity_orientation_and_controller_are_exact(self):
        request = spec()
        self.assertEqual(request.method_id, ADAPTIVE_RKF78_METHOD_ID)
        self.assertEqual(request.source_report, RKF78_SOURCE_REPORT)
        self.assertEqual(request.source_document_id, RKF78_SOURCE_DOCUMENT_ID)
        self.assertEqual(request.stage_count, 13)
        self.assertEqual(request.force_evaluations_per_attempt, 13)
        self.assertEqual(request.principal_order, 8)
        self.assertEqual(request.accepted_order, 8)
        self.assertEqual(request.embedded_order, 7)
        self.assertEqual(request.accepted_solution, RKF78_ACCEPTED_SOLUTION)
        self.assertEqual(request.embedded_solution, RKF78_EMBEDDED_SOLUTION)
        self.assertEqual(request.defect_orientation, CONTRACT_DEFECT_ORIENTATION)
        self.assertEqual(request.controller_exponent, RKF78_CONTROLLER_EXPONENT)
        self.assertEqual(request.error_norm, RKF78_ERROR_NORM)
        self.assertEqual(request.error_scale, RKF78_ERROR_SCALE)
        self.assertEqual(request.checkpoint_policy, RKF78_CHECKPOINT_POLICY)
        self.assertEqual(
            request.time_step_representation,
            RKF78_TIME_STEP_REPRESENTATION,
        )
        self.assertEqual(
            request.checkpoint_proposal_policy,
            RKF78_CHECKPOINT_PROPOSAL_POLICY,
        )
        self.assertEqual(
            request.accepted_state_accumulation,
            RKF78_ACCEPTED_STATE_ACCUMULATION,
        )
        self.assertEqual(request.direction, "FORWARD")
        self.assertFalse(request.dense_output)
        self.assertTrue(request.supports_velocity_dependent_forces)
        self.assertEqual(request.evidence_class, "MODEL_OUTPUT")
        self.assertFalse(request.registry_authorized)
        self.assertFalse(request.qualification_authorized)

        for field, value in (
            ("accepted_order", 7),
            ("embedded_order", 8),
            ("accepted_solution", request.embedded_solution),
            ("defect_orientation", "EMBEDDED_MINUS_ACCEPTED"),
            ("controller_exponent", 1.0 / 7.0),
            ("accepted_state_accumulation", "NAIVE"),
            ("time_step_representation", "REQUESTED_STEP"),
            ("checkpoint_proposal_policy", "RESET_TO_CLIPPED_STEP"),
            ("dense_output", True),
            ("registry_authorized", True),
        ):
            with self.subTest(field=field), self.assertRaises(ContractError):
                dataclasses.replace(request, **{field: value})

    def test_checkpoint_direction_is_strict_and_exact(self):
        self.assertEqual(spec(checkpoint_epochs=(1.0, 0.5, 0.0)).direction, "BACKWARD")
        invalid = (
            [0.0, 1.0],
            (0.0,),
            (0.0, 0.0),
            (0.0, 1.0, 0.5),
            (0.0, math.nan),
            (False, 1.0),
        )
        for checkpoints in invalid:
            with self.subTest(checkpoints=checkpoints), self.assertRaises(ContractError):
                spec(checkpoint_epochs=checkpoints)

    def test_step_tolerance_budget_and_controller_controls_fail_closed(self):
        invalid: tuple[dict[str, object], ...] = (
            {"initial_step": 0.0},
            {"minimum_step": 0.2},
            {"maximum_step": 0.05},
            {"position_atol": None},
            {"velocity_atol": None},
            {"position_rtol": 0.0},
            {"velocity_rtol": math.inf},
            {"maximum_steps": 0},
            {"maximum_steps": True},
            {"maximum_rejections": -1},
            {"maximum_rejections": False},
            {"safety_factor": 1.0},
            {"minimum_scale_factor": 1.1},
            {"maximum_scale_factor": 0.9},
            {"minimum_scale_factor": 0.8, "maximum_scale_factor": 0.7},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ContractError):
                spec(**change)

    def test_backend_arrays_are_deferred_and_spec_has_identity_semantics(self):
        opaque = object()
        request = spec(position_atol=opaque, velocity_atol=opaque)
        self.assertIs(request.position_atol, opaque)
        self.assertIs(request.velocity_atol, opaque)
        self.assertNotEqual(request, dataclasses.replace(request))
        self.assertEqual(request, request)
        self.assertIsInstance(hash(request), int)


class AcceptedStepLedgerChecksumTests(unittest.TestCase):
    EXPECTED_SHA256 = (
        "ac590df4a3a432928502c9acce4ddde1df960b0e14ad51ae202ee1ec0933fb53"
    )

    def test_v1_literal_ascii_preimage_has_fixed_sha256(self):
        preimage = (
            b"jxplanetx.accepted-step-ledger.content-integrity.v1\x00"
            b'{"accepted_step_epochs_hex":["0x1.0000000000000p-1"],'
            b'"accepted_step_magnitudes_hex":["0x1.0000000000000p-1"],'
            b'"checkpoint_endpoint_bindings":[{"accepted_steps":0,'
            b'"epoch_hex":"-0x0.0p+0","index":0,"rejected_steps":0},'
            b'{"accepted_steps":1,"epoch_hex":"0x1.0000000000000p-1",'
            b'"index":1,"rejected_steps":2}],'
            b'"checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1",'
            b'"counters":{"accepted_steps":1,"attempted_steps":3,'
            b'"force_evaluations":39,"rejected_steps":2},'
            b'"direction":"FORWARD","extrema_hex":{'
            b'"last":"0x1.0000000000000p-1",'
            b'"maximum":"0x1.0000000000000p-1",'
            b'"minimum":"0x1.0000000000000p-1"},'
            b'"magnitude_source":"ABS_SIGNED_RKF78_STEP_ARGUMENT"}'
        )
        self.assertEqual(hashlib.sha256(preimage).hexdigest(), self.EXPECTED_SHA256)

    def test_production_v1_serializer_matches_vector_and_mutation_drifts(self):
        zero_state = np.zeros((1, 3), dtype=np.float64)
        checkpoints = (
            TrajectoryCheckpoint(
                0,
                -0.0,
                ("BODY",),
                "numpy",
                "cpu",
                zero_state.copy(),
                zero_state.copy(),
                0,
                0,
            ),
            TrajectoryCheckpoint(
                1,
                0.5,
                ("BODY",),
                "numpy",
                "cpu",
                zero_state.copy(),
                zero_state.copy(),
                1,
                2,
            ),
        )
        inputs = {
            "accepted_step_epochs": (0.5,),
            "accepted_step_magnitudes": (0.5,),
            "checkpoints": checkpoints,
            "direction": "FORWARD",
            "attempted_steps": 3,
            "accepted_steps": 1,
            "rejected_steps": 2,
            "force_evaluations": 39,
            "minimum_accepted_step": 0.5,
            "maximum_accepted_step": 0.5,
            "last_accepted_step": 0.5,
            "magnitude_source": RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
        }
        self.assertEqual(
            _accepted_step_ledger_content_sha256(**inputs),
            self.EXPECTED_SHA256,
        )
        mutated = dict(inputs)
        mutated["accepted_step_magnitudes"] = (math.nextafter(0.5, 0.0),)
        self.assertNotEqual(
            _accepted_step_ledger_content_sha256(**mutated),
            self.EXPECTED_SHA256,
        )


class RKF78TableauTests(unittest.TestCase):
    def test_nasa_tableau_shape_abscissae_and_row_sums(self):
        self.assertEqual(RKF78_TABLEAU_SOURCE, "NASA-TR-R-287")
        self.assertEqual(RKF78_STAGE_COUNT, 13)
        self.assertEqual(RKF78_ACCEPTED_ORDER, 8)
        self.assertEqual(RKF78_EMBEDDED_ORDER, 7)
        self.assertEqual(
            RKF78_DEFECT_ORIENTATION,
            "ACCEPTED_ORDER_8_MINUS_EMBEDDED_ORDER_7",
        )
        expected_c = tuple(
            float(value)
            for value in (
                Fraction(0),
                Fraction(2, 27),
                Fraction(1, 9),
                Fraction(1, 6),
                Fraction(5, 12),
                Fraction(1, 2),
                Fraction(5, 6),
                Fraction(1, 6),
                Fraction(2, 3),
                Fraction(1, 3),
                Fraction(1),
                Fraction(0),
                Fraction(1),
            )
        )
        self.assertEqual(_C, expected_c)
        self.assertEqual(tuple(len(row) for row in _A), tuple(range(13)))
        for stage, (row, abscissa) in enumerate(zip(_A, _C)):
            with self.subTest(stage=stage):
                self.assertAlmostEqual(sum(row), abscissa, places=14)

    def test_hatted_eighth_and_ordinary_seventh_weights_are_not_swapped(self):
        common = {
            5: Fraction(34, 105),
            6: Fraction(9, 35),
            7: Fraction(9, 35),
            8: Fraction(9, 280),
            9: Fraction(9, 280),
        }
        expected_eighth = tuple(
            float(
                common.get(
                    index,
                    Fraction(41, 840) if index in {11, 12} else Fraction(0),
                )
            )
            for index in range(13)
        )
        expected_seventh = tuple(
            float(
                common.get(
                    index,
                    Fraction(41, 840) if index in {0, 10} else Fraction(0),
                )
            )
            for index in range(13)
        )
        self.assertEqual(_B_EIGHTH, expected_eighth)
        self.assertEqual(_B_SEVENTH, expected_seventh)
        self.assertEqual(
            _B_DEFECT,
            tuple(high - low for high, low in zip(expected_eighth, expected_seventh)),
        )
        self.assertAlmostEqual(sum(_B_EIGHTH), 1.0, places=15)
        self.assertAlmostEqual(sum(_B_SEVENTH), 1.0, places=15)

    def test_independent_exponential_ode_confirms_local_order_orientation(self):
        backend = resolve_backend("numpy")

        def derivative(_epoch, positions, velocities):
            return positions, np.zeros_like(velocities)

        errors: list[tuple[float, float]] = []
        for step in (0.4, 0.2):
            trial = _rkf78_step(
                backend=backend,
                epoch=0.0,
                endpoint_epoch=step,
                step=step,
                positions=np.array((1.0,), dtype=np.float64),
                velocities=np.array((0.0,), dtype=np.float64),
                position_carry=np.zeros((1,), dtype=np.float64),
                velocity_carry=np.zeros((1,), dtype=np.float64),
                derivative=derivative,
            )
            exact = math.exp(step)
            errors.append(
                (
                    abs(float(trial.accepted_positions[0]) - exact),
                    abs(float(trial.embedded_positions[0]) - exact),
                )
            )
        high_ratio = errors[0][0] / errors[1][0]
        low_ratio = errors[0][1] / errors[1][1]
        self.assertGreater(high_ratio, 450.0)  # local h^9; ideal ratio 512
        self.assertLess(high_ratio, 650.0)
        self.assertGreater(low_ratio, 200.0)  # local h^8; ideal ratio 256
        self.assertLess(low_ratio, 350.0)
        self.assertLess(errors[0][0], errors[0][1])

    def test_constant_acceleration_is_exact_and_inputs_are_unchanged(self):
        backend = resolve_backend("numpy")
        positions = np.array(((2.0, -1.0, 0.5),), dtype=np.float64)
        velocities = np.array(((0.5, 1.0, -2.0),), dtype=np.float64)
        acceleration = np.array(((3.0, -4.0, 2.0),), dtype=np.float64)
        original_positions = positions.copy()
        original_velocities = velocities.copy()

        def derivative(_epoch, stage_positions, stage_velocities):
            self.assertEqual(stage_positions.shape, positions.shape)
            return stage_velocities, acceleration

        step = 0.25
        trial = _rkf78_step(
            backend=backend,
            epoch=1.0,
            endpoint_epoch=1.25,
            step=step,
            positions=positions,
            velocities=velocities,
            position_carry=np.zeros_like(positions),
            velocity_carry=np.zeros_like(velocities),
            derivative=derivative,
        )
        expected_positions = positions + step * velocities + 0.5 * step * step * acceleration
        expected_velocities = velocities + step * acceleration
        np.testing.assert_allclose(trial.accepted_positions, expected_positions, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(trial.accepted_velocities, expected_velocities, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(trial.embedded_positions, expected_positions, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(trial.embedded_velocities, expected_velocities, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(trial.position_defect, np.zeros_like(positions), rtol=0.0, atol=2e-16)
        np.testing.assert_allclose(trial.velocity_defect, np.zeros_like(velocities), rtol=0.0, atol=2e-16)
        np.testing.assert_array_equal(positions, original_positions)
        np.testing.assert_array_equal(velocities, original_velocities)

    def test_componentwise_compensated_add_handles_sign_and_exact_zero(self):
        for initial, increment, expected in (
            (1.0e16, 1.0, 1.0e16 + 10.0),
            (-1.0e16, -1.0, -1.0e16 - 10.0),
        ):
            with self.subTest(initial=initial, increment=increment):
                state = np.full((1, 1), initial, dtype=np.float64)
                carry = np.zeros_like(state)
                delta = np.full_like(state, increment)
                for _ in range(10):
                    state, carry = _compensated_add(state, delta, carry)
                np.testing.assert_array_equal(
                    state, np.full_like(state, expected)
                )

        state = np.array(((1.0, -2.0),), dtype=np.float64)
        updated, carry = _compensated_add(
            state,
            np.zeros_like(state),
            np.zeros_like(state),
        )
        np.testing.assert_array_equal(updated, state)
        np.testing.assert_array_equal(carry, np.zeros_like(state))
        self.assertFalse(np.any(np.signbit(carry)))

    def test_step_carries_must_be_native_float64_shaped_and_finite(self):
        backend = resolve_backend("numpy")
        state = np.zeros((1, 3), dtype=np.float64)

        def derivative(_epoch, positions, velocities):
            return np.zeros_like(positions), np.zeros_like(velocities)

        invalid = (
            ([0.0, 0.0, 0.0], BackendArrayError),
            (np.zeros((1, 3), dtype=np.float32), ValueError),
            (np.zeros((3,), dtype=np.float64), ValueError),
            (np.full((1, 3), np.nan, dtype=np.float64), ValueError),
        )
        for bad_carry, expected_error in invalid:
            with self.subTest(carry=bad_carry), self.assertRaises(expected_error):
                _rkf78_step(
                    backend=backend,
                    epoch=0.0,
                    endpoint_epoch=0.1,
                    step=0.1,
                    positions=state,
                    velocities=state,
                    position_carry=bad_carry,
                    velocity_carry=np.zeros_like(state),
                    derivative=derivative,
                )

    def test_public_fixed_cap_long_arc_compensation_gate(self):
        period = 2.0 * math.pi
        step = period / 256.0
        total_steps = 100 * 256
        checkpoint_epochs = tuple(
            float(index * period / 4.0) for index in range(100 * 4 + 1)
        )
        initial_positions = np.array(
            ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64
        )
        initial_velocities = np.array(
            ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64
        )
        binary_state = StateSnapshot(
            "fixture.trajectory.equal-mass-binary",
            0.0,
            "TDB",
            "BARYCENTRIC_INERTIAL",
            "BARYCENTER",
            "ICRS_ALIGNED",
            "AU",
            "DAY",
            "SOLAR_MASS",
            UNIT_SYSTEM_ID,
            ("BODY_A", "BODY_B"),
            initial_positions,
            initial_velocities,
            np.full(2, 0.5, dtype=np.float64),
            np.full(2, 0.5, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.ones(2, dtype=np.bool_),
            provenance(),
        )
        binary_metadata = metadata(
            "state.gravitational_parameters",
            "AU^3/DAY^2",
            validity_start=0.0,
            validity_end=checkpoint_epochs[-1],
        )
        binary_plan = ForcePlan(
            "fixture.trajectory.equal-mass-binary.plan",
            backend_spec(),
            (
                NewtonianPointMass(
                    ("BODY_A", "BODY_B"),
                    ("BODY_A", "BODY_B"),
                    UNIT_SYSTEM_ID,
                    (binary_metadata,),
                ),
            ),
        )
        loose = np.full((2, 3), 1.0e6, dtype=np.float64)
        result = integrate_trajectory(
            binary_state,
            binary_plan,
            AdaptiveRKF78Spec(
                checkpoint_epochs,
                step,
                1.0e-15,
                step,
                loose,
                1.0e-6,
                loose.copy(),
                1.0e-6,
                30_000,
                0,
                0.9,
                0.2,
                5.0,
            ),
        )
        positions = np.stack(result.positions)
        velocities = np.stack(result.velocities)
        exact_positions = np.empty_like(positions)
        exact_velocities = np.empty_like(velocities)
        for index, epoch in enumerate(checkpoint_epochs):
            cosine = math.cos(epoch)
            sine = math.sin(epoch)
            exact_positions[index] = (
                (-0.5 * cosine, -0.5 * sine, 0.0),
                (0.5 * cosine, 0.5 * sine, 0.0),
            )
            exact_velocities[index] = (
                (0.5 * sine, -0.5 * cosine, 0.0),
                (-0.5 * sine, 0.5 * cosine, 0.0),
            )
        position_error = float(
            np.max(np.linalg.norm(positions - exact_positions, axis=2))
        )
        velocity_error = float(
            np.max(np.linalg.norm(velocities - exact_velocities, axis=2))
        )
        separation = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
        energy = (
            0.25 * np.sum(velocities * velocities, axis=(1, 2))
            - 0.25 / separation
        )
        angular_momentum = 0.5 * np.sum(
            np.cross(positions, velocities)[:, :, 2], axis=1
        )
        self.assertLessEqual(position_error, 2.0e-12)
        self.assertLessEqual(velocity_error, 2.0e-12)
        self.assertLessEqual(float(np.max(np.abs(energy + 0.125))), 1.0e-14)
        self.assertLessEqual(
            float(np.max(np.abs(angular_momentum - 0.25))), 5.0e-15
        )
        self.assertEqual(result.rejected_steps, 0)
        self.assertGreaterEqual(result.accepted_steps, total_steps)
        self.assertLessEqual(result.accepted_steps, total_steps + 400)
        checkpoint_step_counts = tuple(
            right.accepted_steps - left.accepted_steps
            for left, right in zip(result.checkpoints, result.checkpoints[1:])
        )
        self.assertTrue(
            all(64 <= count <= 65 for count in checkpoint_step_counts)
        )
        self.assertEqual(result.attempted_steps, result.accepted_steps)
        self.assertEqual(result.force_evaluations, result.accepted_steps * 13)
        self.assertLessEqual(result.maximum_accepted_step, step)
        previous_epochs = (checkpoint_epochs[0],) + result.accepted_step_epochs[:-1]
        self.assertEqual(
            result.accepted_step_magnitudes,
            tuple(
                abs(endpoint - previous)
                for previous, endpoint in zip(
                    previous_epochs, result.accepted_step_epochs
                )
            ),
        )
        summed_steps = math.fsum(result.accepted_step_magnitudes)
        epoch_span = checkpoint_epochs[-1] - checkpoint_epochs[0]
        self.assertLessEqual(abs(summed_steps - epoch_span), math.ulp(epoch_span))


class TrajectoryRuntimeTests(unittest.TestCase):
    def test_public_result_binds_copied_inputs_exact_checkpoints_and_nonauthority(self):
        state = trajectory_state()
        force_plan = trajectory_plan(newtonian(), one_pn(), srp())
        request = runtime_spec()
        state_originals = {
            name: getattr(state, name).copy()
            for name in (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            )
        }
        tolerance_originals = (
            request.position_atol.copy(),
            request.velocity_atol.copy(),
        )
        srp_originals = (
            force_plan.models[2].area_to_mass.copy(),
            force_plan.models[2].radiation_pressure_coefficient.copy(),
        )

        result = integrate_trajectory(state, force_plan, request)

        self.assertIsInstance(result, TrajectoryResult)
        self.assertEqual(result.snapshot_id, state.snapshot_id)
        self.assertEqual(result.plan_id, force_plan.plan_id)
        self.assertEqual(result.backend_id, "numpy")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.dtype, "float64")
        self.assertEqual(result.checkpoint_epochs, request.checkpoint_epochs)
        self.assertEqual(result.direction, "FORWARD")
        self.assertEqual(len(result.positions), 3)
        self.assertEqual(len(result.velocities), 3)
        self.assertTrue(all(array.shape == (3, 3) for array in result.positions))
        self.assertTrue(all(array.shape == (3, 3) for array in result.velocities))
        self.assertTrue(
            all(array.dtype == np.dtype("float64") for array in result.positions)
        )
        self.assertEqual(result.force_evaluations, result.attempted_steps * 13)
        self.assertEqual(
            result.accepted_state_accumulation,
            RKF78_ACCEPTED_STATE_ACCUMULATION,
        )
        self.assertEqual(
            result.accepted_step_magnitude_source,
            RKF78_ACCEPTED_STEP_MAGNITUDE_SOURCE,
        )
        self.assertEqual(
            result.accepted_step_ledger_checksum_algorithm,
            RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_ALGORITHM,
        )
        self.assertEqual(
            result.accepted_step_ledger_checksum_domain,
            RKF78_ACCEPTED_STEP_LEDGER_CHECKSUM_DOMAIN,
        )
        self.assertEqual(
            result.time_step_representation,
            RKF78_TIME_STEP_REPRESENTATION,
        )
        self.assertEqual(
            result.checkpoint_proposal_policy,
            RKF78_CHECKPOINT_PROPOSAL_POLICY,
        )
        self.assertRegex(result.accepted_step_ledger_content_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(len(result.accepted_step_epochs), result.accepted_steps)
        self.assertEqual(len(result.accepted_step_magnitudes), result.accepted_steps)
        self.assertEqual(
            (
                result.minimum_accepted_step,
                result.maximum_accepted_step,
                result.last_accepted_step,
            ),
            (
                min(result.accepted_step_magnitudes),
                max(result.accepted_step_magnitudes),
                result.accepted_step_magnitudes[-1],
            ),
        )
        self.assertEqual(
            result.force_model_ids,
            tuple(model.model_id for model in force_plan.models),
        )
        self.assertEqual(
            result.force_model_ids,
            tuple(entry.model_id for entry in result.force_ledger),
        )
        self.assertEqual(result.evidence_class, "MODEL_OUTPUT")
        self.assertEqual(result.scope, "ADAPTIVE_TRAJECTORY_INTEGRATION")
        self.assertTrue(result.integrated)
        self.assertFalse(result.qualified)
        self.assertFalse(result.registry_authorized)
        self.assertFalse(result.qualification_authorized)
        self.assertFalse(result.dense_output)

        self.assertIsNot(result.initial_snapshot, state)
        self.assertIsNot(result.force_plan, force_plan)
        self.assertIsNot(result.integration_spec, request)
        self.assertIsNot(result.backend_spec, force_plan.backend)
        self.assertEqual(result.initial_snapshot.snapshot_id, state.snapshot_id)
        self.assertEqual(result.force_plan.plan_id, force_plan.plan_id)
        self.assertEqual(
            result.integration_spec.checkpoint_epochs,
            request.checkpoint_epochs,
        )
        self.assertFalse(np.shares_memory(result.initial_snapshot.positions, state.positions))
        self.assertFalse(
            np.shares_memory(result.integration_spec.position_atol, request.position_atol)
        )
        self.assertFalse(
            np.shares_memory(
                result.force_plan.models[2].area_to_mass,
                force_plan.models[2].area_to_mass,
            )
        )

        self.assertEqual(len(result.checkpoints), len(request.checkpoint_epochs))
        for index, (checkpoint, epoch) in enumerate(
            zip(result.checkpoints, request.checkpoint_epochs)
        ):
            with self.subTest(index=index):
                self.assertIsInstance(checkpoint, TrajectoryCheckpoint)
                self.assertEqual(checkpoint.index, index)
                self.assertEqual(checkpoint.epoch, epoch)
                self.assertEqual(checkpoint.positions.shape, (3, 3))
                self.assertEqual(checkpoint.velocities.shape, (3, 3))
                self.assertIs(checkpoint.positions, result.positions[index])
                self.assertIs(checkpoint.velocities, result.velocities[index])
                self.assertEqual(checkpoint.evidence_class, "MODEL_OUTPUT")
                self.assertFalse(checkpoint.qualified)
        self.assertEqual(result.checkpoints[0].accepted_steps, 0)
        self.assertEqual(result.checkpoints[0].rejected_steps, 0)
        self.assertEqual(result.checkpoints[-1].accepted_steps, result.accepted_steps)
        self.assertEqual(result.checkpoints[-1].rejected_steps, result.rejected_steps)
        np.testing.assert_array_equal(result.final_positions, result.positions[-1])
        np.testing.assert_array_equal(result.final_velocities, result.velocities[-1])
        self.assertEqual(result.final_epoch, request.checkpoint_epochs[-1])

        for name, original in state_originals.items():
            np.testing.assert_array_equal(getattr(state, name), original)
        np.testing.assert_array_equal(request.position_atol, tolerance_originals[0])
        np.testing.assert_array_equal(request.velocity_atol, tolerance_originals[1])
        np.testing.assert_array_equal(force_plan.models[2].area_to_mass, srp_originals[0])
        np.testing.assert_array_equal(
            force_plan.models[2].radiation_pressure_coefficient,
            srp_originals[1],
        )

    def test_forward_and_backward_checkpoints_are_hit_exactly_without_interpolation(self):
        cases = (
            (0.0, (0.0, 0.03, 0.1), "FORWARD"),
            (0.1, (0.1, 0.03, 0.0), "BACKWARD"),
        )
        for epoch, checkpoints, direction in cases:
            with self.subTest(direction=direction):
                state = trajectory_state(epoch=epoch)
                result = integrate_trajectory(
                    state,
                    trajectory_plan(),
                    runtime_spec(checkpoint_epochs=checkpoints),
                )
                self.assertEqual(result.checkpoint_epochs, checkpoints)
                self.assertEqual(
                    tuple(checkpoint.epoch for checkpoint in result.checkpoints),
                    checkpoints,
                )
                self.assertEqual(result.direction, direction)
                self.assertEqual(result.integration_spec.direction, direction)
                self.assertEqual(result.final_epoch, checkpoints[-1])
                self.assertTrue(
                    all(step > 0.0 for step in result.accepted_step_magnitudes)
                )

    def test_rejected_candidate_carries_are_discarded_and_accepted_ones_persist(self):
        real_step = _rkf78_step
        seen_position_carries: list[np.ndarray] = []

        def recording_step(**kwargs):
            seen_position_carries.append(kwargs["position_carry"].copy())
            trial = real_step(**kwargs)
            marker = np.full_like(
                trial.accepted_position_carry,
                0.125 * len(seen_position_carries),
            )
            return dataclasses.replace(
                trial,
                accepted_position_carry=marker,
                accepted_velocity_carry=marker.copy(),
            )

        def controlled_error(**_kwargs):
            return 2.0 if len(seen_position_carries) == 1 else 0.0

        with mock.patch(
            "jxplanetx.engine.trajectory._rkf78_step",
            side_effect=recording_step,
        ), mock.patch(
            "jxplanetx.engine.trajectory._normalized_max_error",
            side_effect=controlled_error,
        ):
            result = integrate_trajectory(
                trajectory_state(),
                trajectory_plan(),
                runtime_spec(
                    checkpoint_epochs=(0.0, 0.01, 0.02),
                    initial_step=0.01,
                    maximum_step=0.01,
                ),
            )

        self.assertEqual(result.rejected_steps, 1)
        self.assertGreaterEqual(len(seen_position_carries), 4)
        np.testing.assert_array_equal(
            seen_position_carries[0], np.zeros_like(seen_position_carries[0])
        )
        np.testing.assert_array_equal(
            seen_position_carries[1], np.zeros_like(seen_position_carries[1])
        )
        for index, observed in enumerate(seen_position_carries[2:], start=2):
            np.testing.assert_array_equal(
                observed,
                np.full_like(observed, 0.125 * index),
            )

    def test_tiny_checkpoint_residual_preserves_the_next_interval_proposal(self):
        tiny_offset_checkpoint = math.nextafter(0.1, math.inf)
        observed_signed_steps: list[float] = []
        real_step = _rkf78_step

        def recording_step(**kwargs):
            observed_signed_steps.append(kwargs["step"])
            return real_step(**kwargs)

        with mock.patch(
            "jxplanetx.engine.trajectory._rkf78_step",
            side_effect=recording_step,
        ), mock.patch(
            "jxplanetx.engine.trajectory._normalized_max_error",
            return_value=0.0,
        ):
            result = integrate_trajectory(
                trajectory_state(),
                trajectory_plan(),
                runtime_spec(
                    checkpoint_epochs=(0.0, tiny_offset_checkpoint, 0.3),
                    initial_step=0.1,
                    maximum_step=0.1,
                ),
            )

        self.assertGreaterEqual(len(observed_signed_steps), 3)
        self.assertEqual(observed_signed_steps[0], 0.1)
        self.assertLess(abs(observed_signed_steps[1]), 1.0e-12)
        self.assertGreater(abs(observed_signed_steps[2]), 0.09)
        self.assertLessEqual(abs(observed_signed_steps[2]), 0.1)
        self.assertEqual(
            tuple(abs(step) for step in observed_signed_steps),
            result.accepted_step_magnitudes,
        )
        self.assertLessEqual(result.accepted_steps, 5)

    def test_trial_endpoint_returns_the_exact_representable_delta(self):
        base = 1.0e10
        for direction, checkpoint in (
            (1.0, base + 1.0),
            (-1.0, base - 1.0),
        ):
            with self.subTest(direction=direction):
                endpoint, actual_step, clipped = _trial_endpoint(
                    base,
                    checkpoint,
                    direction,
                    0.1,
                )
                self.assertFalse(clipped)
                self.assertEqual(actual_step, endpoint - base)
                self.assertGreater(direction * actual_step, 0.0)
                self.assertLessEqual(abs(actual_step), 0.1)
                self.assertLess(abs(actual_step), 0.1)

    def test_large_epoch_step_cap_uses_actual_rkf78_argument_in_both_directions(self):
        base = 1.0e10
        bound_metadata = (
            metadata(
                "state.gravitational_parameters",
                "AU^3/DAY^2",
                validity_start=base,
                validity_end=base + 1.0,
            ),
        )
        plan = trajectory_plan(
            newtonian(parameter_metadata=bound_metadata)
        )
        for start, stop in ((base, base + 1.0), (base + 1.0, base)):
            with self.subTest(start=start, stop=stop):
                result = integrate_trajectory(
                    trajectory_state(epoch=start),
                    plan,
                    runtime_spec(
                        checkpoint_epochs=(start, stop),
                        initial_step=0.1,
                        maximum_step=0.1,
                        position_atol=np.full((3, 3), 1.0e6),
                        velocity_atol=np.full((3, 3), 1.0e6),
                    ),
                )
                endpoint_differences = tuple(
                    abs(right - left)
                    for left, right in zip(
                        (start,) + result.accepted_step_epochs[:-1],
                        result.accepted_step_epochs,
                    )
                )
                self.assertEqual(
                    endpoint_differences,
                    result.accepted_step_magnitudes,
                )
                self.assertTrue(any(value < 0.1 for value in endpoint_differences))
                self.assertLessEqual(result.maximum_accepted_step, 0.1)
                self.assertTrue(
                    all(value <= 0.1 for value in result.accepted_step_magnitudes)
                )
                signed_sum = math.copysign(
                    math.fsum(result.accepted_step_magnitudes), stop - start
                )
                span = stop - start
                self.assertLessEqual(abs(signed_sum - span), math.ulp(abs(span)))
                self.assertLessEqual(result.accepted_steps, 12)
                self.assertEqual(
                    result.time_step_representation,
                    RKF78_TIME_STEP_REPRESENTATION,
                )

    def test_static_central_two_body_solution_converges_at_eighth_order(self):
        state = trajectory_state()
        plan = trajectory_plan(newtonian(target_ids=("ORBITER",)))
        exact_position = np.array((math.cos(1.0), math.sin(1.0), 0.0))
        exact_velocity = np.array((-math.sin(1.0), math.cos(1.0), 0.0))
        errors: list[float] = []
        steps: list[int] = []
        for maximum_step in (0.5, 0.25):
            request = runtime_spec(
                checkpoint_epochs=(0.0, 1.0),
                initial_step=maximum_step,
                maximum_step=maximum_step,
                position_atol=np.full((3, 3), 1e6, dtype=np.float64),
                velocity_atol=np.full((3, 3), 1e6, dtype=np.float64),
                position_rtol=1e-6,
                velocity_rtol=1e-6,
                maximum_rejections=0,
            )
            result = integrate_trajectory(state, plan, request)
            errors.append(
                max(
                    float(np.max(np.abs(result.final_positions[1] - exact_position))),
                    float(np.max(np.abs(result.final_velocities[1] - exact_velocity))),
                )
            )
            steps.append(result.accepted_steps)
            self.assertEqual(result.minimum_accepted_step, maximum_step)
            self.assertEqual(result.maximum_accepted_step, maximum_step)
            self.assertEqual(result.last_accepted_step, maximum_step)
        self.assertEqual(steps, [2, 4])
        self.assertLess(errors[1], errors[0] / 100.0)
        self.assertLess(errors[1], 1e-9)

    def test_every_implemented_force_composition_integrates(self):
        plans = (
            (newtonian(),),
            (newtonian(), one_pn()),
            (newtonian(), srp()),
            (newtonian(), one_pn(), srp()),
        )
        for models in plans:
            with self.subTest(models=tuple(model.model_id for model in models)):
                result = integrate_trajectory(
                    trajectory_state(),
                    trajectory_plan(*models),
                    runtime_spec(checkpoint_epochs=(0.0, 0.01)),
                )
                self.assertEqual(
                    result.force_model_ids,
                    tuple(model.model_id for model in models),
                )
                self.assertEqual(result.force_evaluations, result.attempted_steps * 13)

    def test_separate_position_and_velocity_scales_feed_one_max_norm(self):
        backend = resolve_backend("numpy")
        zeros = np.zeros((1, 3), dtype=np.float64)
        position_current = np.full((1, 3), 1.0, dtype=np.float64)
        position_candidate = np.full((1, 3), 2.0, dtype=np.float64)
        velocity_current = np.full((1, 3), 10.0, dtype=np.float64)
        velocity_candidate = np.full((1, 3), 20.0, dtype=np.float64)
        normalized = _normalized_max_error(
            backend=backend,
            current_positions=position_current,
            current_velocities=velocity_current,
            candidate_positions=position_candidate,
            candidate_velocities=velocity_candidate,
            position_defect=np.full((1, 3), 0.6, dtype=np.float64),
            velocity_defect=np.full((1, 3), 6.0, dtype=np.float64),
            position_atol=np.full((1, 3), 0.2, dtype=np.float64),
            position_rtol=0.2,
            velocity_atol=np.full((1, 3), 2.0, dtype=np.float64),
            velocity_rtol=0.2,
            state_shape=(1, 3),
        )
        self.assertAlmostEqual(normalized, 1.0)

        velocity_dominates = _normalized_max_error(
            backend=backend,
            current_positions=zeros,
            current_velocities=zeros,
            candidate_positions=zeros,
            candidate_velocities=zeros,
            position_defect=np.full((1, 3), 1.0, dtype=np.float64),
            velocity_defect=np.full((1, 3), 2.0, dtype=np.float64),
            position_atol=np.full((1, 3), 4.0, dtype=np.float64),
            position_rtol=1e-6,
            velocity_atol=np.full((1, 3), 0.5, dtype=np.float64),
            velocity_rtol=1e-6,
            state_shape=(1, 3),
        )
        self.assertAlmostEqual(velocity_dominates, 4.0)


class TrajectoryFailureTests(unittest.TestCase):
    def test_explicit_gpu_request_never_falls_back_when_cupy_is_unavailable(self):
        plan = trajectory_plan(
            newtonian(),
            backend=backend_spec("cupy", "cuda:0"),
        )
        with mock.patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaisesRegex(BackendUnavailableError, "will not fall back"):
                integrate_trajectory(trajectory_state(), plan, runtime_spec())

    def test_collision_contact_fails_closed_without_event_or_response_model(self):
        touching = trajectory_state(
            radii=np.array((0.5, 0.5, 0.0), dtype=np.float64)
        )
        with self.assertRaises(ForceCollisionError):
            integrate_trajectory(
                touching,
                trajectory_plan(newtonian(target_ids=("ORBITER",))),
                runtime_spec(checkpoint_epochs=(0.0, 0.01)),
            )

    def test_initial_epoch_atol_backend_shape_dtype_finiteness_and_sign_fail_closed(self):
        state = trajectory_state()
        plan = trajectory_plan()
        cases: tuple[tuple[dict[str, object], type[BaseException]], ...] = (
            ({"checkpoint_epochs": (1.0, 2.0)}, TrajectoryContractError),
            ({"position_atol": [[1e-12] * 3] * 3}, BackendArrayError),
            ({"position_atol": np.ones((3, 3), dtype=np.float32)}, TrajectoryContractError),
            ({"velocity_atol": np.ones((2, 3), dtype=np.float64)}, TrajectoryContractError),
            ({"position_atol": np.zeros((3, 3), dtype=np.float64)}, TrajectoryDomainError),
            ({"velocity_atol": np.full((3, 3), np.nan)}, TrajectoryDomainError),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes), self.assertRaises(expected):
                integrate_trajectory(state, plan, runtime_spec(**changes))

    def test_metadata_must_cover_the_complete_trajectory(self):
        limited = metadata(
            "state.gravitational_parameters",
            "AU^3/DAY^2",
            validity_start=0.0,
            validity_end=0.05,
        )
        with self.assertRaisesRegex(TrajectoryContractError, "full trajectory"):
            integrate_trajectory(
                trajectory_state(),
                trajectory_plan(newtonian(parameter_metadata=(limited,))),
                runtime_spec(checkpoint_epochs=(0.0, 0.1)),
            )

    def test_maximum_step_rejection_and_minimum_step_failures_are_distinct(self):
        state = trajectory_state()
        plan = trajectory_plan()
        with self.assertRaisesRegex(TrajectoryStepLimitError, "maximum_steps"):
            integrate_trajectory(
                state,
                plan,
                runtime_spec(
                    checkpoint_epochs=(0.0, 1.0),
                    initial_step=0.1,
                    maximum_step=0.1,
                    maximum_steps=1,
                ),
            )

        strict = {
            "checkpoint_epochs": (0.0, 1.0),
            "initial_step": 1.0,
            "minimum_step": 0.5,
            "maximum_step": 1.0,
            "position_atol": np.full((3, 3), 1e-30, dtype=np.float64),
            "position_rtol": 1e-30,
            "velocity_atol": np.full((3, 3), 1e-30, dtype=np.float64),
            "velocity_rtol": 1e-30,
        }
        with self.assertRaisesRegex(TrajectoryStepLimitError, "maximum_rejections"):
            integrate_trajectory(
                state,
                plan,
                runtime_spec(maximum_rejections=0, **strict),
            )
        with self.assertRaisesRegex(TrajectoryStepLimitError, "minimum_step"):
            integrate_trajectory(
                state,
                plan,
                runtime_spec(maximum_rejections=100, **strict),
            )

    def test_binary64_epoch_underflow_fails_instead_of_stalling(self):
        epoch = 1e20
        for stop in (
            math.nextafter(epoch, math.inf),
            math.nextafter(epoch, -math.inf),
        ):
            with self.subTest(stop=stop):
                full_range = metadata(
                    "state.gravitational_parameters",
                    "AU^3/DAY^2",
                    validity_start=min(epoch, stop),
                    validity_end=max(epoch, stop),
                )
                with self.assertRaisesRegex(
                    TrajectoryStepLimitError, "cannot advance"
                ):
                    integrate_trajectory(
                        trajectory_state(epoch=epoch),
                        trajectory_plan(
                            newtonian(parameter_metadata=(full_range,))
                        ),
                        runtime_spec(
                            checkpoint_epochs=(epoch, stop),
                            initial_step=1.0,
                            minimum_step=1.0,
                            maximum_step=1.0,
                        ),
                    )

    def test_result_fixed_claim_fields_and_accounting_cannot_be_rewritten(self):
        result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(),
            runtime_spec(checkpoint_epochs=(0.0, 0.1)),
        )
        mutations = (
            {"accepted_order": 7},
            {"defect_orientation": "EMBEDDED_MINUS_ACCEPTED"},
            {"accepted_state_accumulation": "NAIVE"},
            {"accepted_step_magnitude_source": "ENDPOINT_SUBTRACTION"},
            {"accepted_step_ledger_checksum_algorithm": "SHA256"},
            {"accepted_step_ledger_checksum_domain": "other.domain"},
            {"time_step_representation": "REQUESTED_STEP"},
            {"checkpoint_proposal_policy": "RESET_TO_CLIPPED_STEP"},
            {"attempted_steps": result.attempted_steps + 1},
            {"force_evaluations": result.force_evaluations + 1},
            {"registry_authorized": True},
            {"qualification_authorized": True},
            {"dense_output": True},
        )
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(TrajectoryContractError):
                dataclasses.replace(result, **changes)

    def test_result_step_magnitude_ledger_and_extrema_fail_closed_on_tamper(self):
        result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(),
            runtime_spec(checkpoint_epochs=(0.0, 0.1)),
        )
        magnitudes = result.accepted_step_magnitudes
        mutations = (
            {"accepted_step_magnitudes": magnitudes[:-1]},
            {"accepted_step_magnitudes": (0.0,) + magnitudes[1:]},
            {"accepted_step_magnitudes": (math.nan,) + magnitudes[1:]},
            {"accepted_step_magnitudes": (magnitudes[0] * 2.0,) + magnitudes[1:]},
            {"minimum_accepted_step": result.minimum_accepted_step * 0.5},
            {"maximum_accepted_step": result.maximum_accepted_step * 0.5},
            {"last_accepted_step": result.last_accepted_step * 0.5},
            {"accepted_step_ledger_content_sha256": "0" * 64},
        )
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(
                TrajectoryContractError
            ):
                dataclasses.replace(result, **changes)

    def test_coherent_step_metadata_rewrite_requires_a_new_checksum(self):
        result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(),
            runtime_spec(checkpoint_epochs=(0.0, 0.1)),
        )
        self.assertGreaterEqual(result.accepted_steps, 2)
        mutated_epochs = list(result.accepted_step_epochs)
        mutated_epochs[0] = math.nextafter(mutated_epochs[0], 0.0)
        mutated_epoch_tuple = tuple(mutated_epochs)
        previous_epochs = (result.checkpoint_epochs[0],) + mutated_epoch_tuple[:-1]
        mutated_magnitudes = tuple(
            abs(endpoint - previous)
            for previous, endpoint in zip(previous_epochs, mutated_epoch_tuple)
        )
        minimum_step = min(mutated_magnitudes)
        maximum_step = max(mutated_magnitudes)
        last_step = mutated_magnitudes[-1]

        with self.assertRaisesRegex(TrajectoryContractError, "bit-exactly"):
            dataclasses.replace(
                result,
                accepted_step_magnitudes=mutated_magnitudes,
            )

        replacement_checksum = _accepted_step_ledger_content_sha256(
            accepted_step_epochs=mutated_epoch_tuple,
            accepted_step_magnitudes=mutated_magnitudes,
            checkpoints=result.checkpoints,
            direction=result.direction,
            attempted_steps=result.attempted_steps,
            accepted_steps=result.accepted_steps,
            rejected_steps=result.rejected_steps,
            force_evaluations=result.force_evaluations,
            minimum_accepted_step=minimum_step,
            maximum_accepted_step=maximum_step,
            last_accepted_step=last_step,
            magnitude_source=result.accepted_step_magnitude_source,
        )
        coherent_model_output = dataclasses.replace(
            result,
            accepted_step_epochs=mutated_epoch_tuple,
            accepted_step_magnitudes=mutated_magnitudes,
            minimum_accepted_step=minimum_step,
            maximum_accepted_step=maximum_step,
            last_accepted_step=last_step,
            accepted_step_ledger_content_sha256=replacement_checksum,
        )
        self.assertFalse(coherent_model_output.qualified)
        self.assertFalse(coherent_model_output.registry_authorized)
        self.assertFalse(coherent_model_output.qualification_authorized)

    def test_result_cross_bindings_and_array_shapes_fail_closed(self):
        result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(),
            runtime_spec(checkpoint_epochs=(0.0, 0.1)),
        )
        wrong_epochs = (0.0, 0.2)
        mutations = (
            {"snapshot_id": "other.snapshot"},
            {"plan_id": "other.plan"},
            {"backend_id": "cupy"},
            {"checkpoint_epochs": wrong_epochs},
            {"direction": "BACKWARD"},
            {"force_model_ids": ("force.nongrav.srp_cannonball",)},
        )
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(TrajectoryContractError):
                dataclasses.replace(result, **changes)

        checkpoint = result.checkpoints[0]
        checkpoint_mutations = (
            {"body_ids": ("OTHER",)},
            {"positions": np.zeros((1, 3), dtype=np.float64)},
            {"velocities": np.zeros((3, 3), dtype=np.float32)},
        )
        for changes in checkpoint_mutations:
            with self.subTest(checkpoint_changes=changes), self.assertRaises(
                TrajectoryContractError
            ):
                dataclasses.replace(checkpoint, **changes)

        wrong_backend_checkpoint = dataclasses.replace(checkpoint, backend_id="cupy")
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(
                result,
                checkpoints=(wrong_backend_checkpoint,) + result.checkpoints[1:],
            )

    def test_restricted_1pn_result_cannot_rebind_a_nonstatic_or_displaced_central_body(self):
        result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(newtonian(), one_pn()),
            runtime_spec(checkpoint_epochs=(0.0, 0.1)),
        )
        for field in ("positions", "velocities"):
            with self.subTest(field=field):
                retained = getattr(result.initial_snapshot, field).copy()
                retained[0, 0] = 1.0
                tampered_snapshot = dataclasses.replace(
                    result.initial_snapshot,
                    **{field: retained},
                )

                checkpoint_array = getattr(result.checkpoints[0], field).copy()
                checkpoint_array[0, 0] = 1.0
                tampered_checkpoint = dataclasses.replace(
                    result.checkpoints[0],
                    **{field: checkpoint_array},
                )
                with self.assertRaises(EvaluationError):
                    dataclasses.replace(
                        result,
                        initial_snapshot=tampered_snapshot,
                        checkpoints=(tampered_checkpoint,) + result.checkpoints[1:],
                    )


class OptionalCuPyTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module("cupy")
            count = int(cls.cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            raise unittest.SkipTest(f"CuPy GPU runtime unavailable: {exc}") from exc
        if count < 1:
            raise unittest.SkipTest("CuPy found no GPU")

    def test_gpu_trajectory_matches_cpu_and_all_result_arrays_remain_resident(self):
        cpu_result = integrate_trajectory(
            trajectory_state(),
            trajectory_plan(newtonian(), one_pn(), srp()),
            runtime_spec(checkpoint_epochs=(0.0, 0.05)),
        )
        cp = self.cp
        with cp.cuda.Device(0):
            state = trajectory_state(cp)
            plan = trajectory_plan(
                newtonian(),
                one_pn(),
                srp(cp),
                backend=backend_spec("cupy", "cuda:0"),
            )
            request = runtime_spec(cp, checkpoint_epochs=(0.0, 0.05))
            observed_carry_calls = 0

            def observe_native_carries(**kwargs):
                nonlocal observed_carry_calls
                observed_carry_calls += 1
                for name in ("position_carry", "velocity_carry"):
                    carry = kwargs[name]
                    self.assertIsInstance(carry, cp.ndarray)
                    self.assertEqual(int(carry.device.id), 0)
                    self.assertEqual(carry.dtype, cp.dtype("float64"))
                trial = _rkf78_step(**kwargs)
                for carry in (
                    trial.accepted_position_carry,
                    trial.accepted_velocity_carry,
                ):
                    self.assertIsInstance(carry, cp.ndarray)
                    self.assertEqual(int(carry.device.id), 0)
                    self.assertEqual(carry.dtype, cp.dtype("float64"))
                return trial

            with mock.patch.object(
                cp,
                "asnumpy",
                side_effect=AssertionError("implicit device-to-host transfer"),
            ), mock.patch(
                "jxplanetx.engine.trajectory._rkf78_step",
                side_effect=observe_native_carries,
            ):
                result = integrate_trajectory(state, plan, request)
            self.assertGreater(observed_carry_calls, 0)

            arrays = [
                result.positions,
                result.velocities,
                result.initial_snapshot.positions,
                result.initial_snapshot.velocities,
                result.integration_spec.position_atol,
                result.integration_spec.velocity_atol,
                result.force_plan.models[2].area_to_mass,
                result.force_plan.models[2].radiation_pressure_coefficient,
            ]
            arrays.extend(checkpoint.positions for checkpoint in result.checkpoints)
            arrays.extend(checkpoint.velocities for checkpoint in result.checkpoints)
            for array in arrays:
                self.assertIsInstance(array, cp.ndarray)
                self.assertEqual(int(array.device.id), 0)
                self.assertEqual(array.dtype, cp.dtype("float64"))
            np.testing.assert_allclose(
                cp.asnumpy(cp.stack(result.positions, axis=0)),
                np.stack(cpu_result.positions, axis=0),
                rtol=5e-13,
                atol=2e-14,
            )
            np.testing.assert_allclose(
                cp.asnumpy(cp.stack(result.velocities, axis=0)),
                np.stack(cpu_result.velocities, axis=0),
                rtol=5e-13,
                atol=2e-14,
            )

    def test_gpu_request_rejects_host_tolerances_without_transfer(self):
        cp = self.cp
        with cp.cuda.Device(0):
            state = trajectory_state(cp)
            plan = trajectory_plan(
                newtonian(),
                backend=backend_spec("cupy", "cuda:0"),
            )
            request = runtime_spec(cp, position_atol=np.ones((3, 3), dtype=np.float64))
            with self.assertRaises(BackendArrayError):
                integrate_trajectory(state, plan, request)


if __name__ == "__main__":
    unittest.main()
