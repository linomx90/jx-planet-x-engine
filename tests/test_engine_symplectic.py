import dataclasses
import hashlib
import math
import unittest
from types import SimpleNamespace

import numpy as np

from jxplanetx.engine import (
    BackendSpec,
    CannonballSRP,
    FixedStepKDKSpec,
    ForcePlan,
    KDKTrajectoryResult,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
    TrajectoryContractError,
    TrajectoryDomainError,
    TrajectoryStepLimitError,
    integrate_kdk_trajectory,
)
from jxplanetx.engine.backends import resolve_backend
from jxplanetx.engine.symplectic import (
    _kdk_step,
    _result_content_sha256,
    _schedule_content_sha256,
)


UNIT_SYSTEM_ID = "fixture.kdk.units"


def provenance() -> Provenance:
    return Provenance("fixture.kdk", "Synthetic KDK fixture", "1", "d" * 64)


def metadata(
    *, validity_start: float = -10_000.0, validity_end: float = 10_000.0
) -> ParameterMetadata:
    return ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance(),
        None,
        None,
        validity_start,
        validity_end,
    )


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


def binary_state(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "fixture.kdk.equal-binary",
        "epoch": 0.0,
        "time_scale": "TDB",
        "frame": "BARYCENTRIC_INERTIAL",
        "origin": "BARYCENTER",
        "axes": "ICRS_ALIGNED",
        "length_unit": "L",
        "time_unit": "T",
        "mass_unit": "M",
        "unit_system_id": UNIT_SYSTEM_ID,
        "body_ids": ("A", "B"),
        "positions": np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))),
        "velocities": np.array(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0))),
        "gravitational_parameters": np.full(2, 0.5, dtype=np.float64),
        "masses": np.full(2, 0.5, dtype=np.float64),
        "radii": np.zeros(2, dtype=np.float64),
        "massive": np.ones(2, dtype=np.bool_),
        "provenance": provenance(),
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def mutual_plan(
    state: StateSnapshot | None = None,
    *,
    backend_spec: BackendSpec | None = None,
    model: object | None = None,
    validity_start: float = -10_000.0,
    validity_end: float = 10_000.0,
) -> ForcePlan:
    state = state or binary_state()
    newtonian = model or NewtonianPointMass(
        state.body_ids,
        state.body_ids,
        state.unit_system_id,
        (metadata(validity_start=validity_start, validity_end=validity_end),),
    )
    return ForcePlan("fixture.kdk.plan", backend_spec or backend(), (newtonian,))


def kdk_spec(
    checkpoint_step_indices: tuple[int, ...] = (0, 32, 64),
    fixed_step: float = 2.0 * math.pi / 64.0,
    **changes: object,
) -> FixedStepKDKSpec:
    values: dict[str, object] = {
        "checkpoint_step_indices": checkpoint_step_indices,
        "fixed_step": fixed_step,
        "maximum_steps": checkpoint_step_indices[-1],
        "minimum_swept_pair_separation": 0.1,
        "maximum_pair_frequency_step": 0.1,
    }
    values.update(changes)
    return FixedStepKDKSpec(**values)  # type: ignore[arg-type]


def integrate(spec: FixedStepKDKSpec | None = None) -> KDKTrajectoryResult:
    state = binary_state()
    return integrate_kdk_trajectory(state, mutual_plan(state), spec or kdk_spec())


def recomputed_result_digest(
    result: KDKTrajectoryResult, **changes: object
) -> str:
    values: dict[str, object] = {
        "initial_snapshot": result.initial_snapshot,
        "force_plan": result.force_plan,
        "integration_spec": result.integration_spec,
        "checkpoint_step_indices": result.checkpoint_step_indices,
        "checkpoint_epochs": result.checkpoint_epochs,
        "checkpoints": result.checkpoints,
        "force_model_ids": result.force_model_ids,
        "force_ledger": result.force_ledger,
        "completed_steps": result.completed_steps,
        "force_evaluations": result.force_evaluations,
        "direction": result.direction,
        "schedule_content_sha256": result.schedule_content_sha256,
        "minimum_observed_pair_separations": result.minimum_observed_pair_separations,
        "minimum_observed_swept_pair_separation": result.minimum_observed_swept_pair_separation,
        "maximum_observed_pair_frequency_step": result.maximum_observed_pair_frequency_step,
    }
    values.update(changes)
    return _result_content_sha256(**values)  # type: ignore[arg-type]


class KDKContractTests(unittest.TestCase):
    def test_spec_is_closed_signed_and_integer_lattice_only(self):
        forward = kdk_spec()
        backward = kdk_spec((0, 1), -0.01, maximum_steps=1)
        self.assertEqual(forward.direction, "FORWARD")
        self.assertEqual(backward.direction, "BACKWARD")
        self.assertEqual(forward.completed_steps, 64)
        cases = (
            {"checkpoint_step_indices": [0, 1]},
            {"checkpoint_step_indices": (1, 2)},
            {"checkpoint_step_indices": (0, 1, 1)},
            {"checkpoint_step_indices": (0, True)},
            {"fixed_step": 0.0},
            {"fixed_step": 1},
            {"maximum_steps": 63},
            {"minimum_swept_pair_separation": 0.0},
            {"maximum_pair_frequency_step": 1.01},
            {"drift_coefficients": (1,)},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                kdk_spec(**changes)
        with self.assertRaises(ValueError):
            dataclasses.replace(forward, adaptive=True)

    def test_preflight_rejects_ndarray_subclasses_before_state_arithmetic(self):
        class ArraySubclass(np.ndarray):
            pass

        state = binary_state()
        subclass_positions = state.positions.view(ArraySubclass)
        candidate = dataclasses.replace(state, positions=subclass_positions)
        with self.assertRaisesRegex(
            TrajectoryContractError, "input positions must be an exact NumPy ndarray"
        ):
            integrate_kdk_trajectory(candidate, mutual_plan(candidate), kdk_spec())

    def test_preflight_rejects_nonmutual_passive_and_nonnewtonian_plans(self):
        state = binary_state()
        base = mutual_plan(state).models[0]
        passive = dataclasses.replace(
            state,
            gravitational_parameters=np.array((0.5, 0.0)),
            massive=np.array((True, False)),
        )
        cases = (
            (state, mutual_plan(state, model=dataclasses.replace(base, source_ids=("A",)))),
            (state, mutual_plan(state, model=dataclasses.replace(base, target_ids=("B", "A")))),
            (passive, mutual_plan(passive)),
        )
        for candidate, plan in cases:
            with self.subTest(candidate=candidate), self.assertRaises(TrajectoryContractError):
                integrate_kdk_trajectory(candidate, plan, kdk_spec())

        dummy_srp = object.__new__(CannonballSRP)
        object.__setattr__(dummy_srp, "MODEL_ID", CannonballSRP.MODEL_ID)
        with self.assertRaises(TrajectoryContractError):
            integrate_kdk_trajectory(
                state,
                ForcePlan("bad", backend(), (base, dummy_srp)),
                kdk_spec(),
            )

    def test_preflight_rejects_backend_frame_time_axes_and_expired_metadata(self):
        cases = (
            (binary_state(frame="CENTRAL_BODY_INERTIAL"), None),
            (binary_state(origin="SOLAR_SYSTEM_BARYCENTER"), None),
            (binary_state(axes="UNKNOWN"), None),
            (binary_state(time_scale="UTC"), None),
        )
        for state, _ in cases:
            with self.subTest(state=state), self.assertRaises(TrajectoryContractError):
                integrate_kdk_trajectory(state, mutual_plan(state), kdk_spec())
        state = binary_state()
        with self.assertRaisesRegex(TrajectoryContractError, "NumPy CPU"):
            integrate_kdk_trajectory(
                state,
                mutual_plan(state, backend_spec=backend(backend_id="cupy", device="cuda:0")),
                kdk_spec(),
            )
        with self.assertRaisesRegex(TrajectoryContractError, "full KDK schedule"):
            integrate_kdk_trajectory(
                state,
                mutual_plan(state, validity_end=1.0),
                kdk_spec(),
            )

    def test_binary64_schedule_must_advance_before_state_arithmetic(self):
        state = binary_state(epoch=1.0e16)
        before = state.positions.copy()
        with self.assertRaises(TrajectoryStepLimitError):
            integrate_kdk_trajectory(
                state,
                mutual_plan(
                    state,
                    validity_start=1.0e16 - 100.0,
                    validity_end=1.0e16 + 100.0,
                ),
                kdk_spec((0, 1), 0.1, maximum_steps=1),
            )
        np.testing.assert_array_equal(state.positions, before)

    def test_swept_contact_and_pair_frequency_guards_fail_closed(self):
        state = binary_state()
        with self.assertRaisesRegex(TrajectoryDomainError, "pair-frequency"):
            integrate_kdk_trajectory(
                state,
                mutual_plan(state),
                kdk_spec((0, 1), 0.2, maximum_steps=1, maximum_pair_frequency_step=0.1),
            )
        contact = binary_state(radii=np.full(2, 0.5, dtype=np.float64))
        with self.assertRaisesRegex(TrajectoryDomainError, "contact/caller"):
            integrate_kdk_trajectory(
                contact,
                mutual_plan(contact),
                kdk_spec((0, 1), 0.01, maximum_steps=1),
            )

        tunneling = binary_state(
            positions=np.array(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            velocities=np.array(((10.0, 0.0, 0.0), (-10.0, 0.0, 0.0))),
            gravitational_parameters=np.full(2, 1.0e-12, dtype=np.float64),
        )
        with self.assertRaisesRegex(
            TrajectoryDomainError, "swept pair separation.*contact/caller"
        ):
            integrate_kdk_trajectory(
                tunneling,
                mutual_plan(tunneling),
                kdk_spec(
                    (0, 1),
                    0.15,
                    maximum_steps=1,
                    minimum_swept_pair_separation=0.01,
                    maximum_pair_frequency_step=0.1,
                ),
            )


class KDKMapTests(unittest.TestCase):
    def test_one_harmonic_step_has_exact_kick_drift_kick_order(self):
        backend_impl = resolve_backend("numpy")
        positions = np.array(((1.0, 0.0, 0.0),), dtype=np.float64)
        velocities = np.zeros((1, 3), dtype=np.float64)
        acceleration = -positions
        calls = 0

        def acceleration_at(candidate_positions, half_velocities):
            nonlocal calls
            calls += 1
            np.testing.assert_array_equal(
                half_velocities, np.array(((-0.05, 0.0, 0.0),))
            )
            return -candidate_positions

        new_positions, new_velocities, new_acceleration = _kdk_step(
            backend=backend_impl,
            signed_step=0.1,
            positions=positions,
            velocities=velocities,
            acceleration=acceleration,
            acceleration_at_drifted_state=acceleration_at,
        )
        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(new_positions, np.array(((0.995, 0.0, 0.0),)))
        np.testing.assert_array_equal(new_velocities, np.array(((-0.09975, 0.0, 0.0),)))
        np.testing.assert_array_equal(new_acceleration, -new_positions)

    def test_equal_binary_shows_second_order_convergence(self):
        errors = []
        for steps in (32, 64):
            result = integrate(
                kdk_spec(
                    (0, steps),
                    2.0 * math.pi / steps,
                    maximum_steps=steps,
                    maximum_pair_frequency_step=0.25,
                )
            )
            errors.append(float(np.max(np.abs(result.final_positions - binary_state().positions))))
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertLess(errors[0] / errors[1], 4.2)

    def test_forward_then_backward_map_is_reversible_to_roundoff(self):
        steps = 256
        h = 2.0 * math.pi / 128.0
        state = binary_state()
        forward = integrate_kdk_trajectory(
            state,
            mutual_plan(state),
            kdk_spec((0, steps), h, maximum_steps=steps),
        )
        reverse_state = dataclasses.replace(
            state,
            epoch=forward.final_epoch,
            positions=forward.final_positions.copy(),
            velocities=forward.final_velocities.copy(),
        )
        backward = integrate_kdk_trajectory(
            reverse_state,
            mutual_plan(reverse_state),
            kdk_spec((0, steps), -h, maximum_steps=steps),
        )
        np.testing.assert_allclose(backward.final_positions, state.positions, rtol=0.0, atol=8e-14)
        np.testing.assert_allclose(backward.final_velocities, state.velocities, rtol=0.0, atol=8e-14)

    def test_checkpoint_cadence_accounting_momentum_and_custody(self):
        state = binary_state()
        state_arrays = tuple(
            getattr(state, field).copy()
            for field in (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            )
        )
        result = integrate_kdk_trajectory(
            state,
            mutual_plan(state),
            kdk_spec((0, 7, 64), 2.0 * math.pi / 64.0),
        )
        self.assertEqual(result.checkpoint_step_indices, (0, 7, 64))
        self.assertEqual(result.completed_steps, 64)
        self.assertEqual(result.force_evaluations, 65)
        self.assertEqual(tuple(c.accepted_steps for c in result.checkpoints), (0, 7, 64))
        self.assertTrue(result.exact_arithmetic_symplectic)
        self.assertFalse(result.floating_point_symplectic)
        self.assertFalse(result.qualified)
        self.assertFalse(result.registry_authorized)
        self.assertFalse(result.qualification_authorized)
        momentum = np.sum(
            result.initial_snapshot.gravitational_parameters[:, None]
            * result.final_velocities,
            axis=0,
        )
        np.testing.assert_allclose(momentum, np.zeros(3), rtol=0.0, atol=2e-15)
        retained_ids = {
            id(getattr(result.initial_snapshot, field))
            for field in (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            )
        }
        self.assertEqual(len(retained_ids), 6)
        for checkpoint in result.checkpoints:
            self.assertFalse(checkpoint.positions.flags.writeable)
            self.assertFalse(checkpoint.velocities.flags.writeable)
            self.assertNotIn(id(checkpoint.positions), retained_ids)
            self.assertNotIn(id(checkpoint.velocities), retained_ids)
        for field, original in zip(
            (
                "positions",
                "velocities",
                "gravitational_parameters",
                "masses",
                "radii",
                "massive",
            ),
            state_arrays,
        ):
            np.testing.assert_array_equal(getattr(state, field), original)

    def test_checkpoint_cadence_does_not_change_the_fixed_map(self):
        state = binary_state()
        plan = mutual_plan(state)
        steps = 128
        h = 2.0 * math.pi / 128.0
        sparse = integrate_kdk_trajectory(
            state,
            plan,
            kdk_spec((0, steps), h, maximum_steps=steps),
        )
        dense = integrate_kdk_trajectory(
            state,
            plan,
            kdk_spec(tuple(range(0, steps + 1, 8)), h, maximum_steps=steps),
        )
        np.testing.assert_array_equal(sparse.final_positions, dense.final_positions)
        np.testing.assert_array_equal(sparse.final_velocities, dense.final_velocities)
        self.assertEqual(sparse.force_evaluations, dense.force_evaluations)
        self.assertEqual(sparse.force_evaluations, steps + 1)

    def test_bounded_long_arc_oracle_and_invariants(self):
        periods = 10
        steps_per_period = 256
        h = 2.0 * math.pi / steps_per_period
        checkpoints = tuple(
            range(0, periods * steps_per_period + 1, steps_per_period // 4)
        )
        state = binary_state()
        result = integrate_kdk_trajectory(
            state,
            mutual_plan(state),
            kdk_spec(
                checkpoints,
                h,
                maximum_steps=checkpoints[-1],
                maximum_pair_frequency_step=0.05,
            ),
        )
        positions = np.stack(result.positions)
        velocities = np.stack(result.velocities)
        exact_positions = []
        exact_velocities = []
        for epoch in result.checkpoint_epochs:
            cosine, sine = math.cos(epoch), math.sin(epoch)
            exact_positions.append(((-0.5 * cosine, -0.5 * sine, 0.0), (0.5 * cosine, 0.5 * sine, 0.0)))
            exact_velocities.append(((0.5 * sine, -0.5 * cosine, 0.0), (-0.5 * sine, 0.5 * cosine, 0.0)))
        self.assertLessEqual(float(np.max(np.abs(positions - np.array(exact_positions)))), 0.007)
        self.assertLessEqual(float(np.max(np.abs(velocities - np.array(exact_velocities)))), 0.007)
        separation = positions[:, 1] - positions[:, 0]
        relative_velocity = velocities[:, 1] - velocities[:, 0]
        energy = 0.125 * np.sum(relative_velocity * relative_velocity, axis=1) - 0.25 / np.linalg.norm(separation, axis=1)
        angular = 0.25 * np.cross(separation, relative_velocity)[:, 2]
        relative_energy_drift = float(np.max(np.abs((energy + 0.125) / 0.125)))
        self.assertGreater(relative_energy_drift, 1e-8)
        self.assertLessEqual(relative_energy_drift, 2e-7)
        self.assertLessEqual(float(np.max(np.abs((angular - 0.25) / 0.25))), 1e-12)
        self.assertEqual(result.force_evaluations, checkpoints[-1] + 1)


class KDKResultIntegrityTests(unittest.TestCase):
    def test_schedule_checksum_has_independent_literal_known_answer(self):
        payload = (
            b'{"checkpoint_epochs_hex":["0x0.0p+0","0x1.0000000000000p-2"],'
            b'"checkpoint_step_indices":[0,2],'
            b'"checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1",'
            b'"completed_steps":2,"direction":"FORWARD",'
            b'"fixed_step_hex":"0x1.0000000000000p-3","force_evaluations":3,'
            b'"method_id":"integrator.symplectic.kdk_leapfrog_2"}'
        )
        preimage = b"jxplanetx.kdk-schedule.content-integrity.v1\x00" + payload
        expected = "ab02b920c2c080159b9745b1a6538420307ed77d5213d0704e949554f3ee16cd"
        self.assertEqual(hashlib.sha256(preimage).hexdigest(), expected)
        self.assertEqual(
            _schedule_content_sha256(
                checkpoint_step_indices=(0, 2),
                checkpoint_epochs=(0.0, 0.25),
                fixed_step=0.125,
                direction="FORWARD",
                completed_steps=2,
                force_evaluations=3,
            ),
            expected,
        )

    def test_result_rejects_checksum_accounting_guard_and_contact_tampering(self):
        result = integrate()
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(result, schedule_content_sha256="0" * 64)
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(result, force_evaluations=result.force_evaluations + 1)
        with self.assertRaises(TrajectoryContractError):
            dataclasses.replace(
                result,
                minimum_observed_swept_pair_separation=0.05,
            )
        altered_positions = result.final_positions.copy()
        altered_positions[0, 0] += 0.25
        altered_positions.setflags(write=False)
        altered_final = dataclasses.replace(
            result.checkpoints[-1], positions=altered_positions
        )
        with self.assertRaisesRegex(TrajectoryContractError, "result content checksum"):
            dataclasses.replace(
                result,
                checkpoints=result.checkpoints[:-1] + (altered_final,),
            )
        ledger = result.force_ledger[0]
        ledger_attacks = (
            dataclasses.replace(
                ledger,
                state_metadata=dataclasses.replace(
                    ledger.state_metadata, frame="ALTERED_FRAME"
                ),
            ),
            dataclasses.replace(ledger, tile_size=ledger.tile_size + 1),
            dataclasses.replace(ledger, determinism_scope="ALTERED_SCOPE"),
            dataclasses.replace(ledger, evidence_class="ALTERED_EVIDENCE"),
        )
        for altered_ledger in ledger_attacks:
            with self.subTest(altered_ledger=altered_ledger), self.assertRaises(
                TrajectoryContractError
            ):
                dataclasses.replace(result, force_ledger=(altered_ledger,))
        radii = np.full(2, 0.6, dtype=np.float64)
        radii.setflags(write=False)
        altered_snapshot = dataclasses.replace(result.initial_snapshot, radii=radii)
        with self.assertRaisesRegex(TrajectoryContractError, "contact/caller floor"):
            dataclasses.replace(result, initial_snapshot=altered_snapshot)

        shared_view = result.initial_snapshot.positions.view()
        self.assertFalse(shared_view.flags.writeable)
        self.assertTrue(
            np.shares_memory(shared_view, result.initial_snapshot.positions)
        )
        aliased_initial = dataclasses.replace(
            result.checkpoints[0], positions=shared_view
        )
        with self.assertRaisesRegex(TrajectoryContractError, "overlap retained memory"):
            dataclasses.replace(
                result,
                checkpoints=(aliased_initial,) + result.checkpoints[1:],
            )

        external_owner = result.final_positions.copy()
        external_view = external_owner.view()
        external_view.setflags(write=False)
        self.assertFalse(external_view.flags.owndata)
        self.assertFalse(
            any(
                np.shares_memory(external_view, retained)
                for retained in (
                    result.initial_snapshot.positions,
                    result.initial_snapshot.velocities,
                    result.final_positions,
                    result.final_velocities,
                )
            )
        )
        external_final = dataclasses.replace(
            result.checkpoints[-1], positions=external_view
        )
        with self.assertRaisesRegex(TrajectoryContractError, "must own their memory"):
            dataclasses.replace(
                result,
                checkpoints=result.checkpoints[:-1] + (external_final,),
            )

        metadata_namespace = SimpleNamespace(
            **{
                field.name: getattr(result.force_ledger[0].state_metadata, field.name)
                for field in dataclasses.fields(result.force_ledger[0].state_metadata)
            }
        )
        altered_ledger = dataclasses.replace(
            result.force_ledger[0], state_metadata=metadata_namespace
        )
        with self.assertRaisesRegex(TrajectoryContractError, "exact StateMetadataBinding"):
            dataclasses.replace(result, force_ledger=(altered_ledger,))

    def test_result_rejects_equal_value_noncanonical_dynamic_types(self):
        class TextSubclass(str):
            pass

        class TupleSubclass(tuple):
            pass

        class ResultSubclass(KDKTrajectoryResult):
            pass

        result = integrate(
            kdk_spec(
                (0, 1),
                0.125,
                maximum_steps=1,
                maximum_pair_frequency_step=0.2,
            )
        )
        attacks = (
            lambda: dataclasses.replace(result, completed_steps=True),
            lambda: dataclasses.replace(result, direction=TextSubclass(result.direction)),
            lambda: dataclasses.replace(
                result,
                checkpoint_step_indices=TupleSubclass(result.checkpoint_step_indices),
            ),
            lambda: dataclasses.replace(
                result,
                force_ledger=(
                    dataclasses.replace(result.force_ledger[0], order=False),
                ),
            ),
            lambda: dataclasses.replace(
                result,
                force_ledger=(
                    dataclasses.replace(
                        result.force_ledger[0], registry_authorized=0
                    ),
                ),
            ),
            lambda: dataclasses.replace(
                result,
                checkpoints=(
                    dataclasses.replace(result.checkpoints[0], epoch=0),
                )
                + result.checkpoints[1:],
            ),
            lambda: dataclasses.replace(
                result,
                initial_snapshot=dataclasses.replace(
                    result.initial_snapshot, epoch=0
                ),
            ),
            lambda: ResultSubclass(
                **{
                    field.name: getattr(result, field.name)
                    for field in dataclasses.fields(result)
                }
            ),
        )
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(TrajectoryContractError):
                attack()

    def test_result_content_checksum_has_independent_literal_known_answer(self):
        preimage = b'jxplanetx.kdk-result.content-integrity.v1\x00{"backend":{"allow_fallback":false,"backend_id":"numpy","determinism_scope":"SAME_RUNTIME_DEVICE","deterministic_reductions":true,"device":"cpu","dtype":"float64","fast_math":false,"tile_size":2},"checkpoint_states":[{"accepted_steps":0,"epoch_hex":"0x0.0p+0","index":0,"positions_hex":[["-0x1.0000000000000p-1","0x0.0p+0","0x0.0p+0"],["0x1.0000000000000p-1","0x0.0p+0","0x0.0p+0"]],"rejected_steps":0,"step_index":0,"velocities_hex":[["0x0.0p+0","-0x1.0000000000000p-1","0x0.0p+0"],["0x0.0p+0","0x1.0000000000000p-1","0x0.0p+0"]]},{"accepted_steps":1,"epoch_hex":"0x1.0000000000000p-3","index":1,"positions_hex":[["-0x1.fc00000000000p-2","-0x1.0000000000000p-4","0x0.0p+0"],["0x1.fc00000000000p-2","0x1.0000000000000p-4","0x0.0p+0"]],"rejected_steps":0,"step_index":1,"velocities_hex":[["0x1.fdfa0c1dc3752p-5","-0x1.fc0017ff88023p-2","0x0.0p+0"],["-0x1.fdfa0c1dc3752p-5","0x1.fc0017ff88023p-2","0x0.0p+0"]]}],"checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","control":{"adaptive":false,"dense_output":false,"evidence_class":"MODEL_OUTPUT","exact_arithmetic_symplectic":true,"exact_arithmetic_time_reversible":true,"floating_point_symplectic":false,"qualification_authorized":false,"registry_authorized":false,"scope":"FIXED_STEP_KDK_TRAJECTORY_INTEGRATION"},"force_ledger":[{"assumptions":["direct unsoftened point masses","finite-radius contact and singular coincidence are errors","repeatability is limited to the same backend, device, software stack, force order, body order, and tile size; no cross-device bitwise guarantee"],"backend":{"backend_id":"numpy","device":"cpu","dtype":"float64","tile_size":2},"determinism_scope":"SAME_RUNTIME_DEVICE","evidence_class":"MODEL_OUTPUT","model_id":"force.newtonian.point_mass","order":0,"qualification_authorized":false,"registry_authorized":false,"role":"NEWTONIAN_BASE","source_ids":["A","B"],"state_metadata":{"axes":"ICRS_ALIGNED","body_ids":["A","B"],"epoch_hex":"0x0.0p+0","frame":"BARYCENTRIC_INERTIAL","length_unit":"L","mass_unit":"M","origin":"BARYCENTER","provenance_citation":"Synthetic KDK fixture","provenance_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","provenance_source_id":"fixture.kdk","provenance_version":"1","snapshot_id":"fixture.kdk.equal-binary","time_scale":"TDB","time_unit":"T","unit_system_id":"fixture.kdk.units"},"target_ids":["A","B"],"tile_size":2}],"force_plan":{"evidence_class":"MODEL_OUTPUT","model":{"model_id":"force.newtonian.point_mass","parameter_metadata":[{"covariance_group":null,"parameter_id":"state.gravitational_parameters","provenance":{"citation":"Synthetic KDK fixture","sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","source_id":"fixture.kdk","version":"1"},"uncertainty_hex":null,"units":"L^3/T^2","validity_end_hex":"0x1.3880000000000p+13","validity_start_hex":"-0x1.3880000000000p+13"}],"source_ids":["A","B"],"target_ids":["A","B"],"unit_system_id":"fixture.kdk.units"},"plan_id":"fixture.kdk.plan","qualification_authorized":false,"registry_authorized":false},"guards":{"maximum_observed_pair_frequency_step_hex":"0x1.00bfe809fa63bp-3","minimum_observed_pair_separations_hex":["0x1.ff00bf608b827p-1"],"minimum_observed_swept_pair_separation_hex":"0x1.ff00bf608b827p-1"},"integration_spec":{"checkpoint_step_indices":[0,1],"encounter_guard":"LINEAR_DRIFT_PAIRWISE_MINIMUM_SEPARATION_STRICTLY_ABOVE_MAX_CONTACT_AND_CALLER_FLOOR","fixed_step_hex":"0x1.0000000000000p-3","force_plan_scope":"MUTUAL_ALL_BODY_NEWTONIAN_POINT_MASS_ONLY","maximum_pair_frequency_step_hex":"0x1.999999999999ap-3","maximum_steps":1,"minimum_swept_pair_separation_hex":"0x1.999999999999ap-4","pair_frequency_guard":"ABS_H_TIMES_SQRT_PAIR_GM_OVER_SWEPT_MINIMUM_SEPARATION_CUBED_LESS_THAN_OR_EQUAL_TO_CALLER_LIMIT"},"method":{"backend_scope":"NUMPY_CPU_ONLY","checkpoint_policy":"INTEGER_STEP_LATTICE_NO_CLIPPING_NO_INTERPOLATION","composition":"KICK_HALF_DRIFT_FULL_KICK_HALF","drift_coefficients_hex":["0x1.0000000000000p+0"],"force_evaluation_accounting":"ONE_INITIAL_PLUS_ONE_PER_COMPLETED_MAP_STEP","kick_coefficients_hex":["0x1.0000000000000p-1","0x1.0000000000000p-1"],"method_class":"EXPLICIT_SYMMETRIC_KICK_DRIFT_KICK_SPLITTING","method_id":"integrator.symplectic.kdk_leapfrog_2","principal_order":2,"step_representation":"CONSTANT_SIGNED_BINARY64_MAP_STEP","time_semantics":"CONTINUOUS_COORDINATE_TIME"},"runtime":{"checkpoint_epochs_hex":["0x0.0p+0","0x1.0000000000000p-3"],"completed_steps":1,"direction":"FORWARD","force_evaluations":2,"force_model_ids":["force.newtonian.point_mass"],"schedule_content_sha256":"e25a51ce3c62a7d9a4fb8d18e7554fff1e41da67fcf5d48ff46f9bb0989d9e94"},"snapshot":{"axes":"ICRS_ALIGNED","body_ids":["A","B"],"epoch_hex":"0x0.0p+0","frame":"BARYCENTRIC_INERTIAL","gravitational_parameters_hex":["0x1.0000000000000p-1","0x1.0000000000000p-1"],"length_unit":"L","mass_unit":"M","masses_hex":["0x1.0000000000000p-1","0x1.0000000000000p-1"],"massive":[true,true],"origin":"BARYCENTER","positions_hex":[["-0x1.0000000000000p-1","0x0.0p+0","0x0.0p+0"],["0x1.0000000000000p-1","0x0.0p+0","0x0.0p+0"]],"provenance":{"citation":"Synthetic KDK fixture","sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","source_id":"fixture.kdk","version":"1"},"radii_hex":["0x0.0p+0","0x0.0p+0"],"snapshot_id":"fixture.kdk.equal-binary","time_scale":"TDB","time_unit":"T","unit_system_id":"fixture.kdk.units","velocities_hex":[["0x0.0p+0","-0x1.0000000000000p-1","0x0.0p+0"],["0x0.0p+0","0x1.0000000000000p-1","0x0.0p+0"]]}}'
        expected = "8b772531a0092e8493537533d351a2b7463618dad383b9fa143b03417e1e8ef8"
        self.assertEqual(len(preimage), 5555)
        self.assertEqual(hashlib.sha256(preimage).hexdigest(), expected)
        state = binary_state()
        result = integrate_kdk_trajectory(
            state,
            mutual_plan(state),
            kdk_spec(
                (0, 1),
                0.125,
                maximum_steps=1,
                maximum_pair_frequency_step=0.2,
            ),
        )
        self.assertEqual(result.result_content_sha256, expected)

    def test_coherent_checksum_rewrite_cannot_rescue_invalid_retained_semantics(self):
        result = integrate()
        model = result.force_plan.models[0]
        retained_metadata = model.parameter_metadata[0]
        altered_models = (
            dataclasses.replace(model, unit_system_id="altered.units"),
            dataclasses.replace(
                model,
                parameter_metadata=(
                    dataclasses.replace(retained_metadata, units="WRONG"),
                ),
            ),
            dataclasses.replace(
                model,
                parameter_metadata=(
                    dataclasses.replace(
                        retained_metadata,
                        validity_start=-2.0,
                        validity_end=-1.0,
                    ),
                ),
            ),
        )
        for altered_model in altered_models:
            altered_plan = dataclasses.replace(
                result.force_plan, models=(altered_model,)
            )
            coherent_digest = recomputed_result_digest(
                result, force_plan=altered_plan
            )
            with self.subTest(altered_model=altered_model), self.assertRaises(
                ValueError
            ):
                dataclasses.replace(
                    result,
                    force_plan=altered_plan,
                    result_content_sha256=coherent_digest,
                )

        altered_ledger = dataclasses.replace(
            result.force_ledger[0], assumptions=("invented assumption",)
        )
        coherent_digest = recomputed_result_digest(
            result, force_ledger=(altered_ledger,)
        )
        with self.assertRaisesRegex(TrajectoryContractError, "force-ledger binding"):
            dataclasses.replace(
                result,
                force_ledger=(altered_ledger,),
                result_content_sha256=coherent_digest,
            )

    def test_per_pair_contact_revalidation_does_not_mix_unrelated_extrema(self):
        state = binary_state(
            body_ids=("A", "B", "C"),
            positions=np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1000.0, 0.0, 0.0))),
            velocities=np.zeros((3, 3), dtype=np.float64),
            gravitational_parameters=np.full(3, 1.0e-12, dtype=np.float64),
            masses=np.ones(3, dtype=np.float64),
            radii=np.array((0.0, 0.0, 100.0), dtype=np.float64),
            massive=np.ones(3, dtype=np.bool_),
        )
        result = integrate_kdk_trajectory(
            state,
            mutual_plan(state),
            kdk_spec(
                (0, 1),
                0.001,
                maximum_steps=1,
                minimum_swept_pair_separation=0.01,
                maximum_pair_frequency_step=0.1,
            ),
        )
        self.assertEqual(len(result.minimum_observed_pair_separations), 3)
        closest_radii = np.array((0.6, 0.6, 100.0), dtype=np.float64)
        closest_radii.setflags(write=False)
        altered_snapshot = dataclasses.replace(
            result.initial_snapshot, radii=closest_radii
        )
        with self.assertRaisesRegex(TrajectoryContractError, "contact/caller floor"):
            dataclasses.replace(result, initial_snapshot=altered_snapshot)


if __name__ == "__main__":
    unittest.main()
