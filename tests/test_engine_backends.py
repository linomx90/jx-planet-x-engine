import dataclasses
import importlib
import sys
import unittest
from unittest import mock

import numpy as np

from jxplanetx.engine import (
    BackendArrayError,
    BackendError,
    BackendSpec,
    BackendUnavailableError,
    CannonballSRP,
    ContractError,
    EvaluationError,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    StateSnapshot,
    evaluate,
)
from jxplanetx.engine.backends import resolve_backend


UNIT_SYSTEM_ID = "fixture.au_day_solar_mass"


def provenance() -> Provenance:
    return Provenance("fixture.backend", "Synthetic backend fixture", "1", "d" * 64)


def metadata(parameter_id: str, units: str) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id, units, provenance(), None, None, -1.0, 1.0
    )


def backend_spec(
    backend_id: str = "numpy",
    device: str = "cpu",
    **changes: object,
) -> BackendSpec:
    values: dict[str, object] = {
        "backend_id": backend_id,
        "device": device,
        "tile_size": 2,
        "dtype": "float64",
        "allow_fallback": False,
        "deterministic_reductions": True,
        "fast_math": False,
        "determinism_scope": "SAME_RUNTIME_DEVICE",
    }
    values.update(changes)
    return BackendSpec(**values)  # type: ignore[arg-type]


def state_and_plan(xp, backend_id: str, device: str):
    state = StateSnapshot(
        snapshot_id="fixture.backend.snapshot",
        epoch=0.0,
        time_scale="TDB",
        frame="CENTRAL_BODY_INERTIAL",
        origin="SUN",
        axes="ICRS_ALIGNED",
        length_unit="AU",
        time_unit="DAY",
        mass_unit="SOLAR_MASS",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=("SUN", "REL", "SRP"),
        positions=xp.asarray(
            ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=xp.float64,
        ),
        velocities=xp.asarray(
            ((0.0, 0.0, 0.0), (2.0, -1.0, 1.0), (0.0, 0.0, 0.0)),
            dtype=xp.float64,
        ),
        gravitational_parameters=xp.asarray((7.0, 0.0, 0.0), dtype=xp.float64),
        masses=xp.asarray((1.0, 0.0, 0.0), dtype=xp.float64),
        radii=xp.zeros(3, dtype=xp.float64),
        massive=xp.asarray((True, False, False), dtype=xp.bool_),
        provenance=provenance(),
    )
    newtonian = NewtonianPointMass(
        ("SUN",),
        ("SUN", "REL", "SRP"),
        UNIT_SYSTEM_ID,
        (metadata("state.gravitational_parameters", "AU^3/DAY^2"),),
    )
    relativity = RestrictedStaticCentral1PN(
        "SUN",
        ("REL",),
        100.0,
        0.1,
        0.1,
        UNIT_SYSTEM_ID,
        (
            metadata("speed_of_light", "AU/DAY"),
            metadata("maximum_compactness", "1"),
            metadata("maximum_speed_fraction_squared", "1"),
        ),
    )
    srp = CannonballSRP(
        "SUN",
        ("SRP",),
        5.0,
        2.0,
        xp.asarray((3.0,), dtype=xp.float64),
        xp.asarray((2.0,), dtype=xp.float64),
        "QPR",
        "ISOTROPIC_CANNONBALL",
        "NONE",
        UNIT_SYSTEM_ID,
        (
            metadata("reference_pressure", "SOLAR_MASS/(AU*DAY^2)"),
            metadata("reference_distance", "AU"),
            metadata("area_to_mass", "AU^2/SOLAR_MASS"),
            metadata("radiation_pressure_coefficient", "1"),
        ),
    )
    plan = ForcePlan(
        "fixture.backend.plan",
        backend_spec(backend_id, device),
        (newtonian, relativity, srp),
    )
    return state, plan


class BackendSelectionTests(unittest.TestCase):
    def test_numpy_backend_is_explicit_cpu_float64(self):
        resolved = resolve_backend(backend_spec())
        self.assertEqual(resolved.name, "numpy")
        self.assertEqual(resolved.device, "cpu")
        self.assertIs(resolved.array_type, np.ndarray)
        self.assertIs(resolved.float64, np.float64)

    def test_backend_identifiers_are_exact_lowercase_and_never_auto_selected(self):
        for requested in ("NumPy", "NUMPY", "CuPy", "CUPY", "auto", "cuda", ""):
            with self.subTest(requested=requested), self.assertRaises(BackendError):
                resolve_backend(requested)
        with self.assertRaises(BackendError):
            resolve_backend(None)

    def test_numpy_rejects_non_cpu_and_weak_arithmetic_policies(self):
        with self.assertRaises(BackendError):
            resolve_backend(backend_spec(device="cuda:0"))
        with self.assertRaises(ContractError):
            backend_spec(deterministic_reductions=False)
        for dtype in ("float32", "FLOAT64", np.dtype("float64")):
            with self.subTest(dtype=dtype), self.assertRaises(ContractError):
                backend_spec(dtype=dtype)

    def test_explicit_cupy_request_without_cupy_fails_without_numpy_fallback(self):
        state, plan = state_and_plan(np, "cupy", "cuda:0")
        originals = state.positions.copy(), state.velocities.copy()
        with mock.patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaisesRegex(BackendUnavailableError, "will not fall back"):
                evaluate(state, plan)
        np.testing.assert_array_equal(state.positions, originals[0])
        np.testing.assert_array_equal(state.velocities, originals[1])

    def test_numpy_runtime_rejects_non_native_arrays_instead_of_coercing(self):
        state, plan = state_and_plan(np, "numpy", "cpu")
        wrong = dataclasses.replace(state, positions=state.positions.tolist())
        with self.assertRaisesRegex(BackendArrayError, "never performs an implicit"):
            evaluate(wrong, plan)

    def test_float32_state_and_force_buffers_are_never_accepted(self):
        state, plan = state_and_plan(np, "numpy", "cpu")
        with self.assertRaises(EvaluationError):
            evaluate(
                dataclasses.replace(state, positions=state.positions.astype(np.float32)),
                plan,
            )
        srp = plan.models[2]
        wrong_srp = dataclasses.replace(
            srp, area_to_mass=srp.area_to_mass.astype(np.float32)
        )
        with self.assertRaises(EvaluationError):
            evaluate(
                state,
                dataclasses.replace(plan, models=(plan.models[0], plan.models[1], wrong_srp)),
            )


class OptionalCuPyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module("cupy")
            count = int(cls.cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            raise unittest.SkipTest(f"CuPy GPU runtime unavailable: {exc}") from exc
        if count < 1:
            raise unittest.SkipTest("CuPy found no GPU")

    def test_all_three_force_terms_match_numpy_and_remain_device_native(self):
        cpu_state, cpu_plan = state_and_plan(np, "numpy", "cpu")
        expected = evaluate(cpu_state, cpu_plan)

        cp = self.cp
        with cp.cuda.Device(0):
            gpu_state, gpu_plan = state_and_plan(cp, "cupy", "cuda:0")
            # Any internal call to this explicit transfer helper is a test
            # failure.  Conversion below happens only after evaluation.
            with mock.patch.object(
                cp,
                "asnumpy",
                side_effect=AssertionError("implicit device-to-host transfer"),
            ):
                actual = evaluate(gpu_state, gpu_plan)

            self.assertEqual(actual.backend_id, "cupy")
            self.assertEqual(actual.device, "cuda:0")
            self.assertIsInstance(actual.total_acceleration, cp.ndarray)
            self.assertEqual(int(actual.total_acceleration.device.id), 0)
            for contribution in actual.contributions:
                self.assertIsInstance(contribution.acceleration, cp.ndarray)
                self.assertEqual(int(contribution.acceleration.device.id), 0)

            np.testing.assert_allclose(
                cp.asnumpy(actual.total_acceleration),
                expected.total_acceleration,
                rtol=5e-14,
                atol=2e-15,
            )
            for gpu_term, cpu_term in zip(actual.contributions, expected.contributions):
                np.testing.assert_allclose(
                    cp.asnumpy(gpu_term.acceleration),
                    cpu_term.acceleration,
                    rtol=5e-14,
                    atol=2e-15,
                )

    def test_mixed_host_device_inputs_fail_instead_of_transferring(self):
        cp = self.cp
        cpu_state, cpu_plan = state_and_plan(np, "numpy", "cpu")
        with cp.cuda.Device(0):
            gpu_state, gpu_plan = state_and_plan(cp, "cupy", "cuda:0")
            with self.assertRaises(BackendArrayError):
                evaluate(cpu_state, gpu_plan)
            with self.assertRaises(BackendArrayError):
                evaluate(gpu_state, cpu_plan)

    def test_cupy_float32_is_rejected(self):
        cp = self.cp
        with cp.cuda.Device(0):
            state, plan = state_and_plan(cp, "cupy", "cuda:0")
            with self.assertRaises(EvaluationError):
                evaluate(
                    dataclasses.replace(
                        state, velocities=state.velocities.astype(cp.float32)
                    ),
                    plan,
                )


if __name__ == "__main__":
    unittest.main()
