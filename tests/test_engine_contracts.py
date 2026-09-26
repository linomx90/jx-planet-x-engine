import dataclasses
import math
import unittest

import numpy as np

from jxplanetx.engine.contracts import (
    BackendSpec,
    CannonballSRP,
    ContractError,
    ForcePlan,
    MutualEIH1PN,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    StateSnapshot,
)


SHA256 = "a" * 64
UNIT_SYSTEM_ID = "fixture.au_day_solar_mass"


def provenance() -> Provenance:
    return Provenance(
        source_id="fixture.state",
        citation="Synthetic contract fixture",
        version="1",
        sha256=SHA256,
    )


def metadata(parameter_id: str, units: str) -> ParameterMetadata:
    return ParameterMetadata(
        parameter_id,
        units,
        provenance(),
        None,
        None,
        -1.0,
        1.0,
    )


def newtonian_metadata() -> tuple[ParameterMetadata, ...]:
    return (
        metadata(
            "state.gravitational_parameters",
            "STATE_LENGTH_UNIT^3/STATE_TIME_UNIT^2",
        ),
    )


def one_pn_metadata() -> tuple[ParameterMetadata, ...]:
    return (
        metadata("speed_of_light", "STATE_LENGTH_UNIT/STATE_TIME_UNIT"),
        metadata("maximum_compactness", "1"),
        metadata("maximum_speed_fraction_squared", "1"),
    )


def srp_metadata() -> tuple[ParameterMetadata, ...]:
    return (
        metadata(
            "reference_pressure",
            "STATE_MASS_UNIT/(STATE_LENGTH_UNIT*STATE_TIME_UNIT^2)",
        ),
        metadata("reference_distance", "STATE_LENGTH_UNIT"),
        metadata("area_to_mass", "STATE_LENGTH_UNIT^2/STATE_MASS_UNIT"),
        metadata("radiation_pressure_coefficient", "1"),
    )


def state(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "fixture.snapshot",
        "epoch": 0.0,
        "time_scale": "TDB",
        "frame": "ICRS",
        "origin": "SOLAR_SYSTEM_BARYCENTER",
        "axes": "ICRS",
        "length_unit": "AU",
        "time_unit": "DAY",
        "mass_unit": "SOLAR_MASS",
        "unit_system_id": UNIT_SYSTEM_ID,
        "body_ids": ("SUN", "TEST"),
        "positions": np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        "velocities": np.array(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        "gravitational_parameters": np.array((1.0, 0.0)),
        "masses": np.array((1.0, 0.0)),
        "radii": np.array((0.01, 0.0)),
        "massive": np.array((True, False)),
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
    }
    values.update(changes)
    return BackendSpec(**values)  # type: ignore[arg-type]


class ProvenanceAndParameterContractTests(unittest.TestCase):
    def test_provenance_requires_exact_lowercase_sha256_and_trimmed_text(self):
        self.assertEqual(provenance().sha256, SHA256)
        for digest in ("A" * 64, "a" * 63, "g" * 64, 0):
            with self.subTest(digest=digest), self.assertRaises(ContractError):
                Provenance("source", "citation", "1", digest)  # type: ignore[arg-type]
        for field in ("source_id", "citation", "version"):
            with self.subTest(field=field), self.assertRaises(ContractError):
                dataclasses.replace(provenance(), **{field: " untrimmed"})

    def test_parameter_metadata_is_finite_nonnegative_and_ordered(self):
        valid = ParameterMetadata(
            parameter_id="speed_of_light",
            units="AU/DAY",
            provenance=provenance(),
            uncertainty=0.0,
            covariance_group="constants",
            validity_start=-1.0,
            validity_end=1.0,
        )
        self.assertEqual(valid.uncertainty, 0.0)
        for uncertainty in (-1.0, math.nan, math.inf, True):
            with self.subTest(uncertainty=uncertainty), self.assertRaises(ContractError):
                dataclasses.replace(valid, uncertainty=uncertainty)
        with self.assertRaises(ContractError):
            dataclasses.replace(valid, validity_start=2.0, validity_end=1.0)
        with self.assertRaises(ContractError):
            dataclasses.replace(valid, covariance_group="")


class BackendContractTests(unittest.TestCase):
    def test_initial_backend_contract_is_binary64_without_fallback_or_fast_math(self):
        contract = backend()
        self.assertEqual(contract.dtype, "float64")
        self.assertFalse(contract.allow_fallback)
        self.assertTrue(contract.deterministic_reductions)
        self.assertFalse(contract.fast_math)

        invalid = (
            {"dtype": "float32"},
            {"dtype": "FLOAT64"},
            {"allow_fallback": True},
            {"allow_fallback": 0},
            {"deterministic_reductions": 1},
            {"fast_math": True},
            {"fast_math": 0},
            {"tile_size": 0},
            {"tile_size": True},
            {"determinism_scope": "CROSS_DEVICE_BITWISE"},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ContractError):
                backend(**change)

    def test_backend_and_device_are_explicit_nonempty_trimmed_strings(self):
        for field in ("backend_id", "device"):
            for value in ("", " cpu", "cpu ", None):
                with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                    backend(**{field: value})


class StateSnapshotContractTests(unittest.TestCase):
    def test_valid_snapshot_is_immutable_and_has_stable_body_lookup(self):
        snapshot = state()
        self.assertEqual(snapshot.index_of("TEST"), 1)
        with self.assertRaises(ContractError):
            snapshot.index_of("MISSING")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.epoch = 1.0  # type: ignore[misc]

    def test_array_bearing_snapshot_uses_identity_equality_and_hashing(self):
        first = state()
        second = state()
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)
        self.assertIsInstance(hash(first), int)

    def test_body_ids_are_exact_but_numeric_buffers_are_deferred_to_the_backend(self):
        invalid_ids: tuple[object, ...] = (
            ["SUN", "TEST"],
            ("SUN", "SUN"),
            (),
        )
        for body_ids in invalid_ids:
            with self.subTest(body_ids=body_ids), self.assertRaises(ContractError):
                state(body_ids=body_ids)

        for field in (
            "positions",
            "velocities",
            "gravitational_parameters",
            "masses",
            "radii",
            "massive",
        ):
            with self.subTest(field=field), self.assertRaises(ContractError):
                state(**{field: None})

        # Descriptor construction must not inspect, materialize, or transfer a
        # backend buffer.  Shape/dtype/finiteness are checked by evaluation.
        opaque = object()
        self.assertIs(state(positions=opaque).positions, opaque)

    def test_snapshot_scalar_context_rejects_invalid_values(self):
        invalid: tuple[dict[str, object], ...] = (
            {"body_ids": ["SUN", "TEST"]},
            {"body_ids": ("SUN", "SUN")},
            {"body_ids": ()},
            {"epoch": math.nan},
            {"epoch": math.inf},
            {"epoch": True},
            {"time_scale": ""},
            {"frame": " ICRS"},
            {"origin": None},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ContractError):
                state(**change)


class ForceConfigurationContractTests(unittest.TestCase):
    def test_mutual_eih_requires_full_identity_and_explicit_domain_bounds(self):
        model = MutualEIH1PN(
            body_ids=("SUN", "EARTH"),
            speed_of_light=299792.458,
            maximum_compactness=1.0e-4,
            maximum_speed_fraction_squared=1.0e-4,
            unit_system_id=UNIT_SYSTEM_ID,
            parameter_metadata=one_pn_metadata(),
        )
        self.assertEqual(model.model_id, "force.relativity.eih_1pn_gr")
        for change in (
            {"body_ids": ("SUN",)},
            {"body_ids": ("SUN", "SUN")},
            {"speed_of_light": 0.0},
            {"maximum_compactness": 1.0},
            {"maximum_speed_fraction_squared": 1.0},
            {"unit_system_id": ""},
            {"parameter_metadata": one_pn_metadata()[::-1]},
        ):
            with self.subTest(change=change), self.assertRaises(ContractError):
                dataclasses.replace(model, **change)

    def test_newtonian_selection_and_parameter_metadata_are_exact(self):
        gm_metadata = metadata(
            "state.gravitational_parameters",
            "STATE_LENGTH_UNIT^3/STATE_TIME_UNIT^2",
        )
        model = NewtonianPointMass(
            source_ids=("SUN",),
            target_ids=("SUN", "TEST"),
            unit_system_id=UNIT_SYSTEM_ID,
            parameter_metadata=newtonian_metadata(),
        )
        self.assertEqual(model.model_id, "force.newtonian.point_mass")

        for field, value in (
            ("source_ids", ()),
            ("target_ids", ()),
            ("source_ids", ("SUN", "SUN")),
            ("target_ids", ["TEST"]),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                dataclasses.replace(model, **{field: value})

        wrong = dataclasses.replace(gm_metadata, parameter_id="softening_length")
        with self.assertRaises(ContractError):
            dataclasses.replace(model, parameter_metadata=(wrong,))
        with self.assertRaises(ContractError):
            dataclasses.replace(model, parameter_metadata=())

    def test_restricted_1pn_requires_disjoint_scope_and_explicit_domain_bounds(self):
        model = RestrictedStaticCentral1PN(
            central_source_id="SUN",
            target_ids=("TEST",),
            speed_of_light=100.0,
            maximum_compactness=0.1,
            maximum_speed_fraction_squared=0.1,
            unit_system_id=UNIT_SYSTEM_ID,
            parameter_metadata=one_pn_metadata(),
        )
        self.assertEqual(
            model.model_id,
            "relativity.solar_schwarzschild_test_particle_1pn",
        )
        invalid = (
            {"target_ids": ("SUN",)},
            {"speed_of_light": 0.0},
            {"speed_of_light": math.inf},
            {"maximum_compactness": 0.0},
            {"maximum_compactness": 1.0},
            {"maximum_speed_fraction_squared": -0.1},
            {"maximum_speed_fraction_squared": 1.0},
            {"unit_system_id": ""},
            {"parameter_metadata": one_pn_metadata()[:-1]},
            {"parameter_metadata": tuple(reversed(one_pn_metadata()))},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ContractError):
                dataclasses.replace(model, **change)

    def test_srp_requires_explicit_convention_attitude_shadow_and_aligned_parameters(self):
        model = CannonballSRP(
            radiation_source_id="SUN",
            target_ids=("TEST",),
            reference_pressure=1.0,
            reference_distance=1.0,
            area_to_mass=np.array((2.0,)),
            radiation_pressure_coefficient=np.array((1.5,)),
            coefficient_convention="QPR",
            attitude_model="ISOTROPIC_CANNONBALL",
            shadow_model="NONE",
            unit_system_id=UNIT_SYSTEM_ID,
            parameter_metadata=srp_metadata(),
        )
        self.assertEqual(model.model_id, "force.nongrav.srp_cannonball")
        invalid = (
            {"target_ids": ("SUN",)},
            {"reference_pressure": 0.0},
            {"reference_distance": math.nan},
            {"area_to_mass": None},
            {"radiation_pressure_coefficient": None},
            {"coefficient_convention": "UNKNOWN"},
            {"attitude_model": "SPACECRAFT_ATTITUDE"},
            {"shadow_model": "UMBRA"},
            {"unit_system_id": ""},
            {"parameter_metadata": srp_metadata()[:-1]},
            {"parameter_metadata": tuple(reversed(srp_metadata()))},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ContractError):
                dataclasses.replace(model, **change)
        self.assertEqual(model, model)
        self.assertNotEqual(model, dataclasses.replace(model))
        self.assertIsInstance(hash(model), int)

    def test_force_plan_preserves_order_rejects_duplicates_and_cannot_claim_authority(self):
        newtonian = NewtonianPointMass(
            ("SUN",), ("SUN", "TEST"), UNIT_SYSTEM_ID, newtonian_metadata()
        )
        relativity = RestrictedStaticCentral1PN(
            "SUN",
            ("TEST",),
            100.0,
            0.1,
            0.1,
            UNIT_SYSTEM_ID,
            one_pn_metadata(),
        )
        plan = ForcePlan("fixture.plan", backend(), (newtonian, relativity))
        self.assertEqual(
            tuple(model.model_id for model in plan.models),
            (newtonian.model_id, relativity.model_id),
        )
        self.assertEqual(plan.evidence_class, "MODEL_OUTPUT")
        self.assertFalse(plan.registry_authorized)
        self.assertFalse(plan.qualification_authorized)

        for change in (
            {"models": ()},
            {"models": (newtonian, newtonian)},
            {"models": [newtonian]},
            {"evidence_class": "QUALIFIED_RESULT"},
            {"registry_authorized": True},
            {"qualification_authorized": True},
        ):
            with self.subTest(change=change), self.assertRaises(ContractError):
                dataclasses.replace(plan, **change)


if __name__ == "__main__":
    unittest.main()
