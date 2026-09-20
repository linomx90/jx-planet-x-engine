import dataclasses
import hashlib
import json
import math
import unittest

import numpy as np

import jxplanetx.engine as engine
import jxplanetx.engine.api as engine_api
from jxplanetx.engine import (
    DynamicsScenario,
    ForcePlan,
    RestrictedStaticCentral1PN,
    ScenarioManifestError,
    dump_dynamics_scenario_manifest,
    dynamics_scenario_manifest_sha256,
    load_dynamics_scenario_manifest,
    run_dynamics_scenario,
)
from tests.test_engine_scenario import (
    UNIT_SYSTEM_ID,
    backend,
    gravity,
    integration_spec,
    metadata,
    scenario_pair,
    snapshot,
    srp,
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


def complete_scenario() -> DynamicsScenario:
    state = snapshot()
    state.velocities[0, 2] = -0.0
    return DynamicsScenario(
        scenario_id="fixture.scenario.portable.complete",
        role="STANDALONE",
        description="Portable Newtonian, restricted 1PN, and SRP fixture",
        comparison_id=None,
        initial_snapshot=state,
        force_plan=ForcePlan(
            "fixture.scenario.portable.complete.plan",
            backend(),
            (gravity(), one_pn(), srp()),
        ),
        integration_spec=integration_spec(),
    )


def canonical(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


class PortableScenarioManifestTests(unittest.TestCase):
    def test_complete_force_roster_round_trips_and_executes_from_readonly_arrays(self):
        original = complete_scenario()
        raw = dump_dynamics_scenario_manifest(original)
        digest = dynamics_scenario_manifest_sha256(raw)
        loaded = load_dynamics_scenario_manifest(raw, digest)

        self.assertEqual(dump_dynamics_scenario_manifest(loaded), raw)
        self.assertEqual(loaded.force_model_ids, original.force_model_ids)
        self.assertIn(b'"-0x0.0p+0"', raw)
        arrays = (
            loaded.initial_snapshot.positions,
            loaded.initial_snapshot.velocities,
            loaded.initial_snapshot.gravitational_parameters,
            loaded.initial_snapshot.masses,
            loaded.initial_snapshot.radii,
            loaded.initial_snapshot.massive,
            loaded.integration_spec.position_atol,
            loaded.integration_spec.velocity_atol,
            loaded.force_plan.models[-1].area_to_mass,
            loaded.force_plan.models[-1].radiation_pressure_coefficient,
        )
        self.assertTrue(all(type(array) is np.ndarray for array in arrays))
        self.assertTrue(all(not array.flags.writeable for array in arrays))

        original_result = run_dynamics_scenario(original)
        loaded_result = run_dynamics_scenario(loaded)
        self.assertEqual(
            original_result.trajectory.accepted_step_ledger_content_sha256,
            loaded_result.trajectory.accepted_step_ledger_content_sha256,
        )
        for left, right in zip(
            original_result.trajectory.checkpoints,
            loaded_result.trajectory.checkpoints,
        ):
            self.assertTrue(np.array_equal(left.positions, right.positions))
            self.assertTrue(np.array_equal(left.velocities, right.velocities))

    def test_manifest_has_a_frozen_known_answer_identity(self):
        raw = dump_dynamics_scenario_manifest(scenario_pair()[0])
        self.assertEqual(len(raw), 5_298)
        self.assertEqual(
            dynamics_scenario_manifest_sha256(raw),
            "66fa6faca6242deabb0612a6500fd692d2a4e534b449dfb224d12058de95b917",
        )

    def test_external_hash_and_canonical_bytes_are_mandatory(self):
        raw = dump_dynamics_scenario_manifest(scenario_pair()[0])
        digest = dynamics_scenario_manifest_sha256(raw)
        with self.assertRaisesRegex(ScenarioManifestError, "identity mismatch"):
            load_dynamics_scenario_manifest(raw, "0" * 64)

        document = json.loads(raw)
        noncanonical = json.dumps(document, indent=2, sort_keys=True).encode("ascii")
        with self.assertRaisesRegex(ScenarioManifestError, "not canonical"):
            load_dynamics_scenario_manifest(
                noncanonical,
                hashlib.sha256(noncanonical).hexdigest(),
            )
        with self.assertRaisesRegex(ScenarioManifestError, "exact bytes"):
            dynamics_scenario_manifest_sha256(bytearray(raw))  # type: ignore[arg-type]
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_duplicate_keys_json_float_tokens_and_noncanonical_hex_fail_closed(self):
        raw = dump_dynamics_scenario_manifest(scenario_pair()[0])

        duplicate = b'{"schema":"duplicate",' + raw[1:]
        with self.assertRaisesRegex(ScenarioManifestError, "repeats object key"):
            load_dynamics_scenario_manifest(
                duplicate,
                hashlib.sha256(duplicate).hexdigest(),
            )

        floating = raw.replace(b'"epoch_hex":"0x0.0p+0"', b'"epoch_hex":0.0', 1)
        with self.assertRaisesRegex(ScenarioManifestError, "floating token"):
            load_dynamics_scenario_manifest(
                floating,
                hashlib.sha256(floating).hexdigest(),
            )

        noncanonical_hex = raw.replace(b'"epoch_hex":"0x0.0p+0"', b'"epoch_hex":"0x0p+0"', 1)
        with self.assertRaisesRegex(ScenarioManifestError, "canonical finite"):
            load_dynamics_scenario_manifest(
                noncanonical_hex,
                hashlib.sha256(noncanonical_hex).hexdigest(),
            )

    def test_fixed_method_and_exact_json_types_cannot_be_rewritten(self):
        raw = dump_dynamics_scenario_manifest(scenario_pair()[0])
        document = json.loads(raw)
        document["scenario"]["integration_spec"]["source_report"] = "UNKNOWN"
        changed_method = canonical(document)
        with self.assertRaisesRegex(ScenarioManifestError, "complete fixed RKF78"):
            load_dynamics_scenario_manifest(
                changed_method,
                hashlib.sha256(changed_method).hexdigest(),
            )

        document = json.loads(raw)
        document["scenario"]["accuracy_claimed"] = 0
        changed_type = canonical(document)
        with self.assertRaisesRegex(ScenarioManifestError, "exact boolean"):
            load_dynamics_scenario_manifest(
                changed_type,
                hashlib.sha256(changed_type).hexdigest(),
            )

    def test_tampering_is_detected_but_a_rehashed_valid_document_is_a_new_scenario(self):
        raw = dump_dynamics_scenario_manifest(scenario_pair()[0])
        old_digest = dynamics_scenario_manifest_sha256(raw)
        document = json.loads(raw)
        values = document["scenario"]["initial_snapshot"]["positions"]["values_hex"]
        values[3] = (float.fromhex(values[3]) + 0.125).hex()
        changed = canonical(document)
        with self.assertRaisesRegex(ScenarioManifestError, "identity mismatch"):
            load_dynamics_scenario_manifest(changed, old_digest)

        changed_digest = dynamics_scenario_manifest_sha256(changed)
        loaded = load_dynamics_scenario_manifest(changed, changed_digest)
        self.assertNotEqual(changed_digest, old_digest)
        self.assertEqual(float(loaded.initial_snapshot.positions[1, 0]), 1.125)

    def test_dump_rejects_nonportable_backends_and_invalid_array_storage(self):
        scenario = scenario_pair()[0]
        gpu_backend = dataclasses.replace(
            scenario.force_plan.backend,
            backend_id="cupy",
            device="cuda:0",
        )
        gpu_scenario = dataclasses.replace(
            scenario,
            force_plan=dataclasses.replace(
                scenario.force_plan,
                backend=gpu_backend,
            ),
        )
        with self.assertRaisesRegex(ScenarioManifestError, "only NumPy/CPU"):
            dump_dynamics_scenario_manifest(gpu_scenario)

        float32_snapshot = dataclasses.replace(
            scenario.initial_snapshot,
            positions=scenario.initial_snapshot.positions.astype(np.float32),
        )
        with self.assertRaisesRegex(ScenarioManifestError, "dtype float64"):
            dump_dynamics_scenario_manifest(
                dataclasses.replace(scenario, initial_snapshot=float32_snapshot)
            )

        nonfinite_snapshot = dataclasses.replace(
            scenario.initial_snapshot,
            positions=scenario.initial_snapshot.positions.copy(),
        )
        nonfinite_snapshot.positions[0, 0] = math.inf
        with self.assertRaisesRegex(ScenarioManifestError, "only finite"):
            dump_dynamics_scenario_manifest(
                dataclasses.replace(scenario, initial_snapshot=nonfinite_snapshot)
            )

    def test_public_api_exports_the_manifest_surface_with_identical_objects(self):
        expected = (
            "PORTABLE_SCENARIO_ARRAY_ORDER",
            "PORTABLE_SCENARIO_FLOAT_ENCODING",
            "PORTABLE_SCENARIO_HASH_ALGORITHM",
            "PORTABLE_SCENARIO_MANIFEST_MAX_BYTES",
            "PORTABLE_SCENARIO_MANIFEST_SCHEMA",
            "PORTABLE_SCENARIO_MANIFEST_SCOPE",
            "ScenarioManifestError",
            "dump_dynamics_scenario_manifest",
            "dynamics_scenario_manifest_sha256",
            "load_dynamics_scenario_manifest",
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
