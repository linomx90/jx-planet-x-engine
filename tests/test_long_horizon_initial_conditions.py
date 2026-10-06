from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import unittest

from jxplanetx.long_horizon.initial_conditions import (
    BINARY64_SEMANTICS,
    CONVERTER_IMPLEMENTATION_SHA256,
    DE440_GM_ARTIFACT_SHA256,
    DE440_SOLAR_GM_AU3_PER_JULIAN_YEAR2,
    DE440_SUN_GENESIS_MODEL,
    DE441_CONTROL_PRIMARY_GM_AU3_PER_JULIAN_YEAR2,
    DE441_CONTROL_SOURCE_MODEL_MANIFEST_PACKAGE_PATH,
    DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256,
    DE441_CONTROL_STATE_PACKAGE_PATH,
    DE441_CONTROL_STATE_SHA256,
    DEFAULT_GENESIS_MODEL,
    GENESIS_NONCLAIMS,
    GENESIS_STATUS,
    InitialConditionContractError,
    SOURCE_ALIGNED_GENESIS_MODEL,
    build_initial_condition_bundle,
    build_predeclared_initial_condition_inventory,
    orbital_elements_to_primary_relative_cartesian,
    primary_relative_cartesian_to_orbital_elements,
)
from jxplanetx.long_horizon.science_inputs import (
    DETACHED_TNO_SCREEN_V1,
    DETACHED_TNO_SENSITIVITY_V1,
    OrbitalElements,
    SCATTERING_DISK_CONTROL_V1,
    SCATTERING_DISK_SENSITIVITY_V1,
    SCIENCE_INPUT_CLAIM_CEILING,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_STATE = (
    ROOT
    / "runs"
    / "de441_population_100k"
    / "states"
    / "de441_control_state.csv"
)


def _angular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % math.tau - math.pi)


class LongHorizonInitialConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = build_predeclared_initial_condition_inventory()

    def test_model_is_explicit_binary64_and_content_addressed(self) -> None:
        model = DEFAULT_GENESIS_MODEL
        self.assertEqual(model.epoch_jd_tdb, 2_461_200.5)
        self.assertEqual(model.epoch_scale, "TDB")
        self.assertEqual(model.epoch_unit, "JULIAN_DATE_DAY")
        self.assertEqual(
            model.source_element_frame,
            "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING",
        )
        self.assertEqual(
            model.output_cartesian_frame,
            "PRIMARY_RELATIVE_IAU76_80_ECLIPTIC_J2000",
        )
        self.assertIs(model, SOURCE_ALIGNED_GENESIS_MODEL)
        self.assertEqual(model.primary_id, "DE441-INNER-SYSTEM-MONOPOLE")
        self.assertIn("INNER_SYSTEM_MONOPOLE", model.primary_definition)
        self.assertEqual(
            model.primary_gm_AU3_per_year2,
            DE441_CONTROL_PRIMARY_GM_AU3_PER_JULIAN_YEAR2,
        )
        self.assertEqual(
            model.primary_gm_AU3_per_year2.hex(),
            "0x1.3bd13a81d2262p+5",
        )
        self.assertEqual(model.length_unit, "AU")
        self.assertEqual(model.time_unit, "JULIAN_YEAR_365.25_D")
        self.assertEqual(model.binary64_semantics, BINARY64_SEMANTICS)
        self.assertEqual(
            model.primary_gm_source_artifact_sha256,
            DE441_CONTROL_STATE_SHA256,
        )
        self.assertIn(
            "NO_EPHEMERIS_ORIGIN_TRANSLATION",
            model.element_origin_to_output_origin_policy,
        )
        self.assertEqual(
            DE440_SUN_GENESIS_MODEL.primary_gm_AU3_per_year2,
            DE440_SOLAR_GM_AU3_PER_JULIAN_YEAR2,
        )
        self.assertEqual(
            DE440_SUN_GENESIS_MODEL.primary_gm_AU3_per_year2.hex(),
            "0x1.3bd0beccac6afp+5",
        )
        self.assertEqual(
            DE440_SUN_GENESIS_MODEL.primary_gm_source_artifact_sha256,
            DE440_GM_ARTIFACT_SHA256,
        )
        self.assertRegex(CONVERTER_IMPLEMENTATION_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            model.converter_implementation_sha256,
            CONVERTER_IMPLEMENTATION_SHA256,
        )
        module_bytes = (
            ROOT / "src/jxplanetx/long_horizon/initial_conditions.py"
        ).read_bytes()
        self.assertEqual(
            CONVERTER_IMPLEMENTATION_SHA256,
            hashlib.sha256(
                b"jx.long-horizon.cartesian-genesis-implementation.v1\0"
                + module_bytes
            ).hexdigest(),
        )
        self.assertRegex(model.content_sha256, r"^[0-9a-f]{64}$")

    def test_default_primary_gm_is_exact_frozen_force_model_row(self) -> None:
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.primary_gm_source_artifact_path,
            DE441_CONTROL_STATE_PACKAGE_PATH,
        )
        packaged = resources.files("jxplanetx").joinpath(
            "data",
            "long_horizon_source_model_v1",
            "de441_control_state.csv",
        )
        self.assertEqual(packaged.read_bytes(), CONTROL_STATE.read_bytes())
        self.assertEqual(
            hashlib.sha256(packaged.read_bytes()).hexdigest(),
            DE441_CONTROL_STATE_SHA256,
        )
        manifest = resources.files("jxplanetx").joinpath(
            "data",
            "long_horizon_source_model_v1",
            "MANIFEST.json",
        )
        manifest_bytes = manifest.read_bytes()
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(),
            DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256,
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.source_model_manifest_path,
            DE441_CONTROL_SOURCE_MODEL_MANIFEST_PACKAGE_PATH,
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.source_model_manifest_sha256,
            DE441_CONTROL_SOURCE_MODEL_MANIFEST_SHA256,
        )
        manifest_record = json.loads(manifest_bytes)
        self.assertEqual(
            manifest_record["artifact"]["sha256"],
            DE441_CONTROL_STATE_SHA256,
        )
        self.assertEqual(
            manifest_record["source_ids_ordered"],
            ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"],
        )
        with CONTROL_STATE.open(newline="", encoding="utf-8") as stream:
            first = next(csv.DictReader(stream))
        self.assertEqual(first["index"], "0")
        self.assertEqual(first["name"], "Sun")
        self.assertEqual(
            float(first["mass"]),
            DEFAULT_GENESIS_MODEL.primary_gm_AU3_per_year2,
        )
        self.assertNotEqual(
            DEFAULT_GENESIS_MODEL.primary_gm_AU3_per_year2,
            DE440_SUN_GENESIS_MODEL.primary_gm_AU3_per_year2,
        )

    def test_predeclared_inventory_is_exactly_108_bound_states(self) -> None:
        inventory = self.inventory
        self.assertEqual(inventory.state_count, 108)
        self.assertEqual(inventory.expected_state_count, 108)
        self.assertEqual(len(inventory.bundles), 2)
        self.assertEqual(
            [bundle.state_count for bundle in inventory.bundles], [54, 54]
        )
        self.assertEqual(
            [
                sum(state.source_kind == "NOMINAL" for state in bundle.states)
                for bundle in inventory.bundles
            ],
            [6, 6],
        )
        self.assertEqual(
            [
                sum(state.source_kind == "PERTURBED" for state in bundle.states)
                for bundle in inventory.bundles
            ],
            [48, 48],
        )
        self.assertEqual(len({state.object_id for state in inventory.states}), 108)
        self.assertEqual(
            inventory.content_sha256,
            build_predeclared_initial_condition_inventory().content_sha256,
        )

        expected = (
            (DETACHED_TNO_SCREEN_V1, DETACHED_TNO_SENSITIVITY_V1),
            (SCATTERING_DISK_CONTROL_V1, SCATTERING_DISK_SENSITIVITY_V1),
        )
        for bundle, (population, ensemble) in zip(
            inventory.bundles, expected, strict=True
        ):
            self.assertEqual(bundle.population_sha256, population.content_sha256)
            self.assertEqual(
                bundle.raw_source_artifact_sha256,
                population.source_artifact_sha256,
            )
            self.assertEqual(bundle.ensemble_sha256, ensemble.content_sha256)
            self.assertEqual(bundle.model, DEFAULT_GENESIS_MODEL)
            self.assertRegex(bundle.materialized_ensemble_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(bundle.content_sha256, r"^[0-9a-f]{64}$")
            for state in bundle.states:
                self.assertEqual(state.population_sha256, bundle.population_sha256)
                self.assertEqual(state.ensemble_sha256, bundle.ensemble_sha256)
                self.assertEqual(
                    state.materialized_ensemble_sha256,
                    bundle.materialized_ensemble_sha256,
                )
                self.assertEqual(
                    state.raw_source_artifact_sha256,
                    bundle.raw_source_artifact_sha256,
                )
                self.assertEqual(state.model.content_sha256, bundle.model.content_sha256)

    def test_every_state_passes_independent_element_round_trip(self) -> None:
        model = DEFAULT_GENESIS_MODEL
        for state in self.inventory.states:
            recovered = primary_relative_cartesian_to_orbital_elements(
                state.object_id,
                state.position_AU,
                state.velocity_AU_per_year,
                model=model,
            )
            source = state.source_elements
            self.assertLessEqual(
                abs(recovered.semimajor_axis_AU - source.semimajor_axis_AU),
                max(
                    model.roundtrip_scalar_absolute_tolerance,
                    model.roundtrip_semimajor_axis_relative_tolerance
                    * source.semimajor_axis_AU,
                ),
            )
            self.assertLessEqual(
                abs(recovered.eccentricity - source.eccentricity),
                model.roundtrip_scalar_absolute_tolerance,
            )
            self.assertLessEqual(
                abs(recovered.inclination_rad - source.inclination_rad),
                model.roundtrip_angle_absolute_tolerance_rad,
            )
            for left, right in (
                (
                    recovered.longitude_of_ascending_node_rad,
                    source.longitude_of_ascending_node_rad,
                ),
                (
                    recovered.argument_of_periapsis_rad,
                    source.argument_of_periapsis_rad,
                ),
                (recovered.mean_anomaly_rad, source.mean_anomaly_rad),
            ):
                self.assertLessEqual(
                    _angular_distance(left, right),
                    model.roundtrip_angle_absolute_tolerance_rad,
                )
            position, velocity = orbital_elements_to_primary_relative_cartesian(
                source, model=model
            )
            self.assertEqual(position, state.position_AU)
            self.assertEqual(velocity, state.velocity_AU_per_year)

    def test_states_are_bound_and_tampering_fails_closed(self) -> None:
        bundle = self.inventory.bundles[0]
        state = bundle.states[0]
        tampered_position = (
            math.nextafter(state.position_AU[0], math.inf),
            state.position_AU[1],
            state.position_AU[2],
        )
        with self.assertRaises(InitialConditionContractError):
            replace(state, position_AU=tampered_position)
        forged_state = replace(state, population_sha256="0" * 64)
        self.assertNotEqual(forged_state.content_sha256, state.content_sha256)
        with self.assertRaises(InitialConditionContractError):
            replace(bundle, states=(forged_state,) + bundle.states[1:])
        with self.assertRaises(InitialConditionContractError):
            replace(state, inference_authorized=True)
        with self.assertRaises(InitialConditionContractError):
            replace(bundle, states=bundle.states[::-1])

        forged_population = replace(
            DETACHED_TNO_SCREEN_V1,
            source_artifact_sha256="0" * 64,
        )
        with self.assertRaises(InitialConditionContractError):
            build_initial_condition_bundle(
                forged_population, DETACHED_TNO_SENSITIVITY_V1
            )

    def test_exact_types_singularities_and_model_changes_are_closed(self) -> None:
        with self.assertRaises(InitialConditionContractError):
            replace(DEFAULT_GENESIS_MODEL, primary_gm_AU3_per_year2=39)
        with self.assertRaises(InitialConditionContractError):
            replace(DEFAULT_GENESIS_MODEL, kepler_maximum_iterations=True)
        with self.assertRaises(InitialConditionContractError):
            primary_relative_cartesian_to_orbital_elements(
                "bad-state", [1.0, 0.0, 0.0], (0.0, 1.0, 0.0)
            )
        singular = OrbitalElements(
            "singular",
            50.0,
            0.0,
            0.2,
            0.3,
            0.4,
            0.5,
        )
        with self.assertRaises(InitialConditionContractError):
            orbital_elements_to_primary_relative_cartesian(singular)

        changed_model = replace(
            DEFAULT_GENESIS_MODEL,
            primary_gm_AU3_per_year2=math.nextafter(
                DEFAULT_GENESIS_MODEL.primary_gm_AU3_per_year2, math.inf
            ),
        )
        changed_bundle = build_initial_condition_bundle(
            DETACHED_TNO_SCREEN_V1,
            DETACHED_TNO_SENSITIVITY_V1,
            model=changed_model,
        )
        self.assertNotEqual(
            changed_model.content_sha256,
            DEFAULT_GENESIS_MODEL.content_sha256,
        )
        self.assertNotEqual(
            changed_bundle.content_sha256,
            self.inventory.bundles[0].content_sha256,
        )

    def test_claim_ceiling_cannot_be_widened(self) -> None:
        inventory = self.inventory
        self.assertEqual(inventory.status, GENESIS_STATUS)
        self.assertFalse(inventory.inference_authorized)
        self.assertEqual(inventory.claim_ceiling, SCIENCE_INPUT_CLAIM_CEILING)
        self.assertEqual(inventory.nonclaims, GENESIS_NONCLAIMS)
        self.assertIn("not an ephemeris evaluation or observational orbit fit", GENESIS_NONCLAIMS)
        for state in inventory.states:
            self.assertFalse(state.inference_authorized)
            self.assertEqual(state.claim_ceiling, SCIENCE_INPUT_CLAIM_CEILING)
        with self.assertRaises(InitialConditionContractError):
            replace(inventory, claim_ceiling="SCIENTIFIC_RESULT")


if __name__ == "__main__":
    unittest.main()
