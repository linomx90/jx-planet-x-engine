from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from importlib import resources
import math
import unittest
from unittest.mock import patch

import numpy as np

from jxplanetx.engine import Provenance
from jxplanetx.long_horizon.initial_conditions import DE440_SUN_GENESIS_MODEL
from jxplanetx.long_horizon import physical_models as physical_models_module
from jxplanetx.long_horizon.physical_models import (
    CHECKSUM_ARTIFACT_SHA256,
    COMPACT_ELEMENT_FRAME,
    DERIVED_ARTIFACT_SHA256,
    DE440_EARTH_GM_AU3_PER_JULIAN_YEAR2,
    DE440_EARTH_GM_KM3_S2_EXACT,
    GM_ARTIFACT_SHA256,
    KNOWN_SOLAR_SYSTEM_CONTENT_SHA256,
    KNOWN_SOLAR_SYSTEM_CONVERTED_STATE_SHA256,
    KNOWN_SOLAR_SYSTEM_SOURCE_STATE_SHA256,
    KNOWN_SOLAR_SYSTEM_V2,
    KNOWN_SYSTEM_BODY_IDS,
    KNOWN_SYSTEM_BODY_NAMES,
    MANIFEST_ARTIFACT_SHA256,
    PHYSICAL_CLAIM_FLAGS,
    README_ARTIFACT_SHA256,
    SOURCE_ARTIFACT_SHA256,
    BoundCompactPlanetMember,
    CompactPlanetEnsemble,
    MatchedPhysicalArmSet,
    PhysicalModelContractError,
    build_matched_physical_arm_set,
    compact_planet_gm_from_earth_masses,
    load_known_solar_system_source_model,
)


PROVENANCE_SHA = hashlib.sha256(b"jx-physical-model-test-provenance").hexdigest()


def _provider_input_sha256(
    source_ids: tuple[str, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.restricted-wh.source-provider.input.v1\0")
    for source_id in source_ids:
        encoded = source_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for array in (positions, velocities, gm):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _member(member_id: str = "candidate-a", **changes) -> BoundCompactPlanetMember:
    arguments = {
        "member_id": member_id,
        "epoch_jd_tdb": 2_461_200.5,
        "mass_earth_masses": 5.0,
        "gm_AU3_per_julian_year2": compact_planet_gm_from_earth_masses(5.0),
        "semimajor_axis_AU": 400.0,
        "eccentricity": 0.2,
        "inclination_rad": 0.3,
        "longitude_of_ascending_node_rad": 1.0,
        "argument_of_periapsis_rad": 2.0,
        "mean_anomaly_rad": 3.0,
        "prior_weight": 0.25,
        "provenance_id": "test-candidate-catalog-v1",
        "provenance_artifact_path": "tests/data/candidate-catalog-v1.json",
        "provenance_artifact_sha256": PROVENANCE_SHA,
    }
    arguments.update(changes)
    return BoundCompactPlanetMember(**arguments)


def _ensemble() -> CompactPlanetEnsemble:
    return CompactPlanetEnsemble(
        ensemble_id="test-compact-ensemble-v1",
        members=(
            _member(),
            _member(
                "candidate-b",
                semimajor_axis_AU=550.0,
                eccentricity=0.4,
                inclination_rad=0.4,
                mean_anomaly_rad=4.0,
                prior_weight=0.5,
            ),
        ),
        provenance_id="test-ensemble-catalog-v1",
        provenance_artifact_path="tests/data/test-ensemble-v1.json",
        provenance_artifact_sha256=PROVENANCE_SHA,
    )


class KnownSolarSystemSourceModelTests(unittest.TestCase):
    def test_packaged_model_is_exact_real_sun_then_nine_barycenters(self) -> None:
        model = KNOWN_SOLAR_SYSTEM_V2
        self.assertEqual(model.body_ids, KNOWN_SYSTEM_BODY_IDS)
        self.assertEqual(model.body_names, KNOWN_SYSTEM_BODY_NAMES)
        self.assertEqual(model.source_count, 10)
        self.assertEqual(model.body_ids[0], 10)
        self.assertEqual(model.body_names[0], "Sun")
        self.assertEqual(model.body_names[-1], "Pluto Barycenter")
        self.assertNotEqual(model.source_positions_barycentric_icrf_AU[0], (0.0,) * 3)
        self.assertEqual(model.positions_primary_relative_ecliptic_AU[0], (0.0,) * 3)
        self.assertEqual(
            model.velocities_primary_relative_ecliptic_AU_per_julian_year[0],
            (0.0,) * 3,
        )
        self.assertTrue(
            all(
                component.hex() == "0x0.0p+0"
                for component in (
                    *model.positions_primary_relative_ecliptic_AU[0],
                    *model.velocities_primary_relative_ecliptic_AU_per_julian_year[0],
                )
            )
        )
        self.assertEqual(
            model.primary_gm_AU3_per_julian_year2.hex(),
            "0x1.3bd0beccac6afp+5",
        )
        self.assertEqual(
            model.primary_gm_AU3_per_julian_year2,
            DE440_SUN_GENESIS_MODEL.primary_gm_AU3_per_year2,
        )
        self.assertFalse(any(value for _, value in model.claim_flags))
        self.assertEqual(model.claim_flags, PHYSICAL_CLAIM_FLAGS)

    def test_loader_repeats_byte_and_conversion_custody_deterministically(self) -> None:
        rebuilt = load_known_solar_system_source_model()
        self.assertEqual(rebuilt, KNOWN_SOLAR_SYSTEM_V2)
        self.assertEqual(rebuilt.content_sha256, KNOWN_SOLAR_SYSTEM_CONTENT_SHA256)
        self.assertEqual(
            rebuilt.source_barycentric_state_sha256,
            KNOWN_SOLAR_SYSTEM_SOURCE_STATE_SHA256,
        )
        self.assertEqual(
            rebuilt.converted_state_sha256,
            KNOWN_SOLAR_SYSTEM_CONVERTED_STATE_SHA256,
        )
        self.assertEqual(
            KNOWN_SOLAR_SYSTEM_CONTENT_SHA256,
            "19d33011f45d07d15a2aff8acf56a48c57e726828d8a34f1354f070987b86387",
        )
        self.assertEqual(
            KNOWN_SOLAR_SYSTEM_SOURCE_STATE_SHA256,
            "d545472608abc19360b5fd020918c48bcd8cb9fe95da4b8adc315ba8a1513e29",
        )
        self.assertEqual(
            KNOWN_SOLAR_SYSTEM_CONVERTED_STATE_SHA256,
            "3b6de264608aa07fca7281c28c2dfbccc66bd9bf7bcb84c098cb0b81decbdbee",
        )

    def test_packaged_artifact_hashes_are_frozen(self) -> None:
        expected = {
            "MANIFEST.json": MANIFEST_ARTIFACT_SHA256,
            "SHA256SUMS": CHECKSUM_ARTIFACT_SHA256,
            "README.md": README_ARTIFACT_SHA256,
            "gm_de440_major_barycenters.csv": GM_ARTIFACT_SHA256,
            "horizons_de441_barycentric_icrf_jd2461200p5.csv": (
                SOURCE_ARTIFACT_SHA256
            ),
            "known_system_primary_relative_ecliptic_float_hex.csv": (
                DERIVED_ARTIFACT_SHA256
            ),
        }
        root = resources.files("jxplanetx").joinpath(
            "data", "long_horizon_known_system_v2"
        )
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest(), digest)

    def test_loader_reads_and_verifies_packaged_readme(self) -> None:
        observed: list[tuple[str, str]] = []
        original = physical_models_module._resource_bytes

        def recording_resource_bytes(name: str, expected_sha256: str) -> bytes:
            observed.append((name, expected_sha256))
            return original(name, expected_sha256)

        with patch.object(
            physical_models_module,
            "_resource_bytes",
            side_effect=recording_resource_bytes,
        ):
            rebuilt = load_known_solar_system_source_model()
        self.assertEqual(rebuilt.content_sha256, KNOWN_SOLAR_SYSTEM_CONTENT_SHA256)
        self.assertIn(("README.md", README_ARTIFACT_SHA256), observed)

    def test_retained_model_rejects_a_one_ulp_state_change(self) -> None:
        positions = list(KNOWN_SOLAR_SYSTEM_V2.positions_primary_relative_ecliptic_AU)
        mercury = list(positions[1])
        mercury[0] = math.nextafter(mercury[0], math.inf)
        positions[1] = tuple(mercury)
        with self.assertRaisesRegex(
            PhysicalModelContractError, "retained source or converted state"
        ):
            replace(
                KNOWN_SOLAR_SYSTEM_V2,
                positions_primary_relative_ecliptic_AU=tuple(positions),
            )

    def test_claim_widening_and_non_sun_primary_are_rejected(self) -> None:
        widened = PHYSICAL_CLAIM_FLAGS[:-1] + (("uncertainty_calibrated", True),)
        with self.assertRaisesRegex(PhysicalModelContractError, "all-false"):
            replace(KNOWN_SOLAR_SYSTEM_V2, claim_flags=widened)
        positions = list(KNOWN_SOLAR_SYSTEM_V2.positions_primary_relative_ecliptic_AU)
        positions[0] = (math.ulp(1.0), 0.0, 0.0)
        with self.assertRaisesRegex(PhysicalModelContractError, "Sun"):
            replace(
                KNOWN_SOLAR_SYSTEM_V2,
                positions_primary_relative_ecliptic_AU=tuple(positions),
            )


class CompactPlanetPhysicalContractTests(unittest.TestCase):
    def test_earth_mass_gm_uses_the_complete_retained_body399_decimal(self) -> None:
        self.assertEqual(DE440_EARTH_GM_KM3_S2_EXACT, "398600.43550702266")
        exact = (
            Fraction(DE440_EARTH_GM_KM3_S2_EXACT)
            * Fraction(86_400 * 36_525, 100) ** 2
            / Fraction(1_495_978_707, 10) ** 3
        )
        self.assertEqual(
            DE440_EARTH_GM_AU3_PER_JULIAN_YEAR2, float(exact)
        )
        self.assertEqual(
            DE440_EARTH_GM_AU3_PER_JULIAN_YEAR2.hex(),
            "0x1.f14fffb533692p-14",
        )
        self.assertEqual(
            compact_planet_gm_from_earth_masses(5.0).hex(),
            "0x1.36d1ffd14021cp-11",
        )

    def test_member_binds_mass_gm_elements_weight_and_provenance(self) -> None:
        member = _member()
        self.assertGreater(member.mass_earth_masses, 0.0)
        self.assertGreater(member.gm_AU3_per_julian_year2, 0.0)
        self.assertEqual(member.element_frame, COMPACT_ELEMENT_FRAME)
        self.assertEqual(member.perihelion_AU, 320.0)
        self.assertEqual(member.as_orbital_elements().object_id, member.member_id)
        self.assertRegex(member.content_sha256, r"^[0-9a-f]{64}$")
        self.assertFalse(any(value for _, value in member.claim_flags))

    def test_member_rejects_mass_gm_mismatch(self) -> None:
        with self.assertRaisesRegex(PhysicalModelContractError, "does not exactly match"):
            _member(gm_AU3_per_julian_year2=5.0e-4)

    def test_member_rejects_unbound_high_eccentricity_and_missing_custody(self) -> None:
        for value in (1.0, 1.1, -0.01):
            with self.subTest(eccentricity=value):
                with self.assertRaisesRegex(PhysicalModelContractError, "bound"):
                    _member(eccentricity=value)
        with self.assertRaisesRegex(PhysicalModelContractError, "may not exceed"):
            _member(eccentricity=0.9000000000000001)
        with self.assertRaisesRegex(PhysicalModelContractError, "provenance"):
            _member(provenance_id="")
        with self.assertRaisesRegex(PhysicalModelContractError, "SHA-256"):
            _member(provenance_artifact_sha256="")
        for weight in (0.0, -1.0, 1.0000000000000002):
            with self.subTest(weight=weight):
                with self.assertRaises(PhysicalModelContractError):
                    _member(prior_weight=weight)

    def test_ensemble_requires_explicit_members_and_does_not_assume_sum_one(self) -> None:
        ensemble = _ensemble()
        self.assertEqual(ensemble.weights, (0.25, 0.5))
        self.assertEqual(ensemble.total_weight, 0.75)
        self.assertRegex(ensemble.content_sha256, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(PhysicalModelContractError, "nonempty"):
            replace(ensemble, members=())
        with self.assertRaisesRegex(PhysicalModelContractError, "unique"):
            replace(ensemble, members=(ensemble.members[0], ensemble.members[0]))
        with self.assertRaisesRegex(PhysicalModelContractError, "provenance"):
            replace(ensemble, provenance_artifact_path="")

    def test_matched_arms_share_exact_prefix_and_append_after_pluto(self) -> None:
        ensemble = _ensemble()
        arms = build_matched_physical_arm_set(
            "test-matched-arms-v1",
            KNOWN_SOLAR_SYSTEM_V2,
            ensemble,
            provenance_id="test-matched-arm-build-v1",
            provenance_artifact_path="tests/data/test-matched-arm-build-v1.json",
            provenance_artifact_sha256=PROVENANCE_SHA,
        )
        self.assertIsInstance(arms, MatchedPhysicalArmSet)
        self.assertEqual(arms.arm_count, 2)
        self.assertEqual(len(set(arms.arm_state_sha256)), 2)
        for index, member in enumerate(ensemble.members):
            self.assertEqual(
                arms.arm_body_names[index][:-1], KNOWN_SYSTEM_BODY_NAMES
            )
            self.assertEqual(arms.arm_body_names[index][-2], "Pluto Barycenter")
            self.assertEqual(arms.arm_body_names[index][-1], member.member_id)
            self.assertEqual(
                arms.arm_gm_AU3_per_julian_year2[index][:-1],
                KNOWN_SOLAR_SYSTEM_V2.gm_AU3_per_julian_year2,
            )
            self.assertEqual(
                arms.arm_positions_primary_relative_ecliptic_AU[index][:-1],
                KNOWN_SOLAR_SYSTEM_V2.positions_primary_relative_ecliptic_AU,
            )
            self.assertEqual(
                arms.arm_velocities_primary_relative_ecliptic_AU_per_julian_year[
                    index
                ][:-1],
                KNOWN_SOLAR_SYSTEM_V2.velocities_primary_relative_ecliptic_AU_per_julian_year,
            )
        self.assertFalse(any(value for _, value in arms.claim_flags))

    def test_matched_arms_reject_changed_candidate_state(self) -> None:
        arms = build_matched_physical_arm_set(
            "test-matched-arms-v1",
            KNOWN_SOLAR_SYSTEM_V2,
            _ensemble(),
            provenance_id="test-matched-arm-build-v1",
            provenance_artifact_path="tests/data/test-matched-arm-build-v1.json",
            provenance_artifact_sha256=PROVENANCE_SHA,
        )
        candidate_positions = list(
            arms.candidate_positions_primary_relative_ecliptic_AU
        )
        candidate = list(candidate_positions[0])
        candidate[0] = math.nextafter(candidate[0], math.inf)
        candidate_positions[0] = tuple(candidate)
        with self.assertRaisesRegex(PhysicalModelContractError, "locked element"):
            replace(
                arms,
                candidate_positions_primary_relative_ecliptic_AU=tuple(
                    candidate_positions
                ),
            )


class KnownSystemSourceProviderRegressionTests(unittest.TestCase):
    def test_appended_candidate_arm_passes_bounded_canonicalization(self) -> None:
        from jxplanetx.restricted_wh_source import (
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
            WisdomHolmanSourceTileSpec,
            build_wisdom_holman_source_tile,
        )

        model = KNOWN_SOLAR_SYSTEM_V2
        arms = build_matched_physical_arm_set(
            "test-provider-candidate-arm-v1",
            model,
            CompactPlanetEnsemble(
                ensemble_id="test-provider-candidate-ensemble-v1",
                members=(
                    _member(
                        "candidate-provider-regression",
                        mass_earth_masses=14.530761600753994,
                        gm_AU3_per_julian_year2=(
                            compact_planet_gm_from_earth_masses(
                                14.530761600753994
                            )
                        ),
                        semimajor_axis_AU=913.607983690281,
                        eccentricity=0.2991370384345897,
                        inclination_rad=0.3409144389768314,
                        longitude_of_ascending_node_rad=4.240630197599849,
                        argument_of_periapsis_rad=4.462597433266458,
                        mean_anomaly_rad=4.500712706279397,
                    ),
                ),
                provenance_id="test-provider-candidate-catalog-v1",
                provenance_artifact_path="tests/data/provider-candidate-v1.json",
                provenance_artifact_sha256=PROVENANCE_SHA,
            ),
            provenance_id="test-provider-candidate-build-v1",
            provenance_artifact_path="tests/data/provider-candidate-build-v1.json",
            provenance_artifact_sha256=PROVENANCE_SHA,
        )
        positions = np.ascontiguousarray(
            arms.arm_positions_primary_relative_ecliptic_AU[0], dtype=np.float64
        )
        velocities = np.ascontiguousarray(
            arms.arm_velocities_primary_relative_ecliptic_AU_per_julian_year[0],
            dtype=np.float64,
        )
        gm = np.ascontiguousarray(
            arms.arm_gm_AU3_per_julian_year2[0], dtype=np.float64
        )
        build = build_wisdom_holman_source_tile(
            tile_id="test-provider-candidate-arm-v1",
            epoch_origin_years=0.0,
            global_source_step=0,
            time_scale="TDB",
            frame=model.output_frame,
            axes="J2000_ECLIPTIC",
            source_ids=arms.arm_body_names[0],
            primary_source_id=arms.arm_body_names[0][0],
            source_schedule_sha256=hashlib.sha256(
                b"test-provider-candidate-schedule-v1"
            ).hexdigest(),
            initial_source_relative_positions_AU=positions,
            initial_source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=gm,
            source_encounter_radii_AU=np.full(11, 1.0e-6, dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                11, 1.0e6, dtype=np.float64
            ),
            provenance=Provenance(
                "jx.test.provider-candidate-arm-v1",
                "matched physical candidate-arm canonicalization regression",
                "v1",
                arms.content_sha256,
            ),
            spec=WisdomHolmanSourceTileSpec(
                1.0 / 365.25,
                1,
                minimum_encounter_pair_separation_AU=1.0e-6,
                minimum_jacobi_periapse_AU=1.0e-6,
            ),
        )
        receipt = build.runtime_identity["coordinate_canonicalization"]
        self.assertEqual(receipt["fixed_point_iteration_count"], 3)
        self.assertGreater(receipt["position_maximum_component_ulps"], 4.0)
        self.assertLessEqual(
            receipt["position_maximum_forward_error_bound_fraction"], 1.0
        )
        self.assertLessEqual(
            receipt["velocity_maximum_forward_error_bound_fraction"], 1.0
        )
        self.assertLessEqual(
            receipt["position_maximum_input_relative_error"],
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
        )
        self.assertLessEqual(
            receipt["velocity_maximum_input_relative_error"],
            MAXIMUM_COORDINATE_CANONICALIZATION_RELATIVE_ERROR,
        )
        self.assertTrue(receipt["canonicalization_idempotent"])
        self.assertTrue(receipt["tile_node_zero_exact"])

    def test_real_known_system_source_provider_preserves_raw_model_custody(self) -> None:
        """Exercise the real arrays after provider canonicalization is available."""

        from jxplanetx.restricted_wh_source import (
            WisdomHolmanSourceTileSpec,
            build_wisdom_holman_source_tile,
        )

        model = KNOWN_SOLAR_SYSTEM_V2
        positions = np.ascontiguousarray(
            model.positions_primary_relative_ecliptic_AU, dtype=np.float64
        )
        velocities = np.ascontiguousarray(
            model.velocities_primary_relative_ecliptic_AU_per_julian_year,
            dtype=np.float64,
        )
        gm = np.ascontiguousarray(model.gm_AU3_per_julian_year2, dtype=np.float64)
        positions_before = positions.copy()
        velocities_before = velocities.copy()
        schedule = hashlib.sha256(b"known-system-provider-regression-v2").hexdigest()
        provenance = Provenance(
            "jx.test.known-system-provider-v2",
            "retained DE441/DE440 known-system source regression",
            "v2",
            model.content_sha256,
        )
        build = build_wisdom_holman_source_tile(
            tile_id="known-system-provider-regression-v2",
            epoch_origin_years=0.0,
            global_source_step=0,
            time_scale="TDB",
            frame=model.output_frame,
            axes="J2000_ECLIPTIC",
            source_ids=model.body_names,
            primary_source_id=model.body_names[0],
            source_schedule_sha256=schedule,
            initial_source_relative_positions_AU=positions,
            initial_source_relative_velocities_AU_per_year=velocities,
            source_gm_AU3_per_year2=gm,
            source_encounter_radii_AU=np.full(10, 1.0e-6, dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                10, 1.0e6, dtype=np.float64
            ),
            provenance=provenance,
            spec=WisdomHolmanSourceTileSpec(
                1.0 / 365.25,
                1,
                minimum_encounter_pair_separation_AU=1.0e-6,
                minimum_jacobi_periapse_AU=1.0e-6,
            ),
        )
        np.testing.assert_array_equal(positions, positions_before)
        np.testing.assert_array_equal(velocities, velocities_before)
        self.assertEqual(build.tile.node_count, 2)
        total_gm = np.sum(gm, dtype=np.float64)
        barycentric_positions = np.ascontiguousarray(
            positions
            - np.sum(positions * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        barycentric_velocities = np.ascontiguousarray(
            velocities
            - np.sum(velocities * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        canonical_positions = np.ascontiguousarray(
            barycentric_positions - barycentric_positions[0]
        )
        canonical_velocities = np.ascontiguousarray(
            barycentric_velocities - barycentric_velocities[0]
        )
        self.assertTrue(
            np.array_equal(
                build.tile.source_relative_positions_AU[0],
                canonical_positions,
            )
        )
        self.assertTrue(
            np.array_equal(
                build.tile.source_relative_velocities_AU_per_year[0],
                canonical_velocities,
            )
        )
        correction = build.runtime_identity["coordinate_canonicalization"]
        self.assertEqual(
            correction["raw_relative_input_sha256"],
            _provider_input_sha256(model.body_names, positions, velocities, gm),
        )
        self.assertEqual(
            correction["canonical_relative_input_sha256"],
            _provider_input_sha256(
                model.body_names, canonical_positions, canonical_velocities, gm
            ),
        )
        self.assertEqual(
            correction["raw_relative_input_sha256"],
            correction["canonical_relative_input_sha256"],
        )
        self.assertTrue(correction["canonicalization_idempotent"])
        self.assertTrue(correction["tile_node_zero_exact"])
        self.assertEqual(correction["position_changed_component_count"], 0)
        self.assertEqual(correction["position_maximum_component_ulps"], 0.0)
        self.assertEqual(correction["velocity_changed_component_count"], 0)
        self.assertEqual(correction["velocity_maximum_component_ulps"], 0.0)


if __name__ == "__main__":
    unittest.main()
