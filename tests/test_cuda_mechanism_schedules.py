from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jxplanetx.cuda_mechanism_schedules import (
    ARM_KINDS,
    CONTROL_ARM,
    DE441_CONTROL_EPOCH_JD_TDB,
    DE441_CONTROL_SOURCE_IDS,
    DE441_CONTROL_STATE_SHA256,
    DISTANT_PLANET_ARM,
    INITIAL_CONDITION_SCOPE,
    MASSIVE_FORCE_SEMANTICS,
    PROJECT_SCOPE,
    RESULT_CLAIMS,
    SCIENTIFIC_CLASSIFICATION,
    SOURCE_ACTIVITY_SEMANTICS,
    STELLAR_FLYBY_ARM,
    TEMPORARY_ROGUE_ARM,
    TRACER_BACKREACTION_SEMANTICS,
    TRACER_FORCE_SEMANTICS,
    UNSUPPORTED_CAPABILITIES,
    MechanismArm,
    MechanismScheduleError,
    MatchedMechanismScheduleSet,
    SourceActivityInterval,
    build_de441_four_arm_schedule_set,
    build_synthetic_persistent_distant_planet,
    build_synthetic_stellar_flyby,
    build_synthetic_temporary_rogue,
    load_frozen_de441_outer_control,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = (
    ROOT
    / "runs"
    / "de441_population_100k"
    / "states"
    / "de441_control_state.csv"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


class FrozenDE441ControlTests(unittest.TestCase):
    def test_exact_current_epoch_reduced_state_is_loaded_read_only(self) -> None:
        self.assertEqual(_sha256_file(CONTROL_PATH), DE441_CONTROL_STATE_SHA256)
        state = load_frozen_de441_outer_control(CONTROL_PATH)
        self.assertEqual(state.source_ids, DE441_CONTROL_SOURCE_IDS)
        self.assertEqual(state.primary_source_id, "Sun")
        self.assertEqual(state.epoch_jd_tdb, DE441_CONTROL_EPOCH_JD_TDB)
        self.assertEqual(state.positions_AU.shape, (6, 3))
        self.assertEqual(state.velocities_AU_per_year.shape, (6, 3))
        self.assertEqual(state.gm_AU3_per_year2.shape, (6,))
        self.assertFalse(state.positions_AU.flags.writeable)
        self.assertFalse(state.velocities_AU_per_year.flags.writeable)
        self.assertFalse(state.gm_AU3_per_year2.flags.writeable)
        self.assertFalse(state.primordial_initial_conditions)
        self.assertEqual(state.initial_condition_scope, INITIAL_CONDITION_SCOPE)
        self.assertIn("INNER_SYSTEM_MONOPOLE", state.state_classification)
        with self.assertRaises(ValueError):
            state.positions_AU[0, 0] = 1.0

    def test_loader_refuses_modified_or_wrong_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "control.csv"
            data = bytearray(CONTROL_PATH.read_bytes())
            data[-2] = ord("1") if data[-2] != ord("1") else ord("2")
            copy.write_bytes(data)
            with self.assertRaisesRegex(MechanismScheduleError, "SHA-256"):
                load_frozen_de441_outer_control(copy)


class FourArmScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule_set = build_de441_four_arm_schedule_set(
            CONTROL_PATH,
            step_years=0.5,
            total_steps=2_000,
            nominal_passage_global_step=1_000,
            diagnostic_half_window_steps=20,
        )

    def test_exact_four_arm_order_and_stable_hashes(self) -> None:
        schedules = self.schedule_set
        self.assertEqual(
            tuple(arm.schedule.arm_kind for arm in schedules.arms),
            ARM_KINDS,
        )
        rebuilt = build_de441_four_arm_schedule_set(
            CONTROL_PATH,
            step_years=0.5,
            total_steps=2_000,
            nominal_passage_global_step=1_000,
            diagnostic_half_window_steps=20,
        )
        self.assertEqual(
            schedules.matched_set_sha256,
            rebuilt.matched_set_sha256,
        )
        self.assertEqual(
            schedules.matched_set_sha256,
            "eb03744a73d65fa2894233373fcba6efdff6ab8e9ee988eb5189db6c98cdbd22",
        )
        self.assertEqual(
            tuple(arm.schedule.schedule_sha256 for arm in schedules.arms),
            (
                "bbedf24b2d99e21d54b90a5d0edcae00029b645221be818cb0587bee83542d3d",
                "115432ad9d9de32e509935aa642c322f0ea80cd06cc1923d3c184cf8db4e9bfa",
                "f175cd52ff3e026fc79656d68e728392439f7f008f7c79a1fc5d03ffe990a7fc",
                "1a98f388dc63829e9f4d5fb0b7770fe72811c81fc789d9bad9b8e4facd479992",
            ),
        )

    def test_candidates_preserve_byte_identical_base_prefix(self) -> None:
        control = self.schedule_set.control.initial_state
        count = len(control.source_ids)
        self.assertEqual(count, control.base_source_count)
        for arm in self.schedule_set.arms[1:]:
            candidate = arm.initial_state
            self.assertEqual(candidate.source_ids[:count], control.source_ids)
            self.assertEqual(len(candidate.source_ids), count + 1)
            for control_array, candidate_array in (
                (control.positions_AU, candidate.positions_AU),
                (control.velocities_AU_per_year, candidate.velocities_AU_per_year),
                (control.gm_AU3_per_year2, candidate.gm_AU3_per_year2),
            ):
                self.assertEqual(
                    control_array.tobytes(order="C"),
                    candidate_array[:count].tobytes(order="C"),
                )
            self.assertEqual(
                candidate.base_prefix_sha256,
                control.base_prefix_sha256,
            )

    def test_all_sources_including_perturbers_are_active_full_horizon(self) -> None:
        for arm in self.schedule_set.arms:
            schedule = arm.schedule
            self.assertEqual(schedule.force_transition_steps, ())
            self.assertEqual(schedule.source_activity_semantics, SOURCE_ACTIVITY_SEMANTICS)
            self.assertEqual(
                tuple(row.source_id for row in schedule.source_activity),
                schedule.source_ids,
            )
            for interval in schedule.source_activity:
                self.assertEqual(interval.start_global_step, 0)
                self.assertEqual(
                    interval.end_global_step_exclusive,
                    schedule.total_steps,
                )
            if arm is not self.schedule_set.control:
                self.assertEqual(
                    schedule.source_activity[-1].source_id,
                    schedule.perturber.source_id,
                )

    def test_force_semantics_require_mutual_backreaction_and_all_source_tracer_force(self) -> None:
        for arm in self.schedule_set.arms:
            schedule = arm.schedule
            self.assertEqual(schedule.massive_force_semantics, MASSIVE_FORCE_SEMANTICS)
            self.assertEqual(schedule.tracer_force_semantics, TRACER_FORCE_SEMANTICS)
            self.assertEqual(
                schedule.tracer_backreaction_semantics,
                TRACER_BACKREACTION_SEMANTICS,
            )
            record = schedule.canonical_record()
            self.assertNotIn("force_callback", record)
            self.assertEqual(record["force_transition_steps"], [])

    def test_passage_windows_are_annotations_only(self) -> None:
        self.assertIsNone(self.schedule_set.control.schedule.diagnostic_window)
        self.assertIsNone(
            self.schedule_set.persistent_distant_planet.schedule.diagnostic_window
        )
        for arm in (
            self.schedule_set.stellar_flyby,
            self.schedule_set.temporary_rogue,
        ):
            window = arm.schedule.diagnostic_window
            self.assertEqual(window.nominal_event_global_step, 1_000)
            self.assertEqual(window.start_global_step, 980)
            self.assertEqual(window.end_global_step_exclusive, 1_021)
            self.assertEqual(window.dynamics_effect, "NONE")
            self.assertEqual(
                arm.schedule.perturber.trajectory_semantics,
                "DYNAMIC_MUTUAL_SOURCE_NOT_PRESCRIBED",
            )

    def test_synthetic_states_have_declared_nominal_geometry(self) -> None:
        base = self.schedule_set.control.initial_state
        primary_position = base.positions_AU[0]
        primary_velocity = base.velocities_AU_per_year[0]
        planet = self.schedule_set.persistent_distant_planet.schedule.perturber
        planet_position = np.asarray(planet.initial_position_AU) - primary_position
        planet_velocity = (
            np.asarray(planet.initial_velocity_AU_per_year) - primary_velocity
        )
        self.assertAlmostEqual(float(np.linalg.norm(planet_position)), 500.0)
        self.assertAlmostEqual(float(np.dot(planet_position, planet_velocity)), 0.0)

        for arm, impact in (
            (self.schedule_set.stellar_flyby, 1_000.0),
            (self.schedule_set.temporary_rogue, 100.0),
        ):
            perturber = arm.schedule.perturber
            relative_position = (
                np.asarray(perturber.initial_position_AU) - primary_position
            )
            relative_velocity = (
                np.asarray(perturber.initial_velocity_AU_per_year)
                - primary_velocity
            )
            nominal_position = relative_position + relative_velocity * 500.0
            self.assertAlmostEqual(float(np.linalg.norm(nominal_position)), impact)
            self.assertAlmostEqual(
                float(np.dot(nominal_position, relative_velocity)),
                0.0,
                places=10,
            )

    def test_manifests_retain_screening_nonclaims_and_hex_float_clock(self) -> None:
        for arm in self.schedule_set.arms:
            schedule = arm.schedule
            record = schedule.canonical_record()
            self.assertEqual(record["scientific_classification"], SCIENTIFIC_CLASSIFICATION)
            self.assertEqual(record["project_scope"], PROJECT_SCOPE)
            self.assertEqual(record["initial_condition_scope"], INITIAL_CONDITION_SCOPE)
            self.assertFalse(record["primordial_initial_conditions"])
            self.assertEqual(record["clock"]["step_years_hex"], float(0.5).hex())
            self.assertTrue(all(value is False for value in record["claims"].values()))
            self.assertEqual(record["claims"], dict(RESULT_CLAIMS))
            self.assertEqual(
                record["unsupported_capabilities"],
                list(UNSUPPORTED_CAPABILITIES),
            )
            self.assertIn(
                "SOURCE_ACTIVATION_DEACTIVATION_OR_GM_RAMP",
                record["unsupported_capabilities"],
            )
            self.assertFalse(any(type(value) is float for value in _walk(record)))
            self.assertFalse(
                any(
                    type(value) is float
                    for value in _walk(arm.initial_state.canonical_metadata())
                )
            )

    def test_schedule_hash_binds_clock_and_diagnostic_window(self) -> None:
        changed_clock = build_de441_four_arm_schedule_set(
            CONTROL_PATH,
            step_years=0.25,
            total_steps=2_000,
            nominal_passage_global_step=1_000,
            diagnostic_half_window_steps=20,
        )
        changed_window = build_de441_four_arm_schedule_set(
            CONTROL_PATH,
            step_years=0.5,
            total_steps=2_000,
            nominal_passage_global_step=1_000,
            diagnostic_half_window_steps=21,
        )
        self.assertNotEqual(
            self.schedule_set.matched_set_sha256,
            changed_clock.matched_set_sha256,
        )
        self.assertNotEqual(
            self.schedule_set.matched_set_sha256,
            changed_window.matched_set_sha256,
        )

    def test_contract_rejects_force_switch_or_state_schedule_mismatch(self) -> None:
        control = self.schedule_set.control
        wrong_activity = tuple(
            SourceActivityInterval(source_id, 0, 1_999)
            for source_id in control.schedule.source_ids
        )
        with self.assertRaisesRegex(MechanismScheduleError, "complete modeled horizon"):
            replace(control.schedule, source_activity=wrong_activity)
        with self.assertRaisesRegex(MechanismScheduleError, "force-transition"):
            replace(control.schedule, force_transition_steps=(1_000,))
        with self.assertRaisesRegex(MechanismScheduleError, "semantic hash"):
            MechanismArm(
                replace(control.schedule, source_state_sha256="1" * 64),
                control.initial_state,
            )

    def test_matched_set_rejects_duplicate_or_misordered_arm(self) -> None:
        schedules = self.schedule_set
        with self.assertRaisesRegex(MechanismScheduleError, "arm order"):
            MatchedMechanismScheduleSet(
                matched_set_id=schedules.matched_set_id,
                control=schedules.control,
                persistent_distant_planet=schedules.stellar_flyby,
                stellar_flyby=schedules.persistent_distant_planet,
                temporary_rogue=schedules.temporary_rogue,
            )


class SyntheticBuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_frozen_de441_outer_control(CONTROL_PATH)

    def test_builders_reject_nonphysical_or_ambiguous_values(self) -> None:
        with self.assertRaises(MechanismScheduleError):
            build_synthetic_persistent_distant_planet(self.base, mass_earth=0.0)
        with self.assertRaises(MechanismScheduleError):
            build_synthetic_persistent_distant_planet(
                self.base, source_id="Jupiter"
            )
        with self.assertRaises(MechanismScheduleError):
            build_synthetic_stellar_flyby(
                self.base,
                nominal_event_global_step=0,
                step_years=0.5,
            )
        with self.assertRaises(MechanismScheduleError):
            build_synthetic_temporary_rogue(
                self.base,
                nominal_event_global_step=100,
                step_years=-0.5,
            )

    def test_four_arm_builder_requires_interior_lattice_passage(self) -> None:
        for invalid in (0, 2_000, 2_001):
            with self.assertRaises(MechanismScheduleError):
                build_de441_four_arm_schedule_set(
                    CONTROL_PATH,
                    step_years=0.5,
                    total_steps=2_000,
                    nominal_passage_global_step=invalid,
                )


if __name__ == "__main__":
    unittest.main()
