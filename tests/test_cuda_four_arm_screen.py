from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmarks.jx_cuda_four_arm_screen import (
    ARM_DIRECTORY_NAMES,
    ARM_KINDS,
    LIFECYCLE_ACTIVE,
    MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
    MAXIMUM_TOTAL_ROUTE_RECORDS_PER_STAGE,
    MINIMUM_FREE_STORAGE_BYTES,
    ROUTE_CAPACITY_ADMISSION_LIMIT,
    ROUTE_CAPACITY_PILOT_STEPS,
    ROUTE_CAPACITY_REQUIRED_HEADROOM,
    ROUTE_EPISODE_RESERVED_RECORDS,
    STAGES,
    DEFAULT_ACCEPTANCE_PROTOCOL,
    DEFAULT_ACCEPTANCE_STATE,
    DEFAULT_DE441_CONTROL,
    _build_stage_schedule_set,
    _build_protocol,
    _capacity_admission_document,
    _capacity_target_context,
    _derive_population,
    _finalized_summary_gates,
    _has_stage_finalization_artifact,
    _load_partial_summary_commit,
    _load_verified_capacity_admission,
    _route_storage_guard,
    _schedule_prefix_physics_binding,
    _source_activity_is_exact_prefix,
    _route_episode_capacity_screen,
    main,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_ARMS = (
    ROOT / "runs/cuda_four_arm_screen_v1/qualification/arms"
)


def _route(
    tracer_index: int,
    global_step: int,
    *,
    source_id: str = "Neptune",
    reason: str = "SWEPT_FLOOR",
    outcome: str = "REJOINED",
) -> dict[str, object]:
    return {
        "tracer_index": tracer_index,
        "global_step": global_step,
        "source_id": source_id,
        "reason": reason,
        "outcome": outcome,
    }


def _screen(
    records: list[dict[str, object]],
    *,
    tracer_ids: tuple[int, ...] = (0,),
    terminal_ids: tuple[int, ...] = (),
    observed_steps: int = ROUTE_CAPACITY_PILOT_STEPS,
    target_steps: int = STAGES["one_myr"].total_steps,
) -> dict[str, object]:
    return _route_episode_capacity_screen(
        route_records_by_arm={"ARM": records},
        tracer_ids_by_arm={"ARM": tracer_ids},
        terminal_tracer_ids_by_arm={"ARM": terminal_ids},
        observed_steps=observed_steps,
        target_steps=target_steps,
        expected_arm_kinds=("ARM",),
        required_pilot_steps=ROUTE_CAPACITY_PILOT_STEPS,
        required_tracer_count=len(tracer_ids),
    )


class RouteEpisodeDefinitionTests(unittest.TestCase):
    def test_source_and_reason_changes_do_not_split_but_one_quiet_step_does(self) -> None:
        records = [
            _route(0, 5),
            _route(0, 2, source_id="Neptune", reason="SWEPT_FLOOR"),
            _route(0, 3, source_id="P9_SYNTHETIC", reason="ACCELERATION"),
        ]
        result = _screen(
            records,
            observed_steps=ROUTE_CAPACITY_PILOT_STEPS,
            target_steps=ROUTE_CAPACITY_PILOT_STEPS,
        )
        arm = result["arm_diagnostics"]["ARM"]
        self.assertEqual(arm["route_record_count"], 3)
        self.assertEqual(arm["episode_count"], 2)
        self.assertEqual(arm["completed_episode_count"], 2)
        self.assertEqual(arm["longest_episode_route_records"], 2)
        self.assertEqual(
            arm["episode_source_membership_counts"],
            {"Neptune": 2, "P9_SYNTHETIC": 1},
        )

    def test_unsorted_input_has_deterministic_episode_commitment(self) -> None:
        ordered = [_route(0, step) for step in (10, 11, 12, 20)]
        left = _screen(ordered)
        right = _screen(list(reversed(ordered)))
        self.assertEqual(
            left["episode_descriptor_sha256"],
            right["episode_descriptor_sha256"],
        )
        self.assertEqual(left["arm_diagnostics"], right["arm_diagnostics"])

    def test_duplicate_tracer_step_is_rejected_even_when_source_differs(self) -> None:
        records = [
            _route(0, 10, source_id="Neptune"),
            _route(0, 10, source_id="P9_SYNTHETIC"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate routes"):
            _screen(records)

    def test_invalid_route_outcome_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid outcome"):
            _screen([_route(0, 10, outcome="NOT_AN_ENGINE_OUTCOME")])

    def test_checkpoint_split_consecutive_routes_form_one_episode(self) -> None:
        before_checkpoint = [_route(0, step) for step in range(3_837, 4_000)]
        after_checkpoint = [_route(0, step) for step in range(4_000, 4_033)]
        result = _screen(before_checkpoint + after_checkpoint)
        arm = result["arm_diagnostics"]["ARM"]
        self.assertEqual(len(before_checkpoint), 163)
        self.assertEqual(len(after_checkpoint), 33)
        self.assertEqual(arm["route_record_count"], 196)
        self.assertEqual(arm["episode_count"], 1)
        self.assertEqual(arm["completed_episode_count"], 1)
        self.assertEqual(arm["longest_episode_route_records"], 196)

    def test_terminal_route_closes_preceding_rejoined_steps(self) -> None:
        records = [_route(0, step) for step in range(100, 105)]
        records.append(_route(0, 105, outcome="REMOVED_OUTER_BOUNDARY"))
        result = _screen(records, terminal_ids=(0,))
        arm = result["arm_diagnostics"]["ARM"]
        self.assertEqual(arm["episode_count"], 1)
        self.assertEqual(arm["completed_episode_count"], 1)
        self.assertEqual(arm["right_censored_rejoined_episode_count"], 0)
        self.assertEqual(result["worst_lane"]["reserved_future_episode_slots"], 0)
        self.assertTrue(result["one_myr_capacity_screen_passed"])

    def test_mismatched_arm_tracer_rosters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tracer rosters"):
            _route_episode_capacity_screen(
                route_records_by_arm={"LEFT": [], "RIGHT": []},
                tracer_ids_by_arm={"LEFT": (0, 1), "RIGHT": (0, 2)},
                terminal_tracer_ids_by_arm={"LEFT": (), "RIGHT": ()},
                observed_steps=ROUTE_CAPACITY_PILOT_STEPS,
                target_steps=STAGES["one_myr"].total_steps,
                expected_arm_kinds=("LEFT", "RIGHT"),
                required_pilot_steps=ROUTE_CAPACITY_PILOT_STEPS,
                required_tracer_count=2,
            )

    def test_unknown_route_outcome_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid outcome"):
            _screen([_route(0, 10, outcome="UNKNOWN")])


class RouteEpisodeAdmissionFormulaTests(unittest.TestCase):
    def test_add_one_exposure_reserves_nine_slots_for_unseen_episode(self) -> None:
        result = _screen([], tracer_ids=(0, 1))
        arm = result["arm_diagnostics"]["ARM"]
        self.assertEqual(arm["future_episode_slot_distribution"], {"9": 2})
        self.assertEqual(
            arm["maximum_planned_routes_for_one_tracer"],
            9 * ROUTE_EPISODE_RESERVED_RECORDS,
        )
        self.assertTrue(result["one_myr_capacity_screen_passed"])

    def test_one_completed_episode_passes_but_two_fail_budget(self) -> None:
        one_episode = [_route(0, step) for step in range(100, 110)]
        passing = _screen(one_episode)
        passing_lane = passing["worst_lane"]
        self.assertEqual(passing_lane["completed_episode_count"], 1)
        self.assertEqual(passing_lane["reserved_future_episode_slots"], 18)
        self.assertEqual(
            passing_lane["planned_route_records"],
            10 + 18 * ROUTE_EPISODE_RESERVED_RECORDS,
        )
        self.assertTrue(passing["one_myr_capacity_screen_passed"])

        two_episodes = one_episode + [
            _route(0, step) for step in range(200, 210)
        ]
        failing = _screen(two_episodes)
        failing_lane = failing["worst_lane"]
        self.assertEqual(failing_lane["completed_episode_count"], 2)
        self.assertEqual(failing_lane["reserved_future_episode_slots"], 27)
        self.assertGreater(
            failing_lane["planned_route_records"],
            ROUTE_CAPACITY_ADMISSION_LIMIT,
        )
        self.assertFalse(failing["one_myr_capacity_screen_passed"])

    def test_active_frontier_episode_fails_but_terminal_frontier_does_not(self) -> None:
        frontier = ROUTE_CAPACITY_PILOT_STEPS - 1
        active = _screen([_route(0, frontier)])
        self.assertEqual(active["right_censored_rejoined_episode_count"], 1)
        self.assertFalse(
            active["gates"]["no_right_censored_rejoined_episode"]
        )
        self.assertFalse(active["one_myr_capacity_screen_passed"])

        terminal = _screen(
            [_route(0, frontier, outcome="REMOVED_OUTER_BOUNDARY")],
            terminal_ids=(0,),
        )
        self.assertEqual(terminal["right_censored_rejoined_episode_count"], 0)
        self.assertEqual(terminal["worst_lane"]["reserved_future_episode_slots"], 0)
        self.assertTrue(terminal["one_myr_capacity_screen_passed"])

    def test_terminal_outcome_closes_a_prior_rejoined_run(self) -> None:
        terminal = _screen(
            [
                _route(0, ROUTE_CAPACITY_PILOT_STEPS - 2),
                _route(
                    0,
                    ROUTE_CAPACITY_PILOT_STEPS - 1,
                    outcome="REMOVED_SOURCE_RADIUS",
                ),
            ],
            terminal_ids=(0,),
        )
        arm = terminal["arm_diagnostics"]["ARM"]
        self.assertEqual(arm["episode_count"], 1)
        self.assertEqual(arm["completed_episode_count"], 1)
        self.assertEqual(arm["right_censored_rejoined_episode_count"], 0)
        self.assertTrue(terminal["one_myr_capacity_screen_passed"])

    def test_active_episode_larger_than_reservation_fails_independently_of_budget(self) -> None:
        records = [
            _route(0, step)
            for step in range(100, 100 + ROUTE_EPISODE_RESERVED_RECORDS + 1)
        ]
        result = _screen(
            records,
            target_steps=ROUTE_CAPACITY_PILOT_STEPS,
        )
        self.assertLessEqual(
            result["maximum_planned_routes_for_one_tracer"],
            ROUTE_CAPACITY_ADMISSION_LIMIT,
        )
        self.assertFalse(
            result["gates"]["active_observed_episodes_fit_future_reservation"]
        )
        self.assertFalse(result["one_myr_capacity_screen_passed"])

    def test_fixed_headroom_is_below_unchanged_hard_cap(self) -> None:
        result = _screen([])
        self.assertEqual(result["engine_hard_capacity_per_tracer"], 16_384)
        self.assertEqual(
            result["required_runtime_headroom_per_tracer"],
            ROUTE_CAPACITY_REQUIRED_HEADROOM,
        )
        self.assertEqual(
            result["admission_limit_per_tracer"]
            + result["required_runtime_headroom_per_tracer"],
            result["engine_hard_capacity_per_tracer"],
        )

    def test_one_episode_budget_boundary_is_3072_routes(self) -> None:
        at_limit = _screen([_route(0, step) for step in range(100, 3_172)])
        over_limit = _screen([_route(0, step) for step in range(100, 3_173)])

        self.assertEqual(at_limit["worst_lane"]["completed_episode_count"], 1)
        self.assertEqual(
            at_limit["worst_lane"]["planned_route_records"],
            ROUTE_CAPACITY_ADMISSION_LIMIT,
        )
        self.assertTrue(
            at_limit["gates"][
                "every_lane_planned_routes_at_or_below_admission_limit"
            ]
        )
        self.assertFalse(
            at_limit["gates"]["active_observed_episodes_fit_future_reservation"]
        )

        self.assertEqual(
            over_limit["worst_lane"]["planned_route_records"],
            ROUTE_CAPACITY_ADMISSION_LIMIT + 1,
        )
        self.assertFalse(
            over_limit["gates"][
                "every_lane_planned_routes_at_or_below_admission_limit"
            ]
        )


class PreservedQualificationEpisodeRegressionTests(unittest.TestCase):
    def test_old_mechanics_qualification_is_descriptive_but_non_authorizing(self) -> None:
        route_records_by_arm = {}
        tracer_ids_by_arm = {}
        terminal_ids_by_arm = {}
        for arm_kind in ARM_KINDS:
            arm_directory = QUALIFICATION_ARMS / ARM_DIRECTORY_NAMES[arm_kind]
            route_document = json.loads(
                (arm_directory / "ENCOUNTER_ROUTES.json").read_text(
                    encoding="utf-8"
                )
            )
            route_records_by_arm[arm_kind] = route_document["records"]
            with np.load(
                arm_directory / "ENDPOINT_STATE.npz", allow_pickle=False
            ) as archive:
                tracer_indices = np.asarray(archive["tracer_indices"])
                lifecycle = np.asarray(archive["tracer_lifecycle"])
            tracer_ids_by_arm[arm_kind] = tuple(
                int(value) for value in tracer_indices
            )
            terminal_ids_by_arm[arm_kind] = tuple(
                int(tracer_indices[index])
                for index in np.flatnonzero(lifecycle != LIFECYCLE_ACTIVE)
            )

        result = _route_episode_capacity_screen(
            route_records_by_arm=route_records_by_arm,
            tracer_ids_by_arm=tracer_ids_by_arm,
            terminal_tracer_ids_by_arm=terminal_ids_by_arm,
            observed_steps=STAGES["qualification"].total_steps,
            target_steps=STAGES["one_myr"].total_steps,
            expected_arm_kinds=ARM_KINDS,
            required_pilot_steps=ROUTE_CAPACITY_PILOT_STEPS,
            required_tracer_count=STAGES["one_myr"].tracer_count,
        )
        self.assertEqual(result["total_route_record_count"], 1_178)
        self.assertEqual(result["total_episode_count"], 12)
        self.assertEqual(result["total_completed_episode_count"], 11)
        self.assertEqual(result["longest_episode_route_records"], 196)
        self.assertEqual(result["right_censored_rejoined_episode_count"], 1)
        self.assertFalse(
            result["gates"]["required_capacity_pilot_step_count"]
        )
        self.assertFalse(
            result["gates"]["required_tracer_count_in_every_arm"]
        )
        self.assertFalse(result["one_myr_capacity_screen_passed"])


class CapacityPilotOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pilot_stage = STAGES["capacity_pilot"]
        cls.target_stage = STAGES["one_myr"]
        cls.qualification_stage = STAGES["qualification"]
        cls.pilot_schedules = _build_stage_schedule_set(
            cls.pilot_stage, DEFAULT_DE441_CONTROL
        )
        cls.target_schedules = _build_stage_schedule_set(
            cls.target_stage, DEFAULT_DE441_CONTROL
        )
        cls.qualification_schedules = _build_stage_schedule_set(
            cls.qualification_stage, DEFAULT_DE441_CONTROL
        )
        cls.pilot_population, cls.pilot_receipt = _derive_population(
            DEFAULT_ACCEPTANCE_STATE, cls.pilot_schedules
        )
        cls.target_context = _capacity_target_context(
            pilot_stage=cls.pilot_stage,
            pilot_schedule_set=cls.pilot_schedules,
            pilot_population=cls.pilot_population,
            pilot_receipt=cls.pilot_receipt,
            de441_control=DEFAULT_DE441_CONTROL,
            acceptance_state=DEFAULT_ACCEPTANCE_STATE,
        )

    def test_stage_clock_population_and_passage_are_exact(self) -> None:
        self.assertEqual(self.pilot_stage.duration_years, 100_000.0)
        self.assertEqual(self.pilot_stage.total_steps, 800_000)
        self.assertEqual(self.pilot_stage.tracer_count, 16_800)
        self.assertEqual(self.pilot_stage.checkpoint_years, 25_000.0)
        self.assertEqual(self.target_stage.total_steps, 8_000_000)
        for schedule_set in (self.pilot_schedules, self.target_schedules):
            for arm in (schedule_set.stellar_flyby, schedule_set.temporary_rogue):
                self.assertEqual(
                    arm.schedule.diagnostic_window.nominal_event_global_step,
                    400_000,
                )

    def test_protocol_freezes_global_route_and_storage_budgets(self) -> None:
        protocol = _build_protocol(
            self.pilot_stage,
            self.pilot_stage.checkpoint_years,
            self.pilot_schedules,
            self.pilot_receipt,
            "cuda:0",
            capacity_target_context=self.target_context,
        )
        budget = protocol["route_ledger_budget"]
        self.assertEqual(
            protocol["integrator"]["maximum_total_encounter_routes"],
            MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
        )
        self.assertEqual(
            budget["maximum_total_route_records_per_arm"],
            MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
        )
        self.assertEqual(
            budget["maximum_total_route_records_across_four_arms"],
            MAXIMUM_TOTAL_ROUTE_RECORDS_PER_STAGE,
        )
        self.assertEqual(
            budget["minimum_free_storage_bytes_before_every_segment"],
            MINIMUM_FREE_STORAGE_BYTES,
        )
        self.assertEqual(
            budget["minimum_free_storage_bytes_before_every_checkpoint"],
            MINIMUM_FREE_STORAGE_BYTES,
        )

    def test_pilot_is_exact_physical_prefix_but_not_same_schedule_identity(self) -> None:
        pilot_physics = _schedule_prefix_physics_binding(self.pilot_schedules)
        target_physics = _schedule_prefix_physics_binding(self.target_schedules)
        self.assertEqual(pilot_physics, target_physics)
        self.assertTrue(
            _source_activity_is_exact_prefix(
                self.pilot_schedules, self.target_schedules
            )
        )
        self.assertEqual(
            [arm.initial_state.state_sha256 for arm in self.pilot_schedules.arms],
            [arm.initial_state.state_sha256 for arm in self.target_schedules.arms],
        )
        self.assertNotEqual(
            [arm.schedule.schedule_sha256 for arm in self.pilot_schedules.arms],
            [arm.schedule.schedule_sha256 for arm in self.target_schedules.arms],
        )

    def test_old_qualification_cannot_substitute_for_capacity_pilot(self) -> None:
        self.assertNotEqual(
            _schedule_prefix_physics_binding(self.qualification_schedules),
            _schedule_prefix_physics_binding(self.target_schedules),
        )

    def test_target_context_binds_roster_population_and_initial_bytes(self) -> None:
        context = self.target_context
        self.assertTrue(context["gates"]["all_pass"])
        self.assertEqual(context["target_stage_id"], "one_myr")
        self.assertEqual(context["target_total_steps"], 8_000_000)
        self.assertEqual(context["tracer_count_per_arm"], 16_800)
        self.assertEqual(
            context["pilot_schedule_prefix_physics"],
            context["target_schedule_prefix_physics"],
        )
        self.assertEqual(
            context["pilot_full_initial_state_semantic_sha256"],
            context["target_full_initial_state_semantic_sha256"],
        )
        self.assertEqual(
            context["pilot_population_transform_receipt_sha256"],
            context["target_population_transform_receipt_sha256"],
        )

    def test_admission_rejects_wrong_target_frontier_or_roster_binding(self) -> None:
        protocol = _build_protocol(
            self.pilot_stage,
            self.pilot_stage.checkpoint_years,
            self.pilot_schedules,
            self.pilot_receipt,
            "cuda:0",
            capacity_target_context=self.target_context,
        )
        reports = {
            arm_kind: {
                "content_sha256": f"{index + 1:064x}",
                "gates": {"all_pass": True},
            }
            for index, arm_kind in enumerate(ARM_KINDS)
        }
        screen = {
            "one_myr_capacity_screen_passed": True,
            "observed_steps": ROUTE_CAPACITY_PILOT_STEPS,
            "target_steps": self.target_stage.total_steps,
            "required_tracer_count_per_arm": self.target_stage.tracer_count,
            "arm_diagnostics": {arm_kind: {} for arm_kind in ARM_KINDS},
        }
        initial_hashes = self.target_context[
            "pilot_full_initial_state_semantic_sha256"
        ]
        with mock.patch(
            "benchmarks.jx_cuda_four_arm_screen._artifact_initial_state_hashes",
            return_value=initial_hashes,
        ), mock.patch(
            "benchmarks.jx_cuda_four_arm_screen._sha256_file",
            return_value="f" * 64,
        ):
            passing = _capacity_admission_document(
                stage=self.pilot_stage,
                checkpoint_years=self.pilot_stage.checkpoint_years,
                stage_directory=Path("/not-read-under-mock"),
                schedule_set=self.pilot_schedules,
                receipt=self.pilot_receipt,
                protocol=protocol,
                runtime_identity={"device": "test"},
                runtime_receipt_sha256="e" * 64,
                reports=reports,
                route_episode_capacity_screen=screen,
                target_context=self.target_context,
            )
            self.assertTrue(passing["one_myr_authorized"])

            wrong_target = dict(screen)
            wrong_target["target_steps"] -= 1
            rejected_target = _capacity_admission_document(
                stage=self.pilot_stage,
                checkpoint_years=self.pilot_stage.checkpoint_years,
                stage_directory=Path("/not-read-under-mock"),
                schedule_set=self.pilot_schedules,
                receipt=self.pilot_receipt,
                protocol=protocol,
                runtime_identity={"device": "test"},
                runtime_receipt_sha256="e" * 64,
                reports=reports,
                route_episode_capacity_screen=wrong_target,
                target_context=self.target_context,
            )
            self.assertFalse(rejected_target["one_myr_authorized"])
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "CAPACITY_ADMISSION.json"
                path.write_text(
                    json.dumps(rejected_target, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                loaded = _load_verified_capacity_admission(
                    path, expected_authorized=False
                )
                self.assertFalse(loaded["one_myr_authorized"])
                with self.assertRaisesRegex(RuntimeError, "terminal decision"):
                    _load_verified_capacity_admission(path)

            wrong_roster = copy.deepcopy(self.target_context)
            wrong_roster["gates"]["exact_arm_rosters"] = False
            wrong_roster["gates"]["all_pass"] = False
            rejected_roster = _capacity_admission_document(
                stage=self.pilot_stage,
                checkpoint_years=self.pilot_stage.checkpoint_years,
                stage_directory=Path("/not-read-under-mock"),
                schedule_set=self.pilot_schedules,
                receipt=self.pilot_receipt,
                protocol=protocol,
                runtime_identity={"device": "test"},
                runtime_receipt_sha256="e" * 64,
                reports=reports,
                route_episode_capacity_screen=screen,
                target_context=wrong_roster,
            )
            self.assertFalse(rejected_roster["one_myr_authorized"])

    def test_capacity_pilot_prepare_only_needs_no_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run(
                stage=self.pilot_stage,
                output_root=Path(temporary),
                acceptance_state=DEFAULT_ACCEPTANCE_STATE,
                acceptance_protocol=DEFAULT_ACCEPTANCE_PROTOCOL,
                de441_control=DEFAULT_DE441_CONTROL,
                checkpoint_years=self.pilot_stage.checkpoint_years,
                device="cuda:0",
                selected_arm_kinds=ARM_KINDS,
                stop_after_checkpoints=None,
                prepare_only=True,
            )
            self.assertEqual(result["status"], "PREPARED_NOT_EXECUTED")
            self.assertEqual(result["stage_id"], "capacity_pilot")
            protocol_path = Path(temporary) / "capacity_pilot/PROTOCOL.json"
            self.assertTrue(protocol_path.is_file())
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertTrue(
                protocol["capacity_target_binding"]["gates"]["all_pass"]
            )
            self.assertFalse(
                (Path(temporary) / "capacity_pilot/CAPACITY_ADMISSION.json").exists()
            )

    def test_one_myr_prepare_only_fails_without_capacity_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "capacity_pilot"):
                run(
                    stage=self.target_stage,
                    output_root=output_root,
                    acceptance_state=DEFAULT_ACCEPTANCE_STATE,
                    acceptance_protocol=DEFAULT_ACCEPTANCE_PROTOCOL,
                    de441_control=DEFAULT_DE441_CONTROL,
                    checkpoint_years=self.target_stage.checkpoint_years,
                    device="cuda:0",
                    selected_arm_kinds=ARM_KINDS,
                    stop_after_checkpoints=None,
                    prepare_only=True,
                )
            self.assertFalse((output_root / "one_myr").exists())

    def test_one_myr_runtime_mismatch_fails_before_target_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.jx_cuda_four_arm_screen._capacity_pilot_prerequisite",
            side_effect=(None, RuntimeError("pilot runtime mismatch")),
        ) as prerequisite, mock.patch(
            "benchmarks.jx_cuda_four_arm_screen._current_runtime_identity",
            return_value={"device_index": 0},
        ):
            output_root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
                run(
                    stage=self.target_stage,
                    output_root=output_root,
                    acceptance_state=DEFAULT_ACCEPTANCE_STATE,
                    acceptance_protocol=DEFAULT_ACCEPTANCE_PROTOCOL,
                    de441_control=DEFAULT_DE441_CONTROL,
                    checkpoint_years=self.target_stage.checkpoint_years,
                    device="cuda:0",
                    selected_arm_kinds=ARM_KINDS,
                    stop_after_checkpoints=None,
                    prepare_only=False,
                )
            self.assertEqual(prerequisite.call_count, 2)
            self.assertFalse((output_root / "one_myr").exists())


class StorageAndFinalizationGuardTests(unittest.TestCase):
    def test_storage_guard_enforces_count_and_free_space_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.jx_cuda_four_arm_screen.shutil.disk_usage",
            return_value=mock.Mock(free=MINIMUM_FREE_STORAGE_BYTES),
        ):
            observed = _route_storage_guard(
                Path(temporary),
                phase="unit-test checkpoint",
                cumulative_route_records=MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
            )
            self.assertTrue(observed["passed"])
            self.assertEqual(
                observed["required_minimum_free_storage_bytes"],
                MINIMUM_FREE_STORAGE_BYTES,
            )

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.jx_cuda_four_arm_screen.shutil.disk_usage",
            return_value=mock.Mock(free=MINIMUM_FREE_STORAGE_BYTES - 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient free storage"):
                _route_storage_guard(
                    Path(temporary),
                    phase="unit-test segment",
                    cumulative_route_records=0,
                )
            with self.assertRaisesRegex(RuntimeError, "budget was exceeded"):
                _route_storage_guard(
                    Path(temporary),
                    phase="unit-test checkpoint",
                    cumulative_route_records=(
                        MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM + 1
                    ),
                )

    def test_failed_capacity_summary_is_a_supported_non_authorizing_terminal(self) -> None:
        failed = _finalized_summary_gates(
            STAGES["capacity_pilot"], "FAILED_ROUTE_CAPACITY_GATE"
        )
        self.assertTrue(failed["all_four_arms_complete"])
        self.assertTrue(failed["all_arm_gates_pass"])
        self.assertFalse(failed["route_episode_capacity_gate_passed"])
        self.assertFalse(failed["all_pass"])
        with self.assertRaisesRegex(RuntimeError, "supported finalized status"):
            _finalized_summary_gates(
                STAGES["one_myr"], "FAILED_ROUTE_CAPACITY_GATE"
            )

    def test_summary_only_commit_is_recoverable_and_sha_without_summary_is_not(self) -> None:
        source = ROOT / "runs/cuda_four_arm_screen_v1/qualification/SUMMARY.json"
        with tempfile.TemporaryDirectory() as temporary:
            stage_directory = Path(temporary)
            destination = stage_directory / "SUMMARY.json"
            destination.write_bytes(source.read_bytes())
            recovered = _load_partial_summary_commit(destination)
            self.assertEqual(recovered, json.loads(source.read_text(encoding="utf-8")))
            (stage_directory / "SHA256SUMS").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not in the recoverable"):
                _load_partial_summary_commit(destination)

    def test_main_never_appends_failure_inside_finalization_artifacts(self) -> None:
        for artifact_name in ("SUMMARY.json", "SHA256SUMS"):
            with self.subTest(artifact_name=artifact_name), tempfile.TemporaryDirectory() as temporary:
                stage_directory = Path(temporary) / "capacity_pilot"
                stage_directory.mkdir(parents=True)
                (stage_directory / artifact_name).write_text("committed\n", encoding="utf-8")
                self.assertTrue(_has_stage_finalization_artifact(stage_directory))
                with mock.patch(
                    "sys.argv",
                    [
                        "jx_cuda_four_arm_screen.py",
                        "--stage",
                        "capacity_pilot",
                        "--output-root",
                        temporary,
                    ],
                ), mock.patch(
                    "benchmarks.jx_cuda_four_arm_screen.run",
                    side_effect=RuntimeError("rerun identity mismatch"),
                ), mock.patch(
                    "benchmarks.jx_cuda_four_arm_screen._write_failure"
                ) as write_failure:
                    self.assertEqual(main(), 1)
                    write_failure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
