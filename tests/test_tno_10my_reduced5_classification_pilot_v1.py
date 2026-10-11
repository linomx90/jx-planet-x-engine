from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_classification_pilot_v1.py"


def _source_only_load(name: str) -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(
            compile(
                payload,
                str(SOURCE),
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


PILOT = _source_only_load("_test_jx_tno_10my_reduced5_classification_pilot_v1")


class Tno10MyrReduced5ClassificationPilotV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = PILOT.build_protocol()

    def test_source_only_import_has_no_dynamics_runtime_or_execution_surface(self) -> None:
        before = set(sys.modules)
        module = _source_only_load(
            f"_test_jx_tno_10my_reduced5_import_{sys.flags.optimize}"
        )
        added = set(sys.modules) - before
        self.assertFalse(any(name == "cupy" or name.startswith("cupy.") for name in added))
        self.assertFalse(any(name == "numpy" or name.startswith("numpy.") for name in added))
        self.assertFalse(any(name == "rebound" or name.startswith("rebound.") for name in added))
        self.assertFalse(hasattr(module, "execute"))
        self.assertFalse(hasattr(module, "run"))
        self.assertEqual(
            module.build_protocol()["implementation"]["cli_surface"],
            "BUILD_PREPARE_VALIDATE_ONLY",
        )

    def test_scope_and_reduced_five_dynamics_are_exact(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], PILOT.SCHEMA)
        self.assertEqual(protocol["classification"], PILOT.CLASSIFICATION)
        self.assertEqual(protocol["design_status"], "PREPARE_ONLY_EXECUTION_BLOCKED")
        cohort = protocol["cohort"]
        self.assertEqual(cohort["object_ids_in_exact_order"], list(PILOT.OBJECT_IDS))
        self.assertEqual(cohort["holdout_clone_indices"], list(PILOT.HOLDOUT_CLONE_INDICES))
        self.assertEqual(cohort["total_state_count"], 108)
        self.assertIs(cohort["full_1548_state_execution"], False)

        dynamics = protocol["dynamics"]
        self.assertEqual(dynamics["ephemeris"], "DE441_NATIVE_EPOCH_STATE_SOURCE")
        self.assertEqual(dynamics["source_ids_in_exact_order"], list(PILOT.REDUCED_SOURCE_IDS))
        self.assertEqual(dynamics["horizon_julian_years"], 10_000_000)
        self.assertEqual(dynamics["step_julian_years"]["exact_rational"], "1/64")
        self.assertEqual(dynamics["step_julian_years"]["binary64_hex"], "0x1.0000000000000p-6")
        self.assertEqual(dynamics["production_lane_count"], 12)
        self.assertEqual(dynamics["refinement_lane_count"], 0)
        self.assertIs(dynamics["another_timestep_ladder_authorized"], False)
        roles = dynamics["input_role_separation"]
        self.assertEqual(
            roles["massive_source_initial_states"],
            "DE441_RESOLVED11_CUSTODY_COLLAPSED_TO_REDUCED5",
        )
        self.assertIs(roles["de440s_de441_equivalence_claimed"], False)

    def test_retained_h64_entry_gate_and_predecessor_failure_are_bound(self) -> None:
        records = {row["role"]: row for row in self.protocol["bound_inputs"]["records"]}
        self.assertEqual(records["h64_result"]["sha256"], PILOT.BOUND_FILES[3][3])
        self.assertEqual(
            self.protocol["bound_inputs"]["content_identities"]["h64_result"],
            "ea0627dd3c48c444cb4a0d9d5c11dedd41b3c3be886daa06208c9d4930896836",
        )
        self.assertEqual(
            self.protocol["bound_inputs"]["content_identities"]["stage1_v4_result"],
            "df4e089aad8d436e99871c943343ea8b631ca714c946edb388387307c75d87d2",
        )
        self.assertIs(
            self.protocol["claims"]["retained_h64_numerical_entry_gate_passed"],
            True,
        )
        self.assertIs(
            self.protocol["claims"]["resolved11_timestep_refinement_required_before_this_pilot"],
            False,
        )
        self.assertEqual(
            self.protocol["bound_inputs"]["large_h64_evidence_custody"],
            "LOCAL_CONTENT_ADDRESSED_EXTERNAL_TO_COMPACT_GITHUB_SOURCE_COMMIT",
        )

    def test_classifier_cadence_windows_and_precedence_are_exact(self) -> None:
        classifier = self.protocol["classifier_input"]
        self.assertEqual(
            classifier["input_frame"],
            "ICRF_J2000_EQUATORIAL_MODEL_BARYCENTRIC_INERTIAL",
        )
        self.assertEqual(
            classifier["element_frame"],
            "ECLIPJ2000_MODEL_BARYCENTRIC_INERTIAL",
        )
        self.assertEqual(classifier["cadence_julian_years"], 200)
        self.assertEqual(classifier["node_count_including_final_endpoint"], 50_001)
        self.assertEqual(
            (PILOT.CLASSIFIER_NODE_COUNT - 1) * PILOT.CLASSIFIER_CADENCE_JULIAN_YEARS,
            PILOT.HORIZON_JULIAN_YEARS,
        )
        self.assertEqual(
            PILOT.RESONANCE_WINDOW_COUNT * PILOT.RESONANCE_WINDOW_JULIAN_YEARS,
            PILOT.HORIZON_JULIAN_YEARS,
        )
        self.assertEqual(
            PILOT.POINTS_PER_RESONANCE_WINDOW * PILOT.CLASSIFIER_CADENCE_JULIAN_YEARS,
            PILOT.RESONANCE_WINDOW_JULIAN_YEARS,
        )
        self.assertEqual(
            self.protocol["classifier_precedence"],
            [
                "RESONANT_OR_RESONANT_CANDIDATE",
                "OORT_IF_A0_STRICTLY_GREATER_THAN_2000_AU",
                "CENTAUR_IF_A0_STRICTLY_LESS_THAN_NEPTUNE_A0",
                "SCATTERING_IF_MAX_ABS_A_MINUS_A0_OVER_A0_STRICTLY_GREATER_THAN_3_OVER_80",
                "DETACHED_IF_E0_STRICTLY_GREATER_THAN_6_OVER_25",
                "CLASSICAL_OTHERWISE",
            ],
        )
        self.assertEqual(
            self.protocol["classifier_input"]["missing_nonfinite_nonelliptic_or_removed_trajectory_policy"],
            "FAIL_CLOSED_NO_LABEL",
        )

        extraction = self.protocol["element_extraction_contract"]
        self.assertEqual(extraction["bound_source_role"], "stage1_v4_implementation")
        self.assertIs(extraction["reimplementation_permitted"], False)
        self.assertEqual(
            extraction["state_origin"],
            "INSTANTANEOUS_GM_WEIGHTED_BARYCENTER_OF_EXACT_REDUCED5_SOURCE_ROSTER",
        )
        self.assertEqual(
            extraction["j2000_equatorial_to_eclipj2000_rotation"][
                "cos_obliquity_binary64_hex"
            ],
            "0x1.d5c0357681ef3p-1",
        )
        self.assertIs(
            extraction["j2000_equatorial_to_eclipj2000_rotation"][
                "fused_multiply_add_permitted"
            ],
            False,
        )
        self.assertEqual(
            extraction["period_binary64_evaluation"],
            "(2.0*math.pi)*math.sqrt(((a*a)*a)/element_mu)",
        )

    def test_jx_b22_operationalization_fills_paper_gaps_explicitly(self) -> None:
        resonance = self.protocol["resonance_operationalization"]
        self.assertEqual(
            resonance["name"],
            "JX_B22_OPERATIONALIZATION_V1_NOT_EXACT_PAPER_REPRODUCTION",
        )
        self.assertEqual(resonance["p_domain_inclusive"], [1, 29])
        self.assertEqual(resonance["q_domain_inclusive"], [1, 29])
        self.assertEqual(resonance["m_domain_inclusive"], [-29, 29])
        self.assertEqual(resonance["n_domain_inclusive"], [-29, 29])
        self.assertIs(resonance["zero_coefficients_allowed"], True)
        self.assertEqual(resonance["angle_bin_count"], 18)
        self.assertEqual(resonance["angle_bin_edges_degrees"], list(range(0, 361, 20)))
        self.assertEqual(resonance["minimum_empty_angle_bins"], 2)
        self.assertEqual(resonance["window_count"], 50)
        self.assertEqual(resonance["minimum_contiguous_windows"], 46)
        self.assertIs(resonance["paper_underspecification_disclosed"], True)
        self.assertEqual(len(resonance["paper_missing_items_filled_by_jx"]), 6)

    def test_nonresonant_and_aggregation_boundaries_are_frozen(self) -> None:
        thresholds = self.protocol["nonresonant_thresholds"]
        self.assertEqual(thresholds["scattering_fraction"]["exact_rational"], "3/80")
        self.assertEqual(thresholds["detached_e0"]["exact_rational"], "6/25")
        self.assertEqual(thresholds["all_threshold_comparisons"], "STRICT_GREATER_THAN")

        aggregation = self.protocol["aggregation"]
        self.assertIs(
            aggregation["this_108_state_pilot"]["object_classification_authorized"],
            False,
        )
        full = aggregation["future_full_128_clone_jx_rule"]
        self.assertEqual(full["minimum_same_class_clones"], 123)
        self.assertEqual(full["minimum_same_class_fraction"], "123/128")
        self.assertIs(full["nominal_must_match"], True)
        self.assertIs(full["resonance_consensus_requires_same_exact_argument"], True)
        reference = aggregation["bernardinelli_2022_literal_30_clone_reference_only"]
        self.assertEqual(reference["scattering_at_least"], 15)
        self.assertEqual(reference["resonant_candidate_strictly_over_half_at_least"], 16)
        self.assertEqual(reference["secure_same_argument_strictly_over_80_percent_at_least"], 25)
        self.assertIs(reference["transferred_to_jx_128_clone_rule"], False)

    def test_fixtures_blockers_and_claim_ceiling_are_fail_closed(self) -> None:
        fixtures = self.protocol["required_classifier_fixtures"]
        self.assertEqual(fixtures["qualification_status"], "REQUIRED_NOT_YET_RETAINED_OR_PASSED")
        self.assertIn("KHAIN_S12_GOOD_5_2_TO_7_RESONANT", fixtures["source_backed"])
        self.assertIn("45_OF_50_CONTIGUOUS_NEGATIVE_AND_46_OF_50_POSITIVE", fixtures["mechanical_boundaries"])
        gate = self.protocol["execution_gate"]
        self.assertEqual(gate["blockers"], list(PILOT.BLOCKERS))
        self.assertIs(gate["all_blockers_closed"], False)
        self.assertIs(gate["execution_authorized"], False)
        self.assertIs(gate["no_automatic_run_after_prepare"], True)
        self.assertIs(self.protocol["outcomes_generated"], False)
        self.assertIs(self.protocol["execution_authorized"], False)
        for key, expected in PILOT.CLAIMS.items():
            self.assertIs(self.protocol["claims"][key], expected, key)

    def test_protocol_validation_rejects_rehashed_bool_numeric_and_policy_tamper(self) -> None:
        cases = (
            (("execution_authorized",), 0),
            (("cohort", "total_state_count"), 108.0),
            (("resonance_operationalization", "minimum_contiguous_windows"), 45),
            (("claims", "exact_bernardinelli_2022_classifier_reproduction"), True),
        )
        with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
            for path, replacement in cases:
                forged = copy.deepcopy(self.protocol)
                target = forged
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                forged["content_sha256"] = PILOT._content_sha256(forged)
                with self.subTest(path=path):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "exact frozen reconstruction",
                    ):
                        PILOT.validate_protocol(forged)

    def test_bound_file_drift_fails_before_protocol_construction(self) -> None:
        changed = list(PILOT.BOUND_FILES)
        role, path, size, _digest = changed[0]
        changed[0] = (role, path, size, "0" * 64)
        with mock.patch.object(PILOT, "BOUND_FILES", tuple(changed)):
            with self.assertRaisesRegex(PILOT.PilotProtocolError, "bound file differs"):
                PILOT._bound_records()

    def test_create_only_prepare_and_validation_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                prepared = PILOT.prepare(root)
                self.assertEqual(prepared, self.protocol)
                self.assertEqual([path.name for path in root.iterdir()], ["PROTOCOL.json"])
                path = root / "PROTOCOL.json"
                self.assertEqual(path.read_bytes(), PILOT._canonical_json(self.protocol))
                info = path.lstat()
                self.assertTrue(stat.S_ISREG(info.st_mode))
                self.assertEqual(info.st_nlink, 1)
                self.assertEqual(PILOT.validate_prepared(root), self.protocol)
                with self.assertRaisesRegex(PILOT.PilotProtocolError, "create-only"):
                    PILOT.prepare(root)

    def test_prepare_cleanup_never_deletes_a_replacement_inode(self) -> None:
        replacement = b"replacement owned by another process\n"

        def replace_then_fail(_root_fd: int):
            path = root / "PROTOCOL.json"
            path.unlink()
            path.write_bytes(replacement)
            raise RuntimeError("simulated validation failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(
                    PILOT,
                    "_validate_prepared_directory_fd",
                    side_effect=replace_then_fail,
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated validation failure"):
                        PILOT.prepare(root)
            self.assertEqual((root / "PROTOCOL.json").read_bytes(), replacement)

    def test_prepare_rejects_root_replacement_before_success_path_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            displaced = Path(directory) / "displaced"
            real_open = PILOT.os.open
            swapped = False

            def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal swapped
                if (
                    not swapped
                    and path == root.name
                    and dir_fd is not None
                    and flags & getattr(PILOT.os, "O_DIRECTORY", 0)
                ):
                    root.rename(displaced)
                    root.mkdir(mode=0o700)
                    swapped = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(PILOT.os, "open", side_effect=swapping_open):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "replaced before publication",
                    ):
                        PILOT.prepare(root)
            self.assertTrue(swapped)
            self.assertTrue(root.is_dir())
            self.assertEqual(list(root.iterdir()), [])
            self.assertTrue(displaced.is_dir())

    def test_prepare_rejects_lexical_parent_replacement_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parent = base / "publication"
            displaced_parent = base / "displaced-publication"
            parent.mkdir()
            root = parent / "prepared"
            real_validate = PILOT._validate_prepared_directory_fd

            def validate_then_swap(root_fd: int):
                value = real_validate(root_fd)
                parent.rename(displaced_parent)
                parent.mkdir()
                return value

            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(
                    PILOT,
                    "_validate_prepared_directory_fd",
                    side_effect=validate_then_swap,
                ):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "parent path was replaced",
                    ):
                        PILOT.prepare(root)
            self.assertFalse(root.exists())
            self.assertEqual(list(parent.iterdir()), [])
            self.assertEqual(list(displaced_parent.iterdir()), [])

    def test_validate_rejects_lexical_parent_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parent = base / "publication"
            displaced_parent = base / "displaced-publication"
            parent.mkdir()
            root = parent / "prepared"
            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                PILOT.prepare(root)
            real_validate = PILOT._validate_prepared_directory_fd

            def validate_then_swap(root_fd: int):
                value = real_validate(root_fd)
                parent.rename(displaced_parent)
                parent.mkdir()
                return value

            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(
                    PILOT,
                    "_validate_prepared_directory_fd",
                    side_effect=validate_then_swap,
                ):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "parent path was replaced",
                    ):
                        PILOT.validate_prepared(root)
            self.assertFalse(root.exists())
            self.assertTrue((displaced_parent / "prepared" / "PROTOCOL.json").is_file())

    def test_opposite_optimization_build_is_byte_identical(self) -> None:
        if sys.flags.optimize != 0:
            self.skipTest("normal-mode suite performs the direct -OO equivalence check")
        completed = subprocess.run(
            [sys.executable, "-OO", "-B", str(SOURCE), "build"],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(completed.stdout, PILOT._canonical_json(self.protocol))
        self.assertEqual(json.loads(completed.stdout), self.protocol)

    def test_default_root_is_absent_or_exact_after_publication(self) -> None:
        if not os.path.lexists(PILOT.DEFAULT_OUTPUT_ROOT):
            return
        retained = PILOT.validate_prepared(PILOT.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(retained, self.protocol)


if __name__ == "__main__":
    unittest.main()
