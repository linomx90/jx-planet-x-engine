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
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_classification_pilot_v2.py"


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


PILOT = _source_only_load("_test_jx_tno_10my_reduced5_classification_pilot_v2")


class Tno10MyrReduced5ClassificationPilotV2Tests(unittest.TestCase):
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

    def test_three_retention_grids_and_union_are_exact_without_interpolation(self) -> None:
        grids = self.protocol["retention_grids"]
        self.assertIs(grids["interpolation_permitted"], False)
        self.assertEqual(grids["union_grid"]["node_count"], 57_501)
        short_times = tuple(50 * index for index in range(10_001))
        b22_times = tuple(200 * index for index in range(50_001))
        long_times = tuple(1_000 * index for index in range(10_001))
        union_times = short_times + tuple(
            range(500_200, 10_000_001, 200)
        )
        self.assertEqual(len(union_times), 57_501)
        self.assertEqual(union_times[0], 0)
        self.assertEqual(union_times[-1], 10_000_000)
        self.assertEqual(short_times[::4], b22_times[:2_501])
        self.assertEqual(b22_times[::5], long_times)
        self.assertEqual(short_times[::20], long_times[:501])
        self.assertEqual(50 * 64, 3_200)
        self.assertEqual(200 * 64, 12_800)
        self.assertEqual(1_000 * 64, 64_000)
        self.assertEqual(
            grids["exact_cross_grid_mappings"],
            [
                "NODE_SHA256(B22_STATES[k])_EQUALS_B22_NODE_SHA256[k]_FOR_EVERY_k_0_THROUGH_50000",
                "EXTRACT_ELEMENTS(B22_STATES[k])_BITWISE_EQUALS_B22_ELEMENTS[k]_FOR_EVERY_k_0_THROUGH_50000",
                "ML_LONG_VIEW[l]_IS_B22_STATES[5*l]_FOR_l_0_THROUGH_10000",
                "ML_LONG_ELEMENT_VIEW[l]_IS_B22_ELEMENTS[5*l]_FOR_l_0_THROUGH_10000",
                "ML_SHORT[4*k]_BITWISE_EQUALS_B22_STATES[k]_FOR_k_0_THROUGH_2500",
                "NODE_SHA256(ML_SHORT[4*k])_EQUALS_B22_NODE_SHA256[k]_FOR_k_0_THROUGH_2500",
                "ML_SHORT[20*l]_BITWISE_EQUALS_B22_STATES[5*l]_FOR_l_0_THROUGH_500",
                "EXTRACT_ELEMENTS(ML_SHORT[4*k])_BITWISE_EQUALS_B22_ELEMENTS[k]_FOR_k_0_THROUGH_2500",
            ],
        )
        comparison_types = grids["cross_grid_comparison_types"]
        self.assertEqual(
            comparison_types["cartesian_to_node_digest"],
            "RECOMPUTE_EXACT_NODE_SHA256_THEN_DIGEST_EQUALITY",
        )
        self.assertIs(
            comparison_types["direct_cartesian_bytes_equal_orbital_element_bytes"],
            False,
        )
        self.assertEqual(
            comparison_types["offline_replay_coverage"],
            "EVERY_B22_NODE_IN_EVERY_BATCH",
        )
        self.assertEqual(
            grids["ml_long"]["cartesian_view"],
            "EXACT_B22_STATES_STRIDE_5_NO_DUPLICATE_CARTESIAN_FILE",
        )

    def test_prospective_binary_roster_sizes_and_result_last_are_exact(self) -> None:
        artifacts = self.protocol["prospective_binary_artifacts"]
        files = {row["name"]: row for row in artifacts["files"]}
        self.assertEqual(
            files["B22_STATES.f64le"]["byte_count"],
            12 * 50_001 * 14 * 6 * 8,
        )
        self.assertEqual(
            files["B22_ELEMENTS.f64le"]["byte_count"],
            12 * 10 * 6 * 50_001 * 8,
        )
        self.assertEqual(
            files["B22_NODE_SHA256.raw"]["byte_count"],
            12 * 50_001 * 32,
        )
        self.assertEqual(
            files["ML_SHORT_STATES.f64le"]["byte_count"],
            12 * 10_001 * 14 * 6 * 8,
        )
        self.assertNotIn("ML_LONG_STATES.f64le", files)
        self.assertEqual(
            artifacts["fixed_binary_byte_count"],
            sum(row["byte_count"] for row in files.values()),
        )
        self.assertEqual(artifacts["fixed_binary_byte_count"], 791_062_272)
        self.assertEqual(artifacts["publication_order"][-1], "RESULT.json_LAST")
        self.assertTrue(
            all(row["padding_or_trailing_bytes"] is False for row in files.values())
        )

    def test_classifier_v2_and_qualification_are_live_revalidated(self) -> None:
        successor = self.protocol["bound_inputs"]["classifier_successor_validation"]
        classifier = successor["classifier"]
        self.assertEqual(
            classifier["classifier"],
            "JX_B22_OPERATIONALIZATION_V2_36_TO_1_NOT_EXACT_PAPER_REPRODUCTION",
        )
        self.assertEqual(
            classifier["search_domain"]["principal_36_to_1_argument"],
            [36, 1, -35, 0, 0, 0],
        )
        self.assertIs(
            classifier["claims"]["search_domain_includes_36_to_1"], True
        )
        self.assertIs(
            classifier["claims"]["classifier_scientifically_qualified"], False
        )
        self.assertEqual(
            successor["qualification_protocol_content_sha256"],
            PILOT.EXPECTED_CONTENT["b22_qualification_protocol"],
        )
        records = {
            row["role"]: row for row in self.protocol["bound_inputs"]["records"]
        }
        self.assertEqual(
            records["b22_classifier_v2_implementation"]["sha256"],
            "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98",
        )
        self.assertEqual(
            records["predecessor_pilot_protocol"]["sha256"],
            "9a00d2d30cf59267ff9d2474d6e3d42dde8bb4e5713d1a0c3c073d56ff8f9c2c",
        )

    def test_jx_b22_operationalization_fills_paper_gaps_explicitly(self) -> None:
        resonance = self.protocol["resonance_operationalization"]
        self.assertEqual(
            resonance["name"],
            "JX_B22_OPERATIONALIZATION_V2_36_TO_1_NOT_EXACT_PAPER_REPRODUCTION",
        )
        self.assertEqual(
            resonance["period_pair_union"],
            [
                {
                    "band": "INHERITED_V1",
                    "p_domain_inclusive": [1, 29],
                    "q_domain_inclusive": [1, 29],
                    "coefficient_domain_inclusive": [-29, 29],
                },
                {
                    "band": "NEW_P30_TO_P36",
                    "p_domain_inclusive": [30, 36],
                    "q_domain_inclusive": [1, 29],
                    "coefficient_domain_inclusive": [-35, 35],
                },
            ],
        )
        self.assertEqual(resonance["principal_36_to_1_argument"], [36, 1, -35, 0, 0, 0])
        self.assertEqual(resonance["search_universe_counts"]["combined_period_pairs"], 653)
        self.assertEqual(resonance["search_universe_counts"]["combined_arguments"], 1_712_871)
        self.assertIs(resonance["zero_coefficients_allowed"], True)
        self.assertEqual(resonance["angle_bin_count"], 18)
        self.assertEqual(resonance["angle_bin_edges_degrees"], list(range(0, 361, 20)))
        self.assertEqual(resonance["minimum_empty_angle_bins"], 2)
        self.assertEqual(resonance["window_count"], 50)
        self.assertEqual(resonance["minimum_contiguous_windows"], 46)
        self.assertIs(resonance["paper_underspecification_disclosed"], True)
        self.assertEqual(len(resonance["paper_missing_items_filled_by_jx"]), 7)

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
        self.assertEqual(
            fixtures["qualification_status"],
            "CATALOG_LABEL_CUSTODY_PASSED_NUMERICAL_TRAJECTORY_FIXTURES_NOT_YET_RETAINED_OR_PASSED",
        )
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

    def test_implementation_identity_is_bound_to_module_bootstrap_observation(self) -> None:
        identity = PILOT._implementation_identity()
        self.assertEqual(
            identity["sha256"],
            PILOT._BOOTSTRAP_IMPLEMENTATION_SHA256,
        )
        self.assertEqual(
            identity["byte_length"],
            PILOT._BOOTSTRAP_IMPLEMENTATION_BYTE_LENGTH,
        )
        with mock.patch.object(PILOT, "_BOOTSTRAP_IMPLEMENTATION_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                PILOT.PilotProtocolError,
                "module-bootstrap observation",
            ):
                PILOT._implementation_identity()

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

    def test_validate_rejects_protocol_path_substitution_even_if_root_mtime_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                PILOT.prepare(root)
            protocol_path = root / "PROTOCOL.json"
            root_before = root.stat()
            real_read = PILOT.os.read
            swapped = False

            def swapping_read(descriptor: int, count: int) -> bytes:
                nonlocal swapped
                if not swapped:
                    protocol_path.unlink()
                    protocol_path.write_bytes(b"{}\n")
                    os.utime(
                        root,
                        ns=(root_before.st_atime_ns, root_before.st_mtime_ns),
                    )
                    swapped = True
                return real_read(descriptor, count)

            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(PILOT.os, "read", side_effect=swapping_read):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "prepared file changed while being read",
                    ):
                        PILOT.validate_prepared(root)
            self.assertTrue(swapped)
            self.assertEqual(protocol_path.read_bytes(), b"{}\n")

    def test_validate_rejects_root_ctime_change_when_roster_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                PILOT.prepare(root)
            root_before = root.stat()
            real_read = PILOT._read_regular_bytes_at
            changed = False

            def read_then_mutate(directory_fd: int, name: str) -> bytes:
                nonlocal changed
                payload = real_read(directory_fd, name)
                marker = root / "transient"
                marker.write_bytes(b"x")
                marker.unlink()
                os.utime(
                    root,
                    ns=(root_before.st_atime_ns, root_before.st_mtime_ns),
                )
                changed = True
                return payload

            with mock.patch.object(PILOT, "build_protocol", return_value=self.protocol):
                with mock.patch.object(
                    PILOT,
                    "_read_regular_bytes_at",
                    side_effect=read_then_mutate,
                ):
                    with self.assertRaisesRegex(
                        PILOT.PilotProtocolError,
                        "prepared root changed during validation",
                    ):
                        PILOT.validate_prepared(root)
            self.assertTrue(changed)
            self.assertEqual([path.name for path in root.iterdir()], ["PROTOCOL.json"])

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
