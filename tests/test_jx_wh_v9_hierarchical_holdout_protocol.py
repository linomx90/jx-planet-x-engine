from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "jx_wh_v9_hierarchical_holdout_v1"
PROTOCOL = RUN / "protocol_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("content_integrity")
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    domain = payload["content_integrity"]["domain"].encode("ascii")
    return hashlib.sha256(domain + b"\0" + serialized).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_literal(value: str) -> None:
    raise ValueError(f"nonfinite JSON literal: {value}")


class WisdomHolmanV9HierarchicalHoldoutProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            PROTOCOL.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_literal,
        )

    def test_protocol_tree_contains_only_finite_numbers(self) -> None:
        pending: list[object] = [self.protocol]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, float):
                self.assertTrue(math.isfinite(value))

    def test_protocol_is_design_only_and_cannot_self_approve(self) -> None:
        protocol = self.protocol
        self.assertEqual(
            protocol["status"],
            "REGISTERED_DESIGN_ONLY_AWAITING_SOURCE_BOUND_RUNNER_AND_INDEPENDENT_EXECUTION",
        )
        blindness = protocol["outcome_blindness"]
        self.assertEqual(blindness["holdout_candidate_executions_before_registration"], 0)
        self.assertEqual(blindness["holdout_oracle_executions_before_registration"], 0)
        self.assertFalse(blindness["holdout_scientific_outcomes_generated"])
        self.assertFalse(blindness["holdout_scientific_outcomes_inspected"])
        self.assertFalse(blindness["post_outcome_case_replacement_allowed"])
        self.assertFalse(blindness["post_outcome_gate_or_threshold_change_allowed"])
        controls = protocol["claim_controls"]
        for name in (
            "input_manifest_registered",
            "source_bound_runner_registered",
            "holdout_execution_completed",
            "outcome_blind_holdout_completed",
            "all_32_cases_passed",
            "independent_host_reproduction_completed",
            "external_review_completed",
            "scientific_approval_granted",
            "broad_n_body_qualified",
            "default_route_authorized",
            "production_ephemeris_authorized",
        ):
            self.assertIs(controls[name], False, name)

    def test_future_public_seed_cannot_be_known_at_registration(self) -> None:
        beacon = self.protocol["future_public_randomness"]
        self.assertEqual(beacon["chain_index"], 2)
        self.assertEqual(beacon["pulse_timestamp_utc"], "2026-10-23T04:00:00.000Z")
        self.assertEqual(beacon["pulse_timestamp_unix_milliseconds"], 1792728000000)
        self.assertEqual(beacon["api_path"], "/pulse/time/1792728000000")
        self.assertFalse(beacon["pulse_available_at_registration"])
        self.assertFalse(beacon["pulse_accessed_at_registration"])
        self.assertIsNone(beacon["pulse_raw_response_sha256"])
        self.assertIsNone(beacon["pulse_output_value"])
        self.assertFalse(beacon["alternate_beacon_or_timestamp_allowed"])
        self.assertFalse(beacon["manual_seed_allowed"])
        derivation = self.protocol["public_case_derivation"]
        self.assertIsNone(derivation["seed_sha256"])
        self.assertEqual(
            derivation["seed_sha256_text_encoding"],
            "EXACT_64_CHARACTER_LOWERCASE_ASCII_HEXADECIMAL",
        )
        self.assertEqual(
            derivation["literal_field_name_table"],
            {
                "case_level_with_body_index_dash": [
                    "interior_cell_rank",
                    "boundary_base_cell",
                ],
                "body_level_with_canonical_integer_index": [
                    "eccentricity_u",
                    "inclination_u",
                    "longitude_of_ascending_node_u",
                    "argument_of_periapsis_u",
                    "mean_anomaly_u",
                ],
            },
        )
        self.assertFalse(derivation["unknown_or_additional_draw_field_names_allowed"])
        self.assertFalse(derivation["boundary_override_consumes_additional_draws"])
        self.assertIn("no leading zero", derivation["body_index_token_rule"])
        self.assertTrue(derivation["seed_must_be_registered_additively_before_input_construction"])
        self.assertFalse(derivation["rejection_sampling"])
        self.assertFalse(derivation["duplicate_case_replacement"])

    def test_exact_32_case_stratified_roster(self) -> None:
        roster = self.protocol["case_roster"]
        self.assertEqual(len(roster), 32)
        ids = [case["case_id"] for case in roster]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            Counter(case["body_count"] for case in roster),
            Counter({3: 6, 4: 6, 8: 6, 16: 6, 32: 8}),
        )
        self.assertEqual(
            Counter(case["class"] for case in roster),
            Counter({"INTERIOR": 20, "BOUNDARY_STRESS": 12}),
        )
        self.assertEqual(
            Counter(case["stress"] for case in roster if case["stress"] is not None),
            Counter(
                {
                    "MASS_RATIO_0_99": 3,
                    "PERIAPSE_STEP_DYNAMIC_SAFE": 2,
                    "ECCENTRICITY_DYNAMIC_SAFE_0_80": 2,
                    "HILL_SWEPT_CLEARANCE_JUST_INSIDE": 2,
                    "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION": 2,
                    "MAXIMUM_BODY_COUNT": 1,
                }
            ),
        )
        for case in roster:
            self.assertEqual(case["horizon_inner_periods"], 10)

    def test_interior_and_boundary_constants_are_frozen(self) -> None:
        materialization = self.protocol["public_case_derivation"][
            "input_numeric_materialization"
        ]
        self.assertEqual(materialization["construction_precision_decimal_digits"], 160)
        self.assertEqual(materialization["verification_precision_decimal_digits"], 220)
        self.assertIn("2^-400", materialization["kepler_equation"])
        self.assertIn("not a formal correctly-rounded", materialization["dual_precision_stability"])
        spec = self.protocol["fixed_candidate_spec_inputs"]
        self.assertEqual(
            spec["minimum_encounter_pair_separation_float_hex"],
            "0x1.0000000000000p-40",
        )
        self.assertEqual(
            spec["minimum_jacobi_periapse_float_hex"],
            "0x1.0000000000000p-12",
        )
        self.assertEqual(
            spec["maximum_initial_barycenter_position_norm_float_hex"],
            "0x1.0000000000000p+0",
        )
        self.assertEqual(
            spec["maximum_initial_barycenter_velocity_norm_float_hex"],
            "0x1.0000000000000p+0",
        )
        interior = self.protocol["interior_case_family"]
        self.assertEqual(interior["body_counts"], [3, 4, 8, 16, 32])
        self.assertEqual(interior["cases_per_body_count"], 4)
        self.assertEqual(
            interior["total_secondary_to_primary_gm_ratio_by_cell"],
            [2**-10, 2**-9, 2**-8, 2**-7],
        )
        self.assertLessEqual(
            max(interior["total_secondary_to_primary_gm_ratio_by_cell"]),
            0.008,
        )
        ratio_cells = interior[
            "adjacent_jacobi_semimajor_axis_ratio_by_body_count_and_cell"
        ]
        self.assertEqual(ratio_cells["3"], [4.0, 6.0, 10.0, 14.0])
        self.assertEqual(ratio_cells["32"], [1.6, 1.65, 1.7, 1.75])
        self.assertEqual(
            interior["eccentricity_max_by_body_count_and_cell"]["32"],
            [0.01, 0.01, 0.01, 0.01],
        )
        self.assertIn("equal GM", interior["mass_weight_rule"])
        stresses = self.protocol["boundary_stress_construction"]
        self.assertEqual(stresses["MASS_RATIO_0_99"]["total_secondary_to_primary_gm_ratio"], 0.0099)
        self.assertEqual(stresses["ECCENTRICITY_DYNAMIC_SAFE_0_80"]["eccentricity"], 0.8)
        self.assertEqual(
            stresses["PERIAPSE_STEP_DYNAMIC_SAFE"]["target_periapse_step_fraction"],
            0.03125,
        )
        self.assertEqual(stresses["HILL_SWEPT_CLEARANCE_JUST_INSIDE"]["initial_node_utilization_target"], 0.8)
        self.assertFalse(stresses["HILL_SWEPT_CLEARANCE_JUST_INSIDE"]["swept_path_endpoint_may_be_pre_tuned"])
        interaction = stresses["INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION"]
        self.assertEqual(interaction["interaction_ratio_target"], 0.015)
        self.assertEqual(interaction["target_tolerance_absolute"], 0.0005)
        self.assertEqual(interaction["reconstructed_target_band"], [0.0145, 0.0155])
        self.assertEqual(interaction["minimum_initial_adjacent_mutual_hill_separation"], 3.5)
        self.assertEqual(
            interaction["closed_interval_float_hex"],
            ["0x1.00068db8bac71p+0", "0x1.0000000000000p+1"],
        )
        self.assertEqual(interaction["secondary_02_gm_float_hex"], "0x1.26e978d4fdf3bp-7")
        self.assertIn("interaction_ratio(low)-0.015>0", interaction["endpoint_residual_assertion"])
        self.assertEqual(interaction["total_secondary_to_primary_gm_ratio_override"], 0.0099)
        self.assertFalse(stresses["target_search"]["candidate_or_oracle_import_permitted"])
        self.assertFalse(stresses["target_search"]["case_replacement"])
        self.assertIn("before the beacon pulse", stresses["target_search"]["freeze_scope"])
        self.assertNotIn("runner registration", json.dumps(stresses["target_search"]).lower())
        self.assertIn("boundary_base_cell", stresses["base_cell_assignment"])

    def test_razor_boundary_controls_are_separate_mandatory_twins(self) -> None:
        controls = self.protocol["guard_semantics_controls"]
        self.assertFalse(controls["included_in_32_case_oracle_accuracy_denominator"])
        self.assertTrue(controls["mandatory_for_protocol_pass"])
        self.assertTrue(controls["no_control_is_a_scientific_accuracy_case"])
        roster = controls["controls"]
        self.assertEqual(len(roster), 10)
        self.assertEqual(Counter(row["pair_id"] for row in roster), Counter({
            "ECCENTRICITY_GUARD": 2,
            "PERIAPSE_STEP_GUARD": 2,
            "TOTAL_SECONDARY_GM_GUARD": 2,
            "INITIAL_NODE_HILL_GUARD": 2,
            "MAXIMUM_BODY_COUNT_GUARD": 2,
        }))
        self.assertEqual(Counter(row["expected"] for row in roster), Counter({"ACCEPT": 5, "REJECT": 5}))
        by_id = {row["control_id"]: row for row in roster}
        self.assertEqual(by_id["JXHHO-C-ECC-IN"]["target_float_hex"], "0x1.cc49ba5e353f8p-1")
        self.assertEqual(by_id["JXHHO-C-ECC-OUT"]["reconstructed_acceptance_band_min"], 0.900001)
        self.assertEqual(by_id["JXHHO-C-ECC-OUT"]["reconstructed_acceptance_band_max"], 0.900002)
        self.assertEqual(by_id["JXHHO-C-STEP-OUT"]["reconstructed_guard_utilization_band_min"], 1.0001)
        self.assertEqual(by_id["JXHHO-C-STEP-OUT"]["reconstructed_guard_utilization_band_max"], 1.0002)
        self.assertEqual(by_id["JXHHO-C-MASS-OUT"]["target_float_hex"], "0x1.47ae147ae147cp-7")
        self.assertEqual(by_id["JXHHO-C-HILL-OUT"]["reconstructed_initial_node_utilization_band_min"], 1.0001)
        self.assertEqual(by_id["JXHHO-C-HILL-OUT"]["reconstructed_initial_node_utilization_band_max"], 1.0002)
        self.assertEqual(by_id["JXHHO-C-N33-OUT"]["body_count"], 33)
        self.assertEqual(
            by_id["JXHHO-C-MASS-OUT"]["expected_top_level_exception_message"],
            "V9 public input validation failed",
        )
        self.assertEqual(
            by_id["JXHHO-C-MASS-OUT"]["expected_cause_exception_family"],
            "TrajectoryDomainError",
        )
        self.assertEqual(
            by_id["JXHHO-C-N33-OUT"]["expected_cause_exception_message"],
            "complete native loop supports at most 32 bodies",
        )
        for prefix in ("ECC", "STEP"):
            self.assertEqual(by_id[f"JXHHO-C-{prefix}-IN"]["body_count"], 2)
            self.assertTrue(
                by_id[f"JXHHO-C-{prefix}-IN"]
                ["interaction_force_is_exactly_zero_for_single_secondary"]
            )
        self.assertEqual(
            by_id["JXHHO-C-HILL-IN"]["fixed_step_rule"],
            "h=P_inner/1048576",
        )
        self.assertEqual(
            Counter(row["execution_mode"] for row in roster),
            Counter({"ONE_NATIVE_MAP_STEP": 6, "PREPARE_WORKSPACE_ONLY": 4}),
        )
        self.assertIn("not constructively guaranteed", controls["interaction_boundary_control_exclusion"])
        self.assertEqual(self.protocol["input_manifest_freeze"]["required_guard_control_count"], 10)
        required_case_fields = self.protocol["input_manifest_freeze"]["required_fields_per_case"]
        self.assertIn("cumulative_gm_float_hex", required_case_fields)
        self.assertIn("cumulative_gm_binding_sha256", required_case_fields)

    def test_all_cases_are_conjunctive_and_oracle_resolvable(self) -> None:
        gates = self.protocol["gates"]
        oracle = gates["oracle_resolution"]
        self.assertTrue(oracle["all_32_cases_must_be_oracle_resolvable"])
        self.assertEqual(oracle["oracle_position_floor_absolute_max"], 1e-8)
        self.assertEqual(oracle["oracle_velocity_floor_absolute_max"], 1e-8)
        self.assertEqual(oracle["oracle_floor_fraction_of_fine_4D0_error_max"], 0.05)
        self.assertEqual(oracle["oracle_failure_status"], "INCONCLUSIVE_ORACLE_RESOLUTION")
        state = gates["state_accuracy_and_convergence"]
        self.assertTrue(state["all_32_cases_conjunctive"])
        self.assertEqual(state["fine_4D0_state_error_max"], 2e-4)
        for metric in ("position", "velocity"):
            for ratio in ("D0_over_2D0", "2D0_over_4D0"):
                self.assertEqual(state[f"{ratio}_{metric}_error_ratio_min"], 3.2)
                self.assertEqual(state[f"{ratio}_{metric}_error_ratio_max"], 4.8)
        decision = self.protocol["decision_rule"]
        self.assertTrue(decision["all_32_cases_and_all_gates_conjunctive"])
        self.assertTrue(decision["all_ten_guard_semantics_controls_conjunctive"])
        self.assertTrue(decision["no_case_attrition"])
        self.assertTrue(decision["no_case_replacement"])
        self.assertTrue(decision["no_threshold_adjustment"])

    def test_oracle_and_invariant_gates_are_exact(self) -> None:
        oracle = self.protocol["independent_oracles"]
        self.assertEqual(oracle["dop853"]["required_scipy_version"], "1.17.0")
        self.assertEqual(oracle["ias15"]["required_rebound_version"], "5.1.1")
        self.assertEqual(oracle["dop853"]["baseline_relative_tolerance"], 1e-13)
        self.assertEqual(oracle["dop853"]["tight_relative_tolerance"], 3e-14)
        self.assertIn("solver is restarted", oracle["dop853"]["checkpoint_execution"])
        self.assertIn("returned_end_time_float_hex", oracle["dop853"]["segment_records_required"])
        self.assertEqual(oracle["ias15"]["baseline_epsilon"], 1e-13)
        self.assertEqual(oracle["ias15"]["tight_epsilon"], 1e-14)
        self.assertEqual(oracle["ias15"]["adaptive_mode"], "PRS23")
        self.assertEqual(oracle["ias15"]["initial_dt"], "P_inner/1024")
        self.assertEqual(oracle["ias15"]["min_dt"], 0.0)
        self.assertEqual(oracle["ias15"]["G"], 1.0)
        self.assertEqual(oracle["ias15"]["gravity"], "basic")
        self.assertEqual(oracle["ias15"]["collision"], "none")
        self.assertEqual(oracle["ias15"]["boundary"], "none")
        self.assertEqual(oracle["ias15"]["softening"], 0.0)
        self.assertEqual(oracle["ias15"]["N_active"], "N")
        self.assertEqual(oracle["ias15"]["testparticle_type"], 0)
        self.assertEqual(oracle["ias15"]["omp_num_threads"], 1)
        self.assertTrue(oracle["ias15"]["exact_finish_time"])
        metrics = self.protocol["metric_definitions"]
        self.assertIn("DOP853_tight-vs-IAS15_tight", metrics["oracle_position_floor"])
        self.assertIn("both tight oracles", metrics["position_error"])
        self.assertIn("norm(L(0))", metrics["relative_angular_momentum_error"])
        lattice = self.protocol["execution_lattice"]
        self.assertEqual(lattice["case_step_divisors"], ["D0", "2*D0", "4*D0"])
        self.assertEqual(lattice["base_step_divisor_min"], 64)
        self.assertTrue(lattice["base_step_divisor_must_be_power_of_two"])
        self.assertIn("32*max_i", lattice["base_step_divisor_rule"])
        self.assertEqual(set(lattice["horizon_inner_periods_by_body_count"].values()), {10})
        invariants = self.protocol["gates"]["invariants"]
        self.assertEqual(invariants["D0_relative_energy_error_max"], 1e-5)
        self.assertEqual(invariants["2D0_relative_energy_error_max"], 2.5e-6)
        self.assertEqual(invariants["4D0_relative_energy_error_max"], 6.25e-7)
        self.assertEqual(invariants["relative_angular_momentum_error_max_each_divisor"], 2e-12)
        self.assertEqual(invariants["center_of_mass_position_drift_max_each_divisor"], 2e-12)
        self.assertEqual(invariants["total_momentum_drift_max_each_divisor"], 2e-12)
        self.assertEqual(
            self.protocol["gates"]["candidate_integrity"]["fallback_count_required_each_case_each_divisor"],
            0,
        )
        serialized = json.dumps(self.protocol, sort_keys=True).lower()
        for stale in ("p64", "p128", "p256"):
            self.assertNotIn(stale, serialized)
        self.assertNotIn("100_digit", serialized)

    def test_frozen_prior_evidence_and_candidate_files_match(self) -> None:
        for record in self.protocol["frozen_evidence_bindings"].values():
            if not isinstance(record, dict) or "path" not in record:
                continue
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(_sha256(path), record["sha256"])
        for name in ("wrapper", "native_wrapper", "native_loop"):
            record = self.protocol["candidate"][name]
            path = ROOT / record["path"]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(_sha256(path), record["sha256"])

    def test_protocol_content_integrity(self) -> None:
        integrity = self.protocol["content_integrity"]
        self.assertEqual(integrity["algorithm"], "SHA-256")
        self.assertEqual(_content_sha256(self.protocol), integrity["content_sha256"])


if __name__ == "__main__":
    unittest.main()
