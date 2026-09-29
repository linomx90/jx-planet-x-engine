from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "contract_v1.json"
REGISTRATION_PATH = ROOT / "registration_v1.json"
RUNNER_PATH = ROOT / "run_numerics.py"
VERIFIER_PATH = ROOT / "verify_replay.py"


def load_module(name: str, path: Path):
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


runner = load_module("jx_e2_runner_test", RUNNER_PATH)
verifier = load_module("jx_e2_verifier_test", VERIFIER_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_classification_bundle(contract: dict) -> dict:
    frames = [item["id"] for item in contract["frames"]]
    regimes = [item["id"] for item in contract["numerical_regimes"]]
    arms = []
    for frame in frames:
        for regime in regimes:
            if regime == "MERCURIUS_0125":
                linear = 2e-12
            elif regime == "MERCURIUS_00625":
                linear = 4e-12
            elif regime == "MERCURIUS_003125":
                linear = 8e-12
            else:
                linear = 1e-12
            origin = (
                5e-8
                if frame == "F0_E1_UNSHIFTED" and regime == "MERCURIUS_003125"
                else 1e-8
            )
            maximum = {
                "relative_engine_energy_drift": 0.0,
                "scale_normalized_intrinsic_energy_residual": 0.0,
                "scale_normalized_linear_momentum_residual": linear,
                "scale_normalized_origin_angular_momentum_residual": origin,
                "scale_normalized_com_angular_momentum_residual": (
                    8e-12 if regime == "MERCURIUS_003125" else 1e-12
                ),
                "relative_initial_linear_momentum_residual": 0.0,
                "relative_initial_origin_angular_momentum_residual": 0.0,
                "com_ballistic_position_residual_AU": 0.0,
                "angular_decomposition_residual_over_scale": 0.0,
            }
            semantic = {
                "arm_key": f"{frame}/{regime}",
                "invariant_metrics": {"maximum": maximum},
            }
            arms.append(
                {
                    "semantic": semantic,
                    "semantic_sha256": runner.sha256_bytes(
                        runner.canonical_bytes(semantic)
                    ),
                }
            )
    comparisons = []
    for definition in runner.pair_definitions(contract):
        if definition["kind"] == "IAS15_REFERENCE":
            discrepancy = 4e-12 if definition["left"].endswith("IAS15_1E10") else 1e-12
        elif definition["kind"] == "MERCURIUS_TO_IAS15":
            discrepancy = {
                "MERCURIUS_0125": 8e-8,
                "MERCURIUS_00625": 4e-8,
                "MERCURIUS_003125": 2e-8,
            }[definition["left"].split("/")[1]]
        else:
            discrepancy = 1e-12
        comparisons.append(
            {
                **definition,
                "maximum": {
                    "maximum_dimensionless_state_discrepancy": discrepancy,
                    "maximum_position_separation_AU": discrepancy,
                    "maximum_velocity_separation_AU_per_year": discrepancy,
                },
                "endpoint": {
                    "time_year": 50000.0,
                    "maximum_dimensionless_state_discrepancy": discrepancy,
                    "maximum_position_separation_AU": discrepancy,
                    "maximum_velocity_separation_AU_per_year": discrepancy,
                },
            }
        )
    return {
        "configuration_id": "M0",
        "arms": arms,
        "comparisons": comparisons,
        "e1_context_comparisons": [{"exact": True} for _ in range(20)],
    }


class JXE2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = runner.strict_json(CONTRACT_PATH)
        cls.registration = runner.strict_json(REGISTRATION_PATH)

    def test_identity_and_claim_boundary_are_exact(self) -> None:
        self.assertEqual(self.contract["schema"], "jx-e2-numerics-contract/v1")
        self.assertEqual(
            self.contract["experiment_id"], "jx-e2-active-frame-integrator-50k-v1"
        )
        self.assertEqual(
            self.contract["artifact_class"], "LOCAL_NUMERICAL_METHOD_FORENSICS_ONLY"
        )
        self.assertEqual(
            self.contract["claim_ceiling"], "NUMERICAL_METHOD_FORENSICS_ONLY"
        )
        self.assertFalse(self.contract["outcomes_generated_at_registration"])
        self.assertIn("provides no evidence for or against Planet X", self.contract["mandatory_nonclaim"])

    def test_permissions_are_fail_closed(self) -> None:
        self.assertEqual(
            self.contract["permissions"],
            {
                "local_cpu_numerical_diagnostic_authorized": True,
                "gpu_execution_authorized": False,
                "network_access_authorized": False,
                "observed_data_access_authorized": False,
                "jx_e1_reclassification_authorized": False,
                "jx_e1_execution_b_authorized": False,
                "jx_o2_execution_authorized": False,
                "scientific_planet_x_claim_authorized": False,
            },
        )

    def test_e1_bindings_match_immutable_bytes(self) -> None:
        boundary = self.contract["e1_immutable_boundary"]
        pairs = {
            "contract_path": "contract_sha256",
            "runner_path": "runner_sha256",
            "verifier_path": "verifier_sha256",
            "result_path": "result_sha256",
            "post_failure_audit_path": "post_failure_audit_sha256",
            "audit_script_path": "audit_script_sha256",
        }
        for path_key, hash_key in pairs.items():
            path = (ROOT / boundary[path_key]).resolve()
            self.assertTrue(path.is_file())
            self.assertEqual(sha256(path), boundary[hash_key])
        result = runner.strict_json((ROOT / boundary["result_path"]).resolve())
        self.assertEqual(result["verdict"], "ENGINEERING_LONG_INVALID")
        self.assertEqual(result["semantic_sha256"], boundary["result_semantic_sha256"])
        audit = runner.strict_json((ROOT / boundary["post_failure_audit_path"]).resolve())
        self.assertFalse(audit["execution_b_authorized_or_started"])
        self.assertFalse(audit["thresholds_changed"])

    def test_matrix_is_exact_complete_and_unique(self) -> None:
        self.assertEqual(
            [item["id"] for item in self.contract["configuration_set"]],
            ["M0", "CI01-A", "CI03-C", "CI05-C", "CI06-C", "CI07-B", "CI09-A", "CI09-D"],
        )
        self.assertEqual(
            [item["id"] for item in self.contract["frames"]],
            ["F0_E1_UNSHIFTED", "FCM_ACTIVE_BARYCENTRIC"],
        )
        self.assertEqual(
            [item["id"] for item in self.contract["numerical_regimes"]],
            [
                "MERCURIUS_0125",
                "MERCURIUS_00625",
                "MERCURIUS_003125",
                "IAS15_1E10",
                "IAS15_1E12",
                "IAS15_1E14",
            ],
        )
        keys = {
            (configuration["id"], frame["id"], regime["id"])
            for configuration in self.contract["configuration_set"]
            for frame in self.contract["frames"]
            for regime in self.contract["numerical_regimes"]
        }
        self.assertEqual(len(keys), 96)
        self.assertEqual(self.contract["dynamics"]["expected_arm_count"], 96)
        self.assertEqual(
            self.contract["dynamics"]["expected_pairwise_comparison_count"], 160
        )
        self.assertEqual(
            self.contract["dynamics"]["expected_e1_context_record_count"], 160
        )

    def test_pair_matrix_is_exact(self) -> None:
        pairs = runner.pair_definitions(self.contract)
        self.assertEqual(len(pairs), 20)
        self.assertEqual(
            {item["kind"] for item in pairs},
            {"FRAME", "MERCURIUS_REFINEMENT", "IAS15_REFERENCE", "MERCURIUS_TO_IAS15"},
        )
        self.assertEqual(len({(item["kind"], item["left"], item["right"]) for item in pairs}), 20)

    def test_classification_implementations_have_exact_parity(self) -> None:
        bundle = synthetic_classification_bundle(self.contract)
        first = runner.configuration_classification(self.contract, bundle)
        second = verifier.recompute_configuration_classification(self.contract, bundle)
        self.assertEqual(first, second)
        self.assertEqual(
            first["classification"],
            "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT",
        )
        runner_overall_input = []
        verifier_overall_input = []
        for configuration in self.contract["configuration_set"]:
            left = copy.deepcopy(first)
            right = copy.deepcopy(second)
            left["configuration_id"] = configuration["id"]
            right["configuration_id"] = configuration["id"]
            runner_overall_input.append(left)
            verifier_overall_input.append(right)
        self.assertEqual(
            runner.overall_classification(runner_overall_input),
            verifier.recompute_overall(verifier_overall_input),
        )

    def test_classification_fails_closed_without_established_reference(self) -> None:
        bundle = synthetic_classification_bundle(self.contract)
        for item in bundle["comparisons"]:
            if item["kind"] == "IAS15_REFERENCE" and item["left"].endswith(
                "IAS15_1E12"
            ):
                item["maximum"]["maximum_dimensionless_state_discrepancy"] = 1e-8
        result = runner.configuration_classification(self.contract, bundle)
        self.assertEqual(result["classification"], "MIXED_OR_INCONCLUSIVE")
        self.assertFalse(result["frame_state_equivalent"])
        self.assertFalse(
            result["forensic_signature_indicators"][
                "frame_intrinsic_metric_sensitivity"
            ]
        )
        self.assertFalse(
            result["mechanism_details"][
                "origin_angular_frame_effect_consistent"
            ]
        )

    def test_classification_forbids_cross_metric_substitution(self) -> None:
        bundle = synthetic_classification_bundle(self.contract)
        for wrapper in bundle["arms"]:
            arm = wrapper["semantic"]
            if arm["arm_key"].endswith("MERCURIUS_003125"):
                arm["invariant_metrics"]["maximum"][
                    "scale_normalized_com_angular_momentum_residual"
                ] = 1e-12
            wrapper["semantic_sha256"] = runner.sha256_bytes(
                runner.canonical_bytes(arm)
            )
        result = runner.configuration_classification(self.contract, bundle)
        self.assertFalse(
            all(
                item["both_metrics"]
                for item in result["mechanism_details"][
                    "ias15_quarter_residual_by_frame"
                ].values()
            )
        )
        self.assertNotEqual(
            result["classification"],
            "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT",
        )

    def test_classification_rejects_floor_and_unbounded_step_ratios(self) -> None:
        for values in ((1e-15, 2e-15, 4e-15), (1e-12, 4e-12, 16e-12)):
            bundle = synthetic_classification_bundle(self.contract)
            by_regime = dict(
                zip(
                    ("MERCURIUS_0125", "MERCURIUS_00625", "MERCURIUS_003125"),
                    values,
                    strict=True,
                )
            )
            for wrapper in bundle["arms"]:
                arm = wrapper["semantic"]
                regime = arm["arm_key"].split("/")[1]
                if regime in by_regime:
                    arm["invariant_metrics"]["maximum"][
                        "scale_normalized_linear_momentum_residual"
                    ] = by_regime[regime]
                wrapper["semantic_sha256"] = runner.sha256_bytes(
                    runner.canonical_bytes(arm)
                )
            result = runner.configuration_classification(self.contract, bundle)
            self.assertFalse(
                result["forensic_signature_indicators"]["step_count_scaling"]
            )

    def test_overall_classification_rejects_empty_or_partial_input(self) -> None:
        with self.assertRaises(ValueError):
            runner.overall_classification([])
        with self.assertRaises(ValueError):
            verifier.recompute_overall([])

    def test_inherited_thresholds_cannot_control_validity(self) -> None:
        reference = self.contract["inherited_reference_flags"]
        self.assertFalse(reference["affect_e2_validity"])
        self.assertEqual(reference["maximum_relative_energy_drift"], 1e-6)
        self.assertEqual(reference["maximum_relative_origin_angular_momentum_drift"], 1e-10)
        self.assertFalse(reference["linear_momentum_threshold_transferable_to_Pstar_metric"])
        self.assertFalse(reference["angular_momentum_threshold_transferable_to_Lstar_com_metric"])
        self.assertIn("mathematically corresponding", reference["source"])
        self.assertIn("not an exact reproduction", reference["source"])
        self.assertFalse(self.contract["result_policy"]["threshold_edit_or_adaptive_extension_allowed"])
        self.assertFalse(self.contract["e1_immutable_boundary"]["e2_can_rehabilitate_or_replace_e1"])

    def test_legacy_angular_flag_is_explicitly_nonclaiming(self) -> None:
        required = {
            "legacy_E1_reference_applicable",
            "relative_energy_within_legacy_E1_reference",
            "compensated_origin_angular_within_legacy_numeric_value_illustration",
            "origin_angular_is_not_exact_E1_evaluator",
            "Pstar_and_Lstar_metrics_are_not_legacy_E1_thresholds",
        }
        old_claiming_name = "origin_angular_within_legacy_E1_reference"
        for path in (RUNNER_PATH, VERIFIER_PATH):
            source = path.read_text(encoding="utf-8")
            for field in required:
                self.assertIn(f'"{field}"', source)
            self.assertNotIn(f'"{old_claiming_name}"', source)

    def test_preflight_disclosure_is_explicit(self) -> None:
        disclosure = self.contract["pre_registration_disclosure"]
        self.assertTrue(disclosure["runtime_only_preflight_executed"])
        self.assertFalse(disclosure["state_values_or_invariant_metrics_saved_or_inspected"])
        self.assertTrue(disclosure["decoded_state_equality_flags_inspected"])
        self.assertTrue(disclosure["checkpoint_capability_preflight"]["ias15_immediate_and_continued_exact"])
        self.assertFalse(disclosure["preflight_may_affect_numerical_classification"])

    def test_registration_locks_exact_file_set_and_hashes(self) -> None:
        self.assertEqual(
            self.registration["schema"], "jx-e2-numerics-local-registration/v1"
        )
        expected_paths = {
            "README.md",
            "contract_v1.json",
            "run_numerics.py",
            "verify_replay.py",
            "test_jx_e2.py",
        }
        self.assertEqual(set(self.registration["locked_files"]), expected_paths)
        for relative, expected in self.registration["locked_files"].items():
            self.assertRegex(expected, r"^[0-9a-f]{64}$")
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_runner_and_verifier_have_no_placeholders(self) -> None:
        self.assertRegex(runner.EXPECTED_CONTRACT_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(runner.EXPECTED_CONTRACT_SHA256, sha256(CONTRACT_PATH))
        self.assertEqual(verifier.EXPECTED_CONTRACT_SHA256, sha256(CONTRACT_PATH))
        self.assertRegex(verifier.EXPECTED_RUNNER_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(verifier.EXPECTED_RUNNER_SHA256, sha256(RUNNER_PATH))

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        cases = (
            '{"a": 1, "a": 2}',
            '{"a": NaN}',
            '{"a": Infinity}',
            '{"a": 1e400}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for payload in cases:
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.strict_json(path)
                with self.assertRaises(ValueError):
                    verifier.strict_json(path)

    def test_runtime_and_sources_validate_without_dynamics(self) -> None:
        runner.validate_contract(self.contract, CONTRACT_PATH)
        runner.validate_registration(REGISTRATION_PATH, CONTRACT_PATH, RUNNER_PATH)
        runtime = runner.validate_runtime()
        self.assertEqual(runtime["python_version"], "3.12.13")
        module, e1_contract, _ = runner.load_e1(self.contract, CONTRACT_PATH)
        self.assertTrue(callable(module.build_simulation))
        self.assertEqual(e1_contract["experiment_id"], "jx-e1-p9-9x4-50k-v1")
        self.assertFalse(hasattr(module, "__cached__"))
        self.assertEqual(
            Path(module.build_simulation.__code__.co_filename).resolve(),
            (ROOT / self.contract["e1_immutable_boundary"]["runner_path"]).resolve(),
        )
        self.assertEqual(
            runtime["rebound_python_source_sha256"],
            self.contract["runtime_lock"]["rebound_python_source_sha256"],
        )

    def test_active_builders_preserve_identity_across_regimes(self) -> None:
        module, e1_contract, _ = runner.load_e1(self.contract, CONTRACT_PATH)
        configuration = self.contract["configuration_set"][6]
        physical = []
        for regime in self.contract["numerical_regimes"]:
            simulation, names, _ = runner.build_active_simulation(
                module,
                e1_contract,
                configuration,
                "F0_E1_UNSHIFTED",
                regime,
            )
            physical.append(runner.physical_state_digest(simulation))
            self.assertEqual(simulation.N, simulation.N_active)
            self.assertEqual(len(names), 6)
        self.assertEqual(len(set(physical)), 1)

    def test_frame_translation_preserves_sun_relative_state(self) -> None:
        module, e1_contract, _ = runner.load_e1(self.contract, CONTRACT_PATH)
        configuration = self.contract["configuration_set"][7]
        regime = self.contract["numerical_regimes"][0]
        native, names, _ = runner.build_active_simulation(
            module, e1_contract, configuration, "F0_E1_UNSHIFTED", regime
        )
        barycentric, other_names, _ = runner.build_active_simulation(
            module, e1_contract, configuration, "FCM_ACTIVE_BARYCENTRIC", regime
        )
        self.assertEqual(names, other_names)
        identity = runner.frame_identity_error(native, barycentric, names, 64.0)
        self.assertTrue(identity["within_locked_bound"])
        self.assertEqual(
            runner.canonical_bytes(runner.invariant_snapshot(native)),
            verifier.canonical_bytes(verifier.independent_invariant_snapshot(native)),
        )
        self.assertEqual(
            runner.state_discrepancy(
                native,
                barycentric,
                names,
                runner.build_active_simulation(
                    module,
                    e1_contract,
                    configuration,
                    "F0_E1_UNSHIFTED",
                    regime,
                )[2],
                self.contract["analytic_benchmark"]["G_AU3_Msun_yr2"],
                self.contract["analytic_benchmark"]["sun_mass_Msun"],
            ),
            verifier.independent_state_discrepancy(
                native,
                barycentric,
                names,
                runner.build_active_simulation(
                    module,
                    e1_contract,
                    configuration,
                    "F0_E1_UNSHIFTED",
                    regime,
                )[2],
                self.contract["analytic_benchmark"]["G_AU3_Msun_yr2"],
                self.contract["analytic_benchmark"]["sun_mass_Msun"],
            ),
        )

    def test_no_forbidden_scientific_verdicts_or_execution_authority(self) -> None:
        serialized = json.dumps(self.contract, sort_keys=True).upper()
        for token in ("PLANET_X_DETECTED", "PLANET_X_EXCLUDED", "PLANET_X_CONFIRMED"):
            self.assertNotIn(token, serialized)
        allowed = set(self.contract["classification_policy"]["allowed_configuration_classifications"])
        self.assertEqual(
            allowed,
            {
                "FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT",
                "MERCURIUS_REFINEMENT_CONSISTENT",
                "MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED",
                "MIXED_OR_INCONCLUSIVE",
            },
        )

    def test_local_registration_has_no_external_timestamp_claim(self) -> None:
        self.assertEqual(
            self.registration["timestamp_authority"],
            "LOCAL_CONTENT_HASH_REGISTRATION_ONLY_NO_EXTERNAL_TIMESTAMP",
        )
        self.assertFalse(self.registration["externally_timestamped"])
        self.assertFalse(self.registration["scientific_evidence_artifact"])


if __name__ == "__main__":
    unittest.main()
