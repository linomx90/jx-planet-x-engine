from __future__ import annotations

import ast
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks import jx_long_horizon_observed_qualification as qualification


ROOT = Path(__file__).resolve().parents[1]


class LongHorizonObservedQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = qualification.build_protocol(ROOT)

    def test_exact_108_state_identity_is_content_bound(self) -> None:
        inputs = self.protocol["observed_inputs"]
        self.assertEqual(inputs["state_count"], 108)
        self.assertEqual(inputs["nominal_state_count"], 12)
        self.assertEqual(inputs["sensitivity_state_count"], 96)
        self.assertEqual(
            inputs["science_inputs_content_sha256"],
            qualification.EXPECTED_SCIENCE_INPUTS_SHA256,
        )
        self.assertEqual(
            inputs["content_sha256"],
            qualification.EXPECTED_OBSERVED_INPUT_SHA256,
        )
        states = inputs["states"]
        self.assertEqual([row["state_index"] for row in states], list(range(108)))
        self.assertEqual(len({row["state_id"] for row in states}), 108)
        self.assertEqual(
            sum(row["kind"] == "OBSERVED_NOMINAL" for row in states), 12
        )
        self.assertEqual(
            sum(
                row["kind"] == "BOUNDED_INDEPENDENT_SENSITIVITY"
                for row in states
            ),
            96,
        )
        self.assertEqual(
            [len(block["state_indices"]) for block in inputs["population_blocks"]],
            [54, 54],
        )
        for block in inputs["population_blocks"]:
            raw = block["raw_source_artifact"]
            payload = (ROOT / raw["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), raw["sha256"])
            self.assertEqual(len(payload), raw["bytes"])

    def test_source_model_and_all_three_lanes_are_frozen(self) -> None:
        source = self.protocol["source_model"]
        self.assertEqual(source["source_ids_ordered"], list(qualification.SOURCE_IDS))
        self.assertEqual(source["primary_source_id"], "Sun")
        self.assertEqual(
            source["primary_physical_model"], "COLLAPSED_INNER_SYSTEM_MONOPOLE"
        )
        self.assertEqual(source["primary_collapsed_naif_body_ids"], [10, 1, 2, 3, 4])
        self.assertEqual(
            float.fromhex(source["primary_gm_AU3_per_julian_year2_float64_hex"]),
            qualification.SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2,
        )
        self.assertEqual(source["artifact"]["sha256"], qualification.SOURCE_STATE_SHA256)
        self.assertEqual(
            source["packaged_artifact"]["sha256"],
            qualification.SOURCE_STATE_SHA256,
        )
        self.assertEqual(
            (ROOT / source["packaged_artifact"]["path"]).read_bytes(),
            (ROOT / source["artifact"]["path"]).read_bytes(),
        )
        self.assertEqual(
            source["construction_manifest"]["sha256"],
            qualification.SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            source["packaged_semantic_manifest"]["sha256"],
            qualification.PACKAGED_SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / qualification.PACKAGED_SOURCE_MANIFEST_RELATIVE_PATH)
                .read_bytes()
            ).hexdigest(),
            qualification.PACKAGED_SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            source["epoch_jd_tdb_float64_hex"],
            qualification.SOURCE_EPOCH_JD_TDB.hex(),
        )
        self.assertEqual(
            source["massive_force_semantics"],
            "FULL_MUTUAL_NEWTONIAN_ALL_ACTIVE_SOURCES",
        )
        self.assertEqual(
            source["tracer_backreaction_semantics"],
            "EXACTLY_MASSLESS_NO_BACKREACTION",
        )

        lanes = {lane["role"]: lane for lane in self.protocol["lanes"]}
        self.assertEqual(set(lanes), {
            "JX_CANDIDATE",
            "JX_REFINEMENT_REFERENCE",
            "INDEPENDENT_NUMERICAL_COMPARATOR",
        })
        candidate = lanes["JX_CANDIDATE"]
        refinement = lanes["JX_REFINEMENT_REFERENCE"]
        external = lanes["INDEPENDENT_NUMERICAL_COMPARATOR"]
        self.assertEqual(
            float.fromhex(candidate["step_years_float64_hex"]), 0.0625
        )
        self.assertEqual(candidate["fixed_step_count"], 16_000)
        self.assertEqual(
            float.fromhex(refinement["step_years_float64_hex"]), 0.03125
        )
        self.assertEqual(refinement["fixed_step_count"], 32_000)
        self.assertEqual(external["required_exact_version"], "5.1.1")
        self.assertEqual(external["integrator"], "ias15")
        self.assertEqual(
            float.fromhex(external["epsilon_float64_hex"]), 1.0e-15
        )
        self.assertTrue(external["exact_finish_time_at_every_annual_target"])
        self.assertFalse(external["reference_truth"])
        for lane in lanes.values():
            self.assertEqual(lane["massive_source_state_sha256"], source["artifact"]["sha256"])
            self.assertEqual(lane["state_count"], 108)
            self.assertEqual(lane["output_target_count"], 1_001)

    def test_annual_budgets_restart_gates_and_resources_are_bounded(self) -> None:
        budgets = self.protocol["output_and_event_budgets"]
        annual = budgets["annual_output"]
        self.assertEqual(annual["cadence_years"], 1)
        self.assertEqual(annual["exact_epochs_per_lane"], 1_001)
        self.assertEqual(annual["maximum_state_rows_per_lane"], 108_108)
        self.assertEqual(annual["missing_or_duplicate_epoch_budget"], 0)
        self.assertEqual(annual["nonfinite_value_budget"], 0)
        events = budgets["events"]
        self.assertEqual(events["maximum_terminal_removal_records_per_lane"], 108)
        self.assertEqual(events["maximum_unserialized_or_dropped_events"], 0)
        self.assertTrue(events["event_chain_digest_required"])

        restart = self.protocol["split_restart_qualification"]
        self.assertEqual(restart["split_epoch_years"], 500)
        self.assertEqual(restart["jx_candidate_split_global_step"], 8_000)
        self.assertEqual(restart["jx_refinement_split_global_step"], 16_000)
        self.assertEqual(len(restart["required_executions_per_jx_lane"]), 3)
        for key, value in restart["gates"].items():
            if key == "checkpoint_contract_status_must_equal":
                self.assertEqual(value, "VERIFIED")
            elif key == "cross_runtime_bitwise_claimed":
                self.assertFalse(value)
            else:
                self.assertTrue(value)

        resources = self.protocol["resource_limits"]
        self.assertEqual(resources["physical_cpu_workers"], 6)
        self.assertEqual(resources["maximum_gpu_devices"], 1)
        self.assertEqual(resources["concurrent_qualification_lanes"], 1)
        self.assertEqual(resources["blas_threads_per_process"], 1)
        self.assertTrue(resources["fail_closed_on_limit"])

    def test_precision_gates_are_predeclared_and_cannot_mask_failures(self) -> None:
        gates = self.protocol["measurements_and_gates"]
        expected = {
            "position_vector_l2_AU",
            "velocity_vector_l2_AU_per_year",
            "absolute_perihelion_delta_AU",
            "absolute_inclination_delta_rad",
            "absolute_minimum_source_distance_delta_AU",
            "absolute_matched_terminal_event_time_delta_years",
        }
        self.assertEqual(set(gates["metrics"]), expected)
        for comparison in (
            "jx_candidate_vs_jx_refinement_maxima_float64_hex",
            "jx_refinement_vs_rebound_ias15_maxima_float64_hex",
        ):
            limits = gates[comparison]
            self.assertEqual(set(limits), expected)
            self.assertTrue(all(float.fromhex(value) > 0.0 for value in limits.values()))
        self.assertEqual(
            gates["event_correspondence"]["unmatched_terminal_event_budget"], 0
        )
        self.assertIn("ALL_LIMITS", gates["acceptance_semantics"])

    def test_preparation_binds_surfaces_by_ast_without_importing_them(self) -> None:
        conversion = self.protocol["initial_condition_conversion"]
        builder = conversion["builder_binding"]
        runner = self.protocol["execution_surface"]["runner_binding"]
        self.assertEqual(builder["module"], qualification.INITIAL_CONDITIONS_MODULE)
        self.assertEqual(runner["module"], qualification.RUNNER_MODULE)
        for binding in (builder, runner):
            self.assertEqual(binding["inspection"], "AST_ONLY_NO_IMPORT")
            self.assertEqual(
                binding["surface_ready"],
                binding["declared_required_symbols"] == binding["required_symbols"],
            )
            if binding["present"]:
                payload = (ROOT / binding["path"]).read_bytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), binding["source_sha256"]
                )
        self.assertEqual(
            conversion["required_converter_implementation_sha256"],
            qualification.EXPECTED_GENESIS_CONVERTER_SHA256,
        )
        self.assertEqual(
            conversion["required_genesis_model_sha256"],
            qualification.EXPECTED_SOURCE_ALIGNED_GENESIS_MODEL_SHA256,
        )
        self.assertEqual(
            conversion["required_primary_gm_AU3_per_julian_year2_float64_hex"],
            qualification.SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2.hex(),
        )
        self.assertEqual(
            conversion["required_cartesian_inventory_sha256"],
            qualification.EXPECTED_CARTESIAN_INVENTORY_SHA256,
        )
        self.assertEqual(
            conversion["required_cartesian_bundle_sha256"],
            qualification.EXPECTED_CARTESIAN_BUNDLE_SHA256,
        )
        self.assertFalse(
            conversion["cartesian_inventory_materialized_during_preparation"]
        )
        self.assertIsNone(conversion["conversion_receipt_sha256"])
        execution = self.protocol["execution_surface"]
        self.assertTrue(execution["runner_module_loaded_during_preparation"])
        self.assertTrue(execution["builder_module_loaded_during_preparation"])
        self.assertTrue(execution["numpy_module_loaded_during_preparation"])
        self.assertFalse(execution["runner_entrypoint_called_during_preparation"])
        self.assertFalse(execution["builder_entrypoint_called_during_preparation"])
        self.assertTrue(execution["deterministic_input_materialization_performed"])
        self.assertFalse(execution["dynamics_integration_performed"])

    def test_fresh_process_import_receipt_matches_actual_module_state(self) -> None:
        code = """
import json
import sys
from benchmarks import jx_long_horizon_observed_qualification as qualification
record = qualification.build_protocol('.')
surface = record['execution_surface']
print(json.dumps({
    'runner_actual': qualification.RUNNER_MODULE in sys.modules,
    'builder_actual': qualification.INITIAL_CONDITIONS_MODULE in sys.modules,
    'numpy_actual': 'numpy' in sys.modules,
    'runner_recorded': surface['runner_module_loaded_during_preparation'],
    'builder_recorded': surface['builder_module_loaded_during_preparation'],
    'numpy_recorded': surface['numpy_module_loaded_during_preparation'],
    'input_materialization_performed': surface['deterministic_input_materialization_performed'],
    'dynamics_integration_performed': surface['dynamics_integration_performed'],
}, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)
        self.assertIs(receipt["runner_actual"], receipt["runner_recorded"])
        self.assertIs(receipt["builder_actual"], receipt["builder_recorded"])
        self.assertIs(receipt["numpy_actual"], receipt["numpy_recorded"])
        self.assertTrue(receipt["input_materialization_performed"])
        self.assertFalse(receipt["dynamics_integration_performed"])

    def test_expected_cartesian_identity_matches_source_aligned_builder(self) -> None:
        from jxplanetx.long_horizon.initial_conditions import (
            CONVERTER_IMPLEMENTATION_SHA256,
            DEFAULT_GENESIS_MODEL,
            build_predeclared_initial_condition_inventory,
        )

        inventory = build_predeclared_initial_condition_inventory()
        self.assertEqual(
            CONVERTER_IMPLEMENTATION_SHA256,
            qualification.EXPECTED_GENESIS_CONVERTER_SHA256,
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.content_sha256,
            qualification.EXPECTED_SOURCE_ALIGNED_GENESIS_MODEL_SHA256,
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.primary_gm_AU3_per_year2.hex(),
            qualification.SOURCE_PRIMARY_GM_AU3_PER_JULIAN_YEAR2.hex(),
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.primary_gm_source_artifact_sha256,
            qualification.SOURCE_STATE_SHA256,
        )
        self.assertEqual(
            DEFAULT_GENESIS_MODEL.source_model_manifest_sha256,
            qualification.PACKAGED_SOURCE_MANIFEST_SHA256,
        )
        self.assertIn(
            "NO_EPHEMERIS_ORIGIN_TRANSLATION",
            DEFAULT_GENESIS_MODEL.element_origin_to_output_origin_policy,
        )
        self.assertEqual(inventory.state_count, 108)
        self.assertEqual(
            inventory.content_sha256,
            qualification.EXPECTED_CARTESIAN_INVENTORY_SHA256,
        )
        self.assertEqual(
            {
                bundle.population_id: bundle.content_sha256
                for bundle in inventory.bundles
            },
            qualification.EXPECTED_CARTESIAN_BUNDLE_SHA256,
        )

    def test_harness_source_has_no_direct_numerical_or_gpu_import(self) -> None:
        path = ROOT / "benchmarks/jx_long_horizon_observed_qualification.py"
        tree = ast.parse(path.read_bytes(), filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue({"numpy", "cupy", "rebound"}.isdisjoint(imported_roots))

    def test_protocol_is_deterministic_content_addressed_and_fail_closed(self) -> None:
        second = qualification.build_protocol(ROOT)
        self.assertEqual(self.protocol, second)
        self.assertRegex(self.protocol["protocol_sha256"], r"^[0-9a-f]{64}$")
        qualification.validate_protocol(deepcopy(self.protocol), ROOT)

        bound_paths = {
            item["path"] for item in self.protocol["implementation_sources"]
        }
        required_transitive_paths = {
            "src/jxplanetx/_all_conic_kepler.py",
            "src/jxplanetx/cuda_tracer_events.py",
            "src/jxplanetx/fast_wisdom_holman.py",
            "src/jxplanetx/_wisdom_holman_cpu_v3_module.c",
            "src/jxplanetx/_wisdom_holman_loop.c",
            "src/jxplanetx/_wisdom_holman_loop_v2.c",
            "src/jxplanetx/_wisdom_holman_loop_v3.c",
        }
        required_transitive_paths.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src/jxplanetx/engine").glob("*.py")
        )
        self.assertTrue(required_transitive_paths <= bound_paths)

        mutators = (
            lambda value: value["lanes"][0].update(
                {"step_years_float64_hex": (0.125).hex()}
            ),
            lambda value: value["source_model"]["artifact"].update(
                {"sha256": "0" * 64}
            ),
            lambda value: value["observed_inputs"].update({"state_count": 107}),
            lambda value: value["resource_limits"].update(
                {"physical_cpu_workers": 12}
            ),
            lambda value: value["nonclaims"].update({"science_result": True}),
            lambda value: value.update({"execution_authorized": True}),
        )
        for mutate in mutators:
            changed = deepcopy(self.protocol)
            mutate(changed)
            with self.assertRaises(qualification.QualificationProtocolError):
                qualification.validate_protocol(changed, ROOT)

    def test_prepare_only_cli_is_create_only_and_cannot_execute(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qualification.main([])
            with self.assertRaises(SystemExit):
                qualification.main(["--execute"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "PROTOCOL.json"
            self.assertEqual(
                qualification.main(
                    [
                        "--prepare-only",
                        "--project-root",
                        str(ROOT),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            document = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(document, self.protocol)
            self.assertEqual(document["status"], "PREPARED_NOT_EXECUTED")
            self.assertTrue(document["prepared_only"])
            self.assertFalse(document["execution_authorized"])
            self.assertFalse(document["long_run_started"])
            self.assertFalse(any(document["nonclaims"].values()))
            with self.assertRaises(qualification.QualificationProtocolError):
                qualification.main(
                    [
                        "--prepare-only",
                        "--project-root",
                        str(ROOT),
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
