from __future__ import annotations

import ast
from decimal import Decimal, getcontext
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_common_epoch_full129_bridge_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("jx_tno_common_epoch_full129_bridge_v1_test_subject", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bridge preregistration module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CommonEpochFull129BridgeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.protocol = cls.m.build_protocol()

    def test_prepare_only_claim_ceiling_is_fail_closed(self) -> None:
        self.assertEqual(
            self.protocol["status"],
            "PREPARED_BLOCKED_COMMON_EPOCH_FULL129_BRIDGE_PREREGISTRATION_NO_EXECUTION",
        )
        authority = self.protocol["authority"]
        self.assertIs(authority["execution_in_this_artifact"], False)
        self.assertEqual(
            authority["explicitly_absent_public_cli"],
            ["execute", "execute-worker", "resume", "validate-result"],
        )
        claims = self.protocol["claims"]
        self.assertIs(claims["prepare_only_protocol_frozen"], True)
        self.assertIs(claims["all12_full129_identity_frozen"], True)
        self.assertIs(claims["source_result_ancestry_bound"], True)
        for name in (
            "de441_kernel_bound",
            "common_epoch_nominal_oracle_bound",
            "source_node_gate_frozen",
            "execution_authorized",
            "dynamics_executed",
            "gpu_executed",
            "common_epoch_bridge_executed",
            "common_epoch_bridge_qualified",
            "autonomous_resolved11_authoritative",
            "short_horizon_tracer_qualification_authorized",
            "three_gyr_execution_authorized",
            "classification_generated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            with self.subTest(name=name):
                self.assertIs(claims[name], False)

    def test_target_and_all12_full129_identity_are_exact(self) -> None:
        target = self.protocol["target"]
        self.assertEqual(target["jd_tdb_decimal"], "2461200.5")
        self.assertEqual(target["jd_tdb_binary64_hex"], (2_461_200.5).hex())
        geometry = self.protocol["geometry"]
        self.assertEqual(geometry["batch_count"], 12)
        self.assertEqual(geometry["tracers_per_batch"], 129)
        self.assertEqual(geometry["tracer_rows"], 1548)
        self.assertEqual(geometry["common_epoch_state_bytes_per_lane"], 74_304)
        self.assertEqual(geometry["object_order"], list(self.m.OBJECT_IDS))
        roster = self.protocol["inputs"]["native_roster"]
        self.assertEqual(len(roster), 12)
        for order, row in enumerate(roster):
            self.assertEqual(row["batch_order"], order)
            self.assertEqual(row["object_id"], self.m.OBJECT_IDS[order])
            self.assertEqual(row["spk_id"], self.m.SPK_IDS[order])
            self.assertEqual(row["sbdb_solution_id"], self.m.SBDB_SOLUTION_IDS[order])
            self.assertEqual(row["tracer_orders"], list(range(129)))
            self.assertGreater(row["forward_delta_days"], 0)

    def test_authoritative_model_is_time_dependent_de441_not_autonomous(self) -> None:
        model = self.protocol["physical_model"]
        self.assertEqual(
            model["authoritative_bridge"],
            "DE441_TIME_DEPENDENT_RESTRICTED_RESOLVED11",
        )
        self.assertEqual(model["source_naif_ids"], list(self.m.RESOLVED11_NAIF_IDS))
        self.assertIs(model["identical_external_source_for_every_numerical_lane"], True)
        self.assertIs(model["cpu_only"], True)
        self.assertIs(model["gpu_allowed"], False)
        self.assertIn("DIAGNOSTIC_ONLY", model["autonomous_resolved11"])
        gm = model["source_gm_contract"]
        self.assertEqual(gm["naif_ids_in_exact_force_sum_order"], list(self.m.RESOLVED11_NAIF_IDS))
        self.assertEqual(gm["gm_km3_per_s2_exact_decimal_tokens"], list(self.m.SOURCE_GM_KM3_PER_S2_DECIMAL))
        self.assertIn("NOT_THE_LATER_BODY_ASSIGNMENT_VALUES", gm["authority_semantics"])
        getcontext().prec = 80
        factor = Decimal(31_557_600) ** 2 / Decimal("149597870.7") ** 3
        expected_gm_hex = [float(Decimal(token) * factor).hex() for token in self.m.SOURCE_GM_KM3_PER_S2_DECIMAL]
        self.assertEqual(gm["gm_AU3_per_julian_year2_binary64_hex"], expected_gm_hex)
        initial = self.protocol["initial_state_contract"]
        self.assertIn("NATIVE_EPOCH_CARTESIAN_STATES", initial["authoritative_tracer_rows"])
        self.assertIn("ECLIPTIC_J2000_TO_ICRF", initial["input_transform"])
        self.assertIn("ARE_NOT_THE_AUTHORITATIVE_INPUT_FIELDS", initial["native_full129_bridge_role"])
        rotation = initial["frame_rotation_contract"]
        self.assertEqual(rotation["obliquity_arcseconds_exact_decimal"], "84381.448")
        self.assertEqual(rotation["sine_binary64_hex"], "0x1.9752e50f4b399p-2")
        self.assertEqual(rotation["cosine_binary64_hex"], "0x1.d5c0357681ef3p-1")
        self.assertIs(rotation["applies_identically_to_position_and_velocity"], True)
        blockers = set(self.protocol["execution_blockers"])
        self.assertIn(
            "EXACT_RETAINED_HASHED_DE441_KERNEL_WITH_FULL_INTERVAL_COVERAGE_NOT_BOUND",
            blockers,
        )
        self.assertIn(
            "COMMON_EPOCH_CARTESIAN_NOMINAL_DIAGNOSTIC_ORACLE_NOT_BOUND",
            blockers,
        )
        kernel = model["de441_kernel_acquisition_requirement"]
        self.assertEqual(kernel["byte_length"], 1_656_830_976)
        self.assertEqual(kernel["official_md5"], "ad8dfa4e505ef0e3a5d587a5b4705632")
        self.assertTrue(kernel["official_url"].endswith("/de441_part-2.bsp"))

    def test_three_required_cpu_numerical_lanes_are_frozen(self) -> None:
        lanes = self.protocol["numerical_lanes"]
        production = lanes["production"]
        self.assertEqual(production["required_version"], "5.1.1")
        self.assertEqual(production["integrator"], "IAS15")
        self.assertEqual(production["epsilon_hex"], (1e-13).hex())
        self.assertEqual(production["adaptive_mode"], "PRS23")
        self.assertEqual(production["initial_dt_julian_years_hex"], (1 / 32).hex())
        independent = lanes["independent"]
        self.assertEqual(independent["method"], "DOP853")
        self.assertEqual(independent["rtol_hex"], (1e-13).hex())
        self.assertEqual(independent["atol_hex"], (1e-15).hex())
        self.assertEqual(independent["max_step_hex"], (1 / 128).hex())
        replicate = lanes["required_convergence_replicate"]
        self.assertIs(replicate["optional"], False)
        self.assertEqual(replicate["epsilon_hex"], (1e-14).hex())
        self.assertIs(replicate["all_12_times_129_required"], True)

    def test_method_and_roundtrip_gates_are_exact_and_oracle_remains_unbound(self) -> None:
        acceptance = self.protocol["acceptance"]
        self.assertEqual(
            acceptance["position_AU_max"],
            {"numerator": 1, "denominator": 1_000_000_000, "nearest_binary64_hex": (1e-9).hex()},
        )
        self.assertEqual(
            acceptance["velocity_AU_per_julian_year_max"],
            {"numerator": 1, "denominator": 10_000_000_000, "nearest_binary64_hex": (1e-10).hex()},
        )
        self.assertEqual(len(acceptance["method_endpoint_pairs"]), 3)
        self.assertEqual(len(acceptance["same_method_roundtrip_lanes"]), 3)
        self.assertEqual(len(acceptance["cross_method_roundtrip_paths"]), 4)
        self.assertIs(acceptance["source_node_gate"]["required"], True)
        self.assertEqual(acceptance["source_node_gate"]["status"], "UNBOUND_EXECUTION_BLOCKER")
        oracle = acceptance["common_epoch_nominal_cartesian_oracle"]
        self.assertIs(oracle["required"], True)
        self.assertIsNone(oracle["exact_artifact"])
        self.assertIn("DIAGNOSTIC_NOT_NUMERICAL", oracle["comparison_role"])
        self.assertEqual(oracle["tolerance_policy"].split(";", 1)[0], "NO_PASS_TOLERANCE")
        self.assertIs(oracle["horizons_force_model_equivalence_claimed"], False)
        self.assertIn("DESIGNATION_QUERY_ONLY", oracle["acquisition"])
        self.assertIn("USER_SUPPLIED_ELEMENT_PROPAGATION_IS_OUT_OF_SCOPE", oracle["acquisition"])
        self.assertEqual(
            oracle["frozen_designation_solution_ids_in_object_order"],
            list(self.m.SBDB_SOLUTION_IDS),
        )
        self.assertIn("TP_BINARY64_ULP_DISTANCE_LE_1", oracle["lineage_gates"])
        self.assertEqual(len(oracle["hard_acceptance_gates"]), 5)
        self.assertIn("DIFFERENT_FORCE_MODELS", oracle["parent_oracle_tolerance_requirement_resolution"])
        self.assertEqual(oracle["binding_without_every_hard_acceptance_gate_passing"], "FORBIDDEN")
        self.assertEqual(
            self.protocol["future_result_claim_ceiling"],
            "QUALIFIED_COMMON_EPOCH_FULL129_INPUT_BRIDGE_FOR_THIS_FROZEN_DE441_RESTRICTED_RESOLVED11_FORCE_MODEL_ONLY",
        )
        self.assertTrue(all(value is False for value in self.protocol["nonclaims"].values()))

    def test_every_declared_input_binding_is_exactly_audited(self) -> None:
        audit = self.m.audit_prerequisites()
        groups = audit["input_bindings"]
        self.assertEqual(sum(len(rows) for rows in groups.values()), len(self.m.INPUT_BINDINGS))
        self.assertEqual(len(groups["authoritative_native_cartesian"]), 2)
        self.assertEqual(len(groups["authoritative_de441_resolved11"]), 2)
        self.assertEqual(len(groups["source_gm_authority"]), 1)
        self.assertEqual(len(groups["native_full129"]), 2)
        self.assertEqual(len(groups["native_resolved11"]), 3)
        self.assertEqual(len(groups["sbdb"]), 14)
        self.assertEqual(len(groups["source_result"]), 9)
        self.assertEqual(len(groups["runtime"]), 1)
        self.assertIn("authoritative_native_cartesian_states", audit["input_content_bindings"])
        self.assertIn("authoritative_de441_resolved11_custody", audit["input_content_bindings"])
        git = audit["git_source_result"]
        self.assertEqual(git["tag_object"], self.m.SOURCE_RESULT_TAG_OBJECT)
        self.assertEqual(git["commit"], self.m.SOURCE_RESULT_COMMIT)
        self.assertEqual(git["tree"], self.m.SOURCE_RESULT_TREE)

    def test_protocol_is_canonical_and_domain_hashed(self) -> None:
        payload = self.m._canonical_json(self.protocol)
        self.assertEqual(payload, self.m._canonical_json(self.m._strict_json(payload)))
        self.assertEqual(self.protocol["content_sha256"], self.m._content_sha256(self.protocol))
        self.assertEqual(self.m.build_protocol(), self.protocol)

    def test_source_has_no_scientific_runtime_import_or_execution_surface(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
        self.assertTrue({"numpy", "scipy", "rebound", "cupy"}.isdisjoint(imported))
        self.assertTrue({"execute", "resume", "execute_worker", "validate_result"}.isdisjoint(functions))

    def test_create_only_prepare_and_read_only_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            created = self.m.prepare(root)
            self.assertEqual(created, self.protocol)
            self.assertEqual(set(os.listdir(root)), {"PROTOCOL.json"})
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "PROTOCOL.json").stat().st_mode), 0o600)
            before = (root.stat().st_mtime_ns, (root / "PROTOCOL.json").stat().st_mtime_ns)
            self.assertEqual(self.m.validate_prepared(root), self.protocol)
            after = (root.stat().st_mtime_ns, (root / "PROTOCOL.json").stat().st_mtime_ns)
            self.assertEqual(after, before)
            with self.assertRaisesRegex(self.m.BridgeProtocolError, "already exists"):
                self.m.prepare(root)

    def test_validation_rejects_noncanonical_or_tampered_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            self.m.prepare(root)
            path = root / "PROTOCOL.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["claims"]["execution_authorized"] = True
            path.write_text(json.dumps(value), encoding="ascii")
            os.chmod(path, 0o600)
            with self.assertRaises(self.m.BridgeProtocolError):
                self.m.validate_prepared(root)

    def test_prepublication_failure_leaves_no_root_or_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            with mock.patch.object(
                self.m,
                "_write_all",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(
                    self.m.BridgeProtocolError,
                    "cannot create prepared protocol",
                ):
                    self.m.prepare(root)
            self.assertFalse(root.exists())

            with mock.patch.object(
                self.m,
                "_rename_noreplace",
                side_effect=self.m.BridgeProtocolError("injected publication failure"),
            ):
                with self.assertRaisesRegex(
                    self.m.BridgeProtocolError,
                    "injected publication failure",
                ):
                    self.m.prepare(root)
            self.assertFalse(root.exists())

    def test_optimized_build_matches_normal_build(self) -> None:
        process = subprocess.run(
            [sys.executable, "-OO", str(SOURCE), "build"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), self.protocol)


if __name__ == "__main__":
    unittest.main()
