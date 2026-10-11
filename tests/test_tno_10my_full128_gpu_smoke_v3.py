from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v3.py"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def load_module() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    name = f"_jx_test_full128_smoke_v3_{os.getpid()}_{id(object())}"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(
            compile(payload, str(SOURCE), "exec", dont_inherit=True, optimize=sys.flags.optimize),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class Full128SmokeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.protocol = cls.module.build_protocol()
        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-B", str(SOURCE),
             "cpu-relocation-probe"],
            cwd=PROJECT_ROOT, env=cls.module._production_child_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=300,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))
        cls.probe = json.loads(completed.stdout)
        cls.module.validate_cpu_relocation_probe(cls.probe)
        cls.live_extension = cls.module._read_live_native_exact()
        cls.historical_extension = {
            "path": cls.module.HISTORICAL_NATIVE_PATH,
            "size_bytes": cls.module.NATIVE_BYTES,
            "sha256": cls.module.NATIVE_SHA256,
        }
        cls.synthetic_identity = {
            "provider_id": "CPU_SYNTHETIC_RELOCATION_FIXTURE_NO_SOURCE_BUILD",
            "python_version": "synthetic",
            "source_map_runtime_identity": {
                "kernel_id": "CPU_SYNTHETIC_RELOCATION_FIXTURE",
                "extension_binary": copy.deepcopy(cls.live_extension),
            },
            "exact_replay_required": True,
        }
        neutral = copy.deepcopy(cls.synthetic_identity)
        del neutral["source_map_runtime_identity"]["extension_binary"]["path"]
        rebased = copy.deepcopy(cls.synthetic_identity)
        rebased["source_map_runtime_identity"]["extension_binary"]["path"] = (
            cls.module.HISTORICAL_NATIVE_PATH
        )
        cls.synthetic_neutral_sha = hashlib.sha256(canonical(neutral)).hexdigest()
        cls.synthetic_rebased_sha = hashlib.sha256(canonical(rebased)).hexdigest()
        cls.base = cls.module._validate_relocation_identity(
            cls.synthetic_identity,
            live_extension=cls.live_extension,
            historical_extension=cls.historical_extension,
            expected_neutral_sha256=cls.synthetic_neutral_sha,
            expected_rebased_sha256=cls.synthetic_rebased_sha,
        )
        cls.runtime_post = cls.synthetic_runtime_post()
        cls.receipt = cls.module._bind_relocation_to_runtime_post(
            cls.base, cls.runtime_post,
        )

    @classmethod
    def synthetic_runtime_post(cls) -> dict[str, object]:
        native = cls.module._read_live_native_exact()
        return {
            "content_sha256": "2" * 64,
            "v3_runtime_identity": {
                "content_sha256": "1" * 64,
                "critical_jx_modules": [{
                    "module": cls.module.NATIVE_MODULE,
                    "path": native["path"],
                    "bytes": native["size_bytes"],
                    "sha256": native["sha256"],
                }],
                "engine_runtime_projection": {
                    "method_id": "CPU_TEST_SYNTHETIC_RUNTIME_POST_NO_EXECUTION",
                },
            },
        }

    def rehash_receipt(self, value: dict[str, object]) -> None:
        value["content_sha256"] = self.module._domain_hash(
            self.module.RELOCATION_DOMAIN, value,
        )

    def test_protocol_binds_v2_failure_and_v3_namespace(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], self.module.PROTOCOL_SCHEMA)
        self.assertEqual(
            protocol["preregistration"]["branch"],
            "research/tno-10my-reduced5-execution-v3-preregistered-20261011",
        )
        self.assertEqual(
            protocol["preregistration"]["annotated_tag"],
            "tno-10my-reduced5-execution-v3-preregistered-20261011",
        )
        predecessor = protocol["predecessor_v2"]
        self.assertEqual(predecessor["commit"], self.module.V2_COMMIT)
        self.assertEqual(predecessor["tag_object"], self.module.V2_TAG_OBJECT)
        self.assertEqual(predecessor["root_roster"], ["PROTOCOL.json"])
        failed = predecessor["failed_attempt"]
        self.assertEqual(failed["attempt_count"], 1)
        self.assertTrue(failed["runtime_preflight_and_setup_reached"])
        self.assertTrue(failed["massive_source_build_completed"])
        self.assertEqual(failed["lifecycle_initialize_call_count"], 0)
        self.assertEqual(failed["tracer_advance_call_count"], 0)
        self.assertEqual(failed["completed_segment_count"], 0)
        self.assertFalse(failed["result_published"])
        self.assertFalse(failed["scientific_outcome_generated"])
        self.assertFalse(failed["traceback_authenticated_by_retained_v2_artifact"])
        self.assertTrue(all(value is False for value in protocol["claims"].values()))
        self.module.validate_protocol(protocol)

    def test_v2_source_test_protocol_root_and_git_pins_exact(self) -> None:
        predecessor = self.module._predecessor_v2()
        self.assertEqual(predecessor["source"]["sha256"], self.module.V2_SHA256)
        self.assertEqual(predecessor["test"]["sha256"], self.module.V2_TEST_SHA256)
        self.assertEqual(
            predecessor["protocol"]["content_sha256"],
            self.module.V2_PROTOCOL_CONTENT_SHA256,
        )
        self.assertEqual(
            {item.name for item in self.module.V2_ROOT.iterdir()}, {"PROTOCOL.json"},
        )

    def test_protocol_declares_only_path_relocation_and_no_physics_drift(self) -> None:
        contract = self.protocol["source_build_runtime_relocation"]
        self.assertEqual(contract["relocated_field_token"], self.module.RELOCATED_FIELD_TOKEN)
        self.assertEqual(
            contract["historical_path_rebased_runtime_sha256"],
            self.module.HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256,
        )
        self.assertFalse(contract["physics_force_step_grid_threshold_or_gate_changed"])
        self.assertFalse(self.protocol["claims"]["bounded_gpu_smoke_executed"])
        self.assertFalse(self.protocol["claims"]["ten_myr_execution_authorized"])
        self.assertFalse(self.protocol["claims"]["classification_generated"])
        self.assertFalse(self.protocol["claims"]["science_result"])

    def test_prepare_validate_is_create_only_and_default_is_not_materialized(self) -> None:
        self.assertFalse(self.module.DEFAULT_OUTPUT_ROOT.exists())
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared-v3"
            self.module.prepare(root)
            retained = self.module.validate_prepared(root)
            self.assertEqual(canonical(retained), canonical(self.protocol))
            self.assertEqual({item.name for item in root.iterdir()}, {"PROTOCOL.json"})
            with self.assertRaises(self.module.Full128SmokeError):
                self.module.prepare(root)

    def test_inner_production_globals_and_calls_are_v3_wired(self) -> None:
        inner = self.module._INNER
        self.assertEqual(inner.SCRIPT_PATH, self.module.SCRIPT_PATH)
        self.assertEqual(inner.TEST_PATH, self.module.TEST_PATH)
        self.assertEqual(inner.DEFAULT_OUTPUT_ROOT, self.module.DEFAULT_OUTPUT_ROOT)
        self.assertIs(inner.build_protocol, self.module.build_protocol)
        self.assertIs(inner._source_build_identity, self.module._source_build_identity)
        self.assertIs(inner._run_full129_lane, self.module._run_full129_lane)
        self.assertIs(inner._build_result_and_artifacts, self.module._build_result_and_artifacts)
        self.assertIs(inner._validate_output_snapshot, self.module._validate_output_snapshot)
        self.assertIs(inner.validate_output, self.module.validate_output)
        self.assertIsNot(
            self.module._ORIGINAL_SOURCE_BUILD_IDENTITY,
            self.module._source_build_identity,
        )

    def test_fresh_production_child_uses_current_source_not_editable_pth(self) -> None:
        value = self.module.validate_cpu_relocation_probe(self.probe)
        expected = self.module.PROJECT_ROOT / self.module.NATIVE_RELATIVE_PATH
        self.assertEqual(value["native_module_path"], str(expected))
        self.assertTrue(value["native_module_loaded_from_current_project"])
        self.assertFalse(value["stale_editable_path_selected"])
        self.assertEqual(
            value["production_pythonpath"],
            os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))),
        )
        self.assertFalse(value["source_builder_called"])
        self.assertFalse(value["source_map_invoked"])
        self.assertFalse(value["gpu_imported"])
        self.assertFalse(value["gpu_executed"])
        self.assertFalse(value["dynamics_api_invoked"])
        self.assertFalse(value["outcome_generated"])
        self.assertNotIn("massive_source_step_count", value)

    def test_public_offline_import_scope_overrides_old_editable_path(self) -> None:
        self.assertEqual(self.module._jx_module_names(), ())
        with self.module._current_project_jx_import_scope():
            __import__(self.module.NATIVE_MODULE)
            native = sys.modules[self.module.NATIVE_MODULE]
            self.assertEqual(
                Path(native.__file__).resolve(),
                self.module.PROJECT_ROOT / self.module.NATIVE_RELATIVE_PATH,
            )
        self.assertEqual(self.module._jx_module_names(), ())

    def test_public_offline_import_scope_rejects_preloaded_jx(self) -> None:
        counterfeit = types.ModuleType("jxplanetx.counterfeit_preload")
        counterfeit.__file__ = self.module.HISTORICAL_NATIVE_PATH
        sys.modules[counterfeit.__name__] = counterfeit
        try:
            with self.assertRaises(self.module.Full128SmokeError):
                with self.module._current_project_jx_import_scope():
                    pass
        finally:
            sys.modules.pop(counterfeit.__name__, None)

    def test_path_neutral_and_historical_rebase_are_exact(self) -> None:
        base = self.base
        live = copy.deepcopy(base["live_runtime_identity"])
        neutral = copy.deepcopy(live)
        del neutral["source_map_runtime_identity"]["extension_binary"]["path"]
        self.assertEqual(neutral, base["path_neutral_runtime_identity"])
        self.assertEqual(
            hashlib.sha256(canonical(neutral)).hexdigest(),
            self.synthetic_neutral_sha,
        )
        live["source_map_runtime_identity"]["extension_binary"]["path"] = (
            self.module.HISTORICAL_NATIVE_PATH
        )
        self.assertEqual(
            hashlib.sha256(canonical(live)).hexdigest(),
            self.synthetic_rebased_sha,
        )

    def test_relocation_receipt_crossbinds_runtime_post_native_module(self) -> None:
        value = self.receipt
        extension = value["live_extension_binary"]
        critical = value["runtime_post_critical_native_module"]
        self.assertEqual(critical["path"], extension["path"])
        self.assertEqual(critical["bytes"], extension["size_bytes"])
        self.assertEqual(critical["sha256"], extension["sha256"])
        self.assertEqual(value["runtime_post_content_sha256"], "2" * 64)
        self.assertEqual(value["runtime_post_v3_identity_content_sha256"], "1" * 64)

    def test_coordinated_result_runtime_post_outside_path_forgery_rejected(self) -> None:
        identity = copy.deepcopy(self.synthetic_identity)
        post = copy.deepcopy(self.runtime_post)
        outside = "/tmp/counterfeit/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
        identity["source_map_runtime_identity"]["extension_binary"]["path"] = outside
        post["v3_runtime_identity"]["critical_jx_modules"][0]["path"] = outside
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_relocation_identity(
                identity, live_extension=self.live_extension,
                historical_extension=self.historical_extension,
                expected_neutral_sha256=self.synthetic_neutral_sha,
                expected_rebased_sha256=self.synthetic_rebased_sha,
            )
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._critical_native_row(post)

    def test_coordinated_native_size_forgery_rejected(self) -> None:
        identity = copy.deepcopy(self.synthetic_identity)
        new_size = self.module.NATIVE_BYTES + 1
        identity["source_map_runtime_identity"]["extension_binary"]["size_bytes"] = new_size
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_relocation_identity(
                identity, live_extension=self.live_extension,
                historical_extension=self.historical_extension,
                expected_neutral_sha256=self.synthetic_neutral_sha,
                expected_rebased_sha256=self.synthetic_rebased_sha,
            )

    def test_coordinated_native_sha_forgery_rejected(self) -> None:
        identity = copy.deepcopy(self.synthetic_identity)
        fake_sha = "0" * 64
        identity["source_map_runtime_identity"]["extension_binary"]["sha256"] = fake_sha
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_relocation_identity(
                identity, live_extension=self.live_extension,
                historical_extension=self.historical_extension,
                expected_neutral_sha256=self.synthetic_neutral_sha,
                expected_rebased_sha256=self.synthetic_rebased_sha,
            )

    def test_other_runtime_field_drift_rejected_after_coordinated_rehash(self) -> None:
        identity = copy.deepcopy(self.synthetic_identity)
        identity["platform"] = "counterfeit-platform"
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_relocation_identity(
                identity, live_extension=self.live_extension,
                historical_extension=self.historical_extension,
                expected_neutral_sha256=self.synthetic_neutral_sha,
                expected_rebased_sha256=self.synthetic_rebased_sha,
            )

    def test_runtime_post_critical_row_replacement_rejected(self) -> None:
        post = copy.deepcopy(self.runtime_post)
        post["v3_runtime_identity"]["critical_jx_modules"][0]["module"] = (
            "jxplanetx.counterfeit"
        )
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_relocation_receipt(self.receipt, post)

    def test_bool_alias_and_claim_escalation_remain_rejected(self) -> None:
        years = list(self.module.CHECKPOINT_YEARS)
        years[0] = False
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._INNER._validate_checkpoint_years(years)
        protocol = copy.deepcopy(self.protocol)
        protocol["claims"]["gpu_executed"] = True
        protocol["content_sha256"] = self.module._domain_hash(
            self.module.PROTOCOL_DOMAIN, protocol,
        )
        with self.assertRaises(self.module.Full128SmokeError):
            self.module.validate_protocol(protocol)

    def test_real_lane_wires_nonoverridable_frozen_relocation_hashes(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        source_identity = functions["_source_build_identity"]
        called = {
            node.func.id for node in ast.walk(source_identity)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_relocation_base", called)
        production = functions["_relocation_base"]
        calls = [
            node for node in ast.walk(production)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_relocation_identity"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords["expected_neutral_sha256"], ast.Name)
        self.assertEqual(
            keywords["expected_neutral_sha256"].id,
            "HISTORICAL_PATH_NEUTRAL_RUNTIME_SHA256",
        )
        self.assertIsInstance(keywords["expected_rebased_sha256"], ast.Name)
        self.assertEqual(
            keywords["expected_rebased_sha256"].id,
            "HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256",
        )
        lane = functions["_run_full129_lane"]
        lane_calls = {
            node.func.id for node in ast.walk(lane)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_ORIGINAL_RUN_FULL129_LANE", lane_calls)

    @unittest.skipIf(sys.flags.optimize > 0, "normal mode owns opposite-mode byte check")
    def test_optimized_protocol_bytes_identical(self) -> None:
        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-OO", str(SOURCE), "build"],
            cwd=PROJECT_ROOT, env=self.module._production_child_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        self.assertEqual(completed.stdout, canonical(self.protocol))


if __name__ == "__main__":
    unittest.main()
