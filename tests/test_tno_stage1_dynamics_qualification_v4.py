from __future__ import annotations

import ast
from contextlib import contextmanager
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import struct
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"


def _source_only_load() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    name = "_test_jx_tno_stage1_dynamics_qualification_v4"
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


@contextmanager
def _patched(module: types.ModuleType, **values):
    originals = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in originals.items():
            setattr(module, key, value)


def _cartesian_from_elements(
    a: float,
    e: float,
    inclination: float,
    node: float,
    peri: float,
    true_anomaly: float,
    mu: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    p = a * (1.0 - e * e)
    radius = p / (1.0 + e * math.cos(true_anomaly))
    rp = (radius * math.cos(true_anomaly), radius * math.sin(true_anomaly), 0.0)
    factor = math.sqrt(mu / p)
    vp = (-factor * math.sin(true_anomaly), factor * (e + math.cos(true_anomaly)), 0.0)
    co, so = math.cos(node), math.sin(node)
    cw, sw = math.cos(peri), math.sin(peri)
    ci, si = math.cos(inclination), math.sin(inclination)
    rotation = (
        (co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si),
        (so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si),
        (sw * si, cw * si, ci),
    )
    def apply(value):
        return tuple(sum(rotation[row][column] * value[column] for column in range(3)) for row in range(3))
    return apply(rp), apply(vp)


class ImportAndProtocolTests(unittest.TestCase):
    def test_source_only_import_does_not_load_dynamics_runtimes(self) -> None:
        before = set(sys.modules)
        module = _source_only_load()
        after = set(sys.modules) - before
        self.assertNotIn("rebound", after)
        self.assertNotIn("cupy", after)
        self.assertNotIn("numpy", after)
        self.assertEqual(module.HORIZON_JULIAN_YEARS, 1000)

    def test_protocol_freezes_exact_science_and_lane_scope(self) -> None:
        module = _source_only_load()
        protocol = module.build_protocol()
        self.assertEqual(protocol["selection"]["native_epoch_batch_count"], 12)
        self.assertEqual(protocol["selection"]["exact_tracer_count_per_ephemeris"], 108)
        self.assertEqual(protocol["time_contract"]["checkpoint_count_per_lane"], 1001)
        self.assertEqual(
            protocol["time_contract"]["jx_engine_time_scale_token"], "TDB"
        )
        self.assertEqual(
            protocol["time_contract"]["jx_time_coordinate_unit"],
            module.JX_TIME_COORDINATE_UNIT,
        )
        self.assertEqual(
            protocol["time_contract"]["jx_epoch_origin_semantics"],
            module.JX_EPOCH_ORIGIN_SEMANTICS,
        )
        self.assertEqual(protocol["expected_batch_lane_count"], 96)
        self.assertEqual(
            [lane["lane_id"] for lane in protocol["lanes_per_ephemeris"]],
            [
                "jx_reduced5_h_1_16",
                "jx_reduced5_h_1_32",
                "rebound511_ias15_reduced5",
                "rebound511_ias15_resolved11",
            ],
        )
        self.assertEqual(
            protocol["rebound_control"]["retained_wheel_sha256"],
            module.REBOUND_WHEEL_SHA256,
        )
        self.assertEqual(
            protocol["rebound_control"]["heartbeat_binding"],
            "PINNED_REBOUND_5_1_1_C_FIELD_WITH_LOCAL_CALLBACK_REFERENCE_DUE_MISSING_HB_SLOT",
        )
        smoke = protocol["rebound_control"][
            "prepreregistration_callback_smoke"
        ]
        self.assertTrue(smoke["completed"])
        self.assertEqual(
            [lane["heartbeat_callback_count"] for lane in smoke["lanes"]],
            [13, 1060],
        )
        self.assertEqual(
            [lane["encounter_or_guard_event_count"] for lane in smoke["lanes"]],
            [0, 0],
        )
        self.assertFalse(smoke["retained_scientific_output"])
        self.assertFalse(smoke["qualification_gates_evaluated"])
        self.assertFalse(protocol["claims"]["dynamical_classification_generated"])
        self.assertFalse(protocol["claims"]["resonance_classification_generated"])
        self.assertFalse(protocol["claims"]["science_result"])
        self.assertFalse(protocol["outcomes_generated"])
        self.assertEqual(
            protocol["retained_evidence_size_projection"][
                "ias15_trace_validation_hard_maximum_bytes"
            ],
            module.IAS_TRACE_MAXIMUM_BYTES,
        )
        self.assertEqual(module.IAS_TRACE_MAXIMUM_BYTES, 100_000_000_000)
        self.assertEqual(
            protocol["predeclared_runtime_contract"]["numpy_distribution"]["version"],
            module.NUMPY_DISTRIBUTION_VERSION,
        )
        self.assertEqual(
            protocol["predeclared_runtime_contract"]["numpy_distribution"]
            ["metadata"]["sha256"],
            module.NUMPY_METADATA_SHA256,
        )
        self.assertEqual(
            protocol["predeclared_runtime_contract"]["project_venv_launcher"]["launcher_path"],
            str(module.PROJECT_VENV_PYTHON),
        )
        self.assertTrue(
            protocol["publication"][
                "preparation_failure_never_unlinks_created_names"
            ]
        )
        self.assertEqual(
            protocol["result_artifact_roster"],
            [
                "PROTOCOL.json", "JX_MIDPOINTS.bin",
                "JX_RESTART_STREAMS.bin", "JX_SOURCE_NODES.bin",
                "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin", "RESULT.json",
            ],
        )
        self.assertEqual(
            protocol["jx_adapter"]["retained_full_source_nodes"]["exact_file_bytes"],
            module.EXPECTED_JX_SOURCE_BYTES,
        )
        self.assertEqual(
            protocol["jx_adapter"]["retained_restart_stream"]["exact_file_bytes"],
            1_731_456,
        )
        self.assertFalse(protocol["claims"]["externally_preregistered"])
        self.assertEqual(
            protocol["exact_gate_budgets"]["total_scattering_disagreement_chain"],
            {
                "numerator": 3,
                "denominator": 800,
                "nearest_binary64_hex": float(Fraction(3, 800)).hex(),
                "outward_ceiling_binary64_hex": "0x1.eb851eb851eb9p-9",
            },
        )
        self.assertEqual(
            len(protocol["jx_adapter"]["fresh_process_restart_cases"]), 4
        )
        self.assertEqual(
            protocol["jx_adapter"]["time_scale_token"], module.JX_TIME_SCALE
        )
        self.assertEqual(
            protocol["jx_adapter"]["legacy_verbose_time_scale_token_forbidden"],
            module.JX_REJECTED_VERBOSE_TIME_SCALE,
        )
        module.validate_protocol(protocol)

    def test_predecessor_v3_is_byte_preserved_and_failure_is_explicit(self) -> None:
        module = _source_only_load()
        self.assertEqual(
            module._sha256(module.PREDECESSOR_V3_SOURCE_PATH.read_bytes()),
            module.PREDECESSOR_V3_SOURCE_SHA256,
        )
        self.assertEqual(
            module._sha256(module.PREDECESSOR_V3_TEST_PATH.read_bytes()),
            module.PREDECESSOR_V3_TEST_SHA256,
        )
        self.assertEqual(
            module._sha256(module.PREDECESSOR_V3_PROTOCOL_PATH.read_bytes()),
            module.PREDECESSOR_V3_PROTOCOL_SHA256,
        )
        self.assertEqual(
            sorted(path.name for path in module.PREDECESSOR_V3_ROOT.iterdir()),
            ["PROTOCOL.json"],
        )
        record = module.build_protocol()["predecessor_v3_setup_failure"]
        self.assertEqual(record["completed_unretained_jx_lane_count"], 2)
        self.assertEqual(record["retained_completed_lane_count"], 0)
        self.assertEqual(record["completed_ias15_step_count"], 0)
        self.assertTrue(record["bounded_real_rebound_callback_smoke_completed"])
        self.assertTrue(record["dynamics_executed_but_unretained"])
        self.assertFalse(record["result_artifacts_published"])
        self.assertFalse(record["dynamics_outcome_generated"])
        self.assertFalse(record["qualification_gates_evaluated"])
        self.assertFalse(record["qualification_established"])
        self.assertFalse(record["scientific_disproof"])
        self.assertFalse(record["predecessor_bytes_mutated"])
        self.assertEqual(
            record["failure_class"], module.PREDECESSOR_V3_SETUP_FAILURE_CLASS
        )
        self.assertEqual(
            record["failure_message"],
            module.PREDECESSOR_V3_SETUP_FAILURE_MESSAGE,
        )
        self.assertEqual(record["observed_output_root_roster"], ["PROTOCOL.json"])

    def test_rebound_511_heartbeat_binds_the_c_field_and_retains_callback(self) -> None:
        module = _source_only_load()
        simulation = types.SimpleNamespace(_heartbeat=None)
        created: list[object] = []

        def callback_factory(function):
            created.append(function)
            return function

        rebound = types.SimpleNamespace(
            simulation=types.SimpleNamespace(AFF=callback_factory)
        )
        heartbeat = lambda pointer: None
        callback = module._bind_rebound_heartbeat(
            simulation=simulation,
            rebound=rebound,
            heartbeat=heartbeat,
        )
        self.assertIs(callback, heartbeat)
        self.assertIs(simulation._heartbeat, heartbeat)
        self.assertEqual(created, [heartbeat])
        self.assertNotIn(
            "simulation.heartbeat = heartbeat",
            SOURCE.read_text(encoding="utf-8"),
        )

    def test_runtime_receipt_plaining_is_recursive_exact_and_narrow(self) -> None:
        module = _source_only_load()
        nested = types.MappingProxyType(
            {
                "source_map_runtime_identity": types.MappingProxyType(
                    {
                        "kernel_id": "fixture",
                        "coordinate_canonicalization": types.MappingProxyType(
                            {"origin": "PRIMARY", "axes": ("X", "Y", "Z")}
                        ),
                    }
                ),
                "exact_replay_required": True,
            }
        )
        plain = module._plain_runtime_receipt(nested)
        self.assertIs(type(plain), dict)
        self.assertIs(type(plain["source_map_runtime_identity"]), dict)
        self.assertEqual(
            plain["source_map_runtime_identity"]["coordinate_canonicalization"],
            {"origin": "PRIMARY", "axes": ["X", "Y", "Z"]},
        )
        self.assertEqual(
            json.loads(module._canonical_json(plain)),
            plain,
        )
        with self.assertRaises(module.Stage1QualificationError):
            module._plain_runtime_receipt({"bad": {1, 2}})

    def test_primary_and_restart_builders_use_exact_tdb_token(self) -> None:
        module = _source_only_load()
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for function_name in ("_run_jx_lane", "_run_jx_restart_worker"):
            with self.subTest(function=function_name):
                calls = [
                    node
                    for node in ast.walk(functions[function_name])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "build_wisdom_holman_source_tile"
                ]
                self.assertEqual(len(calls), 1)
                keywords = {item.arg: item.value for item in calls[0].keywords}
                self.assertIn("time_scale", keywords)
                self.assertIsInstance(keywords["time_scale"], ast.Name)
                self.assertEqual(keywords["time_scale"].id, "JX_TIME_SCALE")
        self.assertEqual(module.JX_TIME_SCALE, "TDB")
        self.assertEqual(
            module.JX_REJECTED_VERBOSE_TIME_SCALE,
            "TDB_JULIAN_YEARS_SINCE_J2000",
        )

    def test_public_cpu_source_builder_time_scale_semantic_preflight(self) -> None:
        module = _source_only_load()
        command = [str(module.PROJECT_VENV_PYTHON), "-B"]
        if sys.flags.optimize >= 2:
            command.append("-OO")
        command.extend(
            [str(SOURCE), "source-builder-semantic-preflight"]
        )
        output_root_existed = module.DEFAULT_OUTPUT_ROOT.exists()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=module._production_child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        receipt = json.loads(completed.stdout.decode("ascii"))
        module.validate_cpu_source_builder_time_scale_semantic_preflight(receipt)
        self.assertEqual(
            [row["construction_path"] for row in receipt["records"]],
            ["PRIMARY_EXECUTION", "RESTART_WORKER"],
        )
        self.assertTrue(
            all(row["accepted_time_scale"] == "TDB" for row in receipt["records"])
        )
        self.assertTrue(
            all(
                row["rejected_time_scale"]
                == "TDB_JULIAN_YEARS_SINCE_J2000"
                for row in receipt["records"]
            )
        )
        self.assertFalse(receipt["gpu_modules_loaded"])
        self.assertFalse(receipt["gpu_device_queried"])
        mutations = (
            lambda value: value.__setitem__("construction_path_count", True),
            lambda value: value["records"][0].__setitem__(
                "accepted_completed_source_steps", True
            ),
            lambda value: value["records"][1].__setitem__(
                "rejection_message", "forged"
            ),
        )
        for mutate in mutations:
            forged = json.loads(json.dumps(receipt))
            mutate(forged)
            forged["content_sha256"] = module._content_sha256(
                module.SOURCE_BUILDER_PREFLIGHT_DOMAIN, forged
            )
            with self.assertRaises(module.Stage1QualificationError):
                module.validate_cpu_source_builder_time_scale_semantic_preflight(
                    forged
                )
        self.assertEqual(module.DEFAULT_OUTPUT_ROOT.exists(), output_root_existed)

    def test_prepare_is_create_only_and_validates_exact_reconstruction(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage1"
            protocol = module.prepare(root)
            self.assertEqual(module.validate_prepared(root), protocol)
            with self.assertRaises(module.Stage1QualificationError):
                module.prepare(root)
            path = root / "PROTOCOL.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["claims"]["science_result"] = True
            path.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaises(module.Stage1QualificationError):
                module.validate_prepared(root)

    def test_execute_blocks_missing_de441_before_runtime_import(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            module.prepare(output)
            missing = Path(directory) / "missing-de441"
            runtime = mock.Mock(side_effect=AssertionError("runtime must not load"))
            with _patched(module, DE441_ROOT=missing, _load_dynamics_runtime=runtime):
                with self.assertRaises(module.Stage1QualificationError):
                    module.execute(output, de441_root=missing)
            runtime.assert_not_called()

    def test_fresh_runtime_registry_rejects_spoofed_preloads(self) -> None:
        module = _source_only_load()
        for name, fake in (
            ("cupy", types.ModuleType("cupy")),
            ("rebound", object()),
            ("numpy", None),
            ("jxplanetx.cuda_restricted_wh", object()),
        ):
            with mock.patch.dict(sys.modules, {name: fake}):
                with self.assertRaises(module.Stage1QualificationError):
                    module._reject_preloaded_runtime_modules()

    def test_project_venv_launcher_checks_each_parent_chain(self) -> None:
        module = _source_only_load()
        failure = module.Stage1QualificationError("synthetic symlinked parent")
        with mock.patch.object(
            module, "_require_no_symlink_ancestors", side_effect=failure
        ) as guard:
            with self.assertRaises(module.Stage1QualificationError):
                module._project_venv_launcher_receipt()
        guard.assert_called_once_with(
            module.PROJECT_VENV_PYTHON, "project venv launcher link"
        )

    def test_import_only_child_uses_exact_project_venv_and_runtime_manifests(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            path, receipt = module._run_import_only_preflight(
                module.DEFAULT_REBOUND_TARGET, Path(directory)
            )
            self.assertEqual(path, Path(directory) / "IMPORT_PROBE.json")
            self.assertFalse(receipt["cuda_device_queried"])
            self.assertFalse(receipt["dynamics_executed"])
            self.assertEqual(
                receipt["fresh_child_interpreter"]["observed_sys_executable"],
                str(module.PROJECT_VENV_PYTHON),
            )
            names = {
                row["module"] for row in receipt["runtime_module_registry"]
            }
            self.assertTrue(set(module.REQUIRED_RUNTIME_MODULE_NAMES) <= names)
            module._validate_import_probe_receipt(receipt)
            mutations = (
                ("guard extra key", lambda value: value["source_only_runtime_import_guard"].__setitem__("extra", True)),
                ("guard prefix", lambda value: value["source_only_runtime_import_guard"].__setitem__("pycache_prefix", "/forged")),
                ("guard claim", lambda value: value["source_only_runtime_import_guard"].__setitem__("claim", "forged")),
                ("kernel directory", lambda value: value["source_only_runtime_import_guard"]["cupy_external_kernel_cache"].__setitem__("directory", "relative")),
                ("kernel memory", lambda value: value["source_only_runtime_import_guard"]["cupy_external_kernel_cache"].__setitem__("cupy_cache_in_memory", "0")),
                ("cuda cache", lambda value: value["source_only_runtime_import_guard"]["cupy_external_kernel_cache"].__setitem__("cuda_cache_disable", "0")),
                ("numpy version", lambda value: value["numpy_runtime_identity"].__setitem__("distribution_version", "0")),
                ("fft module", lambda value: value["cupy_import_identity"]["fft_config_module"].__setitem__("sha256", "0" * 64)),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    forged = json.loads(json.dumps(receipt))
                    mutate(forged)
                    forged["content_sha256"] = module._content_sha256(
                        module.IMPORT_PROBE_DOMAIN, forged
                    )
                    with self.assertRaises(module.Stage1QualificationError):
                        module._validate_import_probe_receipt(forged)
            forged = json.loads(json.dumps(receipt))
            forged_path = Path(directory) / "forged" / "IMPORT_PROBE.json"
            forged["probe_output_path"] = str(forged_path)
            forged["source_only_runtime_import_guard"]["pycache_prefix"] = str(
                forged_path.parent / ".import-probe-pycache-must-remain-absent"
            )
            forged["content_sha256"] = module._content_sha256(
                module.IMPORT_PROBE_DOMAIN, forged
            )
            with self.assertRaises(module.Stage1QualificationError):
                module._validate_import_probe_receipt(
                    forged, expected_path=path
                )

    def test_offline_numpy_binding_rejects_spoof_before_numerical_use(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            _path, preflight = module._run_import_only_preflight(
                module.DEFAULT_REBOUND_TARGET, Path(directory)
            )
        retained = preflight["numpy_runtime_identity"]
        fake = types.ModuleType("numpy")
        fake.asarray = mock.Mock(
            side_effect=AssertionError("spoofed NumPy must not be used")
        )
        with mock.patch.dict(sys.modules, {"numpy": fake}):
            with self.assertRaises(module.Stage1QualificationError):
                with module._bound_offline_numpy_runtime(retained):
                    self.fail("spoofed NumPy registry was accepted")
            fake.asarray.assert_not_called()

        with module._bound_offline_numpy_runtime(retained) as numpy_module:
            self.assertIs(sys.modules["numpy"], numpy_module)
            self.assertEqual(numpy_module.asarray([1.0]).shape, (1,))
            original_import = __import__

            def reject_fallback_numpy_import(name, *args, **kwargs):
                if name == "numpy":
                    raise AssertionError(
                        "provided-module replay must not import NumPy again"
                    )
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", reject_fallback_numpy_import):
                summary = module._source_node_invariant_summary(
                    (
                        ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
                        ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
                    ),
                    (
                        ((0.0, -0.1, 0.0), (0.0, 0.1, 0.0)),
                        ((0.0, -0.1, 0.0), (0.0, 0.1, 0.0)),
                    ),
                    (1.0, 1.0),
                    numpy_module=numpy_module,
                )
            self.assertEqual(
                summary["maximum_relative_massive_energy_drift_exact"],
                {
                    "exact_numerator": 0,
                    "exact_denominator": 1,
                    "outward_binary64_hex": "0x0.0p+0",
                },
            )
        self.assertFalse(
            any(
                name == "numpy" or name.startswith("numpy.")
                for name in sys.modules
                if type(name) is str
            )
        )

    def test_validate_output_rejects_numpy_spoof_before_replay(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            _path, preflight = module._run_import_only_preflight(
                module.DEFAULT_REBOUND_TARGET, Path(directory)
            )
            root = Path(directory) / "retained"
            root.mkdir()
            for name in (
                "PROTOCOL.json", "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
                "JX_SOURCE_NODES.bin", "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin",
                "RESULT.json",
            ):
                (root / name).write_bytes(b"")
            protocol = {"fixture": "protocol"}
            result = {
                "runtime_identity": {
                    "numpy_runtime_identity": preflight["numpy_runtime_identity"]
                }
            }

            def load_json(path, _label):
                if path.name == "PROTOCOL.json":
                    return module._canonical_json(protocol), protocol
                if path.name == "RESULT.json":
                    return b"", result
                raise AssertionError(f"unexpected JSON path: {path}")

            fake = types.ModuleType("numpy")
            fake.asarray = mock.Mock(
                side_effect=AssertionError("spoofed NumPy must not be used")
            )
            with mock.patch.object(
                module, "_load_json", side_effect=load_json
            ), mock.patch.object(
                module, "build_protocol", return_value=protocol
            ), mock.patch.object(
                module, "_regular_bytes", return_value=b""
            ), mock.patch.object(
                module, "_verify_input_documents", return_value=({}, {}, [])
            ), mock.patch.object(
                module, "validate_result_document"
            ) as replay, mock.patch.dict(sys.modules, {"numpy": fake}):
                with self.assertRaises(module.Stage1QualificationError):
                    module.validate_output(root)
            replay.assert_not_called()
            fake.asarray.assert_not_called()

    def test_offline_numpy_module_is_wired_through_every_replay_layer(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def calls(function_name, callee_name):
            return [
                node
                for node in ast.walk(functions[function_name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == callee_name
            ]

        output_calls = calls("validate_output", "validate_result_document")
        self.assertEqual(len(output_calls), 1)
        self.assertIsInstance(output_calls[0].args[-1], ast.Name)
        self.assertEqual(output_calls[0].args[-1].id, "numpy_module")

        result_calls = calls(
            "validate_result_document", "_validate_jx_source_evidence"
        )
        self.assertEqual(len(result_calls), 1)
        self.assertIsInstance(result_calls[0].args[-1], ast.Name)
        self.assertEqual(result_calls[0].args[-1].id, "offline_numpy_module")

        evidence_calls = calls(
            "_validate_jx_source_evidence", "_source_node_invariant_summary"
        )
        self.assertEqual(len(evidence_calls), 1)
        keywords = {item.arg: item.value for item in evidence_calls[0].keywords}
        self.assertIn("numpy_module", keywords)
        self.assertIsInstance(keywords["numpy_module"], ast.Name)
        self.assertEqual(keywords["numpy_module"].id, "numpy_module")

    def test_runtime_registry_filters_lazy_values_but_required_receipts_reject_them(self) -> None:
        module = _source_only_load()
        lazy = object()
        with mock.patch.dict(
            sys.modules,
            {"cupy.fft.config": lazy, "cupyx": lazy, "cupy.testing": lazy},
        ):
            names = module._runtime_module_names()
            self.assertNotIn("cupy.fft.config", names)
            self.assertNotIn("cupyx", names)
            self.assertNotIn("cupy.testing", names)
            with self.assertRaises(module.Stage1QualificationError):
                module._reject_preloaded_runtime_modules()
        with mock.patch.dict(sys.modules, {"cupy": object()}):
            with self.assertRaises(module.Stage1QualificationError):
                module._module_runtime_receipt("cupy")

    def test_manifest_authority_is_required_for_zero_byte_runtime_modules(self) -> None:
        module = _source_only_load()
        manifest = module._cupy_record_manifest()
        zero_members = {
            relative for relative, (size, digest) in manifest.items()
            if size == 0 and digest == module.EMPTY_SHA256
        }
        self.assertTrue(
            {
                "cupy_backends/__init__.py",
                "cupy_backends/cuda/__init__.py",
                "cupy_backends/cuda/api/__init__.py",
                "cupy/_functional/__init__.py",
                "cupy/_misc/__init__.py",
            }
            <= zero_members
        )
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.py"
            empty.write_bytes(b"")
            receipt = {"path": str(empty), "bytes": 0, "sha256": module.EMPTY_SHA256}
            module._validate_runtime_file_receipt(
                receipt, "manifest-authorized empty fixture", allow_empty=True
            )
            with self.assertRaises(module.Stage1QualificationError):
                module._validate_runtime_file_receipt(
                    receipt, "unauthorized empty fixture"
                )
            name = "cupy.synthetic_empty_manifest_member"
            spec = module.importlib.util.spec_from_file_location(name, empty)
            self.assertIsNotNone(spec)
            fake = types.ModuleType(name)
            fake.__file__ = str(empty)
            fake.__spec__ = spec
            fake.__loader__ = spec.loader
            fake.__cached__ = None
            expected = {
                str(empty.resolve()): (
                    0, module.EMPTY_SHA256, "PINNED_SYNTHETIC_RECORD"
                )
            }
            with mock.patch.dict(sys.modules, {name: fake}):
                exact = module._module_runtime_receipt(name, expected)
                self.assertEqual(exact["bytes"], 0)
                self.assertEqual(exact["sha256"], module.EMPTY_SHA256)
                with self.assertRaises(module.Stage1QualificationError):
                    module._module_runtime_receipt(name, {})
                with self.assertRaises(module.Stage1QualificationError):
                    module._module_runtime_receipt(
                        name,
                        {
                            str(empty.resolve()): (
                                0, "0" * 64, "PINNED_SYNTHETIC_RECORD"
                            )
                        },
                    )
            symlink = Path(directory) / "manifest-link.py"
            symlink.symlink_to(empty)
            link_name = "cupy.synthetic_symlink_manifest_member"
            link_spec = module.importlib.util.spec_from_file_location(
                link_name, symlink
            )
            self.assertIsNotNone(link_spec)
            link_module = types.ModuleType(link_name)
            link_module.__file__ = str(symlink)
            link_module.__spec__ = link_spec
            link_module.__loader__ = link_spec.loader
            link_module.__cached__ = None
            with mock.patch.dict(sys.modules, {link_name: link_module}):
                with self.assertRaises(module.Stage1QualificationError):
                    module._module_runtime_receipt(
                        link_name,
                        {
                            str(symlink.absolute()): (
                                0, module.EMPTY_SHA256,
                                "PINNED_SYNTHETIC_RECORD",
                            )
                        },
                    )
            real_parent = Path(directory) / "real-parent"
            real_parent.mkdir()
            parent_member = real_parent / "module.py"
            parent_member.write_bytes(b"pass\n")
            linked_parent = Path(directory) / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            lexical_member = linked_parent / "module.py"
            parent_name = "cupy.synthetic_symlinked_parent_member"
            parent_spec = module.importlib.util.spec_from_file_location(
                parent_name, lexical_member
            )
            self.assertIsNotNone(parent_spec)
            parent_module = types.ModuleType(parent_name)
            parent_module.__file__ = str(lexical_member)
            parent_module.__spec__ = parent_spec
            parent_module.__loader__ = parent_spec.loader
            parent_module.__cached__ = None
            with mock.patch.dict(sys.modules, {parent_name: parent_module}):
                with self.assertRaises(module.Stage1QualificationError):
                    module._module_runtime_receipt(
                        parent_name,
                        {
                            str(lexical_member.absolute()): (
                                5, module._sha256(b"pass\n"),
                                "PINNED_SYNTHETIC_RECORD",
                            )
                        },
                    )

    def test_numpy_runtime_receipt_rejects_every_custody_drift(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            _path, preflight = module._run_import_only_preflight(
                module.DEFAULT_REBOUND_TARGET, Path(directory)
            )
        identity = preflight["numpy_runtime_identity"]
        registry = {
            row["module"]: row
            for row in preflight["runtime_module_registry"]
        }
        module._validate_numpy_runtime_identity(identity, registry)
        mutations = (
            ("module hash", lambda value: value["module"].__setitem__("sha256", "0" * 64)),
            ("module path", lambda value: value["module"].__setitem__("path", str(module.SCRIPT_PATH))),
            ("core hash", lambda value: value["core_module"].__setitem__("sha256", "0" * 64)),
            ("core path", lambda value: value["core_module"].__setitem__("path", str(module.SCRIPT_PATH))),
            ("RECORD hash", lambda value: value["distribution_record"].__setitem__("sha256", "0" * 64)),
            ("RECORD path", lambda value: value["distribution_record"].__setitem__("path", str(module.SCRIPT_PATH))),
            ("METADATA hash", lambda value: value["distribution_metadata"].__setitem__("sha256", "0" * 64)),
            ("METADATA path", lambda value: value["distribution_metadata"].__setitem__("path", str(module.SCRIPT_PATH))),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                forged = json.loads(json.dumps(identity))
                mutate(forged)
                with self.assertRaises(module.Stage1QualificationError):
                    module._validate_numpy_runtime_identity(forged, registry)
        forged_registry = json.loads(json.dumps(registry))
        forged_registry["numpy"]["predeclared_authority"] = "FORGED"
        with self.assertRaises(module.Stage1QualificationError):
            module._validate_numpy_runtime_identity(identity, forged_registry)

    def test_prepare_failure_retains_concurrent_or_partial_names_fail_closed(self) -> None:
        module = _source_only_load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage1"

            def concurrent_replacement(path, payload):
                path.write_bytes(b"concurrent-owner")
                raise module.Stage1QualificationError("synthetic prepare failure")

            with mock.patch.object(
                module, "_write_create_only", side_effect=concurrent_replacement
            ):
                with self.assertRaises(module.Stage1QualificationError):
                    module.prepare(root)
            self.assertTrue(root.is_dir())
            self.assertEqual(
                (root / "PROTOCOL.json").read_bytes(), b"concurrent-owner"
            )


class PureOrbitalMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _source_only_load()

    def test_circular_equatorial_convention(self) -> None:
        mu = 4.0 * math.pi * math.pi
        result = self.module.extract_elliptic_orbital_elements(
            (1.0, 0.0, 0.0), (0.0, 2.0 * math.pi, 0.0), mu
        )
        self.assertAlmostEqual(result["semimajor_axis_AU"], 1.0, places=14)
        self.assertLess(result["eccentricity"], 1.0e-15)
        self.assertEqual(result["longitude_of_ascending_node_radians"], 0.0)
        self.assertEqual(result["argument_of_perihelion_radians"], 0.0)
        self.assertEqual(result["mean_anomaly_radians"], 0.0)

    def test_inclined_eccentric_roundtrip_oracle(self) -> None:
        mu = 4.0 * math.pi * math.pi
        a, e, inc, node, peri, true = 40.0, 0.2, 0.3, 0.7, 1.1, 0.4
        position, velocity = _cartesian_from_elements(a, e, inc, node, peri, true, mu)
        result = self.module.extract_elliptic_orbital_elements(position, velocity, mu)
        eccentric_anomaly = math.atan2(
            math.sqrt(1.0 - e * e) * math.sin(true), e + math.cos(true)
        )
        expected_m = (eccentric_anomaly - e * math.sin(eccentric_anomaly)) % (2.0 * math.pi)
        self.assertAlmostEqual(result["semimajor_axis_AU"], a, places=12)
        self.assertAlmostEqual(result["eccentricity"], e, places=14)
        self.assertAlmostEqual(result["inclination_radians"], inc, places=14)
        self.assertAlmostEqual(result["longitude_of_ascending_node_radians"], node, places=14)
        self.assertAlmostEqual(result["argument_of_perihelion_radians"], peri, places=14)
        self.assertAlmostEqual(result["mean_anomaly_radians"], expected_m, places=14)

    def test_invalid_and_unbound_states_fail_closed(self) -> None:
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.extract_elliptic_orbital_elements((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.extract_elliptic_orbital_elements((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), 1.0)
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.extract_elliptic_orbital_elements((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), True)
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.extract_elliptic_orbital_elements((1.0, math.nan, 0.0), (0.0, 1.0, 0.0), 1.0)

    def test_scattering_metric_boundary_is_strict_and_not_a_class(self) -> None:
        below = self.module.preliminary_scattering_metric([80.0, 82.0])
        equal = self.module.preliminary_scattering_metric([80.0, 83.0])
        above = self.module.preliminary_scattering_metric([80.0, 83.0000000001])
        self.assertFalse(below["threshold_exceeded_diagnostic_only"])
        self.assertFalse(equal["threshold_exceeded_diagnostic_only"])
        self.assertTrue(above["threshold_exceeded_diagnostic_only"])
        self.assertFalse(equal["dynamical_classification_generated"])
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.preliminary_scattering_metric([80.0], expected_count=2)
        with self.assertRaises(self.module.Stage1QualificationError):
            self.module.preliminary_scattering_metric([80.0, math.inf])


class AdapterMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _source_only_load()

    def _synthetic_batch(self):
        m = self.module
        massive = tuple(
            m.MassiveState(
                naif_id,
                body_id,
                float(index + 1),
                (float(index), float(2 * index), 0.0),
                (0.0, float(index) / 10.0, 0.0),
            )
            for index, (naif_id, body_id) in enumerate(
                zip(m.RESOLVED11_NAIF_IDS, m.RESOLVED11_BODY_IDS, strict=True)
            )
        )
        tracers = tuple(
            m.TracerState("synthetic", 1, "NOMINAL" if index == 0 else "COVARIANCE_CLONE", None if index == 0 else m.HOLDOUT_CLONE_INDICES[index - 1], (40.0 + index, 2.0, 1.0), (0.0, 1.0, 0.0))
            for index in range(9)
        )
        return m.BatchState("DE440S", 0, "synthetic", 1, 2_451_545.0, massive, tracers)

    def test_reduced_model_collapses_exact_inner_roster_and_omits_pluto(self) -> None:
        m = self.module
        batch = self._synthetic_batch()
        reduced = m.collapse_reduced_model(batch)
        self.assertEqual(tuple(item.naif_id for item in reduced.massive), (10, 5, 6, 7, 8))
        self.assertEqual(
            reduced.massive[0].gm_AU3_per_julian_year2,
            math.fsum(item.gm_AU3_per_julian_year2 for item in batch.massive[:6]),
        )
        self.assertNotIn(9, tuple(item.naif_id for item in reduced.massive))
        center_position, center_velocity = m._weighted_centroid(reduced.massive)
        self.assertLess(max(abs(value) for value in center_position), 1.0e-14)
        self.assertLess(max(abs(value) for value in center_velocity), 1.0e-14)
        self.assertEqual(len(reduced.tracers), 9)

    def test_annual_slice_bounds_cover_each_step_once_and_share_nodes(self) -> None:
        m = self.module
        for steps in m.JX_STEPS_PER_YEAR:
            first = m.annual_source_node_bounds(0, steps)
            second = m.annual_source_node_bounds(1, steps)
            last = m.annual_source_node_bounds(999, steps)
            self.assertEqual(first, (0, steps + 1))
            self.assertEqual(first[1] - 1, second[0])
            self.assertEqual(last[1], 1000 * steps + 1)
        with self.assertRaises(m.Stage1QualificationError):
            m.annual_source_node_bounds(1000, 16)
        with self.assertRaises(m.Stage1QualificationError):
            m.annual_source_node_bounds(0, 64)

    def test_de441_tno_reconstruction_preserves_heliocentric_j2000_vector(self) -> None:
        m = self.module
        bridge, custody, _bindings = m._verify_input_documents()
        batch = m._batch_states(bridge, custody, "DE441")[0]
        source = bridge["batches"][0]["tnos"][0]
        expected_position = tuple(float.fromhex(item) for item in source["heliocentric_j2000_position_AU_hex"])
        expected_velocity = tuple(float.fromhex(item) for item in source["heliocentric_j2000_velocity_AU_per_julian_year_hex"])
        actual_position = tuple(batch.tracers[0].position_AU[i] - batch.massive[0].position_AU[i] for i in range(3))
        actual_velocity = tuple(batch.tracers[0].velocity_AU_per_julian_year[i] - batch.massive[0].velocity_AU_per_julian_year[i] for i in range(3))
        self.assertLess(max(abs(actual_position[i] - expected_position[i]) for i in range(3)), 2.0e-14)
        self.assertLess(max(abs(actual_velocity[i] - expected_velocity[i]) for i in range(3)), 2.0e-15)

    def _comparison_lane(
        self,
        initial_a: float,
        peak_a: float,
        peak_position: float,
        peak_velocity: float,
    ):
        m = self.module
        checkpoints = []
        for year in range(m.CHECKPOINT_COUNT):
            is_peak = year == 500
            tracers = []
            positions = []
            velocities = []
            for tracer_order in range(m.TNO_STATES_PER_BATCH):
                a = peak_a if is_peak and tracer_order == 0 else initial_a
                tracers.append(
                    {
                        "tracer_order": tracer_order,
                        "sample_kind": "NOMINAL" if tracer_order == 0 else "COVARIANCE_CLONE",
                        "clone_index": None if tracer_order == 0 else m.HOLDOUT_CLONE_INDICES[tracer_order - 1],
                        "elements": {"semimajor_axis_AU": a.hex()},
                    }
                )
                positions.append([peak_position if is_peak and tracer_order == 0 else 0.0, 0.0, 0.0])
                velocities.append([0.0, peak_velocity if is_peak and tracer_order == 0 else 0.0, 0.0])
            checkpoints.append(
                {
                    "year_index": year,
                    "time_julian_years_hex": float(year).hex(),
                    "tracers": tracers,
                    "tracer_positions_AU": positions,
                    "tracer_velocities_AU_per_julian_year": velocities,
                }
            )
        return {"model": "REDUCED_5_BODY", "checkpoints": checkpoints}

    def test_comparison_uses_fixed_initial_denominator_and_intermediate_cartesian_peak(self) -> None:
        m = self.module
        left = self._comparison_lane(100.0, 110.0, 1.0, 0.25)
        right = self._comparison_lane(80.0, 80.0, 0.0, 0.0)
        metrics = m._comparison_metrics(left, right, 0)
        self.assertEqual(metrics["semimajor_axis"], Fraction(3, 10))
        self.assertEqual(metrics["position_AU_squared"], Fraction(1, 1))
        self.assertEqual(
            metrics["velocity_AU_per_julian_year_squared"], Fraction(1, 16)
        )

    def test_exact_fraction_chain_gate_accepts_exact_boundary_only(self) -> None:
        m = self.module
        boundary = m._exact_fraction_gate([Fraction(3, 800)], Fraction(3, 800))
        over = m._exact_fraction_gate(
            [Fraction(3, 800) + Fraction(1, 2**200)], Fraction(3, 800)
        )
        self.assertTrue(boundary["passed"])
        self.assertFalse(over["passed"])

    def test_exact_decision_boundaries_and_outward_reporting(self) -> None:
        m = self.module
        quantum = Fraction(1, 2**200)
        threshold = Fraction(3, 4000)
        self.assertTrue(m._exact_fraction_gate([threshold], threshold)["passed"])
        self.assertFalse(
            m._exact_fraction_gate([threshold + quantum], threshold)["passed"]
        )
        squared_threshold = Fraction(1, 500)
        self.assertTrue(
            m._exact_squared_gate(
                [squared_threshold * squared_threshold], squared_threshold
            )["passed"]
        )
        self.assertFalse(
            m._exact_squared_gate(
                [squared_threshold * squared_threshold + quantum],
                squared_threshold,
            )["passed"]
        )
        fine = Fraction(1, 2**100)
        floor = fine
        self.assertTrue(
            m._exact_scalar_contraction_passes(fine, Fraction(0), floor)
        )
        self.assertTrue(
            m._exact_scalar_contraction_passes(fine, 3 * fine, Fraction(0))
        )
        self.assertFalse(
            m._exact_scalar_contraction_passes(
                fine + quantum, 3 * fine - quantum, floor
            )
        )
        fine_squared = Fraction(1, 2**100)
        self.assertTrue(
            m._exact_squared_contraction_passes(
                fine_squared, Fraction(0), fine_squared
            )
        )
        self.assertTrue(
            m._exact_squared_contraction_passes(
                fine_squared, 9 * fine_squared, Fraction(0)
            )
        )
        self.assertFalse(
            m._exact_squared_contraction_passes(
                fine_squared + quantum,
                9 * fine_squared - quantum,
                fine_squared,
            )
        )
        floor_energy = Fraction(1, 2**46)
        self.assertTrue(
            m._exact_energy_contraction_passes(floor_energy, Fraction(0))
        )
        self.assertTrue(
            m._exact_energy_contraction_passes(Fraction(1, 3), Fraction(1))
        )
        self.assertFalse(
            m._exact_energy_contraction_passes(
                Fraction(1, 100), Fraction(3, 100) - quantum
            )
        )
        self.assertFalse(
            m._exact_energy_contraction_passes(
                floor_energy + quantum, Fraction(0)
            )
        )
        tiny_squared = Fraction(1, 2**1200)
        outward = m._sqrt_fraction_to_outward_float(tiny_squared)
        self.assertGreaterEqual(Fraction.from_float(outward) ** 2, tiny_squared)
        predecessor = math.nextafter(outward, -math.inf)
        self.assertLess(Fraction.from_float(predecessor) ** 2, tiny_squared)
        reference = (1.0, 2.0, 0.0)
        self.assertEqual(
            m._exact_squared_reference_ulp_floor(reference),
            sum(
                (
                    Fraction.from_float(64.0 * math.ulp(component)) ** 2
                    for component in reference
                ),
                Fraction(0, 1),
            ),
        )

    def test_exact_scattering_and_energy_fraction_records(self) -> None:
        m = self.module
        metric = m.preliminary_scattering_metric([80.0, 83.0])
        expected = (
            abs(Fraction.from_float(83.0) - Fraction.from_float(80.0))
            / Fraction.from_float(80.0)
        )
        self.assertEqual(
            m._fraction_from_record(
                metric["maximum_absolute_delta_over_initial_exact"],
                "test scattering",
            ),
            expected,
        )
        energy = m._maximum_exact_relative_energy_drift(
            (-1.0, -1.0 + 2.0**-52)
        )
        self.assertEqual(energy, Fraction(1, 2**52))
        record = m._fraction_record(energy)
        self.assertEqual(m._fraction_from_record(record, "test energy"), energy)
        damaged = dict(record)
        damaged["exact_denominator"] *= 2
        with self.assertRaises(m.Stage1QualificationError):
            m._fraction_from_record(damaged, "test energy")

    def _synthetic_gate_records(self):
        m = self.module

        def checkpoints(massive_count):
            result = []
            for year in range(m.CHECKPOINT_COUNT):
                result.append(
                    {
                        "year_index": year,
                        "time_julian_years_hex": float(year).hex(),
                        "tracers": [
                            {
                                "tracer_order": tracer_order,
                                "sample_kind": "NOMINAL" if tracer_order == 0 else "COVARIANCE_CLONE",
                                "clone_index": None if tracer_order == 0 else m.HOLDOUT_CLONE_INDICES[tracer_order - 1],
                                "elements": {"semimajor_axis_AU": float(80 + tracer_order).hex()},
                            }
                            for tracer_order in range(m.TNO_STATES_PER_BATCH)
                        ],
                        "massive_positions_AU": [
                            [float(source_order), 0.0, 0.0]
                            for source_order in range(massive_count)
                        ],
                        "massive_velocities_AU_per_julian_year": [
                            [0.0, float(source_order), 0.0]
                            for source_order in range(massive_count)
                        ],
                        "tracer_positions_AU": [
                            [float(40 + tracer_order), 1.0, 0.0]
                            for tracer_order in range(m.TNO_STATES_PER_BATCH)
                        ],
                        "tracer_velocities_AU_per_julian_year": [
                            [0.0, 0.5, 0.0]
                            for _tracer_order in range(m.TNO_STATES_PER_BATCH)
                        ],
                    }
                )
            return result

        reduced_checkpoints = checkpoints(len(m.REDUCED_BODY_IDS))
        resolved_checkpoints = checkpoints(len(m.RESOLVED11_BODY_IDS))
        zero_fraction = m._fraction_record(Fraction(0, 1))
        summary = {
            "maximum_relative_massive_energy_drift_hex": 0.0.hex(),
            "maximum_relative_massive_energy_drift_exact": zero_fraction,
            "maximum_relative_massive_angular_momentum_drift_hex": 0.0.hex(),
        }
        records = []
        for ephemeris in ("DE440S", "DE441"):
            for batch_order in range(m.OBJECT_COUNT):
                for lane_spec in m.LANE_SPECS:
                    resolved = lane_spec["model"] == "RESOLVED_11_BODY"
                    lane = {
                        "lane_id": lane_spec["lane_id"],
                        "engine": lane_spec["engine"],
                        "model": lane_spec["model"],
                        "checkpoints": resolved_checkpoints if resolved else reduced_checkpoints,
                        "summary": dict(summary),
                    }
                    if lane_spec["engine"] == "REBOUND_5.1.1_IAS15":
                        massive_count = len(m.RESOLVED11_BODY_IDS) if resolved else len(m.REDUCED_BODY_IDS)
                        lane["steps_done"] = 10
                        lane["accepted_step_guard"] = {
                            "sampling": "INITIAL_PLUS_REBOUND_HEARTBEAT_ACCEPTED_STEPS_WITH_STRAIGHT_CHORD_BETWEEN_SAMPLES_NOT_CONTINUOUS_CERTIFIED_GEOMETRY",
                            "initial_observation_count": 1,
                            "heartbeat_callback_count": 1010,
                            "integrate_call_count": 1000,
                            "duplicate_entry_record_count": 1000,
                            "total_observation_count": 1011,
                            "source_tracer_pair_observation_count": 1011 * m.TNO_STATES_PER_BATCH * massive_count,
                            "minimum_source_tracer_endpoint_distance_AU_hex": float(2.0).hex(),
                            "minimum_source_tracer_chord_distance_AU_hex": float(2.0).hex(),
                            "encounter_radius_AU_hex": float(1.0).hex(),
                            "outer_boundary_AU_hex": float(4096.0).hex(),
                            "encounter_or_guard_event_count": 0,
                            "callback_exception": None,
                            "trace": {
                                "schema": m.IAS_TRACE_SCHEMA,
                                "record_count": 1011,
                            },
                        }
                    records.append(
                        {
                            "ephemeris": ephemeris,
                            "batch_order": batch_order,
                            "lane": lane,
                        }
                    )
        return records

    def test_full_96_lane_1001_epoch_gate_evaluator(self) -> None:
        m = self.module
        result = m.evaluate_qualification_gates(self._synthetic_gate_records())
        self.assertTrue(result["all_predeclared_gates_passed"])
        self.assertEqual(
            result["gates"]["scattering_disagreement_chain"]["observation_count"],
            12 * 2 * 9 * 2,
        )

    def test_jx_midpoint_binary_roundtrip_is_bitwise_and_rejects_nonfinite(self) -> None:
        m = self.module
        positions = tuple((float(i), float(i + 1), float(i + 2)) for i in range(9))
        velocities = tuple((float(-i), float(i) / 10.0, 0.0) for i in range(9))
        first = m._pack_jx_internal_tracer_state(positions, velocities)
        second = m._pack_jx_internal_tracer_state(positions, velocities)
        self.assertEqual(first, second)
        self.assertEqual(len(first), m.JX_MIDPOINT_RECORD_BYTES)
        self.assertEqual(m._unpack_jx_internal_tracer_state(first), (positions, velocities))
        tampered = bytearray(first)
        tampered[:8] = bytes.fromhex("000000000000f87f")
        with self.assertRaises(m.Stage1QualificationError):
            m._unpack_jx_internal_tracer_state(bytes(tampered))

    def test_restart_pair_and_forward_array_binding_helpers(self) -> None:
        m = self.module
        midpoint = bytes(m.JX_MIDPOINT_RECORD_BYTES)
        suffix = midpoint * m.JX_RESTART_SUFFIX_CHECKPOINT_COUNT
        m._require_restart_stream_pair(suffix, suffix, midpoint)
        worker_suffix_receipt = {
            "suffix_checkpoint_count": m.JX_RESTART_SUFFIX_CHECKPOINT_COUNT,
            "suffix_byte_length": len(suffix),
            "suffix_file_sha256": m._sha256(suffix),
            "suffix_content_sha256": m._sha256(m.RESTART_DOMAIN + suffix),
            "final_state_sha256": m._sha256(suffix[-m.JX_MIDPOINT_RECORD_BYTES:]),
        }
        m._require_restart_worker_suffix_matches(suffix, worker_suffix_receipt)
        changed = bytearray(suffix)
        changed[m.JX_MIDPOINT_RECORD_BYTES] ^= 1
        with self.assertRaises(m.Stage1QualificationError):
            m._require_restart_stream_pair(suffix, bytes(changed), midpoint)
        with self.assertRaises(m.Stage1QualificationError):
            m._require_restart_worker_suffix_matches(
                bytes(changed), worker_suffix_receipt
            )
        both_changed = bytes(changed)
        m._require_restart_stream_pair(both_changed, both_changed, midpoint)
        sources = tuple((float(index), 0.0, 0.0) for index in range(5))
        velocities = tuple((0.0, float(index), 0.0) for index in range(5))
        gms = (1.0, 2.0, 3.0, 4.0, 5.0)
        tracers = tuple((40.0 + index, 0.0, 0.0) for index in range(9))
        tracer_velocities = tuple((0.0, 1.0, 0.0) for _ in range(9))
        expected = m._forward_center_primary_relative_tracers(
            sources, velocities, gms, tracers, tracer_velocities
        )
        checkpoint = {
            "tracer_positions_AU": expected[0],
            "tracer_velocities_AU_per_julian_year": expected[1],
        }
        m._require_centered_restart_checkpoint_matches(
            sources, velocities, gms, tracers, tracer_velocities, checkpoint
        )
        forged = list(tracers)
        forged[0] = (math.nextafter(forged[0][0], math.inf), 0.0, 0.0)
        observed = m._forward_center_primary_relative_tracers(
            sources, velocities, gms, forged, tracer_velocities
        )
        self.assertNotEqual(observed, expected)
        with self.assertRaises(m.Stage1QualificationError):
            m._require_centered_restart_checkpoint_matches(
                sources, velocities, gms, forged, tracer_velocities, checkpoint
            )

    def test_restart_receipt_rejects_forged_final_step(self) -> None:
        m = self.module
        ledger = {
            "full_source_tile_sha256": "0" * 64,
            "annual_source_slice_sha256_years_500_through_999": ["1" * 64] * 500,
            "steps_per_year": 16,
            "tracer_indices": list(range(9)),
            "maximum_kepler_iterations": 7,
            "final_global_source_step": 16_000,
            "fused_tracer_runtime_identity": {},
            "fresh_process_registry": {
                "checked_prefixes": list(m.RUNTIME_MODULE_PREFIXES),
                "preloaded_module_names": [],
                "fresh_registry_observed": True,
                "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
            },
        }
        receipt = {
            "schema": m.RESTART_RECEIPT_SCHEMA,
            "request_content_sha256": "2" * 64,
            "request_file_sha256": "3" * 64,
            "implementation_file_sha256": m._sha256(m._regular_bytes(m.SCRIPT_PATH, "test implementation")),
            "midpoint_record_sha256": "4" * 64,
            "midpoint_consumed": True,
            "fresh_process_resume_start_year": 500,
            "suffix_checkpoint_count": 501,
            "suffix_byte_length": m.JX_RESTART_SUFFIX_BYTES,
            "suffix_file_sha256": "5" * 64,
            "suffix_content_sha256": "6" * 64,
            "final_state_sha256": "7" * 64,
            "ledger": ledger,
            "ledger_content_sha256": m._sha256(m.RESTART_DOMAIN + m._canonical_json(ledger)),
            "runtime_identity": {},
            "bitwise_match_checked_by_parent": False,
            "content_sha256": "",
        }
        receipt["content_sha256"] = m._content_sha256(m.RESTART_DOMAIN, receipt)
        with mock.patch.object(m, "_validate_fused_tracer_runtime_identity"), mock.patch.object(
            m, "_validate_restart_runtime_identity"
        ):
            m._validate_restart_worker_receipt(receipt)
            forged = json.loads(json.dumps(receipt))
            forged["ledger"]["final_global_source_step"] = -999
            forged["ledger_content_sha256"] = m._sha256(
                m.RESTART_DOMAIN + m._canonical_json(forged["ledger"])
            )
            forged["content_sha256"] = m._content_sha256(
                m.RESTART_DOMAIN, forged
            )
            with self.assertRaises(m.Stage1QualificationError):
                m._validate_restart_worker_receipt(forged)

    def test_restart_runtime_identity_cross_binds_native_and_fused_cuda(self) -> None:
        m = self.module
        extension = {
            "path": "/synthetic/jxplanetx/_wisdom_holman_cpu_v3.so",
            "size_bytes": 123,
            "sha256": "8" * 64,
        }
        native = {
            "kernel_id": "synthetic",
            "g_kernel_id": "synthetic-g",
            "exact_fallback_kernel_id": "synthetic-fallback",
            "scientific_claim_state": "synthetic",
            "maximum_bodies": 64,
            "metric_count": 1,
            "counter_count": 1,
            "compiler_version": "synthetic",
            "compiler_flags": [],
            "release_wrapper_source_sha256": "9" * 64,
            "release_loop_source_sha256": "a" * 64,
            "release_base_loop_source_sha256": "b" * 64,
            "extension_binary": extension,
        }
        source = m._regular_bytes(m.SCRIPT_PATH, "test implementation")
        def module_receipt(name, path, size, digest):
            return {
                "module": name,
                "spec_name": name,
                "spec_origin": path,
                "loader_type": "synthetic.Loader",
                "file_kind": "PYTHON_SOURCE" if path.endswith(".py") else "EXTENSION_BINARY",
                "cached_path": None,
                "path": path,
                "bytes": size,
                "sha256": digest,
            }

        cupy_module = module_receipt(
            "cupy", str((m.CUPY_SITE_PACKAGES / m.CUPY_INIT_RELATIVE_PATH).resolve()),
            m.CUPY_INIT_BYTES, m.CUPY_INIT_SHA256,
        )
        cupy_core = module_receipt(
            "cupy._core.core",
            str((m.CUPY_SITE_PACKAGES / m.CUPY_CORE_RELATIVE_PATH).resolve()),
            m.CUPY_CORE_BYTES, m.CUPY_CORE_SHA256,
        )
        cupy_fft = module_receipt(
            "cupy.fft._config",
            str((m.CUPY_SITE_PACKAGES / m.CUPY_FFT_CONFIG_RELATIVE_PATH).resolve()),
            m.CUPY_FFT_CONFIG_BYTES, m.CUPY_FFT_CONFIG_SHA256,
        )
        numpy_module = module_receipt(
            "numpy",
            str((m.CUPY_SITE_PACKAGES / m.NUMPY_INIT_RELATIVE_PATH).resolve()),
            m.NUMPY_INIT_BYTES, m.NUMPY_INIT_SHA256,
        )
        numpy_core = module_receipt(
            "numpy._core._multiarray_umath",
            str((m.CUPY_SITE_PACKAGES / m.NUMPY_CORE_RELATIVE_PATH).resolve()),
            m.NUMPY_CORE_BYTES, m.NUMPY_CORE_SHA256,
        )
        numpy_identity = {
            "distribution_name": "numpy",
            "distribution_version": m.NUMPY_DISTRIBUTION_VERSION,
            "module_version": m.NUMPY_DISTRIBUTION_VERSION,
            "module": numpy_module,
            "core_module": numpy_core,
            "distribution_record": {
                "path": str(m.NUMPY_RECORD_PATH.resolve()),
                "bytes": m.NUMPY_RECORD_BYTES,
                "sha256": m.NUMPY_RECORD_SHA256,
            },
            "distribution_metadata": {
                "path": str(m.NUMPY_METADATA_PATH.resolve()),
                "bytes": m.NUMPY_METADATA_BYTES,
                "sha256": m.NUMPY_METADATA_SHA256,
            },
            "claim": "EXACT_LIVE_FILE_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION",
        }
        cupy = {
            "cupy_version": m.CUPY_DISTRIBUTION_VERSION,
            "cupy_module": cupy_module,
            "cupy_core_module": cupy_core,
            "cupy_fft_config_module": cupy_fft,
            "cuda_runtime_version": 13000,
            "cuda_local_runtime_version": 13000,
            "cuda_driver_version": 13000,
            "cuda_device_id": 0,
            "cuda_device_name": "synthetic GPU",
            "cuda_device_uuid": "00" * 16,
            "cuda_compute_capability": "9.0",
            "cuda_pci_domain_bus_device": "0000:01:00",
        }
        fused = {
            "cupy_version": cupy["cupy_version"],
            "cuda_runtime_version": cupy["cuda_runtime_version"],
            "cuda_local_runtime_version": cupy["cuda_local_runtime_version"],
            "cuda_driver_version": cupy["cuda_driver_version"],
            "device_index": cupy["cuda_device_id"],
            "device_name": cupy["cuda_device_name"],
            "device_uuid": cupy["cuda_device_uuid"],
            "device_compute_capability": cupy["cuda_compute_capability"],
            "device_pci_domain_bus_device": cupy["cuda_pci_domain_bus_device"],
        }
        identity = {
            "schema": "jx.tno.stage1-dynamics-qualification.restart-runtime-identity.v4",
            "python_executable": {
                "path": str(m.PYTHON_EXECUTABLE_PATH.resolve(strict=True)),
                "bytes": m.PYTHON_EXECUTABLE_BYTES,
                "sha256": m.PYTHON_EXECUTABLE_SHA256,
            },
            "fresh_child_interpreter": {},
            "stage1_implementation": {
                "path": str(m.SCRIPT_PATH.resolve(strict=True)),
                "bytes": len(source),
                "sha256": m._sha256(source),
            },
            "runtime_module_registry": [
                {
                    "module": "jxplanetx._wisdom_holman_cpu_v3",
                    "path": extension["path"],
                    "bytes": extension["size_bytes"],
                    "sha256": extension["sha256"],
                },
                {**cupy_module, "predeclared_authority": "PINNED_CUPY_RECORD"},
                {**cupy_core, "predeclared_authority": "PINNED_CUPY_RECORD"},
                {**cupy_fft, "predeclared_authority": "PINNED_CUPY_RECORD"},
            ],
            "numpy_runtime_identity": numpy_identity,
            "native_wisdom_holman_runtime_identity": native,
            "cupy_cuda": cupy,
            "fresh_process_registry": {
                "checked_prefixes": list(m.RUNTIME_MODULE_PREFIXES),
                "preloaded_module_names": [],
                "fresh_registry_observed": True,
                "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
            },
            "claim": "TRUSTED_FRESH_PROCESS_AND_EXACT_LIVE_FILE_REREAD_OBSERVATIONS_NOT_RUNTIME_AUTHENTICATION",
        }
        main = {
            "native_wisdom_holman_runtime_identity": native,
            "numpy_runtime_identity": json.loads(json.dumps(numpy_identity)),
        }
        main["numpy_runtime_identity"]["module"]["cached_path"] = (
            "/another/fresh-child/cache/numpy.pyc"
        )
        with mock.patch.object(m, "_validate_runtime_file_receipt"), mock.patch.object(
            m, "_validate_runtime_module_registry"
        ), mock.patch.object(
            m, "_validate_fresh_child_interpreter_receipt"
        ), mock.patch.object(m, "_validate_numpy_runtime_identity"):
            m._validate_restart_runtime_identity(identity, fused, main)
            mismatched = dict(fused)
            mismatched["cuda_driver_version"] += 1
            with self.assertRaises(m.Stage1QualificationError):
                m._validate_restart_runtime_identity(identity, mismatched, main)
            changed_main = {
                "native_wisdom_holman_runtime_identity": {
                    **native, "kernel_id": "forged"
                },
                "numpy_runtime_identity": main["numpy_runtime_identity"],
            }
            with self.assertRaises(m.Stage1QualificationError):
                m._validate_restart_runtime_identity(identity, fused, changed_main)
            changed_numpy = json.loads(json.dumps(main))
            changed_numpy["numpy_runtime_identity"]["core_module"]["sha256"] = (
                "0" * 64
            )
            with self.assertRaises(m.Stage1QualificationError):
                m._validate_restart_runtime_identity(
                    identity, fused, changed_numpy
                )

    def test_every_ias_integer_time_record_must_match_arrays(self) -> None:
        m = self.module
        exact = {
            key: [[0.0, 0.0, 0.0]]
            for key in m._CARTESIAN_KEYS
        }
        m._require_ias_integer_checkpoint_matches(exact, exact)
        forged = json.loads(json.dumps(exact))
        forged[m._CARTESIAN_KEYS[0]][0][0] = math.nextafter(0.0, math.inf)
        # A later correct duplicate cannot rescue this mismatched integer-time row:
        # validation raises on each row before marking that year observed.
        with self.assertRaises(m.Stage1QualificationError):
            m._require_ias_integer_checkpoint_matches(forged, exact)
        m._require_ias_integer_checkpoint_matches(exact, exact)

    def test_jx_source_tile_digest_and_invariant_evidence_are_tamper_evident(self) -> None:
        m = self.module
        steps = 16
        node_count = 2
        contract = {
            "tile_id": "synthetic",
            "epoch_origin_years_hex": 0.0.hex(),
            "global_source_step": 0,
            "source_step_years_hex": (1.0 / steps).hex(),
            "time_scale": m.JX_TIME_SCALE,
            "frame": "ICRF_J2000",
            "axes": "J2000_EQUATORIAL",
            "coordinate_system": "PRIMARY_RELATIVE_NONROTATING",
            "length_unit": "AU",
            "time_unit": "year",
            "unit_system_id": "AU-YEAR",
            "source_ids": list(m.REDUCED_BODY_IDS),
            "primary_source_id": m.REDUCED_BODY_IDS[0],
            "source_schedule_sha256": "0" * 64,
            "provenance": {"citation": "x", "sha256": "0" * 64, "source_id": "x", "version": "v3"},
            "node_count": node_count,
            "source_count": 5,
            "encounter_radii_AU_hex": [1.0.hex()] * 5,
            "certified_relative_acceleration_bounds_AU_per_year2_hex": [2.0.hex()] * 5,
        }
        positions = b"".join(
            struct.pack("<d", float(value))
            for value in range(node_count * 5 * 3)
        )
        velocities = bytes(len(positions))
        gms = struct.pack("<5d", 1.0, 2.0, 3.0, 4.0, 5.0)
        digest = m._jx_tile_sha256_from_raw(
            contract, positions, velocities, gms,
            node_count=node_count, tile_id="synthetic", global_source_step=0,
        )
        damaged = bytearray(positions)
        damaged[-1] ^= 1
        self.assertNotEqual(
            digest,
            m._jx_tile_sha256_from_raw(
                contract, bytes(damaged), velocities, gms,
                node_count=node_count, tile_id="synthetic", global_source_step=0,
            ),
        )
        invariant = {
            "massive_invariant_sampling": "ALL_JX_FIXED_SOURCE_NODES",
            "massive_invariant_sample_count": 2,
            "maximum_relative_massive_energy_drift_hex": 0.0.hex(),
            "maximum_relative_massive_energy_drift_exact": m._fraction_record(Fraction(0)),
            "maximum_relative_massive_angular_momentum_drift_hex": 0.0.hex(),
        }
        retained = {"unrelated": True, **invariant}
        m._require_source_invariant_summary(retained, invariant)
        retained["maximum_relative_massive_energy_drift_hex"] = (2.0**-52).hex()
        with self.assertRaises(m.Stage1QualificationError):
            m._require_source_invariant_summary(retained, invariant)

    def test_ias_trace_predicates_cover_chord_boundaries_and_step_gaps(self) -> None:
        m = self.module
        massive = tuple((100.0 * index, 0.0, 0.0) for index in range(5))
        massive_velocity = tuple((0.0, 0.0, 0.0) for _ in range(5))
        tracer_velocity = tuple((0.0, 0.0, 0.0) for _ in range(9))
        start_tracers = ((-2.0, 0.0, 0.0),) + tuple(
            (20.0 + index, 20.0, 0.0) for index in range(1, 9)
        )
        stop_tracers = ((2.0, 0.0, 0.0),) + start_tracers[1:]

        def decoded(tracers, flags=0):
            raw = m._pack_ias_trace_record(
                steps_done=0, observed_n=14, observed_n_active=5,
                observed_pair_count=45, event_flags=flags,
                time_julian_years=0.0, massive_positions=massive,
                massive_velocities=massive_velocity,
                tracer_positions=tracers, tracer_velocities=tracer_velocity,
            )
            return m._decode_ias_trace_record(raw, 5)

        first = m._ias_trace_observation_metrics(decoded(start_tracers), None, 5)
        second = m._ias_trace_observation_metrics(
            decoded(stop_tracers), first["relative_rows"], 5
        )
        self.assertTrue(second["event_flags"] & m.IAS_TRACE_EVENT_CHORD)
        boundary = list(start_tracers)
        boundary[0] = (1.0, 0.0, 0.0)
        endpoint = m._ias_trace_observation_metrics(decoded(tuple(boundary)), None, 5)
        self.assertTrue(endpoint["event_flags"] & m.IAS_TRACE_EVENT_ENDPOINT)
        outer = list(start_tracers)
        outer[0] = (4096.0, 0.0, 0.0)
        outer_result = m._ias_trace_observation_metrics(decoded(tuple(outer)), None, 5)
        self.assertTrue(outer_result["event_flags"] & m.IAS_TRACE_EVENT_OUTER)
        initial_raw = m._pack_ias_trace_record(
            steps_done=0, observed_n=14, observed_n_active=5,
            observed_pair_count=45, event_flags=0,
            time_julian_years=0.0, massive_positions=massive,
            massive_velocities=massive_velocity,
            tracer_positions=start_tracers,
            tracer_velocities=tracer_velocity,
        )
        self.assertEqual(
            m._validate_ias_trace_transition(
                0, initial_raw, 0, 0.0, None, -1, -math.inf, 0
            ),
            (False, 0),
        )
        self.assertEqual(
            m._validate_ias_trace_transition(
                1, initial_raw, 0, 0.0, initial_raw, 0, 0.0, 0
            ),
            (True, 1),
        )
        advanced_raw = m._pack_ias_trace_record(
            steps_done=1, observed_n=14, observed_n_active=5,
            observed_pair_count=45, event_flags=0,
            time_julian_years=0.1, massive_positions=massive,
            massive_velocities=massive_velocity,
            tracer_positions=stop_tracers,
            tracer_velocities=tracer_velocity,
        )
        self.assertEqual(
            m._validate_ias_trace_transition(
                2, advanced_raw, 1, 0.1, initial_raw, 0, 0.0, 1
            ),
            (False, 1),
        )
        for bad_step, bad_time in ((2, 0.1), (1, 0.0)):
            with self.assertRaises(m.Stage1QualificationError):
                m._validate_ias_trace_transition(
                    2, advanced_raw, bad_step, bad_time,
                    initial_raw, 0, 0.0, 1,
                )
        with self.assertRaises(m.Stage1QualificationError):
            m._validate_ias_trace_transition(
                2, initial_raw, 0, 0.0, initial_raw, 0, 0.0, 1
            )

    def test_ias_trace_cap_is_checked_before_any_record_bytes_are_written(self) -> None:
        m = self.module

        class Sink:
            def __init__(self, offset):
                self.offset = offset
                self.write_called = False

            def tell(self):
                return self.offset

            def write(self, payload):
                self.write_called = True
                return len(payload)

        record = b"12345678"
        blocked = Sink(m.IAS_TRACE_MAXIMUM_BYTES - len(record) + 1)
        with self.assertRaises(m.Stage1QualificationError):
            m._write_ias_trace_record_bounded(blocked, record)
        self.assertFalse(blocked.write_called)
        boundary = Sink(m.IAS_TRACE_MAXIMUM_BYTES - len(record))
        m._write_ias_trace_record_bounded(boundary, record)
        self.assertTrue(boundary.write_called)

    def test_stable_evidence_descriptor_detects_path_replacement_without_reopen(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.bin"
            replacement = Path(directory) / "replacement.bin"
            path.write_bytes(b"sealed-evidence")
            replacement.write_bytes(b"replacement")
            with self.assertRaises(m.Stage1QualificationError):
                with m._stable_read_handle(
                    path, "synthetic evidence", maximum=1024
                ) as handle:
                    with mock.patch.object(
                        m.os, "open", side_effect=AssertionError("must not reopen")
                    ):
                        self.assertEqual(
                            m._read_exact_handle_region(
                                handle, 0, len(b"sealed-evidence"),
                                "synthetic region",
                            ),
                            b"sealed-evidence",
                        )
                    m.os.replace(replacement, path)
            self.assertEqual(path.read_bytes(), b"replacement")

    def test_every_declared_json_integer_identity_rejects_bool(self) -> None:
        m = self.module
        for index, key in enumerate(sorted(m._EXACT_JSON_INT_KEYS)):
            with self.subTest(key=key):
                integer = index & 1
                forged = {key: bool(integer)}
                payload = m._canonical_json(forged)
                with self.assertRaises(m.Stage1QualificationError):
                    m._strict_json(payload, f"bool mutation {key}")
        for key in sorted(m._EXACT_JSON_INT_LIST_KEYS):
            with self.subTest(list_key=key):
                with self.assertRaises(m.Stage1QualificationError):
                    m._strict_json(
                        m._canonical_json({key: [False, 1]}),
                        f"bool roster mutation {key}",
                    )
        with self.assertRaises(m.Stage1QualificationError):
            m._strict_json(
                m._canonical_json(
                    {"naif_ids": {"__kind__": "tuple", "items": [10, True]}}
                ),
                "encoded tuple bool mutation",
            )

    def test_every_integer_leaf_in_protocol_and_pinned_json_has_bool_guard(self) -> None:
        m = self.module
        paths = (
            m.PILOT_PROTOCOL_PATH, m.STAGE0_PROTOCOL_PATH,
            m.STAGE0_STATES_PATH, m.BRIDGE_PROTOCOL_PATH,
            m.BRIDGE_RECEIPTS_PATH, m.BRIDGE_STATES_PATH,
            m.DE441_ROOT / "PROTOCOL.json", m.DE441_ROOT / "CUSTODY.json",
        )
        self.assertEqual(len(paths), 8)
        documents = [m.build_protocol()]
        documents.extend(json.loads(path.read_bytes()) for path in paths)
        scalar_examples = {}
        list_examples = {}
        typed_tuple_integer_count = 0

        def walk(value):
            nonlocal typed_tuple_integer_count
            if type(value) is dict:
                typed_tuple = (
                    value.get("__kind__") == "tuple"
                    and type(value.get("items")) is list
                )
                if typed_tuple:
                    typed_tuple_integer_count += sum(
                        type(item) is int for item in value["items"]
                    )
                for key, item in value.items():
                    if type(item) is int:
                        self.assertTrue(
                            key in m._EXACT_JSON_INT_KEYS
                            or key.endswith(m._EXACT_JSON_INT_SUFFIXES)
                            or key in m._EXACT_JSON_INT_OR_STRING_KEYS,
                            key,
                        )
                        scalar_examples.setdefault(key, item)
                    if type(item) is list and any(
                        type(entry) is int for entry in item
                    ) and not typed_tuple:
                        self.assertIn(key, m._EXACT_JSON_INT_LIST_KEYS)
                        list_examples.setdefault(key, item)
                    walk(item)
            elif type(value) is list:
                for item in value:
                    walk(item)

        for document in documents:
            walk(document)
        self.assertGreater(len(scalar_examples), 100)
        self.assertGreater(len(list_examples), 10)
        self.assertGreater(typed_tuple_integer_count, 1000)
        for key, integer in scalar_examples.items():
            with self.subTest(integer_key=key):
                with self.assertRaises(m.Stage1QualificationError):
                    m._guard_exact_json_integer_identities(
                        {key: bool(integer)}, "real-fixture scalar mutation"
                    )
        for key, roster in list_examples.items():
            with self.subTest(integer_list_key=key):
                forged = list(roster)
                target = next(
                    index for index, item in enumerate(forged)
                    if type(item) is int
                )
                forged[target] = bool(forged[target])
                with self.assertRaises(m.Stage1QualificationError):
                    m._guard_exact_json_integer_identities(
                        {key: forged}, "real-fixture roster mutation"
                    )
        with self.assertRaises(m.Stage1QualificationError):
            m._guard_exact_json_integer_identities(
                {"exact_pair": {"__kind__": "tuple", "items": ["x", False]}},
                "real-fixture typed-tuple mutation",
            )
        m._guard_exact_json_integer_identities(
            {"massive": {"__kind__": "tuple", "items": [True, True]}},
            "declared boolean tuple",
        )

    def test_guard_rejects_realistic_zero_and_one_bool_identity_mutations(self) -> None:
        m = self.module
        fixtures = (
            {"common_gm_byte_offset": False},
            {"initial_observation_count": True},
            {"device_index": False},
            {"source_order": False},
            {"tracer_indices": [False, 1, 2, 3, 4, 5, 6, 7, 8]},
            {"record_count": True},
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(m.Stage1QualificationError):
                    m._guard_exact_json_integer_identities(
                        fixture, "hostile retained JSON"
                    )

    def test_cartesian_lane_payload_is_deterministic_and_tamper_evident(self) -> None:
        m = self.module
        batch = self._synthetic_batch()
        reduced = m.collapse_reduced_model(batch)
        checkpoints = [
            m._checkpoint(year, reduced.massive, reduced.tracers)
            for year in range(m.CHECKPOINT_COUNT)
        ]
        summary = m._lane_summary(checkpoints)
        summary.update(
            {
                "massive_invariant_sampling": "ANNUAL_CHECKPOINTS_ONLY",
                "massive_invariant_sample_count": m.CHECKPOINT_COUNT,
            }
        )
        lane = {
            "lane_id": "rebound511_ias15_reduced5",
            "engine": "REBOUND_5.1.1_IAS15",
            "model": "REDUCED_5_BODY",
            "summary": summary,
            "checkpoints": checkpoints,
        }
        first, compact, _gate = m._packed_lane_arrays(lane)
        second, compact_second, _gate_second = m._packed_lane_arrays(lane)
        self.assertEqual(first, second)
        self.assertEqual(compact, compact_second)
        compact["cartesian_array"]["byte_offset"] = 0
        record = {
            "ephemeris": "DE440S",
            "batch_order": 0,
            "object_id": batch.object_id,
            "native_epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
            "lane": compact,
        }
        decoded = m._unpack_lane_arrays(first, record, batch)
        self.assertEqual(len(decoded["lane"]["checkpoints"]), m.CHECKPOINT_COUNT)
        damaged = bytearray(first)
        damaged[-1] ^= 1
        with self.assertRaises(m.Stage1QualificationError):
            m._unpack_lane_arrays(bytes(damaged), record, batch)
        lane["_private_bytes"] = b"forbidden"
        with self.assertRaises(m.Stage1QualificationError):
            m._packed_lane_arrays(lane)

    def test_source_only_import_guard_stays_absent(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "absent-cache"
            previous_environment = {
                key: m.os.environ.get(key)
                for key in ("CUPY_CACHE_DIR", "CUPY_CACHE_IN_MEMORY", "CUDA_CACHE_DISABLE")
            }
            with m._source_only_runtime_import_guard(cache) as receipt:
                self.assertFalse(cache.exists())
                self.assertTrue(sys.dont_write_bytecode)
                kernel = receipt["cupy_external_kernel_cache"]
                self.assertEqual(m.os.environ["CUPY_CACHE_IN_MEMORY"], "1")
                self.assertEqual(m.os.environ["CUDA_CACHE_DISABLE"], "1")
                self.assertEqual(m.os.environ["CUPY_CACHE_DIR"], kernel["directory"])
            self.assertFalse(cache.exists())
            self.assertTrue(receipt["verified_absent_after"])
            self.assertTrue(
                receipt["cupy_external_kernel_cache"]["verified_empty_after"]
            )
            self.assertTrue(
                receipt["cupy_external_kernel_cache"]["directory_removed_after"]
            )
            self.assertEqual(
                previous_environment,
                {
                    key: m.os.environ.get(key)
                    for key in (
                        "CUPY_CACHE_DIR",
                        "CUPY_CACHE_IN_MEMORY",
                        "CUDA_CACHE_DISABLE",
                    )
                },
            )

    def test_staged_publication_is_create_only_and_rolls_back_partial_links(self) -> None:
        m = self.module
        artifacts = (
            ("JX_MIDPOINTS.bin", b"midpoint"),
            ("JX_RESTART_STREAMS.bin", b"restart"),
            ("JX_SOURCE_NODES.bin", b"source"),
            ("IAS15_ACCEPTED_STEPS.bin", b"ias"),
            ("ARRAYS.bin", b"arrays"),
            ("RESULT.json", b"{}\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            root.mkdir()
            (root / "PROTOCOL.json").write_bytes(b"{}\n")
            real_link = m.os.link
            calls = 0

            def fail_second(source, destination, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic link failure")
                return real_link(source, destination, **kwargs)

            with mock.patch.object(m.os, "link", side_effect=fail_second):
                with self.assertRaises(m.Stage1QualificationError):
                    m._publish_result_artifacts(root, artifacts)
            self.assertEqual(
                {entry.name for entry in root.iterdir()},
                {"PROTOCOL.json", "JX_MIDPOINTS.bin"},
            )
            clean = Path(directory) / "clean-result"
            clean.mkdir()
            (clean / "PROTOCOL.json").write_bytes(b"{}\n")
            m._publish_result_artifacts(clean, artifacts)
            self.assertEqual(
                {entry.name for entry in clean.iterdir()},
                {
                    "PROTOCOL.json", "JX_MIDPOINTS.bin",
                    "JX_RESTART_STREAMS.bin", "JX_SOURCE_NODES.bin",
                    "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin", "RESULT.json",
                },
            )
            self.assertEqual((clean / "RESULT.json").read_bytes(), b"{}\n")
            with self.assertRaises(m.Stage1QualificationError):
                m._publish_result_artifacts(clean, artifacts)

    def test_publication_rollback_never_unlinks_concurrent_replacement(self) -> None:
        m = self.module
        artifacts = (
            ("JX_MIDPOINTS.bin", b"midpoint"),
            ("JX_RESTART_STREAMS.bin", b"restart"),
            ("JX_SOURCE_NODES.bin", b"source"),
            ("IAS15_ACCEPTED_STEPS.bin", b"ias"),
            ("ARRAYS.bin", b"arrays"),
            ("RESULT.json", b"{}\n"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            root.mkdir()
            (root / "PROTOCOL.json").write_bytes(b"{}\n")
            real_link = m.os.link
            calls = 0

            def replace_then_fail(source, destination, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_link(source, destination, **kwargs)
                first = root / "JX_MIDPOINTS.bin"
                first.unlink()
                first.write_bytes(b"concurrent-owner")
                raise OSError("synthetic concurrent replacement")

            with mock.patch.object(m.os, "link", side_effect=replace_then_fail):
                with self.assertRaises(m.Stage1QualificationError):
                    m._publish_result_artifacts(root, artifacts)
            self.assertEqual(
                (root / "JX_MIDPOINTS.bin").read_bytes(), b"concurrent-owner"
            )


if __name__ == "__main__":
    unittest.main()
