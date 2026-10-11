from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v2.py"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def load_module() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    name = f"_jx_test_full128_smoke_v2_{os.getpid()}_{id(object())}"
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


class Full128SmokeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.fixture, cls.payloads = cls.module.build_cpu_synthetic_fixture()

    def fresh_fixture(self) -> tuple[dict[str, object], dict[str, bytes]]:
        return copy.deepcopy(self.fixture), dict(self.payloads)

    def test_protocol_binds_immutable_v1_and_v2_namespace(self) -> None:
        protocol = self.module.build_protocol()
        self.assertEqual(protocol["schema"], self.module.PROTOCOL_SCHEMA)
        self.assertEqual(
            protocol["preregistration"]["branch"],
            "research/tno-10my-reduced5-execution-v2-preregistered-20261011",
        )
        self.assertEqual(
            protocol["preregistration"]["annotated_tag"],
            "tno-10my-reduced5-execution-v2-preregistered-20261011",
        )
        predecessor = protocol["predecessor_v1"]
        self.assertEqual(predecessor["commit"], "13c17fab588ec6b073a036d245be4f67c48ec002")
        self.assertEqual(predecessor["tag_object"], "9ea2950aadd358e2276e774cf39780b3dcfd2a25")
        self.assertEqual(predecessor["root_roster"], ["PROTOCOL.json"])
        self.assertFalse(predecessor["disposition"]["v1_production_execute_invoked"])
        self.assertTrue(
            predecessor["disposition"][
                "post_discovery_v3_runtime_preflight_operator_session_check_run"
            ]
        )
        self.assertFalse(predecessor["disposition"]["post_discovery_check_retained"])
        self.assertFalse(predecessor["disposition"]["post_discovery_check_authenticated"])
        self.assertFalse(protocol["claims"]["bounded_gpu_smoke_executed"])
        self.assertFalse(protocol["cpu_fixture_contract"]["accepted_by_production_validator"])
        self.module.validate_protocol(protocol)

    def test_v1_root_and_bytes_remain_exact(self) -> None:
        predecessor = self.module._predecessor_v1()
        self.assertEqual(predecessor["source"]["sha256"], self.module.V1_SHA256)
        self.assertEqual(predecessor["test"]["sha256"], self.module.V1_TEST_SHA256)
        self.assertEqual(predecessor["protocol"]["sha256"], self.module.V1_PROTOCOL_SHA256)
        self.assertEqual(
            set(item.name for item in self.module.V1_ROOT.iterdir()), {"PROTOCOL.json"},
        )

    def test_prepare_validate_create_only_and_default_roster(self) -> None:
        if self.module.DEFAULT_OUTPUT_ROOT.exists():
            retained_default = self.module.validate_prepared(
                self.module.DEFAULT_OUTPUT_ROOT
            )
            self.assertEqual(retained_default["schema"], self.module.PROTOCOL_SCHEMA)
            self.assertEqual(
                {item.name for item in self.module.DEFAULT_OUTPUT_ROOT.iterdir()},
                {"PROTOCOL.json"},
            )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared"
            self.module.prepare(root)
            retained = self.module.validate_prepared(root)
            self.assertEqual(retained["schema"], self.module.PROTOCOL_SCHEMA)
            with self.assertRaises(self.module.Full128SmokeError):
                self.module.prepare(root)

    def test_cpu_fixture_full_scientific_chain_replays(self) -> None:
        value = self.module.validate_cpu_synthetic_fixture(self.fixture, self.payloads)
        self.assertEqual(value["status"], "CPU_SYNTHETIC_FIXTURE_NO_GPU_NO_EXECUTION")
        self.assertTrue(all(item is False for item in value["claims"].values()))
        self.assertEqual(len(value["checkpoint_sha256"]), 21)
        self.assertEqual(len(value["continuation_sha256"]), 21)
        self.assertEqual(len(value["source_endpoint_sha256"]), 21)
        self.assertEqual(len(value["source_segment_sha256"]), 20)
        self.assertEqual(len(value["segment_content_sha256"]), 20)

    def test_fixture_is_relative_source_only_and_not_production_output(self) -> None:
        for receipt in self.fixture["implementation_receipts"]:
            self.assertFalse(Path(receipt["relative_path"]).is_absolute())
            self.assertEqual(len(receipt["sha256"]), 64)
        historical = self.fixture["historical_v3_runtime_data"]
        self.assertTrue(historical["historical_observation_only"])
        self.assertFalse(historical["current_runtime_observation"])
        self.assertEqual(
            historical["engine_projection_sha256"],
            self.module.HISTORICAL_V3_ENGINE_PROJECTION_SHA256,
        )
        self.assertTrue(self.module.FIXTURE_ROSTER.isdisjoint(self.module.EXPECTED_OUTPUT_ROSTER))
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_output_snapshot(self.payloads)

    def test_coordinated_nonholdout_fixture_forgery_rejected(self) -> None:
        document, payloads = self.fresh_fixture()
        checkpoints = bytearray(payloads["FIXTURE_CHECKPOINTS.bin"])
        offset = self.module.STATE_RECORD_BYTES + (self.module.MASSIVE_COUNT + 2) * 48
        old = struct.unpack("<d", checkpoints[offset:offset + 8])[0]
        checkpoints[offset:offset + 8] = struct.pack("<d", math.nextafter(old, math.inf))
        payloads["FIXTURE_CHECKPOINTS.bin"] = bytes(checkpoints)
        changed = bytes(checkpoints[
            self.module.STATE_RECORD_BYTES:2 * self.module.STATE_RECORD_BYTES
        ])
        changed_sha = hashlib.sha256(changed).hexdigest()
        document["checkpoint_sha256"][1] = changed_sha
        document["fixture_files"]["FIXTURE_CHECKPOINTS.bin"] = {
            "bytes": len(checkpoints), "sha256": hashlib.sha256(checkpoints).hexdigest(),
        }
        segments = self.module._IMPL._jsonl(
            payloads["FIXTURE_SEGMENTS.jsonl"], "fixture segments",
        )
        segments[0]["checkpoint_sha256"] = changed_sha
        segments[0]["content_sha256"] = self.module._domain_hash(
            self.module.FIXTURE_SEGMENT_DOMAIN, segments[0],
        )
        payloads["FIXTURE_SEGMENTS.jsonl"] = b"".join(canonical(item) for item in segments)
        document["segment_content_sha256"][0] = segments[0]["content_sha256"]
        document["fixture_files"]["FIXTURE_SEGMENTS.jsonl"] = {
            "bytes": len(payloads["FIXTURE_SEGMENTS.jsonl"]),
            "sha256": hashlib.sha256(payloads["FIXTURE_SEGMENTS.jsonl"]).hexdigest(),
        }
        document["content_sha256"] = self.module._domain_hash(
            self.module.FIXTURE_DOMAIN, document,
        )
        with self.assertRaises(self.module.Full128SmokeError):
            self.module.validate_cpu_synthetic_fixture(document, payloads)

    def test_fixture_claim_escalation_rejected(self) -> None:
        document, payloads = self.fresh_fixture()
        document["claims"]["gpu_executed"] = True
        document["content_sha256"] = self.module._domain_hash(
            self.module.FIXTURE_DOMAIN, document,
        )
        with self.assertRaises(self.module.Full128SmokeError):
            self.module.validate_cpu_synthetic_fixture(document, payloads)

    def test_historical_v3_runtime_rejected_as_current_production_runtime(self) -> None:
        pre = self.module._strict_json(
            self.module._IMPL._read_regular(
                self.module.V3_ROOT / "RUNTIME_PREFLIGHT.json", "historical V3 preflight",
            )
        )
        with self.module._delegates() as (_preflight, v3):
            with v3._delegates() as (_grid, _h64, stage1):
                with self.assertRaises(Exception):
                    v3._validate_runtime_preflight_document(pre, stage1)

    def test_publication_roster_and_result_last_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "publication"
            self.module.prepare(root)
            names = [
                "CHECKPOINTS.bin", "LIFECYCLE.bin", "CONTINUATION.bin",
                "RUNTIME_PRE.json", "RUNTIME_POST.json", "SEGMENTS.jsonl",
                "RESULT.json",
            ]
            artifacts = tuple((name, f"fixture:{name}".encode("ascii")) for name in names)
            self.module._publish(root, artifacts)
            self.assertEqual(set(item.name for item in root.iterdir()), self.module.EXPECTED_OUTPUT_ROSTER)
            self.assertEqual((root / "RESULT.json").read_bytes(), b"fixture:RESULT.json")
            for item in root.iterdir():
                self.assertEqual(item.stat().st_nlink, 1)

    def test_production_core_is_exact_v1_delegate(self) -> None:
        self.assertIs(self.module._IMPL._run_full129_lane, self.module._run_full129_lane)
        self.assertIs(self.module._IMPL._validate_output_snapshot, self.module._validate_output_snapshot)
        self.assertEqual(
            hashlib.sha256(self.module.V1_PATH.read_bytes()).hexdigest(), self.module.V1_SHA256,
        )

    def test_strict_bool_year_alias_rejected_by_production_helper(self) -> None:
        years = list(self.module.CHECKPOINT_YEARS)
        years[0] = False
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._IMPL._validate_checkpoint_years(years)

    @unittest.skipIf(sys.flags.optimize > 0, "normal mode owns opposite-mode byte check")
    def test_optimized_protocol_bytes_identical(self) -> None:
        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-OO", str(SOURCE), "build"],
            cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        self.assertEqual(completed.stdout, canonical(self.module.build_protocol()))


if __name__ == "__main__":
    unittest.main()
