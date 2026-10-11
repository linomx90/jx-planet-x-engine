from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "benchmarks" / "jx_tno_full128_reduced5_bridge_v1.py"


def _load_module():
    name = "jx_tno_full128_reduced5_bridge_v1_test_target"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("bridge source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_module()


class Full128Reduced5BridgeV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="jx-full128-bridge-test-")
        cls.parent = Path(cls._temporary.name)
        cls.root = cls.parent / "artifact"
        cls.artifact = bridge.prepare(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def clone_artifact(self, name: str) -> Path:
        target = self.parent / name
        shutil.copytree(self.root, target)
        return target

    def test_01_build_has_exact_full_geometry_and_claim_ceiling(self) -> None:
        protocol, artifact = bridge.build_documents()
        self.assertEqual(protocol["schema"], bridge.PROTOCOL_SCHEMA)
        self.assertEqual(artifact["schema"], bridge.BRIDGE_SCHEMA)
        self.assertEqual(artifact["batch_count"], 12)
        self.assertEqual(artifact["massive_state_rows_total"], 60)
        self.assertEqual(artifact["tracer_state_rows_total"], 1_548)
        self.assertEqual(artifact["tracer_count_per_batch"], 129)
        self.assertFalse(artifact["execution_authorized"])
        self.assertFalse(artifact["outcomes_generated"])
        self.assertTrue(artifact["claims"]["artifact_prepared"])
        for name, value in artifact["claims"].items():
            if name != "artifact_prepared":
                self.assertIs(value, False, name)
        self.assertFalse(protocol["claims"]["artifact_prepared"])
        self.assertEqual(
            artifact["protocol_content_sha256"], protocol["content_sha256"]
        )

    def test_02_every_batch_is_nominal_then_clone_0_through_127(self) -> None:
        for order, batch in enumerate(self.artifact["batches"]):
            self.assertEqual(batch["batch_order"], order)
            self.assertEqual(batch["object_id"], bridge.OBJECT_IDS[order])
            self.assertEqual(len(batch["massive_bodies"]), 5)
            self.assertEqual(len(batch["tracers"]), 129)
            self.assertEqual(
                [row["naif_id"] for row in batch["massive_bodies"]],
                list(bridge.REDUCED5_NAIF_IDS),
            )
            tracers = batch["tracers"]
            self.assertIsNone(tracers[0]["clone_index"])
            self.assertEqual(tracers[0]["sample_kind"], "NOMINAL")
            self.assertEqual(
                [row["clone_index"] for row in tracers[1:]], list(range(128))
            )
            self.assertTrue(all(
                row["sample_kind"] == "COVARIANCE_CLONE" for row in tracers[1:]
            ))
            self.assertTrue(all(
                row["lifecycle_status"] == "ACTIVE_AT_NATIVE_EPOCH_NOT_PROPAGATED"
                for row in tracers
            ))

    def test_03_selected_projection_is_exact_and_covers_required_lanes(self) -> None:
        self.assertTrue(self.artifact["all_twelve_selected9_projections_bitwise_equal"])
        global_payload = bytearray()
        for batch in self.artifact["batches"]:
            receipt = batch["selected9_projection"]
            self.assertIs(receipt["heliocentric_and_reduced5_bitwise_equal"], True)
            self.assertEqual(
                receipt["full_tracer_indices"], list(bridge.PROJECTED_TRACER_INDICES)
            )
            self.assertEqual(
                [row["clone_index"] for row in receipt["rows"]],
                [None, *bridge.HOLDOUT_CLONE_INDICES],
            )
            projected = [
                batch["tracers"][index] for index in bridge.PROJECTED_TRACER_INDICES
            ]
            payload = bridge._projection_payload(projected)
            global_payload.extend(payload)
            self.assertEqual(receipt["binary_payload_byte_length"], len(payload))
            self.assertEqual(
                receipt["binary_payload_sha256"],
                hashlib.sha256(bridge.PROJECTION_DOMAIN + payload).hexdigest(),
            )
        self.assertEqual(
            self.artifact["selected9_projection_binary_payload_sha256"],
            hashlib.sha256(bridge.PROJECTION_DOMAIN + bytes(global_payload)).hexdigest(),
        )

    def test_04_batch_and_global_binary_receipts_reconstruct(self) -> None:
        global_payload = bytearray()
        global_direct = bytearray()
        global_primary = bytearray()
        global_checkpoint = bytearray()
        v3_prefix = bridge._input_bundle()["v3_prefix"]
        for batch in self.artifact["batches"]:
            payload = bridge._binary_payload(batch["massive_bodies"], batch["tracers"])
            global_payload.extend(payload)
            self.assertEqual(batch["binary_payload_byte_length"], len(payload))
            self.assertEqual(
                batch["binary_payload_sha256"],
                hashlib.sha256(bridge.BINARY_DOMAIN + payload).hexdigest(),
            )
            self.assertEqual(
                batch["content_sha256"],
                bridge._content_sha256(bridge.BATCH_DOMAIN, batch),
            )
            direct, primary, checkpoint = bridge._consumer_payloads(
                batch["massive_bodies"], batch["tracers"]
            )
            global_direct.extend(direct)
            global_primary.extend(primary)
            global_checkpoint.extend(checkpoint)
            receipt = batch["consumer_state_payloads"]
            self.assertEqual(len(direct), 6_432)
            self.assertEqual(len(primary), 6_432)
            self.assertEqual(len(checkpoint), 6_432)
            self.assertEqual(
                receipt["direct_reduced5_barycentric"]["sha256"],
                hashlib.sha256(direct).hexdigest(),
            )
            self.assertEqual(
                receipt["engine_primary_relative"]["sha256"],
                hashlib.sha256(primary).hexdigest(),
            )
            self.assertEqual(
                receipt["engine_checkpoint_equivalent"]["sha256"],
                hashlib.sha256(checkpoint).hexdigest(),
            )
            self.assertEqual(
                bridge._selected9_checkpoint_payload(checkpoint),
                bridge._v3_year_zero_checkpoint(
                    v3_prefix, batch["batch_order"]
                ),
            )
        self.assertEqual(self.artifact["binary_payload_byte_length"], len(global_payload))
        self.assertEqual(
            self.artifact["binary_payload_sha256"],
            hashlib.sha256(bridge.BINARY_DOMAIN + bytes(global_payload)).hexdigest(),
        )
        aggregates = self.artifact["consumer_state_payload_aggregates"]
        self.assertEqual(aggregates["direct_reduced5_barycentric"]["sha256"], hashlib.sha256(bytes(global_direct)).hexdigest())
        self.assertEqual(aggregates["engine_primary_relative"]["sha256"], hashlib.sha256(bytes(global_primary)).hexdigest())
        self.assertEqual(aggregates["engine_checkpoint_equivalent"]["sha256"], hashlib.sha256(bytes(global_checkpoint)).hexdigest())
        self.assertEqual(aggregates["batch0_direct_reduced5_barycentric_sha256"], bridge.EXPECTED_BATCH0_DIRECT_BARYCENTRIC_SHA256)
        self.assertEqual(aggregates["batch0_engine_checkpoint_equivalent_sha256"], bridge.EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256)
        self.assertEqual(bridge.EXPECTED_BATCH0_ENGINE_CHECKPOINT_SHA256, "25394a0f2a31caf3c9b47f593ec0511806b94a0ac140d1d44fc9e1939656e00c")

    def test_05_consumer_operation_order_drift_and_relabel_fail_closed(self) -> None:
        bundle = bridge._input_bundle()
        protocol = bridge.build_protocol(bundle)
        original = bridge._consumer_payloads

        def drifted(massive, tracers):
            direct, primary, checkpoint = original(massive, tracers)
            changed = bytearray(checkpoint)
            value = struct.unpack_from("<d", changed, 0)[0]
            struct.pack_into("<d", changed, 0, math.nextafter(value, math.inf))
            return direct, primary, bytes(changed)

        with mock.patch.object(bridge, "_consumer_payloads", side_effect=drifted):
            with self.assertRaisesRegex(bridge.FullBridgeError, "checkpoint selected9"):
                bridge.build_bridge(protocol, bundle)

        def relabelled(massive, tracers):
            direct, primary, _checkpoint = original(massive, tracers)
            return direct, primary, direct

        with mock.patch.object(bridge, "_consumer_payloads", side_effect=relabelled):
            with self.assertRaisesRegex(bridge.FullBridgeError, "relabelled or collapsed"):
                bridge.build_bridge(protocol, bundle)

    def test_05_v3_lineage_is_full_and_does_not_authorize_10myr(self) -> None:
        lineage = self.artifact["v3_lineage"]
        self.assertEqual(len(lineage["exact_root_roster"]), 8)
        self.assertEqual(
            lineage["protocol_content_sha256"], bridge.V3_PROTOCOL_CONTENT_SHA256
        )
        self.assertEqual(
            lineage["result_content_sha256"], bridge.V3_RESULT_CONTENT_SHA256
        )
        self.assertEqual(lineage["git_preregistration"], bridge.V3_GIT)
        self.assertIn("NO_10MYR_AUTHORIZATION", lineage["interpretation"])
        self.assertFalse(self.artifact["claims"]["ten_myr_execution_authorized"])
        self.assertFalse(self.artifact["claims"]["classification_generated"])
        self.assertFalse(self.artifact["claims"]["science_result"])
        protocol, _artifact = bridge.build_documents()
        self.assertEqual(
            protocol["local_threat_boundary"],
            "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        )

    def test_06_prepare_publishes_only_canonical_protocol_then_bridge(self) -> None:
        self.assertEqual({item.name for item in self.root.iterdir()}, bridge.EXPECTED_OUTPUT_ROSTER)
        for name in bridge.EXPECTED_OUTPUT_ROSTER:
            path = self.root / name
            observed = path.stat()
            self.assertTrue(path.is_file())
            self.assertEqual(observed.st_nlink, 1)
            payload = path.read_bytes()
            value = bridge._strict_json(payload)
            self.assertEqual(payload, bridge._canonical_json(value))
        self.assertEqual(bridge.validate_output(self.root), self.artifact)
        with self.assertRaises(bridge.FullBridgeError):
            bridge.prepare(self.root)

    def test_07_exact_type_and_nonfinite_boundaries_fail_closed(self) -> None:
        for value in (True, 1, 1.0, None, "nan", "inf", "0X1.0P+0", "0x1p+0"):
            with self.subTest(value=value):
                with self.assertRaises(bridge.FullBridgeError):
                    bridge._hex_float(value, "probe")
        with self.assertRaises(bridge.FullBridgeError):
            bridge._vector(("0x0.0p+0",) * 3, "tuple probe")
        with self.assertRaises(bridge.FullBridgeError):
            bridge._strict_json(b'{"a":1,"a":2}\n')
        with self.assertRaises(bridge.FullBridgeError):
            bridge._strict_json(b'{"a":NaN}\n')
        with self.assertRaises(bridge.FullBridgeError):
            bridge._weighted_centroid((True,), ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),))
        self.assertEqual(bridge._hex_float("-0x0.0p+0", "signed zero").hex(), "-0x0.0p+0")

    def test_08_protocol_or_bridge_tampering_fails_exact_reconstruction(self) -> None:
        root = self.clone_artifact("tampered")
        value = json.loads((root / "BRIDGE.json").read_text(encoding="ascii"))
        value["claims"]["classification_generated"] = True
        value["content_sha256"] = bridge._content_sha256(bridge.BRIDGE_DOMAIN, value)
        (root / "BRIDGE.json").write_bytes(bridge._canonical_json(value))
        with self.assertRaises(bridge.FullBridgeError):
            bridge.validate_output(root)

    def test_09_extra_missing_symlink_and_hardlink_outputs_fail_closed(self) -> None:
        extra = self.clone_artifact("extra")
        (extra / "EXTRA").write_bytes(b"")
        with self.assertRaises(bridge.FullBridgeError):
            bridge.validate_output(extra)

        missing = self.clone_artifact("missing")
        (missing / "BRIDGE.json").unlink()
        with self.assertRaises(bridge.FullBridgeError):
            bridge.validate_output(missing)

        linked = self.clone_artifact("linked")
        os.link(linked / "PROTOCOL.json", linked / "PROTOCOL.link")
        (linked / "PROTOCOL.link").unlink()
        # The link count returns to one; make the second link outside the root so
        # the exact root roster remains plausible while held-file validation sees it.
        outside = self.parent / "outside-protocol-link"
        os.link(linked / "PROTOCOL.json", outside)
        try:
            with self.assertRaises(bridge.FullBridgeError):
                bridge.validate_output(linked)
        finally:
            outside.unlink()

        symlink = self.parent / "root-symlink"
        symlink.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(bridge.FullBridgeError):
            bridge.validate_output(symlink)

    def test_10_held_validation_detects_mid_validation_replacement(self) -> None:
        root = self.clone_artifact("toctou")
        original = bridge._read_output_file
        count = 0

        def replacing(root_fd: int, name: str):
            nonlocal count
            count += 1
            if count == 3:
                payload = (root / "PROTOCOL.json").read_bytes()
                (root / "PROTOCOL.json").write_bytes(payload)
            return original(root_fd, name)

        with mock.patch.object(bridge, "_read_output_file", side_effect=replacing):
            with self.assertRaises(bridge.FullBridgeError):
                bridge.validate_output(root)

    def test_11_input_byte_tampering_is_rejected_before_use(self) -> None:
        original = bridge._read_project
        target = bridge.STAGE0_BINDINGS[-1][1]

        def altered(relative: str) -> bytes:
            payload = original(relative)
            return payload + b" " if relative == target else payload

        with mock.patch.object(bridge, "_read_project", side_effect=altered):
            with self.assertRaises(bridge.FullBridgeError):
                bridge.build_documents()

    def test_12_wrong_protocol_and_relabelled_projection_fail_closed(self) -> None:
        bundle = bridge._input_bundle()
        protocol = bridge.build_protocol(bundle)
        altered = json.loads(json.dumps(protocol))
        altered["geometry"]["tracer_count_per_batch"] = 9
        altered["content_sha256"] = bridge._content_sha256(bridge.PROTOCOL_DOMAIN, altered)
        with self.assertRaises(bridge.FullBridgeError):
            bridge.build_bridge(altered, bundle)

        legacy = bundle["legacy_batches"][0]["tnos"]
        first = bundle["groups"][0][0]
        final = first["native_vector_sha256"][-1]
        legacy[0]["native_vector_sha256"] = first["native_vector_sha256"][:-1] + ("0" if final != "0" else "1")
        with self.assertRaises(bridge.FullBridgeError):
            bridge.build_bridge(protocol, bundle)

        bundle = bridge._input_bundle()
        protocol = bridge.build_protocol(bundle)
        bundle["groups"][0][1]["clone_index"] = True
        with self.assertRaises(bridge.FullBridgeError):
            bridge.build_bridge(protocol, bundle)

    def test_13_source_only_import_has_no_dynamics_or_array_runtime(self) -> None:
        code = (
            "import runpy,sys; before=set(sys.modules); "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='bridge_source_only'); "
            "bad=sorted(name for name in set(sys.modules)-before if "
            "name.split('.')[0] in {'numpy','cupy','rebound','jxplanetx'}); "
            "print('\\n'.join(bad)); raise SystemExit(bool(bad))"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stdout, b"\n")

    def test_14_build_is_identical_under_normal_and_optimized_python(self) -> None:
        command = [str(SCRIPT), "build"]
        normal = subprocess.run(
            [sys.executable, "-B", *command], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        optimized = subprocess.run(
            [sys.executable, "-B", "-OO", *command], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(normal.returncode, 0, normal.stderr.decode())
        self.assertEqual(optimized.returncode, 0, optimized.stderr.decode())
        self.assertEqual(normal.stdout, optimized.stdout)
        summary = json.loads(normal.stdout)
        self.assertEqual(summary["tracer_state_rows_total"], 1_548)
        self.assertFalse(summary["execution_authorized"])

    def test_15_validate_cli_is_identical_under_normal_and_optimized_python(self) -> None:
        command = [str(SCRIPT), "validate", "--output-root", str(self.root)]
        normal = subprocess.run(
            [sys.executable, "-B", *command], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        optimized = subprocess.run(
            [sys.executable, "-B", "-OO", *command], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(normal.returncode, 0, normal.stderr.decode())
        self.assertEqual(optimized.returncode, 0, optimized.stderr.decode())
        self.assertEqual(normal.stdout, optimized.stdout)


if __name__ == "__main__":
    unittest.main()
