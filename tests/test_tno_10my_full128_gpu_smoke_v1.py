from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v1.py"
HAS_LOCAL_H64_SOURCE = (
    PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v2" / "JX_SOURCE_NODES.bin"
).is_file()


def load_module() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    name = f"_jx_test_full128_smoke_{os.getpid()}_{id(object())}"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(compile(payload, str(SOURCE), "exec", dont_inherit=True, optimize=sys.flags.optimize), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def inverse_sub(desired: float, center: float) -> float:
    candidate = desired + center
    if candidate - center == desired:
        return candidate
    values = [candidate]
    lower = candidate
    upper = candidate
    for _ in range(8):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
        values.extend((lower, upper))
    for value in values:
        if value - center == desired:
            return value
    raise AssertionError("could not invert exact checkpoint subtraction")


def fake_git_custody(module: types.ModuleType, protocol: dict[str, object]) -> dict[str, object]:
    source_payload = SOURCE.read_bytes()
    test_payload = Path(__file__).read_bytes()
    protocol_payload = canonical(protocol)
    records = []
    for relative, payload in zip(
        protocol["preregistration"]["exact_clean_tracked_paths"],
        (source_payload, test_payload, protocol_payload), strict=True,
    ):
        records.append({
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    value = {
        "schema": module.GIT_CUSTODY_SCHEMA,
        "externally_preregistered": True,
        "branch": module.PREREGISTRATION_BRANCH,
        "annotated_tag": module.PREREGISTRATION_TAG,
        "commit": "1" * 40,
        "tag_object": "2" * 40,
        "tag_peeled_commit": "1" * 40,
        "tag_object_type": "tag",
        "clean_tracked_paths": records,
        "network_remote_verification_performed": False,
        "point_observation_only": True,
        "content_sha256": "",
    }
    value["content_sha256"] = module._domain_hash(module.GIT_CUSTODY_DOMAIN, value)
    return value


def build_synthetic_snapshot(module: types.ModuleType) -> dict[str, bytes]:
    protocol = module.build_protocol()
    custody = fake_git_custody(module, protocol)
    with module._delegates() as (preflight, v3):
        v3_pre = v3._strict_json(module._read_regular(module.V3_ROOT / "RUNTIME_PREFLIGHT.json", "V3 preflight"))
        v3_post = v3._strict_json(module._read_regular(module.V3_ROOT / "RUNTIME.json", "V3 runtime"))
        runtime_pre = module._runtime_pre(protocol, custody, v3_pre)
        runtime_post = module._runtime_post(protocol, custody, v3_post, v3)
        source_evidence = module._reconstruct_cpu_source_evidence(preflight, v3)
        v3_result = module._strict_json(module._read_regular(module.V3_ROOT / "RESULT.json", "V3 result"))
        v3_lane = v3_result["batch_lanes"][0]
        v3_segments = module._jsonl(module._read_regular(module.V3_ROOT / "SEGMENTS.jsonl", "V3 segments"), "V3 segments")[:20]
        v3_prefix = module._read_regular(module.V3_ROOT / "PREFIX_STATES.bin", "V3 prefix", 169_344)
        primary = module._batch0_primary_payload(preflight)
        primary_rows = module._unpack_state_record(primary)
        checkpoints: list[bytes] = []
        continuation_records: list[bytes] = []
        lifecycle_rows = [bytes(module.TRACER_COUNT) for _ in module.CHECKPOINT_YEARS]
        source_segment_sha: list[str] = []
        source_endpoint_sha: list[str] = []
        source_nodes: list[bytes] = []
        for node in source_evidence["node_payloads"]:
            source_nodes.append(node)
            source_endpoint_sha.append(hashlib.sha256(node).hexdigest())
        with v3._delegates() as (grid, _h64, _d):
            with grid.FrozenClassifierKernelV1() as kernel:
                for index, node in enumerate(source_nodes):
                    source_positions, source_velocities = module._source_node_rows(node)
                    if index == 0:
                        tracer_positions = tuple(tuple(row[:3]) for row in primary_rows[module.MASSIVE_COUNT:])
                        tracer_velocities = tuple(tuple(row[3:]) for row in primary_rows[module.MASSIVE_COUNT:])
                    else:
                        selected_payload = preflight._v3_checkpoint(v3_prefix, 0, index)
                        selected_values = struct.unpack("<84d", selected_payload)
                        selected_rows = tuple(
                            tuple(selected_values[row * 6:(row + 1) * 6]) for row in range(14)
                        )
                        center_position = tuple(-item for item in selected_rows[0][:3])
                        center_velocity = tuple(-item for item in selected_rows[0][3:])
                        nominal_position = tuple(inverse_sub(selected_rows[5][axis], center_position[axis]) for axis in range(3))
                        nominal_velocity = tuple(inverse_sub(selected_rows[5][3 + axis], center_velocity[axis]) for axis in range(3))
                        tracer_positions_list = [nominal_position for _ in range(module.TRACER_COUNT)]
                        tracer_velocities_list = [nominal_velocity for _ in range(module.TRACER_COUNT)]
                        for selected_order, tracer_index in enumerate(preflight.HOLDOUT_TRACER_INDICES):
                            row = selected_rows[module.MASSIVE_COUNT + selected_order]
                            tracer_positions_list[tracer_index] = tuple(
                                inverse_sub(row[axis], center_position[axis]) for axis in range(3)
                            )
                            tracer_velocities_list[tracer_index] = tuple(
                                inverse_sub(row[3 + axis], center_velocity[axis]) for axis in range(3)
                            )
                        tracer_positions = tuple(tracer_positions_list)
                        tracer_velocities = tuple(tracer_velocities_list)
                    checkpoint = module._checkpoint_full129(
                        preflight, grid, kernel, source_positions, source_velocities,
                        tracer_positions, tracer_velocities,
                    )
                    checkpoints.append(checkpoint)
                    continuation_records.append(
                        module._pack_continuation(tracer_positions, tracer_velocities)
                    )
        continuation = b"".join(continuation_records)
        engine_sha = hashlib.sha256(canonical(v3_post["engine_runtime_projection"])).hexdigest()
        segments = []
        for index in range(module.SEGMENT_COUNT):
            source_sha = hashlib.sha256(source_evidence["segment_payloads"][index]).hexdigest()
            source_segment_sha.append(source_sha)
            segments.append(module._segment_record(
                segment_index=index,
                source_tile_sha256=v3_segments[index]["source_tile_sha256"],
                source_segment_sha256=source_sha,
                source_start_sha256=source_endpoint_sha[index],
                source_end_sha256=source_endpoint_sha[index + 1],
                checkpoint_sha256=hashlib.sha256(checkpoints[index + 1]).hexdigest(),
                lifecycle_sha256=hashlib.sha256(lifecycle_rows[index + 1]).hexdigest(),
                continuation_start_sha256=hashlib.sha256(continuation_records[index]).hexdigest(),
                continuation_end_sha256=hashlib.sha256(continuation_records[index + 1]).hexdigest(),
                minimum_distance=2.0,
                maximum_kepler_iterations=0,
                engine_runtime_identity_sha256=engine_sha,
                elapsed_seconds=0.01,
            ))
        lane = {
            "checkpoints": b"".join(checkpoints),
            "lifecycle": b"".join(lifecycle_rows),
            "continuation": continuation,
            "segments": segments,
            "source_segment_sha256": source_segment_sha,
            "source_endpoint_sha256": source_endpoint_sha,
            "source_full_lane_sha256": v3_lane["retained_source_payload_sha256"],
            "source_full_tile_sha256": v3_lane["full_source_tile_sha256"],
            "source_build_identity": module._retained_source_build_identity(v3_lane),
            "source_invariants": v3_lane["source_invariants"],
            "elapsed_seconds": 1.0,
        }
        result, artifacts = module._build_result_and_artifacts(
            protocol, custody, runtime_pre, runtime_post, lane, preflight,
        )
    snapshot = {"PROTOCOL.json": canonical(protocol), **dict(artifacts)}
    if canonical(result) != snapshot["RESULT.json"]:
        raise AssertionError("synthetic result bytes differ")
    return snapshot


class Full128SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.snapshot = build_synthetic_snapshot(cls.module)

    def fresh(self) -> dict[str, bytes]:
        return dict(self.snapshot)

    def rewrite_result(self, snapshot: dict[str, bytes], transform) -> None:
        result = self.module._strict_json(snapshot["RESULT.json"])
        transform(result)
        result["content_sha256"] = self.module._domain_hash(self.module.RESULT_DOMAIN, result)
        snapshot["RESULT.json"] = canonical(result)

    def test_protocol_exact_geometry_and_claim_ceiling(self) -> None:
        protocol = self.module.build_protocol()
        self.assertEqual(protocol["execution"]["tracer_count"], 129)
        self.assertEqual(protocol["execution"]["segment_count"], 20)
        self.assertEqual(protocol["outputs"]["checkpoints_bytes"], 135_072)
        self.assertEqual(protocol["outputs"]["lifecycle_bytes"], 2_709)
        self.assertEqual(protocol["outputs"]["continuation_bytes"], 130_032)
        self.assertEqual(protocol["outputs"]["continuation_record_count"], 21)
        self.assertEqual(protocol["outputs"]["continuation_record_bytes"], 6_192)
        self.assertEqual(
            protocol["bound_inputs"]["full128_bridge_states"]["sha256"],
            self.module.FULL_BRIDGE_STATES_SHA256,
        )
        self.assertEqual(
            protocol["bound_inputs"]["full128_trajectory_grid_protocol"]["sha256"],
            self.module.TRAJECTORY_GRID_PROTOCOL_SHA256,
        )
        self.assertEqual(
            protocol["execution"]["initial_state_authority_content_sha256"],
            self.module.FULL_BRIDGE_STATES_CONTENT_SHA256,
        )
        self.assertFalse(protocol["claims"]["ten_myr_execution_authorized"])
        self.assertEqual(protocol["implementation"]["local_threat_boundary"], self.module.LOCAL_THREAT_BOUNDARY)
        self.assertTrue(
            protocol["interpretation"][
                "clones_are_local_formal_covariance_sensitivity_draws_not_posterior_or_likelihood_samples"
            ]
        )
        large = protocol["bound_inputs"]["required_large_runtime_artifact"]
        self.assertEqual(large["relative_path"], self.module.H64_SOURCE_RELATIVE_PATH)
        self.assertEqual(large["bytes"], 368_645_800)
        self.assertEqual(large["sha256"], self.module.H64_SOURCE_SHA256)
        self.assertEqual(large["exact_link_count"], 1)
        self.assertFalse(large["availability_checked_by_prepare"])
        self.assertTrue(large["availability_and_full_hash_checked_before_gpu_lane"])
        self.assertTrue(protocol["acceptance"]["source_tile_identity_reuses_exact_v3_batch0_labels"])
        self.assertTrue(
            protocol["acceptance"]["this_smoke_validator_is_sole_authoritative_full129_smoke_validator"]
        )
        self.assertTrue(
            protocol["acceptance"]["preflight_projection_receipt_is_secondary_non_authoritative_evidence"]
        )
        self.module.validate_protocol(protocol)

    def test_cpu_source_tile_identities_match_retained_v3(self) -> None:
        with self.module._delegates() as (preflight, v3):
            source = self.module._reconstruct_cpu_source_evidence(preflight, v3)
        retained = [
            json.loads(line)["source_tile_sha256"]
            for line in (self.module.V3_ROOT / "SEGMENTS.jsonl").read_text().splitlines()[:20]
        ]
        self.assertEqual(list(source["source_tile_sha256"]), retained)

    def test_full129_initial_payload_exact(self) -> None:
        with self.module._delegates() as (preflight, _v3):
            payload = self.module._batch0_primary_payload(preflight)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.module.YEAR_ZERO_PRIMARY_SHA256)
        self.assertEqual(
            hashlib.sha256(self.snapshot["CHECKPOINTS.bin"][:self.module.STATE_RECORD_BYTES]).hexdigest(),
            self.module.YEAR_ZERO_CHECKPOINT_SHA256,
        )

    def test_prepare_validate_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared"
            self.module.prepare(root)
            self.module.validate_prepared(root)
            with self.assertRaises(self.module.Full128SmokeError):
                self.module.prepare(root)

    def test_prepared_extra_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared"
            self.module.prepare(root)
            (root / "EXTRA").write_bytes(b"x")
            with self.assertRaises(self.module.Full128SmokeError):
                self.module.validate_prepared(root)

    @unittest.skipUnless(HAS_LOCAL_H64_SOURCE, "exact V3 runtime replay requires the separately retained 369 MB source file")
    def test_exact_synthetic_output_replays(self) -> None:
        result = self.module._validate_output_snapshot(self.fresh())
        self.assertTrue(result["all_predeclared_gates_passed"])
        self.assertTrue(result["claims"]["bounded_gpu_smoke_passed"])
        self.assertFalse(result["claims"]["ten_myr_execution_authorized"])

    @unittest.skipUnless(HAS_LOCAL_H64_SOURCE, "exact V3 runtime replay requires the separately retained 369 MB source file")
    def test_retained_root_validate_and_roster(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "result"
            root.mkdir()
            for name, payload in self.snapshot.items():
                (root / name).write_bytes(payload)
            self.module.validate_output(root)
            (root / "EXTRA").write_bytes(b"")
            with self.assertRaises(self.module.Full128SmokeError):
                self.module.validate_output(root)

    def test_nonholdout_nonfinite_checkpoint_rejected(self) -> None:
        snapshot = self.fresh()
        payload = bytearray(snapshot["CHECKPOINTS.bin"])
        offset = self.module.STATE_RECORD_BYTES + (self.module.MASSIVE_COUNT + 2) * 48
        payload[offset:offset + 8] = struct.pack("<d", float("nan"))
        snapshot["CHECKPOINTS.bin"] = bytes(payload)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_output_snapshot(snapshot)

    def test_nonholdout_finite_checkpoint_forgery_rejected_by_chain(self) -> None:
        snapshot = self.fresh()
        payload = bytearray(snapshot["CHECKPOINTS.bin"])
        offset = self.module.STATE_RECORD_BYTES + (self.module.MASSIVE_COUNT + 2) * 48
        payload[offset] ^= 1
        snapshot["CHECKPOINTS.bin"] = bytes(payload)
        records = self.module._jsonl(snapshot["SEGMENTS.jsonl"], "segments")
        record = records[0]
        checkpoint = snapshot["CHECKPOINTS.bin"][self.module.STATE_RECORD_BYTES:2 * self.module.STATE_RECORD_BYTES]
        lifecycle = snapshot["LIFECYCLE.bin"][self.module.TRACER_COUNT:2 * self.module.TRACER_COUNT]
        continuations = self.module._continuation_records(snapshot["CONTINUATION.bin"])
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_segment_record(
                record, segment_index=0, checkpoint=checkpoint, lifecycle=lifecycle,
                continuation_start=continuations[0], continuation_end=continuations[1],
                retained_segment_sha256=record["retained_source_segment_sha256"],
                source_start_sha256=record["source_start_node_sha256"],
                source_end_sha256=record["source_end_node_sha256"],
                v3_source_tile_sha256=record["source_tile_sha256"],
                engine_projection_sha256=record["engine_runtime_identity_sha256"],
            )

    def test_coordinated_nonholdout_checkpoint_forgery_rejected_by_continuation(self) -> None:
        snapshot = self.fresh()
        payload = bytearray(snapshot["CHECKPOINTS.bin"])
        offset = self.module.STATE_RECORD_BYTES + (self.module.MASSIVE_COUNT + 2) * 48
        original = struct.unpack("<d", payload[offset:offset + 8])[0]
        payload[offset:offset + 8] = struct.pack("<d", math.nextafter(original, math.inf))
        snapshot["CHECKPOINTS.bin"] = bytes(payload)
        forged_checkpoint = snapshot["CHECKPOINTS.bin"][
            self.module.STATE_RECORD_BYTES:2 * self.module.STATE_RECORD_BYTES
        ]
        forged_checkpoint_sha = hashlib.sha256(forged_checkpoint).hexdigest()

        segments = self.module._jsonl(snapshot["SEGMENTS.jsonl"], "segments")
        segments[0]["checkpoint_sha256"] = forged_checkpoint_sha
        segments[0]["content_sha256"] = self.module._domain_hash(
            self.module.SEGMENT_DOMAIN, segments[0],
        )
        snapshot["SEGMENTS.jsonl"] = b"".join(canonical(record) for record in segments)

        result = self.module._strict_json(snapshot["RESULT.json"])
        result["checkpoint_sha256"][1] = forged_checkpoint_sha
        for name in ("CHECKPOINTS.bin", "SEGMENTS.jsonl"):
            result["files"][name] = {
                "bytes": len(snapshot[name]),
                "sha256": hashlib.sha256(snapshot[name]).hexdigest(),
            }
        with self.module._delegates() as (preflight, v3):
            result["secondary_preflight_evidence"] = preflight.validate_smoke_evidence_v1(
                snapshot["CHECKPOINTS.bin"],
                active_counts=[self.module.TRACER_COUNT] * self.module.CHECKPOINT_COUNT,
                routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=1.0,
            )
            result["content_sha256"] = self.module._domain_hash(self.module.RESULT_DOMAIN, result)
            snapshot["RESULT.json"] = canonical(result)

            continuations = self.module._continuation_records(snapshot["CONTINUATION.bin"])
            self.module._validate_segment_record(
                segments[0], segment_index=0, checkpoint=forged_checkpoint,
                lifecycle=snapshot["LIFECYCLE.bin"][self.module.TRACER_COUNT:2 * self.module.TRACER_COUNT],
                continuation_start=continuations[0], continuation_end=continuations[1],
                retained_segment_sha256=segments[0]["retained_source_segment_sha256"],
                source_start_sha256=segments[0]["source_start_node_sha256"],
                source_end_sha256=segments[0]["source_end_node_sha256"],
                v3_source_tile_sha256=segments[0]["source_tile_sha256"],
                engine_projection_sha256=segments[0]["engine_runtime_identity_sha256"],
            )
            source = self.module._reconstruct_cpu_source_evidence(preflight, v3)
            checkpoints = tuple(
                snapshot["CHECKPOINTS.bin"][index * self.module.STATE_RECORD_BYTES:(index + 1) * self.module.STATE_RECORD_BYTES]
                for index in range(self.module.CHECKPOINT_COUNT)
            )
            with self.assertRaisesRegex(self.module.Full128SmokeError, "checkpoint 1"):
                self.module._validate_continuation_records(
                    preflight, v3, source["node_payloads"],
                    snapshot["CONTINUATION.bin"], checkpoints,
                )

    def test_checkpoint_year_bool_alias_rejected(self) -> None:
        snapshot = self.fresh()
        result = self.module._strict_json(snapshot["RESULT.json"])
        result["checkpoint_years"][0] = False
        result["content_sha256"] = self.module._domain_hash(self.module.RESULT_DOMAIN, result)
        snapshot["RESULT.json"] = canonical(result)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_output_snapshot(snapshot)

    def test_lifecycle_tamper_rejected(self) -> None:
        snapshot = self.fresh()
        payload = bytearray(snapshot["LIFECYCLE.bin"])
        payload[-1] = 1
        snapshot["LIFECYCLE.bin"] = bytes(payload)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_output_snapshot(snapshot)

    def test_continuation_tamper_rejected(self) -> None:
        snapshot = self.fresh()
        payload = bytearray(snapshot["CONTINUATION.bin"])
        payload[-1] ^= 1
        with self.module._delegates() as (preflight, v3):
            source = self.module._reconstruct_cpu_source_evidence(preflight, v3)
            checkpoints = tuple(
                snapshot["CHECKPOINTS.bin"][index * self.module.STATE_RECORD_BYTES:(index + 1) * self.module.STATE_RECORD_BYTES]
                for index in range(self.module.CHECKPOINT_COUNT)
            )
            with self.assertRaises(self.module.Full128SmokeError):
                self.module._validate_continuation_records(
                    preflight, v3, source["node_payloads"], bytes(payload), checkpoints,
                )

    def test_continuation_is_exact_engine_valid_state(self) -> None:
        snapshot = self.fresh()
        with self.module._delegates() as (preflight, v3):
            source = self.module._reconstruct_cpu_source_evidence(preflight, v3)
            checkpoints = tuple(
                snapshot["CHECKPOINTS.bin"][index * self.module.STATE_RECORD_BYTES:(index + 1) * self.module.STATE_RECORD_BYTES]
                for index in range(self.module.CHECKPOINT_COUNT)
            )
            self.module._validate_continuation_records(
                preflight, v3, source["node_payloads"],
                snapshot["CONTINUATION.bin"], checkpoints,
            )

    def test_segment_source_relabel_rejected(self) -> None:
        snapshot = self.fresh()
        records = self.module._jsonl(snapshot["SEGMENTS.jsonl"], "segments")
        expected = copy.deepcopy(records[0])
        records[0]["source_end_node_sha256"] = records[1]["source_end_node_sha256"]
        records[0]["content_sha256"] = self.module._domain_hash(self.module.SEGMENT_DOMAIN, records[0])
        continuations = self.module._continuation_records(snapshot["CONTINUATION.bin"])
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_segment_record(
                records[0], segment_index=0,
                checkpoint=snapshot["CHECKPOINTS.bin"][self.module.STATE_RECORD_BYTES:2 * self.module.STATE_RECORD_BYTES],
                lifecycle=snapshot["LIFECYCLE.bin"][self.module.TRACER_COUNT:2 * self.module.TRACER_COUNT],
                continuation_start=continuations[0], continuation_end=continuations[1],
                retained_segment_sha256=expected["retained_source_segment_sha256"],
                source_start_sha256=expected["source_start_node_sha256"],
                source_end_sha256=expected["source_end_node_sha256"],
                v3_source_tile_sha256=expected["source_tile_sha256"],
                engine_projection_sha256=expected["engine_runtime_identity_sha256"],
            )

    def test_runtime_preregistration_relabel_rejected(self) -> None:
        snapshot = self.fresh()
        pre = self.module._strict_json(snapshot["RUNTIME_PRE.json"])
        pre["git_preregistration"]["branch"] = "wrong"
        pre["git_preregistration"]["content_sha256"] = self.module._domain_hash(
            self.module.GIT_CUSTODY_DOMAIN, pre["git_preregistration"],
        )
        pre["content_sha256"] = self.module._domain_hash(self.module.RUNTIME_PRE_DOMAIN, pre)
        snapshot["RUNTIME_PRE.json"] = canonical(pre)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_output_snapshot(snapshot)

    def test_result_claim_escalation_rejected(self) -> None:
        result = self.module._strict_json(self.snapshot["RESULT.json"])
        result["claims"]["ten_myr_execution_authorized"] = True
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_exact_claims(result["claims"], self.module._result_claims())

    def test_result_file_receipt_forgery_rejected(self) -> None:
        snapshot = self.fresh()
        result = self.module._strict_json(snapshot["RESULT.json"])
        result["files"]["CHECKPOINTS.bin"]["sha256"] = "0" * 64
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_file_receipts(result["files"], snapshot)

    def test_strict_json_duplicate_and_nan_rejected(self) -> None:
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._strict_json(b'{"a":1,"a":2}\n')
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._strict_json(b'{"a":NaN}\n')

    def test_execute_requires_explicit_preregistration_before_git(self) -> None:
        with self.assertRaisesRegex(self.module.Full128SmokeError, "external preregistration"):
            self.module._git_custody(
                externally_preregistered=False, prereg_commit="", prereg_tag_object="",
            )

    def test_offline_git_custody_rejects_nonhex_and_wrong_length_receipt(self) -> None:
        protocol = self.module.build_protocol()
        custody = fake_git_custody(self.module, protocol)
        custody["commit"] = "g" * 40
        custody["tag_peeled_commit"] = "g" * 40
        custody["content_sha256"] = self.module._domain_hash(self.module.GIT_CUSTODY_DOMAIN, custody)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_git_custody_offline(custody, protocol)
        custody = fake_git_custody(self.module, protocol)
        custody["clean_tracked_paths"][0]["bytes"] += 1
        custody["content_sha256"] = self.module._domain_hash(self.module.GIT_CUSTODY_DOMAIN, custody)
        with self.assertRaises(self.module.Full128SmokeError):
            self.module._validate_git_custody_offline(custody, protocol)

    @unittest.skipIf(sys.flags.optimize > 0, "normal mode owns the opposite-mode byte comparison")
    def test_optimized_protocol_bytes_identical(self) -> None:
        completed = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-OO", str(SOURCE), "build"],
            cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        self.assertEqual(completed.stdout, canonical(self.module.build_protocol()))


if __name__ == "__main__":
    unittest.main()
