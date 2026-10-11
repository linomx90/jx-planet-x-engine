from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_1kyr_full129_restart_qualification_v1.py"


def load_module() -> types.ModuleType:
    name = f"_test_tno_full129_restart_v1_{os.getpid()}_{id(object())}_{sys.flags.optimize}"
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
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


m = load_module()


class FakeArray:
    def __init__(self, value: object) -> None:
        import numpy as np

        self.value = np.asarray(value)


class FakeCP:
    import numpy as _np

    uint64 = _np.uint64

    @staticmethod
    def asarray(value: object) -> FakeArray:
        return FakeArray(value)

    @staticmethod
    def arange(count: int, dtype: object) -> FakeArray:
        import numpy as np

        return FakeArray(np.arange(count, dtype=dtype))

    @staticmethod
    def asnumpy(value: object) -> object:
        return value.value if type(value) is FakeArray else value


@dataclass(frozen=True)
class FakeState:
    snapshot_id: str
    global_source_step: int
    tracer_relative_positions_AU: object
    tracer_relative_velocities_AU_per_year: object
    tracer_indices: object
    sealed_metadata: str


def synthetic_v3_segment(index: int = 10) -> dict[str, object]:
    projection = {
        "active_tracer_count": m.TRACER_COUNT,
        "batch_order": m.BATCH_ORDER,
        "checkpoint_sha256": "a" * 64,
        "continuation_end_sha256": "b" * 64,
        "continuation_start_sha256": "c" * 64,
        "end_global_source_step": (index + 1) * m.SEGMENT_STEPS,
        "end_year": (index + 1) * m.SEGMENT_YEARS,
        "engine_runtime_identity_sha256": m.REFERENCE_ENGINE_PROJECTION_SHA256,
        "generated_source_segment_sha256": "d" * 64,
        "lifecycle_sha256": "e" * 64,
        "maximum_kepler_iterations": 6,
        "minimum_source_distance_AU_hex": float(240.0).hex(),
        "object_id": m.OBJECT_ID,
        "removal_record_count": 0,
        "retained_source_segment_sha256": "d" * 64,
        "route_record_count": 0,
        "source_end_node_sha256": "f" * 64,
        "source_start_node_sha256": "1" * 64,
        "source_tile_sha256": "2" * 64,
        "start_global_source_step": index * m.SEGMENT_STEPS,
        "start_year": index * m.SEGMENT_YEARS,
    }
    return {
        "schema": "jx.tno.10my-full128-gpu-smoke.segment.v3",
        "segment_order": index,
        "segment_index": index,
        **projection,
        "elapsed_seconds_hex": float(0.2).hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE",
        "content_sha256": "3" * 64,
    }


class RestartQualificationTests(unittest.TestCase):
    def test_protocol_exact_predecessor_geometry_and_false_claim_ceiling(self) -> None:
        before = set(sys.modules)
        protocol = m.build_protocol()
        self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
        self.assertEqual(protocol["preregistration"]["branch"], m.PREREGISTRATION_BRANCH)
        self.assertEqual(protocol["preregistration"]["annotated_tag"], m.PREREGISTRATION_TAG)
        self.assertEqual(protocol["preregistration"]["future_result_tag"], m.FUTURE_RESULT_TAG)
        lineage = protocol["predecessor_v3"]["git_lineage"]
        self.assertEqual(lineage["preregistration"]["peeled_commit"], m.V3_PREREG_COMMIT)
        self.assertEqual(lineage["preregistration"]["tag_object"], m.V3_PREREG_TAG_OBJECT)
        self.assertEqual(lineage["result"]["peeled_commit"], m.V3_RESULT_COMMIT)
        self.assertEqual(lineage["result"]["tag_object"], m.V3_RESULT_TAG_OBJECT)
        self.assertEqual(protocol["restart"]["tracer_indices_exact"], list(range(129)))
        self.assertEqual(protocol["restart"]["checkpoint_years"], list(range(500, 1001, 50)))
        self.assertFalse(protocol["claim_ceiling"]["general_disk_checkpoint_restart_qualified"])
        self.assertFalse(protocol["claim_ceiling"]["ten_myr_execution_authorized"])
        self.assertFalse(protocol["claim_ceiling"]["classification_generated"])
        self.assertFalse(protocol["claim_ceiling"]["science_result"])
        added = set(sys.modules) - before
        self.assertFalse(any(name == "cupy" or name.startswith("cupy.") for name in added))
        self.assertFalse(any(name == "rebound" or name.startswith("rebound.") for name in added))
        self.assertFalse(any(name == "jxplanetx" or name.startswith("jxplanetx.") for name in added))

    def test_prepare_validate_prepared_create_only_temp_root(self) -> None:
        self.assertFalse(m.DEFAULT_OUTPUT_ROOT.exists())
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared"
            expected = m.prepare(root)
            self.assertEqual(set(path.name for path in root.iterdir()), {"PROTOCOL.json"})
            self.assertEqual(m.validate_prepared(root), expected)
            with self.assertRaises(m.Full129RestartError):
                m.prepare(root)
        self.assertFalse(m.DEFAULT_OUTPUT_ROOT.exists())

    def test_reference_v3_roster_suffixes_and_all_active_boundary(self) -> None:
        with m._held_v3_snapshot() as snapshot:
            reference = m._reference_suffix(snapshot)
        self.assertEqual(hashlib.sha256(reference["continuation"]).hexdigest(), m.REFERENCE_CONTINUATION_SUFFIX_SHA256)
        self.assertEqual(hashlib.sha256(reference["checkpoints"]).hexdigest(), m.REFERENCE_CHECKPOINT_SUFFIX_SHA256)
        self.assertEqual(hashlib.sha256(reference["lifecycle"]).hexdigest(), m.REFERENCE_LIFECYCLE_SUFFIX_SHA256)
        self.assertEqual(reference["lifecycle"], bytes(m.LIFECYCLE_SUFFIX_BYTES))
        self.assertEqual(len(reference["segments"]), 10)
        self.assertTrue(all(row["route_record_count"] == row["removal_record_count"] == 0 for row in reference["segments"]))

    def test_continuation_codec_is_exact_and_rejects_nonfinite(self) -> None:
        with m._held_v3_snapshot() as snapshot:
            reference = m._reference_suffix(snapshot)
        midpoint = reference["continuation"][:m.CONTINUATION_RECORD_BYTES]
        positions, velocities = m._unpack_continuation(midpoint)
        self.assertEqual(m._pack_continuation(positions, velocities), midpoint)
        bad = list(positions)
        bad[0] = (math.nan, bad[0][1], bad[0][2])
        with self.assertRaises(m.Full129RestartError):
            m._pack_continuation(tuple(bad), velocities)

    def test_restart_state_preserves_sealed_metadata_and_exact_indices(self) -> None:
        midpoint = struct.pack(f"<{m.TRACER_COUNT * 6}d", *([1.0] * (m.TRACER_COUNT * 6)))
        base = FakeState(
            snapshot_id=f"tno-full129-smoke-v1-DE441-{m.OBJECT_ID}",
            global_source_step=0,
            tracer_relative_positions_AU=FakeArray([[0.0] * 3] * m.TRACER_COUNT),
            tracer_relative_velocities_AU_per_year=FakeArray([[0.0] * 3] * m.TRACER_COUNT),
            tracer_indices=FakeArray(range(m.TRACER_COUNT)),
            sealed_metadata="unchanged",
        )
        state = m._restart_state(FakeCP, base, midpoint)
        self.assertEqual(state.global_source_step, m.START_GLOBAL_SOURCE_STEP)
        self.assertEqual(state.snapshot_id, base.snapshot_id)
        self.assertEqual(state.sealed_metadata, "unchanged")
        self.assertEqual(tuple(FakeCP.asnumpy(state.tracer_indices).tolist()), tuple(range(129)))
        wrong = FakeState("restart-specific-id", 0, base.tracer_relative_positions_AU, base.tracer_relative_velocities_AU_per_year, base.tracer_indices, "unchanged")
        with self.assertRaises(m.Full129RestartError):
            m._restart_state(FakeCP, wrong, midpoint)

    def test_request_receipts_exact_types_and_distinct_child(self) -> None:
        protocol = m.build_protocol()
        transfer = m._request_transfer(protocol, parent_pid=123456789, challenge_hex="a" * 64)
        self.assertEqual(m._validate_request_transfer(transfer, protocol), transfer)
        request = m._published_request(transfer)
        self.assertEqual(m._validate_published_request(request, protocol), request)
        bad = dict(request)
        bad["start_year"] = False
        bad["content_sha256"] = m._domain_hash(m.REQUEST_DOMAIN, bad)
        with self.assertRaises(m.Full129RestartError):
            m._validate_published_request(bad, protocol)
        same_pid = dict(transfer)
        same_pid["parent_pid"] = os.getpid()
        same_pid["content_sha256"] = m._domain_hash(m.REQUEST_TRANSFER_DOMAIN, same_pid)
        with self.assertRaises(m.Full129RestartError):
            m._published_request(same_pid)

    def test_runtime_post_uses_exact_stage15_gate_projection_callables(self) -> None:
        protocol = m.build_protocol()
        with m._held_v3_snapshot() as snapshot:
            reference = m._reference_suffix(snapshot)
        retained_post = reference["runtime_post"]
        runtime = retained_post["v3_runtime_identity"]
        request = {"content_sha256": "a" * 64}
        custody = {"point_observation_only": True}
        with m._v3_delegate() as v3:
            with v3._INNER._delegates() as (_preflight, gate):
                resolved_gate = gate
            # Production constructs this receipt after the delegate context
            # has removed its private sys.modules registrations.
            post = m._runtime_post(protocol, custody, request, runtime, resolved_gate)
            self.assertEqual(
                post["v3_runtime_comparison_projection_sha256"],
                retained_post["v3_runtime_comparison_projection_sha256"],
            )
            self.assertEqual(
                post["v3_runtime_prelane_binding_projection_sha256"],
                retained_post["v3_runtime_prelane_binding_projection_sha256"],
            )
            self.assertEqual(
                post["engine_runtime_projection_sha256"],
                m.REFERENCE_ENGINE_PROJECTION_SHA256,
            )
            with self.assertRaises(m.Full129RestartError):
                m._runtime_post(protocol, custody, request, runtime, v3)

    def test_deterministic_segment_projection_rejects_relabel_and_bool_alias(self) -> None:
        reference = synthetic_v3_segment()
        projection = m._deterministic_v3_segment_projection(reference)
        self.assertEqual(projection["start_year"], 500)
        wrong = dict(reference)
        wrong["route_record_count"] = False
        with self.assertRaises(m.Full129RestartError):
            m._deterministic_v3_segment_projection(wrong)
        extra = dict(reference)
        extra["dynamical_classification"] = "SCATTERING"
        with self.assertRaises(m.Full129RestartError):
            m._deterministic_v3_segment_projection(extra)

    def test_segment_record_exact_crossbinding_and_hostile_mutations(self) -> None:
        checkpoint = b"checkpoint"
        lifecycle = b"lifecycle"
        continuation_start = b"start"
        continuation_end = b"end"
        source_segment = b"segment"
        source_start = b"source-start"
        source_end = b"source-end"
        reference = synthetic_v3_segment()
        projection = reference.copy()
        projection.update({
            "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            "lifecycle_sha256": hashlib.sha256(lifecycle).hexdigest(),
            "continuation_start_sha256": hashlib.sha256(continuation_start).hexdigest(),
            "continuation_end_sha256": hashlib.sha256(continuation_end).hexdigest(),
            "generated_source_segment_sha256": hashlib.sha256(source_segment).hexdigest(),
            "retained_source_segment_sha256": hashlib.sha256(source_segment).hexdigest(),
            "source_start_node_sha256": hashlib.sha256(source_start).hexdigest(),
            "source_end_node_sha256": hashlib.sha256(source_end).hexdigest(),
        })
        record = m._segment_record(
            restart_order=0, reference=projection, checkpoint=checkpoint, lifecycle=lifecycle,
            continuation_start=continuation_start, continuation_end=continuation_end,
            source_segment=source_segment, source_start=source_start, source_end=source_end,
            source_tile_sha256=projection["source_tile_sha256"],
            minimum_distance=240.0, maximum_kepler_iterations=6,
            engine_projection_sha256=m.REFERENCE_ENGINE_PROJECTION_SHA256,
            elapsed_seconds=0.2,
        )
        self.assertEqual(m._validate_segment_record(
            record, restart_order=0, reference=projection, checkpoint=checkpoint,
            lifecycle=lifecycle, continuation_start=continuation_start,
            continuation_end=continuation_end, source_segment=source_segment,
            source_start=source_start, source_end=source_end,
        ), record)
        for key, replacement in (("restart_segment_order", False), ("reference_segment_index", 99)):
            bad = json.loads(json.dumps(record))
            bad[key] = replacement
            bad["content_sha256"] = m._domain_hash(m.SEGMENT_DOMAIN, bad)
            with self.assertRaises(m.Full129RestartError):
                m._validate_segment_record(
                    bad, restart_order=0, reference=projection, checkpoint=checkpoint,
                    lifecycle=lifecycle, continuation_start=continuation_start,
                    continuation_end=continuation_end, source_segment=source_segment,
                    source_start=source_start, source_end=source_end,
                )

    def test_git_receipt_requires_exact_40_character_object_ids(self) -> None:
        protocol = m.build_protocol()
        row = {
            "schema": m.GIT_CUSTODY_SCHEMA,
            "externally_preregistered": True,
            "branch": m.PREREGISTRATION_BRANCH,
            "annotated_tag": m.PREREGISTRATION_TAG,
            "commit": "a" * 40,
            "tag_object": "b" * 40,
            "tag_peeled_commit": "a" * 40,
            "tag_object_type": "tag",
            "clean_tracked_paths": [
                {
                    "relative_path": relative,
                    "bytes": size,
                    "sha256": digest,
                }
                for relative, (size, digest) in zip(
                    protocol["preregistration"]["exact_clean_tracked_paths"],
                    (
                        (protocol["implementation"]["bytes"], protocol["implementation"]["sha256"]),
                        (protocol["implementation"]["test"]["bytes"], protocol["implementation"]["test"]["sha256"]),
                        (len(m._canonical_json(protocol)), hashlib.sha256(m._canonical_json(protocol)).hexdigest()),
                    ),
                    strict=True,
                )
            ],
            "network_remote_verification_performed": False,
            "point_observation_only": True,
            "content_sha256": "",
        }
        row["content_sha256"] = m._domain_hash(m.GIT_CUSTODY_DOMAIN, row)
        self.assertEqual(m._validate_git_custody(row, protocol), row)
        for field in ("commit", "tag_object"):
            bad = json.loads(json.dumps(row))
            bad[field] = "c" * 64
            if field == "commit":
                bad["tag_peeled_commit"] = bad[field]
            bad["content_sha256"] = m._domain_hash(m.GIT_CUSTODY_DOMAIN, bad)
            with self.assertRaises(m.Full129RestartError):
                m._validate_git_custody(bad, protocol)

    def test_source_locators_bind_year500_and_first_segment(self) -> None:
        descriptor = os.open(m.H64_SOURCE_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            node = m._source_node_payload(descriptor, m.START_GLOBAL_SOURCE_STEP)
            segment = m._source_segment_payload(descriptor, 10)
        finally:
            os.close(descriptor)
        self.assertEqual(hashlib.sha256(node).hexdigest(), m.REFERENCE_SOURCE_START_SHA256)
        self.assertEqual(hashlib.sha256(segment).hexdigest(), "266608bca8cfb37094d3672658f11f5c30d5a38cec545b5863eef350f76d159c")
        with self.assertRaises(m.Full129RestartError):
            descriptor = os.open(m.H64_SOURCE_PATH, os.O_RDONLY)
            try:
                m._source_segment_payload(descriptor, 9)
            finally:
                os.close(descriptor)

    def test_offline_replay_rederives_all_checkpoints_segments_and_source_invariants(self) -> None:
        before = set(sys.modules)
        with m._held_v3_snapshot() as retained:
            reference = m._reference_suffix(retained)
        continuations = m._fixed_records(
            reference["continuation"], m.CONTINUATION_RECORD_BYTES, 11, "continuation",
        )
        checkpoints = m._fixed_records(
            reference["checkpoints"], m.STATE_RECORD_BYTES, 11, "checkpoints",
        )
        lifecycle = m._fixed_records(
            reference["lifecycle"], m.LIFECYCLE_RECORD_BYTES, 11, "lifecycle",
        )
        records: list[dict[str, object]] = []
        with m._held_regular_file(
            m.H64_SOURCE_PATH, "test H64 evidence",
            expected_bytes=m.H64_SOURCE_BYTES, expected_sha256=m.H64_SOURCE_SHA256,
        ) as descriptor:
            for order, row in enumerate(reference["segments"]):
                segment_index = m.REFERENCE_START_RECORD + order
                records.append(m._segment_record(
                    restart_order=order, reference=row,
                    checkpoint=checkpoints[order + 1], lifecycle=lifecycle[order + 1],
                    continuation_start=continuations[order],
                    continuation_end=continuations[order + 1],
                    source_segment=m._source_segment_payload(descriptor, segment_index),
                    source_start=m._source_node_payload(descriptor, segment_index * m.SEGMENT_STEPS),
                    source_end=m._source_node_payload(descriptor, (segment_index + 1) * m.SEGMENT_STEPS),
                    source_tile_sha256=row["source_tile_sha256"],
                    minimum_distance=float.fromhex(row["minimum_source_distance_AU_hex"]),
                    maximum_kepler_iterations=row["maximum_kepler_iterations"],
                    engine_projection_sha256=m.REFERENCE_ENGINE_PROJECTION_SHA256,
                    elapsed_seconds=0.1,
                ))
        snapshot = {
            "RESTART_SUFFIX.bin": reference["continuation"],
            "CHECKPOINT_SUFFIX.bin": reference["checkpoints"],
            "LIFECYCLE_SUFFIX.bin": reference["lifecycle"],
            "SEGMENTS.jsonl": b"".join(m._canonical_json(row) for row in records),
            "ROUTES.jsonl": b"",
            "REMOVALS.jsonl": b"",
        }
        with m._v3_delegate() as v3:
            source_segments, source_endpoints, invariant, _segments_sha = m._replay_suffix_with_source(
                snapshot, reference, v3,
            )
        self.assertEqual(len(source_segments), 10)
        self.assertEqual(len(source_endpoints), 11)
        self.assertEqual(invariant, reference["result"]["source_invariants"])
        added = set(sys.modules) - before
        self.assertFalse(any(name == "cupy" or name.startswith("cupy.") for name in added))

    def test_pure_suffix_gate_rejects_coordinated_state_and_route_tamper(self) -> None:
        with m._held_v3_snapshot() as snapshot:
            reference = m._reference_suffix(snapshot)
        synthetic = {
            "RESTART_SUFFIX.bin": reference["continuation"],
            "CHECKPOINT_SUFFIX.bin": reference["checkpoints"],
            "LIFECYCLE_SUFFIX.bin": reference["lifecycle"],
            "ROUTES.jsonl": b"",
            "REMOVALS.jsonl": b"",
        }

        def exact_prefix_gate(value: dict[str, bytes]) -> None:
            continuations = m._fixed_records(value["RESTART_SUFFIX.bin"], m.CONTINUATION_RECORD_BYTES, 11, "continuation")
            checkpoints = m._fixed_records(value["CHECKPOINT_SUFFIX.bin"], m.STATE_RECORD_BYTES, 11, "checkpoint")
            lifecycle = m._fixed_records(value["LIFECYCLE_SUFFIX.bin"], m.LIFECYCLE_RECORD_BYTES, 11, "lifecycle")
            if (
                value["RESTART_SUFFIX.bin"] != reference["continuation"]
                or value["CHECKPOINT_SUFFIX.bin"] != reference["checkpoints"]
                or value["LIFECYCLE_SUFFIX.bin"] != reference["lifecycle"]
                or any(row != bytes(129) for row in lifecycle)
                or value["ROUTES.jsonl"] != b"" or value["REMOVALS.jsonl"] != b""
                or len(continuations) != len(checkpoints) != 11
            ):
                raise m.Full129RestartError("synthetic exact prefix gate failed")

        exact_prefix_gate(synthetic)
        for names in (("RESTART_SUFFIX.bin",), ("CHECKPOINT_SUFFIX.bin",), ("RESTART_SUFFIX.bin", "CHECKPOINT_SUFFIX.bin")):
            bad = dict(synthetic)
            for name in names:
                value = bytearray(bad[name])
                value[m.CONTINUATION_RECORD_BYTES + 7 if name == "RESTART_SUFFIX.bin" else m.STATE_RECORD_BYTES + 7] ^= 1
                bad[name] = bytes(value)
            with self.assertRaises(m.Full129RestartError):
                exact_prefix_gate(bad)
        bad_route = dict(synthetic)
        bad_route["ROUTES.jsonl"] = b"{}\n"
        with self.assertRaises(m.Full129RestartError):
            exact_prefix_gate(bad_route)

    def test_publication_is_create_only_and_result_last(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "output"
            m.prepare(root)
            artifacts = tuple((name, name.encode("ascii")) for name in m.EXPECTED_OUTPUT_ORDER)
            calls: list[str] = []
            original = os.link

            def recording_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
                calls.append(str(destination))
                original(source, destination, *args, **kwargs)

            with mock.patch.object(m.os, "link", side_effect=recording_link):
                m._publish(root, artifacts)
            self.assertEqual(calls, list(m.EXPECTED_OUTPUT_ORDER))
            self.assertEqual(calls[-1], "RESULT.json")
            self.assertEqual(set(path.name for path in root.iterdir()), set(m.EXPECTED_OUTPUT_ROSTER))

    def test_publication_failure_before_result_leaves_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "output"
            m.prepare(root)
            artifacts = tuple((name, name.encode("ascii")) for name in m.EXPECTED_OUTPUT_ORDER)
            original = os.link

            def fail_before_result(source: object, destination: object, *args: object, **kwargs: object) -> None:
                if destination == "RUNTIME_POST.json":
                    raise OSError("injected precommit failure")
                original(source, destination, *args, **kwargs)

            with mock.patch.object(m.os, "link", side_effect=fail_before_result):
                with self.assertRaises(OSError):
                    m._publish(root, artifacts)
            self.assertFalse((root / "RESULT.json").exists())

    def test_held_regular_file_rejects_link_count_change_during_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "evidence.bin"
            path.write_bytes(b"evidence")
            alias = Path(raw) / "alias.bin"
            with self.assertRaises(m.Full129RestartError):
                with m._held_regular_file(
                    path, "synthetic evidence", expected_bytes=8,
                    expected_sha256=hashlib.sha256(b"evidence").hexdigest(),
                ):
                    os.link(path, alias)

    def test_execute_rejects_nondefault_before_git_or_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(m.Full129RestartError):
                m.execute(Path(raw) / "other", externally_preregistered=True, prereg_commit="0" * 40, prereg_tag_object="1" * 40)

    def test_claim_validator_rejects_integer_boolean_alias(self) -> None:
        claims = m._result_claims()
        m._validate_exact_claims(claims, claims)
        bad = dict(claims)
        bad["gpu_executed"] = 1
        with self.assertRaises(m.Full129RestartError):
            m._validate_exact_claims(bad, claims)

    def test_cli_defaults_are_v1_restart_root(self) -> None:
        args = m._parser().parse_args(["validate-prepared"])
        self.assertEqual(args.output_root, m.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(m.DEFAULT_OUTPUT_ROOT.name, "tno_1kyr_full129_restart_qualification_v1")


if __name__ == "__main__":
    unittest.main()
