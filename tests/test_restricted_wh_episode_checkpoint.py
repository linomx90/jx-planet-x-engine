from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import jxplanetx.restricted_wh_episode_checkpoint as checkpoint_module
from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec
from jxplanetx.cuda_restricted_wh_episode_owner import (
    advance_cuda_restricted_wh_owned_episodes,
)
from jxplanetx.restricted_wh_episode_checkpoint import (
    CHECKPOINT_CLAIMS,
    OwnedEpisodeCheckpointContractError,
    OwnedEpisodeCheckpointIntegrityError,
    dump_owned_episode_delta_checkpoint,
    load_latest_owned_episode_delta_checkpoint,
    owned_episode_state_sha256,
)

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)
from tests.test_cuda_restricted_wh_episode import _assert_device_array_equal


RUN_ID = "owned-episode-delta-test"
PROTOCOL_SHA256 = hashlib.sha256(b"owned-episode-delta-test-v1").hexdigest()


def _dense_state(lane_count: int = 6, *, global_source_step: int = 0):
    state = _state(lane_count, global_source_step=global_source_step)
    positions = state.tracer_relative_positions_AU.copy()
    velocities = state.tracer_relative_velocities_AU_per_year.copy()
    positions[:, 0] = 1.0
    velocities[:] = velocities[0]
    return replace(
        state,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
    )


def _segment(
    state,
    *,
    step_count: int = 4,
    worker_count: int = 1,
    lookahead: int = 4,
    encounter_radius: float = 0.997,
):
    tile = _tile(
        start=state.global_source_step,
        node_count=step_count + 1,
        encounter_radius=encounter_radius,
        acceleration_bound=100.0,
    )
    spec = FixedStepCUDARestrictedWHSpec(step_count, 1, _backend())
    result = advance_cuda_restricted_wh_owned_episodes(
        state,
        tile,
        spec,
        route_worker_count=worker_count,
        route_lookahead_steps=lookahead,
    )
    return tile, spec, result


def _dump(
    root: Path,
    *,
    index: int,
    genesis: str,
    input_state,
    tile,
    spec,
    result,
    predecessor: str | None,
    worker_count: int = 1,
    lookahead: int = 4,
):
    return dump_owned_episode_delta_checkpoint(
        root,
        run_id=RUN_ID,
        protocol_sha256=PROTOCOL_SHA256,
        genesis_state_sha256=genesis,
        checkpoint_index=index,
        input_state=input_state,
        result=result,
        source_tile=tile,
        spec=spec,
        route_worker_count=worker_count,
        route_lookahead_steps=lookahead,
        previous_checkpoint_content_sha256=predecessor,
    )


def _load(root: Path, genesis: str, head):
    return load_latest_owned_episode_delta_checkpoint(
        root,
        run_id=RUN_ID,
        protocol_sha256=PROTOCOL_SHA256,
        genesis_state_sha256=genesis,
        expected_checkpoint_index=head["checkpoint_index"],
        expected_checkpoint_content_sha256=head["checkpoint_content_sha256"],
        backend=_backend(),
    )


def _namespace(root: Path) -> Path:
    return checkpoint_module._checkpoint_namespace(root, RUN_ID, PROTOCOL_SHA256)


def _record_path(root: Path, index: int) -> Path:
    return _namespace(root) / f"checkpoint_{index:020d}.json"


def _rehash_record(path: Path, record: dict) -> None:
    record.pop("checkpoint_content_sha256", None)
    record["checkpoint_content_sha256"] = hashlib.sha256(
        checkpoint_module.CHECKPOINT_DOMAIN
        + checkpoint_module._checkpoint_codec._canonical_json(record)
    ).hexdigest()
    path.write_text(
        json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


@GPU_REQUIRED
class OwnedEpisodeDeltaCheckpointTests(unittest.TestCase):
    def assertStateEqual(self, left, right) -> None:
        self.assertEqual(left.global_source_step, right.global_source_step)
        self.assertEqual(left.snapshot_id, right.snapshot_id)
        for left_array, right_array in (
            (
                left.tracer_relative_positions_AU,
                right.tracer_relative_positions_AU,
            ),
            (
                left.tracer_relative_velocities_AU_per_year,
                right.tracer_relative_velocities_AU_per_year,
            ),
            (left.tracer_indices, right.tracer_indices),
        ):
            _assert_device_array_equal(self, left_array, right_array)

    def _two_segment_chain(
        self, root: Path, *, worker_count: int = 1, lookahead: int = 4
    ):
        initial = _dense_state()
        genesis = owned_episode_state_sha256(initial, _backend())
        tile0, spec0, result0 = _segment(
            initial, worker_count=worker_count, lookahead=lookahead
        )
        row0 = _dump(
            root,
            index=0,
            genesis=genesis,
            input_state=initial,
            tile=tile0,
            spec=spec0,
            result=result0,
            predecessor=None,
            worker_count=worker_count,
            lookahead=lookahead,
        )
        tile1, spec1, result1 = _segment(
            result0.state, worker_count=worker_count, lookahead=lookahead
        )
        row1 = _dump(
            root,
            index=1,
            genesis=genesis,
            input_state=result0.state,
            tile=tile1,
            spec=spec1,
            result=result1,
            predecessor=row0["checkpoint_content_sha256"],
            worker_count=worker_count,
            lookahead=lookahead,
        )
        return initial, genesis, result0, result1, row0, row1

    def test_two_segment_chain_stores_only_route_deltas_and_replays_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                _initial,
                genesis,
                result0,
                result1,
                row0,
                row1,
            ) = self._two_segment_chain(root, worker_count=6, lookahead=4)
            expected_delta = 6 * 4
            self.assertEqual(row0["route_record_count_delta"], expected_delta)
            self.assertEqual(row1["route_record_count_delta"], expected_delta)
            self.assertEqual(
                len(row1["result"]["route_records"]), expected_delta
            )
            self.assertEqual(
                row1["route_record_count_cumulative"], 2 * expected_delta
            )
            self.assertEqual(row0["segment_episode_count_delta"], 6)
            self.assertEqual(row1["segment_episode_count_delta"], 6)
            self.assertNotIn("prior_route_records", row1)
            self.assertNotIn("cumulative_route_records", row1)
            self.assertEqual(
                row1["previous_route_chain_sha256"], row0["route_chain_sha256"]
            )
            self.assertNotEqual(
                row1["route_delta_sha256"], row1["route_chain_sha256"]
            )
            loaded = _load(root, genesis, row1)
            self.assertStateEqual(loaded.state, result1.state)
            self.assertEqual(
                loaded.route_record_count_cumulative, 2 * expected_delta
            )
            self.assertEqual(loaded.segment_episode_count_cumulative, 12)
            self.assertEqual(
                loaded.route_chain_sha256, row1["route_chain_sha256"]
            )
            self.assertFalse(any(loaded.claims.values()))
            self.assertEqual(dict(loaded.claims), dict(CHECKPOINT_CLAIMS))

    def test_quiet_segment_has_empty_delta_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _state(5)
            genesis = owned_episode_state_sha256(initial, _backend())
            tile = _tile(node_count=5)
            spec = FixedStepCUDARestrictedWHSpec(4, 1, _backend())
            result = advance_cuda_restricted_wh_owned_episodes(
                initial, tile, spec, route_worker_count=1
            )
            row = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile,
                spec=spec,
                result=result,
                predecessor=None,
                lookahead=32,
            )
            self.assertEqual(row["route_record_count_delta"], 0)
            self.assertEqual(row["route_record_count_cumulative"], 0)
            self.assertEqual(row["segment_episode_count_delta"], 0)
            self.assertEqual(row["result"]["route_records"], ())
            self.assertStateEqual(_load(root, genesis, row).state, result.state)

    def test_loaded_head_can_continue_with_exact_state_and_source_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile0, spec0, result0 = _segment(initial)
            row0 = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile0,
                spec=spec0,
                result=result0,
                predecessor=None,
            )
            loaded0 = _load(root, genesis, row0)
            self.assertStateEqual(loaded0.state, result0.state)
            tile1, spec1, result1 = _segment(loaded0.state)
            row1 = _dump(
                root,
                index=1,
                genesis=genesis,
                input_state=loaded0.state,
                tile=tile1,
                spec=spec1,
                result=result1,
                predecessor=loaded0.checkpoint_content_sha256,
            )
            loaded1 = _load(root, genesis, row1)
            self.assertStateEqual(loaded1.state, result1.state)

    def test_external_head_anchor_detects_tail_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _initial, genesis, _r0, _r1, _row0, row1 = self._two_segment_chain(
                root
            )
            _record_path(root, 1).unlink()
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointIntegrityError, "head differs"
            ):
                _load(root, genesis, row1)

    def test_rehashed_worker_assignment_tamper_fails_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(
                initial, worker_count=2, lookahead=4
            )
            row = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile,
                spec=spec,
                result=result,
                predecessor=None,
                worker_count=2,
            )
            path = _record_path(root, 0)
            record = json.loads(path.read_text(encoding="utf-8"))
            first_tracer = record["result"]["route_records"][0]["tracer_index"]
            first_worker = record["result"]["route_worker_indices"][0]
            replacement_worker = 1 - first_worker
            assignment_payload = []
            for route_row, worker in zip(
                record["result"]["route_records"],
                record["result"]["route_worker_indices"],
                strict=True,
            ):
                if route_row["tracer_index"] == first_tracer:
                    worker = replacement_worker
                assignment_payload.append(
                    {
                        "map_step_index": route_row["map_step_index"],
                        "lane_index": route_row["lane_index"],
                        "tracer_index": route_row["tracer_index"],
                        "worker_index": worker,
                    }
                )
            record["result"]["route_worker_indices"] = [
                item["worker_index"] for item in assignment_payload
            ]
            for episode in record["result"]["segment_episode_summaries"]:
                if episode["tracer_index"] == first_tracer:
                    episode["worker_index"] = replacement_worker
            record["result"]["runtime_identity"][
                "route_worker_assignment_sha256"
            ] = hashlib.sha256(
                b"jx.cuda-restricted-wh-owned-assignments.v1\0"
                + json.dumps(
                    assignment_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest()
            record["route_delta_sha256"] = checkpoint_module._route_delta_sha256(
                record["result"]
            )
            record["route_chain_sha256"] = checkpoint_module._route_chain_sha256(
                previous_route_chain_sha256=None,
                checkpoint_index=0,
                input_global_source_step=initial.global_source_step,
                output_global_source_step=result.state.global_source_step,
                route_delta_sha256=record["route_delta_sha256"],
            )
            segment_record = {
                "input_state_sha256": record["input_state_sha256"],
                "state_sha256": record["state_sha256"],
                "lineage_sha256": record["lineage_sha256"],
                "source_tile_sha256": record["result"]["source_tile_sha256"],
                "spec": record["spec"],
                "route_policy": record["route_policy"],
                "route_worker_count": record["route_worker_count"],
                "route_lookahead_steps": record["route_lookahead_steps"],
                "route_delta_sha256": record["route_delta_sha256"],
                "route_chain_sha256": record["route_chain_sha256"],
                "result": record["result"],
            }
            record["segment_sha256"] = checkpoint_module._segment_sha256(
                segment_record
            )
            _rehash_record(path, record)
            forged = dict(row)
            forged["checkpoint_content_sha256"] = json.loads(
                path.read_text(encoding="utf-8")
            )["checkpoint_content_sha256"]
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointIntegrityError, "route delta differs"
            ):
                _load(root, genesis, forged)

    def test_successor_source_lineage_change_is_rejected_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile0, spec0, result0 = _segment(initial)
            row0 = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile0,
                spec=spec0,
                result=result0,
                predecessor=None,
            )
            tile1, spec1, result1 = _segment(
                result0.state, encounter_radius=0.998
            )
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointContractError, "source lineage"
            ):
                _dump(
                    root,
                    index=1,
                    genesis=genesis,
                    input_state=result0.state,
                    tile=tile1,
                    spec=spec1,
                    result=result1,
                    predecessor=row0["checkpoint_content_sha256"],
                )
            self.assertFalse(_record_path(root, 1).exists())

    def test_wrong_genesis_and_checkpoint_gap_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointContractError, "exactly extend"
            ):
                _dump(
                    root,
                    index=1,
                    genesis=genesis,
                    input_state=initial,
                    tile=tile,
                    spec=spec,
                    result=result,
                    predecessor=None,
                )
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointContractError, "genesis"
            ):
                _dump(
                    root,
                    index=0,
                    genesis="f" * 64,
                    input_state=initial,
                    tile=tile,
                    spec=spec,
                    result=result,
                    predecessor=None,
                )

    def test_minimum_array_dtype_tamper_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            row = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile,
                spec=spec,
                result=result,
                predecessor=None,
            )
            path = _record_path(root, 0)
            record = json.loads(path.read_text(encoding="utf-8"))
            old_array = path.parent / record["array_file"]
            with np.load(old_array, allow_pickle=False) as archive:
                arrays = {
                    key: np.array(archive[key], copy=True)
                    for key in archive.files
                }
            arrays["minimum_source_distance_AU"] = arrays[
                "minimum_source_distance_AU"
            ].astype(np.float32)
            temporary_array = path.parent / ".tampered.npz"
            with temporary_array.open("wb") as stream:
                np.savez(stream, **arrays)
            digest = checkpoint_module._checkpoint_codec._sha256_file(
                temporary_array
            )
            array_path = path.parent / (
                f"checkpoint_{0:020d}.array-{digest}.npz"
            )
            temporary_array.replace(array_path)
            record["array_file"] = array_path.name
            record["array_file_size_bytes"] = array_path.stat().st_size
            record["array_file_sha256"] = digest
            _rehash_record(path, record)
            forged = dict(row)
            forged["checkpoint_content_sha256"] = json.loads(
                path.read_text(encoding="utf-8")
            )["checkpoint_content_sha256"]
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointIntegrityError, "minimum-distance"
            ):
                _load(root, genesis, forged)

    def test_claim_type_tamper_is_rejected_after_outer_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            row = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile,
                spec=spec,
                result=result,
                predecessor=None,
            )
            path = _record_path(root, 0)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["claim_ceiling"]["claims"]["science_result"] = 0
            _rehash_record(path, record)
            forged = dict(row)
            forged["checkpoint_content_sha256"] = json.loads(
                path.read_text(encoding="utf-8")
            )["checkpoint_content_sha256"]
            with self.assertRaisesRegex(
                OwnedEpisodeCheckpointIntegrityError, "claim ceiling"
            ):
                _load(root, genesis, forged)

    def test_runtime_and_implementation_tamper_fail_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            row = _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile,
                spec=spec,
                result=result,
                predecessor=None,
            )
            path = _record_path(root, 0)
            original_text = path.read_text(encoding="utf-8")
            cases = (
                (
                    "implementation_identity",
                    "checkpoint_implementation_source_sha256",
                    "0" * 64,
                    "implementation/kernel identity",
                ),
                (
                    "checkpoint_runtime_identity",
                    "cuda_driver_version",
                    0,
                    "runtime identity",
                ),
            )
            for container, key, value, message in cases:
                with self.subTest(container=container, key=key):
                    record = json.loads(original_text)
                    record[container][key] = value
                    _rehash_record(path, record)
                    forged = dict(row)
                    forged["checkpoint_content_sha256"] = json.loads(
                        path.read_text(encoding="utf-8")
                    )["checkpoint_content_sha256"]
                    with self.assertRaisesRegex(
                        OwnedEpisodeCheckpointIntegrityError, message
                    ):
                        _load(root, genesis, forged)
                    path.write_text(original_text, encoding="utf-8")

    def test_tampered_predecessor_cannot_authorize_successor_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile0, spec0, result0 = _segment(initial)
            _dump(
                root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile0,
                spec=spec0,
                result=result0,
                predecessor=None,
            )
            tile1, spec1, result1 = _segment(result0.state)
            path = _record_path(root, 0)
            original_text = path.read_text(encoding="utf-8")
            cases = (
                (
                    "implementation_identity",
                    "checkpoint_implementation_source_sha256",
                    "0" * 64,
                    "implementation/kernel identity",
                ),
                (
                    "checkpoint_runtime_identity",
                    "cuda_driver_version",
                    0,
                    "runtime identity",
                ),
            )
            for container, key, value, message in cases:
                with self.subTest(container=container, key=key):
                    path.write_text(original_text, encoding="utf-8")
                    record = json.loads(original_text)
                    record[container][key] = value
                    _rehash_record(path, record)
                    forged_head = json.loads(path.read_text(encoding="utf-8"))[
                        "checkpoint_content_sha256"
                    ]
                    with self.assertRaisesRegex(
                        OwnedEpisodeCheckpointIntegrityError, message
                    ):
                        _dump(
                            root,
                            index=1,
                            genesis=genesis,
                            input_state=result0.state,
                            tile=tile1,
                            spec=spec1,
                            result=result1,
                            predecessor=forged_head,
                        )
                    self.assertFalse(_record_path(root, 1).exists())
            path.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
