from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

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
    open_owned_episode_delta_append_session,
    owned_episode_state_sha256,
)
from jxplanetx.restricted_wh_route import RestrictedWHRouteSpec

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
    route_spec: RestrictedWHRouteSpec | None = None,
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
        route_spec=route_spec,
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
    route_spec: RestrictedWHRouteSpec | None = None,
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
        route_spec=route_spec,
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


def _array_path(root: Path, index: int) -> Path:
    record = json.loads(_record_path(root, index).read_text(encoding="utf-8"))
    return _record_path(root, index).parent / record["array_file"]


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

    def test_distinct_terminal_surface_policy_round_trips_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state(lane_count=1)
            genesis = owned_episode_state_sha256(initial, _backend())
            route_spec = RestrictedWHRouteSpec(
                terminal_source_radii_AU=(0.5,)
            )
            tile, spec, result = _segment(
                initial,
                step_count=2,
                route_spec=route_spec,
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
                route_spec=route_spec,
            )
            self.assertEqual(
                row["route_policy"]["terminal_source_radii_AU_hex"],
                (0.5.hex(),),
            )
            loaded = _load(root, genesis, row)
            self.assertStateEqual(loaded.state, result.state)
            self.assertEqual(loaded.route_record_count_cumulative, 2)

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

    def test_locked_append_session_matches_standalone_records_exactly(self):
        initial = _dense_state()
        genesis = owned_episode_state_sha256(initial, _backend())
        tile0, spec0, result0 = _segment(initial, worker_count=2)
        tile1, spec1, result1 = _segment(result0.state, worker_count=2)
        with (
            tempfile.TemporaryDirectory() as standalone_temporary,
            tempfile.TemporaryDirectory() as session_temporary,
        ):
            standalone_root = Path(standalone_temporary)
            session_root = Path(session_temporary)
            row0 = _dump(
                standalone_root,
                index=0,
                genesis=genesis,
                input_state=initial,
                tile=tile0,
                spec=spec0,
                result=result0,
                predecessor=None,
                worker_count=2,
            )
            row1 = _dump(
                standalone_root,
                index=1,
                genesis=genesis,
                input_state=result0.state,
                tile=tile1,
                spec=spec1,
                result=result1,
                predecessor=row0["checkpoint_content_sha256"],
                worker_count=2,
            )
            with open_owned_episode_delta_append_session(
                session_root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
            ) as session:
                head0 = session.append(
                    input_state=initial,
                    result=result0,
                    source_tile=tile0,
                    spec=spec0,
                    route_worker_count=2,
                    route_lookahead_steps=4,
                )
                head1 = session.append(
                    input_state=result0.state,
                    result=result1,
                    source_tile=tile1,
                    spec=spec1,
                    route_worker_count=2,
                    route_lookahead_steps=4,
                )
                self.assertIs(session.loaded_head, head1)
            self.assertEqual(
                head0.checkpoint_content_sha256,
                row0["checkpoint_content_sha256"],
            )
            self.assertEqual(
                head1.checkpoint_content_sha256,
                row1["checkpoint_content_sha256"],
            )
            loaded = _load(session_root, genesis, row1)
            self.assertStateEqual(loaded.state, result1.state)

    def test_append_session_resumes_once_from_external_head(self):
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
            tile1, spec1, result1 = _segment(result0.state)
            with open_owned_episode_delta_append_session(
                root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
                expected_checkpoint_index=0,
                expected_checkpoint_content_sha256=row0[
                    "checkpoint_content_sha256"
                ],
            ) as session:
                self.assertEqual(
                    session.loaded_head.checkpoint_content_sha256,
                    row0["checkpoint_content_sha256"],
                )
                head1 = session.append(
                    input_state=result0.state,
                    result=result1,
                    source_tile=tile1,
                    spec=spec1,
                    route_worker_count=1,
                    route_lookahead_steps=4,
                )
            self.assertEqual(head1.checkpoint_index, 1)
            self.assertStateEqual(_load(root, genesis, head1.record).state, result1.state)

    def test_append_session_rejects_changed_locked_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile0, spec0, result0 = _segment(initial)
            with open_owned_episode_delta_append_session(
                root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
            ) as session:
                head0 = session.append(
                    input_state=initial,
                    result=result0,
                    source_tile=tile0,
                    spec=spec0,
                    route_worker_count=1,
                    route_lookahead_steps=4,
                )
                path = _record_path(root, 0)
                record = json.loads(path.read_text(encoding="utf-8"))
                record["route_record_count_cumulative"] += 1
                _rehash_record(path, record)
                tile1, spec1, result1 = _segment(result0.state)
                with self.assertRaisesRegex(
                    OwnedEpisodeCheckpointIntegrityError,
                    "previous checkpoint digest|artifact inventory changed",
                ):
                    session.append(
                        input_state=result0.state,
                        result=result1,
                        source_tile=tile1,
                        spec=spec1,
                        route_worker_count=1,
                        route_lookahead_steps=4,
                    )
            self.assertFalse(_record_path(root, 1).exists())
            self.assertNotEqual(
                json.loads(_record_path(root, 0).read_text(encoding="utf-8"))[
                    "checkpoint_content_sha256"
                ],
                head0.checkpoint_content_sha256,
            )

    def test_append_session_rehashes_old_arrays_before_every_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile0, spec0, result0 = _segment(initial)
            tile1, spec1, result1 = _segment(result0.state)
            with open_owned_episode_delta_append_session(
                root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
            ) as session:
                session.append(
                    input_state=initial,
                    result=result0,
                    source_tile=tile0,
                    spec=spec0,
                    route_worker_count=1,
                    route_lookahead_steps=4,
                )
                path = _array_path(root, 0)
                payload = bytearray(path.read_bytes())
                payload[len(payload) // 2] ^= 0x01
                path.write_bytes(payload)
                with self.assertRaisesRegex(
                    OwnedEpisodeCheckpointIntegrityError, "array SHA-256"
                ):
                    session.append(
                        input_state=result0.state,
                        result=result1,
                        source_tile=tile1,
                        spec=spec1,
                        route_worker_count=1,
                        route_lookahead_steps=4,
                    )
            self.assertFalse(_record_path(root, 1).exists())

    def test_append_session_head_comes_from_published_replay_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            expected_positions = CP.asnumpy(
                result.state.tracer_relative_positions_AU
            ).copy()
            with open_owned_episode_delta_append_session(
                root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
            ) as session:
                head = session.append(
                    input_state=initial,
                    result=result,
                    source_tile=tile,
                    spec=spec,
                    route_worker_count=1,
                    route_lookahead_steps=4,
                )
                result.state.tracer_relative_positions_AU[0, 0] += 1.0
                CP.cuda.get_current_stream().synchronize()
                self.assertTrue(
                    np.array_equal(
                        CP.asnumpy(head.state.tracer_relative_positions_AU),
                        expected_positions,
                    )
                )
            loaded = _load(root, genesis, head.record)
            self.assertTrue(
                np.array_equal(
                    CP.asnumpy(loaded.state.tracer_relative_positions_AU),
                    expected_positions,
                )
            )

    def test_append_session_rejects_process_thread_and_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = _dense_state()
            genesis = owned_episode_state_sha256(initial, _backend())
            tile, spec, result = _segment(initial)
            session = open_owned_episode_delta_append_session(
                root,
                run_id=RUN_ID,
                protocol_sha256=PROTOCOL_SHA256,
                genesis_state_sha256=genesis,
                backend=_backend(),
            )
            errors: list[BaseException] = []

            def use_from_other_thread() -> None:
                try:
                    session.append(
                        input_state=initial,
                        result=result,
                        source_tile=tile,
                        spec=spec,
                        route_worker_count=1,
                        route_lookahead_steps=4,
                    )
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=use_from_other_thread)
            worker.start()
            worker.join()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(
                errors[0], OwnedEpisodeCheckpointIntegrityError
            )
            with mock.patch.object(
                checkpoint_module.os,
                "getpid",
                return_value=session._process_id + 1,
            ):
                with self.assertRaisesRegex(
                    OwnedEpisodeCheckpointIntegrityError, "process fork"
                ):
                    session.append(
                        input_state=initial,
                        result=result,
                        source_tile=tile,
                        spec=spec,
                        route_worker_count=1,
                        route_lookahead_steps=4,
                    )

            runtime = dict(
                checkpoint_module._checkpoint_codec._plain_tree(
                    checkpoint_module.owned_episode_checkpoint_runtime_identity(
                        _backend()
                    )
                )
            )
            runtime["checkpoint_method_id"] = "changed"
            with mock.patch.object(
                checkpoint_module,
                "owned_episode_checkpoint_runtime_identity",
                return_value=runtime,
            ):
                with self.assertRaisesRegex(
                    OwnedEpisodeCheckpointIntegrityError, "runtime identity"
                ):
                    session.append(
                        input_state=initial,
                        result=result,
                        source_tile=tile,
                        spec=spec,
                        route_worker_count=1,
                        route_lookahead_steps=4,
                    )
            self.assertTrue(session._closed)
            self.assertFalse(_record_path(root, 0).exists())

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
