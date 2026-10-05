from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

import jxplanetx.restricted_wh_checkpoint as checkpoint_module
from jxplanetx.restricted_wh_checkpoint import (
    CHECKPOINT_CLAIMS,
    RestrictedWHCheckpointContractError,
    RestrictedWHCheckpointIntegrityError,
    dump_restricted_wh_lifecycle_checkpoint,
    load_latest_restricted_wh_lifecycle_checkpoint,
    restricted_wh_lifecycle_state_sha256,
)
from jxplanetx.restricted_wh_lifecycle import (
    REMOVED_SOURCE_RADIUS,
    advance_cuda_restricted_wh_lifecycle,
    initialize_restricted_wh_lifecycle,
)

from tests.test_cuda_restricted_wh import (
    CP,
    GPU_REQUIRED,
    _backend,
    _state,
    _tile,
)
from tests.test_restricted_wh_lifecycle import _source_contact_state
from jxplanetx.cuda_restricted_wh import FixedStepCUDARestrictedWHSpec


PROTOCOL_SHA256 = hashlib.sha256(
    b"restricted-wh-checkpoint-test-protocol-v1"
).hexdigest()


_CRASH_WORKER_MODULE = "tests.restricted_wh_checkpoint_crash_worker"


def _kill_writer_at_publication_boundary(root: Path, phase: str) -> None:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    repository_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, repository_root, existing)
        if value
    )
    completed = subprocess.run(
        [sys.executable, "-m", _CRASH_WORKER_MODULE, phase, str(root)],
        cwd=repository_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60.0,
    )
    expected = {
        "before_array_link": 91,
        "after_array_link": 92,
        "before_record_link": 93,
        "after_record_link": 94,
    }[phase]
    if completed.returncode != expected:
        raise AssertionError(
            f"crash worker exited {completed.returncode}, expected {expected}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )


def _seed_successor_fixture(root: Path):
    initial = initialize_restricted_wh_lifecycle(_state(1))
    first, tile0, spec0 = _run_segment(initial, start=0, steps=1)
    row0 = _dump(
        root,
        index=0,
        genesis_state=initial,
        input_state=initial,
        result=first,
        tile=tile0,
        spec=spec0,
    )
    second, tile1, spec1 = _run_segment(first.state, start=1, steps=1)
    return initial, first, second, tile1, spec1, row0


def _assert_append_lock_released(testcase: unittest.TestCase, root: Path) -> None:
    lock_path = _namespace(root) / ".append.lock"
    testcase.assertTrue(lock_path.is_file())
    testcase.assertEqual(lock_path.stat().st_size, 0)
    with lock_path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_segment(
    state,
    *,
    start: int,
    steps: int,
    terminal: bool = False,
    route_worker_count: int = 1,
):
    tile = _tile(
        start=start,
        node_count=steps + 1,
        encounter_radius=1.0 if terminal else 1.0e-4,
        acceleration_bound=100.0,
    )
    spec = FixedStepCUDARestrictedWHSpec(steps, 1, _backend())
    result = advance_cuda_restricted_wh_lifecycle(
        state,
        tile,
        spec,
        route_worker_count=route_worker_count,
    )
    return result, tile, spec


def _dump(
    directory: Path,
    *,
    index: int,
    genesis_state,
    input_state,
    result,
    tile,
    spec,
    predecessor: str | None = None,
    route_worker_count: int = 1,
):
    return dump_restricted_wh_lifecycle_checkpoint(
        directory,
        run_id="checkpoint-test-run",
        protocol_sha256=PROTOCOL_SHA256,
        genesis_state_sha256=restricted_wh_lifecycle_state_sha256(
            genesis_state, _backend()
        ),
        checkpoint_index=index,
        input_state=input_state,
        result=result,
        source_tile=tile,
        spec=spec,
        route_worker_count=route_worker_count,
        previous_checkpoint_content_sha256=predecessor,
    )


def _load(directory: Path, genesis_state, head):
    return load_latest_restricted_wh_lifecycle_checkpoint(
        directory,
        run_id="checkpoint-test-run",
        protocol_sha256=PROTOCOL_SHA256,
        genesis_state_sha256=restricted_wh_lifecycle_state_sha256(
            genesis_state, _backend()
        ),
        expected_checkpoint_index=head["checkpoint_index"],
        expected_checkpoint_content_sha256=head[
            "checkpoint_content_sha256"
        ],
        backend=_backend(),
    )


def _namespace(directory: Path) -> Path:
    return checkpoint_module._checkpoint_namespace(
        directory,
        "checkpoint-test-run",
        PROTOCOL_SHA256,
    )


def _rewrite_record(path: Path, mutate) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("checkpoint_content_sha256")
    mutate(record)
    record["checkpoint_content_sha256"] = hashlib.sha256(
        checkpoint_module.CHECKPOINT_DOMAIN
        + checkpoint_module._canonical_json(record)
    ).hexdigest()
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_segment_sha256(record: dict) -> None:
    record["segment_sha256"] = checkpoint_module._segment_sha256(
        {
            "input_state_sha256": record["input_state_sha256"],
            "state_sha256": record["state_sha256"],
            "lineage_sha256": record["lineage_sha256"],
            "source_tile_sha256": record["source_tile_metadata"][
                "content_sha256"
            ],
            "spec": record["spec"],
            "route_policy": record["route_policy"],
            "route_worker_count": record["route_worker_count"],
            "result": record["result"],
        }
    )


def _replace_minimum_with_float32(namespace: Path, record_path: Path) -> dict:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source = namespace / record["array_file"]
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    arrays["minimum_source_distance_AU"] = np.ascontiguousarray(
        arrays["minimum_source_distance_AU"].astype(np.float32)
    )
    with tempfile.NamedTemporaryFile(
        dir=namespace,
        prefix=".tampered-array.",
        suffix=".npz",
        delete=False,
    ) as stream:
        np.savez(stream, **arrays)
        stream.flush()
        temporary = Path(stream.name)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    replacement = namespace / (
        f"checkpoint_{record['checkpoint_index']:020d}.array-{digest}.npz"
    )
    temporary.replace(replacement)
    record.pop("checkpoint_content_sha256")
    record["array_file"] = replacement.name
    record["array_file_size_bytes"] = replacement.stat().st_size
    record["array_file_sha256"] = digest
    minimum_digest = hashlib.sha256()
    checkpoint_module._array_digest_update(
        minimum_digest,
        "minimum_source_distance_AU",
        arrays["minimum_source_distance_AU"],
    )
    record["result"][
        "minimum_source_distance_array_sha256"
    ] = minimum_digest.hexdigest()
    _refresh_segment_sha256(record)
    record["checkpoint_content_sha256"] = hashlib.sha256(
        checkpoint_module.CHECKPOINT_DOMAIN
        + checkpoint_module._canonical_json(record)
    ).hexdigest()
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


@GPU_REQUIRED
class RestrictedWHCheckpointTests(unittest.TestCase):
    def test_process_death_before_array_link_leaves_temporary_and_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, first, result, tile, spec, row0 = _seed_successor_fixture(root)
            namespace = _namespace(root)
            row0_path = namespace / "checkpoint_00000000000000000000.json"
            row0_sha256 = hashlib.sha256(row0_path.read_bytes()).hexdigest()
            _kill_writer_at_publication_boundary(root, "before_array_link")
            _assert_append_lock_released(self, root)
            self.assertEqual(len(list(namespace.glob("checkpoint_*.json"))), 1)
            self.assertEqual(len(list(namespace.glob("checkpoint_*.npz"))), 1)
            self.assertEqual(len(list(namespace.glob(".checkpoint_*.npz"))), 1)
            self.assertEqual(_load(root, initial, row0).checkpoint_index, 0)
            row = _dump(
                root,
                index=1,
                genesis_state=initial,
                input_state=first.state,
                result=result,
                tile=tile,
                spec=spec,
                predecessor=row0["checkpoint_content_sha256"],
            )
            self.assertEqual(hashlib.sha256(row0_path.read_bytes()).hexdigest(), row0_sha256)
            self.assertEqual(_load(root, initial, row).checkpoint_index, 1)

    def test_process_death_after_array_link_preserves_orphan_and_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, first, result, tile, spec, row0 = _seed_successor_fixture(root)
            _kill_writer_at_publication_boundary(root, "after_array_link")
            namespace = _namespace(root)
            arrays = list(namespace.glob("checkpoint_*.npz"))
            hidden = list(namespace.glob(".checkpoint_*.npz"))
            self.assertEqual(len(arrays), 2)
            self.assertEqual(len(list(namespace.glob("checkpoint_*.json"))), 1)
            self.assertEqual(len(hidden), 1)
            canonical = next(
                path
                for path in arrays
                if path.stat().st_ino == hidden[0].stat().st_ino
            )
            self.assertEqual(canonical.stat().st_nlink, 2)
            self.assertEqual(hidden[0].stat().st_nlink, 2)
            _assert_append_lock_released(self, root)
            self.assertEqual(_load(root, initial, row0).checkpoint_index, 0)
            row = _dump(
                root,
                index=1,
                genesis_state=initial,
                input_state=first.state,
                result=result,
                tile=tile,
                spec=spec,
                predecessor=row0["checkpoint_content_sha256"],
            )
            self.assertEqual(len(list(namespace.glob("checkpoint_*.npz"))), 2)
            self.assertEqual(_load(root, initial, row).checkpoint_index, 1)

    def test_process_death_before_record_link_keeps_old_head_and_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, first, result, tile, spec, row0 = _seed_successor_fixture(root)
            _kill_writer_at_publication_boundary(root, "before_record_link")
            namespace = _namespace(root)
            self.assertEqual(len(list(namespace.glob("checkpoint_*.npz"))), 2)
            self.assertEqual(len(list(namespace.glob("checkpoint_*.json"))), 1)
            self.assertEqual(
                len(list(namespace.glob(".checkpoint_*.json.*.tmp"))), 1
            )
            _assert_append_lock_released(self, root)
            self.assertEqual(_load(root, initial, row0).checkpoint_index, 0)
            row1 = _dump(
                root,
                index=1,
                genesis_state=initial,
                input_state=first.state,
                result=result,
                tile=tile,
                spec=spec,
                predecessor=row0["checkpoint_content_sha256"],
            )
            self.assertEqual(_load(root, initial, row1).checkpoint_index, 1)

    def test_process_death_after_record_link_requires_explicit_head_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, first, result, tile, spec, row0 = _seed_successor_fixture(root)
            _kill_writer_at_publication_boundary(root, "after_record_link")
            record_path = (
                _namespace(root) / "checkpoint_00000000000000000001.json"
            )
            self.assertTrue(record_path.is_file())
            hidden = list(_namespace(root).glob(".checkpoint_*.json.*.tmp"))
            self.assertEqual(len(hidden), 1)
            self.assertEqual(record_path.stat().st_ino, hidden[0].stat().st_ino)
            self.assertEqual(record_path.stat().st_nlink, 2)
            _assert_append_lock_released(self, root)
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, row0)
            recovered_head = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(
                RestrictedWHCheckpointContractError,
                "checkpoint indices must increase monotonically",
            ):
                _dump(
                    root,
                    index=1,
                    genesis_state=initial,
                    input_state=first.state,
                    result=result,
                    tile=tile,
                    spec=spec,
                    predecessor=row0["checkpoint_content_sha256"],
                )
            loaded = _load(root, initial, recovered_head)
            self.assertEqual(loaded.checkpoint_index, 1)
            self.assertEqual(
                loaded.checkpoint_content_sha256,
                recovered_head["checkpoint_content_sha256"],
            )

    def test_quiet_roundtrip_is_replay_verified_and_immutable(self):
        initial = initialize_restricted_wh_lifecycle(_state(3))
        result, tile, spec = _run_segment(initial, start=0, steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            loaded = _load(root, initial, record)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.checkpoint_index, 0)
            self.assertEqual(
                loaded.checkpoint_content_sha256,
                record["checkpoint_content_sha256"],
            )
            self.assertEqual(dict(loaded.claims), dict(CHECKPOINT_CLAIMS))
            self.assertFalse(any(loaded.claims.values()))
            implementation = record["implementation_identity"]
            self.assertEqual(
                implementation["backend_implementation_source_sha256"],
                checkpoint_module.BACKENDS_IMPLEMENTATION_SOURCE_SHA256,
            )
            self.assertEqual(
                implementation["contracts_implementation_source_sha256"],
                checkpoint_module.CONTRACTS_IMPLEMENTATION_SOURCE_SHA256,
            )
            self.assertEqual(
                implementation["wisdom_holman_contracts_source_sha256"],
                checkpoint_module.WH_CONTRACTS_IMPLEMENTATION_SOURCE_SHA256,
            )
            with self.assertRaises(TypeError):
                loaded.record["method_id"] = "forged"
            for left, right in (
                (
                    loaded.state.dynamics_state.tracer_relative_positions_AU,
                    result.state.dynamics_state.tracer_relative_positions_AU,
                ),
                (
                    loaded.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                    result.state.dynamics_state.tracer_relative_velocities_AU_per_year,
                ),
                (
                    loaded.state.dynamics_state.tracer_indices,
                    result.state.dynamics_state.tracer_indices,
                ),
            ):
                self.assertTrue(bool(CP.array_equal(left, right)))
            np.testing.assert_array_equal(
                loaded.state.tracer_lifecycle,
                result.state.tracer_lifecycle,
            )

    def test_terminal_lane_roundtrip_preserves_frozen_event(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        result, tile, spec = _run_segment(
            initial,
            start=0,
            steps=2,
            terminal=True,
        )
        self.assertEqual(result.state.tracer_lifecycle.tolist(), [1, 0])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            loaded = _load(root, initial, record)
            assert loaded is not None
            self.assertEqual(len(loaded.state.removal_records), 1)
            removal = loaded.state.removal_records[0]
            self.assertEqual(removal.lifecycle_code, REMOVED_SOURCE_RADIUS)
            self.assertEqual(
                removal.content_sha256,
                result.state.removal_records[0].content_sha256,
            )
            np.testing.assert_array_equal(
                CP.asnumpy(
                    loaded.state.dynamics_state.tracer_relative_positions_AU[0]
                ),
                removal.event.relative_position_AU,
            )
            np.testing.assert_array_equal(
                CP.asnumpy(
                    loaded.state.dynamics_state.tracer_relative_velocities_AU_per_year[
                        0
                    ]
                ),
                removal.event.relative_velocity_AU_per_year,
            )

    def test_six_worker_route_policy_survives_checkpoint_replay(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        result, tile, spec = _run_segment(
            initial,
            start=0,
            steps=1,
            terminal=True,
            route_worker_count=6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
                route_worker_count=6,
            )
            loaded = _load(root, initial, record)
            self.assertEqual(record["route_worker_count"], 6)
            self.assertEqual(
                loaded.state.removal_records[0].content_sha256,
                result.state.removal_records[0].content_sha256,
            )

    def test_nonterminal_route_witness_roundtrips_and_replays(self):
        initial = initialize_restricted_wh_lifecycle(_state(4))
        tile = _tile(
            start=0,
            node_count=2,
            encounter_radius=0.997,
            acceleration_bound=100.0,
        )
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        result = advance_cuda_restricted_wh_lifecycle(
            initial,
            tile,
            spec,
            route_worker_count=6,
        )
        self.assertEqual(len(result.route_records), 1)
        self.assertEqual(result.new_removal_records, ())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
                route_worker_count=6,
            )
            loaded = _load(root, initial, record)
            self.assertEqual(len(record["result"]["route_records"]), 1)
            self.assertEqual(
                record["result"]["route_records"][0][
                    "route_result"
                ]["source_segment_sha256"],
                result.route_records[0].route_result.source_segment_sha256,
            )
            self.assertTrue(
                bool(
                    CP.array_equal(
                        loaded.state.dynamics_state.tracer_relative_positions_AU,
                        result.state.dynamics_state.tracer_relative_positions_AU,
                    )
                )
            )

    def test_two_segment_chain_selects_numeric_latest_and_resumes(self):
        initial = initialize_restricted_wh_lifecycle(_state(2))
        first, tile0, spec0 = _run_segment(initial, start=0, steps=2)
        second, tile1, spec1 = _run_segment(first.state, start=2, steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            row1000 = _dump(
                root,
                index=1000,
                genesis_state=initial,
                input_state=first.state,
                result=second,
                tile=tile1,
                spec=spec1,
                predecessor=row0["checkpoint_content_sha256"],
            )
            loaded = _load(root, initial, row1000)
            assert loaded is not None
            self.assertEqual(loaded.checkpoint_index, 1000)
            self.assertEqual(
                loaded.checkpoint_content_sha256,
                row1000["checkpoint_content_sha256"],
            )
            self.assertEqual(loaded.state.dynamics_state.global_source_step, 4)
            self.assertTrue(
                bool(
                    CP.array_equal(
                        loaded.state.dynamics_state.tracer_relative_positions_AU,
                        second.state.dynamics_state.tracer_relative_positions_AU,
                    )
                )
            )

    def test_terminal_removal_delta_reconstructs_across_successor(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        first, tile0, spec0 = _run_segment(
            initial,
            start=0,
            steps=1,
            terminal=True,
        )
        second, tile1, spec1 = _run_segment(
            first.state,
            start=1,
            steps=1,
            terminal=True,
        )
        self.assertEqual(len(first.new_removal_records), 1)
        self.assertEqual(second.new_removal_records, ())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            row1 = _dump(
                root,
                index=1,
                genesis_state=initial,
                input_state=first.state,
                result=second,
                tile=tile1,
                spec=spec1,
                predecessor=row0["checkpoint_content_sha256"],
            )
            loaded = _load(root, initial, row1)
            assert loaded is not None
            self.assertEqual(len(loaded.state.removal_records), 1)
            self.assertEqual(
                loaded.state.removal_records[0].content_sha256,
                first.state.removal_records[0].content_sha256,
            )
            records = sorted(_namespace(root).glob("checkpoint_*.json"))
            first_record = json.loads(records[0].read_text(encoding="utf-8"))
            second_record = json.loads(records[1].read_text(encoding="utf-8"))
            self.assertEqual(first_record["genesis_removal_records"], [])
            self.assertIsNone(second_record["genesis_removal_records"])
            self.assertEqual(
                len(first_record["result"]["new_removal_records"]),
                1,
            )
            self.assertEqual(
                second_record["result"]["new_removal_records"],
                [],
            )

    def test_wrong_input_and_wrong_predecessor_reject_before_publish(self):
        initial = initialize_restricted_wh_lifecycle(_state(2))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        changed_positions = initial.dynamics_state.tracer_relative_positions_AU.copy()
        changed_positions[0, 0] += 1.0e-9
        wrong_input = replace(
            initial,
            dynamics_state=replace(
                initial.dynamics_state,
                tracer_relative_positions_AU=changed_positions,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RestrictedWHCheckpointContractError):
                _dump(
                    root,
                    index=0,
                    genesis_state=initial,
                    input_state=wrong_input,
                    result=result,
                    tile=tile,
                    spec=spec,
                )
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            next_result, next_tile, next_spec = _run_segment(
                result.state,
                start=1,
                steps=1,
            )
            with self.assertRaises(RestrictedWHCheckpointContractError):
                _dump(
                    root,
                    index=1,
                    genesis_state=initial,
                    input_state=result.state,
                    result=next_result,
                    tile=next_tile,
                    spec=next_spec,
                    predecessor="1" * 64,
                )
            self.assertEqual(
                len(list(_namespace(root).glob("checkpoint_*.json"))),
                1,
            )
            self.assertEqual(row0["checkpoint_index"], 0)

    def test_create_only_record_and_namespace_file_fail_closed(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            with self.assertRaises((FileExistsError, RestrictedWHCheckpointContractError)):
                _dump(
                    root,
                    index=0,
                    genesis_state=initial,
                    input_state=initial,
                    result=result,
                    tile=tile,
                    spec=spec,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = _namespace(root)
            namespace.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _dump(
                    root,
                    index=0,
                    genesis_state=initial,
                    input_state=initial,
                    result=result,
                    tile=tile,
                    spec=spec,
                )

    def test_orphan_array_is_tolerated_but_unknown_artifact_rejects(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            namespace = _namespace(root)
            orphan = namespace / (
                "checkpoint_00000000000000000099.array-"
                + "a" * 64
                + ".npz"
            )
            orphan.write_bytes(b"crash orphan")
            loaded = _load(root, initial, row)
            assert loaded is not None
            self.assertEqual(
                loaded.checkpoint_content_sha256,
                row["checkpoint_content_sha256"],
            )
            hidden = namespace / ".unexpected"
            hidden.write_text("hidden", encoding="utf-8")
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, row)
            hidden.unlink()
            (namespace / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, row)

    def test_record_array_and_rehashed_claim_tampering_fail(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            record_path = _namespace(root) / "checkpoint_00000000000000000000.json"
            _rewrite_record(
                record_path,
                lambda record: record["claim_ceiling"]["claims"].__setitem__(
                    "science_result", True
                ),
            )
            tampered = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, tampered)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            record_path = _namespace(root) / "checkpoint_00000000000000000000.json"
            _rewrite_record(
                record_path,
                lambda record: record["claim_ceiling"]["claims"].__setitem__(
                    "science_result", 0
                ),
            )
            tampered = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, tampered)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            array_path = _namespace(root) / row["array_file"]
            with array_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, row)

    def test_rehashed_runtime_result_tampering_is_caught_by_replay(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            record_path = _namespace(root) / "checkpoint_00000000000000000000.json"

            def mutate(record):
                record["result"]["runtime_identity"]["route_worker_count"] = 31
                segment = {
                    "input_state_sha256": record["input_state_sha256"],
                    "state_sha256": record["state_sha256"],
                    "lineage_sha256": record["lineage_sha256"],
                    "source_tile_sha256": record["source_tile_metadata"][
                        "content_sha256"
                    ],
                    "spec": record["spec"],
                    "route_policy": record["route_policy"],
                    "route_worker_count": record["route_worker_count"],
                    "result": record["result"],
                }
                record["segment_sha256"] = checkpoint_module._segment_sha256(
                    segment
                )

            _rewrite_record(record_path, mutate)
            tampered = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, tampered)

    def test_append_replays_and_verifies_predecessor_before_publication(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        first, tile0, spec0 = _run_segment(initial, start=0, steps=1)
        second, tile1, spec1 = _run_segment(first.state, start=1, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            array_path = _namespace(root) / row0["array_file"]
            with array_path.open("ab") as stream:
                stream.write(b"tampered predecessor")
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _dump(
                    root,
                    index=1,
                    genesis_state=initial,
                    input_state=first.state,
                    result=second,
                    tile=tile1,
                    spec=spec1,
                    predecessor=row0["checkpoint_content_sha256"],
                )
            self.assertFalse(
                (_namespace(root) / "checkpoint_00000000000000000001.json").exists()
            )

    def test_external_head_anchor_rejects_tail_rollback(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        first, tile0, spec0 = _run_segment(initial, start=0, steps=1)
        second, tile1, spec1 = _run_segment(first.state, start=1, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            row1 = _dump(
                root,
                index=1,
                genesis_state=initial,
                input_state=first.state,
                result=second,
                tile=tile1,
                spec=spec1,
                predecessor=row0["checkpoint_content_sha256"],
            )
            (_namespace(root) / "checkpoint_00000000000000000001.json").unlink()
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, row1)

    def test_checkpoint_index_must_fit_canonical_filename(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RestrictedWHCheckpointContractError):
                _dump(
                    root,
                    index=10**20,
                    genesis_state=initial,
                    input_state=initial,
                    result=result,
                    tile=tile,
                    spec=spec,
                )
            self.assertEqual(list(_namespace(root).glob("checkpoint_*.json")), [])

    def test_rehashed_float32_minimum_array_is_rejected(self):
        initial = initialize_restricted_wh_lifecycle(_source_contact_state())
        result, tile, spec = _run_segment(
            initial,
            start=0,
            steps=1,
            terminal=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            namespace = _namespace(root)
            record_path = namespace / "checkpoint_00000000000000000000.json"
            tampered = _replace_minimum_with_float32(namespace, record_path)
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, tampered)

    def test_successor_requires_exact_source_boundary_and_lineage(self):
        initial = initialize_restricted_wh_lifecycle(
            replace(_state(1), source_ids=("Sun", "Jupiter"))
        )
        tile0 = _tile(start=0, node_count=2, secondary=True)
        spec0 = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        first = advance_cuda_restricted_wh_lifecycle(
            initial,
            tile0,
            spec0,
            route_worker_count=1,
        )
        canonical_tile1 = _tile(start=1, node_count=2, secondary=True)
        changed_positions = canonical_tile1.source_relative_positions_AU.copy()
        changed_positions[0, 1, 0] += 1.0e-6
        discontinuous_tile1 = replace(
            canonical_tile1,
            source_relative_positions_AU=changed_positions,
            content_sha256=None,
        )
        spec1 = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        second = advance_cuda_restricted_wh_lifecycle(
            first.state,
            discontinuous_tile1,
            spec1,
            route_worker_count=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            with self.assertRaises(RestrictedWHCheckpointContractError):
                _dump(
                    root,
                    index=1,
                    genesis_state=initial,
                    input_state=first.state,
                    result=second,
                    tile=discontinuous_tile1,
                    spec=spec1,
                    predecessor=row0["checkpoint_content_sha256"],
                )

        changed_gm = canonical_tile1.source_gm_AU3_per_year2.copy()
        changed_gm[1] = np.nextafter(changed_gm[1], np.inf)
        changed_lineage_tile1 = replace(
            canonical_tile1,
            source_gm_AU3_per_year2=changed_gm,
            content_sha256=None,
        )
        changed_result = advance_cuda_restricted_wh_lifecycle(
            first.state,
            changed_lineage_tile1,
            spec1,
            route_worker_count=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            with self.assertRaises(RestrictedWHCheckpointContractError):
                _dump(
                    root,
                    index=1,
                    genesis_state=initial,
                    input_state=first.state,
                    result=changed_result,
                    tile=changed_lineage_tile1,
                    spec=spec1,
                    predecessor=row0["checkpoint_content_sha256"],
                )

    def test_checkpoint_runtime_identity_tamper_is_rejected(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        result, tile, spec = _run_segment(initial, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            record_path = _namespace(root) / "checkpoint_00000000000000000000.json"

            def mutate(record):
                record["checkpoint_runtime_identity"][
                    "python_version"
                ] = "forged-runtime"

            _rewrite_record(record_path, mutate)
            tampered = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, initial, tampered)

    def test_rehashed_first_input_must_equal_external_genesis(self):
        expected_genesis = initialize_restricted_wh_lifecycle(_state(1))
        changed_positions = (
            expected_genesis.dynamics_state.tracer_relative_positions_AU.copy()
        )
        changed_positions[0, 0] += 1.0e-9
        actual_input = replace(
            expected_genesis,
            dynamics_state=replace(
                expected_genesis.dynamics_state,
                tracer_relative_positions_AU=changed_positions,
            ),
        )
        result, tile, spec = _run_segment(actual_input, start=0, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dump(
                root,
                index=0,
                genesis_state=actual_input,
                input_state=actual_input,
                result=result,
                tile=tile,
                spec=spec,
            )
            record_path = _namespace(root) / "checkpoint_00000000000000000000.json"
            expected_digest = restricted_wh_lifecycle_state_sha256(
                expected_genesis,
                _backend(),
            )
            _rewrite_record(
                record_path,
                lambda record: record.__setitem__(
                    "genesis_state_sha256", expected_digest
                ),
            )
            tampered = json.loads(record_path.read_text(encoding="utf-8"))
            with self.assertRaises(RestrictedWHCheckpointIntegrityError):
                _load(root, expected_genesis, tampered)

    def test_source_boundary_hash_accepts_large_python_lattice_integer(self):
        tile = _tile(start=2**64, node_count=2)
        digest = checkpoint_module._source_boundary_sha256(tile, 1)
        self.assertEqual(len(digest), 64)

    def test_integer_valued_time_metadata_roundtrips_canonically(self):
        dynamics = _state(1)
        positions = dynamics.tracer_relative_positions_AU.copy()
        velocities = dynamics.tracer_relative_velocities_AU_per_year.copy()
        positions[0, 0] = 100.0
        velocities[0, 1] /= 10.0
        dynamics = replace(
            dynamics,
            epoch_origin_years=0,
            source_step_years=1,
            tracer_relative_positions_AU=positions,
            tracer_relative_velocities_AU_per_year=velocities,
        )
        initial = initialize_restricted_wh_lifecycle(dynamics)
        tile = replace(
            _tile(
                start=0,
                node_count=2,
                step=1,
                acceleration_bound=1.0,
            ),
            epoch_origin_years=0,
            content_sha256=None,
        )
        spec = FixedStepCUDARestrictedWHSpec(1, 1, _backend())
        result = advance_cuda_restricted_wh_lifecycle(
            initial,
            tile,
            spec,
            route_worker_count=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=result,
                tile=tile,
                spec=spec,
            )
            loaded = _load(root, initial, record)
            self.assertEqual(loaded.state.dynamics_state.epoch_origin_years, 0.0)
            self.assertEqual(loaded.state.dynamics_state.source_step_years, 1.0)

    def test_namespace_lock_prevents_concurrent_append_fork(self):
        initial = initialize_restricted_wh_lifecycle(_state(1))
        first, tile0, spec0 = _run_segment(initial, start=0, steps=1)
        second, tile1, spec1 = _run_segment(first.state, start=1, steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row0 = _dump(
                root,
                index=0,
                genesis_state=initial,
                input_state=initial,
                result=first,
                tile=tile0,
                spec=spec0,
            )
            barrier = threading.Barrier(2)

            def contender(index):
                barrier.wait()
                try:
                    return (
                        "success",
                        _dump(
                            root,
                            index=index,
                            genesis_state=initial,
                            input_state=first.state,
                            result=second,
                            tile=tile1,
                            spec=spec1,
                            predecessor=row0["checkpoint_content_sha256"],
                        ),
                    )
                except RestrictedWHCheckpointContractError as exc:
                    return ("rejected", exc)

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(contender, (1, 2)))
            successes = [value for status, value in outcomes if status == "success"]
            rejections = [value for status, value in outcomes if status == "rejected"]
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(rejections), 1)
            self.assertEqual(
                len(list(_namespace(root).glob("checkpoint_*.json"))),
                2,
            )
            loaded = _load(root, initial, successes[0])
            self.assertEqual(
                loaded.checkpoint_content_sha256,
                successes[0]["checkpoint_content_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
