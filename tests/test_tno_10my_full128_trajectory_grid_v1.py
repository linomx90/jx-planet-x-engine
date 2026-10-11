from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_10my_full128_trajectory_grid_v1.py"


def _load():
    name = f"_jx_tno_full128_grid_test_{sys.flags.optimize}"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("source module spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = _load()


class Full128TrajectoryGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = m.GridGeometry(
            batch_count=1,
            total_steps=8,
            shard_steps=4,
            b22_interval_steps=2,
            short_interval_steps=1,
            short_final_step=4,
            massive_count=2,
            tracer_count=3,
        )

    def _state(self, step: int) -> bytes:
        rows = tuple(
            tuple(float(step * 100_000 + body * 10 + component) for component in range(6))
            for body in range(self.geometry.body_count)
        )
        return m.state_record_v1(rows, self.geometry)

    def _life(self, step: int) -> bytes:
        codes = (m.ACTIVE, m.REMOVED_SOURCE_RADIUS if step >= 4 else m.ACTIVE,
                 m.REMOVED_OUTER_BOUNDARY if step >= 6 else m.ACTIVE)
        return m.lifecycle_record_v1(codes, self.geometry)

    def _locators(self, root: Path) -> dict[str, object]:
        result = {}
        for index, role in enumerate(m.RECEIPT_ROLES):
            result[role] = m.write_synthetic_receipt_v1(
                root, f"receipt-{index}-{role}.json", role, f"receipt {role}"
            )
        return result

    def _write_synthetic_shards(self, evidence: Path) -> None:
        locators = self._locators(evidence)
        for shard in range(self.geometry.shard_count):
            checkpoints = tuple(
                (step, self._state(step), self._life(step))
                for step in self.geometry.expected_steps(shard)
            )
            m.write_shard_v1(
                evidence,
                batch_order=0,
                shard_index=shard,
                checkpoints=checkpoints,
                receipt_locators=locators,
                geometry=self.geometry,
                test_only_geometry=True,
            )

    def test_production_geometry_identity_and_protocol_claim_ceiling(self) -> None:
        geometry = m.PRODUCTION_GEOMETRY
        self.assertEqual(geometry.state_record_bytes, 6_432)
        self.assertEqual(geometry.lifecycle_record_bytes, 129)
        self.assertEqual(geometry.shard_count, 1_000)
        self.assertEqual(geometry.b22_node_count, 50_001)
        self.assertEqual(geometry.short_node_count, 10_001)
        self.assertEqual(sum(geometry.output_sizes().values()), 4_743_277_848)
        protocol = m.build_protocol()
        self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
        self.assertFalse(protocol["claims"]["dynamics_executed"])
        self.assertFalse(protocol["claims"]["ten_myr_trajectory_acquired"])
        self.assertFalse(protocol["claims"]["classification_generated"])
        self.assertFalse(protocol["claims"]["science_result"])
        self.assertEqual(protocol["implementation"]["cli_surface"], ["prepare", "validate-prepared"])
        self.assertEqual(
            protocol["implementation"]["filesystem_trust_boundary"],
            m.FILESYSTEM_TRUST_BOUNDARY,
        )
        self.assertEqual(
            protocol["evidence"]["filesystem_trust_boundary"],
            "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        )

    def test_identity_order_is_nominal_then_all_128_clones(self) -> None:
        nominal = m.tracer_identity_v1(3, 0)
        first = m.tracer_identity_v1(3, 1)
        last = m.tracer_identity_v1(3, 128)
        self.assertEqual((nominal["sample_kind"], nominal["clone_index"]), ("NOMINAL", None))
        self.assertEqual((first["sample_kind"], first["clone_index"]), ("COVARIANCE_CLONE", 0))
        self.assertEqual((last["sample_kind"], last["clone_index"]), ("COVARIANCE_CLONE", 127))
        self.assertEqual(last["stage0_state_order"], 3 * 129 + 128)
        with self.assertRaises(m.TrajectoryGridError):
            m.tracer_identity_v1(False, 0)
        with self.assertRaises(m.TrajectoryGridError):
            m.tracer_identity_v1(0, True)

    def test_state_and_lifecycle_codecs_are_exact_and_terminal_states_stay_finite(self) -> None:
        state = self._state(6)
        life = self._life(6)
        self.assertEqual(len(state), self.geometry.state_record_bytes)
        self.assertEqual(len(life), 3)
        self.assertEqual(m.decode_lifecycle_record_v1(life, self.geometry), (0, 1, 2))
        self.assertTrue(all(math.isfinite(value) for row in m.decode_state_record_v1(state, self.geometry) for value in row))
        with self.assertRaises(m.TrajectoryGridError):
            m.lifecycle_record_v1((False, 1, 2), self.geometry)
        with self.assertRaises(m.TrajectoryGridError):
            m.lifecycle_record_v1((0, 1, 3), self.geometry)
        rows = [list(row) for row in m.decode_state_record_v1(state, self.geometry)]
        rows[0][0] = float("nan")
        with self.assertRaises(m.TrajectoryGridError):
            m.state_record_v1(rows, self.geometry)
        rows[0][0] = 1
        with self.assertRaises(m.TrajectoryGridError):
            m.state_record_v1(rows, self.geometry)

    def test_node_digest_binds_all_state_lifecycle_and_identity_bytes(self) -> None:
        state = self._state(2)
        life = self._life(2)
        digest = m.node_digest_v1(0, 2, state, life, self.geometry)
        self.assertEqual(len(digest), 32)
        changed = bytearray(state)
        changed[-1] ^= 1
        self.assertNotEqual(digest, m.node_digest_v1(0, 2, bytes(changed), life, self.geometry))
        changed_life = bytearray(life)
        changed_life[-1] = m.REMOVED_OUTER_BOUNDARY
        self.assertNotEqual(digest, m.node_digest_v1(0, 2, state, bytes(changed_life), self.geometry))
        with self.assertRaises(m.TrajectoryGridError):
            m.node_digest_v1(False, 2, state, life, self.geometry)

    def test_full_record_projection_bitwise_replays_all_v3_prefix_rows(self) -> None:
        prefix = (ROOT / "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin").read_bytes()
        batch = 0
        checkpoints = []
        for index, year in enumerate(m.V3_CHECKPOINT_YEARS):
            offset = (batch * len(m.V3_CHECKPOINT_YEARS) + index) * m.V3_RECORD_BYTES
            legacy = prefix[offset:offset + m.V3_RECORD_BYTES]
            legacy_rows = tuple(
                tuple(struct.unpack("<84d", legacy)[row * 6:(row + 1) * 6])
                for row in range(14)
            )
            rows = [[0.0] * 6 for _ in range(m.PRODUCTION_GEOMETRY.body_count)]
            for row in range(5):
                rows[row] = list(legacy_rows[row])
            for projected_index, lane in enumerate(m.HOLDOUT_LANES):
                rows[5 + lane] = list(legacy_rows[5 + projected_index])
            state = m.state_record_v1(tuple(tuple(row) for row in rows))
            life = m.lifecycle_record_v1((m.ACTIVE,) * 129)
            checkpoints.append((year, state, life))
        receipt = m.validate_first_1000_year_projection_v1(batch, tuple(checkpoints))
        self.assertTrue(receipt["passed"])
        self.assertEqual(len(receipt["checkpoint_sha256"]), 21)
        year, state, life = checkpoints[7]
        corrupted = bytearray(state)
        projected_row_offset = (5 + m.HOLDOUT_LANES[0]) * 48
        corrupted[projected_row_offset] ^= 1
        bad = list(checkpoints)
        bad[7] = (year, bytes(corrupted), life)
        with self.assertRaises(m.TrajectoryGridError):
            m.validate_first_1000_year_projection_v1(batch, tuple(bad))
        statuses = bytearray(life)
        statuses[m.HOLDOUT_LANES[-1]] = m.REMOVED_SOURCE_RADIUS
        bad[7] = (year, state, bytes(statuses))
        with self.assertRaises(m.TrajectoryGridError):
            m.validate_first_1000_year_projection_v1(batch, tuple(bad))

    def test_prepare_and_validate_prepared_are_create_only_and_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            protocol = m.prepare(root)
            self.assertEqual(set(path.name for path in root.iterdir()), {"PROTOCOL.json"})
            self.assertEqual(m.validate_prepared(root), protocol)
            with self.assertRaises(m.TrajectoryGridError):
                m.prepare(root)
            mutated = json.loads((root / "PROTOCOL.json").read_text())
            mutated["geometry"]["batch_count"] = True
            mutated["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, mutated)
            (root / "PROTOCOL.json").write_bytes(m._canonical_json(mutated))
            with self.assertRaises(m.TrajectoryGridError):
                m.validate_prepared(root)

    def test_created_root_and_shard_inode_substitution_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "prepared"
            real_open = os.open
            root_attacked = False

            def substitute_root(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal root_attacked
                if (
                    not root_attacked and path == root.name and dir_fd is not None
                    and flags & getattr(os, "O_DIRECTORY", 0)
                ):
                    os.rename(path, f"{path}-original", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                    os.mkdir(path, 0o700, dir_fd=dir_fd)
                    root_attacked = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(m.os, "open", side_effect=substitute_root):
                with self.assertRaisesRegex(m.TrajectoryGridError, "substituted"):
                    m.prepare(root)
            self.assertTrue(root_attacked)

        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            locators = self._locators(evidence)
            checkpoints = tuple(
                (step, self._state(step), self._life(step))
                for step in self.geometry.expected_steps(0)
            )
            shard_name = m._shard_name(0, 0)
            real_open = os.open
            shard_attacked = False

            def substitute_shard(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal shard_attacked
                if (
                    not shard_attacked and path == shard_name and dir_fd is not None
                    and flags & getattr(os, "O_DIRECTORY", 0)
                ):
                    os.rename(path, f"{path}-original", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                    os.mkdir(path, 0o700, dir_fd=dir_fd)
                    shard_attacked = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(m.os, "open", side_effect=substitute_shard):
                with self.assertRaisesRegex(m.TrajectoryGridError, "substituted"):
                    m.write_shard_v1(
                        evidence,
                        batch_order=0,
                        shard_index=0,
                        checkpoints=checkpoints,
                        receipt_locators=locators,
                        geometry=self.geometry,
                        test_only_geometry=True,
                    )
            self.assertTrue(shard_attacked)

    def test_shard_writer_validator_and_result_last_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            evidence = base / "evidence"
            evidence.mkdir()
            self._write_synthetic_shards(evidence)
            for shard in range(self.geometry.shard_count):
                observed = m.validate_shard_v1(
                    evidence, batch_order=0, shard_index=shard,
                    geometry=self.geometry, test_only_geometry=True,
                )
                self.assertEqual(observed["shard_index"], shard)
            output = base / "output"
            m.prepare(output)
            writes = []
            original_write = m._write_create_only
            def tracked_write(directory_fd, name, payload):
                writes.append(name)
                return original_write(directory_fd, name, payload)
            m._write_create_only = tracked_write
            try:
                result = m.finalize_shards_v1(
                    evidence, output, geometry=self.geometry, test_only_geometry=True
                )
            finally:
                m._write_create_only = original_write
            self.assertEqual(writes[-1], "RESULT.json")
            self.assertEqual(set(path.name for path in output.iterdir()), set(m.FINAL_ROSTER))
            self.assertFalse(result["claims"]["dynamics_executed"])
            self.assertFalse(result["claims"]["science_result"])
            self.assertEqual(
                m.validate_finalized_v1(
                    evidence, output, geometry=self.geometry, test_only_geometry=True
                ),
                result,
            )
            expected = self.geometry.output_sizes()
            for name, length in expected.items():
                self.assertEqual((output / name).stat().st_size, length)

    def test_relabel_bool_and_receipt_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            self._write_synthetic_shards(evidence)
            manifest_path = evidence / m._shard_name(0, 0) / "MANIFEST.json"
            original = json.loads(manifest_path.read_text())
            for key, value in (("batch_order", False), ("object_id", "jpl-forged")):
                forged = json.loads(json.dumps(original))
                forged[key] = value
                forged["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, forged)
                manifest_path.write_bytes(m._canonical_json(forged))
                with self.assertRaises(m.TrajectoryGridError):
                    m.validate_shard_v1(
                        evidence, batch_order=0, shard_index=0,
                        geometry=self.geometry, test_only_geometry=True,
                    )
            forged = json.loads(json.dumps(original))
            forged["receipt_locators"]["runtime"]["role"] = "route"
            forged["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, forged)
            manifest_path.write_bytes(m._canonical_json(forged))
            with self.assertRaises(m.TrajectoryGridError):
                m.validate_shard_v1(
                    evidence, batch_order=0, shard_index=0,
                    geometry=self.geometry, test_only_geometry=True,
                )

            forged = json.loads(json.dumps(original))
            forged["claims"]["trajectory_bytes_retained"] = 1
            forged["claims"]["dynamics_qualified"] = 0
            forged["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, forged)
            manifest_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.TrajectoryGridError, "claim ceiling"):
                m.validate_shard_v1(
                    evidence, batch_order=0, shard_index=0,
                    geometry=self.geometry, test_only_geometry=True,
                )

    def test_final_manifest_integer_claim_aliases_fail_closed_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            evidence = base / "evidence"
            evidence.mkdir()
            self._write_synthetic_shards(evidence)
            output = base / "output"
            m.prepare(output)
            m.finalize_shards_v1(
                evidence, output,
                geometry=self.geometry, test_only_geometry=True,
            )
            manifest_path = output / "MANIFEST.json"
            result_path = output / "RESULT.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["claims"]["trajectory_bytes_finalized"] = 1
            manifest["claims"]["science_result"] = 0
            manifest["content_sha256"] = m._domain_hash(
                m.FINAL_MANIFEST_DOMAIN, manifest
            )
            manifest_payload = m._canonical_json(manifest)
            manifest_path.write_bytes(manifest_payload)
            result = json.loads(result_path.read_text())
            result["manifest_content_sha256"] = manifest["content_sha256"]
            result["manifest_file_sha256"] = hashlib.sha256(
                manifest_payload
            ).hexdigest()
            result["content_sha256"] = m._domain_hash(
                m.FINAL_RESULT_DOMAIN, result
            )
            result_path.write_bytes(m._canonical_json(result))
            with self.assertRaisesRegex(m.TrajectoryGridError, "claim ceiling"):
                m.validate_finalized_v1(
                    evidence, output,
                    geometry=self.geometry, test_only_geometry=True,
                )

    def test_fragment_and_cross_grid_tamper_are_rederived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            self._write_synthetic_shards(evidence)
            shard = evidence / m._shard_name(0, 0)
            fragment = shard / "ML_SHORT_STATES.fragment"
            payload = bytearray(fragment.read_bytes())
            payload[0] ^= 1
            fragment.write_bytes(payload)
            manifest_path = shard / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["fragments"]["ML_SHORT_STATES.fragment"]["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, manifest)
            manifest_path.write_bytes(m._canonical_json(manifest))
            with self.assertRaises(m.TrajectoryGridError):
                m.validate_shard_v1(
                    evidence, batch_order=0, shard_index=0,
                    geometry=self.geometry, test_only_geometry=True,
                )

    def test_held_file_detects_hardlink_and_directory_entry_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "value.bin"
            target.write_bytes(b"original")
            parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            held = m._HeldFile(parent_fd, target.name, "hostile held file")
            link = root / "link.bin"
            os.link(target, link)
            with self.assertRaises(m.TrajectoryGridError):
                held.close()
            link.unlink()
            os.close(parent_fd)

            target.write_bytes(b"original")
            parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            held = m._HeldFile(parent_fd, target.name, "hostile replaced file")
            target.unlink()
            target.write_bytes(b"replacement")
            with self.assertRaises(m.TrajectoryGridError):
                held.close()
            os.close(parent_fd)

            held_root = root / "held-root"
            held_root.mkdir()
            moved_root = root / "held-root-old"
            with self.assertRaises(m.TrajectoryGridError):
                with m._held_directory(held_root, "hostile held directory"):
                    held_root.rename(moved_root)
                    held_root.mkdir()

    def test_terminal_lifecycle_reversion_is_rejected_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            locators = self._locators(evidence)
            checkpoints = []
            for step in self.geometry.expected_steps(0):
                codes = (0, 1 if step in (1, 2) else 0, 0)
                checkpoints.append((step, self._state(step), m.lifecycle_record_v1(codes, self.geometry)))
            with self.assertRaises(m.TrajectoryGridError):
                m.write_shard_v1(
                    evidence,
                    batch_order=0,
                    shard_index=0,
                    checkpoints=tuple(checkpoints),
                    receipt_locators=locators,
                    geometry=self.geometry,
                    test_only_geometry=True,
                )
            self.assertFalse((evidence / m._shard_name(0, 0) / "MANIFEST.json").exists())

    def test_receipt_locator_rejects_symlink_path_and_type_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locator = m.write_synthetic_receipt_v1(root, "runtime.json", "runtime", "runtime")
            parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                forged = dict(locator)
                forged["byte_length"] = True
                with self.assertRaises(m.TrajectoryGridError):
                    m.validate_receipt_locator_v1(parent_fd, forged, "runtime")
                target = root / "target.json"
                (root / "runtime.json").rename(target)
                (root / "runtime.json").symlink_to(target.name)
                with self.assertRaises((m.TrajectoryGridError, OSError)):
                    m.validate_receipt_locator_v1(parent_fd, locator, "runtime")
            finally:
                os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
