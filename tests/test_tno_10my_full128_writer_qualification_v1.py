from __future__ import annotations

import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/jx_tno_10my_full128_writer_qualification_v1.py"


def _load():
    name = f"_jx_tno_full128_writer_qualification_test_{sys.flags.optimize}"
    sys.modules.pop(name, None)
    payload = SOURCE.read_bytes()
    code = compile(
        payload,
        os.fspath(SOURCE),
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    module = types.ModuleType(name)
    module.__file__ = os.fspath(SOURCE)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


m = _load()


class Full128WriterQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        geometry = m.GridGeometry(
            batch_count=2,
            total_steps=8,
            shard_steps=4,
            b22_interval_steps=2,
            short_interval_steps=1,
            short_final_step=4,
            massive_count=2,
            tracer_count=3,
        )
        self.profile = m.WriterProfile("tiny-hostile-test", geometry, 4, False)

    def _prepared(self, base: Path, name: str = "qualified") -> Path:
        root = base / name
        m.prepare(root)
        return root

    def _run(self, base: Path, name: str = "qualified") -> tuple[Path, dict[str, object]]:
        root = self._prepared(base, name)
        result = m.qualify(root, profile=self.profile, test_only=True, unreserved_floor_bytes=0)
        return root, result

    def _attempt(self, root: Path) -> Path:
        return root.parent / m._attempt_name(root)

    def _first_manifest(self, evidence: Path) -> tuple[Path, dict[str, object]]:
        path = evidence / m._shard_name(0, 0) / "MANIFEST.json"
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_production_metadata_plan_is_exact_and_dry(self) -> None:
        plan = m.production_metadata_plan_v1()
        self.assertEqual(plan["shard_total"], 12_000)
        self.assertEqual(plan["receipt_total"], 60_000)
        self.assertEqual(plan["final_binary_bytes"], 4_743_277_848)
        self.assertEqual(plan["first_shard"]["shard_name"], "shard-b000-s0000")
        self.assertEqual(plan["last_shard"]["shard_name"], "shard-b011-s0999")
        self.assertLess(plan["estimated_digest_roster_bytes"], 100_000_000)
        reserve = plan["reservation"]
        self.assertEqual(reserve["final_logical_bytes"], 4_743_277_848)
        self.assertEqual(reserve["required_inodes"], 156_021)
        self.assertFalse(plan["claims"]["dynamics_executed"])
        self.assertFalse(plan["claims"]["ten_myr_execution_authorized"])

    def test_protocol_freezes_branch_grid_custody_archive_boundary_and_claim_ceiling(self) -> None:
        protocol = m.build_protocol()
        self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
        self.assertEqual(protocol["branch"], m.EXPECTED_BRANCH)
        self.assertEqual(protocol["preregistration_tag"], m.PREREGISTRATION_TAG)
        self.assertEqual(protocol["future_result_tag"], m.FUTURE_RESULT_TAG)
        self.assertEqual(protocol["custody"]["exact_production_receipt_count"], 60_000)
        self.assertEqual(protocol["storage"]["unreserved_floor_bytes"], 2 * 1024**3)
        self.assertEqual(
            protocol["pinned_trajectory_grid"]["source"]["sha256"],
            m.EXPECTED_GRID_SOURCE_SHA256,
        )
        self.assertTrue(protocol["archive_surface"]["live_revalidation_requires_local_large_payloads"])
        self.assertIn("not the large payload bytes", protocol["archive_surface"]["statement"])
        self.assertEqual(protocol["filesystem_trust_boundary"], m.FILESYSTEM_TRUST_BOUNDARY)
        self.assertEqual(
            protocol["production_git_authority"]["required_cli_inputs"],
            ["--externally-preregistered", "--prereg-commit", "--prereg-tag-object"],
        )
        self.assertEqual(
            protocol["production_git_authority"]["gate_order"],
            "VERIFY_BEFORE_STORAGE_RESERVATION_OR_ANY_ATTEMPT_WRITE",
        )
        self.assertEqual(protocol["claims"], m._claim_ceiling(False))

    def test_grid_executes_exact_hashed_source_bytes_and_ignores_cached_module_or_pyc_loader(self) -> None:
        module_name = "_jx_tno_full128_trajectory_grid_v1_for_writer_qualification"
        poisoned = object()
        sys.modules[module_name] = poisoned
        with mock.patch.object(
            importlib.util,
            "spec_from_file_location",
            side_effect=AssertionError("source/pyc loader must not be consulted"),
        ):
            loaded = m._load_grid()
        self.assertIsNot(loaded, poisoned)
        self.assertIs(sys.modules[module_name], loaded)
        self.assertEqual(
            m._geometry_record(loaded.PRODUCTION_GEOMETRY),
            m._geometry_record(m.PRODUCTION_GEOMETRY),
        )
        self.assertEqual(
            m._sha256(m.GRID_SOURCE_PATH.read_bytes()),
            m.EXPECTED_GRID_SOURCE_SHA256,
        )

    def test_production_git_authority_cannot_be_omitted_retyped_or_relabelled(self) -> None:
        with self.assertRaisesRegex(m.WriterQualificationError, "requires explicit external"):
            m._resolve_git_custody(
                m.PRODUCTION_PROFILE,
                externally_preregistered=False,
                prereg_commit=None,
                prereg_tag_object=None,
            )
        with self.assertRaises(m.WriterQualificationError):
            m._resolve_git_custody(
                m.PRODUCTION_PROFILE,
                externally_preregistered=1,
                prereg_commit="a" * 40,
                prereg_tag_object="b" * 40,
            )
        with self.assertRaisesRegex(m.WriterQualificationError, "test profile cannot"):
            m._resolve_git_custody(
                self.profile,
                externally_preregistered=True,
                prereg_commit="a" * 40,
                prereg_tag_object="b" * 40,
            )
        expected_files = m._expected_git_file_records()
        tracked = {
            path: {
                **record,
                "git_blob_oid": (str(index + 1) * 40),
                "committed_byte_identical": True,
            }
            for index, (path, record) in enumerate(expected_files.items())
        }
        record = {
            "schema": m.GIT_CUSTODY_SCHEMA,
            "mode": "EXTERNALLY_PREREGISTERED_LOCAL_GIT_CUSTODY",
            "externally_preregistered": True,
            "repository_root": ".",
            "object_format": "sha1",
            "branch": m.EXPECTED_BRANCH,
            "head_commit": "a" * 40,
            "head_tree": "b" * 40,
            "preregistration_tag": m.PREREGISTRATION_TAG,
            "preregistration_tag_object": "c" * 40,
            "preregistration_tag_peeled_commit": "a" * 40,
            "annotated_tag": True,
            "tag_object_byte_length": 128,
            "tag_object_payload_sha256": "d" * 64,
            "future_result_tag": m.FUTURE_RESULT_TAG,
            "future_result_tag_absent": True,
            "tracked_files": tracked,
            "status_porcelain_byte_length": 0,
            "status_porcelain_sha256": m._sha256(b""),
            "clean_bound_paths": True,
        }
        self.assertEqual(m._validate_git_custody_record(record, m.PRODUCTION_PROFILE), record)
        for field, replacement in (
            ("branch", "research/wrong"),
            ("externally_preregistered", 1),
            ("annotated_tag", False),
            ("future_result_tag_absent", False),
            ("clean_bound_paths", False),
        ):
            forged = dict(record)
            forged[field] = replacement
            with self.assertRaises(m.WriterQualificationError):
                m._validate_git_custody_record(forged, m.PRODUCTION_PROFILE)
        forged = {**record, "tracked_files": {path: dict(value) for path, value in tracked.items()}}
        forged["tracked_files"][m.PREREG_PROTOCOL_RELATIVE_PATH]["committed_byte_identical"] = 1
        with self.assertRaises(m.WriterQualificationError):
            m._validate_git_custody_record(forged, m.PRODUCTION_PROFILE)

    def test_production_git_gate_fails_before_prepare_or_physical_reservation(self) -> None:
        with (
            mock.patch.object(
                m,
                "_git_preregistration_custody",
                side_effect=m.WriterQualificationError("injected preregistration gate failure"),
            ) as git_gate,
            mock.patch.object(m, "validate_prepared") as prepared,
            mock.patch.object(m, "_initialize_attempt") as initialize,
            mock.patch.object(m.os, "posix_fallocate") as fallocate,
        ):
            with self.assertRaisesRegex(m.WriterQualificationError, "preregistration gate"):
                m.qualify(
                    m.DEFAULT_OUTPUT_ROOT,
                    externally_preregistered=True,
                    prereg_commit="a" * 40,
                    prereg_tag_object="b" * 40,
                )
        git_gate.assert_called_once_with("a" * 40, "b" * 40)
        prepared.assert_not_called()
        initialize.assert_not_called()
        fallocate.assert_not_called()

    def test_exact_type_sentinels_are_finite_deterministic_and_terminal(self) -> None:
        state = m._sentinel_state(self.profile, 1, 4)
        self.assertEqual(state, m._sentinel_state(self.profile, 1, 4))
        decoded = m.grid.decode_state_record_v1(state, self.profile.geometry)
        self.assertTrue(all(math.isfinite(value) for row in decoded for value in row))
        before = m.grid.decode_lifecycle_record_v1(m._sentinel_lifecycle(self.profile, 1, 4), self.profile.geometry)
        after = m.grid.decode_lifecycle_record_v1(m._sentinel_lifecycle(self.profile, 1, 8), self.profile.geometry)
        m.grid._require_terminal_transition(before, after)
        with self.assertRaises(m.WriterQualificationError):
            m._sentinel_state(self.profile, False, 0)
        with self.assertRaises(m.WriterQualificationError):
            m.WriterProfile("alias", self.profile.geometry, 4, 0)
        with self.assertRaises(m.WriterQualificationError):
            m.qualify(Path("irrelevant"), profile=self.profile, test_only=1)

    def test_prepare_is_create_only_one_file_and_production_cannot_relabel_or_move(self) -> None:
        with self.assertRaisesRegex(m.WriterQualificationError, "test-only profile"):
            m.qualify(
                m.DEFAULT_OUTPUT_ROOT,
                profile=self.profile,
                test_only=True,
                unreserved_floor_bytes=0,
            )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._prepared(base)
            self.assertEqual({path.name for path in root.iterdir()}, {"PROTOCOL.json"})
            self.assertEqual(m.validate_prepared(root)["schema"], m.PROTOCOL_SCHEMA)
            with self.assertRaises(m.WriterQualificationError):
                m.prepare(root)
            with self.assertRaises(m.WriterQualificationError):
                m.qualify(root, profile=m.PRODUCTION_PROFILE)
            self.assertEqual({path.name for path in root.iterdir()}, {"PROTOCOL.json"})

    def test_tiny_full_writer_streaming_replay_rosters_and_archive_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, result = self._run(Path(temporary))
            self.assertEqual({path.name for path in root.iterdir()}, set(m.SUCCESS_ROOT_ROSTER))
            evidence_names = {path.name for path in (root / "EVIDENCE").iterdir()}
            self.assertEqual(len(evidence_names), self.profile.shard_total + 4)
            self.assertIn("ATTEMPT.json", evidence_names)
            self.assertIn("STORAGE_RESERVATION.json", evidence_names)
            self.assertIn("FINALIZATION_PLAN.json", evidence_names)
            self.assertIn("FINALIZATION_STORAGE.json", evidence_names)
            self.assertEqual(
                {path.name for path in (root / "FINAL").iterdir()},
                {*m.FINAL_BINARY_NAMES, "SHARD_MANIFESTS.jsonl", "MANIFEST.json"},
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)
            self.assertEqual(m.validate_archive_surface(root, profile=self.profile, test_only=True), result)
            self.assertFalse(self._attempt(root).exists())
            self.assertEqual(result["digest_roster"]["record_count"], self.profile.shard_total + 1)
            self.assertEqual(result["digest_roster"]["shard_record_count"], self.profile.shard_total)
            self.assertEqual(result["claims"], m._claim_ceiling(False))
            self.assertFalse(result["archive_surface"]["large_payload_bytes_in_git_custody"])
            self.assertLess((root / "DIGEST_ROSTER.jsonl").stat().st_size, 100_000_000)

    def test_crash_after_committed_shards_discards_only_current_partial_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=2, unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            evidence = attempt / "EVIDENCE"
            committed = sorted(path.name for path in evidence.iterdir() if path.name.startswith("shard-"))
            self.assertEqual(committed, [m._shard_name(0, 0), m._shard_name(0, 1)])
            current_partial = evidence / m._partial_shard_name(1, 0)
            current_partial.mkdir()
            (current_partial / "abandoned").write_bytes(b"partial")
            result = m.qualify(
                root, profile=self.profile, test_only=True, resume=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)
            self.assertFalse(attempt.exists())

    def test_finalized_attempt_is_validated_and_published_without_o_excl_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "after finalized attempt"):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_finalize=True, unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            self.assertTrue((attempt / "FINAL/MANIFEST.json").is_file())
            self.assertTrue((attempt / "DIGEST_ROSTER.jsonl").is_file())
            self.assertEqual((attempt / "RESERVATION.bin").stat().st_size, 0)
            plan = json.loads((attempt / "EVIDENCE/FINALIZATION_PLAN.json").read_text(encoding="utf-8"))
            reconciliation = json.loads(
                (attempt / "EVIDENCE/FINALIZATION_STORAGE.json").read_text(encoding="utf-8")
            )
            self.assertGreater(plan["stream_allocation_budget_bytes"], 0)
            self.assertEqual(reconciliation["reservation_byte_length_after_release"], 0)
            self.assertEqual(reconciliation["reservation_allocated_bytes_after_release"], 0)
            self.assertGreater(
                sum(row["allocated_bytes"] for row in reconciliation["streams"].values()),
                0,
            )
            self.assertGreaterEqual(
                reconciliation["statvfs_after_streams"]["available_bytes"],
                plan["unreserved_floor_bytes"] + plan["tail_allocation_budget_bytes"],
            )
            result = m.qualify(
                root, profile=self.profile, test_only=True, resume=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)

    def test_pre_manifest_finalizer_crash_discards_only_uncommitted_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=self.profile.shard_total,
                    unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            (attempt / "FINAL/SHARD_MANIFESTS.jsonl").write_bytes(b"incomplete\n")
            (attempt / "FINAL/.MANIFEST.json.partial").write_bytes(b"incomplete\n")
            (attempt / "DIGEST_ROSTER.jsonl").write_bytes(b"incomplete\n")
            result = m.qualify(
                root, profile=self.profile, test_only=True, resume=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)

    def test_digest_parent_is_durable_before_final_manifest_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root,
                    profile=self.profile,
                    test_only=True,
                    fault_after_shards=self.profile.shard_total,
                    unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            identity = (attempt.stat().st_dev, attempt.stat().st_ino)
            events: list[str] = []
            original_fsync = m.os.fsync
            original_publish = m._write_atomic_noreplace

            def observe_fsync(fd):
                metadata = os.fstat(fd)
                if (metadata.st_dev, metadata.st_ino) == identity:
                    events.append("digest-parent-fsync")
                return original_fsync(fd)

            def observe_publish(parent_fd, name, payload):
                if name == "MANIFEST.json":
                    events.append("final-manifest-commit")
                return original_publish(parent_fd, name, payload)

            with (
                mock.patch.object(m.os, "fsync", side_effect=observe_fsync),
                mock.patch.object(m, "_write_atomic_noreplace", side_effect=observe_publish),
            ):
                m._finalize_attempt(attempt, self.profile)
            self.assertIn("digest-parent-fsync", events)
            self.assertIn("final-manifest-commit", events)
            self.assertLess(
                events.index("digest-parent-fsync"),
                events.index("final-manifest-commit"),
            )

    def test_digest_parent_fsync_failure_prevents_final_manifest_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root,
                    profile=self.profile,
                    test_only=True,
                    fault_after_shards=self.profile.shard_total,
                    unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            identity = (attempt.stat().st_dev, attempt.stat().st_ino)
            original_fsync = m.os.fsync
            observed = False

            def fail_digest_parent(fd):
                nonlocal observed
                metadata = os.fstat(fd)
                if (metadata.st_dev, metadata.st_ino) == identity:
                    observed = True
                    raise m.WriterQualificationError("injected digest-parent fsync failure")
                return original_fsync(fd)

            with mock.patch.object(m.os, "fsync", side_effect=fail_digest_parent):
                with self.assertRaisesRegex(m.WriterQualificationError, "digest-parent fsync"):
                    m._finalize_attempt(attempt, self.profile)
            self.assertTrue(observed)
            self.assertFalse((attempt / "FINAL/MANIFEST.json").exists())
            finalized = m._finalize_or_recover_attempt(attempt, self.profile)
            self.assertEqual(finalized["manifest"]["schema"], m.FINAL_SCHEMA)

    def test_post_finalization_floor_is_enforced_before_result_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root,
                    profile=self.profile,
                    test_only=True,
                    fault_after_shards=self.profile.shard_total,
                    unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            manifest_committed = False
            original_publish = m._write_atomic_noreplace
            original_statvfs = m._statvfs_receipt

            def observe_publish(parent_fd, name, payload):
                nonlocal manifest_committed
                value = original_publish(parent_fd, name, payload)
                if name == "MANIFEST.json":
                    manifest_committed = True
                return value

            def exhaust_after_manifest(path):
                value = original_statvfs(path)
                if manifest_committed:
                    value = dict(value)
                    value["f_bavail"] = 0
                    value["available_bytes"] = 0
                return value

            with (
                mock.patch.object(m, "_write_atomic_noreplace", side_effect=observe_publish),
                mock.patch.object(m, "_statvfs_receipt", side_effect=exhaust_after_manifest),
            ):
                with self.assertRaisesRegex(m.WriterQualificationError, "post-finalization storage floor"):
                    m._finalize_attempt(attempt, self.profile)
            self.assertTrue((attempt / "FINAL/MANIFEST.json").is_file())
            self.assertFalse((root / "QUALIFICATION_RESULT.json").exists())

    def test_result_last_publication_crash_resumes_from_mixed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            original = m._write_create_only

            def fail_result(parent_fd, name, payload):
                if name == ".QUALIFICATION_RESULT.json.partial":
                    original(parent_fd, name, payload)
                    raise m.WriterQualificationError("injected result-last failure")
                return original(parent_fd, name, payload)

            with mock.patch.object(m, "_write_create_only", side_effect=fail_result):
                with self.assertRaisesRegex(m.WriterQualificationError, "result-last"):
                    m.qualify(root, profile=self.profile, test_only=True, unreserved_floor_bytes=0)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"PROTOCOL.json", "EVIDENCE", "FINAL", "DIGEST_ROSTER.jsonl", ".QUALIFICATION_RESULT.json.partial"},
            )
            result = m.qualify(
                root, profile=self.profile, test_only=True, resume=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)

    def test_published_resume_crosschecks_the_current_preregistration_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._run(Path(temporary))
            forged_gate = {
                "schema": m.GIT_CUSTODY_SCHEMA,
                "mode": "TEST_ONLY_NO_GIT_OR_PREREGISTRATION_AUTHORITY",
                "externally_preregistered": True,
            }
            with self.assertRaisesRegex(m.WriterQualificationError, "live gate"):
                m._resume_publication_state(
                    root,
                    self._attempt(root),
                    self.profile,
                    forged_gate,
                )

    def test_physical_reservation_unsupported_fails_before_any_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))

            def unsupported(fd, offset, length):
                raise OSError(errno.EOPNOTSUPP, "not supported")

            with mock.patch.object(m.os, "posix_fallocate", side_effect=unsupported):
                with self.assertRaisesRegex(m.WriterQualificationError, "EOPNOTSUPP"):
                    m.qualify(root, profile=self.profile, test_only=True, unreserved_floor_bytes=0)
            self.assertNotIn("QUALIFICATION_RESULT.json", {path.name for path in root.iterdir()})
            self.assertEqual({path.name for path in root.iterdir()}, {"PROTOCOL.json"})
            result = m.qualify(
                root, profile=self.profile, test_only=True, resume=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)

    def test_live_fragment_st_blocks_forgery_fails_even_after_manifest_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaises(m.WriterQualificationError):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=1, unreserved_floor_bytes=0,
                )
            evidence = self._attempt(root) / "EVIDENCE"
            manifest_path, manifest = self._first_manifest(evidence)
            manifest["fragments"][m.FRAGMENT_NAMES[0]]["st_blocks"] += 1
            manifest["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, manifest)
            manifest_path.write_bytes(m._canonical_json(manifest))
            with self.assertRaisesRegex(m.WriterQualificationError, "fragment identity"):
                m.validate_shard(
                    evidence, self.profile, batch=0, shard=0,
                    predecessor_manifest_sha=m.ZERO_SHA256,
                    previous_role_sha={role: m.ZERO_SHA256 for role in m.RECEIPT_ROLES},
                    previous_first_1k_sha=m.ZERO_SHA256,
                )

    def test_coordinated_receipt_and_manifest_rehash_cannot_relabel_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaises(m.WriterQualificationError):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=1, unreserved_floor_bytes=0,
                )
            evidence = self._attempt(root) / "EVIDENCE"
            manifest_path, manifest = self._first_manifest(evidence)
            receipt_path = evidence / m._shard_name(0, 0) / "RECEIPTS/source.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["payload_sha256"] = "f" * 64
            receipt["content_sha256"] = m._domain_hash(m.RECEIPT_DOMAIN, receipt)
            receipt_payload = m._canonical_json(receipt)
            receipt_path.write_bytes(receipt_payload)
            locator = manifest["receipt_locators"]["source"]
            locator["byte_length"] = len(receipt_payload)
            locator["file_sha256"] = m._sha256(receipt_payload)
            locator["content_sha256"] = receipt["content_sha256"]
            manifest["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, manifest)
            manifest_path.write_bytes(m._canonical_json(manifest))
            with self.assertRaisesRegex(m.WriterQualificationError, "exact receipt"):
                m.validate_shard(
                    evidence, self.profile, batch=0, shard=0,
                    predecessor_manifest_sha=m.ZERO_SHA256,
                    previous_role_sha={role: m.ZERO_SHA256 for role in m.RECEIPT_ROLES},
                    previous_first_1k_sha=m.ZERO_SHA256,
                )

    def test_false_identity_alias_and_unexpected_resume_entry_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaises(m.WriterQualificationError):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=1, unreserved_floor_bytes=0,
                )
            evidence = self._attempt(root) / "EVIDENCE"
            manifest_path, manifest = self._first_manifest(evidence)
            manifest["batch_order"] = False
            manifest["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, manifest)
            manifest_path.write_bytes(m._canonical_json(manifest))
            with self.assertRaises(m.WriterQualificationError):
                m.validate_shard(
                    evidence, self.profile, batch=0, shard=0,
                    predecessor_manifest_sha=m.ZERO_SHA256,
                    previous_role_sha={role: m.ZERO_SHA256 for role in m.RECEIPT_ROLES},
                    previous_first_1k_sha=m.ZERO_SHA256,
                )
            (evidence / "rogue").write_bytes(b"unexpected")
            with self.assertRaisesRegex(m.WriterQualificationError, "unexpected entry"):
                m.qualify(
                    root, profile=self.profile, test_only=True, resume=True,
                    unreserved_floor_bytes=0,
                )

    def test_coordinated_attempt_rehash_cannot_forge_storage_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with self.assertRaises(m.WriterQualificationError):
                m.qualify(
                    root, profile=self.profile, test_only=True,
                    fault_after_shards=1, unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            record_path = attempt / "ATTEMPT.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["statvfs_before"]["f_favail"] = 0
            record["content_sha256"] = m._domain_hash(m.ATTEMPT_DOMAIN, record)
            payload = m._canonical_json(record)
            record_path.write_bytes(payload)
            (attempt / "EVIDENCE/ATTEMPT.json").write_bytes(payload)
            (attempt / "EVIDENCE/STORAGE_RESERVATION.json").write_bytes(payload)
            with self.assertRaisesRegex(m.WriterQualificationError, "admission"):
                m.qualify(
                    root, profile=self.profile, test_only=True, resume=True,
                    unreserved_floor_bytes=0,
                )

    def test_archive_surface_rejects_coordinated_claim_escalation_and_works_without_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, result = self._run(base, "full")
            archive = base / "archive"
            archive.mkdir()
            for name in m.ARCHIVE_ROSTER:
                shutil.copy2(root / name, archive / name)
            self.assertEqual(
                m.validate_archive_surface(archive, profile=self.profile, test_only=True),
                result,
            )
            digest_path = archive / "DIGEST_ROSTER.jsonl"
            lines = digest_path.read_bytes().splitlines(keepends=True)
            forged_row = json.loads(lines[1].decode("ascii"))
            forged_row["batch_order"] = False
            forged_row["content_sha256"] = m._domain_hash(m.DIGEST_ROSTER_DOMAIN, forged_row)
            lines[1] = m._canonical_json(forged_row)
            forged_digest = b"".join(lines)
            digest_path.write_bytes(forged_digest)
            result_path = archive / "QUALIFICATION_RESULT.json"
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            forged["digest_roster"]["byte_length"] = len(forged_digest)
            forged["digest_roster"]["sha256"] = m._sha256(forged_digest)
            forged["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, forged)
            result_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.WriterQualificationError, "shard-row semantics"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)
            shutil.copy2(root / "DIGEST_ROSTER.jsonl", digest_path)
            shutil.copy2(root / "QUALIFICATION_RESULT.json", result_path)
            lines = digest_path.read_bytes().splitlines(keepends=True)
            forged_row = json.loads(lines[1].decode("ascii"))
            forged_row["receipt_content_sha256"]["source"] = "e" * 64
            forged_row["receipt_file_sha256"]["source"] = "f" * 64
            forged_row["content_sha256"] = m._domain_hash(m.DIGEST_ROSTER_DOMAIN, forged_row)
            lines[1] = m._canonical_json(forged_row)
            forged_digest = b"".join(lines)
            digest_path.write_bytes(forged_digest)
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            forged["digest_roster"]["byte_length"] = len(forged_digest)
            forged["digest_roster"]["sha256"] = m._sha256(forged_digest)
            forged["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, forged)
            result_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.WriterQualificationError, "exact receipt semantics"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)
            shutil.copy2(root / "DIGEST_ROSTER.jsonl", digest_path)
            shutil.copy2(root / "QUALIFICATION_RESULT.json", result_path)
            lines = digest_path.read_bytes().splitlines(keepends=True)
            forged_row = json.loads(lines[1].decode("ascii"))
            forged_row["fragment_sha256"][m.FRAGMENT_NAMES[0]] = "e" * 64
            forged_row["content_sha256"] = m._domain_hash(m.DIGEST_ROSTER_DOMAIN, forged_row)
            lines[1] = m._canonical_json(forged_row)
            forged_digest = b"".join(lines)
            digest_path.write_bytes(forged_digest)
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            forged["digest_roster"]["byte_length"] = len(forged_digest)
            forged["digest_roster"]["sha256"] = m._sha256(forged_digest)
            forged["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, forged)
            result_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.WriterQualificationError, "fragment semantics"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)
            shutil.copy2(root / "DIGEST_ROSTER.jsonl", digest_path)
            shutil.copy2(root / "QUALIFICATION_RESULT.json", result_path)
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            reconciliation = forged["finalization_storage_reconciliation"]
            reconciliation["reservation_byte_length_after_release"] = 1
            reconciliation["content_sha256"] = m._domain_hash(
                m.FINALIZATION_STORAGE_DOMAIN,
                reconciliation,
            )
            forged["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, forged)
            result_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.WriterQualificationError, "reconciliation identity"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)
            shutil.copy2(root / "QUALIFICATION_RESULT.json", result_path)
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            forged["claims"]["dynamics_executed"] = True
            forged["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, forged)
            result_path.write_bytes(m._canonical_json(forged))
            with self.assertRaisesRegex(m.WriterQualificationError, "archive custody"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)

    def test_digest_roster_is_opened_no_follow_and_payload_validation_needs_local_large_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _ = self._run(base)
            digest = root / "DIGEST_ROSTER.jsonl"
            saved = base / "saved-digest"
            shutil.copy2(digest, saved)
            digest.unlink()
            digest.symlink_to(saved)
            with self.assertRaises((m.WriterQualificationError, OSError)):
                m.validate(root, profile=self.profile, test_only=True)

    def test_cleanup_failure_after_durable_result_does_not_turn_success_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._prepared(Path(temporary))
            with mock.patch.object(
                m, "_cleanup_empty_attempt_shell",
                side_effect=m.WriterQualificationError("injected cleanup failure"),
            ):
                result = m.qualify(root, profile=self.profile, test_only=True, unreserved_floor_bytes=0)
            self.assertEqual(m.validate(root, profile=self.profile, test_only=True), result)
            self.assertTrue((root / "QUALIFICATION_RESULT.json").is_file())


if __name__ == "__main__":
    unittest.main()
