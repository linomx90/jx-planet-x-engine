from __future__ import annotations

import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/jx_tno_10my_full128_writer_qualification_v3.py"


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

    def _runtime_sha(self, evidence: Path) -> str:
        attempt = json.loads((evidence / "ATTEMPT.json").read_text(encoding="utf-8"))
        return attempt["producer_runtime_identity"]["content_sha256"]

    def _snapshot_tree(self, root: Path) -> tuple[tuple[str, str, str], ...]:
        rows: list[tuple[str, str, str]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if path.is_symlink():
                rows.append((relative, "symlink", os.readlink(path)))
            elif path.is_dir():
                rows.append((relative, "directory", oct(metadata.st_mode)))
            elif path.is_file():
                rows.append((relative, "file", m._sha256(path.read_bytes())))
            else:
                rows.append((relative, "other", oct(metadata.st_mode)))
        return tuple(rows)

    def _fresh_child(
        self,
        operation: str,
        root: Path,
        archive: Path | None = None,
        *,
        optimize: int,
    ) -> dict[str, object]:
        child = r'''
import json
import os
from pathlib import Path
import shutil
import sys
import types

source = Path(sys.argv[1])
operation = sys.argv[2]
root = Path(sys.argv[3])
archive = None if sys.argv[4] == "-" else Path(sys.argv[4])
payload = source.read_bytes()
name = f"_jx_writer_v3_fresh_child_{os.getpid()}_{sys.flags.optimize}"
module = types.ModuleType(name)
module.__file__ = os.fspath(source)
module.__package__ = ""
module.__loader__ = None
module.__spec__ = None
sys.modules[name] = module
exec(compile(payload, os.fspath(source), "exec", dont_inherit=True,
             optimize=sys.flags.optimize), module.__dict__)
geometry = module.GridGeometry(
    batch_count=2, total_steps=8, shard_steps=4,
    b22_interval_steps=2, short_interval_steps=1, short_final_step=4,
    massive_count=2, tracer_count=3,
)
profile = module.WriterProfile("tiny-hostile-test", geometry, 4, False)

if operation in ("create", "partial"):
    module.prepare(root)
if operation == "create":
    result = module.qualify(
        root, profile=profile, test_only=True, unreserved_floor_bytes=0,
    )
    if archive is None:
        raise RuntimeError("archive path is required")
    archive.mkdir()
    for item in ("PROTOCOL.json", "DIGEST_ROSTER.jsonl", "QUALIFICATION_RESULT.json"):
        shutil.copy2(root / item, archive / item)
    output = {
        "result_content_sha256": result["content_sha256"],
        "producer_runtime_identity": result["producer_runtime_identity"],
    }
elif operation == "validate":
    result = module.validate(root, profile=profile, test_only=True)
    archived = module.validate_archive_surface(archive, profile=profile, test_only=True)
    attempt = json.loads((root / "EVIDENCE/ATTEMPT.json").read_text(encoding="utf-8"))
    runtime_receipt = json.loads(
        (root / "EVIDENCE/shard-b000-s0000/RECEIPTS/runtime.json").read_text(encoding="utf-8")
    )
    output = {
        "result_content_sha256": result["content_sha256"],
        "archive_content_sha256": archived["content_sha256"],
        "producer_runtime_identity": attempt["producer_runtime_identity"],
        "runtime_receipt_payload_sha256": runtime_receipt["payload_sha256"],
        "validator_python_optimize": sys.flags.optimize,
    }
elif operation == "partial":
    try:
        module.qualify(
            root, profile=profile, test_only=True, fault_after_shards=1,
            unreserved_floor_bytes=0,
        )
    except module.WriterQualificationError as exc:
        if "injected crash after committed shard" not in str(exc):
            raise
    else:
        raise RuntimeError("partial child did not fault")
    attempt = json.loads(
        (root.parent / module._attempt_name(root) / "ATTEMPT.json").read_text(encoding="utf-8")
    )
    output = {"producer_runtime_identity": attempt["producer_runtime_identity"]}
elif operation == "resume":
    try:
        result = module.qualify(
            root, profile=profile, test_only=True, resume=True,
            unreserved_floor_bytes=0,
        )
    except module.WriterQualificationError as exc:
        output = {"error": str(exc), "validator_python_optimize": sys.flags.optimize}
    else:
        output = {
            "result_content_sha256": result["content_sha256"],
            "validator_python_optimize": sys.flags.optimize,
        }
else:
    raise RuntimeError("unknown operation")
print(json.dumps(output, sort_keys=True))
'''
        arguments = [sys.executable]
        if optimize == 2:
            arguments.append("-OO")
        elif optimize != 0:
            raise AssertionError("fresh-child optimize must be 0 or 2")
        arguments.extend(
            [
                "-B", "-c", child, os.fspath(SOURCE), operation,
                os.fspath(root), "-" if archive is None else os.fspath(archive),
            ]
        )
        environment = dict(os.environ)
        environment.pop("PYTHONOPTIMIZE", None)
        completed = subprocess.run(
            arguments,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"fresh child failed ({operation}, optimize={optimize}):\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        return json.loads(completed.stdout)

    def test_production_metadata_plan_is_exact_and_dry(self) -> None:
        plan = m.production_metadata_plan_v3()
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

    def test_v2_ext4_receipts_validate_in_production_profile_without_allocation(self) -> None:
        extra_extent_names = {"B22_STATES.f64le", "ML_SHORT_STATES.f64le"}
        receipts = {
            name: m._physical_allocation_receipt(
                size,
                4096,
                m._round_up(size, 4096) // 512 + (8 if name in extra_extent_names else 0),
                f"frozen V2 {name}",
            )
            for name, size in m.PRODUCTION_PROFILE.geometry.output_sizes().items()
        }
        with mock.patch.object(
            m.os,
            "posix_fallocate",
            side_effect=AssertionError("nonallocating regression must not fallocate"),
        ):
            summary = m._validate_final_reservation_set(
                receipts, m.PRODUCTION_PROFILE, 4096,
            )
        self.assertEqual(summary["final_minimum_allocated_bytes"], 4_743_286_784)
        self.assertEqual(summary["observed_final_allocated_bytes"], 4_743_294_976)
        self.assertEqual(summary["observed_final_allocation_overhead_bytes"], 8_192)

    def test_minimum_observed_receipts_accept_bounded_overhead_and_reject_aliases(self) -> None:
        size = self.profile.geometry.output_sizes()[m.FINAL_BINARY_NAMES[0]]
        minimum = m._round_up(size, 4096)
        blocks = minimum // 512 + 8
        receipt = m._physical_allocation_receipt(size, 4096, blocks, "injected final")
        self.assertEqual(receipt["observed_allocation_overhead_bytes"], 4096)
        self.assertEqual(
            m._validate_physical_allocation_receipt(
                receipt, logical_byte_length=size, allocation_unit=4096,
                label="injected final",
            ),
            receipt,
        )
        shard = m._shard_allocation_receipt(
            fragment_minimum_allocated_bytes=minimum,
            fragment_observed_allocated_bytes=minimum + 4096,
            nonfragment_observed_allocated_bytes=4096,
        )
        self.assertEqual(shard["allocation_allowance_used_bytes"], 8192)
        with self.assertRaisesRegex(m.WriterQualificationError, "allowance"):
            m._shard_allocation_receipt(
                fragment_minimum_allocated_bytes=minimum,
                fragment_observed_allocated_bytes=(
                    minimum + m.SHARD_ALLOCATION_ALLOWANCE_BYTES + 512
                ),
                nonfragment_observed_allocated_bytes=0,
            )
        below = dict(receipt)
        below["observed_st_blocks"] = max(0, minimum // 512 - 1)
        below["observed_allocated_bytes"] = below["observed_st_blocks"] * 512
        below["observed_allocation_overhead_bytes"] = below["observed_allocated_bytes"] - minimum
        with self.assertRaisesRegex(m.WriterQualificationError, "below"):
            m._validate_physical_allocation_receipt(
                below, logical_byte_length=size, allocation_unit=4096,
                label="below minimum",
            )
        boolean = dict(receipt)
        boolean["observed_st_blocks"] = False
        with self.assertRaisesRegex(m.WriterQualificationError, "type"):
            m._validate_physical_allocation_receipt(
                boolean, logical_byte_length=size, allocation_unit=4096,
                label="bool alias",
            )

    def test_frozen_v2_inconclusive_lineage_is_self_contained_and_rehash_tamper_fails(self) -> None:
        protocol = m.build_protocol()
        lineage = protocol["predecessor_result_lineage"]
        self.assertEqual(lineage["predecessor"]["source"], m.V2_SOURCE)
        self.assertEqual(lineage["predecessor"]["test"], m.V2_TEST)
        self.assertEqual(lineage["predecessor"]["protocol"], m.V2_PROTOCOL)
        self.assertEqual(lineage["retained_complete_root"]["result"], m.V2_RESULT)
        self.assertEqual(
            lineage["retained_complete_root"]["output_root_roster"],
            ["DIGEST_ROSTER.jsonl", "EVIDENCE", "FINAL", "PROTOCOL.json", "QUALIFICATION_RESULT.json"],
        )
        observation = lineage["cross_mode_observation"]
        self.assertEqual(observation["normal_interpreter"]["live_validate"], "PASS")
        self.assertEqual(observation["normal_interpreter"]["archive_validate"], "PASS")
        self.assertEqual(observation["python_oo_interpreter"]["live_validate"], "FAIL")
        self.assertEqual(observation["python_oo_interpreter"]["archive_validate"], "FAIL")
        self.assertEqual(
            observation["normal_interpreter"]["stored_runtime_payload_sha256"],
            m.V2_RUNTIME_PAYLOAD_OPTIMIZE_0,
        )
        self.assertEqual(
            observation["python_oo_interpreter"]["reconstructed_runtime_payload_sha256"],
            m.V2_RUNTIME_PAYLOAD_OPTIMIZE_2,
        )
        self.assertFalse(lineage["predecessor"]["future_result_tag_present"])
        self.assertFalse(lineage["custody_policy"]["v3_reads_live_v2_root"])
        self.assertTrue(lineage["custody_policy"]["fresh_v3_qualification_required"])
        self.assertEqual(lineage["claims"], m._claim_ceiling(False))
        forged = json.loads(json.dumps(protocol))
        forged_lineage = forged["predecessor_result_lineage"]
        forged_lineage["cross_mode_observation"]["python_oo_interpreter"]["archive_validate"] = "PASS"
        forged_lineage["content_sha256"] = m._domain_hash(m.PREDECESSOR_RESULT_DOMAIN, forged_lineage)
        forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
        with self.assertRaises(m.WriterQualificationError):
            m.validate_protocol(forged)

    def test_protocol_prepare_and_validation_never_access_live_v2_bytes(self) -> None:
        forbidden = (
            os.fspath(ROOT / "benchmarks/jx_tno_10my_full128_writer_qualification_v2.py"),
            os.fspath(ROOT / "tests/test_tno_10my_full128_writer_qualification_v2.py"),
            os.fspath(ROOT / "runs/tno_10my_full128_writer_qualification_v2"),
            os.fspath(ROOT / "runs/.tno_10my_full128_writer_qualification_v2.writer-qualification-v2.partial"),
        )
        original_stat = Path.stat
        original_lstat = Path.lstat
        original_read_regular = m._read_regular

        def forbidden_path(path: object) -> bool:
            absolute = os.path.abspath(os.fspath(path))
            return any(absolute.startswith(item) for item in forbidden)

        def guarded_stat(path: Path, *args, **kwargs):
            if forbidden_path(path):
                raise AssertionError("live V2 path access is forbidden")
            return original_stat(path, *args, **kwargs)

        def guarded_lstat(path: Path, *args, **kwargs):
            if forbidden_path(path):
                raise AssertionError("live V2 path access is forbidden")
            return original_lstat(path, *args, **kwargs)

        def guarded_read_regular(path: Path, *args, **kwargs):
            if forbidden_path(path):
                raise AssertionError("live V2 bytes are forbidden")
            return original_read_regular(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            Path, "stat", guarded_stat,
        ), mock.patch.object(Path, "lstat", guarded_lstat), mock.patch.object(
            m, "_read_regular", guarded_read_regular,
        ):
            protocol = m.build_protocol()
            self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
            root = Path(temporary) / "v3-prepared"
            self.assertEqual(m.prepare(root), protocol)
            self.assertEqual(m.validate_prepared(root), protocol)
            result = m.qualify(
                root, profile=self.profile, test_only=True,
                unreserved_floor_bytes=0,
            )
            self.assertEqual(
                m.validate(root, profile=self.profile, test_only=True),
                result,
            )

    def test_finalization_stream_overhead_is_explicit_bounded_and_floor_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, result = self._run(Path(temporary))
            plan = result["finalization_plan"]
            reconciliation = json.loads(json.dumps(result["finalization_storage_reconciliation"]))
            stream = reconciliation["streams"]["FINAL/SHARD_MANIFESTS.jsonl"]
            stream["observed_st_blocks"] += 8
            stream["observed_allocated_bytes"] += 4096
            stream["observed_allocation_overhead_bytes"] += 4096
            reconciliation["observed_stream_allocated_bytes"] += 4096
            reconciliation["observed_stream_allocation_overhead_bytes"] += 4096
            reconciliation["remaining_tail_allocation_allowance_bytes"] -= 4096
            reconciliation["content_sha256"] = m._domain_hash(
                m.FINALIZATION_STORAGE_DOMAIN, reconciliation,
            )
            self.assertEqual(
                m._validate_finalization_storage_record(reconciliation, plan),
                reconciliation,
            )
            excessive = json.loads(json.dumps(reconciliation))
            extra = excessive["remaining_tail_allocation_allowance_bytes"] + 512
            extra_blocks = (extra + 511) // 512
            delta = extra_blocks * 512
            stream = excessive["streams"]["FINAL/SHARD_MANIFESTS.jsonl"]
            stream["observed_st_blocks"] += extra_blocks
            stream["observed_allocated_bytes"] += delta
            stream["observed_allocation_overhead_bytes"] += delta
            excessive["observed_stream_allocated_bytes"] += delta
            excessive["observed_stream_allocation_overhead_bytes"] += delta
            excessive["remaining_tail_allocation_allowance_bytes"] -= delta
            excessive["content_sha256"] = m._domain_hash(
                m.FINALIZATION_STORAGE_DOMAIN, excessive,
            )
            with self.assertRaisesRegex(m.WriterQualificationError, "allowance"):
                m._validate_finalization_storage_record(excessive, plan)
            for field in (
                "reservation_byte_length_after_release",
                "reservation_observed_st_blocks_after_release",
                "reservation_observed_allocated_bytes_after_release",
            ):
                boolean = json.loads(json.dumps(reconciliation))
                boolean[field] = False
                boolean["content_sha256"] = m._domain_hash(
                    m.FINALIZATION_STORAGE_DOMAIN, boolean,
                )
                with self.subTest(field=field), self.assertRaisesRegex(
                    m.WriterQualificationError, "reconciliation identity",
                ):
                    m._validate_finalization_storage_record(boolean, plan)

    def test_reservation_shrink_uses_exact_observed_committed_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt"
            attempt.mkdir()
            reservation = attempt / "RESERVATION.bin"
            fd = os.open(reservation, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.posix_fallocate(fd, 0, 65_536)
            finally:
                os.close(fd)
            record = {
                "reservation_plan": {"initial_evidence_reservation_byte_length": 65_536}
            }
            observed_committed = 12_288
            m._reservation_remaining(
                attempt, record, observed_committed, floor_bytes=0,
            )
            self.assertEqual(reservation.stat().st_size, 65_536 - observed_committed)
            with self.assertRaises(m.WriterQualificationError):
                m._reservation_remaining(attempt, record, False, floor_bytes=0)

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
        runtime = protocol["producer_runtime_custody"]
        self.assertEqual(runtime["schema"], m.PRODUCER_RUNTIME_SCHEMA)
        self.assertEqual(runtime["production_python_optimize"], 0)
        self.assertEqual(
            runtime["runtime_receipt_payload"],
            "PRODUCER_RUNTIME_CONTENT_SHA256",
        )
        self.assertIn("WITHOUT_SUBSTITUTING_VALIDATOR_RUNTIME", runtime["offline_validation_rule"])
        self.assertFalse(runtime["test_profile_nonzero_optimize_has_production_authority"])
        self.assertIn(
            "EXTERNAL_RESULT_TAG_OR_PINNED_RESULT_HASH",
            protocol["archive_surface"]["authenticity_boundary"],
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
            expected = "producer python optimize 0" if sys.flags.optimize else "preregistration gate"
            with self.assertRaisesRegex(m.WriterQualificationError, expected):
                m.qualify(
                    m.DEFAULT_OUTPUT_ROOT,
                    externally_preregistered=True,
                    prereg_commit="a" * 40,
                    prereg_tag_object="b" * 40,
                )
        if sys.flags.optimize:
            git_gate.assert_not_called()
        else:
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
            self.assertGreater(plan["stream_minimum_allocated_bytes"], 0)
            self.assertEqual(reconciliation["reservation_byte_length_after_release"], 0)
            self.assertEqual(reconciliation["reservation_observed_allocated_bytes_after_release"], 0)
            self.assertGreater(
                sum(row["observed_allocated_bytes"] for row in reconciliation["streams"].values()),
                0,
            )
            self.assertGreaterEqual(
                reconciliation["statvfs_after_streams"]["available_bytes"],
                plan["unreserved_floor_bytes"]
                + reconciliation["remaining_tail_allocation_allowance_bytes"],
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
            fragment = manifest["fragments"][m.FRAGMENT_NAMES[0]]
            fragment["observed_st_blocks"] += 1
            fragment["observed_allocated_bytes"] += 512
            fragment["observed_allocation_overhead_bytes"] += 512
            manifest["content_sha256"] = m._domain_hash(m.SHARD_DOMAIN, manifest)
            manifest_path.write_bytes(m._canonical_json(manifest))
            with self.assertRaisesRegex(m.WriterQualificationError, "fragment identity"):
                m.validate_shard(
                    evidence, self.profile,
                    producer_runtime_content_sha256=self._runtime_sha(evidence),
                    batch=0, shard=0,
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
                    evidence, self.profile,
                    producer_runtime_content_sha256=self._runtime_sha(evidence),
                    batch=0, shard=0,
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
                    evidence, self.profile,
                    producer_runtime_content_sha256=self._runtime_sha(evidence),
                    batch=0, shard=0,
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
            forged["digest_roster"]["logical_byte_length"] = len(forged_digest)
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
            forged["digest_roster"]["logical_byte_length"] = len(forged_digest)
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
            forged["digest_roster"]["logical_byte_length"] = len(forged_digest)
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

    def test_archive_summary_cannot_relabel_minimum_or_maximum_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _ = self._run(base, "full")
            archive = base / "archive"
            archive.mkdir()

            def forge_summary(mutator) -> None:
                for name in m.ARCHIVE_ROSTER:
                    shutil.copy2(root / name, archive / name)
                digest_path = archive / "DIGEST_ROSTER.jsonl"
                result_path = archive / "QUALIFICATION_RESULT.json"
                lines = digest_path.read_bytes().splitlines(keepends=True)
                header = json.loads(lines[0].decode("ascii"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
                summary = result["evidence_allocation_summary"]
                mutator(summary)
                header["evidence_allocation_summary"] = json.loads(json.dumps(summary))
                header["content_sha256"] = m._domain_hash(
                    m.DIGEST_ROSTER_HEADER_DOMAIN, header,
                )
                lines[0] = m._canonical_json(header)
                forged_digest = b"".join(lines)
                self.assertEqual(len(forged_digest), digest_path.stat().st_size)
                digest_path.write_bytes(forged_digest)
                plan = result["finalization_plan"]
                plan["evidence_allocation_summary_sha256"] = m._sha256(
                    m._canonical_json(summary)
                )
                plan["content_sha256"] = m._domain_hash(
                    m.FINALIZATION_PLAN_DOMAIN, plan,
                )
                reconciliation = result["finalization_storage_reconciliation"]
                reconciliation["plan_content_sha256"] = plan["content_sha256"]
                reconciliation["content_sha256"] = m._domain_hash(
                    m.FINALIZATION_STORAGE_DOMAIN, reconciliation,
                )
                result["digest_roster_header_content_sha256"] = header["content_sha256"]
                result["digest_roster"]["sha256"] = m._sha256(forged_digest)
                result["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, result)
                result_path.write_bytes(m._canonical_json(result))

            def relabel_minimum(summary) -> None:
                summary["fragment_minimum_allocated_bytes"] -= 1
                summary["fragment_observed_allocation_overhead_bytes"] += 1
                summary["allocation_allowance_used_bytes"] += 1
                summary["maximum_shard_allocation_allowance_used_bytes"] += 1
                self.assertLessEqual(
                    summary["maximum_shard_allocation_allowance_used_bytes"],
                    summary["allocation_allowance_used_bytes"],
                )
                self.assertLessEqual(
                    summary["allocation_allowance_used_bytes"],
                    summary["maximum_shard_allocation_allowance_used_bytes"]
                    * summary["shard_count"],
                )

            forge_summary(relabel_minimum)
            with self.assertRaisesRegex(m.WriterQualificationError, "archive custody"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)

            def relabel_maximum(summary) -> None:
                summary["maximum_shard_allocation_allowance_used_bytes"] -= 4096

            forge_summary(relabel_maximum)
            with self.assertRaisesRegex(m.WriterQualificationError, "archive custody"):
                m.validate_archive_surface(archive, profile=self.profile, test_only=True)

    def test_archive_reconstructs_exact_final_manifest_locators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, result = self._run(base, "full")
            archive = base / "archive"
            archive.mkdir()
            for name in m.ARCHIVE_ROSTER:
                shutil.copy2(root / name, archive / name)
            reconstructed, payload = m._reconstruct_final_manifest_from_result(
                self.profile, result,
            )
            self.assertEqual(
                reconstructed["content_sha256"],
                result["final_manifest_content_sha256"],
            )
            self.assertEqual(
                m._sha256(payload),
                result["final_manifest_file_sha256"],
            )
            result_path = archive / "QUALIFICATION_RESULT.json"
            forged = json.loads(result_path.read_text(encoding="utf-8"))
            forged["final_manifest_content_sha256"] = "e" * 64
            forged["final_manifest_file_sha256"] = "f" * 64
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

    def test_fresh_normal_artifact_validates_live_and_archive_under_oo_and_inverse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            normal_root = base / "normal-result"
            normal_archive = base / "normal-archive"
            created = self._fresh_child(
                "create", normal_root, normal_archive, optimize=0,
            )
            before_root = self._snapshot_tree(normal_root)
            before_archive = self._snapshot_tree(normal_archive)
            validated_oo = self._fresh_child(
                "validate", normal_root, normal_archive, optimize=2,
            )
            validated_normal = self._fresh_child(
                "validate", normal_root, normal_archive, optimize=0,
            )
            self.assertEqual(created["result_content_sha256"], validated_oo["result_content_sha256"])
            self.assertEqual(validated_oo["result_content_sha256"], validated_oo["archive_content_sha256"])
            self.assertEqual(validated_normal["result_content_sha256"], validated_oo["result_content_sha256"])
            self.assertEqual(validated_oo["validator_python_optimize"], 2)
            producer = validated_oo["producer_runtime_identity"]
            self.assertEqual(producer["python_optimize"], 0)
            self.assertEqual(
                validated_oo["runtime_receipt_payload_sha256"],
                producer["content_sha256"],
            )
            self.assertEqual(self._snapshot_tree(normal_root), before_root)
            self.assertEqual(self._snapshot_tree(normal_archive), before_archive)

            # The inverse is test-fixture-only: an -OO producer has no
            # production authority, but normal offline replay still must not
            # substitute its own runtime.
            oo_root = base / "oo-result"
            oo_archive = base / "oo-archive"
            oo_created = self._fresh_child("create", oo_root, oo_archive, optimize=2)
            inverse = self._fresh_child("validate", oo_root, oo_archive, optimize=0)
            self.assertEqual(oo_created["result_content_sha256"], inverse["result_content_sha256"])
            self.assertEqual(inverse["result_content_sha256"], inverse["archive_content_sha256"])
            self.assertEqual(inverse["producer_runtime_identity"]["python_optimize"], 2)

    def test_cross_mode_resume_fails_before_mutation_then_matching_runtime_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "resume-result"
            partial = self._fresh_child("partial", root, optimize=0)
            self.assertEqual(partial["producer_runtime_identity"]["python_optimize"], 0)
            before = self._snapshot_tree(base)
            rejected = self._fresh_child("resume", root, optimize=2)
            self.assertIn("mutating execution runtime differs", rejected["error"])
            self.assertEqual(self._snapshot_tree(base), before)
            resumed = self._fresh_child("resume", root, optimize=0)
            self.assertIn("result_content_sha256", resumed)
            self.assertTrue((root / "QUALIFICATION_RESULT.json").is_file())

            forged_root = base / "forged-resume-result"
            self._fresh_child("partial", forged_root, optimize=0)
            forged_attempt = forged_root.parent / m._attempt_name(forged_root)
            record = json.loads(
                (forged_attempt / "ATTEMPT.json").read_text(encoding="utf-8")
            )
            record["producer_runtime_identity"]["python_optimize"] = 2
            record["producer_runtime_identity"]["content_sha256"] = m._domain_hash(
                m.PRODUCER_RUNTIME_DOMAIN,
                record["producer_runtime_identity"],
            )
            record["content_sha256"] = m._domain_hash(m.ATTEMPT_DOMAIN, record)
            record_payload = m._canonical_json(record)
            for relative in (
                "ATTEMPT.json", "EVIDENCE/ATTEMPT.json",
                "EVIDENCE/STORAGE_RESERVATION.json",
            ):
                (forged_attempt / relative).write_bytes(record_payload)
            forged_before = self._snapshot_tree(forged_attempt)
            forged_rejected = self._fresh_child("resume", forged_root, optimize=0)
            self.assertIn("mutating execution runtime differs", forged_rejected["error"])
            self.assertEqual(self._snapshot_tree(forged_attempt), forged_before)

    def test_published_and_partial_result_authority_cannot_be_shadowed_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "published-result"
            archive = base / "published-archive"
            self._fresh_child("create", root, archive, optimize=0)
            attempt = self._attempt(root)
            attempt.mkdir()
            shutil.copy2(root / "EVIDENCE/ATTEMPT.json", attempt / "ATTEMPT.json")
            (attempt / "RESERVATION.bin").write_bytes(b"")
            before_cleanup_shell = self._snapshot_tree(base)
            rejected_shell = self._fresh_child("resume", root, optimize=2)
            self.assertIn("mutating execution runtime differs", rejected_shell["error"])
            self.assertEqual(self._snapshot_tree(base), before_cleanup_shell)

            (attempt / "ATTEMPT.json").unlink()
            (attempt / "RESERVATION.bin").unlink()
            before_empty_shell = self._snapshot_tree(base)
            rejected_empty = self._fresh_child("resume", root, optimize=2)
            self.assertIn("mutating execution runtime differs", rejected_empty["error"])
            self.assertEqual(self._snapshot_tree(base), before_empty_shell)

            partial_root = base / "partial-result"
            partial_archive = base / "partial-archive"
            self._fresh_child("create", partial_root, partial_archive, optimize=0)
            partial_attempt = self._attempt(partial_root)
            partial_attempt.mkdir()
            (partial_attempt / "RESERVATION.bin").write_bytes(b"")
            (partial_root / "QUALIFICATION_RESULT.json").rename(
                partial_root / ".QUALIFICATION_RESULT.json.partial"
            )
            before_partial = self._snapshot_tree(base)
            rejected_partial = self._fresh_child(
                "resume", partial_root, optimize=2,
            )
            self.assertIn("uncommitted attempt coexists", rejected_partial["error"])
            self.assertEqual(self._snapshot_tree(base), before_partial)

    def test_runtime_gate_precedes_direct_recovery_unlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._prepared(base, "finalizer-recovery")
            with self.assertRaisesRegex(m.WriterQualificationError, "injected crash"):
                m.qualify(
                    root,
                    profile=self.profile,
                    test_only=True,
                    fault_after_shards=self.profile.shard_total,
                    unreserved_floor_bytes=0,
                )
            attempt = self._attempt(root)
            (attempt / "FINAL/SHARD_MANIFESTS.jsonl").write_bytes(b"incomplete\n")
            (attempt / "DIGEST_ROSTER.jsonl").write_bytes(b"incomplete\n")
            (attempt / "EVIDENCE/FINALIZATION_STORAGE.json").write_bytes(b"incomplete\n")
            retained = json.loads(
                (attempt / "ATTEMPT.json").read_text(encoding="utf-8")
            )["producer_runtime_identity"]
            alternate = dict(retained)
            alternate["python_optimize"] = (retained["python_optimize"] + 1) % 3
            alternate["content_sha256"] = m._domain_hash(
                m.PRODUCER_RUNTIME_DOMAIN, alternate,
            )
            before_finalizer = self._snapshot_tree(attempt)
            with mock.patch.object(
                m, "_live_producer_runtime_identity", return_value=alternate,
            ):
                with self.assertRaisesRegex(
                    m.WriterQualificationError,
                    "mutating execution runtime differs",
                ):
                    m._finalize_or_recover_attempt(attempt, self.profile)
            self.assertEqual(self._snapshot_tree(attempt), before_finalizer)

            recovery_root = self._prepared(base, "initialization-recovery")
            recovery_attempt = self._attempt(recovery_root)
            recovery_attempt.mkdir()
            (recovery_attempt / "uncommitted").write_bytes(b"must remain")
            producer = m._live_producer_runtime_identity(self.profile)
            alternate = dict(producer)
            alternate["python_optimize"] = (producer["python_optimize"] + 1) % 3
            alternate["content_sha256"] = m._domain_hash(
                m.PRODUCER_RUNTIME_DOMAIN, alternate,
            )
            before_initialization = self._snapshot_tree(base)
            with mock.patch.object(
                m, "_live_producer_runtime_identity", return_value=alternate,
            ):
                with self.assertRaisesRegex(
                    m.WriterQualificationError,
                    "mutating execution runtime differs",
                ):
                    m._recover_uncommitted_initialization(
                        recovery_root,
                        recovery_attempt,
                        self.profile,
                        0,
                        m._test_git_custody(),
                        producer,
                    )
            self.assertEqual(self._snapshot_tree(base), before_initialization)

    def test_producer_runtime_identity_exact_types_source_binding_and_receipt_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, result = self._run(base)
            attempt = json.loads(
                (root / "EVIDENCE/ATTEMPT.json").read_text(encoding="utf-8")
            )
            identity = attempt["producer_runtime_identity"]
            self.assertEqual(
                m._validate_producer_runtime_identity(identity, self.profile),
                identity,
            )
            header = json.loads(
                (root / "DIGEST_ROSTER.jsonl").read_bytes().splitlines()[0]
            )
            final_manifest = json.loads(
                (root / "FINAL/MANIFEST.json").read_text(encoding="utf-8")
            )
            runtime_receipt = json.loads(
                (root / "EVIDENCE/shard-b000-s0000/RECEIPTS/runtime.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(header["producer_runtime_identity"], identity)
            self.assertEqual(header["attempt_record"]["producer_runtime_identity"], identity)
            self.assertEqual(final_manifest["producer_runtime_identity"], identity)
            self.assertEqual(result["producer_runtime_identity"], identity)
            self.assertEqual(runtime_receipt["payload_sha256"], identity["content_sha256"])

            production_identity = json.loads(json.dumps(identity))
            production_identity["profile_content_sha256"] = m._profile_record(
                m.PRODUCTION_PROFILE
            )["content_sha256"]
            production_identity["python_optimize"] = 0
            production_identity["content_sha256"] = m._domain_hash(
                m.PRODUCER_RUNTIME_DOMAIN, production_identity,
            )
            self.assertEqual(
                m._validate_producer_runtime_identity(
                    production_identity, m.PRODUCTION_PROFILE,
                ),
                production_identity,
            )

            for field, replacement in (
                ("python_optimize", False),
                ("python_version_hex", False),
                ("profile_content_sha256", "0" * 64),
                ("writer_source_sha256", "f" * 64),
            ):
                forged = json.loads(json.dumps(identity))
                forged[field] = replacement
                forged["content_sha256"] = m._domain_hash(
                    m.PRODUCER_RUNTIME_DOMAIN, forged,
                )
                with self.assertRaises(m.WriterQualificationError):
                    m._validate_producer_runtime_identity(forged, self.profile)
            forged = json.loads(json.dumps(identity))
            forged["content_sha256"] = "f" * 64
            with self.assertRaises(m.WriterQualificationError):
                m._validate_producer_runtime_identity(forged, self.profile)

            alternate = json.loads(json.dumps(identity))
            alternate["python_optimize"] = (identity["python_optimize"] + 1) % 3
            alternate["content_sha256"] = m._domain_hash(
                m.PRODUCER_RUNTIME_DOMAIN, alternate,
            )
            self.assertEqual(
                m._validate_producer_runtime_identity(alternate, self.profile),
                alternate,
            )
            receipt_arguments = {
                "role": "runtime", "batch": 0, "shard": 0,
                "predecessor_manifest_content_sha256": m.ZERO_SHA256,
                "previous_role_content_sha256": m.ZERO_SHA256,
                "first_1k_chain_sha256": m.ZERO_SHA256,
            }
            original_receipt = m._receipt_record(
                self.profile,
                producer_runtime_content_sha256=identity["content_sha256"],
                **receipt_arguments,
            )
            alternate_receipt = m._receipt_record(
                self.profile,
                producer_runtime_content_sha256=alternate["content_sha256"],
                **receipt_arguments,
            )
            repeated_original = m._receipt_record(
                self.profile,
                producer_runtime_content_sha256=identity["content_sha256"],
                **receipt_arguments,
            )
            self.assertNotEqual(original_receipt, alternate_receipt)
            self.assertEqual(original_receipt, repeated_original)

            archive = base / "archive"
            archive.mkdir()
            for name in m.ARCHIVE_ROSTER:
                shutil.copy2(root / name, archive / name)
            result_path = archive / "QUALIFICATION_RESULT.json"
            relabelled = json.loads(result_path.read_text(encoding="utf-8"))
            relabelled["producer_runtime_identity"] = alternate
            relabelled["content_sha256"] = m._domain_hash(m.RESULT_DOMAIN, relabelled)
            result_path.write_bytes(m._canonical_json(relabelled))
            with self.assertRaises(m.WriterQualificationError):
                m.validate_archive_surface(
                    archive, profile=self.profile, test_only=True,
                )

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
