from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "benchmarks/jx_tno_10my_full129_trajectory_acquisition_v1.py"


def _load_source(name: str = "jx_tno_10my_full129_trajectory_acquisition_v1_tested"):
    payload = SOURCE_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE_PATH)
    sys.modules[name] = module
    exec(compile(payload, str(SOURCE_PATH), "exec"), module.__dict__)
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    if not root.exists():
        return ()
    records: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((relative, "symlink", 0, os.readlink(path)))
        elif path.is_dir():
            records.append((relative, "directory", 0, ""))
        else:
            payload = path.read_bytes()
            records.append((relative, "file", len(payload), _sha256(payload)))
    return tuple(records)


CHILD = r'''
from pathlib import Path
import sys
import types

source = Path(sys.argv[1]).resolve()
name = "jx_tno_10my_full129_trajectory_acquisition_v1_child"
module = types.ModuleType(name)
module.__file__ = str(source)
sys.modules[name] = module
exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
root = Path(sys.argv[3])
if sys.argv[2] == "prepare":
    value = module.prepare(root)
elif sys.argv[2] == "validate":
    value = module.validate_prepared(root)
else:
    raise RuntimeError("unknown child command")
print(value["content_sha256"])
'''


class TrajectoryAcquisitionV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_source()
        cls.protocol = cls.module.build_protocol()

    def test_protocol_is_canonical_deterministic_and_prepare_only(self) -> None:
        m = self.module
        first = self.protocol
        second = m.build_protocol()
        self.assertTrue(m._exact_equal(first, second))
        self.assertEqual(
            first["content_sha256"], m._domain_hash(m.PROTOCOL_DOMAIN, first)
        )
        self.assertEqual(
            m._canonical_json(first),
            json.dumps(
                first, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii") + b"\n",
        )
        self.assertEqual(
            first["status"],
            "PREREGISTRATION_CANDIDATE_PREPARE_ONLY_EXECUTION_NOT_IMPLEMENTED",
        )
        self.assertEqual(
            first["public_cli"],
            ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        )
        self.assertFalse(hasattr(m, "execute"))
        self.assertFalse(hasattr(m, "resume"))
        self.assertFalse(hasattr(m, "execute_worker"))

    def test_claim_ceiling_and_geometry_are_exact_typed(self) -> None:
        claims = self.protocol["claims"]
        self.assertIs(claims["prepare_only_protocol_frozen"], True)
        for key, value in claims.items():
            self.assertIs(type(value), bool, key)
            if key != "prepare_only_protocol_frozen":
                self.assertIs(value, False, key)
        geometry = self.protocol["production_geometry"]
        expected = {
            "batch_count": 12,
            "tracer_count_per_batch": 129,
            "body_count": 134,
            "total_steps_per_batch": 640_000_000,
            "shards_per_batch": 1_000,
            "total_shards": 12_000,
            "final_binary_bytes": 4_743_277_848,
        }
        for key, value in expected.items():
            self.assertIs(type(geometry[key]), int, key)
            self.assertEqual(geometry[key], value, key)
        self.assertIs(geometry["trajectory_only_no_elements_during_dynamics"], True)
        self.assertIs(geometry["removed_states_retained"], True)

    def test_all_mandatory_stop_gates_and_gaps_are_frozen(self) -> None:
        readiness = self.protocol["execution_readiness"]
        blockers = set(readiness["blocking_contracts"])
        required_blockers = {
            "ALL_12_FULL129_GLOBAL_1KYR_STOP_GATE_NOT_EXECUTED",
            "REAL_ROUTE_TRIGGER_INPUT_EVIDENCE_NOT_CAPTURED_OR_QUALIFIED",
            "OUTCOME_BLIND_CAPPED_VARIABLE_EVIDENCE_WRITER_NOT_QUALIFIED",
            "ROUTED_AND_REMOVED_5MYR_RESTART_NOT_QUALIFIED",
            "FOUR_PREREGISTERED_5MYR_BATCH_PILOTS_NOT_COMPLETED",
            "DETERMINISTIC_SOURCE_CHUNK_REGENERATION_AND_RECEIPTS_NOT_QUALIFIED",
        }
        self.assertTrue(required_blockers.issubset(blockers))
        phases = readiness["mandatory_phase_order"]
        self.assertLess(
            phases.index("EXECUTE_ALL_12_FULL129_GLOBAL_1KYR_STOP_GATE"),
            phases.index("STOP_AND_REVIEW_ALL_12_1KYR_RESULTS_BEFORE_ANY_BUILD_BEYOND_1KYR"),
        )
        self.assertLess(
            phases.index("QUALIFY_ROUTED_AND_REMOVED_5MYR_RESTART"),
            phases.index("EXECUTE_FOUR_SEPARATELY_PREREGISTERED_5MYR_BATCH_PILOTS"),
        )
        successor = readiness["next_executable_successor_requirements"]
        self.assertIs(type(successor["source_corpus_if_fully_retained_bytes"]), int)
        self.assertEqual(
            successor["source_corpus_if_fully_retained_bytes"], 1_843_200_002_880
        )
        self.assertEqual(successor["source_corpus_decimal_terabytes"], "1.843200002880")
        self.assertIn("DETERMINISTICALLY_REGENERATE", successor["source_custody"])
        adapter = self.protocol["prerequisite_custody"]["semantic_gate_summary"]["grid_adapter_v1"]
        self.assertEqual(adapter["source_corpus_if_fully_retained_bytes"], 1_843_200_002_880)
        self.assertIs(adapter["source_corpus_retained"], False)
        self.assertIs(adapter["deterministic_source_chunk_replay_specified"], True)

    def test_writer_v3_commit_tag_archive_and_content_are_exact(self) -> None:
        custody = self.protocol["prerequisite_custody"]
        objects = {
            value["role"]: value for value in custody["annotated_git_objects"]
        }
        writer_result = objects["writer_v3_result"]
        self.assertEqual(
            writer_result["tag_object"],
            "b8a441cf8e126174e9901ad256d03b816ce0762d",
        )
        self.assertEqual(
            writer_result["peeled_commit"],
            "b4bd47ccee94be51aa8606c73dfc8d447f6a6d65",
        )
        files = {
            item["role"]: item
            for item in custody["exact_files_by_gate"]["writer_v3"]
        }
        self.assertEqual(
            files["digest_roster"]["sha256"],
            "0c07031eeef5488b700486dc3007105143e3c4c5cc5514e7365bcfbdbb4cc0e3",
        )
        self.assertEqual(
            files["result"]["sha256"],
            "c63d880e14c172130de57d6469027809cb1d50dc5900f103b8757cad08f25a07",
        )
        summary = custody["semantic_gate_summary"]["writer_v3"]
        self.assertEqual(
            summary["result_content_sha256"],
            "dd6d9fe6528e9353a38029ec3e7a06260857c4ab3dfd861423b20a54d4a84a49",
        )
        self.assertIs(summary["production_scale_synthetic_writer_qualified"], True)
        self.assertIs(summary["real_dynamics_adapter_qualified"], False)
        interpretation = self.protocol["preregistration"]["git_custody_interpretation"]
        self.assertIn("CURRENT_LOCAL_ANNOTATED_RESULT_TAG_OBJECTS", interpretation)
        self.assertIn("DID_NOT_SELF_AUTHENTICATE", interpretation)

    def test_every_file_pin_is_unique_exact_and_commit_bound(self) -> None:
        m = self.module
        identities = [(pin.gate, pin.role) for pin in m.FILE_PINS]
        self.assertEqual(len(identities), len(set(identities)))
        paths = [pin.relative_path for pin in m.FILE_PINS]
        self.assertEqual(len(paths), len(set(paths)))
        records = self.protocol["prerequisite_custody"]["exact_files_by_gate"]
        self.assertEqual(sum(map(len, records.values())), len(m.FILE_PINS))
        for gate_records in records.values():
            for record in gate_records:
                self.assertIs(record["live_equals_committed"], True)
                self.assertIs(type(record["byte_length"]), int)
                self.assertTrue(m._is_sha256(record["sha256"]))

    def test_gate_scopes_do_not_escalate_restart_or_science(self) -> None:
        summary = self.protocol["prerequisite_custody"]["semantic_gate_summary"]
        self.assertEqual(summary["smoke_v3"]["scope"], "BATCH0_FULL129_1KYR_ALL_ACTIVE")
        self.assertIs(summary["smoke_v3"]["ten_myr_authorized"], False)
        self.assertEqual(
            summary["restart_v1"]["scope"],
            "BATCH0_FULL129_ALL_ACTIVE_YEAR500_TO1000",
        )
        self.assertIs(
            summary["restart_v1"]["general_disk_checkpoint_restart_qualified"],
            False,
        )
        self.assertIs(
            summary["restart_v1"]["routed_or_removed_restart_qualified"],
            False,
        )
        self.assertIs(summary["lifecycle_v3"]["encounter_route_qualified"], False)
        self.assertIs(summary["bridge_v1"]["dynamics_executed"], False)
        self.assertIs(summary["grid_v1"]["execution_authorized"], False)

    def test_writer_archive_roster_is_explicitly_bounded(self) -> None:
        policy = self.protocol["prerequisite_custody"]["root_roster_policies"]["writer_v3"]
        self.assertIn(
            ["DIGEST_ROSTER.jsonl", "PROTOCOL.json", "QUALIFICATION_RESULT.json"],
            policy["allowed_rosters"],
        )
        self.assertIs(policy["local_large_payload_directories_optional_for_archive_binding"], True)
        for gate, value in self.protocol["prerequisite_custody"]["root_roster_policies"].items():
            if gate != "writer_v3":
                self.assertIs(
                    value["local_large_payload_directories_optional_for_archive_binding"],
                    False,
                )

    def test_source_and_test_self_receipts_match_exact_bytes(self) -> None:
        implementation = self.protocol["implementation"]
        for role, path in (("source", SOURCE_PATH), ("test", Path(__file__).resolve())):
            payload = path.read_bytes()
            self.assertEqual(implementation[role]["byte_length"], len(payload))
            self.assertEqual(implementation[role]["sha256"], _sha256(payload))
            self.assertEqual(
                implementation[role]["relative_path"],
                path.relative_to(PROJECT_ROOT).as_posix(),
            )

    def test_strict_json_rejects_duplicates_nonfinite_and_nonobjects(self) -> None:
        m = self.module
        for payload in (
            b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n',
            b'[1,2]\n', b'"object"\n', b'\xff',
        ):
            with self.subTest(payload=payload), self.assertRaises(m.TrajectoryAcquisitionError):
                m._strict_json(payload)

    def test_exact_equality_rejects_bool_integer_aliases(self) -> None:
        m = self.module
        self.assertFalse(m._exact_equal(True, 1))
        self.assertFalse(m._exact_equal(False, 0))
        self.assertFalse(m._exact_equal({"v": True}, {"v": 1}))
        forged = copy.deepcopy(self.protocol)
        forged["production_geometry"]["batch_count"] = True
        forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
        with self.assertRaises(m.TrajectoryAcquisitionError):
            m.validate_protocol(forged)

    def test_claim_or_status_relabel_fails_even_after_coordinated_rehash(self) -> None:
        m = self.module
        pin = m._pin_by("writer_v3", "result")
        payload = (m.PROJECT_ROOT / pin.relative_path).read_bytes()
        original = m._strict_json(payload)
        for mutation in ("claim", "status"):
            forged = copy.deepcopy(original)
            if mutation == "claim":
                forged["claims"]["classification_generated"] = True
            else:
                forged["status"] = "SCIENCE_COMPLETE"
            forged["content_sha256"] = m._domain_hash(
                b"jx.tno.10my-full128-writer-qualification.result.v3\0", forged
            )
            forged_payload = m._canonical_json(forged)
            with self.subTest(mutation=mutation), self.assertRaises(m.TrajectoryAcquisitionError):
                m._validate_canonical_record(
                    forged_payload,
                    schema="jx.tno.10my-full128-writer-qualification.result.v3",
                    status="SYNTHETIC_IO_WRITER_QUALIFICATION_COMPLETE_NO_DYNAMICS_OR_SCIENCE_CLAIM",
                    content_sha256=forged["content_sha256"],
                    domain=b"jx.tno.10my-full128-writer-qualification.result.v3\0",
                    claims=m.WRITER_CLAIMS,
                    label="forged writer result",
                )

    def test_held_file_read_rejects_digest_size_symlink_and_hardlink(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-held-") as temporary:
            root = Path(temporary)
            exact = root / "exact.bin"
            exact.write_bytes(b"exact-payload")
            digest = _sha256(b"exact-payload")
            self.assertEqual(
                m._read_exact_file(exact, 13, digest, "exact fixture"),
                b"exact-payload",
            )
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_exact_file(exact, 12, digest, "wrong size")
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_exact_file(exact, 13, "0" * 64, "wrong digest")
            symlink = root / "symlink.bin"
            symlink.symlink_to(exact.name)
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_exact_file(symlink, 13, digest, "symlink")
            hard = root / "hard.bin"
            os.link(exact, hard)
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_exact_file(exact, 13, digest, "hardlink source")

    def test_unpinned_self_read_is_held_bounded_and_single_link(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-self-") as temporary:
            root = Path(temporary)
            exact = root / "exact.py"
            exact.write_bytes(b"pass\n")
            self.assertEqual(
                m._read_single_link_regular_file(exact, "exact", 5), b"pass\n"
            )
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_single_link_regular_file(exact, "bounded", 4)
            hard = root / "hard.py"
            os.link(exact, hard)
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m._read_single_link_regular_file(exact, "hardlink", 5)

    def test_annotated_tag_tree_tamper_is_rejected(self) -> None:
        m = self.module
        pin = m.GIT_PINS[0]
        forged = replace(pin, commit_tree="0" * 40)
        with self.assertRaises(m.TrajectoryAcquisitionError):
            m._verify_git_pin(forged)

    def test_reverse_result_lineage_is_rejected(self) -> None:
        m = self.module
        with self.assertRaises(m.TrajectoryAcquisitionError):
            m._verify_result_lineage(
                "reverse writer lineage", m.WRITER_RESULT_COMMIT, m.WRITER_PREREG_COMMIT
            )

    def test_prepare_is_create_only_and_validation_is_read_only(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-prepare-") as temporary:
            root = Path(temporary) / "result"
            prepared = m.prepare(root)
            self.assertTrue(m._exact_equal(prepared, self.protocol))
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["PROTOCOL.json"])
            snapshot = _tree_snapshot(root)
            validated = m.validate_prepared(root)
            self.assertTrue(m._exact_equal(validated, prepared))
            self.assertEqual(_tree_snapshot(root), snapshot)
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m.prepare(root)
            self.assertEqual(_tree_snapshot(root), snapshot)
            (root / "unexpected").write_bytes(b"")
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m.validate_prepared(root)

    def test_failed_publication_removes_only_new_attempt(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-fault-") as temporary:
            root = Path(temporary) / "result"
            with mock.patch.object(
                m, "_rename_noreplace", side_effect=m.TrajectoryAcquisitionError("fault")
            ):
                with self.assertRaises(m.TrajectoryAcquisitionError):
                    m.prepare(root)
            self.assertFalse(root.exists())

    def test_claim_tamper_with_rehash_fails_prepared_validation(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-forge-") as temporary:
            root = Path(temporary) / "result"
            m.prepare(root)
            protocol_path = root / "PROTOCOL.json"
            forged = m._strict_json(protocol_path.read_bytes())
            forged["claims"]["science_result"] = True
            forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
            protocol_path.write_bytes(m._canonical_json(forged))
            with self.assertRaises(m.TrajectoryAcquisitionError):
                m.validate_prepared(root)

    def test_fresh_normal_create_then_optimized_validate_is_read_only(self) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONOPTIMIZE", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory(prefix="jx-acquisition-crossmode-") as temporary:
            root = Path(temporary) / "result"
            creator = subprocess.run(
                [sys.executable, "-B", "-c", CHILD, str(SOURCE_PATH), "prepare", str(root)],
                cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True,
            )
            self.assertEqual(creator.returncode, 0, creator.stderr)
            snapshot = _tree_snapshot(root)
            validator = subprocess.run(
                [sys.executable, "-OO", "-B", "-c", CHILD, str(SOURCE_PATH), "validate", str(root)],
                cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True,
            )
            self.assertEqual(validator.returncode, 0, validator.stderr)
            self.assertEqual(creator.stdout.strip(), validator.stdout.strip())
            self.assertEqual(_tree_snapshot(root), snapshot)

    def test_cli_rejects_execute_and_resume(self) -> None:
        m = self.module
        for command in ("execute", "execute-worker", "resume", "validate-result"):
            with self.subTest(command=command):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        m._parser().parse_args([command])


if __name__ == "__main__":
    unittest.main()
