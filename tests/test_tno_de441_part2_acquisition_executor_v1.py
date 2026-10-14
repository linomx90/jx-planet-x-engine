from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_part2_acquisition_executor_v1.py"


def _load_source_module() -> types.ModuleType:
    name = "jx_tno_de441_part2_acquisition_executor_v1_tested_source"
    payload = SCRIPT_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[name] = module
    exec(compile(payload, str(SCRIPT_PATH), "exec"), module.__dict__)
    return module


EXECUTOR = _load_source_module()
FIXTURE_BYTES = (b"JX-DE441-FIXTURE\x00" * 4096) + b"EOF"
PUBLISHED_AT = "2026-10-14T19:20:21Z"


class FakeSemanticBackend:
    def __init__(self, payload: bytes = FIXTURE_BYTES) -> None:
        self.payload = payload
        self.calls = 0

    def runtime_record(self) -> dict[str, object]:
        return {
            "runtime": "PRIVATE_TEST_FIXTURE_ONLY",
            "network_access": False,
            "production_runtime_claimed": False,
        }

    def inspect(self, held_fd: int, contract: object) -> dict[str, object]:
        self.calls += 1
        observed = bytearray()
        offset = 0
        while offset < len(self.payload):
            block = os.pread(held_fd, min(4096, len(self.payload) - offset), offset)
            if not block:
                raise EXECUTOR.AcquisitionExecutorError("fixture semantic bytes ended")
            observed.extend(block)
            offset += len(block)
        if bytes(observed) != self.payload:
            raise EXECUTOR.AcquisitionExecutorError("fixture semantic bytes differ")
        return {
            "architecture": "DAF",
            "kernel_type": "SPK",
            "fixture_sha256": hashlib.sha256(self.payload).hexdigest(),
            "status": "PASSED_PRIVATE_FIXTURE_ONLY",
        }


class FakeSpice:
    def __init__(self) -> None:
        self.rows = list(EXECUTOR.STORED_TARGET_CENTER_ROSTER)
        self.index = -1
        self.frame = 1
        self.coverage_start = EXECUTOR._jd_to_et(EXECUTOR.KERNEL_COVERAGE_JD_TDB[0])
        self.coverage_stop = EXECUTOR._jd_to_et(EXECUTOR.KERNEL_COVERAGE_JD_TDB[1])

    def getfat(self, path: str) -> tuple[str, str]:
        return ("DAF", "SPK")

    def spkobj(self, path: str) -> list[int]:
        return [row[0] for row in self.rows]

    def dafopr(self, path: str) -> int:
        return 7

    def dafbfs(self, handle: int) -> None:
        self.index = -1

    def daffna(self) -> bool:
        self.index += 1
        return self.index < len(self.rows)

    def dafgs(self, count: int) -> list[float]:
        return [float(self.index)] * count

    def dafgn(self, length: int) -> str:
        return f"DE441 TEST SEGMENT {self.index:02d}"

    def dafus(self, summary: object, doubles: int, integers: int):
        target, center, _semantic = self.rows[self.index]
        return (
            [self.coverage_start, self.coverage_stop],
            [
                target,
                center,
                self.frame,
                2,
                100 + 10 * self.index,
                109 + 10 * self.index,
            ],
        )

    def dafcls(self, handle: int) -> None:
        return None

    def spkcov(self, path: str, target: int) -> int:
        return target

    def wncard(self, window: int) -> int:
        return 1

    def wnfetd(self, window: int, index: int) -> tuple[float, float]:
        return self.coverage_start, self.coverage_stop

    def kclear(self) -> None:
        return None

    def failed(self) -> bool:
        return False

    def ktotal(self, kind: str) -> int:
        return 0


class FixtureLayout:
    def __init__(self, directory: str, payload: bytes = FIXTURE_BYTES) -> None:
        self.base = Path(directory)
        self.worktree = self.base / "worktree"
        self.asset = self.base / "asset-root"
        self.stage = self.asset / "staging"
        self.staged = self.stage / EXECUTOR.KERNEL_FILENAME
        self.receipt_parent = self.base / "receipts"
        self.custody = self.receipt_parent / "CUSTODY.json"
        self.worktree.mkdir(mode=0o700)
        self.asset.mkdir(mode=0o700)
        self.stage.mkdir(mode=0o700)
        self.receipt_parent.mkdir(mode=0o700)
        self.staged.write_bytes(payload)
        self.staged.chmod(0o600)

    @property
    def published(self) -> Path:
        digest = hashlib.sha256(FIXTURE_BYTES).hexdigest()
        return self.asset / "objects" / "sha256" / digest / EXECUTOR.KERNEL_FILENAME


def _fixture_contract(payload: bytes = FIXTURE_BYTES):
    return EXECUTOR._fixture_contract(payload, EXECUTOR._FIXTURE_CAPABILITY)


def _ingest(layout: FixtureLayout, backend: FakeSemanticBackend | None = None, **kwargs):
    selected = backend if backend is not None else FakeSemanticBackend()
    value = EXECUTOR.ingest_staged_file(
        layout.asset,
        layout.staged,
        PUBLISHED_AT,
        custody_path=layout.custody,
        _contract=_fixture_contract(),
        _backend=selected,
        _asset_project_root=layout.worktree,
        **kwargs,
    )
    return value, selected


def _validate(layout: FixtureLayout, backend: FakeSemanticBackend | None = None):
    selected = backend if backend is not None else FakeSemanticBackend()
    value = EXECUTOR.validate_custody(
        layout.asset,
        custody_path=layout.custody,
        _contract=_fixture_contract(),
        _backend=selected,
        _asset_project_root=layout.worktree,
    )
    return value, selected


class TnoDe441Part2AcquisitionExecutorV1Tests(unittest.TestCase):
    def test_exact_preregistration_runtime_constants_and_cli_surface(self) -> None:
        self.assertEqual(
            EXECUTOR.PREREGISTRATION_TAG_OBJECT,
            "0a6d1c752489d08f73a318cf8de16942998a6992",
        )
        self.assertEqual(
            EXECUTOR.PREREGISTRATION_COMMIT,
            "b86f8da9ffdd44a89dadb2a56bbb565919400f3a",
        )
        self.assertEqual(
            EXECUTOR.PREREGISTRATION_TREE,
            "980243fd12b40cf0939102ceaeb70a58daa89342",
        )
        self.assertEqual(EXECUTOR.KERNEL_BYTE_LENGTH, 1_656_830_976)
        self.assertEqual(
            EXECUTOR.KERNEL_OFFICIAL_MD5, "ad8dfa4e505ef0e3a5d587a5b4705632"
        )
        self.assertEqual(EXECUTOR.SPICEYPY_VERSION, "8.2.0")
        self.assertEqual(EXECUTOR.CSPICE_TOOLKIT_VERSION, "CSPICE_N0067")
        self.assertEqual(
            EXECUTOR.CSPICE_LIBRARY_SHA256,
            "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84",
        )
        parser = EXECUTOR._parser()
        self.assertEqual(
            set(parser._subparsers._group_actions[0].choices),
            {"audit-preregistration", "ingest", "validate-custody"},
        )
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertTrue(
            {"urllib", "requests", "httpx", "socket", "ftplib"}.isdisjoint(imported)
        )
        for forbidden in ("download", "resume", "urlopen", "requests.get"):
            self.assertNotIn(forbidden, set(parser._subparsers._group_actions[0].choices))

    def test_real_sealed_preregistration_audit_and_exact_bindings(self) -> None:
        value = EXECUTOR.audit_preregistration()
        self.assertEqual(value["tag_object"], EXECUTOR.PREREGISTRATION_TAG_OBJECT)
        self.assertEqual(value["commit"], EXECUTOR.PREREGISTRATION_COMMIT)
        self.assertEqual(value["tree"], EXECUTOR.PREREGISTRATION_TREE)
        self.assertEqual(len(value["bindings"]), 4)
        self.assertEqual(
            [row["sha256"] for row in value["bindings"]],
            [row.sha256 for row in EXECUTOR.PREREGISTRATION_BINDINGS],
        )
        process = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "audit-preregistration"],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "PASSED")

    def test_authority_failure_precedes_staged_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            with mock.patch.object(
                EXECUTOR,
                "audit_preregistration",
                side_effect=EXECUTOR.AcquisitionExecutorError("authority fault"),
            ), mock.patch.object(EXECUTOR, "_open_staged") as opened:
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "authority fault"
                ):
                    _ingest(layout)
                opened.assert_not_called()
                self.assertFalse(layout.custody.exists())
                self.assertFalse(layout.published.exists())

    def test_final_authority_reaudit_precedes_receipt_publication(self) -> None:
        authority = EXECUTOR.audit_preregistration()
        changed = dict(authority)
        changed["tree"] = "0" * 40
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            with mock.patch.object(
                EXECUTOR,
                "audit_preregistration",
                side_effect=(authority, changed),
            ):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError,
                    "authority changed during acquisition",
                ):
                    _ingest(layout)
            self.assertTrue(layout.published.is_file())
            self.assertFalse(layout.custody.exists())

    def test_fixture_ingest_content_address_and_offline_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            custody, backend = _ingest(layout)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(custody, json.loads(layout.custody.read_bytes()))
            self.assertEqual(stat.S_IMODE(layout.custody.stat().st_mode), 0o600)
            self.assertTrue(layout.published.is_file())
            self.assertEqual(layout.published.read_bytes(), FIXTURE_BYTES)
            self.assertEqual(stat.S_IMODE(layout.published.stat().st_mode), 0o600)
            self.assertEqual(layout.published.stat().st_nlink, 1)
            self.assertEqual(layout.staged.stat().st_nlink, 1)
            self.assertNotEqual(layout.published.stat().st_ino, layout.staged.stat().st_ino)
            digest = hashlib.sha256(FIXTURE_BYTES).hexdigest()
            self.assertEqual(
                custody["artifact"]["logical_locator"],
                f"objects/sha256/{digest}/{EXECUTOR.KERNEL_FILENAME}",
            )
            self.assertNotIn(str(layout.base), layout.custody.read_text(encoding="ascii"))
            self.assertEqual(custody["schema"], EXECUTOR.FIXTURE_CUSTODY_SCHEMA)
            self.assertEqual(
                custody["classification"], EXECUTOR.FIXTURE_CLASSIFICATION
            )
            self.assertEqual(
                custody["status"],
                "PASSED_PRIVATE_TEST_FIXTURE_BYTE_PIPELINE_NO_DE441_CUSTODY",
            )
            self.assertIs(custody["source"]["official_source_used"], False)
            self.assertIs(custody["claims"]["private_test_fixture_only"], True)
            self.assertIs(custody["claims"]["kernel_custody_established"], False)
            self.assertIs(custody["claims"]["spk_semantics_validated"], False)
            for claim in (
                "provider_transport_authenticated",
                "continuous_custody_proven",
                "source_evaluator_qualified",
                "source_node_gate_passed",
                "common_epoch_bridge_executed",
                "dynamics_executed",
                "gpu_executed",
                "classification_generated",
                "population_inference",
                "planet_nine_evidence",
                "science_result",
                "scientific_execution_authorized",
            ):
                self.assertIs(custody["claims"][claim], False)
            validated, validator = _validate(layout)
            self.assertEqual(validated, custody)
            self.assertEqual(validator.calls, 1)

    def test_staging_symlink_hardlink_mode_size_and_digest_fail_closed(self) -> None:
        cases = ("symlink", "hardlink", "mode", "size", "digest")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                layout = FixtureLayout(directory)
                if case == "symlink":
                    target = layout.stage / "target"
                    layout.staged.rename(target)
                    layout.staged.symlink_to(target.name)
                elif case == "hardlink":
                    os.link(layout.staged, layout.stage / "second-link")
                elif case == "mode":
                    layout.staged.chmod(0o640)
                elif case == "size":
                    layout.staged.write_bytes(FIXTURE_BYTES[:-1])
                    layout.staged.chmod(0o600)
                elif case == "digest":
                    changed = b"X" + FIXTURE_BYTES[1:]
                    layout.staged.write_bytes(changed)
                    layout.staged.chmod(0o600)
                with self.assertRaises(EXECUTOR.AcquisitionExecutorError):
                    _ingest(layout)
                self.assertFalse(layout.custody.exists())
                self.assertFalse(layout.published.exists())

    def test_staged_path_swap_after_open_is_detected_from_held_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)

            def swap(event: str) -> None:
                if event == "staged-opened":
                    layout.staged.rename(layout.stage / "displaced-original")
                    layout.staged.write_bytes(FIXTURE_BYTES)
                    layout.staged.chmod(0o600)

            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "changed while held"
            ):
                _ingest(layout, _fault_hook=swap)
            self.assertFalse(layout.custody.exists())
            self.assertFalse(layout.published.exists())

    def test_root_and_staging_directory_privacy_fail_closed(self) -> None:
        for target in ("root", "staging"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                layout = FixtureLayout(directory)
                (layout.asset if target == "root" else layout.stage).chmod(0o750)
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "private"
                ):
                    _ingest(layout)
                self.assertFalse(layout.custody.exists())

    def test_existing_destination_and_copy_fault_never_publish_custody(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            layout.published.parent.mkdir(mode=0o700, parents=True)
            for private_directory in (
                layout.asset / "objects",
                layout.asset / "objects" / "sha256",
                layout.published.parent,
            ):
                private_directory.chmod(0o700)
            layout.published.write_bytes(FIXTURE_BYTES)
            layout.published.chmod(0o600)
            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "already exists"
            ):
                _ingest(layout)
            self.assertFalse(layout.custody.exists())
            self.assertEqual(layout.published.read_bytes(), FIXTURE_BYTES)

        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            with mock.patch.object(
                EXECUTOR,
                "_copy_fd",
                side_effect=EXECUTOR.AcquisitionExecutorError("copy fault"),
            ):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "copy fault"
                ):
                    _ingest(layout)
            self.assertFalse(layout.custody.exists())
            self.assertFalse(layout.published.exists())
            digest_parent = layout.published.parent
            self.assertFalse((digest_parent / f".{EXECUTOR.KERNEL_FILENAME}.partial").exists())

    def test_post_publication_fault_retains_object_but_never_claims_custody(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)

            def fail(event: str) -> None:
                if event == "published-independently-validated":
                    raise EXECUTOR.AcquisitionExecutorError("post-publication fault")

            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "post-publication fault"
            ):
                _ingest(layout, _fault_hook=fail)
            self.assertTrue(layout.published.is_file())
            self.assertFalse(layout.custody.exists())

    def test_content_address_directory_detach_before_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            detached = layout.asset / "detached-objects"

            def detach(event: str) -> None:
                if event == "published-independently-validated":
                    (layout.asset / "objects").rename(detached)
                    (layout.asset / "objects").mkdir(mode=0o700)

            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "objects.*held private directory"
            ):
                _ingest(layout, _fault_hook=detach)
            self.assertFalse(layout.custody.exists())
            self.assertTrue(
                (detached / "sha256" / hashlib.sha256(FIXTURE_BYTES).hexdigest()
                 / EXECUTOR.KERNEL_FILENAME).is_file()
            )

    def test_ambiguous_rename_is_fully_checked_retained_and_unclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            backend = FakeSemanticBackend()
            original = EXECUTOR._rename_noreplace

            def ambiguous(directory_fd: int, source: str, destination: str) -> None:
                original(directory_fd, source, destination)
                if destination == EXECUTOR.KERNEL_FILENAME:
                    raise OSError("injected lost rename acknowledgement")

            with mock.patch.object(EXECUTOR, "_rename_noreplace", side_effect=ambiguous):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "ambiguous publication"
                ):
                    _ingest(layout, backend=backend)
            self.assertEqual(backend.calls, 2)
            self.assertTrue(layout.published.is_file())
            self.assertEqual(layout.published.read_bytes(), FIXTURE_BYTES)
            self.assertFalse(layout.custody.exists())

    def test_ambiguous_custody_rename_resolves_exact_validated_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "receipt"
            parent.mkdir(mode=0o700)
            destination = parent / "CUSTODY.json"
            payload = b'{"fixture":"canonical-bytes"}\n'
            original = EXECUTOR._rename_noreplace

            def ambiguous(directory_fd: int, source: str, target: str) -> None:
                original(directory_fd, source, target)
                raise OSError("injected lost custody acknowledgement")

            with mock.patch.object(EXECUTOR, "_rename_noreplace", side_effect=ambiguous):
                EXECUTOR._create_custody_last(
                    destination,
                    payload,
                    _fixture_contract(),
                )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse((parent / ".CUSTODY.json.partial").exists())

    def test_custody_parent_detach_and_post_rename_tamper_leave_no_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parent = base / "receipt"
            detached = base / "detached-receipt"
            parent.mkdir(mode=0o700)
            destination = parent / "CUSTODY.json"
            payload = b'{"fixture":"canonical-bytes"}\n'

            def detach() -> None:
                parent.rename(detached)
                parent.mkdir(mode=0o700)

            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "custody parent.*held directory"
            ):
                EXECUTOR._create_custody_last(
                    destination,
                    payload,
                    _fixture_contract(),
                    final_gate=detach,
                )
            self.assertFalse(destination.exists())
            self.assertFalse((detached / "CUSTODY.json").exists())

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "receipt"
            parent.mkdir(mode=0o700)
            destination = parent / "CUSTODY.json"
            payload = b'{"fixture":"canonical-bytes"}\n'
            original_read = EXECUTOR._read_fd_all

            def corrupt(fd: int, size: int, label: str) -> bytes:
                observed = original_read(fd, size, label)
                if label == "published custody receipt":
                    return b"X" + observed[1:]
                return observed

            with mock.patch.object(EXECUTOR, "_read_fd_all", side_effect=corrupt):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "receipt bytes differ"
                ):
                    EXECUTOR._create_custody_last(
                        destination,
                        payload,
                        _fixture_contract(),
                    )
            self.assertFalse(destination.exists())
            self.assertFalse((parent / ".CUSTODY.json.partial").exists())

    def test_kernel_mutation_during_custody_rename_rolls_back_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            original = EXECUTOR._rename_noreplace

            def mutate_after_rename(
                directory_fd: int, source: str, destination: str
            ) -> None:
                original(directory_fd, source, destination)
                if destination == "CUSTODY.json":
                    layout.published.write_bytes(b"X" + FIXTURE_BYTES[1:])
                    layout.published.chmod(0o600)

            with mock.patch.object(
                EXECUTOR, "_rename_noreplace", side_effect=mutate_after_rename
            ):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError,
                    "published kernel.*changed",
                ):
                    _ingest(layout)
            self.assertTrue(layout.published.is_file())
            self.assertFalse(layout.custody.exists())
            self.assertFalse(
                (layout.receipt_parent / ".CUSTODY.json.partial").exists()
            )

    def test_roster_mutation_during_postpublication_gate_rolls_back_receipt(self) -> None:
        authority = EXECUTOR.audit_preregistration()
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            calls = 0

            def authority_with_late_extra() -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 3:
                    extra = layout.receipt_parent / "unregistered"
                    extra.write_bytes(b"extra")
                    extra.chmod(0o600)
                return authority

            with mock.patch.object(
                EXECUTOR,
                "audit_preregistration",
                side_effect=authority_with_late_extra,
            ):
                with self.assertRaisesRegex(
                    EXECUTOR.AcquisitionExecutorError, "post-gate custody roster differs"
                ):
                    _ingest(layout)
            self.assertFalse(layout.custody.exists())
            self.assertTrue((layout.receipt_parent / "unregistered").is_file())

    def test_offline_validator_rejects_kernel_tamper_mode_link_and_receipt_tamper(self) -> None:
        cases = ("bytes", "mode", "hardlink", "directory-mode", "extra", "receipt")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                layout = FixtureLayout(directory)
                _ingest(layout)
                if case == "bytes":
                    layout.published.write_bytes(b"X" + FIXTURE_BYTES[1:])
                    layout.published.chmod(0o600)
                elif case == "mode":
                    layout.published.chmod(0o640)
                elif case == "hardlink":
                    os.link(layout.published, layout.published.parent / "second-link")
                elif case == "directory-mode":
                    layout.published.parent.chmod(0o750)
                elif case == "extra":
                    extra = layout.published.parent / "unregistered"
                    extra.write_bytes(b"extra")
                    extra.chmod(0o600)
                elif case == "receipt":
                    value = json.loads(layout.custody.read_bytes())
                    value["claims"]["science_result"] = True
                    layout.custody.write_text(
                        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="ascii",
                    )
                    layout.custody.chmod(0o600)
                with self.assertRaises(EXECUTOR.AcquisitionExecutorError):
                    _validate(layout)

    def test_offline_validator_detects_midrun_asset_and_receipt_directory_detach(self) -> None:
        for target in ("asset", "receipt"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                layout = FixtureLayout(directory)
                _ingest(layout)
                backend = FakeSemanticBackend()
                original_inspect = backend.inspect

                def inspect(held_fd: int, contract: object) -> dict[str, object]:
                    value = original_inspect(held_fd, contract)
                    if target == "asset":
                        source = layout.asset / "objects"
                        source.rename(layout.asset / "detached-objects")
                        source.mkdir(mode=0o700)
                    else:
                        layout.receipt_parent.rename(layout.base / "detached-receipts")
                        layout.receipt_parent.mkdir(mode=0o700)
                    return value

                backend.inspect = inspect  # type: ignore[method-assign]
                with self.assertRaises(EXECUTOR.AcquisitionExecutorError):
                    _validate(layout, backend=backend)

    def test_noncanonical_and_absolute_receipt_values_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = FixtureLayout(directory)
            _ingest(layout)
            value = json.loads(layout.custody.read_bytes())
            layout.custody.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
            layout.custody.chmod(0o600)
            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "not canonical"
            ):
                _validate(layout)

        value = {"x": "/home/person/secret"}
        with self.assertRaisesRegex(EXECUTOR.AcquisitionExecutorError, "absolute path"):
            EXECUTOR._reject_absolute_receipt_values(value)

    def test_private_fixture_capability_cannot_change_production_constants(self) -> None:
        with self.assertRaisesRegex(
            EXECUTOR.AcquisitionExecutorError, "fixture capability"
        ):
            EXECUTOR._fixture_contract(FIXTURE_BYTES, object())
        forged = EXECUTOR._KernelContract(
            EXECUTOR.KERNEL_ARTIFACT_ID,
            EXECUTOR.KERNEL_FILENAME,
            1,
            "0" * 32,
            EXECUTOR.KERNEL_COVERAGE_JD_TDB,
            True,
            None,
        )
        with self.assertRaisesRegex(
            EXECUTOR.AcquisitionExecutorError, "unrecognized"
        ):
            EXECUTOR._assert_contract(forged)
        self.assertIs(EXECUTOR._PRODUCTION_CONTRACT.production, True)
        self.assertEqual(EXECUTOR._PRODUCTION_CONTRACT.byte_length, 1_656_830_976)
        with self.assertRaisesRegex(
            EXECUTOR.AcquisitionExecutorError, "private fixture cannot target production"
        ):
            EXECUTOR._enforce_path_overrides(
                _fixture_contract(),
                EXECUTOR.DEFAULT_CUSTODY_PATH,
                PROJECT_ROOT,
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "override is forbidden"
            ):
                EXECUTOR.ingest_staged_file(
                    Path(directory) / "asset",
                    Path(directory) / EXECUTOR.KERNEL_FILENAME,
                    PUBLISHED_AT,
                    custody_path=Path(directory) / "CUSTODY.json",
                )

    def test_exact_spk_semantic_roster_frame_type_and_coverage_logic(self) -> None:
        backend = object.__new__(EXECUTOR._PinnedSpiceBackend)
        backend._spice = FakeSpice()
        backend._libc = types.SimpleNamespace(fegetround=lambda: 0)
        fd = os.open("/dev/null", os.O_RDONLY)
        try:
            value = backend.inspect(fd, EXECUTOR._PRODUCTION_CONTRACT)
        finally:
            os.close(fd)
        self.assertEqual(value["architecture"], "DAF")
        self.assertEqual(value["kernel_type"], "SPK")
        self.assertEqual(value["reference_frame_naif_id"], 1)
        self.assertEqual(
            {(row["target_naif_id"], row["center_naif_id"]) for row in value["segments"]},
            {(row[0], row[1]) for row in EXECUTOR.STORED_TARGET_CENTER_ROSTER},
        )
        self.assertIs(value["all_required_chains_covered_at_every_query_time"], True)

        for mutation, message in (("frame", "descriptor"), ("coverage", "coverage"), ("roster", "roster")):
            fake = FakeSpice()
            if mutation == "frame":
                fake.frame = 17
            elif mutation == "coverage":
                fake.coverage_stop -= 1.0
            elif mutation == "roster":
                fake.rows.pop()
            backend = object.__new__(EXECUTOR._PinnedSpiceBackend)
            backend._spice = fake
            backend._libc = types.SimpleNamespace(fegetround=lambda: 0)
            fd = os.open("/dev/null", os.O_RDONLY)
            try:
                with self.assertRaisesRegex(EXECUTOR.AcquisitionExecutorError, message):
                    backend.inspect(fd, EXECUTOR._PRODUCTION_CONTRACT)
            finally:
                os.close(fd)

    def test_pinned_runtime_rejects_preconfigured_library_without_mutating_it(self) -> None:
        sentinel = "/private/untrusted/libcspice.so"
        with mock.patch.dict(os.environ, {"CSPICE_SHARED_LIB": sentinel}, clear=False):
            with self.assertRaisesRegex(
                EXECUTOR.AcquisitionExecutorError, "already configured"
            ):
                EXECUTOR._PinnedSpiceBackend(
                    Path("/does/not/matter.whl"),
                    Path("/does/not/matter.so"),
                )
            self.assertEqual(os.environ["CSPICE_SHARED_LIB"], sentinel)


if __name__ == "__main__":
    unittest.main()
