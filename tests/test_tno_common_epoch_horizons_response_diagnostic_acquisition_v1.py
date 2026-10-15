from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "tools"
    / "jx_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "jx_tno_common_epoch_horizons_response_diagnostic_acquisition_v1_tested",
    SOURCE,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load diagnostic acquisition executor")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContractTests(unittest.TestCase):
    def test_exact_prepare_identity_and_claim_ceiling(self) -> None:
        identity = MODULE._preregistration_identity()
        self.assertEqual(identity["commit"], MODULE.PREPARE_COMMIT)
        self.assertEqual(identity["tree"], MODULE.PREPARE_TREE)
        self.assertEqual(identity["tag_object"], MODULE.PREPARE_TAG_OBJECT)
        self.assertEqual(identity["protocol_sha256"], MODULE.PREPARE_PROTOCOL_SHA256)
        self.assertEqual(set(MODULE.CLAIMS.values()), {False})
        self.assertEqual(set(MODULE.NONCLAIMS.values()), {False})

    def test_protocol_exactly_matches_executor_constants(self) -> None:
        payload = (ROOT / MODULE.PREPARE_PROTOCOL_RELATIVE).read_bytes()
        self.assertEqual(len(payload), MODULE.PREPARE_PROTOCOL_BYTES)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), MODULE.PREPARE_PROTOCOL_SHA256)
        protocol = MODULE._strict_json(payload, "protocol", MODULE.PREPARE_PROTOCOL_BYTES)
        MODULE._verify_protocol_contract(protocol)
        self.assertEqual(payload, MODULE._canonical_json(protocol))

    def test_attempt_is_exact_canonical_domain_hashed_json(self) -> None:
        value = MODULE._make_attempt()
        MODULE._validate_attempt_record(value)
        payload = MODULE._canonical_json(value)
        replay = MODULE._strict_json(payload, "attempt", MODULE.MAX_ATTEMPT_BYTES)
        self.assertEqual(replay, value)
        self.assertEqual(value["content_sha256"], MODULE._domain_hash(MODULE.ATTEMPT_DOMAIN, value))
        self.assertEqual(value["request_budget"]["maximum_request_invocations"], 1)
        self.assertEqual(value["request_budget"]["retries"], 0)

    def test_attempt_rejects_bool_byte_count_and_hash_mutation(self) -> None:
        value = MODULE._make_attempt()
        value["preregistration_identity"] = dict(value["preregistration_identity"])
        value["preregistration_identity"]["source_bytes"] = True
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE._validate_attempt_record(value)
        value = MODULE._make_attempt()
        value["content_sha256"] = "0" * 64
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE._validate_attempt_record(value)

    def test_strict_json_rejects_duplicates_nonfinite_and_nonobject(self) -> None:
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
            with self.subTest(payload=payload):
                with self.assertRaises(MODULE.AcquisitionError):
                    MODULE._strict_json(payload, "hostile", 100)

    def test_diagnosis_total_for_hostile_and_lone_surrogate_payloads(self) -> None:
        payloads = (
            b"",
            b"not-json",
            b"[]",
            b'{"result":"\\ud800"}',
            json.dumps({"result": "$$EOE x $$SOE"}).encode(),
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                value = MODULE._diagnosis(payload)
                MODULE._validate_diagnosis(value)
                self.assertIn(
                    value["v1_gate_outcome"],
                    {
                        "REJECTED_BY_FROZEN_V1_GATE",
                        "FROZEN_V1_VALIDATOR_INTERNAL_EXCEPTION",
                    },
                )

    def test_reversed_delimiters_preserve_strict_object_and_result_hash(self) -> None:
        payload = json.dumps({"result": "$$EOE x $$SOE"}).encode()
        with mock.patch.object(
            MODULE, "_v1_gate", return_value=("REJECTED_BY_FROZEN_V1_GATE", "x")
        ):
            value = MODULE._diagnosis(payload)
        self.assertEqual(value["json_envelope_status"], "STRICT_OBJECT")
        self.assertRegex(value["result_string_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(value["state_vector_epoch_token_candidates"], [])

    def test_custody_accepts_zero_byte_raw_and_keeps_every_claim_false(self) -> None:
        attempt = MODULE._make_attempt()
        attempt_payload = MODULE._canonical_json(attempt)
        with mock.patch.object(
            MODULE, "_v1_gate", return_value=("REJECTED_BY_FROZEN_V1_GATE", "empty")
        ):
            diagnosis = MODULE._diagnosis(b"")
        value = MODULE._make_custody(b"", diagnosis, attempt_payload, attempt)
        MODULE._validate_custody(value)
        self.assertEqual(value["response"]["raw_bytes"], 0)
        self.assertEqual(set(value["claims"].values()), {False})
        self.assertEqual(set(value["nonclaims"].values()), {False})

    def test_diagnosis_rejects_inconsistent_error_message(self) -> None:
        value = {
            "element_epoch_line_candidates": [],
            "json_envelope_status": "STRICT_JSON_REJECTED",
            "result_string_sha256": None,
            "state_vector_epoch_token_candidates": [],
            "v1_gate_error_message": "not allowed on pass",
            "v1_gate_outcome": "PASS_DIAGNOSTIC_ONLY",
        }
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE._validate_diagnosis(value)

    def test_public_cli_has_no_resume_or_oracle_command(self) -> None:
        parser = MODULE._parser()
        choices = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        self.assertEqual(
            set(choices),
            {"acquire", "audit-prerequisites", "validate-attempt", "validate-result"},
        )
        self.assertNotIn("resume", choices)


class GitAuthorityTests(unittest.TestCase):
    @staticmethod
    def _commit_payload(parent: str, tree: str = "1" * 40) -> bytes:
        return (
            f"tree {tree}\n"
            f"parent {parent}\n"
            "author A <a@example.invalid> 1 +0000\n"
            "committer A <a@example.invalid> 1 +0000\n\n"
            "message\n"
        ).encode("ascii")

    def test_raw_commit_proof_rehashes_and_requires_exact_parent(self) -> None:
        payload = self._commit_payload(MODULE.PREPARE_COMMIT)
        commit = MODULE._git_object_sha1(b"commit", payload)
        with mock.patch.object(MODULE, "_git", return_value=payload) as git:
            tree = MODULE._raw_direct_child_tree(commit, MODULE.PREPARE_COMMIT, "candidate")
        self.assertEqual(tree, "1" * 40)
        self.assertEqual(git.call_args.args, ("cat-file", "commit", commit))

    def test_raw_commit_rejects_hash_substitution_wrong_parent_and_merges(self) -> None:
        valid = self._commit_payload(MODULE.PREPARE_COMMIT)
        cases = [
            ("0" * 40, valid),
            (
                MODULE._git_object_sha1(b"commit", self._commit_payload("2" * 40)),
                self._commit_payload("2" * 40),
            ),
        ]
        merged = valid.replace(
            f"parent {MODULE.PREPARE_COMMIT}\n".encode(),
            (
                f"parent {MODULE.PREPARE_COMMIT}\n"
                f"parent {'3' * 40}\n"
            ).encode(),
        )
        cases.append((MODULE._git_object_sha1(b"commit", merged), merged))
        for commit, payload in cases:
            with self.subTest(commit=commit):
                with mock.patch.object(MODULE, "_git", return_value=payload):
                    with self.assertRaises(MODULE.AcquisitionError):
                        MODULE._raw_direct_child_tree(commit, MODULE.PREPARE_COMMIT, "candidate")

    def test_raw_commit_authority_never_uses_rev_list(self) -> None:
        payload = self._commit_payload(MODULE.PREPARE_COMMIT)
        commit = MODULE._git_object_sha1(b"commit", payload)

        def fake_git(*arguments: str) -> bytes:
            self.assertNotIn("rev-list", arguments)
            self.assertEqual(arguments[:2], ("cat-file", "commit"))
            return payload

        with mock.patch.object(MODULE, "_git", side_effect=fake_git):
            self.assertEqual(
                MODULE._raw_direct_child_tree(commit, MODULE.PREPARE_COMMIT, "candidate"),
                "1" * 40,
            )

    def test_git_strips_unsafe_environment_and_is_bounded(self) -> None:
        completed = types.SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")
        with mock.patch.dict(
            os.environ,
            {
                "GIT_DIR": "/attacker",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "alias.x",
                "GIT_CONFIG_VALUE_0": "evil",
            },
            clear=False,
        ), mock.patch.object(subprocess, "run", return_value=completed) as run:
            self.assertEqual(MODULE._git("status"), b"ok")
        kwargs = run.call_args.kwargs
        command = run.call_args.args[0]
        self.assertEqual(kwargs["timeout"], 30)
        self.assertEqual(kwargs["env"]["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
        self.assertNotIn("GIT_DIR", kwargs["env"])
        self.assertNotIn("GIT_CONFIG_COUNT", kwargs["env"])
        self.assertIn("core.fsmonitor=false", command)
        self.assertIn("core.untrackedCache=false", command)

    def test_executor_tag_is_raw_rehashed_and_bound_to_head(self) -> None:
        head = "4" * 40
        payload = (
            f"object {head}\n"
            "type commit\n"
            f"tag {MODULE.EXECUTOR_TAG}\n"
            "tagger A <a@example.invalid> 1 +0000\n\nmessage\n"
        ).encode()
        object_id = MODULE._git_object_sha1(b"tag", payload)

        def fake_git(*arguments: str) -> bytes:
            if arguments[0] == "rev-parse" and arguments[1].startswith("refs/tags/"):
                return object_id.encode() + b"\n"
            if arguments == ("cat-file", "-t", object_id):
                return b"tag\n"
            if arguments == ("cat-file", "tag", object_id):
                return payload
            if arguments[0] == "rev-parse" and arguments[1].endswith("^{}"):
                return head.encode() + b"\n"
            self.fail(f"unexpected Git call {arguments}")

        with mock.patch.object(MODULE, "_git", side_effect=fake_git):
            record = MODULE._verify_executor_tag(head)
        self.assertEqual(record["tag_object"], object_id)
        self.assertEqual(record["tag_payload_sha256"], hashlib.sha256(payload).hexdigest())

    def test_expected_manifest_is_one_sorted_insertion(self) -> None:
        prepared = MODULE._git("cat-file", "blob", f"{MODULE.PREPARE_TREE}:{MODULE.MANIFEST_RELATIVE}")
        expected = MODULE._expected_manifest(prepared)
        current = (ROOT / MODULE.MANIFEST_RELATIVE).read_bytes()
        self.assertEqual(current, expected)
        manifest = json.loads(current)
        matches = [
            profile
            for profile in manifest["profiles"]
            if MODULE.SUCCESSOR_TEST_RELATIVE in profile["files"]
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["files"], sorted(matches[0]["files"]))

    def test_live_authority_passes_in_precommit_or_sealed_mode(self) -> None:
        value = MODULE.audit_prerequisites()
        mode = value["authority"]["authority_mode"]
        head = MODULE._git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        if head == MODULE.PREPARE_COMMIT:
            self.assertEqual(
                mode,
                "PRECOMMIT_PREPARE_HEAD_WITH_EXACT_THREE_PATH_WORKTREE_ROSTER",
            )
        else:
            self.assertEqual(
                mode,
                "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY",
            )
            self.assertEqual(value["authority"]["head"], head)
            self.assertEqual(value["authority"]["commit_paths"], sorted(MODULE.SUCCESSOR_PATHS))
        self.assertEqual(value["status"], "PASSED_WITH_ZERO_REQUESTS")

    def test_acquisition_refuses_precommit_authority(self) -> None:
        record = {
            "authority_mode": "PRECOMMIT_PREPARE_HEAD_WITH_EXACT_THREE_PATH_WORKTREE_ROSTER"
        }
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with self.assertRaisesRegex(MODULE.AcquisitionError, "sealed executor"):
                MODULE._acquire_to_roots(
                    parent / "result",
                    parent / "attempt",
                    transport=lambda _url: self.fail("transport used"),
                    authority=(record, []),
                )

    def test_no_status_sealed_recheck_detects_head_movement(self) -> None:
        record = {
            "authority_mode": "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY",
            "head": "4" * 40,
        }
        calls: list[tuple[str, ...]] = []

        def fake_git(*arguments: str) -> bytes:
            calls.append(arguments)
            self.assertNotEqual(arguments[0], "status")
            if arguments == ("branch", "--show-current"):
                return MODULE.SUCCESSOR_BRANCH.encode() + b"\n"
            if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                return b"5555555555555555555555555555555555555555\n"
            self.fail(f"unexpected Git call {arguments}")

        with mock.patch.object(MODULE, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "HEAD moved"):
                MODULE._recheck_sealed_git_authority(record)
        self.assertEqual(len(calls), 2)

    def test_no_status_sealed_recheck_detects_executor_tag_movement(self) -> None:
        head = "4" * 40
        record = {
            "authority_mode": "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY",
            "head": head,
            "tree": "6" * 40,
            "prepare_identity": MODULE._preregistration_identity(),
            "executor_tag": {"tag_object": "7" * 40},
        }

        def fake_git(*arguments: str) -> bytes:
            self.assertNotEqual(arguments[0], "status")
            if arguments == ("branch", "--show-current"):
                return MODULE.SUCCESSOR_BRANCH.encode() + b"\n"
            if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                return head.encode() + b"\n"
            return b"ignored"

        with mock.patch.object(MODULE, "_git", side_effect=fake_git), mock.patch.object(
            MODULE, "_raw_direct_child_tree", return_value="6" * 40
        ), mock.patch.object(
            MODULE, "_nul_paths", return_value=MODULE.SUCCESSOR_PATHS
        ), mock.patch.object(
            MODULE,
            "_verify_prepare_tag_and_bytes",
            return_value=(MODULE._preregistration_identity(), b"manifest"),
        ), mock.patch.object(
            MODULE, "_verify_executor_tag", return_value={"tag_object": "8" * 40}
        ):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "tag moved"):
                MODULE._recheck_sealed_git_authority(record)

    def test_no_status_sealed_recheck_detects_late_tag_ref_movement(self) -> None:
        head = "4" * 40
        tree = "6" * 40
        executor_tag_object = "7" * 40
        payload = b"x"
        record = {
            "authority_mode": "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY",
            "head": head,
            "tree": tree,
            "prepare_identity": MODULE._preregistration_identity(),
            "executor_tag": {"tag_object": executor_tag_object},
            "successor_identity": {
                relative: {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for relative in MODULE.SUCCESSOR_PATHS
            },
        }

        for moved_tag in ("prepare", "executor"):
            with self.subTest(moved_tag=moved_tag):
                def fake_git(*arguments: str) -> bytes:
                    self.assertNotEqual(arguments[0], "status")
                    if arguments == ("branch", "--show-current"):
                        return MODULE.SUCCESSOR_BRANCH.encode() + b"\n"
                    if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                        return head.encode() + b"\n"
                    if arguments[0] in ("diff-tree", "diff"):
                        return b"ignored"
                    if arguments[:2] == ("cat-file", "blob"):
                        return payload
                    if arguments == ("rev-parse", f"refs/tags/{MODULE.PREPARE_TAG}"):
                        observed = (
                            "8" * 40
                            if moved_tag == "prepare"
                            else MODULE.PREPARE_TAG_OBJECT
                        )
                        return observed.encode() + b"\n"
                    if arguments == ("rev-parse", f"refs/tags/{MODULE.EXECUTOR_TAG}"):
                        observed = "8" * 40 if moved_tag == "executor" else executor_tag_object
                        return observed.encode() + b"\n"
                    self.fail(f"unexpected Git call {arguments}")

                with mock.patch.object(MODULE, "_git", side_effect=fake_git), mock.patch.object(
                    MODULE, "_raw_direct_child_tree", return_value=tree
                ), mock.patch.object(
                    MODULE, "_nul_paths", return_value=MODULE.SUCCESSOR_PATHS
                ), mock.patch.object(
                    MODULE,
                    "_verify_prepare_tag_and_bytes",
                    return_value=(MODULE._preregistration_identity(), b"manifest"),
                ), mock.patch.object(
                    MODULE,
                    "_verify_executor_tag",
                    return_value={"tag_object": executor_tag_object},
                ):
                    with self.assertRaisesRegex(MODULE.AcquisitionError, "tag moved"):
                        MODULE._recheck_sealed_git_authority(record)


class FilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name)
        self.attempt = self.parent / "attempt"
        self.result = self.parent / "result"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _authority() -> tuple[dict[str, object], list[object]]:
        return (
            {
                "authority_mode": "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY"
            },
            [],
        )

    def _acquire(self, payload: bytes) -> tuple[dict[str, object], list[str]]:
        calls: list[str] = []

        def transport(url: str) -> bytes:
            calls.append(url)
            self.assertTrue(self.attempt.is_dir())
            MODULE.validate_attempt(self.attempt)
            return payload

        value = MODULE._acquire_to_roots(
            self.result,
            self.attempt,
            transport=transport,
            authority=self._authority(),
        )
        return value, calls

    def test_attempt_publication_is_create_only_canonical_and_private(self) -> None:
        parent_path = MODULE._primitive("_open_anchored_directory_path")(
            self.parent, "test parent"
        )
        try:
            value = MODULE._make_attempt()
            payload = MODULE._canonical_json(value)
            MODULE._publish_attempt(parent_path[-1], self.attempt, payload)
        finally:
            MODULE._primitive("_close_directory_path")(parent_path)
        self.assertEqual(MODULE.validate_attempt(self.attempt), value)
        self.assertEqual(stat.S_IMODE(self.attempt.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.attempt / "ATTEMPT.json").stat().st_mode), 0o600)
        original = (self.attempt / "ATTEMPT.json").read_bytes()
        parent_path = MODULE._primitive("_open_anchored_directory_path")(
            self.parent, "test parent"
        )
        try:
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE._publish_attempt(parent_path[-1], self.attempt, payload)
        finally:
            MODULE._primitive("_close_directory_path")(parent_path)
        self.assertEqual((self.attempt / "ATTEMPT.json").read_bytes(), original)

    def test_attempt_name_swap_during_parent_fsync_sends_no_request(self) -> None:
        original_fsync = os.fsync
        displaced = self.parent / "displaced-attempt"
        injected = False
        calls = 0

        def replace_at_attempt_parent_fsync(descriptor: int) -> None:
            nonlocal injected
            metadata = os.fstat(descriptor)
            parent_metadata = self.parent.stat()
            if (
                not injected
                and stat.S_ISDIR(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino)
                == (parent_metadata.st_dev, parent_metadata.st_ino)
                and (self.attempt / "ATTEMPT.json").is_file()
            ):
                self.attempt.rename(displaced)
                self.attempt.mkdir(mode=0o700)
                replacement = self.attempt / "ATTEMPT.json"
                replacement.write_bytes((displaced / "ATTEMPT.json").read_bytes())
                replacement.chmod(0o600)
                injected = True
            original_fsync(descriptor)

        def forbidden(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(MODULE.os, "fsync", side_effect=replace_at_attempt_parent_fsync):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "dirent|identity"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=forbidden,
                    authority=self._authority(),
                )
        self.assertTrue(injected)
        self.assertEqual(calls, 0)
        self.assertNotEqual(self.attempt.stat().st_ino, displaced.stat().st_ino)
        MODULE.validate_attempt(self.attempt)
        MODULE.validate_attempt(displaced)
        self.assertFalse(self.result.exists())

    def test_semantic_rejection_still_publishes_exact_raw_and_custody(self) -> None:
        payload = b'{"result":"schema surprise"}'
        value, calls = self._acquire(payload)
        self.assertEqual(calls, [MODULE.QUERY_URL])
        self.assertEqual((self.result / "raw" / MODULE.RAW_NAME).read_bytes(), payload)
        self.assertEqual(value["diagnosis"]["v1_gate_outcome"], "REJECTED_BY_FROZEN_V1_GATE")
        self.assertEqual(set(value["claims"].values()), {False})
        self.assertEqual(MODULE.validate_result(self.result, self.attempt), value)

    def test_zero_byte_http_entity_is_retained_and_replayed(self) -> None:
        value, calls = self._acquire(b"")
        self.assertEqual(len(calls), 1)
        self.assertEqual((self.result / "raw" / MODULE.RAW_NAME).stat().st_size, 0)
        self.assertEqual(value["response"]["raw_bytes"], 0)
        self.assertEqual(MODULE.validate_result(self.result, self.attempt), value)

    def test_raw_file_and_directory_are_fsynced_before_diagnosis(self) -> None:
        events: list[tuple[str, int, int]] = []
        original_fsync = os.fsync
        original_diagnosis = MODULE._diagnosis

        def tracked_fsync(descriptor: int) -> None:
            metadata = os.fstat(descriptor)
            events.append(("dir" if stat.S_ISDIR(metadata.st_mode) else "file", metadata.st_dev, metadata.st_ino))
            original_fsync(descriptor)

        def checked_diagnosis(payload: bytes) -> dict[str, object]:
            staging = next(self.parent.glob(".result.staging-*"))
            raw_directory = (staging / "raw").stat()
            raw_file = (staging / "raw" / MODULE.RAW_NAME).stat()
            self.assertIn(("file", raw_file.st_dev, raw_file.st_ino), events)
            self.assertIn(("dir", raw_directory.st_dev, raw_directory.st_ino), events)
            return original_diagnosis(payload)

        with mock.patch.object(MODULE.os, "fsync", side_effect=tracked_fsync), mock.patch.object(
            MODULE, "_diagnosis", side_effect=checked_diagnosis
        ):
            self._acquire(b"{}")

    def test_raw_exact_copy_swap_before_diagnosis_cannot_publish(self) -> None:
        original_diagnosis = MODULE._diagnosis
        displaced = self.parent / "displaced-created-raw"
        injected = False
        calls = 0

        def replace_raw(payload: bytes) -> dict[str, object]:
            nonlocal injected
            staging = next(self.parent.glob(".result.staging-*"))
            raw_path = staging / "raw" / MODULE.RAW_NAME
            raw_path.rename(displaced)
            raw_path.write_bytes(payload)
            raw_path.chmod(0o600)
            injected = True
            return original_diagnosis(payload)

        def transport(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(MODULE, "_diagnosis", side_effect=replace_raw):
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=transport,
                    authority=self._authority(),
                )
        self.assertTrue(injected)
        self.assertEqual(calls, 1)
        self.assertEqual(displaced.read_bytes(), b"{}")
        self.assertFalse(self.result.exists())
        MODULE.validate_attempt(self.attempt)

    def test_custody_exact_copy_swap_before_replay_cannot_publish(self) -> None:
        original_validate = MODULE._validate_result_fds
        displaced = self.parent / "displaced-created-custody"
        injected = False

        def replace_custody(*args: object, **kwargs: object) -> object:
            nonlocal injected
            staging = next(self.parent.glob(".result.staging-*"))
            custody_path = staging / "CUSTODY.json"
            custody_path.rename(displaced)
            custody_path.write_bytes(displaced.read_bytes())
            custody_path.chmod(0o600)
            injected = True
            return original_validate(*args, **kwargs)

        with mock.patch.object(
            MODULE, "_validate_result_fds", side_effect=replace_custody
        ):
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=lambda _url: b"{}",
                    authority=self._authority(),
                )
        self.assertTrue(injected)
        self.assertGreater(displaced.stat().st_size, 0)
        self.assertFalse(self.result.exists())
        MODULE.validate_attempt(self.attempt)

    def test_transport_failure_keeps_attempt_removes_result_and_never_retries(self) -> None:
        calls = 0

        def failed(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            raise MODULE.AcquisitionError("terminal transport failure")

        with self.assertRaisesRegex(MODULE.AcquisitionError, "terminal transport"):
            MODULE._acquire_to_roots(
                self.result,
                self.attempt,
                transport=failed,
                authority=self._authority(),
            )
        self.assertEqual(calls, 1)
        MODULE.validate_attempt(self.attempt)
        self.assertFalse(self.result.exists())
        self.assertEqual(list(self.parent.glob(".result.staging-*")), [])

    def test_existing_attempt_refuses_before_transport_and_is_not_removed(self) -> None:
        first_calls = 0

        def failed(_url: str) -> bytes:
            nonlocal first_calls
            first_calls += 1
            raise MODULE.AcquisitionError("terminal")

        with self.assertRaises(MODULE.AcquisitionError):
            MODULE._acquire_to_roots(
                self.result,
                self.attempt,
                transport=failed,
                authority=self._authority(),
            )
        before = (self.attempt / "ATTEMPT.json").read_bytes()
        second_calls = 0

        def forbidden(_url: str) -> bytes:
            nonlocal second_calls
            second_calls += 1
            return b"{}"

        with self.assertRaisesRegex(MODULE.AcquisitionError, "attempt root already exists"):
            MODULE._acquire_to_roots(
                self.result,
                self.attempt,
                transport=forbidden,
                authority=self._authority(),
            )
        self.assertEqual(first_calls, 1)
        self.assertEqual(second_calls, 0)
        self.assertEqual((self.attempt / "ATTEMPT.json").read_bytes(), before)

    def test_existing_result_refuses_before_attempt_or_transport(self) -> None:
        self.result.mkdir(mode=0o700)
        calls = 0

        def forbidden(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with self.assertRaisesRegex(MODULE.AcquisitionError, "result root already exists"):
            MODULE._acquire_to_roots(
                self.result,
                self.attempt,
                transport=forbidden,
                authority=self._authority(),
            )
        self.assertEqual(calls, 0)
        self.assertFalse(self.attempt.exists())

    def test_result_appearing_after_attempt_refuses_before_transport(self) -> None:
        original = MODULE._publish_attempt
        calls = 0

        def publish_then_compete(parent_fd: int, attempt_root: Path, payload: bytes) -> None:
            original(parent_fd, attempt_root, payload)
            self.result.mkdir(mode=0o700)

        def forbidden(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(MODULE, "_publish_attempt", side_effect=publish_then_compete):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "result root already exists"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=forbidden,
                    authority=self._authority(),
                )
        self.assertEqual(calls, 0)
        MODULE.validate_attempt(self.attempt)
        self.assertTrue(self.result.is_dir())
        self.assertEqual(list(self.parent.glob(".result.staging-*")), [])

    def test_extra_injected_at_final_rename_can_never_return_success(self) -> None:
        base = MODULE._base_module()
        original = base._rename_noreplace
        calls = 0

        def inject_extra(
            source_directory_fd: int,
            source: str,
            destination_directory_fd: int,
            destination: str,
        ) -> None:
            if destination == self.result.name:
                (self.parent / source / "EXTRA").write_bytes(b"hostile")
            original(
                source_directory_fd,
                source,
                destination_directory_fd,
                destination,
            )

        def transport(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(base, "_rename_noreplace", side_effect=inject_extra):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "roster differs"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=transport,
                    authority=self._authority(),
                )
        self.assertEqual(calls, 1)
        self.assertTrue((self.result / "EXTRA").is_file())
        MODULE.validate_attempt(self.attempt)
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE.validate_result(self.result, self.attempt)

    def test_final_result_name_replacement_during_parent_fsync_cannot_succeed(self) -> None:
        original_fsync = os.fsync
        displaced = self.parent / "displaced-result"
        injected = False

        def replace_at_final_parent_fsync(descriptor: int) -> None:
            nonlocal injected
            metadata = os.fstat(descriptor)
            parent_metadata = self.parent.stat()
            if (
                not injected
                and stat.S_ISDIR(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino)
                == (parent_metadata.st_dev, parent_metadata.st_ino)
                and (self.result / "CUSTODY.json").is_file()
            ):
                self.result.rename(displaced)
                self.result.mkdir(mode=0o700)
                injected = True
            original_fsync(descriptor)

        with mock.patch.object(MODULE.os, "fsync", side_effect=replace_at_final_parent_fsync):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "dirent|identity"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=lambda _url: b"{}",
                    authority=self._authority(),
                )
        self.assertTrue(injected)
        self.assertEqual(list(self.result.iterdir()), [])
        self.assertTrue((displaced / "CUSTODY.json").is_file())
        MODULE.validate_attempt(self.attempt)

    def test_final_raw_name_replacement_during_parent_fsync_cannot_succeed(self) -> None:
        original_fsync = os.fsync
        displaced_raw = self.parent / "displaced-raw"
        injected = False

        def replace_at_final_parent_fsync(descriptor: int) -> None:
            nonlocal injected
            metadata = os.fstat(descriptor)
            parent_metadata = self.parent.stat()
            if (
                not injected
                and stat.S_ISDIR(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino)
                == (parent_metadata.st_dev, parent_metadata.st_ino)
                and (self.result / "CUSTODY.json").is_file()
            ):
                (self.result / "raw").rename(displaced_raw)
                (self.result / "raw").mkdir(mode=0o700)
                injected = True
            original_fsync(descriptor)

        with mock.patch.object(MODULE.os, "fsync", side_effect=replace_at_final_parent_fsync):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "dirent|identity"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=lambda _url: b"{}",
                    authority=self._authority(),
                )
        self.assertTrue(injected)
        self.assertTrue((displaced_raw / MODULE.RAW_NAME).is_file())
        self.assertEqual(list((self.result / "raw").iterdir()), [])
        MODULE.validate_attempt(self.attempt)

    def test_git_authority_is_rechecked_after_attempt_before_transport(self) -> None:
        record = self._authority()[0]
        calls = 0

        def forbidden(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(
            MODULE,
            "_verify_execution_authority",
            return_value=(record, []),
        ), mock.patch.object(
            MODULE,
            "_recheck_sealed_git_authority",
            side_effect=MODULE.AcquisitionError("HEAD moved"),
        ) as recheck:
            with self.assertRaisesRegex(MODULE.AcquisitionError, "HEAD moved"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=forbidden,
                )
        self.assertEqual(recheck.call_count, 1)
        self.assertEqual(calls, 0)
        MODULE.validate_attempt(self.attempt)
        self.assertFalse(self.result.exists())

    def test_stage_swap_during_pretransport_git_recheck_sends_no_request(self) -> None:
        record = self._authority()[0]
        moved = self.parent / "moved-stage"
        calls = 0

        def swap_stage(_record: object) -> None:
            staging = next(self.parent.glob(".result.staging-*"))
            staging.rename(moved)
            staging.mkdir(mode=0o700)

        def forbidden(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(
            MODULE, "_verify_execution_authority", return_value=(record, [])
        ), mock.patch.object(
            MODULE, "_recheck_sealed_git_authority", side_effect=swap_stage
        ):
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=forbidden,
                )
        self.assertEqual(calls, 0)
        MODULE.validate_attempt(self.attempt)
        self.assertTrue((moved / "raw").is_dir())

    def test_raw_swap_during_prefinal_git_recheck_cannot_publish(self) -> None:
        record = self._authority()[0]
        moved_raw = self.parent / "moved-prefinal-raw"
        git_rechecks = 0
        calls = 0

        def swap_on_second_recheck(_record: object) -> None:
            nonlocal git_rechecks
            git_rechecks += 1
            if git_rechecks == 2:
                staging = next(self.parent.glob(".result.staging-*"))
                (staging / "raw").rename(moved_raw)
                (staging / "raw").mkdir(mode=0o700)

        def transport(_url: str) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        with mock.patch.object(
            MODULE, "_verify_execution_authority", return_value=(record, [])
        ), mock.patch.object(
            MODULE,
            "_recheck_sealed_git_authority",
            side_effect=swap_on_second_recheck,
        ):
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=transport,
                )
        self.assertEqual(git_rechecks, 2)
        self.assertEqual(calls, 1)
        MODULE.validate_attempt(self.attempt)
        self.assertFalse(self.result.exists())
        self.assertTrue((moved_raw / MODULE.RAW_NAME).is_file())

    def test_handled_diagnosis_failure_keeps_attempt_and_cleans_staging(self) -> None:
        with mock.patch.object(
            MODULE, "_diagnosis", side_effect=MODULE.AcquisitionError("diagnosis bug")
        ):
            with self.assertRaisesRegex(MODULE.AcquisitionError, "diagnosis bug"):
                MODULE._acquire_to_roots(
                    self.result,
                    self.attempt,
                    transport=lambda _url: b"{}",
                    authority=self._authority(),
                )
        MODULE.validate_attempt(self.attempt)
        self.assertFalse(self.result.exists())
        self.assertEqual(list(self.parent.glob(".result.staging-*")), [])

    def test_process_control_baseexception_leaves_private_staging_and_attempt(self) -> None:
        def crash(_url: str) -> bytes:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            MODULE._acquire_to_roots(
                self.result,
                self.attempt,
                transport=crash,
                authority=self._authority(),
            )
        MODULE.validate_attempt(self.attempt)
        self.assertFalse(self.result.exists())
        leftovers = list(self.parent.glob(".result.staging-*"))
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(set(item.name for item in leftovers[0].iterdir()), {"raw"})

    def test_result_and_attempt_modes_rosters_and_single_links(self) -> None:
        self._acquire(b"{}")
        self.assertEqual(stat.S_IMODE(self.result.stat().st_mode), 0o700)
        self.assertEqual(set(item.name for item in self.result.iterdir()), {"CUSTODY.json", "raw"})
        self.assertEqual(set(item.name for item in (self.result / "raw").iterdir()), {MODULE.RAW_NAME})
        for path in (
            self.result / "CUSTODY.json",
            self.result / "raw" / MODULE.RAW_NAME,
            self.attempt / "ATTEMPT.json",
        ):
            metadata = path.stat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)

    def test_offline_replay_rejects_raw_mutation_and_extra_entries(self) -> None:
        self._acquire(b"{}")
        raw = self.result / "raw" / MODULE.RAW_NAME
        raw.write_bytes(b"changed")
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE.validate_result(self.result, self.attempt)
        raw.write_bytes(b"{}")
        extra = self.result / "extra"
        extra.write_text("x")
        with self.assertRaisesRegex(MODULE.AcquisitionError, "roster"):
            MODULE.validate_result(self.result, self.attempt)

    def test_offline_replay_rejects_attempt_mutation(self) -> None:
        self._acquire(b"{}")
        path = self.attempt / "ATTEMPT.json"
        value = json.loads(path.read_text())
        value["status"] = "changed"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(path, 0o600)
        with self.assertRaises(MODULE.AcquisitionError):
            MODULE.validate_result(self.result, self.attempt)

    def test_offline_replay_cannot_succeed_after_late_result_name_swap(self) -> None:
        self._acquire(b"{}")
        original = MODULE._recheck_attempt_hold
        displaced = self.parent / "offline-displaced-result"
        rechecks = 0

        def swap_after_second_attempt_recheck(hold: object) -> None:
            nonlocal rechecks
            original(hold)
            rechecks += 1
            if rechecks == 2:
                self.result.rename(displaced)
                self.result.mkdir(mode=0o700)

        with mock.patch.object(
            MODULE,
            "_recheck_attempt_hold",
            side_effect=swap_after_second_attempt_recheck,
        ):
            with self.assertRaises(MODULE.AcquisitionError):
                MODULE.validate_result(self.result, self.attempt)
        self.assertEqual(rechecks, 2)
        self.assertTrue((displaced / "CUSTODY.json").is_file())

    def test_result_custody_is_last_commit_marker_and_canonical(self) -> None:
        value, _calls = self._acquire(b"{}")
        custody = self.result / "CUSTODY.json"
        payload = custody.read_bytes()
        self.assertEqual(payload, MODULE._canonical_json(value))
        self.assertFalse((self.result / ".CUSTODY.json.partial").exists())
        self.assertEqual(
            value["content_sha256"], MODULE._domain_hash(MODULE.CUSTODY_DOMAIN, value)
        )

    def test_validate_commands_are_offline(self) -> None:
        self._acquire(b"{}")
        with mock.patch.object(MODULE, "_transport", side_effect=AssertionError("network")):
            MODULE.validate_attempt(self.attempt)
            MODULE.validate_result(self.result, self.attempt)

    def test_public_acquire_uses_only_frozen_production_roots(self) -> None:
        sentinel = {"ok": True}
        with mock.patch.object(MODULE, "_acquire_to_roots", return_value=sentinel) as acquire:
            self.assertIs(MODULE.acquire(), sentinel)
        acquire.assert_called_once_with(MODULE.DEFAULT_RESULT_ROOT, MODULE.DEFAULT_ATTEMPT_ROOT)


if __name__ == "__main__":
    unittest.main()
