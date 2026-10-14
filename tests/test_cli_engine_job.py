import contextlib
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from jxplanetx.cli import parser
from jxplanetx.engine.runner import run_scenario_manifest
from jxplanetx.engine_job_cli import (
    ENGINE_JOB_CLI_CONTRACT_FAILURE,
    ENGINE_JOB_CLI_IO_FAILURE,
    ENGINE_JOB_CLI_SUCCESS,
    EngineJobCLIIOError,
    _HeldOutput,
    _read_regular_file,
)
from tests.test_engine_runner import fixture_manifest


def invoke_cli(arguments: list[str]) -> tuple[int, str, str]:
    args = parser().parse_args(arguments)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = args.func(args)
    return result, stdout.getvalue(), stderr.getvalue()


class EngineJobCLITests(unittest.TestCase):
    def test_official_display_name_preserves_command_identifier(self):
        cli_parser = parser()
        self.assertEqual(cli_parser.prog, "jxplanetx")
        self.assertIn("JX Scientific Engine", cli_parser.format_help())

    def test_base_cli_import_and_help_do_not_require_engine_extra(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)
        probe = (
            "import jxplanetx.cli as cli; "
            "parser = cli.parser(); "
            "print(parser.prog); "
            "print('engine' in parser.format_help()); "
            "args = parser.parse_args(['engine', 'list']); "
            "print(args.func(args))"
        )
        completed = subprocess.run(
            (sys.executable, "-S", "-c", probe),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines(), ["jxplanetx", "True", "2"])
        self.assertIn("install jxplanetx[engine]", completed.stderr)

    def test_list_run_and_verify_one_create_only_job(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            job_path = root / "result.jxjob"
            manifest_path.write_bytes(manifest)

            code, stdout, stderr = invoke_cli(["engine", "list"])
            self.assertEqual(code, ENGINE_JOB_CLI_SUCCESS)
            self.assertEqual(stderr, "")
            listing = json.loads(stdout)
            self.assertTrue(any(row["engine_job_v1"] for row in listing["routes"]))

            code, stdout, stderr = invoke_cli(
                [
                    "engine",
                    "run",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-sha256",
                    manifest_sha256,
                    "--output",
                    str(job_path),
                ]
            )
            self.assertEqual(code, ENGINE_JOB_CLI_SUCCESS)
            self.assertEqual(stderr, "")
            run_summary = json.loads(stdout)
            self.assertTrue(job_path.is_file())
            self.assertEqual(run_summary["manifest_sha256"], manifest_sha256)
            for key in (
                "accuracy_claimed",
                "cross_platform_replay_claimed",
                "execution_authenticated",
                "general_superiority_claimed",
                "improvement_claimed",
                "native_result_digest_recomputed",
                "numerical_execution_replayed",
                "physics_claimed",
                "production_authorized",
                "qualification_authorized",
                "registry_authorized",
                "source_runtime_reproduction_claimed",
            ):
                self.assertIs(run_summary[key], False)
            self.assertIs(run_summary["content_integrity_only"], True)
            self.assertIs(run_summary["deterministic_encoding_only"], True)
            self.assertEqual(run_summary["evidence_class"], "MODEL_OUTPUT")

            original_job = job_path.read_bytes()
            code, stdout, stderr = invoke_cli(
                [
                    "engine",
                    "verify",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-sha256",
                    manifest_sha256,
                    "--job",
                    str(job_path),
                    "--job-sha256",
                    run_summary["job_sha256"],
                ]
            )
            self.assertEqual(code, ENGINE_JOB_CLI_SUCCESS)
            self.assertEqual(stderr, "")
            self.assertEqual(job_path.read_bytes(), original_job)
            verification = json.loads(stdout)
            self.assertIs(verification["bundle_exact_bytes_verified"], True)
            self.assertIs(
                verification["manifest_artifact_restart_crossbindings_verified"],
                True,
            )
            self.assertNotIn("content_verified", verification)
            for key in (
                "accuracy_claimed",
                "cross_platform_replay_claimed",
                "execution_authenticated",
                "general_superiority_claimed",
                "improvement_claimed",
                "native_result_digest_recomputed",
                "numerical_execution_replayed",
                "physics_claimed",
                "production_authorized",
                "qualification_authorized",
                "registry_authorized",
                "source_runtime_reproduction_claimed",
            ):
                self.assertIs(verification[key], False)
            self.assertIs(verification["content_integrity_only"], True)
            self.assertIs(verification["deterministic_encoding_only"], True)
            self.assertEqual(verification["evidence_class"], "MODEL_OUTPUT")

    def test_existing_or_racing_output_is_never_replaced(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            job_path = root / "result.jxjob"
            manifest_path.write_bytes(manifest)
            job_path.write_bytes(b"retained")
            with mock.patch(
                "jxplanetx.engine_job_cli.run_scenario_manifest"
            ) as numerical_runner:
                code, stdout, _ = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(manifest_path),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(job_path),
                    ]
                )
            self.assertEqual(code, ENGINE_JOB_CLI_IO_FAILURE)
            self.assertEqual(stdout, "")
            numerical_runner.assert_not_called()
            self.assertEqual(job_path.read_bytes(), b"retained")

            job_path.unlink()

            def race_into_place(raw: bytes, digest: str):
                bundle = run_scenario_manifest(raw, digest)
                job_path.write_bytes(b"racer")
                return bundle

            with mock.patch(
                "jxplanetx.engine_job_cli.run_scenario_manifest",
                side_effect=race_into_place,
            ):
                code, stdout, _ = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(manifest_path),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(job_path),
                    ]
                )
            self.assertEqual(code, ENGINE_JOB_CLI_IO_FAILURE)
            self.assertEqual(stdout, "")
            self.assertEqual(job_path.read_bytes(), b"racer")
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

    def test_symlink_components_and_special_inputs_fail_before_numerics(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            manifest_path = real / "scenario.json"
            manifest_path.write_bytes(manifest)
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            output = root / "result.jxjob"
            with mock.patch(
                "jxplanetx.engine_job_cli.run_scenario_manifest"
            ) as numerical_runner:
                code, stdout, _ = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(linked / "scenario.json"),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, ENGINE_JOB_CLI_IO_FAILURE)
            self.assertEqual(stdout, "")
            numerical_runner.assert_not_called()
            self.assertFalse(output.exists())

            fifo = root / "input.fifo"
            os.mkfifo(fifo)
            with mock.patch(
                "jxplanetx.engine_job_cli.os.read",
                side_effect=AssertionError("FIFO payload must never be read"),
            ):
                with self.assertRaises(EngineJobCLIIOError):
                    _read_regular_file(
                        str(fifo),
                        label="fixture",
                        maximum_bytes=1024,
                    )

    def test_input_mutation_and_publication_failure_leave_no_output(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            job_path = root / "result.jxjob"
            manifest_path.write_bytes(manifest)
            real_read = os.read
            mutated = False

            def mutate_after_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                chunk = real_read(descriptor, size)
                if chunk and not mutated:
                    mutated = True
                    manifest_path.write_bytes(b"changed")
                return chunk

            with mock.patch(
                "jxplanetx.engine_job_cli.os.read",
                side_effect=mutate_after_read,
            ):
                with self.assertRaisesRegex(EngineJobCLIIOError, "changed"):
                    _read_regular_file(
                        str(manifest_path),
                        label="manifest",
                        maximum_bytes=4 * 1024 * 1024,
                    )

            manifest_path.write_bytes(manifest)
            with mock.patch(
                "jxplanetx.engine_job_cli.os.write",
                side_effect=OSError(errno.ENOSPC, "fixture full"),
            ):
                code, stdout, _ = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(manifest_path),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(job_path),
                    ]
                )
            self.assertEqual(code, ENGINE_JOB_CLI_IO_FAILURE)
            self.assertEqual(stdout, "")
            self.assertFalse(job_path.exists())
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

    def test_publication_fsync_boundaries_preserve_atomic_contract(self):
        payload = b"complete-engine-job-fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            file_fsync_path = root / "file-fsync-failure.jxjob"
            held = _HeldOutput(str(file_fsync_path))
            try:
                file_error = OSError(errno.EIO, "fixture file fsync failure")
                with mock.patch(
                    "jxplanetx.engine_job_cli.os.fsync",
                    side_effect=(file_error, None),
                ) as synchronized:
                    with self.assertRaises(EngineJobCLIIOError):
                        held.publish(payload)
                self.assertEqual(synchronized.call_count, 2)
            finally:
                held.close()
            self.assertFalse(file_fsync_path.exists())
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

            first_directory_fsync_path = root / "directory-fsync-failure.jxjob"
            held = _HeldOutput(str(first_directory_fsync_path))
            try:
                directory_error = OSError(
                    errno.EIO,
                    "fixture publication directory fsync failure",
                )
                with mock.patch(
                    "jxplanetx.engine_job_cli.os.fsync",
                    side_effect=(None, directory_error, None),
                ) as synchronized:
                    with self.assertRaises(EngineJobCLIIOError):
                        held.publish(payload)
                self.assertEqual(synchronized.call_count, 3)
            finally:
                held.close()
            self.assertFalse(first_directory_fsync_path.exists())
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

            staging_unlink_path = root / "staging-unlink-failure.jxjob"
            held = _HeldOutput(str(staging_unlink_path))
            try:
                real_unlink = os.unlink
                staging_failure_injected = False

                def fail_first_staging_unlink(name, *args, **kwargs):
                    nonlocal staging_failure_injected
                    if (
                        not staging_failure_injected
                        and str(name).startswith(".jxengine-")
                    ):
                        staging_failure_injected = True
                        raise OSError(errno.EIO, "fixture staging unlink failure")
                    return real_unlink(name, *args, **kwargs)

                with mock.patch(
                    "jxplanetx.engine_job_cli.os.unlink",
                    side_effect=fail_first_staging_unlink,
                ):
                    with self.assertRaises(EngineJobCLIIOError):
                        held.publish(payload)
            finally:
                held.close()
            self.assertFalse(staging_unlink_path.exists())
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

    def test_initial_staging_fstat_failure_cleans_temporary_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path = root / "result.jxjob"
            held = _HeldOutput(str(job_path))
            try:
                with mock.patch(
                    "jxplanetx.engine_job_cli.os.fstat",
                    side_effect=OSError(errno.EIO, "fixture fstat failure"),
                ):
                    with self.assertRaises(EngineJobCLIIOError):
                        held.publish(b"complete-engine-job-fixture")
            finally:
                held.close()
            self.assertFalse(job_path.exists())
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

    def test_unsafe_output_parent_is_rejected_before_execution(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            manifest_path.write_bytes(manifest)
            unsafe = root / "unsafe"
            unsafe.mkdir(mode=0o700)
            unsafe.chmod(0o770)
            output = unsafe / "result.jxjob"

            with mock.patch(
                "jxplanetx.engine_job_cli.run_scenario_manifest"
            ) as numerical_runner:
                code, stdout, _ = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(manifest_path),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, ENGINE_JOB_CLI_IO_FAILURE)
            self.assertEqual(stdout, "")
            numerical_runner.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(list(unsafe.glob(".jxengine-*.tmp")), [])

    def test_reporting_failure_keeps_complete_hash_matching_final(self):
        manifest, manifest_sha256 = fixture_manifest()
        bundle = run_scenario_manifest(manifest, manifest_sha256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            job_path = root / "result.jxjob"
            manifest_path.write_bytes(manifest)

            with mock.patch(
                "jxplanetx.engine_job_cli.run_scenario_manifest",
                return_value=bundle,
            ), mock.patch(
                "builtins.print",
                side_effect=BrokenPipeError(errno.EPIPE, "fixture closed stdout"),
            ) as reporter:
                code, stdout, stderr = invoke_cli(
                    [
                        "engine",
                        "run",
                        "--manifest",
                        str(manifest_path),
                        "--manifest-sha256",
                        manifest_sha256,
                        "--output",
                        str(job_path),
                    ]
                )

            self.assertEqual(code, ENGINE_JOB_CLI_SUCCESS)
            self.assertEqual(stdout, "")
            self.assertEqual(stderr, "")
            reporter.assert_called_once()
            self.assertEqual(job_path.read_bytes(), bundle.job_bytes)
            self.assertEqual(
                hashlib.sha256(job_path.read_bytes()).hexdigest(),
                bundle.job_sha256,
            )
            self.assertEqual(list(root.glob(".jxengine-*.tmp")), [])

    def test_maximum_portable_leaf_and_wrong_hash_exit_contract(self):
        manifest, manifest_sha256 = fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "scenario.json"
            manifest_path.write_bytes(manifest)
            job_path = root / ("a" * 249 + ".jxjob")
            code, stdout, _ = invoke_cli(
                [
                    "engine",
                    "run",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-sha256",
                    manifest_sha256,
                    "--output",
                    str(job_path),
                ]
            )
            self.assertEqual(code, ENGINE_JOB_CLI_SUCCESS)
            summary = json.loads(stdout)
            self.assertTrue(job_path.is_file())

            code, stdout, _ = invoke_cli(
                [
                    "engine",
                    "verify",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-sha256",
                    manifest_sha256,
                    "--job",
                    str(job_path),
                    "--job-sha256",
                    "0" * 64,
                ]
            )
            self.assertEqual(code, ENGINE_JOB_CLI_CONTRACT_FAILURE)
            self.assertEqual(stdout, "")
            self.assertNotEqual(summary["job_sha256"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
