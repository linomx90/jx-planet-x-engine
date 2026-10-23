from __future__ import annotations

import base64
import copy
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from benchmarks import jx_wh_v9_scientific_qualification as qualification


_BASE_REPORT: dict[str, object] | None = None


def _base_report() -> dict[str, object]:
    global _BASE_REPORT
    if _BASE_REPORT is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        _BASE_REPORT = qualification.run_qualification(
            qualification.SMOKE, include_ias15=False
        )
    return _BASE_REPORT


def _reseal(report: dict[str, object]) -> dict[str, object]:
    report["content_integrity"] = {
        "algorithm": "SHA256_DOMAIN_SEPARATED_CANONICAL_ASCII_JSON_V1",
        "domain": qualification.REPORT_DOMAIN,
        "sha256": qualification._domain_sha256(
            qualification.REPORT_DOMAIN,
            {key: value for key, value in report.items() if key != "content_integrity"},
        ),
    }
    return report


def _reseal_source_closure(report: dict[str, object]) -> None:
    sources = report["sources"]
    sources["closure_sha256"] = qualification._domain_sha256(
        qualification.SOURCE_CLOSURE_DOMAIN,
        {"execution_head": sources["execution_head"], "files": sources["files"]},
    )


def _refresh_derived_report(report: dict[str, object]) -> dict[str, object]:
    gates = qualification._derive_gates(report)
    report["gates"] = gates
    passed = all(gate["passed"] is True for gate in gates)
    report["decision"] = {
        "status": (
            qualification.INTERNAL_PASS if passed else qualification.INTERNAL_FAIL
        ),
        "all_numeric_and_custody_gates_conjunctive": True,
        "all_gates_passed": passed,
        "scientific_approval_granted": False,
        "external_review_completed": False,
        "next_gate": qualification._protocol()["next_gate"],
    }
    return _reseal(report)


def _engineering_log(
    level: int,
    sources: dict[str, object],
    *,
    exit_code: int = 0,
    tests_run: int = 39,
    elapsed: str = "1.234",
) -> dict[str, object]:
    stdout = b""
    stderr = (
        "----------------------------------------------------------------------\n"
        f"Ran {tests_run} tests in {elapsed}s\n\nOK\n"
    ).encode("ascii")
    core = {
        "schema": "jx.wh-v9.qualification.engineering-command-result.v1",
        "command": qualification._logical_engineering_command(level),
        "optimization_level": level,
        "working_directory": "REPOSITORY_ROOT",
        "environment": qualification._engineering_environment_record(),
        "execution_head": sources["execution_head"],
        "source_closure_sha256": sources["closure_sha256"],
        "exit_code": exit_code,
        "stdout": qualification._bytes_record(stdout),
        "stderr": qualification._bytes_record(stderr),
        "semantic_evidence": qualification._unittest_semantic_evidence(
            stdout, stderr
        ),
        "python_executable_sha256": _base_report()["runtime"]["python"][
            "executable_sha256"
        ],
    }
    return core | {
        "content_sha256": qualification._domain_sha256(
            "jx.wh-v9.qualification.engineering-command-result.v1", core
        )
    }


def _wheel_bytes(
    sources: dict[str, object],
    *,
    extra_member: bytes | None = None,
    extension_bytes: bytes | None = None,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        relative_paths = (
            "src/jxplanetx/fast_wisdom_holman_v9.py",
            *(f"src/jxplanetx/{name}" for name in qualification.V9_NATIVE_SOURCE_SHA256),
        )
        for relative in relative_paths:
            record = qualification._source_record(sources, relative)
            raw = (qualification.ROOT / relative).read_bytes()
            if len(raw) != record["size_bytes"]:
                raise RuntimeError("source changed while making test wheel")
            archive.writestr(relative.removeprefix("src/"), raw)
        archive.writestr(
            "jxplanetx/_wisdom_holman_cpu_v9.cpython-314-x86_64-linux-gnu.so",
            (
                Path(qualification.v9._native.__file__).read_bytes()
                if extension_bytes is None
                else extension_bytes
            ),
        )
        prefix = "jxplanetx-0.6.0rc17.dist-info/"
        archive.writestr(
            prefix + "METADATA",
            b"Metadata-Version: 2.4\nName: jxplanetx\nVersion: 0.6.0rc17\n\n",
        )
        archive.writestr(prefix + "WHEEL", b"Wheel-Version: 1.0\n")
        archive.writestr(prefix + "RECORD", b"qualified-test-record\n")
        if extra_member is not None:
            archive.writestr("jxplanetx/FORMAL_BUILD_MARKER", extra_member)
    return stream.getvalue()


def _mock_release_execution(
    label: str,
    output_directory: Path,
    sources: dict[str, object],
    wheel: bytes,
    sdist: bytes,
) -> tuple[dict[str, object], dict[str, Path]]:
    output_directory.mkdir()
    wheel_name = "jxplanetx-0.6.0rc17-cp314-cp314-linux_x86_64.whl"
    sdist_name = "jxplanetx-0.6.0rc17.tar.gz"
    wheel_path, sdist_path = output_directory / wheel_name, output_directory / sdist_name
    wheel_path.write_bytes(wheel)
    sdist_path.write_bytes(sdist)
    manifest = {
        "schema": "jxplanetx.reproducible-release-artifacts.v1",
        "version": "0.6.0rc17",
        "scientific_claim_state": "SCREENING_ONLY",
        "source_date_epoch": qualification.RELEASE_SOURCE_DATE_EPOCH,
        "build_backend": "setuptools==78.1.1",
        "independent_clean_builds": 2,
        "byte_identical": {"wheel": True, "sdist": True},
        "artifacts": {
            "wheel": {
                "filename": wheel_name,
                "size_bytes": len(wheel),
                "sha256": qualification._sha256(wheel),
            },
            "sdist": {
                "filename": sdist_name,
                "size_bytes": len(sdist),
                "sha256": qualification._sha256(sdist),
            },
        },
    }
    record = {
        "schema": "jx.wh-v9.qualification.release-build-execution.v1",
        "logical_build": label,
        "command": qualification._logical_release_build_command(label),
        "working_directory": "REPOSITORY_ROOT",
        "environment": qualification._release_environment_record(),
        "execution_head": sources["execution_head"],
        "source_closure_sha256": sources["closure_sha256"],
        "python_executable_sha256": _base_report()["runtime"]["python"][
            "executable_sha256"
        ],
        "exit_code": 0,
        "stdout_manifest_matches": True,
        "compiler": qualification._compiler_custody(),
        "python_build_configuration": (
            qualification._build_configuration_custody()
        ),
        "stderr": qualification._bytes_record(b""),
        "manifest": manifest,
    }
    return record, {"wheel": wheel_path, "sdist": sdist_path}


def _sdist_bytes(sources: dict[str, object]) -> bytes:
    root = "jxplanetx-0.6.0rc17/"
    payloads = {
        root + "PKG-INFO": (
            b"Metadata-Version: 2.4\nName: jxplanetx\nVersion: 0.6.0rc17\n\n"
        )
    }
    relative_paths = (
        "pyproject.toml",
        "src/jxplanetx/fast_wisdom_holman_v9.py",
        *(f"src/jxplanetx/{name}" for name in qualification.V9_NATIVE_SOURCE_SHA256),
    )
    for relative in relative_paths:
        record = qualification._source_record(sources, relative)
        raw = (qualification.ROOT / relative).read_bytes()
        if len(raw) != record["size_bytes"]:
            raise RuntimeError("source changed while making test sdist")
        payloads[root + relative] = raw
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(raw))
    return stream.getvalue()


def _artifact_paths(
    directory: Path, sources: dict[str, object], *, write_logs: bool = True
) -> qualification.EngineeringEvidencePaths:
    build_a, build_b = directory / "build-a", directory / "build-b"
    build_a.mkdir()
    build_b.mkdir()
    wheel_name = "jxplanetx-0.6.0rc17-cp314-cp314-linux_x86_64.whl"
    sdist_name = "jxplanetx-0.6.0rc17.tar.gz"
    wheel, sdist = _wheel_bytes(sources), _sdist_bytes(sources)
    for build in (build_a, build_b):
        (build / wheel_name).write_bytes(wheel)
        (build / sdist_name).write_bytes(sdist)
    normal, optimized = directory / "normal.json", directory / "optimized.json"
    if write_logs:
        normal.write_bytes(
            qualification._canonical_bytes(_engineering_log(0, sources), newline=True)
        )
        optimized.write_bytes(
            qualification._canonical_bytes(_engineering_log(2, sources), newline=True)
        )
    return qualification.EngineeringEvidencePaths(
        normal,
        optimized,
        build_a / wheel_name,
        build_b / wheel_name,
        build_a / sdist_name,
        build_b / sdist_name,
    )


def _valid_rebound_runtime() -> dict[str, object]:
    wrappers = [
        "rebound/__init__.py",
        "rebound/integrator.py",
        "rebound/particle.py",
        "rebound/particles.py",
        "rebound/simulation.py",
    ]
    files = [
        {"path": path, "size_bytes": 1, "sha256": "1" * 64}
        for path in wrappers
    ]
    sources = {
        "schema": qualification.fixture._PROVENANCE_SUPPORT.REBOUND_SOURCE_MANIFEST_SCHEMA,
        "selection": "ALL_DISTRIBUTION_RECORDED_REBOUND_PACKAGE_PYTHON_SOURCES",
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "required_execution_wrappers": wrappers,
        "files": files,
        "file_count": len(files),
        "tree_hash_algorithm": "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1",
        "tree_hash_domain": qualification.fixture._PROVENANCE_SUPPORT.REBOUND_SOURCE_TREE_HASH_DOMAIN,
        "tree_sha256": qualification.fixture._PROVENANCE_SUPPORT._rebound_source_tree_sha256(files),
    }

    def identity(logical_path: str) -> dict[str, object]:
        return {"logical_path": logical_path, "size_bytes": 1, "sha256": "2" * 64}

    return {
        "schema": "jx.wh-v9.qualification.rebound-ias15-runtime.v1",
        "classification": "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR",
        "authority_authorized": False,
        "distribution_name": "rebound",
        "distribution_version": "5.1.1",
        "rebound_version": "5.1.1",
        "rebound_githash": "test-githash",
        "rebound_build": "test-build",
        "module_source": identity("rebound/__init__.py"),
        "c_library": identity("rebound/NATIVE_LIBREBOUND_BINARY"),
        "python_sources": sources,
        "distribution_metadata": identity("rebound-5.1.1.dist-info/METADATA"),
        "distribution_license_file": identity("rebound-5.1.1.dist-info/licenses/LICENSE"),
        "installed_license_expression": "GPL-3.0-only",
        "project_description_mentions_v3_or_later": True,
        "license_text_inconsistency_observed": True,
        "official_repository": "https://github.com/hannorein/rebound",
        "official_license_page": "https://github.com/hannorein/rebound/blob/5.1.1/LICENSE",
        "legal_or_redistribution_conclusion_authorized": False,
        "rebound_code_vendored": False,
        "rebound_code_copied": False,
        "jx_core_linked_to_rebound": False,
        "omp_num_threads": "1",
    }


class QualificationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _base_report()
        cls.encoded = qualification.report_bytes(cls.report)

    def test_smoke_is_closed_and_never_eligible(self) -> None:
        qualification.validate_report(self.report)
        self.assertEqual(self.report["decision"]["status"], qualification.INTERNAL_FAIL)
        self.assertIs(self.report["decision"]["scientific_approval_granted"], False)
        self.assertEqual(self.report["analytic_two_body"]["case_count"], 1)
        self.assertEqual(
            [lane["step_divisor"] for lane in self.report["weak_three_body"]["short"]["lanes"]],
            [64, 128, 256],
        )

    def test_retained_bytes_are_intrinsic_and_live_check_is_separate(self) -> None:
        with (
            mock.patch.object(qualification, "_source_custody", side_effect=AssertionError),
            mock.patch.object(qualification, "_runtime_custody", side_effect=AssertionError),
        ):
            self.assertEqual(qualification.load_report_bytes(self.encoded), self.report)
            self.assertEqual(qualification.load_report_bytes(self.encoded[:-1]), self.report)
        with (
            mock.patch.object(qualification, "_source_custody", return_value=copy.deepcopy(self.report["sources"])),
            mock.patch.object(qualification, "_runtime_custody", return_value=copy.deepcopy(self.report["runtime"])),
        ):
            qualification.validate_live_reproduction(self.report)
        changed = copy.deepcopy(self.report["sources"])
        changed["execution_head"] = "f" * 40
        with mock.patch.object(qualification, "_source_custody", return_value=changed), self.assertRaisesRegex(
            qualification.QualificationError, "live source"
        ):
            qualification.validate_live_reproduction(self.report)

    def test_roundtrip_scalar_is_recomputed(self) -> None:
        forged = copy.deepcopy(self.report)
        forged["weak_three_body"]["p128_roundtrip"]["normalized_state_error_hex"] = 0.0.hex()
        with self.assertRaisesRegex(qualification.QualificationError, "roundtrip"):
            qualification.validate_report(_reseal(forged))

    def test_wrong_divisor_receipt_splice_fails_after_metric_recompute(self) -> None:
        forged = copy.deepcopy(self.report)
        lanes = forged["weak_three_body"]["short"]["lanes"]
        lanes[1]["candidate"] = copy.deepcopy(lanes[0]["candidate"])
        positions, velocities, _, _ = qualification._validate_v9_receipt(lanes[1]["candidate"])
        _, dop_p, dop_v = qualification._oracle_arrays(lanes[1]["dop853"], "DOP853")
        metrics = qualification._weak_lane_metrics(positions, velocities, dop_p, dop_v)
        lanes[1]["metrics_hex"] = {
            "position_vector_l2_max": metrics["position_vector_l2_max"].hex(),
            "velocity_vector_l2_max": metrics["velocity_vector_l2_max"].hex(),
            "phase_error_radians_max": metrics["phase_error_radians_max"].hex(),
            "invariants": {key: float(value).hex() for key, value in metrics["invariants"].items()},
        }
        with self.assertRaises(
            (qualification.QualificationError, qualification.v9.FastWisdomHolmanV9Error)
        ):
            qualification.validate_report(_reseal(forged))

    def test_array_and_dop853_accounting_forgeries_fail(self) -> None:
        forged = copy.deepcopy(self.report)
        evidence = forged["analytic_two_body"]["cases"][0]["candidate"]["arrays"]["positions"]
        raw = bytearray(base64.b64decode(evidence["data_base64"], validate=True))
        raw[-1] ^= 1
        evidence["data_base64"] = base64.b64encode(raw).decode("ascii")
        metadata = {key: evidence[key] for key in ("dtype", "shape", "byte_order")}
        evidence["sha256"] = qualification._sha256(
            qualification.ARRAY_DOMAIN.encode("ascii")
            + b"\0"
            + qualification._canonical_bytes(metadata)
            + b"\0"
            + bytes(raw)
        )
        with self.assertRaises(
            (qualification.QualificationError, qualification.v9.FastWisdomHolmanV9Error)
        ):
            qualification.validate_report(_reseal(forged))
        for key, replacement in (
            ("function_evaluations", 999999),
            ("relative_tolerance_hex", float(1.0e-9).hex()),
        ):
            with self.subTest(key=key):
                forged = copy.deepcopy(self.report)
                forged["weak_three_body"]["short"]["lanes"][0]["dop853"]["settings"][key] = replacement
                with self.assertRaisesRegex(qualification.QualificationError, "DOP853"):
                    qualification.validate_report(_reseal(forged))

    def test_live_reproduction_rejects_structurally_resealed_oracle_and_nfev(self) -> None:
        for mutation in ("nfev", "oracle"):
            forged = copy.deepcopy(self.report)
            lane = forged["weak_three_body"]["short"]["lanes"][0]
            dop = lane["dop853"]
            if mutation == "nfev":
                dop["settings"]["function_evaluations"] += 2
            else:
                positions = qualification._decode_array(dop["positions"]).copy()
                positions[-1, -1, -1] = np.nextafter(
                    positions[-1, -1, -1], np.inf
                )
                dop["positions"] = qualification._array_evidence(positions)
            dop_core = {
                key: dop[key] for key in dop if key != "content_sha256"
            }
            dop["content_sha256"] = qualification._domain_sha256(
                qualification.DOP853_RECEIPT_DOMAIN, dop_core
            )
            candidate_p, candidate_v, _, _ = qualification._validate_v9_receipt(
                lane["candidate"]
            )
            _, dop_p, dop_v = qualification._oracle_arrays(dop, "DOP853")
            metrics = qualification._weak_lane_metrics(
                candidate_p, candidate_v, dop_p, dop_v
            )
            lane["metrics_hex"] = {
                "position_vector_l2_max": metrics[
                    "position_vector_l2_max"
                ].hex(),
                "velocity_vector_l2_max": metrics[
                    "velocity_vector_l2_max"
                ].hex(),
                "phase_error_radians_max": metrics[
                    "phase_error_radians_max"
                ].hex(),
                "invariants": {
                    key: float(value).hex()
                    for key, value in metrics["invariants"].items()
                },
            }
            forged = _refresh_derived_report(forged)
            qualification.validate_report(forged)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                qualification.QualificationError, "live scientific reproduction"
            ):
                qualification.validate_live_reproduction(forged)

    def test_invariant_container_roster_is_exact(self) -> None:
        for operation in ("reverse", "extra", "missing"):
            forged = copy.deepcopy(self.report)
            lanes = forged["weak_three_body"]["invariant_horizon"]["lanes"]
            if operation == "reverse":
                lanes.reverse()
            elif operation == "extra":
                lanes.append(copy.deepcopy(lanes[-1]))
            else:
                lanes.pop()
            with self.subTest(operation=operation), self.assertRaisesRegex(
                qualification.QualificationError, "invariant"
            ):
                qualification.validate_report(_reseal(forged))

    def test_native_source_closure_forgery_fails_after_inner_reseal(self) -> None:
        forged = copy.deepcopy(self.report)
        record = qualification._source_record(
            forged["sources"], "src/jxplanetx/_wisdom_holman_loop_v8.c"
        )
        record["sha256"] = "0" * 64
        _reseal_source_closure(forged)
        with self.assertRaisesRegex(qualification.QualificationError, "file identity"):
            qualification.validate_report(_reseal(forged))

    def test_claim_protocol_and_metric_forgeries_fail(self) -> None:
        cases = (
            (lambda r: r["claim_controls"].__setitem__("scientific_approval_granted", True), "claim"),
            (lambda r: r["protocol"].__setitem__("sha256", "0" * 64), "protocol"),
            (
                lambda r: r["analytic_two_body"]["cases"][0]["metrics_hex"].__setitem__(
                    "normalized_position_error_max", 0.0.hex()
                ),
                "metrics",
            ),
        )
        for mutate, message in cases:
            forged = copy.deepcopy(self.report)
            mutate(forged)
            with self.assertRaisesRegex(qualification.QualificationError, message):
                qualification.validate_report(_reseal(forged))


class StrictInputAndPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.encoded = qualification.report_bytes(_base_report())

    def test_bad_json_forms_fail(self) -> None:
        cases = (
            (b'{"x":1,"x":2}', "duplicate"),
            (b'{"x":NaN}', "nonfinite"),
            (b'{ "x": 1 }', "canonical"),
            (b"x" * (qualification.MAX_REPORT_BYTES + 1), "cap"),
        )
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                qualification.QualificationError, message
            ):
                qualification.load_report_bytes(data)

    def test_path_json_is_size_bounded_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "oversized.json"
            with oversized.open("wb") as stream:
                stream.truncate(qualification.MAX_ENGINEERING_JSON_BYTES + 1)
            with self.assertRaisesRegex(
                qualification.QualificationError, "bounded|cap"
            ):
                qualification._load_json_file(oversized)

    def test_create_only_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            qualification._write_new_file(target, self.encoded)
            with self.assertRaisesRegex(qualification.QualificationError, "must not exist"):
                qualification._write_new_file(target, b"replacement")
            self.assertEqual(target.read_bytes(), self.encoded)

    def test_dangling_leaf_and_symlink_ancestor_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dangling = root / "dangling"
            dangling.symlink_to("missing")
            with self.assertRaises(qualification.QualificationError):
                qualification._write_new_file(dangling, b"data")
            self.assertTrue(dangling.is_symlink())
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(qualification.QualificationError):
                qualification._write_new_file(alias / "result", b"data")
            self.assertFalse((real / "result").exists())

    def test_leaf_replacement_race_is_detected_and_not_unlinked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "race.json"
            real_stat, leaf_stats = os.stat, 0

            def racing_stat(path: object, *args: object, **kwargs: object):
                nonlocal leaf_stats
                result = real_stat(path, *args, **kwargs)
                if path == target.name and kwargs.get("dir_fd") is not None and kwargs.get("follow_symlinks") is False:
                    leaf_stats += 1
                    if leaf_stats == 1:
                        directory_fd = kwargs["dir_fd"]
                        os.unlink(target.name, dir_fd=directory_fd)
                        os.symlink("attacker-owned", target.name, dir_fd=directory_fd)
                        return real_stat(path, *args, **kwargs)
                return result

            with mock.patch.object(qualification.os, "stat", side_effect=racing_stat), self.assertRaisesRegex(
                qualification.QualificationError, "durably"
            ):
                qualification._write_new_file(target, b"data")
            self.assertTrue(target.is_symlink())
            self.assertEqual(os.readlink(target), "attacker-owned")

    def test_fsync_failure_rolls_back_created_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "fsync.json"
            with mock.patch.object(qualification.os, "fsync", side_effect=OSError("injected")), self.assertRaisesRegex(
                qualification.QualificationError, "durably"
            ):
                qualification._write_new_file(target, b"data")
            self.assertFalse(os.path.lexists(target))

    def test_post_parent_fsync_replacement_is_detected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "post-fsync.json"
            real_fsync = os.fsync
            calls = 0
            created_descriptor: int | None = None

            def swapping_fsync(descriptor: int) -> None:
                nonlocal calls, created_descriptor
                calls += 1
                real_fsync(descriptor)
                if calls == 1:
                    created_descriptor = descriptor
                if calls == 2:
                    self.assertIsNotNone(created_descriptor)
                    self.assertEqual(
                        os.fstat(created_descriptor).st_size,
                        len(b"qualified bytes"),
                    )
                    target.unlink()
                    target.symlink_to("attacker-after-fsync")

            with mock.patch.object(
                qualification.os, "fsync", side_effect=swapping_fsync
            ), self.assertRaisesRegex(
                qualification.QualificationError, "durably"
            ):
                qualification._write_new_file(target, b"qualified bytes")
            self.assertTrue(target.is_symlink())
            self.assertEqual(os.readlink(target), "attacker-after-fsync")

    def test_formal_rejects_standalone_receipt_before_science(self) -> None:
        with self.assertRaisesRegex(qualification.QualificationError, "path-backed"):
            qualification.run_qualification(
                qualification.FORMAL,
                engineering_evidence={"self_attested": True},
                include_ias15=True,
            )
        with self.assertRaisesRegex(qualification.QualificationError, "acknowledge-formal-cost"):
            qualification.main(["--profile", "formal", "--output", "/tmp/unused.json"])


class EngineeringEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = _base_report()["sources"]
        cls.runtime = _base_report()["runtime"]

    def test_path_backed_real_archives_derive_path_free_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            with mock.patch.object(
                qualification, "_build_tool_versions", return_value=qualification.ENGINEERING_TOOL_VERSIONS
            ):
                receipt = qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=False,
                )
            self.assertEqual(
                qualification._engineering_receipt(
                    receipt,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                ),
                receipt,
            )
            self.assertNotIn(directory.encode(), qualification._canonical_bytes(receipt))

    def test_engineering_lanes_ignore_ambient_python_optimize(self) -> None:
        stderr = (
            b"----------------------------------------------------------------------\n"
            b"Ran 45 tests in 1.000s\n\nOK\n"
        )

        def execute(
            command: list[str], *, env: dict[str, str], **kwargs: object
        ) -> mock.Mock:
            del kwargs
            self.assertNotIn("PYTHONOPTIMIZE", env)
            expected_level = 2 if "-OO" in command else 0
            self.assertEqual(command.count("-OO"), 1 if expected_level else 0)
            return mock.Mock(returncode=0, stdout=b"", stderr=stderr)

        with (
            mock.patch.dict(os.environ, {"PYTHONOPTIMIZE": "2"}),
            mock.patch.object(
                qualification.subprocess, "run", side_effect=execute
            ),
        ):
            normal = qualification._execute_engineering_lane(0, self.sources)
            optimized = qualification._execute_engineering_lane(2, self.sources)
        self.assertEqual(normal["optimization_level"], 0)
        self.assertEqual(optimized["optimization_level"], 2)

    def test_formal_reruns_tests_and_builds_with_stable_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            wheel = paths.wheel_build_a.read_bytes()
            sdist = paths.sdist_build_a.read_bytes()

            def execute(level: int, sources: dict[str, object]):
                return _engineering_log(level, sources, elapsed="9.876")

            def build(label: str, output: Path, sources: dict[str, object]):
                return _mock_release_execution(
                    label, output, sources, wheel, sdist
                )

            with (
                mock.patch.object(
                    qualification, "_execute_engineering_lane", side_effect=execute
                ),
                mock.patch.object(
                    qualification, "_execute_release_build", side_effect=build
                ),
                mock.patch.object(
                    qualification,
                    "_build_tool_versions",
                    return_value=qualification.ENGINEERING_TOOL_VERSIONS,
                ),
            ):
                receipt = qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=True,
                )
            self.assertIsNotNone(receipt["formal_execution_verification"])
            qualification._engineering_receipt(
                receipt,
                qualification._protocol(),
                self.sources,
                self.runtime,
            )

    def test_fabricated_logs_fail_formal_semantic_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)

            def execute(level: int, sources: dict[str, object]):
                return _engineering_log(level, sources, tests_run=40)

            with mock.patch.object(
                qualification, "_execute_engineering_lane", side_effect=execute
            ), self.assertRaisesRegex(
                qualification.QualificationError, "formal test reruns"
            ):
                qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=True,
                )

    def test_copied_synthetic_archives_fail_formal_clean_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            generated_wheel = _wheel_bytes(
                self.sources, extra_member=b"actual-formal-build"
            )
            generated_sdist = paths.sdist_build_a.read_bytes()

            def execute(level: int, sources: dict[str, object]):
                return _engineering_log(level, sources)

            def build(label: str, output: Path, sources: dict[str, object]):
                return _mock_release_execution(
                    label,
                    output,
                    sources,
                    generated_wheel,
                    generated_sdist,
                )

            with (
                mock.patch.object(
                    qualification, "_execute_engineering_lane", side_effect=execute
                ),
                mock.patch.object(
                    qualification, "_execute_release_build", side_effect=build
                ),
                self.assertRaisesRegex(
                    qualification.QualificationError, "do not match"
                ),
            ):
                qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=True,
                )

    def test_synthetic_wheel_extension_cannot_match_science_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            synthetic = _wheel_bytes(
                self.sources, extension_bytes=b"\x7fELFsynthetic-not-runtime"
            )
            paths.wheel_build_a.write_bytes(synthetic)
            paths.wheel_build_b.write_bytes(synthetic)
            with self.assertRaisesRegex(
                qualification.QualificationError, "extension payload"
            ):
                qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=False,
                )

    def test_formal_science_loads_validated_wheel_a_native_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            pair = qualification._artifact_pair(
                paths.wheel_build_a,
                paths.wheel_build_b,
                "wheel",
                self.sources,
                self.runtime["v9"],
            )
            loading = qualification._science_runtime_loading_record(
                pair, self.sources
            )
            original_wrapper = qualification.v9
            original = qualification.v9._native
            with qualification._formal_science_native_from_wheel(
                paths.wheel_build_a,
                loading,
                self.sources,
                self.runtime,
            ):
                self.assertIsNot(qualification.v9, original_wrapper)
                self.assertIsNot(qualification.v9._native, original)
                self.assertEqual(
                    qualification._sha256(
                        Path(qualification.v9.__file__).read_bytes()
                    ),
                    loading["wrapper_sha256"],
                )
                loaded = Path(qualification.v9._native.__file__).read_bytes()
                self.assertEqual(
                    qualification._sha256(loaded),
                    self.runtime["v9"]["extension_binary"]["sha256"],
                )
            self.assertIs(qualification.v9._native, original)
            self.assertIs(qualification.v9, original_wrapper)

    def test_formal_gate_wrapper_is_bound_to_retained_source(self) -> None:
        qualification._validate_formal_loaded_source_modules(self.sources)
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "fast_wisdom_holman_v9.py"
            alternate.write_bytes(Path(qualification.v9.__file__).read_bytes())
            with (
                mock.patch.object(
                    qualification.v9, "__file__", os.fspath(alternate)
                ),
                self.assertRaisesRegex(
                    qualification.QualificationError, "loaded source module"
                ),
            ):
                qualification._validate_formal_loaded_source_modules(
                    self.sources
                )

    def test_tiny_named_artifacts_and_byte_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            paths.wheel_build_a.write_bytes(b"x")
            paths.wheel_build_b.write_bytes(b"x")
            with self.assertRaisesRegex(qualification.QualificationError, "ZIP|wheel"):
                qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=False,
                )
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifact_paths(Path(directory), self.sources)
            paths.sdist_build_b.write_bytes(
                paths.sdist_build_b.read_bytes() + b"different"
            )
            with self.assertRaisesRegex(
                qualification.QualificationError, "differ"
            ):
                qualification._derive_engineering_receipt(
                    paths,
                    qualification._protocol(),
                    self.sources,
                    self.runtime,
                    formal_verify=False,
                )

    def test_sdist_member_cap_is_enforced_during_streaming(self) -> None:
        payload = io.BytesIO()
        root = "jxplanetx-0.6.0rc17/"
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for index in range(qualification.MAX_ARCHIVE_MEMBERS + 1):
                member = tarfile.TarInfo(f"{root}empty-{index}")
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
        with self.assertRaisesRegex(
            qualification.QualificationError, "structure|cap"
        ):
            qualification._validate_sdist_bytes(
                payload.getvalue(),
                "jxplanetx-0.6.0rc17.tar.gz",
                self.sources,
            )

    def test_zip_member_cap_does_not_trust_forged_eocd_count(self) -> None:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
            for index in range(qualification.MAX_ARCHIVE_MEMBERS + 1):
                archive.writestr(f"jxplanetx/empty-{index}", b"")
        forged = bytearray(payload.getvalue())
        eocd = forged.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd, 0)
        forged[eocd + 8 : eocd + 10] = (1).to_bytes(2, "little")
        forged[eocd + 10 : eocd + 12] = (1).to_bytes(2, "little")
        with self.assertRaisesRegex(
            qualification.QualificationError, "directory|entry"
        ):
            qualification._validate_wheel_bytes(
                bytes(forged),
                "jxplanetx-0.6.0rc17-cp314-cp314-linux_x86_64.whl",
                self.sources,
                self.runtime["v9"]["extension_binary"],
            )

    def test_logs_revalidate_command_exit_and_source(self) -> None:
        for field in ("exit_code", "command", "execution_head"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                paths = _artifact_paths(Path(directory), self.sources)
                log = qualification._load_json_file(paths.normal_log)
                log[field] = 1 if field == "exit_code" else ["wrong"] if field == "command" else "f" * 40
                core = {key: log[key] for key in log if key != "content_sha256"}
                log["content_sha256"] = qualification._domain_sha256(
                    "jx.wh-v9.qualification.engineering-command-result.v1", core
                )
                paths.normal_log.write_bytes(qualification._canonical_bytes(log, newline=True))
                with self.assertRaisesRegex(qualification.QualificationError, "engineering command"):
                    qualification._derive_engineering_receipt(
                        paths,
                        qualification._protocol(),
                        self.sources,
                        self.runtime,
                        formal_verify=False,
                    )

    def test_safe_cli_creates_evidence_without_scientific_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _artifact_paths(root, self.sources, write_logs=False)
            receipt_path = root / "receipt.json"

            def execute(level: int, sources: dict[str, object]):
                return _engineering_log(level, sources)

            with (
                mock.patch.object(qualification, "_source_custody", return_value=copy.deepcopy(self.sources)),
                mock.patch.object(qualification, "_execute_engineering_lane", side_effect=execute),
                mock.patch.object(qualification, "_build_tool_versions", return_value=qualification.ENGINEERING_TOOL_VERSIONS),
                mock.patch.object(qualification, "run_qualification", side_effect=AssertionError("forbidden")),
            ):
                status = qualification.main(
                    [
                        "--create-engineering-receipt", str(receipt_path),
                        "--normal-log", str(paths.normal_log),
                        "--optimized-log", str(paths.optimized_log),
                        "--wheel-build-a", str(paths.wheel_build_a),
                        "--wheel-build-b", str(paths.wheel_build_b),
                        "--sdist-build-a", str(paths.sdist_build_a),
                        "--sdist-build-b", str(paths.sdist_build_b),
                    ]
                )
            self.assertEqual(status, 0)
            receipt = qualification._load_json_file(receipt_path)
            qualification._engineering_receipt(
                receipt,
                qualification._protocol(),
                self.sources,
                self.runtime,
            )


class OracleAndRuntimeContractTests(unittest.TestCase):
    def test_decimal_oracle_and_standalone_rhs(self) -> None:
        case = qualification.AnalyticCase(1.0, 0.6, np.pi / 3.0, 0.009, -1)
        position_0, velocity_0 = qualification._analytic_state(case, 0.0)
        period = 2.0 * np.pi * np.sqrt(1.0 / 1.009)
        position_1, velocity_1 = qualification._analytic_state(case, float(period))
        gm = np.array((1.0, 0.009))
        np.testing.assert_allclose(gm @ position_0, np.zeros(3), atol=2e-18)
        np.testing.assert_allclose(gm @ velocity_0, np.zeros(3), atol=2e-18)
        np.testing.assert_allclose(position_1, position_0, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(velocity_1, velocity_0, rtol=0.0, atol=2e-15)
        positions, velocities = qualification.fixture._initial_arrays()
        masses = qualification.fixture.GRAVITATIONAL_PARAMETERS.copy()
        derivative = qualification._standalone_newtonian_rhs(
            0.0, np.concatenate((positions.ravel(), velocities.ravel())), masses
        )
        np.testing.assert_allclose(masses @ derivative[9:].reshape(3, 3), np.zeros(3), atol=5e-19)

    def test_rebound_manifest_is_exact_and_path_free(self) -> None:
        runtime = _valid_rebound_runtime()
        qualification._validate_rebound_runtime_record(runtime)
        for mutation in ("wrapper", "absolute", "tree"):
            forged = copy.deepcopy(runtime)
            if mutation == "wrapper":
                forged["python_sources"]["required_execution_wrappers"].pop()
            elif mutation == "absolute":
                forged["python_sources"]["files"][0]["path"] = "/tmp/x.py"
            else:
                forged["python_sources"]["tree_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                qualification.QualificationError, "REBOUND"
            ):
                qualification._validate_rebound_runtime_record(forged)


if __name__ == "__main__":
    unittest.main()
