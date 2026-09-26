from __future__ import annotations

import gzip
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import tomllib
import unittest

from tools.build_release import (
    STAGE_DIRECTORIES,
    STAGE_FILES,
    canonicalize_sdist,
    project_metadata,
    sha256,
    stage_source,
)
from tools.run_local_test_matrix import require_python
from tools.run_cuda_release_gate import CUDA_TEST_TARGETS, EXPECTED_TEST_COUNT


ROOT = Path(__file__).resolve().parents[1]


class ReproducibleReleaseTests(unittest.TestCase):
    @staticmethod
    def _write_archive(path: Path, *, mtime: int, uid: int, uname: str) -> None:
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename=path.name,
                mode="wb",
                fileobj=raw,
                mtime=mtime,
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    directory = tarfile.TarInfo("jxplanetx-0.6.0rc16")
                    directory.type = tarfile.DIRTYPE
                    directory.mode = 0o775
                    directory.mtime = mtime
                    directory.uid = uid
                    directory.gid = uid
                    directory.uname = uname
                    directory.gname = uname
                    archive.addfile(directory)

                    payload = b"screening only\n"
                    member = tarfile.TarInfo("jxplanetx-0.6.0rc16/CLAIMS.txt")
                    member.mode = 0o664
                    member.mtime = mtime + 1
                    member.uid = uid
                    member.gid = uid
                    member.uname = uname
                    member.gname = uname
                    member.size = len(payload)
                    archive.addfile(member, BytesIO(payload))

    def test_release_identity_and_exact_backend_pin(self) -> None:
        version, backend = project_metadata(ROOT)
        self.assertEqual(version, "0.6.0rc16")
        self.assertEqual(backend, "78.1.1")
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            metadata["project"]["license"], "LicenseRef-JX-Proprietary"
        )
        self.assertEqual(
            metadata["project"]["license-files"],
            ["LICENSE", "AUTHORS.md", "THIRD_PARTY_NOTICES.md"],
        )
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 Lino Avila", license_text)
        self.assertIn("All rights reserved", license_text)
        self.assertNotIn("Permission is hereby granted", license_text)

    def test_distribution_stage_excludes_research_evidence(self) -> None:
        self.assertEqual(
            STAGE_FILES,
            (
                "pyproject.toml",
                "README.md",
                "LICENSE",
                "AUTHORS.md",
                "THIRD_PARTY_NOTICES.md",
                "benchmarks/jx_smalln_cpu_rkf78.c",
            ),
        )
        self.assertEqual(STAGE_DIRECTORIES, ("src/jxplanetx",))
        declared = set(STAGE_FILES + STAGE_DIRECTORIES)
        self.assertFalse(
            declared & {"library", "runs", "evidence", "archives", "benchmarks"}
        )
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            stage_source(ROOT, stage)
            staged_files = tuple(
                path.relative_to(stage).as_posix()
                for path in stage.rglob("*")
                if path.is_file()
            )
            self.assertFalse(
                any(
                    name.endswith((".so", ".pyd", ".dylib", ".pyc", ".pyo"))
                    for name in staged_files
                )
            )
            self.assertIn(
                "src/jxplanetx/_wisdom_holman_cpu_module.c",
                staged_files,
            )
            self.assertIn(
                "src/jxplanetx/_wisdom_holman_loop.c",
                staged_files,
            )
            self.assertIn(
                "src/jxplanetx/_wisdom_holman_cpu_v2_module.c",
                staged_files,
            )
            self.assertIn(
                "src/jxplanetx/_wisdom_holman_loop_v2.c",
                staged_files,
            )
            self.assertIn(
                "src/jxplanetx/fast_wisdom_holman.py",
                staged_files,
            )
            self.assertIn("src/jxplanetx/_gr15_core.c", staged_files)
            self.assertIn("src/jxplanetx/_gr15_core.h", staged_files)
            self.assertIn("src/jxplanetx/_gr15_cpu_module.c", staged_files)
            self.assertIn("src/jxplanetx/_gr15_v2_reference.py", staged_files)
            self.assertIn("src/jxplanetx/_gr15_v3_core.c", staged_files)
            self.assertIn("src/jxplanetx/_gr15_v3_core.h", staged_files)
            self.assertIn("src/jxplanetx/_gr15_v3_cpu_module.c", staged_files)
            self.assertIn("src/jxplanetx/gr15.py", staged_files)
            self.assertIn("src/jxplanetx/_eih_1pn_core.c", staged_files)
            self.assertIn("src/jxplanetx/_eih_1pn_core.h", staged_files)
            self.assertIn("src/jxplanetx/_eih_1pn_cpu_module.c", staged_files)
            self.assertIn("src/jxplanetx/_gr15_eih_1pn_core.c", staged_files)
            self.assertIn("src/jxplanetx/_gr15_eih_1pn_core.h", staged_files)
            self.assertIn("src/jxplanetx/_gr15_eih_1pn_cpu_module.c", staged_files)
            self.assertIn("src/jxplanetx/gr15_eih_1pn.py", staged_files)
            self.assertIn("src/jxplanetx/gr15_eih_1pn_batch.py", staged_files)
            self.assertIn("src/jxplanetx/eih_1pn_cuda.py", staged_files)
            self.assertIn("src/jxplanetx/solar_system/eih_1pn.py", staged_files)
            for path in stage.rglob("*"):
                expected_mode = 0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
                self.assertEqual(path.stat().st_mode & 0o777, expected_mode)

    def test_native_extensions_are_exactly_scoped_and_screening_only(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extensions = metadata["tool"]["setuptools"]["ext-modules"]
        self.assertEqual(len(extensions), 9)
        self.assertEqual(extensions[0]["name"], "jxplanetx._smalln_cpu")
        self.assertEqual(
            extensions[0]["sources"],
            [
                "src/jxplanetx/_smalln_cpu_module.c",
                "benchmarks/jx_smalln_cpu_rkf78.c",
            ],
        )
        self.assertEqual(extensions[0]["libraries"], ["m"])
        self.assertEqual(
            extensions[0]["extra-compile-args"],
            ["-O3", "-g0", "-std=c11", "-fno-fast-math", "-ffp-contract=off"],
        )
        self.assertEqual(
            sha256(ROOT / "benchmarks/jx_smalln_cpu_rkf78.c"),
            "3952fb198482261d2ac210f3a9ef3f941af60c56c9e97bcf403b3b937c73397c",
        )
        wrapper = (ROOT / "src/jxplanetx/_smalln_cpu_module.c").read_text()
        self.assertIn('"SCREENING_ONLY"', wrapper)
        self.assertEqual(
            extensions[1],
            {
                "name": "jxplanetx._wisdom_holman_cpu",
                "sources": [
                    "src/jxplanetx/_wisdom_holman_cpu_module.c",
                    "src/jxplanetx/_wisdom_holman_loop.c",
                ],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_cpu_module.c"),
            "85c454431e3414aa0d44be3887e6d4e83a43a040dbe5a8b6eb692a9b53c98ea5",
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_loop.c"),
            "2d66d776bd493a5c3ef0bed2deb5d78bb27f397a9e302daa186290d52588315c",
        )
        wh_wrapper = (
            ROOT / "src/jxplanetx/_wisdom_holman_cpu_module.c"
        ).read_text()
        self.assertIn('"SCREENING_ONLY"', wh_wrapper)
        self.assertIn(
            '"jx.wisdom-holman.native-g-bundle.prototype.v1"',
            wh_wrapper,
        )
        self.assertIn(
            '"jx.wisdom-holman.native-complete-map.prototype.v1"',
            wh_wrapper,
        )
        self.assertEqual(
            extensions[2],
            {
                "name": "jxplanetx._wisdom_holman_cpu_v2",
                "sources": [
                    "src/jxplanetx/_wisdom_holman_cpu_v2_module.c",
                    "src/jxplanetx/_wisdom_holman_loop_v2.c",
                ],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_cpu_v2_module.c"),
            "65e4ce5b9c40879e81eabc19641697085ab7f052953839bad9ae4f70eeed1a84",
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_loop_v2.c"),
            "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750",
        )
        wh_v2_wrapper = (
            ROOT / "src/jxplanetx/_wisdom_holman_cpu_v2_module.c"
        ).read_text()
        self.assertIn('"SCREENING_ONLY"', wh_v2_wrapper)
        self.assertIn(
            '"jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"',
            wh_v2_wrapper,
        )
        self.assertIn(
            '"jx.wisdom-holman.native-g-bundle.prototype.v1"',
            wh_v2_wrapper,
        )
        self.assertIn(
            '"jx.wisdom-holman.native-complete-map.prototype.v2"',
            wh_v2_wrapper,
        )
        self.assertEqual(
            extensions[3]["name"], "jxplanetx._wisdom_holman_cpu_v3"
        )
        self.assertEqual(
            extensions[3]["sources"],
            [
                "src/jxplanetx/_wisdom_holman_cpu_v3_module.c",
                "src/jxplanetx/_wisdom_holman_loop_v3.c",
            ],
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_cpu_v3_module.c"),
            "ee0e40a075b481516d8d30f8824b1fe31c9051c8c0de1dc8b321155882c411b3",
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_wisdom_holman_loop_v3.c"),
            "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629",
        )
        self.assertEqual(
            extensions[4]["name"], "jxplanetx._wisdom_holman_cpu_v4"
        )
        self.assertEqual(
            extensions[4]["sources"],
            [
                "src/jxplanetx/_wisdom_holman_cpu_v4_module.c",
                "src/jxplanetx/_wisdom_holman_loop_v4.c",
            ],
        )
        self.assertIn(
            '"jx.wisdom-holman.native-complete-map.endpoint-seed.prototype.v4"',
            (ROOT / "src/jxplanetx/_wisdom_holman_cpu_v4_module.c").read_text(),
        )
        self.assertEqual(
            extensions[5],
            {
                "name": "jxplanetx._gr15_cpu",
                "sources": [
                    "src/jxplanetx/_gr15_cpu_module.c",
                    "src/jxplanetx/_gr15_core.c",
                ],
                "depends": ["src/jxplanetx/_gr15_core.h"],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        gr15_wrapper = (ROOT / "src/jxplanetx/_gr15_cpu_module.c").read_text()
        self.assertIn('"SCREENING_ONLY"', gr15_wrapper)
        self.assertIn('"JX_GAUSS_RADAU15_V2"', gr15_wrapper)
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_gr15_core.c"),
            "9a6cb532e88c732f6653c2763d9510036432d9930d930c19a5d259a03b3b514c",
        )
        self.assertEqual(
            extensions[6],
            {
                "name": "jxplanetx._gr15_v3_cpu",
                "sources": [
                    "src/jxplanetx/_gr15_v3_cpu_module.c",
                    "src/jxplanetx/_gr15_v3_core.c",
                ],
                "depends": ["src/jxplanetx/_gr15_v3_core.h"],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        gr15_v3_wrapper = (
            ROOT / "src/jxplanetx/_gr15_v3_cpu_module.c"
        ).read_text()
        self.assertIn('"SCREENING_ONLY"', gr15_v3_wrapper)
        self.assertIn('"JX_GAUSS_RADAU15_V3"', gr15_v3_wrapper)
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_gr15_v3_core.c"),
            "036d1dc7e4e2e26c908c6dde9ff27bdf6a5804b88aa3f3e374f1f512642c1e41",
        )
        self.assertEqual(
            extensions[7],
            {
                "name": "jxplanetx._eih_1pn_cpu",
                "sources": [
                    "src/jxplanetx/_eih_1pn_cpu_module.c",
                    "src/jxplanetx/_eih_1pn_core.c",
                ],
                "depends": ["src/jxplanetx/_eih_1pn_core.h"],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_eih_1pn_core.c"),
            "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746",
        )
        self.assertEqual(
            extensions[8],
            {
                "name": "jxplanetx._gr15_eih_1pn_cpu",
                "sources": [
                    "src/jxplanetx/_gr15_eih_1pn_cpu_module.c",
                    "src/jxplanetx/_gr15_eih_1pn_core.c",
                    "src/jxplanetx/_eih_1pn_core.c",
                ],
                "depends": [
                    "src/jxplanetx/_gr15_eih_1pn_core.h",
                    "src/jxplanetx/_eih_1pn_core.h",
                ],
                "libraries": ["m"],
                "extra-compile-args": [
                    "-O3",
                    "-g0",
                    "-std=c11",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ],
            },
        )
        self.assertEqual(
            sha256(ROOT / "src/jxplanetx/_gr15_eih_1pn_core.c"),
            "bef3a607c4c3674d342745570c68b462a6178e789089c84419d6f23dd810fad1",
        )
        eih_wrapper = (ROOT / "src/jxplanetx/_gr15_eih_1pn_cpu_module.c").read_text()
        self.assertIn('"JX_GR15_EIH1PN_V1"', eih_wrapper)
        self.assertIn('"SCREENING_ONLY"', eih_wrapper)

    def test_packaging_workflow_separates_package_and_registry_versions(self) -> None:
        workflow = (ROOT / ".github/workflows/test.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("JX_PACKAGE_VERSION", workflow)
        self.assertIn("jxplanetx-release", workflow)
        self.assertIn(
            "assert jxplanetx.__version__ == os.environ[\"JX_PACKAGE_VERSION\"]",
            workflow,
        )
        self.assertIn(
            'assert summary["version"] == "0.6.0rc3"',
            workflow,
        )
        self.assertIn(
            "_wisdom_holman_cpu_v2",
            workflow,
        )
        self.assertIn("_wisdom_holman_cpu_v3", workflow)
        self.assertIn("_wisdom_holman_cpu_v4", workflow)
        self.assertIn("_gr15_cpu", workflow)
        self.assertIn("_gr15_v3_cpu", workflow)
        self.assertIn("_eih_1pn_cpu", workflow)
        self.assertIn("_gr15_eih_1pn_cpu", workflow)
        self.assertIn("JX_GAUSS_RADAU15_V2", workflow)
        self.assertIn("JX_GAUSS_RADAU15_V3", workflow)

    def test_sdist_canonicalization_removes_host_metadata(self) -> None:
        epoch = 1_700_000_000
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.tar.gz"
            second = Path(directory) / "second.tar.gz"
            self._write_archive(first, mtime=1_800_000_000, uid=1000, uname="alpha")
            self._write_archive(second, mtime=1_900_000_000, uid=2000, uname="beta")
            self.assertNotEqual(sha256(first), sha256(second))

            canonicalize_sdist(first, epoch)
            canonicalize_sdist(second, epoch)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first, "r:gz") as archive:
                for member in archive.getmembers():
                    self.assertEqual(member.mtime, epoch)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.uname, "")
                    self.assertEqual(member.gname, "")
                    self.assertIn(member.mode, (0o644, 0o755))

    def test_artifact_manifest_shape_is_json_safe(self) -> None:
        example = {
            "schema": "jxplanetx.reproducible-release-artifacts.v1",
            "version": "0.6.0rc16",
            "scientific_claim_state": "SCREENING_ONLY",
        }
        self.assertEqual(json.loads(json.dumps(example, allow_nan=False)), example)

    def test_rc11_manifest_binds_gr15_eih_sources_and_claim_ceiling(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc11.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["package_version"], "0.6.0rc11")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["status"],
            "LOCAL_RELEASE_CANDIDATE_READY_NOT_PUBLISHED",
        )
        self.assertEqual(
            manifest["gr15_eih_1pn"]["newtonian_gr15_v3_core_sha256"],
            sha256(ROOT / "src/jxplanetx/_gr15_v3_core.c"),
        )
        self.assertFalse(
            manifest["gr15_eih_1pn"]["newtonian_gr15_v3_source_changed"]
        )
        source_paths = {
            "de440_attribution_runner_sha256": (
                "benchmarks/jx_gr15_eih_1pn_de440_attribution.py"
            ),
            "de440_test_sha256": "tests/test_gr15_eih_1pn_de440.py",
            "external_test_sha256": "tests/test_gr15_eih_1pn_external.py",
            "gr15_eih_python_sha256": "src/jxplanetx/gr15_eih_1pn.py",
            "local_test_sha256": "tests/test_gr15_eih_1pn.py",
        }
        for key, relative in source_paths.items():
            self.assertEqual(manifest["source_bindings"][key], sha256(ROOT / relative))
        self.assertEqual(
            manifest["source_bindings"]["package_init_sha256"],
            "984db22f933e7c9cc70e6d851bd287c6811090c9d15023edc6c645adf27f5147",
        )
        self.assertEqual(
            manifest["source_bindings"]["pyproject_sha256"],
            "f20cb8b0f7fc4ead8b810ca903a010deb66435845f6ca5d2055e9dd88cc14086",
        )
        self.assertTrue(
            all(
                value == "PASS"
                for value in manifest["scientific_validation"]["layers"].values()
            )
        )
        self.assertTrue(
            manifest["scientific_validation"]["de440_attribution"][
                "all_locked_gates_passed"
            ]
        )
        self.assertTrue(
            manifest["reproducible_packaging"][
                "fresh_wheel_public_api_smoke_passed"
            ]
        )
        self.assertTrue(
            manifest["reproducible_packaging"][
                "fresh_sdist_public_api_smoke_passed"
            ]
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])
        self.assertFalse(manifest["release_actions"]["push_performed"])
        self.assertFalse(manifest["claim_controls"]["exact_general_relativity_claimed"])
        self.assertFalse(manifest["claim_controls"]["de440_equivalence_claimed"])
        self.assertFalse(
            manifest["claim_controls"]["navigation_or_production_authorized"]
        )

    def test_rc13_manifest_binds_cuda_gr15_and_claim_ceiling(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc13.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["package_version"], "0.6.0rc13")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["status"], "LOCAL_RELEASE_CANDIDATE_PACKAGED_NOT_PUBLISHED"
        )
        unchanged_bindings = {
            "benchmark_sha256": "benchmarks/jx_gr15_eih_1pn_cuda_scale.py",
            "cuda_gr15_module_sha256": "src/jxplanetx/gr15_eih_1pn_cuda.py",
            "cuda_gr15_test_sha256": "tests/test_gr15_eih_1pn_cuda.py",
        }
        for key, relative in unchanged_bindings.items():
            self.assertEqual(manifest["source_bindings"][key], sha256(ROOT / relative))
        historical_mutable_bindings = {
            "package_init_sha256": "49b3d4559d658eed842481f2b7a8b8d0cef944adc52f8ab54d4586e6c9d547fb",
            "pyproject_sha256": "0310e02d165915afd1f523049cf7c8772af4946ea5ba544cbeb2d757f59a0c88",
            "readme_sha256": "4821c4d55e396579624f20bd8d3480afcf27e050bef3991641e6005d2b0a59dd",
            "test_matrix_manifest_sha256": "88e57588bb4dbabe518a7e562c2d1ed384fe2987f97fc121f9a2aab856d3a386",
        }
        for key, expected in historical_mutable_bindings.items():
            self.assertEqual(manifest["source_bindings"][key], expected)
        cuda = manifest["cuda_gr15_eih_1pn"]
        self.assertEqual(cuda["gr15_stage_count"], 8)
        self.assertEqual(cuda["body_count"], {"minimum": 2, "maximum": 32})
        self.assertTrue(cuda["complete_gr15_cuda_integrator"])
        self.assertFalse(cuda["implicit_host_device_input_transfer"])
        self.assertEqual(
            manifest["verification"]["full_live_matrix"]["isolated_file_lanes"],
            220,
        )
        self.assertEqual(
            manifest["verification"]["cuda_hardware_profile"]["test_count"],
            36,
        )
        self.assertTrue(
            manifest["reproducible_packaging"]["fresh_wheel_cpu_cuda_smoke_passed"]
        )
        self.assertTrue(
            manifest["reproducible_packaging"]["fresh_sdist_cpu_cuda_smoke_passed"]
        )
        self.assertEqual(
            manifest["reproducible_packaging"]["twine_check"],
            {"status": "PASS", "version": "6.2.0"},
        )
        self.assertFalse(
            manifest["claim_controls"]["production_cuda_gr15_authorized"]
        )
        self.assertFalse(
            manifest["claim_controls"]["general_cpu_gpu_superiority_claimed"]
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_rc14_manifest_binds_supported_batch_dispatch_and_claim_ceiling(
        self,
    ) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc14.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["package_version"], "0.6.0rc14")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["status"], "LOCAL_RELEASE_CANDIDATE_PACKAGED_NOT_PUBLISHED"
        )
        unchanged_source_paths = {
            "batch_module_sha256": "src/jxplanetx/gr15_eih_1pn_batch.py",
            "batch_cpu_test_sha256": "tests/test_gr15_eih_1pn_batch.py",
            "batch_cuda_test_sha256": "tests/test_gr15_eih_1pn_batch_cuda.py",
            "dispatch_benchmark_sha256": (
                "benchmarks/jx_gr15_dispatch_rc14_acceptance.py"
            ),
            "dispatch_protocol_sha256": (
                "benchmarks/jx_gr15_dispatch_rc14_protocol.json"
            ),
        }
        for key, relative in unchanged_source_paths.items():
            self.assertEqual(manifest["source_bindings"][key], sha256(ROOT / relative))
        historical_mutable_bindings = {
            "package_init_sha256": "da497667bb842892f07b776e1f61bb33ebafea2ad6960efadd2d8cbce1577183",
            "pyproject_sha256": "a17a5aa08a60d4cb46e0f4246ebcd8bcd107bdd238b995e0beb3ad68dd62bd9f",
            "readme_sha256": "a3b9f200e002f6ab2c13ea7b1ef26715b0ecc95b1837b7ab2c049d01ab0fb24a",
            "reproducible_release_test_sha256": "82c87329a41970c66dced02770ed86cc4b69c44d69f50b8291585370116706b2",
            "test_matrix_manifest_sha256": "4978ff9f0cb6a3ea6b8e6175fcb81cf4fa022fb2d82cb01be666affae10ece89",
        }
        for key, expected in historical_mutable_bindings.items():
            self.assertEqual(manifest["source_bindings"][key], expected)
        batch = manifest["gr15_eih_1pn_batch"]
        self.assertEqual(batch["cpu_workers_default"], 8)
        self.assertEqual(batch["automatic_cuda_minimum_system_count"], 128)
        self.assertEqual(batch["public_auto_api"], "integrate_gr15_eih_1pn_batch")
        self.assertEqual(
            batch["public_cpu_api"], "integrate_gr15_eih_1pn_cpu_batch"
        )
        self.assertTrue(batch["unmatched_auto_route_is_cpu"])
        self.assertTrue(batch["explicit_cuda_never_silently_falls_back"])
        screen = manifest["supported_dispatch_acceptance"]
        self.assertEqual(screen["status"], "PASS")
        self.assertEqual(screen["observed_stable_cuda_crossover"], 96)
        self.assertEqual(screen["automatic_cuda_cutoff"], 128)
        self.assertFalse(screen["portable_crossover_claimed"])
        self.assertEqual(
            manifest["verification"]["full_live_matrix"]["isolated_file_lanes"],
            222,
        )
        self.assertEqual(
            manifest["verification"]["cuda_hardware_profile"]["test_count"],
            37,
        )
        packaging = manifest["reproducible_packaging"]
        self.assertTrue(packaging["fresh_wheel_cpu_cuda_smoke_passed"])
        self.assertTrue(packaging["fresh_sdist_cpu_cuda_smoke_passed"])
        self.assertEqual(
            packaging["twine_check"], {"status": "PASS", "version": "6.2.0"}
        )
        self.assertFalse(manifest["claim_controls"]["production_authorized"])
        self.assertFalse(
            manifest["claim_controls"]["general_rebound_superiority_claimed"]
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_rc15_manifest_binds_verified_execution_and_claim_ceiling(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc15.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["package_version"], "0.6.0rc15")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["status"], "LOCAL_RELEASE_CANDIDATE_PACKAGED_NOT_PUBLISHED"
        )
        # Rc15 is immutable history.  Later candidates intentionally change the
        # active package, README, matrix, and this test, so comparing a frozen
        # manifest to the current tree would make every legitimate successor
        # fail.  Keep custody by asserting the exact recorded rc15 bindings.
        self.assertEqual(
            manifest["source_bindings"],
            {
                "eih_force_core_sha256": "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746",
                "gr15_eih_core_sha256": "bef3a607c4c3674d342745570c68b462a6178e789089c84419d6f23dd810fad1",
                "gr15_eih_module_sha256": "aa694614e55feb827db5075c46d1489bcdca580dad7d0843957de948d2dddc85",
                "package_init_sha256": "d340b73ae1eb207fd345e45877395bf5ae74dba584c27b2452fee63674663123",
                "pyproject_sha256": "2f4ff1b9cddc1565f62ed24c6114c52e5eddbd8715a878a26dc864a354927463",
                "readme_sha256": "24ff5653e5fe610574c87d6d2811f7bc258bca5a744c100c43b3bf8f80184339",
                "reproducible_release_test_sha256": "f38cc78cd7557a4991bdf92ffc02b3a71bd9bdba31aa460db03dc89a47debb60",
                "test_matrix_manifest_sha256": "be045e4b47ee3af13ffa86df4dd5032b095a126c89bd6892c03c4eaeddc07b34",
                "verified_acceptance_runner_sha256": "97ef6ccf2b42275f9b5c4724d6a94bc08d47298e460c51a5d997c566f63e1209",
                "verified_module_sha256": "4884fee9a85c0513b44c3f038217562f9f85b48155e9ab77ae1096bb505b79a8",
                "verified_protocol_sha256": "3edd208fef7f4fbf1aae2b5dd0f1737f46f767200cb53762424d2376517c163f",
                "verified_test_sha256": "8af3f00217b1e765e9131fc2e39623899df0f7440e9a4923d93523d6832550a1",
            },
        )
        component = manifest["verified_gr15_eih_1pn"]
        self.assertEqual(
            component["public_api"], "integrate_verified_gr15_eih_1pn"
        )
        self.assertTrue(component["full_checkpoint_trajectory_comparison"])
        self.assertTrue(component["confirmation_must_be_no_looser"])
        self.assertFalse(component["independent_implementation_confirmation"])
        acceptance = manifest["verified_execution_acceptance"]
        self.assertEqual(acceptance["status"], "PASS")
        self.assertEqual(acceptance["case_count"], 4)
        self.assertTrue(acceptance["all_non_timing_repeats_exact"])
        self.assertTrue(acceptance["all_primary_results_exact"])
        self.assertEqual(
            acceptance["ten_year_eleven_body"][
                "maximum_position_difference_km"
            ],
            0.002440939204001141,
        )
        self.assertEqual(
            manifest["verification"]["full_live_matrix"]["isolated_file_lanes"],
            223,
        )
        packaging = manifest["reproducible_packaging"]
        self.assertTrue(packaging["fresh_wheel_verified_smoke_passed"])
        self.assertTrue(packaging["fresh_sdist_verified_smoke_passed"])
        self.assertEqual(
            packaging["twine_check"], {"status": "PASS", "version": "6.2.0"}
        )
        self.assertTrue(
            manifest["claim_controls"]["verified_numerical_execution_claimed"]
        )
        self.assertFalse(manifest["claim_controls"]["production_authorized"])
        self.assertFalse(
            manifest["claim_controls"]["independent_confirmation_claimed"]
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_explicit_runtime_keeps_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "python"
            runtime.symlink_to(Path(sys.executable))
            selected = require_python(runtime.absolute(), "test runtime")
            self.assertEqual(selected, runtime.absolute())
            self.assertNotEqual(selected, runtime.resolve())

    def test_cuda_release_gate_has_an_exact_non_skippable_core_roster(self) -> None:
        self.assertEqual(EXPECTED_TEST_COUNT, 5)
        self.assertEqual(
            CUDA_TEST_TARGETS,
            (
                "tests.test_engine_backends.OptionalCuPyParityTests",
                "tests.test_engine_trajectory.OptionalCuPyTrajectoryTests",
            ),
        )

    def test_rc3_manifest_binds_exact_artifacts_and_honest_claim_limits(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc3.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["version"], "0.6.0rc3")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(manifest["general_dynamics_registry"]["generation"], 18)
        self.assertEqual(
            manifest["general_dynamics_registry"]["sha256"],
            sha256(ROOT / "runs/jx_general_dynamics_registry_v18/REGISTRY.json"),
        )
        self.assertEqual(
            manifest["verification"]["coupled_lunar_increment"]["binding_sha256"],
            sha256(ROOT / "runs/jx_lunar_deformation_geodetic_screen_v1/SUMMARY.json"),
        )
        self.assertFalse(
            manifest["verification"]["full_live_research_matrix"]["rerun_for_rc3"]
        )
        self.assertEqual(
            manifest["verification"]["cuda_core_release_gate"]["skips"], 0
        )
        self.assertFalse(manifest["release_actions"]["publication_performed"])
        self.assertIn(
            "The full live and historical research matrix was not rerun for rc3.",
            manifest["claim_limits"],
        )

    def test_rc4_manifest_records_supported_smalln_and_two_machine_screen(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc4.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["version"], "0.6.0rc4")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(manifest["general_dynamics_registry"]["generation"], 18)
        self.assertEqual(
            manifest["general_dynamics_registry"]["registry_version"],
            "0.6.0rc3",
        )
        self.assertTrue(
            manifest["general_dynamics_registry"]["historical_binding_preserved"]
        )
        native = manifest["reproducible_packaging"]["native_smalln_extension"]
        self.assertEqual(native["supported_public_wrapper"], "jxplanetx.smalln")
        self.assertEqual(native["scientific_claim_state"], "SCREENING_ONLY")
        comparison = manifest["verification"]["accuracy_matched_rebound_v8"]
        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(comparison["single_system_fastest"], "rebound-ias15-sequential")
        second = manifest["verification"]["second_machine_kit"]
        self.assertEqual(second["rtx4050_execution_status"], "PASS")
        self.assertTrue(second["initial_v2_import_closure_failure_before_timing"])
        self.assertFalse(second["organizational_independence_claimed"])
        aggregate = manifest["verification"]["multi_machine_aggregate"]
        self.assertEqual(aggregate["status"], "PASS_MULTI_MACHINE_SCREENING_ONLY")
        self.assertEqual(aggregate["machine_count"], 2)
        self.assertFalse(aggregate["organizational_independence_claimed"])
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_rc5_manifest_binds_public_latency_and_noise_sensitive_route(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc5.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["version"], "0.6.0rc5")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(manifest["general_dynamics_registry"]["generation"], 18)
        self.assertTrue(
            manifest["general_dynamics_registry"]["historical_binding_preserved"]
        )
        native = manifest["reproducible_packaging"]["native_smalln_extension"]
        self.assertEqual(
            native["public_wrapper_sha256"],
            sha256(ROOT / "src/jxplanetx/smalln.py"),
        )
        self.assertFalse(native["native_arithmetic_changed_for_rc5"])
        comparison = manifest["verification"][
            "accuracy_matched_public_cpu_cuda_rebound"
        ]
        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(
            comparison["single_system_fastest_for_recorded_rows"],
            "jx-compiled-cpu-rkf78",
        )
        self.assertTrue(
            all(
                row["public_cpu_to_rebound_latency_ratio"] < 1.0
                for row in comparison["rows"]
            )
        )
        boundary = manifest["verification"]["discrete_crossover_boundary_audit"]
        self.assertEqual(boundary["timed_repetitions"], 101)
        self.assertTrue(boundary["authoritative_for_public_route_boundary"])
        route = manifest["verification"]["public_dispatch_route"]
        self.assertEqual(route["eight_lane_cuda_body_counts"], [32])
        self.assertFalse(route["portable_routing_threshold_claimed"])
        parallel = manifest["verification"][
            "accuracy_matched_fused_cuda_rebound_eight_workers"
        ]
        self.assertEqual(parallel["status"], "PASS")
        self.assertEqual(parallel["jx_version"], "0.6.0rc5")
        self.assertEqual(parallel["rebound_process_workers"], 8)
        self.assertEqual(parallel["independent_systems"], 256)
        self.assertTrue(
            all(
                row["jx_to_rebound_eight_worker_integration_throughput_ratio"]
                > 1.0
                for row in parallel["rows"]
            )
        )
        self.assertFalse(parallel["portable_speed_ranking_claimed"])
        long_horizon = manifest["verification"][
            "solar_system_point_mass_100_year_screen"
        ]
        self.assertEqual(long_horizon["status"], "PASS_SCREENING_ONLY")
        self.assertEqual(long_horizon["jx_version"], "0.6.0rc5")
        self.assertFalse(long_horizon["timing_comparable"])
        self.assertFalse(long_horizon["coupled_lunar_physics_tested"])
        portfolio = manifest["verification"][
            "appropriate_rebound_solver_portfolio"
        ]
        self.assertEqual(
            portfolio["status"],
            "PASS_CURRENT_SOURCE_SOLVER_PORTFOLIO_SCREENING",
        )
        self.assertEqual(
            {study["study_id"] for study in portfolio["studies"]},
            {"leapfrog_binary", "whfast_weak_three_body", "hybrid_close_scatter"},
        )
        self.assertFalse(portfolio["performance_superiority_claimed"])
        self.assertFalse(portfolio["cross_study_ranking_authorized"])
        self.assertFalse(portfolio["timings_comparable"])
        long_race = manifest["verification"]["prospective_long_timing_race"]
        self.assertEqual(
            long_race["status"], "PASS_PROSPECTIVE_TIMING_SCREENING_ONLY"
        )
        self.assertEqual(long_race["timed_repetitions"], 3)
        self.assertTrue(long_race["protocol"]["v1_stopped_before_measured_execution"])
        self.assertTrue(
            long_race["protocol"]["timing_sample_locked_before_measured_execution"]
        )
        self.assertTrue(
            long_race["timing_contract"]["jx_public_mandatory_semantic_replay_included"]
        )
        self.assertTrue(
            all(
                "EXTERNAL_FASTER" in value
                for value in (
                    long_race["studies"]["solar_system_100_year"]["classification"],
                    long_race["studies"]["smooth_hierarchy_100_period"]["classification"],
                    long_race["studies"]["close_scatter"]["mercurius_classification"],
                    long_race["studies"]["close_scatter"]["trace_classification"],
                )
            )
        )
        self.assertFalse(long_race["portable_performance_claimed"])
        self.assertFalse(long_race["general_solver_superiority_claimed"])
        self.assertFalse(manifest["release_actions"]["publication_performed"])

    def test_rc16_manifest_binds_public_lunar_component_and_claim_ceiling(self) -> None:
        manifest = json.loads(
            (ROOT / "RELEASE_MANIFEST_v0.6.0rc16.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["package_version"], "0.6.0rc16")
        self.assertEqual(manifest["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            manifest["status"], "LOCAL_RELEASE_CANDIDATE_PACKAGED_NOT_PUBLISHED"
        )
        # Rc16 is immutable history. Successor development changes the active
        # package, catalog, README, matrix, and this test, so bind the exact
        # recorded manifest rather than requiring historical bytes to equal
        # the live tree.
        self.assertEqual(
            manifest["source_bindings"],
            {
                "engine_api_sha256": "54bc4e3ca7224b225b3885deaaf8f86a104467376a3a8d7ac13b3c8e7cc830eb",
                "engine_catalog_sha256": "91ef9d6a1691ecded6b98400701a7dd250fafa85d95db88afc1cdd4e1312a7a0",
                "holdout_runner_sha256": "7f058dd7bc6a28db5b23a6be03a9f2521df190090fbb446afbbe89984926ff23",
                "holdout_test_sha256": "b8a6cfb918c671830149c13ef2fdd97c6799322dff8e6fa736491df331c760e5",
                "lunar_ephemeris_contracts_sha256": "ab03eff40ca0a0ee89d5803d74282a30a0a05b706b16369ba4d1a3a7b0b840ed",
                "lunar_ephemeris_sha256": "efede8495675fc11a264a1b9e7ff90201892318e7e26bc7223a1a7834332f202",
                "lunar_ephemeris_test_sha256": "a07cad3d4e44634df65b313fee7fed0340d611f9f30485a8f5e0e2d7d65b6cf0",
                "package_init_sha256": "1063f1915ca864b07cf1f9b15d15542eb9f6dc67f0b192b1372c42e736e02a63",
                "pyproject_sha256": "d8193e41f5c72dd7cd8b7af60ca087a590341762bd5f283f2306d933a062ab9d",
                "readme_sha256": "23f9c03ebb1b7b3d7ba537c672bc6b3eee9d9f5853b39fd9ec01080fd9f9b334",
                "reproducible_release_test_sha256": "b1151a713e54ab3a2cbd694a834a330af69b990385d09948c532171873b03101",
                "test_matrix_manifest_sha256": "22c5b1202d8bf563c248eb8bd852a9f9f45c1ccc4cddca930ce03eee19eb4643",
            },
        )
        component = manifest["public_lunar_ephemeris_v1"]
        self.assertEqual(
            component["model_id"],
            "solar-system.screening.resolved-eleven-eih1pn-coupled-lunar.v1",
        )
        self.assertEqual(component["accepted_steps"], 11_680)
        self.assertEqual(component["force_evaluations"], 151_840)
        self.assertEqual(component["holdout_decision"], "PASS_SCREENING_ONLY")
        self.assertFalse(component["independent_physical_validation"])
        self.assertFalse(component["production_ephemeris_qualified"])
        packaging = manifest["reproducible_packaging"]
        self.assertTrue(packaging["wheel_byte_identical"])
        self.assertTrue(packaging["sdist_byte_identical"])
        self.assertTrue(packaging["fresh_wheel_import_passed"])
        self.assertTrue(packaging["fresh_sdist_import_passed"])
        self.assertFalse(manifest["claim_controls"]["production_authorized"])
        self.assertFalse(
            manifest["claim_controls"]["general_rebound_superiority_claimed"]
        )
        self.assertFalse(manifest["release_actions"]["commit_performed"])
        self.assertFalse(manifest["release_actions"]["push_performed"])
        self.assertFalse(manifest["release_actions"]["publication_performed"])


if __name__ == "__main__":
    unittest.main()
