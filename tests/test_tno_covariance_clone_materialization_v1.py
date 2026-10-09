from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_covariance_clone_materialization_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_covariance_clone_materialization_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load covariance clone materializer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLONES = _load_module()


class TnoCovarianceCloneMaterializationV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = CLONES.build_protocol()
        cls.artifact = CLONES.build_clones(cls.protocol)

    def test_protocol_is_exact_input_only_and_preregisters_128_draws(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], CLONES.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["classification"], CLONES.CLASSIFICATION)
        self.assertEqual(protocol["sampler"]["clones_per_object"], 128)
        self.assertEqual(protocol["sampler"]["total_objects"], 12)
        self.assertEqual(protocol["sampler"]["total_clones"], 1536)
        self.assertEqual(
            protocol["sampler"]["object_seed_rule"],
            "SHA256(ASCII('jx.tno.covariance-clones.object-seed.v1') || "
            "NUL || ASCII_HEX64(protocol_content_sha256) || NUL || "
            "ASCII(object_id))",
        )
        self.assertIs(protocol["sampler"]["no_redraw"], True)
        self.assertIs(protocol["sampler"]["no_clipping"], True)
        self.assertIs(protocol["sampler"]["selection_cuts_reapplied_to_clones"], False)
        self.assertIs(protocol["execution_authorized"], False)
        self.assertIs(protocol["claims"]["science_result"], False)
        self.assertEqual(
            CLONES.validate_protocol(protocol), protocol["content_sha256"]
        )

    def test_materialization_is_deterministic_correlated_and_physical(self) -> None:
        first = self.artifact
        second = CLONES.build_clones(self.protocol)
        self.assertEqual(first, second)
        self.assertEqual(first["object_count"], 12)
        self.assertEqual(first["total_clone_count"], 1536)
        self.assertEqual(len(first["binary_payload_sha256"]), 64)
        self.assertEqual(
            first["sampling_outcome"],
            {
                "redraw_count": 0,
                "clipping_count": 0,
                "selection_rejection_count": 0,
                "nonphysical_sample_count": 0,
                "duplicate_vector_count": 0,
            },
        )
        for object_record in first["objects"]:
            self.assertEqual(object_record["clone_count"], 128)
            self.assertEqual(len(object_record["clones"]), 128)
            self.assertEqual(
                [row["clone_index"] for row in object_record["clones"]],
                list(range(128)),
            )
            lower = object_record["cholesky_lower_binary64_hex"]
            self.assertTrue(
                any(
                    float.fromhex(lower[row][column]) != 0.0
                    for row in range(1, 6)
                    for column in range(row)
                )
            )
            for clone in object_record["clones"]:
                values = tuple(float.fromhex(value) for value in clone["native_elements_hex"])
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertGreater(values[0], 0.0)
                self.assertLess(values[0], 1.0)
                self.assertGreater(values[1], 0.0)
                self.assertGreaterEqual(values[5], 0.0)
                self.assertLessEqual(values[5], 180.0)

    def test_hash_open_uniform_and_box_muller_are_strict_and_repeatable(self) -> None:
        seed = bytes(range(32))
        uniforms = [CLONES._open_uniform(seed, index, slot) for index in range(128) for slot in range(6)]
        self.assertTrue(all(0.0 < value < 1.0 for value in uniforms))
        first = CLONES._standard_normals(seed, 7)
        second = CLONES._standard_normals(seed, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(math.isfinite(value) for value in first))
        self.assertNotEqual(first, CLONES._standard_normals(seed, 8))

    def test_acquisition_loader_executes_pinned_source_bytes_not_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fake_acquisition.py"
            safe = b"SENTINEL = 'PINNED_SOURCE'".ljust(96, b" ") + b"\n"
            malicious = b"SENTINEL = 'UNPINNED_PYC'".ljust(96, b" ") + b"\n"
            self.assertEqual(len(safe), len(malicious))
            source.write_bytes(malicious)
            timestamp_ns = 1_700_000_000_000_000_000
            os.utime(source, ns=(timestamp_ns, timestamp_ns))
            optimization = "" if sys.flags.optimize == 0 else str(sys.flags.optimize)
            cache = Path(importlib.util.cache_from_source(str(source), optimization=optimization))
            cache.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(
                str(source),
                cfile=str(cache),
                doraise=True,
                optimize=sys.flags.optimize,
            )
            source.write_bytes(safe)
            os.utime(source, ns=(timestamp_ns, timestamp_ns))
            with mock.patch.object(CLONES, "ACQUISITION_TOOL", source):
                with mock.patch.object(
                    CLONES,
                    "ACQUISITION_TOOL_SHA256",
                    hashlib.sha256(safe).hexdigest(),
                ):
                    loaded = CLONES._load_acquisition_module()
            self.assertEqual(loaded.SENTINEL, "PINNED_SOURCE")

    def test_create_only_output_validates_and_rehashed_tamper_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            artifact = CLONES.materialize(output)
            self.assertEqual(
                CLONES.validate_output(output)["content_sha256"],
                artifact["content_sha256"],
            )
            with self.assertRaisesRegex(
                CLONES.CloneMaterializationError, "already exists"
            ):
                CLONES.materialize(output)
            path = output / "CLONES.json"
            forged = json.loads(path.read_bytes())
            forged["claims"]["science_result"] = True
            forged["content_sha256"] = CLONES._content_sha256(
                CLONES.CLONES_DOMAIN, forged
            )
            path.write_bytes(CLONES._canonical_json(forged))
            with self.assertRaisesRegex(
                CLONES.CloneMaterializationError, "reconstruction"
            ):
                CLONES.validate_output(output)

    def test_input_tamper_and_symlink_fail_closed(self) -> None:
        raw = CLONES.INPUT_ROOT / "raw" / "20090377.json"
        with self.assertRaisesRegex(
            CLONES.CloneMaterializationError, "differs from sealed custody"
        ):
            CLONES._provider_mean_matrix(raw, "0" * 64)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "input"
            shutil.copytree(CLONES.INPUT_ROOT, copied)
            raw = copied / "raw" / "20090377.json"
            original = raw.read_bytes()
            raw.write_bytes(original.replace(b'"2458411.5"', b'"2458412.5"'))
            with self.assertRaises(
                (
                    CLONES.CloneMaterializationError,
                    CLONES.ACQUISITION.CovarianceAcquisitionError,
                )
            ):
                CLONES.build_protocol(copied)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "input"
            shutil.copytree(CLONES.INPUT_ROOT, copied)
            raw = copied / "raw" / "20090377.json"
            payload = raw.read_bytes()
            raw.unlink()
            source = Path(directory) / "source.json"
            source.write_bytes(payload)
            raw.symlink_to(source)
            with self.assertRaises((CLONES.CloneMaterializationError, CLONES.ACQUISITION.CovarianceAcquisitionError)):
                CLONES.build_protocol(copied)

    def test_optimized_execution_builds_same_content_digest(self) -> None:
        script = (
            "import importlib.util,sys;"
            f"p={str(SCRIPT_PATH)!r};"
            "s=importlib.util.spec_from_file_location('jx_clone_oo',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "q=m.build_protocol();a=m.build_clones(q);print(q['content_sha256']);print(a['content_sha256'])"
        )
        completed = subprocess.run(
            [sys.executable, "-OO", "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip().splitlines(),
            [self.protocol["content_sha256"], self.artifact["content_sha256"]],
        )


if __name__ == "__main__":
    unittest.main()
