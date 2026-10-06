"""Direct contract and isolated-runtime tests for the public SPICE ephemeris."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from jxplanetx import ephemeris as module
from jxplanetx.ephemeris import (
    EphemerisContractError,
    EphemerisCoverageError,
    EphemerisDependencyError,
    EphemerisEpoch,
    EphemerisExecutionError,
    EphemerisKernelError,
    EphemerisRequest,
    EphemerisState,
    JplEphemeris,
    KernelArtifact,
    KernelBundle,
    KernelIdentity,
)


def _contract_fixture() -> tuple[KernelArtifact, KernelBundle, EphemerisRequest, EphemerisState]:
    artifact = KernelArtifact(
        logical_name="fixture.bsp",
        kind="SPK",
        path="/kernels/fixture.bsp",
        byte_length=4096,
        sha256="a" * 64,
    )
    bundle = KernelBundle(
        bundle_id="fixture-bundle",
        ephemeris_name="fixture-ephemeris",
        artifacts=(artifact,),
    )
    request = EphemerisRequest(
        request_id="fixture-request",
        target=1,
        observer=0,
        epoch=EphemerisEpoch.tdb_seconds(5.0),
    )
    state = EphemerisState(
        request_id=request.request_id,
        request_sha256=request.content_sha256,
        bundle_id=bundle.bundle_id,
        bundle_sha256=bundle.content_sha256,
        ephemeris_name=bundle.ephemeris_name,
        target=request.target,
        target_naif_id=1,
        observer=request.observer,
        observer_naif_id=0,
        input_epoch=request.epoch,
        ephemeris_time_seconds=5.0,
        frame=request.frame,
        aberration_correction=request.aberration_correction,
        position_km=(1.0, 2.0, 3.0),
        velocity_km_s=(0.1, 0.2, 0.3),
        light_time_seconds=0.01,
        kernels=(
            KernelIdentity(
                logical_name=artifact.logical_name,
                kind=artifact.kind,
                byte_length=artifact.byte_length,
                sha256=artifact.sha256,
            ),
        ),
        worker_sha256="b" * 64,
        worker_byte_length=8192,
        python_implementation="CPython",
        python_version="3.12.0",
        python_cache_tag="cpython-312",
        python_executable_path="/runtime/python",
        python_executable_byte_length=65536,
        python_executable_sha256="c" * 64,
        spiceypy_version="8.2.0",
        spiceypy_module_path="/runtime/spiceypy.py",
        spiceypy_module_byte_length=131072,
        spiceypy_module_sha256="d" * 64,
        cspice_version="CSPICE_N0067",
        cspice_library_path="/runtime/libcspice.so",
        cspice_library_byte_length=262144,
        cspice_library_sha256="e" * 64,
    )
    return artifact, bundle, request, state


class EphemerisContractTests(unittest.TestCase):
    def test_signed_int32_minimum_is_accepted_and_bounds_are_exact(self) -> None:
        request = EphemerisRequest(
            request_id="signed-minimum",
            target=-(1 << 31),
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(0.0),
        )
        self.assertEqual(request.target, -(1 << 31))
        for invalid in (-(1 << 31) - 1, 1 << 31, True):
            with self.subTest(invalid=invalid), self.assertRaises(EphemerisContractError):
                EphemerisRequest(
                    request_id="out-of-range",
                    target=invalid,
                    observer=0,
                    epoch=EphemerisEpoch.tdb_seconds(0.0),
                )

    def test_kernel_identity_is_exact_and_self_validating(self) -> None:
        valid = KernelIdentity("de441.bsp", "SPK", 1, "0" * 64)
        self.assertEqual(valid.logical_name, "de441.bsp")
        invalid_values = (
            {"logical_name": "../de441.bsp"},
            {"kind": "BSP"},
            {"byte_length": True},
            {"byte_length": 0},
            {"sha256": "A" * 64},
        )
        for replacement in invalid_values:
            values = {
                "logical_name": "de441.bsp",
                "kind": "SPK",
                "byte_length": 1,
                "sha256": "0" * 64,
            }
            values.update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(EphemerisContractError):
                KernelIdentity(**values)

    def test_state_rejects_forged_request_bundle_and_classification(self) -> None:
        _, _, _, state = _contract_fixture()
        for changes in (
            {"request_id": "forged-request"},
            {"request_sha256": "0" * 64},
            {"bundle_id": "forged-bundle"},
            {"bundle_sha256": "0" * 64},
            {"source_kind": "NATIVE_JX_EPHEMERIS"},
            {"scientific_status": "VALIDATED"},
            {"coverage_status": "EXTRAPOLATED"},
            {"integrity_class": "AUTHENTICATED"},
        ):
            with self.subTest(changes=changes), self.assertRaises(EphemerisContractError):
                dataclasses.replace(state, **changes)

    def test_state_rejects_nonexact_or_inconsistent_values(self) -> None:
        _, _, _, state = _contract_fixture()
        invalid = (
            ({"position_km": [1.0, 2.0, 3.0]}, EphemerisExecutionError),
            ({"velocity_km_s": (0.1, 0.2, float("nan"))}, EphemerisExecutionError),
            ({"light_time_seconds": -0.1}, EphemerisExecutionError),
            ({"target_naif_id": True}, EphemerisContractError),
            ({"observer_naif_id": 1}, EphemerisContractError),
            ({"ephemeris_time_seconds": 6.0}, EphemerisExecutionError),
            ({"kernels": list(state.kernels)}, EphemerisContractError),
            ({"python_executable_path": "relative/python"}, EphemerisContractError),
            ({"python_executable_byte_length": True}, EphemerisContractError),
            ({"cspice_library_sha256": "E" * 64}, EphemerisContractError),
        )
        for changes, error in invalid:
            with self.subTest(changes=changes), self.assertRaises(error):
                dataclasses.replace(state, **changes)

    def test_state_digest_binds_state_and_every_runtime_identity_group(self) -> None:
        _, _, _, state = _contract_fixture()
        changes = (
            {"position_km": (1.5, 2.0, 3.0)},
            {"velocity_km_s": (0.2, 0.2, 0.3)},
            {"light_time_seconds": 0.02},
            {"worker_sha256": "1" * 64},
            {"worker_byte_length": 8193},
            {"python_implementation": "OtherPython"},
            {"python_version": "3.12.1"},
            {"python_cache_tag": "cpython-312-alt"},
            {"python_executable_path": "/runtime/python-alt"},
            {"python_executable_byte_length": 65537},
            {"python_executable_sha256": "2" * 64},
            {"spiceypy_version": "8.2.1"},
            {"spiceypy_module_path": "/runtime/spiceypy-alt.py"},
            {"spiceypy_module_byte_length": 131073},
            {"spiceypy_module_sha256": "3" * 64},
            {"cspice_version": "CSPICE_N0068"},
            {"cspice_library_path": "/runtime/libcspice-alt.so"},
            {"cspice_library_byte_length": 262145},
            {"cspice_library_sha256": "4" * 64},
        )
        for replacement in changes:
            with self.subTest(replacement=replacement):
                changed = dataclasses.replace(state, **replacement)
                self.assertNotEqual(changed.content_sha256, state.content_sha256)

    def test_state_digest_binds_coherent_request_and_bundle_identities(self) -> None:
        artifact, bundle, request, state = _contract_fixture()
        changed_request = EphemerisRequest(
            request_id="second-request",
            target=request.target,
            observer=request.observer,
            epoch=request.epoch,
            frame=request.frame,
            aberration_correction=request.aberration_correction,
        )
        request_state = dataclasses.replace(
            state,
            request_id=changed_request.request_id,
            request_sha256=changed_request.content_sha256,
        )
        self.assertNotEqual(request_state.content_sha256, state.content_sha256)

        changed_bundle = KernelBundle(
            bundle_id="second-bundle",
            ephemeris_name=bundle.ephemeris_name,
            artifacts=(artifact,),
        )
        bundle_state = dataclasses.replace(
            state,
            bundle_id=changed_bundle.bundle_id,
            bundle_sha256=changed_bundle.content_sha256,
        )
        self.assertNotEqual(bundle_state.content_sha256, state.content_sha256)

    @unittest.skipUnless(sys.platform == "linux", "sealed memfd custody is Linux-only")
    def test_worker_source_copy_is_kernel_sealed(self) -> None:
        source = b"print('sealed')\n"
        descriptor = module._sealed_worker_fd(source)
        try:
            length, digest, _ = module._hash_fd(descriptor)
            self.assertEqual(length, len(source))
            self.assertEqual(digest, hashlib.sha256(source).hexdigest())
            with self.assertRaises(OSError) as caught:
                os.write(descriptor, b"tamper")
            self.assertEqual(caught.exception.errno, errno.EPERM)
        finally:
            os.close(descriptor)

    def test_worker_output_is_bounded_without_pipe_buffering(self) -> None:
        command = (
            "import os; "
            f"os.write(1, b'x' * ({module._MAX_WORKER_OUTPUT} + 1))"
        )
        with self.assertRaisesRegex(EphemerisExecutionError, "output cap"):
            module._run_bounded_worker(
                argv=[sys.executable, "-I", "-c", command],
                executable=sys.executable,
                encoded_request=b"",
                pass_fds=(),
                timeout_seconds=5.0,
            )

    def test_calendar_epoch_requires_an_explicit_lsk(self) -> None:
        _, bundle, _, _ = _contract_fixture()
        evaluator = JplEphemeris(bundle)
        request = EphemerisRequest(
            request_id="calendar-without-lsk",
            target=1,
            observer=0,
            epoch=EphemerisEpoch.calendar("2000-01-01T12:00:00", time_scale="UTC"),
        )
        with self.assertRaisesRegex(EphemerisKernelError, "explicitly bound LSK"):
            evaluator.state(request)


@unittest.skipUnless(
    sys.platform == "linux" and importlib.util.find_spec("spiceypy") is not None,
    "the optional SpiceyPy runtime is unavailable",
)
class EphemerisTemporarySpkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import spiceypy as spice

        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.kernel_path = cls.root / "linear-test.bsp"
        handle = spice.spkopn(str(cls.kernel_path), "JX EPHEMERIS TEST", 0)
        try:
            spice.spkw08(
                handle,
                1_000_001,
                0,
                "J2000",
                0.0,
                10.0,
                "JX LINEAR TEST SEGMENT",
                1,
                2,
                (
                    (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
                    (11.0, 22.0, 33.0, 0.1, 0.2, 0.3),
                ),
                0.0,
                10.0,
            )
        finally:
            spice.spkcls(handle)
        cls.kernel_bytes = cls.kernel_path.read_bytes()
        cls.kernel_sha256 = hashlib.sha256(cls.kernel_bytes).hexdigest()
        cls.artifact = KernelArtifact(
            logical_name="linear-test.bsp",
            kind="SPK",
            path=str(cls.kernel_path),
            byte_length=len(cls.kernel_bytes),
            sha256=cls.kernel_sha256,
        )
        cls.bundle = KernelBundle(
            bundle_id="temporary-linear-spk",
            ephemeris_name="temporary-test-only",
            artifacts=(cls.artifact,),
        )
        cls.evaluator = JplEphemeris(cls.bundle, timeout_seconds=10.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_exact_midpoint_and_complete_runtime_identity(self) -> None:
        request = EphemerisRequest(
            request_id="midpoint",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(5.0),
        )
        real_runner = module._run_bounded_worker

        def observe_custody(**kwargs: object) -> tuple[int, bytes, bytes]:
            argv = kwargs["argv"]
            executable = kwargs["executable"]
            self.assertTrue(argv[-1].startswith("/proc/self/fd/"))
            self.assertTrue(executable.startswith("/proc/self/fd/"))
            self.assertNotIn(str(Path(module.__file__).with_name("_ephemeris_worker.py")), argv)
            worker_fd = int(argv[-1].rsplit("/", 1)[1])
            with self.assertRaises(OSError) as caught:
                os.write(worker_fd, b"tamper")
            self.assertEqual(caught.exception.errno, errno.EPERM)
            return real_runner(**kwargs)

        with mock.patch.object(module, "_run_bounded_worker", side_effect=observe_custody):
            state = self.evaluator.state(request)
        self.assertEqual(state.position_km, (6.0, 12.0, 18.0))
        self.assertEqual(state.velocity_km_s, (0.1, 0.2, 0.3))
        self.assertEqual(state.target_naif_id, 1_000_001)
        self.assertEqual(state.observer_naif_id, 0)
        self.assertEqual(state.kernels[0].sha256, self.kernel_sha256)
        self.assertEqual(state.spiceypy_version, "8.2.0")
        self.assertTrue(state.cspice_version.startswith("CSPICE_"))
        self.assertEqual(
            (state.worker_byte_length, state.worker_sha256),
            (
                Path(module.__file__).with_name("_ephemeris_worker.py").stat().st_size,
                hashlib.sha256(
                    Path(module.__file__).with_name("_ephemeris_worker.py").read_bytes()
                ).hexdigest(),
            ),
        )
        for path, length, digest in (
            (
                state.python_executable_path,
                state.python_executable_byte_length,
                state.python_executable_sha256,
            ),
            (
                state.spiceypy_module_path,
                state.spiceypy_module_byte_length,
                state.spiceypy_module_sha256,
            ),
            (
                state.cspice_library_path,
                state.cspice_library_byte_length,
                state.cspice_library_sha256,
            ),
        ):
            payload = Path(path).read_bytes()
            self.assertEqual(length, len(payload))
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_out_of_coverage_epoch_fails_closed(self) -> None:
        request = EphemerisRequest(
            request_id="outside-coverage",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(11.0),
        )
        with self.assertRaises(EphemerisCoverageError):
            self.evaluator.state(request)

    def test_kernel_content_tamper_fails_before_query(self) -> None:
        tampered_path = self.root / "tampered.bsp"
        shutil.copyfile(self.kernel_path, tampered_path)
        with tampered_path.open("ab") as stream:
            stream.write(b"tamper")
        artifact = KernelArtifact(
            logical_name="tampered.bsp",
            kind="SPK",
            path=str(tampered_path),
            byte_length=len(self.kernel_bytes),
            sha256=self.kernel_sha256,
        )
        evaluator = JplEphemeris(
            KernelBundle("tampered-bundle", "temporary-test-only", (artifact,))
        )
        request = EphemerisRequest(
            request_id="tampered-kernel",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(5.0),
        )
        with self.assertRaisesRegex(EphemerisKernelError, "byte identity"):
            evaluator.state(request)

    def test_kernel_symlink_is_rejected(self) -> None:
        symlink_path = self.root / "linked.bsp"
        symlink_path.symlink_to(self.kernel_path)
        artifact = KernelArtifact(
            logical_name="linked.bsp",
            kind="SPK",
            path=str(symlink_path),
            byte_length=len(self.kernel_bytes),
            sha256=self.kernel_sha256,
        )
        evaluator = JplEphemeris(
            KernelBundle("symlink-bundle", "temporary-test-only", (artifact,))
        )
        request = EphemerisRequest(
            request_id="symlink-kernel",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(5.0),
        )
        with self.assertRaisesRegex(EphemerisKernelError, "securely open"):
            evaluator.state(request)

    def test_invalid_worker_response_is_rejected(self) -> None:
        request = EphemerisRequest(
            request_id="bad-response",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(5.0),
        )
        with mock.patch.object(module, "_run_bounded_worker", return_value=(0, b"{}", b"")):
            with self.assertRaisesRegex(EphemerisExecutionError, "wrong schema"):
                self.evaluator.state(request)

    def test_missing_dependency_is_explicit(self) -> None:
        request = EphemerisRequest(
            request_id="missing-dependency",
            target=1_000_001,
            observer=0,
            epoch=EphemerisEpoch.tdb_seconds(5.0),
        )
        with mock.patch.object(module.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(EphemerisDependencyError):
                self.evaluator.state(request)


if __name__ == "__main__":
    unittest.main()
