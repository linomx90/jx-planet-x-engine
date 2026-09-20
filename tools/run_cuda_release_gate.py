#!/usr/bin/env python3
"""Run the fail-closed JX CUDA core release gate on a real NVIDIA device."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for entry in (str(ROOT), str(SOURCE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from jxplanetx import __version__  # noqa: E402


CUDA_TEST_TARGETS = (
    "tests.test_engine_backends.OptionalCuPyParityTests",
    "tests.test_engine_trajectory.OptionalCuPyTrajectoryTests",
)
EXPECTED_TEST_COUNT = 5


class CudaGateError(RuntimeError):
    """The required CUDA runtime, device, or exact test contract failed."""


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink():
        raise CudaGateError("output parent must not be a symlink")
    target = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def probe_cuda(device_index: int) -> tuple[object, dict[str, object]]:
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise CudaGateError(f"CuPy import failed: {exc}") from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise CudaGateError(f"CUDA device discovery failed: {exc}") from exc
    if device_count < 1:
        raise CudaGateError("CUDA reported no devices")
    if device_index < 0 or device_index >= device_count:
        raise CudaGateError(
            f"requested CUDA device {device_index} is outside 0..{device_count - 1}"
        )

    try:
        with cp.cuda.Device(device_index):
            properties = cp.cuda.runtime.getDeviceProperties(device_index)
            values = cp.arange(1, 1_000_001, dtype=cp.float64)
            observed = float(cp.sum(values * values).item())
            cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        raise CudaGateError(f"CUDA float64 allocation/kernel probe failed: {exc}") from exc
    if not math.isfinite(observed) or observed <= 0.0:
        raise CudaGateError("CUDA float64 kernel probe returned an invalid scalar")

    report = {
        "cupy_version": cp.__version__,
        "cupy_distribution": importlib.metadata.version("cupy-cuda13x"),
        "device_count": device_count,
        "device_index": device_index,
        "device_name": _text(properties["name"]),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "compute_capability": (
            f"{int(properties['major'])}.{int(properties['minor'])}"
        ),
        "total_global_memory_bytes": int(properties["totalGlobalMem"]),
        "float64_kernel_probe": "PASS",
    }
    return cp, report


def run_gate(device_index: int) -> tuple[int, dict[str, object]]:
    base: dict[str, object] = {
        "schema": "jxplanetx.cuda-core-release-gate.v1",
        "jxplanetx_version": __version__,
        "scientific_claim_state": "SCREENING_ONLY",
        "production_ready": False,
        "general_gpu_qualification": False,
        "test_targets": list(CUDA_TEST_TARGETS),
        "expected_test_count": EXPECTED_TEST_COUNT,
    }
    try:
        _cp, probe = probe_cuda(device_index)
    except CudaGateError as exc:
        base.update(
            {
                "status": "BLOCKED_RUNTIME_OR_DEVICE_UNAVAILABLE",
                "reason": str(exc),
                "tests_run": 0,
                "skips": 0,
            }
        )
        return 2, base

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for target in CUDA_TEST_TARGETS:
        suite.addTests(loader.loadTestsFromName(target))
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    skips = len(result.skipped)
    passed = (
        result.wasSuccessful()
        and result.testsRun == EXPECTED_TEST_COUNT
        and skips == 0
    )
    base.update(
        {
            "status": "PASS" if passed else "FAIL",
            "device": probe,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skips": skips,
        }
    )
    return (0 if passed else 1), base


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=0, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    status, report = run_gate(arguments.device)
    payload = canonical(report)
    write_new(arguments.output, payload)
    sys.stdout.buffer.write(payload)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
