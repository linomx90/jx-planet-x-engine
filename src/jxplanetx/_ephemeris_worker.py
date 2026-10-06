"""Private, isolated CSPICE worker for :mod:`jxplanetx.ephemeris`."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import resource
import signal
import stat
import sys


_REQUEST_SCHEMA = "jx.ephemeris-worker-request.v2"
_RESPONSE_SCHEMA = "jx.ephemeris-worker-response.v2"
_RUNTIME_SCHEMA = "jx.ephemeris-runtime.v1"
_MAX_REQUEST_BYTES = 1 << 20
_MAX_OUTPUT_BYTES = 1 << 20
_MAX_WORKER_BYTES = 1 << 20
_MAX_RUNTIME_FILE_BYTES = 1 << 31
_MAX_KERNELS = 64


def _emit(value: dict[str, object], return_code: int) -> int:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("ascii")) > _MAX_OUTPUT_BYTES:
        encoded = json.dumps(
            {
                "category": "EXECUTION",
                "message": "CSPICE worker response exceeded its output cap",
                "ok": False,
                "schema": _RESPONSE_SCHEMA,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return_code = 2
    sys.stdout.write(encoded)
    return return_code


def _failure(category: str, message: str) -> int:
    bounded = " ".join(message.split())[:2048] or "CSPICE worker failed"
    return _emit(
        {"category": category, "message": bounded, "ok": False, "schema": _RESPONSE_SCHEMA},
        2,
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fd_identity(
    descriptor: int,
    *,
    maximum: int,
) -> tuple[int, str, tuple[int, int, int, int, int]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
        raise ValueError("runtime descriptor is not a bounded regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if offset != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != identity:
        raise ValueError("runtime descriptor changed while it was hashed")
    return offset, digest.hexdigest(), identity


def _file_identity(path_value: object) -> tuple[dict[str, object], tuple[int, int, int, int, int]]:
    if type(path_value) is not str or not path_value:
        raise ValueError("runtime file path is unavailable")
    path = os.path.realpath(path_value)
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError("runtime file path is not absolute and normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        length, digest, identity = _fd_identity(
            descriptor,
            maximum=_MAX_RUNTIME_FILE_BYTES,
        )
    finally:
        os.close(descriptor)
    return {"byte_length": length, "path": path, "sha256": digest}, identity


def _body_code(spice: object, value: str) -> int:
    code = spice.bods2c(value)
    if type(code) is tuple:
        number, found = code
        if found is not True:
            raise ValueError(f"NAIF body {value!r} is unknown")
        code = number
    if type(code) is not int or not -(1 << 31) <= code < (1 << 31):
        raise ValueError(f"NAIF body {value!r} did not resolve to a signed 32-bit integer")
    return code


def _limit_output_files() -> None:
    # Parent stdout/stderr are regular temporary files. This makes their
    # maximum growth a kernel-enforced property rather than a post-hoc check.
    limit = _MAX_OUTPUT_BYTES + 1
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)


def main() -> int:
    try:
        _limit_output_files()
    except Exception as exc:
        return _failure("DEPENDENCY", f"could not enforce worker output cap: {exc}")

    try:
        encoded = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
        if len(encoded) > _MAX_REQUEST_BYTES:
            return _failure("REQUEST", "worker request exceeded its byte cap")
        body = json.loads(encoded.decode("ascii"), parse_constant=_reject_json_constant)
    except Exception as exc:
        return _failure("REQUEST", f"worker request JSON is invalid: {exc}")
    expected_body_keys = {
        "kernel_fds",
        "python_executable_byte_length",
        "python_executable_fd",
        "python_executable_path",
        "python_executable_sha256",
        "request",
        "schema",
        "worker_source_byte_length",
        "worker_source_fd",
        "worker_source_sha256",
    }
    if type(body) is not dict or set(body) != expected_body_keys or body.get("schema") != _REQUEST_SCHEMA:
        return _failure("REQUEST", "worker request schema or fields are invalid")
    descriptors = body["kernel_fds"]
    request = body["request"]
    worker_fd = body["worker_source_fd"]
    python_fd = body["python_executable_fd"]
    if (
        type(descriptors) is not list
        or not 0 < len(descriptors) <= _MAX_KERNELS
        or type(request) is not dict
        or any(type(value) is not int or value < 0 for value in descriptors)
        or len(set(descriptors)) != len(descriptors)
        or type(worker_fd) is not int
        or worker_fd < 0
        or type(python_fd) is not int
        or python_fd < 0
        or worker_fd == python_fd
        or worker_fd in descriptors
        or python_fd in descriptors
    ):
        return _failure("REQUEST", "worker descriptors or request are invalid")
    if (
        type(body["worker_source_byte_length"]) is not int
        or not 0 < body["worker_source_byte_length"] <= _MAX_WORKER_BYTES
        or not _sha256(body["worker_source_sha256"])
        or type(body["python_executable_byte_length"]) is not int
        or not 0 < body["python_executable_byte_length"] <= _MAX_RUNTIME_FILE_BYTES
        or not _sha256(body["python_executable_sha256"])
        or type(body["python_executable_path"]) is not str
    ):
        return _failure("REQUEST", "worker source or Python identity is invalid")

    try:
        worker_length, worker_sha256, worker_identity = _fd_identity(
            worker_fd,
            maximum=_MAX_WORKER_BYTES,
        )
        python_length, python_sha256, python_identity = _fd_identity(
            python_fd,
            maximum=_MAX_RUNTIME_FILE_BYTES,
        )
    except Exception as exc:
        return _failure("DEPENDENCY", f"runtime descriptor verification failed: {exc}")
    if (worker_length, worker_sha256) != (
        body["worker_source_byte_length"],
        body["worker_source_sha256"],
    ):
        return _failure("DEPENDENCY", "sealed worker source identity mismatch")
    if (python_length, python_sha256) != (
        body["python_executable_byte_length"],
        body["python_executable_sha256"],
    ):
        return _failure("DEPENDENCY", "executed Python binary identity mismatch")
    expected_python_path = body["python_executable_path"]
    if (
        not os.path.isabs(expected_python_path)
        or os.path.normpath(expected_python_path) != expected_python_path
        or os.path.realpath(sys.executable) != expected_python_path
    ):
        return _failure("DEPENDENCY", "executed Python path identity mismatch")

    try:
        import spiceypy as spice
        from spiceypy.utils import libspicehelper
        from spiceypy.utils.exceptions import SpiceyError

        spice_module = importlib.import_module("spiceypy.spiceypy")
        spiceypy_module, spiceypy_module_identity = _file_identity(spice_module.__file__)
        cspice_library, cspice_library_identity = _file_identity(libspicehelper.libspice_path)
        spiceypy_version = spice.__version__
        cspice_version = spice.tkvrsn("TOOLKIT")
        python_implementation = platform.python_implementation()
        python_version = platform.python_version()
        python_cache_tag = sys.implementation.cache_tag
        if any(
            type(value) is not str or not value
            for value in (
                spiceypy_version,
                cspice_version,
                python_implementation,
                python_version,
                python_cache_tag,
            )
        ):
            raise ValueError("runtime version identity is unavailable")
    except Exception as exc:
        return _failure("DEPENDENCY", f"spiceypy/CSPICE runtime identity failed: {exc}")

    runtime = {
        "cspice_library": cspice_library,
        "cspice_version": cspice_version,
        "python_cache_tag": python_cache_tag,
        "python_executable": {
            "byte_length": python_length,
            "path": expected_python_path,
            "sha256": python_sha256,
        },
        "python_implementation": python_implementation,
        "python_version": python_version,
        "schema": _RUNTIME_SCHEMA,
        "spiceypy_module": spiceypy_module,
        "spiceypy_version": spiceypy_version,
        "worker": {"byte_length": worker_length, "sha256": worker_sha256},
    }

    phase = "KERNEL"
    spice.kclear()
    try:
        for descriptor in descriptors:
            path = f"/proc/self/fd/{descriptor}"
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not os.path.isfile(path):
                return _failure("KERNEL", "passed kernel descriptor is unavailable")
            spice.furnsh(path)

        if set(request) != {"aberration_correction", "epoch", "frame", "observer", "target"}:
            return _failure("REQUEST", "state request fields are invalid")
        epoch = request["epoch"]
        if type(epoch) is not dict or set(epoch) != {"format", "time_scale", "value"}:
            return _failure("REQUEST", "worker epoch is invalid")
        phase = "TIME"
        if epoch["format"] == "SECONDS_PAST_J2000":
            et = epoch["value"]
            if type(et) is not float or not math.isfinite(et) or epoch["time_scale"] != "TDB":
                return _failure("REQUEST", "native TDB epoch is invalid")
        elif epoch["format"] == "CALENDAR":
            scale = {"UTC": "UTC", "TT": "TDT", "TDB": "TDB"}.get(epoch["time_scale"])
            value = epoch["value"]
            if scale is None or type(value) is not str:
                return _failure("REQUEST", "calendar epoch is invalid")
            et = float(spice.str2et(f"{value} {scale}"))
        else:
            return _failure("REQUEST", "epoch format is unsupported")

        phase = "QUERY"
        target = request["target"]
        observer = request["observer"]
        frame = request["frame"]
        correction = request["aberration_correction"]
        if any(type(value) is not str or not value for value in (target, observer, frame, correction)):
            return _failure("REQUEST", "state request values are invalid")
        state, light_time = spice.spkezr(target, et, frame, correction, observer)
        target_id = _body_code(spice, target)
        observer_id = _body_code(spice, observer)
        values = [float(component) for component in state]
        et = float(et)
        light_time = float(light_time)
        if (
            len(values) != 6
            or target_id == observer_id
            or any(not math.isfinite(value) for value in (*values, et, light_time))
            or light_time < 0.0
        ):
            return _failure("EXECUTION", "CSPICE returned an invalid state")

        worker_after = _fd_identity(worker_fd, maximum=_MAX_WORKER_BYTES)
        python_after = _fd_identity(python_fd, maximum=_MAX_RUNTIME_FILE_BYTES)
        spiceypy_after, spiceypy_after_identity = _file_identity(spice_module.__file__)
        cspice_after, cspice_after_identity = _file_identity(libspicehelper.libspice_path)
        if worker_after != (worker_length, worker_sha256, worker_identity):
            return _failure("DEPENDENCY", "sealed worker source changed during execution")
        if python_after != (python_length, python_sha256, python_identity):
            return _failure("DEPENDENCY", "Python binary changed during execution")
        if (
            spiceypy_after != spiceypy_module
            or spiceypy_after_identity != spiceypy_module_identity
            or cspice_after != cspice_library
            or cspice_after_identity != cspice_library_identity
        ):
            return _failure("DEPENDENCY", "SpiceyPy/CSPICE runtime changed during execution")

        return _emit(
            {
                "ephemeris_time_seconds": et,
                "light_time_seconds": light_time,
                "observer_naif_id": observer_id,
                "ok": True,
                "runtime": runtime,
                "schema": _RESPONSE_SCHEMA,
                "state": values,
                "target_naif_id": target_id,
            },
            0,
        )
    except SpiceyError as exc:
        text = str(exc)
        upper = text.upper()
        if "SPKINSUFFDATA" in upper or "NOSEGMENTSFOUND" in upper:
            category = "COVERAGE"
        elif phase == "KERNEL":
            category = "KERNEL"
        elif phase == "TIME":
            category = "TIME"
        else:
            category = "EXECUTION"
        return _failure(category, text)
    except Exception as exc:
        return _failure("EXECUTION" if phase == "QUERY" else phase, str(exc))
    finally:
        spice.kclear()


if __name__ == "__main__":
    raise SystemExit(main())
