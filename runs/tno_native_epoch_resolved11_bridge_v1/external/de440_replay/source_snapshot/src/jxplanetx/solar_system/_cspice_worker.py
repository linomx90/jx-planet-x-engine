"""Private isolated worker for the closed M4C2B CSPICE execution profile.

This file is executed as a held file descriptor by :mod:`cspice_execution`.
It is not an importable provider API.  The parent gives it a held private
wheel-extraction directory and a held SPK file descriptor.  The worker emits
only bounded primitive observations; the parent constructs every public
sealed contract.
"""

from __future__ import annotations

import base64
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import platform
import stat
import struct
import sys
from pathlib import PurePosixPath


_REQUEST_SCHEMA = "jxplanetx.cspice-worker.request.v1"
_RESULT_SCHEMA = "jxplanetx.cspice-worker.result.v1"
_REQUEST_TOKEN = "ONE_HELD_STAGE_ONE_HELD_SPK_DIRECT_SPKGEO_J2000_V1"
_MAXIMUM_REQUEST_BYTES = 65_536
_MAXIMUM_RESULT_BYTES = 2_097_152
_MAXIMUM_TARGETS = 64
_MAXIMUM_CHAIN_LENGTH = 16
_MAXIMUM_RECORD_ROWS = 4_096
_MAXIMUM_FILE_BYTES = 1 << 30
_MAXIMUM_DISTRIBUTION_MEMBER_BYTES = 1 << 27
_MAXIMUM_STAGE_ENTRIES = 2_048
_MAXIMUM_STAGE_DEPTH = 16
_MAXIMUM_STAGE_TOTAL_BYTES = 1 << 27
_MAXIMUM_MAPS_BYTES = 1 << 22
_MAXIMUM_MAPS_LINES = 65_536
_READ_CHUNK = 1 << 20
_SPICE_INT_MAX = (1 << 31) - 1
_CSPICE_SHA256 = (
    "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84"
)
_CPU_PAYLOAD_LENGTH = 1_086
_CPU_PAYLOAD_SHA256 = (
    "b050bd98f1d010d3f10c7d685e2eb6875793ddb9909110de8474aba96740fe74"
)

_PROFILES = {
    "cp312-cp312": {
        "python_version": "3.12.13",
        "spiceypy_record": "da050a7e1cb4abbea030a804090f40ca7ec3283a8b059478fcd5d4754a515d47",
        "spiceypy_tree": (
            29,
            7_197_075,
            2_950,
            "1a03ac00e4816b237c93d1d0643494f50f0d04a9df2a253cdf211b519d6bd4ab",
        ),
        "numpy_record": "ddd8a4670b2a83f7b3ca989ae0256137e5b75dd78b5faede406301db81fe9d79",
        "numpy_tree": (
            900,
            57_522_497,
            95_533,
            "c45b6e7c5c50537127e5f7cc7f861c6ecc032471a78de972ea1d02ac066dfaf7",
        ),
        "numpy_native_tree": (
            22,
            45_593_806,
            2_816,
            "c6e2662d4564486d186fa8dc066225f67b16e44bb55e8216ce8497a289d45c8d",
        ),
        "loaded_numpy_natives": (
            ("numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0", 2_833_617, "c6090048eccc763522c12ef016f81da6b627cb3a044f55cf0479a839c41c0980"),
            ("numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0", 250_985, "6ed5137f412781ad7863439fb543613f620b43c32b63292a0029246162f5bbc6"),
            ("numpy.libs/libscipy_openblas64_-fdde5778.so", 25_034_001, "ee2039cf13a45e1e89f04d0cea9585657d93ffaa425594e1935161013fdf4e12"),
            ("numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so", 10_808_937, "ba96467ae494babaf7c3458b09cc03c4200c2e10010860d80e452ffb8b01c8cd"),
            ("numpy/linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so", 231_833, "64bfb8bb8a1a09ca31f7a6e307a87b4c78a75f3489aaf3c61ed8aa4ffada9845"),
        ),
    },
    "cp314-cp314": {
        "python_version": "3.14.4",
        "spiceypy_record": "ae6ccb8132cec7e02cf020785df7e6c2b36c344eb215253757f8dcb100ee1002",
        "spiceypy_tree": (
            29,
            7_209_395,
            2_950,
            "f8f65e07e738b4beab2a277bac53ced6338ce9ed001f9c84930e1b54988700db",
        ),
        "numpy_record": "47521114855987c11cea7b07e2414bd40fafac4fe21921c389da6cbee9d82f4b",
        "numpy_tree": (
            900,
            57_505_893,
            95_533,
            "685ec34fdf369dc90dd4cc8719e36b24279a8adba0f0b118304406e1f04f8ee4",
        ),
        "numpy_native_tree": (
            22,
            45_577_206,
            2_816,
            "2c3a6d532868e754cba081d053fd3e3b18ad5cf4f08b2a692be2734ced194192",
        ),
        "loaded_numpy_natives": (
            ("numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0", 2_833_617, "c6090048eccc763522c12ef016f81da6b627cb3a044f55cf0479a839c41c0980"),
            ("numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0", 250_985, "6ed5137f412781ad7863439fb543613f620b43c32b63292a0029246162f5bbc6"),
            ("numpy.libs/libscipy_openblas64_-fdde5778.so", 25_034_001, "ee2039cf13a45e1e89f04d0cea9585657d93ffaa425594e1935161013fdf4e12"),
            ("numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so", 10_800_697, "61052ef771d926665ca135b1840662516f058d5c98843aa429ef2abdd7f46a04"),
            ("numpy/linalg/_umath_linalg.cpython-314-x86_64-linux-gnu.so", 231_833, "ce9f59d6d4adf07b7ad262400d87484af53a3c3bafbfcca10c25327cbc75f5e9"),
        ),
    },
}


class _WorkerError(RuntimeError):
    pass


class _CoverageError(_WorkerError):
    pass


class _DependencyError(_WorkerError):
    pass


def _require(condition: object, message: str) -> None:
    if not condition:
        raise _WorkerError(message)


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _WorkerError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _read_exact(stream: io.BufferedReader, length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        block = stream.read(length - len(result))
        if not block:
            raise _WorkerError("truncated request frame")
        result.extend(block)
    return bytes(result)


def _receive_request() -> tuple[dict[str, object], str]:
    header = _read_exact(sys.stdin.buffer, 4)
    length = struct.unpack(">I", header)[0]
    if length < 2 or length > _MAXIMUM_REQUEST_BYTES:
        raise _WorkerError("request frame length is outside its hard cap")
    encoded = _read_exact(sys.stdin.buffer, length)
    if sys.stdin.buffer.read(1) not in (b"", None):
        raise _WorkerError("request contains trailing bytes")
    try:
        value = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _WorkerError("request is not canonical ASCII JSON") from exc
    if _canonical_json(value) != encoded:
        raise _WorkerError("request JSON is not canonical")
    _require(type(value) is dict and set(value) == {"body", "sha256"}, "invalid request envelope")
    body = value["body"]
    digest = value["sha256"]
    _require(type(body) is dict, "request body must be an object")
    _require(type(digest) is str and len(digest) == 64, "invalid request digest")
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()
    _require(digest == expected and digest != "0" * 64, "request digest mismatch")
    expected_keys = {
        "abi_tag",
        "effective_et_be_hex",
        "nonce",
        "python_fd",
        "request_token",
        "role",
        "schema",
        "spk_fd",
        "stage_fd",
        "targets",
        "worker_fd",
        "worker_source_sha256",
    }
    _require(set(body) == expected_keys, "request body has an invalid key roster")
    _require(body["schema"] == _REQUEST_SCHEMA, "request schema mismatch")
    _require(body["request_token"] == _REQUEST_TOKEN, "request token mismatch")
    role = body["role"]
    _require(type(role) is str and role in ("PRIMARY", "REPLAY"), "invalid lane role")
    nonce = body["nonce"]
    _require(type(nonce) is str and len(nonce) == 64 and nonce != "0" * 64, "invalid nonce")
    _require(all(character in "0123456789abcdef" for character in nonce), "invalid nonce")
    abi = body["abi_tag"]
    _require(type(abi) is str and abi in _PROFILES, "unsupported ABI profile")
    targets = body["targets"]
    _require(type(targets) is list and 1 <= len(targets) <= _MAXIMUM_TARGETS, "invalid target roster")
    _require(all(type(item) is int and -(1 << 31) < item < (1 << 31) and item != 0 for item in targets), "invalid target ID")
    _require(len(set(targets)) == len(targets), "duplicate target ID")
    for name in ("stage_fd", "spk_fd", "worker_fd", "python_fd"):
        fd = body[name]
        _require(type(fd) is int and 0 <= fd <= _SPICE_INT_MAX, f"invalid {name}")
    for name in ("effective_et_be_hex", "worker_source_sha256"):
        item = body[name]
        expected_length = 16 if name.startswith("effective") else 64
        _require(type(item) is str and len(item) == expected_length, f"invalid {name}")
        _require(all(character in "0123456789abcdef" for character in item), f"invalid {name}")
    _require(body["worker_source_sha256"] != "0" * 64, "invalid worker source digest")
    return body, digest


def _fd_hash(fd: int, expected_length: int | None = None) -> tuple[int, str]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise _WorkerError("held descriptor is not a regular file")
    if info.st_size < 0 or info.st_size > _MAXIMUM_FILE_BYTES:
        raise _WorkerError("held file length is outside the cap")
    if expected_length is not None and info.st_size != expected_length:
        raise _WorkerError("held file length mismatch")
    digest = hashlib.sha256()
    offset = 0
    while offset < info.st_size:
        block = os.pread(fd, min(_READ_CHUNK, info.st_size - offset), offset)
        if not block:
            raise _WorkerError("short held-file read")
        digest.update(block)
        offset += len(block)
    if os.pread(fd, 1, info.st_size) != b"":
        raise _WorkerError("held file grew during observation")
    return info.st_size, digest.hexdigest()


def _f64_from_bits(value: object, label: str) -> float:
    _require(type(value) is str and len(value) == 16, f"invalid {label}")
    _require(all(character in "0123456789abcdef" for character in value), f"invalid {label}")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    _require(math.isfinite(result), f"nonfinite {label}")
    return result


def _f64_bits(value: object) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise _WorkerError("provider returned a nonfinite binary64 value")
    return struct.pack(">d", result).hex()


def _safe_relative(value: str) -> str:
    _require(type(value) is str and value and len(value) <= 1_024, "invalid relative path")
    _require("\x00" not in value and "\\" not in value and not value.startswith("/"), "invalid relative path")
    parts = value.split("/")
    _require(1 <= len(parts) <= 64 and all(part not in ("", ".", "..") and len(part) <= 256 for part in parts), "invalid relative path")
    _require(PurePosixPath(value).as_posix() == value, "nonnormalized relative path")
    return value


def _hash_path(path: str, expected_length: int | None = None) -> tuple[int, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        return _fd_hash(fd, expected_length)
    finally:
        os.close(fd)


def _record_rows(stage_path: str, distribution: str, profile: dict[str, object]) -> tuple[set[str], list[tuple[str, int, str]]]:
    if distribution == "spiceypy":
        record_relative = "spiceypy-8.2.0.dist-info/RECORD"
        expected_record = profile["spiceypy_record"]
    else:
        record_relative = "numpy-2.3.5.dist-info/RECORD"
        expected_record = profile["numpy_record"]
    record_path = os.path.join(stage_path, record_relative)
    record_fd = os.open(record_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        record_stat = os.fstat(record_fd)
        _require(stat.S_ISREG(record_stat.st_mode) and 0 < record_stat.st_size <= _MAXIMUM_REQUEST_BYTES * 2, "RECORD length/type is outside its cap")
        record_length, record_digest = _fd_hash(record_fd, record_stat.st_size)
        _require(record_digest == expected_record, "RECORD identity mismatch")
        encoded = bytearray()
        while len(encoded) < record_length:
            block = os.pread(record_fd, min(_READ_CHUNK, record_length - len(encoded)), len(encoded))
            _require(bool(block), "short RECORD read")
            encoded.extend(block)
        _require(os.pread(record_fd, 1, record_length) == b"", "RECORD grew during read")
    finally:
        os.close(record_fd)
    encoded = bytes(encoded)
    _require(hashlib.sha256(encoded).hexdigest() == record_digest, "parsed RECORD bytes differ from verified bytes")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkerError("RECORD is not strict UTF-8") from exc
    parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    _require(1 <= len(parsed) <= _MAXIMUM_RECORD_ROWS, "RECORD row count outside cap")
    members: set[str] = set()
    declared: list[tuple[str, str, int]] = []
    prospective_total = 0
    for row in parsed:
        _require(len(row) == 3, "RECORD row must have three columns")
        relative = _safe_relative(row[0])
        _require(relative not in members, "duplicate RECORD member")
        members.add(relative)
        if relative == record_relative:
            _require(row[1] == "" and row[2] == "", "RECORD self row must be unhashed")
            length = record_length
        else:
            _require(row[1].startswith("sha256=") and row[2].isdigit() and (row[2] == "0" or not row[2].startswith("0")), "invalid RECORD hash/size")
            length = int(row[2])
            _require(0 <= length <= _MAXIMUM_DISTRIBUTION_MEMBER_BYTES, "RECORD member exceeds its byte cap")
        prospective_total += length
        _require(prospective_total <= _MAXIMUM_STAGE_TOTAL_BYTES, "RECORD aggregate exceeds its byte cap")
        declared.append((relative, row[1], length))
    expected_tree = profile["spiceypy_tree"] if distribution == "spiceypy" else profile["numpy_tree"]
    _require(prospective_total == expected_tree[1], "RECORD prospective aggregate differs from calibrated profile")
    rows: list[tuple[str, int, str]] = []
    for relative, encoded_digest, length in declared:
        member_path = os.path.join(stage_path, relative)
        observed_length, digest = _hash_path(member_path, length)
        _require(observed_length == length, "RECORD member length mismatch")
        if relative != record_relative:
            expected = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
            _require(encoded_digest[7:] == expected, "RECORD member hash mismatch")
        rows.append((relative, length, digest))
    return members, rows


def _manifest(rows: list[tuple[str, int, str]]) -> tuple[int, int, int, str]:
    ordered = sorted(rows, key=lambda item: item[0].encode("utf-8"))
    preimage = b"".join(
        path.encode("utf-8") + b"\0" + str(length).encode("ascii") + b"\0" + digest.encode("ascii") + b"\n"
        for path, length, digest in ordered
    )
    return len(ordered), sum(item[1] for item in ordered), len(preimage), hashlib.sha256(preimage).hexdigest()


def _distribution_files(stage_path: str) -> tuple[set[str], set[str], set[str]]:
    spice_files: set[str] = set()
    numpy_files: set[str] = set()
    all_files: set[str] = set()
    entry_count = 0
    total_bytes = 0
    for root, directories, files in os.walk(stage_path, topdown=True, followlinks=False):
        relative_root = os.path.relpath(root, stage_path)
        depth = 0 if relative_root == "." else len(relative_root.split(os.sep))
        _require(depth <= _MAXIMUM_STAGE_DEPTH, "stage depth exceeds its cap")
        entry_count += len(directories) + len(files)
        _require(entry_count <= _MAXIMUM_STAGE_ENTRIES, "stage entry count exceeds its cap")
        directories.sort()
        files.sort()
        for name in tuple(directories) + tuple(files):
            full = os.path.join(root, name)
            info = os.lstat(full)
            _require(not stat.S_ISLNK(info.st_mode), "stage contains a symlink")
            _require(stat.S_ISDIR(info.st_mode) if name in directories else stat.S_ISREG(info.st_mode), "stage contains a special node")
        for name in files:
            relative = os.path.relpath(os.path.join(root, name), stage_path).replace(os.sep, "/")
            _safe_relative(relative)
            info = os.lstat(os.path.join(root, name))
            total_bytes += info.st_size
            _require(total_bytes <= _MAXIMUM_STAGE_TOTAL_BYTES, "stage bytes exceed their cap")
            all_files.add(relative)
            if relative.startswith("spiceypy/") or relative.startswith("spiceypy-8.2.0.dist-info/"):
                spice_files.add(relative)
            elif relative.startswith("numpy/") or relative.startswith("numpy.libs/") or relative.startswith("numpy-2.3.5.dist-info/"):
                numpy_files.add(relative)
    return spice_files, numpy_files, all_files


def _found_tuple(value: tuple[object, ...], expected_length: int) -> tuple[object, ...]:
    if len(value) == expected_length + 1:
        *items, found = value
        if not found:
            raise _CoverageError("SPK selector reported no applicable segment")
        return tuple(items)
    if len(value) != expected_length:
        raise _WorkerError("unexpected SpiceyPy found-result shape")
    return tuple(value)


def _pool_counts(spice: object) -> list[int]:
    return [int(spice.ktotal("ALL")), int(spice.ktotal("SPK"))]


def _selected_chain(
    spice: object,
    target: int,
    et: float,
    loaded_handle: int,
    coverage_error_types: tuple[type[Exception], ...],
) -> list[dict[str, object]]:
    body = target
    seen: set[int] = set()
    legs: list[dict[str, object]] = []
    while body != 0:
        if body in seen:
            raise _WorkerError("selected SPK chain contains a cycle")
        if len(legs) >= _MAXIMUM_CHAIN_LENGTH:
            raise _WorkerError("selected SPK chain exceeds its hard profile cap")
        seen.add(body)
        try:
            handle_raw, descriptor_raw, identifier_raw = _found_tuple(tuple(spice.spksfs(body, et, 41)), 3)
        except coverage_error_types as exc:
            raise _CoverageError("SPK target chain is unavailable") from exc
        handle = int(handle_raw)
        _require(handle == loaded_handle and 1 <= handle <= _SPICE_INT_MAX, "selector handle mismatch")
        descriptor_values = tuple(float(value) for value in descriptor_raw)
        _require(len(descriptor_values) == 5 and all(math.isfinite(value) for value in descriptor_values), "invalid SPK descriptor")
        decoded = tuple(spice.spkuds(descriptor_raw))
        _require(len(decoded) == 8, "SPKUDS returned an invalid shape")
        segment_body = int(decoded[0])
        center = int(decoded[1])
        frame = int(decoded[2])
        segment_type = int(decoded[3])
        first = float(decoded[4])
        last = float(decoded[5])
        begin = int(decoded[6])
        end = int(decoded[7])
        _require(segment_body == body and body != center, "SPK descriptor body/center mismatch")
        _require(frame == 1 and segment_type == 2, "SPK segment is outside frame-1/type-2 profile")
        _require(math.isfinite(first) and math.isfinite(last) and first <= et <= last, "SPK segment does not contain ET")
        _require(1 <= begin <= end <= _SPICE_INT_MAX, "SPK DAF address outside SpiceInt profile")
        returned_frame_raw, state_raw, returned_center_raw = spice.spkpvn(handle, descriptor_raw, et)
        returned_frame = int(returned_frame_raw)
        returned_center = int(returned_center_raw)
        state_values = tuple(float(value) for value in state_raw)
        _require(len(state_values) == 6 and all(math.isfinite(value) for value in state_values), "invalid SPKPVN state")
        _require(returned_frame == frame and returned_center == center, "SPKPVN readback mismatch")
        identifier = str(identifier_raw)
        _require(identifier and len(identifier) <= 40 and identifier.strip() == identifier, "invalid SPK segment identifier")
        _require(all(0x20 <= ord(character) <= 0x7E for character in identifier), "non-ASCII SPK segment identifier")
        legs.append(
            {
                "begin_daf_address": begin,
                "center_naif_id": center,
                "end_daf_address": end,
                "first_et_be_hex": _f64_bits(first),
                "frame_naif_id": frame,
                "last_et_be_hex": _f64_bits(last),
                "segment_identifier": identifier,
                "segment_type": segment_type,
                "selected_body_naif_id": body,
                "spkpvn_returned_center_naif_id": returned_center,
                "spkpvn_returned_frame_naif_id": returned_frame,
                "spkpvn_state_be_hex": [_f64_bits(value) for value in state_values],
            }
        )
        body = center
    _require(bool(legs), "selected SPK chain is empty")
    return legs


def _mapped_runtime_observation(
    stage_path: str,
    expected_paths: tuple[str, ...],
    all_stage_native_paths: tuple[str, ...],
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    maps_path = "/proc/self/maps"
    if not os.path.isfile(maps_path):
        raise _DependencyError("Linux procfs maps are unavailable")
    mapped: dict[tuple[int, int], str] = {}
    with open(maps_path, "rb", buffering=0) as stream:
        maps_buffer = bytearray()
        while True:
            block = stream.read(min(65_536, _MAXIMUM_MAPS_BYTES - len(maps_buffer) + 1))
            if not block:
                break
            maps_buffer.extend(block)
            _require(len(maps_buffer) <= _MAXIMUM_MAPS_BYTES, "proc maps exceeds its byte cap")
    maps_bytes = bytes(maps_buffer)
    try:
        map_lines = maps_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise _DependencyError("proc maps is not UTF-8") from exc
    _require(len(map_lines) <= _MAXIMUM_MAPS_LINES, "proc maps exceeds its line cap")
    for line in map_lines:
            fields = line.split()
            if not fields or not fields[-1].startswith("/"):
                continue
            path = os.path.realpath(fields[-1])
            try:
                info = os.stat(path)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                mapped[(info.st_dev, info.st_ino)] = path
    stage_native_by_inode: dict[tuple[int, int], str] = {}
    for relative in all_stage_native_paths:
        info = os.stat(os.path.join(stage_path, relative))
        stage_native_by_inode[(info.st_dev, info.st_ino)] = relative
    mapped_stage_paths = {
        relative
        for identity, relative in stage_native_by_inode.items()
        if identity in mapped
    }
    missing = set(expected_paths) - mapped_stage_paths
    if missing:
        raise _DependencyError("required distribution native maps are absent")
    _require(mapped_stage_paths == set(expected_paths), "mapped stage-native roster has extra members")
    distribution: list[tuple[str, int, str]] = []
    for relative in expected_paths:
        expected = os.path.join(stage_path, relative)
        info = os.stat(expected)
        mapped_path = mapped.get((info.st_dev, info.st_ino))
        if mapped_path is None:
            raise _DependencyError(f"expected native map absent: {relative}")
        length, digest = _hash_path(expected)
        distribution.append((relative, length, digest))
    host: list[tuple[str, int, str]] = []
    for basename in ("ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6"):
        candidates = sorted({path for path in mapped.values() if os.path.basename(path) == basename})
        if len(candidates) != 1:
            raise _DependencyError(f"host native mapping is absent or ambiguous: {basename}")
        length, digest = _hash_path(candidates[0])
        host.append((basename, length, digest))
    return distribution, host


def _finish_provider_observation(
    provider_failure: Exception | None,
    *,
    cleanup_valid: bool,
    fenv_after: int,
    spk_offset_before: int,
    spk_offset_after: int,
    spk_pre_stat: tuple[int, ...],
    spk_post_stat: tuple[int, ...],
    spk_length_before: int,
    spk_length_after: int,
    spk_sha_before: str,
    spk_sha_after: str,
) -> None:
    """Make local data/custody failures dominate a provider coverage failure."""

    _require(cleanup_valid, "CSPICE cleanup did not clear state")
    _require(fenv_after == 0, "worker rounding mode changed")
    _require(spk_offset_before == 0 and spk_offset_after == 0, "CSPICE changed held SPK descriptor offset")
    _require(spk_pre_stat == spk_post_stat, "held SPK metadata changed")
    _require(spk_length_after == spk_length_before and spk_sha_after == spk_sha_before, "held SPK bytes changed")
    if provider_failure is not None:
        raise provider_failure


def _execute(body: dict[str, object], request_sha256: str) -> dict[str, object]:
    abi = str(body["abi_tag"])
    profile = _PROFILES[abi]
    if platform.python_implementation() != "CPython" or platform.python_version() != profile["python_version"]:
        raise _DependencyError("worker version/ABI mismatch")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise _DependencyError("worker platform mismatch")
    if sys.byteorder != "little" or struct.calcsize("P") * 8 != 64 or float.__getformat__("double") != "IEEE, little-endian":
        raise _DependencyError("worker numerical architecture mismatch")
    flags = sys.flags
    if not (flags.isolated == 1 and flags.ignore_environment == 1 and flags.dont_write_bytecode == 1 and flags.no_site == 1):
        raise _DependencyError("worker was not invoked with exact -I -B -S isolation flags")
    allowed_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if dict(os.environ) != allowed_environment:
        raise _DependencyError("worker environment differs from the exact minimal roster")
    if len(sys.path) > 16 or any(not item or "site-packages" in item or "dist-packages" in item for item in sys.path):
        raise _DependencyError("worker pre-stage import path is not isolated")
    stage_fd = int(body["stage_fd"])
    spk_fd = int(body["spk_fd"])
    worker_fd = int(body["worker_fd"])
    python_fd = int(body["python_fd"])
    stage_info = os.fstat(stage_fd)
    _require(stat.S_ISDIR(stage_info.st_mode), "held stage descriptor is not a directory")
    _require(stage_info.st_uid == os.geteuid() and stat.S_IMODE(stage_info.st_mode) == 0o700, "held stage is not an euid-owned private directory")
    python_info = os.fstat(python_fd)
    actual_python_info = os.stat("/proc/self/exe")
    _require(stat.S_ISREG(python_info.st_mode) and (python_info.st_dev, python_info.st_ino) == (actual_python_info.st_dev, actual_python_info.st_ino), "held Python FD is not the executing interpreter")
    _require(os.path.samefile(__file__, f"/proc/self/fd/{worker_fd}"), "held worker FD is not the executing source")
    worker_length, worker_sha = _fd_hash(worker_fd)
    _require(worker_sha == body["worker_source_sha256"], "worker source digest mismatch")
    stage_path = f"/proc/self/fd/{stage_fd}"
    _require(os.path.isdir(stage_path), "held stage procfd is unavailable")
    spice_members, spice_rows = _record_rows(stage_path, "spiceypy", profile)
    numpy_members, numpy_rows = _record_rows(stage_path, "numpy", profile)
    actual_spice, actual_numpy, all_files = _distribution_files(stage_path)
    _require(actual_spice == spice_members and actual_numpy == numpy_members and all_files == spice_members | numpy_members, "extracted distribution is not globally RECORD-closed")
    _require(_manifest(spice_rows) == profile["spiceypy_tree"], "SpiceyPy tree manifest mismatch")
    _require(_manifest(numpy_rows) == profile["numpy_tree"], "NumPy tree manifest mismatch")
    numpy_native_rows = [row for row in numpy_rows if row[0].endswith((".so", ".dll", ".dylib")) or ".so." in PurePosixPath(row[0]).name]
    _require(_manifest(numpy_native_rows) == profile["numpy_native_tree"], "NumPy native tree manifest mismatch")
    sys.path.insert(0, stage_path)
    try:
        import numpy as np
        import numpy._core._multiarray_umath as multiarray
        import numpy.linalg._umath_linalg  # noqa: F401
        import spiceypy as spice
        from spiceypy.utils import libspicehelper
        from spiceypy.utils.exceptions import NotFoundError, SpiceSPKINSUFFDATA
    except (ImportError, OSError) as exc:
        raise _DependencyError("exact staged NumPy/SpiceyPy runtime cannot be imported") from exc
    coverage_error_types = (NotFoundError, SpiceSPKINSUFFDATA)

    _require(np.__version__ == "2.3.5" and spice.__version__ == "8.2.0", "provider version mismatch")
    expected_numpy = os.path.join(stage_path, "numpy/__init__.py")
    expected_spice = os.path.join(stage_path, "spiceypy/__init__.py")
    _require(os.path.samefile(np.__file__, expected_numpy), "NumPy module escaped held stage")
    _require(os.path.samefile(spice.__file__, expected_spice), "SpiceyPy module escaped held stage")
    _require(not any(name == "spiceypy.cyice" or name.startswith("spiceypy.cyice.") for name in sys.modules), "cyice imported")
    cspice_path = os.path.join(stage_path, "spiceypy/utils/libcspice.so")
    _require(os.path.samefile(libspicehelper.libspice_path, cspice_path), "loaded CSPICE escaped held stage")
    cspice_length, cspice_sha = _hash_path(cspice_path)
    _require(cspice_length == 3_561_056 and cspice_sha == _CSPICE_SHA256, "loaded CSPICE bytes mismatch")
    try:
        libc = ctypes.CDLL(None)
        libc.fegetround.argtypes = []
        libc.fegetround.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise _DependencyError("host fenv symbols are unavailable") from exc
    fenv_before = int(libc.fegetround())
    if fenv_before != 0:
        raise _DependencyError("worker rounding mode is not FE_TONEAREST")
    if spice.failed():
        raise _DependencyError("fresh CSPICE error state is failed")
    fresh_counts = _pool_counts(spice)
    if fresh_counts != [0, 0]:
        raise _DependencyError("fresh kernel pool is not empty")
    spice.kclear()
    _require(not spice.failed() and _pool_counts(spice) == [0, 0] and libc.fegetround() == 0, "initial CSPICE cleanup/profile check failed")
    _require(spice.tkvrsn("TOOLKIT") == "CSPICE_N0067", "CSPICE toolkit readback mismatch")
    _require(not spice.failed() and libc.fegetround() == 0, "CSPICE state changed during toolkit readback")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(os.environ.get(variable) == "1", "numerical thread environment mismatch")
    cpu_payload = _canonical_json({"baseline": multiarray.__cpu_baseline__, "dispatch": multiarray.__cpu_dispatch__, "features": multiarray.__cpu_features__})
    if len(cpu_payload) != _CPU_PAYLOAD_LENGTH or hashlib.sha256(cpu_payload).hexdigest() != _CPU_PAYLOAD_SHA256:
        raise _DependencyError("NumPy CPU feature payload mismatch")
    openblas_path = next(iter(sorted(PurePosixPath(path).as_posix() for path, _, _ in profile["loaded_numpy_natives"] if "openblas" in path)))
    try:
        openblas = ctypes.CDLL(os.path.join(stage_path, openblas_path))
        get_threads = openblas.scipy_openblas_get_num_threads64_
        get_threads.argtypes = []
        get_threads.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise _DependencyError("calibrated OpenBLAS runtime symbol is unavailable") from exc
    if int(get_threads()) != 1:
        raise _DependencyError("OpenBLAS thread count differs from one")
    expected_native_paths = ("spiceypy/utils/libcspice.so",) + tuple(path for path, _, _ in profile["loaded_numpy_natives"])
    all_stage_native_paths = tuple(sorted(
        path for path in all_files
        if path.endswith((".so", ".dll", ".dylib")) or ".so." in PurePosixPath(path).name
    ))
    mapped_distribution, host_dependencies = _mapped_runtime_observation(stage_path, expected_native_paths, all_stage_native_paths)
    expected_distribution = [("spiceypy/utils/libcspice.so", 3_561_056, _CSPICE_SHA256)] + list(profile["loaded_numpy_natives"])
    _require(sorted(mapped_distribution) == sorted(expected_distribution), "mapped native roster mismatch")
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or not libc_version:
        raise _DependencyError("host libc identity is unavailable")
    executable_length, executable_sha = _fd_hash(python_fd)
    effective_et = _f64_from_bits(body["effective_et_be_hex"], "effective ET")
    targets = [int(value) for value in body["targets"]]
    spk_pre_stat = os.fstat(spk_fd)
    spk_length, spk_sha_before = _fd_hash(spk_fd, 32_726_016)
    _require(spk_sha_before == "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2", "held SPK bytes are outside exact de440s profile")
    spk_offset_before = os.lseek(spk_fd, 0, os.SEEK_CUR)
    spk_proc_path = f"/proc/self/fd/{spk_fd}"
    _require(tuple(spice.getfat(spk_proc_path)) == ("DAF", "SPK"), "held SPK getfat mismatch")
    _require(not spice.failed() and libc.fegetround() == 0 and _pool_counts(spice) == [0, 0], "CSPICE state changed during held-SPK classification")
    loaded_counts: list[int] = []
    post_query_counts: list[int] = []
    after_clear_counts: list[int] = []
    loaded_handle = -1
    target_rows: list[dict[str, object]] = []
    provider_failure: Exception | None = None
    try:
        spice.furnsh(spk_proc_path)
        _require(libc.fegetround() == 0 and not spice.failed(), "CSPICE state changed after furnsh")
        loaded_counts = _pool_counts(spice)
        _require(loaded_counts == [1, 1], "kernel pool does not contain exactly one SPK")
        loaded_file, loaded_type, loaded_source, loaded_handle_raw = _found_tuple(tuple(spice.kdata(0, "ALL", 1_024, 33, 1_024)), 4)
        loaded_handle = int(loaded_handle_raw)
        _require(loaded_file == spk_proc_path and loaded_type == "SPK" and loaded_source == "", "kdata inventory mismatch")
        _require(1 <= loaded_handle <= _SPICE_INT_MAX, "invalid kernel handle")
        authoritative: dict[int, tuple[list[str], str]] = {}
        for target in targets:
            try:
                state_raw, light_time_raw = spice.spkgeo(target, effective_et, "J2000", 0)
            except coverage_error_types as exc:
                raise _CoverageError("SPKGEO target state is unavailable") from exc
            state = [float(value) for value in state_raw]
            light_time = float(light_time_raw)
            _require(len(state) == 6 and all(math.isfinite(value) for value in state), "SPKGEO returned invalid state")
            _require(math.isfinite(light_time) and light_time >= 0.0 and not (light_time == 0.0 and math.copysign(1.0, light_time) < 0.0), "SPKGEO returned invalid light time")
            authoritative[target] = ([ _f64_bits(value) for value in state ], _f64_bits(light_time))
            _require(libc.fegetround() == 0 and not spice.failed(), "CSPICE state changed during SPKGEO phase")
        for target in targets:
            state_bits, light_bits = authoritative[target]
            legs = _selected_chain(spice, target, effective_et, loaded_handle, coverage_error_types)
            _require(libc.fegetround() == 0 and not spice.failed(), "CSPICE state changed during selector phase")
            target_rows.append({"legs": legs, "spkgeo_light_time_be_hex": light_bits, "spkgeo_state_be_hex": state_bits, "target_naif_id": target})
        post_query_counts = _pool_counts(spice)
        _require(post_query_counts == [1, 1], "kernel pool changed during query")
        _require(not spice.failed() and libc.fegetround() == 0, "post-query CSPICE/fenv state mismatch")
    except Exception as exc:
        provider_failure = exc
    finally:
        try:
            if spice.failed():
                spice.reset()
            spice.kclear()
            if spice.failed():
                spice.reset()
            after_clear_counts = _pool_counts(spice)
            _require(after_clear_counts == [0, 0] and not spice.failed() and libc.fegetround() == 0, "CSPICE failure cleanup did not restore clear state")
        except Exception as cleanup_exc:
            provider_failure = cleanup_exc
    fenv_after = int(libc.fegetround())
    spk_offset_after = os.lseek(spk_fd, 0, os.SEEK_CUR)
    spk_post_stat = os.fstat(spk_fd)
    spk_length_after, spk_sha_after = _fd_hash(spk_fd, 32_726_016)
    stat_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")
    _finish_provider_observation(
        provider_failure,
        cleanup_valid=after_clear_counts == [0, 0] and not spice.failed(),
        fenv_after=fenv_after,
        spk_offset_before=spk_offset_before,
        spk_offset_after=spk_offset_after,
        spk_pre_stat=tuple(getattr(spk_pre_stat, name) for name in stat_fields),
        spk_post_stat=tuple(getattr(spk_post_stat, name) for name in stat_fields),
        spk_length_before=spk_length,
        spk_length_after=spk_length_after,
        spk_sha_before=spk_sha_before,
        spk_sha_after=spk_sha_after,
    )
    return {
        "abi_tag": abi,
        "after_clear_pool_counts": after_clear_counts,
        "cspice_toolkit_version": "CSPICE_N0067",
        "effective_et_be_hex": str(body["effective_et_be_hex"]),
        "fegetround_after": fenv_after,
        "fegetround_before": fenv_before,
        "fresh_pool_counts": fresh_counts,
        "host_native_dependencies": [[name, length, digest] for name, length, digest in host_dependencies],
        "libc_name": libc_name,
        "libc_version": libc_version,
        "loaded_distribution_native_files": [[path, length, digest] for path, length, digest in sorted(mapped_distribution)],
        "loaded_handle": loaded_handle,
        "loaded_pool_counts": loaded_counts,
        "numpy_cpu_feature_payload_byte_length": len(cpu_payload),
        "numpy_cpu_feature_payload_sha256": hashlib.sha256(cpu_payload).hexdigest(),
        "post_query_pool_counts": post_query_counts,
        "python_executable_byte_length": executable_length,
        "python_executable_sha256": executable_sha,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "request_sha256": request_sha256,
        "role": body["role"],
        "nonce": body["nonce"],
        "schema": _RESULT_SCHEMA,
        "spiceypy_record_sha256": profile["spiceypy_record"],
        "spiceypy_tree": list(profile["spiceypy_tree"]),
        "numpy_record_sha256": profile["numpy_record"],
        "numpy_tree": list(profile["numpy_tree"]),
        "numpy_native_tree": list(profile["numpy_native_tree"]),
        "spk_byte_length": spk_length,
        "spk_sha256": spk_sha_before,
        "spk_offset_before": spk_offset_before,
        "spk_offset_after": spk_offset_after,
        "targets": target_rows,
        "worker_source_byte_length": worker_length,
        "worker_source_sha256": worker_sha,
    }


def _error_kind(exc: BaseException) -> str:
    if isinstance(exc, _CoverageError):
        return "COVERAGE"
    if isinstance(exc, (_DependencyError, ImportError, ModuleNotFoundError)):
        return "DEPENDENCY"
    return "DATA"


def _send(value: object) -> None:
    encoded = _canonical_json(value)
    if len(encoded) > _MAXIMUM_RESULT_BYTES:
        encoded = _canonical_json({"error_kind": "DATA", "error_status": "WORKER_RESULT_EXCEEDED_CAP", "schema": _RESULT_SCHEMA})
    sys.stdout.buffer.write(struct.pack(">I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> int:
    try:
        body, request_sha = _receive_request()
        result = _execute(body, request_sha)
        _send({"result": result, "status": "OK"})
        return 0
    except Exception as exc:
        try:
            _send({"error_kind": _error_kind(exc), "error_status": "FAIL_CLOSED", "schema": _RESULT_SCHEMA, "status": "ERROR"})
        except Exception:
            pass
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
