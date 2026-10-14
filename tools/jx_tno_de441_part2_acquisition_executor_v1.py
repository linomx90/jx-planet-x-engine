#!/usr/bin/env python3
"""Offline, fail-closed acquisition executor for the sealed DE441 part-2 SPK.

The transport is deliberately external.  This tool never uses a network and
never resumes a transfer.  It accepts one already-complete staged file, proves
the exact preregistration authority before opening that file, validates and
copies the bytes into private content-addressed storage, independently rehashes
and revalidates the published object, and creates the portable custody receipt
last.  It does not run dynamics or make a scientific claim.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import errno
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_part2_acquisition_executor_v1.py"
GIT_EXECUTABLE = "/usr/bin/git"
DEFAULT_CUSTODY_PATH = (
    PROJECT_ROOT / "runs" / "tno_de441_part2_acquisition_v1" / "CUSTODY.json"
)

PREREGISTRATION_TAG = "tno-de441-part2-acquisition-v1-preregistered-20261014"
PREREGISTRATION_TAG_OBJECT = "0a6d1c752489d08f73a318cf8de16942998a6992"
PREREGISTRATION_TAG_PAYLOAD_BYTES = 246
PREREGISTRATION_TAG_PAYLOAD_SHA256 = (
    "afdaa2339002f8005b8933fcab2a76fa4ffa309305461328294f1c2ebbcc7aa2"
)
PREREGISTRATION_COMMIT = "b86f8da9ffdd44a89dadb2a56bbb565919400f3a"
PREREGISTRATION_TREE = "980243fd12b40cf0939102ceaeb70a58daa89342"

PROTOCOL_SCHEMA = "jx.tno.de441-part2-acquisition.protocol.v1"
CUSTODY_SCHEMA = "jx.tno.de441-part2-acquisition.custody.v1"
CUSTODY_DOMAIN = b"jx.tno.de441-part2-acquisition.custody.v1\0"
CLASSIFICATION = "DE441_PART2_OFFICIAL_KERNEL_LOCAL_BYTE_CUSTODY_ONLY"
CLAIM_CEILING = "LOCAL_EXACT_KERNEL_BYTE_CUSTODY_AND_OFFICIAL_METADATA_LINEAGE_ONLY"
FIXTURE_CUSTODY_SCHEMA = "jx.test.tno.de441-part2-acquisition.fixture-custody.v1"
FIXTURE_CUSTODY_DOMAIN = b"jx.test.tno.de441-part2-acquisition.fixture-custody.v1\0"
FIXTURE_CLASSIFICATION = "PRIVATE_TEST_FIXTURE_ONLY_NO_DE441_CUSTODY"
FIXTURE_CLAIM_CEILING = "PRIVATE_BYTE_PIPELINE_TEST_ONLY_NO_PRODUCTION_CUSTODY"

KERNEL_ARTIFACT_ID = "artifact.naif.de441-part2.spk.v1"
KERNEL_FILENAME = "de441_part-2.bsp"
KERNEL_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    "de441_part-2.bsp"
)
KERNEL_BYTE_LENGTH = 1_656_830_976
KERNEL_OFFICIAL_MD5 = "ad8dfa4e505ef0e3a5d587a5b4705632"
KERNEL_COVERAGE_JD_TDB = ("2440400.50", "8000016.50")
REQUIRED_QUERY_JD_TDB = (
    "2458411.5",
    "2456187.5",
    "2460034.5",
    "2458052.5",
    "2456596.5",
    "2452568.5",
    "2458299.5",
    "2458767.5",
    "2458156.5",
    "2458324.5",
    "2457961.5",
    "2458098.5",
    "2461200.5",
)

SPICEYPY_VERSION = "8.2.0"
SPICEYPY_WHEEL_BYTES = 2_442_346
SPICEYPY_WHEEL_SHA256 = (
    "00c2714d3dd67180794ad0b66735d6cac470e19627d6e99f9628a50b52133d90"
)
CSPICE_LIBRARY_BYTES = 3_561_056
CSPICE_LIBRARY_SHA256 = (
    "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84"
)
CSPICE_TOOLKIT_VERSION = "CSPICE_N0067"

OFFICIAL_METADATA = (
    {
        "role": "official_md5_roster",
        "url": (
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
            "aa_checksums.txt"
        ),
        "byte_length": 559,
        "sha256": "3c618bed7d51a48ee059269097eeeab1f69f3b2641ec2aeb96132757418c845b",
    },
    {
        "role": "official_spk_summary_roster",
        "url": (
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
            "aa_summaries.txt"
        ),
        "byte_length": 12_911,
        "sha256": "2b34f045b63731a6a315a83b1e2f240dd208a4b5a19ee3419be0573485c15f87",
    },
    {
        "role": "official_de441_technical_comments",
        "url": (
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
            "de441_tech-comments.txt"
        ),
        "byte_length": 59_453,
        "sha256": "eda144896f7a09fd810c6261036d05d767ce8c77cb2b3795bc824fb623881007",
    },
)

STORED_TARGET_CENTER_ROSTER = (
    (1, 0, "MERCURY_BARYCENTER"),
    (2, 0, "VENUS_BARYCENTER"),
    (3, 0, "EARTH_BARYCENTER"),
    (4, 0, "MARS_BARYCENTER"),
    (5, 0, "JUPITER_BARYCENTER"),
    (6, 0, "SATURN_BARYCENTER"),
    (7, 0, "URANUS_BARYCENTER"),
    (8, 0, "NEPTUNE_BARYCENTER"),
    (9, 0, "PLUTO_BARYCENTER"),
    (10, 0, "SUN"),
    (199, 1, "MERCURY"),
    (299, 2, "VENUS"),
    (301, 3, "MOON"),
    (399, 3, "EARTH"),
)

RESOLVED11_CHAINS = (
    (10, (10, 0), "SUN"),
    (199, (199, 1, 0), "MERCURY"),
    (299, (299, 2, 0), "VENUS"),
    (399, (399, 3, 0), "EARTH"),
    (301, (301, 3, 0), "MOON"),
    (4, (4, 0), "MARS_BARYCENTER"),
    (5, (5, 0), "JUPITER_BARYCENTER"),
    (6, (6, 0), "SATURN_BARYCENTER"),
    (7, (7, 0), "URANUS_BARYCENTER"),
    (8, (8, 0), "NEPTUNE_BARYCENTER"),
    (9, (9, 0), "PLUTO_BARYCENTER"),
)


@dataclass(frozen=True, slots=True)
class SealedBinding:
    role: str
    relative_path: str
    byte_length: int
    sha256: str
    require_unchanged_worktree_copy: bool


PREREGISTRATION_BINDINGS = (
    SealedBinding(
        "preregistration_source",
        "tools/jx_tno_de441_part2_acquisition_v1.py",
        45_084,
        "5b2d1ac961724331a8484f9f737b60240480ad0e8ca9dca7b7056f0a5dbd4f5a",
        True,
    ),
    SealedBinding(
        "preregistration_test",
        "tests/test_tno_de441_part2_acquisition_v1.py",
        22_198,
        "9017f4d3ab12553f94b6b26e6d241a5c70b3ea42b46b15b7d2701eac9dfc5527",
        True,
    ),
    SealedBinding(
        "preregistration_protocol",
        "runs/tno_de441_part2_acquisition_v1/PROTOCOL.json",
        14_275,
        "93625b46883d121e8968d9ae4e1a48a6579ad5f2e7e433346f5001d0da2ed74d",
        True,
    ),
    SealedBinding(
        "preregistration_sorted_test_matrix",
        "tools/test_matrix_manifest.json",
        13_062,
        "a94b3876358aaecb6cf17ee581a0e74d5e0c106fe40b49a88973da2b219407a3",
        False,
    ),
)


class AcquisitionExecutorError(RuntimeError):
    """The authority, byte-custody, publication, or validation gate failed."""


@dataclass(frozen=True, slots=True)
class _KernelContract:
    artifact_id: str
    filename: str
    byte_length: int
    official_md5: str
    coverage_jd_tdb: tuple[str, str]
    production: bool
    fixture_capability: object | None


_PRODUCTION_CONTRACT = _KernelContract(
    KERNEL_ARTIFACT_ID,
    KERNEL_FILENAME,
    KERNEL_BYTE_LENGTH,
    KERNEL_OFFICIAL_MD5,
    KERNEL_COVERAGE_JD_TDB,
    True,
    None,
)
_FIXTURE_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class _HeldSourceBinding:
    fd: int
    initial: os.stat_result
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _RuntimeInput:
    path: Path
    fd: int
    initial: os.stat_result
    byte_length: int
    sha256: str


def _fixture_contract(payload: bytes, capability: object) -> _KernelContract:
    """Return a tiny internal-only contract; never reachable from the CLI."""
    if capability is not _FIXTURE_CAPABILITY or type(payload) is not bytes or not payload:
        raise AcquisitionExecutorError("invalid private fixture capability")
    return _KernelContract(
        "artifact.test.synthetic-spk.v1",
        KERNEL_FILENAME,
        len(payload),
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        KERNEL_COVERAGE_JD_TDB,
        False,
        _FIXTURE_CAPABILITY,
    )


class _SemanticBackend(Protocol):
    def inspect(self, held_fd: int, contract: _KernelContract) -> dict[str, Any]: ...

    def runtime_record(self) -> dict[str, Any]: ...


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise AcquisitionExecutorError("value is not strict canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise AcquisitionExecutorError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any], contract: _KernelContract) -> str:
    if type(value) is not dict:
        raise AcquisitionExecutorError("custody content must be an exact object")
    _assert_contract(contract)
    record = dict(value)
    record.pop("content_sha256", None)
    domain = CUSTODY_DOMAIN if contract is _PRODUCTION_CONTRACT else FIXTURE_CUSTODY_DOMAIN
    return _sha256(domain + _canonical_json(record))


def _pairs(rows: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if type(key) is not str or key in value:
            raise AcquisitionExecutorError("JSON contains a duplicate or invalid key")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise AcquisitionExecutorError(f"JSON contains non-finite constant {token}")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise AcquisitionExecutorError(f"{label} is empty or not exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except AcquisitionExecutorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AcquisitionExecutorError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AcquisitionExecutorError(f"{label} root is not an exact object")
    return value


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _rename_stable_identity(info: os.stat_result) -> tuple[int, ...]:
    """Identity fields a same-directory rename cannot legitimately change."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
    )


def _git_environment() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", "/nonexistent"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git(*arguments: str) -> bytes:
    process = subprocess.run(
        [GIT_EXECUTABLE, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    if process.returncode != 0:
        raise AcquisitionExecutorError("sealed Git authority check failed")
    return process.stdout


def _read_fd_all(fd: int, expected_size: int, label: str) -> bytes:
    payload = bytearray()
    offset = 0
    while offset < expected_size:
        block = os.pread(fd, min(1 << 20, expected_size - offset), offset)
        if not block:
            raise AcquisitionExecutorError(f"{label} ended before its declared length")
        payload.extend(block)
        offset += len(block)
    if os.pread(fd, 1, expected_size):
        raise AcquisitionExecutorError(f"{label} has bytes after its declared length")
    return bytes(payload)


def _read_bound_worktree_file(binding: SealedBinding) -> bytes:
    path = PROJECT_ROOT.joinpath(*PurePosixPath(binding.relative_path).parts)
    try:
        fd = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise AcquisitionExecutorError(f"cannot open {binding.role}") from exc
    try:
        before = os.fstat(fd)
        path_before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != binding.byte_length
            or _stat_identity(before) != _stat_identity(path_before)
        ):
            raise AcquisitionExecutorError(f"{binding.role} identity differs")
        payload = _read_fd_all(fd, binding.byte_length, binding.role)
        after = os.fstat(fd)
        path_after = path.stat(follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise AcquisitionExecutorError(f"{binding.role} changed while held")
    finally:
        os.close(fd)
    if _sha256(payload) != binding.sha256:
        raise AcquisitionExecutorError(f"{binding.role} SHA-256 differs")
    return payload


def _binding_record(binding: SealedBinding) -> dict[str, object]:
    return {
        "role": binding.role,
        "relative_path": binding.relative_path,
        "bytes": binding.byte_length,
        "sha256": binding.sha256,
    }


def audit_preregistration() -> dict[str, Any]:
    """Bind the exact annotated authority and rebuild its protocol."""
    tag_object = _git("rev-parse", f"{PREREGISTRATION_TAG}^{{tag}}").decode("ascii").strip()
    commit = _git("rev-parse", f"{PREREGISTRATION_TAG}^{{commit}}").decode("ascii").strip()
    tree = _git("rev-parse", f"{PREREGISTRATION_TAG}^{{tree}}").decode("ascii").strip()
    if (
        tag_object != PREREGISTRATION_TAG_OBJECT
        or commit != PREREGISTRATION_COMMIT
        or tree != PREREGISTRATION_TREE
    ):
        raise AcquisitionExecutorError("annotated preregistration identity differs")
    tag_payload = _git("cat-file", "tag", PREREGISTRATION_TAG_OBJECT)
    if (
        len(tag_payload) != PREREGISTRATION_TAG_PAYLOAD_BYTES
        or _sha256(tag_payload) != PREREGISTRATION_TAG_PAYLOAD_SHA256
    ):
        raise AcquisitionExecutorError("annotated preregistration payload differs")
    try:
        _git("merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD")
    except AcquisitionExecutorError as exc:
        raise AcquisitionExecutorError(
            "HEAD does not descend from preregistration"
        ) from exc

    worktree_payloads: dict[str, bytes] = {}
    for binding in PREREGISTRATION_BINDINGS:
        sealed = _git("show", f"{PREREGISTRATION_COMMIT}:{binding.relative_path}")
        if len(sealed) != binding.byte_length or _sha256(sealed) != binding.sha256:
            raise AcquisitionExecutorError(f"sealed {binding.role} blob differs")
        if binding.require_unchanged_worktree_copy:
            current = _read_bound_worktree_file(binding)
            if current != sealed:
                raise AcquisitionExecutorError(f"working {binding.role} differs from sealed bytes")
            worktree_payloads[binding.role] = current

    protocol_payload = worktree_payloads["preregistration_protocol"]
    protocol = _strict_json(protocol_payload, "preregistration protocol")
    if protocol_payload != _canonical_json(protocol):
        raise AcquisitionExecutorError("preregistration protocol is not canonical")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("authority", {}).get("future_preregistration_tag_name")
        != PREREGISTRATION_TAG
        or protocol.get("official_source", {}).get("expected_byte_length")
        != KERNEL_BYTE_LENGTH
        or protocol.get("official_source", {}).get("official_md5")
        != KERNEL_OFFICIAL_MD5
        or protocol.get("claims", {}).get("kernel_custody_established") is not False
        or protocol.get("claims", {}).get("scientific_execution_authorized") is not False
    ):
        raise AcquisitionExecutorError("preregistration protocol authority differs")

    source_path = PROJECT_ROOT / PREREGISTRATION_BINDINGS[0].relative_path
    source_payload = worktree_payloads["preregistration_source"]
    module_name = "_jx_tno_de441_part2_preregistration_bound"
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(source_path)
    sys.modules[module_name] = module
    try:
        exec(compile(source_payload, os.fspath(source_path), "exec"), module.__dict__)
        rebuilt = module._canonical_json(module.build_protocol())
    except Exception as exc:
        raise AcquisitionExecutorError("exact preregistration rebuild failed") from exc
    finally:
        sys.modules.pop(module_name, None)
    if rebuilt != protocol_payload:
        raise AcquisitionExecutorError("rebuilt preregistration protocol differs")

    return {
        "tag": PREREGISTRATION_TAG,
        "tag_object": PREREGISTRATION_TAG_OBJECT,
        "tag_payload_bytes": PREREGISTRATION_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": PREREGISTRATION_TAG_PAYLOAD_SHA256,
        "commit": PREREGISTRATION_COMMIT,
        "tree": PREREGISTRATION_TREE,
        "bindings": [_binding_record(item) for item in PREREGISTRATION_BINDINGS],
        "protocol_content_sha256": protocol["content_sha256"],
    }


def _absolute_nosymlink(path: Path, label: str, *, must_exist: bool = True) -> Path:
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raise AcquisitionExecutorError(f"{label} must be absolute")
    absolute = Path(os.path.abspath(raw))
    for component in reversed((absolute, *absolute.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            if must_exist:
                raise AcquisitionExecutorError(f"{label} does not exist")
            continue
        except OSError as exc:
            raise AcquisitionExecutorError(f"cannot inspect {label} ancestry") from exc
        if stat.S_ISLNK(info.st_mode):
            raise AcquisitionExecutorError(f"{label} has a symlink ancestor")
    if must_exist and absolute.resolve(strict=True) != absolute:
        raise AcquisitionExecutorError(f"{label} is not an exact real path")
    return absolute


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _worktree_roots() -> tuple[Path, ...]:
    payload = _git("worktree", "list", "--porcelain").decode("utf-8", errors="strict")
    roots: list[Path] = []
    for line in payload.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line[9:]).resolve(strict=True))
    if not roots:
        raise AcquisitionExecutorError("Git reported no worktree roots")
    return tuple(roots)


def _reject_git_controlled_directory_fd(fd: int, label: str) -> None:
    discovery = subprocess.run(
        [
            GIT_EXECUTABLE,
            "-C",
            f"/proc/self/fd/{fd}",
            "rev-parse",
            "--show-toplevel",
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_git_environment(),
    )
    if discovery.returncode == 0:
        raise AcquisitionExecutorError(f"{label} is inside a Git worktree")


def _open_asset_root(asset_root: Path, *, project_root: Path = PROJECT_ROOT) -> tuple[Path, int]:
    root = _absolute_nosymlink(asset_root, "asset root")
    expected_parent = project_root.resolve(strict=True).parent
    if root.parent != expected_parent or root == project_root.resolve(strict=True):
        raise AcquisitionExecutorError("asset root is not a real sibling of the worktree")
    for worktree in _worktree_roots():
        if root == worktree or _is_within(root, worktree) or _is_within(worktree, root):
            raise AcquisitionExecutorError("asset root overlaps a Git worktree")
    if (root / ".git").exists():
        raise AcquisitionExecutorError("asset root is itself Git-controlled")
    discovery = subprocess.run(
        [GIT_EXECUTABLE, "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_git_environment(),
    )
    if discovery.returncode == 0:
        raise AcquisitionExecutorError("asset root is inside a Git worktree")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        raise AcquisitionExecutorError("cannot open asset root") from exc
    info = os.fstat(fd)
    path_info = root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or _stat_identity(info) != _stat_identity(path_info)
    ):
        os.close(fd)
        raise AcquisitionExecutorError("asset root is not an owner-private stable directory")
    return root, fd


def _verify_directory_anchor(fd: int, path: Path, label: str) -> None:
    held = os.fstat(fd)
    try:
        named = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionExecutorError(f"{label} path disappeared") from exc
    held_identity = (
        held.st_dev,
        held.st_ino,
        held.st_mode,
        held.st_uid,
        held.st_gid,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_mode,
        named.st_uid,
        named.st_gid,
    )
    if (
        not stat.S_ISDIR(held.st_mode)
        or held.st_uid != os.geteuid()
        or stat.S_IMODE(held.st_mode) != 0o700
        or held_identity != named_identity
    ):
        raise AcquisitionExecutorError(f"{label} path no longer names the held directory")


def _verify_directory_edge(
    parent_fd: int, name: str, child_fd: int, label: str
) -> None:
    held = os.fstat(child_fd)
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionExecutorError(f"{label} directory edge disappeared") from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or held.st_uid != os.geteuid()
        or stat.S_IMODE(held.st_mode) != 0o700
        or _stat_identity(held) != _stat_identity(named)
    ):
        raise AcquisitionExecutorError(f"{label} no longer names the held private directory")


def _verify_private_chain_anchor(
    root_fd: int,
    parts: Sequence[str],
    held_leaf_fd: int,
    label: str,
) -> None:
    reopened = _open_private_directory_chain(root_fd, parts, label)
    try:
        if _stat_identity(os.fstat(reopened)) != _stat_identity(os.fstat(held_leaf_fd)):
            raise AcquisitionExecutorError(f"{label} directory chain was detached")
    finally:
        os.close(reopened)


def _verify_content_address_chain(
    root_fd: int,
    object_fd: int,
    sha_fd: int,
    digest_fd: int,
    digest: str,
) -> None:
    _verify_directory_edge(root_fd, "objects", object_fd, "objects")
    _verify_directory_edge(object_fd, "sha256", sha_fd, "sha256")
    _verify_directory_edge(sha_fd, digest, digest_fd, "digest")


def _relative_under(root: Path, path: Path, label: str) -> tuple[str, ...]:
    absolute = _absolute_nosymlink(path, label)
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise AcquisitionExecutorError(f"{label} is outside the asset root") from exc
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise AcquisitionExecutorError(f"{label} has an invalid relative path")
    return relative.parts


def _open_private_directory_chain(root_fd: int, parts: Sequence[str], label: str) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            next_fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current,
            )
            info = os.fstat(next_fd)
            path_info = os.stat(part, dir_fd=current, follow_symlinks=False)
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or _stat_identity(info) != _stat_identity(path_info)
            ):
                os.close(next_fd)
                raise AcquisitionExecutorError(f"{label} directory chain is not private")
            os.close(current)
            current = next_fd
    except BaseException:
        os.close(current)
        raise
    return current


def _open_staged(
    root: Path,
    root_fd: int,
    staged_file: Path,
    contract: _KernelContract,
) -> tuple[int, int, str, os.stat_result]:
    parts = _relative_under(root, staged_file, "staged file")
    if parts[-1] != contract.filename or parts[0] == "objects":
        raise AcquisitionExecutorError("staged file name or location differs")
    parent_fd = _open_private_directory_chain(root_fd, parts[:-1], "staged")
    try:
        _reject_git_controlled_directory_fd(parent_fd, "staging directory")
        fd = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        os.close(parent_fd)
        raise AcquisitionExecutorError("cannot open complete staged file") from exc
    before = os.fstat(fd)
    path_before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size != contract.byte_length
        or _stat_identity(before) != _stat_identity(path_before)
    ):
        os.close(fd)
        os.close(parent_fd)
        raise AcquisitionExecutorError("staged file type, mode, link, size, or identity differs")
    if frozenset(os.listdir(parent_fd)) != frozenset({parts[-1]}):
        os.close(fd)
        os.close(parent_fd)
        raise AcquisitionExecutorError("staging directory contains a partial or extra entry")
    return parent_fd, fd, parts[-1], before


def _stream_hashes(fd: int, expected_size: int, label: str) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        block = os.pread(fd, min(8 << 20, expected_size - offset), offset)
        if not block:
            raise AcquisitionExecutorError(f"{label} ended early")
        md5.update(block)
        sha256.update(block)
        offset += len(block)
    if os.pread(fd, 1, expected_size):
        raise AcquisitionExecutorError(f"{label} has trailing bytes")
    return md5.hexdigest(), sha256.hexdigest()


def _verify_held_path(
    fd: int,
    initial: os.stat_result,
    parent_fd: int,
    name: str,
    label: str,
) -> None:
    after = os.fstat(fd)
    try:
        path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionExecutorError(f"{label} path disappeared") from exc
    if (
        _stat_identity(initial) != _stat_identity(after)
        or _stat_identity(initial) != _stat_identity(path_after)
    ):
        raise AcquisitionExecutorError(f"{label} changed while held")


def _verify_held_after_rename(
    fd: int,
    pre_rename: os.stat_result,
    parent_fd: int,
    name: str,
    label: str,
) -> os.stat_result:
    after = os.fstat(fd)
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionExecutorError(f"{label} path disappeared") from exc
    if (
        _rename_stable_identity(pre_rename) != _rename_stable_identity(after)
        or _stat_identity(after) != _stat_identity(named)
    ):
        raise AcquisitionExecutorError(f"{label} changed across atomic rename")
    return after


def _mkdir_private(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise AcquisitionExecutorError("cannot create private content-address directory") from exc
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    info = os.fstat(fd)
    path_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or _stat_identity(info) != _stat_identity(path_info)
    ):
        os.close(fd)
        raise AcquisitionExecutorError("content-address directory is not private")
    try:
        _reject_git_controlled_directory_fd(fd, "content-address directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise AcquisitionExecutorError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        directory_fd,
        source.encode("ascii"),
        directory_fd,
        destination.encode("ascii"),
        1,
    ) != 0:
        observed = ctypes.get_errno()
        if observed == errno.EEXIST:
            raise AcquisitionExecutorError("publication destination already exists")
        raise AcquisitionExecutorError(f"renameat2 failed: {os.strerror(observed)}")


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise AcquisitionExecutorError("write made no progress")
        view = view[count:]


def _copy_fd(source_fd: int, destination_fd: int, expected_size: int) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        block = os.pread(source_fd, min(8 << 20, expected_size - offset), offset)
        if not block:
            raise AcquisitionExecutorError("staged file ended during publication copy")
        _write_all(destination_fd, block)
        md5.update(block)
        sha256.update(block)
        offset += len(block)
    if os.pread(source_fd, 1, expected_size):
        raise AcquisitionExecutorError("staged file gained trailing bytes")
    return md5.hexdigest(), sha256.hexdigest()


def _unlink_held_regular_name(directory_fd: int, name: str, held_fd: int) -> bool:
    """Unlink only when ``name`` still denotes the exact held single-link file."""
    held = os.fstat(held_fd)
    try:
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or _stat_identity(held) != _stat_identity(named)
    ):
        return False
    os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)
    return True


def _available_bytes(root_fd: int) -> int:
    values = os.fstatvfs(root_fd)
    return int(values.f_bavail) * int(values.f_frsize)


def _jd_to_et(jd: str) -> float:
    value = (Decimal(jd) - Decimal("2451545.0")) * Decimal(86400)
    result = float(value)
    if not math.isfinite(result):
        raise AcquisitionExecutorError("JD-to-ET conversion is non-finite")
    return result


def _open_runtime_input(
    path: Path, label: str, byte_length: int, expected_sha256: str
) -> _RuntimeInput:
    absolute = _absolute_nosymlink(path, label)
    fd = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        initial = os.fstat(fd)
        path_initial = absolute.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != os.geteuid()
            or initial.st_nlink != 1
            or initial.st_size != byte_length
            or _stat_identity(initial) != _stat_identity(path_initial)
        ):
            raise AcquisitionExecutorError(f"{label} type, owner, link, size, or identity differs")
        _md5, observed = _stream_hashes(fd, byte_length, label)
        if observed != expected_sha256:
            raise AcquisitionExecutorError(f"{label} SHA-256 differs")
        _verify_runtime_input(
            _RuntimeInput(absolute, fd, initial, byte_length, expected_sha256),
            label,
            rehash=False,
        )
    except BaseException:
        os.close(fd)
        raise
    return _RuntimeInput(absolute, fd, initial, byte_length, expected_sha256)


def _verify_runtime_input(
    runtime: _RuntimeInput, label: str, *, rehash: bool = True
) -> None:
    parent_fd = os.open(
        runtime.path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        _verify_held_path(
            runtime.fd,
            runtime.initial,
            parent_fd,
            runtime.path.name,
            label,
        )
        if rehash:
            _md5, observed = _stream_hashes(runtime.fd, runtime.byte_length, label)
            if observed != runtime.sha256:
                raise AcquisitionExecutorError(f"{label} changed while held")
            _verify_held_path(
                runtime.fd,
                runtime.initial,
                parent_fd,
                runtime.path.name,
                label,
            )
    finally:
        os.close(parent_fd)


def _sealed_runtime_memfd(runtime: _RuntimeInput, label: str) -> int:
    if not hasattr(os, "memfd_create"):
        raise AcquisitionExecutorError("Linux sealed memfd runtime binding is unavailable")
    flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
    fd = os.memfd_create(f"jx-{label}", flags)
    try:
        md5, sha256 = _copy_fd(runtime.fd, fd, runtime.byte_length)
        del md5
        if sha256 != runtime.sha256:
            raise AcquisitionExecutorError(f"{label} sealed copy SHA-256 differs")
        os.fsync(fd)
        required_seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, required_seals)
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & required_seals != required_seals:
            raise AcquisitionExecutorError(f"{label} memfd seals differ")
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != runtime.byte_length:
            raise AcquisitionExecutorError(f"{label} sealed copy identity differs")
        _md5, sealed_sha = _stream_hashes(fd, runtime.byte_length, f"sealed {label}")
        if sealed_sha != runtime.sha256:
            raise AcquisitionExecutorError(f"{label} sealed copy rehash differs")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _verify_sealed_runtime_fd(
    fd: int, label: str, byte_length: int, expected_sha256: str
) -> None:
    required_seals = (
        fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_WRITE
    )
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size != byte_length
        or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & required_seals != required_seals
    ):
        raise AcquisitionExecutorError(f"sealed {label} identity differs")
    _md5, observed = _stream_hashes(fd, byte_length, f"sealed {label}")
    if observed != expected_sha256:
        raise AcquisitionExecutorError(f"sealed {label} SHA-256 differs")


def _mapped_library_is_held(helper: Any, library_fd: int) -> bool:
    try:
        address = int(ctypes.cast(helper.libspice.tkvrsn_c, ctypes.c_void_p).value)
    except (AttributeError, TypeError, ValueError):
        return False
    held = os.fstat(library_fd)
    maps_fd = os.open(
        "/proc/self/maps",
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(maps_fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(maps_fd)
    try:
        lines = b"".join(chunks).decode("ascii", errors="strict").splitlines()
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                continue
            start_text, stop_text = fields[0].split("-", 1)
            if int(start_text, 16) <= address < int(stop_text, 16):
                major_text, minor_text = fields[3].split(":", 1)
                return (
                    int(major_text, 16) == os.major(held.st_dev)
                    and int(minor_text, 16) == os.minor(held.st_dev)
                    and int(fields[4]) == held.st_ino
                )
    except (UnicodeDecodeError, ValueError):
        return False
    return False


class _PinnedSpiceBackend:
    def __init__(self, wheel: Path, cspice_library: Path) -> None:
        self._wheel_fd = -1
        self._library_fd = -1
        if any(
            name == "spiceypy" or name.startswith("spiceypy.")
            for name in sys.modules
        ):
            raise AcquisitionExecutorError("SpiceyPy was imported before runtime binding")
        if "CSPICE_SHARED_LIB" in os.environ:
            raise AcquisitionExecutorError("CSPICE_SHARED_LIB was already configured")
        wheel_input = _open_runtime_input(
            wheel,
            "SpiceyPy wheel",
            SPICEYPY_WHEEL_BYTES,
            SPICEYPY_WHEEL_SHA256,
        )
        try:
            library_input = _open_runtime_input(
                cspice_library,
                "CSPICE library",
                CSPICE_LIBRARY_BYTES,
                CSPICE_LIBRARY_SHA256,
            )
        except BaseException:
            os.close(wheel_input.fd)
            raise
        wheel_memfd = library_memfd = -1
        environment_bound = False
        try:
            wheel_memfd = _sealed_runtime_memfd(wheel_input, "spiceypy-wheel")
            library_memfd = _sealed_runtime_memfd(library_input, "cspice-library")
            wheel_proc = f"/proc/self/fd/{wheel_memfd}"
            library_proc = f"/proc/self/fd/{library_memfd}"
            os.environ["CSPICE_SHARED_LIB"] = library_proc
            environment_bound = True
            sys.path.insert(0, wheel_proc)
            try:
                spice = importlib.import_module("spiceypy")
                helper = importlib.import_module("spiceypy.utils.libspicehelper")
                version = importlib.metadata.version("spiceypy")
            except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as exc:
                raise AcquisitionExecutorError("cannot load exact pinned SpiceyPy runtime") from exc
            finally:
                if sys.path and sys.path[0] == wheel_proc:
                    sys.path.pop(0)
                if environment_bound:
                    os.environ.pop("CSPICE_SHARED_LIB", None)
                    environment_bound = False

            loader = getattr(getattr(spice, "__spec__", None), "loader", None)
            origin = getattr(getattr(spice, "__spec__", None), "origin", None)
            helper_origin = getattr(getattr(helper, "__spec__", None), "origin", None)
            try:
                libc = ctypes.CDLL(None)
                libc.fegetround.argtypes = []
                libc.fegetround.restype = ctypes.c_int
            except (OSError, AttributeError) as exc:
                raise AcquisitionExecutorError(
                    "host floating-point environment is unavailable"
                ) from exc
            if (
                version != SPICEYPY_VERSION
                or getattr(spice, "__version__", None) != SPICEYPY_VERSION
                or getattr(loader, "archive", None) != wheel_proc
                or origin != f"{wheel_proc}/spiceypy/__init__.py"
                or helper_origin
                != f"{wheel_proc}/spiceypy/utils/libspicehelper.py"
                or os.fspath(helper.libspice_path) != library_proc
                or not _mapped_library_is_held(helper, library_memfd)
                or spice.tkvrsn("TOOLKIT") != CSPICE_TOOLKIT_VERSION
                or spice.failed()
                or int(libc.fegetround()) != 0
            ):
                raise AcquisitionExecutorError("loaded SpiceyPy/CSPICE identity differs")
            _verify_runtime_input(wheel_input, "post-import SpiceyPy wheel")
            _verify_runtime_input(library_input, "post-import CSPICE library")
            _verify_sealed_runtime_fd(
                wheel_memfd,
                "SpiceyPy wheel",
                SPICEYPY_WHEEL_BYTES,
                SPICEYPY_WHEEL_SHA256,
            )
            _verify_sealed_runtime_fd(
                library_memfd,
                "CSPICE library",
                CSPICE_LIBRARY_BYTES,
                CSPICE_LIBRARY_SHA256,
            )
            spice.kclear()
            if spice.failed() or int(spice.ktotal("ALL")) != 0:
                raise AcquisitionExecutorError(
                    "CSPICE did not begin with a clear kernel pool"
                )
            self._wheel_fd = wheel_memfd
            self._library_fd = library_memfd
            self._spice = spice
            self._helper = helper
            self._libc = libc
            wheel_memfd = library_memfd = -1
        finally:
            if environment_bound:
                os.environ.pop("CSPICE_SHARED_LIB", None)
            for fd in (wheel_memfd, library_memfd, wheel_input.fd, library_input.fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def verify_runtime(self) -> None:
        _verify_sealed_runtime_fd(
            self._wheel_fd,
            "SpiceyPy wheel",
            SPICEYPY_WHEEL_BYTES,
            SPICEYPY_WHEEL_SHA256,
        )
        _verify_sealed_runtime_fd(
            self._library_fd,
            "CSPICE library",
            CSPICE_LIBRARY_BYTES,
            CSPICE_LIBRARY_SHA256,
        )
        if (
            not _mapped_library_is_held(self._helper, self._library_fd)
            or int(self._libc.fegetround()) != 0
        ):
            raise AcquisitionExecutorError("loaded runtime mapping or rounding mode differs")

    def close(self) -> None:
        for name in ("_wheel_fd", "_library_fd"):
            fd = getattr(self, name, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)

    def runtime_record(self) -> dict[str, Any]:
        self.verify_runtime()
        return {
            "spiceypy_version": SPICEYPY_VERSION,
            "spiceypy_wheel_bytes": SPICEYPY_WHEEL_BYTES,
            "spiceypy_wheel_sha256": SPICEYPY_WHEEL_SHA256,
            "cspice_toolkit_version": CSPICE_TOOLKIT_VERSION,
            "loaded_cspice_bytes": CSPICE_LIBRARY_BYTES,
            "loaded_cspice_sha256": CSPICE_LIBRARY_SHA256,
            "kernel_path_binding": "LINUX_PROC_SELF_FD_HELD_DESCRIPTOR",
            "floating_point_rounding_mode": "FE_TONEAREST",
            "runtime_inputs_full_rehash_before_and_after_import": True,
            "runtime_loaded_from_immutable_sealed_memfds": True,
            "loaded_cspice_mapping_matches_held_memfd_inode": True,
        }

    def inspect(self, held_fd: int, contract: _KernelContract) -> dict[str, Any]:
        if contract is not _PRODUCTION_CONTRACT:
            raise AcquisitionExecutorError("pinned backend refuses a fixture contract")
        spice = self._spice
        if int(self._libc.fegetround()) != 0:
            raise AcquisitionExecutorError("floating-point rounding mode differs")
        proc_path = f"/proc/self/fd/{held_fd}"
        if tuple(spice.getfat(proc_path)) != ("DAF", "SPK"):
            raise AcquisitionExecutorError("kernel is not an exact DAF/SPK")
        object_ids = tuple(sorted(int(value) for value in spice.spkobj(proc_path)))
        expected_ids = tuple(sorted(row[0] for row in STORED_TARGET_CENTER_ROSTER))
        if object_ids != expected_ids:
            raise AcquisitionExecutorError("SPK target roster differs")

        descriptors: list[dict[str, Any]] = []
        handle = int(spice.dafopr(proc_path))
        try:
            spice.dafbfs(handle)
            while bool(spice.daffna()):
                doubles, integers = spice.dafus(spice.dafgs(5), 2, 6)
                if len(doubles) != 2 or len(integers) != 6:
                    raise AcquisitionExecutorError("SPK descriptor shape differs")
                first, last = (float(value) for value in doubles)
                target, center, frame, data_type, begin, end = (
                    int(value) for value in integers
                )
                identifier = str(spice.dafgn(41))
                if (
                    not math.isfinite(first)
                    or not math.isfinite(last)
                    or first > last
                    or target == center
                    or frame != 1
                    or data_type != 2
                    or begin < 1
                    or end < begin
                    or end > 2_147_483_647
                    or not identifier
                    or identifier != identifier.strip()
                    or len(identifier) > 40
                    or any(not 0x20 <= ord(character) <= 0x7E for character in identifier)
                ):
                    raise AcquisitionExecutorError("SPK segment descriptor is invalid")
                descriptors.append(
                    {
                        "target_naif_id": target,
                        "center_naif_id": center,
                        "frame_naif_id": frame,
                        "segment_type": data_type,
                        "first_et_hex": first.hex(),
                        "last_et_hex": last.hex(),
                        "begin_daf_address": begin,
                        "end_daf_address": end,
                        "segment_identifier": identifier,
                    }
                )
        finally:
            spice.dafcls(handle)
        if not descriptors:
            raise AcquisitionExecutorError("SPK has no segment descriptors")
        observed_pairs = {(row["target_naif_id"], row["center_naif_id"]) for row in descriptors}
        expected_pairs = {(row[0], row[1]) for row in STORED_TARGET_CENTER_ROSTER}
        if observed_pairs != expected_pairs:
            raise AcquisitionExecutorError("SPK target-center roster differs")
        address_ranges = [
            (row["begin_daf_address"], row["end_daf_address"])
            for row in descriptors
        ]
        if len(set(address_ranges)) != len(address_ranges):
            raise AcquisitionExecutorError("SPK segment address ranges are not unique")
        ordered_ranges = sorted(address_ranges)
        if any(
            later_begin <= earlier_end
            for (_earlier_begin, earlier_end), (later_begin, _later_end) in zip(
                ordered_ranges, ordered_ranges[1:]
            )
        ):
            raise AcquisitionExecutorError("SPK segment address ranges overlap")

        coverage_rows: list[dict[str, Any]] = []
        required_ets = tuple(_jd_to_et(value) for value in REQUIRED_QUERY_JD_TDB)
        full_start, full_stop = (_jd_to_et(value) for value in contract.coverage_jd_tdb)
        for target in expected_ids:
            window = spice.spkcov(proc_path, target)
            interval_count = int(spice.wncard(window))
            intervals: list[list[str]] = []
            numeric: list[tuple[float, float]] = []
            for index in range(interval_count):
                start, stop = (float(value) for value in spice.wnfetd(window, index))
                if not math.isfinite(start) or not math.isfinite(stop) or start > stop:
                    raise AcquisitionExecutorError("SPK coverage interval is invalid")
                intervals.append([start.hex(), stop.hex()])
                numeric.append((start, stop))
            if not numeric or numeric[0][0] != full_start or numeric[-1][1] != full_stop:
                raise AcquisitionExecutorError("SPK exact endpoint coverage differs")
            if any(not any(start <= et <= stop for start, stop in numeric) for et in required_ets):
                raise AcquisitionExecutorError("SPK misses a required bridge query time")
            coverage_rows.append(
                {"target_naif_id": target, "intervals_et_hex": intervals}
            )

        chain_checks: list[dict[str, Any]] = []
        for target, chain, semantic in RESOLVED11_CHAINS:
            for edge_target, edge_center in zip(chain, chain[1:]):
                matching = [
                    row
                    for row in descriptors
                    if row["target_naif_id"] == edge_target
                    and row["center_naif_id"] == edge_center
                ]
                if not matching:
                    raise AcquisitionExecutorError("required resolved11 chain edge is absent")
                ranges = [
                    (float.fromhex(row["first_et_hex"]), float.fromhex(row["last_et_hex"]))
                    for row in matching
                ]
                if any(
                    not any(start <= et <= stop for start, stop in ranges)
                    for et in required_ets
                ):
                    raise AcquisitionExecutorError(
                        "required resolved11 chain edge lacks query-time coverage"
                    )
                chain_checks.append(
                    {
                        "semantic_id": semantic,
                        "resolved_target_naif_id": target,
                        "edge_target_naif_id": edge_target,
                        "edge_center_naif_id": edge_center,
                        "all_required_query_times_covered": True,
                    }
                )

        spice.kclear()
        if (
            spice.failed()
            or int(spice.ktotal("ALL")) != 0
            or int(self._libc.fegetround()) != 0
        ):
            raise AcquisitionExecutorError("CSPICE cleanup failed")
        return {
            "architecture": "DAF",
            "kernel_type": "SPK",
            "reference_frame_naif_id": 1,
            "segment_type": 2,
            "segment_count": len(descriptors),
            "segments": descriptors,
            "object_ids": list(object_ids),
            "coverage": coverage_rows,
            "exact_stored_target_center_roster": [
                {
                    "target_naif_id": target,
                    "center_naif_id": center,
                    "semantic_id": semantic,
                }
                for target, center, semantic in STORED_TARGET_CENTER_ROSTER
            ],
            "resolved11_chains": [
                {
                    "target_naif_id": target,
                    "target_to_ssb_chain": list(chain),
                    "semantic_id": semantic,
                }
                for target, chain, semantic in RESOLVED11_CHAINS
            ],
            "resolved11_chain_edge_checks": chain_checks,
            "required_query_jd_tdb": list(REQUIRED_QUERY_JD_TDB),
            "coverage_jd_tdb": list(contract.coverage_jd_tdb),
            "all_required_chains_covered_at_every_query_time": True,
            "status": "PASSED",
        }


def _open_executor_source_binding() -> _HeldSourceBinding:
    fd = os.open(
        SOURCE_PATH,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(fd)
        path_before = SOURCE_PATH.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= 2_000_000
            or _stat_identity(before) != _stat_identity(path_before)
        ):
            raise AcquisitionExecutorError("executor source identity differs")
        payload = _read_fd_all(fd, before.st_size, "executor source")
        after = os.fstat(fd)
        path_after = SOURCE_PATH.stat(follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise AcquisitionExecutorError("executor source changed while held")
    except BaseException:
        os.close(fd)
        raise
    return _HeldSourceBinding(fd, before, len(payload), _sha256(payload))


def _verify_executor_source_binding(binding: _HeldSourceBinding) -> None:
    parent_fd = os.open(
        SOURCE_PATH.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        _verify_held_path(
            binding.fd,
            binding.initial,
            parent_fd,
            SOURCE_PATH.name,
            "executor source",
        )
    finally:
        os.close(parent_fd)


def _executor_record(binding: _HeldSourceBinding) -> dict[str, Any]:
    _verify_executor_source_binding(binding)
    return {
        "relative_path": "tools/jx_tno_de441_part2_acquisition_executor_v1.py",
        "bytes": binding.byte_length,
        "sha256": binding.sha256,
        "binding_scope": "SOURCE_PATH_BYTES_HELD_STABLE_FOR_COMPLETE_OPERATION",
        "python_code_object_attestation_claimed": False,
    }


def _published_at(token: str) -> str:
    if type(token) is not str or len(token) != 20 or not token.endswith("Z"):
        raise AcquisitionExecutorError("published-at UTC must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        value = datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise AcquisitionExecutorError("published-at UTC is invalid") from exc
    if value.strftime("%Y-%m-%dT%H:%M:%SZ") != token:
        raise AcquisitionExecutorError("published-at UTC is not canonical")
    return token


def _artifact_record(contract: _KernelContract, sha256: str) -> dict[str, Any]:
    return {
        "artifact_id": contract.artifact_id,
        "logical_locator": f"objects/sha256/{sha256}/{contract.filename}",
        "filename": contract.filename,
        "byte_length": contract.byte_length,
        "official_md5": contract.official_md5,
        "sha256": sha256,
    }


def _claims(contract: _KernelContract) -> dict[str, bool]:
    production = contract is _PRODUCTION_CONTRACT
    claims = {
        "official_source_identity_bound": production,
        "exact_local_kernel_bytes_retained": production,
        "kernel_locally_sha256_bound": production,
        "kernel_custody_established": production,
        "spk_semantics_validated": production,
        "network_accessed_by_executor": False,
        "kernel_downloaded_by_executor": False,
        "executor_resume_surface_present": False,
        "external_transport_method_or_resume_history_proven": False,
        "official_metadata_source_bytes_revalidated_offline": False,
        "provider_transport_authenticated": False,
        "continuous_custody_proven": False,
        "source_evaluator_qualified": False,
        "source_node_gate_passed": False,
        "common_epoch_bridge_executed": False,
        "dynamics_executed": False,
        "gpu_executed": False,
        "classification_generated": False,
        "population_inference": False,
        "planet_nine_evidence": False,
        "science_result": False,
        "scientific_execution_authorized": False,
        "executor_loaded_code_attested": False,
    }
    if not production:
        claims["private_test_fixture_only"] = True
        claims["fixture_byte_pipeline_exercised"] = True
    return claims


def _build_custody(
    preregistration: dict[str, Any],
    contract: _KernelContract,
    sha256: str,
    semantics: dict[str, Any],
    backend: _SemanticBackend,
    published_at_utc: str,
    source_binding: _HeldSourceBinding,
) -> dict[str, Any]:
    production = contract is _PRODUCTION_CONTRACT
    source: dict[str, Any]
    if production:
        source = {
            "official_url": KERNEL_URL,
            "official_metadata": [dict(row) for row in OFFICIAL_METADATA],
            "provider_evidence_limit": (
                "Frozen official metadata and MD5 establish lineage checks only; "
                "they do not prove continuous custody or authenticate transport."
            ),
            "network_access_by_executor": False,
            "transport_external_to_executor": True,
            "executor_resume_surface_present": False,
            "external_transport_method_or_resume_history_proven": False,
        }
    else:
        source = {
            "private_test_fixture_only": True,
            "official_source_used": False,
            "network_access_by_executor": False,
            "transport_external_to_executor": False,
            "executor_resume_surface_present": False,
        }
    record: dict[str, Any] = {
        "schema": CUSTODY_SCHEMA if production else FIXTURE_CUSTODY_SCHEMA,
        "classification": CLASSIFICATION if production else FIXTURE_CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING if production else FIXTURE_CLAIM_CEILING,
        "status": (
            "VALIDATED_LOCAL_EXACT_DE441_PART2_KERNEL_CUSTODY"
            if production
            else "PASSED_PRIVATE_TEST_FIXTURE_BYTE_PIPELINE_NO_DE441_CUSTODY"
        ),
        "protocol": {
            "schema": PROTOCOL_SCHEMA,
            "relative_path": "runs/tno_de441_part2_acquisition_v1/PROTOCOL.json",
            "file_sha256": PREREGISTRATION_BINDINGS[2].sha256,
            "content_sha256": preregistration["protocol_content_sha256"],
        },
        "preregistration_git_custody": preregistration,
        "source": source,
        "artifact": _artifact_record(contract, sha256),
        "spk_semantics": {
            "runtime": backend.runtime_record(),
            "observation": semantics,
        },
        "validation": {
            "executor_source": _executor_record(source_binding),
            "authority_bound_before_staged_open": True,
            "staged_complete_stream_md5_passed": True,
            "staged_complete_stream_sha256_computed": True,
            "copy_stream_md5_and_sha256_equal_staged": True,
            "atomic_noreplace_publication": True,
            "independent_complete_post_publication_rehash_passed": True,
            "published_spk_semantics_recomputed": True,
            "published_semantics_equal_staged_semantics": True,
            "receipt_created_last": True,
            "published_at_utc_provenance": "CALLER_SUPPLIED_CANONICAL_UTC_TOKEN",
            "offline_validator_required_for_every_future_use": True,
        },
        "claims": _claims(contract),
        "published_at_utc": _published_at(published_at_utc),
        "content_sha256": "",
    }
    record["content_sha256"] = _content_sha256(record, contract)
    _reject_absolute_receipt_values(record)
    return record


def _reject_absolute_receipt_values(value: object) -> None:
    if type(value) is str:
        if value.startswith("/") or value.startswith("file:"):
            raise AcquisitionExecutorError("portable custody contains an absolute path")
    elif type(value) is list:
        for item in value:
            _reject_absolute_receipt_values(item)
    elif type(value) is dict:
        for key, item in value.items():
            if key in {"device_id", "inode_number", "hostname", "username", "local_path"}:
                raise AcquisitionExecutorError("portable custody contains a forbidden field")
            _reject_absolute_receipt_values(item)


def _custody_schema(contract: _KernelContract) -> str:
    return CUSTODY_SCHEMA if contract is _PRODUCTION_CONTRACT else FIXTURE_CUSTODY_SCHEMA


def _custody_roster(contract: _KernelContract) -> frozenset[str]:
    return (
        frozenset({"PROTOCOL.json", "CUSTODY.json"})
        if contract is _PRODUCTION_CONTRACT
        else frozenset({"CUSTODY.json"})
    )


def _create_custody_last(
    path: Path,
    payload: bytes,
    contract: _KernelContract,
    final_gate: Callable[[], None] | None = None,
    postpublication_gate: Callable[[], None] | None = None,
) -> None:
    _assert_contract(contract)
    absolute = _absolute_nosymlink(path, "custody parent", must_exist=False)
    if absolute.name != "CUSTODY.json":
        raise AcquisitionExecutorError("custody filename differs")
    parent = absolute.parent
    _absolute_nosymlink(parent, "custody parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    parent_info = os.fstat(parent_fd)
    parent_path_info = parent.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.geteuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or _stat_identity(parent_info) != _stat_identity(parent_path_info)
    ):
        os.close(parent_fd)
        raise AcquisitionExecutorError("custody parent is not an owner-private directory")
    pre_roster = (
        frozenset({"PROTOCOL.json"})
        if contract is _PRODUCTION_CONTRACT
        else frozenset()
    )
    if frozenset(os.listdir(parent_fd)) != pre_roster:
        os.close(parent_fd)
        raise AcquisitionExecutorError("pre-custody tracked roster differs")
    partial = ".CUSTODY.json.partial"
    fd = -1
    try:
        try:
            os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AcquisitionExecutorError("custody destination already exists")
        fd = os.open(
            partial,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        partial_info = os.fstat(fd)
        if (
            not stat.S_ISREG(partial_info.st_mode)
            or partial_info.st_nlink != 1
            or partial_info.st_size != len(payload)
            or stat.S_IMODE(partial_info.st_mode) != 0o600
        ):
            raise AcquisitionExecutorError("custody partial identity differs")
        _verify_held_path(fd, partial_info, parent_fd, partial, "custody partial")
        if _read_fd_all(fd, len(payload), "custody partial") != payload:
            raise AcquisitionExecutorError("custody partial bytes differ")
        if final_gate is not None:
            final_gate()
        _verify_directory_anchor(parent_fd, parent, "custody parent")
        try:
            _rename_noreplace(parent_fd, partial, absolute.name)
        except BaseException as exc:
            held = os.fstat(fd)
            try:
                final_info = os.stat(
                    absolute.name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                final_info = None
            final_is_held = (
                final_info is not None
                and _stat_identity(final_info) == _stat_identity(held)
            )
            if final_is_held:
                try:
                    partial_path = os.stat(
                        partial, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    partial_path = None
                if partial_path is not None:
                    raise AcquisitionExecutorError(
                        "ambiguous custody publication left both names"
                    ) from exc
            elif final_info is not None:
                raise AcquisitionExecutorError(
                    "ambiguous custody publication found an unverified destination"
                ) from exc
            else:
                _unlink_held_regular_name(parent_fd, partial, fd)
                raise
        published_info = _verify_held_after_rename(
            fd,
            partial_info,
            parent_fd,
            absolute.name,
            "published custody receipt",
        )
        if _read_fd_all(fd, len(payload), "published custody receipt") != payload:
            raise AcquisitionExecutorError("published custody receipt bytes differ")
        os.fsync(parent_fd)
        if frozenset(os.listdir(parent_fd)) != _custody_roster(contract):
            raise AcquisitionExecutorError("post-custody tracked roster differs")
        _verify_held_path(
            fd,
            published_info,
            parent_fd,
            absolute.name,
            "published custody receipt",
        )
        if _read_fd_all(fd, len(payload), "published custody receipt") != payload:
            raise AcquisitionExecutorError("published custody receipt bytes differ")
        if frozenset(os.listdir(parent_fd)) != _custody_roster(contract):
            raise AcquisitionExecutorError("post-gate custody roster differs")
        _verify_directory_anchor(parent_fd, parent, "custody parent")
        if postpublication_gate is not None:
            postpublication_gate()
        _verify_held_path(
            fd,
            published_info,
            parent_fd,
            absolute.name,
            "published custody receipt",
        )
        if _read_fd_all(fd, len(payload), "published custody receipt") != payload:
            raise AcquisitionExecutorError("published custody receipt bytes differ")
        if frozenset(os.listdir(parent_fd)) != _custody_roster(contract):
            raise AcquisitionExecutorError("post-gate custody roster differs")
        _verify_directory_anchor(parent_fd, parent, "custody parent")
    except BaseException:
        if fd >= 0:
            for name in (partial, absolute.name):
                try:
                    _unlink_held_regular_name(parent_fd, name, fd)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass
            fd = -1
        try:
            os.close(parent_fd)
        except OSError:
            pass
        raise
    if fd >= 0:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        os.close(parent_fd)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class _CustodyHandle:
    absolute: Path
    parent: Path
    parent_fd: int
    fd: int
    initial: os.stat_result
    payload: bytes
    value: dict[str, Any]
    contract: _KernelContract


def _open_custody_handle(path: Path, contract: _KernelContract) -> _CustodyHandle:
    _assert_contract(contract)
    absolute = _absolute_nosymlink(path, "custody receipt")
    if absolute.name != "CUSTODY.json":
        raise AcquisitionExecutorError("custody filename differs")
    parent_fd = os.open(
        absolute.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    parent_info = os.fstat(parent_fd)
    parent_path_info = absolute.parent.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.geteuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or _stat_identity(parent_info) != _stat_identity(parent_path_info)
    ):
        os.close(parent_fd)
        raise AcquisitionExecutorError("custody parent is not owner-private")
    if frozenset(os.listdir(parent_fd)) != _custody_roster(contract):
        os.close(parent_fd)
        raise AcquisitionExecutorError("custody tracked roster differs")
    try:
        fd = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= 2_000_000
        ):
            raise AcquisitionExecutorError("custody receipt mode, link, or size differs")
        payload = _read_fd_all(fd, before.st_size, "custody receipt")
        after = os.fstat(fd)
        path_after = os.stat(
            absolute.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise AcquisitionExecutorError("custody receipt changed while held")
        _verify_directory_anchor(parent_fd, absolute.parent, "custody parent")
        value = _strict_json(payload, "custody receipt")
        if payload != _canonical_json(value):
            raise AcquisitionExecutorError("custody receipt is not canonical JSON")
        if (
            value.get("schema") != _custody_schema(contract)
            or value.get("content_sha256") != _content_sha256(value, contract)
        ):
            raise AcquisitionExecutorError("custody receipt schema or content hash differs")
        _reject_absolute_receipt_values(value)
    except BaseException:
        os.close(fd)
        os.close(parent_fd)
        raise
    return _CustodyHandle(
        absolute,
        absolute.parent,
        parent_fd,
        fd,
        before,
        payload,
        value,
        contract,
    )


def _verify_custody_handle(handle: _CustodyHandle) -> None:
    _verify_held_path(
        handle.fd,
        handle.initial,
        handle.parent_fd,
        handle.absolute.name,
        "custody receipt",
    )
    if _read_fd_all(handle.fd, len(handle.payload), "custody receipt") != handle.payload:
        raise AcquisitionExecutorError("custody receipt bytes changed while held")
    if frozenset(os.listdir(handle.parent_fd)) != _custody_roster(handle.contract):
        raise AcquisitionExecutorError("custody tracked roster differs")
    _verify_directory_anchor(handle.parent_fd, handle.parent, "custody parent")


def _close_custody_handle(handle: _CustodyHandle) -> None:
    for fd in (handle.fd, handle.parent_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def _load_custody(
    path: Path, contract: _KernelContract
) -> tuple[bytes, dict[str, Any]]:
    handle = _open_custody_handle(path, contract)
    try:
        _verify_custody_handle(handle)
        return handle.payload, handle.value
    finally:
        _close_custody_handle(handle)


def _make_backend(
    contract: _KernelContract,
    backend: _SemanticBackend | None,
    spiceypy_wheel: Path | None,
    cspice_library: Path | None,
) -> _SemanticBackend:
    if contract is _PRODUCTION_CONTRACT:
        if backend is not None or spiceypy_wheel is None or cspice_library is None:
            raise AcquisitionExecutorError("production requires exact pinned runtime paths")
        return _PinnedSpiceBackend(spiceypy_wheel, cspice_library)
    if backend is None or spiceypy_wheel is not None or cspice_library is not None:
        raise AcquisitionExecutorError("private fixture runtime boundary differs")
    return backend


def _verify_backend_runtime(backend: _SemanticBackend) -> None:
    verifier = getattr(backend, "verify_runtime", None)
    if verifier is not None:
        verifier()


def _close_backend(backend: _SemanticBackend | None) -> None:
    if backend is None:
        return
    closer = getattr(backend, "close", None)
    if closer is not None:
        try:
            closer()
        except BaseException:
            pass


def _assert_contract(contract: _KernelContract) -> None:
    if contract is _PRODUCTION_CONTRACT:
        return
    if (
        contract.production
        or contract.artifact_id != "artifact.test.synthetic-spk.v1"
        or contract.fixture_capability is not _FIXTURE_CAPABILITY
    ):
        raise AcquisitionExecutorError("unrecognized kernel contract")


def _hook(hook: Callable[[str], None] | None, event: str) -> None:
    if hook is not None:
        hook(event)


def _enforce_path_overrides(
    contract: _KernelContract,
    custody_path: Path,
    asset_project_root: Path,
    fault_hook: Callable[[str], None] | None = None,
) -> None:
    custody_absolute = Path(os.path.abspath(os.fspath(custody_path)))
    project_absolute = asset_project_root.resolve(strict=True)
    if contract is not _PRODUCTION_CONTRACT:
        if (
            custody_absolute == DEFAULT_CUSTODY_PATH
            or project_absolute == PROJECT_ROOT.resolve(strict=True)
            or _is_within(custody_absolute, PROJECT_ROOT.resolve(strict=True))
        ):
            raise AcquisitionExecutorError(
                "private fixture cannot target production paths or project root"
            )
        return
    if (
        custody_absolute != DEFAULT_CUSTODY_PATH
        or project_absolute != PROJECT_ROOT.resolve(strict=True)
        or fault_hook is not None
    ):
        raise AcquisitionExecutorError("production internal override is forbidden")


def ingest_staged_file(
    asset_root: Path,
    staged_file: Path,
    published_at_utc: str,
    *,
    spiceypy_wheel: Path | None = None,
    cspice_library: Path | None = None,
    custody_path: Path = DEFAULT_CUSTODY_PATH,
    _contract: _KernelContract = _PRODUCTION_CONTRACT,
    _backend: _SemanticBackend | None = None,
    _asset_project_root: Path = PROJECT_ROOT,
    _fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate one complete externally staged file and publish exact custody."""
    _assert_contract(_contract)
    _enforce_path_overrides(
        _contract, custody_path, _asset_project_root, _fault_hook
    )
    published_at = _published_at(published_at_utc)
    source_binding = _open_executor_source_binding()
    root_fd = -1
    staged_parent_fd = -1
    staged_fd = -1
    object_fd = -1
    sha_fd = -1
    digest_fd = -1
    partial_fd = -1
    custody_guard_fd = -1
    backend: _SemanticBackend | None = None
    try:
        preregistration = audit_preregistration()  # Must precede any staged-file open/read.
        _hook(_fault_hook, "authority-bound-before-staged-open")
        root, root_fd = _open_asset_root(asset_root, project_root=_asset_project_root)
        if _available_bytes(root_fd) < 2 * _contract.byte_length:
            raise AcquisitionExecutorError("asset filesystem lacks frozen two-copy free space")
        backend = _make_backend(
            _contract, _backend, spiceypy_wheel, cspice_library
        )
        staged_parent_parts = _relative_under(root, staged_file, "staged file")[:-1]
        staged_parent_fd, staged_fd, staged_name, staged_initial = _open_staged(
            root, root_fd, staged_file, _contract
        )
        _hook(_fault_hook, "staged-opened")
        staged_md5, staged_sha256 = _stream_hashes(
            staged_fd, _contract.byte_length, "staged kernel"
        )
        if staged_md5 != _contract.official_md5:
            raise AcquisitionExecutorError("staged kernel official MD5 differs")
        _verify_held_path(
            staged_fd,
            staged_initial,
            staged_parent_fd,
            staged_name,
            "staged kernel",
        )
        staged_semantics = backend.inspect(staged_fd, _contract)
        _verify_held_path(
            staged_fd,
            staged_initial,
            staged_parent_fd,
            staged_name,
            "staged kernel",
        )
        _hook(_fault_hook, "staged-fully-validated")

        object_fd = _mkdir_private(root_fd, "objects")
        sha_fd = _mkdir_private(object_fd, "sha256")
        digest_fd = _mkdir_private(sha_fd, staged_sha256)
        final_name = _contract.filename
        partial_name = f".{final_name}.partial"
        try:
            os.stat(final_name, dir_fd=digest_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AcquisitionExecutorError("publication destination already exists")
        try:
            os.stat(partial_name, dir_fd=digest_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AcquisitionExecutorError("publication partial already exists")
        if os.listdir(digest_fd):
            raise AcquisitionExecutorError("publication digest directory is not empty")

        partial_fd = os.open(
            partial_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=digest_fd,
        )
        try:
            os.fchmod(partial_fd, 0o600)
            copy_md5, copy_sha256 = _copy_fd(
                staged_fd, partial_fd, _contract.byte_length
            )
            if copy_md5 != staged_md5 or copy_sha256 != staged_sha256:
                raise AcquisitionExecutorError("publication copy digest differs")
            os.fsync(partial_fd)
            partial_info = os.fstat(partial_fd)
            if (
                not stat.S_ISREG(partial_info.st_mode)
                or partial_info.st_nlink != 1
                or partial_info.st_size != _contract.byte_length
                or stat.S_IMODE(partial_info.st_mode) != 0o600
            ):
                raise AcquisitionExecutorError("publication partial identity differs")
        except BaseException:
            try:
                _unlink_held_regular_name(digest_fd, partial_name, partial_fd)
            except OSError:
                pass
            os.close(partial_fd)
            partial_fd = -1
            raise
        _verify_held_path(
            partial_fd,
            partial_info,
            digest_fd,
            partial_name,
            "publication partial",
        )
        _verify_held_path(
            staged_fd,
            staged_initial,
            staged_parent_fd,
            staged_name,
            "staged kernel",
        )
        try:
            _hook(_fault_hook, "partial-fsynced")
            _rename_noreplace(digest_fd, partial_name, final_name)
        except BaseException as exc:
            # Rename errors are ambiguous: inspect both names and never unlink final.
            held_partial = os.fstat(partial_fd)
            final_info: os.stat_result | None = None
            partial_path_info: os.stat_result | None = None
            try:
                final_info = os.stat(
                    final_name, dir_fd=digest_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            try:
                partial_path_info = os.stat(
                    partial_name, dir_fd=digest_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            final_is_held = (
                final_info is not None
                and _stat_identity(final_info) == _stat_identity(held_partial)
            )
            partial_is_held = (
                partial_path_info is not None
                and _stat_identity(partial_path_info) == _stat_identity(held_partial)
            )
            if final_is_held:
                verify_fd = os.open(
                    final_name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=digest_fd,
                )
                try:
                    ambiguous_initial = os.fstat(verify_fd)
                    if (
                        _stat_identity(ambiguous_initial)
                        != _stat_identity(held_partial)
                        or not stat.S_ISREG(ambiguous_initial.st_mode)
                        or ambiguous_initial.st_uid != os.geteuid()
                        or ambiguous_initial.st_nlink != 1
                        or ambiguous_initial.st_size != _contract.byte_length
                        or stat.S_IMODE(ambiguous_initial.st_mode) != 0o600
                    ):
                        raise AcquisitionExecutorError(
                            "ambiguous destination is not the exact held publication"
                        ) from exc
                    observed_md5, observed_sha = _stream_hashes(
                        verify_fd, _contract.byte_length, "ambiguous destination"
                    )
                    if observed_md5 != staged_md5 or observed_sha != staged_sha256:
                        raise AcquisitionExecutorError(
                            "ambiguous destination bytes differ; destination retained"
                        ) from exc
                    ambiguous_semantics = backend.inspect(verify_fd, _contract)
                    if ambiguous_semantics != staged_semantics:
                        raise AcquisitionExecutorError(
                            "ambiguous destination semantics differ; destination retained"
                        ) from exc
                    _verify_held_path(
                        verify_fd,
                        ambiguous_initial,
                        digest_fd,
                        final_name,
                        "ambiguous destination",
                    )
                finally:
                    os.close(verify_fd)
            if partial_is_held:
                _unlink_held_regular_name(digest_fd, partial_name, partial_fd)
            if final_info is not None and not final_is_held:
                raise AcquisitionExecutorError(
                    "ambiguous publication found an unverified destination; nothing removed"
                ) from exc
            if final_is_held:
                raise AcquisitionExecutorError(
                    "ambiguous publication retained a fully hashed destination; no receipt created"
                ) from exc
            raise
        _verify_held_after_rename(
            partial_fd,
            partial_info,
            digest_fd,
            final_name,
            "atomically published kernel",
        )
        os.close(partial_fd)
        partial_fd = -1
        if frozenset(os.listdir(digest_fd)) != frozenset({final_name}):
            raise AcquisitionExecutorError("published digest directory roster differs")
        os.fsync(digest_fd)
        os.fsync(sha_fd)
        os.fsync(object_fd)
        os.fsync(root_fd)
        _hook(_fault_hook, "published-before-independent-validation")

        published_fd = os.open(
            final_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=digest_fd,
        )
        try:
            published_initial = os.fstat(published_fd)
            path_initial = os.stat(final_name, dir_fd=digest_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(published_initial.st_mode)
                or published_initial.st_uid != os.geteuid()
                or published_initial.st_nlink != 1
                or published_initial.st_size != _contract.byte_length
                or stat.S_IMODE(published_initial.st_mode) != 0o600
                or _stat_identity(published_initial) != _stat_identity(path_initial)
            ):
                raise AcquisitionExecutorError("published kernel identity differs")
            published_md5, published_sha256 = _stream_hashes(
                published_fd, _contract.byte_length, "published kernel"
            )
            if published_md5 != staged_md5 or published_sha256 != staged_sha256:
                raise AcquisitionExecutorError("independent published rehash differs")
            published_semantics = backend.inspect(published_fd, _contract)
            if published_semantics != staged_semantics:
                raise AcquisitionExecutorError("published SPK semantics differ from staging")
            _verify_held_path(
                published_fd,
                published_initial,
                digest_fd,
                final_name,
                "published kernel",
            )
        finally:
            os.close(published_fd)
        _verify_held_path(
            staged_fd,
            staged_initial,
            staged_parent_fd,
            staged_name,
            "staged kernel",
        )
        _verify_directory_anchor(root_fd, root, "asset root")
        _hook(_fault_hook, "published-independently-validated")

        custody_guard_fd = os.open(
            final_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=digest_fd,
        )
        custody_guard_info = os.fstat(custody_guard_fd)
        guard_path_info = os.stat(
            final_name, dir_fd=digest_fd, follow_symlinks=False
        )
        if (
            _stat_identity(custody_guard_info) != _stat_identity(guard_path_info)
            or _stat_identity(custody_guard_info)
            != _stat_identity(published_initial)
            or not stat.S_ISREG(custody_guard_info.st_mode)
            or custody_guard_info.st_nlink != 1
            or custody_guard_info.st_size != _contract.byte_length
            or stat.S_IMODE(custody_guard_info.st_mode) != 0o600
        ):
            raise AcquisitionExecutorError("pre-custody published guard differs")

        custody = _build_custody(
            preregistration,
            _contract,
            staged_sha256,
            published_semantics,
            backend,
            published_at,
            source_binding,
        )
        custody_payload = _canonical_json(custody)
        _hook(_fault_hook, "before-custody-create-last")

        def final_gate() -> None:
            _verify_held_path(
                custody_guard_fd,
                custody_guard_info,
                digest_fd,
                final_name,
                "pre-receipt published kernel",
            )
            _verify_held_path(
                staged_fd,
                staged_initial,
                staged_parent_fd,
                staged_name,
                "pre-receipt staged kernel",
            )
            _verify_private_chain_anchor(
                root_fd,
                staged_parent_parts,
                staged_parent_fd,
                "staged",
            )
            _verify_content_address_chain(
                root_fd,
                object_fd,
                sha_fd,
                digest_fd,
                staged_sha256,
            )
            if frozenset(os.listdir(digest_fd)) != frozenset({final_name}):
                raise AcquisitionExecutorError("published digest directory roster differs")
            _verify_directory_anchor(root_fd, root, "pre-receipt asset root")
            _verify_backend_runtime(backend)
            _verify_executor_source_binding(source_binding)
            if audit_preregistration() != preregistration:
                raise AcquisitionExecutorError(
                    "preregistration authority changed during acquisition"
                )

        _create_custody_last(
            custody_path,
            custody_payload,
            _contract,
            final_gate=final_gate,
            postpublication_gate=final_gate,
        )
        return custody
    finally:
        _close_backend(backend)
        for fd in (
            partial_fd,
            custody_guard_fd,
            digest_fd,
            sha_fd,
            object_fd,
            staged_fd,
            staged_parent_fd,
            root_fd,
        ):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        try:
            os.close(source_binding.fd)
        except OSError:
            pass


def validate_custody(
    asset_root: Path,
    *,
    spiceypy_wheel: Path | None = None,
    cspice_library: Path | None = None,
    custody_path: Path = DEFAULT_CUSTODY_PATH,
    _contract: _KernelContract = _PRODUCTION_CONTRACT,
    _backend: _SemanticBackend | None = None,
    _asset_project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Read-only offline reconstruction and full revalidation of custody."""
    _assert_contract(_contract)
    _enforce_path_overrides(_contract, custody_path, _asset_project_root)
    source_binding = _open_executor_source_binding()
    custody_handle: _CustodyHandle | None = None
    root_fd = -1
    object_fd = sha_fd = digest_fd = -1
    object_guard_fd = -1
    backend: _SemanticBackend | None = None
    try:
        preregistration = audit_preregistration()
        custody_handle = _open_custody_handle(custody_path, _contract)
        observed = custody_handle.value
        root, root_fd = _open_asset_root(asset_root, project_root=_asset_project_root)
        backend = _make_backend(
            _contract, _backend, spiceypy_wheel, cspice_library
        )
        artifact = observed.get("artifact")
        if type(artifact) is not dict or artifact.get("artifact_id") != _contract.artifact_id:
            raise AcquisitionExecutorError("custody artifact identity differs")
        digest = artifact.get("sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or artifact != _artifact_record(_contract, digest)
        ):
            raise AcquisitionExecutorError("custody artifact record differs")
        object_fd = _open_private_directory_chain(root_fd, ("objects",), "objects")
        sha_fd = _open_private_directory_chain(object_fd, ("sha256",), "sha256")
        digest_fd = _open_private_directory_chain(sha_fd, (digest,), "digest")
        if frozenset(os.listdir(digest_fd)) != frozenset({_contract.filename}):
            raise AcquisitionExecutorError("published digest directory roster differs")
        fd = os.open(
            _contract.filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=digest_fd,
        )
        try:
            initial = os.fstat(fd)
            path_initial = os.stat(
                _contract.filename, dir_fd=digest_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(initial.st_mode)
                or initial.st_uid != os.geteuid()
                or initial.st_nlink != 1
                or initial.st_size != _contract.byte_length
                or stat.S_IMODE(initial.st_mode) != 0o600
                or _stat_identity(initial) != _stat_identity(path_initial)
            ):
                raise AcquisitionExecutorError("published kernel identity differs")
            md5, sha256 = _stream_hashes(fd, _contract.byte_length, "published kernel")
            if md5 != _contract.official_md5 or sha256 != digest:
                raise AcquisitionExecutorError("published kernel full rehash differs")
            semantics = backend.inspect(fd, _contract)
            _verify_held_path(
                fd,
                initial,
                digest_fd,
                _contract.filename,
                "published kernel",
            )
        finally:
            os.close(fd)
        object_guard_fd = os.open(
            _contract.filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=digest_fd,
        )
        object_guard_info = os.fstat(object_guard_fd)
        object_guard_path = os.stat(
            _contract.filename, dir_fd=digest_fd, follow_symlinks=False
        )
        if (
            _stat_identity(object_guard_info) != _stat_identity(object_guard_path)
            or _stat_identity(object_guard_info) != _stat_identity(initial)
        ):
            raise AcquisitionExecutorError("offline object guard identity differs")
        expected = _build_custody(
            preregistration,
            _contract,
            digest,
            semantics,
            backend,
            observed.get("published_at_utc"),
            source_binding,
        )
        if observed != expected:
            raise AcquisitionExecutorError("custody receipt differs from offline reconstruction")
        _verify_custody_handle(custody_handle)
        _verify_held_path(
            object_guard_fd,
            object_guard_info,
            digest_fd,
            _contract.filename,
            "offline published kernel",
        )
        _verify_content_address_chain(
            root_fd,
            object_fd,
            sha_fd,
            digest_fd,
            digest,
        )
        _verify_directory_anchor(root_fd, root, "asset root")
        _verify_backend_runtime(backend)
        _verify_executor_source_binding(source_binding)
        _verify_held_path(
            object_guard_fd,
            object_guard_info,
            digest_fd,
            _contract.filename,
            "offline published kernel",
        )
        _verify_content_address_chain(
            root_fd,
            object_fd,
            sha_fd,
            digest_fd,
            digest,
        )
        _verify_directory_anchor(root_fd, root, "asset root")
        if audit_preregistration() != preregistration:
            raise AcquisitionExecutorError(
                "preregistration authority changed during offline validation"
            )
        _verify_custody_handle(custody_handle)
        return observed
    finally:
        _close_backend(backend)
        if custody_handle is not None:
            _close_custody_handle(custody_handle)
        for fd in (object_guard_fd, digest_fd, sha_fd, object_fd, root_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        try:
            os.close(source_binding.fd)
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "audit-preregistration", help="verify exact sealed authority without kernel access"
    )
    ingest_parser = subparsers.add_parser(
        "ingest", help="ingest one complete externally transported staged file"
    )
    ingest_parser.add_argument("--asset-root", type=Path, required=True)
    ingest_parser.add_argument("--staged-file", type=Path, required=True)
    ingest_parser.add_argument("--published-at-utc", required=True)
    ingest_parser.add_argument("--spiceypy-wheel", type=Path, required=True)
    ingest_parser.add_argument("--cspice-library", type=Path, required=True)
    validate_parser = subparsers.add_parser(
        "validate-custody", help="read-only offline full custody validation"
    )
    validate_parser.add_argument("--asset-root", type=Path, required=True)
    validate_parser.add_argument("--spiceypy-wheel", type=Path, required=True)
    validate_parser.add_argument("--cspice-library", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit-preregistration":
        value = audit_preregistration()
        print(
            json.dumps(
                {
                    "commit": value["commit"],
                    "status": "PASSED",
                    "tag_object": value["tag_object"],
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "ingest":
        value = ingest_staged_file(
            arguments.asset_root,
            arguments.staged_file,
            arguments.published_at_utc,
            spiceypy_wheel=arguments.spiceypy_wheel,
            cspice_library=arguments.cspice_library,
        )
        print(
            json.dumps(
                {
                    "content_sha256": value["content_sha256"],
                    "logical_locator": value["artifact"]["logical_locator"],
                    "status": value["status"],
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "validate-custody":
        value = validate_custody(
            arguments.asset_root,
            spiceypy_wheel=arguments.spiceypy_wheel,
            cspice_library=arguments.cspice_library,
        )
        print(
            json.dumps(
                {
                    "content_sha256": value["content_sha256"],
                    "status": "VALID",
                },
                sort_keys=True,
            )
        )
    else:  # pragma: no cover - argparse enforces the exact command roster.
        raise AcquisitionExecutorError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
