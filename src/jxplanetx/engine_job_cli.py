"""Filesystem boundary for the portable Engine Job V1 byte API.

The numerical runner remains path-free.  This module adds bounded, no-follow
input reads and one create-only publication point for a complete ``.jxjob``
bundle.  It never replaces an existing path.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import secrets
import stat
import sys

from .engine.integrator_registry import list_integrator_routes
from .engine.runner import (
    ENGINE_JOB_BUNDLE_MAX_BYTES,
    ENGINE_JOB_BUNDLE_SCHEMA,
    ENGINE_JOB_EXPECTED_SHA256_SOURCE,
    ENGINE_JOB_V1_EXECUTION_POLICY,
    ENGINE_JOB_V1_MAX_MANIFEST_BYTES,
    EngineRunnerError,
    engine_job_claim_ceiling,
    engine_job_sha256,
    run_scenario_manifest,
    verify_engine_job_bytes,
)
from .engine.trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID


ENGINE_JOB_CLI_SUCCESS = 0
ENGINE_JOB_CLI_CONTRACT_FAILURE = 2
ENGINE_JOB_CLI_IO_FAILURE = 3
_READ_CHUNK_BYTES = 1024 * 1024


class EngineJobCLIIOError(OSError):
    """A held-file input or create-only output operation failed."""


def _public_claim_projection() -> dict[str, object]:
    return {
        **engine_job_claim_ceiling(),
        "bundle_hash_and_structure_verified": True,
        "manifest_artifact_restart_crossbindings_verified": True,
        "native_result_digest_verification": (
            "DECLARED_CROSS_BOUND_NOT_RECOMPUTED"
        ),
    }


def _required_open_flags() -> tuple[int, int, int]:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise EngineJobCLIIOError(
            "Engine Job CLI requires O_CLOEXEC, O_DIRECTORY, and O_NOFOLLOW"
        )
    return os.O_CLOEXEC, os.O_DIRECTORY, os.O_NOFOLLOW


def _path_components(value: object, label: str) -> tuple[bool, tuple[str, ...]]:
    if type(value) is not str or not value or "\x00" in value:
        raise EngineJobCLIIOError(f"{label} must be a nonempty filesystem path")
    path = Path(value)
    parts = path.parts
    absolute = path.is_absolute()
    components = parts[1:] if absolute else parts
    if (
        not components
        or any(part in {"", ".", ".."} for part in components)
    ):
        raise EngineJobCLIIOError(
            f"{label} must use normalized components without '.' or '..'"
        )
    return absolute, tuple(components)


def _open_parent(value: object, label: str) -> tuple[int, str]:
    cloexec, directory, nofollow = _required_open_flags()
    absolute, components = _path_components(value, label)
    try:
        current = os.open(
            "/" if absolute else ".",
            os.O_RDONLY | directory | nofollow | cloexec,
        )
    except OSError as exc:
        raise EngineJobCLIIOError(f"{label} root could not be opened safely") from exc
    try:
        for component in components[:-1]:
            try:
                following = os.open(
                    component,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=current,
                )
            except OSError as exc:
                raise EngineJobCLIIOError(
                    f"{label} parent contains an unavailable or symbolic component"
                ) from exc
            os.close(current)
            current = following
        return current, components[-1]
    except BaseException:
        os.close(current)
        raise


def _read_regular_file(value: object, *, label: str, maximum_bytes: int) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise EngineJobCLIIOError("maximum_bytes must be a positive exact integer")
    parent_fd, leaf = _open_parent(value, label)
    anchor: int | None = None
    descriptor: int | None = None
    try:
        if not hasattr(os, "O_PATH") or not os.path.isdir("/proc/self/fd"):
            raise EngineJobCLIIOError(
                "Engine Job CLI requires Linux O_PATH and /proc/self/fd custody"
            )
        try:
            anchor = os.open(
                leaf,
                os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise EngineJobCLIIOError(
                f"{label} is unavailable or is a symbolic link"
            ) from exc
        before = os.fstat(anchor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise EngineJobCLIIOError(
                f"{label} must be a nonempty regular file within its byte ceiling"
            )
        try:
            descriptor = os.open(
                f"/proc/self/fd/{anchor}",
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOCTTY", 0),
            )
        except OSError as exc:
            raise EngineJobCLIIOError(
                f"{label} pinned regular file could not be opened for reading"
            ) from exc
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise EngineJobCLIIOError(f"{label} inode custody changed before reading")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise EngineJobCLIIOError(f"{label} exceeds its byte ceiling")
        after = os.fstat(descriptor)
        anchored_after = os.fstat(anchor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_anchored_after = (
            anchored_after.st_dev,
            anchored_after.st_ino,
            anchored_after.st_size,
            anchored_after.st_mtime_ns,
            anchored_after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_before != identity_anchored_after
            or total != after.st_size
        ):
            raise EngineJobCLIIOError(f"{label} changed while it was being read")
        return b"".join(chunks)
    except EngineJobCLIIOError:
        raise
    except OSError as exc:
        raise EngineJobCLIIOError(f"{label} could not be read safely") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if anchor is not None:
            try:
                os.close(anchor)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass


class _HeldOutput:
    __slots__ = ("parent_fd", "leaf", "closed")

    def __init__(self, value: object) -> None:
        parent_fd, leaf = _open_parent(value, "output")
        try:
            parent_stat = os.fstat(parent_fd)
            if (
                parent_stat.st_uid != os.geteuid()
                or parent_stat.st_mode & 0o022
            ):
                raise EngineJobCLIIOError(
                    "output parent must be owner-controlled and not group/world writable"
                )
            try:
                encoded = leaf.encode("ascii")
            except UnicodeEncodeError as exc:
                raise EngineJobCLIIOError(
                    "output leaf must contain only portable ASCII"
                ) from exc
            try:
                name_max = os.fpathconf(parent_fd, "PC_NAME_MAX")
            except (OSError, ValueError) as exc:
                raise EngineJobCLIIOError(
                    "output parent name limit could not be established"
                ) from exc
            if (
                len(encoded) > name_max
                or not encoded
                or not leaf[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in leaf
                )
            ):
                raise EngineJobCLIIOError(
                    "output leaf must be normalized portable ASCII"
                )
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise EngineJobCLIIOError(
                    "output target could not be checked safely"
                ) from exc
            else:
                raise EngineJobCLIIOError("output target already exists")
        except EngineJobCLIIOError:
            try:
                os.close(parent_fd)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                os.close(parent_fd)
            except OSError:
                pass
            raise EngineJobCLIIOError(
                "output parent could not be retained safely"
            ) from exc
        except BaseException:
            try:
                os.close(parent_fd)
            except OSError:
                pass
            raise
        self.parent_fd = parent_fd
        self.leaf = leaf
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            try:
                os.close(self.parent_fd)
            except OSError:
                pass
            self.closed = True

    def _unlink_if_identity(self, name: str, identity: tuple[int, int]) -> None:
        try:
            observed = os.stat(
                name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            return
        if (observed.st_dev, observed.st_ino) == identity:
            try:
                os.unlink(name, dir_fd=self.parent_fd)
            except OSError:
                pass

    def _unlink_required_identity(
        self,
        name: str,
        identity: tuple[int, int],
    ) -> None:
        observed = os.stat(
            name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        if (observed.st_dev, observed.st_ino) != identity:
            raise EngineJobCLIIOError(
                "output staging identity changed before publication commit"
            )
        os.unlink(name, dir_fd=self.parent_fd)

    def publish(self, payload: bytes) -> None:
        if self.closed:
            raise EngineJobCLIIOError("output directory custody is closed")
        if type(payload) is not bytes or not payload:
            raise EngineJobCLIIOError("published payload must be nonempty exact bytes")
        temporary = f".jxengine-{secrets.token_hex(16)}.tmp"
        descriptor: int | None = None
        temporary_created = False
        temporary_identity: tuple[int, int] | None = None
        linked = False
        committed = False
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=self.parent_fd,
            )
            temporary_created = True
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != 0:
                raise EngineJobCLIIOError("output staging file is not new and regular")
            temporary_identity = (opened.st_dev, opened.st_ino)
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise EngineJobCLIIOError("output staging write made no progress")
                written += count
            os.fsync(descriptor)
            staged = os.fstat(descriptor)
            if (
                (staged.st_dev, staged.st_ino) != temporary_identity
                or staged.st_size != len(payload)
                or not stat.S_ISREG(staged.st_mode)
            ):
                raise EngineJobCLIIOError("output staging identity is inconsistent")
            os.close(descriptor)
            descriptor = None
            os.link(
                temporary,
                self.leaf,
                src_dir_fd=self.parent_fd,
                dst_dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            linked = True
            final_descriptor = os.open(
                self.leaf,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=self.parent_fd,
            )
            try:
                final = os.fstat(final_descriptor)
            finally:
                os.close(final_descriptor)
            if (
                (final.st_dev, final.st_ino) != temporary_identity
                or final.st_size != len(payload)
                or not stat.S_ISREG(final.st_mode)
            ):
                raise EngineJobCLIIOError("published output identity changed")
            # The final name and temporary name are hard links to the same
            # complete inode.  Removing the staging name is part of the
            # transaction: success must expose exactly one output path.
            self._unlink_required_identity(temporary, temporary_identity)
            temporary_created = False
            os.fsync(self.parent_fd)
            committed = True
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise EngineJobCLIIOError("output target already exists") from exc
            raise EngineJobCLIIOError("output could not be published atomically") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_identity is not None:
                if linked and not committed:
                    self._unlink_if_identity(self.leaf, temporary_identity)
                if not committed:
                    self._unlink_if_identity(temporary, temporary_identity)
                    try:
                        os.fsync(self.parent_fd)
                    except OSError:
                        raise EngineJobCLIIOError(
                            "output rollback could not be synchronized"
                        )
            elif temporary_created:
                try:
                    os.unlink(temporary, dir_fd=self.parent_fd)
                except OSError:
                    pass


def list_engine_routes_cli(_args: argparse.Namespace) -> int:
    rows = []
    for route in list_integrator_routes():
        rows.append(
            {
                "engine_job_v1": route.integrator_id == ADAPTIVE_RKF78_METHOD_ID,
                "integrator_id": route.integrator_id,
                "portable_manifest_backend_ids": list(
                    route.portable_manifest_backend_ids
                ),
                "production_authorized": route.production_authorized,
                "scientific_claim_state": route.scientific_claim_state,
                "supported_backend_ids": list(route.supported_backend_ids),
            }
        )
    print(
        json.dumps(
            {
                "engine_job_bundle_schema": ENGINE_JOB_BUNDLE_SCHEMA,
                "engine_job_v1_execution_policy": ENGINE_JOB_V1_EXECUTION_POLICY,
                "routes": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return ENGINE_JOB_CLI_SUCCESS


def run_engine_job_cli(args: argparse.Namespace) -> int:
    output = _HeldOutput(args.output)
    try:
        manifest_bytes = _read_regular_file(
            args.manifest,
            label="manifest",
            maximum_bytes=ENGINE_JOB_V1_MAX_MANIFEST_BYTES,
        )
        bundle = run_scenario_manifest(manifest_bytes, args.manifest_sha256)
        output.publish(bundle.job_bytes)
    finally:
        output.close()
    try:
        print(
            json.dumps(
                {
                    **_public_claim_projection(),
                    "job_sha256": bundle.job_sha256,
                    "manifest_sha256": bundle.manifest_sha256,
                    "run_id": bundle.run_id,
                    "schema": ENGINE_JOB_BUNDLE_SCHEMA,
                    "scientific_claim_state": bundle.scientific_claim_state,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError):
        # Publication is already complete; a closed stdout cannot unpublish it.
        pass
    return ENGINE_JOB_CLI_SUCCESS


def verify_engine_job_cli(args: argparse.Namespace) -> int:
    manifest_bytes = _read_regular_file(
        args.manifest,
        label="manifest",
        maximum_bytes=ENGINE_JOB_V1_MAX_MANIFEST_BYTES,
    )
    job_bytes = _read_regular_file(
        args.job,
        label="job",
        maximum_bytes=ENGINE_JOB_BUNDLE_MAX_BYTES,
    )
    verification = verify_engine_job_bytes(
        manifest_bytes,
        expected_manifest_sha256=args.manifest_sha256,
        job_bytes=job_bytes,
        expected_job_sha256=args.job_sha256,
    )
    print(
        json.dumps(
            {
                **_public_claim_projection(),
                "bundle_exact_bytes_verified": True,
                "expected_job_sha256_source": ENGINE_JOB_EXPECTED_SHA256_SOURCE,
                "job_sha256": verification.job_sha256,
                "manifest_sha256": verification.manifest_sha256,
                "run_id": verification.run_id,
                "schema": ENGINE_JOB_BUNDLE_SCHEMA,
                "scientific_claim_state": verification.scientific_claim_state,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return ENGINE_JOB_CLI_SUCCESS


def dispatch_engine_job_cli(args: argparse.Namespace) -> int:
    try:
        return args.engine_job_func(args)
    except EngineRunnerError as exc:
        print(f"jxplanetx engine: {exc}", file=sys.stderr)
        return ENGINE_JOB_CLI_CONTRACT_FAILURE
    except EngineJobCLIIOError as exc:
        print(f"jxplanetx engine: {exc}", file=sys.stderr)
        return ENGINE_JOB_CLI_IO_FAILURE


__all__ = [
    "ENGINE_JOB_CLI_CONTRACT_FAILURE",
    "ENGINE_JOB_CLI_IO_FAILURE",
    "ENGINE_JOB_CLI_SUCCESS",
    "EngineJobCLIIOError",
    "dispatch_engine_job_cli",
    "list_engine_routes_cli",
    "run_engine_job_cli",
    "verify_engine_job_cli",
]
