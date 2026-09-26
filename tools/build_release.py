#!/usr/bin/env python3
"""Build byte-reproducible JX wheel and source archives from declared inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATE_EPOCH = 1_700_000_000
RELEASE_TOOL_VERSIONS = {
    "packaging": "26.3",
    "wheel": "0.48.0",
}
STAGE_FILES = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "AUTHORS.md",
    "THIRD_PARTY_NOTICES.md",
    "benchmarks/jx_smalln_cpu_rkf78.c",
)
STAGE_DIRECTORIES = ("src/jxplanetx",)
IGNORED_STAGE_NAMES = {"__pycache__", ".pytest_cache"}


class ReleaseBuildError(RuntimeError):
    """Raised when a release artifact cannot be built reproducibly."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_metadata(source_root: Path) -> tuple[str, str]:
    metadata = tomllib.loads(
        (source_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = metadata["project"]["version"]
    requirements = metadata["build-system"]["requires"]
    if (
        type(version) is not str
        or len(requirements) != 1
        or type(requirements[0]) is not str
        or not requirements[0].startswith("setuptools==")
    ):
        raise ReleaseBuildError(
            "release builds require one exact setuptools build-system pin"
        )
    return version, requirements[0].split("==", 1)[1]


def verify_backend(source_root: Path) -> tuple[str, str]:
    version, required_backend = project_metadata(source_root)
    actual_backend = importlib.metadata.version("setuptools")
    if actual_backend != required_backend:
        raise ReleaseBuildError(
            "setuptools backend mismatch: "
            f"required {required_backend}, running {actual_backend}"
        )
    for distribution, required_version in RELEASE_TOOL_VERSIONS.items():
        actual_version = importlib.metadata.version(distribution)
        if actual_version != required_version:
            raise ReleaseBuildError(
                f"release tool mismatch for {distribution}: "
                f"required {required_version}, running {actual_version}"
            )
    return version, required_backend


def _ignore_stage(directory: str, names: list[str]) -> set[str]:
    del directory
    return {
        name
        for name in names
        if name in IGNORED_STAGE_NAMES
        or name.endswith(".pyc")
        or name.endswith(".pyo")
        or name.endswith(".so")
        or name.endswith(".pyd")
        or name.endswith(".dylib")
        or name.endswith(".egg-info")
    }


def stage_source(source_root: Path, destination: Path) -> None:
    """Copy the intentionally small distribution input boundary."""

    destination.mkdir(mode=0o755, parents=True)
    for relative in STAGE_FILES:
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise ReleaseBuildError(f"required regular stage input is missing: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in STAGE_DIRECTORIES:
        source = source_root / relative
        if not source.is_dir() or source.is_symlink():
            raise ReleaseBuildError(f"required stage directory is missing: {relative}")
        shutil.copytree(source, destination / relative, ignore=_ignore_stage)

    # Git does not preserve group-write bits, while copytree does.  Normalize
    # the complete stage so wheel ZIP metadata is independent of the checkout
    # umask and source-directory permission policy.
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise ReleaseBuildError(f"symbolic link entered release stage: {path}")
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        else:
            raise ReleaseBuildError(f"unsupported release stage entry: {path}")


def _build_environment(source_date_epoch: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "SOURCE_DATE_EPOCH": str(source_date_epoch),
            "TZ": "UTC",
        }
    )
    return environment


def _run_backend(stage: Path, output: Path, source_date_epoch: int) -> None:
    output.mkdir(mode=0o755)
    script = (
        "import setuptools.build_meta as backend, sys; "
        "destination = sys.argv[1]; "
        "backend.build_sdist(destination); "
        "backend.build_wheel(destination)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(output)],
        cwd=stage,
        env=_build_environment(source_date_epoch),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stdout[-4000:]
        raise ReleaseBuildError(f"setuptools backend failed:\n{detail}")


def _safe_tar_name(name: str) -> None:
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ReleaseBuildError(f"unsafe source archive member: {name!r}")


def canonicalize_sdist(path: Path, source_date_epoch: int) -> None:
    """Rewrite an sdist with stable ordering, ownership, modes, and timestamps."""

    members: list[tuple[str, bool, bool, bytes | None]] = []
    seen: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_tar_name(member.name)
            if member.name in seen:
                raise ReleaseBuildError(
                    f"duplicate source archive member: {member.name!r}"
                )
            seen.add(member.name)
            if member.isdir():
                members.append((member.name, True, False, None))
                continue
            if not member.isfile():
                raise ReleaseBuildError(
                    f"unsupported source archive member type: {member.name!r}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ReleaseBuildError(
                    f"cannot read source archive member: {member.name!r}"
                )
            data = extracted.read()
            if len(data) != member.size:
                raise ReleaseBuildError(
                    f"short read for source archive member: {member.name!r}"
                )
            members.append((member.name, False, bool(member.mode & 0o111), data))

    temporary = path.with_name(path.name + ".canonical.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=source_date_epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for name, is_directory, executable, data in sorted(members):
                        member = tarfile.TarInfo(name)
                        member.mtime = source_date_epoch
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.pax_headers = {}
                        if is_directory:
                            member.type = tarfile.DIRTYPE
                            member.mode = 0o755
                            member.size = 0
                            archive.addfile(member)
                        else:
                            assert data is not None
                            member.type = tarfile.REGTYPE
                            member.mode = 0o755 if executable else 0o644
                            member.size = len(data)
                            archive.addfile(member, BytesIO(data))
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_once(
    source_root: Path,
    workspace: Path,
    *,
    source_date_epoch: int,
) -> dict[str, Path]:
    stage = workspace / "source"
    output = workspace / "dist"
    stage_source(source_root, stage)
    _run_backend(stage, output, source_date_epoch)
    wheels = sorted(output.glob("*.whl"))
    sdists = sorted(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseBuildError(
            f"expected one wheel and one sdist, got {wheels!r} and {sdists!r}"
        )
    canonicalize_sdist(sdists[0], source_date_epoch)
    return {"wheel": wheels[0], "sdist": sdists[0]}


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_reproducibly(
    source_root: Path,
    output_directory: Path,
    *,
    source_date_epoch: int,
) -> dict[str, object]:
    version, backend_version = verify_backend(source_root)
    if output_directory.exists() or output_directory.is_symlink():
        raise ReleaseBuildError(
            f"output directory must not already exist: {output_directory}"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jx-release-build-") as temporary:
        temporary_root = Path(temporary)
        first = build_once(
            source_root,
            temporary_root / "first",
            source_date_epoch=source_date_epoch,
        )
        second = build_once(
            source_root,
            temporary_root / "second",
            source_date_epoch=source_date_epoch,
        )
        for kind in ("wheel", "sdist"):
            if first[kind].name != second[kind].name:
                raise ReleaseBuildError(f"{kind} filenames differ across clean builds")
            if first[kind].read_bytes() != second[kind].read_bytes():
                raise ReleaseBuildError(f"{kind} is not byte-reproducible")

        output_directory.mkdir(mode=0o755)
        copied: dict[str, Path] = {}
        for kind, artifact in first.items():
            target = output_directory / artifact.name
            shutil.copyfile(artifact, target)
            target.chmod(0o644)
            os.utime(target, (source_date_epoch, source_date_epoch))
            copied[kind] = target

    manifest: dict[str, object] = {
        "schema": "jxplanetx.reproducible-release-artifacts.v1",
        "version": version,
        "scientific_claim_state": "SCREENING_ONLY",
        "source_date_epoch": source_date_epoch,
        "build_backend": f"setuptools=={backend_version}",
        "independent_clean_builds": 2,
        "byte_identical": {"wheel": True, "sdist": True},
        "artifacts": {kind: _artifact_record(path) for kind, path in copied.items()},
    }
    manifest_path = output_directory / "ARTIFACTS.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o644)
    os.utime(manifest_path, (source_date_epoch, source_date_epoch))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-date-epoch",
        default=DEFAULT_SOURCE_DATE_EPOCH,
        type=int,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.source_date_epoch < 315_532_800:
        raise ReleaseBuildError("SOURCE_DATE_EPOCH must be at or after 1980-01-01")
    manifest = build_reproducibly(
        ROOT,
        arguments.output_dir.resolve(),
        source_date_epoch=arguments.source_date_epoch,
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
