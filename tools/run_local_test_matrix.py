#!/usr/bin/env python3
"""Run local JX tests in the mutually compatible runtime profiles.

The repository contains source-closed evidence whose dependency and runtime
contracts cannot all be loaded into one Python process.  This runner keeps the
profiles separate, verifies retained binary inputs before using them, and can
reconstruct the exact temporary virtual environment required by the R2 lunar
launcher tests.  It never installs packages or changes frozen run artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_ROOT = ROOT.parents[1] / "library"
SYSTEM_PYTHON = Path("/usr/bin/python3")
LOCAL_VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
REFERENCE_RUNTIME_ROOT = Path("/tmp/jx-reference-core-runtime")
MANIFEST_PATH = Path(__file__).with_name("test_matrix_manifest.json")
MANIFEST_SCHEMA = "jx.test-profile-manifest.v1"


@dataclass(frozen=True)
class LockedFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def resolve(self, library_root: Path) -> Path:
        path = library_root / self.relative_path
        if not path.is_file():
            raise RuntimeError(f"required retained file is missing: {path}")
        size = path.stat().st_size
        if size != self.size_bytes:
            raise RuntimeError(
                f"retained file size mismatch for {path}: "
                f"expected {self.size_bytes}, got {size}"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256 != self.sha256:
            raise RuntimeError(
                f"retained file hash mismatch for {path}: "
                f"expected {self.sha256}, got {actual_sha256}"
            )
        return path


@dataclass(frozen=True)
class TestLane:
    name: str
    python: Path
    targets: tuple[str, ...]
    pythonpath: tuple[Path, ...] = field(default_factory=tuple)
    environment: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TestProfile:
    profile_id: str
    disposition: str
    interpreter: str
    isolation: str
    ci: bool
    description: str
    requirements: tuple[str, ...]
    files: tuple[str, ...]


COUPLED_CORE_MPMATH_WHEEL = LockedFile(
    relative_path=(
        "jx-ultra-private-lunar-coupled-core-endpoint-metric-resolution-"
        "oracle-r2-final-execution-prereg-20260912/checkpoint/inputs/"
        "high-precision/mpmath-1.3.0-py3-none-any.whl"
    ),
    size_bytes=536_198,
    sha256="a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
)

TORQUE_MPMATH_WHEEL = LockedFile(
    relative_path=(
        "jx-ultra-private-lunar-torque-chebyshev-derivative-oracle-"
        "confirmation-supplement-20260911/checkpoint/inputs/high-precision/"
        "mpmath-1.3.0-py3-none-any.whl"
    ),
    size_bytes=536_198,
    sha256="a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
)

SPICEYPY_WHEEL = LockedFile(
    relative_path=(
        "jx-ultra-private-lunar-torque-chebyshev-derivative-oracle-"
        "confirmation-supplement-20260911/checkpoint/inputs/lunar-orientation/"
        "spiceypy-8.2.0-cp314-cp314-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
    ),
    size_bytes=2_442_346,
    sha256="00c2714d3dd67180794ad0b66735d6cac470e19627d6e99f9628a50b52133d90",
)

CSPICE_LIBRARY = LockedFile(
    relative_path=(
        "jx-v5-published-v0.4.0a1-20260906/external/step5-de440/"
        "runtime-root-components/native/libcspice.so"
    ),
    size_bytes=3_561_056,
    sha256="1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, TestProfile]:
    """Load the profile inventory and prove exact, non-overlapping coverage."""

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read test profile manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported test profile manifest schema in {path}")
    if raw.get("inventory_glob") != "tests/test_*.py":
        raise RuntimeError("test profile manifest inventory glob changed")

    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise RuntimeError("test profile manifest has no profiles")

    profiles: dict[str, TestProfile] = {}
    assigned: dict[str, str] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise RuntimeError("test profile entries must be objects")
        profile_id = raw_profile.get("id")
        files = raw_profile.get("files")
        requirements = raw_profile.get("requirements", [])
        if not isinstance(profile_id, str) or not profile_id:
            raise RuntimeError("test profile id must be a non-empty string")
        if profile_id in profiles:
            raise RuntimeError(f"duplicate test profile id: {profile_id}")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise RuntimeError(f"test profile {profile_id} has an invalid file list")
        if files != sorted(files) or len(files) != len(set(files)):
            raise RuntimeError(f"test profile {profile_id} file list is not unique and sorted")
        if not isinstance(requirements, list) or not all(
            isinstance(item, str) for item in requirements
        ):
            raise RuntimeError(f"test profile {profile_id} has invalid requirements")

        required_strings = ("disposition", "interpreter", "isolation", "description")
        if any(not isinstance(raw_profile.get(key), str) for key in required_strings):
            raise RuntimeError(f"test profile {profile_id} metadata is invalid")
        if not isinstance(raw_profile.get("ci"), bool):
            raise RuntimeError(f"test profile {profile_id} CI flag is invalid")
        for relative in files:
            if relative in assigned:
                raise RuntimeError(
                    f"test file {relative} is assigned to both "
                    f"{assigned[relative]} and {profile_id}"
                )
            assigned[relative] = profile_id
        profiles[profile_id] = TestProfile(
            profile_id=profile_id,
            disposition=raw_profile["disposition"],
            interpreter=raw_profile["interpreter"],
            isolation=raw_profile["isolation"],
            ci=raw_profile["ci"],
            description=raw_profile["description"],
            requirements=tuple(requirements),
            files=tuple(files),
        )

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_*.py")
        if path.is_file()
    }
    declared = set(assigned)
    missing = sorted(actual - declared)
    stale = sorted(declared - actual)
    if missing or stale:
        details: list[str] = []
        if missing:
            details.append(f"unassigned={missing}")
        if stale:
            details.append(f"missing_from_tree={stale}")
        raise RuntimeError("test profile coverage mismatch: " + "; ".join(details))
    return profiles


def print_inventory(profiles: dict[str, TestProfile], *, include_files: bool) -> None:
    total = sum(len(profile.files) for profile in profiles.values())
    print(f"{MANIFEST_SCHEMA}: {total} test files in {len(profiles)} profiles")
    for profile in profiles.values():
        print(
            f"{profile.profile_id}: files={len(profile.files)} "
            f"disposition={profile.disposition} ci={str(profile.ci).lower()}"
        )
        if include_files:
            for relative in profile.files:
                print(f"  {relative}")


def require_python(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{label} is not executable: {path}")
    return path


def require_rebound_version(python: Path, expected: str) -> None:
    script = (
        "import importlib.metadata, rebound; "
        "print(importlib.metadata.version('rebound')); "
        "print(rebound.__version__)"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    versions = completed.stdout.splitlines()
    if completed.returncode != 0 or versions != [expected, expected]:
        detail = completed.stderr.strip() or repr(versions)
        raise RuntimeError(
            f"{python} does not provide exact REBOUND {expected}: {detail}"
        )


def require_reboundx_version(python: Path, expected: str) -> None:
    script = (
        "import importlib.metadata, reboundx; "
        "print(importlib.metadata.version('reboundx')); "
        "print(reboundx.__version__)"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    versions = completed.stdout.splitlines()
    if completed.returncode != 0 or versions != [expected, expected]:
        detail = completed.stderr.strip() or repr(versions)
        raise RuntimeError(
            f"{python} does not provide exact REBOUNDx {expected}: {detail}"
        )


def require_spiceypy_version(python: Path, expected: str) -> None:
    script = (
        "import importlib.metadata, spiceypy; "
        "print(importlib.metadata.version('spiceypy'))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    versions = completed.stdout.splitlines()
    if completed.returncode != 0 or versions != [expected]:
        detail = completed.stderr.strip() or repr(versions)
        raise RuntimeError(
            f"{python} does not provide exact SpiceyPy {expected}: {detail}"
        )


def cuda_device_available(python: Path) -> bool:
    script = (
        "import cupy; "
        "raise SystemExit(0 if int(cupy.cuda.runtime.getDeviceCount()) > 0 else 1)"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def ensure_reference_runtime() -> Path:
    """Create the exact ephemeral runtime if absent, then verify its custody."""

    invocation = REFERENCE_RUNTIME_ROOT / "bin" / "python"
    intermediate = REFERENCE_RUNTIME_ROOT / "bin" / "python3"
    pyvenv = REFERENCE_RUNTIME_ROOT / "pyvenv.cfg"

    if not REFERENCE_RUNTIME_ROOT.exists():
        require_python(SYSTEM_PYTHON, "frozen system Python")
        subprocess.run(
            [
                str(SYSTEM_PYTHON),
                "-m",
                "venv",
                "--system-site-packages",
                str(REFERENCE_RUNTIME_ROOT),
            ],
            check=True,
            cwd=ROOT,
        )

    expected_links = ((invocation, "python3"), (intermediate, "/usr/bin/python3"))
    for path, expected_target in expected_links:
        if not path.is_symlink():
            raise RuntimeError(f"frozen runtime link is missing: {path}")
        actual_target = os.readlink(path)
        if actual_target != expected_target:
            raise RuntimeError(
                f"frozen runtime link mismatch for {path}: "
                f"expected {expected_target!r}, got {actual_target!r}"
            )

    if not pyvenv.is_file():
        raise RuntimeError(f"frozen runtime metadata is missing: {pyvenv}")
    expected_size = 191
    expected_sha256 = "e96028a8e3a9cf77a59de7de012ae5baf08f8b00ab783b15187f78c8203722a4"
    actual_size = pyvenv.stat().st_size
    actual_sha256 = file_sha256(pyvenv)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "the existing /tmp/jx-reference-core-runtime is not the frozen "
            "runtime; refusing to replace it automatically"
        )
    return invocation


def base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("CSPICE_SHARED_LIB", "PYTHONOPTIMIZE", "PYTHONPATH"):
        environment.pop(name, None)
    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "LC_ALL": "C",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    return environment


def lane_for_file(
    profile_id: str,
    relative: str,
    python: Path,
    *,
    pythonpath: tuple[Path, ...],
    environment: tuple[tuple[str, str], ...] = (),
) -> TestLane:
    return TestLane(
        name=f"{profile_id}:{Path(relative).name}",
        python=python,
        targets=("discover", "-s", "tests", "-p", Path(relative).name),
        pythonpath=pythonpath,
        environment=environment,
    )


def build_lanes(
    requested: str,
    library_root: Path,
    profiles: dict[str, TestProfile],
    *,
    runtime_python: Path | None,
) -> list[TestLane]:
    source_path = ROOT / "src"
    current_python = require_python(Path(sys.executable), "current Python")
    selected: list[str]
    if requested == "engine":
        files = tuple(
            relative
            for relative in profiles["live-current"].files
            if Path(relative).name.startswith("test_engine")
        )
        return [
            lane_for_file(
                "engine",
                relative,
                current_python,
                pythonpath=(source_path,),
            )
            for relative in files
        ]
    if requested == "live":
        selected = ["live-current", "cuda-optional", "cuda-hardware-only"]
    elif requested == "ci-generic":
        selected = ["live-current", "cuda-optional"]
    elif requested in {"all", "all-compatible"}:
        selected = [
            "live-current",
            "cuda-optional",
            "cuda-hardware-only",
            "rebound-5.1.1",
            "gr15-eih-external",
            "performance-local",
            "library-catalog-local",
            "source-closed-reph",
            "retained-lunar-coupled",
            "retained-lunar-torque-spice",
            "exact-lunar-launcher",
        ]
    elif requested == "evidence":
        selected = [
            "rebound-5.1.1",
            "gr15-eih-external",
            "performance-local",
            "library-catalog-local",
            "source-closed-reph",
            "retained-lunar-coupled",
            "retained-lunar-torque-spice",
            "exact-lunar-launcher",
        ]
    elif requested in profiles:
        selected = [requested]
    else:
        choices = sorted(
            set(profiles) | {"all", "all-compatible", "ci-generic", "engine", "evidence", "live"}
        )
        raise RuntimeError(
            f"unknown test profile {requested!r}; choose one of {', '.join(choices)}"
        )

    non_runnable = {
        "frozen-history": "replay from its matching frozen source snapshot",
        "superseded-preregistration": "inspect as superseded evidence state",
    }
    for profile_id in selected:
        if profile_id in non_runnable:
            raise RuntimeError(
                f"profile {profile_id} is accounted but not live-runnable; "
                f"{non_runnable[profile_id]}"
            )

    lanes: list[TestLane] = []
    evidence_python = runtime_python
    if evidence_python is None and any(
        profile_id
        in {
            "rebound-5.1.1",
            "gr15-eih-external",
            "performance-local",
            "source-closed-reph",
        }
        for profile_id in selected
    ):
        evidence_python = (
            LOCAL_VENV_PYTHON if LOCAL_VENV_PYTHON.is_file() else current_python
        )

    for profile_id in selected:
        profile = profiles[profile_id]
        if profile_id in {"live-current", "cuda-optional"}:
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    current_python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "cuda-hardware-only":
            if not cuda_device_available(current_python):
                print(
                    "Skipping cuda-hardware-only: no CUDA device is visible to "
                    f"{current_python}.",
                    flush=True,
                )
                continue
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    current_python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "rebound-4.4.11":
            python = require_python(runtime_python or current_python, "REBOUND 4.4.11 Python")
            require_rebound_version(python, "4.4.11")
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "rebound-5.1.1":
            if evidence_python is None:
                raise RuntimeError("REBOUND 5.1.1 Python was not selected")
            python = require_python(evidence_python, "REBOUND 5.1.1 Python")
            require_rebound_version(python, "5.1.1")
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "gr15-eih-external":
            if evidence_python is None:
                raise RuntimeError("GR15-EIH external Python was not selected")
            python = require_python(evidence_python, "GR15-EIH external Python")
            require_rebound_version(python, "5.1.1")
            require_reboundx_version(python, "5.1.0")
            require_spiceypy_version(python, "8.2.0")
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "performance-local":
            if evidence_python is None:
                raise RuntimeError("local performance evidence Python was not selected")
            python = require_python(evidence_python, "local performance evidence Python")
            require_rebound_version(python, "5.1.1")
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "library-catalog-local":
            authority = library_root / "JX_AUTHORITY_CATALOG.json"
            if not authority.is_file():
                raise RuntimeError(
                    f"local library authority catalog is missing: {authority}"
                )
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    current_python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "source-closed-reph":
            if evidence_python is None:
                raise RuntimeError("source-closed REPH Python was not selected")
            python = require_python(evidence_python, "source-closed REPH Python")
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        elif profile_id == "retained-lunar-coupled":
            python = require_python(SYSTEM_PYTHON, "frozen system Python")
            mpmath = COUPLED_CORE_MPMATH_WHEEL.resolve(library_root)
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path, mpmath),
                )
                for relative in profile.files
            )
        elif profile_id == "retained-lunar-torque-spice":
            python = require_python(SYSTEM_PYTHON, "frozen system Python")
            mpmath = TORQUE_MPMATH_WHEEL.resolve(library_root)
            spiceypy = SPICEYPY_WHEEL.resolve(library_root)
            cspice = CSPICE_LIBRARY.resolve(library_root)
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path, mpmath, spiceypy),
                    environment=(("CSPICE_SHARED_LIB", str(cspice)),),
                )
                for relative in profile.files
            )
        elif profile_id == "exact-lunar-launcher":
            python = ensure_reference_runtime()
            lanes.extend(
                lane_for_file(
                    profile_id,
                    relative,
                    python,
                    pythonpath=(source_path,),
                )
                for relative in profile.files
            )
        else:
            raise RuntimeError(f"runner has no implementation for profile {profile_id}")
    return lanes


def run_lane(lane: TestLane, *, quiet: bool) -> int:
    environment = base_environment()
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in lane.pythonpath)
    environment.update(dict(lane.environment))
    verbosity = "-q" if quiet else "-v"
    command = [str(lane.python), "-m", "unittest", *lane.targets, verbosity]
    print(f"\n[{lane.name}] {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return result.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="engine",
        help=(
            "profile or aggregate to run: engine, live, ci-generic, evidence, "
            "all-compatible, or a manifest profile id (default: engine)"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"test inventory manifest (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=DEFAULT_LIBRARY_ROOT,
        help=f"retained cloud-library mirror (default: {DEFAULT_LIBRARY_ROOT})",
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        help="Python executable for REBOUND and source-closed runtime profiles",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate exact manifest coverage without running tests",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list profile counts and dispositions without running tests",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="include every assigned test file in --list output",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="quiet unittest output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profiles = load_manifest(args.manifest.resolve())
        if args.list or args.list_files:
            print_inventory(profiles, include_files=args.list_files)
            return 0
        if args.validate_only:
            total = sum(len(profile.files) for profile in profiles.values())
            print(
                f"Test profile manifest is valid: {total} files, "
                f"{len(profiles)} profiles, exact non-overlapping coverage."
            )
            return 0
        lanes = build_lanes(
            args.profile,
            args.library_root.resolve(),
            profiles,
            runtime_python=(
                args.runtime_python.absolute()
                if args.runtime_python is not None
                else None
            ),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"test matrix setup failed: {error}", file=sys.stderr)
        return 2

    failed: list[str] = []
    for lane in lanes:
        if run_lane(lane, quiet=args.quiet) != 0:
            failed.append(lane.name)

    print(f"\nCompleted {len(lanes)} isolated test lane(s).", flush=True)
    if failed:
        print(f"Failed lanes: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All selected lanes passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
