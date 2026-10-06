#!/usr/bin/env python3
"""Reproducible, external-only qualification harness for GENGA 3.208.

This module deliberately treats GENGA as a separately built external
executable.  Its binary and optional build receipt (including any compatibility
patch) are evidence inputs; GENGA is never imported or bundled into JX.
The toy experiment qualifies test-particle isolation and binary restart
behavior.  The JX adapter only prepares an engineering workload from the
tracked four-arm-screen inputs.  Neither path is a Planet Nine, flyby, Rubin,
primordial-population, or 4.5-Gyr scientific result.

The harness is fail-closed: run directories are create-only, input identities
are committed before execution, a zero process return code is insufficient
without GENGA's terminal success message, and all evidence files are hashed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "jxplanetx.genga-external-qualification.v1"
PROTOCOL_SCHEMA = "jxplanetx.genga-external-protocol.v1"
ARTIFACT_SCHEMA = "jxplanetx.genga-external-artifacts.v1"
RESULT_SCHEMA = "jxplanetx.genga-external-result.v1"
JX_ADAPTER_SCHEMA = "jxplanetx.genga-jx-adapter.v1"

# Constants are copied exactly from GENGA 3.208 source/define.h.  They are
# explicit here so that an adapter result does not depend on an ambient build.
GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN = 2.959122082855911e-4
GENGA_DAY_UNIT = 0.01720209895
JX_DAYS_PER_YEAR = 365.25

TRACKED_SOURCE_STATE = (
    ROOT
    / "runs/cuda_four_arm_screen_v1/qualification/arms/control/SOURCE_STATE.npz"
)
TRACKED_TRACER_STATE = (
    ROOT / "runs/cuda_four_arm_screen_v1/qualification/TRANSFORMED_TRACERS.npz"
)
TRACKED_SOURCE_SHA256 = (
    "92c5f14b04027d4d72d62402a4bf3279ef74117b0f716a13e97a276d98510aec"
)
TRACKED_TRACER_SHA256 = (
    "f4ceb3b811a97b2f82f8017e788b2d74b91b7c6a0a935e8b64dbd08dffb6338f"
)
TRACKED_SOURCE_IDS = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
JX_SOURCE_BODY_IDS = (0, 1, 2, 3, 4)
JX_TRACER_ID_OFFSET = 5
JX_EXPECTED_TRACER_COUNT = 16_800
JX_DEFAULT_QUALIFICATION_PREFIX = 256
JX_FROZEN_QUALIFICATION_STEPS = 1_024

TOY_SOURCE_IDS = (100, 101)
TOY_TRACER_ID = 1_000_000
TOY_FINITE_TRACER_MASS = 4.0e-4
TOY_PARTICLE_MINIMUM_MASS = 5.0e-4
TOY_MIDPOINT_STEP = 32
TOY_ENDPOINT_STEP = 64
TOY_PAIRWISE_ABS_TOLERANCE = 2.0e-13
TOY_FEEDBACK_DETECTION_FLOOR = 1.0e-15
TOY_CONSERVATION_LIMIT = 1.0e-8
TERMINAL_SUCCESS_TEXT = "GENGA terminated successfully"

OUTPUT_FIELDS = (
    "t",
    "i",
    "m",
    "r",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "Sx",
    "Sy",
    "Sz",
    "amin",
    "amax",
    "emin",
    "emax",
    "aec",
    "aecT",
    "encc",
    "test",
)

# GENGA writes each field separately with fwrite; this is therefore an
# unaligned 140-byte little-endian record, not an ABI-aligned C struct.
GENGA_BINARY_DTYPE = np.dtype(
    [
        ("t", "<f8"),
        ("i", "<i4"),
        ("m", "<f8"),
        ("r", "<f8"),
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("vx", "<f8"),
        ("vy", "<f8"),
        ("vz", "<f8"),
        ("Sx", "<f8"),
        ("Sy", "<f8"),
        ("Sz", "<f8"),
        ("amin", "<f4"),
        ("amax", "<f4"),
        ("emin", "<f4"),
        ("emax", "<f4"),
        ("aec", "<f4"),
        ("aecT", "<f4"),
        ("encc", "<u8"),
        ("test", "<f8"),
    ],
    align=False,
)
if GENGA_BINARY_DTYPE.itemsize != 140:  # pragma: no cover - import-time guard
    raise RuntimeError("unexpected GENGA binary record size")

TUNING_PARAMETERS = """\
FTX 64
RTX 128
KP 1
KTX 1
KTY 128
KP2 1
KTX2 1
KTY2 256
FrTX 128
UseAcc 1
UseBVH 1
"""


@dataclass(frozen=True)
class Body:
    body_id: int
    mass_msun: float
    radius_au: float
    position_au: tuple[float, float, float]
    velocity_genga: tuple[float, float, float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_new_text(path, text)


def _require_fresh_directory(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _artifact_inventory(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "inventory_sha256": _canonical_sha256(files),
    }


def _verify_artifact_inventory(root: Path, manifest_path: Path) -> None:
    """Fail closed unless a prepared tree exactly matches its committed manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("prepared artifact manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("prepared artifact manifest has no file list")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in files:
        relative = str(item.get("path", ""))
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact path: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate artifact path: {relative}")
        expected[relative] = item
        target = root / relative_path
        if not target.is_file():
            raise ValueError(f"prepared artifact is missing: {relative}")
        if target.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"prepared artifact size changed: {relative}")
        if _sha256_file(target) != item["sha256"]:
            raise ValueError(f"prepared artifact hash changed: {relative}")
    if _canonical_sha256(files) != manifest.get("inventory_sha256"):
        raise ValueError("prepared artifact inventory digest mismatch")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        raise ValueError("prepared artifact roster changed before execution")


def _runner_identity() -> dict[str, str]:
    runner = Path(__file__).resolve()
    test_source = ROOT / "tests/test_jx_genga_external_qualification.py"
    identity = {"path": str(runner), "sha256": _sha256_file(runner)}
    if test_source.is_file():
        identity["test_path"] = str(test_source.resolve())
        identity["test_sha256"] = _sha256_file(test_source)
    return identity


def _engine_identity(engine: Path) -> dict[str, str]:
    identity = {"path": str(engine), "sha256": _sha256_file(engine)}
    build_receipt = engine.parent.parent / "JX_BUILD_RECEIPT.json"
    if build_receipt.is_file():
        identity["build_receipt_path"] = str(build_receipt.resolve())
        identity["build_receipt_sha256"] = _sha256_file(build_receipt)
    return identity


def _runtime_identity(engine: Path) -> dict[str, Mapping[str, str]]:
    """Snapshot every local executable/source identity that affects a run."""

    return {"engine": _engine_identity(engine), "runner": _runner_identity()}


def gm_au3_per_year2_to_msun(gm: np.ndarray | float) -> np.ndarray:
    """Convert JX AU^3/year^2 GM to the GENGA solar-mass convention."""

    value = np.asarray(gm, dtype=np.float64)
    return value / (
        GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN * JX_DAYS_PER_YEAR**2
    )


def velocity_au_per_year_to_genga(velocity: np.ndarray | float) -> np.ndarray:
    """Convert physical AU/year to GENGA's stored velocity number.

    A stored GENGA velocity multiplied by ``dayUnit`` is in AU/day.
    """

    value = np.asarray(velocity, dtype=np.float64)
    return value / (JX_DAYS_PER_YEAR * GENGA_DAY_UNIT)


def render_initial_conditions(bodies: Sequence[Body]) -> str:
    seen: set[int] = set()
    lines: list[str] = []
    for body in bodies:
        if body.body_id in seen:
            raise ValueError(f"duplicate body id: {body.body_id}")
        if body.body_id < 0 or body.body_id > np.iinfo(np.int32).max:
            raise ValueError(f"body id outside GENGA int32 range: {body.body_id}")
        values = (
            body.body_id,
            body.mass_msun,
            body.radius_au,
            *body.position_au,
            *body.velocity_genga,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite initial condition for body {body.body_id}")
        seen.add(body.body_id)
        lines.append(
            f"{body.body_id:d} "
            + " ".join(format(float(value), ".17g") for value in values[1:])
        )
    return "\n".join(lines) + "\n"


def render_param(
    *,
    output_name: str,
    input_name: str,
    central_mass_msun: float,
    use_test_particles: int,
    particle_minimum_mass: float,
    integration_steps: int,
    timestep_days: float,
    coordinate_interval: int,
    energy_interval: int,
    minimum_massive_bodies: int,
    outer_radius_au: float,
) -> str:
    if not output_name.isalnum():
        raise ValueError("output_name must be alphanumeric")
    if use_test_particles not in (0, 1):
        raise ValueError("this qualification only supports TP0 and TP1")
    if integration_steps <= 0 or coordinate_interval == 0 or energy_interval == 0:
        raise ValueError("invalid integration or output interval")
    if outer_radius_au <= 50.0:
        raise ValueError("outer radius must explicitly exceed GENGA's unsafe 50 AU default")
    lines = [
        f"Time step in days = {timestep_days:.17g}",
        f"Output name = {output_name}",
        f"Energy output interval = {energy_interval:d}",
        f"Coordinates output interval = {coordinate_interval:d}",
        "Number of outputs per interval = 1",
        "Coordinate output buffer = 1",
        "Irregular output calendar = -",
        f"Integration steps = {integration_steps:d}",
        f"Central Mass = {central_mass_msun:.17g}",
        "n1 = 3.0",
        "n2 = 0.4",
        f"Input file = {input_name}",
        "Input file Format: << i m r x y z vx vy vz >>",
        "Output file Format: << " + " ".join(OUTPUT_FIELDS) + " >>",
        "Use output binary files = 1",
        "Angle units = radians",
        "Default rho = 2.0",
        f"Use Test Particles = {use_test_particles:d}",
        f"Particle Minimum Mass = {particle_minimum_mass:.17g}",
        "Restart timestep = 0",
        "Symplectic recursion levels = 1",
        "Symplectic recursion sub steps = 2",
        f"Minimum number of bodies = {minimum_massive_bodies:d}",
        "Minimum number of test particles = 0",
        "Inner truncation radius = 0.05",
        f"Outer truncation radius = {outer_radius_au:.17g}",
        "Order of integrator = 2",
        "Maximum encounter pairs = 1024",
        "Star Radius = 1.0e-12",
        "J2 = 0.0",
        "J2 radius = 0.0",
        "Use aeGrid = 0",
        "Use gas disk = 0",
        "Use force = 0",
        "Use GR = 0",
        "Use Tides = 0",
        "Use Rotational Deformation = 0",
        "Use Yarkovsky = 0",
        "Use Poynting-Robertson = 0",
        "Use Small Collisions = 0",
        "Create Particles file name = -",
        "Set Elements file name = -",
        "FormatS = 0",
        "FormatT = 0",
        "FormatP = 1",
        "FormatO = 0",
        "Print Transits = 0",
        "Print RV = 0",
        "Print MCMC = 0",
        "Report Encounters = 0",
        "Stop at Encounter = 0",
        "Stop at Collision = 1",
        "Stop Minimum Mass = 0.0",
        "Collision Model = 0",
        "Serial Grouping = 1",
        "Do kernel tuning = 0",
        "Do Kick in single precision = 0",
    ]
    return "\n".join(lines) + "\n"


def _toy_sources() -> tuple[Body, Body]:
    # Deliberately non-symmetric, well-separated massive bodies.  A non-zero
    # z component on the second body exercises all phase-space columns.
    return (
        Body(100, 1.0e-3, 1.0e-12, (5.2, 0.1, 0.02), (-0.01, 0.438, 0.003)),
        # Keep every source strictly above the finite-TP classification
        # threshold (5e-4); otherwise GENGA silently moves it into the test
        # population and the two-source pairwise control is no longer valid.
        Body(101, 8.0e-4, 1.0e-12, (-8.7, 1.3, -0.08), (-0.052, -0.331, 0.007)),
    )


def _toy_tracer(mass_msun: float) -> Body:
    # vz=0.4 is intentional: it guards against an adapter/parser that silently
    # drops the third velocity component.
    return Body(
        TOY_TRACER_ID,
        mass_msun,
        1.0e-12,
        (35.0, -2.0, 1.0),
        (0.01, 0.165, 0.4),
    )


def toy_protocol(engine: Path, device: int) -> dict[str, Any]:
    lanes = {
        "base": {"tp_mode": 0, "min_mass": 0.0, "tracer_mass": None},
        "zero_tp1": {"tp_mode": 1, "min_mass": 0.0, "tracer_mass": 0.0},
        "finite_tp1": {
            "tp_mode": 1,
            "min_mass": TOY_PARTICLE_MINIMUM_MASS,
            "tracer_mass": TOY_FINITE_TRACER_MASS,
        },
        "finite_tp0": {
            "tp_mode": 0,
            "min_mass": TOY_PARTICLE_MINIMUM_MASS,
            "tracer_mass": TOY_FINITE_TRACER_MASS,
        },
        "continuous_tp1": {
            "tp_mode": 1,
            "min_mass": 0.0,
            "tracer_mass": 0.0,
        },
        "split_tp1": {
            "tp_mode": 1,
            "min_mass": 0.0,
            "tracer_mass": 0.0,
            "split_at_step": TOY_MIDPOINT_STEP,
        },
    }
    return {
        "schema": PROTOCOL_SCHEMA,
        "qualification": "TP1_QUIET_PATH_NO_BACKREACTION_AND_BINARY_RESTART",
        "scientific_scope": "ENGINEERING_ONLY_NO_ASTROPHYSICAL_INFERENCE",
        "engine": _engine_identity(engine),
        "runner": _runner_identity(),
        "device": device,
        "genga_constants": {
            "def_Kg_AU3_day2_Msun": GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN,
            "dayUnit": GENGA_DAY_UNIT,
        },
        "fixed_numerics": {
            "steps": TOY_ENDPOINT_STEP,
            "restart_step": TOY_MIDPOINT_STEP,
            "timestep_days": 0.25,
            "order": 2,
            "symplectic_recursion_levels": 1,
            "symplectic_recursion_sub_steps": 2,
            "serial_grouping": 1,
            "kernel_tuning": 0,
            "kick_single_precision": 0,
            "binary_record_size": GENGA_BINARY_DTYPE.itemsize,
            "tuning_parameters": TUNING_PARAMETERS.splitlines(),
        },
        "lanes": lanes,
        "gates": {
            "all_invocations_return_zero_and_print_terminal_success": True,
            "engine_runner_and_test_identities_unchanged": True,
            "zero_mass_tp1_raw_source_records_bitwise_equal_to_base": True,
            "finite_positive_tp1_frame_transform_limitation_detected": {
                "operator": ">",
                "threshold": TOY_FEEDBACK_DETECTION_FLOOR,
                "interpretation": (
                    "expected GENGA 3.208 limitation: positive-mass TP1 bodies enter "
                    "HelioToDemo/DemoToHelio mass transforms; only exact-zero tracers "
                    "are qualified for production"
                ),
            },
            "finite_tp1_pairwise_source_phase_space_max_abs": {
                "classification": "DIAGNOSTIC_ONLY_NOT_A_PRODUCTION_GATE",
                "interpretation": (
                    "positive-mass TP1 is unsupported; retain the pairwise delta "
                    "to characterize the external code, never to authorize JX"
                ),
            },
            "finite_tp0_pairwise_feedback_max_abs": {
                "operator": ">",
                "threshold": TOY_FEEDBACK_DETECTION_FLOOR,
            },
            "continuous_split_binary_endpoint_bitwise_equal": True,
            "info_echo_body_classification_matches_protocol": True,
            "absolute_relative_energy_and_angular_momentum": {
                "operator": "<=",
                "threshold": TOY_CONSERVATION_LIMIT,
                "excludes": ["finite_tp1"],
            },
            "no_collisions_or_ejections": True,
        },
    }


def _prepare_lane(
    lane_dir: Path,
    *,
    lane_name: str,
    tp_mode: int,
    min_mass: float,
    tracer_mass: float | None,
    steps: int,
    timestep_days: float,
    outer_radius_au: float,
) -> None:
    lane_dir.mkdir(parents=True, exist_ok=False)
    bodies: tuple[Body, ...] = _toy_sources()
    if tracer_mass is not None:
        bodies += (_toy_tracer(tracer_mass),)
    _write_new_text(lane_dir / "initial.dat", render_initial_conditions(bodies))
    _write_new_text(
        lane_dir / "param.dat",
        render_param(
            output_name=lane_name.replace("_", ""),
            input_name="initial.dat",
            central_mass_msun=1.0,
            use_test_particles=tp_mode,
            particle_minimum_mass=min_mass,
            integration_steps=steps,
            timestep_days=timestep_days,
            coordinate_interval=TOY_MIDPOINT_STEP,
            energy_interval=TOY_MIDPOINT_STEP,
            minimum_massive_bodies=2,
            outer_radius_au=outer_radius_au,
        ),
    )
    _write_new_text(lane_dir / "tuningParameters.dat", TUNING_PARAMETERS)


def prepare_toy(output_root: Path, engine: Path, device: int) -> tuple[Path, dict[str, Any]]:
    engine = engine.expanduser().resolve(strict=True)
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise ValueError(f"GENGA executable is not executable: {engine}")
    root = _require_fresh_directory(output_root)
    protocol = toy_protocol(engine, device)
    _write_new_json(root / "PROTOCOL.json", protocol)
    raw = root / "raw"
    raw.mkdir()
    for name, lane in protocol["lanes"].items():
        _prepare_lane(
            raw / name,
            lane_name=name,
            tp_mode=int(lane["tp_mode"]),
            min_mass=float(lane["min_mass"]),
            tracer_mass=lane["tracer_mass"],
            steps=TOY_ENDPOINT_STEP,
            timestep_days=0.25,
            outer_radius_au=10_000.0,
        )
    return root, protocol


def terminal_success(returncode: int, stdout: str, stderr: str) -> bool:
    return returncode == 0 and TERMINAL_SUCCESS_TEXT in (stdout + "\n" + stderr)


def _genga_command(
    engine: Path, device: int, steps: int, restart_step: int | None = None
) -> list[str]:
    # GENGA 3.208's ``-dev`` parser consumes its device values by advancing
    # the outer argv cursor internally.  Any option placed after ``-dev`` can
    # therefore be skipped and its value misread as an option.  Keep ``-dev``
    # last; this is bound into each command record below.
    command = [str(engine), "-I", str(steps)]
    if restart_step is not None:
        command += ["-R", str(restart_step)]
    command += ["-dev", str(device)]
    return command


def _execute_genga(
    engine: Path,
    lane_dir: Path,
    *,
    device: int,
    steps: int,
    phase: str,
    restart_step: int | None = None,
    timeout_seconds: float = 300.0,
    expected_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_identity_start = _runtime_identity(engine)
    if (
        expected_runtime_identity is not None
        and runtime_identity_start != expected_runtime_identity
    ):
        raise ValueError("engine, build receipt, runner, or test identity changed before lane")
    command = _genga_command(engine, device, steps, restart_step)
    command_record = {
        "argv": command,
        "cwd": str(lane_dir),
        "started_utc": _utc_now(),
        "runtime_identity_start": runtime_identity_start,
    }
    _write_new_json(lane_dir / f"{phase}.command.json", command_record)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=lane_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        timed_out = False
        stdout = process.stdout
        stderr = process.stderr
        returncode: int | None = process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        returncode = None
    elapsed = time.perf_counter() - started
    _write_new_text(lane_dir / f"{phase}.stdout.txt", stdout)
    _write_new_text(lane_dir / f"{phase}.stderr.txt", stderr)
    identity_error: str | None = None
    try:
        runtime_identity_end: Mapping[str, Any] | None = _runtime_identity(engine)
    except Exception as error:
        runtime_identity_end = None
        identity_error = f"{type(error).__name__}: {error}"
    runtime_identity_unchanged = (
        runtime_identity_end == runtime_identity_start
        and (
            expected_runtime_identity is None
            or runtime_identity_end == expected_runtime_identity
        )
    )
    success = (
        not timed_out
        and returncode is not None
        and terminal_success(returncode, stdout, stderr)
        and runtime_identity_unchanged
    )
    return {
        "phase": phase,
        "argv": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "terminal_success_text_present": TERMINAL_SUCCESS_TEXT
        in (stdout + "\n" + stderr),
        "runtime_identity_start": runtime_identity_start,
        "runtime_identity_end": runtime_identity_end,
        "runtime_identity_unchanged": runtime_identity_unchanged,
        "runtime_identity_error": identity_error,
        "success": success,
    }


def read_binary_output(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size == 0 or size % GENGA_BINARY_DTYPE.itemsize != 0:
        raise ValueError(
            f"invalid GENGA binary output size {size} for {path}; "
            f"record size is {GENGA_BINARY_DTYPE.itemsize}"
        )
    rows = np.fromfile(path, dtype=GENGA_BINARY_DTYPE)
    ids = rows["i"].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"duplicate body ids in {path}")
    return rows


def _output_path(lane_dir: Path, output_name: str, step: int) -> Path:
    return lane_dir / f"Out{output_name}_{step:012d}.bin"


def _records_by_id(rows: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    table = {int(row["i"]): row for row in rows}
    missing = [body_id for body_id in ids if body_id not in table]
    if missing:
        raise ValueError(f"missing body ids in output: {missing}")
    result = np.empty(len(ids), dtype=GENGA_BINARY_DTYPE)
    for index, body_id in enumerate(ids):
        result[index] = table[body_id]
    return result


def _phase_space(rows: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    selected = _records_by_id(rows, ids)
    return np.column_stack(
        [selected[field].astype(np.float64) for field in ("x", "y", "z", "vx", "vy", "vz")]
    )


def pairwise_source_phase_space(rows: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    state = _phase_space(rows, ids)
    if len(state) < 2:
        raise ValueError("pairwise comparison requires at least two massive bodies")
    differences = [
        state[left] - state[right]
        for left in range(len(state))
        for right in range(left + 1, len(state))
    ]
    return np.asarray(differences, dtype=np.float64)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _parse_energy(path: Path) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = [float(token) for token in line.split()]
            if len(values) != 10:
                raise ValueError(f"unexpected energy row in {path}: {line!r}")
            rows.append(values)
    if not rows:
        raise ValueError(f"empty energy file: {path}")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite energy value: {path}")
    return {
        "row_count": len(rows),
        "final_time_years": float(values[-1, 0]),
        "final_body_count": int(values[-1, 1]),
        "final_relative_angular_momentum": float(values[-1, 8]),
        "final_relative_energy": float(values[-1, 9]),
        "max_abs_relative_angular_momentum": float(np.max(np.abs(values[:, 8]))),
        "max_abs_relative_energy": float(np.max(np.abs(values[:, 9]))),
    }


def _parse_info_echo(path: Path) -> dict[str, str]:
    """Return the last value echoed for each simple ``key: value`` field."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def _expected_toy_info(lane: str) -> dict[str, int]:
    if lane == "base":
        return {"Number of bodies": 2, "Number of test particles": 0, "Test Particle Mode": 0}
    if lane == "finite_tp0":
        return {"Number of bodies": 3, "Number of test particles": 0, "Test Particle Mode": 0}
    return {"Number of bodies": 2, "Number of test particles": 1, "Test Particle Mode": 1}


def _empty_event_file(path: Path) -> bool:
    return path.exists() and not path.read_bytes().strip()


def _lane_output_name(name: str) -> str:
    return name.replace("_", "")


def run_toy(
    output_root: Path,
    engine: Path,
    *,
    device: int = 0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    root, protocol = prepare_toy(output_root, engine, device)
    engine = engine.expanduser().resolve(strict=True)
    runtime_identity_start = {
        "engine": protocol["engine"],
        "runner": protocol["runner"],
    }
    execution: dict[str, list[dict[str, Any]]] = {}

    for name in ("base", "zero_tp1", "finite_tp1", "finite_tp0", "continuous_tp1"):
        record = _execute_genga(
            engine,
            root / "raw" / name,
            device=device,
            steps=TOY_ENDPOINT_STEP,
            phase="full",
            timeout_seconds=timeout_seconds,
            expected_runtime_identity=runtime_identity_start,
        )
        execution[name] = [record]

    split_dir = root / "raw" / "split_tp1"
    first = _execute_genga(
        engine,
        split_dir,
        device=device,
        steps=TOY_MIDPOINT_STEP,
        phase="first_half",
        timeout_seconds=timeout_seconds,
        expected_runtime_identity=runtime_identity_start,
    )
    execution["split_tp1"] = [first]
    split_name = _lane_output_name("split_tp1")
    midpoint = _output_path(split_dir, split_name, TOY_MIDPOINT_STEP)
    if first["success"] and midpoint.exists():
        phase1_copy = split_dir / "phase1_endpoint_000000000032.bin"
        if phase1_copy.exists():
            raise FileExistsError(phase1_copy)
        shutil.copy2(midpoint, phase1_copy)
        second = _execute_genga(
            engine,
            split_dir,
            device=device,
            steps=TOY_ENDPOINT_STEP,
            phase="restart_second_half",
            restart_step=TOY_MIDPOINT_STEP,
            timeout_seconds=timeout_seconds,
            expected_runtime_identity=runtime_identity_start,
        )
    else:
        second = {
            "phase": "restart_second_half",
            "success": False,
            "skipped": True,
            "reason": "first half failed or midpoint output missing",
        }
    execution["split_tp1"].append(second)

    all_invocations_success = all(
        bool(record.get("success"))
        for records in execution.values()
        for record in records
    )
    try:
        runtime_identity_end: Mapping[str, Any] | None = _runtime_identity(engine)
        runtime_identity_error: str | None = None
    except Exception as error:
        runtime_identity_end = None
        runtime_identity_error = f"{type(error).__name__}: {error}"
    runtime_identity_unchanged = runtime_identity_end == runtime_identity_start
    metrics: dict[str, Any] = {}
    gates: dict[str, bool] = {
        "all_invocations_return_zero_and_print_terminal_success": all_invocations_success
    }
    gates["engine_runner_and_test_identities_unchanged"] = runtime_identity_unchanged

    try:
        for step in (TOY_MIDPOINT_STEP, TOY_ENDPOINT_STEP):
            base = read_binary_output(
                _output_path(root / "raw/base", _lane_output_name("base"), step)
            )
            zero = read_binary_output(
                _output_path(root / "raw/zero_tp1", _lane_output_name("zero_tp1"), step)
            )
            base_source = _records_by_id(base, TOY_SOURCE_IDS)
            zero_source = _records_by_id(zero, TOY_SOURCE_IDS)
            equal = base_source.tobytes() == zero_source.tobytes()
            metrics[f"zero_tp1_raw_source_equal_step_{step}"] = equal
        gates["zero_mass_tp1_raw_source_records_bitwise_equal_to_base"] = all(
            metrics[f"zero_tp1_raw_source_equal_step_{step}"]
            for step in (TOY_MIDPOINT_STEP, TOY_ENDPOINT_STEP)
        )

        finite_tp1 = read_binary_output(
            _output_path(
                root / "raw/finite_tp1",
                _lane_output_name("finite_tp1"),
                TOY_ENDPOINT_STEP,
            )
        )
        finite_tp0 = read_binary_output(
            _output_path(
                root / "raw/finite_tp0",
                _lane_output_name("finite_tp0"),
                TOY_ENDPOINT_STEP,
            )
        )
        base_endpoint = read_binary_output(
            _output_path(
                root / "raw/base", _lane_output_name("base"), TOY_ENDPOINT_STEP
            )
        )
        base_pairwise = pairwise_source_phase_space(base_endpoint, TOY_SOURCE_IDS)
        tp1_pairwise = pairwise_source_phase_space(finite_tp1, TOY_SOURCE_IDS)
        tp0_pairwise = pairwise_source_phase_space(finite_tp0, TOY_SOURCE_IDS)
        tp1_delta = _max_abs(base_pairwise, tp1_pairwise)
        tp1_raw_delta = _max_abs(
            _phase_space(base_endpoint, TOY_SOURCE_IDS),
            _phase_space(finite_tp1, TOY_SOURCE_IDS),
        )
        feedback_delta = _max_abs(tp1_pairwise, tp0_pairwise)
        metrics["finite_tp1_pairwise_source_phase_space_max_abs"] = tp1_delta
        metrics["finite_tp1_raw_source_phase_space_max_abs"] = tp1_raw_delta
        metrics["finite_tp0_pairwise_feedback_max_abs"] = feedback_delta
        gates["finite_positive_tp1_frame_transform_limitation_detected"] = (
            tp1_raw_delta > TOY_FEEDBACK_DETECTION_FLOOR
        )
        metrics["finite_tp1_pairwise_tolerance_for_context"] = (
            TOY_PAIRWISE_ABS_TOLERANCE
        )
        gates["finite_tp0_feedback_detected"] = (
            feedback_delta > TOY_FEEDBACK_DETECTION_FLOOR
        )

        continuous = read_binary_output(
            _output_path(
                root / "raw/continuous_tp1",
                _lane_output_name("continuous_tp1"),
                TOY_ENDPOINT_STEP,
            )
        )
        split = read_binary_output(
            _output_path(
                root / "raw/split_tp1", split_name, TOY_ENDPOINT_STEP
            )
        )
        restart_equal = continuous.tobytes() == split.tobytes()
        metrics["continuous_split_endpoint_bitwise_equal"] = restart_equal
        gates["continuous_split_binary_endpoint_bitwise_equal"] = restart_equal

        conservation: dict[str, Any] = {}
        event_files_empty: dict[str, bool] = {}
        info_echo: dict[str, Any] = {}
        for lane in protocol["lanes"]:
            lane_dir = root / "raw" / lane
            output_name = _lane_output_name(lane)
            energy = _parse_energy(lane_dir / f"Energy{output_name}.dat")
            conservation[lane] = energy
            event_files_empty[lane] = _empty_event_file(
                lane_dir / f"Collisions{output_name}.dat"
            ) and _empty_event_file(lane_dir / f"Ejections{output_name}.dat")
            echoed = _parse_info_echo(lane_dir / f"info{output_name}.dat")
            expected = _expected_toy_info(lane)
            observed = {key: int(echoed[key]) for key in expected}
            info_echo[lane] = {
                "expected": expected,
                "observed": observed,
                "matched": observed == expected,
            }
        metrics["conservation"] = conservation
        metrics["conservation_gate_exclusion"] = {
            "finite_tp1": (
                "positive-mass TP1 is a known one-way/frame-transform limitation; "
                "production tracers must have exactly zero mass"
            )
        }
        metrics["event_files_empty"] = event_files_empty
        metrics["info_echo"] = info_echo
        gates["conservation_within_predeclared_limit"] = all(
            item["max_abs_relative_angular_momentum"] <= TOY_CONSERVATION_LIMIT
            and item["max_abs_relative_energy"] <= TOY_CONSERVATION_LIMIT
            for lane, item in conservation.items()
            if lane != "finite_tp1"
        )
        gates["no_collisions_or_ejections"] = all(event_files_empty.values())
        gates["info_echo_body_classification_matches_protocol"] = all(
            item["matched"] for item in info_echo.values()
        )
        analysis_error: str | None = None
    except Exception as error:  # evidence remains usable on fail-closed analysis
        analysis_error = f"{type(error).__name__}: {error}"
        for key in (
            "zero_mass_tp1_raw_source_records_bitwise_equal_to_base",
            "finite_positive_tp1_frame_transform_limitation_detected",
            "finite_tp0_feedback_detected",
            "continuous_split_binary_endpoint_bitwise_equal",
            "conservation_within_predeclared_limit",
            "no_collisions_or_ejections",
            "info_echo_body_classification_matches_protocol",
        ):
            gates.setdefault(key, False)

    passed = all(gates.values()) and analysis_error is None
    result = {
        "schema": RESULT_SCHEMA,
        "qualification": "TP1_QUIET_PATH_NO_BACKREACTION_AND_BINARY_RESTART",
        "scientific_scope": "ENGINEERING_ONLY_NO_ASTROPHYSICAL_INFERENCE",
        "status": "PASS" if passed else "FAIL",
        "started_from_protocol_sha256": _sha256_file(root / "PROTOCOL.json"),
        "completed_utc": _utc_now(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "execution": execution,
        "runtime_identity_start": runtime_identity_start,
        "runtime_identity_end": runtime_identity_end,
        "runtime_identity_error": runtime_identity_error,
        "metrics": metrics,
        "gates": gates,
        "analysis_error": analysis_error,
    }
    _write_new_json(root / "RESULT.json", result)
    inventory = _artifact_inventory(root, exclude=("ARTIFACTS.json",))
    _write_new_json(root / "ARTIFACTS.json", inventory)
    return result


def _load_jx_inputs(
    source_path: Path,
    tracer_path: Path,
    *,
    enforce_tracked_hashes: bool = True,
) -> dict[str, np.ndarray]:
    source_path = source_path.resolve(strict=True)
    tracer_path = tracer_path.resolve(strict=True)
    source_sha = _sha256_file(source_path)
    tracer_sha = _sha256_file(tracer_path)
    if enforce_tracked_hashes and source_sha != TRACKED_SOURCE_SHA256:
        raise ValueError(f"tracked SOURCE_STATE.npz hash mismatch: {source_sha}")
    if enforce_tracked_hashes and tracer_sha != TRACKED_TRACER_SHA256:
        raise ValueError(f"tracked TRANSFORMED_TRACERS.npz hash mismatch: {tracer_sha}")
    with np.load(source_path, allow_pickle=False) as source:
        if set(source.files) != {"source_positions", "source_velocities", "source_gm"}:
            raise ValueError(f"unexpected source arrays: {sorted(source.files)}")
        positions = np.asarray(source["source_positions"], dtype=np.float64)
        velocities = np.asarray(source["source_velocities"], dtype=np.float64)
        gm = np.asarray(source["source_gm"], dtype=np.float64)
    with np.load(tracer_path, allow_pickle=False) as tracers:
        if set(tracers.files) != {
            "tracer_positions",
            "tracer_velocities",
            "tracer_indices",
        }:
            raise ValueError(f"unexpected tracer arrays: {sorted(tracers.files)}")
        tracer_positions = np.asarray(tracers["tracer_positions"], dtype=np.float64)
        tracer_velocities = np.asarray(tracers["tracer_velocities"], dtype=np.float64)
        tracer_indices = np.asarray(tracers["tracer_indices"], dtype=np.uint64)
    if positions.shape != (6, 3) or velocities.shape != (6, 3) or gm.shape != (6,):
        raise ValueError("JX source state must contain six 3D sources")
    if tracer_positions.shape != tracer_velocities.shape or tracer_positions.ndim != 2:
        raise ValueError("tracer phase-space shapes differ")
    if tracer_positions.shape[1] != 3 or tracer_indices.shape != (len(tracer_positions),):
        raise ValueError("invalid tracer array dimensions")
    arrays = (positions, velocities, gm, tracer_positions, tracer_velocities)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("JX source or tracer state contains non-finite values")
    if np.any(gm <= 0.0):
        raise ValueError("JX source GM must be positive")
    if len(np.unique(tracer_indices)) != len(tracer_indices):
        raise ValueError("JX tracer ids are not unique")
    if enforce_tracked_hashes and len(tracer_indices) != JX_EXPECTED_TRACER_COUNT:
        raise ValueError("tracked tracer count changed")
    return {
        "source_positions": positions,
        "source_velocities": velocities,
        "source_gm": gm,
        "tracer_positions": tracer_positions,
        "tracer_velocities": tracer_velocities,
        "tracer_indices": tracer_indices,
        "source_sha256": np.asarray(source_sha),
        "tracer_sha256": np.asarray(tracer_sha),
    }


def adapt_jx_bodies(
    arrays: Mapping[str, np.ndarray], prefix_count: int
) -> tuple[float, tuple[Body, ...], tuple[Body, ...]]:
    source_positions = np.asarray(arrays["source_positions"], dtype=np.float64)
    source_velocities = np.asarray(arrays["source_velocities"], dtype=np.float64)
    source_gm = np.asarray(arrays["source_gm"], dtype=np.float64)
    tracer_positions = np.asarray(arrays["tracer_positions"], dtype=np.float64)
    tracer_velocities = np.asarray(arrays["tracer_velocities"], dtype=np.float64)
    tracer_indices = np.asarray(arrays["tracer_indices"], dtype=np.uint64)
    if prefix_count < 1 or prefix_count > len(tracer_indices):
        raise ValueError(f"prefix_count must be in [1, {len(tracer_indices)}]")

    primary_position = source_positions[0]
    primary_velocity = source_velocities[0]
    masses = gm_au3_per_year2_to_msun(source_gm)
    source_relative_position = source_positions[1:] - primary_position
    source_relative_velocity = velocity_au_per_year_to_genga(
        source_velocities[1:] - primary_velocity
    )
    sources = tuple(
        Body(
            JX_SOURCE_BODY_IDS[index],
            float(masses[index + 1]),
            1.0e-12,
            tuple(float(value) for value in source_relative_position[index]),
            tuple(float(value) for value in source_relative_velocity[index]),
        )
        for index in range(5)
    )

    prefix_position = tracer_positions[:prefix_count] - primary_position
    prefix_velocity = velocity_au_per_year_to_genga(
        tracer_velocities[:prefix_count] - primary_velocity
    )
    tracers: list[Body] = []
    for index in range(prefix_count):
        original_id = int(tracer_indices[index])
        body_id = JX_TRACER_ID_OFFSET + original_id
        tracers.append(
            Body(
                body_id,
                0.0,
                0.0,
                tuple(float(value) for value in prefix_position[index]),
                tuple(float(value) for value in prefix_velocity[index]),
            )
        )
    return float(masses[0]), sources, tuple(tracers)


def prepare_jx(
    output_root: Path,
    *,
    source_path: Path = TRACKED_SOURCE_STATE,
    tracer_path: Path = TRACKED_TRACER_STATE,
    prefix_count: int = JX_DEFAULT_QUALIFICATION_PREFIX,
    integration_steps: int = JX_FROZEN_QUALIFICATION_STEPS,
) -> dict[str, Any]:
    if integration_steps <= 0:
        raise ValueError("integration_steps must be positive")
    arrays = _load_jx_inputs(source_path, tracer_path, enforce_tracked_hashes=True)
    central_mass, sources, tracers = adapt_jx_bodies(arrays, prefix_count)
    root = _require_fresh_directory(output_root)
    lane_names = ("source_only", f"plus_{prefix_count}")
    protocol = {
        "schema": JX_ADAPTER_SCHEMA,
        "operation": "PREPARE_FROZEN_EXTERNAL_GENGA_WORKLOAD",
        "scientific_scope": "ENGINEERING_PERFORMANCE_ONLY_NO_ASTROPHYSICAL_INFERENCE",
        "runner": _runner_identity(),
        "source_inputs": {
            "source_state": {
                "path": str(source_path.resolve()),
                "sha256": str(arrays["source_sha256"]),
            },
            "transformed_tracers": {
                "path": str(tracer_path.resolve()),
                "sha256": str(arrays["tracer_sha256"]),
                "classification": "MODEL_OUTPUT_SYNTHETIC_POPULATION_REFRAME_ONLY",
            },
        },
        "conversion": {
            "mass": "GM_AU3_year2 / (def_Kg * 365.25^2)",
            "velocity": "velocity_AU_year / (365.25 * dayUnit)",
            "position": "heliocentric = value - primary_value",
            "def_Kg": GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN,
            "dayUnit": GENGA_DAY_UNIT,
            "days_per_year": JX_DAYS_PER_YEAR,
            "central_mass_msun_hex": central_mass.hex(),
        },
        "identity": {
            "primary_omitted_from_input": TRACKED_SOURCE_IDS[0],
            "source_names": list(TRACKED_SOURCE_IDS[1:]),
            "source_genga_ids": list(JX_SOURCE_BODY_IDS),
            "tracer_id_formula": f"{JX_TRACER_ID_OFFSET} + tracked_uint64_id",
            "prefix_count": prefix_count,
            "selection": "IMMUTABLE_PREFIX_BY_TRACKED_ARRAY_ORDER",
        },
        "frozen_run": {
            "lanes": list(lane_names),
            "timestep_days": 45.65625,
            "integration_steps": integration_steps,
            "nominal_span_years": integration_steps * 0.125,
            "test_particle_mode": 1,
            "particle_minimum_mass": 0.0,
            "tracer_mass": 0.0,
            "outer_truncation_radius_au": 10_000.0,
            "coordinate_output_interval": -1,
            "energy_output_interval": -1,
            "order": 2,
            "symplectic_recursion_levels": 1,
            "symplectic_recursion_sub_steps": 2,
            "serial_grouping": 1,
            "kernel_tuning": 0,
            "kick_single_precision": 0,
        },
        "claims_excluded": [
            "physics parity with JX",
            "observed or Rubin-selected population",
            "primordial population",
            "Planet Nine or flyby evidence",
            "4.5-Gyr stability or reconstruction",
        ],
    }
    _write_new_json(root / "PROTOCOL.json", protocol)
    raw = root / "raw"
    raw.mkdir()
    lane_bodies = {lane_names[0]: sources, lane_names[1]: sources + tracers}
    for lane_name, bodies in lane_bodies.items():
        lane = raw / lane_name
        lane.mkdir()
        output_name = lane_name.replace("_", "")
        _write_new_text(lane / "initial.dat", render_initial_conditions(bodies))
        _write_new_text(
            lane / "param.dat",
            render_param(
                output_name=output_name,
                input_name="initial.dat",
                central_mass_msun=central_mass,
                use_test_particles=1,
                particle_minimum_mass=0.0,
                integration_steps=integration_steps,
                timestep_days=45.65625,
                coordinate_interval=-1,
                energy_interval=-1,
                minimum_massive_bodies=5,
                outer_radius_au=10_000.0,
            ),
        )
        _write_new_text(lane / "tuningParameters.dat", TUNING_PARAMETERS)
    id_map = {
        "central": {"name": TRACKED_SOURCE_IDS[0], "genga_implicit": True},
        "sources": [
            {"name": name, "genga_id": body_id}
            for name, body_id in zip(TRACKED_SOURCE_IDS[1:], JX_SOURCE_BODY_IDS)
        ],
        "tracers": {
            "count": prefix_count,
            "genga_id_offset": JX_TRACER_ID_OFFSET,
            "first_tracked_id": int(arrays["tracer_indices"][0]),
            "last_tracked_id": int(arrays["tracer_indices"][prefix_count - 1]),
        },
    }
    _write_new_json(root / "ID_MAP.json", id_map)
    result = {
        "schema": RESULT_SCHEMA,
        "qualification": "JX_TO_GENGA_ADAPTER_PREPARATION",
        "status": "PREPARED_NOT_EXECUTED",
        "scientific_scope": "ENGINEERING_PERFORMANCE_ONLY_NO_ASTROPHYSICAL_INFERENCE",
        "gates": {
            "tracked_source_sha256_exact": str(arrays["source_sha256"])
            == TRACKED_SOURCE_SHA256,
            "tracked_tracer_sha256_exact": str(arrays["tracer_sha256"])
            == TRACKED_TRACER_SHA256,
            "all_production_tracer_masses_exact_zero": all(
                body.mass_msun == 0.0 for body in tracers
            ),
            "all_ids_unique": len(
                {body.body_id for body in sources + tracers}
            )
            == len(sources) + len(tracers),
            "default_outer_50_au_not_used": True,
        },
        "counts": {
            "central": 1,
            "explicit_massive_sources": len(sources),
            "massless_tracers": len(tracers),
        },
        "protocol_sha256": _sha256_file(root / "PROTOCOL.json"),
    }
    if not all(result["gates"].values()):
        raise RuntimeError(f"JX adapter preparation gate failed: {result['gates']}")
    _write_new_json(root / "RESULT.json", result)
    inventory = _artifact_inventory(root, exclude=("ARTIFACTS.json",))
    _write_new_json(root / "ARTIFACTS.json", inventory)
    return result


def _parse_execution_seconds(lane_dir: Path, output_name: str) -> float | None:
    path = lane_dir / f"time{output_name}.dat"
    if not path.exists():
        return None
    values = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        return None
    return float(values[-1][-1])


def run_prepared_jx(
    prepared_root: Path,
    engine: Path,
    *,
    device: int = 0,
    timeout_seconds: float = 86_400.0,
) -> dict[str, Any]:
    root = prepared_root.resolve(strict=True)
    engine = engine.expanduser().resolve(strict=True)
    if (root / "EXECUTION_RESULT.json").exists():
        raise FileExistsError("prepared workload has already been executed")
    _verify_artifact_inventory(root, root / "ARTIFACTS.json")
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != JX_ADAPTER_SCHEMA:
        raise ValueError("not a JX GENGA adapter directory")
    expected_protocol_sha = json.loads(
        (root / "RESULT.json").read_text(encoding="utf-8")
    )["protocol_sha256"]
    if _sha256_file(root / "PROTOCOL.json") != expected_protocol_sha:
        raise ValueError("prepared protocol identity changed")
    runtime_identity_start = _runtime_identity(engine)
    if protocol.get("runner") != runtime_identity_start["runner"]:
        raise ValueError("prepared runner or test source identity changed")
    steps = int(protocol["frozen_run"]["integration_steps"])
    lane_results: dict[str, Any] = {}
    for lane_name in protocol["frozen_run"]["lanes"]:
        lane = root / "raw" / lane_name
        if (lane / "lock.dat").exists():
            raise FileExistsError(f"lane already has a GENGA lock: {lane}")
        execution = _execute_genga(
            engine,
            lane,
            device=device,
            steps=steps,
            phase="frozen_run",
            timeout_seconds=timeout_seconds,
            expected_runtime_identity=runtime_identity_start,
        )
        output_name = lane_name.replace("_", "")
        lane_results[lane_name] = {
            "execution": execution,
            "genga_reported_seconds": _parse_execution_seconds(lane, output_name),
        }
    all_success = all(item["execution"]["success"] for item in lane_results.values())
    try:
        runtime_identity_end: Mapping[str, Any] | None = _runtime_identity(engine)
        runtime_identity_error: str | None = None
    except Exception as error:
        runtime_identity_end = None
        runtime_identity_error = f"{type(error).__name__}: {error}"
    runtime_identity_unchanged = runtime_identity_end == runtime_identity_start
    source_lane = root / "raw" / "source_only"
    prefix_lane_name = next(
        name for name in protocol["frozen_run"]["lanes"] if name != "source_only"
    )
    prefix_lane = root / "raw" / prefix_lane_name
    massive_exact = False
    comparison_error: str | None = None
    if all_success:
        try:
            source_rows = read_binary_output(
                _output_path(source_lane, "sourceonly", steps)
            )
            prefix_rows = read_binary_output(
                _output_path(prefix_lane, prefix_lane_name.replace("_", ""), steps)
            )
            source_massive = _records_by_id(source_rows, JX_SOURCE_BODY_IDS)
            prefix_massive = _records_by_id(prefix_rows, JX_SOURCE_BODY_IDS)
            massive_exact = source_massive.tobytes() == prefix_massive.tobytes()
        except Exception as error:
            comparison_error = f"{type(error).__name__}: {error}"
    passed = (
        all_success
        and massive_exact
        and comparison_error is None
        and runtime_identity_unchanged
    )
    result = {
        "schema": RESULT_SCHEMA,
        "qualification": "JX_SIZED_GENGA_FROZEN_PERFORMANCE_RUN",
        "status": "PASS" if passed else "FAIL",
        "scientific_scope": "ENGINEERING_PERFORMANCE_ONLY_NO_ASTROPHYSICAL_INFERENCE",
        "engine": runtime_identity_start["engine"],
        "runner": runtime_identity_start["runner"],
        "runtime_identity_end": runtime_identity_end,
        "runtime_identity_error": runtime_identity_error,
        "device": device,
        "protocol_sha256": expected_protocol_sha,
        "lanes": lane_results,
        "gate_all_invocations_return_zero_and_print_terminal_success": all_success,
        "gate_source_only_vs_prefix_massive_endpoint_bitwise_equal": massive_exact,
        "gate_engine_runner_and_test_identities_unchanged": runtime_identity_unchanged,
        "comparison_error": comparison_error,
        "completed_utc": _utc_now(),
    }
    _write_new_json(root / "EXECUTION_RESULT.json", result)
    inventory = _artifact_inventory(root, exclude=("EXECUTION_ARTIFACTS.json",))
    _write_new_json(root / "EXECUTION_ARTIFACTS.json", inventory)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy = subparsers.add_parser("toy", help="run the bounded TP1/restart qualification")
    toy.add_argument("--engine", type=Path, required=True)
    toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--device", type=int, default=0)
    toy.add_argument("--timeout-seconds", type=float, default=300.0)

    adapter = subparsers.add_parser("prepare-jx", help="prepare the frozen JX-sized workload")
    adapter.add_argument("--output", type=Path, required=True)
    adapter.add_argument("--source", type=Path, default=TRACKED_SOURCE_STATE)
    adapter.add_argument("--tracers", type=Path, default=TRACKED_TRACER_STATE)
    adapter.add_argument(
        "--prefix-count", type=int, default=JX_DEFAULT_QUALIFICATION_PREFIX
    )
    adapter.add_argument(
        "--integration-steps",
        type=int,
        default=JX_FROZEN_QUALIFICATION_STEPS,
        help="frozen run length; 1024=128 years, 8000=1000 years",
    )

    run_jx = subparsers.add_parser("run-jx", help="execute a prepared frozen JX workload")
    run_jx.add_argument("--prepared", type=Path, required=True)
    run_jx.add_argument("--engine", type=Path, required=True)
    run_jx.add_argument("--device", type=int, default=0)
    run_jx.add_argument("--timeout-seconds", type=float, default=86_400.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "toy":
        result = run_toy(
            arguments.output,
            arguments.engine,
            device=arguments.device,
            timeout_seconds=arguments.timeout_seconds,
        )
    elif arguments.command == "prepare-jx":
        result = prepare_jx(
            arguments.output,
            source_path=arguments.source,
            tracer_path=arguments.tracers,
            prefix_count=arguments.prefix_count,
            integration_steps=arguments.integration_steps,
        )
    elif arguments.command == "run-jx":
        result = run_prepared_jx(
            arguments.prepared,
            arguments.engine,
            device=arguments.device,
            timeout_seconds=arguments.timeout_seconds,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"PASS", "PREPARED_NOT_EXECUTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
