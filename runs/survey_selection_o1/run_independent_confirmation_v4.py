#!/usr/bin/env python3
"""Checkpointed driver for the prelocked independent JX-O1 V4 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jxplanetx.provenance import sha256_file
from jxplanetx.survey_selection_v2 import (
    load_survey_contract,
    parse_ossos_tracked_file,
    register_official_ossos_pool,
    run_analytic_survey_pilot,
    verify_external_simulator,
    write_ossos_model_file,
)
from jxplanetx.survey_selection_v4 import (
    _require_independent_contract,
    finalize_survey_selection,
)


STATE_SCHEMA = "jx-survey-selection-execution-state/v4-independent"
QUALIFICATION_SCHEMA = "jx-survey-selection-official-qualification/v4-independent"
READ_MODEL_CHUNK_SIZE = 100
DEFAULT_BATCH_SIZE = 100_001


def _atomic_replace_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable batch artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fortran_seed(seed_key: str, seed_block: int, batch_index: int) -> int:
    message = f"official-ossos-v4-independent\x1f{seed_block}\x1f{batch_index}".encode(
        "utf-8"
    )
    digest = hmac.new(seed_key.encode("utf-8"), message, hashlib.sha256).digest()
    return 1 + int.from_bytes(digest[:8], "big") % 2_147_483_646


def _compiler_version(compiler: str | None) -> str | None:
    if compiler is None:
        return None
    try:
        completed = subprocess.run(
            [compiler, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.splitlines()[0].strip()


def _preflight(contract: dict[str, Any], simulator_root: Path) -> dict[str, Any]:
    verification = verify_external_simulator(contract, simulator_root)
    source = simulator_root / contract["external_simulator"]["source_subdirectory"]
    executable = source / "Driver"
    compiler = shutil.which("gfortran")
    make = shutil.which("make")
    return {
        "schema": "jx-survey-selection-preflight/v4-independent",
        "external_verification": verification,
        "source_directory": str(source),
        "gfortran": compiler,
        "gfortran_version": _compiler_version(compiler),
        "make": make,
        "driver_executable": str(executable),
        "driver_exists": executable.is_file(),
        "driver_sha256": sha256_file(executable) if executable.is_file() else None,
        "ready": verification["passed"] and make is not None and compiler is not None,
    }


def _compile_driver(contract: dict[str, Any], simulator_root: Path) -> Path:
    preflight = _preflight(contract, simulator_root)
    if not preflight["external_verification"]["passed"]:
        raise RuntimeError("pinned OSSOS source or characterization verification failed")
    if preflight["gfortran"] is None or preflight["make"] is None:
        raise RuntimeError("gfortran and make are required to build the pinned OSSOS F95 driver")
    source = Path(preflight["source_directory"])
    executable = Path(preflight["driver_executable"])
    clean = subprocess.run(
        [preflight["make"], "clean"],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError(
            "OSSOS driver clean build preparation failed\n"
            + clean.stdout[-4000:]
            + "\n"
            + clean.stderr[-4000:]
        )
    completed = subprocess.run(
        [preflight["make"], "Driver", "GIMEOBJ=ReadModelFromFile"],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    adapter_link = source / "GiMeObj.f95"
    expected_adapter = (source / "ReadModelFromFile.f95").resolve()
    if (
        completed.returncode != 0
        or not executable.is_file()
        or not adapter_link.is_symlink()
        or adapter_link.resolve() != expected_adapter
    ):
        raise RuntimeError(
            "OSSOS ReadModelFromFile driver compilation failed\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    return executable


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    if batch_size % READ_MODEL_CHUNK_SIZE == 0:
        raise ValueError(
            "batch size must not be divisible by 100; the pinned ReadModelFromFile "
            "EOF path is invalid when its final internal chunk is exactly full"
        )
    return batch_size


def _driver_totals(path: Path) -> dict[str, int]:
    patterns = {
        "objects": re.compile(r"^# Total number of objects:\s+([0-9]+)(?:\.0*)?\s*$"),
        "detections": re.compile(r"^# Number of detections:\s+([0-9]+)\s*$"),
        "tracked": re.compile(r"^# Number of tracked objects:\s+([0-9]+)\s*$"),
    }
    totals: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for name, pattern in patterns.items():
            match = pattern.fullmatch(line)
            if match is not None:
                totals[name] = int(match.group(1))
    if set(totals) != set(patterns):
        raise RuntimeError(f"official detected output lacks complete footer totals: {path}")
    if totals["tracked"] > totals["detections"] or totals["detections"] > totals["objects"]:
        raise RuntimeError(f"official detected output has inconsistent footer totals: {path}")
    return totals


def _new_state(
    contract_path: Path,
    contract: dict[str, Any],
    model_id: str,
    seed_block: int,
    driver: Path,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "contract_sha256": sha256_file(contract_path),
        "simulator_commit": contract["external_simulator"]["commit"],
        "driver_sha256": sha256_file(driver),
        "model_id": model_id,
        "seed_block": seed_block,
        "batches": [],
        "intrinsic_draws": 0,
        "tracked_detections": 0,
    }


def _load_and_verify_state(
    state_path: Path,
    contract_path: Path,
    contract: dict[str, Any],
    simulator_root: Path,
    driver: Path,
    model_id: str,
    seed_block: int,
) -> tuple[dict[str, Any], bool]:
    if not state_path.exists():
        return _new_state(contract_path, contract, model_id, seed_block, driver), False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise RuntimeError(f"invalid state schema: {state_path}")
    expected = {
        "contract_sha256": sha256_file(contract_path),
        "simulator_commit": contract["external_simulator"]["commit"],
        "driver_sha256": sha256_file(driver),
        "model_id": model_id,
        "seed_block": seed_block,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"checkpoint {state_path} has mismatched {key}")
    survey_target = (
        simulator_root / contract["external_simulator"]["characterization_path"]
    ).resolve()
    intrinsic = 0
    tracked = 0
    for expected_index, batch in enumerate(state.get("batches", [])):
        if batch.get("batch_index") != expected_index or batch.get("start_index") != intrinsic:
            raise RuntimeError(f"checkpoint batch sequence mismatch: {state_path}")
        for path_key, hash_key in (
            ("model_path", "model_sha256"),
            ("driver_input_path", "driver_input_sha256"),
            ("detected_path", "detected_sha256"),
            ("tracked_path", "tracked_sha256"),
            ("model_used_path", "model_used_sha256"),
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256"),
        ):
            artifact = (state_path.parent / batch[path_key]).resolve()
            if not artifact.is_file() or sha256_file(artifact) != batch[hash_key]:
                raise RuntimeError(f"checkpoint artifact mismatch: {artifact}")
        survey_link = state_path.parent / batch["survey_link_path"]
        if not survey_link.is_symlink() or survey_link.resolve() != survey_target:
            raise RuntimeError(f"checkpoint survey link mismatch: {survey_link}")
        detected = (state_path.parent / batch["detected_path"]).resolve()
        raw = (state_path.parent / batch["tracked_path"]).resolve()
        totals = _driver_totals(detected)
        parsed = parse_ossos_tracked_file(raw, model_id, seed_block)
        if totals != batch["driver_totals"]:
            raise RuntimeError(f"checkpoint driver totals mismatch: {detected}")
        if totals["objects"] != batch["intrinsic_draws"]:
            raise RuntimeError(f"checkpoint intrinsic count mismatch: {detected}")
        if len(parsed) != batch["tracked_count"] or totals["tracked"] != len(parsed):
            raise RuntimeError(f"checkpoint tracked count mismatch: {raw}")
        intrinsic += int(batch["intrinsic_draws"])
        tracked += int(batch["tracked_count"])
    if intrinsic != state.get("intrinsic_draws") or tracked != state.get("tracked_detections"):
        raise RuntimeError(f"checkpoint totals mismatch: {state_path}")
    return state, True


def _run_batch(
    contract: dict[str, Any],
    driver: Path,
    simulator_root: Path,
    state_path: Path,
    state: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    batch_size = _validate_batch_size(batch_size)
    model_id = state["model_id"]
    seed_block = int(state["seed_block"])
    batch_index = len(state["batches"])
    start_index = int(state["intrinsic_draws"])
    batch_root = state_path.parent / f"batch_{batch_index:05d}"
    batch_root.mkdir(parents=True, exist_ok=False)
    model_path = batch_root / "intrinsic_model.dat"
    driver_input = batch_root / "Driver.in"
    detected_path = batch_root / "SimulDetect.dat"
    tracked_path = batch_root / "SimulTrack.dat"
    model_used_path = batch_root / "ModelUsed.dat"
    stdout_path = batch_root / "stdout.txt"
    stderr_path = batch_root / "stderr.txt"
    survey_link = batch_root / "survey"
    write_ossos_model_file(
        model_path,
        contract,
        model_id,
        seed_block,
        batch_size,
        namespace="final-v4-independent",
        start_index=start_index,
    )
    survey_path = (
        simulator_root / contract["external_simulator"]["characterization_path"]
    ).resolve()
    survey_link.symlink_to(survey_path, target_is_directory=True)
    seed = _fortran_seed(contract["population"]["seed_key"], seed_block, batch_index)
    input_lines = (
        str(seed),
        "0",
        survey_link.name,
        model_path.name,
        detected_path.name,
        tracked_path.name,
    )
    if len(input_lines[2]) > 100 or any(len(value) > 80 for value in input_lines[3:]):
        raise RuntimeError("short-path adapter exceeded the pinned Driver character limits")
    driver_text = "\n".join((*input_lines, ""))
    _write_text_once(driver_input, driver_text)
    completed = subprocess.run(
        [str(driver.resolve())],
        cwd=batch_root,
        input=driver_text,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_text_once(stdout_path, completed.stdout)
    _write_text_once(stderr_path, completed.stderr)
    if (
        completed.returncode != 0
        or not detected_path.is_file()
        or not tracked_path.is_file()
        or not model_used_path.is_file()
    ):
        raise RuntimeError(
            f"OSSOS batch failed for {model_id} block {seed_block} batch {batch_index}: "
            f"exit {completed.returncode}"
        )
    totals = _driver_totals(detected_path)
    tracked = parse_ossos_tracked_file(tracked_path, model_id, seed_block)
    if totals["objects"] != batch_size:
        raise RuntimeError(
            f"OSSOS processed {totals['objects']} objects but the batch contains {batch_size}"
        )
    if totals["tracked"] != len(tracked):
        raise RuntimeError("OSSOS footer and tracked-file counts disagree")
    relative = lambda path: os.path.relpath(path.absolute(), state_path.parent.absolute())
    record = {
        "batch_index": batch_index,
        "start_index": start_index,
        "intrinsic_draws": batch_size,
        "fortran_seed": seed,
        "tracked_count": len(tracked),
        "driver_totals": totals,
        "survey_link_path": relative(survey_link),
        "model_path": relative(model_path),
        "model_sha256": sha256_file(model_path),
        "driver_input_path": relative(driver_input),
        "driver_input_sha256": sha256_file(driver_input),
        "detected_path": relative(detected_path),
        "detected_sha256": sha256_file(detected_path),
        "tracked_path": relative(tracked_path),
        "tracked_sha256": sha256_file(tracked_path),
        "model_used_path": relative(model_used_path),
        "model_used_sha256": sha256_file(model_used_path),
        "stdout_path": relative(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": relative(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    state["batches"].append(record)
    state["intrinsic_draws"] += batch_size
    state["tracked_detections"] += len(tracked)
    _atomic_replace_json(state_path, state)
    return state


def _verified_block_state(
    state_path: Path,
    contract_path: Path,
    contract: dict[str, Any],
    simulator_root: Path,
    driver: Path,
    model_id: str,
    seed_block: int,
) -> dict[str, Any]:
    verified, _ = _load_and_verify_state(
        state_path,
        contract_path,
        contract,
        simulator_root,
        driver,
        model_id,
        seed_block,
    )
    return verified


def run_qualification(
    contract_path: Path,
    simulator_root: Path,
    run_dir: Path,
    output: Path,
    model_id: str,
    seed_block: int,
    draws: int,
) -> dict[str, Any]:
    contract = load_survey_contract(contract_path)
    _require_independent_contract(contract)
    draws = _validate_batch_size(draws)
    driver = _compile_driver(contract, simulator_root)
    state_path = run_dir / model_id / f"block_{seed_block:02d}" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state, _ = _load_and_verify_state(
        state_path,
        contract_path,
        contract,
        simulator_root,
        driver,
        model_id,
        seed_block,
    )
    if not state["batches"]:
        state = _run_batch(contract, driver, simulator_root, state_path, state, draws)
    if len(state["batches"]) != 1 or state["intrinsic_draws"] != draws:
        raise RuntimeError("qualification directory does not contain exactly the declared batch")
    replay = _verified_block_state(
        state_path,
        contract_path,
        contract,
        simulator_root,
        driver,
        model_id,
        seed_block,
    )
    result = {
        "schema": QUALIFICATION_SCHEMA,
        "contract_sha256": sha256_file(contract_path),
        "simulator_commit": contract["external_simulator"]["commit"],
        "driver_sha256": sha256_file(driver),
        "model_id": model_id,
        "seed_block": seed_block,
        "intrinsic_draws": replay["intrinsic_draws"],
        "tracked_detections": replay["tracked_detections"],
        "tracked_rate": replay["tracked_detections"] / replay["intrinsic_draws"],
        "checkpoint_replay_passed": replay == state,
        "status": "COMPLETED",
    }
    if output.exists():
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != result:
            raise RuntimeError("immutable qualification output does not match replay")
        return observed
    _atomic_replace_json(output, result)
    return result


def run_official(
    contract_path: Path,
    simulator_root: Path,
    run_dir: Path,
    output: Path,
    batch_size: int,
    max_draws_per_block: int,
) -> dict[str, Any]:
    contract = load_survey_contract(contract_path)
    _require_independent_contract(contract)
    batch_size = _validate_batch_size(batch_size)
    if max_draws_per_block < batch_size:
        raise ValueError("max draws per block must cover at least one batch")
    driver = _compile_driver(contract, simulator_root)
    target = int(contract["population"]["minimum_tracked_detections_per_seed_block"])
    raw: dict[str, list[tuple[int, Path, int]]] = {"correct": [], "wrong": []}
    replay_passed = True
    for model_id in ("correct", "wrong"):
        for seed_block in range(int(contract["population"]["seed_blocks"])):
            block_root = run_dir / model_id / f"block_{seed_block:02d}"
            state_path = block_root / "state.json"
            block_root.mkdir(parents=True, exist_ok=True)
            state, _ = _load_and_verify_state(
                state_path,
                contract_path,
                contract,
                simulator_root,
                driver,
                model_id,
                seed_block,
            )
            while state["tracked_detections"] < target:
                if state["intrinsic_draws"] + batch_size > max_draws_per_block:
                    raise RuntimeError(
                        f"draw ceiling reached for {model_id} block {seed_block} before "
                        f"{target} detections"
                    )
                state = _run_batch(
                    contract,
                    driver,
                    simulator_root,
                    state_path,
                    state,
                    batch_size,
                )
            verified = _verified_block_state(
                state_path,
                contract_path,
                contract,
                simulator_root,
                driver,
                model_id,
                seed_block,
            )
            replay_passed = replay_passed and verified == state
            if not replay_passed:
                raise RuntimeError(f"checkpoint replay changed state: {state_path}")
            for batch in state["batches"]:
                raw[model_id].append(
                    (
                        seed_block,
                        (state_path.parent / batch["tracked_path"]).resolve(),
                        int(batch["intrinsic_draws"]),
                    )
                )

    manifests: dict[str, Path] = {}
    for model_id in ("correct", "wrong"):
        detections = run_dir / f"{model_id}_detections.csv"
        manifest = run_dir / f"{model_id}_pool.json"
        if not manifest.exists():
            register_official_ossos_pool(
                contract,
                simulator_root,
                model_id,
                raw[model_id],
                detections,
                manifest,
                checkpoint_replay_passed=replay_passed,
            )
        manifests[model_id] = manifest
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return finalize_survey_selection(
        contract_path,
        manifests["correct"],
        manifests["wrong"],
        output,
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--contract", required=True, type=Path)
    preflight.add_argument("--simulator-root", required=True, type=Path)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--contract", required=True, type=Path)
    pilot.add_argument("--run-dir", required=True, type=Path)
    pilot.add_argument("--output", required=True, type=Path)
    pilot.add_argument("--draws-per-block", type=int, default=10_000)
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--contract", required=True, type=Path)
    qualify.add_argument("--simulator-root", required=True, type=Path)
    qualify.add_argument("--run-dir", required=True, type=Path)
    qualify.add_argument("--output", required=True, type=Path)
    qualify.add_argument("--model-id", required=True, choices=("correct", "wrong"))
    qualify.add_argument("--seed-block", required=True, type=int)
    qualify.add_argument("--draws", type=int, default=DEFAULT_BATCH_SIZE)
    run = sub.add_parser("run")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--simulator-root", required=True, type=Path)
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--max-draws-per-block", type=int, default=5_000_050)
    return command


def main() -> int:
    args = parser().parse_args()
    action = sys.argv[1]
    if getattr(args, "simulator_root", None) is not None:
        args.simulator_root = args.simulator_root.resolve()
    if action == "preflight":
        contract = load_survey_contract(args.contract)
        _require_independent_contract(contract)
        result = _preflight(contract, args.simulator_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 3
    if action == "pilot":
        contract = load_survey_contract(args.contract)
        _require_independent_contract(contract)
        result = run_analytic_survey_pilot(
            args.contract,
            args.run_dir,
            args.output,
            draws_per_block=args.draws_per_block,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if result["verdict"] == "INVALID" else 3
    if action == "qualify":
        result = run_qualification(
            args.contract,
            args.simulator_root,
            args.run_dir,
            args.output,
            args.model_id,
            args.seed_block,
            args.draws,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = run_official(
        args.contract,
        args.simulator_root,
        args.run_dir,
        args.output,
        args.batch_size,
        args.max_draws_per_block,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == "PASSED":
        return 0
    return 2 if result["verdict"] == "INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
