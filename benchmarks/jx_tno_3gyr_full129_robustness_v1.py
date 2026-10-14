#!/usr/bin/env python3
"""Preregister the precision-first three-gigayear full129 TNO experiment.

The physical horizon is exactly 3,000,000,000 Julian years.  There is no
compute- or storage-budget ceiling.  The protocol is deliberately
prepare-only: it binds all twelve frozen full129 object batches and freezes a
precision-first CPU--GPU architecture in which the CPU constructs
exact-replay massive-source tiles and CUDA advances 1,548 massless initial
trajectories through 1,024 statistical stellar histories plus one tide-only
control.  H128 is the proposed primary resolution, with a complete H64
control, but neither long route is authorized until its staged qualification
and physical-model gates pass.  No execute or resume surface exists here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "benchmarks/jx_tno_3gyr_full129_robustness_v1.py"
TEST_PATH = PROJECT_ROOT / "tests/test_tno_3gyr_full129_robustness_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/tno_3gyr_full129_robustness_v1"

EXPECTED_BRANCH = "research/tno-3gyr-full129-robustness-v1-preregistered-20261014"
PREREGISTRATION_TAG = "tno-3gyr-full129-robustness-v1-preregistered-20261014"
FUTURE_RESULT_TAG = "tno-3gyr-full129-robustness-v1-result-20261014"
BASE_COMMIT = "db8f403fc7acaad90bcad0a8484438d12949232a"
BASE_TREE = "9ae4df89307a604a1c3276d2d156db5ee90e0372"

PREDECESSOR_TAG = "tno-10my-full129-trajectory-acquisition-v1-preregistered-20261011"
PREDECESSOR_TAG_OBJECT = "82f01f0a6b672b23a0a229ac2b4fd5a1dc95ae74"
PREDECESSOR_TAG_BYTES = 275
PREDECESSOR_TAG_SHA256 = "447b278c400e29eebe1e7ae806e771c7f3cf64bc8707497b511b192f5d5a946e"
PREDECESSOR_PROTOCOL_CONTENT_SHA256 = (
    "6bbf6cfd4ae7915a99a9573e96eff672a5ed6a31cafa183f04f43a5b59d31868"
)

PROTOCOL_SCHEMA = "jx.tno.3gyr-full129-robustness.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.3gyr-full129-robustness.protocol.v1\0"
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})
FILESYSTEM_TRUST_BOUNDARY = "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"


class TNOThreeBillionError(RuntimeError):
    """The prepare-only three-gigayear experiment boundary failed closed."""


@dataclass(frozen=True)
class FilePin:
    role: str
    relative_path: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class Target:
    batch_order: int
    object_id: str
    spk_id: int
    designation: str
    cohort: str


FILE_PINS = (
    FilePin(
        "predecessor_source",
        "benchmarks/jx_tno_10my_full129_trajectory_acquisition_v1.py",
        63_357,
        "0ca064027184b22f3100fce1cfb787a5e03e3d38c7081bbf9960e4f28d795fa7",
    ),
    FilePin(
        "predecessor_test",
        "tests/test_tno_10my_full129_trajectory_acquisition_v1.py",
        19_686,
        "369248855730e942bdaa571da7288bfb060ebdd05aa37156c38820e2c0fc8c5d",
    ),
    FilePin(
        "predecessor_protocol",
        "runs/tno_10my_full129_trajectory_acquisition_v1/PROTOCOL.json",
        27_099,
        "99e5cc4bf90e3a7e232e96538f2ad1e780d5113f12f362ad5f0e2c8e2f4a0e31",
    ),
    FilePin(
        "full129_bridge",
        "runs/tno_full128_reduced5_bridge_v1/BRIDGE.json",
        1_208_834,
        "0570e7d81917caad871de2b2a143d4fc083851894d3c72aab1b77f8e9591fcec",
    ),
    FilePin(
        "cohort_custody",
        "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json",
        16_520,
        "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288",
    ),
    FilePin(
        "grid_adapter_source",
        "benchmarks/jx_tno_10my_reduced5_grid_adapter_v1.py",
        111_493,
        "324498cd36726e461ab6910fdf0fe13a1b024346293558f535bea557cdb61b3a",
    ),
    FilePin(
        "grid_adapter_test",
        "tests/test_tno_10my_reduced5_grid_adapter_v1.py",
        37_454,
        "909e6f63ff229e8909245001728acd58a8d4bb7df80993a14b578659c6dd96b8",
    ),
    FilePin(
        "grid_adapter_protocol",
        "runs/tno_10my_reduced5_grid_adapter_v1/PROTOCOL.json",
        11_844,
        "6b10412f06f3c398646179fd163347ac269c03b08486c867e4da821e082be3f2",
    ),
    FilePin(
        "cpu_massive_source",
        "src/jxplanetx/restricted_wh_source.py",
        61_803,
        "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52",
    ),
    FilePin(
        "gpu_quiet_tracer",
        "src/jxplanetx/cuda_restricted_wh.py",
        74_758,
        "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927",
    ),
    FilePin(
        "gpu_hybrid_route",
        "src/jxplanetx/cuda_restricted_wh_hybrid.py",
        37_420,
        "6b3ea46436dbdb7dc2c434cff005a88395fea6e00cd97c7be104650c75769b10",
    ),
    FilePin(
        "lifecycle_route",
        "src/jxplanetx/restricted_wh_lifecycle.py",
        38_996,
        "b0211778fcf51dbfef78049e74bafd05d5daee4db7dd18f743f4a5b5a14af346",
    ),
    FilePin(
        "b22_classifier_source",
        "benchmarks/jx_tno_b22_classifier_v2.py",
        94_534,
        "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98",
    ),
    FilePin(
        "classification_pilot_source",
        "benchmarks/jx_tno_10my_reduced5_classification_pilot_v2.py",
        59_307,
        "33ad38cbc89a15a83cf180161544a721b8635affd0d4890a8a39d04389694332",
    ),
    FilePin(
        "classification_pilot_test",
        "tests/test_tno_10my_reduced5_classification_pilot_v2.py",
        28_773,
        "060793a45cd91590d484a610b9c6d48f6fae6c2a468c6e2663541599a831fbd8",
    ),
    FilePin(
        "classification_pilot_protocol",
        "runs/tno_10my_reduced5_classification_pilot_v2/PROTOCOL.json",
        34_347,
        "b510cbf7a62f45d8713ca8ef02dd2229f5a79966d00d2482843e814b939cded6",
    ),
)

TARGETS = (
    Target(0, "jpl-20090377", 20_090_377, "90377", "DETACHED_TNO_SCREEN_V1"),
    Target(1, "jpl-20474640", 20_474_640, "474640", "DETACHED_TNO_SCREEN_V1"),
    Target(2, "jpl-54526835", 54_526_835, "2023 KQ14", "DETACHED_TNO_SCREEN_V1"),
    Target(3, "jpl-20541132", 20_541_132, "541132", "DETACHED_TNO_SCREEN_V1"),
    Target(4, "jpl-50666516", 50_666_516, "2012 VP113", "DETACHED_TNO_SCREEN_V1"),
    Target(5, "jpl-20148209", 20_148_209, "148209", "DETACHED_TNO_SCREEN_V1"),
    Target(6, "jpl-20574372", 20_574_372, "574372", "SCATTERING_DISK_CONTROL_V1"),
    Target(7, "jpl-20303775", 20_303_775, "303775", "SCATTERING_DISK_CONTROL_V1"),
    Target(8, "jpl-20026181", 20_026_181, "26181", "SCATTERING_DISK_CONTROL_V1"),
    Target(9, "jpl-20042301", 20_042_301, "42301", "SCATTERING_DISK_CONTROL_V1"),
    Target(10, "jpl-20026375", 20_026_375, "26375", "SCATTERING_DISK_CONTROL_V1"),
    Target(11, "jpl-20015874", 20_015_874, "15874", "SCATTERING_DISK_CONTROL_V1"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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
    except (TypeError, ValueError, RecursionError) as exc:
        raise TNOThreeBillionError("value is not canonical-JSON encodable") from exc


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TNOThreeBillionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_number(token: str) -> None:
    raise TNOThreeBillionError(f"floating or nonfinite JSON token rejected: {token}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except TNOThreeBillionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TNOThreeBillionError("payload is not strict ASCII JSON") from exc
    if type(value) is not dict:
        raise TNOThreeBillionError("JSON root must be an object")
    return value


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = [absolute]
    chain.extend(absolute.parents)
    for component in reversed(chain):
        if component.exists() and component.is_symlink():
            raise TNOThreeBillionError(f"{label} has a symlink ancestor")


@contextmanager
def _held_directory(path: Path, label: str) -> Iterator[int]:
    _require_no_symlink_ancestors(path, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TNOThreeBillionError(f"cannot open {label}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise TNOThreeBillionError(f"{label} is not a directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_file(path: Path, *, expected_bytes: int | None, expected_sha256: str | None, label: str) -> bytes:
    _require_no_symlink_ancestors(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TNOThreeBillionError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TNOThreeBillionError(f"{label} must be a single-link regular file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise TNOThreeBillionError(f"{label} byte length differs")
        if before.st_size > 2_000_000:
            raise TNOThreeBillionError(f"{label} exceeds the bounded read ceiling")
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - len(chunks)), len(chunks))
            if not chunk:
                raise TNOThreeBillionError(f"{label} ended early")
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise TNOThreeBillionError(f"{label} changed while held")
    finally:
        os.close(descriptor)
    payload = bytes(chunks)
    if expected_sha256 is not None and _sha256(payload) != expected_sha256:
        raise TNOThreeBillionError(f"{label} SHA-256 differs")
    return payload


def _git(arguments: Sequence[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(PROJECT_ROOT), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise TNOThreeBillionError(f"Git check failed for {label}")
    return completed.stdout


def _git_text(arguments: Sequence[str], label: str) -> str:
    try:
        return _git(arguments, label).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise TNOThreeBillionError(f"Git output is not ASCII for {label}") from exc


def _verify_git_authority() -> dict[str, Any]:
    branch = _git_text(["symbolic-ref", "--short", "HEAD"], "current branch")
    if branch != EXPECTED_BRANCH:
        raise TNOThreeBillionError("current branch differs from the preregistered branch")
    if _git_text(["rev-parse", f"{BASE_COMMIT}^{{tree}}"], "base tree") != BASE_TREE:
        raise TNOThreeBillionError("base tree differs")
    ancestor = subprocess.run(
        ["git", "-C", os.fspath(PROJECT_ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise TNOThreeBillionError("base commit is not an ancestor of HEAD")
    if _git_text(["cat-file", "-t", PREDECESSOR_TAG], "predecessor tag type") != "tag":
        raise TNOThreeBillionError("predecessor tag is not annotated")
    if _git_text(["rev-parse", PREDECESSOR_TAG], "predecessor tag object") != PREDECESSOR_TAG_OBJECT:
        raise TNOThreeBillionError("predecessor tag object differs")
    tag_payload = _git(["cat-file", "tag", PREDECESSOR_TAG], "predecessor tag payload")
    if len(tag_payload) != PREDECESSOR_TAG_BYTES or _sha256(tag_payload) != PREDECESSOR_TAG_SHA256:
        raise TNOThreeBillionError("predecessor tag payload differs")
    if _git_text(["rev-parse", f"{PREDECESSOR_TAG}^{{commit}}"], "predecessor commit") != BASE_COMMIT:
        raise TNOThreeBillionError("predecessor tag commit differs")
    return {
        "base_commit": BASE_COMMIT,
        "base_tree": BASE_TREE,
        "expected_branch": EXPECTED_BRANCH,
        "predecessor_tag": PREDECESSOR_TAG,
        "predecessor_tag_object": PREDECESSOR_TAG_OBJECT,
        "predecessor_tag_payload_bytes": PREDECESSOR_TAG_BYTES,
        "predecessor_tag_payload_sha256": PREDECESSOR_TAG_SHA256,
    }


def _verify_file_pin(pin: FilePin) -> tuple[dict[str, Any], bytes]:
    if Path(pin.relative_path).is_absolute() or ".." in Path(pin.relative_path).parts:
        raise TNOThreeBillionError("file pin path is not contained")
    live = _read_file(
        PROJECT_ROOT / pin.relative_path,
        expected_bytes=pin.byte_length,
        expected_sha256=pin.sha256,
        label=pin.role,
    )
    committed = _git(["show", f"{BASE_COMMIT}:{pin.relative_path}"], pin.role)
    if committed != live:
        raise TNOThreeBillionError(f"{pin.role} live bytes differ from the base commit")
    return (
        {
            "role": pin.role,
            "relative_path": pin.relative_path,
            "byte_length": pin.byte_length,
            "sha256": pin.sha256,
            "base_commit": BASE_COMMIT,
            "live_equals_committed": True,
        },
        live,
    )


def _target_records() -> list[dict[str, Any]]:
    return [
        {
            "batch_order": target.batch_order,
            "object_id": target.object_id,
            "spk_id": target.spk_id,
            "designation": target.designation,
            "cohort": target.cohort,
            "included": True,
        }
        for target in TARGETS
    ]


def _validate_rosters(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    predecessor = _strict_json(payloads["predecessor_protocol"])
    if (
        predecessor.get("schema") != "jx.tno.10my-full129-trajectory-acquisition.protocol.v1"
        or predecessor.get("content_sha256") != PREDECESSOR_PROTOCOL_CONTENT_SHA256
        or predecessor.get("status") != "PREREGISTRATION_CANDIDATE_PREPARE_ONLY_EXECUTION_NOT_IMPLEMENTED"
    ):
        raise TNOThreeBillionError("predecessor protocol identity differs")
    geometry = predecessor.get("production_geometry")
    if type(geometry) is not dict or any(
        type(geometry.get(key)) is not int or geometry.get(key) != expected
        for key, expected in {
            "batch_count": 12,
            "tracer_count_per_batch": 129,
            "steps_per_julian_year": 64,
            "shard_steps": 640_000,
            "shards_per_batch": 1_000,
            "b22_node_count_per_batch": 50_001,
            "final_binary_bytes": 4_743_277_848,
        }.items()
    ):
        raise TNOThreeBillionError("predecessor production geometry differs")

    bridge = json.loads(payloads["full129_bridge"].decode("ascii"))
    bridge_rows = bridge.get("batches")
    if type(bridge_rows) is not list or len(bridge_rows) != 12:
        raise TNOThreeBillionError("full129 bridge roster differs")
    bridge_identity = [
        (row.get("batch_order"), row.get("object_id"), row.get("spk_id"))
        for row in bridge_rows
    ]
    expected_identity = [(t.batch_order, t.object_id, t.spk_id) for t in TARGETS]
    if bridge_identity != expected_identity:
        raise TNOThreeBillionError("full129 bridge identities differ")

    custody = _strict_json(payloads["cohort_custody"])
    records = custody.get("records")
    if type(records) is not list or len(records) != 12:
        raise TNOThreeBillionError("cohort custody roster differs")
    custody_identity = [
        (row.get("object_id"), row.get("spk_id"), row.get("primary_designation"), row.get("cohort"))
        for row in records
    ]
    expected_custody = [(t.object_id, t.spk_id, t.designation, t.cohort) for t in TARGETS]
    if custody_identity != expected_custody:
        raise TNOThreeBillionError("cohort custody identities differ")
    return {
        "predecessor_protocol_content_sha256": PREDECESSOR_PROTOCOL_CONTENT_SHA256,
        "full129_batch_count": 12,
        "tracer_count_per_batch": 129,
        "cohort_counts": {
            "DETACHED_TNO_SCREEN_V1": 6,
            "SCATTERING_DISK_CONTROL_V1": 6,
        },
        "bridge_content_sha256": bridge.get("content_sha256"),
        "cohort_custody_content_sha256": custody.get("content_sha256"),
    }


def validate_bound_prerequisites() -> dict[str, Any]:
    git_authority = _verify_git_authority()
    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for pin in FILE_PINS:
        record, payload = _verify_file_pin(pin)
        records.append(record)
        payloads[pin.role] = payload
    roster_summary = _validate_rosters(payloads)
    return {
        "git_authority": git_authority,
        "exact_files": records,
        "roster_summary": roster_summary,
        "authority_chain": (
            "THE_ANNOTATED_PREDECESSOR_COMMIT_AND_EXACT_COMMITTED_FILES_BIND_INPUTS_"
            "INTEGRATOR_COMPONENTS_AND_CLASSIFIER_BYTES;THEY_DO_NOT_AUTHORIZE_EXECUTION"
        ),
    }


def _implementation_record(path: Path, role: str) -> dict[str, Any]:
    payload = _read_file(path, expected_bytes=None, expected_sha256=None, label=role)
    return {
        "role": role,
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def _claims() -> dict[str, bool]:
    return {
        "prepare_only_protocol_frozen": True,
        "execution_implemented": False,
        "execution_authorized": False,
        "cpu_source_executed": False,
        "gpu_tracers_executed": False,
        "all_12_h64_qualification_passed": False,
        "all_12_h128_qualification_passed": False,
        "h128_route_qualified": False,
        "canonical_shared_source_service_qualified": False,
        "distributed_batch_runtime_qualified": False,
        "guarded_cpu_route_qualified": False,
        "independent_all129_ias15_reference_pinned": False,
        "external_forcing_work_torque_balance_qualified": False,
        "three_gyr_physical_model_qualified": False,
        "h64_three_gyr_control_completed": False,
        "h128_three_gyr_primary_completed": False,
        "first_ten_myr_b22_classification_generated": False,
        "three_gyr_survival_record_generated": False,
        "cross_resolution_statistical_comparison_generated": False,
        "population_inference": False,
        "planet_nine_evidence": False,
        "science_result": False,
    }


def build_protocol() -> dict[str, Any]:
    prerequisites = validate_bound_prerequisites()
    roster = _target_records()
    if [record["batch_order"] for record in roster] != list(range(12)):
        raise TNOThreeBillionError("the complete frozen roster differs")

    horizon_years = 3_000_000_000
    batch_count = 12
    tracers_per_batch = 129
    statistical_stellar_history_count = 1_024
    tide_only_control_count = 1
    forcing_context_count = statistical_stellar_history_count + tide_only_control_count
    h64_steps_per_batch = horizon_years * 64
    h128_steps_per_batch = horizon_years * 128
    h64_batch_steps_per_context = batch_count * h64_steps_per_batch
    h128_batch_steps_per_context = batch_count * h128_steps_per_batch
    dual_batch_steps_per_context = (
        h64_batch_steps_per_context + h128_batch_steps_per_context
    )
    h64_ensemble_batch_steps = h64_batch_steps_per_context * forcing_context_count
    h128_ensemble_batch_steps = h128_batch_steps_per_context * forcing_context_count
    dual_ensemble_batch_steps = h64_ensemble_batch_steps + h128_ensemble_batch_steps
    maximum_tracer_lane_slot_steps = dual_ensemble_batch_steps * tracers_per_batch
    canonical_source_steps_per_context = h64_steps_per_batch + h128_steps_per_batch
    ensemble_canonical_source_steps = (
        canonical_source_steps_per_context * forcing_context_count
    )
    exact_replay_cpu_advances = ensemble_canonical_source_steps * 2
    h64_shards_per_batch = h64_steps_per_batch // 640_000
    h128_shards_per_batch = h128_steps_per_batch // 640_000
    h64_total_shards_per_context = batch_count * h64_shards_per_batch
    h128_total_shards_per_context = batch_count * h128_shards_per_batch
    dual_total_shards_per_context = (
        h64_total_shards_per_context + h128_total_shards_per_context
    )
    h64_ensemble_total_shards = h64_total_shards_per_context * forcing_context_count
    h128_ensemble_total_shards = h128_total_shards_per_context * forcing_context_count
    dual_ensemble_total_shards = (
        h64_ensemble_total_shards + h128_ensemble_total_shards
    )
    canonical_source_tiles = (
        h64_shards_per_batch + h128_shards_per_batch
    ) * forcing_context_count
    h64_physical_fragments_per_batch = h64_shards_per_batch + 2
    h128_physical_fragments_per_batch = h128_shards_per_batch + 1
    ensemble_scheduled_physical_batch_fragments = (
        h64_physical_fragments_per_batch
        + h128_physical_fragments_per_batch
    ) * batch_count * forcing_context_count
    unique_canonical_source_physical_fragments = (
        h64_physical_fragments_per_batch
        + h128_physical_fragments_per_batch
    ) * forcing_context_count
    exact_replay_source_fragment_attempts = (
        unique_canonical_source_physical_fragments * 2
    )
    distributed_worker_units = forcing_context_count * 2 * batch_count
    retained_nodes_per_batch = horizon_years // 200 + 1
    ten_myr_window_count = horizon_years // 10_000_000
    geometry = (
        h64_steps_per_batch,
        h128_steps_per_batch,
        h64_batch_steps_per_context,
        h128_batch_steps_per_context,
        dual_batch_steps_per_context,
        h64_ensemble_batch_steps,
        h128_ensemble_batch_steps,
        dual_ensemble_batch_steps,
        maximum_tracer_lane_slot_steps,
        canonical_source_steps_per_context,
        ensemble_canonical_source_steps,
        exact_replay_cpu_advances,
        h64_shards_per_batch,
        h128_shards_per_batch,
        h64_total_shards_per_context,
        h128_total_shards_per_context,
        dual_total_shards_per_context,
        h64_ensemble_total_shards,
        h128_ensemble_total_shards,
        dual_ensemble_total_shards,
        canonical_source_tiles,
        h64_physical_fragments_per_batch,
        h128_physical_fragments_per_batch,
        ensemble_scheduled_physical_batch_fragments,
        unique_canonical_source_physical_fragments,
        exact_replay_source_fragment_attempts,
        distributed_worker_units,
        retained_nodes_per_batch,
        ten_myr_window_count,
    )
    expected_geometry = (
        192_000_000_000,
        384_000_000_000,
        2_304_000_000_000,
        4_608_000_000_000,
        6_912_000_000_000,
        2_361_600_000_000_000,
        4_723_200_000_000_000,
        7_084_800_000_000_000,
        913_939_200_000_000_000,
        576_000_000_000,
        590_400_000_000_000,
        1_180_800_000_000_000,
        300_000,
        600_000,
        3_600_000,
        7_200_000,
        10_800_000,
        3_690_000_000,
        7_380_000_000,
        11_070_000_000,
        922_500_000,
        300_002,
        600_001,
        11_070_036_900,
        922_503_075,
        1_845_006_150,
        24_600,
        15_000_001,
        300,
    )
    if geometry != expected_geometry:
        raise TNOThreeBillionError("three-gigayear geometry arithmetic differs")

    stop_years = (
        1_000,
        5_000,
        10_000,
        100_000,
        500_000,
        5_000_000,
        10_000_000,
        100_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
    )

    def stop_records(steps_per_year: int) -> list[dict[str, int]]:
        records: list[dict[str, int]] = []
        previous_step = 0
        for year in stop_years:
            global_step = year * steps_per_year
            delta_steps = global_step - previous_step
            records.append({
                "julian_year": year,
                "global_step": global_step,
                "completed_640k_shards_per_batch": global_step // 640_000,
                "remainder_steps_in_current_fragment": global_step % 640_000,
                "newly_completed_logical_640k_shards_since_previous_stop": (
                    global_step // 640_000 - previous_step // 640_000
                ),
                "partial_fragment_remainder_steps_since_previous_stop": (
                    delta_steps % 640_000
                ),
            })
            previous_step = global_step
        return records

    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREREGISTRATION_CANDIDATE_PREPARE_ONLY_EXECUTION_NOT_IMPLEMENTED",
        "title": (
            "JX Scientific Engine three-gigayear H128 CPU-GPU TNO "
            "robustness experiment v1"
        ),
        "objective": (
            "Evolve all twelve frozen full129 object batches forward for exactly "
            "3,000,000,000 Julian years through 1,024 statistical stellar histories "
            "and one Galactic-tide-only control at H128, repeat every paired context "
            "at H64, and report only model-conditional chaotic-ensemble, lifecycle, "
            "and resolution-sensitivity records after every gate passes."
        ),
        "evidence_class": "PREPARED_PREREGISTRATION_NO_MODEL_OUTPUT",
        "claim_ceiling": "PREPARED_PROTOCOL_ONLY_NO_EXECUTION_OR_SCIENCE_RESULT",
        "future_result_evidence_class": "MODEL_OUTPUT",
        "future_result_claim_ceiling": (
            "TWELVE_OBJECT_1025_FORCING_CONTEXT_FORWARD_3GYR_CHAOTIC_ENSEMBLE_MODEL_OUTPUT_"
            "CONDITIONAL_ON_FROZEN_INITIAL_COVARIANCE_FORCE_MODEL_AND_RUNTIME"
        ),
        "implementation": {
            "source": _implementation_record(SOURCE_PATH, "3 Gyr preregistration source"),
            "test": _implementation_record(TEST_PATH, "3 Gyr preregistration test"),
        },
        "authority": prerequisites,
        "time_contract": {
            "direction": "FORWARD",
            "candidate_common_epoch_jd_tdb": "2461200.5",
            "candidate_epoch_basis": (
                "STANDARD_EPOCH_RECORDED_FOR_ALL_TWELVE_SBDB_CUSTODY_ROWS"
            ),
            "native_covariance_epochs_are_object_specific": True,
            "qualified_full_covariance_propagation_to_common_epoch_required": True,
            "duration_julian_years": horizon_years,
            "duration_julian_days": 1_095_750_000_000,
            "time_representation": (
                "SIGNED_INT64_GLOBAL_STEP_PLUS_EXACT_BINARY_POWER_OF_TWO_STEP"
            ),
            "floating_point_time_accumulation_forbidden": True,
        },
        "roster": {
            "policy": "ALL_TWELVE_FROZEN_BATCHES_NO_SELECTION_OR_SUBSTITUTION",
            "records": roster,
            "batch_count": batch_count,
            "tracers_per_batch": tracers_per_batch,
            "trajectory_count": batch_count * tracers_per_batch,
            "forcing_context_count": forcing_context_count,
            "trajectory_instances_across_forcing_contexts": (
                batch_count * tracers_per_batch * forcing_context_count
            ),
            "nominal_trajectories": 12,
            "covariance_clones_per_batch": 128,
            "clones_are_alternative_massless_worlds_not_simultaneous_bodies": True,
            "outcome_blind_scope": (
                "PROSPECTIVE_BEYOND_ALREADY_OBSERVED_SHORT_HORIZON_LINEAGE"
            ),
        },
        "resource_policy": {
            "compute_budget_limit_imposed": False,
            "storage_budget_limit_imposed": False,
            "precision_and_complete_custody_precede_speed_or_cost": True,
            "charged_step_definition": (
                "ONE_ATTEMPTED_FIXED_MAP_STEP_FOR_ONE_FULL129_BATCH;"
                "FAILED_AND_REPLAYED_ATTEMPTS_REMAIN_REPORTED"
            ),
            "maximum_automatic_precommit_interruption_retries_per_physical_fragment": 1,
            "automatic_retry_scope": (
                "PROCESS_DEVICE_OR_IO_INTERRUPTION_BEFORE_CREATE_ONLY_COMMIT_ONLY"
            ),
            "nan_hash_numerical_or_route_failure_automatic_retries": 0,
            "after_retry_failure": (
                "STOP_RETAIN_FAILURE_AND_REQUIRE_A_NEW_PREREGISTRATION"
            ),
            "selective_outcome_dependent_retries_forbidden": True,
            "all_attempts_and_adaptive_route_accounting_retained": True,
            "wall_runtime_projection_frozen": False,
            "runtime_requires_measured_qualified_cpu_gpu_route_and_io_throughput": True,
        },
        "work_geometry": {
            "h64_steps_per_batch": h64_steps_per_batch,
            "h128_steps_per_batch": h128_steps_per_batch,
            "forcing_context_count": forcing_context_count,
            "per_context_h64_batch_map_steps": h64_batch_steps_per_context,
            "per_context_h128_batch_map_steps": h128_batch_steps_per_context,
            "per_context_dual_resolution_batch_map_steps": (
                dual_batch_steps_per_context
            ),
            "ensemble_h64_batch_map_steps": h64_ensemble_batch_steps,
            "ensemble_h128_batch_map_steps": h128_ensemble_batch_steps,
            "ensemble_dual_resolution_batch_map_steps": dual_ensemble_batch_steps,
            "ensemble_maximum_all_lanes_active_tracer_step_capacity": (
                maximum_tracer_lane_slot_steps
            ),
            "per_context_canonical_shared_source_steps": (
                canonical_source_steps_per_context
            ),
            "ensemble_canonical_shared_source_steps": ensemble_canonical_source_steps,
            "ensemble_canonical_shared_source_advances_with_complete_exact_replay": (
                exact_replay_cpu_advances
            ),
            "actual_attempted_tracer_steps_depend_on_frozen_terminal_lifecycle": True,
            "qualification_failed_and_crash_replay_work_is_additional": True,
            "h64_shards_per_batch": h64_shards_per_batch,
            "h128_shards_per_batch": h128_shards_per_batch,
            "per_context_h64_total_batch_shards": h64_total_shards_per_context,
            "per_context_h128_total_batch_shards": h128_total_shards_per_context,
            "per_context_dual_resolution_total_batch_shards": (
                dual_total_shards_per_context
            ),
            "ensemble_h64_total_batch_shards": h64_ensemble_total_shards,
            "ensemble_h128_total_batch_shards": h128_ensemble_total_shards,
            "ensemble_dual_resolution_total_batch_shards": (
                dual_ensemble_total_shards
            ),
            "batch_shard_counts_are_640k_logical_equivalents": True,
            "h64_scheduled_physical_fragments_per_batch": (
                h64_physical_fragments_per_batch
            ),
            "h128_scheduled_physical_fragments_per_batch": (
                h128_physical_fragments_per_batch
            ),
            "ensemble_scheduled_physical_batch_fragment_count_with_prefix_stops": (
                ensemble_scheduled_physical_batch_fragments
            ),
            "ensemble_canonical_source_tile_count": canonical_source_tiles,
            "canonical_source_tile_count_is_640k_logical_equivalents": True,
            "ensemble_unique_canonical_source_physical_fragment_count_with_prefix_stops": (
                unique_canonical_source_physical_fragments
            ),
            "ensemble_exact_replay_canonical_source_physical_fragment_attempts_before_retries": (
                exact_replay_source_fragment_attempts
            ),
            "distributed_context_resolution_batch_worker_units": (
                distributed_worker_units
            ),
            "source_steps_per_logical_640k_tile": 640_000,
            "logical_640k_tile_within_provider_one_million_step_ceiling": True,
            "h64_logical_640k_tile_julian_years": 10_000,
            "h128_logical_640k_tile_julian_years": 5_000,
            "h64_stop_schedule": stop_records(64),
            "h128_stop_schedule": stop_records(128),
        },
        "cpu_gpu_integrator": {
            "architecture": (
                "CPU_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN_MASSIVE_SOURCE_"
                "PLUS_GPU_FUSED_RESTRICTED_WISDOM_HOLMAN_QUIET_TRACERS_"
                "PLUS_CPU_DOP853_GUARDED_ROUTE"
            ),
            "precision_precedes_speed": True,
            "massive_source": {
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
                "callable": (
                    "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile"
                ),
                "source_body_count": 5,
                "complete_same_runtime_bitwise_replay_required": True,
                "canonical_stream_scope": "ONE_PER_FORCING_CONTEXT_AND_RESOLUTION",
                "redundant_per_batch_source_rebuild_forbidden": True,
                "full_physical_fragment_sha256_required": True,
                "physical_fragment_start_and_end_state_receipts_required": True,
            },
            "tracer_propagation": {
                "backend": "cupy",
                "device": "cuda:0",
                "dtype": "float64",
                "quiet_map": "CUDA_RESTRICTED_WH_FIXED_STEP",
                "tracers_per_batch": 129,
                "threads_per_block": 128,
                "deterministic_reductions": True,
                "fast_math": False,
                "cpu_fallback_allowed": False,
            },
            "guarded_route": {
                "backend": "numpy_cpu_workers",
                "method": "CONTINUOUS_RESTRICTED_FRAME_DOP853",
                "transaction": (
                    "DISCARD_PROVISIONAL_GPU_STEP_AND_REDO_ONLY_THE_GUARDED_LANE"
                ),
                "currently_qualified_for_3gyr": False,
                "route_and_removal_counts_are_outcomes_not_gates": True,
            },
            "coupling": {
                "source_steps_per_gpu_map_step": 1,
                "source_physical_fragment_exact_replay_and_sha256_required": True,
                "gpu_consumes_exact_verified_cpu_source_fragment_bytes": True,
                "canonical_source_fragment_broadcast_to_all_twelve_batch_workers": True,
                "every_batch_worker_verifies_full_fragment_sha256_before_use": True,
                "external_history_event_stream_sha256_required": True,
                "gpu_quiet_and_cpu_guarded_routes_share_exact_external_history_cursor": True,
                "external_forcing_gpu_and_cpu_route_kernels_implemented": False,
                "mismatch_action": "STOP_BEFORE_WRITING_SUCCESS",
                "runtime_device_driver_library_kernel_receipts_required": True,
                "device_uuid_is_runtime_evidence_not_preregistration_identity": True,
                "twelve_gpu_batches_are_independent_workers_after_source_verification": True,
                "one_batch_context_and_resolution_stays_on_one_qualified_runtime": True,
                "paired_h64_h128_batch_context_runs_use_same_device_uuid_sequentially": True,
                "runtime_migration_without_separate_bitwise_qualification_allowed": False,
            },
            "distributed_execution_contract": {
                "worker_unit_count": distributed_worker_units,
                "worker_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                ],
                "signed_int64_fields": [
                    "forcing_context_index",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                    "logical_640k_shard_index",
                    "physical_fragment_index",
                    "start_global_step",
                    "end_global_step",
                ],
                "physical_batch_fragment_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                    "start_global_step",
                    "end_global_step",
                ],
                "canonical_source_fragment_identity_fields_in_order": [
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "start_global_step",
                    "end_global_step",
                ],
                "physical_fragment_index_scope": "ZERO_BASED_PER_STREAM",
                "first_physical_fragment_index_is_zero": True,
                "physical_fragment_indices_increment_by_exactly_one": True,
                "physical_fragments_are_nonoverlapping_and_contiguous": True,
                "first_physical_fragment_start_global_step_is_zero": True,
                "successor_start_step_equals_predecessor_end_step": True,
                "final_physical_fragment_end_global_step_equals_resolution_horizon_step": True,
                "logical_640k_shard_index_retained_for_capacity_accounting_only": True,
                "canonical_aggregation_order": (
                    "FORCING_CONTEXT_THEN_RESOLUTION_H64_H128_THEN_BATCH_0_TO_11_"
                    "THEN_PHYSICAL_FRAGMENT_START_STEP_ASCENDING"
                ),
                "idempotent_create_only_physical_fragment_commit_required": True,
                "fixed_integer_counts_and_fixed_reduction_order_required": True,
                "runtime_migration_requires_separate_bitwise_qualification": True,
                "h64_h128_pair_same_device_uuid_sequential_execution_required": True,
            },
            "fragment_and_restart_contract": {
                "canonical_incremental_prefix_fragments": {
                    "h64": [
                        {"stop_julian_year": 1_000, "steps": 64_000},
                        {"stop_julian_year": 5_000, "steps": 256_000},
                        {"stop_julian_year": 10_000, "steps": 320_000},
                    ],
                    "h128": [
                        {"stop_julian_year": 1_000, "steps": 128_000},
                        {"stop_julian_year": 5_000, "steps": 512_000},
                        {"stop_julian_year": 10_000, "steps": 640_000},
                    ],
                },
                "all_stops_after_10k_are_640k_step_aligned": True,
                "restart_payload_fields": [
                    "signed_int64_start_global_step",
                    "signed_int64_end_global_step",
                    "signed_int64_logical_640k_shard_index",
                    "signed_int64_physical_fragment_index",
                    "forcing_context_id",
                    "resolution_steps_per_julian_year",
                    "batch_order",
                    "all_129_tracer_state_and_lifecycle",
                    "terminal_and_removal_ledger",
                    "pending_route_state_or_exact_proof_none",
                    "external_history_cursor",
                    "canonical_source_start_state",
                    "canonical_source_end_state",
                    "canonical_source_full_physical_fragment_sha256",
                    "predecessor_manifest_sha256",
                    "code_model_runtime_receipts",
                ],
                "fresh_process_bitwise_suffix_validation_required": True,
                "retry_eligible_failures": [
                    "PRECOMMIT_PROCESS_INTERRUPTION",
                    "PRECOMMIT_DEVICE_INTERRUPTION",
                    "PRECOMMIT_IO_INTERRUPTION",
                ],
                "retry_requires_identical_pinned_runtime_and_inputs": True,
                "nan_hash_numerical_or_route_failure_action": "STOP_WITHOUT_AUTOMATIC_RETRY",
            },
            "h128_primary": {
                "current_status": (
                    "CANDIDATE_REQUIRES_STAGED_QUALIFICATION_BEFORE_EXECUTION"
                ),
                "source_step_julian_years_binary64_hex": "0x1.0000000000000p-7",
                "steps_per_julian_year": 128,
                "binary64_not_128_bit_arithmetic": True,
                "horizon_julian_years": horizon_years,
                "shards_per_batch": h128_shards_per_batch,
                "qualification_stop_years": list(stop_years),
            },
            "h64_complete_control": {
                "source_step_julian_years_binary64_hex": "0x1.0000000000000p-6",
                "steps_per_julian_year": 64,
                "horizon_julian_years": horizon_years,
                "shards_per_batch": h64_shards_per_batch,
            },
            "short_horizon_qualification_contract": {
                "scope": {
                    "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
                    "horizon_julian_years": 1_000,
                    "checkpoint_interval_julian_years": 1,
                    "checkpoint_count_endpoint_inclusive": 1_001,
                    "batch_orders": list(range(12)),
                    "tracer_orders": list(range(129)),
                    "reference_integrator": (
                        "INDEPENDENT_CPU_IAS15_EXACT_IMPLEMENTATION_AND_RUNTIME_TO_BE_PINNED"
                    ),
                    "common_epoch_bridge_must_pass_first": True,
                },
                "h64_h128_richardson_divisor": 3,
                "richardson_normalized_semimajor_axis_max": "3/8000",
                "richardson_normalized_semimajor_axis_formula": (
                    "MAX_OVER_CHECKPOINTS_ABS_A_LEFT_MINUS_A_RIGHT_DIV_"
                    "MAX_A_LEFT_INITIAL_A_RIGHT_INITIAL"
                ),
                "richardson_cartesian_position_au_max": "1/500",
                "richardson_cartesian_velocity_au_per_julian_year_max": "1/50000",
                "h128_vs_ias15_normalized_semimajor_axis_max": "3/4000",
                "h128_vs_ias15_scattering_diagnostic_disagreement_max": "3/4000",
                "h128_vs_ias15_cartesian_position_au_max": "1/500",
                "h128_vs_ias15_cartesian_velocity_au_per_julian_year_max": "1/50000",
                "scalar_contraction_rule": (
                    "FINE_LE_EXACT_64_ULP_FLOOR_OR_COARSE_GE_3_TIMES_FINE"
                ),
                "vector_contraction_rule": (
                    "FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_"
                    "COARSE_SQUARED_GE_9_TIMES_FINE_SQUARED"
                ),
                "ulp_floor_rule": (
                    "PER_CHECKPOINT_EXACT_64_TIMES_REFERENCE_SCALAR_ULP;_"
                    "VECTOR_FLOOR_IS_EXACT_SUM_OF_THREE_SQUARED_COMPONENT_FLOORS;_"
                    "TAKE_MAXIMUM_OVER_THE_SAME_1001_CHECKPOINTS"
                ),
                "energy_contraction_rule": (
                    "FINE_LE_MAX_OF_COARSE_OVER_3_AND_2_POW_MINUS_46"
                ),
                "all_node_fine_massive_source_relative_energy_drift_max": (
                    "1/100000000"
                ),
                "all_node_fine_massive_source_angular_vector_drift_max": (
                    "1/10000000000"
                ),
                "all_individual_predicates_must_pass": True,
                "energy_and_angular_gates_apply_only_to_autonomous_control": True,
                "external_forcing_work_and_torque_balance_residual_contract_frozen": False,
                "external_forcing_lanes_cannot_use_autonomous_energy_or_angular_gates": True,
            },
            "resolved11_h128_prior_evidence": {
                "status": "UNBOUND_LOCAL_DIAGNOSTIC_ONLY",
                "qualification": "EXECUTED_GATES_NOT_ALL_PASS",
                "passed_gate_count": 12,
                "total_gate_count": 20,
                "selected9_tracer_gates_all_passed": True,
                "massive_source_gate_failure_count": 8,
                "reduced5_or_full129_h128_authority": False,
                "scientific_disproof": False,
            },
            "pointwise_h64_h128_closeness_after_chaotic_divergence_is_a_gate": False,
            "globally_symplectic_with_external_source_tile": False,
            "long_horizon_accuracy_qualified": False,
            "silent_backend_integrator_step_or_force_change_allowed": False,
        },
        "physical_model": {
            "bound_control_model": (
                "ISOLATED_NEWTONIAN_REDUCED5_PLUS_129_MASSLESS_TNO_STATES_PER_BATCH"
            ),
            "reduced5_massive_sources": [
                "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
                "JUPITER_BARYCENTER",
                "SATURN_BARYCENTER",
                "URANUS_BARYCENTER",
                "NEPTUNE_BARYCENTER",
            ],
            "tno_backreaction": False,
            "mutual_tno_gravity": False,
            "control_model_is_a_real_solar_system_prediction": False,
            "minimum_model_ladder_before_broad_physical_interpretation": [
                "QUALIFIED_COMMON_EPOCH_FULL_COVARIANCE_BRIDGE",
                "RESOLVED_OR_VALIDATED_MULTIRATE_KNOWN_PLANETARY_MODEL_WITH_PLUTO_CHARON_SENSITIVITY",
                "GALACTIC_TIDE_IN_A_FROZEN_GALACTIC_FRAME_AND_POTENTIAL",
                "PREREGISTERED_STATISTICAL_STELLAR_ENCOUNTER_HISTORIES",
                "PHYSICAL_COLLISION_SURFACES_WITH_CONTINUOUS_EVENT_LOCATION",
                "OUTER_BARYCENTRIC_OORT_DOMAIN_ROUTE_AND_ESCAPE_OR_CENSORING_RULE",
                "PLANETARY_INITIAL_CONDITION_AND_PARAMETER_SENSITIVITY",
            ],
            "solar_mass_evolution_1pn_j2_dwarf_planet_kbo_ring_ablations_required": True,
            "current_4096_au_crossing_label": (
                "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION"
            ),
            "current_zero_terminal_source_radii_support_collision_claim": False,
            "external_perturber_model_implemented_and_qualified": False,
            "external_forcing_ensemble": {
                "statistical_stellar_history_count": statistical_stellar_history_count,
                "galactic_tide_only_control_count": tide_only_control_count,
                "total_forcing_context_count": forcing_context_count,
                "statistical_history_ids": "STELLAR_HISTORY_0000_THROUGH_1023",
                "control_id": "GALACTIC_TIDE_ONLY_NO_DISCRETE_STARS",
                "same_forcing_context_applied_to_all_twelve_objects": True,
                "identical_context_ids_paired_across_h64_and_h128": True,
                "history_seed_domain": (
                    "jx.tno.3gyr-full129-robustness.stellar-history.v1"
                ),
                "history_seed_preimage": (
                    "ASCII_DOMAIN_NUL_PLUS_FOUR_DIGIT_ASCII_HISTORY_INDEX"
                ),
                "generator_rng_catalog_probability_law_and_truncations_frozen": False,
                "catalog_stars_may_be_used_only_inside_their_validated_time_window": True,
                "catalog_stars_extrapolated_for_three_gyr": False,
                "synthetic_history_frequencies_are_model_sensitivity_not_real_probabilities": True,
                "execution_forbidden_until_generator_and_every_history_digest_are_bound": True,
            },
        },
        "retention_geometry": {
            "precision_maximal_200_year_state_retention_selected": True,
            "state_interval_julian_years": 200,
            "h64_state_interval_steps": 12_800,
            "h128_state_interval_steps": 25_600,
            "state_nodes_per_batch_endpoint_inclusive": retained_nodes_per_batch,
            "state_body_count_per_batch": 134,
            "phase_components_per_body": 6,
            "phase_component_bytes": 8,
            "lifecycle_lanes_per_batch": 129,
            "node_digest_bytes_per_batch": 32,
            "first_500k_interval_julian_years": 50,
            "h64_first_500k_interval_steps": 3_200,
            "h128_first_500k_interval_steps": 6_400,
            "h64_first_500k_final_step": 32_000_000,
            "h128_first_500k_final_step": 64_000_000,
            "first_500k_nodes_per_batch_endpoint_inclusive": 10_001,
            "ten_myr_windows": ten_myr_window_count,
            "nodes_per_ten_myr_window_endpoint_inclusive": 50_001,
            "every_integrator_step_source_nodes_fully_retained": False,
            "source_fragments_regenerated_and_bound_by_full_fragment_sha256": True,
            "source_fragment_start_and_end_state_receipts_required": True,
            "hierarchical_append_only_64_bit_manifest_offset_format_required": True,
        },
        "storage": {
            "forcing_context_count": forcing_context_count,
            "geometry_scope": "REDUCED5_FIXED_ARRAY_SUBTOTAL_ONLY",
            "per_context_per_resolution_all12_fixed_output_logical_bytes": (
                1_187_527_477_848
            ),
            "per_context_per_resolution_finalized_monolithic_stream_4k_lower_bound_bytes": (
                1_187_527_487_488
            ),
            "per_context_per_resolution_finalized_per_batch_stream_4k_sum_bytes": (
                1_187_527_606_272
            ),
            "per_context_per_resolution_fixed_array_breakdown_bytes": {
                "states_200_year": 1_157_760_077_184,
                "lifecycle_200_year": 23_220_001_548,
                "all12_node_sha256_200_year": 5_760_000_384,
                "first_500k_states_50_year": 771_917_184,
                "first_500k_lifecycle_50_year": 15_481_548,
            },
            "node_sha256_200_year_per_batch_bytes": 480_000_032,
            "per_context_dual_resolution_fixed_output_logical_bytes": (
                2_375_054_955_696
            ),
            "per_context_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes": (
                2_375_054_974_976
            ),
            "per_context_dual_resolution_finalized_per_batch_stream_4k_sum_bytes": (
                2_375_055_212_544
            ),
            "ensemble_dual_resolution_fixed_output_logical_bytes": (
                2_434_431_329_588_400
            ),
            "ensemble_dual_resolution_finalized_monolithic_stream_4k_lower_bound_bytes": (
                2_434_431_349_350_400
            ),
            "ensemble_dual_resolution_finalized_per_batch_stream_4k_sum_bytes": (
                2_434_431_592_857_600
            ),
            "per_context_canonical_continuous_source_stream_bytes_if_fully_retained": (
                138_240_000_000_480
            ),
            "ensemble_canonical_continuous_source_stream_bytes_if_fully_retained": (
                141_696_000_000_492_000
            ),
            "per_context_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints": (
                138_240_216_000_000
            ),
            "ensemble_canonical_logical_640k_tile_materialized_source_bytes_with_duplicate_endpoints": (
                141_696_221_400_000_000
            ),
            "per_context_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints": (
                138_240_216_000_720
            ),
            "ensemble_canonical_physical_fragment_materialized_source_bytes_with_duplicate_endpoints": (
                141_696_221_400_738_000
            ),
            "fixed_array_subtotals_exclude": [
                "ROUTE_AND_REMOVAL_RECORDS",
                "RESTART_PAYLOADS",
                "MANIFESTS_AND_RECEIPTS",
                "INTERMEDIATE_PHYSICAL_FRAGMENT_PAYLOADS_AND_PER_FRAGMENT_RECEIPTS_AND_FILESYSTEM_ALLOCATION",
                "REPLICATION",
                "TRANSIENT_STAGING",
                "EXPANDED_PHYSICAL_MODEL_BODIES",
            ],
            "current_writer_physical_fragment_inode_and_roster_format_qualified": False,
            "exact_external_storage_replication_route_transient_plan_required": True,
            "success_forbidden_if_space_or_inode_floor_is_not_met": True,
        },
        "mandatory_phase_order": [
            "PREPARE_COMMIT_AND_ANNOTATED_TAG_THIS_PREPARE_ONLY_PREREGISTRATION",
            "FREEZE_AND_QUALIFY_COMMON_EPOCH_AND_COMPLETE_3GYR_PHYSICAL_MODEL_LADDER",
            "FREEZE_1024_STELLAR_HISTORIES_PLUS_TIDE_ONLY_CONTROL_AND_BIND_EVERY_DIGEST",
            "IMPLEMENT_AND_QUALIFY_H64_H128_3GYR_ADAPTERS_WITH_INT64_STEPS",
            "IMPLEMENT_AND_QUALIFY_HIERARCHICAL_WRITER_RESTART_AND_EXTERNAL_STORAGE",
            "QUALIFY_CPU_SOURCE_REPLAY_GPU_TRANSFER_QUIET_MAP_AND_CPU_DOP853_ROUTE",
            "QUALIFY_ACTIVE_ROUTED_REMOVED_AND_CRASHED_FRESH_PROCESS_RESTART",
            "EXECUTE_ALL12_FULL129_H64_H128_1KYR_WITH_FROZEN_REFERENCE_LANES",
            "STOP_AND_REVIEW_ONLY_ENGINEERING_EVIDENCE_WITH_SCIENCE_OUTCOMES_SEALED",
            "CONTINUE_PREFIXES_THROUGH_5KYR_10KYR_100KYR_500KYR_5MYR_AND_10MYR",
            "QUALIFY_LONG_HORIZON_STATE_MACHINE_AND_WINDOWED_DIAGNOSTICS",
            "PASS_EXPANDED_FORCE_MODEL_AND_BOUNDARY_SENSITIVITY_GATES",
            "CONTINUE_WITHOUT_DROPPING_OR_TUNING_THROUGH_100MYR_500MYR_1GYR_2GYR_3GYR",
            "VERIFY_OUTPUT_RESTART_SOURCE_ROUTE_MODEL_RUNTIME_STORAGE_CROSSBINDINGS",
            "REPORT_PAIRED_H64_H128_SURVIVAL_EVENT_RESIDENCE_TRANSITION_SENSITIVITY",
            "PUBLISH_ONLY_CONDITIONAL_CHAOTIC_ENSEMBLE_MODEL_OUTPUT",
        ],
        "execution_blockers": [
            "COMMON_EPOCH_FULL_COVARIANCE_PROPAGATION_NOT_QUALIFIED",
            "3GYR_GALACTIC_TIDE_STELLAR_HISTORY_COLLISION_AND_OUTER_BOUNDARY_MODEL_NOT_IMPLEMENTED",
            "STELLAR_HISTORY_GENERATOR_RNG_DISTRIBUTIONS_TRUNCATIONS_AND_DIGESTS_NOT_FROZEN",
            "EXTERNAL_FORCING_WORK_AND_TORQUE_BALANCE_RESIDUAL_CONTRACT_NOT_FROZEN",
            "INDEPENDENT_ALL129_CPU_IAS15_REFERENCE_IMPLEMENTATION_AND_RUNTIME_NOT_PINNED",
            "REDUCED5_FULL129_H128_CPU_GPU_ROUTE_NOT_QUALIFIED",
            "RESOLVED11_H128_SOURCE_GATES_FAILED_AND_H1024_WAS_INCONCLUSIVE",
            "CANONICAL_SHARED_SOURCE_SERVICE_AND_FULL_FRAGMENT_BROADCAST_NOT_IMPLEMENTED_OR_QUALIFIED",
            "CURRENT_10MYR_ADAPTER_IS_H64_AND_EXACTLY_1000_SHARDS_PER_BATCH_ONLY",
            "CURRENT_WRITER_CANNOT_ADDRESS_11_070_036_900_SCHEDULED_PHYSICAL_BATCH_FRAGMENTS_OR_2_434_431_329_588_400_BYTE_FIXED_SUBTOTAL",
            "ROUTED_REMOVED_CRASHED_FRESH_PROCESS_RESTART_NOT_QUALIFIED",
            "LONG_HORIZON_STATE_MACHINE_AND_WINDOWED_CLASSIFICATION_NOT_QUALIFIED",
            "EXTERNAL_STORAGE_AND_DISTRIBUTED_RUNTIME_PLAN_NOT_FROZEN",
            "GPU_DRIVER_AND_COMPLETE_RUNTIME_IDENTITY_NOT_CURRENTLY_VERIFIED",
        ],
        "outcome_contract": {
            "first_10myr_diagnostic": (
                "JX_B22_OPERATIONALIZATION_V2_ONLY_AFTER_ITS_OWN_QUALIFICATION"
            ),
            "first_10myr_b22_nodes": 50_001,
            "single_b22_label_over_3gyr_forbidden": True,
            "candidate_windowing": (
                "300_NONOVERLAPPING_10MYR_WINDOWS_SHARING_ONLY_BOUNDARY_NODES"
            ),
            "windowing_requires_separate_qualification": True,
            "primary_3gyr_outputs": [
                "SURVIVED_TO_HORIZON_UNDER_FROZEN_MODEL",
                "FIRST_TERMINAL_OR_CENSORING_EVENT_TYPE_AND_INTEGER_STEP",
                "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION",
                "STATE_RESIDENCE_AND_TRANSITION_COUNTS",
                "ORBITAL_ELEMENT_ENVELOPES_AMONG_RETAINED_ACTIVE_STATES",
                "PAIRED_H64_H128_OBJECT_AND_CLONE_SENSITIVITY",
            ],
            "numerical_failure_is_never_a_physical_removal": True,
            "pointwise_prediction_after_chaotic_divergence": False,
            "clone_frequencies_are_covariance_sensitivity_not_probabilities": True,
            "no_cross_object_population_statistic": True,
        },
        "nonclaims": [
            "NO_EXACT_REAL_TNO_POSITION_CLASS_OR_EVENT_PREDICTION_AFTER_CHAOS",
            "NO_PHYSICAL_STABILITY_CLAIM_ONLY_CONDITIONAL_HORIZON_SURVIVAL",
            "NO_PHYSICAL_EJECTION_OR_COLLISION_FROM_CURRENT_BOUNDARIES",
            "NO_REAL_FUTURE_PROBABILITY_FROM_SYNTHETIC_STELLAR_HISTORY_FREQUENCIES",
            "NO_POPULATION_FREQUENCY_OR_SURVEY_SELECTION_INFERENCE",
            "NO_FORMATION_CAUSALITY_OR_ANGULAR_CLUSTERING_INFERENCE",
            "NO_PLANET_NINE_DETECTION_EXCLUSION_OR_RANKING",
            "NO_STABILITY_CLAIM_BEYOND_3GYR",
            "NO_PRODUCTION_NAVIGATION_OR_GENERAL_ENGINE_QUALIFICATION",
        ],
        "preregistration": {
            "expected_branch": EXPECTED_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "future_result_tag": FUTURE_RESULT_TAG,
            "prepared_default_root": DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            "prepared_roster": ["PROTOCOL.json"],
            "execution_successor_must_pin": [
                "ANNOTATED_PREREGISTRATION_TAG_OBJECT",
                "TAG_PEELED_COMMIT",
                "SOURCE_BYTES_AND_SHA256",
                "TEST_BYTES_AND_SHA256",
                "PROTOCOL_BYTES_CONTENT_SHA256_AND_FILE_SHA256",
            ],
            "no_automatic_git_network_gpu_or_execution_action": True,
        },
        "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        "public_cli": ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        "explicitly_absent_public_cli": [
            "execute",
            "execute-worker",
            "resume",
            "validate-result",
        ],
        "claims": _claims(),
        "content_sha256": "",
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise TNOThreeBillionError("protocol must be an exact object")
    expected = build_protocol()
    if (
        not _exact_equal(value, expected)
        or value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value)
        or not _exact_equal(value.get("claims"), _claims())
    ):
        raise TNOThreeBillionError("protocol bytes, semantics, or claim ceiling differ")
    return value["content_sha256"]


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise TNOThreeBillionError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TNOThreeBillionError("renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        source.encode("ascii"),
        directory_fd,
        destination.encode("ascii"),
        1,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise TNOThreeBillionError("protocol destination already exists")
        raise TNOThreeBillionError(f"renameat2 failed: {os.strerror(observed_errno)}")


def _validate_prepared_directory(
    root_fd: int,
    expected_payload: bytes,
) -> dict[str, Any]:
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise TNOThreeBillionError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
    ):
        raise TNOThreeBillionError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                raise TNOThreeBillionError("prepared protocol ended early")
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise TNOThreeBillionError("prepared protocol changed while held")
    finally:
        os.close(descriptor)
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise TNOThreeBillionError("prepared roster changed while held")
    payload = bytes(chunks)
    if payload != expected_payload:
        raise TNOThreeBillionError("prepared protocol bytes differ")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TNOThreeBillionError("prepared protocol is not canonical")
    validate_protocol(value)
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise TNOThreeBillionError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(root.parent, flags)
    created = False
    prepared: dict[str, Any] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise TNOThreeBillionError("prepared root already exists; create-only publication refused") from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            if os.listdir(root_fd):
                raise TNOThreeBillionError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            descriptor = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_directory(root_fd, payload)
            os.fsync(parent_fd)
        except Exception:
            for name in tuple(os.listdir(root_fd)):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    os.unlink(name, dir_fd=root_fd)
            os.fsync(root_fd)
            raise
        finally:
            os.close(root_fd)
    except Exception:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise TNOThreeBillionError("prepared protocol validation did not complete")
    return prepared


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    expected_payload = _canonical_json(build_protocol())
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared three-gigayear root") as root_fd:
        return _validate_prepared_directory(root_fd, expected_payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build", "audit-prerequisites", "prepare", "validate-prepared"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
    elif arguments.command == "audit-prerequisites":
        value = validate_bound_prerequisites()
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
    else:
        value = validate_prepared(arguments.output_root)
    sys.stdout.buffer.write(_canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
