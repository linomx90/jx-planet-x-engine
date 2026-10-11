#!/usr/bin/env python3
"""Prepare the full-128 TNO production geometry and bounded smoke contract.

This module does not execute dynamics.  It revalidates the exact retained
inputs, reconstructs all nominal-plus-128-clone initial states, and freezes a
single-batch 1 kyr GPU smoke acceptance contract.  A future smoke result can
be checked with :func:`validate_smoke_evidence_v1`; that check does not
authorize the 10 Myr production run or any scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_production_preflight_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_production_preflight_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_production_preflight_v1"

PROTOCOL_SCHEMA = "jx.tno.10my-full128-production-preflight.protocol.v1"
SMOKE_EVIDENCE_SCHEMA = "jx.tno.10my-full128-production-preflight.smoke-projection-receipt.v1"
PROTOCOL_DOMAIN = b"jx.tno.10my-full128-production-preflight.protocol.v1\0"
SMOKE_EVIDENCE_DOMAIN = b"jx.tno.10my-full128-production-preflight.smoke-projection-receipt.v1\0"
FILESYSTEM_TRUST_BOUNDARY = "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
SPK_IDS = (
    20090377, 20474640, 54526835, 20541132, 50666516, 20148209,
    20574372, 20303775, 20026181, 20042301, 20026375, 20015874,
)
NATIVE_EPOCHS_JD_TDB_HEX = (
    "0x1.2c195c0000000p+21", "0x1.2bd3dc0000000p+21",
    "0x1.2c4c140000000p+21", "0x1.2c0e240000000p+21",
    "0x1.2be0a40000000p+21", "0x1.2b62c40000000p+21",
    "0x1.2c15dc0000000p+21", "0x1.2c247c0000000p+21",
    "0x1.2c11640000000p+21", "0x1.2c16a40000000p+21",
    "0x1.2c0b4c0000000p+21", "0x1.2c0f940000000p+21",
)
RESOLVED11_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
REDUCED5_NAIF_IDS = (10, 5, 6, 7, 8)
REDUCED5_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER", "SATURN_BARYCENTER",
    "URANUS_BARYCENTER", "NEPTUNE_BARYCENTER",
)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
FULL_CLONE_INDICES = tuple(range(128))
HOLDOUT_TRACER_INDICES = (0, *(index + 1 for index in HOLDOUT_CLONE_INDICES))

AU_METRES_EXACT = 149_597_870_700
SECONDS_PER_JULIAN_YEAR_EXACT = 31_557_600
OBLIQUITY_RADIANS_HEX = "0x1.a2e9395ca7c40p-2"
OBLIQUITY_SIN_HEX = "0x1.9752e50f4b399p-2"
OBLIQUITY_COS_HEX = "0x1.d5c0357681ef3p-1"

BATCH_COUNT = 12
MASSIVE_COUNT = 5
TRACER_COUNT = 129
BODY_COUNT = MASSIVE_COUNT + TRACER_COUNT
ELEMENT_ENTITY_COUNT = 1 + TRACER_COUNT  # Neptune plus all tracers.
ELEMENT_FIELD_COUNT = 6
STEPS_PER_JULIAN_YEAR = 64
TEN_MYR_STEPS = 640_000_000
B22_INTERVAL_STEPS = 12_800
ML_SHORT_INTERVAL_STEPS = 3_200
ML_SHORT_FINAL_STEP = 32_000_000
SHARD_STEPS = 640_000
B22_NODE_COUNT = TEN_MYR_STEPS // B22_INTERVAL_STEPS + 1
ML_SHORT_NODE_COUNT = ML_SHORT_FINAL_STEP // ML_SHORT_INTERVAL_STEPS + 1
OVERLAP_NODE_COUNT = ML_SHORT_FINAL_STEP // B22_INTERVAL_STEPS + 1
UNION_NODE_COUNT = B22_NODE_COUNT + ML_SHORT_NODE_COUNT - OVERLAP_NODE_COUNT
ML_LONG_STRIDE = 5
ML_LONG_NODE_COUNT = (B22_NODE_COUNT - 1) // ML_LONG_STRIDE + 1
STATE_RECORD_BYTES = BODY_COUNT * 6 * 8
ELEMENT_RECORD_BYTES = ELEMENT_ENTITY_COUNT * ELEMENT_FIELD_COUNT * 8
OUTPUT_SIZES = {
    "B22_STATES.f64le": BATCH_COUNT * B22_NODE_COUNT * STATE_RECORD_BYTES,
    "B22_ELEMENTS.f64le": BATCH_COUNT * B22_NODE_COUNT * ELEMENT_RECORD_BYTES,
    "B22_NODE_SHA256.raw": BATCH_COUNT * B22_NODE_COUNT * 32,
    "ML_SHORT_STATES.f64le": BATCH_COUNT * ML_SHORT_NODE_COUNT * STATE_RECORD_BYTES,
}
TOTAL_RETAINED_BINARY_BYTES = sum(OUTPUT_SIZES.values())

SMOKE_BATCH_ORDER = 0
SMOKE_OBJECT_ID = OBJECT_IDS[SMOKE_BATCH_ORDER]
SMOKE_HORIZON_YEARS = 1_000
SMOKE_STEPS = SMOKE_HORIZON_YEARS * STEPS_PER_JULIAN_YEAR
SMOKE_CHECKPOINT_YEARS = tuple(range(0, SMOKE_HORIZON_YEARS + 1, 50))
SMOKE_PREFIX_BYTES = len(SMOKE_CHECKPOINT_YEARS) * STATE_RECORD_BYTES

V3_ARCHIVE_COMMIT = "d63e1cc9ddba00ab883776b0bce10fd00051a586"
V3_ARCHIVE_TAG_OBJECT = "1734a1f59168f5cca11a969b02a3eed809ff2675"
V3_PREREG_COMMIT = "9ec5151e5a979e46a516ec7efdf7c6e4b90da620"
V3_PREREG_TAG_OBJECT = "94297db889f5b5ba2f4d9eaef2adfa792a3740cd"

# Every scientific input is an exact current-byte dependency.  The V3 result
# root is deliberately represented by all eight files, not by RESULT alone.
BOUND_FILES = (
    ("pilot_v2_source", "benchmarks/jx_tno_10my_reduced5_classification_pilot_v2.py", 59_307, "33ad38cbc89a15a83cf180161544a721b8635affd0d4890a8a39d04389694332"),
    ("pilot_v2_test", "tests/test_tno_10my_reduced5_classification_pilot_v2.py", 28_773, "060793a45cd91590d484a610b9c6d48f6fae6c2a468c6e2663541599a831fbd8"),
    ("pilot_v2_protocol", "runs/tno_10my_reduced5_classification_pilot_v2/PROTOCOL.json", 34_347, "b510cbf7a62f45d8713ca8ef02dd2229f5a79966d00d2482843e814b939cded6"),
    ("grid_v1_source", "benchmarks/jx_tno_10my_reduced5_grid_adapter_v1.py", 111_493, "324498cd36726e461ab6910fdf0fe13a1b024346293558f535bea557cdb61b3a"),
    ("grid_v1_test", "tests/test_tno_10my_reduced5_grid_adapter_v1.py", 37_454, "909e6f63ff229e8909245001728acd58a8d4bb7df80993a14b578659c6dd96b8"),
    ("grid_v1_protocol", "runs/tno_10my_reduced5_grid_adapter_v1/PROTOCOL.json", 11_844, "6b10412f06f3c398646179fd163347ac269c03b08486c867e4da821e082be3f2"),
    ("classifier_v2_source", "benchmarks/jx_tno_b22_classifier_v2.py", 94_534, "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98"),
    ("stage0_source", "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", 70_064, "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367"),
    ("stage0_test", "tests/test_tno_native_epoch_cartesian_qualification_v1.py", 15_944, "1225cc19a78aa4ee412c11bf42e931ac8b6881c41845ad9b359c8ff547cb9d0c"),
    ("stage0_protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"),
    ("stage0_states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
    ("clone_source", "tools/jx_tno_covariance_clone_materialization_v1.py", 27_114, "49f00b856810225a495cbd16e34282db2199e2295d755c2bb2aaba4897b7ba53"),
    ("clone_test", "tests/test_tno_covariance_clone_materialization_v1.py", 9_607, "c2fdd4eddd85a9a5655e72bb8ad899bd6d3dc5db9c904b4259cee23b5cacbf0d"),
    ("clone_protocol", "runs/tno_covariance_clones_v1/PROTOCOL.json", 3_358, "88745a6f8286cf1c6f717033973fa972457f8e088eebbfe1d6047b1286a1f21b"),
    ("clone_artifact", "runs/tno_covariance_clones_v1/CLONES.json", 531_609, "395ce9914df0efe235e2144c24abd157f5aa3ba87beba87fa39891ec42312014"),
    ("bridge_source", "benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py", 81_161, "5cb0b5380e7eac3e1c3c5932715dcbe43372b03581e65ef981ba491e78087e7e"),
    ("bridge_test", "tests/test_tno_native_epoch_resolved11_bridge_v1.py", 13_008, "ad52731288e89e1425cc315bcbcdef705144a4bf22fa92f9a2c78576ae4a3497"),
    ("bridge_protocol", "runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", 11_386, "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459"),
    ("bridge_receipts", "runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", 5_810_062, "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5"),
    ("bridge_states", "runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", 168_684, "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"),
    ("de441_source", "tools/jx_tno_de441_resolved11_acquisition_v1.py", 37_145, "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00"),
    ("de441_test", "tests/test_tno_de441_resolved11_acquisition_v1.py", 14_556, "220215bae324d8119d03417f44997920b7a4304fb59eae19a375d94a8555d8b3"),
    ("de441_protocol", "runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", 11_986, "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b"),
    ("de441_custody", "runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", 141_501, "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"),
    ("v3_source", "benchmarks/jx_tno_1kyr_reduced5_lifecycle_gate_v3.py", 151_995, "02e72fb6a635b74dd31dc364e18cbaf03c5b9e1833caf5aba4a063d7198e47c7"),
    ("v3_test", "tests/test_tno_1kyr_reduced5_lifecycle_gate_v3.py", 42_017, "b4eb43b00e1fab0055a43cc4f26db2bef6bf29d51656bcfab1d792596b7845a8"),
    ("v3_protocol", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PROTOCOL.json", 9_178, "9c1e7f77b3a2fd9f6af5542aa2ee3fa7dc00d669d496c809da28b675c8519199"),
    ("v3_prefix", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin", 169_344, "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b"),
    ("v3_restart", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESTART_STREAMS.bin", 9_504, "3330f6d83b71f32c2a0fb14bc9f2e26ab13def77f1ea17527d69552473c80173"),
    ("v3_segments", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/SEGMENTS.jsonl", 239_895, "e9452debe803a7213d0878c6c2bd35199bfe966e7e6891f12d072113c825f64f"),
    ("v3_routes", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/ROUTES.jsonl", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("v3_preflight", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME_PREFLIGHT.json", 26_179, "7df113a70bcb1012ea05b2145f141bd8b788c504fc61c0f2b075671d0810ea80"),
    ("v3_runtime", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RUNTIME.json", 12_980, "43bc1edba432bed75f8a9eb1db9784c13116e1a93e20bb4b43cb634d6abd1dcd"),
    ("v3_result", "runs/tno_1kyr_reduced5_lifecycle_gate_v3/RESULT.json", 66_669, "3d85ae01fa826d0132fc0cbe6188159f2810bc9cf240a9daec7e2b3525054aa4"),
)

DE441_RAW_FILES = (
    ("raw/body_004.json", 7_194, "0669acdf63bb9b4cd9824a3fd46d90542f0c722102b2022dcf991f31be9a11bb"),
    ("raw/body_005.json", 7_194, "6e68920a0354cf5a521cb84c16235e4acb635a6a26ad5b225303de45372b4974"),
    ("raw/body_006.json", 7_194, "d912e2eb7d855e53dc2d6cf2dbcaae6f946d505f7f25036c0db88e9123924cc4"),
    ("raw/body_007.json", 7_194, "cc51de7b5da54b95bb1e26fbb88def2498d4b6a989cf55d5b89209fe818b271f"),
    ("raw/body_008.json", 7_194, "a8c45995d1a885c9b0d85ba5c5367d83369beee4f2d78a35fa1bb870fa94d32a"),
    ("raw/body_009.json", 7_194, "0b166885a491f49a5c1c0cc2c2da499bcea734d98662c0072366b22ac40150aa"),
    ("raw/body_010.json", 7_194, "913f48d219d635c6a5e262cc74239ea86bafb802d00732567461bf2b93755354"),
    ("raw/body_199.json", 7_194, "7d183ca3370a483dc739090f02915dc11ce4265a167260d90f9635bf856a03ef"),
    ("raw/body_299.json", 7_194, "4d721c4b6d88aa2ec2de4efdfb3161c1427ad00d789428e8590ea832ed2ede8c"),
    ("raw/body_301.json", 7_194, "3d919ab6ad860173f393023ea3670d904d9f226de4b994592fa5fbc6484912c3"),
    ("raw/body_399.json", 7_194, "0b8802edeaa692a18e747f670280fa4a275300a2342eb56eb093a56c5c32e97a"),
)

EXPECTED_CONTENT_SHA256 = {
    "pilot_v2_protocol": "eb6fe0b5ac6485f4175ebe44ab779d88ebf92986ef279d949e8eaea2b2a5c7d3",
    "grid_v1_protocol": "f13b551ea074ce6233c9f3b8aa5e0b75421aa26639141564312087af331812a9",
    "stage0_protocol": "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e",
    "stage0_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
    "clone_protocol": "939e6c3f8e3773f2af4675bab95369e702461ae00edf311b71c02f16785ab798",
    "clone_artifact": "9f1c0df03a2001f6249aafd423c4c9feb8bde9a75467416d35b81b88bb2d59d8",
    "bridge_protocol": "a7dbc361ffa4ccaf138f3c7afce269cad8bbb505b77930bcea9d04f5167b3c15",
    "bridge_receipts": "7e5f3ef0a731647a9adaebedae65d6f703e44b8a923eb624d6dc872d935b8058",
    "bridge_states": "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13",
    "de441_protocol": "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e",
    "de441_custody": "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3",
    "v3_protocol": "538e076274ab5e1c850cce069405ba67c3cab41226fa684a686aaaa81cdc32d9",
    "v3_preflight": "80743cbaeba28be1b131a9470a779df95131752b14a02cb364f59afc4b8a95c2",
    "v3_runtime": "fbbc50a05f3faa87df98052e42bf51e5c9bd9f0feb11bf5acd2516ef857548b5",
    "v3_result": "d1dffed406f02774f4af874489db4a8209570ebeadffdc6a2e5cc2ea43b39264",
}


class ProductionPreflightError(RuntimeError):
    """The prepare-only production preflight failed closed."""


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise ProductionPreflightError("SHA-256 payload must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ProductionPreflightError("value is not canonical-JSON encodable") from exc


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ProductionPreflightError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProductionPreflightError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionPreflightError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ProductionPreflightError("JSON root must be an exact object")
    return value


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev, left.st_ino, left.st_mode, left.st_size,
        left.st_mtime_ns, left.st_ctime_ns, left.st_nlink,
    ) == (
        right.st_dev, right.st_ino, right.st_mode, right.st_size,
        right.st_mtime_ns, right.st_ctime_ns, right.st_nlink,
    )


def _read_regular(path: Path, label: str, expected_bytes: int | None = None) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        lexical = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProductionPreflightError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1
        or (expected_bytes is not None and lexical.st_size != expected_bytes)
    ):
        raise ProductionPreflightError(f"{label} is not the expected single-link regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not _same_stat(lexical, before):
            raise ProductionPreflightError(f"{label} changed before open")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = absolute.stat(follow_symlinks=False)
    payload = b"".join(chunks)
    if remaining or not _same_stat(before, after) or not _same_stat(before, final):
        raise ProductionPreflightError(f"{label} changed during read")
    return payload


def _load_canonical_bound(role: str, records: Mapping[str, bytes]) -> dict[str, Any]:
    payload = records[role]
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise ProductionPreflightError(f"{role} is not canonical JSON")
    expected = EXPECTED_CONTENT_SHA256.get(role)
    if expected is not None and value.get("content_sha256") != expected:
        raise ProductionPreflightError(f"{role} content identity differs")
    return value


def _require_directory_roster(path: Path, expected: set[str], label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    lexical = absolute.stat(follow_symlinks=False)
    if not stat.S_ISDIR(lexical.st_mode):
        raise ProductionPreflightError(f"{label} is not a lexical directory")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino):
            raise ProductionPreflightError(f"{label} changed before open")
        if set(os.listdir(descriptor)) != expected:
            raise ProductionPreflightError(f"{label} roster differs")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or (before.st_dev, before.st_ino) != (final.st_dev, final.st_ino)
    ):
        raise ProductionPreflightError(f"{label} changed during inspection")


def _bound_inputs() -> tuple[list[dict[str, object]], dict[str, bytes]]:
    _require_directory_roster(
        PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_lifecycle_gate_v3",
        {
            "PROTOCOL.json", "PREFIX_STATES.bin", "RESTART_STREAMS.bin",
            "SEGMENTS.jsonl", "ROUTES.jsonl", "RUNTIME_PREFLIGHT.json",
            "RUNTIME.json", "RESULT.json",
        },
        "V3 retained result root",
    )
    _require_directory_roster(
        PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1",
        {"PROTOCOL.json", "CUSTODY.json", "raw"},
        "DE441 retained root",
    )
    _require_directory_roster(
        PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1" / "raw",
        {Path(relative).name for relative, _size, _sha in DE441_RAW_FILES},
        "DE441 raw response root",
    )
    records: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for role, relative, expected_bytes, expected_sha in BOUND_FILES:
        payload = _read_regular(PROJECT_ROOT / relative, role, expected_bytes)
        if _sha256(payload) != expected_sha:
            raise ProductionPreflightError(f"{role} SHA-256 differs")
        records.append({
            "role": role, "relative_path": relative,
            "bytes": expected_bytes, "sha256": expected_sha,
        })
        payloads[role] = payload
    raw_root = PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1"
    for relative, expected_bytes, expected_sha in DE441_RAW_FILES:
        role = "de441_" + Path(relative).stem
        payload = _read_regular(raw_root / relative, role, expected_bytes)
        if _sha256(payload) != expected_sha:
            raise ProductionPreflightError(f"{role} SHA-256 differs")
        records.append({
            "role": role,
            "relative_path": f"runs/tno_de441_resolved11_native_epochs_v1/{relative}",
            "bytes": expected_bytes, "sha256": expected_sha,
        })
    return records, payloads


def _exact_float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise ProductionPreflightError(f"{label} is not an exact hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ProductionPreflightError(f"{label} is not binary64 hex") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ProductionPreflightError(f"{label} is not canonical finite binary64 hex")
    return result


def _vector_hex(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise ProductionPreflightError(f"{label} is not an exact three-vector")
    return tuple(_exact_float_hex(item, f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[k]) + float(right[k]) for k in range(3))  # type: ignore[return-value]


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[k]) - float(right[k]) for k in range(3))  # type: ignore[return-value]


def _weighted_center(
    gms: Sequence[float], positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if type(gms) not in (tuple, list) or not gms or len(positions) != len(gms) or len(velocities) != len(gms):
        raise ProductionPreflightError("weighted-center roster differs")
    if any(type(gm) is not float or not math.isfinite(gm) or gm <= 0.0 for gm in gms):
        raise ProductionPreflightError("weighted-center GM differs")
    total = math.fsum(gms)
    position = tuple(
        math.fsum(gms[i] * float(positions[i][axis]) for i in range(len(gms))) / total
        for axis in range(3)
    )
    velocity = tuple(
        math.fsum(gms[i] * float(velocities[i][axis]) for i in range(len(gms))) / total
        for axis in range(3)
    )
    return position, velocity  # type: ignore[return-value]


def _rotate_ecliptic_to_j2000(vector: Sequence[float]) -> tuple[float, float, float]:
    if type(vector) not in (tuple, list) or len(vector) != 3:
        raise ProductionPreflightError("rotation input is not an exact three-vector")
    x, y, z = vector
    if any(type(item) is not float or not math.isfinite(item) for item in (x, y, z)):
        raise ProductionPreflightError("rotation input is not finite built-in binary64")
    sine = float.fromhex(OBLIQUITY_SIN_HEX)
    cosine = float.fromhex(OBLIQUITY_COS_HEX)
    rotated_y = float(Fraction.from_float(cosine) * Fraction.from_float(y) - Fraction.from_float(sine) * Fraction.from_float(z))
    rotated_z = float(Fraction.from_float(sine) * Fraction.from_float(y) + Fraction.from_float(cosine) * Fraction.from_float(z))
    return x, rotated_y, rotated_z


def _pack_rows(rows: Sequence[Sequence[float]]) -> bytes:
    if type(rows) not in (tuple, list) or len(rows) != BODY_COUNT:
        raise ProductionPreflightError("full-clone state row count differs")
    values: list[float] = []
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 6:
            raise ProductionPreflightError("full-clone state row shape differs")
        for value in row:
            if type(value) is not float or not math.isfinite(value):
                raise ProductionPreflightError("full-clone state is not finite built-in binary64")
            values.append(value)
    result = struct.pack(f"<{len(values)}d", *values)
    if len(result) != STATE_RECORD_BYTES:
        raise ProductionPreflightError("full-clone state payload length differs")
    return result


def _unpack_rows(payload: bytes) -> tuple[tuple[float, ...], ...]:
    if type(payload) is not bytes or len(payload) != STATE_RECORD_BYTES:
        raise ProductionPreflightError("full-clone checkpoint payload length differs")
    values = struct.unpack(f"<{BODY_COUNT * 6}d", payload)
    if any(not math.isfinite(value) for value in values):
        raise ProductionPreflightError("full-clone checkpoint contains a non-finite value")
    return tuple(tuple(values[index * 6:(index + 1) * 6]) for index in range(BODY_COUNT))


def _v3_checkpoint(prefix: bytes, batch_order: int, checkpoint_order: int) -> bytes:
    record_bytes = 14 * 6 * 8
    offset = (batch_order * len(SMOKE_CHECKPOINT_YEARS) + checkpoint_order) * record_bytes
    result = prefix[offset:offset + record_bytes]
    if len(result) != record_bytes:
        raise ProductionPreflightError("V3 prefix locator exceeds retained bytes")
    return result


def _selected_holdout_payload(full_payload: bytes) -> bytes:
    rows = _unpack_rows(full_payload)
    selected = (*rows[:MASSIVE_COUNT], *(rows[MASSIVE_COUNT + index] for index in HOLDOUT_TRACER_INDICES))
    values = tuple(value for row in selected for value in row)
    return struct.pack("<84d", *values)


def _validate_semantic_inputs(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    pilot = _load_canonical_bound("pilot_v2_protocol", payloads)
    grid = _load_canonical_bound("grid_v1_protocol", payloads)
    stage0_protocol = _load_canonical_bound("stage0_protocol", payloads)
    stage0 = _load_canonical_bound("stage0_states", payloads)
    clone_protocol = _load_canonical_bound("clone_protocol", payloads)
    clones = _load_canonical_bound("clone_artifact", payloads)
    bridge_protocol = _load_canonical_bound("bridge_protocol", payloads)
    _load_canonical_bound("bridge_receipts", payloads)
    bridge = _load_canonical_bound("bridge_states", payloads)
    de441_protocol = _load_canonical_bound("de441_protocol", payloads)
    custody = _load_canonical_bound("de441_custody", payloads)
    v3_protocol = _load_canonical_bound("v3_protocol", payloads)
    _load_canonical_bound("v3_preflight", payloads)
    _load_canonical_bound("v3_runtime", payloads)
    result = _load_canonical_bound("v3_result", payloads)
    prefix = payloads["v3_prefix"]
    if (
        pilot.get("schema") != "jx.tno.10my-reduced5-classification-pilot.protocol.v2"
        or pilot.get("claims", {}).get("ten_myr_execution_authorized") is not False
        or grid.get("schema") != "jx.tno.10my-reduced5-grid-adapter.protocol.v1"
        or grid.get("claims", {}).get("production_scale_writer_qualified") is not False
        or stage0_protocol.get("schema") != "jx.tno.native-epoch-cartesian.protocol.v1"
        or stage0.get("schema") != "jx.tno.native-epoch-cartesian.states.v1"
        or stage0.get("total_state_count") != 1548
        or clone_protocol.get("schema") != "jx.tno.covariance-clones.protocol.v1"
        or clones.get("schema") != "jx.tno.covariance-clones.artifact.v1"
        or clones.get("total_clone_count") != 1536
        or bridge_protocol.get("schema") != "jx.tno.native-epoch-resolved11-bridge.protocol.v1"
        or bridge.get("schema") != "jx.tno.native-epoch-resolved11-bridge.states.v1"
        or bridge.get("batch_count") != 12
        or bridge.get("source_stage0_state_count") != 1548
        or de441_protocol.get("schema") != "jx.tno.de441-resolved11-native-epochs.protocol.v1"
        or custody.get("schema") != "jx.tno.de441-resolved11-native-epochs.custody.v1"
        or custody.get("state_record_count") != 132
        or v3_protocol.get("schema") != "jx.tno.1kyr-reduced5-lifecycle-gate.protocol.v3"
        or result.get("schema") != "jx.tno.1kyr-reduced5-lifecycle-gate.result.v3"
        or result.get("status") != "ONE_THOUSAND_YEAR_LIFECYCLE_PREFIX_PASS_TEN_MYR_NOT_AUTHORIZED"
        or result.get("all_predeclared_gates_passed") is not True
        or result.get("batch_count") != 12
        or result.get("route_record_count") != 0
        or result.get("removal_record_count") != 0
        or result.get("claims", {}).get("ten_myr_execution_authorized") is not False
        or result.get("git_preregistration", {}).get("commit") != V3_PREREG_COMMIT
        or result.get("git_preregistration", {}).get("tag_object") != V3_PREREG_TAG_OBJECT
        or len(prefix) != 12 * 21 * 672
    ):
        raise ProductionPreflightError("bound scientific input semantics differ")
    return stage0, bridge, custody, prefix


def _initial_state_records(
    stage0: Mapping[str, Any], bridge: Mapping[str, Any],
    custody: Mapping[str, Any], v3_prefix: bytes,
) -> tuple[list[dict[str, object]], str, str, tuple[bytes, ...], tuple[bytes, ...]]:
    states = stage0.get("states")
    batches = bridge.get("batches")
    de441_rows = custody.get("state_records")
    if type(states) is not list or len(states) != 1548 or type(batches) is not list or len(batches) != 12 or type(de441_rows) is not list or len(de441_rows) != 132:
        raise ProductionPreflightError("initial-state source roster differs")
    primary_payloads: list[bytes] = []
    checkpoint_payloads: list[bytes] = []
    summaries: list[dict[str, object]] = []
    for batch_order, object_id in enumerate(OBJECT_IDS):
        group = [row for row in states if type(row) is dict and row.get("object_id") == object_id]
        if len(group) != 129:
            raise ProductionPreflightError("Stage0 per-object state count differs")
        expected_indices = [None, *FULL_CLONE_INDICES]
        observed_indices = [row.get("clone_index") for row in group]
        if any(
            (expected is None and observed is not None)
            or (expected is not None and (type(observed) is not int or observed != expected))
            for observed, expected in zip(observed_indices, expected_indices, strict=True)
        ):
            raise ProductionPreflightError("Stage0 full-clone order differs")
        if any(
            row.get("sample_kind") != ("NOMINAL" if index == 0 else "COVARIANCE_CLONE")
            or row.get("spk_id") != SPK_IDS[batch_order]
            or row.get("covariance_epoch_jd_tdb_hex") != NATIVE_EPOCHS_JD_TDB_HEX[batch_order]
            for index, row in enumerate(group)
        ):
            raise ProductionPreflightError("Stage0 full-clone identity differs")
        bridge_batch = batches[batch_order]
        if type(bridge_batch) is not dict or bridge_batch.get("object_id") != object_id:
            raise ProductionPreflightError("bridge batch identity differs")
        massive_rows = bridge_batch.get("massive_bodies")
        holdouts = bridge_batch.get("tnos")
        if type(massive_rows) is not list or len(massive_rows) != 11 or type(holdouts) is not list or len(holdouts) != 9:
            raise ProductionPreflightError("bridge batch roster differs")
        gms = tuple(
            _exact_float_hex(row.get("gravitational_parameter_m3_per_s2_hex"), "resolved11 GM")
            * (SECONDS_PER_JULIAN_YEAR_EXACT ** 2) / (AU_METRES_EXACT ** 3)
            for row in massive_rows
        )
        rows = de441_rows[batch_order * 11:(batch_order + 1) * 11]
        if any(
            type(row.get("naif_id")) is not int or row.get("naif_id") != expected
            for row, expected in zip(rows, RESOLVED11_NAIF_IDS, strict=True)
        ) or any(
            type(row.get("epoch_order")) is not int or row.get("epoch_order") != batch_order
            or row.get("object_id") != object_id
            or row.get("spk_id") != SPK_IDS[batch_order]
            or row.get("ephemeris_source") != "DE441"
            or row.get("frame") != "ICRF"
            or type(row.get("center_naif_id")) is not int or row.get("center_naif_id") != 0
            for row in rows
        ):
            raise ProductionPreflightError("DE441 epoch/body roster differs")
        ssb_positions = tuple(_vector_hex(row.get("position_AU_binary64_hex"), "DE441 position") for row in rows)
        ssb_velocities = tuple(
            tuple(value * 365.25 for value in _vector_hex(row.get("velocity_AU_per_day_binary64_hex"), "DE441 velocity"))
            for row in rows
        )
        resolved_center_position, resolved_center_velocity = _weighted_center(gms, ssb_positions, ssb_velocities)
        resolved_positions = tuple(_sub(row, resolved_center_position) for row in ssb_positions)
        resolved_velocities = tuple(_sub(row, resolved_center_velocity) for row in ssb_velocities)

        heliocentric_positions: list[tuple[float, float, float]] = []
        heliocentric_velocities: list[tuple[float, float, float]] = []
        for row in group:
            position = _rotate_ecliptic_to_j2000(_vector_hex(row.get("position_AU_hex"), "Stage0 position"))
            velocity = _rotate_ecliptic_to_j2000(_vector_hex(row.get("velocity_AU_per_julian_year_hex"), "Stage0 velocity"))
            heliocentric_positions.append(position)
            heliocentric_velocities.append(velocity)
        for holdout_order, tracer_index in enumerate(HOLDOUT_TRACER_INDICES):
            retained = holdouts[holdout_order]
            if (
                retained.get("sample_kind") != ("NOMINAL" if tracer_index == 0 else "COVARIANCE_CLONE")
                or retained.get("clone_index") != expected_indices[tracer_index]
                or [item.hex() for item in heliocentric_positions[tracer_index]] != retained.get("heliocentric_j2000_position_AU_hex")
                or [item.hex() for item in heliocentric_velocities[tracer_index]] != retained.get("heliocentric_j2000_velocity_AU_per_julian_year_hex")
            ):
                raise ProductionPreflightError("Stage0-to-bridge holdout rotation differs bitwise")

        tracer_resolved_positions = tuple(
            _sub(_add(position, ssb_positions[0]), resolved_center_position)
            for position in heliocentric_positions
        )
        tracer_resolved_velocities = tuple(
            _sub(_add(velocity, ssb_velocities[0]), resolved_center_velocity)
            for velocity in heliocentric_velocities
        )
        reduced_gms = (math.fsum(gms[:6]), *gms[6:10])
        active_positions = (resolved_positions[0], *resolved_positions[6:10])
        active_velocities = (resolved_velocities[0], *resolved_velocities[6:10])
        reduced_center_position, reduced_center_velocity = _weighted_center(
            reduced_gms, active_positions, active_velocities
        )
        centered_massive_positions = tuple(_sub(row, reduced_center_position) for row in active_positions)
        centered_massive_velocities = tuple(_sub(row, reduced_center_velocity) for row in active_velocities)
        centered_tracer_positions = tuple(_sub(row, reduced_center_position) for row in tracer_resolved_positions)
        centered_tracer_velocities = tuple(_sub(row, reduced_center_velocity) for row in tracer_resolved_velocities)
        central_position = centered_massive_positions[0]
        central_velocity = centered_massive_velocities[0]
        primary_massive_positions = tuple(_sub(row, central_position) for row in centered_massive_positions)
        primary_massive_velocities = tuple(_sub(row, central_velocity) for row in centered_massive_velocities)
        primary_massive_positions = ((0.0, 0.0, 0.0), *primary_massive_positions[1:])
        primary_massive_velocities = ((0.0, 0.0, 0.0), *primary_massive_velocities[1:])
        primary_tracer_positions = tuple(_sub(row, central_position) for row in centered_tracer_positions)
        primary_tracer_velocities = tuple(_sub(row, central_velocity) for row in centered_tracer_velocities)
        primary_rows = tuple(
            (*primary_massive_positions[i], *primary_massive_velocities[i]) for i in range(5)
        ) + tuple(
            (*primary_tracer_positions[i], *primary_tracer_velocities[i]) for i in range(129)
        )
        primary_payload = _pack_rows(primary_rows)

        checkpoint_center_position, checkpoint_center_velocity = _weighted_center(
            reduced_gms, primary_massive_positions, primary_massive_velocities
        )
        checkpoint_rows = tuple(
            (*_sub(primary_massive_positions[i], checkpoint_center_position),
             *_sub(primary_massive_velocities[i], checkpoint_center_velocity))
            for i in range(5)
        ) + tuple(
            (*_sub(primary_tracer_positions[i], checkpoint_center_position),
             *_sub(primary_tracer_velocities[i], checkpoint_center_velocity))
            for i in range(129)
        )
        checkpoint_payload = _pack_rows(checkpoint_rows)
        expected_v3 = _v3_checkpoint(v3_prefix, batch_order, 0)
        selected = _selected_holdout_payload(checkpoint_payload)
        if selected != expected_v3:
            raise ProductionPreflightError("full-clone year-zero holdout projection differs from V3")
        primary_payloads.append(primary_payload)
        checkpoint_payloads.append(checkpoint_payload)
        summaries.append({
            "batch_order": batch_order,
            "object_id": object_id,
            "spk_id": SPK_IDS[batch_order],
            "native_epoch_jd_tdb_hex": NATIVE_EPOCHS_JD_TDB_HEX[batch_order],
            "tracer_count": 129,
            "primary_relative_payload_bytes": len(primary_payload),
            "primary_relative_payload_sha256": _sha256(primary_payload),
            "barycentric_checkpoint_payload_bytes": len(checkpoint_payload),
            "barycentric_checkpoint_payload_sha256": _sha256(checkpoint_payload),
            "bridge_holdout_rotation_crosscheck_count": 9,
            "v3_year_zero_bitwise_crosscheck_count": 14,
            "v3_year_zero_selected_payload_sha256": _sha256(selected),
        })
    return (
        summaries,
        _sha256(b"".join(primary_payloads)),
        _sha256(b"".join(checkpoint_payloads)),
        tuple(primary_payloads),
        tuple(checkpoint_payloads),
    )


def _implementation_record(path: Path, label: str) -> dict[str, object]:
    payload = _read_regular(path, label)
    if not payload:
        raise ProductionPreflightError(f"{label} is empty")
    return {
        "label": label, "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload), "sha256": _sha256(payload),
    }


def build_protocol() -> dict[str, Any]:
    """Rebuild the complete prepare-only protocol from exact retained inputs."""

    bound_records, payloads = _bound_inputs()
    stage0, bridge, custody, prefix = _validate_semantic_inputs(payloads)
    batches, primary_sha, checkpoint_sha, _primary_payloads, _checkpoint_payloads = _initial_state_records(
        stage0, bridge, custody, prefix
    )
    source = _implementation_record(SCRIPT_PATH, "full128_preflight_source")
    test = _implementation_record(TEST_PATH, "full128_preflight_test")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARE_ONLY_FULL128_GEOMETRY_AND_SMOKE_PROJECTION_RECEIPT_FROZEN",
        "objective": "QUALIFY_INPUT_GEOMETRY_AND_FREEZE_A_NONQUALIFYING_ONE_BATCH_1KYR_PROJECTION_RECEIPT_BEFORE_AUTHORITATIVE_SMOKE_VALIDATION",
        "implementation": {
            **source,
            "test": test,
            "source_identity_scope": "LIVE_SINGLE_LINK_FILE_POINT_OBSERVATION_NOT_EXECUTED_BYTE_PROOF",
            "runtime_trust_boundary": "NO_HOSTILE_CODE_IN_CALLING_PROCESS",
            "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        },
        "lineage": {
            "bound_inputs": bound_records,
            "v3_result_root_exact_file_count": 8,
            "v3_result_archive_commit": V3_ARCHIVE_COMMIT,
            "v3_result_archive_tag_object": V3_ARCHIVE_TAG_OBJECT,
            "v3_preregistration_commit": V3_PREREG_COMMIT,
            "v3_preregistration_tag_object": V3_PREREG_TAG_OBJECT,
            "v3_archive_git_identity_is_external_custody_not_rederived_by_this_protocol": True,
        },
        "full_clone_initial_states": {
            "batch_count": BATCH_COUNT,
            "objects": list(OBJECT_IDS),
            "sample_order": [
                {"sample_kind": "NOMINAL", "clone_index": None},
                *({"sample_kind": "COVARIANCE_CLONE", "clone_index": index} for index in FULL_CLONE_INDICES),
            ],
            "tracers_per_batch": TRACER_COUNT,
            "total_tracers": BATCH_COUNT * TRACER_COUNT,
            "massive_count": MASSIVE_COUNT,
            "reduced_massive_body_ids": list(REDUCED5_BODY_IDS),
            "reduced_massive_naif_ids": list(REDUCED5_NAIF_IDS),
            "source_frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
            "engine_frame": "ICRF_J2000_EQUATORIAL_PRIMARY_RELATIVE",
            "checkpoint_frame": "ICRF_J2000_EQUATORIAL_INSTANTANEOUS_REDUCED5_GM_BARYCENTER",
            "time_scale": "TDB",
            "units": "AU_AND_AU_PER_JULIAN_YEAR",
            "rotation": {
                "obliquity_radians_hex": OBLIQUITY_RADIANS_HEX,
                "sine_hex": OBLIQUITY_SIN_HEX,
                "cosine_hex": OBLIQUITY_COS_HEX,
                "operation": "X_IDENTITY_Y_CY_MINUS_SZ_Z_SY_PLUS_CZ_EXACT_DYADS_THEN_ONE_RN_EVEN",
            },
            "translation_and_collapse_operation_order": [
                "DE441_SSB_RESOLVED11_GM_CENTER_USING_MATH_FSUM_BINARY64_PRODUCTS",
                "ADD_DE441_SUN_SSB_TO_ROTATED_STAGE0_HELIOCENTRIC_TNO",
                "SUBTRACT_RESOLVED11_CENTER",
                "COLLAPSE_INNER_SIX_GMS_AT_SUN_STATE_AND_RETAIN_JUPITER_THROUGH_NEPTUNE",
                "SUBTRACT_REDUCED5_GM_CENTER",
                "SUBTRACT_REDUCED5_CENTRAL_STATE_FOR_PRIMARY_RELATIVE_ENGINE_INPUT",
                "REAPPLY_STAGE1_CHECKPOINT_GM_CENTERING_FOR_RETAINED_OUTPUT",
            ],
            "bridge_holdout_rotation_bitwise_crosschecks": BATCH_COUNT * 9,
            "v3_year_zero_row_bitwise_crosschecks": BATCH_COUNT * 14,
            "primary_relative_aggregate_sha256": primary_sha,
            "barycentric_checkpoint_aggregate_sha256": checkpoint_sha,
            "batches": batches,
        },
        "production_grid": {
            "steps_per_julian_year": STEPS_PER_JULIAN_YEAR,
            "step_julian_years_hex": (1.0 / STEPS_PER_JULIAN_YEAR).hex(),
            "horizon_julian_years": 10_000_000,
            "total_steps": TEN_MYR_STEPS,
            "shard_steps": SHARD_STEPS,
            "shard_count_per_batch": TEN_MYR_STEPS // SHARD_STEPS,
            "b22_cadence_years": 200,
            "b22_interval_steps": B22_INTERVAL_STEPS,
            "b22_node_count": B22_NODE_COUNT,
            "ml_short_cadence_years": 50,
            "ml_short_final_year": 500_000,
            "ml_short_interval_steps": ML_SHORT_INTERVAL_STEPS,
            "ml_short_node_count": ML_SHORT_NODE_COUNT,
            "ml_long_cadence_years": 1_000,
            "ml_long_is_stride_5_view_of_b22": True,
            "ml_long_node_count": ML_LONG_NODE_COUNT,
            "union_node_count": UNION_NODE_COUNT,
            "state_body_count": BODY_COUNT,
            "state_record_bytes": STATE_RECORD_BYTES,
            "element_entity_count": ELEMENT_ENTITY_COUNT,
            "element_entities": "NEPTUNE_PLUS_NOMINAL_AND_128_CLONES",
            "element_field_count": ELEMENT_FIELD_COUNT,
            "element_record_bytes": ELEMENT_RECORD_BYTES,
            "output_sizes": dict(OUTPUT_SIZES),
            "total_retained_binary_bytes": TOTAL_RETAINED_BINARY_BYTES,
            "formula": "SUM(12*50001*134*6*8,12*50001*130*6*8,12*50001*32,12*10001*134*6*8)",
        },
        "bounded_gpu_smoke": {
            "execution_authorized_by_this_protocol": False,
            "batch_selection_and_acceptance_preregistered_in_protocol": True,
            "future_external_preregistration_required": True,
            "batch_order": SMOKE_BATCH_ORDER,
            "object_id": SMOKE_OBJECT_ID,
            "ephemeris": "DE441",
            "steps_per_julian_year": STEPS_PER_JULIAN_YEAR,
            "step_julian_years_hex": (1.0 / STEPS_PER_JULIAN_YEAR).hex(),
            "horizon_julian_years": SMOKE_HORIZON_YEARS,
            "step_count": SMOKE_STEPS,
            "checkpoint_years": list(SMOKE_CHECKPOINT_YEARS),
            "checkpoint_count": len(SMOKE_CHECKPOINT_YEARS),
            "checkpoint_record_bytes": STATE_RECORD_BYTES,
            "prefix_bytes": SMOKE_PREFIX_BYTES,
            "acceptance": {
                "all_129_tracers_active_at_every_checkpoint": True,
                "all_state_components_finite": True,
                "route_record_count": 0,
                "removal_record_count": 0,
                "v3_massive_plus_nine_holdout_projection_bitwise_equal_at_all_21_checkpoints": True,
                "elapsed_time_is_diagnostic_only": True,
                "no_performance_threshold": True,
            },
            "local_helper_scope": "PROJECTION_ONLY_RECEIPT_NOT_GPU_SOURCE_RUNTIME_LIFECYCLE_OR_EXECUTION_CUSTODY",
            "local_helper_cannot_satisfy_smoke_blocker": True,
        },
        "publication": {
            "prepared_roster": ["PROTOCOL.json"],
            "create_only": True,
            "canonical_json": True,
            "no_automatic_gpu_or_dynamics_execution": True,
            "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
            "partial_root_on_failure": "INVALID_REQUIRES_MANUAL_INSPECTION_AND_NEW_ROOT",
        },
        "blockers": [
            "FULL129_ONE_BATCH_1KYR_GPU_SMOKE_NOT_EXECUTED",
            "FULL129_GPU_RUNTIME_IDENTITY_NOT_OBSERVED",
            "AUTHORITATIVE_FULL129_GPU_SMOKE_ARTIFACT_REPLAY_IS_DELEGATED_TO_THE_EXECUTION_DOSSIER",
            "CURRENT_GRID_ADAPTER_IS_FIXED_TO_NINE_TRACERS_AND_CANNOT_BE_REUSED_UNCHANGED",
            "FULL128_WRITER_FINALIZER_AND_RESTART_LIFECYCLE_NOT_IMPLEMENTED_OR_QUALIFIED",
            "REQUIRED_8_394_469_632_OUTPUT_BYTES_PLUS_STAGING_FREE_SPACE_NOT_ATTESTED",
            "FUTURE_10MY_EXECUTION_NOT_EXTERNALLY_PREREGISTERED",
            "TEN_MYR_DYNAMICS_AND_CLASSIFIER_RUNTIME_NOT_EXECUTED",
        ],
        "claims": {
            "prepare_only_protocol_frozen": True,
            "full_clone_initial_state_geometry_constructed": True,
            "full_clone_initial_states_dynamically_qualified": False,
            "bounded_gpu_smoke_contract_frozen": False,
            "bounded_gpu_smoke_projection_receipt_contract_frozen": True,
            "bounded_gpu_smoke_executed": False,
            "gpu_executed": False,
            "ten_myr_execution_authorized": False,
            "ten_myr_execution_completed": False,
            "classification_generated": False,
            "robust_128_clone_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise ProductionPreflightError("protocol must be an exact object")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise ProductionPreflightError("protocol differs from exact reconstruction")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise ProductionPreflightError("protocol content hash differs")
    return str(value["content_sha256"])


def validate_smoke_evidence_v1(
    prefix_states: bytes,
    *,
    active_counts: Sequence[int],
    routes_jsonl: bytes,
    removals_jsonl: bytes,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Validate only a smoke projection receipt; never certify GPU execution."""

    protocol = build_protocol()
    if type(prefix_states) is not bytes or len(prefix_states) != SMOKE_PREFIX_BYTES:
        raise ProductionPreflightError("smoke prefix byte length differs")
    if (
        type(active_counts) not in (tuple, list)
        or len(active_counts) != len(SMOKE_CHECKPOINT_YEARS)
        or any(type(value) is not int or value != 129 for value in active_counts)
    ):
        raise ProductionPreflightError("smoke active-count series differs")
    if type(routes_jsonl) is not bytes or routes_jsonl != b"":
        raise ProductionPreflightError("smoke route evidence is not exactly empty")
    if type(removals_jsonl) is not bytes or removals_jsonl != b"":
        raise ProductionPreflightError("smoke removal evidence is not exactly empty")
    if type(elapsed_seconds) is not float or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ProductionPreflightError("smoke elapsed time must be a positive finite built-in float")
    v3_prefix = _read_regular(
        PROJECT_ROOT / "runs/tno_1kyr_reduced5_lifecycle_gate_v3/PREFIX_STATES.bin",
        "V3 prefix during smoke validation", 169_344,
    )
    if _sha256(v3_prefix) != "a27a929f6f440fb7484ef1bc3c9d55c94a05b24cdb28be0f57904e319b1a9a9b":
        raise ProductionPreflightError("V3 prefix changed during smoke validation")
    checkpoint_sha: list[str] = []
    selected_sha: list[str] = []
    for order, _year in enumerate(SMOKE_CHECKPOINT_YEARS):
        start = order * STATE_RECORD_BYTES
        payload = prefix_states[start:start + STATE_RECORD_BYTES]
        _unpack_rows(payload)
        if order == 0 and _sha256(payload) != protocol["full_clone_initial_states"]["batches"][0]["barycentric_checkpoint_payload_sha256"]:
            raise ProductionPreflightError("smoke full129 year-zero state differs from frozen initial state")
        selected = _selected_holdout_payload(payload)
        if selected != _v3_checkpoint(v3_prefix, SMOKE_BATCH_ORDER, order):
            raise ProductionPreflightError("smoke holdout projection differs bitwise from V3")
        checkpoint_sha.append(_sha256(payload))
        selected_sha.append(_sha256(selected))
    record: dict[str, Any] = {
        "schema": SMOKE_EVIDENCE_SCHEMA,
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_source_sha256": protocol["implementation"]["sha256"],
        "batch_order": SMOKE_BATCH_ORDER,
        "object_id": SMOKE_OBJECT_ID,
        "checkpoint_years": list(SMOKE_CHECKPOINT_YEARS),
        "checkpoint_sha256": checkpoint_sha,
        "selected_v3_projection_sha256": selected_sha,
        "active_counts": list(active_counts),
        "route_record_count": 0,
        "removal_record_count": 0,
        "all_components_finite": True,
        "all_129_tracers_active": True,
        "v3_holdout_projection_bitwise_equal": True,
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "elapsed_time_claim": "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE",
        "receipt_scope": "PROJECTION_ONLY_NOT_GPU_SOURCE_RUNTIME_LIFECYCLE_OR_EXECUTION_CUSTODY",
        "claims": {
            "projection_only_receipt_validated": True,
            "bounded_gpu_smoke_passed": False,
            "gpu_executed": False,
            "ten_myr_execution_authorized": False,
            "classification_generated": False,
            "science_result": False,
        },
    }
    record["content_sha256"] = _domain_hash(SMOKE_EVIDENCE_DOMAIN, record)
    return record


@contextmanager
def _held_parent_and_root(root: Path) -> Iterator[tuple[int, int, os.stat_result, os.stat_result]]:
    absolute = Path(os.path.abspath(os.fspath(root)))
    if absolute.name in ("", ".", ".."):
        raise ProductionPreflightError("prepared root name is invalid")
    parent = absolute.parent
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_before = os.fstat(parent_fd)
        root_lstat = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(root_lstat.st_mode):
            raise ProductionPreflightError("prepared root is not a lexical directory")
        root_fd = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            root_before = os.fstat(root_fd)
            if (root_lstat.st_dev, root_lstat.st_ino) != (root_before.st_dev, root_before.st_ino):
                raise ProductionPreflightError("prepared root changed before open")
            yield parent_fd, root_fd, parent_before, root_before
            parent_after = os.fstat(parent_fd)
            root_after = os.fstat(root_fd)
            lexical_parent = parent.stat(follow_symlinks=False)
            lexical_root = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            if (parent_before.st_dev, parent_before.st_ino) != (parent_after.st_dev, parent_after.st_ino) or (parent_before.st_dev, parent_before.st_ino) != (lexical_parent.st_dev, lexical_parent.st_ino):
                raise ProductionPreflightError("prepared root parent changed during use")
            if (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino) or (root_before.st_dev, root_before.st_ino) != (lexical_root.st_dev, lexical_root.st_ino):
                raise ProductionPreflightError("prepared root changed during use")
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProductionPreflightError("prepared protocol is not a single-link regular file")
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if len(payload) != before.st_size or not _same_stat(before, after) or not _same_stat(before, final):
        raise ProductionPreflightError("prepared protocol changed during read")
    return bytes(payload)


def _validate_prepared_fd(root_fd: int) -> dict[str, Any]:
    if set(os.listdir(root_fd)) != {"PROTOCOL.json"}:
        raise ProductionPreflightError("prepared root roster differs")
    payload = _read_regular_at(root_fd, "PROTOCOL.json")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise ProductionPreflightError("prepared protocol is not canonical")
    validate_protocol(value)
    if set(os.listdir(root_fd)) != {"PROTOCOL.json"}:
        raise ProductionPreflightError("prepared root roster changed during validation")
    return value


def validate_prepared(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    with _held_parent_and_root(root) as (_parent_fd, root_fd, _parent, _root):
        return _validate_prepared_fd(root_fd)


def prepare(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    absolute = Path(os.path.abspath(os.fspath(root)))
    parent = absolute.parent
    if absolute.name in ("", ".", ".."):
        raise ProductionPreflightError("prepared root name is invalid")
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    created = False
    try:
        parent_before = os.fstat(parent_fd)
        try:
            os.mkdir(absolute.name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise ProductionPreflightError("prepared root already exists; create-only publication refused") from exc
        created_stat = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(created_stat.st_mode):
            raise ProductionPreflightError("created root was substituted before open")
        root_fd = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            root_stat = os.fstat(root_fd)
            if (created_stat.st_dev, created_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino):
                raise ProductionPreflightError("created root was substituted before open")
            file_fd = os.open(
                "PROTOCOL.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o644, dir_fd=root_fd,
            )
            try:
                view = memoryview(payload)
                written = 0
                while written < len(view):
                    count = os.write(file_fd, view[written:])
                    if count <= 0:
                        raise ProductionPreflightError("short protocol write")
                    written += count
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.fsync(root_fd)
            value = _validate_prepared_fd(root_fd)
        finally:
            os.close(root_fd)
        parent_after = os.fstat(parent_fd)
        lexical_parent = parent.stat(follow_symlinks=False)
        lexical_root = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if (parent_before.st_dev, parent_before.st_ino) != (parent_after.st_dev, parent_after.st_ino) or (parent_before.st_dev, parent_before.st_ino) != (lexical_parent.st_dev, lexical_parent.st_ino):
            raise ProductionPreflightError("publication parent changed")
        if (created_stat.st_dev, created_stat.st_ino) != (lexical_root.st_dev, lexical_root.st_ino):
            raise ProductionPreflightError("published root changed")
        return value
    except Exception:
        # A failed create-only publication remains visible for manual custody;
        # this function never deletes or retries it.
        raise
    finally:
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate-prepared")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
    else:
        value = validate_prepared(arguments.output_root)
    sys.stdout.buffer.write(_canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
