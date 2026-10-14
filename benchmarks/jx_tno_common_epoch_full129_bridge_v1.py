#!/usr/bin/env python3
"""Preregister the blocked all-12/full-129 common-epoch TNO bridge.

This module is deliberately prepare-only.  It freezes the input custody,
identity, physical model, numerical lanes, and known acceptance gates for a
future CPU execution.  It does not expose an execution or resume command.

The authoritative physical model is a time-dependent restricted resolved-11
field read from an exact retained DE441 kernel.  No such kernel, common-epoch
nominal Cartesian oracle, or corresponding source/oracle gates are currently
bound, so execution remains fail-closed.  Autonomous resolved-11 propagation
may later be recorded only as a separate model-sensitivity diagnostic.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_common_epoch_full129_bridge_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_common_epoch_full129_bridge_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_common_epoch_full129_bridge_v1"
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})

PROTOCOL_SCHEMA = "jx.tno.common-epoch-full129-bridge.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.common-epoch-full129-bridge.protocol.v1\0"
EXPECTED_BRANCH = "research/tno-1kyr-common-epoch-full129-cpu-gpu-qualification-v1-20261014"

SOURCE_RESULT_TAG = "tno-1kyr-reduced5-h128-source-preflight-v1-result-20261014"
SOURCE_RESULT_TAG_OBJECT = "81a2c8d1d860b6685f84627971d0bbd880b696b1"
SOURCE_RESULT_TAG_PAYLOAD_BYTES = 543
SOURCE_RESULT_TAG_PAYLOAD_SHA256 = "edd5e707ab27a099ad5c58976288ea5daca4d7a216ae777e6fee237c2323c5e0"
SOURCE_RESULT_COMMIT = "661de5a83087b5050d66eff63f2be8fec0d3cff9"
SOURCE_RESULT_TREE = "295bba3bac50332b340404320ea0dddf24832ebe"
SOURCE_RESULT_CONTENT_SHA256 = "4e9c054491b867100879a04731eb1ae8a6217fff7eea97c20e0bee5259187f61"
SOURCE_RUNTIME_CONTENT_SHA256 = "6e1905c1b26c0198ca40560ae0563e3f1293c55598b0a26e96ba5036f005b15c"

TARGET_JD_TDB_DECIMAL = "2461200.5"
TARGET_JD_TDB_HEX = "0x1.2c70840000000p+21"
J2000_JD_TDB_DECIMAL = "2451545.0"
OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
SPK_IDS = (
    20090377, 20474640, 54526835, 20541132, 50666516, 20148209,
    20574372, 20303775, 20026181, 20042301, 20026375, 20015874,
)
SBDB_SOLUTION_IDS = (51, 12, 3, 7, 11, 5, 14, 29, 25, 28, 38, 34)
RESOLVED11_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
SOURCE_GM_KM3_PER_S2_DECIMAL = (
    "1.3271244004127942E+11", "2.2031868551400003E+04", "3.2485859200000000E+05",
    "3.9860043550702266E+05", "4.9028001184575496E+03", "4.2828375815756102E+04",
    "1.2671276409999998E+08", "3.7940584841799997E+07", "5.7945563999999985E+06",
    "6.8365271005803989E+06", "9.7550000000000000E+02",
)
SOURCE_GM_AU3_PER_JULIAN_YEAR2_HEX = (
    "0x1.3bd0beccac6afp+5", "0x1.b7cecb4d5b351p-18", "0x1.954f090f44727p-14",
    "0x1.f14fffb533692p-14", "0x1.877c4ea8180a5p-20", "0x1.ab7a3929b393ap-17",
    "0x1.34c6609ebf23fp-5", "0x1.71d0e2fd3b84cp-7", "0x1.c3d8f106aa649p-10",
    "0x1.0a8c8f8c38c6ap-9", "0x1.379282522bdf1p-22",
)
BATCH_COUNT = 12
TRACERS_PER_BATCH = 129
TRACER_ROWS = BATCH_COUNT * TRACERS_PER_BATCH
STATE_COMPONENTS = 6
STATE_BYTES = TRACER_ROWS * STATE_COMPONENTS * 8


@dataclass(frozen=True, slots=True)
class FileBinding:
    group: str
    role: str
    relative_path: str
    byte_length: int
    sha256: str


def _b(group: str, role: str, path: str, size: int, digest: str) -> FileBinding:
    return FileBinding(group, role, path, size, digest)


INPUT_BINDINGS = (
    _b("authoritative_native_cartesian", "protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"),
    _b("authoritative_native_cartesian", "states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
    _b("authoritative_de441_resolved11", "protocol", "runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", 11_986, "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b"),
    _b("authoritative_de441_resolved11", "custody", "runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", 141_501, "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"),
    _b("source_gm_authority", "horizons_perturbation_mass_parameters", "runs/tno_native_epoch_horizons_oracle_v1/source/gm_Horizons.pck", 15_428, "169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c"),
    _b("native_full129", "protocol", "runs/tno_full128_reduced5_bridge_v1/PROTOCOL.json", 10_469, "a97ce6daefd2417606219744cb168ab3dcb1cb2de676a1f1671bef10493b5ccb"),
    _b("native_full129", "states", "runs/tno_full128_reduced5_bridge_v1/BRIDGE.json", 1_208_834, "0570e7d81917caad871de2b2a143d4fc083851894d3c72aab1b77f8e9591fcec"),
    _b("native_resolved11", "protocol", "runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", 11_386, "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459"),
    _b("native_resolved11", "receipts", "runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", 5_810_062, "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5"),
    _b("native_resolved11", "states", "runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", 168_684, "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"),
    _b("sbdb", "protocol", "runs/tno_sbdb_covariance_acquisition_v1/PROTOCOL.json", 6_261, "4f3de80ba98b43c907bb846bd35724a0f1bd0ef9ef152e7642e3af65a2229acc"),
    _b("sbdb", "custody", "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json", 16_520, "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288"),
    _b("sbdb", "raw_20090377", "runs/tno_sbdb_covariance_acquisition_v1/raw/20090377.json", 3_855, "62bad94b65a7ab2472e20d771b5289003eafb9dedc6eca351935b98ba447bcc8"),
    _b("sbdb", "raw_20474640", "runs/tno_sbdb_covariance_acquisition_v1/raw/20474640.json", 3_855, "4b6e601b1cb41afb72428cfe7112622307c5a08c032a98277b1c9a688459035c"),
    _b("sbdb", "raw_54526835", "runs/tno_sbdb_covariance_acquisition_v1/raw/54526835.json", 3_808, "a47327d6de9c5b30a5d12aec549aef1454ec575fc4dd764bc49105a6c6471aef"),
    _b("sbdb", "raw_20541132", "runs/tno_sbdb_covariance_acquisition_v1/raw/20541132.json", 3_857, "6ce8dac67eaf465bde689d4e782ffa65a7c551195075d06d865c69d860d5b297"),
    _b("sbdb", "raw_50666516", "runs/tno_sbdb_covariance_acquisition_v1/raw/50666516.json", 3_796, "a43a00f133ad1b435416917030512038e23a4f7b52619f42a76a9c8763bc80ce"),
    _b("sbdb", "raw_20148209", "runs/tno_sbdb_covariance_acquisition_v1/raw/20148209.json", 3_798, "e2e4e08acff0fb4a4d37bec7cecf9656ac4633adc480676e3ac9b6ba25dae4b5"),
    _b("sbdb", "raw_20574372", "runs/tno_sbdb_covariance_acquisition_v1/raw/20574372.json", 3_824, "bcf5b50ee3bc4437ab0c55333249a128ba239a9fc8ba23d765acfcc6db9aeb91"),
    _b("sbdb", "raw_20303775", "runs/tno_sbdb_covariance_acquisition_v1/raw/20303775.json", 3_837, "a1e779145a7c91c7d954c63047a5e03302d4c38b0e9869f022937ba647a4a2ef"),
    _b("sbdb", "raw_20026181", "runs/tno_sbdb_covariance_acquisition_v1/raw/20026181.json", 3_823, "13ae906ea1c001d02b5c528ee3a2e5bfc9d0884860699bed2f0350fc0d17da73"),
    _b("sbdb", "raw_20042301", "runs/tno_sbdb_covariance_acquisition_v1/raw/20042301.json", 3_835, "0856c73adc67c8fb70813afe09fd4f933aa3216a15683935d62e24d73bdbdac8"),
    _b("sbdb", "raw_20026375", "runs/tno_sbdb_covariance_acquisition_v1/raw/20026375.json", 3_834, "09722418980836833b11ee8fae108964408ff7a8397c70c03e46e24a4e88a293"),
    _b("sbdb", "raw_20015874", "runs/tno_sbdb_covariance_acquisition_v1/raw/20015874.json", 3_839, "db9a0bd37a44e52527bbcb061dd65e725564240da773d36e1f5746f4bfcd00e3"),
    _b("source_result", "protocol", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/PROTOCOL.json", 20_740, "e37730efc8cee79fadfc8324d8e33b1a97b9732df92b0cb1162529c439601141"),
    _b("source_result", "runtime", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/RUNTIME.json", 185_599, "371616918961ebc9b9f970a69f11c8ad5418658c3d219a7a0243b66aa707f83a"),
    _b("source_result", "result", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/RESULT.json", 43_306, "8474bb1396885e46e16fca2e3ba6e5ff7bba50ede6c289c814afdc74c28f7ddc"),
    _b("source_result", "source_gm", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/SOURCE_GM.bin", 40, "85629de4232faa9a25eff5ec4e7e3a259467f4de528ab71f4f075f752eba3cb5"),
    _b("source_result", "h64_nodes", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/JX_H64_SOURCE_NODES.bin", 15_360_240, "db6ed3e1af1add38634a9942a9ba361c47732bef1fd57055923df1e5fc836e69"),
    _b("source_result", "h128_nodes", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/JX_H128_SOURCE_NODES.bin", 30_720_240, "83938788ac34028f2a484ca1a25ef9e1e1b12f95131dfe0b699fe898bb2225a4"),
    _b("source_result", "h64_invariants", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/JX_H64_ALL_NODE_INVARIANTS.bin", 2_048_032, "969b65cd2766d7f11c8848191d587f06fc00065052f7d7492a6223754d72ebf1"),
    _b("source_result", "h128_invariants", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/JX_H128_ALL_NODE_INVARIANTS.bin", 4_096_032, "aae313cfe29ecac5064e45570b745c6c0e10e87de2ea933a4a1dd36e3e97786c"),
    _b("source_result", "ias15_annual", "runs/tno_1kyr_reduced5_h128_source_preflight_v1_result/IAS15_ANNUAL_SOURCE_STATES.bin", 240_240, "c10af9779bab6742c2e6a025d32f32d9903d7a74d55eac78326e2a5036a3226f"),
    _b("runtime", "rebound_5_1_1_wheel", "runs/tno_stage1_rebound511_runtime_v1/rebound-5.1.1-cp314-manylinux_x86_64.whl", 903_973, "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19"),
)

CONTENT_BINDINGS = {
    "authoritative_native_cartesian_protocol": "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e",
    "authoritative_native_cartesian_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
    "authoritative_de441_resolved11_protocol": "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e",
    "authoritative_de441_resolved11_custody": "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3",
    "native_full129_protocol": "0458f700c64e798ecbfa927e6d722cd6d3b2b4ae8d6f70765f3fa2ea5684a0cd",
    "native_full129_states": "2c095be924f10481ced30af205836ebd18b4113398a7688b77f06259a2ac780b",
    "native_resolved11_protocol": "a7dbc361ffa4ccaf138f3c7afce269cad8bbb505b77930bcea9d04f5167b3c15",
    "native_resolved11_receipts": "7e5f3ef0a731647a9adaebedae65d6f703e44b8a923eb624d6dc872d935b8058",
    "native_resolved11_states": "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13",
    "sbdb_protocol": "176b56c4784bef58a8eacc6e34d99edb17d21d70c559a245b7c0677cdb1a4a60",
    "sbdb_custody": "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f",
    "source_protocol": "1c275624d21f815c7a2b9c9a8e18f77f63caf9356cf383bd1de3a6ad7228ae66",
    "source_runtime": SOURCE_RUNTIME_CONTENT_SHA256,
    "source_result": SOURCE_RESULT_CONTENT_SHA256,
}


class BridgeProtocolError(RuntimeError):
    """A frozen input, custody, schema, or publication invariant failed."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise BridgeProtocolError("record is not strict canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(PROTOCOL_DOMAIN + _canonical_json(record))


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise BridgeProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BridgeProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except BridgeProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise BridgeProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise BridgeProtocolError("JSON root must be an exact object")
    return value


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise BridgeProtocolError("binding path is not a contained POSIX path")
    return pure.parts


def _read_relative(root_fd: int, binding: FileBinding) -> bytes:
    parts = _relative_parts(binding.relative_path)
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size != binding.byte_length:
                raise BridgeProtocolError(f"{binding.role} type or byte length differs")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(file_fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise BridgeProtocolError(f"{binding.role} changed while held")
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)
    payload = b"".join(chunks)
    if _sha256(payload) != binding.sha256:
        raise BridgeProtocolError(f"{binding.role} SHA-256 differs")
    return payload


def _git(*args: str) -> bytes:
    process = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise BridgeProtocolError("Git ancestry check failed")
    return process.stdout


def _verify_git_ancestry() -> dict[str, object]:
    if _git("rev-parse", f"{SOURCE_RESULT_TAG}^{{tag}}").decode("ascii").strip() != SOURCE_RESULT_TAG_OBJECT:
        raise BridgeProtocolError("source result tag object differs")
    if _git("rev-parse", f"{SOURCE_RESULT_TAG}^{{commit}}").decode("ascii").strip() != SOURCE_RESULT_COMMIT:
        raise BridgeProtocolError("source result tag commit differs")
    if _git("rev-parse", f"{SOURCE_RESULT_TAG}^{{tree}}").decode("ascii").strip() != SOURCE_RESULT_TREE:
        raise BridgeProtocolError("source result tree differs")
    tag_payload = _git("cat-file", "tag", SOURCE_RESULT_TAG)
    if len(tag_payload) != SOURCE_RESULT_TAG_PAYLOAD_BYTES or _sha256(tag_payload) != SOURCE_RESULT_TAG_PAYLOAD_SHA256:
        raise BridgeProtocolError("source result tag payload differs")
    if subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_RESULT_COMMIT, "HEAD"], cwd=PROJECT_ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise BridgeProtocolError("HEAD does not descend from the sealed source result")
    return {
        "tag": SOURCE_RESULT_TAG,
        "tag_object": SOURCE_RESULT_TAG_OBJECT,
        "tag_payload_bytes": SOURCE_RESULT_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": SOURCE_RESULT_TAG_PAYLOAD_SHA256,
        "commit": SOURCE_RESULT_COMMIT,
        "tree": SOURCE_RESULT_TREE,
    }


def _binding_record(binding: FileBinding) -> dict[str, object]:
    return {"role": binding.role, "relative_path": binding.relative_path, "bytes": binding.byte_length, "sha256": binding.sha256}


def audit_prerequisites() -> dict[str, object]:
    root_fd = os.open(PROJECT_ROOT, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    payloads: dict[tuple[str, str], bytes] = {}
    try:
        for binding in INPUT_BINDINGS:
            payloads[(binding.group, binding.role)] = _read_relative(root_fd, binding)
    finally:
        os.close(root_fd)

    authoritative_cartesian_protocol = _strict_json(payloads[("authoritative_native_cartesian", "protocol")])
    authoritative_cartesian = _strict_json(payloads[("authoritative_native_cartesian", "states")])
    authoritative_de441_protocol = _strict_json(payloads[("authoritative_de441_resolved11", "protocol")])
    authoritative_de441 = _strict_json(payloads[("authoritative_de441_resolved11", "custody")])
    full = _strict_json(payloads[("native_full129", "states")])
    resolved = _strict_json(payloads[("native_resolved11", "states")])
    custody = _strict_json(payloads[("sbdb", "custody")])
    source_protocol = _strict_json(payloads[("source_result", "protocol")])
    source_runtime = _strict_json(payloads[("source_result", "runtime")])
    source_result = _strict_json(payloads[("source_result", "result")])
    checks = (
        (authoritative_cartesian_protocol.get("schema"), "jx.tno.native-epoch-cartesian.protocol.v1", "authoritative native Cartesian protocol schema"),
        (authoritative_cartesian_protocol.get("content_sha256"), CONTENT_BINDINGS["authoritative_native_cartesian_protocol"], "authoritative native Cartesian protocol content"),
        (authoritative_cartesian.get("schema"), "jx.tno.native-epoch-cartesian.states.v1", "authoritative native Cartesian state schema"),
        (authoritative_cartesian.get("content_sha256"), CONTENT_BINDINGS["authoritative_native_cartesian_states"], "authoritative native Cartesian state content"),
        (authoritative_de441_protocol.get("schema"), "jx.tno.de441-resolved11-native-epochs.protocol.v1", "authoritative DE441 protocol schema"),
        (authoritative_de441_protocol.get("content_sha256"), CONTENT_BINDINGS["authoritative_de441_resolved11_protocol"], "authoritative DE441 protocol content"),
        (authoritative_de441.get("schema"), "jx.tno.de441-resolved11-native-epochs.custody.v1", "authoritative DE441 custody schema"),
        (authoritative_de441.get("content_sha256"), CONTENT_BINDINGS["authoritative_de441_resolved11_custody"], "authoritative DE441 custody content"),
        (full.get("schema"), "jx.tno.full128-reduced5-bridge.states.v1", "native full129 schema"),
        (full.get("content_sha256"), CONTENT_BINDINGS["native_full129_states"], "native full129 content"),
        (resolved.get("schema"), "jx.tno.native-epoch-resolved11-bridge.states.v1", "resolved11 schema"),
        (resolved.get("content_sha256"), CONTENT_BINDINGS["native_resolved11_states"], "resolved11 content"),
        (custody.get("schema"), "jx.tno.sbdb-covariance-acquisition.custody.v1", "SBDB custody schema"),
        (custody.get("content_sha256"), CONTENT_BINDINGS["sbdb_custody"], "SBDB custody content"),
        (source_protocol.get("content_sha256"), CONTENT_BINDINGS["source_protocol"], "source protocol content"),
        (source_runtime.get("content_sha256"), SOURCE_RUNTIME_CONTENT_SHA256, "source runtime content"),
        (source_result.get("content_sha256"), SOURCE_RESULT_CONTENT_SHA256, "source result content"),
        (source_result.get("status"), "PASSED_BOUNDED_AUTONOMOUS_REDUCED5_SOURCE_PREFLIGHT", "source result status"),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise BridgeProtocolError(f"{label} differs")
    full_batches = full.get("batches")
    resolved_batches = resolved.get("batches")
    records = custody.get("records")
    cartesian_states = authoritative_cartesian.get("states")
    de441_states = authoritative_de441.get("state_records")
    if type(full_batches) is not list or type(resolved_batches) is not list or type(records) is not list or type(cartesian_states) is not list or type(de441_states) is not list:
        raise BridgeProtocolError("input batch rosters are not arrays")
    if len(full_batches) != BATCH_COUNT or len(resolved_batches) != BATCH_COUNT or len(records) != BATCH_COUNT:
        raise BridgeProtocolError("input batch counts differ")
    if (
        authoritative_cartesian.get("object_count") != BATCH_COUNT
        or authoritative_cartesian.get("nominal_state_count") != BATCH_COUNT
        or authoritative_cartesian.get("clone_state_count") != BATCH_COUNT * (TRACERS_PER_BATCH - 1)
        or authoritative_cartesian.get("total_state_count") != TRACER_ROWS
        or len(cartesian_states) != TRACER_ROWS
    ):
        raise BridgeProtocolError("authoritative native Cartesian state geometry differs")
    if (
        authoritative_de441.get("epoch_count") != BATCH_COUNT
        or authoritative_de441.get("body_count") != len(RESOLVED11_NAIF_IDS)
        or authoritative_de441.get("state_record_count") != BATCH_COUNT * len(RESOLVED11_NAIF_IDS)
        or len(de441_states) != BATCH_COUNT * len(RESOLVED11_NAIF_IDS)
    ):
        raise BridgeProtocolError("authoritative native DE441 state geometry differs")
    native_roster: list[dict[str, object]] = []
    target = Decimal(TARGET_JD_TDB_DECIMAL)
    for order, (full_batch, resolved_batch, record, object_id, spk_id, solution_id) in enumerate(zip(full_batches, resolved_batches, records, OBJECT_IDS, SPK_IDS, SBDB_SOLUTION_IDS, strict=True)):
        if type(full_batch) is not dict or type(resolved_batch) is not dict or type(record) is not dict:
            raise BridgeProtocolError("input batch row is not an exact object")
        epoch_hex = full_batch.get("covariance_epoch_jd_tdb_hex")
        expected = (order, object_id, spk_id, 129)
        observed = (full_batch.get("batch_order"), full_batch.get("object_id"), full_batch.get("spk_id"), full_batch.get("tracer_count"))
        if observed != expected or type(full_batch.get("tracers")) is not list or len(full_batch["tracers"]) != TRACERS_PER_BATCH:
            raise BridgeProtocolError("native full129 identity or geometry differs")
        for tracer_order, tracer in enumerate(full_batch["tracers"]):
            if type(tracer) is not dict or tracer.get("tracer_order") != tracer_order or tracer.get("object_id") != object_id or tracer.get("spk_id") != spk_id:
                raise BridgeProtocolError("native tracer identity or order differs")
            expected_clone = None if tracer_order == 0 else tracer_order - 1
            if tracer.get("clone_index") != expected_clone:
                raise BridgeProtocolError("native tracer clone identity differs")
            cartesian = cartesian_states[order * TRACERS_PER_BATCH + tracer_order]
            if (
                type(cartesian) is not dict
                or cartesian.get("object_id") != object_id
                or cartesian.get("spk_id") != spk_id
                or cartesian.get("clone_index") != expected_clone
                or cartesian.get("covariance_epoch_jd_tdb_hex") != epoch_hex
                or cartesian.get("native_vector_sha256") != tracer.get("native_vector_sha256")
                or type(cartesian.get("position_AU_hex")) is not list
                or len(cartesian["position_AU_hex"]) != 3
                or type(cartesian.get("velocity_AU_per_julian_year_hex")) is not list
                or len(cartesian["velocity_AU_per_julian_year_hex"]) != 3
            ):
                raise BridgeProtocolError("authoritative native Cartesian state identity differs")
        if resolved_batch.get("object_id") != object_id or resolved_batch.get("covariance_epoch_jd_tdb_hex") != epoch_hex or resolved_batch.get("massive_body_count") != 11:
            raise BridgeProtocolError("resolved11 batch identity differs")
        if [row.get("naif_id") for row in resolved_batch.get("massive_bodies", [])] != list(RESOLVED11_NAIF_IDS):
            raise BridgeProtocolError("resolved11 source identity differs")
        if record.get("object_id") != object_id or record.get("spk_id") != spk_id or record.get("standard_epoch_jd_tdb") != TARGET_JD_TDB_DECIMAL:
            raise BridgeProtocolError("SBDB identity or standard epoch differs")
        raw = _strict_json(payloads[("sbdb", f"raw_{spk_id}")])
        orbit = raw.get("orbit")
        if (
            type(orbit) is not dict
            or orbit.get("orbit_id") != str(solution_id)
            or orbit.get("epoch") != TARGET_JD_TDB_DECIMAL
            or orbit.get("sb_used") != "SB441-N16"
            or orbit.get("pe_used") != "DE441"
        ):
            raise BridgeProtocolError("SBDB frozen solution or standard-epoch model differs")
        native_decimal = Decimal(str(record.get("covariance_epoch_jd_tdb")))
        if float(native_decimal).hex() != epoch_hex or native_decimal >= target:
            raise BridgeProtocolError("native epoch cross-binding differs")
        for body_order, naif_id in enumerate(RESOLVED11_NAIF_IDS):
            source_state = de441_states[order * len(RESOLVED11_NAIF_IDS) + body_order]
            if (
                type(source_state) is not dict
                or source_state.get("epoch_order") != order
                or source_state.get("body_order") != body_order
                or source_state.get("object_id") != object_id
                or source_state.get("spk_id") != spk_id
                or source_state.get("naif_id") != naif_id
                or source_state.get("requested_jd_tdb") != str(native_decimal)
                or source_state.get("ephemeris_source") != "DE441"
                or source_state.get("frame") != "ICRF"
                or source_state.get("center_naif_id") != 0
                or source_state.get("corrections") != "GEOMETRIC_NONE"
            ):
                raise BridgeProtocolError("authoritative native DE441 state identity differs")
        native_roster.append({
            "batch_order": order,
            "object_id": object_id,
            "spk_id": spk_id,
            "sbdb_solution_id": solution_id,
            "native_epoch_jd_tdb_decimal": str(native_decimal),
            "native_epoch_jd_tdb_hex": epoch_hex,
            "forward_delta_days": int(target - native_decimal),
            "tracer_orders": list(range(TRACERS_PER_BATCH)),
        })
    grouped: dict[str, list[dict[str, object]]] = {}
    for binding in INPUT_BINDINGS:
        grouped.setdefault(binding.group, []).append(_binding_record(binding))
    return {
        "git_source_result": _verify_git_ancestry(),
        "input_bindings": grouped,
        "input_content_bindings": dict(CONTENT_BINDINGS),
        "native_roster": native_roster,
    }


def _implementation_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "bytes": len(payload), "sha256": _sha256(payload)}


def build_protocol() -> dict[str, object]:
    prerequisites = audit_prerequisites()
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "status": "PREPARED_BLOCKED_COMMON_EPOCH_FULL129_BRIDGE_PREREGISTRATION_NO_EXECUTION",
        "evidence_class": "CPU_ONLY_PREPARE_ONLY_INPUT_AND_METHOD_CONTRACT_NO_MODEL_OUTPUT",
        "authority": {
            "expected_branch_for_preparation": EXPECTED_BRANCH,
            "source_result_ancestry": prerequisites["git_source_result"],
            "preregistration_required_before_any_executor_is_added": True,
            "preregistration_absence_of_a_future_executor_mechanically_proven_by_this_builder": False,
            "successor_enforcement": "ANY_EXECUTOR_MUST_BIND_THE_FUTURE_ANNOTATED_PREREGISTRATION_TAG_OBJECT_PEELED_COMMIT_TREE_AND_EXACT_SOURCE_TEST_PROTOCOL_BYTES_BEFORE_LANE_ZERO",
            "execution_in_this_artifact": False,
            "public_cli": ["build", "audit-prerequisites", "prepare", "validate-prepared"],
            "explicitly_absent_public_cli": ["execute", "execute-worker", "resume", "validate-result"],
            "network_access": False,
        },
        "implementation": {"source": _implementation_record(SOURCE_PATH), "test": _implementation_record(TEST_PATH)},
        "inputs": {
            "bindings": prerequisites["input_bindings"],
            "content_bindings": prerequisites["input_content_bindings"],
            "native_roster": prerequisites["native_roster"],
            "identity_policy": "ALL_12_OBJECTS_TIMES_NOMINAL_PLUS_CLONES_0_THROUGH_127_EXACT_ORDER_NO_RESAMPLING_NO_RELABELING_NO_DROPPING",
            "source_result_reuse": "READ_ONLY_ANCESTRY_AND_FUTURE_CONSUMER_FRAME_EVIDENCE;SEALED_FILES_MUST_NOT_BE_COPIED_REGENERATED_OR_MUTATED",
        },
        "target": {
            "epoch_scale": "TDB",
            "jd_tdb_decimal": TARGET_JD_TDB_DECIMAL,
            "jd_tdb_binary64_hex": TARGET_JD_TDB_HEX,
            "basis": "STANDARD_EPOCH_RECORDED_FOR_ALL_TWELVE_BOUND_SBDB_CUSTODY_ROWS",
            "direction": "FORWARD_FROM_EACH_OBJECT_SPECIFIC_NATIVE_COVARIANCE_EPOCH",
            "batch_local_time_origin": "NATIVE_EPOCH_IS_T_EQUALS_ZERO_JULIAN_YEARS",
            "source_query_time": "EXACT_TWO_PART_JD_TDB_DERIVED_FROM_NATIVE_EPOCH_PLUS_BATCH_LOCAL_TIME;NO_FLOATING_POINT_TIME_ACCUMULATION",
        },
        "geometry": {
            "batch_count": BATCH_COUNT,
            "tracers_per_batch": TRACERS_PER_BATCH,
            "tracer_rows": TRACER_ROWS,
            "state_components_per_tracer": STATE_COMPONENTS,
            "common_epoch_state_bytes_per_lane": STATE_BYTES,
            "tracer_order": "NOMINAL_THEN_CLONE_0_THROUGH_CLONE_127",
            "object_order": list(OBJECT_IDS),
        },
        "physical_model": {
            "authoritative_bridge": "DE441_TIME_DEPENDENT_RESTRICTED_RESOLVED11",
            "cpu_only": True,
            "source_naif_ids": list(RESOLVED11_NAIF_IDS),
            "source_requirement": "ONE_EXACT_RETAINED_HASHED_DE441_KERNEL_WITH_COVERAGE_OVER_EVERY_NATIVE_TO_TARGET_AND_ROUNDTRIP_INTERVAL",
            "de441_kernel_acquisition_requirement": {
                "official_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-2.bsp",
                "official_checksum_roster_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_checksums.txt",
                "official_summary_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_summaries.txt",
                "official_technical_comments_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_tech-comments.txt",
                "byte_length": 1_656_830_976,
                "official_md5": "ad8dfa4e505ef0e3a5d587a5b4705632",
                "coverage_jd_tdb": ["2440400.50", "8000016.50"],
                "untouched_official_file_required_for_first_qualification": True,
                "local_sha256_and_before_after_execution_hashes_required": True,
            },
            "source_frame": "SOLAR_SYSTEM_BARYCENTRIC_J2000_ICRF",
            "source_time_scale": "TDB",
            "source_units": "AU_AND_AU_PER_JULIAN_YEAR_WITH_GM_IN_AU3_PER_JULIAN_YEAR2",
            "source_gm_contract": {
                "authority_binding_group": "source_gm_authority",
                "authority_semantics": "EXACT_HORIZONS_SMALL_BODY_NUMERICAL_INTEGRATION_PERTURBATION_MASSES_DOCUMENTED_IN_THE_BOUND_GM_PCK;NOT_THE_LATER_BODY_ASSIGNMENT_VALUES_USED_FOR_OSCULATING_ELEMENTS",
                "naif_ids_in_exact_force_sum_order": list(RESOLVED11_NAIF_IDS),
                "gm_km3_per_s2_exact_decimal_tokens": list(SOURCE_GM_KM3_PER_S2_DECIMAL),
                "unit_conversion": "CORRECTLY_ROUNDED_BINARY64_ONCE_FROM_EXACT_DECIMAL_GM_TIMES_31557600_SQUARED_DIVIDED_BY_149597870_7_KM_CUBED",
                "julian_year_seconds_exact": 31_557_600,
                "iau_2012_astronomical_unit_km_exact_decimal": "149597870.7",
                "gm_AU3_per_julian_year2_binary64_hex": list(SOURCE_GM_AU3_PER_JULIAN_YEAR2_HEX),
                "force_accumulation_order": "ASCENDING_INDEX_IN_THIS_EXACT_RESOLVED11_ROSTER_WITH_A_SEPARATELY_FROZEN_SUMMATION_IMPLEMENTATION_REQUIRED_IN_THE_EXECUTION_SUCCESSOR",
            },
            "tracer_dynamics": "MASSLESS_NO_MUTUAL_INTERACTION_NO_BACKREACTION;ACCELERATION_IS_SUM_GM_I_TIMES_R_I_OF_T_MINUS_R_DIV_NORM_CUBED",
            "identical_external_source_for_every_numerical_lane": True,
            "autonomous_resolved11": "SEPARATE_MODEL_SENSITIVITY_DIAGNOSTIC_ONLY_NEVER_NUMERICAL_OR_PHYSICAL_BRIDGE_AUTHORITY",
            "gpu_allowed": False,
        },
        "initial_state_contract": {
            "authoritative_tracer_rows": "BOUND_NATIVE_EPOCH_CARTESIAN_STATES_HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_ROWS",
            "authoritative_position_field": "position_AU_hex",
            "authoritative_velocity_field": "velocity_AU_per_julian_year_hex",
            "native_full129_bridge_role": "IDENTITY_ORDER_NATIVE_VECTOR_DIGEST_AND_DERIVED_CONSUMER_PROJECTION_CROSSCHECK_ONLY;ITS_ALREADY_ROTATED_HELIOCENTRIC_J2000_EQUATORIAL_ICRF_COMPONENTS_ARE_NOT_THE_AUTHORITATIVE_INPUT_FIELDS",
            "native_ssb_conversion": "ADD_THE_EXACT_DE441_SUN_SSB_STATE_FROM_THE_BOUND_FUTURE_KERNEL_AT_EACH_NATIVE_EPOCH",
            "input_transform": "ROTATE_HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_TO_ICRF_THEN_ADD_DE441_SUN_SSB_STATE",
            "frame_rotation_contract": {
                "authority": "BOUND_NATIVE_FULL129_PROTOCOL_COORDINATE_CONTRACT",
                "obliquity_arcseconds_exact_decimal": "84381.448",
                "obliquity_radians_binary64_hex": "0x1.a2e9395ca7c40p-2",
                "sine_binary64_hex": "0x1.9752e50f4b399p-2",
                "cosine_binary64_hex": "0x1.d5c0357681ef3p-1",
                "forward_ecliptic_to_icrf": "X_EQ=X_ECL;Y_EQ=ROUND_ONCE(COS*Y_ECL-SIN*Z_ECL);Z_EQ=ROUND_ONCE(SIN*Y_ECL+COS*Z_ECL)",
                "inverse_icrf_to_ecliptic": "X_ECL=X_EQ;Y_ECL=ROUND_ONCE(COS*Y_EQ+SIN*Z_EQ);Z_ECL=ROUND_ONCE(-SIN*Y_EQ+COS*Z_EQ)",
                "arithmetic": "EVERY_MULTIPLY_AND_ADD_EXPRESSION_IS_EVALUATED_AS_AN_EXACT_BINARY64_DYADIC_RATIONAL_THEN_THE_COMPLETE_COMPONENT_IS_ROUNDED_ONCE_TO_NEAREST_BINARY64_TIES_TO_EVEN",
                "applies_identically_to_position_and_velocity": True,
            },
            "integration_coordinates": "SOLAR_SYSTEM_BARYCENTRIC_ICRF_J2000_AU_AND_AU_PER_JULIAN_YEAR",
            "authoritative_output_transform": "SUBTRACT_DE441_SUN_SSB_STATE_AT_TARGET_THEN_ROTATE_ICRF_TO_IAU76_80_ECLIPTIC_J2000",
            "authoritative_output_coordinates": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_AU_AND_AU_PER_JULIAN_YEAR",
            "reduced5_consumer_projection": "DERIVED_ONLY_AFTER_BRIDGE_PASS_USING_ONE_EXACT_COMMON_EPOCH_DE441_SOURCE_NODE;NEVER_ALTERS_AUTHORITATIVE_TRACER_BYTES",
            "reduced5_native_payload_not_authoritative_for_physical_propagation": True,
            "all_rows_finite_and_massless_required": True,
        },
        "numerical_lanes": {
            "production": {
                "library": "REBOUND", "required_version": "5.1.1", "integrator": "IAS15",
                "retained_wheel_sha256": "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19",
                "gravity": "NONE_BUILTIN;EXTERNAL_RESTRICTED_ACCELERATION_CALLBACK_AT_EVERY_FORCE_EVALUATION",
                "particle_contract": "N_129;N_ACTIVE_0;EVERY_PARTICLE_MASS_POSITIVE_ZERO;TESTPARTICLE_TYPE_0",
                "additional_forces_callback": "REQUIRED_NON_NULL_PINNED_CALLBACK_WITH_ONE_DE441_SOURCE_QUERY_PER_FORCE_EVALUATION",
                "epsilon_decimal": "1e-13", "epsilon_hex": (1e-13).hex(),
                "adaptive_mode": "PRS23", "minimum_dt": "0", "initial_dt_julian_years_hex": (1.0 / 32.0).hex(),
                "exact_finish_time": True, "batch_particle_count": TRACERS_PER_BATCH,
            },
            "independent": {
                "library": "SciPy", "api": "scipy.integrate.solve_ivp", "method": "DOP853",
                "rtol_decimal": "1e-13", "rtol_hex": (1e-13).hex(),
                "atol_decimal": "1e-15", "atol_hex": (1e-15).hex(),
                "max_step_julian_years": "1/128", "max_step_hex": (1.0 / 128.0).hex(),
                "identical_external_source_required": True, "exact_distribution_and_loaded_module_receipts_required": True,
            },
            "required_convergence_replicate": {
                "optional": False, "library": "REBOUND", "required_version": "5.1.1", "integrator": "IAS15",
                "epsilon_decimal": "1e-14", "epsilon_hex": (1e-14).hex(),
                "all_other_settings_identical_to_production": True, "all_12_times_129_required": True,
            },
            "execution_order": "EACH_BATCH_ALL129_TOGETHER;BATCH_OUTPUTS_CAN_BE_COMPUTED_IN_PARALLEL_BUT_ARE_PUBLISHED_IN_FROZEN_ORDER",
            "forward_and_backward_roundtrip_required_for_every_lane": True,
        },
        "acceptance": {
            "arithmetic": "COMPARE_EXACT_BINARY64_SQUARED_L2_NORMS_TO_EXACT_RATIONAL_THRESHOLD_SQUARES;NO_SQRT_DECISION",
            "method_endpoint_pairs": ["PRODUCTION_VS_DOP853", "PRODUCTION_VS_IAS15_EPSILON_1E_MINUS_14", "DOP853_VS_IAS15_EPSILON_1E_MINUS_14"],
            "same_method_roundtrip_lanes": ["PRODUCTION", "DOP853", "IAS15_EPSILON_1E_MINUS_14"],
            "cross_method_roundtrip_paths": [
                "PRODUCTION_FORWARD_THEN_DOP853_BACKWARD",
                "DOP853_FORWARD_THEN_PRODUCTION_BACKWARD",
                "IAS15_EPSILON_1E_MINUS_14_FORWARD_THEN_DOP853_BACKWARD",
                "DOP853_FORWARD_THEN_IAS15_EPSILON_1E_MINUS_14_BACKWARD",
            ],
            "position_AU_max": {"numerator": 1, "denominator": 1_000_000_000, "nearest_binary64_hex": (1e-9).hex()},
            "velocity_AU_per_julian_year_max": {"numerator": 1, "denominator": 10_000_000_000, "nearest_binary64_hex": (1e-10).hex()},
            "scope": "EVERY_BATCH_EVERY_TRACER_EVERY_POSITION_AND_VELOCITY_VECTOR;NO_OMISSION_COUNTS_AS_PASS",
            "nonfinite_or_identity_difference": "FAIL",
            "parameter_tuning_retry_or_selective_rerun": "FORBIDDEN",
            "source_node_gate": {"required": True, "status": "UNBOUND_EXECUTION_BLOCKER", "exact_artifact": None, "tolerances": None},
            "common_epoch_nominal_cartesian_oracle": {
                "required": True,
                "coverage": "NOMINAL_TRACER_ORDER_0_FOR_ALL_TWELVE_OBJECTS",
                "status": "UNBOUND_EXECUTION_BLOCKER",
                "exact_artifact": None,
                "acquisition": "HORIZONS_DESIGNATION_QUERY_ONLY_WITH_COMMAND_DES_EQUALS_FROZEN_SPK_ID;REQUIRE_EXACT_FROZEN_RETURNED_SOLUTION_ID;USER_SUPPLIED_ELEMENT_PROPAGATION_IS_OUT_OF_SCOPE_AND_REQUIRES_A_SEPARATELY_VERSIONED_DIAGNOSTIC",
                "frozen_designation_solution_ids_in_object_order": list(SBDB_SOLUTION_IDS),
                "provider_contract": "CENTER_500_AT_10;TLIST_2461200_5;TDB;ECLIPTIC_J2000;AU_D;GEOMETRIC_NO_CORRECTIONS;EXACTLY_ONE_FINITE_VECTOR_ROW",
                "lineage_gates": "EPOCH_E_Q_NODE_PERI_I_DECIMAL_TOKENS_EXACTLY_EQUAL_BOUND_SBDB_STANDARD_ELEMENTS;TP_BINARY64_ULP_DISTANCE_LE_1;SB_USED_SB441_N16;PE_USED_DE441",
                "hard_acceptance_gates": [
                    "EXACT_TWELVE_OBJECT_AND_FROZEN_SOLUTION_ID_ROSTER",
                    "EXACT_QUERY_CENTER_EPOCH_TIME_SCALE_FRAME_UNITS_AND_NO_CORRECTIONS",
                    "EXACTLY_ONE_FINITE_SIX_COMPONENT_VECTOR_ROW_PER_OBJECT",
                    "EXACT_STANDARD_ELEMENT_LINEAGE_AND_TP_BINARY64_ULP_DISTANCE_LE_1",
                    "EXACT_RESPONSE_BYTE_HASH_ROSTER_AND_CANONICAL_PARSE_REPLAY",
                ],
                "comparison_role": "PHYSICAL_MODEL_DISCREPANCY_DIAGNOSTIC_NOT_NUMERICAL_ACCEPTANCE_GATE",
                "tolerance_policy": "NO_PASS_TOLERANCE;RETAIN_EXACT_COMPONENT_NORM_AND_RADIAL_TRANSVERSE_NORMAL_RESIDUALS",
                "parent_oracle_tolerance_requirement_resolution": "THE_ORACLE_HAS_EXACT_SCHEMA_LINEAGE_AND_CUSTODY_ACCEPTANCE_GATES_BUT_NO_PHYSICAL_RESIDUAL_TOLERANCE;A_HORIZONS_SB441_N16_TRAJECTORY_AND_THIS_RESTRICTED_RESOLVED11_MODEL_HAVE_DIFFERENT_FORCE_MODELS_SO_A_RESIDUAL_THRESHOLD_WOULD_CONFOUND_MODEL_DISCREPANCY_WITH_NUMERICAL_ERROR",
                "binding_without_every_hard_acceptance_gate_passing": "FORBIDDEN",
                "horizons_force_model_equivalence_claimed": False,
            },
        },
        "successor_evidence_geometry": {
            "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
            "shape_each": [BATCH_COUNT, TRACERS_PER_BATCH, STATE_COMPONENTS],
            "bytes_each": STATE_BYTES,
            "core_state_files": [
                "PRODUCTION_COMMON_EPOCH_STATES.bin", "DOP853_COMMON_EPOCH_STATES.bin", "IAS15_E14_COMMON_EPOCH_STATES.bin",
                "PRODUCTION_ROUNDTRIP_STATES.bin", "DOP853_ROUNDTRIP_STATES.bin", "IAS15_E14_ROUNDTRIP_STATES.bin",
                "PRODUCTION_FORWARD_DOP853_BACKWARD_STATES.bin", "DOP853_FORWARD_PRODUCTION_BACKWARD_STATES.bin",
                "IAS15_E14_FORWARD_DOP853_BACKWARD_STATES.bin", "DOP853_FORWARD_IAS15_E14_BACKWARD_STATES.bin",
            ],
            "publication_contract_must_be_completed_by_blocker_resolving_successor": True,
        },
        "execution_blockers": [
            "EXACT_RETAINED_HASHED_DE441_KERNEL_WITH_FULL_INTERVAL_COVERAGE_NOT_BOUND",
            "EXACT_DE441_SOURCE_EVALUATOR_RUNTIME_AND_SOURCE_NODE_GATES_NOT_BOUND",
            "COMMON_EPOCH_CARTESIAN_NOMINAL_DIAGNOSTIC_ORACLE_NOT_BOUND",
            "EXACT_SCIPY_DOP853_DISTRIBUTION_AND_LOADED_RUNTIME_NOT_BOUND",
            "EXECUTOR_RESULT_SCHEMA_AND_COMPLETE_PUBLICATION_ROSTER_NOT_PREREGISTERED",
        ],
        "claims": {
            "prepare_only_protocol_frozen": True,
            "all12_full129_identity_frozen": True,
            "source_result_ancestry_bound": True,
            "de441_kernel_bound": False,
            "common_epoch_nominal_oracle_bound": False,
            "source_node_gate_frozen": False,
            "execution_authorized": False,
            "dynamics_executed": False,
            "gpu_executed": False,
            "common_epoch_bridge_executed": False,
            "common_epoch_bridge_qualified": False,
            "autonomous_resolved11_authoritative": False,
            "short_horizon_tracer_qualification_authorized": False,
            "three_gyr_execution_authorized": False,
            "classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "future_result_claim_ceiling": "QUALIFIED_COMMON_EPOCH_FULL129_INPUT_BRIDGE_FOR_THIS_FROZEN_DE441_RESTRICTED_RESOLVED11_FORCE_MODEL_ONLY",
        "nonclaims": {
            "horizons_full_force_model_equivalence": False,
            "covariance_posterior_or_probability": False,
            "dynamical_classification": False,
            "physical_stability_collision_or_ejection": False,
            "population_inference": False,
            "planet_nine_detection_exclusion_or_ranking": False,
            "long_horizon_accuracy_or_three_gyr_execution_authority": False,
        },
        "publication": {
            "prepared_roster": ["PROTOCOL.json"], "root_must_not_exist": True,
            "create_only_no_overwrite": True, "directory_mode_octal": "0700", "file_mode_octal": "0600",
            "file_and_directory_fsync_required": True, "validation_is_read_only": True,
        },
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    return protocol


def _validate_protocol(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or value.get("schema") != PROTOCOL_SCHEMA:
        raise BridgeProtocolError("protocol schema differs")
    if value.get("content_sha256") != _content_sha256(value):
        raise BridgeProtocolError("protocol content SHA-256 differs")
    if value.get("claims", {}).get("execution_authorized") is not False:
        raise BridgeProtocolError("protocol cannot authorize execution")
    if value.get("acceptance", {}).get("common_epoch_nominal_cartesian_oracle", {}).get("status") != "UNBOUND_EXECUTION_BLOCKER":
        raise BridgeProtocolError("oracle blocker is not preserved")


def _current_branch() -> str:
    return _git("branch", "--show-current").decode("utf-8").strip()


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
    for component in reversed([absolute, *absolute.parents]):
        if component.exists() and component.is_symlink():
            raise BridgeProtocolError(f"{label} has a symlink ancestor")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise BridgeProtocolError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise BridgeProtocolError("renameat2 is unavailable")
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
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise BridgeProtocolError("protocol destination already exists")
        raise BridgeProtocolError(
            f"renameat2 failed: {os.strerror(observed_errno)}"
        )


def _validate_prepared_fd(
    root_fd: int,
    expected_payload: bytes,
) -> dict[str, object]:
    root_metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != os.geteuid()
    ):
        raise BridgeProtocolError("prepared root is not an owner-private directory")
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise BridgeProtocolError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise BridgeProtocolError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(payload)),
                len(payload),
            )
            if not chunk:
                raise BridgeProtocolError("prepared protocol ended early")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise BridgeProtocolError("prepared protocol changed while held")
    finally:
        os.close(descriptor)
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise BridgeProtocolError("prepared root changed while held")
    exact_payload = bytes(payload)
    if exact_payload != expected_payload:
        raise BridgeProtocolError("prepared protocol bytes differ")
    value = _strict_json(exact_payload)
    if exact_payload != _canonical_json(value):
        raise BridgeProtocolError("prepared protocol is not canonical JSON")
    _validate_protocol(value)
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if _current_branch() != EXPECTED_BRANCH:
        raise BridgeProtocolError("preparation branch differs")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise BridgeProtocolError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(root.parent, flags)
    except OSError as exc:
        raise BridgeProtocolError("cannot open prepared parent") from exc
    created = False
    prepared: dict[str, object] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise BridgeProtocolError(
                "prepared root already exists; create-only publication refused"
            ) from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            if os.listdir(root_fd):
                raise BridgeProtocolError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            file_fd = os.open(
                partial,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                os.fchmod(file_fd, 0o600)
                _write_all(file_fd, payload)
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_fd(root_fd, payload)
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
    except BridgeProtocolError:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    except OSError as exc:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise BridgeProtocolError("cannot create prepared protocol") from exc
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise BridgeProtocolError("prepared protocol validation did not complete")
    return prepared


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    protocol = build_protocol()
    expected_payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = os.open(root, flags)
    try:
        value = _validate_prepared_fd(root_fd, expected_payload)
    finally:
        os.close(root_fd)
    if value != protocol:
        raise BridgeProtocolError("prepared protocol differs from rebuilt protocol")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="build and print the protocol without writing")
    subparsers.add_parser("audit-prerequisites", help="verify every frozen input without writing")
    prepare_parser = subparsers.add_parser("prepare", help="create only PROTOCOL.json")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate-prepared", help="read-only validation of a prepared root")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        value = build_protocol()
        print(_canonical_json(value).decode("ascii"), end="")
    elif args.command == "audit-prerequisites":
        value = audit_prerequisites()
        print(json.dumps({"binding_count": len(INPUT_BINDINGS), "status": "PASSED", "source_result_commit": value["git_source_result"]["commit"]}, sort_keys=True))
    elif args.command == "prepare":
        value = prepare(args.output_root)
        print(json.dumps({"content_sha256": value["content_sha256"], "status": value["status"]}, sort_keys=True))
    elif args.command == "validate-prepared":
        value = validate_prepared(args.output_root)
        print(json.dumps({"content_sha256": value["content_sha256"], "status": "VALID"}, sort_keys=True))
    else:
        raise BridgeProtocolError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
