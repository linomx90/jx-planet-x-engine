#!/usr/bin/env python3
"""Freeze the fail-closed Stage-1.5 resolved-11 encounter qualification.

This module verifies retained inputs and can create or validate one canonical
protocol.  It intentionally has no dynamics execution mode.  The currently
available encounter-capable CUDA engine remains ``MODEL_OUTPUT /
SCREENING_ONLY`` and does not expose all evidence required to reproduce the
inherited Stage-1 acceptance gates.  A later, independently reviewed adapter
must close every blocker recorded here before any GPU dynamics are launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_stage15_resolved11_encounter_qualification_v1.py"
)
TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_tno_stage15_resolved11_encounter_qualification_v1.py"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "tno_stage15_resolved11_encounter_qualification_v1"
    / "PROTOCOL.json"
)

SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.protocol.v1"
DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.protocol.v1\0"
CLASSIFICATION = (
    "TNO_STAGE15_DE441_RESOLVED11_ENCOUNTER_ENGINE_QUALIFICATION_PREPARATION_ONLY"
)
CLAIM_CEILING = (
    "PREPARED_1000_YEAR_RESOLVED11_ENCOUNTER_NUMERICAL_QUALIFICATION_NO_EXECUTION_NO_CLASSIFICATION"
)

OBJECT_IDS = (
    "jpl-20090377",
    "jpl-20474640",
    "jpl-54526835",
    "jpl-20541132",
    "jpl-50666516",
    "jpl-20148209",
    "jpl-20574372",
    "jpl-20303775",
    "jpl-20026181",
    "jpl-20042301",
    "jpl-20026375",
    "jpl-20015874",
)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
TRACER_INDICES = tuple(range(9))
MASSIVE_ROSTER = (
    (10, "SUN"),
    (199, "MERCURY"),
    (299, "VENUS"),
    (399, "EARTH"),
    (301, "MOON"),
    (4, "MARS_BARYCENTER"),
    (5, "JUPITER_BARYCENTER"),
    (6, "SATURN_BARYCENTER"),
    (7, "URANUS_BARYCENTER"),
    (8, "NEPTUNE_BARYCENTER"),
    (9, "PLUTO_BARYCENTER"),
)

# role, relative path, exact byte count, exact SHA-256, retain payload
FILE_BINDINGS = (
    ("pilot_implementation", "benchmarks/jx_tno_covariance_classification_pilot_v1.py", 31_585, "7d90ff40e71400071eb7d0ecf315113342e7c76cf8ae91935310f8895fcde2e1", False),
    ("pilot_test", "tests/test_tno_covariance_classification_pilot_v1.py", 13_134, "10d3a8b162ecc325370b7e973bceb4da5b9c6bbc4dacff93a3cfa2d3d93bfc6c", False),
    ("pilot_protocol", "runs/tno_covariance_classification_pilot_v1/PROTOCOL.json", 13_599, "583260b2d9886f0745f1cffd37bd800d18238dfb791d976d5a539a72d5d87f4f", True),
    ("clone_implementation", "tools/jx_tno_covariance_clone_materialization_v1.py", 27_114, "49f00b856810225a495cbd16e34282db2199e2295d755c2bb2aaba4897b7ba53", False),
    ("clone_test", "tests/test_tno_covariance_clone_materialization_v1.py", 9_607, "c2fdd4eddd85a9a5655e72bb8ad899bd6d3dc5db9c904b4259cee23b5cacbf0d", False),
    ("clone_protocol", "runs/tno_covariance_clones_v1/PROTOCOL.json", 3_358, "88745a6f8286cf1c6f717033973fa972457f8e088eebbfe1d6047b1286a1f21b", True),
    ("clone_artifact", "runs/tno_covariance_clones_v1/CLONES.json", 531_609, "395ce9914df0efe235e2144c24abd157f5aa3ba87beba87fa39891ec42312014", True),
    ("stage0_implementation", "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", 70_064, "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367", False),
    ("stage0_test", "tests/test_tno_native_epoch_cartesian_qualification_v1.py", 15_944, "1225cc19a78aa4ee412c11bf42e931ac8b6881c41845ad9b359c8ff547cb9d0c", False),
    ("stage0_protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184", True),
    ("stage0_states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3", True),
    ("de441_acquisition_implementation", "tools/jx_tno_de441_resolved11_acquisition_v1.py", 37_145, "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00", False),
    ("de441_acquisition_test", "tests/test_tno_de441_resolved11_acquisition_v1.py", 14_556, "220215bae324d8119d03417f44997920b7a4304fb59eae19a375d94a8555d8b3", False),
    ("de441_protocol", "runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", 11_986, "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b", True),
    ("de441_custody", "runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", 141_501, "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3", True),
    ("de441_raw_010", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_010.json", 7_194, "913f48d219d635c6a5e262cc74239ea86bafb802d00732567461bf2b93755354", False),
    ("de441_raw_199", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_199.json", 7_194, "7d183ca3370a483dc739090f02915dc11ce4265a167260d90f9635bf856a03ef", False),
    ("de441_raw_299", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_299.json", 7_194, "4d721c4b6d88aa2ec2de4efdfb3161c1427ad00d789428e8590ea832ed2ede8c", False),
    ("de441_raw_399", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_399.json", 7_194, "0b8802edeaa692a18e747f670280fa4a275300a2342eb56eb093a56c5c32e97a", False),
    ("de441_raw_301", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_301.json", 7_194, "3d919ab6ad860173f393023ea3670d904d9f226de4b994592fa5fbc6484912c3", False),
    ("de441_raw_004", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_004.json", 7_194, "0669acdf63bb9b4cd9824a3fd46d90542f0c722102b2022dcf991f31be9a11bb", False),
    ("de441_raw_005", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_005.json", 7_194, "6e68920a0354cf5a521cb84c16235e4acb635a6a26ad5b225303de45372b4974", False),
    ("de441_raw_006", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_006.json", 7_194, "d912e2eb7d855e53dc2d6cf2dbcaae6f946d505f7f25036c0db88e9123924cc4", False),
    ("de441_raw_007", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_007.json", 7_194, "cc51de7b5da54b95bb1e26fbb88def2498d4b6a989cf55d5b89209fe818b271f", False),
    ("de441_raw_008", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_008.json", 7_194, "a8c45995d1a885c9b0d85ba5c5367d83369beee4f2d78a35fa1bb870fa94d32a", False),
    ("de441_raw_009", "runs/tno_de441_resolved11_native_epochs_v1/raw/body_009.json", 7_194, "0b166885a491f49a5c1c0cc2c2da499bcea734d98662c0072366b22ac40150aa", False),
    ("bridge_implementation", "benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py", 81_161, "5cb0b5380e7eac3e1c3c5932715dcbe43372b03581e65ef981ba491e78087e7e", False),
    ("bridge_test", "tests/test_tno_native_epoch_resolved11_bridge_v1.py", 13_008, "ad52731288e89e1425cc315bcbcdef705144a4bf22fa92f9a2c78576ae4a3497", False),
    ("bridge_protocol", "runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", 11_386, "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459", True),
    ("bridge_receipts", "runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", 5_810_062, "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5", True),
    ("bridge_states", "runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", 168_684, "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f", True),
    ("stage1_v4_implementation", "benchmarks/jx_tno_stage1_dynamics_qualification_v4.py", 436_125, "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2", False),
    ("stage1_v4_test", "tests/test_tno_stage1_dynamics_qualification_v4.py", 89_619, "dc71007054e9afd1cb4d4182de253ac17f495ff835cc6b8677cdf2f36b5c2f8d", False),
    ("stage1_v4_protocol", "runs/tno_stage1_dynamics_qualification_v4/PROTOCOL.json", 74_186, "5aaa8a2db5808c1b343fb25473a50aa2bcbc28059e2d724170cd94063f72c05d", True),
    ("stage1_v4_result", "runs/tno_stage1_dynamics_qualification_v4/RESULT.json", 10_602_296, "bebd9186da0ef8943e05a8c4afc057073d48a8a47e9a4d006dae6e59921d5c42", True),
    ("stage1_v4_arrays", "runs/tno_stage1_dynamics_qualification_v4/ARRAYS.bin", 71_495_424, "e7ce12bed175e51831b4d55e558b3e813fbd555fd18c5833f91203feb2921346", True),
    ("h64_implementation", "benchmarks/jx_tno_stage1_jx_h64_closure_v2.py", 66_627, "04337f7fa63abbcbafd2dcacf6ad26ea5cd73cbccb8562e2a233b0e2c4aaefbd", False),
    ("h64_test", "tests/test_tno_stage1_jx_h64_closure_v2.py", 23_647, "8dc67ff8ad87c3a34a333fd3f862852f7cf7dfe6ba20f09601952101c458ca5d", False),
    ("h64_protocol", "runs/tno_stage1_jx_h64_closure_v2/PROTOCOL.json", 6_965, "f3ce7587145ddbda17442e4a47a97c4973d3d656aee7337aec49771731eaf1a0", True),
    ("h64_result", "runs/tno_stage1_jx_h64_closure_v2/RESULT.json", 5_464_509, "3dd021735f817d4dbfb1a5c92370058347eff3606b47453137b5a3b44d41d54d", True),
    ("jx_package_init", "src/jxplanetx/__init__.py", 6_057, "86bd065b9c16a448b217d960c6dd8d79a7d7ccc79209e7708ff79bc07ce681e0", False),
    ("jx_engine_package_init", "src/jxplanetx/engine/__init__.py", 16_617, "0eb03741a4b37ef6cc8a4a8c982639d4fbbbe1b1caedb376b8c419be3e51c500", False),
    ("jx_engine_api", "src/jxplanetx/engine/api.py", 16_757, "1664c4c69e4b9fa30b983c8390f76aa7a05259e89dd009d57f7099f8ffe088f2", False),
    ("encounter_engine", "src/jxplanetx/cuda_massless_tracer.py", 218_776, "a7b74e8ba5203a3e94c2c466de8f94ba10c3f6c8a0dd9d13009dced5a2d0139a", True),
    ("event_locator", "src/jxplanetx/cuda_tracer_events.py", 67_504, "a3c5f1f9de29f504420aa892bd50f8a947b24d6df21cadf91dc243c7976dad89", False),
    ("backend_contract", "src/jxplanetx/engine/backends.py", 7_367, "c039ae04aab43c0809b200304a47e9d6e63bb7cbea76e0665724c75ab513deba", False),
    ("state_contract", "src/jxplanetx/engine/contracts.py", 16_390, "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d", False),
    ("encounter_engine_test", "tests/test_cuda_massless_tracer.py", 52_536, "0e08ee0a6054a1ff180d2acf72193102ef283cbd39587ca16acd14b9c89966a0", False),
    ("project_manifest", "pyproject.toml", 5_865, "8797a0fab9693872e36a6259e84bd82ca5104b4ffe7d0191d248252232b33e7a", False),
)

CONTENT_BINDINGS = {
    "pilot_protocol": "af27a4c5014b83d6b7c0241edba807e10ae635aaf641e6c5ad1eaf1fc1176db0",
    "clone_protocol": "939e6c3f8e3773f2af4675bab95369e702461ae00edf311b71c02f16785ab798",
    "clone_artifact": "9f1c0df03a2001f6249aafd423c4c9feb8bde9a75467416d35b81b88bb2d59d8",
    "stage0_protocol": "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e",
    "stage0_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
    "de441_protocol": "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e",
    "de441_custody": "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3",
    "bridge_protocol": "a7dbc361ffa4ccaf138f3c7afce269cad8bbb505b77930bcea9d04f5167b3c15",
    "bridge_receipts": "7e5f3ef0a731647a9adaebedae65d6f703e44b8a923eb624d6dc872d935b8058",
    "bridge_states": "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13",
    "stage1_v4_protocol": "3ea78828498b71e688ad12e6210f3c532689265a6ba202a4d5a6bae5d08f3647",
    "stage1_v4_result": "df4e089aad8d436e99871c943343ea8b631ca714c946edb388387307c75d87d2",
    "h64_protocol": "16712543d49432d110bf96e6c8e5be0436d922fdeaef74068606447c88b1a14e",
    "h64_result": "ea0627dd3c48c444cb4a0d9d5c11dedd41b3c3be886daa06208c9d4930896836",
}

BLOCKERS = (
    "PINNED_CUDA_MASSLESS_TRACER_IS_EXPLICITLY_MODEL_OUTPUT_SCREENING_ONLY_NOT_A_QUALIFIED_TNO_ENGINE",
    "NO_REVIEWED_DE441_RESOLVED11_PLUS_TNO_ADAPTER_TO_CUDA_MASSLESS_TRACER_STATE_EXISTS",
    "PUBLIC_ENGINE_RESULT_DOES_NOT_EXPOSE_THE_FULL_FIXED_STEP_MASSIVE_SOURCE_NODE_STREAM_REQUIRED_TO_REDERIVE_ALL_NODE_INVARIANT_GATES",
    "NO_REVIEWED_ANNUAL_BINARY_EVIDENCE_WRITER_AND_OFFLINE_REPLAYER_EXISTS_FOR_THIS_ENGINE",
    "NO_REVIEWED_EXACT_V4_DE441_IAS15_ARRAY_EXTRACTOR_AND_STAGE1_GATE_COMPARATOR_EXISTS",
    "NO_REVIEWED_ROUTE_REMOVAL_LEDGER_MATERIALIZER_WITH_STREAMING_CAP_ENFORCEMENT_EXISTS",
    "NO_REVIEWED_MIDPOINT_FRESH_PROCESS_RESTART_ORACLE_EXISTS_FOR_THIS_ENGINE",
    "NO_REVIEWED_FRESH_PROCESS_NUMPY_SCIPY_CUPY_CUDA_GPU_AND_JX_RUNTIME_CUSTODY_BOUNDARY_EXISTS_FOR_STAGE15",
    "RESOLVED11_H64_H128_ONE_THOUSAND_YEAR_QUALIFICATION_HAS_NOT_BEEN_EXECUTED",
)


class Stage15ProtocolError(ValueError):
    """A bound input or preparation-only contract failed closed."""


def _canonical_json(value: object) -> bytes:
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256_bytes(DOMAIN + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise Stage15ProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Stage15ProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage15ProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise Stage15ProtocolError("JSON root must be an object")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage15ProtocolError(f"required regular file cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise Stage15ProtocolError(f"file is linked or nonregular: {path}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as exc:
        raise Stage15ProtocolError(f"file disappeared while read: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if identity_before != identity_after or identity_before != identity_path:
        raise Stage15ProtocolError(f"file changed while read: {path}")
    if len(payload) != before.st_size:
        raise Stage15ProtocolError(f"file length changed while read: {path}")
    return payload


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if set(left) != set(right):
            return False
        return all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise Stage15ProtocolError(f"{label} differs from the frozen contract")


def _load_canonical_payload(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise Stage15ProtocolError(f"retained JSON is not canonical: {label}")
    return value


def _verify_bound_inputs(
    *, retain_payloads: bool = True
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    records: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for role, relative_path, expected_size, expected_sha256, retain in FILE_BINDINGS:
        payload = _read_regular_bytes(PROJECT_ROOT / relative_path)
        if len(payload) != expected_size:
            raise Stage15ProtocolError(f"bound file size differs: {relative_path}")
        if _sha256_bytes(payload) != expected_sha256:
            raise Stage15ProtocolError(f"bound file digest differs: {relative_path}")
        records.append(
            {
                "role": role,
                "relative_path": relative_path,
                "bytes": expected_size,
                "file_sha256": expected_sha256,
            }
        )
        if retain and retain_payloads:
            payloads[role] = payload
    return records, payloads


def _json_inputs(payloads: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for role, expected_content in CONTENT_BINDINGS.items():
        value = _load_canonical_payload(payloads[role], role)
        _require_exact(value.get("content_sha256"), expected_content, f"{role} content digest")
        values[role] = value
    return values


def _validate_selection_and_custody(values: Mapping[str, Mapping[str, Any]]) -> None:
    pilot = values["pilot_protocol"]
    _require_exact(pilot.get("execution_authorized"), False, "pilot execution claim")
    _require_exact(pilot.get("outcomes_generated"), False, "pilot outcome claim")

    clones = values["clone_artifact"]
    _require_exact(clones.get("object_count"), 12, "clone object count")
    _require_exact(clones.get("clones_per_object"), 128, "clone count per object")
    _require_exact(clones.get("total_clone_count"), 1_536, "total clone count")
    _require_exact(
        clones.get("binary_payload_sha256"),
        "3f296d6bf9489598674e152fc7511641023c9d513fd4f6ca6d33ab6ec7293278",
        "clone binary payload digest",
    )

    stage0 = values["stage0_states"]
    _require_exact(stage0.get("object_count"), 12, "Stage-0 object count")
    _require_exact(stage0.get("nominal_state_count"), 12, "Stage-0 nominal count")
    _require_exact(stage0.get("clone_state_count"), 1_536, "Stage-0 clone count")
    _require_exact(stage0.get("total_state_count"), 1_548, "Stage-0 state count")
    _require_exact(len(stage0.get("states", [])), 1_548, "Stage-0 state roster length")
    _require_exact(
        stage0.get("binary_payload_sha256"),
        "334d3fb579e3be752e44cee91680a0c73bd4ac2cb5ab2bff108d66c389118b83",
        "Stage-0 binary payload digest",
    )

    custody = values["de441_custody"]
    _require_exact(
        custody.get("raw_roster_sha256"),
        "11274439a0d1f5d318a806dbf1d94eed8e726bce2acb68a9880b505a3d216024",
        "DE441 raw response roster digest",
    )
    _require_exact(
        custody.get("state_roster_sha256"),
        "f57f04d68909312d5fe53918eb8c7527310282034d678690b2f99619cafa3cd2",
        "DE441 state roster digest",
    )
    state_records = custody.get("state_records")
    if type(state_records) is not list:
        raise Stage15ProtocolError("DE441 state_records must be a list")
    _require_exact(len(state_records), 132, "DE441 state record count")
    expected_naif = [row[0] for row in MASSIVE_ROSTER]
    requests = custody.get("requests")
    if type(requests) is not list:
        raise Stage15ProtocolError("DE441 requests must be a list")
    binding_by_role = {binding[0]: binding for binding in FILE_BINDINGS}
    expected_raw = [
        (
            naif_id,
            f"raw/body_{naif_id:03d}.json",
            binding_by_role[f"de441_raw_{naif_id:03d}"][3],
        )
        for naif_id, _ in MASSIVE_ROSTER
    ]
    _require_exact(
        [
            (
                row.get("naif_id"),
                row.get("retained_relative_path"),
                row.get("raw_sha256"),
            )
            for row in requests
        ],
        expected_raw,
        "DE441 raw request roster",
    )
    for epoch_order, object_id in enumerate(OBJECT_IDS):
        rows = state_records[epoch_order * 11 : (epoch_order + 1) * 11]
        _require_exact([row.get("object_id") for row in rows], [object_id] * 11, "DE441 epoch object roster")
        _require_exact([row.get("naif_id") for row in rows], expected_naif, "DE441 body roster")
        _require_exact([row.get("epoch_order") for row in rows], [epoch_order] * 11, "DE441 epoch ordering")

    bridge = values["bridge_states"]
    _require_exact(bridge.get("batch_count"), 12, "bridge batch count")
    _require_exact(bridge.get("massive_state_rows_total"), 132, "bridge massive rows")
    _require_exact(bridge.get("tno_state_rows_total"), 108, "bridge TNO rows")
    _require_exact(bridge.get("source_stage0_state_count"), 1_548, "bridge Stage-0 source count")
    batches = bridge.get("batches")
    if type(batches) is not list:
        raise Stage15ProtocolError("bridge batches must be a list")
    _require_exact([row.get("object_id") for row in batches], list(OBJECT_IDS), "bridge object order")
    for row in batches:
        _require_exact(row.get("massive_body_count"), 11, "bridge massive count")
        _require_exact(row.get("tno_state_count"), 9, "bridge tracer count")
        tnos = row.get("tnos")
        if type(tnos) is not list:
            raise Stage15ProtocolError("bridge TNO roster must be a list")
        _require_exact(
            [entry.get("clone_index") for entry in tnos],
            [None, *HOLDOUT_CLONE_INDICES],
            "bridge holdout clone order",
        )


def _validate_stage1_lineage(
    values: Mapping[str, Mapping[str, Any]], arrays_payload: bytes
) -> list[dict[str, object]]:
    protocol = values["stage1_v4_protocol"]
    result = values["stage1_v4_result"]
    _require_exact(protocol.get("status"), "PREDECLARED_V4_READY_FOR_EXPLICIT_QUALIFICATION_EXECUTION", "Stage-1 v4 protocol status")
    _require_exact(result.get("status"), "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS", "Stage-1 v4 result status")
    _require_exact(result.get("batch_lane_count"), 96, "Stage-1 v4 lane count")
    _require_exact(result.get("dynamical_classification_generated"), False, "Stage-1 classification claim")
    _require_exact(result.get("resonance_classification_generated"), False, "Stage-1 resonance claim")
    _require_exact(result.get("science_result"), False, "Stage-1 science claim")

    selection = protocol.get("selection")
    if type(selection) is not dict:
        raise Stage15ProtocolError("Stage-1 selection is absent")
    _require_exact(selection.get("native_epoch_batch_count"), 12, "Stage-1 batch count")
    _require_exact(selection.get("tracers_per_batch"), 9, "Stage-1 tracer count per batch")
    _require_exact(selection.get("exact_tracer_count_per_ephemeris"), 108, "Stage-1 ephemeris tracer count")
    _require_exact(selection.get("holdout_clone_indices"), list(HOLDOUT_CLONE_INDICES), "Stage-1 holdout indices")
    _require_exact(
        [row.get("object_id") for row in selection.get("batches", [])],
        list(OBJECT_IDS),
        "Stage-1 object order",
    )

    arrays = result.get("cartesian_arrays")
    if type(arrays) is not dict:
        raise Stage15ProtocolError("Stage-1 Cartesian array receipt is absent")
    _require_exact(
        arrays.get("file_sha256"),
        "e7ce12bed175e51831b4d55e558b3e813fbd555fd18c5833f91203feb2921346",
        "Stage-1 array digest receipt",
    )
    _require_exact(arrays.get("byte_length"), len(arrays_payload), "Stage-1 array length receipt")
    _require_exact(arrays.get("batch_lane_count"), 96, "Stage-1 array lane count")
    _require_exact(
        arrays.get("all_vectors_origin"),
        "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER",
        "Stage-1 array origin",
    )

    gate_map = result.get("qualification_gates", {}).get("gates", {})
    for gate in (
        "ias15_accepted_step_source_tracer_guard",
        "ias15_max_relative_energy_drift",
        "ias15_max_relative_angular_momentum_drift",
    ):
        _require_exact(gate_map.get(gate, {}).get("passed"), True, f"retained IAS15 gate {gate}")

    lanes = result.get("batch_lanes")
    if type(lanes) is not list:
        raise Stage15ProtocolError("Stage-1 batch lanes are absent")
    oracle_rows: list[dict[str, object]] = []
    selected = [
        row
        for row in lanes
        if type(row) is dict
        and row.get("ephemeris") == "DE441"
        and type(row.get("lane")) is dict
        and row["lane"].get("lane_id") == "rebound511_ias15_resolved11"
    ]
    _require_exact(len(selected), 12, "DE441 resolved11 IAS15 oracle lane count")
    for batch_order, (expected_object, row) in enumerate(zip(OBJECT_IDS, selected, strict=True)):
        _require_exact(row.get("batch_order"), batch_order, "oracle batch order")
        _require_exact(row.get("object_id"), expected_object, "oracle object order")
        lane = row["lane"]
        _require_exact(lane.get("engine"), "REBOUND_5.1.1_IAS15", "oracle engine")
        _require_exact(lane.get("model"), "RESOLVED_11_BODY", "oracle model")
        locator = lane.get("cartesian_array")
        if type(locator) is not dict:
            raise Stage15ProtocolError("oracle Cartesian locator is absent")
        for key, expected in (
            ("checkpoint_count", 1_001),
            ("massive_count", 11),
            ("tracer_count", 9),
            ("components", 3),
            ("record_bytes", 960),
            ("byte_length", 960_960),
        ):
            _require_exact(locator.get(key), expected, f"oracle locator {key}")
        offset = locator.get("byte_offset")
        length = locator.get("byte_length")
        if type(offset) is not int or type(length) is not int:
            raise Stage15ProtocolError("oracle array offsets must be exact integers")
        if offset < 0 or length <= 0 or offset + length > len(arrays_payload):
            raise Stage15ProtocolError("oracle array locator is outside the retained payload")
        lane_payload = arrays_payload[offset : offset + length]
        _require_exact(
            _sha256_bytes(lane_payload),
            locator.get("lane_payload_sha256"),
            "oracle lane payload digest",
        )
        oracle_rows.append(
            {
                "batch_order": batch_order,
                "object_id": expected_object,
                "lane_id": "rebound511_ias15_resolved11",
                "byte_offset": offset,
                "byte_length": length,
                "lane_payload_sha256": locator["lane_payload_sha256"],
                "checkpoint_count": 1_001,
                "record_bytes": 960,
                "massive_count": 11,
                "tracer_count": 9,
            }
        )

    h64 = values["h64_result"]
    _require_exact(h64.get("status"), "H64_REFINEMENT_CLOSURE_PASS", "h64 closure status")
    _require_exact(h64.get("predecessor_v4_result_content_sha256"), CONTENT_BINDINGS["stage1_v4_result"], "h64 predecessor binding")
    _require_exact(h64.get("dynamical_classification_generated"), False, "h64 classification claim")
    _require_exact(h64.get("resonance_classification_generated"), False, "h64 resonance claim")
    _require_exact(h64.get("science_result"), False, "h64 science claim")
    _require_exact(h64.get("claims", {}).get("retrospective_outcome_seen"), True, "h64 retrospective status")
    _require_exact(h64.get("qualification_gates", {}).get("all_predeclared_gates_passed"), True, "h64 closure gate status")
    return oracle_rows


def _validate_engine_boundary(payload: bytes) -> None:
    required_tokens = (
        b'SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"',
        b'PROJECT_SCOPE = "SCREENING_ONLY"',
        b'METHOD_ID = "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2"',
        b'class FixedStepCUDAMasslessTracerSpec:',
        b'def advance_cuda_massless_tracers(',
        b'class TracerEncounterRouteRecord:',
        b'class TracerRemovalEventRecord:',
        b'class CUDAMasslessTracerResult:',
    )
    for token in required_tokens:
        if token not in payload:
            raise Stage15ProtocolError(
                f"pinned encounter engine semantic token is absent: {token!r}"
            )


def _validate_inputs() -> dict[str, object]:
    files, payloads = _verify_bound_inputs()
    values = _json_inputs(payloads)
    _validate_selection_and_custody(values)
    oracle_rows = _validate_stage1_lineage(values, payloads["stage1_v4_arrays"])
    _validate_engine_boundary(payloads["encounter_engine"])
    # Close the ordinary point-observation window for every pinned input after
    # parsing and semantic checks.  This is not concurrent-mutation immunity.
    files_after, _ = _verify_bound_inputs(retain_payloads=False)
    _require_exact(files_after, files, "input bindings after semantic validation")
    return {
        "files": files,
        "content_sha256": dict(CONTENT_BINDINGS),
        "binary_payload_sha256": {
            "clone_artifact": "3f296d6bf9489598674e152fc7511641023c9d513fd4f6ca6d33ab6ec7293278",
            "stage0_states": "334d3fb579e3be752e44cee91680a0c73bd4ac2cb5ab2bff108d66c389118b83",
            "bridge_states": "a0a7c85edec30b2f494c9db7053abbd807cbf24dd94f0ca2645e490a2700e9bc",
        },
        "stage1_v4_de441_resolved11_ias15_oracle_lanes": oracle_rows,
        "stage1_v4_ias15_trace_not_consumed_by_stage15_comparison": {
            "relative_path": "runs/tno_stage1_dynamics_qualification_v4/IAS15_ACCEPTED_STEPS.bin",
            "bytes": 25_665_735_616,
            "file_sha256": "c5678695f709478cde083aaec165c6797b59bc55bfe6ef243405384ed228bd32",
            "content_sha256": "7bc9c8d1abcb3903065777d45956e66feaba4f591ceef11d699c5104a18954c5",
            "reason_not_rehashed": "ANNUAL_REFERENCE_IS_FULLY_LOCATED_IN_BOUND_ARRAYS_BIN;_25_7_GB_ACCEPTED_STEP_TRACE_IS_LINEAGE_ONLY",
        },
        "h64_lineage_payloads_bound_by_retained_h64_result": [
            {"relative_path": "ARRAYS.bin", "bytes": 16_144_128, "file_sha256": "9da897bb9a386db8429cd607147f241a3a5a1113f716a2f05d2c2d07c876ad41"},
            {"relative_path": "JX_MIDPOINTS.bin", "bytes": 10_368, "file_sha256": "92d80bab7b4c39c70f0eada943ec625c8665dc0e1414f48693a3fdb0587e3f04"},
            {"relative_path": "JX_RESTART_STREAMS.bin", "bytes": 865_728, "file_sha256": "e18d17a8b4b51cf840110bb01730476f0535295acc875a2e994761fccb50c96a"},
            {"relative_path": "JX_SOURCE_NODES.bin", "bytes": 368_645_800, "file_sha256": "fd9b51e18aa6bb5fac095799955ecc6b3d6afeb2253b1000e2def41ccf376dbd"},
        ],
    }


def _exact_gate_budgets() -> dict[str, object]:
    return {
        "preliminary_scattering_diagnostic": {"numerator": 3, "denominator": 80, "binary64_hex": float(3 / 80).hex()},
        "numerical_richardson_semimajor_axis_tail": {"numerator": 3, "denominator": 8_000, "binary64_hex": float(3 / 8_000).hex()},
        "fine_vs_ias15_semimajor_axis": {"numerator": 3, "denominator": 4_000, "binary64_hex": float(3 / 4_000).hex()},
        "fine_vs_ias15_scattering_diagnostic": {"numerator": 3, "denominator": 4_000, "binary64_hex": float(3 / 4_000).hex()},
        "cartesian_position_AU": {"numerator": 1, "denominator": 500, "binary64_hex": float(1 / 500).hex()},
        "cartesian_velocity_AU_per_julian_year": {"numerator": 1, "denominator": 50_000, "binary64_hex": float(1 / 50_000).hex()},
        "jx_resolved11_max_relative_energy_drift": {"numerator": 1, "denominator": 100_000_000, "binary64_hex": float(1 / 100_000_000).hex()},
        "jx_resolved11_max_relative_angular_momentum_drift": {"numerator": 1, "denominator": 10_000_000_000, "binary64_hex": float(1 / 10_000_000_000).hex()},
        "energy_contraction_floor": {"numerator": 1, "denominator": 2**46, "binary64_hex": float(2**-46).hex()},
    }


def _planned_lanes() -> list[dict[str, object]]:
    return [
        {
            "lane_order": order,
            "lane_id": lane_id,
            "step_julian_years_hex": step.hex(),
            "steps_per_julian_year": steps,
            "total_fixed_steps": 1_000 * steps,
            "annual_segment_step_count": steps,
            "integrator": "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2_WITH_SCIPY_DOP853_ENCOUNTER_ROUTE",
            "model": "RESOLVED_11_BODY",
        }
        for order, (lane_id, step, steps) in enumerate(
            (
                ("jx_resolved11_encounter_h64", 1.0 / 64.0, 64),
                ("jx_resolved11_encounter_h128", 1.0 / 128.0, 128),
            )
        )
    ]


def build_protocol() -> dict[str, object]:
    inputs = _validate_inputs()
    source_payload = _read_regular_bytes(SCRIPT_PATH)
    test_payload = _read_regular_bytes(TEST_PATH)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PREPARED_FAIL_CLOSED_EXECUTION_BLOCKED",
        "claim_ceiling": CLAIM_CEILING,
        "scope": {
            "purpose": "QUALIFY_THE_ENCOUNTER_CAPABLE_RESOLVED11_JX_PATH_BEFORE_ANY_LONG_HORIZON_TNO_CLASSIFICATION",
            "ephemeris": "DE441",
            "native_epoch_batch_count": 12,
            "object_ids_in_order": list(OBJECT_IDS),
            "tracers_per_batch": 9,
            "total_tracer_count": 108,
            "tracer_order": "NOMINAL_THEN_COVARIANCE_CLONES_IN_FROZEN_HOLDOUT_INDEX_ORDER",
            "holdout_clone_indices": list(HOLDOUT_CLONE_INDICES),
            "tracer_indices_required_exactly": list(TRACER_INDICES),
            "horizon_julian_years": 1_000,
            "checkpoint_indices_inclusive": [0, 1_000],
            "checkpoint_rule": "EVERY_INTEGER_JULIAN_YEAR_INCLUDING_ZERO_AND_1000",
            "checkpoint_count_per_lane": 1_001,
            "lane_count_per_batch": 2,
            "total_lane_count": 24,
            "lanes": _planned_lanes(),
            "qualification_only": True,
            "classification_or_science_output_permitted": False,
        },
        "inputs": inputs,
        "physics": {
            "force_law": "NEWTONIAN_POINT_MASS",
            "massive_roster": [
                {"body_order": order, "naif_id": naif_id, "body_id": body_id}
                for order, (naif_id, body_id) in enumerate(MASSIVE_ROSTER)
            ],
            "massive_body_count": 11,
            "gm_policy": "EXACT_COMMON_DE440_BRIDGE_BINARY64_GMS_AS_USED_BY_STAGE1_V4_FOR_DE441",
            "tno_mass_binary64_hex": "0x0.0p+0",
            "tno_backreaction": False,
            "tracer_tracer_interactions": False,
            "frame": "ICRF_J2000_INERTIAL",
            "length_unit": "AU",
            "time_unit": "JULIAN_YEAR_EXACTLY_365_25_TDB_DAYS",
            "time_scale_token": "TDB",
            "epoch_origin_semantics": "NATIVE_COVARIANCE_EPOCH_JD_TDB_MINUS_2451545_0_DIVIDED_BY_365_25",
            "initial_state_rule": "RECONSTRUCT_DE441_SSB_RESOLVED11_AND_TNO_HELIOCENTRIC_STATE_THEN_COMMON_GM_RECENTER; REQUIRE_EXACT_FORWARD_CROSSCHECK_TO_RETAINED_STAGE1_V4_DE441_IAS15_RESOLVED11_YEAR_ZERO_ARRAY_RECORD",
            "annual_output_origin": "INSTANTANEOUS_RESOLVED11_COMMON_GM_BARYCENTER",
            "source_encounter_radii_AU": ["0x1.0000000000000p+0"] * 11,
            "source_collision_radii_AU": ["0x0.0p+0"] * 11,
            "relative_acceleration_bound_rule": "ALL_11_ENTRIES_EQUAL_NEXTAFTER_TOWARD_POSITIVE_INFINITY_OF_2_TIMES_SUM_COMMON_GM_AU3_PER_YEAR2_DIVIDED_BY_1_AU_SQUARED",
            "primary_outer_boundary_AU_hex": "0x1.0000000000000p+12",
            "removal_policy": "ANY_SOURCE_RADIUS_OR_OUTER_BOUNDARY_REMOVAL_FAILS_THE_WHOLE_QUALIFICATION;_NO_TRACER_MAY_BE_DROPPED",
        },
        "engine_adapter_contract": {
            "public_module": "jxplanetx.cuda_massless_tracer",
            "public_entrypoint": "advance_cuda_massless_tracers",
            "method_id": "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2",
            "current_scientific_classification": "MODEL_OUTPUT",
            "current_project_scope": "SCREENING_ONLY",
            "backend": {"backend_id": "cupy", "device": "cuda:0", "float64": True, "fallback": False, "fast_math": False},
            "annual_segmentation": "1000_SUCCESSIVE_PUBLIC_CALLS_EACH_WITH_STEP_COUNT_EQUAL_TO_STEPS_PER_JULIAN_YEAR_AND_CARRIED_CUMULATIVE_STATE",
            "source_path_chunk_steps": 65_536,
            "source_node_evidence_requirement": "RETAIN_EVERY_FIXED_KDK_SOURCE_NODE_FOR_OFFLINE_ALL_NODE_ENERGY_AND_ANGULAR_MOMENTUM_REDERIVATION",
            "public_result_exposes_full_source_node_stream": False,
            "adapter_implemented": False,
            "execution_authorized": False,
        },
        "encounter_route_contract": {
            "method": "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2",
            "relative_tolerance_hex": float(1.0e-12).hex(),
            "absolute_tolerance_hex": float(1.0e-14).hex(),
            "maximum_substep_fraction_hex": float(0.125).hex(),
            "maximum_primary_dynamical_phase_hex": float(0.05).hex(),
            "event_time_tolerance_fraction_hex": float(1.0e-12).hex(),
            "maximum_routes_per_tracer": 16_384,
            "maximum_total_routes": 100_000,
            "maximum_total_removal_records": 108,
            "maximum_total_evidence_records": 100_108,
            "maximum_route_or_removal_json_line_bytes": 65_536,
            "maximum_route_jsonl_bytes": 6_560_677_888,
            "resource_cap_policy": "CHECK_BEFORE_EVERY_APPEND_AND_FAIL_BEFORE_BYTES_OR_STATE_BEYOND_THE_FROZEN_CAP",
            "retained_record_encoding": "CANONICAL_JSON_LINES_ONE_EXACT_ASDICT_RECORD_PER_NEW_ROUTE_OR_REMOVAL_WITH_LANE_AND_ANNUAL_SEGMENT_PREFIX",
            "retained_fields": [
                "tracer_index",
                "source_index_and_id",
                "global_step_and_reason",
                "source_segment_sha256_epoch_endpoints_velocities_gms_and_step",
                "tracer_endpoint_positions_and_velocities",
                "all_route_tolerances",
                "cpu_accepted_and_rejected_steps",
                "evaluated_and_committed_minimum_distance",
                "unbound_outcome_and_committed_fraction",
                "accepted_and_committed_primary_phase",
                "complete_removal_event_record_if_any",
            ],
            "offline_replay": "RECONSTRUCT_EVERY_SOURCE_SEGMENT_HASH_AND_RECORD_INVARIANT;_VERIFY_CUMULATIVE_COUNTS_PREFIX_ORDER_AND_NO_MISSING_DUPLICATE_OR_RELABELED_RECORD",
            "route_is_not_a_physical_encounter_claim": True,
            "materializer_implemented": False,
        },
        "restart_contract": {
            "midpoint_year": 500,
            "retain_all_lane_midpoints": True,
            "midpoint_lane_count": 24,
            "selected_fresh_process_cases": [
                {"restart_order": 0, "lane_id": "jx_resolved11_encounter_h64", "object_id": OBJECT_IDS[0]},
                {"restart_order": 1, "lane_id": "jx_resolved11_encounter_h64", "object_id": OBJECT_IDS[-1]},
                {"restart_order": 2, "lane_id": "jx_resolved11_encounter_h128", "object_id": OBJECT_IDS[0]},
                {"restart_order": 3, "lane_id": "jx_resolved11_encounter_h128", "object_id": OBJECT_IDS[-1]},
            ],
            "acceptance": "BITWISE_IDENTICAL_YEARS_500_THROUGH_1000_SOURCE_AND_TRACER_STATES_FINAL_STATE_ROUTE_COUNTS_ROUTE_RECORD_SUFFIX_REMOVAL_LEDGER_AND_RUNTIME_IDENTITY_PROJECTION",
            "fresh_process_required": True,
            "checkpoint_schema_required": "jxplanetx.cuda-massless-tracer-checkpoint.v4",
            "checkpoint_roundtrip_validate_before_continuation": True,
            "restart_oracle_implemented": False,
        },
        "runtime_custody_contract": {
            "fresh_project_venv_child_required": True,
            "predeclared_distribution_versions": {
                "python": "3.14.4",
                "numpy": "2.5.3",
                "scipy": "1.17.0",
                "cupy_distribution": "cupy-cuda13x",
                "cupy": "14.2.0",
            },
            "reject_preloaded_prefixes": ["jxplanetx", "numpy", "scipy", "cupy", "cupyx"],
            "python_bytecode_policy": "FRESH_VERIFIED_ABSENT_PYCACHE_PREFIX_AND_DONT_WRITE_BYTECODE_FOR_PROCESS_LIFETIME",
            "cupy_cache_policy": "FRESH_VERIFIED_ABSENT_EXTERNAL_KERNEL_CACHE_AND_IN_MEMORY_CACHE_ONLY",
            "bind_before_and_reread_after": [
                "PROJECT_VENV_LAUNCHER_CHAIN_PYVENV_CFG_AND_RESOLVED_PYTHON_BINARY",
                "STAGE15_IMPLEMENTATION_AND_ALL_PINNED_JX_SOURCES",
                "NUMPY_DISTRIBUTION_RECORD_METADATA_MODULE_CORE_AND_LOADED_NATIVE_OBJECTS",
                "SCIPY_DISTRIBUTION_RECORD_METADATA_INTEGRATE_DOP853_MODULES_AND_LOADED_NATIVE_OBJECTS",
                "CUPY_DISTRIBUTION_RECORD_METADATA_MODULES_CORE_AND_LOADED_NATIVE_OBJECTS",
                "CUDA_RUNTIME_DRIVER_DEVICE_PCI_UUID_COMPUTE_CAPABILITY_AND_GPU_NAME",
            ],
            "registry_rule": "ONLY_EXACT_MODULETYPE_ENTRIES_RETAINED_AFTER_RAW_PREFIX_KEY_PRELOAD_REJECTION",
            "trusted_process_ceiling": "POINT_OBSERVATIONS_IN_A_FRESH_TRUSTED_PROCESS_WITH_NO_CONCURRENT_MUTATION_NOT_SOFTWARE_AUTHENTICATION",
            "runtime_boundary_implemented": False,
        },
        "retained_evidence_contract": {
            "planned_exact_roster": [
                "PROTOCOL.json",
                "RESULT.json",
                "ARRAYS.bin",
                "SOURCE_NODES.bin",
                "ROUTES.jsonl",
                "MIDPOINTS.bin",
                "RESTART_STREAMS.bin",
                "RUNTIME.json",
            ],
            "arrays": {
                "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
                "order": "BATCH_THEN_H64_H128_THEN_YEAR_THEN_11_MASSIVE_POS_VEL_THEN_9_TRACER_POS_VEL",
                "exact_bytes": 23_063_040,
            },
            "source_nodes": {
                "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
                "order": "BATCH_THEN_H64_H128_THEN_NODE_THEN_11_SOURCE_POSITIONS_THEN_VELOCITIES",
                "h64_exact_bytes": 405_510_336,
                "h128_exact_bytes": 811_014_336,
                "combined_exact_bytes": 1_216_524_672,
            },
            "midpoints_exact_bytes": 10_368,
            "restart_streams_exact_bytes": 1_731_456,
            "route_jsonl_maximum_record_count": 100_108,
            "large_artifacts_must_be_streamed_with_one_held_no_follow_descriptor": True,
            "result_committed_last_and_exact_roster_required": True,
            "multi_file_filesystem_atomicity_claimed": False,
        },
        "comparison_oracle": {
            "source": "RETAINED_STAGE1_V4_DE441_REBOUND_5_1_1_IAS15_RESOLVED11_ANNUAL_CARTESIAN_LANES",
            "oracle_lane_count": 12,
            "checkpoint_count_per_lane": 1_001,
            "same_initial_state_required": True,
            "same_resolved11_gm_roster_required": True,
            "same_instantaneous_common_gm_barycenter_output_convention_required": True,
            "oracle_result_overall_status_caution": "V4_OVERALL_GATES_DID_NOT_ALL_PASS_BUT_ITS_RESOLVED11_IAS15_GUARD_AND_INVARIANT_GATES_PASSED",
            "h64_lineage_caution": "RETAINED_H64_CLOSURE_IS_POST_OUTCOME_REDUCED5_EVIDENCE_AND_IS_NOT_STAGE15_RESOLVED11_ACCEPTANCE_EVIDENCE",
        },
        "observables": {
            "retained_annual_cartesian_states": "ALL_11_MASSIVE_AND_ALL_9_TRACER_POSITIONS_AU_AND_VELOCITIES_AU_PER_JULIAN_YEAR",
            "barycentric_element_origin": "INSTANTANEOUS_RESOLVED11_COMMON_GM_BARYCENTER",
            "barycentric_element_mu": "SUM_OF_ALL_11_COMMON_BINARY64_GMS",
            "derived_elliptic_elements": ["a", "e", "q", "i", "node", "peri", "M"],
            "semimajor_axis_unit": "AU",
            "angle_unit": "RADIAN",
            "preliminary_scattering_metric": "MAX_YEAR_0_THROUGH_1000_ABS_A_YEAR_MINUS_A0_DIVIDED_BY_A0",
            "preliminary_scattering_threshold": "3/80",
            "threshold_comparison": "STRICTLY_GREATER_THAN",
            "preliminary_scattering_bit_is_not_a_dynamical_class": True,
            "resonant_angles_computed": False,
            "classification_generated": False,
        },
        "exact_gate_budgets": _exact_gate_budgets(),
        "predeclared_gates": {
            "structural": [
                "EXACT_24_LANE_12_BATCH_9_TRACER_1001_CHECKPOINT_ROSTER",
                "NO_NONFINITE_STATE_NO_REMOVAL_NO_DROP_NO_INDEX_OR_ORDER_CHANGE",
                "EXACT_YEAR_ZERO_CROSS_BINDING_TO_STAGE0_BRIDGE_DE441_AND_V4_ORACLE",
                "EVERY_ROUTE_AND_REMOVAL_RECORD_RETAINED_AND_OFFLINE_REPLAYED",
                "EVERY_FIXED_SOURCE_NODE_RETAINED_AND_ALL_NODE_INVARIANTS_REDERIVED",
                "ALL_24_MIDPOINTS_ROUNDTRIP_AND_FOUR_FRESH_PROCESS_SUFFIX_REPLAYS_PASS",
                "EXACT_RUNTIME_AND_SOURCE_RECEIPTS_PASS_BEFORE_AND_AFTER",
            ],
            "fine_h128_vs_ias15_per_tracer": {
                "semimajor_axis": "EXACT_MAX_ABS_A_DIFFERENCE_DIVIDED_BY_MAX_INITIAL_A_LE_3_OVER_4000",
                "position": "EXACT_MAX_SQUARED_CARTESIAN_DIFFERENCE_LE_SQUARE_OF_1_OVER_500_AU",
                "velocity": "EXACT_MAX_SQUARED_CARTESIAN_DIFFERENCE_LE_SQUARE_OF_1_OVER_50000_AU_PER_JULIAN_YEAR",
                "scattering_diagnostic": "EXACT_ABS_DIFFERENCE_LE_3_OVER_4000_NOT_A_CLASS",
            },
            "h64_h128_richardson_per_tracer_and_source": {
                "semimajor_axis": "EXACT_MAX_ABS_COARSE_FINE_A_DIFFERENCE_DIVIDED_BY_FIXED_MAX_OF_BOTH_INITIAL_A_VALUES_THEN_DIVIDED_BY_3_LE_3_OVER_8000",
                "position": "EXACT_COARSE_FINE_MAX_SQUARED_DIVIDED_BY_9_LE_SQUARE_OF_1_OVER_500_AU",
                "velocity": "EXACT_COARSE_FINE_MAX_SQUARED_DIVIDED_BY_9_LE_SQUARE_OF_1_OVER_50000_AU_PER_JULIAN_YEAR",
            },
            "contraction_per_tracer_and_source": "FOR_EACH_A_R_V_FINE_ERROR_LE_EXACT_PROPAGATED_64_ULP_REFERENCE_FLOOR_OR_COARSE_ERROR_AT_LEAST_3_TIMES_FINE;_VECTOR_PREDICATE_USES_EXACT_SQUARES_AND_FACTOR_9",
            "massive_source_fine_vs_ias15": "PER_SOURCE_POSITION_LE_1_OVER_500_AU_AND_VELOCITY_LE_1_OVER_50000_AU_PER_JULIAN_YEAR_WITH_RICHARDSON_AND_CONTRACTION",
            "resolved11_invariants": "H128_ALL_NODE_ENERGY_DRIFT_LE_1E_MINUS_8_AND_ANGULAR_MOMENTUM_DRIFT_LE_1E_MINUS_10;_H128_ENERGY_LE_MAX_H64_DIVIDED_BY_3,2_POW_MINUS_46",
            "decision_arithmetic": "EXACT_FRACTIONS_FROM_BINARY64_FOR_SEMIMAJOR_SCATTERING_AND_ENERGY;_EXACT_SQUARED_FRACTIONS_FOR_CARTESIAN;_OUTWARD_BINARY64_ONLY_FOR_DISPLAY",
            "all_individual_predicates_must_pass": True,
            "preliminary_scattering_bit_used_for_acceptance": False,
            "stage1_v4_threshold_values_changed": False,
        },
        "execution_gate": {
            "blockers": list(BLOCKERS),
            "all_blockers_require_separate_content_addressed_reviewed_implementation_and_tests": True,
            "execution_mode_present_in_this_implementation": False,
            "execution_authorized": False,
        },
        "implementation": {
            "relative_path": "benchmarks/jx_tno_stage15_resolved11_encounter_qualification_v1.py",
            "bytes": len(source_payload),
            "file_sha256": _sha256_bytes(source_payload),
            "test_relative_path": "tests/test_tno_stage15_resolved11_encounter_qualification_v1.py",
            "test_bytes": len(test_payload),
            "test_file_sha256": _sha256_bytes(test_payload),
            "cli_surface": "PREPARE_OR_VALIDATE_PROTOCOL_ONLY_NO_EXECUTE_SUBCOMMAND",
            "dynamics_modules_imported": False,
        },
        "claims": {
            "input_hashes_and_stage1_oracle_slices_revalidated": True,
            "stage15_protocol_predeclared_before_execution": True,
            "externally_preregistered": False,
            "encounter_engine_qualified": False,
            "adapter_implemented": False,
            "execution_authorized": False,
            "dynamics_executed": False,
            "result_artifacts_published": False,
            "qualification_established": False,
            "dynamical_classification_generated": False,
            "resonance_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "preparation_only": True,
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": "THIS_IS_NOT_A_DYNAMICS_RESULT_CLASSIFICATION_RESONANCE_RESULT_POPULATION_RESULT_OR_PLANET_NINE_RESULT",
    }
    record["content_sha256"] = _content_sha256(record)
    return record


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or not _exact_equal(value, expected):
        raise Stage15ProtocolError("protocol differs from exact frozen reconstruction")
    digest = _content_sha256(value)
    _require_exact(value.get("content_sha256"), digest, "protocol content digest")
    _require_exact(value.get("execution_authorized"), False, "execution claim")
    _require_exact(value.get("outcomes_generated"), False, "outcome claim")
    _require_exact(value.get("preparation_only"), True, "preparation-only claim")
    claims = value.get("claims")
    if type(claims) is not dict:
        raise Stage15ProtocolError("claims record is absent")
    for key in (
        "encounter_engine_qualified",
        "adapter_implemented",
        "execution_authorized",
        "dynamics_executed",
        "result_artifacts_published",
        "qualification_established",
        "dynamical_classification_generated",
        "resonance_classification_generated",
        "population_inference",
        "planet_nine_evidence",
        "science_result",
    ):
        _require_exact(claims.get(key), False, f"claim {key}")
    _require_exact(
        value.get("execution_gate", {}).get("blockers"),
        list(BLOCKERS),
        "execution blockers",
    )
    return digest


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise Stage15ProtocolError(f"protocol path is not create-only: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    if output.name != "PROTOCOL.json":
        raise Stage15ProtocolError("output basename must be PROTOCOL.json")
    parent = output.parent
    if parent.is_symlink():
        raise Stage15ProtocolError("output directory cannot be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise Stage15ProtocolError("output directory must be a real directory")
    if any(parent.iterdir()):
        raise Stage15ProtocolError("dedicated output directory is not empty")
    protocol = build_protocol()
    _write_create_only(output, _canonical_json(protocol))
    try:
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Stage15ProtocolError("output directory fsync failed") from exc
    retained = _load_canonical_payload(_read_regular_bytes(output), str(output))
    validate_protocol(retained)
    return protocol


def validate_file(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = _load_canonical_payload(_read_regular_bytes(path), str(path))
    validate_protocol(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="verify inputs and create the protocol; never run dynamics",
    )
    modes.add_argument(
        "--validate-protocol",
        action="store_true",
        help="validate an existing protocol; never run dynamics",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.prepare_only:
            value = prepare(args.output)
        else:
            value = validate_file(args.output)
    except Stage15ProtocolError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
