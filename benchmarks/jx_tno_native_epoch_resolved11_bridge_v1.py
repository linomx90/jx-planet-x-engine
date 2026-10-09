#!/usr/bin/env python3
"""Materialize a prepare-only native-epoch TNO/resolved-11 state bridge.

The bridge executes the already-qualified, replayed DE440s CSPICE acquisition
at twelve object-specific JD(TDB) epochs, applies the existing exact resolved-
eleven operational-GM centroid contract, and translates a preregistered
nominal/eight-clone TNO holdout into the same J2000/SI origin.  It never
integrates a trajectory, creates a source tile, classifies an orbit, or treats
DE440s planetary states as equivalent to the DE441 small-body orbit fits.
"""

from __future__ import annotations

import argparse
import dataclasses
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from jxplanetx.solar_system import artifacts as artifact_contracts
from jxplanetx.solar_system import contracts as solar_contracts
from jxplanetx.solar_system import cspice_execution as cspice_execution
from jxplanetx.solar_system import cspice_execution_contracts as cspice_contracts
from jxplanetx.solar_system import cspice_time
from jxplanetx.solar_system import engine_preparation as engine_preparation
from jxplanetx.solar_system import engine_preparation_contracts as preparation_contracts
from jxplanetx.solar_system import serialization as solar_serialization
from jxplanetx.solar_system import states as state_contracts
from jxplanetx.solar_system import units as unit_contracts
from jxplanetx.solar_system.contracts import (
    ArtifactBinding,
    CoordinateEpoch,
    CoverageInterval,
    EphemerisProviderSpec,
    EphemerisQuerySpec,
    ExactUnitScale,
    FrameRealization,
    ProviderIdentity,
    UnitSystemDefinition,
    validate_integrity,
)
from jxplanetx.solar_system.cspice_execution import execute_cspice_spkgeo_replay
from jxplanetx.solar_system.cspice_time import project_cspice_binary64_query
from jxplanetx.solar_system.engine_preparation import (
    prepare_de440_newtonian_engine_inputs,
    validate_prepared_de440_newtonian_inputs,
)
from jxplanetx.solar_system.engine_preparation_contracts import (
    validate_de440_newtonian_initial_state,
)
from jxplanetx.solar_system.units import KILOGRAM, KILOMETRE, SECOND


SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_native_epoch_resolved11_bridge_v1.py"
STAGE0_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_resolved11_bridge_v1"
EXTERNAL_RELATIVE_ROOT = Path("external/de440_replay")

PROTOCOL_SCHEMA = "jx.tno.native-epoch-resolved11-bridge.protocol.v1"
RECEIPTS_SCHEMA = "jx.tno.native-epoch-resolved11-bridge.receipts.v1"
BRIDGE_SCHEMA = "jx.tno.native-epoch-resolved11-bridge.states.v1"
PROTOCOL_DOMAIN = b"jx.tno.native-epoch-resolved11-bridge.protocol.v1\0"
RECEIPTS_DOMAIN = b"jx.tno.native-epoch-resolved11-bridge.receipts.v1\0"
BRIDGE_DOMAIN = b"jx.tno.native-epoch-resolved11-bridge.states.v1\0"
BATCH_DOMAIN = b"jx.tno.native-epoch-resolved11-bridge.batch.v1\0"
BINARY_DOMAIN = b"jx.tno.native-epoch-resolved11-bridge.binary.v1\0"
CLASSIFICATION = "DE440S_NATIVE_EPOCH_RESOLVED11_PREPARE_ONLY_INPUT_BRIDGE"
CONTENT_INTEGRITY = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

ABI = "cp314-cp314"
PROVIDER_ID = "provider.spiceypy-8.2.0.cspice-n0067-derived.v1"
IMPLEMENTATION_ID = "implementation.spiceypy-8.2.0.cspice-n0067-derived.v1"
PROVIDER_SPEC_ID = "provider-spec.spiceypy-8.2.0.cspice-n0067-derived.single-spk-j2000.v1"
RUNTIME_ID = "runtime.cpython-3.14.4.cp314-linux-x86_64.host-local-observed.v1"
RULES_ID = "artifact.cspice.naif-rules-license"
COMPOSITE_ID = "artifact.cspice.license"
NUMPY_LICENSE_ID = "artifact.cspice.numpy-license"
RULES_SPDX = "LicenseRef-NAIF-SPICE-Rules"
COMPOSITE_SPDX = "MIT AND LicenseRef-NAIF-SPICE-Rules"
NUMPY_SPDX = "LicenseRef-NumPy-2.3.5-Wheel-License-Manifest"
RESOLVED11_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)

AU_METRES_EXACT = 149_597_870_700
SECONDS_PER_JULIAN_YEAR_EXACT = 31_557_600
OBLIQUITY_ARCSECONDS_EXACT = "84381.448"
OBLIQUITY_RADIANS_HEX = "0x1.a2e9395ca7c40p-2"
OBLIQUITY_SIN_HEX = "0x1.9752e50f4b399p-2"
OBLIQUITY_COS_HEX = "0x1.d5c0357681ef3p-1"
PI_DECIMAL_120 = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
    "208998628034825342117067982148086513282306647"
)

STAGE0_BINDINGS = (
    ("implementation", "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", 70_064, "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367"),
    ("test", "tests/test_tno_native_epoch_cartesian_qualification_v1.py", 15_944, "1225cc19a78aa4ee412c11bf42e931ac8b6881c41845ad9b359c8ff547cb9d0c"),
    ("protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"),
    ("states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
)
STAGE0_PROTOCOL_CONTENT_SHA256 = "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e"
STAGE0_STATES_CONTENT_SHA256 = "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b"
STAGE0_BINARY_PAYLOAD_SHA256 = "334d3fb579e3be752e44cee91680a0c73bd4ac2cb5ab2bff108d66c389118b83"

EXTERNAL_TOP_BINDINGS = (
    ("README.md", 1_224, "9e9e151d1a708c2c6d8b1cb65a5458f6d841d4903f9c0c0c057d0e74b9fe6d30"),
    ("INPUT_MANIFEST.json", 2_524, "7c0e5ad456d6d2750c55058037256e61323e09a57d1821e74bc20a24be6df3a8"),
    ("SOURCE_MANIFEST.json", 13_049, "858d5c1554f78bc51f1de0189cd502d19b4e647f14e5aa8273a981f24857268b"),
    ("SHA256SUMS", 10_085, "d3bebd2e75fd024a117d4074c3bf9214e4d6c08879b804ed9cb47cd64276deac"),
    ("requirements.txt", 29, "879a453d23ff662968b116569aa9a41a05979192dcae6cc50ded5fca0cc83aa7"),
)
EXTERNAL_SOURCE_TREE_SHA256 = "84cc729c6ac29fa6f95b8a36762d530962279ce3d3dac97ccff5b2cb4a4df88b"
EXTERNAL_SOURCE_BASE_COMMIT = "750206382bd6bf323bd72b3b779cd61ddc79b3a0"

RUNTIME_SOURCE_BINDINGS = (
    ("src/jxplanetx/solar_system/contracts.py", 73_474, "8c3c940ad6d08cddac75e78a0837de7b3f310d5d66dae036e844a3939e477dbb"),
    ("src/jxplanetx/solar_system/serialization.py", 11_953, "f4fbd0f9cfb25171e3fd32e329b6f223512d6054ee76ad30c23c2e08c383de4a"),
    ("src/jxplanetx/solar_system/units.py", 20_389, "348a6a996a1c39571772499100896a466caa1178956ad8acd97bb1db34ae829b"),
    ("src/jxplanetx/solar_system/artifacts.py", 34_663, "a9e4d553d034eb6b49e90507b3f3e8d97f1851151214866784cb69344f1bf025"),
    ("src/jxplanetx/solar_system/states.py", 11_965, "67c8726569f7e39a9c21d49b64b4337d8b1bfaa9d8787b2760282b93ba0e4869"),
    ("src/jxplanetx/solar_system/cspice_time.py", 20_519, "60aaecc001af508c632603ae86712c155bfa54ba8ac9ed350dc0c341e72bea00"),
    ("src/jxplanetx/solar_system/cspice_execution_contracts.py", 100_339, "b1f7b22094ca152ba5ec7b720db4127b2b3526d69955b830d5625fd8f4f43de8"),
    ("src/jxplanetx/solar_system/cspice_execution.py", 88_890, "bee84c9e5cb29dcca9e2eb5d4a30eb13d2ebe10ffe02b2cec2b5483b62f673d8"),
    ("src/jxplanetx/solar_system/_cspice_worker.py", 40_504, "09824658b6acb407ea683ff6d9c24df07bff3684549f5b478844f24c7d6e0951"),
    ("src/jxplanetx/solar_system/engine_preparation_contracts.py", 73_122, "db4437f49b9bbe72bd6ebdc33d05623471eca3b172efa4c4057a516d9b26f369"),
    ("src/jxplanetx/solar_system/engine_preparation.py", 38_195, "f16c7e4d67c6b147f8a441e46947d0718c5c5125f8755cdc26704e604afd7030"),
)
ISOLATED_WORKER_SOURCE_RELATIVE_PATH = "src/jxplanetx/solar_system/_cspice_worker.py"


class BridgeError(ValueError):
    """A sealed input, execution receipt, or bridge invariant failed closed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(payload))


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise BridgeError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BridgeError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise BridgeError("JSON root must be an object")
    return value


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BridgeError(f"required file cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BridgeError(f"required path is linked or nonregular: {path}")
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
    current = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(current) or len(payload) != before.st_size:
        raise BridgeError(f"required file changed while read: {path}")
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = _read_regular(path)
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise BridgeError(f"retained JSON is not canonical: {path}")
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _require(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise BridgeError(f"{label} differs from the frozen contract")


def _bound_record(relative_path: str, size: int, digest: str, role: str | None = None) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    payload = _read_regular(path)
    if len(payload) != size or _sha256(payload) != digest:
        raise BridgeError(f"bound bytes differ: {relative_path}")
    result: dict[str, object] = {"relative_path": relative_path, "bytes": size, "file_sha256": digest}
    if role is not None:
        result["role"] = role
    return result


def _load_stage0_module():
    role, relative_path, size, digest = STAGE0_BINDINGS[0]
    payload = _read_regular(PROJECT_ROOT / relative_path)
    if len(payload) != size or _sha256(payload) != digest:
        raise BridgeError("Stage-0 implementation binding differs")
    name = "jx_tno_native_epoch_cartesian_for_resolved11_bridge"
    module = types.ModuleType(name)
    module.__file__ = str(PROJECT_ROOT / relative_path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(payload, module.__file__, "exec", dont_inherit=True, optimize=-1), module.__dict__)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise BridgeError("Stage-0 verifier source failed to load") from exc
    return module


def _validate_stage0() -> dict[str, Any]:
    records = [_bound_record(path, size, digest, role) for role, path, size, digest in STAGE0_BINDINGS]
    module = _load_stage0_module()
    try:
        states = module.validate_output(STAGE0_ROOT)
    except Exception as exc:
        raise BridgeError("sealed Stage-0 output failed exact validation") from exc
    _require(states.get("content_sha256"), STAGE0_STATES_CONTENT_SHA256, "Stage-0 states content digest")
    _require(states.get("protocol_content_sha256"), STAGE0_PROTOCOL_CONTENT_SHA256, "Stage-0 protocol content digest")
    _require(states.get("binary_payload_sha256"), STAGE0_BINARY_PAYLOAD_SHA256, "Stage-0 binary payload digest")
    _require(states.get("total_state_count"), 1_548, "Stage-0 state count")
    _require([_bound_record(path, size, digest, role) for role, path, size, digest in STAGE0_BINDINGS], records, "Stage-0 rebound files")
    return states


def _external_root(output_root: Path) -> Path:
    return output_root / EXTERNAL_RELATIVE_ROOT


def _validate_external_bundle(output_root: Path) -> dict[str, object]:
    root = _external_root(output_root)
    if root.is_symlink() or not root.is_dir():
        raise BridgeError("retained replay bundle root is unavailable or linked")
    top = []
    for relative, size, digest in EXTERNAL_TOP_BINDINGS:
        payload = _read_regular(root / relative)
        if len(payload) != size or _sha256(payload) != digest:
            raise BridgeError(f"external top-level binding differs: {relative}")
        top.append({"relative_path": str(EXTERNAL_RELATIVE_ROOT / relative), "bytes": size, "file_sha256": digest})
    sums_payload = _read_regular(root / "SHA256SUMS")
    expected_leaves: dict[str, tuple[int, str]] = {}
    for line in sums_payload.decode("ascii").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise BridgeError("external SHA256SUMS syntax differs")
        digest, relative = line[:64], line[66:]
        if relative in expected_leaves or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise BridgeError("external SHA256SUMS path roster is invalid")
        payload = _read_regular(root / relative)
        if _sha256(payload) != digest:
            raise BridgeError(f"external replay leaf digest differs: {relative}")
        expected_leaves[relative] = (len(payload), digest)
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    expected_files = set(expected_leaves) | {"SHA256SUMS"}
    if actual_files != expected_files or len(actual_files) != 86:
        raise BridgeError("external replay bundle file roster differs")
    input_manifest = _strict_json(_read_regular(root / "INPUT_MANIFEST.json"))
    source_manifest = _strict_json(_read_regular(root / "SOURCE_MANIFEST.json"))
    _require(input_manifest.get("schema"), "jx.eleven-body-replay-input-manifest.v1", "external input schema")
    _require(input_manifest.get("input_count"), 9, "external input count")
    inputs = input_manifest.get("inputs")
    if type(inputs) is not list or len(inputs) != 9:
        raise BridgeError("external input manifest roster differs")
    for row in inputs:
        if type(row) is not dict:
            raise BridgeError("external input manifest row must be an object")
        relative = row.get("logical_locator")
        if type(relative) is not str or relative not in expected_leaves:
            raise BridgeError("external input manifest locator differs")
        size, digest = expected_leaves[relative]
        _require(row.get("size_bytes"), size, "external input size")
        _require(row.get("sha256"), digest, "external input digest")
    _require(source_manifest.get("schema"), "jx.eleven-body-replay-source-manifest.v1", "external source schema")
    _require(source_manifest.get("source_base_commit"), EXTERNAL_SOURCE_BASE_COMMIT, "external source base commit")
    _require(source_manifest.get("source_tree_sha256"), EXTERNAL_SOURCE_TREE_SHA256, "external source tree digest")
    _require(source_manifest.get("file_count"), 72, "external source file count")
    source_rows = source_manifest.get("files")
    if type(source_rows) is not list or len(source_rows) != 72:
        raise BridgeError("external source manifest file roster differs")
    for row in source_rows:
        if type(row) is not dict or type(row.get("path")) is not str:
            raise BridgeError("external source manifest row differs")
        relative = "source_snapshot/" + row["path"]
        if relative not in expected_leaves:
            raise BridgeError("external source manifest path is absent")
        size, digest = expected_leaves[relative]
        _require(row.get("size_bytes"), size, "external source size")
        _require(row.get("sha256"), digest, "external source digest")
    return {
        "root_relative_path": str(EXTERNAL_RELATIVE_ROOT),
        "file_count": 86,
        "sha256sums_entry_count": len(expected_leaves),
        "top_level_bindings": top,
        "source_base_commit": EXTERNAL_SOURCE_BASE_COMMIT,
        "source_tree_sha256": EXTERNAL_SOURCE_TREE_SHA256,
        "input_manifest_sha256": EXTERNAL_TOP_BINDINGS[1][2],
        "custody_classification": CONTENT_INTEGRITY,
    }


def _epoch(et_seconds: int) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale="TDB",
        representation="SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        whole=et_seconds,
        fraction=0.0,
        coordinate_unit_id=SECOND.unit_id,
        origin_id="SPICE_J2000_TDB_ORIGIN",
        realization_id="de440s.spice-et.v1",
        artifact_ids=(),
    )


def _artifact_binding(
    artifact_id: str,
    role: str,
    version: str,
    locator: str,
    length: int,
    digest: str,
    media_type: str,
    spdx: str,
    license_id: str | None,
    *,
    coverage: tuple[CoverageInterval, ...] | None = None,
    load_order: int | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id=artifact_id,
        artifact_role=role,
        provider_id=PROVIDER_ID,
        version=version,
        logical_locator=locator,
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=length,
        artifact_sha256=digest,
        media_type=media_type,
        coverage_status="COARSE_ARTIFACT_TIME_ENVELOPE" if coverage is not None else "NOT_APPLICABLE",
        coverage=coverage,
        license_evidence_status="RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING" if role == "LICENSE" else "RETAINED_HASH_BOUND_LICENSE_ARTIFACT",
        license_spdx=spdx,
        license_artifact_id=None if role == "LICENSE" else license_id,
        redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        load_order_status="ORDERED_LOAD_MEMBER" if load_order is not None else "NOT_LOADABLE",
        load_order=load_order,
        extrapolation_policy="FORBID",
        content_integrity_class=CONTENT_INTEGRITY,
    )


def production_provider() -> EphemerisProviderSpec:
    """Reconstruct the sealed provider from production contracts/constants only."""

    profile = cspice_contracts._ABI_PROFILES[ABI]
    coverage = (CoverageInterval(_epoch(-4_734_072_000), _epoch(4_735_368_000), "CLOSED_CLOSED"),)
    artifacts = (
        _artifact_binding(cspice_execution._SPK_ID, "SPK", "de440s", "kernels/de440s.bsp", cspice_execution._SPK_LENGTH, cspice_execution._SPK_SHA256, "application/octet-stream", RULES_SPDX, RULES_ID, coverage=coverage, load_order=0),
        _artifact_binding(COMPOSITE_ID, "LICENSE", "SpiceyPy-8.2.0-bundled-N0067-derived-patched-composite-license-provenance.v1", "licenses/spiceypy-cspice-composite-license-provenance.v1.json", cspice_execution._COMPOSITE_LENGTH, cspice_execution._COMPOSITE_SHA256, "application/json", COMPOSITE_SPDX, None),
        _artifact_binding(RULES_ID, "LICENSE", "NAIF-SPICE-Rules-ae85f851.v1", "licenses/naif-spice-rules.html", cspice_execution._RULES_LENGTH, cspice_execution._RULES_SHA256, "text/html", RULES_SPDX, None),
        _artifact_binding(cspice_execution._NATIVE_ID, "NATIVE_LIBRARY", "SpiceyPy-8.2.0-bundled-N0067-derived-patched", "native/libcspice.so", cspice_execution._NATIVE_LENGTH, cspice_execution._NATIVE_SHA256, "application/x-sharedlib", RULES_SPDX, RULES_ID),
        _artifact_binding(NUMPY_LICENSE_ID, "LICENSE", "NumPy-2.3.5-wheel-license-manifest-2046a313.v1", "licenses/numpy-2.3.5-LICENSE.txt", cspice_execution._NUMPY_LICENSE_LENGTH, cspice_execution._NUMPY_LICENSE_SHA256, "text/plain", NUMPY_SPDX, None),
        _artifact_binding(cspice_execution._NUMPY_ID, "SOFTWARE_DISTRIBUTION", "2.3.5", "wheels/numpy-2.3.5-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", profile["numpy_wheel_length"], profile["numpy_wheel_sha256"], "application/zip", NUMPY_SPDX, NUMPY_LICENSE_ID),
        _artifact_binding(cspice_execution._SPICEYPY_ID, "SOFTWARE_DISTRIBUTION", "8.2.0", "wheels/spiceypy-8.2.0-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", profile["spiceypy_wheel_length"], profile["spiceypy_wheel_sha256"], "application/zip", COMPOSITE_SPDX, COMPOSITE_ID),
    )
    identity = ProviderIdentity(
        provider_id=PROVIDER_ID,
        implementation_id=IMPLEMENTATION_ID,
        provider_kind="LOCAL_OFFLINE_SPK",
        version="8.2.0",
        runtime_id=RUNTIME_ID,
        module_name="spiceypy",
        distribution_name="spiceypy",
        distribution_version="8.2.0",
        ordered_implementation_artifact_ids=(cspice_execution._SPICEYPY_ID, cspice_execution._NUMPY_ID, cspice_execution._NATIVE_ID),
    )
    frame = FrameRealization(
        frame_id="frame.naif-j2000.ssb.tdb-compatible.v1",
        frame_kind="TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
        origin_kind="SOLAR_SYSTEM_BARYCENTER",
        origin_naif_id=0,
        origin_realization_id="NAIF_BODY_0_SOLAR_SYSTEM_BARYCENTER",
        axes_realization_id="NAIF_BUILTIN_INERTIAL_FRAME_1_J2000",
        orientation_model_id="NAIF_BUILTIN_J2000_STATIC_ORIENTATION",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="TDB",
        coverage_status="RETAINED",
        coverage=coverage,
        artifact_ids=(cspice_execution._SPK_ID,),
    )
    gm = ExactUnitScale(
        unit_id="unit.kilometre3-per-second2.defined-exact.v1",
        dimension="GRAVITATIONAL_PARAMETER",
        si_unit_id="si.metre3-per-second2",
        numerator=1_000_000_000,
        denominator=1,
        definition_classification="DEFINED_EXACT",
        artifact_ids=(),
    )
    units = UnitSystemDefinition(
        unit_system_id="units.cspice.kilometre-second-kilogram.v1",
        length=KILOMETRE,
        time=SECOND,
        mass=KILOGRAM,
        gravitational_parameter=gm,
        coordinate_time_scale="TDB",
    )
    provider = EphemerisProviderSpec(
        spec_id=PROVIDER_SPEC_ID,
        identity=identity,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
        ordered_load_artifact_ids=(cspice_execution._SPK_ID,),
        capabilities=("COARSE_ARTIFACT_TIME_ENVELOPE", "GEOMETRIC_CARTESIAN_STATE", "TDB_COORDINATE_EPOCH"),
        native_time_scale="TDB",
        native_frame=frame,
        native_output_unit_system=units,
        network_access=False,
        fallback_allowed=False,
        extrapolation_policy="FORBID",
        evidence_class="MODEL_OUTPUT",
        registry_authorized=False,
        qualification_authorized=False,
    )
    validate_integrity(provider)
    _require(provider.content_sha256, cspice_execution._PROVIDER_SEALS[ABI], "production provider seal")
    return provider


def projection_for_et(et_seconds: int, object_id: str):
    if type(et_seconds) is not int or type(object_id) is not str:
        raise BridgeError("projection inputs have wrong exact types")
    provider = production_provider()
    requested = EphemerisQuerySpec(
        query_id=f"query.tno-resolved11.requested.{object_id}.v1",
        provider=provider,
        epoch=_epoch(et_seconds),
        target_naif_ids=RESOLVED11_NAIF_IDS,
        observer_naif_id=0,
        frame=provider.native_frame,
        aberration_correction="NONE",
        output_unit_system=provider.native_output_unit_system,
        state_kind="GEOMETRIC",
        target_chain_availability_status="REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION",
    )
    projection = project_cspice_binary64_query(requested, effective_query_id=f"query.tno-resolved11.effective.{object_id}.v1")
    if projection.binary64_projection.rounding_status != "EXACT":
        raise BridgeError("native epoch is not exactly representable at the CSPICE ET boundary")
    return projection


def _decimal_sin(value: Decimal) -> Decimal:
    term = value
    total = term
    index = 1
    while True:
        term = -term * value * value / Decimal((2 * index) * (2 * index + 1))
        updated = total + term
        if updated == total:
            return updated
        total = updated
        index += 1


def _decimal_cos(value: Decimal) -> Decimal:
    term = Decimal(1)
    total = term
    index = 1
    while True:
        term = -term * value * value / Decimal((2 * index - 1) * (2 * index))
        updated = total + term
        if updated == total:
            return updated
        total = updated
        index += 1


def independently_rounded_obliquity() -> tuple[float, float, float]:
    with localcontext() as context:
        context.prec = 110
        radians = Decimal(OBLIQUITY_ARCSECONDS_EXACT) * Decimal(PI_DECIMAL_120) / (Decimal(180) * Decimal(3600))
        values = (float(radians), float(_decimal_sin(radians)), float(_decimal_cos(radians)))
    _require(tuple(value.hex() for value in values), (OBLIQUITY_RADIANS_HEX, OBLIQUITY_SIN_HEX, OBLIQUITY_COS_HEX), "independently rounded obliquity constants")
    return values


def _pxform_kat() -> dict[str, object]:
    try:
        import spiceypy
        from spiceypy.utils import libspicehelper
    except (ImportError, OSError) as exc:
        raise BridgeError("SpiceyPy is unavailable for the fixed ECLIPJ2000 PXFORM KAT") from exc
    library_path = Path(libspicehelper.libspice._name).resolve()
    library = _read_regular(library_path)
    if len(library) != cspice_execution._NATIVE_LENGTH or _sha256(library) != cspice_execution._NATIVE_SHA256:
        raise BridgeError("PXFORM KAT did not load the exact bound CSPICE library")
    matrix = spiceypy.pxform("ECLIPJ2000", "J2000", 0.0)
    observed = [[float(component).hex() for component in row] for row in matrix]
    expected = [
        ["0x1.0000000000000p+0", "0x0.0p+0", "0x0.0p+0"],
        ["0x0.0p+0", OBLIQUITY_COS_HEX, "-" + OBLIQUITY_SIN_HEX],
        ["0x0.0p+0", OBLIQUITY_SIN_HEX, OBLIQUITY_COS_HEX],
    ]
    _require(observed, expected, "CSPICE ECLIPJ2000-to-J2000 PXFORM matrix")
    return {
        "spiceypy_version": spiceypy.__version__,
        "loaded_cspice_byte_length": len(library),
        "loaded_cspice_sha256": _sha256(library),
        "et_seconds_hex": "0x0.0p+0",
        "matrix_binary64_hex": observed,
        "classification": "HOST_LOCAL_EXACT_LIBRARY_BYTE_MATCH_CORROBORATION_NOT_INDEPENDENT_PROVIDER_AUTHENTICATION",
    }


def _rotate_ecliptic_to_j2000(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    if type(vector) is not tuple or len(vector) != 3 or any(type(item) is not float or not math.isfinite(item) for item in vector):
        raise BridgeError("rotation input must be an immutable finite binary64 triplet")
    _radians, sine, cosine = independently_rounded_obliquity()
    x, y, z = vector
    # Treat every binary64 input as its exact dyad, evaluate each two-term dot
    # product exactly, and round only the completed output component.
    exact_sine = Fraction.from_float(sine)
    exact_cosine = Fraction.from_float(cosine)
    exact_y = Fraction.from_float(y)
    exact_z = Fraction.from_float(z)
    result = (
        x,
        _round_fraction(exact_cosine * exact_y - exact_sine * exact_z),
        _round_fraction(exact_sine * exact_y + exact_cosine * exact_z),
    )
    if any(not math.isfinite(item) for item in result):
        raise BridgeError("rotation produced a nonfinite component")
    return result


def _inverse_rotate_j2000_to_ecliptic(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    _radians, sine, cosine = independently_rounded_obliquity()
    x, y, z = vector
    exact_sine = Fraction.from_float(sine)
    exact_cosine = Fraction.from_float(cosine)
    exact_y = Fraction.from_float(y)
    exact_z = Fraction.from_float(z)
    return (
        x,
        _round_fraction(exact_cosine * exact_y + exact_sine * exact_z),
        _round_fraction(-exact_sine * exact_y + exact_cosine * exact_z),
    )


def _round_fraction(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BridgeError("exact bridge arithmetic overflowed binary64")
    return result


def _position_au_to_metres(value: float) -> float:
    return _round_fraction(Fraction.from_float(value) * AU_METRES_EXACT)


def _velocity_auy_to_metres_per_second(value: float) -> float:
    return _round_fraction(Fraction.from_float(value) * AU_METRES_EXACT / SECONDS_PER_JULIAN_YEAR_EXACT)


def _source_km_to_si(value: float) -> float:
    if value == 0.0:
        return math.copysign(0.0, value)
    return _round_fraction(Fraction.from_float(value) * 1000)


def _fraction_pair(value: tuple[int, int]) -> Fraction:
    if type(value) is not tuple or len(value) != 2 or type(value[0]) is not int or type(value[1]) is not int or value[1] <= 0:
        raise BridgeError("centroid pair is not an exact reduced rational")
    result = Fraction(value[0], value[1])
    if (result.numerator, result.denominator) != value:
        raise BridgeError("centroid pair is not canonical")
    return result


_FREEZE_MODULES = (
    solar_contracts,
    unit_contracts,
    artifact_contracts,
    state_contracts,
    cspice_time,
    cspice_contracts,
    preparation_contracts,
)
_DATACLASS_ALLOWLIST = {
    f"{item.__module__}.{item.__qualname__}": item
    for module in _FREEZE_MODULES
    for item in vars(module).values()
    if inspect.isclass(item) and dataclasses.is_dataclass(item) and item.__module__ == module.__name__
}


def _freeze(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        if name not in _DATACLASS_ALLOWLIST:
            raise BridgeError(f"receipt dataclass is outside the exact allowlist: {name}")
        return {"__kind__": "dataclass", "type": name, "fields": {field.name: _freeze(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if type(value) is tuple:
        return {"__kind__": "tuple", "items": [_freeze(item) for item in value]}
    if type(value) is float:
        if not math.isfinite(value):
            raise BridgeError("receipt contains a nonfinite binary64 value")
        return {"__kind__": "binary64", "hex": value.hex()}
    if type(value) is list:
        return {"__kind__": "list", "items": [_freeze(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise BridgeError("receipt mapping key is not a string")
        return {"__kind__": "mapping", "items": {key: _freeze(item) for key, item in value.items()}}
    if value is None or type(value) in (bool, int, str):
        return value
    raise BridgeError(f"receipt contains an unsupported exact type: {type(value).__name__}")


def _thaw(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is not dict or type(value.get("__kind__")) is not str:
        raise BridgeError("frozen receipt node has an invalid exact shape")
    kind = value["__kind__"]
    if kind == "binary64":
        if set(value) != {"__kind__", "hex"} or type(value["hex"]) is not str:
            raise BridgeError("frozen binary64 node differs")
        result = float.fromhex(value["hex"])
        if not math.isfinite(result) or result.hex() != value["hex"]:
            raise BridgeError("frozen binary64 token is not canonical finite hex")
        return result
    if kind in ("tuple", "list"):
        if set(value) != {"__kind__", "items"} or type(value["items"]) is not list:
            raise BridgeError("frozen sequence node differs")
        items = [_thaw(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "mapping":
        if set(value) != {"__kind__", "items"} or type(value["items"]) is not dict:
            raise BridgeError("frozen mapping node differs")
        return {key: _thaw(item) for key, item in value["items"].items()}
    if kind == "dataclass":
        if set(value) != {"__kind__", "type", "fields"} or type(value["type"]) is not str or type(value["fields"]) is not dict:
            raise BridgeError("frozen dataclass node differs")
        cls = _DATACLASS_ALLOWLIST.get(value["type"])
        if cls is None:
            raise BridgeError("frozen dataclass type is outside the exact allowlist")
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(value["fields"]) != expected:
            raise BridgeError("frozen dataclass field roster differs")
        try:
            return cls(**{key: _thaw(item) for key, item in value["fields"].items()})
        except Exception as exc:
            raise BridgeError(f"frozen dataclass failed exact reconstruction: {value['type']}") from exc
    raise BridgeError("frozen receipt node kind differs")


def _receipt_inventory(value: object, path: str = "initial_state") -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest = getattr(value, "content_sha256", None)
        if type(digest) is str and len(digest) == 64:
            result.append({"path": path, "type": f"{type(value).__module__}.{type(value).__qualname__}", "content_sha256": digest})
        for field in dataclasses.fields(value):
            result.extend(_receipt_inventory(getattr(value, field.name), f"{path}.{field.name}"))
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            result.extend(_receipt_inventory(item, f"{path}[{index}]"))
    elif type(value) is dict:
        for key in sorted(value):
            result.extend(_receipt_inventory(value[key], f"{path}.{key}"))
    return result


def _et_seconds(epoch_hex: str) -> int:
    jd = float.fromhex(epoch_hex)
    exact = (Fraction.from_float(jd) - Fraction(2_451_545, 1)) * 86_400
    if exact.denominator != 1:
        raise BridgeError("Stage-0 native epoch does not map to exact integer TDB ET seconds")
    return exact.numerator


def _selected_stage0_rows(states: Mapping[str, Any]) -> list[tuple[str, float, list[dict[str, Any]]]]:
    rows = states.get("states")
    if type(rows) is not list or len(rows) != 1_548:
        raise BridgeError("Stage-0 row roster differs")
    groups: list[tuple[str, float, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or type(row.get("object_id")) is not str:
            raise BridgeError("Stage-0 state row differs")
        object_id = row["object_id"]
        if object_id in seen:
            continue
        seen.add(object_id)
        group = [item for item in rows if item.get("object_id") == object_id]
        if len(group) != 129 or group[0].get("clone_index") is not None:
            raise BridgeError("Stage-0 per-object roster differs")
        selected = [group[0]]
        for index in HOLDOUT_CLONE_INDICES:
            matches = [item for item in group if item.get("clone_index") == index]
            if len(matches) != 1:
                raise BridgeError("Stage-0 holdout clone selection differs")
            selected.append(matches[0])
        epoch_hex = group[0].get("covariance_epoch_jd_tdb_hex")
        if type(epoch_hex) is not str or any(item.get("covariance_epoch_jd_tdb_hex") != epoch_hex for item in group):
            raise BridgeError("Stage-0 per-object native epoch differs")
        groups.append((object_id, float.fromhex(epoch_hex), selected))
    if len(groups) != 12 or sum(len(group[2]) for group in groups) != 108:
        raise BridgeError("selected Stage-0 holdout roster differs")
    return groups


def _source_records() -> list[dict[str, object]]:
    return [_bound_record(path, size, digest) for path, size, digest in RUNTIME_SOURCE_BINDINGS]


def _loaded_project_source_observations() -> list[dict[str, object]]:
    """Observe every currently loaded project Python source after preparation.

    This is a point observation of source paths/bytes, not proof that imports did
    not use bytecode or that the observations are the executed instructions.
    """

    observed: dict[str, dict[str, object]] = {}
    project = PROJECT_ROOT.resolve()
    for name, loaded in tuple(sys.modules.items()):
        if name != "jxplanetx" and not name.startswith("jxplanetx."):
            continue
        raw = getattr(loaded, "__file__", None)
        if type(raw) is not str:
            raise BridgeError(f"loaded project module has no exact source path: {name}")
        path = Path(raw).resolve()
        if path.suffix in (".pyc", ".pyo"):
            try:
                path = Path(importlib.util.source_from_cache(str(path))).resolve()
            except (ValueError, ImportError) as exc:
                raise BridgeError(f"loaded project bytecode has no source mapping: {name}") from exc
        try:
            relative = path.relative_to(project).as_posix()
        except ValueError as exc:
            raise BridgeError(f"loaded project module escaped the project root: {name}") from exc
        if path.suffix != ".py":
            raise BridgeError(f"loaded project module is not backed by Python source: {name}")
        payload = _read_regular(path)
        row = {
            "relative_path": relative,
            "bytes": len(payload),
            "file_sha256": _sha256(payload),
            "classification": "POST_IMPORT_SOURCE_BYTE_POINT_OBSERVATION_NOT_EXECUTED_BYTE_PROOF",
        }
        prior = observed.get(relative)
        if prior is not None and not _exact_equal(prior, row):
            raise BridgeError("loaded project source observation is inconsistent")
        observed[relative] = row
    required = {
        row[0]
        for row in RUNTIME_SOURCE_BINDINGS
        if row[0] != ISOLATED_WORKER_SOURCE_RELATIVE_PATH
    }
    if not required.issubset(observed):
        raise BridgeError("loaded project source roster omits a required bridge module")
    return [observed[key] for key in sorted(observed)]


def _revalidate_source_observations(rows: object) -> list[dict[str, object]]:
    if type(rows) is not list or not rows:
        raise BridgeError("loaded project source observation roster differs")
    result: list[dict[str, object]] = []
    prior = ""
    for row in rows:
        if type(row) is not dict or set(row) != {"relative_path", "bytes", "file_sha256", "classification"}:
            raise BridgeError("loaded project source observation row differs")
        relative = row["relative_path"]
        if type(relative) is not str or relative <= prior or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise BridgeError("loaded project source observation order/path differs")
        payload = _read_regular(PROJECT_ROOT / relative)
        expected = {
            "relative_path": relative,
            "bytes": len(payload),
            "file_sha256": _sha256(payload),
            "classification": "POST_IMPORT_SOURCE_BYTE_POINT_OBSERVATION_NOT_EXECUTED_BYTE_PROOF",
        }
        if not _exact_equal(row, expected):
            raise BridgeError("loaded project source observation bytes differ")
        result.append(expected)
        prior = relative
    required = {
        row[0]
        for row in RUNTIME_SOURCE_BINDINGS
        if row[0] != ISOLATED_WORKER_SOURCE_RELATIVE_PATH
    }
    if not required.issubset({row["relative_path"] for row in result}):
        raise BridgeError("loaded project source observations omit required modules")
    return result


def build_protocol(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    stage0 = _validate_stage0()
    external = _validate_external_bundle(output_root)
    groups = _selected_stage0_rows(stage0)
    provider = production_provider()
    obliquity = independently_rounded_obliquity()
    kat = _pxform_kat()
    source_records = _source_records()
    stage0_records = [_bound_record(path, size, digest, role) for role, path, size, digest in STAGE0_BINDINGS]
    record: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PREPARE_ONLY_12_NATIVE_EPOCH_BATCHES_NO_DYNAMICS",
        "claim_ceiling": "DE440S_RESOLVED11_PLUS_DE441_FIT_TNO_COORDINATE_BRIDGE_INPUT_ONLY",
        "input_bindings": {
            "stage0_files": stage0_records,
            "stage0_protocol_content_sha256": STAGE0_PROTOCOL_CONTENT_SHA256,
            "stage0_states_content_sha256": STAGE0_STATES_CONTENT_SHA256,
            "stage0_binary_payload_sha256": STAGE0_BINARY_PAYLOAD_SHA256,
            "stage0_total_state_count": 1_548,
            "external_replay_bundle": external,
            "runtime_source_files": source_records,
        },
        "selection": {
            "object_count": 12,
            "per_object_tno_state_count": 9,
            "selected_tno_state_count": 108,
            "selected_clone_indices": list(HOLDOUT_CLONE_INDICES),
            "selection_semantics": "NOMINAL_PLUS_EIGHT_FROZEN_STAGE1_HOLDOUT_CLONES_PER_OBJECT",
            "object_order": [group[0] for group in groups],
            "native_epoch_roster": [
                {
                    "object_id": object_id,
                    "covariance_epoch_jd_tdb_hex": epoch_jd.hex(),
                    "spice_et_seconds": _et_seconds(epoch_jd.hex()),
                }
                for object_id, epoch_jd, _selected in groups
            ],
            "mixed_epoch_snapshot_created": False,
            "full_1548_state_dynamics_tile_created": False,
        },
        "provider_contract": {
            "abi": ABI,
            "provider_content_sha256": provider.content_sha256,
            "provider_expected_seal": cspice_execution._PROVIDER_SEALS[ABI],
            "resolved11_naif_ids": list(RESOLVED11_NAIF_IDS),
            "observer_naif_id": 0,
            "source_frame": "NAIF_BUILTIN_INERTIAL_FRAME_1_J2000",
            "source_origin": "NAIF_BODY_0_SOLAR_SYSTEM_BARYCENTER",
            "source_units": "KILOMETRE_AND_KILOMETRE_PER_DEFINED_SI_SECOND",
            "query_time": "TDB_SECONDS_FROM_JD_2451545_0_TDB",
            "aberration_correction": "NONE",
            "spk": "DE440S",
            "network_access": False,
            "fallback_allowed": False,
            "primary_and_replay_required_per_epoch": True,
        },
        "rotation": {
            "source_frame": "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000",
            "target_axes": "NAIF_J2000_EQUATORIAL",
            "obliquity_arcseconds_exact": OBLIQUITY_ARCSECONDS_EXACT,
            "pi_decimal_constant": PI_DECIMAL_120,
            "radians_binary64_hex": obliquity[0].hex(),
            "sin_binary64_hex": obliquity[1].hex(),
            "cos_binary64_hex": obliquity[2].hex(),
            "rounding": "110_DIGIT_DECIMAL_SERIES_THEN_CORRECT_BINARY64_ROUND_TO_NEAREST_TIES_TO_EVEN",
            "matrix_operation": "BINARY64_INPUTS_INTERPRETED_AS_EXACT_DYADS;EACH_TWO_TERM_MATRIX_DOT_EVALUATED_EXACTLY;ONE_RN_EVEN_PER_OUTPUT_COMPONENT",
            "cspice_pxform_kat": kat,
            "legacy_one_ulp_low_radian_or_sine_used": False,
        },
        "origin_and_units": {
            "target_length_unit": "METRE",
            "target_velocity_unit": "METRE_PER_DEFINED_SI_SECOND",
            "AU_metres_exact": AU_METRES_EXACT,
            "seconds_per_julian_year_exact": SECONDS_PER_JULIAN_YEAR_EXACT,
            "operation_order": "EXACT_DYAD_ROTATION_DOT_ONE_RN_EVEN;EXACT_UNIT_RATIO_ONE_RN_EVEN;ADD_CONVERTED_SUN_SSB_AND_SUBTRACT_IDENTICAL_PREPARATION_CENTROID_AS_EXACT_DYADS;ONE_RN_EVEN",
            "target_origin": "OPERATIONAL_GM_WEIGHTED_SELECTED_RESOLVED11_MODEL_BARYCENTER_NOT_DE440_SOLAR_SYSTEM_BARYCENTER",
            "centroid_members": list(RESOLVED11_NAIF_IDS),
            "tno_gravitational_parameter": "CANONICAL_POSITIVE_ZERO",
            "tno_included_in_centroid": False,
            "massive_body_states_copied_from_preparation_initial_state": True,
            "massive_sun_primary_cross_binding": "SUBTRACT_SOURCE_SUN_FROM_EACH_SOURCE_MASSIVE_STATE_AS_EXACT_BINARY64_DYADS;APPLY_EXACT_KM_TO_AU_OR_KM_PER_S_TO_AU_PER_JULIAN_YEAR_RATIO;ONE_RN_EVEN_PER_COMPONENT;SUN_CANONICAL_POSITIVE_ZERO",
            "heliocentric_j2000_AU_and_AU_per_julian_year_cross_binding_retained": True,
        },
        "model_mismatch_gate": {
            "small_body_fit_and_covariance_model": "JPL_SBDB_HORIZONS_DE441_DECLARED_CONTEXT",
            "massive_body_source": "DE440S",
            "de440_de441_equivalence_established": False,
            "future_required_gate": "PAIRED_DE440_DE441_1000_YEAR_108_STATE_OUTCOME_SENSITIVITY",
            "future_gate_executed": False,
            "coordinate_difference_cutoff_sufficient": False,
            "reason": "INITIAL_COORDINATE_PROXIMITY_CANNOT_REPLACE_PAIRED_DYNAMICAL_OUTCOME_SENSITIVITY",
        },
        "receipt_policy": {
            "full_validated_initial_state_receipt_graph_frozen": True,
            "nested_receipt_path_and_content_hash_inventory_retained": True,
            "runtime_and_artifact_receipts_classification": CONTENT_INTEGRITY,
            "provider_origin_authenticated": False,
            "continuous_custody_proven": False,
            "process_independence_proven": False,
        },
        "downstream_blockers": {
            "de440_de441_sensitivity_gate": True,
            "hybrid_close_encounter_engine_qualification": True,
            "classifier_and_resonance_rules_freeze": True,
            "independent_numerical_budget": True,
            "runtime_identity_for_future_dynamics": True,
            "external_commit_or_timestamp_before_preregistration_claim": True,
        },
        "implementation": {
            "relative_path": "benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py",
            "file_sha256": _sha256(_read_regular(SCRIPT_PATH)),
            "surface": "CREATE_ONLY_PREPARE_BRIDGE_NO_PROPAGATION_ENTRY_POINT",
        },
        "claims": {
            "external_de440s_replay_bundle_retained": True,
            "stage0_full_artifact_revalidated": True,
            "twelve_native_epoch_resolved11_replays_required": True,
            "bridge_materialized": True,
            "de440_de441_model_mismatch_resolved": False,
            "source_tile_created": False,
            "dynamics_executed": False,
            "one_thousand_year_holdout_executed": False,
            "paired_de440_de441_sensitivity_executed": False,
            "gpu_executed": False,
            "dynamical_classifications_generated": False,
            "long_horizon_execution_authorized": False,
            "observational_likelihood_evaluated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": "Prepared coordinate batches and unauthenticated content-integrity receipts are not trajectories, dynamical outcomes, classifications, likelihoods, population inference, or Planet Nine evidence.",
    }
    record["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, record)
    return record


def _nonce(object_id: str, role: str) -> str:
    return _sha256(b"jx.tno.native-epoch-resolved11-bridge.nonce.v1\0" + object_id.encode("ascii") + b"\0" + role.encode("ascii"))


def _source_sun_values(initial) -> tuple[tuple[float, ...], tuple[float, ...]]:
    batch = initial.preparation_receipt.source_execution.primary_lane.state_batch
    if batch.target_naif_ids != RESOLVED11_NAIF_IDS:
        raise BridgeError("source execution target roster differs")
    return tuple(_source_km_to_si(value) for value in batch.positions[:3]), tuple(_source_km_to_si(value) for value in batch.velocities[:3])


def _assert_initial_identity(initial, object_id: str, epoch_jd: float, et_seconds: int) -> None:
    """Cross-bind the full nested receipt graph to one outer object/epoch label."""

    if type(object_id) is not str or type(epoch_jd) is not float or type(et_seconds) is not int:
        raise BridgeError("initial-state identity inputs have wrong exact types")
    if not math.isfinite(epoch_jd) or _et_seconds(epoch_jd.hex()) != et_seconds:
        raise BridgeError("outer JD(TDB) and ET labels are inconsistent")
    try:
        validate_de440_newtonian_initial_state(initial)
    except Exception as exc:
        raise BridgeError("initial-state graph failed its native contract validation") from exc
    expected_projection = projection_for_et(et_seconds, object_id)
    expected = {
        "receipt_id": f"receipt.tno-resolved11.{object_id}.v1",
        "preparation_id": f"preparation.tno-resolved11.{object_id}.v1",
        "state_id": f"state.tno-resolved11.{object_id}.v1",
        "plan_id": f"plan.tno-resolved11.{object_id}.v1",
        "primary_batch_id": f"batch.tno-resolved11.{object_id}.primary.v1",
        "replay_batch_id": f"batch.tno-resolved11.{object_id}.replay.v1",
        "primary_nonce": _nonce(object_id, "PRIMARY"),
        "replay_nonce": _nonce(object_id, "REPLAY"),
        "projection_content_sha256": expected_projection.content_sha256,
        "requested_query_id": f"query.tno-resolved11.requested.{object_id}.v1",
        "effective_query_id": f"query.tno-resolved11.effective.{object_id}.v1",
    }
    receipt = initial.preparation_receipt
    source = receipt.source_execution
    primary = source.primary_lane
    replay = source.replay_lane
    actual = {
        "receipt_id": source.receipt_id,
        "preparation_id": receipt.receipt_id,
        "state_id": initial.state_id,
        "plan_id": initial.force_plan_id,
        "primary_batch_id": primary.state_batch.batch_id,
        "replay_batch_id": replay.state_batch.batch_id,
        "primary_nonce": primary.execution_instance_nonce,
        "replay_nonce": replay.execution_instance_nonce,
        "projection_content_sha256": primary.projection.content_sha256,
        "requested_query_id": primary.projection.requested_query.query_id,
        "effective_query_id": primary.projection.effective_query.query_id,
    }
    if not _exact_equal(actual, expected):
        raise BridgeError("nested receipt/query/batch/state/plan identity differs from the outer object label")
    for label, lane in (("primary", primary), ("replay", replay)):
        projection = lane.projection
        if (
            projection.content_sha256 != expected_projection.content_sha256
            or projection.requested_query.content_sha256 != expected_projection.requested_query.content_sha256
            or projection.effective_query.content_sha256 != expected_projection.effective_query.content_sha256
            or lane.state_batch.query.content_sha256 != expected_projection.effective_query.content_sha256
            or projection.binary64_projection.rounding_status != "EXACT"
            or projection.binary64_projection.rounded_value.hex() != float(et_seconds).hex()
        ):
            raise BridgeError(f"{label} projection/state batch differs from the exact object-native epoch")
        requested_epoch = projection.requested_query.epoch
        effective_epoch = projection.effective_query.epoch
        if (
            requested_epoch.whole != et_seconds
            or requested_epoch.fraction.hex() != "0x0.0p+0"
            or effective_epoch.whole != et_seconds
            or effective_epoch.fraction.hex() != "0x0.0p+0"
        ):
            raise BridgeError(f"{label} requested/effective epoch differs from exact integer ET")
    epoch_binding = receipt.epoch_binding
    if (
        epoch_binding.effective_epoch.content_sha256 != expected_projection.effective_query.epoch.content_sha256
        or epoch_binding.effective_binary64_et.hex() != float(et_seconds).hex()
        or epoch_binding.engine_epoch_at_origin.hex() != "0x0.0p+0"
    ):
        raise BridgeError("preparation epoch binding differs from the exact object-native epoch")


def _receipt_summary(initial, object_id: str, epoch_jd: float, et_seconds: int) -> dict[str, object]:
    _assert_initial_identity(initial, object_id, epoch_jd, et_seconds)
    receipt = initial.preparation_receipt
    source = receipt.source_execution
    sun_position, sun_velocity = _source_sun_values(initial)
    inventory = _receipt_inventory(initial)
    if not inventory or inventory[0]["content_sha256"] != initial.content_sha256:
        raise BridgeError("receipt graph inventory does not begin at the initial state")
    return {
        "object_id": object_id,
        "covariance_epoch_jd_tdb_hex": epoch_jd.hex(),
        "spice_et_seconds": et_seconds,
        "spice_et_seconds_hex": float(et_seconds).hex(),
        "provider_content_sha256": source.primary_lane.projection.effective_query.provider.content_sha256,
        "projection_content_sha256": source.primary_lane.projection.content_sha256,
        "execution_receipt_content_sha256": source.content_sha256,
        "primary_lane_content_sha256": source.primary_lane.content_sha256,
        "primary_lane_semantic_content_sha256": source.primary_lane.semantic_content_sha256,
        "primary_runtime_content_sha256": source.primary_lane.runtime.content_sha256,
        "replay_lane_content_sha256": source.replay_lane.content_sha256,
        "replay_lane_semantic_content_sha256": source.replay_lane.semantic_content_sha256,
        "replay_runtime_content_sha256": source.replay_lane.runtime.content_sha256,
        "preparation_receipt_content_sha256": receipt.content_sha256,
        "preparation_receipt_semantic_content_sha256": receipt.semantic_content_sha256,
        "initial_state_content_sha256": initial.content_sha256,
        "initial_state_semantic_content_sha256": initial.semantic_content_sha256,
        "source_sun_ssb_position_m_hex": [value.hex() for value in sun_position],
        "source_sun_ssb_velocity_m_per_s_hex": [value.hex() for value in sun_velocity],
        "centroid_position_exact_pairs": [list(pair) for pair in receipt.centroid_position_exact_pairs],
        "centroid_velocity_exact_pairs": [list(pair) for pair in receipt.centroid_velocity_exact_pairs],
        "recentered_massive_state_payload_sha256": receipt.recentered_payload_sha256,
        "gm_payload_sha256": receipt.gm_payload_sha256,
        "nested_content_receipt_inventory": inventory,
        "sealed_initial_state": _freeze(initial),
    }


def execute_receipts(protocol: Mapping[str, Any], stage0: Mapping[str, Any], output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if protocol.get("content_sha256") != _content_sha256(PROTOCOL_DOMAIN, protocol):
        raise BridgeError("protocol content digest differs before execution")
    _validate_external_bundle(output_root)
    artifact_root = _external_root(output_root) / "artifact-root"
    parameter_root = _external_root(output_root) / "parameter-root"
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    artifact_fd = os.open(artifact_root, flags)
    parameter_fd = os.open(parameter_root, flags)
    entries: list[dict[str, object]] = []
    try:
        for ordinal, (object_id, epoch_jd, _selected) in enumerate(_selected_stage0_rows(stage0), start=1):
            et_seconds = _et_seconds(epoch_jd.hex())
            projection = projection_for_et(et_seconds, object_id)
            execution = execute_cspice_spkgeo_replay(
                projection,
                artifact_root_directory_fd=artifact_fd,
                receipt_id=f"receipt.tno-resolved11.{object_id}.v1",
                primary_batch_id=f"batch.tno-resolved11.{object_id}.primary.v1",
                replay_batch_id=f"batch.tno-resolved11.{object_id}.replay.v1",
                primary_execution_instance_nonce=_nonce(object_id, "PRIMARY"),
                replay_execution_instance_nonce=_nonce(object_id, "REPLAY"),
            )
            prepared = prepare_de440_newtonian_engine_inputs(
                execution,
                parameter_root_directory_fd=parameter_fd,
                receipt_id=f"preparation.tno-resolved11.{object_id}.v1",
                state_id=f"state.tno-resolved11.{object_id}.v1",
                plan_id=f"plan.tno-resolved11.{object_id}.v1",
            )
            validate_prepared_de440_newtonian_inputs(prepared)
            entries.append(_receipt_summary(prepared.initial_state, object_id, epoch_jd, et_seconds))
            print(f"prepared native epoch {ordinal}/12: {object_id}", file=sys.stderr, flush=True)
    finally:
        os.close(parameter_fd)
        os.close(artifact_fd)
    artifact: dict[str, object] = {
        "schema": RECEIPTS_SCHEMA,
        "classification": CLASSIFICATION,
        "protocol_content_sha256": protocol["content_sha256"],
        "epoch_receipt_count": len(entries),
        "entries": entries,
        "host_project_source_observations": _loaded_project_source_observations(),
        "host_project_source_observation_scope": "POST_IMPORT_SOURCE_BYTE_POINT_OBSERVATION_NOT_EXECUTED_BYTE_PROOF",
        "content_integrity_class": CONTENT_INTEGRITY,
        "provider_origin_authenticated": False,
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    artifact["content_sha256"] = _content_sha256(RECEIPTS_DOMAIN, artifact)
    return artifact


def validate_receipts(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[object, ...]:
    if type(value) is not dict:
        raise BridgeError("receipt artifact must be a built-in dict")
    _require(value.get("schema"), RECEIPTS_SCHEMA, "receipt schema")
    _require(value.get("classification"), CLASSIFICATION, "receipt classification")
    _require(value.get("protocol_content_sha256"), protocol.get("content_sha256"), "receipt protocol binding")
    _require(value.get("content_sha256"), _content_sha256(RECEIPTS_DOMAIN, value), "receipt content digest")
    _require(value.get("epoch_receipt_count"), 12, "receipt epoch count")
    _require(value.get("content_integrity_class"), CONTENT_INTEGRITY, "receipt integrity class")
    for key in ("provider_origin_authenticated", "execution_authorized", "outcomes_generated"):
        _require(value.get(key), False, f"receipt {key}")
    entries = value.get("entries")
    if type(entries) is not list or len(entries) != 12:
        raise BridgeError("receipt entry roster differs")
    _require(
        value.get("host_project_source_observation_scope"),
        "POST_IMPORT_SOURCE_BYTE_POINT_OBSERVATION_NOT_EXECUTED_BYTE_PROOF",
        "host project source observation scope",
    )
    _revalidate_source_observations(value.get("host_project_source_observations"))
    reconstructed: list[object] = []
    expected_roster = protocol["selection"]["native_epoch_roster"]
    if type(expected_roster) is not list or len(expected_roster) != 12:
        raise BridgeError("protocol native-epoch roster differs")
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise BridgeError("receipt entry must be an object")
        roster = expected_roster[index]
        if type(roster) is not dict:
            raise BridgeError("protocol native-epoch roster row differs")
        _require(entry.get("object_id"), roster.get("object_id"), "receipt object order")
        _require(entry.get("covariance_epoch_jd_tdb_hex"), roster.get("covariance_epoch_jd_tdb_hex"), "receipt native epoch")
        _require(entry.get("spice_et_seconds"), roster.get("spice_et_seconds"), "receipt ET seconds")
        initial = _thaw(entry.get("sealed_initial_state"))
        try:
            validate_de440_newtonian_initial_state(initial)
        except Exception as exc:
            raise BridgeError("retained initial-state receipt graph failed validation") from exc
        epoch_jd = float.fromhex(roster["covariance_epoch_jd_tdb_hex"])
        et_seconds = roster["spice_et_seconds"]
        expected = _receipt_summary(initial, entry["object_id"], epoch_jd, et_seconds)
        if not _exact_equal(entry, expected):
            raise BridgeError("receipt entry differs from exact reconstruction")
        reconstructed.append(initial)
    return tuple(reconstructed)


def _source_massive_sun_primary(initial) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    batch = initial.preparation_receipt.source_execution.primary_lane.state_batch
    if batch.target_naif_ids != RESOLVED11_NAIF_IDS or len(batch.positions) != 33 or len(batch.velocities) != 33:
        raise BridgeError("source massive state batch differs from the exact resolved-eleven roster")
    sun_position = batch.positions[:3]
    sun_velocity = batch.velocities[:3]
    result: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for index in range(11):
        offset = index * 3
        position: list[float] = []
        velocity: list[float] = []
        for component in range(3):
            relative_km = Fraction.from_float(batch.positions[offset + component]) - Fraction.from_float(sun_position[component])
            relative_km_per_s = Fraction.from_float(batch.velocities[offset + component]) - Fraction.from_float(sun_velocity[component])
            position.append(_round_fraction(relative_km * 1000 / AU_METRES_EXACT))
            velocity.append(_round_fraction(relative_km_per_s * 1000 * SECONDS_PER_JULIAN_YEAR_EXACT / AU_METRES_EXACT))
        result.append((tuple(position), tuple(velocity)))
    if any(item.hex() != "0x0.0p+0" for vector in result[0] for item in vector):
        raise BridgeError("Sun-primary source cross-binding does not retain canonical Sun zeros")
    return tuple(result)


def _massive_rows(initial) -> list[dict[str, object]]:
    primary_relative = _source_massive_sun_primary(initial)
    rows: list[dict[str, object]] = []
    for index, (body_id, naif_id) in enumerate(zip(initial.body_ids, initial.naif_ids, strict=True)):
        offset = index * 3
        rows.append({
            "body_id": body_id,
            "naif_id": naif_id,
            "massive": True,
            "position_m_hex": [value.hex() for value in initial.positions[offset:offset + 3]],
            "velocity_m_per_s_hex": [value.hex() for value in initial.velocities[offset:offset + 3]],
            "gravitational_parameter_m3_per_s2_hex": initial.gravitational_parameters[index].hex(),
            "sun_primary_position_AU_hex": [value.hex() for value in primary_relative[index][0]],
            "sun_primary_velocity_AU_per_julian_year_hex": [value.hex() for value in primary_relative[index][1]],
        })
    return rows


def _massive_sun_primary_crosscheck(initial, massive: list[dict[str, object]]) -> dict[str, object]:
    batch = initial.preparation_receipt.source_execution.primary_lane.state_batch
    maximum_position_m = 0.0
    maximum_velocity_m_per_s = 0.0
    maximum_operational_position_m = 0.0
    maximum_operational_velocity_m_per_s = 0.0
    for index, row in enumerate(massive):
        offset = index * 3
        for component in range(3):
            exact_relative_position_m = _round_fraction(
                (Fraction.from_float(batch.positions[offset + component]) - Fraction.from_float(batch.positions[component])) * 1000
            )
            exact_relative_velocity_m_per_s = _round_fraction(
                (Fraction.from_float(batch.velocities[offset + component]) - Fraction.from_float(batch.velocities[component])) * 1000
            )
            retained_position_AU = float.fromhex(row["sun_primary_position_AU_hex"][component])
            retained_velocity_AUy = float.fromhex(row["sun_primary_velocity_AU_per_julian_year_hex"][component])
            reconstructed_position_m = _position_au_to_metres(retained_position_AU)
            reconstructed_velocity_m_per_s = _velocity_auy_to_metres_per_second(retained_velocity_AUy)
            operational_relative_position_m = _round_fraction(
                Fraction.from_float(initial.positions[offset + component]) - Fraction.from_float(initial.positions[component])
            )
            operational_relative_velocity_m_per_s = _round_fraction(
                Fraction.from_float(initial.velocities[offset + component]) - Fraction.from_float(initial.velocities[component])
            )
            maximum_position_m = max(maximum_position_m, abs(reconstructed_position_m - exact_relative_position_m))
            maximum_velocity_m_per_s = max(maximum_velocity_m_per_s, abs(reconstructed_velocity_m_per_s - exact_relative_velocity_m_per_s))
            maximum_operational_position_m = max(maximum_operational_position_m, abs(operational_relative_position_m - exact_relative_position_m))
            maximum_operational_velocity_m_per_s = max(maximum_operational_velocity_m_per_s, abs(operational_relative_velocity_m_per_s - exact_relative_velocity_m_per_s))
    position_gate = 0.01
    velocity_gate = 1.0e-8
    if max(maximum_position_m, maximum_operational_position_m) > position_gate or max(maximum_velocity_m_per_s, maximum_operational_velocity_m_per_s) > velocity_gate:
        raise BridgeError("massive Sun-primary cross-binding exceeded its frozen roundoff gate")
    payload = bytearray()
    for row in massive:
        for key in ("sun_primary_position_AU_hex", "sun_primary_velocity_AU_per_julian_year_hex"):
            payload.extend(struct.pack(">3d", *(float.fromhex(item) for item in row[key])))
    return {
        "source_primary_relative_payload_sha256": _sha256(BINARY_DOMAIN + bytes(payload)),
        "maximum_AU_roundtrip_position_error_m_hex": maximum_position_m.hex(),
        "maximum_AUy_roundtrip_velocity_error_m_per_s_hex": maximum_velocity_m_per_s.hex(),
        "maximum_operational_centroid_state_difference_position_error_m_hex": maximum_operational_position_m.hex(),
        "maximum_operational_centroid_state_difference_velocity_error_m_per_s_hex": maximum_operational_velocity_m_per_s.hex(),
        "position_gate_m_hex": position_gate.hex(),
        "velocity_gate_m_per_s_hex": velocity_gate.hex(),
        "all_eleven_pass": True,
    }


def _tno_row(source: Mapping[str, Any], initial) -> dict[str, object]:
    position_ecliptic = tuple(float.fromhex(item) for item in source["position_AU_hex"])
    velocity_ecliptic = tuple(float.fromhex(item) for item in source["velocity_AU_per_julian_year_hex"])
    position_j2000 = _rotate_ecliptic_to_j2000(position_ecliptic)
    velocity_j2000 = _rotate_ecliptic_to_j2000(velocity_ecliptic)
    position_si = tuple(_position_au_to_metres(item) for item in position_j2000)
    velocity_si = tuple(_velocity_auy_to_metres_per_second(item) for item in velocity_j2000)
    sun_position, sun_velocity = _source_sun_values(initial)
    centroid_position = tuple(_fraction_pair(pair) for pair in initial.preparation_receipt.centroid_position_exact_pairs)
    centroid_velocity = tuple(_fraction_pair(pair) for pair in initial.preparation_receipt.centroid_velocity_exact_pairs)
    centered_position = tuple(_round_fraction(Fraction.from_float(position_si[i]) + Fraction.from_float(sun_position[i]) - centroid_position[i]) for i in range(3))
    centered_velocity = tuple(_round_fraction(Fraction.from_float(velocity_si[i]) + Fraction.from_float(sun_velocity[i]) - centroid_velocity[i]) for i in range(3))
    return {
        "object_id": source["object_id"],
        "spk_id": source["spk_id"],
        "sample_kind": source["sample_kind"],
        "clone_index": source["clone_index"],
        "native_vector_sha256": source["native_vector_sha256"],
        "massive": False,
        "gravitational_parameter_m3_per_s2_hex": "0x0.0p+0",
        "heliocentric_j2000_position_AU_hex": [item.hex() for item in position_j2000],
        "heliocentric_j2000_velocity_AU_per_julian_year_hex": [item.hex() for item in velocity_j2000],
        "resolved11_barycenter_position_m_hex": [item.hex() for item in centered_position],
        "resolved11_barycenter_velocity_m_per_s_hex": [item.hex() for item in centered_velocity],
    }


def _batch_binary(rows: list[dict[str, object]], massive: list[dict[str, object]]) -> bytes:
    payload = bytearray()
    for row in massive:
        payload.extend(struct.pack(">q", row["naif_id"]))
        for key in ("position_m_hex", "velocity_m_per_s_hex"):
            payload.extend(struct.pack(">3d", *(float.fromhex(item) for item in row[key])))
        payload.extend(struct.pack(">d", float.fromhex(row["gravitational_parameter_m3_per_s2_hex"])))
        for key in ("sun_primary_position_AU_hex", "sun_primary_velocity_AU_per_julian_year_hex"):
            payload.extend(struct.pack(">3d", *(float.fromhex(item) for item in row[key])))
    for row in rows:
        payload.extend(struct.pack(">q", row["spk_id"]))
        clone = -1 if row["clone_index"] is None else row["clone_index"]
        payload.extend(struct.pack(">i", clone))
        for key in ("heliocentric_j2000_position_AU_hex", "heliocentric_j2000_velocity_AU_per_julian_year_hex", "resolved11_barycenter_position_m_hex", "resolved11_barycenter_velocity_m_per_s_hex"):
            payload.extend(struct.pack(">3d", *(float.fromhex(item) for item in row[key])))
        payload.extend(struct.pack(">d", 0.0))
    return bytes(payload)


def _roundtrip_summary(selected: list[dict[str, Any]], tnos: list[dict[str, object]], initial) -> dict[str, object]:
    max_position = 0.0
    max_velocity = 0.0
    max_operational_position_m = 0.0
    max_operational_velocity_m_per_s = 0.0
    sun_position, sun_velocity = _source_sun_values(initial)
    centroid_position = tuple(_fraction_pair(pair) for pair in initial.preparation_receipt.centroid_position_exact_pairs)
    centroid_velocity = tuple(_fraction_pair(pair) for pair in initial.preparation_receipt.centroid_velocity_exact_pairs)
    for source, row in zip(selected, tnos, strict=True):
        centered_position = tuple(float.fromhex(item) for item in row["resolved11_barycenter_position_m_hex"])
        centered_velocity = tuple(float.fromhex(item) for item in row["resolved11_barycenter_velocity_m_per_s_hex"])
        recovered_j2000_position = tuple(_round_fraction((Fraction.from_float(centered_position[i]) - Fraction.from_float(sun_position[i]) + centroid_position[i]) / AU_METRES_EXACT) for i in range(3))
        recovered_j2000_velocity = tuple(_round_fraction((Fraction.from_float(centered_velocity[i]) - Fraction.from_float(sun_velocity[i]) + centroid_velocity[i]) * SECONDS_PER_JULIAN_YEAR_EXACT / AU_METRES_EXACT) for i in range(3))
        recovered_position = _inverse_rotate_j2000_to_ecliptic(recovered_j2000_position)
        recovered_velocity = _inverse_rotate_j2000_to_ecliptic(recovered_j2000_velocity)
        expected_position = tuple(float.fromhex(item) for item in source["position_AU_hex"])
        expected_velocity = tuple(float.fromhex(item) for item in source["velocity_AU_per_julian_year_hex"])
        max_position = max(max_position, *(abs(a - b) for a, b in zip(recovered_position, expected_position, strict=True)))
        max_velocity = max(max_velocity, *(abs(a - b) for a, b in zip(recovered_velocity, expected_velocity, strict=True)))
        expected_j2000_position_m = tuple(_position_au_to_metres(item) for item in _rotate_ecliptic_to_j2000(expected_position))
        expected_j2000_velocity_m_per_s = tuple(_velocity_auy_to_metres_per_second(item) for item in _rotate_ecliptic_to_j2000(expected_velocity))
        for component in range(3):
            operational_position = _round_fraction(
                Fraction.from_float(centered_position[component]) - Fraction.from_float(initial.positions[component])
            )
            operational_velocity = _round_fraction(
                Fraction.from_float(centered_velocity[component]) - Fraction.from_float(initial.velocities[component])
            )
            max_operational_position_m = max(max_operational_position_m, abs(operational_position - expected_j2000_position_m[component]))
            max_operational_velocity_m_per_s = max(max_operational_velocity_m_per_s, abs(operational_velocity - expected_j2000_velocity_m_per_s[component]))
    if max_position > 2.0e-13 or max_velocity > 2.0e-14:
        raise BridgeError("bridge inverse roundtrip exceeded its frozen numerical gate")
    if max_operational_position_m > 0.01 or max_operational_velocity_m_per_s > 1.0e-8:
        raise BridgeError("TNO operational-SI/Sun-primary cross-binding exceeded its frozen roundoff gate")
    return {
        "maximum_position_component_error_AU_hex": max_position.hex(),
        "maximum_velocity_component_error_AU_per_julian_year_hex": max_velocity.hex(),
        "position_gate_AU_hex": (2.0e-13).hex(),
        "velocity_gate_AU_per_julian_year_hex": (2.0e-14).hex(),
        "maximum_operational_Sun_difference_position_error_m_hex": max_operational_position_m.hex(),
        "maximum_operational_Sun_difference_velocity_error_m_per_s_hex": max_operational_velocity_m_per_s.hex(),
        "operational_position_gate_m_hex": (0.01).hex(),
        "operational_velocity_gate_m_per_s_hex": (1.0e-8).hex(),
        "all_nine_pass": True,
    }


def build_bridge(protocol: Mapping[str, Any], receipts: Mapping[str, Any], stage0: Mapping[str, Any]) -> dict[str, object]:
    initials = validate_receipts(receipts, protocol)
    groups = _selected_stage0_rows(stage0)
    batches: list[dict[str, object]] = []
    global_payload = bytearray()
    for (object_id, epoch_jd, selected), initial in zip(groups, initials, strict=True):
        if initial.naif_ids != RESOLVED11_NAIF_IDS:
            raise BridgeError("prepared massive-body roster differs")
        massive = _massive_rows(initial)
        tnos = [_tno_row(source, initial) for source in selected]
        binary = _batch_binary(tnos, massive)
        global_payload.extend(binary)
        batch: dict[str, object] = {
            "object_id": object_id,
            "covariance_epoch_jd_tdb_hex": epoch_jd.hex(),
            "spice_et_seconds": _et_seconds(epoch_jd.hex()),
            "preparation_initial_state_content_sha256": initial.content_sha256,
            "massive_body_count": 11,
            "massive_bodies": massive,
            "massive_sun_primary_crosscheck": _massive_sun_primary_crosscheck(initial, massive),
            "tno_state_count": 9,
            "tnos": tnos,
            "tno_excluded_from_centroid": True,
            "roundtrip": _roundtrip_summary(selected, tnos, initial),
            "binary_payload_byte_length": len(binary),
            "binary_payload_sha256": _sha256(BINARY_DOMAIN + binary),
        }
        batch["content_sha256"] = _content_sha256(BATCH_DOMAIN, batch)
        batches.append(batch)
    claims = dict(protocol["claims"])
    artifact: dict[str, object] = {
        "schema": BRIDGE_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "TWELVE_NATIVE_EPOCH_PREPARED_BATCHES_NO_DYNAMICS",
        "claim_ceiling": protocol["claim_ceiling"],
        "protocol_content_sha256": protocol["content_sha256"],
        "receipts_content_sha256": receipts["content_sha256"],
        "batch_count": 12,
        "massive_state_rows_total": 132,
        "tno_state_rows_total": 108,
        "source_stage0_state_count": 1_548,
        "batches": batches,
        "binary_payload_byte_length": len(global_payload),
        "binary_payload_sha256": _sha256(BINARY_DOMAIN + bytes(global_payload)),
        "claims": claims,
        "execution_authorized": False,
        "outcomes_generated": False,
    }
    artifact["content_sha256"] = _content_sha256(BRIDGE_DOMAIN, artifact)
    return artifact


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise BridgeError(f"artifact path is not create-only: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def materialize(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise BridgeError("output root must be a pre-staged real directory")
    if {entry.name for entry in output_root.iterdir()} != {"external"}:
        raise BridgeError("pre-materialization root must contain only the retained external bundle")
    protocol = build_protocol(output_root)
    stage0 = _validate_stage0()
    receipts = execute_receipts(protocol, stage0, output_root)
    bridge = build_bridge(protocol, receipts, stage0)
    _write_create_only(output_root / "PROTOCOL.json", _canonical_json(protocol))
    _write_create_only(output_root / "RECEIPTS.json", _canonical_json(receipts))
    _write_create_only(output_root / "BRIDGE.json", _canonical_json(bridge))
    validate_output(output_root)
    return bridge


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise BridgeError("output root must be a real directory")
    if {entry.name for entry in output_root.iterdir()} != {"external", "PROTOCOL.json", "RECEIPTS.json", "BRIDGE.json"}:
        raise BridgeError("output root roster differs")
    protocol = _load_canonical(output_root / "PROTOCOL.json")
    expected_protocol = build_protocol(output_root)
    if not _exact_equal(protocol, expected_protocol):
        raise BridgeError("protocol differs from exact reconstruction")
    receipts = _load_canonical(output_root / "RECEIPTS.json")
    validate_receipts(receipts, protocol)
    stage0 = _validate_stage0()
    actual = _load_canonical(output_root / "BRIDGE.json")
    expected = build_bridge(protocol, receipts, stage0)
    if not _exact_equal(actual, expected):
        raise BridgeError("bridge artifact differs from exact reconstruction")
    _require(actual.get("content_sha256"), _content_sha256(BRIDGE_DOMAIN, actual), "bridge content digest")
    for claim in ("de440_de441_model_mismatch_resolved", "source_tile_created", "dynamics_executed", "one_thousand_year_holdout_executed", "paired_de440_de441_sensitivity_executed", "gpu_executed", "dynamical_classifications_generated", "long_horizon_execution_authorized", "observational_likelihood_evaluated", "population_inference", "planet_nine_evidence", "science_result"):
        _require(actual["claims"].get(claim), False, f"bridge claim {claim}")
    _require(actual.get("execution_authorized"), False, "bridge execution claim")
    _require(actual.get("outcomes_generated"), False, "bridge outcome claim")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true", help="execute only the twelve source replays and prepare coordinate batches")
    group.add_argument("--validate-only", action="store_true", help="offline-validate retained content-addressed bridge artifacts")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = materialize(args.output_root) if args.prepare_only else validate_output(args.output_root)
    except BridgeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps({"schema": artifact["schema"], "content_sha256": artifact["content_sha256"], "batch_count": artifact["batch_count"], "tno_state_rows_total": artifact["tno_state_rows_total"], "execution_authorized": artifact["execution_authorized"], "outcomes_generated": artifact["outcomes_generated"]}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
