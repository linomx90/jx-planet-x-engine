#!/usr/bin/env python3
"""V3 relocation-safe successor for the bounded full-129 TNO GPU smoke.

The numerical implementation and every physics gate remain delegated to the
exact preregistered V2/V1 implementation.  V3 changes only custody around the
pathful native source-builder receipt: the live receipt is retained in full,
the native binary is required to live under this checkout, and equality with
the historical V3 execution is proved after replacing that one absolute path.
"""

from __future__ import annotations

import contextvars
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v3.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v3.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v3"

V2_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_gpu_smoke_v2.py"
V2_BYTES = 23_821
V2_SHA256 = "99c760443c2bfbb38ea05ee9e47203017ad6e3ce0c6a80697f41c1ccb6bf6bae"
V2_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_full128_gpu_smoke_v2.py"
V2_TEST_BYTES = 10_817
V2_TEST_SHA256 = "e81a6b609b1c897dcd71c87681f29a50e64d4a9fab8f5938fbce942c7411dc1f"
V2_ROOT = PROJECT_ROOT / "runs" / "tno_10my_full128_gpu_smoke_v2"
V2_PROTOCOL_BYTES = 20_134
V2_PROTOCOL_SHA256 = "3233177b11ba8cb18ae9ddb7a3075b8797e8d92127556336d297d1d87d35aac5"
V2_PROTOCOL_CONTENT_SHA256 = "b7acad6826962e1f20241bac59de42c5e5ce793a461d502dafbf12d7de0a0941"
V2_BRANCH = "research/tno-10my-reduced5-execution-v2-preregistered-20261011"
V2_COMMIT = "90ea10e4d9c95f2a8eb49a834636ed1ddbb3a4fa"
V2_TAG = "tno-10my-reduced5-execution-v2-preregistered-20261011"
V2_TAG_OBJECT = "0694db84113b3cc5263c573c3346cb36a4f8c451"

PREREGISTRATION_BRANCH = "research/tno-10my-reduced5-execution-v3-preregistered-20261011"
PREREGISTRATION_TAG = "tno-10my-reduced5-execution-v3-preregistered-20261011"

PROTOCOL_SCHEMA = "jx.tno.10my-full128-gpu-smoke.protocol.v3"
SEGMENT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.segment.v3"
RUNTIME_PRE_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-pre.v3"
RUNTIME_POST_SCHEMA = "jx.tno.10my-full128-gpu-smoke.runtime-post.v3"
RESULT_SCHEMA = "jx.tno.10my-full128-gpu-smoke.result.v3"
GIT_CUSTODY_SCHEMA = "jx.tno.10my-full128-gpu-smoke.git-custody.v3"
PROTOCOL_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.protocol.v3\0"
SEGMENT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.segment.v3\0"
RUNTIME_PRE_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-pre.v3\0"
RUNTIME_POST_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.runtime-post.v3\0"
RESULT_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.result.v3\0"
GIT_CUSTODY_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.git-custody.v3\0"

RELOCATION_SCHEMA = "jx.tno.10my-full128-gpu-smoke.source-build-relocation.v3"
RELOCATION_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.source-build-relocation.v3\0"
CPU_PROBE_SCHEMA = "jx.tno.10my-full128-gpu-smoke.cpu-relocation-probe.v3"
CPU_PROBE_DOMAIN = b"jx.tno.10my-full128-gpu-smoke.cpu-relocation-probe.v3\0"
RELOCATED_FIELD_TOKEN = "source_map_runtime_identity.extension_binary.path"
LOCAL_THREAT_BOUNDARY = (
    "NO_HOSTILE_CODE_IN_FRESH_CHILD_AND_NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"
)

NATIVE_MODULE = "jxplanetx._wisdom_holman_cpu_v3"
NATIVE_RELATIVE_PATH = "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
NATIVE_PATH = PROJECT_ROOT / NATIVE_RELATIVE_PATH
NATIVE_BYTES = 42_520
NATIVE_SHA256 = "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3"
HISTORICAL_NATIVE_PATH = (
    "/home/linoavila/Documents/Codex/2026-10-02/"
    "referenced-chatgpt-conversation-this-is-an/work/jx-cuda-tracer/"
    "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
)
HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256 = (
    "6cb05c6891023e2a65bbec605746cecc67e3b026526e275a1a4bdc24be21b70d"
)
HISTORICAL_PATH_NEUTRAL_RUNTIME_SHA256 = (
    "ab6f07706b1d246c363e0f783251230282adbc74e76a8130fae1da23f7f6a100"
)
EXPECTED_SOURCE_BUILD_DIGESTS = {
    "input_content_sha256": "c4ab0cdd5b4ce459b5cdd57f9f23871354a2068ce6396ffbe361b464c4aced08",
    "map_result_content_sha256": "d2bd674e9c49524cb49467b88d9d5a32f6d4cdd689ee5d6ad42fdf8d0dccd5a9",
    "map_verification_content_sha256": "a7a34f1cb38d3e130337da3cfec927e9780108f19980ca90209e485882ca1934",
    "runtime_identity_sha256": HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256,
}


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _load_v2() -> types.ModuleType:
    payload = V2_PATH.read_bytes()
    if len(payload) != V2_BYTES or _sha256(payload) != V2_SHA256:
        raise RuntimeError("frozen V2 source binding differs")
    name = f"_jx_tno_full128_smoke_v3_v2_{os.getpid()}_{id(object())}"
    module = types.ModuleType(name)
    module.__file__ = str(V2_PATH)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[name] = module
    try:
        exec(
            compile(payload, str(V2_PATH), "exec", dont_inherit=True, optimize=sys.flags.optimize),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_V2 = _load_v2()
_INNER = _V2._IMPL
Full128SmokeError = _INNER.Full128SmokeError

_NAMESPACE_REPLACEMENTS = {
    "SCRIPT_PATH": SCRIPT_PATH,
    "TEST_PATH": TEST_PATH,
    "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
    "PREREGISTRATION_BRANCH": PREREGISTRATION_BRANCH,
    "PREREGISTRATION_TAG": PREREGISTRATION_TAG,
    "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
    "SEGMENT_SCHEMA": SEGMENT_SCHEMA,
    "RUNTIME_PRE_SCHEMA": RUNTIME_PRE_SCHEMA,
    "RUNTIME_POST_SCHEMA": RUNTIME_POST_SCHEMA,
    "RESULT_SCHEMA": RESULT_SCHEMA,
    "GIT_CUSTODY_SCHEMA": GIT_CUSTODY_SCHEMA,
    "PROTOCOL_DOMAIN": PROTOCOL_DOMAIN,
    "SEGMENT_DOMAIN": SEGMENT_DOMAIN,
    "RUNTIME_PRE_DOMAIN": RUNTIME_PRE_DOMAIN,
    "RUNTIME_POST_DOMAIN": RUNTIME_POST_DOMAIN,
    "RESULT_DOMAIN": RESULT_DOMAIN,
    "GIT_CUSTODY_DOMAIN": GIT_CUSTODY_DOMAIN,
}
for _module in (_V2, _INNER):
    for _key, _value in _NAMESPACE_REPLACEMENTS.items():
        setattr(_module, _key, _value)

_V2_BUILD_PROTOCOL = _V2.build_protocol
_ORIGINAL_SOURCE_BUILD_IDENTITY = _INNER._source_build_identity
_ORIGINAL_RUN_FULL129_LANE = _INNER._run_full129_lane
_ORIGINAL_BUILD_RESULT_AND_ARTIFACTS = _INNER._build_result_and_artifacts
_ORIGINAL_VALIDATE_OUTPUT_SNAPSHOT = _INNER._validate_output_snapshot
_ORIGINAL_VALIDATE_OUTPUT = _INNER.validate_output
_RELOCATION_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("jx_tno_full128_v3_source_relocation", default=None)
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_file(path: Path, size: int, digest: str, label: str) -> dict[str, object]:
    payload = _INNER._read_regular(path, label, size)
    if _sha256(payload) != digest:
        raise Full128SmokeError(f"{label} SHA-256 differs")
    return {
        "label": label,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": size,
        "sha256": digest,
    }


def _predecessor_v2() -> dict[str, Any]:
    source = _exact_file(V2_PATH, V2_BYTES, V2_SHA256, "V2 predecessor source")
    test = _exact_file(V2_TEST_PATH, V2_TEST_BYTES, V2_TEST_SHA256, "V2 predecessor test")
    with _INNER._held_directory(
        V2_ROOT, "V2 predecessor root", expected_roster={"PROTOCOL.json"},
    ) as root_fd:
        payload = _INNER._read_regular_at(
            root_fd, "PROTOCOL.json", "V2 predecessor protocol", V2_PROTOCOL_BYTES,
        )
    protocol = _INNER._strict_json(payload)
    claims = protocol.get("claims")
    if (
        payload != _canonical_json(protocol)
        or _sha256(payload) != V2_PROTOCOL_SHA256
        or protocol.get("content_sha256") != V2_PROTOCOL_CONTENT_SHA256
        or protocol.get("schema") != "jx.tno.10my-full128-gpu-smoke.protocol.v2"
        or type(claims) is not dict or not claims
        or any(type(value) is not bool or value for value in claims.values())
    ):
        raise Full128SmokeError("V2 predecessor protocol custody differs")
    return {
        "source": source,
        "test": test,
        "protocol": {
            "relative_path": "runs/tno_10my_full128_gpu_smoke_v2/PROTOCOL.json",
            "bytes": V2_PROTOCOL_BYTES,
            "sha256": V2_PROTOCOL_SHA256,
            "content_sha256": V2_PROTOCOL_CONTENT_SHA256,
        },
        "branch": V2_BRANCH,
        "commit": V2_COMMIT,
        "annotated_tag": V2_TAG,
        "tag_object": V2_TAG_OBJECT,
        "root_roster": ["PROTOCOL.json"],
        "failed_attempt": {
            "attempt_count": 1,
            "operator_session_observed": True,
            "observation_custody": (
                "OPERATOR_SESSION_AND_EXACT_SOURCE_CODE_ORDER_NOT_RETAINED_BY_V2_ROOT"
            ),
            "traceback_authenticated_by_retained_v2_artifact": False,
            "failure_class": "Full128SmokeError",
            "failure_message": "source construction receipts differ from retained V3",
            "failure_phase": (
                "BATCH0_AFTER_EXACT_H64_MASSIVE_SOURCE_REPLAY_BEFORE_LIFECYCLE_INITIALIZE"
            ),
            "runtime_preflight_and_setup_reached": True,
            "massive_source_step_count": 64_000,
            "massive_source_build_completed": True,
            "massive_source_payload_gm_tile_and_invariants_matched": True,
            "lifecycle_initialize_call_count": 0,
            "tracer_advance_call_count": 0,
            "completed_segment_count": 0,
            "tracer_checkpoint_count": 0,
            "result_published": False,
            "scientific_outcome_generated": False,
            "repair_scope": "ABSOLUTE_NATIVE_EXTENSION_PATH_RELOCATION_CUSTODY_ONLY",
        },
    }


def build_protocol() -> dict[str, Any]:
    value = _V2_BUILD_PROTOCOL()
    value["status"] = "PREREGISTERED_EXACT_BATCH0_FULL129_1KYR_GPU_SMOKE_V3_NOT_EXECUTED"
    value["predecessor_v2"] = _predecessor_v2()
    value["source_build_runtime_relocation"] = {
        "schema": RELOCATION_SCHEMA,
        "relocated_field_token": RELOCATED_FIELD_TOKEN,
        "live_native_relative_path": NATIVE_RELATIVE_PATH,
        "historical_native_absolute_path": HISTORICAL_NATIVE_PATH,
        "native_bytes": NATIVE_BYTES,
        "native_sha256": NATIVE_SHA256,
        "native_regular_non_symlink_single_link_required": True,
        "native_ancestry_under_current_project_root_must_be_non_symlink": True,
        "full_live_pathful_runtime_identity_retained": True,
        "historical_path_rebased_runtime_sha256": HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256,
        "historical_path_neutral_runtime_sha256": HISTORICAL_PATH_NEUTRAL_RUNTIME_SHA256,
        "only_absolute_path_field_allowed_to_differ": RELOCATED_FIELD_TOKEN,
        "runtime_post_critical_native_module_crossbinding_required": True,
        "physics_force_step_grid_threshold_or_gate_changed": False,
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
    }
    value["content_sha256"] = _INNER._domain_hash(PROTOCOL_DOMAIN, value)
    return value


def _historical_native_receipt() -> dict[str, object]:
    payload = _INNER._read_regular(
        _INNER.V3_ROOT / "RUNTIME.json", "historical V3 runtime", None,
    )
    value = _INNER._strict_json(payload)
    if (
        payload != _canonical_json(value)
        or _sha256(payload) != _INNER.V3_RUNTIME_SHA256
        or value.get("content_sha256") != _V2.HISTORICAL_V3_RUNTIME_CONTENT_SHA256
    ):
        raise Full128SmokeError("historical V3 runtime custody differs")
    rows = value.get("critical_jx_modules")
    matches = (
        [row for row in rows if type(row) is dict and row.get("module") == NATIVE_MODULE]
        if type(rows) is list else []
    )
    if len(matches) != 1:
        raise Full128SmokeError("historical V3 native-module receipt roster differs")
    row = matches[0]
    if (
        set(row) != {"module", "path", "bytes", "sha256"}
        or row.get("path") != HISTORICAL_NATIVE_PATH
        or row.get("bytes") != NATIVE_BYTES or type(row.get("bytes")) is not int
        or row.get("sha256") != NATIVE_SHA256
    ):
        raise Full128SmokeError("historical V3 native-module receipt differs")
    return dict(row)


def _read_live_native_exact() -> dict[str, object]:
    expected = Path(os.path.abspath(os.fspath(NATIVE_PATH)))
    ancestry = (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "jxplanetx")
    for item in ancestry:
        metadata = os.lstat(item)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise Full128SmokeError("live native binary ancestry differs")
    before = os.lstat(expected)
    if (
        stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1 or before.st_size != NATIVE_BYTES
    ):
        raise Full128SmokeError("live native binary file identity differs")
    descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_nlink != 1 or opened.st_size != NATIVE_BYTES
        ):
            raise Full128SmokeError("live native binary open identity differs")
        chunks: list[bytes] = []
        remaining = NATIVE_BYTES
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise Full128SmokeError("live native binary read was incomplete")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise Full128SmokeError("live native binary grew during read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(expected)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_nlink,
    )
    if identity(before) != identity(opened) or identity(opened) != identity(after) or identity(after) != identity(final):
        raise Full128SmokeError("live native binary changed during read")
    payload = b"".join(chunks)
    if _sha256(payload) != NATIVE_SHA256:
        raise Full128SmokeError("live native binary SHA-256 differs")
    return {
        "path": str(expected), "size_bytes": NATIVE_BYTES, "sha256": NATIVE_SHA256,
    }


def _extension_receipt(identity: Mapping[str, Any]) -> dict[str, object]:
    source_map = identity.get("source_map_runtime_identity")
    extension = source_map.get("extension_binary") if type(source_map) is dict else None
    if (
        type(extension) is not dict
        or set(extension) != {"path", "size_bytes", "sha256"}
        or type(extension.get("path")) is not str
        or type(extension.get("size_bytes")) is not int
        or extension.get("size_bytes") <= 0
        or not _is_sha256(extension.get("sha256"))
    ):
        raise Full128SmokeError("source-build extension-binary receipt differs")
    return dict(extension)


def _validate_relocation_identity(
    live_identity: Mapping[str, Any], *, live_extension: Mapping[str, Any],
    historical_extension: Mapping[str, Any], expected_neutral_sha256: str,
    expected_rebased_sha256: str,
) -> dict[str, Any]:
    """Validate path-only relocation arithmetic without building a source path."""
    if (
        type(live_identity) is not dict
        or type(live_extension) is not dict
        or type(historical_extension) is not dict
        or not _is_sha256(expected_neutral_sha256)
        or not _is_sha256(expected_rebased_sha256)
    ):
        raise Full128SmokeError("source-build relocation inputs differ")
    expected_live = dict(live_extension)
    supplied_live = _extension_receipt(live_identity)
    if supplied_live != expected_live:
        raise Full128SmokeError("source-build native receipt differs")
    historical_native = dict(historical_extension)
    if set(historical_native) != {"path", "size_bytes", "sha256"}:
        raise Full128SmokeError("historical native extension receipt shape differs")
    if (
        type(historical_native.get("path")) is not str
        or type(historical_native.get("size_bytes")) is not int
        or not _is_sha256(historical_native.get("sha256"))
        or historical_native["size_bytes"] != supplied_live["size_bytes"]
        or historical_native["sha256"] != supplied_live["sha256"]
    ):
        raise Full128SmokeError("live and historical native binaries differ")
    live_sha = _sha256(_canonical_json(live_identity))
    neutral = copy.deepcopy(live_identity)
    del neutral["source_map_runtime_identity"]["extension_binary"]["path"]
    neutral_sha = _sha256(_canonical_json(neutral))
    if neutral_sha != expected_neutral_sha256:
        raise Full128SmokeError("path-neutral source-build runtime identity differs")
    rebased = copy.deepcopy(live_identity)
    rebased["source_map_runtime_identity"]["extension_binary"]["path"] = historical_native["path"]
    rebased_sha = _sha256(_canonical_json(rebased))
    if rebased_sha != expected_rebased_sha256:
        raise Full128SmokeError("historically rebased source-build runtime identity differs")
    return {
        "relocated_field_token": RELOCATED_FIELD_TOKEN,
        "live_runtime_identity": live_identity,
        "live_runtime_identity_sha256": live_sha,
        "path_neutral_runtime_identity": neutral,
        "path_neutral_runtime_identity_sha256": neutral_sha,
        "historical_rebased_runtime_identity_sha256": rebased_sha,
        "historical_runtime_identity_sha256": expected_rebased_sha256,
        "live_extension_binary": supplied_live,
        "historical_extension_binary": historical_native,
        "path_only_relocation_verified": True,
    }


def _relocation_base(v3: types.ModuleType, build: object) -> dict[str, Any]:
    """Production-only adapter with non-overridable frozen historical hashes."""
    live_identity = v3._plain(build.runtime_identity)
    if type(live_identity) is not dict:
        raise Full128SmokeError("live source-build runtime identity differs")
    exact_file = _read_live_native_exact()
    historical_native = _historical_native_receipt()
    return _validate_relocation_identity(
        live_identity,
        live_extension=exact_file,
        historical_extension={
            "path": historical_native["path"],
            "size_bytes": historical_native["bytes"],
            "sha256": historical_native["sha256"],
        },
        expected_neutral_sha256=HISTORICAL_PATH_NEUTRAL_RUNTIME_SHA256,
        expected_rebased_sha256=HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256,
    )


def _source_build_identity(v3: types.ModuleType, build: object) -> dict[str, str]:
    live = _ORIGINAL_SOURCE_BUILD_IDENTITY(v3, build)
    expected = _INNER._retained_source_build_identity(_INNER._v3_gate_batch0_lane())
    if expected != EXPECTED_SOURCE_BUILD_DIGESTS:
        raise Full128SmokeError("historical V3 source-build digest roster differs")
    for key in (
        "input_content_sha256", "map_result_content_sha256",
        "map_verification_content_sha256",
    ):
        if live.get(key) != expected[key]:
            raise Full128SmokeError(f"source construction receipt differs: {key}")
    base = _relocation_base(v3, build)
    if live.get("runtime_identity_sha256") != base["live_runtime_identity_sha256"]:
        raise Full128SmokeError("live source-build runtime digest differs")
    _RELOCATION_CONTEXT.set(base)
    return dict(EXPECTED_SOURCE_BUILD_DIGESTS)


def _run_full129_lane(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _RELOCATION_CONTEXT.set(None)
    lane = _ORIGINAL_RUN_FULL129_LANE(*args, **kwargs)
    base = _RELOCATION_CONTEXT.get()
    if type(base) is not dict:
        raise Full128SmokeError("source-build relocation receipt was not produced")
    output = dict(lane)
    output["source_build_runtime_relocation_base"] = base
    return output


def _critical_native_row(runtime_post: Mapping[str, Any]) -> dict[str, object]:
    runtime = runtime_post.get("v3_runtime_identity")
    rows = runtime.get("critical_jx_modules") if type(runtime) is dict else None
    matches = (
        [row for row in rows if type(row) is dict and row.get("module") == NATIVE_MODULE]
        if type(rows) is list else []
    )
    if len(matches) != 1:
        raise Full128SmokeError("runtime-post native-module receipt roster differs")
    row = matches[0]
    if set(row) != {"module", "path", "bytes", "sha256"}:
        raise Full128SmokeError("runtime-post native-module receipt shape differs")
    expected = _read_live_native_exact()
    if row != {
        "module": NATIVE_MODULE, "path": expected["path"],
        "bytes": expected["size_bytes"], "sha256": expected["sha256"],
    }:
        raise Full128SmokeError("runtime-post native-module receipt differs")
    return dict(row)


def _bind_relocation_to_runtime_post(
    base: Mapping[str, Any], runtime_post: Mapping[str, Any],
) -> dict[str, Any]:
    if type(base) is not dict or type(runtime_post) is not dict:
        raise Full128SmokeError("source-build relocation binding shape differs")
    live_extension = base.get("live_extension_binary")
    row = _critical_native_row(runtime_post)
    if (
        type(live_extension) is not dict
        or row["path"] != live_extension.get("path")
        or row["bytes"] != live_extension.get("size_bytes")
        or row["sha256"] != live_extension.get("sha256")
    ):
        raise Full128SmokeError("source-build and runtime-post native receipts differ")
    runtime = runtime_post.get("v3_runtime_identity")
    if type(runtime) is not dict:
        raise Full128SmokeError("runtime-post V3 identity differs")
    runtime_content = runtime.get("content_sha256")
    post_content = runtime_post.get("content_sha256")
    projection = runtime.get("engine_runtime_projection")
    if not _is_sha256(runtime_content) or not _is_sha256(post_content) or type(projection) is not dict:
        raise Full128SmokeError("runtime-post identity hash binding differs")
    value: dict[str, Any] = {
        "schema": RELOCATION_SCHEMA,
        **copy.deepcopy(dict(base)),
        "runtime_post_critical_native_module": row,
        "runtime_post_content_sha256": post_content,
        "runtime_post_v3_identity_content_sha256": runtime_content,
        "runtime_post_engine_projection_sha256": _sha256(_canonical_json(projection)),
        "local_threat_boundary": LOCAL_THREAT_BOUNDARY,
        "content_sha256": "",
    }
    value["content_sha256"] = _INNER._domain_hash(RELOCATION_DOMAIN, value)
    return value


def _validate_relocation_receipt(
    value: object, runtime_post: Mapping[str, Any], *, expected_base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    keys = {
        "schema", "relocated_field_token", "live_runtime_identity",
        "live_runtime_identity_sha256", "path_neutral_runtime_identity",
        "path_neutral_runtime_identity_sha256", "historical_rebased_runtime_identity_sha256",
        "historical_runtime_identity_sha256", "live_extension_binary",
        "historical_extension_binary", "path_only_relocation_verified",
        "runtime_post_critical_native_module", "runtime_post_content_sha256",
        "runtime_post_v3_identity_content_sha256", "runtime_post_engine_projection_sha256",
        "local_threat_boundary", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full128SmokeError("source-build relocation receipt shape differs")
    if (
        value.get("schema") != RELOCATION_SCHEMA
        or value.get("relocated_field_token") != RELOCATED_FIELD_TOKEN
        or value.get("path_only_relocation_verified") is not True
        or value.get("local_threat_boundary") != LOCAL_THREAT_BOUNDARY
        or value.get("content_sha256") != _INNER._domain_hash(RELOCATION_DOMAIN, value)
    ):
        raise Full128SmokeError("source-build relocation receipt identity differs")
    live_identity = value.get("live_runtime_identity")
    neutral = value.get("path_neutral_runtime_identity")
    if type(live_identity) is not dict or type(neutral) is not dict:
        raise Full128SmokeError("source-build relocation runtime projection differs")
    extension = _extension_receipt(live_identity)
    exact = _read_live_native_exact()
    if extension != exact or value.get("live_extension_binary") != exact:
        raise Full128SmokeError("source-build live extension receipt differs")
    rebuilt_neutral = copy.deepcopy(live_identity)
    del rebuilt_neutral["source_map_runtime_identity"]["extension_binary"]["path"]
    rebased = copy.deepcopy(live_identity)
    rebased["source_map_runtime_identity"]["extension_binary"]["path"] = HISTORICAL_NATIVE_PATH
    if (
        neutral != rebuilt_neutral
        or value.get("live_runtime_identity_sha256") != _sha256(_canonical_json(live_identity))
        or value.get("path_neutral_runtime_identity_sha256") != _sha256(_canonical_json(neutral))
        or value.get("path_neutral_runtime_identity_sha256") != HISTORICAL_PATH_NEUTRAL_RUNTIME_SHA256
        or value.get("historical_rebased_runtime_identity_sha256") != _sha256(_canonical_json(rebased))
        or value.get("historical_rebased_runtime_identity_sha256") != HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256
        or value.get("historical_runtime_identity_sha256") != HISTORICAL_SOURCE_BUILD_RUNTIME_SHA256
    ):
        raise Full128SmokeError("source-build relocation projection differs")
    historical = _historical_native_receipt()
    if value.get("historical_extension_binary") != {
        "path": historical["path"], "size_bytes": historical["bytes"],
        "sha256": historical["sha256"],
    }:
        raise Full128SmokeError("source-build historical extension receipt differs")
    expected = _bind_relocation_to_runtime_post(
        dict(expected_base) if expected_base is not None else {
            key: copy.deepcopy(value[key])
            for key in (
                "relocated_field_token", "live_runtime_identity", "live_runtime_identity_sha256",
                "path_neutral_runtime_identity", "path_neutral_runtime_identity_sha256",
                "historical_rebased_runtime_identity_sha256", "historical_runtime_identity_sha256",
                "live_extension_binary", "historical_extension_binary",
                "path_only_relocation_verified",
            )
        },
        runtime_post,
    )
    if _canonical_json(value) != _canonical_json(expected):
        raise Full128SmokeError("source-build relocation runtime-post binding differs")
    return dict(value)


def _build_result_and_artifacts(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], tuple[tuple[str, bytes], ...]]:
    result, artifacts = _ORIGINAL_BUILD_RESULT_AND_ARTIFACTS(*args, **kwargs)
    lane = args[4] if len(args) >= 5 else kwargs.get("lane")
    runtime_post = args[3] if len(args) >= 4 else kwargs.get("runtime_post")
    base = lane.get("source_build_runtime_relocation_base") if type(lane) is dict else None
    if type(base) is not dict or type(runtime_post) is not dict:
        raise Full128SmokeError("source-build relocation result inputs differ")
    value = dict(result)
    value["source_build_runtime_relocation"] = _bind_relocation_to_runtime_post(base, runtime_post)
    value["content_sha256"] = _INNER._domain_hash(RESULT_DOMAIN, value)
    replaced = tuple(
        (name, _canonical_json(value) if name == "RESULT.json" else payload)
        for name, payload in artifacts
    )
    if [name for name, _payload in replaced].count("RESULT.json") != 1:
        raise Full128SmokeError("result artifact roster differs")
    return value, replaced


def _validate_output_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    if type(snapshot) is not dict or set(snapshot) != _INNER.EXPECTED_OUTPUT_ROSTER:
        raise Full128SmokeError("retained output snapshot roster differs")
    result = _INNER._strict_json(snapshot["RESULT.json"])
    post = _INNER._strict_json(snapshot["RUNTIME_POST.json"])
    if snapshot["RESULT.json"] != _canonical_json(result):
        raise Full128SmokeError("retained V3 result is not canonical")
    receipt = result.get("source_build_runtime_relocation") if type(result) is dict else None
    sanitized_result = dict(result)
    if sanitized_result.pop("source_build_runtime_relocation", None) is None:
        raise Full128SmokeError("retained source-build relocation receipt is absent")
    sanitized_result["content_sha256"] = _INNER._domain_hash(RESULT_DOMAIN, sanitized_result)
    sanitized = dict(snapshot)
    sanitized["RESULT.json"] = _canonical_json(sanitized_result)
    _RELOCATION_CONTEXT.set(None)
    base_result = _ORIGINAL_VALIDATE_OUTPUT_SNAPSHOT(sanitized)
    expected_base = _RELOCATION_CONTEXT.get()
    if type(expected_base) is not dict:
        raise Full128SmokeError("offline source-build relocation replay is absent")
    _validate_relocation_receipt(receipt, post, expected_base=expected_base)
    rebuilt = dict(base_result)
    rebuilt["source_build_runtime_relocation"] = receipt
    rebuilt["content_sha256"] = _INNER._domain_hash(RESULT_DOMAIN, rebuilt)
    if _canonical_json(rebuilt) != _canonical_json(result):
        raise Full128SmokeError("retained V3 result relocation binding differs")
    return result


# Production calls resolve these names in the inner V1 globals.  Replacing the
# inner objects, not merely the public V3 aliases, is part of the repair.
_INNER.build_protocol = build_protocol
_INNER._source_build_identity = _source_build_identity
_INNER._run_full129_lane = _run_full129_lane
_INNER._build_result_and_artifacts = _build_result_and_artifacts
_INNER._validate_output_snapshot = _validate_output_snapshot


def validate_protocol(value: Mapping[str, Any]) -> str:
    return _INNER.validate_protocol(value)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _INNER.prepare(output_root)


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _INNER.validate_prepared(output_root)


def _jx_module_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name for name in sys.modules
            if name == "jxplanetx" or name.startswith("jxplanetx.")
        )
    )


@contextmanager
def _current_project_jx_import_scope() -> Iterator[None]:
    if _jx_module_names():
        raise Full128SmokeError("public offline validation requires no preloaded JX modules")
    previous_path = list(sys.path)
    current_src = str(PROJECT_ROOT / "src")
    current_root = str(PROJECT_ROOT)
    sys.path[:] = [current_src, current_root, *(
        item for item in previous_path if item not in {current_src, current_root}
    )]
    try:
        yield
        source_root = PROJECT_ROOT / "src" / "jxplanetx"
        names = _jx_module_names()
        if not names:
            raise Full128SmokeError("public offline validation imported no JX modules")
        for name in names:
            module = sys.modules.get(name)
            path_token = getattr(module, "__file__", None)
            if type(module) is not types.ModuleType or type(path_token) is not str:
                raise Full128SmokeError("public offline validation JX module identity differs")
            path = Path(path_token).resolve(strict=True)
            if path != source_root / "__init__.py" and source_root not in path.parents:
                raise Full128SmokeError("public offline validation selected a foreign JX module")
    finally:
        for name in reversed(_jx_module_names()):
            sys.modules.pop(name, None)
        sys.path[:] = previous_path


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    with _current_project_jx_import_scope():
        return _ORIGINAL_VALIDATE_OUTPUT(output_root)


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, externally_preregistered: bool = False,
    prereg_commit: str = "", prereg_tag_object: str = "",
) -> dict[str, Any]:
    return _INNER.execute(
        output_root, externally_preregistered=externally_preregistered,
        prereg_commit=prereg_commit, prereg_tag_object=prereg_tag_object,
    )


def _production_child_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def build_cpu_relocation_probe() -> dict[str, Any]:
    if _jx_module_names():
        raise Full128SmokeError("CPU relocation probe requires no preloaded JX modules")
    if any(
        name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx.")
        for name in sys.modules
    ):
        raise Full128SmokeError("CPU relocation probe must not preload CuPy")
    fast_module = importlib.import_module("jxplanetx.fast_wisdom_holman")
    identity_function = getattr(fast_module, "fast_wisdom_holman_runtime_identity", None)
    if type(fast_module) is not types.ModuleType or type(identity_function) is not types.FunctionType:
        raise Full128SmokeError("CPU import probe runtime-identity callable differs")
    source_map_identity = identity_function()
    if type(source_map_identity) is not dict:
        raise Full128SmokeError("CPU import probe native runtime identity differs")
    if any(
        name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx.")
        for name in sys.modules
    ):
        raise Full128SmokeError("CPU relocation probe imported CuPy")
    native_module = sys.modules.get(NATIVE_MODULE)
    native_module_path = str(Path(str(getattr(native_module, "__file__", ""))).resolve(strict=True))
    fast_module_path = str(Path(str(getattr(fast_module, "__file__", ""))).resolve(strict=True))
    exact_native = _read_live_native_exact()
    if _extension_receipt({"source_map_runtime_identity": source_map_identity}) != exact_native:
        raise Full128SmokeError("CPU import probe native extension receipt differs")
    value: dict[str, Any] = {
        "schema": CPU_PROBE_SCHEMA,
        "status": "CPU_IMPORT_ONLY_NATIVE_RUNTIME_IDENTITY",
        "project_root": str(PROJECT_ROOT),
        "production_pythonpath": os.environ.get("PYTHONPATH", ""),
        "fast_wisdom_holman_module_path": fast_module_path,
        "native_module_path": native_module_path,
        "native_module_loaded_from_current_project": native_module_path == str(NATIVE_PATH),
        "stale_editable_path_selected": False,
        "source_map_runtime_identity": source_map_identity,
        "source_map_runtime_identity_sha256": _sha256(_canonical_json(source_map_identity)),
        "source_builder_called": False,
        "source_map_invoked": False,
        "gpu_imported": False,
        "gpu_executed": False,
        "dynamics_api_invoked": False,
        "outcome_generated": False,
        "content_sha256": "",
    }
    if (
        value["production_pythonpath"]
        != os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
        or value["native_module_loaded_from_current_project"] is not True
        or fast_module_path != str(PROJECT_ROOT / "src" / "jxplanetx" / "fast_wisdom_holman.py")
    ):
        raise Full128SmokeError("CPU relocation probe did not use the production project root")
    value["content_sha256"] = _INNER._domain_hash(CPU_PROBE_DOMAIN, value)
    return value


def validate_cpu_relocation_probe(value: object) -> dict[str, Any]:
    keys = {
        "schema", "status", "project_root", "production_pythonpath",
        "fast_wisdom_holman_module_path", "native_module_path",
        "native_module_loaded_from_current_project", "stale_editable_path_selected",
        "source_map_runtime_identity", "source_map_runtime_identity_sha256",
        "source_builder_called", "source_map_invoked", "gpu_imported", "gpu_executed",
        "dynamics_api_invoked", "outcome_generated", "content_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise Full128SmokeError("CPU relocation probe shape differs")
    if (
        value.get("schema") != CPU_PROBE_SCHEMA
        or value.get("status") != "CPU_IMPORT_ONLY_NATIVE_RUNTIME_IDENTITY"
        or value.get("project_root") != str(PROJECT_ROOT)
        or value.get("production_pythonpath")
        != os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
        or value.get("fast_wisdom_holman_module_path")
        != str(PROJECT_ROOT / "src" / "jxplanetx" / "fast_wisdom_holman.py")
        or value.get("native_module_path") != str(NATIVE_PATH)
        or value.get("native_module_loaded_from_current_project") is not True
        or value.get("stale_editable_path_selected") is not False
        or any(value.get(key) is not False for key in (
            "source_builder_called", "source_map_invoked", "gpu_imported", "gpu_executed",
            "dynamics_api_invoked", "outcome_generated",
        ))
        or value.get("content_sha256") != _INNER._domain_hash(CPU_PROBE_DOMAIN, value)
    ):
        raise Full128SmokeError("CPU relocation probe semantics differ")
    source_map_identity = value.get("source_map_runtime_identity")
    if (
        type(source_map_identity) is not dict
        or _extension_receipt({"source_map_runtime_identity": source_map_identity})
        != _read_live_native_exact()
        or value.get("source_map_runtime_identity_sha256")
        != _sha256(_canonical_json(source_map_identity))
    ):
        raise Full128SmokeError("CPU import probe native identity differs")
    return dict(value)


# Explicit public aliases for audit/tests.
_publish = _INNER._publish
_strict_json = _INNER._strict_json
_domain_hash = _INNER._domain_hash
_delegates = _INNER._delegates

# The delegated CLI and execute parent resolve this global from the inner V1
# module, so direct offline validation also receives current-checkout import
# precedence rather than the copied virtualenv's historical editable .pth.
_INNER.validate_output = validate_output


def __getattr__(name: str) -> Any:
    return getattr(_INNER, name)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["cpu-relocation-probe"]:
        sys.stdout.buffer.write(_canonical_json(build_cpu_relocation_probe()))
        return 0
    return _INNER.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
