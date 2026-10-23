#!/usr/bin/env python3
"""Explicit supported-platform route to the V9 Wisdom--Holman kernel.

This module is additive.  It does not replace the V3 route exported by
``jxplanetx.fast_wisdom_holman``.  V9 is available only through the names in
this module, only when its narrow runtime policy is satisfied, and remains a
screening-only scientific implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import struct
import sys
import time
from typing import Any

import numpy as np

from . import _wisdom_holman_cpu_v9 as _native
from . import fast_wisdom_holman as _v3
from .engine.contracts import ForcePlan, StateSnapshot
from .engine.wisdom_holman_contracts import FixedStepWisdomHolmanSpec


SCHEMA = "jxplanetx.fast-wisdom-holman-v9.opt-in-screening-api.v1"
API_ID = "jx.fast-wisdom-holman-v9.opt-in-screening-api.v1"
RESULT_TYPE_ID = "jx.fast-wisdom-holman-v9.loop-result.v1"
KERNEL_ID = (
    "jx.wisdom-holman.native-complete-map."
    "bundled-series-bit-next.prototype.v9"
)
G_KERNEL_ID = (
    "jx.wisdom-holman.native-g-bundle."
    "bundled-series-bit-next.prototype.v9"
)
EXACT_FALLBACK_KERNEL_ID = (
    "jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
SUPPORT_STATUS = "SUPPORTED_OPT_IN_LINUX_X86_64_GCC"
PUBLIC_DEFAULT = False
EXECUTION_SCOPE = (
    "EXPLICIT_OPT_IN_COMPLETE_FIXED_STEP_NUMERICAL_MAP_AND_GUARDS_IN_"
    "NATIVE_V9_C_ON_FROZEN_LINUX_X86_64_GCC_BINARY64_RUNTIME"
)
REPLAY_SCOPE = "EXPLICIT_OPTIONAL_COMPLETE_NATIVE_V9_NUMERICAL_REPLAY"
CERTIFICATE_VERIFICATION_MODE = "V9_COMPLETE_MAP_POSTCONDITION_CERTIFICATE"
CERTIFICATE_WITH_EXACT_REPLAY_MODE = (
    "V9_COMPLETE_MAP_POSTCONDITION_CERTIFICATE_PLUS_EXACT_REPLAY"
)
CERTIFICATE_VERIFICATION_SCOPE = (
    "ALL_NATIVE_V9_STEPS_KEPLER_POSTCONDITIONS_TRANSLATION_NODE_PATH_"
    "GUARDS_EXACT_ACCOUNTING_SOURCE_CLOSURE_RUNTIME_FP_AND_SHA256_CUSTODY"
)
CERTIFICATE_WITH_EXACT_REPLAY_SCOPE = (
    CERTIFICATE_VERIFICATION_SCOPE
    + "_PLUS_SECOND_COMPLETE_BITWISE_NATIVE_V9_EXECUTION"
)
RESULT_CONTENT_CHECKSUM_DOMAIN = (
    "jx.fast-wisdom-holman-v9.result-content.v1"
)
CERTIFICATE_CONTENT_CHECKSUM_DOMAIN = (
    "jx.fast-wisdom-holman-v9.postcondition-certificate.v1"
)
RUNTIME_IDENTITY_CHECKSUM_DOMAIN = (
    "jx.fast-wisdom-holman-v9.runtime-identity.v1"
)
SOURCE_CLOSURE_CHECKSUM_DOMAIN = (
    "jx.fast-wisdom-holman-v9.compiled-source-closure.v1"
)
CHECKSUM_ALGORITHM = "SHA256_DOMAIN_SEPARATED_CANONICAL_BINARY_AND_JSON_V1"

_EXPECTED_COMPILER_FAMILY = "GCC"
_EXPECTED_COMPILER_FLAGS = (
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"
)
_NATIVE_WRAPPER_SOURCE_SHA256 = (
    "90468c9f527eefb6347b31e8e371310d1e054e2703bb6043baaae71d4078d654"
)
_SOURCE_CLOSURE_SHA256 = {
    "_wisdom_holman_cpu_v9_module.c": _NATIVE_WRAPPER_SOURCE_SHA256,
    "_wisdom_holman_cpu_v8_module.c": (
        "fcf77481d241902f2c05eabaaac9bfa89f208e645d9ae760baf67dcd03e9d288"
    ),
    "_wisdom_holman_loop_v9.c": (
        "680b4f887fa82561bb503a7982367f68c9c6648cd67ed8b926f1ae87e0d41bd2"
    ),
    "_wisdom_holman_loop_v8.c": (
        "6fe8b4fcda8947bc21aacce56d9e3383a1d5d579176c742a994628967927b004"
    ),
    "_wisdom_holman_loop_v7.c": (
        "337c92c1ea4aba54f64e8e336889a02dccb23d2bd6c275dedfac39e73fd9c0bb"
    ),
    "_wisdom_holman_loop_v6.c": (
        "c1da285ec602185f093c7f702097650f009827285332e2aeb1c5f06aa848590d"
    ),
    "_wisdom_holman_loop_v4.c": (
        "10981b416e4bb90a89babfc05cdd9c97bf1e5b49a0956490300bc5cbc591a16c"
    ),
    "_wisdom_holman_loop_v3.c": (
        "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629"
    ),
    "_wisdom_holman_loop_v2.c": (
        "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750"
    ),
}

METRIC_NAMES = _v3.METRIC_NAMES
COUNTER_NAMES = _v3.COUNTER_NAMES
STATUS_NAMES = dict(_v3.STATUS_NAMES)
CACHE_STAT_NAMES = ("entries", "hits", "misses")
FP_ENVIRONMENT_KEYS = (
    "compiler_family",
    "compiled_linux",
    "compiled_x86_64",
    "compiled_little_endian",
    "runtime_linux",
    "runtime_x86_64",
    "runtime_little_endian",
    "ieee_binary64_layout",
    "rounding_mode",
    "fe_tonearest",
    "round_to_nearest",
    "mxcsr_available",
    "mxcsr",
    "mxcsr_all_exceptions_masked",
    "mxcsr_rounding_control",
    "mxcsr_round_to_nearest",
    "mxcsr_ftz",
    "mxcsr_daz",
    "supported",
)
_FP_BOOL_KEYS = frozenset(
    {
        "compiled_linux",
        "compiled_x86_64",
        "compiled_little_endian",
        "runtime_linux",
        "runtime_x86_64",
        "runtime_little_endian",
        "ieee_binary64_layout",
        "round_to_nearest",
        "mxcsr_available",
        "mxcsr_all_exceptions_masked",
        "mxcsr_round_to_nearest",
        "mxcsr_ftz",
        "mxcsr_daz",
        "supported",
    }
)
_FP_INT_KEYS = frozenset(
    {"rounding_mode", "fe_tonearest", "mxcsr", "mxcsr_rounding_control"}
)


class FastWisdomHolmanV9Error(RuntimeError):
    """An explicit V9 request failed its narrow supported contract."""


FastWisdomHolmanV9Workspace = _v3.FastWisdomHolmanWorkspace


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _domain_hash(domain: str, payload: Any) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _canonical_json(payload))


def _file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise FastWisdomHolmanV9Error(
            f"V9 custody source is a symbolic link: {path.name}"
        )
    try:
        resolved = path.resolve(strict=True)
        raw = resolved.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise FastWisdomHolmanV9Error(
            f"V9 custody source is unavailable: {path.name}"
        ) from exc
    if not resolved.is_file() or not raw:
        raise FastWisdomHolmanV9Error(
            f"V9 custody source is not a nonempty regular file: {path.name}"
        )
    return {
        "path": resolved.as_posix(),
        "name": path.name,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _source_closure_identity() -> tuple[tuple[dict[str, Any], ...], str]:
    package_dir = Path(__file__).resolve(strict=True).parent
    records: list[dict[str, Any]] = []
    for name, expected_sha256 in _SOURCE_CLOSURE_SHA256.items():
        record = _file_identity(package_dir / name)
        if record["sha256"] != expected_sha256:
            raise FastWisdomHolmanV9Error(
                f"V9 compiled source custody changed: {name}"
            )
        records.append(record)
    canonical_records = tuple(
        {
            "name": record["name"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in records
    )
    closure_sha256 = _domain_hash(
        SOURCE_CLOSURE_CHECKSUM_DOMAIN, canonical_records
    )
    return tuple(records), closure_sha256


def _validate_python_runtime() -> None:
    float_info = sys.float_info
    little_one = struct.pack("=d", 1.0)
    finfo = np.finfo(np.float64)
    if (
        sys.platform != "linux"
        or platform.machine().lower() != "x86_64"
        or sys.byteorder != "little"
        or little_one != bytes.fromhex("000000000000f03f")
        or np.dtype(np.float64).itemsize != 8
        or float_info.radix != 2
        or float_info.mant_dig != 53
        or float_info.max_exp != 1024
        or float_info.min_exp != -1021
        or finfo.nmant != 52
        or finfo.iexp != 11
        or finfo.maxexp != 1024
        or finfo.minexp != -1022
        or float(finfo.eps) != math.ldexp(1.0, -52)
        or float(finfo.smallest_subnormal) != math.ldexp(1.0, -1074)
    ):
        raise FastWisdomHolmanV9Error(
            "V9 requires Linux x86_64 little-endian IEEE binary64"
        )


def _validate_native_identity() -> None:
    expected = {
        "MAP_KERNEL_ID": KERNEL_ID,
        "KERNEL_ID": G_KERNEL_ID,
        "EXACT_FALLBACK_KERNEL_ID": EXACT_FALLBACK_KERNEL_ID,
        "MAP_MAX_BODIES": 32,
        "MAP_METRIC_COUNT": len(METRIC_NAMES),
        "MAP_COUNTER_COUNT": len(COUNTER_NAMES),
        "SCIENTIFIC_CLAIM_STATE": SCIENTIFIC_CLAIM_STATE,
        "COMPILER_FLAGS": _EXPECTED_COMPILER_FLAGS,
    }
    if any(getattr(_native, name, None) != value for name, value in expected.items()):
        raise FastWisdomHolmanV9Error("native V9 identity changed")
    if (
        type(getattr(_native, "COMPILER_VERSION", None)) is not str
        or not _native.COMPILER_VERSION
        or not callable(getattr(_native, "_runtime_fp_environment", None))
        or not callable(getattr(_native, "integrate_map", None))
        or not callable(getattr(_native, "last_fallback_count", None))
        or not callable(getattr(_native, "last_cache_stats", None))
    ):
        raise FastWisdomHolmanV9Error("native V9 build identity is incomplete")


def fast_wisdom_holman_v9_runtime_identity() -> dict[str, Any]:
    """Return and verify the opt-in V9 binary and compiled source identity."""

    _validate_native_identity()
    runtime_fp_environment = _checked_fp_environment()
    sources, source_closure_sha256 = _source_closure_identity()
    extension = _file_identity(Path(_native.__file__))
    canonical = {
        "schema": SCHEMA,
        "api_id": API_ID,
        "result_type_id": RESULT_TYPE_ID,
        "kernel_id": _native.MAP_KERNEL_ID,
        "g_kernel_id": _native.KERNEL_ID,
        "exact_fallback_kernel_id": _native.EXACT_FALLBACK_KERNEL_ID,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        "support_status": SUPPORT_STATUS,
        "public_default": PUBLIC_DEFAULT,
        "registry_authorized": False,
        "qualification_authorized": False,
        "production_authorized": False,
        "superiority_claimed": False,
        "maximum_bodies": _native.MAP_MAX_BODIES,
        "metric_count": _native.MAP_METRIC_COUNT,
        "counter_count": _native.MAP_COUNTER_COUNT,
        "compiler_family": _EXPECTED_COMPILER_FAMILY,
        "compiler_version": _native.COMPILER_VERSION,
        "compiler_flags": _native.COMPILER_FLAGS,
        "runtime_fp_environment": runtime_fp_environment,
        "platform_policy": {
            "system": "Linux",
            "machine": "x86_64",
            "byteorder": "little",
            "floating_point": "IEEE_754_BINARY64",
            "rounding": "FE_TONEAREST",
            "mxcsr_all_exceptions_masked": True,
            "mxcsr_rounding_control": 0,
            "mxcsr_ftz": False,
            "mxcsr_daz": False,
        },
        "source_closure_sha256": source_closure_sha256,
        "compiled_sources": tuple(
            {
                "name": record["name"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in sources
        ),
        "extension_binary": {
            "name": extension["name"],
            "size_bytes": extension["size_bytes"],
            "sha256": extension["sha256"],
        },
    }
    content_sha256 = _domain_hash(RUNTIME_IDENTITY_CHECKSUM_DOMAIN, canonical)
    return canonical | {"content_sha256": content_sha256}


def _checked_native_fp_environment() -> tuple[
    tuple[str, str | bool | int], ...
]:
    """Gate raw floating-point controls before any Python numerical work."""

    try:
        environment = _native._runtime_fp_environment()
    except Exception as exc:
        raise FastWisdomHolmanV9Error(
            "native V9 floating-point environment probe failed"
        ) from exc
    if type(environment) is not dict or set(environment) != set(FP_ENVIRONMENT_KEYS):
        raise FastWisdomHolmanV9Error(
            "native V9 floating-point environment schema changed"
        )
    if (
        type(environment["compiler_family"]) is not str
        or environment["compiler_family"] != _EXPECTED_COMPILER_FAMILY
        or any(type(environment[name]) is not bool for name in _FP_BOOL_KEYS)
        or any(type(environment[name]) is not int for name in _FP_INT_KEYS)
    ):
        raise FastWisdomHolmanV9Error(
            "native V9 floating-point environment types changed"
        )
    required_true = (
        "compiled_linux",
        "compiled_x86_64",
        "compiled_little_endian",
        "runtime_linux",
        "runtime_x86_64",
        "runtime_little_endian",
        "ieee_binary64_layout",
        "round_to_nearest",
        "mxcsr_available",
        "mxcsr_all_exceptions_masked",
        "mxcsr_round_to_nearest",
        "supported",
    )
    expected_supported = (
        environment["compiler_family"] == _EXPECTED_COMPILER_FAMILY
        and environment["compiled_linux"]
        and environment["compiled_x86_64"]
        and environment["compiled_little_endian"]
        and environment["runtime_linux"]
        and environment["runtime_x86_64"]
        and environment["runtime_little_endian"]
        and environment["ieee_binary64_layout"]
        and environment["round_to_nearest"]
        and environment["mxcsr_available"]
        and environment["mxcsr_all_exceptions_masked"]
        and environment["mxcsr_round_to_nearest"]
        and not environment["mxcsr_ftz"]
        and not environment["mxcsr_daz"]
    )
    if (
        any(environment[name] is not True for name in required_true)
        or not 0 <= environment["mxcsr"] <= 0xFFFFFFFF
        or environment["fe_tonearest"] != 0
        or environment["rounding_mode"] != 0
        or environment["rounding_mode"] != environment["fe_tonearest"]
        or environment["round_to_nearest"]
        is not (environment["rounding_mode"] == environment["fe_tonearest"])
        or environment["mxcsr_rounding_control"] != 0
        or environment["mxcsr_rounding_control"]
        != ((environment["mxcsr"] >> 13) & 0x3)
        or environment["mxcsr_round_to_nearest"]
        is not (environment["mxcsr_rounding_control"] == 0)
        or environment["mxcsr_all_exceptions_masked"]
        is not ((environment["mxcsr"] & 0x1F80) == 0x1F80)
        or environment["mxcsr_ftz"]
        is not bool(environment["mxcsr"] & (1 << 15))
        or environment["mxcsr_daz"]
        is not bool(environment["mxcsr"] & (1 << 6))
        or environment["supported"] is not expected_supported
        or environment["mxcsr_ftz"] is not False
        or environment["mxcsr_daz"] is not False
    ):
        raise FastWisdomHolmanV9Error(
            "native V9 floating-point environment is unsupported"
        )
    # MXCSR bits 0--5 are ambient sticky exception-status flags, not control
    # state.  Keep every policy/control bit while normalizing only those six
    # status bits so custody hashes do not depend on unrelated earlier math.
    environment["mxcsr"] &= ~0x3F
    return tuple((name, environment[name]) for name in FP_ENVIRONMENT_KEYS)


def _checked_fp_environment() -> tuple[tuple[str, str | bool | int], ...]:
    environment = _checked_native_fp_environment()
    _validate_python_runtime()
    return environment


def _fp_control_fingerprint(
    environment: tuple[tuple[str, str | bool | int], ...]
) -> tuple[tuple[str, str | bool | int], ...]:
    """Exclude only MXCSR's six sticky exception-status bits."""

    return environment


def _owned_readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    if not array.flags.owndata or not array.flags.aligned:
        raise FastWisdomHolmanV9Error("failed to allocate aligned owned storage")
    array.setflags(write=False)
    return array


def _validate_owned_array(
    value: object,
    dtype: np.dtype[Any],
    label: str,
    *,
    writeable: bool,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or not value.flags.owndata
        or not value.flags.aligned
        or not value.flags.c_contiguous
        or value.flags.writeable is not writeable
    ):
        raise FastWisdomHolmanV9Error(
            f"{label} must be an aligned owned C-contiguous {dtype.name} array"
        )
    return value


def _validate_no_sharing(arrays: tuple[tuple[str, np.ndarray], ...]) -> None:
    for index, (left_name, left) in enumerate(arrays):
        for right_name, right in arrays[index + 1 :]:
            if np.shares_memory(left, right):
                raise FastWisdomHolmanV9Error(
                    f"{left_name} and {right_name} must not share storage"
                )


def _validate_workspace_custody(workspace: FastWisdomHolmanV9Workspace) -> None:
    arrays = (
        (
            "checkpoint_steps",
            _validate_owned_array(
                workspace.checkpoint_steps,
                np.dtype(np.int64),
                "checkpoint_steps",
                writeable=False,
            ),
        ),
        (
            "checkpoint_epochs",
            _validate_owned_array(
                workspace.checkpoint_epochs,
                np.dtype(np.float64),
                "checkpoint_epochs",
                writeable=False,
            ),
        ),
        (
            "positions",
            _validate_owned_array(
                workspace.positions,
                np.dtype(np.float64),
                "positions",
                writeable=False,
            ),
        ),
        (
            "velocities",
            _validate_owned_array(
                workspace.velocities,
                np.dtype(np.float64),
                "velocities",
                writeable=False,
            ),
        ),
        (
            "gravitational_parameters",
            _validate_owned_array(
                workspace.gravitational_parameters,
                np.dtype(np.float64),
                "gravitational_parameters",
                writeable=False,
            ),
        ),
        (
            "radii",
            _validate_owned_array(
                workspace.radii,
                np.dtype(np.float64),
                "radii",
                writeable=False,
            ),
        ),
    )
    _validate_no_sharing(arrays)
    try:
        _v3._validate_workspace_custody(workspace)
    except Exception as exc:
        raise FastWisdomHolmanV9Error("V9 workspace custody failed") from exc


def prepare_fast_wisdom_holman_v9_workspace(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepWisdomHolmanSpec,
) -> FastWisdomHolmanV9Workspace:
    """Validate public inputs and retain aligned, owned V9 input custody."""

    fast_wisdom_holman_v9_runtime_identity()
    try:
        workspace = _v3.prepare_fast_wisdom_holman_workspace(
            snapshot, plan, spec
        )
    except Exception as exc:
        raise FastWisdomHolmanV9Error("V9 public input validation failed") from exc
    _validate_workspace_custody(workspace)
    return workspace


def _update_array_hash(hasher: Any, label: str, array: np.ndarray) -> None:
    descriptor = _canonical_json(
        {"label": label, "dtype": array.dtype.str, "shape": list(array.shape)}
    )
    hasher.update(len(descriptor).to_bytes(8, "big"))
    hasher.update(descriptor)
    raw = array.tobytes(order="C")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _result_hash(
    *,
    input_content_sha256: str,
    runtime_identity_sha256: str,
    source_closure_sha256: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    checkpoint_epochs: np.ndarray,
    metrics: tuple[tuple[str, float], ...],
    counters: tuple[tuple[str, int], ...],
    fallback_count: int,
    cache_stats: tuple[tuple[str, int], ...],
    fp_environment_before: tuple[tuple[str, str | bool | int], ...],
    fp_environment_after: tuple[tuple[str, str | bool | int], ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(RESULT_CONTENT_CHECKSUM_DOMAIN.encode("ascii") + b"\0")
    scalar = {
        "schema": SCHEMA,
        "api_id": API_ID,
        "result_type_id": RESULT_TYPE_ID,
        "kernel_id": KERNEL_ID,
        "g_kernel_id": G_KERNEL_ID,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        "support_status": SUPPORT_STATUS,
        "public_default": PUBLIC_DEFAULT,
        "input_content_sha256": input_content_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "source_closure_sha256": source_closure_sha256,
        "metrics_hex": tuple((name, value.hex()) for name, value in metrics),
        "counters": counters,
        "fallback_count": fallback_count,
        "cache_stats": cache_stats,
        "fp_environment_before": fp_environment_before,
        "fp_environment_after": fp_environment_after,
    }
    encoded = _canonical_json(scalar)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    for label, array in (
        ("positions", positions),
        ("velocities", velocities),
        ("checkpoint_epochs", checkpoint_epochs),
    ):
        _update_array_hash(hasher, label, array)
    return hasher.hexdigest()


def _certificate_hash(
    *,
    input_content_sha256: str,
    result_content_sha256: str,
    runtime_identity_sha256: str,
    source_closure_sha256: str,
    verification_mode: str,
    certified_steps: int,
    certified_kepler_solves: int,
    primary_counters: tuple[tuple[str, int], ...],
    replay_counters: tuple[tuple[str, int], ...],
    primary_fallback_count: int,
    replay_fallback_count: int,
    primary_cache_stats: tuple[tuple[str, int], ...],
    replay_cache_stats: tuple[tuple[str, int], ...],
    primary_fp_environment_before: tuple[
        tuple[str, str | bool | int], ...
    ],
    primary_fp_environment_after: tuple[
        tuple[str, str | bool | int], ...
    ],
    replay_fp_environment_before: tuple[
        tuple[str, str | bool | int], ...
    ]
    | None,
    replay_fp_environment_after: tuple[
        tuple[str, str | bool | int], ...
    ]
    | None,
) -> str:
    payload = {
        "schema": SCHEMA,
        "api_id": API_ID,
        "result_type_id": RESULT_TYPE_ID,
        "kernel_id": KERNEL_ID,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        "support_status": SUPPORT_STATUS,
        "public_default": PUBLIC_DEFAULT,
        "registry_authorized": False,
        "qualification_authorized": False,
        "production_authorized": False,
        "superiority_claimed": False,
        "input_content_sha256": input_content_sha256,
        "result_content_sha256": result_content_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "source_closure_sha256": source_closure_sha256,
        "verification_mode": verification_mode,
        "certified_steps": certified_steps,
        "certified_kepler_solves": certified_kepler_solves,
        "full_step_postcondition_coverage": True,
        "independent_numerical_method_verified": False,
        "deterministic_implementation_defect_excluded": False,
        "primary_counters": primary_counters,
        "replay_counters": replay_counters,
        "primary_fallback_count": primary_fallback_count,
        "replay_fallback_count": replay_fallback_count,
        "primary_cache_stats": primary_cache_stats,
        "replay_cache_stats": replay_cache_stats,
        "primary_fp_environment_before": primary_fp_environment_before,
        "primary_fp_environment_after": primary_fp_environment_after,
        "replay_fp_environment_before": replay_fp_environment_before,
        "replay_fp_environment_after": replay_fp_environment_after,
    }
    return _domain_hash(CERTIFICATE_CONTENT_CHECKSUM_DOMAIN, payload)


@dataclass(frozen=True, eq=False)
class _RawV9Run:
    positions: np.ndarray
    velocities: np.ndarray
    metrics: np.ndarray
    counters: np.ndarray
    fallback_count: int
    cache_stats: tuple[tuple[str, int], ...]
    fp_environment_before: tuple[tuple[str, str | bool | int], ...]
    fp_environment_after: tuple[tuple[str, str | bool | int], ...]
    seconds: float
    runtime_identity_sha256: str
    source_closure_sha256: str
    content_sha256: str


def _read_native_diagnostics() -> tuple[int, tuple[tuple[str, int], ...]]:
    try:
        fallback_count = _native.last_fallback_count()
        raw_cache = _native.last_cache_stats()
    except Exception as exc:
        raise FastWisdomHolmanV9Error(
            "native V9 per-call diagnostics are unavailable"
        ) from exc
    if (
        type(fallback_count) is not int
        or fallback_count < 0
        or type(raw_cache) is not tuple
        or len(raw_cache) != len(CACHE_STAT_NAMES)
        or any(type(value) is not int or value < 0 for value in raw_cache)
    ):
        raise FastWisdomHolmanV9Error(
            "native V9 per-call diagnostics are invalid"
        )
    return fallback_count, tuple(
        (name, value)
        for name, value in zip(CACHE_STAT_NAMES, raw_cache, strict=True)
    )


def _execute_once(workspace: FastWisdomHolmanV9Workspace) -> _RawV9Run:
    # This raw native control gate must precede workspace/NumPy validation:
    # unmasked MXCSR exceptions can otherwise terminate the process while
    # merely inspecting an already-prepared workspace.
    _checked_native_fp_environment()
    _validate_workspace_custody(workspace)
    identity = fast_wisdom_holman_v9_runtime_identity()
    runtime_identity_sha256 = identity["content_sha256"]
    source_closure_sha256 = identity["source_closure_sha256"]
    shape = (workspace.checkpoint_count, workspace.body_count, 3)
    positions = np.empty(shape, dtype=np.float64, order="C")
    velocities = np.empty(shape, dtype=np.float64, order="C")
    metrics = np.empty(len(METRIC_NAMES), dtype=np.float64, order="C")
    counters = np.empty(len(COUNTER_NAMES), dtype=np.int64, order="C")
    for label, array in (
        ("output positions", positions),
        ("output velocities", velocities),
        ("output metrics", metrics),
        ("output counters", counters),
    ):
        _validate_owned_array(array, array.dtype, label, writeable=True)
    _validate_no_sharing(
        (
            ("output positions", positions),
            ("output velocities", velocities),
            ("output metrics", metrics),
            ("output counters", counters),
        )
    )

    started = time.perf_counter()
    before = _checked_fp_environment()
    call_error: BaseException | None = None
    status: object = None
    diagnostics: tuple[int, tuple[tuple[str, int], ...]] | None = None
    diagnostic_error: BaseException | None = None
    after_error: BaseException | None = None
    after: tuple[tuple[str, str | bool | int], ...] | None = None
    seconds: float | None = None
    try:
        status = _native.integrate_map(
            workspace.body_count,
            workspace.checkpoint_count,
            workspace.positions,
            workspace.velocities,
            workspace.gravitational_parameters,
            workspace.radii,
            workspace.checkpoint_steps,
            workspace.fixed_step,
            workspace.minimum_encounter_pair_separation,
            workspace.minimum_jacobi_periapse,
            workspace.maximum_initial_barycenter_position_norm,
            workspace.maximum_initial_barycenter_velocity_norm,
            positions,
            velocities,
            metrics,
            counters,
        )
    except BaseException as exc:
        call_error = exc
    finally:
        # Native execution may return with hostile FP controls.  Probe those
        # raw controls before diagnostics, Python numerical validation, or
        # timing arithmetic; any of those can otherwise trap the process.
        try:
            after = _checked_native_fp_environment()
        except BaseException as exc:
            after_error = exc
        if after_error is None:
            try:
                _validate_python_runtime()
            except BaseException as exc:
                after_error = exc
        if after_error is None:
            try:
                diagnostics = _read_native_diagnostics()
            except BaseException as exc:
                diagnostic_error = exc
            seconds = time.perf_counter() - started

    if after_error is not None:
        raise FastWisdomHolmanV9Error(
            "native V9 post-run floating-point environment failed"
        ) from after_error
    if after is None or _fp_control_fingerprint(before) != _fp_control_fingerprint(after):
        raise FastWisdomHolmanV9Error(
            "native V9 floating-point environment changed during execution"
        )
    if diagnostic_error is not None:
        raise FastWisdomHolmanV9Error(
            "native V9 immediate diagnostics failed"
        ) from diagnostic_error
    if call_error is not None:
        raise FastWisdomHolmanV9Error("native V9 call failed") from call_error
    if type(status) is not int:
        raise FastWisdomHolmanV9Error("native V9 status type changed")
    if status != 0:
        status_name = STATUS_NAMES.get(status, "UNKNOWN")
        raise FastWisdomHolmanV9Error(
            f"native V9 loop failed closed with status {status}: {status_name}"
        )
    if diagnostics is None:
        raise FastWisdomHolmanV9Error("native V9 diagnostics were not captured")
    if seconds is None:
        raise FastWisdomHolmanV9Error("native V9 elapsed time was not captured")
    fallback_count, cache_stats = diagnostics
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(metrics))
    ):
        raise FastWisdomHolmanV9Error(
            "native V9 loop returned nonfinite retained output"
        )

    expected_steps = int(workspace.checkpoint_steps[-1])
    expected = {
        "completed_steps": expected_steps,
        "force_evaluations": expected_steps + 1,
        "interaction_assemblies": expected_steps + 1,
        "kepler_solves": expected_steps * (workspace.body_count - 1),
        "node_guards": 2 * expected_steps + 1,
        "path_guards": expected_steps,
        "forward_transforms": 1,
        "inverse_transforms": expected_steps + workspace.checkpoint_count - 1,
    }
    observed = dict(zip(COUNTER_NAMES, map(int, counters), strict=True))
    for name, expected_value in expected.items():
        if observed[name] != expected_value:
            raise FastWisdomHolmanV9Error(
                f"native V9 accounting mismatch for {name}: expected "
                f"{expected_value}, observed {observed[name]}"
            )
    if fallback_count > observed["kepler_solves"]:
        raise FastWisdomHolmanV9Error("native V9 fallback accounting is invalid")

    metric_tuple = tuple(
        (name, float(value))
        for name, value in zip(METRIC_NAMES, metrics, strict=True)
    )
    counter_tuple = tuple(
        (name, int(value))
        for name, value in zip(COUNTER_NAMES, counters, strict=True)
    )
    content_sha256 = _result_hash(
        input_content_sha256=workspace.input_content_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_closure_sha256=source_closure_sha256,
        positions=positions,
        velocities=velocities,
        checkpoint_epochs=workspace.checkpoint_epochs,
        metrics=metric_tuple,
        counters=counter_tuple,
        fallback_count=fallback_count,
        cache_stats=cache_stats,
        fp_environment_before=before,
        fp_environment_after=after,
    )
    return _RawV9Run(
        positions=positions,
        velocities=velocities,
        metrics=metrics,
        counters=counters,
        fallback_count=fallback_count,
        cache_stats=cache_stats,
        fp_environment_before=before,
        fp_environment_after=after,
        seconds=float(seconds),
        runtime_identity_sha256=runtime_identity_sha256,
        source_closure_sha256=source_closure_sha256,
        content_sha256=content_sha256,
    )


def _validate_named_float_tuple(
    value: object, names: tuple[str, ...], label: str
) -> tuple[tuple[str, float], ...]:
    if type(value) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not float
            or not math.isfinite(item[1])
            for item in value
        ) or tuple(item[0] for item in value) != names:
        raise FastWisdomHolmanV9Error(f"{label} are invalid")
    return value


def _validate_named_int_tuple(
    value: object, names: tuple[str, ...], label: str
) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
            for item in value
        ) or tuple(item[0] for item in value) != names:
        raise FastWisdomHolmanV9Error(f"{label} are invalid")
    return value


def _validate_stored_fp_environment(
    value: object, label: str
) -> tuple[tuple[str, str | bool | int], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple or len(item) != 2 for item in value
    ) or tuple(item[0] for item in value) != FP_ENVIRONMENT_KEYS:
        raise FastWisdomHolmanV9Error(f"{label} schema is invalid")
    environment = dict(value)
    if (
        type(environment["compiler_family"]) is not str
        or any(type(environment[name]) is not bool for name in _FP_BOOL_KEYS)
        or any(type(environment[name]) is not int for name in _FP_INT_KEYS)
    ):
        raise FastWisdomHolmanV9Error(f"{label} types are invalid")
    expected_supported = (
        environment["compiler_family"] == _EXPECTED_COMPILER_FAMILY
        and environment["compiled_linux"]
        and environment["compiled_x86_64"]
        and environment["compiled_little_endian"]
        and environment["runtime_linux"]
        and environment["runtime_x86_64"]
        and environment["runtime_little_endian"]
        and environment["ieee_binary64_layout"]
        and environment["round_to_nearest"]
        and environment["mxcsr_available"]
        and environment["mxcsr_all_exceptions_masked"]
        and environment["mxcsr_round_to_nearest"]
        and not environment["mxcsr_ftz"]
        and not environment["mxcsr_daz"]
    )
    if (
        environment["compiler_family"] != _EXPECTED_COMPILER_FAMILY
        or any(
            environment[name] is not True
            for name in (
                "compiled_linux",
                "compiled_x86_64",
                "compiled_little_endian",
                "runtime_linux",
                "runtime_x86_64",
                "runtime_little_endian",
                "ieee_binary64_layout",
                "round_to_nearest",
                "mxcsr_available",
                "mxcsr_all_exceptions_masked",
                "mxcsr_round_to_nearest",
                "supported",
            )
        )
        or not 0 <= environment["mxcsr"] <= 0xFFFFFFFF
        or environment["mxcsr"] & 0x3F != 0
        or environment["fe_tonearest"] != 0
        or environment["rounding_mode"] != 0
        or environment["rounding_mode"] != environment["fe_tonearest"]
        or environment["round_to_nearest"]
        is not (environment["rounding_mode"] == environment["fe_tonearest"])
        or environment["mxcsr_rounding_control"] != 0
        or environment["mxcsr_rounding_control"]
        != ((environment["mxcsr"] >> 13) & 0x3)
        or environment["mxcsr_round_to_nearest"]
        is not (environment["mxcsr_rounding_control"] == 0)
        or environment["mxcsr_all_exceptions_masked"]
        is not ((environment["mxcsr"] & 0x1F80) == 0x1F80)
        or environment["mxcsr_ftz"]
        is not bool(environment["mxcsr"] & (1 << 15))
        or environment["mxcsr_daz"]
        is not bool(environment["mxcsr"] & (1 << 6))
        or environment["supported"] is not expected_supported
        or environment["mxcsr_ftz"] is not False
        or environment["mxcsr_daz"] is not False
    ):
        raise FastWisdomHolmanV9Error(f"{label} is unsupported")
    return value


@dataclass(frozen=True, eq=False)
class FastWisdomHolmanV9LoopResult:
    """Owned V9 output plus primary/replay custody and diagnostics."""

    positions: np.ndarray
    velocities: np.ndarray
    checkpoint_epochs: np.ndarray
    metrics: tuple[tuple[str, float], ...]
    primary_counters: tuple[tuple[str, int], ...]
    replay_counters: tuple[tuple[str, int], ...]
    total_counters: tuple[tuple[str, int], ...]
    primary_fallback_count: int
    replay_fallback_count: int
    total_fallback_count: int
    primary_cache_stats: tuple[tuple[str, int], ...]
    replay_cache_stats: tuple[tuple[str, int], ...]
    primary_fp_environment_before: tuple[tuple[str, str | bool | int], ...]
    primary_fp_environment_after: tuple[tuple[str, str | bool | int], ...]
    replay_fp_environment_before: tuple[tuple[str, str | bool | int], ...] | None
    replay_fp_environment_after: tuple[tuple[str, str | bool | int], ...] | None
    primary_seconds: float
    replay_seconds: float
    replay_count: int
    verification_mode: str
    verification_scope: str
    certified_steps: int
    certified_kepler_solves: int
    full_step_postcondition_coverage: bool
    exact_replay_verified: bool
    independent_numerical_method_verified: bool
    deterministic_implementation_defect_excluded: bool
    verification_content_sha256: str
    result_content_sha256: str
    input_content_sha256: str
    runtime_identity_sha256: str
    source_closure_sha256: str
    schema: str = SCHEMA
    api_id: str = API_ID
    result_type_id: str = RESULT_TYPE_ID
    kernel_id: str = KERNEL_ID
    g_kernel_id: str = G_KERNEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    support_status: str = SUPPORT_STATUS
    public_default: bool = PUBLIC_DEFAULT
    execution_scope: str = EXECUTION_SCOPE
    replay_scope: str = REPLAY_SCOPE
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_authorized: bool = False
    superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or self.api_id != API_ID
            or self.result_type_id != RESULT_TYPE_ID
            or self.kernel_id != KERNEL_ID
            or self.g_kernel_id != G_KERNEL_ID
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or self.support_status != SUPPORT_STATUS
            or self.public_default is not False
            or self.execution_scope != EXECUTION_SCOPE
            or self.replay_scope != REPLAY_SCOPE
            or type(self.registry_authorized) is not bool
            or self.registry_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.superiority_claimed) is not bool
            or self.superiority_claimed
        ):
            raise FastWisdomHolmanV9Error(
                "V9 result identity, support state, or claim ceiling changed"
            )
        arrays = (
            (
                "positions",
                _validate_owned_array(
                    self.positions,
                    np.dtype(np.float64),
                    "positions",
                    writeable=False,
                ),
            ),
            (
                "velocities",
                _validate_owned_array(
                    self.velocities,
                    np.dtype(np.float64),
                    "velocities",
                    writeable=False,
                ),
            ),
            (
                "checkpoint_epochs",
                _validate_owned_array(
                    self.checkpoint_epochs,
                    np.dtype(np.float64),
                    "checkpoint_epochs",
                    writeable=False,
                ),
            ),
        )
        _validate_no_sharing(arrays)
        if (
            self.positions.ndim != 3
            or self.positions.shape != self.velocities.shape
            or self.positions.shape[0] != self.checkpoint_epochs.size
            or self.positions.shape[2] != 3
            or self.checkpoint_epochs.ndim != 1
            or not np.all(np.isfinite(self.positions))
            or not np.all(np.isfinite(self.velocities))
            or not np.all(np.isfinite(self.checkpoint_epochs))
        ):
            raise FastWisdomHolmanV9Error("V9 result arrays are invalid")

        metrics = _validate_named_float_tuple(
            self.metrics, METRIC_NAMES, "V9 metrics"
        )
        metric_values = dict(metrics)
        if any(value < 0.0 for value in metric_values.values()):
            raise FastWisdomHolmanV9Error("V9 metric is negative")
        for name in (
            "minimum_jacobi_periapse",
            "minimum_pair_endpoint",
            "minimum_pair_path_lower_bound",
            "minimum_pair_clearance_after_margin",
        ):
            if metric_values[name] <= 0.0:
                raise FastWisdomHolmanV9Error("V9 minimum guard is invalid")
        if (
            metric_values["maximum_kepler_time_residual"]
            > metric_values["maximum_kepler_residual_tolerance"]
        ):
            raise FastWisdomHolmanV9Error("V9 Kepler residual certificate failed")

        primary = dict(
            _validate_named_int_tuple(
                self.primary_counters, COUNTER_NAMES, "V9 primary counters"
            )
        )
        replay = dict(
            _validate_named_int_tuple(
                self.replay_counters, COUNTER_NAMES, "V9 replay counters"
            )
        )
        total = dict(
            _validate_named_int_tuple(
                self.total_counters, COUNTER_NAMES, "V9 total counters"
            )
        )
        _validate_named_int_tuple(
            self.primary_cache_stats, CACHE_STAT_NAMES, "V9 primary cache stats"
        )
        _validate_named_int_tuple(
            self.replay_cache_stats, CACHE_STAT_NAMES, "V9 replay cache stats"
        )
        primary_before = _validate_stored_fp_environment(
            self.primary_fp_environment_before, "V9 primary pre-run environment"
        )
        primary_after = _validate_stored_fp_environment(
            self.primary_fp_environment_after, "V9 primary post-run environment"
        )
        if _fp_control_fingerprint(primary_before) != _fp_control_fingerprint(
            primary_after
        ):
            raise FastWisdomHolmanV9Error(
                "V9 primary floating-point environment changed"
            )

        if type(self.replay_count) is not int or self.replay_count not in (0, 1):
            raise FastWisdomHolmanV9Error("V9 replay count must be zero or one")
        expected_replay = self.replay_count == 1
        expected_mode = (
            CERTIFICATE_WITH_EXACT_REPLAY_MODE
            if expected_replay
            else CERTIFICATE_VERIFICATION_MODE
        )
        expected_scope = (
            CERTIFICATE_WITH_EXACT_REPLAY_SCOPE
            if expected_replay
            else CERTIFICATE_VERIFICATION_SCOPE
        )
        if (
            self.verification_mode != expected_mode
            or self.verification_scope != expected_scope
            or type(self.exact_replay_verified) is not bool
            or self.exact_replay_verified is not expected_replay
        ):
            raise FastWisdomHolmanV9Error("V9 replay certificate is inconsistent")
        if expected_replay:
            replay_before = _validate_stored_fp_environment(
                self.replay_fp_environment_before,
                "V9 replay pre-run environment",
            )
            replay_after = _validate_stored_fp_environment(
                self.replay_fp_environment_after,
                "V9 replay post-run environment",
            )
            if (
                _fp_control_fingerprint(replay_before)
                != _fp_control_fingerprint(replay_after)
                or _fp_control_fingerprint(replay_before)
                != _fp_control_fingerprint(primary_before)
                or replay != primary
                or self.replay_cache_stats != self.primary_cache_stats
                or self.replay_fallback_count != self.primary_fallback_count
            ):
                raise FastWisdomHolmanV9Error(
                    "V9 exact replay diagnostics differ"
                )
        elif (
            self.replay_fp_environment_before is not None
            or self.replay_fp_environment_after is not None
            or any(replay.values())
            or any(value for _, value in self.replay_cache_stats)
            or self.replay_fallback_count != 0
        ):
            raise FastWisdomHolmanV9Error(
                "V9 no-replay diagnostics must be empty"
            )

        if (
            type(self.primary_fallback_count) is not int
            or self.primary_fallback_count < 0
            or type(self.replay_fallback_count) is not int
            or self.replay_fallback_count < 0
            or type(self.total_fallback_count) is not int
            or self.total_fallback_count
            != self.primary_fallback_count + self.replay_fallback_count
            or any(
                total[name] != primary[name] + replay[name]
                for name in COUNTER_NAMES
            )
            or type(self.certified_steps) is not int
            or self.certified_steps <= 0
            or self.certified_steps != primary["completed_steps"]
            or type(self.certified_kepler_solves) is not int
            or self.certified_kepler_solves <= 0
            or self.certified_kepler_solves != primary["kepler_solves"]
            or self.primary_fallback_count > self.certified_kepler_solves
        ):
            raise FastWisdomHolmanV9Error("V9 accounting binding failed")
        if (
            type(self.primary_seconds) is not float
            or not math.isfinite(self.primary_seconds)
            or self.primary_seconds < 0.0
            or type(self.replay_seconds) is not float
            or not math.isfinite(self.replay_seconds)
            or self.replay_seconds < 0.0
            or (not expected_replay and self.replay_seconds != 0.0)
            or type(self.full_step_postcondition_coverage) is not bool
            or not self.full_step_postcondition_coverage
            or type(self.independent_numerical_method_verified) is not bool
            or self.independent_numerical_method_verified
            or type(self.deterministic_implementation_defect_excluded) is not bool
            or self.deterministic_implementation_defect_excluded
        ):
            raise FastWisdomHolmanV9Error("V9 certificate claim boundary changed")

        for digest in (
            self.verification_content_sha256,
            self.result_content_sha256,
            self.input_content_sha256,
            self.runtime_identity_sha256,
            self.source_closure_sha256,
        ):
            if not _valid_digest(digest):
                raise FastWisdomHolmanV9Error("V9 custody digest is invalid")
        expected_result = _result_hash(
            input_content_sha256=self.input_content_sha256,
            runtime_identity_sha256=self.runtime_identity_sha256,
            source_closure_sha256=self.source_closure_sha256,
            positions=self.positions,
            velocities=self.velocities,
            checkpoint_epochs=self.checkpoint_epochs,
            metrics=self.metrics,
            counters=self.primary_counters,
            fallback_count=self.primary_fallback_count,
            cache_stats=self.primary_cache_stats,
            fp_environment_before=self.primary_fp_environment_before,
            fp_environment_after=self.primary_fp_environment_after,
        )
        if self.result_content_sha256 != expected_result:
            raise FastWisdomHolmanV9Error("V9 result content digest changed")
        expected_certificate = _certificate_hash(
            input_content_sha256=self.input_content_sha256,
            result_content_sha256=self.result_content_sha256,
            runtime_identity_sha256=self.runtime_identity_sha256,
            source_closure_sha256=self.source_closure_sha256,
            verification_mode=self.verification_mode,
            certified_steps=self.certified_steps,
            certified_kepler_solves=self.certified_kepler_solves,
            primary_counters=self.primary_counters,
            replay_counters=self.replay_counters,
            primary_fallback_count=self.primary_fallback_count,
            replay_fallback_count=self.replay_fallback_count,
            primary_cache_stats=self.primary_cache_stats,
            replay_cache_stats=self.replay_cache_stats,
            primary_fp_environment_before=self.primary_fp_environment_before,
            primary_fp_environment_after=self.primary_fp_environment_after,
            replay_fp_environment_before=self.replay_fp_environment_before,
            replay_fp_environment_after=self.replay_fp_environment_after,
        )
        if self.verification_content_sha256 != expected_certificate:
            raise FastWisdomHolmanV9Error("V9 certificate digest changed")


def integrate_fast_wisdom_holman_v9_workspace(
    workspace: FastWisdomHolmanV9Workspace,
    *,
    exact_replay: bool = False,
) -> FastWisdomHolmanV9LoopResult:
    """Run V9 explicitly and return its screening-only certificate."""

    if type(workspace) is not FastWisdomHolmanV9Workspace:
        raise FastWisdomHolmanV9Error(
            "workspace must be an exact FastWisdomHolmanWorkspace"
        )
    if type(exact_replay) is not bool:
        raise FastWisdomHolmanV9Error("exact_replay must be a built-in bool")
    primary = _execute_once(workspace)
    if exact_replay:
        replay = _execute_once(workspace)
        if (
            primary.runtime_identity_sha256 != replay.runtime_identity_sha256
            or primary.source_closure_sha256 != replay.source_closure_sha256
            or primary.content_sha256 != replay.content_sha256
            or primary.positions.tobytes(order="C")
            != replay.positions.tobytes(order="C")
            or primary.velocities.tobytes(order="C")
            != replay.velocities.tobytes(order="C")
            or primary.metrics.tobytes(order="C")
            != replay.metrics.tobytes(order="C")
            or primary.counters.tobytes(order="C")
            != replay.counters.tobytes(order="C")
            or primary.fallback_count != replay.fallback_count
            or primary.cache_stats != replay.cache_stats
            or _fp_control_fingerprint(primary.fp_environment_before)
            != _fp_control_fingerprint(replay.fp_environment_before)
            or _fp_control_fingerprint(primary.fp_environment_after)
            != _fp_control_fingerprint(replay.fp_environment_after)
        ):
            raise FastWisdomHolmanV9Error(
                "native V9 deterministic replay differs bitwise"
            )
        replay_counters_array = replay.counters
        replay_fallback_count = replay.fallback_count
        replay_cache_stats = replay.cache_stats
        replay_before = replay.fp_environment_before
        replay_after = replay.fp_environment_after
        replay_seconds = replay.seconds
        replay_count = 1
        verification_mode = CERTIFICATE_WITH_EXACT_REPLAY_MODE
        verification_scope = CERTIFICATE_WITH_EXACT_REPLAY_SCOPE
    else:
        replay_counters_array = np.zeros_like(primary.counters)
        replay_fallback_count = 0
        replay_cache_stats = tuple((name, 0) for name in CACHE_STAT_NAMES)
        replay_before = None
        replay_after = None
        replay_seconds = 0.0
        replay_count = 0
        verification_mode = CERTIFICATE_VERIFICATION_MODE
        verification_scope = CERTIFICATE_VERIFICATION_SCOPE

    positions = _owned_readonly(primary.positions, np.dtype(np.float64))
    velocities = _owned_readonly(primary.velocities, np.dtype(np.float64))
    epochs = _owned_readonly(workspace.checkpoint_epochs, np.dtype(np.float64))
    metrics = tuple(
        (name, float(value))
        for name, value in zip(METRIC_NAMES, primary.metrics, strict=True)
    )
    primary_counters = tuple(
        (name, int(value))
        for name, value in zip(COUNTER_NAMES, primary.counters, strict=True)
    )
    replay_counters = tuple(
        (name, int(value))
        for name, value in zip(
            COUNTER_NAMES, replay_counters_array, strict=True
        )
    )
    total_counters = tuple(
        (name, primary_value + replay_value)
        for (name, primary_value), (_, replay_value) in zip(
            primary_counters, replay_counters, strict=True
        )
    )
    counter_map = dict(primary_counters)
    certified_steps = counter_map["completed_steps"]
    certified_kepler_solves = counter_map["kepler_solves"]
    result_content_sha256 = _result_hash(
        input_content_sha256=workspace.input_content_sha256,
        runtime_identity_sha256=primary.runtime_identity_sha256,
        source_closure_sha256=primary.source_closure_sha256,
        positions=positions,
        velocities=velocities,
        checkpoint_epochs=epochs,
        metrics=metrics,
        counters=primary_counters,
        fallback_count=primary.fallback_count,
        cache_stats=primary.cache_stats,
        fp_environment_before=primary.fp_environment_before,
        fp_environment_after=primary.fp_environment_after,
    )
    if result_content_sha256 != primary.content_sha256:
        raise FastWisdomHolmanV9Error("V9 retained-result custody changed")
    verification_content_sha256 = _certificate_hash(
        input_content_sha256=workspace.input_content_sha256,
        result_content_sha256=result_content_sha256,
        runtime_identity_sha256=primary.runtime_identity_sha256,
        source_closure_sha256=primary.source_closure_sha256,
        verification_mode=verification_mode,
        certified_steps=certified_steps,
        certified_kepler_solves=certified_kepler_solves,
        primary_counters=primary_counters,
        replay_counters=replay_counters,
        primary_fallback_count=primary.fallback_count,
        replay_fallback_count=replay_fallback_count,
        primary_cache_stats=primary.cache_stats,
        replay_cache_stats=replay_cache_stats,
        primary_fp_environment_before=primary.fp_environment_before,
        primary_fp_environment_after=primary.fp_environment_after,
        replay_fp_environment_before=replay_before,
        replay_fp_environment_after=replay_after,
    )
    return FastWisdomHolmanV9LoopResult(
        positions=positions,
        velocities=velocities,
        checkpoint_epochs=epochs,
        metrics=metrics,
        primary_counters=primary_counters,
        replay_counters=replay_counters,
        total_counters=total_counters,
        primary_fallback_count=primary.fallback_count,
        replay_fallback_count=replay_fallback_count,
        total_fallback_count=primary.fallback_count + replay_fallback_count,
        primary_cache_stats=primary.cache_stats,
        replay_cache_stats=replay_cache_stats,
        primary_fp_environment_before=primary.fp_environment_before,
        primary_fp_environment_after=primary.fp_environment_after,
        replay_fp_environment_before=replay_before,
        replay_fp_environment_after=replay_after,
        primary_seconds=primary.seconds,
        replay_seconds=float(replay_seconds),
        replay_count=replay_count,
        verification_mode=verification_mode,
        verification_scope=verification_scope,
        certified_steps=certified_steps,
        certified_kepler_solves=certified_kepler_solves,
        full_step_postcondition_coverage=True,
        exact_replay_verified=exact_replay,
        independent_numerical_method_verified=False,
        deterministic_implementation_defect_excluded=False,
        verification_content_sha256=verification_content_sha256,
        result_content_sha256=result_content_sha256,
        input_content_sha256=workspace.input_content_sha256,
        runtime_identity_sha256=primary.runtime_identity_sha256,
        source_closure_sha256=primary.source_closure_sha256,
    )


def integrate_fast_wisdom_holman_v9_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepWisdomHolmanSpec,
    *,
    exact_replay: bool = False,
) -> FastWisdomHolmanV9LoopResult:
    """Validate inputs and execute the explicit supported-platform V9 route."""

    workspace = prepare_fast_wisdom_holman_v9_workspace(snapshot, plan, spec)
    return integrate_fast_wisdom_holman_v9_workspace(
        workspace, exact_replay=exact_replay
    )


__all__ = [
    "API_ID",
    "CACHE_STAT_NAMES",
    "CERTIFICATE_CONTENT_CHECKSUM_DOMAIN",
    "CERTIFICATE_VERIFICATION_MODE",
    "CERTIFICATE_VERIFICATION_SCOPE",
    "CERTIFICATE_WITH_EXACT_REPLAY_MODE",
    "CERTIFICATE_WITH_EXACT_REPLAY_SCOPE",
    "CHECKSUM_ALGORITHM",
    "COUNTER_NAMES",
    "EXECUTION_SCOPE",
    "EXACT_FALLBACK_KERNEL_ID",
    "FP_ENVIRONMENT_KEYS",
    "FastWisdomHolmanV9Error",
    "FastWisdomHolmanV9LoopResult",
    "FastWisdomHolmanV9Workspace",
    "G_KERNEL_ID",
    "KERNEL_ID",
    "METRIC_NAMES",
    "PUBLIC_DEFAULT",
    "REPLAY_SCOPE",
    "RESULT_CONTENT_CHECKSUM_DOMAIN",
    "RESULT_TYPE_ID",
    "RUNTIME_IDENTITY_CHECKSUM_DOMAIN",
    "SCHEMA",
    "SCIENTIFIC_CLAIM_STATE",
    "SOURCE_CLOSURE_CHECKSUM_DOMAIN",
    "SUPPORT_STATUS",
    "fast_wisdom_holman_v9_runtime_identity",
    "integrate_fast_wisdom_holman_v9_trajectory",
    "integrate_fast_wisdom_holman_v9_workspace",
    "prepare_fast_wisdom_holman_v9_workspace",
]
