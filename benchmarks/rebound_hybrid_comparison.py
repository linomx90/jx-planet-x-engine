#!/usr/bin/env python3
"""Optional close-scatter comparison for the JX hybrid and REBOUND 5.1.1.

The study is measurement-only.  REBOUND is loaded lazily as a separate
GPL-family external comparator.  It reports fixture-specific errors, empirical
orders, and descriptive thresholds; no general/theoretical accuracy or order,
timing, qualification, production, authority, or superiority claim is authorized.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from jxplanetx import __version__ as JX_VERSION
from jxplanetx.engine import (
    BackendSpec,
    EncounterExactRationalResourceSpec,
    EncounterDomainError,
    FixedStepWisdomHolmanSpec,
    ForcePlan,
    HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER,
    HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
    HybridDomainError,
    HybridEncounterControlProfile,
    HybridWisdomHolmanRKF78Spec,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
    integrate_hybrid_wisdom_holman_rkf78_trajectory,
    integrate_wisdom_holman_trajectory,
)


SCHEMA = "jx.rebound_hybrid.close_scatter.v1"
BENCHMARK_ID = "jx.rebound_hybrid.all_active_close_scatter.v1"
REQUIRED_REBOUND_VERSION = "5.1.1"
REBOUND_MODES = ("auto", "required", "disabled")
PROFILE_NAMES = ("smoke", "full-close-scatter")
BODY_IDS = ("STAR", "INNER", "OUTER")
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))
PERIOD = float.fromhex("0x1.921fb54442d18p+2")
JX_DIVISORS = (256, 512, 1024)
EXTERNAL_DIVISORS = (128, 256, 512, 1024)
JX_IAS15_INITIAL_DT = PERIOD / 128.0
JX_IAS15_SENSITIVITY_DT = PERIOD / 1024.0
INITIAL_ROWS_HEX = (
    (
        "0x1.0000000000000p+0",
        "0x0.0p+0",
        "-0x1.d8a0a98cbf91ep-8",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x0.0p+0",
        "-0x1.65b6be128f912p-8",
        "0x0.0p+0",
    ),
    (
        "0x1.0624dd2f1a9fcp-10",
        "0x0.0p+0",
        "0x1.fc4ebeace680ep-1",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.fd761789f219cp-1",
        "0x0.0p+0",
    ),
    (
        "0x1.47ae147ae147bp-8",
        "0x0.0p+0",
        "0x1.3e68d7cb11d03p+0",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.c9091de16980bp-1",
        "0x0.0p+0",
    ),
)
INITIAL_RAW_SHA256 = (
    "c2c5eb2a44a58fe1d8f61838cf1adb35030ed816f4b9038bc6b8a68b1b661f8c"
)

EXTERNAL_FINAL_STATE_SHA256 = {
    "mercurius": {
        128: "2caa86c640f18258b7729f1d2770a9ac4a3c99f15d50f46e79b992916a849221",
        256: "bed47b2dc26841dee8df14a9b00b20c6bd4d8ab68acd66f33662ad61a1708d18",
        512: "bd5ebc238757413ae8a5066b4f0299317de39d31a9f71eb988b5131eebfdb366",
        1024: "7f29ac73d32d34cee3862a023d86b94ca9fa0f2a774102989036edd362eaab3b",
    },
    "trace": {
        128: "f51743817ca9a414b11f444590f0cc8b98c18ba165684429abab4da286c68430",
        256: "e6a4d93c3091c1ece6bf4c5adaa56858ad403dc4bcbe911580a4d65bf6250628",
        512: "17012cfd9f129a8fdba80938c43ee31af8c780d5afda1200f64e82812daec8ac",
        1024: "38097470a3bb0b027d195b9de8bcf731f1ab137e06261e162a3f35cc5d6dd9af",
    },
}
EXTERNAL_TRAJECTORY_SHA256 = {
    "mercurius": {
        128: "e1fed42e10945ab8e1f443f43a803fdf34edd802a29adc3528f4c5124d497a1e",
        256: "b58ad0b86f75d499800fbf40439d78ad372184cf2558dc7c2fa01623df6d12d0",
        512: "77d2c349ad42d403de47daf5519fbfeb6eaee40b6d03ff369e7bd05b5f188c6a",
        1024: "f507833c936092cea9bfb98f9b4d3241759fd79ba6ccd60cc35df69f6d9d268a",
    },
    "trace": {
        128: "4110eac879f4e0d0b4edbe194a8f44c5af98279a3f34cf5bcaf2c16f12840dfe",
        256: "0a29643affa14469870041c48b6d9cd2f3a62ed702dd7f9ed3fd123ccd953c09",
        512: "9aa986bb919dc240e2dd96bf1965c39555f276585e491288f5a74f8b57851d38",
        1024: "fef92da5d75b57a61f658eae1f90daeb4969a19ad077df4a33b98683ab245395",
    },
}
EXTERNAL_IAS15_FINAL_STATE_SHA256 = {
    128: "0970d08aa11d1f383f1feda970a02dd977b59c3760d8b5953b1b3076bce4db4e",
    256: "9086d77c679bcba79e28ab4a2d20f95a1dd3224f1f5115dafc560d94eb28109c",
    512: "2b8883d661ec9f891aca261bb582e1a9438660ad4e4c24833dad69bbe7560926",
    1024: "23a14094443cd3f64c25ff83b4a11933a53020513ff5f86b8a383a2cbb089ded",
}
EXTERNAL_IAS15_TRAJECTORY_SHA256 = {
    128: "e8e8a4e5bd1c94d8b149889433907104790b17ec4bae2526b06b983ebba06a56",
    256: "55587ff17cb9b61b5c4f7a33ccd987eb18ba8752e41b639477569163cc24538e",
    512: "c57d883880f4240e50f6a87be1496f7dfa2553c0d063eaebfbf8716d74556ebf",
    1024: "da12cf998f7c1ba1c3846a3ea29ae833563f92fc90f35d8df06c19281c1c1cbf",
}
EXTERNAL_OBSERVED_FINAL_EPOCH_HEX = {
    128: "0x1.400cbc85142fbp+5",
    256: "0x1.400cbc8514323p+5",
    512: "0x1.400cbc851442bp+5",
    1024: "0x1.400cbc8514223p+5",
}
EXTERNAL_OBSERVED_CLOCK_VECTOR_SHA256 = {
    128: "4fdd2c5e21954aaafb637b95a0ef96c3a833874d226a15abb5b0b7b2c6b175fa",
    256: "dd1e6b90f75e4277d68b4b100257db1f5cd4bcf0ae3c21ccd150aa5fa4165500",
    512: "eb50bfca4e4fe9b3644b58b307d8015d30b08c7244a374129e63a538ea21b075",
    1024: "1650c724c179239f387b061ff0df286e6932c985f3d473ced4a9dd01646e2c85",
}
JX_IAS15_REFERENCE_KATS = {
    128: {
        "final_state_sha256": "bedca205d160af2b2cd7c13a7524a9d3db09b9937440fe6866bd77ad7a8915c9",
        "trajectory_sha256": "cdbca9002597c51f732767871ed464118cc84a41f6298929fff3460c01f9d736",
        "steps_done": 1551,
        "post_dt_hex": "0x1.1b5e52dc04702p-5",
        "post_dt_last_done_hex": "0x1.c0b2eddde2000p-6",
    },
    1024: {
        "final_state_sha256": "5e3d78231c5ac94feb9f5f77773db8f994a3ce981756f3c4a137753dc699bb6e",
        "trajectory_sha256": "eca943f2a5685291cb02e8b9119059293e714a3616950abc58fe1b4b3e2ad2ac",
        "steps_done": 1553,
        "post_dt_hex": "0x1.1b5e52dd6b9b9p-5",
        "post_dt_last_done_hex": "0x1.c0b2edd9bf000p-6",
    },
}

JX_FINEST_ENVELOPE = {
    "position_vector_l2_max": 4.0e-4,
    "position_vector_l2_rms": 8.0e-5,
    "velocity_vector_l2_max": 1.0e-4,
    "velocity_vector_l2_rms": 2.5e-5,
    "phase_max_abs_radians": 3.0e-4,
    "relative_total_energy_max": 5.0e-8,
    "relative_angular_momentum_max": 2.0e-14,
    "center_of_mass_position_max": 1.0e-12,
    "total_momentum_max": 1.0e-12,
}
JX_FINE_STATE_ORDER_MINIMUM = 1.5
JX_FINE_ENERGY_ORDER_MINIMUM = 1.7
EXTERNAL_ORDER_WINDOW = (1.8, 2.2)
REFERENCE_PERTURBATION_MAXIMUM = 5.0e-12
JX_EXPECTED_MODE_COUNTS = {
    256: (1022, 608),
    512: (2068, 1192),
    1024: (4153, 2367),
}
JX_FINAL_STATE_SHA256 = {
    256: "981afa9528ae2fd3ba3438288b1d57f44c183cfe4afc1063e7e891a98217e2c6",
    512: "5f6231cdf01585d1dac4ec3733a149047099b94b64e7d3505511bfb9ce60dd6b",
    1024: "ba0696c32c283c87bb4558a53dfcb5084d8e950fc7e3b8ac48675d1021d17930",
}
JX_TRAJECTORY_SHA256 = {
    256: "c0cd3cc6114b80619e48c94854eeaf6d021dc46d6d808ed586ae6c5a94ec2178",
    512: "6efb95e42207667c4cd2b584ed789b76d9788881e578d829685e74f0b125d3e3",
    1024: "6555ec6d6dd472a4a64104a9e0b5df741c8308116db2d3fbb4de9cc3b3814925",
}
MAXIMUM_RETAINED_NODES = 6521
MAXIMUM_CANONICAL_BYTES = 64 * 1024 * 1024
MAXIMUM_CANONICAL_INTEGER_BITS = 4096
CLAIM_CONTROLS = {
    "timing_comparable": False,
    "superiority_claimed": False,
    "qualification_claimed": False,
    "production_use_authorized": False,
    "reference_truth_claimed": False,
    "registry_authorized": False,
    "external_comparator_authorizes_jx": False,
    "global_order_claimed": False,
    "global_clearance_claimed": False,
    "collision_response_claimed": False,
    "regularization_claimed": False,
    "dense_output_claimed": False,
    "floating_point_symplecticity_claimed": False,
    "floating_point_reversibility_claimed": False,
    "long_term_boundedness_claimed": False,
}
FORBIDDEN_RESULT_KEYS = frozenset(("winner", "defeated", "speedup"))
__all__ = (
    "BENCHMARK_ID",
    "BenchmarkError",
    "Profile",
    "ReboundUnavailable",
    "SCHEMA",
    "build_report",
    "main",
    "run_comparison",
)


class BenchmarkError(RuntimeError):
    """The comparison contract could not be satisfied exactly."""


class ReboundUnavailable(BenchmarkError):
    """The exact optional REBOUND runtime is unavailable."""


def _load_provenance_support() -> Any:
    path = Path(__file__).resolve().with_name("rebound_leapfrog_comparison.py")
    if not path.is_file() or path.is_symlink():
        raise BenchmarkError("the frozen provenance-support source is unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"jx_rebound_leapfrog_provenance_support_{digest}"
    if name in sys.modules:
        raise BenchmarkError("preloaded provenance-support substitution detected")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BenchmarkError("the provenance-support module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    required = (
        "_exact_tree_equal",
        "_file_identity",
        "_engine_sources_manifest",
        "_numpy_runtime_identity",
        "_load_rebound",
        "_rebound_runtime_provenance",
        "_validate_rebound_runtime_provenance",
    )
    if (
        Path(module.__file__).resolve() != path
        or module.__loader__ is not spec.loader
        or module.__spec__ is not spec
        or Path(spec.origin).resolve() != path
        or module.REQUIRED_REBOUND_VERSION != REQUIRED_REBOUND_VERSION
    ):
        raise BenchmarkError("provenance-support identity is inconsistent")
    for name in required:
        value = getattr(module, name, None)
        source = inspect.getsourcefile(value) if callable(value) else None
        if source is None or Path(source).resolve() != path:
            raise BenchmarkError("provenance-support callable substitution detected")
    return module


_PROVENANCE_SUPPORT = _load_provenance_support()


@dataclass(frozen=True)
class Profile:
    name: str = "smoke"
    include_limitations: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in PROFILE_NAMES:
            raise ValueError(f"name must be one of {PROFILE_NAMES!r}")
        if type(self.include_limitations) is not bool:
            raise ValueError("include_limitations must be an exact bool")

    @classmethod
    def full(cls, *, include_limitations: bool = False) -> "Profile":
        return cls("full-close-scatter", include_limitations)

    @property
    def named_full_profile(self) -> bool:
        return self.name == "full-close-scatter"

    @property
    def base_steps_at_256(self) -> int:
        return 1630 if self.named_full_profile else 128

    @property
    def jx_divisors(self) -> tuple[int, ...]:
        return JX_DIVISORS if self.named_full_profile else (256,)

    @property
    def external_divisors(self) -> tuple[int, ...]:
        return EXTERNAL_DIVISORS if self.named_full_profile else (256,)

    def step_count(self, divisor: int) -> int:
        if type(divisor) is not int or divisor not in EXTERNAL_DIVISORS:
            raise ValueError("divisor is outside the exact comparison lattice")
        product = self.base_steps_at_256 * divisor
        if product % 256:
            raise ValueError("profile horizon is not an exact integer lattice")
        return product // 256

    def jx_checkpoint_indices(self, divisor: int) -> tuple[int, ...]:
        if divisor not in self.jx_divisors:
            raise ValueError("divisor is outside the JX profile")
        count = self.step_count(divisor)
        stride = divisor // 16
        values = tuple(range(0, count + 1, stride))
        if values[-1] != count:
            values += (count,)
        if self.named_full_profile and len(values) != 103:
            raise ValueError("full JX checkpoint roster must contain 103 labels")
        return values


def _initial_rows() -> np.ndarray:
    rows = np.array(
        [[float.fromhex(value) for value in row] for row in INITIAL_ROWS_HEX],
        dtype="<f8",
    )
    if rows.shape != (3, 8) or hashlib.sha256(rows.tobytes()).hexdigest() != INITIAL_RAW_SHA256:
        raise BenchmarkError("the locked initial Cartesian bytes changed")
    return rows


def _validate_tree_envelope(value: Any, *, label: str) -> None:
    remaining = 1_000_000
    reserved_bytes = 0

    def reserve(amount: int) -> None:
        nonlocal reserved_bytes
        if type(amount) is not int or amount < 0:
            raise BenchmarkError(f"{label} has an invalid canonical reservation")
        if amount > MAXIMUM_CANONICAL_BYTES - reserved_bytes:
            raise BenchmarkError(f"{label} exceeds its cumulative canonical byte cap")
        reserved_bytes += amount

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise BenchmarkError(f"{label} exceeds its canonical node/depth cap")
        reserve(16)
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item.bit_length() > MAXIMUM_CANONICAL_INTEGER_BITS:
                raise BenchmarkError(f"{label} integer exceeds its bit cap")
            reserve(max(2, 2 * item.bit_length()))
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise BenchmarkError(f"{label} contains a nonfinite float")
            reserve(64)
            return
        if type(item) is str:
            if len(item) > 16_384:
                raise BenchmarkError(f"{label} string exceeds its code-point cap")
            try:
                encoded_length = _strict_utf8_length(item)
            except UnicodeEncodeError as exc:
                raise BenchmarkError(f"{label} string is not valid Unicode") from exc
            if encoded_length > 16_384:
                raise BenchmarkError(f"{label} string exceeds its byte cap")
            reserve(2 + 6 * encoded_length)
            return
        if type(item) is np.ndarray:
            if item.size > 200_000:
                raise BenchmarkError(f"{label} array exceeds its element cap")
            if item.dtype == np.dtype(np.float64):
                if not item.flags.c_contiguous or not np.all(np.isfinite(item)):
                    raise BenchmarkError(f"{label} float array is not finite C-order")
                reserve(128 + 32 * int(item.size) + 32 * item.ndim)
            elif item.dtype == np.dtype(np.bool_):
                if not item.flags.c_contiguous:
                    raise BenchmarkError(f"{label} bool array is not C-order")
                reserve(128 + 6 * int(item.size) + 32 * item.ndim)
            else:
                raise BenchmarkError(f"{label} array has unsupported dtype")
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            descriptors = fields(item)
            if len(descriptors) > 256:
                raise BenchmarkError(f"{label} dataclass exceeds its field cap")
            reserve(128 + 8 * len(descriptors))
            visit(f"{type(item).__module__}.{type(item).__qualname__}", depth + 1)
            for descriptor in descriptors:
                visit(descriptor.name, depth + 1)
                visit(getattr(item, descriptor.name), depth + 1)
            return
        if isinstance(item, Mapping):
            if len(item) > 256:
                raise BenchmarkError(f"{label} mapping exceeds its entry cap")
            reserve(2 + 2 * len(item))
            for key, nested in item.items():
                if type(key) is not str:
                    raise BenchmarkError(f"{label} mapping key is not built-in str")
                visit(key, depth + 1)
                visit(nested, depth + 1)
            return
        if type(item) in (tuple, list):
            if len(item) > 65_536:
                raise BenchmarkError(f"{label} sequence exceeds its entry cap")
            reserve(2 + len(item))
            for nested in item:
                visit(nested, depth + 1)
            return
        raise BenchmarkError(f"{label} has unsupported type {type(item).__name__}")

    visit(value, 0)


def _strict_utf8_length(value: str) -> int:
    return len(value.encode("utf-8", errors="strict"))


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise BenchmarkError("canonical JSON rejects nonfinite floats")
        return {"float_hex": value.hex()}
    if type(value) is np.ndarray:
        if value.dtype == np.dtype(np.float64):
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise BenchmarkError("canonical arrays must be finite C-order float64")
            values = [float(item).hex() for item in value.ravel(order="C")]
        elif value.dtype == np.dtype(np.bool_):
            if not value.flags.c_contiguous:
                raise BenchmarkError("canonical boolean arrays must be C-order")
            values = [bool(item) for item in value.ravel(order="C")]
        else:
            raise BenchmarkError("canonical arrays have unsupported dtype")
        return {"dtype": value.dtype.name, "shape": list(value.shape), "values": values}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _json_value(getattr(value, field.name)) for field in fields(value)},
        }
    if type(value) in (tuple, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise BenchmarkError("canonical mapping keys must be built-in strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    raise BenchmarkError(f"unsupported canonical value type {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    _validate_tree_envelope(value, label="canonical value")
    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > MAXIMUM_CANONICAL_BYTES:
        raise BenchmarkError("canonical JSON exceeds its final byte cap")
    return encoded


def _freeze_tree(value: Any) -> Any:
    _validate_tree_envelope(value, label="canonical custody tree")
    canonical = _json_value(value)

    def freeze(item: Any) -> Any:
        if type(item) is dict:
            return MappingProxyType({key: freeze(item[key]) for key in sorted(item)})
        if type(item) is list:
            return tuple(freeze(nested) for nested in item)
        return item

    return freeze(canonical)


_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _thaw_tree(value: Any) -> Any:
    if type(value) is _MAPPING_PROXY_TYPE:
        return {key: _thaw_tree(value[key]) for key in value}
    if type(value) is tuple:
        return [_thaw_tree(item) for item in value]
    return value


def _domain_sha256(domain: str, value: Any) -> str:
    if type(domain) is not str or not domain or domain.strip() != domain:
        raise BenchmarkError("checksum domain must be an explicit built-in string")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_json(value)).hexdigest()


def _readonly(array: Any, shape: tuple[int, ...]) -> np.ndarray:
    value = np.array(array, dtype=np.float64, order="C", copy=True)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise BenchmarkError(f"trajectory array must be finite with shape {shape!r}")
    value.setflags(write=False)
    return value


def _load_rebound() -> Any:
    try:
        return _PROVENANCE_SUPPORT._load_rebound()
    except _PROVENANCE_SUPPORT.ReboundUnavailable as exc:
        raise ReboundUnavailable(str(exc)) from exc
    except _PROVENANCE_SUPPORT.BenchmarkError as exc:
        raise BenchmarkError(str(exc)) from exc


def _jx_runtime_provenance() -> dict[str, Any]:
    return {
        "classification": "PROVENANCE_DIAGNOSTIC",
        "authority_authorized": False,
        "jxplanetx_version": JX_VERSION,
        "engine_sources": _PROVENANCE_SUPPORT._engine_sources_manifest(),
        "benchmark_script": _PROVENANCE_SUPPORT._file_identity(__file__),
        "provenance_support_script": _PROVENANCE_SUPPORT._file_identity(
            _PROVENANCE_SUPPORT.__file__
        ),
        "numpy": _PROVENANCE_SUPPORT._numpy_runtime_identity(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _validate_jx_runtime_provenance(runtime: Any) -> None:
    if not _PROVENANCE_SUPPORT._exact_tree_equal(runtime, _jx_runtime_provenance()):
        raise BenchmarkError("JX runtime provenance differs from loaded source bytes")


@dataclass(frozen=True, eq=False)
class Lane:
    engine_id: str
    family: str
    method_id: str
    divisor: int
    fixed_step: float
    step_count: int
    declared_epochs: np.ndarray
    observed_epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    settings: Mapping[str, Any]
    accounting: Mapping[str, Any]
    runtime: Mapping[str, Any]
    raw_timing_seconds: float
    final_state_sha256: str
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not Lane:
            raise BenchmarkError("lane must have its exact type")
        for name in ("engine_id", "family", "method_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise BenchmarkError(f"{name} must be an explicit built-in string")
        if type(self.divisor) is not int or self.divisor not in EXTERNAL_DIVISORS:
            raise BenchmarkError("lane divisor is outside the locked roster")
        if type(self.step_count) is not int or self.step_count <= 0:
            raise BenchmarkError("step_count must be a positive built-in integer")
        if self.step_count > MAXIMUM_RETAINED_NODES - 1:
            raise BenchmarkError("step_count exceeds the locked retained-node cap")
        if (
            type(self.fixed_step) is not float
            or not math.isfinite(self.fixed_step)
            or self.fixed_step <= 0.0
            or self.fixed_step.hex() != (PERIOD / self.divisor).hex()
        ):
            raise BenchmarkError("fixed_step differs from the exact positive lattice")
        if (
            type(self.declared_epochs) is not np.ndarray
            or type(self.observed_epochs) is not np.ndarray
        ):
            raise BenchmarkError("epoch arrays must be exact ndarray objects")
        count = len(self.declared_epochs)
        if not 2 <= count <= MAXIMUM_RETAINED_NODES:
            raise BenchmarkError("lane epoch count exceeds the locked retained-node cap")
        if (
            self.declared_epochs.dtype != np.dtype(np.float64)
            or self.observed_epochs.dtype != np.dtype(np.float64)
            or self.declared_epochs.shape != (count,)
            or self.observed_epochs.shape != (count,)
            or count < 2
            or not self.declared_epochs.flags.c_contiguous
            or not self.observed_epochs.flags.c_contiguous
            or self.declared_epochs.flags.writeable
            or self.observed_epochs.flags.writeable
            or not self.declared_epochs.flags.owndata
            or not self.observed_epochs.flags.owndata
            or self.declared_epochs.base is not None
            or self.observed_epochs.base is not None
            or not np.all(np.isfinite(self.declared_epochs))
            or not np.all(np.isfinite(self.observed_epochs))
            or np.any(np.diff(self.declared_epochs) <= 0.0)
            or np.any(np.diff(self.observed_epochs) <= 0.0)
        ):
            raise BenchmarkError("lane epochs lost finite readonly increasing custody")
        for name in ("positions", "velocities"):
            value = getattr(self, name)
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float64)
                or value.shape != (count, 3, 3)
                or not value.flags.c_contiguous
                or value.flags.writeable
                or not value.flags.owndata
                or value.base is not None
                or not np.all(np.isfinite(value))
            ):
                raise BenchmarkError(f"{name} lost exact readonly Cartesian custody")
        arrays = (
            self.declared_epochs,
            self.observed_epochs,
            self.positions,
            self.velocities,
        )
        if any(
            np.shares_memory(left, right)
            for index, left in enumerate(arrays)
            for right in arrays[index + 1 :]
        ):
            raise BenchmarkError("lane arrays must be mutually disjoint")
        for name in ("declared_epochs", "observed_epochs", "positions", "velocities"):
            value = getattr(self, name)
            object.__setattr__(self, name, _readonly(value, value.shape))
        for name in ("settings", "accounting", "runtime"):
            value = getattr(self, name)
            if type(value) in (dict, _MAPPING_PROXY_TYPE):
                maximum_keys = {"settings": 64, "accounting": 128, "runtime": 32}[name]
                if len(value) > maximum_keys:
                    raise BenchmarkError(f"{name} exceeds its top-level field cap")
                _validate_tree_envelope(value, label=name)
                value = _freeze_tree(value)
                object.__setattr__(self, name, value)
            if type(value) is not _MAPPING_PROXY_TYPE or any(
                type(key) is not str for key in value
            ):
                raise BenchmarkError(f"{name} must be an immutable canonical mapping")
            maximum_keys = {"settings": 64, "accounting": 128, "runtime": 32}[name]
            if len(value) > maximum_keys:
                raise BenchmarkError(f"{name} exceeds its top-level field cap")
            _canonical_json(value)
        if (
            type(self.raw_timing_seconds) is not float
            or not math.isfinite(self.raw_timing_seconds)
            or self.raw_timing_seconds < 0.0
        ):
            raise BenchmarkError("raw_timing_seconds must be a finite nonnegative float")
        for name in ("final_state_sha256", "trajectory_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise BenchmarkError(f"{name} must be a lowercase SHA-256 digest")
        if self.final_state_sha256 != _final_state_sha256(self.positions[-1], self.velocities[-1]):
            raise BenchmarkError("final-state raw digest differs from retained state")
        if self.trajectory_sha256 != _trajectory_sha256(
            self.observed_epochs, self.positions, self.velocities
        ):
            raise BenchmarkError("trajectory raw digest differs from retained state")


@dataclass(frozen=True, eq=False)
class StudyRun:
    profile: Profile
    rebound_mode: str
    jx_lanes: tuple[Lane, ...]
    external_lanes: tuple[Lane, ...]
    jx_reference: Lane | None
    jx_reference_sensitivity: Lane | None
    external_references: tuple[Lane, ...]
    external_status: Mapping[str, Any]
    all_far_control: Mapping[str, Any]
    limitation_probes: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self) is not StudyRun or type(self.profile) is not Profile:
            raise BenchmarkError("study/profile must have exact types")
        self.profile.__post_init__()
        if type(self.rebound_mode) is not str or self.rebound_mode not in REBOUND_MODES:
            raise BenchmarkError("study rebound_mode is outside the exact roster")
        for name in ("jx_lanes", "external_lanes", "external_references"):
            value = getattr(self, name)
            if type(value) is not tuple:
                raise BenchmarkError(f"{name} must be an exact tuple of Lane")
            maximum = {"jx_lanes": 3, "external_lanes": 8, "external_references": 4}[name]
            if len(value) > maximum:
                raise BenchmarkError(f"{name} exceeds its exact lane-count cap")
            if any(type(item) is not Lane for item in value):
                raise BenchmarkError(f"{name} must be an exact tuple of Lane")
            for item in value:
                item.__post_init__()
        if tuple(lane.divisor for lane in self.jx_lanes) != self.profile.jx_divisors:
            raise BenchmarkError("JX lanes differ from their profile divisor roster")
        for name in ("jx_reference", "jx_reference_sensitivity"):
            value = getattr(self, name)
            if value is not None and type(value) is not Lane:
                raise BenchmarkError(f"{name} must be None or an exact Lane")
            if value is not None:
                value.__post_init__()
        for name in ("external_status", "all_far_control", "limitation_probes"):
            value = getattr(self, name)
            if type(value) in (dict, _MAPPING_PROXY_TYPE):
                if len(value) > 64:
                    raise BenchmarkError(f"{name} exceeds its top-level field cap")
                _validate_tree_envelope(value, label=name)
                value = _freeze_tree(value)
                object.__setattr__(self, name, value)
            if type(value) is not _MAPPING_PROXY_TYPE:
                raise BenchmarkError(f"{name} must be an immutable canonical mapping")
            if len(value) > 64:
                raise BenchmarkError(f"{name} exceeds its top-level field cap")
            _canonical_json(value)
        _validate_study_rosters(self)


def _final_state_sha256(positions: Any, velocities: Any) -> str:
    rows = _initial_rows()
    pos = np.ascontiguousarray(positions, dtype="<f8")
    vel = np.ascontiguousarray(velocities, dtype="<f8")
    if pos.shape != (3, 3) or vel.shape != (3, 3):
        raise BenchmarkError("final state arrays must have shape (3,3)")
    payload = np.concatenate((rows[:, :2], pos, vel), axis=1).astype("<f8")
    return hashlib.sha256(payload.tobytes(order="C")).hexdigest()


def _trajectory_sha256(epochs: Any, positions: Any, velocities: Any) -> str:
    epoch = np.ascontiguousarray(epochs, dtype="<f8")
    pos = np.ascontiguousarray(positions, dtype="<f8")
    vel = np.ascontiguousarray(velocities, dtype="<f8")
    if epoch.ndim != 1 or pos.shape != vel.shape or pos.shape != (epoch.size, 3, 3):
        raise BenchmarkError("trajectory digest arrays have inconsistent shapes")
    state = np.concatenate((pos, vel), axis=2).astype("<f8")
    return hashlib.sha256(epoch.tobytes(order="C") + state.tobytes(order="C")).hexdigest()


def _lane_content_payload(lane: Lane) -> dict[str, Any]:
    if type(lane) is not Lane:
        raise BenchmarkError("lane content requires an exact Lane")
    lane.__post_init__()
    return {
        "schema": "jx.rebound-hybrid.lane-content.payload.v1",
        "engine_id": lane.engine_id,
        "family": lane.family,
        "method_id": lane.method_id,
        "divisor": lane.divisor,
        "fixed_step": lane.fixed_step,
        "step_count": lane.step_count,
        "declared_epochs_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.declared-epochs.v1", lane.declared_epochs
        ),
        "observed_epochs_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.observed-epochs.v1", lane.observed_epochs
        ),
        "positions_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.positions.v1", lane.positions
        ),
        "velocities_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.velocities.v1", lane.velocities
        ),
        "settings": lane.settings,
        "accounting": lane.accounting,
        "runtime": lane.runtime,
        "final_state_sha256": lane.final_state_sha256,
        "trajectory_sha256": lane.trajectory_sha256,
    }


def _lane_content_sha256(lane: Lane) -> str:
    return _domain_sha256(
        "jx.rebound-hybrid.lane-content.v1", _lane_content_payload(lane)
    )


def _array_component_sha256(domain: str, value: np.ndarray) -> str:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise BenchmarkError("array component requires an exact float64 ndarray")
    if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise BenchmarkError("array component requires finite C-order storage")
    header = _canonical_json(
        {"dtype": "float64", "shape": tuple(int(item) for item in value.shape)}
    )
    payload = np.ascontiguousarray(value, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\0"
        + len(header).to_bytes(8, "big")
        + header
        + len(payload).to_bytes(8, "big")
        + payload
    ).hexdigest()


def _cartesian_error(candidate: Any, reference: Any) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (3, 3):
        raise BenchmarkError("Cartesian error arrays must share shape (count,3,3)")
    difference = left - right
    if not np.all(np.isfinite(difference)):
        raise BenchmarkError("Cartesian error arrays became nonfinite")
    vector = np.linalg.norm(difference, axis=2)
    return {
        "component_max_abs": float(np.max(np.abs(difference))),
        "component_rms": float(np.sqrt(np.mean(difference * difference))),
        "vector_l2_max": float(np.max(vector)),
        "vector_l2_rms": float(np.sqrt(np.mean(vector * vector))),
        "final_vector_l2_max": float(np.max(vector[-1])),
    }


def _pair_arrays(values: Any) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    if checked.ndim != 3 or checked.shape[1:] != (3, 3):
        raise BenchmarkError("pair arrays require shape (count,3,3)")
    return checked[:, 2] - checked[:, 1]


def _pair_error(candidate: Any, reference: Any) -> dict[str, float]:
    left = _pair_arrays(candidate)
    right = _pair_arrays(reference)
    difference = left - right
    vector = np.linalg.norm(difference, axis=1)
    return {
        "component_max_abs": float(np.max(np.abs(difference))),
        "component_rms": float(np.sqrt(np.mean(difference * difference))),
        "vector_l2_max": float(np.max(vector)),
        "vector_l2_rms": float(np.sqrt(np.mean(vector * vector))),
        "final_vector_l2": float(vector[-1]),
    }


def _separation_error(candidate: Any, reference: Any) -> dict[str, float]:
    left = np.linalg.norm(_pair_arrays(candidate), axis=1)
    right = np.linalg.norm(_pair_arrays(reference), axis=1)
    difference = np.abs(left - right)
    return {
        "maximum_abs": float(np.max(difference)),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "final_abs": float(difference[-1]),
        "candidate_minimum": float(np.min(left)),
        "reference_minimum": float(np.min(right)),
    }


def _phase_error(candidate: Any, reference: Any) -> dict[str, Any]:
    left = _pair_arrays(candidate)[:, :2]
    right = _pair_arrays(reference)[:, :2]
    if np.any(np.linalg.norm(left, axis=1) <= 0.0) or np.any(np.linalg.norm(right, axis=1) <= 0.0):
        raise BenchmarkError("phase proxy requires nonzero inner-outer vectors")
    cross = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    dot = np.sum(left * right, axis=1)
    phase = np.unwrap(np.arctan2(cross, dot))
    return {
        "proxy": "INNER_TO_OUTER_RELATIVE_POSITION_VECTOR_ANGLE_IN_XY_PLANE",
        "sign_convention": "ATAN2_CROSS_CANDIDATE_REFERENCE_REFERENCE_MINUS_CANDIDATE",
        "maximum_abs_radians": float(np.max(np.abs(phase))),
        "rms_radians": float(np.sqrt(np.mean(phase * phase))),
        "final_signed_radians": float(phase[-1]),
    }


def _invariant_series(positions: Any, velocities: Any) -> tuple[np.ndarray, ...]:
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.shape != vel.shape or pos.ndim != 3 or pos.shape[1:] != (3, 3):
        raise BenchmarkError("invariant arrays require shared shape (count,3,3)")
    masses = _initial_rows()[:, 0].astype(np.float64)
    count = pos.shape[0]
    energy = np.empty(count, dtype=np.float64)
    angular = np.empty((count, 3), dtype=np.float64)
    momentum = np.empty((count, 3), dtype=np.float64)
    center = np.empty((count, 3), dtype=np.float64)
    total_mass = float(sum(float(value) for value in masses))
    for index in range(count):
        kinetic = 0.0
        potential = 0.0
        angular_value = np.zeros(3, dtype=np.float64)
        momentum_value = np.zeros(3, dtype=np.float64)
        center_value = np.zeros(3, dtype=np.float64)
        for body in range(3):
            mass = float(masses[body])
            kinetic += 0.5 * mass * float(np.dot(vel[index, body], vel[index, body]))
            angular_value += mass * np.cross(pos[index, body], vel[index, body])
            momentum_value += mass * vel[index, body]
            center_value += mass * pos[index, body]
        for first, second in PAIR_ORDER:
            separation = float(np.linalg.norm(pos[index, second] - pos[index, first]))
            if not math.isfinite(separation) or separation <= 0.0:
                raise BenchmarkError("invariant evaluation reached nonpositive separation")
            potential -= float(masses[first]) * float(masses[second]) / separation
        energy[index] = kinetic + potential
        angular[index] = angular_value
        momentum[index] = momentum_value
        center[index] = center_value / total_mass
    return energy, angular, momentum, center


def _invariant_metrics(positions: Any, velocities: Any) -> dict[str, float]:
    energy, angular, momentum, center = _invariant_series(positions, velocities)
    energy_error = np.abs((energy - energy[0]) / energy[0])
    angular_error = np.linalg.norm(angular - angular[0], axis=1) / np.linalg.norm(angular[0])
    momentum_error = np.linalg.norm(momentum - momentum[0], axis=1)
    center_error = np.linalg.norm(center - center[0], axis=1)
    momentum_absolute = np.linalg.norm(momentum, axis=1)
    center_absolute = np.linalg.norm(center, axis=1)
    result: dict[str, float] = {}
    for label, values in (
        ("relative_total_energy", energy_error),
        ("relative_angular_momentum", angular_error),
        ("total_momentum_drift", momentum_error),
        ("center_of_mass_position_drift", center_error),
        ("total_momentum_absolute", momentum_absolute),
        ("center_of_mass_position_absolute", center_absolute),
    ):
        result[f"{label}_max"] = float(np.max(values))
        result[f"{label}_rms"] = float(np.sqrt(np.mean(values * values)))
        result[f"{label}_final"] = float(values[-1])
    return result


def _accuracy_metrics(candidate: Lane, reference: Lane) -> dict[str, Any]:
    if candidate.observed_epochs.shape != reference.observed_epochs.shape or any(
        float(left).hex() != float(right).hex()
        for left, right in zip(candidate.observed_epochs, reference.observed_epochs)
    ):
        raise BenchmarkError("accuracy lanes do not share exact observed epochs")
    return {
        "position": _cartesian_error(candidate.positions, reference.positions),
        "velocity": _cartesian_error(candidate.velocities, reference.velocities),
        "inner_outer_pair_position": _pair_error(candidate.positions, reference.positions),
        "inner_outer_pair_velocity": _pair_error(candidate.velocities, reference.velocities),
        "inner_outer_separation": _separation_error(candidate.positions, reference.positions),
        "inner_outer_relative_vector_phase_proxy": _phase_error(
            candidate.positions, reference.positions
        ),
    }


def _order(coarse: float, fine: float) -> float:
    if not (math.isfinite(coarse) and math.isfinite(fine) and coarse > 0.0 and fine > 0.0):
        raise BenchmarkError("refinement order requires finite positive errors")
    return float(math.log(coarse / fine, 2.0))


def _resources() -> EncounterExactRationalResourceSpec:
    return EncounterExactRationalResourceSpec(
        maximum_body_count=16,
        maximum_pair_count=120,
        maximum_integer_bits=8192,
        maximum_rational_exponent_magnitude=4096,
        maximum_gcd_iterations_per_reduction=16384,
        maximum_initialization_operations=100_000,
        maximum_initialization_gcd_iterations=500_000,
        maximum_initialization_transcript_bytes=1_048_576,
        maximum_gcd_iterations_per_proposal=500_000,
        maximum_gcd_iterations_per_segment=64_500_000,
        maximum_operations_per_proposal=650_000,
        maximum_operations_per_segment=83_450_000,
        maximum_witness_transcript_bytes_per_proposal=1_310_720,
        maximum_witness_transcript_bytes_per_segment=168_820_736,
        maximum_witness_diagnostic_bytes_per_proposal=4096,
        maximum_witness_ledger_bytes=16_777_216,
    )


def _state_and_plan(profile: Profile) -> tuple[StateSnapshot, ForcePlan]:
    if type(profile) is not Profile:
        raise BenchmarkError("state construction requires an exact Profile")
    rows = _initial_rows()
    source_sha256 = _PROVENANCE_SUPPORT._file_identity(__file__)["sha256"]
    provenance = Provenance(
        "jx.benchmark.hybrid.close-scatter.v1",
        "benchmarks/rebound_hybrid_comparison.py",
        "1",
        source_sha256,
    )
    unit_system_id = "jx.benchmark.hybrid.synthetic.v1"
    snapshot = StateSnapshot(
        "jx.benchmark.hybrid.close-scatter.snapshot.v1",
        0.0,
        "TDB",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "ICRS_ALIGNED",
        "L",
        "T",
        "M",
        unit_system_id,
        BODY_IDS,
        rows[:, 2:5].astype(np.float64).copy(),
        rows[:, 5:8].astype(np.float64).copy(),
        rows[:, 0].astype(np.float64).copy(),
        rows[:, 0].astype(np.float64).copy(),
        rows[:, 1].astype(np.float64).copy(),
        np.ones(3, dtype=np.bool_),
        provenance,
    )
    horizon = float((PERIOD / 256.0) * profile.base_steps_at_256)
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        -1.0,
        float(horizon + PERIOD),
    )
    plan = ForcePlan(
        "jx.benchmark.hybrid.close-scatter.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                BODY_IDS,
                BODY_IDS,
                unit_system_id,
                (metadata,),
            ),
        ),
    )
    return snapshot, plan


def _hybrid_spec(
    profile: Profile, state: StateSnapshot, divisor: int
) -> HybridWisdomHolmanRKF78Spec:
    count = profile.step_count(divisor)
    step = float(PERIOD / divisor)
    wh = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=profile.jx_checkpoint_indices(divisor),
        fixed_step=step,
        maximum_steps=count,
        jacobi_body_order=state.body_ids,
        minimum_encounter_pair_separation=0.01,
        minimum_jacobi_periapse=0.05,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )
    encounter = HybridEncounterControlProfile(
        initial_step_magnitude=step,
        minimum_step_magnitude=step / 128.0,
        maximum_step_magnitude=step,
        pair_certification_floors=(0.25, 0.25, 0.01),
        pair_position_atols=(1.0e-10, 1.0e-10, 1.0e-10),
        pair_position_rtol=1.0e-12,
        pair_velocity_atols=(1.0e-10, 1.0e-10, 1.0e-10),
        pair_velocity_rtol=1.0e-12,
        gm_centroid_position_atol=1.0e-12,
        gm_centroid_position_rtol=1.0e-12,
        gm_centroid_velocity_atol=1.0e-12,
        gm_centroid_velocity_rtol=1.0e-12,
        maximum_substep_proposals=128,
        maximum_accepted_substeps=128,
        maximum_rejected_substeps=64,
        maximum_consecutive_rejections=32,
        maximum_force_evaluations=1664,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
        exact_rational_resources=_resources(),
    )
    return HybridWisdomHolmanRKF78Spec(wh, encounter)


def _mode_run_length(records: Any) -> tuple[dict[str, Any], ...]:
    checked = tuple(records)
    if not checked:
        raise BenchmarkError("hybrid mode ledger cannot be empty")
    runs: list[dict[str, Any]] = []
    start = checked[0].outer_step_index
    previous = None
    for record in checked:
        key = (
            record.mode,
            record.decision.reason,
            record.decision.phase,
            record.decision.body_indices,
            record.decision.pair_indices,
        )
        if previous is None:
            previous = key
        elif key != previous:
            runs.append(
                {
                    "first_outer_step": start,
                    "last_outer_step": record.outer_step_index - 1,
                    "mode": previous[0],
                    "reason": previous[1],
                    "phase": previous[2],
                    "body_indices": previous[3],
                    "pair_indices": previous[4],
                }
            )
            start = record.outer_step_index
            previous = key
    assert previous is not None
    runs.append(
        {
            "first_outer_step": start,
            "last_outer_step": checked[-1].outer_step_index,
            "mode": previous[0],
            "reason": previous[1],
            "phase": previous[2],
            "body_indices": previous[3],
            "pair_indices": previous[4],
        }
    )
    return tuple(runs)


def _dataclass_numeric_double(primary: Any, total: Any) -> bool:
    if not (
        dataclasses.is_dataclass(primary)
        and dataclasses.is_dataclass(total)
        and not isinstance(primary, type)
        and type(primary) is type(total)
    ):
        return False
    for descriptor in fields(primary):
        left = getattr(primary, descriptor.name)
        right = getattr(total, descriptor.name)
        if dataclasses.is_dataclass(left) and not isinstance(left, type):
            if not _dataclass_numeric_double(left, right):
                return False
        elif type(left) is int:
            if type(right) is not int or right != 2 * left:
                return False
        else:
            return False
    return True


def _hybrid_accounting(result: Any) -> dict[str, Any]:
    records = tuple(result.outer_step_records)
    children = tuple(
        record.private_encounter
        for record in records
        if record.private_encounter is not None
    )
    mode_counts = Counter(record.mode for record in records)
    reason_counts = Counter(
        record.decision.reason for record in records if not record.probe_committed
    )
    phase_counts = Counter(
        record.decision.phase for record in records if not record.probe_committed
    )
    trigger_counts = Counter(
        (
            record.decision.reason,
            record.decision.body_indices,
            record.decision.pair_indices,
        )
        for record in records
        if not record.probe_committed
    )
    child_digests = tuple(child.private_execution_content_sha256 for child in children)
    child_summary = {
        "count": len(children),
        "proposal_count": sum(child.proposal_count for child in children),
        "accepted_substep_count": sum(
            child.accepted_substep_count for child in children
        ),
        "rejected_substep_count": sum(
            child.rejected_substep_count for child in children
        ),
        "force_evaluations": sum(child.force_evaluations for child in children),
        "maximum_exact_operations": max(
            (
                child.general_rational_operation_count
                + child.dyadic_operation_count
                for child in children
            ),
            default=0,
        ),
        "maximum_gcd_iterations": max(
            (child.gcd_iteration_count for child in children), default=0
        ),
        "maximum_transcript_bytes": max(
            (child.transcript_byte_count for child in children), default=0
        ),
        "ordered_private_execution_digest_count": len(child_digests),
        "ordered_private_execution_digests_sha256": _domain_sha256(
            "jx.rebound-hybrid.ordered-private-child-digests.v1", child_digests
        ),
        "accepted_certified_substeps_for_this_locked_workload": sum(
            child.accepted_substep_count for child in children
        ),
        "certificate_rejections": result.primary_counts.private_encounter_counts.certificate_rejections,
        "exact_margin_retained_by_public_result": False,
        "each_private_child_one_proposal_one_accept_zero_reject_thirteen_forces": all(
            child.proposal_count == 1
            and child.accepted_substep_count == 1
            and child.rejected_substep_count == 0
            and child.force_evaluations == 13
            for child in children
        ),
    }
    return {
        "custody_source": "HybridWisdomHolmanRKF78Result",
        "method_id": result.method_id,
        "completed_steps": result.completed_steps,
        "checkpoint_count": result.checkpoint_count,
        "schedule_content_sha256": result.schedule_content_sha256,
        "step_ledger_content_sha256": result.step_ledger_content_sha256,
        "result_content_sha256": result.result_content_sha256,
        "wisdom_holman_schedule_content_sha256": (
            result.wisdom_holman_schedule_content_sha256
        ),
        "wisdom_holman_result_content_sha256": (
            result.wisdom_holman_result_content_sha256
        ),
        "primary_counts": result.primary_counts,
        "validation_replay_counts": result.validation_replay_counts,
        "total_public_call_counts": result.total_public_call_counts,
        "primary_counts_equal_validation_replay_counts": _canonical_equal(
            result.primary_counts, result.validation_replay_counts
        ),
        "total_public_counts_exactly_double_primary": _dataclass_numeric_double(
            result.primary_counts, result.total_public_call_counts
        ),
        "mode_counts": {key: mode_counts[key] for key in sorted(mode_counts)},
        "near_reason_counts": {
            key: reason_counts[key] for key in sorted(reason_counts)
        },
        "near_phase_counts": {
            key: phase_counts[key] for key in sorted(phase_counts)
        },
        "near_trigger_counts": tuple(
            {
                "reason": key[0],
                "body_indices": key[1],
                "pair_indices": key[2],
                "count": trigger_counts[key],
            }
            for key in sorted(trigger_counts)
        ),
        "mode_reason_runs": _mode_run_length(records),
        "child_summary": child_summary,
        "force_model_ids": result.force_model_ids,
        "force_ledger_sha256": _domain_sha256(
            "jx.rebound-hybrid.force-ledger.v1", result.force_ledger
        ),
    }


def _jx_lane(profile: Profile, divisor: int) -> Lane:
    state, plan = _state_and_plan(profile)
    spec = _hybrid_spec(profile, state, divisor)
    started = time.perf_counter()
    result = integrate_hybrid_wisdom_holman_rkf78_trajectory(state, plan, spec)
    elapsed = float(time.perf_counter() - started)
    epochs = np.array(result.checkpoint_epochs, dtype=np.float64, copy=True)
    positions = np.stack(result.positions).astype(np.float64, copy=False)
    velocities = np.stack(result.velocities).astype(np.float64, copy=False)
    expected_epochs = np.array(
        [
            float(index * spec.wisdom_holman_spec.fixed_step)
            for index in spec.wisdom_holman_spec.checkpoint_step_indices
        ],
        dtype=np.float64,
    )
    if not np.array_equal(epochs.view(np.uint64), expected_epochs.view(np.uint64)):
        raise BenchmarkError("JX returned a checkpoint outside its exact integer lattice")
    for value in (epochs, positions, velocities):
        value.setflags(write=False)
    settings = {
        "method_id": result.method_id,
        "wisdom_holman_spec": result.integration_spec.wisdom_holman_spec,
        "encounter_control": result.integration_spec.encounter_control,
        "checkpoint_step_indices": result.checkpoint_step_indices,
        "checkpoint_epoch_policy": "EXACT_JX_INTEGER_OUTER_LATTICE",
        "outer_semantic_replay_count": result.validation_replay_count,
        "nested_public_child_replay_count": 0,
    }
    accounting = _hybrid_accounting(result)
    runtime = _jx_runtime_provenance()
    lane = Lane(
        engine_id=f"jx_hybrid_p{divisor}",
        family="JX_TRANSACTIONAL_WISDOM_HOLMAN_RKF78_HYBRID",
        method_id=HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
        divisor=divisor,
        fixed_step=float(PERIOD / divisor),
        step_count=profile.step_count(divisor),
        declared_epochs=_readonly(epochs, epochs.shape),
        observed_epochs=_readonly(epochs, epochs.shape),
        positions=_readonly(positions, positions.shape),
        velocities=_readonly(velocities, velocities.shape),
        settings=settings,
        accounting=accounting,
        runtime=runtime,
        raw_timing_seconds=elapsed,
        final_state_sha256=_final_state_sha256(positions[-1], velocities[-1]),
        trajectory_sha256=_trajectory_sha256(epochs, positions, velocities),
    )
    lane.__post_init__()
    if profile.named_full_profile and (
        lane.final_state_sha256 != JX_FINAL_STATE_SHA256[divisor]
        or lane.trajectory_sha256 != JX_TRAJECTORY_SHA256[divisor]
    ):
        raise BenchmarkError("JX named-full raw state fingerprint changed")
    return lane


def _new_rebound_simulation(rebound: Any, *, initial_dt: float) -> Any:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    rows = _initial_rows()
    for row in rows:
        simulation.add(
            m=float(row[0]),
            r=float(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            z=float(row[4]),
            vx=float(row[5]),
            vy=float(row[6]),
            vz=float(row[7]),
        )
    simulation.N_active = 3
    simulation.testparticle_type = 0
    simulation.dt = initial_dt
    readback_rows = np.array(
        [
            (
                body.m, body.r, body.x, body.y, body.z,
                body.vx, body.vy, body.vz,
            )
            for body in simulation.particles
        ],
        dtype="<f8",
    )
    if (
        int(simulation.N) != 3
        or int(simulation.N_active) != 3
        or int(simulation.testparticle_type) != 0
        or float(simulation.t).hex() != "0x0.0p+0"
        or float(simulation.dt).hex() != initial_dt.hex()
        or int(simulation.steps_done) != 0
        or float(simulation.dt_last_done).hex() != "0x0.0p+0"
        or int(simulation.is_synchronized) != 1
        or readback_rows.shape != (3, 8)
        or hashlib.sha256(readback_rows.tobytes(order="C")).hexdigest()
        != INITIAL_RAW_SHA256
    ):
        raise BenchmarkError("fresh REBOUND simulation readback is inconsistent")
    return simulation


def _particle_arrays(simulation: Any) -> tuple[np.ndarray, np.ndarray]:
    if int(simulation.N) != 3:
        raise BenchmarkError("REBOUND returned the wrong particle roster")
    positions = np.array(
        [[body.x, body.y, body.z] for body in simulation.particles],
        dtype=np.float64,
    )
    velocities = np.array(
        [[body.vx, body.vy, body.vz] for body in simulation.particles],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise BenchmarkError("REBOUND returned nonfinite Cartesian state")
    return positions, velocities


def _configuration_name(configuration: Any) -> str:
    name = configuration.name
    if type(name) is not bytes:
        raise BenchmarkError("REBOUND integrator name readback must be bytes")
    try:
        decoded = name.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BenchmarkError("REBOUND integrator name is not ASCII") from exc
    if not decoded or decoded.strip() != decoded:
        raise BenchmarkError("REBOUND integrator name is invalid")
    return decoded


def _external_fixed_every_node(
    rebound: Any, profile: Profile, method: str, divisor: int, runtime: dict[str, Any]
) -> Lane:
    """Execute every outer node; callers must retain only `_headline_lane`."""
    if type(profile) is not Profile:
        raise BenchmarkError("external fixed lane requires an exact Profile")
    profile.__post_init__()
    if type(method) is not str or method not in ("mercurius", "trace"):
        raise BenchmarkError("external fixed lane method is outside the roster")
    if type(divisor) is not int or divisor not in profile.external_divisors:
        raise BenchmarkError("external fixed lane divisor is outside the profile")
    if type(runtime) is not dict:
        raise BenchmarkError("external fixed lane runtime must be exact dict")
    step = float(PERIOD / divisor)
    count = profile.step_count(divisor)
    simulation = _new_rebound_simulation(rebound, initial_dt=step)
    requested_gravity = str(simulation.gravity)
    simulation.integrator = method
    configuration = simulation.integrator
    fresh_integrator_readback = {
        "integrator_name": _configuration_name(configuration),
        "r_crit_hill": float(configuration.r_crit_hill),
    }
    if method == "mercurius":
        fresh_integrator_readback["safe_mode"] = int(configuration.safe_mode)
        configuration.r_crit_hill = 3.0
        configuration.safe_mode = 1
        requested = {
            "integrator": "mercurius",
            "gravity": "basic",
            "r_crit_hill": 3.0,
            "safe_mode": 1,
        }
    else:
        fresh_integrator_readback.update(
            {
                "peri_crit_eta": float(configuration.peri_crit_eta),
                "peri_mode": str(configuration.peri_mode),
                "documentation_default_r_crit_hill": 4.0,
                "default_discrepancy_observed": (
                    float(configuration.r_crit_hill).hex() != float(4.0).hex()
                ),
            }
        )
        configuration.r_crit_hill = 3.0
        configuration.peri_crit_eta = 1.0
        configuration.peri_mode = "FULL_BS"
        requested = {
            "integrator": "trace",
            "gravity": "basic",
            "r_crit_hill": 3.0,
            "peri_crit_eta": 1.0,
            "peri_mode": "FULL_BS",
            "custom_S_callback": False,
            "custom_S_peri_callback": False,
        }
    initial_effective = {
        "integrator_name": _configuration_name(configuration),
        "gravity": str(simulation.gravity),
        "r_crit_hill": float(configuration.r_crit_hill),
        "G": float(simulation.G),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "dt": float(simulation.dt),
        "dt_last_done": float(simulation.dt_last_done),
        "steps_done": int(simulation.steps_done),
        "is_synchronized": int(simulation.is_synchronized),
        "initial_state_raw_sha256": INITIAL_RAW_SHA256,
    }
    if method == "mercurius":
        initial_effective["safe_mode"] = int(configuration.safe_mode)
    else:
        initial_effective.update(
            {
                "peri_crit_eta": float(configuration.peri_crit_eta),
                "peri_mode": str(configuration.peri_mode),
            }
        )
        if (
            fresh_integrator_readback["r_crit_hill"] != 3.0
            or fresh_integrator_readback["documentation_default_r_crit_hill"] != 4.0
            or fresh_integrator_readback["default_discrepancy_observed"] is not True
        ):
            raise BenchmarkError("TRACE fresh/default discrepancy witness changed")
    if (
        requested_gravity != "basic"
        or initial_effective["integrator_name"] != method
        or initial_effective["gravity"] != "basic"
        or float(initial_effective["r_crit_hill"]).hex() != "0x1.8000000000000p+1"
        or float(initial_effective["G"]).hex() != "0x1.0000000000000p+0"
        or initial_effective["N"] != 3
        or initial_effective["N_active"] != 3
        or initial_effective["testparticle_type"] != 0
        or initial_effective["collision"] != "none"
        or initial_effective["boundary"] != "none"
        or float(initial_effective["softening"]).hex() != "0x0.0p+0"
        or float(initial_effective["dt"]).hex() != step.hex()
        or float(initial_effective["dt_last_done"]).hex() != "0x0.0p+0"
        or initial_effective["steps_done"] != 0
        or initial_effective["is_synchronized"] != 1
        or (method == "mercurius" and initial_effective["safe_mode"] != 1)
        or (
            method == "trace"
            and (
                float(initial_effective["peri_crit_eta"]).hex()
                != "0x1.0000000000000p+0"
                or initial_effective["peri_mode"] != "FULL_BS"
            )
        )
    ):
        raise BenchmarkError("external fixed initial effective readback changed")
    epochs = np.empty(count + 1, dtype=np.float64)
    positions = np.empty((count + 1, 3, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    epochs[0] = float(simulation.t)
    positions[0], velocities[0] = _particle_arrays(simulation)
    started = time.perf_counter()
    for index in range(1, count + 1):
        simulation.steps(1)
        simulation.synchronize()
        epochs[index] = float(simulation.t)
        positions[index], velocities[index] = _particle_arrays(simulation)
    elapsed = float(time.perf_counter() - started)
    if int(simulation.steps_done) != count:
        raise BenchmarkError("external fixed lane steps_done is inconsistent")
    post_effective = {
        "integrator_name": _configuration_name(configuration),
        "gravity": str(simulation.gravity),
        "r_crit_hill": float(configuration.r_crit_hill),
        "G": float(simulation.G),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "dt": float(simulation.dt),
        "dt_last_done": float(simulation.dt_last_done),
        "steps_done": int(simulation.steps_done),
        "is_synchronized": int(simulation.is_synchronized),
        "initial_state_raw_sha256": INITIAL_RAW_SHA256,
    }
    if method == "mercurius":
        post_effective["safe_mode"] = int(configuration.safe_mode)
    else:
        post_effective.update(
            {
                "peri_crit_eta": float(configuration.peri_crit_eta),
                "peri_mode": str(configuration.peri_mode),
            }
        )
    if (
        post_effective["integrator_name"] != method
        or post_effective["gravity"] != "custom"
        or float(post_effective["r_crit_hill"]).hex() != "0x1.8000000000000p+1"
        or float(post_effective["G"]).hex() != "0x1.0000000000000p+0"
        or post_effective["N"] != 3
        or post_effective["N_active"] != 3
        or post_effective["testparticle_type"] != 0
        or post_effective["collision"] != "none"
        or post_effective["boundary"] != "none"
        or float(post_effective["softening"]).hex() != "0x0.0p+0"
        or float(post_effective["dt"]).hex() != step.hex()
        or float(post_effective["dt_last_done"]).hex() != step.hex()
        or post_effective["steps_done"] != count
        or post_effective["is_synchronized"] != 1
        or (method == "mercurius" and post_effective["safe_mode"] != 1)
        or (
            method == "trace"
            and (
                float(post_effective["peri_crit_eta"]).hex()
                != "0x1.0000000000000p+0"
                or post_effective["peri_mode"] != "FULL_BS"
            )
        )
    ):
        raise BenchmarkError("external fixed post-step effective readback changed")
    declared = np.array([float(index * step) for index in range(count + 1)], dtype=np.float64)
    for value in (declared, epochs, positions, velocities):
        value.setflags(write=False)
    settings = {
        "requested": requested,
        "fresh_integrator_readback_before_explicit_configuration": (
            fresh_integrator_readback
        ),
        "initial_effective_readback": initial_effective,
        "post_step_effective_readback": post_effective,
        "provenance_categories": {
            "requested": "CALLER_REQUEST",
            "fresh_integrator_readback_before_explicit_configuration": (
                "PINNED_5_1_1_EFFECTIVE_READBACK_BEFORE_EXPLICIT_CONFIGURATION"
            ),
            "initial_effective_readback": "EFFECTIVE_READBACK_BEFORE_STEPPING",
            "post_step_effective_readback": "EFFECTIVE_READBACK_AFTER_STEPPING",
        },
        "actual_clock_policy": "OBSERVED_SIMULATION_T_AFTER_EACH_STEPS_1_CALL",
    }
    accounting = {
        "steps_call_count": count,
        "steps_argument_each_call": 1,
        "synchronize_call_count": count,
        "steps_done_delta": int(simulation.steps_done),
        "internal_force_evaluations": "NOT_EXPOSED",
        "internal_switch_events": "NOT_EXPOSED",
        "internal_substeps": "NOT_EXPOSED",
        "internal_rejections": "NOT_EXPOSED",
        "encounter_N_authoritative": False,
    }
    lane = Lane(
        engine_id=f"rebound_{method}_p{divisor}",
        family=f"REBOUND_5_1_1_{method.upper()}_EXTERNAL_COMPARATOR",
        method_id=f"rebound.integrator.{method}.5.1.1",
        divisor=divisor,
        fixed_step=step,
        step_count=count,
        declared_epochs=_readonly(declared, declared.shape),
        observed_epochs=_readonly(epochs, epochs.shape),
        positions=_readonly(positions, positions.shape),
        velocities=_readonly(velocities, velocities.shape),
        settings=settings,
        accounting=accounting,
        runtime=runtime,
        raw_timing_seconds=elapsed,
        final_state_sha256=_final_state_sha256(positions[-1], velocities[-1]),
        trajectory_sha256=_trajectory_sha256(epochs, positions, velocities),
    )
    if profile.named_full_profile:
        if lane.final_state_sha256 != EXTERNAL_FINAL_STATE_SHA256[method][divisor]:
            raise BenchmarkError(f"{method} final-state fingerprint changed")
        if lane.trajectory_sha256 != EXTERNAL_TRAJECTORY_SHA256[method][divisor]:
            raise BenchmarkError(f"{method} every-node trajectory fingerprint changed")
        if (
            _array_component_sha256(
                "jx.rebound-hybrid.every-node-observed-epochs.v1",
                lane.observed_epochs,
            )
            != EXTERNAL_OBSERVED_CLOCK_VECTOR_SHA256[divisor]
        ):
            raise BenchmarkError(f"{method} complete observed clock vector changed")
        if float(lane.observed_epochs[-1]).hex() != EXTERNAL_OBSERVED_FINAL_EPOCH_HEX[divisor]:
            raise BenchmarkError(f"{method} observed final clock changed")
    return lane


def _ias15_lane(
    rebound: Any,
    *,
    engine_id: str,
    divisor: int,
    targets: np.ndarray,
    epsilon: float,
    initial_dt: float,
    runtime: dict[str, Any],
    expected_full_hashes: bool,
) -> Lane:
    if (
        type(engine_id) is not str
        or not engine_id
        or engine_id.strip() != engine_id
        or type(divisor) is not int
        or divisor not in EXTERNAL_DIVISORS
        or type(epsilon) is not float
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or type(initial_dt) is not float
        or not math.isfinite(initial_dt)
        or initial_dt <= 0.0
        or type(runtime) is not dict
        or type(expected_full_hashes) is not bool
    ):
        raise BenchmarkError("IAS lane scalar schema is invalid")
    if type(targets) is not np.ndarray or targets.dtype != np.dtype(np.float64):
        raise BenchmarkError("IAS targets must be an exact float64 ndarray")
    if not 2 <= len(targets) <= MAXIMUM_RETAINED_NODES:
        raise BenchmarkError("IAS target count exceeds its retained-node cap")
    if (
        targets.ndim != 1
        or not targets.flags.c_contiguous
        or not np.all(np.isfinite(targets))
        or float(targets[0]).hex() != "0x0.0p+0"
        or np.any(np.diff(targets) <= 0.0)
    ):
        raise BenchmarkError("IAS targets must be finite increasing from +0")
    simulation = _new_rebound_simulation(rebound, initial_dt=initial_dt)
    simulation.integrator = "ias15"
    configuration = simulation.integrator
    configuration.epsilon = epsilon
    configuration.min_dt = 0.0
    configuration.adaptive_mode = "PRS23"
    requested = {
        "integrator": "ias15",
        "epsilon": epsilon,
        "minimum_dt": 0.0,
        "adaptive_mode": "PRS23",
        "initial_dt": initial_dt,
        "exact_finish_time": 1,
        "gravity": "basic",
    }
    initial_effective = {
        "integrator_name": _configuration_name(configuration),
        "epsilon": float(configuration.epsilon),
        "minimum_dt": float(configuration.min_dt),
        "adaptive_mode": str(configuration.adaptive_mode),
        "dt": float(simulation.dt),
        "dt_last_done": float(simulation.dt_last_done),
        "steps_done": int(simulation.steps_done),
        "is_synchronized": int(simulation.is_synchronized),
        "gravity": str(simulation.gravity),
        "G": float(simulation.G),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "initial_state_raw_sha256": INITIAL_RAW_SHA256,
    }
    if (
        initial_effective["integrator_name"] != "ias15"
        or float(initial_effective["epsilon"]).hex() != epsilon.hex()
        or float(initial_effective["minimum_dt"]).hex() != "0x0.0p+0"
        or initial_effective["adaptive_mode"] != "PRS23"
        or float(initial_effective["dt"]).hex() != initial_dt.hex()
        or float(initial_effective["dt_last_done"]).hex() != "0x0.0p+0"
        or initial_effective["steps_done"] != 0
        or initial_effective["is_synchronized"] != 1
        or initial_effective["gravity"] != "basic"
    ):
        raise BenchmarkError("IAS15 initial effective readback changed")
    positions = np.empty((len(targets), 3, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    observed = np.empty(len(targets), dtype=np.float64)
    positions[0], velocities[0] = _particle_arrays(simulation)
    observed[0] = float(simulation.t)
    started = time.perf_counter()
    for index, target in enumerate(targets[1:], start=1):
        simulation.integrate(float(target), exact_finish_time=1)
        simulation.synchronize()
        observed[index] = float(simulation.t)
        positions[index], velocities[index] = _particle_arrays(simulation)
    elapsed = float(time.perf_counter() - started)
    if any(float(left).hex() != float(right).hex() for left, right in zip(observed, targets)):
        raise BenchmarkError("IAS15 did not stop at every exact requested target")
    for value in (positions, velocities, observed):
        value.setflags(write=False)
    post_effective = {
        "integrator_name": _configuration_name(configuration),
        "epsilon": float(configuration.epsilon),
        "minimum_dt": float(configuration.min_dt),
        "adaptive_mode": str(configuration.adaptive_mode),
        "dt": float(simulation.dt),
        "dt_last_done": float(simulation.dt_last_done),
        "steps_done": int(simulation.steps_done),
        "gravity": str(simulation.gravity),
        "is_synchronized": int(simulation.is_synchronized),
        "G": float(simulation.G),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "initial_state_raw_sha256": INITIAL_RAW_SHA256,
    }
    if (
        post_effective["integrator_name"] != "ias15"
        or float(post_effective["epsilon"]).hex() != epsilon.hex()
        or float(post_effective["minimum_dt"]).hex() != "0x0.0p+0"
        or post_effective["adaptive_mode"] != "PRS23"
        or post_effective["gravity"] != "basic"
        or type(post_effective["steps_done"]) is not int
        or post_effective["steps_done"] <= 0
        or post_effective["is_synchronized"] != 1
        or not math.isfinite(float(post_effective["dt"]))
        or not math.isfinite(float(post_effective["dt_last_done"]))
    ):
        raise BenchmarkError("IAS15 post-run effective readback changed")
    settings = {
        "requested": requested,
        "initial_effective_readback": initial_effective,
        "post_run_effective_readback": post_effective,
        "provenance_categories": {
            "requested": "CALLER_REQUEST",
            "initial_effective_readback": "EFFECTIVE_READBACK_BEFORE_INTEGRATION",
            "post_run_effective_readback": "EFFECTIVE_READBACK_AFTER_INTEGRATION",
            "target_epochs": "EXACT_CALL_ARGUMENTS",
        },
        "target_basis": "EXACT_CALLER_TARGET_VECTOR",
        "target_count": len(targets),
    }
    accounting = {
        "integrate_call_count": len(targets) - 1,
        "synchronize_call_count": len(targets) - 1,
        "steps_done": int(simulation.steps_done),
        "iterations_max_exceeded": int(configuration.iterations_max_exceeded),
        "internal_force_evaluations": "NOT_EXPOSED",
    }
    lane = Lane(
        engine_id=engine_id,
        family="REBOUND_5_1_1_IAS15_NUMERICAL_REFERENCE",
        method_id="rebound.integrator.ias15.5.1.1",
        divisor=divisor,
        fixed_step=float(PERIOD / divisor),
        step_count=len(targets) - 1,
        declared_epochs=_readonly(targets, targets.shape),
        observed_epochs=_readonly(observed, observed.shape),
        positions=_readonly(positions, positions.shape),
        velocities=_readonly(velocities, velocities.shape),
        settings=settings,
        accounting=accounting,
        runtime=runtime,
        raw_timing_seconds=elapsed,
        final_state_sha256=_final_state_sha256(positions[-1], velocities[-1]),
        trajectory_sha256=_trajectory_sha256(observed, positions, velocities),
    )
    if expected_full_hashes:
        if lane.final_state_sha256 != EXTERNAL_IAS15_FINAL_STATE_SHA256[divisor]:
            raise BenchmarkError("external-clock IAS15 final-state fingerprint changed")
        if lane.trajectory_sha256 != EXTERNAL_IAS15_TRAJECTORY_SHA256[divisor]:
            raise BenchmarkError("external-clock IAS15 trajectory fingerprint changed")
    return lane


def _headline_indices(profile: Profile, divisor: int) -> tuple[int, ...]:
    count = profile.step_count(divisor)
    stride = divisor // 16
    values = tuple(range(0, count + 1, stride))
    if values[-1] != count:
        values += (count,)
    if profile.named_full_profile and len(values) != 103:
        raise BenchmarkError("full headline roster must contain 103 nodes")
    return values


def _headline_lane(
    every_node: Lane,
    profile: Profile,
    *,
    all_node_accuracy: dict[str, Any] | None,
) -> Lane:
    indices = _headline_indices(profile, every_node.divisor)
    if len(every_node.observed_epochs) != every_node.step_count + 1:
        raise BenchmarkError("headline source must be a transient every-node lane")
    declared = np.array(every_node.declared_epochs[list(indices)], dtype=np.float64)
    observed = np.array(every_node.observed_epochs[list(indices)], dtype=np.float64)
    positions = np.array(every_node.positions[list(indices)], dtype=np.float64)
    velocities = np.array(every_node.velocities[list(indices)], dtype=np.float64)
    accounting = _thaw_tree(every_node.accounting)
    accounting.update(
        {
            "headline_checkpoint_step_indices": indices,
            "headline_retained_node_count": len(indices),
            "every_node_count": every_node.step_count + 1,
            "every_node_final_state_raw_sha256": every_node.final_state_sha256,
            "every_node_trajectory_raw_sha256": every_node.trajectory_sha256,
            "every_node_declared_epochs_content_sha256": _array_component_sha256(
                "jx.rebound-hybrid.every-node-declared-epochs.v1",
                every_node.declared_epochs,
            ),
            "every_node_observed_epochs_content_sha256": _array_component_sha256(
                "jx.rebound-hybrid.every-node-observed-epochs.v1",
                every_node.observed_epochs,
            ),
            "every_node_invariants": _invariant_metrics(
                every_node.positions, every_node.velocities
            ),
            "every_node_accuracy_against_actual_clock_ias15": all_node_accuracy,
            "every_node_states_retained_in_report": False,
        }
    )
    lane = Lane(
        engine_id=every_node.engine_id,
        family=every_node.family,
        method_id=every_node.method_id,
        divisor=every_node.divisor,
        fixed_step=every_node.fixed_step,
        step_count=every_node.step_count,
        declared_epochs=_readonly(declared, declared.shape),
        observed_epochs=_readonly(observed, observed.shape),
        positions=_readonly(positions, positions.shape),
        velocities=_readonly(velocities, velocities.shape),
        settings=_thaw_tree(every_node.settings),
        accounting=accounting,
        runtime=_thaw_tree(every_node.runtime),
        raw_timing_seconds=every_node.raw_timing_seconds,
        final_state_sha256=_final_state_sha256(positions[-1], velocities[-1]),
        trajectory_sha256=_trajectory_sha256(observed, positions, velocities),
    )
    return lane


def _binary_state_plan() -> tuple[StateSnapshot, ForcePlan]:
    gm = np.array((1.0, 0.001), dtype=np.float64)
    total = float(gm[0] + gm[1])
    speed = math.sqrt(total)
    provenance = Provenance(
        "jx.benchmark.hybrid.all-far-control.v1",
        "benchmarks/rebound_hybrid_comparison.py",
        "1",
        _PROVENANCE_SUPPORT._file_identity(__file__)["sha256"],
    )
    state = StateSnapshot(
        "jx.benchmark.hybrid.all-far-control.snapshot.v1",
        0.0,
        "SYNTHETIC",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "CARTESIAN_RIGHT_HANDED",
        "L",
        "T",
        "M",
        "jx.benchmark.hybrid.synthetic.v1",
        ("STAR", "PLANET"),
        np.array(((-gm[1] / total, 0.0, 0.0), (gm[0] / total, 0.0, 0.0))),
        np.array(
            (
                (0.0, -gm[1] * speed / total, 0.0),
                (0.0, gm[0] * speed / total, 0.0),
            )
        ),
        gm.copy(),
        gm.copy(),
        np.zeros(2, dtype=np.float64),
        np.ones(2, dtype=np.bool_),
        provenance,
    )
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        -10.0,
        10.0,
    )
    plan = ForcePlan(
        "jx.benchmark.hybrid.all-far-control.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                state.body_ids,
                state.body_ids,
                state.unit_system_id,
                (metadata,),
            ),
        ),
    )
    return state, plan


def _binary_hybrid_spec(
    state: StateSnapshot,
    *,
    signed_step: float,
    checkpoints: tuple[int, ...],
    wh_floor: float = 0.01,
    child_floor: float = 0.01,
) -> HybridWisdomHolmanRKF78Spec:
    wh = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=checkpoints,
        fixed_step=signed_step,
        maximum_steps=3,
        jacobi_body_order=state.body_ids,
        minimum_encounter_pair_separation=wh_floor,
        minimum_jacobi_periapse=0.05,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )
    magnitude = abs(signed_step)
    encounter = HybridEncounterControlProfile(
        initial_step_magnitude=magnitude,
        minimum_step_magnitude=0.001,
        maximum_step_magnitude=magnitude,
        pair_certification_floors=(child_floor,),
        pair_position_atols=(1.0e-6,),
        pair_position_rtol=1.0e-12,
        pair_velocity_atols=(1.0e-6,),
        pair_velocity_rtol=1.0e-12,
        gm_centroid_position_atol=1.0e-6,
        gm_centroid_position_rtol=1.0e-12,
        gm_centroid_velocity_atol=1.0e-6,
        gm_centroid_velocity_rtol=1.0e-12,
        maximum_substep_proposals=128,
        maximum_accepted_substeps=128,
        maximum_rejected_substeps=64,
        maximum_consecutive_rejections=32,
        maximum_force_evaluations=1664,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=5.0,
        exact_rational_resources=_resources(),
    )
    return HybridWisdomHolmanRKF78Spec(wh, encounter)


def _all_far_control() -> dict[str, Any]:
    state, plan = _binary_state_plan()
    cases: list[dict[str, Any]] = []
    for signed_step in (0.125, -0.125):
        for checkpoints in ((0, 3), (0, 1, 2, 3)):
            spec = _binary_hybrid_spec(
                state, signed_step=signed_step, checkpoints=checkpoints
            )
            hybrid = integrate_hybrid_wisdom_holman_rkf78_trajectory(
                state, plan, spec
            )
            wh = integrate_wisdom_holman_trajectory(
                state, plan, spec.wisdom_holman_spec
            )
            if not hybrid.all_far or any(
                record.mode != "WISDOM_HOLMAN_FAR"
                for record in hybrid.outer_step_records
            ):
                raise BenchmarkError("all-far control unexpectedly switched")
            if (
                hybrid.wisdom_holman_schedule_content_sha256
                != wh.schedule_content_sha256
                or hybrid.wisdom_holman_result_content_sha256
                != wh.result_content_sha256
            ):
                raise BenchmarkError("all-far WH checksum projection changed")
            if any(
                left.positions.tobytes(order="C")
                != right.positions.tobytes(order="C")
                or left.velocities.tobytes(order="C")
                != right.velocities.tobytes(order="C")
                for left, right in zip(hybrid.checkpoints, wh.checkpoints)
            ):
                raise BenchmarkError("all-far WH state projection changed")
            hybrid_epochs = np.array(hybrid.checkpoint_epochs, dtype=np.float64)
            wh_epochs = np.array(wh.checkpoint_epochs, dtype=np.float64)
            if hybrid_epochs.tobytes(order="C") != wh_epochs.tobytes(order="C"):
                raise BenchmarkError("all-far WH checkpoint epoch bytes changed")
            diagnostics_equal = all(
                _canonical_json(
                    getattr(hybrid.wisdom_holman_diagnostics, descriptor.name)
                )
                == _canonical_json(getattr(wh, descriptor.name))
                for descriptor in fields(hybrid.wisdom_holman_diagnostics)
            )
            if not diagnostics_equal:
                raise BenchmarkError("all-far WH diagnostic projection changed")
            work = hybrid.primary_counts.accepted_wh_work
            work_vector = {
                name: getattr(work, name)
                for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
            }
            expected_work_vector = {
                "force_calls_entered": wh.primary_map_force_evaluations,
                "force_evaluations_completed": wh.primary_map_force_evaluations,
                "interaction_force_assemblies": (
                    wh.primary_map_interaction_force_assemblies
                ),
                "kepler_subflow_calls_entered": wh.primary_map_kepler_subflow_solves,
                "kepler_subflow_solves_completed": wh.primary_map_kepler_subflow_solves,
                "universal_solver_iterations": (
                    wh.primary_map_universal_solver_iterations
                ),
                "universal_solver_bracket_expansions": (
                    wh.primary_map_universal_solver_bracket_expansions
                ),
                "universal_g_bundle_calls_entered": (
                    wh.primary_map_universal_g_function_evaluations
                ),
                "universal_g_bundle_calls_completed": (
                    wh.primary_map_universal_g_function_evaluations
                ),
                "universal_series_terms_evaluated": (
                    wh.primary_map_universal_series_terms_evaluated
                ),
                "cartesian_to_jacobi_calls_entered": (
                    wh.primary_map_coordinate_forward_transforms
                ),
                "cartesian_to_jacobi_transforms_completed": (
                    wh.primary_map_coordinate_forward_transforms
                ),
                "jacobi_to_cartesian_calls_entered": (
                    wh.primary_map_coordinate_inverse_transforms
                ),
                "jacobi_to_cartesian_transforms_completed": (
                    wh.primary_map_coordinate_inverse_transforms
                ),
                "node_guard_evaluations": wh.primary_map_node_guard_evaluations,
                "path_guard_evaluations": wh.primary_map_path_guard_evaluations,
                "first_half_kicks_completed": wh.completed_steps,
                "center_of_mass_drifts_completed": wh.completed_steps,
                "second_half_kicks_completed": wh.completed_steps,
            }
            if work_vector != expected_work_vector or any(
                getattr(hybrid.primary_counts.discarded_wh_work, name) != 0
                for name in HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
            ):
                raise BenchmarkError("all-far nineteen-counter projection changed")
            cases.append(
                {
                    "signed_step": signed_step,
                    "checkpoint_step_indices": checkpoints,
                    "checkpoint_state_bytes_equal": True,
                    "checkpoint_epoch_bytes_equal": True,
                    "diagnostics_equal": True,
                    "schedule_content_sha256": wh.schedule_content_sha256,
                    "result_content_sha256": wh.result_content_sha256,
                    "nineteen_counter_projection": work_vector,
                    "public_wh_accounting_projection_equal": True,
                }
            )
    return {
        "classification": "SEPARATE_UNEQUAL_MASS_BINARY_ALL_FAR_CONTROL",
        "case_count": len(cases),
        "cases": tuple(cases),
        "scope": "THREE_STEP_FORWARD_BACKWARD_SPARSE_DENSE_PROJECTION_CUSTODY_ONLY",
        "analytic_accuracy_claimed": False,
        "broad_finite_step_equivalence_claimed": False,
    }


def _initial_floor_fatal_control() -> dict[str, Any]:
    state, plan = _binary_state_plan()
    spec = _binary_hybrid_spec(
        state,
        signed_step=0.125,
        checkpoints=(0, 3),
        wh_floor=1.0,
        child_floor=math.nextafter(1.0, math.inf),
    )
    try:
        integrate_hybrid_wisdom_holman_rkf78_trajectory(state, plan, spec)
    except (HybridDomainError, EncounterDomainError) as exc:
        return {
            "classification": "ORIGINAL_NODE_ENCOUNTER_FLOOR_FATAL_CONTROL",
            "exception_type": type(exc).__name__,
            "partial_result_released": False,
            "message_not_used_for_classification": True,
        }
    raise BenchmarkError("initial encounter-floor equality did not fail fatally")


TUNNEL_ROWS_HEX = (
    ("0x1p+0", "0x0p+0", "0x0p+0", "0x0p+0", "0x0p+0", "0x0p+0", "0x0p+0", "0x0p+0"),
    ("0x0p+0", "0x1.999999999999ap-5", "0x1.9000000000000p+6", "-0x1p+0", "0x0p+0", "0x0p+0", "0x1p+2", "0x0p+0"),
    ("0x0p+0", "0x1.999999999999ap-5", "0x1.9000000000000p+6", "0x1p+0", "0x0p+0", "0x0p+0", "-0x1p+2", "0x0p+0"),
)
TUNNEL_RAW_SHA256 = "24076364a0f8a74f2362f340b27263164c93f2f8b947f22f44212fe00215bc94"
CENTRAL_PERIAPSE_ROWS_HEX = (
    (
        "0x1.0000000000000p+0", "0x0.0p+0", "0x1.04929cf1f45c0p-9",
        "-0x1.1f6da5f9758dep-62", "0x0.0p+0", "0x1.0019352d4c738p-60",
        "0x1.292d82b3cee7bp-14", "0x0.0p+0",
    ),
    (
        "0x1.0624dd2f1a9fcp-10", "0x0.0p+0", "-0x1.fcee5a8891439p+0",
        "0x1.18b114159ccc8p-52", "0x0.0p+0", "-0x1.f4313bdc79519p-51",
        "-0x1.223671a3980e3p-4", "0x0.0p+0",
    ),
)
CENTRAL_PERIAPSE_RAW_SHA256 = "d246a85d2c2c78409c73cf29337e9e31639aca8410650f342b5901c4f19a0c3f"
CENTRAL_PERIAPSE_FINAL_STATE_SHA256 = {
    "mercurius": "4ec290025f0cbdc4c62ca9563f0d427d3bf20424344edf9b842357d98b2d4308",
    "trace": "624bdf8c7e885866d82f7987a4cc526f29bc76affd844c6d247432d8cb1cec09",
    "ias15": "a1258fe7578115a7697fcc07dbd3aacfe5d38dc0c08f55b4f3a3b0ed2393ffc0",
}


def _locked_rows(rows_hex: tuple[tuple[str, ...], ...], digest: str) -> np.ndarray:
    if type(rows_hex) is not tuple or not 1 <= len(rows_hex) <= 3:
        raise BenchmarkError("locked limitation rows have an invalid roster")
    rows = np.array(
        [[float.fromhex(value) for value in row] for row in rows_hex], dtype="<f8"
    )
    if (
        rows.shape != (len(rows_hex), 8)
        or hashlib.sha256(rows.tobytes(order="C")).hexdigest() != digest
    ):
        raise BenchmarkError("locked limitation Cartesian bytes changed")
    return rows


def _limitation_final_state_sha256(
    rows: np.ndarray, positions: np.ndarray, velocities: np.ndarray
) -> str:
    count = len(rows)
    if (
        type(rows) is not np.ndarray
        or rows.dtype != np.dtype(np.float64)
        or rows.shape != (count, 8)
        or type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or positions.shape != (count, 3)
        or velocities.shape != (count, 3)
        or not np.all(np.isfinite(rows))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
    ):
        raise BenchmarkError("limitation final state has an invalid exact schema")
    payload = np.concatenate((rows[:, :2], positions, velocities), axis=1).astype(
        "<f8", copy=False
    )
    return hashlib.sha256(payload.tobytes(order="C")).hexdigest()


def _new_rebound_from_rows(
    rebound: Any,
    rows: np.ndarray,
    *,
    active_count: int,
    fixed_step: float,
) -> Any:
    if type(rows) is not np.ndarray or rows.dtype != np.dtype(np.float64):
        raise BenchmarkError("limitation rows must be an exact float64 ndarray")
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    for row in rows:
        simulation.add(
            m=float(row[0]), r=float(row[1]), x=float(row[2]), y=float(row[3]),
            z=float(row[4]), vx=float(row[5]), vy=float(row[6]), vz=float(row[7]),
        )
    simulation.N_active = active_count
    simulation.testparticle_type = 0
    simulation.dt = fixed_step
    return simulation


def _configure_external_integrator(simulation: Any, method: str) -> None:
    simulation.integrator = method
    configuration = simulation.integrator
    configuration.r_crit_hill = 3.0
    if method == "mercurius":
        configuration.safe_mode = 1
    elif method == "trace":
        configuration.peri_crit_eta = 1.0
        configuration.peri_mode = "FULL_BS"
    else:
        raise BenchmarkError("limitation integrator is outside the exact roster")


def _tunneling_limitation_probe(rebound: Any) -> dict[str, Any]:
    rows = _locked_rows(TUNNEL_ROWS_HEX, TUNNEL_RAW_SHA256)
    fixed_step = float.fromhex("0x1p-1")
    exit_min_distance = float.fromhex("0x1.999999999999ap-4")
    initial_relative_y = float(rows[2, 3] - rows[1, 3])
    outcomes: dict[str, Any] = {}
    for method in ("mercurius", "trace"):
        simulation = _new_rebound_from_rows(
            rebound, rows, active_count=1, fixed_step=fixed_step
        )
        _configure_external_integrator(simulation, method)
        simulation.exit_min_distance = exit_min_distance
        configuration = simulation.integrator

        def readback() -> dict[str, Any]:
            result = {
                "integrator_name": _configuration_name(configuration),
                "r_crit_hill": float(configuration.r_crit_hill),
                "fixed_step": float(simulation.dt),
                "exit_min_distance": float(simulation.exit_min_distance),
                "gravity": str(simulation.gravity),
                "collision": str(simulation.collision),
                "boundary": str(simulation.boundary),
                "N": int(simulation.N),
                "N_active": int(simulation.N_active),
                "testparticle_type": int(simulation.testparticle_type),
                "steps_done": int(simulation.steps_done),
                "dt_last_done": float(simulation.dt_last_done),
                "is_synchronized": int(simulation.is_synchronized),
            }
            if method == "mercurius":
                result["safe_mode"] = int(configuration.safe_mode)
            else:
                result["peri_crit_eta"] = float(configuration.peri_crit_eta)
                result["peri_mode"] = str(configuration.peri_mode)
            return result

        initial_effective = readback()
        raised = False
        try:
            simulation.steps(1)
            simulation.synchronize()
        except Exception as exc:  # Exact observation only; no message classification.
            raised = True
            exception_type = type(exc).__name__
        first = simulation.particles[1]
        second = simulation.particles[2]
        post_effective = readback()
        endpoint_relative_y = float(second.y) - float(first.y)
        separation = math.sqrt(
            (float(second.x) - float(first.x)) ** 2
            + (float(second.y) - float(first.y)) ** 2
            + (float(second.z) - float(first.z)) ** 2
        )
        outcomes[method] = {
            "exception_raised": raised,
            "exception_type": exception_type if raised else None,
            "endpoint_separation": separation,
            "endpoint_clock": float(simulation.t),
            "requested": {
                "integrator": method,
                "fixed_step": fixed_step,
                "exit_min_distance": exit_min_distance,
                "r_crit_hill": 3.0,
                "safe_mode": 1 if method == "mercurius" else None,
                "peri_crit_eta": 1.0 if method == "trace" else None,
                "peri_mode": "FULL_BS" if method == "trace" else None,
            },
            "initial_effective_readback": initial_effective,
            "post_step_effective_readback": post_effective,
            "signed_relative_y_bracket": {
                "initial_epoch": 0.0,
                "initial_relative_y": initial_relative_y,
                "endpoint_epoch": float(simulation.t),
                "endpoint_relative_y": endpoint_relative_y,
                "strict_sign_change": initial_relative_y * endpoint_relative_y < 0.0,
            },
        }
    line = _new_rebound_from_rows(
        rebound, rows, active_count=1, fixed_step=float.fromhex("0x1p-1")
    )
    line.integrator = "leapfrog"
    line.collision = "line"
    line_collision_type = None
    try:
        line.steps(1)
    except Exception as exc:  # Observation is exception type, never message text.
        line_collision_type = type(exc).__name__
    if any(
        outcomes[method]["exception_raised"] is not False
        or float(outcomes[method]["endpoint_clock"]).hex() != "0x1.0000000000000p-1"
        or float(outcomes[method]["endpoint_separation"]).hex()
        != "0x1.000000b2f0dbcp+1"
        or float(
            outcomes[method]["signed_relative_y_bracket"]["initial_relative_y"]
        ).hex()
        != "0x1.0000000000000p+1"
        or float(
            outcomes[method]["signed_relative_y_bracket"]["endpoint_relative_y"]
        ).hex()
        != "-0x1.000000b2f0dbcp+1"
        or outcomes[method]["signed_relative_y_bracket"]["strict_sign_change"]
        is not True
        or outcomes[method]["initial_effective_readback"]["integrator_name"]
        != method
        or float(
            outcomes[method]["initial_effective_readback"]["fixed_step"]
        ).hex()
        != "0x1.0000000000000p-1"
        or float(
            outcomes[method]["initial_effective_readback"]["exit_min_distance"]
        ).hex()
        != "0x1.999999999999ap-4"
        or outcomes[method]["initial_effective_readback"]["steps_done"] != 0
        or float(
            outcomes[method]["initial_effective_readback"]["dt_last_done"]
        ).hex()
        != "0x0.0p+0"
        or outcomes[method]["post_step_effective_readback"]["integrator_name"]
        != method
        or float(outcomes[method]["post_step_effective_readback"]["fixed_step"]).hex()
        != "0x1.0000000000000p-1"
        or float(
            outcomes[method]["post_step_effective_readback"]["exit_min_distance"]
        ).hex()
        != "0x1.999999999999ap-4"
        or outcomes[method]["post_step_effective_readback"]["steps_done"] != 1
        or float(
            outcomes[method]["post_step_effective_readback"]["dt_last_done"]
        ).hex()
        != "0x1.0000000000000p-1"
        for method in ("mercurius", "trace")
    ) or line_collision_type != "Collision":
        raise BenchmarkError("endpoint-tunneling limitation witness changed")
    return {
        "classification": "ENDPOINT_TUNNELING_LIMITATION_WITNESS",
        "fixture_raw_sha256": TUNNEL_RAW_SHA256,
        "configured_fixed_step": fixed_step,
        "configured_exit_min_distance": exit_min_distance,
        "straight_line_relative_motion_crossing_epoch": float.fromhex("0x1p-2"),
        "straight_line_epoch_is_exact_newtonian_root": False,
        "continuous_crossing_basis": (
            "LOCKED_SYMMETRIC_NEWTONIAN_IVP_WITH_OPPOSITE_INITIAL_Y_AND_VY"
        ),
        "event_or_root_location_claimed": False,
        "configured_exit_min_distance_outcomes": outcomes,
        "line_collision_exception_type": line_collision_type,
        "collision_response_claimed": False,
        "common_success_gate": False,
    }


def _central_periapse_limitation_probe(rebound: Any) -> dict[str, Any]:
    rows = _locked_rows(CENTRAL_PERIAPSE_ROWS_HEX, CENTRAL_PERIAPSE_RAW_SHA256)
    step = float(PERIOD / 64.0)
    states: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for method in ("mercurius", "trace"):
        simulation = _new_rebound_from_rows(
            rebound, rows, active_count=2, fixed_step=step
        )
        _configure_external_integrator(simulation, method)
        for _ in range(64):
            simulation.steps(1)
        simulation.synchronize()
        position, velocity = _particle_arrays_n(simulation, 2)
        states[method] = (position, velocity, float(simulation.t))
    reference_simulation = _new_rebound_from_rows(
        rebound, rows, active_count=2, fixed_step=step
    )
    reference_simulation.integrator = "ias15"
    reference_configuration = reference_simulation.integrator
    reference_configuration.epsilon = 1.0e-12
    reference_configuration.min_dt = 0.0
    reference_configuration.adaptive_mode = "PRS23"
    target = states["mercurius"][2]
    if (
        target.hex() != "0x1.921fb54442d13p+2"
        or target.hex() != states["trace"][2].hex()
    ):
        raise BenchmarkError("central-periapse external clocks differ")
    reference_requested = {
        "integrator": "ias15",
        "epsilon": 1.0e-12,
        "minimum_dt": 0.0,
        "adaptive_mode": "PRS23",
        "initial_dt": step,
        "target_epoch": target,
        "exact_finish_time": 1,
    }
    reference_initial_effective = {
        "integrator_name": _configuration_name(reference_configuration),
        "epsilon": float(reference_configuration.epsilon),
        "minimum_dt": float(reference_configuration.min_dt),
        "adaptive_mode": str(reference_configuration.adaptive_mode),
        "dt": float(reference_simulation.dt),
        "dt_last_done": float(reference_simulation.dt_last_done),
        "steps_done": int(reference_simulation.steps_done),
        "iterations_max_exceeded": int(
            reference_configuration.iterations_max_exceeded
        ),
        "is_synchronized": int(reference_simulation.is_synchronized),
        "gravity": str(reference_simulation.gravity),
    }
    if (
        reference_initial_effective["integrator_name"] != "ias15"
        or float(reference_initial_effective["epsilon"]).hex()
        != "0x1.19799812dea11p-40"
        or float(reference_initial_effective["minimum_dt"]).hex() != "0x0.0p+0"
        or reference_initial_effective["adaptive_mode"] != "PRS23"
        or float(reference_initial_effective["dt"]).hex()
        != "0x1.921fb54442d18p-4"
        or float(reference_initial_effective["dt_last_done"]).hex() != "0x0.0p+0"
        or reference_initial_effective["steps_done"] != 0
        or reference_initial_effective["iterations_max_exceeded"] != 0
        or reference_initial_effective["is_synchronized"] != 1
        or reference_initial_effective["gravity"] != "basic"
    ):
        raise BenchmarkError("central-periapse IAS15 initial readback changed")
    reference_simulation.integrate(target, exact_finish_time=1)
    reference_simulation.synchronize()
    reference_observed_epoch = float(reference_simulation.t)
    reference_post_effective = {
        "integrator_name": _configuration_name(reference_configuration),
        "epsilon": float(reference_configuration.epsilon),
        "minimum_dt": float(reference_configuration.min_dt),
        "adaptive_mode": str(reference_configuration.adaptive_mode),
        "dt": float(reference_simulation.dt),
        "dt_last_done": float(reference_simulation.dt_last_done),
        "steps_done": int(reference_simulation.steps_done),
        "iterations_max_exceeded": int(
            reference_configuration.iterations_max_exceeded
        ),
        "is_synchronized": int(reference_simulation.is_synchronized),
        "gravity": str(reference_simulation.gravity),
    }
    if (
        reference_observed_epoch.hex() != target.hex()
        or reference_post_effective["integrator_name"] != "ias15"
        or float(reference_post_effective["epsilon"]).hex()
        != "0x1.19799812dea11p-40"
        or float(reference_post_effective["minimum_dt"]).hex() != "0x0.0p+0"
        or reference_post_effective["adaptive_mode"] != "PRS23"
        or float(reference_post_effective["dt"]).hex()
        != "0x1.72388bb63622ep-3"
        or float(reference_post_effective["dt_last_done"]).hex()
        != "0x1.0309e58782600p-5"
        or reference_post_effective["steps_done"] != 429
        or reference_post_effective["iterations_max_exceeded"] != 0
        or reference_post_effective["is_synchronized"] != 1
        or reference_post_effective["gravity"] != "basic"
    ):
        raise BenchmarkError("central-periapse IAS15 post-run readback changed")
    reference_position, reference_velocity = _particle_arrays_n(
        reference_simulation, 2
    )
    final_state_sha256 = {
        method: _limitation_final_state_sha256(
            rows, states[method][0], states[method][1]
        )
        for method in ("mercurius", "trace")
    }
    final_state_sha256["ias15"] = _limitation_final_state_sha256(
        rows, reference_position, reference_velocity
    )
    if final_state_sha256 != CENTRAL_PERIAPSE_FINAL_STATE_SHA256:
        raise BenchmarkError("central-periapse final-state fingerprint changed")
    metrics: dict[str, Any] = {}
    masses = rows[:, 0]
    initial_energy = _two_body_energy(rows[:, 2:5], rows[:, 5:8], masses)
    for method in ("mercurius", "trace"):
        position, velocity, _ = states[method]
        metrics[method] = {
            "position_vector_l2_max": float(
                np.max(np.linalg.norm(position - reference_position, axis=1))
            ),
            "velocity_vector_l2_max": float(
                np.max(np.linalg.norm(velocity - reference_velocity, axis=1))
            ),
            "relative_total_energy_final": abs(
                (_two_body_energy(position, velocity, masses) - initial_energy)
                / initial_energy
            ),
        }
    expected_hex = {
        "mercurius": {
            "position_vector_l2_max": "0x1.30613300e95e9p+2",
            "velocity_vector_l2_max": "0x1.f2afc0b49479bp+0",
            "relative_total_energy_final": "0x1.1ad8af0f489fbp+2",
        },
        "trace": {
            "position_vector_l2_max": "0x1.c0df22dd50bf1p-17",
            "velocity_vector_l2_max": "0x1.258d20b561134p-18",
            "relative_total_energy_final": "0x1.d90e0391f0005p-19",
        },
    }
    if any(
        float(metrics[method][name]).hex() != expected
        for method in ("mercurius", "trace")
        for name, expected in expected_hex[method].items()
    ):
        raise BenchmarkError("central-periapse limitation witness changed")
    return {
        "classification": "CENTRAL_PERIAPSE_METHOD_SCOPE_LIMITATION",
        "fixture_raw_sha256": CENTRAL_PERIAPSE_RAW_SHA256,
        "fixed_step": step,
        "step_count": 64,
        "final_state_raw_sha256_payload": (
            "EXACT_2_BY_8_ROW_MAJOR_LITTLE_ENDIAN_FLOAT64_M_R_X_Y_Z_VX_VY_VZ"
        ),
        "final_state_raw_sha256": final_state_sha256,
        "ias15_reference_protocol": {
            "target_epoch": target,
            "observed_epoch": reference_observed_epoch,
            "target_equals_observed_bit_exact": (
                target.hex() == reference_observed_epoch.hex()
            ),
            "requested": reference_requested,
            "initial_effective_readback": reference_initial_effective,
            "post_run_effective_readback": reference_post_effective,
            "accounting": {
                "integrate_call_count": 1,
                "synchronize_call_count": 1,
                "steps_done": reference_post_effective["steps_done"],
                "iterations_max_exceeded": reference_post_effective[
                    "iterations_max_exceeded"
                ],
            },
        },
        "metrics_against_actual_clock_ias15": metrics,
        "mercurius_common_success_gate": False,
        "trace_forward_only": True,
    }


def _particle_arrays_n(simulation: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    if type(count) is not int or int(simulation.N) != count:
        raise BenchmarkError("limitation simulation returned the wrong body roster")
    positions = np.array(
        [[body.x, body.y, body.z] for body in simulation.particles], dtype=np.float64
    )
    velocities = np.array(
        [[body.vx, body.vy, body.vz] for body in simulation.particles], dtype=np.float64
    )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise BenchmarkError("limitation simulation returned nonfinite state")
    return positions, velocities


def _two_body_energy(positions: Any, velocities: Any, masses: Any) -> float:
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    kinetic = sum(
        0.5 * float(mass[index]) * float(np.dot(vel[index], vel[index]))
        for index in range(2)
    )
    separation = float(np.linalg.norm(pos[1] - pos[0]))
    if separation <= 0.0:
        raise BenchmarkError("two-body limitation reached nonpositive separation")
    return kinetic - float(mass[0] * mass[1]) / separation


def _limitation_probes(
    profile: Profile, rebound: Any | None, rebound_mode: str
) -> dict[str, Any]:
    if type(rebound_mode) is not str or rebound_mode not in REBOUND_MODES:
        raise BenchmarkError("limitation probes require an exact REBOUND mode")
    result: dict[str, Any] = {
        "requested": profile.include_limitations,
        "initial_committed_node_floor_fatal": _initial_floor_fatal_control(),
        "all_active_main_lane_collisions_enabled": False,
        "trace_protocol_direction": "FORWARD_ONLY",
        "global_clearance_claimed": False,
        "event_location_claimed": False,
    }
    if not profile.include_limitations:
        result.update(
            {
                "external_limitation_lanes_executed": False,
                "reason": "NOT_REQUESTED",
            }
        )
    elif rebound is None:
        result.update(
            {
                "external_limitation_lanes_executed": False,
                "reason": (
                    "EXPLICITLY_DISABLED"
                    if rebound_mode == "disabled"
                    else "EXACT_REBOUND_5_1_1_UNAVAILABLE"
                ),
            }
        )
    else:
        result.update(
            {
                "external_limitation_lanes_executed": True,
                "endpoint_tunneling": _tunneling_limitation_probe(rebound),
                "central_periapse": _central_periapse_limitation_probe(rebound),
            }
        )
    return result


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (BenchmarkError, TypeError, ValueError):
        return False


def _exact_epoch_bytes(values: Iterable[float]) -> bytes:
    array = np.array(tuple(values), dtype=np.float64)
    return array.tobytes(order="C")


def _validate_lane_epoch_roster(
    lane: Lane,
    profile: Profile,
    *,
    exact_jx_lattice: bool,
) -> None:
    indices = _headline_indices(profile, lane.divisor)
    if len(lane.declared_epochs) != len(indices):
        raise BenchmarkError("lane retained-node count differs from headline roster")
    expected = tuple(float(index * (PERIOD / lane.divisor)) for index in indices)
    if lane.declared_epochs.tobytes(order="C") != _exact_epoch_bytes(expected):
        raise BenchmarkError("lane declared epochs differ from headline lattice")
    if exact_jx_lattice and (
        lane.observed_epochs.tobytes(order="C")
        != lane.declared_epochs.tobytes(order="C")
    ):
        raise BenchmarkError("JX lane observed epochs differ from exact labels")


def _validate_lane_mapping_roster(
    lane: Lane,
    *,
    settings: tuple[str, ...],
    accounting_required: tuple[str, ...],
) -> None:
    if tuple(lane.settings.keys()) != tuple(sorted(settings)):
        raise BenchmarkError(f"{lane.engine_id} settings field roster changed")
    if any(key not in lane.accounting for key in accounting_required):
        raise BenchmarkError(f"{lane.engine_id} accounting field roster is incomplete")


def _validate_ias_lane_schema(
    lane: Lane,
    *,
    epsilon: float,
    initial_dt: float,
    target_count: int,
    integrate_call_count: int,
) -> None:
    settings = _plain_tree(lane.settings)
    accounting = _plain_tree(lane.accounting)
    requested = settings.get("requested")
    initial = settings.get("initial_effective_readback")
    post = settings.get("post_run_effective_readback")
    if not all(type(value) is dict for value in (requested, initial, post)):
        raise BenchmarkError("IAS settings components lost exact mapping schema")
    assert type(requested) is dict and type(initial) is dict and type(post) is dict
    if (
        requested.get("integrator") != "ias15"
        or type(requested.get("epsilon")) is not float
        or requested["epsilon"].hex() != epsilon.hex()
        or type(requested.get("minimum_dt")) is not float
        or requested["minimum_dt"].hex() != "0x0.0p+0"
        or requested.get("adaptive_mode") != "PRS23"
        or type(requested.get("initial_dt")) is not float
        or requested["initial_dt"].hex() != initial_dt.hex()
        or requested.get("exact_finish_time") != 1
        or requested.get("gravity") != "basic"
        or settings.get("target_basis") != "EXACT_CALLER_TARGET_VECTOR"
        or settings.get("target_count") != target_count
    ):
        raise BenchmarkError("IAS requested settings differ from exact protocol")
    for readback, stage in ((initial, "initial"), (post, "post")):
        if (
            readback.get("integrator_name") != "ias15"
            or type(readback.get("epsilon")) is not float
            or readback["epsilon"].hex() != epsilon.hex()
            or type(readback.get("minimum_dt")) is not float
            or readback["minimum_dt"].hex() != "0x0.0p+0"
            or readback.get("adaptive_mode") != "PRS23"
            or readback.get("gravity") != "basic"
            or readback.get("G") != 1.0
            or readback.get("N") != 3
            or readback.get("N_active") != 3
            or readback.get("testparticle_type") != 0
            or readback.get("collision") != "none"
            or readback.get("boundary") != "none"
            or readback.get("softening") != 0.0
            or readback.get("initial_state_raw_sha256") != INITIAL_RAW_SHA256
        ):
            raise BenchmarkError(f"IAS {stage} effective readback changed")
    if (
        type(initial.get("dt")) is not float
        or initial["dt"].hex() != initial_dt.hex()
        or type(initial.get("dt_last_done")) is not float
        or initial["dt_last_done"].hex() != "0x0.0p+0"
        or initial.get("steps_done") != 0
        or accounting.get("integrate_call_count") != integrate_call_count
        or accounting.get("synchronize_call_count") != integrate_call_count
        or type(accounting.get("steps_done")) is not int
        or accounting["steps_done"] <= 0
        or accounting.get("iterations_max_exceeded") != 0
    ):
        raise BenchmarkError("IAS target/accounting schema changed")


def _validate_study_rosters(study: StudyRun) -> None:
    profile = study.profile
    expected_jx_ids = tuple(f"jx_hybrid_p{value}" for value in profile.jx_divisors)
    if tuple(lane.engine_id for lane in study.jx_lanes) != expected_jx_ids:
        raise BenchmarkError("JX lane identifier/order roster changed")
    for lane in study.jx_lanes:
        if (
            lane.family != "JX_TRANSACTIONAL_WISDOM_HOLMAN_RKF78_HYBRID"
            or lane.method_id != HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID
            or lane.step_count != profile.step_count(lane.divisor)
        ):
            raise BenchmarkError("JX lane method/count schema changed")
        _validate_lane_epoch_roster(lane, profile, exact_jx_lattice=True)
        _validate_lane_mapping_roster(
            lane,
            settings=(
                "method_id", "wisdom_holman_spec", "encounter_control",
                "checkpoint_step_indices", "checkpoint_epoch_policy",
                "outer_semantic_replay_count", "nested_public_child_replay_count",
            ),
            accounting_required=(
                "completed_steps", "schedule_content_sha256",
                "step_ledger_content_sha256", "result_content_sha256",
                "primary_counts", "validation_replay_counts",
                "total_public_call_counts", "mode_counts", "near_reason_counts",
                "near_phase_counts",
                "child_summary",
            ),
        )
        expected_runtime = _jx_runtime_provenance()
        if not _canonical_equal(lane.runtime, expected_runtime):
            raise BenchmarkError("JX lane runtime source custody changed")

    status = _thaw_tree(study.external_status)
    executed = status.get("external_lanes_executed")
    if type(executed) is not bool:
        raise BenchmarkError("external execution status must be exact bool")
    if executed:
        expected_external_ids = tuple(
            f"rebound_{method}_p{divisor}"
            for method in ("mercurius", "trace")
            for divisor in profile.external_divisors
        )
        expected_reference_ids = tuple(
            f"rebound_ias15_actual_clock_p{divisor}"
            for divisor in profile.external_divisors
        )
        if tuple(lane.engine_id for lane in study.external_lanes) != expected_external_ids:
            raise BenchmarkError("external lane identifier/order roster changed")
        if tuple(lane.engine_id for lane in study.external_references) != expected_reference_ids:
            raise BenchmarkError("external IAS lane identifier/order roster changed")
        if study.jx_reference is None or study.jx_reference_sensitivity is None:
            raise BenchmarkError("executed external study lacks JX IAS sensitivity pair")
        for lane in study.external_lanes:
            method = lane.engine_id.split("_")[1]
            if (
                lane.family
                != f"REBOUND_5_1_1_{method.upper()}_EXTERNAL_COMPARATOR"
                or lane.method_id != f"rebound.integrator.{method}.5.1.1"
                or lane.step_count != profile.step_count(lane.divisor)
            ):
                raise BenchmarkError("external fixed lane method/count schema changed")
            _validate_lane_epoch_roster(lane, profile, exact_jx_lattice=False)
            _validate_lane_mapping_roster(
                lane,
                settings=(
                    "requested",
                    "fresh_integrator_readback_before_explicit_configuration",
                    "initial_effective_readback",
                    "post_step_effective_readback",
                    "provenance_categories",
                    "actual_clock_policy",
                ),
                accounting_required=(
                    "steps_call_count", "synchronize_call_count",
                    "every_node_observed_epochs_content_sha256",
                    "every_node_trajectory_raw_sha256",
                    "headline_checkpoint_step_indices",
                ),
            )
        for lane in study.external_references:
            if (
                lane.family != "REBOUND_5_1_1_IAS15_NUMERICAL_REFERENCE"
                or lane.method_id != "rebound.integrator.ias15.5.1.1"
                or lane.step_count != profile.step_count(lane.divisor)
            ):
                raise BenchmarkError("external IAS method/count schema changed")
            _validate_lane_mapping_roster(
                lane,
                settings=(
                    "requested", "initial_effective_readback",
                    "post_run_effective_readback", "provenance_categories",
                    "target_basis", "target_count",
                ),
                accounting_required=(
                    "integrate_call_count", "synchronize_call_count", "steps_done",
                    "every_node_observed_epochs_content_sha256",
                    "every_node_trajectory_raw_sha256",
                    "headline_checkpoint_step_indices",
                ),
            )
            _validate_ias_lane_schema(
                lane,
                epsilon=1.0e-12,
                initial_dt=float(PERIOD / lane.divisor),
                target_count=profile.step_count(lane.divisor) + 1,
                integrate_call_count=profile.step_count(lane.divisor),
            )
        for lane in study.external_lanes + study.external_references:
            expected_runtime = _PROVENANCE_SUPPORT._rebound_runtime_provenance(
                _load_rebound()
            )
            if not _canonical_equal(lane.runtime, expected_runtime):
                raise BenchmarkError("external lane runtime custody changed")
    else:
        if (
            study.external_lanes
            or study.external_references
            or study.jx_reference is not None
            or study.jx_reference_sensitivity is not None
        ):
            raise BenchmarkError("unexecuted external study retained external lanes")

    if executed:
        assert study.jx_reference is not None
        assert study.jx_reference_sensitivity is not None
        references = (
            (
                study.jx_reference,
                "rebound_ias15_jx_exact_labels_dt_p128",
                128,
            ),
            (
                study.jx_reference_sensitivity,
                "rebound_ias15_jx_exact_labels_dt_p1024_sensitivity",
                1024,
            ),
        )
        expected_epochs = study.jx_lanes[0].observed_epochs.tobytes(order="C")
        for lane, engine_id, divisor in references:
            if (
                lane.engine_id != engine_id
                or lane.divisor != divisor
                or lane.family != "REBOUND_5_1_1_IAS15_NUMERICAL_REFERENCE"
                or lane.method_id != "rebound.integrator.ias15.5.1.1"
                or lane.declared_epochs.tobytes(order="C") != expected_epochs
                or lane.observed_epochs.tobytes(order="C") != expected_epochs
            ):
                raise BenchmarkError("JX-label IAS reference schema changed")
            _validate_ias_lane_schema(
                lane,
                epsilon=1.0e-14,
                initial_dt=float(PERIOD / divisor),
                target_count=len(study.jx_lanes[0].observed_epochs),
                integrate_call_count=len(study.jx_lanes[0].observed_epochs) - 1,
            )
            expected_runtime = _PROVENANCE_SUPPORT._rebound_runtime_provenance(
                _load_rebound()
            )
            if not _canonical_equal(lane.runtime, expected_runtime):
                raise BenchmarkError("JX-label IAS runtime custody changed")


def _execute_study(profile: Profile, rebound_mode: str) -> StudyRun:
    if type(profile) is not Profile:
        raise BenchmarkError("profile must be an exact Profile")
    profile.__post_init__()
    if type(rebound_mode) is not str or rebound_mode not in REBOUND_MODES:
        raise BenchmarkError(f"rebound_mode must be one of {REBOUND_MODES!r}")
    jx_lanes = tuple(_jx_lane(profile, divisor) for divisor in profile.jx_divisors)
    all_far = _all_far_control()
    if rebound_mode == "disabled":
        return StudyRun(
            profile=profile,
            rebound_mode=rebound_mode,
            jx_lanes=jx_lanes,
            external_lanes=(),
            jx_reference=None,
            jx_reference_sensitivity=None,
            external_references=(),
            external_status={
                "requested_mode": rebound_mode,
                "external_lanes_executed": False,
                "reason": "EXPLICITLY_DISABLED",
                "required_version": REQUIRED_REBOUND_VERSION,
                "authority_authorized": False,
            },
            all_far_control=all_far,
            limitation_probes=_limitation_probes(profile, None, rebound_mode),
        )
    try:
        rebound = _load_rebound()
    except ReboundUnavailable:
        if rebound_mode == "required":
            raise
        return StudyRun(
            profile=profile,
            rebound_mode=rebound_mode,
            jx_lanes=jx_lanes,
            external_lanes=(),
            jx_reference=None,
            jx_reference_sensitivity=None,
            external_references=(),
            external_status={
                "requested_mode": rebound_mode,
                "external_lanes_executed": False,
                "reason": "EXACT_REBOUND_5_1_1_UNAVAILABLE",
                "required_version": REQUIRED_REBOUND_VERSION,
                "authority_authorized": False,
            },
            all_far_control=all_far,
            limitation_probes=_limitation_probes(profile, None, rebound_mode),
        )

    runtime = _PROVENANCE_SUPPORT._rebound_runtime_provenance(rebound)
    _PROVENANCE_SUPPORT._validate_rebound_runtime_provenance(runtime)
    common_jx_epochs = np.array(jx_lanes[0].observed_epochs, dtype=np.float64)
    if any(
        lane.observed_epochs.tobytes(order="C") != common_jx_epochs.tobytes(order="C")
        for lane in jx_lanes[1:]
    ):
        raise BenchmarkError("JX refinement lanes do not share exact checkpoint labels")
    jx_reference = _ias15_lane(
        rebound,
        engine_id="rebound_ias15_jx_exact_labels_dt_p128",
        divisor=128,
        targets=common_jx_epochs,
        epsilon=1.0e-14,
        initial_dt=JX_IAS15_INITIAL_DT,
        runtime=runtime,
        expected_full_hashes=False,
    )
    jx_reference_sensitivity = _ias15_lane(
        rebound,
        engine_id="rebound_ias15_jx_exact_labels_dt_p1024_sensitivity",
        divisor=1024,
        targets=np.array(common_jx_epochs, dtype=np.float64),
        epsilon=1.0e-14,
        initial_dt=JX_IAS15_SENSITIVITY_DT,
        runtime=runtime,
        expected_full_hashes=False,
    )
    if profile.named_full_profile:
        for lane, seed_divisor in (
            (jx_reference, 128),
            (jx_reference_sensitivity, 1024),
        ):
            expected = JX_IAS15_REFERENCE_KATS[seed_divisor]
            settings = _plain_tree(lane.settings)
            accounting = _plain_tree(lane.accounting)
            post = settings["post_run_effective_readback"]
            if (
                lane.final_state_sha256 != expected["final_state_sha256"]
                or lane.trajectory_sha256 != expected["trajectory_sha256"]
                or accounting["steps_done"] != expected["steps_done"]
                or float(post["dt"]).hex() != expected["post_dt_hex"]
                or float(post["dt_last_done"]).hex()
                != expected["post_dt_last_done_hex"]
            ):
                raise BenchmarkError("JX-label IAS15 reference fingerprint changed")

    transient_fixed: dict[tuple[str, int], Lane] = {}
    for method in ("mercurius", "trace"):
        for divisor in profile.external_divisors:
            transient_fixed[(method, divisor)] = _external_fixed_every_node(
                rebound, profile, method, divisor, runtime
            )
    external_references: list[Lane] = []
    external_headline: dict[tuple[str, int], Lane] = {}
    for divisor in profile.external_divisors:
        mercurius = transient_fixed[("mercurius", divisor)]
        trace = transient_fixed[("trace", divisor)]
        if (
            mercurius.observed_epochs.tobytes(order="C")
            != trace.observed_epochs.tobytes(order="C")
        ):
            raise BenchmarkError("MERCURIUS and TRACE observed clock vectors differ")
        reference_every_node = _ias15_lane(
            rebound,
            engine_id=f"rebound_ias15_actual_clock_p{divisor}",
            divisor=divisor,
            targets=np.array(mercurius.observed_epochs, dtype=np.float64),
            epsilon=1.0e-12,
            initial_dt=float(PERIOD / divisor),
            runtime=runtime,
            expected_full_hashes=profile.named_full_profile,
        )
        reference_headline = _headline_lane(
            reference_every_node, profile, all_node_accuracy=None
        )
        external_references.append(reference_headline)
        for method in ("mercurius", "trace"):
            candidate = transient_fixed[(method, divisor)]
            all_node_accuracy = _accuracy_metrics(candidate, reference_every_node)
            external_headline[(method, divisor)] = _headline_lane(
                candidate, profile, all_node_accuracy=all_node_accuracy
            )
    external_lanes = tuple(
        external_headline[(method, divisor)]
        for method in ("mercurius", "trace")
        for divisor in profile.external_divisors
    )
    return StudyRun(
        profile=profile,
        rebound_mode=rebound_mode,
        jx_lanes=jx_lanes,
        external_lanes=external_lanes,
        jx_reference=jx_reference,
        jx_reference_sensitivity=jx_reference_sensitivity,
        external_references=tuple(external_references),
        external_status={
            "requested_mode": rebound_mode,
            "external_lanes_executed": True,
            "reason": "EXACT_REBOUND_5_1_1_LOADED",
            "required_version": REQUIRED_REBOUND_VERSION,
            "authority_authorized": False,
            "external_state_transfer_between_lanes": False,
            "orbital_reconstruction_at_execution": False,
        },
        all_far_control=all_far,
        limitation_probes=_limitation_probes(profile, rebound, rebound_mode),
    )


def _lane_equal_ignoring_timing(left: Lane, right: Lane) -> bool:
    if type(left) is not Lane or type(right) is not Lane:
        return False
    try:
        left.__post_init__()
        right.__post_init__()
    except BenchmarkError:
        return False
    for name in (
        "engine_id", "family", "method_id", "divisor", "fixed_step", "step_count",
        "final_state_sha256", "trajectory_sha256",
    ):
        first = getattr(left, name)
        second = getattr(right, name)
        if type(first) is not type(second):
            return False
        if type(first) is float:
            if first.hex() != second.hex():
                return False
        elif first != second:
            return False
    for name in ("declared_epochs", "observed_epochs", "positions", "velocities"):
        first = getattr(left, name)
        second = getattr(right, name)
        if (
            first.dtype != second.dtype
            or first.shape != second.shape
            or first.tobytes(order="C") != second.tobytes(order="C")
        ):
            return False
    if any(
        not _canonical_equal(getattr(left, name), getattr(right, name))
        for name in ("settings", "accounting", "runtime")
    ):
        return False
    return all(
        type(value) is float and math.isfinite(value) and value >= 0.0
        for value in (left.raw_timing_seconds, right.raw_timing_seconds)
    )


def _validate_study_against_replay(primary: StudyRun, replay: StudyRun) -> None:
    if type(primary) is not StudyRun or type(replay) is not StudyRun:
        raise BenchmarkError("semantic replay requires exact StudyRun objects")
    primary.__post_init__()
    replay.__post_init__()
    if primary.profile != replay.profile or primary.rebound_mode != replay.rebound_mode:
        raise BenchmarkError("semantic replay profile/mode changed")
    for name in ("jx_lanes", "external_lanes", "external_references"):
        left = getattr(primary, name)
        right = getattr(replay, name)
        if len(left) != len(right) or any(
            not _lane_equal_ignoring_timing(first, second)
            for first, second in zip(left, right)
        ):
            raise BenchmarkError(f"{name} differs from independent semantic replay")
    for name in ("jx_reference", "jx_reference_sensitivity"):
        left = getattr(primary, name)
        right = getattr(replay, name)
        if (left is None) != (right is None) or (
            left is not None and right is not None and not _lane_equal_ignoring_timing(left, right)
        ):
            raise BenchmarkError(f"{name} differs from independent semantic replay")
    for name in ("external_status", "all_far_control", "limitation_probes"):
        if not _canonical_equal(getattr(primary, name), getattr(replay, name)):
            raise BenchmarkError(f"{name} differs from independent semantic replay")


def _plain_tree(value: Any) -> Any:
    if type(value) is _MAPPING_PROXY_TYPE or type(value) is dict:
        if tuple(value.keys()) == ("float_hex",) and type(value["float_hex"]) is str:
            result = float.fromhex(value["float_hex"])
            if not math.isfinite(result):
                raise BenchmarkError("canonical custody contained a nonfinite float")
            return result
        return {key: _plain_tree(value[key]) for key in value}
    if type(value) in (tuple, list):
        return [_plain_tree(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    raise BenchmarkError("canonical custody tree has an unsupported retained value")


def _lane_summary(lane: Lane, reference: Lane | None) -> dict[str, Any]:
    lane.__post_init__()
    accuracy = None if reference is None else _accuracy_metrics(lane, reference)
    clock_delta = lane.observed_epochs - lane.declared_epochs
    adaptive_reference = lane.method_id == "rebound.integrator.ias15.5.1.1"
    return {
        "engine_id": lane.engine_id,
        "family": lane.family,
        "method_id": lane.method_id,
        "divisor": lane.divisor,
        "lattice_step_or_adaptive_initial_dt_seed": lane.fixed_step,
        "lattice_step_or_adaptive_initial_dt_seed_hex": lane.fixed_step.hex(),
        "step_value_role": (
            "ADAPTIVE_IAS15_INITIAL_DT_SEED"
            if adaptive_reference else "FIXED_OUTER_MACROSTEP"
        ),
        "outer_step_or_target_interval_count": lane.step_count,
        "count_role": (
            "IAS15_TARGET_INTERVAL_COUNT_BEFORE_HEADLINE_DOWNSAMPLING"
            if adaptive_reference else "FIXED_OUTER_MACROSTEP_COUNT"
        ),
        "headline_retained_node_count": len(lane.observed_epochs),
        "headline_declared_epoch_hex": tuple(
            float(value).hex() for value in lane.declared_epochs
        ),
        "headline_observed_epoch_hex": tuple(
            float(value).hex() for value in lane.observed_epochs
        ),
        "headline_clock_drift": {
            "maximum_abs": float(np.max(np.abs(clock_delta))),
            "final_signed": float(clock_delta[-1]),
            "state_relabelled_to_declared_epoch": False,
        },
        "headline_accuracy_against_numerical_reference": accuracy,
        "headline_invariants": _invariant_metrics(
            lane.positions, lane.velocities
        ),
        "headline_final_state_raw_sha256": lane.final_state_sha256,
        "headline_trajectory_raw_sha256": lane.trajectory_sha256,
        "declared_epochs_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.declared-epochs.v1", lane.declared_epochs
        ),
        "observed_epochs_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.observed-epochs.v1", lane.observed_epochs
        ),
        "positions_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.positions.v1", lane.positions
        ),
        "velocities_content_sha256": _array_component_sha256(
            "jx.rebound-hybrid.velocities.v1", lane.velocities
        ),
        "lane_content_sha256": _lane_content_sha256(lane),
        "digest_classification": "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
        "digest_authority_authorized": False,
        "settings": _plain_tree(lane.settings),
        "accounting": _plain_tree(lane.accounting),
        "runtime": _plain_tree(lane.runtime),
        "raw_primary_lane_timing_seconds": lane.raw_timing_seconds,
    }


def _reference_summary(lane: Lane) -> dict[str, Any]:
    return {
        "engine_id": lane.engine_id,
        "role": "HIGH_ACCURACY_EXTERNAL_NUMERICAL_REFERENCE",
        "reference_truth_claimed": False,
        "headline_retained_node_count": len(lane.observed_epochs),
        "target_epoch_hex": tuple(float(value).hex() for value in lane.declared_epochs),
        "observed_epoch_hex": tuple(float(value).hex() for value in lane.observed_epochs),
        "headline_final_state_raw_sha256": lane.final_state_sha256,
        "headline_trajectory_raw_sha256": lane.trajectory_sha256,
        "lane_content_sha256": _lane_content_sha256(lane),
        "digest_classification": "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
        "digest_authority_authorized": False,
        "settings": _plain_tree(lane.settings),
        "accounting": _plain_tree(lane.accounting),
        "runtime": _plain_tree(lane.runtime),
        "raw_primary_lane_timing_seconds": lane.raw_timing_seconds,
    }


def _refinement_summary(
    lanes: tuple[Lane, ...], reference_for: Mapping[int, Lane]
) -> dict[str, Any]:
    if type(lanes) is not tuple or len(lanes) < 1:
        raise BenchmarkError("refinement summary requires an exact nonempty lane tuple")
    ordered = tuple(sorted(lanes, key=lambda lane: lane.divisor))
    points: list[dict[str, Any]] = []
    for lane in ordered:
        reference = reference_for[lane.divisor]
        accuracy = _accuracy_metrics(lane, reference)
        invariants = _invariant_metrics(lane.positions, lane.velocities)
        points.append(
            {
                "divisor": lane.divisor,
                "position_vector_l2_max": accuracy["position"]["vector_l2_max"],
                "position_vector_l2_rms": accuracy["position"]["vector_l2_rms"],
                "velocity_vector_l2_max": accuracy["velocity"]["vector_l2_max"],
                "velocity_vector_l2_rms": accuracy["velocity"]["vector_l2_rms"],
                "relative_vector_phase_proxy_max_abs_radians": (
                    accuracy["inner_outer_relative_vector_phase_proxy"][
                        "maximum_abs_radians"
                    ]
                ),
                "relative_total_energy_max": invariants["relative_total_energy_max"],
            }
        )
    metric_names = tuple(key for key in points[0] if key != "divisor")
    orders = []
    for coarse, fine in zip(points, points[1:]):
        orders.append(
            {
                "coarse_divisor": coarse["divisor"],
                "fine_divisor": fine["divisor"],
                **{
                    name: _order(float(coarse[name]), float(fine[name]))
                    for name in metric_names
                },
            }
        )
    return {
        "checkpoint_cadence": (
            "COMMON_103_HEADLINE_LABELS"
            if len(ordered[0].observed_epochs) == 103
            else "COMMON_PROFILE_HEADLINE_LABELS"
        ),
        "retained_node_count": len(ordered[0].observed_epochs),
        "points": tuple(points),
        "observed_orders": tuple(orders),
        "fixture_specific_empirical_order_only": True,
        "global_or_theoretical_order_claimed": False,
    }


def _all_true(value: Any) -> bool:
    if type(value) is bool:
        return value
    if type(value) is dict:
        return all(_all_true(item) for item in value.values())
    if type(value) in (tuple, list):
        return all(_all_true(item) for item in value)
    return True


def _maximum_gate(observed: float, maximum: float) -> dict[str, Any]:
    return {
        "observed": float(observed),
        "operator": "<=",
        "threshold": float(maximum),
        "passed": float(observed) <= float(maximum),
    }


def _minimum_gate(observed: float, minimum: float) -> dict[str, Any]:
    return {
        "observed": float(observed),
        "operator": ">=",
        "threshold": float(minimum),
        "passed": float(observed) >= float(minimum),
    }


def _window_gate(observed: float, window: tuple[float, float]) -> dict[str, Any]:
    return {
        "observed": float(observed),
        "operator": "CLOSED_INTERVAL",
        "window": window,
        "passed": window[0] <= float(observed) <= window[1],
    }


def _evaluate_regression_gates(
    study: StudyRun,
    jx_refinement: dict[str, Any] | None,
    external_refinements: dict[str, Any] | None,
) -> dict[str, Any]:
    if not study.profile.named_full_profile:
        return {
            "evaluated": False,
            "reason": "ONLY_THE_NAMED_FULL_PROFILE_HAS_CALIBRATED_GATES",
            "passed": None,
        }
    if jx_refinement is None or study.jx_reference is None:
        return {
            "evaluated": False,
            "reason": "EXACT_REBOUND_5_1_1_REFERENCE_UNAVAILABLE_OR_DISABLED",
            "jx_fixed_lanes_executed": True,
            "calibrated_accuracy_and_order_gates_evaluated": False,
            "passed": None,
        }
    points = tuple(jx_refinement["points"])
    orders = tuple(jx_refinement["observed_orders"])
    monotone_metrics = tuple(key for key in points[0] if key != "divisor")
    monotone = {
        key: {
            "observed_by_divisor": tuple(
                (point["divisor"], float(point[key])) for point in points
            ),
            "operator": "NONINCREASING_WITH_REFINEMENT",
            "passed": all(
                float(fine[key]) <= float(coarse[key])
                for coarse, fine in zip(points, points[1:])
            ),
        }
        for key in monotone_metrics
    }
    finest_lane = study.jx_lanes[-1]
    finest_accuracy = _accuracy_metrics(finest_lane, study.jx_reference)
    finest_invariants = _invariant_metrics(
        finest_lane.positions, finest_lane.velocities
    )
    fine_order = orders[-1]
    jx_gates = {
        "monotone_all_levels": monotone,
        "fine_pair_state_and_phase_order": {
            "position": _minimum_gate(
                fine_order["position_vector_l2_max"], JX_FINE_STATE_ORDER_MINIMUM
            ),
            "velocity": _minimum_gate(
                fine_order["velocity_vector_l2_max"], JX_FINE_STATE_ORDER_MINIMUM
            ),
            "phase_proxy": _minimum_gate(
                fine_order["relative_vector_phase_proxy_max_abs_radians"],
                JX_FINE_STATE_ORDER_MINIMUM,
            ),
            "energy": _minimum_gate(
                fine_order["relative_total_energy_max"],
                JX_FINE_ENERGY_ORDER_MINIMUM,
            ),
        },
        "finest_envelope": {
            "position_vector_l2_max": _maximum_gate(
                finest_accuracy["position"]["vector_l2_max"],
                JX_FINEST_ENVELOPE["position_vector_l2_max"],
            ),
            "position_vector_l2_rms": _maximum_gate(
                finest_accuracy["position"]["vector_l2_rms"],
                JX_FINEST_ENVELOPE["position_vector_l2_rms"],
            ),
            "velocity_vector_l2_max": _maximum_gate(
                finest_accuracy["velocity"]["vector_l2_max"],
                JX_FINEST_ENVELOPE["velocity_vector_l2_max"],
            ),
            "velocity_vector_l2_rms": _maximum_gate(
                finest_accuracy["velocity"]["vector_l2_rms"],
                JX_FINEST_ENVELOPE["velocity_vector_l2_rms"],
            ),
            "relative_vector_phase_proxy": _maximum_gate(
                finest_accuracy["inner_outer_relative_vector_phase_proxy"]
                ["maximum_abs_radians"],
                JX_FINEST_ENVELOPE["phase_max_abs_radians"],
            ),
            "relative_total_energy": _maximum_gate(
                finest_invariants["relative_total_energy_max"],
                JX_FINEST_ENVELOPE["relative_total_energy_max"],
            ),
            "relative_angular_momentum": _maximum_gate(
                finest_invariants["relative_angular_momentum_max"],
                JX_FINEST_ENVELOPE["relative_angular_momentum_max"],
            ),
            "center_of_mass_position_absolute": _maximum_gate(
                finest_invariants["center_of_mass_position_absolute_max"],
                JX_FINEST_ENVELOPE["center_of_mass_position_max"],
            ),
            "total_momentum_absolute": _maximum_gate(
                finest_invariants["total_momentum_absolute_max"],
                JX_FINEST_ENVELOPE["total_momentum_max"],
            ),
        },
    }
    assert study.jx_reference_sensitivity is not None
    sensitivity = _accuracy_metrics(
        study.jx_reference, study.jx_reference_sensitivity
    )
    jx_gates["ias15_initial_dt_sensitivity"] = {
        "authoritative_initial_dt": JX_IAS15_INITIAL_DT,
        "sensitivity_initial_dt": JX_IAS15_SENSITIVITY_DT,
        "position_vector_l2_max": _maximum_gate(
            sensitivity["position"]["vector_l2_max"],
            REFERENCE_PERTURBATION_MAXIMUM,
        ),
        "velocity_vector_l2_max": _maximum_gate(
            sensitivity["velocity"]["vector_l2_max"],
            REFERENCE_PERTURBATION_MAXIMUM,
        ),
    }
    accounting_gates: dict[str, Any] = {}
    for lane in study.jx_lanes:
        accounting = _plain_tree(lane.accounting)
        mode_counts = accounting["mode_counts"]
        reason_counts = accounting["near_reason_counts"]
        phase_counts = accounting["near_phase_counts"]
        child = accounting["child_summary"]
        expected_far, expected_near = JX_EXPECTED_MODE_COUNTS[lane.divisor]
        observed_far = mode_counts.get("WISDOM_HOLMAN_FAR", 0)
        observed_near = mode_counts.get("CARTESIAN_RKF78_NEAR_FULL_INTERVAL", 0)
        path_reason = "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN"
        path_phase = "POST_DRIFT_PRE_FORCE_PATH_SCREEN"
        path_switches = reason_counts.get(path_reason, 0)
        accounting_gates[f"p{lane.divisor}"] = {
            "far_count": {
                "observed": observed_far,
                "expected": expected_far,
                "passed": observed_far == expected_far,
            },
            "near_count": {
                "observed": observed_near,
                "expected": expected_near,
                "passed": observed_near == expected_near,
            },
            "real_path_screen_switch_reason": {
                "observed": reason_counts,
                "required_reason": path_reason,
                "required_nonzero_observed_count": path_switches,
                "total_expected": expected_near,
                "passed": path_switches > 0
                and sum(reason_counts.values()) == expected_near,
            },
            "real_path_screen_switch_phase": {
                "observed": phase_counts,
                "path_phase": path_phase,
                "path_phase_expected_from_reason_count": path_switches,
                "accepted_start_phase_expected": expected_near - path_switches,
                "passed": (
                    set(phase_counts)
                    <= {"ACCEPTED_START_NODE", path_phase}
                    and phase_counts.get(path_phase, 0) == path_switches
                    and phase_counts.get("ACCEPTED_START_NODE", 0)
                    == expected_near - path_switches
                    and sum(phase_counts.values()) == expected_near
                ),
            },
            "far_plus_near_equals_outer_steps": {
                "observed": observed_far + observed_near,
                "expected": lane.step_count,
                "passed": observed_far + observed_near == lane.step_count,
            },
            "private_child_count_equals_near": {
                "observed": child["count"],
                "expected": expected_near,
                "passed": child["count"] == expected_near,
            },
            "child_proposals": {
                "observed": child["proposal_count"],
                "expected": expected_near,
                "passed": child["proposal_count"] == expected_near,
            },
            "child_accepted_substeps": {
                "observed": child["accepted_substep_count"],
                "expected": expected_near,
                "passed": child["accepted_substep_count"] == expected_near,
            },
            "child_rejected_substeps": {
                "observed": child["rejected_substep_count"],
                "expected": 0,
                "passed": child["rejected_substep_count"] == 0,
            },
            "child_force_evaluations": {
                "observed": child["force_evaluations"],
                "expected": 13 * expected_near,
                "passed": child["force_evaluations"] == 13 * expected_near,
            },
            "certificate_rejections": {
                "observed": child["certificate_rejections"],
                "expected": 0,
                "passed": child["certificate_rejections"] == 0,
            },
            "each_near_child_exact_one_step_cadence": {
                "passed": child[
                    "each_private_child_one_proposal_one_accept_zero_reject_thirteen_forces"
                ]
                is True,
            },
            "primary_equals_semantic_replay": {
                "passed": accounting[
                    "primary_counts_equal_validation_replay_counts"
                ]
                is True,
            },
            "public_totals_exactly_double_primary": {
                "passed": accounting[
                    "total_public_counts_exactly_double_primary"
                ]
                is True,
            },
        }
    jx_gates["switch_child_and_replay_accounting"] = accounting_gates

    external_gates: dict[str, Any] = {}
    if external_refinements is not None:
        for method, refinement in external_refinements.items():
            method_orders = tuple(refinement["observed_orders"])
            selected = method_orders if method == "mercurius" else method_orders[1:]
            order_metrics = (
                "position_vector_l2_max", "velocity_vector_l2_max",
                "relative_vector_phase_proxy_max_abs_radians",
            )
            external_gates[method] = {
                "order_window_intervals": tuple(
                    (row["coarse_divisor"], row["fine_divisor"])
                    for row in selected
                ),
                "fixture_specific_second_order_window": {
                    metric: tuple(
                        {
                            "coarse_divisor": row["coarse_divisor"],
                            "fine_divisor": row["fine_divisor"],
                            **_window_gate(float(row[metric]), EXTERNAL_ORDER_WINDOW),
                        }
                        for row in selected
                    )
                    for metric in order_metrics
                },
            }
        for method in ("mercurius", "trace"):
            lane = next(
                item
                for item in study.external_lanes
                if item.engine_id == f"rebound_{method}_p256"
            )
            accounting = _plain_tree(lane.accounting)
            accuracy = accounting["every_node_accuracy_against_actual_clock_ias15"]
            invariants = accounting["every_node_invariants"]
            external_gates[method]["p256_descriptive_workload_envelope"] = {
                "position": _maximum_gate(
                    accuracy["position"]["vector_l2_max"], 6.0e-3
                ),
                "velocity": _maximum_gate(
                    accuracy["velocity"]["vector_l2_max"], 1.5e-3
                ),
                "relative_energy": _maximum_gate(
                    invariants["relative_total_energy_max"], 1.0e-6
                ),
                "relative_angular_momentum": _maximum_gate(
                    invariants["relative_angular_momentum_max"], 3.0e-14
                ),
                "center_of_mass_absolute": _maximum_gate(
                    invariants["center_of_mass_position_absolute_max"], 1.0e-12
                ),
                "momentum_absolute": _maximum_gate(
                    invariants["total_momentum_absolute_max"], 1.0e-12
                ),
            }
    result = {
        "evaluated": True,
        "jx": jx_gates,
        "external": external_gates,
        "descriptive_regression_only": True,
        "qualification_or_superiority_authorized": False,
    }
    result["passed"] = _all_true({"jx": jx_gates, "external": external_gates})
    if result["passed"] is not True:
        raise BenchmarkError("named full-profile descriptive regression gate failed")
    return result


def _authorized_timing_exclusion_path(path: tuple[str, ...]) -> bool:
    if path in (
        ("timing_disclosure", "summed_primary_lane_timing_seconds"),
        ("timing_disclosure", "semantic_replay_wall_seconds"),
    ):
        return True
    return (
        len(path) == 3
        and path[0] in ("headline_lanes", "numerical_references")
        and path[2] == "raw_primary_lane_timing_seconds"
    )


def _strip_timing(value: Any, _path: tuple[str, ...] = ()) -> Any:
    if type(value) is dict:
        return {
            key: _strip_timing(nested, _path + (key,))
            for key, nested in value.items()
            if not _authorized_timing_exclusion_path(_path + (key,))
        }
    if type(value) in (tuple, list):
        return [_strip_timing(item, _path + (str(index),)) for index, item in enumerate(value)]
    return value


def _assert_no_forbidden_keys(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if key in FORBIDDEN_RESULT_KEYS:
                raise BenchmarkError(f"forbidden comparative key {key!r}")
            _assert_no_forbidden_keys(nested)
    elif type(value) in (tuple, list):
        for nested in value:
            _assert_no_forbidden_keys(nested)


REPORT_FIELD_ROSTER = (
    "schema", "benchmark_id", "profile", "problem", "method_provenance",
    "runtime_provenance", "headline_lanes", "numerical_references",
    "refinement", "regression_gates", "all_far_control", "limitations",
    "timing_disclosure", "interpretation", "claim_controls", "serialization",
    "content_integrity",
)


def _validate_report_timing_fields(report: dict[str, Any]) -> None:
    timing = report.get("timing_disclosure")
    if type(timing) is not dict:
        raise BenchmarkError("report timing disclosure must be an exact mapping")

    def require_nonnegative_float(container: dict[str, Any], key: str) -> None:
        value = container.get(key)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise BenchmarkError(f"report timing field {key!r} is missing or invalid")

    require_nonnegative_float(timing, "summed_primary_lane_timing_seconds")
    require_nonnegative_float(timing, "semantic_replay_wall_seconds")
    for section_name in ("headline_lanes", "numerical_references"):
        section = report.get(section_name)
        if type(section) is not dict:
            raise BenchmarkError(f"report {section_name} must be an exact mapping")
        for engine_id, lane in section.items():
            if type(engine_id) is not str or type(lane) is not dict:
                raise BenchmarkError(f"report {section_name} lane schema changed")
            require_nonnegative_float(lane, "raw_primary_lane_timing_seconds")


def _validate_report_content_integrity(report: Any) -> None:
    if (
        type(report) is not dict
        or len(report) != len(REPORT_FIELD_ROSTER)
        or set(report) != set(REPORT_FIELD_ROSTER)
    ):
        raise BenchmarkError("report content validation requires the exact field roster")
    _validate_tree_envelope(report, label="saved report content")
    _validate_report_timing_fields(report)
    integrity = report.get("content_integrity")
    if type(integrity) is not dict:
        raise BenchmarkError("report content-integrity manifest is missing")
    digest = integrity.get("report_semantic_content_sha256")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise BenchmarkError("report semantic digest is malformed")
    payload = dict(report)
    manifest = dict(integrity)
    del manifest["report_semantic_content_sha256"]
    payload["content_integrity"] = manifest
    expected = _domain_sha256(
        "jx.rebound-hybrid.report-semantic-content.v1", _strip_timing(payload)
    )
    if digest != expected:
        raise BenchmarkError("report semantic content digest changed")


def build_report(primary: StudyRun) -> dict[str, Any]:
    if type(primary) is not StudyRun:
        raise BenchmarkError("build_report requires an exact StudyRun")
    primary.__post_init__()
    replay_started = time.perf_counter()
    replay = _execute_study(primary.profile, primary.rebound_mode)
    replay_seconds = float(time.perf_counter() - replay_started)
    _validate_study_against_replay(primary, replay)
    external_executed = bool(
        _plain_tree(primary.external_status)["external_lanes_executed"]
    )
    jx_reference_map: dict[int, Lane] = {}
    if primary.jx_reference is not None:
        jx_reference_map = {
            lane.divisor: primary.jx_reference for lane in primary.jx_lanes
        }
    jx_refinement = (
        _refinement_summary(primary.jx_lanes, jx_reference_map)
        if jx_reference_map
        else None
    )
    external_refinement: dict[str, Any] | None = None
    external_reference_by_divisor = {
        lane.divisor: lane for lane in primary.external_references
    }
    if external_executed:
        external_refinement = {
            method: _refinement_summary(
                tuple(
                    lane
                    for lane in primary.external_lanes
                    if lane.engine_id.startswith(f"rebound_{method}_")
                ),
                external_reference_by_divisor,
            )
            for method in ("mercurius", "trace")
        }
    lane_summaries: dict[str, Any] = {}
    for lane in primary.jx_lanes:
        lane_summaries[lane.engine_id] = _lane_summary(
            lane, primary.jx_reference
        )
    for lane in primary.external_lanes:
        lane_summaries[lane.engine_id] = _lane_summary(
            lane, external_reference_by_divisor[lane.divisor]
        )
    reference_lanes = tuple(
        lane
        for lane in (
            primary.jx_reference,
            primary.jx_reference_sensitivity,
            *primary.external_references,
        )
        if lane is not None
    )
    gates = _evaluate_regression_gates(
        primary, jx_refinement, external_refinement
    )
    initial = _initial_rows()
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "profile": {
            "name": primary.profile.name,
            "named_full_profile": primary.profile.named_full_profile,
            "include_limitations": primary.profile.include_limitations,
            "jx_divisors": primary.profile.jx_divisors,
            "external_divisors": primary.profile.external_divisors,
            "base_steps_at_p256": primary.profile.base_steps_at_256,
            "headline_checkpoint_policy": (
                "COMMON_P_OVER_16_LABELS_PLUS_EXACT_FINAL_LABEL"
            ),
            "headline_full_profile_node_count": 103,
        },
        "problem": {
            "model": "LOCKED_ALL_ACTIVE_CLOSE_SCATTER_NEWTONIAN_THREE_BODY",
            "G": 1.0,
            "body_order": BODY_IDS,
            "initial_rows_float_hex": INITIAL_ROWS_HEX,
            "initial_rows_raw_little_endian_float64_sha256": INITIAL_RAW_SHA256,
            "initial_rows_reconstructed_from_orbital_elements_at_runtime": False,
            "initial_rows_moved_to_center_of_mass_at_runtime": False,
            "all_bodies_active": True,
            "massless_test_particles": False,
            "initial_center_of_mass_position_absolute": float(
                np.linalg.norm(
                    np.sum(initial[:, 0, None] * initial[:, 2:5], axis=0)
                    / np.sum(initial[:, 0])
                )
            ),
        },
        "method_provenance": {
            "jx_method_id": HYBRID_WISDOM_HOLMAN_RKF78_METHOD_ID,
            "mercurius": {
                "role": "OPTIONAL_EXTERNAL_GPL_FAMILY_COMPARATOR",
                "documentation": "https://rebound.hanno-rein.de/integrators/mercurius/",
                "paper": "https://academic.oup.com/mnras/article/485/4/5490/5380811",
            },
            "trace": {
                "role": "OPTIONAL_EXTERNAL_GPL_FAMILY_COMPARATOR",
                "documentation": "https://rebound.hanno-rein.de/integrators/trace/",
                "implementation_paper": "https://academic.oup.com/mnras/article/533/3/3708/7735374",
                "method_paper": "https://academic.oup.com/mnras/article/522/3/4639/7068101",
                "protocol_direction": "FORWARD_ONLY",
            },
            "ias15": {
                "role": "HIGH_ACCURACY_EXTERNAL_NUMERICAL_REFERENCE",
                "documentation": "https://rebound.hanno-rein.de/integrators/ias15/",
                "reference_truth_claimed": False,
            },
            "rebound_tag": "https://github.com/hannorein/rebound/tree/5.1.1",
            "rebound_license": "https://github.com/hannorein/rebound/blob/5.1.1/LICENSE",
            "legal_or_redistribution_conclusion_authorized": False,
            "rebound_code_vendored_or_copied_or_linked_into_jx": False,
        },
        "runtime_provenance": {
            "jx": _plain_tree(primary.jx_lanes[0].runtime),
            "rebound": (
                _plain_tree(reference_lanes[0].runtime)
                if reference_lanes else None
            ),
            "external_status": _plain_tree(primary.external_status),
        },
        "headline_lanes": lane_summaries,
        "numerical_references": {
            lane.engine_id: _reference_summary(lane) for lane in reference_lanes
        },
        "refinement": {
            "jx_hybrid": jx_refinement,
            "external": external_refinement,
            "jx_coarse_pair_is_monotonicity_only": True,
            "trace_order_window_excludes_p128_to_p256": True,
        },
        "regression_gates": gates,
        "all_far_control": _plain_tree(primary.all_far_control),
        "limitations": _plain_tree(primary.limitation_probes),
        "timing_disclosure": {
            "summed_primary_lane_timing_seconds": float(
                sum(lane.raw_timing_seconds for lane in (
                    primary.jx_lanes + primary.external_lanes + reference_lanes
                ))
            ),
            "semantic_replay_wall_seconds": replay_seconds,
            "one_outer_report_semantic_replay": True,
            "jx_public_hybrid_internal_semantic_replay_included_in_lane_timing": True,
            "no_nested_public_child_replay": True,
            "raw_timings_are_diagnostic_only": True,
            "timing_comparable": False,
        },
        "interpretation": {
            "headline_metrics_share_common_profile_label_cadence": True,
            "headline_metrics_share_103_labels": (
                primary.profile.named_full_profile
            ),
            "headline_retained_node_count": len(primary.jx_lanes[0].observed_epochs),
            "external_every_node_extrema_are_separately_labeled": True,
            "external_state_compared_to_ias15_at_actual_observed_clock": True,
            "external_state_relabelled_to_nominal_grid": False,
            "jx_ias15_authoritative_initial_dt": JX_IAS15_INITIAL_DT,
            "jx_ias15_sensitivity_initial_dt": JX_IAS15_SENSITIVITY_DT,
            "phase_metric_is_inner_outer_relative_vector_angle_proxy": True,
            "phase_proxy_sign_is_reference_minus_candidate": True,
            "phase_proxy_is_mean_longitude_or_general_orbital_phase": False,
            "small_invariant_or_energy_error_implies_small_state_or_phase_error": False,
            "fixture_specific_empirical_order_may_be_reported": True,
            "global_or_theoretical_order_claimed": False,
            "external_internal_force_switch_substep_counts_exposed": False,
            "local_ivp_certificate_is_not_global_clearance": True,
            "collision_event_location_or_response_claimed": False,
            "cross_divisor_external_orders_use_divisor_specific_observed_clocks": True,
            "cross_divisor_external_observed_epoch_vectors_are_identical": False,
            "numerical_reference_protocols": {
                "jx_headline_reference": (
                    "IAS15 epsilon=1e-14, PRS23, min_dt=0, initial_dt=P/128; "
                    "sequential exact-finish stops only at common JX headline labels"
                ),
                "jx_headline_sensitivity": (
                    "same exact labels/settings with initial_dt=P/1024"
                ),
                "external_actual_clock_reference": (
                    "IAS15 epsilon=1e-12, PRS23, min_dt=0, initial_dt=lane h; "
                    "sequential exact-finish stops at every observed outer node before "
                    "headline downsampling"
                ),
                "adaptive_steps_done_or_timing_cross_compared": False,
            },
            "metric_definitions": {
                "position_or_velocity_vector_l2_max": (
                    "maximum Euclidean three-vector error over retained nodes and bodies"
                ),
                "position_or_velocity_vector_l2_rms": (
                    "sqrt(mean of squared Euclidean three-vector errors over retained "
                    "node-by-body samples)"
                ),
                "relative_vector_phase_proxy_max_or_rms": (
                    "maximum absolute value or RMS over retained nodes of the signed "
                    "inner-to-outer relative-position angle in the xy plane"
                ),
                "invariant_drift_baseline": "first retained state in that lane",
                "center_of_mass_position_absolute": (
                    "Euclidean norm of the GM-weighted center-of-mass position"
                ),
                "total_momentum_absolute": "Euclidean norm of total linear momentum",
                "center_of_mass_or_momentum_drift": (
                    "Euclidean norm of value minus its first retained-state value"
                ),
            },
        },
        "claim_controls": dict(CLAIM_CONTROLS),
        "serialization": {
            "canonical_json": "SORTED_ASCII_JSON_WITH_BINARY64_FLOAT_HEX_OBJECTS",
            "canonical_pretraversal_node_cap": 1_000_000,
            "canonical_pretraversal_cumulative_byte_cap": MAXIMUM_CANONICAL_BYTES,
            "component_digests_are_domain_separated": True,
            "report_semantic_digest_excludes_raw_timing_fields": True,
            "byte_determinism_of_pretty_report_claimed": False,
        },
    }
    ordered_lanes = primary.jx_lanes + primary.external_lanes + reference_lanes
    core["content_integrity"] = {
        "classification": "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
        "authority_authorized": False,
        "array_component_domain_roster": (
            "jx.rebound-hybrid.declared-epochs.v1",
            "jx.rebound-hybrid.observed-epochs.v1",
            "jx.rebound-hybrid.positions.v1",
            "jx.rebound-hybrid.velocities.v1",
        ),
        "lane_content_domain": "jx.rebound-hybrid.lane-content.v1",
        "report_semantic_content_domain": (
            "jx.rebound-hybrid.report-semantic-content.v1"
        ),
        "ordered_lane_ids": tuple(lane.engine_id for lane in ordered_lanes),
        "ordered_lane_content_sha256": tuple(
            _lane_content_sha256(lane) for lane in ordered_lanes
        ),
        "aggregate_digest_excludes_itself": True,
        "raw_timing_fields_excluded": (
            "raw_primary_lane_timing_seconds",
            "summed_primary_lane_timing_seconds",
            "semantic_replay_wall_seconds",
        ),
    }
    core["content_integrity"]["report_semantic_content_sha256"] = _domain_sha256(
        "jx.rebound-hybrid.report-semantic-content.v1", _strip_timing(core)
    )
    if tuple(core) != REPORT_FIELD_ROSTER:
        raise BenchmarkError("report top-level field roster changed")
    _assert_no_forbidden_keys(core)
    _validate_report_content_integrity(core)
    _canonical_json(core)
    json.dumps(core, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    return core


def run_comparison(profile: Profile, rebound_mode: str = "auto") -> dict[str, Any]:
    return build_report(_execute_study(profile, rebound_mode))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the JX transactional WH/RKF78 hybrid and optional exact "
            "REBOUND 5.1.1 MERCURIUS/TRACE comparators on a locked close scatter."
        )
    )
    parser.add_argument(
        "--profile", choices=("smoke", "full"), default="smoke",
        help="full selects the named calibrated P/256,P/512,P/1024 JX ladder",
    )
    parser.add_argument(
        "--rebound-mode", choices=REBOUND_MODES, default="auto",
        help="required fails closed unless exact REBOUND 5.1.1 is available",
    )
    parser.add_argument(
        "--include-limitations", action="store_true",
        help="also execute the separate endpoint-tunnel and central-periapse witnesses",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    profile = (
        Profile.full(include_limitations=arguments.include_limitations)
        if arguments.profile == "full"
        else Profile(include_limitations=arguments.include_limitations)
    )
    report = run_comparison(profile, arguments.rebound_mode)
    text = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    )
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
